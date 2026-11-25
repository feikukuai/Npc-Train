"""在留存集上搜索 beat 先验与决策头的融合方式，并用**无标签**验证器做 Bo-N。

GRPO 那轮的 Bo-N 用了 gold 做选择（泄漏），这里把选择器限制成部署时真能
拿到的信息：事实规则、重复、长度、占位符。同时比较三种融合：
  1. log beat_prob + α·z(dec)        （现状）
  2. z(dec) + β·z(log beat_prob)     （先验改成标准化，权重可很小）
  3. top-K beat 门控后再用 z(dec) 排序
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import canonical_text, load_rows, real_samples
from grpo_head import fact_head, parse_context

HERE = "expertnet/zh"


def deploy_reward(cands, texts, meta):
    """部署时可用的规则：事实命中、近期重复、长度、缺名字占位。"""
    message, replies, name = meta
    recent = {r.rstrip("。！？…，,.!? ") for r in replies[-8:]}
    head = fact_head(message)
    scores = []
    for c in cands:
        text = texts[c]
        value = 0.0
        if head and text.startswith(head):
            value += 0.5
        if text.rstrip("。！？…，,.!? ") in recent:
            value -= 0.2
        if len(text) < 4 or len(text) > 40:
            value -= 0.1
        if "{player_name}" in text and not name:
            value -= 0.5
        if "{last_player}" in text and not replies:
            value -= 0.5
        scores.append(value)
    return torch.tensor(scores, dtype=torch.float32)


def metrics(chosen, beats, gold, gold_beat):
    beat = sum(int(beats[c] == b) for c, b in zip(chosen, gold_beat)) / len(gold)
    exact = sum(int(c == g) for c, g in zip(chosen, gold)) / len(gold)
    return round(beat, 4), round(exact, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()
    torch.manual_seed(5)

    _, fixed = load_rows(EX, 1, 31)
    texts = [r["text"] for r in fixed]
    beats = [r.get("beat", "fixed") for r in fixed]
    index = {t: i for i, t in enumerate(texts)}
    by_beat = {}
    for i, beat in enumerate(beats):
        by_beat.setdefault(beat, []).append(i)
    canon = {b: texts[index[canonical_text([texts[i] for i in pool])]]
             for b, pool in by_beat.items()}
    valid = [i for i, t in enumerate(texts) if "{" not in t]
    pos = {i: j for j, i in enumerate(valid)}
    test = [r for r in real_samples(index, canon, 3000, 1031, 4, heldout=True)
            if r[1] in pos]
    gold = [r[1] for r in test]
    gold_beat = [beats[g] for g in gold]
    metas = [parse_context(r[0]) for r in test]
    print("留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cuda")
    ckpt = torch.load(os.path.join(HERE, "factorized_v3_deploy.pt"),
                      map_location="cuda", weights_only=False)
    fixed_beats = ckpt["fixed_beats"]
    names = sorted(set(fixed_beats))
    beat_idx = {b: i for i, b in enumerate(names)}
    beat_col = torch.tensor([beat_idx[beats[i]] for i in valid], device="cuda")

    beatp = torch.tensor(np.load(os.path.join(HERE, "grpo_beat_test.npz"))["p"],
                         device="cuda")
    estate = torch.load(os.path.join(HERE, "decision_states_test.pt"),
                        map_location="cpu", weights_only=False)["state"].cuda()
    assert len(estate) == len(test) and len(beatp) == len(test)

    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1).cuda()
    proj = nn.Linear(opt.shape[1], estate.shape[1], bias=False).cuda()
    proj.load_state_dict(torch.load(os.path.join(HERE, "decision_head.pt"),
                                    map_location="cuda",
                                    weights_only=False)["state"])
    table = proj(opt)
    dec = estate @ table.T
    zdec = (dec - dec.mean(1, keepdim=True)) / dec.std(1, keepdim=True).clamp_min(1e-6)
    lb = torch.log(beatp[:, beat_col] + 1e-8)
    zlb = (lb - lb.mean(1, keepdim=True)) / lb.std(1, keepdim=True).clamp_min(1e-6)

    report = {"rows": [], "test": len(test)}

    def record(name, chosen, note=""):
        beat, exact = metrics(chosen, beats, gold, gold_beat)
        row = {"name": name, "beat": beat, "exact": exact, "note": note}
        report["rows"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    for alpha in (0.3, 0.4, 0.5, 0.6, 0.8):
        chosen = [valid[p] for p in (lb + alpha * zdec).argmax(1).tolist()]
        record(f"logbeat+{alpha}z", chosen)

    for beta in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        chosen = [valid[p] for p in (zdec + beta * zlb).argmax(1).tolist()]
        record(f"zdec+{beta}zlogbeat", chosen)

    # top-K beat 门控：先按 beat 概率保留前 K 个 beat，再让决策头在门内排序。
    order = beatp.argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(order.size(1), device=order.device)
                  .unsqueeze(0).expand_as(order))
    for k in (1, 2, 3, 5, 8):
        keep = rank[:, beat_col] < k
        record(f"top{k}beat+zdec",
               [valid[p] for p in zdec.masked_fill(~keep, -1e9).argmax(1).tolist()])

    # 无标签 Bo-N：用最好的融合分做采样，验证器只做部署可用的检查。
    best = max(report["rows"], key=lambda r: r["beat"] + r["exact"])
    if best["name"].startswith("zdec+"):
        beta = float(best["name"].split("+")[1].rstrip("zlogbeat"))
        score = zdec + beta * zlb
    elif best["name"].startswith("top"):
        k = int(best["name"][3:].split("beat")[0])
        score = zdec.masked_fill(~(rank[:, beat_col] < k), -1e9)
    else:
        alpha = float(best["name"].split("+")[1].rstrip("z"))
        score = lb + alpha * zdec
    top = score.topk(args.top_m, dim=1)
    probs = F.softmax(top.values / args.temp, dim=1)
    picks = torch.multinomial(probs, args.group, replacement=True)
    chosen, covered = [], 0
    for i in range(len(gold)):
        cands = [valid[int(top.indices[i, j])] for j in picks[i].tolist()]
        covered += int(gold[i] in cands)
        rewards = deploy_reward(cands, texts, metas[i])
        best_j = int(rewards.argmax())
        # 规则分相同时，回退到策略分最高的那条。
        if rewards[best_j] <= 0:
            best_j = int(top.values[i, picks[i]].argmax())
        chosen.append(cands[best_j])
    row = record(f"bon{args.group}@{best['name']}", chosen)
    row["coverage"] = round(covered / len(gold), 4)
    print("coverage", row["coverage"], flush=True)

    with open(os.path.join(HERE, "rerank_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("best greedy", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
