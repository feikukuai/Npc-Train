"""新评测口径 + 输入闭集化实验。

不再把 exact 当主指标（教师的 gold 是 LINES[beat][turn % n]，轮换随机）。
这里报告：
  beat      选中句 beat 是否等于教师 beat（真正的路由准确率）
  legal     合法率：无占位符、长度 4–40
  repeat    与最近 8 句重复的比例
  fact      事实问句是否命中正确事实句
  diversity 各 gold beat 内，不同候选句的占比（越高越不呆板）

--closure 打开输入闭集化：最后一句玩家话先映射到 SEEDS 里最近的标准说法
（相似度 < 阈值则保持原句），再走同一条管线，看 beat/legal 是否变化。
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.corpus import SEEDS
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import (MAX_LINES, FactorizedNet,
                                           canonical_text, load_rows,
                                           real_samples, split_lines)
from grpo_head import fact_head, parse_context

HERE = "expertnet/zh"


def load_shared():
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
    return texts, beats, valid, test


def encode_all(model, enc, contexts, beat_of, n_beats, batch=192):
    """一次编码所有上下文，同时给出 state 向量和 beat 概率。"""
    split = [split_lines(c) for c in contexts]
    unique = sorted({line for lines in split for line in lines})
    table = {line: i for i, line in enumerate(unique)}
    vec = enc.embed(unique, batch=256)
    states, beatps = [], []
    for start in range(0, len(split), batch):
        chunk = split[start:start + batch]
        x = torch.zeros(len(chunk), MAX_LINES, vec.shape[1])
        lengths = torch.zeros(len(chunk), dtype=torch.long)
        for i, lines in enumerate(chunk):
            lengths[i] = len(lines)
            for j, line in enumerate(lines):
                x[i, j] = vec[table[line]]
        with torch.no_grad():
            out = model(x.cuda(), lengths.cuda())
            probs = F.softmax(out["fixed"] + out["fixed_local"], dim=-1)
        bp = torch.zeros(len(chunk), n_beats, device=probs.device)
        bp.scatter_add_(1, beat_of.unsqueeze(0).expand(len(chunk), -1), probs)
        beatps.append(bp.cpu())
        norm = F.normalize(x, dim=-1)
        for i in range(len(chunk)):
            n = int(lengths[i])
            last = norm[i, n - 1]
            hist = norm[i, :n - 1].mean(0) if n > 1 else last
            states.append(torch.cat([last, hist]))
    return torch.stack(states), torch.cat(beatps)


def metrics(chosen, test, beats, texts, dump=0):
    gold = [r[1] for r in test]
    metas = [parse_context(r[0]) for r in test]
    beat_hit = sum(int(beats[c] == beats[g]) for c, g in zip(chosen, gold)) / len(gold)
    exact = sum(int(c == g) for c, g in zip(chosen, gold)) / len(gold)
    legal, repeat, fact_total, fact_hit = 0, 0, 0, 0
    misses = []
    for c, (_, replies, _) in zip(chosen, metas):
        text = texts[c]
        recent = {r.rstrip("。！？…，,.!? ") for r in replies[-8:]}
        legal += int("{" not in text and 4 <= len(text) <= 40)
        repeat += int(text.rstrip("。！？…，,.!? ") in recent)
    for c, g, (message, _, _) in zip(chosen, gold, metas):
        if beats[g] != "personal":
            continue
        head = fact_head(message)
        if head:
            fact_total += 1
            fact_hit += int(texts[c].startswith(head))
            if not texts[c].startswith(head):
                misses.append((message, head, texts[c], beats[g], beats[c]))
    for message, head, got, gb, pb in misses[:dump]:
        print(f"  问：{message[:24]} | 应：{head} | 实：{got[:22]} "
              f"| gold beat {gb} / 预测 {pb}", flush=True)
    by_gold = collections.defaultdict(list)
    for c, g in zip(chosen, gold):
        by_gold[beats[g]].append(texts[c])
    diversity = sum(1 - collections.Counter(v).most_common(1)[0][1] / len(v)
                    for v in by_gold.values()) / max(1, len(by_gold))
    return {
        "beat": round(beat_hit, 4), "exact": round(exact, 4),
        "legal": round(legal / len(gold), 4),
        "repeat": round(repeat / len(gold), 4),
        "fact": round(fact_hit / fact_total, 4) if fact_total else None,
        "fact_n": fact_total,
        "diversity": round(diversity, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure", action="store_true")
    parser.add_argument("--ckpt", default=os.path.join(HERE, "factorized_v3_deploy.pt"))
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--dump", type=int, default=0)
    args = parser.parse_args()
    texts, beats, valid, test = load_shared()
    print("留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cuda")
    ckpt = torch.load(args.ckpt,
                      map_location="cuda", weights_only=False)
    model = FactorizedNet(**ckpt["config"]).cuda().eval()
    model.load_state_dict(ckpt["state"])
    fixed_beats = ckpt["fixed_beats"]
    names = sorted(set(fixed_beats))
    beat_idx = {b: i for i, b in enumerate(names)}
    beat_of = torch.tensor([beat_idx[b] for b in fixed_beats], device="cuda")
    beat_col = torch.tensor([beat_idx[beats[i]] for i in valid])

    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1)
    proj = nn.Linear(opt.shape[1], 1024, bias=False)
    proj.load_state_dict(torch.load(os.path.join(HERE, "decision_head.pt"),
                                    map_location="cpu",
                                    weights_only=False)["state"])
    with torch.no_grad():
        table = proj(opt)

    contexts = [r[0] for r in test]
    hit = None
    if args.closure:
        seeds = sorted({line for lines in SEEDS.values() for line in lines})
        seed_vec = F.normalize(enc.embed(seeds, batch=256), dim=-1)
        metas = [parse_context(c) for c in contexts]
        last_vec = F.normalize(enc.embed([m[0] for m in metas], batch=256),
                               dim=-1)
        sims = last_vec @ seed_vec.T
        best = sims.max(1)
        new_contexts = []
        hit = 0
        for i, context in enumerate(contexts):
            score, j = float(best.values[i]), int(best.indices[i])
            if score >= args.threshold:
                hit += 1
                lines = context.split("\n")
                lines[-1] = f"玩家：{seeds[j]}"
                new_contexts.append("\n".join(lines))
            else:
                new_contexts.append(context)
        print("归一化命中率", round(hit / len(contexts), 3),
              "阈值", args.threshold, flush=True)
        contexts = new_contexts

    state, beatp = encode_all(model, enc, contexts, beat_of, len(names))
    dec = state @ table.T
    z = (dec - dec.mean(1, keepdim=True)) / dec.std(1, keepdim=True).clamp_min(1e-6)
    score = torch.log(beatp[:, beat_col] + 1e-8) + 0.5 * z
    chosen = [valid[p] for p in score.argmax(1).tolist()]
    # 与部署一致：事实关键词命中且 beat 属于模糊类时，规则优先。
    fact_block = {"gift", "threat", "insult", "apology", "apology_meta",
                  "farewell", "farewell_hurt", "boundary", "trade", "promise",
                  "name_saved", "taunt"}
    secret_keys = ("秘密", "藏", "偷偷", "柴房")
    unknown_keys = ("穿", "衣服", "村子", "村外", "山那边", "河", "天上", "蓝",
                    "镇子", "远处")
    pool_oow = [j for j in valid if beats[j] == "out_of_world"]
    metas = [parse_context(r[0]) for r in test]
    fixed_chosen = []
    for idx, meta in zip(chosen, metas):
        message = meta[0]
        secret_ask = beats[idx] in ("secret_denied", "secret") and \
            any(k in message for k in secret_keys)
        if beats[idx] not in fact_block and not secret_ask:
            head = fact_head(message)
            if head:
                hit = next((j for j in valid if texts[j].startswith(head)), None)
                if hit is not None:
                    idx = hit
        elif any(k in message for k in unknown_keys) and pool_oow:
            idx = pool_oow[0]
        fixed_chosen.append(idx)
    chosen = fixed_chosen
    row = metrics(chosen, test, beats, texts, dump=args.dump)
    print(json.dumps(row, ensure_ascii=False), flush=True)

    out = os.path.join(HERE, "quality_report_closure.json" if args.closure
                       else "quality_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": row, "closure": args.closure,
                   "threshold": args.threshold if args.closure else None,
                   "closure_hit": hit}, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
