"""Cross-encoder 验证器：给“上下文 + 候选句”打分，兑现多路径采样的上限。

R1 式流程在本项目里的版本：
  策略（beat 概率 + 决策头）采样 G 条候选 → 验证器逐条打分 → 选最高。
  验证器用教师标签训练（正例=教师目标句，负例=同 beat 难例 + 其他 beat），
  部署时不需要人理解理由，只要分数；这也是后面 loop/自对弈的奖励来源。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import canonical_text, load_rows, real_samples
from grpo_head import parse_context, policy_scores

HERE = "expertnet/zh"
OUT = os.path.join(HERE, "verifier.pt")
REPORT = os.path.join(HERE, "verifier_report.json")


def build_pairs(samples, beats, valid, rng, n_same=2, n_other=1):
    """每条样本：1 正 + 同 beat 难例 + 其他 beat 负例。"""
    same = {}
    for i in valid:
        same.setdefault(beats[i], []).append(i)
    others = list(valid)
    pairs = []
    for context, gold, _ in samples:
        beat = beats[gold]
        pool = [i for i in same[beat] if i != gold]
        negs = rng.sample(pool, min(n_same, len(pool)))
        while len(negs) < n_same + n_other:
            pick = rng.choice(others)
            if pick != gold and beats[pick] != beat:
                negs.append(pick)
        pairs.append((context, gold, 1))
        pairs.extend((context, cand, 0) for cand in negs)
    return pairs


def encode(enc, texts, batch=96):
    rows = []
    for start in range(0, len(texts), batch):
        rows.append(enc.embed(texts[start:start + batch], batch=batch))
    return torch.cat(rows)


def pair_text(context: str, cand: str) -> str:
    return f"{context}\n候选：{cand}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    torch.manual_seed(13)
    rng = random.Random(13)

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

    train = [r for r in real_samples(index, canon, args.samples, 31, 4,
                                     augment_rate=0.5) if r[1] in pos]
    test = [r for r in real_samples(index, canon, 3000, 1031, 4, heldout=True)
            if r[1] in pos]
    print("训练", len(train), "留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=320, device="cuda")
    dim = enc.embed(["测试"], batch=1).shape[1]
    head = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(256, 1)).cuda()
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    gold_set = {r[1] for r in test}
    print("验证器目标：把教师目标句排在 1 正 3 负之上", flush=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        pairs = build_pairs(train, beats, valid, rng)
        feats = encode(enc, [pair_text(ctx, texts[c]) for ctx, c, _ in pairs])
        labels = torch.tensor([lbl for _, _, lbl in pairs], dtype=torch.float32)
        perm = torch.randperm(len(pairs))
        total, correct, seen = 0.0, 0, 0
        for start in range(0, len(pairs), 512):
            idx = perm[start:start + 512]
            logits = head(feats[idx].cuda()).squeeze(-1)
            target = labels[idx].cuda()
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(idx)
            correct += int(((logits > 0) == (target > 0.5)).sum())
            seen += len(idx)
        print(f"epoch {epoch} loss {total / seen:.4f} acc {correct / seen:.3f} "
              f"{time.time() - t0:.0f}s", flush=True)

        # 在留存集上测“4 选 1”的排序准确率。
        sub = test[:600]
        sample_pairs = build_pairs([(c, g, b) for c, g, b in
                                    [(r[0], r[1], 0) for r in sub]],
                                   beats, valid, random.Random(99))
        feats2 = encode(enc, [pair_text(ctx, texts[c]) for ctx, c, _ in sample_pairs])
        with torch.no_grad():
            score = head(feats2.cuda()).squeeze(-1).cpu()
        right = 0
        for i in range(0, len(sample_pairs), 4):
            if int(score[i:i + 4].argmax()) == 0:
                right += 1
        print(f"  留存 4 选 1 排序准确率 {right / len(sub):.3f}", flush=True)

    # ---- 多路径选择评测 ----
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
    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1).cuda()
    proj = nn.Linear(opt.shape[1], estate.shape[1], bias=False).cuda()
    proj.load_state_dict(torch.load(os.path.join(HERE, "decision_head.pt"),
                                    map_location="cuda",
                                    weights_only=False)["state"])
    with torch.no_grad():
        table = proj(opt)
    score = policy_scores(estate, table, beatp, beat_col, args.alpha)

    gold = [r[1] for r in test]
    gold_beat = [beats[g] for g in gold]

    def report(chosen, name):
        beat = sum(int(beats[c] == b) for c, b in zip(chosen, gold_beat)) / len(gold)
        exact = sum(int(c == g) for c, g in zip(chosen, gold)) / len(gold)
        print(f"{name}: beat {beat:.4f} exact {exact:.4f}", flush=True)
        return {"name": name, "beat": round(beat, 4), "exact": round(exact, 4)}

    rows = []
    greedy = [valid[p] for p in score.argmax(1).tolist()]
    rows.append(report(greedy, "greedy"))

    top = score.topk(args.top_m, dim=1)
    probs = F.softmax(top.values / args.temp, dim=1)
    picks = torch.multinomial(probs, args.group, replacement=True)
    cands_all, covered, flat_texts = [], 0, []
    for i in range(len(gold)):
        cands = [valid[int(top.indices[i, j])] for j in picks[i].tolist()]
        covered += int(gold[i] in cands)
        cands_all.append(cands)
        flat_texts.extend(pair_text(test[i][0], texts[c]) for c in cands)
    print("coverage", round(covered / len(gold), 4), flush=True)
    feats3 = encode(enc, flat_texts)
    with torch.no_grad():
        vscore = head(feats3.cuda()).squeeze(-1).cpu().view(len(gold), args.group)
    for lam in (0.0, 0.5, 1.0, 2.0):
        base = torch.tensor([[float(score[i, int(picks[i, j])])
                              for j in range(args.group)]
                             for i in range(len(gold))])
        total = vscore + lam * base
        chosen = [cands_all[i][int(total[i].argmax())] for i in range(len(gold))]
        rows.append(report(chosen, f"bo{args.group}+verifier(lam={lam})"))

    torch.save({"state": head.state_dict(), "dim": dim,
                "max_length": 320}, OUT)
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump({"train": len(train), "test": len(test),
                   "coverage": round(covered / len(gold), 4),
                   "rows": rows}, fh, ensure_ascii=False, indent=1)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
