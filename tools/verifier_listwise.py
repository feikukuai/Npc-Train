"""Listwise 验证器：直接优化“在策略采样的 G 条候选里选中教师目标句”。

上一版用随机同 beat 负例 + 逐条 BCE，合成 4 选 1 准确率 0.85，但部署时
候选都来自策略 top-M，分布对不上，Bo-8 exact 只有 0.28（贪心 0.44）。
这一版把训练分布对齐部署：候选 = 策略 softmax 采样（与推理同一套分数），
loss = 候选内 softmax 的交叉熵（名单式），位置随机，避免利用顺序。
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
from grpo_head import policy_scores

HERE = "expertnet/zh"
OUT = os.path.join(HERE, "verifier.pt")
REPORT = os.path.join(HERE, "verifier_report.json")


def encode(enc, texts, batch=96):
    rows = []
    for start in range(0, len(texts), batch):
        rows.append(enc.embed(texts[start:start + batch], batch=batch))
    return torch.cat(rows)


def pair_text(context: str, cand: str) -> str:
    return f"{context}\n候选：{cand}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fill", type=float, default=0.7,
                        help="候选里没有 gold 时，按这个概率把它塞进去")
    args = parser.parse_args()
    torch.manual_seed(17)
    rng = random.Random(17)

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
    n = len(train)
    print("训练", n, "留存", len(test), flush=True)

    ckpt = torch.load(os.path.join(HERE, "factorized_v3_deploy.pt"),
                      map_location="cpu", weights_only=False)
    fixed_beats = ckpt["fixed_beats"]
    names = sorted(set(fixed_beats))
    beat_idx = {b: i for i, b in enumerate(names)}
    beat_col = torch.tensor([beat_idx[beats[i]] for i in valid])
    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=320, device="cuda")
    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1)
    tstate = torch.load(os.path.join(HERE, "decision_states.pt"),
                        map_location="cpu", weights_only=False)["state"]
    estate = torch.load(os.path.join(HERE, "decision_states_test.pt"),
                        map_location="cpu", weights_only=False)["state"]
    train_beat = torch.tensor(np.load(os.path.join(HERE, "grpo_beat_train.npz"))["p"])
    test_beat = torch.tensor(np.load(os.path.join(HERE, "grpo_beat_test.npz"))["p"])
    assert len(tstate) >= n and len(train_beat) >= n
    proj = nn.Linear(opt.shape[1], tstate.shape[1], bias=False)
    proj.load_state_dict(torch.load(os.path.join(HERE, "decision_head.pt"),
                                    map_location="cpu",
                                    weights_only=False)["state"])

    with torch.no_grad():
        table = proj(opt)
        train_score = policy_scores(tstate[:n], table, train_beat[:n],
                                    beat_col, args.alpha)
        test_score = policy_scores(estate, table, test_beat, beat_col, args.alpha)

    # 训练候选：策略 top-M 上采样 G 条，gold 不在里面时按 fill 概率补进去。
    top = train_score.topk(args.top_m, dim=1)
    probs = F.softmax(top.values / args.temp, dim=1)
    picks = torch.multinomial(probs, args.group, replacement=True)
    lists, labels = [], []
    for i in range(n):
        gold = train[i][1]
        cands = [valid[int(top.indices[i, j])] for j in picks[i].tolist()]
        if gold not in cands:
            if rng.random() >= args.fill:
                continue
            cands[rng.randrange(args.group)] = gold
        label = cands.index(gold)
        rng.shuffle(cands)
        lists.append((train[i][0], cands))
        labels.append(cands.index(gold))
    print("训练候选组", len(lists), flush=True)

    dim = opt.shape[1]
    head = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(256, 1)).cuda()
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    y = torch.tensor(labels, dtype=torch.long, device="cuda")

    feats = encode(enc, [pair_text(ctx, texts[c]) for ctx, cands in lists
                         for c in cands]).cuda()
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(lists), device="cuda")
        total, right = 0.0, 0
        for start in range(0, len(perm), 128):
            idx = perm[start:start + 128]
            logits = head(feats[idx.unsqueeze(1) * args.group
                               + torch.arange(args.group, device="cuda")])
            logits = logits.view(len(idx), args.group)
            loss = F.cross_entropy(logits, y[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(idx)
            right += int((logits.argmax(1) == y[idx]).sum())
        print(f"epoch {epoch} listwise loss {total / len(lists):.4f} "
              f"acc {right / len(lists):.3f} {time.time() - t0:.0f}s", flush=True)

    # ---- 部署同分布评测 ----
    gold = [r[1] for r in test]
    gold_beat = [beats[g] for g in gold]
    rows = []

    def report(chosen, name):
        beat = sum(int(beats[c] == b) for c, b in zip(chosen, gold_beat)) / len(gold)
        exact = sum(int(c == g) for c, g in zip(chosen, gold)) / len(gold)
        print(f"{name}: beat {beat:.4f} exact {exact:.4f}", flush=True)
        rows.append({"name": name, "beat": round(beat, 4),
                     "exact": round(exact, 4)})

    def pick(cands_all, name):
        flat = [pair_text(test[i][0], texts[c]) for i in range(len(gold))
                for c in cands_all[i]]
        feats = encode(enc, flat).cuda()
        with torch.no_grad():
            score = head(feats).squeeze(-1).cpu().view(len(gold), args.group)
        chosen = [cands_all[i][int(score[i].argmax())] for i in range(len(gold))]
        report(chosen, name)

    report([valid[p] for p in test_score.argmax(1).tolist()], "greedy")

    ttop = test_score.topk(args.top_m, dim=1)
    tprobs = F.softmax(ttop.values / args.temp, dim=1)
    for g in (4, 8, 16):
        picks_t = torch.multinomial(tprobs, g, replacement=True)
        cands_all, covered = [], 0
        for i in range(len(gold)):
            cands = [valid[int(ttop.indices[i, j])] for j in picks_t[i].tolist()]
            covered += int(gold[i] in cands)
            cands_all.append(cands)
        print(f"bo{g} coverage {covered / len(gold):.3f}", flush=True)
        flat = [pair_text(test[i][0], texts[c]) for i in range(len(gold))
                for c in cands_all[i]]
        feats = encode(enc, flat).cuda()
        with torch.no_grad():
            vscore = head(feats).squeeze(-1).cpu().view(len(gold), g)
        chosen = [cands_all[i][int(vscore[i].argmax())] for i in range(len(gold))]
        report(chosen, f"bo{g}+verifier")

    torch.save({"state": head.state_dict(), "dim": dim, "max_length": 320,
                "top_m": args.top_m, "temp": args.temp}, OUT)
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump({"train": n, "test": len(test), "rows": rows},
                  fh, ensure_ascii=False, indent=1)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
