"""导出多路径候选，供外部大模型裁判打分。

每条样本：贪心句 + 策略 top-M 上采样的 4 条候选（去重后可能少于 4）。
输出 JSONL：context / message / greedy / cands，供 judge_eval.py 逐条打分。
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import (MAX_LINES, FactorizedNet,
                                           canonical_text, load_rows,
                                           real_samples, split_lines)
from grpo_head import parse_context, policy_scores

HERE = "expertnet/zh"


def encode_all(model, enc, contexts, beat_of, n_beats, batch=192):
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
            device = next(model.parameters()).device
            out = model(x.to(device), lengths.to(device))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(HERE, "factorized_v3_deploy.pt"))
    parser.add_argument("--head", default=os.path.join(HERE, "decision_head.pt"))
    parser.add_argument("--rows", default=EX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(HERE, "judge_candidates.jsonl"))
    args = parser.parse_args()
    torch.manual_seed(41)
    random.seed(41)

    _, fixed = load_rows(args.rows, 1, 31)
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

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device=args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = FactorizedNet(**ckpt["config"]).to(args.device).eval()
    model.load_state_dict(ckpt["state"])
    fixed_beats = ckpt["fixed_beats"]
    names = sorted(set(fixed_beats))
    beat_idx = {b: i for i, b in enumerate(names)}
    beat_of = torch.tensor([beat_idx[b] for b in fixed_beats],
                           device=args.device)
    beat_col = torch.tensor([beat_idx[beats[i]] for i in valid])

    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1)
    proj = nn.Linear(opt.shape[1], 1024, bias=False)
    proj.load_state_dict(torch.load(args.head, map_location="cpu",
                                    weights_only=False)["state"])
    with torch.no_grad():
        table = proj(opt)

    contexts = [r[0] for r in test]
    state, beatp = encode_all(model, enc, contexts, beat_of, len(names))
    score = policy_scores(state, table, beatp, beat_col, 0.5)
    greedy = [valid[p] for p in score.argmax(1).tolist()]

    top = score.topk(args.top_m, dim=1)
    probs = F.softmax(top.values / args.temp, dim=1)
    picks = torch.multinomial(probs, args.group, replacement=True)

    idxs = list(range(len(test)))
    random.shuffle(idxs)
    idxs = idxs[:args.limit]
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in idxs:
            message, _, _ = parse_context(contexts[i])
            cands = []
            for j in picks[i].tolist():
                c = valid[int(top.indices[i, j])]
                if c not in cands:
                    cands.append(c)
            fh.write(json.dumps({
                "i": i, "message": message,
                "beat": beats[test[i][1]],
                "greedy": texts[greedy[i]],
                "cands": [texts[c] for c in cands],
            }, ensure_ascii=False) + "\n")
    print("写出", args.out, "条数", len(idxs), flush=True)


if __name__ == "__main__":
    main()
