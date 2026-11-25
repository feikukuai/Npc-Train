"""小决策模型：冻结 BGE 特征 + 双线性指针，在整张固定句表里选一句。

对照 TAMEV/Jev 的 decoupled bi-encoder 思路：state 与每个候选独立编码，
用对称打分排序，天然与候选顺序无关。这里用 BGE-zh 做编码器，在教师对话上
训练指针权重，然后和“最后一句余弦相似度”基线比较。
"""
from __future__ import annotations

import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import (canonical_text, load_rows,
                                           real_samples, split_lines)

REPORT = "expertnet/zh/decision_report.json"


def state_vectors(enc, contexts, cache="expertnet/zh/decision_states.pt"):
    if os.path.exists(cache):
        data = torch.load(cache, map_location="cpu", weights_only=False)
        if len(data["state"]) == len(contexts):
            return data["state"], data["last"]
    split = [split_lines(c) for c in contexts]
    unique = sorted({line for lines in split for line in lines})
    print("唯一行", len(unique), "上下文", len(contexts), flush=True)
    table = {line: i for i, line in enumerate(unique)}
    vec = F.normalize(enc.embed(unique, batch=256), dim=-1).cpu()
    state = torch.zeros(len(split), vec.shape[1] * 2)
    last_vec = torch.zeros(len(split), vec.shape[1])
    for i, lines in enumerate(split):
        rows = vec[[table[line] for line in lines]]
        last_vec[i] = rows[-1]
        hist = rows[:-1].mean(0) if len(rows) > 1 else rows[-1]
        state[i] = torch.cat([rows[-1], hist])
    torch.save({"state": state, "last": last_vec}, cache)
    return state, last_vec


def main():
    torch.manual_seed(7)
    random.seed(7)
    _, fixed = load_rows(EX, 1, 31)
    texts = [r["text"] for r in fixed]
    beats = [r.get("beat", "fixed") for r in fixed]
    index = {t: i for i, t in enumerate(texts)}
    by_beat = {}
    for i, beat in enumerate(beats):
        by_beat.setdefault(beat, []).append(i)
    canon = {b: canonical_text([texts[i] for i in pool])
             for b, pool in by_beat.items()}
    canon = {b: index[t] for b, t in canon.items()}

    train = real_samples(index, {b: texts[i] for b, i in canon.items()},
                         60000, 31, 4, augment_rate=0.5)
    test = real_samples(index, {b: texts[i] for b, i in canon.items()},
                        3000, 1031, 4, heldout=True)
    print("训练", len(train), "测试", len(test), flush=True)

    valid = [i for i, t in enumerate(texts) if "{" not in t]
    pos = {i: j for j, i in enumerate(valid)}
    train = [r for r in train if r[1] in pos]
    test = [r for r in test if r[1] in pos]

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cuda")
    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1).cuda()
    opt_beats = [beats[i] for i in valid]
    same = {}
    for j, beat in enumerate(opt_beats):
        same.setdefault(beat, []).append(j)

    state, last = state_vectors(enc, [r[0] for r in train])
    state = state.cuda()
    last = last.cuda()
    tstate, tlast = state_vectors(enc, [r[0] for r in test], cache="expertnet/zh/decision_states_test.pt")
    tstate = tstate.cuda()
    tlast = tlast.cuda()
    y = torch.tensor([pos[r[1]] for r in train], device="cuda")
    ty = torch.tensor([pos[r[1]] for r in test], device="cuda")

    rng = random.Random(3)
    options = torch.zeros(len(train), 8, dtype=torch.long)
    for i, row in enumerate(train):
        gold = pos[row[1]]
        beat = opt_beats[gold]
        pool = [j for j in same[beat] if j != gold]
        same_neg = rng.sample(pool, min(3, len(pool)))
        other = [j for j in range(len(valid)) if opt_beats[j] != beat]
        neg = same_neg + rng.sample(other, 8 - 1 - len(same_neg))
        choices = [gold] + neg
        rng.shuffle(choices)
        options[i] = torch.tensor(choices)
    options = options.cuda()

    proj = nn.Linear(opt.shape[1], state.shape[1], bias=False).cuda()
    optim = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 8
    for epoch in range(epochs):
        perm = torch.randperm(len(train), device="cuda")
        total = 0.0
        for step, i in enumerate(range(0, len(perm), 256)):
            idx = perm[i:i + 256]
            table = proj(opt)[options[idx]]
            logits = torch.einsum("bd,bkd->bk", state[idx], table)
            gold = (options[idx] == y[idx].unsqueeze(1)).float().argmax(1)
            loss = F.cross_entropy(logits, gold)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
        with torch.no_grad():
            table = proj(opt)
            pred = (tstate @ table.T).argmax(1)
            exact = (pred == ty).float().mean().item()
            beat_acc = sum(int(opt_beats[p] == opt_beats[ty[j]])
                           for j, p in enumerate(pred.tolist())) / len(ty)
        print(f"epoch {epoch}: 全表 exact {exact:.3f} beat {beat_acc:.3f} "
              f"loss {total/max(1, step+1):.3f}", flush=True)

    with torch.no_grad():
        cos = tlast @ opt.T
        pred = cos.argmax(1)
        base_exact = (pred == ty).float().mean().item()
        base_beat = sum(int(opt_beats[p] == opt_beats[ty[j]])
                        for j, p in enumerate(pred.tolist())) / len(ty)

    report = {"train": len(train), "test": len(test),
              "baseline_exact": base_exact, "baseline_beat": base_beat,
              "decision_exact": exact, "decision_beat": beat_acc}
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    torch.save({"state": proj.state_dict()}, "expertnet/zh/decision_head.pt")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
