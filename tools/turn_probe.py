"""探针：把“当前是第几轮”显式喂给决策头，看 exact 能到多少。

教师的 gold 其实是 `LINES[beat][turn % len]`（persona._pick），
而 context 只保留最近 4 轮，turn 信息被截断了。这个脚本验证：
  exact 的上限是不是由“看不清 turn”决定的。
只改决策头输入（state 拼接 turn one-hot），不重训因子化模型。
"""
from __future__ import annotations

import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.corpus import HELDOUT, SEEDS
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import (augment, canonical_text,
                                           format_line_context, load_rows)
from expertnet.zh.persona import format_line, new_state, teacher_turn

HERE = "expertnet/zh"
TURNS = 16


def samples_with_turn(index, canon, count, seed, turns, heldout=False,
                      augment_rate=0.0):
    phrases = HELDOUT if heldout else SEEDS
    rng = random.Random(seed)
    samples = []
    while len(samples) < count:
        state = new_state()
        history = []
        for _ in range(rng.randint(6, 12)):
            intent = rng.choice(list(phrases))
            player = rng.choice(phrases[intent])
            surface = augment(player, rng, augment_rate)
            context = format_line_context(history, surface, turns)
            tpl, beat = teacher_turn(player, state)
            target = tpl if tpl in index else canon.get(beat)
            if target is None or target not in index:
                continue
            samples.append((context, index[target], beat,
                            min(int(state["turn"]), TURNS - 1)))
            history.append((surface, format_line(target, state)))
            if len(samples) >= count:
                break
    return samples


def onehot(turns, device="cuda"):
    table = torch.eye(TURNS, device=device)
    return table[torch.tensor(turns, device=device)]


def main():
    torch.manual_seed(23)
    random.seed(23)
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

    train = [r for r in samples_with_turn(index, canon, 60000, 31, 4,
                                          augment_rate=0.5) if r[1] in pos]
    test = [r for r in samples_with_turn(index, canon, 3000, 1031, 4,
                                         heldout=True) if r[1] in pos]
    print("训练", len(train), "留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cuda")
    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1).cuda()
    opt_beats = [beats[i] for i in valid]
    same = {}
    for j, beat in enumerate(opt_beats):
        same.setdefault(beat, []).append(j)

    tstate = torch.load(f"{HERE}/decision_states.pt", map_location="cpu",
                        weights_only=False)["state"].cuda()
    estate = torch.load(f"{HERE}/decision_states_test.pt", map_location="cpu",
                        weights_only=False)["state"].cuda()
    tvalid = [i for i in range(len(test)) if test[i][1] in pos]

    def run(with_turn: bool):
        k, n_same = 24, 9
        rng = random.Random(3)
        others = {beat: [j for j in range(len(valid)) if opt_beats[j] != beat]
                  for beat in same}
        options = torch.zeros(len(train), k, dtype=torch.long)
        for i, row in enumerate(train):
            gold = pos[row[1]]
            beat = opt_beats[gold]
            pool = [j for j in same[beat] if j != gold]
            same_neg = rng.sample(pool, min(n_same, len(pool), k - 1))
            neg = same_neg + rng.sample(others[beat], k - 1 - len(same_neg))
            choices = [gold] + neg
            rng.shuffle(choices)
            options[i] = torch.tensor(choices)
        options = options.cuda()

        width = tstate.shape[1] + (TURNS if with_turn else 0)
        proj = nn.Linear(opt.shape[1], width, bias=False).cuda()
        optim = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)
        y = torch.tensor([pos[r[1]] for r in train], device="cuda")
        tindex = torch.tensor([pos[test[i][1]] for i in tvalid])
        tst = estate[tvalid]
        if with_turn:
            tturn = onehot([test[i][3] for i in tvalid])
            train_state = torch.cat([tstate, onehot([r[3] for r in train])], 1)
        else:
            train_state = tstate
        for epoch in range(12):
            perm = torch.randperm(len(train), device="cuda")
            for start in range(0, len(perm), 256):
                idx = perm[start:start + 256]
                flat = options[idx].reshape(-1)
                table = proj(opt[flat]).view(len(idx), -1, width)
                logits = torch.einsum("bd,bkd->bk", train_state[idx], table)
                gold = (options[idx] == y[idx].unsqueeze(1)).float().argmax(1)
                loss = F.cross_entropy(logits, gold)
                optim.zero_grad()
                loss.backward()
                optim.step()
            with torch.no_grad():
                table = proj(opt)
                if with_turn:
                    base = torch.cat([tst, tturn], dim=1)
                else:
                    base = tst
                pred = (base @ table.T).argmax(1)
                exact = (pred.cpu() == tindex).float().mean().item()
                beat_acc = sum(int(opt_beats[p] == opt_beats[tindex[j]])
                               for j, p in enumerate(pred.cpu().tolist())) \
                    / len(tindex)
            print(f"  epoch {epoch} exact {exact:.3f} beat {beat_acc:.3f}",
                  flush=True)
        return {"with_turn": with_turn, "exact": round(exact, 4),
                "beat": round(beat_acc, 4)}

    print("=== 不带轮次（基准） ===", flush=True)
    base = run(False)
    print("=== 带轮次 one-hot ===", flush=True)
    turn = run(True)
    report = {"baseline": base, "with_turn": turn}
    with open(f"{HERE}/turn_probe.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
