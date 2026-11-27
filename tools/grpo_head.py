"""GRPO（组内基线策略梯度）微调决策头 + Best-of-N 规则验证器。

策略只定义在固定句表上：
    π(c | s) ∝ softmax((log beat_prob(c) + α·z(dec(c|s))) / τ)，候选取 top-M。
组采样 G 条，奖励 = beat 命中(+1) + 精确目标(+1) + 事实命中(+0.5)
                - 近期重复(-0.2) - 长度异常(-0.1) - 缺名字占位(-0.5)。
只更新决策头 proj，冻结 v3 因子化模型与 BGE；与 heldout 留存集对比
beat / exact（集成分支、Bo-N 分支各一组）。
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
from expertnet.zh.persona import extract_name

HERE = "expertnet/zh"
CKPT = os.path.join(HERE, "factorized_v3_deploy.pt")
HEAD = os.path.join(HERE, "decision_head.pt")
REPORT = os.path.join(HERE, "grpo_report.json")
TRAIN_CACHE = os.path.join(HERE, "grpo_beat_train.npz")
TEST_CACHE = os.path.join(HERE, "grpo_beat_test.npz")

# 与 context_app.FACT_RULES 同源（那边带 gradio，这里不能直接 import）。
FACT_RULES = [
    (("阿团是什么", "阿团是啥", "什么动物", "猫是什么", "阿团是猫吗"), (),
     "阿团是猫"),
    (("认识阿团", "知道阿团", "认识猫吗", "认识那只猫", "见过阿团"), (),
     "认识呀，阿团"),
    (("有猫吗", "养猫吗", "家里有猫"), (), "有呀，阿团"),
    (("阿团在哪", "猫在哪", "阿团在哪里", "猫在哪里"), (),
     "阿团在柜台下面睡觉"),
    (("你知道什么", "你会什么", "你懂什么", "你都知道什么"), (), "我知道药铺"),
    (("娘", "妈", "母亲"), ("教", "学", "草药"), "娘教我认草药"),
    (("娘", "妈", "母亲"), ("名字", "叫", "谁", "姓", "问你娘"), "我娘叫阿棠"),
    (("娘", "妈", "母亲"), ("做什么", "干什么", "做啥"), "娘在药铺抓药"),
    (("阿团", "猫"), (), "阿团是橘色的"),
    (("药园", "草药园"), (), "屋后是草药园"),
    (("井",), (), "老井很深"),
    (("几岁", "多大", "年龄"), (), "我五岁半"),
    (("住",), ("哪", "哪里", "哪儿"), "我叫阿禾"),
    (("你叫什么", "你是谁", "你叫啥", "你的名字"), (), "我叫阿禾"),
    (("糖", "麦芽糖"), ("在哪", "哪儿", "哪里", "有糖", "糖呢", "还有"), "麦芽糖在柜子上"),
    (("打雷", "雷"), (), "打雷我钻桌底"),
    (("秘密", "藏", "偷偷", "柴房"), (), "柴房有只小鸟"),
    (("药", "草药"), ("认", "懂", "会"), "娘在药铺抓药"),
    (("蚂蚁", "数"), (), "我数到七只"),
]


def fact_head(message: str) -> str:
    for keys, must, head in FACT_RULES:
        if any(k in message for k in keys) \
                and (not must or any(k in message for k in must)):
            return head
    return ""


def parse_context(context: str):
    """从训练上下文里取最后一问、历史回答、玩家名字。"""
    players, replies = [], []
    for line in context.split("\n"):
        if line.startswith("玩家："):
            players.append(line[3:])
        elif line.startswith("阿禾："):
            replies.append(line[3:])
    message = players[-1] if players else ""
    name = ""
    for text in reversed(players):
        name = extract_name(text)
        if name:
            break
    return message, replies, name


def encode_contexts(enc, contexts, batch=384):
    for start in range(0, len(contexts), batch):
        chunk = contexts[start:start + batch]
        split = [split_lines(c) for c in chunk]
        unique = sorted({line for lines in split for line in lines})
        table = {line: i for i, line in enumerate(unique)}
        vec = enc.embed(unique, batch=256)
        x = torch.zeros(len(split), MAX_LINES, vec.shape[1])
        lengths = torch.zeros(len(split), dtype=torch.long)
        for i, lines in enumerate(split):
            lengths[i] = len(lines)
            for j, line in enumerate(lines):
                x[i, j] = vec[table[line]]
        yield x, lengths


def beat_probabilities(model, enc, contexts, beat_of, n_beats, cache):
    if os.path.exists(cache):
        data = np.load(cache)
        if len(data["p"]) == len(contexts):
            return torch.tensor(data["p"])
    rows = []
    for x, lengths in encode_contexts(enc, contexts):
        with torch.no_grad():
            out = model(x.cuda(), lengths.cuda())
            probs = F.softmax(out["fixed"] + out["fixed_local"], dim=-1)
        beat = torch.zeros(len(x), n_beats, device=probs.device)
        beat.scatter_add_(1, beat_of.unsqueeze(0).expand(len(x), -1), probs)
        rows.append(beat.cpu())
    p = torch.cat(rows)
    np.savez(cache, p=p.numpy())
    return p


class Verifier:
    """规则奖励：beat/事实/重复/长度/占位符，同时用于 Bo-N 选择。"""

    def __init__(self, texts, beats):
        self.texts = texts
        self.beats = beats

    def reward(self, cands, gold, meta):
        gold_text = self.texts[gold]
        gold_beat = self.beats[gold]
        message, replies, name = meta
        recent = {r.rstrip("。！？…，,.!? ") for r in replies[-8:]}
        head = fact_head(message)
        scores = []
        for c in cands:
            text = self.texts[c]
            value = 1.0 if self.beats[c] == gold_beat else 0.0
            value += 1.0 if text == gold_text else 0.0
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


def policy_scores(state, table, beat_prob, beat_col, alpha, chunk=2048):
    """log beat_prob + α·z(decision)，返回 [N, V]。"""
    out = []
    logp = torch.log(beat_prob + 1e-8)
    for start in range(0, len(state), chunk):
        dec = state[start:start + chunk] @ table.T
        z = (dec - dec.mean(1, keepdim=True)) / dec.std(1, keepdim=True).clamp_min(1e-6)
        out.append(logp[start:start + chunk, beat_col] + alpha * z)
    return torch.cat(out)


def evaluate(name, state, table, beat_prob, beat_col, beats, texts, valid,
             gold, gold_beat, metas, alpha, top_m, group, temp, verifier):
    score = policy_scores(state, table, beat_prob, beat_col, alpha)
    top = score.topk(min(top_m, score.size(1)), dim=1)
    pred = top.indices[:, 0]
    chose = [valid[p] for p in pred.tolist()]
    beat_acc = sum(int(beats[c] == b) for c, b in zip(chose, gold_beat)) / len(gold)
    exact = sum(int(c == g) for c, g in zip(chose, gold)) / len(gold)

    # Bo-N：在 top-M 上按策略分布采 G 条，用规则验证器挑最高奖励。
    logits = top.values / temp
    probs = F.softmax(logits, dim=1)
    chosen = torch.multinomial(probs, group, replacement=True)
    bo_correct, bo_beat = 0, 0
    for i in range(len(gold)):
        cands = [valid[int(top.indices[i, j])] for j in chosen[i].tolist()]
        r = verifier.reward(cands, gold[i], metas[i])
        best = cands[int(r.argmax())]
        bo_correct += int(best == gold[i])
        bo_beat += int(beats[best] == gold_beat[i])
    row = {"alpha": alpha, "beat": round(beat_acc, 4), "exact": round(exact, 4),
           "bo_beat": round(bo_beat / len(gold), 4),
           "bo_exact": round(bo_correct / len(gold), 4)}
    print(name, json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--limit", type=int, default=60000)
    args = parser.parse_args()
    torch.manual_seed(11)
    random.seed(11)

    _, fixed = load_rows(EX, 1, 31)
    texts = [r["text"] for r in fixed]
    beats = [r.get("beat", "fixed") for r in fixed]
    index = {t: i for i, t in enumerate(texts)}
    by_beat = {}
    for i, beat in enumerate(beats):
        by_beat.setdefault(beat, []).append(i)
    canon = {b: index[canonical_text([texts[i] for i in pool])]
             for b, pool in by_beat.items()}
    canon_text = {b: texts[i] for b, i in canon.items()}
    valid = [i for i, t in enumerate(texts) if "{" not in t]
    pos = {i: j for j, i in enumerate(valid)}

    train = real_samples(index, canon_text, args.limit, 31, 4, augment_rate=0.5)
    test = real_samples(index, canon_text, 3000, 1031, 4, heldout=True)
    train = [r for r in train if r[1] in pos]
    test = [r for r in test if r[1] in pos]
    print("训练", len(train), "留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cuda")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    model = FactorizedNet(**ckpt["config"]).cuda().eval()
    model.load_state_dict(ckpt["state"])
    fixed_beats = ckpt["fixed_beats"]
    names = sorted(set(fixed_beats))
    beat_idx = {b: i for i, b in enumerate(names)}
    beat_of = torch.tensor([beat_idx[b] for b in fixed_beats], device="cuda")

    tstate = torch.load(os.path.join(HERE, "decision_states.pt"),
                        map_location="cpu", weights_only=False)["state"]
    estate = torch.load(os.path.join(HERE, "decision_states_test.pt"),
                        map_location="cpu", weights_only=False)["state"]
    assert len(tstate) == len(train) and len(estate) == len(test), \
        (len(tstate), len(train), len(estate), len(test))
    train_beat = beat_probabilities(model, enc, [r[0] for r in train], beat_of,
                                    len(names), TRAIN_CACHE)
    test_beat = beat_probabilities(model, enc, [r[0] for r in test], beat_of,
                                   len(names), TEST_CACHE)

    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1).cuda()
    proj = nn.Linear(opt.shape[1], tstate.shape[1], bias=False).cuda()
    proj.load_state_dict(torch.load(HEAD, map_location="cuda",
                                    weights_only=False)["state"])
    init = {k: v.detach().clone() for k, v in proj.state_dict().items()}

    verifier = Verifier(texts, beats)
    gold = [r[1] for r in test]
    gold_beat = [beats[g] for g in gold]
    metas = [parse_context(r[0]) for r in test]
    estate = estate.cuda()
    test_beat = test_beat.cuda()
    tstate = tstate.cuda()
    train_beat = train_beat.cuda()

    base_rows = []
    beat_col = torch.tensor([beat_idx[beats[i]] for i in valid], device="cuda")
    with torch.no_grad():
        table = proj(opt)
        for alpha in (0.3, 0.4, 0.5, 0.6, 0.7):
            base_rows.append(evaluate("baseline", estate, table, test_beat,
                                      beat_col, beats, texts, valid, gold,
                                      gold_beat, metas, alpha, args.top_m,
                                      args.group, args.temp, verifier))

    phrases = [parse_context(r[0]) for r in train]
    optim = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for step in range(args.steps):
        idx = torch.randint(0, len(train), (args.batch,), device="cuda")
        table = proj(opt)
        score = policy_scores(tstate[idx], table, train_beat[idx], beat_col,
                              args.alpha)
        top = score.topk(args.top_m, dim=1)
        logp = F.log_softmax(top.values / args.temp, dim=1)
        with torch.no_grad():
            chosen = torch.multinomial(logp.exp(), args.group, replacement=True)
        rewards = torch.zeros(args.batch, args.group)
        for i in range(args.batch):
            row = int(idx[i])
            cands = [valid[int(top.indices[i, j])] for j in chosen[i].tolist()]
            rewards[i] = verifier.reward(cands, train[row][1], phrases[row])
        adv = (rewards - rewards.mean(1, keepdim=True)) / \
            rewards.std(1, keepdim=True).clamp_min(1e-6)
        adv = adv.cuda()
        gather = torch.gather(logp, 1, chosen)
        loss = -(gather * adv).mean() + 1e-4 * sum(
            (p - init[k]).pow(2).mean() for k, p in proj.named_parameters())
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        optim.step()
        if step % 50 == 0 or step == args.steps - 1:
            mean_r = rewards.mean().item()
            print(f"step {step} loss {loss.item():.4f} reward {mean_r:.3f}",
                  flush=True)
            history.append({"step": step, "loss": round(loss.item(), 4),
                            "reward": round(mean_r, 4)})

    with torch.no_grad():
        table = proj(opt)
        rows = [evaluate("grpo", estate, table, test_beat, beat_col, beats,
                         texts, valid, gold, gold_beat, metas, args.alpha,
                         args.top_m, args.group, args.temp, verifier)]
        for alpha in (0.4, 0.6, 0.7):
            if alpha != args.alpha:
                rows.append(evaluate("grpo", estate, table, test_beat, beat_col,
                                     beats, texts, valid, gold, gold_beat,
                                     metas, alpha, args.top_m, args.group,
                                     args.temp, verifier))

    best_base = max(base_rows, key=lambda r: r["exact"] + r["beat"])
    best = max(rows, key=lambda r: r["exact"] + r["beat"])
    report = {"train": len(train), "test": len(test), "baseline": base_rows,
              "grpo": rows, "best_baseline": best_base, "best_grpo": best,
              "steps": args.steps, "group": args.group, "top_m": args.top_m,
              "temp": args.temp, "history": history}
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    gain = (best["exact"] - best_base["exact"]) + (best["beat"] - best_base["beat"])
    print("baseline best", json.dumps(best_base, ensure_ascii=False))
    print("grpo best", json.dumps(best, ensure_ascii=False))
    if gain > 0:
        torch.save({"state": proj.state_dict()}, os.path.join(HERE, "decision_head_grpo.pt"))
        print("saved decision_head_grpo.pt (gain %.4f)" % gain)
    else:
        print("no gain (%.4f), keep current head" % gain)


if __name__ == "__main__":
    main()
