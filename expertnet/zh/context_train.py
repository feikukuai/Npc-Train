"""纯上下文 → 句表 ID 的小规模训练原型。

这个脚本故意只做一件事：输入最近几轮对话，输出句表里的一句。没有状态向量、
没有名字参数、没有额外输入；名字和“刚才的话”只能从上下文里自己学。

第一版把句表缩到约 1000 句，方便在 CPU / 单卡 T4 上先验证链路：
    python -m expertnet.zh.context_train --samples 8000 --epochs 6

放大到 1B 时，把这里的 MLP 换成预训练 Transformer，输出层换成检索指针即可；
数据格式和评测口径不变。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bank import get_bank
from .corpus import HELDOUT, SEEDS
from .persona import BEATS, FACTS, LINES, format_line, new_state, teacher_turn

HERE = os.path.dirname(os.path.abspath(__file__))
ATOMS = os.path.join(HERE, "atoms_zh.json")
CKPT = os.path.join(HERE, "context_net.pt")
REPORT = os.path.join(HERE, "context_report.json")
ENC_MODEL = "BAAI/bge-small-zh-v1.5"
FACT_BEATS = {"personal", "lore", "secret", "boundary", "threat"}


def atom_beats() -> dict[str, str]:
    """给 86 个手写句标上 beat，作为句表里的事实/安全条目。"""
    alias = {"memory_first": "memory_recall", "memory_hurt": "memory_recall",
             "memory_quote": "memory_recall", "memory_name": "memory_recall",
             "memory_repeat": "memory_recall"}
    table = {fact["text"]: fact["beat"] for fact in FACTS.values()}
    for beat, lines in LINES.items():
        beat = alias.get(beat, beat)
        if beat not in BEATS:
            continue
        for line in lines:
            table.setdefault(line, beat)
    return table


def build_table(size: int = 1000, seed: int = 31):
    """手写句 + 分层抽样的整句库，返回 (texts, beats, bank_map)。"""
    bank = get_bank()
    with open(ATOMS, encoding="utf-8") as fh:
        atoms = json.load(fh)["atoms"]
    known = atom_beats()
    base = [(text, known[text]) for text in atoms if text in known]

    rng = random.Random(seed)
    room = max(0, size - len(base))
    per_beat = max(1, room // max(1, len(bank.by_beat)))
    chosen: list[int] = []
    for beat, ids in bank.by_beat.items():
        pool = list(ids)
        rng.shuffle(pool)
        chosen.extend(pool[:per_beat])

    seen = {text for text, _ in base}
    texts = [text for text, _ in base]
    beats = [beat for _, beat in base]
    bank_map: dict[int, int] = {}
    for idx in chosen:
        text = bank.texts[idx]
        if text in seen:
            continue
        seen.add(text)
        bank_map[idx] = len(texts)
        texts.append(text)
        beats.append(bank.beats[idx])

    # 不够就从整句库随机补齐。
    if len(texts) < size:
        rest = list(range(bank.size))
        rng.shuffle(rest)
        for idx in rest:
            if len(texts) >= size:
                break
            text = bank.texts[idx]
            if text in seen:
                continue
            seen.add(text)
            bank_map[idx] = len(texts)
            texts.append(text)
            beats.append(bank.beats[idx])
    return texts[:size], beats[:size], bank_map


def format_context(history: list[tuple[str, str]], player: str, turns: int) -> str:
    parts = []
    for prev_player, prev_reply in history[-turns:]:
        parts.append(f"玩家：{prev_player}")
        parts.append(f"阿禾：{prev_reply}")
    parts.append(f"玩家：{player}")
    return "\n".join(parts)


def make_samples(bank, texts, beats, bank_map, count: int, seed: int,
                 context_turns: int, heldout: bool = False):
    """Teacher-driven dialogues; target is always an index in the small table."""
    from .corpus import HELDOUT as TEST

    phrases = TEST if heldout else SEEDS
    index = {text: i for i, text in enumerate(texts)}
    beat_ids = {beat: i for i, beat in enumerate(BEATS)}
    bank_candidates: dict[str, list[int]] = {}
    for idx, gid in bank_map.items():
        bank_candidates.setdefault(bank.beats[idx], []).append(idx)
    by_beat: dict[str, list[int]] = {}
    for gid, beat in enumerate(beats):
        by_beat.setdefault(beat, []).append(gid)

    rng = random.Random(seed)
    samples = []
    while len(samples) < count:
        state = new_state()
        history: list[tuple[str, str]] = []
        for _ in range(rng.randint(6, 12)):
            intent = rng.choice(list(phrases))
            player = rng.choice(phrases[intent])
            context = format_context(history, player, context_turns)
            rule_tpl, beat = teacher_turn(player, state)
            gid = None
            if beat in bank_candidates:
                bidx = bank.pick_index(beat, state, player, avoid=[],
                                       candidates=bank_candidates[beat])
                if bidx is not None:
                    gid = bank_map.get(bidx)
            else:
                gid = index.get(rule_tpl)
            if gid is None:
                continue
            samples.append((context, gid, beat_ids[beat]))
            reply = format_line(texts[gid], state)
            history.append((player, reply))
            if len(samples) >= count:
                break
    return samples


def encode(enc, texts: list[str], cache: str, rebuild: bool = False):
    if not rebuild and os.path.exists(cache):
        data = np.load(cache)
        if len(data["x"]) == len(texts):
            print(f"features from {os.path.basename(cache)}: {len(texts)}")
            return data["x"]
    chunks = []
    start = time.perf_counter()
    for i in range(0, len(texts), 256):
        chunks.append(enc.embed(texts[i:i + 256]))
        done = min(i + 256, len(texts))
        print(f"  encode {done}/{len(texts)}  {time.perf_counter() - start:.1f}s")
    array = torch.cat(chunks).numpy()
    np.savez_compressed(cache, x=array)
    return array


def build_candidates(texts, beats, enc, pos_k: int, neg_k: int,
                     temperature: float, seed: int = 7):
    """Soft listwise targets: 同 beat 近邻是正例，其他 beat 是负例。"""
    table = F.normalize(enc.embed(texts).detach(), dim=-1)
    by_beat: dict[str, list[int]] = {}
    for idx, beat in enumerate(beats):
        by_beat.setdefault(beat, []).append(idx)
    rng = random.Random(seed)
    cand_ids = np.zeros((len(texts), pos_k + 1 + neg_k), dtype=np.int64)
    cand_probs = np.zeros((len(texts), pos_k + 1 + neg_k), dtype=np.float32)
    for gid in range(len(texts)):
        beat = beats[gid]
        same = [i for i in by_beat[beat] if i != gid]
        if same:
            sims = (table[same] @ table[gid]).numpy()
            order = np.argsort(-sims)[:pos_k]
            pos = [same[i] for i in order]
            sims = sims[order]
        else:
            pos, sims = [], np.zeros(0, dtype=np.float32)
        pool = [gid] + pos
        pool_sims = np.concatenate([[1.0], sims])
        weights = np.exp((pool_sims - pool_sims.max()) / max(1e-3, temperature))
        weights /= weights.sum()
        others = [i for i in range(len(texts)) if beats[i] != beat]
        neg = rng.sample(others, min(neg_k, len(others))) if others else []
        ids = pool + neg
        cand_ids[gid, :len(ids)] = ids
        cand_probs[gid, :len(pool)] = weights
    return cand_ids, cand_probs


def cluster_table(texts, beats, enc, threshold: float = 0.9):
    """把同 beat、语义相近的句子折成一个语义 ID；返回代表句与映射。"""
    table = F.normalize(enc.embed(texts).detach(), dim=-1)
    parent = list(range(len(texts)))
    # 这些词成对出现时，不能因为向量像就合并（“外面热”和“外面冷”）。
    conflicts = [("热", "冷"), ("大", "小"), ("多", "少"), ("快", "慢"),
                 ("高", "低"), ("远", "近"), ("开心", "难过"), ("喜欢", "讨厌"),
                 ("甜", "苦"), ("干", "湿"), ("亮", "暗"), ("进", "出"),
                 ("来", "去"), ("新", "旧"), ("胖", "瘦"), ("真", "假")]

    def clash(a: str, b: str) -> bool:
        return any((x in a and y in b) or (y in a and x in b)
                   for x, y in conflicts)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: dict[str, list[int]] = {}
    for idx, beat in enumerate(beats):
        groups.setdefault(beat, []).append(idx)
    for idx in groups.values():
        block = table[idx]
        sims = block @ block.T
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if float(sims[i, j]) >= threshold \
                        and not clash(texts[idx[i]], texts[idx[j]]):
                    union(idx[i], idx[j])

    clusters: dict[int, list[int]] = {}
    for i in range(len(texts)):
        clusters.setdefault(find(i), []).append(i)
    new_texts, new_beats, mapping = [], [], [0] * len(texts)
    for members in clusters.values():
        center = table[members].mean(0)
        medoid = members[int((table[members] @ center).argmax())]
        cid = len(new_texts)
        new_texts.append(texts[medoid])
        new_beats.append(beats[medoid])
        for old in members:
            mapping[old] = cid
    print(f"语义去重：{len(texts)} → {len(new_texts)} 个语义 ID"
          f"（cos ≥ {threshold}，同 beat 内合并）")
    return new_texts, new_beats, mapping


class TableNet(nn.Module):
    def __init__(self, dim: int, hidden: int, n_sent: int, n_beats: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.sent = nn.Linear(hidden, n_sent)
        self.beat = nn.Linear(hidden, n_beats)

    def forward(self, x):
        z = self.trunk(x)
        return self.sent(z), self.beat(z)


class ContextGRUNet(nn.Module):
    """逐轮编码：GRU 保留轮次顺序，再把最后一轮和整段池化拼起来。"""

    def __init__(self, dim: int, hidden: int, n_sent: int, n_beats: int):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1),
        )
        self.sent = nn.Linear(hidden, n_sent)
        self.beat = nn.Linear(hidden, n_beats)

    def forward(self, x, mask):
        z = torch.relu(self.proj(x))
        out, _ = self.gru(z)
        lengths = mask.sum(1).clamp(min=1).long().squeeze(-1) - 1
        cur = out[torch.arange(out.size(0), device=out.device), lengths]
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        zz = self.trunk(torch.cat([cur, pooled], dim=-1))
        return self.sent(zz), self.beat(zz)


def line_list(context: str) -> list[str]:
    return [line for line in context.split("\n") if line.strip()]


def pack_lines(rows, line_index: dict[str, int], line_x):
    width = line_x.shape[1]
    length = max(len(line_list(row[0])) for row in rows)
    x = torch.zeros(len(rows), length, width)
    mask = torch.zeros(len(rows), length, 1)
    for i, row in enumerate(rows):
        for j, line in enumerate(line_list(row[0])):
            x[i, j] = torch.from_numpy(line_x[line_index[line]])
            mask[i, j] = 1.0
    return x, mask


def evaluate(model, x, y, b, class_beats, n_beats, mask=None):
    model.eval()
    with torch.no_grad():
        logits, beat_logits = model(x, mask) if mask is not None else model(x)
    top1 = (logits.argmax(1) == y).float().mean().item()
    top5 = (logits.topk(5, dim=1).indices == y[:, None]).any(1).float().mean().item()
    beat_acc = (beat_logits.argmax(1) == b).float().mean().item()
    pred_beat = beat_logits.argmax(1)
    mask = class_beats[None, :] == pred_beat[:, None]
    masked = logits.masked_fill(~mask, -1e9)
    masked_top1 = (masked.argmax(1) == y).float().mean().item()
    acceptable = (class_beats[logits.argmax(1)] == class_beats[y]).float().mean().item()
    return {"top1": top1, "top5": top5, "beat": beat_acc,
            "masked_top1": masked_top1, "acceptable": acceptable,
            "unique_pred": int(logits.argmax(1).unique().numel())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--eval-samples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--arch", choices=["mlp", "gru"], default="mlp")
    parser.add_argument("--loss", choices=["ce", "listwise"], default="ce")
    parser.add_argument("--pos-k", type=int, default=12)
    parser.add_argument("--neg-k", type=int, default=12)
    parser.add_argument("--sim-temp", type=float, default=0.1)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dedup-threshold", type=float, default=0.0)
    parser.add_argument("--table-size", type=int, default=1000)
    parser.add_argument("--context-turns", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = get_bank()
    texts, beats, bank_map = build_table(args.table_size, args.seed)
    print(f"句表 {len(texts)} 句，覆盖 {len(set(beats))} 个 beat；设备 {device}")

    train_raw = make_samples(bank, texts, beats, bank_map, args.samples,
                             args.seed, args.context_turns, heldout=False)
    eval_raw = make_samples(bank, texts, beats, bank_map, args.eval_samples,
                            args.seed + 1000, args.context_turns, heldout=True)
    print(f"训练 {len(train_raw)} 条，评测 {len(eval_raw)} 条（HELDOUT 说法）")

    from ..sentence import HFSentenceEncoder
    enc = HFSentenceEncoder(ENC_MODEL, pooling="cls", max_length=args.max_length)
    if args.dedup_threshold > 0:
        texts, beats, mapping = cluster_table(texts, beats, enc,
                                              args.dedup_threshold)
        train_raw = [(c, mapping[g], b) for c, g, b in train_raw]
        eval_raw = [(c, mapping[g], b) for c, g, b in eval_raw]
    tag = f"{args.table_size}_{args.samples}_{args.seed}_{args.context_turns}_{args.max_length}"
    x_mask = ex_mask = None
    if args.arch == "gru":
        unique = sorted({line for row in train_raw + eval_raw
                         for line in line_list(row[0])})
        line_x = encode(enc, unique,
                        os.path.join(HERE, f"context_lines_{tag}.npz"), args.rebuild)
        line_index = {line: i for i, line in enumerate(unique)}
        x, x_mask = pack_lines(train_raw, line_index, line_x)
        ex, ex_mask = pack_lines(eval_raw, line_index, line_x)
        ex, ex_mask = ex.to(device), ex_mask.to(device)
        dim = line_x.shape[1]
        print(f"轮级编码：{len(unique)} 条唯一行，序列长度 {x.shape[1]}")
    else:
        train_x = encode(enc, [row[0] for row in train_raw],
                         os.path.join(HERE, f"context_train_{tag}.npz"), args.rebuild)
        eval_x = encode(enc, [row[0] for row in eval_raw],
                        os.path.join(HERE, f"context_eval_{tag}.npz"), args.rebuild)
        x = torch.tensor(train_x, dtype=torch.float32)
        ex = torch.tensor(eval_x, dtype=torch.float32).to(device)
        dim = x.shape[1]
    train_y = np.array([row[1] for row in train_raw], dtype=np.int64)
    train_b = np.array([row[2] for row in train_raw], dtype=np.int64)
    eval_y = np.array([row[1] for row in eval_raw], dtype=np.int64)
    eval_b = np.array([row[2] for row in eval_raw], dtype=np.int64)
    if args.data_only:
        print(Counter(beats[i] for i in train_y).most_common())
        return 0

    y = torch.tensor(train_y)
    b = torch.tensor(train_b)
    ey = torch.tensor(eval_y).to(device)
    eb = torch.tensor(eval_b).to(device)
    class_beats = torch.tensor([BEATS.index(beat) for beat in beats]).to(device)

    model_cls = ContextGRUNet if args.arch == "gru" else TableNet
    model = model_cls(dim, args.hidden, len(texts), len(BEATS)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数 {params/1e6:.2f}M，输出 {len(texts)} 类，架构 {args.arch}")

    cand_ids = cand_probs = None
    if args.loss == "listwise":
        cand_ids_np, cand_probs_np = build_candidates(
            texts, beats, enc, args.pos_k, args.neg_k, args.sim_temp)
        cand_ids = torch.tensor(cand_ids_np, device=device)
        cand_probs = torch.tensor(cand_probs_np, device=device)
        print(f"listwise 候选 {cand_ids.shape[1]} 个/句："
              f"{args.pos_k} 个同 beat 近邻 + {args.neg_k} 个其他 beat")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(x) / args.batch)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warm_steps = max(1, int(total_steps * args.warmup))

    def lr_lambda(step: int) -> float:
        if step < warm_steps:
            return (step + 1) / warm_steps
        progress = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        total = sent_total = beat_total = 0.0
        start = time.perf_counter()
        for step, i in enumerate(range(0, len(order), args.batch)):
            idx = order[i:i + args.batch]
            xb, yb, bb = x[idx].to(device), y[idx].to(device), b[idx].to(device)
            if x_mask is not None:
                logits, beat_logits = model(xb, x_mask[idx].to(device))
            else:
                logits, beat_logits = model(xb)
            sent_loss = F.cross_entropy(logits, yb,
                                        label_smoothing=args.label_smoothing) \
                if args.loss == "ce" else None
            if args.loss == "listwise":
                cand = cand_ids[yb]
                target = cand_probs[yb]
                logp = F.log_softmax(logits.gather(1, cand), dim=1)
                sent_loss = -(target * logp).sum(1).mean()
            beat_loss = F.cross_entropy(beat_logits, bb)
            loss = sent_loss + 0.3 * beat_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            total += loss.item()
            sent_total += sent_loss.item()
            beat_total += beat_loss.item()
            if step % 40 == 0:
                print(f"epoch {epoch} step {step}/{len(order)//args.batch} "
                      f"loss {loss.item():.3f} sent {sent_loss.item():.3f} "
                      f"beat {beat_loss.item():.3f}")
        metrics = evaluate(model, ex, ey, eb, class_beats, len(BEATS), ex_mask)
        metrics.update(epoch=epoch, seconds=round(time.perf_counter() - start, 1),
                       train_loss=round(total / max(1, step + 1), 3),
                       sent_loss=round(sent_total / max(1, step + 1), 3),
                       beat_loss=round(beat_total / max(1, step + 1), 3))
        print(f"epoch {epoch}: {metrics}")
        if best is None or metrics["top1"] > best["top1"]:
            best = metrics
            torch.save({"state": model.state_dict(),
                        "config": {"dim": dim, "hidden": args.hidden,
                                   "n_sent": len(texts), "n_beats": len(BEATS),
                                   "arch": args.arch},
                        "texts": texts, "beats": beats}, CKPT)
    report = {"table_size": len(texts), "params": params,
              "train": len(train_raw), "eval": len(eval_raw),
              "loss": args.loss, "dedup_threshold": args.dedup_threshold,
              "best": best,
              "random": {"top1": 1 / len(texts), "top5": 5 / len(texts),
                         "beat": 1 / len(BEATS)}}
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    model.eval()
    with torch.no_grad():
        logits, _ = (model(ex[:8], ex_mask[:8]) if ex_mask is not None
                     else model(ex[:8]))
    print("\n样例：")
    for (context, target, _), pred in zip(eval_raw[:8], logits.argmax(1).tolist()):
        print("上下文：", context.replace("\n", " / ")[:90])
        print("  目标：", texts[target])
        print("  预测：", texts[pred])
    print(f"\n报告 {REPORT}，权重 {CKPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
