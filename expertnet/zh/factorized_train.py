"""因子化句表训练：模型只输出模式 + 槽位，组合结果必然在 exhaustive 表内。

上下文仍由 BGE + MLP 编码；输出不是 33 万类 softmax，而是：
    mode(固定句/组合句) + pattern + subj + verb + obj + adj + place + adv

目标来自当前语义簇模型（distillation），再用教师 DAgger 纠正补强。
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

from ..sentence import HFSentenceEncoder
from .bank import get_bank
from .context_train import build_table, cluster_table, make_samples
from .exhaustive import (ADJ, ADV, OBJ, PATTERNS, PLACE, SUBJ, VERB,
                         DEFAULT_OUT as EXHAUSTIVE_OUT)

HERE = os.path.dirname(os.path.abspath(__file__))
ENC_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_CKPT = os.path.join(HERE, "context_net_dedup09_30k.pt")
DEFAULT_MODEL = os.path.join(HERE, "factorized.pt")
REPORT = os.path.join(HERE, "factorized_report.json")

FIELDS = {
    "svo": ("subj", "verb", "obj"),
    "svo_neg": ("subj", "verb", "obj"),
    "ba": ("subj", "verb", "obj"),
    "place_svo": ("subj", "place", "verb", "obj"),
    "svc": ("subj", "adv", "adj"),
    "o_is_a": ("obj", "adj"),
    "s_de_o_a": ("subj", "obj", "adj"),
    "s_think_o_a": ("subj", "obj", "adj"),
    "place_has_o": ("place", "obj"),
    "want_svo": ("subj", "verb", "obj"),
    "can_svo": ("subj", "verb", "obj"),
    "like_svo": ("subj", "verb", "obj"),
    "no_svo": ("subj", "verb", "obj"),
    "le_svo": ("subj", "verb", "obj"),
    "s_place": ("subj", "place"),
    "o_place": ("obj", "place"),
}
VOCAB = {
    "subj": SUBJ, "verb": VERB, "obj": OBJ, "adj": ADJ,
    "place": PLACE, "adv": ADV,
}
TEMPLATE_KEY = {"subj": "s", "verb": "v", "obj": "o",
                "adj": "a", "place": "p", "adv": "adv"}
PATTERN_NAMES = [row[0] for row in PATTERNS]
PATTERN_TEMPLATE = {row[0]: row[1] for row in PATTERNS}


def slot_context(row: dict) -> str:
    """用行元数据造一个“已知槽位”的问题，给因子化头干净的监督。"""
    p = row["pattern"]
    s, v, o = row.get("subj", ""), row.get("verb", ""), row.get("obj", "")
    a, place, adv = row.get("adj", ""), row.get("place", ""), row.get("adv", "")
    return {
        "svo": f"{s}是不是{v}{o}？",
        "svo_neg": f"{s}是不是不{v}{o}？",
        "ba": f"{s}是不是把{o}{v}了？",
        "place_svo": f"{s}是不是在{place}{v}{o}？",
        "svc": f"{s}是不是{adv}{a}？",
        "o_is_a": f"{o}是不是{a}的？",
        "s_de_o_a": f"{s}的{o}是不是很{a}？",
        "s_think_o_a": f"{s}是不是觉得{o}很{a}？",
        "place_has_o": f"{place}是不是有{o}？",
        "want_svo": f"{s}是不是想{v}{o}？",
        "can_svo": f"{s}是不是会{v}{o}？",
        "like_svo": f"{s}是不是喜欢{v}{o}？",
        "no_svo": f"{s}是不是没有{v}{o}？",
        "le_svo": f"{s}是不是{v}了{o}？",
        "s_place": f"{s}是不是在{place}？",
        "o_place": f"{o}是不是在{place}？",
    }[p]


def slot_contexts(row: dict) -> list[str]:
    """同一个槽位组合的多种问法，避免模型只背一个模板。"""
    base = slot_context(row)
    return [
        base,
        f"我想问一下，{base}",
        f"听说{base}",
        f"{base}对不对？",
        f"你告诉我，{base}",
    ]


def row_label(row: dict) -> dict:
    return {"mode": 0, "pattern": row["pattern"],
            "fixed_text": row["text"], **row}


HISTORY = [
    [("你好。", "你回来啦。")],
    [("我叫小明。", "我记住小明了。"), ("今天有点冷。", "要下雨了，蚂蚁都在搬家。")],
    [("你在干嘛？", "我在数蚂蚁。")],
    [("阿团呢？", "阿团在柜台下面睡觉。")],
]


def with_history(context: str, rng: random.Random) -> str:
    """一半样本加 0–2 轮无关历史，训练“只看最后一句”。"""
    if rng.random() >= 0.5:
        return context
    turns = rng.choice(HISTORY)
    lines = []
    for player, reply in turns:
        lines.append(f"玩家：{player}")
        lines.append(f"阿禾：{reply}")
    lines.append(f"玩家：{context}")
    return "\n".join(lines)


def load_rows(path: str, limit: int, seed: int):
    rows, fixed = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["pattern"] == "fixed":
                fixed.append(row)
            else:
                rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit], fixed


def encode(enc, texts, cache: str, rebuild=False):
    if not rebuild and os.path.exists(cache):
        data = np.load(cache)
        if len(data["x"]) == len(texts):
            return data["x"]
    chunks = []
    for i in range(0, len(texts), 256):
        chunks.append(enc.embed(texts[i:i + 256]))
        print(f"  encode {min(i + 256, len(texts))}/{len(texts)}")
    array = torch.cat(chunks).numpy()
    np.savez_compressed(cache, x=array)
    return array


class FactorizedNet(nn.Module):
    def __init__(self, dim: int, hidden: int, n_fixed: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mode = nn.Linear(hidden, 2)
        self.fixed = nn.Linear(hidden, n_fixed)
        self.pattern = nn.Linear(hidden, len(PATTERN_NAMES))
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden, len(vocab)) for name, vocab in VOCAB.items()
        })

    def forward(self, x):
        z = self.trunk(x)
        out = {"mode": self.mode(z), "fixed": self.fixed(z),
               "pattern": self.pattern(z)}
        for name, head in self.heads.items():
            out[name] = head(z)
        return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--rows", default=EXHAUSTIVE_OUT)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--row-limit", type=int, default=30000)
    parser.add_argument("--synth-per-row", type=int, default=4)
    parser.add_argument("--history-rate", type=float, default=0.5)
    parser.add_argument("--table-size", type=int, default=30000)
    parser.add_argument("--dedup-threshold", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--extra", default=None,
                        help="教师 DAgger JSONL（context/target/beat）")
    parser.add_argument("--out", default=DEFAULT_MODEL)
    args = parser.parse_args()

    bank = get_bank()
    raw_texts, raw_beats, bank_map = build_table(args.table_size, args.seed)
    enc = HFSentenceEncoder(ENC_MODEL, pooling="cls", max_length=160)
    texts, beats, mapping = cluster_table(raw_texts, raw_beats, enc,
                                          args.dedup_threshold)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt["config"]["n_sent"] != len(texts):
        raise SystemExit("checkpoint / table mismatch")
    cluster_bank_map = {bidx: mapping[ridx] for bidx, ridx in bank_map.items()}

    train_raw = make_samples(bank, texts, beats, cluster_bank_map, args.samples,
                             args.seed, 4, heldout=False)
    eval_raw = make_samples(bank, texts, beats, cluster_bank_map, args.eval_samples,
                            args.seed + 1000, 4, heldout=True)
    extra_count = 0
    if args.extra and os.path.exists(args.extra):
        with open(args.extra, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                train_raw.append((row["context"], int(row["target"]),
                                  int(row["beat"])))
                extra_count += 1
    print(f"教师回流 {extra_count} 条")
    rows, fixed = load_rows(args.rows, args.row_limit, args.seed)
    split = max(1, int(len(rows) * 0.9))
    synth_train_rows, synth_eval_rows = rows[:split], rows[split:]
    synth_train = []
    rng = random.Random(args.seed + 7)
    for row in synth_train_rows:
        for context in slot_contexts(row)[:args.synth_per_row]:
            if args.history_rate > 0:
                context = with_history(context, rng)
            synth_train.append((context, row))
    synth_eval = [(slot_context(r), r) for r in synth_eval_rows]
    print(f"组合句候选 {len(rows)}，固定句 {len(fixed)}；"
          f"训练 {len(train_raw)} + 合成 {len(synth_train)}，"
          f"评测 {len(eval_raw)} + 合成 {len(synth_eval)}")

    fixed_texts = [row["text"] for row in fixed]
    fixed_index = {text: i for i, text in enumerate(fixed_texts)}
    row_texts = [row["text"] for row in rows]
    row_x = torch.tensor(encode(enc, row_texts,
                                os.path.join(HERE, f"factorized_rows_{args.row_limit}_{args.seed}.npz")))
    row_x = F.normalize(row_x, dim=-1)
    target_texts = sorted({texts[gid] for _, gid, _ in train_raw + eval_raw})

    # 目标句 -> 固定句或最近的组合行
    target_rows = {}
    target_x = F.normalize(torch.tensor(encode(
        enc, target_texts,
        os.path.join(HERE, f"factorized_targets_{args.row_limit}_{args.seed}.npz"))), dim=-1)
    nearest = []
    for i in range(0, len(target_x), 128):
        sims = target_x[i:i + 128] @ row_x.T
        nearest.extend(sims.argmax(1).tolist())
        print(f"  nearest {min(i + 128, len(target_x))}/{len(target_x)}")
    for text, idx in zip(target_texts, nearest):
        target_rows[text] = {"mode": 0, "row": rows[idx]}

    def labels(items):
        out = []
        for _, gid, _ in items:
            text = texts[gid]
            if text in fixed_index:
                out.append({"mode": 1, "fixed": fixed_index[text],
                            "fixed_text": text})
            else:
                row = target_rows[text]["row"]
                out.append({"mode": 0, "pattern": row["pattern"],
                            "fixed_text": text, **row})
        return out

    train_y = labels(train_raw) + [row_label(r) for _, r in synth_train]
    eval_y = labels(eval_raw) + [row_label(r) for _, r in synth_eval]
    train_contexts = ([row[0] for row in train_raw]
                      + [context for context, _ in synth_train])
    eval_contexts = ([row[0] for row in eval_raw]
                     + [context for context, _ in synth_eval])
    train_x = encode(enc, train_contexts,
                     os.path.join(HERE, f"factorized_train_x_{args.samples}_{args.seed}_{args.synth_per_row}_{args.history_rate}.npz"))
    eval_x = encode(enc, eval_contexts,
                    os.path.join(HERE, f"factorized_eval_x_{args.eval_samples}_{args.seed}_{args.synth_per_row}_{args.history_rate}.npz"))
    x = torch.tensor(train_x, dtype=torch.float32)
    ex = torch.tensor(eval_x, dtype=torch.float32)

    def pack(rows_y):
        n = len(rows_y)
        mode = torch.tensor([r["mode"] for r in rows_y])
        fixed_id = torch.tensor([r.get("fixed", 0) for r in rows_y])
        pattern = torch.tensor([PATTERN_NAMES.index(r["pattern"])
                                if r["mode"] == 0 else 0 for r in rows_y])
        slots = {name: torch.zeros(n, dtype=torch.long) for name in VOCAB}
        masks = {name: torch.zeros(n, dtype=torch.bool) for name in VOCAB}
        for i, r in enumerate(rows_y):
            if r["mode"] == 1:
                continue
            for name in FIELDS[r["pattern"]]:
                slots[name][i] = VOCAB[name].index(r[name])
                masks[name][i] = True
        return mode, fixed_id, pattern, slots, masks

    ty, tf, tp, ts, tm = pack(train_y)
    ey, ef, ep, es, em = pack(eval_y)
    model = FactorizedNet(x.shape[1], args.hidden, len(fixed_texts))
    print(f"模型参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        start = time.perf_counter()
        for step, i in enumerate(range(0, len(order), args.batch)):
            idx = order[i:i + args.batch]
            out = model(x[idx])
            loss = F.cross_entropy(out["mode"], ty[idx])
            fmask = ty[idx] == 1
            if fmask.any():
                loss += F.cross_entropy(out["fixed"][fmask], tf[idx][fmask])
            pmask = ~fmask
            if pmask.any():
                loss += F.cross_entropy(out["pattern"][pmask], tp[idx][pmask])
                for name in VOCAB:
                    m = tm[name][idx] & pmask
                    if m.any():
                        loss += F.cross_entropy(out[name][m], ts[name][idx][m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            out = model(ex)
        mode_acc = (out["mode"].argmax(1) == ey).float().mean().item()
        pm = ey == 0
        pattern_acc = ((out["pattern"].argmax(1) == ep).float()[pm].mean().item()
                       if pm.any() else 0.0)
        slot_correct = slot_total = 0
        for name in VOCAB:
            m = em[name]
            if m.any():
                slot_correct += (out[name].argmax(1)[m] == es[name][m]).sum().item()
                slot_total += m.sum().item()
        slot_acc = slot_correct / max(1, slot_total)
        print(f"epoch {epoch}: mode {mode_acc:.3f} pattern {pattern_acc:.3f} "
              f"slot {slot_acc:.3f} loss {total/max(1,step+1):.3f} "
              f"{time.perf_counter()-start:.1f}s")

    torch.save({"state": model.state_dict(),
                "config": {"dim": x.shape[1], "hidden": args.hidden,
                           "n_fixed": len(fixed_texts)},
                "fixed_texts": fixed_texts, "rows": rows}, args.out)

    syn_y = [row_label(r) for _, r in synth_eval]
    _, _, syn_p, syn_s, syn_m = pack(syn_y)
    syn_x = torch.tensor(encode(
        enc, [context for context, _ in synth_eval],
        os.path.join(HERE, f"factorized_synth_eval_{args.row_limit}_{args.seed}.npz")))
    with torch.no_grad():
        out_s = model(syn_x)
    syn_pattern = (out_s["pattern"].argmax(1) == syn_p).float().mean().item()
    correct = total = 0
    for name in VOCAB:
        m = syn_m[name]
        if m.any():
            correct += (out_s[name].argmax(1)[m] == syn_s[name][m]).sum().item()
            total += m.sum().item()
    syn_slot = correct / max(1, total)
    exact = 0
    for i, (_, row) in enumerate(synth_eval[:2000]):
        p = PATTERN_NAMES[int(out_s["pattern"][i].argmax())]
        vals = {TEMPLATE_KEY[name]: VOCAB[name][int(out_s[name][i].argmax())]
                for name in FIELDS[p]}
        exact += int(PATTERN_TEMPLATE[p].format(**vals) == row["text"])
    syn_exact = exact / max(1, min(2000, len(synth_eval_rows)))

    report = {"samples": len(train_raw), "extra": extra_count,
              "rows": len(rows), "fixed": len(fixed_texts),
              "row_limit": args.row_limit, "synth_per_row": args.synth_per_row,
              "hidden": args.hidden, "history_rate": args.history_rate,
              "mode_acc": mode_acc, "pattern_acc": pattern_acc,
              "slot_acc": slot_acc,
              "synth_pattern": syn_pattern, "synth_slot": syn_slot,
              "synth_exact": syn_exact}
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
