"""因子化句表训练：模型只输出模式 + 槽位，组合结果必然在 exhaustive 表内。

上下文仍由 BGE + MLP 编码；输出不是 33 万类 softmax，而是：
    mode(固定句/组合句) + pattern + subj + verb + obj + adj + place + adv

真实对话的监督直接落在固定句上：教师先给 beat，再从该 beat 的固定句里取
一句。组合模式只由“显式槽位问句”合成数据监督，负责覆盖穷尽句表。
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
from .corpus import HELDOUT, SEEDS
from .exhaustive import (ADJ, ADV, OBJ, PATTERNS, PLACE, SUBJ, VERB,
                         DEFAULT_OUT as EXHAUSTIVE_OUT)
from .persona import format_line, new_state, teacher_turn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL_ENC = os.path.join(ROOT, "models", "bge-small-zh-v1.5")
ENC_MODEL = os.environ.get(
    "ENC_MODEL", LOCAL_ENC if os.path.isdir(LOCAL_ENC)
    else "BAAI/bge-small-zh-v1.5")
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
    combos, fixed = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["pattern"] == "fixed":
                fixed.append(row)
            else:
                combos.append(row)
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:limit], fixed


MAX_LINES = 12


def split_lines(context: str, max_lines: int = MAX_LINES) -> list[str]:
    return [line for line in context.split("\n") if line.strip()][-max_lines:]


def line_tensor(enc, contexts, cache: str):
    """每行单独编码，得到 [N, 12, dim] 和真实长度。"""
    if os.path.exists(cache):
        data = np.load(cache)
        if len(data["x"]) == len(contexts):
            return torch.tensor(data["x"]), torch.tensor(data["len"])
    split = [split_lines(context) for context in contexts]
    unique = sorted({line for lines in split for line in lines})
    print(f"  唯一对话行 {len(unique)} / 上下文 {len(contexts)}", flush=True)
    table = {line: i for i, line in enumerate(unique)}
    vectors = enc.embed(unique, batch=256)
    dim = vectors.shape[1]
    x = torch.zeros(len(split), MAX_LINES, dim)
    lengths = torch.zeros(len(split), dtype=torch.long)
    for i, lines in enumerate(split):
        lengths[i] = len(lines)
        for j, line in enumerate(lines):
            x[i, j] = vectors[table[line]]
    np.savez(cache, x=x.numpy(), len=lengths.numpy())
    return x, lengths


LEAD = set("嗯啊哦唔咦哼哎唉嘿")


def canonical_text(pool: list[str]) -> str:
    """每个 beat 选一句干净、长度适中的台词作为训练目标。"""
    candidates = [text for text in pool if "{" not in text] or pool

    def score(text: str) -> int:
        return abs(len(text) - 10) + (4 if text[0] in LEAD else 0)

    return min(candidates, key=score)


AUG_PREFIX = ["", "阿禾，", "喂，", "那个……", "我说，", "小阿禾，"]
AUG_SUFFIX = ["", "呀", "啊", "哦", "呢"]
AUG_SYN = [("娘", "妈妈"), ("名字", "称呼"), ("走", "离开"), ("笨", "蠢"),
           ("生气", "不高兴"), ("糖", "糖果"), ("看", "瞧"), ("说", "讲"),
           ("怕", "害怕"), ("喜欢", "爱")]


def augment(text: str, rng: random.Random, rate: float) -> str:
    """标签不变的说法扰动：加称呼/语气词、换同义词，教模型别被表面词带偏。"""
    if rate <= 0 or rng.random() >= rate:
        return text
    if rng.random() < 0.5:
        for old, new in AUG_SYN:
            if old in text:
                text = text.replace(old, new, 1)
                break
    body = text.rstrip("。？！")
    return f"{rng.choice(AUG_PREFIX)}{body}{rng.choice(AUG_SUFFIX)}。"


def real_samples(fixed_index: dict[str, int], canon: dict[str, str], count: int,
                 seed: int, turns: int, heldout: bool = False,
                 augment_rate: float = 0.0):
    """教师驱动的真实对话：标签直接是固定句 ID，不再映射到组合行。"""
    phrases = HELDOUT if heldout else SEEDS
    rng = random.Random(seed)
    samples = []
    while len(samples) < count:
        state = new_state()
        history: list[tuple[str, str]] = []
        for _ in range(rng.randint(6, 12)):
            intent = rng.choice(list(phrases))
            player = rng.choice(phrases[intent])
            surface = augment(player, rng, augment_rate)
            context = format_line_context(history, surface, turns)
            tpl, beat = teacher_turn(player, state)
            target = tpl if tpl in fixed_index else canon.get(beat)
            if target is None or target not in fixed_index:
                continue
            samples.append((context, fixed_index[target], beat))
            history.append((surface, format_line(target, state)))
            if len(samples) >= count:
                break
    return samples


def format_line_context(history, player: str, turns: int) -> str:
    parts = []
    for prev_player, prev_reply in history[-turns:]:
        parts.append(f"玩家：{prev_player}")
        parts.append(f"阿禾：{prev_reply}")
    parts.append(f"玩家：{player}")
    return "\n".join(parts)


class FactorizedNet(nn.Module):
    """逐行编码整段对话，再用 GRU 读最后状态。

    整段拼成一个 BGE 句向量会把最后一句话淹没在历史里（实测 4 轮历史时
    行为准确率从 0.68 掉到 0.14）；这里每行单独编码，历史只通过 GRU 传递。
    """

    def __init__(self, dim: int, hidden: int, n_fixed: int,
                 max_lines: int = MAX_LINES):
        super().__init__()
        self.max_lines = max_lines
        self.inp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(),
        )
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, hidden // 2), nn.GELU(), nn.Dropout(0.1),
        )
        # 双视角：GRU 读上下文，另一路只看最后一句，推理时相加。
        width = hidden // 2
        self.local_trunk = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, width), nn.GELU(),
        )
        self.mode = nn.Linear(width, 2)
        self.fixed = nn.Linear(width, n_fixed)
        self.pattern = nn.Linear(width, len(PATTERN_NAMES))
        self.heads = nn.ModuleDict({
            name: nn.Linear(width, len(vocab)) for name, vocab in VOCAB.items()
        })
        self.local = nn.ModuleDict({
            name: nn.Linear(width, out)
            for name, out in [("mode", 2), ("fixed", n_fixed),
                              ("pattern", len(PATTERN_NAMES))]
            + [(name, len(vocab)) for name, vocab in VOCAB.items()]
        })

    def forward(self, x, lengths):
        z = self.inp(x)
        out, _ = self.gru(z)
        idx = torch.arange(out.size(0), device=out.device)
        last = out[idx, lengths - 1]
        zc = self.trunk(last)
        zl = self.local_trunk(z[idx, lengths - 1])
        result = {"mode": self.mode(zc), "fixed": self.fixed(zc),
                  "pattern": self.pattern(zc)}
        for name, head in self.heads.items():
            result[name] = head(zc)
        for name, head in self.local.items():
            result[name + "_local"] = head(zl)
        return result


def weighted_ce(logits, target, weight):
    loss = F.cross_entropy(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=EXHAUSTIVE_OUT)
    parser.add_argument("--samples", type=int, default=120000,
                        help="真实对话训练样本数")
    parser.add_argument("--eval-samples", type=int, default=4000)
    parser.add_argument("--row-limit", type=int, default=200000)
    parser.add_argument("--synth-per-row", type=int, default=1)
    parser.add_argument("--synth-eval-limit", type=int, default=3000)
    parser.add_argument("--history-rate", type=float, default=0.5)
    parser.add_argument("--real-weight", type=float, default=2.5,
                        help="真实对话样本的损失权重，平衡合成槽位数据")
    parser.add_argument("--augment", type=float, default=0.0,
                        help="标签不变的说法扰动比例（称呼/语气词/同义词）")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cpu",
                        help="cpu / cuda；cuda 时句向量和训练都放 GPU")
    parser.add_argument("--out", default=DEFAULT_MODEL)
    parser.add_argument("--deploy-out", default=None,
                        help="只保留推理所需字段的部署权重")
    args = parser.parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) \
        else "cpu"
    print(f"设备 {device}", flush=True)

    combos, fixed = load_rows(args.rows, args.row_limit, args.seed)
    fixed_texts = [row["text"] for row in fixed]
    fixed_beats = [row.get("beat", "fixed") for row in fixed]
    fixed_index = {text: i for i, text in enumerate(fixed_texts)}
    by_beat: dict[str, list[str]] = {}
    for text, beat in zip(fixed_texts, fixed_beats):
        by_beat.setdefault(beat, []).append(text)
    canon = {beat: canonical_text(pool) for beat, pool in by_beat.items()}
    print(f"组合句 {len(combos)}，固定句 {len(fixed_texts)}，"
          f"beat {len(by_beat)}", flush=True)

    enc = HFSentenceEncoder(ENC_MODEL, pooling="cls", max_length=160,
                            device=device)
    train_real = real_samples(fixed_index, canon, args.samples, args.seed, 4,
                              augment_rate=args.augment)
    eval_real = real_samples(fixed_index, canon, args.eval_samples,
                             args.seed + 1000, 4, heldout=True)

    split = max(1, int(len(combos) * 0.9))
    synth_rows, synth_eval_rows = combos[:split], combos[split:]
    rng = random.Random(args.seed + 7)
    synth_train = []
    for row in synth_rows:
        for context in slot_contexts(row)[:args.synth_per_row]:
            if args.history_rate > 0:
                context = with_history(context, rng)
            synth_train.append((context, row))
    synth_eval = [(slot_context(r), r)
                  for r in synth_eval_rows[:args.synth_eval_limit]]
    print(f"真实 {len(train_real)}（留存 {len(eval_real)}）+ 合成 "
          f"{len(synth_train)}，合成评测 {len(synth_eval)}", flush=True)

    tag = (f"{args.samples}_{args.row_limit}_{args.synth_per_row}_"
           f"{args.seed}_{args.augment}")
    train_contexts = ([row[0] for row in train_real]
                      + [context for context, _ in synth_train])
    train_x, train_len = line_tensor(
        enc, train_contexts, os.path.join(HERE, f"fact_lines_{tag}.npz"))
    train_x = train_x.to(device)
    train_len = train_len.to(device)
    eval_contexts = ([row[0] for row in eval_real]
                     + [context for context, _ in synth_eval])
    eval_x, eval_len = line_tensor(
        enc, eval_contexts, os.path.join(HERE, f"fact_lines_eval_{tag}.npz"))
    eval_x = eval_x.to(device)
    eval_len = eval_len.to(device)
    train_y = ([{"mode": 1, "fixed": gid} for _, gid, _ in train_real]
               + [row_label(row) for _, row in synth_train])
    weight = torch.tensor([args.real_weight] * len(train_real)
                          + [1.0] * len(synth_train), dtype=torch.float32) \
        .to(device)
    eval_y = ([{"mode": 1, "fixed": gid} for _, gid, _ in eval_real]
              + [row_label(row) for _, row in synth_eval])

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
        return (mode.to(device), fixed_id.to(device), pattern.to(device),
                {k: v.to(device) for k, v in slots.items()},
                {k: v.to(device) for k, v in masks.items()})

    ty, tf, tp, ts, tm = pack(train_y)
    ey, ef, ep, es, em = pack(eval_y)
    model = FactorizedNet(train_x.shape[2], args.hidden, len(fixed_texts))
    model.to(device)
    print(f"模型参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        total = 0.0
        start = time.perf_counter()
        for step, i in enumerate(range(0, len(order), args.batch)):
            idx = order[i:i + args.batch]
            w = weight[idx]
            out = model(train_x[idx], train_len[idx])
            loss = (weighted_ce(out["mode"], ty[idx], w)
                    + 0.5 * weighted_ce(out["mode_local"], ty[idx], w))
            fmask = ty[idx] == 1
            if fmask.any():
                loss += weighted_ce(out["fixed"][fmask], tf[idx][fmask],
                                    w[fmask])
                loss += 0.5 * weighted_ce(out["fixed_local"][fmask],
                                          tf[idx][fmask], w[fmask])
            pmask = ~fmask
            if pmask.any():
                loss += weighted_ce(out["pattern"][pmask], tp[idx][pmask],
                                    w[pmask])
                loss += 0.5 * weighted_ce(out["pattern_local"][pmask],
                                          tp[idx][pmask], w[pmask])
                for name in VOCAB:
                    m = tm[name][idx] & pmask
                    if m.any():
                        loss += weighted_ce(out[name][m], ts[name][idx][m],
                                            w[m])
                        loss += 0.5 * weighted_ce(out[name + "_local"][m],
                                                  ts[name][idx][m], w[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(order)//args.batch} "
                      f"loss {total/(step+1):.3f}", flush=True)
        model.eval()
        with torch.no_grad():
            raw = model(eval_x, eval_len)
        out = {key: raw[key] + raw[key + "_local"]
               for key in ["mode", "fixed", "pattern", *VOCAB]}
        n_real = len(eval_real)
        mode_pred = out["mode"].argmax(1)
        fixed_pred = out["fixed"].argmax(1)
        real_mode = (mode_pred[:n_real] == 1).float().mean().item()
        beat_acc = 0.0
        for i, (_, gid, beat) in enumerate(eval_real):
            if int(mode_pred[i]) != 1:
                continue
            beat_acc += int(fixed_beats[int(fixed_pred[i])] == beat)
        beat_acc /= max(1, n_real)
        synth_mode_acc = (mode_pred[n_real:] == 0).float().mean().item()
        pattern_acc = ((out["pattern"].argmax(1)[n_real:] == ep[n_real:])
                       .float().mean().item())
        slot_correct = slot_total = 0
        for name in VOCAB:
            m = em[name][n_real:]
            if m.any():
                slot_correct += (out[name].argmax(1)[n_real:][m]
                                 == es[name][n_real:][m]).sum().item()
                slot_total += m.sum().item()
        slot_acc = slot_correct / max(1, slot_total)
        exact = 0
        for i, (context, row) in enumerate(synth_eval[:2000]):
            pos = n_real + i
            p = PATTERN_NAMES[int(out["pattern"][pos].argmax())]
            vals = {TEMPLATE_KEY[name]: VOCAB[name][int(out[name][pos].argmax())]
                    for name in FIELDS[p]}
            exact += int(PATTERN_TEMPLATE[p].format(**vals) == row["text"])
        synth_exact = exact / max(1, min(2000, len(synth_eval)))
        print(f"epoch {epoch}: 真实模式 {real_mode:.3f} beat {beat_acc:.3f} | "
              f"合成模式 {synth_mode_acc:.3f} pattern {pattern_acc:.3f} "
              f"slot {slot_acc:.3f} exact {synth_exact:.3f} "
              f"loss {total/max(1,step+1):.3f} "
              f"{time.perf_counter()-start:.1f}s", flush=True)

    print("\n留存说法样例：", flush=True)
    for i, (context, gid, beat) in enumerate(eval_real[:10]):
        with torch.no_grad():
            raw = model(eval_x[i:i + 1], eval_len[i:i + 1])
        out = {key: raw[key] + raw[key + "_local"]
               for key in ["mode", "fixed", "pattern", *VOCAB]}
        mode = int(out["mode"][0].argmax())
        if mode == 1:
            pred = fixed_texts[int(out["fixed"][0].argmax())]
        else:
            p = PATTERN_NAMES[int(out["pattern"][0].argmax())]
            vals = {TEMPLATE_KEY[name]: VOCAB[name][int(out[name][0].argmax())]
                    for name in FIELDS[p]}
            pred = PATTERN_TEMPLATE[p].format(**vals)
        last = context.split("\n")[-1]
        print(f"  {last}  ->  [{beat}] {pred}", flush=True)

    config = {"dim": train_x.shape[2], "hidden": args.hidden,
              "n_fixed": len(fixed_texts), "max_lines": MAX_LINES}
    torch.save({"state": model.state_dict(), "config": config,
                "fixed_texts": fixed_texts, "fixed_beats": fixed_beats,
                "rows": combos}, args.out)
    if args.deploy_out:
        torch.save({"state": model.state_dict(), "config": config,
                    "fixed_texts": fixed_texts,
                    "fixed_beats": fixed_beats}, args.deploy_out)
        print(f"写出部署权重 {args.deploy_out}", flush=True)

    report = {"samples": len(train_real), "eval_samples": len(eval_real),
              "rows": len(combos), "fixed": len(fixed_texts),
              "beats": len(by_beat), "row_limit": args.row_limit,
              "synth_per_row": args.synth_per_row, "hidden": args.hidden,
              "real_weight": args.real_weight,
              "real_mode": real_mode, "real_beat": beat_acc,
              "synth_mode": synth_mode_acc, "pattern_acc": pattern_acc,
              "slot_acc": slot_acc, "synth_exact": synth_exact}
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
