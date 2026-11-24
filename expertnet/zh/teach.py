"""教师-学生一轮 DAgger：学生先说，规则老师纠正，写出可回流训练的样本。

    python -m expertnet.zh.teach --dialogues 200 --turns 8
    python -m expertnet.zh.context_train --init expertnet/zh/context_net_dedup09_30k.pt \
        --extra expertnet/zh/dagger.jsonl --epochs 3 --lr 3e-4 ...

学生看到的是自己的历史（on-policy），老师只给“这一轮应该落到哪个语义簇”。
这样能收集到学生真实会走到的状态，而不是只模仿老师的理想对话。
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F

from ..sentence import HFSentenceEncoder
from .bank import get_bank
from .context_train import (ContextGRUNet, build_table, cluster_table,
                            format_context, line_list)
from .corpus import HELDOUT, SEEDS
from .persona import BEATS, extract_name, format_line, new_state, teacher_turn

HERE = os.path.dirname(os.path.abspath(__file__))
ENC_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_CKPT = os.path.join(HERE, "context_net_dedup09_30k.pt")
DEFAULT_OUT = os.path.join(HERE, "dagger.jsonl")


def load_student(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    cfg.pop("arch", None)
    model = ContextGRUNet(**cfg).eval()
    model.load_state_dict(ckpt["state"])
    return model, ckpt


def student_predict(model, enc, history, player: str):
    context = format_context(history, player, 4)
    lines = line_list(context)
    vecs = enc.embed(lines).unsqueeze(0)
    mask = torch.ones(1, vecs.size(1), 1)
    with torch.no_grad():
        logits, _ = model(vecs, mask)
    probs = F.softmax(logits[0], dim=-1)
    idx = int(probs.argmax())
    return idx, float(probs[idx]), context


def render(text: str, history, player: str) -> str:
    name = extract_name(player)
    for prev, _ in reversed(history):
        if name:
            break
        name = extract_name(prev)
    last = history[-1][0] if history else ""
    return format_line(text, {"player_name": name, "last_player": last})


def teacher_target(bank, raw_index, raw_candidates, mapping, state, player: str):
    template, beat = teacher_turn(player, state)
    raw = raw_index.get(template)
    if raw is None:
        pool = raw_candidates.get(beat, [])
        raw = bank.pick_index(beat, state, player, candidates=pool)
    if raw is None or raw not in mapping:
        return None, beat
    return mapping[raw], beat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--dialogues", type=int, default=200)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--table-size", type=int, default=30000)
    parser.add_argument("--dedup-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--source", choices=["train", "eval"], default="eval")
    parser.add_argument("--low-confidence", type=float, default=0.25)
    args = parser.parse_args()

    bank = get_bank()
    raw_texts, raw_beats, bank_map = build_table(args.table_size, args.seed)
    enc = HFSentenceEncoder(ENC_MODEL, pooling="cls", max_length=160)
    texts, beats, mapping = cluster_table(raw_texts, raw_beats, enc,
                                          args.dedup_threshold)
    model, ckpt = load_student(args.checkpoint)
    if ckpt["config"]["n_sent"] != len(texts):
        raise SystemExit(f"checkpoint expects {ckpt['config']['n_sent']} clusters, "
                         f"built {len(texts)}; check table-size/threshold/seed")

    raw_index = {text: i for i, text in enumerate(raw_texts)}
    raw_candidates: dict[str, list[int]] = {}
    for bidx, ridx in bank_map.items():
        raw_candidates.setdefault(bank.beats[bidx], []).append(ridx)

    phrases = HELDOUT if args.source == "eval" else SEEDS
    intents = list(phrases)
    beat_ids = {beat: i for i, beat in enumerate(BEATS)}
    rng = random.Random(args.seed)
    rows = []
    turns = matches = low_conf = 0
    confusion = Counter()

    for _ in range(args.dialogues):
        tstate = new_state()
        history: list[tuple[str, str]] = []
        for _ in range(args.turns):
            intent = rng.choice(intents)
            player = rng.choice(phrases[intent])
            idx, conf, context = student_predict(model, enc, history, player)
            target, beat = teacher_target(bank, raw_index, raw_candidates,
                                          mapping, tstate, player)
            turns += 1
            student_beat = beats[idx]
            if target is not None:
                matches += int(idx == target)
                if idx != target or conf < args.low_confidence:
                    rows.append({"context": context, "target": int(target),
                                 "beat": beat_ids[beat], "student": int(idx),
                                 "confidence": round(conf, 4)})
                if student_beat != beat:
                    confusion[(student_beat, beat)] += 1
            if conf < args.low_confidence:
                low_conf += 1
            reply = render(texts[idx], history, player)
            history.append((player, reply))
            del history[:-8]
            # 老师的状态也按学生真实回复延续，记忆才对得上。
            if tstate.get("recent"):
                tstate["recent"][-1]["reply"] = reply
            if target is not None:
                tstate["last_beat"] = beat

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "turns": turns,
        "teacher_agreement": round(matches / max(1, turns), 4),
        "corrections": len(rows),
        "low_confidence": low_conf,
        "confusions": confusion.most_common(10),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
