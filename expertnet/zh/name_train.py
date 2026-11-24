"""Targeted name-memory curriculum (the weakest raw category).

Training chains always put 1-4 distractor turns between "我叫X" and the recall
question. Evaluation uses different HELDOUT recall phrasings, so improvement is
generalization, not memorising one sentence.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch

from ..atoms import make_model
from .adversary import Attacker, audit, label_session
from .adversarial import clone_state, export, fine_tune
from .chat import META, components
from .corpus import HELDOUT, SEEDS
from .scaling import deploy_audit, test_chains
from .train import play

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["小明", "阿木", "小竹", "小满", "阿杰", "小禾"]
TRAIN_RECALL = ["你记得我叫什么吗？", "我叫什么名字？", "还记得我叫什么吗？",
                "我的名字你还记得吗？"]
TRAIN_DISTRACT = ["你好。", "你在干嘛？", "草药园在哪里？", "今天天气怎么样？",
                  "我无聊。", "讲个笑话。", "这个给你。", "谢谢。", "你多大了？"]


def _recall_pool(heldout: bool) -> list[str]:
    if not heldout:
        return TRAIN_RECALL
    pool = HELDOUT["name"]
    return [t for t in pool if "记得" in t or "名字" in t or "我叫什么" in t]


def make_chains(count: int, seed: int, heldout: bool = False) -> list[dict]:
    rng = random.Random(seed)
    recalls = _recall_pool(heldout)
    distract = HELDOUT["personal"] + HELDOUT["greeting"] if heldout else TRAIN_DISTRACT
    chains = []
    for _ in range(count):
        texts = []
        if rng.random() < 0.65:
            texts.append(f"我叫{rng.choice(NAMES)}。")
        for _ in range(rng.randint(1, 4)):
            texts.append(rng.choice(distract))
        texts.append(rng.choice(recalls))
        for _ in range(rng.randint(1, 2)):
            texts.append(rng.choice(distract))
        texts.append(rng.choice(recalls))
        chains.append({"texts": texts, "trust": 35, "player_name": ""})
    return chains


def _name_metrics(records: list[dict]) -> dict:
    rows = [r for r in records if r["category"] == "name"]
    if not rows:
        return {"n": 0, "atom": 0.0, "behaviour": 0.0, "thought": 0.0}
    return {"n": len(rows),
            "atom": round(sum(r["hit"] for r in rows) / len(rows), 3),
            "behaviour": round(sum(r["behaviour"] for r in rows) / len(rows), 3),
            "thought": round(sum(r["thought_hit"] for r in rows) / len(rows), 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--general", type=int, default=300)
    parser.add_argument("--eval", type=int, default=200)
    parser.add_argument("--init", default=None,
                        help="optional candidate .pt; promote to main only if accepted")
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    loaded = components()
    if loaded is None:
        return 1
    _, hf, _, _ = loaded
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu")
        model = make_model(ckpt["config"])
        model.load_state_dict(ckpt["state"])
        model.eval()
        atoms, beats = ckpt["atoms"], ckpt["beats"]
        config = ckpt["config"]
        encoder = ckpt.get("encoder", {})
    else:
        model, _, atoms, beats = loaded
        with open(META, encoding="utf-8") as fh:
            meta = json.load(fh)
        config, encoder = meta["config"], meta.get("encoder", {})
    atom_index = {a: i for i, a in enumerate(atoms)}
    beat_index = {b: i for i, b in enumerate(beats)}
    base_rows = play(SEEDS, 300, 6, seed=100)
    line_beat = {}
    for session in base_rows:
        for row in session:
            line_beat.setdefault(row[2], row[3])
    atom_beats = torch.tensor([beat_index.get(line_beat.get(a, ""), -1)
                               for a in atoms])
    base_state = clone_state(model)

    name_test = make_chains(args.eval, args.seed + 900, heldout=True)
    overall = test_chains(150, 14)
    records, _ = audit(model, hf, name_test, atoms, atom_index, beat_index,
                       atom_beats, tag="baseline-name")
    base_name = _name_metrics(records)
    _, base_overall = audit(model, hf, overall, atoms, atom_index, beat_index,
                            atom_beats, tag="baseline-all")
    base_deploy = deploy_audit(model, hf, atoms, overall)
    print(f"baseline name={base_name} overall_atom={base_overall['atom']:.3f} "
          f"deploy={base_deploy['overall']:.3f}")

    train = make_chains(args.chains, args.seed, heldout=False)
    replay = [{"texts": [row[0] for row in s], "trust": 35, "player_name": ""}
              for s in play(SEEDS, 300, 6, seed=300)]
    general = Attacker(args.seed + 5).generate(args.general, 14, seed=args.seed + 6)
    rows = ([label_session(s) for s in train]
            + [label_session(s) for s in replay]
            + [label_session(s) for s in general])
    fine_tune(model, rows, hf, atoms, beats, args.epochs, args.lr)

    records, _ = audit(model, hf, name_test, atoms, atom_index, beat_index,
                       atom_beats, tag="tuned-name")
    name_after = _name_metrics(records)
    _, overall_after = audit(model, hf, overall, atoms, atom_index, beat_index,
                             atom_beats, tag="tuned-all")
    deploy_after = deploy_audit(model, hf, atoms, overall)
    print(f"tuned    name={name_after} overall_atom={overall_after['atom']:.3f} "
          f"deploy={deploy_after['overall']:.3f}")

    accepted = (name_after["atom"] >= base_name["atom"] + 0.10
                and overall_after["atom"] >= base_overall["atom"] - 0.015
                and deploy_after["overall"] >= base_deploy["overall"] - 0.003)
    print("PROMOTED" if accepted else "kept baseline")
    if accepted:
        export(model, atoms, beats, config, encoder)
    else:
        model.load_state_dict(base_state)

    report = {"baseline": {"name": base_name, "overall": base_overall,
                           "deploy": base_deploy},
              "tuned": {"name": name_after, "overall": overall_after,
                        "deploy": deploy_after},
              "accepted": accepted, "train_chains": args.chains,
              "init": args.init}
    with open(os.path.join(HERE, "name_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
