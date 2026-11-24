"""Data-efficiency ablation: what do sentence vectors buy per labelled sentence?

Trains v1 / v2h / v2t on 25%, 50% and 100% of the hand-labelled seeds, then reports
held-out accuracy, expert-intent MI and context gain (carried h vs reset, with
last_intent hidden). Writes expertnet/ablation.json.

    python -m expertnet.ablation
"""
from __future__ import annotations

import argparse
import json
import os
import types

from .probe import routing, scripted_sequence, session_accuracy
from .sentence import HFSentenceEncoder
from .train import run

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHS = ["v1", "v2h", "v2t"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fracs", default="0.25,0.5,1.0")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    hf = HFSentenceEncoder()
    rows = []
    print(f"{'arch':5} {'frac':>5} {'heldout':>8} {'regex':>6} {'ctx-gain':>9} {'MI':>6}")
    for kind in ARCHS:
        for frac in [float(x) for x in args.fracs.split(",")]:
            run_args = types.SimpleNamespace(
                arch=kind, rounds=args.rounds, epochs=args.epochs, batch=32,
                steps=8, sessions=args.sessions, experts=6, hidden=64, copies=8,
                train_frac=frac, distill=False, scripted=False, seed=args.seed,
                quiet=True)
            result = run(run_args, hf=hf)
            data = scripted_sequence()
            carried = session_accuracy(result["arch_obj"], result["model"], data,
                                       carry=True, use_last=False)
            reset = session_accuracy(result["arch_obj"], result["model"], data,
                                     carry=False, use_last=False)
            mi, usage, _ = routing(result["arch_obj"], result["model"], data)
            row = {"arch": kind, "train_frac": frac,
                   "heldout": round(result["heldout"], 3),
                   "regex": round(result["baseline"], 3),
                   "context_gain": round(carried - reset, 3),
                   "context_carried": round(carried, 3),
                   "expert_intent_mi": round(mi, 3), "gate_usage": usage}
            rows.append(row)
            print(f"{kind:5} {frac:5.2f} {row['heldout']:8.3f} {row['regex']:6.3f} "
                  f"{row['context_gain']:+9.3f} {row['expert_intent_mi']:6.3f}")

    with open(os.path.join(HERE, "ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("saved expertnet/ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
