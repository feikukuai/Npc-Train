"""Evaluate all scaling variants on one fixed test set."""
from __future__ import annotations

import json
import os

import torch

from ..atoms import make_model
from .adversary import audit
from .chat import META, components
from .corpus import SEEDS
from .scaling import deploy_audit, test_chains
from .train import play

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "scaling_summary.json")


def main() -> int:
    loaded = components()
    if loaded is None:
        print("no checkpoint")
        return 1
    model, hf, atoms, beats = loaded
    atom_index = {a: i for i, a in enumerate(atoms)}
    beat_index = {b: i for i, b in enumerate(beats)}
    base_rows = play(SEEDS, 300, 6, seed=100)
    line_beat = {}
    for session in base_rows:
        for row in session:
            line_beat.setdefault(row[2], row[3])
    atom_beats = torch.tensor([beat_index.get(line_beat.get(a, ""), -1)
                               for a in atoms])

    test = test_chains(200, 14)
    rows = []
    models = [("baseline", model)]
    for size in (500, 5000, 50000):
        path = os.path.join(HERE, f"atoms_zh_scale{size}.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu")
        m = make_model(ckpt["config"])
        m.load_state_dict(ckpt["state"])
        models.append((f"scale{size}", m))

    print(f"{'model':10} {'atom':>6} {'behav':>6} {'think':>6} "
          f"{'general':>8} {'reason':>7} {'persona':>8} {'safety':>7} {'deploy':>7}")
    for name, m in models:
        _, raw = audit(m, hf, test, atoms, atom_index, beat_index, atom_beats,
                       tag=name)
        deploy = deploy_audit(m, hf, atoms, test)
        row = {"model": name, "raw": raw, "deploy": deploy}
        rows.append(row)
        print(f"{name:10} {raw['atom']:6.3f} {raw['behaviour']:6.3f} "
              f"{raw['thought']:6.3f} {deploy['general']:8.3f} "
              f"{deploy['reasoning']:7.3f} {deploy['persona']:8.3f} "
              f"{deploy['safety']:7.3f} {deploy['overall']:7.3f}")

    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump({"test": {"fixed_suite": 14, "random_chains": 200,
                            "turns_per_chain": 14},
                   "rows": rows}, fh, ensure_ascii=False, indent=1)
    print(f"saved {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
