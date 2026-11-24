"""Adversarial long-dialogue loop for 阿禾.

    python -m expertnet.zh.adversarial --rounds 3

Each round:
  1. Attacker writes 12-16 turn conversations and hard paraphrases.
  2. Student answers with its 64-d memory; Auditor records every turn.
  3. Misses (with their preceding context) are added to the training pool.
  4. Fine-tune from the current best model, then re-evaluate on HELDOUT and the
     fixed long suite. Keep the round only when the suite improves.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from ..atoms import evaluate, tensors
from .adversary import Attacker, audit, label_session
from .chat import META, PT, SAFE, components
from .corpus import HELDOUT, SEEDS
from .long_suite import SUITE
from .persona import zh_state_vector
from .thought import THOUGHTS, thought_of
from .train import play

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "adversarial_report.json")
LOG = os.path.join(HERE, "adversarial.jsonl")


def build_indices(atoms: list[str], beats: list[str], base_rows):
    atom_index = {atom: i for i, atom in enumerate(atoms)}
    beat_index = {beat: i for i, beat in enumerate(beats)}
    line_beat = {}
    for session in base_rows:
        for row in session:
            line_beat.setdefault(row[2], row[3])
    atom_beats = torch.tensor([beat_index.get(line_beat.get(atom, ""), -1)
                               for atom in atoms])
    return atom_index, beat_index, atom_beats


def fine_tune(model, sessions, hf, atoms, beats, epochs: int, lr: float) -> float:
    atom_index = {atom: i for i, atom in enumerate(atoms)}
    beat_index = {beat: i for i, beat in enumerate(beats)}
    thought_index = {name: i for i, name in enumerate(THOUGHTS)}
    groups = defaultdict(list)
    for session in sessions:
        groups[len(session)].append(session)
    prepared = []
    for group in groups.values():
        data = tensors(group, hf, state_fn=zh_state_vector)
        yi = torch.tensor([[atom_index.get(row[2], -1) for row in session]
                           for session in group])
        bi = torch.tensor([[beat_index.get(row[3], -1) for row in session]
                           for session in group])
        ti = torch.tensor([[
            thought_index[thought_of(row[1]["last_beat"], row[1]["player_name"], row[3])]
            for row in session] for session in group])
        prepared.append((data, yi, bi, ti))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = 0.0
    for _ in range(epochs):
        for data, yi, bi, ti in prepared:
            order = torch.randperm(len(yi))
            for start in range(0, len(order), 64):
                idx = order[start:start + 64]
                logits_i, logits_b, _ = model(data["vecs"][idx], data["states"][idx])
                flat_i = logits_i.reshape(-1, logits_i.size(-1))
                flat_b = logits_b.reshape(-1, logits_b.size(-1))
                target_i = yi[idx].reshape(-1)
                target_b = bi[idx].reshape(-1)
                mask_i = target_i >= 0
                mask_b = target_b >= 0
                loss = flat_i.sum() * 0.0
                if bool(mask_i.any()):
                    loss = loss + F.cross_entropy(flat_i[mask_i], target_i[mask_i])
                if bool(mask_b.any()):
                    loss = loss + 0.3 * F.cross_entropy(flat_b[mask_b], target_b[mask_b])
                if getattr(model, "last_thoughts", None) is not None:
                    loss = loss + 0.2 * F.cross_entropy(
                        model.last_thoughts.reshape(-1, len(THOUGHTS)),
                        ti[idx].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.detach())
    model.eval()
    return last


def clone_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def score(summary: dict) -> float:
    return summary["behaviour"] + 0.5 * summary["atom"]


def export(model, atoms, beats, config, encoder) -> None:
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    torch.save({"config": config, "state": state, "atoms": atoms,
                "beats": beats, "encoder": encoder}, PT)
    from safetensors.torch import save_file
    save_file(state, SAFE)
    with open(META, "w", encoding="utf-8") as fh:
        json.dump({"config": config, "atoms": atoms, "beats": beats,
                   "encoder": encoder}, fh, ensure_ascii=False, indent=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=80)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and log, but never overwrite the checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    loaded = components()
    if loaded is None:
        print("no checkpoint/encoder available")
        return 1
    model, hf, atoms, beats = loaded
    with open(META, encoding="utf-8") as fh:
        meta = json.load(fh)
    config, encoder = meta["config"], meta.get("encoder", {})

    base_rows = play(SEEDS, 500, 6, seed=100)
    eval_rows = play(HELDOUT, 120, 6, seed=200)
    atom_index, beat_index, atom_beats = build_indices(atoms, beats, base_rows)

    held = evaluate(model, hf, eval_rows, atom_index, beat_index, atom_beats,
                    state_fn=zh_state_vector)
    records, summary = audit(model, hf, SUITE, atoms, atom_index, beat_index,
                             atom_beats, tag="baseline")
    baseline_held, baseline_summary = held, summary
    baseline_score = score(summary)
    print(f"baseline: heldout={held['atom']:.3f} long_atom={summary['atom']:.3f} "
          f"long_behaviour={summary['behaviour']:.3f}")
    print(f"  categories: {json.dumps(summary['by_category'], ensure_ascii=False)}")

    best_state = clone_state(model)
    best_held, best_summary = held, summary
    best_score = score(summary)
    all_records = list(records)
    rounds = []
    hard_pool: list[str] = []

    for round_no in range(1, args.rounds + 1):
        attacker = Attacker(args.seed + round_no)
        fresh = attacker.generate(args.sessions, args.steps,
                                  seed=args.seed + round_no * 17,
                                  hard_texts=hard_pool or None)
        fresh_records, fresh_summary = audit(model, hf, fresh, atoms, atom_index,
                                             beat_index, atom_beats,
                                             tag=f"round{round_no}")
        misses = [r["text"] for r in fresh_records if not r["hit"]]
        hard_pool = (hard_pool + misses)[-300:]
        hard_sessions = attacker.harden(fresh, fresh_records)
        hard_rows = [label_session(s) for s in hard_sessions]
        long_rows = [label_session(s) for s in fresh]

        train_rows = base_rows + long_rows + hard_rows
        model.load_state_dict(best_state)
        loss = fine_tune(model, train_rows, hf, atoms, beats, args.epochs, args.lr)

        held = evaluate(model, hf, eval_rows, atom_index, beat_index, atom_beats,
                        state_fn=zh_state_vector)
        records, summary = audit(model, hf, SUITE, atoms, atom_index, beat_index,
                                 atom_beats, tag=f"round{round_no}-suite")
        gained = score(summary) > best_score + 0.002
        kept = gained and held["atom"] >= best_held["atom"] - 0.02
        if kept:
            best_state = clone_state(model)
            best_held, best_summary, best_score = held, summary, score(summary)
        else:
            model.load_state_dict(best_state)
        all_records.extend(fresh_records)
        all_records.extend(records)
        rounds.append({"round": round_no, "loss": round(loss, 3),
                       "attacker_turns": len(fresh) * args.steps,
                       "fresh": fresh_summary, "hard_sessions": len(hard_sessions),
                       "heldout": held["atom"], "long": summary,
                       "kept": kept})
        print(f"round {round_no}: fresh_atom={fresh_summary['atom']:.3f} "
              f"fresh_behaviour={fresh_summary['behaviour']:.3f} "
              f"fresh_thought={fresh_summary['thought']:.3f} "
              f"hard={len(hard_sessions)} -> heldout={held['atom']:.3f} "
              f"long_atom={summary['atom']:.3f} "
              f"long_behaviour={summary['behaviour']:.3f} "
              f"long_thought={summary['thought']:.3f} "
              f"{'KEPT' if kept else 'dropped'}")

    saved = best_score > baseline_score + 0.002 and not args.dry_run
    if saved:
        model.load_state_dict(best_state)
        export(model, atoms, beats, config, encoder)

    print(f"\nbest long: atom={best_summary['atom']:.3f} "
          f"behaviour={best_summary['behaviour']:.3f} "
          f"heldout={best_held['atom']:.3f}")
    print(f"categories: {json.dumps(best_summary['by_category'], ensure_ascii=False)}")
    misses = [r for r in all_records if not r["hit"]][:8]
    for miss in misses:
        print(f"  miss [{miss['tag']}] {miss['text']} -> {miss['pred']} "
              f"(want {miss['target']})")
    print("checkpoint updated" if saved else "checkpoint left unchanged")

    with open(LOG, "w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump({"baseline": {"heldout": baseline_held["atom"],
                                "long": baseline_summary},
                   "baseline_score": baseline_score, "rounds": rounds,
                   "best": {"heldout": best_held["atom"], "long": best_summary},
                   "saved": saved}, fh, ensure_ascii=False, indent=1)
    print(f"logged {len(all_records)} turns to {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
