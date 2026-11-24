"""Scaling study: do 500 / 5,000 / 50,000 long synthetic dialogues add reasoning?

Player / Teacher / Attacker agents generate long chains, the teacher labels every
turn, and the current checkpoint is fine-tuned from the same baseline for each
size. Evaluation uses disjoint HELDOUT phrasings plus the fixed long suite, and
reports raw and deployment (rule-gated) scores.

    python -m expertnet.zh.scaling --sizes 500,5000 --eval-chains 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from ..atoms import make_model, tensors
from .adversary import Attacker, audit, label_session, summarize
from .adversarial import fine_tune
from .chat import META, RULE_ONLY, SAFE, SECRET_ATOMS, components
from .corpus import HELDOUT, SEEDS
from .long_suite import SUITE
from .persona import new_state, teacher_turn, format_line, zh_state_vector
from .thought import THOUGHTS

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "scaling_report.json")


def test_chains(count: int = 200, steps: int = 14) -> list[dict]:
    """HELDOUT phrasings plus the fixed suite; never used for training."""
    rng = random.Random(9901)
    texts = [t for group in HELDOUT.values() for t in group]
    sessions = [{"texts": list(s["texts"]), "trust": s["trust"],
                 "player_name": ""} for s in SUITE]
    for _ in range(count):
        sessions.append({"texts": [rng.choice(texts) for _ in range(steps)],
                         "trust": rng.choice([10, 25, 35, 55, 70]),
                         "player_name": ""})
    return sessions


def _format(line: str, state: dict) -> str:
    if "{player_name}" in line:
        line = line.replace("{player_name}", state.get("player_name") or "你")
    if "{last_player}" in line:
        line = line.replace("{last_player}", state.get("last_player") or "你刚才的话")
    return line


@torch.no_grad()
def deploy_audit(model, hf, atoms, sessions) -> dict:
    """Score the actual hybrid policy, not just the raw selector."""
    model.eval()
    blocked_secret = {atom for atom in atoms if atom in SECRET_ATOMS}
    buckets = defaultdict(lambda: [0, 0])
    for session in sessions:
        state = new_state(session.get("trust", 35))
        if session.get("player_name"):
            state["player_name"] = session["player_name"]
        h = torch.zeros(1, model.hidden)
        for turn, text in enumerate(session["texts"]):
            previous = state.get("last_beat")
            target, beat = teacher_turn(text, state)
            rule = _format(target, state)
            if beat in RULE_ONLY:
                pred = rule
            else:
                atom_state = {**state, "last_beat": previous}
                vec = hf.embed([text]).unsqueeze(0)
                st = zh_state_vector(atom_state).reshape(1, 1, -1)
                logits, _, h = model(vec, st, h)
                logits = logits[0, 0]
                lock = state["trust"] < 40 and "secret" not in state.get("revealed", [])
                mask = torch.tensor([
                    ("{player_name}" in atom and not state.get("player_name"))
                    or ("{last_player}" in atom and not state.get("last_player"))
                    or (lock and atom in blocked_secret)
                    for atom in atoms])
                if bool(mask.any()):
                    logits = logits.masked_fill(mask, -1e9)
                pred = _format(atoms[int(logits.argmax())], state)
            hit = int(pred == rule)
            if beat in ("boundary", "threat", "secret_denied") or \
                    (beat == "secret" and state["trust"] < 40):
                key = "safety"
            elif beat in ("memory_recall", "taunt", "apology_meta") or \
                    previous in ("insult", "threat", "taunt"):
                key = "reasoning"
            elif beat in ("personal", "lore", "secret", "name_recall", "name_saved",
                          "name_unknown", "meta", "trade"):
                key = "persona"
            else:
                key = "general"
            buckets[key][0] += hit
            buckets[key][1] += 1
    result = {key: round(hits / n, 3) if n else 0.0
              for key, (hits, n) in buckets.items()}
    result["overall"] = round(sum(h for h, _ in buckets.values()) /
                              max(sum(n for _, n in buckets.values()), 1), 3)
    return result


def fit_variant(model, sessions, hf, atoms, beats, epochs: int, lr: float) -> None:
    rows = []
    for session in sessions:
        rows.append(label_session(session))
    fine_tune(model, rows, hf, atoms, beats, epochs, lr)


def save_variant(model, atoms, beats, config, encoder, size: int) -> str:
    path = os.path.join(HERE, f"atoms_zh_scale{size}.pt")
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    torch.save({"config": config, "state": state, "atoms": atoms,
                "beats": beats, "encoder": encoder}, path)
    try:
        from safetensors.torch import save_file
        save_file(state, path.replace(".pt", ".safetensors"))
        with open(path.replace(".pt", ".json"), "w", encoding="utf-8") as fh:
            json.dump({"config": config, "atoms": atoms, "beats": beats,
                       "encoder": encoder}, fh, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="500,5000")
    parser.add_argument("--eval-chains", type=int, default=200)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--replay", type=int, default=300)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    loaded = components()
    if loaded is None:
        print("no checkpoint/encoder")
        return 1
    model, hf, atoms, beats = loaded
    with open(META, encoding="utf-8") as fh:
        meta = json.load(fh)
    config, encoder = meta["config"], meta.get("encoder", {})
    atom_index = {atom: i for i, atom in enumerate(atoms)}
    beat_index = {beat: i for i, beat in enumerate(beats)}
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # line_beat from a replay corpus, for raw behaviour
    from .train import play
    base_rows = play(SEEDS, args.replay, 6, seed=100)
    line_beat = {}
    for session in base_rows:
        for row in session:
            line_beat.setdefault(row[2], row[3])
    atom_beats = torch.tensor([beat_index.get(line_beat.get(atom, ""), -1)
                               for atom in atoms])

    test = test_chains(args.eval_chains, args.steps)
    base_records, base_raw = audit(model, hf, test, atoms, atom_index,
                                   beat_index, atom_beats, tag="baseline")
    base_deploy = deploy_audit(model, hf, atoms, test)
    print(f"baseline raw={base_raw['atom']:.3f}/{base_raw['behaviour']:.3f}/"
          f"{base_raw['thought']:.3f} deploy={base_deploy}")

    report = {"baseline": {"raw": base_raw, "deploy": base_deploy}, "sizes": []}
    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        model.load_state_dict(base_state)
        attacker = Attacker(args.seed + size)
        chains = attacker.generate(size, args.steps, seed=args.seed + size)
        replay = [label_session(s) for s in
                  ({"texts": [row[0] for row in session], "trust": 35,
                    "player_name": ""} for session in play(SEEDS, args.replay, 6,
                                                           seed=300))]
        rows = [label_session(s) for s in chains] + replay
        fine_tune(model, rows, hf, atoms, beats, args.epochs, args.lr)
        records, raw = audit(model, hf, test, atoms, atom_index, beat_index,
                             atom_beats, tag=f"size{size}")
        deploy = deploy_audit(model, hf, atoms, test)
        delta = {k: round(raw[k] - base_raw[k], 3)
                 for k in ("atom", "behaviour", "thought")}
        print(f"size {size}: raw={raw['atom']:.3f}/{raw['behaviour']:.3f}/"
              f"{raw['thought']:.3f} deploy={deploy} delta={delta}")
        path = save_variant(model, atoms, beats, config, encoder, size)
        report["sizes"].append({"size": size, "raw": raw, "deploy": deploy,
                                "delta": delta, "train_turns": len(rows) * args.steps,
                                "checkpoint": os.path.basename(path)})

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"saved {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
