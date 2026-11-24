"""Hand-written training loop with adversarial self-play and distillation.

    python -m expertnet.train --rounds 3 --epochs 6
    python -m expertnet.train --rounds 2 --epochs 4 --distill

Round 1 trains on the hand-labelled seed corpus plus attacker variants. Each later
round asks the attacker for the variants the student finds hardest, adds them with
their true labels, and retrains. Evaluation always uses the untouched HELDOUT set.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from .agents import (Attacker, EnsembleTeacher, RuleTeacher, build_sessions,
                     eval_set, labelled, session_tensors, train_pool)
from .model import ExpertNet, encode, save
from .schema import HELDOUT, INTENTS, SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))


def _batches(sessions, batch: int):
    xs, yi, yr = session_tensors(sessions)
    for i in range(0, len(xs), batch):
        yield xs[i:i + batch], yi[i:i + batch], yr[i:i + batch]


def train_epochs(model, sessions, epochs: int, batch: int, lr: float = 1e-3) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = {}
    prepared = list(_batches(sessions, batch))
    for _ in range(epochs):
        for xs, yi, yr in prepared:
            intent_logits, risk_logits, gates, _ = model(xs)
            loss_i = F.cross_entropy(intent_logits.reshape(-1, len(INTENTS)),
                                     yi.reshape(-1), label_smoothing=0.05)
            loss_r = F.cross_entropy(risk_logits.reshape(-1, 2), yr.reshape(-1))
            balance = model.experts * (gates.mean(dim=(0, 1)) ** 2).sum()
            loss = loss_i + 0.2 * loss_r + 0.1 * balance
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = {"loss": float(loss.detach()), "intent": float(loss_i.detach()),
                    "risk": float(loss_r.detach()), "balance": float(balance.detach())}
    return last


def evaluate(model, data) -> dict:
    model.eval()
    xs = torch.stack([encode(text, {"trust": 35}, None) for text, _, _ in data])
    yi = torch.tensor([i for _, i, _ in data])
    yr = torch.tensor([r for _, _, r in data])
    with torch.no_grad():
        intent_logits, risk_logits, gates, _ = model(xs.unsqueeze(1), None)
    pred = intent_logits[:, 0].argmax(-1)
    risk_pred = risk_logits[:, 0].argmax(-1)
    per_intent = {}
    for (text, idx, _), p in zip(data, pred.tolist()):
        hit, total = per_intent.get(INTENTS[idx], (0, 0))
        per_intent[INTENTS[idx]] = (hit + int(p == idx), total + 1)
    usage = gates.reshape(-1, model.experts).argmax(-1).bincount(
        minlength=model.experts).tolist()
    return {
        "intent_acc": float((pred == yi).float().mean()),
        "risk_acc": float((risk_pred == yr).float().mean()),
        "gate_usage": usage,
        "per_intent": {k: round(v[0] / v[1], 2) for k, v in sorted(per_intent.items())},
        "misses": [text for (text, idx, _), p in zip(data, pred.tolist()) if p != idx],
    }


def regex_baseline(data) -> float:
    teacher = RuleTeacher()
    hits = 0
    for text, idx, _ in data:
        probs, _ = teacher.probs(text)
        hits += int(int(probs.argmax()) == idx)
    return hits / max(len(data), 1)


def distill(model, pool, teacher: EnsembleTeacher, epochs: int = 1, batch: int = 32) -> float:
    """Soft-label pass. Keeps hard labels for the CE part and trusts only confident
    teacher answers, so a weak teacher cannot overwrite what the student learned."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    last = 0.0
    for _ in range(epochs):
        for i in range(0, len(pool), batch):
            chunk = pool[i:i + batch]
            xs = torch.stack([encode(text, {"trust": 35}, None) for text, _, _ in chunk])
            labels = torch.tensor([idx for _, idx, _ in chunk])
            targets = torch.stack([teacher.probs(text) for text, _, _ in chunk])
            targets = targets / targets.sum(dim=-1, keepdim=True)
            logits = model(xs.unsqueeze(1))[0][:, 0]
            ce = F.cross_entropy(logits, labels, label_smoothing=0.05)
            per_item = F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="none").sum(-1)
            confident = targets.max(dim=-1).values >= 0.6
            # Only learn the teacher where it agrees with the curated label; a weak
            # teacher must not overwrite the hand-labelled truth.
            agree = confident & (targets.argmax(dim=-1) == labels)
            kl = per_item[agree].mean() if bool(agree.any()) else logits.sum() * 0.0
            loss = ce + 0.3 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
    return last


def predict_idx(model, text: str) -> int:
    model.eval()
    x = encode(text, {"trust": 35}, None).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        logits = model(x)[0][0, 0]
    return int(logits.argmax())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=160)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--distill", action="store_true",
                        help="add a soft-label pass from the rule teacher")
    parser.add_argument("--scripted", action="store_true",
                        help="sessions follow a social phase path, so context matters")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = ExpertNet(experts=args.experts)
    attacker = Attacker(args.seed)
    teacher = EnsembleTeacher([RuleTeacher()])
    data = eval_set()
    baseline = regex_baseline(data)
    pool = train_pool(copies=8, seed=args.seed)
    history = []

    for round_no in range(1, args.rounds + 1):
        sessions = build_sessions(pool, steps=args.steps, count=args.sessions,
                                  seed=args.seed + round_no, scripted=args.scripted)
        stats = train_epochs(model, sessions, args.epochs, args.batch)
        if args.distill:
            stats["distill"] = distill(model, pool, teacher)

        held = evaluate(model, data)
        seeds_eval = evaluate(model, list(labelled(SEEDS)))
        print(f"round {round_no}: heldout={held['intent_acc']:.3f} "
              f"seed={seeds_eval['intent_acc']:.3f} risk={held['risk_acc']:.3f} "
              f"loss={stats['loss']:.3f} gate={held['gate_usage']}")
        history.append({"round": round_no, **{k: v for k, v in stats.items()},
                        "heldout": held["intent_acc"], "seed": seeds_eval["intent_acc"],
                        "risk": held["risk_acc"], "gate": held["gate_usage"]})

        # The attacker may only touch the training seeds; HELDOUT stays untouched.
        seed_misses = [(text, idx, risk) for text, idx, risk in labelled(SEEDS)
                       if predict_idx(model, text) != idx]
        if seed_misses:
            hard = attacker.hardest(model, seed_misses, k=2)
            pool.extend(hard)
            print(f"  attacker added {len(hard)} hard variants")

    print(f"regex baseline on heldout: {baseline:.3f}")
    print(f"student on heldout:        {history[-1]['heldout']:.3f}")
    save(model, os.path.join(HERE, "checkpoint.pt"))
    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"baseline": baseline, "history": history,
                   "heldout": evaluate(model, data)["per_intent"]}, fh,
                  ensure_ascii=False, indent=1)
    print("saved expertnet/checkpoint.pt and expertnet/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
