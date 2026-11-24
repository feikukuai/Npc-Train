"""Hand-written training loop with adversarial self-play and distillation.

    python -m expertnet.train --arch v1  --rounds 3 --epochs 6
    python -m expertnet.train --arch v2h --rounds 2 --epochs 4 --distill
    python -m expertnet.train --arch v2t --rounds 2 --epochs 4

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
                     eval_set, labelled, train_pool)
from .arch import make_arch, save_arch
from .schema import HELDOUT, INTENTS, SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))


def _slices(batch: dict, size: int):
    for i in range(0, batch["yi"].size(0), size):
        yield {key: value[i:i + size] for key, value in batch.items()}


def train_epochs(arch, model, batch: dict, epochs: int, size: int,
                 lr: float = 1e-3) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = {}
    prepared = list(_slices(batch, size))
    for _ in range(epochs):
        for chunk in prepared:
            intent_logits, risk_logits, gates = arch.forward_batch(model, chunk)
            loss_i = F.cross_entropy(intent_logits.reshape(-1, len(INTENTS)),
                                     chunk["yi"].reshape(-1), label_smoothing=0.05)
            loss_r = F.cross_entropy(risk_logits.reshape(-1, 2), chunk["yr"].reshape(-1))
            balance = model.experts * (gates.mean(dim=(0, 1)) ** 2).sum()
            loss = loss_i + 0.2 * loss_r + 0.1 * balance
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = {"loss": float(loss.detach()), "intent": float(loss_i.detach()),
                    "risk": float(loss_r.detach()), "balance": float(balance.detach())}
    return last


def evaluate(arch, model, data) -> dict:
    model.eval()
    texts = [text for text, _, _ in data]
    with torch.no_grad():
        intent_logits, risk_logits, gates = arch.forward_texts(
            model, arch.text_tensors(texts))
    yi = torch.tensor([i for _, i, _ in data])
    yr = torch.tensor([r for _, _, r in data])
    pred = intent_logits.argmax(-1)
    risk_pred = risk_logits.argmax(-1)
    per_intent = {}
    for (text, idx, _), p in zip(data, pred.tolist()):
        hit, total = per_intent.get(INTENTS[idx], (0, 0))
        per_intent[INTENTS[idx]] = (hit + int(p == idx), total + 1)
    usage = gates.argmax(-1).bincount(minlength=model.experts).tolist()
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


def distill(arch, model, pool, teacher: EnsembleTeacher, epochs: int = 1,
            batch: int = 32) -> float:
    """Soft-label pass. Keeps hard labels for the CE part and trusts only confident
    teacher answers, so a weak teacher cannot overwrite what the student learned."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    last = 0.0
    for _ in range(epochs):
        for i in range(0, len(pool), batch):
            chunk = pool[i:i + batch]
            texts = [text for text, _, _ in chunk]
            labels = torch.tensor([idx for _, idx, _ in chunk])
            targets = torch.stack([teacher.probs(text) for text, _, _ in chunk])
            targets = targets / targets.sum(dim=-1, keepdim=True)
            logits = arch.forward_texts(model, arch.text_tensors(texts))[0]
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


def predict_idx(arch, model, text: str) -> int:
    model.eval()
    with torch.no_grad():
        logits = arch.forward_texts(model, arch.text_tensors([text]))[0][0]
    return int(logits.argmax())


def run(args, hf=None) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    arch = make_arch(args.arch, experts=args.experts, hidden=args.hidden, hf=hf)
    model = arch.model
    attacker = Attacker(args.seed)
    teacher = EnsembleTeacher([RuleTeacher()])
    data = eval_set()
    baseline = regex_baseline(data)
    pool = train_pool(copies=args.copies, seed=args.seed, frac=args.train_frac)
    history = []

    for round_no in range(1, args.rounds + 1):
        sessions = build_sessions(pool, steps=args.steps, count=args.sessions,
                                  seed=args.seed + round_no, scripted=args.scripted)
        stats = train_epochs(arch, model, arch.batches(sessions), args.epochs, args.batch)
        if args.distill:
            stats["distill"] = distill(arch, model, pool, teacher)

        held = evaluate(arch, model, data)
        seeds_eval = evaluate(arch, model, list(labelled(SEEDS)))
        if not getattr(args, "quiet", False):
            print(f"round {round_no}: heldout={held['intent_acc']:.3f} "
                  f"seed={seeds_eval['intent_acc']:.3f} risk={held['risk_acc']:.3f} "
                  f"loss={stats['loss']:.3f} gate={held['gate_usage']}")
        history.append({"round": round_no, **stats,
                        "heldout": held["intent_acc"], "seed": seeds_eval["intent_acc"],
                        "risk": held["risk_acc"], "gate": held["gate_usage"]})

        # The attacker may only touch the training seeds; HELDOUT stays untouched.
        seed_misses = [(text, idx, risk) for text, idx, risk in labelled(SEEDS)
                       if predict_idx(arch, model, text) != idx]
        if seed_misses:
            hard = attacker.hardest(model, seed_misses, k=2, arch=arch)
            pool.extend(hard)
            if not getattr(args, "quiet", False):
                print(f"  attacker added {len(hard)} hard variants")

    held = evaluate(arch, model, data)
    if not getattr(args, "quiet", False):
        print(f"regex baseline on heldout: {baseline:.3f}")
        print(f"student on heldout:        {held['intent_acc']:.3f}")
    return {"arch": args.arch, "frac": args.train_frac, "baseline": baseline,
            "heldout": held["intent_acc"], "per_intent": held["per_intent"],
            "gate_usage": held["gate_usage"], "history": history,
            "model": model, "arch_obj": arch}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["v1", "v2h", "v2t"], default="v1")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=160)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--copies", type=int, default=8,
                        help="attacker variants kept per hand-labelled seed")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="share of the seed corpus used (data-efficiency ablation)")
    parser.add_argument("--distill", action="store_true",
                        help="add a soft-label pass from the rule teacher")
    parser.add_argument("--scripted", action="store_true",
                        help="sessions follow a social phase path, so context matters")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=os.path.join(HERE, "checkpoint.pt"))
    parser.add_argument("--metrics", default=os.path.join(HERE, "metrics.json"))
    args = parser.parse_args()

    result = run(args)
    save_arch(result["arch_obj"], args.out)
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump({"arch": result["arch"], "train_frac": result["frac"],
                   "baseline": result["baseline"], "history": result["history"],
                   "heldout": result["per_intent"], "gate_usage": result["gate_usage"]},
                  fh, ensure_ascii=False, indent=1)
    print(f"saved {args.out} and {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
