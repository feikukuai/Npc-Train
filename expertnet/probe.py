"""Emergence probes: does context help, and do experts specialise?

Two measurements on a trained checkpoint, both cheap and deterministic:

1. context gain   - same held-out sentences, carried h_t vs reset h_t every turn;
2. expert routing - mutual information between the gate's choice and the true intent,
   plus the top intents each expert handles.

Run:  python -m expertnet.probe [--checkpoint expertnet/checkpoint.pt]
"""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict

import torch

from .model import encode, load
from .schema import HELDOUT, INTENTS
from .train import eval_set

HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = ["greeting", "smalltalk_how", "smalltalk_weather", "personal",
          "lore", "secret", "farewell"]


def scripted_sequence():
    """Held-out sentences ordered by social phase, never used for training."""
    return [(text, INTENTS.index(intent), 0)
            for intent in PHASES for text in HELDOUT.get(intent, [])]


def session_accuracy(model, data, carry: bool, use_last: bool = True) -> float:
    model.eval()
    h = torch.zeros(1, model.wz.out_features)
    last = None
    hits = 0
    for turn, (text, idx, _) in enumerate(data):
        # Reset mode must not see the current or previous label either, otherwise the
        # ablation leaks the answer through the state features.
        hint = last if (carry and use_last) else None
        x = encode(text, {"trust": 35, "turn": turn + 1}, hint).unsqueeze(0)
        with torch.no_grad():
            logits, _, h, _ = model.step(x, h if carry else torch.zeros_like(h))
        pred = int(logits.argmax(-1))
        hits += int(pred == idx)
        if carry and use_last:
            last = INTENTS[pred]
    return hits / max(len(data), 1)


def routing(model, data):
    model.eval()
    xs = torch.stack([encode(text, {"trust": 35}, None) for text, _, _ in data])
    with torch.no_grad():
        _, _, gates, _ = model(xs.unsqueeze(1), None)
    choices = gates[:, 0].argmax(-1).tolist()
    pairs = list(zip(choices, [INTENTS[idx] for _, idx, _ in data]))
    counts = Counter(pairs)
    experts = Counter(x for x, _ in pairs)
    intents = Counter(y for _, y in pairs)
    n = len(pairs)
    mi = 0.0
    for (expert, intent), c in counts.items():
        pxy = c / n
        px = experts[expert] / n
        py = intents[intent] / n
        mi += pxy * math.log(pxy / (px * py))
    top = defaultdict(list)
    for (expert, intent), c in counts.most_common():
        if len(top[expert]) < 3:
            top[expert].append(f"{intent}:{c}")
    return mi, dict(sorted(experts.items())), dict(sorted(top.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(HERE, "checkpoint.pt"))
    parser.add_argument("--steps", type=int, default=0,
                        help="limit the held-out sequence length (0 = all)")
    parser.add_argument("--sequence", choices=["scripted", "raw"], default="scripted")
    parser.add_argument("--drop-last", action="store_true",
                        help="hide last_intent so only the recurrent state can carry context")
    args = parser.parse_args()

    model = load(args.checkpoint)
    data = scripted_sequence() if args.sequence == "scripted" else eval_set()
    if args.steps:
        data = data[:args.steps]

    carried = session_accuracy(model, data, carry=True, use_last=not args.drop_last)
    reset = session_accuracy(model, data, carry=False, use_last=not args.drop_last)
    mi, usage, top = routing(model, data)

    print(f"sentences:        {len(data)}")
    print(f"context carried:  {carried:.3f}")
    print(f"context reset:    {reset:.3f}")
    print(f"context gain:     {carried - reset:+.3f}")
    print(f"expert-intent MI: {mi:.3f} nats")
    print(f"expert usage:     {usage}")
    for expert, intents in top.items():
        print(f"  expert {expert}: {', '.join(intents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
