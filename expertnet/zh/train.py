"""训练一个中文的整句选择器（阿禾，五岁半）。

    python -m expertnet.zh.train --sessions 800 --epochs 30 --demo

数据来自 persona 里的规则老师：记录（玩家句、状态、回复句、beat），学生只学
“从句库里挑哪一句”，不学逐字生成。测试用 corpus.HELDOUT 的换说法。
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F

from ..atoms import AtomNet, evaluate, tensors
from ..model import state_vector
from ..sentence import HFSentenceEncoder
from .corpus import HELDOUT, SEEDS
from .persona import new_state, teacher_turn

HERE = os.path.dirname(os.path.abspath(__file__))
TRUSTS = (10, 25, 35, 55, 70)


def play(texts_by_intent: dict[str, list[str]], count: int, steps: int,
         seed: int) -> list[list[tuple]]:
    rng = random.Random(seed)
    intents = list(texts_by_intent)
    out = []
    for _ in range(count):
        state = new_state(rng.choice(TRUSTS))
        rows = []
        for _ in range(steps):
            intent = rng.choice(intents)
            text = rng.choice(texts_by_intent[intent])
            line, beat = teacher_turn(text, state)
            snapshot = {"trust": state["trust"], "turn": state["turn"],
                        "guild": False, "revealed": list(state["revealed"])}
            rows.append((text, snapshot, line, beat))
        out.append(rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=800)
    parser.add_argument("--eval-sessions", type=int, default=150)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", default=os.path.join(HERE, "atoms_zh.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    hf = HFSentenceEncoder("BAAI/bge-small-zh-v1.5", pooling="cls")

    train = play(SEEDS, args.sessions, args.steps, args.seed)
    eval_set = play(HELDOUT, args.eval_sessions, args.steps, args.seed + 1)
    atom_counts = Counter(row[2] for session in train for row in session)
    atoms = [line for line, _ in atom_counts.most_common()]
    atom_index = {line: i for i, line in enumerate(atoms)}
    beats = sorted({row[3] for session in train for row in session})
    beat_index = {beat: i for i, beat in enumerate(beats)}
    line_beat = {}
    for session in train:
        for row in session:
            line_beat.setdefault(row[2], row[3])
    atom_beats = torch.tensor([beat_index.get(line_beat[line], -1) for line in atoms])

    data = tensors(train, hf)
    targets = torch.tensor([[atom_index[row[2]] for row in session] for session in train])
    beat_targets = torch.tensor([[beat_index[row[3]] for row in session] for session in train])

    model = AtomNet(hf.dim, len(atoms), len(beats), hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(train))
        total = 0.0
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            atom_logits, beat_logits, _ = model(data["vecs"][idx], data["states"][idx])
            loss = (F.cross_entropy(atom_logits.reshape(-1, len(atoms)),
                                    targets[idx].reshape(-1))
                    + 0.3 * F.cross_entropy(beat_logits.reshape(-1, len(beats)),
                                            beat_targets[idx].reshape(-1)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:2d}  loss={total / len(train):.3f}")

    metrics = evaluate(model, hf, eval_set, atom_index, beat_index, atom_beats)
    print(f"句子原子:          {len(atoms)}")
    print(f"训练回合:          {len(train) * args.steps}")
    print(f"测试回合:          {len(eval_set) * args.steps}")
    print(f"beat 命中:         {metrics['beat']:.3f}")
    print(f"整句完全命中:      {metrics['atom']:.3f}")
    print(f"  句子类型正确:    {metrics['behaviour']:.3f}")
    print(f"  目标句覆盖:      {metrics['coverage']:.3f}")
    print(f"整句 top-3:        {metrics['top3']:.3f}")
    print(f"携带上下文:        {metrics['carried']:.3f}")
    print(f"每轮重置:          {metrics['reset']:.3f}")
    print(f"上下文增益:        {metrics['carried'] - metrics['reset']:+.3f}")

    torch.save({"config": {"text_dim": hf.dim, "n_atoms": len(atoms),
                           "n_beats": len(beats), "hidden": args.hidden},
                "atoms": atoms, "beats": beats, "state": model.state_dict(),
                "encoder": {"name": hf.name, "pooling": hf.pooling}}, args.out)
    with open(os.path.join(HERE, "atoms_zh_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=1)
    print(f"saved {args.out}")

    if args.demo:
        demo(model, hf, atoms)
    return 0


def demo(model, hf, atoms) -> None:
    state = new_state()
    model.eval()
    h = torch.zeros(1, model.hidden)
    print("\n--- 演示（模型选整句 vs 规则老师）---")
    for text in ["你好，阿禾。", "你在干嘛？", "草药园在哪里？",
                 "糖多少钱？", "告诉我一个秘密。", "我走啦。"]:
        vec = hf.embed([text]).unsqueeze(0)
        st = state_vector(state, None).reshape(1, 1, -1)
        with torch.no_grad():
            logits, _, h = model(vec, st, h)
        probs = F.softmax(logits[0, 0], -1)
        line, conf = atoms[int(probs.argmax())], float(probs.max())
        truth, _ = teacher_turn(text, state)
        print(f"玩家: {text}")
        print(f"  句库: {line}  ({conf:.2f})")
        print(f"  规则: {truth}")


if __name__ == "__main__":
    raise SystemExit(main())
