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
from ..sentence import HFSentenceEncoder
from .corpus import HELDOUT, SEEDS
from .persona import extract_name, format_line, new_state, teacher_turn, zh_state_vector

HERE = os.path.dirname(os.path.abspath(__file__))
TRUSTS = (10, 25, 35, 55, 70)

# Context-dependent trajectories: the reply depends on what happened just before.
PATHS = [
    ["greeting", "smalltalk_how", "personal", "lore", "farewell"],
    ["insult", "farewell"],
    ["insult", "farewell", "greeting"],
    ["insult", "greeting"],
    ["gift", "thanks", "smalltalk_bored", "farewell"],
    ["secret", "secret", "farewell"],
    ["name", "name", "name", "farewell"],
    ["name_unknown", "name", "name", "farewell"],
    ["name", "personal", "name_unknown", "name", "farewell"],
    ["threat", "apology", "farewell"],
    ["out_of_world", "out_of_world", "farewell"],
]


def play(texts_by_intent: dict[str, list[str]], count: int, steps: int,
         seed: int, scripted: bool = True) -> list[list[tuple]]:
    rng = random.Random(seed)
    intents = list(texts_by_intent)
    out = []
    for _ in range(count):
        state = new_state(rng.choice(TRUSTS))
        path = rng.choice(PATHS) if scripted and rng.random() < 0.8 else None
        rows = []
        for step in range(steps):
            if path and step < len(path) and rng.random() < 0.85:
                intent = path[step]
            else:
                intent = rng.choice(intents)
            base = "name" if intent.startswith("name") else intent
            pool = texts_by_intent.get(base) or texts_by_intent["fallback"]
            if intent in ("name", "name_set", "name_unknown"):
                sets = [t for t in pool if extract_name(t)]
                gets = [t for t in pool if not extract_name(t)]
                if intent == "name_set":
                    pool = sets or pool
                elif intent == "name_unknown":
                    pool = gets or pool
                else:
                    pool = gets if state.get("player_name") else (sets or pool)
            text = rng.choice(pool)
            prev = state.get("last_beat")
            line, beat = teacher_turn(text, state)
            snapshot = {"trust": state["trust"], "turn": state["turn"],
                        "guild": False, "revealed": list(state["revealed"]),
                        "last_beat": prev, "mood": state["mood"],
                        "player_name": state["player_name"]}
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

    data = tensors(train, hf, state_fn=zh_state_vector)
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

    metrics = evaluate(model, hf, eval_set, atom_index, beat_index, atom_beats,
                       state_fn=zh_state_vector)
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


def run_script(model, hf, atoms, texts: list[str]) -> tuple[int, int]:
    state = new_state()
    model.eval()
    h = torch.zeros(1, model.hidden)
    hits = 0
    for text in texts:
        prev = state.get("last_beat")
        truth, _ = teacher_turn(text, state)
        atom_state = {**state, "last_beat": prev}
        vec = hf.embed([text]).unsqueeze(0)
        st = zh_state_vector(atom_state).reshape(1, 1, -1)
        with torch.no_grad():
            logits, _, h = model(vec, st, h)
        probs = F.softmax(logits[0, 0], -1)
        line = atoms[int(probs.argmax())]
        hit = line == truth
        hits += int(hit)
        print(f"玩家: {text}")
        print(f"  句库: {format_line(line, state)}  ({float(probs.max()):.2f})"
              f"{'  [命中]' if hit else ''}")
        print(f"  规则: {format_line(truth, state)}")
    return hits, len(texts)


CASES = [
    ["你好。", "傻瓜。", "再见。"],
    ["傻瓜，我走了。", "你好。"],
    ["我叫小明。", "你知道我叫什么吗？"],
    ["核弹是什么？", "我问你核弹是什么？"],
    ["这个给你。", "谢谢。", "我走了。"],
    ["告诉我一个秘密。", "告诉我一个秘密。", "再见。"],
]


def demo(model, hf, atoms) -> None:
    print("\n--- 演示：被打断的告别 ---")
    run_script(model, hf, atoms, ["你好。", "傻瓜。", "再见。"])
    print("\n--- 上下文脚本命中 ---")
    total_hit = total = 0
    for case in CASES:
        hits, n = run_script(model, hf, atoms, case)
        total_hit += hits
        total += n
        print(f"  -> {hits}/{n}")
    print(f"上下文脚本合计: {total_hit}/{total}")


if __name__ == "__main__":
    raise SystemExit(main())
