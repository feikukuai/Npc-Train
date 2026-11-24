"""Predict a whole sentence, not the next character.

A small character does not need an open-ended language model. It needs to pick, from
the sentences it already knows, the one that fits this turn. That turns generation
into classification over a finite sentence vocabulary ("sentence atoms"), which is
why it needs thousands of turns instead of millions of tokens.

    v_t = SentenceEncoder(player_text)      # frozen MiniLM, one vector per sentence
    h_t = GRU([v_t | state_t], h_{t-1})     # one 64-d memory vector, nothing longer
    P(line_t) = softmax(W h_t)              # one sentence atom out of N
    P(beat_t) = softmax(U h_t)              # auxiliary: what kind of reply it is

Training data is distilled from the rule engine: play scripted turns against the
existing NPC, record (player sentence, state, reply sentence, beat). The student
imitates the teacher's choices, and the evaluation uses held-out player paraphrases.

    python -m expertnet.atoms --sessions 300 --epochs 20
    python -m expertnet.atoms --demo
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import STATE_DIM, state_vector
from .schema import HELDOUT, SEEDS
from .sentence import HFSentenceEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
TRUSTS = (10, 25, 35, 55, 70)


def play_sessions(texts_by_intent: dict[str, list[str]], count: int, steps: int,
                  seed: int, profile: str = "xiaohong",
                  scripted: bool = False) -> list[list[tuple]]:
    """Run the rule NPC and record (player, state, reply, beat) per turn."""
    from bot.npc import engine

    rng = random.Random(seed)
    intents = list(texts_by_intent)
    out = []
    for _ in range(count):
        memory: dict = {}
        engine.activate(memory, profile, trust=rng.choice(TRUSTS))
        phase = rng.choice(["greeting", "smalltalk_how", "personal", "lore", "farewell"])
        rows = []
        for _ in range(steps):
            if scripted and rng.random() < 0.6:
                intent = phase
            else:
                intent = rng.choice(intents)
            text = rng.choice(texts_by_intent[intent])
            result = engine.turn(text, memory)
            state = memory["npc"]
            snapshot = {"trust": state["trust"], "turn": state["turn"],
                        "guild": bool(state.get("guild")),
                        "revealed": list(state.get("revealed") or [])}
            rows.append((text, snapshot, result["reply"], result["npc"]["beat"]))
            if scripted and rng.random() < 0.4:
                phase = rng.choice(intents)
        out.append(rows)
    return out


class AtomNet(nn.Module):
    """Tiny recurrent selector over a fixed sentence vocabulary."""

    def __init__(self, text_dim: int, n_atoms: int, n_beats: int,
                 hidden: int = 64):
        super().__init__()
        self.hidden, self.n_atoms = hidden, n_atoms
        inp = text_dim + STATE_DIM
        self.wz = nn.Linear(inp + hidden, hidden)
        self.wr = nn.Linear(inp + hidden, hidden)
        self.wn = nn.Linear(inp + hidden, hidden)
        # Read out from both the memory and the current sentence: the memory is short,
        # the sentence itself carries most of what this turn is about.
        self.atoms = nn.Linear(hidden + inp, n_atoms)
        self.beats = nn.Linear(hidden + inp, n_beats)

    def step(self, x: torch.Tensor, h: torch.Tensor):
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.wz(xh))
        r = torch.sigmoid(self.wr(xh))
        n = torch.tanh(self.wn(torch.cat([x, r * h], dim=-1)))
        h = (1 - z) * h + z * n
        out = torch.cat([h, x], dim=-1)
        return self.atoms(out), self.beats(out), h

    def forward(self, vecs: torch.Tensor, states: torch.Tensor, h=None):
        batch, steps, _ = vecs.shape
        if h is None:
            h = torch.zeros(batch, self.hidden)
        atoms, beats = [], []
        for t in range(steps):
            a, b, h = self.step(torch.cat([vecs[:, t], states[:, t]], -1), h)
            atoms.append(a)
            beats.append(b)
        return torch.stack(atoms, 1), torch.stack(beats, 1), h


def tensors(sessions, hf: HFSentenceEncoder, state_fn=None):
    rows = [row for session in sessions for row in session]
    encode_state = state_fn or state_vector
    return {"vecs": hf.embed([row[0] for row in rows]).view(len(sessions), len(sessions[0]), -1),
            "states": torch.stack([encode_state(row[1]) for row in rows])
                            .view(len(sessions), len(sessions[0]), -1)}


class AtomSession:
    """One player's 64-d memory plus the frozen sentence encoder."""

    def __init__(self, model: AtomNet, hf: HFSentenceEncoder, atoms: list[str],
                 state_fn=None):
        self.model = model.eval()
        self.hf = hf
        self.atoms = atoms
        self.state_fn = state_fn or state_vector
        self.reset()

    def reset(self) -> None:
        self.h = torch.zeros(1, self.model.hidden)

    def reply(self, text: str, state: dict | None = None) -> tuple[str, float]:
        vec = self.hf.embed([text]).unsqueeze(0)
        st = self.state_fn(state).reshape(1, 1, -1)
        with torch.no_grad():
            logits, _, self.h = self.model(vec, st, self.h)
        logits = logits[0, 0]
        if not (state or {}).get("player_name"):
            # A name template needs a name; without one it would render as "你叫你".
            blocked = torch.tensor(["{player_name}" in atom for atom in self.atoms])
            logits = logits.masked_fill(blocked, -1e9)
        probs = F.softmax(logits, dim=-1)
        line = self.atoms[int(probs.argmax())]
        if "{player_name}" in line:
            line = line.format(player_name=(state or {}).get("player_name") or "你")
        return line, float(probs.max())


def load_atoms(path: str, hf: HFSentenceEncoder, state_fn=None) -> AtomSession:
    ckpt = torch.load(path, map_location="cpu")
    model = AtomNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return AtomSession(model, hf, ckpt["atoms"], state_fn)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=300)
    parser.add_argument("--eval-sessions", type=int, default=120)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--profile", default="xiaohong")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="print a short conversation after training")
    parser.add_argument("--out", default=os.path.join(HERE, "atoms.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    hf = HFSentenceEncoder()

    train = play_sessions(SEEDS, args.sessions, args.steps, args.seed,
                          args.profile, args.scripted)
    eval_set = play_sessions(HELDOUT, args.eval_sessions, args.steps, args.seed + 1,
                             args.profile, args.scripted)
    atom_counts = Counter(row[2] for session in train for row in session)
    atoms = [line for line, _ in atom_counts.most_common()]
    atom_index = {line: i for i, line in enumerate(atoms)}
    beats = sorted({row[3] for session in train for row in session})
    beat_index = {beat: i for i, beat in enumerate(beats)}
    line_beat: dict[str, str] = {}
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
    print(f"sentence atoms:    {len(atoms)}")
    print(f"training turns:    {len(train) * args.steps}")
    print(f"eval turns:        {len(eval_set) * args.steps}")
    print(f"beat accuracy:     {metrics['beat']:.3f}")
    print(f"atom accuracy:     {metrics['atom']:.3f}")
    print(f"  right sentence type: {metrics['behaviour']:.3f}")
    print(f"  seen targets:    {metrics['coverage']:.3f} of eval turns")
    print(f"atom top-3:        {metrics['top3']:.3f}")
    print(f"context carried:   {metrics['carried']:.3f}")
    print(f"context reset:     {metrics['reset']:.3f}")
    print(f"context gain:      {metrics['carried'] - metrics['reset']:+.3f}")

    torch.save({"config": {"text_dim": hf.dim, "n_atoms": len(atoms),
                           "n_beats": len(beats), "hidden": args.hidden},
                "atoms": atoms, "beats": beats, "state": model.state_dict()}, args.out)
    with open(os.path.join(HERE, "atoms_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=1)
    print(f"saved {args.out}")

    if args.demo:
        demo(model, hf, atoms, args.profile)
    return 0


def evaluate(model, hf, sessions, atom_index, beat_index, atom_beats,
             state_fn=None) -> dict:
    """Carried vs reset memory; accuracy over held-out player paraphrases."""
    model.eval()
    rows = [row for session in sessions for row in session]
    data = tensors(sessions, hf, state_fn=state_fn)
    atom_ids = [atom_index.get(row[2], -1) for row in rows]
    beat_ids = [beat_index.get(row[3], -1) for row in rows]
    target = torch.tensor(atom_ids)
    beat_target = torch.tensor(beat_ids)
    steps = len(sessions[0])
    stats = {"atom": 0, "beat": 0, "behaviour": 0, "top3": 0, "coverage": 0,
             "carried": 0, "reset": 0}
    with torch.no_grad():
        atom_logits, beat_logits, _ = model(data["vecs"], data["states"])
    flat = atom_logits.reshape(-1, atom_logits.size(-1))
    pred = flat.argmax(-1)
    top3 = flat.topk(3, dim=-1).indices
    seen = target >= 0
    stats["coverage"] = float(seen.float().mean())
    stats["atom"] = float((pred[seen] == target[seen]).float().mean()) if seen.any() else 0.0
    stats["top3"] = float((top3[seen] == target[seen].unsqueeze(1)).any(-1).float().mean())
    valid_beats = beat_target >= 0
    stats["beat"] = float((beat_logits.reshape(-1, beat_logits.size(-1)).argmax(-1)[valid_beats]
                           == beat_target[valid_beats]).float().mean())
    chosen_beats = atom_beats[pred]
    both = (chosen_beats >= 0) & valid_beats
    stats["behaviour"] = float((chosen_beats[both] == beat_target[both]).float().mean())

    # Reset memory every turn: same inputs, no session context.
    per_turn = []
    with torch.no_grad():
        for t in range(steps):
            logits, _, _ = model(data["vecs"][:, t:t + 1], data["states"][:, t:t + 1])
            per_turn.append(logits[:, 0])
    reset_logits = torch.cat(per_turn, 1).reshape(-1, atom_logits.size(-1))
    reset_pred = reset_logits.argmax(-1)
    stats["reset"] = float((reset_pred[seen] == target[seen]).float().mean()) if seen.any() else 0.0
    stats["carried"] = stats["atom"]
    return stats


def demo(model, hf, atoms, profile: str = "xiaohong") -> None:
    from bot.npc import engine

    memory: dict = {}
    engine.activate(memory, profile, trust=35)
    model.eval()
    h = torch.zeros(1, model.hidden)
    print("\n--- demo (model picks a sentence; rule engine shown for comparison) ---")
    for text in ["Hello.", "How are you?", "Tell me about the bees.",
                 "What do you sell?", "Tell me a secret.", "Goodbye."]:
        state = memory["npc"]
        vec = hf.embed([text])
        st = state_vector(state, None).reshape(1, 1, -1)
        with torch.no_grad():
            logits, _, h = model(vec.unsqueeze(0), st, h)
        probs = F.softmax(logits[0, 0], -1)
        line, conf = atoms[int(probs.argmax())], float(probs.max())
        truth = engine.turn(text, memory)["reply"]
        print(f"player: {text}")
        print(f"  atom: {line}  ({conf:.2f})")
        print(f"  rule: {truth}")


if __name__ == "__main__":
    raise SystemExit(main())
