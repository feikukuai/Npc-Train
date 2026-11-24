"""Three agents for long-dialogue adversarial training.

Attacker  builds 12-16 turn conversations and hard paraphrases;
Student   answers with its 64-d memory;
Auditor   replays the rule teacher, records every turn, and mines the misses.

Nothing here changes the output vocabulary: the student may only choose among the
sentences the teacher already produced, so a failed turn is a routing/context
failure rather than an unknown sentence.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from .corpus import SEEDS
from .persona import extract_name, new_state, teacher_turn, zh_state_vector
from .thought import THOUGHTS, thought_of

ARCS = [
    ["insult", "farewell"],
    ["insult", "farewell", "greeting"],
    ["insult", "greeting"],
    ["gift", "thanks", "insult", "farewell"],
    ["name", "name", "name"],
    ["name", "personal", "name", "memory"],
    ["secret", "secret", "secret"],
    ["threat", "taunt", "apology", "apology_meta", "memory"],
    ["threat", "apology", "farewell"],
    ["out_of_world", "out_of_world", "personal"],
    ["greeting", "smalltalk_how", "personal", "lore", "farewell"],
    ["greeting", "personal", "memory", "farewell"],
    ["smalltalk_bored", "smalltalk_joke", "smalltalk_weather", "farewell"],
    ["trade", "gift", "thanks", "farewell"],
]
PREFIXES = ["阿禾，", "喂，", "那个……", "我跟你说，"]
SUFFIXES = ["啊。", "呀。", "……"]


def label_session(session: dict) -> list[tuple]:
    """Turn a text list into training rows with the teacher's reply and state."""
    state = new_state(session.get("trust", 35))
    if session.get("player_name"):
        state["player_name"] = session["player_name"]
    rows = []
    for text in session["texts"]:
        previous = state.get("last_beat")
        line, beat = teacher_turn(text, state)
        snapshot = {"trust": state["trust"], "turn": state["turn"],
                    "guild": False, "revealed": list(state["revealed"]),
                    "last_beat": previous, "mood": state["mood"],
                    "player_name": state["player_name"],
                    "last_player": state.get("last_player", "")}
        rows.append((text, snapshot, line, beat))
    return rows


class Attacker:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def _line(self, intent: str) -> str:
        if intent in ("name", "name_set", "name_unknown"):
            pool = SEEDS["name"]
            sets = [t for t in pool if extract_name(t)]
            gets = [t for t in pool if not extract_name(t)]
            pool = sets if intent == "name_set" else gets
        else:
            pool = SEEDS.get(intent) or SEEDS["fallback"]
        text = self.rng.choice(pool)
        if self.rng.random() < 0.35:
            prefix = self.rng.choice(PREFIXES)
            if not text.startswith(prefix):
                text = prefix + text
        if self.rng.random() < 0.2:
            text = text.rstrip("。！？") + self.rng.choice(SUFFIXES)
        return text

    def generate(self, count: int = 80, steps: int = 14, seed: int | None = None,
                 hard_texts: list[str] | None = None) -> list[dict]:
        rng = random.Random(seed) if seed is not None else self.rng
        sessions = []
        for _ in range(count):
            trust = rng.choice([10, 25, 35, 55, 70])
            has_name = False
            texts: list[str] = []
            while len(texts) < steps:
                arc = rng.choice(ARCS)
                for intent in arc:
                    if intent == "name":
                        intent = "name_unknown" if has_name else "name_set"
                    text = self._line(intent)
                    if intent == "name_set":
                        has_name = bool(extract_name(text)) or has_name
                    if intent == "farewell" and rng.random() < 0.3:
                        text = "傻瓜，" + text
                    texts.append(text)
                    if len(texts) >= steps:
                        break
            if hard_texts and rng.random() < 0.4:
                texts[rng.randrange(len(texts))] = rng.choice(hard_texts)
            sessions.append({"texts": texts, "trust": trust, "player_name": ""})
        return sessions

    def mutate(self, text: str) -> str:
        r = self.rng.random()
        if r < 0.4:
            return self.rng.choice(PREFIXES) + text
        if r < 0.7:
            return text.rstrip("。！？") + self.rng.choice(SUFFIXES)
        return text

    def harden(self, sessions: list[dict], records: list[dict]) -> list[dict]:
        """Keep the whole prefix before a miss, so name/mood state is preserved."""
        misses = defaultdict(list)
        for record in records:
            if not record["hit"]:
                misses[record["session"]].append(record)
        out = []
        for index, failed in misses.items():
            base = sessions[index]
            for record in failed[:2]:
                turn = record["turn"]
                prefix = base["texts"][:turn + 1]
                out.append({"texts": prefix, "trust": base["trust"],
                            "player_name": base["player_name"]})
                mutated = self.mutate(record["text"])
                if mutated != record["text"]:
                    out.append({"texts": prefix[:-1] + [mutated],
                                "trust": base["trust"],
                                "player_name": base["player_name"]})
        return out


def _name_mask(atoms: list[str]) -> torch.Tensor:
    return torch.tensor(["{player_name}" in atom for atom in atoms])


@torch.no_grad()
def audit(model, hf, sessions, atoms, atom_index, beat_index, atom_beats,
          tag: str = "") -> tuple[list[dict], dict]:
    model.eval()
    blocked = _name_mask(atoms)
    records = []
    for sid, session in enumerate(sessions):
        state = new_state(session.get("trust", 35))
        if session.get("player_name"):
            state["player_name"] = session["player_name"]
        h = torch.zeros(1, model.hidden)
        for turn, text in enumerate(session["texts"]):
            previous = state.get("last_beat")
            target, beat = teacher_turn(text, state)
            atom_state = {**state, "last_beat": previous}
            vec = hf.embed([text]).unsqueeze(0)
            st = zh_state_vector(atom_state).reshape(1, 1, -1)
            logits, _, h = model(vec, st, h)
            logits = logits[0, 0]
            if not state.get("player_name"):
                logits = logits.masked_fill(blocked, -1e9)
            probs = F.softmax(logits, dim=-1)
            pred_idx = int(probs.argmax())
            pred = atoms[pred_idx]
            hit = pred == target
            behaviour = int(atom_beats[pred_idx]) == beat_index.get(beat, -1)
            thought_target = thought_of(previous, state["player_name"], beat)
            thought_pred = (THOUGHTS[int(model.last_thoughts[0, 0].argmax())]
                            if getattr(model, "last_thoughts", None) is not None else "-")
            if "名字" in text or "我叫什么" in text:
                category = "name"
            elif previous in ("insult", "threat") and beat in (
                    "farewell", "farewell_hurt", "greeting", "greeting_hurt"):
                category = "mood"
            elif beat in ("secret", "secret_denied"):
                category = "secret"
            elif turn >= 8:
                category = "long"
            else:
                category = "base"
            records.append({
                "tag": tag, "session": sid, "turn": turn, "text": text,
                "trust": state["trust"], "category": category,
                "target": target, "pred": pred, "hit": bool(hit),
                "behaviour": bool(behaviour), "beat": beat,
                "thought_target": thought_target, "thought": thought_pred,
                "thought_hit": bool(thought_target == thought_pred),
                "confidence": round(float(probs.max()), 3),
            })
    return records, summarize(records)


def summarize(records: list[dict]) -> dict:
    def agg(rows):
        if not rows:
            return {"n": 0, "atom": 0.0, "behaviour": 0.0, "thought": 0.0}
        return {"n": len(rows),
                "atom": round(sum(r["hit"] for r in rows) / len(rows), 3),
                "behaviour": round(sum(r["behaviour"] for r in rows) / len(rows), 3),
                "thought": round(sum(r["thought_hit"] for r in rows) / len(rows), 3)}

    by_category = {}
    for category in sorted({r["category"] for r in records}):
        by_category[category] = agg([r for r in records if r["category"] == category])
    by_beat = {}
    for beat, _ in Counter(r["beat"] for r in records).most_common():
        by_beat[beat] = agg([r for r in records if r["beat"] == beat])
    return {**agg(records), "by_category": by_category, "by_beat": by_beat}
