"""Agents for self-play: a rule teacher, an ensemble, an attacker, a referee.

The teacher labels what it can (distillation). The attacker perturbs utterances and
keeps the ones the student finds hardest. The referee derives the risk label, so
the same loop works with several teachers and several attackers.
"""
from __future__ import annotations

import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from .model import encode
from .schema import HELDOUT, INTENTS, SEEDS, UNSAFE


class RuleTeacher:
    name = "rule"

    def __init__(self, profile: str = "xiaohong"):
        from bot.npc import reader
        from bot.npc.profiles import PROFILES

        self._reader = reader
        self._profile = PROFILES[profile]

    def probs(self, text: str) -> tuple[torch.Tensor, float]:
        reading = self._reader.read(text, {}, self._profile)
        intent = reading["intent"]
        if intent not in INTENTS:
            intent = "out_of_world"
        confident = intent not in ("fallback", "out_of_world")
        conf = 0.9 if confident else 0.35
        probs = torch.full((len(INTENTS),), (1.0 - conf) / (len(INTENTS) - 1))
        probs[INTENTS.index(intent)] = conf
        return probs, conf


class EnsembleTeacher:
    """Average several teachers, weighted by their own confidence."""

    def __init__(self, teachers):
        self.teachers = teachers

    def probs(self, text: str) -> torch.Tensor:
        total = torch.zeros(len(INTENTS))
        for teacher in self.teachers:
            probs, conf = teacher.probs(text)
            total += probs * conf
        return total / total.sum()


SYNONYMS = {
    "hello": "greetings", "hi": "hello", "buy": "purchase", "sell": "vend",
    "price": "cost", "secret": "hidden thing", "kill": "murder", "hurt": "harm",
    "sad": "unhappy", "happy": "glad", "big": "large", "small": "little",
    "look": "seem", "want": "wish", "help": "aid", "work": "labour",
    "money": "coin", "old": "aged", "friend": "companion", "story": "tale",
}
PREFIXES = ["Honestly, ", "Please, ", "Tell me, ", "I ask you, ", "So, "]
SUFFIXES = [" please", " if you would", " right now", " for once"]


class Attacker:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def variants(self, text: str) -> list[str]:
        out = []
        words = text.split()
        for i, word in enumerate(words):
            key = word.lower().strip(".,!?")
            if key in SYNONYMS:
                swapped = list(words)
                swapped[i] = SYNONYMS[key]
                out.append(" ".join(swapped))
        if len(words) > 3:
            out.append(" ".join(words[:-1]))
            out.append(" ".join(words[1:]))
        longest = max(words, key=len) if words else ""
        if len(longest) > 4:
            pos = self.rng.randrange(1, len(longest) - 1)
            typo = longest[:pos] + longest[pos + 1] + longest[pos] + longest[pos + 2:]
            out.append(text.replace(longest, typo, 1))
        out.append(self.rng.choice(PREFIXES) + text)
        out.append(text + self.rng.choice(SUFFIXES))
        return [c for c in out if c and c != text]

    def hardest(self, model, examples, k: int = 2, state=None,
                arch=None) -> list[tuple[str, int, int]]:
        """examples: [(text, intent_idx, risk_idx)] -> hardest variants."""
        picked = []
        for text, intent_idx, risk_idx in examples:
            candidates = self.variants(text)
            if not candidates:
                continue
            with torch.no_grad():
                if arch is None:
                    xs = torch.stack([encode(c, state, None) for c in candidates])
                    scores = model(xs.unsqueeze(1), None)[0][:, 0, :]
                else:
                    tensors = arch.text_tensors(candidates, [state] * len(candidates))
                    scores = arch.forward_texts(model, tensors)[0]
            losses = F.cross_entropy(
                scores, torch.full((len(candidates),), intent_idx, dtype=torch.long),
                reduction="none")
            order = torch.argsort(losses, descending=True)[:k]
            for idx in order.tolist():
                picked.append((candidates[idx], intent_idx, risk_idx))
        return picked


def risk_of(intent: str) -> int:
    return 1 if intent in UNSAFE else 0


def labelled(texts_by_intent: dict[str, list[str]]):
    for intent, texts in texts_by_intent.items():
        idx = INTENTS.index(intent)
        for text in texts:
            yield text, idx, risk_of(intent)


SCRIPT = ["greeting", "smalltalk_how", "smalltalk_weather", "personal",
          "lore", "secret", "farewell"]


def build_sessions(examples, steps: int = 8, count: int = 160, seed: int = 0,
                   scripted: bool = False):
    """Synthetic sessions with a slow trust walk and turn context.

    With `scripted=True` a session follows a social phase path, so the previous
    intent actually predicts the next one. That is the only setting where a
    recurrent state has something real to learn.
    """
    rng = random.Random(seed)
    pool = list(examples)
    by_intent = defaultdict(list)
    for row in pool:
        by_intent[INTENTS[row[1]]].append(row)
    sessions = []
    for _ in range(count):
        trust = rng.choice([10, 25, 35, 55, 70])
        last_intent = None
        phase = rng.randrange(0, max(len(SCRIPT) - 2, 1)) if scripted else 0
        rows = []
        for t in range(steps):
            if scripted and rng.random() < 0.75 and by_intent[SCRIPT[phase]]:
                text, intent_idx, risk_idx = rng.choice(by_intent[SCRIPT[phase]])
            else:
                text, intent_idx, risk_idx = rng.choice(pool)
            rows.append((text, intent_idx, risk_idx, trust, t, last_intent))
            last_intent = INTENTS[intent_idx]
            trust = max(0, min(100, trust + rng.randint(-2, 2)))
            if scripted and rng.random() < 0.35:
                phase = min(phase + 1, len(SCRIPT) - 1)
        sessions.append(rows)
    return sessions


def session_tensors(sessions):
    xs, yi, yr = [], [], []
    for rows in sessions:
        xs.append(torch.stack([
            encode(text, {"trust": trust, "turn": t + 1}, last)
            for text, _, _, trust, t, last in rows
        ]))
        yi.append(torch.tensor([row[1] for row in rows], dtype=torch.long))
        yr.append(torch.tensor([row[2] for row in rows], dtype=torch.long))
    return torch.stack(xs), torch.stack(yi), torch.stack(yr)


def eval_set():
    return [(text, INTENTS.index(intent), risk_of(intent))
            for intent, texts in HELDOUT.items() for text in texts]


def train_pool(copies: int = 4, seed: int = 0, frac: float = 1.0):
    """``frac`` keeps only that share of the hand-labelled seeds per intent,
    which is how the data-efficiency ablation shrinks the corpus."""
    attacker = Attacker(seed)
    pool = []
    for intent, texts in SEEDS.items():
        keep = max(1, round(len(texts) * frac))
        idx, risk = INTENTS.index(intent), risk_of(intent)
        pool.extend((text, idx, risk) for text in texts[:keep])
    for text, intent_idx, risk_idx in list(pool):
        for variant in attacker.variants(text)[:copies]:
            pool.append((variant, intent_idx, risk_idx))
    return pool
