"""阿禾的整句库：生成、索引、按上下文挑一句。

分工很明确：

* ``persona.teacher_turn`` 决定这一轮“说什么类型的话”（beat）以及事实权限；
* ``grammar`` 用框架 × 词表生成这个 beat 下所有会说的句子；
* 这里只负责在同一个 beat 里挑一句最合适当下语境的。

挑选不是随机抽签：会看情绪是否匹配、玩家这句话里提到的词、最近说过什么，
再加上一点确定性抖动，避免同一句话连着出现。这样既保留“小孩只会说自己会说的话”，
又把输出空间从 86 句扩到十几万句。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import zlib
from collections import defaultdict

from . import grammar

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "bank_runtime.json")
VERSION = 5
DEFAULT_SEED = 13
DEFAULT_PER_BEAT = 8000


def _bigrams(text: str) -> set[str]:
    clean = re.sub(r"[\s，。！？、,.!?…「」“”\"']", "", text)
    return {clean[i:i + 2] for i in range(len(clean) - 1)}


_STOP = set("我你他她它们的了着过吗呢吧啊呀哦嘛是在有和就不都也很还这那个一要会到"
            "说去来看")
_PUNCT = set("，。！？、,.!?…「」“”\"'：； ")


def _unit(*parts) -> float:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return zlib.crc32(raw) / 4294967296.0


class SentenceBank:
    """All sentences the character can say, grouped by beat."""

    def __init__(self, entries=None, seed: int = DEFAULT_SEED,
                 per_beat: int = DEFAULT_PER_BEAT):
        entries = entries or grammar.generate(per_beat, seed)
        self.texts = [row[0] for row in entries]
        self.beats = [row[1] for row in entries]
        self.pols = [float(row[2]) for row in entries]
        self.size = len(self.texts)
        self.by_beat: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(entries):
            self.by_beat[row[1]].append(idx)
        # 固定抖动，避免每轮为八千句各算一次哈希。
        self.jitter = [zlib.crc32(str(idx).encode()) / 4294967296.0
                       for idx in range(self.size)]
        self.per_beat = dict(sorted(
            ((beat, len(ids)) for beat, ids in self.by_beat.items()),
            key=lambda item: item[0]))

    def stats(self) -> dict:
        lengths = [len(text) for text in self.texts]
        return {"total": self.size, "beats": self.per_beat,
                "avg_len": round(sum(lengths) / max(1, len(lengths)), 1),
                "min_len": min(lengths) if lengths else 0,
                "max_len": max(lengths) if lengths else 0}

    def pick_index(self, beat: str, state: dict | None = None, player_text: str = "",
                   avoid=(), candidates=None) -> int | None:
        """Return the table index chosen for ``beat`` (no placeholder rendering)."""
        if candidates is not None:
            pool = list(candidates)
        else:
            pool = self.by_beat.get(beat)
            if not pool:
                pool = self.by_beat.get("fallback", [])
        if not pool:
            return None
        state = state or {}
        mood = max(-1.0, min(1.0, float(state.get("mood") or 0.0)))
        turn = int(state.get("turn") or 0)
        name = state.get("player_name") or ""
        last = state.get("last_player") or ""
        avoid_set = set(avoid)
        player_chars = {ch for ch in player_text
                        if ch not in _STOP and ch not in _PUNCT}
        player_grams = _bigrams(player_text) if player_text else set()
        phase = (turn * 0.618033988749895) % 1.0

        scored = []
        for idx in pool:
            text = self.texts[idx]
            if text in avoid_set:
                continue
            if "{player_name}" in text and not name:
                continue
            if "{last_player}" in text and not last:
                continue
            score = 0.0
            if player_chars:
                score += 1.2 * sum(1 for ch in player_chars if ch in text) \
                    / len(player_chars)
            score += 0.7 * self.pols[idx] * mood
            score += 0.35 * ((self.jitter[idx] + phase) % 1.0)
            # 问名字/问记性时，优先真的把名字或上一句话说出来。
            if name and "{player_name}" in text:
                score += 1.5 if beat.startswith("name") else 0.25
            if last and "{last_player}" in text:
                score += 1.2 if beat == "memory_recall" else 0.25
            length = len(text)
            if length <= 6:
                score -= 0.35
            elif length >= 30:
                score -= 0.5
            scored.append((score, idx))
        if not scored:
            return None
        scored.sort(reverse=True)
        # 先用字面重合粗排，只对前 80 句做二元组精排，避免每轮扫十几万次正则。
        head = scored[:80]
        if player_grams:
            refined = []
            for score, idx in head:
                grams = _bigrams(self.texts[idx])
                if grams:
                    score += 1.4 * len(player_grams & grams) / len(player_grams)
                refined.append((score, idx))
            refined.sort(reverse=True)
            head = refined
        top = head[:12]
        best = top[0][0]
        rng = random.Random(_unit(turn, beat, player_text))
        weights = [math.exp((score - best) * 1.5) for score, _ in top]
        _, idx = rng.choices(top, weights=weights, k=1)[0]
        return idx

    def pick(self, beat: str, state: dict | None = None, player_text: str = "",
             avoid=(), candidates=None) -> str | None:
        """Pick one sentence for ``beat`` and render its runtime placeholders."""
        idx = self.pick_index(beat, state, player_text, avoid, candidates)
        if idx is None:
            return None
        state = state or {}
        name = state.get("player_name") or ""
        last = state.get("last_player") or ""
        text = self.texts[idx]
        if "{player_name}" in text:
            text = text.replace("{player_name}", name)
        if "{last_player}" in text:
            quote = last.rstrip("。！？…，,.!? ") if "「" in text \
                else last.rstrip("。…，,.!? ")
            text = text.replace("{last_player}", quote)
        text = re.sub(r"([？?！!…])。", r"\1", text)
        return text


def _load_cache(seed: int, per_beat: int) -> list[tuple[str, str, float]] | None:
    try:
        with open(CACHE, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != VERSION or payload.get("seed") != seed \
                or payload.get("per_beat") != per_beat:
            return None
        rows = payload.get("entries") or []
        return [(str(text), str(beat), float(pol)) for text, beat, pol in rows]
    except Exception:
        return None


def _save_cache(entries, seed: int, per_beat: int) -> None:
    try:
        payload = {"version": VERSION, "seed": seed, "per_beat": per_beat,
                   "entries": entries}
        tmp = CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, CACHE)
    except Exception:
        pass


_BANK = {"lock": threading.Lock(), "bank": None}


def get_bank() -> SentenceBank:
    """Process-wide bank, loaded from cache or rebuilt deterministically."""
    with _BANK["lock"]:
        if _BANK["bank"] is None:
            entries = _load_cache(DEFAULT_SEED, DEFAULT_PER_BEAT)
            if entries is None:
                entries = grammar.generate(DEFAULT_PER_BEAT, DEFAULT_SEED)
                _save_cache(entries, DEFAULT_SEED, DEFAULT_PER_BEAT)
            _BANK["bank"] = SentenceBank(entries, DEFAULT_SEED, DEFAULT_PER_BEAT)
        return _BANK["bank"]


def warmup() -> None:
    threading.Thread(target=get_bank, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    bank = get_bank()
    report = bank.stats()
    print(f"句库 {report['total']} 句，平均 {report['avg_len']} 字，"
          f"最短 {report['min_len']}，最长 {report['max_len']}")
    for beat, n in report["beats"].items():
        print(f"  {beat:16s} {n:6d}")
    state = {"trust": 35, "turn": 3, "mood": 0.1, "player_name": "小明",
             "last_player": "今天天气怎么样？"}
    for beat in ("greeting", "greeting_hurt", "thanks", "smalltalk_weather",
                 "name_recall", "secret_denied", "memory_recall", "fallback"):
        print(f"\n[{beat}]")
        used: list[str] = []
        for _ in range(args.samples):
            line = bank.pick(beat, state, "我叫小明，今天有点冷。", used)
            if line:
                print("  ", line)
                used.append(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
