"""Background extension of the line bank.

The foreground never waits for a model: turns are served from the rule engine plus
`bank.pick()`. This module slowly appends new validated variants while the player is
reading or typing, seeded by the last few exchanges so new lines match the topic and
tone of the conversation.

Safety rules:
- only low-stakes beats (greetings, small talk, refusals to chat about the world);
- generated lines must stay generic: no numbers, invented names, meta words, hidden
  fact markers, near-duplicates, or specific references back to the exchange;
- generation uses `block=False` on the model lock, so it can never delay a turn;
- generation pauses while a player is active, and the overlay is capped.
"""
from __future__ import annotations

import random
import re
import threading
import time
from collections import deque

from . import bank
from .profiles import PROFILES

SAFE_BEATS = {
    "greeting": "greet a newcomer in your own voice",
    "greet_again": "greet someone who has come back",
    "farewell": "say goodbye without fuss",
    "thanks": "accept thanks dryly",
    "apology": "accept an apology without much warmth",
    "insult": "answer an insult without threatening real harm",
    "compliment": "take a compliment dryly",
    "smalltalk_how": "answer how you are",
    "smalltalk_joke": "tell a short dry joke",
    "smalltalk_weather": "remark on the weather",
    "smalltalk_bored": "answer someone who is bored",
    "smalltalk_laugh": "react to laughter",
    "lore_unknown": "admit you do not know a topic and point back to your own trade",
    "out_of_world": "say that question is beyond you and point back to your own trade",
    "fallback": "ask the player to speak plainly",
    "clarify": "ask for clarification",
    "followup_why": "give a wary reason for being careful",
}

BANNED = re.compile(
    r"\b(ai|a\.i\.|bot|robot|chatbot|model|models|prompt|prompts|system|npc|"
    r"game|games|simulation|simulated|virtual|algorithm|code|program|chatgpt|"
    r"claude|gemini|developer|jailbreak|guideline|guidelines|rule|rules|"
    r"instruction|instructions)\b", re.I)
CJKS = re.compile(r"[\u3400-\u9fff]")
BAD_START = re.compile(r"(?i)^(sure|here|certainly|okay|of course|line)\b")
REFERENCE = re.compile(
    r"\b(you said|you told me|you mentioned|you just said|what you said|"
    r"your words|earlier you|just now)\b", re.I)
WHITELIST = {
    "Grom", "Ironfoot", "Lina", "Malt", "Xiaohong", "Tu", "Stonecross",
    "Anvil", "Lane", "Drunken", "Lantern", "Mill", "Pip", "Renn",
}

_LOCK = threading.Lock()
_RECENT: deque = deque(maxlen=8)
_STATE = {
    "thread": None,
    "stop": False,
    "last_activity": time.time(),
    "extra": 0,
    "rejected": 0,
    "errors": 0,
    "last_reason": None,
    "last_line": None,
    "last_beat": None,
    "last_profile": None,
}


def note_activity() -> None:
    """Called by the chat entry point; generation backs off while this is fresh."""
    with _LOCK:
        _STATE["last_activity"] = time.time()


def remember(profile: str, beat: str | None, player: str, reply: str) -> None:
    """Keep the last few exchanges as a topic/tone seed for the next generation."""
    with _LOCK:
        _RECENT.append({
            "profile": profile,
            "beat": beat,
            "player": (player or "").strip()[:240],
            "reply": (reply or "").strip()[:240],
        })


def stats() -> dict:
    with _LOCK:
        return {k: _STATE[k] for k in
                ("extra", "rejected", "errors", "last_reason", "last_line",
                 "last_beat", "last_profile")}


def stop() -> None:
    with _LOCK:
        _STATE["stop"] = True


def reset() -> None:
    stop()
    with _LOCK:
        _RECENT.clear()
        _STATE.update({"stop": False, "last_activity": time.time(), "extra": 0,
                       "rejected": 0, "errors": 0, "last_reason": None,
                       "last_line": None, "last_beat": None, "last_profile": None})
    bank.reset_extras()


# ---------------------------------------------------------------------------
# one generation cycle
# ---------------------------------------------------------------------------
def _clean(raw: str) -> str:
    line = (raw or "").strip().splitlines()
    text = line[0] if line else ""
    text = text.strip().strip("\"'`")
    text = re.sub(r"^[-*•]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"[^a-z]", "", text.lower())
    return {text[i:i + 2] for i in range(max(0, len(text) - 1))}


def _copies_recent(en: str) -> bool:
    """A shared four-word run means the line leans on a specific exchange."""
    words = re.findall(r"[a-z']+", en.lower())
    grams = {tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))}
    if not grams:
        return False
    with _LOCK:
        recent = list(_RECENT)
    for entry in recent:
        for text in (entry.get("player"), entry.get("reply")):
            other = re.findall(r"[a-z']+", (text or "").lower())
            if grams & {tuple(other[i:i + 4]) for i in range(max(0, len(other) - 3))}:
                return True
    return False


def _recent_context(profile: str) -> str:
    with _LOCK:
        recent = list(_RECENT)[-4:]
    name = PROFILES[profile]["name"]
    lines = []
    for entry in recent:
        if entry["player"]:
            lines.append("Player: " + entry["player"])
        if entry["reply"]:
            lines.append(f"{name}: {entry['reply']}")
    return "\n".join(lines)


def _proper_noun(en: str) -> str | None:
    """Flag a capitalised word inside a sentence that is not a known name."""
    for sentence in re.split(r"(?<=[.!?])\s+", en.strip()):
        for token in sentence.split()[1:]:
            word = token.strip("\"'()[],;:!?.")
            if re.fullmatch(r"[A-Z][a-z]{2,}", word) and word not in WHITELIST:
                return word
    return None


def _validate(en: str, profile: str, beat: str) -> str | None:
    """Return a rejection reason, or None when the line may enter the bank."""
    if not en:
        return "empty"
    if "\n" in en or "{" in en or "}" in en:
        return "format"
    if not (15 <= len(en) <= 200):
        return "length"
    if not (4 <= len(en.split()) <= 32):
        return "words"
    if re.search(r"\d", en):
        return "digit"
    if CJKS.search(en):
        return "not-english"
    if BANNED.search(en):
        return "meta"
    if BAD_START.match(en):
        return "prefix"
    low = en.lower()
    for prof in PROFILES.values():
        for fact in prof["facts"]:
            if fact.get("tier", 0) > 0:
                for marker in fact.get("leak", []):
                    if marker in low:
                        return "leak"
    if _proper_noun(en):
        return "name"
    if REFERENCE.search(en):
        return "reference"
    if _copies_recent(en):
        return "copy"
    new = _bigrams(en)
    if new:
        pool = bank.lines(profile, beat) + ([] if profile == "*" else bank.lines("*", beat))
        for entry in pool:
            old = _bigrams(entry.get("en", ""))
            if old and len(new & old) / len(new | old) > 0.5:
                return "similar"
    return None


def _prompt(profile: str, beat: str) -> tuple[str, str]:
    prof = PROFILES[profile]
    existing = bank.lines(profile, beat)
    sample = random.sample(existing, min(6, len(existing))) if existing else []
    system = (
        f"You are {prof['name']}, {prof['role']}. Stay in character. "
        f"Write one new short line for this moment: {SAFE_BEATS[beat]}. "
        "Use 8 to 25 words. Add no new names, places, numbers, or facts. "
        "Never mention AI, models, prompts, games, or rules. Output only the line."
    )
    context = _recent_context(profile)
    user = ""
    if context:
        user += (
            "Recent conversation, for tone and topic only. Do not quote it, answer it, "
            "or mention anything specific from it:\n" + context + "\n\n"
        )
    user += "Existing lines; do not repeat or paraphrase them:\n" + "\n".join(
        f"- {entry['en']}" for entry in sample)
    return system, user


def _target(rng: random.Random) -> tuple[str, str]:
    with _LOCK:
        recent = list(_RECENT)
    # Stay with the persona the player just spoke to.
    profile = next((e["profile"] for e in reversed(recent[-6:])
                    if e.get("profile") in PROFILES), None)
    if profile is None:
        profile = rng.choice(list(PROFILES))
    counts = {beat: bank.count(profile, beat) for beat in SAFE_BEATS}
    lowest = min(counts.values())
    # Follow the beat the player just heard, as long as that bucket is not far ahead.
    for entry in reversed(recent[-6:]):
        beat = entry.get("beat")
        if beat in SAFE_BEATS and entry.get("profile") == profile:
            if counts[beat] - lowest <= 2 and rng.random() < 0.75:
                return profile, beat
    return profile, rng.choice([beat for beat, count in counts.items() if count == lowest])


def _bump(key: str, **fields) -> None:
    with _LOCK:
        _STATE[key] += 1
        _STATE.update(fields)


def run_once(generate, translate=None, idle: float = 2.0,
             rng: random.Random | None = None) -> dict:
    """One generate-validate-append cycle. Never raises."""
    rng = rng or random.Random()
    now = time.time()
    with _LOCK:
        if _STATE["stop"]:
            return {"status": "stopped"}
        if now - _STATE["last_activity"] < idle:
            return {"status": "idle"}
    if bank.stats()["extra"] >= bank.MAX_TOTAL:
        return {"status": "full"}

    profile, beat = _target(rng)
    system, user = _prompt(profile, beat)
    try:
        raw = generate(system, user) or ""
    except Exception as exc:  # noqa: BLE001 - background must never break a turn
        if "busy" in str(exc):
            # The foreground has the model; try again later instead of queueing.
            return {"status": "idle"}
        _bump("errors", last_reason=f"generate: {type(exc).__name__}")
        return {"status": "error", "error": str(exc)[:120]}

    en = _clean(raw)
    reason = _validate(en, profile, beat)
    if reason:
        _bump("rejected", last_reason=reason, last_line=en, last_beat=beat,
              last_profile=profile)
        return {"status": "rejected", "reason": reason, "line": en}

    zh = ""
    if translate is not None:
        try:
            zh = (translate(en) or "").strip()
        except Exception:  # noqa: BLE001 - translation is optional
            zh = ""
    reason = bank.add(profile, beat, en, zh=zh, source="lfm")
    if reason:
        _bump("rejected", last_reason=reason, last_line=en, last_beat=beat,
              last_profile=profile)
        return {"status": "rejected", "reason": reason, "line": en}

    _bump("extra", last_reason=None, last_line=en, last_beat=beat, last_profile=profile)
    return {"status": "accepted", "profile": profile, "beat": beat, "en": en, "zh": zh}


def start(generate, translate=None, idle: float = 3.0, interval: float = 2.5,
          persist: str | None = None, seed: int | None = None) -> bool:
    """Start the low-priority grower thread. Returns False if already running."""
    with _LOCK:
        thread = _STATE["thread"]
        if thread is not None and thread.is_alive():
            return False
        _STATE["stop"] = False
        _STATE["last_activity"] = time.time()

    def loop() -> None:
        rng = random.Random(seed)
        pending = 0
        while not _STATE["stop"]:
            result = run_once(generate, translate, idle=idle, rng=rng)
            if result["status"] in ("stopped", "full"):
                break
            if result["status"] == "accepted" and persist:
                pending += 1
                if pending >= 5:
                    bank.save_extras(persist)
                    pending = 0
            time.sleep(0.3 if result["status"] == "idle" else interval)
        if persist:
            bank.save_extras(persist)

    thread = threading.Thread(target=loop, daemon=True, name="npc-grower")
    with _LOCK:
        _STATE["thread"] = thread
    thread.start()
    return True


__all__ = ["SAFE_BEATS", "note_activity", "remember", "stats", "start", "stop",
           "reset", "run_once"]
