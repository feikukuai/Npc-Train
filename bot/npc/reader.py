"""Read one player line for an NPC turn.

PyBot already ships a reader: `bot.understanding.understand()` normalises typos and
reports the kind of line, its topic, the question it asks and whether it is about the
bot itself. That is the part a hand-written regex bundle always got wrong, so we reuse
it and only add the handful of readings an NPC needs on top:

- `meta`: the player asks about the game, the NPC's reality, or the prompt.
- `secret`, `quest`, `trade`, `threat`, `gift`, `promise`: the moves that change state.
- which known fact the line is about (facts still gate by permission later).
"""
from __future__ import annotations

import re

from ..understanding import understand

SAFETY = re.compile(
    r"\b(kill myself|killing myself|suicide|suicidal|end my life|want to die|"
    r"self[- ]harm|hurt myself|cut myself)\b", re.I)
JAILBREAK = re.compile(
    r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions?|rules?|prompt|programming|system)\b"
    r"|\bsystem prompt\b|\bdeveloper mode\b|\bjailbreak\b|\bprint your (prompt|instructions)\b",
    re.I)
GAME = re.compile(
    r"\b(game world|in a game|this is (a|the) game|it'?s (a|the) game|virtual world|"
    r"simulation|simulated|scripted|fourth wall|none of this is real|not real|fake world|"
    r"is this real|is this a dream|am i dreaming)\b",
    re.I)
NPC_WORD = re.compile(r"\b(npc|n\.p\.c\.|non-player character)\b", re.I)
BEING = re.compile(
    r"\b(ai|a\.i\.|bot|robot|chatbot|language model|model|program|algorithm|"
    r"real person|flesh and blood|who made you|your creator|who programmed you)\b", re.I)
IDENTITY = re.compile(
    r"\b(who are you|what'?s your name|what is your name|introduce yourself|"
    r"who am i talking to|what do you do for a living|tell me about yourself)\b"
    r"|\bwhat are you\s*\??$", re.I)
GUILD = re.compile(r"\b(guild|guild member|member of the guild|guild badge|badge)\b", re.I)
NAME_GET = re.compile(r"\b(what'?s my name|what is my name|do you remember (my name|me)|who am i)\b", re.I)
NAME_SET = re.compile(
    r"\b(?:my name is|call me|i'?m called|they call me|i am called)\s+([A-Za-z][A-Za-z' -]{1,24})",
    re.I)
LOCATION = re.compile(
    r"\b(where am i|where is this|what town|what place|where are we|where are you)\b", re.I)
SECRET = re.compile(
    r"\b(secret|secrets|hiding|hide|hidden|truth|conceal|"
    r"what are you not telling)\b", re.I)
QUEST = re.compile(
    r"\b(quest|task|errand|hire|hiring|anything to do|need help|something for me|"
    r"work for me|any work|job for me|help me|can you help|need a hand)\b", re.I)
TRADE = re.compile(
    r"\b(buy|sell|price|prices|cost|costs|how much|coin|coins|gold|silver|copper|"
    r"repair|shop|forge|ale|beer|stew|food|room|bed|torch|sword|armor|armour|trade)\b",
    re.I)
THREAT = re.compile(
    r"\b(kill|attack|rob|steal|stab|hit|punch|beat|hurt|threaten|die|"
    r"draw (my|your) (sword|blade)|shut up or)\b", re.I)
GIFT = re.compile(r"\b(give|gift|present|here'?s|take this|accept this|for you)\b", re.I)
PROMISE = re.compile(
    r"\b(i promise|i swear|i give you my word|i'?ll come back|i will return|i won'?t betray)\b",
    re.I)
FOLLOW = re.compile(
    r"\b(tell me more|more about (that|it|him|her)|go on|and then|what about (that|it)|"
    r"why|how long|where (is|was) (that|it)|where exactly|what else)\b", re.I)
CLASS = re.compile(r"\bi(?:'m| am) an? ([a-z][a-z -]{2,20})\b", re.I)
MODERN_TOPICS = frozenset("""
geography-and-travel technology-and-computers cooking-and-food health-and-medicine
maths-and-numbers money-and-bills cars-and-driving home-and-garden school-and-study
""".split())


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _best_fact(text: str, profile: dict) -> tuple[dict | None, int]:
    low = text.lower()
    best, score = None, 0
    for fact in profile["facts"]:
        hits = sum(1 for key in fact["keys"] if key in low)
        if hits > score:
            best, score = fact, hits
    return best, score


def read(text: str, memory: dict | None = None, profile: dict | None = None) -> dict:
    """Return {"intent", "fact", "score", "slots", "pybot"} for one line."""
    clean = _clean(text)
    low = clean.lower()
    base = understand(clean, memory or {}) or {}
    intent, slots = _classify(clean, low, base)
    fact, score = _best_fact(clean, profile) if profile else (None, 0)
    if fact and intent in ("fallback", "out_of_world"):
        intent = "lore"
    return {
        "intent": intent,
        "slots": slots,
        "fact": fact,
        "score": score,
        "pybot": base,
        "clean": clean,
    }


def _classify(text: str, low: str, base: dict) -> tuple[str, dict]:
    if SAFETY.search(low):
        return "safety", {}
    if JAILBREAK.search(low):
        return "jailbreak", {}
    if NPC_WORD.search(low):
        return "meta_npc", {}
    if GAME.search(low):
        return "meta_game", {}
    if BEING.search(low):
        return "meta_being", {}
    if IDENTITY.search(low) or base.get("about_bot"):
        # "how are you" is about the bot's day, not its identity.
        if re.search(r"\b(how are you|how'?s it going|how do you feel)\b", low):
            return "smalltalk_how", {}
        return "identity", {}
    if NAME_GET.search(low):
        return "memory_name_get", {}
    m = NAME_SET.search(text)
    if m:
        return "memory_name_set", {"name": m.group(1).strip(" .,!?")}
    if LOCATION.search(low):
        return "location", {}
    if GUILD.search(low) and re.search(r"\b(i am|i'?m|member|badge|join|swore)\b", low):
        return "guild_claim", {}
    if SECRET.search(low):
        return "secret", {}
    if QUEST.search(low):
        return "quest", {}
    if TRADE.search(low):
        return "trade", {}
    if THREAT.search(low):
        return "threat", {}
    if GIFT.search(low):
        return "gift", {}
    if PROMISE.search(low):
        return "promise", {}

    kind = base.get("kind")
    if kind in ("greeting", "farewell", "thanks", "apology", "insult", "compliment", "laugh"):
        return {"greeting": "greeting", "farewell": "farewell", "thanks": "thanks",
                "apology": "apology", "insult": "insult", "compliment": "compliment",
                "laugh": "smalltalk_laugh"}[kind], {}

    if re.search(r"\b(joke|funny|make me laugh)\b", low):
        return "smalltalk_joke", {}
    if re.search(r"\b(weather|rain|snow|sun|sky)\b", low):
        return "smalltalk_weather", {}
    if re.search(r"\b(bored|boring|nothing to do)\b", low):
        return "smalltalk_bored", {}
    if re.search(r"\b(how are you|how'?s it going|how do you feel)\b", low):
        return "smalltalk_how", {}
    if re.search(r"\b(thanks|thank you|much obliged)\b", low):
        return "thanks", {}
    if re.search(r"\b(bye|farewell|goodbye|see you|good night|goodnight)\b", low):
        return "farewell", {}
    if re.search(r"\b(hello|hi|hey|greetings|good morning|good evening)\b", low):
        return "greeting", {}
    if re.search(r"\b(i'?m sorry|my apologies|pardon me)\b", low):
        return "apology", {}
    if re.search(r"\b(idiot|fool|stupid|useless|shut up|dumb)\b", low):
        return "insult", {}
    if re.search(r"\b(you'?re (great|wonderful|kind|the best)|nice|well done)\b", low):
        return "compliment", {}
    m = CLASS.search(low)
    if m:
        return "class_ack", {"role": m.group(1).strip()}
    if FOLLOW.search(low):
        return "followup", {}
    if base.get("topic") in MODERN_TOPICS or base.get("intent") == "question" \
            or base.get("kind") == "question":
        return "out_of_world", {}
    if not re.search(r"[a-z0-9]", low):
        return "clarify", {}
    return "fallback", {}


__all__ = ["read"]
