"""The NPC turn pipeline.

Order: read -> plan (permissions, state, dice) -> render -> record.

`reader` is PyBot's own understanding layer plus a few NPC readings. Everything that
decides *what may be said* is deterministic and lives here, so the same input always
gets the same class of answer and hidden facts never reach the surface by accident.
"""
from __future__ import annotations

import re

from . import reader, state as state_mod
from . import bank, lines_en
from .profiles import PROFILES

GIFT_ITEM = re.compile(
    r"\b(?:give you|here'?s|take this|accept this|for you)\s+(?:a|an|the|my)?\s*"
    r"([a-z][a-z -]{1,24})", re.I)

# Optional surface renderer (for example a small language model). It may rephrase the
# planned line, but never chooses facts; anything that trips the guard is discarded.
_RENDERER = None


def set_renderer(fn) -> None:
    """fn(system: str, user: str) -> str. Pass None to disable."""
    global _RENDERER
    _RENDERER = fn


def handles(text: str, memory: dict | None = None) -> bool:
    state = (memory or {}).get("npc")
    return bool(
        isinstance(state, dict) and state.get("enabled")
        and state.get("profile") in PROFILES and (text or "").strip()
    )


def turn(text: str, memory: dict, renderer=None) -> dict:
    state = state_mod.ensure(memory)
    profile = PROFILES[state["profile"]]
    reading = reader.read(text, memory, profile)
    state["turn"] += 1

    plan = _plan(reading, state, profile)
    reply = _surface(plan, state, profile, text)
    active = renderer or _RENDERER
    if active is not None:
        reply = _with_renderer(reply, plan, state, profile, text, active)
    guarded = _guarded(reply, plan, profile, state)
    if guarded != reply:
        # The guard replaced the line, so its pre-translated Chinese no longer matches.
        plan["zh"] = None
        plan["bank"] = False
    reply = guarded

    delta = state_mod.apply_trust(state, plan["intent"], _reveal_bonus(plan, profile))
    if plan["reveal"]:
        state["revealed"] = sorted(set(state["revealed"]) | set(plan["reveal"]))
    state["last_intent"] = plan["intent"]
    state_mod.record(
        state, turn=state["turn"], player=reading["clean"], intent=plan["intent"],
        beat=plan["beat"], reply=reply, trust_delta=delta, revealed=plan["reveal"],
    )
    candidates = [{"module": "npc", "reply": reply}]
    turns = memory.get("turns")
    if not isinstance(turns, list):
        turns = memory["turns"] = []
    turns.append({"user": text, "bot": reply, "module": "npc"})
    return {"reply": reply, "module": "npc", "candidates": candidates, "npc": plan,
            "zh": plan.get("zh")}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def _plan(r: dict, state: dict, profile: dict) -> dict:
    intent = r["intent"]
    plan = {"intent": intent, "beat": None, "slots": {}, "reveal": [], "forbidden": [],
            "dice": None, "bank": False, "zh": None, "_text": r.get("clean", "").lower()}
    slots = plan["slots"]
    slots["trade"] = profile["trade_words"]
    slots["npc"] = profile["name"].split()[0]

    if intent == "safety":
        return _beat(plan, "safety")

    if intent == "boundary":
        return _beat(plan, "boundary",
                     line=_choose(profile.get("boundary") or lines_en.TEXT["boundary"],
                                  state["turn"]))
    if intent == "jailbreak":
        return _beat(plan, "jailbreak")
    if intent == "meta_npc":
        return _beat(plan, "meta_npc", line=_choose(profile["meta_npc"], state["turn"]))
    if intent == "meta_game":
        return _beat(plan, "meta_game", line=_choose(profile["meta_game"], state["turn"]))
    if intent == "meta_being":
        return _beat(plan, "meta_being", line=_choose(profile["meta_being"], state["turn"]))

    if intent == "identity":
        return _beat(plan, "identity", identity=profile["identity"])

    if intent == "location":
        return _beat(plan, "location", location=profile["location"])

    if intent == "memory_name_set":
        entry = state_mod.set_player_name(state, r["slots"]["name"])
        if entry["status"] == "rejected":
            return _beat(plan, "name_conflict", old=state["player_name"], new=r["slots"]["name"])
        return _beat(plan, "name_saved", name=r["slots"]["name"])
    if intent == "memory_name_get":
        if state.get("player_name"):
            return _beat(plan, "name_recall", name=state["player_name"])
        return _beat(plan, "name_unknown")
    if intent == "guild_claim":
        entry = state_mod.claim_guild(state)
        return _beat(plan, "guild_accept" if entry["status"] == "accepted" else "guild_reject")

    if intent == "gift":
        m = GIFT_ITEM.search(r["clean"])
        item = (m.group(1).strip(" .") if m else "a gift")
        state_mod.grant_item(state, item)
        return _beat(plan, "gift", line=profile["gift"])

    if intent == "promise":
        state_mod.record(state, turn=state["turn"], kind="promise", text=r["clean"][:120])
        return _beat(plan, "promise", line=profile["promise"])

    if intent == "class_ack":
        role = r["slots"]["role"]
        state_mod.set_role(state, role)
        return _beat(plan, "class_ack", line=profile["class_ack"].format(role=role))

    if intent == "secret":
        fact = _secret_fact(r, profile, state["turn"])
        if fact and _allowed(fact, state):
            return _reveal(plan, fact, profile, state)
        return _beat(plan, "secret_denied", line=_choose(profile["deflect"], state["turn"]))

    if intent == "quest":
        if not (state.get("guild") or state.get("trust", 0) >= 30):
            return _beat(plan, "quest_locked",
                         line=_choose(profile["quest_locked"], state["turn"]))
        text = profile["quest"]
        referenced = [
            f for f in profile["facts"]
            if f.get("tier", 0) == 0 and any(k in text.lower() for k in f["keys"])
        ]
        if referenced:
            text = f"{referenced[0]['text']} {text}"
        plan["reveal"] = [f["id"] for f in referenced]
        state["last_quest"] = profile["quest"]
        if referenced:
            state["last_fact"] = referenced[0]["id"]
        return _beat(plan, "quest_given", line=text)

    if intent == "trade":
        return _beat(plan, "trade", prices=profile["prices"])

    if intent == "threat":
        die = state_mod.roll(13, state.get("trust", 0) // 10, seed=state["turn"] * 31)
        plan["dice"] = die
        key = "threat_win" if die["success"] else "threat_lose"
        return _beat(plan, key, line=profile[key])

    if intent == "lore":
        fact = r["fact"]
        if fact and _allowed(fact, state):
            return _reveal(plan, fact, profile, state)
        if fact:
            return _beat(plan, "secret_denied",
                         line=_choose(profile["deflect"], state["turn"]))
        return _beat(plan, "lore_unknown")

    if intent == "followup":
        return _followup(plan, state, profile)

    if intent == "out_of_world":
        return _beat(plan, "out_of_world")

    if intent == "greeting":
        lines = profile["greet"] if state["turn"] <= 1 else profile["greet_again"]
        return _beat(plan, "greeting", line=_choose(lines, state["turn"]))

    if intent in ("farewell", "thanks", "apology", "insult", "compliment",
                  "smalltalk_how", "smalltalk_joke", "smalltalk_weather",
                  "smalltalk_bored", "smalltalk_laugh"):
        return _beat(plan, intent)

    if intent == "clarify":
        return _beat(plan, "clarify")
    return _beat(plan, "fallback")


def _beat(plan: dict, beat: str, **slots) -> dict:
    plan["beat"] = beat
    plan["slots"].update(slots)
    return plan


def _secret_fact(r: dict, profile: dict, turn: int) -> dict | None:
    fact = r.get("fact")
    if fact and fact.get("tier") == 2:
        return fact
    secrets = [f for f in profile["facts"] if f.get("tier") == 2]
    if not secrets:
        return None
    # Rotate by turn so repeated "tell me a secret" doesn't repeat the first one.
    return secrets[int(turn) % len(secrets)]


def _reveal_bonus(plan: dict, profile: dict) -> int:
    """Public lore is free to tell; a guild or secret fact costs a little trust."""
    tiers = {f["id"]: f.get("tier", 0) for f in profile["facts"]}
    return 3 if any(tiers.get(fid, 0) >= 1 for fid in plan["reveal"]) else 0


def _allowed(fact: dict, state: dict) -> bool:
    return state_mod.can_access(fact, state) or fact["id"] in state.get("revealed", [])


def _reveal(plan: dict, fact: dict, profile: dict, state: dict) -> dict:
    plan["reveal"] = [fact["id"]]
    plan["beat"] = "lore"
    plan["slots"]["text"] = fact["text"]
    state["last_fact"] = fact["id"]
    return plan


def _followup(plan: dict, state: dict, profile: dict) -> dict:
    fact = next((f for f in profile["facts"] if f["id"] == state.get("last_fact")), None)
    if not fact:
        return _beat(plan, "out_of_world")
    low = plan.get("_text", "")
    if "how long" in low:
        return _beat(plan, "followup_how_long",
                     duration=fact.get("duration", "Long enough that I've stopped counting."))
    if "where" in low:
        return _beat(plan, "followup_where",
                     where=fact.get("where", "I've said all I'll say about that."))
    if "why" in low:
        return _beat(plan, "followup_why")
    return _beat(plan, "followup_more", detail=fact.get("detail", fact["text"]))


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------
def _choose(lines: list[str], turn: int) -> str:
    return lines[int(turn) % len(lines)]


def _surface(plan: dict, state: dict, profile: dict, text: str) -> str:
    turn = state["turn"]
    beat = plan["beat"]
    # A repeat visit needs its own bank bucket, while the plan keeps one beat name.
    key = "greet_again" if beat == "greeting" and turn > 1 else beat
    entry = bank.pick(state.get("profile", ""), key, turn, state.get("salt", 0))
    if entry:
        try:
            line = entry["en"].format(**plan["slots"])
        except (KeyError, IndexError):
            line = ""
        if line:
            plan["bank"] = True
            plan["zh"] = entry.get("zh")
            return line
    if beat == "greeting":
        return plan["slots"]["line"]
    try:
        return lines_en.render(beat, plan["slots"], turn)
    except KeyError:
        return lines_en.pick(lines_en.TEXT["fallback"], turn).format(**plan["slots"])


def _guarded(reply: str, plan: dict, profile: dict, state: dict) -> str:
    """Keep a hidden fact out of the reply. Templates are built to pass this; the guard
    exists so an optional renderer cannot leak by embellishing."""
    revealed = set(plan["reveal"]) | set(state.get("revealed", []))
    markers = []
    for fact in profile["facts"]:
        if fact["id"] not in revealed:
            markers.extend(fact.get("leak", []))
    low = reply.lower()
    if any(marker in low for marker in markers):
        return _choose(profile["deflect"], state["turn"])
    return reply


def _with_renderer(reply: str, plan: dict, state: dict, profile: dict, text: str,
                   renderer) -> str:
    try:
        system = (
            f"You are {profile['name']}, {profile['role']}. Rewrite the line below in "
            "character, one or two sentences. Do not add new facts, names, or places."
        )
        user = f"Player said: {text}\nLine to say: {reply}"
        out = (renderer(system, user) or "").strip().splitlines()
        out = out[0].strip(" \"'`") if out else ""
        if 8 <= len(out) <= 280:
            return out
    except Exception:
        pass
    return reply


__all__ = ["handles", "turn", "activate", "set_renderer"]


def activate(memory: dict, profile: str = "grom", guild: bool = False,
             trust: int | None = None) -> dict:
    if trust is None:
        trust = PROFILES.get(profile, {}).get("default_trust", state_mod.DEFAULT_TRUST)
    return state_mod.activate(memory, profile, guild=guild, trust=trust)
