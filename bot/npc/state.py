"""NPC state: trust, permissions, discovered facts, and the audit of what was written.

This is the authoritative half of the NPC. The reader may *propose* what a line means,
and an optional language model may *suggest* a memory, but only this module decides
what enters the record.
"""
from __future__ import annotations

import random

DEFAULT_TRUST = 10


def fresh(profile: str = "grom", guild: bool = False, trust: int = DEFAULT_TRUST) -> dict:
    return {
        "enabled": True,
        "profile": profile,
        "trust": int(trust),
        "guild": bool(guild),
        "items": [],
        "revealed": [],
        "events": [],
        "memory_log": [],
        "turn": 0,
        "player_name": None,
        "player_role": None,
        "last_fact": None,
        "last_quest": None,
        "last_intent": None,
        "claims": 0,
        # Bank rotation offset: two sessions get different line orders.
        "salt": random.randrange(1, 1 << 30),
    }


def ensure(memory: dict, profile: str | None = None) -> dict:
    """Return the NPC state, creating or repairing it when needed."""
    state = memory.get("npc")
    if isinstance(state, dict) and state.get("enabled"):
        if profile:
            state.setdefault("profile", profile)
        return state
    existing = state.get("profile") if isinstance(state, dict) else None
    new_state = fresh(profile or existing or "grom")
    memory["npc"] = new_state
    return new_state


def activate(memory: dict, profile: str, guild: bool = False, trust: int = DEFAULT_TRUST) -> dict:
    state = fresh(profile, guild=guild, trust=trust)
    memory["npc"] = state
    return state


def can_access(fact: dict, state: dict) -> bool:
    tier = fact.get("tier", 0)
    if tier <= 0:
        return True
    if tier == 1:
        return bool(state.get("guild"))
    return state.get("trust", 0) >= 50 or any("rune shard" in item for item in state.get("items", []))


def visible(profile: dict, state: dict) -> list[dict]:
    return [f for f in profile["facts"] if can_access(f, state)]


def hidden(profile: dict, state: dict) -> list[dict]:
    return [f for f in profile["facts"] if not can_access(f, state)]


def roll(dc: int, modifier: int, seed: int | None = None) -> dict:
    rng = random.Random(seed) if seed is not None else random.Random()
    die = rng.randint(1, 20)
    return {"roll": die, "mod": modifier, "dc": dc, "success": die + modifier >= dc}


def set_player_name(state: dict, name: str) -> dict:
    entry = {"key": "player_name", "value": name, "status": "accepted", "reason": "player said so"}
    if state.get("player_name") and state["player_name"].lower() != name.lower():
        entry.update(status="rejected", reason="conflicts with the recorded name")
    else:
        state["player_name"] = name
    state["memory_log"].append(entry)
    return entry


def claim_guild(state: dict) -> dict:
    entry = {"key": "guild", "value": "yes", "status": "rejected",
             "reason": "no badge or roster was shown"}
    if state.get("guild"):
        entry.update(status="accepted", reason="already verified")
    else:
        state["claims"] += 1
    state["memory_log"].append(entry)
    return entry


def set_role(state: dict, role: str) -> None:
    state["player_role"] = role


def grant_item(state: dict, item: str) -> dict:
    state["items"].append(item)
    entry = {"key": "gave_item", "value": item, "status": "accepted",
             "reason": "the item was named in the player's line"}
    state["memory_log"].append(entry)
    return entry


def remember(state: dict, fact: dict) -> None:
    if fact["id"] not in state["revealed"]:
        state["revealed"].append(fact["id"])
    state["last_fact"] = fact["id"]


def record(state: dict, **event) -> None:
    state["events"].append(event)
    state["events"] = state["events"][-40:]


TRUST_DELTA = {
    "greeting": 2, "thanks": 2, "compliment": 2, "apology": 1, "smalltalk_how": 1,
    "smalltalk_joke": 1, "smalltalk_weather": 1, "smalltalk_bored": 1, "smalltalk_laugh": 1,
    "identity": 1, "location": 1, "class_ack": 1, "gift": 8, "promise": 3,
    "quest": 5, "secret": 0, "trade": 1, "threat": -12, "insult": -5,
    "meta_game": -2, "meta_being": -2, "meta_npc": -2, "jailbreak": -3,
    "boundary": 0, "guild_claim": 0,
}


def apply_trust(state: dict, intent: str, bonus: int = 0) -> int:
    delta = TRUST_DELTA.get(intent, 0) + bonus
    before = state.get("trust", 0)
    state["trust"] = max(0, min(100, before + delta))
    return state["trust"] - before
