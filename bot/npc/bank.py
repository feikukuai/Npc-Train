"""Offline-written surface variants.

The rule engine decides what a turn means and which beat to play; this module only
supplies a differently-worded way to say the same beat. Everything here was written
and validated offline, so a turn costs one dict lookup instead of a model call.

`bank.json` is the content. `pick()` never raises: a missing or broken file simply
falls back to `lines_en.py`.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

PATH = os.path.join(os.path.dirname(__file__), "bank.json")


@lru_cache(maxsize=1)
def load() -> dict:
    try:
        with open(PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def pick(profile: str, beat: str, turn: int = 0, salt: int = 0) -> dict | None:
    """Return {"en", "zh"} for this persona and beat, or None.

    Selection is deterministic per session (`salt`) and rotates with the turn, so two
    players see different orders while one player does not hear the same line twice
    in a row.
    """
    data = load()
    for bucket in (profile, "*"):
        entries = (data.get(bucket) or {}).get(beat)
        if isinstance(entries, list) and entries:
            index = (int(turn) + int(salt)) % len(entries)
            entry = entries[index]
            if isinstance(entry, dict) and entry.get("en"):
                return entry
    return None


__all__ = ["load", "pick"]
