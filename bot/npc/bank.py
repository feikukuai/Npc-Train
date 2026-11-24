"""Surface variants: a fixed offline bank plus a bounded runtime overlay.

The rule engine decides what a turn means and which beat to play; this module only
supplies a differently-worded way to say the same beat. Base content lives in
`bank.json`; `add()` lets the background grower append validated lines at runtime.

`pick()` never raises: a missing or broken file simply falls back to `lines_en.py`.
"""
from __future__ import annotations

import json
import os
import threading
from functools import lru_cache

PATH = os.path.join(os.path.dirname(__file__), "bank.json")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MAX_PER_BEAT = _env_int("NPC_BANK_MAX_PER_BEAT", 16)
MAX_TOTAL = _env_int("NPC_BANK_MAX_TOTAL", 900)

_EXTRA: dict[str, dict[str, list[dict]]] = {}
_LOCK = threading.Lock()
_extra_total = 0


@lru_cache(maxsize=1)
def load() -> dict:
    try:
        with open(PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _base(bucket: str, beat: str) -> list[dict]:
    entries = (load().get(bucket) or {}).get(beat)
    return entries if isinstance(entries, list) else []


def lines(bucket: str, beat: str) -> list[dict]:
    """Base entries plus runtime extras for one bucket/beat."""
    with _LOCK:
        extra = list((_EXTRA.get(bucket) or {}).get(beat) or [])
    return _base(bucket, beat) + extra


def count(bucket: str, beat: str) -> int:
    return len(lines(bucket, beat))


def add(bucket: str, beat: str, en: str, zh: str = "", source: str = "lfm") -> str:
    """Append one validated variant. Returns "" on success, else a reason."""
    global _extra_total
    en = (en or "").strip()
    if not en:
        return "empty"
    low = en.lower()
    with _LOCK:
        bucket_map = _EXTRA.setdefault(bucket, {})
        extra = bucket_map.setdefault(beat, [])
        if len(extra) >= MAX_PER_BEAT or _extra_total >= MAX_TOTAL:
            return "cap"
        existing = [e.get("en", "").lower() for e in extra]
        existing += [e.get("en", "").lower() for e in _base(bucket, beat)]
        existing += [e.get("en", "").lower() for e in _base("*", beat)]
        if low in existing:
            return "duplicate"
        extra.append({"en": en, "zh": zh, "source": source})
        _extra_total += 1
    return ""


def update_zh(bucket: str, beat: str, en: str, zh: str) -> None:
    if not zh:
        return
    with _LOCK:
        for entry in (_EXTRA.get(bucket) or {}).get(beat) or []:
            if entry.get("en") == en:
                entry["zh"] = zh
                return


def pick(profile: str, beat: str, turn: int = 0, salt: int = 0) -> dict | None:
    """Return {"en", "zh"} for this persona and beat, or None.

    Selection is deterministic per session (`salt`) and rotates with the turn, so two
    players see different orders while one player does not hear the same line twice
    in a row.
    """
    for bucket in (profile, "*"):
        entries = lines(bucket, beat)
        if entries:
            entry = entries[(int(turn) + int(salt)) % len(entries)]
            if isinstance(entry, dict) and entry.get("en"):
                return entry
    return None


def stats() -> dict:
    with _LOCK:
        return {"extra": _extra_total, "max": MAX_TOTAL}


def reset_extras() -> None:
    """Drop the runtime overlay (tests, or a deliberate rebuild)."""
    global _extra_total
    with _LOCK:
        _EXTRA.clear()
        _extra_total = 0


def save_extras(path: str) -> bool:
    try:
        with _LOCK:
            snapshot = {b: {k: list(v) for k, v in beats.items()} for b, beats in _EXTRA.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, ensure_ascii=False, indent=1)
        return True
    except (OSError, ValueError):
        return False


def load_extras(path: str) -> int:
    """Best-effort restore of a previous runtime overlay."""
    global _extra_total
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return 0
    if not isinstance(data, dict):
        return 0
    loaded = 0
    with _LOCK:
        for bucket, beats in data.items():
            if not isinstance(beats, dict) or bucket == "_meta":
                continue
            for beat, entries in beats.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict) or not entry.get("en"):
                        continue
                    if _extra_total >= MAX_TOTAL:
                        break
                    target = _EXTRA.setdefault(bucket, {}).setdefault(beat, [])
                    target.append({"en": str(entry["en"]), "zh": str(entry.get("zh") or ""),
                                   "source": str(entry.get("source") or "restored")})
                    _extra_total += 1
                    loaded += 1
    return loaded


__all__ = ["load", "lines", "count", "add", "update_zh", "pick", "stats",
           "reset_extras", "save_extras", "load_extras", "MAX_PER_BEAT", "MAX_TOTAL"]
