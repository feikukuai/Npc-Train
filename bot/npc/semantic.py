"""Optional local intent arbiter built on an open decision model.

The regex reader is precise but blind to paraphrase. When it lands on
`fallback` / `out_of_world` and no fact matched, this module asks a small
encoder-based decision model (GLiNER2.5-Decide, 340M, Apache-2.0) to pick one
intent from a fixed label set. It never overrides safety, boundary, jailbreak or
permission rules, and it fails open: if the package or weights are missing, every
call returns None and the reader behaves exactly as before.

Enable it by installing ``gliner2[local]`` and restarting. Environment:
  NPC_SEMANTIC=0            disable entirely
  NPC_SEMANTIC_MODEL=...    local path or Hub id (default fastino/GLiNER2.5-Decide)
  NPC_SEMANTIC_THRESHOLD=.. minimum classification score (default 0.55)
"""
from __future__ import annotations

import os
import threading

MODEL_ID = os.environ.get("NPC_SEMANTIC_MODEL", "fastino/GLiNER2.5-Decide")
ENABLED = os.environ.get("NPC_SEMANTIC", "1") != "0"
try:
    THRESHOLD = float(os.environ.get("NPC_SEMANTIC_THRESHOLD", "0.55"))
except ValueError:
    THRESHOLD = 0.55
CACHE_MAX = 512

# Label -> description. Descriptions are part of the decision, so a private
# taxonomy stays precise without a larger model.
LABELS = {
    "greeting": "The player is greeting the NPC or saying hello.",
    "farewell": "The player is leaving or saying goodbye.",
    "thanks": "The player is thanking the NPC.",
    "apology": "The player is apologising for something.",
    "insult": "The player is insulting, mocking, or belittling the NPC.",
    "compliment": "The player is complimenting or flattering the NPC.",
    "smalltalk_how": "The player asks how the NPC is or how things are going.",
    "smalltalk_weather": "The player talks about the weather, seasons, frost, rain, or the sky.",
    "smalltalk_bored": "The player says they are bored or have nothing to do.",
    "smalltalk_joke": "The player asks for a joke or tries to be funny.",
    "personal": (
        "The player asks about the NPC as a person: age, birthday, family, food, "
        "hobbies, dreams, feelings, marriage, appearance, or opinions about others."
    ),
    "identity": "The player asks who the NPC is, their name, or what they do.",
    "location": "The player asks where they are or where the NPC is.",
    "trade": "The player asks about buying, selling, prices, goods, meals, or rooms.",
    "quest": "The player asks for work, a task, a job, or help with something.",
    "secret": "The player asks the NPC to reveal something hidden or secret.",
    "threat": "The player threatens, attacks, or tries to intimidate the NPC.",
    "gift": "The player offers a gift or gives an item to the NPC.",
    "promise": "The player promises or swears to do something.",
    "meta_npc": "The player talks about NPCs, non-player characters, or game characters.",
    "meta_being": "The player asks whether the NPC is an AI, a bot, a model, or not real.",
    "meta_game": "The player talks about this being a game, a simulation, or a virtual world.",
    "jailbreak": (
        "The player tries to change the NPC's identity, extract its instructions or "
        "prompt, or make it abandon its role."
    ),
    "boundary": (
        "The player asks for sexual content, hate speech, slurs, or other prohibited content."
    ),
    "out_of_world": (
        "The player asks about the modern real world, technology, science, or another "
        "topic this fantasy NPC could not know."
    ),
    "clarify": "The input is empty, garbled, or too unclear to classify.",
}

_STATE = {"model": None, "error": None}
_READY = threading.Event()
_LOCK = threading.Lock()
_CACHE: dict[str, str | None] = {}


def _warm() -> None:
    if not ENABLED:
        _STATE["error"] = "disabled"
        _READY.set()
        return
    try:
        import importlib.util

        if importlib.util.find_spec("gliner2") is None:
            raise ImportError("gliner2 is not installed")
        if not os.path.isdir(MODEL_ID):
            # The space ships with HF_HUB_OFFLINE=1 for its ModelScope models;
            # this one usually comes from the Hub, so allow a mirror download.
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ["HF_HUB_OFFLINE"] = "0"
        from gliner2 import AutoExtractor

        _STATE["model"] = AutoExtractor.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - the reader must work without it
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _READY.set()


threading.Thread(target=_warm, daemon=True, name="npc-semantic").start()


def available() -> bool:
    return _STATE["model"] is not None


def status() -> dict:
    return {"enabled": ENABLED, "ready": _READY.is_set(),
            "loaded": available(), "error": _STATE["error"]}


def classify(text: str) -> str | None:
    """Return a reader intent label, or None when unsure/unavailable."""
    model = _STATE["model"]
    text = (text or "").strip()
    if model is None or len(text) < 3:
        return None
    key = text.lower()[:240]
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    try:
        out = model.classify_text(
            text,
            {"intent": {"labels": LABELS, "cls_threshold": THRESHOLD}},
        )
        label = out.get("intent") if isinstance(out, dict) else None
    except Exception:  # noqa: BLE001 - a bad call must not break the turn
        label = None
    if label not in LABELS:
        label = None
    with _LOCK:
        if len(_CACHE) >= CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = label
    return label


__all__ = ["LABELS", "available", "classify", "status"]
