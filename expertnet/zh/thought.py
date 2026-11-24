"""Interpretable intermediate labels for the thinking head."""
from __future__ import annotations

THOUGHTS = [
    "fresh",
    "after_insult",
    "after_gift",
    "after_apology",
    "name_context",
    "memory_probe",
    "fact_probe",
    "safety",
    "social",
    "unknown",
]


def thought_of(previous_beat: str | None, player_name: str, target_beat: str) -> str:
    if target_beat in ("boundary", "threat"):
        return "safety"
    if target_beat == "memory_recall":
        return "memory_probe"
    if target_beat in ("personal", "lore", "secret", "secret_denied"):
        return "fact_probe"
    if previous_beat in ("insult", "threat", "taunt"):
        return "after_insult"
    if previous_beat == "gift":
        return "after_gift"
    if previous_beat in ("apology", "apology_meta"):
        return "after_apology"
    if player_name:
        return "name_context"
    if previous_beat is None:
        return "fresh"
    return "social"
