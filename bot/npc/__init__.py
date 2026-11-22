"""English NPC layer built on PyBot's reader and rule engine.

Enable it per conversation by calling `activate(memory, "grom")`. While active,
`bot.runtime.turn()` routes lines here before the general modules, so the NPC keeps its
persona, permissions and memory across turns. Without `activate`, PyBot behaves exactly
as before.
"""
from .engine import activate, handles, set_renderer, turn

__all__ = ["activate", "handles", "set_renderer", "turn"]
