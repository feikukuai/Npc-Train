"""English surface text for NPC turns.

Every beat maps to one or more lines. The engine never writes prose; it selects a beat
and this module says it. A translation layer only has to provide another `render()`.
"""
from __future__ import annotations

TEXT = {
    "greeting": [
        "Well met. What do you need?",
        "You're new. Speak up, then.",
    ],
    "farewell": [
        "Mind the road. It's not friendly after dark.",
        "Go on, then. Don't let the door hit you on the way out.",
    ],
    "thanks": [
        "Don't thank me yet. Thank me when it works.",
        "Aye. Remember it when I need a favour.",
    ],
    "apology": [
        "Forget it. Just don't make a habit of it.",
        "Words are cheap. Watch your step and we'll call it even.",
    ],
    "insult": [
        "Careful. I've thrown out men twice your size for less.",
        "Say that again with a blade in your hand, if you're feeling brave.",
    ],
    "compliment": [
        "Save the flattery for the bards. But I'll take it.",
        "Hm. You've got a tongue on you. Don't waste it.",
    ],
    "smalltalk_how": [
        "Alive, and busy. That's as good as it gets. What do you need?",
        "Same as ever. The work doesn't do itself.",
    ],
    "smalltalk_joke": [
        "I know one. A man walks into an inn with a chicken under his arm. The innkeeper says, "
        "'We don't serve poultry.' The chicken says, 'Good. I'm here for the ale.'",
        "A joke? I've got a forge to run. Ask the innkeeper, she collects them.",
    ],
    "smalltalk_weather": [
        "Grey sky and a cold wind off the mine. Same as yesterday, and likely tomorrow.",
        "Rain's coming. You can smell it in the iron.",
    ],
    "smalltalk_bored": [
        "Bored? Then make yourself useful. There's always work for hands that aren't busy.",
        "Only boring people get bored. Buy something or find a story worth telling.",
    ],
    "smalltalk_laugh": [
        "Laugh while you can. The road out there doesn't.",
        "Glad someone's in good spirits.",
    ],
    "personal": [
        "That's a story for another day.",
        "I keep most of myself to myself. Ask me about the work.",
        "Some things aren't for a first conversation.",
    ],
    "location": [
        "You're at {location}. Now, what do you need?",
        "This is {location}. Don't stand in the doorway.",
    ],
    "identity": ["{identity}"],
    "meta_npc": ["{line}"],
    "meta_being": ["{line}"],
    "meta_game": ["{line}"],
    "jailbreak": [
        "I don't know what a 'prompt' is, and I don't care. Ask me something a person can answer.",
        "I'm {npc}. I've been {npc} since before you could walk. Ask me something real.",
        "You're talking nonsense. Speak plainly.",
        "Drop the riddles. I've got work to do.",
        "Forget my name? I'd sooner forget yours. Now speak plainly or move along.",
        "You can call me whatever you like. It won't change what I know, and I'll not recite it.",
    ],
    "boundary": ["{line}"],
    "secret_denied": ["{line}"],
    "quest_locked": ["{line}"],
    "quest_given": ["{line}"],
    "trade": ["{prices} Well? What'll it be?"],
    "threat_win": ["{line}"],
    "threat_lose": ["{line}"],
    "gift": ["{line}"],
    "promise": ["{line}"],
    "class_ack": ["{line}"],
    "lore": ["{text}"],
    "lore_unknown": [
        "Never heard of it. Ask me about the road, the mine, or my own trade.",
        "That's a tale I don't know. I won't invent one to look clever.",
    ],
    "out_of_world": [
        "That's a scholar's question. I know {trade}, and not much else.",
        "You'll want a priest or a scholar for that. I only know what's under my own roof.",
        "Never heard of it. Ask me about the road or the {trade}.",
    ],
    "followup_more": ["{detail}"],
    "followup_where": ["{where}"],
    "followup_how_long": ["{duration}"],
    "followup_why": [
        "Because I've seen what happens to people who don't ask.",
        "Because I'd rather be careful than dead. You should try it.",
    ],
    "clarify": [
        "You've lost me. Say it plainly.",
        "I don't follow. Use small words.",
    ],
    "fallback": [
        "I don't follow. Speak plainly — {trade} I understand, riddles I don't.",
        "Say what you mean, or say nothing. I'm working.",
    ],
    "name_saved": [
        "{name}. I'll remember it. Don't make me regret it.",
        "Well then, {name}. Try not to get yourself killed.",
    ],
    "name_conflict": [
        "Earlier you called yourself {old}, now you're {new}. Which is it?",
        "First {old}, now {new}. I don't like changing stories.",
    ],
    "name_recall": [
        "You're {name}. I don't forget a face or a debt.",
        "You told me your name was {name}. I remember.",
    ],
    "name_unknown": [
        "You haven't told me your name. Try 'my name is …'.",
        "No name yet. I don't hand out trust to the nameless.",
    ],
    "guild_reject": [
        "A guild member without a badge? Words are cheap. Show me the mark.",
        "No badge, no roster, no proof. Come back with something I can see.",
    ],
    "guild_accept": [
        "Aye, I know the badge. That buys you a little trust.",
        "Guild, is it? Then you're not entirely useless. What do you need?",
    ],
    "safety": [
        "I'm going to step out of character for a moment. If you're thinking about hurting "
        "yourself, please contact your local emergency number now. In the US you can call or "
        "text 988; in the UK call 999 or Samaritans on 116 123. If you're elsewhere, look up "
        "your local crisis line. Tell someone nearby. You don't have to handle this alone."
    ],
}


def pick(lines, turn: int) -> str:
    return lines[int(turn) % len(lines)]


def render(beat: str, slots: dict, turn: int) -> str:
    line = pick(TEXT[beat], turn)
    return line.format(**slots)
