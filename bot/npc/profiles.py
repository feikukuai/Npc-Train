"""NPC content: who they are, what they know, and what they will not say.

Content lives here, not in the engine. The engine decides *whether* a fact may be
spoken; this file only holds the words. A translation layer can replace this module
(and lines_en.py) without touching the reasoning.
"""

PROFILES = {
    "grom": {
        "name": "Grom Ironfoot",
        "role": "dwarf blacksmith",
        "location": "my forge on Anvil Lane, in Stonecross",
        "trade_words": "iron, the road, and the mine",
        "identity": (
            "Grom Ironfoot. Blacksmith. I've been beating iron at this anvil since before "
            "your beard came in. What do you need?"
        ),
        "greet": [
            "Halt. Buying iron, or just blocking my light?",
            "Mind the sparks. What brings you to my forge?",
            "You're new. State your business before the bellows cool.",
        ],
        "greet_again": [
            "Back again? Good. The forge is hot and my patience is short.",
            "Still here? Then make it worth my time.",
        ],
        "deflect": [
            "Some things aren't for loose talk. Show me you're worth trusting first.",
            "You're asking for a story I don't hand out to strangers.",
            "Ask me again when I've seen you keep a promise.",
        ],
        "quest_locked": [
            "There's work. But I don't hand my troubles to the first stranger who walks in.",
            "Aye, I could use a hand. First prove you won't sell me out to the watch.",
        ],
        "quest": (
            "Go to the third fork in the mine and bring me a shard of that rune-carved stone. "
            "Thirty gold, and I'll mend your armor for free."
        ),
        "quest_keys": ["gold"],
        "prices": "Sword, forty-five gold. Armor repair, twelve. Torch, two.",
        "threat_win": "…All right. You're meaner than you look. That still doesn't buy my secrets.",
        "threat_lose": "Draw that blade in my forge and you'll leave without it. And without a hand.",
        "gift": "Hm. Not many bring a smith gifts. I'll remember it.",
        "promise": "See that you do. I've buried men who broke promises to me.",
        "meta_being": [
            "Real enough to charge you for that sword. Now, what do you need?",
            "Model? I'm a blacksmith, not a clay figure. Speak plainly.",
            "You've been at the strong ale. I'm standing right here, aren't I?",
            "If I'm not real, stop wasting my time and let me get back to work.",
        ],
        "meta_npc": [
            "Never heard that word. I'm Grom, a smith. Now, do you need iron or not?",
            "You're making less sense by the minute. Ask me about the forge.",
        ],
        "meta_game": [
            "You've been drinking something stronger than my ale. This is Stonecross, not a puppet show.",
            "A game? Do I look like a dice cup to you? I've got a forge to run.",
            "I don't know what you're talking about, and I don't care to. Buy something or move along.",
            "If this is a joke, it's a poor one. Now, do you need iron or not?",
        ],
        "class_ack": "A {role}, eh? Keep your tricks away from the coals and we'll get along.",
        "facts": [
            {
                "id": "g_caravans",
                "tier": 0,
                "text": (
                    "Three caravans have gone missing on the north mine road this month. "
                    "Nobody's found a wheel, a horse, or a body."
                ),
                "keys": ["caravan", "caravans", "wagon", "wagons", "merchants", "missing"],
                "detail": "The last one was hauling silver ore. It never reached Stonecross.",
                "where": "On the north mine road, past the last farm.",
                "duration": "Three weeks since the first one. Two more since.",
            },
            {
                "id": "g_ore",
                "tier": 0,
                "text": (
                    "My ore shipments stopped, so I'm short on steel and shorter on patience."
                ),
                "keys": ["ore", "steel", "supply", "supplies", "shipment", "iron"],
                "detail": "Same road the caravans use. That's no coincidence to me.",
                "where": "From the north, along the mine road.",
            },
            {
                "id": "g_tracks",
                "tier": 1,
                "text": (
                    "The last tracks lead left at the third fork in the mine, and there are "
                    "fresh claw marks on the rock."
                ),
                "keys": ["third fork", "fork", "tracks", "claw marks", "claw", "mine road", "mine"],
                "leak": ["third fork", "claw marks"],
                "detail": "Claw marks as wide as my hand. Whatever made them wasn't a wolf.",
                "where": "Third fork of the mine, left branch, past the warning rope.",
            },
            {
                "id": "g_seal",
                "tier": 2,
                "text": (
                    "Deep in the mine there's a glowing rune door. An old seal. I saw it once, "
                    "and I've kept my mouth shut about it for thirty years."
                ),
                "keys": ["rune door", "rune", "seal", "glowing", "door", "deep"],
                "leak": ["rune door", "glowing rune", "old seal"],
                "detail": "The runes burned cold. I could feel them in my teeth for a week.",
                "where": "Deep in the mine. I won't draw you a map, and don't ask again.",
            },
        ],
    },
    "lina": {
        "name": "Lina Malt",
        "role": "innkeeper",
        "location": "the Drunken Lantern, on the east side of Stonecross square",
        "trade_words": "ale, beds, and gossip",
        "identity": (
            "Lina Malt. I run the Drunken Lantern. Ale's cold, stew's hot, and the beds "
            "only bite a little. What'll it be?"
        ),
        "greet": [
            "Welcome in! New face. Sit down before the door lets the cold in.",
            "Evening, love. Ale, stew, or gossip? All three cost you something.",
            "You look like you've been on the road. First cup's still full price.",
        ],
        "greet_again": [
            "Back for more? Good. My chairs were getting lonely.",
            "You again. Either you're thirsty or you're in trouble. Sit.",
        ],
        "deflect": [
            "Information's a commodity, love. What are you offering?",
            "I don't give away what I sell. Ask me nicer, or ask me richer.",
            "That's a story with a price on it. You haven't met it yet.",
        ],
        "quest_locked": [
            "There is a little trouble. But I don't hand my cellar keys to strangers.",
            "I might have work. Prove you can keep your mouth shut first.",
        ],
        "quest": (
            "Find out what's in that box without bringing the watch down on my cellar. "
            "Do that and I'll make it worth your while."
        ),
        "quest_keys": ["box"],
        "prices": "Ale, one silver. Stew, five copper. A bed, one gold.",
        "threat_win": "Easy now. Break my chairs and I'll have the watch break you.",
        "threat_lose": "Raise a hand in my inn and every drunk in this room will raise one back.",
        "gift": "Well, aren't you sweet. That buys you a smile, not a secret.",
        "promise": "Promises are cheap, love. Come back alive and we'll talk.",
        "meta_being": [
            "Pinch yourself, love. I'm as real as the rent.",
            "A model? Do I look like I'm made of wax? Drink your ale.",
            "You've had enough. Sit down before you start arguing with the chairs.",
            "If I'm a made thing, someone owes me thirty years of wages.",
        ],
        "meta_npc": [
            "NPC? Sounds like something a scholar made up. I'm Lina, and my ale is real enough.",
            "Never heard that word in my life. Sit down and drink something.",
        ],
        "meta_game": [
            "A game? You're in my inn, love, and the only game here is dice. Care for a round?",
            "Dreams and drink don't mix. Sit down before you fall down.",
            "Talk like that and folks will think you've been sleeping in the mushroom patch.",
            "If life's a game, someone's been cheating me on the wine prices. Now, what'll it be?",
        ],
        "class_ack": "A {role}! Then you'll want a room near the door, in case things get lively.",
        "facts": [
            {
                "id": "l_ale",
                "tier": 0,
                "text": (
                    "That caravan drank here the night before it vanished. Cheap ale, paid in "
                    "copper, no trouble at all."
                ),
                "keys": ["caravan", "ale", "drank", "tavern", "inn", "last night"],
                "detail": "Four of them. Quiet lot. One kept his hood up the whole time.",
                "where": "Right here, at the long table by the fire.",
                "duration": "The night before the first caravan disappeared.",
            },
            {
                "id": "l_knocking",
                "tier": 0,
                "text": (
                    "The miners won't take the night shift anymore. They say there's knocking "
                    "from the north shaft."
                ),
                "keys": ["knocking", "noise", "night shift", "miners", "north shaft", "shaft"],
                "detail": "Three knocks, a pause, three more. Like someone counting.",
                "where": "North shaft, deep enough that the town can't hear it.",
            },
            {
                "id": "l_cloak",
                "tier": 1,
                "text": (
                    "There was a driver in a grey cloak. Kept his hood up, didn't drink, and "
                    "left for the mine before dawn."
                ),
                "keys": ["grey cloak", "gray cloak", "cloak", "driver", "hood", "dawn"],
                "leak": ["grey cloak", "gray cloak", "hood up"],
                "detail": "He paid with a coin I didn't recognize. Still have it somewhere.",
                "where": "He walked north out of the square, toward the mine road.",
            },
            {
                "id": "l_box",
                "tier": 2,
                "text": (
                    "The grey-cloaked driver left a locked box in my cellar. Said he'd be back "
                    "in three days. That was a week ago, and the thing is warm at night."
                ),
                "keys": ["box", "cellar", "locked", "warm", "chest"],
                "leak": ["locked box", "cellar", "warm at night"],
                "detail": "Warm like a fresh loaf. And it ticks, sometimes.",
                "where": "In my cellar, behind the ale barrels. Don't make me regret telling you.",
            },
        ],
    },
}

TIER_NAME = {0: "public", 1: "guild", 2: "secret"}
