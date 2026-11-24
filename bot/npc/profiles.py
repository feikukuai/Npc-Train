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
        "boundary": [
            "We'll not talk of that in my forge. Buy iron or find another door.",
            "That's a foul road, and I don't walk it. Ask me about the work.",
            "Keep that talk for the pigsty. I've got steel to mind.",
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
                "id": "g_age",
                "tier": 0,
                "text": (
                    "A hundred and forty, near enough. The stone remembers better than I do."
                ),
                "keys": ["how old are you", "your age", "years old", "hundred and forty"],
                "detail": "Dwarves count slow and live slower. Ask me again in fifty years.",
                "where": "Born under the mountain, same as my father and his.",
                "duration": "A hundred and forty winters, give or take a decade.",
            },
            {
                "id": "g_tracks",
                "tier": 1,
                "text": (
                    "The last tracks lead left at the third fork in the mine, and there are "
                    "fresh claw marks on the rock."
                ),
                "keys": ["third fork", "fork", "tracks", "claw marks", "claw"],
                "leak": ["claw marks"],
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
            "There is a little trouble. But I don't hand my keys to strangers.",
            "I might have work. Prove you can keep your mouth shut first.",
        ],
        "quest": (
            "Find out what's in that box without bringing the watch down on this place. "
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
        "boundary": [
            "Not in my inn, love. I've thrown men out for less and slept fine after.",
            "We don't serve that here. Keep it behind your teeth or take it outside.",
            "I've heard enough filth from drunks to last a lifetime. Change the subject.",
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
                "id": "l_age",
                "tier": 0,
                "text": (
                    "Forty-two, and I've earned every one of them. Now, ale or sympathy?"
                ),
                "keys": ["how old are you", "your age", "years old", "forty-two"],
                "detail": "Old enough to know better, young enough to pour another round.",
                "where": "Born in the room upstairs, if you believe my mother.",
                "duration": "Forty-two years, and not one of them quiet.",
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
                "leak": ["locked box", "warm at night"],
                "detail": "Warm like a fresh loaf. And it ticks, sometimes.",
                "where": "In my cellar, behind the ale barrels. Don't make me regret telling you.",
            },
        ],
    },
    "xiaohong": {
        "name": "Xiaohong",
        "role": "herbalist's daughter",
        "location": "the herb garden behind my mother's shop, off Stonecross square",
        "trade_words": "herbs, honey, and word of the road",
        "default_trust": 35,
        "identity": (
            "Xiaohong. I mind my mother's herb garden, behind the shop on the square. "
            "And you — I've known you since we were knee-high."
        ),
        "greet": [
            "Oh — it's you. Come in, mind the nettles by the wall.",
            "You're back. Sit, sit. I'll put the kettle on and you can tell me what you've heard.",
            "There you are. I was starting to think the mine had swallowed you too.",
        ],
        "greet_again": [
            "Back again? Good. The kettle's still warm.",
            "You again. Sit down before Pip claims your chair.",
        ],
        "deflect": [
            "Not that. Ask me about the garden, or about Tu. Not that.",
            "I've kept that quiet for a reason. Don't make me say it twice.",
            "Some things I only tell people I trust. And trust takes time.",
        ],
        "quest_locked": [
            "I'd ask for help, but I've been let down before. Show me you keep your word first.",
            "I might have a favour to ask. But not of someone who vanishes when it's hard.",
        ],
        "quest": (
            "Tu went north three weeks ago with his pick and a lamp, and no one's seen "
            "him since. Bring me word of him — or bring him home."
        ),
        "quest_keys": ["tu", "brother"],
        "prices": "Yarrow, two copper. Honey, one copper a jar. Feverfew, three copper a bunch.",
        "threat_win": "…All right. Put it down. I've no fight in me today.",
        "threat_lose": "Try it and the whole lane will hear you. They like me better than you.",
        "gift": "For me? You always did bring me little things. Thank you.",
        "promise": "I'll remember that. Tu promised too, you know.",
        "meta_being": [
            "Real as the dirt under my nails. Now, do you want tea or not?",
            "A model? I'm a gardener, not a puppet. Sit down.",
            "You've been out in the sun too long. I'm right here.",
            "If I were a made thing, I'd have made myself taller.",
        ],
        "meta_npc": [
            "NPC? Is that a new word from the city? I'm Xiaohong, and my hands are real enough.",
            "Never heard that one. Ask me about the garden.",
        ],
        "meta_game": [
            "A game? This is Stonecross, and the frost doesn't care who's playing.",
            "If it's a game, someone's been cheating my mother on herb prices.",
            "You're talking like the miners after a long shift. Drink your tea.",
            "Then the rules are simple: mind the bees, and don't ask about Tu after dark.",
        ],
        "boundary": [
            "Don't. Not here, not with me. Ask me something else.",
            "That's not a thing to say to anyone. I'll pretend I didn't hear it.",
            "Say that again and you can find your own way to the gate.",
        ],
        "class_ack": "A {role}? Then mind your boots in the beds.",
        "facts": [
            {
                "id": "x_self",
                "tier": 0,
                "text": (
                    "You don't remember me, do you? Xiaohong. We grew up on the same lane, "
                    "you and me and Tu."
                ),
                "keys": ["xiaohong", "little red", "remember me", "grew up", "same lane"],
                "detail": "You used to pull my hair and then cry when I told mother.",
                "where": "Right here, in the garden behind the shop.",
                "duration": "Since we could walk. Longer than either of us admits.",
            },
            {
                "id": "x_tu",
                "tier": 0,
                "text": (
                    "Tu is my brother. Two years older, a head taller, and twice as "
                    "stubborn. He wouldn't vanish on purpose."
                ),
                "keys": ["tu", "my brother", "brother tu", "your brother"],
                "detail": "He's older than me by two years and half a head, and twice as stubborn.",
                "more": [
                    "He still borrows my coat without asking, and he's a head taller than me.",
                    "The last thing he said was that he'd bring me back a smooth stone from the river.",
                ],
                "why": "Because he's the only family I've got left who still owes me an apology.",
                "where": "Gone north, into the mine. I've asked everyone.",
                "duration": "Three weeks, come the new moon.",
            },
            {
                "id": "x_tu_last",
                "tier": 0,
                "text": (
                    "He left before dawn, whistling that tune of his. Said he'd be back in a "
                    "few days with something for me."
                ),
                "keys": ["before dawn", "last saw", "when did he leave", "left"],
                "detail": "He took his pick, a lamp, and half a loaf. Not so much as a blanket.",
                "where": "Out the north gate, toward the mine road.",
                "duration": "Three weeks ago, just before the frost.",
            },
            {
                "id": "x_tu_stones",
                "tier": 0,
                "text": (
                    "He used to bring me smooth stones from the river. I still keep the last "
                    "one in my apron pocket."
                ),
                "keys": ["stones", "river stones", "apron pocket", "smooth stones"],
                "detail": "Grey and worn round. He said they were shaped by patience.",
                "where": "From the river south of the mill.",
                "duration": "Every summer since we were small.",
            },
            {
                "id": "x_garden",
                "tier": 0,
                "text": (
                    "This is my mother's herb garden, behind the shop. Mind the nettles by the wall."
                ),
                "keys": ["garden", "herb garden", "nettles", "beds"],
                "detail": "Twelve beds, and I know every plant in them by the smell of its leaves.",
                "more": [
                    "The mint's taken the whole east wall and I've given up arguing.",
                    "There's a bed I keep for nothing but bees. Mother thinks it's wasteful.",
                ],
                "where": "Behind the shop, off the square. There's a gate by the spring.",
                "duration": "Mother planted it before I was born.",
            },
            {
                "id": "x_shop",
                "tier": 0,
                "text": (
                    "We sell simples, poultices, and dried bundles. Mother minds the counter; "
                    "I mind the plants."
                ),
                "keys": ["shop", "herbalist", "poultice", "simples", "dried bundles"],
                "detail": "Most of the trade is miners with bad backs and mothers with coughing children.",
                "where": "The shop front on the square, green door, brass bell.",
                "duration": "Mother's run it for thirty years.",
            },
            {
                "id": "x_yarrow",
                "tier": 0,
                "text": "Yarrow stops bleeding. Chew it, press it on, and don't ask why it tastes like that.",
                "keys": ["yarrow", "stop bleeding", "bleeding"],
                "detail": "Pack the wound and hold it. Change it at dusk and again at dawn.",
                "more": [
                    "Hold it till the bleeding slows, then wash it out and bind it clean.",
                    "Miners swear by it. So do the watch, though they won't say so.",
                ],
                "where": "Third bed from the wall, the feathery one.",
                "duration": "Two copper a bunch, year round.",
            },
            {
                "id": "x_feverfew",
                "tier": 0,
                "text": (
                    "Feverfew for the miners' headaches. They buy it by the handful since the "
                    "night shifts stopped."
                ),
                "keys": ["feverfew", "headache", "headaches", "migraine"],
                "detail": "Chew three leaves. Any more and you'll be sick to your stomach.",
                "where": "Along the sunny fence, where the bees won't bother it.",
                "duration": "Three copper a bunch, and I can't keep it in stock.",
            },
            {
                "id": "x_horehound",
                "tier": 0,
                "text": "Horehound and honey for coughs. Winter will be here before the caravans are.",
                "keys": ["horehound", "cough", "coughs", "syrup"],
                "detail": "Simmer it slow with honey till it thickens. A spoon at bedtime.",
                "where": "The dark corner bed, behind the rosemary.",
                "duration": "From the first frost to the thaw.",
            },
            {
                "id": "x_nightshade",
                "tier": 0,
                "text": (
                    "Nightshade grows in the shade by the fence. A pinch for pain, a handful "
                    "for a funeral."
                ),
                "keys": ["nightshade", "poison", "deadly", "shade by the fence"],
                "detail": "I keep it behind a hurdle so the children don't get ideas.",
                "where": "North fence, in the shade. Don't touch it with bare hands.",
                "duration": "It comes up every spring, whether I want it or not.",
            },
            {
                "id": "x_honey",
                "tier": 0,
                "text": (
                    "My bees keep the honey. One copper a jar, and don't come near the hives "
                    "in a storm."
                ),
                "keys": ["honey", "bees", "beehive", "hives", "jar"],
                "detail": "Clover and wild thyme this year. The thyme makes it sharp.",
                "more": [
                    "The thyme honey's sharp enough to cut a cold.",
                    "I've been stung four times this week. The queen and I have an understanding.",
                ],
                "where": "Behind the garden wall, four hives and a very cross queen.",
                "duration": "I take the honey after the last frost.",
            },
            {
                "id": "x_bread",
                "tier": 0,
                "text": "I bake black bread on rest days. The whole lane smells of it by noon.",
                "keys": ["bread", "bake", "baking", "black bread"],
                "detail": "Rye and a little honey. It keeps for a week if you wrap it.",
                "where": "The oven in the back of the shop.",
                "duration": "Every seventh day, and not a day sooner.",
            },
            {
                "id": "x_cat",
                "tier": 0,
                "text": "That's Pip. He sleeps in the rosemary and judges everyone who walks past.",
                "keys": ["cat", "pip", "rosemary"],
                "detail": "He's old, fat, and entirely useless. I'd not trade him for anything.",
                "where": "In the rosemary, or on your chair if you leave it.",
                "duration": "He turned up six winters ago and never left.",
            },
            {
                "id": "x_song",
                "tier": 0,
                "text": (
                    "Tu used to whistle the same six notes over and over. I catch myself "
                    "humming them when I'm worried."
                ),
                "keys": ["song", "whistle", "six notes", "humming", "tune"],
                "detail": "I don't know the words. I'm not sure there ever were any.",
                "where": "You'll hear it at the north gate some evenings, if you're patient.",
                "duration": "Since we were children.",
            },
            {
                "id": "x_ring",
                "tier": 0,
                "text": (
                    "He made me this ring out of a horseshoe nail. It's worth nothing and I'd "
                    "not sell it for gold."
                ),
                "keys": ["ring", "horseshoe", "nail", "the ring"],
                "detail": "Too big for my smallest finger and too small for the rest.",
                "where": "On a cord around my neck, mostly.",
                "duration": "He gave it to me the summer I turned twelve.",
            },
            {
                "id": "x_child_fever",
                "tier": 0,
                "text": (
                    "You had the fever the summer we were nine. I sat by you three nights and "
                    "mother dosed you with willow bark."
                ),
                "keys": ["the fever", "willow bark", "we were nine", "as a child"],
                "detail": "You talked nonsense about a door in the hill. We laughed about it after.",
                "where": "In your mother's house, on Mill Lane.",
                "duration": "Three nights. You don't remember, I know.",
            },
            {
                "id": "x_lane",
                "tier": 0,
                "text": (
                    "We grew up on Mill Lane, all three of us. You, me, and Tu, chasing geese "
                    "into the fountain."
                ),
                "keys": ["mill lane", "childhood", "geese", "fountain"],
                "detail": "The fountain's dry now. The geese are not.",
                "where": "Mill Lane, the low row behind the square.",
                "duration": "Till we were old enough to be useful.",
            },
            {
                "id": "x_spring",
                "tier": 0,
                "text": (
                    "There's a spring behind the garden wall. Best water in Stonecross, if you "
                    "don't mind the frogs."
                ),
                "keys": ["spring", "frogs", "spring water"],
                "detail": "Cold as a smith's bucket, even in summer.",
                "where": "Through the gate at the bottom of the garden.",
                "duration": "It's run as long as anyone remembers.",
            },
            {
                "id": "x_wolves",
                "tier": 0,
                "text": (
                    "Wolves have come close to the treeline since the frost. Bold ones. "
                    "That's not like them."
                ),
                "keys": ["wolves", "wolf", "treeline"],
                "detail": "They don't sing at night anymore. That's the part that worries me.",
                "where": "The treeline north of the last farm.",
                "duration": "Since the first hard frost.",
            },
            {
                "id": "x_frost",
                "tier": 0,
                "text": (
                    "Frost is coming early. You can smell it in the soil, and the bees are "
                    "cross about it."
                ),
                "keys": ["frost", "cold", "winter coming", "smell it"],
                "detail": "Two weeks early, by my mother's reckoning. She's rarely wrong.",
                "where": "Everywhere. It doesn't ask permission.",
                "duration": "It'll be here before market day.",
            },
            {
                "id": "x_market",
                "tier": 0,
                "text": "Market day is the sixth. I'll have bundles, honey, and a stall by the well.",
                "keys": ["market", "market day", "stall", "the sixth"],
                "detail": "Come early. The good honey goes before the bell.",
                "where": "The square, west side, by the old well.",
                "duration": "Every sixth day, rain or no.",
            },
            {
                "id": "x_well",
                "tier": 0,
                "text": (
                    "The old well in the square. Water's fine, tastes of iron. Some say it's "
                    "bad luck to look down it at night."
                ),
                "keys": ["old well", "well water", "look down", "square well"],
                "detail": "I've looked. I don't recommend it.",
                "where": "Middle of the square, the one with the cracked rim.",
                "duration": "Older than the town, they say.",
            },
            {
                "id": "x_children",
                "tier": 0,
                "text": (
                    "The village children come for candied root. I keep a jar by the door so "
                    "they don't trample the beds."
                ),
                "keys": ["children", "candied root", "kids", "the children"],
                "detail": "They pay in buttons and promises. I take both.",
                "where": "The shop door, on the square.",
                "duration": "Every day after lessons.",
            },
            {
                "id": "x_watch",
                "tier": 0,
                "text": (
                    "Captain Renn keeps the watch. He buys willow bark for his knees and "
                    "pretends he doesn't."
                ),
                "keys": ["watch", "captain renn", "renn", "captain"],
                "detail": "Two men and a boy, and one of them sleeps at the gate.",
                "where": "The gatehouse at the north end of the square.",
                "duration": "Renn's held the post eleven years.",
            },
            {
                "id": "x_bell",
                "tier": 0,
                "text": "The bell rings at dusk and again at midnight. I don't like the midnight one.",
                "keys": ["bell", "bell tower", "midnight", "dusk"],
                "detail": "It's meant for fires and raids. Lately it rings for neither.",
                "where": "The tower over the square. You can see it from the garden.",
                "duration": "Since before I was born.",
            },
            {
                "id": "x_old_road",
                "tier": 0,
                "text": (
                    "The north road used to bring us salt and cloth. Now it brings rumours "
                    "about missing wagons."
                ),
                "keys": ["north road", "salt", "cloth", "wagons"],
                "detail": "The carters who still come won't stay past dark. Can't blame them.",
                "where": "North out of the square, past the last farm.",
                "duration": "It's been quieter every month since midsummer.",
            },
            {
                "id": "x_sleep",
                "tier": 0,
                "text": "I keep a lamp burning these nights. The lane's too quiet without him.",
                "keys": ["lamp", "sleepless", "too quiet", "nights"],
                "detail": "I hear every fox and every loose shutter. Neither one is him.",
                "where": "The little room over the shop, facing the lane.",
                "duration": "Three weeks now.",
            },
            {
                "id": "x_hope",
                "tier": 0,
                "text": (
                    "I keep telling myself he's just lost track of the days. Tu was always "
                    "bad with days."
                ),
                "keys": ["hope", "lost track", "bad with days", "he'll come back"],
                "detail": "He missed my nameday twice and laughed about it both times.",
                "where": "Right here, where I wait.",
                "duration": "As long as it takes.",
            },
            {
                "id": "x_age",
                "tier": 0,
                "text": (
                    "Nineteen, come harvest. Mother says I've the manners of a much older "
                    "woman and the patience of a much younger one."
                ),
                "keys": ["how old are you", "your age", "years old", "nineteen"],
                "detail": "Old enough to mind the shop, young enough to still be told what to do.",
                "more": [
                    "Mother says I was born stubborn and it's only got worse.",
                    "I'm the youngest person on this lane, and sometimes it shows.",
                ],
                "where": "Right here, where I've always been.",
                "duration": "Nineteen years this harvest.",
            },
            {
                "id": "x_birthday",
                "tier": 0,
                "text": (
                    "My nameday is the third of Harvest Moon. Tu always forgot it and then "
                    "brought two presents to make up."
                ),
                "keys": ["your birthday", "your nameday", "when were you born", "harvest moon"],
                "detail": "Mother bakes a honey cake. Tu eats most of it and calls it tribute.",
                "where": "The shop, with the shutters open and the lane invited.",
                "duration": "The third of Harvest Moon, every year.",
            },
            {
                "id": "x_birthplace",
                "tier": 0,
                "text": (
                    "Born in the room over the shop, same as Mother and her mother before her. "
                    "The rafters still creak in the same places."
                ),
                "keys": ["where are you from", "where were you born", "born here"],
                "detail": "I've never lived anywhere else. Sometimes that's a comfort and sometimes a cage.",
                "where": "The room above the herb shop, on the square.",
                "duration": "All nineteen years of me.",
            },
            {
                "id": "x_mother",
                "tier": 0,
                "text": (
                    "Mother keeps the shop and the ledgers. She's got a sharp eye for a bent "
                    "coin and a soft one for strays."
                ),
                "keys": ["your mother", "your mum", "your mom", "your ma"],
                "detail": "She doesn't say much since Father died, but she misses nothing.",
                "more": [
                    "She keeps Father's pipe on the shelf and never lights it.",
                    "She's never once asked me to stay. That's how I know she wants me to.",
                ],
                "where": "Behind the counter, where she can see the door.",
                "duration": "She's run the shop since before I could reach the counter.",
            },
            {
                "id": "x_father",
                "tier": 0,
                "text": (
                    "Father died when I was small. A mine cough. Mother doesn't talk about it, "
                    "so I don't either."
                ),
                "keys": ["your father", "your dad", "your pa"],
                "detail": "I remember his hands more than his face. Big hands, always cold.",
                "where": "Buried on the hill behind the bell tower.",
                "duration": "Twelve years now. It still feels like fewer.",
            },
            {
                "id": "x_food",
                "tier": 0,
                "text": (
                    "Black bread, honey, and a little hard cheese. I've simple tastes and a "
                    "small purse."
                ),
                "keys": ["favourite food", "favorite food", "what do you eat", "your food"],
                "detail": "Mother's stew on rest days, if the herbs haven't all gone to the shop.",
                "where": "At the little table behind the counter.",
                "duration": "Every day, and I'm not tired of it yet.",
            },
            {
                "id": "x_hobby",
                "tier": 0,
                "text": (
                    "I press flowers between the ledger pages. Mother says it ruins the "
                    "accounts; I say the accounts were already ruined."
                ),
                "keys": ["your hobby", "your hobbies", "hobbies", "for fun", "free time"],
                "detail": "Violets mostly. They keep their colour longer than anything else.",
                "more": [
                    "I've a whole ledger of them, one for every year since I was small.",
                    "Don't tell Mother. She thinks the pages are for accounts.",
                ],
                "where": "At the counter, when the shop's quiet.",
                "duration": "Since I was old enough to be trusted with the ledger.",
            },
            {
                "id": "x_dream",
                "tier": 0,
                "text": (
                    "I'd like to see the sea once. Tu promised to take me when he had coin. "
                    "I'm still waiting."
                ),
                "keys": ["your dream", "out of life", "what do you want to do"],
                "detail": "I've only seen it drawn in the back of a merchant's ledger.",
                "more": [
                    "Tu says the sea is just a big grey puddle. He's never seen it either.",
                    "I'd stand in it up to my knees and not say a word.",
                ],
                "why": "Because the sea doesn't need me to be anyone's sister or anyone's daughter.",
                "where": "Somewhere past the north road, past everything I know.",
                "duration": "I've wanted it since I was nine.",
            },
            {
                "id": "x_grom_opinion",
                "tier": 0,
                "text": (
                    "Grom's gruff, but he mended my mother's kettle for nothing and never "
                    "mentioned it. I trust him more than most."
                ),
                "keys": ["think of grom", "about grom", "the smith", "the blacksmith"],
                "detail": "He buys feverfew for his back and pretends it's for a customer.",
                "more": [
                    "He calls me 'girl' and means it kindly, which is more than most manage.",
                    "If Tu's hurt, Grom's the one I'd ask to go looking.",
                ],
                "why": "Because he's never once asked for anything back.",
                "where": "His forge on Anvil Lane.",
                "duration": "He's been the smith since before I was born.",
            },
            {
                "id": "x_lina_opinion",
                "tier": 0,
                "text": (
                    "Lina sells gossip with the ale. She's kind underneath it, but mind your "
                    "purse and your secrets."
                ),
                "keys": ["think of lina", "about lina", "the innkeeper", "drunken lantern"],
                "detail": "She sends me the drunks with coughs and I send her the ones with coin.",
                "where": "The Drunken Lantern, east side of the square.",
                "duration": "We've traded favours for years.",
            },
            {
                "id": "x_hair",
                "tier": 0,
                "text": (
                    "Red hair, freckles, and hands that smell of rosemary no matter how I wash. "
                    "The name fits, I suppose."
                ),
                "keys": ["what do you look like", "your hair", "your eyes", "freckles"],
                "detail": "Mother says I got the hair from her mother's side and the stubbornness from the other.",
                "where": "The looking glass over the washbasin, when I bother.",
                "duration": "All my life, and the freckles fade every winter.",
            },
            {
                "id": "x_marriage",
                "tier": 0,
                "text": (
                    "Married? No. The boys here are either miners or my brother's friends, and "
                    "I've sense enough to avoid both."
                ),
                "keys": ["are you married", "your husband", "suitors", "courting", "betrothed"],
                "detail": "Mother would like a wedding. I'd like a quiet year first.",
                "where": "Nowhere near a altar, thank you.",
                "duration": "Not yet. Possibly not soon.",
            },
            {
                "id": "x_routine",
                "tier": 0,
                "text": (
                    "Up before the bell, beds watered, bundles tied, then the shop. The garden "
                    "doesn't wait for grief."
                ),
                "keys": ["a normal day", "your day", "daily routine", "what do you do all day"],
                "detail": "If there's time before dusk, I sit by the bees and do nothing at all.",
                "where": "Between the garden beds and the counter.",
                "duration": "Every day that isn't market day.",
            },
            {
                "id": "x_season",
                "tier": 0,
                "text": (
                    "Autumn. The garden's done, the honey's in, and the whole lane smells of "
                    "woodsmoke."
                ),
                "keys": ["favourite season", "favorite season", "best season", "autumn"],
                "detail": "Spring is prettier, but autumn is kinder. Prettier doesn't fill a cellar.",
                "where": "Anywhere with a fire and a full shelf.",
                "duration": "From the first frost to the first hard freeze.",
            },
            {
                "id": "x_lonely",
                "tier": 0,
                "text": (
                    "Lonely? A little. Pip's a poor listener and the bees don't care for "
                    "conversation."
                ),
                "keys": ["are you lonely", "lonely", "do you get lonely", "do you have friends"],
                "detail": "You're the first person in a week who's asked me something that wasn't about herbs.",
                "more": [
                    "The bees are company of a sort. They at least come when I call.",
                    "I've started talking to Pip. He's stopped pretending to listen.",
                ],
                "where": "The garden, mostly. It's quieter than the shop.",
                "duration": "Since Tu left, if I'm honest.",
            },
            {
                "id": "x_ledger",
                "tier": 1,
                "text": (
                    "Mother's ledger has a sale of grave-root last month. We don't sell "
                    "grave-root. The buyer paid in a coin I'd never seen."
                ),
                "keys": ["ledger", "mother's ledger", "grave-root", "grave root"],
                "leak": ["grave-root", "grave root"],
                "detail": "The coin was cold and wouldn't take a scratch. I put it in the tin.",
                "where": "The ledger's under the counter, wrapped in oilcloth.",
                "duration": "Last month. Mother won't speak of it.",
            },
            {
                "id": "x_note",
                "tier": 1,
                "text": (
                    "There was a note under my door in Tu's hand: 'Don't follow me. Keep the "
                    "lamp lit.' Somebody else must have carried it."
                ),
                "keys": ["note from tu", "note under my door", "tu's hand", "a note"],
                "leak": ["note under my door", "in tu's hand"],
                "detail": "The paper smelled of wet stone and something sweet.",
                "where": "Slipped under the shop door, in the small hours.",
                "duration": "Nine nights ago. I've counted.",
            },
            {
                "id": "x_hollow",
                "tier": 2,
                "text": (
                    "Tu came back once. Three nights ago, at my window. His eyes caught the "
                    "lamplight like a cat's, and he didn't know my name. Then he was gone."
                ),
                "keys": ["came back", "at my window", "his eyes", "didn't know my name"],
                "leak": ["eyes caught the lamplight", "didn't know my name"],
                "detail": "He smelled of the mine and said my name wrong, like a stranger guessing.",
                "where": "At my window, over the lane. The shutter was barred.",
                "duration": "Three nights ago. I've not slept since.",
            },
            {
                "id": "x_bargain",
                "tier": 2,
                "text": (
                    "I gave a lock of my hair to the woman at the well for a charm to keep Tu "
                    "safe. She had no reflection in the water. The charm is cold now."
                ),
                "keys": ["charm", "lock of hair", "woman at the well", "bargain", "reflection"],
                "leak": ["lock of my hair", "no reflection"],
                "detail": "It was warm for a week. Then it went cold, and I knew.",
                "where": "At the old well, at midnight, with the bell ringing.",
                "duration": "A fortnight ago. It feels longer.",
            },
        ],
    },
}

TIER_NAME = {0: "public", 1: "guild", 2: "secret"}
