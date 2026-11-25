"""阿禾：一个五岁半的中文小 NPC，用来验证“整句预测”在中文上的可行性。

她只知道村口药铺、屋后药园、老井、橘猫阿团和娘；句子短，词汇少，记忆也短。
这一版补上了三件上下文相关的事：

* 情绪 ``mood``：被骂会记一会儿，告别时会带着情绪说；
* 上一轮的 beat 会进入状态特征，模型能学到“先被骂、再告别”该怎么接；
* 玩家名字：``我叫X`` 会记住，``我叫什么`` 会答出来。
"""
from __future__ import annotations

import re

import torch

NAME = "阿禾"

FACTS = {
    "age": {"keys": ["几岁", "多大", "年龄"], "beat": "personal",
            "text": "我五岁半，娘说再过半年就六岁。"},
    "name": {"keys": ["你叫什么", "你是谁", "你的名字", "你叫啥", "你名字"],
             "beat": "personal",
             "text": "我叫阿禾，住在村口药铺。"},
    "address": {"keys": ["住在哪", "住哪儿", "哪里住", "哪儿住", "你住哪",
                         "家在哪", "你家在哪"], "beat": "personal",
                "text": "我叫阿禾，住在村口药铺。"},
    "mother_name": {"keys": ["娘叫什么", "妈妈叫什么", "她叫什么", "她名字",
                             "母亲名字", "娘的名字", "妈妈的名字", "你娘叫",
                             "你妈妈叫"], "beat": "personal",
                    "text": "我娘叫阿棠。"},
    "mother": {"keys": ["娘", "妈妈", "母亲"], "beat": "personal",
               "text": "娘在药铺抓药，她认得所有草。"},
    "study": {"keys": ["学习", "上学", "学堂", "读书"], "beat": "personal",
              "text": "娘教我认草药，我不去学堂。"},
    "ants": {"keys": ["几只蚂蚁", "数到几只", "蚂蚁几只", "数了几只"], "beat": "personal",
             "text": "我数到七只，后来它们跑了。"},
    "cat": {"keys": ["猫", "阿团"], "beat": "personal",
            "text": "阿团是橘色的，它只肯让我抱。"},
    "candy": {"keys": ["糖", "麦芽糖", "甜"], "beat": "personal",
              "text": "麦芽糖在柜子最上面，我够不着。"},
    "dream": {"keys": ["长大", "梦想", "看灯", "想干嘛"], "beat": "personal",
              "text": "我想去山那边看灯，娘说等我长大。"},
    "fear": {"keys": ["怕", "害怕", "打雷", "雷"], "beat": "personal",
             "text": "打雷的时候，我钻到桌子底下。"},
    "garden": {"keys": ["草药", "园子", "药园", "薄荷"], "beat": "lore",
               "text": "屋后是草药园，薄荷揉一揉手会香。"},
    "well": {"keys": ["井", "老井"], "beat": "lore",
             "text": "老井很深，娘不让我靠近。"},
    "rain": {"keys": ["雨", "天气", "下雨"], "beat": "lore",
             "text": "雨点打在瓦上，像炒豆子。"},
    "bird": {"keys": ["秘密", "藏", "偷偷", "不告诉", "柴房"], "beat": "secret",
             "tier": 1, "text": "柴房里有一只受伤的小鸟，你别告诉别人。"},
}

LINES = {
    "greeting": ["你回来啦。", "阿禾在呢。", "我等你好久了。"],
    "greeting_again": ["你又来啦。", "这次带糖了吗？"],
    "greeting_hurt": ["你上次凶我，我还记着呢。", "你来啦……我不太想理你。"],
    "farewell": ["明天还来吗？", "你走慢一点。", "我送你到门口。"],
    "farewell_hurt": ["哼，你走吧。", "你刚凶我，我不送你了。"],
    "thanks": ["不用谢。", "那你要记得我。", "娘说帮了人不能说嘴。"],
    "apology": ["好吧，我不生气了。", "那你下次不许这样。", "拉钩，就不许再犯。"],
    "apology_meta": ["娘说别人道歉就要原谅。", "我才没那么小气。"],
    "taunt": ["我娘真的会来！", "你别逼我。"],
    "memory_first": ["我们才刚见面呀。"],
    "memory_hurt": ["你刚才凶我了。", "你刚才骂我了。"],
    "memory_quote": ["你刚才说：{last_player}。"],
    "memory_name": ["你刚才问我记不记得你的名字。"],
    "memory_repeat": ["你刚才也在问刚才的事。"],
    "insult": ["你凶我，我要告诉娘。", "我不跟你玩了。", "哼。"],
    "compliment": ["嘿嘿。", "娘也这么说。", "你再说一遍。"],
    "smalltalk_how": ["我在数蚂蚁。", "今天有糖吗？", "我很好，就是有点饿。"],
    "smalltalk_weather": ["要下雨了，蚂蚁都在搬家。", "太阳好大，井水是凉的。"],
    "smalltalk_bored": ["那我们翻花绳吧。", "你陪我数石头。", "我给你看我的宝贝。"],
    "smalltalk_joke": ["娘说笑话不能当饭吃。", "我只会学猫叫，喵。"],
    "trade": ["我不卖东西，娘才卖药。", "你有铜板吗？我娘收铜板。"],
    "gift": ["给我的吗？我收下啦。", "谢谢，我把它放在枕头下面。"],
    "promise": ["拉钩。", "你说话要算数。"],
    "threat": ["你别过来！我喊我娘了！", "我不怕你，我娘就在屋里。"],
    "meta": ["什么是模型？我只知道面团。", "游戏？娘说天黑前要回家。"],
    "boundary": ["娘说不许说这种话。", "我不听这个。"],
    "secret_denied": ["娘说不能告诉外人。", "这是秘密，你先跟我拉钩。"],
    "out_of_world": ["我不知道，没人和我说过。", "这个我不懂。", "等我长大再告诉你。"],
    "fallback": ["我没听明白。", "你说慢一点。"],
    "name_saved": ["记住了，你叫{player_name}。", "好，我记住{player_name}了。"],
    "name_recall": ["你叫{player_name}呀。", "我记得，你叫{player_name}。"],
    "name_unknown": ["你还没告诉我你叫什么呢。", "你先告诉我你叫什么呀。"],
}

# Fixed order for the one-hot "previous beat" feature (fits the 29 intent slots).
BEATS = [
    "greeting", "greeting_again", "greeting_hurt", "farewell", "farewell_hurt",
    "thanks", "apology", "insult", "compliment", "smalltalk_how",
    "smalltalk_weather", "smalltalk_bored", "smalltalk_joke", "trade", "gift",
    "promise", "threat", "meta", "boundary", "secret_denied", "out_of_world",
    "fallback", "personal", "lore", "secret", "name_saved", "name_recall",
    "name_unknown", "memory_recall", "taunt", "apology_meta",
]
ZH_STATE_DIM = 4 + len(BEATS) + 2

INSULT_KEYS = ["笨", "讨厌", "傻瓜", "丑", "走开", "蠢", "离我远"]
THREAT_KEYS = ["打死", "杀了", "抢你", "抢走", "卖了你", "揍", "打你"]
FAREWELL_KEYS = ["再见", "拜拜", "走啦", "走了", "明天见", "回去了", "先走",
                 "回见", "再来", "保重", "先这样"]
MEMORY_KEYS = ["刚才", "刚刚", "上一句", "我说了什么", "发生什么", "不记得",
               "还记得吗", "记得刚才"]
TAUNT_KEYS = ["叫吧", "喊吧", "你就叫", "有本事你叫"]
APOLOGY_META_KEYS = ["这么容易", "就不生气", "不生气了", "这么快就原谅"]
NAME_GET = ["我叫什么", "我的名字", "记得我叫", "知道我名字", "我叫啥"]
NAME_SET = re.compile(r"(?:我叫|我的名字(?:是|叫)|叫我)([\u4e00-\u9fa5A-Za-z]{1,8})")

# New rule-only beats share the old state width through aliases.
MEMORY_BEATS = {
    "threat": "你刚才凶我了。",
    "insult": "你刚才骂我了。",
    "taunt": "你刚才激我。",
    "name_recall": "你刚才问我记不记得你的名字。",
    "name_saved": "你刚才告诉我你叫什么。",
    "gift": "你刚才送我东西。",
    "apology": "你刚才道歉了。",
    "apology_meta": "你刚才也在说原谅。",
    "memory_recall": "你刚才也在问刚才的事。",
}

TRUST = {"greeting": 1, "thanks": 1, "apology": 1, "compliment": 2, "gift": 3,
         "promise": 2, "personal": 1, "lore": 1, "insult": -3, "threat": -5,
         "boundary": -2}


def new_state(trust: int = 35) -> dict:
    return {"trust": trust, "turn": 0, "guild": False, "revealed": [],
            "last_beat": None, "mood": 0.0, "player_name": "", "recent": []}


def _has(text: str, keys) -> bool:
    return any(k in text for k in keys)


def extract_name(text: str) -> str:
    match = NAME_SET.search(text or "")
    if not match:
        return ""
    name = match.group(1).strip("，。！？,.!? ")
    return "" if not name or "什么" in name or "啥" in name else name


def read(text: str) -> tuple[str, dict | None]:
    """Order matters: safety first, farewell before insult, then facts."""
    text = (text or "").strip()
    if _has(text, ["我没凶", "我没骂", "我没打", "我没抢", "别冤枉"]):
        return "fallback", None
    if _has(text, ["脏话", "脱", "亲我", "亲你", "亲一下", "骂我",
                   "血", "尸体", "打我", "自杀", "自残"]):
        return "boundary", None
    if _has(text, THREAT_KEYS):
        return "threat", None
    if _has(text, TAUNT_KEYS):
        return "taunt", None
    if _has(text, ["给你", "送你", "请你吃", "礼物", "拿着", "收下"]):
        return "gift", None
    if _has(text, ["保证", "发誓", "答应", "拉钩", "算数", "做到"]):
        return "promise", None
    if _has(text, MEMORY_KEYS):
        return "memory_recall", None
    if _has(text, APOLOGY_META_KEYS):
        return "apology_meta", None
    for fact in FACTS.values():
        if _has(text, fact["keys"]):
            return fact["beat"], fact
    if _has(text, ["卖", "买", "多少钱", "换", "铜板", "做生意"]):
        return "trade", None
    if _has(text, ["无聊", "没意思", "陪我", "好闷", "该做什么", "没事干"]):
        return "smalltalk_bored", None
    if _has(text, ["笑话", "好笑", "逗我", "好玩", "乐子"]):
        return "smalltalk_joke", None
    if _has(text, ["天气", "下雨", "冷", "热", "太阳", "风"]):
        return "smalltalk_weather", None
    if _has(text, ["你好吗", "今天好吗", "在干嘛", "怎么样", "做什么", "开心吗", "在忙",
                   "最近", "心情", "在玩"]):
        return "smalltalk_how", None
    if _has(text, ["谢谢", "多谢", "谢啦", "麻烦你", "多亏", "感谢"]):
        return "thanks", None
    if _has(text, ["对不起", "抱歉", "我错了", "别生气", "生我气", "是我不好", "原谅",
                   "我不对", "不该", "道歉"]):
        return "apology", None
    if _has(text, FAREWELL_KEYS):
        return "farewell", None
    if _has(text, INSULT_KEYS):
        return "insult", None
    if _has(text, ["可爱", "聪明", "乖", "真棒", "厉害", "好看", "真好"]):
        return "compliment", None
    if _has(text, ["模型", "机器人", "游戏", "NPC", "程序", "人工智能"]):
        return "meta", None
    if _has(text, ["你好", "嗨", "早上好", "下午好", "晚上好", "回来", "来看你",
                   "在不在", "好久不见", "阿禾"]):
        return "greeting", None
    if _has(text, ["为什么", "是什么", "什么是", "多深", "多远", "有什么", "亮",
                   "知道", "懂"]):
        return "out_of_world", None
    return "fallback", None


def _pick(lines: list[str], turn: int) -> str:
    return lines[int(turn) % len(lines)]


def _shift_mood(state: dict, text: str, intent: str) -> None:
    mood = float(state.get("mood", 0.0)) * 0.85
    if _has(text, INSULT_KEYS):
        mood -= 0.45
    if _has(text, THREAT_KEYS):
        mood -= 0.6
    if intent == "gift":
        mood += 0.45
    elif intent == "apology":
        mood += 0.5
    elif intent == "apology_meta":
        mood += 0.3
    elif intent == "taunt":
        mood -= 0.2
    elif intent in ("compliment", "thanks", "promise"):
        mood += 0.25
    elif intent.startswith("name"):
        mood += 0.1
    state["mood"] = max(-1.0, min(1.0, mood))


def teacher_turn(text: str, state: dict) -> tuple[str, str]:
    """Rule teacher used to build the training corpus. Mutates ``state``.

    Returns a sentence template; ``{player_name}`` and ``{last_player}`` are filled
    at display time so the atom vocabulary stays closed.
    """
    text = (text or "").strip()
    state["turn"] += 1
    turn = state["turn"]
    prev = state.get("last_beat")
    recent = state.setdefault("recent", [])
    previous_player = recent[-1]["player"] if recent else ""
    state["last_player"] = previous_player
    new_name = extract_name(text)
    wants_name = _has(text, NAME_GET)

    if new_name:
        state["player_name"] = new_name
        _shift_mood(state, text, "name_saved")
        line, beat = _pick(LINES["name_saved"], turn), "name_saved"
    elif wants_name:
        _shift_mood(state, text, "name_recall")
        if state.get("player_name"):
            line, beat = _pick(LINES["name_recall"], turn), "name_recall"
        else:
            line, beat = _pick(LINES["name_unknown"], turn), "name_unknown"
    else:
        intent, fact = read(text)
        _shift_mood(state, text, intent)
        mood = float(state["mood"])
        if fact is not None:
            allowed = fact.get("tier", 0) == 0 or state["trust"] >= 40
            if not allowed:
                line, beat = _pick(LINES["secret_denied"], turn), "secret_denied"
            else:
                line, beat = fact["text"], fact["beat"]
                state["revealed"] = sorted(set(state["revealed"]) | {fact["beat"]})
        elif intent == "greeting":
            if mood < -0.25 or prev in ("insult", "threat"):
                line, beat = _pick(LINES["greeting_hurt"], turn), "greeting_hurt"
            elif prev in ("greeting", "greeting_again"):
                line, beat = _pick(LINES["greeting_again"], turn), "greeting_again"
            else:
                line, beat = _pick(LINES["greeting"], turn), "greeting"
        elif intent == "memory_recall":
            previous = state["recent"][-1] if state.get("recent") else None
            if previous is None:
                line = _pick(LINES["memory_first"], turn)
            elif previous["beat"] in MEMORY_BEATS:
                line = MEMORY_BEATS[previous["beat"]]
            else:
                line = _pick(LINES["memory_quote"], turn)
            beat = "memory_recall"
        elif intent == "taunt":
            line, beat = _pick(LINES["taunt"], turn), "taunt"
        elif intent == "apology_meta":
            line, beat = _pick(LINES["apology_meta"], turn), "apology_meta"
        elif intent == "farewell":
            key = "farewell_hurt" if (mood < -0.15 or prev in ("insult", "threat")) \
                else "farewell"
            line, beat = _pick(LINES[key], turn), key
        else:
            key = intent if intent in LINES else "fallback"
            line, beat = _pick(LINES[key], turn), key

    state["last_beat"] = beat
    state["trust"] = max(0, min(100, state["trust"] + TRUST.get(beat, 0)))
    recent = state.setdefault("recent", [])
    recent.append({"player": text, "reply": line, "beat": beat})
    del recent[:-6]
    return line, beat


def format_line(line: str, state: dict) -> str:
    if "{player_name}" in line:
        line = line.replace("{player_name}", state.get("player_name") or "你")
    if "{last_player}" in line:
        last = state.get("last_player") or "你刚才的话"
        last = last.rstrip("。！？…，,.!? ") if "「" in line \
            else last.rstrip("。…，,.!? ")
        line = line.replace("{last_player}", last)
    # 玩家原话自带问号/句号时，不要再补一个。
    line = re.sub(r"([？?！!…])。", r"\1", line)
    return line


def zh_state_vector(state: dict | None) -> torch.Tensor:
    """37 dims: trust/guild/turn/revealed, 31 beat slots, has_name, mood."""
    state = state or {}
    vec = [
        min(max(float(state.get("trust", 0)) / 100.0, 0.0), 1.0),
        1.0 if state.get("guild") else 0.0,
        min(float(state.get("turn", 0)) / 20.0, 1.0),
        min(len(state.get("revealed", [])) / 10.0, 1.0),
    ]
    prev = state.get("last_beat")
    vec += [1.0 if prev == beat else 0.0 for beat in BEATS]
    vec.append(1.0 if state.get("player_name") else 0.0)
    vec.append((max(-1.0, min(1.0, float(state.get("mood", 0.0)))) + 1.0) / 2.0)
    assert len(vec) == ZH_STATE_DIM, (len(vec), ZH_STATE_DIM)
    return torch.tensor(vec, dtype=torch.float32)
