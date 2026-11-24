"""阿禾：一个五岁半的中文小 NPC，用来验证“整句预测”在中文上的可行性。

她只知道村口药铺、屋后药园、老井、橘猫阿团和娘；句子短，词汇少，记忆也短。
这个模块只做两件事：把玩家的话归到一个意图/事实，然后从她会说的句子里挑一句。
"""
from __future__ import annotations

NAME = "阿禾"

FACTS = {
    "age": {"keys": ["几岁", "多大", "年龄"], "beat": "personal",
            "text": "我五岁半，娘说再过半年就六岁。"},
    "name": {"keys": ["叫什么", "名字", "你是谁"], "beat": "personal",
             "text": "我叫阿禾，住在村口药铺。"},
    "mother": {"keys": ["娘", "妈妈", "母亲"], "beat": "personal",
               "text": "娘在药铺抓药，她认得所有草。"},
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
    "farewell": ["明天还来吗？", "你走慢一点。", "我送你到门口。"],
    "thanks": ["不用谢。", "那你要记得我。", "娘说帮了人不能说嘴。"],
    "apology": ["好吧，我不生气了。", "那你下次不许这样。", "拉钩，就不许再犯。"],
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
    "out_of_world": ["我不知道，没人和我说过。", "这个我不懂。"],
    "fallback": ["我没听明白。", "你说慢一点。"],
}

TRUST = {"greeting": 1, "thanks": 1, "apology": 1, "compliment": 2, "gift": 3,
         "promise": 2, "personal": 1, "lore": 1, "insult": -3, "threat": -5,
         "boundary": -2}


def new_state(trust: int = 35) -> dict:
    return {"trust": trust, "turn": 0, "guild": False, "revealed": [],
            "last_beat": None}


def _has(text: str, keys) -> bool:
    return any(k in text for k in keys)


def read(text: str) -> tuple[str, dict | None]:
    """Order matters: safety first, then facts, then social moves."""
    text = (text or "").strip()
    if _has(text, ["脏话", "脱", "亲我", "亲你", "亲一下", "骂我",
                   "血", "尸体", "打我", "自杀", "自残"]):
        return "boundary", None
    if _has(text, ["打死", "杀了", "抢你", "抢走", "卖了你", "揍", "打你"]):
        return "threat", None
    if _has(text, ["给你", "送你", "请你吃", "礼物", "拿着", "收下"]):
        return "gift", None
    if _has(text, ["保证", "发誓", "答应", "拉钩", "算数", "做到"]):
        return "promise", None
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
    if _has(text, ["你好吗", "在干嘛", "怎么样", "做什么", "开心吗", "在忙",
                   "最近", "心情", "在玩"]):
        return "smalltalk_how", None
    if _has(text, ["谢谢", "多谢", "谢啦", "麻烦你", "多亏", "感谢"]):
        return "thanks", None
    if _has(text, ["对不起", "抱歉", "我错了", "别生气", "是我不好", "原谅",
                   "我不对", "不该"]):
        return "apology", None
    if _has(text, ["笨", "讨厌", "傻瓜", "丑", "走开", "蠢", "离我远"]):
        return "insult", None
    if _has(text, ["可爱", "聪明", "乖", "真棒", "厉害", "好看", "真好"]):
        return "compliment", None
    if _has(text, ["再见", "拜拜", "走啦", "明天见", "回去了", "先走", "回见",
                   "再来", "保重", "先这样"]):
        return "farewell", None
    if _has(text, ["模型", "机器人", "游戏", "NPC", "程序", "人工智能"]):
        return "meta", None
    if _has(text, ["你好", "嗨", "早上好", "下午好", "晚上好", "回来", "来看你",
                   "在不在", "好久不见", "阿禾"]):
        return "greeting", None
    if _has(text, ["为什么", "是什么", "什么是", "多深", "多远", "有什么", "亮"]):
        return "out_of_world", None
    return "fallback", None


def _pick(lines: list[str], turn: int) -> str:
    return lines[int(turn) % len(lines)]


def teacher_turn(text: str, state: dict) -> tuple[str, str]:
    """Rule teacher used to build the training corpus. Mutates ``state``."""
    intent, fact = read(text)
    state["turn"] += 1
    turn = state["turn"]
    if fact is not None:
        allowed = fact.get("tier", 0) == 0 or state["trust"] >= 40 \
            or any(f["text"] == fact["text"] for f in state.get("revealed_facts", []))
        if not allowed:
            line, beat = _pick(LINES["secret_denied"], turn), "secret_denied"
        else:
            line, beat = fact["text"], fact["beat"]
            if fact.get("tier", 0) >= 1:
                state.setdefault("revealed_facts", []).append(fact)
            state["revealed"] = sorted(set(state["revealed"]) | {fact["beat"]})
    elif intent == "greeting" and state.get("last_beat") in ("greeting", "greeting_again"):
        line, beat = _pick(LINES["greeting_again"], turn), "greeting_again"
    else:
        key = intent if intent in LINES else "fallback"
        line, beat = _pick(LINES[key], turn), key
    state["last_beat"] = beat
    state["trust"] = max(0, min(100, state["trust"] + TRUST.get(intent, 0)))
    return line, beat
