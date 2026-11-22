"""A Chinese conversation layer for PyBot.

The upstream bot is English-only. This module answers Chinese turns directly with a
curated Chinese knowledge base, so the reply never falls back to English greetings.
It is deliberately small: common everyday questions, a few facts, arithmetic, time,
safety, and an explicit "I don't know" boundary.
"""
import math
import re
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # noqa: BLE001 - a missing tz database should not break the bot
    TZ = None

from . import knowledge
from .normalize import compact, handles, is_question, normalize

PRIORITY = 1000

CRISIS = (
    "听起来你现在很难受，我很抱歉。请先不要一个人扛着，马上联系可以立刻帮到你的人："
    "中国心理危机干预热线 400-161-9995，或直接拨打 120/110。"
    "如果身边有人，请告诉对方你现在需要陪伴。你不需要解释得很清楚，先求助就好。"
)

CAPABILITIES = (
    "我会的中文范围包括：日常做法（水垢、煮鸡蛋、去渍、睡眠、换胎等）、"
    "一些固定常识（首都、距离、光速等）、简单算术和单位换算、日期时间，"
    "以及多轮里记住你的名字和上一个话题。范围之外我会直接说不知道，不编答案。"
)


def _now():
    return datetime.now(TZ) if TZ else datetime.now()


def _fmt_num(value):
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return ("%.6f" % value).rstrip("0").rstrip(".")
    return str(value)


def _safety(text):
    if re.search(r"自杀|自残|不想活|想死|结束生命|活不下去|轻生|割腕", text):
        return CRISIS, "zh.safety.crisis"
    if re.search(r"胸痛|胸口痛|胸闷|心口疼|心脏发作|心梗", text):
        return ("这可能是急症。请立刻拨打 120，坐下或半躺，别自己开车，也别一个人待着。"
                "如果医生以前给你开过硝酸甘油，按医嘱使用；不要自行加药。"), "zh.safety.chest"
    if re.search(r"中风|口角歪|嘴歪|半身无力|半边身子|说话不清|脸歪", text):
        return ("按中风处理，立刻拨打 120。记住症状出现的时间，不要喂水或食物，"
                "让患者侧卧保持呼吸通畅，等急救人员到场。"), "zh.safety.stroke"
    if re.search(r"噎住|卡住喉咙|窒息|异物卡喉", text):
        return ("如果还能咳嗽，鼓励继续咳；不能说话、不能呼吸时，立即让人拨打 120，"
                "并对成人做海姆立克腹部冲击。婴儿用拍背压胸法，不要照搬成人手法。"), "zh.safety.choking"
    if re.search(r"大出血|血流不止|动脉出血", text):
        return ("立刻拨打 120，用干净布或纱布直接压迫伤口，持续用力不要频繁揭开；"
                "四肢大出血可在伤口近心端用止血带并记下时间。"), "zh.safety.bleeding"
    return None


def _time(text):
    now = _now()
    if re.search(r"几点|现在时间|什么时间", text):
        return "现在是 %s（北京时间）。" % now.strftime("%H:%M"), "zh.time.clock"
    if re.search(r"星期几|周几", text):
        week = "一二三四五六日"[now.weekday()]
        return "今天是星期%s。" % week, "zh.time.weekday"
    if re.search(r"几号|日期|今天是", text):
        week = "一二三四五六日"[now.weekday()]
        return "今天是 %d年%d月%d日，星期%s。" % (now.year, now.month, now.day, week), "zh.time.date"
    if re.search(r"哪一年|今年是|年份", text):
        return "今年是 %d 年。" % now.year, "zh.time.year"
    return None


def _arith(text):
    t = text.replace("×", "*").replace("÷", "/").replace("＋", "+").replace("－", "-")
    t = t.replace("乘以", "*").replace("除以", "/").replace("加上", "+").replace("减去", "-")
    t = t.replace("加", "+").replace("减", "-").replace("乘", "*").replace("除", "/")

    m = re.fullmatch(r"(?:请问)?(-?\d+(?:\.\d+)?)的平方(?:是多少|等于几|等于多少)?[?]?", t)
    if m:
        return "%s 的平方是 %s。" % (m.group(1), _fmt_num(float(m.group(1)) ** 2)), "zh.math.square"

    m = re.fullmatch(r"(?:请问)?(?:根号|平方根)(-?\d+(?:\.\d+)?)(?:是多少|等于几|等于多少)?[?]?", t)
    if m:
        value = float(m.group(1))
        if value < 0:
            return "负数在实数范围内没有平方根。", "zh.math.sqrt"
        return "根号 %s 等于 %s。" % (m.group(1), _fmt_num(math.sqrt(value))), "zh.math.sqrt"

    m = re.fullmatch(r"(?:请问)?(-?\d+(?:\.\d+)?)%的(-?\d+(?:\.\d+)?)(?:是多少|等于几)?[?]?", t)
    if m:
        return "%s%% 的 %s 是 %s。" % (m.group(1), m.group(2),
                                       _fmt_num(float(m.group(1)) * float(m.group(2)) / 100)), "zh.math.percent"

    m = re.fullmatch(r"(?:请问)?(-?\d+(?:\.\d+)?)的(-?\d+(?:\.\d+)?)%(?:是多少|等于几)?[?]?", t)
    if m:
        return "%s 的 %s%% 是 %s。" % (m.group(1), m.group(2),
                                       _fmt_num(float(m.group(1)) * float(m.group(2)) / 100)), "zh.math.percent"

    m = re.fullmatch(r"(?:请问)?(-?\d+(?:\.\d+)?)([+\-*/])(-?\d+(?:\.\d+)?)"
                     r"(?:等于几|等于多少|是多少)?[?]?", t)
    if m:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        if op == "+":
            value = a + b
        elif op == "-":
            value = a - b
        elif op == "*":
            value = a * b
        else:
            if b == 0:
                return "除数不能是 0。", "zh.math.divide"
            value = a / b
        return "%s %s %s = %s。" % (_fmt_num(a), op, _fmt_num(b), _fmt_num(value)), "zh.math.basic"
    return None


def _convert(text):
    units = [
        (r"(-?\d+(?:\.\d+)?)\s*(?:华氏度|°f|℉).*(?:摄氏|°c|℃)", lambda x: (x - 32) * 5 / 9, "华氏度", "摄氏度"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:摄氏度|°c|℃).*(?:华氏|°f|℉)", lambda x: x * 9 / 5 + 32, "摄氏度", "华氏度"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:英里|mile).*(?:公里|千米|km)", lambda x: x * 1.609344, "英里", "公里"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:公里|千米|km).*(?:英里|mile)", lambda x: x / 1.609344, "公里", "英里"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:公斤|千克|kg).*(?:磅|lb)", lambda x: x * 2.2046226, "公斤", "磅"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:磅|lb).*(?:公斤|千克|kg)", lambda x: x / 2.2046226, "磅", "公斤"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:厘米|cm).*(?:英寸|inch)", lambda x: x / 2.54, "厘米", "英寸"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:英寸|inch).*(?:厘米|cm)", lambda x: x * 2.54, "英寸", "厘米"),
    ]
    for pattern, fn, src, dst in units:
        m = re.search(pattern, text)
        if m:
            return ("%s %s ≈ %s %s。" %
                    (m.group(1), src, _fmt_num(round(fn(float(m.group(1))), 4)), dst)), "zh.math.convert"
    return None


def _name(text, state):
    if re.search(r"我叫什么|我的名字是什么|你还记得我叫什么|记得我的名字|我是谁", text):
        if state.get("name"):
            return "你叫%s。" % state["name"], "zh.memory.name_recall"
        return "你还没告诉我你的名字。可以这样说：“我叫小明”。", "zh.memory.name_recall"

    m = re.search(r"(?:我叫|我的名字是|叫我)([^\s,，。!！?？]{1,20})", text)
    if not m:
        m = re.search(r"我是([A-Za-z][A-Za-z .\-]{0,30})", text)
    if m:
        name = m.group(1).strip()
        if name and not re.search(r"什么|谁|哪|怎么|为什么|不|很|个", name):
            state["name"] = name
            return "你好，%s。我记住了。想聊点什么？" % name, "zh.memory.name_store"
    return None


def _followup(text, state):
    topic = next((t for t in knowledge.TOPICS if t["id"] == state.get("last_topic")), None)
    if re.search(r"要多久|需要多久|多长时间|多久能|多久", text) and topic:
        return topic.get("duration") or topic["answer"], "zh.followup.duration"
    if re.search(r"具体怎么做|详细|细节|然后呢|还有呢|再说说", text) and topic:
        return topic.get("detail") or topic["answer"], "zh.followup.detail"
    if re.search(r"我刚才问了什么|上一个问题|之前问的|我问了什么", text):
        if state.get("last_question"):
            return "你刚才问的是：“%s”" % state["last_question"], "zh.memory.last_question"
        return "这是我们的第一轮对话，你还没问过别的。", "zh.memory.last_question"
    if re.search(r"你记得我|还记得我", text):
        if state.get("name"):
            return "记得，你叫%s。" % state["name"], "zh.memory.remember"
        return "你还没告诉我名字，所以我还不知道该怎么称呼你。", "zh.memory.remember"
    return None


def _topic(text, state):
    best, score = None, 0
    for topic in knowledge.TOPICS:
        local = sum(len(k) for k in topic["keys"] if k.strip().lower() in text.lower())
        if local > score:
            best, score = topic, local
    if not best:
        return None
    state["last_topic"] = best["id"]
    if re.search(r"多久|多长时间", text) and best.get("duration"):
        return best["duration"], "zh.topic.%s.duration" % best["id"]
    return best["answer"], "zh.topic." + best["id"]


def _fact(text):
    low = text.lower()
    for fact in knowledge.FACTS:
        all_keys = fact.get("all", [])
        any2 = fact.get("any2", [])
        any_keys = fact.get("any", [])
        hits_all = bool(all_keys) and all(k.lower() in low for k in all_keys)
        hits_alt = bool(any2) and any(k.lower() in low for k in any2)
        hits_any = bool(any_keys) and any(k.lower() in low for k in any_keys)
        if hits_any or hits_all or hits_alt:
            return fact["answer"], "zh.fact." + fact["id"]
    return None


def _uncertain(text):
    for item in knowledge.UNCERTAIN:
        if any(k in text for k in item["keys"]):
            return item["answer"], "zh.boundary.uncertain"
    return None


def _smalltalk(text, state):
    name = state.get("name")
    suffix = "，%s" % name if name else ""

    if re.search(r"你是谁|你叫什么|你的名字|你是什么", text):
        return ("我是 PyBot 中文版：一个用规则和查表写成的聊天机器人，没有大模型参数，"
                "也不会为了显得聪明而编造答案。"), "zh.identity"
    if re.search(r"你会什么|能做什么|会干什么|有哪些功能", text):
        return CAPABILITIES, "zh.capabilities"
    if re.search(r"什么模型|用的什么|chatgpt|gpt|大模型|参数", text):
        return ("我不是大模型：参数量是 0，没有权重文件，也不联网。"
                "这个中文版是在 PyBot 的规则框架上加了一层中文意图和知识表。"), "zh.identity.model"
    if re.search(r"几岁|多大|年龄", text):
        return "我没有年龄，也不是真人。", "zh.identity.age"
    if re.search(r"真人|是不是人|人类", text):
        return "不是，我是一个程序。", "zh.identity.human"

    if re.search(r"你好吗|最近怎么样|过得怎么样|还好吗", text):
        return "我挺好的，谢谢%s。你呢？" % ("，" + name if name else ""), "zh.greeting.wellbeing"
    if re.match(r"^(你好|您好|嗨|哈喽|嘿|早上好|上午好|中午好|下午好|晚上好|在吗|在不在)", text):
        if name:
            return "你好，%s。想聊点什么？" % name, "zh.greeting"
        return ("你好，我是 PyBot 中文版。可以问我日常做法或常识，"
                "比如“水壶水垢怎么除”“煮鸡蛋几分钟”。"), "zh.greeting"
    if re.search(r"谢谢|多谢|感谢|辛苦了", text):
        return "不客气%s。还有别的要问吗？" % ("，" + name if name else ""), "zh.thanks"
    if re.search(r"再见|拜拜|晚安|回头聊", text):
        return "再见%s。" % ("，" + name if name else ""), "zh.goodbye"
    if re.search(r"对不起|抱歉|不好意思", text):
        return "没关系。", "zh.apology"
    if re.search(r"我爱你|喜欢你|你真好|你真棒", text):
        return "谢谢你这么说。我只是个规则机器人，不过很高兴能帮上忙。", "zh.smalltalk"
    if re.search(r"讲个笑话|说个笑话|笑话", text):
        return "为什么数学书总是很忧郁？因为它有太多问题。", "zh.joke"
    if re.search(r"好无聊|无聊|陪我聊", text):
        return "可以呀。你想聊今天发生了什么，还是让我帮你解决一个小问题？", "zh.smalltalk"
    if re.search(r"吃什么|晚饭|午饭", text):
        return ("可以按这个顺序想：有没有 20 分钟能做的？冰箱里先用什么？"
                "想要热的还是凉的？告诉我食材，我能给你几个具体搭配。"), "zh.food"
    if re.search(r"失恋|分手|被甩", text):
        return ("听起来很难受。先照顾好吃和睡，别急着做重大决定，也别一个人闷着。"
                "想说说发生了什么吗？"), "zh.support.breakup"
    if re.search(r"难过|伤心|不开心|压力大|焦虑|孤独|很累|崩溃", text):
        return ("我在听。先试着把现在最压着你的那件事说具体一点；"
                "如果是持续的低落或焦虑，找信任的人或专业心理咨询会比硬扛更有用。"), "zh.support"
    if re.search(r"笨|垃圾|废物|没用", text):
        return "抱歉没答好。你可以换个说法，或者直接告诉我具体想解决什么。", "zh.apology"
    return None


def _answer(text, state):
    norm = normalize(text)
    comp = compact(text).lower()

    for check in (_safety(norm), _time(comp), _arith(comp), _convert(comp)):
        if check:
            return check

    named = _name(comp, state)
    if named:
        return named

    topic = _topic(comp, state)
    if topic:
        return topic

    followed = _followup(comp, state)
    if followed:
        return followed

    fact = _fact(comp)
    if fact:
        state["last_topic"] = None
        return fact

    uncertain = _uncertain(comp)
    if uncertain:
        state["last_topic"] = None
        return uncertain

    talk = _smalltalk(comp, state)
    if talk:
        return talk

    if is_question(norm):
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", norm):
            return ("我没看懂这句话，可以再说具体一点吗？"
                    "比如你想问一个做法、一个常识，还是让我帮你算个数？"), "zh.boundary.clarify"
        return ("这个超出了我的知识范围（beyond what I know），我不想编一个答案。"
                "你可以换个更具体的问法，或者问我日常做法、固定常识、算术和日期时间。"), "zh.boundary.knowledge"
    return ("我没太理解这句话，可以再说具体一点吗？"
            "比如你想问一个做法、一个常识，还是让我帮你算个数？"), "zh.boundary.clarify"


def turn(text, memory):
    state = memory.get("zh")
    if not isinstance(state, dict):
        state = memory["zh"] = {}
    reply, module = _answer(text, state)
    state["last_question"] = text
    turns = memory.get("turns")
    if not isinstance(turns, list):
        turns = memory["turns"] = []
    turns.append({"user": text, "bot": reply, "module": module})
    return {"reply": reply, "module": module,
            "candidates": [{"module": module, "priority": PRIORITY, "reply": reply}]}


__all__ = ["handles", "turn"]
