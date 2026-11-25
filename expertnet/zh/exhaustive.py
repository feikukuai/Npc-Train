"""穷尽式句表：有限词典 × 简单句式，全量组合。

不追求复杂句法，只保证：
* 每个句子都能由固定词典解释；
* 句子表是封闭的，模型输出只能是表里的 ID；
* 以后接润色/扩写模型时，输入就是这些“骨架句”。

    python -m expertnet.zh.exhaustive --out expertnet/zh/exhaustive.jsonl
    python -m expertnet.zh.exhaustive --stats
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "exhaustive.jsonl")
ATOMS = os.path.join(HERE, "atoms_zh.json")

# 阿禾的封闭世界。名词/动词/形容词都是有意的“小词表”。
SUBJ = ["我", "你", "娘", "阿团", "王伯", "李婶", "客人"]
VERB = [
    "吃", "喝", "拿", "抱", "数", "看", "找", "摘", "洗", "晒",
    "收", "放", "藏", "给", "送", "买", "卖", "煮", "切", "闻",
    "摸", "追", "喂", "叫", "等", "陪", "帮", "教", "学", "怕",
    "喜欢", "讨厌", "问", "告诉", "借", "还", "丢", "捡", "画", "写",
]
OBJ = [
    "糖", "麦芽糖", "薄荷", "甘草", "艾草", "陈皮", "药", "药罐", "药匾",
    "石头", "花绳", "小鸟", "蚂蚁", "井水", "瓦片", "铜板", "竹篮",
    "小木匣", "凳子", "柜台", "门", "窗", "灯", "饼", "粥", "馒头",
    "糕", "花生", "鸡蛋", "菜", "花", "草", "叶子", "树枝", "泥巴",
    "水", "雨", "云", "太阳", "月亮", "星星", "名字", "秘密", "故事",
    "歌", "画", "字", "鞋", "衣服", "帽子", "窝",
]
ADJ = [
    "甜", "香", "苦", "凉", "热", "冷", "大", "小", "多", "少",
    "快", "慢", "高", "低", "远", "近", "亮", "暗", "干", "湿",
    "新", "旧", "胖", "瘦", "圆", "方", "长", "短", "硬", "软",
    "干净", "脏", "好看", "难吃", "好玩", "可怕", "可爱", "聪明",
    "笨", "乖", "安静", "吵", "满", "空", "疼", "痒", "累", "饿",
    "渴", "困", "开心", "难过", "生气", "害怕",
]
PLACE = [
    "药铺", "药园", "老井边", "柴房", "门口", "柜台后",
    "屋后", "井台", "门槛上", "桌子底下", "墙角", "篱笆边",
]
ADV = ["很", "有点", "太", "真"]

# 组合是穷尽的，但“吃娘/卖阿团”这类人不作为食物/商品的搭配直接剔除。
ANIMATE = {"娘", "阿团", "王伯", "李婶", "客人"}
BODY = {"手", "脸", "头", "脚", "尾巴", "毛"}
INGEST = {"吃", "喝", "煮", "切", "尝", "咬", "舔"}


def allowed(row: dict) -> bool:
    verb, obj = row.get("verb"), row.get("obj")
    if verb in INGEST and obj in ANIMATE | BODY:
        return False
    if verb in {"卖", "丢"} and obj in ANIMATE | BODY:
        return False
    return True

PATTERNS = [
    ("svo", "{s}{v}{o}。", 1),
    ("svo_neg", "{s}不{v}{o}。", -1),
    ("ba", "{s}把{o}{v}了。", 1),
    ("place_svo", "{s}在{p}{v}{o}。", 1),
    ("svc", "{s}{adv}{a}。", 1),
    ("o_is_a", "{o}是{a}的。", 1),
    ("s_de_o_a", "{s}的{o}很{a}。", 1),
    ("s_think_o_a", "{s}觉得{o}很{a}。", 1),
    ("place_has_o", "{p}有{o}。", 1),
    ("want_svo", "{s}想{v}{o}。", 1),
    ("can_svo", "{s}会{v}{o}。", 1),
    ("like_svo", "{s}喜欢{v}{o}。", 1),
    ("no_svo", "{s}没有{v}{o}。", -1),
    ("le_svo", "{s}{v}了{o}。", 1),
    ("s_place", "{s}在{p}。", 1),
    ("o_place", "{o}在{p}。", 1),
]


def fixed_rows():
    """事实、名字、安全边界等固定句也必须锁在表里。"""
    if not os.path.exists(ATOMS):
        return
    from .persona import FACTS, LINES

    beats: dict[str, str] = {}
    for fact in FACTS.values():
        beats.setdefault(fact["text"], "personal")
    for beat, lines in LINES.items():
        for line in lines:
            beats.setdefault(line, beat)
    with open(ATOMS, encoding="utf-8") as fh:
        atoms = json.load(fh).get("atoms", [])
    seen = set()
    for text in atoms:
        seen.add(text)
        yield {"text": text, "pattern": "fixed", "polarity": 0,
               "beat": beats.get(text, "fixed")}
    for text, beat in beats.items():
        if text not in seen:
            seen.add(text)
            yield {"text": text, "pattern": "fixed", "polarity": 0,
                   "beat": beat}


def fixed_bank_rows(per_beat: int = 265, seed: int = 13):
    """从整句库给每个 beat 精选一批自然台词，补进固定句。

    组合句只能表达“谁对谁做什么”，问候、道歉、回忆这类社交句必须直接落在
    句表里，否则模型只能把“你好”硬映射成“你的药很生气”。
    用固定步长横跨整个 beat，保证不同句法框架/话题都被抽到。
    """
    if per_beat <= 0:
        return
    from .bank import get_bank

    bank = get_bank()
    rng = random.Random(seed)
    for beat, ids in sorted(bank.by_beat.items()):
        pool = [i for i in ids if 4 <= len(bank.texts[i]) <= 26]
        rng.shuffle(pool)
        pool.sort(key=lambda i: len(bank.texts[i]))
        seen, chosen = set(), []

        def take(index: int) -> bool:
            text = bank.texts[index]
            if text in seen:
                return False
            seen.add(text)
            chosen.append(index)
            return True

        # 先保证名字/回忆类有带占位符的台词。
        for i in pool:
            text = bank.texts[i]
            if "{player_name}" in text or "{last_player}" in text:
                take(i)
            if len(chosen) >= min(4, per_beat):
                break
        step = max(1, len(pool) // max(1, per_beat))
        for i in pool[::step]:
            if len(chosen) >= per_beat:
                break
            take(i)
        for i in pool:
            if len(chosen) >= per_beat:
                break
            take(i)
        for i in chosen:
            yield {"text": bank.texts[i], "pattern": "fixed",
                   "polarity": 0, "beat": beat}


def rows(fixed_bank: int = 0):
    """Yield every combination, with metadata for later factorized training."""
    yield from fixed_rows()
    yield from fixed_bank_rows(fixed_bank)
    for pattern, template, polarity in PATTERNS:
        if pattern in ("svo", "svo_neg", "ba", "want_svo", "can_svo",
                       "like_svo", "no_svo", "le_svo"):
            for s, v, o in itertools.product(SUBJ, VERB, OBJ):
                yield {"text": template.format(s=s, v=v, o=o),
                       "pattern": pattern, "subj": s, "verb": v,
                       "obj": o, "polarity": polarity}
        elif pattern in ("place_svo",):
            for s, p, v, o in itertools.product(SUBJ, PLACE, VERB, OBJ):
                yield {"text": template.format(s=s, p=p, v=v, o=o),
                       "pattern": pattern, "subj": s, "place": p,
                       "verb": v, "obj": o, "polarity": polarity}
        elif pattern in ("svc",):
            for s, adv, a in itertools.product(SUBJ, ADV, ADJ):
                yield {"text": template.format(s=s, adv=adv, a=a),
                       "pattern": pattern, "subj": s, "adj": a,
                       "adv": adv, "polarity": polarity}
        elif pattern in ("o_is_a", "s_de_o_a", "s_think_o_a"):
            if pattern == "o_is_a":
                combos = ((o, a) for o, a in itertools.product(OBJ, ADJ))
                for o, a in combos:
                    yield {"text": template.format(o=o, a=a),
                           "pattern": pattern, "obj": o, "adj": a,
                           "polarity": polarity}
            else:
                for s, o, a in itertools.product(SUBJ, OBJ, ADJ):
                    yield {"text": template.format(s=s, o=o, a=a),
                           "pattern": pattern, "subj": s, "obj": o,
                           "adj": a, "polarity": polarity}
        elif pattern in ("place_has_o",):
            for p, o in itertools.product(PLACE, OBJ):
                yield {"text": template.format(p=p, o=o),
                       "pattern": pattern, "place": p, "obj": o,
                       "polarity": polarity}
        elif pattern in ("s_place", "o_place"):
            if pattern == "s_place":
                for s, p in itertools.product(SUBJ, PLACE):
                    yield {"text": template.format(s=s, p=p),
                           "pattern": pattern, "subj": s, "place": p,
                           "polarity": polarity}
            else:
                for o, p in itertools.product(OBJ, PLACE):
                    yield {"text": template.format(o=o, p=p),
                           "pattern": pattern, "obj": o, "place": p,
                           "polarity": polarity}


def build(limit: int = 0, fixed_bank: int = 0):
    seen = set()
    out = []
    for row in rows(fixed_bank):
        if not allowed(row) or row["text"] in seen:
            continue
        seen.add(row["text"])
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def stats(entries: list[dict]) -> dict:
    per = Counter(row["pattern"] for row in entries)
    return {"total": len(entries), "patterns": dict(sorted(per.items())),
            "avg_len": round(sum(len(r["text"]) for r in entries)
                             / max(1, len(entries)), 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fixed-bank", type=int, default=265,
                        help="每个 beat 从整句库精选多少条固定台词")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    entries = build(args.limit, args.fixed_bank)
    report = stats(entries)
    if args.stats:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, row in enumerate(entries):
            row["id"] = i
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"写出 {args.out}")
    step = max(1, len(entries) // max(1, args.samples))
    for row in entries[::step][:args.samples]:
        print(f"  [{row['pattern']}] {row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
