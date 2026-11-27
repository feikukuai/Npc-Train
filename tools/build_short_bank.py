"""生成“短句版”输出表：组合句全保留（99.8% ≤10 字），固定句只留 ≤10 字，
并把事实句替换成短句版本。输出 short_bank.jsonl 与统计。
"""
from __future__ import annotations

import collections
import json

SRC = "expertnet/zh/exhaustive.jsonl"
OUT = "expertnet/zh/short_bank.jsonl"
MAX = 10

# 长事实句 → 短句版本（保持 beat=personal）。
SHORT_FACTS = {
    "我五岁半，娘说再过半年就六岁。": "我五岁半。",
    "我叫阿禾，住在村口药铺。": "我叫阿禾。",
    "娘在药铺抓药，她认得所有草。": "娘在药铺抓药。",
    "娘教我认草药，我不去学堂。": "娘教我认草药。",
    "我数到七只，后来它们跑了。": "我数到七只。",
    "阿团是橘色的，它只肯让我抱。": "阿团是橘色的。",
    "阿团是猫呀，橘色的。": "阿团是猫。",
    "阿团在柜台下面睡觉。": "阿团在柜台下面睡觉。",
    "认识呀，阿团天天在柜台下打盹。": "认识呀，阿团。",
    "有呀，阿团天天在柜台下打盹。": "有呀，阿团。",
    "老井很深，娘不让我靠近。": "老井很深。",
    "麦芽糖在柜子最上面，我够不着。": "麦芽糖在柜子上。",
    "屋后是草药园，薄荷揉一揉手会香。": "屋后是草药园。",
    "柴房里有一只受伤的小鸟，你别告诉别人。": "柴房有只小鸟。",
    "打雷的时候，我钻到桌子底下。": "打雷我钻桌底。",
    "我想去山那边看灯，娘说等我长大。": "我想去看灯。",
    "我知道药铺、药园、老井、阿团和娘。": "我知道药铺。",
    "太阳好大，井水是凉的。": "太阳好大。",
    "要下雨了，蚂蚁都在搬家。": "要下雨了。",
}


# 长句筛选后安全/记忆类太少，补一批 ≤10 字的短句。
EXTRA = [
    ("我不听这个", "boundary"), ("娘不许说", "boundary"),
    ("我不说这个", "boundary"), ("你别说了", "boundary"),
    ("我不怕你", "threat"), ("我喊我娘", "threat"),
    ("你别过来", "threat"), ("我娘在屋里", "threat"),
    ("娘说不能告诉外人", "secret_denied"), ("这是秘密", "secret_denied"),
    ("你先拉钩", "secret_denied"), ("我不告诉你", "secret_denied"),
    ("你刚才说的", "memory_quote"),
    ("你叫{player_name}", "name_recall"),
    ("{player_name}，对不对", "name_recall"),
    ("我记住{player_name}了", "name_saved"),
]


def main():
    combo = collections.Counter()
    fixed = collections.Counter()
    kept = []
    short_fact_seen = set()
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = row["text"]
            if row["pattern"] != "fixed":
                combo[row["pattern"]] += 1
                kept.append(row)
                continue
            beat = row.get("beat", "fixed")
            if text in SHORT_FACTS:
                if text in short_fact_seen:
                    continue
                short_fact_seen.add(text)
                row = dict(row, text=SHORT_FACTS[text])
                text = row["text"]
            if len(text.rstrip("。？！")) <= MAX:
                fixed[beat] += 1
                kept.append(row)
    seen_text = {row["text"] for row in kept if row["pattern"] == "fixed"}
    base_id = max((row.get("id", 0) for row in kept), default=0) + 1
    for text, beat in EXTRA:
        if text in seen_text:
            continue
        kept.append({"id": base_id, "text": text + "。", "pattern": "fixed",
                     "polarity": 0, "beat": beat})
        base_id += 1
        fixed[beat] += 1
        seen_text.add(text + "。")
    with open(OUT, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("短句表", len(kept), "句；组合句", sum(combo.values()), "；固定句", sum(fixed.values()))
    print("固定句按 beat：")
    for beat, n in fixed.most_common():
        print(f"  {beat:18s} {n}")
    print("事实短句：")
    for old, new in SHORT_FACTS.items():
        if new in short_fact_seen or True:
            print(f"  {old}  ->  {new}  ({len(new.rstrip('。？！'))} 字)")


if __name__ == "__main__":
    main()
