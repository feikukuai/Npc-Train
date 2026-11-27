"""带搭配约束的全量 QA 表：只保留语义成立的“动词×宾语”组合。

机械笛卡尔积会产生“我吃太阳”这种荒谬对；这里用白名单约束，
再叠否定/想/会/喜欢、判断句、处所句，保证每条都说得通。
"""
from __future__ import annotations

import collections
import csv
import itertools
import json

from expertnet.zh.exhaustive import ADJ, ADV, OBJ, PLACE, SUBJ

MAX = 10
SUBJ_IN = ["你", "阿禾", "娘", "阿团", "王伯", "李婶", "客人"]

# 动词 → 允许的宾语（只保留日常成立的搭配）。
VERB_OBJ = {
    "吃": ["糖", "麦芽糖", "药", "花生", "饼", "饭"],
    "喝": ["水", "井水", "药"],
    "拿": ["糖", "药", "铜板", "竹篮", "小木匣"],
    "抱": ["阿团", "猫", "药罐"],
    "数": ["蚂蚁", "石头", "瓦片", "铜板"],
    "看": ["井", "猫", "花", "药园", "药匾"],
    "找": ["猫", "糖", "药", "铜板"],
    "摘": ["薄荷", "甘草", "艾草", "陈皮", "花"],
    "洗": ["药罐", "药匾", "手", "石头"],
    "晒": ["草药", "药匾", "陈皮"],
    "收": ["药", "铜板", "草药"],
    "放": ["糖", "药罐", "竹篮", "药匾"],
    "藏": ["糖", "小木匣", "铜板"],
    "给": ["糖", "药", "铜板"],
    "送": ["糖", "花", "药"],
    "买": ["糖", "药", "铜板"],
    "卖": ["药", "草药", "糖"],
    "煮": ["药", "饭"],
    "切": ["药", "草药", "陈皮"],
    "闻": ["花", "薄荷", "药", "甘草"],
    "摸": ["猫", "药罐", "石头"],
    "喂": ["阿团", "猫"],
    "追": ["阿团", "猫", "蚂蚁"],
    "画": ["猫", "花", "井"],
    "背": ["药", "竹篮"],
    "提": ["竹篮", "药罐"],
    "搬": ["凳子", "药匾", "石头"],
    "扫": ["瓦片", "石头"],
    "捡": ["石头", "铜板", "瓦片"],
    "擦": ["药罐", "柜台", "凳子"],
    "摆": ["药", "药罐", "药匾"],
    "尝": ["糖", "药", "薄荷"],
    "舔": ["糖", "麦芽糖"],
    "咬": ["糖", "饼"],
    "剥": ["花生", "陈皮"],
    "浇": ["花", "薄荷", "草药"],
    "拔": ["草", "薄荷", "艾草"],
    "种": ["花", "薄荷", "草药"],
}

FACT_QA = [
    ("你叫什么", "我叫阿禾。"), ("你是谁", "我叫阿禾。"),
    ("你几岁", "我五岁半。"), ("你多大了", "我五岁半。"),
    ("你住在哪", "我叫阿禾。"), ("你家在哪", "我叫阿禾。"),
    ("你娘叫什么", "我娘叫阿棠。"), ("你妈妈叫什么", "我娘叫阿棠。"),
    ("娘做什么", "娘在药铺抓药。"), ("妈妈做什么", "娘在药铺抓药。"),
    ("你在做什么", "我在数蚂蚁。"), ("阿团是什么", "阿团是猫。"),
    ("阿团在哪", "阿团在柜台下面睡觉。"), ("有猫吗", "有呀，阿团。"),
    ("你认识阿团吗", "认识呀，阿团。"), ("井有多深", "老井很深。"),
    ("糖在哪", "麦芽糖在柜子上。"), ("药园在哪", "屋后是草药园。"),
    ("你有什么秘密", "柴房有只小鸟。"), ("柴房里有什么", "柴房有只小鸟。"),
    ("打雷你怕吗", "打雷我钻桌底。"), ("你数几只蚂蚁", "我数到七只。"),
    ("你知道什么", "我知道药铺。"), ("你想做什么", "我想去看灯。"),
]


def who(s: str) -> str:
    return "我" if s == "你" else s


def main():
    rows, seen = [], set()

    def add(inp: str, ans: str, pattern: str):
        key = (inp, ans)
        if key in seen:
            return
        seen.add(key)
        if len(inp.rstrip("。？！")) > MAX or len(ans.rstrip("。？！")) > MAX:
            return
        rows.append({"input": inp, "answer": ans, "pattern": pattern})

    for inp, ans in FACT_QA:
        add(inp + "？", ans, "fact")

    for v, objs in VERB_OBJ.items():
        for s, o in itertools.product(SUBJ_IN, objs):
            add(f"{s}{v}{o}吗？", f"{who(s)}{v}{o}。", "svo")
            add(f"{s}不{v}{o}吗？", f"{who(s)}不{v}{o}。", "svo_neg")
            add(f"{s}想{v}{o}吗？", f"{who(s)}想{v}{o}。", "want_svo")
            add(f"{s}会{v}{o}吗？", f"{who(s)}会{v}{o}。", "can_svo")
            add(f"{s}喜欢{v}{o}吗？", f"{who(s)}喜欢{v}{o}。", "like_svo")
    for s, a in itertools.product(SUBJ_IN, ADJ):
        add(f"{s}很{a}吗？", f"{who(s)}很{a}。", "svc")
    for o, a in itertools.product(OBJ, ADJ):
        add(f"{o}是{a}的吗？", f"{o}是{a}的。", "o_is_a")
    for o, p in itertools.product(OBJ, PLACE):
        add(f"{o}在{p}吗？", f"{o}在{p}。", "o_place")
    for p, o in itertools.product(PLACE, OBJ):
        add(f"{p}有{o}吗？", f"{p}有{o}。", "place_has_o")
    for s, p in itertools.product(SUBJ_IN, PLACE):
        add(f"{s}在{p}吗？", f"{who(s)}在{p}。", "s_place")

    with open("expertnet/zh/qa_pairs.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open("expertnet/zh/qa_pairs_sample.csv", "w", encoding="utf-8-sig",
              newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["玩家输入", "阿禾回答", "句式", "输入字数", "回答字数"])
        for row in rows[:3000]:
            writer.writerow([row["input"], row["answer"], row["pattern"],
                             len(row["input"].rstrip("。？！")),
                             len(row["answer"].rstrip("。？！"))])

    counts = collections.Counter(r["pattern"] for r in rows)
    print("QA 对", len(rows), "条")
    for pattern, n in counts.most_common():
        print(f"  {pattern:12s} {n}")
    print("样例：")
    for row in rows[24:34]:
        print(f"  {row['input']}  ->  {row['answer']}")


if __name__ == "__main__":
    main()
