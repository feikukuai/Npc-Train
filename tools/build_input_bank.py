"""生成“玩家输入句表”原型：主谓宾/主系表/判断句 + 疑问，限制 10 字以内。

复用输出表的词表，验证输入闭集的真实规模，并输出 CSV 样例供审阅。
"""
from __future__ import annotations

import collections
import csv
import itertools

from expertnet.zh.exhaustive import ADJ, ADV, OBJ, PLACE, SUBJ, VERB

SUBJ_IN = ["你", "阿禾", "娘", "阿团"]
MAX_LEN = 10

TEMPLATES = [
    ("svo_q", "{s}吃{o}吗？"),
    ("svo_neg_q", "{s}不吃{o}吗？"),
    ("place_q", "{s}在{p}吗？"),
    ("o_where", "{o}在哪？"),
    ("s_what", "{s}在做什么？"),
    ("svc_q", "{s}很{a}吗？"),
    ("o_is_a_q", "{o}是{a}的吗？"),
    ("s_think_q", "{s}觉得{o}怎么样？"),
    ("o_place_q", "{o}在{p}吗？"),
    ("s_who", "{s}是谁？"),
]

# 只取一部分谓语/形容词，避免句子过长无法落进 10 字。
VERB_IN = VERB[:24]
ADJ_IN = ADJ[:24]


def clean(text: str) -> str:
    return text.rstrip("。？！")


def main():
    rows = []
    seen = set()
    counts = collections.Counter()

    def add(pattern, text):
        body = clean(text)
        if len(body) > MAX_LEN or body in seen:
            return
        seen.add(body)
        counts[pattern] += 1
        rows.append({"pattern": pattern, "text": body})

    for s, o in itertools.product(SUBJ_IN, OBJ):
        add("svo_q", f"{s}吃{o}吗？")
        add("svo_neg_q", f"{s}不吃{o}吗？")
        add("o_where", f"{o}在哪？")
    for s, p in itertools.product(SUBJ_IN, PLACE):
        add("place_q", f"{s}在{p}吗？")
        add("s_what", f"{s}在做什么？")
    for s, a in itertools.product(SUBJ_IN, ADJ_IN):
        add("svc_q", f"{s}很{a}吗？")
    for o, a in itertools.product(OBJ, ADJ_IN):
        add("o_is_a_q", f"{o}是{a}的吗？")
    for o, p in itertools.product(OBJ, PLACE):
        add("o_place_q", f"{o}在{p}吗？")
    for s in SUBJ_IN:
        add("s_who", f"{s}是谁？")
    for s, o in itertools.product(SUBJ_IN, OBJ[:20]):
        add("s_think_q", f"{s}觉得{o}怎么样？")

    with open("expertnet/zh/input_patterns.csv", "w", encoding="utf-8-sig",
              newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["句式", "玩家输入", "字数"])
        for row in rows[:3000]:
            writer.writerow([row["pattern"], row["text"], len(row["text"])])

    print("生成输入句", len(rows), "条（≤10 字，去重）")
    for pattern, n in counts.most_common():
        print(f"  {pattern:12s} {n:6d}")
    print("样例：")
    for row in rows[:12]:
        print("  ", row["pattern"], row["text"])


if __name__ == "__main__":
    main()
