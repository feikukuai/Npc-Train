"""全量 QA 标注：33 万组合句逐条生成“玩家问法 → 阿禾回答”。

规则是确定性的：输入 = 输出句做主语视角翻转 + 疑问化；回答 = 原句。
社交固定句用“意图种子说法 × 固定句”做笛卡尔积。
输出 qa_full.jsonl（全量）+ 统计 + 样例。
"""
from __future__ import annotations

import collections
import json

from tools.build_qa_pairs import VERB_OBJ

SRC = "expertnet/zh/exhaustive.jsonl"
OUT = "expertnet/zh/qa_full.jsonl"
MAX_IN = 12
VERB_PATTERNS = {"svo", "svo_neg", "ba", "place_svo", "want_svo", "can_svo",
                 "like_svo", "no_svo", "le_svo", "s_de_o_a", "s_think_o_a"}

Q = {
    "svo": "{s}{v}{o}吗？",
    "svo_neg": "{s}不{v}{o}吗？",
    "ba": "{s}把{o}{v}了吗？",
    "place_svo": "{s}在{p}{v}{o}吗？",
    "svc": "{s}{adv}{a}吗？",
    "o_is_a": "{o}是{a}的吗？",
    "s_de_o_a": "{s}的{o}很{a}吗？",
    "s_think_o_a": "{s}觉得{o}很{a}吗？",
    "place_has_o": "{p}有{o}吗？",
    "want_svo": "{s}想{v}{o}吗？",
    "can_svo": "{s}会{v}{o}吗？",
    "like_svo": "{s}喜欢{v}{o}吗？",
    "no_svo": "{s}没有{v}{o}吗？",
    "le_svo": "{s}{v}了{o}吗？",
    "s_place": "{s}在{p}吗？",
    "o_place": "{o}在{p}吗？",
}
# 玩家视角：阿禾说“我” → 问“你”；阿禾说“你” → 问“我”。
FLIP = {"我": "你", "你": "我"}
# 社交意图的玩家说法（少量种子，够做笛卡尔积）。
SOCIAL_SEED = {
    "greeting": ["你好", "嗨", "我回来啦", "阿禾在吗", "好久不见"],
    "farewell": ["我走了", "再见", "拜拜", "明天见"],
    "thanks": ["谢谢你", "多谢", "谢啦", "麻烦你了"],
    "apology": ["对不起", "抱歉", "我错了", "别生气"],
    "insult": ["你真笨", "傻瓜", "讨厌", "蠢死了"],
    "threat": ["我打你", "把你卖了", "小心点"],
    "taunt": ["你敢吗", "有本事你来", "你不敢吧"],
    "compliment": ["你真可爱", "你好聪明", "真乖"],
    "gift": ["给你一颗糖", "这个送你", "请你吃"],
    "promise": ["我保证", "我发誓", "拉钩"],
    "trade": ["这个多少钱", "我买", "换不换"],
    "smalltalk_weather": ["今天天气怎么样", "会下雨吗", "好冷"],
    "smalltalk_bored": ["好无聊", "陪我玩", "没意思"],
    "smalltalk_joke": ["讲个笑话", "有什么好笑的"],
    "smalltalk_how": ["你在干嘛", "最近好吗", "心情怎么样"],
    "boundary": ["说点脏话", "你脱衣服"],
    "secret_denied": ["你有什么秘密", "你藏了什么"],
    "name_saved": ["我叫小明", "我的名字是小明", "叫我小明"],
    "name_unknown": ["我叫什么", "你记得我的名字吗"],
    "memory_recall": ["我刚才说了什么", "我们刚才聊什么"],
    "meta": ["你是机器人吗", "这是游戏吗"],
    "fallback": ["那个", "嗯", "随便说说"],
    "out_of_world": ["天为什么是蓝的", "附近有村子吗", "你穿什么衣服"],
}


def main():
    rows = []
    stats = collections.Counter()
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = row["text"].rstrip("。？！")
            if row["pattern"] == "fixed":
                beat = row.get("beat", "fallback")
                seeds = SOCIAL_SEED.get(beat)
                if not seeds:
                    continue
                for utter in seeds:
                    rows.append({"input": utter + "？", "answer": text + "。",
                                 "pattern": "social", "beat": beat})
                    stats["social"] += 1
                continue
            pattern = row["pattern"]
            template = Q.get(pattern)
            if not template:
                continue
            verb, obj = row.get("verb", ""), row.get("obj", "")
            if pattern in VERB_PATTERNS and obj not in VERB_OBJ.get(verb, []):
                continue
            values = {
                "s": FLIP.get(row.get("subj", ""), row.get("subj", "")),
                "v": row.get("verb", ""), "o": row.get("obj", ""),
                "a": row.get("adj", ""), "p": row.get("place", ""),
                "adv": row.get("adv", ""),
            }
            try:
                inp = template.format(**values)
            except KeyError:
                continue
            if len(inp.rstrip("？")) > MAX_IN:
                continue
            rows.append({"input": inp, "answer": text + "。",
                         "pattern": pattern,
                         "beat": row.get("beat", "fixed")})
            stats[pattern] += 1

    with open(OUT, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("全量 QA", len(rows), "条")
    for pattern, n in stats.most_common(12):
        print(f"  {pattern:14s} {n}")
    print("样例：")
    for row in rows[:6] + rows[len(rows) // 2:len(rows) // 2 + 6]:
        print(f"  {row['input']}  ->  {row['answer']}")


if __name__ == "__main__":
    main()
