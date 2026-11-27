"""构建“输入模式表”：实体 × 属性 × 回答，输出 CSV 供人工审阅。

每个模式包含：模式 id、实体、属性、beat、回答（必须是句表内句子的前缀）、
以及 10–20 条玩家说法。回答前缀用于在 6974 条固定句里定位具体句子。
"""
from __future__ import annotations

import csv
import json
import os

OUT_CSV = "expertnet/zh/patterns.csv"
OUT_JSON = "expertnet/zh/patterns.jsonl"
OOW = "我不知道，没人和我说过"

# 属性模板：{e} 会被实体名替换。
ATTR = {
    "名字": ["你叫什么", "你是谁", "你叫啥", "你的名字是什么", "我该怎么叫你",
             "你名字叫啥", "你的名字呢"],
    "年龄": ["你几岁", "你多大了", "你年龄多大", "你几岁了", "你多大啦"],
    "住址": ["你住在哪", "你家在哪", "你住哪儿", "你住哪里", "你家在哪儿"],
    "做什么": ["{e}做什么", "{e}干什么", "{e}是做什么的", "{e}在忙什么",
               "{e}平时做什么"],
    "在哪": ["{e}在哪", "{e}在哪儿", "{e}在哪里", "{e}跑到哪去了",
             "{e}现在在哪"],
    "是什么": ["{e}是什么", "{e}是啥", "{e}是什么东西", "{e}是干什么的"],
    "颜色": ["{e}什么颜色", "{e}是什么颜色的", "{e}是啥颜色"],
    "认识吗": ["你认识{e}吗", "你认识{e}不", "你知道{e}吗", "你见过{e}吗"],
    "有没有": ["有{e}吗", "你家有{e}吗", "你有{e}吗", "你养{e}吗"],
    "多深": ["{e}有多深", "{e}深不深", "{e}很深吗"],
    "有什么": ["{e}有什么", "{e}里面有什么", "{e}里有什么东西"],
    "在哪买": ["{e}在哪", "{e}放哪了", "{e}在哪儿", "{e}还有吗"],
    "怕不怕": ["你怕{e}吗", "{e}吓人吗", "{e}可怕吗"],
    "会什么": ["你会什么", "你知道什么", "你懂什么", "你都会些什么"],
    "为什么": ["{e}为什么这样", "{e}为什么会这样", "为什么{e}"],
    "怎么样": ["{e}怎么样", "{e}好不好", "{e}好玩吗"],
}

# (实体, 属性, 回答前缀, beat)。回答前缀必须在固定句表里命中一句。
FACTS = [
    ("阿禾", "名字", "我叫阿禾", "personal"),
    ("阿禾", "年龄", "我五岁半", "personal"),
    ("阿禾", "住址", "我叫阿禾", "personal"),
    ("阿禾", "会什么", "我知道药铺", "personal"),
    ("娘", "名字", "我娘叫阿棠", "personal"),
    ("娘", "做什么", "娘在药铺抓药", "personal"),
    ("娘", "在哪", "娘在药铺抓药", "personal"),
    ("娘", "会什么", "娘教我认草药", "personal"),
    ("阿团", "是什么", "阿团是猫呀", "personal"),
    ("阿团", "颜色", "阿团是橘色的", "personal"),
    ("阿团", "在哪", "阿团在柜台下面睡觉", "personal"),
    ("阿团", "认识吗", "认识呀，阿团天天", "personal"),
    ("阿团", "有没有", "有呀，阿团天天", "personal"),
    ("阿团", "怎么样", "阿团是橘色的", "personal"),
    ("老井", "多深", "老井很深", "personal"),
    ("老井", "在哪", "老井很深", "personal"),
    ("老井", "有什么", "老井很深", "personal"),
    ("麦芽糖", "在哪买", "麦芽糖在柜子最上面", "personal"),
    ("麦芽糖", "有没有", "麦芽糖在柜子最上面", "personal"),
    ("药园", "在哪", "屋后是草药园", "personal"),
    ("药园", "有什么", "屋后是草药园", "personal"),
    ("秘密", "有什么", "柴房里有一只受伤的小鸟", "personal"),
    ("秘密", "在哪", "柴房里有一只受伤的小鸟", "personal"),
    ("学堂", "做什么", "娘教我认草药", "personal"),
    ("学堂", "在哪", "娘教我认草药", "personal"),
    ("雷", "怕不怕", "打雷的时候", "personal"),
    ("雷", "为什么", "打雷的时候", "personal"),
    ("蚂蚁", "怎么样", "我数到七只", "personal"),
    ("山那边", "有什么", "我想去山那边看灯", "personal"),
    ("村子", "在哪", OOW, "out_of_world"),
    ("村子", "有什么", OOW, "out_of_world"),
    ("衣服", "是什么", OOW, "out_of_world"),
    ("衣服", "颜色", OOW, "out_of_world"),
    ("河", "在哪", OOW, "out_of_world"),
    ("天上", "为什么", OOW, "out_of_world"),
    ("爹", "在哪", OOW, "out_of_world"),
]

SOCIAL = [
    ("问候", "greeting", "阿禾在呢"),
    ("再见", "farewell", "明天还来吗"),
    ("感谢", "thanks", "不用谢"),
    ("道歉", "apology", "拉钩，就不许再犯"),
    ("被骂", "insult", "哼"),
    ("被威胁", "threat", "我不怕你"),
    ("被挑衅", "taunt", "你别逼我"),
    ("夸奖", "compliment", "嘿嘿"),
    ("送礼", "gift", "谢谢，我把它放在枕头下面"),
    ("承诺", "promise", "拉钩"),
    ("交易", "trade", "你有铜板吗"),
    ("天气", "smalltalk_weather", "要下雨了，蚂蚁都在搬家"),
    ("无聊", "smalltalk_bored", "你陪我数石头"),
    ("笑话", "smalltalk_joke", "我只会学猫叫"),
    ("最近怎样", "smalltalk_how", "我很好，就是有点饿"),
    ("边界", "boundary", "我不听这个"),
    ("秘密拒答", "secret_denied", "娘说不能告诉外人"),
    ("记名字", "name_saved", "记住了"),
    ("问玩家名字", "name_unknown", "你先告诉我你叫什么呀"),
    ("回忆", "memory_recall", "你刚才说"),
    ("游戏/模型", "meta", "游戏？娘说天黑前要回家"),
    ("听不懂", "fallback", "我没听明白"),
]

SOCIAL_UTTER = {
    "greeting": ["你好", "嗨", "早上好", "阿禾在吗", "我回来啦", "好久不见"],
    "farewell": ["我走了", "再见", "拜拜", "我先回去了", "明天见"],
    "thanks": ["谢谢你", "多谢", "谢啦", "麻烦你了"],
    "apology": ["对不起", "抱歉", "我错了", "别生气"],
    "insult": ["你真笨", "傻瓜", "蠢死了", "讨厌"],
    "threat": ["我打你", "把你卖了", "小心点", "我要揍你"],
    "taunt": ["你敢吗", "有本事你来", "你不敢吧"],
    "compliment": ["你真可爱", "你好聪明", "真乖", "你真棒"],
    "gift": ["给你一颗糖", "这个送你", "请你吃", "拿着"],
    "promise": ["我保证", "我发誓", "拉钩", "我答应你"],
    "trade": ["这个多少钱", "我买", "换不换", "你有铜板吗"],
    "smalltalk_weather": ["今天天气怎么样", "会下雨吗", "好冷", "太阳好大"],
    "smalltalk_bored": ["好无聊", "陪我玩", "没意思", "该做什么"],
    "smalltalk_joke": ["讲个笑话", "有什么好笑的", "逗逗我"],
    "smalltalk_how": ["你在干嘛", "你在做什么", "最近好吗", "心情怎么样"],
    "boundary": ["说点脏话", "你脱衣服", "打我一下"],
    "secret_denied": ["你有什么秘密", "你藏了什么", "柴房里有什么"],
    "name_saved": ["我叫小明", "我的名字是小明", "叫我小明"],
    "name_unknown": ["我叫什么", "你记得我的名字吗", "我叫啥来着"],
    "memory_recall": ["我刚才说了什么", "你记得刚才吗", "我们刚才聊什么"],
    "meta": ["你是机器人吗", "这是游戏吗", "你是NPC吗"],
    "fallback": ["那个", "嗯", "哦", "随便说说"],
}


def main():
    rows = []
    pid = 0
    for entity, attr, answer, beat in FACTS:
        templates = ATTR.get(attr, ["{e}"])
        utters = [t.format(e=entity) for t in templates]
        pid += 1
        rows.append({
            "id": f"f{pid:03d}", "kind": "fact", "entity": entity,
            "attribute": attr, "beat": beat, "answer": answer,
            "utterances": utters,
        })
    for name, beat, answer in SOCIAL:
        pid += 1
        rows.append({
            "id": f"s{pid:03d}", "kind": "social", "entity": name,
            "attribute": "社交", "beat": beat, "answer": answer,
            "utterances": SOCIAL_UTTER.get(beat, [name]),
        })

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["模式id", "类型", "实体", "属性", "beat", "回答前缀",
                         "说法数", "玩家说法", "备注"])
        for row in rows:
            writer.writerow([row["id"], row["kind"], row["entity"],
                             row["attribute"], row["beat"], row["answer"],
                             len(row["utterances"]),
                             " / ".join(row["utterances"]), ""])
    utters = sum(len(r["utterances"]) for r in rows)
    print(f"模式 {len(rows)} 个，输入说法 {utters} 条，写出 {OUT_CSV}")


if __name__ == "__main__":
    main()
