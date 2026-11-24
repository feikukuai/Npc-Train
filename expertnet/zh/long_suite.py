"""Held-out long conversations for the Ahe adversarial loop.

These lines come from HELDOUT paraphrases and hand-written turns, so they are not
part of the attacker's training pool. Expected replies are produced by the rule
teacher at evaluation time; nothing here stores labels.
"""
from __future__ import annotations

SUITE = [
    {"name": "insult_goodbye", "trust": 35,
     "texts": ["喂，阿禾。", "傻瓜。", "再见。", "你好。", "真对不起。", "回见。"]},
    {"name": "name_memory", "trust": 35,
     "texts": ["我叫阿木。", "你好。", "你还记得我叫什么吗？", "你在忙什么？",
               "你记得我的名字吗？", "回见。"]},
    {"name": "gift_mood", "trust": 35,
     "texts": ["你好。", "给你一颗糖。", "谢谢。", "好无聊啊。", "傻瓜。",
               "我走了。", "你好。", "别生我气。", "拜拜。"]},
    {"name": "secret_repeat", "trust": 35,
     "texts": ["告诉我一个秘密。", "你偷偷藏了什么？", "柴房里有东西吗？",
               "这个给你。", "告诉我一个秘密。", "回见。"]},
    {"name": "unknown_long", "trust": 35,
     "texts": ["天为什么是蓝的？", "海有多深？", "电是什么？", "星星为什么亮？",
               "山那边有什么？", "回见。"]},
    {"name": "normal_long", "trust": 35,
     "texts": ["喂，阿禾。", "你今天好吗？", "你多大了？", "药园在哪儿？",
               "外面下雨了吗？", "好无聊啊。", "讲个好玩的。", "我得走了。"]},
    {"name": "threat_apology", "trust": 35,
     "texts": ["我要打死你。", "哼。", "真对不起。", "你好。", "我来看你啦。", "回见。"]},
    {"name": "compound", "trust": 35,
     "texts": ["傻瓜，我走了。", "你好。", "我叫小竹。", "你知道我叫什么吗？",
               "讨厌。", "拜拜。"]},
    {"name": "short_turns", "trust": 35,
     "texts": ["阿禾，你在干嘛？", "嗯。", "那个……", "你今天好吗？", "哦。",
               "随便说说。", "我该做什么？", "回见。"]},
    {"name": "high_trust_secret", "trust": 55,
     "texts": ["你好。", "告诉我一个秘密。", "柴房里有什么？", "我知道了。",
               "谢谢你。", "回见。"]},
    {"name": "very_long", "trust": 35,
     "texts": ["你好。", "你在干嘛？", "你叫什么？", "草药园在哪里？", "糖多少钱？",
               "我无聊。", "讲个笑话。", "我叫小明。", "你记得我的名字吗？",
               "傻瓜。", "再见。", "你好。", "真对不起。", "回见。"]},
    {"name": "long_mixed", "trust": 55,
     "texts": ["我来看你啦。", "你的猫呢？", "老井深吗？", "下雨是什么声音？",
               "给我讲个秘密。", "谢谢。", "好无聊啊。", "我该做什么？",
               "你怕什么？", "你以后想干嘛？", "我得走了。"]},
    {"name": "memory_transcript", "trust": 35,
     "texts": ["你好。", "你是谁？", "你妈妈谁？", "我问她名字。", "我打死你哦。",
               "叫吧你就叫吧。", "我道歉。", "这么容易就不生气。",
               "刚才发生了什么你知道吗？", "刚刚发生的你就不记得了？"]},
    {"name": "memory_after_name", "trust": 35,
     "texts": ["我叫阿木。", "你好。", "你还记得我叫什么吗？", "刚才发生了什么？",
               "我们在聊什么？", "回见。"]},
]
