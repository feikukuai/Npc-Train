"""输入侧闭集映射：玩家最后一句 → SEEDS 最近句 → 意图 → beat。

不训练、不调 API。目的：验证“检索式编码”能否比模型 beat 分类更准，
为后续“输入句表 → 输出句表 → 润色”的编码-解码结构提供依据。
"""
from __future__ import annotations

import collections
import json
import os

import torch
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.corpus import SEEDS
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import canonical_text, load_rows, real_samples

FAMILY = {
    "greeting": {"greeting", "greeting_again", "greeting_hurt"},
    "farewell": {"farewell", "farewell_hurt"},
    "memory_recall": {"memory_recall", "memory_first", "memory_quote",
                      "memory_hurt", "memory_repeat", "memory_name"},
    "name_saved": {"name_saved", "name_recall", "name_unknown"},
    "smalltalk_how": {"smalltalk_how", "fallback"},
    "personal": {"personal", "lore", "secret", "secret_denied"},
}


def family(beat: str) -> str:
    for key, members in FAMILY.items():
        if beat in members:
            return key
    return beat


def last_message(context: str) -> str:
    for line in reversed(context.split("\n")):
        if line.startswith("玩家："):
            return line[3:]
    return ""


def main():
    _, fixed = load_rows(EX, 1, 31)
    texts = [r["text"] for r in fixed]
    beats = [r.get("beat", "fixed") for r in fixed]
    index = {t: i for i, t in enumerate(texts)}
    by_beat = {}
    for i, beat in enumerate(beats):
        by_beat.setdefault(beat, []).append(i)
    canon = {b: texts[index[canonical_text([texts[i] for i in pool])]]
             for b, pool in by_beat.items()}
    valid = [i for i, t in enumerate(texts) if "{" not in t]
    pos = {i: j for j, i in enumerate(valid)}
    test = [r for r in real_samples(index, canon, 3000, 1031, 4, heldout=True)
            if r[1] in pos]
    print("留存", len(test), flush=True)

    phrases, intents = [], []
    for intent, items in SEEDS.items():
        for phrase in items:
            phrases.append(phrase)
            intents.append(intent)
    print("输入句库", len(phrases), "意图", len(SEEDS), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cpu")
    lib = F.normalize(enc.embed(phrases, batch=128), dim=-1)
    msgs = [last_message(r[0]) for r in test]
    msg = F.normalize(enc.embed(msgs, batch=128), dim=-1)
    sims = msg @ lib.T
    top = sims.topk(5, dim=1)

    gold = [beats[r[1]] for r in test]
    exact = family_hit = 0
    for i in range(len(test)):
        got = intents[int(top.indices[i, 0])]
        exact += int(got == gold[i])
        family_hit += int(family(got) == family(gold[i]))

    # 按检索相似度分桶，看阈值能否筛掉不可靠映射。
    buckets = collections.Counter()
    bucket_hit = collections.Counter()
    for i in range(len(test)):
        score = float(top.values[i, 0])
        b = round(score, 1)
        buckets[b] += 1
        got = intents[int(top.indices[i, 0])]
        bucket_hit[b] += int(family(got) == family(gold[i]))

    print(json.dumps({
        "n": len(test),
        "intent_exact": round(exact / len(test), 4),
        "family_exact": round(family_hit / len(test), 4),
    }, ensure_ascii=False), flush=True)
    for b in sorted(buckets):
        print(f"sim {b:.1f}: n={buckets[b]:4d} "
              f"family_acc={bucket_hit[b] / buckets[b]:.3f}", flush=True)


if __name__ == "__main__":
    main()
