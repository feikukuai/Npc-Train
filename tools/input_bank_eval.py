"""输入闭集扩容实验：把 SEEDS 扩成说法变体库，测检索映射的覆盖率。

回答两个问题：
  1) 把输入句库从 142 扩到几万条，HELDOUT 的检索映射准不准；
  2) 限制输入长度（≤5 / ≤10 字）时，映射准确率是否更高。
只做检索，不训练、不调 API。
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

PREFIX = ["", "阿禾，", "喂，", "那个……", "我说，", "小阿禾，", "请问，",
          "我想问一下，"]
SUFFIX = ["", "呀", "啊", "哦", "呢", "嘛", "？"]
SYN = [("娘", "妈妈"), ("名字", "称呼"), ("走", "离开"), ("笨", "蠢"),
       ("生气", "不高兴"), ("糖", "糖果"), ("看", "瞧"), ("说", "讲"),
       ("怕", "害怕"), ("喜欢", "爱")]

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


def build_bank(level: int):
    rows = []
    for intent, items in SEEDS.items():
        for phrase in items:
            body = phrase.rstrip("。？！，,.!? ")
            rows.append((phrase, intent))
            if level >= 1:
                for pre in PREFIX:
                    for suf in SUFFIX:
                        rows.append((f"{pre}{body}{suf}。", intent))
            if level >= 2:
                for old, new in SYN:
                    if old in phrase:
                        rows.append((phrase.replace(old, new, 1), intent))
    seen, out = set(), []
    for text, intent in rows:
        if text not in seen:
            seen.add(text)
            out.append((text, intent))
    return out


def retrieve(enc, bank, messages, chunk=2048):
    lib = F.normalize(enc.embed([t for t, _ in bank], batch=128), dim=-1)
    msg = F.normalize(enc.embed(messages, batch=128), dim=-1)
    best = torch.full((len(msg),), -1.0)
    best_idx = torch.zeros(len(msg), dtype=torch.long)
    for start in range(0, len(lib), chunk):
        block = lib[start:start + chunk]
        sims = msg @ block.T
        values, indices = sims.max(1)
        mask = values > best
        best[mask] = values[mask]
        best_idx[mask] = indices[mask] + start
    return best, best_idx


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
    messages = [last_message(r[0]) for r in test]
    gold = [family(beats[r[1]]) for r in test]
    print("留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cpu")
    report = {"n": len(test), "levels": []}
    for level in (0, 1, 2):
        bank = build_bank(level)
        sims, idxs = retrieve(enc, bank, messages)
        hits = 0
        buckets = collections.Counter()
        bucket_hit = collections.Counter()
        for i, message in enumerate(messages):
            got = family(bank[int(idxs[i])][1])
            ok = int(got == gold[i])
            hits += ok
            length = len(message.rstrip("。？！"))
            key = "≤5" if length <= 5 else "6-10" if length <= 10 else \
                "11-20" if length <= 20 else ">20"
            buckets[key] += 1
            bucket_hit[key] += ok
        row = {"level": level, "bank": len(bank),
               "family_acc": round(hits / len(test), 4),
               "by_len": {k: [buckets[k], round(bucket_hit[k] / buckets[k], 3)]
                          for k in ("≤5", "6-10", "11-20", ">20") if buckets[k]},
               "sim_mean": round(float(sims.mean()), 3)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        report["levels"].append(row)

    with open("expertnet/zh/input_bank_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
