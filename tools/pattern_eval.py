"""用模式表的真实说法当输入库，测检索映射准确率。"""
from __future__ import annotations

import collections
import json

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import canonical_text, load_rows, real_samples
from tools.input_bank_eval import family, last_message, retrieve

import torch.nn.functional as F


def main():
    bank = []
    for line in open("expertnet/zh/patterns.jsonl", encoding="utf-8"):
        row = json.loads(line)
        for text in row["utterances"]:
            bank.append((text, row["beat"]))
    print("模式表输入库", len(bank), flush=True)

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

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cpu")
    sims, idxs = retrieve(enc, bank, messages)
    buckets = collections.Counter()
    hit_bucket = collections.Counter()
    hits = 0
    for i, message in enumerate(messages):
        got = family(bank[int(idxs[i])][1])
        ok = int(got == gold[i])
        hits += ok
        length = len(message.rstrip("。？！"))
        key = "≤5" if length <= 5 else "6-10" if length <= 10 else "11-20" \
            if length <= 20 else ">20"
        buckets[key] += 1
        hit_bucket[key] += ok
    print(json.dumps({
        "n": len(test),
        "family_acc": round(hits / len(test), 4),
        "by_len": {k: [buckets[k], round(hit_bucket[k] / buckets[k], 3)]
                   for k in ("≤5", "6-10", "11-20", ">20") if buckets[k]},
        "sim_mean": round(float(sims.mean()), 3),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
