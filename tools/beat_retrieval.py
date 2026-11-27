"""意图分库检索评测：全库混检 vs 先定 beat 再在子库检索。

对 HELDOUT 的每一轮，取 gold beat 的社交子库做检索，比较相似度覆盖；
同时统计每个 beat 子库需要多少条说法才能覆盖住。
"""
from __future__ import annotations

import collections
import json
import os

import torch
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import canonical_text, load_rows, real_samples


def last_message(context: str) -> str:
    for line in reversed(context.split("\n")):
        if line.startswith("玩家："):
            return line[3:]
    return ""


def main():
    qa = [json.loads(line) for line in open("expertnet/zh/qa_full.jsonl",
                                           encoding="utf-8")]
    qa += [json.loads(line) for line in
           open("expertnet/zh/qa_pairs.jsonl", encoding="utf-8")]
    qa += [json.loads(line) for line in
           open("expertnet/zh/intent_qa.jsonl", encoding="utf-8")]
    social = [r for r in qa if r.get("beat") not in (None, "fixed")]
    by_beat = collections.defaultdict(list)
    for row in social:
        by_beat[row["beat"]].append(row)
    print("QA", len(qa), "社交", len(social), "beat", len(by_beat), flush=True)

    _, fixed = load_rows(EX, 1, 31)
    texts = [r["text"] for r in fixed]
    beats = [r.get("beat", "fixed") for r in fixed]
    index = {t: i for i, t in enumerate(texts)}
    by_beat_fixed = {}
    for i, beat in enumerate(beats):
        by_beat_fixed.setdefault(beat, []).append(i)
    canon = {b: texts[index[canonical_text([texts[i] for i in pool])]]
             for b, pool in by_beat_fixed.items()}
    valid = [i for i, t in enumerate(texts) if "{" not in t]
    pos = {i: j for j, i in enumerate(valid)}
    test = [r for r in real_samples(index, canon, 3000, 1031, 4, heldout=True)
            if r[1] in pos]
    messages = [last_message(r[0]) for r in test]
    gold = [beats[r[1]] for r in test]
    print("留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cpu")
    msg = F.normalize(enc.embed(messages, batch=128), dim=-1)

    # 全库（含事实/组合）矩阵
    lib_all = F.normalize(enc.embed([r["input"] for r in qa], batch=128), dim=-1)
    all_best = torch.full((len(msg),), -1.0)
    for start in range(0, len(lib_all), 2048):
        block = lib_all[start:start + 2048]
        all_best = torch.maximum(all_best, (msg @ block.T).max(1).values)

    # 只对 gold beat 有社交子库的样本比较
    rows = []
    hit = collections.Counter()
    total = 0
    sub_best_list = []
    for i, beat in enumerate(gold):
        if beat not in by_beat or len(by_beat[beat]) < 5:
            continue
        total += 1
        subs = by_beat[beat]
        sub_vec = F.normalize(enc.embed([r["input"] for r in subs], batch=128),
                              dim=-1)
        score = float((msg[i] @ sub_vec.T).max())
        sub_best_list.append(score)
        hit["all≥0.9"] += int(float(all_best[i]) >= 0.9)
        hit["sub≥0.9"] += int(score >= 0.9)
        hit["all≥0.8"] += int(float(all_best[i]) >= 0.8)
        hit["sub≥0.8"] += int(score >= 0.8)
    n = max(1, total)
    print(json.dumps({
        "samples": total,
        "all≥0.9": round(hit["all≥0.9"] / n, 3),
        "sub≥0.9": round(hit["sub≥0.9"] / n, 3),
        "all≥0.8": round(hit["all≥0.8"] / n, 3),
        "sub≥0.8": round(hit["sub≥0.8"] / n, 3),
        "sub_sim_mean": round(sum(sub_best_list) / n, 3),
    }, ensure_ascii=False), flush=True)

    print("子库规模 top10：")
    for beat, items in sorted(by_beat.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  {beat:18s} {len(items)}")


if __name__ == "__main__":
    main()
