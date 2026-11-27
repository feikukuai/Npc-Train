"""评测 QA 表作为检索式回答的效果：HELDOUT 最后一句 → 最近 QA 输入 → 回答。"""
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
    qa = [json.loads(line) for line in open("expertnet/zh/qa_pairs.jsonl",
                                           encoding="utf-8")]
    if os.path.exists("expertnet/zh/intent_qa.jsonl"):
        qa += [json.loads(line) for line in
               open("expertnet/zh/intent_qa.jsonl", encoding="utf-8")]
    if os.path.exists("expertnet/zh/qa_full.jsonl"):
        qa += [json.loads(line) for line in
               open("expertnet/zh/qa_full.jsonl", encoding="utf-8")]
    print("QA", len(qa), flush=True)

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
    print("留存", len(test), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=160, device="cpu")
    lib = F.normalize(enc.embed([r["input"] for r in qa], batch=128), dim=-1)
    msg = F.normalize(enc.embed(messages, batch=128), dim=-1)
    best = torch.full((len(msg),), -1.0)
    best_idx = torch.zeros(len(msg), dtype=torch.long)
    for start in range(0, len(lib), 2048):
        block = lib[start:start + 2048]
        values, indices = (msg @ block.T).max(1)
        mask = values > best
        best[mask] = values[mask]
        best_idx[mask] = indices[mask] + start

    buckets = collections.Counter()
    for score in best.tolist():
        key = "≥0.95" if score >= 0.95 else "0.9-0.95" if score >= 0.9 else \
            "0.8-0.9" if score >= 0.8 else "<0.8"
        buckets[key] += 1
    legal = sum(1 for i in best_idx.tolist()
                if len(qa[i]["answer"].rstrip("。？！")) <= 10)
    print(json.dumps({
        "n": len(test),
        "sim_mean": round(float(best.mean()), 3),
        "coverage": {k: round(buckets[k] / len(test), 3) for k in
                     ("≥0.95", "0.9-0.95", "0.8-0.9", "<0.8")},
        "answer_legal": round(legal / len(test), 3),
    }, ensure_ascii=False), flush=True)
    print("样例：")
    for i in range(0, len(test), 240):
        row = qa[int(best_idx[i])]
        print(f"  问：{messages[i][:20]}  →  {row['answer']} "
              f"(sim {float(best[i]):.2f})", flush=True)


if __name__ == "__main__":
    main()
