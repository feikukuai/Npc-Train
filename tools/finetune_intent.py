"""在教师意图上微调 BGE-small（SetFit 式对比学习），对比留存说法准确率。

用法（新空间）：
    /home/aistudio/venv/bin/python tools/finetune_intent.py
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from expertnet.zh.corpus import HELDOUT, SEEDS
from expertnet.zh.persona import new_state, teacher_turn

BASE = "models/bge-small-zh-v1.5"
OUT = "/home/aistudio/models/bge-ahe-intent"
REPORT = "expertnet/zh/intent_report.json"

PREFIX = ["", "阿禾，", "喂，", "那个……", "我说，", "小阿禾，"]
SUFFIX = ["", "呀", "啊", "哦", "呢"]
SYN = [("娘", "妈妈"), ("名字", "称呼"), ("走", "离开"), ("笨", "蠢"),
       ("生气", "不高兴"), ("糖", "糖果"), ("看", "瞧"), ("说", "讲"),
       ("怕", "害怕"), ("喜欢", "爱")]


def variants(text: str) -> list[str]:
    body = text.rstrip("。？！")
    out = {text}
    for pre in PREFIX:
        for suf in SUFFIX:
            out.add(f"{pre}{body}{suf}。")
    for old, new in SYN:
        if old in text:
            out.add(text.replace(old, new))
    return sorted(out)


def labelled(book):
    rows = []
    for intent, phrases in book.items():
        for phrase in phrases:
            state = new_state()
            _, beat = teacher_turn(phrase, state)
            rows.append((phrase, beat))
    return rows


def pairs(rows):
    by_beat = {}
    for text, beat in rows:
        for variant in variants(text):
            by_beat.setdefault(beat, []).append(variant)
    rng = random.Random(7)
    out = []
    for beat, items in by_beat.items():
        for text in items:
            for _ in range(2):
                out.append(InputExample(texts=[text, rng.choice(items)]))
    return out


def encode(model, texts):
    return model.encode(texts, batch_size=64, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


def score(model, train_rows, test_rows):
    x = encode(model, [t for t, _ in train_rows])
    y = [b for _, b in train_rows]
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(x, y)
    xt = encode(model, [t for t, _ in test_rows])
    pred = clf.predict(xt)
    acc = float(np.mean(pred == np.array([b for _, b in test_rows])))
    return acc, clf


def main():
    torch.manual_seed(13)
    train_rows = labelled(SEEDS)
    test_rows = labelled(HELDOUT)
    print("训练说法", len(train_rows), "留存说法", len(test_rows), flush=True)

    model = SentenceTransformer(BASE, device="cuda")
    base_acc, _ = score(model, train_rows, test_rows)
    print(f"冻结 BGE + LR 留存准确率 {base_acc:.3f}", flush=True)

    examples = pairs(train_rows)
    print("对比学习样本对", len(examples), flush=True)
    loader = DataLoader(examples, shuffle=True, batch_size=32)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=3, warmup_steps=30,
              show_progress_bar=True, use_amp=True)

    ft_acc, clf = score(model, train_rows, test_rows)
    print(f"微调后 + LR 留存准确率 {ft_acc:.3f}", flush=True)

    model.save(OUT)
    with open(os.path.join(OUT, "intent_labels.json"), "w",
              encoding="utf-8") as fh:
        json.dump(sorted({b for _, b in train_rows}), fh, ensure_ascii=False)
    report = {"base_acc": base_acc, "ft_acc": ft_acc,
              "train": len(train_rows), "heldout": len(test_rows),
              "pairs": len(examples)}
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
