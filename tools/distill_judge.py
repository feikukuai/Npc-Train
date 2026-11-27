"""把外部大模型裁判的分数蒸馏成本地小 cross-encoder（BGE 冻结 + 回归头）。

用 judge_scores_300.jsonl 训练：输入「玩家话 + 候选回复」，输出 0-10 分。
验证集上看：本地裁判选出的候选是否与 judge 选择一致、实际分数是否提升。
"""
from __future__ import annotations

import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder

SRC = os.path.join(os.environ["TEMP"], "judge_scores_300.jsonl")
OUT = r"C:\Users\Administrator\Documents\Trae-CN\temp_20261124_03\PyBot-ZH\expertnet\zh\judge_distill.pt"


def main():
    torch.manual_seed(37)
    random.seed(37)
    rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]
    rows = [r for r in rows if r.get("cand_scores")]
    rng = random.Random(37)
    rng.shuffle(rows)
    split = int(len(rows) * 0.85)
    train, val = rows[:split], rows[split:]
    print("训练", len(train), "验证", len(val), flush=True)

    enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                            max_length=192, device="cpu")

    def pairs(rows):
        texts, labels, owners = [], [], []
        for k, row in enumerate(rows):
            replies = [row["greedy"]] + row["cands"]
            scores = [row["greedy_score"]] + row["cand_scores"]
            for reply, score in zip(replies, scores):
                texts.append(f"玩家：{row['message']}\n候选：{reply}")
                labels.append(score / 10)
                owners.append(k)
        return texts, torch.tensor(labels), owners

    ttext, tlabel, _ = pairs(train)
    vtext, vlabel, vowner = pairs(val)
    tfeat = enc.embed(ttext, batch=64)
    vfeat = enc.embed(vtext, batch=64)

    head = nn.Sequential(nn.Linear(tfeat.shape[1], 128), nn.GELU(),
                         nn.Dropout(0.1), nn.Linear(128, 1))
    optim = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    for epoch in range(30):
        perm = torch.randperm(len(tfeat))
        total = 0.0
        for start in range(0, len(perm), 128):
            idx = perm[start:start + 128]
            pred = head(tfeat[idx]).squeeze(-1)
            loss = F.mse_loss(pred, tlabel[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == 29:
            with torch.no_grad():
                vpred = head(vfeat).squeeze(-1)
            mae = (vpred - vlabel).abs().mean().item() * 10
            print(f"epoch {epoch} train_mse {total / len(tfeat):.4f} "
                  f"val_mae {mae:.2f}", flush=True)

    with torch.no_grad():
        vpred = head(vfeat).squeeze(-1) * 10
    agree = 0
    pick_gain = 0.0
    judge_best = 0.0
    greedy_sum = 0.0
    for k in range(len(val)):
        idx = [j for j, owner in enumerate(vowner) if owner == k]
        pred = [float(vpred[j]) for j in idx]
        row = val[k]
        judge_scores = [row["greedy_score"]] + row["cand_scores"]
        agree += int(pred.index(max(pred)) == judge_scores.index(max(judge_scores)))
        greedy_sum += judge_scores[0]
        judge_best += max(judge_scores)
        pick_gain += judge_scores[pred.index(max(pred))]
    print(json.dumps({
        "val": len(val),
        "argmax_agree_with_judge": round(agree / len(val), 3),
        "greedy_avg": round(greedy_sum / len(val), 2),
        "judge_best_avg": round(judge_best / len(val), 2),
        "distilled_pick_avg": round(pick_gain / len(val), 2),
    }, ensure_ascii=False), flush=True)

    torch.save({"state": head.state_dict(), "dim": int(tfeat.shape[1]),
                "max_length": 192}, OUT)
    print("写出", OUT)


if __name__ == "__main__":
    main()
