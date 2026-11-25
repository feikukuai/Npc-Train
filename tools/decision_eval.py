"""决策头在“已定 beat”时选句的准确率：对比余弦基线。"""
from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import canonical_text, load_rows, real_samples

_, fixed = load_rows(EX, 1, 31)
texts = [r["text"] for r in fixed]
beats = [r.get("beat", "fixed") for r in fixed]
index = {t: i for i, t in enumerate(texts)}
by_beat = {}
for i, beat in enumerate(beats):
    by_beat.setdefault(beat, []).append(i)
canon = {b: canonical_text([texts[i] for i in pool])
         for b, pool in by_beat.items()}
canon = {b: index[t] for b, t in canon.items()}

test = real_samples(index, {b: texts[i] for b, i in canon.items()},
                    2851, 1031, 4, heldout=True)
valid = [i for i, t in enumerate(texts) if "{" not in t]
pos = {i: j for j, i in enumerate(valid)}
test = [r for r in test if r[1] in pos]

enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                        max_length=160, device="cuda")
opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                  dim=-1).cuda()
data = torch.load("expertnet/zh/decision_states_test.pt", map_location="cpu",
                  weights_only=False)
state = data["state"].cuda()
last = data["last"].cuda()
proj = nn.Linear(opt.shape[1], state.shape[1], bias=False).cuda()
proj.load_state_dict(torch.load("expertnet/zh/decision_head.pt",
                                map_location="cuda", weights_only=False)["state"])
with torch.no_grad():
    table = proj(opt)

gold = [pos[r[1]] for r in test]
beat_of = [beats[valid[j]] for j in range(len(valid))]
dec_exact = cos_exact = 0
for i, g in enumerate(gold):
    pool = [j for j in range(len(valid)) if beat_of[j] == beat_of[g]]
    s = state[i]
    pred = pool[int((table[pool] @ s).argmax())]
    dec_exact += int(pred == g)
    pred2 = pool[int((opt[pool] @ last[i]).argmax())]
    cos_exact += int(pred2 == g)

print(json.dumps({"test": len(gold),
                  "decision_within_beat": dec_exact / len(gold),
                  "cosine_within_beat": cos_exact / len(gold)},
                 ensure_ascii=False))
