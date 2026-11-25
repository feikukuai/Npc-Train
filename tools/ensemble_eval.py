"""把 beat 分类器与决策头做集成，看能否同时提高 beat 和整句准确率。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from expertnet.sentence import HFSentenceEncoder
from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import (FactorizedNet, canonical_text,
                                           line_tensor, load_rows, real_samples)

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
                    3000, 1031, 4, heldout=True)
valid = [i for i, t in enumerate(texts) if "{" not in t]
valid_set = set(valid)
test = [r for r in test if r[1] in valid_set]
gold = [r[1] for r in test]
gold_beat = [beats[g] for g in gold]

enc = HFSentenceEncoder("models/bge-small-zh-v1.5", pooling="cls",
                        max_length=160, device="cuda")
ckpt = torch.load("expertnet/zh/factorized_v3_deploy.pt", map_location="cuda",
                  weights_only=False)
model = FactorizedNet(**ckpt["config"]).cuda().eval()
model.load_state_dict(ckpt["state"])
x, lengths = line_tensor(enc, [r[0] for r in test],
                         "expertnet/zh/ensemble_lines.npz")
x, lengths = x.cuda(), lengths.cuda()
with torch.no_grad():
    out = model(x, lengths)
    fixed_logits = out["fixed"] + out["fixed_local"]
    probs = F.softmax(fixed_logits, dim=-1)
    fbeats = ckpt["fixed_beats"]
    beat_names = sorted(set(fbeats))
    beat_idx = {b: i for i, b in enumerate(beat_names)}
    beat_prob = torch.zeros(probs.size(0), len(beat_names), device="cuda")
    beat_of = torch.tensor([beat_idx[b] for b in beats], device="cuda")
    beat_prob.scatter_add_(1, beat_of.unsqueeze(0).expand(probs.size(0), -1),
                           probs)

    opt = F.normalize(enc.embed([texts[i] for i in valid], batch=256),
                      dim=-1).cuda()
    data = torch.load("expertnet/zh/decision_states_test.pt",
                      map_location="cpu", weights_only=False)
    state = data["state"].cuda()
    proj = nn.Linear(opt.shape[1], state.shape[1], bias=False).cuda()
    proj.load_state_dict(torch.load("expertnet/zh/decision_head.pt",
                                    map_location="cuda",
                                    weights_only=False)["state"])
    dec = state @ proj(opt).T
    dec = (dec - dec.mean(1, keepdim=True)) / dec.std(1, keepdim=True)

    beat_prob_v = beat_prob[:, [beat_idx[beats[i]] for i in valid]]
    logp = torch.log(beat_prob_v + 1e-8)
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        total = logp + alpha * dec
        pred = total.argmax(1)
        chose = [valid[p] for p in pred.tolist()]
        exact = sum(int(c == g) for c, g in zip(chose, gold)) / len(gold)
        beat_acc = sum(int(beats[c] == b) for c, b in zip(chose, gold_beat)) \
            / len(gold)
        print(f"alpha {alpha}: beat {beat_acc:.3f} exact {exact:.3f}")

    pred = dec.argmax(1)
    chose = [valid[p] for p in pred.tolist()]
    print("decision only: beat",
          round(sum(int(beats[c] == b) for c, b in zip(chose, gold_beat))
                / len(gold), 3),
          "exact", round(sum(int(c == g) for c, g in zip(chose, gold))
                         / len(gold), 3))
