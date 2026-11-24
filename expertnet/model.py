"""A hand-written expert-form network: gated experts + a recurrent context cell.

It does not predict the next token. Each step maps a structured input (hashed text
features + NPC state) to a decision over a fixed label space. A small hand-written
GRU-style cell carries context between turns; a gate mixes several expert heads.

    x_t = [text features | state features]
    h_t = (1-z) * h_{t-1} + z * tanh(W_n [x_t, r * h_{t-1}])
    g   = softmax(W_g [x_t, h_t])
    out = sum_k g_k * expert_k([x_t, h_t])
"""
from __future__ import annotations

import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import INTENTS

FEAT_DIM = 2048
STATE_DIM = 4 + len(INTENTS) + 1


def _bucket(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) % FEAT_DIM


def featurize(text: str) -> torch.Tensor:
    vec = torch.zeros(FEAT_DIM)
    words = [w for w in "".join(
        c.lower() if (c.isalnum() or c in "' ") else " " for c in (text or "")
    ).split() if w]
    for word in words:
        vec[_bucket(word)] += 1.0
        padded = f"#{word}#"
        for i in range(len(padded) - 2):
            vec[_bucket("3:" + padded[i:i + 3])] += 0.5
    for a, b in zip(words, words[1:]):
        vec[_bucket(a + "_" + b)] += 1.0
    norm = vec.norm()
    return vec / norm if norm > 0 else vec


def state_vector(state: dict | None, last_intent: str | None) -> torch.Tensor:
    state = state or {}
    vec = torch.zeros(STATE_DIM)
    vec[0] = min(max(float(state.get("trust", 0)) / 100.0, 0.0), 1.0)
    vec[1] = 1.0 if state.get("guild") else 0.0
    vec[2] = min(float(state.get("turn", 0)) / 20.0, 1.0)
    vec[3] = min(len(state.get("revealed", [])) / 10.0, 1.0)
    if last_intent in INTENTS:
        vec[4 + INTENTS.index(last_intent)] = 1.0
    vec[-1] = 0.0 if last_intent in INTENTS else 1.0
    return vec


def encode(text: str, state: dict | None = None, last_intent: str | None = None) -> torch.Tensor:
    return torch.cat([featurize(text), state_vector(state, last_intent)])


class ExpertNet(nn.Module):
    def __init__(self, hidden: int = 64, experts: int = 6,
                 n_intents: int = len(INTENTS), n_risk: int = 2):
        super().__init__()
        inp = FEAT_DIM + STATE_DIM
        self.n_intents, self.n_risk, self.experts = n_intents, n_risk, experts
        self.wz = nn.Linear(inp + hidden, hidden)
        self.wr = nn.Linear(inp + hidden, hidden)
        self.wn = nn.Linear(inp + hidden, hidden)
        self.gate = nn.Linear(inp + hidden, experts)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(inp + hidden, 128), nn.ReLU(),
                          nn.Linear(128, n_intents + n_risk))
            for _ in range(experts)
        ])

    def step(self, x: torch.Tensor, h: torch.Tensor):
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.wz(xh))
        r = torch.sigmoid(self.wr(xh))
        n = torch.tanh(self.wn(torch.cat([x, r * h], dim=-1)))
        h = (1 - z) * h + z * n
        xh = torch.cat([x, h], dim=-1)
        gate = F.softmax(self.gate(xh), dim=-1)
        logits = torch.stack([head(xh) for head in self.heads], dim=1)
        mixed = (gate.unsqueeze(-1) * logits).sum(dim=1)
        return mixed[:, :self.n_intents], mixed[:, self.n_intents:], h, gate

    def forward(self, xs: torch.Tensor, h: torch.Tensor | None = None):
        """xs: [B, T, D] -> intent logits [B, T, I], risk logits, gates."""
        batch, steps, _ = xs.shape
        if h is None:
            h = torch.zeros(batch, self.wz.out_features)
        intents, risks, gates = [], [], []
        for t in range(steps):
            i, r, h, g = self.step(xs[:, t], h)
            intents.append(i)
            risks.append(r)
            gates.append(g)
        return torch.stack(intents, 1), torch.stack(risks, 1), torch.stack(gates, 1), h


class Session:
    """Carries the recurrent context between turns for one player."""

    def __init__(self, model: ExpertNet):
        self.model = model
        self.model.eval()
        self.h = torch.zeros(1, model.wz.out_features)
        self.last_intent = None

    def reset(self) -> None:
        self.h = torch.zeros(1, self.model.wz.out_features)
        self.last_intent = None

    def predict(self, text: str, state: dict | None = None) -> tuple[str, float, int]:
        x = encode(text, state, self.last_intent).unsqueeze(0)
        with torch.no_grad():
            intent_logits, _, h, gate = self.model.step(x, self.h)
        self.h = h
        probs = F.softmax(intent_logits, dim=-1)[0]
        idx = int(probs.argmax())
        self.last_intent = INTENTS[idx]
        return INTENTS[idx], float(probs[idx]), int(gate[0].argmax())


def load(path: str) -> ExpertNet:
    ckpt = torch.load(path, map_location="cpu")
    model = ExpertNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model


def save(model: ExpertNet, path: str) -> None:
    config = {"hidden": model.wz.out_features, "experts": model.experts,
              "n_intents": model.n_intents, "n_risk": model.n_risk}
    torch.save({"config": config, "state": model.state_dict()}, path)
