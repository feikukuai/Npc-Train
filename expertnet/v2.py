"""v2 of the expert-form network: a sentence vector instead of hashed word features.

    s_t = SentenceEncoder(text_t)            # 384-d frozen MiniLM or 96-d capsule
    x_t = [s_t | state_t]
    h_t = (1 - z) * h_{t-1} + z * tanh(W_n [x_t, r * h_{t-1}])
    g   = softmax(W_g [x_t, h_t])
    out = sum_k g_k * expert_k([x_t, h_t])

The internal work of the sentence (attention over characters or the pretrained
transformer) is deliberately hidden behind one vector, exactly as requested: the
outer network only sees "this sentence means this direction in space".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import STATE_DIM, state_vector
from .schema import INTENTS
from .sentence import HFSentenceEncoder, TrainedSentenceEncoder


class ExpertNetV2(nn.Module):
    def __init__(self, text_dim: int = 384, hidden: int = 64, experts: int = 6,
                 n_intents: int = len(INTENTS), n_risk: int = 2,
                 trained_encoder: bool = False):
        super().__init__()
        self.text_dim, self.hidden = text_dim, hidden
        self.experts, self.n_intents, self.n_risk = experts, n_intents, n_risk
        self.encoder = TrainedSentenceEncoder(out_dim=text_dim) if trained_encoder else None
        inp = text_dim + STATE_DIM
        self.wz = nn.Linear(inp + hidden, hidden)
        self.wr = nn.Linear(inp + hidden, hidden)
        self.wn = nn.Linear(inp + hidden, hidden)
        self.gate = nn.Linear(inp + hidden, experts)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(inp + hidden, 128), nn.ReLU(),
                          nn.Linear(128, n_intents + n_risk))
            for _ in range(experts)
        ])

    def _sentences(self, xs, tokens):
        if self.encoder is not None:
            if tokens is None:
                raise ValueError("this model needs character tokens")
            batch, steps, length = tokens.shape
            vectors, _ = self.encoder(tokens.reshape(-1, length))
            return vectors.view(batch, steps, -1)
        if xs is None:
            raise ValueError("this model needs precomputed sentence vectors")
        return xs

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

    def forward(self, xs: torch.Tensor | None = None, state: torch.Tensor | None = None,
                tokens: torch.Tensor | None = None, h: torch.Tensor | None = None):
        """Returns intent logits [B,T,I], risk logits [B,T,2], gates [B,T,K], h."""
        sentences = self._sentences(xs, tokens)
        x = torch.cat([sentences, state], dim=-1)
        batch, steps, _ = x.shape
        if h is None:
            h = torch.zeros(batch, self.hidden)
        intents, risks, gates = [], [], []
        for t in range(steps):
            i, r, h, g = self.step(x[:, t], h)
            intents.append(i)
            risks.append(r)
            gates.append(g)
        return torch.stack(intents, 1), torch.stack(risks, 1), torch.stack(gates, 1), h

    def config(self) -> dict:
        return {"text_dim": self.text_dim, "hidden": self.hidden,
                "experts": self.experts, "n_intents": self.n_intents,
                "n_risk": self.n_risk, "trained_encoder": self.encoder is not None}


class SessionV2:
    """Carries ``h_t`` between turns for one player."""

    def __init__(self, model: ExpertNetV2, hf: HFSentenceEncoder | None = None):
        self.model = model.eval()
        if model.encoder is None and hf is None:
            hf = HFSentenceEncoder()
        self.hf = hf
        self.reset()

    def reset(self) -> None:
        self.h = torch.zeros(1, self.model.hidden)
        self.last_intent = None

    def predict(self, text: str, state: dict | None = None) -> tuple[str, float, int]:
        st = state_vector(state, self.last_intent).reshape(1, 1, -1)
        with torch.no_grad():
            if self.model.encoder is not None:
                tokens = self.model.encoder.tokenize([text]).unsqueeze(0)
                logits, _, gate, h = self.model(tokens=tokens, state=st, h=self.h)
            else:
                vec = self.hf.embed([text]).unsqueeze(0)
                logits, _, gate, h = self.model(xs=vec, state=st, h=self.h)
        probs = F.softmax(logits[0, 0], dim=-1)
        idx = int(probs.argmax())
        self.h, self.last_intent = h, INTENTS[idx]
        return INTENTS[idx], float(probs[idx]), int(gate[0, 0].argmax())


def save_v2(model: ExpertNetV2, path: str) -> None:
    torch.save({"config": model.config(), "state": model.state_dict()}, path)


def load_v2(path: str, hf: HFSentenceEncoder | None = None) -> ExpertNetV2:
    ckpt = torch.load(path, map_location="cpu")
    model = ExpertNetV2(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model
