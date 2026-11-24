"""One adapter per architecture, so train / probe / ablation share the same calls.

``v1``  hashed word n-grams       (model.py)
``v2h`` frozen MiniLM sentence    (v2.py, 384-d, no gradient)
``v2t`` trained character capsule (v2.py, 96-d, trained jointly)

Each adapter hides the tensor shapes: sessions -> batches, texts -> one-step
logits, plus save/load that records the architecture in the checkpoint.
"""
from __future__ import annotations

import torch

from .agents import session_tensors
from .model import ExpertNet, encode, state_vector
from .sentence import HFSentenceEncoder
from .v2 import ExpertNetV2

DEFAULT_STATE = {"trust": 35}


def _state_rows(rows) -> torch.Tensor:
    return torch.stack([
        state_vector({"trust": trust, "turn": t + 1}, last)
        for _, _, _, trust, t, last in rows
    ])


class V1Arch:
    kind = "v1"

    def __init__(self, experts: int = 6, hidden: int = 64):
        self.model = ExpertNet(hidden=hidden, experts=experts)

    def batches(self, sessions) -> dict:
        xs, yi, yr = session_tensors(sessions)
        return {"xs": xs, "yi": yi, "yr": yr}

    def forward_batch(self, model, batch):
        intents, risks, gates, _ = model(batch["xs"])
        return intents, risks, gates

    def text_tensors(self, texts, states=None) -> dict:
        states = states or [DEFAULT_STATE] * len(texts)
        return {"xs": torch.stack([encode(t, s, None)
                                   for t, s in zip(texts, states)]).unsqueeze(1)}

    def forward_texts(self, model, tensors):
        intents, risks, gates, _ = model(tensors["xs"])
        return intents[:, 0], risks[:, 0], gates[:, 0]

    def step(self, model, text, state=None, last_intent=None, h=None):
        x = encode(text, state or DEFAULT_STATE, last_intent).unsqueeze(0)
        intents, _, h, gate = model.step(x, h if h is not None
                                         else torch.zeros(1, model.wz.out_features))
        return intents, h, gate

    def config(self) -> dict:
        return {"hidden": self.model.wz.out_features, "experts": self.model.experts,
                "n_intents": self.model.n_intents, "n_risk": self.model.n_risk}


class V2HFArch:
    kind = "v2h"

    def __init__(self, experts: int = 6, hidden: int = 64,
                 encoder: HFSentenceEncoder | None = None):
        self.hf = encoder or HFSentenceEncoder()
        self.model = ExpertNetV2(text_dim=self.hf.dim, hidden=hidden, experts=experts)

    def batches(self, sessions) -> dict:
        rows = [row for rows in sessions for row in rows]
        batch, steps = len(sessions), len(sessions[0])
        vectors = self.hf.embed([row[0] for row in rows]).view(batch, steps, -1)
        return {"xs": vectors, "state": _state_rows(rows).view(batch, steps, -1),
                "yi": torch.tensor([row[1] for row in rows]).view(batch, steps),
                "yr": torch.tensor([row[2] for row in rows]).view(batch, steps)}

    def forward_batch(self, model, batch):
        intents, risks, gates, _ = model(xs=batch["xs"], state=batch["state"])
        return intents, risks, gates

    def text_tensors(self, texts, states=None) -> dict:
        states = states or [DEFAULT_STATE] * len(texts)
        return {"xs": self.hf.embed(texts).unsqueeze(1),
                "state": torch.stack([state_vector(s, None) for s in states]).unsqueeze(1)}

    def forward_texts(self, model, tensors):
        intents, risks, gates, _ = model(xs=tensors["xs"], state=tensors["state"])
        return intents[:, 0], risks[:, 0], gates[:, 0]

    def step(self, model, text, state=None, last_intent=None, h=None):
        xs = self.hf.embed([text]).unsqueeze(0)
        st = state_vector(state or DEFAULT_STATE, last_intent).reshape(1, 1, -1)
        intents, _, gates, h = model(xs=xs, state=st, h=h)
        return intents[:, 0], h, gates[:, 0]

    def config(self) -> dict:
        return {**self.model.config(), "hf_model": self.hf.name}


class V2TArch:
    kind = "v2t"

    def __init__(self, experts: int = 6, hidden: int = 64, text_dim: int = 96):
        self.model = ExpertNetV2(text_dim=text_dim, hidden=hidden, experts=experts,
                                 trained_encoder=True)

    def batches(self, sessions) -> dict:
        rows = [row for rows in sessions for row in rows]
        batch, steps = len(sessions), len(sessions[0])
        tokens = self.model.encoder.tokenize([row[0] for row in rows])
        return {"tokens": tokens.view(batch, steps, -1),
                "state": _state_rows(rows).view(batch, steps, -1),
                "yi": torch.tensor([row[1] for row in rows]).view(batch, steps),
                "yr": torch.tensor([row[2] for row in rows]).view(batch, steps)}

    def forward_batch(self, model, batch):
        intents, risks, gates, _ = model(tokens=batch["tokens"], state=batch["state"])
        return intents, risks, gates

    def text_tensors(self, texts, states=None) -> dict:
        states = states or [DEFAULT_STATE] * len(texts)
        return {"tokens": self.model.encoder.tokenize(texts).unsqueeze(1),
                "state": torch.stack([state_vector(s, None) for s in states]).unsqueeze(1)}

    def forward_texts(self, model, tensors):
        intents, risks, gates, _ = model(tokens=tensors["tokens"],
                                         state=tensors["state"])
        return intents[:, 0], risks[:, 0], gates[:, 0]

    def step(self, model, text, state=None, last_intent=None, h=None):
        tokens = self.model.encoder.tokenize([text]).unsqueeze(0)
        st = state_vector(state or DEFAULT_STATE, last_intent).reshape(1, 1, -1)
        intents, _, gates, h = model(tokens=tokens, state=st, h=h)
        return intents[:, 0], h, gates[:, 0]

    def config(self) -> dict:
        return {**self.model.config(), "text_dim": self.model.text_dim}


def make_arch(kind: str, experts: int = 6, hidden: int = 64,
              hf: HFSentenceEncoder | None = None) -> V1Arch | V2HFArch | V2TArch:
    if kind == "v1":
        return V1Arch(experts=experts, hidden=hidden)
    if kind == "v2h":
        return V2HFArch(experts=experts, hidden=hidden, encoder=hf)
    if kind == "v2t":
        return V2TArch(experts=experts, hidden=hidden)
    raise ValueError(f"unknown architecture: {kind}")


def save_arch(arch, path: str) -> None:
    torch.save({"config": {**arch.config(), "arch": arch.kind},
                "state": arch.model.state_dict()}, path)


def load_arch(path: str, encoder: HFSentenceEncoder | None = None):
    """Checkpoints from v1 carry no ``arch`` key, so they still load as v1."""
    ckpt = torch.load(path, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    kind = config.pop("arch", "v1")
    if kind == "v1":
        arch = V1Arch(experts=config.get("experts", 6), hidden=config.get("hidden", 64))
    elif kind == "v2h":
        arch = V2HFArch(experts=config.get("experts", 6), hidden=config.get("hidden", 64),
                        encoder=encoder)
    else:
        arch = V2TArch(experts=config.get("experts", 6), hidden=config.get("hidden", 64),
                       text_dim=config.get("text_dim", 96))
    arch.model.load_state_dict(ckpt["state"])
    return arch
