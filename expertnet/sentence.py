"""Sentence-level black boxes: one sentence in, one vector out.

Two interchangeable front ends:

* ``HFSentenceEncoder``  - a frozen MiniLM (Sentence-BERT family). The sentence is
  embedded before the recurrent loop, so the trainable part only ever sees 384-d
  vectors. Same idea as Sentence-BERT / SimCSE: a sentence is a point in space.
* ``TrainedSentenceEncoder`` - a hand-written character capsule. Character
  embeddings go through one self-attention block and an attention pool; the pooled
  vector is the sentence. Trained jointly with the expert net, and it exposes the
  per-character attention weights, so the "black box" is at least inspectable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LEN = 72


def char_ids(text: str, max_len: int = MAX_LEN) -> list[int]:
    """Printable characters -> ids 1..127, padded with 0. No vocabulary to train."""
    ids = [0] * max_len
    for i, ch in enumerate((text or "").lower()[:max_len]):
        ids[i] = ord(ch) % 127 + 1
    return ids


class HFSentenceEncoder:
    """Frozen MiniLM with mean pooling and L2 normalisation.

    Vectors are cached per string, so repeated turns and attacker variants cost one
    forward pass each at most.
    """

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str | None = None, pooling: str = "mean",
                 max_length: int = 96, device: str = "cpu"):
        import os

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.pooling = pooling
        self.max_length = max_length
        self.device = device
        # Prefer the local cache: a hub check per process is slow and fails offline.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                name, cache_dir=cache_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(
                name, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir)
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def embed(self, texts: list[str], batch: int = 64) -> torch.Tensor:
        todo: list[str] = []
        for text in texts:
            if text not in self._cache and text not in todo:
                todo.append(text)
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            encoded = self.tokenizer(chunk, padding=True, truncation=True,
                                     max_length=self.max_length,
                                     return_tensors="pt")
            encoded = {key: value.to(self.device)
                       for key, value in encoded.items()}
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            if self.pooling == "cls":
                vec = hidden[:, 0]
            else:
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            vec = F.normalize(vec, dim=-1)
            for text, row in zip(chunk, vec):
                self._cache[text] = row.detach().cpu()
        return torch.stack([self._cache[text] for text in texts])


class TrainedSentenceEncoder(nn.Module):
    """Character capsule: embed -> one self-attention block -> attention pool.

    The output is L2-normalised, so downstream layers see directions not lengths.
    ``forward`` also returns the pool weights over characters.
    """

    def __init__(self, out_dim: int = 96, char_dim: int = 32, max_len: int = MAX_LEN):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(128, char_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, char_dim)
        self.q = nn.Linear(char_dim, char_dim, bias=False)
        self.k = nn.Linear(char_dim, char_dim, bias=False)
        self.v = nn.Linear(char_dim, char_dim, bias=False)
        self.norm = nn.LayerNorm(char_dim)
        self.pool = nn.Linear(char_dim, 1)
        self.proj = nn.Linear(char_dim, out_dim)

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([char_ids(t, self.max_len) for t in texts], dtype=torch.long)

    def forward(self, ids: torch.Tensor):
        mask = ids != 0
        emb = self.embed(ids) + self.pos.weight[:ids.size(1)].unsqueeze(0)
        scores = (self.q(emb) @ self.k(emb).transpose(-1, -2)) / (emb.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        emb = self.norm(emb + F.softmax(scores, dim=-1) @ self.v(emb))
        weights = F.softmax(self.pool(emb).squeeze(-1).masked_fill(~mask, -1e9), dim=-1)
        pooled = (weights.unsqueeze(-1) * emb).sum(1)
        return F.normalize(self.proj(pooled), dim=-1), weights
