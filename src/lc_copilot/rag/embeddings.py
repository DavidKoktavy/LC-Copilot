"""Embedding providers.

``HashingEmbedder`` is a deterministic, offline feature-hashing embedder (word unigrams,
bigrams and character 4-grams). It is *not* semantic — it exists so retrieval, tests and the
eval regression suite run in CI without keys. Production uses Azure OpenAI / OpenAI embeddings.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .text import tokens


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _h(self, feat: str) -> tuple[int, float]:
        d = hashlib.blake2b(feat.encode(), digest_size=8).digest()
        v = int.from_bytes(d, "little")
        return v % self.dim, 1.0 if (v >> 63) & 1 else -1.0

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        feats += [f"#{t[i:i + 4]}" for t in toks if len(t) > 4 for i in range(len(t) - 3)]
        for f in feats:
            i, s = self._h(f)
            vec[i] += s
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class RemoteEmbedder:
    """OpenAI-compatible embeddings (Azure OpenAI or OpenAI) — delegates to the LLM client's HTTP layer."""

    def __init__(self, client, dim: int = 1536) -> None:  # client: OpenAICompatibleLLM
        self.client = client
        self.dim = dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            out.extend(await self.client.embeddings(texts[i : i + 64]))
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)
