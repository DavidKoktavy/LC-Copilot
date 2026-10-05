"""Vector + keyword retrieval with Reciprocal Rank Fusion and metadata filtering.

``InMemoryHybridStore`` is the zero-dependency default (numpy cosine + BM25).
``AzureAISearchStore`` runs the same hybrid query server-side on Azure AI Search
(keyword + vector in one request, RRF fused by the service). Both satisfy ``VectorStore``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from ..resilience import PermanentError, RetryPolicy, TransientError, call_with_retry
from .chunking import Chunk
from .embeddings import Embedder
from .text import tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[Chunk]) -> int: ...
    async def search(self, query: str, k: int, filters: dict[str, list[str]] | None = None) -> list[Hit]: ...


def _passes(chunk: Chunk, filters: dict[str, list[str]] | None) -> bool:
    """A chunk passes when, for every filter key it *has*, its value is in the allowed list.

    Chunks without the key (e.g. bank-wide policies have no ``lc_number``) always pass, so a
    question about LC-2026-0187 still sees the policies but never another credit's terms.
    """
    if not filters:
        return True
    for key, allowed in filters.items():
        if allowed and key in chunk.metadata and chunk.metadata[key] not in allowed:
            return False
    return True


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(
                self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf
            )
        return out


class InMemoryHybridStore:
    def __init__(self, embedder: Embedder, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self.bm25: BM25 | None = None

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        by_id = {c.chunk_id: c for c in self.chunks}
        by_id.update({c.chunk_id: c for c in chunks})  # idempotent re-ingestion
        self.chunks = list(by_id.values())
        self.vectors = await self.embedder.embed([c.indexed_text for c in self.chunks])
        self.bm25 = BM25([tokens(c.indexed_text) for c in self.chunks])
        return len(chunks)

    async def search(self, query: str, k: int, filters: dict[str, list[str]] | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        qv = (await self.embedder.embed([query]))[0]
        dense = self.vectors @ qv
        sparse = self.bm25.scores(tokens(query))
        allowed = [i for i, c in enumerate(self.chunks) if _passes(c, filters)]
        d_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -dense[i]), 1)}
        s_rank = {
            i: r for r, i in enumerate(sorted((i for i in allowed if sparse[i] > 0), key=lambda i: -sparse[i]), 1)
        }
        fused = {
            i: 1 / (self.rrf_k + d_rank[i]) + (1 / (self.rrf_k + s_rank[i]) if i in s_rank else 0.0)
            for i in allowed
        }
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.chunks[i], round(fused[i], 5), d_rank.get(i), s_rank.get(i)) for i in top]


class AzureAISearchStore:
    """Hybrid (BM25 + HNSW vector) search on Azure AI Search via REST.

    Index schema: see ``infra/azure-search-index.json``.
    """

    API = "2024-07-01"

    def __init__(self, endpoint: str, api_key: str, index: str, embedder: Embedder) -> None:
        self.base = f"{endpoint.rstrip('/')}/indexes/{index}/docs"
        self.headers = {"api-key": api_key, "Content-Type": "application/json"}
        self.embedder = embedder
        self.policy = RetryPolicy(attempts=3, timeout_s=10)

    async def _post(self, path: str, body: dict) -> dict:
        async def go() -> dict:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(f"{self.base}/{path}?api-version={self.API}", headers=self.headers, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                raise TransientError(f"azure search {r.status_code}")
            if r.status_code >= 400:
                raise PermanentError(f"azure search {r.status_code}: {r.text[:300]}")
            return r.json()

        return await call_with_retry(go, self.policy)

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        vecs = await self.embedder.embed([c.indexed_text for c in chunks])
        docs = [
            {
                "@search.action": "mergeOrUpload",
                "id": c.chunk_id.replace("#", "_"),
                "doc_id": c.doc_id,
                "title": c.title,
                "section": c.section,
                "content": c.text,
                "lc_number": c.metadata.get("lc_number", ""),
                "embedding": v.tolist(),
            }
            for c, v in zip(chunks, vecs)
        ]
        for i in range(0, len(docs), 500):
            await self._post("index", {"value": docs[i : i + 500]})
        return len(docs)

    async def search(self, query: str, k: int, filters: dict[str, list[str]] | None = None) -> list[Hit]:
        qv = (await self.embedder.embed([query]))[0]
        body: dict = {
            "search": query,
            "top": k,
            "select": "id,doc_id,title,section,content,lc_number",
            "vectorQueries": [{"kind": "vector", "vector": qv.tolist(), "fields": "embedding", "k": k * 4}],
        }
        if filters and (lcs := filters.get("lc_number")):
            allowed = " or ".join(f"lc_number eq '{x}'" for x in lcs if x.replace("-", "").isalnum())
            body["filter"] = f"(lc_number eq '' or {allowed})"
        data = await self._post("search", body)
        return [
            Hit(
                Chunk(
                    d["id"].replace("_", "#"), d["doc_id"], d["title"], d["section"], d["content"],
                    {"lc_number": d["lc_number"]} if d.get("lc_number") else {},
                ),
                d.get("@search.score", 0.0),
            )
            for d in data.get("value", [])
        ]
