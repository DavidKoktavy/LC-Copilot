"""Retrieval front door: entity-aware filtering + hybrid search + tracing."""

from __future__ import annotations

from ..observability import span
from .store import Hit, VectorStore
from .text import entity_ids


class Retriever:
    def __init__(self, store: VectorStore, top_k: int = 5) -> None:
        self.store = store
        self.top_k = top_k

    async def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        # Self-query style filtering: identifiers in the question constrain the search space,
        # so "partial shipments under LC-2026-0187" can't be answered from LC-2026-0142's terms.
        filters = {key: vals for key, vals in entity_ids(query).items() if vals}
        with span("retrieval.search", query=query, filters=str(filters)) as sp:
            hits = await self.store.search(query, k or self.top_k, filters or None)
            sp.set(n_hits=len(hits), top=[h.chunk.chunk_id for h in hits])
        return hits
