"""Composition root: builds every dependency once from settings."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings
from .data.db import ReadOnlyDB, ensure_db
from .llm import LLMClient, build_llm
from .rag.ingest import ingest
from .rag.retriever import Retriever
from .rag.store import AzureAISearchStore, InMemoryHybridStore, VectorStore
from .tools import ToolRegistry


@dataclass
class Services:
    settings: Settings
    llm: LLMClient
    store: VectorStore
    retriever: Retriever
    db: ReadOnlyDB
    tools: ToolRegistry
    n_chunks: int


async def build_services(s: Settings | None = None, scopes: set[str] | None = None) -> Services:
    s = s or get_settings()
    llm, embedder = build_llm(s)
    if s.vector_backend == "azure_ai_search":
        store: VectorStore = AzureAISearchStore(s.azure_search_endpoint, s.azure_search_api_key,
                                                s.azure_search_index, embedder)
    else:
        store = InMemoryHybridStore(embedder)
    n = await ingest(store, s)
    ensure_db(s.db_path, s.seed_sql)
    retriever = Retriever(store, s.top_k)
    db = ReadOnlyDB(s.db_path, s.sql_timeout_s, s.sql_max_rows)
    # the agent itself may draft notices; sending still requires a human approval in the graph
    agent_scopes = scopes if scopes is not None else {"docs:read", "data:read", "notices:write"}
    return Services(s, llm, store, retriever, db, ToolRegistry(s, retriever, db, agent_scopes), n)
