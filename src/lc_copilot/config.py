"""Centralised, typed configuration. Every knob is an env var (prefix ``LC_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LC_", env_file=".env", extra="ignore")

    # --- providers -------------------------------------------------------
    # "offline" is deterministic and needs no keys: CI, demos and evals run on it.
    llm_provider: Literal["offline", "azure_openai", "openai"] = "offline"
    vector_backend: Literal["memory", "azure_ai_search"] = "memory"

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index: str = "lc-copilot"

    azure_docintel_endpoint: str = ""
    azure_docintel_api_key: str = ""

    azure_content_safety_endpoint: str = ""
    azure_content_safety_api_key: str = ""

    # --- data ------------------------------------------------------------
    docs_dir: Path = ROOT / "data" / "docs"
    seed_sql: Path = ROOT / "data" / "seed.sql"
    db_path: Path = ROOT / "data" / "portfolio.db"

    # --- retrieval -------------------------------------------------------
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 5
    min_grounding_score: float = 0.5

    # --- safety / resilience --------------------------------------------
    sql_max_rows: int = 200
    sql_timeout_s: float = 2.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    api_rate_limit_per_min: int = 60
    mcp_scopes: str = "docs:read,data:read"

    # --- observability ---------------------------------------------------
    trace_dir: Path = ROOT / "runs"
    otel_enabled: bool = False

    # USD per 1M tokens (input, output) — used by the eval harness for cost.
    price_in_per_m: float = 0.15
    price_out_per_m: float = 0.60


@lru_cache
def get_settings() -> Settings:
    return Settings()
