"""Task-typed model interface.

The graph never sends free-form prompts; it calls typed tasks (route, answer, text-to-SQL, ...)
that return validated pydantic objects. That gives one place to version prompts, one place to
enforce structured output, and lets the offline baseline implement the exact same contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from ..rag.chunking import Chunk

Route = Literal["docs", "data", "hybrid", "action", "out_of_scope"]


class RouteDecision(BaseModel):
    route: Route
    reason: str = Field(description="one short sentence")


class SqlPlan(BaseModel):
    sql: str | None = Field(description="a single SQLite SELECT, or null if the question cannot be answered")
    rationale: str = ""


class LLMClient(ABC):
    name: str = "abstract"

    @abstractmethod
    async def route(self, question: str) -> RouteDecision: ...

    @abstractmethod
    async def answer_from_context(self, question: str, contexts: list[Chunk], strict: bool = False) -> str:
        """Answer using only ``contexts``; cite with [n] (1-based). Return INSUFFICIENT_CONTEXT if unanswerable."""

    @abstractmethod
    async def text_to_sql(self, question: str, schema: str, previous_error: str | None = None) -> SqlPlan: ...

    @abstractmethod
    async def answer_from_rows(self, question: str, sql: str, rows: list[dict]) -> str: ...

    @abstractmethod
    async def draft_refusal_notice(
        self, presentation_id: str, discrepancies: list[dict], policy: list[Chunk]
    ) -> str: ...


INSUFFICIENT = "INSUFFICIENT_CONTEXT"
