"""Tool registry with explicit contracts.

One registry backs both the LangGraph agent and the MCP server, so the guarantees are identical
regardless of who calls a tool:
  - typed input (pydantic) → bad arguments fail fast with a clear error, never reach the backend
  - scope check (least privilege) → a "docs:read" client cannot touch the database
  - per-tool rate limit, timeout and retry policy (retry only for idempotent tools)
  - idempotency keys for side-effecting tools
  - audit log (JSONL) with a hash of the arguments, outcome and latency for every call
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .data.db import ReadOnlyDB
from .observability import current_trace_id, span
from .rag.retriever import Retriever
from .resilience import (
    IdempotencyStore,
    KeyedRateLimiter,
    PermanentError,
    RetryPolicy,
    call_with_retry,
    fingerprint,
)


class ToolError(PermanentError):
    pass


# ----------------------------------------------------------------- contracts
class SearchDocumentsIn(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class QueryPortfolioIn(BaseModel):
    sql: str = Field(min_length=6, max_length=4000, description="one read-only SQLite SELECT statement")


class GetDiscrepanciesIn(BaseModel):
    presentation_id: str = Field(pattern=r"^PRS-\d{4}-\d{2}$")


class SubmitRefusalNoticeIn(BaseModel):
    presentation_id: str = Field(pattern=r"^PRS-\d{4}-\d{2}$")
    notice_text: str = Field(min_length=20, max_length=8000)
    approved_by: str = Field(min_length=2, max_length=100)
    idempotency_key: str = Field(min_length=8, max_length=128)


@dataclass
class ToolSpec:
    name: str
    description: str
    scope: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Awaitable[Any]]
    timeout_s: float = 10.0
    idempotent: bool = True  # only idempotent tools are retried automatically


class ToolRegistry:
    def __init__(self, s: Settings, retriever: Retriever, db: ReadOnlyDB, scopes: set[str] | None = None) -> None:
        self.s = s
        self.retriever = retriever
        self.db = db
        self.scopes = scopes if scopes is not None else set(s.mcp_scopes.split(","))
        self.rate = KeyedRateLimiter(per_minute=120)
        self.idem = IdempotencyStore()
        self.audit_path: Path = s.trace_dir / "tool_audit.jsonl"
        self.outbox_path: Path = s.trace_dir / "outbox.jsonl"
        self.specs: dict[str, ToolSpec] = {}
        for spec in self._all_specs():
            if spec.scope in self.scopes:  # tools outside the granted scopes are not even registered
                self.specs[spec.name] = spec

    def _all_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("search_documents", "Hybrid search over credit terms, examiner notes and bank policies. "
                     "Returns passages with stable chunk ids for citation.", "docs:read",
                     SearchDocumentsIn, self._search, timeout_s=10),
            ToolSpec("query_portfolio", "Run ONE read-only SELECT on the trade-finance portfolio "
                     "(letters_of_credit, presentations, discrepancies, counterparties). PII columns are blocked.",
                     "data:read", QueryPortfolioIn, self._query, timeout_s=self.s.sql_timeout_s + 1),
            ToolSpec("get_discrepancies", "List discrepancies recorded for one presentation.", "data:read",
                     GetDiscrepanciesIn, self._discrepancies, timeout_s=5),
            ToolSpec("submit_refusal_notice", "Queue an APPROVED refusal notice for sending. Requires a human "
                     "approver and an idempotency key.", "notices:write", SubmitRefusalNoticeIn, self._submit,
                     timeout_s=5, idempotent=False),
        ]

    # ------------------------------------------------------------- handlers
    async def _search(self, a: SearchDocumentsIn) -> list[dict]:
        hits = await self.retriever.retrieve(a.query, a.top_k)
        return [
            {"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "section": h.chunk.section,
             "text": h.chunk.text, "score": h.score}
            for h in hits
        ]

    async def _query(self, a: QueryPortfolioIn) -> dict:
        return (await self.db.query(a.sql)).as_dict()

    async def _discrepancies(self, a: GetDiscrepanciesIn) -> list[dict]:
        # parameter-free, server-built SQL: the id is pattern-validated above
        res = await self.db.query(
            "SELECT discrepancy_id, code, description, waived FROM discrepancies "
            f"WHERE presentation_id = '{a.presentation_id}' ORDER BY discrepancy_id"
        )
        return res.rows

    async def _submit(self, a: SubmitRefusalNoticeIn) -> dict:
        async def send() -> dict:
            record = {
                "message_id": f"MSG-{fingerprint(a.idempotency_key)[:12]}",
                "presentation_id": a.presentation_id,
                "approved_by": a.approved_by,
                "queued_at": datetime.now(UTC).isoformat(),
                "text": a.notice_text,
            }
            self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
            with self.outbox_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            return {"message_id": record["message_id"], "status": "queued"}

        result, replayed = await self.idem.run_once(a.idempotency_key, a.model_dump(), send)
        return result | {"replayed": replayed}

    # ------------------------------------------------------------- invocation
    def list(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "scope": t.scope,
             "input_schema": t.input_model.model_json_schema()}
            for t in self.specs.values()
        ]

    async def invoke(self, name: str, args: dict, caller: str = "agent") -> Any:
        spec = self.specs.get(name)
        if spec is None:
            self._audit(caller, name, args, "denied", 0, "tool not available for granted scopes")
            raise ToolError(f"tool '{name}' is not available (granted scopes: {sorted(self.scopes)})")
        t0 = time.perf_counter()
        outcome, err = "ok", ""
        with span(f"tool.{name}", caller=caller) as sp:
            try:
                await self.rate.check(f"{caller}:{name}")
                try:
                    parsed = spec.input_model.model_validate(args)
                except ValidationError as e:
                    raise ToolError(f"invalid arguments for {name}: {e.errors()[0]['msg']}") from e
                policy = RetryPolicy(attempts=3 if spec.idempotent else 1, timeout_s=spec.timeout_s)
                return await call_with_retry(lambda: spec.handler(parsed), policy)
            except Exception as e:
                outcome, err = "error", f"{type(e).__name__}: {e}"
                sp.set(error=err)
                raise
            finally:
                self._audit(caller, name, args, outcome, (time.perf_counter() - t0) * 1000, err)

    def _audit(self, caller: str, tool: str, args: dict, outcome: str, ms: float, err: str) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(UTC).isoformat(),
            "trace_id": current_trace_id(),
            "caller": caller,
            "tool": tool,
            "args_sha256": fingerprint(args),
            "args_preview": {k: (str(v)[:120]) for k, v in args.items() if k != "notice_text"},
            "outcome": outcome,
            "error": err[:300],
            "latency_ms": round(ms, 2),
        }
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
