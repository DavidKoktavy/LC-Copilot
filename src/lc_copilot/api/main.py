"""Production-style HTTP surface.

- ``POST /v1/ask``                         Talk2Document / Talk2Data / hybrid Q&A (Idempotency-Key aware)
- ``POST /v1/threads/{id}/approval``       human-in-the-loop decision for paused runs
- ``GET  /v1/tools``                       tool contracts (same registry the MCP server exposes)
- ``GET  /.well-known/agent-card.json``    A2A agent card;  ``POST /a2a`` minimal JSON-RPC ``message/send``
- ``GET  /health`` · ``GET /ready``
Rate limited per client (X-Client-Id or IP); every response carries X-Request-Id and X-Trace-Id.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..agent.graph import Agent
from ..resilience import IdempotencyConflict, IdempotencyStore, KeyedRateLimiter, RateLimitedError
from ..services import build_services

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("lc_copilot.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["Are partial shipments allowed under LC-2026-0187?"])
    thread_id: str | None = Field(default=None, max_length=64)


class ApprovalRequest(BaseModel):
    approved: bool
    approver: str = Field(min_length=2, max_length=100)
    edited_text: str | None = Field(default=None, max_length=8000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = await build_services()
    app.state.svc = svc
    app.state.agent = Agent(svc)
    app.state.idem = IdempotencyStore()
    app.state.limiter = KeyedRateLimiter(svc.settings.api_rate_limit_per_min)
    log.info(json.dumps({"event": "startup", "chunks": svc.n_chunks, "llm": svc.llm.name}))
    yield


app = FastAPI(title="LC Copilot", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def edge(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    client = request.headers.get("X-Client-Id") or (request.client.host if request.client else "anon")
    t0 = time.perf_counter()
    if request.url.path.startswith(("/v1/", "/a2a")):
        try:
            await request.app.state.limiter.check(client)
        except RateLimitedError:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                                headers={"Retry-After": "1", "X-Request-Id": rid})
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    log.info(json.dumps({"event": "http", "rid": rid, "client": client, "method": request.method,
                         "path": request.url.path, "status": response.status_code,
                         "ms": round((time.perf_counter() - t0) * 1000, 1)}))
    return response


def _respond(result: dict) -> JSONResponse:
    code = 202 if result["status"] == "pending_approval" else 200
    headers = {"X-Trace-Id": result.get("meta", {}).get("trace_id", "")}
    return JSONResponse(result, status_code=code, headers=headers)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/ready")
async def ready(request: Request) -> dict:
    svc = request.app.state.svc
    return {"status": "ready", "chunks": svc.n_chunks, "llm": svc.llm.name,
            "vector_backend": svc.settings.vector_backend}


@app.post("/v1/ask")
async def ask(body: AskRequest, request: Request, idempotency_key: str | None = Header(default=None)) -> JSONResponse:
    agent: Agent = request.app.state.agent
    if not idempotency_key:
        return _respond(await agent.ask(body.question, body.thread_id))
    try:
        result, replayed = await request.app.state.idem.run_once(
            idempotency_key, body.model_dump(), lambda: agent.ask(body.question, body.thread_id)
        )
    except IdempotencyConflict as e:
        raise HTTPException(409, str(e)) from e
    resp = _respond(result)
    resp.headers["Idempotent-Replayed"] = str(replayed).lower()
    return resp


@app.post("/v1/threads/{thread_id}/approval")
async def approve(thread_id: str, body: ApprovalRequest, request: Request) -> JSONResponse:
    try:
        result = await request.app.state.agent.resume(thread_id, body.approved, body.approver, body.edited_text)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return _respond(result)


@app.get("/v1/tools")
async def tools(request: Request) -> list[dict]:
    return request.app.state.svc.tools.list()


# ----------------------------------------------------------------- A2A (agent-to-agent)
AGENT_CARD: dict[str, Any] = {
    "name": "lc-copilot",
    "description": "Answers grounded questions about documentary credits, presentations, discrepancies "
                   "and trade-operations policy. Read-only; refusal notices require human approval.",
    "version": __version__,
    "url": "/a2a",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["application/json", "text/plain"],
    "skills": [
        {"id": "talk2document", "name": "Grounded document Q&A", "tags": ["rag", "citations", "trade-finance"]},
        {"id": "talk2data", "name": "Safe portfolio analytics", "tags": ["text-to-sql", "read-only"]},
    ],
}


@app.get("/.well-known/agent-card.json")
async def agent_card() -> dict:
    return AGENT_CARD


@app.post("/a2a")
async def a2a(request: Request) -> JSONResponse:
    """Minimal JSON-RPC 2.0 ``message/send`` so other agents (e.g. an orchestrator) can delegate questions."""
    try:
        rpc = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
    rid = rpc.get("id")
    if rpc.get("method") != "message/send":
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}})
    msg = rpc.get("params", {}).get("message", {})
    text = " ".join(p.get("text", "") for p in msg.get("parts", []) if p.get("kind", "text") == "text").strip()
    if not text:
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "no text part"}})
    result = await request.app.state.agent.ask(text[:2000], msg.get("contextId"))
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": {
        "kind": "message",
        "role": "agent",
        "messageId": uuid.uuid4().hex,
        "contextId": result["thread_id"],
        "parts": [
            {"kind": "text", "text": result["answer"] or json.dumps(result.get("approval_request"))},
            {"kind": "data", "data": {k: result.get(k) for k in ("status", "route", "citations", "sql", "meta")}},
        ],
    }})
