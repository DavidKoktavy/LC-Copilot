"""MCP server exposing the trade-finance knowledge base and portfolio to any MCP client
(Claude Desktop, VS Code, Azure AI Foundry agents, another LangGraph agent, ...).

Least privilege: ``LC_MCP_SCOPES`` decides which tools exist at all. The default
(``docs:read,data:read``) never registers ``submit_refusal_notice`` — a client cannot call what it
cannot see. Every call goes through the same ToolRegistry as the agent, so it gets the same input
validation, SQL guard, rate limit, timeout and JSONL audit trail (``runs/tool_audit.jsonl``).

Run:  ``LC_MCP_SCOPES=docs:read python -m lc_copilot.mcp_server``   (stdio transport)
"""

from __future__ import annotations

import asyncio
from typing import Any

try:  # MCP Python SDK v2
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError as _SdkToolError
except ImportError:  # pragma: no cover — SDK v1
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]
    from mcp.server.fastmcp.exceptions import ToolError as _SdkToolError  # type: ignore[no-redef]

from .config import get_settings
from .data.sql_guard import schema_description
from .resilience import PermanentError, RateLimitedError
from .services import Services, build_services

_svc: Services | None = None
_lock = asyncio.Lock()


async def services() -> Services:
    global _svc
    async with _lock:
        if _svc is None:
            s = get_settings()
            _svc = await build_services(s, scopes=set(filter(None, s.mcp_scopes.split(","))))
    return _svc


async def _call(tool: str, **args: Any) -> Any:
    svc = await services()
    try:
        return await svc.tools.invoke(tool, args, caller="mcp")
    except (PermanentError, RateLimitedError) as e:
        # deliberate refusals (policy, validation, rate limit) are shown to the model so it can adapt;
        # unexpected crashes stay generic and server-side.
        raise _SdkToolError(str(e)) from e


def create_server(scopes: set[str] | None = None) -> _Server:
    scopes = scopes if scopes is not None else set(filter(None, get_settings().mcp_scopes.split(",")))
    server = _Server(
        "lc-copilot",
        instructions="Read-only access to documentary-credit terms, bank trade policies and the LC portfolio. "
                     "Cite chunk_id values when quoting search results.",
    )

    if "docs:read" in scopes:
        @server.tool()
        async def search_documents(query: str, top_k: int = 5) -> list[dict]:
            """Hybrid (keyword + vector) search over credit terms, examiner notes and bank policies.
            Returns passages with chunk_id, title, section and text."""
            return await _call("search_documents", query=query, top_k=top_k)

    if "data:read" in scopes:
        @server.tool()
        async def query_portfolio(sql: str) -> dict:
            """Run ONE read-only SQLite SELECT against the LC portfolio. Tables: letters_of_credit,
            presentations, discrepancies, counterparties. PII columns are blocked; a row cap is enforced."""
            return await _call("query_portfolio", sql=sql)

        @server.tool()
        async def get_discrepancies(presentation_id: str) -> list[dict]:
            """List discrepancies recorded for a presentation id such as PRS-0142-01."""
            return await _call("get_discrepancies", presentation_id=presentation_id)

        @server.resource("schema://portfolio")
        def portfolio_schema() -> str:
            """Tables and columns the data tools may use."""
            return schema_description()

    if "notices:write" in scopes:
        @server.tool()
        async def submit_refusal_notice(presentation_id: str, notice_text: str, approved_by: str,
                                        idempotency_key: str) -> dict:
            """Queue a refusal notice that a human signatory has ALREADY approved. Idempotent per key."""
            return await _call("submit_refusal_notice", presentation_id=presentation_id,
                               notice_text=notice_text, approved_by=approved_by, idempotency_key=idempotency_key)

    return server


def main() -> None:
    create_server().run()  # stdio by default


if __name__ == "__main__":
    main()
