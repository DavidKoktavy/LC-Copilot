"""MCP client side: connect to any MCP server over stdio and call tools with timeouts and retries.

Used by the integration test and ``scripts/mcp_client_demo.py``. The same wrapper lets a LangGraph
node consume *remote* tools (e.g. a sanctions-screening MCP server owned by another team) with the
resilience guarantees this project applies to its own tools.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .observability import span
from .resilience import PermanentError, TransientError


class MCPToolError(PermanentError):
    pass


class MCPToolClient:
    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None,
                 timeout_s: float = 20.0) -> None:
        self.params = StdioServerParameters(command=command, args=args, env={**os.environ, **(env or {})})
        self.timeout_s = timeout_s
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    @classmethod
    def for_lc_copilot(cls, scopes: str = "docs:read,data:read") -> MCPToolClient:
        # make the package importable in the child even when running from a source checkout
        src = str(Path(__file__).resolve().parents[1])
        pypath = os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH", "")]))
        return cls(sys.executable, ["-m", "lc_copilot.mcp_server"], {"LC_MCP_SCOPES": scopes, "PYTHONPATH": pypath})

    async def __aenter__(self) -> MCPToolClient:
        read, write = await self._stack.enter_async_context(stdio_client(self.params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> list[str]:
        assert self.session
        return [t.name for t in (await self.session.list_tools()).tools]

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        assert self.session
        # anyio cancel scope (not asyncio.wait_for): the MCP session must stay in the caller's task.
        # No automatic retry here — the client can't know whether a remote tool is idempotent.
        with span(f"mcp.call.{name}"):
            try:
                with anyio.fail_after(self.timeout_s):
                    res = await self.session.call_tool(name, args)
            except TimeoutError as e:
                raise TransientError(f"MCP tool {name} timed out after {self.timeout_s}s") from e
        text = res.content[0].text if res.content else ""
        if res.is_error:
            raise MCPToolError(text)
        if res.structured_content is not None:
            sc = res.structured_content
            return sc.get("result", sc) if isinstance(sc, dict) else sc
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
