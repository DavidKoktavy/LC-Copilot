"""Spawn the LC Copilot MCP server over stdio and exercise it like an external agent would.

    python scripts/mcp_client_demo.py
"""

import asyncio

from lc_copilot.mcp_client import MCPToolClient, MCPToolError


async def main() -> None:
    async with MCPToolClient.for_lc_copilot("docs:read,data:read") as mcp:
        print("tools exposed:", await mcp.list_tools())

        hits = await mcp.call("search_documents", {"query": "waiver of discrepancies", "top_k": 2})
        for h in hits:
            print(f"  [{h['chunk_id']}] {h['section']}: {h['text'][:90]}...")

        rows = await mcp.call("query_portfolio", {
            "sql": "SELECT status, COUNT(*) AS n FROM presentations GROUP BY status ORDER BY n DESC"})
        print("presentations by status:", rows["rows"])

        for blocked in ("SELECT name, tax_id FROM counterparties", "DELETE FROM presentations"):
            try:
                await mcp.call("query_portfolio", {"sql": blocked})
            except MCPToolError as e:
                print(f"blocked as expected: {e}")

        try:
            await mcp.call("submit_refusal_notice", {})
        except MCPToolError as e:
            print(f"write tool not exposed under read-only scopes: {e}")


if __name__ == "__main__":
    asyncio.run(main())
