import pytest

from lc_copilot.mcp_client import MCPToolClient, MCPToolError
from lc_copilot.mcp_server import create_server
from tests.conftest import run


def _names(server):
    return {t.name for t in run(server.list_tools())}


def test_scopes_control_which_tools_exist():
    assert _names(create_server({"docs:read"})) == {"search_documents"}
    assert _names(create_server({"docs:read", "data:read"})) == {"search_documents", "query_portfolio",
                                                                 "get_discrepancies"}
    assert "submit_refusal_notice" in _names(create_server({"docs:read", "notices:write"}))


def test_end_to_end_over_stdio():
    async def scenario():
        async with MCPToolClient.for_lc_copilot("docs:read,data:read") as mcp:
            assert "submit_refusal_notice" not in await mcp.list_tools()
            hits = await mcp.call("search_documents", {"query": "waiver in writing", "top_k": 2})
            assert hits and "chunk_id" in hits[0]
            res = await mcp.call("query_portfolio", {"sql": "SELECT COUNT(*) AS n FROM discrepancies"})
            assert res["rows"] == [{"n": 5}]
            with pytest.raises(MCPToolError, match="tax_id"):
                await mcp.call("query_portfolio", {"sql": "SELECT tax_id FROM counterparties"})
            with pytest.raises(MCPToolError, match="pattern"):
                await mcp.call("get_discrepancies", {"presentation_id": "x' OR 1=1 --"})

    run(scenario())
