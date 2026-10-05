"""The Azure OpenAI client against a mocked transport: URL shape, auth header, retry on 429,
fail-fast on 4xx, JSON-mode parsing and token accounting."""

import json

import httpx
import pytest

from lc_copilot.config import Settings
from lc_copilot.llm.openai_compat import OpenAICompatibleLLM
from lc_copilot.observability import collect_spans, usage_totals
from lc_copilot.resilience import PermanentError, RetryPolicy
from tests.conftest import run


def _llm(handler) -> OpenAICompatibleLLM:
    s = Settings(llm_provider="azure_openai", azure_openai_endpoint="https://unit.openai.azure.com",
                 azure_openai_api_key="test-key", azure_openai_chat_deployment="gpt-4o-mini")
    llm = OpenAICompatibleLLM(s, azure=True)
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm.policy = RetryPolicy(attempts=3, base_delay_s=0.001, max_delay_s=0.002, timeout_s=2)
    return llm


def _completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 15},
    })


def test_retries_429_then_parses_structured_output():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if len(seen) == 1:
            return httpx.Response(429, text="slow down")
        body = json.loads(req.content)
        assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0
        return _completion('{"route": "data", "reason": "aggregate"}')

    llm = _llm(handler)
    with collect_spans() as spans:
        d = run(llm.route("How many presentations are discrepant?"))
    assert d.route == "data"
    assert len(seen) == 2
    assert seen[0].url.path == "/openai/deployments/gpt-4o-mini/chat/completions"
    assert seen[0].url.params["api-version"] and seen[0].headers["api-key"] == "test-key"
    totals = usage_totals(spans)
    assert totals["tokens_in"] == 120 and totals["cost_usd"] > 0


def test_client_errors_fail_fast_without_retry():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(PermanentError):
        run(_llm(handler).answer_from_context("q", []))
    assert len(calls) == 1


def test_invalid_json_gets_one_repair_attempt():
    replies = iter(["not json", '{"sql": "SELECT 1", "rationale": "x"}'])

    def handler(req):
        return _completion(next(replies))

    plan = run(_llm(handler).text_to_sql("q", "schema"))
    assert plan.sql == "SELECT 1"
