"""Azure OpenAI / OpenAI client over raw HTTP (no SDK lock-in), wrapped in the resilience layer.

- 429 / 5xx / timeouts → retried with jittered backoff; 4xx → fail fast
- circuit breaker per provider so a degraded endpoint fails quickly instead of piling up latency
- outbound token bucket to stay under the deployment's RPM quota
- JSON-mode + pydantic validation for every structured task, one repair attempt on bad JSON
"""

from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..observability import span
from ..rag.chunking import Chunk
from ..resilience import (
    CircuitBreaker,
    PermanentError,
    RetryPolicy,
    TokenBucket,
    TransientError,
    call_with_retry,
)
from . import prompts
from .base import LLMClient, RouteDecision, SqlPlan


def _ctx(contexts: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.title} — {c.section})\n{c.text}" for i, c in enumerate(contexts, 1))


class OpenAICompatibleLLM(LLMClient):
    def __init__(self, s: Settings, azure: bool) -> None:
        self.s = s
        self.azure = azure
        if azure:
            base = s.azure_openai_endpoint.rstrip("/")
            v = s.azure_openai_api_version
            self.chat_url = f"{base}/openai/deployments/{s.azure_openai_chat_deployment}/chat/completions?api-version={v}"
            self.emb_url = f"{base}/openai/deployments/{s.azure_openai_embedding_deployment}/embeddings?api-version={v}"
            self.headers = {"api-key": s.azure_openai_api_key}
            self.model, self.emb_model = s.azure_openai_chat_deployment, s.azure_openai_embedding_deployment
            self.name = f"azure_openai:{self.model}"
        else:
            base = s.openai_base_url.rstrip("/")
            self.chat_url, self.emb_url = f"{base}/chat/completions", f"{base}/embeddings"
            self.headers = {"Authorization": f"Bearer {s.openai_api_key}"}
            self.model, self.emb_model = s.openai_chat_model, s.openai_embedding_model
            self.name = f"openai:{self.model}"
        self.policy = RetryPolicy(attempts=s.llm_max_retries, timeout_s=s.llm_timeout_s)
        self.breaker = CircuitBreaker(self.name, failure_threshold=5, reset_after_s=30)
        self.bucket = TokenBucket(rate_per_s=5, capacity=10)
        self._http = httpx.AsyncClient(timeout=s.llm_timeout_s)

    # ------------------------------------------------------------------ transport
    async def _post(self, url: str, body: dict) -> dict:
        async def once() -> dict:
            await self.bucket.acquire()
            try:
                r = await self._http.post(url, headers=self.headers, json=body)
            except httpx.TransportError as e:
                raise TransientError(str(e)) from e
            if r.status_code == 429 or r.status_code >= 500:
                raise TransientError(f"{r.status_code}: {r.text[:200]}")
            if r.status_code >= 400:
                raise PermanentError(f"{r.status_code}: {r.text[:300]}")
            return r.json()

        return await self.breaker.call(lambda: call_with_retry(once, self.policy))

    async def _chat(self, task: str, system: str, user: str, json_mode: bool = False) -> str:
        body: dict = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }
        if not self.azure:
            body["model"] = self.model
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        with span(f"llm.{task}", provider=self.name, prompt_version=prompts.PROMPT_VERSION) as sp:
            data = await self._post(self.chat_url, body)
            usage = data.get("usage", {})
            sp.set(tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0))
            return data["choices"][0]["message"]["content"] or ""

    async def _structured(self, task: str, system: str, user: str, model: type[BaseModel]) -> BaseModel:
        raw = await self._chat(task, system, user, json_mode=True)
        try:
            return model.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            fix = f"{user}\n\nYour previous output was invalid ({e.__class__.__name__}). Return valid JSON only."
            return model.model_validate(json.loads(await self._chat(task, system, fix, json_mode=True)))

    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        body: dict = {"input": texts}
        if not self.azure:
            body["model"] = self.emb_model
        with span("llm.embed", provider=self.name, n=len(texts)) as sp:
            data = await self._post(self.emb_url, body)
            sp.set(tokens_in=data.get("usage", {}).get("prompt_tokens", 0))
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    # ------------------------------------------------------------------ tasks
    async def route(self, question: str) -> RouteDecision:
        return await self._structured("route", prompts.ROUTER, question, RouteDecision)  # type: ignore[return-value]

    async def answer_from_context(self, question: str, contexts: list[Chunk], strict: bool = False) -> str:
        system = prompts.ANSWER_DOCS_STRICT if strict else prompts.ANSWER_DOCS
        return (await self._chat("answer_docs", system, f"Context:\n{_ctx(contexts)}\n\nQuestion: {question}")).strip()

    async def text_to_sql(self, question: str, schema: str, previous_error: str | None = None) -> SqlPlan:
        system = prompts.TEXT_TO_SQL.format(schema=schema)
        if previous_error:
            system += prompts.TEXT_TO_SQL_REPAIR.format(error=previous_error)
        return await self._structured("text_to_sql", system, question, SqlPlan)  # type: ignore[return-value]

    async def answer_from_rows(self, question: str, sql: str, rows: list[dict]) -> str:
        user = f"Question: {question}\nSQL: {sql}\nRows (JSON): {json.dumps(rows, default=str)[:6000]}"
        return (await self._chat("answer_data", prompts.ANSWER_DATA, user)).strip()

    async def draft_refusal_notice(self, presentation_id: str, discrepancies: list[dict], policy: list[Chunk]) -> str:
        user = (
            f"Presentation: {presentation_id}\nDiscrepancies: {json.dumps(discrepancies)}\n\n"
            f"Policy excerpts:\n{_ctx(policy)}"
        )
        return (await self._chat("draft_notice", prompts.DRAFT_NOTICE, user)).strip()
