"""Input and output guardrails.

Input: prompt-injection heuristics, write/destructive intent, optional Azure AI Content Safety.
Output: PII redaction as the last line of defence (the SQL guard should already have blocked it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from ..config import Settings
from ..observability import span

INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)|disregard (your|the) (rules|instructions)|"
    r"system prompt|you are now|developer mode|jailbreak|act as (an? )?(admin|dba|root)|reveal (your|the) (prompt|instructions)",
    re.IGNORECASE,
)
WRITE_INTENT = re.compile(
    r"\b(delete|drop|truncate|insert|alter|update|overwrite|wipe)\b[^?.]*\b(table|tables|records?|rows?|data|"
    r"database|presentations|discrepancies|letters of credit|counterparties|status)\b",
    re.IGNORECASE,
)
PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email redacted]"),
    (re.compile(r"\b(CZ|CN|IT|TR|HU)[0-9A-Z]{8,15}\b"), "[tax id redacted]"),
]


@dataclass
class GuardVerdict:
    allowed: bool
    reason: str = ""
    category: str = ""


async def check_input(question: str, s: Settings) -> GuardVerdict:
    with span("guardrail.input") as sp:
        if len(question) > 2000:
            v = GuardVerdict(False, "question too long (max 2000 characters)", "length")
        elif INJECTION.search(question):
            v = GuardVerdict(False, "the request tries to override the assistant's instructions", "prompt_injection")
        elif WRITE_INTENT.search(question):
            v = GuardVerdict(False, "the assistant has read-only access and cannot modify records", "write_intent")
        elif s.azure_content_safety_endpoint:
            v = await _azure_content_safety(question, s)
        else:
            v = GuardVerdict(True)
        sp.set(allowed=v.allowed, category=v.category)
        return v


async def _azure_content_safety(text: str, s: Settings) -> GuardVerdict:
    url = f"{s.azure_content_safety_endpoint.rstrip('/')}/contentsafety/text:analyze?api-version=2024-09-01"
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(url, headers={"Ocp-Apim-Subscription-Key": s.azure_content_safety_api_key},
                             json={"text": text[:10000]})
        r.raise_for_status()
        worst = max((x.get("severity", 0) for x in r.json().get("categoriesAnalysis", [])), default=0)
    except httpx.HTTPError:
        return GuardVerdict(True, "content safety unavailable — failing open for read-only Q&A", "cs_unavailable")
    if worst >= 4:
        return GuardVerdict(False, "blocked by content safety policy", "content_safety")
    return GuardVerdict(True)


def redact(text: str) -> tuple[str, int]:
    n = 0
    for pat, repl in PII:
        text, k = pat.subn(repl, text)
        n += k
    return text, n
