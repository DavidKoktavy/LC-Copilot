"""Deterministic offline baseline that implements the full ``LLMClient`` contract without a model.

- routing: keyword/entity rules
- document QA: extractive — selects the best-supported sentences and cites them
- text-to-SQL: an intent-pattern planner over the allowed schema

It is intentionally simple and honest about it: its job is to make the whole system (graph,
guards, grounding, API, MCP, evals) runnable and regression-testable in CI with zero keys.
Swap ``LC_LLM_PROVIDER=azure_openai`` for real generation; the graph does not change.
"""

from __future__ import annotations

import math
import re

from ..observability import span
from ..rag.chunking import Chunk, split_sentences
from ..rag.text import entity_ids, tokens
from .base import INSUFFICIENT, LLMClient, RouteDecision, SqlPlan

ACTION = re.compile(r"\b(draft|prepare|write|send|issue)\b[^?]*\b(notice|refusal|message|letter|email)\b")
DATA = re.compile(
    r"\b(how many|number of|count|total|sum|average|list|which|show|top|largest|biggest)\b[^?]*"
    r"\b(letters of credit|lcs|credits|presentations|discrepancies|counterparties|beneficiaries|applicants)\b"
    r"|\b(exposure|outstanding|portfolio|expiring|expire before|tax[ _-]?ids?|e-?mails?)\b"
)
DOC = re.compile(
    r"\b(policy|policies|rules?|ucp|eucp|allowed|prohibited|permitted|required|tolerance|presentation period|"
    r"deadline|what does|according to|waivers?|refusal notice|incoterms?|bill of lading|insurance|examin\w*|"
    r"electronic records?|mletr|partial shipments?|transhipment)\b"
)
DOMAIN = re.compile(
    r"\b(lc|l/c|letters? of credit|credits?|documentary|presentations?|discrepan\w*|banks?|banking|shipments?|"
    r"waivers?|invoices?|ucp|eucp|policy|beneficiar\w*|applicants?|counterpart\w*|exposure|portfolio|"
    r"bills? of lading|examin\w*|refusal|tolerance|incoterms?|trade|expir\w*|mletr|electronic records?|"
    r"tax[ _-]?ids?|amount)\b|LC-\d{4}-\d{4}|PRS-\d{4}-\d{2}",
    re.IGNORECASE,
)
STATUSES = ["active", "expired", "cancelled", "fully_drawn", "discrepant", "compliant", "pending", "paid"]


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


class OfflineLLM(LLMClient):
    name = "offline-baseline"

    # ------------------------------------------------------------------ routing
    async def route(self, question: str) -> RouteDecision:
        with span("llm.route", provider=self.name):
            q = question.lower()
            if ACTION.search(q):
                return RouteDecision(route="action", reason="asks to draft or send a communication")
            if not DOMAIN.search(question):
                return RouteDecision(route="out_of_scope", reason="no trade-finance subject matter")
            data, doc = bool(DATA.search(q)), bool(DOC.search(q))
            if data and doc:
                return RouteDecision(route="hybrid", reason="needs portfolio data and document/policy text")
            if data:
                return RouteDecision(route="data", reason="aggregate or lookup over portfolio records")
            return RouteDecision(route="docs", reason="answer lives in credit terms or policy text")

    # ------------------------------------------------------------------ document QA
    async def answer_from_context(self, question: str, contexts: list[Chunk], strict: bool = False) -> str:
        with span("llm.answer_docs", provider=self.name, strict=strict):
            ids = {t.lower() for v in entity_ids(question).values() for t in v}
            q = set(tokens(question)) - ids  # identifiers already filtered retrieval; score on content words
            if not q or not contexts:
                return INSUFFICIENT
            cands = [
                (i, s, set(tokens(s)), set(tokens(c.section)))
                for i, c in enumerate(contexts, 1)
                for s in split_sentences(c.text)
                if len(s) > 25
            ]
            n = len(cands)
            df = {t: sum(1 for _, _, st, _ in cands if t in st) for t in q}
            idf = {t: math.log(1 + n / (1 + df[t])) for t in q}
            total = sum(idf.values()) or 1.0

            def score(item) -> float:
                i, _, st, sec = item
                lexical = sum(idf[t] for t in q & st) + 0.5 * sum(idf[t] for t in (q & sec) - st)
                return lexical / total + 0.03 * (len(contexts) - i)

            ranked = sorted(cands, key=score, reverse=True)
            best = ranked[0]
            coverage = sum(idf[t] for t in q & (best[2] | best[3])) / total
            if coverage < 0.35:
                return INSUFFICIENT
            picked = [best]
            # a second sentence only from the same passage, so answers don't stitch unrelated documents
            if not strict and len(ranked) > 1 and ranked[1][0] == best[0] and score(ranked[1]) >= 0.75 * score(best):
                picked.append(ranked[1])
            picked.sort(key=lambda x: (x[0], contexts[x[0] - 1].text.find(x[1])))
            return " ".join(re.sub(r"\.$", f" [{i}].", s) if s.endswith(".") else f"{s} [{i}]" for i, s, *_ in picked)

    # ------------------------------------------------------------------ text-to-SQL
    async def text_to_sql(self, question: str, schema: str, previous_error: str | None = None) -> SqlPlan:
        with span("llm.text_to_sql", provider=self.name):
            if previous_error:
                return SqlPlan(sql=None, rationale="offline planner cannot repair a rejected query")
            return self._plan(question)

    def _plan(self, question: str) -> SqlPlan:
        q = question.lower()
        ids = entity_ids(question)
        cur = next((c for c in ("usd", "eur", "czk") if re.search(rf"\b{c}\b", q)), None)
        status = next((s for s in STATUSES if s.replace("_", " ") in q), None)
        date = re.search(r"\d{4}-\d{2}-\d{2}", q)

        if re.search(r"tax[ _-]?ids?|e-?mails?|contact", q):
            return SqlPlan(sql="SELECT name, tax_id, contact_email FROM counterparties", rationale="identifier lookup")
        if ids["presentation_id"] and "discrepanc" in q:
            pid = ids["presentation_id"][0]
            return SqlPlan(
                sql=f"SELECT code, description, waived FROM discrepancies WHERE presentation_id = '{pid}'",
                rationale="discrepancies of one presentation",
            )
        if "discrepanc" in q and re.search(r"how many|number of|count", q):
            where = " WHERE waived = 0" if re.search(r"not (yet )?waived|unwaived|open|outstanding", q) else ""
            return SqlPlan(sql=f"SELECT COUNT(*) AS discrepancies FROM discrepancies{where}", rationale="count")
        if "presentation" in q and re.search(r"how many|number of|count", q):
            where = f" WHERE status = '{status}'" if status else ""
            return SqlPlan(sql=f"SELECT COUNT(*) AS presentations FROM presentations{where}", rationale="count")
        if re.search(r"expir", q) and date:
            return SqlPlan(
                sql="SELECT lc_number, currency, amount, expiry_date FROM letters_of_credit "
                f"WHERE status = 'active' AND expiry_date < '{date.group()}' ORDER BY expiry_date",
                rationale="active credits expiring before a date",
            )
        if re.search(r"high[ -]risk", q):
            return SqlPlan(
                sql="SELECT l.lc_number, c.name AS beneficiary, c.country, l.currency, l.amount, l.status "
                "FROM letters_of_credit l JOIN counterparties c ON c.counterparty_id = l.beneficiary_id "
                "WHERE c.risk_rating = 'high' ORDER BY l.issue_date",
                rationale="credits with high-risk beneficiaries",
            )
        if re.search(r"total|sum|exposure|outstanding", q):
            conds = [f"status = '{status or 'active'}'"] + ([f"currency = '{cur.upper()}'"] if cur else [])
            return SqlPlan(
                sql="SELECT currency, SUM(amount) AS total_amount, COUNT(*) AS lc_count FROM letters_of_credit "
                f"WHERE {' AND '.join(conds)} GROUP BY currency ORDER BY currency",
                rationale="aggregate credit amount",
            )
        if ids["lc_number"] and "presentation" in q:
            return SqlPlan(
                sql="SELECT presentation_id, presented_on, amount, channel, status FROM presentations "
                f"WHERE lc_number = '{ids['lc_number'][0]}' ORDER BY presented_on",
                rationale="presentations under one credit",
            )
        if re.search(r"\b(list|which|show)\b", q) and re.search(r"letters? of credit|lcs|credits", q):
            where = f" WHERE status = '{status}'" if status else ""
            return SqlPlan(
                sql=f"SELECT lc_number, currency, amount, expiry_date, status FROM letters_of_credit{where} "
                "ORDER BY expiry_date",
                rationale="list credits",
            )
        if ids["lc_number"]:
            return SqlPlan(
                sql="SELECT lc_number, currency, amount, issue_date, expiry_date, latest_shipment, status, rules "
                f"FROM letters_of_credit WHERE lc_number = '{ids['lc_number'][0]}'",
                rationale="single credit lookup",
            )
        return SqlPlan(sql=None, rationale="no supported query pattern")

    async def answer_from_rows(self, question: str, sql: str, rows: list[dict]) -> str:
        with span("llm.answer_data", provider=self.name):
            if not rows:
                return "No matching records were found."

            def render(r: dict) -> str:
                return "; ".join(
                    f"{k.replace('_', ' ')}: {('yes' if v else 'no') if k == 'waived' else _fmt(v)}"
                    for k, v in r.items()
                )

            if len(rows) == 1:
                return f"{render(rows[0])}."
            return f"{len(rows)} records found:\n" + "\n".join(f"- {render(r)}" for r in rows)

    async def draft_refusal_notice(self, presentation_id: str, discrepancies: list[dict], policy: list[Chunk]) -> str:
        with span("llm.draft_notice", provider=self.name):
            items = "\n".join(f"  {i}. {d['description']} ({d['code']})" for i, d in enumerate(discrepancies, 1))
            return (
                f"NOTICE OF REFUSAL — presentation {presentation_id}\n\n"
                "We refer to the above presentation. We refuse to honour it on the basis of the following "
                f"discrepancies:\n{items}\n\n"
                "We are holding the documents pending further instructions from the presenter.\n\n"
                "[DRAFT — requires approval by a second authorised signatory under policy TO-07]"
            )
