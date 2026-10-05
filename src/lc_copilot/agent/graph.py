"""The agent: a LangGraph state machine with explicit routing, retries, fallbacks and a human gate.

    guard_input ─► route ─┬─ docs ─────► retrieve ► answer_docs ► check_docs ─┬─► compose
                          │                              ▲   (retry strict)   │
                          │                              └────────────────────┤
                          ├─ hybrid ───► (docs branch) ────────────────────► plan_sql
                          ├─ data ─────► plan_sql ► run_sql ─(repair once)─► answer_data ► compose
                          ├─ action ───► gather ► draft_notice ► human_approval ⏸ ► submit ► compose
                          └─ out_of_scope / blocked ─────────────────────────────────────► compose

Every node is traced; every external call goes through the tool registry or resilient LLM client.
"""

from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..data.sql_guard import SQLPolicyViolation, schema_description
from ..llm.base import INSUFFICIENT
from ..llm.offline import OfflineLLM
from ..observability import collect_spans, span, usage_totals
from ..rag.chunking import Chunk
from ..rag.grounding import check_data_answer, check_document_answer
from ..rag.text import entity_ids
from ..services import Services
from .guardrails import check_input, redact

MAX_DOC_ATTEMPTS = 2
MAX_SQL_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    blocked_reason: str
    contexts: list[dict]
    doc_answer: str
    doc_grounding: dict
    doc_attempts: int
    sql: str
    sql_error: str
    sql_attempts: int
    rows: list[dict]
    data_answer: str
    data_grounding: dict
    presentation_id: str
    draft: str
    approval: dict
    submission: dict
    final_answer: str
    citations: list[dict]
    status: str
    degraded: Annotated[list[str], operator.add]
    steps: Annotated[list[str], operator.add]


def _chunks(state: AgentState) -> list[Chunk]:
    return [Chunk(**c) for c in state.get("contexts", [])]


def build_graph(svc: Services, checkpointer=None):
    llm, fallback = svc.llm, OfflineLLM()

    # ------------------------------------------------------------------ nodes
    async def guard_input(state: AgentState) -> dict:
        v = await check_input(state["question"], svc.settings)
        out: dict = {"steps": ["guard_input"]}
        if not v.allowed:
            out |= {"blocked_reason": v.reason, "route": "blocked", "status": "refused"}
        return out

    async def route(state: AgentState) -> dict:
        try:
            d = await llm.route(state["question"])
            return {"route": d.route, "route_reason": d.reason, "steps": ["route"]}
        except Exception as e:  # router outage → deterministic fallback router
            d = await fallback.route(state["question"])
            return {"route": d.route, "route_reason": d.reason, "steps": ["route"],
                    "degraded": [f"router fallback: {type(e).__name__}"]}

    async def retrieve(state: AgentState) -> dict:
        hits = await svc.tools.invoke("search_documents", {"query": state["question"], "top_k": svc.settings.top_k})
        ctx = [
            {"chunk_id": h["chunk_id"], "doc_id": h["chunk_id"].split("#")[0], "title": h["title"],
             "section": h["section"], "text": h["text"], "metadata": {}}
            for h in hits
        ]
        return {"contexts": ctx, "steps": ["retrieve"]}

    async def answer_docs(state: AgentState) -> dict:
        attempt = state.get("doc_attempts", 0) + 1
        strict = attempt > 1
        try:
            ans = await llm.answer_from_context(state["question"], _chunks(state), strict=strict)
            degraded: list[str] = []
        except Exception as e:  # model outage → extractive fallback keeps the service answering
            ans = await fallback.answer_from_context(state["question"], _chunks(state), strict=True)
            degraded = [f"answer fallback: {type(e).__name__}"]
        return {"doc_answer": ans, "doc_attempts": attempt, "steps": [f"answer_docs#{attempt}"], "degraded": degraded}

    async def check_docs(state: AgentState) -> dict:
        ans = state["doc_answer"]
        if ans.strip() == INSUFFICIENT:
            return {"doc_grounding": {"score": 0.0, "insufficient": True}, "steps": ["check_docs"]}
        with span("grounding.docs") as sp:
            rep = check_document_answer(ans, _chunks(state), svc.settings.min_grounding_score)
            sp.set(**rep.as_dict())
        return {"doc_grounding": rep.as_dict() | {"ok": rep.ok}, "steps": ["check_docs"]}

    def after_check(state: AgentState) -> Literal["answer_docs", "plan_sql", "compose"]:
        g = state["doc_grounding"]
        if not g.get("ok") and not g.get("insufficient") and state["doc_attempts"] < MAX_DOC_ATTEMPTS:
            return "answer_docs"
        return "plan_sql" if state["route"] == "hybrid" else "compose"

    async def plan_sql(state: AgentState) -> dict:
        attempt = state.get("sql_attempts", 0) + 1
        try:
            plan = await llm.text_to_sql(state["question"], schema_description(), state.get("sql_error"))
        except Exception as e:
            return {"sql": "", "sql_error": f"planner unavailable: {type(e).__name__}", "sql_attempts": attempt,
                    "steps": [f"plan_sql#{attempt}"], "degraded": ["sql planner unavailable"]}
        return {"sql": plan.sql or "", "sql_attempts": attempt, "steps": [f"plan_sql#{attempt}"]}

    async def run_sql(state: AgentState) -> dict:
        if not state.get("sql"):
            return {"rows": [], "sql_error": state.get("sql_error") or "no safe query could be planned",
                    "steps": ["run_sql"]}
        try:
            res = await svc.tools.invoke("query_portfolio", {"sql": state["sql"]})
            return {"rows": res["rows"], "sql": res["sql"], "sql_error": "", "steps": ["run_sql"]}
        except SQLPolicyViolation as e:
            return {"rows": [], "sql_error": f"blocked by data policy: {e}", "steps": ["run_sql"]}
        except Exception as e:
            return {"rows": [], "sql_error": f"{type(e).__name__}: {e}", "steps": ["run_sql"]}

    def after_sql(state: AgentState) -> Literal["plan_sql", "answer_data"]:
        if state.get("sql_error") and state["sql_attempts"] < MAX_SQL_ATTEMPTS and state.get("sql"):
            return "plan_sql"  # one repair round with the error fed back to the model
        return "answer_data"

    async def answer_data(state: AgentState) -> dict:
        if state.get("sql_error"):
            return {"data_answer": "", "steps": ["answer_data"]}
        ans = await llm.answer_from_rows(state["question"], state["sql"], state["rows"])
        with span("grounding.data") as sp:
            rep = check_data_answer(ans, state["rows"], state["question"])
            sp.set(**rep.as_dict())
        if not rep.ok:  # never show numbers that aren't in the result set
            ans = await fallback.answer_from_rows(state["question"], state["sql"], state["rows"])
            rep = check_data_answer(ans, state["rows"], state["question"])
        return {"data_answer": ans, "data_grounding": rep.as_dict() | {"ok": rep.ok}, "steps": ["answer_data"]}

    async def gather(state: AgentState) -> dict:
        pids = entity_ids(state["question"])["presentation_id"]
        if not pids:
            return {"presentation_id": "", "steps": ["gather"]}
        rows = await svc.tools.invoke("get_discrepancies", {"presentation_id": pids[0]})
        hits = await svc.tools.invoke("search_documents", {"query": "refusal notice discrepancies four-eyes", "top_k": 3})
        ctx = [{"chunk_id": h["chunk_id"], "doc_id": h["chunk_id"].split("#")[0], "title": h["title"],
                "section": h["section"], "text": h["text"], "metadata": {}} for h in hits]
        return {"presentation_id": pids[0], "rows": rows, "contexts": ctx, "steps": ["gather"]}

    def after_gather(state: AgentState) -> Literal["draft_notice", "compose"]:
        open_items = [r for r in state.get("rows", []) if not r.get("waived")]
        return "draft_notice" if state.get("presentation_id") and open_items else "compose"

    async def draft_notice(state: AgentState) -> dict:
        open_items = [r for r in state["rows"] if not r.get("waived")]
        draft = await llm.draft_refusal_notice(state["presentation_id"], open_items, _chunks(state))
        return {"draft": draft, "steps": ["draft_notice"]}

    async def human_approval(state: AgentState) -> dict:
        # ⏸ Execution pauses here; the API returns 202 + thread_id. Policy TO-07 requires four-eyes.
        decision = interrupt({
            "type": "approve_refusal_notice",
            "presentation_id": state["presentation_id"],
            "draft": state["draft"],
            "policy": "TO-07 four-eyes approval",
        })
        return {"approval": dict(decision), "steps": ["human_approval"]}

    def after_approval(state: AgentState) -> Literal["submit", "compose"]:
        return "submit" if state["approval"].get("approved") else "compose"

    async def submit(state: AgentState, config) -> dict:
        thread = config["configurable"]["thread_id"]
        text = state["approval"].get("edited_text") or state["draft"]
        res = await svc.tools.invoke("submit_refusal_notice", {
            "presentation_id": state["presentation_id"],
            "notice_text": text,
            "approved_by": state["approval"].get("approver", "unknown"),
            "idempotency_key": f"{thread}:{state['presentation_id']}",
        })
        return {"submission": res, "steps": ["submit"]}

    async def compose(state: AgentState) -> dict:
        r = state.get("route")
        parts, status, citations = [], "answered", []
        if r == "blocked":
            return {"final_answer": f"I can't help with that: {state['blocked_reason']}.", "status": "refused",
                    "steps": ["compose"]}
        if r == "out_of_scope":
            return {"final_answer": "That's outside what I can help with. I answer questions about the bank's "
                    "letters of credit, presentations, discrepancies and trade-operations policies.",
                    "status": "out_of_scope", "steps": ["compose"]}
        if r == "action":
            if not state.get("presentation_id"):
                return {"final_answer": "Please name the presentation (e.g. PRS-0142-01) the notice is for.",
                        "status": "needs_input", "steps": ["compose"]}
            if not state.get("draft"):
                return {"final_answer": f"Presentation {state['presentation_id']} has no open discrepancies, "
                        "so there is nothing to refuse.", "status": "answered", "steps": ["compose"]}
            if state.get("submission"):
                sub = state["submission"]
                return {"final_answer": f"Refusal notice approved and queued as {sub['message_id']}.",
                        "status": "sent", "steps": ["compose"]}
            return {"final_answer": "Draft discarded by the reviewer; nothing was sent.", "status": "discarded",
                    "steps": ["compose"]}

        if r in ("docs", "hybrid"):
            g = state.get("doc_grounding", {})
            ctx = state.get("contexts", [])
            if g.get("insufficient"):
                parts.append("The documents I can access don't answer this.")
                status = "insufficient"
            elif g.get("ok"):
                parts.append(state["doc_answer"])
                citations = [
                    {"ref": i, "chunk_id": ctx[i - 1]["chunk_id"], "title": ctx[i - 1]["title"],
                     "section": ctx[i - 1]["section"]}
                    for i in g.get("cited", [])
                ]
            else:  # graceful degradation: say so and point at the sources instead of guessing
                parts.append("I couldn't produce a fully grounded answer. The most relevant passages are: "
                             + "; ".join(f"{c['title']} — {c['section']}" for c in ctx[:3]) + ".")
                status = "low_confidence"
        if r in ("data", "hybrid"):
            if state.get("sql_error"):
                parts.append(f"I couldn't answer from the portfolio data ({state['sql_error']}).")
                status = "refused" if "data policy" in state["sql_error"] else "error"
            else:
                parts.append(state.get("data_answer", ""))
        text, n = redact("\n\n".join(p for p in parts if p))
        return {"final_answer": text, "status": status, "citations": citations, "steps": ["compose"],
                "degraded": [f"redacted {n} PII value(s)"] if n else []}

    # ------------------------------------------------------------------ wiring
    g = StateGraph(AgentState)
    for name, fn in [("guard_input", guard_input), ("route", route), ("retrieve", retrieve),
                     ("answer_docs", answer_docs), ("check_docs", check_docs), ("plan_sql", plan_sql),
                     ("run_sql", run_sql), ("answer_data", answer_data), ("gather", gather),
                     ("draft_notice", draft_notice), ("human_approval", human_approval), ("submit", submit),
                     ("compose", compose)]:
        g.add_node(name, _traced(name, fn))
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", lambda s: "compose" if s.get("route") == "blocked" else "route")
    g.add_conditional_edges("route", lambda s: {
        "docs": "retrieve", "hybrid": "retrieve", "data": "plan_sql", "action": "gather",
    }.get(s["route"], "compose"))
    g.add_edge("retrieve", "answer_docs")
    g.add_edge("answer_docs", "check_docs")
    g.add_conditional_edges("check_docs", after_check)
    g.add_edge("plan_sql", "run_sql")
    g.add_conditional_edges("run_sql", after_sql)
    g.add_edge("answer_data", "compose")
    g.add_conditional_edges("gather", after_gather)
    g.add_edge("draft_notice", "human_approval")
    g.add_conditional_edges("human_approval", after_approval)
    g.add_edge("submit", "compose")
    g.add_edge("compose", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def _traced(name: str, fn):
    import inspect

    if "config" in inspect.signature(fn).parameters:
        async def wrapped(state, config):
            with span(f"node.{name}"):
                return await fn(state, config)
    else:
        async def wrapped(state):
            with span(f"node.{name}"):
                return await fn(state)
    wrapped.__name__ = name
    return wrapped


class Agent:
    """Thin façade used by the API, the A2A endpoint and the eval harness."""

    def __init__(self, svc: Services) -> None:
        self.svc = svc
        self.graph = build_graph(svc)

    async def _run(self, payload: Any, thread_id: str) -> dict:
        cfg = {"configurable": {"thread_id": thread_id}}
        t0 = time.perf_counter()
        with collect_spans() as spans, span("agent.run", thread_id=thread_id) as root:
            out = await self.graph.ainvoke(payload, cfg)
            trace_id = root.trace_id
        result = self._result(out, thread_id)
        result["meta"] = {
            "trace_id": trace_id,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "llm": self.svc.llm.name,
            **usage_totals(spans),
        }
        return result

    async def ask(self, question: str, thread_id: str | None = None) -> dict:
        # a thread may carry several questions; reset per-question scratch fields explicitly
        fresh = {"question": question, "route": "", "blocked_reason": "", "contexts": [], "doc_answer": "",
                 "doc_grounding": {}, "doc_attempts": 0, "sql": "", "sql_error": "", "sql_attempts": 0,
                 "rows": [], "data_answer": "", "data_grounding": {}, "presentation_id": "", "draft": "",
                 "approval": {}, "submission": {}, "citations": []}
        return await self._run(fresh, thread_id or uuid.uuid4().hex)

    async def resume(self, thread_id: str, approved: bool, approver: str, edited_text: str | None = None) -> dict:
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not state.next:
            raise KeyError(f"thread {thread_id} has nothing awaiting approval")
        decision = {"approved": approved, "approver": approver, "edited_text": edited_text}
        return await self._run(Command(resume=decision), thread_id)

    @staticmethod
    def _result(out: dict, thread_id: str) -> dict:
        if "__interrupt__" in out:
            req = out["__interrupt__"][0].value
            return {"thread_id": thread_id, "status": "pending_approval", "answer": None,
                    "approval_request": req, "route": out.get("route"), "steps": out.get("steps", []),
                    "citations": [], "sql": None, "degraded": out.get("degraded", [])}
        return {
            "thread_id": thread_id,
            "status": out.get("status"),
            "answer": out.get("final_answer"),
            "route": out.get("route"),
            "route_reason": out.get("route_reason"),
            "citations": out.get("citations", []),
            "sql": out.get("sql") or None,
            "rows": out.get("rows", [])[:50] if out.get("route") in ("data", "hybrid") else [],
            "grounding": {"docs": out.get("doc_grounding"), "data": out.get("data_grounding")},
            "steps": out.get("steps", []),
            "degraded": out.get("degraded", []),
        }
