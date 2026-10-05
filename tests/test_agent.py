import json

from lc_copilot.agent.graph import Agent
from lc_copilot.llm.offline import OfflineLLM
from lc_copilot.resilience import TransientError
from tests.conftest import run


def test_routes(svc):
    a = Agent(svc)
    cases = {
        "Are partial shipments allowed under LC-2026-0187?": "docs",
        "How many presentations are discrepant?": "data",
        "Which discrepancies were recorded for presentation PRS-0142-01, and what does policy say about waivers?": "hybrid",
        "Draft a refusal notice for presentation PRS-0142-01": "action",
        "Best pizza in Brno?": "out_of_scope",
    }
    for q, route in cases.items():
        assert run(a.ask(q))["route"] == route, q


def test_docs_answer_is_cited_and_grounded(svc):
    r = run(Agent(svc).ask("Are partial shipments allowed under LC-2026-0187?"))
    assert r["status"] == "answered" and "prohibited" in r["answer"]
    assert r["citations"][0]["chunk_id"].startswith("LC-2026-0187#")
    assert r["grounding"]["docs"]["ok"] is True
    assert r["meta"]["trace_id"]


def test_abstains_instead_of_guessing(svc):
    r = run(Agent(svc).ask("Who is the confirming bank on LC-2026-0142?"))
    assert r["status"] == "insufficient"


def test_pii_never_leaves_the_database(svc):
    r = run(Agent(svc).ask("Show the tax ids of all counterparties"))
    assert r["status"] == "refused"
    assert "CZ27845123" not in r["answer"]
    assert r["steps"].count("plan_sql#2") == 1  # one repair attempt, then refuse


class HallucinatingOnceLLM(OfflineLLM):
    """First answer cites a passage that doesn't support it; the strict retry is fine."""

    async def answer_from_context(self, question, contexts, strict=False):
        if not strict:
            return "Partial shipments are allowed in up to 7 lots [1]."
        return await super().answer_from_context(question, contexts, strict)


def test_grounding_failure_triggers_strict_retry(svc):
    svc.llm = HallucinatingOnceLLM()
    r = run(Agent(svc).ask("Are partial shipments allowed under LC-2026-0187?"))
    assert "answer_docs#2" in r["steps"]
    assert r["status"] == "answered" and "prohibited" in r["answer"]


class DownLLM(OfflineLLM):
    name = "down"

    async def route(self, q):
        raise TransientError("503")

    async def answer_from_context(self, *a, **k):
        raise TransientError("503")


def test_model_outage_degrades_gracefully(svc):
    svc.llm = DownLLM()
    r = run(Agent(svc).ask("What tolerance applies to the amount of LC-2026-0142?"))
    assert r["status"] == "answered" and "5%" in r["answer"]
    assert any("fallback" in d for d in r["degraded"])


def test_human_in_the_loop_approve_sends_once(svc):
    a = Agent(svc)
    r = run(a.ask("Draft a refusal notice for presentation PRS-0142-01"))
    assert r["status"] == "pending_approval"
    assert "INS-COVER" in r["approval_request"]["draft"]
    outbox = svc.tools.outbox_path
    before = outbox.read_text().count("\n") if outbox.exists() else 0

    done = run(a.resume(r["thread_id"], approved=True, approver="J. Novak"))
    assert done["status"] == "sent"
    lines = outbox.read_text().splitlines()
    assert len(lines) == before + 1
    assert json.loads(lines[-1])["approved_by"] == "J. Novak"


def test_human_in_the_loop_reject_sends_nothing(svc):
    a = Agent(svc)
    r = run(a.ask("Draft a refusal notice for presentation PRS-0161-01"))
    assert "COO-ISSUER" not in r["approval_request"]["draft"]  # waived discrepancy excluded
    done = run(a.resume(r["thread_id"], approved=False, approver="J. Novak"))
    assert done["status"] == "discarded"


def test_tool_audit_log_written(svc):
    run(Agent(svc).ask("How many presentations are discrepant?"))
    last = json.loads(svc.tools.audit_path.read_text().splitlines()[-1])
    assert last["tool"] == "query_portfolio" and last["outcome"] == "ok" and last["trace_id"]
