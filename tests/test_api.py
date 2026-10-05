import pytest
from fastapi.testclient import TestClient

from lc_copilot.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["chunks"] > 10


def test_ask_returns_grounded_answer_with_trace(client):
    r = client.post("/v1/ask", json={"question": "What tolerance applies to the amount of LC-2026-0142?"})
    assert r.status_code == 200
    body = r.json()
    assert "5%" in body["answer"] and body["citations"]
    assert r.headers["X-Trace-Id"] and r.headers["X-Request-Id"]


def test_validation_rejects_bad_input(client):
    assert client.post("/v1/ask", json={"question": "x"}).status_code == 422


def test_idempotency_key_replays_and_conflicts(client):
    h = {"Idempotency-Key": "ask-123456"}
    q = {"question": "How many presentations are discrepant?"}
    a = client.post("/v1/ask", json=q, headers=h)
    b = client.post("/v1/ask", json=q, headers=h)
    assert a.headers["Idempotent-Replayed"] == "false" and b.headers["Idempotent-Replayed"] == "true"
    assert a.json()["meta"]["trace_id"] == b.json()["meta"]["trace_id"]
    assert client.post("/v1/ask", json={"question": "something else entirely"}, headers=h).status_code == 409


def test_human_approval_flow(client):
    r = client.post("/v1/ask", json={"question": "Draft a refusal notice for presentation PRS-0142-01"})
    assert r.status_code == 202
    tid = r.json()["thread_id"]
    done = client.post(f"/v1/threads/{tid}/approval", json={"approved": True, "approver": "J. Novak"})
    assert done.json()["status"] == "sent"
    again = client.post(f"/v1/threads/{tid}/approval", json={"approved": True, "approver": "J. Novak"})
    assert again.status_code == 404  # nothing left to approve — no double send


def test_tools_endpoint_lists_contracts(client):
    names = {t["name"] for t in client.get("/v1/tools").json()}
    assert {"search_documents", "query_portfolio"} <= names


def test_a2a_agent_card_and_message_send(client):
    assert client.get("/.well-known/agent-card.json").json()["name"] == "lc-copilot"
    rpc = {"jsonrpc": "2.0", "id": 7, "method": "message/send",
           "params": {"message": {"role": "user", "parts": [{"kind": "text",
                                                             "text": "How many presentations are discrepant?"}]}}}
    res = client.post("/a2a", json=rpc).json()
    assert res["id"] == 7 and res["result"]["parts"][1]["data"]["route"] == "data"


def test_rate_limit(client):
    app.state.limiter.per_minute = 2
    app.state.limiter._buckets.clear()
    codes = [client.get("/v1/tools", headers={"X-Client-Id": "burst"}).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes[2:]
    app.state.limiter.per_minute = 60
