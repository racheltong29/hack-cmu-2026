import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGAR_DB_PATH", str(tmp_path / "test.db"))
    from edgar_analyst.config import get_settings

    get_settings.cache_clear()
    from edgar_analyst.api.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health_reports_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0
    assert body["llm"] == "stub"


def test_ask_returns_cited_answer(client):
    resp = client.post("/ask", json={"question": "What was NOVA's revenue in fiscal 2025?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "20.1" in body["answer"]
    assert body["citations"]


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_ask_stream_emits_trace_and_answer(client):
    with client.stream(
        "POST", "/ask/stream", json={"question": "What risks does BLDR face from aluminum costs?"}
    ) as resp:
        assert resp.status_code == 200
        payload = "".join(resp.iter_text())
    assert "event: trace" in payload
    assert "event: answer" in payload
    assert "event: done" in payload
    assert "verifier" in payload


def test_ingest_fixture_is_idempotent(client):
    first = client.post("/ingest", json={}).json()
    second = client.post("/ingest", json={}).json()
    assert first["total_chunks"] == second["total_chunks"]
