from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agentic_rag.api import _query_trace_step, create_api_app, run
from agentic_rag.errors import ConfigurationError, LLMError
from agentic_rag.service import AgenticRagService, HealthStatus, IndexStatus, QueryResult, QueryStep
from agentic_rag.settings import AgenticRagSettings


def make_settings(tmp_path) -> AgenticRagSettings:
    return AgenticRagSettings(
        source_urls=("https://example.com/doc",),
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        chat_api_key="chat-key",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
        index_cache_dir=str(tmp_path),
        collection_name="documents",
    )


def make_index_status(tmp_path, *, collection_name: str, exists: bool = True) -> IndexStatus:
    return IndexStatus(
        cache_path=Path(tmp_path) / collection_name,
        fingerprint=f"fingerprint-{collection_name}",
        exists=exists,
        document_count=3 if exists else None,
        collection_name=collection_name,
    )


def test_query_endpoint_returns_structured_trace_and_metrics(monkeypatch, tmp_path):
    def fake_query(self, question: str):
        assert question == "What is reward hacking?"
        assert self.settings.collection_name == "team-a"
        return QueryResult(
            final_message=AIMessage(content="Reward hacking exploits flaws in the reward signal."),
            answer_text="Reward hacking exploits flaws in the reward signal.",
            document_count=3,
            index_status=make_index_status(tmp_path, collection_name=self.settings.collection_name),
            steps=(
                QueryStep(
                    node_name="prepare_retrieval",
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "retrieve-0",
                                "name": "retrieve_blog_posts",
                                "args": {"query": question},
                                "type": "tool_call",
                            }
                        ],
                    ),
                ),
                QueryStep(
                    node_name="generate_answer",
                    message=AIMessage(
                        content="Reward hacking exploits flaws in the reward signal."
                    ),
                ),
            ),
            rewrites_used=0,
        )

    monkeypatch.setattr(AgenticRagService, "query", fake_query)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.post(
        "/query",
        json={"question": "What is reward hacking?", "collection_name": "team-a"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["answer"] == "Reward hacking exploits flaws in the reward signal."
    assert payload["trace"]["step_count"] == 2
    assert payload["trace"]["steps"][0]["tool_calls"][0]["name"] == "retrieve_blog_posts"
    assert payload["metrics"]["collection_name"] == "team-a"
    assert payload["index_status"]["collection_name"] == "team-a"


def test_query_endpoint_returns_null_answer_for_insufficient_context(monkeypatch, tmp_path):
    def fake_query(self, question: str):
        return QueryResult(
            final_message=AIMessage(
                content="I do not have enough relevant context to answer reliably."
            ),
            answer_text="I do not have enough relevant context to answer reliably.",
            document_count=0,
            index_status=make_index_status(tmp_path, collection_name=self.settings.collection_name),
            steps=(),
            termination_reason="insufficient_context",
            rewrites_used=2,
        )

    monkeypatch.setattr(AgenticRagService, "query", fake_query)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.post("/query", json={"question": "Unknown question"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient_context"
    assert payload["answer"] is None
    assert payload["termination_reason"] == "insufficient_context"


def test_ingest_endpoint_returns_index_status(monkeypatch, tmp_path):
    def fake_ingest(self):
        assert self.settings.collection_name == "knowledge-base"
        return make_index_status(tmp_path, collection_name=self.settings.collection_name)

    monkeypatch.setattr(AgenticRagService, "ingest", fake_ingest)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.post("/ingest", json={"collection_name": "knowledge-base"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["index_status"]["collection_name"] == "knowledge-base"
    assert payload["metrics"]["collection_name"] == "knowledge-base"


def test_ingest_endpoint_accepts_missing_body_and_uses_default_collection(monkeypatch, tmp_path):
    def fake_ingest(self):
        assert self.settings.collection_name == "documents"
        return make_index_status(tmp_path, collection_name=self.settings.collection_name)

    monkeypatch.setattr(AgenticRagService, "ingest", fake_ingest)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.post("/ingest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["index_status"]["collection_name"] == "documents"


def test_health_endpoint_reports_collection_override(monkeypatch, tmp_path):
    def fake_health(self):
        return HealthStatus(
            ok=True,
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            embedding_provider="google",
            embedding_model="gemini-embedding-2-preview",
            ingestion_mode="auto",
            retrieval_mode="dense",
            index_cache_dir=Path(self.settings.index_cache_dir),
            collection_name=self.settings.collection_name,
        )

    monkeypatch.setattr(AgenticRagService, "health", fake_health)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.get("/health", params={"collection_name": "docs-v2"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["collection_name"] == "docs-v2"
    assert payload["metrics"]["collection_name"] == "docs-v2"


def test_index_status_endpoint_reports_collection_override(monkeypatch, tmp_path):
    def fake_index_status(self):
        return make_index_status(
            tmp_path,
            collection_name=self.settings.collection_name,
            exists=False,
        )

    monkeypatch.setattr(AgenticRagService, "index_status", fake_index_status)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.get("/index/status", params={"collection_name": "docs-v3"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["index_status"]["exists"] is False
    assert payload["index_status"]["collection_name"] == "docs-v3"


def test_api_maps_configuration_errors_to_400(monkeypatch, tmp_path):
    monkeypatch.setattr(
        AgenticRagService,
        "query",
        lambda self, question: (_ for _ in ()).throw(ConfigurationError("bad request")),
    )

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.post("/query", json={"question": "bad"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_type"] == "ConfigurationError"
    assert payload["detail"] == "bad request"
    assert payload["request_id"]


def test_api_maps_domain_errors_to_503(monkeypatch, tmp_path):
    monkeypatch.setattr(
        AgenticRagService,
        "ingest",
        lambda self: (_ for _ in ()).throw(LLMError("upstream unavailable")),
    )

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.post("/ingest", json={})

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_type"] == "LLMError"
    assert payload["detail"] == "upstream unavailable"


def test_api_rejects_empty_collection_override_in_post_body(monkeypatch, tmp_path):
    query_called = {"value": False}

    def fake_query(self, question: str):
        query_called["value"] = True
        raise AssertionError("query should not be called for invalid collection_name")

    monkeypatch.setattr(AgenticRagService, "query", fake_query)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.post("/query", json={"question": "bad", "collection_name": ""})

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_type"] == "ConfigurationError"
    assert "collection_name" in payload["detail"]
    assert query_called["value"] is False


def test_api_rejects_empty_collection_override_in_get_query(monkeypatch, tmp_path):
    health_called = {"value": False}

    def fake_health(self):
        health_called["value"] = True
        raise AssertionError("health should not be called for invalid collection_name")

    monkeypatch.setattr(AgenticRagService, "health", fake_health)

    client = TestClient(create_api_app(make_settings(tmp_path)))
    response = client.get("/health", params={"collection_name": ""})

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_type"] == "ConfigurationError"
    assert "collection_name" in payload["detail"]
    assert health_called["value"] is False


def test_api_enables_cors_for_configured_origins(tmp_path):
    settings = AgenticRagSettings(
        **{
            **make_settings(tmp_path).__dict__,
            "cors_allow_origins": ("http://localhost:3000",),
        }
    )
    client = TestClient(create_api_app(settings))
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "GET" in response.headers["access-control-allow-methods"]


def test_query_trace_step_ignores_non_dict_tool_calls():
    step = QueryStep(
        node_name="prepare_retrieval",
        message=SimpleNamespace(content="trace message", tool_calls=["skip-me"]),
    )

    trace_step = _query_trace_step(step)

    assert trace_step.message == "trace message"
    assert trace_step.tool_calls == []


def test_api_run_starts_uvicorn(monkeypatch):
    uvicorn_calls = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: uvicorn_calls.append((args, kwargs))),
    )

    run()

    assert uvicorn_calls == [
        (
            ("agentic_rag.api:create_api_app",),
            {
                "factory": True,
                "host": "127.0.0.1",
                "port": 8000,
            },
        )
    ]


def test_api_module_main_guard(monkeypatch):
    uvicorn_calls = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: uvicorn_calls.append((args, kwargs))),
    )

    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "src/agentic_rag/api.py"),
        run_name="__main__",
    )

    assert uvicorn_calls
