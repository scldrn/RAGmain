from __future__ import annotations

import runpy
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from agentic_rag.app import (
    AgenticRagApp,
    create_agentic_rag_app,
    export_graph_mermaid,
    run_question,
)
from agentic_rag.errors import DocumentLoadError
from agentic_rag.service import IndexStatus, QueryResult, QueryStep
from agentic_rag.settings import AgenticRagSettings


def _make_settings(tmp_path) -> AgenticRagSettings:
    return AgenticRagSettings(
        source_urls=("https://example.com/doc",),
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        chat_api_key="key",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
        index_cache_dir=str(tmp_path),
    )


def _make_index_status(tmp_path) -> IndexStatus:
    from pathlib import Path

    return IndexStatus(
        cache_path=Path(tmp_path) / "fake-index",
        fingerprint="abc123",
        exists=True,
        document_count=5,
    )


def test_create_agentic_rag_app_delegates_to_service(monkeypatch, tmp_path):
    settings = _make_settings(tmp_path)
    status = _make_index_status(tmp_path)

    mock_service = MagicMock()
    mock_service.ingest.return_value = status
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr("agentic_rag.app.AgenticRagService", MockServiceClass)

    app = create_agentic_rag_app(settings)

    mock_service.ingest.assert_called_once()
    assert app.document_count == 5
    assert app.settings is settings


def test_create_agentic_rag_app_uses_default_settings_when_none(monkeypatch, tmp_path):
    status = _make_index_status(tmp_path)

    mock_service = MagicMock()
    mock_service.ingest.return_value = status
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr("agentic_rag.app.AgenticRagService", MockServiceClass)
    monkeypatch.setattr(
        "agentic_rag.app.AgenticRagSettings",
        lambda: _make_settings(tmp_path),
    )

    app = create_agentic_rag_app()
    assert app.document_count == 5


def test_run_question_returns_final_message(monkeypatch, tmp_path):
    settings = _make_settings(tmp_path)
    status = _make_index_status(tmp_path)
    expected_message = AIMessage(content="Final answer")
    query_result = QueryResult(
        final_message=expected_message,
        answer_text="Final answer",
        document_count=5,
        index_status=status,
    )

    mock_service = MagicMock()
    mock_service.query.return_value = query_result
    app = AgenticRagApp(settings=settings, service=mock_service, document_count=5)

    result = run_question(app, "What is reward hacking?")
    assert result is expected_message


def test_run_question_with_show_steps_passes_callback(monkeypatch, tmp_path):
    settings = _make_settings(tmp_path)
    status = _make_index_status(tmp_path)
    step = QueryStep(node_name="generate_answer", message=AIMessage(content="step"))
    query_result = QueryResult(
        final_message=AIMessage(content="Answer"),
        answer_text="Answer",
        document_count=5,
        index_status=status,
        steps=(step,),
    )

    def fake_query(q, on_step=None):
        if on_step:
            on_step(step)
        return query_result

    mock_service = MagicMock()
    mock_service.query.side_effect = fake_query
    app = AgenticRagApp(settings=settings, service=mock_service, document_count=5)

    run_question(app, "q", show_steps=True)
    _, kwargs = mock_service.query.call_args
    assert kwargs.get("on_step") is not None


def test_run_question_with_show_steps_prints_tool_calls(capsys, tmp_path):
    settings = _make_settings(tmp_path)
    status = _make_index_status(tmp_path)
    step = QueryStep(
        node_name="retrieve",
        message=AIMessage(
            content="step",
            tool_calls=[
                {
                    "id": "tool-1",
                    "name": "retrieve_blog_posts",
                    "args": {"query": "reward hacking"},
                }
            ],
        ),
    )
    query_result = QueryResult(
        final_message=AIMessage(content="Answer"),
        answer_text="Answer",
        document_count=5,
        index_status=status,
        steps=(step,),
    )

    def fake_query(q, on_step=None):
        if on_step:
            on_step(step)
        return query_result

    mock_service = MagicMock()
    mock_service.query.side_effect = fake_query
    app = AgenticRagApp(settings=settings, service=mock_service, document_count=5)

    run_question(app, "q", show_steps=True)
    captured = capsys.readouterr()

    assert "tool: retrieve_blog_posts" in captured.out


def test_export_graph_mermaid_delegates(tmp_path):
    settings = _make_settings(tmp_path)
    mock_service = MagicMock()
    app = AgenticRagApp(settings=settings, service=mock_service, document_count=0)

    export_graph_mermaid(app, "/tmp/graph.md")
    mock_service.export_graph_mermaid.assert_called_once_with("/tmp/graph.md")


def test_create_agentic_rag_app_propagates_ingest_error(monkeypatch, tmp_path):
    settings = _make_settings(tmp_path)

    mock_service = MagicMock()
    mock_service.ingest.side_effect = DocumentLoadError("network fail")
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr("agentic_rag.app.AgenticRagService", MockServiceClass)

    with pytest.raises(DocumentLoadError):
        create_agentic_rag_app(settings)


def test_create_agentic_rag_app_explicit_mode_does_not_ingest(monkeypatch, tmp_path):
    settings = AgenticRagSettings(
        **{**_make_settings(tmp_path).__dict__, "ingestion_mode": "explicit"}
    )
    status = IndexStatus(
        cache_path=_make_index_status(tmp_path).cache_path,
        fingerprint="abc123",
        exists=False,
        document_count=None,
    )

    mock_service = MagicMock()
    mock_service.index_status.return_value = status
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr("agentic_rag.app.AgenticRagService", MockServiceClass)

    app = create_agentic_rag_app(settings)

    mock_service.ingest.assert_not_called()
    mock_service.index_status.assert_called_once_with()
    assert app.document_count == 0


def test_package_main_exits_with_cli_status(monkeypatch):
    monkeypatch.setattr("agentic_rag.cli.main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("agentic_rag.__main__", run_name="__main__")

    assert exc_info.value.code == 7
