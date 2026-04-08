from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_rag.cli import (
    _default_embedding_provider,
    build_parser,
    build_settings,
    configure_logging,
    configure_runtime_warnings,
    log_startup_diagnostics,
    main,
    normalize_cli_argv,
)
from agentic_rag.errors import AgenticRagError, ConfigurationError, LLMError
from agentic_rag.service import HealthStatus, IndexStatus

# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_build_parser_requires_question():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])
    assert exc_info.value.code == 2


def test_build_parser_accepts_query_question():
    parser = build_parser()
    args = parser.parse_args(["query", "--question", "What is reward hacking?"])
    assert args.command == "query"
    assert args.question == "What is reward hacking?"


def test_build_parser_accepts_ingest_command():
    parser = build_parser()
    args = parser.parse_args(["ingest"])

    assert args.command == "ingest"


def test_build_parser_accepts_multiple_urls():
    parser = build_parser()
    args = parser.parse_args(
        ["query", "--question", "q", "--url", "https://a.com", "--url", "https://b.com"]
    )
    assert args.source_urls == ["https://a.com", "https://b.com"]


def test_build_parser_show_steps_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args(["query", "--question", "q"])
    assert args.show_steps is False
    assert args.verbose is False


def test_build_parser_accepts_verbose_flag():
    parser = build_parser()

    query_args = parser.parse_args(["query", "--question", "q", "--verbose"])
    ingest_args = parser.parse_args(["ingest", "--verbose"])

    assert query_args.verbose is True
    assert ingest_args.verbose is True


def test_normalize_cli_argv_preserves_explicit_commands():
    assert normalize_cli_argv(["ingest"]) == ["ingest"]
    assert normalize_cli_argv(["query", "--question", "q"]) == ["query", "--question", "q"]


def test_normalize_cli_argv_defaults_empty_input_to_query():
    assert normalize_cli_argv([]) == ["query"]


def test_normalize_cli_argv_maps_legacy_query_flags():
    assert normalize_cli_argv(["--question", "q"]) == ["query", "--question", "q"]


def test_normalize_cli_argv_maps_legacy_verbose_query_flags():
    assert normalize_cli_argv(["--verbose", "--question", "q"]) == [
        "query",
        "--verbose",
        "--question",
        "q",
    ]


# ---------------------------------------------------------------------------
# build_settings
# ---------------------------------------------------------------------------


def test_build_settings_uses_cli_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "query",
            "--question",
            "test",
            "--chat-provider",
            "openai",
            "--chat-model",
            "gpt-4.1-mini",
            "--chat-api-key",
            "sk-test",
            "--embedding-provider",
            "openai",
            "--embedding-model",
            "text-embedding-3-small",
            "--embedding-api-key",
            "sk-test",
        ]
    )
    settings = build_settings(args)
    assert settings.chat_provider == "openai"
    assert settings.chat_model == "gpt-4.1-mini"
    assert settings.chat_api_key == "sk-test"


def test_build_settings_uses_default_urls_when_none_provided():
    parser = build_parser()
    args = parser.parse_args(
        [
            "query",
            "--question",
            "test",
            "--chat-provider",
            "google",
            "--chat-api-key",
            "key",
            "--embedding-provider",
            "google",
        ]
    )
    settings = build_settings(args)
    assert len(settings.source_urls) > 0


def test_build_settings_allows_cli_embedding_overrides_over_invalid_env(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "anthropic")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "query",
            "--question",
            "test",
            "--embedding-provider",
            "openai",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    settings = build_settings(args)

    assert settings.chat_provider == "anthropic"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"


def test_default_embedding_provider_falls_back_to_chat_provider(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    assert _default_embedding_provider("google") == "google"


def test_default_embedding_provider_returns_none_for_chat_only_provider(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    assert _default_embedding_provider("anthropic") is None


# ---------------------------------------------------------------------------
# configure_runtime_warnings
# ---------------------------------------------------------------------------


def test_configure_logging_uses_warning_level_by_default(monkeypatch):
    basic_config_calls = []
    logger_mock = MagicMock()

    monkeypatch.setattr(
        "agentic_rag.cli.logging.basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )
    monkeypatch.setattr("agentic_rag.cli.logging.getLogger", lambda *args: logger_mock)

    configure_logging(verbose=False)

    assert basic_config_calls == [
        {
            "level": 30,
            "format": "%(levelname)s %(name)s: %(message)s",
        }
    ]
    logger_mock.setLevel.assert_called_once_with(30)


def test_configure_logging_uses_debug_level_when_verbose(monkeypatch):
    basic_config_calls = []
    logger_mock = MagicMock()

    monkeypatch.setattr(
        "agentic_rag.cli.logging.basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )
    monkeypatch.setattr("agentic_rag.cli.logging.getLogger", lambda *args: logger_mock)

    configure_logging(verbose=True)

    assert basic_config_calls == [
        {
            "level": 10,
            "format": "%(levelname)s %(name)s: %(message)s",
        }
    ]
    logger_mock.setLevel.assert_called_once_with(10)


def test_configure_runtime_warnings_runs_without_error(monkeypatch):
    monkeypatch.delenv("AGENTIC_RAG_DEBUG", raising=False)
    configure_runtime_warnings()  # must not raise


def test_configure_runtime_warnings_skips_in_debug_mode(monkeypatch):
    monkeypatch.setenv("AGENTIC_RAG_DEBUG", "1")
    configure_runtime_warnings()  # must not raise


def test_log_startup_diagnostics_logs_expected_context(caplog):
    health = HealthStatus(
        ok=True,
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
        ingestion_mode="auto",
        retrieval_mode="dense",
        index_cache_dir=Path("/tmp/cache"),
    )
    index_status = IndexStatus(
        cache_path=Path("/tmp/cache/index"),
        fingerprint="abc123",
        exists=True,
        document_count=7,
    )

    with caplog.at_level("INFO", logger="agentic_rag.cli"):
        log_startup_diagnostics(
            MagicMock(),
            command="query",
            health=health,
            index_status=index_status,
        )

    assert "startup command=query" in caplog.text
    assert "chat=google/gemini-2.5-flash" in caplog.text
    assert "startup_index cache=/tmp/cache/index" in caplog.text
    assert "document_count=7" in caplog.text


def test_log_startup_diagnostics_tolerates_index_status_failures(caplog):
    health = HealthStatus(
        ok=True,
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
        ingestion_mode="auto",
        retrieval_mode="dense",
        index_cache_dir=Path("/tmp/cache"),
    )
    service = MagicMock()
    service.health.return_value = health
    service.index_status.side_effect = OSError("bad cache")

    with caplog.at_level("WARNING", logger="agentic_rag.cli"):
        log_startup_diagnostics(service, command="query")

    assert "startup_index_status_unavailable" in caplog.text


# ---------------------------------------------------------------------------
# main — happy path
# ---------------------------------------------------------------------------


def _make_mock_service(answer: str = "42") -> MagicMock:
    mock_result = MagicMock()
    mock_result.answer_text = answer

    service = MagicMock()
    service.query.return_value = mock_result
    return service


def test_main_prints_answer_on_success(monkeypatch, capsys):
    mock_service = _make_mock_service("The answer is 42.")
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "The answer is 42." in captured.out


def test_main_exports_diagram_when_flag_set(monkeypatch, tmp_path):
    diagram_file = str(tmp_path / "graph.md")
    mock_service = _make_mock_service()
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "q", "--diagram", diagram_file])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    exit_code = main()
    assert exit_code == 0
    mock_service.export_graph_mermaid.assert_called_once_with(diagram_file)


def test_main_runs_explicit_ingest_command(monkeypatch, capsys):
    mock_status = MagicMock()
    mock_status.document_count = 5
    mock_status.cache_path = "/tmp/index"
    mock_status.fingerprint = "abc123"

    mock_service = MagicMock()
    mock_service.ingest.return_value = mock_status
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "ingest"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Index ready" in captured.out
    mock_service.ingest.assert_called_once_with()
    mock_service.query.assert_not_called()


def test_main_verbose_configures_logging(monkeypatch):
    mock_service = _make_mock_service()
    MockServiceClass = MagicMock(return_value=mock_service)
    configure_calls = []

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--verbose", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)
    monkeypatch.setattr(
        "agentic_rag.cli.configure_logging",
        lambda *, verbose: configure_calls.append(verbose),
    )

    exit_code = main()

    assert exit_code == 0
    assert configure_calls == [True]


def test_main_skips_startup_diagnostics_without_verbose(monkeypatch):
    mock_service = _make_mock_service()
    MockServiceClass = MagicMock(return_value=mock_service)
    log_calls = []

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)
    monkeypatch.setattr(
        "agentic_rag.cli.log_startup_diagnostics",
        lambda *_args, **_kwargs: log_calls.append("called"),
    )

    exit_code = main()

    assert exit_code == 0
    assert log_calls == []


def test_main_verbose_diagnostics_do_not_block_query(monkeypatch, capsys):
    mock_result = MagicMock()
    mock_result.answer_text = "The answer is 42."
    mock_service = MagicMock()
    mock_service.health.return_value = HealthStatus(
        ok=True,
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
        ingestion_mode="auto",
        retrieval_mode="dense",
        index_cache_dir=Path("/tmp/cache"),
    )
    mock_service.index_status.side_effect = OSError("bad cache")
    mock_service.query.return_value = mock_result
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--verbose", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "The answer is 42." in captured.out
    mock_service.query.assert_called_once_with("What?", on_step=None)


# ---------------------------------------------------------------------------
# main — error paths (build_settings inside try after our fix)
# ---------------------------------------------------------------------------


def test_main_exits_2_on_configuration_error_from_build_settings(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.setattr(
        "agentic_rag.cli.build_settings",
        lambda _args: (_ for _ in ()).throw(ConfigurationError("bad config")),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_main_exits_1_on_llm_error(monkeypatch):
    mock_service = MagicMock()
    mock_service.query.side_effect = LLMError("model failed")
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_exits_1_on_generic_rag_error(monkeypatch):
    mock_service = MagicMock()
    mock_service.query.side_effect = AgenticRagError("rag failure")
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_exits_1_on_unexpected_error(monkeypatch):
    mock_service = MagicMock()
    mock_service.query.side_effect = RuntimeError("unexpected")
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_show_steps_calls_on_step(monkeypatch, capsys):
    from langchain_core.messages import AIMessage

    from agentic_rag.service import QueryStep

    step = QueryStep(node_name="generate_answer", message=AIMessage(content="step text"))
    mock_result = MagicMock()
    mock_result.answer_text = "done"

    def fake_query(q, on_step=None):
        if on_step:
            on_step(step)
        return mock_result

    mock_service = MagicMock()
    mock_service.query.side_effect = fake_query
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "q", "--show-steps"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "generate_answer" in captured.out


def test_main_show_steps_prints_tool_call_details(monkeypatch, capsys):
    from langchain_core.messages import AIMessage

    from agentic_rag.service import QueryStep

    step = QueryStep(
        node_name="retrieve",
        message=AIMessage(
            content="step text",
            tool_calls=[
                {
                    "id": "tool-1",
                    "name": "retrieve_blog_posts",
                    "args": {"query": "reward hacking"},
                }
            ],
        ),
    )
    mock_result = MagicMock()
    mock_result.answer_text = "done"

    def fake_query(q, on_step=None):
        if on_step:
            on_step(step)
        return mock_result

    mock_service = MagicMock()
    mock_service.query.side_effect = fake_query
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "q", "--show-steps"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "tool: retrieve_blog_posts" in captured.out


def test_main_reraises_unexpected_error_in_debug_mode(monkeypatch):
    mock_service = MagicMock()
    mock_service.query.side_effect = RuntimeError("unexpected")
    MockServiceClass = MagicMock(return_value=mock_service)

    monkeypatch.setenv("AGENTIC_RAG_DEBUG", "1")
    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.setattr("agentic_rag.cli.build_settings", lambda _args: MagicMock())
    monkeypatch.setattr("agentic_rag.cli.AgenticRagService", MockServiceClass)

    with pytest.raises(RuntimeError, match="unexpected"):
        main()


def test_cli_module_main_guard(monkeypatch):
    class FakeService:
        def __init__(self, settings):
            self.settings = settings

        def health(self):
            return HealthStatus(
                ok=True,
                chat_provider="google",
                chat_model="gemini-2.5-flash",
                embedding_provider="google",
                embedding_model="gemini-embedding-2-preview",
                ingestion_mode="auto",
                retrieval_mode="dense",
                index_cache_dir=Path(".cache/vectorstores"),
            )

        def index_status(self):
            return IndexStatus(
                cache_path=Path(".cache/vectorstores/fake-index"),
                fingerprint="abc123",
                exists=False,
                document_count=None,
            )

        def export_graph_mermaid(self, _path):
            return None

        def query(self, question, on_step=None):
            result = MagicMock()
            result.answer_text = f"answer:{question}"
            result.termination_reason = None
            result.rewrites_used = 0
            return result

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("AGENTIC_RAG_DEBUG", raising=False)
    monkeypatch.setattr("agentic_rag.service.AgenticRagService", FakeService)
    monkeypatch.setattr(sys, "argv", ["agentic-rag", "--question", "What?"])
    monkeypatch.delitem(sys.modules, "agentic_rag.cli", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("agentic_rag.cli", run_name="__main__")

    assert exc_info.value.code == 0
