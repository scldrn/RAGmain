from __future__ import annotations

import pytest

from agentic_rag import settings as settings_module
from agentic_rag.errors import ConfigurationError
from agentic_rag.settings import AgenticRagSettings


def test_settings_normalize_modes_and_provider_aliases():
    settings = AgenticRagSettings(
        source_urls=[" https://example.com/doc "],
        chat_provider="gemini",
        chat_model="gemini-2.5-flash",
        chat_api_key="chat-key",
        chat_api_base="http://localhost:1234/v1",
        embedding_provider="openai_compatible",
        embedding_model="text-embedding-3-small",
        embedding_api_key="embed-key",
        embedding_api_base="http://localhost:1234/v1",
        ingestion_mode="AUTO",
        retrieval_mode="HYBRID",
    )

    assert settings.source_urls == ("https://example.com/doc",)
    assert settings.chat_provider == "google"
    assert settings.embedding_provider == "openai-compatible"
    assert settings.ingestion_mode == "auto"
    assert settings.retrieval_mode == "hybrid"


def test_settings_reject_invalid_chunk_overlap():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chunk_size=100,
            chunk_overlap=100,
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="chat-key",
            embedding_provider="google",
            embedding_model="gemini-embedding-2-preview",
        )


def test_settings_reject_missing_embedding_provider():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="anthropic",
            chat_model="claude-3-5-sonnet-20241022",
            chat_api_key="anthropic-key",
            embedding_provider=None,
            embedding_model=None,
        )


def test_settings_reject_invalid_timeouts():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="chat-key",
            embedding_provider="google",
            embedding_model="gemini-embedding-2-preview",
            fetch_timeout_seconds=0,
        )


def _google_settings(**kwargs) -> AgenticRagSettings:
    defaults: dict[str, object] = dict(
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        chat_api_key="key",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
    )
    defaults.update(kwargs)
    return AgenticRagSettings(**defaults)


def test_settings_reject_chunk_size_zero():
    with pytest.raises(ConfigurationError):
        _google_settings(chunk_size=0)


def test_settings_reject_negative_retrieval_k():
    with pytest.raises(ConfigurationError):
        _google_settings(retrieval_k=0)


def test_settings_reject_temperature_above_range():
    with pytest.raises(ConfigurationError):
        _google_settings(chat_temperature=2.1)


def test_settings_reject_temperature_below_range():
    with pytest.raises(ConfigurationError):
        _google_settings(chat_temperature=-0.1)


def test_settings_reject_max_rewrites_zero():
    with pytest.raises(ConfigurationError):
        _google_settings(max_rewrites=0)


def test_settings_reject_model_timeout_zero():
    with pytest.raises(ConfigurationError):
        _google_settings(model_timeout_seconds=0.0)


def test_settings_reject_invalid_chat_provider():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="unsupported-provider",
            embedding_provider="google",
        )


def test_settings_reject_invalid_ingestion_mode():
    with pytest.raises(ConfigurationError):
        _google_settings(ingestion_mode="batch")


def test_settings_reject_invalid_retrieval_mode():
    with pytest.raises(ConfigurationError):
        _google_settings(retrieval_mode="sparse")


def test_settings_reject_empty_source_urls():
    with pytest.raises(ConfigurationError):
        _google_settings(source_urls=[])


def test_settings_reject_openai_compatible_without_chat_api_base():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="openai-compatible",
            chat_model="local-model",
            chat_api_key="key",
            chat_api_base=None,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_api_key="key",
            embedding_api_base=None,
            # no chat_api_base — should fail
        )


def test_read_float_env_raises_on_non_numeric(monkeypatch):
    monkeypatch.setenv("CHAT_TEMPERATURE", "not-a-number")
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="key",
            embedding_provider="google",
            embedding_model="gemini-embedding-2-preview",
        )


def test_default_chat_provider_reads_llm_provider_fallback(monkeypatch):
    monkeypatch.delenv("CHAT_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from agentic_rag.settings import default_chat_provider

    assert default_chat_provider() == "openai"


def test_default_embedding_provider_returns_none_for_anthropic(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "anthropic")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    from agentic_rag.settings import default_embedding_provider

    assert default_embedding_provider() is None


def test_default_embedding_provider_reads_explicit_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("CHAT_PROVIDER", raising=False)
    from agentic_rag.settings import default_embedding_provider

    assert default_embedding_provider() == "openai"


def test_default_embedding_provider_reuses_supported_chat_provider(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("CHAT_PROVIDER", "google")
    from agentic_rag.settings import default_embedding_provider

    assert default_embedding_provider() == "google"


def test_default_embedding_model_returns_none_without_provider(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "anthropic")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    from agentic_rag.settings import default_embedding_model

    assert default_embedding_model() is None


def test_read_int_env_raises_on_non_numeric(monkeypatch):
    monkeypatch.setenv("MAX_REWRITES", "not-a-number")
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="key",
            embedding_provider="google",
            embedding_model="gemini-embedding-2-preview",
        )


def test_settings_reject_invalid_embedding_provider():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="key",
            embedding_provider="unsupported-embedder",
            embedding_model="some-model",
        )


def test_settings_reject_empty_chat_model_when_default_missing(monkeypatch):
    monkeypatch.delitem(settings_module.DEFAULT_CHAT_MODELS, "google", raising=False)

    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="google",
            chat_model="   ",
            chat_api_key="key",
            embedding_provider="google",
            embedding_model="gemini-embedding-2-preview",
        )


def test_settings_reject_empty_embedding_model_when_default_missing(monkeypatch):
    monkeypatch.delitem(settings_module.DEFAULT_EMBEDDING_MODELS, "google", raising=False)

    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="key",
            embedding_provider="google",
            embedding_model="   ",
        )


def test_settings_reject_negative_chunk_overlap():
    with pytest.raises(ConfigurationError):
        _google_settings(chunk_overlap=-1)


def test_settings_reject_openai_compatible_embedding_without_effective_base():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="key",
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_api_key="key",
        )
