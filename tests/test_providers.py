from __future__ import annotations

import pytest

from agentic_rag.providers import resolve_chat_config, resolve_embedding_config
from agentic_rag.settings import AgenticRagSettings, normalize_provider_name


def test_google_chat_config_accepts_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    settings = AgenticRagSettings(
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
    )

    config = resolve_chat_config(settings)

    assert config.api_key == "gemini-key"


def test_openai_compatible_embedding_inherits_chat_credentials():
    settings = AgenticRagSettings(
        chat_provider="openai-compatible",
        chat_model="local-chat-model",
        chat_api_key="local-key",
        chat_api_base="http://localhost:1234/v1",
        embedding_provider="openai-compatible",
        embedding_model="local-embedding-model",
    )

    chat_config = resolve_chat_config(settings)
    embedding_config = resolve_embedding_config(settings, chat_config)

    assert embedding_config.api_key == "local-key"
    assert embedding_config.api_base == "http://localhost:1234/v1"


def test_anthropic_requires_explicit_embedding_provider():
    settings = AgenticRagSettings(
        chat_provider="anthropic",
        chat_model="claude-3-5-sonnet-20241022",
        chat_api_key="anthropic-key",
        embedding_provider=None,
        embedding_model=None,
    )

    chat_config = resolve_chat_config(settings)

    with pytest.raises(RuntimeError):
        resolve_embedding_config(settings, chat_config)


def test_openai_compatible_requires_api_base():
    settings = AgenticRagSettings(
        chat_provider="openai-compatible",
        chat_model="local-model",
        chat_api_key="key",
    )

    with pytest.raises(RuntimeError):
        resolve_chat_config(settings)


def test_normalize_provider_name_handles_aliases():
    assert normalize_provider_name("openai_compatible") == "openai-compatible"
    assert normalize_provider_name("gemini") == "google"
