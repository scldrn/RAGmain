from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.errors import ConfigurationError
from agentic_rag.providers import (
    LiteLLMEmbeddingsAdapter,
    ResolvedProviderConfig,
    _credential_hint,
    _normalize_supported_provider,
    _resolve_api_key,
    _supported_providers_text,
    create_chat_model,
    create_embeddings,
    resolve_chat_config,
    resolve_embedding_config,
)
from agentic_rag.settings import AgenticRagSettings, normalize_provider_name


def _config(
    provider: str, model: str = "test-model", api_key: str = "key"
) -> ResolvedProviderConfig:
    return ResolvedProviderConfig(provider=provider, model=model, api_key=api_key)


def _mock_modules(**mocks: MagicMock):
    """Return a sys.modules patch dict with the given module mocks."""
    return patch.dict("sys.modules", mocks)


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
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="anthropic",
            chat_model="claude-3-5-sonnet-20241022",
            chat_api_key="anthropic-key",
            embedding_provider=None,
            embedding_model=None,
        )


def test_openai_compatible_requires_api_base():
    with pytest.raises(ConfigurationError):
        AgenticRagSettings(
            chat_provider="openai-compatible",
            chat_model="local-model",
            chat_api_key="key",
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
        )


def test_normalize_provider_name_handles_aliases():
    assert normalize_provider_name("openai_compatible") == "openai-compatible"
    assert normalize_provider_name("gemini") == "google"


def test_resolve_chat_config_raises_typed_error_for_missing_credentials():
    settings = AgenticRagSettings(
        chat_provider="openai",
        chat_model="gpt-4.1-mini",
        chat_api_key=None,
        chat_api_key_env=None,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_api_key="embedding-key",
    )

    with pytest.raises(ConfigurationError):
        resolve_chat_config(settings)


# ---------------------------------------------------------------------------
# create_chat_model — provider factory tests (all providers mocked)
# ---------------------------------------------------------------------------


def test_create_chat_model_google():
    mock_google = MagicMock(ChatGoogleGenerativeAI=MagicMock(return_value=MagicMock()))
    with _mock_modules(langchain_google_genai=mock_google):
        config = _config("google")
        model = create_chat_model(config)
        assert model is not None


def test_create_chat_model_openai():
    mock_openai = MagicMock(ChatOpenAI=MagicMock(return_value=MagicMock()))
    with _mock_modules(langchain_openai=mock_openai):
        config = _config("openai")
        model = create_chat_model(config)
        assert model is not None


def test_create_chat_model_openai_compatible():
    mock_openai = MagicMock(ChatOpenAI=MagicMock(return_value=MagicMock()))
    with _mock_modules(langchain_openai=mock_openai):
        config = ResolvedProviderConfig(
            provider="openai-compatible",
            model="local-model",
            api_key="key",
            api_base="http://localhost:1234/v1",
        )
        model = create_chat_model(config)
        assert model is not None


def test_create_chat_model_anthropic():
    mock_anthropic = MagicMock(ChatAnthropic=MagicMock(return_value=MagicMock()))
    with _mock_modules(langchain_anthropic=mock_anthropic):
        config = _config("anthropic")
        model = create_chat_model(config)
        assert model is not None


def test_create_chat_model_litellm():
    mock_litellm_pkg = MagicMock(ChatLiteLLM=MagicMock(return_value=MagicMock()))
    with _mock_modules(langchain_litellm=mock_litellm_pkg):
        config = _config("litellm", model="openai/gpt-4.1-mini", api_key=None)
        model = create_chat_model(config)
        assert model is not None


def test_create_chat_model_unsupported_raises():
    config = ResolvedProviderConfig(provider="unsupported", model="x")
    with pytest.raises(ConfigurationError):
        create_chat_model(config)


# ---------------------------------------------------------------------------
# create_embeddings — provider factory tests
# ---------------------------------------------------------------------------


def test_create_embeddings_google():
    mock_google = MagicMock(GoogleGenerativeAIEmbeddings=MagicMock(return_value=MagicMock()))
    with _mock_modules(langchain_google_genai=mock_google):
        config = _config("google")
        emb = create_embeddings(config)
        assert emb is not None


def test_create_embeddings_openai():
    mock_openai = MagicMock(OpenAIEmbeddings=MagicMock(return_value=MagicMock()))
    with _mock_modules(langchain_openai=mock_openai):
        config = _config("openai")
        emb = create_embeddings(config)
        assert emb is not None


def test_create_embeddings_litellm():
    config = ResolvedProviderConfig(provider="litellm", model="openai/text-embedding-3-small")
    emb = create_embeddings(config)
    assert isinstance(emb, LiteLLMEmbeddingsAdapter)


def test_create_embeddings_unsupported_raises():
    config = ResolvedProviderConfig(provider="unsupported", model="x")
    with pytest.raises(ConfigurationError):
        create_embeddings(config)


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingsAdapter
# ---------------------------------------------------------------------------


def test_litellm_embeddings_adapter_embed_documents():
    fake_item = MagicMock()
    fake_item.embedding = [0.1, 0.2, 0.3]
    fake_response = MagicMock()
    fake_response.data = [fake_item]

    mock_litellm = MagicMock()
    mock_litellm.embedding.return_value = fake_response

    adapter = LiteLLMEmbeddingsAdapter(model="openai/text-embedding-3-small", api_key="key")

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        result = adapter.embed_documents(["hello world"])

    assert result == [[0.1, 0.2, 0.3]]
    mock_litellm.embedding.assert_called_once_with(
        model="openai/text-embedding-3-small",
        input=["hello world"],
        api_key="key",
        api_base=None,
    )


def test_litellm_embeddings_adapter_embed_query():
    fake_item = MagicMock()
    fake_item.embedding = [0.4, 0.5]
    fake_response = MagicMock()
    fake_response.data = [fake_item]

    mock_litellm = MagicMock()
    mock_litellm.embedding.return_value = fake_response

    adapter = LiteLLMEmbeddingsAdapter(model="openai/text-embedding-3-small")

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        result = adapter.embed_query("hello")

    assert result == [0.4, 0.5]


def test_litellm_embeddings_adapter_embedding_from_item_dict():
    result = LiteLLMEmbeddingsAdapter._embedding_from_item({"embedding": [0.1, 0.2, 0.3]})
    assert result == [0.1, 0.2, 0.3]


def test_litellm_embeddings_adapter_embedding_from_item_object():
    item = MagicMock()
    item.embedding = [0.4, 0.5]
    result = LiteLLMEmbeddingsAdapter._embedding_from_item(item)
    assert result == [0.4, 0.5]


def test_resolve_embedding_config_missing_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = AgenticRagSettings(
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        chat_api_key="chat-key",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_api_key=None,
        embedding_api_key_env=None,
    )

    chat_config = resolve_chat_config(settings)
    with pytest.raises(ConfigurationError):
        resolve_embedding_config(settings, chat_config)


def test_supported_providers_text_joins_values():
    assert _supported_providers_text(("a", "b")) == "a, b"


def test_normalize_supported_provider_none_returns_none():
    assert _normalize_supported_provider(None, kind="chat") is None


def test_normalize_supported_provider_rejects_unsupported():
    with pytest.raises(ConfigurationError):
        _normalize_supported_provider("bad-provider", kind="chat")


def test_resolve_api_key_prefers_explicit_env_name(monkeypatch):
    monkeypatch.setenv("CUSTOM_KEY_ENV", "custom-key")

    assert (
        _resolve_api_key(
            "openai",
            explicit_value=None,
            explicit_env_name="CUSTOM_KEY_ENV",
        )
        == "custom-key"
    )


def test_credential_hint_handles_litellm():
    assert "CHAT_API_KEY" in _credential_hint("litellm")


def test_resolve_chat_config_raises_when_provider_is_none():
    settings = SimpleNamespace(
        chat_provider=None,
        chat_model="model",
        chat_api_key=None,
        chat_api_key_env=None,
        chat_api_base=None,
        chat_temperature=0.0,
    )

    with pytest.raises(ConfigurationError):
        resolve_chat_config(settings)


def test_resolve_chat_config_openai_compatible_requires_api_base_with_simple_namespace():
    settings = SimpleNamespace(
        chat_provider="openai-compatible",
        chat_model="local-model",
        chat_api_key="key",
        chat_api_key_env=None,
        chat_api_base=None,
        chat_temperature=0.0,
    )

    with pytest.raises(ConfigurationError):
        resolve_chat_config(settings)


def test_resolve_embedding_config_requires_provider_when_none():
    settings = SimpleNamespace(
        embedding_provider=None,
        embedding_model=None,
        embedding_api_key=None,
        embedding_api_key_env=None,
        embedding_api_base=None,
    )
    chat_config = ResolvedProviderConfig(provider="anthropic", model="claude")

    with pytest.raises(ConfigurationError):
        resolve_embedding_config(settings, chat_config)


def test_resolve_embedding_config_openai_compatible_requires_api_base_with_simple_namespace():
    settings = SimpleNamespace(
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_api_key="key",
        embedding_api_key_env=None,
        embedding_api_base=None,
    )
    chat_config = ResolvedProviderConfig(provider="openai", model="gpt-4.1-mini", api_key="key")

    with pytest.raises(ConfigurationError):
        resolve_embedding_config(settings, chat_config)


def test_create_chat_model_anthropic_passes_base_url():
    constructor = MagicMock(return_value=MagicMock())
    mock_anthropic = MagicMock(ChatAnthropic=constructor)
    with _mock_modules(langchain_anthropic=mock_anthropic):
        config = ResolvedProviderConfig(
            provider="anthropic",
            model="claude",
            api_key="key",
            api_base="https://anthropic.example",
        )
        create_chat_model(config)

    constructor.assert_called_once()
    assert constructor.call_args.kwargs["base_url"] == "https://anthropic.example"


def test_create_chat_model_litellm_passes_api_key_and_base():
    constructor = MagicMock(return_value=MagicMock())
    mock_litellm_pkg = MagicMock(ChatLiteLLM=constructor)
    with _mock_modules(langchain_litellm=mock_litellm_pkg):
        config = ResolvedProviderConfig(
            provider="litellm",
            model="openai/gpt-4.1-mini",
            api_key="litellm-key",
            api_base="https://litellm.example",
        )
        create_chat_model(config)

    constructor.assert_called_once()
    assert constructor.call_args.kwargs["api_key"] == "litellm-key"
    assert constructor.call_args.kwargs["api_base"] == "https://litellm.example"


def test_create_embeddings_openai_compatible_passes_base_url():
    constructor = MagicMock(return_value=MagicMock())
    mock_openai = MagicMock(OpenAIEmbeddings=constructor)
    with _mock_modules(langchain_openai=mock_openai):
        config = ResolvedProviderConfig(
            provider="openai-compatible",
            model="text-embedding-3-small",
            api_key="key",
            api_base="https://openai.example",
        )
        create_embeddings(config)

    constructor.assert_called_once()
    assert constructor.call_args.kwargs["base_url"] == "https://openai.example"
