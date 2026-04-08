from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from agentic_rag.errors import ConfigurationError
from agentic_rag.settings import (
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBEDDING_MODELS,
    SUPPORTED_CHAT_PROVIDERS,
    SUPPORTED_EMBEDDING_PROVIDERS,
    AgenticRagSettings,
    normalize_provider_name,
)

PROVIDER_ENV_FALLBACKS = {
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai-compatible": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "litellm": (),
}


@dataclass(frozen=True)
class ResolvedProviderConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = None
    timeout_seconds: Optional[float] = None
    max_retries: Optional[int] = None


class LiteLLMEmbeddingsAdapter(Embeddings):
    """LangChain embeddings adapter backed by LiteLLM's embedding endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout_seconds = timeout_seconds

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import litellm

        response = litellm.embedding(
            model=self.model,
            input=list(texts),
            api_key=self.api_key,
            api_base=self.api_base,
            timeout=self.timeout_seconds or 600,
        )
        return [self._embedding_from_item(item) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @staticmethod
    def _embedding_from_item(item: Any) -> list[float]:
        embedding = item["embedding"] if isinstance(item, dict) else item.embedding
        return list(embedding)


def _supported_providers_text(values: Tuple[str, ...]) -> str:
    return ", ".join(values)


def _normalize_supported_provider(provider: Optional[str], *, kind: str) -> Optional[str]:
    normalized = normalize_provider_name(provider)
    supported = SUPPORTED_CHAT_PROVIDERS if kind == "chat" else SUPPORTED_EMBEDDING_PROVIDERS

    if normalized is None:
        return None

    if normalized not in supported:
        raise ConfigurationError(
            f"Unsupported {kind} provider '{provider}'. Supported values: "
            f"{_supported_providers_text(supported)}."
        )

    return normalized


def _resolve_api_key(
    provider: str,
    *,
    explicit_value: Optional[str],
    explicit_env_name: Optional[str],
) -> Optional[str]:
    if explicit_value:
        return explicit_value

    if explicit_env_name and os.getenv(explicit_env_name):
        return os.getenv(explicit_env_name)

    for env_name in PROVIDER_ENV_FALLBACKS.get(provider, ()):
        if os.getenv(env_name):
            return os.getenv(env_name)

    return None


def _credential_hint(provider: str) -> str:
    env_names = PROVIDER_ENV_FALLBACKS.get(provider, ())
    if provider == "litellm":
        return (
            "Set CHAT_API_KEY/EMBEDDING_API_KEY, CHAT_API_KEY_ENV/EMBEDDING_API_KEY_ENV, "
            "or use whatever provider-specific env vars your LiteLLM model expects."
        )

    return " or ".join(env_names)


def resolve_chat_config(settings: AgenticRagSettings) -> ResolvedProviderConfig:
    """Resolve chat provider settings into an executable model config."""
    provider = _normalize_supported_provider(settings.chat_provider, kind="chat")
    if provider is None:
        raise ConfigurationError(
            f"Could not resolve chat provider from '{settings.chat_provider}'."
        )

    api_key = _resolve_api_key(
        provider,
        explicit_value=settings.chat_api_key,
        explicit_env_name=settings.chat_api_key_env,
    )

    if provider == "openai-compatible" and not settings.chat_api_base:
        raise ConfigurationError(
            "The 'openai-compatible' chat provider requires CHAT_API_BASE or --chat-api-base."
        )

    if provider != "litellm" and not api_key:
        raise ConfigurationError(
            f"Missing credentials for chat provider '{provider}'. "
            f"Set {settings.chat_api_key_env or _credential_hint(provider)}."
        )

    return ResolvedProviderConfig(
        provider=provider,
        model=settings.chat_model or DEFAULT_CHAT_MODELS[provider],
        api_key=api_key,
        api_base=settings.chat_api_base,
        temperature=settings.chat_temperature,
        timeout_seconds=getattr(settings, "model_timeout_seconds", None),
        max_retries=0,
    )


def resolve_embedding_config(
    settings: AgenticRagSettings,
    chat_config: ResolvedProviderConfig,
) -> ResolvedProviderConfig:
    """Resolve embedding provider settings into an executable embeddings config."""
    provider = _normalize_supported_provider(
        settings.embedding_provider,
        kind="embedding",
    )

    if provider is None:
        raise ConfigurationError(
            f"Chat provider '{chat_config.provider}' does not define embeddings automatically. "
            "Set EMBEDDING_PROVIDER and EMBEDDING_MODEL explicitly."
        )

    api_key = _resolve_api_key(
        provider,
        explicit_value=settings.embedding_api_key,
        explicit_env_name=settings.embedding_api_key_env,
    )
    api_base = settings.embedding_api_base

    if provider == chat_config.provider:
        api_key = api_key or chat_config.api_key
        api_base = api_base or chat_config.api_base

    if provider == "openai-compatible" and not api_base:
        raise ConfigurationError(
            "The 'openai-compatible' embedding provider requires "
            "EMBEDDING_API_BASE or --embedding-api-base."
        )

    if provider != "litellm" and not api_key:
        raise ConfigurationError(
            f"Missing credentials for embedding provider '{provider}'. "
            f"Set {settings.embedding_api_key_env or _credential_hint(provider)}."
        )

    return ResolvedProviderConfig(
        provider=provider,
        model=settings.embedding_model or DEFAULT_EMBEDDING_MODELS[provider],
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=getattr(settings, "model_timeout_seconds", None),
        max_retries=0,
    )


def create_chat_model(config: ResolvedProviderConfig) -> BaseChatModel:
    """Create a chat model for the configured provider."""
    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        google_kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            google_kwargs["api_key"] = config.api_key
        if config.timeout_seconds is not None:
            google_kwargs["timeout"] = config.timeout_seconds
        if config.max_retries is not None:
            google_kwargs["max_retries"] = config.max_retries
        return ChatGoogleGenerativeAI(**google_kwargs)

    if config.provider in {"openai", "openai-compatible"}:
        from langchain_openai import ChatOpenAI

        openai_kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            openai_kwargs["api_key"] = config.api_key
        if config.api_base:
            openai_kwargs["base_url"] = config.api_base
        if config.timeout_seconds is not None:
            openai_kwargs["request_timeout"] = config.timeout_seconds
        if config.max_retries is not None:
            openai_kwargs["max_retries"] = config.max_retries
        return ChatOpenAI(**openai_kwargs)

    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs: dict[str, Any] = {
            "model_name": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            anthropic_kwargs["api_key"] = config.api_key
        if config.api_base:
            anthropic_kwargs["base_url"] = config.api_base
        if config.timeout_seconds is not None:
            anthropic_kwargs["default_request_timeout"] = config.timeout_seconds
        if config.max_retries is not None:
            anthropic_kwargs["max_retries"] = config.max_retries
        return ChatAnthropic(**anthropic_kwargs)

    if config.provider == "litellm":
        from langchain_litellm import ChatLiteLLM

        litellm_kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            litellm_kwargs["api_key"] = config.api_key
        if config.api_base:
            litellm_kwargs["api_base"] = config.api_base
        if config.timeout_seconds is not None:
            litellm_kwargs["request_timeout"] = config.timeout_seconds
        if config.max_retries is not None:
            litellm_kwargs["max_retries"] = config.max_retries
        return ChatLiteLLM(**litellm_kwargs)

    raise ConfigurationError(f"Unsupported chat provider '{config.provider}'.")


def create_embeddings(config: ResolvedProviderConfig) -> Embeddings:
    """Create embeddings for the configured provider."""
    if config.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        google_kwargs: dict[str, Any] = {"model": config.model}
        if config.api_key:
            google_kwargs["google_api_key"] = config.api_key
        if config.timeout_seconds is not None:
            google_kwargs["request_options"] = {"timeout": config.timeout_seconds}
        return GoogleGenerativeAIEmbeddings(**google_kwargs)

    if config.provider in {"openai", "openai-compatible"}:
        from langchain_openai import OpenAIEmbeddings

        openai_kwargs: dict[str, Any] = {"model": config.model}
        if config.api_key:
            openai_kwargs["api_key"] = config.api_key
        if config.api_base:
            openai_kwargs["base_url"] = config.api_base
        if config.timeout_seconds is not None:
            openai_kwargs["request_timeout"] = config.timeout_seconds
        if config.max_retries is not None:
            openai_kwargs["max_retries"] = config.max_retries
        return OpenAIEmbeddings(**openai_kwargs)

    if config.provider == "litellm":
        return LiteLLMEmbeddingsAdapter(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            timeout_seconds=config.timeout_seconds,
        )

    raise ConfigurationError(f"Unsupported embedding provider '{config.provider}'.")
