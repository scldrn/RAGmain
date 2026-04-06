from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from langchain_core.embeddings import Embeddings

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


class LiteLLMEmbeddingsAdapter(Embeddings):
    """LangChain embeddings adapter backed by LiteLLM's embedding endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    def embed_documents(self, texts):
        import litellm

        response = litellm.embedding(
            model=self.model,
            input=list(texts),
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return [self._embedding_from_item(item) for item in response.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    @staticmethod
    def _embedding_from_item(item):
        if isinstance(item, dict):
            return item["embedding"]
        return item.embedding


def _supported_providers_text(values: Tuple[str, ...]) -> str:
    return ", ".join(values)


def _normalize_supported_provider(provider: Optional[str], *, kind: str) -> Optional[str]:
    normalized = normalize_provider_name(provider)
    supported = (
        SUPPORTED_CHAT_PROVIDERS if kind == "chat" else SUPPORTED_EMBEDDING_PROVIDERS
    )

    if normalized is None:
        return None

    if normalized not in supported:
        raise RuntimeError(
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
    assert provider is not None

    api_key = _resolve_api_key(
        provider,
        explicit_value=settings.chat_api_key,
        explicit_env_name=settings.chat_api_key_env,
    )

    if provider == "openai-compatible" and not settings.chat_api_base:
        raise RuntimeError(
            "The 'openai-compatible' chat provider requires CHAT_API_BASE or --chat-api-base."
        )

    if provider != "litellm" and not api_key:
        raise RuntimeError(
            f"Missing credentials for chat provider '{provider}'. "
            f"Set {settings.chat_api_key_env or _credential_hint(provider)}."
        )

    return ResolvedProviderConfig(
        provider=provider,
        model=settings.chat_model or DEFAULT_CHAT_MODELS[provider],
        api_key=api_key,
        api_base=settings.chat_api_base,
        temperature=settings.chat_temperature,
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
        raise RuntimeError(
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
        raise RuntimeError(
            "The 'openai-compatible' embedding provider requires "
            "EMBEDDING_API_BASE or --embedding-api-base."
        )

    if provider != "litellm" and not api_key:
        raise RuntimeError(
            f"Missing credentials for embedding provider '{provider}'. "
            f"Set {settings.embedding_api_key_env or _credential_hint(provider)}."
        )

    return ResolvedProviderConfig(
        provider=provider,
        model=settings.embedding_model or DEFAULT_EMBEDDING_MODELS[provider],
        api_key=api_key,
        api_base=api_base,
    )


def create_chat_model(config: ResolvedProviderConfig):
    """Create a chat model for the configured provider."""
    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if config.provider in {"openai", "openai-compatible"}:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["base_url"] = config.api_base
        return ChatOpenAI(**kwargs)

    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model_name": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["base_url"] = config.api_base
        return ChatAnthropic(**kwargs)

    if config.provider == "litellm":
        from langchain_litellm import ChatLiteLLM

        kwargs = {
            "model": config.model,
            "temperature": config.temperature or 0,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["api_base"] = config.api_base
        return ChatLiteLLM(**kwargs)

    raise RuntimeError(f"Unsupported chat provider '{config.provider}'.")


def create_embeddings(config: ResolvedProviderConfig):
    """Create embeddings for the configured provider."""
    if config.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        kwargs = {"model": config.model}
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return GoogleGenerativeAIEmbeddings(**kwargs)

    if config.provider in {"openai", "openai-compatible"}:
        from langchain_openai import OpenAIEmbeddings

        kwargs = {"model": config.model}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["base_url"] = config.api_base
        return OpenAIEmbeddings(**kwargs)

    if config.provider == "litellm":
        return LiteLLMEmbeddingsAdapter(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
        )

    raise RuntimeError(f"Unsupported embedding provider '{config.provider}'.")
