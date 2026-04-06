from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SOURCE_URLS: Tuple[str, ...] = (
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
)

SUPPORTED_CHAT_PROVIDERS: Tuple[str, ...] = (
    "google",
    "openai",
    "openai-compatible",
    "anthropic",
    "litellm",
)

SUPPORTED_EMBEDDING_PROVIDERS: Tuple[str, ...] = (
    "google",
    "openai",
    "openai-compatible",
    "litellm",
)

DEFAULT_CHAT_MODELS = {
    "google": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "openai-compatible": "gpt-4.1-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "litellm": "openai/gpt-4.1-mini",
}

DEFAULT_EMBEDDING_MODELS = {
    "google": "gemini-embedding-2-preview",
    "openai": "text-embedding-3-small",
    "openai-compatible": "text-embedding-3-small",
    "litellm": "openai/text-embedding-3-small",
}

PROVIDER_ALIASES = {
    "gemini": "google",
    "google-ai-studio": "google",
    "google-genai": "google",
    "openai_compatible": "openai-compatible",
    "compat": "openai-compatible",
}


def normalize_provider_name(provider: Optional[str]) -> Optional[str]:
    """Normalize provider names and aliases."""
    if provider is None:
        return None

    value = provider.strip().lower().replace("_", "-")
    return PROVIDER_ALIASES.get(value, value)


def default_chat_provider() -> str:
    """Read the configured chat provider from the environment."""
    provider = os.getenv("CHAT_PROVIDER", os.getenv("LLM_PROVIDER", "google"))
    return normalize_provider_name(provider) or "google"


def default_chat_model() -> str:
    """Read the configured chat model from the environment."""
    provider = default_chat_provider()
    return os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", DEFAULT_CHAT_MODELS[provider]))


def default_embedding_provider() -> Optional[str]:
    """Read the configured embedding provider from the environment."""
    explicit_provider = os.getenv("EMBEDDING_PROVIDER")
    if explicit_provider:
        return normalize_provider_name(explicit_provider)

    provider = default_chat_provider()
    if provider in SUPPORTED_EMBEDDING_PROVIDERS:
        return provider

    return None


def default_embedding_model() -> Optional[str]:
    """Read the configured embedding model from the environment."""
    provider = default_embedding_provider()
    if provider is None:
        return None

    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS[provider])


@dataclass(frozen=True)
class AgenticRagSettings:
    source_urls: Sequence[str] = field(default_factory=lambda: DEFAULT_SOURCE_URLS)
    chunk_size: int = 100
    chunk_overlap: int = 50
    retrieval_k: int = 4
    chat_provider: str = field(default_factory=default_chat_provider)
    chat_model: str = field(default_factory=default_chat_model)
    chat_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("CHAT_API_KEY", os.getenv("LLM_API_KEY"))
    )
    chat_api_key_env: Optional[str] = field(
        default_factory=lambda: os.getenv("CHAT_API_KEY_ENV", os.getenv("LLM_API_KEY_ENV"))
    )
    chat_api_base: Optional[str] = field(
        default_factory=lambda: os.getenv("CHAT_API_BASE", os.getenv("LLM_API_BASE"))
    )
    chat_temperature: float = field(
        default_factory=lambda: float(
            os.getenv("CHAT_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0"))
        )
    )
    embedding_provider: Optional[str] = field(default_factory=default_embedding_provider)
    embedding_model: Optional[str] = field(default_factory=default_embedding_model)
    embedding_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY")
    )
    embedding_api_key_env: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY_ENV")
    )
    embedding_api_base: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_BASE")
    )
