from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple

from dotenv import load_dotenv

from agentic_rag.errors import ConfigurationError

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

SUPPORTED_INGESTION_MODES: Tuple[str, ...] = ("auto", "explicit")
SUPPORTED_RETRIEVAL_MODES: Tuple[str, ...] = ("dense", "hybrid")

IngestionMode = Literal["auto", "explicit"]
RetrievalMode = Literal["dense", "hybrid"]


def _clean_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def _read_float_env(*names: str, default: float) -> float:
    raw_value = next((os.getenv(name) for name in names if os.getenv(name) is not None), None)
    normalized = _clean_optional_str(raw_value)
    if normalized is None:
        return default

    try:
        return float(normalized)
    except ValueError as exc:
        joined_names = ", ".join(names)
        raise ConfigurationError(f"{joined_names} must be a valid float value.") from exc


def _read_int_env(*names: str, default: int) -> int:
    raw_value = next((os.getenv(name) for name in names if os.getenv(name) is not None), None)
    normalized = _clean_optional_str(raw_value)
    if normalized is None:
        return default

    try:
        return int(normalized)
    except ValueError as exc:
        joined_names = ", ".join(names)
        raise ConfigurationError(f"{joined_names} must be a valid integer value.") from exc


def normalize_provider_name(provider: Optional[str]) -> Optional[str]:
    """Normalize provider names and aliases."""
    if provider is None:
        return None

    value = provider.strip().lower().replace("_", "-")
    return PROVIDER_ALIASES.get(value, value)


def default_chat_provider() -> str:
    """Read the configured chat provider from the environment."""
    provider = _clean_optional_str(os.getenv("CHAT_PROVIDER", os.getenv("LLM_PROVIDER")))
    return normalize_provider_name(provider) or "google"


def default_chat_model() -> str:
    """Read the configured chat model from the environment."""
    provider = default_chat_provider()
    return (
        _clean_optional_str(
            os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", DEFAULT_CHAT_MODELS.get(provider, "")))
        )
        or ""
    )


def default_embedding_provider() -> Optional[str]:
    """Read the configured embedding provider from the environment."""
    explicit_provider = _clean_optional_str(os.getenv("EMBEDDING_PROVIDER"))
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

    return _clean_optional_str(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS.get(provider)))


def default_ingestion_mode() -> str:
    """Read the configured ingestion mode from the environment."""
    return (_clean_optional_str(os.getenv("INGESTION_MODE")) or "auto").lower()


def default_retrieval_mode() -> str:
    """Read the configured retrieval mode from the environment."""
    return (_clean_optional_str(os.getenv("RETRIEVAL_MODE")) or "dense").lower()


@dataclass(frozen=True)
class AgenticRagSettings:
    source_urls: Sequence[str] = field(default_factory=lambda: DEFAULT_SOURCE_URLS)
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 6
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
        default_factory=lambda: _read_float_env("CHAT_TEMPERATURE", "LLM_TEMPERATURE", default=0.0)
    )
    embedding_provider: Optional[str] = field(default_factory=default_embedding_provider)
    embedding_model: Optional[str] = field(default_factory=default_embedding_model)
    embedding_api_key: Optional[str] = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY"))
    embedding_api_key_env: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY_ENV")
    )
    embedding_api_base: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_BASE")
    )
    index_cache_dir: str = field(
        default_factory=lambda: os.getenv("INDEX_CACHE_DIR", ".cache/vectorstores")
    )
    ingestion_mode: str = field(default_factory=default_ingestion_mode)
    retrieval_mode: str = field(default_factory=default_retrieval_mode)
    max_rewrites: int = field(default_factory=lambda: _read_int_env("MAX_REWRITES", default=3))
    fetch_timeout_seconds: float = field(
        default_factory=lambda: _read_float_env("FETCH_TIMEOUT_SECONDS", default=20.0)
    )
    model_timeout_seconds: float = field(
        default_factory=lambda: _read_float_env("MODEL_TIMEOUT_SECONDS", default=60.0)
    )

    def __post_init__(self) -> None:
        normalized_urls = tuple(url.strip() for url in self.source_urls if url and url.strip())
        if not normalized_urls:
            raise ConfigurationError("source_urls must contain at least one non-empty URL.")

        chat_provider = normalize_provider_name(_clean_optional_str(self.chat_provider))
        if chat_provider not in SUPPORTED_CHAT_PROVIDERS:
            raise ConfigurationError(
                "Unsupported chat provider "
                f"'{self.chat_provider}'. Supported values: {', '.join(SUPPORTED_CHAT_PROVIDERS)}."
            )

        embedding_provider = normalize_provider_name(_clean_optional_str(self.embedding_provider))
        if embedding_provider is None:
            raise ConfigurationError(
                "embedding_provider must be set. Providers without default embeddings must "
                "configure EMBEDDING_PROVIDER and EMBEDDING_MODEL explicitly."
            )
        if embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ConfigurationError(
                "Unsupported embedding provider "
                f"'{self.embedding_provider}'. Supported values: "
                f"{', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}."
            )

        chat_model = _clean_optional_str(self.chat_model) or DEFAULT_CHAT_MODELS.get(chat_provider)
        if not chat_model:
            raise ConfigurationError("chat_model must be a non-empty string.")

        embedding_model = _clean_optional_str(self.embedding_model) or DEFAULT_EMBEDDING_MODELS.get(
            embedding_provider
        )
        if not embedding_model:
            raise ConfigurationError("embedding_model must be a non-empty string.")

        ingestion_mode = (_clean_optional_str(self.ingestion_mode) or "auto").lower()
        if ingestion_mode not in SUPPORTED_INGESTION_MODES:
            raise ConfigurationError(
                f"Unsupported ingestion_mode '{self.ingestion_mode}'. Supported values: "
                f"{', '.join(SUPPORTED_INGESTION_MODES)}."
            )

        retrieval_mode = (_clean_optional_str(self.retrieval_mode) or "dense").lower()
        if retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
            raise ConfigurationError(
                f"Unsupported retrieval_mode '{self.retrieval_mode}'. Supported values: "
                f"{', '.join(SUPPORTED_RETRIEVAL_MODES)}."
            )

        chat_api_base = _clean_optional_str(self.chat_api_base)
        embedding_api_base = _clean_optional_str(self.embedding_api_base)
        effective_embedding_api_base = embedding_api_base or (
            chat_api_base if embedding_provider == chat_provider else None
        )

        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be greater than zero.")
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be zero or greater.")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be smaller than chunk_size.")
        if self.retrieval_k <= 0:
            raise ConfigurationError("retrieval_k must be greater than zero.")
        if not 0 <= self.chat_temperature <= 2:
            raise ConfigurationError("chat_temperature must be between 0 and 2.")
        if self.max_rewrites <= 0:
            raise ConfigurationError("max_rewrites must be greater than zero.")
        if self.fetch_timeout_seconds <= 0:
            raise ConfigurationError("fetch_timeout_seconds must be greater than zero.")
        if self.model_timeout_seconds <= 0:
            raise ConfigurationError("model_timeout_seconds must be greater than zero.")
        if chat_provider == "openai-compatible" and not chat_api_base:
            raise ConfigurationError(
                "The 'openai-compatible' chat provider requires CHAT_API_BASE or --chat-api-base."
            )
        if embedding_provider == "openai-compatible" and not effective_embedding_api_base:
            raise ConfigurationError(
                "The 'openai-compatible' embedding provider requires EMBEDDING_API_BASE "
                "or a compatible CHAT_API_BASE."
            )

        object.__setattr__(self, "source_urls", normalized_urls)
        object.__setattr__(self, "chat_provider", chat_provider)
        object.__setattr__(self, "chat_model", chat_model)
        object.__setattr__(self, "chat_api_key", _clean_optional_str(self.chat_api_key))
        object.__setattr__(self, "chat_api_key_env", _clean_optional_str(self.chat_api_key_env))
        object.__setattr__(self, "chat_api_base", chat_api_base)
        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "embedding_model", embedding_model)
        object.__setattr__(self, "embedding_api_key", _clean_optional_str(self.embedding_api_key))
        object.__setattr__(
            self, "embedding_api_key_env", _clean_optional_str(self.embedding_api_key_env)
        )
        object.__setattr__(self, "embedding_api_base", embedding_api_base)
        object.__setattr__(
            self,
            "index_cache_dir",
            _clean_optional_str(self.index_cache_dir) or ".cache/vectorstores",
        )
        object.__setattr__(self, "ingestion_mode", ingestion_mode)
        object.__setattr__(self, "retrieval_mode", retrieval_mode)
