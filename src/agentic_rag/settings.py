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
DEFAULT_CHAT_PROVIDER = "google"
DEFAULT_INGESTION_MODE = "auto"
DEFAULT_RETRIEVAL_MODE = "dense"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 6
DEFAULT_CHAT_TEMPERATURE = 0.0
DEFAULT_CHAT_MAX_TOKENS: int | None = None
DEFAULT_INDEX_CACHE_DIR = ".cache/vectorstores"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_CORS_ALLOW_ORIGINS: Tuple[str, ...] = ()
DEFAULT_MAX_REWRITES = 3
DEFAULT_FETCH_TIMEOUT_SECONDS = 20.0
DEFAULT_MODEL_TIMEOUT_SECONDS = 60.0

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


def _read_optional_env(*names: str) -> Optional[str]:
    raw_value = next((os.getenv(name) for name in names if os.getenv(name) is not None), None)
    return _clean_optional_str(raw_value)


def _read_csv_env(*names: str) -> Tuple[str, ...]:
    raw_value = _read_optional_env(*names)
    if raw_value is None:
        return ()
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _read_float_env(*names: str, default: float) -> float:
    normalized = _read_optional_env(*names)
    if normalized is None:
        return default

    try:
        return float(normalized)
    except ValueError as exc:
        joined_names = ", ".join(names)
        raise ConfigurationError(f"{joined_names} must be a valid float value.") from exc


def _read_int_env(*names: str, default: int) -> int:
    normalized = _read_optional_env(*names)
    if normalized is None:
        return default

    try:
        return int(normalized)
    except ValueError as exc:
        joined_names = ", ".join(names)
        raise ConfigurationError(f"{joined_names} must be a valid integer value.") from exc


def _read_optional_int_env(*names: str) -> int | None:
    normalized = _read_optional_env(*names)
    if normalized is None:
        return None

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
    return normalize_provider_name(_read_optional_env("CHAT_PROVIDER", "LLM_PROVIDER")) or (
        DEFAULT_CHAT_PROVIDER
    )


def default_chat_model() -> str:
    """Read the configured chat model from the environment."""
    provider = default_chat_provider()
    return _read_optional_env("CHAT_MODEL", "LLM_MODEL") or DEFAULT_CHAT_MODELS.get(provider, "")


def default_embedding_provider() -> Optional[str]:
    """Read the configured embedding provider from the environment."""
    explicit_provider = _read_optional_env("EMBEDDING_PROVIDER")
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

    return _read_optional_env("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS.get(provider)


def default_ingestion_mode() -> str:
    """Read the configured ingestion mode from the environment."""
    return (_read_optional_env("INGESTION_MODE") or DEFAULT_INGESTION_MODE).lower()


def default_retrieval_mode() -> str:
    """Read the configured retrieval mode from the environment."""
    return (_read_optional_env("RETRIEVAL_MODE") or DEFAULT_RETRIEVAL_MODE).lower()


def _normalize_mode_setting(
    value: Optional[str],
    *,
    default: str,
    supported: Tuple[str, ...],
    field_name: str,
) -> str:
    normalized = (_clean_optional_str(value) or default).lower()
    if normalized not in supported:
        raise ConfigurationError(
            f"Unsupported {field_name} '{value}'. Supported values: {', '.join(supported)}."
        )
    return normalized


def _normalized_source_urls(source_urls: Sequence[str]) -> Tuple[str, ...]:
    return tuple(url.strip() for url in source_urls if url and url.strip())


def _normalized_values(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(value.strip() for value in values if value and value.strip())


def _resolve_model_name(
    value: Optional[str],
    *,
    provider: str,
    defaults: dict[str, str],
    field_name: str,
) -> str:
    resolved = _clean_optional_str(value) or defaults.get(provider)
    if not resolved:
        raise ConfigurationError(f"{field_name} must be a non-empty string.")
    return resolved


@dataclass(frozen=True)
class AgenticRagSettings:
    source_urls: Sequence[str] = field(default_factory=lambda: DEFAULT_SOURCE_URLS)
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    chat_provider: str = field(default_factory=default_chat_provider)
    chat_model: str = field(default_factory=default_chat_model)
    chat_api_key: Optional[str] = field(
        default_factory=lambda: _read_optional_env("CHAT_API_KEY", "LLM_API_KEY")
    )
    chat_api_key_env: Optional[str] = field(
        default_factory=lambda: _read_optional_env("CHAT_API_KEY_ENV", "LLM_API_KEY_ENV")
    )
    chat_api_base: Optional[str] = field(
        default_factory=lambda: _read_optional_env("CHAT_API_BASE", "LLM_API_BASE")
    )
    chat_temperature: float = field(
        default_factory=lambda: _read_float_env(
            "CHAT_TEMPERATURE",
            "LLM_TEMPERATURE",
            default=DEFAULT_CHAT_TEMPERATURE,
        )
    )
    chat_max_tokens: int | None = field(
        default_factory=lambda: _read_optional_int_env("CHAT_MAX_TOKENS")
    )
    embedding_provider: Optional[str] = field(default_factory=default_embedding_provider)
    embedding_model: Optional[str] = field(default_factory=default_embedding_model)
    embedding_api_key: Optional[str] = field(
        default_factory=lambda: _read_optional_env("EMBEDDING_API_KEY")
    )
    embedding_api_key_env: Optional[str] = field(
        default_factory=lambda: _read_optional_env("EMBEDDING_API_KEY_ENV")
    )
    embedding_api_base: Optional[str] = field(
        default_factory=lambda: _read_optional_env("EMBEDDING_API_BASE")
    )
    index_cache_dir: str = field(
        default_factory=lambda: _read_optional_env("INDEX_CACHE_DIR") or DEFAULT_INDEX_CACHE_DIR
    )
    collection_name: str = field(
        default_factory=lambda: _read_optional_env("COLLECTION_NAME") or DEFAULT_COLLECTION_NAME
    )
    cors_allow_origins: Sequence[str] = field(
        default_factory=lambda: _read_csv_env("CORS_ALLOW_ORIGINS", "API_CORS_ALLOW_ORIGINS")
    )
    ingestion_mode: str = field(default_factory=default_ingestion_mode)
    retrieval_mode: str = field(default_factory=default_retrieval_mode)
    max_rewrites: int = field(
        default_factory=lambda: _read_int_env("MAX_REWRITES", default=DEFAULT_MAX_REWRITES)
    )
    fetch_timeout_seconds: float = field(
        default_factory=lambda: _read_float_env(
            "FETCH_TIMEOUT_SECONDS",
            default=DEFAULT_FETCH_TIMEOUT_SECONDS,
        )
    )
    model_timeout_seconds: float = field(
        default_factory=lambda: _read_float_env(
            "MODEL_TIMEOUT_SECONDS",
            default=DEFAULT_MODEL_TIMEOUT_SECONDS,
        )
    )

    def __post_init__(self) -> None:
        normalized_urls = _normalized_source_urls(self.source_urls)
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

        chat_model = _resolve_model_name(
            self.chat_model,
            provider=chat_provider,
            defaults=DEFAULT_CHAT_MODELS,
            field_name="chat_model",
        )
        embedding_model = _resolve_model_name(
            self.embedding_model,
            provider=embedding_provider,
            defaults=DEFAULT_EMBEDDING_MODELS,
            field_name="embedding_model",
        )

        ingestion_mode = _normalize_mode_setting(
            self.ingestion_mode,
            default=DEFAULT_INGESTION_MODE,
            supported=SUPPORTED_INGESTION_MODES,
            field_name="ingestion_mode",
        )
        retrieval_mode = _normalize_mode_setting(
            self.retrieval_mode,
            default=DEFAULT_RETRIEVAL_MODE,
            supported=SUPPORTED_RETRIEVAL_MODES,
            field_name="retrieval_mode",
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
        if self.chat_max_tokens is not None and self.chat_max_tokens <= 0:
            raise ConfigurationError("chat_max_tokens must be greater than zero.")
        if self.max_rewrites <= 0:
            raise ConfigurationError("max_rewrites must be greater than zero.")
        if self.fetch_timeout_seconds <= 0:
            raise ConfigurationError("fetch_timeout_seconds must be greater than zero.")
        if self.model_timeout_seconds <= 0:
            raise ConfigurationError("model_timeout_seconds must be greater than zero.")
        collection_name = _clean_optional_str(self.collection_name)
        if not collection_name:
            raise ConfigurationError("collection_name must be a non-empty string.")
        cors_allow_origins = _normalized_values(self.cors_allow_origins)
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
        object.__setattr__(self, "chat_max_tokens", self.chat_max_tokens)
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
            _clean_optional_str(self.index_cache_dir) or DEFAULT_INDEX_CACHE_DIR,
        )
        object.__setattr__(self, "collection_name", collection_name)
        object.__setattr__(self, "cors_allow_origins", cors_allow_origins)
        object.__setattr__(self, "ingestion_mode", ingestion_mode)
        object.__setattr__(self, "retrieval_mode", retrieval_mode)
