from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from collections.abc import Sequence
from typing import cast

from agentic_rag.errors import AgenticRagError, ConfigurationError
from agentic_rag.presentation import format_index_status, format_query_step
from agentic_rag.service import AgenticRagService, HealthStatus, IndexStatus, QueryStep
from agentic_rag.settings import (
    DEFAULT_CHAT_MODELS,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_INDEX_CACHE_DIR,
    DEFAULT_MAX_REWRITES,
    DEFAULT_MODEL_TIMEOUT_SECONDS,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SOURCE_URLS,
    SUPPORTED_CHAT_PROVIDERS,
    SUPPORTED_EMBEDDING_PROVIDERS,
    AgenticRagSettings,
    _read_float_env,
    _read_int_env,
    _read_optional_env,
    _read_optional_int_env,
    default_chat_provider,
    default_ingestion_mode,
    default_retrieval_mode,
    normalize_provider_name,
)

CLI_COMMANDS = ("query", "ingest")
logger = logging.getLogger(__name__)


def configure_runtime_warnings() -> None:
    """Suppress noisy dependency warnings that are irrelevant during normal CLI usage."""
    if os.getenv("AGENTIC_RAG_DEBUG"):
        return

    warning_patterns = (
        r".*non-supported Python version.*",
        r".*Python version 3\.9 past its end of life.*",
    )
    for pattern in warning_patterns:
        warnings.filterwarnings(
            "ignore",
            message=pattern,
            category=FutureWarning,
        )


def add_shared_settings_arguments(parser: argparse.ArgumentParser) -> None:
    """Add provider and indexing settings shared across CLI commands."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable startup diagnostics and verbose runtime logging.",
    )
    parser.add_argument(
        "--url",
        dest="source_urls",
        action="append",
        help="Source URL to index. Repeat this flag to add multiple URLs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size used during document splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap used during document splitting.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=None,
        help="Number of chunks returned by the retriever.",
    )
    parser.add_argument(
        "--chat-provider",
        default=None,
        help=(f"Chat provider to use. Supported values: {', '.join(SUPPORTED_CHAT_PROVIDERS)}."),
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Chat model to use for the selected provider.",
    )
    parser.add_argument(
        "--chat-api-key",
        default=None,
        help="Optional explicit API key for the chat provider.",
    )
    parser.add_argument(
        "--chat-api-key-env",
        default=None,
        help="Name of the environment variable that stores the chat API key.",
    )
    parser.add_argument(
        "--chat-api-base",
        default=None,
        help="Optional base URL for chat providers that support custom endpoints.",
    )
    parser.add_argument(
        "--chat-temperature",
        type=float,
        default=None,
        help="Sampling temperature for the chat model.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help=(
            "Embedding provider to use. Supported values: "
            f"{', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model to use for the selected embedding provider.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        help="Optional explicit API key for the embedding provider.",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default=None,
        help="Name of the environment variable that stores the embedding API key.",
    )
    parser.add_argument(
        "--embedding-api-base",
        default=None,
        help="Optional base URL for embedding providers that support custom endpoints.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with `query` and `ingest` commands."""
    parser = argparse.ArgumentParser(
        description="Run the LangGraph agentic RAG example as a local Python project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Run a question through the RAG graph.")
    add_shared_settings_arguments(query_parser)
    query_parser.add_argument(
        "--question",
        required=True,
        help="Question to send through the agentic RAG graph.",
    )
    query_parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Print node-by-node updates while the graph runs.",
    )
    query_parser.add_argument(
        "--diagram",
        help="Optional output file for the Mermaid graph diagram.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Build or refresh the local vector index.",
    )
    add_shared_settings_arguments(ingest_parser)

    return parser


def _default_embedding_provider(chat_provider: str) -> str | None:
    explicit_provider = normalize_provider_name(_read_optional_env("EMBEDDING_PROVIDER"))
    if explicit_provider:
        return explicit_provider
    if chat_provider in SUPPORTED_EMBEDDING_PROVIDERS:
        return chat_provider
    return None


def build_settings(args: argparse.Namespace) -> AgenticRagSettings:
    """Translate parsed CLI arguments into validated settings."""
    source_urls = cast(list[str] | None, getattr(args, "source_urls", None))
    chunk_size = cast(int | None, getattr(args, "chunk_size", None))
    chunk_overlap = cast(int | None, getattr(args, "chunk_overlap", None))
    retrieval_k = cast(int | None, getattr(args, "retrieval_k", None))
    chat_provider_override = cast(str | None, getattr(args, "chat_provider", None))
    chat_provider = normalize_provider_name(chat_provider_override or default_chat_provider()) or ""
    chat_model = cast(str | None, getattr(args, "chat_model", None))
    chat_api_key = cast(str | None, getattr(args, "chat_api_key", None))
    chat_api_key_env = cast(str | None, getattr(args, "chat_api_key_env", None))
    chat_api_base = cast(str | None, getattr(args, "chat_api_base", None))
    chat_temperature = cast(float | None, getattr(args, "chat_temperature", None))
    embedding_provider_override = cast(str | None, getattr(args, "embedding_provider", None))
    embedding_provider = normalize_provider_name(
        embedding_provider_override or _default_embedding_provider(chat_provider)
    )
    embedding_model = cast(str | None, getattr(args, "embedding_model", None))
    embedding_api_key = cast(str | None, getattr(args, "embedding_api_key", None))
    embedding_api_key_env = cast(str | None, getattr(args, "embedding_api_key_env", None))
    embedding_api_base = cast(str | None, getattr(args, "embedding_api_base", None))

    return AgenticRagSettings(
        source_urls=tuple(source_urls or DEFAULT_SOURCE_URLS),
        chunk_size=chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP,
        retrieval_k=retrieval_k if retrieval_k is not None else DEFAULT_RETRIEVAL_K,
        chat_provider=chat_provider,
        chat_model=chat_model
        or _read_optional_env("CHAT_MODEL", "LLM_MODEL")
        or DEFAULT_CHAT_MODELS.get(chat_provider, ""),
        chat_api_key=chat_api_key or _read_optional_env("CHAT_API_KEY", "LLM_API_KEY"),
        chat_api_key_env=chat_api_key_env
        or _read_optional_env("CHAT_API_KEY_ENV", "LLM_API_KEY_ENV"),
        chat_api_base=chat_api_base or _read_optional_env("CHAT_API_BASE", "LLM_API_BASE"),
        chat_temperature=chat_temperature
        if chat_temperature is not None
        else _read_float_env(
            "CHAT_TEMPERATURE",
            "LLM_TEMPERATURE",
            default=DEFAULT_CHAT_TEMPERATURE,
        ),
        chat_max_tokens=_read_optional_int_env("CHAT_MAX_TOKENS"),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
        or _read_optional_env("EMBEDDING_MODEL")
        or (DEFAULT_EMBEDDING_MODELS.get(embedding_provider) if embedding_provider else None),
        embedding_api_key=embedding_api_key or _read_optional_env("EMBEDDING_API_KEY"),
        embedding_api_key_env=embedding_api_key_env or _read_optional_env("EMBEDDING_API_KEY_ENV"),
        embedding_api_base=embedding_api_base or _read_optional_env("EMBEDDING_API_BASE"),
        index_cache_dir=_read_optional_env("INDEX_CACHE_DIR") or DEFAULT_INDEX_CACHE_DIR,
        ingestion_mode=default_ingestion_mode(),
        retrieval_mode=default_retrieval_mode(),
        max_rewrites=_read_int_env("MAX_REWRITES", default=DEFAULT_MAX_REWRITES),
        fetch_timeout_seconds=_read_float_env(
            "FETCH_TIMEOUT_SECONDS",
            default=DEFAULT_FETCH_TIMEOUT_SECONDS,
        ),
        model_timeout_seconds=_read_float_env(
            "MODEL_TIMEOUT_SECONDS",
            default=DEFAULT_MODEL_TIMEOUT_SECONDS,
        ),
    )


def normalize_cli_argv(argv: Sequence[str]) -> list[str]:
    """Preserve the legacy `--question ...` flow by treating it as `query`."""
    if not argv:
        return ["query"]

    first_arg = argv[0]
    if first_arg in CLI_COMMANDS or first_arg in {"-h", "--help"}:
        return list(argv)

    return ["query", *argv]


def configure_logging(*, verbose: bool) -> None:
    """Configure CLI logging without forcing handlers across the repo."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("agentic_rag").setLevel(logging.DEBUG if verbose else logging.WARNING)


def log_startup_diagnostics(
    service: AgenticRagService,
    *,
    command: str,
    health: HealthStatus | None = None,
    index_status: IndexStatus | None = None,
) -> None:
    """Log the startup configuration and current index state."""
    current_health = health or service.health()
    logger.info(
        "startup command=%s chat=%s/%s embedding=%s/%s ingestion_mode=%s retrieval_mode=%s",
        command,
        current_health.chat_provider,
        current_health.chat_model,
        current_health.embedding_provider,
        current_health.embedding_model,
        current_health.ingestion_mode,
        current_health.retrieval_mode,
    )
    current_index_status = index_status
    if current_index_status is None:
        try:
            current_index_status = service.index_status()
        except Exception as exc:
            logger.warning(
                "startup_index_status_unavailable; continuing without index diagnostics",
                exc_info=exc,
            )
            return

    logger.info(
        "startup_index cache=%s index_exists=%s document_count=%s",
        current_index_status.cache_path,
        current_index_status.exists,
        current_index_status.document_count,
    )


def print_step(step: QueryStep) -> None:
    """Render a single streamed query step to stdout."""
    print()
    print(format_query_step(step))


def print_index_status(status: IndexStatus) -> None:
    """Render the current index status after an explicit ingest."""
    print()
    print(format_index_status(status))


def main() -> int:
    configure_runtime_warnings()
    parser = build_parser()
    args = parser.parse_args(normalize_cli_argv(sys.argv[1:]))
    verbose = cast(bool, getattr(args, "verbose", False))
    configure_logging(verbose=verbose)

    try:
        settings = build_settings(args)
        service = AgenticRagService(settings=settings)
        if verbose:
            log_startup_diagnostics(service, command=args.command)

        if args.command == "ingest":
            status = service.ingest()
            logger.info(
                "ingest_complete cache=%s document_count=%s fingerprint=%s",
                status.cache_path,
                status.document_count,
                status.fingerprint,
            )
            print_index_status(status)
            return 0

        if args.diagram:
            service.export_graph_mermaid(args.diagram)

        result = service.query(
            args.question,
            on_step=print_step if args.show_steps else None,
        )
    except ConfigurationError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    except AgenticRagError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    except Exception as exc:
        if os.getenv("AGENTIC_RAG_DEBUG"):
            raise
        parser.exit(status=1, message=f"error: {exc}\n")

    logger.info(
        "query_complete termination_reason=%s rewrites_used=%s document_count=%s",
        result.termination_reason,
        result.rewrites_used,
        result.document_count,
    )
    print("\nFinal answer:\n")
    print(result.answer_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
