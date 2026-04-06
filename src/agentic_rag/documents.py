from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence

os.environ.setdefault("USER_AGENT", "agentic-rag/0.1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.errors import DocumentLoadError

DEFAULT_FETCH_MAX_ATTEMPTS = 3
DEFAULT_FETCH_RETRY_BACKOFF_SECONDS = 0.5
logger = logging.getLogger(__name__)


def _build_web_loader(url: str, *, timeout_seconds: float) -> WebBaseLoader:
    return WebBaseLoader(
        url,
        requests_kwargs={"timeout": timeout_seconds},
        raise_for_status=True,
        show_progress=False,
    )


def _retry_delay_seconds(attempt: int) -> float:
    return DEFAULT_FETCH_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))


def _sleep_before_retry(delay_seconds: float) -> None:
    time.sleep(delay_seconds)


def _load_documents_with_retries(
    url: str,
    *,
    timeout_seconds: float,
    max_attempts: int,
) -> list[Document]:
    last_error: Exception = DocumentLoadError(f"Failed to load source URL '{url}'.")

    for attempt in range(1, max_attempts + 1):
        try:
            return list(_build_web_loader(url, timeout_seconds=timeout_seconds).load())
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break

            logger.warning(
                "Failed to load '%s' on attempt %s/%s; retrying.",
                url,
                attempt,
                max_attempts,
                exc_info=exc,
            )
            _sleep_before_retry(_retry_delay_seconds(attempt))

    raise DocumentLoadError(
        f"Failed to load source URL '{url}' after {max_attempts} attempts."
    ) from last_error


def load_documents(
    urls: Sequence[str],
    *,
    timeout_seconds: float = 20.0,
    max_attempts: int = DEFAULT_FETCH_MAX_ATTEMPTS,
) -> list[Document]:
    """Fetch all source documents, retrying per URL and tolerating partial failures."""
    loaded_documents: list[Document] = []
    failed_urls: list[str] = []

    for url in urls:
        try:
            loaded_documents.extend(
                _load_documents_with_retries(
                    url,
                    timeout_seconds=timeout_seconds,
                    max_attempts=max_attempts,
                )
            )
        except DocumentLoadError as exc:
            failed_urls.append(url)
            logger.warning("Skipping source URL '%s' after repeated failures.", url, exc_info=exc)

    if loaded_documents:
        if failed_urls:
            logger.warning(
                "Loaded %s of %s source URLs; skipped failures: %s",
                len(urls) - len(failed_urls),
                len(urls),
                ", ".join(failed_urls),
            )
        return loaded_documents

    joined_failed_urls = ", ".join(failed_urls) or "the configured sources"
    raise DocumentLoadError(f"Failed to load any source documents from: {joined_failed_urls}.")


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into smaller chunks for semantic search."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def preprocess_documents(
    urls: Sequence[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    fetch_timeout_seconds: float = 20.0,
) -> list[Document]:
    """Load and split source documents in one step."""
    documents = load_documents(urls, timeout_seconds=fetch_timeout_seconds)
    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
