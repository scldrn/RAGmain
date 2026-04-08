from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterable, Sequence

os.environ.setdefault("USER_AGENT", "agentic-rag/0.1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.errors import DocumentLoadError

DEFAULT_FETCH_MAX_ATTEMPTS = 3
DEFAULT_FETCH_RETRY_BACKOFF_SECONDS = 0.5
DOCUMENT_CLEANER_FINGERPRINT = "web-clean-v6"
EXACT_NOISE_LINES = {
    "",
    "|",
    "«",
    "»",
    "citation",
    "references",
    "lil'log",
    "table of contents",
}
NAVIGATION_WORDS = {
    "archive",
    "author",
    "date",
    "estimated",
    "faq",
    "factuality",
    "hallucination",
    "language-model",
    "language",
    "long-read",
    "lil",
    "log",
    "nlp",
    "papermod",
    "posts",
    "powered",
    "read",
    "reinforcement-learning",
    "rlhf",
    "safety",
    "search",
    "tags",
    "time",
}
TITLE_CASE_CONNECTORS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "does",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "via",
    "with",
    "without",
}
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


def _normalize_line(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    return re.sub(r"\s*#\s*$", "", normalized).strip()


def _line_words(line: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", line.lower())


def _is_noise_line(line: str) -> bool:
    normalized = _normalize_line(line)
    lowered = normalized.lower()
    if lowered in EXACT_NOISE_LINES:
        return True
    if lowered.startswith("powered by") or lowered.startswith("© "):
        return True
    if lowered.startswith("date:") or "estimated reading time" in lowered:
        return True

    words = _line_words(normalized)
    if not words:
        return True

    if len(words) <= 12 and set(words) <= NAVIGATION_WORDS:
        return True

    return False


def _is_reference_boundary(line: str, *, current_text: str) -> bool:
    lowered = _normalize_line(line).lower()
    if not current_text:
        return False
    current_text_length = len(current_text)
    sentence_like_count = len(re.findall(r"[.!?]", current_text))
    if current_text_length < 800 and sentence_like_count < 2:
        return False
    return lowered in {"citation", "citation#", "references", "references#"} or lowered.startswith(
        "or @article"
    )


def clean_document_content(text: str) -> str:
    """Remove obvious navigation, tags, and trailing references from loaded page text."""
    lines = [_normalize_line(line) for line in text.splitlines()]
    cleaned_lines: list[str] = []
    previous_line: str | None = None

    for line in lines:
        current_text = "\n".join(cleaned_lines)
        if _is_reference_boundary(line, current_text=current_text):
            break
        if _is_noise_line(line):
            continue
        if line == previous_line:
            continue
        cleaned_lines.append(line)
        previous_line = line

    cleaned_lines = _strip_leading_outline(cleaned_lines)

    if cleaned_lines:
        return "\n".join(cleaned_lines).strip()

    return _normalize_line(text)


def _strip_leading_outline(lines: list[str]) -> list[str]:
    """Drop heading-heavy intro blocks until the first sentence-like line."""
    if not lines:
        return lines

    for index, line in enumerate(lines):
        if _looks_like_sentence_line(line):
            if index >= 1:
                return lines[index:]
            return lines

    return lines


def _looks_like_sentence_line(line: str) -> bool:
    lowered = line.lower()
    word_count = len(_line_words(line))
    if _looks_like_outline_heading(line):
        return False
    if "." in line or "!" in line:
        return True
    if ("?" in line or ":" in line) and word_count >= 8:
        return True
    if word_count >= 12:
        return True
    if "reward hacking occurs" in lowered:
        return True
    return False


def _looks_like_outline_heading(line: str) -> bool:
    stripped = line.strip()
    if "?" not in stripped and ":" not in stripped:
        return False

    heading_candidate = stripped.split(":", 1)[0].rstrip("?").strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", heading_candidate)
    if len(words) < 3:
        return False

    return all(_is_title_case_heading_word(word) for word in words)


def _is_title_case_heading_word(word: str) -> bool:
    lowered = word.lower()
    if lowered in TITLE_CASE_CONNECTORS:
        return True
    if len(word) >= 2 and word[:2].isupper():
        return True
    return word[0].isupper() and word[1:].islower()


def clean_documents(documents: Sequence[Document]) -> list[Document]:
    """Normalize loaded documents while preserving metadata."""
    cleaned_documents: list[Document] = []
    for document in documents:
        cleaned_documents.append(
            Document(
                page_content=clean_document_content(document.page_content),
                metadata=dict(document.metadata),
            )
        )
    return cleaned_documents


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
        return clean_documents(loaded_documents)

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
