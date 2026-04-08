from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_qdrant import SparseEmbeddings, SparseVector

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CITATION_MARKER_PATTERN = re.compile(r"\[\d+\]")
OUTLINE_HEADING_PREFIX_PATTERN = re.compile(r"^(?:[A-Z][^#\n]{1,60}#\s*){1,4}")
SPARSE_ENCODER_FINGERPRINT = "token-hash-v1"
SPARSE_HASH_SPACE = 1_000_003
DEFAULT_CONTEXT_MAX_TOTAL_CHARS = 4_500
DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS = 1_200
DEFAULT_MAX_DOCUMENTS_PER_SOURCE = 2
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "which",
    "with",
    "why",
}
BOILERPLATE_MARKERS = (
    "@article{",
    "references#",
    "citation#",
    "posts archive search tags faq",
    "powered by",
    "table of contents",
    "© ",
)


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _normalized_tokens(text: str) -> tuple[str, ...]:
    base_tokens = [
        token for token in TOKEN_PATTERN.findall(text.lower()) if token and token not in STOPWORDS
    ]
    phrase_tokens = [
        f"{left}_{right}"
        for left, right in zip(base_tokens, base_tokens[1:], strict=False)
        if left != right
    ]
    return tuple(base_tokens + phrase_tokens)


def _line_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", text.lower())


@dataclass(frozen=True)
class _ScoredDocument:
    document: Document
    score: float
    source: str
    content_fingerprint: str
    original_rank: int
    boilerplate: bool


def retrieval_fetch_k(final_k: int) -> int:
    return max(final_k * 2, final_k + 3)


def score_document_for_query(query: str, document: Document, *, rank: int) -> float:
    query_terms = set(_normalized_tokens(query))
    content_counts = Counter(_normalized_tokens(document.page_content))
    title_terms = set(_normalized_tokens(str(document.metadata.get("title", ""))))
    phrase_overlap = sum(min(content_counts.get(term, 0), 2) for term in query_terms if "_" in term)
    term_overlap = sum(
        min(content_counts.get(term, 0), 3) for term in query_terms if "_" not in term
    )
    title_overlap = len(query_terms & title_terms)
    base_score = (
        (term_overlap * 4.0) + (phrase_overlap * 6.0) + (title_overlap * 2.0) + (1.0 / (rank + 1))
    )
    richness_factor = min(max(len(content_counts), 1), 8) / 8.0
    return base_score * richness_factor


def curate_retrieved_documents(
    query: str,
    documents: Sequence[Document],
    *,
    final_k: int,
    max_total_chars: int = DEFAULT_CONTEXT_MAX_TOTAL_CHARS,
    max_document_chars: int = DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS,
    max_documents_per_source: int = DEFAULT_MAX_DOCUMENTS_PER_SOURCE,
) -> list[Document]:
    scored_documents = []
    for index, document in enumerate(documents):
        trimmed_document = _trim_document_for_query(
            query,
            document,
            max_chars=max_document_chars,
        )
        scored_documents.append(
            _ScoredDocument(
                document=trimmed_document,
                score=score_document_for_query(query, trimmed_document, rank=index),
                source=str(document.metadata.get("source", "unknown")),
                content_fingerprint=_document_fingerprint(trimmed_document),
                original_rank=index,
                boilerplate=_looks_like_boilerplate(trimmed_document.page_content),
            )
        )
    scored_documents.sort(key=lambda item: (-item.score, item.original_rank))
    minimum_score = max(1.5, scored_documents[0].score * 0.55) if scored_documents else 0.0

    selected: list[Document] = []
    seen_fingerprints: set[str] = set()
    source_counts: dict[str, int] = {}
    total_chars = 0

    for source_limit in (1, max_documents_per_source):
        for item in scored_documents:
            if len(selected) >= final_k:
                return selected
            if item.score < minimum_score:
                continue
            if item.boilerplate:
                continue
            if item.content_fingerprint in seen_fingerprints:
                continue
            if source_counts.get(item.source, 0) >= source_limit:
                continue
            content_length = len(item.document.page_content)
            if selected and total_chars + content_length > max_total_chars:
                continue

            selected.append(item.document)
            seen_fingerprints.add(item.content_fingerprint)
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
            total_chars += content_length

    return selected


def _document_fingerprint(document: Document) -> str:
    normalized = " ".join(document.page_content.lower().split())
    payload = f"{document.metadata.get('source', 'unknown')}::{normalized[:600]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trim_document_for_query(query: str, document: Document, *, max_chars: int) -> Document:
    best_excerpt = _best_excerpt_for_query(query, document.page_content, max_chars=max_chars)
    return Document(
        page_content=best_excerpt,
        metadata=dict(document.metadata),
    )


def _best_excerpt_for_query(query: str, text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    sentences = [
        re.sub(r"\s+", " ", part).strip()
        for part in SENTENCE_PATTERN.split(stripped)
        if part.strip()
    ]
    if len(stripped) <= max_chars:
        if not sentences:
            return stripped
        return _assemble_excerpt(sentences, start_index=0, max_chars=max_chars)

    if not sentences:
        return stripped[:max_chars].rstrip()

    query_terms = set(_normalized_tokens(query))
    ranked_segments = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _normalized_tokens(sentence)
        sentence_overlap = sum(1 for term in query_terms if term in sentence_terms)
        sentence_richness = len(set(sentence_terms))
        ranked_segments.append(
            (
                0 if _looks_like_boilerplate(sentence) else 1,
                sentence_overlap,
                min(sentence_richness, 12),
                min(len(sentence), max_chars),
                -index,
            )
        )
    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda index: ranked_segments[index],
        reverse=True,
    )
    start_index = ranked_indices[0]
    return _assemble_excerpt(sentences, start_index=start_index, max_chars=max_chars)


def _looks_like_boilerplate(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in BOILERPLATE_MARKERS):
        return True
    if len(CITATION_MARKER_PATTERN.findall(lowered)) >= 3:
        return True

    words = _line_words(text)
    if words and not any(mark in text for mark in ".?!:") and len(words) <= 12:
        return True

    return False


def _looks_like_heading_tail(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    prefix_match = OUTLINE_HEADING_PREFIX_PATTERN.match(stripped)
    if prefix_match is None:
        return False

    prefix = prefix_match.group(0)
    return prefix.count("#") >= 1 and len(prefix) <= max(140, len(stripped) // 2)


def _assemble_excerpt(sentences: Sequence[str], *, start_index: int, max_chars: int) -> str:
    excerpt_parts: list[str] = []
    total_chars = 0
    for sentence in sentences[start_index:]:
        if excerpt_parts and _looks_like_heading_tail(sentence):
            break
        sentence_length = len(sentence) + (1 if excerpt_parts else 0)
        if excerpt_parts and total_chars + sentence_length > max_chars:
            break
        excerpt_parts.append(sentence)
        total_chars += sentence_length

    excerpt = " ".join(excerpt_parts) or sentences[start_index][:max_chars]
    return excerpt[:max_chars].rstrip()


class RerankingRetriever:
    """Wrap a retriever with deterministic reranking and context assembly."""

    def __init__(
        self,
        base_retriever: Any,
        *,
        final_k: int,
        max_total_chars: int = DEFAULT_CONTEXT_MAX_TOTAL_CHARS,
        max_document_chars: int = DEFAULT_CONTEXT_MAX_DOCUMENT_CHARS,
        max_documents_per_source: int = DEFAULT_MAX_DOCUMENTS_PER_SOURCE,
    ) -> None:
        self.base_retriever = base_retriever
        self.final_k = final_k
        self.max_total_chars = max_total_chars
        self.max_document_chars = max_document_chars
        self.max_documents_per_source = max_documents_per_source

    def invoke(self, query: str) -> list[Document]:
        documents = list(self.base_retriever.invoke(query))
        return curate_retrieved_documents(
            query,
            documents,
            final_k=self.final_k,
            max_total_chars=self.max_total_chars,
            max_document_chars=self.max_document_chars,
            max_documents_per_source=self.max_documents_per_source,
        )


class TokenSparseEmbeddings(SparseEmbeddings):
    """Deterministic lexical sparse embeddings for local Qdrant hybrid retrieval."""

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text)

    def _embed(self, text: str) -> SparseVector:
        counts = Counter(_normalized_tokens(text))
        if not counts:
            return SparseVector(indices=[], values=[])

        merged_counts: dict[int, float] = {}
        for token, count in counts.items():
            index = _stable_hash(token) % SPARSE_HASH_SPACE
            merged_counts[index] = merged_counts.get(index, 0.0) + (1.0 + math.log(float(count)))

        indices = sorted(merged_counts)
        return SparseVector(
            indices=indices,
            values=[merged_counts[index] for index in indices],
        )
