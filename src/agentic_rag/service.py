from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from agentic_rag.documents import preprocess_documents
from agentic_rag.errors import ConfigurationError, DocumentLoadError, LLMError, VectorStoreError
from agentic_rag.graph import (
    build_agentic_rag_graph,
    create_retriever_tool,
    message_text,
)
from agentic_rag.providers import (
    create_chat_model,
    create_embeddings,
    resolve_chat_config,
    resolve_embedding_config,
)
from agentic_rag.settings import AgenticRagSettings


@dataclass(frozen=True)
class QueryStep:
    """Single graph step emitted while answering a query."""

    node_name: str
    message: BaseMessage


@dataclass(frozen=True)
class IndexStatus:
    """Summary of the persisted index state for the current settings."""

    cache_path: Path
    fingerprint: str
    exists: bool
    document_count: int | None = None


@dataclass(frozen=True)
class HealthStatus:
    """High-level runtime health information without performing network calls."""

    ok: bool
    chat_provider: str
    chat_model: str
    embedding_provider: str
    embedding_model: str
    ingestion_mode: str
    retrieval_mode: str
    index_cache_dir: Path
    detail: str | None = None


@dataclass(frozen=True)
class QueryResult:
    """Structured query response returned by the shared service layer."""

    final_message: BaseMessage
    answer_text: str
    document_count: int
    index_status: IndexStatus
    steps: tuple[QueryStep, ...] = ()
    termination_reason: str | None = None
    rewrites_used: int = 0


StepHandler = Callable[[QueryStep], None]
QDRANT_COLLECTION_NAME = "documents"
QDRANT_RETRIEVAL_MODE = RetrievalMode.DENSE
logger = logging.getLogger(__name__)


def _effective_embedding_api_base(settings: AgenticRagSettings) -> str | None:
    if settings.embedding_api_base:
        return settings.embedding_api_base
    if settings.embedding_provider == settings.chat_provider:
        return settings.chat_api_base
    return None


def index_fingerprint(settings: AgenticRagSettings) -> str:
    """Compute a stable fingerprint for the current source/index configuration."""
    payload = {
        "source_urls": list(settings.source_urls),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_api_base": _effective_embedding_api_base(settings),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def index_cache_path(settings: AgenticRagSettings) -> Path:
    """Return the persisted cache path for the current settings."""
    return Path(settings.index_cache_dir) / index_fingerprint(settings)


def _close_qdrant_client(client: QdrantClient | None) -> None:
    if client is None:
        return

    close = getattr(client, "close", None)
    if callable(close):
        close()


def _collection_document_count(client: QdrantClient) -> int:
    return int(client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count or 0)


def _remove_cache_path(cache_path: Path) -> None:
    if not cache_path.exists():
        return

    if cache_path.is_dir():
        shutil.rmtree(cache_path)
        return

    cache_path.unlink()


def _load_existing_vectorstore(
    cache_path: Path,
    *,
    embeddings: Embeddings,
) -> QdrantVectorStore | None:
    if not cache_path.exists():
        return None

    client = QdrantClient(path=str(cache_path))
    try:
        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            _close_qdrant_client(client)
            return None

        return QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings,
            retrieval_mode=QDRANT_RETRIEVAL_MODE,
        )
    except Exception:
        _close_qdrant_client(client)
        raise


def _build_vectorstore(
    settings: AgenticRagSettings,
    *,
    cache_path: Path,
    embeddings: Embeddings,
) -> tuple[QdrantVectorStore, int]:
    try:
        documents = preprocess_documents(
            settings.source_urls,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            fetch_timeout_seconds=settings.fetch_timeout_seconds,
        )
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(
            "Failed to load and preprocess the configured source URLs."
        ) from exc

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore = QdrantVectorStore.from_documents(
            documents=list(documents),
            embedding=embeddings,
            path=str(cache_path),
            collection_name=QDRANT_COLLECTION_NAME,
            retrieval_mode=QDRANT_RETRIEVAL_MODE,
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to build or persist the Qdrant index at '{cache_path}'."
        ) from exc

    return vectorstore, len(documents)


def _rebuild_corrupted_vectorstore(
    settings: AgenticRagSettings,
    *,
    cache_path: Path,
    embeddings: Embeddings,
    reason: Exception | None = None,
) -> tuple[QdrantVectorStore, int]:
    if reason is None:
        logger.warning("Detected an incomplete Qdrant index at '%s'; rebuilding.", cache_path)
    else:
        logger.warning(
            "Detected a corrupted Qdrant index at '%s'; rebuilding.",
            cache_path,
            exc_info=reason,
        )

    try:
        _remove_cache_path(cache_path)
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to remove the invalid Qdrant index at '{cache_path}'."
        ) from exc

    return _build_vectorstore(settings, cache_path=cache_path, embeddings=embeddings)


def load_or_create_vectorstore(
    settings: AgenticRagSettings,
    *,
    embeddings: Embeddings,
    create_if_missing: bool = True,
) -> tuple[QdrantVectorStore, int]:
    """Load a persisted Qdrant index when possible, otherwise build it locally on disk."""
    cache_path = index_cache_path(settings)
    cache_exists = cache_path.exists()

    try:
        vectorstore = _load_existing_vectorstore(cache_path, embeddings=embeddings)
    except Exception as exc:
        return _rebuild_corrupted_vectorstore(
            settings,
            cache_path=cache_path,
            embeddings=embeddings,
            reason=exc,
        )
    if vectorstore is not None:
        return vectorstore, _collection_document_count(vectorstore.client)

    if cache_exists:
        return _rebuild_corrupted_vectorstore(
            settings,
            cache_path=cache_path,
            embeddings=embeddings,
        )

    if not create_if_missing:
        raise VectorStoreError(
            f"No persisted Qdrant index exists at '{cache_path}'. "
            "Run ingest() first or set ingestion_mode='auto'."
        )

    return _build_vectorstore(settings, cache_path=cache_path, embeddings=embeddings)


@dataclass(frozen=True)
class AgenticRagService:
    """Shared service layer for indexing, querying, and runtime diagnostics."""

    settings: AgenticRagSettings = field(default_factory=AgenticRagSettings)

    def health(self) -> HealthStatus:
        """Return configuration health without performing expensive work."""
        try:
            chat_cfg = resolve_chat_config(self.settings)
            resolve_embedding_config(self.settings, chat_cfg)
        except ConfigurationError as exc:
            return HealthStatus(
                ok=False,
                chat_provider=self.settings.chat_provider,
                chat_model=self.settings.chat_model,
                embedding_provider=self.settings.embedding_provider or "",
                embedding_model=self.settings.embedding_model or "",
                ingestion_mode=self.settings.ingestion_mode,
                retrieval_mode=self.settings.retrieval_mode,
                index_cache_dir=Path(self.settings.index_cache_dir),
                detail=str(exc),
            )

        return HealthStatus(
            ok=True,
            chat_provider=self.settings.chat_provider,
            chat_model=self.settings.chat_model,
            embedding_provider=self.settings.embedding_provider or "",
            embedding_model=self.settings.embedding_model or "",
            ingestion_mode=self.settings.ingestion_mode,
            retrieval_mode=self.settings.retrieval_mode,
            index_cache_dir=Path(self.settings.index_cache_dir),
        )

    def index_status(self) -> IndexStatus:
        """Return the current persisted index status for this settings profile."""
        cache_path = index_cache_path(self.settings)
        if not cache_path.exists():
            return IndexStatus(
                cache_path=cache_path,
                fingerprint=index_fingerprint(self.settings),
                exists=False,
            )

        client = QdrantClient(path=str(cache_path))
        try:
            exists = client.collection_exists(QDRANT_COLLECTION_NAME)
            document_count = _collection_document_count(client) if exists else None
        finally:
            _close_qdrant_client(client)

        return IndexStatus(
            cache_path=cache_path,
            fingerprint=index_fingerprint(self.settings),
            exists=exists,
            document_count=document_count,
        )

    def ingest(self) -> IndexStatus:
        """Build or reuse the persisted index for the current settings."""
        logger.info(
            "ingest_start cache=%s ingestion_mode=%s",
            index_cache_path(self.settings),
            self.settings.ingestion_mode,
        )
        _chat_config, embedding_config = self._resolve_provider_configs()
        embeddings = self._create_embeddings(embedding_config)
        vectorstore, document_count = load_or_create_vectorstore(
            self.settings, embeddings=embeddings
        )
        try:
            logger.info("ingest_ready document_count=%s", document_count)
            return self._index_status(document_count=document_count)
        finally:
            _close_qdrant_client(vectorstore.client)

    def query(
        self,
        question: str,
        *,
        on_step: StepHandler | None = None,
    ) -> QueryResult:
        """Run a single question through the graph and return a structured result."""
        if not question.strip():
            raise ConfigurationError("Question must be a non-empty string.")

        logger.info(
            "query_start cache=%s retrieval_mode=%s rewrite_limit=%s question_chars=%s",
            index_cache_path(self.settings),
            self.settings.retrieval_mode,
            self.settings.max_rewrites,
            len(question.strip()),
        )
        graph, document_count, vectorstore = self._build_query_graph()
        final_message: BaseMessage | None = None
        final_node_name: str | None = None
        steps: list[QueryStep] = []

        try:
            for chunk in graph.stream(
                {
                    "messages": [{"role": "user", "content": question}],
                    "rewrite_count": 0,
                }
            ):
                for node_name, update in chunk.items():
                    message = update["messages"][-1]
                    step = QueryStep(node_name=node_name, message=message)
                    steps.append(step)
                    logger.debug(
                        "query_step node=%s message=%s",
                        node_name,
                        message_text(message).strip(),
                    )
                    if on_step is not None:
                        on_step(step)
                    final_message = message
                    final_node_name = node_name
        except Exception as exc:
            raise LLMError("Failed while running the query graph.") from exc
        finally:
            _close_qdrant_client(vectorstore.client)

        if final_message is None:
            raise LLMError("The query graph did not return any messages.")

        termination_reason = (
            "insufficient_context" if final_node_name == "insufficient_context" else None
        )
        rewrites_used = sum(1 for step in steps if step.node_name == "rewrite_question")
        logger.info(
            "query_finished termination_reason=%s rewrites_used=%s document_count=%s",
            termination_reason,
            rewrites_used,
            document_count,
        )
        return QueryResult(
            final_message=final_message,
            answer_text=message_text(final_message).strip(),
            document_count=document_count,
            index_status=self._index_status(document_count=document_count),
            steps=tuple(steps),
            termination_reason=termination_reason,
            rewrites_used=rewrites_used,
        )

    def render_graph_mermaid(self) -> str:
        """Render the current graph as Mermaid text."""
        graph, _document_count, vectorstore = self._build_query_graph()
        try:
            return graph.get_graph().draw_mermaid()
        except Exception as exc:
            raise LLMError("Failed to render the graph diagram.") from exc
        finally:
            _close_qdrant_client(vectorstore.client)

    def export_graph_mermaid(self, output_path: str | Path) -> None:
        """Write the current graph diagram to disk."""
        path = Path(output_path)
        path.write_text(self.render_graph_mermaid(), encoding="utf-8")

    def _build_query_graph(self) -> tuple[Any, int, QdrantVectorStore]:
        chat_config, embedding_config = self._resolve_provider_configs()
        embeddings = self._create_embeddings(embedding_config)
        vectorstore, document_count = load_or_create_vectorstore(
            self.settings,
            embeddings=embeddings,
            create_if_missing=self.settings.ingestion_mode == "auto",
        )

        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": self.settings.retrieval_k})
            retriever_tool = create_retriever_tool(retriever)
            response_model = self._create_chat_model(chat_config)
            grader_model = self._create_chat_model(chat_config)
            graph = build_agentic_rag_graph(
                response_model=response_model,
                grader_model=grader_model,
                retriever_tool=retriever_tool,
                max_rewrites=self.settings.max_rewrites,
            )
        except Exception as exc:
            _close_qdrant_client(vectorstore.client)
            if isinstance(exc, LLMError):
                raise
            raise LLMError("Failed to build the query graph.") from exc

        return graph, document_count, vectorstore

    def _create_chat_model(self, config: Any) -> Any:
        try:
            return create_chat_model(config)
        except Exception as exc:
            raise LLMError(
                f"Failed to create the chat model for provider '{config.provider}'."
            ) from exc

    def _create_embeddings(self, config: Any) -> Embeddings:
        try:
            return create_embeddings(config)
        except Exception as exc:
            raise LLMError(
                f"Failed to create the embeddings client for provider '{config.provider}'."
            ) from exc

    def _resolve_provider_configs(self) -> tuple[Any, Any]:
        chat_config = resolve_chat_config(self.settings)
        embedding_config = resolve_embedding_config(self.settings, chat_config)
        return chat_config, embedding_config

    def _index_status(self, *, document_count: int | None = None) -> IndexStatus:
        status = self.index_status()
        return IndexStatus(
            cache_path=status.cache_path,
            fingerprint=status.fingerprint,
            exists=status.exists,
            document_count=document_count,
        )
