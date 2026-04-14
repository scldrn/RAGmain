from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient

from agentic_rag.documents import DOCUMENT_CLEANER_FINGERPRINT, preprocess_documents
from agentic_rag.errors import ConfigurationError, DocumentLoadError, LLMError, VectorStoreError
from agentic_rag.graph import (
    build_agentic_rag_graph,
    create_retriever_tool,
    message_text,
)
from agentic_rag.providers import (
    ResolvedProviderConfig,
    create_chat_model,
    create_embeddings,
    resolve_chat_config,
    resolve_embedding_config,
)
from agentic_rag.retrieval import (
    SPARSE_ENCODER_FINGERPRINT,
    RerankingRetriever,
    TokenSparseEmbeddings,
    retrieval_fetch_k,
)
from agentic_rag.settings import AgenticRagSettings

QDRANT_COLLECTION_NAME = "documents"
DEFAULT_MODEL_MAX_ATTEMPTS = 3
DEFAULT_MODEL_RETRY_BACKOFF_SECONDS = 0.5
logger = logging.getLogger(__name__)
T = TypeVar("T")


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
    collection_name: str = QDRANT_COLLECTION_NAME


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
    collection_name: str = QDRANT_COLLECTION_NAME


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


def _optional_httpx() -> Any | None:
    try:
        import httpx
    except Exception:  # pragma: no cover - httpx is installed in normal runtime/tests
        return None
    return httpx


def _effective_embedding_api_base(settings: AgenticRagSettings) -> str | None:
    if settings.embedding_api_base:
        return settings.embedding_api_base
    if settings.embedding_provider == settings.chat_provider:
        return settings.chat_api_base
    return None


def _qdrant_collection_name(settings: AgenticRagSettings) -> str:
    return settings.collection_name or QDRANT_COLLECTION_NAME


def _llm_error_message(exc: Exception, *, default: str) -> str:
    httpx = _optional_httpx()

    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 401:
            return (
                "Model request failed with 401 Unauthorized. "
                "Check the provider API key and API base."
            )
        if status_code == 429:
            return (
                "Model request failed with 429 Too Many Requests. "
                "Check quota or rate limits and retry."
            )

    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return (
            "Model request timed out after the configured timeout. "
            "Retry the request or increase MODEL_TIMEOUT_SECONDS."
        )

    if httpx is not None and isinstance(exc, httpx.RemoteProtocolError):
        return "Model request failed due to a transient network/protocol error. Retry the request."

    return default


def _model_retry_delay_seconds(attempt: int) -> float:
    return DEFAULT_MODEL_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    direct_status_code = getattr(exc, "status_code", None)
    if isinstance(direct_status_code, int):
        return direct_status_code

    return None


def _should_retry_model_error(exc: Exception) -> bool:
    status_code = _http_status_code(exc)
    if status_code in {408, 409, 425, 429}:
        return True
    if status_code is not None and 500 <= status_code < 600:
        return True

    httpx = _optional_httpx()

    if httpx is not None and isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    return isinstance(exc, (TimeoutError, ConnectionError))


def _invoke_with_retries(
    *,
    operation_name: str,
    max_attempts: int,
    call: Callable[[], T],
    root_cause_lost_message: str,
) -> T:
    if max_attempts <= 0:
        raise LLMError(root_cause_lost_message)

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts or not _should_retry_model_error(exc):
                break

            delay_seconds = _model_retry_delay_seconds(attempt)
            logger.warning(
                "%s failed on attempt %s/%s; retrying after %.2fs.",
                operation_name,
                attempt,
                max_attempts,
                delay_seconds,
                exc_info=exc,
            )
            time.sleep(delay_seconds)

    if last_error is None:  # pragma: no cover - loop always sets or returns
        raise LLMError(root_cause_lost_message)
    raise last_error


@dataclass(frozen=True)
class RetryingChatModel:
    """Wrap a chat model with deterministic retry/backoff behavior."""

    model: Any
    max_attempts: int = DEFAULT_MODEL_MAX_ATTEMPTS

    def invoke(self, messages: Any) -> Any:
        return _invoke_with_retries(
            operation_name="Model call",
            max_attempts=self.max_attempts,
            call=lambda: self.model.invoke(messages),
            root_cause_lost_message="Model call failed without preserving the root cause.",
        )


@dataclass(frozen=True)
class RetryingEmbeddings(Embeddings):
    """Wrap embeddings with deterministic retry/backoff behavior."""

    embeddings: Embeddings
    max_attempts: int = DEFAULT_MODEL_MAX_ATTEMPTS

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _invoke_with_retries(
            operation_name="Embeddings call",
            max_attempts=self.max_attempts,
            call=lambda: self.embeddings.embed_documents(texts),
            root_cause_lost_message="Embeddings call failed without preserving the root cause.",
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class _DiagramChatModel:
    """Minimal model stub used only to compile the static graph diagram."""

    def invoke(self, _messages: Any) -> AIMessage:
        return AIMessage(content="")


class _DiagramRetriever:
    """Minimal retriever stub used only to compile the static graph diagram."""

    def invoke(self, _query: str) -> list[Any]:
        return []


def _qdrant_retrieval_mode(settings: AgenticRagSettings) -> RetrievalMode:
    if settings.retrieval_mode == "dense":
        return RetrievalMode.DENSE
    if settings.retrieval_mode == "hybrid":
        return RetrievalMode.HYBRID
    raise ConfigurationError(f"Unsupported retrieval_mode '{settings.retrieval_mode}'.")


def _qdrant_sparse_embeddings(settings: AgenticRagSettings) -> SparseEmbeddings | None:
    retrieval_mode = _qdrant_retrieval_mode(settings)
    if retrieval_mode is RetrievalMode.HYBRID:
        return TokenSparseEmbeddings()
    return None


def index_fingerprint(settings: AgenticRagSettings) -> str:
    """Compute a stable fingerprint for the current source/index configuration."""
    retrieval_mode = _qdrant_retrieval_mode(settings)
    sparse_fingerprint = (
        SPARSE_ENCODER_FINGERPRINT if retrieval_mode is RetrievalMode.HYBRID else None
    )
    payload = {
        "source_urls": list(settings.source_urls),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "collection_name": _qdrant_collection_name(settings),
        "retrieval_mode": retrieval_mode.value,
        "document_cleaner": DOCUMENT_CLEANER_FINGERPRINT,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_api_base": _effective_embedding_api_base(settings),
        "sparse_encoder": sparse_fingerprint,
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


def _collection_document_count(
    client: QdrantClient,
    *,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> int:
    return int(client.count(collection_name=collection_name, exact=True).count or 0)


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
    retrieval_mode: RetrievalMode,
    sparse_embeddings: SparseEmbeddings | None,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> QdrantVectorStore | None:
    if not cache_path.exists():
        return None

    client = QdrantClient(path=str(cache_path))
    try:
        if not client.collection_exists(collection_name):
            _close_qdrant_client(client)
            return None

        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            retrieval_mode=retrieval_mode,
            sparse_embedding=sparse_embeddings,
        )
    except Exception:
        _close_qdrant_client(client)
        raise


def _build_vectorstore(
    settings: AgenticRagSettings,
    *,
    cache_path: Path,
    embeddings: Embeddings,
    retrieval_mode: RetrievalMode,
    sparse_embeddings: SparseEmbeddings | None,
    collection_name: str = QDRANT_COLLECTION_NAME,
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
            collection_name=collection_name,
            retrieval_mode=retrieval_mode,
            sparse_embedding=sparse_embeddings,
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
    retrieval_mode: RetrievalMode,
    sparse_embeddings: SparseEmbeddings | None,
    collection_name: str = QDRANT_COLLECTION_NAME,
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

    return _build_vectorstore(
        settings,
        cache_path=cache_path,
        embeddings=embeddings,
        retrieval_mode=retrieval_mode,
        sparse_embeddings=sparse_embeddings,
        collection_name=collection_name,
    )


def load_or_create_vectorstore(
    settings: AgenticRagSettings,
    *,
    embeddings: Embeddings,
    create_if_missing: bool = True,
) -> tuple[QdrantVectorStore, int]:
    """Load a persisted Qdrant index when possible, otherwise build it locally on disk."""
    cache_path = index_cache_path(settings)
    cache_exists = cache_path.exists()
    retrieval_mode = _qdrant_retrieval_mode(settings)
    sparse_embeddings = _qdrant_sparse_embeddings(settings)
    collection_name = _qdrant_collection_name(settings)

    try:
        vectorstore = _load_existing_vectorstore(
            cache_path,
            embeddings=embeddings,
            retrieval_mode=retrieval_mode,
            sparse_embeddings=sparse_embeddings,
            collection_name=collection_name,
        )
    except Exception as exc:
        return _rebuild_corrupted_vectorstore(
            settings,
            cache_path=cache_path,
            embeddings=embeddings,
            retrieval_mode=retrieval_mode,
            sparse_embeddings=sparse_embeddings,
            collection_name=collection_name,
            reason=exc,
        )
    if vectorstore is not None:
        return vectorstore, _collection_document_count(
            vectorstore.client,
            collection_name=collection_name,
        )

    if cache_exists:
        return _rebuild_corrupted_vectorstore(
            settings,
            cache_path=cache_path,
            embeddings=embeddings,
            retrieval_mode=retrieval_mode,
            sparse_embeddings=sparse_embeddings,
            collection_name=collection_name,
        )

    if not create_if_missing:
        raise VectorStoreError(
            f"No persisted Qdrant index exists at '{cache_path}'. "
            "Run ingest() first or set ingestion_mode='auto'."
        )

    return _build_vectorstore(
        settings,
        cache_path=cache_path,
        embeddings=embeddings,
        retrieval_mode=retrieval_mode,
        sparse_embeddings=sparse_embeddings,
        collection_name=collection_name,
    )


@dataclass(frozen=True)
class AgenticRagService:
    """Shared service layer for indexing, querying, and runtime diagnostics."""

    settings: AgenticRagSettings = field(default_factory=AgenticRagSettings)

    def health(self) -> HealthStatus:
        """Return configuration health without performing expensive work."""
        try:
            chat_cfg = resolve_chat_config(self.settings)
            resolve_embedding_config(self.settings, chat_cfg)
            _qdrant_retrieval_mode(self.settings)
            _qdrant_sparse_embeddings(self.settings)
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
                collection_name=_qdrant_collection_name(self.settings),
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
            collection_name=_qdrant_collection_name(self.settings),
        )

    def index_status(self) -> IndexStatus:
        """Return the current persisted index status for this settings profile."""
        cache_path = index_cache_path(self.settings)
        if not cache_path.exists():
            return self._known_index_status(exists=False)

        client: QdrantClient | None = None
        try:
            client = QdrantClient(path=str(cache_path))
            collection_name = _qdrant_collection_name(self.settings)
            exists = client.collection_exists(collection_name)
            document_count = (
                _collection_document_count(client, collection_name=collection_name)
                if exists
                else None
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to inspect the persisted Qdrant index at '{cache_path}'."
            ) from exc
        finally:
            _close_qdrant_client(client)

        return self._known_index_status(exists=exists, document_count=document_count)

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
            return self._known_index_status(exists=True, document_count=document_count)
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
            raise LLMError(
                _llm_error_message(exc, default="Failed while running the query graph.")
            ) from exc
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
            index_status=self._known_index_status(exists=True, document_count=document_count),
            steps=tuple(steps),
            termination_reason=termination_reason,
            rewrites_used=rewrites_used,
        )

    def render_graph_mermaid(self) -> str:
        """Render the current graph as Mermaid text."""
        try:
            graph = build_agentic_rag_graph(
                response_model=cast(Any, _DiagramChatModel()),
                grader_model=cast(Any, _DiagramChatModel()),
                retriever_tool=create_retriever_tool(_DiagramRetriever()),
                max_rewrites=self.settings.max_rewrites,
            )
            return graph.get_graph().draw_mermaid()
        except Exception as exc:
            raise LLMError("Failed to render the graph diagram.") from exc

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
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": retrieval_fetch_k(self.settings.retrieval_k)}
            )
            reranking_retriever = RerankingRetriever(
                retriever,
                final_k=self.settings.retrieval_k,
            )
            retriever_tool = create_retriever_tool(reranking_retriever)
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

    def _create_chat_model(self, config: ResolvedProviderConfig) -> RetryingChatModel:
        return RetryingChatModel(
            model=self._build_provider_client(
                config,
                factory=create_chat_model,
                error_message=f"Failed to create the chat model for provider '{config.provider}'.",
            )
        )

    def _create_embeddings(self, config: ResolvedProviderConfig) -> Embeddings:
        return RetryingEmbeddings(
            embeddings=self._build_provider_client(
                config,
                factory=create_embeddings,
                error_message=(
                    f"Failed to create the embeddings client for provider '{config.provider}'."
                ),
            )
        )

    def _resolve_provider_configs(
        self,
    ) -> tuple[ResolvedProviderConfig, ResolvedProviderConfig]:
        chat_config = resolve_chat_config(self.settings)
        embedding_config = resolve_embedding_config(self.settings, chat_config)
        return chat_config, embedding_config

    def _known_index_status(
        self,
        *,
        exists: bool,
        document_count: int | None = None,
    ) -> IndexStatus:
        return IndexStatus(
            cache_path=index_cache_path(self.settings),
            fingerprint=index_fingerprint(self.settings),
            exists=exists,
            document_count=document_count,
            collection_name=_qdrant_collection_name(self.settings),
        )

    @staticmethod
    def _build_provider_client(
        config: ResolvedProviderConfig,
        *,
        factory: Callable[[ResolvedProviderConfig], T],
        error_message: str,
    ) -> T:
        try:
            return factory(config)
        except Exception as exc:
            raise LLMError(error_message) from exc
