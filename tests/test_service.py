from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_qdrant import RetrievalMode

from agentic_rag.errors import (
    ConfigurationError,
    DocumentLoadError,
    LLMError,
    VectorStoreError,
)
from agentic_rag.retrieval import retrieval_fetch_k
from agentic_rag.service import (
    DEFAULT_MODEL_RETRY_BACKOFF_SECONDS,
    QDRANT_COLLECTION_NAME,
    AgenticRagService,
    IndexStatus,
    RetryingChatModel,
    RetryingEmbeddings,
    _build_vectorstore,
    _close_qdrant_client,
    _collection_document_count,
    _DiagramChatModel,
    _DiagramRetriever,
    _effective_embedding_api_base,
    _http_status_code,
    _llm_error_message,
    _load_existing_vectorstore,
    _qdrant_retrieval_mode,
    _remove_cache_path,
    _should_retry_model_error,
    index_cache_path,
    index_fingerprint,
    load_or_create_vectorstore,
)
from agentic_rag.settings import AgenticRagSettings


@dataclass(frozen=True)
class FakeConfig:
    provider: str
    model: str


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        total = sum(ord(char) for char in text)
        return [float(len(text)), float(total % 97), float((total // 97) % 89), 1.0]


class ClosingClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeVectorStore:
    def __init__(self, client: ClosingClient | None = None):
        self.client = client or ClosingClient()
        self.retriever_calls: list[dict[str, int]] = []

    def as_retriever(self, *, search_kwargs):
        self.retriever_calls.append(search_kwargs)
        return {"search_kwargs": search_kwargs}


class FakeGraph:
    def __init__(
        self,
        *,
        message: AIMessage | None = None,
        error: Exception | None = None,
        node_name: str = "generate_answer",
        chunks: list[tuple[str, AIMessage]] | None = None,
    ):
        self.message = message or AIMessage(content="Final answer")
        self.error = error
        self.node_name = node_name
        self.chunks = chunks

    def stream(self, _payload):
        if self.error is not None:
            raise self.error
        for node_name, message in self.chunks or [(self.node_name, self.message)]:
            yield {node_name: {"messages": [message]}}

    def get_graph(self):
        return self

    def draw_mermaid(self):
        return "graph TD"


class EmptyGraph:
    def stream(self, _payload):
        if False:
            yield {}
        return


class BrokenDrawGraph(FakeGraph):
    def draw_mermaid(self):
        raise RuntimeError("draw failed")


class FakeQdrantClient:
    def __init__(
        self,
        *,
        exists: bool = True,
        count: int = 0,
        error: Exception | None = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
    ):
        self.exists = exists
        self.document_count = count
        self.error = error
        self.collection_name = collection_name
        self.closed = False

    def collection_exists(self, name: str) -> bool:
        if self.error is not None:
            raise self.error
        if name != self.collection_name:
            return False
        return self.exists

    def count(self, *, collection_name: str, exact: bool):
        assert collection_name == self.collection_name
        assert exact is True
        return SimpleNamespace(count=self.document_count)

    def close(self):
        self.closed = True


def make_settings(tmp_path) -> AgenticRagSettings:
    return AgenticRagSettings(
        source_urls=("https://example.com/doc",),
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        chat_api_key="chat-key",
        embedding_provider="google",
        embedding_model="gemini-embedding-2-preview",
        index_cache_dir=str(tmp_path),
        retrieval_mode="dense",
    )


def make_documents() -> list[Document]:
    return [
        Document(page_content="alpha content", metadata={"source": "https://example.com/1"}),
        Document(page_content="beta content", metadata={"source": "https://example.com/2"}),
    ]


def test_close_qdrant_client_handles_none_and_closeable():
    client = ClosingClient()

    _close_qdrant_client(None)
    _close_qdrant_client(object())
    _close_qdrant_client(client)

    assert client.closed is True


def test_collection_document_count_reads_qdrant_response():
    client = FakeQdrantClient(count=3)

    assert _collection_document_count(client) == 3


def test_remove_cache_path_handles_files_and_directories(tmp_path):
    _remove_cache_path(tmp_path / "missing")

    file_path = tmp_path / "corrupt-file"
    directory_path = tmp_path / "corrupt-dir"
    nested_file = directory_path / "data.txt"
    file_path.write_text("bad", encoding="utf-8")
    directory_path.mkdir()
    nested_file.write_text("bad", encoding="utf-8")

    _remove_cache_path(file_path)
    _remove_cache_path(directory_path)

    assert file_path.exists() is False
    assert directory_path.exists() is False


def test_load_existing_vectorstore_returns_none_for_missing_path(tmp_path):
    assert (
        _load_existing_vectorstore(
            tmp_path / "missing",
            embeddings=FakeEmbeddings(),
            retrieval_mode=RetrievalMode.DENSE,
            sparse_embeddings=None,
        )
        is None
    )


def test_load_existing_vectorstore_returns_none_when_collection_is_missing(monkeypatch, tmp_path):
    cache_path = tmp_path / "index"
    cache_path.mkdir()
    client = FakeQdrantClient(exists=False)

    monkeypatch.setattr("agentic_rag.service.QdrantClient", lambda path: client)

    result = _load_existing_vectorstore(
        cache_path,
        embeddings=FakeEmbeddings(),
        retrieval_mode=RetrievalMode.DENSE,
        sparse_embeddings=None,
    )

    assert result is None
    assert client.closed is True


def test_load_existing_vectorstore_closes_client_when_vectorstore_init_fails(monkeypatch, tmp_path):
    cache_path = tmp_path / "index"
    cache_path.mkdir()
    client = FakeQdrantClient(exists=True)

    monkeypatch.setattr("agentic_rag.service.QdrantClient", lambda path: client)

    def fail_vectorstore(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("agentic_rag.service.QdrantVectorStore", fail_vectorstore)

    with pytest.raises(RuntimeError, match="boom"):
        _load_existing_vectorstore(
            cache_path,
            embeddings=FakeEmbeddings(),
            retrieval_mode=RetrievalMode.DENSE,
            sparse_embeddings=None,
        )

    assert client.closed is True


def test_load_existing_vectorstore_passes_retrieval_mode(monkeypatch, tmp_path):
    cache_path = tmp_path / "index"
    cache_path.mkdir()
    client = FakeQdrantClient(exists=True)
    seen_kwargs: dict[str, Any] = {}

    monkeypatch.setattr("agentic_rag.service.QdrantClient", lambda path: client)

    def fake_vectorstore(**kwargs):
        seen_kwargs.update(kwargs)
        return SimpleNamespace(client=client)

    monkeypatch.setattr("agentic_rag.service.QdrantVectorStore", fake_vectorstore)

    vectorstore = _load_existing_vectorstore(
        cache_path,
        embeddings=FakeEmbeddings(),
        retrieval_mode=RetrievalMode.DENSE,
        sparse_embeddings=None,
    )

    assert vectorstore is not None
    assert seen_kwargs["retrieval_mode"] is RetrievalMode.DENSE


def test_load_existing_vectorstore_passes_sparse_embedding_for_hybrid(monkeypatch, tmp_path):
    cache_path = tmp_path / "index"
    cache_path.mkdir()
    client = FakeQdrantClient(exists=True)
    seen_kwargs: dict[str, Any] = {}

    monkeypatch.setattr("agentic_rag.service.QdrantClient", lambda path: client)

    def fake_vectorstore(**kwargs):
        seen_kwargs.update(kwargs)
        return SimpleNamespace(client=client)

    monkeypatch.setattr("agentic_rag.service.QdrantVectorStore", fake_vectorstore)

    vectorstore = _load_existing_vectorstore(
        cache_path,
        embeddings=FakeEmbeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embeddings=SimpleNamespace(),
    )

    assert vectorstore is not None
    assert seen_kwargs["retrieval_mode"] is RetrievalMode.HYBRID
    assert seen_kwargs["sparse_embedding"] is not None


def test_index_status_reports_missing_cache(tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    status = service.index_status()

    assert status.exists is False
    assert status.cache_path.parent == tmp_path
    assert status.document_count is None


def test_index_status_reports_directory_without_collection(tmp_path):
    settings = make_settings(tmp_path)
    cache_path = index_cache_path(settings)
    cache_path.mkdir(parents=True)
    service = AgenticRagService(settings)

    status = service.index_status()

    assert status.cache_path == cache_path
    assert status.exists is False
    assert status.document_count is None


def test_index_status_reports_existing_qdrant_collection(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    embeddings = FakeEmbeddings()
    documents = make_documents()
    monkeypatch.setattr(
        "agentic_rag.service.preprocess_documents", lambda *args, **kwargs: documents
    )

    vectorstore, _document_count = load_or_create_vectorstore(settings, embeddings=embeddings)
    _close_qdrant_client(vectorstore.client)

    status = AgenticRagService(settings).index_status()

    assert status.exists is True
    assert status.document_count == len(documents)


def test_index_status_wraps_corrupted_cache_errors(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    service = AgenticRagService(settings)
    cache_path = index_cache_path(settings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("corrupt", encoding="utf-8")

    monkeypatch.setattr(
        "agentic_rag.service.QdrantClient",
        lambda path: (_ for _ in ()).throw(FileExistsError("bad cache")),
    )

    with pytest.raises(VectorStoreError, match="Failed to inspect the persisted Qdrant index"):
        service.index_status()


def test_effective_embedding_api_base_prefers_embedding_base(tmp_path):
    settings = make_settings(tmp_path)
    settings = AgenticRagSettings(
        **{**settings.__dict__, "embedding_api_base": "https://embed.example"}
    )

    assert _effective_embedding_api_base(settings) == "https://embed.example"


def test_effective_embedding_api_base_falls_back_to_chat_base(tmp_path):
    settings = AgenticRagSettings(
        source_urls=("https://example.com/doc",),
        chat_provider="openai-compatible",
        chat_model="local-chat",
        chat_api_key="chat-key",
        chat_api_base="https://chat.example",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_api_key="embed-key",
        index_cache_dir=str(tmp_path),
    )

    assert _effective_embedding_api_base(settings) == "https://chat.example"


def test_effective_embedding_api_base_returns_none_when_providers_differ(tmp_path):
    settings = AgenticRagSettings(
        source_urls=("https://example.com/doc",),
        chat_provider="google",
        chat_model="gemini-2.5-flash",
        chat_api_key="chat-key",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_api_key="embed-key",
        index_cache_dir=str(tmp_path),
    )

    assert _effective_embedding_api_base(settings) is None


def test_llm_error_message_handles_timeout():
    timeout_error = httpx.ReadTimeout("timed out")

    assert "timed out after the configured timeout" in _llm_error_message(
        timeout_error,
        default="fallback",
    )


def test_http_status_code_reads_response_and_direct_status_code():
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(408, request=request)
    http_error = httpx.HTTPStatusError("timeout", request=request, response=response)

    assert _http_status_code(http_error) == 408
    assert _http_status_code(SimpleNamespace(status_code=503)) == 503


def test_should_retry_model_error_handles_http_status_and_transport_errors():
    request = httpx.Request("POST", "https://example.com")
    retryable_response = httpx.Response(429, request=request)
    server_response = httpx.Response(500, request=request)

    assert (
        _should_retry_model_error(
            httpx.HTTPStatusError("rate limit", request=request, response=retryable_response)
        )
        is True
    )
    assert (
        _should_retry_model_error(
            httpx.HTTPStatusError("server", request=request, response=server_response)
        )
        is True
    )
    assert _should_retry_model_error(ConnectionError("network")) is True


def test_qdrant_retrieval_mode_returns_dense(tmp_path):
    settings = make_settings(tmp_path)

    assert _qdrant_retrieval_mode(settings) is RetrievalMode.DENSE


def test_qdrant_retrieval_mode_returns_hybrid(tmp_path):
    settings = AgenticRagSettings(
        **{**make_settings(tmp_path).__dict__, "retrieval_mode": "hybrid"}
    )

    assert _qdrant_retrieval_mode(settings) is RetrievalMode.HYBRID


def test_qdrant_retrieval_mode_rejects_unknown_values():
    with pytest.raises(ConfigurationError, match="Unsupported retrieval_mode 'sparse'"):
        _qdrant_retrieval_mode(SimpleNamespace(retrieval_mode="sparse"))


def test_health_reports_configuration_problems(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    monkeypatch.setattr(
        "agentic_rag.service.resolve_chat_config",
        lambda _settings: (_ for _ in ()).throw(ConfigurationError("missing credentials")),
    )

    health = service.health()

    assert health.ok is False
    assert health.detail == "missing credentials"


def test_health_returns_ok_true_when_config_valid(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    monkeypatch.setattr(
        "agentic_rag.service.resolve_chat_config",
        lambda _settings: FakeConfig(provider="google", model="gemini-2.5-flash"),
    )
    monkeypatch.setattr(
        "agentic_rag.service.resolve_embedding_config",
        lambda _settings, _chat_config: FakeConfig(
            provider="google", model="gemini-embedding-2-preview"
        ),
    )

    health = service.health()

    assert health.ok is True
    assert health.chat_provider == "google"


def test_health_accepts_hybrid_retrieval_mode(monkeypatch, tmp_path):
    settings = AgenticRagSettings(
        **{**make_settings(tmp_path).__dict__, "retrieval_mode": "hybrid"}
    )
    service = AgenticRagService(settings)

    monkeypatch.setattr(
        "agentic_rag.service.resolve_chat_config",
        lambda _settings: FakeConfig(provider="google", model="gemini-2.5-flash"),
    )
    monkeypatch.setattr(
        "agentic_rag.service.resolve_embedding_config",
        lambda _settings, _chat_config: FakeConfig(
            provider="google", model="gemini-embedding-2-preview"
        ),
    )

    health = service.health()

    assert health.ok is True
    assert health.detail is None
    assert health.retrieval_mode == "hybrid"


def test_index_fingerprint_differs_between_dense_and_hybrid(tmp_path):
    dense_settings = make_settings(tmp_path)
    hybrid_settings = AgenticRagSettings(
        **{**make_settings(tmp_path).__dict__, "retrieval_mode": "hybrid"}
    )

    assert index_fingerprint(dense_settings) != index_fingerprint(hybrid_settings)


def test_index_fingerprint_differs_between_collections(tmp_path):
    default_settings = make_settings(tmp_path)
    alternate_settings = AgenticRagSettings(
        **{**make_settings(tmp_path).__dict__, "collection_name": "research-notes"}
    )

    assert index_fingerprint(default_settings) != index_fingerprint(alternate_settings)


def test_ingest_wraps_document_loader_failures(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    monkeypatch.setattr(
        "agentic_rag.service.resolve_chat_config",
        lambda _settings: FakeConfig(provider="google", model="gemini-2.5-flash"),
    )
    monkeypatch.setattr(
        "agentic_rag.service.resolve_embedding_config",
        lambda _settings, _chat_config: FakeConfig(
            provider="google", model="gemini-embedding-2-preview"
        ),
    )
    monkeypatch.setattr("agentic_rag.service.create_embeddings", lambda _config: object())
    monkeypatch.setattr(
        "agentic_rag.service.preprocess_documents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
    )

    with pytest.raises(DocumentLoadError):
        service.ingest()


def test_build_vectorstore_wraps_document_loader_failures(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        "agentic_rag.service.preprocess_documents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
    )

    with pytest.raises(DocumentLoadError):
        _build_vectorstore(
            settings,
            cache_path=index_cache_path(settings),
            embeddings=FakeEmbeddings(),
            retrieval_mode=RetrievalMode.DENSE,
            sparse_embeddings=None,
        )


def test_build_vectorstore_preserves_typed_document_load_errors(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        "agentic_rag.service.preprocess_documents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DocumentLoadError("typed failure")),
    )

    with pytest.raises(DocumentLoadError, match="typed failure"):
        _build_vectorstore(
            settings,
            cache_path=index_cache_path(settings),
            embeddings=FakeEmbeddings(),
            retrieval_mode=RetrievalMode.DENSE,
            sparse_embeddings=None,
        )


def test_load_or_create_vectorstore_builds_and_reuses_persisted_qdrant(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    embeddings = FakeEmbeddings()
    calls: list[str] = []

    def fake_preprocess(*_args, **_kwargs):
        calls.append("preprocess")
        return make_documents()

    monkeypatch.setattr("agentic_rag.service.preprocess_documents", fake_preprocess)

    vectorstore, document_count = load_or_create_vectorstore(settings, embeddings=embeddings)
    try:
        assert document_count == 2
        assert index_cache_path(settings).exists()
        assert _collection_document_count(vectorstore.client) == 2
    finally:
        _close_qdrant_client(vectorstore.client)

    second_vectorstore, second_document_count = load_or_create_vectorstore(
        settings,
        embeddings=embeddings,
    )
    try:
        assert second_document_count == 2
    finally:
        _close_qdrant_client(second_vectorstore.client)

    assert calls == ["preprocess"]


def test_load_or_create_vectorstore_rebuilds_incomplete_cache(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    cache_path = index_cache_path(settings)
    cache_path.mkdir(parents=True)
    calls: list[str] = []

    def fake_preprocess(*_args, **_kwargs):
        calls.append("preprocess")
        return make_documents()

    monkeypatch.setattr("agentic_rag.service.preprocess_documents", fake_preprocess)

    vectorstore, document_count = load_or_create_vectorstore(
        settings,
        embeddings=FakeEmbeddings(),
        create_if_missing=False,
    )
    try:
        assert document_count == 2
        assert cache_path.is_dir() is True
    finally:
        _close_qdrant_client(vectorstore.client)

    assert calls == ["preprocess"]


def test_load_or_create_vectorstore_rebuilds_corrupted_cache(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    cache_path = index_cache_path(settings)
    cache_path.write_text("corrupt", encoding="utf-8")
    calls: list[str] = []

    def fail_load(*_args, **_kwargs):
        raise ValueError("bad cache")

    def fake_preprocess(*_args, **_kwargs):
        calls.append("preprocess")
        return make_documents()

    monkeypatch.setattr("agentic_rag.service._load_existing_vectorstore", fail_load)
    monkeypatch.setattr("agentic_rag.service.preprocess_documents", fake_preprocess)

    vectorstore, document_count = load_or_create_vectorstore(
        settings,
        embeddings=FakeEmbeddings(),
        create_if_missing=False,
    )
    try:
        assert document_count == 2
        assert cache_path.is_dir() is True
    finally:
        _close_qdrant_client(vectorstore.client)

    assert calls == ["preprocess"]


def test_load_or_create_vectorstore_raises_when_corrupt_cache_cannot_be_removed(
    monkeypatch, tmp_path
):
    settings = make_settings(tmp_path)
    cache_path = index_cache_path(settings)
    cache_path.write_text("corrupt", encoding="utf-8")

    def fail_load(*_args, **_kwargs):
        raise ValueError("bad cache")

    def fail_remove(_cache_path):
        raise OSError("permission denied")

    monkeypatch.setattr("agentic_rag.service._load_existing_vectorstore", fail_load)
    monkeypatch.setattr("agentic_rag.service._remove_cache_path", fail_remove)

    with pytest.raises(VectorStoreError, match="Failed to remove the invalid Qdrant index"):
        load_or_create_vectorstore(settings, embeddings=FakeEmbeddings())


def test_load_or_create_vectorstore_wraps_build_errors(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        "agentic_rag.service.preprocess_documents", lambda *args, **kwargs: make_documents()
    )

    def fail_build(*args, **kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr("agentic_rag.service.QdrantVectorStore.from_documents", fail_build)

    with pytest.raises(VectorStoreError, match="Failed to build or persist the Qdrant index"):
        load_or_create_vectorstore(settings, embeddings=FakeEmbeddings())


def test_load_or_create_vectorstore_explicit_mode_requires_existing_index(tmp_path):
    settings = make_settings(tmp_path)

    with pytest.raises(VectorStoreError, match="Run ingest\\(\\) first"):
        load_or_create_vectorstore(
            settings,
            embeddings=FakeEmbeddings(),
            create_if_missing=False,
        )


def test_load_or_create_vectorstore_builds_hybrid_index(monkeypatch, tmp_path):
    settings = AgenticRagSettings(
        **{**make_settings(tmp_path).__dict__, "retrieval_mode": "hybrid"}
    )

    monkeypatch.setattr(
        "agentic_rag.service.preprocess_documents", lambda *args, **kwargs: make_documents()
    )

    vectorstore, document_count = load_or_create_vectorstore(settings, embeddings=FakeEmbeddings())
    try:
        assert document_count == 2
        assert vectorstore.retrieval_mode is RetrievalMode.HYBRID
        assert vectorstore.sparse_embeddings is not None
    finally:
        _close_qdrant_client(vectorstore.client)


def test_query_empty_question_raises_configuration_error(tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    with pytest.raises(ConfigurationError):
        service.query("   ")


def test_ingest_returns_index_status_and_closes_client(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_resolve_provider_configs",
        lambda self: (
            FakeConfig(provider="google", model="gemini-2.5-flash"),
            FakeConfig(provider="google", model="gemini-embedding-2-preview"),
        ),
    )
    monkeypatch.setattr(
        AgenticRagService, "_create_embeddings", lambda self, _config: FakeEmbeddings()
    )
    monkeypatch.setattr(
        "agentic_rag.service.load_or_create_vectorstore",
        lambda _settings, *, embeddings: (fake_vectorstore, 4),
    )

    status = service.ingest()

    assert status == IndexStatus(
        cache_path=index_cache_path(service.settings),
        fingerprint=status.fingerprint,
        exists=True,
        document_count=4,
    )
    assert fake_vectorstore.client.closed is True


def test_ingest_does_not_reopen_index_status_while_qdrant_is_open(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_resolve_provider_configs",
        lambda self: (
            FakeConfig(provider="google", model="gemini-2.5-flash"),
            FakeConfig(provider="google", model="gemini-embedding-2-preview"),
        ),
    )
    monkeypatch.setattr(
        AgenticRagService, "_create_embeddings", lambda self, _config: FakeEmbeddings()
    )
    monkeypatch.setattr(
        "agentic_rag.service.load_or_create_vectorstore",
        lambda _settings, *, embeddings: (fake_vectorstore, 4),
    )
    monkeypatch.setattr(
        AgenticRagService,
        "index_status",
        lambda self: (_ for _ in ()).throw(AssertionError("index_status should not be called")),
    )

    status = service.ingest()

    assert status.exists is True
    assert status.document_count == 4
    assert fake_vectorstore.client.closed is True


def test_query_returns_structured_result_and_closes_vectorstore(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (FakeGraph(message=AIMessage(content="Final answer")), 2, fake_vectorstore),
    )
    monkeypatch.setattr(
        AgenticRagService,
        "index_status",
        lambda self: (_ for _ in ()).throw(AssertionError("index_status should not be called")),
    )

    result = service.query("What is reward hacking?")

    assert result.answer_text == "Final answer"
    assert result.document_count == 2
    assert result.index_status == IndexStatus(
        cache_path=index_cache_path(service.settings),
        fingerprint=index_fingerprint(service.settings),
        exists=True,
        document_count=2,
    )
    assert result.steps[0].node_name == "generate_answer"
    assert result.termination_reason is None
    assert result.rewrites_used == 0
    assert fake_vectorstore.client.closed is True


def test_query_invokes_step_callback(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()
    seen_steps = []

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (FakeGraph(message=AIMessage(content="Final answer")), 2, fake_vectorstore),
    )

    result = service.query("What is reward hacking?", on_step=seen_steps.append)

    assert result.answer_text == "Final answer"
    assert len(seen_steps) == 1


def test_query_returns_insufficient_context_result(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()
    insufficient_message = AIMessage(
        content="I do not have enough relevant context to answer reliably.",
        additional_kwargs={"reason": "insufficient_context"},
    )

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (
            FakeGraph(
                chunks=[
                    ("rewrite_question", AIMessage(content="Rewritten question")),
                    ("insufficient_context", insufficient_message),
                ]
            ),
            2,
            fake_vectorstore,
        ),
    )

    result = service.query("What is reward hacking?")

    assert result.termination_reason == "insufficient_context"
    assert result.rewrites_used == 1
    assert result.answer_text == "I do not have enough relevant context to answer reliably."
    assert fake_vectorstore.client.closed is True


def test_query_raises_when_graph_returns_no_messages_and_closes_vectorstore(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (EmptyGraph(), 0, fake_vectorstore),
    )

    with pytest.raises(LLMError, match="did not return any messages"):
        service.query("What is reward hacking?")

    assert fake_vectorstore.client.closed is True


def test_query_wraps_graph_failures_and_closes_vectorstore(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (FakeGraph(error=RuntimeError("graph failed")), 2, fake_vectorstore),
    )

    with pytest.raises(LLMError, match="Failed while running the query graph"):
        service.query("What is reward hacking?")

    assert fake_vectorstore.client.closed is True


def test_query_surfaces_401_with_actionable_message(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(401, request=request)
    error = httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (FakeGraph(error=error), 2, fake_vectorstore),
    )

    with pytest.raises(LLMError, match="401 Unauthorized"):
        service.query("What is reward hacking?")

    assert fake_vectorstore.client.closed is True


def test_query_surfaces_429_with_actionable_message(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request)
    error = httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (FakeGraph(error=error), 2, fake_vectorstore),
    )

    with pytest.raises(LLMError, match="429 Too Many Requests"):
        service.query("What is reward hacking?")

    assert fake_vectorstore.client.closed is True


def test_query_surfaces_transient_protocol_errors(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (
            FakeGraph(error=httpx.RemoteProtocolError("server disconnected")),
            2,
            fake_vectorstore,
        ),
    )

    with pytest.raises(LLMError, match="transient network/protocol error"):
        service.query("What is reward hacking?")

    assert fake_vectorstore.client.closed is True


def test_query_explicit_mode_requires_prior_ingest(monkeypatch, tmp_path):
    settings = AgenticRagSettings(
        **{**make_settings(tmp_path).__dict__, "ingestion_mode": "explicit"}
    )
    service = AgenticRagService(settings)

    monkeypatch.setattr(
        AgenticRagService,
        "_resolve_provider_configs",
        lambda self: (
            FakeConfig(provider="google", model="gemini-2.5-flash"),
            FakeConfig(provider="google", model="gemini-embedding-2-preview"),
        ),
    )
    monkeypatch.setattr(
        AgenticRagService,
        "_create_embeddings",
        lambda self, _config: FakeEmbeddings(),
    )

    with pytest.raises(VectorStoreError, match="ingestion_mode='auto'"):
        service.query("What is reward hacking?")


def test_retrying_chat_model_retries_transient_errors(monkeypatch):
    calls: list[str] = []
    sleep_calls: list[float] = []

    class FlakyModel:
        def invoke(self, _messages):
            calls.append("invoke")
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("temporary disconnect")
            return AIMessage(content="Recovered answer")

    monkeypatch.setattr(
        "agentic_rag.service.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    model = RetryingChatModel(model=FlakyModel())

    result = model.invoke([{"role": "user", "content": "What is reward hacking?"}])

    assert result.content == "Recovered answer"
    assert calls == ["invoke", "invoke"]
    assert sleep_calls == [DEFAULT_MODEL_RETRY_BACKOFF_SECONDS]


def test_retrying_chat_model_does_not_retry_non_transient_errors(monkeypatch):
    sleep_calls: list[float] = []

    class BrokenModel:
        def invoke(self, _messages):
            raise RuntimeError("bad prompt")

    monkeypatch.setattr(
        "agentic_rag.service.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    model = RetryingChatModel(model=BrokenModel())

    with pytest.raises(RuntimeError, match="bad prompt"):
        model.invoke([{"role": "user", "content": "What is reward hacking?"}])

    assert sleep_calls == []


def test_retrying_chat_model_raises_internal_error_when_max_attempts_is_zero():
    model = RetryingChatModel(model=object(), max_attempts=0)

    with pytest.raises(LLMError, match="without preserving the root cause"):
        model.invoke([])


def test_retrying_embeddings_retries_transient_errors(monkeypatch):
    calls: list[str] = []
    sleep_calls: list[float] = []

    class FlakyEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append("embed_documents")
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("temporary disconnect")
            return [[float(len(texts[0]))]]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    monkeypatch.setattr(
        "agentic_rag.service.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    embeddings = RetryingEmbeddings(embeddings=FlakyEmbeddings())

    result = embeddings.embed_query("reward hacking")

    assert result == [14.0]
    assert calls == ["embed_documents", "embed_documents"]
    assert sleep_calls == [DEFAULT_MODEL_RETRY_BACKOFF_SECONDS]


def test_retrying_embeddings_does_not_retry_non_transient_errors(monkeypatch):
    sleep_calls: list[float] = []

    class BrokenEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("bad embeddings request")

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    monkeypatch.setattr(
        "agentic_rag.service.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    embeddings = RetryingEmbeddings(embeddings=BrokenEmbeddings())

    with pytest.raises(RuntimeError, match="bad embeddings request"):
        embeddings.embed_query("reward hacking")

    assert sleep_calls == []


def test_retrying_embeddings_raises_internal_error_when_max_attempts_is_zero():
    embeddings = RetryingEmbeddings(embeddings=FakeEmbeddings(), max_attempts=0)

    with pytest.raises(LLMError, match="without preserving the root cause"):
        embeddings.embed_query("reward hacking")


def test_diagram_stubs_return_minimal_values():
    assert _DiagramChatModel().invoke([]).content == ""
    assert _DiagramRetriever().invoke("query") == []


def test_render_graph_mermaid_returns_static_mermaid_text(tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    mermaid = service.render_graph_mermaid()

    assert mermaid.startswith("---\nconfig:")
    assert "prepare_retrieval" in mermaid
    assert "insufficient_context" in mermaid


def test_render_graph_mermaid_does_not_require_index_or_provider_setup(monkeypatch, tmp_path):
    service = AgenticRagService(
        AgenticRagSettings(**{**make_settings(tmp_path).__dict__, "ingestion_mode": "explicit"})
    )

    monkeypatch.setattr(
        AgenticRagService,
        "_build_query_graph",
        lambda self: (_ for _ in ()).throw(AssertionError("_build_query_graph should not run")),
    )

    mermaid = service.render_graph_mermaid()

    assert mermaid.startswith("---\nconfig:")
    assert "prepare_retrieval" in mermaid


def test_render_graph_mermaid_wraps_draw_errors(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    broken_graph = BrokenDrawGraph()
    monkeypatch.setattr(
        "agentic_rag.service.build_agentic_rag_graph",
        lambda **kwargs: broken_graph,
    )

    with pytest.raises(LLMError, match="render the graph diagram"):
        service.render_graph_mermaid()


def test_export_graph_mermaid_writes_file(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    output_path = Path(tmp_path) / "graph.md"

    monkeypatch.setattr(AgenticRagService, "render_graph_mermaid", lambda self: "graph TD")

    service.export_graph_mermaid(output_path)

    assert output_path.read_text(encoding="utf-8") == "graph TD"


def test_build_query_graph_returns_graph_document_count_and_vectorstore(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()
    sentinel_retriever = object()
    sentinel_tool = object()
    sentinel_graph = object()
    response_model = object()
    grader_model = object()
    seen_kwargs: dict[str, Any] = {}
    seen_retriever = []
    models = [response_model, grader_model]

    monkeypatch.setattr(
        AgenticRagService,
        "_resolve_provider_configs",
        lambda self: (
            FakeConfig(provider="google", model="gemini-2.5-flash"),
            FakeConfig(provider="google", model="gemini-embedding-2-preview"),
        ),
    )
    monkeypatch.setattr(
        AgenticRagService, "_create_embeddings", lambda self, _config: FakeEmbeddings()
    )
    monkeypatch.setattr(
        "agentic_rag.service.load_or_create_vectorstore",
        lambda _settings, *, embeddings, create_if_missing: (fake_vectorstore, 2),
    )
    monkeypatch.setattr(
        "agentic_rag.service.create_retriever_tool",
        lambda retriever: seen_retriever.append(retriever) or sentinel_tool,
    )
    monkeypatch.setattr(
        AgenticRagService,
        "_create_chat_model",
        lambda self, _config: models.pop(0),
    )

    def fake_build_graph(**kwargs):
        seen_kwargs.update(kwargs)
        return sentinel_graph

    def fake_as_retriever(*, search_kwargs):
        fake_vectorstore.retriever_calls.append(search_kwargs)
        return sentinel_retriever

    fake_vectorstore.as_retriever = fake_as_retriever
    monkeypatch.setattr("agentic_rag.service.build_agentic_rag_graph", fake_build_graph)

    graph, document_count, vectorstore = service._build_query_graph()

    assert graph is sentinel_graph
    assert document_count == 2
    assert vectorstore is fake_vectorstore
    assert fake_vectorstore.retriever_calls == [
        {"k": retrieval_fetch_k(service.settings.retrieval_k)}
    ]
    assert len(seen_retriever) == 1
    assert seen_retriever[0].base_retriever is sentinel_retriever
    assert seen_retriever[0].final_k == service.settings.retrieval_k
    assert seen_kwargs == {
        "response_model": response_model,
        "grader_model": grader_model,
        "retriever_tool": sentinel_tool,
        "max_rewrites": service.settings.max_rewrites,
    }


def test_build_query_graph_closes_vectorstore_on_model_errors(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_resolve_provider_configs",
        lambda self: (
            FakeConfig(provider="google", model="gemini-2.5-flash"),
            FakeConfig(provider="google", model="gemini-embedding-2-preview"),
        ),
    )
    monkeypatch.setattr(
        AgenticRagService, "_create_embeddings", lambda self, _config: FakeEmbeddings()
    )
    monkeypatch.setattr(
        "agentic_rag.service.load_or_create_vectorstore",
        lambda _settings, *, embeddings, create_if_missing: (fake_vectorstore, 2),
    )
    monkeypatch.setattr(
        AgenticRagService,
        "_create_chat_model",
        lambda self, _config: (_ for _ in ()).throw(LLMError("chat failed")),
    )

    with pytest.raises(LLMError, match="chat failed"):
        service._build_query_graph()

    assert fake_vectorstore.client.closed is True


def test_build_query_graph_wraps_graph_construction_errors_and_closes_vectorstore(
    monkeypatch, tmp_path
):
    service = AgenticRagService(make_settings(tmp_path))
    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        AgenticRagService,
        "_resolve_provider_configs",
        lambda self: (
            FakeConfig(provider="google", model="gemini-2.5-flash"),
            FakeConfig(provider="google", model="gemini-embedding-2-preview"),
        ),
    )
    monkeypatch.setattr(
        AgenticRagService, "_create_embeddings", lambda self, _config: FakeEmbeddings()
    )
    monkeypatch.setattr(
        "agentic_rag.service.load_or_create_vectorstore",
        lambda _settings, *, embeddings, create_if_missing: (fake_vectorstore, 2),
    )
    monkeypatch.setattr(AgenticRagService, "_create_chat_model", lambda self, _config: object())

    def fail_build_graph(**kwargs):
        raise RuntimeError("graph build failed")

    monkeypatch.setattr("agentic_rag.service.build_agentic_rag_graph", fail_build_graph)

    with pytest.raises(LLMError, match="build the query graph"):
        service._build_query_graph()

    assert fake_vectorstore.client.closed is True


def test_create_chat_model_wraps_factory_errors(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    def fail_create(_config):
        raise RuntimeError("chat failed")

    monkeypatch.setattr("agentic_rag.service.create_chat_model", fail_create)

    with pytest.raises(LLMError, match="create the chat model"):
        service._create_chat_model(FakeConfig(provider="google", model="gemini-2.5-flash"))


def test_create_chat_model_wraps_model_in_retrying_chat_model(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))
    sentinel_model = object()

    monkeypatch.setattr("agentic_rag.service.create_chat_model", lambda _config: sentinel_model)

    wrapped = service._create_chat_model(FakeConfig(provider="google", model="gemini-2.5-flash"))

    assert isinstance(wrapped, RetryingChatModel)
    assert wrapped.model is sentinel_model


def test_create_embeddings_wraps_factory_errors(monkeypatch, tmp_path):
    service = AgenticRagService(make_settings(tmp_path))

    def fail_create(_config):
        raise RuntimeError("embed failed")

    monkeypatch.setattr("agentic_rag.service.create_embeddings", fail_create)

    with pytest.raises(LLMError, match="create the embeddings client"):
        service._create_embeddings(
            FakeConfig(provider="google", model="gemini-embedding-2-preview")
        )
