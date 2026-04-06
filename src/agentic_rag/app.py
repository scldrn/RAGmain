from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.vectorstores import InMemoryVectorStore

from agentic_rag.documents import preprocess_documents
from agentic_rag.graph import (
    build_agentic_rag_graph,
    build_vectorstore,
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
class AgenticRagApp:
    settings: AgenticRagSettings
    graph: Any
    document_count: int


def index_cache_path(
    settings: AgenticRagSettings,
    *,
    embedding_provider: str,
    embedding_model: str,
) -> Path:
    """Compute the cache path for the current source/index configuration."""
    payload = {
        "source_urls": list(settings.source_urls),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_api_base": settings.embedding_api_base,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return Path(settings.index_cache_dir) / f"{digest}.json"


def load_or_create_vectorstore(
    settings: AgenticRagSettings,
    *,
    embedding_provider: str,
    embedding_model: str,
    embeddings,
) -> tuple[InMemoryVectorStore, int]:
    """Load a cached vector store when possible, otherwise build and cache it."""
    cache_path = index_cache_path(
        settings,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )

    if cache_path.exists():
        vectorstore = InMemoryVectorStore.load(str(cache_path), embedding=embeddings)
        return vectorstore, len(getattr(vectorstore, "store", {}))

    documents = preprocess_documents(
        settings.source_urls,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vectorstore = build_vectorstore(documents, embeddings=embeddings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.dump(str(cache_path))
    return vectorstore, len(documents)


def create_agentic_rag_app(settings: Optional[AgenticRagSettings] = None) -> AgenticRagApp:
    """Build the full Agentic RAG application from the configured sources."""
    settings = settings or AgenticRagSettings()
    chat_config = resolve_chat_config(settings)
    embedding_config = resolve_embedding_config(settings, chat_config)

    embeddings = create_embeddings(embedding_config)
    vectorstore, document_count = load_or_create_vectorstore(
        settings,
        embedding_provider=embedding_config.provider,
        embedding_model=embedding_config.model,
        embeddings=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
    retriever_tool = create_retriever_tool(retriever)
    response_model = create_chat_model(chat_config)
    grader_model = create_chat_model(chat_config)
    graph = build_agentic_rag_graph(
        response_model=response_model,
        grader_model=grader_model,
        retriever_tool=retriever_tool,
    )

    return AgenticRagApp(
        settings=settings,
        graph=graph,
        document_count=document_count,
    )


def run_question(app: AgenticRagApp, question: str, *, show_steps: bool = False) -> BaseMessage:
    """Run a user question through the graph and return the last message."""
    final_message = None

    for chunk in app.graph.stream({"messages": [{"role": "user", "content": question}]}):
        for node_name, update in chunk.items():
            message = update["messages"][-1]
            final_message = message
            if show_steps:
                print(f"\n[{node_name}]")
                tool_calls = getattr(message, "tool_calls", None) or []
                for tool_call in tool_calls:
                    print(f"tool: {tool_call['name']} args={tool_call['args']}")
                text = message_text(message).strip()
                if text:
                    print(text)

    if final_message is None:
        raise RuntimeError("The graph did not return any messages.")

    return final_message


def export_graph_mermaid(app: AgenticRagApp, output_path: str) -> None:
    """Write the graph diagram as Mermaid text."""
    mermaid = app.graph.get_graph().draw_mermaid()
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(mermaid)
