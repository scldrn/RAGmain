from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.messages import BaseMessage

from agentic_rag.documents import preprocess_documents
from agentic_rag.graph import (
    build_agentic_rag_graph,
    build_retriever,
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


def create_agentic_rag_app(settings: Optional[AgenticRagSettings] = None) -> AgenticRagApp:
    """Build the full Agentic RAG application from the configured sources."""
    settings = settings or AgenticRagSettings()
    chat_config = resolve_chat_config(settings)
    embedding_config = resolve_embedding_config(settings, chat_config)

    documents = preprocess_documents(
        settings.source_urls,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = create_embeddings(embedding_config)
    retriever = build_retriever(
        documents,
        embeddings=embeddings,
        retrieval_k=settings.retrieval_k,
    )
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
        document_count=len(documents),
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
