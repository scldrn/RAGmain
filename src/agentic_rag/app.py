from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.messages import BaseMessage

from agentic_rag.documents import preprocess_documents
from agentic_rag.graph import (
    build_agentic_rag_graph,
    build_retriever,
    create_chat_model,
    create_retriever_tool,
    message_text,
)
from agentic_rag.settings import AgenticRagSettings


@dataclass(frozen=True)
class AgenticRagApp:
    settings: AgenticRagSettings
    graph: Any
    document_count: int


def ensure_google_api_key() -> None:
    """Fail early when the Google AI Studio API key is missing."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not (google_api_key or gemini_api_key):
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is not set. Export one of them before running the agentic RAG CLI."
        )

    if not google_api_key and gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key


def create_agentic_rag_app(settings: Optional[AgenticRagSettings] = None) -> AgenticRagApp:
    """Build the full Agentic RAG application from the configured sources."""
    settings = settings or AgenticRagSettings()
    ensure_google_api_key()

    documents = preprocess_documents(
        settings.source_urls,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever = build_retriever(
        documents,
        embedding_model=settings.embedding_model,
        retrieval_k=settings.retrieval_k,
    )
    retriever_tool = create_retriever_tool(retriever)
    response_model = create_chat_model(settings.chat_model)
    grader_model = create_chat_model(settings.chat_model)
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
