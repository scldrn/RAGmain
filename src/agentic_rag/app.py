from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import BaseMessage

from agentic_rag.graph import message_text
from agentic_rag.service import AgenticRagService, QueryStep
from agentic_rag.settings import AgenticRagSettings


@dataclass(frozen=True)
class AgenticRagApp:
    settings: AgenticRagSettings
    service: AgenticRagService
    document_count: int


def create_agentic_rag_app(settings: Optional[AgenticRagSettings] = None) -> AgenticRagApp:
    """Compatibility wrapper that prepares the shared service layer eagerly."""
    settings = settings or AgenticRagSettings()
    service = AgenticRagService(settings=settings)
    index_status = service.ingest()

    return AgenticRagApp(
        settings=settings,
        service=service,
        document_count=index_status.document_count or 0,
    )


def run_question(app: AgenticRagApp, question: str, *, show_steps: bool = False) -> BaseMessage:
    """Compatibility wrapper around the service-layer query interface."""

    def print_step(step: QueryStep) -> None:
        print(f"\n[{step.node_name}]")
        tool_calls = getattr(step.message, "tool_calls", None) or []
        for tool_call in tool_calls:
            print(f"tool: {tool_call['name']} args={tool_call['args']}")
        text = message_text(step.message).strip()
        if text:
            print(text)

    result = app.service.query(question, on_step=print_step if show_steps else None)
    return result.final_message


def export_graph_mermaid(app: AgenticRagApp, output_path: str) -> None:
    """Write the graph diagram using the shared service layer."""
    app.service.export_graph_mermaid(output_path)
