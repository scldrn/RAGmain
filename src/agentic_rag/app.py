from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import BaseMessage

from agentic_rag.presentation import format_query_step
from agentic_rag.service import AgenticRagService, QueryStep
from agentic_rag.settings import AgenticRagSettings


@dataclass(frozen=True)
class AgenticRagApp:
    settings: AgenticRagSettings
    service: AgenticRagService
    document_count: int


def create_agentic_rag_app(settings: Optional[AgenticRagSettings] = None) -> AgenticRagApp:
    """Compatibility wrapper that prepares the shared service layer."""
    settings = settings or AgenticRagSettings()
    service = AgenticRagService(settings=settings)
    if settings.ingestion_mode == "auto":
        index_status = service.ingest()
    else:
        index_status = service.index_status()

    return AgenticRagApp(
        settings=settings,
        service=service,
        document_count=index_status.document_count or 0,
    )


def run_question(app: AgenticRagApp, question: str, *, show_steps: bool = False) -> BaseMessage:
    """Compatibility wrapper around the service-layer query interface."""

    def print_step(step: QueryStep) -> None:
        print()
        print(format_query_step(step))

    result = app.service.query(question, on_step=print_step if show_steps else None)
    return result.final_message


def export_graph_mermaid(app: AgenticRagApp, output_path: str) -> None:
    """Write the graph diagram using the shared service layer."""
    app.service.export_graph_mermaid(output_path)
