from __future__ import annotations

from agentic_rag.graph import message_text
from agentic_rag.service import IndexStatus, QueryStep


def format_query_step(step: QueryStep) -> str:
    """Render a streamed query step for console output."""
    lines = [f"[{step.node_name}]"]
    tool_calls = getattr(step.message, "tool_calls", None) or []
    for tool_call in tool_calls:
        lines.append(f"tool: {tool_call['name']} args={tool_call['args']}")
    text = message_text(step.message).strip()
    if text:
        lines.append(text)
    return "\n".join(lines)


def format_index_status(status: IndexStatus) -> str:
    """Render the current index status for console output."""
    return "\n".join(
        (
            "Index ready:",
            "",
            f"documents: {status.document_count or 0}",
            f"cache: {status.cache_path}",
            f"fingerprint: {status.fingerprint}",
        )
    )
