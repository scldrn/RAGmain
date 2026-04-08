from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from agentic_rag.presentation import format_index_status, format_query_step
from agentic_rag.service import IndexStatus, QueryStep


def test_format_query_step_includes_tool_calls_and_message_text():
    step = QueryStep(
        node_name="retrieve",
        message=AIMessage(
            content="retrieved context",
            tool_calls=[
                {
                    "id": "tool-1",
                    "name": "retrieve_blog_posts",
                    "args": {"query": "reward hacking"},
                }
            ],
        ),
    )

    rendered = format_query_step(step)

    assert rendered == (
        "[retrieve]\ntool: retrieve_blog_posts args={'query': 'reward hacking'}\nretrieved context"
    )


def test_format_query_step_omits_empty_message_text():
    step = QueryStep(node_name="generate_answer", message=AIMessage(content="   "))

    assert format_query_step(step) == "[generate_answer]"


def test_format_index_status_renders_expected_lines(tmp_path):
    status = IndexStatus(
        cache_path=Path(tmp_path) / "index",
        fingerprint="abc123",
        exists=True,
        document_count=7,
    )

    rendered = format_index_status(status)

    assert rendered == (
        f"Index ready:\n\ndocuments: 7\ncache: {status.cache_path}\nfingerprint: abc123"
    )
