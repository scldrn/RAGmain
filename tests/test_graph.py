from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentic_rag.graph import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    INSUFFICIENT_CONTEXT_REASON,
    _context_has_no_relevant_documents,
    _normalize_rewritten_question,
    _parse_grade_documents_response,
    build_agentic_rag_graph,
    create_retriever_tool,
    current_retrieval_query,
    format_retrieved_documents,
    make_generate_answer,
    make_generate_insufficient_context,
    make_grade_documents,
    make_prepare_retrieval,
    make_rewrite_question,
    message_text,
    rewrite_count,
)


class FakeResponseModel:
    def __init__(self, *, text_response: str = ""):
        self.text_response = text_response

    def invoke(self, _messages):
        return AIMessage(content=self.text_response)


class FakeSequencedResponseModel:
    def __init__(self, *, invoke_messages: list[str]):
        self.invoke_messages = list(invoke_messages)

    def invoke(self, _messages):
        return AIMessage(content=self.invoke_messages.pop(0))


class FakeGraderModel:
    def __init__(self, score: str):
        self.score = score

    def invoke(self, _messages):
        return AIMessage(content=f'{{"binary_score":"{self.score}"}}')


class DummyMessage:
    def __init__(self, content):
        self.content = content


def test_format_retrieved_documents_includes_source_labels():
    documents = [
        Document(
            page_content="Reward hacking appears when the reward signal is exploitable.",
            metadata={"source": "https://example.com/reward-hacking", "title": "Reward Hacking"},
        )
    ]

    result = format_retrieved_documents(documents)

    assert "[Source 1] Reward Hacking" in result
    assert "https://example.com/reward-hacking" in result


def test_message_text_handles_content_list():
    message = DummyMessage([{"text": "alpha"}, {"ignored": "x"}, 42])

    assert message_text(message) == "alpha\n42"


def test_message_text_handles_non_string_non_list_content():
    message = DummyMessage({"key": "value"})

    assert message_text(message) == "{'key': 'value'}"


def test_format_retrieved_documents_handles_empty_input():
    assert format_retrieved_documents([]) == "No relevant documents were found."


def test_context_has_no_relevant_documents_detects_empty_retrieval_message():
    assert _context_has_no_relevant_documents("No relevant documents were found.") is True
    assert _context_has_no_relevant_documents("[Source 1] Useful context") is False


def test_current_retrieval_query_prefers_latest_human_message():
    state = {
        "messages": [
            HumanMessage(content="Original question"),
            AIMessage(content="Thinking"),
            HumanMessage(content="Rewritten question"),
        ]
    }

    assert current_retrieval_query(state) == "Rewritten question"


def test_current_retrieval_query_falls_back_to_first_message_when_no_human_message_exists():
    state = {"messages": [AIMessage(content="Fallback question")]}

    assert current_retrieval_query(state) == "Fallback question"


def test_normalize_rewritten_question_extracts_concise_query():
    text = """
    A clearer version would be:

    **"In the reward hacking experiments, what training budget was used?"**

    Why this works:
    - It is more specific.
    """.strip()

    assert _normalize_rewritten_question(text) == (
        "In the reward hacking experiments, what training budget was used?"
    )


def test_prepare_retrieval_emits_tool_call_for_latest_question():
    retriever_tool = MagicMock(name="retrieve_blog_posts")
    retriever_tool.name = "retrieve_blog_posts"
    prepare_retrieval = make_prepare_retrieval(retriever_tool)

    result = prepare_retrieval(
        {
            "messages": [
                HumanMessage(content="Original question"),
                HumanMessage(content="Improved question"),
            ],
            "rewrite_count": 1,
        }
    )
    message = result["messages"][-1]

    assert message.tool_calls == [
        {
            "id": "retrieve-1",
            "name": "retrieve_blog_posts",
            "args": {"query": "Improved question"},
            "type": "tool_call",
        }
    ]


def test_parse_grade_documents_response_accepts_json_and_text_fallback():
    assert (
        _parse_grade_documents_response(AIMessage(content='{"binary_score":"yes"}')).binary_score
        == "yes"
    )
    assert _parse_grade_documents_response(AIMessage(content="no")).binary_score == "no"
    assert (
        _parse_grade_documents_response(AIMessage(content="yes, definitely")).binary_score == "yes"
    )
    assert (
        _parse_grade_documents_response(AIMessage(content="Answer: no (not yes)")).binary_score
        == "no"
    )
    assert (
        _parse_grade_documents_response(AIMessage(content="Answer: yes (not no)")).binary_score
        == "yes"
    )


def test_parse_grade_documents_response_raises_on_unparseable_content():
    from agentic_rag.errors import LLMError

    with pytest.raises(LLMError, match="Failed to parse the document relevance grade"):
        _parse_grade_documents_response(AIMessage(content="maybe"))


def test_grade_documents_routes_to_answer_when_context_is_relevant():
    grade_documents = make_grade_documents(FakeGraderModel("yes"), max_rewrites=3)
    state = {
        "messages": [
            HumanMessage(content="What is reward hacking?"),
            ToolMessage(content="Reward hacking is when agents exploit rewards.", tool_call_id="1"),
        ],
        "rewrite_count": 0,
    }

    assert grade_documents(state) == "generate_answer"


def test_grade_documents_uses_latest_retrieval_query_in_prompt():
    grader_model = MagicMock()
    grader_model.invoke.return_value = AIMessage(content='{"binary_score":"yes"}')
    grade_documents = make_grade_documents(grader_model, max_rewrites=3)
    state = {
        "messages": [
            HumanMessage(content="Original question"),
            HumanMessage(content="Improved retrieval query"),
            ToolMessage(content="Relevant context", tool_call_id="1"),
        ],
        "rewrite_count": 1,
    }

    assert grade_documents(state) == "generate_answer"
    prompt = grader_model.invoke.call_args.args[0][0]["content"]
    assert "Improved retrieval query" in prompt
    assert "Original question" not in prompt


def test_grade_documents_routes_to_rewrite_when_context_is_not_relevant():
    grade_documents = make_grade_documents(FakeGraderModel("no"), max_rewrites=3)
    state = {
        "messages": [
            HumanMessage(content="What is reward hacking?"),
            ToolMessage(content="Unrelated context", tool_call_id="1"),
        ],
        "rewrite_count": 0,
    }

    assert grade_documents(state) == "rewrite_question"


def test_grade_documents_returns_insufficient_context_when_rewrite_stalled():
    grade_documents = make_grade_documents(FakeGraderModel("no"), max_rewrites=3)
    state = {
        "messages": [
            HumanMessage(content="What is reward hacking?"),
            ToolMessage(content="Unrelated context", tool_call_id="1"),
        ],
        "rewrite_count": 1,
        "rewrite_stalled": True,
    }

    assert grade_documents(state) == "insufficient_context"


def test_grade_documents_routes_to_insufficient_context_when_limit_is_reached():
    grade_documents = make_grade_documents(FakeGraderModel("no"), max_rewrites=1)
    state = {
        "messages": [
            HumanMessage(content="What is reward hacking?"),
            ToolMessage(content="Unrelated context", tool_call_id="1"),
        ],
        "rewrite_count": 1,
    }

    assert grade_documents(state) == "insufficient_context"


def test_grade_documents_rewrites_without_grader_when_no_documents_found():
    grader_model = MagicMock()
    grade_documents = make_grade_documents(grader_model, max_rewrites=3)
    state = {
        "messages": [
            HumanMessage(content="What is reward hacking?"),
            ToolMessage(content="No relevant documents were found.", tool_call_id="1"),
        ],
        "rewrite_count": 0,
    }

    assert grade_documents(state) == "rewrite_question"
    grader_model.invoke.assert_not_called()


def test_grade_documents_stops_without_grader_when_no_documents_and_limit_hit():
    grader_model = MagicMock()
    grade_documents = make_grade_documents(grader_model, max_rewrites=1)
    state = {
        "messages": [
            HumanMessage(content="What is reward hacking?"),
            ToolMessage(content="No relevant documents were found.", tool_call_id="1"),
        ],
        "rewrite_count": 1,
    }

    assert grade_documents(state) == "insufficient_context"
    grader_model.invoke.assert_not_called()


def test_rewrite_question_returns_human_message():
    rewrite_question = make_rewrite_question(FakeResponseModel(text_response="Improved question"))
    state = {"messages": [HumanMessage(content="Original question")], "rewrite_count": 0}

    result = rewrite_question(state)

    assert result["messages"][-1].content == "Improved question"
    assert result["rewrite_count"] == 1
    assert result["rewrite_stalled"] is False


def test_rewrite_question_uses_latest_human_message():
    rewrite_question = make_rewrite_question(FakeResponseModel(text_response="Improved again"))
    state = {
        "messages": [
            HumanMessage(content="Original question"),
            HumanMessage(content="First rewrite"),
        ],
        "rewrite_count": 1,
    }

    result = rewrite_question(state)

    assert result["messages"][-1].content == "Improved again"
    assert result["rewrite_count"] == 2


def test_rewrite_question_normalizes_verbose_model_output():
    rewrite_question = make_rewrite_question(
        FakeResponseModel(
            text_response=(
                "A clearer version would be:\n\n"
                '**"In the reward hacking experiments, what training budget was used?"**\n\n'
                "Why this works:\n- Better retrieval."
            )
        )
    )
    state = {"messages": [HumanMessage(content="Original question")], "rewrite_count": 0}

    result = rewrite_question(state)

    assert result["messages"][-1].content == (
        "In the reward hacking experiments, what training budget was used?"
    )
    assert result["rewrite_stalled"] is False


def test_rewrite_question_marks_stalled_when_query_does_not_change():
    rewrite_question = make_rewrite_question(FakeResponseModel(text_response="Original question"))
    state = {"messages": [HumanMessage(content="Original question")], "rewrite_count": 0}

    result = rewrite_question(state)

    assert result["messages"][-1].content == "Original question"
    assert result["rewrite_stalled"] is True


def test_rewrite_count_reads_state_with_default():
    assert rewrite_count({}) == 0
    assert rewrite_count({"rewrite_count": 2}) == 2


def test_generate_answer_uses_model_output():
    generate_answer = make_generate_answer(FakeResponseModel(text_response="Final answer"))
    state = {
        "messages": [
            HumanMessage(content="Question"),
            ToolMessage(content="[Source 1] Context", tool_call_id="1"),
        ]
    }

    result = generate_answer(state)

    assert result["messages"][-1].content == "Final answer"


def test_generate_insufficient_context_returns_structured_message():
    generate_insufficient_context = make_generate_insufficient_context()

    result = generate_insufficient_context({"messages": [], "rewrite_count": 2})
    message = result["messages"][-1]

    assert message.content == INSUFFICIENT_CONTEXT_MESSAGE
    assert message.additional_kwargs["reason"] == INSUFFICIENT_CONTEXT_REASON
    assert message.additional_kwargs["answer"] is None
    assert message.additional_kwargs["rewrites_used"] == 2


def test_create_retriever_tool_invokes_retriever():
    class FakeRetriever:
        def __init__(self):
            self.queries: list[str] = []

        def invoke(self, query: str):
            self.queries.append(query)
            return [Document(page_content="context", metadata={"source": "https://example.com"})]

    retriever = FakeRetriever()
    tool_instance = create_retriever_tool(retriever)

    result = tool_instance.invoke("reward hacking")

    assert retriever.queries == ["reward hacking"]
    assert "[Source 1]" in result


def test_graph_can_run_retrieve_then_answer_path():
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Return matching context."""
        return f"[Source 1] query={query}"

    response_model = FakeResponseModel(text_response="Final answer")
    grader_model = FakeGraderModel("yes")
    graph = build_agentic_rag_graph(
        response_model=response_model,
        grader_model=grader_model,
        retriever_tool=retrieve_blog_posts,
        max_rewrites=3,
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What is reward hacking?"}], "rewrite_count": 0}
    )

    assert result["messages"][-1].content == "Final answer"


def test_graph_always_retrieves_before_answering():
    seen_queries: list[str] = []

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Return matching context."""
        seen_queries.append(query)
        return f"[Source 1] query={query}"

    graph = build_agentic_rag_graph(
        response_model=FakeResponseModel(text_response="Answer after retrieval"),
        grader_model=FakeGraderModel("yes"),
        retriever_tool=retrieve_blog_posts,
        max_rewrites=3,
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What is reward hacking?"}], "rewrite_count": 0}
    )

    assert result["messages"][-1].content == "Answer after retrieval"
    assert seen_queries == ["What is reward hacking?"]


def test_graph_returns_insufficient_context_after_max_rewrites():
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Return matching context."""
        return f"[Source 1] query={query}"

    response_model = FakeSequencedResponseModel(
        invoke_messages=["Improved question"],
    )
    graph = build_agentic_rag_graph(
        response_model=response_model,
        grader_model=FakeGraderModel("no"),
        retriever_tool=retrieve_blog_posts,
        max_rewrites=1,
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What is reward hacking?"}], "rewrite_count": 0}
    )
    final_message = result["messages"][-1]

    assert final_message.content == INSUFFICIENT_CONTEXT_MESSAGE
    assert final_message.additional_kwargs["reason"] == INSUFFICIENT_CONTEXT_REASON
    assert final_message.additional_kwargs["rewrites_used"] == 1


def test_graph_uses_rewritten_question_for_followup_retrieval():
    seen_queries: list[str] = []

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Return matching context."""
        seen_queries.append(query)
        return f"[Source 1] query={query}"

    class SequencedGraderModel:
        def __init__(self, scores: list[str]):
            self.scores = list(scores)

        def invoke(self, _messages):
            return AIMessage(content=f'{{"binary_score":"{self.scores.pop(0)}"}}')

    graph = build_agentic_rag_graph(
        response_model=FakeSequencedResponseModel(
            invoke_messages=["Improved question", "Final answer"]
        ),
        grader_model=SequencedGraderModel(["no", "yes"]),
        retriever_tool=retrieve_blog_posts,
        max_rewrites=2,
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Original question"}], "rewrite_count": 0}
    )

    assert result["messages"][-1].content == "Final answer"
    assert seen_queries == ["Original question", "Improved question"]


def test_graph_stops_without_retrieving_again_when_rewrite_stalls():
    seen_queries: list[str] = []

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Return matching context."""
        seen_queries.append(query)
        return "No relevant documents were found."

    graph = build_agentic_rag_graph(
        response_model=FakeResponseModel(text_response="Original question"),
        grader_model=FakeGraderModel("no"),
        retriever_tool=retrieve_blog_posts,
        max_rewrites=3,
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Original question"}], "rewrite_count": 0}
    )
    final_message = result["messages"][-1]

    assert final_message.content == INSUFFICIENT_CONTEXT_MESSAGE
    assert final_message.additional_kwargs["reason"] == INSUFFICIENT_CONTEXT_REASON
    assert seen_queries == ["Original question"]
