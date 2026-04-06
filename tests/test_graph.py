from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentic_rag.graph import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    INSUFFICIENT_CONTEXT_REASON,
    build_agentic_rag_graph,
    create_retriever_tool,
    format_retrieved_documents,
    make_generate_answer,
    make_generate_insufficient_context,
    make_generate_query_or_respond,
    make_grade_documents,
    make_rewrite_question,
    message_text,
    rewrite_count,
)


class FakeBoundToolModel:
    def __init__(self, message: AIMessage):
        self.message = message

    def invoke(self, _messages):
        return self.message


class FakeResponseModel:
    def __init__(self, *, tool_message: AIMessage | None = None, text_response: str = ""):
        self.tool_message = tool_message
        self.text_response = text_response

    def bind_tools(self, _tools):
        return FakeBoundToolModel(self.tool_message or AIMessage(content=self.text_response))

    def invoke(self, _messages):
        return AIMessage(content=self.text_response)


class FakeSequencedBoundToolModel:
    def __init__(self, messages: list[AIMessage]):
        self.messages = list(messages)

    def invoke(self, _messages):
        return self.messages.pop(0)


class FakeSequencedResponseModel:
    def __init__(self, *, bound_messages: list[AIMessage], invoke_messages: list[str]):
        self.bound_messages = list(bound_messages)
        self.invoke_messages = list(invoke_messages)

    def bind_tools(self, _tools):
        return FakeSequencedBoundToolModel(self.bound_messages)

    def invoke(self, _messages):
        return AIMessage(content=self.invoke_messages.pop(0))


class FakeStructuredOutput:
    def __init__(self, score: str):
        self.score = score

    def invoke(self, _messages):
        return type("GradeResult", (), {"binary_score": self.score})()


class FakeGraderModel:
    def __init__(self, score: str):
        self.score = score

    def with_structured_output(self, _schema):
        return FakeStructuredOutput(self.score)


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


def test_generate_query_or_respond_returns_bound_model_response():
    generate_query_or_respond = make_generate_query_or_respond(
        FakeResponseModel(text_response="Use the retriever"),
        MagicMock(),
    )

    result = generate_query_or_respond({"messages": [HumanMessage(content="Question")]})

    assert result["messages"][-1].content == "Use the retriever"


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


def test_rewrite_question_returns_human_message():
    rewrite_question = make_rewrite_question(FakeResponseModel(text_response="Improved question"))
    state = {"messages": [HumanMessage(content="Original question")], "rewrite_count": 0}

    result = rewrite_question(state)

    assert result["messages"][-1].content == "Improved question"
    assert result["rewrite_count"] == 1


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

    response_model = FakeResponseModel(
        tool_message=AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool-1",
                    "name": "retrieve_blog_posts",
                    "args": {"query": "reward hacking"},
                }
            ],
        ),
        text_response="Final answer",
    )
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


def test_graph_can_finish_without_retrieval():
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Return matching context."""
        return f"[Source 1] query={query}"

    graph = build_agentic_rag_graph(
        response_model=FakeResponseModel(text_response="Direct answer"),
        grader_model=FakeGraderModel("yes"),
        retriever_tool=retrieve_blog_posts,
        max_rewrites=3,
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What is reward hacking?"}], "rewrite_count": 0}
    )

    assert result["messages"][-1].content == "Direct answer"


def test_graph_returns_insufficient_context_after_max_rewrites():
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Return matching context."""
        return f"[Source 1] query={query}"

    response_model = FakeSequencedResponseModel(
        bound_messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool-1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "reward hacking"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool-2",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "reward hacking"},
                    }
                ],
            ),
        ],
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
