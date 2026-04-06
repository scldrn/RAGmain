from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentic_rag.graph import (
    build_agentic_rag_graph,
    format_retrieved_documents,
    make_generate_answer,
    make_grade_documents,
    make_rewrite_question,
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
        return FakeBoundToolModel(self.tool_message)

    def invoke(self, _messages):
        return AIMessage(content=self.text_response)


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


def test_grade_documents_routes_to_answer_when_context_is_relevant():
    grade_documents = make_grade_documents(FakeGraderModel("yes"))
    state = {
        "messages": [
            HumanMessage(content="What is reward hacking?"),
            ToolMessage(content="Reward hacking is when agents exploit rewards.", tool_call_id="1"),
        ]
    }

    assert grade_documents(state) == "generate_answer"


def test_rewrite_question_returns_human_message():
    rewrite_question = make_rewrite_question(FakeResponseModel(text_response="Improved question"))
    state = {"messages": [HumanMessage(content="Original question")]}

    result = rewrite_question(state)

    assert result["messages"][-1].content == "Improved question"


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
    )

    result = graph.invoke({"messages": [{"role": "user", "content": "What is reward hacking?"}]})

    assert result["messages"][-1].content == "Final answer"
