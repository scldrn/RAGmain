from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

RELEVANCE_PROMPT = """
You are checking whether retrieved context can help answer a question.

Question:
{question}

Retrieved context:
{context}

Reply with "yes" when the context is meaningfully relevant, otherwise reply "no".
""".strip()

REWRITE_PROMPT = """
Rewrite the user's question so retrieval is more likely to succeed.
Focus on the underlying intent and keep the improved question specific.

Original question:
{question}
""".strip()

ANSWER_PROMPT = """
You answer questions using only the retrieved context.
If the context is insufficient, say you do not know.
Keep the answer concise and cite source labels like [Source 1] when possible.

Question:
{question}

Context:
{context}
""".strip()

INSUFFICIENT_CONTEXT_REASON = "insufficient_context"
INSUFFICIENT_CONTEXT_MESSAGE = (
    "I do not have enough relevant context to answer reliably after the allowed retrieval attempts."
)

GraphRoute = Literal["generate_answer", "rewrite_question", "insufficient_context"]


class AgenticRagState(MessagesState):
    rewrite_count: int


class GradeDocuments(BaseModel):
    """Binary relevance decision for retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(
        description='Relevance score: "yes" when the context helps answer the question.'
    )


def message_text(message: BaseMessage) -> str:
    """Extract text from a LangChain message."""
    content = message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def format_retrieved_documents(documents: Sequence[Document]) -> str:
    """Serialize retrieved documents into a readable tool response."""
    if not documents:
        return "No relevant documents were found."

    sections = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        title = document.metadata.get("title")
        heading = title or source
        sections.append(
            f"[Source {index}] {heading}\nURL: {source}\nContent: {document.page_content.strip()}"
        )

    return "\n\n".join(sections)


def create_retriever_tool(retriever: BaseRetriever) -> BaseTool:
    """Wrap the retriever as a LangChain tool."""

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search the indexed blog posts for relevant context."""
        documents = retriever.invoke(query)
        return format_retrieved_documents(documents)

    return retrieve_blog_posts


def rewrite_count(state: AgenticRagState) -> int:
    """Return the current number of rewrite attempts recorded in state."""
    return int(state.get("rewrite_count", 0))


def make_generate_query_or_respond(
    response_model: BaseChatModel,
    retriever_tool: BaseTool,
) -> Callable[[AgenticRagState], dict[str, Any]]:
    """Create the node that decides between direct response and retrieval."""
    model_with_tools = response_model.bind_tools([retriever_tool])

    def generate_query_or_respond(state: AgenticRagState) -> dict[str, Any]:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return generate_query_or_respond


def make_grade_documents(
    grader_model: BaseChatModel,
    *,
    max_rewrites: int,
) -> Callable[[AgenticRagState], GraphRoute]:
    """Create the edge function that grades retrieved context."""
    structured_grader = grader_model.with_structured_output(GradeDocuments)

    def grade_documents(state: AgenticRagState) -> GraphRoute:
        question = message_text(state["messages"][0])
        context = message_text(state["messages"][-1])
        prompt = RELEVANCE_PROMPT.format(question=question, context=context)
        response = cast(
            GradeDocuments,
            structured_grader.invoke([{"role": "user", "content": prompt}]),
        )

        if response.binary_score == "yes":
            return "generate_answer"
        if rewrite_count(state) >= max_rewrites:
            return "insufficient_context"
        return "rewrite_question"

    return grade_documents


def make_rewrite_question(
    response_model: BaseChatModel,
) -> Callable[[AgenticRagState], dict[str, Any]]:
    """Create the node that rewrites the user's question."""

    def rewrite_question(state: AgenticRagState) -> dict[str, Any]:
        question = message_text(state["messages"][0])
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {
            "messages": [HumanMessage(content=message_text(response))],
            "rewrite_count": rewrite_count(state) + 1,
        }

    return rewrite_question


def make_generate_answer(
    response_model: BaseChatModel,
) -> Callable[[AgenticRagState], dict[str, Any]]:
    """Create the final answer-generation node."""

    def generate_answer(state: AgenticRagState) -> dict[str, Any]:
        question = message_text(state["messages"][0])
        context = message_text(state["messages"][-1])
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    return generate_answer


def make_generate_insufficient_context() -> Callable[[AgenticRagState], dict[str, Any]]:
    """Create the terminal node returned when rewrite attempts are exhausted."""

    def generate_insufficient_context(state: AgenticRagState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content=INSUFFICIENT_CONTEXT_MESSAGE,
                    additional_kwargs={
                        "reason": INSUFFICIENT_CONTEXT_REASON,
                        "answer": None,
                        "rewrites_used": rewrite_count(state),
                    },
                )
            ]
        }

    return generate_insufficient_context


def build_agentic_rag_graph(
    *,
    response_model: BaseChatModel,
    grader_model: BaseChatModel,
    retriever_tool: BaseTool,
    max_rewrites: int = 3,
) -> Any:
    """Assemble the LangGraph workflow for agentic RAG."""
    workflow = StateGraph(AgenticRagState)

    workflow.add_node(  # type: ignore[call-overload]
        "generate_query_or_respond",
        make_generate_query_or_respond(response_model, retriever_tool),
    )
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", make_rewrite_question(response_model))  # type: ignore[call-overload]
    workflow.add_node("generate_answer", make_generate_answer(response_model))  # type: ignore[call-overload]
    workflow.add_node("insufficient_context", make_generate_insufficient_context())  # type: ignore[call-overload]

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        make_grade_documents(grader_model, max_rewrites=max_rewrites),
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("insufficient_context", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()
