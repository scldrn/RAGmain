from __future__ import annotations

import re
from typing import Any, Callable, Literal, Protocol, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from agentic_rag.errors import LLMError

RELEVANCE_PROMPT = """
You are checking whether retrieved context can help answer a question.

Question:
{question}

Retrieved context:
{context}

Reply with a strict JSON object: {{"binary_score":"yes"}} when the context is meaningfully
relevant, otherwise {{"binary_score":"no"}}.
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

NO_RELEVANT_DOCUMENTS_MESSAGE = "No relevant documents were found."
INSUFFICIENT_CONTEXT_REASON = "insufficient_context"
INSUFFICIENT_CONTEXT_MESSAGE = (
    "I do not have enough relevant context to answer reliably after the allowed retrieval attempts."
)

GraphRoute = Literal["generate_answer", "rewrite_question", "insufficient_context"]


class AgenticRagState(MessagesState):
    rewrite_count: int
    rewrite_stalled: bool


class GradeDocuments(BaseModel):
    """Binary relevance decision for retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(
        description='Relevance score: "yes" when the context helps answer the question.'
    )


class RetrieverLike(Protocol):
    def invoke(self, query: str) -> Sequence[Document]: ...


class ChatModelLike(Protocol):
    def invoke(self, messages: Any) -> BaseMessage: ...


def _context_has_no_relevant_documents(context: str) -> bool:
    return context.strip() == NO_RELEVANT_DOCUMENTS_MESSAGE


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
        return NO_RELEVANT_DOCUMENTS_MESSAGE

    sections = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        title = document.metadata.get("title")
        heading = title or source
        sections.append(
            f"[Source {index}] {heading}\nURL: {source}\nContent: {document.page_content.strip()}"
        )

    return "\n\n".join(sections)


def create_retriever_tool(retriever: RetrieverLike | BaseRetriever) -> BaseTool:
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


def rewrite_stalled(state: AgenticRagState) -> bool:
    """Return whether the latest rewrite failed to improve the retrieval query."""
    return bool(state.get("rewrite_stalled", False))


def current_retrieval_query(state: AgenticRagState) -> str:
    """Return the latest human-authored question to use for retrieval."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            text = message_text(message).strip()
            if text:
                return text

    return message_text(state["messages"][0]).strip()


def _normalize_rewritten_question(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        normalized = re.sub(r"\s+", " ", raw_line.strip()).strip("*-• \t")
        normalized = normalized.strip()
        if not normalized:
            continue

        lowered = normalized.strip(":").lower()
        if lowered.startswith(
            ("a clearer version would be", "why this works", "original question")
        ):
            continue

        lines.append(normalized.strip("\"'` "))

    if not lines:
        return re.sub(r"\s+", " ", text).strip()

    for line in lines:
        if "?" in line:
            return line[: line.rfind("?") + 1].strip()

    return lines[0]


def make_prepare_retrieval(
    retriever_tool: BaseTool,
) -> Callable[[AgenticRagState], dict[str, Any]]:
    """Create the node that always routes through retrieval first."""

    def prepare_retrieval(state: AgenticRagState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"retrieve-{rewrite_count(state)}",
                            "name": retriever_tool.name,
                            "args": {"query": current_retrieval_query(state)},
                        }
                    ],
                )
            ]
        }

    return prepare_retrieval


def make_grade_documents(
    grader_model: ChatModelLike,
    *,
    max_rewrites: int,
) -> Callable[[AgenticRagState], GraphRoute]:
    """Create the edge function that grades retrieved context."""

    def grade_documents(state: AgenticRagState) -> GraphRoute:
        question = current_retrieval_query(state)
        context = message_text(state["messages"][-1])
        if _context_has_no_relevant_documents(context):
            if _should_stop_after_failed_retrieval(state, max_rewrites=max_rewrites):
                return "insufficient_context"
            return "rewrite_question"
        prompt = RELEVANCE_PROMPT.format(question=question, context=context)
        response = _parse_grade_documents_response(
            grader_model.invoke([{"role": "user", "content": prompt}])
        )

        if response.binary_score == "yes":
            return "generate_answer"
        if _should_stop_after_failed_retrieval(state, max_rewrites=max_rewrites):
            return "insufficient_context"
        return "rewrite_question"

    return grade_documents


def make_rewrite_question(
    response_model: ChatModelLike,
) -> Callable[[AgenticRagState], dict[str, Any]]:
    """Create the node that rewrites the user's question."""

    def rewrite_question(state: AgenticRagState) -> dict[str, Any]:
        question = current_retrieval_query(state)
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        rewritten_question = _normalize_rewritten_question(message_text(response)) or question
        return {
            "messages": [HumanMessage(content=rewritten_question)],
            "rewrite_count": rewrite_count(state) + 1,
            "rewrite_stalled": rewritten_question.casefold() == question.casefold(),
        }

    return rewrite_question


def route_after_rewrite(
    state: AgenticRagState,
) -> Literal["prepare_retrieval", "insufficient_context"]:
    """Stop early when a rewrite fails to improve the retrieval query."""
    if rewrite_stalled(state):
        return "insufficient_context"
    return "prepare_retrieval"


def _should_stop_after_failed_retrieval(
    state: AgenticRagState,
    *,
    max_rewrites: int,
) -> bool:
    return rewrite_count(state) >= max_rewrites or rewrite_stalled(state)


def _parse_grade_documents_response(message: BaseMessage) -> GradeDocuments:
    text = message_text(message).strip()
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate_payloads = [json_match.group(0)] if json_match else []
    if text and text not in candidate_payloads:
        candidate_payloads.append(text)

    for payload in candidate_payloads:
        try:
            return GradeDocuments.model_validate_json(payload)
        except ValidationError:
            continue

    lowered = text.lower()
    fallback_match = re.search(r"\b(yes|no)\b", lowered)
    if fallback_match is not None:
        token = fallback_match.group(1)
        return GradeDocuments(binary_score="yes" if token == "yes" else "no")

    raise LLMError("Failed to parse the document relevance grade.")


def make_generate_answer(
    response_model: ChatModelLike,
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
    response_model: ChatModelLike,
    grader_model: ChatModelLike,
    retriever_tool: BaseTool,
    max_rewrites: int = 3,
) -> Any:
    """Assemble the LangGraph workflow for agentic RAG."""
    workflow = StateGraph(AgenticRagState)

    workflow.add_node(  # type: ignore[call-overload]
        "prepare_retrieval",
        make_prepare_retrieval(retriever_tool),
    )
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", make_rewrite_question(response_model))  # type: ignore[call-overload]
    workflow.add_node("generate_answer", make_generate_answer(response_model))  # type: ignore[call-overload]
    workflow.add_node("insufficient_context", make_generate_insufficient_context())  # type: ignore[call-overload]

    workflow.add_edge(START, "prepare_retrieval")
    workflow.add_edge("prepare_retrieval", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        make_grade_documents(grader_model, max_rewrites=max_rewrites),
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("insufficient_context", END)
    workflow.add_conditional_edges("rewrite_question", route_after_rewrite)

    return workflow.compile()
