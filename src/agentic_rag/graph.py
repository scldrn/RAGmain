from __future__ import annotations

from typing import Literal, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
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
            f"[Source {index}] {heading}\n"
            f"URL: {source}\n"
            f"Content: {document.page_content.strip()}"
        )

    return "\n\n".join(sections)


def build_retriever(
    documents: Sequence[Document],
    *,
    embeddings: Embeddings,
    retrieval_k: int,
) -> BaseRetriever:
    """Create an in-memory vector retriever from chunked documents."""
    vectorstore = InMemoryVectorStore.from_documents(
        documents=list(documents),
        embedding=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": retrieval_k})


def create_retriever_tool(retriever: BaseRetriever) -> BaseTool:
    """Wrap the retriever as a LangChain tool."""

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search the indexed blog posts for relevant context."""
        documents = retriever.invoke(query)
        return format_retrieved_documents(documents)

    return retrieve_blog_posts

def make_generate_query_or_respond(response_model, retriever_tool: BaseTool):
    """Create the node that decides between direct response and retrieval."""
    model_with_tools = response_model.bind_tools([retriever_tool])

    def generate_query_or_respond(state: MessagesState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return generate_query_or_respond


def make_grade_documents(grader_model):
    """Create the edge function that grades retrieved context."""
    structured_grader = grader_model.with_structured_output(GradeDocuments)

    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        question = message_text(state["messages"][0])
        context = message_text(state["messages"][-1])
        prompt = RELEVANCE_PROMPT.format(question=question, context=context)
        response = structured_grader.invoke([{"role": "user", "content": prompt}])

        if response.binary_score == "yes":
            return "generate_answer"
        return "rewrite_question"

    return grade_documents


def make_rewrite_question(response_model):
    """Create the node that rewrites the user's question."""

    def rewrite_question(state: MessagesState):
        question = message_text(state["messages"][0])
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=message_text(response))]}

    return rewrite_question


def make_generate_answer(response_model):
    """Create the final answer-generation node."""

    def generate_answer(state: MessagesState):
        question = message_text(state["messages"][0])
        context = message_text(state["messages"][-1])
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    return generate_answer


def build_agentic_rag_graph(
    *,
    response_model,
    grader_model,
    retriever_tool: BaseTool,
):
    """Assemble the LangGraph workflow for agentic RAG."""
    workflow = StateGraph(MessagesState)

    workflow.add_node(
        "generate_query_or_respond",
        make_generate_query_or_respond(response_model, retriever_tool),
    )
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", make_rewrite_question(response_model))
    workflow.add_node("generate_answer", make_generate_answer(response_model))

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges("retrieve", make_grade_documents(grader_model))
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()
