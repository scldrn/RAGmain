from __future__ import annotations

import argparse

from agentic_rag.app import create_agentic_rag_app, export_graph_mermaid, run_question
from agentic_rag.graph import message_text
from agentic_rag.settings import AgenticRagSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph agentic RAG example as a local Python project."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to send through the agentic RAG graph.",
    )
    parser.add_argument(
        "--url",
        dest="source_urls",
        action="append",
        help="Source URL to index. Repeat this flag to add multiple URLs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size used during document splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap used during document splitting.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=4,
        help="Number of chunks returned by the retriever.",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Gemini chat model to use. Defaults to GOOGLE_CHAT_MODEL or gemini-2.5-flash.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Gemini embedding model to use. Defaults to GOOGLE_EMBEDDING_MODEL or gemini-embedding-2-preview.",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Print node-by-node updates while the graph runs.",
    )
    parser.add_argument(
        "--diagram",
        help="Optional output file for the Mermaid graph diagram.",
    )
    return parser


def build_settings(args: argparse.Namespace) -> AgenticRagSettings:
    base = AgenticRagSettings()
    return AgenticRagSettings(
        source_urls=args.source_urls or base.source_urls,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_k=args.retrieval_k,
        chat_model=args.chat_model or base.chat_model,
        embedding_model=args.embedding_model or base.embedding_model,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = build_settings(args)

    try:
        app = create_agentic_rag_app(settings)
    except RuntimeError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    if args.diagram:
        export_graph_mermaid(app, args.diagram)

    final_message = run_question(app, args.question, show_steps=args.show_steps)
    print("\nFinal answer:\n")
    print(message_text(final_message).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
