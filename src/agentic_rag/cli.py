from __future__ import annotations

import argparse
import os

from agentic_rag.app import create_agentic_rag_app, export_graph_mermaid, run_question
from agentic_rag.graph import message_text
from agentic_rag.settings import (
    SUPPORTED_CHAT_PROVIDERS,
    SUPPORTED_EMBEDDING_PROVIDERS,
    AgenticRagSettings,
)


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
        "--chat-provider",
        default=None,
        help=(
            "Chat provider to use. Supported values: "
            f"{', '.join(SUPPORTED_CHAT_PROVIDERS)}."
        ),
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Chat model to use for the selected provider.",
    )
    parser.add_argument(
        "--chat-api-key",
        default=None,
        help="Optional explicit API key for the chat provider.",
    )
    parser.add_argument(
        "--chat-api-key-env",
        default=None,
        help="Name of the environment variable that stores the chat API key.",
    )
    parser.add_argument(
        "--chat-api-base",
        default=None,
        help="Optional base URL for chat providers that support custom endpoints.",
    )
    parser.add_argument(
        "--chat-temperature",
        type=float,
        default=None,
        help="Sampling temperature for the chat model.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help=(
            "Embedding provider to use. Supported values: "
            f"{', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model to use for the selected embedding provider.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        help="Optional explicit API key for the embedding provider.",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default=None,
        help="Name of the environment variable that stores the embedding API key.",
    )
    parser.add_argument(
        "--embedding-api-base",
        default=None,
        help="Optional base URL for embedding providers that support custom endpoints.",
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
        chat_provider=args.chat_provider or base.chat_provider,
        chat_model=args.chat_model or base.chat_model,
        chat_api_key=args.chat_api_key or base.chat_api_key,
        chat_api_key_env=args.chat_api_key_env or base.chat_api_key_env,
        chat_api_base=args.chat_api_base or base.chat_api_base,
        chat_temperature=(
            args.chat_temperature
            if args.chat_temperature is not None
            else base.chat_temperature
        ),
        embedding_provider=args.embedding_provider or base.embedding_provider,
        embedding_model=args.embedding_model or base.embedding_model,
        embedding_api_key=args.embedding_api_key or base.embedding_api_key,
        embedding_api_key_env=args.embedding_api_key_env or base.embedding_api_key_env,
        embedding_api_base=args.embedding_api_base or base.embedding_api_base,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = build_settings(args)

    try:
        app = create_agentic_rag_app(settings)
    except RuntimeError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    except Exception as exc:
        if os.getenv("AGENTIC_RAG_DEBUG"):
            raise
        parser.exit(status=1, message=f"error: {exc}\n")

    if args.diagram:
        export_graph_mermaid(app, args.diagram)

    try:
        final_message = run_question(app, args.question, show_steps=args.show_steps)
    except Exception as exc:
        if os.getenv("AGENTIC_RAG_DEBUG"):
            raise
        parser.exit(status=1, message=f"error: {exc}\n")

    print("\nFinal answer:\n")
    print(message_text(final_message).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
