# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env  # then add API keys
pre-commit install
```

The project requires Python `3.10+`. Use Python `3.11` locally unless there is a specific compatibility reason not to.

## Commands

```bash
# Run a question
python -m agentic_rag --question "What does Lilian Weng say about reward hacking?"

# Run the explicit query command
python -m agentic_rag query --question "What does Lilian Weng say about reward hacking?"

# Build the index explicitly
python -m agentic_rag ingest

# Run with step-by-step tracing
python -m agentic_rag --question "..." --show-steps

# Run with startup diagnostics and verbose logging
python -m agentic_rag --verbose --question "..."

# Export graph diagram
python -m agentic_rag --question "test" --diagram graph.md

# Run tests
pytest

# Run a single test
pytest tests/test_graph.py::test_name

# Run the canonical local CI gate
make check

# Equivalent individual quality gates
ruff check .
ruff format --check .
mypy src/agentic_rag
pytest --cov=src/agentic_rag --cov-report=term-missing --cov-fail-under=85
```

## Architecture

The system is an agentic RAG pipeline built with LangGraph. The module dependency flow is:

```
cli.py → service.py → providers.py
 app.py ↗        ↓
            graph.py → documents.py
               ↓
           settings.py
```

**`settings.py`** — `AgenticRagSettings` dataclass holding all configuration: source URLs, chunk params, retrieval-k, provider configs, and cache path. Default sources are 3 Lilian Weng blog posts.

**`errors.py`** — Typed domain exceptions for configuration, document loading, vector store lifecycle, and model/runtime failures.

**`service.py`** — Shared orchestration layer with `ingest()`, `query()`, `health()`, and `index_status()`. This is the main business-logic entry point used by the CLI and compatibility wrappers.

**`providers.py`** — Multi-provider abstraction for chat and embeddings. Supported providers: `google`, `openai`, `openai-compatible`, `anthropic`, `litellm`. `resolve_chat_config()` / `resolve_embedding_config()` walk a credential fallback chain: explicit value → user-specified env var → provider-specific env var. Factory functions `create_chat_model()` and `create_embeddings()` instantiate the correct LangChain object.

**`documents.py`** — Fetches URLs via `WebBaseLoader`, applies per-URL retries and request timeouts, tolerates partial source failures, and chunks with `RecursiveCharacterTextSplitter` using tiktoken token counts.

**`graph.py`** — Defines the LangGraph state machine:
```
START → generate_query_or_respond
          ├─(tool call)→ retrieve → grade_documents
          │                            ├─(relevant)→ generate_answer → END
          │                            ├─(not relevant, rewrites left)→ rewrite_question → generate_query_or_respond
          │                            └─(rewrite limit reached)→ insufficient_context → END
          └─(no tool call)→ END
```
The grading step uses structured output (`GradeDocuments` schema) for reliable yes/no relevance decisions, and the graph state tracks `rewrite_count` so loops terminate deterministically at `max_rewrites`.

**`app.py`** — Compatibility wrapper around `AgenticRagService`. It preserves `create_agentic_rag_app()` / `run_question()` while delegating the real work to the service layer.

**`cli.py`** — Thin CLI wrapper over `service.py`. It exposes `query` and `ingest` commands, keeps the legacy `--question ...` entry path as a compatibility alias for `query`, and owns logging setup plus startup diagnostics via `--verbose`.

## Key Design Notes

- **Persistence**: Vector stores persist as local-file Qdrant indexes under `.cache/vectorstores/{sha256-hash}/`. The hash covers source URLs, chunk size/overlap, and embedding provider+model — any change to these invalidates the cache automatically.
- **Ingestion modes**: `INGESTION_MODE=auto` keeps the current CLI/dev behavior and lazily builds a missing index on query. `INGESTION_MODE=explicit` is for service-style flows and requires calling `AgenticRagService.ingest()` before querying.
- **Corruption recovery**: If a persisted Qdrant directory exists but cannot be reopened cleanly, the service removes it and rebuilds it automatically.
- **No infinite rewrite loops**: `MAX_REWRITES` is enforced in the graph. Exhausted retrieval paths end in an `insufficient_context` result with a graceful fallback answer.
- **Observability**: `--verbose` enables `agentic_rag` logging, startup diagnostics, and final command summaries without changing the normal stdout answer format.
- **Quality gates**: `make check` is the canonical local gate, and `.github/workflows/ci.yml` mirrors the same lint/format/type/test sequence in GitHub Actions.
- **Provider credentials**: Never pass API keys directly via CLI in scripts; prefer `--chat-api-key-env MY_KEY_VAR` to reference an env var by name.
- **Adding a provider**: Add to the `if/elif` chains in `create_chat_model()` and `create_embeddings()` in `providers.py`, and add the alias to `resolve_chat_config()`.
- **LiteLLM embeddings**: Handled by `LiteLLMEmbeddingsAdapter` (custom `Embeddings` subclass) since LangChain has no native LiteLLM embeddings wrapper.
