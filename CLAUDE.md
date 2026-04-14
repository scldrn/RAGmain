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

# Run the offline dense baseline snapshot
make eval-baseline

# Run the offline hybrid baseline snapshot
make eval-hybrid

# Compare the current hybrid run against the checked-in dense baseline
make eval-compare

# Equivalent individual quality gates
ruff check .
ruff format --check .
mypy src/agentic_rag
pytest --cov=src/agentic_rag --cov-report=term-missing --cov-fail-under=85
```

## Architecture

The system is an agentic RAG pipeline built with LangGraph. The module dependency flow is:

```
cli.py ─┬→ presentation.py
app.py ─┘
   ↓
service.py ─→ graph.py
   ├→ retrieval.py
   ├→ documents.py
   ├→ providers.py
   └→ settings.py

evaluation.py ─→ retrieval.py
```

**`settings.py`** — `AgenticRagSettings` dataclass holding all configuration: source URLs, chunk params, retrieval mode, retrieval-k, rewrite limits, provider configs, timeouts, and cache path. Default sources are 3 Lilian Weng blog posts.

**`errors.py`** — Typed domain exceptions for configuration, document loading, vector store lifecycle, and model/runtime failures.

**`service.py`** — Shared orchestration layer with `ingest()`, `query()`, `health()`, and `index_status()`. This is the main business-logic entry point used by the CLI and compatibility wrappers. It owns index fingerprints, Qdrant lifecycle, retry-wrapped models/embeddings, and graph construction.

**`providers.py`** — Multi-provider abstraction for chat and embeddings. Supported providers: `google`, `openai`, `openai-compatible`, `anthropic`, `litellm`. `resolve_chat_config()` / `resolve_embedding_config()` walk a credential fallback chain: explicit value → user-specified env var → provider-specific env var. Factory functions `create_chat_model()` and `create_embeddings()` instantiate the correct LangChain object.

**`documents.py`** — Fetches URLs via `WebBaseLoader`, applies per-URL retries and request timeouts, tolerates partial source failures, cleans navigation/outline/reference noise, and chunks with `RecursiveCharacterTextSplitter` using tiktoken token counts.

**`retrieval.py`** — Phase 2 retrieval-quality layer. It expands the candidate pool, applies deterministic reranking, trims excerpts, filters boilerplate-heavy chunks, deduplicates repeated content, balances sources, and provides lexical sparse embeddings for hybrid retrieval.

**`graph.py`** — Defines the LangGraph state machine:
```
START → prepare_retrieval → retrieve → grade_documents
                                      ├─(relevant)→ generate_answer → END
                                      ├─(not relevant, rewrites left)→ rewrite_question → prepare_retrieval
                                      └─(rewrite limit reached or stalled)→ insufficient_context → END
```
The grading step uses structured output (`GradeDocuments` schema) plus a text fallback for reliable yes/no relevance decisions, and the graph state tracks both `rewrite_count` and `rewrite_stalled` so loops terminate deterministically at `max_rewrites`.

**`app.py`** — Compatibility wrapper around `AgenticRagService`. It preserves `create_agentic_rag_app()` / `run_question()` while delegating the real work to the service layer.

**`cli.py`** — Thin CLI wrapper over `service.py`. It exposes `query` and `ingest` commands, keeps the legacy `--question ...` entry path as a compatibility alias for `query`, and owns logging setup plus startup diagnostics via `--verbose`.

**`presentation.py`** — Formatting helpers for streamed `QueryStep` output and `IndexStatus` rendering so the CLI and app wrappers share the same console presentation.

**`evaluation.py`** — Offline Phase 2 harness. It builds a deterministic local eval index, runs dense or hybrid retrieval without external APIs, emits checked-in JSON baselines, and supports summary comparisons for retrieval regressions.

## Key Design Notes

- **Persistence**: Vector stores persist as local-file Qdrant indexes under `.cache/vectorstores/{sha256-hash}/`. The hash covers source URLs, chunk size/overlap, retrieval mode, document cleaner fingerprint, and embedding provider+model — any change to these invalidates the cache automatically.
- **Ingestion modes**: `INGESTION_MODE=auto` keeps the current CLI/dev behavior and lazily builds a missing index on query. `INGESTION_MODE=explicit` is for service-style flows and requires calling `AgenticRagService.ingest()` before querying.
- **Corruption recovery**: If a persisted Qdrant directory exists but cannot be reopened cleanly, the service removes it and rebuilds it automatically.
- **Retrieval modes**: `RETRIEVAL_MODE=dense` and `RETRIEVAL_MODE=hybrid` keep separate cache fingerprints. Hybrid uses local deterministic sparse embeddings layered on Qdrant hybrid search.
- **Retrieval quality**: Query-time retrieval expands `retrieval_k` into a larger fetch pool, reranks deterministically, trims excerpts to budget, removes boilerplate, deduplicates repeated chunks, and balances sources before context is sent to the answer model.
- **No infinite rewrite loops**: `MAX_REWRITES` is enforced in the graph. Exhausted retrieval paths end in an `insufficient_context` result with a graceful fallback answer.
- **Offline regression checks**: `tests/eval/` contains the starter corpus plus dense/hybrid baseline snapshots. Use `make eval-baseline`, `make eval-hybrid`, and `make eval-compare` when changing retrieval behavior.
- **Observability**: `--verbose` enables `agentic_rag` logging, startup diagnostics, and final command summaries without changing the normal stdout answer format.
- **Quality gates**: `make check` is the canonical local gate, and `.github/workflows/ci.yml` mirrors the same lint/format/type/test sequence in GitHub Actions.
- **Test strategy**: The suite currently runs at 100% coverage and includes end-to-end service integration tests for grounded answer generation, rewrite recovery, and explicit `insufficient_context` termination.
- **Provider credentials**: Never pass API keys directly via CLI in scripts; prefer `--chat-api-key-env MY_KEY_VAR` to reference an env var by name.
- **Adding a provider**: Add to the `if/elif` chains in `create_chat_model()` and `create_embeddings()` in `providers.py`, and add the alias to `resolve_chat_config()`.
- **LiteLLM embeddings**: Handled by `LiteLLMEmbeddingsAdapter` (custom `Embeddings` subclass) since LangChain has no native LiteLLM embeddings wrapper.
