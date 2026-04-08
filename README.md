# Agentic RAG

Multi-provider agentic RAG built with LangGraph and LangChain.

This project indexes a set of source documents, persists a local Qdrant vector index on disk, and answers questions with a retrieval-first workflow that can rewrite queries, call retrieval as a tool, and generate grounded answers with citations. Rewrite loops are bounded by `MAX_REWRITES`, and exhausted retrieval paths terminate gracefully with an `insufficient_context` result instead of looping indefinitely.

## Features

- LangGraph-based agentic RAG flow
- Multi-provider chat model support
- Multi-provider embedding support
- OpenAI-compatible endpoint support
- Persistent local Qdrant index in `.cache/vectorstores`
- CLI with step-by-step execution tracing
- Source-aware answers with citations

## Supported Providers

### Chat

- `google`
- `openai`
- `openai-compatible`
- `anthropic`
- `litellm`

### Embeddings

- `google`
- `openai`
- `openai-compatible`
- `litellm`

## Quickstart

```bash
git clone git@github.com:scldrn/RAGmain.git
cd RAGmain
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Python `3.10+` is required. Python `3.11` is the preferred local baseline.

Create a `.env` file with your provider config. Minimal Google AI Studio example:

```env
CHAT_PROVIDER=google
CHAT_MODEL=gemini-2.5-flash
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=gemini-embedding-2-preview
INDEX_CACHE_DIR=.cache/vectorstores
INGESTION_MODE=auto
FETCH_TIMEOUT_SECONDS=20
GOOGLE_API_KEY=your_api_key
```

Run a query:

```bash
python -m agentic_rag --question "What does Lilian Weng say about reward hacking?"
```

Run the explicit query command:

```bash
python -m agentic_rag query --question "What does Lilian Weng say about reward hacking?"
```

Pre-build the index explicitly:

```bash
python -m agentic_rag ingest
```

Run with trace output:

```bash
python -m agentic_rag \
  --question "What does Lilian Weng say about reward hacking?" \
  --show-steps
```

Run with startup diagnostics and verbose logs:

```bash
python -m agentic_rag \
  --verbose \
  --question "What does Lilian Weng say about reward hacking?"
```

## Provider Examples

OpenAI:

```env
CHAT_PROVIDER=openai
CHAT_MODEL=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_api_key
```

Anthropic chat + OpenAI embeddings:

```env
CHAT_PROVIDER=anthropic
CHAT_MODEL=claude-3-5-sonnet-20241022
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
ANTHROPIC_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
```

OpenAI-compatible local endpoint:

```env
CHAT_PROVIDER=openai-compatible
CHAT_MODEL=local-model-name
CHAT_API_BASE=http://localhost:1234/v1
CHAT_API_KEY=lm-studio
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_BASE=http://localhost:1234/v1
EMBEDDING_API_KEY=lm-studio
```

## CLI Options

Common options:

- `query`
- `ingest`
- `--question`
- `--url`
- `--chat-provider`
- `--chat-model`
- `--embedding-provider`
- `--embedding-model`
- `--chat-api-base`
- `--embedding-api-base`
- `--show-steps`
- `--diagram`
- `--verbose`

Show full help:

```bash
python -m agentic_rag --help
```

## How It Works

1. Load and split source documents.
2. Build or reuse a cached vector index.
3. Retrieve relevant chunks as a tool call.
4. Grade retrieved context against the current retrieval query.
5. Rewrite the question if retrieval quality is weak.
6. Normalize rewrites down to a concise retrieval query instead of feeding explanation-heavy text back into retrieval.
7. Stop at `insufficient_context` if rewrites are exhausted or if a rewrite stalls without improving the query, otherwise generate a final cited answer.

## Default Sources

The current default setup indexes a small set of Lilian Weng blog posts so the project works immediately as a reference RAG example.

You can add your own sources at runtime:

```bash
python -m agentic_rag \
  --url https://example.com/doc1 \
  --url https://example.com/doc2 \
  --question "Summarize the main ideas"
```

## Development

Run the local quality gates:

```bash
make check
```

Run the offline Phase 2 dense baseline:

```bash
make eval-baseline
```

Run the checked-in hybrid snapshot:

```bash
make eval-hybrid
```

Compare the current hybrid run against the checked-in dense baseline:

```bash
make eval-compare
```

Equivalent individual commands:

```bash
ruff check .
ruff format --check .
mypy src/agentic_rag
pytest --cov=src/agentic_rag --cov-report=term-missing --cov-fail-under=85
```

The Phase 2 starter corpus and checked-in dense/hybrid baselines live under `tests/eval/`. The harness is implemented in `src/agentic_rag/evaluation.py` and runs without external APIs so retrieval changes can be compared locally before promoting them.

Main files:

- `src/agentic_rag/service.py`
- `src/agentic_rag/errors.py`
- `src/agentic_rag/providers.py`
- `src/agentic_rag/app.py`
- `src/agentic_rag/graph.py`
- `src/agentic_rag/cli.py`
- `src/agentic_rag/settings.py`

## Notes

- `python -m agentic_rag --question "..."` remains supported as a legacy shortcut for `python -m agentic_rag query --question "..."`.
- `INGESTION_MODE=auto` builds a missing index on first query. `INGESTION_MODE=explicit` requires calling `AgenticRagService.ingest()` before querying.
- `MAX_REWRITES` bounds rewrite loops. When retrieval remains weak after that limit, the service returns a structured `insufficient_context` termination reason and a graceful fallback answer.
- `RETRIEVAL_MODE=dense` remains the baseline. `RETRIEVAL_MODE=hybrid` now enables a local Qdrant dense+lexical hybrid index and keeps a separate cache fingerprint from the dense index.
- Query-time retrieval now expands the candidate pool, applies deterministic reranking, trims boilerplate-heavy chunks, deduplicates repeated context, and balances sources before passing context to the model.
- The offline Phase 2 dense and hybrid snapshots are currently aligned under the deterministic reranking path; keep the harness for regression checks when retrieval changes.
- Rewrite loops now sanitize verbose rewrite outputs into a concise query and stop earlier when a rewrite fails to improve the retrieval query.
- Corrupted or incomplete Qdrant cache directories are detected and rebuilt automatically.
- Document fetches use per-request timeouts, retry each URL, and continue indexing with the remaining sources when only some URLs fail.
- `MODEL_TIMEOUT_SECONDS` applies to provider clients, and transient chat/embedding failures retry with exponential backoff before surfacing a typed error.
- Graph diagram export is static and does not require an existing index.
- `--verbose` enables startup diagnostics and runtime logging for command, providers, cache path, index state, and final query outcome.
- The first run for a new source/config combination builds embeddings and writes a local Qdrant index.
- Later runs reuse the cached vector index and are much cheaper/faster.
- Install Git hooks with `pre-commit install` after setting up the virtualenv.
- `make check` is the canonical local CI gate and matches the GitHub Actions workflow.
- The shared runtime entry point is `AgenticRagService`, which powers both indexing and queries.
