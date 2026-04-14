# Agentic RAG

Multi-provider agentic RAG built with LangGraph, LangChain, and local Qdrant.

This project indexes a set of source documents, persists a local Qdrant vector index on disk, and answers questions with a retrieval-first workflow that can rewrite queries, call retrieval as a tool, and generate grounded answers with citations. Rewrite loops are bounded by `MAX_REWRITES`, retrieval is reranked and cleaned before it reaches the model, and exhausted retrieval paths terminate gracefully with an `insufficient_context` result instead of looping indefinitely or guessing.

## Features

- LangGraph-based agentic RAG flow
- Multi-provider chat model support
- Multi-provider embedding support
- OpenAI-compatible endpoint support
- Dense and hybrid retrieval modes
- Deterministic reranking, deduplication, and context trimming
- Persistent local Qdrant index in `.cache/vectorstores`
- CLI with step-by-step execution tracing
- FastAPI runtime over the shared service layer
- Open WebUI integration through an external Pipe adapter
- Source-aware answers with citations
- Offline evaluation harness with checked-in baselines
- Multi-collection support through `collection_name`
- Full local quality gate with 100% test coverage

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
make install-dev
cp .env.example .env
pre-commit install
```

Python `3.10+` is required. Python `3.11` is the preferred local baseline.
The commands below assume the virtualenv is activated. If it is not, prefer the `make` targets or
run `.venv/bin/python -m ...`.

Create a `.env` file with your provider config. Minimal Google AI Studio example:

```env
CHAT_PROVIDER=google
CHAT_MODEL=gemini-2.5-flash
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=gemini-embedding-2-preview
INDEX_CACHE_DIR=.cache/vectorstores
COLLECTION_NAME=documents
INGESTION_MODE=auto
FETCH_TIMEOUT_SECONDS=20
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:5173
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

Run the API server:

```bash
agentic-rag-api
```

Or run it explicitly with Uvicorn:

```bash
python -m uvicorn agentic_rag.api:create_api_app --factory --reload
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
CHAT_MAX_TOKENS=2048
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

## API Runtime

The FastAPI app is a thin wrapper around `AgenticRagService`, so the HTTP surface reuses the same
ingestion, query, health, and index lifecycle as the CLI.

Endpoints:

- `POST /query`
- `POST /ingest`
- `GET /health`
- `GET /index/status`

`POST /query` returns a structured payload with:

- nullable `answer`
- final `message`
- `termination_reason`
- `rewrites_used`
- `index_status`
- structured `trace.steps`
- per-request `metrics`

The API supports multi-collection operation through `collection_name`. You can pass it in the
request body for `POST` endpoints or as a query parameter for `GET` endpoints. Different
collections keep separate cache fingerprints and index state.

For browser-based clients, CORS is opt-in. Set `CORS_ALLOW_ORIGINS` to a comma-separated list of
allowed origins. Example:

```env
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:5173
```

Example query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is reward hacking?",
    "collection_name": "research-notes"
  }'
```

Example ingest using the default collection:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Example ingest for a named collection:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_name": "research-notes"
  }'
```

Health check:

```bash
curl 'http://127.0.0.1:8000/health?collection_name=research-notes'
```

Index status:

```bash
curl 'http://127.0.0.1:8000/index/status?collection_name=research-notes'
```

## Open WebUI Integration

Open WebUI support is implemented as an external adapter, not as a dependency of the core package.
The backend remains a standalone CLI/API service, and Open WebUI connects to it through
[`integrations/open_webui/agentic_rag_pipe.py`](integrations/open_webui/agentic_rag_pipe.py).

That means you can remove Open WebUI later without undoing application logic or changing
`src/agentic_rag/`.

See [`docs/open-webui.md`](docs/open-webui.md) for import steps, valve configuration, and
multi-collection setup inside Open WebUI.

For a fully local one-command launcher, run:

```bash
./scripts/isabella
```

That script starts the Agentic RAG API, boots an isolated Open WebUI runtime, registers the Pipe
automatically, and exposes the UI on `http://127.0.0.1:8081` with login disabled for that local
runtime. Use `./scripts/isabella --no-browser`, `./scripts/isabella --webui-port 8090`, or
`./scripts/isabella --runtime-dir /custom/path` if you want to adjust the local setup.

To stop services previously started by Isabella for the same runtime directory:

```bash
./scripts/isabella-stop
```

Or point it at a specific runtime:

```bash
./scripts/isabella-stop --runtime-dir /custom/path
```

## How It Works

1. Load and split source documents.
2. Build or reuse a cached vector index.
3. Retrieve relevant chunks as a tool call.
4. Grade retrieved context against the current retrieval query.
5. Rewrite the question if retrieval quality is weak.
6. Normalize rewrites down to a concise retrieval query instead of feeding explanation-heavy text back into retrieval.
7. Stop at `insufficient_context` if rewrites are exhausted or if a rewrite stalls without improving the query, otherwise generate a final cited answer.

## Retrieval Quality

- Retrieval expands the initial candidate pool and applies deterministic reranking before context reaches the answer model.
- Boilerplate-heavy chunks, duplicate chunks, and low-value context are filtered out before final context assembly.
- Weak retrieval does not silently fall through to answer generation: the graph rewrites the query or terminates with `insufficient_context`.
- Offline evals under `tests/eval/` let you compare retrieval behavior without calling external model APIs.
- Integration tests exercise grounded-answer, rewrite-recovery, and insufficient-context paths end to end.

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

Run the canonical local quality gate:

```bash
make check
```

Run the individual targets that the GitHub Actions workflow executes:

```bash
make lint
make format-check
make typecheck
make test
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
python -m ruff check .
python -m ruff format --check .
python -m mypy src/agentic_rag
python -m pytest --cov=src/agentic_rag --cov-report=term-missing --cov-fail-under=85
```

The Phase 2 starter corpus and checked-in dense/hybrid baselines live under `tests/eval/`. The harness is implemented in `src/agentic_rag/evaluation.py` and runs without external APIs so retrieval changes can be compared locally before promoting them.
GitHub Actions runs the same lint, format, type-check, and test targets on Python `3.10` and
`3.11`, creating `.venv` inside the runner before calling `make`.

Main files:

- `src/agentic_rag/service.py`
- `src/agentic_rag/api.py`
- `src/agentic_rag/retrieval.py`
- `src/agentic_rag/evaluation.py`
- `src/agentic_rag/presentation.py`
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
- `CHAT_MAX_TOKENS` lets you cap completion size for chat providers; this is useful with OpenRouter routes that otherwise default to a very large token budget.
- Graph diagram export is static and does not require an existing index.
- `--verbose` enables startup diagnostics and runtime logging for command, providers, cache path, index state, and final query outcome.
- The first run for a new source/config combination builds embeddings and writes a local Qdrant index.
- Later runs reuse the cached vector index and are much cheaper/faster.
- Install Git hooks with `pre-commit install` after setting up the virtualenv.
- `make check` is the canonical local CI gate and matches the GitHub Actions workflow.
- The shared runtime entry point is `AgenticRagService`, which powers both indexing and queries.
