# Agentic RAG

Multi-provider agentic RAG built with LangGraph and LangChain.

This project indexes a set of source documents, caches the vector index locally, and answers questions with a retrieval-first workflow that can rewrite queries, call retrieval as a tool, and generate grounded answers with citations.

## Features

- LangGraph-based agentic RAG flow
- Multi-provider chat model support
- Multi-provider embedding support
- OpenAI-compatible endpoint support
- Local vector index cache in `.cache/vectorstores`
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
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Create a `.env` file with your provider config. Minimal Google AI Studio example:

```env
CHAT_PROVIDER=google
CHAT_MODEL=gemini-2.5-flash
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=gemini-embedding-2-preview
INDEX_CACHE_DIR=.cache/vectorstores
GOOGLE_API_KEY=your_api_key
```

Run a query:

```bash
python -m agentic_rag --question "What does Lilian Weng say about reward hacking?"
```

Run with trace output:

```bash
python -m agentic_rag \
  --question "What does Lilian Weng say about reward hacking?" \
  --show-steps
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

Show full help:

```bash
python -m agentic_rag --help
```

## How It Works

1. Load and split source documents.
2. Build or reuse a cached vector index.
3. Let the graph decide whether to answer directly or retrieve.
4. Retrieve relevant chunks as a tool call.
5. Grade retrieved context.
6. Rewrite the question if retrieval quality is weak.
7. Generate a final cited answer.

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

Run tests:

```bash
pytest
```

Main files:

- `src/agentic_rag/providers.py`
- `src/agentic_rag/app.py`
- `src/agentic_rag/graph.py`
- `src/agentic_rag/cli.py`
- `src/agentic_rag/settings.py`

## Notes

- The first run for a new source/config combination builds embeddings and writes a cache file.
- Later runs reuse the cached vector index and are much cheaper/faster.
- Python `3.10+` is recommended even though the project currently runs on `3.9`.
