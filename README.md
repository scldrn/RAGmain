# Robust RAG System

A production-ready Retrieval-Augmented Generation (RAG) system designed for **accuracy, robustness, traceability, and continuous evaluation**.

This project implements a modern RAG architecture based on:

- Structural chunking
- Contextual chunk enrichment
- Hybrid retrieval (semantic + keyword)
- Reranking
- Hierarchical retrieval
- Grounded generation with citations
- Continuous evaluation

---

## Objective

Build a RAG system that is:

- Accurate
- Explainable
- Stable
- Scalable
- Measurable

Avoid naive pipelines like:

embeddings → vector search → top-k → prompt

Instead, use:

query rewrite → hybrid retrieval → reranking → context assembly → grounded generation → evaluation

---

## Architecture Overview

### Core Components

1. Document ingestion
2. Structural chunking
3. Chunk contextualization
4. Hybrid indexing (vector + BM25)
5. Retrieval + query rewriting
6. Reranking
7. Context assembly
8. LLM generation with citations
9. Evaluation pipeline

---

## Key Features

- Hybrid retrieval (semantic + keyword)
- Metadata filtering
- Context-aware chunks
- Reranking layer
- Citation-based answers
- Evaluation framework
- Modular and extensible design

---

## Project Structure

robust-rag/
├── README.md
├── configs/
├── data/
├── notebooks/
├── src/
├── scripts/
└── tests/

---

## System Flow

### Ingestion

1. Load documents
2. Extract text
3. Clean & normalize
4. Split structurally
5. Generate contextual chunks
6. Create embeddings
7. Index (vector + BM25)

### Query

1. Rewrite query
2. Hybrid retrieval
3. Merge results
4. Rerank
5. Expand context
6. Generate answer
7. Return citations

---

## Tech Stack (Suggested)

- LLM: OpenAI / Anthropic
- Embeddings: OpenAI
- Vector DB: Qdrant / Pinecone / Weaviate
- BM25: Elasticsearch / Whoosh
- Framework: LangChain or LlamaIndex
- API: FastAPI
- Evaluation: RAGAS / TruLens

---

## Installation

```bash
git clone https://github.com/your-org/robust-rag.git
cd robust-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Ingest documents

```bash
python scripts/ingest.py --input ./data/raw
```

### Run API

```bash
python scripts/serve.py
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the data retention policy?"}'
```

---

## Evaluation

Evaluate using:

- Recall@k
- Precision@k
- Faithfulness
- Answer relevance
- Context relevance

Tools:
- RAGAS
- TruLens
- LangSmith

---

## Best Practices

- Always use hybrid retrieval
- Avoid fixed chunking blindly
- Use metadata filters
- Add reranking
- Always evaluate
- Always return citations

---

## Roadmap

v1:
- ingestion
- vector retrieval

v2:
- hybrid search
- metadata filters

v3:
- reranking
- evaluation

v4:
- agentic RAG
- multimodal RAG
- GraphRAG (optional)

---

## License

MIT

---

## Summary

Recommended formula:

**Structural chunking + contextualization + hybrid retrieval + reranking + evaluation**

This is the most robust baseline for modern RAG systems.
