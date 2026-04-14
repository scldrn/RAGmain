# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/agentic_rag/`. Keep CLI entry points in `cli.py` and `__main__.py`, shared orchestration in `service.py`, compatibility wrappers in `app.py`, graph logic in `graph.py`, retrieval/reranking logic in `retrieval.py`, offline eval tooling in `evaluation.py`, console formatting in `presentation.py`, provider integrations in `providers.py`, typed failures in `errors.py`, document loading/cleaning in `documents.py`, and configuration in `settings.py`. Tests live in `tests/` and follow the module split, including `tests/test_graph.py`, `tests/test_service.py`, `tests/test_retrieval.py`, `tests/test_evaluation.py`, and `tests/test_service_integration.py`. Offline eval fixtures and checked-in baselines live in `tests/eval/`. Runtime configuration starts from `.env.example`; local Qdrant index data is written under `.cache/vectorstores/`.

## Build, Test, and Development Commands
Set up a local environment before making changes:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
pre-commit install
```

Use `python -m agentic_rag --question "..."` to run the CLI, add `--show-steps` for node-by-node tracing, and add `--verbose` when you need startup diagnostics or runtime logs. Export the graph with `python -m agentic_rag --question "test" --diagram graph.md`. Run `make check` before opening a PR; it is the canonical local CI gate. The equivalent individual commands are `ruff check .`, `ruff format --check .`, `mypy src/agentic_rag`, and `pytest --cov=src/agentic_rag --cov-report=term-missing --cov-fail-under=85`.
Use `python -m agentic_rag ingest` when you need an explicit pre-build step for `INGESTION_MODE=explicit`.
Use `make eval-baseline`, `make eval-hybrid`, and `make eval-compare` when you change retrieval, reranking, chunk cleaning, or other behavior that can affect the Phase 2 offline harness.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public functions, concise docstrings where behavior is not obvious, and `snake_case` for functions, variables, and test names. Classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Keep modules focused and small. Use `ruff format` for formatting and let `ruff check` manage import ordering and common correctness issues.

## Testing Guidelines
Tests use `pytest` with `src` on the Python path. Add or update tests whenever graph behavior, retrieval quality, provider resolution, document cleaning, or CLI-facing settings change. Prefer deterministic tests with fake models/tools or local deterministic embeddings over live provider calls. Name tests `test_<behavior>()` and group them by module under `tests/`. When changing retrieval behavior, update both targeted unit tests and the deeper service/eval coverage when appropriate; the repo currently maintains 100% coverage, so avoid adding behavior without meaningful assertions.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Add project README` and `Generalize RAG providers and model backends`. Keep commit messages in that style and scoped to one logical change. Pull requests should explain the user-visible impact, list any new environment variables or provider assumptions, link related issues, and include CLI examples or output snippets when behavior changes.

## Security & Configuration Tips
Do not hardcode API keys in code, tests, or scripts. Prefer `.env` values or `--chat-api-key-env` / `--embedding-api-key-env` so secrets stay outside the shell history. Treat `.cache/vectorstores/` as generated data and avoid committing local cache artifacts. Keep `INGESTION_MODE=auto` for the default CLI/dev flow; use `INGESTION_MODE=explicit` only when your service or setup path calls `AgenticRagService.ingest()` before queries. Corrupted local Qdrant caches are rebuildable and should not be committed or patched manually. `FETCH_TIMEOUT_SECONDS` controls per-request document fetch timeout; individual source failures should be handled by tests as partial-failure cases, not as hard repo-wide failures. `MAX_REWRITES` bounds retrieval rewrites, and tests that exercise weak-context flows should assert the explicit `insufficient_context` termination path. `RETRIEVAL_MODE=dense` and `RETRIEVAL_MODE=hybrid` keep distinct cache fingerprints, and changes to document cleaning or retrieval assembly can invalidate checked-in eval assumptions, so keep `tests/eval/` and the retrieval-focused tests in sync.
