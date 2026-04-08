.PHONY: install-dev lint format-check typecheck test eval-baseline eval-hybrid eval-compare check

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
RUFF ?= $(VENV)/bin/ruff
MYPY ?= $(VENV)/bin/mypy
PYTEST ?= $(VENV)/bin/pytest

install-dev:
	$(PIP) install -e '.[dev]'

lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

typecheck:
	$(MYPY) src/agentic_rag

test:
	$(PYTEST) --cov=src/agentic_rag --cov-report=term-missing --cov-fail-under=85

eval-baseline:
	$(PYTHON) -m agentic_rag.evaluation --check-baseline

eval-hybrid:
	$(PYTHON) -m agentic_rag.evaluation --retrieval-mode hybrid --check-baseline

eval-compare:
	$(PYTHON) -m agentic_rag.evaluation --retrieval-mode hybrid --compare-to tests/eval/baseline_dense.json

check: lint format-check typecheck test
