.PHONY: install-dev lint format-check typecheck test check

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

check: lint format-check typecheck test
