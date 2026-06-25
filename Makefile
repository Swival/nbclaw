.PHONY: all install test lint format check clean dist

all: check

install:
	uv sync

test:
	uv run python -m pytest tests/ -v --durations=25

lint:
	uv run ruff check nbclaw/ tests/

format:
	uv run ruff format nbclaw/ tests/

check: lint
	uv run ruff format --check nbclaw/ tests/

clean:
	rm -rf dist/ build/ __pycache__ nbclaw/__pycache__ tests/__pycache__ .pytest_cache .ruff_cache
	find . -name '*.pyc' -delete

dist: clean
	uv build
