.PHONY: check format lint typecheck test

check: format lint typecheck test

format:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

test:
	uv run pytest
