.PHONY: install install-dev test cov lint typecheck fmt check clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff format .
	ruff check --fix .

typecheck:
	mypy

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
