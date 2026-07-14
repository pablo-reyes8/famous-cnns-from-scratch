.PHONY: help install install-dev test lint format type check build clean docker-build docker-smoke

help:
	@echo "install       Install the library"
	@echo "install-dev   Install library and development tools"
	@echo "test          Run tests with coverage"
	@echo "lint          Run Ruff checks"
	@echo "format        Format maintained Python code"
	@echo "type          Run mypy on the public package"
	@echo "check         Run lint, format check, tests, and package build"
	@echo "docker-build  Build the local container"
	@echo "docker-smoke  Run a synthetic LeNet training container"

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

test:
	python -m pytest --cov=famous_cnns --cov-report=term-missing --cov-fail-under=50

lint:
	ruff check famous_cnns scripts tests */scripts

format:
	ruff format famous_cnns scripts tests */scripts

type:
	mypy famous_cnns

check: lint
	ruff format --check famous_cnns scripts tests */scripts
	python -m pytest
	python -m build

build:
	python -m build

clean:
	rm -rf build dist htmlcov .coverage coverage.xml .pytest_cache .ruff_cache .mypy_cache

docker-build:
	docker build -t famous-cnns:local .

docker-smoke:
	docker compose --profile test run --rm smoke-test
