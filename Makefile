SHELL := /bin/bash

.PHONY: all
all: lint test

.PHONY: format
format:
	ruff format src tests

.PHONY: lint
lint:
	ruff check src tests

.PHONY: test
test:
	pytest --maxfail=1 --disable-warnings -q
