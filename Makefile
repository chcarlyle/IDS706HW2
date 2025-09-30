# Makefile for IDS706 Powerlifting Project

.PHONY: all test lint format clean

all: test

# Files to operate on by default (can be overridden with FILE=...)
TARGETS := powerlifting.py test_powerlifting.py
FILE ?= $(TARGETS)

# Tools (can be overridden in environment)
BLACK ?= black
FLAKE8 ?= flake8

# Flake8 ignore codes (comma-separated)
FLAKE8_IGNORE ?= E501,W503

# Run all unit tests
test:
	python test_powerlifting.py

# Format Python files with black. Usage:
#   make format          # formats default TARGETS
#   make format FILE=abc.py  # formats a specific file
format:
	@if ! command -v $(BLACK) >/dev/null 2>&1; then \
		echo "black not found; install with: pip install black"; exit 1; \
	fi
	$(BLACK) $(FILE)

# Lint Python files with flake8. Usage:
#   make lint                    # lints default TARGETS
#   make lint FILE=abc.py        # lints a specific file
# You can set ignore codes with FLAKE8_IGNORE, e.g.:
#   make lint FLAKE8_IGNORE=E203,E266
lint:
	@if ! command -v $(FLAKE8) >/dev/null 2>&1; then \
		echo "flake8 not found; install with: pip install flake8"; exit 1; \
	fi
	$(FLAKE8) --ignore=$(FLAKE8_IGNORE) $(FILE)

# Remove Python cache and temporary files (Unix-friendly)
clean:
	rm -rf __pycache__
	rm -f *.pyc *.pyo
	rm -rf .pytest_cache .mypy_cache
