# Makefile for IDS706 Powerlifting Project

.PHONY: all test lint clean

all: test

# Run all unit tests

test:
	python test_powerlifting.py

# Lint Python files with flake8 (if installed)
lint:
	flake8 powerlifting.py test_powerlifting.py || echo "flake8 not installed"

# Remove Python cache and temporary files
clean:
	del /s /q __pycache__ 2>nul || true
	del /s /q *.pyc 2>nul || true
	del /s /q *.pyo 2>nul || true
	del /s /q .pytest_cache 2>nul || true
	del /s /q .mypy_cache 2>nul || true
