#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

echo "[1/3] Running unit tests"
"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_*.py' -v

echo "[2/3] Running one benchmark experiment"
"${PYTHON_BIN}" run_experiment.py

echo "[3/3] Finished. Check results/ for logs and plots."
