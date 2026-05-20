#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-}

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.14 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3.14)
  elif [[ -x /home/linuxbrew/.linuxbrew/bin/python3.14 ]]; then
    PYTHON_BIN=/home/linuxbrew/.linuxbrew/bin/python3.14
  else
    echo "psycopg smoke requires Python 3.14 with pip; set PYTHON_BIN to a compatible interpreter" >&2
    exit 1
  fi
fi

DEPS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-psycopg-deps.XXXXXX")
cleanup() {
  rm -rf "$DEPS_DIR"
}
trap cleanup EXIT

"$PYTHON_BIN" -m pip install --quiet --target "$DEPS_DIR" -r "$SCRIPT_DIR/requirements.txt"
PYTHONPATH="$DEPS_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/smoke.py" "$REPO_ROOT"
