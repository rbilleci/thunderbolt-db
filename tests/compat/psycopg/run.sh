#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-}

python_has_pip() {
  local python="$1"
  if [[ "$python" == */* ]]; then
    [[ -x "$python" ]] || return 1
  elif ! command -v "$python" >/dev/null 2>&1; then
    return 1
  fi
  "$python" -m pip --version >/dev/null 2>&1
}

if [[ -z "$PYTHON_BIN" ]]; then
  PATH_PYTHON=$(command -v python3.14 2>/dev/null || true)
  if [[ -n "$PATH_PYTHON" ]] && python_has_pip "$PATH_PYTHON"; then
    PYTHON_BIN=$PATH_PYTHON
  elif python_has_pip /home/linuxbrew/.linuxbrew/bin/python3.14; then
    PYTHON_BIN=/home/linuxbrew/.linuxbrew/bin/python3.14
  else
    echo "psycopg smoke requires Python 3.14 with pip; set PYTHON_BIN to a compatible interpreter" >&2
    exit 1
  fi
elif ! python_has_pip "$PYTHON_BIN"; then
  echo "psycopg smoke PYTHON_BIN must name an executable Python 3.14 interpreter with pip" >&2
  exit 1
fi

DEPS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-psycopg-deps.XXXXXX")
cleanup() {
  rm -rf "$DEPS_DIR"
}
trap cleanup EXIT

"$PYTHON_BIN" -m pip install --quiet --target "$DEPS_DIR" -r "$SCRIPT_DIR/requirements.txt"
PYTHONPATH="$DEPS_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/smoke.py" "$REPO_ROOT"
