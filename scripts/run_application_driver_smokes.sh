#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "required command not found for application driver smokes: $name" >&2
    exit 1
  fi
}

python_has_pip() {
  local python="$1"
  if [[ "$python" == */* ]]; then
    [[ -x "$python" ]] || return 1
  elif ! command -v "$python" >/dev/null 2>&1; then
    return 1
  fi
  "$python" -m pip --version >/dev/null 2>&1
}

require_python314() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if python_has_pip "$PYTHON_BIN"; then
      export PYTHON_BIN
      return
    fi
    echo "application driver smoke PYTHON_BIN must name an executable Python 3.14 interpreter with pip" >&2
    exit 1
  fi
  local path_python
  path_python=$(command -v python3.14 2>/dev/null || true)
  if [[ -n "$path_python" ]] && "$path_python" -m pip --version >/dev/null 2>&1; then
    PYTHON_BIN=$path_python
    export PYTHON_BIN
    return
  fi
  if [[ -x /home/linuxbrew/.linuxbrew/bin/python3.14 ]] \
    && /home/linuxbrew/.linuxbrew/bin/python3.14 -m pip --version >/dev/null 2>&1; then
    PYTHON_BIN=/home/linuxbrew/.linuxbrew/bin/python3.14
    export PYTHON_BIN
    return
  fi
  echo "application driver smokes require Python 3.14 for asyncpg/psycopg; set PYTHON_BIN to a compatible interpreter" >&2
  exit 1
}

require_canonical_driver_source() {
  local driver="$1"
  if rg -q 'gpu_db_protocol|gpu-db-server' "tests/compat/$driver"; then
    echo "canonical application driver still names the legacy server: $driver" >&2
    exit 1
  fi
  if ! rg -q 'gpu_db_server|gpu-db-engine-server' "tests/compat/$driver"; then
    echo "canonical application driver does not name gpu-db-engine-server: $driver" >&2
    exit 1
  fi
}

require_command cargo
require_command node
require_command npm
require_command go
require_command javac
require_command mvn
require_command rg
require_python314

for canonical_driver in node-postgres asyncpg psycopg pgx jdbc r2dbc; do
  require_canonical_driver_source "$canonical_driver"
done

cargo test -p gpu_db_server --test tokio_postgres_smoke -- --color never
echo "application_driver_smoke_tokio_postgres=passed"

cargo test -p gpu_db_server --test sqlx_smoke -- --color never
echo "application_driver_smoke_sqlx=passed"

tests/compat/node-postgres/run.sh
echo "application_driver_smoke_node_postgres=passed"

tests/compat/asyncpg/run.sh
echo "application_driver_smoke_asyncpg=passed"

tests/compat/psycopg/run.sh
echo "application_driver_smoke_psycopg=passed"

tests/compat/pgx/run.sh
echo "application_driver_smoke_pgx=passed"

tests/compat/jdbc/run.sh
echo "application_driver_smoke_jdbc=passed"

tests/compat/r2dbc/run.sh
echo "application_driver_smoke_r2dbc=passed"
echo "application_driver_smoke_r2dbc_target=canonical_gpu_catalog"

echo "application_driver_smoke_canonical_targets=tokio-postgres,sqlx,node-postgres,asyncpg,psycopg,pgx,jdbc,r2dbc"
echo "application_driver_smoke_scope=supported_sql_protocol_subset"
echo "application driver smoke gate passed"
