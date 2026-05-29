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

require_python314() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    return
  fi
  if command -v python3.14 >/dev/null 2>&1; then
    return
  fi
  if [[ -x /home/linuxbrew/.linuxbrew/bin/python3.14 ]]; then
    return
  fi
  echo "application driver smokes require Python 3.14 for asyncpg/psycopg; set PYTHON_BIN to a compatible interpreter" >&2
  exit 1
}

require_command cargo
require_command node
require_command npm
require_command go
require_command javac
require_command mvn
require_python314

cargo test -p gpu_db_protocol --test tokio_postgres_smoke -- --color never
echo "application_driver_smoke_tokio_postgres=passed"

cargo test -p gpu_db_protocol --test sqlx_smoke -- --color never
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

echo "application_driver_smoke_scope=supported_sql_protocol_subset"
echo "application driver smoke gate passed"
