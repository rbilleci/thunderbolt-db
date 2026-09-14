#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${PG_DUMPALL_SMOKE_OUT_DIR:-$ROOT_DIR/target/pg-dumpall-smoke}"
PG16_BIN="${PG16_BIN:-/usr/lib/postgresql/16/bin}"
SERVER_BIN="${GPU_DB_ENGINE_SERVER_BIN:-$ROOT_DIR/target/debug/thunderbolt-db-server}"
SOURCE_PORT="${PG_DUMPALL_SMOKE_SOURCE_PORT:-55457}"
RESTORE_PORT="${PG_DUMPALL_SMOKE_RESTORE_PORT:-55458}"

for pg_tool in psql pg_dumpall; do
  if [[ ! -x "$PG16_BIN/$pg_tool" ]]; then
    echo "PostgreSQL 16 tool is not executable: $PG16_BIN/$pg_tool" >&2
    exit 1
  fi
done
export PATH="$PG16_BIN:$PATH"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/wal"

source_pid=""
restore_pid=""

cleanup() {
  if [[ -n "$source_pid" ]]; then
    kill "$source_pid" 2>/dev/null || true
    wait "$source_pid" 2>/dev/null || true
  fi
  if [[ -n "$restore_pid" ]]; then
    kill "$restore_pid" 2>/dev/null || true
    wait "$restore_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

port_is_open() {
  local port="$1"
  (echo >"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
}

wait_for_port() {
  local port="$1"
  local child_pid="$2"
  for _ in $(seq 1 160); do
    if ! kill -0 "$child_pid" 2>/dev/null; then
      wait "$child_pid" 2>/dev/null || true
      echo "server process $child_pid exited before port $port became ready" >&2
      return 2
    fi
    if port_is_open "$port" && kill -0 "$child_pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  echo "server on port $port did not become ready" >&2
  return 1
}

start_server() {
  local port="$1"
  local log_name="$2"
  local wal_name="$3"
  local -n pid_ref="$4"
  if port_is_open "$port"; then
    echo "refusing to start test server: port $port is already occupied" >&2
    return 1
  fi
  GPU_DB_WAL_SEGMENT="$OUT_DIR/wal/$wal_name.wal" \
    "$SERVER_BIN" --listen "127.0.0.1:$port" \
    >"$OUT_DIR/$log_name" 2>&1 &
  pid_ref=$!
  local wait_status=0
  wait_for_port "$port" "$pid_ref" || wait_status=$?
  if (( wait_status == 2 )); then
    pid_ref=""
  fi
  (( wait_status == 0 ))
}

cd "$ROOT_DIR"
cargo build -p gpu_db_server --bin thunderbolt-db-server

start_server "$SOURCE_PORT" source-server.log source source_pid

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q <<'SQL'
CREATE ROLE global_reader WITH LOGIN;
COMMENT ON ROLE global_reader IS 'global metadata reader';
CREATE TABLESPACE global_space LOCATION '/tmp/gpu-db-global-space';
COMMENT ON TABLESPACE global_space IS 'global metadata tablespace';
GRANT CREATE ON TABLESPACE global_space TO global_reader;
SQL

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dumpall --globals-only --no-role-passwords \
  >"$OUT_DIR/globals.sql" 2>"$OUT_DIR/pg_dumpall.err"

grep -F "CREATE ROLE global_reader;" "$OUT_DIR/globals.sql" >/dev/null
grep -F "ALTER ROLE global_reader WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS;" "$OUT_DIR/globals.sql" >/dev/null
grep -F "COMMENT ON ROLE global_reader IS 'global metadata reader';" "$OUT_DIR/globals.sql" >/dev/null
grep -F "CREATE TABLESPACE global_space OWNER postgres LOCATION '/tmp/gpu-db-global-space';" "$OUT_DIR/globals.sql" >/dev/null
grep -F "COMMENT ON TABLESPACE global_space IS 'global metadata tablespace';" "$OUT_DIR/globals.sql" >/dev/null
grep -F "GRANT ALL ON TABLESPACE global_space TO global_reader;" "$OUT_DIR/globals.sql" >/dev/null

grep -Ev '^(CREATE ROLE postgres;|ALTER ROLE postgres WITH )' "$OUT_DIR/globals.sql" \
  >"$OUT_DIR/globals-restore.sql"

start_server "$RESTORE_PORT" restore-server.log restore restore_pid

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -f "$OUT_DIR/globals-restore.sql" \
  >"$OUT_DIR/restore.out" 2>"$OUT_DIR/restore.err"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q \
  -c '\du+' \
  -c '\db+' \
  >"$OUT_DIR/verify.out" 2>"$OUT_DIR/verify.err"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -A -t \
  -c "SELECT rolname, rolcanlogin FROM pg_catalog.pg_roles WHERE rolname = 'global_reader';" \
  >"$OUT_DIR/role-verify.out" 2>"$OUT_DIR/role-verify.err"

grep -F "global_reader" "$OUT_DIR/verify.out" >/dev/null
grep -F "global metadata reader" "$OUT_DIR/verify.out" >/dev/null
grep -F "global_space" "$OUT_DIR/verify.out" >/dev/null
grep -F "/tmp/gpu-db-global-space" "$OUT_DIR/verify.out" >/dev/null
grep -F "global metadata tablespace" "$OUT_DIR/verify.out" >/dev/null
grep -F "global_reader=C/postgres" "$OUT_DIR/verify.out" >/dev/null
grep -Fx "global_reader|t" "$OUT_DIR/role-verify.out" >/dev/null

printf 'pg_dumpall_globals_restore=passed\n'
printf 'pg_dumpall_globals_scope=roles_login_attribute_tablespaces_comments_tablespace_acls_no_role_passwords\n'
printf 'pg_dumpall_globals_gap_bootstrap_role_restore=filtered_existing_bootstrap_role\n'
printf 'pg_dumpall_globals_gap_database_acl_restore=not_emitted_by_globals_only\n'
