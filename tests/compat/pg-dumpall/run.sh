#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${PG_DUMPALL_SMOKE_OUT_DIR:-$ROOT_DIR/target/pg-dumpall-smoke}"
SOURCE_PORT="${PG_DUMPALL_SMOKE_SOURCE_PORT:-55457}"
RESTORE_PORT="${PG_DUMPALL_SMOKE_RESTORE_PORT:-55458}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

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

wait_for_port() {
  local port="$1"
  for _ in $(seq 1 160); do
    if (echo >"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  echo "server on port $port did not become ready" >&2
  return 1
}

cd "$ROOT_DIR"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$SOURCE_PORT" --shared-catalog \
  >"$OUT_DIR/source-server.log" 2>&1 &
source_pid=$!
wait_for_port "$SOURCE_PORT"

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
grep -F "COMMENT ON ROLE global_reader IS 'global metadata reader';" "$OUT_DIR/globals.sql" >/dev/null
grep -F "CREATE TABLESPACE global_space OWNER postgres LOCATION '/tmp/gpu-db-global-space';" "$OUT_DIR/globals.sql" >/dev/null
grep -F "COMMENT ON TABLESPACE global_space IS 'global metadata tablespace';" "$OUT_DIR/globals.sql" >/dev/null
grep -F "GRANT ALL ON TABLESPACE global_space TO global_reader;" "$OUT_DIR/globals.sql" >/dev/null

grep -Ev '^(CREATE ROLE postgres;|ALTER ROLE .* WITH )' "$OUT_DIR/globals.sql" \
  >"$OUT_DIR/globals-restore.sql"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/restore-server.log" 2>&1 &
restore_pid=$!
wait_for_port "$RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -f "$OUT_DIR/globals-restore.sql" \
  >"$OUT_DIR/restore.out" 2>"$OUT_DIR/restore.err"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q \
  -c '\du+' \
  -c '\db+' \
  >"$OUT_DIR/verify.out" 2>"$OUT_DIR/verify.err"

grep -F "global_reader" "$OUT_DIR/verify.out" >/dev/null
grep -F "global metadata reader" "$OUT_DIR/verify.out" >/dev/null
grep -F "global_space" "$OUT_DIR/verify.out" >/dev/null
grep -F "/tmp/gpu-db-global-space" "$OUT_DIR/verify.out" >/dev/null
grep -F "global metadata tablespace" "$OUT_DIR/verify.out" >/dev/null
grep -F "global_reader=C/postgres" "$OUT_DIR/verify.out" >/dev/null

printf 'pg_dumpall_globals_restore=passed\n'
printf 'pg_dumpall_globals_scope=roles_tablespaces_comments_tablespace_acls_no_role_passwords\n'
printf 'pg_dumpall_globals_gap_bootstrap_role_restore=filtered_existing_bootstrap_role\n'
printf 'pg_dumpall_globals_gap_database_acl_restore=not_emitted_by_globals_only\n'
