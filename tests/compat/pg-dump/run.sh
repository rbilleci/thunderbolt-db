#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${PG_DUMP_SMOKE_OUT_DIR:-$ROOT_DIR/target/pg-dump-smoke}"
SOURCE_PORT="${PG_DUMP_SMOKE_SOURCE_PORT:-55444}"
RESTORE_PORT="${PG_DUMP_SMOKE_RESTORE_PORT:-55445}"
CUSTOM_RESTORE_PORT="${PG_DUMP_SMOKE_CUSTOM_RESTORE_PORT:-55446}"
DIRECTORY_RESTORE_PORT="${PG_DUMP_SMOKE_DIRECTORY_RESTORE_PORT:-55447}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

source_pid=""
restore_pid=""
custom_restore_pid=""
directory_restore_pid=""

cleanup() {
  if [[ -n "$source_pid" ]]; then
    kill "$source_pid" 2>/dev/null || true
    wait "$source_pid" 2>/dev/null || true
  fi
  if [[ -n "$restore_pid" ]]; then
    kill "$restore_pid" 2>/dev/null || true
    wait "$restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$custom_restore_pid" ]]; then
    kill "$custom_restore_pid" 2>/dev/null || true
    wait "$custom_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$directory_restore_pid" ]]; then
    kill "$directory_restore_pid" 2>/dev/null || true
    wait "$directory_restore_pid" 2>/dev/null || true
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
CREATE TABLE accounts (id int4, name text);
INSERT INTO accounts (id, name) VALUES (1, 'Ada');
INSERT INTO accounts (id, name) VALUES (2, 'Grace');
CREATE TABLE events (event_id int4, note text);
INSERT INTO events (event_id, note) VALUES (10, 'created');
INSERT INTO events (event_id, note) VALUES (11, 'updated');
SQL

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dump --schema=public --no-owner --no-privileges --format=plain \
  >"$OUT_DIR/dump.sql" 2>"$OUT_DIR/pg_dump.err"

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dump --schema=public --no-owner --no-privileges --format=custom \
  --file="$OUT_DIR/dump.custom" 2>"$OUT_DIR/pg_dump_custom.err"

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dump --schema=public --no-owner --no-privileges --format=directory \
  --file="$OUT_DIR/dump.dir" 2>"$OUT_DIR/pg_dump_directory.err"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/restore-server.log" 2>&1 &
restore_pid=$!
wait_for_port "$RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -f "$OUT_DIR/dump.sql" \
  >"$OUT_DIR/restore.out" 2>"$OUT_DIR/restore.err"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/verify.out" 2>"$OUT_DIR/verify.err"

cat >"$OUT_DIR/verify.expected" <<'EOF'
1|Ada
2|Grace
10|created
11|updated
EOF

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/verify.out"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$CUSTOM_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/custom-restore-server.log" 2>&1 &
custom_restore_pid=$!
wait_for_port "$CUSTOM_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$CUSTOM_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.custom" \
  >"$OUT_DIR/custom-restore.out" 2>"$OUT_DIR/custom-restore.err"

PGHOST=127.0.0.1 PGPORT="$CUSTOM_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/custom-verify.out" 2>"$OUT_DIR/custom-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/custom-verify.out"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$DIRECTORY_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/directory-restore-server.log" 2>&1 &
directory_restore_pid=$!
wait_for_port "$DIRECTORY_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$DIRECTORY_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.dir" \
  >"$OUT_DIR/directory-restore.out" 2>"$OUT_DIR/directory-restore.err"

PGHOST=127.0.0.1 PGPORT="$DIRECTORY_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/directory-verify.out" 2>"$OUT_DIR/directory-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/directory-verify.out"

pg_restore --list "$OUT_DIR/dump.custom" >"$OUT_DIR/dump.custom.toc"
pg_restore --list "$OUT_DIR/dump.dir" >"$OUT_DIR/dump.dir.toc"

grep -F "COPY public.accounts (id, name) FROM stdin;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COPY public.events (event_id, note) FROM stdin;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE SCHEMA public;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE TABLE public.accounts (" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE TABLE public.events (" "$OUT_DIR/dump.sql" >/dev/null
grep -F "SCHEMA - public" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE public accounts" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE public events" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE DATA public accounts" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE DATA public events" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "SCHEMA - public" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE public accounts" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE public events" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE DATA public accounts" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE DATA public events" "$OUT_DIR/dump.dir.toc" >/dev/null

echo "pg_dump_plain_public_schema_restore=passed"
echo "pg_dump_custom_public_schema_pg_restore=passed"
echo "pg_dump_directory_public_schema_pg_restore=passed"
echo "dump_file=$OUT_DIR/dump.sql"
echo "custom_dump_file=$OUT_DIR/dump.custom"
echo "directory_dump_dir=$OUT_DIR/dump.dir"
