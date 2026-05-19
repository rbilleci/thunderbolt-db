#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${PG_DUMP_SMOKE_OUT_DIR:-$ROOT_DIR/target/pg-dump-smoke}"
SOURCE_PORT="${PG_DUMP_SMOKE_SOURCE_PORT:-55444}"
RESTORE_PORT="${PG_DUMP_SMOKE_RESTORE_PORT:-55445}"
CUSTOM_RESTORE_PORT="${PG_DUMP_SMOKE_CUSTOM_RESTORE_PORT:-55446}"
DIRECTORY_RESTORE_PORT="${PG_DUMP_SMOKE_DIRECTORY_RESTORE_PORT:-55447}"
TAR_RESTORE_PORT="${PG_DUMP_SMOKE_TAR_RESTORE_PORT:-55448}"
PARALLEL_DIRECTORY_RESTORE_PORT="${PG_DUMP_SMOKE_PARALLEL_DIRECTORY_RESTORE_PORT:-55449}"
CLEAN_RESTORE_PORT="${PG_DUMP_SMOKE_CLEAN_RESTORE_PORT:-55450}"
INSERT_RESTORE_PORT="${PG_DUMP_SMOKE_INSERT_RESTORE_PORT:-55451}"
SPLIT_RESTORE_PORT="${PG_DUMP_SMOKE_SPLIT_RESTORE_PORT:-55452}"
CUSTOM_SPLIT_RESTORE_PORT="${PG_DUMP_SMOKE_CUSTOM_SPLIT_RESTORE_PORT:-55453}"
DIRECTORY_SPLIT_RESTORE_PORT="${PG_DUMP_SMOKE_DIRECTORY_SPLIT_RESTORE_PORT:-55454}"
TAR_SPLIT_RESTORE_PORT="${PG_DUMP_SMOKE_TAR_SPLIT_RESTORE_PORT:-55455}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

source_pid=""
restore_pid=""
custom_restore_pid=""
directory_restore_pid=""
tar_restore_pid=""
parallel_directory_restore_pid=""
clean_restore_pid=""
insert_restore_pid=""
split_restore_pid=""
custom_split_restore_pid=""
directory_split_restore_pid=""
tar_split_restore_pid=""

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
  if [[ -n "$tar_restore_pid" ]]; then
    kill "$tar_restore_pid" 2>/dev/null || true
    wait "$tar_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$parallel_directory_restore_pid" ]]; then
    kill "$parallel_directory_restore_pid" 2>/dev/null || true
    wait "$parallel_directory_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$clean_restore_pid" ]]; then
    kill "$clean_restore_pid" 2>/dev/null || true
    wait "$clean_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$insert_restore_pid" ]]; then
    kill "$insert_restore_pid" 2>/dev/null || true
    wait "$insert_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$split_restore_pid" ]]; then
    kill "$split_restore_pid" 2>/dev/null || true
    wait "$split_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$custom_split_restore_pid" ]]; then
    kill "$custom_split_restore_pid" 2>/dev/null || true
    wait "$custom_split_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$directory_split_restore_pid" ]]; then
    kill "$directory_split_restore_pid" 2>/dev/null || true
    wait "$directory_split_restore_pid" 2>/dev/null || true
  fi
  if [[ -n "$tar_split_restore_pid" ]]; then
    kill "$tar_split_restore_pid" 2>/dev/null || true
    wait "$tar_split_restore_pid" 2>/dev/null || true
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
CREATE TABLE accounts (id int4 PRIMARY KEY, name text DEFAULT 'unknown'::text, tier int4 DEFAULT 7);
INSERT INTO accounts (id, name) VALUES (1, 'Ada');
INSERT INTO accounts (id) VALUES (2);
CREATE INDEX accounts_name_idx ON accounts (name);
COMMENT ON TABLE public.accounts IS 'accounts table';
COMMENT ON COLUMN public.accounts.name IS 'account display name';
COMMENT ON INDEX public.accounts_name_idx IS 'accounts name lookup';
COMMENT ON CONSTRAINT accounts_pkey ON public.accounts IS 'accounts row identity';
CREATE TABLE events (event_id int4, note text);
INSERT INTO events (event_id, note) VALUES (10, 'created');
INSERT INTO events (event_id, note) VALUES (11, 'updated');
CREATE INDEX events_note_idx ON events (note);
COMMENT ON TABLE public.events IS 'events table';
COMMENT ON COLUMN public.events.note IS 'event note';
COMMENT ON INDEX public.events_note_idx IS 'events note lookup';
CREATE VIEW public.account_lookup AS SELECT id, name FROM accounts WHERE id > 1 ORDER BY id;
COMMENT ON VIEW public.account_lookup IS 'active account lookup';
CREATE MATERIALIZED VIEW public.account_snapshot AS SELECT id, name FROM accounts ORDER BY id;
COMMENT ON MATERIALIZED VIEW public.account_snapshot IS 'account snapshot';
CREATE SEQUENCE public.account_seq;
COMMENT ON SEQUENCE public.account_seq IS 'account sequence';
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

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dump --schema=public --no-owner --no-privileges --format=tar \
  --file="$OUT_DIR/dump.tar" 2>"$OUT_DIR/pg_dump_tar.err"

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dump --schema=public --no-owner --no-privileges --format=plain \
  --inserts --rows-per-insert=2 \
  >"$OUT_DIR/dump-inserts.sql" 2>"$OUT_DIR/pg_dump_inserts.err"

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dump --schema=public --no-owner --no-privileges --format=plain \
  --schema-only \
  >"$OUT_DIR/dump-schema.sql" 2>"$OUT_DIR/pg_dump_schema.err"

PGHOST=127.0.0.1 PGPORT="$SOURCE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_dump --schema=public --no-owner --no-privileges --format=plain \
  --data-only \
  >"$OUT_DIR/dump-data.sql" 2>"$OUT_DIR/pg_dump_data.err"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/restore-server.log" 2>&1 &
restore_pid=$!
wait_for_port "$RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -f "$OUT_DIR/dump.sql" \
  >"$OUT_DIR/restore.out" 2>"$OUT_DIR/restore.err"

PGHOST=127.0.0.1 PGPORT="$RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/verify.out" 2>"$OUT_DIR/verify.err"

cat >"$OUT_DIR/verify.expected" <<'EOF'
1|Ada|7
2|unknown|7
10|created
11|updated
EOF

cat >"$OUT_DIR/index-verify.expected" <<'EOF'
accounts|accounts_name_idx|CREATE INDEX accounts_name_idx ON public.accounts USING btree (name)
accounts|accounts_pkey|CREATE UNIQUE INDEX accounts_pkey ON public.accounts USING btree (id)
events|events_note_idx|CREATE INDEX events_note_idx ON public.events USING btree (note)
EOF

cat >"$OUT_DIR/constraint-verify.expected" <<'EOF'
public|accounts|accounts_pkey|p
EOF

cat >"$OUT_DIR/comment-verify.expected" <<'EOF'
public|accounts_name_idx|i||accounts name lookup
public|events_note_idx|i||events note lookup
public|accounts|r||accounts table
public|accounts|r|name|account display name
public|events|r||events table
public|events|r|note|event note
public|account_lookup|v||active account lookup
public|account_snapshot|m||account snapshot
public|account_seq|s||account sequence
EOF

cat >"$OUT_DIR/constraint-comment-verify.expected" <<'EOF'
public|accounts|accounts_pkey|accounts row identity
EOF

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/verify.out"

verify_indexes() {
  local port="$1"
  local prefix="$2"
  PGHOST=127.0.0.1 PGPORT="$port" PGDATABASE=postgres PGUSER=postgres \
    psql -v ON_ERROR_STOP=1 -X -A -t \
    -c "SELECT tablename, indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname = 'public' ORDER BY tablename, indexname;" \
    >"$OUT_DIR/${prefix}-index-verify.out" 2>"$OUT_DIR/${prefix}-index-verify.err"
  diff -u "$OUT_DIR/index-verify.expected" "$OUT_DIR/${prefix}-index-verify.out"
}

verify_indexes "$RESTORE_PORT" "restore"

verify_constraints() {
  local port="$1"
  local prefix="$2"
  PGHOST=127.0.0.1 PGPORT="$port" PGDATABASE=postgres PGUSER=postgres \
    psql -v ON_ERROR_STOP=1 -X -A -t \
    -c "SELECT n.nspname, c.relname, con.conname, con.contype FROM pg_catalog.pg_constraint con JOIN pg_catalog.pg_class c ON c.oid = con.conrelid JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' ORDER BY c.relname, con.conname;" \
    >"$OUT_DIR/${prefix}-constraint-verify.out" 2>"$OUT_DIR/${prefix}-constraint-verify.err"
  diff -u "$OUT_DIR/constraint-verify.expected" "$OUT_DIR/${prefix}-constraint-verify.out"
}

verify_constraints "$RESTORE_PORT" "restore"

verify_comments() {
  local port="$1"
  local prefix="$2"
  PGHOST=127.0.0.1 PGPORT="$port" PGDATABASE=postgres PGUSER=postgres \
    psql -v ON_ERROR_STOP=1 -X -A -t \
    -c "SELECT n.nspname, c.relname, c.relkind, a.attname, d.description FROM pg_catalog.pg_description d JOIN pg_catalog.pg_class c ON c.oid = d.objoid JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid WHERE n.nspname = 'public' AND c.relkind IN ('r','i','v','m','s') ORDER BY c.relkind, c.relname, d.objsubid;" \
    >"$OUT_DIR/${prefix}-comment-verify.out" 2>"$OUT_DIR/${prefix}-comment-verify.err"
  diff -u "$OUT_DIR/comment-verify.expected" "$OUT_DIR/${prefix}-comment-verify.out"
}

verify_comments "$RESTORE_PORT" "restore"

verify_constraint_comments() {
  local port="$1"
  local prefix="$2"
  PGHOST=127.0.0.1 PGPORT="$port" PGDATABASE=postgres PGUSER=postgres \
    psql -v ON_ERROR_STOP=1 -X -A -t \
    -c "SELECT n.nspname, c.relname, con.conname, d.description FROM pg_catalog.pg_description d JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid JOIN pg_catalog.pg_class c ON c.oid = con.conrelid JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' ORDER BY c.relname, con.conname;" \
    >"$OUT_DIR/${prefix}-constraint-comment-verify.out" 2>"$OUT_DIR/${prefix}-constraint-comment-verify.err"
  diff -u "$OUT_DIR/constraint-comment-verify.expected" "$OUT_DIR/${prefix}-constraint-comment-verify.out"
}

verify_constraint_comments "$RESTORE_PORT" "restore"

cat >"$OUT_DIR/view-verify.expected" <<'EOF'
public|account_lookup|postgres|SELECT id, name FROM accounts WHERE id > 1 ORDER BY id
2|unknown
EOF

verify_views() {
  local port="$1"
  local prefix="$2"
  PGHOST=127.0.0.1 PGPORT="$port" PGDATABASE=postgres PGUSER=postgres \
    psql -v ON_ERROR_STOP=1 -X -A -t \
    -c "SELECT schemaname, viewname, viewowner, definition FROM pg_catalog.pg_views WHERE schemaname = 'public' ORDER BY viewname;" \
    -c "SELECT * FROM account_lookup;" \
    >"$OUT_DIR/${prefix}-view-verify.out" 2>"$OUT_DIR/${prefix}-view-verify.err"
  diff -u "$OUT_DIR/view-verify.expected" "$OUT_DIR/${prefix}-view-verify.out"
}

verify_views "$RESTORE_PORT" "restore"

cat >"$OUT_DIR/materialized-view-verify.expected" <<'EOF'
1|Ada
2|unknown
EOF
touch "$OUT_DIR/materialized-view-empty.expected"

verify_materialized_views() {
  local port="$1"
  local prefix="$2"
  local expected="${3:-$OUT_DIR/materialized-view-verify.expected}"
  PGHOST=127.0.0.1 PGPORT="$port" PGDATABASE=postgres PGUSER=postgres \
    psql -v ON_ERROR_STOP=1 -X -A -t \
    -c "SELECT * FROM account_snapshot;" \
    >"$OUT_DIR/${prefix}-materialized-view-verify.out" 2>"$OUT_DIR/${prefix}-materialized-view-verify.err"
  diff -u "$expected" "$OUT_DIR/${prefix}-materialized-view-verify.out"
}

verify_materialized_views "$RESTORE_PORT" "restore"

cat >"$OUT_DIR/sequence-verify.expected" <<'EOF'
public|account_seq|s|p
EOF

verify_sequences() {
  local port="$1"
  local prefix="$2"
  PGHOST=127.0.0.1 PGPORT="$port" PGDATABASE=postgres PGUSER=postgres \
    psql -v ON_ERROR_STOP=1 -X -A -t \
    -c "SELECT c.oid, n.nspname, c.relname, c.relkind, c.relpersistence FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relkind = 's' ORDER BY c.relname;" \
    2>"$OUT_DIR/${prefix}-sequence-verify.err" \
    | cut -d'|' -f2- >"$OUT_DIR/${prefix}-sequence-verify.out"
  diff -u "$OUT_DIR/sequence-verify.expected" "$OUT_DIR/${prefix}-sequence-verify.out"
}

verify_sequences "$RESTORE_PORT" "restore"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$CUSTOM_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/custom-restore-server.log" 2>&1 &
custom_restore_pid=$!
wait_for_port "$CUSTOM_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$CUSTOM_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.custom" \
  >"$OUT_DIR/custom-restore.out" 2>"$OUT_DIR/custom-restore.err"

PGHOST=127.0.0.1 PGPORT="$CUSTOM_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/custom-verify.out" 2>"$OUT_DIR/custom-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/custom-verify.out"
verify_indexes "$CUSTOM_RESTORE_PORT" "custom"
verify_constraints "$CUSTOM_RESTORE_PORT" "custom"
verify_comments "$CUSTOM_RESTORE_PORT" "custom"
verify_constraint_comments "$CUSTOM_RESTORE_PORT" "custom"
verify_views "$CUSTOM_RESTORE_PORT" "custom"
verify_materialized_views "$CUSTOM_RESTORE_PORT" "custom"
verify_sequences "$CUSTOM_RESTORE_PORT" "custom"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$DIRECTORY_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/directory-restore-server.log" 2>&1 &
directory_restore_pid=$!
wait_for_port "$DIRECTORY_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$DIRECTORY_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.dir" \
  >"$OUT_DIR/directory-restore.out" 2>"$OUT_DIR/directory-restore.err"

PGHOST=127.0.0.1 PGPORT="$DIRECTORY_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/directory-verify.out" 2>"$OUT_DIR/directory-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/directory-verify.out"
verify_indexes "$DIRECTORY_RESTORE_PORT" "directory"
verify_constraints "$DIRECTORY_RESTORE_PORT" "directory"
verify_comments "$DIRECTORY_RESTORE_PORT" "directory"
verify_constraint_comments "$DIRECTORY_RESTORE_PORT" "directory"
verify_views "$DIRECTORY_RESTORE_PORT" "directory"
verify_materialized_views "$DIRECTORY_RESTORE_PORT" "directory"
verify_sequences "$DIRECTORY_RESTORE_PORT" "directory"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$TAR_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/tar-restore-server.log" 2>&1 &
tar_restore_pid=$!
wait_for_port "$TAR_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$TAR_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.tar" \
  >"$OUT_DIR/tar-restore.out" 2>"$OUT_DIR/tar-restore.err"

PGHOST=127.0.0.1 PGPORT="$TAR_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/tar-verify.out" 2>"$OUT_DIR/tar-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/tar-verify.out"
verify_indexes "$TAR_RESTORE_PORT" "tar"
verify_constraints "$TAR_RESTORE_PORT" "tar"
verify_comments "$TAR_RESTORE_PORT" "tar"
verify_constraint_comments "$TAR_RESTORE_PORT" "tar"
verify_views "$TAR_RESTORE_PORT" "tar"
verify_materialized_views "$TAR_RESTORE_PORT" "tar"
verify_sequences "$TAR_RESTORE_PORT" "tar"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$PARALLEL_DIRECTORY_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/directory-parallel-restore-server.log" 2>&1 &
parallel_directory_restore_pid=$!
wait_for_port "$PARALLEL_DIRECTORY_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$PARALLEL_DIRECTORY_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --jobs=2 --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.dir" \
  >"$OUT_DIR/directory-parallel-restore.out" 2>"$OUT_DIR/directory-parallel-restore.err"

PGHOST=127.0.0.1 PGPORT="$PARALLEL_DIRECTORY_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/directory-parallel-verify.out" 2>"$OUT_DIR/directory-parallel-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/directory-parallel-verify.out"
verify_indexes "$PARALLEL_DIRECTORY_RESTORE_PORT" "directory-parallel"
verify_constraints "$PARALLEL_DIRECTORY_RESTORE_PORT" "directory-parallel"
verify_comments "$PARALLEL_DIRECTORY_RESTORE_PORT" "directory-parallel"
verify_constraint_comments "$PARALLEL_DIRECTORY_RESTORE_PORT" "directory-parallel"
verify_views "$PARALLEL_DIRECTORY_RESTORE_PORT" "directory-parallel"
verify_materialized_views "$PARALLEL_DIRECTORY_RESTORE_PORT" "directory-parallel"
verify_sequences "$PARALLEL_DIRECTORY_RESTORE_PORT" "directory-parallel"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$CLEAN_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/clean-restore-server.log" 2>&1 &
clean_restore_pid=$!
wait_for_port "$CLEAN_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$CLEAN_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q <<'SQL'
CREATE TABLE accounts (id int4, name text);
INSERT INTO accounts (id, name) VALUES (99, 'stale account');
CREATE VIEW public.account_lookup AS SELECT id, name FROM accounts WHERE id = 99 ORDER BY id;
CREATE MATERIALIZED VIEW public.account_snapshot AS SELECT id, name FROM accounts ORDER BY id;
CREATE SEQUENCE public.account_seq;
CREATE TABLE events (event_id int4, note text);
INSERT INTO events (event_id, note) VALUES (99, 'stale event');
SQL

PGHOST=127.0.0.1 PGPORT="$CLEAN_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --clean --if-exists --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.custom" \
  >"$OUT_DIR/clean-restore.out" 2>"$OUT_DIR/clean-restore.err"

PGHOST=127.0.0.1 PGPORT="$CLEAN_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/clean-verify.out" 2>"$OUT_DIR/clean-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/clean-verify.out"
verify_indexes "$CLEAN_RESTORE_PORT" "clean"
verify_constraints "$CLEAN_RESTORE_PORT" "clean"
verify_comments "$CLEAN_RESTORE_PORT" "clean"
verify_constraint_comments "$CLEAN_RESTORE_PORT" "clean"
verify_views "$CLEAN_RESTORE_PORT" "clean"
verify_materialized_views "$CLEAN_RESTORE_PORT" "clean"
verify_sequences "$CLEAN_RESTORE_PORT" "clean"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$INSERT_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/insert-restore-server.log" 2>&1 &
insert_restore_pid=$!
wait_for_port "$INSERT_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$INSERT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -f "$OUT_DIR/dump-inserts.sql" \
  >"$OUT_DIR/insert-restore.out" 2>"$OUT_DIR/insert-restore.err"

PGHOST=127.0.0.1 PGPORT="$INSERT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/insert-verify.out" 2>"$OUT_DIR/insert-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/insert-verify.out"
verify_indexes "$INSERT_RESTORE_PORT" "insert"
verify_constraints "$INSERT_RESTORE_PORT" "insert"
verify_comments "$INSERT_RESTORE_PORT" "insert"
verify_constraint_comments "$INSERT_RESTORE_PORT" "insert"
verify_views "$INSERT_RESTORE_PORT" "insert"
verify_materialized_views "$INSERT_RESTORE_PORT" "insert"
verify_sequences "$INSERT_RESTORE_PORT" "insert"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$SPLIT_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/split-restore-server.log" 2>&1 &
split_restore_pid=$!
wait_for_port "$SPLIT_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -f "$OUT_DIR/dump-schema.sql" \
  >"$OUT_DIR/split-schema-restore.out" 2>"$OUT_DIR/split-schema-restore.err"

PGHOST=127.0.0.1 PGPORT="$SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -q -f "$OUT_DIR/dump-data.sql" \
  >"$OUT_DIR/split-data-restore.out" 2>"$OUT_DIR/split-data-restore.err"

PGHOST=127.0.0.1 PGPORT="$SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/split-verify.out" 2>"$OUT_DIR/split-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/split-verify.out"
verify_indexes "$SPLIT_RESTORE_PORT" "split"
verify_constraints "$SPLIT_RESTORE_PORT" "split"
verify_comments "$SPLIT_RESTORE_PORT" "split"
verify_constraint_comments "$SPLIT_RESTORE_PORT" "split"
verify_views "$SPLIT_RESTORE_PORT" "split"
verify_materialized_views "$SPLIT_RESTORE_PORT" "split" "$OUT_DIR/materialized-view-empty.expected"
verify_sequences "$SPLIT_RESTORE_PORT" "split"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$CUSTOM_SPLIT_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/custom-split-restore-server.log" 2>&1 &
custom_split_restore_pid=$!
wait_for_port "$CUSTOM_SPLIT_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$CUSTOM_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --schema-only --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.custom" \
  >"$OUT_DIR/custom-split-schema-restore.out" 2>"$OUT_DIR/custom-split-schema-restore.err"

PGHOST=127.0.0.1 PGPORT="$CUSTOM_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --data-only --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.custom" \
  >"$OUT_DIR/custom-split-data-restore.out" 2>"$OUT_DIR/custom-split-data-restore.err"

PGHOST=127.0.0.1 PGPORT="$CUSTOM_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/custom-split-verify.out" 2>"$OUT_DIR/custom-split-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/custom-split-verify.out"
verify_indexes "$CUSTOM_SPLIT_RESTORE_PORT" "custom-split"
verify_constraints "$CUSTOM_SPLIT_RESTORE_PORT" "custom-split"
verify_comments "$CUSTOM_SPLIT_RESTORE_PORT" "custom-split"
verify_constraint_comments "$CUSTOM_SPLIT_RESTORE_PORT" "custom-split"
verify_views "$CUSTOM_SPLIT_RESTORE_PORT" "custom-split"
verify_materialized_views "$CUSTOM_SPLIT_RESTORE_PORT" "custom-split" "$OUT_DIR/materialized-view-empty.expected"
verify_sequences "$CUSTOM_SPLIT_RESTORE_PORT" "custom-split"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$DIRECTORY_SPLIT_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/directory-split-restore-server.log" 2>&1 &
directory_split_restore_pid=$!
wait_for_port "$DIRECTORY_SPLIT_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$DIRECTORY_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --schema-only --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.dir" \
  >"$OUT_DIR/directory-split-schema-restore.out" 2>"$OUT_DIR/directory-split-schema-restore.err"

PGHOST=127.0.0.1 PGPORT="$DIRECTORY_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --data-only --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.dir" \
  >"$OUT_DIR/directory-split-data-restore.out" 2>"$OUT_DIR/directory-split-data-restore.err"

PGHOST=127.0.0.1 PGPORT="$DIRECTORY_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/directory-split-verify.out" 2>"$OUT_DIR/directory-split-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/directory-split-verify.out"
verify_indexes "$DIRECTORY_SPLIT_RESTORE_PORT" "directory-split"
verify_constraints "$DIRECTORY_SPLIT_RESTORE_PORT" "directory-split"
verify_comments "$DIRECTORY_SPLIT_RESTORE_PORT" "directory-split"
verify_constraint_comments "$DIRECTORY_SPLIT_RESTORE_PORT" "directory-split"
verify_views "$DIRECTORY_SPLIT_RESTORE_PORT" "directory-split"
verify_materialized_views "$DIRECTORY_SPLIT_RESTORE_PORT" "directory-split" "$OUT_DIR/materialized-view-empty.expected"
verify_sequences "$DIRECTORY_SPLIT_RESTORE_PORT" "directory-split"

cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen "127.0.0.1:$TAR_SPLIT_RESTORE_PORT" --shared-catalog \
  >"$OUT_DIR/tar-split-restore-server.log" 2>&1 &
tar_split_restore_pid=$!
wait_for_port "$TAR_SPLIT_RESTORE_PORT"

PGHOST=127.0.0.1 PGPORT="$TAR_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --schema-only --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.tar" \
  >"$OUT_DIR/tar-split-schema-restore.out" 2>"$OUT_DIR/tar-split-schema-restore.err"

PGHOST=127.0.0.1 PGPORT="$TAR_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  pg_restore --data-only --no-owner --no-privileges --dbname=postgres "$OUT_DIR/dump.tar" \
  >"$OUT_DIR/tar-split-data-restore.out" 2>"$OUT_DIR/tar-split-data-restore.err"

PGHOST=127.0.0.1 PGPORT="$TAR_SPLIT_RESTORE_PORT" PGDATABASE=postgres PGUSER=postgres \
  psql -v ON_ERROR_STOP=1 -X -A -t \
  -c "SELECT id, name, tier FROM accounts ORDER BY id;" \
  -c "SELECT event_id, note FROM events ORDER BY event_id;" \
  >"$OUT_DIR/tar-split-verify.out" 2>"$OUT_DIR/tar-split-verify.err"

diff -u "$OUT_DIR/verify.expected" "$OUT_DIR/tar-split-verify.out"
verify_indexes "$TAR_SPLIT_RESTORE_PORT" "tar-split"
verify_constraints "$TAR_SPLIT_RESTORE_PORT" "tar-split"
verify_comments "$TAR_SPLIT_RESTORE_PORT" "tar-split"
verify_constraint_comments "$TAR_SPLIT_RESTORE_PORT" "tar-split"
verify_views "$TAR_SPLIT_RESTORE_PORT" "tar-split"
verify_materialized_views "$TAR_SPLIT_RESTORE_PORT" "tar-split" "$OUT_DIR/materialized-view-empty.expected"
verify_sequences "$TAR_SPLIT_RESTORE_PORT" "tar-split"

pg_restore --list "$OUT_DIR/dump.custom" >"$OUT_DIR/dump.custom.toc"
pg_restore --list "$OUT_DIR/dump.dir" >"$OUT_DIR/dump.dir.toc"
pg_restore --list "$OUT_DIR/dump.tar" >"$OUT_DIR/dump.tar.toc"

grep -F "COPY public.accounts (id, name, tier) FROM stdin;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COPY public.events (event_id, note) FROM stdin;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "INSERT INTO public.accounts VALUES" "$OUT_DIR/dump-inserts.sql" >/dev/null
grep -F "	(1, 'Ada', 7)," "$OUT_DIR/dump-inserts.sql" >/dev/null
grep -F "	(2, 'unknown', 7);" "$OUT_DIR/dump-inserts.sql" >/dev/null
grep -F "INSERT INTO public.events VALUES" "$OUT_DIR/dump-inserts.sql" >/dev/null
grep -F "	(10, 'created')," "$OUT_DIR/dump-inserts.sql" >/dev/null
grep -F "CREATE SCHEMA public;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE TABLE public.accounts (" "$OUT_DIR/dump.sql" >/dev/null
grep -F "    name text DEFAULT 'unknown'::text," "$OUT_DIR/dump.sql" >/dev/null
grep -F "    tier integer DEFAULT 7" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE TABLE public.events (" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE INDEX accounts_name_idx ON public.accounts USING btree (name);" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE INDEX events_note_idx ON public.events USING btree (note);" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE VIEW public.account_lookup AS" "$OUT_DIR/dump.sql" >/dev/null
grep -F "SELECT id, name FROM accounts WHERE id > 1 ORDER BY id;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON VIEW public.account_lookup IS 'active account lookup';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE MATERIALIZED VIEW public.account_snapshot AS" "$OUT_DIR/dump.sql" >/dev/null
grep -F "SELECT id, name FROM accounts ORDER BY id" "$OUT_DIR/dump.sql" >/dev/null
grep -F "  WITH NO DATA;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "REFRESH MATERIALIZED VIEW public.account_snapshot;" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON MATERIALIZED VIEW public.account_snapshot IS 'account snapshot';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE SEQUENCE public.account_seq" "$OUT_DIR/dump.sql" >/dev/null
grep -F "START WITH 1" "$OUT_DIR/dump.sql" >/dev/null
grep -F "SELECT pg_catalog.setval('public.account_seq', 1, false);" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON SEQUENCE public.account_seq IS 'account sequence';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON TABLE public.accounts IS 'accounts table';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON COLUMN public.accounts.name IS 'account display name';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON INDEX public.accounts_name_idx IS 'accounts name lookup';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON TABLE public.events IS 'events table';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON COLUMN public.events.note IS 'event note';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "COMMENT ON INDEX public.events_note_idx IS 'events note lookup';" "$OUT_DIR/dump.sql" >/dev/null
grep -F "CREATE SCHEMA public;" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "CREATE TABLE public.accounts (" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "    name text DEFAULT 'unknown'::text," "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "    tier integer DEFAULT 7" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "CREATE TABLE public.events (" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "CREATE INDEX accounts_name_idx ON public.accounts USING btree (name);" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "CREATE INDEX events_note_idx ON public.events USING btree (note);" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "CREATE VIEW public.account_lookup AS" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "SELECT id, name FROM accounts WHERE id > 1 ORDER BY id;" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON VIEW public.account_lookup IS 'active account lookup';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "CREATE MATERIALIZED VIEW public.account_snapshot AS" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "  WITH NO DATA;" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON MATERIALIZED VIEW public.account_snapshot IS 'account snapshot';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "CREATE SEQUENCE public.account_seq" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON SEQUENCE public.account_seq IS 'account sequence';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON TABLE public.accounts IS 'accounts table';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON COLUMN public.accounts.name IS 'account display name';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON INDEX public.accounts_name_idx IS 'accounts name lookup';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON TABLE public.events IS 'events table';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON COLUMN public.events.note IS 'event note';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COMMENT ON INDEX public.events_note_idx IS 'events note lookup';" "$OUT_DIR/dump-schema.sql" >/dev/null
grep -F "COPY public.accounts (id, name, tier) FROM stdin;" "$OUT_DIR/dump-data.sql" >/dev/null
grep -F "COPY public.events (event_id, note) FROM stdin;" "$OUT_DIR/dump-data.sql" >/dev/null
grep -F "SELECT pg_catalog.setval('public.account_seq', 1, false);" "$OUT_DIR/dump-data.sql" >/dev/null
if grep -F "REFRESH MATERIALIZED VIEW public.account_snapshot;" "$OUT_DIR/dump-data.sql" >/dev/null; then
  echo "plain data-only dump unexpectedly included materialized view refresh" >&2
  exit 1
fi
grep -F "SCHEMA - public" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE public accounts" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE public events" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "INDEX public accounts_name_idx" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "INDEX public events_note_idx" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "VIEW public account_lookup" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "MATERIALIZED VIEW public account_snapshot" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "MATERIALIZED VIEW DATA public account_snapshot" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "SEQUENCE public account_seq" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "SEQUENCE SET public account_seq" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE DATA public accounts" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "TABLE DATA public events" "$OUT_DIR/dump.custom.toc" >/dev/null
grep -F "SCHEMA - public" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE public accounts" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE public events" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "INDEX public accounts_name_idx" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "INDEX public events_note_idx" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "VIEW public account_lookup" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "MATERIALIZED VIEW public account_snapshot" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "MATERIALIZED VIEW DATA public account_snapshot" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "SEQUENCE public account_seq" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "SEQUENCE SET public account_seq" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE DATA public accounts" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "TABLE DATA public events" "$OUT_DIR/dump.dir.toc" >/dev/null
grep -F "SCHEMA - public" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "TABLE public accounts" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "TABLE public events" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "INDEX public accounts_name_idx" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "INDEX public events_note_idx" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "VIEW public account_lookup" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "MATERIALIZED VIEW public account_snapshot" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "MATERIALIZED VIEW DATA public account_snapshot" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "SEQUENCE public account_seq" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "SEQUENCE SET public account_seq" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "TABLE DATA public accounts" "$OUT_DIR/dump.tar.toc" >/dev/null
grep -F "TABLE DATA public events" "$OUT_DIR/dump.tar.toc" >/dev/null

echo "pg_dump_plain_public_schema_restore=passed"
echo "pg_dump_custom_public_schema_pg_restore=passed"
echo "pg_dump_directory_public_schema_pg_restore=passed"
echo "pg_dump_tar_public_schema_pg_restore=passed"
echo "pg_dump_directory_parallel_public_schema_pg_restore=passed"
echo "pg_dump_custom_clean_if_exists_pg_restore=passed"
echo "pg_dump_plain_insert_style_restore=passed"
echo "pg_dump_plain_split_schema_data_restore=passed"
echo "pg_dump_custom_split_schema_data_restore=passed"
echo "pg_dump_directory_split_schema_data_restore=passed"
echo "pg_dump_tar_split_schema_data_restore=passed"
echo "pg_dump_metadata_index_restore=passed"
echo "pg_dump_bounded_view_restore=passed"
echo "pg_dump_bounded_materialized_view_restore=passed"
echo "pg_dump_bounded_sequence_restore=passed"
echo "dump_file=$OUT_DIR/dump.sql"
echo "insert_dump_file=$OUT_DIR/dump-inserts.sql"
echo "schema_dump_file=$OUT_DIR/dump-schema.sql"
echo "data_dump_file=$OUT_DIR/dump-data.sql"
echo "custom_dump_file=$OUT_DIR/dump.custom"
echo "directory_dump_dir=$OUT_DIR/dump.dir"
echo "tar_dump_file=$OUT_DIR/dump.tar"
