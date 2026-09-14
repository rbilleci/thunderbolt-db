#!/usr/bin/env bash
#
# GPU-native source-release smoke: a durable, process-boundary pgwire route.
# It is deliberately a small release check, not a performance benchmark. The
# script owns its temporary directory, loopback port, WAL segment, and server
# process, and leaves no database state behind.

set -euo pipefail

readonly SCRIPT_NAME="run_oss_release_smoke.sh"
readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly PSQL_BIN="${PSQL_BIN:-psql}"
readonly TEMP_PARENT="$(cd -- "${TMPDIR:-/tmp}" && pwd -P)"
readonly WORKDIR_PREFIX="${TEMP_PARENT}/gpu-db-oss-release-smoke."

workdir=""
server_pid=""
port=""
assertions=0

die() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}

owns_workdir() {
  local parent
  [[ -n "$workdir" && -d "$workdir" && "$workdir" == "${WORKDIR_PREFIX}"* ]] || return 1
  parent="$(cd -- "$(dirname -- "$workdir")" && pwd -P)" || return 1
  [[ "$parent" == "$TEMP_PARENT" ]]
}

cleanup() {
  local status=$?
  cleanup_server
  if [[ "$status" -ne 0 ]]; then
    printf 'oss_release_smoke_executed_assertions=%s\n' "$assertions"
    printf 'oss_release_smoke_completion=failed\n'
  fi
  if owns_workdir; then
    rm -rf -- "$workdir"
  elif [[ -n "$workdir" ]]; then
    printf '%s: refusing to remove unowned temporary path: %s\n' \
      "$SCRIPT_NAME" "$workdir" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

assert_passed() {
  assertions=$((assertions + 1))
  printf 'oss_release_smoke_assertion=%s status=passed\n' "$1"
}

reserve_loopback_port() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

wait_for_server() {
  local deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      sed -n '1,200p' "$workdir/server.stderr" >&2 || true
      die "server exited before accepting loopback connections"
    fi
    if python3 - "$port" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    try:
        sock.connect(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        sys.exit(1)
PY
    then
      return 0
    fi
    sleep 0.1
  done
  sed -n '1,200p' "$workdir/server.stderr" >&2 || true
  die "server did not listen on 127.0.0.1:${port} within 30 seconds"
}

start_server() {
  (
    unset GPU_DB_SECURITY_PROFILE GPU_DB_TLS_CERT GPU_DB_TLS_KEY GPU_DB_AUTH_USER
    unset GPU_DB_AUTH_PASSWORD GPU_DB_AUTH_SCRAM_VERIFIER GPU_DB_AUTH_SCRAM_VERIFIER_FILE
    export GPU_DB_WAL_SEGMENT="$workdir/release-smoke.wal"
    export GPU_DB_WAL_DURABILITY="serial"
    export GPU_DB_INTENT_LANES="1"
    exec "$server_bin" --listen "127.0.0.1:${port}"
  ) >"$workdir/server.stdout" 2>"$workdir/server.stderr" &
  server_pid=$!
  wait_for_server
}

run_sql() {
  "$PSQL_BIN" -X -q -A -t -v ON_ERROR_STOP=1 \
    "postgresql://postgres@127.0.0.1:${port}/postgres?sslmode=disable" \
    -c "$1"
}

assert_query() {
  local name=$1
  local expected=$2
  local sql=$3
  local actual
  actual="$(run_sql "$sql")"
  if [[ "$actual" != "$expected" ]]; then
    printf '%s: assertion %s failed\nexpected:\n%s\nactual:\n%s\n' \
      "$SCRIPT_NAME" "$name" "$expected" "$actual" >&2
    exit 1
  fi
  assert_passed "$name"
}

cd "$ROOT_DIR"
require_command cargo
require_command "$PSQL_BIN"
require_command nvidia-smi
require_command python3

device_lines="$(nvidia-smi -L 2>&1)"
[[ -n "$device_lines" ]] || die "nvidia-smi reported no visible NVIDIA device"
device_count="$(printf '%s\n' "$device_lines" | awk 'NF { count += 1 } END { print count + 0 }')"
[[ "$device_count" -gt 0 ]] || die "nvidia-smi reported no visible NVIDIA device"
printf 'oss_release_smoke_gpu_devices=%s\n' "$device_count"

if [[ -n "${OSS_SERVER_BIN:-}" ]]; then
  server_bin="$OSS_SERVER_BIN"
  [[ -x "$server_bin" ]] || die "OSS_SERVER_BIN is not an executable file: $server_bin"
else
  cargo build --locked --release -p gpu_db_server --bin thunderbolt-db-server
  server_bin="${CARGO_TARGET_DIR:-target}/release/thunderbolt-db-server"
  [[ -x "$server_bin" ]] || die "release server binary was not produced: $server_bin"
fi

workdir="$(mktemp -d "${WORKDIR_PREFIX}XXXXXX")"
port="$(reserve_loopback_port)"
start_server
assert_passed "server_started_with_serial_durable_wal_and_one_intent_lane"

run_sql "
  CREATE TABLE oss_release_rows (
    id INT4 PRIMARY KEY,
    amount INT4,
    note TEXT,
    active BOOLEAN
  );
  INSERT INTO oss_release_rows (id, amount, note, active) VALUES
    (1, 10, 'kept', TRUE),
    (2, NULL, NULL, FALSE),
    (3, 30, 'rollback', TRUE);
" >/dev/null
assert_passed "typed_and_null_insert_acknowledged"

run_sql "BEGIN; UPDATE oss_release_rows SET amount = 11 WHERE id = 1; COMMIT;" >/dev/null
assert_passed "committed_update_acknowledged"

run_sql "BEGIN; DELETE FROM oss_release_rows WHERE id = 3; ROLLBACK;" >/dev/null
assert_passed "rolled_back_delete_acknowledged"

assert_query \
  "ordered_gpu_read_before_restart" \
  $'1|11|kept|t\n2|||f\n3|30|rollback|t' \
  "SELECT id, amount, note, active FROM oss_release_rows ORDER BY id"
assert_query \
  "gpu_sum_before_restart" \
  "41" \
  "SELECT SUM(amount) FROM oss_release_rows"

cleanup_server
assert_passed "server_killed_after_acknowledged_durable_commits"
start_server
assert_passed "server_restarted_from_same_wal"

assert_query \
  "ordered_gpu_read_recovered_after_restart" \
  $'1|11|kept|t\n2|||f\n3|30|rollback|t' \
  "SELECT id, amount, note, active FROM oss_release_rows ORDER BY id"
assert_query \
  "gpu_sum_recovered_after_restart" \
  "41" \
  "SELECT SUM(amount) FROM oss_release_rows"

[[ "$assertions" -gt 0 ]] || die "no release-smoke assertions executed"
printf 'oss_release_smoke_executed_assertions=%s\n' "$assertions"
printf 'oss_release_smoke_completion=passed\n'
