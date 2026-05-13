#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SUITE_DIR="$ROOT_DIR/tests/compat/psql-golden"
SCENARIOS_DIR="$SUITE_DIR/scenarios"
EXPECTED_DIR="$SUITE_DIR/expected"
OUT_DIR="${PSQL_GOLDEN_OUT_DIR:-$ROOT_DIR/target/psql-golden}"
PSQL_BIN="${PSQL_BIN:-psql}"
BOOT_CMD="${PSQL_GOLDEN_BOOT_CMD:-}"
BOOT_CWD="${PSQL_GOLDEN_BOOT_CWD:-$ROOT_DIR}"
STOP_CMD="${PSQL_GOLDEN_STOP_CMD:-}"
WAIT_HOST="${PSQL_GOLDEN_WAIT_HOST:-${PGHOST:-127.0.0.1}}"
WAIT_PORT="${PSQL_GOLDEN_WAIT_PORT:-${PGPORT:-5432}}"
WAIT_TIMEOUT_SEC="${PSQL_GOLDEN_WAIT_TIMEOUT_SEC:-30}"
BOOT_LOG="$OUT_DIR/boot.log"
REPORT_PATH="${PSQL_GOLDEN_REPORT:-$ROOT_DIR/target/compat/psql-golden-report.json}"
REPORT_ROWS="$OUT_DIR/report.rows.tsv"
boot_pid=""
suite_status="passed"
suite_error=""

mkdir -p "$OUT_DIR"

write_report() {
  local output_path=$1
  local status_value=$2
  local error_value=$3

  python3 - "$REPORT_ROWS" "$output_path" "$status_value" "$error_value" <<'PY'
import json
import sys
from pathlib import Path

rows_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
status = sys.argv[3]
error = sys.argv[4]

scenarios = []
if rows_path.exists():
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, scenario_status, expected_rc, actual_rc = line.split("\t")
        scenarios.append(
            {
                "id": f"psql_golden::{name}",
                "name": name,
                "status": scenario_status,
                "expected_rc": int(expected_rc),
                "actual_rc": int(actual_rc),
            }
        )

report = {
    "suite": "psql_golden",
    "status": status,
    "error": error or None,
    "scenarios": scenarios,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

if ! command -v "$PSQL_BIN" >/dev/null 2>&1; then
  suite_status="failed"
  suite_error="psql not found (set PSQL_BIN or install psql)"
  write_report "$REPORT_PATH" "$suite_status" "$suite_error"
  echo "error: psql not found (set PSQL_BIN or install psql)" >&2
  exit 2
fi
:
>"$REPORT_ROWS"

if [ -z "${PGHOST:-}" ] || [ -z "${PGPORT:-}" ] || [ -z "${PGDATABASE:-}" ] || [ -z "${PGUSER:-}" ]; then
  suite_status="failed"
  suite_error="missing libpq env (PGHOST, PGPORT, PGDATABASE, PGUSER)"
  write_report "$REPORT_PATH" "$suite_status" "$suite_error"
  cat >&2 <<'MSG'
error: missing libpq env. Set at least PGHOST, PGPORT, PGDATABASE, PGUSER (and PGPASSWORD if needed).
MSG
  exit 2
fi

normalize() {
  # Keep output deterministic across environments and psql versions.
  sed -E \
    -e 's/[[:space:]]+$//' \
    -e 's#^psql:.*/tests/compat/psql-golden/scenarios/#psql:#' \
    -e '/^$/N;/^\n$/D' \
    -e 's#(^|\n)psql:.*/tests/compat/psql-golden/scenarios/#\1psql:#' \
    -e '/^Time: [0-9.]+ ms$/d' \
    -e '/^SSL connection \(.+\)$/d' \
    -e '/^psql \([0-9.]+\).*$/d' \
    | perl -pe 's/[ \t\r]+$//' \
    | awk '{ lines[NR] = $0 } END { end = NR; while (end > 0 && lines[end] == "") end--; for (i = 1; i <= end; i++) print lines[i] }'
}

wait_for_endpoint() {
  python3 - "$WAIT_HOST" "$WAIT_PORT" "$WAIT_TIMEOUT_SEC" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect((host, port))
    except OSError as exc:
        last_error = exc
        time.sleep(0.2)
    else:
        sock.close()
        print(f"endpoint ready at {host}:{port}")
        sys.exit(0)
    finally:
        sock.close()

print(f"timed out waiting for {host}:{port}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

cleanup() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ -z "$suite_error" ]; then
    suite_status="failed"
  fi
  write_report "$REPORT_PATH" "$suite_status" "$suite_error"
  if [ -n "$STOP_CMD" ]; then
    bash -lc "$STOP_CMD" >>"$BOOT_LOG" 2>&1 || true
  fi
  if [ -n "$boot_pid" ] && kill -0 "$boot_pid" 2>/dev/null; then
    kill "$boot_pid" 2>/dev/null || true
    wait "$boot_pid" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT

if [ -n "$BOOT_CMD" ]; then
  : >"$BOOT_LOG"
  (
    cd "$BOOT_CWD"
    exec bash -lc "$BOOT_CMD"
  ) >>"$BOOT_LOG" 2>&1 &
  boot_pid=$!
  if ! wait_for_endpoint; then
    suite_status="failed"
    suite_error="timed out waiting for booted compatibility endpoint"
    exit 1
  fi
fi

status=0
for scenario in "$SCENARIOS_DIR"/*.sql; do
  name=$(basename "$scenario" .sql)
  expected="$EXPECTED_DIR/$name.txt"
  expected_rc_file="$EXPECTED_DIR/$name.rc"
  scenario_args_file="$SCENARIOS_DIR/$name.psqlargs"
  if [ ! -f "$expected" ]; then
    echo "error: missing expected artifact for scenario '$name': $expected" >&2
    status=1
    continue
  fi

  expected_rc=0
  if [ -f "$expected_rc_file" ]; then
    expected_rc=$(tr -d '[:space:]' <"$expected_rc_file")
    if ! [[ "$expected_rc" =~ ^[0-9]+$ ]]; then
      echo "error: invalid expected rc for scenario '$name': $expected_rc_file" >&2
      status=1
      continue
    fi
  fi

  scenario_args=()
  if [ -f "$scenario_args_file" ]; then
    mapfile -t scenario_args <"$scenario_args_file"
  fi

  raw="$OUT_DIR/$name.raw.txt"
  norm="$OUT_DIR/$name.txt"

  set +e
  "$PSQL_BIN" \
    --no-psqlrc \
    --set ON_ERROR_STOP=0 \
    --set VERBOSITY=default \
    --set SHOW_CONTEXT=never \
    --file "$scenario" \
    "${scenario_args[@]}" \
    >"$raw" 2>&1
  rc=$?
  set -e

  normalize <"$raw" >"$norm"

  if ! diff -u "$expected" "$norm"; then
    echo "scenario '$name' mismatch (psql exit=$rc)" >&2
    printf '%s\t%s\t%s\t%s\n' "$name" "failed" "$expected_rc" "$rc" >>"$REPORT_ROWS"
    status=1
  elif [ "$rc" -ne "$expected_rc" ]; then
    echo "scenario '$name' exit mismatch (expected=$expected_rc actual=$rc)" >&2
    printf '%s\t%s\t%s\t%s\n' "$name" "failed" "$expected_rc" "$rc" >>"$REPORT_ROWS"
    status=1
  else
    printf '%s\t%s\t%s\t%s\n' "$name" "passed" "$expected_rc" "$rc" >>"$REPORT_ROWS"
    echo "scenario '$name' ok (psql exit=$rc)"
  fi
done

if [ "$status" -ne 0 ]; then
  suite_status="failed"
  echo "psql golden: FAILED" >&2
  exit 1
fi

echo "psql golden: PASS"
