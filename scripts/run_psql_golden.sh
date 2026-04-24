#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SUITE_DIR="$ROOT_DIR/tests/compat/psql-golden"
SCENARIOS_DIR="$SUITE_DIR/scenarios"
EXPECTED_DIR="$SUITE_DIR/expected"
OUT_DIR="${PSQL_GOLDEN_OUT_DIR:-$ROOT_DIR/target/psql-golden}"
PSQL_BIN="${PSQL_BIN:-psql}"

if ! command -v "$PSQL_BIN" >/dev/null 2>&1; then
  echo "error: psql not found (set PSQL_BIN or install psql)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

if [ -z "${PGHOST:-}" ] || [ -z "${PGPORT:-}" ] || [ -z "${PGDATABASE:-}" ] || [ -z "${PGUSER:-}" ]; then
  cat >&2 <<'MSG'
error: missing libpq env. Set at least PGHOST, PGPORT, PGDATABASE, PGUSER (and PGPASSWORD if needed).
MSG
  exit 2
fi

normalize() {
  # Keep output deterministic across environments and psql versions.
  sed -E \
    -e 's/[[:space:]]+$//' \
    -e '/^$/N;/^\n$/D' \
    -e '/^Time: [0-9.]+ ms$/d' \
    -e '/^SSL connection \(.+\)$/d' \
    -e '/^psql \([0-9.]+\).*$/d'
}

status=0
for scenario in "$SCENARIOS_DIR"/*.sql; do
  name=$(basename "$scenario" .sql)
  expected="$EXPECTED_DIR/$name.txt"
  if [ ! -f "$expected" ]; then
    echo "error: missing expected artifact for scenario '$name': $expected" >&2
    status=1
    continue
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
    >"$raw" 2>&1
  rc=$?
  set -e

  normalize <"$raw" >"$norm"

  if ! diff -u "$expected" "$norm"; then
    echo "scenario '$name' mismatch (psql exit=$rc)" >&2
    status=1
  else
    echo "scenario '$name' ok (psql exit=$rc)"
  fi
done

if [ "$status" -ne 0 ]; then
  echo "psql golden: FAILED" >&2
  exit 1
fi

echo "psql golden: PASS"
