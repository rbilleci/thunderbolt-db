#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-resident-warmup.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

run_preflight() {
  cargo run -q -p gpu_db_engine --example resident_warmup_preflight -- "$@"
}

run_preflight --dry-run --include-missing > "$TMP_DIR/dry-run.out"
run_preflight --apply --include-missing > "$TMP_DIR/apply.out"

grep -q '^mode=dry-run$' "$TMP_DIR/dry-run.out"
grep -q '^scenario=basic$' "$TMP_DIR/dry-run.out"
grep -q '^entry.table=events$' "$TMP_DIR/dry-run.out"
grep -q '^entry.table=aux$' "$TMP_DIR/dry-run.out"
grep -q '^entry.table=missing$' "$TMP_DIR/dry-run.out"
grep -q '^entry.action=warmed$' "$TMP_DIR/dry-run.out"
grep -q '^entry.action=skipped$' "$TMP_DIR/dry-run.out"
grep -q '^entry.route.query_shape=count_all$' "$TMP_DIR/dry-run.out"
grep -q '^entry.route.h2d_bytes_if_resident=0$' "$TMP_DIR/dry-run.out"
grep -q '^entry.route.d2h_bytes_estimate=8$' "$TMP_DIR/dry-run.out"

grep -q '^mode=apply$' "$TMP_DIR/apply.out"
grep -q '^scenario=basic$' "$TMP_DIR/apply.out"
grep -q '^entry.table=events$' "$TMP_DIR/apply.out"
grep -q '^entry.table=missing$' "$TMP_DIR/apply.out"
grep -q '^entry.action=warmed$' "$TMP_DIR/apply.out"
grep -q '^entry.action=skipped$' "$TMP_DIR/apply.out"
grep -q '^entry.route.query_shape=count_all$' "$TMP_DIR/apply.out"
grep -q '^entry.route.h2d_bytes_if_resident=0$' "$TMP_DIR/apply.out"
grep -q '^entry.route.d2h_bytes_estimate=8$' "$TMP_DIR/apply.out"

sed '/^mode=/d;/^verification\./d' "$TMP_DIR/dry-run.out" > "$TMP_DIR/dry-run.normalized"
sed '/^mode=/d;/^verification\./d' "$TMP_DIR/apply.out" > "$TMP_DIR/apply.normalized"
diff -u "$TMP_DIR/dry-run.normalized" "$TMP_DIR/apply.normalized"

run_preflight --apply --scenario invalidated --refresh-invalidated > "$TMP_DIR/invalidated.out"
grep -q '^scenario=invalidated$' "$TMP_DIR/invalidated.out"
grep -q '^entry.table=events$' "$TMP_DIR/invalidated.out"
grep -q '^entry.action=refreshed$' "$TMP_DIR/invalidated.out"
grep -q '^entry.route.query_shape=count_all$' "$TMP_DIR/invalidated.out"
grep -q '^entry.route.d2h_bytes_estimate=8$' "$TMP_DIR/invalidated.out"

run_preflight --apply --scenario memory-pressure > "$TMP_DIR/memory-pressure.out"
grep -q '^scenario=memory-pressure$' "$TMP_DIR/memory-pressure.out"
grep -q '^entry.action=skipped$' "$TMP_DIR/memory-pressure.out"
grep -q '^entry.reason=GPU 0 is memory pressured$' "$TMP_DIR/memory-pressure.out"
grep -q '^entry.route.accepted=none$' "$TMP_DIR/memory-pressure.out"

run_preflight --apply --scenario oversized-budget > "$TMP_DIR/oversized.out"
grep -q '^scenario=oversized-budget$' "$TMP_DIR/oversized.out"
grep -q '^budget_bytes=1$' "$TMP_DIR/oversized.out"
grep -q '^entry.action=error$' "$TMP_DIR/oversized.out"
grep -q 'exceeding GPU 0' "$TMP_DIR/oversized.out"

echo "p8 resident warmup preflight smoke passed"
