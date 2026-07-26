#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-resident-maintenance.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

run_tick() {
  cargo run -q -p gpu_db_engine --example resident_maintenance_tick -- "$@"
}

run_tick --scenario basic > "$TMP_DIR/basic.out"
grep -q '^scenario=basic$' "$TMP_DIR/basic.out"
grep -q '^entry_count=3$' "$TMP_DIR/basic.out"
grep -q '^warmed_count=3$' "$TMP_DIR/basic.out"
grep -q '^skipped_count=0$' "$TMP_DIR/basic.out"
grep -q '^error_count=0$' "$TMP_DIR/basic.out"
grep -q '^entry.table=events$' "$TMP_DIR/basic.out"
grep -q '^entry.table=aux$' "$TMP_DIR/basic.out"

run_tick --scenario invalidated > "$TMP_DIR/invalidated.out"
grep -q '^scenario=invalidated$' "$TMP_DIR/invalidated.out"
grep -q '^entry_count=3$' "$TMP_DIR/invalidated.out"
grep -q '^refreshed_count=0$' "$TMP_DIR/invalidated.out"
grep -q '^warmed_count=3$' "$TMP_DIR/invalidated.out"
grep -q '^entry.table=events$' "$TMP_DIR/invalidated.out"
grep -q '^entry.action=Warmed$' "$TMP_DIR/invalidated.out"

run_tick --scenario memory-pressure > "$TMP_DIR/memory-pressure.out"
grep -q '^scenario=memory-pressure$' "$TMP_DIR/memory-pressure.out"
grep -q '^entry_count=3$' "$TMP_DIR/memory-pressure.out"
grep -q '^skipped_count=3$' "$TMP_DIR/memory-pressure.out"
grep -q '^route_ready_count=0$' "$TMP_DIR/memory-pressure.out"
grep -q '^route_blocker.reason=GPU 0 is memory pressured$' "$TMP_DIR/memory-pressure.out"

run_tick --scenario oversized-budget > "$TMP_DIR/oversized.out"
grep -q '^scenario=oversized-budget$' "$TMP_DIR/oversized.out"
grep -q '^entry_count=1$' "$TMP_DIR/oversized.out"
grep -q '^error_count=1$' "$TMP_DIR/oversized.out"
grep -q '^route_ready_count=0$' "$TMP_DIR/oversized.out"
grep -q 'requested GPU 0 residency budget 1 bytes cannot replace device-authoritative relation' "$TMP_DIR/oversized.out"

echo "p8 resident maintenance smoke passed"
