#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

run_preflight() {
  local fixture_dir="$1"
  shift
  cargo run -q -p gpu_db_engine --example wal_archive_maintenance_preflight -- \
    --control "$fixture_dir/base/CONTROL" \
    --archive-manifest "$fixture_dir/archive/MANIFEST" \
    --timeline-registry "$fixture_dir/TIMELINE_REGISTRY" \
    --retain-timeline timeline-keep-0002 \
    --current-timestamp-micros 6000 \
    --pitr-window-micros 3000 \
    --recover-retained \
    "$@"
}

cd "$ROOT"

fixture="$TMP_DIR/fixture"
cargo run -q -p gpu_db_engine --example wal_archive_maintenance_preflight -- \
  --write-demo-fixture "$fixture" >"$TMP_DIR/fixture.out"

run_preflight "$fixture" >"$TMP_DIR/dry-run.out"
run_preflight "$fixture" --apply >"$TMP_DIR/apply.out"

grep -v '^mode=' "$TMP_DIR/dry-run.out" >"$TMP_DIR/dry-run.plan"
grep -v '^mode=' "$TMP_DIR/apply.out" >"$TMP_DIR/apply.plan"
diff -u "$TMP_DIR/dry-run.plan" "$TMP_DIR/apply.plan"
grep -q '^mode=dry-run$' "$TMP_DIR/dry-run.out"
grep -q '^mode=apply$' "$TMP_DIR/apply.out"
grep -q '^base_txn_id=2$' "$TMP_DIR/apply.out"
grep -q '^retained_record_count=4$' "$TMP_DIR/apply.out"
grep -q '^removed_record_count=1$' "$TMP_DIR/apply.out"
grep -q '^retained_timeline_ids=timeline-main-0001,timeline-keep-0002$' "$TMP_DIR/apply.out"
grep -q '^removed_timeline_ids=timeline-prune-0003$' "$TMP_DIR/apply.out"
grep -q '^retained_timeline_recovery_wal_records=4$' "$TMP_DIR/apply.out"
test ! -e "$fixture/timeline-prune/TIMELINE"
test ! -e "$fixture/timeline-prune/MANIFEST"
test -e "$fixture/timeline-keep/TIMELINE"
test -e "$fixture/timeline-keep/MANIFEST"

stale_fixture="$TMP_DIR/stale-fixture"
cargo run -q -p gpu_db_engine --example wal_archive_maintenance_preflight -- \
  --write-demo-fixture "$stale_fixture" >"$TMP_DIR/stale-fixture.out"
sha256sum "$stale_fixture/archive/MANIFEST" "$stale_fixture/TIMELINE_REGISTRY" \
  >"$TMP_DIR/stale-before.sha256"
sed -i 's/^fork_txn_id=4$/fork_txn_id=3/' "$stale_fixture/timeline-keep/TIMELINE"

if run_preflight "$stale_fixture" --apply >"$TMP_DIR/stale.out" 2>"$TMP_DIR/stale.err"; then
  echo "expected stale sidecar maintenance preflight to fail" >&2
  exit 1
fi
grep -q 'does not match sidecar' "$TMP_DIR/stale.err"
sha256sum "$stale_fixture/archive/MANIFEST" "$stale_fixture/TIMELINE_REGISTRY" \
  >"$TMP_DIR/stale-after.sha256"
diff -u "$TMP_DIR/stale-before.sha256" "$TMP_DIR/stale-after.sha256"

recent_fixture="$TMP_DIR/recent-fixture"
cargo run -q -p gpu_db_engine --example wal_archive_maintenance_preflight -- \
  --write-demo-fixture "$recent_fixture" >"$TMP_DIR/recent-fixture.out"
sha256sum "$recent_fixture/archive/MANIFEST" "$recent_fixture/TIMELINE_REGISTRY" \
  >"$TMP_DIR/recent-before.sha256"
if cargo run -q -p gpu_db_engine --example wal_archive_maintenance_preflight -- \
  --control "$recent_fixture/base/CONTROL" \
  --archive-manifest "$recent_fixture/archive/MANIFEST" \
  --timeline-registry "$recent_fixture/TIMELINE_REGISTRY" \
  --retain-timeline timeline-keep-0002 \
  --current-timestamp-micros 6000 \
  --pitr-window-micros 5000 \
  --recover-retained \
  --apply >"$TMP_DIR/recent.out" 2>"$TMP_DIR/recent.err"; then
  echo "expected unsafe recent-base maintenance preflight to fail" >&2
  exit 1
fi
grep -q 'newer than PITR retention cutoff' "$TMP_DIR/recent.err"
sha256sum "$recent_fixture/archive/MANIFEST" "$recent_fixture/TIMELINE_REGISTRY" \
  >"$TMP_DIR/recent-after.sha256"
diff -u "$TMP_DIR/recent-before.sha256" "$TMP_DIR/recent-after.sha256"

echo "wal archive maintenance preflight smoke passed"
