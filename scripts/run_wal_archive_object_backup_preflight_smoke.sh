#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cd "$ROOT"

fixture="$TMP_DIR/fixture"
cargo run -q -p gpu_db_engine --example wal_archive_object_backup_preflight -- \
  --write-demo-fixture "$fixture" >"$TMP_DIR/fixture.out"

cargo run -q -p gpu_db_engine --example wal_archive_object_backup_preflight -- \
  --archive-manifest "$fixture/archive/MANIFEST" \
  --backup-manifest "$fixture/backup/BACKUP" \
  --object-dir "$fixture/backup/objects" \
  --restored-manifest "$fixture/restored/MANIFEST" \
  --restored-segment-dir "$fixture/restored/segments" \
  --recover-timestamp-micros 3000 >"$TMP_DIR/backup.out"

grep -q '^mode=export-restore$' "$TMP_DIR/backup.out"
grep -q '^source_record_count=3$' "$TMP_DIR/backup.out"
grep -q '^object_count=4$' "$TMP_DIR/backup.out"
grep -q '^restored_record_count=3$' "$TMP_DIR/backup.out"
grep -q '^restored_segments=3$' "$TMP_DIR/backup.out"
grep -q '^recovered_timestamp_micros=3000$' "$TMP_DIR/backup.out"
grep -q '^recovered_wal_records=3$' "$TMP_DIR/backup.out"
grep -q '^recovered_grace_rows=1$' "$TMP_DIR/backup.out"
test -e "$fixture/restored/MANIFEST"
test -e "$fixture/restored/segments"

corrupt_fixture="$TMP_DIR/corrupt-fixture"
cargo run -q -p gpu_db_engine --example wal_archive_object_backup_preflight -- \
  --write-demo-fixture "$corrupt_fixture" >"$TMP_DIR/corrupt-fixture.out"
cargo run -q -p gpu_db_engine --example wal_archive_object_backup_preflight -- \
  --archive-manifest "$corrupt_fixture/archive/MANIFEST" \
  --backup-manifest "$corrupt_fixture/backup/BACKUP" \
  --object-dir "$corrupt_fixture/backup/objects" \
  --restored-manifest "$corrupt_fixture/restored-good/MANIFEST" \
  --restored-segment-dir "$corrupt_fixture/restored-good/segments" \
  >"$TMP_DIR/corrupt-export.out"
printf '\ncorruption\n' >>"$corrupt_fixture/backup/objects/segment-0003.wal.object"

if cargo run -q -p gpu_db_engine --example wal_archive_object_backup_preflight -- \
  --restore-only \
  --backup-manifest "$corrupt_fixture/backup/BACKUP" \
  --restored-manifest "$corrupt_fixture/restored-corrupt/MANIFEST" \
  --restored-segment-dir "$corrupt_fixture/restored-corrupt/segments" \
  >"$TMP_DIR/corrupt.out" 2>"$TMP_DIR/corrupt.err"; then
  echo "expected corrupt object restore to fail" >&2
  exit 1
fi
grep -Eq 'checksum mismatch|expected [0-9]+ bytes but read' "$TMP_DIR/corrupt.err"
test ! -e "$corrupt_fixture/restored-corrupt/MANIFEST"
test ! -e "$corrupt_fixture/restored-corrupt/segments"

echo "wal archive object backup preflight smoke passed"
