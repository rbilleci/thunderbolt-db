#!/usr/bin/env bash
#
# WRITE-001 PostgreSQL semantic differential.
#
# This preserves the accepted INSERT-001 17-case comparison as a repeatable
# frozen-candidate gate. It deliberately exercises real simple-query pgwire
# bytes against both the GPU server and the pinned PostgreSQL 16 container.
# Extended/bound ingress remains covered by the mandatory tokio-postgres gate;
# this script is the independent PostgreSQL oracle for literal/simple results
# and SQLSTATEs.

set -euo pipefail

readonly SCRIPT_NAME="write001_postgresql_differential.sh"
readonly POSTGRES_IMAGE="postgres@sha256:33f923b05f64ca54ac4401c01126a6b92afe839a0aa0a52bc5aeb5cc958e5f20"
readonly GPU_PORT="56410"
readonly POSTGRES_PORT="56411"

artifact_arg=""

usage() {
  cat <<'USAGE'
usage: scripts/write001_postgresql_differential.sh [--artifact-dir DIR]

Runs the preserved 17-case INSERT-001 PostgreSQL differential on an exact
frozen candidate. The index may contain the intended candidate, but the
worktree must have no unstaged or untracked files. The result is written to a
new artifact directory (default: target/write001-postgresql-differential.*).
USAGE
}

die() {
  echo "${SCRIPT_NAME}: $*" >&2
  exit 2
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      [[ "$#" -ge 2 ]] || die "--artifact-dir requires a value"
      artifact_arg="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

preflight_unstaged_paths="$(git diff --name-only | wc -l)"
preflight_untracked_paths="$(git ls-files --others --exclude-standard | wc -l)"
[[ "$preflight_unstaged_paths" == "0" ]] || die "candidate has unstaged changes; stage the exact candidate first"
[[ "$preflight_untracked_paths" == "0" ]] || die "candidate has untracked files; reconcile them before the differential"

if [[ -n "$artifact_arg" ]]; then
  artifact="$(realpath -m "$artifact_arg")"
  [[ "$artifact" == "$repo_root"/target/* ]] || die "--artifact-dir must be below $repo_root/target"
  [[ ! -e "$artifact" ]] || die "artifact directory already exists: $artifact"
  mkdir -- "$artifact"
else
  artifact="$(mktemp -d "$repo_root/target/write001-postgresql-differential.XXXXXX")"
fi

container="write001-pgdiff-$$"
gpu_pid=""

cleanup() {
  if [[ -n "$gpu_pid" ]] && kill -0 "$gpu_pid" 2>/dev/null; then
    kill -TERM "$gpu_pid" 2>/dev/null || true
    wait "$gpu_pid" 2>/dev/null || true
  fi
  if docker container inspect "$container" >/dev/null 2>&1; then
    docker logs "$container" >"$artifact/postgres-server.log" 2>&1 || true
    docker rm --force "$container" >"$artifact/postgres-remove.log" 2>&1 || true
  fi
}
trap cleanup EXIT

command -v cargo >/dev/null || die "cargo is required"
command -v docker >/dev/null || die "docker is required"
command -v psql >/dev/null || die "psql is required"
command -v pg_isready >/dev/null || die "pg_isready is required"

index_tree="$(git write-tree)"
cached_diff_sha256="$(git diff --cached --binary | sha256sum | awk '{print $1}')"
unstaged_paths="$(git diff --name-only | wc -l)"
untracked_paths="$(git ls-files --others --exclude-standard | wc -l)"
{
  printf 'label=start\n'
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  printf 'index_tree=%s\n' "$index_tree"
  printf 'cached_diff_sha256=%s\n' "$cached_diff_sha256"
  printf 'unstaged_paths=%s\n' "$unstaged_paths"
  printf 'untracked_paths=%s\n' "$untracked_paths"
} >"$artifact/seal-start.txt"

[[ "$unstaged_paths" == "0" && "$untracked_paths" == "0" ]] ||
  die "the artifact directory is not ignored; candidate seal would be ambiguous"

cargo build -p gpu_db_server --bin gpu-db-engine-server \
  >"$artifact/build.stdout" 2>"$artifact/build.stderr"
sha256sum target/debug/gpu-db-engine-server | awk '{print $1}' >"$artifact/gpu-server.sha256"

target/debug/gpu-db-engine-server --listen "127.0.0.1:${GPU_PORT}" \
  >"$artifact/gpu-server.stdout" 2>"$artifact/gpu-server.stderr" &
gpu_pid=$!

docker run --detach --name "$container" \
  --env POSTGRES_HOST_AUTH_METHOD=trust \
  --publish "127.0.0.1:${POSTGRES_PORT}:5432" \
  "$POSTGRES_IMAGE" >"$artifact/postgres-container-id"

gpu_ready=0
postgres_ready=0
for _ in $(seq 1 200); do
  if [[ "$gpu_ready" == "0" ]] &&
    timeout 1 bash -c 'exec 3<>"/dev/tcp/$1/$2"' -- 127.0.0.1 "$GPU_PORT" 2>/dev/null; then
    gpu_ready=1
  fi
  if [[ "$postgres_ready" == "0" ]] &&
    pg_isready -h 127.0.0.1 -p "$POSTGRES_PORT" -U postgres -d postgres \
      >"$artifact/postgres-ready.log" 2>&1; then
    postgres_ready=1
  fi
  [[ "$gpu_ready" == "1" && "$postgres_ready" == "1" ]] && break
  sleep 0.1
done
[[ "$gpu_ready" == "1" && "$postgres_ready" == "1" ]] || die "GPU or PostgreSQL server did not become ready"

names=(
  create_plain
  insert_typed_multi
  read_typed_multi
  atomic_multi_overflow
  read_after_overflow
  duplicate_column
  unknown_column
  insert_reordered
  insert_null
  insert_returning
  transaction_rollback
  read_final_plain
  create_pk
  seed_pk
  atomic_unique
  atomic_not_null
  read_guarded
)
kinds=(
  success success success failure success failure failure success success success success success
  success success failure failure success
)
states=(
  none none none 22003 none 42701 42703 none none none none none none none 23505 23502 none
)
sqls=(
  'CREATE TABLE diff_i32 (a int4, b int4)'
  'INSERT INTO diff_i32 (a,b) VALUES (1,10),(2,20)'
  'SELECT a,b FROM diff_i32 ORDER BY a'
  'INSERT INTO diff_i32 (a,b) VALUES (3,30),(2147483648,40)'
  'SELECT a,b FROM diff_i32 ORDER BY a'
  'INSERT INTO diff_i32 (a,a) VALUES (3,30)'
  'INSERT INTO diff_i32 (missing,b) VALUES (3,30)'
  'INSERT INTO diff_i32 (b,a) VALUES (30,3)'
  'INSERT INTO diff_i32 (a,b) VALUES (4,NULL)'
  'INSERT INTO diff_i32 (a,b) VALUES (5,50) RETURNING a,b'
  'BEGIN; INSERT INTO diff_i32 (a,b) VALUES (6,60); ROLLBACK;'
  'SELECT a,b FROM diff_i32 ORDER BY a'
  'CREATE TABLE guarded_i32 (a int4 PRIMARY KEY, b int4)'
  'INSERT INTO guarded_i32 (a,b) VALUES (1,10)'
  'INSERT INTO guarded_i32 (a,b) VALUES (2,20),(1,99)'
  'INSERT INTO guarded_i32 (a,b) VALUES (2,20),(NULL,30)'
  'SELECT a,b FROM guarded_i32 ORDER BY a'
)

printf 'case\tkind\texpected_sqlstate\tsql\n' >"$artifact/cases.tsv"
: >"$artifact/results.txt"

for i in "${!names[@]}"; do
  ordinal="$(printf '%02d' "$((i + 1))")"
  name="${names[$i]}"
  kind="${kinds[$i]}"
  expected_state="${states[$i]}"
  sql="${sqls[$i]}"
  prefix="$artifact/${ordinal}_${name}"
  printf '%s\t%s\t%s\t%s\n' "$name" "$kind" "$expected_state" "$sql" >>"$artifact/cases.tsv"

  gpu_url="postgresql://postgres@127.0.0.1:${GPU_PORT}/postgres?sslmode=disable"
  postgres_url="postgresql://postgres@127.0.0.1:${POSTGRES_PORT}/postgres?sslmode=disable"

  set +e
  PGCONNECT_TIMEOUT=5 psql -X -A -t -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$gpu_url" -c "$sql" >"${prefix}.gpu.stdout" 2>"${prefix}.gpu.stderr"
  gpu_rc=$?
  PGCONNECT_TIMEOUT=5 psql -X -A -t -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$postgres_url" -c "$sql" >"${prefix}.postgresql.stdout" 2>"${prefix}.postgresql.stderr"
  postgres_rc=$?
  set -e
  printf '%s\n' "$gpu_rc" >"${prefix}.gpu.rc"
  printf '%s\n' "$postgres_rc" >"${prefix}.postgresql.rc"

  if [[ "$kind" == "success" ]]; then
    [[ "$gpu_rc" == "0" && "$postgres_rc" == "0" ]]
    diff -u "${prefix}.postgresql.stdout" "${prefix}.gpu.stdout" >"${prefix}.stdout.diff"
    printf '%s\tpass\tstdout-byte-equal\n' "$name" >>"$artifact/results.txt"
  else
    [[ "$gpu_rc" != "0" && "$postgres_rc" != "0" ]]
    gpu_state="$(sed -n 's/^ERROR:  \([0-9A-Z][0-9A-Z]*\):.*/\1/p' "${prefix}.gpu.stderr" | head -n 1)"
    postgres_state="$(sed -n 's/^ERROR:  \([0-9A-Z][0-9A-Z]*\):.*/\1/p' "${prefix}.postgresql.stderr" | head -n 1)"
    [[ "$gpu_state" == "$expected_state" ]]
    [[ "$postgres_state" == "$expected_state" ]]
    printf '%s\tpass\tsqlstate=%s\n' "$name" "$expected_state" >>"$artifact/results.txt"
  fi
done

post_index_tree="$(git write-tree)"
post_cached_diff_sha256="$(git diff --cached --binary | sha256sum | awk '{print $1}')"
post_unstaged_paths="$(git diff --name-only | wc -l)"
post_untracked_paths="$(git ls-files --others --exclude-standard | wc -l)"
[[ "$post_index_tree" == "$index_tree" ]] || die "candidate index changed during differential"
[[ "$post_cached_diff_sha256" == "$cached_diff_sha256" ]] || die "candidate staged diff changed during differential"
[[ "$post_unstaged_paths" == "0" ]] || die "candidate gained unstaged changes during differential"
[[ "$post_untracked_paths" == "0" ]] || die "candidate gained untracked files during differential"

{
  printf 'label=end\n'
  printf 'index_tree=%s\n' "$post_index_tree"
  printf 'cached_diff_sha256=%s\n' "$post_cached_diff_sha256"
  printf 'unstaged_paths=%s\n' "$post_unstaged_paths"
  printf 'untracked_paths=%s\n' "$post_untracked_paths"
} >"$artifact/seal-end.txt"

{
  printf 'write001_postgresql_differential_status=complete '
  printf 'cases=17 backends=gpu,postgresql success_stdout=byte_equal '
  printf 'failure_sqlstates=22003,42701,42703,23505,23502 '
  printf 'null=pass rollback=pass returning=pass atomicity=pass '
  printf 'gpu_server_sha256=%s ' "$(cat "$artifact/gpu-server.sha256")"
  printf 'index_tree=%s cached_diff_sha256=%s ' "$index_tree" "$cached_diff_sha256"
  printf 'artifact=%s\n' "$artifact"
} >"$artifact/status.txt"

sha256sum "$artifact"/status.txt "$artifact"/cases.tsv "$artifact"/results.txt \
  "$artifact"/gpu-server.sha256 >"$artifact/evidence.sha256"
cat "$artifact/status.txt"
