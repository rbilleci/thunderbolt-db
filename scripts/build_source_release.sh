#!/usr/bin/env bash
#
# Produce a reproducible GPL source archive from one committed tree.  This
# intentionally archives Git content only, so build products (including
# target/) and other local state cannot enter the release artifact.

set -euo pipefail

readonly SCRIPT_NAME="build_source_release.sh"
readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Keep this manifest aligned with the notice references in THIRD_PARTY_NOTICES.
# It is used for both the committed tree and the final archive checks.
readonly REQUIRED_RELEASE_FILES=(
  LICENSE
  COPYRIGHT
  CUDA_EXCEPTION
  THIRD_PARTY_NOTICES.md
  THIRD_PARTY_LICENSES/AWS-LC.txt
  THIRD_PARTY_LICENSES/MPL-2.0.txt
  THIRD_PARTY_LICENSES/PostgreSQL.txt
  THIRD_PARTY_LICENSES/jemalloc.txt
  THIRD_PARTY_LICENSES/libpg_query-BSD-3-Clause.txt
  THIRD_PARTY_LICENSES/pg_query-MIT.txt
  THIRD_PARTY_LICENSES/protobuf-c-BSD-2-Clause.txt
  THIRD_PARTY_LICENSES/xxHash-BSD-2-Clause.txt
)

metadata_dir=""
temporary_archive=""

die() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

cleanup() {
  local status=$?
  if [[ -n "$temporary_archive" && -f "$temporary_archive" ]]; then
    rm -f -- "$temporary_archive"
  fi
  if [[ -n "$metadata_dir" && -d "$metadata_dir" ]]; then
    rm -rf -- "$metadata_dir"
  fi
  exit "$status"
}
trap cleanup EXIT

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

cd "$ROOT_DIR"
require_command cargo
require_command git
require_command gzip
require_command python3
require_command sha256sum
require_command stat
require_command tar

requested_ref="${RELEASE_REF:-HEAD}"
commit="$(git rev-parse --verify "${requested_ref}^{commit}")" || \
  die "RELEASE_REF does not resolve to a commit: ${requested_ref}"

for required_file in "${REQUIRED_RELEASE_FILES[@]}"; do
  git cat-file -e "${commit}:${required_file}" 2>/dev/null || \
    die "release commit is missing required file: ${required_file}"
done

# Cargo metadata runs against a Git-exported copy of the exact release commit,
# rather than the caller's possibly dirty checkout.  --no-deps keeps this a
# manifest check; --locked proves the exported manifest matches its lockfile.
metadata_dir="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-source-release-metadata.XXXXXX")"
git archive --format=tar "$commit" | tar -xf - -C "$metadata_dir"
metadata_json="$(cargo metadata --locked --offline --no-deps --format-version=1 \
  --manifest-path "$metadata_dir/Cargo.toml")"
version="$(printf '%s' "$metadata_json" | python3 -c '
import json
import sys

metadata = json.load(sys.stdin)
workspace_members = set(metadata["workspace_members"])
versions = {
    package["version"]
    for package in metadata["packages"]
    if package["id"] in workspace_members
}
if len(versions) != 1:
    raise SystemExit(
        "workspace members must have one release version; found " + repr(sorted(versions))
    )
print(versions.pop())
')" || die "could not determine one workspace version from Cargo metadata"
[[ -n "$version" ]] || die "Cargo metadata returned an empty workspace version"

if [[ -n "${RELEASE_OUT_DIR:-}" ]]; then
  [[ -d "$RELEASE_OUT_DIR" && -w "$RELEASE_OUT_DIR" ]] || \
    die "RELEASE_OUT_DIR must be an existing writable caller-owned directory"
  release_out_dir="$(cd -- "$RELEASE_OUT_DIR" && pwd -P)"
else
  mkdir -p -- "$ROOT_DIR/target/releases"
  release_out_dir="$(cd -- "$ROOT_DIR/target/releases" && pwd -P)"
fi

archive_name="gpu-database-engine-${version}.tar.gz"
archive_path="${release_out_dir}/${archive_name}"
[[ ! -e "$archive_path" ]] || die "refusing to overwrite existing archive: ${archive_path}"
prefix="gpu-database-engine-${version}"
temporary_archive="$(mktemp "${release_out_dir}/.${archive_name}.XXXXXX")"

# Historical research reports and build output are not source-release inputs.
# The final listing check below makes a future pathspec regression fail rather
# than silently enlarge the artifact.
git archive --format=tar --prefix="${prefix}/" "$commit" -- . \
  ':(exclude)target/**' \
  ':(exclude)docs/archive/research/**' | gzip -n >"$temporary_archive"

while IFS= read -r archive_entry; do
  case "$archive_entry" in
    "${prefix}/target/"* | "${prefix}/docs/archive/research/"*)
      die "source archive unexpectedly contains excluded path: ${archive_entry}"
      ;;
  esac
done < <(tar -tzf "$temporary_archive")
archive_listing="$(tar -tzf "$temporary_archive")"
for required_file in "${REQUIRED_RELEASE_FILES[@]}"; do
  grep -Fx "${prefix}/${required_file}" <<<"$archive_listing" >/dev/null || \
    die "source archive is missing required file: ${required_file}"
done

mv -- "$temporary_archive" "$archive_path"
temporary_archive=""
archive_sha256="$(sha256sum "$archive_path" | awk '{print $1}')"
archive_size="$(stat -c '%s' "$archive_path")"

printf 'source_release_ref=%s\n' "$commit"
printf 'source_release_version=%s\n' "$version"
printf 'source_release_path=%s\n' "$archive_path"
printf 'source_release_sha256=%s\n' "$archive_sha256"
printf 'source_release_size_bytes=%s\n' "$archive_size"
printf 'source_release_completion=passed\n'
