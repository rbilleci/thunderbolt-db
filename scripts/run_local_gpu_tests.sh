#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# These crates carry the #[ignore]-gated end-to-end GPU tests
# ("requires a local NVIDIA driver and GPU" / "requires local NVIDIA driver
# and CUDA-capable hardware"). Every #[ignore] in them is GPU-gated, so we run
# the whole ignored set, including the resident parity gate
# (cuda_resident_i32_equal_any_project_submit_complete_matches_sync) and the
# pinned host buffer pool isolation test
# (gpu_pinned_host_buffer_pool_reuses_buffers_and_isolates_concurrent_leases).
#
# Run one libtest case at a time: these tests share one physical CUDA device
# and deliberately exercise global memory pressure, lifecycle, and failure
# controls whose simultaneous execution would interfere with one another.
GPU_TEST_CRATES=(-p gpu_db_execution -p gpu_db_engine)

# Harmless on a non-GPU box: skip cleanly and succeed so this can sit in shared
# preflights without forcing a GPU on every host.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "local_gpu_tests=skipped reason=no_gpu_detected"
  exit 0
fi

OUT_DIR="${LOCAL_GPU_TESTS_OUT_DIR:-"$ROOT/target/local-gpu-tests"}"
LOG="$OUT_DIR/local-gpu-tests.log"
mkdir -p "$OUT_DIR"

git_sha="$(git rev-parse HEAD)"
device_info="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' -)"

echo "local_gpu_tests_git_sha=${git_sha}"
echo "local_gpu_tests_device_info=${device_info}"

# Preserve the cargo exit code through tee so a failing GPU test fails this
# script (pipefail is on via `set -euo pipefail`).
cargo test "${GPU_TEST_CRATES[@]}" -- --ignored --color never --test-threads=1 2>&1 | tee "$LOG"

# Roll the per-binary `test result:` lines up into one summary so the preflight
# can grep for stable counts.
passed_total=0
failed_total=0
ignored_total=0
while read -r passed failed ignored; do
  passed_total=$((passed_total + passed))
  failed_total=$((failed_total + failed))
  ignored_total=$((ignored_total + ignored))
done < <(
  sed -n 's/^test result:.* \([0-9]\+\) passed; \([0-9]\+\) failed; \([0-9]\+\) ignored.*/\1 \2 \3/p' "$LOG"
)

echo "local_gpu_tests_log=${LOG}"
echo "local_gpu_tests_passed_count=${passed_total}"
echo "local_gpu_tests_failed_count=${failed_total}"
echo "local_gpu_tests_skipped_in_run_count=${ignored_total}"
echo "local_gpu_tests=passed"
