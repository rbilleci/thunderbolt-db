#!/usr/bin/env bash
#
# STANDARD BENCHMARK REPORT CARD -- GPU-native OLTP database.
#
# A recurring canonical artifact. It ALWAYS reports BOTH layers x BOTH cache regimes:
#
#   Layer 1 -- RAW READ KERNELS         (crates/execution -- read_kernel_roofline)
#   Layer 2 -- lpb/wave POINT-READ PATH (crates/engine    -- r2_wave_engine_ab)
#
#   Cache regime IN-L2     : gathered i32 column FITS this card's L2 (cache-resident, flattered).
#   Cache regime OUT-OF-L2 : gathered i32 column EXCEEDS L2 (HBM/GDDR7-bound, honest).
#
# Every line in every section reports BOTH p50 latency AND throughput (both examples already do).
#
# This card's L2 = 128 MB (measured via cudaDevAttrL2CacheSize; RTX PRO 6000 Blackwell Max-Q, 96GB).
# OUT-OF-L2 needs the gathered i32 column (4B/row) to exceed 128MB => > 32M rows.
#   - read_kernel_roofline emits IN-L2 (32MB/col, 8M rows) + OUT-OF-L2 (256MB/col, 64M rows) in ONE run.
#   - r2_wave_engine_ab builds its table via a CPU-bound SQL INSERT loop. At 48M rows (192MB/col =
#     1.5x L2 -> clearly out-of-L2), the final R3-004 tree retains each device-authoritative insert
#     publication instead of late-converting the fixture. A 2026-07-18 run measured 1620.4s insert +
#     0.0s final residency; the former 1200s timeout expired during the build.
#
# Discipline: this is a shared GPU box. Never run two GPU examples back-to-back without a gap (sleep 12).
# Never pass --gpu-reset anywhere. set -uo pipefail (NOT -e) so a timeout in one section does not abort
# the whole card.
#
# Run:  scripts/benchmark_report_card.sh
#
# Tunables (env overrides, with the defaults this card was calibrated to):
#   OUT_OF_L2_ROWS     out-of-L2 lpb table size           (default 48000000 = 192MB/col = 1.5x L2)
#   OUT_OF_L2_BATCHES  measured batches for the large pass (default 300; p50 is stable at ~300)
#   SECTION_A_TIMEOUT  roofline timeout, seconds           (default 280)
#   SECTION_B_TIMEOUT  in-L2 engine timeout, seconds       (default 280)
#   SECTION_C_TIMEOUT  out-of-L2 engine timeout, seconds   (default 2400; measured 1620.4s build + sweeps)
#   GPU_GAP            inter-section GPU cool-down, seconds (default 12)

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

OUT_OF_L2_ROWS="${OUT_OF_L2_ROWS:-48000000}"
OUT_OF_L2_BATCHES="${OUT_OF_L2_BATCHES:-300}"
SECTION_A_TIMEOUT="${SECTION_A_TIMEOUT:-280}"
SECTION_B_TIMEOUT="${SECTION_B_TIMEOUT:-280}"
SECTION_C_TIMEOUT="${SECTION_C_TIMEOUT:-2400}"
GPU_GAP="${GPU_GAP:-12}"

l2_mb=128
out_of_l2_col_mb=$(( OUT_OF_L2_ROWS * 4 / 1000000 ))

device_info="nvidia-smi unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  device_info="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
fi

echo "########################################################################################"
echo "# STANDARD BENCHMARK REPORT CARD -- GPU-native OLTP database"
echo "# date    : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "# device  : ${device_info}"
echo "# L2 cache: ${l2_mb} MB  [cudaDevAttrL2CacheSize, RTX PRO 6000 Blackwell Max-Q]"
echo "#"
echo "# COVERS: 2 layers x 2 cache regimes, p50 latency + throughput on EVERY line."
echo "#   Layer 1  RAW READ KERNELS         (Section A: IN-L2 + OUT-OF-L2 in one run)"
echo "#   Layer 2  lpb/wave POINT-READ PATH (Section B: IN-L2  |  Section C: OUT-OF-L2)"
echo "#   IN-L2 = gathered i32 col fits ${l2_mb}MB L2 (flattered);  OUT-OF-L2 = exceeds it (HBM-bound)."
echo "#   Section C size = ${OUT_OF_L2_ROWS} rows = ${out_of_l2_col_mb}MB/i32-col (out-of-L2), batches=${OUT_OF_L2_BATCHES}."
echo "########################################################################################"

run_section() {
  # run_section <id> <title> <timeout_secs> <cmd...>
  local id="$1"; shift
  local title="$1"; shift
  local tmo="$1"; shift
  echo ""
  echo "### SECTION ${id} -- ${title}"
  echo "### cmd: timeout ${tmo} $*"
  echo "### ----------------------------------------------------------------------------------"
  timeout "${tmo}" "$@"
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    if [ "${rc}" -eq 124 ]; then
      echo "[section ${id} TIMED OUT after ${tmo}s rc=${rc}]"
    else
      echo "[section ${id} FAILED rc=${rc}]"
    fi
  fi
  return 0
}

# ---- Section A: RAW READ KERNELS (emits IN-L2 + OUT-OF-L2 in one invocation) ----
run_section A "RAW READ KERNELS (read_kernel_roofline -- IN-L2 + OUT-OF-L2)" "${SECTION_A_TIMEOUT}" \
  cargo run --release --example read_kernel_roofline -p gpu_db_execution

echo ""
echo "### GPU cool-down: sleep ${GPU_GAP}"
sleep "${GPU_GAP}"

# ---- Section B: lpb/wave ENGINE POINT READS, IN-L2 (default 1M rows; full concurrent section) ----
run_section B "lpb/wave ENGINE POINT READS -- IN-L2 (1M rows = 4MB/col, cache-resident)" "${SECTION_B_TIMEOUT}" \
  env GPU_DB_BENCH_ROWS=1048576 \
  cargo run --release --example r2_wave_engine_ab -p gpu_db_engine

echo ""
echo "### GPU cool-down: sleep ${GPU_GAP}"
sleep "${GPU_GAP}"

# ---- Section C: lpb/wave ENGINE POINT READS, OUT-OF-L2 (large table; skip concurrent; fewer batches) ----
run_section C "lpb/wave ENGINE POINT READS -- OUT-OF-L2 (${OUT_OF_L2_ROWS} rows = ${out_of_l2_col_mb}MB/col, HBM-bound)" "${SECTION_C_TIMEOUT}" \
  env GPU_DB_BENCH_ROWS="${OUT_OF_L2_ROWS}" GPU_DB_BENCH_BATCHES="${OUT_OF_L2_BATCHES}" GPU_DB_BENCH_THREADS="" \
  cargo run --release --example r2_wave_engine_ab -p gpu_db_engine

echo ""
echo "########################################################################################"
echo "# END REPORT CARD"
echo "########################################################################################"
