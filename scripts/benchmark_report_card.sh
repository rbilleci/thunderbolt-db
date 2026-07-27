#!/usr/bin/env bash
#
# STANDARD BENCHMARK REPORT CARD -- GPU-native OLTP database.
#
# Full mode is the recurring canonical artifact and ALWAYS reports BOTH layers x BOTH cache regimes.
# Quick mode is a non-acceptance development screen that runs Sections A+B and never substitutes for full mode.
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
# Discipline: this is a shared GPU box. One process-wide lock prevents overlapping report-card invocations.
# Never run two GPU examples back-to-back without a gap (sleep 12), and never pass --gpu-reset anywhere.
# Both modes build the examples once in a fresh target, record toolchain/candidate/artifact identities, and invoke
# those exact binaries. Full mode requires a frozen candidate (staged changes are allowed; unstaged and untracked
# files are not) and verifies that identity again at closeout. set -uo pipefail (NOT -e) keeps bounded diagnostics.
#
# Run:
#   scripts/benchmark_report_card.sh              # full (backward-compatible default)
#   scripts/benchmark_report_card.sh --full       # canonical acceptance artifact: Sections A+B+C
#   scripts/benchmark_report_card.sh --quick      # non-acceptance screen: Sections A+B only
#   scripts/benchmark_report_card.sh --self-check # no build/GPU; validates mode and fail-fast planning
#
# Full mode fixes every workload dimension below; ambient benchmark variables cannot weaken a canonical card.
# Operational tunables:
#   SECTION_A_TIMEOUT  roofline timeout, seconds           (default 280)
#   SECTION_B_TIMEOUT  in-L2 engine timeout, seconds       (default 280)
#   SECTION_C_TIMEOUT  out-of-L2 engine timeout, seconds   (default 2400; measured 1620.4s build + sweeps)
#   GPU_GAP            quick-screen cool-down, seconds      (default 12; full mode is fixed at 12)
#   BENCH_TARGET_DIR   caller-owned EMPTY build directory   (default: fresh target/benchmark-report-card.*)
#   BENCH_KEEP_TARGET  retain the auto-created target       (default 0; set to 1 for diagnosis)
#
# Acceptance floor:
#   Section B batch-65,536 production-compact whole-run wall throughput must be
#   at least 260,000,000 lookups/s. Missing, malformed, or duplicate metrics fail closed.

set -uo pipefail

readonly REPORT_CARD_GPU_LOCK_FILE="/tmp/gpu-db-benchmark-report-card.lock"
readonly FULL_RAW_ROWS="8388608"
readonly FULL_RAW_ROWS_LARGE="67108864"
readonly FULL_RAW_SORT_N="1048576"
readonly FULL_RAW_ITERS="20"
readonly FULL_RAW_ITERS_LARGE="10"
readonly FULL_POINT_ROWS_IN_L2="1048576"
readonly FULL_POINT_ROWS_OUT_OF_L2="48000000"
readonly FULL_POINT_BATCH_SIZES="1,8,32,256,4096,16384,65536"
readonly FULL_POINT_BATCHES_IN_L2="2000"
readonly FULL_POINT_BATCHES_OUT_OF_L2="300"
readonly FULL_POINT_WARMUP="20"
readonly FULL_POINT_THREADS_IN_L2="1,2,4,8"
readonly FULL_POINT_INSERT_CHUNK="1000"
readonly FULL_POINT_IN_L2_FLOOR_BATCH="65536"
readonly FULL_POINT_IN_L2_MIN_LOOKUPS_PER_S="260000000"
readonly FULL_GPU_GAP="12"

usage() {
  cat <<'USAGE'
usage: scripts/benchmark_report_card.sh [--full|--quick|--self-check|--help]

  --full        canonical clean-build Sections A+B+C card (default)
  --quick       non-acceptance clean-build Sections A+B development screen
  --self-check  validate mode planning without Cargo or GPU work
  --help        show this help
USAGE
}

mode_from_args() {
  if [[ "$#" -gt 1 ]]; then
    return 2
  fi
  if [[ "$#" -eq 0 ]]; then
    printf '%s\n' "full"
    return 0
  fi
  case "$1" in
    --full) printf '%s\n' "full" ;;
    --quick) printf '%s\n' "quick" ;;
    --self-check) printf '%s\n' "self-check" ;;
    --help|-h) printf '%s\n' "help" ;;
    *) return 2 ;;
  esac
}

sections_for_mode() {
  case "$1" in
    full) printf '%s\n' "A B C" ;;
    quick) printf '%s\n' "A B" ;;
    *) return 1 ;;
  esac
}

should_run_section_c() {
  local requested_mode="$1"
  local prior_failure_count="$2"
  [[ "$requested_mode" == "full" && "$prior_failure_count" -eq 0 ]]
}

section_output_complete() {
  local command_rc="$1"
  local tee_rc="$2"
  local completion_marker="$3"
  local section_log="$4"
  [[ "$command_rc" -eq 0 ]] &&
    [[ "$tee_rc" -eq 0 ]] &&
    grep -Fq "$completion_marker" "$section_log"
}

point_production_throughput_for_batch() {
  local section_log="$1"
  local target_batch="$2"
  awk -v target_batch="$target_batch" '
    $0 == "### batch=" target_batch {
      target_headers += 1
      in_target_batch = 1
      next
    }
    /^##/ {
      in_target_batch = 0
    }
    in_target_batch && $1 == "prod-compact" {
      prod_lines += 1
      for (field = 1; field <= NF; field += 1) {
        if ($field == "lookups/s") {
          candidate = $(field - 1)
          throughput_tokens += 1
          if ((candidate == "0" || candidate ~ /^[1-9][0-9]*$/) &&
              length(candidate) <= 10) {
            valid_tokens += 1
            throughput = candidate
          }
        }
      }
    }
    END {
      if (target_headers == 1 &&
          prod_lines == 1 &&
          throughput_tokens == 1 &&
          valid_tokens == 1) {
        print throughput
        exit 0
      }
      exit 1
    }
  ' "$section_log"
}

enforce_point_production_throughput_floor() {
  local section_log="$1"
  local target_batch="$2"
  local minimum_lookups_per_s="$3"
  local measured_lookups_per_s
  if ! [[
    "$minimum_lookups_per_s" =~ ^(0|[1-9][0-9]*)$ &&
      "${#minimum_lookups_per_s}" -le 10
  ]]; then
    echo "point_read_throughput_gate_status=fail cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s minimum=${minimum_lookups_per_s} reason=invalid_minimum"
    return 1
  fi
  if ! measured_lookups_per_s="$(
    point_production_throughput_for_batch "$section_log" "$target_batch"
  )"; then
    echo "point_read_throughput_gate_status=fail cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s minimum=${minimum_lookups_per_s} reason=missing_malformed_or_duplicate"
    return 1
  fi
  if ((10#$measured_lookups_per_s < 10#$minimum_lookups_per_s)); then
    echo "point_read_throughput_gate_status=fail cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s measured=${measured_lookups_per_s} minimum=${minimum_lookups_per_s} reason=below_floor"
    return 1
  fi
  echo "point_read_throughput_gate_status=pass cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s measured=${measured_lookups_per_s} minimum=${minimum_lookups_per_s}"
}

candidate_remained_frozen_in() {
  local candidate_repo="$1"
  local expected_head="$2"
  local expected_tree="$3"
  local expected_cached_diff_sha256="$4"
  local current_cached_diff_sha256
  [[ "$(git -C "$candidate_repo" rev-parse HEAD 2>/dev/null || echo unavailable)" == "$expected_head" ]] ||
    return 1
  [[ "$(git -C "$candidate_repo" write-tree 2>/dev/null || echo unavailable)" == "$expected_tree" ]] ||
    return 1
  current_cached_diff_sha256="$(
    git -C "$candidate_repo" diff --cached --binary | sha256sum | awk '{print $1}'
  )"
  [[ "$current_cached_diff_sha256" == "$expected_cached_diff_sha256" ]] || return 1
  git -C "$candidate_repo" diff --quiet --ignore-submodules -- || return 1
  [[ -z "$(git -C "$candidate_repo" ls-files --others --exclude-standard)" ]] || return 1
}

cleanup_owned_bench_target() {
  local cleanup_repo_root="$1"
  local cleanup_target="$2"
  local cleanup_owned="$3"
  local cleanup_keep="$4"
  if [[ "$cleanup_owned" -ne 1 ]]; then
    echo "# benchmark target retained (caller-owned): $cleanup_target"
    return 0
  fi
  if [[ "$cleanup_keep" == "1" ]]; then
    echo "# benchmark target retained (BENCH_KEEP_TARGET=1): $cleanup_target"
    return 0
  fi
  case "$cleanup_target" in
    "$cleanup_repo_root"/target/benchmark-report-card.??????)
      rm -rf -- "$cleanup_target"
      echo "# removed isolated benchmark target: $cleanup_target"
      ;;
    *)
      echo "refusing to remove unexpected benchmark target: $cleanup_target" >&2
      return 1
      ;;
  esac
}

run_section() {
  # run_section <id> <title> <timeout_secs> <completion_marker> <cmd...>
  local id="$1"; shift
  local title="$1"; shift
  local tmo="$1"; shift
  local completion_marker="$1"; shift
  local section_log="$bench_target_dir/report-card-section-${id}.log"
  local command_rc
  local tee_rc
  local -a pipeline_status
  echo ""
  echo "### SECTION ${id} -- ${title}"
  echo "### cmd: timeout ${tmo} $*"
  echo "### ----------------------------------------------------------------------------------"
  timeout "${tmo}" "$@" 2>&1 | tee "$section_log"
  pipeline_status=("${PIPESTATUS[@]}")
  command_rc="${pipeline_status[0]}"
  tee_rc="${pipeline_status[1]}"
  if [[ "$tee_rc" -ne 0 ]]; then
    echo "[section ${id} INCOMPLETE: output capture failed rc=${tee_rc}]"
    section_failures+=("${id}:output")
  elif [[ "$command_rc" -ne 0 ]]; then
    if [[ "$command_rc" -eq 124 ]]; then
      echo "[section ${id} TIMED OUT after ${tmo}s rc=${command_rc}]"
    else
      echo "[section ${id} FAILED rc=${command_rc}]"
    fi
    section_failures+=("${id}:rc=${command_rc}")
  elif ! section_output_complete "$command_rc" "$tee_rc" "$completion_marker" "$section_log"; then
    echo "[section ${id} INCOMPLETE: missing marker '${completion_marker}']"
    section_failures+=("${id}:marker")
  else
    echo "[section ${id} COMPLETE]"
  fi
  return 0
}

run_self_check() {
  local failures=0
  local scratch
  local self_repo
  local expected_head
  local expected_tree
  local expected_diff
  local first_lock_fd
  local second_lock_fd
  local saved_bench_target_dir="${bench_target_dir:-}"
  local -a saved_section_failures=("${section_failures[@]-}")
  scratch="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-report-card-self-check.XXXXXX")" || return 1

  [[ "$(mode_from_args)" == "full" ]] || failures=$((failures + 1))
  [[ "$(mode_from_args --full)" == "full" ]] || failures=$((failures + 1))
  [[ "$(mode_from_args --quick)" == "quick" ]] || failures=$((failures + 1))
  ! mode_from_args --invalid >/dev/null 2>&1 || failures=$((failures + 1))
  ! mode_from_args --full --quick >/dev/null 2>&1 || failures=$((failures + 1))
  [[ "$(sections_for_mode full)" == "A B C" ]] || failures=$((failures + 1))
  [[ "$(sections_for_mode quick)" == "A B" ]] || failures=$((failures + 1))
  ! sections_for_mode invalid >/dev/null 2>&1 || failures=$((failures + 1))
  should_run_section_c full 0 || failures=$((failures + 1))
  ! should_run_section_c full 1 || failures=$((failures + 1))
  ! should_run_section_c quick 0 || failures=$((failures + 1))

  bench_target_dir="$scratch"
  section_failures=()
  run_section PASS "self-check marker pass" 5 "control-marker=complete" \
    bash -c 'printf "%s\n" "control-marker=complete"' >/dev/null 2>&1
  [[ "${#section_failures[@]}" -eq 0 ]] || failures=$((failures + 1))
  run_section MARKER "self-check missing marker" 5 "control-marker=complete" \
    bash -c 'printf "%s\n" "control-marker=incomplete"' >/dev/null 2>&1
  [[ "${section_failures[*]}" == "MARKER:marker" ]] || failures=$((failures + 1))
  ! should_run_section_c full "${#section_failures[@]}" || failures=$((failures + 1))
  section_failures=()
  run_section COMMAND "self-check command failure" 5 "control-marker=complete" \
    bash -c 'exit 7' >/dev/null 2>&1
  [[ "${section_failures[*]}" == "COMMAND:rc=7" ]] || failures=$((failures + 1))
  section_failures=()
  bench_target_dir="$scratch/missing/capture"
  run_section OUTPUT "self-check tee failure" 5 "control-marker=complete" \
    bash -c 'printf "%s\n" "control-marker=complete"' >/dev/null 2>&1
  [[ "${section_failures[*]}" == "OUTPUT:output" ]] || failures=$((failures + 1))
  mkdir -p -- "$scratch/point-floor"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 260000000 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/pass.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 259999999 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/below.log"
  printf '%s\n' \
    "### batch=65536" \
    "  compat-public p50= 8000us p99= 9000us | 8000000 lookups/s (0.125 us/lookup)" \
    >"$scratch/point-floor/missing.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 270000000 lookups/s (0.004 us/lookup)" \
    "  prod-compact p50= 117us p99= 130us | 270000001 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/duplicate.log"
  printf '%s\n' \
    "### batch=65536" \
    "  compat-public p50= 8000us p99= 9000us | 8000000 lookups/s (0.125 us/lookup)" \
    "## ONE-CALLER summary" \
    "  prod-compact p50= 117us p99= 130us | 270000000 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/post-summary-decoy.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us | 270000000 lookups/s | 270000001 lookups/s" \
    >"$scratch/point-floor/two-tokens.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us | malformed lookups/s | 270000000 lookups/s" \
    >"$scratch/point-floor/malformed-and-valid.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 0259999999 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/zero-prefixed.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 270000000 lookups/s (0.004 us/lookup)" \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 270000000 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/duplicate-header.log"
  enforce_point_production_throughput_floor \
    "$scratch/point-floor/pass.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/below.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/missing.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/duplicate.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/post-summary-decoy.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/two-tokens.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/malformed-and-valid.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/zero-prefixed.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/duplicate-header.log" 65536 260000000 >/dev/null ||
    failures=$((failures + 1))
  ! enforce_point_production_throughput_floor \
    "$scratch/point-floor/pass.log" 65536 0260000000 >/dev/null ||
    failures=$((failures + 1))
  bench_target_dir="$saved_bench_target_dir"
  section_failures=("${saved_section_failures[@]}")

  if command -v flock >/dev/null 2>&1; then
    exec {first_lock_fd}>"$scratch/gpu.lock"
    exec {second_lock_fd}>"$scratch/gpu.lock"
    flock -n "$first_lock_fd" || failures=$((failures + 1))
    ! flock -n "$second_lock_fd" || failures=$((failures + 1))
    flock -u "$first_lock_fd"
    exec {first_lock_fd}>&-
    exec {second_lock_fd}>&-
  else
    failures=$((failures + 1))
  fi

  self_repo="$scratch/repo"
  mkdir -p -- "$self_repo"
  git -C "$self_repo" init -q
  printf '%s\n' "frozen" >"$self_repo/control.txt"
  git -C "$self_repo" add control.txt
  git -C "$self_repo" -c user.name=report-card -c user.email=report-card.invalid \
    commit -qm "self-check seed"
  expected_head="$(git -C "$self_repo" rev-parse HEAD)"
  expected_tree="$(git -C "$self_repo" write-tree)"
  expected_diff="$(
    git -C "$self_repo" diff --cached --binary | sha256sum | awk '{print $1}'
  )"
  candidate_remained_frozen_in "$self_repo" "$expected_head" "$expected_tree" "$expected_diff" ||
    failures=$((failures + 1))
  printf '%s\n' "drift" >>"$self_repo/control.txt"
  ! candidate_remained_frozen_in "$self_repo" "$expected_head" "$expected_tree" "$expected_diff" ||
    failures=$((failures + 1))

  mkdir -p -- "$scratch/cleanup-root/target/benchmark-report-card.ABC123"
  printf '%s\n' "owned" >"$scratch/cleanup-root/target/benchmark-report-card.ABC123/control"
  cleanup_owned_bench_target "$scratch/cleanup-root" \
    "$scratch/cleanup-root/target/benchmark-report-card.ABC123" 1 0 >/dev/null ||
    failures=$((failures + 1))
  [[ ! -e "$scratch/cleanup-root/target/benchmark-report-card.ABC123" ]] ||
    failures=$((failures + 1))

  rm -rf -- "$scratch"
  if [[ "$failures" -ne 0 ]]; then
    echo "benchmark report-card self-check failed: ${failures} assertion(s)" >&2
    return 1
  fi
  echo "benchmark report-card self-check passed"
}

mode="$(mode_from_args "$@")"
mode_rc=$?
if [[ "$mode_rc" -ne 0 ]]; then
  usage >&2
  exit 2
fi
case "$mode" in
  self-check)
    run_self_check
    exit $?
    ;;
  help)
    usage
    exit 0
    ;;
esac
planned_sections="$(sections_for_mode "$mode")" || exit 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

ambient_cargo_target_dir="${CARGO_TARGET_DIR:-<unset>}"
bench_target_owned=0
gpu_lock_file="$REPORT_CARD_GPU_LOCK_FILE"

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to serialize report-card runs on the shared GPU" >&2
  exit 2
fi
exec 9>"$gpu_lock_file"
if ! flock -n 9; then
  echo "another report-card run owns the GPU lock: $gpu_lock_file" >&2
  exit 75
fi

if [[ "${BENCH_KEEP_TARGET:-0}" != "0" && "${BENCH_KEEP_TARGET:-0}" != "1" ]]; then
  echo "BENCH_KEEP_TARGET must be 0 or 1" >&2
  exit 2
fi

git_head="$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
git_state="clean"
if [[ -n "$(git status --short 2>/dev/null)" ]]; then
  git_state="dirty"
fi

candidate_index_tree="unavailable"
candidate_cached_diff_sha256="unavailable"
if git rev-parse --git-dir >/dev/null 2>&1; then
  candidate_index_tree="$(git write-tree)"
  candidate_cached_diff_sha256="$(git diff --cached --binary | sha256sum | awk '{print $1}')"
fi

candidate_remained_frozen() {
  candidate_remained_frozen_in \
    "$repo_root" "$git_head" "$candidate_index_tree" "$candidate_cached_diff_sha256"
}

if [[ "$mode" == "full" ]]; then
  if ! candidate_remained_frozen; then
    echo "full mode requires a frozen candidate: stage/revert tracked changes and remove untracked files" >&2
    exit 2
  fi
fi

if [[ -n "${BENCH_TARGET_DIR:-}" ]]; then
  mkdir -p -- "$BENCH_TARGET_DIR"
  bench_target_dir="$(cd "$BENCH_TARGET_DIR" && pwd -P)"
  if [[ -n "$(find "$bench_target_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "BENCH_TARGET_DIR must be empty: $bench_target_dir" >&2
    exit 2
  fi
else
  mkdir -p -- "$repo_root/target"
  bench_target_dir="$(mktemp -d "$repo_root/target/benchmark-report-card.XXXXXX")" || exit 2
  bench_target_owned=1
fi

cleanup_bench_target() {
  cleanup_owned_bench_target \
    "$repo_root" "$bench_target_dir" "$bench_target_owned" "${BENCH_KEEP_TARGET:-0}"
}
trap cleanup_bench_target EXIT

export CARGO_TARGET_DIR="$bench_target_dir"
# prost-build and other native build helpers may expect the target-local temp directory to exist.
mkdir -p -- "$CARGO_TARGET_DIR/tmp"

SECTION_A_TIMEOUT="${SECTION_A_TIMEOUT:-280}"
SECTION_B_TIMEOUT="${SECTION_B_TIMEOUT:-280}"
SECTION_C_TIMEOUT="${SECTION_C_TIMEOUT:-2400}"
OUT_OF_L2_ROWS="$FULL_POINT_ROWS_OUT_OF_L2"
OUT_OF_L2_BATCHES="$FULL_POINT_BATCHES_OUT_OF_L2"
if [[ "$mode" == "full" ]]; then
  GPU_GAP="$FULL_GPU_GAP"
else
  GPU_GAP="${GPU_GAP:-$FULL_GPU_GAP}"
  if ! [[ "$GPU_GAP" =~ ^[0-9]+$ ]]; then
    echo "GPU_GAP must be a nonnegative integer in quick mode" >&2
    exit 2
  fi
fi

build_root="$repo_root"
if [[ "$mode" == "full" ]]; then
  if ! candidate_remained_frozen; then
    echo "candidate drifted before the source snapshot was captured" >&2
    exit 1
  fi
  build_root="$bench_target_dir/candidate-source"
  mkdir -p -- "$build_root"
  if ! git archive --format=tar "$candidate_index_tree" | tar -xf - -C "$build_root"; then
    echo "failed to export the exact staged candidate tree" >&2
    exit 1
  fi
fi
# The source tree's .cargo/config.toml maps TMPDIR to this source-relative path.
# Recreate it after an intentional target cleanup in both quick and full modes.
mkdir -p -- "$build_root/target/tmp"

l2_mb=128
out_of_l2_col_mb=$(( OUT_OF_L2_ROWS * 4 / 1000000 ))

device_info="nvidia-smi unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  device_info="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
fi

os_info="$(uname -srvmo)"
if [[ -r /etc/os-release ]]; then
  os_name="$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | sed 's/^"//;s/"$//')"
  os_info="${os_name}; ${os_info}"
fi

cc_command="${CC:-cc}"
cc_program="${cc_command%% *}"
cc_version="unavailable"
if command -v "$cc_program" >/dev/null 2>&1; then
  cc_version="$("$cc_program" --version 2>&1 | sed -n '1p')"
fi

echo "########################################################################################"
echo "# STANDARD BENCHMARK REPORT CARD -- GPU-native OLTP database"
echo "# mode    : ${mode}"
echo "# plan    : Sections ${planned_sections}"
echo "# date    : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "# device  : ${device_info}"
echo "# host    : ${os_info}"
echo "# git     : ${git_head} (${git_state})"
echo "# candidate index tree        : ${candidate_index_tree}"
echo "# candidate cached diff sha256: ${candidate_cached_diff_sha256}"
echo "# L2 cache: ${l2_mb} MB  [cudaDevAttrL2CacheSize, RTX PRO 6000 Blackwell Max-Q]"
echo "# target  : ${CARGO_TARGET_DIR} (fresh; ambient CARGO_TARGET_DIR=${ambient_cargo_target_dir})"
echo "# source  : ${build_root}"
echo "# GPU lock: ${gpu_lock_file}"
echo "# cargo   : $(cargo --version)"
echo "# rustc   : $(rustc --version)"
echo "# CC      : ${CC:-<unset>} -> ${cc_version}"
echo "#"
if [[ "$mode" == "full" ]]; then
  echo "# COVERS: 2 layers x 2 cache regimes, p50 latency + throughput on EVERY line."
  echo "#   Layer 1  RAW READ KERNELS         (Section A: IN-L2 + OUT-OF-L2 in one run)"
  echo "#   Layer 2  lpb/wave POINT-READ PATH (Section B: IN-L2  |  Section C: OUT-OF-L2)"
  echo "#   IN-L2 = gathered i32 col fits ${l2_mb}MB L2 (flattered); OUT-OF-L2 exceeds it."
  echo "#   Section C size = ${OUT_OF_L2_ROWS} rows = ${out_of_l2_col_mb}MB/i32-col, batches=${OUT_OF_L2_BATCHES}."
  echo "#   Canonical controls: raw=${FULL_RAW_ROWS}/${FULL_RAW_ROWS_LARGE} rows, iters=${FULL_RAW_ITERS}/${FULL_RAW_ITERS_LARGE},"
  echo "#     point batches=${FULL_POINT_BATCH_SIZES}, B/C samples=${FULL_POINT_BATCHES_IN_L2}/${FULL_POINT_BATCHES_OUT_OF_L2},"
  echo "#     warmup=${FULL_POINT_WARMUP}, B threads=${FULL_POINT_THREADS_IN_L2}, C threads=none, gap=${GPU_GAP}s."
else
  echo "# NON-CANONICAL QUICK SCREEN: Sections A+B only. This is NOT acceptance evidence."
  echo "#   Layer 1 still covers IN-L2 + OUT-OF-L2 raw kernels; Layer 2 covers IN-L2 only."
fi
echo "########################################################################################"

echo ""
echo "### CLEAN RELEASE BUILD -- canonical examples"
echo "### cd ${build_root} && cargo build --locked --release --example read_kernel_roofline -p gpu_db_execution"
if ! (
  cd "$build_root" &&
    cargo build --locked --release --example read_kernel_roofline -p gpu_db_execution
); then
  echo "[clean build FAILED: read_kernel_roofline]" >&2
  exit 1
fi
echo "### cd ${build_root} && cargo build --locked --release --example r2_wave_engine_ab -p gpu_db_engine"
if ! (
  cd "$build_root" &&
    cargo build --locked --release --example r2_wave_engine_ab -p gpu_db_engine
); then
  echo "[clean build FAILED: r2_wave_engine_ab]" >&2
  exit 1
fi

raw_binary="$CARGO_TARGET_DIR/release/examples/read_kernel_roofline"
point_binary="$CARGO_TARGET_DIR/release/examples/r2_wave_engine_ab"
if [[ ! -x "$raw_binary" || ! -x "$point_binary" ]]; then
  echo "clean build did not produce both expected example binaries" >&2
  exit 1
fi

print_artifact_identity() {
  local label="$1"
  local artifact="$2"
  local digest
  local bytes
  digest="$(sha256sum "$artifact" | awk '{print $1}')"
  bytes="$(stat -c '%s' "$artifact")"
  echo "# artifact: ${label} sha256=${digest} bytes=${bytes} path=${artifact}"
}

echo ""
echo "### BUILD ARTIFACT PROVENANCE"
print_artifact_identity "read_kernel_roofline" "$raw_binary"
print_artifact_identity "r2_wave_engine_ab" "$point_binary"
while IFS= read -r native_archive; do
  print_artifact_identity "aws-lc native archive" "$native_archive"
done < <(find "$CARGO_TARGET_DIR/release/build" -type f -path '*/aws-lc-sys-*/out/libaws_lc_*_crypto.a' -print | sort)

section_failures=()

print_execution_status() {
  local status="$1"
  local detail="$2"
  echo ""
  echo "########################################################################################"
  echo "# report_card_execution_status=${status} mode=${mode} ${detail}"
  echo "# Performance acceptance still requires comparison to the applicable accepted baseline."
  echo "########################################################################################"
}

raw_completion_marker="gpu_db_benchmark_status=complete benchmark=read_kernel_roofline cache_regimes=in_l2,out_of_l2"
point_b_completion_marker="gpu_db_benchmark_status=complete benchmark=r2_wave_engine_ab rows=${FULL_POINT_ROWS_IN_L2}"
raw_command=("$raw_binary")
point_b_command=(
  env GPU_DB_BENCH_ROWS="$FULL_POINT_ROWS_IN_L2" "$point_binary"
)
if [[ "$mode" == "full" ]]; then
  raw_completion_marker+=" rows=${FULL_RAW_ROWS} rows_large=${FULL_RAW_ROWS_LARGE} sort_n=${FULL_RAW_SORT_N} iters=${FULL_RAW_ITERS} iters_large=${FULL_RAW_ITERS_LARGE}"
  point_b_completion_marker+=" measured_batches=${FULL_POINT_BATCHES_IN_L2} warmup=${FULL_POINT_WARMUP} batch_sizes=${FULL_POINT_BATCH_SIZES} threads=${FULL_POINT_THREADS_IN_L2} insert_chunk=${FULL_POINT_INSERT_CHUNK}"
  raw_command=(
    env
    ROWS="$FULL_RAW_ROWS"
    ROWS_LARGE="$FULL_RAW_ROWS_LARGE"
    SORT_N="$FULL_RAW_SORT_N"
    ITERS="$FULL_RAW_ITERS"
    ITERS_LARGE="$FULL_RAW_ITERS_LARGE"
    "$raw_binary"
  )
  point_b_command=(
    env
    GPU_DB_BENCH_ROWS="$FULL_POINT_ROWS_IN_L2"
    GPU_DB_BENCH_BATCH="$FULL_POINT_BATCH_SIZES"
    GPU_DB_BENCH_BATCHES="$FULL_POINT_BATCHES_IN_L2"
    GPU_DB_BENCH_WARMUP="$FULL_POINT_WARMUP"
    GPU_DB_BENCH_THREADS="$FULL_POINT_THREADS_IN_L2"
    GPU_DB_BENCH_INSERT_CHUNK="$FULL_POINT_INSERT_CHUNK"
    "$point_binary"
  )
fi

# ---- Section A: RAW READ KERNELS (emits IN-L2 + OUT-OF-L2 in one invocation) ----
run_section A "RAW READ KERNELS (read_kernel_roofline -- IN-L2 + OUT-OF-L2)" "${SECTION_A_TIMEOUT}" \
  "$raw_completion_marker" "${raw_command[@]}"

echo ""
echo "### GPU cool-down: sleep ${GPU_GAP}"
sleep "${GPU_GAP}"

# ---- Section B: lpb/wave ENGINE POINT READS, IN-L2 (default 1M rows; full concurrent section) ----
run_section B "lpb/wave ENGINE POINT READS -- IN-L2 (1M rows = 4MB/col, cache-resident)" "${SECTION_B_TIMEOUT}" \
  "$point_b_completion_marker" "${point_b_command[@]}"

if [[ "${#section_failures[@]}" -eq 0 ]]; then
  if ! enforce_point_production_throughput_floor \
    "$bench_target_dir/report-card-section-B.log" \
    "$FULL_POINT_IN_L2_FLOOR_BATCH" \
    "$FULL_POINT_IN_L2_MIN_LOOKUPS_PER_S" |
    tee -a "$bench_target_dir/report-card-section-B.log"; then
    section_failures+=("B:in-l2-throughput-floor")
  fi
fi

if [[ "${#section_failures[@]}" -ne 0 ]]; then
  print_execution_status "incomplete" "failed_sections=${section_failures[*]}"
  exit 1
fi

if [[ "$mode" == "quick" ]]; then
  print_execution_status "screen-complete" "sections=A,B canonical=false"
  exit 0
fi

if ! candidate_remained_frozen; then
  print_execution_status "invalid" "reason=candidate-drift-before-section-c"
  exit 1
fi

if ! should_run_section_c "$mode" "${#section_failures[@]}"; then
  print_execution_status "incomplete" "reason=section-c-precondition"
  exit 1
fi

echo ""
echo "### GPU cool-down: sleep ${GPU_GAP}"
sleep "${GPU_GAP}"

# ---- Section C: lpb/wave ENGINE POINT READS, OUT-OF-L2 (large table; skip concurrent; fewer batches) ----
point_c_completion_marker="gpu_db_benchmark_status=complete benchmark=r2_wave_engine_ab rows=${OUT_OF_L2_ROWS} measured_batches=${OUT_OF_L2_BATCHES} warmup=${FULL_POINT_WARMUP} batch_sizes=${FULL_POINT_BATCH_SIZES} threads=none insert_chunk=${FULL_POINT_INSERT_CHUNK}"
run_section C "lpb/wave ENGINE POINT READS -- OUT-OF-L2 (${OUT_OF_L2_ROWS} rows = ${out_of_l2_col_mb}MB/col, HBM-bound)" "${SECTION_C_TIMEOUT}" \
  "$point_c_completion_marker" \
  env \
  GPU_DB_BENCH_ROWS="${OUT_OF_L2_ROWS}" \
  GPU_DB_BENCH_BATCH="$FULL_POINT_BATCH_SIZES" \
  GPU_DB_BENCH_BATCHES="${OUT_OF_L2_BATCHES}" \
  GPU_DB_BENCH_WARMUP="$FULL_POINT_WARMUP" \
  GPU_DB_BENCH_THREADS="" \
  GPU_DB_BENCH_INSERT_CHUNK="$FULL_POINT_INSERT_CHUNK" \
  "$point_binary"

if [[ "${#section_failures[@]}" -ne 0 ]]; then
  print_execution_status "incomplete" "failed_sections=${section_failures[*]}"
  exit 1
fi

if ! candidate_remained_frozen; then
  print_execution_status "invalid" "reason=candidate-drift"
  exit 1
fi

print_execution_status "complete" "sections=A,B,C canonical=true"
