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
#   BENCH_REPORT_CARD_LOG
#                      durable full-card stdout/stderr transcript (default: a fresh file under
#                      target/benchmark-report-card-runs; it is deliberately outside the disposable build target)
#
# Acceptance floor:
#   Section B batch-65,536 production-compact whole-run wall throughput must be
#   at least 260,000,000 lookups/s at the median of exactly three independently
#   launched samples (the fixed three-sample median rule). Every sample must have one
#   valid metric; missing, malformed, or duplicate evidence makes execution invalid
#   rather than selecting a passing launch. There are no retries.

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
readonly FULL_POINT_IN_L2_SAMPLE_COUNT="3"
readonly FULL_POINT_IN_L2_REQUIRED_QUALIFYING_SAMPLES="2"
readonly FULL_GPU_GAP="12"

# `run_section` is reused by --self-check, which deliberately does not require a GPU.
gpu_environment_checks_enabled=0

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
  local point_gate_status="$3"
  [[ "$requested_mode" == "full" && "$prior_failure_count" -eq 0 &&
    "$point_gate_status" == "pass" ]]
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

gpu_static_identity() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi -i 0 --query-gpu=uuid,name,driver_version --format=csv,noheader 2>/dev/null |
    awk -F ',' '
      NF != 3 { invalid = 1; next }
      {
        for (field = 1; field <= NF; field += 1) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
          if ($field == "") invalid = 1
        }
        print $1 "," $2 "," $3
        rows += 1
      }
      END { if (invalid || rows == 0) exit 1 }
    '
}

gpu_runtime_telemetry() {
  nvidia-smi -i 0 \
    --query-gpu=temperature.gpu,pstate,clocks.current.graphics,clocks.current.memory,utilization.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null |
    sed -E 's/[[:space:]]*,[[:space:]]*/,/g; s/^[[:space:]]+//; s/[[:space:]]+$//' |
    sed '/^$/d' |
    paste -sd ';' -
}

gpu_compute_process_inventory() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F ',' '
      /No running compute processes found/ { next }
      NF != 4 { invalid = 1; next }
      {
        for (field = 1; field <= NF; field += 1) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        }
        if ($1 == "" || $2 !~ /^[0-9]+$/ || $3 == "" || $4 == "") {
          invalid = 1
          next
        }
        print $1 "," $2 "," $3 "," $4
        rows += 1
      }
      END { if (invalid) exit 1 }
    ' |
    LC_ALL=C sort -t ',' -k1,1 -k2,2n -k3,3 |
    paste -sd ';' -
}

gpu_context_identity_from_inventory() {
  local inventory="$1"
  if [[ -z "$inventory" ]]; then
    printf '%s\n' "none"
    return 0
  fi
  printf '%s\n' "$inventory" |
    tr ';' '\n' |
    awk -F ',' 'NF == 4 { print $1 "," $2 "," $3 }' |
    LC_ALL=C sort -t ',' -k1,1 -k2,2n -k3,3 |
    paste -sd ';' -
}

benchmark_configuration_fingerprint() {
  printf '%s\0' \
    "$mode" \
    "$FULL_RAW_ROWS" "$FULL_RAW_ROWS_LARGE" "$FULL_RAW_SORT_N" \
    "$FULL_RAW_ITERS" "$FULL_RAW_ITERS_LARGE" \
    "$FULL_POINT_ROWS_IN_L2" "$FULL_POINT_ROWS_OUT_OF_L2" \
    "$FULL_POINT_BATCH_SIZES" "$FULL_POINT_BATCHES_IN_L2" \
    "$FULL_POINT_BATCHES_OUT_OF_L2" "$FULL_POINT_WARMUP" \
    "$FULL_POINT_THREADS_IN_L2" "$FULL_POINT_INSERT_CHUNK" \
    "$FULL_POINT_IN_L2_FLOOR_BATCH" "$FULL_POINT_IN_L2_MIN_LOOKUPS_PER_S" \
    "$FULL_POINT_IN_L2_SAMPLE_COUNT" "$FULL_POINT_IN_L2_REQUIRED_QUALIFYING_SAMPLES" \
    "$GPU_GAP" |
    sha256sum | awk '{print $1}'
}

artifact_sha256() {
  local artifact="$1"
  sha256sum "$artifact" 2>/dev/null | awk '{print $1}'
}

artifact_matches_digest() {
  local artifact="$1"
  local expected_digest="$2"
  local observed_digest
  if ! observed_digest="$(artifact_sha256 "$artifact")"; then
    return 1
  fi
  [[ "$observed_digest" == "$expected_digest" ]]
}

environment_probe_failure_rc_for_reason() {
  case "$1" in
    gpu_identity_unavailable | gpu_identity_changed | gpu_telemetry_unavailable | \
      gpu_process_inventory_unavailable | gpu_process_inventory_malformed | \
      gpu_context_inventory_changed)
      return 1
      ;;
    report_card_gpu_lock_lost | candidate_drift | benchmark_configuration_changed | \
      raw_binary_artifact_changed | point_binary_artifact_changed)
      return 2
      ;;
    *)
      # An unclassified probe failure is structural evidence corruption, never
      # authorization to retry an unchanged candidate as an environment issue.
      return 2
      ;;
  esac
}

environment_probe_failure_class_for_rc() {
  case "$1" in
    1) printf '%s\n' "environment_invalid" ;;
    2) printf '%s\n' "execution_invalid" ;;
    *) printf '%s\n' "execution_invalid" ;;
  esac
}

emit_environment_probe_failure() {
  local section_id="$1"
  local reason="$2"
  local probe_rc
  local failure_class
  shift 2
  environment_probe_failure_rc_for_reason "$reason"
  probe_rc=$?
  failure_class="$(environment_probe_failure_class_for_rc "$probe_rc")"
  echo "gpu_environment_sample_status=invalid section=${section_id} failure_class=${failure_class} reason=${reason}${*:+ $*}"
  return "$probe_rc"
}

record_environment_probe_failure() {
  local section_id="$1"
  local probe_rc="$2"
  local failure_class
  failure_class="$(environment_probe_failure_class_for_rc "$probe_rc")"
  case "$failure_class" in
    environment_invalid) section_failures+=("${section_id}:environment") ;;
    *) section_failures+=("${section_id}:execution") ;;
  esac
}

report_card_failure_class() {
  local gate_outcome="$1"
  shift
  local failure
  local saw_environment_invalid=0
  local saw_execution_invalid=0
  local failure_class="execution_invalid"
  case "$gate_outcome" in
    candidate_performance_failure) failure_class="candidate_performance_failure" ;;
    environment_invalid) failure_class="environment_invalid" ;;
    execution_invalid) failure_class="execution_invalid" ;;
  esac
  for failure in "$@"; do
    case "$failure" in
      *:execution) saw_execution_invalid=1 ;;
      *:environment) saw_environment_invalid=1 ;;
    esac
  done
  if [[ "$saw_execution_invalid" -eq 1 ]]; then
    failure_class="execution_invalid"
  elif [[ "$saw_environment_invalid" -eq 1 ]]; then
    failure_class="environment_invalid"
  fi
  printf '%s\n' "$failure_class"
}

benchmark_environment_before_section() {
  local section_id="$1"
  local current_identity
  local current_identity_sha256
  local current_configuration_sha256
  local current_raw_binary_sha256
  local current_point_binary_sha256
  local telemetry
  local process_inventory
  local context_identity
  local context_identity_sha256
  local context_count
  local process_inventory_sha256
  if ! flock -n 9; then
    emit_environment_probe_failure "$section_id" "report_card_gpu_lock_lost"
    return $?
  fi
  if [[ "$mode" == "full" ]] && ! candidate_remained_frozen; then
    emit_environment_probe_failure "$section_id" "candidate_drift"
    return $?
  fi
  if ! current_identity="$(gpu_static_identity)"; then
    emit_environment_probe_failure "$section_id" "gpu_identity_unavailable"
    return $?
  fi
  current_identity_sha256="$(printf '%s' "$current_identity" | sha256sum | awk '{print $1}')"
  if ! telemetry="$(gpu_runtime_telemetry)" || [[ -z "$telemetry" ]]; then
    emit_environment_probe_failure "$section_id" "gpu_telemetry_unavailable"
    return $?
  fi
  if ! process_inventory="$(gpu_compute_process_inventory)"; then
    emit_environment_probe_failure "$section_id" "gpu_process_inventory_unavailable"
    return $?
  fi
  context_identity="$(gpu_context_identity_from_inventory "$process_inventory")" || {
    emit_environment_probe_failure "$section_id" "gpu_process_inventory_malformed"
    return $?
  }
  context_identity_sha256="$(printf '%s' "$context_identity" | sha256sum | awk '{print $1}')"
  if [[ -n "$process_inventory" ]]; then
    process_inventory_sha256="$(printf '%s' "$process_inventory" | sha256sum | awk '{print $1}')"
  else
    process_inventory_sha256="none"
    process_inventory="none"
  fi
  context_count="$(printf '%s' "$context_identity" | awk -F ';' '$0 == "none" { print 0; next } { print NF }')"
  echo "# gpu_environment_identity section=${section_id} uuid_name_driver=${current_identity}"
  echo "# gpu_environment_process_inventory section=${section_id} entries=${process_inventory}"
  echo "# gpu_environment_telemetry section=${section_id} temperature_c_pstate_graphics_mhz_memory_mhz_utilization_pct_power_w=${telemetry}"
  current_configuration_sha256="$(benchmark_configuration_fingerprint)"
  if [[ "$current_identity_sha256" != "$benchmark_gpu_identity_sha256" ]]; then
    emit_environment_probe_failure "$section_id" "gpu_identity_changed" \
      "expected_gpu_identity_sha256=${benchmark_gpu_identity_sha256}" \
      "observed_gpu_identity_sha256=${current_identity_sha256}" \
      "context_inventory_sha256=${process_inventory_sha256}"
    return $?
  fi
  if [[ "$current_configuration_sha256" != "$benchmark_configuration_sha256" ]]; then
    emit_environment_probe_failure "$section_id" "benchmark_configuration_changed" \
      "expected_configuration_sha256=${benchmark_configuration_sha256}" \
      "observed_configuration_sha256=${current_configuration_sha256}" \
      "context_inventory_sha256=${process_inventory_sha256}"
    return $?
  fi
  if ! current_raw_binary_sha256="$(artifact_sha256 "$raw_binary")" ||
    [[ "$current_raw_binary_sha256" != "$raw_binary_sha256" ]]; then
    emit_environment_probe_failure "$section_id" "raw_binary_artifact_changed" \
      "expected_raw_binary_sha256=${raw_binary_sha256}" \
      "observed_raw_binary_sha256=${current_raw_binary_sha256:-unavailable}" \
      "context_inventory_sha256=${process_inventory_sha256}"
    return $?
  fi
  if ! current_point_binary_sha256="$(artifact_sha256 "$point_binary")" ||
    [[ "$current_point_binary_sha256" != "$point_binary_sha256" ]]; then
    emit_environment_probe_failure "$section_id" "point_binary_artifact_changed" \
      "expected_point_binary_sha256=${point_binary_sha256}" \
      "observed_point_binary_sha256=${current_point_binary_sha256:-unavailable}" \
      "context_inventory_sha256=${process_inventory_sha256}"
    return $?
  fi
  if [[ "$section_id" == B* ]]; then
    if [[ -z "${benchmark_b_context_identity_sha256:-}" ]]; then
      benchmark_b_context_identity_sha256="$context_identity_sha256"
    elif [[ "$context_identity_sha256" != "$benchmark_b_context_identity_sha256" ]]; then
      emit_environment_probe_failure "$section_id" "gpu_context_inventory_changed" \
        "expected_context_identity_sha256=${benchmark_b_context_identity_sha256}" \
        "observed_context_identity_sha256=${context_identity_sha256}" \
        "context_inventory_sha256=${process_inventory_sha256}"
      return $?
    fi
  fi
  echo "gpu_environment_sample_status=valid section=${section_id} gpu_identity_sha256=${current_identity_sha256} configuration_sha256=${current_configuration_sha256} raw_binary_sha256=${current_raw_binary_sha256} point_binary_sha256=${current_point_binary_sha256} external_compute_context_count=${context_count} context_identity_sha256=${context_identity_sha256} context_inventory_sha256=${process_inventory_sha256}"
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
  # Exactly three complete fixed-workload Section-B samples are required.  The
  # qualification is their median at the existing 260M floor, not a
  # retry loop that can stop after landing in a high mode.
  local target_batch="$1"
  local minimum_lookups_per_s="$2"
  local expected_sample_count="$3"
  local required_qualifying_samples="$4"
  shift 4
  local observed_sample_count="$#"
  local sample_index=0
  local measured_lookups_per_s
  local sample_status
  local qualified_samples=0
  local invalid_samples=0
  local valid_samples=0
  local below_floor_samples=0
  local measurements
  local minimum_sample
  local median_sample
  local maximum_sample
  local median_index
  local -a measured_samples=()
  local -a reported_samples=()
  local -a sorted_samples=()
  local -A seen_sample_logs=()
  local section_log
  point_gate_outcome="execution_invalid"
  if ! [[
    "$minimum_lookups_per_s" =~ ^(0|[1-9][0-9]*)$ &&
      "${#minimum_lookups_per_s}" -le 10
  ]]; then
    echo "point_read_environment_status=not_evaluated cache_regime=in_l2 batch=${target_batch} reason=execution_invalid"
    echo "point_read_performance_status=not_evaluated cache_regime=in_l2 batch=${target_batch} reason=execution_invalid"
    echo "point_read_throughput_gate_status=execution_invalid cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s minimum=${minimum_lookups_per_s} environment_status=not_evaluated performance_status=not_evaluated reason=invalid_minimum"
    return 2
  fi
  if ! [[
    "$expected_sample_count" =~ ^[1-9][0-9]*$ &&
      "$required_qualifying_samples" =~ ^[1-9][0-9]*$ &&
      "$expected_sample_count" == "$FULL_POINT_IN_L2_SAMPLE_COUNT" &&
      "$required_qualifying_samples" == "$FULL_POINT_IN_L2_REQUIRED_QUALIFYING_SAMPLES"
  ]]; then
    echo "point_read_environment_status=not_evaluated cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} required_qualifying_samples=${required_qualifying_samples} reason=execution_invalid"
    echo "point_read_performance_status=not_evaluated cache_regime=in_l2 batch=${target_batch} reason=execution_invalid"
    echo "point_read_throughput_gate_status=execution_invalid cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s expected_samples=${expected_sample_count} required_qualifying_samples=${required_qualifying_samples} environment_status=not_evaluated performance_status=not_evaluated reason=invalid_qualification"
    return 2
  fi
  if ((10#$required_qualifying_samples > 10#$expected_sample_count)) ||
    [[ "$observed_sample_count" != "$expected_sample_count" ]]; then
    echo "point_read_environment_status=not_evaluated cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} observed_samples=${observed_sample_count} required_qualifying_samples=${required_qualifying_samples} reason=execution_invalid"
    echo "point_read_performance_status=not_evaluated cache_regime=in_l2 batch=${target_batch} reason=execution_invalid"
    echo "point_read_throughput_gate_status=execution_invalid cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s expected_samples=${expected_sample_count} observed_samples=${observed_sample_count} required_qualifying_samples=${required_qualifying_samples} environment_status=not_evaluated performance_status=not_evaluated reason=invalid_sample_count"
    return 2
  fi
  for section_log in "$@"; do
    sample_index=$((sample_index + 1))
    if [[ -n "${seen_sample_logs[$section_log]+present}" ]]; then
      invalid_samples=$((invalid_samples + 1))
      reported_samples+=("B${sample_index}:invalid")
      echo "point_read_throughput_sample_status=invalid cache_regime=in_l2 batch=${target_batch} sample=B${sample_index}/${expected_sample_count} metric=whole_run_wall_lookups_per_s minimum=${minimum_lookups_per_s} reason=duplicate_sample_log"
      continue
    fi
    seen_sample_logs["$section_log"]=1
    if [[ ! -r "$section_log" ]] || ! measured_lookups_per_s="$(
      point_production_throughput_for_batch "$section_log" "$target_batch"
    )"; then
      invalid_samples=$((invalid_samples + 1))
      reported_samples+=("B${sample_index}:invalid")
      echo "point_read_throughput_sample_status=invalid cache_regime=in_l2 batch=${target_batch} sample=B${sample_index}/${expected_sample_count} metric=whole_run_wall_lookups_per_s minimum=${minimum_lookups_per_s} reason=missing_malformed_or_duplicate"
      continue
    fi
    valid_samples=$((valid_samples + 1))
    if ((10#$measured_lookups_per_s >= 10#$minimum_lookups_per_s)); then
      sample_status="qualified"
      qualified_samples=$((qualified_samples + 1))
    else
      sample_status="below_floor"
      below_floor_samples=$((below_floor_samples + 1))
    fi
    measured_samples+=("$measured_lookups_per_s")
    reported_samples+=("B${sample_index}:${measured_lookups_per_s}")
    echo "point_read_throughput_sample_status=${sample_status} cache_regime=in_l2 batch=${target_batch} sample=B${sample_index}/${expected_sample_count} metric=whole_run_wall_lookups_per_s measured=${measured_lookups_per_s} minimum=${minimum_lookups_per_s}"
  done
  measurements="$(IFS=,; echo "${reported_samples[*]}")"
  if [[ "$invalid_samples" -ne 0 ]]; then
    echo "point_read_environment_status=not_evaluated cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} valid_samples=${valid_samples} invalid_samples=${invalid_samples} reason=execution_invalid"
    echo "point_read_performance_status=not_evaluated cache_regime=in_l2 batch=${target_batch} reason=execution_invalid"
    echo "point_read_throughput_gate_status=execution_invalid cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s expected_samples=${expected_sample_count} valid_samples=${valid_samples} invalid_samples=${invalid_samples} minimum=${minimum_lookups_per_s} ordered_measurements=${measurements} environment_status=not_evaluated performance_status=not_evaluated reason=missing_malformed_duplicate_or_reused_sample"
    return 2
  fi
  mapfile -t sorted_samples < <(printf '%s\n' "${measured_samples[@]}" | sort -n)
  minimum_sample="${sorted_samples[0]}"
  median_index=$((10#$expected_sample_count / 2))
  median_sample="${sorted_samples[$median_index]}"
  maximum_sample="${sorted_samples[$((10#$expected_sample_count - 1))]}"
  if ((10#$median_sample >= 10#$minimum_lookups_per_s)); then
    echo "point_read_environment_status=valid cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} valid_samples=${valid_samples} reason=all_samples_valid"
    echo "point_read_performance_status=pass cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} minimum=${minimum_lookups_per_s} min=${minimum_sample} median=${median_sample} max=${maximum_sample} above_floor=${qualified_samples} below_floor=${below_floor_samples}"
    echo "point_read_throughput_gate_status=pass cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s expected_samples=${expected_sample_count} valid_samples=${valid_samples} required_qualifying_samples=${required_qualifying_samples} minimum=${minimum_lookups_per_s} min=${minimum_sample} median=${median_sample} max=${maximum_sample} above_floor=${qualified_samples} below_floor=${below_floor_samples} ordered_measurements=${measurements} rule=fixed_three_sample_median environment_status=valid performance_status=pass reason=median_at_or_above_floor"
    point_gate_outcome="pass"
    return 0
  fi
  if [[ "$qualified_samples" -ne 0 ]]; then
    echo "point_read_environment_status=invalid cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} valid_samples=${valid_samples} minimum=${minimum_lookups_per_s} min=${minimum_sample} median=${median_sample} max=${maximum_sample} above_floor=${qualified_samples} below_floor=${below_floor_samples} reason=indeterminate_bimodality"
    echo "point_read_performance_status=not_evaluated cache_regime=in_l2 batch=${target_batch} reason=environment_invalid"
    echo "point_read_throughput_gate_status=environment_invalid cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s expected_samples=${expected_sample_count} valid_samples=${valid_samples} required_qualifying_samples=${required_qualifying_samples} minimum=${minimum_lookups_per_s} min=${minimum_sample} median=${median_sample} max=${maximum_sample} above_floor=${qualified_samples} below_floor=${below_floor_samples} ordered_measurements=${measurements} rule=fixed_three_sample_median environment_status=invalid performance_status=not_evaluated reason=indeterminate_bimodality"
    point_gate_outcome="environment_invalid"
    return 2
  fi
  echo "point_read_environment_status=valid cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} valid_samples=${valid_samples} reason=all_samples_valid"
  echo "point_read_performance_status=fail cache_regime=in_l2 batch=${target_batch} expected_samples=${expected_sample_count} minimum=${minimum_lookups_per_s} min=${minimum_sample} median=${median_sample} max=${maximum_sample} above_floor=${qualified_samples} below_floor=${below_floor_samples} reason=all_samples_below_floor"
  echo "point_read_throughput_gate_status=performance_fail cache_regime=in_l2 batch=${target_batch} metric=whole_run_wall_lookups_per_s expected_samples=${expected_sample_count} valid_samples=${valid_samples} required_qualifying_samples=${required_qualifying_samples} minimum=${minimum_lookups_per_s} min=${minimum_sample} median=${median_sample} max=${maximum_sample} above_floor=${qualified_samples} below_floor=${below_floor_samples} ordered_measurements=${measurements} rule=fixed_three_sample_median environment_status=valid performance_status=fail reason=all_samples_below_floor"
  point_gate_outcome="candidate_performance_failure"
  return 1
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
      if ! rm -rf -- "$cleanup_target"; then
        echo "failed to remove isolated benchmark target: $cleanup_target" >&2
        return 1
      fi
      echo "# removed isolated benchmark target: $cleanup_target"
      ;;
    *)
      echo "refusing to remove unexpected benchmark target: $cleanup_target" >&2
      return 1
      ;;
  esac
}

# A canonical card may clean its fresh build target, but its result transcript is acceptance evidence and must
# survive that cleanup. Keep it separately and make a failed durable capture fail the runner rather than leaving a
# superficially successful terminal status with no inspectable provenance.
report_card_log=""
report_card_log_tee_pid=""

start_durable_report_card_log() {
  local requested_log="${BENCH_REPORT_CARD_LOG:-}"
  local log_dir
  local log_name

  [[ "$mode" == "full" ]] || return 0
  if [[ -n "$requested_log" ]]; then
    log_dir="$(dirname -- "$requested_log")"
    log_name="$(basename -- "$requested_log")"
    if [[ "$log_name" == "." || "$log_name" == "/" ]]; then
      echo "BENCH_REPORT_CARD_LOG must name a file" >&2
      return 1
    fi
    if ! mkdir -p -- "$log_dir"; then
      echo "failed to create BENCH_REPORT_CARD_LOG parent: $log_dir" >&2
      return 1
    fi
    log_dir="$(cd "$log_dir" && pwd -P)" || return 1
    report_card_log="$log_dir/$log_name"
    case "$report_card_log" in
      "$bench_target_dir"|"$bench_target_dir"/*)
        echo "BENCH_REPORT_CARD_LOG must be outside the disposable benchmark target" >&2
        return 1
        ;;
    esac
    if [[ -e "$report_card_log" ]]; then
      echo "BENCH_REPORT_CARD_LOG must not already exist: $report_card_log" >&2
      return 1
    fi
    if ! (set -C; : >"$report_card_log") 2>/dev/null; then
      echo "failed to create BENCH_REPORT_CARD_LOG: $report_card_log" >&2
      return 1
    fi
  else
    log_dir="$repo_root/target/benchmark-report-card-runs"
    if ! mkdir -p -- "$log_dir"; then
      echo "failed to create durable report-card log directory: $log_dir" >&2
      return 1
    fi
    report_card_log="$(mktemp "$log_dir/runner.${git_head:0:12}.${candidate_index_tree:0:12}.XXXXXX.log")" || {
      echo "failed to create durable report-card log" >&2
      return 1
    }
  fi

  exec 3>&1
  exec 4>&2
  # The host-facing console is best-effort. `-p` keeps its broken pipe from terminating the primary transcript
  # writer, while a failure writing the transcript itself still reaches `wait` below as a hard runner failure.
  exec > >(tee -p -a "$report_card_log" >&3) 2>&1
  report_card_log_tee_pid=$!
  echo "# durable report-card transcript: $report_card_log"
}

finish_durable_report_card_log() {
  local tee_pid="$report_card_log_tee_pid"
  [[ -n "$tee_pid" ]] || return 0
  exec 1>&3 2>&4
  if ! wait "$tee_pid"; then
    echo "durable report-card transcript capture failed: $report_card_log" >&2
    return 1
  fi
  exec 3>&-
  exec 4>&-
  report_card_log_tee_pid=""
  return 0
}

exit_after_durable_cleanup() {
  local original_rc="$1"
  local cleanup_rc="$2"
  if [[ "$original_rc" -ne 0 ]]; then
    exit "$original_rc"
  fi
  exit "$cleanup_rc"
}

cleanup_bench_target() {
  local original_rc=$?
  local cleanup_rc=0
  cleanup_owned_bench_target \
    "$repo_root" "$bench_target_dir" "$bench_target_owned" "${BENCH_KEEP_TARGET:-0}" || cleanup_rc=$?
  if ! finish_durable_report_card_log; then
    cleanup_rc=1
  fi
  exit_after_durable_cleanup "$original_rc" "$cleanup_rc"
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
  local environment_rc
  local -a pipeline_status
  if ! : >"$section_log"; then
    echo "[section ${id} INCOMPLETE: output capture setup failed]"
    section_failures+=("${id}:output")
    return 0
  fi
  if [[ "${gpu_environment_checks_enabled:-0}" == "1" ]]; then
    benchmark_environment_before_section "$id" >>"$section_log"
    environment_rc=$?
    cat "$section_log"
    if [[ "$environment_rc" -ne 0 ]]; then
      echo "[section ${id} INCOMPLETE: environment/provenance qualification failed]"
      record_environment_probe_failure "$id" "$environment_rc"
      return 0
    fi
  fi
  echo ""
  echo "### SECTION ${id} -- ${title}"
  echo "### cmd: timeout ${tmo} $*"
  echo "### ----------------------------------------------------------------------------------"
  timeout "${tmo}" "$@" 2>&1 | tee -a "$section_log"
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
  local point_gate_output
  local point_gate_rc
  local sabotaged_log
  local gate_case
  local gate_case_inputs
  local first_log
  local second_log
  local third_log
  local gate_log
  local saved_gpu_gap
  local saved_gpu_gap_set=0
  local self_context_identity
  local self_context_identity_with_memory_change
  local self_context_identity_drifted
  local self_context_sha256
  local self_context_sha256_with_memory_change
  local self_context_sha256_drifted
  local self_config_sha256
  local self_config_drifted_sha256
  local self_artifact
  local self_artifact_sha256
  local probe_reason
  local probe_rc
  local probe_output
  local terminal_failure_class
  local mode="$mode"
  local repo_root="${repo_root:-}"
  local git_head="${git_head:-}"
  local candidate_index_tree="${candidate_index_tree:-}"
  local bench_target_owned="${bench_target_owned:-0}"
  local BENCH_REPORT_CARD_LOG="${BENCH_REPORT_CARD_LOG:-}"
  local report_card_log="$report_card_log"
  local report_card_log_tee_pid="$report_card_log_tee_pid"
  local self_durable_log
  local self_console_fd
  local self_cleanup_rc
  local self_exit_rc
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
  should_run_section_c full 0 pass || failures=$((failures + 1))
  ! should_run_section_c full 1 pass || failures=$((failures + 1))
  ! should_run_section_c full 0 environment_invalid || failures=$((failures + 1))
  ! should_run_section_c full 0 execution_invalid || failures=$((failures + 1))
  ! should_run_section_c quick 0 pass || failures=$((failures + 1))

  for probe_reason in \
    report_card_gpu_lock_lost candidate_drift benchmark_configuration_changed \
    raw_binary_artifact_changed point_binary_artifact_changed unclassified_failure; do
    environment_probe_failure_rc_for_reason "$probe_reason"
    probe_rc=$?
    section_failures=()
    record_environment_probe_failure B1 "$probe_rc"
    terminal_failure_class="$(
      report_card_failure_class not_evaluated "${section_failures[@]}"
    )"
    [[ "$probe_rc" -eq 2 && "${section_failures[*]}" == "B1:execution" &&
      "$terminal_failure_class" == "execution_invalid" ]] ||
      failures=$((failures + 1))
  done
  for probe_reason in \
    gpu_identity_unavailable gpu_identity_changed gpu_telemetry_unavailable \
    gpu_process_inventory_unavailable gpu_process_inventory_malformed \
    gpu_context_inventory_changed; do
    environment_probe_failure_rc_for_reason "$probe_reason"
    probe_rc=$?
    section_failures=()
    record_environment_probe_failure B1 "$probe_rc"
    terminal_failure_class="$(
      report_card_failure_class not_evaluated "${section_failures[@]}"
    )"
    [[ "$probe_rc" -eq 1 && "${section_failures[*]}" == "B1:environment" &&
      "$terminal_failure_class" == "environment_invalid" ]] ||
      failures=$((failures + 1))
  done
  section_failures=()
  emit_environment_probe_failure B1 candidate_drift >"$scratch/probe-failure.log"
  probe_rc=$?
  probe_output="$(<"$scratch/probe-failure.log")"
  record_environment_probe_failure B1 "$probe_rc"
  terminal_failure_class="$(
    report_card_failure_class environment_invalid \
      "B0:environment" "${section_failures[@]}"
  )"
  [[ "$probe_rc" -eq 2 &&
    "$probe_output" == *"failure_class=execution_invalid reason=candidate_drift"* &&
    "${section_failures[*]}" == "B1:execution" &&
    "$terminal_failure_class" == "execution_invalid" ]] ||
    failures=$((failures + 1))
  [[ "$(report_card_failure_class candidate_performance_failure)" == \
    "candidate_performance_failure" ]] ||
    failures=$((failures + 1))

  bench_target_dir="$scratch"
  section_failures=()
  run_section PASS "self-check marker pass" 5 "control-marker=complete" \
    bash -c 'printf "%s\n" "control-marker=complete"' >/dev/null 2>&1
  [[ "${#section_failures[@]}" -eq 0 ]] || failures=$((failures + 1))
  run_section MARKER "self-check missing marker" 5 "control-marker=complete" \
    bash -c 'printf "%s\n" "control-marker=incomplete"' >/dev/null 2>&1
  [[ "${section_failures[*]}" == "MARKER:marker" ]] || failures=$((failures + 1))
  ! should_run_section_c full "${#section_failures[@]}" execution_invalid || failures=$((failures + 1))
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
    >"$scratch/point-floor/equal.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 270000000 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/above.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 259999999 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/below.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 250000000 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/below-lower.log"
  printf '%s\n' \
    "### batch=65536" \
    "  prod-compact p50= 117us p99= 130us | 240000000 lookups/s (0.004 us/lookup)" \
    >"$scratch/point-floor/below-lowest.log"
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
  [[ "$(point_production_throughput_for_batch "$scratch/point-floor/equal.log" 65536)" == "260000000" ]] ||
    failures=$((failures + 1))
  ! point_production_throughput_for_batch "$scratch/point-floor/missing.log" 65536 >/dev/null ||
    failures=$((failures + 1))
  ! point_production_throughput_for_batch "$scratch/point-floor/duplicate.log" 65536 >/dev/null ||
    failures=$((failures + 1))
  ! point_production_throughput_for_batch "$scratch/point-floor/post-summary-decoy.log" 65536 >/dev/null ||
    failures=$((failures + 1))
  ! point_production_throughput_for_batch "$scratch/point-floor/two-tokens.log" 65536 >/dev/null ||
    failures=$((failures + 1))
  ! point_production_throughput_for_batch "$scratch/point-floor/malformed-and-valid.log" 65536 >/dev/null ||
    failures=$((failures + 1))
  ! point_production_throughput_for_batch "$scratch/point-floor/zero-prefixed.log" 65536 >/dev/null ||
    failures=$((failures + 1))
  ! point_production_throughput_for_batch "$scratch/point-floor/duplicate-header.log" 65536 >/dev/null ||
    failures=$((failures + 1))

  gate_case=0
  for gate_case_inputs in \
    "above.log equal.log below.log" \
    "equal.log below.log above.log" \
    "below.log above.log equal.log"; do
    gate_case=$((gate_case + 1))
    read -r first_log second_log third_log <<<"$gate_case_inputs"
    gate_log="$scratch/point-floor/two-high-${gate_case}.gate"
    enforce_point_production_throughput_floor \
      65536 260000000 3 2 \
      "$scratch/point-floor/${first_log}" \
      "$scratch/point-floor/${second_log}" \
      "$scratch/point-floor/${third_log}" >"$gate_log"
    point_gate_rc=$?
    point_gate_output="$(<"$gate_log")"
    [[ "$point_gate_rc" -eq 0 && "$point_gate_outcome" == "pass" &&
      "$point_gate_output" == *"point_read_throughput_gate_status=pass"* &&
      "$point_gate_output" == *"min=259999999 median=260000000 max=270000000 above_floor=2 below_floor=1"* &&
      "$point_gate_output" == *"ordered_measurements=B1:"*"B2:"*"B3:"* &&
      "$point_gate_output" == *"environment_status=valid performance_status=pass reason=median_at_or_above_floor"* ]] ||
      failures=$((failures + 1))
  done

  gate_case=0
  for gate_case_inputs in \
    "above.log below.log below-lower.log" \
    "below.log above.log below-lower.log" \
    "below.log below-lower.log above.log"; do
    gate_case=$((gate_case + 1))
    read -r first_log second_log third_log <<<"$gate_case_inputs"
    gate_log="$scratch/point-floor/one-high-${gate_case}.gate"
    enforce_point_production_throughput_floor \
      65536 260000000 3 2 \
      "$scratch/point-floor/${first_log}" \
      "$scratch/point-floor/${second_log}" \
      "$scratch/point-floor/${third_log}" >"$gate_log"
    point_gate_rc=$?
    point_gate_output="$(<"$gate_log")"
    [[ "$point_gate_rc" -eq 2 && "$point_gate_outcome" == "environment_invalid" &&
      "$point_gate_output" == *"point_read_environment_status=invalid"* &&
      "$point_gate_output" == *"point_read_performance_status=not_evaluated"* &&
      "$point_gate_output" == *"point_read_throughput_gate_status=environment_invalid"* &&
      "$point_gate_output" == *"above_floor=1 below_floor=2"* &&
      "$point_gate_output" == *"environment_status=invalid performance_status=not_evaluated reason=indeterminate_bimodality"* ]] ||
      failures=$((failures + 1))
  done

  gate_log="$scratch/point-floor/all-below.gate"
  enforce_point_production_throughput_floor \
    65536 260000000 3 2 \
    "$scratch/point-floor/below.log" \
    "$scratch/point-floor/below-lower.log" \
    "$scratch/point-floor/below-lowest.log" >"$gate_log"
  point_gate_rc=$?
  point_gate_output="$(<"$gate_log")"
  [[ "$point_gate_rc" -eq 1 && "$point_gate_outcome" == "candidate_performance_failure" &&
    "$point_gate_output" == *"point_read_environment_status=valid"* &&
    "$point_gate_output" == *"point_read_performance_status=fail"* &&
    "$point_gate_output" == *"point_read_throughput_gate_status=performance_fail"* &&
    "$point_gate_output" == *"above_floor=0 below_floor=3"* &&
    "$point_gate_output" == *"environment_status=valid performance_status=fail reason=all_samples_below_floor"* ]] ||
    failures=$((failures + 1))

  for sabotaged_log in missing.log malformed-and-valid.log duplicate.log duplicate-header.log two-tokens.log post-summary-decoy.log zero-prefixed.log; do
    gate_log="$scratch/point-floor/${sabotaged_log}.gate"
    enforce_point_production_throughput_floor \
      65536 260000000 3 2 \
      "$scratch/point-floor/equal.log" \
      "$scratch/point-floor/${sabotaged_log}" \
      "$scratch/point-floor/above.log" >"$gate_log"
    point_gate_rc=$?
    point_gate_output="$(<"$gate_log")"
    [[ "$point_gate_rc" -eq 2 && "$point_gate_outcome" == "execution_invalid" &&
      "$point_gate_output" == *"point_read_throughput_gate_status=execution_invalid"* &&
      "$point_gate_output" == *"valid_samples=2 invalid_samples=1"* ]] ||
      failures=$((failures + 1))
  done

  gate_log="$scratch/point-floor/duplicate-sample-log.gate"
  enforce_point_production_throughput_floor \
    65536 260000000 3 2 \
    "$scratch/point-floor/equal.log" \
    "$scratch/point-floor/equal.log" \
    "$scratch/point-floor/above.log" >"$gate_log"
  point_gate_rc=$?
  point_gate_output="$(<"$gate_log")"
  [[ "$point_gate_rc" -eq 2 && "$point_gate_outcome" == "execution_invalid" &&
    "$point_gate_output" == *"sample=B2/3"*"reason=duplicate_sample_log"* &&
    "$point_gate_output" == *"ordered_measurements=B1:260000000,B2:invalid,B3:270000000"* ]] ||
    failures=$((failures + 1))

  gate_log="$scratch/point-floor/invalid-count.gate"
  enforce_point_production_throughput_floor \
    65536 260000000 3 2 \
    "$scratch/point-floor/equal.log" \
    "$scratch/point-floor/above.log" >"$gate_log"
  point_gate_rc=$?
  point_gate_output="$(<"$gate_log")"
  [[ "$point_gate_rc" -eq 2 && "$point_gate_outcome" == "execution_invalid" &&
    "$point_gate_output" == *"reason=invalid_sample_count"* ]] ||
    failures=$((failures + 1))

  gate_log="$scratch/point-floor/extra-count.gate"
  enforce_point_production_throughput_floor \
    65536 260000000 3 2 \
    "$scratch/point-floor/equal.log" \
    "$scratch/point-floor/above.log" \
    "$scratch/point-floor/below.log" \
    "$scratch/point-floor/below-lower.log" >"$gate_log"
  point_gate_rc=$?
  point_gate_output="$(<"$gate_log")"
  [[ "$point_gate_rc" -eq 2 && "$point_gate_outcome" == "execution_invalid" &&
    "$point_gate_output" == *"expected_samples=3 observed_samples=4"*"reason=invalid_sample_count"* ]] ||
    failures=$((failures + 1))

  gate_log="$scratch/point-floor/invalid-qualification.gate"
  enforce_point_production_throughput_floor \
    65536 260000000 3 1 \
    "$scratch/point-floor/equal.log" \
    "$scratch/point-floor/above.log" \
    "$scratch/point-floor/below.log" >"$gate_log"
  point_gate_rc=$?
  point_gate_output="$(<"$gate_log")"
  [[ "$point_gate_rc" -eq 2 && "$point_gate_outcome" == "execution_invalid" &&
    "$point_gate_output" == *"reason=invalid_qualification"* ]] ||
    failures=$((failures + 1))

  # Context identity deliberately excludes memory so stable shared-workstation
  # processes remain valid evidence even when their usage changes. A PID/name/UUID
  # change is the B-cohort environment invalidation boundary.
  self_context_identity="$(gpu_context_identity_from_inventory \
    "GPU-self,20,/usr/bin/node,1024;GPU-self,10,VLLM::EngineCore,43894")"
  self_context_identity_with_memory_change="$(gpu_context_identity_from_inventory \
    "GPU-self,10,VLLM::EngineCore,40000;GPU-self,20,/usr/bin/node,2048")"
  self_context_identity_drifted="$(gpu_context_identity_from_inventory \
    "GPU-self,10,VLLM::EngineCore,40000;GPU-self,21,/usr/bin/node,2048")"
  self_context_sha256="$(printf '%s' "$self_context_identity" | sha256sum | awk '{print $1}')"
  self_context_sha256_with_memory_change="$(printf '%s' "$self_context_identity_with_memory_change" | sha256sum | awk '{print $1}')"
  self_context_sha256_drifted="$(printf '%s' "$self_context_identity_drifted" | sha256sum | awk '{print $1}')"
  [[ "$self_context_identity" != "none" &&
    "$self_context_sha256" == "$self_context_sha256_with_memory_change" &&
    "$self_context_sha256" != "$self_context_sha256_drifted" ]] ||
    failures=$((failures + 1))

  if [[ -v GPU_GAP ]]; then
    saved_gpu_gap_set=1
    saved_gpu_gap="$GPU_GAP"
  fi
  GPU_GAP=12
  self_config_sha256="$(benchmark_configuration_fingerprint)"
  GPU_GAP=13
  self_config_drifted_sha256="$(benchmark_configuration_fingerprint)"
  [[ "$self_config_sha256" != "$self_config_drifted_sha256" ]] || failures=$((failures + 1))
  if [[ "$saved_gpu_gap_set" -eq 1 ]]; then
    GPU_GAP="$saved_gpu_gap"
  else
    unset GPU_GAP
  fi

  self_artifact="$scratch/point-floor/artifact"
  printf '%s\n' "artifact-stable" >"$self_artifact"
  self_artifact_sha256="$(artifact_sha256 "$self_artifact")"
  artifact_matches_digest "$self_artifact" "$self_artifact_sha256" ||
    failures=$((failures + 1))
  printf '%s\n' "artifact-drift" >>"$self_artifact"
  ! artifact_matches_digest "$self_artifact" "$self_artifact_sha256" ||
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

  mkdir -p -- "$scratch/cleanup-root/target/benchmark-report-card.ABC124"
  (
    rm() { return 1; }
    cleanup_owned_bench_target "$scratch/cleanup-root" \
      "$scratch/cleanup-root/target/benchmark-report-card.ABC124" 1 0 >/dev/null
  ) >/dev/null 2>&1
  self_cleanup_rc=$?
  [[ "$self_cleanup_rc" -ne 0 ]] || failures=$((failures + 1))

  (
    cleanup_owned_bench_target() { return 0; }
    finish_durable_report_card_log() { return 1; }
    repo_root="$scratch/cleanup-root"
    bench_target_dir="$scratch/cleanup-root/target/benchmark-report-card.ABC125"
    bench_target_owned=1
    trap cleanup_bench_target EXIT
    exit 0
  ) >/dev/null 2>&1
  self_exit_rc=$?
  [[ "$self_exit_rc" -eq 1 ]] || failures=$((failures + 1))

  # The terminal record must outlive the disposable build target. Exercise the same descriptor handoff used by a
  # canonical run; merely creating a side file would not prove that the runner's stdout/stderr reaches it.
  mode="full"
  repo_root="$scratch/cleanup-root"
  git_head="self-check-head"
  candidate_index_tree="self-check-tree"
  bench_target_dir="$scratch/cleanup-root/target/benchmark-report-card.DEF456"
  bench_target_owned=1
  mkdir -p -- "$bench_target_dir"
  BENCH_REPORT_CARD_LOG="$scratch/durable/runner.log"
  exec {self_console_fd}>&1
  exec > >(head -n 0)
  if ! start_durable_report_card_log; then
    failures=$((failures + 1))
  else
    echo "report-card-self-check durable-transcript-marker"
    self_durable_log="$report_card_log"
    if ! finish_durable_report_card_log; then
      failures=$((failures + 1))
    elif [[ "$self_durable_log" == "$bench_target_dir"/* || ! -f "$self_durable_log" ]] ||
      ! grep -Fq "report-card-self-check durable-transcript-marker" "$self_durable_log"; then
      failures=$((failures + 1))
    fi
  fi
  exec 1>&"$self_console_fd"
  exec {self_console_fd}>&-

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

trap cleanup_bench_target EXIT

if ! start_durable_report_card_log; then
  exit 2
fi

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
  echo "#     point batches=${FULL_POINT_BATCH_SIZES}, B runs=${FULL_POINT_IN_L2_SAMPLE_COUNT} fixed samples of ${FULL_POINT_BATCHES_IN_L2}, C samples=${FULL_POINT_BATCHES_OUT_OF_L2},"
  echo "#     B qualification=median >= ${FULL_POINT_IN_L2_MIN_LOOKUPS_PER_S} lookups/s (${FULL_POINT_IN_L2_REQUIRED_QUALIFYING_SAMPLES}/${FULL_POINT_IN_L2_SAMPLE_COUNT}), warmup=${FULL_POINT_WARMUP},"
  echo "#     B threads=${FULL_POINT_THREADS_IN_L2}, C threads=none, gap=${GPU_GAP}s."
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
raw_binary_sha256="$(sha256sum "$raw_binary" | awk '{print $1}')"
point_binary_sha256="$(sha256sum "$point_binary" | awk '{print $1}')"

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

print_failed_execution_status_and_exit() {
  local failure_class
  local failure_reason="section_evidence_incomplete"
  local failure
  local has_environment_probe_failure=0
  local has_execution_probe_failure=0
  failure_class="$(
    report_card_failure_class \
      "${point_gate_outcome:-not_evaluated}" "${section_failures[@]}"
  )"
  for failure in "${section_failures[@]}"; do
    case "$failure" in
      *:environment) has_environment_probe_failure=1 ;;
      *:execution) has_execution_probe_failure=1 ;;
    esac
  done
  case "$failure_class" in
    candidate_performance_failure)
      failure_reason="section_b_all_samples_below_floor"
      ;;
    environment_invalid)
      if [[ "$has_environment_probe_failure" -eq 1 ]]; then
        failure_reason="environment_qualification_invalid"
      else
        failure_reason="section_b_indeterminate_bimodality"
      fi
      ;;
    execution_invalid)
      if [[ "$has_execution_probe_failure" -eq 1 ]]; then
        failure_reason="structural_provenance_invalid"
      elif [[ "${point_gate_outcome:-not_evaluated}" == "execution_invalid" ]]; then
        failure_reason="section_b_execution_evidence_invalid"
      else
        failure_reason="section_evidence_incomplete"
      fi
      ;;
  esac
  if [[ "$mode" == "quick" ]]; then
    print_execution_status "screen-incomplete" "canonical=false failure_class=${failure_class} reason=${failure_reason} failed_sections=${section_failures[*]}"
  elif [[ "$failure_class" == "environment_invalid" ]]; then
    print_execution_status "invalid" "failure_class=${failure_class} reason=${failure_reason} failed_sections=${section_failures[*]}"
  else
    print_execution_status "incomplete" "failure_class=${failure_class} reason=${failure_reason} failed_sections=${section_failures[*]}"
  fi
  exit 1
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

if ! benchmark_gpu_identity="$(gpu_static_identity)"; then
  print_execution_status "invalid" "failure_class=environment_invalid reason=gpu_identity_unavailable"
  exit 1
fi
benchmark_gpu_identity_sha256="$(printf '%s' "$benchmark_gpu_identity" | sha256sum | awk '{print $1}')"
benchmark_configuration_sha256="$(benchmark_configuration_fingerprint)"
benchmark_b_context_identity_sha256=""
gpu_environment_checks_enabled=1
echo "# GPU qualification baseline: gpu_identity_sha256=${benchmark_gpu_identity_sha256} configuration_sha256=${benchmark_configuration_sha256} raw_binary_sha256=${raw_binary_sha256} point_binary_sha256=${point_binary_sha256}"
echo "# GPU qualification identity: ${benchmark_gpu_identity}"

# ---- Section A: RAW READ KERNELS (emits IN-L2 + OUT-OF-L2 in one invocation) ----
run_section A "RAW READ KERNELS (read_kernel_roofline -- IN-L2 + OUT-OF-L2)" "${SECTION_A_TIMEOUT}" \
  "$raw_completion_marker" "${raw_command[@]}"

echo ""
echo "### GPU cool-down: sleep ${GPU_GAP}"
sleep "${GPU_GAP}"

# ---- Section B: fixed lpb/wave ENGINE POINT-READ cohort, IN-L2 ----
# This is one fixed experiment, not retry-until-pass: each B sample runs even if a
# preceding sample is slow or invalid. Section C is eligible only after the cohort's
# complete, valid, median decision passes.
point_gate_outcome="not_evaluated"
point_b_gate_log="$bench_target_dir/report-card-section-B-gate.log"
point_b_post_log="$bench_target_dir/report-card-section-B3-post.log"
point_b_sample_logs=()
for ((point_sample = 1; point_sample <= 10#$FULL_POINT_IN_L2_SAMPLE_COUNT; point_sample += 1)); do
  point_b_sample_logs+=("$bench_target_dir/report-card-section-B${point_sample}.log")
  run_section "B${point_sample}" \
    "lpb/wave ENGINE POINT READS -- IN-L2 sample ${point_sample}/${FULL_POINT_IN_L2_SAMPLE_COUNT} (1M rows = 4MB/col, cache-resident)" \
    "${SECTION_B_TIMEOUT}" "$point_b_completion_marker" "${point_b_command[@]}"
  if ((point_sample < 10#$FULL_POINT_IN_L2_SAMPLE_COUNT)); then
    echo ""
    echo "### GPU cool-down: sleep ${GPU_GAP} before Section B$((point_sample + 1))"
    sleep "${GPU_GAP}"
  fi
done

# The B3-post probe closes the cohort, including its context-inventory stability
# check, before the fixed-sample decision is allowed to consider the measurements.
benchmark_environment_before_section "B3-post" >"$point_b_post_log"
point_b_post_rc=$?
cat "$point_b_post_log"
if [[ "$point_b_post_rc" -ne 0 ]]; then
  echo "[Section B cohort INCOMPLETE: post-sample environment/provenance qualification failed]"
  record_environment_probe_failure "B" "$point_b_post_rc"
fi

if [[ "${#section_failures[@]}" -eq 0 ]]; then
  enforce_point_production_throughput_floor \
    "$FULL_POINT_IN_L2_FLOOR_BATCH" \
    "$FULL_POINT_IN_L2_MIN_LOOKUPS_PER_S" \
    "$FULL_POINT_IN_L2_SAMPLE_COUNT" \
    "$FULL_POINT_IN_L2_REQUIRED_QUALIFYING_SAMPLES" \
    "${point_b_sample_logs[@]}" >"$point_b_gate_log"
  point_gate_rc=$?
  cat "$point_b_gate_log"
  if [[ "$point_gate_rc" -ne 0 ]]; then
    section_failures+=("B:${point_gate_outcome}")
  fi
fi

if [[ "${#section_failures[@]}" -ne 0 ]]; then
  print_failed_execution_status_and_exit
fi

if [[ "$mode" == "quick" ]]; then
  print_execution_status "screen-complete" "sections=A,B canonical=false"
  exit 0
fi

if ! should_run_section_c "$mode" "${#section_failures[@]}" "$point_gate_outcome"; then
  section_failures+=("C:precondition")
  print_failed_execution_status_and_exit
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
  print_failed_execution_status_and_exit
fi

benchmark_environment_before_section "closeout" >"$bench_target_dir/report-card-closeout.log"
closeout_environment_rc=$?
cat "$bench_target_dir/report-card-closeout.log"
if [[ "$closeout_environment_rc" -ne 0 ]]; then
  record_environment_probe_failure "closeout" "$closeout_environment_rc"
  print_failed_execution_status_and_exit
fi

print_execution_status "complete" "sections=A,B,C canonical=true"
