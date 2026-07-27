#!/usr/bin/env bash
#
# INSERT-001 qualification runner.
#
# This is deliberately a qualification artifact, not a throughput shortcut.  It generates one
# deterministic statement stream, builds the canonical pgwire server and external client once,
# and replays those exact bytes through fresh GPU-server and PostgreSQL processes in alternating
# order.  PostgreSQL COPY, outer transactions, and direct Engine callers are not part of this
# runner.
#
# The default is a 1M-row calibration for both durability profiles.  Select 8M for the second
# calibration or 48M for the seal.  Every invocation still runs exactly three complete trials per
# backend, with the fixed 1,000-row statement boundary.
#
#   scripts/benchmark_insert_qualification.sh --profile development --rows 1000000
#   scripts/benchmark_insert_qualification.sh --profile durable --rows 48000000
#   scripts/benchmark_insert_qualification.sh --self-check
#
# Artifacts are retained.  They contain command output, process/container logs, identities, and
# the source/manifest used by every trial.  Local PostgreSQL uses trust authentication only; this
# script neither accepts nor prints credentials.

set -uo pipefail

readonly SCRIPT_NAME="benchmark_insert_qualification.sh"
readonly INSERT_WORKLOAD_ID="insert001-accounts-int4-v1"
readonly INSERT_CHUNK="1000"
readonly INSERT_TRIALS="3"
readonly POSTGRES_IMAGE_DEFAULT="postgres@sha256:33f923b05f64ca54ac4401c01126a6b92afe839a0aa0a52bc5aeb5cc958e5f20"
readonly POSTGRES_IMAGE_ID="sha256:33f923b05f64ca54ac4401c01126a6b92afe839a0aa0a52bc5aeb5cc958e5f20"
readonly POSTGRES_VERSION_NUM="160014"
readonly POSTGRES_VERSION="16.14"
readonly POSTGRES_WAL_SYNC_METHOD="fdatasync"
readonly DEFAULT_CLIENT_TIMEOUT_SECONDS="7200"
# AWK represents integers exactly only through 2^53 - 1. The qualification parser compares
# classifier counters arithmetically, so derive a four-gates-per-INSERT floor only inside that
# exact range and reject any manifest that would exceed it.
readonly AWK_EXACT_UINT_MAX="9007199254740991"
readonly CLASSIFIER_GATE_MAX_INSERT_STATEMENTS="2251799813685247"

frozen_workload_identity() {
  # Byte-exact INSERT-001 v1 identities, generated from the reviewed workload definition and
  # frozen before engine optimization. Output fields are:
  #   source_bytes source_sha256 statement_stream_sha256 manifest_sha256
  # The statement-stream digest hashes each statement as u64-le length || exact statement bytes,
  # so it binds both the bytes and the 1,000-row autocommit boundaries.
  case "$1" in
    1000000)
      printf '%s\n' \
        "15820837 7564cc9e47cce5fe1f2484ff0107f5aa1c73258bb5c8d0fa70f43a2cc1c65f23 ce0d027d434420874d361e1203f0c1094a66c20a68dd7b8b03c60c8d7d2e3393 a0d7472f39059f6694f5cd9cd28632dd5574506dc25d903debfa8642f87f8ea4"
      ;;
    8000000)
      printf '%s\n' \
        "134344137 b5fccd46454bf9bb41fda29f6afb85a9af650fe19fe8b87d415e4bd54d0017d2 4b0bf8519415d560e584aa7a8c95882fb6436d53d0cb2448381f8372d4f69214 bdca1fcc6127739d1ba81db8b247b6546854774547135150ca42fa4937013b18"
      ;;
    48000000)
      printf '%s\n' \
        "849620137 09472115e419b22a685f23cfb956406e57f9b91324e7f52db4ba855f7a4c0cb9 8f5d67db9521bfd345edd8956691ed9ee27c7b780cd68a57d4d00c9449ace44e a2266305f578762641f188f2d99be68762fbd371d27110a0f65386be756b01f0"
      ;;
    *) return 1 ;;
  esac
}

frozen_postgres_rows_per_second_floor() {
  # Floors are added only after a pre-optimization alternating qualification has completed.
  # They are keyed by exact profile and workload size; never extrapolate a calibration size.
  # 1M frozen 2026-07-26 from source SHA-256
  # 7564cc9e47cce5fe1f2484ff0107f5aa1c73258bb5c8d0fa70f43a2cc1c65f23.
  # 8M frozen 2026-07-26 from source SHA-256
  # b5fccd46454bf9bb41fda29f6afb85a9af650fe19fe8b87d415e4bd54d0017d2.
  # 48M frozen 2026-07-26 from source SHA-256
  # 09472115e419b22a685f23cfb956406e57f9b91324e7f52db4ba855f7a4c0cb9.
  case "$1:$2" in
    development:1000000) printf '%s\n' "675859.699" ;;
    durable:1000000) printf '%s\n' "433754.096" ;;
    development:8000000) printf '%s\n' "765543.025" ;;
    durable:8000000) printf '%s\n' "436488.945" ;;
    development:48000000) printf '%s\n' "778301.447" ;;
    durable:48000000) printf '%s\n' "235965.297" ;;
    *) return 1 ;;
  esac
}

usage() {
  cat <<'USAGE'
usage: scripts/benchmark_insert_qualification.sh [options]

  --profile development|durable|all  Run one matched profile or both (default: all).
  --rows 1000000|8000000|48000000   Calibration/seal row count (default: 1000000).
  --artifact-dir DIR                 New, empty evidence directory (default: target/insert-qualification.*).
  --qualification                    Explicitly select the only execution mode (default).
  --self-check                       Exercise parser and sabotage checks without Cargo, GPU, or Docker.
  --help                             Show this help.

Every execution uses exactly three alternating GPU, PostgreSQL trials per selected profile and
the fixed 1,000-literal-row INSERT statement boundary.  --rows 48000000 is the seal workload.
USAGE
}

die() {
  echo "${SCRIPT_NAME}: $*" >&2
  exit 2
}

is_canonical_uint() {
  [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]
}

require_positive_uint() {
  local value="$1"
  local label="$2"
  is_canonical_uint "$value" && ((10#$value > 0)) || die "${label} must be a positive canonical integer"
}

checked_classifier_gate_minimum() {
  # Every INSERT-001 statement traverses COPY, cursor, PREPARE-name, and prepared-action gates.
  # Stay within AWK's exact-integer range because the parser also reconciles these counters.
  local insert_statements="${1-}"
  is_canonical_uint "$insert_statements" && [[ "$insert_statements" != "0" ]] || return 1
  if ((${#insert_statements} > ${#CLASSIFIER_GATE_MAX_INSERT_STATEMENTS})) ||
    ((${#insert_statements} == ${#CLASSIFIER_GATE_MAX_INSERT_STATEMENTS} &&
      insert_statements > CLASSIFIER_GATE_MAX_INSERT_STATEMENTS)); then
    return 1
  fi
  printf '%s\n' "$((10#$insert_statements * 4))"
}

profile="all"
rows="1000000"
artifact_arg=""
mode="qualification"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ "$#" -ge 2 ]] || die "--profile requires a value"
      profile="$2"
      shift 2
      ;;
    --rows)
      [[ "$#" -ge 2 ]] || die "--rows requires a value"
      rows="$2"
      shift 2
      ;;
    --artifact-dir)
      [[ "$#" -ge 2 ]] || die "--artifact-dir requires a value"
      artifact_arg="$2"
      shift 2
      ;;
    --qualification)
      mode="qualification"
      shift
      ;;
    --self-check)
      mode="self-check"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$profile" in
  development|durable|all) ;;
  *) die "--profile must be development, durable, or all" ;;
esac
case "$rows" in
  1000000|8000000|48000000) ;;
  *) die "--rows must be one of 1000000, 8000000, or 48000000" ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

file_size() {
  stat -c '%s' -- "$1"
}

write_exit_status() {
  local destination="$1"
  local status="$2"
  printf '%s\n' "$status" >"$destination"
}

capture_command() {
  # capture_command <artifact-prefix> <command...>
  local prefix="$1"
  shift
  local status
  "$@" >"${prefix}.stdout" 2>"${prefix}.stderr"
  status=$?
  write_exit_status "${prefix}.exit-status" "$status"
  return "$status"
}

capture_optional_command() {
  # Optional host-identity probes are still recorded when their executable is unavailable.
  local prefix="$1"
  shift
  local command_name="$1"
  shift
  if command -v "$command_name" >/dev/null 2>&1; then
    capture_command "$prefix" "$@" || true
  else
    printf 'unavailable: %s is not installed\n' "$command_name" >"${prefix}.stdout"
    : >"${prefix}.stderr"
    write_exit_status "${prefix}.exit-status" "127"
  fi
}

manifest_rows=""
manifest_chunk=""
manifest_insert_statements=""
manifest_total_statements=""
manifest_source_bytes=""
manifest_source_sha256=""
manifest_statement_stream_sha256=""

load_manifest_identity() {
  local manifest_path="$1"
  local parsed
  parsed="$(awk '
    function canonical_uint(value) {
      return value == "0" || value ~ /^[1-9][0-9]*$/
    }
    function digest(value) {
      return length(value) == 64 && value ~ /^[0-9a-f]+$/
    }
    BEGIN {
      expected["manifest_version"] = 1
      expected["workload"] = 1
      expected["rows"] = 1
      expected["chunk"] = 1
      expected["insert_statements"] = 1
      expected["total_statements"] = 1
      expected["source_bytes"] = 1
      expected["source_sha256"] = 1
      expected["statement_stream_sha256"] = 1
    }
    {
      if ($0 == "" || index($0, "=") == 0) bad = 1
      piece_count = split($0, pieces, "=")
      if (piece_count != 2 || pieces[1] == "" || pieces[2] == "") bad = 1
      key = pieces[1]
      value = pieces[2]
      if (!(key in expected) || ++seen[key] != 1) bad = 1
      values[key] = value
    }
    END {
      for (key in expected) if (seen[key] != 1) bad = 1
      if (values["manifest_version"] != "1" ||
          values["workload"] != "insert001-accounts-int4-v1" ||
          !canonical_uint(values["rows"]) || values["rows"] == "0" ||
          !canonical_uint(values["chunk"]) || values["chunk"] == "0" ||
          !canonical_uint(values["insert_statements"]) ||
          !canonical_uint(values["total_statements"]) ||
          !canonical_uint(values["source_bytes"]) ||
          !digest(values["source_sha256"]) ||
          !digest(values["statement_stream_sha256"])) bad = 1
      if (bad) exit 1
      print "rows=" values["rows"]
      print "chunk=" values["chunk"]
      print "insert_statements=" values["insert_statements"]
      print "total_statements=" values["total_statements"]
      print "source_bytes=" values["source_bytes"]
      print "source_sha256=" values["source_sha256"]
      print "statement_stream_sha256=" values["statement_stream_sha256"]
    }
  ' "$manifest_path")" || return 1
  manifest_rows=""
  manifest_chunk=""
  manifest_insert_statements=""
  manifest_total_statements=""
  manifest_source_bytes=""
  manifest_source_sha256=""
  manifest_statement_stream_sha256=""
  local key value
  while IFS='=' read -r key value; do
    case "$key" in
      rows) manifest_rows="$value" ;;
      chunk) manifest_chunk="$value" ;;
      insert_statements) manifest_insert_statements="$value" ;;
      total_statements) manifest_total_statements="$value" ;;
      source_bytes) manifest_source_bytes="$value" ;;
      source_sha256) manifest_source_sha256="$value" ;;
      statement_stream_sha256) manifest_statement_stream_sha256="$value" ;;
      *) return 1 ;;
    esac
  done <<<"$parsed"
  [[ -n "$manifest_rows" && -n "$manifest_chunk" && -n "$manifest_insert_statements" &&
    -n "$manifest_total_statements" && -n "$manifest_source_bytes" &&
    -n "$manifest_source_sha256" && -n "$manifest_statement_stream_sha256" ]]
}

source_path=""
manifest_path=""
source_file_sha256=""
manifest_file_sha256=""
server_binary=""
client_binary=""
server_binary_sha256=""
client_binary_sha256=""

verify_frozen_inputs() {
  [[ "$(sha256_file "$source_path")" == "$source_file_sha256" ]] || return 1
  [[ "$(sha256_file "$manifest_path")" == "$manifest_file_sha256" ]] || return 1
  [[ "$(sha256_file "$server_binary")" == "$server_binary_sha256" ]] || return 1
  [[ "$(sha256_file "$client_binary")" == "$client_binary_sha256" ]] || return 1
  load_manifest_identity "$manifest_path" || return 1
  [[ "$manifest_rows" == "$rows" && "$manifest_chunk" == "$INSERT_CHUNK" &&
    "$manifest_source_sha256" == "$source_file_sha256" ]]
}

# The server attributes raw simple-query source bytes at the Query-message boundary. Canonical
# DML identity is intentionally derived after `split_simple_query` removes this workload's exact
# `;\n` statement boundary. Derive both domains from the already frozen/verified stream instead
# of assuming a byte subtraction from an uninspected source file.
derive_insert_request_identity_bytes() {
  # derive_insert_request_identity_bytes <source> <expected-insert-statements>
  local source_file="$1"
  local expected_inserts="$2"
  LC_ALL=C awk -v expected_inserts="$expected_inserts" '
    BEGIN {
      create = "CREATE TABLE accounts (id int4, balance int4);"
    }
    NR == 1 {
      if ($0 != create) bad = 1
      next
    }
    {
      if (substr($0, length($0), 1) != ";" ||
          substr($0, 1, 42) != "INSERT INTO accounts (id, balance) VALUES ") bad = 1
      insert_source_bytes += length($0) + 1
      canonical_request_bytes += length($0) - 1
      stripped_terminator_bytes += 2
      insert_statements += 1
    }
    END {
      if (bad || NR != expected_inserts + 1 || insert_statements != expected_inserts ||
          stripped_terminator_bytes != insert_statements * 2 ||
          canonical_request_bytes + stripped_terminator_bytes != insert_source_bytes) exit 1
      printf "%.0f %.0f %.0f\n", insert_source_bytes, canonical_request_bytes, stripped_terminator_bytes
    }
  ' "$source_file"
}

# Parse exactly one server-side INSERT probe record. Unlike the client record, this is valid only
# for the GPU backend and seals the server's engine-local aggregate delta at connection close.
# Raw source bytes include each frozen `;\n`; canonical request bytes exclude precisely those two
# parser-stripped bytes per INSERT and are independently required below.
parse_gpu_insert_probe_record() {
  # parse_gpu_insert_probe_record <server-stderr> <profile> <trial>
  local stderr_log="$1"
  local expected_profile="$2"
  local expected_trial="$3"
  local expected_classifier_gate_minimum
  expected_classifier_gate_minimum="$(checked_classifier_gate_minimum "$manifest_insert_statements")" || return 1
  awk \
    -v expected_profile="$expected_profile" \
    -v expected_trial="$expected_trial" \
    -v expected_rows="$rows" \
    -v expected_insert_statements="$manifest_insert_statements" \
    -v expected_classifier_gate_minimum="$expected_classifier_gate_minimum" \
    -v awk_exact_uint_max="$AWK_EXACT_UINT_MAX" \
    -v expected_insert_source_bytes="$insert_source_bytes" \
    -v expected_canonical_request_bytes="$insert_canonical_request_bytes" \
    -v expected_stripped_terminator_bytes="$insert_stripped_terminator_bytes" '
    function canonical_uint(value) {
      return value == "0" || value ~ /^[1-9][0-9]*$/
    }
    function decimal_at_most(value, maximum, position, digit, maximum_digit) {
      if (length(value) != length(maximum)) return length(value) < length(maximum)
      for (position = 1; position <= length(value); position += 1) {
        digit = substr(value, position, 1)
        maximum_digit = substr(maximum, position, 1)
        if (digit != maximum_digit)
          return index("0123456789", digit) < index("0123456789", maximum_digit)
      }
      return 1
    }
    function exact_awk_uint(value) {
      return canonical_uint(value) && decimal_at_most(value, awk_exact_uint_max)
    }
    $1 == "insert_probe_session_delta=complete" {
      records += 1
      delete value
      delete seen
      for (field = 1; field <= NF; field += 1) {
        piece_count = split($field, pieces, "=")
        if (piece_count != 2 || pieces[1] == "" || pieces[2] == "") {
          bad = 1
          continue
        }
        if (++seen[pieces[1]] != 1) bad = 1
        value[pieces[1]] = pieces[2]
      }
      required["insert_probe_session_delta"] = 1
      required["version"] = 1
      required["attribution"] = 1
      required["successful_insert_statements"] = 1
      required["successful_insert_rows"] = 1
      required["fixed_insert_typed_commits"] = 1
      required["fixed_insert_legacy_fallbacks"] = 1
      required["fixed_insert_retryable_declines"] = 1
      required["fixed_insert_legacy_commit_validation_reresolves"] = 1
      required["legacy_insert_delta_builds"] = 1
      required["predicted_row_keys_materialized"] = 1
      required["direct_fixed_insert_carriers"] = 1
      required["raw_request_digest_derivations"] = 1
      required["raw_request_digest_derivation_bytes"] = 1
      required["compat_classifier_gate_checks"] = 1
      required["compat_classifier_gate_rejections"] = 1
      required["compat_classifier_slow_path_admissions"] = 1
      required["compat_classifier_slow_path_bytes"] = 1
      required["successful_insert_source_bytes"] = 1
      required["successful_insert_end_to_end_service_nanos"] = 1
      required["facade_parse_bind_nanos"] = 1
      required["engine_authorization_catalog_admission_nanos"] = 1
      required["offlock_coercion_default_constraint_prepare_nanos"] = 1
      required["commit_validation_reresolve_nanos"] = 1
      required["canonical_wal_encode_append_claim_nanos"] = 1
      required["durability_wait_nanos"] = 1
      required["durability_begin_group_flush_nanos"] = 1
      required["durability_job_wait_nanos"] = 1
      required["fua_logical_groups"] = 1
      required["fua_logical_payload_bytes"] = 1
      required["fua_single_frame_padded_baseline_bytes"] = 1
      required["fua_publish_turn_wait_nanos"] = 1
      required["fua_publish_turn_wait_groups"] = 1
      required["fua_published_frames"] = 1
      required["fua_fenced_frames"] = 1
      required["fua_fence_failures"] = 1
      required["fua_payload_bytes"] = 1
      required["fua_padded_bytes"] = 1
      required["fua_stage_copy_nanos"] = 1
      required["fua_stage_copy_frames"] = 1
      required["fua_publish_to_claim_nanos"] = 1
      required["fua_publish_to_claim_frames"] = 1
      required["fua_claim_to_write_done_nanos"] = 1
      required["fua_claim_to_write_done_frames"] = 1
      required["fua_write_done_to_contiguous_cut_nanos"] = 1
      required["fua_write_done_to_contiguous_cut_frames"] = 1
      required["fua_contiguous_cut_events"] = 1
      required["fua_contiguous_cut_advanced_frames"] = 1
      required["fua_contiguous_cut_advance_max_frames"] = 1
      required["fua_waiter_cut_to_observe_nanos"] = 1
      required["fua_waiter_cut_to_observe_count"] = 1
      required["fua_in_flight_depth_max"] = 1
      required["fua_in_flight_depth_1"] = 1
      required["fua_in_flight_depth_2"] = 1
      required["fua_in_flight_depth_3_to_4"] = 1
      required["fua_in_flight_depth_5_to_8"] = 1
      required["fua_in_flight_depth_9_to_16"] = 1
      required["fua_in_flight_depth_17_to_32"] = 1
      required["fua_in_flight_depth_33_plus"] = 1
      required["fua_controller_sustained_actions"] = 1
      required["fua_controller_pending_probe_cover_actions"] = 1
      required["fua_controller_qd1_samples"] = 1
      required["fua_controller_qd1_sparse_actions"] = 1
      required["fua_controller_qd1_verify_actions"] = 1
      required["fua_controller_qd1_fast_actions"] = 1
      required["fua_controller_unfragmented_actions"] = 1
      required["fua_controller_pool_too_narrow"] = 1
      required["fua_controller_empty_chunk"] = 1
      required["fua_controller_insufficient_free_slots"] = 1
      required["fua_controller_natural_depth"] = 1
      required["fua_controller_segment_boundary"] = 1
      required["fua_controller_amplification_cap"] = 1
      required["fua_controller_fast_samples"] = 1
      required["fua_controller_nonfast_samples"] = 1
      required["fua_controller_transitions_to_verify"] = 1
      required["fua_controller_transitions_to_fast"] = 1
      required["fua_controller_transitions_to_sustained"] = 1
      required["fua_controller_stale_qd1_samples"] = 1
      required["fua_controller_unavailable_qd1_samples"] = 1
      required["fua_controller_abandoned_qd1_samples"] = 1
      required["fua_controller_protocol_faults"] = 1
      required["fua_controller_protocol_fallback_actions"] = 1
      required["fua_controller_phase"] = 1
      required["fua_controller_verify_fast_streak"] = 1
      required["fua_controller_sustained_remaining"] = 1
      required["fua_controller_generation"] = 1
      required["fua_controller_pending_qd1_samples"] = 1
      required["fua_controller_fast_in_flight"] = 1
      required["fua_controller_fast_in_flight_max"] = 1
      required["fua_controller_generation_exhausted"] = 1
      required["fua_controller_ordinal_exhausted"] = 1
      required["fua_controller_action_reconciliation"] = 1
      required["fua_controller_sample_reconciliation"] = 1
      required["device_validate_nanos"] = 1
      required["device_h2d_append_index_apply_nanos"] = 1
      required["publication_status_ack_nanos"] = 1
      required["wave_count"] = 1
      required["wave_item_count"] = 1
      required["peak_host_statement_bytes"] = 1
      required["peak_device_statement_bytes_estimate"] = 1
      required["rollover_count"] = 1
      required["rollover_capacity_rows_total"] = 1
      required["rollover_capacity_rows_max"] = 1
      required["current_shard_count"] = 1
      required["peak_shard_count"] = 1
      required["persistent_allocation_count"] = 1
      required["descriptor_clone_visit_count"] = 1
      required["budget_scan_entries"] = 1
      required["capacity_fit_evaluation_count"] = 1
      required["sidecar_fill_bytes"] = 1
      required["live_h2d_bytes"] = 1
      required["named_index_shard_visits"] = 1
      required["unattributed_or_concurrent_overlap_nanos"] = 1
      required["durability_backend"] = 1
      required["fua_fence_lanes"] = 1
      required["intent_lane_count"] = 1
      required["synchronous_commit_gate"] = 1
      required["auto_admit_on_commit"] = 1
      required["binary_wal_records_enabled"] = 1
      required["device_authoritative_commits"] = 1
      required["active_workload_sessions_at_seal"] = 1
      required["overlap_observed"] = 1
      for (key in required) if (seen[key] != 1) bad = 1
      for (key in seen) if (!(key in required)) bad = 1
      if (value["insert_probe_session_delta"] != "complete" || value["version"] != "7" ||
          value["attribution"] != "engine_local_plus_process_global_classifier_counters_single_active_workload_session_required" ||
          value["successful_insert_statements"] != expected_insert_statements ||
          value["successful_insert_rows"] != expected_rows ||
          value["successful_insert_source_bytes"] != expected_insert_source_bytes ||
          value["raw_request_digest_derivations"] != expected_insert_statements ||
          value["raw_request_digest_derivation_bytes"] != expected_canonical_request_bytes ||
          value["synchronous_commit_gate"] != "strict_rpo0" ||
          value["active_workload_sessions_at_seal"] != "1" ||
          value["overlap_observed"] != "false" ||
          value["binary_wal_records_enabled"] != "true") bad = 1
      if (expected_profile == "development" && (value["durability_backend"] != "memory" ||
          value["intent_lane_count"] != "0" || value["fua_fence_lanes"] != "0")) bad = 1
      if (expected_profile == "durable" && (value["durability_backend"] != "fua" ||
          value["intent_lane_count"] != "10" || value["fua_fence_lanes"] == "0")) bad = 1
      numeric["successful_insert_statements"] = 1
      numeric["successful_insert_rows"] = 1
      numeric["fixed_insert_typed_commits"] = 1
      numeric["fixed_insert_legacy_fallbacks"] = 1
      numeric["fixed_insert_retryable_declines"] = 1
      numeric["fixed_insert_legacy_commit_validation_reresolves"] = 1
      numeric["legacy_insert_delta_builds"] = 1
      numeric["predicted_row_keys_materialized"] = 1
      numeric["direct_fixed_insert_carriers"] = 1
      numeric["raw_request_digest_derivations"] = 1
      numeric["raw_request_digest_derivation_bytes"] = 1
      numeric["compat_classifier_gate_checks"] = 1
      numeric["compat_classifier_gate_rejections"] = 1
      numeric["compat_classifier_slow_path_admissions"] = 1
      numeric["compat_classifier_slow_path_bytes"] = 1
      numeric["successful_insert_source_bytes"] = 1
      numeric["successful_insert_end_to_end_service_nanos"] = 1
      numeric["facade_parse_bind_nanos"] = 1
      numeric["engine_authorization_catalog_admission_nanos"] = 1
      numeric["offlock_coercion_default_constraint_prepare_nanos"] = 1
      numeric["commit_validation_reresolve_nanos"] = 1
      numeric["canonical_wal_encode_append_claim_nanos"] = 1
      numeric["durability_wait_nanos"] = 1
      numeric["durability_begin_group_flush_nanos"] = 1
      numeric["durability_job_wait_nanos"] = 1
      numeric["fua_logical_groups"] = 1
      numeric["fua_logical_payload_bytes"] = 1
      numeric["fua_single_frame_padded_baseline_bytes"] = 1
      numeric["fua_publish_turn_wait_nanos"] = 1
      numeric["fua_publish_turn_wait_groups"] = 1
      numeric["fua_published_frames"] = 1
      numeric["fua_fenced_frames"] = 1
      numeric["fua_fence_failures"] = 1
      numeric["fua_payload_bytes"] = 1
      numeric["fua_padded_bytes"] = 1
      numeric["fua_stage_copy_nanos"] = 1
      numeric["fua_stage_copy_frames"] = 1
      numeric["fua_publish_to_claim_nanos"] = 1
      numeric["fua_publish_to_claim_frames"] = 1
      numeric["fua_claim_to_write_done_nanos"] = 1
      numeric["fua_claim_to_write_done_frames"] = 1
      numeric["fua_write_done_to_contiguous_cut_nanos"] = 1
      numeric["fua_write_done_to_contiguous_cut_frames"] = 1
      numeric["fua_contiguous_cut_events"] = 1
      numeric["fua_contiguous_cut_advanced_frames"] = 1
      numeric["fua_contiguous_cut_advance_max_frames"] = 1
      numeric["fua_waiter_cut_to_observe_nanos"] = 1
      numeric["fua_waiter_cut_to_observe_count"] = 1
      numeric["fua_in_flight_depth_max"] = 1
      numeric["fua_in_flight_depth_1"] = 1
      numeric["fua_in_flight_depth_2"] = 1
      numeric["fua_in_flight_depth_3_to_4"] = 1
      numeric["fua_in_flight_depth_5_to_8"] = 1
      numeric["fua_in_flight_depth_9_to_16"] = 1
      numeric["fua_in_flight_depth_17_to_32"] = 1
      numeric["fua_in_flight_depth_33_plus"] = 1
      numeric["fua_controller_sustained_actions"] = 1
      numeric["fua_controller_pending_probe_cover_actions"] = 1
      numeric["fua_controller_qd1_samples"] = 1
      numeric["fua_controller_qd1_sparse_actions"] = 1
      numeric["fua_controller_qd1_verify_actions"] = 1
      numeric["fua_controller_qd1_fast_actions"] = 1
      numeric["fua_controller_unfragmented_actions"] = 1
      numeric["fua_controller_pool_too_narrow"] = 1
      numeric["fua_controller_empty_chunk"] = 1
      numeric["fua_controller_insufficient_free_slots"] = 1
      numeric["fua_controller_natural_depth"] = 1
      numeric["fua_controller_segment_boundary"] = 1
      numeric["fua_controller_amplification_cap"] = 1
      numeric["fua_controller_fast_samples"] = 1
      numeric["fua_controller_nonfast_samples"] = 1
      numeric["fua_controller_transitions_to_verify"] = 1
      numeric["fua_controller_transitions_to_fast"] = 1
      numeric["fua_controller_transitions_to_sustained"] = 1
      numeric["fua_controller_stale_qd1_samples"] = 1
      numeric["fua_controller_unavailable_qd1_samples"] = 1
      numeric["fua_controller_abandoned_qd1_samples"] = 1
      numeric["fua_controller_protocol_faults"] = 1
      numeric["fua_controller_protocol_fallback_actions"] = 1
      numeric["fua_controller_phase"] = 1
      numeric["fua_controller_verify_fast_streak"] = 1
      numeric["fua_controller_sustained_remaining"] = 1
      numeric["fua_controller_generation"] = 1
      numeric["fua_controller_pending_qd1_samples"] = 1
      numeric["fua_controller_fast_in_flight"] = 1
      numeric["fua_controller_fast_in_flight_max"] = 1
      numeric["fua_controller_generation_exhausted"] = 1
      numeric["fua_controller_ordinal_exhausted"] = 1
      numeric["fua_controller_action_reconciliation"] = 1
      numeric["fua_controller_sample_reconciliation"] = 1
      numeric["device_validate_nanos"] = 1
      numeric["device_h2d_append_index_apply_nanos"] = 1
      numeric["publication_status_ack_nanos"] = 1
      numeric["wave_count"] = 1
      numeric["wave_item_count"] = 1
      numeric["peak_host_statement_bytes"] = 1
      numeric["peak_device_statement_bytes_estimate"] = 1
      numeric["rollover_count"] = 1
      numeric["rollover_capacity_rows_total"] = 1
      numeric["rollover_capacity_rows_max"] = 1
      numeric["current_shard_count"] = 1
      numeric["peak_shard_count"] = 1
      numeric["persistent_allocation_count"] = 1
      numeric["descriptor_clone_visit_count"] = 1
      numeric["budget_scan_entries"] = 1
      numeric["capacity_fit_evaluation_count"] = 1
      numeric["sidecar_fill_bytes"] = 1
      numeric["live_h2d_bytes"] = 1
      numeric["named_index_shard_visits"] = 1
      numeric["unattributed_or_concurrent_overlap_nanos"] = 1
      numeric["fua_fence_lanes"] = 1
      numeric["intent_lane_count"] = 1
      numeric["device_authoritative_commits"] = 1
      for (key in numeric) if (!canonical_uint(value[key])) bad = 1
      if (!exact_awk_uint(expected_insert_statements) ||
          !exact_awk_uint(expected_classifier_gate_minimum) ||
          !exact_awk_uint(expected_insert_source_bytes) ||
          !exact_awk_uint(expected_canonical_request_bytes) ||
          !exact_awk_uint(expected_stripped_terminator_bytes) ||
          !exact_awk_uint(value["compat_classifier_gate_checks"]) ||
          !exact_awk_uint(value["compat_classifier_gate_rejections"]) ||
          !exact_awk_uint(value["compat_classifier_slow_path_admissions"])) bad = 1
      if (expected_classifier_gate_minimum != expected_insert_statements * 4) bad = 1
      if (expected_stripped_terminator_bytes != expected_insert_statements * 2 ||
          expected_canonical_request_bytes + expected_stripped_terminator_bytes != expected_insert_source_bytes ||
          value["raw_request_digest_derivations"] != value["successful_insert_statements"] ||
          value["raw_request_digest_derivation_bytes"] + expected_stripped_terminator_bytes != value["successful_insert_source_bytes"]) bad = 1
      if (value["auto_admit_on_commit"] != "true" ||
          value["device_authoritative_commits"] != expected_insert_statements ||
          value["fixed_insert_typed_commits"] != expected_insert_statements ||
          value["fixed_insert_legacy_fallbacks"] != "0" ||
          value["fixed_insert_retryable_declines"] != "0" ||
          value["fixed_insert_legacy_commit_validation_reresolves"] != "0" ||
          value["legacy_insert_delta_builds"] != "0" ||
          value["predicted_row_keys_materialized"] != "0" ||
          value["direct_fixed_insert_carriers"] != expected_insert_statements ||
          value["compat_classifier_gate_checks"] < expected_classifier_gate_minimum ||
          value["compat_classifier_gate_rejections"] < expected_classifier_gate_minimum ||
          value["compat_classifier_slow_path_admissions"] != "0" ||
          value["compat_classifier_slow_path_bytes"] != "0" ||
          value["commit_validation_reresolve_nanos"] != "0") bad = 1
      if (value["compat_classifier_gate_rejections"] > value["compat_classifier_gate_checks"] ||
          value["compat_classifier_slow_path_admissions"] > value["compat_classifier_gate_checks"] ||
          value["compat_classifier_gate_checks"] - value["compat_classifier_gate_rejections"] != value["compat_classifier_slow_path_admissions"]) bad = 1
      fua_depth_samples = value["fua_in_flight_depth_1"] + value["fua_in_flight_depth_2"] + value["fua_in_flight_depth_3_to_4"] + value["fua_in_flight_depth_5_to_8"] + value["fua_in_flight_depth_9_to_16"] + value["fua_in_flight_depth_17_to_32"] + value["fua_in_flight_depth_33_plus"]
      controller_actions = value["fua_controller_sustained_actions"] + value["fua_controller_pending_probe_cover_actions"] + value["fua_controller_qd1_samples"] + value["fua_controller_unfragmented_actions"] + value["fua_controller_protocol_fallback_actions"]
      controller_samples = value["fua_controller_fast_samples"] + value["fua_controller_nonfast_samples"] + value["fua_controller_stale_qd1_samples"] + value["fua_controller_unavailable_qd1_samples"] + value["fua_controller_abandoned_qd1_samples"] + value["fua_controller_pending_qd1_samples"]
      if (expected_profile == "development") {
        if (value["fua_logical_groups"] != "0" || value["fua_logical_payload_bytes"] != "0" ||
            value["fua_single_frame_padded_baseline_bytes"] != "0" ||
            value["fua_publish_turn_wait_nanos"] != "0" || value["fua_publish_turn_wait_groups"] != "0" ||
            value["fua_published_frames"] != "0" || value["fua_fenced_frames"] != "0" ||
            value["fua_fence_failures"] != "0" || value["fua_payload_bytes"] != "0" ||
            value["fua_padded_bytes"] != "0" || value["fua_stage_copy_nanos"] != "0" ||
            value["fua_stage_copy_frames"] != "0" || value["fua_publish_to_claim_nanos"] != "0" ||
            value["fua_publish_to_claim_frames"] != "0" || value["fua_claim_to_write_done_nanos"] != "0" ||
            value["fua_claim_to_write_done_frames"] != "0" ||
            value["fua_write_done_to_contiguous_cut_nanos"] != "0" ||
            value["fua_write_done_to_contiguous_cut_frames"] != "0" ||
            value["fua_contiguous_cut_events"] != "0" ||
            value["fua_contiguous_cut_advanced_frames"] != "0" ||
            value["fua_contiguous_cut_advance_max_frames"] != "0" ||
            value["fua_waiter_cut_to_observe_nanos"] != "0" ||
            value["fua_waiter_cut_to_observe_count"] != "0" ||
            value["fua_in_flight_depth_max"] != "0" || fua_depth_samples != 0 ||
            controller_actions != 0 || controller_samples != 0 ||
            value["fua_controller_qd1_sparse_actions"] != "0" ||
            value["fua_controller_qd1_verify_actions"] != "0" ||
            value["fua_controller_qd1_fast_actions"] != "0" ||
            value["fua_controller_phase"] != "0" ||
            value["fua_controller_verify_fast_streak"] != "0" ||
            value["fua_controller_sustained_remaining"] != "0" ||
            value["fua_controller_generation"] != "0" ||
            value["fua_controller_fast_in_flight"] != "0" ||
            value["fua_controller_fast_in_flight_max"] != "0" ||
            value["fua_controller_generation_exhausted"] != "0" ||
            value["fua_controller_ordinal_exhausted"] != "0" ||
            value["fua_controller_action_reconciliation"] != "0" ||
            value["fua_controller_sample_reconciliation"] != "0") bad = 1
      }
      if (expected_profile == "durable") {
        if (value["fua_logical_groups"] == "0" || value["fua_logical_payload_bytes"] == "0" ||
            value["fua_single_frame_padded_baseline_bytes"] == "0" ||
            value["fua_publish_turn_wait_groups"] != value["fua_logical_groups"] ||
            value["fua_published_frames"] == "0" ||
            value["fua_published_frames"] < value["fua_logical_groups"] ||
            value["fua_fenced_frames"] != value["fua_published_frames"] ||
            value["fua_fence_failures"] != "0" ||
            value["fua_logical_payload_bytes"] != value["fua_payload_bytes"] ||
            value["fua_padded_bytes"] > value["fua_single_frame_padded_baseline_bytes"] * 2 ||
            value["fua_padded_bytes"] < value["fua_payload_bytes"] ||
            value["fua_stage_copy_frames"] != value["fua_published_frames"] ||
            value["fua_publish_to_claim_frames"] != value["fua_published_frames"] ||
            value["fua_claim_to_write_done_frames"] != value["fua_published_frames"] ||
            value["fua_write_done_to_contiguous_cut_frames"] != value["fua_published_frames"] ||
            value["fua_contiguous_cut_advanced_frames"] != value["fua_published_frames"] ||
            value["fua_contiguous_cut_events"] == "0" ||
            value["fua_contiguous_cut_events"] > value["fua_published_frames"] ||
            value["fua_contiguous_cut_advance_max_frames"] == "0" ||
            value["fua_contiguous_cut_advance_max_frames"] > value["fua_published_frames"] ||
            value["fua_waiter_cut_to_observe_count"] == "0" ||
            value["fua_in_flight_depth_max"] < 16 ||
            value["fua_in_flight_depth_9_to_16"] == "0" ||
            fua_depth_samples != value["fua_published_frames"] ||
            controller_actions != value["fua_logical_groups"] ||
            value["fua_controller_sustained_actions"] == "0" ||
            value["fua_controller_qd1_samples"] != value["fua_controller_qd1_sparse_actions"] + value["fua_controller_qd1_verify_actions"] + value["fua_controller_qd1_fast_actions"] ||
            value["fua_controller_action_reconciliation"] != "1" ||
            value["fua_controller_sample_reconciliation"] != "1" ||
            value["fua_controller_pending_qd1_samples"] != "0" ||
            value["fua_controller_fast_in_flight"] != "0" ||
            value["fua_controller_protocol_faults"] != "0" ||
            value["fua_controller_protocol_fallback_actions"] != "0" ||
            value["fua_controller_unavailable_qd1_samples"] != "0" ||
            value["fua_controller_abandoned_qd1_samples"] != "0" ||
            value["fua_controller_generation_exhausted"] != "0" ||
            value["fua_controller_ordinal_exhausted"] != "0") bad = 1
      }
      if (value["successful_insert_end_to_end_service_nanos"] == "0") bad = 1
      if (value["wave_count"] == "0" || value["wave_item_count"] == "0" ||
          value["peak_host_statement_bytes"] == "0" ||
          value["peak_device_statement_bytes_estimate"] == "0") bad = 1
      # INSERT-001 is two NULL-free int4 columns in fixed 1,000-row statements with the normal
      # implicit 4M-row target. The empty bootstrap descriptor is the sole non-rollover shard;
      # all other geometry is exact for 1M, 8M, and 48M qualification accounts.
      expected_rollovers = int((expected_rows + 3999999) / 4000000)
      expected_capacity_total = expected_rollovers * 4000000
      expected_shards = expected_rollovers + 1
      expected_persistent_allocations = expected_rollovers * 3
      expected_sidecar_fill_bytes = expected_rollovers * 64000000
      expected_live_h2d_bytes = expected_rollovers * 24008
      if (value["rollover_count"] != expected_rollovers ||
          value["rollover_capacity_rows_total"] != expected_capacity_total ||
          value["rollover_capacity_rows_max"] != "4000000" ||
          value["current_shard_count"] != expected_shards ||
          value["peak_shard_count"] != expected_shards ||
          value["persistent_allocation_count"] != expected_persistent_allocations ||
          value["named_index_shard_visits"] != "0" ||
          value["descriptor_clone_visit_count"] < expected_insert_statements ||
          value["budget_scan_entries"] == "0" ||
          value["capacity_fit_evaluation_count"] < expected_rollovers ||
          value["sidecar_fill_bytes"] != expected_sidecar_fill_bytes ||
          value["live_h2d_bytes"] != expected_live_h2d_bytes) bad = 1
      record = $0
    }
    END {
      if (records != 1 || bad) exit 1
      print record
    }
  ' "$stderr_log"
}

# Parse one client completion record and emit selected fields as one tab-separated line.  All
# semantic fields are compared to the already verified workload and requested trial, so a result
# cannot be accidentally associated with another source, profile, backend, or statement shape.
parse_client_record() {
  # parse_client_record <stdout-log> <backend> <profile> <trial>
  local stdout_log="$1"
  local expected_backend="$2"
  local expected_profile="$3"
  local expected_trial="$4"
  awk \
    -v expected_backend="$expected_backend" \
    -v expected_profile="$expected_profile" \
    -v expected_trial="$expected_trial" \
    -v expected_rows="$rows" \
    -v expected_chunk="$INSERT_CHUNK" \
    -v expected_insert_statements="$manifest_insert_statements" \
    -v expected_source_bytes="$manifest_source_bytes" \
    -v expected_source_sha256="$source_file_sha256" \
    -v expected_stream_sha256="$manifest_statement_stream_sha256" '
    function canonical_uint(value) {
      return value == "0" || value ~ /^[1-9][0-9]*$/
    }
    function digest(value) {
      return length(value) == 64 && value ~ /^[0-9a-f]+$/
    }
    function decimal_3(value) {
      return value ~ /^(0|[1-9][0-9]*)\.[0-9][0-9][0-9]$/
    }
    $1 == "insert_workload_replay_status=complete" {
      records += 1
      if (NF != 24) bad = 1
      delete value
      delete seen
      for (field = 1; field <= NF; field += 1) {
        piece_count = split($field, pieces, "=")
        if (piece_count != 2 || pieces[1] == "" || pieces[2] == "") {
          bad = 1
          continue
        }
        if (++seen[pieces[1]] != 1) bad = 1
        value[pieces[1]] = pieces[2]
      }
      required["insert_workload_replay_status"] = 1
      required["backend"] = 1
      required["profile"] = 1
      required["trial"] = 1
      required["manifest_version"] = 1
      required["workload"] = 1
      required["rows"] = 1
      required["chunk"] = 1
      required["insert_statements"] = 1
      required["source_bytes"] = 1
      required["source_sha256"] = 1
      required["statement_stream_sha256"] = 1
      required["source_verification_ms"] = 1
      required["create_roundtrip_us"] = 1
      required["insert_roundtrip_sum_us"] = 1
      required["insert_roundtrip_p50_us"] = 1
      required["insert_roundtrip_p99_us"] = 1
      required["insert_roundtrip_p999_us"] = 1
      required["insert_roundtrip_max_us"] = 1
      required["insert_load_wall_ms"] = 1
      required["validation_roundtrip_sum_us"] = 1
      required["end_to_end_client_wall_ms"] = 1
      required["rows_per_second"] = 1
      required["validation"] = 1
      for (key in required) if (seen[key] != 1) bad = 1
      for (key in seen) if (!(key in required)) bad = 1
      if (value["insert_workload_replay_status"] != "complete" ||
          value["backend"] != expected_backend ||
          value["profile"] != expected_profile ||
          value["trial"] != expected_trial ||
          value["manifest_version"] != "1" ||
          value["workload"] != "insert001-accounts-int4-v1" ||
          value["rows"] != expected_rows || value["chunk"] != expected_chunk ||
          value["insert_statements"] != expected_insert_statements ||
          value["source_bytes"] != expected_source_bytes ||
          value["source_sha256"] != expected_source_sha256 ||
          value["statement_stream_sha256"] != expected_stream_sha256 ||
          value["validation"] != "count_id_sum_balance_sum") bad = 1
      if (!digest(value["source_sha256"]) || !digest(value["statement_stream_sha256"]) ||
          !decimal_3(value["rows_per_second"])) bad = 1
      numeric["source_verification_ms"] = 1
      numeric["create_roundtrip_us"] = 1
      numeric["insert_roundtrip_sum_us"] = 1
      numeric["insert_roundtrip_p50_us"] = 1
      numeric["insert_roundtrip_p99_us"] = 1
      numeric["insert_roundtrip_p999_us"] = 1
      numeric["insert_roundtrip_max_us"] = 1
      numeric["insert_load_wall_ms"] = 1
      numeric["validation_roundtrip_sum_us"] = 1
      numeric["end_to_end_client_wall_ms"] = 1
      for (key in numeric) if (!canonical_uint(value[key])) bad = 1
      parsed_insert_wall = value["insert_load_wall_ms"]
      parsed_rows_per_second = value["rows_per_second"]
      parsed_end_to_end_wall = value["end_to_end_client_wall_ms"]
    }
    END {
      if (records != 1 || bad) exit 1
      print parsed_insert_wall "\t" parsed_rows_per_second "\t" parsed_end_to_end_wall
    }
  ' "$stdout_log"
}

append_result_once() {
  # append_result_once <results-tsv> <profile> <backend> <trial> <client-stdout>
  local results_file="$1"
  local result_profile="$2"
  local backend="$3"
  local trial="$4"
  local client_stdout="$5"
  local parsed
  local insert_wall_ms
  local rows_per_second
  local end_to_end_ms
  if grep -Fqx "${result_profile}"$'\t'"${backend}"$'\t'"${trial}" \
    <(cut -f 1-3 -- "$results_file") 2>/dev/null; then
    return 1
  fi
  parsed="$(parse_client_record "$client_stdout" "$backend" "$result_profile" "$trial")" || return 1
  IFS=$'\t' read -r insert_wall_ms rows_per_second end_to_end_ms <<<"$parsed"
  [[ -n "$insert_wall_ms" && -n "$rows_per_second" && -n "$end_to_end_ms" ]] || return 1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$result_profile" "$backend" "$trial" "$insert_wall_ms" "$rows_per_second" "$end_to_end_ms" \
    >>"$results_file"
}

validate_result_set() {
  # validate_result_set <results-tsv> <profile> <backend>
  local results_file="$1"
  local result_profile="$2"
  local backend="$3"
  awk -F '\t' -v p="$result_profile" -v b="$backend" '
    function canonical_uint(value) { return value == "0" || value ~ /^[1-9][0-9]*$/ }
    function decimal_3(value) { return value ~ /^(0|[1-9][0-9]*)\.[0-9][0-9][0-9]$/ }
    $1 == p && $2 == b {
      matched += 1
      if (NF != 6 || !canonical_uint($3) || $3 == "0" || !canonical_uint($4) ||
          !decimal_3($5) || !canonical_uint($6) || ++seen[$3] != 1) bad = 1
    }
    END { exit (matched == 3 && !bad) ? 0 : 1 }
  ' "$results_file"
}

median_metric() {
  # median_metric <results-tsv> <profile> <backend> <field-number>
  local results_file="$1"
  local result_profile="$2"
  local backend="$3"
  local field_number="$4"
  validate_result_set "$results_file" "$result_profile" "$backend" || return 1
  awk -F '\t' -v p="$result_profile" -v b="$backend" -v field="$field_number" \
    '$1 == p && $2 == b { print $field }' "$results_file" |
    LC_ALL=C sort -n |
    sed -n '2p'
}

active_gpu_pid=""
active_gpu_trial_dir=""
active_pg_container=""
active_pg_trial_dir=""
cleanup_running=0

stop_gpu_server() {
  local pid="$active_gpu_pid"
  local trial_dir="$active_gpu_trial_dir"
  active_gpu_pid=""
  active_gpu_trial_dir=""
  [[ -n "$pid" && -n "$trial_dir" ]] || return 0
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -TERM "$pid" >/dev/null 2>&1 || true
  fi
  local waited=0
  while kill -0 "$pid" >/dev/null 2>&1 && [[ "$waited" -lt 20 ]]; do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
  wait "$pid"
  local status=$?
  write_exit_status "$trial_dir/gpu-server.exit-status" "$status"
}

stop_postgres_server() {
  local container="$active_pg_container"
  local trial_dir="$active_pg_trial_dir"
  active_pg_container=""
  active_pg_trial_dir=""
  [[ -n "$container" && -n "$trial_dir" ]] || return 0
  if docker container inspect "$container" >/dev/null 2>&1; then
    capture_command "$trial_dir/postgres-stop" docker stop --time 10 "$container" || true
    capture_command "$trial_dir/postgres-inspect" docker inspect "$container" || true
    capture_command "$trial_dir/postgres-server" docker logs "$container" || true
    capture_command "$trial_dir/postgres-remove" docker rm --force "$container" || true
  else
    printf 'container was not present during cleanup: %s\n' "$container" >"$trial_dir/postgres-cleanup.stdout"
    : >"$trial_dir/postgres-cleanup.stderr"
    write_exit_status "$trial_dir/postgres-cleanup.exit-status" "1"
  fi
}

cleanup_active_services() {
  [[ "$cleanup_running" -eq 0 ]] || return 0
  cleanup_running=1
  stop_gpu_server || true
  stop_postgres_server || true
}

trap cleanup_active_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_tcp_listener() {
  local port="$1"
  local attempts=0
  while [[ "$attempts" -lt 60 ]]; do
    if timeout 1 bash -c 'exec 3<>"/dev/tcp/$1/$2"' -- 127.0.0.1 "$port" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    attempts=$((attempts + 1))
  done
  return 1
}

wait_for_postgres() {
  local container="$1"
  local trial_dir="$2"
  local attempts=0
  local status
  : >"$trial_dir/postgres-ready.stdout"
  : >"$trial_dir/postgres-ready.stderr"
  while [[ "$attempts" -lt 90 ]]; do
    docker exec "$container" pg_isready -U postgres -d postgres \
      >>"$trial_dir/postgres-ready.stdout" 2>>"$trial_dir/postgres-ready.stderr"
    status=$?
    if [[ "$status" -eq 0 ]]; then
      write_exit_status "$trial_dir/postgres-ready.exit-status" "0"
      return 0
    fi
    sleep 1
    attempts=$((attempts + 1))
  done
  write_exit_status "$trial_dir/postgres-ready.exit-status" "1"
  return 1
}

port_base="${INSERT_QUALIFICATION_PORT_BASE:-56100}"
require_positive_uint "$port_base" "INSERT_QUALIFICATION_PORT_BASE"
((10#$port_base >= 1024 && 10#$port_base <= 65000)) || die "INSERT_QUALIFICATION_PORT_BASE must be in 1024..65000"

trial_port() {
  # profile slot 0/1, backend slot 0/1, trial 1..3.  The names and ports are deterministic in
  # the artifact, avoiding accidental reuse of an unrelated service during a failed cleanup.
  local profile_slot="$1"
  local backend_slot="$2"
  local trial="$3"
  local port=$((10#$port_base + profile_slot * 20 + backend_slot * 5 + trial))
  ((port <= 65535)) || return 1
  printf '%s\n' "$port"
}

start_gpu_server() {
  # start_gpu_server <profile> <trial> <port> <trial-dir>
  local selected_profile="$1"
  local trial="$2"
  local port="$3"
  local trial_dir="$4"
  local wal_path="$trial_dir/gpu-wal/insert-001.wal"
  mkdir -p -- "$trial_dir/gpu-wal"
  if [[ "$selected_profile" == "development" ]]; then
    env -u GPU_DB_WAL_SEGMENT "$server_binary" --listen "127.0.0.1:${port}" \
      >"$trial_dir/gpu-server.stdout" 2>"$trial_dir/gpu-server.stderr" &
  else
    env GPU_DB_WAL_SEGMENT="$wal_path" "$server_binary" --listen "127.0.0.1:${port}" \
      >"$trial_dir/gpu-server.stdout" 2>"$trial_dir/gpu-server.stderr" &
  fi
  active_gpu_pid="$!"
  active_gpu_trial_dir="$trial_dir"
  local wal_record="<unset>"
  if [[ "$selected_profile" == "durable" ]]; then
    wal_record="$wal_path"
  fi
  printf 'profile=%s\nlisten=127.0.0.1:%s\nwal_segment=%s\n' \
    "$selected_profile" "$port" "$wal_record" \
    >"$trial_dir/gpu-runtime-config.txt"
  if ! wait_for_tcp_listener "$port"; then
    return 1
  fi
  kill -0 "$active_gpu_pid" >/dev/null 2>&1 || return 1
  grep -Fqx "gpu-db-engine-server (facade-backed) listening on 127.0.0.1:${port}" \
    "$trial_dir/gpu-server.stderr" || return 1
  if [[ "$selected_profile" == "development" ]]; then
    ! grep -Fq 'durable WAL enabled' "$trial_dir/gpu-server.stderr" || return 1
  else
    grep -Fqx "gpu-db-engine-server: durable WAL enabled (GPU_DB_WAL_SEGMENT=${wal_path})" \
      "$trial_dir/gpu-server.stderr" || return 1
  fi
  ! grep -Fq 'insert_probe_session_delta=' "$trial_dir/gpu-server.stderr" || return 1
}

postgres_runtime_settings_match() {
  # postgres_runtime_settings_match <profile> <settings-output>
  local selected_profile="$1"
  local settings_output="$2"
  local expected
  case "$selected_profile" in
    development) expected="${POSTGRES_VERSION_NUM}|${POSTGRES_VERSION}|off|off|off|${POSTGRES_WAL_SYNC_METHOD}" ;;
    durable) expected="${POSTGRES_VERSION_NUM}|${POSTGRES_VERSION}|on|on|on|${POSTGRES_WAL_SYNC_METHOD}" ;;
    *) return 1 ;;
  esac
  [[ "$(tr -d '\r\n' <"$settings_output")" == "$expected" ]]
}

start_postgres_server() {
  # start_postgres_server <profile> <trial> <port> <container> <trial-dir>
  local selected_profile="$1"
  local trial="$2"
  local port="$3"
  local container="$4"
  local trial_dir="$5"
  local data_dir="$trial_dir/postgres-data"
  local -a postgres_args=(postgres)
  mkdir -p -- "$data_dir"
  case "$selected_profile" in
    development)
      postgres_args+=( -c fsync=off -c synchronous_commit=off -c full_page_writes=off )
      ;;
    durable)
      postgres_args+=( -c fsync=on -c synchronous_commit=on -c full_page_writes=on )
      ;;
    *) return 1 ;;
  esac
  capture_command "$trial_dir/postgres-launch" \
    docker run --detach --name "$container" \
    --mount "type=bind,source=${data_dir},target=/var/lib/postgresql/data" \
    --publish "127.0.0.1:${port}:5432" \
    --env POSTGRES_HOST_AUTH_METHOD=trust \
    "$postgres_image" "${postgres_args[@]}" || return 1
  active_pg_container="$container"
  active_pg_trial_dir="$trial_dir"
  if ! wait_for_postgres "$container" "$trial_dir"; then
    return 1
  fi
  capture_command "$trial_dir/postgres-runtime-version" \
    docker exec "$container" psql -X -A -t -v ON_ERROR_STOP=1 -U postgres -d postgres \
    -c "SHOW server_version;" || return 1
  capture_command "$trial_dir/postgres-runtime-settings" \
    docker exec "$container" psql -X -A -t -F '|' -v ON_ERROR_STOP=1 -U postgres -d postgres \
    -c "SELECT current_setting('server_version_num'), split_part(current_setting('server_version'), ' ', 1), current_setting('fsync'), current_setting('synchronous_commit'), current_setting('full_page_writes'), current_setting('wal_sync_method');" || return 1
  postgres_runtime_settings_match "$selected_profile" "$trial_dir/postgres-runtime-settings.stdout"
}

run_client_trial() {
  # run_client_trial <backend> <profile> <trial> <port> <trial-dir>
  local backend="$1"
  local selected_profile="$2"
  local trial="$3"
  local port="$4"
  local trial_dir="$5"
  local connection="postgresql://postgres@127.0.0.1:${port}/postgres?sslmode=disable"
  local timeout_seconds="${INSERT_QUALIFICATION_CLIENT_TIMEOUT:-$DEFAULT_CLIENT_TIMEOUT_SECONDS}"
  require_positive_uint "$timeout_seconds" "INSERT_QUALIFICATION_CLIENT_TIMEOUT"
  capture_command "$trial_dir/client" \
    timeout "$timeout_seconds" "$client_binary" run \
    --source "$source_path" --manifest "$manifest_path" \
    --connection "$connection" --backend "$backend" --profile "$selected_profile" --trial "$trial"
}

run_one_trial() {
  # run_one_trial <profile> <profile-slot> <backend> <backend-slot> <trial> <results-tsv>
  local selected_profile="$1"
  local profile_slot="$2"
  local backend="$3"
  local backend_slot="$4"
  local trial="$5"
  local results_file="$6"
  local port
  local trial_dir
  local container
  port="$(trial_port "$profile_slot" "$backend_slot" "$trial")" || return 1
  trial_dir="$artifact_dir/trials/${selected_profile}-${backend}-trial-${trial}"
  container="gpu-db-insertq-${run_id}-${selected_profile}-${backend}-${trial}"
  mkdir -p -- "$trial_dir"
  printf 'profile=%s\nbackend=%s\ntrial=%s\nport=%s\ncontainer=%s\n' \
    "$selected_profile" "$backend" "$trial" "$port" "$container" >"$trial_dir/trial-config.txt"
  if ! verify_frozen_inputs; then
    printf 'frozen-input verification failed before service start\n' >"$trial_dir/qualification-failure.txt"
    return 1
  fi
  printf 'source_sha256=%s\nmanifest_sha256=%s\nserver_sha256=%s\nclient_sha256=%s\n' \
    "$source_file_sha256" "$manifest_file_sha256" "$server_binary_sha256" "$client_binary_sha256" \
    >"$trial_dir/reverified-identities.txt"
  case "$backend" in
    gpu)
      start_gpu_server "$selected_profile" "$trial" "$port" "$trial_dir" || return 1
      run_client_trial "$backend" "$selected_profile" "$trial" "$port" "$trial_dir"
      local client_status=$?
      stop_gpu_server || true
      [[ "$client_status" -eq 0 ]] || return 1
      parse_gpu_insert_probe_record "$trial_dir/gpu-server.stderr" "$selected_profile" "$trial" \
        >"$trial_dir/gpu-insert-probe.record" || return 1
      ;;
    postgresql)
      start_postgres_server "$selected_profile" "$trial" "$port" "$container" "$trial_dir" || return 1
      run_client_trial "$backend" "$selected_profile" "$trial" "$port" "$trial_dir"
      local client_status=$?
      stop_postgres_server || true
      [[ "$client_status" -eq 0 ]] || return 1
      ! grep -Fq 'insert_probe_session_delta=' "$trial_dir/client.stdout" || return 1
      ;;
    *) return 1 ;;
  esac
  append_result_once "$results_file" "$selected_profile" "$backend" "$trial" "$trial_dir/client.stdout"
}

emit_profile_medians() {
  # emit_profile_medians <profile> <results-tsv>
  local selected_profile="$1"
  local results_file="$2"
  local gpu_insert_median
  local pg_insert_median
  local gpu_rps_median
  local pg_rps_median
  local frozen_pg_rps_floor
  local floor_result
  validate_result_set "$results_file" "$selected_profile" gpu || return 1
  validate_result_set "$results_file" "$selected_profile" postgresql || return 1
  gpu_insert_median="$(median_metric "$results_file" "$selected_profile" gpu 4)" || return 1
  pg_insert_median="$(median_metric "$results_file" "$selected_profile" postgresql 4)" || return 1
  gpu_rps_median="$(median_metric "$results_file" "$selected_profile" gpu 5)" || return 1
  pg_rps_median="$(median_metric "$results_file" "$selected_profile" postgresql 5)" || return 1
  if frozen_pg_rps_floor="$(frozen_postgres_rows_per_second_floor "$selected_profile" "$rows")"; then
    if awk -v actual="$gpu_rps_median" -v floor="$frozen_pg_rps_floor" \
      'BEGIN { exit !((actual + 0) >= (floor + 0)) }'; then
      floor_result="pass"
    else
      floor_result="fail"
    fi
  else
    frozen_pg_rps_floor="<not-yet-frozen>"
    floor_result="not-enforced"
  fi
  printf '%s\n' \
    "insert_qualification_median_status=complete profile=${selected_profile} backend=gpu rows=${rows} chunk=${INSERT_CHUNK} trials=3 metric=insert_load_wall_ms median=${gpu_insert_median}" \
    "insert_qualification_median_status=complete profile=${selected_profile} backend=gpu rows=${rows} chunk=${INSERT_CHUNK} trials=3 metric=rows_per_second median=${gpu_rps_median}" \
    "insert_qualification_postgresql_median_status=observed profile=${selected_profile} rows=${rows} chunk=${INSERT_CHUNK} trials=3 metric=insert_load_wall_ms median=${pg_insert_median}" \
    "insert_qualification_postgresql_median_status=observed profile=${selected_profile} rows=${rows} chunk=${INSERT_CHUNK} trials=3 metric=rows_per_second median=${pg_rps_median}" \
    "insert_qualification_frozen_floor_status=${floor_result} profile=${selected_profile} rows=${rows} chunk=${INSERT_CHUNK} metric=rows_per_second gpu_median=${gpu_rps_median} frozen_postgresql_floor=${frozen_pg_rps_floor}" \
    | tee -a "$artifact_dir/qualification-summary.txt"
  [[ "$floor_result" != "fail" ]]
}

run_self_check() {
  local scratch
  local failures=0
  local source_digest="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  local stream_digest="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
  scratch="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-insert-qualification-self-check.XXXXXX")" || return 1
  local cleanup_self_check=0
  cleanup_scratch() {
    [[ "$cleanup_self_check" -eq 0 ]] || return 0
    cleanup_self_check=1
    case "$scratch" in
      "${TMPDIR:-/tmp}"/gpu-db-insert-qualification-self-check.??????)
        rm -rf -- "$scratch"
        ;;
      *)
        echo "refusing to remove unexpected self-check directory: $scratch" >&2
        return 1
        ;;
    esac
  }
  printf '%s' \
    $'CREATE TABLE accounts (id int4, balance int4);\nINSERT INTO accounts (id, balance) VALUES (0, 0);\nINSERT INTO accounts (id, balance) VALUES (1, 7);\n' \
    >"$scratch/fixed-terminator-stream.sql"
  local fixture_source_bytes fixture_canonical_request_bytes fixture_stripped_terminator_bytes
  read -r fixture_source_bytes fixture_canonical_request_bytes fixture_stripped_terminator_bytes \
    <<<"$(derive_insert_request_identity_bytes "$scratch/fixed-terminator-stream.sql" 2)" ||
    failures=$((failures + 1))
  [[ "$fixture_stripped_terminator_bytes" == "4" &&
    $((fixture_canonical_request_bytes + fixture_stripped_terminator_bytes)) -eq fixture_source_bytes ]] ||
    failures=$((failures + 1))
  printf '%s' \
    $'CREATE TABLE accounts (id int4, balance int4);\nINSERT INTO accounts (id, balance) VALUES (0, 0);\r\n' \
    >"$scratch/noncanonical-terminator-stream.sql"
  ! derive_insert_request_identity_bytes "$scratch/noncanonical-terminator-stream.sql" 1 >/dev/null ||
    failures=$((failures + 1))
  local valid_line
  valid_line="insert_workload_replay_status=complete backend=gpu profile=development trial=1 manifest_version=1 workload=${INSERT_WORKLOAD_ID} rows=1000000 chunk=1000 insert_statements=1000 source_bytes=12345 source_sha256=${source_digest} statement_stream_sha256=${stream_digest} source_verification_ms=1 create_roundtrip_us=2 insert_roundtrip_sum_us=3 insert_roundtrip_p50_us=4 insert_roundtrip_p99_us=5 insert_roundtrip_p999_us=6 insert_roundtrip_max_us=7 insert_load_wall_ms=8 validation_roundtrip_sum_us=9 end_to_end_client_wall_ms=10 rows_per_second=125000.000 validation=count_id_sum_balance_sum"
  manifest_insert_statements="1000"
  manifest_source_bytes="12345"
  insert_source_bytes="12300"
  insert_canonical_request_bytes="10300"
  insert_stripped_terminator_bytes="2000"
  manifest_statement_stream_sha256="$stream_digest"
  source_file_sha256="$source_digest"
  rows="1000000"
  [[ "$(checked_classifier_gate_minimum "1000")" == "4000" ]] || failures=$((failures + 1))
  ! checked_classifier_gate_minimum "" >/dev/null || failures=$((failures + 1))
  ! checked_classifier_gate_minimum "01" >/dev/null || failures=$((failures + 1))
  ! checked_classifier_gate_minimum "2251799813685248" >/dev/null || failures=$((failures + 1))
  printf '%s\n' "$valid_line" >"$scratch/valid.log"
  parse_client_record "$scratch/valid.log" gpu development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n%s\n' "$valid_line" "$valid_line" >"$scratch/duplicate.log"
  ! parse_client_record "$scratch/duplicate.log" gpu development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${valid_line/backend=gpu/backend=postgresql}" >"$scratch/backend-mismatch.log"
  ! parse_client_record "$scratch/backend-mismatch.log" gpu development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${valid_line/source_sha256=${source_digest}/source_sha256=not-a-digest}" >"$scratch/hash-mismatch.log"
  ! parse_client_record "$scratch/hash-mismatch.log" gpu development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${valid_line/rows_per_second=125000.000/rows_per_second=NaN}" >"$scratch/malformed.log"
  ! parse_client_record "$scratch/malformed.log" gpu development 1 >/dev/null || failures=$((failures + 1))
  local probe_line
  probe_line="insert_probe_session_delta=complete version=7 attribution=engine_local_plus_process_global_classifier_counters_single_active_workload_session_required successful_insert_statements=1000 successful_insert_rows=1000000 fixed_insert_typed_commits=1000 fixed_insert_legacy_fallbacks=0 fixed_insert_retryable_declines=0 fixed_insert_legacy_commit_validation_reresolves=0 legacy_insert_delta_builds=0 predicted_row_keys_materialized=0 direct_fixed_insert_carriers=1000 raw_request_digest_derivations=1000 raw_request_digest_derivation_bytes=10300 compat_classifier_gate_checks=4000 compat_classifier_gate_rejections=4000 compat_classifier_slow_path_admissions=0 compat_classifier_slow_path_bytes=0 successful_insert_source_bytes=12300 successful_insert_end_to_end_service_nanos=100 facade_parse_bind_nanos=1 engine_authorization_catalog_admission_nanos=2 offlock_coercion_default_constraint_prepare_nanos=3 commit_validation_reresolve_nanos=0 canonical_wal_encode_append_claim_nanos=5 durability_wait_nanos=6 durability_begin_group_flush_nanos=1 durability_job_wait_nanos=2 device_validate_nanos=7 device_h2d_append_index_apply_nanos=8 publication_status_ack_nanos=9 wave_count=1 wave_item_count=1000 peak_host_statement_bytes=12 peak_device_statement_bytes_estimate=8 rollover_count=1 rollover_capacity_rows_total=4000000 rollover_capacity_rows_max=4000000 current_shard_count=2 peak_shard_count=2 persistent_allocation_count=3 descriptor_clone_visit_count=1000 budget_scan_entries=5 capacity_fit_evaluation_count=23 sidecar_fill_bytes=64000000 live_h2d_bytes=24008 named_index_shard_visits=0 unattributed_or_concurrent_overlap_nanos=55 durability_backend=memory intent_lane_count=0 synchronous_commit_gate=strict_rpo0 auto_admit_on_commit=true binary_wal_records_enabled=true device_authoritative_commits=1000 active_workload_sessions_at_seal=1 overlap_observed=false"
  probe_line+=" fua_logical_groups=0 fua_logical_payload_bytes=0 fua_single_frame_padded_baseline_bytes=0 fua_publish_turn_wait_nanos=0 fua_publish_turn_wait_groups=0 fua_published_frames=0 fua_fenced_frames=0 fua_fence_failures=0 fua_payload_bytes=0 fua_padded_bytes=0 fua_stage_copy_nanos=0 fua_stage_copy_frames=0 fua_publish_to_claim_nanos=0 fua_publish_to_claim_frames=0 fua_claim_to_write_done_nanos=0 fua_claim_to_write_done_frames=0 fua_write_done_to_contiguous_cut_nanos=0 fua_write_done_to_contiguous_cut_frames=0 fua_contiguous_cut_events=0 fua_contiguous_cut_advanced_frames=0 fua_contiguous_cut_advance_max_frames=0 fua_waiter_cut_to_observe_nanos=0 fua_waiter_cut_to_observe_count=0 fua_in_flight_depth_max=0 fua_in_flight_depth_1=0 fua_in_flight_depth_2=0 fua_in_flight_depth_3_to_4=0 fua_in_flight_depth_5_to_8=0 fua_in_flight_depth_9_to_16=0 fua_in_flight_depth_17_to_32=0 fua_in_flight_depth_33_plus=0 fua_fence_lanes=0"
  probe_line+=" fua_controller_sustained_actions=0 fua_controller_pending_probe_cover_actions=0 fua_controller_qd1_samples=0 fua_controller_qd1_sparse_actions=0 fua_controller_qd1_verify_actions=0 fua_controller_qd1_fast_actions=0 fua_controller_unfragmented_actions=0 fua_controller_pool_too_narrow=0 fua_controller_empty_chunk=0 fua_controller_insufficient_free_slots=0 fua_controller_natural_depth=0 fua_controller_segment_boundary=0 fua_controller_amplification_cap=0 fua_controller_fast_samples=0 fua_controller_nonfast_samples=0 fua_controller_transitions_to_verify=0 fua_controller_transitions_to_fast=0 fua_controller_transitions_to_sustained=0 fua_controller_stale_qd1_samples=0 fua_controller_unavailable_qd1_samples=0 fua_controller_abandoned_qd1_samples=0 fua_controller_protocol_faults=0 fua_controller_protocol_fallback_actions=0 fua_controller_phase=0 fua_controller_verify_fast_streak=0 fua_controller_sustained_remaining=0 fua_controller_generation=0 fua_controller_pending_qd1_samples=0 fua_controller_fast_in_flight=0 fua_controller_fast_in_flight_max=0 fua_controller_generation_exhausted=0 fua_controller_ordinal_exhausted=0 fua_controller_action_reconciliation=0 fua_controller_sample_reconciliation=0"
  printf '%s\n' "$probe_line" >"$scratch/probe-valid.log"
  parse_gpu_insert_probe_record "$scratch/probe-valid.log" development 1 >/dev/null || failures=$((failures + 1))
  local original_manifest_insert_statements="$manifest_insert_statements"
  local original_insert_source_bytes="$insert_source_bytes"
  local original_insert_canonical_request_bytes="$insert_canonical_request_bytes"
  local original_insert_stripped_terminator_bytes="$insert_stripped_terminator_bytes"
  local scaled_probe_line="$probe_line"
  scaled_probe_line="${scaled_probe_line/successful_insert_statements=1000/successful_insert_statements=8000}"
  scaled_probe_line="${scaled_probe_line/fixed_insert_typed_commits=1000/fixed_insert_typed_commits=8000}"
  scaled_probe_line="${scaled_probe_line/direct_fixed_insert_carriers=1000/direct_fixed_insert_carriers=8000}"
  scaled_probe_line="${scaled_probe_line/raw_request_digest_derivations=1000/raw_request_digest_derivations=8000}"
  scaled_probe_line="${scaled_probe_line/raw_request_digest_derivation_bytes=10300/raw_request_digest_derivation_bytes=82400}"
  scaled_probe_line="${scaled_probe_line/compat_classifier_gate_checks=4000/compat_classifier_gate_checks=32000}"
  scaled_probe_line="${scaled_probe_line/compat_classifier_gate_rejections=4000/compat_classifier_gate_rejections=32000}"
  scaled_probe_line="${scaled_probe_line/device_authoritative_commits=1000/device_authoritative_commits=8000}"
  scaled_probe_line="${scaled_probe_line/descriptor_clone_visit_count=1000/descriptor_clone_visit_count=8000}"
  scaled_probe_line="${scaled_probe_line/successful_insert_source_bytes=12300/successful_insert_source_bytes=98400}"
  manifest_insert_statements="8000"
  insert_source_bytes="98400"
  insert_canonical_request_bytes="82400"
  insert_stripped_terminator_bytes="16000"
  printf '%s\n' "$scaled_probe_line" >"$scratch/probe-classifier-scaled-valid.log"
  parse_gpu_insert_probe_record "$scratch/probe-classifier-scaled-valid.log" development 1 >/dev/null || failures=$((failures + 1))
  local scaled_under_minimum="$scaled_probe_line"
  scaled_under_minimum="${scaled_under_minimum/compat_classifier_gate_checks=32000/compat_classifier_gate_checks=4000}"
  scaled_under_minimum="${scaled_under_minimum/compat_classifier_gate_rejections=32000/compat_classifier_gate_rejections=4000}"
  printf '%s\n' "$scaled_under_minimum" >"$scratch/probe-classifier-scaled-under-minimum.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-scaled-under-minimum.log" development 1 >/dev/null || failures=$((failures + 1))
  manifest_insert_statements=""
  ! parse_gpu_insert_probe_record "$scratch/probe-valid.log" development 1 >/dev/null || failures=$((failures + 1))
  manifest_insert_statements="2251799813685248"
  ! parse_gpu_insert_probe_record "$scratch/probe-valid.log" development 1 >/dev/null || failures=$((failures + 1))
  manifest_insert_statements="$original_manifest_insert_statements"
  insert_source_bytes="$original_insert_source_bytes"
  insert_canonical_request_bytes="$original_insert_canonical_request_bytes"
  insert_stripped_terminator_bytes="$original_insert_stripped_terminator_bytes"
  printf '%s\n%s\n' "$probe_line" "$probe_line" >"$scratch/probe-duplicate.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-duplicate.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/ compat_classifier_gate_checks=4000/}" >"$scratch/probe-classifier-missing.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-missing.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/ raw_request_digest_derivations=1000/}" >"$scratch/probe-digest-derivations-missing.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-digest-derivations-missing.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/raw_request_digest_derivations=1000/raw_request_digest_derivations=999}" >"$scratch/probe-digest-derivations-law.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-digest-derivations-law.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/raw_request_digest_derivation_bytes=10300/raw_request_digest_derivation_bytes=10299}" >"$scratch/probe-digest-bytes-law.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-digest-bytes-law.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "$probe_line compat_classifier_gate_rejections=4000" >"$scratch/probe-classifier-duplicate.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-duplicate.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "$probe_line compat_classifier_gate_unknown=0" >"$scratch/probe-classifier-unknown.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-unknown.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/compat_classifier_slow_path_admissions=0/compat_classifier_slow_path_admissions=00}" >"$scratch/probe-classifier-noncanonical.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-noncanonical.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/overlap_observed=false/overlap_observed=true}" >"$scratch/probe-overlap.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-overlap.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/version=7/version=1}" >"$scratch/probe-version-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-version-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/durability_backend=memory/durability_backend=serial}" >"$scratch/probe-profile-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-profile-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/intent_lane_count=0/intent_lane_count=1}" >"$scratch/probe-lane-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-lane-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  local durable_probe_line
  durable_probe_line="${probe_line/durability_backend=memory/durability_backend=fua}"
  durable_probe_line="${durable_probe_line/intent_lane_count=0/intent_lane_count=10}"
  durable_probe_line="${durable_probe_line/fua_logical_groups=0/fua_logical_groups=1}"
  durable_probe_line="${durable_probe_line/fua_logical_payload_bytes=0/fua_logical_payload_bytes=4096}"
  durable_probe_line="${durable_probe_line/fua_single_frame_padded_baseline_bytes=0/fua_single_frame_padded_baseline_bytes=4096}"
  durable_probe_line="${durable_probe_line/fua_publish_turn_wait_nanos=0/fua_publish_turn_wait_nanos=1}"
  durable_probe_line="${durable_probe_line/fua_publish_turn_wait_groups=0/fua_publish_turn_wait_groups=1}"
  durable_probe_line="${durable_probe_line/fua_published_frames=0/fua_published_frames=1}"
  durable_probe_line="${durable_probe_line/fua_fenced_frames=0/fua_fenced_frames=1}"
  durable_probe_line="${durable_probe_line/fua_payload_bytes=0/fua_payload_bytes=4096}"
  durable_probe_line="${durable_probe_line/fua_padded_bytes=0/fua_padded_bytes=4096}"
  durable_probe_line="${durable_probe_line/fua_stage_copy_nanos=0/fua_stage_copy_nanos=1}"
  durable_probe_line="${durable_probe_line/fua_stage_copy_frames=0/fua_stage_copy_frames=1}"
  durable_probe_line="${durable_probe_line/fua_publish_to_claim_nanos=0/fua_publish_to_claim_nanos=1}"
  durable_probe_line="${durable_probe_line/fua_publish_to_claim_frames=0/fua_publish_to_claim_frames=1}"
  durable_probe_line="${durable_probe_line/fua_claim_to_write_done_nanos=0/fua_claim_to_write_done_nanos=1}"
  durable_probe_line="${durable_probe_line/fua_claim_to_write_done_frames=0/fua_claim_to_write_done_frames=1}"
  durable_probe_line="${durable_probe_line/fua_write_done_to_contiguous_cut_nanos=0/fua_write_done_to_contiguous_cut_nanos=1}"
  durable_probe_line="${durable_probe_line/fua_write_done_to_contiguous_cut_frames=0/fua_write_done_to_contiguous_cut_frames=1}"
  durable_probe_line="${durable_probe_line/fua_contiguous_cut_events=0/fua_contiguous_cut_events=1}"
  durable_probe_line="${durable_probe_line/fua_contiguous_cut_advanced_frames=0/fua_contiguous_cut_advanced_frames=1}"
  durable_probe_line="${durable_probe_line/fua_contiguous_cut_advance_max_frames=0/fua_contiguous_cut_advance_max_frames=1}"
  durable_probe_line="${durable_probe_line/fua_waiter_cut_to_observe_nanos=0/fua_waiter_cut_to_observe_nanos=1}"
  durable_probe_line="${durable_probe_line/fua_waiter_cut_to_observe_count=0/fua_waiter_cut_to_observe_count=1}"
  durable_probe_line="${durable_probe_line/fua_in_flight_depth_max=0/fua_in_flight_depth_max=16}"
  durable_probe_line="${durable_probe_line/fua_in_flight_depth_9_to_16=0/fua_in_flight_depth_9_to_16=1}"
  durable_probe_line="${durable_probe_line/fua_fence_lanes=0/fua_fence_lanes=32}"
  durable_probe_line="${durable_probe_line/fua_controller_sustained_actions=0/fua_controller_sustained_actions=1}"
  durable_probe_line="${durable_probe_line/fua_controller_phase=0/fua_controller_phase=1}"
  durable_probe_line="${durable_probe_line/fua_controller_sustained_remaining=0/fua_controller_sustained_remaining=511}"
  durable_probe_line="${durable_probe_line/fua_controller_action_reconciliation=0/fua_controller_action_reconciliation=1}"
  durable_probe_line="${durable_probe_line/fua_controller_sample_reconciliation=0/fua_controller_sample_reconciliation=1}"
  printf '%s\n' "$durable_probe_line" >"$scratch/probe-durable-valid.log"
  parse_gpu_insert_probe_record "$scratch/probe-durable-valid.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/durability_backend=fua/durability_backend=serial}" >"$scratch/probe-durable-backend-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-backend-mismatch.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_fence_lanes=32/fua_fence_lanes=0}" >"$scratch/probe-durable-fua-lanes-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-fua-lanes-mismatch.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_fenced_frames=1/fua_fenced_frames=0}" >"$scratch/probe-durable-fua-fence-reconciliation.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-fua-fence-reconciliation.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_in_flight_depth_9_to_16=1/fua_in_flight_depth_9_to_16=0}" >"$scratch/probe-durable-controller-depth.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-controller-depth.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_controller_sustained_actions=1/fua_controller_sustained_actions=0}" >"$scratch/probe-durable-controller-vacuous.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-controller-vacuous.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_controller_protocol_faults=0/fua_controller_protocol_faults=1}" >"$scratch/probe-durable-controller-fault.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-controller-fault.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_padded_bytes=4096/fua_padded_bytes=8193}" >"$scratch/probe-durable-amplification.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-amplification.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_controller_sustained_actions=1/fua_controller_sustained_actions=2}" >"$scratch/probe-durable-controller-action-law.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-controller-action-law.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${durable_probe_line/fua_controller_sample_reconciliation=1/fua_controller_sample_reconciliation=0}" >"$scratch/probe-durable-controller-sample-law.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-durable-controller-sample-law.log" durable 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/ fua_published_frames=0/}" >"$scratch/probe-fua-missing.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-fua-missing.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "$probe_line fua_published_frames=0" >"$scratch/probe-fua-duplicate.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-fua-duplicate.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "$probe_line fua_unknown=0" >"$scratch/probe-fua-unknown.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-fua-unknown.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/fua_padded_bytes=0/fua_padded_bytes=00}" >"$scratch/probe-fua-noncanonical.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-fua-noncanonical.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/successful_insert_rows=1000000/successful_insert_rows=999999}" >"$scratch/probe-count-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-count-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/device_authoritative_commits=1000/device_authoritative_commits=999}" >"$scratch/probe-device-authority-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-device-authority-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/fixed_insert_typed_commits=1000/fixed_insert_typed_commits=999}" >"$scratch/probe-fixed-cutover-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-fixed-cutover-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/fixed_insert_legacy_fallbacks=0/fixed_insert_legacy_fallbacks=1}" >"$scratch/probe-fixed-fallback.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-fixed-fallback.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/legacy_insert_delta_builds=0/legacy_insert_delta_builds=1}" >"$scratch/probe-legacy-delta-build.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-legacy-delta-build.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/predicted_row_keys_materialized=0/predicted_row_keys_materialized=1}" >"$scratch/probe-predicted-key.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-predicted-key.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/direct_fixed_insert_carriers=1000/direct_fixed_insert_carriers=999}" >"$scratch/probe-direct-carrier.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-direct-carrier.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/compat_classifier_gate_checks=4000/compat_classifier_gate_checks=3999}" >"$scratch/probe-classifier-checks.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-checks.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/compat_classifier_gate_rejections=4000/compat_classifier_gate_rejections=3999}" >"$scratch/probe-classifier-rejections.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-rejections.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/compat_classifier_slow_path_admissions=0/compat_classifier_slow_path_admissions=1}" >"$scratch/probe-classifier-slow-admissions.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-slow-admissions.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/compat_classifier_slow_path_bytes=0/compat_classifier_slow_path_bytes=1}" >"$scratch/probe-classifier-slow-bytes.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-slow-bytes.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/compat_classifier_gate_checks=4000/compat_classifier_gate_checks=4001}" >"$scratch/probe-classifier-torn-accounting.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-classifier-torn-accounting.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/commit_validation_reresolve_nanos=0/commit_validation_reresolve_nanos=1}" >"$scratch/probe-legacy-reresolve.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-legacy-reresolve.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/current_shard_count=2/current_shard_count=3}" >"$scratch/probe-shard-geometry-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-shard-geometry-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/rollover_capacity_rows_max=4000000/rollover_capacity_rows_max=3999999}" >"$scratch/probe-plausible-capacity-geometry-mismatch.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-plausible-capacity-geometry-mismatch.log" development 1 >/dev/null || failures=$((failures + 1))
  printf '%s\n' "${probe_line/capacity_fit_evaluation_count=23/capacity_fit_evaluation_count=0}" >"$scratch/probe-capacity-fit-underflow.log"
  ! parse_gpu_insert_probe_record "$scratch/probe-capacity-fit-underflow.log" development 1 >/dev/null || failures=$((failures + 1))
  capture_optional_command "$scratch/optional-command" printf printf '%s\n' optional-command-ok
  [[ "$(tr -d '\n' <"$scratch/optional-command.stdout")" == "optional-command-ok" ]] ||
    failures=$((failures + 1))
  [[ "$(tr -d '\n' <"$scratch/optional-command.stderr")" == "" ]] || failures=$((failures + 1))
  [[ "$(tr -d '\n' <"$scratch/optional-command.exit-status")" == "0" ]] || failures=$((failures + 1))
  printf '%s\n' \
    'manifest_version=1' \
    "workload=${INSERT_WORKLOAD_ID}" \
    'rows=1000000' \
    'chunk=1000' \
    'insert_statements=1000' \
    'total_statements=1001' \
    'source_bytes=12345' \
    "source_sha256=${source_digest}" \
    "statement_stream_sha256=${stream_digest}" >"$scratch/manifest"
  load_manifest_identity "$scratch/manifest" || failures=$((failures + 1))
  printf '%s\n' 'rows=1000000' >>"$scratch/manifest"
  ! load_manifest_identity "$scratch/manifest" || failures=$((failures + 1))
  printf '%s\n' '160014|16.14|off|off|off|fdatasync' >"$scratch/postgres-development-settings"
  postgres_runtime_settings_match development "$scratch/postgres-development-settings" || failures=$((failures + 1))
  ! postgres_runtime_settings_match durable "$scratch/postgres-development-settings" || failures=$((failures + 1))
  printf '%s\n' '160014|16.14|on|off|on|fdatasync' >"$scratch/postgres-sabotaged-settings"
  ! postgres_runtime_settings_match durable "$scratch/postgres-sabotaged-settings" || failures=$((failures + 1))
  printf '%s\n' \
    $'development\tpostgresql\t1\t30\t100.000\t40' \
    $'development\tpostgresql\t2\t10\t300.000\t20' \
    $'development\tpostgresql\t3\t20\t200.000\t30' >"$scratch/results.tsv"
  [[ "$(median_metric "$scratch/results.tsv" development postgresql 4)" == "20" ]] || failures=$((failures + 1))
  [[ "$(median_metric "$scratch/results.tsv" development postgresql 5)" == "200.000" ]] || failures=$((failures + 1))
  [[ "$(frozen_postgres_rows_per_second_floor development 1000000)" == "675859.699" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_postgres_rows_per_second_floor durable 1000000)" == "433754.096" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_postgres_rows_per_second_floor development 8000000)" == "765543.025" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_postgres_rows_per_second_floor durable 8000000)" == "436488.945" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_postgres_rows_per_second_floor development 48000000)" == "778301.447" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_postgres_rows_per_second_floor durable 48000000)" == "235965.297" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_workload_identity 1000000)" == \
    "15820837 7564cc9e47cce5fe1f2484ff0107f5aa1c73258bb5c8d0fa70f43a2cc1c65f23 ce0d027d434420874d361e1203f0c1094a66c20a68dd7b8b03c60c8d7d2e3393 a0d7472f39059f6694f5cd9cd28632dd5574506dc25d903debfa8642f87f8ea4" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_workload_identity 8000000)" == \
    "134344137 b5fccd46454bf9bb41fda29f6afb85a9af650fe19fe8b87d415e4bd54d0017d2 4b0bf8519415d560e584aa7a8c95882fb6436d53d0cb2448381f8372d4f69214 bdca1fcc6127739d1ba81db8b247b6546854774547135150ca42fa4937013b18" ]] ||
    failures=$((failures + 1))
  [[ "$(frozen_workload_identity 48000000)" == \
    "849620137 09472115e419b22a685f23cfb956406e57f9b91324e7f52db4ba855f7a4c0cb9 8f5d67db9521bfd345edd8956691ed9ee27c7b780cd68a57d4d00c9449ace44e a2266305f578762641f188f2d99be68762fbd371d27110a0f65386be756b01f0" ]] ||
    failures=$((failures + 1))
  printf '%s\n' $'development\tpostgresql\t3\t21\t201.000\t31' >>"$scratch/results.tsv"
  ! validate_result_set "$scratch/results.tsv" development postgresql || failures=$((failures + 1))
  cleanup_scratch || return 1
  if [[ "$failures" -ne 0 ]]; then
    echo "insert qualification self-check failed: ${failures} assertion(s)" >&2
    return 1
  fi
  echo "insert qualification self-check passed"
}

if [[ "$mode" == "self-check" ]]; then
  run_self_check
  exit $?
fi

[[ "$mode" == "qualification" ]] || die "unsupported mode"
[[ -z "$artifact_arg" || "$artifact_arg" != *$'\n'* ]] || die "artifact directory must not contain a newline"

# Ambient engine configuration would silently reshape the candidate and can contain credentials.
# Reject it by variable name before creating an evidence directory, and never record values.
mapfile -t ambient_gpu_db_vars < <(compgen -e | LC_ALL=C sort -u | awk '/^GPU_DB_/')
if [[ "${#ambient_gpu_db_vars[@]}" -ne 0 ]]; then
  die "unset ambient GPU_DB_* configuration before qualification: ${ambient_gpu_db_vars[*]}"
fi
if [[ -n "${INSERT_QUALIFICATION_POSTGRES_IMAGE+x}" ]]; then
  die "INSERT_QUALIFICATION_POSTGRES_IMAGE is not permitted; the PostgreSQL comparator is pinned"
fi

if [[ -n "$artifact_arg" ]]; then
  artifact_dir="$artifact_arg"
  [[ ! -e "$artifact_dir" ]] || die "artifact directory already exists: $artifact_dir"
  mkdir -p -- "$(dirname "$artifact_dir")"
  mkdir -- "$artifact_dir" || die "cannot create artifact directory: $artifact_dir"
else
  mkdir -p -- "$repo_root/target"
  artifact_dir="$(mktemp -d "$repo_root/target/insert-qualification.XXXXXX")" || die "cannot create artifact directory"
fi
artifact_dir="$(cd "$artifact_dir" && pwd -P)"
run_id="$$"
mkdir -p -- "$artifact_dir/preflight" "$artifact_dir/workload" "$artifact_dir/trials"

postgres_image="$POSTGRES_IMAGE_DEFAULT"
client_timeout="${INSERT_QUALIFICATION_CLIENT_TIMEOUT:-$DEFAULT_CLIENT_TIMEOUT_SECONDS}"
require_positive_uint "$client_timeout" "INSERT_QUALIFICATION_CLIENT_TIMEOUT"
development_frozen_floor="$(frozen_postgres_rows_per_second_floor development "$rows" || printf '%s' '<not-yet-frozen>')"
durable_frozen_floor="$(frozen_postgres_rows_per_second_floor durable "$rows" || printf '%s' '<not-yet-frozen>')"

printf '%s\n' \
  "qualification_mode=canonical" \
  "profiles=${profile}" \
  "rows=${rows}" \
  "chunk=${INSERT_CHUNK}" \
  "trials_per_backend=${INSERT_TRIALS}" \
  "sequence=gpu,postgresql,gpu,postgresql,gpu,postgresql" \
  "postgres_image=${postgres_image}" \
  "postgres_auth=local-trust-only" \
  "client_timeout_seconds=${client_timeout}" \
  "frozen_postgresql_rows_per_second_floor_development=${development_frozen_floor}" \
  "frozen_postgresql_rows_per_second_floor_durable=${durable_frozen_floor}" \
  "artifact_dir=${artifact_dir}" \
  "ambient_cargo_target_dir=${CARGO_TARGET_DIR:-<unset>}" \
  >"$artifact_dir/run-config.txt"

capture_optional_command "$artifact_dir/preflight/host-uname" uname uname -a
capture_optional_command "$artifact_dir/preflight/host-os-release" sed sed -n '1,160p' /etc/os-release
capture_optional_command "$artifact_dir/preflight/cpu-lscpu" lscpu lscpu
capture_optional_command "$artifact_dir/preflight/storage-df" df df -PT "$artifact_dir"
capture_optional_command "$artifact_dir/preflight/storage-findmnt" findmnt findmnt -T "$artifact_dir"
capture_optional_command "$artifact_dir/preflight/storage-lsblk" lsblk lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
capture_optional_command "$artifact_dir/preflight/gpu-nvidia-smi" nvidia-smi nvidia-smi -q
capture_optional_command "$artifact_dir/preflight/cargo-version" cargo cargo --version
capture_optional_command "$artifact_dir/preflight/rustc-version" rustc rustc --version --verbose
capture_optional_command "$artifact_dir/preflight/docker-version" docker docker version
capture_optional_command "$artifact_dir/preflight/git-head" git git -C "$repo_root" rev-parse HEAD
capture_optional_command "$artifact_dir/preflight/git-status" git git -C "$repo_root" status --short
capture_optional_command "$artifact_dir/preflight/git-index-tree" git git -C "$repo_root" write-tree

if compgen -G '/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor' >/dev/null; then
  while IFS= read -r governor_file; do
    printf '%s=' "$governor_file"
    cat -- "$governor_file"
  done < <(find /sys/devices/system/cpu -path '*/cpufreq/scaling_governor' -type f | sort) \
    >"$artifact_dir/preflight/cpu-governor.stdout" 2>"$artifact_dir/preflight/cpu-governor.stderr"
  write_exit_status "$artifact_dir/preflight/cpu-governor.exit-status" "0"
else
  printf 'unavailable: CPU governor files are absent\n' >"$artifact_dir/preflight/cpu-governor.stdout"
  : >"$artifact_dir/preflight/cpu-governor.stderr"
  write_exit_status "$artifact_dir/preflight/cpu-governor.exit-status" "127"
fi

command -v cargo >/dev/null 2>&1 || die "cargo is required"
command -v docker >/dev/null 2>&1 || die "docker is required"
command -v timeout >/dev/null 2>&1 || die "timeout is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

capture_command "$artifact_dir/preflight/docker-image-pull" docker pull "$postgres_image" || die "cannot pull PostgreSQL image; see $artifact_dir/preflight/docker-image-pull.stderr"
capture_command "$artifact_dir/preflight/docker-image-inspect" docker image inspect "$postgres_image" || die "cannot inspect PostgreSQL image"
resolved_postgres_image_id="$(docker image inspect --format '{{.Id}}' "$postgres_image")" ||
  die "cannot resolve PostgreSQL image identity"
[[ "$resolved_postgres_image_id" == "$POSTGRES_IMAGE_ID" ]] ||
  die "PostgreSQL image identity drifted: expected ${POSTGRES_IMAGE_ID}"
printf 'postgres_image_ref=%s\npostgres_image_id=%s\n' \
  "$postgres_image" "$resolved_postgres_image_id" \
  >"$artifact_dir/preflight/docker-image-identity.txt"

build_target="$artifact_dir/build-target"
mkdir -p -- "$build_target/tmp"
(
  cd "$repo_root" &&
    TMPDIR="$build_target/tmp" CARGO_TARGET_DIR="$build_target" cargo build --locked --release -p gpu_db_server \
      --features probe-timing --bin gpu-db-engine-server --example insert_workload_client
) >"$artifact_dir/build.stdout" 2>"$artifact_dir/build.stderr"
build_status=$?
write_exit_status "$artifact_dir/build.exit-status" "$build_status"
if [[ "$build_status" -ne 0 ]]; then
  die "release server/client build failed; see $artifact_dir/build.stderr"
fi
server_binary="$build_target/release/gpu-db-engine-server"
client_binary="$build_target/release/examples/insert_workload_client"
[[ -x "$server_binary" && -x "$client_binary" ]] || die "build did not produce exact release server and client binaries"
server_binary_sha256="$(sha256_file "$server_binary")"
client_binary_sha256="$(sha256_file "$client_binary")"
printf '%s\n' \
  "artifact=server path=${server_binary} sha256=${server_binary_sha256} bytes=$(file_size "$server_binary")" \
  "artifact=client path=${client_binary} sha256=${client_binary_sha256} bytes=$(file_size "$client_binary")" \
  >"$artifact_dir/binary-identities.txt"

source_path="$artifact_dir/workload/insert-001-${rows}-chunk-${INSERT_CHUNK}.sql"
manifest_path="$artifact_dir/workload/insert-001-${rows}-chunk-${INSERT_CHUNK}.manifest"
capture_command "$artifact_dir/workload/generate" "$client_binary" generate \
  --rows "$rows" --chunk "$INSERT_CHUNK" --source "$source_path" --manifest "$manifest_path" || die "workload generation failed"
capture_command "$artifact_dir/workload/verify" "$client_binary" verify \
  --source "$source_path" --manifest "$manifest_path" || die "workload verification failed"
source_file_sha256="$(sha256_file "$source_path")"
manifest_file_sha256="$(sha256_file "$manifest_path")"
load_manifest_identity "$manifest_path" || die "manifest is malformed"
[[ "$manifest_rows" == "$rows" && "$manifest_chunk" == "$INSERT_CHUNK" &&
  "$manifest_source_sha256" == "$source_file_sha256" ]] || die "generated workload identity does not match qualification shape"
read -r frozen_source_bytes frozen_source_sha256 frozen_stream_sha256 frozen_manifest_sha256 \
  <<<"$(frozen_workload_identity "$rows")" ||
  die "no frozen workload identity for ${rows} rows"
[[ "$manifest_source_bytes" == "$frozen_source_bytes" &&
  "$source_file_sha256" == "$frozen_source_sha256" &&
  "$manifest_statement_stream_sha256" == "$frozen_stream_sha256" &&
  "$manifest_file_sha256" == "$frozen_manifest_sha256" ]] ||
  die "generated workload drifted from the frozen INSERT-001 v1 identity"
create_statement_source_bytes="$(LC_ALL=C head -n 1 -- "$source_path" | wc -c | tr -d '[:space:]')"
is_canonical_uint "$create_statement_source_bytes" || die "cannot derive CREATE statement source bytes"
((10#$manifest_source_bytes >= 10#$create_statement_source_bytes)) || die "manifest source bytes underflow CREATE statement"
insert_source_bytes=$((10#$manifest_source_bytes - 10#$create_statement_source_bytes))
require_positive_uint "$insert_source_bytes" "derived insert_source_bytes"
read -r derived_insert_source_bytes insert_canonical_request_bytes insert_stripped_terminator_bytes \
  <<<"$(derive_insert_request_identity_bytes "$source_path" "$manifest_insert_statements")" ||
  die "cannot derive exact parser-stripped INSERT request bytes from frozen stream"
[[ "$derived_insert_source_bytes" == "$insert_source_bytes" ]] ||
  die "frozen INSERT source-byte derivation disagrees with manifest/CREATE boundary"
require_positive_uint "$insert_canonical_request_bytes" "derived canonical INSERT request bytes"
require_positive_uint "$insert_stripped_terminator_bytes" "derived stripped INSERT terminator bytes"
((10#$insert_stripped_terminator_bytes == 10#$manifest_insert_statements * 2)) ||
  die "frozen INSERT stream does not have one exact ;\\n boundary per statement"
((10#$insert_canonical_request_bytes + 10#$insert_stripped_terminator_bytes == 10#$insert_source_bytes)) ||
  die "canonical INSERT request bytes do not reconcile with raw source attribution"
printf '%s\n' \
  "source_path=${source_path}" \
  "manifest_path=${manifest_path}" \
  "source_file_sha256=${source_file_sha256}" \
  "manifest_file_sha256=${manifest_file_sha256}" \
  "manifest_source_sha256=${manifest_source_sha256}" \
  "statement_stream_sha256=${manifest_statement_stream_sha256}" \
  "rows=${manifest_rows}" \
  "chunk=${manifest_chunk}" \
  "insert_statements=${manifest_insert_statements}" \
  "total_statements=${manifest_total_statements}" \
  "source_bytes=${manifest_source_bytes}" \
  "create_statement_source_bytes=${create_statement_source_bytes}" \
  "insert_source_bytes=${insert_source_bytes}" \
  "insert_canonical_request_bytes=${insert_canonical_request_bytes}" \
  "insert_stripped_terminator_bytes=${insert_stripped_terminator_bytes}" \
  "insert_parser_stripped_terminator_per_statement_bytes=2" \
  >"$artifact_dir/workload/identity.txt"

results_file="$artifact_dir/qualified-results.tsv"
: >"$results_file"
printf 'profile\tbackend\ttrial\tinsert_load_wall_ms\trows_per_second\tend_to_end_client_wall_ms\n' \
  >"$artifact_dir/qualified-results-header.tsv"

declare -a selected_profiles=()
if [[ "$profile" == "development" ]]; then
  selected_profiles+=(development)
elif [[ "$profile" == "durable" ]]; then
  selected_profiles+=(durable)
else
  selected_profiles+=(development durable)
fi

profile_slot=0
qualification_floor_failed=0
for selected_profile in "${selected_profiles[@]}"; do
  printf 'insert_qualification_profile_status=started profile=%s rows=%s chunk=%s trials_per_backend=%s sequence=gpu,postgresql,gpu,postgresql,gpu,postgresql\n' \
    "$selected_profile" "$rows" "$INSERT_CHUNK" "$INSERT_TRIALS" | tee -a "$artifact_dir/qualification-summary.txt"
  for trial in 1 2 3; do
    run_one_trial "$selected_profile" "$profile_slot" gpu 0 "$trial" "$results_file" || die "GPU trial failed: profile=${selected_profile} trial=${trial}; artifacts retained in $artifact_dir"
    run_one_trial "$selected_profile" "$profile_slot" postgresql 1 "$trial" "$results_file" || die "PostgreSQL trial failed: profile=${selected_profile} trial=${trial}; artifacts retained in $artifact_dir"
  done
  validate_result_set "$results_file" "$selected_profile" gpu ||
    die "GPU result set is malformed or incomplete for profile=${selected_profile}"
  validate_result_set "$results_file" "$selected_profile" postgresql ||
    die "PostgreSQL result set is malformed or incomplete for profile=${selected_profile}"
  if ! emit_profile_medians "$selected_profile" "$results_file"; then
    qualification_floor_failed=1
  fi
  printf 'insert_qualification_profile_status=complete profile=%s rows=%s chunk=%s trials_per_backend=%s\n' \
    "$selected_profile" "$rows" "$INSERT_CHUNK" "$INSERT_TRIALS" | tee -a "$artifact_dir/qualification-summary.txt"
  profile_slot=$((profile_slot + 1))
done

if [[ "$qualification_floor_failed" -ne 0 ]]; then
  printf 'insert_qualification_status=failed mode=qualification reason=frozen-postgresql-floor profiles=%s rows=%s chunk=%s trials_per_backend=%s artifact_dir=%s\n' \
    "$profile" "$rows" "$INSERT_CHUNK" "$INSERT_TRIALS" "$artifact_dir" |
    tee -a "$artifact_dir/qualification-summary.txt"
  exit 2
fi

printf 'insert_qualification_status=complete mode=qualification profiles=%s rows=%s chunk=%s trials_per_backend=%s artifact_dir=%s\n' \
  "$profile" "$rows" "$INSERT_CHUNK" "$INSERT_TRIALS" "$artifact_dir" | tee -a "$artifact_dir/qualification-summary.txt"
