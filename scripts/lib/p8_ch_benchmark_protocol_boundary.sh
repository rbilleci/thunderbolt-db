# shellcheck shell=bash
write_protocol_retained_route_bridge_report() {
  mkdir -p "$OUT_DIR/protocol-retained-route-bridge"
  local bridge_dir="$OUT_DIR/protocol-retained-route-bridge"
  local rows="${GPU_DB_CH_BENCH_PROTOCOL_BRIDGE_ROWS:-64}"
  local report_path="$bridge_dir/protocol-retained-route-bridge.md"
  local metrics_path="$bridge_dir/metrics.jsonl"
  local architecture_path="$bridge_dir/architecture-facts.txt"
  local blocker="engine_backed_protocol_endpoint_required"
  local secondary_blocker="protocol_shared_catalog_to_engine_adapter_required"
  local seed_blocker="protocol_seed_to_resident_cache_admission_required"
  local lookup_blocker="primary_key_lookup_retained_route_required"

  {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "crate_direction=gpu_db_engine_depends_on_gpu_db_sql_not_protocol"
    echo "protocol_crate_depends_on_engine=false"
    echo "protocol_server_target=crates/protocol/src/bin/gpu-db-server.rs"
    echo "protocol_state=Session/SharedCatalog private Table rows"
    echo "protocol_select_executor=execute_select_result over protocol Table rows"
    echo "protocol_copy_admission=apply_copy_in_rows extends protocol Table rows and persists SharedCatalog snapshot"
    echo "retained_engine_target=crates/engine/examples/p8_ch_benchmark_residency_probe.rs"
    echo "retained_engine_entry=Engine::new_local plus install_benchmark_relational_residency_owned_chunks plus execute_relational_select"
    echo "retained_seed_state=benchmark-only generated chunks outside normal SQL/MVCC inserts"
    echo "safe_bridge_this_slice=false"
  } >"$architecture_path"

  cat >"$metrics_path" <<JSON
{"kind":"protocol_retained_route_bridge_decision","rows":$rows,"status":"blocked","blocker":"$blocker","secondary_blocker":"$secondary_blocker","seed_blocker":"$seed_blocker","lookup_blocker":"$lookup_blocker","protocol_client_family_required":"psql/libpq or PostgreSQL-compatible driver","protocol_endpoint_current_route":"protocol_shared_catalog_cpu_scan","retained_route_current_entry":"engine_internal","crate_dependency_cycle_risk":true,"safe_to_label_protocol_smoke_as_retained":false}
{"kind":"protocol_retained_route_required_change","status":"blocked","smallest_next_unblocker":"add an engine-owned PostgreSQL-compatible server/benchmark target, or split shared protocol session/catalog adapters into a crate that can be used by an engine-backed endpoint without making gpu_db_protocol depend on gpu_db_engine"}
JSON

  cat >"$report_path" <<REPORT
# P8 Protocol To Retained Route Bridge

- rows: $rows
- status: blocked
- blocker: $blocker
- secondary_blocker: $secondary_blocker
- seed_blocker: $seed_blocker
- lookup_blocker: $lookup_blocker
- architecture_facts: $architecture_path
- metrics_artifact: $metrics_path

## Result

The checked PostgreSQL-compatible smoke path proves \`psql\`/libpq can seed and
query the GPU DB endpoint, but this bridge cannot be safely closed by wiring a
small call across the current crates. The protocol endpoint is implemented as
\`gpu_db_protocol\`'s \`gpu-db-server\` binary and stores protocol-visible rows in
private \`Session\` / \`SharedCatalog\` \`Table\` state. Its \`COPY FROM STDIN\`
path appends parsed rows to that protocol table state, and its \`SELECT\` path
executes \`execute_select_result(...)\` over those rows.

The retained P8 benchmark route is owned by \`gpu_db_engine\`: it creates an
\`Engine::new_local()\`, installs benchmark-only generated chunks into
\`RelationalResidentCache\` through
\`install_benchmark_relational_residency_owned_chunks(...)\`, and executes
\`execute_relational_select(...)\`. The current crate direction is
\`gpu_db_engine -> gpu_db_protocol\`; making the protocol server directly depend
on \`gpu_db_engine\` would introduce the dependency cycle the benchmark contract
explicitly rejects.

## Decision

Do not relabel \`--gpu-db-protocol-benchmark-smoke\` as retained-route evidence.
The narrow next unblocker is an engine-backed PostgreSQL-compatible target or a
crate-boundary split/shared adapter that lets protocol-visible SQL/COPY table
state enter engine-owned WAL/MVCC state and retained-resident admission before
benchmarking. A benchmark-only resident admission bridge would also need an
explicit product decision because it would bypass normal SQL durability.

Until that target exists, aggregate and key-equality protocol smoke metrics stay
classified as \`protocol_shared_catalog_cpu_scan\`, the existing 25% retained
aggregate result stays \`engine_internal\`, and true concurrency curves remain
blocked behind a retained-route PostgreSQL-compatible target.
REPORT

  cat "$report_path"
  echo "p8_ch_benchmark_protocol_retained_route_bridge=blocked reason=$blocker artifact=$report_path"
}

write_engine_backed_protocol_boundary_probe() {
  mkdir -p "$OUT_DIR/engine-backed-protocol-boundary"
  local boundary_dir="$OUT_DIR/engine-backed-protocol-boundary"
  local rows="${GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS:-64}"
  local report_path="$boundary_dir/engine-backed-protocol-boundary.md"
  local facts_path="$boundary_dir/probe-facts.txt"
  local metrics_path="$boundary_dir/metrics.jsonl"
  local blocker="identical_pg_client_concurrency_harness_required"
  local secondary_blocker="true_concurrent_client_curves_required"
  local retained_blocker="closed"
  declare -A facts=()
  declare -A allowed_facts=()
  local key value line
  local expected_fact_keys=(
    engine_owned_target protocol_parser_reused crate_direction
    create_table_into_engine_wal_mvcc select_parser_reused select_result_rows
    copy_parser_in_sql_lib engine_copy_column_projection_available copy_rows_decoded_by_protocol
    copy_rows_committed_to_engine_wal_mvcc copy_rows_visible_through_execute_relational_select
    backend_writer_api_available wire_session_ready_loop_available startup_packet_parser_reused
    frontend_message_parser_reused engine_owned_session_probe copy_stream_lifecycle_probe
    backend_startup_messages_written backend_copy_in_response_written backend_row_description_written
    backend_data_row_written backend_count_value_matches backend_ready_messages_written
    session_copy_rows_visible_through_engine_select protocol_server_session_catalog_reusable
    sql_visible_resident_warmup_entries sql_visible_resident_warmup_action
    sql_visible_resident_row_count sql_visible_resident_bytes
    sql_visible_resident_device_memory_retained retained_route_accepted retained_route_shape
    retained_route_zero_h2d retained_metadata_count_observed retained_resident_shard_count
    retained_route_h2d_delta retained_route_d2h_delta retained_route_kernel_delta
    retained_route_rows_visible post_mutation_residency_invalidated post_mutation_route_accepted
    post_mutation_route_reason post_mutation_route_shape post_mutation_route_zero_h2d
    post_mutation_metadata_count_observed post_mutation_resident_shard_count
    post_mutation_h2d_delta post_mutation_d2h_delta post_mutation_rows_visible
    post_mutation_resident_row_count post_mutation_incremental_append_observed
    post_mutation_incremental_shard_rollover_observed post_mutation_inserted_row_visible
    post_mutation_inserted_route_accepted post_mutation_inserted_route_shape
    post_mutation_inserted_route_zero_h2d post_mutation_inserted_device_route_witness
    post_mutation_inserted_shards_gathered_delta post_mutation_inserted_h2d_delta
    resident_admission_from_sql_visible_rows endpoint_boundary_status next_blocker
    secondary_blocker retained_blocker
  )
  for key in "${expected_fact_keys[@]}"; do
    allowed_facts[$key]=true
  done

  [[ "$rows" =~ ^[1-9][0-9]*$ ]]
  test "$rows" -le 1000000
  rm -f "$report_path" "$metrics_path" "$facts_path"

  cargo run -q -p gpu_db_server --example p8_engine_protocol_boundary_probe >"$facts_path"
  while IFS= read -r line; do
    [[ "$line" == *=* ]]
    key="${line%%=*}"
    value="${line#*=}"
    test -n "$key"
    [[ "$key" =~ ^[a-z0-9_]+$ ]]
    test "${allowed_facts[$key]:-}" = true
    test -z "${facts[$key]+present}"
    facts[$key]="$value"
  done <"$facts_path"
  test "${#facts[@]}" -eq "${#expected_fact_keys[@]}"

  local required_true=(
    engine_owned_target protocol_parser_reused create_table_into_engine_wal_mvcc
    select_parser_reused copy_parser_in_sql_lib engine_copy_column_projection_available
    copy_rows_committed_to_engine_wal_mvcc copy_rows_visible_through_execute_relational_select
    backend_writer_api_available wire_session_ready_loop_available startup_packet_parser_reused
    frontend_message_parser_reused engine_owned_session_probe copy_stream_lifecycle_probe
    backend_startup_messages_written backend_copy_in_response_written
    backend_row_description_written backend_data_row_written backend_count_value_matches
    backend_ready_messages_written
    session_copy_rows_visible_through_engine_select sql_visible_resident_device_memory_retained
    retained_route_accepted retained_route_zero_h2d retained_metadata_count_observed
    retained_route_rows_visible
    post_mutation_route_accepted post_mutation_route_zero_h2d
    post_mutation_metadata_count_observed post_mutation_rows_visible
    post_mutation_incremental_append_observed post_mutation_incremental_shard_rollover_observed
    post_mutation_inserted_row_visible post_mutation_inserted_route_accepted
    post_mutation_inserted_route_zero_h2d post_mutation_inserted_device_route_witness
    resident_admission_from_sql_visible_rows
  )
  for key in "${required_true[@]}"; do
    test "${facts[$key]:-}" = true
  done
  test "${facts[protocol_server_session_catalog_reusable]:-}" = false
  test "${facts[post_mutation_residency_invalidated]:-}" = false
  test "${facts[crate_direction]:-}" = gpu_db_engine_depends_on_gpu_db_sql_not_protocol
  test "${facts[select_result_rows]:-}" = 1
  test "${facts[sql_visible_resident_warmup_entries]:-}" = 1
  test "${facts[sql_visible_resident_warmup_action]:-}" = Warmed
  test "${facts[retained_route_shape]:-}" = sharded_count_all
  test "${facts[post_mutation_route_shape]:-}" = sharded_count_all
  test "${facts[post_mutation_route_reason]:-}" = 'sharded resident route accepted'
  test "${facts[post_mutation_inserted_route_shape]:-}" = sharded_int4_equality_mixed_column_projection
  test "${facts[endpoint_boundary_status]:-}" = sql_visible_retained_admission_ready
  test "${facts[next_blocker]:-}" = "$blocker"
  test "${facts[secondary_blocker]:-}" = "$secondary_blocker"
  test "${facts[retained_blocker]:-}" = "$retained_blocker"
  [[ "${facts[copy_rows_decoded_by_protocol]:-}" =~ ^[0-9]+$ ]]
  [[ "${facts[sql_visible_resident_row_count]:-}" =~ ^[0-9]+$ ]]
  [[ "${facts[post_mutation_resident_row_count]:-}" =~ ^[0-9]+$ ]]
  test "${facts[copy_rows_decoded_by_protocol]}" -eq "$rows"
  test "${facts[sql_visible_resident_row_count]}" -eq "$rows"
  test "${facts[post_mutation_resident_row_count]}" -eq "$((rows + 1))"
  [[ "${facts[sql_visible_resident_bytes]:-}" =~ ^[1-9][0-9]*$ ]]
  test "${facts[retained_route_h2d_delta]:-}" = 0
  test "${facts[post_mutation_h2d_delta]:-}" = 0
  test "${facts[post_mutation_inserted_h2d_delta]:-}" = 0
  [[ "${facts[retained_resident_shard_count]:-}" =~ ^[1-9][0-9]*$ ]]
  [[ "${facts[post_mutation_resident_shard_count]:-}" =~ ^[1-9][0-9]*$ ]]
  [[ "${facts[post_mutation_inserted_shards_gathered_delta]:-}" =~ ^[1-9][0-9]*$ ]]

  cat >"$metrics_path" <<JSON
{"kind":"engine_backed_protocol_boundary_probe","rows":${facts[sql_visible_resident_row_count]},"status":"closed","engine_owned_target":${facts[engine_owned_target]},"protocol_parser_reused":${facts[protocol_parser_reused]},"startup_packet_parser_reused":${facts[startup_packet_parser_reused]},"frontend_message_parser_reused":${facts[frontend_message_parser_reused]},"wire_count_value_matches":${facts[backend_count_value_matches]},"create_table_into_engine_wal_mvcc":${facts[create_table_into_engine_wal_mvcc]},"copy_rows_visible_through_engine_select":${facts[session_copy_rows_visible_through_engine_select]},"resident_admission_from_sql_visible_rows":${facts[resident_admission_from_sql_visible_rows]},"retained_count_source":"resident_shard_metadata","retained_route_accepted":${facts[retained_route_accepted]},"retained_route_shape":"${facts[retained_route_shape]}","retained_route_zero_h2d_telemetry":${facts[retained_route_zero_h2d]},"retained_resident_shard_count":${facts[retained_resident_shard_count]},"post_mutation_residency_invalidated":${facts[post_mutation_residency_invalidated]},"post_mutation_route_accepted":${facts[post_mutation_route_accepted]},"post_mutation_route_zero_h2d_telemetry":${facts[post_mutation_route_zero_h2d]},"post_mutation_rows_visible":${facts[post_mutation_rows_visible]},"post_mutation_resident_row_count":${facts[post_mutation_resident_row_count]},"post_mutation_resident_shard_count":${facts[post_mutation_resident_shard_count]},"post_mutation_incremental_append_observed":${facts[post_mutation_incremental_append_observed]},"post_mutation_incremental_shard_rollover_observed":${facts[post_mutation_incremental_shard_rollover_observed]},"post_mutation_inserted_row_visible":${facts[post_mutation_inserted_row_visible]},"post_mutation_inserted_route_accepted":${facts[post_mutation_inserted_route_accepted]},"post_mutation_inserted_route_shape":"${facts[post_mutation_inserted_route_shape]}","post_mutation_inserted_route_zero_h2d_telemetry":${facts[post_mutation_inserted_route_zero_h2d]},"post_mutation_inserted_device_route_witness":${facts[post_mutation_inserted_device_route_witness]},"post_mutation_inserted_shards_gathered_delta":${facts[post_mutation_inserted_shards_gathered_delta]},"next_blocker":"$blocker","secondary_blocker":"$secondary_blocker","retained_blocker":"$retained_blocker"}
{"kind":"endpoint_boundary_decision","status":"closed","narrowest_safe_next_step":"run the identical PostgreSQL-compatible client harness and true concurrency curves against the retained engine route"}
JSON

  cat >"$report_path" <<REPORT
# P8 SQL-Visible Retained Admission Probe

- rows: $rows
- status: closed
- adapter_boundary: engine-owned startup/simple-query/COPY/select session probe with retained admission
- next_blocker: $blocker
- secondary_blocker: $secondary_blocker
- retained_blocker: $retained_blocker
- probe_facts: $facts_path
- metrics_artifact: $metrics_path

## Result

The checked \`p8_engine_protocol_boundary_probe\` example now proves the bounded
SQL-visible retained-route admission boundary. The probe owns
\`Engine::new_local()\`, reuses \`gpu_db_protocol\` startup-packet parsing,
frontend-message parsing, SQL parsing, COPY statement/row decoding, and backend
result writers, then routes a PostgreSQL-shaped startup + simple-query
\`CREATE TABLE\` + \`COPY FROM STDIN\` + \`SELECT COUNT(*)\` flow into engine
WAL/MVCC table state.

The same probe writes normal backend startup, \`CopyInResponse\`, command,
\`RowDescription\`, \`DataRow\`, and \`ReadyForQuery\` messages through
\`gpu_db_protocol::backend::BackendWriter\`, then decodes the emitted DataRow to
verify the client-visible COUNT value. Rows loaded through COPY are visible
through \`Engine::execute_relational_select(...)\` from the engine-owned session
state. It then warms \`order_line\` through
\`Engine::warm_relational_residency_with_policy(...)\`, executes
\`SELECT COUNT(*)\` through the default \`Engine::execute_relational_select(...)\`
retained route, records accepted zero-H2D telemetry, and verifies a later engine
mutation preserves the accepted resident route through incremental INSERT
maintenance. This closes the retained-admission
blocker without making \`gpu_db_protocol\` depend on \`gpu_db_engine\`.

The COUNT route is explicitly classified as a resident-shard metadata result;
it is not labeled as a kernel execution. The exact inserted-row projection
records the accepted mixed-column device route, the sharded-source counter,
exact values, and zero-H2D telemetry. This boundary probe is not a substitute
for BENCH-001's missing open-loop client and kernel/event evidence.

## Narrowest Viable Architecture

The next implementation slice should run the identical PostgreSQL-compatible
client harness and true concurrency curves against this retained engine route.
The current protocol server may still keep its private
\`Session\` / \`SharedCatalog\` path for the broader compatibility endpoint, but
the P8 benchmark target now has a bounded engine-owned path for
startup/simple-query/COPY/result writing plus retained admission from
SQL-visible rows.

## Rejected Alternatives

- Do not add \`gpu_db_engine\` as a dependency of \`gpu_db_protocol\`; the
  workspace already depends in the opposite direction.
- Do not relabel the current \`--gpu-db-protocol-benchmark-smoke\` metrics as
  retained-route evidence; those are protocol \`SharedCatalog\` CPU scans.
- Do not bypass WAL/MVCC by admitting COPY output directly to retained chunks;
  this probe warms from engine-visible table rows after COPY has committed.
- Do not collect true-concurrency product curves until the PostgreSQL-compatible
  target reaches the retained engine route.

## Benchmark Gate Impact

The existing 25% aggregate result remains provisional \`engine_internal\`
evidence. This slice proves SQL/COPY-loaded rows can enter engine WAL/MVCC,
be admitted to retained residency, and return through a PostgreSQL-shaped
result-writing boundary with accepted zero-H2D retained-route telemetry.
The benchmark trust gate still needs the same PostgreSQL-compatible client
harness for default PostgreSQL, tuned PostgreSQL, and GPU DB plus true
concurrent-client curves.
The 125% tier remains blocked on \`missing_partitioned_over_resident_execution\`.
REPORT

  cat "$report_path"
  echo "p8_ch_benchmark_sql_visible_retained_admission=closed next_blocker=$blocker artifact=$report_path"
}
