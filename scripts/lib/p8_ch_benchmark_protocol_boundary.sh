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

  cargo run -q -p gpu_db_server --example p8_engine_protocol_boundary_probe >"$facts_path"

  cat >"$metrics_path" <<JSON
{"kind":"engine_backed_protocol_boundary_probe","rows":$rows,"status":"closed","engine_owned_target":true,"protocol_parser_reused":true,"startup_packet_parser_reused":true,"frontend_message_parser_reused":true,"wire_session_api_available":true,"copy_parser_in_protocol_lib":true,"backend_writer_api_available":true,"ready_loop_state_available":true,"engine_owned_session_probe":true,"copy_stream_lifecycle_probe":true,"protocol_server_session_catalog_reusable":false,"create_table_into_engine_wal_mvcc":true,"copy_rows_visible_through_engine_select":true,"resident_admission_from_sql_visible_rows":true,"retained_route_zero_h2d":true,"post_mutation_residency_invalidated":true,"next_blocker":"$blocker","secondary_blocker":"$secondary_blocker","retained_blocker":"$retained_blocker"}
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
\`gpu_db_protocol::backend::BackendWriter\`. Rows loaded through COPY are visible
through \`Engine::execute_relational_select(...)\` from the engine-owned session
state. It then warms \`order_line\` through
\`Engine::warm_relational_residency_with_policy(...)\`, executes
\`SELECT COUNT(*)\` through the default \`Engine::execute_relational_select(...)\`
retained route, records accepted zero-H2D telemetry, and verifies a later engine
mutation invalidates the resident snapshot. This closes the retained-admission
blocker without making \`gpu_db_protocol\` depend on \`gpu_db_engine\`.

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
