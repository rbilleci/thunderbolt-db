# P8 Identical Composite/Text Pgwire Curves

- date: 2026-06-01
- stream: benchmark
- milestone: identical PostgreSQL-compatible target smoke for composite/text lookup curves
- status: closed_with_blocker
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`
- smoke_artifact: `target/p8-identical-composite-text-pgwire-curves-v1-pass/identical-pgwire-target-smoke/identical-pgwire-target-smoke.md`
- metrics_artifact: `target/p8-identical-composite-text-pgwire-curves-v1-pass/identical-pgwire-target-smoke/metrics.jsonl`
- curve_artifact: `target/p8-identical-composite-text-pgwire-curves-v1-pass/identical-pgwire-target-smoke/concurrency-curve.csv`
- endpoint_facts: `target/p8-identical-composite-text-pgwire-curves-v1-pass/identical-pgwire-target-smoke/endpoint-facts.txt`
- next_blocker: `full_25pct_identical_curves_require_operator_long_run`

## Result

This slice extends the identical `psql`/libpq target smoke so default
PostgreSQL, tuned PostgreSQL, and the GPU DB retained endpoint all run the same
composite/text point lookup query through the same client boundary:

`SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = <literal> AND ol_i_id = <literal>`

The tuned PostgreSQL profile remains explicit and bounded. It adds the existing
`ol_o_id` btree/BRIN coverage plus a composite btree on `(ol_o_id, ol_i_id)`;
default PostgreSQL keeps only `CREATE TABLE` plus `COPY FROM STDIN`. The GPU DB
target still seeds through SQL-visible `CREATE TABLE` plus `COPY FROM STDIN`,
commits through Engine WAL/MVCC state, warms into `RelationalResidentCache`, and
serves the lookup through the engine-backed pgwire endpoint.

The checked scaled run used 16 rows and concurrency targets `1,2`. The curve has
18 graph-ready metric rows: 3 target profiles x 3 queries x 2 concurrency
levels. The composite/text rows all passed with zero errors. GPU DB
composite/text rows are marked retained route `true` with route classification
`retained_engine_int4_text_composite_equality_projection`; endpoint facts record
zero H2D and 46 D2H bytes for the selected-row composite/text lookup.

## Remaining Boundary

This is a scaled smoke, not the full 161,061,274-row 25% curve. Full default
PostgreSQL, tuned PostgreSQL, and GPU DB retained curves still require an
operator-approved long-run window and artifact budget. Full 125% remains
blocked by `missing_partitioned_over_resident_execution`, and fully device-side
retained filtering remains blocked by
`retained_match_index_compaction_required_for_fully_device_side_filtering`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- tiny scaled `--identical-pgwire-target-smoke` with rows `16`, concurrency
  `1,2`, GPU DB port `55482`, PostgreSQL Docker port `55483`: passed
- PostgreSQL Docker cleanup after the smoke: passed
