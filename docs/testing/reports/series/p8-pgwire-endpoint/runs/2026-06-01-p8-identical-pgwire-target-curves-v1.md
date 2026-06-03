# P8 Identical Pgwire Target Curves

- date: 2026-06-01
- stream: benchmark
- milestone: P8 identical PostgreSQL-compatible target smoke
- status: closed_with_blocker
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`
- smoke_artifact: `target/p8-identical-pgwire-target-curves-v1-pass/identical-pgwire-target-smoke/identical-pgwire-target-smoke.md`
- metrics_artifact: `target/p8-identical-pgwire-target-curves-v1-pass/identical-pgwire-target-smoke/metrics.jsonl`
- curve_artifact: `target/p8-identical-pgwire-target-curves-v1-pass/identical-pgwire-target-smoke/concurrency-curve.csv`
- next_blocker: `full_25pct_identical_curves_require_operator_long_run`

## Result

This slice closes the reusable shared target/driver primitive for the P8
fairness gate. The new `--identical-pgwire-target-smoke` path runs the same
scaled `order_line` load, the same query texts, the same `psql`/libpq client
boundary, the same concurrency schedule, and the same graph-ready metric schema
against:

- `default_postgresql`
- `tuned_postgresql`
- `gpu_db_retained_endpoint`

The tuned PostgreSQL profile differs only by setup DDL, adding btree and BRIN
indexes on `ol_o_id`. The GPU DB target differs only by endpoint URL and
lifecycle. Rows still arrive through SQL-visible `CREATE TABLE` plus
`COPY FROM STDIN`, commit through Engine WAL/MVCC state, and warm into
`RelationalResidentCache`; no benchmark-only chunk admission is used for the
pgwire endpoint evidence.

The focused run used 16 scaled rows and concurrency targets `1,2` for
`order_line_count_all` and `order_line_lookup_ol_o_id_multi_column`. All twelve
curve rows passed with zero errors. GPU DB rows are marked retained-route
`true`; PostgreSQL rows include profile notes and retained-route `false`.

## Remaining Boundary

This is not the full 161,061,274-row 25% curve. Full default PostgreSQL, tuned
PostgreSQL, and GPU DB retained curves now have a shared pgwire/libpq target
primitive, but still require an operator-approved long-run window and artifact
budget. Composite/text lookup remains
`retained_composite_or_text_lookup_required`. Full 125% remains blocked by
`missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- tiny scaled `--identical-pgwire-target-smoke` with rows `16`, concurrency
  `1,2`, GPU DB port `55466`, PostgreSQL Docker port `55467`: passed
- PostgreSQL Docker lifecycle cleanup after the smoke: passed
