# P8 Benchmark Methodology And Status

This page is the front door for the current P8 benchmark evidence. It explains
what has been proven, what remains guarded, and where the durable artifacts live.

## Scope

The current P8 benchmark lane measures CH-benCHmark-derived retained residency
over deterministic `order_line` data for supported `int4`/`text` table shapes.
It is a PostgreSQL-compatible client-path benchmark lane: trusted headline
comparisons must use the same `psql`/libpq driver, query text, metric schema,
and concurrency schedule for default PostgreSQL, tuned PostgreSQL, and the GPU
DB retained endpoint.

Current query families include:

- analytical `COUNT(*)`, retained `SUM(int4)`, retained `AVG ... BETWEEN`, and
  deterministic empty-domain `MAX ... filter`
- same-column int4 point lookup projections
- bounded multi-column int4 lookup projections
- composite-key int4 lookup projections
- composite/text point lookup projection with `ol_dist_info`

## Harness

The operator harness is:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh
```

Useful bounded modes:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-fairness-audit
scripts/run_p8_ch_benchmark_residency_probe.sh --gpu-db-protocol-benchmark-smoke
scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-benchmark-smoke
scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-full-readiness
```

The full 25% and 125% tiers are deliberately guarded. Do not run them from
unattended automation.

## Current Accepted Evidence

The latest accepted identical-client milestone is:

- report:
  [docs/testing/reports/2026-06-02-p8-identical-10pct-execution-v4.md](../reports/2026-06-02-p8-identical-10pct-execution-v4.md)
- command: `scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`
- rows: `64,424,510`
- concurrency: `1,2,4,8,16,32,64`
- profiles: default PostgreSQL, tuned PostgreSQL, GPU DB retained endpoint
- metric shape: graph-ready p50/p95/p99/throughput/correctness/error count,
  retained-route flag, route classification, and saturation note rows

The tuned PostgreSQL profile is explicit and bounded: it adds the existing
`ol_o_id` btree/BRIN coverage plus a composite btree on `(ol_o_id, ol_i_id)`.
Default PostgreSQL keeps only `CREATE TABLE` plus `COPY FROM STDIN`.

The GPU DB endpoint seeds through SQL-visible `CREATE TABLE` plus
`COPY FROM STDIN`, commits rows through Engine WAL/MVCC state, warms them into
`RelationalResidentCache`, and serves accepted lookup shapes through the
engine-backed pgwire endpoint.

The identical pgwire smoke now streams deterministic setup and `COPY` rows into
each target through `psql`; it does not materialize an
`identical-pgwire-target-smoke/load.sql` file. The GPU DB endpoint now commits
decoded rows to Engine WAL/MVCC in bounded chunks before retained warmup, and
the harness sizes the endpoint lifecycle from the requested GPU DB query/client
schedule instead of a fixed cap. Richard approved a full 25% long-run window,
then pivoted the next benchmark attempt to 10% of GPU memory. The first
  approved 25% attempt proved default PostgreSQL can accept the full
161,061,274-row streamed load. The latest bounded endpoint probes improved
generated row-key MVCC insertion, grouped value-index appends, and COPY
admission phase profiling. The measured decaying phase was an unnecessary
all-visible-row constraint preflight scan for unconstrained COPY chunks; after
that guard, the 1,048,576-row bounded GPU DB probe projected the
64,424,510-row 10% COPY load inside the approved 6h budget. The first 10%
attempt loaded default PostgreSQL and the GPU DB retained endpoint
successfully, but it is blocked because GPU DB sustained COPY admission
measured below the required 30k rows/sec target and retained query timings made
the remaining full concurrency curve indefensible inside the worker budget. A
follow-up retained-query profile removed the measured 1M-row retained query
setup bottlenecks for `COUNT(*)`, multi-column int4 lookup, and composite/text
lookup. A follow-up COPY admission recheck still missed the 30k rows/sec
target, narrowing the next implementation boundary to the WAL commit/flush plus
relational value-index append path. The WAL/current-apply architecture slice
then cleared the bounded 30k rows/sec gate by avoiding duplicate generic
state-machine clone/reparse of the current engine-applied COPY WAL entry. The
full 10% retry is now accepted: default PostgreSQL loaded 64,424,510 rows at
61,660 rows/sec, and the GPU DB retained endpoint loaded the same row count at
30,992 rows/sec with retained query correctness through concurrency 64.

The latest scaled smoke includes:

```sql
SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info
FROM order_line
WHERE ol_o_id = <literal> AND ol_i_id = <literal>
```

For that bounded composite/text lookup, endpoint facts report retained-route
execution with zero H2D, device-side match-index compaction for the equality
filters, and result-sized D2H readback for the selected row plus compacted
match-index output.

## Related Reports

- [2026-05-31 P8 PostgreSQL fairness audit](../reports/2026-05-31-p8-pgsql-fairness-audit-v1.md):
  default/tuned PostgreSQL evidence, settings, EXPLAIN samples, and concurrency
  plan.
- [2026-06-01 identical pgwire target curves](../reports/2026-06-01-p8-identical-pgwire-target-curves-v1.md):
  shared PostgreSQL-compatible target/metric schema before composite/text
  lookup closure.
- [2026-06-01 retained row-id gather route](../reports/2026-06-01-p8-retained-row-id-gather-route-v1.md):
  selected-row retained lookup D2H narrowing.
- [2026-06-01 retained match-index compaction](../reports/2026-06-01-p8-retained-match-index-compaction-v1.md):
  device-side retained equality match-index compaction for selected-row lookup
  projections.
- [2026-06-01 identical full-run streaming guard](../reports/2026-06-01-p8-identical-full-run-streaming-guard-v1.md):
  streamed identical pgwire load contract, full 25% guard, and readiness facts.
- [2026-06-01 engine pgwire COPY/session readiness](../reports/series/p8-copy-admission/runs/2026-06-01-p8-engine-pgwire-full-copy-session-v1.md):
  bounded endpoint COPY admission, full-curve session sizing, and scaled
  default/tuned/GPU retained proof.
- [2026-06-01 engine pgwire full-COPY throughput](../reports/series/p8-copy-admission/runs/2026-06-01-p8-engine-pgwire-full-copy-throughput-v1.md):
  current-process decoded apply and the narrowed SQL-visible MVCC bulk COPY
  admission blocker.
- [2026-06-01 10% copy path single-load curves](../reports/series/p8-copy-admission/runs/2026-06-01-p8-10pct-copy-path-single-load-curves-v1.md):
  10% row-target derivation, first-class harness tier labeling, and the current
  6h-budget blocker for GPU DB endpoint COPY admission.
- [2026-06-01 engine SQL-visible bulk COPY admission](../reports/series/p8-copy-admission/runs/2026-06-01-p8-engine-sql-visible-bulk-copy-admission-v1.md):
  reserved generated-row-key MVCC insertion, bounded 65k and 1M endpoint probes,
  and the narrowed value-index bulk admission blocker.
- [2026-06-01 engine SQL-visible value-index bulk admission](../reports/series/p8-copy-admission/runs/2026-06-01-p8-engine-sql-visible-value-index-bulk-admission-v1.md):
  grouped value-index appends, bounded 1M endpoint evidence, and the narrowed
  storage/WAL/MVCC phase-profile blocker.
- [2026-06-01 engine SQL-visible COPY admission phase profile](../reports/series/p8-copy-admission/runs/2026-06-01-p8-engine-sql-visible-copy-admission-phase-profile-v1.md):
  phase-specific COPY admission evidence, the unconstrained-table preflight
  scan fix, and bounded 1M endpoint evidence projecting 10% COPY admission
  inside the 6h budget.
- [2026-06-01 10% identical single-load curves](../reports/2026-06-01-p8-10pct-identical-single-load-curves-v1.md):
  first 10% default/tuned/GPU retained execution attempt, load metrics, partial
  graph-ready concurrency artifacts, and the narrowed GPU DB COPY/query
  throughput blocker.
- [2026-06-01 10% retained query throughput profile](../reports/2026-06-01-p8-10pct-retained-query-throughput-profile-v1.md):
  retained-query setup phase evidence, the snapshot-clone and conjunctive
  access-path fixes, 1M-row after-fix proof, and the remaining COPY admission
  recheck blocker.
- [2026-06-02 10% COPY admission 30k recheck](../reports/series/p8-copy-admission/runs/2026-06-02-p8-10pct-copy-admission-30000-recheck-v1.md):
  required 1M-row COPY admission recheck, retained-query health smoke, and the
  narrowed WAL/value-index COPY admission blocker.
- [2026-06-02 COPY admission WAL/value-index architecture](../reports/series/p8-copy-admission/runs/2026-06-02-p8-copy-admission-wal-value-index-architecture-v1.md):
  the bounded WAL/current-apply architecture slice that cleared the 30k
  rows/sec COPY gate and left the full 10% retry as the next benchmark decision.
- [2026-06-02 10% identical pgwire execution v4](../reports/2026-06-02-p8-identical-10pct-execution-v4.md):
  accepted 10% default PostgreSQL and GPU DB retained execution through
  concurrency 64 with GPU DB COPY above the required 30k rows/sec gate.
- [2026-06-02 over-resident partitioned readiness](../reports/2026-06-02-p8-over-resident-partitioned-readiness-v1.md):
  narrowed 125% over-resident readiness from a broad missing execution path to
  the partitioned resident route/execution primitive required before full
  125pct scheduling.
- [2026-06-01 retained composite/text lookup route](../reports/2026-06-01-p8-retained-composite-text-lookup-route-v1.md):
  retained composite lookup progression.
- [2026-05-31 25% aggregate refresh after BETWEEN](../reports/2026-05-31-p8-25pct-aggregate-refresh-after-between-v1.md):
  provisional engine-internal retained aggregate evidence.
- [2026-05-31 125% over-resident readiness](../reports/2026-05-31-p8-125pct-over-resident-readiness-v1.md):
  checked over-resident readiness decision and blocker.

## Remaining Blockers

- `pgsql_128_client_count_query_errors_need_classification`: default/tuned
  PostgreSQL 128-client count-query errors appeared in the 10% attempt and must
  be classified before trusting 128-client comparator rows.
- `partitioned_resident_route_execution_primitive_required`: the 125% tier
  requires partition metadata, planner acceptance, per-partition execution, and
  result reduction because the current retained layout expects one resident
  CUDA allocation per table, larger than local RTX 3090 memory at 125%.

## Explicit Non-Claims

The current benchmark evidence does not claim:

- full 25% PostgreSQL-vs-GPU retained curves
- completed 125% PostgreSQL-vs-GPU retained curves
- full CH-benCHmark or BenchBase compatibility
- joins or transaction-mix benchmarking
- external load-generator coverage beyond the checked `psql`/libpq paths
- production cache-daemon scheduling
- durable GPU pages
- external orchestration
- broad retained expressions beyond the documented supported families
- broad CUDA timing coverage beyond current route/event samples

## Artifact Locations

Reports are checked in under
[docs/testing/reports/](../reports/). Raw benchmark JSONL, CSV, endpoint facts,
and generated human-readable smoke artifacts are written under `target/` by each
run and are referenced from the corresponding report.
