# P8 Retained Concurrency Pipeline Phase Profile

- stream: benchmark
- round_id: 2026-06-02-p8-concurrency-pipeline-profile-batched-retained-scheduler-v1
- status: blocked
- next_blocker: persistent_pgwire_client_phase_profile_required_before_scheduler_batching
- smoke_artifact: target/p8-concurrency-phase-smoke-c64/engine-backed-pgwire-concurrency-smoke/engine-backed-pgwire-concurrency-smoke.md
- metrics_artifact: target/p8-concurrency-phase-smoke-c64/engine-backed-pgwire-concurrency-smoke/metrics.jsonl
- curve_artifact: target/p8-concurrency-phase-smoke-c64/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv
- endpoint_facts: target/p8-concurrency-phase-smoke-c64/engine-backed-pgwire-concurrency-smoke/endpoint-facts.txt

## Result

Added retained endpoint phase telemetry for client-visible `SELECT` requests and
threaded it into the bounded pgwire concurrency smoke metrics. The endpoint now
records structured `select_phase_json` facts with owner-thread scheduler queue
wait, engine execute time, endpoint result materialization, pgwire response
write time, retained route wall time, CUDA event time, D2H bytes, kernel sample
delta, matched rows, and result rows.

The harness now attaches per-concurrency retained phase aggregates to
`engine_backed_pgwire_concurrency_metric` and
`identical_pgwire_target_metric` rows. It also sizes the engine-backed endpoint
session budget from the requested concurrency schedule, so the bounded
retained-route smoke can run through concurrency `64` without exhausting the
endpoint's accepted-session count.

## Evidence

The bounded smoke used 64 SQL-visible rows, `CREATE TABLE` plus `COPY FROM
STDIN`, retained warmup from WAL/MVCC table state, and real overlapping
`psql`/libpq sessions at concurrency `1,2,4,8,16,32,64`. It covered one broad
scan/count route and one lookup/projection route:

| query | concurrency | p50 us | p95 us | throughput qps | queue avg us | queue max us | engine avg us | retained wall avg us | CUDA event avg us | D2H avg bytes | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `COUNT(*)` | 64 | 44351 | 49910 | 745.746912 | 1692 | 3873 | 196 | 168 | 11 | 0 | 0 |
| `ol_o_id` multi-column lookup | 64 | 49355 | 56403 | 677.808138 | 5098 | 13867 | 317 | 260 | 11 | 40 | 0 |

The retained endpoint stayed zero-H2D for both route families. The lookup route
had the expected narrow selected-row D2H projection surface (`40` bytes average
in this tiny proof), while count remained `0` D2H.

## Blocker

This evidence does not justify same-shape batching or asynchronous retained
route scheduling yet. The endpoint phases are microsecond-scale in the bounded
profile, while the client-visible `psql` latency is tens of milliseconds. The
current concurrency harness launches one `psql` process per logical request, so
process/client overhead hides whether the production 10pct non-scaling curve is
dominated by owner-thread queueing, CUDA serialization, pgwire writes, or
client-side driver cost.

The next safe implementation slice is a reusable PostgreSQL-compatible client
session driver, or a libpq-based concurrent runner, that keeps the same SQL,
client boundary, and metric schema but removes per-request `psql` process
startup. Then the newly added endpoint phase fields can decide whether retained
route batching, multiple CUDA streams, or pgwire response work is the correct
optimization.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55450 GPU_DB_CH_BENCH_OUT_DIR=target/p8-concurrency-phase-smoke-c64 scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- `git diff --check`

No 25pct, 125pct, full 10pct reload, concurrency `128`, or broad benchmark-tier
command was run for this slice.
