# P8 Engine SQL-Visible Value-Index Bulk Admission

- date: 2026-06-01
- stream: benchmark
- milestone: SQL-visible MVCC value-index bulk admission for engine-backed pgwire COPY
- status: blocked
- previous_blocker: `engine_sql_visible_mvcc_value_index_bulk_admission_required`
- narrowed_blocker: `engine_sql_visible_copy_admission_storage_wal_profile_required`
- code_change: batch construction and append of relational value-index entries for INSERT/COPY and UPDATE rows
- probe_artifact: `target/p8-10pct-copy-path-value-index-probe/identical-pgwire-target-smoke/endpoint-facts.txt`
- cleanup_status: PostgreSQL Docker cleanup passed; no endpoint, benchmark, `psql`, or comparator container process remained

## Result

The worker implemented the smallest value-index-specific bulk admission slice:
rows are still validated through the existing supported `INSERT`/`COPY` subset,
still commit a durable SQL `INSERT` WAL payload before visibility, still insert
normal MVCC tuples with reserved generated row keys, and still preserve normal
equality-filter planning through `relational_value_index`.

The live apply path now builds value-index entries for a committed row batch in a
temporary ordered map and appends each grouped value entry into the main
`relational_value_index` once. The endpoint records
`copy_relational_value_index_bulk_admission=true` so benchmark artifacts prove
the new path was active.

Functional correctness held:

- all `1,048,576` rows committed through SQL-visible `COPY FROM STDIN`
- `128` bounded COPY chunks committed
- maximum buffered decoded rows remained `8,192`
- retained warmup installed a `1,048,576`-row snapshot
- retained `COUNT(*)`, multi-column int4 lookup, and composite/text lookup all
  passed through the `psql`/libpq boundary with zero H2D after warmup
- the focused unit test now asserts COPY rows are visible through the normal
  equality-index access path before and after WAL replay

## Bounded Probe

Command:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-path-value-index-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55556 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55557 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Throughput evidence:

- committed chunks: `128`
- committed rows: `1,048,576`
- total measured GPU DB COPY chunk time: `254,954` ms
- average COPY throughput over committed chunks: about `4,113` rows/sec
- fastest chunk: `28,946` rows/sec
- latest/slowest chunk: `2,241` rows/sec
- latest-rate 10pct COPY-only projection: about `7.99` hours for
  `64,424,510` rows
- average-rate 10pct COPY-only projection: about `4.35` hours before retained
  warmup, PostgreSQL default/tuned work, GPU DB query curves, cleanup, and
  reporting

The value-index grouping path is therefore not enough to make the 10pct
single-load curves defensible inside Richard's 6-hour worker budget. The latest
observed rate is slightly worse than the previous 1M probe's `2,427` rows/sec,
even though the new grouped value-index path was active.

## Narrowed Blocker

`engine_sql_visible_copy_admission_storage_wal_profile_required`: the blocker is
no longer defensibly described as the main value-index BTreeMap append path
alone. A next slice needs phase-specific evidence and then a targeted storage,
WAL-render, MVCC tuple-materialization, row-value encoding, or replay-preserving
bulk path.

Do not run the full 10pct default/tuned/GPU retained curves from this state. The
bounded 1M evidence still projects GPU DB COPY admission alone outside the
approved budget at the latest rate, and average-rate completion leaves too
little room for warmup, query curves, cleanup, and report handoff. Do not run
25pct or 125pct curves from this state.

## Safe Continuation Command

After adding phase timing or a narrower admission slice, retry the same bounded
probe with a fresh output directory:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-path-storage-wal-profile-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55556 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55557 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Only if bounded evidence projects the full 10pct run inside the 6-hour budget
including retained warmup, default/tuned PostgreSQL work, GPU DB query curves,
cleanup, and report handoff should a later worker consider:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v3 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine relational_copy_rows_commit_through_engine_wal_mvcc -- --nocapture`: passed
- required 1,048,576-row endpoint probe: passed functionally, blocked on throughput projection
- `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`: passed
- cleanup process/container check: no matching endpoint, benchmark, `psql`, PostgreSQL comparator, or Docker container remained
- `git diff --check`: passed
