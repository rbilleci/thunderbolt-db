# P8 CH-benCHmark 25% VRAM Blocker

- date: 2026-05-30
- stream: benchmark
- milestone: P8 CH-benCHmark-derived retained-residency 6 GiB tier
- git_baseline: `6604391f docs(testing): record release candidate evidence refresh`
- validation_gate: `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`; `git diff --check`
- preflight_artifacts: `target/p8-ch-benchmark-residency/25pct-preflight.md`; `target/p8-ch-benchmark-residency/25pct-preflight.jsonl`
- cleanup: `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`

## Work Order Result

The 25% / 6 GiB CH-benCHmark-derived retained-residency tier did not start.
The blocker is not disk capacity or operator run-window approval. The blocker
is that the current executable probe still seeds through the in-memory MVCC
engine and builds a resident snapshot by collecting all decoded rows plus the
device payload in process memory before GPU admission.

That path is appropriate for the checked calibration baseline, but it is not a
bounded streaming/on-disk generator for the estimated 161061274-row tier.

## Disk Estimate

The dry-run estimate for the 25% tier is:

- retained_target_bytes: `6442450944`
- estimated_order_line_rows: `161061274`
- generated_table_bytes: `15461882304`
- wal_log_bytes: `7730941152`
- report_bytes: `2097152`
- required_disk_bytes: `23194920608`

At run time the repo filesystem reported `378006937600` bytes available,
so disk was sufficient for this preflight estimate.

## Failed Command

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct
```

The command writes the tier preflight artifacts, then exits non-zero before any
unbounded row generation starts:

```text
p8_ch_benchmark_25pct=blocked reason=missing_bounded_streaming_on_disk_generator_and_chunked_resident_snapshot_builder
```

## Source Inspection

`crates/engine/examples/p8_ch_benchmark_residency_probe.rs` calls
`seed_engine(args.rows)` and inserts one row at a time into the local engine.
`Engine::populate_relational_residency_snapshot(...)` then scans MVCC state,
decodes every row into `resident_rows`, constructs a single `device_payload`,
and only then admits the snapshot.

For the 6 GiB tier this means the current path would require a very large
in-process row vector plus a large contiguous device payload before the
benchmark reaches the resident route. That is not a trustworthy bounded
long-run path.

## Tier Plan

The runnable harness and source-truth docs no longer configure or advertise the
removed 400% / 96 GiB tier. The active retained-residency plan is 25%, 50%,
100%, and 200% of RTX 3090 VRAM, with 50/100/200% still gated behind a checked
25% tier.

## Cleanup Evidence

The generated blocker artifacts live under `target/p8-ch-benchmark-residency/`
and are removed by:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup
```

Cleanup passed after this run with
`p8_ch_benchmark_cleanup=passed path=target/p8-ch-benchmark-residency`.

## Next Decision

The next defensible slice is not another calibration rerun. It is a bounded
streaming/on-disk CH-benCHmark-derived generator plus a chunked resident
snapshot/admission path that can produce the 6 GiB retained table without
materializing all generated rows twice in process memory.
