# Candidate-A SLO and footprint decision report

**Date:** 2026-07-15\
**Source:** committed baseline `f701d8b6` plus the review-only R3-006 correction and benchmark/probe diff listed
in the frozen packet\
**Host:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, synchronous FUA lane WAL\
**Disposition:** **FAIL — the current Candidate-A implementation does not satisfy production graduation; this
report alone did not select the ADR.**

This is retained ADR-014 decision evidence, not a product benchmark or an implementation authority. The harness drives the
real covered `t(id INT PRIMARY KEY, v INT)` GPU intent route: prepared typed parameters, device validation and
append/tombstone apply, FUA lane durability, publication, and optional crash replay. Open-loop latency starts at the
scheduled arrival, so producer slip and queueing are included. The binding latency threshold is W1 single keyed
synchronous mutation p50/p99/p99.9 below 0.8/1.5/5 ms. The charter's >100,000 sustained aggregate target and fixed
400,000-transaction peak cohort now apply only to the deterministic R1/W1/T8/T32 system mix; this standalone/I/U/D-only harness reports TPS
as diagnostic capacity and cannot pass or fail that aggregate gate. The binding schema/data/access/routes/timing are
frozen separately in `oltp-benchmark-workload-v1.md`. This report predates the
classed target decision but its raw measurements remain valid. Its “mixed” rows are I/U/D-only, not read/write,
and the harness pools DML latency samples rather than reporting INSERT, UPDATE, and DELETE distributions separately.
That pooling is insufficient for future graduation, where each operation and the declared mix must pass W1.

## End-to-end matrix

All rows use strict synchronous commit. “Offered” rows are paced arrivals; “saturation” rows are closed-loop. The
reported one-second peak is not treated as success when backlog makes scheduled-arrival latency diverge.

| Workload and load point | Achieved / peak TPS | p50 | p99 | p99.9 | Result |
|---|---:|---:|---:|---:|---|
| INSERT, 1,000 offered | 1,000 / 1,270 | 2.88 ms | 6.57 ms | 27.64 ms | **FAIL** all latency thresholds |
| INSERT, 100,000 offered | 99,902 / 328,610 | 1.20 ms | 210.29 ms | 228.04 ms | **FAIL** all latency thresholds |
| INSERT, 400,000 offered | 399,318 / 883,180 | 3.99 ms | 239.82 ms | 250.98 ms | **FAIL** latency; achieved-rate diagnostic |
| INSERT, closed-loop saturation, 32 drivers x 128 | 473,619 / 560,180 | 6.54 ms | 12.04 ms | 244.25 ms | throughput observed; **FAIL** latency |
| 80% INSERT / 20% UPDATE, saturation | 312,763 / 397,140 | 1.33 ms | 5.56 ms | 10.22 ms | **FAIL** latency; TPS diagnostic |
| 80% INSERT / 20% DELETE, saturation | 275,892 / 387,800 | 1.36 ms | 6.97 ms | 13.71 ms | **FAIL** latency; TPS diagnostic |
| mixed requested 70/20/10 I/U/D, 100,000 offered; actual 73.5/18.4/8.2 | 92,744 / 155,600 | 1.28 ms | 44.88 ms | 223.92 ms | **FAIL** latency; TPS diagnostic |
| same mixed workload, 400,000 offered | 237,110 / 609,900 | 412.14 ms | 1,209.26 ms | 1,222.49 ms | **FAIL**, unstable overload/backlog |

The mixed percentages differ from the requested weights because the harness preserves a lagged committed live-key
pool and computes each mutation quota relative to its insert cursor; the report states the measured mix rather than
relabeling it. UPDATE and DELETE are
non-vacuous: the 100,000-offered run recorded 52,510 updates, 23,340 deletes, 11,098 visible device locates, 75,850
tombstone applies, and two PK-index rebuilds. Recovery reconstructed the exact 186,690-row result in 7.44 seconds.

The low-load result alone rejects the candidate against the declared matrix: its mean FUA fence was about 3.13 ms
per frame and mean publish-to-settle was about 3.14 ms per wave, so batching policy cannot turn the present strict
durability path into a sub-0.8-ms W1 median. At the 100,000 mixed point the mean fence improved to 0.92 ms and
publish-to-settle to 1.76 ms, but p99/p99.9 still diverged. The existing population/deadline policy, automatic FUA
subframing, and one active-lane resize did not keep the tail within the residual client deadline.

The harness currently records one pooled DML end-to-end distribution plus stage averages, not separate INSERT,
UPDATE, and DELETE p50/p99/p99.9, every-stage percentiles, or a read-after-write distribution. That missing
attribution cannot change the rejection: the binding W1 end-to-end threshold already fails at every required load
point. It remains post-selection implementation evidence only if a future candidate first passes the independent
operation and declared-mix end-to-end gates.

## Actual narrow-path bytes

The build-only `probe-timing` instrumentation reads the engine's retained CUDA allocations directly, split into
payload/MVCC regions and device index memory. FUA physical WAL bytes are exact fenced 4 KiB frames, rather than the
64 MiB preallocated file lengths.

| Run | Appended versions | Visible rows | Payload + MVCC regions | Device index | Total retained | Physical WAL |
|---|---:|---:|---:|---:|---:|---:|
| 300,001 INSERT operations | 300,003 including warm-up | 300,003 | 102,291,472 B | 142,606,336 B | 244,897,808 B = 816.3 B/appended version | 129,986,560 B = 433.3 B/op |
| 285,878 mixed operations | 262,540 including warm-up | 186,690 | 104,388,624 B | 142,606,336 B | 246,994,960 B = 940.8 B/appended version, 1,323.0 B/visible row | 123,633,664 B = 432.5 B/op |

These are actual current allocations, including capacity headroom and current index geometry; they are not the
earlier 64-B static extrapolation. They also show why bytes per live row and snapshot-retained history must be
measured rather than inferred from logical column widths.

The canonical width/fanout matrix cannot pass on the current candidate implementation. Route preparation explicitly
rejects a table unless every column is `INT4`, UPDATE requires a full all-INT4 image, compound keys use a different
classic covered path, and the measured table has one int4 PK index. The focused
`covered_insert_route_requires_covered_shape` test confirms the non-INT4 refusal. Therefore there is no honest
current-candidate measurement for variable-width rows, wider fixed types, compound keys, index fanout, cold
placement, or their snapshot-age interaction. Treating the generic resident encoder or a spreadsheet model as the
canonical write candidate would conceal the missing mutation/index implementation.

## Fail-fast disposition

The remaining dead-density, held-snapshot-age, cold-staging, index-rebuild, durable/apply-lag, sparse-lane, skew,
and pressure-hysteresis permutations were not promoted into a larger performance campaign. The base candidate
already fails the binding low-load and mixed-load matrix, lacks the required row-width/index-fanout surface, and
shows a large current allocation. More permutations can diagnose the losing implementation but cannot satisfy the
physical-selection gate.

At this gate, the evidence therefore reopened the physical choice exactly as the then-proposal required. It did not
prove that append/tombstone was intrinsically incapable; it proved that the ADR could not select it from only the
current implementation and controller evidence. Selection then required the bounded competing representation and
corrected Candidate-A comparison recorded below. Automated adaptation remains necessary—the accepted design's
bounded stage credits, oldest-age deadlines, cut-lag admission, and hysteretic maintenance rules are directionally
correct—but adaptation alone cannot erase the measured low-load FUA floor.

## Subsequent physical-selection correction

The first final independent review correctly rejected using this combined failure while the proposal still named
append/tombstone canonical. [`write-path-adr-physical-selection.md`](write-path-adr-physical-selection.md) now
measures the common conveyor/FUA floor directly and compares append/tombstone with dense-latest/undo at the same
resident GPU boundary across width, fanout, and batch dimensions. That evidence selects compact append/tombstone.
This report's end-to-end and actual-current-allocation **FAIL** remains unchanged under W1 as a production graduation result;
it is no longer treated as proof that no physical design can be selected.

## Reproduction

The principal command shape was:

```text
GPU_DB_BENCH_ARM=driver GPU_DB_BENCH_WRITERS=<drivers> \
GPU_DB_BENCH_SECONDS=<seconds> GPU_DB_BENCH_WINDOW=<window> \
GPU_DB_BENCH_OFFERED_TPS=<rate-or-unset> \
GPU_DB_BENCH_MIX_UPDATE=<percent> GPU_DB_BENCH_MIX_DELETE=<percent> \
GPU_DB_BENCH_TIMELINE=1 GPU_DB_BENCH_RECOVER=1 \
cargo run --release -p gpu_db_engine --example intent_fast_path_bench --features probe-timing
```

The benchmark diff adds scheduled-arrival open-loop pacing, p99.9, collision-free UPDATE/DELETE cursors, corrected
mixed recovery parity, and build-only exact retained-byte reporting. It does not alter the product route.
