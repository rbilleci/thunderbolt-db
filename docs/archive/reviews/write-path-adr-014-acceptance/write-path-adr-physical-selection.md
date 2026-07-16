# R3-001 physical-representation selection report

> Archived acceptance-process evidence. Non-actionable; current work lives only in `docs/PLAN.md`.

**Date:** 2026-07-15\
**Source:** baseline `f701d8b6` plus the review-only diff recorded in the next frozen packet\
**Host:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition; local synchronous-FUA WAL device\
**Disposition:** **SELECT CANDIDATE A — compact append/tombstone**

This report resolved the physical-selection blocker from the first final independent review. It did not itself
accept ADR-014 and does not claim that the current engine implementation meets the product SLO or that the compact
layout already serves durable state. The prototype is build-only decision evidence and never migrates or serves
engine state.

## Correct decision boundary

The rejected current-path matrix combined three separable costs:

1. the physical row/version representation;
2. the existing controller, index, allocation, and route restrictions; and
3. the storage device's synchronous-FUA latency.

A representation cannot reduce a durability fence that both candidates require. The fixed-record
`FuaWalSegment` client harness—with the same prewritten-extent, FUA fence-pool, and contiguous-cut physics as the
engine-facing variable-payload `FuaFrameLog`, but not itself the relational lane—measured one client, queue depth
one, one record per fence, and no append grouping over 4,000 synchronous acknowledgements at **1.662 ms p50,
1.723 ms p99, and 5.612 ms maximum**. A queue-depth-16 run over 20,000 records measured its raw FUA writes at
**2.433 ms p50 and 2.951 ms p99**.

The actual engine-facing `FuaFrameLog` was measured separately at queue depth one: 4,000 variable-payload frames
completed in 6.169 seconds, or **1.542 ms per fence on average**. `FuaWalLaneSet` composes those frame logs and adds
global lane-prefix coordination; the existing relational Candidate-A low-load path measured 2.88-ms p50. The
fixed-record percentile distribution is therefore supporting fence-physics attribution, not mislabeled production
lane evidence. Together, the actual frame-log rate and the same-physics distribution establish a
representation-independent millisecond durability envelope before SQL, validation, GPU apply, publication, or
response work is added.

That result does not weaken synchronous commit or the SLO. It establishes two distinct gates:

- **R3-001 design selection:** compare candidate-specific resident GPU mechanics and retained representation bytes
  after removing the common durability floor.
- **Production graduation:** run the complete open-loop synchronous matrix on the canonical implementation and
  qualifying durability hardware. If the observed fence floor plus fixed engine margins cannot fit the declared
  latency budget, the deployment must fail SLO qualification; the engine must not silently switch to asynchronous
  acknowledgement or advertise the low-latency profile.

## Bounded GPU A/B

[`write_path_candidate_ab.rs`](../../crates/execution/examples/write_path_candidate_ab.rs) executes resident-input
CUDA kernels for the two candidates:

- **A — append/tombstone:** copy the complete new image to append storage, stamp the old version's death, write
  creation/row identity plus the new live-death sentinel, publish latest heads, and emit one 32-byte
  history-index record per maintained index/version.
- **B — dense latest plus undo:** mark a per-row seqlock odd, copy the complete old image to undo, overwrite the
  dense latest image, copy the old creation/death interval and row identity to undo, install the new creation/death
  interval, make the seqlock even, publish latest heads, and emit the same index history records. Global-memory
  fences follow the odd marker and precede the even marker; the undo death is the replacement commit, not the old
  live sentinel.

The probe uses one million logical rows, unique update targets within a wave, fixed row widths of 8/32/128 bytes,
index fanout of 1/3/6, and batch sizes of 1/256/4,096. Inputs and destinations are already device resident. Each
sample includes launch plus stream synchronization; three consecutive complete runs were taken. This deliberately
measures the representation-dependent device mutation rather than the common WAL, parsing, admission, or response
path.

### Stable median result

| Row width | A p50 range across cells/runs | B p50 range across cells/runs | B/A p50 range | Result |
|---:|---:|---:|---:|---|
| 8 B | 6.37–13.41 us | 7.56–15.18 us | 1.13–1.28x | A faster in every cell |
| 32 B | 7.29–14.48 us | 9.37–18.33 us | 1.25–1.47x | A faster in every cell |
| 128 B | 10.96–18.27 us | 16.61–29.88 us | 1.50–1.95x | A faster in every cell |

At batch one—the latency-oriented point—B is approximately 18–19% slower for 8-byte rows, 28–29% slower for
32-byte rows, and 50–52% slower for 128-byte rows. Host/driver p99.9 samples contain isolated outliers in both
directions, so they are not used to manufacture a tail winner. In the ordinary range, both candidate kernels are
tens of microseconds or less and are far below the measured millisecond durability floor. The repeatable p50 result
and the increasing width penalty are the relevant representation signals.

The probe also verifies the first appended/latest value, A's matching old-death/new-creation stamps and live new
version, and B's complete undo interval, live new interval, and even seqlock after the run. Explicit visibility
assertions prove the undo version visible and the dense current version hidden at the old snapshot, then the undo
version hidden and current version visible at the replacement snapshot. It does not prove reader reconstruction,
recovery, compaction, or production concurrency for B; those extra mechanisms are costs B would introduce, not
unimplemented benefits credited to A.

## Exact prototype footprint and snapshot-age envelope

The comparison uses explicit logical records, excluding allocator rounding and scratch:

| Component | Candidate A | Candidate B |
|---|---:|---:|
| Retained history per update | row width + 24 B (`row_id`, `created_by`, `deleted_by`) | row width + 24 B (undo `row_id`, `created_by`, `deleted_by`) |
| Latest-head cells | 8 B per logical row per maintained index | same |
| Historical index record | 32 B per maintained index/version | same |
| Current row | row width + 24 B (row identity plus creation/death interval) | row width + 24 B (implicit dense row slot plus creation/death interval and seqlock) |

With a version-history record for every maintained index, retained history is therefore:

| Width | 1 index A / B | 3 indexes A / B | 6 indexes A / B |
|---:|---:|---:|---:|
| 8 B | 64 / 64 B | 128 / 128 B | 224 / 224 B |
| 32 B | 88 / 88 B | 152 / 152 B | 248 / 248 B |
| 128 B | 184 / 184 B | 248 / 248 B | 344 / 344 B |

The semantically complete bounded formats are byte-tied. An earlier draft incorrectly credited B with an 8-byte
history saving by omitting part of its visibility/ownership metadata; the kernel and report now include the complete
interval and row ownership. At 100,000 updates/s, the retained-history envelope ranges from 6.4 MB after one second
for an 8-byte row with one index to 123.84 GB after one hour for a 128-byte row with six indexes. B therefore has no
bounded-format footprint advantage to offset its measured mutation cost or its additional atomic-overwrite, undo
lookup, old-snapshot reconstruction, checkpoint, and recovery machinery.

These are exact bytes for the bounded candidate formats, not forecasts of allocator capacity or compression. The
canonical implementation must report actual resident allocation, cold-artifact, scratch, WAL, and index geometry
under R3-002/003 and DUR-001/002. Snapshot-fenced history may move to device-format STRATA artifacts, but neither
candidate can discard it; hard resident and cold quotas must backpressure or reject before WAL when the retained
horizon cannot fit.

## Decision

Select **compact append/tombstone** as the canonical physical MVCC representation. Dense-latest plus undo is
rejected because its semantically complete bounded format is byte-tied while its GPU mutation is consistently
slower and it requires materially more read, publication, checkpoint, and recovery machinery. The current
approximately 816-B narrow-row allocation is not accepted as the compact layout; it is implementation debt that the
canonical implementation must replace and measure.

The selection preserves one logical row/version/visibility law across resident and cold STRATA placement. It also
preserves the production conveyor boundary: the CPU sequences, fences WAL, and orchestrates; the GPU performs
locate, exact validation, append/death/index mutation, and publication-ready state construction.

## Reproduction

```text
CONVEYOR_CLIENTS=1 CONVEYOR_CLIENT_DRIVER_THREADS=1 CONVEYOR_FUA_QD=1 \
CONVEYOR_EVENTS=4000 CONVEYOR_CLIENT_APPEND_GROUP_US=0 \
timeout 180 target/release/examples/fua_wal_client_bench

CONVEYOR_CLIENTS=1 CONVEYOR_CLIENT_DRIVER_THREADS=1 CONVEYOR_FUA_QD=16 \
CONVEYOR_EVENTS=20000 CONVEYOR_CLIENT_APPEND_GROUP_US=0 \
timeout 180 target/release/examples/fua_wal_client_bench

CONVEYOR_FUA_QD=1 CONVEYOR_EVENTS=4000 \
timeout 180 target/release/examples/fua_frame_log_bench

rustfmt --edition 2021 crates/execution/examples/write_path_candidate_ab.rs
cargo check -p gpu_db_execution --example write_path_candidate_ab
timeout 300 cargo run --quiet --release -p gpu_db_execution \
  --example write_path_candidate_ab
```

Never use `--gpu-reset`.
