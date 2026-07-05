# Write Conveyor Reset

Status: prototype, not wired into the database engine.

The current commit path can be optimized, but it is not the target architecture. It still pays for
SQL text, host MVCC maps, host value-index maintenance, allocation, and commit-time publication.
The write dataplane target starts from a Chronicle/Disruptor-style conveyor:

1. fixed binary write intents;
2. preallocated slots or blocks;
3. producers claim space, write payloads, and publish sequence barriers;
4. consumers advance independent sequence barriers for journal, validation, apply, and visibility;
5. clients wait at the final completion sequence, not at every intermediate stage.

## Prototype

Crate:

```text
crates/write_conveyor
```

Benchmark:

```bash
cargo run --release -p gpu_db_write_conveyor --example write_conveyor_bench
```

Useful knobs:

```text
CONVEYOR_MODE=append|spsc|mpsc|block|both
CONVEYOR_MODE=file-mmap|file-mmap-sync
CONVEYOR_MODE=file-block|file-block-sync
CONVEYOR_MODE=file-block-write|file-block-write-sync
CONVEYOR_MODE=file-block-workers|file-block-workers-sync
CONVEYOR_MODE=file-wal-workers|file-wal-workers-sync
CONVEYOR_MODE=file-wal-payload-workers|file-wal-payload-workers-sync
CONVEYOR_MODE=file-wal-visible-workers|file-wal-visible-workers-sync
CONVEYOR_MODE=file-wal-payload-visible-workers|file-wal-payload-visible-workers-sync
CONVEYOR_MODE=file-wal-staged-workers|file-wal-staged-workers-sync
CONVEYOR_MODE=file-wal-payload-staged-workers|file-wal-payload-staged-workers-sync
CONVEYOR_MODE=file-wal-store-workers|file-wal-store-workers-sync
CONVEYOR_MODE=file-wal-payload-store-workers|file-wal-payload-store-workers-sync
CONVEYOR_EVENTS=100000000
CONVEYOR_PRODUCERS=16
CONVEYOR_WORKERS=16
CONVEYOR_BATCH=1024
CONVEYOR_CAPACITY=1048576
CONVEYOR_FILE=target/write-conveyor-journal-$pid.dat
CONVEYOR_KEEP_FILE=1
CONVEYOR_OVERWRITE=1
CONVEYOR_LATENCY_SAMPLES=4096
```

Modes:

- `append`: pre-faulted append-only 64B intent journal, no tailer.
- `spsc`: single-producer/single-consumer cursor ring.
- `mpsc`: multi-producer/single-consumer per-event sequenced ring.
- `block`: multi-producer/single-consumer block-published ring. Producers publish one sequence per block,
  not per write.
- `file-mmap`: single-threaded append into a preallocated `mmap` journal.
- `file-block`: multi-producer append into a preallocated `mmap` journal with one ordered tailer.
- `file-block-write`: multi-producer append into a preallocated `mmap` journal with no validation consumer;
  this is a raw journal-ingress upper bound.
- `file-block-workers`: multi-producer append into one `mmap` journal with a worker fan-out stage that claims
  published blocks independently. This models validation/apply workers behind a single journal.
- `file-wal-workers`: same worker fan-out shape over a recoverable WAL segment: fixed segment header, fixed block
  stride, block header, payload, block trailer, ordered CRC32C, final commit marker, and a double-buffered durable
  control tail.
- `file-wal-payload-workers`: same single-segment recoverable WAL worker dataplane, but producers publish
  caller-supplied `WriteIntent` slices instead of asking the WAL to synthesize payloads from a sequence range. This is
  the safe real-payload API guardrail, not a manager-throughput measurement.
- `file-wal-visible-workers` and `file-wal-payload-visible-workers`: add a contiguous completion barrier behind
  out-of-order workers. Workers can finish blocks independently, but only the gap-free completed block prefix becomes
  visible. The barrier is block-level, not per-row; the benchmark uses a live waiter thread and reports
  time-to-final-visible-prefix.
- `file-wal-staged-workers` and `file-wal-payload-staged-workers`: split progress into journal/logged and
  apply/visible barriers. Producers publish WAL blocks and mark the logged prefix; apply workers wait on the logged
  prefix before processing; clients wait only on the final applied prefix. In sync variants, this `applied=` timestamp
  is the page-cache/applied prefix; durable visibility still includes the later `sync=` barrier.
- `file-wal-store-workers` and `file-wal-payload-store-workers`: staged WAL apply into an append-only open-shard
  store. The store is fixed-width columnar data with deterministic row-id slots and created sequence values, with no
  host row-key strings. It models the future WAL feeding a shard/store apply target. Store modes validate the stored
  columns after the timed path and report that separately as `store-validate=`.

The file modes create a fresh segment with exclusive `create_new`, `O_NOFOLLOW`, `posix_fallocate`, shared
`mmap`, and pre-touch payload storage before the timed region. The generated default benchmark path may be
removed automatically before a run; caller-specified `CONVEYOR_FILE` paths are not removed unless
`CONVEYOR_OVERWRITE=1` is set. Sync modes include one final full-segment `msync(MS_SYNC)` + `fdatasync`-style
`sync_data` barrier inside the timed region after all publisher/worker threads have exited.

## First Results

Captured 2026-07-04 on this branch, after fixing the benchmark start gate and pre-touching ring
payload storage:

```text
CONVEYOR_MODE=append CONVEYOR_EVENTS=100000000 CONVEYOR_BATCH=1024
append-only   334.011 M/s  2.99 ns/write

CONVEYOR_MODE=block CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=16 CONVEYOR_BATCH=1024
mpsc-block    227.383 M/s  4.40 ns/write

CONVEYOR_MODE=block CONVEYOR_EVENTS=200000000 CONVEYOR_PRODUCERS=16 CONVEYOR_BATCH=1024
mpsc-block    220.340-231.616 M/s  4.32-4.54 ns/write
```

The important result is architectural: append-only and block-published ingress are already in the
hundreds of millions of writes/sec for 64B intents. Per-event publication is not.

Benchmark caveats:

- Ring, `file-mmap`, `file-block`, and `file-block-workers` modes assert a full checksum over every drained
  event; append-only validates first/last only.
- `file-block-workers` runs a full checksum, but block workers may process journal blocks out of order; ordered
  visibility still needs a separate contiguous completion barrier.
- `file-block-write` performs no validation checksum and exists only to isolate raw journal-ingress throughput.
- `file-wal-workers-sync` validates recovery by reopening the segment and scanning the committed prefix outside the
  hot-path elapsed time; the recovery time is reported separately.
- `WalSegmentManager` adds the control-plane layer above a single recoverable segment: rolling segment files,
  a double-buffered external control file, restart recovery across the retained segment range, and a retention guard.
  It is intentionally not in the worker hot-path benchmark yet; the benchmark still measures the single-segment
  journal/worker dataplane directly.
- File-mode setup/preallocation/pre-touch is reported but outside the hot-path timed region, matching the intended
  production model of preparing roll/cycle files off the hot path.
- Non-sync file modes measure writes reaching the OS page cache/mapped file, not durable commits.
- `CONVEYOR_LATENCY_SAMPLES` enables optional block-level latency sampling for recoverable WAL modes. It is off by
  default so throughput guardrails stay comparable. When enabled, producers timestamp every block before the WAL
  publish call and retain timestamps only for sampled block ids; workers timestamp sampled blocks after the block
  reaches that mode's completion stage. The report prints p50/p90/p99/max and sample counts for `publish->logged`,
  `logged-><completion>`, `publish-><completion>`, and, in sync modes, `publish->final-sync`. These are block-unit
  latencies, not per-row timestamps; with `CONVEYOR_BATCH=1024`, one sampled block represents 1024 writes.
  `publish->final-sync` is a closed-burst measurement to the benchmark-wide final sync fence, not an independent
  per-block fsync.
- The safe API intentionally exposes only whole-batch publish operations; raw claim/publish sequence ownership
  is not public because duplicate publication over `UnsafeCell` payloads would be unsound.
- The mmap constructors are `unsafe`: callers must guarantee exclusive ownership of the backing file for the
  mapping lifetime. The payload-only `file-block*` modes are not recoverable WALs; they intentionally lack
  on-disk block headers/trailers, committed tail metadata, and per-block checksums.

`file-wal-workers` closes most of that prototype gap for a single segment: it has recoverable block metadata,
per-block CRC32C, a trailer commit marker, and a durable control record that bounds recovery to the synced prefix.
`WalSegmentManager` now layers segment rolling, an external durable control file, restart-safe retained-range
recovery, and retention metadata over those segments. The segment and manager APIs can now publish real
caller-supplied `WriteIntent` payload blocks; synthetic sequence publishing remains only as a stable benchmark
guardrail. It still lacks replication records and engine replay integration.

## File-Backed Results

Captured 2026-07-04 on XFS over NVMe (`/home/richard/projects`, 64B intents), after mapped-file
preallocation, worker fan-out, full checksum validation, and the safety audit fixes:

```text
CONVEYOR_MODE=append CONVEYOR_EVENTS=100000000
append-only          330.718 M/s  3.02 ns/write

CONVEYOR_MODE=block CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=16 CONVEYOR_BATCH=1024
mpsc-block           222.382 M/s  4.50 ns/write

CONVEYOR_MODE=file-mmap CONVEYOR_EVENTS=100000000
file-mmap            127.618 M/s  7.84 ns/write  setup=0.760s write=0.784s

CONVEYOR_MODE=file-block CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=16 CONVEYOR_BATCH=1024
file-block           235.065 M/s  4.25 ns/write  setup=0.776s produce=0.112s drain=0.425s

CONVEYOR_MODE=file-block-write CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_BATCH=1024
file-block-write    1060.112 M/s  0.94 ns/write  setup=0.796s produce=0.094s validation=none

CONVEYOR_MODE=file-block-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-workers         725.558 M/s  1.38 ns/write  setup=0.839s produce=0.111s workers=0.138s

CONVEYOR_MODE=file-block-workers CONVEYOR_EVENTS=200000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-workers         446.249 M/s  2.24 ns/write  setup=1.570s produce=0.434s workers=0.448s

CONVEYOR_FILE=/tmp/... CONVEYOR_MODE=file-block-workers CONVEYOR_EVENTS=200000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-workers         716.792 M/s  1.40 ns/write  setup=9.216s produce=0.254s workers=0.279s

CONVEYOR_MODE=file-block-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-workers-sync     72.756 M/s 13.74 ns/write  setup=0.888s produce=0.128s workers=0.140s sync=1.203s full-segment-sync

CONVEYOR_MODE=file-wal-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-wal-workers     731.778 M/s  1.37 ns/write  setup=0.749s produce=0.119s workers=0.137s

CONVEYOR_MODE=file-wal-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-wal-workers-sync 72.867 M/s 13.72 ns/write  setup=0.865s produce=0.143s workers=0.143s sync=1.198s recover=2.103s range-sync

CONVEYOR_MODE=file-wal-workers CONVEYOR_EVENTS=200000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-wal-workers     386.721 M/s  2.59 ns/write  setup=1.726s produce=0.509s workers=0.517s

CONVEYOR_FILE=/tmp/... CONVEYOR_MODE=file-wal-workers CONVEYOR_EVENTS=200000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8
file-wal-workers     692.980 M/s  1.44 ns/write  setup=8.899s produce=0.284s workers=0.289s
```

Latest Step 1 guardrail run after adding `WalSegmentManager` retained-range recovery and audit fixes
(2026-07-04):

```text
CONVEYOR_MODE=file-wal-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-workers      531.119 M/s  1.88 ns/write  setup=0.756s produce=0.188s workers=0.188s

CONVEYOR_MODE=file-wal-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-workers-sync  66.971 M/s 14.93 ns/write  setup=0.765s produce=0.140s workers=0.140s sync=1.323s recover=2.172s
```

Latest Step 2 guardrail run after adding real payload-slice publishing (2026-07-04):

```text
CONVEYOR_MODE=file-wal-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-workers       639.867 M/s  1.56 ns/write  setup=0.752s produce=0.156s workers=0.156s

CONVEYOR_MODE=file-wal-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-workers-sync   66.750 M/s 14.98 ns/write  setup=0.786s produce=0.116s workers=0.136s sync=1.328s recover=2.166s

CONVEYOR_MODE=file-wal-payload-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload       732.135 M/s  1.37 ns/write  setup=0.776s produce=0.131s workers=0.137s

CONVEYOR_MODE=file-wal-payload-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-sync   72.714 M/s 13.75 ns/write  setup=0.773s produce=0.138s workers=0.138s sync=1.204s recover=2.181s
```

Latest Step 3 guardrail run after adding the contiguous completion/visibility barrier with a live visibility waiter
(2026-07-04):

```text
CONVEYOR_MODE=file-wal-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-workers             730.615 M/s  1.37 ns/write  setup=0.740s produce=0.113s workers=0.137s

CONVEYOR_MODE=file-wal-visible-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-visible             589.581 M/s  1.70 ns/write  setup=0.805s produce=0.169s workers=0.170s visible=0.170s

CONVEYOR_MODE=file-wal-payload-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload             727.254 M/s  1.38 ns/write  setup=0.772s produce=0.125s workers=0.138s

CONVEYOR_MODE=file-wal-payload-visible-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-visible     699.299 M/s  1.43 ns/write  setup=0.792s produce=0.143s workers=0.143s visible=0.143s

CONVEYOR_MODE=file-wal-payload-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-sync         56.158 M/s 17.81 ns/write  setup=0.779s produce=0.154s workers=0.154s sync=1.602s recover=2.098s

CONVEYOR_MODE=file-wal-payload-visible-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-visible-sync 57.887 M/s 17.27 ns/write  setup=0.761s produce=0.139s workers=0.139s visible=0.139s sync=1.560s recover=2.104s
```

Latest Step 4 guardrail run after adding the staged journal/apply conveyor (2026-07-04):

```text
CONVEYOR_MODE=file-wal-payload-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload             726.600 M/s  1.38 ns/write  setup=0.755s produce=0.137s workers=0.138s

CONVEYOR_MODE=file-wal-payload-visible-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-visible     710.200 M/s  1.41 ns/write  setup=0.801s produce=0.135s workers=0.141s visible=0.141s

CONVEYOR_MODE=file-wal-payload-staged-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-staged      713.643 M/s  1.40 ns/write  setup=0.755s produce=0.140s workers=0.140s applied=0.140s

CONVEYOR_MODE=file-wal-payload-staged-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-staged-sync  58.310 M/s 17.15 ns/write  setup=0.847s produce=0.187s workers=0.196s applied=0.196s sync=1.483s recover=2.100s

CONVEYOR_MODE=file-wal-staged-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-staged              707.897 M/s  1.41 ns/write  setup=0.763s produce=0.140s workers=0.141s applied=0.141s
```

Latest Step 5 guardrail run after adding the append-only open-shard store (2026-07-04):

```text
CONVEYOR_MODE=file-wal-payload-store-workers CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-store       378.537 M/s  2.64 ns/write  setup=0.763s produce=0.136s workers=0.264s applied=0.264s store-validate=1.434s

CONVEYOR_MODE=file-wal-payload-store-workers-sync CONVEYOR_EVENTS=100000000 CONVEYOR_PRODUCERS=32 CONVEYOR_WORKERS=8 CONVEYOR_BATCH=1024
file-wal-payload-store-sync   52.676 M/s 18.98 ns/write  setup=0.779s produce=0.162s workers=0.275s applied=0.275s sync=1.591s recover=2.107s store-validate=1.378s
```

Latency sample run with `CONVEYOR_LATENCY_SAMPLES=4096` over the same 100M-write, 32-producer, 8-worker,
1024-write-block configuration (2026-07-05):

```text
CONVEYOR_MODE=file-wal-payload-staged-workers
file-wal-payload-staged       664.805 M/s  1.50 ns/write  elapsed=0.150s
latency-samples requested=4096 block-stride=24
publish->logged   samples=4070 p50=0.030ms p90=0.051ms p99=0.071ms max=0.383ms
logged->applied   samples=4070 p50=0.024s  p90=0.033s  p99=0.034s  max=0.034s
publish->applied  samples=4070 p50=0.024s  p90=0.033s  p99=0.034s  max=0.034s

CONVEYOR_MODE=file-wal-payload-staged-workers-sync
file-wal-payload-staged-sync   56.078 M/s 17.83 ns/write  elapsed=1.783s
latency-samples requested=4096 block-stride=24
publish->logged      samples=4070 p50=0.029ms p90=0.042ms p99=0.069ms max=0.401ms
logged->applied      samples=4070 p50=0.040s  p90=0.049s  p99=0.050s  max=0.050s
publish->applied     samples=4070 p50=0.040s  p90=0.050s  p99=0.050s  max=0.050s
publish->final-sync  samples=4070 p50=1.699s  p90=1.742s  p99=1.752s  max=1.753s

CONVEYOR_MODE=file-wal-payload-store-workers
file-wal-payload-store        306.697 M/s  3.26 ns/write  elapsed=0.326s
latency-samples requested=4096 block-stride=24
publish->logged        samples=4070 p50=0.032ms p90=0.063ms p99=0.082ms max=0.143ms
logged->store-applied  samples=4070 p50=0.131s  p90=0.166s  p99=0.168s  max=0.168s
publish->store-applied samples=4070 p50=0.131s  p90=0.166s  p99=0.168s  max=0.168s

CONVEYOR_MODE=file-wal-payload-store-workers-sync
file-wal-payload-store-sync    52.605 M/s 19.01 ns/write  elapsed=1.901s
latency-samples requested=4096 block-stride=24
publish->logged        samples=4070 p50=0.030ms p90=0.052ms p99=0.074ms max=0.140ms
logged->store-applied  samples=4070 p50=0.127s  p90=0.170s  p99=0.174s  max=0.174s
publish->store-applied samples=4070 p50=0.127s  p90=0.170s  p99=0.174s  max=0.174s
publish->final-sync    samples=4070 p50=1.811s  p90=1.861s  p99=1.872s  max=1.873s
```

These samples measure closed-burst block residence through the conveyor, not isolated single-client request
latency. The WAL publish/logged stage is tens of microseconds at p99 for 1024-write blocks. The non-durable applied
tail is queue residence behind the worker/apply stage: staged-only p99 was ~34-50ms, and store apply p99 was
~168-174ms. The sync variants show the final durability fence directly: publish-to-final-sync p99 was ~1.75-1.87s
for the full 6.4GB burst.

Interpretation:

- A single ordered tailer is a real stage bottleneck: producers finish 100M mapped writes in ~112ms, while the
  single consumer drains until ~425ms.
- Worker fan-out lifts the validated mapped path from ~235M/s to ~726M/s for a 6.4GB burst.
- The recoverable WAL segment keeps the 100M validated hot path in the same band (~732M/s) despite writing block
  headers/trailers, CRC32C checksums, and the durable-tail-ready layout.
- The 12.8GB XFS/NVMe run falls to ~446M/s while the same run on tmpfs stays ~717M/s. The drop is dirty-page
  writeback/storage pressure, not the conveyor algorithm.
- The same 12.8GB WAL-segment comparison is ~387M/s on XFS/NVMe and ~693M/s on tmpfs.
- The final sync barrier is the durability wall: for 100M records, worker processing finishes in ~140ms, but
  full-segment `msync` + `sync_data` adds ~1.20s. Group commit can hide stage latency before this point, but
  durable commit throughput is bounded by the device.
- The append-only store path is stricter than WAL ingress: workers read WAL payload blocks and write fixed-width
  store columns, so 100M records take ~264ms (~379M/s). That is the first future-WAL-to-store lower bound.
- WAL range-sync for 100M records took ~1.20s, and recovery scan of the 6.4GB committed prefix took ~2.10s with
  ordered CRC32C verification.

## Next Build-Up

1. **Done locally:** add segment rolling, retention metadata, and a small external control file for the synced
   retained range across multiple segment files.
2. **Done locally:** replace synthetic-only `WriteIntent` generation with a real payload publish API: prepared route
   id + typed params as fixed binary `WriteIntent` slices, while keeping the synthetic benchmark path as a stable
   throughput guardrail.
3. **Done locally:** add a contiguous completion/visibility barrier behind the worker fan-out stage. Workers may
   finish blocks out of order, but commits become visible only when every prior block has completed.
4. **Done locally:** add the staged journal/apply conveyor: journal publishes a logged sequence, apply waits on that
   sequence, completion waits on apply, and clients block only at the final completion point.
5. **Done locally:** add an append-only open-shard store: fixed-width columns, row ids, created sequence, no host
   row-key strings.

After those five slices, introduce validation/index maintenance, preferably as block/GPU stages.

Do not wire this into the SQL commit path until the staged conveyor keeps the same order of magnitude.
