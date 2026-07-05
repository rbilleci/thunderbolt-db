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

Raw WAL durability isolator:

```bash
cargo run --release -p gpu_db_write_conveyor --example raw_wal_durability_bench
cargo run --release -p gpu_db_write_conveyor --example raw_direct_wal_bench
cargo run --release -p gpu_db_write_conveyor --example raw_uring_wal_bench
```

FUA-pipelined durable client lane (see "FUA-Pipelined Durable Lane" below):

```bash
CONVEYOR_CLIENTS=2048 CONVEYOR_FUA_QD=16 CONVEYOR_EVENTS=4000000 \
  cargo run --release -p gpu_db_write_conveyor --example fua_wal_client_bench
# extra knobs: CONVEYOR_CLIENT_WAL_BLOCK=62|126 (512B-aligned strides only),
# CONVEYOR_CLIENT_MIN_RECORDS_PER_BLOCK (frame floor; default scales with clients/qd),
# CONVEYOR_RAW_DIRECT_PREWRITE / CONVEYOR_RAW_DIRECT_QD for raw_direct_wal_bench
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
CONVEYOR_MODE=file-wal-client-logged|file-wal-client-store-applied|file-wal-client-durable
CONVEYOR_MODE=file-wal-client-coalesced-logged|file-wal-client-coalesced-store-applied|file-wal-client-coalesced-durable
CONVEYOR_MODE=file-wal-manager-coalesced-logged|file-wal-manager-coalesced-store-applied|file-wal-manager-coalesced-durable
CONVEYOR_EVENTS=100000000
CONVEYOR_PRODUCERS=16
# CONVEYOR_WORKERS=16            # optional override; worker modes default to producers, client-latency modes default to 1
CONVEYOR_BATCH=1024
CONVEYOR_CLIENTS=16
CONVEYOR_CLIENT_WAL_BLOCK=1        # direct client modes default; coalesced modes default to 64
CONVEYOR_CLIENT_APPEND_RING=65536  # coalesced only; rounded up to a power of two and at least clients
CONVEYOR_CLIENT_APPEND_GROUP_US=25 # coalesced client modes only
CONVEYOR_CLIENT_MIN_RECORDS_PER_BLOCK=16 # coalesced effective min is clamped by block size and clients
CONVEYOR_CLIENT_LATENCY_SAMPLES=1000000
CONVEYOR_CLIENT_WAIT_SPINS=0
CONVEYOR_CLIENT_DRIVER_THREADS=$CONVEYOR_CLIENTS # coalesced only; lower values multiplex logical clients
CONVEYOR_CLIENT_DRIVER_ISSUE_BUDGET=1 # multiplexed coalesced only; max new requests issued per driver loop
CONVEYOR_STAGE_TIMINGS=0          # set to 1 for detailed coalesced stage/timeline timing; adds benchmark overhead
CONVEYOR_BACKGROUND_DURABLE=0    # coalesced only; keep advancing durable cut while logged/store-applied clients ack early
CONVEYOR_DURABLE_GROUP_US=25
CONVEYOR_DURABLE_MIN_BLOCKS=0     # coalesced durable: 0=auto, otherwise fixed block threshold
CONVEYOR_DURABLE_LANES=1          # manager-backed coalesced durable/background-durable only; N>1 stripes WAL sync across lanes
CONVEYOR_DURABLE_SYNC_MODE=write-and-file-data # default; range-and-file-data/prewrite-and-file-data/sync-write-data/file-data-only are labeled probes
CONVEYOR_WAL_MANAGER_RECORDS_PER_SEGMENT=262144 # manager-backed coalesced default at block=64
CONVEYOR_RAW_WAL_BLOCK=64                      # raw_wal_durability_bench exact record count; 62 gives a 4 KiB frame
CONVEYOR_RAW_SYNC_MODES=sync-write-data        # comma list for raw_wal_durability_bench
CONVEYOR_RAW_FENCE_BLOCKS=1                   # comma list; WAL blocks per raw durability fence
CONVEYOR_RAW_FENCE_KIND=data                  # data or prefix; prefix ignores sync modes and prints sync_mode=n/a
CONVEYOR_RAW_VALIDATE=1                       # raw benchmark scan-recovers and validates committed blocks
CONVEYOR_RAW_DIRECT_MODES=direct-dsync        # raw_direct_wal_bench: direct-dsync/direct-fdatasync/direct-rwf-dsync
CONVEYOR_RAW_DIRECT_ALIGN=4096                # direct WAL buffer, offset, and frame alignment
CONVEYOR_RAW_URING_MODES=uring-rwf-dsync      # raw_uring_wal_bench: also uring-write-fdatasync and fixed-buffer variants
CONVEYOR_RAW_URING_ENTRIES=8                  # io_uring submission/completion ring entries
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
- `file-wal-client-logged`, `file-wal-client-store-applied`, and `file-wal-client-durable`: client-perspective
  closed-loop latency modes. Logical clients publish one write intent per request and record latency from client
  ingress. `client-logged` acks when the WAL block is published; `client-store-applied` waits for ordered append-store
  apply; `client-durable` waits for ordered append-store apply plus a durable WAL data frontier. The logged and
  store-applied ack points are non-durable lower bounds. These modes default to `CONVEYOR_CLIENT_WAL_BLOCK=1`, the
  latency lane WAL block capacity and stride, not "multiple writes per request." Store waits use bounded yielding by
  default (`CONVEYOR_CLIENT_WAIT_SPINS=0`) because hundreds of spinning client waiters can starve apply workers.
  Durable waits use a small group window by default (`CONVEYOR_DURABLE_GROUP_US=25`) to catch nearby requests without
  letting p90 drift. Client-latency modes default to one apply worker because measured durable runs are fence-bound;
  larger apply-worker pools mainly add scheduler pressure unless store apply becomes the bottleneck.
- `file-wal-client-coalesced-logged`, `file-wal-client-coalesced-store-applied`, and
  `file-wal-client-coalesced-durable`: Chronicle-style OLTP client modes. Clients still issue one logical write at a
  time, but they enqueue into a bounded reusable ring and a single WAL owner coalesces ready singleton requests into
  sequential WAL blocks (`CONVEYOR_CLIENT_WAL_BLOCK=64` by default, `CONVEYOR_CLIENT_APPEND_GROUP_US=25`). Per-request
  acks wait on the assigned WAL block, so this is still a client-latency lane rather than a bulk-load lane. The appender
  drains the ring in sequence order; this preserves deterministic WAL order, but a preempted producer between sequence
  claim and publish can still create a head-of-line stall in this benchmark. Production should make the sequencer own
  reservation/publish or add cancellation semantics. Coalesced durable mode uses a pressure-aware sync threshold by
  default (`CONVEYOR_DURABLE_MIN_BLOCKS=0`): fast/in-memory syncs are fenced immediately, while storage-scale syncs
  raise the minimum block threshold and fall back to the short durable group deadline under sparse traffic. The default
  durable fence mode descriptor-writes the WAL byte range and then calls `sync_data`
  (`CONVEYOR_DURABLE_SYNC_MODE=write-and-file-data`), which beat the mmap `msync(MS_SYNC)` range path on the
  filesystem-backed OLTP latency lane while preserving scan-recoverable durable WAL bytes.
  `CONVEYOR_BACKGROUND_DURABLE=1` applies the Chronicle queue split explicitly: logged/store-applied client modes can
  acknowledge at their visible cut while a background worker continues advancing the requested durable prefix. The
  benchmark reports `chronicle-visible-cuts` at the client-visible boundary, `chronicle-final-cuts` after durable
  catch-up, visible elapsed/throughput, total elapsed including durable catch-up, and scan-recovery validation for the
  final durable prefix. The primary `M/s` line uses total-with-durable elapsed in this mode so it does not hide durable
  backlog. This is not an RPO-0 client commit mode; strict durable clients still use `file-wal-*-coalesced-durable` and
  wait for the durable cut.
  The benchmark can also probe descriptor-written WAL bytes plus `sync_data`
  (`CONVEYOR_DURABLE_SYNC_MODE=write-and-file-data`), background descriptor prewrite before the final `sync_data`
  (`CONVEYOR_DURABLE_SYNC_MODE=prewrite-and-file-data`), Linux-only `pwritev2(RWF_DSYNC)` sync-on-write
  (`CONVEYOR_DURABLE_SYNC_MODE=sync-write-data`), and Linux `sync_data`/fdatasync-only behavior
  (`CONVEYOR_DURABLE_SYNC_MODE=file-data-only`); do not compare those numbers without labeling the mode.
  `CONVEYOR_CLIENT_DRIVER_THREADS` can multiplex many logical clients onto fewer async driver threads; when it is lower
  than `CONVEYOR_CLIENTS`, `CONVEYOR_CLIENT_DRIVER_ISSUE_BUDGET` paces how many new requests each driver injects before
  polling completions again. Sampled latency starts when a logical client becomes ready, so async driver scheduling and
  issue-budget delay are included in `client->logged` and final ack latency. This probes whether a production event-loop
  client shape reduces scheduler overhead or durable-fence burstiness versus one OS thread per logical closed-loop client.
  Low-client closed-loop runs cannot fill a requested minimum larger than the active client count; the benchmark prints
  both `requested-min-block` and effective `min-block`. It also prints `block-budget=compact|sparse-safe`; sparse-safe
  mode intentionally budgets one WAL block per event for liveness in low-concurrency or zero-delay stress cases.
  In compact mode, the appender lowers the current target only as clients finish their final request, so a short append
  deadline cannot create an unbounded stream of under-minimum blocks.
- `file-wal-manager-coalesced-logged`, `file-wal-manager-coalesced-store-applied`, and
  `file-wal-manager-coalesced-durable`: same coalesced OLTP client contract, but publishing goes through
  `WalSegmentManager` instead of one large segment. The manager rolls segment files, returns a cloneable published-block
  handle for asynchronous apply/durable workers, and durable validation uses scan recovery across the manager's segment
  directory. `CONVEYOR_DURABLE_LANES=N` can stripe durable manager mode across N independent WAL manager directories:
  the appender assigns global WAL blocks round-robin, each lane syncs independently, and clients see completion only
  through one global contiguous durable prefix. This probes the big bet that storage sync latency can be hidden with
  multiple Chronicle-style append lanes without weakening ordered commit visibility. Current same-device measurements
  are negative: the extra lanes increase sync contention and global-prefix waiting. Recovery validation for striped
  lanes reconstructs the clean-run global prefix from lane-local block counts, but production replay would still need a
  durable global cut or persisted merge metadata before faster-lane blocks beyond a slower-lane gap can be made visible.
  Background prewrite is still single-lane only. This is the production-shaped WAL basis; the benchmark still uses the
  prototype client ring rather than the engine SQL commit path.

The file modes create a fresh segment with exclusive `create_new`, `O_NOFOLLOW`, `posix_fallocate`, and shared
`mmap`. Non-WAL payload benchmarks may pre-touch payload storage before the timed region; the WAL segment path now
does not zero the whole mapping, because doing so dirties every page and turns the first durability fence into a
segment-sized flush. The generated default benchmark path may be removed automatically before a run; caller-specified
`CONVEYOR_FILE` paths are not removed unless `CONVEYOR_OVERWRITE=1` is set. Sync modes include one final full-segment
`msync(MS_SYNC)` + `fdatasync`-style `sync_data` barrier inside the timed region after all publisher/worker threads
have exited. The client durable mode is different: it uses a background durable frontier that waits for the contiguous
published WAL prefix, then applies the selected `CONVEYOR_DURABLE_SYNC_MODE`: the default `write-and-file-data` mode
rewrites the block bytes through the file descriptor before `sync_data`; `range-and-file-data` flushes the mmap-written
block range with `msync(MS_SYNC)` before `sync_data`; `prewrite-and-file-data` does the descriptor write on a background
prewrite stage in coalesced modes before the durable worker's final `sync_data` (direct modes fall back to the same
inline descriptor-write fence); `sync-write-data` writes the range with Linux `pwritev2(RWF_DSYNC)` and does not issue a
separate `sync_data`; `file-data-only` issues only `sync_data`. The same-process recovery check reopens and scans block
headers/trailers/CRC commit markers; it validates scanner behavior and kernel/filesystem support for the requested
syscall path, not power-fail stable-media semantics. Treat `sync-write-data` and `file-data-only` as labeled probes until
they pass target filesystem/kernel crash-validation. In that mode the double-buffered control tail is a
checkpoint/expected-frontier path, not the per-client latency gate.

`raw_wal_durability_bench` bypasses the client ring, appender, store apply, and manager. It publishes directly into one
recoverable WAL segment, fences every `CONVEYOR_RAW_FENCE_BLOCKS` blocks, prints publish/fence/group p50/p90/p99, then
scan-recovers the file when `CONVEYOR_RAW_VALIDATE=1`. Use it to separate software pipeline latency from
filesystem/device durability latency. Unlike the client-lane block knob, `CONVEYOR_RAW_WAL_BLOCK` is an exact record
count and is not rounded to a power of two; this allows frame-aligned WAL layouts such as 62 records
(`64B header + 62*64B payload + 64B trailer = 4096B`).
`CONVEYOR_RAW_FENCE_KIND=data` measures the data-frontier fence used by the client durable path; `prefix` measures the
full durable-prefix/control-tail path, ignores `CONVEYOR_RAW_SYNC_MODES`, prints `sync_mode=n/a`, and is intentionally
much more expensive.

`raw_direct_wal_bench` writes the same scan-recoverable WAL block format through aligned direct-I/O buffers. It requires
`WAL_SEGMENT_HEADER_BYTES` and the WAL block stride to be aligned to `CONVEYOR_RAW_DIRECT_ALIGN`; for 4096B alignment,
use exact raw block counts such as 62 (`4096B` frame) or 126 (`8192B` frame). It validates by the same
`recover_wal_segment_by_scan` path.

`raw_uring_wal_bench` writes the same scan-recoverable WAL block format through Linux `io_uring`. It currently probes a
synchronous one-operation-at-a-time shape: `IORING_OP_WRITE` with `RWF_DSYNC`, write+blocking `fdatasync`, and
registered-buffer `WriteFixed` variants. This isolates per-fence behavior, but it does not test async fsync opcodes,
linked write+fsync chains, SQPOLL, registered files, or a deeper queued backend. Any production io_uring backend should
beat the page-cache raw baseline in its own raw probe before promotion into the conveyor or engine commit path.

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

## Client Latency Slice

The closed-burst block samples above are useful for saturation and queue-residence analysis, but they are not a real
client latency measurement. The client latency modes timestamp each logical request at client ingress and wait for the
selected ack point. They are lower bounds for SQL client latency: they intentionally exclude SQL parse/plan, protocol
I/O, constraint lookup, update target lookup, and index maintenance. The non-durable modes also exclude durable group
commit. `file-wal-client-durable` includes a prototype durable WAL frontier, but the in-process store is still volatile:
the ack means "the WAL prefix has passed the selected `durable-sync-mode` fence and the live store has applied it", not
"the store has been replayed after restart" or "power-fail stable media has been independently proven".

Captured 2026-07-05 on the single-write latency lane (`CONVEYOR_CLIENT_WAL_BLOCK=1`,
`CONVEYOR_CLIENT_WAIT_SPINS=0`, 1M writes, all writes sampled). The older non-durable snippets below are abbreviated to
the timing fields used for interpretation; current benchmark output also prints `ns/write`, `checksum`, and setup
fields.

```text
CONVEYOR_MODE=file-wal-client-logged CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=1 CONVEYOR_WORKERS=1
file-wal-client-logged   6.649 M/s  elapsed=0.150s setup=0.100s clients=1 workers=0 wal-block=1 wait-spins=0 ack=non-durable-logged
client->logged           p50=0.07us p90=0.07us p99=0.10us max=0.112ms

CONVEYOR_MODE=file-wal-client-store-applied CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=1 CONVEYOR_WORKERS=1
file-wal-client-store-applied  0.737 M/s  elapsed=1.357s setup=0.190s clients=1 workers=1 wal-block=1 wait-spins=0 ack=non-durable-store-applied store-validate=0.015s
client->logged           p50=0.21us p90=0.23us p99=0.25us max=0.072ms
logged->store-applied    p50=1.07us p90=1.08us p99=1.22us max=0.225ms
client->store-applied    p50=1.28us p90=1.30us p99=1.45us max=0.226ms

CONVEYOR_MODE=file-wal-client-store-applied CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=16 CONVEYOR_WORKERS=8
file-wal-client-store-applied  2.936 M/s  elapsed=0.341s setup=0.202s clients=16 workers=8 wal-block=1 wait-spins=0 ack=non-durable-store-applied store-validate=0.018s
client->logged           p50=0.34us p90=0.50us p99=0.68us max=0.118ms
logged->store-applied    p50=4.51us p90=6.01us p99=7.72us max=0.124ms
client->store-applied    p50=4.89us p90=6.39us p99=8.12us max=0.125ms

CONVEYOR_MODE=file-wal-client-store-applied CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8
file-wal-client-store-applied  1.551 M/s  elapsed=0.645s setup=0.210s clients=128 workers=8 wal-block=1 wait-spins=0 ack=non-durable-store-applied store-validate=0.018s
client->logged           p50=0.47us p90=0.59us p99=2.14us max=0.116ms
logged->store-applied    p50=0.079ms p90=0.089ms p99=0.124ms max=2.596ms
client->store-applied    p50=0.079ms p90=0.090ms p99=0.124ms max=2.596ms

CONVEYOR_MODE=file-wal-client-store-applied CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=512 CONVEYOR_WORKERS=8
file-wal-client-store-applied  1.587 M/s  elapsed=0.630s setup=0.210s clients=512 workers=8 wal-block=1 wait-spins=0 ack=non-durable-store-applied store-validate=0.018s
client->logged           p50=0.40us p90=0.57us p99=2.72us max=0.117ms
logged->store-applied    p50=0.297ms p90=0.333ms p99=0.406ms max=5.104ms
client->store-applied    p50=0.298ms p90=0.334ms p99=0.406ms max=5.105ms
```

The dominant latency cost changed by measurement:

- WAL ingress is not the p90 problem. `client->logged` stays below 3us p99 in these runs.
- The earlier high p90 came from measuring closed-burst queue residence and from unbounded spinning waiters. A
  512-client run with spinning client waits had `client->store-applied` p90 ~21ms. Switching client waits to yield
  immediately (`CONVEYOR_CLIENT_WAIT_SPINS=0`) brought the same p90 to ~0.334ms and p99 to ~0.406ms.
- The low-latency lane uses one-write WAL blocks. Bulk `CONVEYOR_BATCH=1024` remains the throughput lane. Production
  should keep both policies: age/latency-capped small blocks for OLTP singleton writes, large blocks for bulk ingest.
- Durable client latency uses a group-commit worker rather than the final full-prefix sync benchmark. The per-client
  durable path waits for a contiguous published WAL prefix, applies the selected `durable-sync-mode` fence, and relies on
  scan recovery over block headers/trailers/CRC markers. The double-buffered control tail remains the stricter checkpoint
  path. Segment creation no longer pre-dirties the mapping, so the intended dirty set is limited to the header plus
  written WAL blocks.
- Scan recovery is best-valid-prefix recovery. It can recover every valid committed WAL block it finds, including a
  block whose client ack was lost in a crash; clients must treat connection loss before commit ack as in-doubt. Without a
  separately persisted expected frontier, scan recovery also cannot prove that storage lost an already acknowledged
  prefix. The control tail is still the stricter persisted-frontier mechanism; wiring replay/checkpoint semantics into
  the engine is a follow-on slice.

Durable client ack samples with `CONVEYOR_CLIENT_WAL_BLOCK=1`, `CONVEYOR_CLIENT_WAIT_SPINS=0`, and unique
`CONVEYOR_FILE` paths so concurrent benchmark processes do not contend on the same storage. These snippets are
summarized tables: the current raw output also prints `client-latency requested=... sample-stride=...`, and each
percentile row includes `samples=...`.

```text
CONVEYOR_MODE=file-wal-client-durable CONVEYOR_EVENTS=20000 CONVEYOR_CLIENTS=16 CONVEYOR_WORKERS=8 CONVEYOR_DURABLE_GROUP_US=125
file-wal-client-durable        0.121 M/s  8268.64 ns/write  elapsed=0.165s checksum=0xb6d1e1771f46c54 setup=0.005s clients=16 workers=8 wal-block=1 wait-spins=0 ack=data-fenced-wal+volatile-store-applied store-validate=0.000s durable-group=125us durable-sync-mode=range-and-file-data recover=0.090s
client->logged                         p50=6.45us p90=0.019ms p99=0.025ms max=0.066ms
logged->data-fenced-wal+volatile-store-applied p50=0.127ms p90=0.129ms p99=0.130ms max=0.189ms
store-applied->wal-data-fenced         p50=0.121ms p90=0.126ms p99=0.128ms max=0.183ms
client->data-fenced-wal+volatile-store-applied p50=0.129ms p90=0.139ms p99=0.142ms max=0.189ms

CONVEYOR_MODE=file-wal-client-durable CONVEYOR_EVENTS=50000 CONVEYOR_CLIENTS=16 CONVEYOR_WORKERS=8 CONVEYOR_DURABLE_GROUP_US=125
file-wal-client-durable        0.124 M/s  8066.24 ns/write  elapsed=0.403s checksum=0x4348c6a7815cbe9 setup=0.013s clients=16 workers=8 wal-block=1 wait-spins=0 ack=data-fenced-wal+volatile-store-applied store-validate=0.001s durable-group=125us durable-sync-mode=range-and-file-data recover=0.224s
client->logged                         p50=0.80us p90=0.018ms p99=0.025ms max=1.800ms
logged->data-fenced-wal+volatile-store-applied p50=0.126ms p90=0.128ms p99=0.129ms max=1.787ms
store-applied->wal-data-fenced         p50=0.122ms p90=0.125ms p99=0.127ms max=1.760ms
client->data-fenced-wal+volatile-store-applied p50=0.128ms p90=0.129ms p99=0.130ms max=1.804ms

CONVEYOR_MODE=file-wal-client-durable CONVEYOR_EVENTS=50000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8 CONVEYOR_DURABLE_GROUP_US=125
file-wal-client-durable        0.587 M/s  1703.55 ns/write  elapsed=0.085s checksum=0x4348c6a7815cbe9 setup=0.014s clients=128 workers=8 wal-block=1 wait-spins=0 ack=data-fenced-wal+volatile-store-applied store-validate=0.001s durable-group=125us durable-sync-mode=range-and-file-data recover=0.219s
client->logged                         p50=0.027ms p90=0.118ms p99=0.189ms max=1.877ms
logged->data-fenced-wal+volatile-store-applied p50=0.130ms p90=0.244ms p99=0.266ms max=1.895ms
store-applied->wal-data-fenced         p50=0.077ms p90=0.123ms p99=0.129ms max=0.404ms
client->data-fenced-wal+volatile-store-applied p50=0.253ms p90=0.261ms p99=0.290ms max=1.923ms
```

The durable result now meets the p90 < 1ms target through 128 closed-loop clients on this host. The main latency
breakthrough was removing the full-mapping zero prewrite during segment creation. That prewrite dirtied every segment
page before the run, so the first durable fence could pay for unrelated setup dirties. The latest path relies on the
file's zero-filled new extent semantics and only dirties pages as WAL blocks are published.

For comparison, the non-durable store-applied lane with 128 clients and 8 workers now reports:

```text
CONVEYOR_MODE=file-wal-client-store-applied CONVEYOR_EVENTS=100000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8
file-wal-client-store-applied        1.130 M/s   885.03 ns/write  elapsed=0.089s checksum=0x7144223f2ea19899 setup=0.026s clients=128 workers=8 wal-block=1 wait-spins=0 ack=non-durable-store-applied store-validate=0.002s
client->logged                 p50=8.27us p90=0.080ms p99=0.156ms max=0.346ms
logged->store-applied          p50=0.084ms p90=0.139ms p99=0.195ms max=0.354ms
client->store-applied          p50=0.093ms p90=0.171ms p99=0.227ms max=0.371ms
```

## Coalesced OLTP Lane

The direct client modes above deliberately used one WAL block per request. That is the lowest-overhead latency probe,
but it is not the right production shape for a majority-OLTP workload: it pays block metadata, store handoff, and durable
frontier work for every single write. The coalesced client modes keep the same client contract — one logical request,
one per-request ack — while hiding WAL block batching behind a single appender.

Current default manager-backed durable point, captured 2026-07-05 after the one-worker and `write-and-file-data`
default changes:

```text
CONVEYOR_MODE=file-wal-manager-coalesced-durable CONVEYOR_EVENTS=500000 CONVEYOR_CLIENTS=128
file-wal-manager-coalesced-durable        0.143 M/s  7002.48 ns/write  elapsed=3.501s clients=128 workers=1 wal-backend=manager wal-block=64 durable-sync-mode=write-and-file-data durable-syncs=4017 blocks/sync=2.0 sync-avg=0.821ms recover-final-durable=0.048s
client->logged        p50=0.022ms p90=0.029ms p99=0.047ms max=9.661ms
client->data-fenced-wal+volatile-store-applied p50=0.878ms p90=0.916ms p99=2.493ms max=0.015s

Same defaults, but with the older mmap-range fence selected explicitly:

CONVEYOR_MODE=file-wal-manager-coalesced-durable CONVEYOR_EVENTS=500000 CONVEYOR_CLIENTS=128 CONVEYOR_DURABLE_SYNC_MODE=range-and-file-data
file-wal-manager-coalesced-durable        0.128 M/s  7829.14 ns/write  elapsed=3.915s clients=128 workers=1 wal-backend=manager wal-block=64 durable-sync-mode=range-and-file-data durable-syncs=4012 blocks/sync=2.0 sync-avg=0.922ms recover-final-durable=0.049s
client->logged        p50=0.022ms p90=0.030ms p99=0.050ms max=0.013s
client->data-fenced-wal+volatile-store-applied p50=0.958ms p90=1.006ms p99=2.914ms max=0.021s
```

The comparison captures below are historical 2026-07-05 probes. Several explicitly set `CONVEYOR_WORKERS=8` and/or
`CONVEYOR_DURABLE_SYNC_MODE=range-and-file-data`; keep those labels when comparing against current defaults.

```text
# Non-durable store-applied, filesystem-backed target/ path.
CONVEYOR_MODE=file-wal-client-coalesced-store-applied CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=512 CONVEYOR_WORKERS=8
file-wal-client-coalesced-store-applied       10.446 M/s    95.73 ns/write  elapsed=0.096s checksum=0x36588915f096952d setup=0.319s clients=512 workers=8 wal-backend=segment wal-block=64 max-blocks=128036 block-budget=compact min-budget=62500 partial-slack=65536 blocks=18996 avg-block=52.6 append-ring=65536 append-group=25us min-block=16 requested-min-block=16 wait-spins=0 ack=coalesced-non-durable-store-applied store-validate=0.018s
client->logged                 p50=0.026ms p90=0.045ms p99=0.062ms max=1.532ms
logged->store-applied          p50=0.28us p90=0.028ms p99=0.047ms max=2.618ms
client->store-applied          p50=0.035ms p90=0.060ms p99=0.091ms max=2.652ms

# Durable WAL + volatile store, tmpfs-like /tmp path, default auto durable policy.
CONVEYOR_FILE=/tmp/write-conveyor-journal-$pid.dat CONVEYOR_MODE=file-wal-client-coalesced-durable CONVEYOR_EVENTS=5000000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8
file-wal-client-coalesced-durable        6.254 M/s   159.89 ns/write  elapsed=0.799s checksum=0xa323c2e8fb0384b1 setup=1.410s clients=128 workers=8 wal-backend=segment wal-block=64 max-blocks=378036 block-budget=compact min-budget=312500 partial-slack=65536 blocks=107326 avg-block=46.6 append-ring=65536 append-group=25us min-block=16 requested-min-block=16 wait-spins=0 ack=coalesced-data-fenced-wal+volatile-store-applied store-validate=0.088s durable-group=25us durable-min-blocks=auto durable-sync-mode=range-and-file-data durable-syncs=107315 blocks/sync=1.0 sync-avg=1.44us sync-max=0.042ms wait-avg=0.20us wait-max=0.026ms flushes=107314/1/0 recover=0.559s
client->logged                 p50=0.011ms p90=0.018ms p99=0.039ms max=3.118ms
logged->data-fenced-wal+volatile-store-applied p50=3.12us p90=6.47us p99=0.013ms max=0.256ms
store-applied->wal-data-fenced p50=0.07us p90=1.44us p99=6.06us max=0.115ms
client->data-fenced-wal+volatile-store-applied p50=0.015ms p90=0.022ms p99=0.040ms max=3.118ms

# Durable WAL + volatile store, filesystem-backed target/ path, default auto durable policy.
CONVEYOR_MODE=file-wal-client-coalesced-durable CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8
file-wal-client-coalesced-durable        0.130 M/s  7678.21 ns/write  elapsed=7.678s checksum=0x36588915f096952d setup=0.322s clients=128 workers=8 wal-backend=segment wal-block=64 max-blocks=128036 block-budget=compact min-budget=62500 partial-slack=65536 blocks=15740 avg-block=63.5 append-ring=65536 append-group=25us min-block=16 requested-min-block=16 wait-spins=0 ack=coalesced-data-fenced-wal+volatile-store-applied store-validate=0.016s durable-group=25us durable-min-blocks=auto durable-sync-mode=range-and-file-data durable-syncs=7937 blocks/sync=2.0 sync-avg=0.914ms sync-max=0.014s wait-avg=0.028ms wait-max=0.072ms flushes=2/7934/1 recover=0.092s
client->logged                 p50=0.023ms p90=0.029ms p99=0.046ms max=0.013s
logged->data-fenced-wal+volatile-store-applied p50=0.938ms p90=0.978ms p99=3.225ms max=0.017s
store-applied->wal-data-fenced p50=0.934ms p90=0.974ms p99=3.222ms max=0.017s
client->data-fenced-wal+volatile-store-applied p50=0.960ms p90=1.000ms p99=3.251ms max=0.019s

# Manager-backed coalesced WAL, same store-applied workload, filesystem-backed target/ path.
CONVEYOR_MODE=file-wal-manager-coalesced-store-applied CONVEYOR_EVENTS=5000000 CONVEYOR_CLIENTS=512 CONVEYOR_WORKERS=8
file-wal-manager-coalesced-store-applied        8.771 M/s   114.02 ns/write  elapsed=0.570s checksum=0xa323c2e8fb0384b1 setup=0.886s clients=512 workers=8 wal-backend=manager wal-block=64 manager-records/segment=262144 max-blocks=378036 block-budget=compact min-budget=312500 partial-slack=65536 blocks=93337 avg-block=53.6 append-ring=65536 append-group=25us min-block=16 requested-min-block=16 wait-spins=0 ack=coalesced-non-durable-store-applied store-validate=0.092s
client->logged                 p50=0.028ms p90=0.052ms p99=0.092ms max=0.012s
logged->store-applied          p50=9.99us p90=0.030ms p99=0.046ms max=1.640ms
client->store-applied          p50=0.039ms p90=0.071ms p99=0.102ms max=0.012s

# Manager-backed coalesced durable WAL, tmpfs-like /tmp path.
CONVEYOR_FILE=/tmp/write-conveyor-journal-$pid.dat CONVEYOR_MODE=file-wal-manager-coalesced-durable CONVEYOR_EVENTS=5000000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8
file-wal-manager-coalesced-durable        5.189 M/s   192.70 ns/write  elapsed=0.964s checksum=0xa323c2e8fb0384b1 setup=0.889s clients=128 workers=8 wal-backend=manager wal-block=64 manager-records/segment=262144 max-blocks=378036 block-budget=compact min-budget=312500 partial-slack=65536 blocks=88946 avg-block=56.2 append-ring=65536 append-group=25us min-block=16 requested-min-block=16 wait-spins=0 ack=coalesced-data-fenced-wal+volatile-store-applied store-validate=0.084s durable-group=25us durable-min-blocks=auto durable-sync-mode=range-and-file-data durable-syncs=88934 blocks/sync=1.0 sync-avg=2.38us sync-max=0.103ms wait-avg=0.04us wait-max=0.072ms flushes=88932/2/0 recover=0.435s
client->logged                 p50=0.012ms p90=0.017ms p99=0.025ms max=0.010s
logged->data-fenced-wal+volatile-store-applied p50=4.25us p90=7.43us p99=0.014ms max=0.406ms
store-applied->wal-data-fenced p50=0.27us p90=2.63us p99=9.05us max=0.125ms
client->data-fenced-wal+volatile-store-applied p50=0.017ms p90=0.021ms p99=0.029ms max=0.010s

# Manager-backed coalesced durable WAL, filesystem-backed target/ path.
CONVEYOR_MODE=file-wal-manager-coalesced-durable CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8
file-wal-manager-coalesced-durable        0.127 M/s  7904.43 ns/write  elapsed=7.904s checksum=0x36588915f096952d setup=0.324s clients=128 workers=8 wal-backend=manager wal-block=64 manager-records/segment=262144 max-blocks=128036 block-budget=compact min-budget=62500 partial-slack=65536 blocks=16055 avg-block=62.3 append-ring=65536 append-group=25us min-block=16 requested-min-block=16 wait-spins=0 ack=coalesced-data-fenced-wal+volatile-store-applied store-validate=0.016s durable-group=25us durable-min-blocks=auto durable-sync-mode=range-and-file-data durable-syncs=8058 blocks/sync=2.0 sync-avg=0.928ms sync-max=0.014s wait-avg=0.028ms wait-max=0.052ms flushes=4/8053/1 recover=0.094s
client->logged                 p50=0.022ms p90=0.029ms p99=0.056ms max=0.012s
logged->data-fenced-wal+volatile-store-applied p50=0.931ms p90=0.976ms p99=3.787ms max=0.016s
store-applied->wal-data-fenced p50=0.927ms p90=0.972ms p99=3.783ms max=0.016s
client->data-fenced-wal+volatile-store-applied p50=0.953ms p90=0.999ms p99=3.845ms max=0.019s

# Same workload, Linux fdatasync-only probe with a 100us group window.
CONVEYOR_DURABLE_SYNC_MODE=file-data-only CONVEYOR_DURABLE_GROUP_US=100 CONVEYOR_MODE=file-wal-manager-coalesced-durable CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8
file-wal-manager-coalesced-durable        0.133 M/s  7504.90 ns/write  elapsed=7.505s checksum=0x36588915f096952d setup=0.321s clients=128 workers=8 wal-backend=manager wal-block=64 manager-records/segment=262144 max-blocks=128036 block-budget=compact min-budget=62500 partial-slack=65536 blocks=15847 avg-block=63.1 append-ring=65536 append-group=25us min-block=16 requested-min-block=16 wait-spins=0 ack=coalesced-data-fenced-wal+volatile-store-applied store-validate=0.016s durable-group=100us durable-min-blocks=auto durable-sync-mode=file-data-only durable-syncs=7914 blocks/sync=2.0 sync-avg=0.822ms sync-max=0.011s wait-avg=0.102ms wait-max=0.193ms flushes=4/7909/1 recover=0.094s
client->logged                 p50=0.023ms p90=0.030ms p99=0.049ms max=4.236ms
logged->data-fenced-wal+volatile-store-applied p50=0.925ms p90=0.962ms p99=2.531ms max=0.014s
store-applied->wal-data-fenced p50=0.921ms p90=0.959ms p99=2.527ms max=0.014s
client->data-fenced-wal+volatile-store-applied p50=0.948ms p90=0.984ms p99=2.565ms max=0.014s

# Direct singleton durable, same filesystem-backed target/ path, historical fixed-window comparison.
CONVEYOR_MODE=file-wal-client-durable CONVEYOR_EVENTS=1000000 CONVEYOR_CLIENTS=128 CONVEYOR_WORKERS=8 CONVEYOR_DURABLE_GROUP_US=125
file-wal-client-durable        0.051 M/s  19731.60 ns/write  elapsed=19.732s checksum=0x36588915f096952d setup=0.174s clients=128 workers=8 wal-block=1 wait-spins=0 ack=data-fenced-wal+volatile-store-applied store-validate=0.018s durable-group=125us durable-sync-mode=range-and-file-data recover=3.811s
client->logged                 p50=0.391ms p90=1.055ms p99=1.367ms max=2.024ms
logged->data-fenced-wal+volatile-store-applied p50=1.851ms p90=3.414ms p99=5.066ms max=0.015s
store-applied->wal-data-fenced p50=1.290ms p90=3.058ms p99=4.917ms max=0.014s
client->data-fenced-wal+volatile-store-applied p50=2.405ms p90=3.585ms p99=5.154ms max=0.015s
```

Interpretation:

- Coalescing is the right OLTP lane shape. It is not a bulk query path: clients still issue and wait for individual
  writes, but the WAL owner turns them into sequential blocks. On the non-durable store-applied path this reached
  10.45M writes/s with p90 0.060ms.
- The adaptive durable policy is not trading throughput for latency on the fast path. In the historical `/tmp`
  comparison, even strict mmap-range flushing plus `sync_data` sustained 5.19M manager writes/s with p90 0.021ms. The
  sync metrics show memory-scale fences (`sync-avg=2.38us`), so auto mode keeps syncing immediately.
- On the filesystem-backed `target/` path, auto mode raises the pressure threshold and keeps the durable deadline short.
  The current default manager durable path (`write-and-file-data`, one apply worker) sustained 0.143M writes/s with
  p90 0.916ms in the 128-client run. The remaining p90/p99 tail is the data fence itself (`sync-avg=0.821ms`, p99
  client latency 2.493ms), not WAL append, store apply, or queueing. The same-default mmap range flush comparison
  reached 0.128M writes/s with p90 1.006ms; keep it labeled as `range-and-file-data`.
- Moving the coalesced path from one segment to `WalSegmentManager` did not materially regress the lane. Store-applied
  manager mode reached 8.77M writes/s with p90 0.071ms in the historical workers=8 run; current manager durable on
  `target/` reaches 0.143M writes/s with p90 0.916ms under the default descriptor-write fence. The extra
  rollover/control-plane shape is below the storage-sync noise floor for durable commits.
- Pushing concurrency past the storage knee raises throughput but loses the p90 target: 512 clients on `target/`
  reached 0.357M writes/s, but p90 was 2.614ms. Smaller durable group windows did not fix that, which confirms the
  storage fence is saturated rather than the benchmark intentionally waiting too long.
- Same-device durable striping is currently worse than the single-lane latency path. With `sync-write-data`,
  128 clients, and 500k events, single-lane manager durable reached ~0.138M writes/s with p90 ~0.918ms. Two durable
  lanes reached ~0.092M writes/s with p90 ~1.534ms, and four lanes reached ~0.060M writes/s with p90 ~2.553ms before
  the adaptive-policy/metric cleanup. Treat `CONVEYOR_DURABLE_LANES` as a hypothesis probe for multiple devices or a
  future global-cut design, not as the next production latency lever on this host.
- With `CONVEYOR_STAGE_TIMINGS=1`, coalesced modes print both block timeline and lower-level worker timing. This is
  benchmark instrumentation, not the default throughput path; it uses per-block/per-sync sample collection and can
  perturb high-throughput store-only runs. Prefer `block-timeline` for pipeline diagnosis: worker `store wait-block` can
  look large because apply workers preclaim future block ids and wait for the appender, which is useful scheduler
  information but not causal client latency. `store->durable slack` is conditional because store apply and durability run
  independently; on very fast durable media, durability can beat store apply for some blocks. Captured with stage timing
  enabled on 2026-07-05:

```text
# Non-durable manager coalesced store path, 1M events, 512 clients.
file-wal-manager-coalesced-store-applied        6.678 M/s
client->store-applied p50=0.040ms p90=0.067ms p99=0.106ms
block publish->store  p50=8.78us p90=0.021ms p99=0.036ms
append wal-publish    p50=1.46us p90=1.75us p99=0.026ms
store apply-block     p50=1.58us p90=2.26us p99=2.67us

# Durable manager coalesced path, sync-write-data probe, 500k events, 128 clients.
file-wal-manager-coalesced-durable              0.133 M/s
client->data-fenced p50=0.881ms p90=0.919ms p99=3.351ms
block publish->store  p50=4.44us p90=5.68us p99=9.95us
store->durable slack  p50=0.851ms p90=0.885ms p99=3.291ms
durable wait/group    p50=0.028ms p90=0.029ms p99=0.030ms
durable sync          p50=0.830ms p90=0.857ms p99=2.463ms

# Historical range-and-file-data probe, same run shape.
client->data-fenced p50=0.959ms p90=0.997ms p99=3.196ms
durable sync        p50=0.909ms p90=0.944ms p99=2.871ms
```

These numbers move the bottleneck out of debate: appender and store are microsecond-scale; durable ack latency is the
storage fence. The `durable wait/group` metric measures policy wait after the durable worker observes pending work; it is
not total durable-worker scheduling delay. With 128 closed-loop clients, ~0.83ms sync p50 and ~122 records/sync puts the
lane near the observed ~0.13M writes/s. Larger WAL blocks and fixed four-block durable thresholds did not improve the
128-client lane: block 128 raised p90 to ~2.424ms, and `CONVEYOR_DURABLE_MIN_BLOCKS=4` still synced about two
blocks/fence because the appender was not far enough ahead before the 25us deadline.

Raw durability isolation, captured 2026-07-05 on `target/` (`/dev/nvme1n1p1`, XFS `rw,noatime`, Samsung SSD 9100 PRO
1TB), confirms the same diagnosis without client/apply noise:

```text
# Data-frontier raw WAL, 200k events, raw block=64, validate by scan recovery.
sync-write-data fence_blocks=1  0.085 M rec/s  publish p50=8.36us p90=20.55us p99=26.16us  fence p50=798.52us p90=855.57us p99=2.381ms
sync-write-data fence_blocks=2  0.152 M rec/s  publish p50=6.93us p90=19.60us p99=25.93us  fence p50=799.03us p90=890.10us p99=2.505ms
sync-write-data fence_blocks=4  0.263 M rec/s  publish p50=6.85us p90=17.44us p99=25.03us  fence p50=815.47us p90=2.315ms p99=2.523ms
sync-write-data fence_blocks=8  0.415 M rec/s  publish p50=6.38us p90=14.86us p99=31.87us  fence p50=838.74us p90=2.453ms p99=2.513ms

range-and-file-data fence_blocks=1 0.076 M rec/s  fence p50=875.14us p90=931.68us p99=2.786ms
range-and-file-data fence_blocks=2 0.138 M rec/s  fence p50=880.58us p90=967.71us p99=2.809ms

file-data-only tracks sync-write-data on this host:
file-data-only fence_blocks=1  0.086 M rec/s  fence p50=790.78us p90=840.77us p99=2.369ms
file-data-only fence_blocks=2  0.156 M rec/s  fence p50=798.11us p90=868.39us p99=2.470ms

# Full durable-prefix/control-tail raw WAL, 100k events, block=64.
durable-prefix fence_blocks=1  0.038 M rec/s  fence p50=1.547ms p90=1.656ms p99=3.752ms
durable-prefix fence_blocks=2  0.070 M rec/s  fence p50=1.555ms p90=3.293ms p99=3.839ms
```

Interpretation:

- The WAL publish stage is not the latency problem. Raw publish is ~6-10us p50 and ~15-21us p90 per 64-record block.
- The client durable benchmark is now explained by the raw fence curve. Around one to two WAL blocks per fence keeps p90
  under 1ms but caps throughput near 0.08-0.16M records/s on this filesystem/device. Larger fence groups raise throughput
  but push p90 to ~2.3-2.8ms because the storage fence itself develops a long tail.
- The full durable-prefix/control-tail path is roughly two data fences and cannot be the per-commit latency path. It
  should remain a checkpoint/recovery publication path, with client visibility gated by a durable data LSN/cut.
- The next big optimization is not another queue policy. The WAL I/O backend has to beat the raw fence curve first, then
  publish a durable global cut only after the data fence completes.

The first aligned-frame/direct-I/O bakeoff did **not** improve durability latency on this host:

```text
# mmap/page-cache raw WAL, exact 62-record blocks = 4096B frames, 200k events.
sync-write-data fence_blocks=1 0.083 M rec/s  publish p50=6.07us p90=14.73us  fence p50=796.79us p90=850.75us p99=2.430ms
sync-write-data fence_blocks=2 0.152 M rec/s  publish p50=5.78us p90=14.03us  fence p50=803.04us p90=873.48us p99=2.441ms
range-and-file-data fence_blocks=1 0.074 M rec/s  fence p50=879.45us p90=932.20us p99=2.725ms
file-data-only fence_blocks=1 0.082 M rec/s  fence p50=791.35us p90=846.48us p99=2.459ms

# O_DIRECT raw WAL, exact 62-record blocks = 4096B frames, 200k events.
direct-dsync fence_blocks=1     0.025 M rec/s  publish p50=2.25us p90=2.62us  fence p50=2.450ms p90=2.490ms p99=2.546ms
direct-dsync fence_blocks=2     0.050 M rec/s  publish p50=2.20us p90=2.33us  fence p50=2.457ms p90=2.495ms p99=2.582ms
direct-fdatasync fence_blocks=1 0.025 M rec/s  fence p50=2.454ms p90=2.491ms p99=2.700ms
direct-rwf-dsync fence_blocks=1 0.025 M rec/s  fence p50=2.465ms p90=2.504ms p99=2.604ms
```

Frame alignment trims raw publish CPU, but the durable fence is still the limiter. On this XFS/NVMe stack, O_DIRECT is
about 3x slower than the mmap/page-cache data-frontier path at the same 4096B frame size. That makes direct synchronous
writes a negative result for the current hardware/filesystem, not the next integration target. The remaining big bets are
larger commit cut publication over a separate durable LSN, deadline-aware group scheduling, and production hardware with
a lower-latency durable-write path (for example PLP NVMe/pmem/DAX-class semantics), validated by the raw benchmarks
before touching the engine.

The first io_uring bakeoff also did **not** improve the low-latency durable fence:

```text
# raw_uring_wal_bench, 500k events, block=64, target/ path.
uring-rwf-dsync fence_blocks=1        0.025 M rec/s  fence p50=2.563ms p90=2.609ms p99=2.891ms
uring-fixed-rwf-dsync fence_blocks=1  0.025 M rec/s  fence p50=2.557ms p90=2.615ms p99=2.967ms
uring-write-fdatasync fence_blocks=1  0.025 M rec/s  fence p50=2.548ms p90=2.695ms p99=2.890ms

uring-rwf-dsync fence_blocks=256        4.817 M rec/s  fence p50=2.870ms p90=2.893ms p99=2.904ms
uring-fixed-rwf-dsync fence_blocks=256  4.708 M rec/s  fence p50=2.912ms p90=2.991ms p99=3.036ms
uring-write-fdatasync fence_blocks=256  4.293 M rec/s  fence p50=2.932ms p90=2.972ms p99=6.364ms
```

Registered buffers reduce none of the durable fence cost in this synchronous probe. This io_uring shape is competitive
only in the same Chronicle-sized large-group regime as the existing raw path; it is materially worse for one/two-block SQL
latency fences. Do not wire this synchronous io_uring shape into the production WAL path on this host unless a later
kernel/filesystem/device result beats the page-cache `sync-write-data` baseline. A deeper queued io_uring design remains
a separate hypothesis.

## FUA-Pipelined Durable Lane (2026-07-05) — SUPERSEDES the direct-I/O and io_uring negative results

The O_DIRECT and synchronous-io_uring "negative results" above were measurement artifacts, not device physics.
Two poisons stacked in every earlier durable measurement:

1. **Unwritten extents.** Every benchmark file was `posix_fallocate`d and never pre-written, so the whole timed
   region ran over XFS *unwritten* extents. Each durable fence then pays an extent-conversion journal force.
   Measured on this host (4KiB commit writes, `/home/richard/projects` XFS, Samsung 9100 PRO):
   fallocate-only extents = ~2.45ms/fence for O_DIRECT+O_DSYNC and ~2.49ms for buffered+fdatasync; the SAME
   operations over pre-written extents = 1.67ms and 0.85ms. `MappedWalSegment::create` still has this defect
   (fallocate, no prewrite); the engine WAL's earlier "fdatasync 2.46ms -> 0.84ms after zero-fill" (W4a) was
   the same effect measured without recognizing it.
2. **Serial fencing.** All five `WalDataSyncMode`s funnel through one `sync_in_progress` slot and end in either
   `fdatasync` (a full NVMe cache FLUSH: unpipelineable, the device drains its whole volatile cache) or a
   buffered `RWF_DSYNC` (degenerates to writeback+FLUSH). A SERIAL FUA write is slower than a serial FLUSH here
   (1.67ms vs 0.85ms), which is why serial-minded probing rejected O_DIRECT. But FUA writes are independent
   NVMe commands: they pipeline, and this drive coalesces concurrent FUA writes internally.

Raw FUA fence curve, pre-written extents, one file, 4KiB frames (62-record WAL blocks), measured three ways
(standalone probe, and `raw_direct_wal_bench` with the new knobs below — scan-recovery validated):

```text
serial FLUSH (fdatasync, prewritten)      ~1360 fences/s  p50=0.85ms   <- ceiling of the current serial design
serial FUA                                  ~647 fences/s  p50=1.67ms
FUA qd=8                                  ~5000 fences/s  p50=1.71ms
FUA qd=16                                ~28600 fences/s  p50=0.68ms  p99=0.75ms   <- fast-mode flip
FUA qd=24                                ~23000 fences/s  p50=1.04ms
FUA qd=32                                ~38000 fences/s  p50=0.84ms
FUA qd=64                                ~68000 fences/s  p50=0.94ms  p99<1ms
```

The device rewards queue depth: between qd=8 and qd=16 it flips into a mode where concurrent FUA writes share
NAND programs, and per-op latency DROPS 2.4x while throughput rises 5.7x. No serial experiment can see this.
Interleaved-consecutive frame offsets behave the same as spread offsets, with or without page cache, so WAL
frame layout needs no interleaving tricks. 28.6K fences/s x 62 records = **1.78M durable records/s at p99
0.75ms per fence** on the same drive the serial design capped at 0.08-0.16M rec/s.

`raw_direct_wal_bench` new knobs, and the reproduction matrix (200K-4M events, block=62, fence_blocks=1):

```text
CONVEYOR_RAW_DIRECT_PREWRITE=1   # default ON: stream real zeros + fsync at setup (written extents)
CONVEYOR_RAW_DIRECT_QD=16        # fence lanes claiming groups from a shared cursor

prewrite=0 qd=1    0.025 M rec/s  fence p50=2.462ms            <- the old "negative result", reproduced
prewrite=1 qd=1    0.040 M rec/s  fence p50=1.682ms
prewrite=1 qd=16   1.775 M rec/s  fence p50=0.681ms p99=0.753ms
prewrite=1 qd=32   2.350 M rec/s  fence p50=0.834ms p99=0.861ms

# open-loop device ceiling, 126-record 8KiB frames:
prewrite=1 qd=64 block=126   11.198 M rec/s  88.9K fences/s  fence p50=0.783ms p99=0.940ms
```

The drive's FUA coalescing improves with BOTH queue depth and frame size: 88.9K 8KiB fences/s at qd=64 is more
fences/s than 4KiB at the same depth. 11.2M rec/s with sub-millisecond p99 fences is the bulk-ingest bound.

### Client lane: `fua_wal_client_bench`

`fua_wal_client_bench` keeps the exact coalesced client contract (closed-loop logical clients, one write per
request, ack = durable WAL frame + volatile store applied, scan recovery + store validation after the timed
region) over the FUA fence pool. Structure: clients -> bounded ring -> one appender packing 62/126-record
frames into ANONYMOUS aligned staging (no file mmap, so no dirty-page writeback/invalidation against the
direct writes) -> N fence lanes each FUA-writing one frame through a shared `O_DIRECT|O_DSYNC` descriptor ->
durable cut = contiguous prefix of completed fences -> ordered store apply.

Two scheduling rules matter more than any queue policy, both learned from failed intermediate runs:

- **Adaptive frame sizing.** Maximally-packed frames starve the fence pool at moderate client counts (128
  clients / 62-record frames = ~4 frames in flight = slow mode). The appender targets
  `frame = clamp(pending / fence_qd, floor, block_size)` so backlog spreads across the pool; frames grow
  toward the cap only at saturation, preserving the throughput ceiling.
- **Fence-pool pacing.** Without it the closed loop convoys: each fence completion releases a few clients,
  their records ship immediately as a tiny frame, pool depth collapses to 1-2, and the drive falls back to
  serial fence latency (measured: 830 fences/s x 4-record frames = 3.3K rec/s death spiral). The appender
  ships a frame only when `published - durable < fence_qd`, accumulating while all lanes are busy. Batch
  boundaries must be driven by DOWNSTREAM stage occupancy, not upstream arrival — this is the Chronicle
  discipline the earlier lane was missing.

Client ladder on the same filesystem-backed `target/` path (2M-record timed regions, recovery-validated):

```text
                                        throughput   client->durable-ack        fence p50
old lane best, 128 clients (doc above)   0.130 M/s   p50=0.96ms  p99=3.25ms     0.9ms serial flush
old lane best, 512 clients (doc above)   0.357 M/s   p90=2.61ms (past its knee)
fua qd=16  128 clients                   0.113 M/s   p50=1.16ms  p99=1.50ms     0.69ms
fua qd=16  512 clients                   0.381 M/s   p50=1.35ms  p99=1.92ms     0.70ms
fua qd=16  1024 clients                  0.664 M/s   p50=1.52ms  p99=2.32ms     0.70ms
fua qd=16  2048 clients                  1.242 M/s   p50=1.48ms  p99=2.32ms     0.69ms
fua qd=16  2048 clients, block=126       1.137 M/s   p50=1.68ms  p99=2.57ms     0.77ms
fua qd=16  4096 clients, block=126       1.942 M/s   p50=1.89ms  p99=3.89ms     0.78ms
fua qd=24  4096 clients, block=126       2.035 M/s   p50=1.87ms  p99=2.69ms     0.80ms
```

At 128 clients FUA only ties the serial-flush design — a FLUSH is a group commit over everything published,
so at low concurrency it is competitive. The FUA lane's win is that it keeps scaling with client population
(closed-loop TPS = clients / ack-latency, and ack latency stays ~1.5-1.9ms while the flush design's fence
saturates): **2.0M durable client writes/s at p99 2.7ms**, ~15.6x the old lane's 128-client number and ~5.7x
its best-ever number at any concurrency. Client ack p50 has a hard floor of one fence (~0.7ms) plus one
pool-pacing cycle on this consumer drive; sub-millisecond p50 durable acks require PLP/enterprise media
(fence ~10-20us), where this same architecture collapses to microsecond acks — the trajectory bet holds.

Production integration order (this replaces the "deadline-aware durable scheduler" plan below):

1. Segment lifecycle: create+fallocate+PRE-WRITE+fsync segments off the hot path, and RECYCLE them
   (PostgreSQL-style) instead of create/unlink, which also avoids unlink/writeback log churn near fences.
2. Replace the single-slot durable worker with a FUA fence pool (qd 16-24 on this host; re-probe per device)
   writing frames from staging through `O_DIRECT|O_DSYNC`; durable cut = contiguous completed-fence prefix.
3. Publishers write staging, not a file mapping; 512B-aligned block strides (62/126-record frames).
4. Appender: adaptive frame sizing + fence-pool-occupancy pacing as above.
5. The control tail stays a checkpoint path; per-commit visibility gates on the durable data cut.

### Library implementation: `FuaWalSegment` (2026-07-05)

Steps 1-3 above are now library code in `crates/write_conveyor/src/fua_wal.rs`:

- `FuaWalSegment::create` — new segment file with WRITTEN extents (streamed zeros + one fsync at setup) and a
  durable epoch-stamped header written through the `O_DIRECT|O_DSYNC` descriptor.
- `FuaWalSegment::recycle` — reuse an existing pre-written file under a NEW segment id: geometry check, header
  rewrite (which also retires the previous life's control records — they live in the header page), no prewrite.
  Measured end to end: recycled runs match created runs' throughput with the prewrite cost gone.
- `FuaWalSegment::appender()` — the single publishing handle; `publish_intents` stages one frame (header +
  records + CRC trailer, epoch stamped) and publishes with one release store. No file I/O on the publish path.
- `FuaWalSegment::spawn_fence_pool(lanes)` — fence lanes claim published frames, FUA-write them, and advance
  the contiguous durable cut (`durable_blocks` / `durable_record_seq`). `free_fence_slots(lanes)` is the
  appender's pacing gate.
- Recycle safety is a WAL FORMAT ADDITION: file header flag `WAL_SEGMENT_FLAG_EPOCH_STAMPED` + block header
  `reserved0` = segment id low 32 bits. Scan recovery requires the epoch to match, so valid-looking frames
  from a recycled file's previous life are rejected (unit-tested). Legacy segments (flag unset) still validate
  with `reserved0 == 0`, so all pre-existing tests and files are unaffected.
- Unit tests cover: publish/fence/recover roundtrip with partial frames, recycle epoch rejection, unfenced-tail
  non-recovery, out-of-order fences gating the durable cut, config/geometry validation, and pacing occupancy.
- Independent adversarial audit (opus, 2026-07-05) found and we fixed: (MAJOR) `segment_id as u32` epoch
  truncation let ids with zero low-32 bits reuse the legacy epoch-0 sentinel and silently accept a recycled
  file's previous-life frames — now rejected in config validation (regression-tested), with the id contract
  (non-zero low-32, monotonic within 2^32 per physical file) documented; (MINOR) a fence-lane IO error had no
  pollable signal, so pacing/ack spin loops would hang instead of surfacing it — `FuaWalSegment::fence_failed()`
  added, set before the lane exits, polled by the bench's appender/ack/store loops; the error itself surfaces
  through `FuaFencePool::join()`. The audit verified durable-cut/fence-claim/join semantics, memory ordering,
  O_DIRECT alignment, StorageFull consistency, symlink hygiene, and legacy/manager recovery compatibility with
  no critical findings.

`fua_wal_client_bench` now runs entirely on the library types, with `CONVEYOR_RECYCLE=1` to exercise recycled
segments. Library-path ladder (recovery + store validated; matches the prototype within noise):

```text
fua qd=16  2048 clients                  1.208 M/s   p50=1.51ms  p99=2.48ms
fua qd=16  2048 clients, recycled file   1.176 M/s   p50=1.54ms  p99=2.45ms   (no prewrite at setup)
fua qd=24  4096 clients, block=126       1.991 M/s   p50=1.88ms  p99=2.46ms
fua qd=24  8192 clients, block=126       2.658 M/s   p50=2.29ms  p99=3.55ms
```

Envelope notes: 190-record (12KiB) frames and qd=32 do NOT beat (qd=24, block=126) at 4096 clients — frame
growth is population-bound before it is device-bound. Latency-lean point: qd=16, block=62. Throughput point:
qd=24, block=126, population >= 4096. The benchmark's setup time is dominated by its worst-case-sized staging
allocation (capacity assumes every frame at the liveness floor); production segments (e.g. 256Ki records =
~17MB staging) do not have this artifact. Remaining engine-side integration: segment rolling under the manager
(recycle pool + background prep thread), replacing yield-waits with event-driven acks, and gating SQL commit
visibility on `durable_record_seq`.

### Optimization rounds (2026-07-05, post-commit `6c05f825`): 2.0 -> 6.3M durable writes/s

Instrumentation-driven rounds over the committed lane. Per-block stage timings
(`FuaWalSegment::enable_stage_timings`, `CONVEYOR_STAGE_TIMINGS=1`) attribute publish -> fence-start ->
fence-done -> durable-cut; handles are captured once per thread (a per-op Mutex read measurably contended the
fence lanes), so enable it BEFORE `spawn_fence_pool`/`appender`.

What the measurements found, in order:

1. **Pacing metric defect (library fix).** `free_fence_slots` gated on `published - durable`, but the durable
   cut is CONTIGUOUS: out-of-order-completed fences still counted as in-flight, idling lanes behind any
   straggler and sagging effective device queue depth below the lane count. Fixed with an order-independent
   `fences_completed` counter. After the fix the pipeline runs at the device floor: publish->fence-start p50
   901ns, FUA write p50 683us, fence-done->cut p50 330ns.
2. **Wait-strategy A/B (thread-per-client).** yield_now ack-waiting costs a scheduler round-trip under
   thousands of threads. Parked waiters (per-seq Mutex<Option<Thread>> registry + waker threads) beat
   bucketed-condvar cohorts decisively — condvars pay a thundering herd on every partial-bucket notify plus
   wake serialization through the bucket mutex. Short `park_timeout` polling is poison (100us: 0.63M/s,
   p99 19ms). But even tuned parking plateaus: direct wake-lag sampling showed unpark->resume at p50 8.3us
   while ~670us of ack latency remained — the waker pool itself saturates at ~5.8us per unpark syscall,
   exactly 100% utilized at 1.4M acks/s. THE PER-REQUEST FUTEX WAKE+PARK PAIR IS UNPAYABLE AT MILLIONS OF
   ACKS/S; no waker-pool width fixes it (32 wakers measured no better than 8).
3. **Event-loop client multiplexing (the Chronicle shape).** `CONVEYOR_CLIENT_DRIVER_THREADS` (default 64)
   drivers sweep logical-client state machines: no parks, no wakes, acks observed by polling two atomics per
   logical client per sweep. This removes the wake wall entirely, moves the standing queue into the ring
   (frames pack to capacity: avg-block 125.6/126), and unlocks population scaling far past OS-thread limits.
4. **Fence-pool width at population.** With drivers, the lane is cleanly device-bound; raising fence lanes
   follows the raw FUA curve (the drive coalesces more concurrent FUA writes).
5. **Post-round audit (opus): no correctness defects** across fences_completed ordering, over-publish,
   stage-timing lifecycle, driver ring-gate/termination/barrier accounting, and waiter aliasing. One MINOR
   benchmark-integrity finding fixed: waker threads no longer spawn in driver mode (they spun at 100% CPU
   scanning empty registries and polluted throughput).

Client-contract frontier (per-request durable ack, scan-recovery + store validated, filesystem-backed
`target/`):

```text
clients   qd  block   throughput   client->durable-ack
 2048     16   62      1.40 M/s    p50=1.35ms p99=2.25ms
 4096     24  126      2.49 M/s    p50=1.66ms p99=3.94ms
 8192     48  126      4.28 M/s    p50=1.91ms p99=3.51ms
16384     48  126      5.87 M/s    p50=2.74ms p99=3.83ms
32768     64  126      6.30 M/s    p50=5.09ms p99=7.31ms   <- 48x the pre-FUA lane's best-ever
```

254-record (16KiB) frames do not beat 126 at these populations (frames run underfull — population-bound
before device-bound). The engine-side lesson is structural: SQL commit acks must be delivered by polling
event loops (gate batches of commits on `durable_record_seq` sweeps), never by per-commit thread wakeups.

**Sub-millisecond p90 regime (population-share frame targeting).** A pending-based frame target self-defeats
below saturation: it sizes frames from the ring LEFTOVER, so a standing queue (~0.4-0.8ms) never drains and
ack latency floors at ~1.15ms even at 256 clients. Targeting `frame >= clients/fence_qd` absorbs the whole
population into the fence pipeline (ring ~= 0) and the ack collapses to the bare fence — the conveyor adds
<10us around the device write:

```text
 512  qd=32  62    0.59 M/s   p50=835us  p90=857us  p99=1.37ms
1024  qd=32  62    1.16 M/s   p50=835us  p90=871us  p99=1.37ms
2048  qd=48  62    2.09 M/s   p50=886us  p90=922us  p99=3.40ms
2560  qd=48  62    2.65 M/s   p50=890us  p90=927us  p99=3.07ms   <- p90<1ms envelope edge
3072  qd=48  62    3.07 M/s   p50=900us  p90=1.10ms             <- past the edge (frames hit cap, ring re-forms)
```

Rule of thumb for the engine: pick lanes ~= clients/frame with frame <= 62 and the durable ack p90 tracks the
raw FUA fence p90 at that queue depth (~0.86-0.93ms on this drive). The p99 tail (~3ms) is device fence
stragglers, within the <5ms SLO. Sub-0.8ms p90 on this consumer drive is not reachable (fence floor); PLP
media collapses the same architecture to tens of microseconds.

### E1 foundation: `FuaFrameLog` — variable-payload frame log for the engine WAL (2026-07-05)

The engine's `gpu_db_wal` records are variable-length (SQL text, binary row-ops, DDL) and its group flush is
single-slot serial `write_all`+`fdatasync` (`io_in_flight` excludes every other writer — at most ONE durable
IO in flight, ~0.86ms/job in the archived diagnostics). `FuaFrameLog` generalizes the FUA lane to opaque
variable payloads so ONE totally ordered log carries every record shape: 4KiB-aligned frames (64B padding-free
CRC'd header + payload + zero pad), bounded slot ring, fence pool + contiguous durable cut reporting
`durable_seq` (the engine's commit-visibility gate), epoch-stamped recycle, chain-walking scan recovery.
Two implementation landmines worth recording: (a) header CRCs must cover PADDING-FREE layouts — implicit
`repr(C)` padding is nondeterministic across stack copies; (b) frame alignment must be the 4KiB FILESYSTEM
block, not the 512B device block — XFS serializes sub-fs-block O_DIRECT writes on the exclusive inode lock
(measured: 3.5K fences/s at qd=16 with 512B alignment vs 28.5K at 4KiB).

Smoke bench (`fua_frame_log_bench`, engine-shaped payload mix, scan-recovery validated): 28.5K durable
frames/s at qd=16, 55.0K at qd=48 — identical fence economics to the fixed-record lane, so an engine group
frame of ~50 records has a ~2.75M commits/s durability budget.

Charter framing for the engine integration: this log is host CONTROL-PLANE work (WAL/durability IO is an
enumerated host duty). The stages behind the durable cut — validation, store apply, index maintenance,
visibility — are DATA PLANE and belong on the GPU (E2: fused device apply kernel replacing host tuple-store /
value-index publication by DELETION, per the retirement program). SQL commit acks gate on `durable_seq` via
batch-polling event loops, never per-commit wakeups.

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
6. **Done locally:** add client-perspective latency modes and a low-latency one-write block lane. This isolates WAL
   ingress, ordered store apply, and waiter contention from the saturated bulk benchmark.
7. **Done locally:** add a durable client frontier with scan recovery over committed WAL blocks. This removes the
   benchmark-wide final sync from client durable latency and avoids a per-commit control-tail fsync.
8. **Done locally:** add a coalesced OLTP client lane: clients submit singleton writes, one WAL owner batches them into
   sequential blocks, and per-request acks wait on the assigned block. This keeps the latency contract while recovering
   high throughput in the same logical lane.
9. **Done locally:** make the coalesced durable policy pressure-aware and instrumented. Fast sync media fence
   immediately; storage-scale syncs use a small block threshold plus a short deadline, and the benchmark reports sync
   count, blocks/sync, sync time, wait time, and flush reasons.
10. **Done locally:** run the coalesced OLTP path over `WalSegmentManager`: rolling segment files, cloneable
    published-block handles for async apply/durable workers, and scan recovery across manager segment directories.
11. **Done locally:** add a raw WAL durability isolator. This proves the remaining latency is the filesystem/device
    fence, not appender/store/conveyor scheduling, and gives a stable benchmark for direct-I/O and io_uring WAL backends.
12. **Done locally:** add exact frame-aligned WAL block counts and a direct-I/O raw WAL bakeoff. On the current XFS/NVMe
    host, 4096B-aligned mmap frames are still ~0.8-0.9ms p90 while O_DIRECT variants are ~2.5ms p90, so synchronous
    direct writes are not the next latency path here.
13. **Done locally:** add a synchronous io_uring raw WAL bakeoff, including registered-buffer `WriteFixed` variants. On
    the current XFS/NVMe host, this io_uring `RWF_DSYNC` shape is ~2.6ms p90 for one-block fences and only competitive at
    large groups, so it is not the low-latency durable WAL backend. Deeper queued io_uring variants remain unproven.
    **(Superseded 2026-07-05: items 12-13 measured fallocate-unwritten-extent conversion plus serial fencing, not the
    device. See "FUA-Pipelined Durable Lane".)**
14. **Done (2026-07-05):** raw FUA fence-pipelining proof: `CONVEYOR_RAW_DIRECT_PREWRITE` (written extents at setup) and
    `CONVEYOR_RAW_DIRECT_QD` (parallel fence lanes) in `raw_direct_wal_bench`. 0.025 -> 1.78M durable rec/s at qd=16 with
    p99 fence 0.75ms, 2.35M at qd=32, scan-recovery validated.
15. **Done (2026-07-05):** `fua_wal_client_bench` — the coalesced client contract over a FUA fence pool with anonymous
    staging, adaptive frame sizing, and fence-pool-occupancy pacing. 2.0M durable client writes/s at p99 2.7ms
    (4096 clients, qd=24, 126-record frames), ~15.6x the prior client-lane number, recovery + store validated.

Next: wire the FUA fence pool + pre-written/recycled segment lifecycle into `MappedWalSegment`/`WalSegmentManager`
(per the production integration order above), re-probe `fence_qd` on target hardware at engine bring-up, then wire the
conveyor through the engine commit path with validation/index maintenance as block/GPU stages behind the durable cut.

Do not wire this into the SQL commit path until the staged conveyor keeps the same order of magnitude.
