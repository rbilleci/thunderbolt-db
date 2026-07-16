# ADR-014 five-minute standalone recovery profile

**Date:** 2026-07-15\
**Scope:** standalone restart/recovery on the reviewed host; replicated node-loss recovery remains HA-001\
**Disposition:** **DESIGN BOUND ESTABLISHED; current recovery implementation does not satisfy it.**

This report supplies ADR-014's R3-001 byte/time argument. It defines a configuration mathematically bounded below
five minutes, identifies the measurements behind its replay coefficient, and makes every unimplemented throughput
assumption a hard qualification/refusal condition. It does not claim that canonical checkpoint artifacts, direct
GPU replay, or automatic cadence already exist.

## Current measurements

`wal_recovery_time_probe` wrote binary covered INSERT records through the durable serial backend and measured
checkpoint-aware reopen. The probe no longer forces the table's elision flag without device backing; that stale
setup now violates the live device-authoritative invariant.

| Rows before rotation | Live logical WAL | Pre-rotation reopen | Post-rotation rows | Post-rotation reopen |
|---:|---:|---:|---:|---:|
| 10,000 | 0.56 MB | 0.310 s | 10,064 | 0.328 s |
| 50,000 | 2.84 MB | 1.201 s | 50,064 | 1.277 s |
| 100,000 | 5.69 MB | 2.335 s | 100,064 | 2.493 s |
| 250,000 | 14.39 MB | 6.259 s | 250,064 | 6.663 s |

The pre-rotation least-squares fit is `T = -0.0326 s + 24.965 us * records` (`R² = 0.99864`), or about 40,056
records/s. Post-rotation remains linear at 26.579 us/row, about 37,624 rows/s: the current checkpoint rotation
bounds the live suffix but still replays the full historical checkpoint. It therefore does not establish a
long-running five-minute bound. At the post-rotation rate, 300 seconds covers only about 11.29 million rows—roughly
28 seconds of history at a conservative 400,000 write-outcomes/s stress rate—before fixed work and any safety
factor. That rate is deliberately harsher than, and is not a relabeling of, the charter's mixed-system peak TPS.

The real FUA intent-lane path independently recovered 300,001 INSERT operations in 7.57 seconds (39,630 ops/s) and
285,878 mixed operations in 7.44 seconds (38,425 ops/s), both with exact row-count parity. Agreement between the two
paths supports a measured current replay coefficient near 38,000–40,000 operations/s. The capacity profile below
halves the lower lane result to a qualification floor of **19,200 records/s**.

## Bound and concrete profile

For a recovery policy allowing one complete fresh-context retry:

```text
T_attempt = T_fixed + B_restore / R_restore + N_suffix / R_replay
T_RTO     = 2 * T_attempt + T_reserve
```

The initial standalone five-minute profile is:

| Parameter | Hard profile value | Meaning |
|---|---:|---|
| `T_fixed` | 15 s/attempt | pointer/manifest discovery, status reconciliation, final validation/publication, and process/context setup outside the two linear terms |
| `B_restore` | at most 32 GiB | all referenced payload, catalog/status, visibility, and mandatory serving-index bytes needed before service |
| `R_restore` | at least 512 MiB/s | end-to-end read, digest verification, staging/H2D, GPU decode, and mandatory-index-ready effective rate |
| `N_suffix` | at most 1,000,000 complete outcomes | marker-complete records after the activated checkpoint |
| `R_replay` | at least 19,200 outcomes/s | half the slower measured FUA lane rate |
| attempts | 2 | initial attempt plus one full retry on a fresh process/context/GPU |
| `T_reserve` | 30 s | scheduler/filesystem variance and activation margin |

This gives:

```text
T_attempt <= 15 + 32 GiB / 512 MiB/s + 1,000,000 / 19,200
          <= 131.09 s
T_RTO     <= 2 * 131.09 + 30
          <= 292.18 s
```

The remaining 7.82 seconds are rounding margin, not capacity to allocate. More than one failed full attempt, a
device/storage floor below the profile, or an artifact set above 32 GiB is outside this standalone RTO class and
must refuse the five-minute configuration rather than publish an unsupported guarantee.

`N_suffix` is additionally byte-bounded:

```text
suffix_limit = min(1,000,000 outcomes, floor(512 MiB / measured_worst_WAL_bytes_per_outcome))
```

The measured narrow FUA path writes about 433 physical bytes/op, so one million outcomes is about 433 MB and the
record cap wins. Under conservative write-only stress, the checkpoint activation interval must therefore be at
most 10 seconds at 100,000 outcomes/s or 2.5 seconds at 400,000 outcomes/s. Wider records shorten the interval
through the byte cap. Immutable content-addressed
payloads and manifests must reuse unchanged artifacts; this cadence is not permission to rewrite 32 GiB every 2.5
seconds.

The actual narrow allocation measurement also makes checkpoint bandwidth part of physical selection. At 816.3
retained bytes per appended INSERT plus about 433 physical WAL bytes/op, 400,000 write outcomes/s would create approximately
326.5 MB/s of retained-allocation pressure and 173.2 MB/s of WAL writes before compaction/reuse and wider rows. A
candidate whose incremental checkpoint/index work cannot remain below the qualified storage and GPU budgets cannot
join this RTO profile even if raw replay is fast enough.

## Enforcement contract

The design closes the capacity argument only with these fail-loud rules:

1. Checkpoint activation records exact referenced bytes, mandatory-index bytes, suffix start, format lineage, and
   measured restore/replay qualification floors.
2. Admission forecasts both record and byte suffix caps. It initiates incremental checkpoint activation early
   enough to retain headroom; if activation cannot finish before either cap, new writes are rejected before WAL.
3. Service startup is refused when the selected active generation exceeds its declared profile or the host/GPU
   qualification run falls below either rate. An operator may select a smaller artifact/suffix cap or a longer RTO,
   but the engine does not silently weaken five minutes.
4. Every index needed by an admitted route is either verified in `B_restore` or its measured rebuild is included in
   `R_restore`. A lazily rebuilding optional index leaves its route explicitly unready.
5. Recovery measures each phase and abandons a poisoned CUDA context. One retry consumes the second attempt budget;
   a second full-attempt failure leaves the node unavailable for repair.
6. STRATA demotion, PITR/status pins, predecessor retention, and GC cannot make bytes disappear from the equation.
   If their reachable set exceeds 32 GiB, this profile is unavailable until a measured higher-throughput profile or
   a smaller recovery working set is activated.

These are accepted ADR-014 durability/admission rules, not prose-owned work. DUR-001/002 own their canonical
implementation and qualification, as recorded in `PLAN.md`.

## Reproduction

```text
GPU_DB_WAL_DURABILITY=serial GPU_DB_PROBE_BINARY=1 GPU_DB_PROBE_ROWS=<N> \
  cargo run --release -p gpu_db_engine --example wal_recovery_time_probe

GPU_DB_BENCH_ARM=driver GPU_DB_BENCH_OFFERED_TPS=100000 \
GPU_DB_BENCH_RECOVER=1 GPU_DB_BENCH_MIX_UPDATE=<0-or-20> \
GPU_DB_BENCH_MIX_DELETE=<0-or-10> \
  cargo run --release -p gpu_db_engine --example intent_fast_path_bench --features probe-timing
```

The current canonical artifact/mandatory-index restore path does not yet exist, so the 512 MiB/s term is a hard
future qualification floor, not a measured fact. That separation is intentional: the design now has an explicit
solvable bound, while production authority remains blocked until DUR-001/002 measure and enforce every term.
