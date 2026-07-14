# Project Agent Guidance

This repository is pursuing a GPU-native database engine. Future agents should
optimize for that thesis unless the user explicitly changes direction.

## GPU-Native North Star

- Treat GPU-resident execution as the product direction, not as an optional
  accelerator around a CPU-first database.
- Optimize for GPU-native OLTP: entity fetches, tenant/security-filtered page
  reads, bounded joins, and computed detail routes, not only analytical scans or
  primary-key microbenchmarks.
- Prefer designs where hot data, lookup structures, encoded columns, and read
  snapshots live in GPU memory.
- The CPU is the host/control plane ONLY: wire protocol, SQL parse/plan,
  transaction coordination, WAL/durability I/O, and GPU orchestration. CPU
  relational execution exists solely as parity-reference plus temporary
  bootstrap scaffold, tracked as debt with a GPU milestone — never product
  direction and never the hot-path design.
- The catalog is GPU-native and joins are GPU operators: `pg_catalog` and
  `information_schema` are GPU-resident system relations, and catalog joins
  (`psql \d`, ORM introspection) run on the GPU join path. Do not build a CPU
  catalog or CPU nested-loop/hash join as the target answer.
- For hot reads, prefer prepared route ids, typed parameters, resident snapshot
  handles, and device-ready projection plans over repeated SQL-text parsing.
- Favor immutable/versioned GPU-resident snapshots for read concurrency.
  Serialize mutation and generation publication until a stronger MVCC model is
  intentionally designed.
- Optimize batching for throughput, but do not make batching the only latency
  answer. GPU-native low latency likely requires concurrent read execution over
  resident snapshots.

## Architecture Bias

When choosing between implementation approaches:

1. Keep the GPU hot path explicit and measurable.
2. Preserve CPU/GPU semantic parity, but treat CPU relational execution as
   parity-reference/bootstrap debt with a milestone — never product direction.
3. Avoid adding CPU caches or CPU indexes as the primary answer for benchmark
   wins unless the change is clearly documented as a non-GPU-native escape
   hatch.
4. Prefer principled concurrency boundaries: immutable read snapshots,
   serialized writers, CUDA stream ownership, epoch/generation retirement.
5. Be cautious with heuristic scheduler complexity. If a policy becomes hard to
   explain, consider a simpler split between latency-oriented prepared reads and
   throughput-oriented batch routes.

## Documentation Expectations

`docs/PLAN.md` is the **only** document that owns open, deferred, blocked, or sequenced work. `STATUS.md`
records facts; `HANDOVER.md` is a short pointer to active PLAN IDs; architecture, ADR, and `docs/design/`
documents do not own tasks. Everything under `docs/archive/` is historical and non-actionable even when it
contains words such as `NEXT`, `TODO`, or `OPEN`.

Permanent analysis instrumentation lives behind the build-only `probe-timing` Cargo feature. Reuse and extend
those probes instead of writing and reverting one-off hot-path timers.

Before major runtime, storage, or scheduler changes, read:

- `docs/CHARTER.md` — mandate, invariants, the OLTP bet, execution discipline + gotchas
- `docs/ARCHITECTURE.md` — the full system design (execution model, residency/STRATA, deterministic OLTP, MVCC,
  durability, multi-GPU)
- `docs/DECISIONS.md` — the decision ledger (ADRs)
- `docs/PLAN.md` (ordered work) · `docs/STATUS.md` (current state) · `docs/HANDOVER.md` (resume baton)
- `docs/CODE_SIZE.md` — source-size thresholds, decomposition method, reference updates, and exceptions

When a change intentionally favors CPU-first behavior, document why it is a
fallback, bootstrap step, or product-scope exception.

## Source File Size and Module Boundaries

Follow `docs/CODE_SIZE.md`. Production source over 2,000 lines and test/example/tool source over 3,000 lines
requires an audited disposition; any file over 5,000 lines must remain owned by a PLAN task until it is split or
accepted in the exception registry. New modules should normally remain below 1,500 lines.

Split by invariant and ownership, not by line range. Preserve stable facades, move the closest tests, update
module/import/re-export/build/test/doc references in the same slice, and verify old paths and symbols are gone.
Do not create `part1`/`part2` shards, catch-all modules, dependency cycles, or broad visibility solely to make a
split compile. Keep behavior changes separate from structural extraction and run the gates prescribed by the
standard and the affected subsystem.

## Read-path performance regression benchmark (standard)

There is ONE standard read-kernel benchmark; run it before/after any change that touches a
read kernel, the residency layout, or the result path, and compare to the baseline below.

```
timeout 300 cargo run --release --example read_kernel_roofline -p gpu_db_execution
#   (no GPU? it prints a skip line. Never use --gpu-reset. ROWS=/SORT_N=/ITERS= override.)
```

It drives each read-kernel FAMILY directly on a resident column and reports effective GB/s
(bandwidth-bound scans / gather) or M-elem/s (sort / join / grouped). The **roofline is `sum_i32`**
(a pure 1-pass read+reduce = the HBM streaming peak). NOTE: `equal_any` is also measured but is NOT
the roofline -- its 8-needle per-element compare is compute-bound, ~2x slower than a pure read.

WHAT IS KERNEL-CLEAN: only section (1) (resident-input scans) is wall ~= kernel. gather/sort/join
take a HOST slice and upload it to the device EVERY call (per-call input H2D the engine does NOT pay
-- its inputs are device-resident), so their wall OVERSTATES the kernel; each line is H2D-labeled.
The single-launch gather kernel is additionally isolated via the CUDA event (~6us vs ~885us wall).
The GROUP BY line shows the aggregate KERNEL (event-timed, ~5ms) AND its full result path (~467ms;
the ~99% non-kernel tail = the 8M-row index H2D + ~2*row_count slot-table setup + host Vec build).

**Compare by RATIO, not absolute GB/s.** Absolute bandwidth varies by GPU/driver, so the portable
signal is `kernel_GB_s / sum_i32_roofline_in_the_same_run`. A material drop in a kernel's ratio
(or in the algorithmic M-elem/s) vs the baseline is a regression to investigate.

Baseline (8M i32 rows, captured 2026-06-30; roofline `sum_i32` was ~1486 GB/s on that box):

| family | kernel | ratio-to-roofline (or Melem/s) | note |
|---|---|---|---|
| scalar reduce (ROOFLINE) | `sum_i32` | 1.00 (~1486 GB/s) | pure 1-pass read+reduce = HBM peak |
| 8-needle scan (NOT roof) | `equal_any` | ~0.43 | compute-bound, ~2x a pure read |
| scan-project ordered (1% sel) | `project_compare` | ~0.12 (2-pass) | near roof/pass — SATURATED |
| ordered compaction (50% sel) | `compare_indices_ordered` | ~0.024 (2-pass) | output-bound — expected |
| arith VM | `arith_filter` | ~0.023 (2-pass) | ok |
| constant mask output | `const_mask_false` | ~0.62 IN-L2 / ~0.98 OUT-OF-L2 | 1-pass i32 device fill; no host vector/H2D or result D2H (~37us / ~188us) |
| **scalar COUNT (reduce)** | `count_i32_compare` | **~0.85** | block-reduced (was KNOWN HEADROOM) |
| | `count_i32_between` | **~0.44** | calls count x2 |
| filter -> indices | `expr_i64_compare_scalar` (1% sel) | ~0.11 | ok (8B col) |
| | `expr_i128_compare_scalar` (1% sel) | ~0.19 | ok (16B col) |
| gather (scattered) | `gather_i32` kernel-only / wall | ~350 GB/s / ~2.4 GB/s | wall = +~4MB idx H2D |
| algorithmic | sort / join / grouped-KERNEL | ~321 / ~259 / ~1679 Melem/s | sort/join wall = +key H2D; grouped = event-timed kernel (full path ~18) |

The scalar COUNT reductions (`count_i32_compare`, `count_i32_between`, `equal_count`) used to be the
KNOWN headroom target (a per-thread `red.global.add` on one counter = ~N serialized atomics); they
are now block-reduced and near roof in this baseline (`count_i32_compare` ~0.85; `count_i32_between`
~0.44 since it calls count twice). (The i64/i128 FILTERS earlier looked "slow" only as a
100%-selectivity output artifact; at ~1% sel they are fine.) Treat a ratio FALLING below the
baseline as the regression signal.

The constant-mask line was added on 2026-07-13 after removing an inherited O(rows) host `Vec<i32>` plus H2D
upload. Its throughput denominator is the four output bytes written per row, not input bytes read; it retains
the synchronized VM mask on device without compaction or result D2H. Compare both cache regimes and treat a
return toward the former ~3.3 GB/s / 10,106us IN-L2 / 80,271us OUT-OF-L2 behavior as host-staging regression.

### Standard benchmark report card (BOTH layers x BOTH cache regimes)

The roofline above is Layer 1 only. The canonical, recurring artifact is the **report card**, which
ALWAYS reports BOTH layers x BOTH cache regimes, p50 latency + throughput on every line:

```
scripts/benchmark_report_card.sh        # ~17-20 min; self-manages timeouts + GPU cool-downs
```

- **Layer 1 -- RAW READ KERNELS** (`read_kernel_roofline`, crates/execution): emits IN-L2 (32MB/col,
  8M rows) AND OUT-OF-L2 (256MB/col, 64M rows) in ONE invocation.
- **Layer 2 -- lpb/wave POINT-READ PATH** (`r2_wave_engine_ab`, crates/engine): batched equal-any
  point reads = the production default route, run IN-L2 (Section B, 1M rows) and OUT-OF-L2 (Section C).

**This card's L2 = 128 MB** (cudaDevAttrL2CacheSize, RTX PRO 6000 Blackwell Max-Q). OUT-OF-L2 needs the
gathered i32 column (4B/row) to exceed 128MB => > 32M rows. Section C uses **48M rows = 192MB/col
(1.5x L2)** by default.

**BUILD-TIME NOTE (why Section C has its own timeout):** `r2_wave_engine_ab` builds its table via a SQL
INSERT loop at ~11 us/row (CPU-bound SQL parse + txn/MVCC apply; in-memory WAL, no fsync). A 2026-07-12
run measured 719s insert + 33s residency; even 1000s cut off during the final batch-65536 scan. Section C
therefore gets `SECTION_C_TIMEOUT=1200` (Sections A/B keep 280). The INSERT-chunk size was measured
non-helpful (250/1000/10000 all ~11 us/row -- the cost is the engine's per-row apply, not per-statement
overhead), so the lever is the timeout, not the chunk. Tunables (env): `OUT_OF_L2_ROWS`,
`OUT_OF_L2_BATCHES` (default 300; p50 stable there), `SECTION_{A,B,C}_TIMEOUT`, `GPU_GAP`. Never
`--gpu-reset`.
