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

Before major runtime, storage, or scheduler changes, read:

- `docs/CHARTER.md` — mandate, invariants, the OLTP bet, execution discipline + gotchas
- `docs/ARCHITECTURE.md` — the full system design (execution model, residency/STRATA, OLTP wave engine, MVCC,
  durability, multi-GPU)
- `docs/DECISIONS.md` — the decision ledger (ADRs)
- `docs/PLAN.md` (ordered work) · `docs/STATUS.md` (current state) · `docs/HANDOVER.md` (resume baton)

When a change intentionally favors CPU-first behavior, document why it is a
fallback, bootstrap step, or product-scope exception.

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
