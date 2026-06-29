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

It drives each read-kernel FAMILY directly on a resident column (kernel-only timing) and reports
effective GB/s (bandwidth-bound scans / gather) or M-elem/s (sort / join / grouped). The first
line, `equal_any`, is a pure coalesced read = the measured **streaming roofline** for the box.

**Compare by RATIO, not absolute GB/s.** Absolute bandwidth varies by GPU/driver, so the portable
signal is `kernel_GB_s / equal_any_roofline_in_the_same_run`. A material drop in a kernel's ratio
(or in the algorithmic M-elem/s) vs the baseline is a regression to investigate.

Baseline (8M i32 rows, captured 2026-06-29; roofline `equal_any` was ~640 GB/s on that box):

| family | kernel | ratio-to-roofline (or Melem/s) | note |
|---|---|---|---|
| scan (ROOFLINE) | `equal_any` | 1.00 | — |
| scan-project ordered (1% sel) | `project_compare` | ~0.29 (2-pass) | near roof/pass — SATURATED |
| ordered compaction (50% sel) | `compare_indices_ordered` | ~0.06 (2-pass) | output-bound — expected |
| arith VM | `arith_filter` | ~0.05 (2-pass) | ok |
| scalar reduce | `sum_i32` | ~0.36 | mild headroom |
| **scalar filter/count** | `count_i32_compare` | **~0.013** | KNOWN HEADROOM (not a regression) |
| | `count_i32_between` | **~0.007** | KNOWN HEADROOM |
| | `expr_i64_compare_scalar` | **~0.003** | KNOWN HEADROOM |
| | `expr_i128_compare_scalar` | **~0.005** | KNOWN HEADROOM |
| gather (scattered) | `gather_i32` / `gather_i64` | ~0.004 / ~0.005 | access-bound (inherent) |
| algorithmic | sort / join / grouped | ~346 / ~255 / ~267 Melem/s | separate programs |

The `scalar filter/count` family is a KNOWN optimization target, NOT a regression — its low
ratios are expected until optimized. `count_i32_compare` is root-caused (`launch_cuda_resident_
i32_compare_count` launches grid=row_count/256 = one thread/row + a per-thread `red.global.add`
on a single counter = ~N atomics serialized); the `between` / i64 / i128 filters are measured-
slow too but not yet individually root-caused (likely the same pattern). Treat a ratio FALLING
below these as the regression signal; a ratio rising (e.g. after that fix) is the win.
