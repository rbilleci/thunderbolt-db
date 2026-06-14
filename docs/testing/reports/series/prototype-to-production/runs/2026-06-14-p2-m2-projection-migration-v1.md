# P2-M2 cont. — projection routes: parallel kernels, migrate the orchestration

Status: closed (first projection route migrated; the wall was orchestration, not the kernel)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 ("Parallel kernels")
Branch: `phase0-m1-engine-facade`
Follows: `.../runs/2026-06-14-gpu-retained-query-mix-v1.md`

## The discovery (which redirected the work)

The GPU-retained query-mix benchmark showed `multi_col_projection` and `mixed_int_text`
walling out at c64 (~24–27 ms p50, ~2k qps), and that report first attributed it to
"single-thread `(1,1,1)` projection kernels." **That was wrong.** Reading the launch code,
the projection kernels are **already parallel**: `gpu_db_resident_i32_equal_project` computes
a global thread id (`blockIdx*blockDim + tid`), one thread per row, and on a filter match does
`atom.global.add.u32` to claim an output slot and writes the projected columns — launched with
`blocks = ceil(rows/128)`, `block = 128`.

The c64 wall was the **unmigrated per-call orchestration**, identical to the count route
*before* P2-M1: per-launch `cuModuleLoadData`/`Unload` (re-JIT every call) + a whole-context
`cuCtxSynchronize` on the default stream — plus a per-call ~800 KB output `cuMemAlloc` and a
synchronous `cuMemsetD8`. So "continue P2-M2" is not "parallelize the kernel"; it is **migrate
the projection routes onto the P2-M1 substrate** (cached module + pooled stream).

## What changed

- **Generalized the pooled-stream helper:** `launch_on_pooled_stream(resident, scratch_out:
  Option<&mut [u8]>, launch)`. Scalar routes (count, equality-count) pass `Some(scratch)` (the
  pooled scratch holds the small result); multi-buffer routes (projection) pass `None` and
  manage their own large device output buffers, doing their own D2H after the (already-synced)
  return. Both get cached-module + pooled-stream + per-stream-sync.
- **Migrated `equal_project`** (`multi_col_projection`): per-launch `cuModuleLoadData` →
  `cached_function`; `launch_with_optional_cuda_event_timing(ctx_sync, default-stream)` →
  `launch_on_pooled_stream(None, …)`.

## Result (GPU-retained query mix, concurrent, 50k rows)

| `multi_col_projection` | before | after |
|---|---:|---:|
| c1 p50 | 202 µs | **106 µs** (per-launch JIT gone) |
| c64 p50 | 24,690 µs | **14,912 µs** |
| c64 qps | 2,142 | **3,793** (~1.77×) |

A real ~1.7× improvement at c64 — confirming the per-launch JIT + whole-context sync was a
wall (the kernel was never the problem). It still trails the scalar routes (count ~17k,
equality-count ~20k qps @c64): the **remaining wall is the per-call 800 KB output `cuMemAlloc`
+ the synchronous `cuMemsetD8`** (a context-wide barrier still issued before the launch). The
fix — pool the output buffer + async-on-stream memset — is the next slice.

## Honest scope

- One route migrated (`equal_project` / `multi_col_projection`). `mixed_int_text`
  (`equal_any_project_text`), `equal_any_project`, `compare_project`, and `row_indices` share
  the exact same shape (parallel kernel + old orchestration) and are mechanical follow-ups via
  the same helper — `mixed_int_text` was **unchanged** in this slice (still ~27 ms p50 @c64).
- The prior report's "serial `(1,1,1)` projection kernels" claim is **corrected** there and in
  the plan's Phase-2 note.
- Validation: 13 execution GPU tests + 39 engine GPU tests (incl. resident projection) green;
  fmt + clippy clean.

## Independent adversarial audit

A reviewer was charged to **refute** (A) the generalized helper's scalar `Some` path, (B) the
`None` path for `equal_project` (memset ordering, sync-before-D2H, module cleanup), (C) cached
function reuse + concurrency, (D) result-ordering regression, (E) the perf claims — reading
the diff and re-running the tests + benchmark.

**Result: no LIVE BLOCKER.** A–C upheld: the `Some` path is byte-identical to the old helper;
the `None` path's synchronous `cuMemsetD8` happens-before the kernel (host-blocking) and the
helper `cuStreamSynchronize`s before the route's D2H; removing the per-call `module_guard` is
correct (the cached module is owned by `GpuPrimaryContext::Drop`); each concurrent call owns
its own output buffers + pooled stream (no aliasing). Perf reproduced (multi_col c1 102 µs,
c64 14.8 ms / 3821 qps). Fixed one **MINOR** it found: a stale duplicated doc comment on the
renamed helper.

**Latent fragility it flagged (pre-existing, NOT introduced or worsened here):** the
projection routes return rows in **atomic-claim order** (`Vec<Vec<i32>>`, no row index, no
sort), while some engine tests assert *exact* row order on multi-match cases. This commit
changed **zero** PTX lines, so the exposure is unchanged — and it's empirically stable
(ordering probe 40/40 green; the benchmark's projection queries are point lookups = 1 match,
so order is moot). But for **large matched sets**, inter-block atomic-claim order can diverge
from row order — a real correctness landmine to fix (carry a row index + sort, or a
prefix-sum compaction) when wide multi-match projections are exercised. Tracked.

## Next

1. Migrate the remaining projection/gather routes (`equal_any_project_text` first — it's
   `mixed_int_text`) via `launch_on_pooled_stream`.
2. Pool the per-call output buffer + async-on-stream memset to remove the residual projection
   wall.
3. **Fix the pre-existing claim-order fragility** (carry a row index + sort, or prefix-sum
   compaction) before exercising wide multi-match projections — today's tests/queries are
   single-match so it's latent.
4. Re-run the GPU-retained mix; expect the projection rows to approach the scalar routes.
