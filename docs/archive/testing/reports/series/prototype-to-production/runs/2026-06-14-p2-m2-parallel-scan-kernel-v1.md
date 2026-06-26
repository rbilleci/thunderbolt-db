# P2-M2 — Parallel scan kernel: the first GPU-native compute win, measured

Status: closed (parallel filtered-count is ~60× the serial scan; correctness-verified)
Date: 2026-06-14
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 2 ("Parallel kernels")
Branch: `phase0-m1-engine-facade`

## Why this milestone

The P2-M1 step-4 finding pointed past the substrate: the resident **COUNT(\*)** is a
precomputed O(1) header read (it can't scale because there's no work), but the real GPU
*compute* routes are the **scans**. Inspection confirmed the filtered-count route
`gpu_db_resident_i32_equal_count` was a **single-thread `(1,1,1)` serial loop** — one GPU
thread walking every row (`loop: … add %idx, 1; bra loop`), exactly the "shallow GPU" the
plan §1.2 calls out. For a large table this is catastrophic. P2-M2 replaces it with a
real parallel kernel and measures the win.

## What changed

`gpu_db_resident_i32_equal_count_parallel` (now the canonical
`launch_cuda_resident_i32_equal_count`): each thread computes a global id + grid stride
from `%tid.x/%ntid.x/%ctaid.x/%nctaid.x`, **grid-strides** over the i32 column counting
matches into a register, then `red.global.add.u64` accumulates into a zeroed scratch
(`block = 256`, `grid = ceil(rows/256)` clamped to 65 535; grid-stride covers any larger
table by wrapping). It runs on the **P2-M1 substrate** — the module is JIT-loaded once and
cached, it launches on a pooled private stream, and the scratch is zeroed with
`cuMemsetD8Async` on that stream immediately before the kernel. The old serial loop is kept
`#[cfg(test)]` as `_serial`, only as the A/B baseline. The public methods
(`count_i32_equal_from_payload`, `count_i32_in_from_payload`) are unchanged — they call the
canonical name and so now use the parallel kernel.

## Result — correct, and ~60× faster as the table grows

A/B spike (`gpu_parallel_i32_equal_count_matches_serial_and_wins_on_large_tables`), RTX PRO
6000 Blackwell, `value[i] = i % 7`, `needle = 3`, best-of-3 latency:

| rows | expected matches | serial (1,1,1) ms | parallel ms | speedup |
|---:|---:|---:|---:|---:|
| 4,096 | 585 | 0.237 | 0.023 | 10.5× |
| 65,536 | 9,362 | 1.960 | 0.051 | 38.6× |
| 1,048,576 | 149,797 | 29.808 | 0.515 | 57.9× |
| 4,194,304 | 599,186 | 118.891 | 2.000 | 59.5× |
| 16,777,216 | 2,396,745 | 475.042 | 7.932 | **59.9×** |

- **Correctness is a hard gate:** at *every* size the parallel count **equals** the serial
  count **equals** the CPU-computed expected — exact match, including the 16,777,216-row
  case where `ceil(rows/256) = 65 536 > 65 535`, so the grid clamps and the grid-stride
  **wraps** (the first 256 threads do two iterations). Wrap + clamp are therefore
  exercised and correct.
- **The win scales with the work:** 10.5× at 4 K rows (launch overhead still visible) up to
  **59.9× at 16 M rows** — the serial single thread takes 475 ms to walk 16 M elements; the
  parallel kernel does it in 7.9 ms.

This is the **first real GPU-native compute win in the project** — a parallel kernel beating
the prototype's shallow single-thread kernel by ~60×, with the result proven correct by
exact count match against the serial and CPU references.

## Honest scope

- **Execution-crate microbenchmark**, not engine/pgwire end-to-end. It is measured by
  building the device column directly (`retain_device_memory_chunks`) because there is no
  bulk-load path to materialize a 16 M-row table through SQL `INSERT` yet; the engine adds a
  small fixed parse/plan overhead on top, negligible against a 475 ms→8 ms scan.
- **One column type / one operator:** `i32` equality count. The other scan routes
  (compare-count, sum, project, the `_equal_any_project` family) are still single-thread
  serial loops — the same parallelization applies and is the tracked follow-up.
- **The A/B is not a pure algorithm-vs-algorithm comparison.** The serial baseline is the
  fully unmigrated shape: it does `cuMemAlloc` + `cuModuleLoadData`/`Unload` + `cuMemFree`
  *every* call, launches on the null stream, and syncs with `cuCtxSynchronize`; the parallel
  kernel uses the cached-module + pooled-stream substrate. So the ratio folds in both the
  parallelization *and* the substrate. At 16 M rows the serial's fixed overhead (~0.2 ms,
  visible in the 4 K serial datapoint) is ~0.04 % of 475 ms, so **the 59.9× is not
  materially inflated** and isolates the parallelization win. The **small-table multipliers
  (10.5× at 4 K, 38.6× at 64 K) ARE inflated** by that per-launch overhead and are
  apples-to-oranges — they overstate the *algorithmic* win at small sizes. The headline
  deliberately uses only the 16 M figure.
- No claim against `DESIGN.md §1.1` targets — this demonstrates the parallel-kernel unlock
  and sizes it.

## Independent adversarial audit

A reviewer was charged to **refute** (A) kernel correctness for all inputs (grid-stride
bounds, non-multiple-of-block, sub-block, `row_count=0`, grid-clamp + grid-stride wrap, the
atomic reduction, integer overflow), (B) zeroing order + readback validity, (C) no OOB
device read, (D) substrate soundness, (E) no overclaim.

**Result: no LIVE BLOCKER; all five charges upheld.** The reviewer wrote a temporary
adversarial `#[ignore]` GPU test and ran it on the RTX PRO 6000, then reverted it —
verifying parallel == serial == CPU at: `n=0`; `n=1` (hit and miss); block boundaries
255/256/257/511/512/513; the **clamp threshold `n=16,776,960`** (full grid, no wrap);
**`n=16,776,961`** (forced single wrap); a **deep ragged wrap at `n=33,566,265`** (~2.4×
saturated threads, non-multiple of 256); and **sentinel probes at the last row and at the
first wrapped index `16,776,960`** (a skip would give 0, a double-visit would give 2 — both
returned exactly 1). Overflow analysis confirmed clear: `stride = gridDim*blockDim` ≤
`65,535*256` ≈ 16.7 M (u32-safe), the loop increment and byte addressing are u64. The
`red.global.add.u64` reduction has no lost updates; the memset runs every call on the same
stream before the kernel; readback is after the stream sync; the bounds check uses checked
arithmetic. Full `cargo test -p gpu_db_execution --release -- --ignored` = 14/14 pass
(existing filtered-count tests now route through the parallel kernel — no regression).

**Acted on:**
- **MINOR (methodology):** the small-table speedups are inflated by the serial baseline's
  per-launch overhead — sharpened in *Honest scope* above (only the 16 M figure isolates the
  parallelization win).
- **LATENT note:** added a code comment at the grid/`BLOCK` computation documenting the
  invariant that keeps `stride` within u32 (grid ≤ 65,535, `BLOCK` ≤ 1024), so a future
  `BLOCK` increase can't silently overflow it. Not reachable today (`BLOCK = 256`).

## Next

- Parallelize the remaining scan routes (compare-count, sum, project) with the same
  grid-stride + reduction pattern on the substrate.
- A bulk device-load path so large tables can be exercised end-to-end through the engine.
- (From P2-M1) async submission / batched completion for concurrent throughput, and
  `partition_device_memory` → `SnapshotCell` before concurrent writes.
