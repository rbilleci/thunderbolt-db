# Cross-kernel transfer: applying the lpb wins to the other routes

**Scope:** what the lpb int4 point-read optimization arc (~3M → ~78M rows/s) can transfer to the *other*
row-producing routes (the `equal_any` SCAN, the general int4 projection / distinct / ordered routes) and the
other kernels. Companion docs: [`read-path-lpb-levers.md`](read-path-lpb-levers.md),
[`scan-kernel-levers.md`](scan-kernel-levers.md), [`small-batch-latency-levers.md`](small-batch-latency-levers.md).

**Status when written:** 2026-06-29, post-wave-retirement (lpb-dense is the production default for the int4
unique-key route; the persistent WaveReadEngine read path is retired). **Code-only analysis** (no benchmarks —
a concurrent agent is active; this is read-only mapping). File:line anchors verified against the current tree.

---

## The decomposition (why most of lpb does NOT auto-transfer)
The lpb arc was three different kinds of change:
1. **Two universal STRUCT wins** — already shared by every route (no action).
2. **A route-specific HOST result-path overhaul** — applied ONLY to the int4 point-read path; the SCAN /
   projection / distinct / ordered routes are **still on the pre-lpb path**. ← the big transfer.
3. **One KERNEL change (dense single-pass compaction)** — narrow; does not generalize to unbounded-output
   kernels. ← small / different technique for the others.

---

## 1. Already shared — DO NOT redo (structural)
`RelationalSelectResult` (`relational_model.rs:310-322`) already has `columns: Arc<..>`, `access_path: Arc<..>`,
and `rows: RowBlock` (flat row-major, `relational_model.rs:176-180`). So **every** route that returns this
struct — scan, projection, distinct, ordered, grouped — already gets lpb levers #1 (Arc-share schema) and #2
(flat RowBlock) for free. Skip them.

## 2. THE BIG TRANSFER — the general row-producing routes are still on the pre-lpb host path
The SCAN and the general int4 projection / distinct / ordered routes (`engine_select_exec.rs:459-479` →
`execute_resident_grouped_via_general` →) run `execute_relational_equality_multi_column_projection_batch_inner`
(`engine_resident_probe.rs:1100-1645`) — which is **lpb's own starting point before the arc**. It still pays
every cost lpb removed:

| lpb lever (point-read path) | What the general path still does | Anchor |
|---|---|---|
| **Batched result model** (`RelationalRetainedBatchResult`, `relational_model.rs:243-268`) | Builds **N per-needle `RelationalSelectResult`** structs, each with its own `Arc::new(columns)` | `engine_resident_probe.rs:1631-1644` (per-needle Arc `:1636`) |
| **Columnar drain** (`complete_detached_columnar`, `lib.rs:2873`; `CudaI32BatchProjectionColumns`) | Materializes `rows_by_select: Vec<Vec<(u64, Vec<SqlValue>)>>` — per-row `Vec<SqlValue>` boxing | `:1367` |
| **O(n) counting-sort scatter** (`assemble_batched_rows`, `engine_retained_read.rs:982-1081`) | Per-needle `slice.sort_by_key(row_index)` (O(n log n) per needle), no scatter | `:1519-1525` |
| **i32 end-to-end** (`RelationalRetainedBatchResult.values: Vec<i32>`, `relational_model.rs:249`) | Converts `i32 → SqlValue::Int4` **inside the loop** — no raw-i32 carry | `:1382` |
| **Slim submission** (`needle_count` + shared Arcs, `relational_model.rs:502-518`) | Builds a per-select `members` Vec with per-needle clones | `:1118-1224` |

**Action:** unify these routes onto the point-read machinery (`RelationalRetainedBatchResult` +
`assemble_batched_rows` + `complete_detached_columnar` + i32 carry + slim submission). It is the same arc that
took lpb 3M → 78M, **and** a de-duplication (two assembly paths collapse into one).

**Scope caveat — where this actually pays off:** it helps routes that are **host-materialization-capped**:
- **High-output / multi-needle int4 projection / distinct / ordered** — structurally identical to lpb's
  situation, almost certainly at the old ~3M cap. **Biggest, most certain win.**
- A **selective** scan (few matches) is GPU-**bandwidth**-bound, not host-bound — its result path is small, so
  this transfer gives little there (use the scan-kernel-levers doc instead).
Confirm with a phase-split measurement on a high-output scan/projection first (mirror `lpb_phase_split_probe`)
before investing — exactly as lpb did.

## 3. KERNEL-side transfer — narrower than it looks
- **Dense single-pass compaction (lpb's kernel win, `gpu_db_resident_i32_index_probe_dense`, `lib.rs:10144`)
  does NOT generalize.** It relies on **bounded** output (unique ⇒ ≤1 row/needle ⇒ a dense `result[needle]`
  slot). The scan family has unbounded output per input — no dense slot exists.
- The four kernels still using `atom.global.add` compaction are candidates for the **generalization** of the
  same insight ("atomic-add compaction serializes"): **warp-aggregated atomics** (ballot + popc, one atomic per
  warp). Matters for **high-selectivity** scans (many matches contending). Kernels:
  - `gpu_db_resident_i32_equal_any_project` — `lib.rs:9405`
  - `gpu_db_resident_i32_equal_any_project_text` — `lib.rs:10745`, `10747`
  - `gpu_db_resident_i32_equal_row_indices` — `lib.rs:11641`
  - `gpu_db_resident_i32_index_probe` (atomic; superseded by the dense default) — `lib.rs:9832`
  (This is the same lever as `scan-kernel-levers.md` #2 — do once.)

## 4. Where the transfer does NOT apply — don't spend effort
The **scalar / aggregation routes** — `sum`, `equal_count`, `compare_count(_blocks/_parallel)`,
`between_stats`, `row_count`, and largely `grouped_*` — produce a tiny result (a scalar or a handful of group
rows). No per-row materialization storm, so the host result-path levers (columnar drain, batched model, i32
carry, scatter) give nothing. Their optimization axis is **kernel-side** (reduction efficiency, the existing
`_parallel`/`_blocks` two-phase structure, memory bandwidth) — a different program. Edge case: a `grouped_*`
with a very large `#groups` could borrow the flat/counting-sort idea for its compact step, but it's generally
far below scan output.

## The meta-lesson
lpb's defining discovery: **the cap was HOST result materialization, not the GPU** (the kernel drained tens of
millions; the host capped it at ~3M). The scan / projection / distinct / ordered routes are *still on that
pre-lpb host path*, so they very likely sit at the same ~3M ceiling — masking kernels that can drain far more.
The highest-leverage cross-kernel action is therefore **not** a kernel change; it is **unifying the
row-producing routes onto the batched-columnar-i32-scatter result path**. One refactor lifts scan + projection
+ distinct + ordered together, the way the result-path arc lifted lpb.

## Suggested sequencing
1. **Phase-split a high-output scan/projection** to confirm the host-materialization cap (cheap; mirror
   `lpb_phase_split_probe`). Don't assume — lpb's whole lesson was "measure the host path."
2. **Unify the general row-producing routes onto the point-read result machinery** (batched result + columnar
   drain + counting-sort scatter + i32 carry + slim submission). Biggest win + a de-dup.
3. **Warp-aggregate the atomic-compaction kernels** for high-selectivity scans (kernel-side; shared with
   `scan-kernel-levers.md` #2).
4. **Leave the scalar/aggregation routes** to their kernel-side program — the result-path work doesn't touch
   them.

## Discipline (charter, non-negotiable)
GPU tests under `timeout`, NEVER `--gpu-reset`; ASCII-only PTX (`ptxas -arch=sm_70` check before launch);
**independent adversarial audit** on any kernel/protocol change (the warp-agg work in #3) — never self-audit;
the SCAN is on the wave-retired byte-identity differential (scan == lpb-dense), so any change must keep that
green; the result-path unification (#2) is behavior-preserving — prove it with the existing GPU differentials +
facade/protocol byte-identity (the lpb arc's standard); commit/push/merge each verified increment with trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; do NOT run a GPU test right after a
`timeout`-killed one. Coordinate with the active agent before editing the shared result-assembly code
(`engine_resident_probe.rs` / `engine_retained_read.rs` / `relational_model.rs`).
