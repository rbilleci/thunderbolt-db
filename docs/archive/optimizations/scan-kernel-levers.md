# ARCHIVED — Scan-kernel optimization analysis

> Historical point-in-time analysis. It is not an executable plan. Any surviving obligation is tracked only
> in `docs/PLAN.md`.

**Scope:** the GPU full-scan `equal_any` predicate kernel — the fallback read path for everything that is
NOT an int4 unique-key index point lookup (non-unique equality, no index, etc.). Main kernel:
`gpu_db_resident_i32_equal_any_project` PTX in `crates/execution/src/lib.rs` (kernel body `~9195-9287`,
launch `~9437`). Sibling per-type variants (i64/text) and `equal_count` exist; the levers below apply to the
i32 kernel and likely its siblings.

**Companion docs (different regime — point lookups are latency-bound, this is bandwidth-bound):**
historical notes `read-path-lpb-levers.md` and `small-batch-latency-levers.md` (not retained in this tree).
**Levers from those docs do NOT transfer here** — see "What does not transfer" below.

**Status when written:** 2026-06-29. **Code-only analysis (no benchmarks run).** Line refs verified against
the PTX directly.

---

## Framing — a scan is MEMORY-BANDWIDTH-bound

The scan reads the whole filter column (and, for matches, the projection columns). The optimization axis is
"move less data / spend less per byte read," not round-trips. So exact ranking depends on
**selectivity x needle-count x table-size** and ultimately needs profiling (deferred). The picks below are
flagged low-risk-regardless vs measure-first vs structural.

## Two corrections to an earlier (sub-agent) map — read before acting

1. **The filter read is ALREADY COALESCED, not uncoalesced.** `idx = ctaid*ntid + tid` (`lib.rs:9207`) →
   `addr = resident + filter_offset + idx*4` (`9217-9220`). Consecutive threads read consecutive i32s → a
   warp's 32 loads coalesce into one 128-byte transaction (textbook pattern). **There is no coalescing win to
   chase on the filter scan.**
2. **"Scalar loads leave 50-75% of bandwidth on the table" is overstated.** A coalesced scalar load already
   fills the bus per transaction; vectorization buys instruction-issue / memory-level-parallelism headroom
   (usually modest), not 2-4x. Do not treat it as a large lever without measuring.

## Already good — leave alone
- **Coalesced filter read** (`9217-9220`).
- **Late materialization** — projection columns read only inside the `MATCHED` block, per matched row
  (`9248-9283`). Reading all projections for all rows would be strictly worse.
- **Single-pass** (column read once), single fused filter+project kernel.

---

## Levers (ranked, with honest magnitudes)

### 1. Single-needle / small-set fast path (cleanest common-case win, low risk)
- **Current:** `equal_any` with one needle still runs the general `NEEDLE_LOOP` (`9222-9233`) and loads the
  needle from **global memory** every comparison (`9229`).
- **Change:** a specialized kernel that passes the needle(s) as a **kernel parameter** (no global needle
  array; fully unrolled for N <= ~4) — removes the loop + the global load on the hot path.
- **Why it matters:** `WHERE col = ?` (single needle) is the dominant filter shape. Workload-independent win.
- **Catch:** keep the general kernel for larger needle sets; route by needle_count.

### 2. Warp-aggregated output atomics (matters for HIGH selectivity)
- **Current:** every matched thread does `atom.global.add` on one global counter (`9237`).
- **Change:** `vote.ballot` the warp's match mask + `popc` to count, **one atomic per warp** to reserve a
  contiguous output range, each lane offset by its intra-warp prefix. Up to 32x less atomic traffic.
- **Why it matters:** high-selectivity predicates (matching a large fraction) serialize all matched lanes on
  the single counter. **Neutral at low selectivity** — so its value is workload-dependent.
- **Catch:** the scattered output writes themselves (each match writes to its slot, `9240-9246`) remain;
  warp-agg only fixes the counter contention.

### 3. Shared-memory needle staging (helps, but partly cache-absorbed — do not oversell)
- **Current:** the needle is re-loaded from global memory per comparison, per row (`9229`); a non-matching row
  loads all N needles (the loop only early-exits on a match, `9231`).
- **Change:** cooperatively stage the needle array into shared memory once per block; the per-row loop reads
  from shared.
- **Honest magnitude:** naively R x N global loads, BUT all warp lanes load the same needle address each
  iteration (broadcast) and a small needle set stays hot in L1/L2 — so most are cache hits, not DRAM. The win
  is **modest for small N**; it grows as N exceeds L1 or combines with #5.

### 4. Grid-stride / multiple-elements-per-thread (an ENABLER; measure standalone)
- **Current:** one thread per row, no grid-stride (`9207`; `blocks = ceil(rows/128)` at `9437`,
  `threads_per_block = 128`).
- **Note:** for a bandwidth-bound scan a huge light-thread grid can already saturate memory via cross-grid
  MLP, so the **standalone** bandwidth gain is uncertain — **must be measured.** Its real value is as
  plumbing: a fixed grid with K rows/thread amortizes setup and lets you stage needles (#3) and widen to
  vector loads once per longer-lived block.
- **Catch:** occupancy/register claims are unknowable without compiling — do not assert them; measure.

### 5. Algorithmic needle test for LARGE needle sets (conditional, lower priority)
- **Current:** O(rows x needles) linear scan (`9222-9233`).
- **Change:** large **sorted** needle set → binary search (O(rows*log N)); dense integer range → shared-memory
  bitmap (O(rows*1)).
- **When:** only past a needle-count threshold (large IN-lists); pairs with #3. Skip unless big IN-lists are a
  real workload.

---

## Higher-ceiling, structural (bigger scope — read LESS data)
These beat every micro-opt for selective scans, because the fastest bytes are the ones never read. Both are
**storage-format + planner** changes, not kernel edits:
- **Zone maps / min-max skipping.** Per-chunk min/max metadata → skip blocks whose range excludes every
  needle. The single biggest bandwidth lever for selective predicates on clustered/sorted data.
- **Compression-aware scan** (dictionary / RLE / bit-packing) — scan encoded data to move fewer bytes.

---

## What does NOT transfer from the point-lookup levers (important)
- **Round-trip collapse / over-fetch** (lpb docs): does NOT apply. A scan's match count is unbounded (up to
  row_count), so the count DtoH is genuinely needed to size results, and over-fetch-to-needle_count is
  invalid. A 2-pass count-then-compact would double the column bandwidth — the wrong trade here.
- **Dense per-needle emit** (the dense index kernel): does NOT apply. A needle can match many rows, so there is
  no dense per-needle slot — the scan genuinely needs atomic compaction (hence #2, warp-agg, instead).

## Correctness flag (NOT a perf lever — verify separately)
The main i32 `equal_any` project kernel has **no validity-bitmap NULL handling** — it loads and compares the
raw value (`9220`, `9230`), whereas the `equal_count` / text variants do the bitmap 3VL check. **Verify**
nullable columns route to a NULL-aware variant (or that NULL-as-0 encoding cannot let a needle of 0 spuriously
match NULLs). Parity question, separate from optimization.

## Sequencing (no profiling done — deferred per constraint)
- **Low-risk, workload-independent:** **#1** (single/small-needle param fast path) and the **NULL parity
  check**.
- **Targeted:** **#2** (warp-agg) IF high-selectivity scans matter.
- **Measure-first:** **#3 / #4 / #5** (need profiling or a known large-N workload).
- **Structural ceiling-raiser:** **zone maps** — biggest bandwidth win for selective scans, but a project, not
  a kernel tweak.
- Exact ranking depends on selectivity x needle-count x table-size — a profiling task for when the GPU is free.

## Discipline (charter, non-negotiable)
GPU tests under `timeout`, NEVER `--gpu-reset`; ASCII-only PTX (`ptxas -arch=sm_70` check before launch);
**independent adversarial audit** on any kernel change (#1, #2, #3, #5 touch PTX) — never self-audit; the scan
is on the wave==lpb==scan byte-identity differential, so any change must keep that differential green;
commit/push/merge each verified increment with trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; do NOT run a GPU test right after a
`timeout`-killed one. Coordinate with the agent doing the dense-index-kernel work (it edits the shared
result-assembly + differentials).
