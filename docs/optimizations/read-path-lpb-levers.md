# Read-path (lpb) optimization levers — remaining backlog

**Scope:** the launch-per-batch (lpb) GPU **int4 unique-key index-probe** point-lookup read path
(`crates/execution/src/lib.rs` `submit_match_project_i32_index_probe_from_payload` +
`complete_detached_columnar`; engine batched completion in `crates/engine/src/engine_retained_read.rs`).
Companion doc for the batch=1 / small-batch latency deep-dive:
[`small-batch-latency-levers.md`](small-batch-latency-levers.md) (shares levers #1, #2, #4 below).

**Status when written:** 2026-06-28, branch context post-`ec4b1a8c`. Analysis is **code-only** (no benchmarks
run — another agent is actively optimizing; do not collide on the result-assembly code they touch).

## STATUS UPDATE (2026-06-29) — DECISIONS "lpb read levers"
- **#1 (dense per-needle emit) + #2 (single sync) + #4-folded + the `row_indices` elision → DONE + MERGED
  default-OFF + AUDIT SHIP (`3603732f`+`7053743f`).** Built as the dense `gpu_db_resident_i32_index_probe_dense`
  kernel + a `status` field on `CudaI32BatchProjectionColumns` + a SINGLE-PASS engine compaction, behind
  `dense_index_probe_enabled` (+ `dense_index_probe_hits` counter). Byte-identical (4/0 GPU differentials).
  MEASURED: lpb 627µs→411µs @b65536 (−34%, ~160M l/s), beats the wave at ≥b4096 with no persistent kernel; tail
  tightens (max 986→609); b256 ~neutral. **Key simplification found: the index route is ALREADY unique-only, so
  no catalog uniqueness flag was needed — which kernel ran encodes it.** First cut compacted in the drain (a
  regression — two passes); fixed via the single-pass. Flip-to-default is the user's call.
- **#3 (spin-poll for the p99 tail) → OBSOLETE.** Its premise (the ~30ms tail is a blocking-sync issue) was
  WRONG: the tail was root-caused to the per-row `Vec` allocation storm (host allocator), not the sync, and
  FIXED by the columnar drain (`10c724e6`; complete max 9073µs→1785µs). Drain instrumentation showed sync #1 was
  1–6µs even on tail events, contradicting the "host blocks behind other processes' kernels" theory. A spin-poll
  could still help a residual contention tail, but it is no longer the showstopper this lever claims.
- **#5 (avoid the pinned→pageable copy) → MEASURED ~52µs, NOT pursued standalone:** `copy_pinned_into` is ~52µs/
  batch at b65536 (not "minor"), but #1's dense path already removes 2 of its 3 copies (no `needle_indices`/
  `row_indices`); a standalone pinned-lifetime refactor wasn't worth it.

---

## Context — what this session already did (DO NOT redo)

lpb went **3M → 78.5M lookups/s** (b65536, 1M rows, ~26x) entirely by killing host-side result
materialization — the GPU was never the cap. Already landed + audited + merged:

- **Arc-share result schema** (`a59add53`) — the per-needle `columns`/`access_path` deep-clone was the ~3M cap.
- **Flat `RowBlock`** then **batched result model** (`c4b0bddb`, `d49aed6f`) — one flat result + per-needle ranges.
- **O(n) counting-sort scatter** (`3f719e3a`) — replaced an O(n log n) global sort (~80% of assembly).
- **Columnar drain** (`10c724e6`) — killed the per-row `Vec<i32>` boxing; fixed the p99 tail AND +3x; lpb
  overtook the wave for large batches.
- **Slim submission** (`07186f1b`) — dropped the per-needle members `Vec`; submit p50 388us → 28us, +78%.
- **i32 batched result** (`c1a27f3a`) — memory/tail win (~6x smaller result buffer), byte-identical.

Already-optimized infra (don't re-touch): device-buffer **pool** (`lib.rs:335`, no per-batch `cuMemAlloc`),
**pinned host-buffer pool** (`lib.rs:410`), batched flat result + O(n) scatter (`engine_retained_read.rs:1143`).

**Strategic frame (read before spending effort):** per the project's own measurements
(`DECISIONS.md` "R2.2c throughput at scale/threads"), the read path delivers **0.9M–78M lookups/s at
batch>=32 = 2–30x the OLTP SLO**; the only sub-SLO regime is un-coalesced batch=1, which the coalescer
batches away under load. So **the levers below are latency / robustness / host-CPU polish, NOT an SLO
throughput bottleneck.** The unmeasured SLO gate is the write path (R3). Weigh each lever against R3 before
committing. Two exceptions are worth doing regardless (levers #3 and #4 — see "Priority" at the end).

---

## Levers (ranked by value)

### 1. Dense per-needle emit for the unique index (kernel) — kills the atomic + the host scatter
- **Current code:** the probe kernel compacts matches with `atom.global.add` into unsorted emit order —
  `lib.rs:9678-9688` (`atom.global.add.u32 ... st.global.u32 [slot]`). That unsorted emit *forces* the host
  counting-sort scatter at `engine_retained_read.rs:1143-1166`.
- **Change:** a unique int4 index has <=1 match/needle, so thread *i* (needle *i*) can write `result[i]`
  **densely** with a presence flag — no atomic. Output is naturally needle-ordered and sized `needle_count`.
- **Payoff:** removes (a) the atomic contention (a hotspot at high match rates), (b) the entire host
  counting-sort scatter, (c) the `needle_indices` array (slot == needle → one fewer device buffer + one fewer
  DtoH). Also **enables lever #2's over-fetch** (no count needed).
- **Catch:** the same PTX serves the `equal_any` **scan** path (non-unique, >1 match/needle, genuinely needs
  atomic compaction). Add a **submission-type flag** (unique-index vs scan) and either a second kernel variant
  or a branch — the unique route takes dense emit, the scan route keeps the atomic.
- **Verify:** GPU differential wave==lpb==scan still byte-identical, incl. the multirow case
  (`r2_batched_completion_matches_per_needle_multirow`); the scan path must be untouched.

### 2. Collapse the COMPLETE-phase sync round-trips (3 → 1)
- **Current code:** three blocking `cuStreamSynchronize` per batch — `lib.rs:2730` (kernel drain),
  `lib.rs:2787` (after DtoH count), `lib.rs:2846` (after DtoH results). The count sync exists only to size the
  result Vecs (`lib.rs:2814-2816`); sync #1 exists only so the timing event read (`lib.rs:2734`) is valid.
- **Change:** for the unique index, over-fetch results to `needle_count` and DtoH count+results behind ONE
  covering sync (lever #1's dense emit makes the count unnecessary entirely). With timing gated (lever #4),
  drop sync #1. Net: **3 round-trips → 1.**
- **Payoff:** the phase-split proved COMPLETE's ~16us *is* the entire lpb single-flight gap to the wave; this
  closes most of it with no persistent kernel. Biggest win for small/medium batches (round-trip-bound).
- **Catch:** the over-fetch is valid only for the unique index (<=1/needle); gate it with the same flag as #1.
  Keep the scan path's count sync.
- **Verify:** byte-identical differentials; check the error-path drain (`drain_err`, `lib.rs:2759`) still
  syncs before any guard Drop on the collapsed path.

### 3. Non-blocking completion (spin-poll) — fixes the p99 ~30ms tail (DO REGARDLESS)
- **Current code:** completion blocks on `cuStreamSynchronize` (`lib.rs:2730`). `DECISIONS.md` records a
  **lpb-path p99 ~30ms vs p50 3.8ms** blocking-sync tail on the shared box (the host thread blocks behind
  other processes' kernels in the GPU queue); the wave is clean because it polls a host-mapped status flag.
- **Change:** have the kernel set a host-mapped `done` flag (`st.release.sys`, the wave's mechanism in
  `crates/execution/src/wave.rs`) and spin-poll it (bounded, then fall back to `cuStreamSynchronize`); or
  poll `cuEventQuery`.
- **Payoff:** a latency-SLO fix (OLTP p99 SLAs care about a 30ms tail even when throughput is fine — the user
  flagged tail latency as a showstopper). Bonus: a non-blocking wait frees the host thread to enqueue the next
  batch (helps any multi-producer-lpb model).
- **Catch:** memory-ordering — the host must observe the kernel's result writes after seeing the flag; reuse
  the wave's validated `.sys`-release + acquire-poll pattern. **Independent audit required** (kernel/protocol).
- **Verify:** hazard-test (repeated runs, zero CUDA 700/716/717); confirm the tail under concurrent shared-box
  load, not just steady-state.

### 4. Gate / sample the per-batch event timing (cheap; UNLOCKS lever #2's sync #1 removal)
- **Current code:** `timed` is on whenever the pooled stream has events (`lib.rs:9842`); per batch it does
  2x `cuEventRecord` (`lib.rs:9881`, `9901`) + `cuEventElapsedTime` (`lib.rs:2734`). The elapsed read is the
  *reason* sync #1 (`lib.rs:2730`) must stay.
- **Change:** gate timing behind a flag (or sample every Nth batch) so production batches skip it.
- **Payoff:** removes the records + the elapsed read (which itself stalls on the event), and lets lever #2 drop
  sync #1. Higher relative impact at small batches. Near-free.
- **Catch:** keep timing available for benchmarks/probes (don't delete the path, gate it).

### 5. Avoid the pinned→pageable result copy / guarantee the pinned path (minor)
- **Current code:** the async path stages DtoH through pooled pinned buffers then `copy_pinned_into` the
  pageable result Vecs (`lib.rs:2847-2849`); if async symbols are absent it blocks straight into pageable
  Vecs (`lib.rs:2798-2806`, `2850+`).
- **Change:** size the pinned pool so the hot path never falls to pageable; consider letting downstream consume
  the pinned buffer directly to drop the extra `copy_pinned_into` host memcpy on large batches.
- **Payoff:** removes a host memcpy from large-batch complete; avoids a slow pageable copy in the tail. Minor.

---

## Priority / sequencing

- **Do regardless of the R3 pivot:** **#3** (the p99 ~30ms tail — a latency-SLO defect already called a
  showstopper) and **#4** (gate timing — near-free, removes per-batch instrumentation from production).
- **Do as one clean bundle IF lpb is confirmed the read default or small-batch latency is wanted:**
  **#1 + #2** (the "unique-index dense fast path" — kernel dense emit + single-sync complete). This is the
  last structural gap to the wave's single-flight number, with none of the persistent-kernel cost. See the
  companion small-batch doc for how this composes with the batch=1 levers.
- **#5** is opportunistic polish.
- Everything here is read-side polish; the **write path (R3)** is the unmeasured SLO gate — do not let the
  read backlog displace it.

## Discipline (charter, non-negotiable)
GPU tests under `timeout`, NEVER `--gpu-reset`; ASCII-only PTX (`ptxas -arch=sm_70` check before launch);
**independent adversarial audit** on any kernel/protocol change (never self-audit) — applies to #1, #2, #3;
commit/push/merge each verified increment with trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; do NOT run a GPU test right after a
`timeout`-killed one (its kernel zombies ~watchdog/backstop). Coordinate with the active result-assembly agent
before editing `engine_retained_read.rs` completion code.
