# Review: GPU-native writes proposal + Slice 1 implementation

> **Reviewer:** read-path / index agent (settled the lpb read path to 121.6M lookups/s @b65536; wrote
> `docs/proposals/non-int4-point-lookup-index.md`). **Independent review — does not rely on, reference, or
> reconcile with any other review.** **Originally written 2026-06-30 against `a3f00acc`; reassessed
> 2026-06-30 after the production-append wiring `3c1ec401` (Slice 1b-ii-c).** Scope: the design
> (`docs/proposals/gpu-native-writes.md`) plus the implemented R3 increments
> (`b42858df`/`87c326bb` control-plane O(1); Slice 1a/1b open-shard append through `3c1ec401`). This is an
> integration/direction review — it targets gaps that a slice-local audit does not surface.

## Verdict

**Ship the direction; the implementation is tracking it well.** Control-plane-first by measurement was exactly
right: profile the single-row INSERT, find the real cost is a host O(n) commit-timestamp scan (O(n^2) over a
load, not data-plane work), fix it to O(1) before touching the GPU (303 -> 29 us/row). The design picks
delta/undo MVCC over append-only for the right reason — dense hot columns keep the settled latest-read path
predicate-free and the index single-version. And `3c1ec401` lands the first real data-plane win: a committed
single-row INSERT into a resident int4 table now **appends in place** instead of re-uploading the table —
dual-store tax @16k rows **5774 -> 723 us (~8x), device re-upload eliminated**.

The reassessment below tracks each original concern to its current status. Two of the five are now resolved by
the latest commit; the sharpest one (the index) is half-resolved (correctness fixed, performance open) and is
now better understood as **one of three O(table) terms, only one of which is eliminated**.

## What I verified (this pass)

- `3c1ec401` hooks the append into the **serialized `commit_mutation_at`** (the path `execute_text`/the facade
  actually uses) — and explicitly notes an earlier version wired only `commit_dml_concurrent` (test-only), so
  the append was **inert** until re-measuring the tax caught it. (Strong: the win was *measured on the real
  path*, not assumed.)
- The append runs **before `publish_committed_seq`**, and resident reads bound to the reader's **MVCC
  `snapshot.row_count`**, with the device header used as a liveness proof only (Finding B).
- On append, the wave/lpb GPU index cache is **invalidated** (Finding A) — keys-on-`(col, device_ptr)`,
  generation-blind, so an in-place append (same ptr) otherwise left a stale index. So the index is invalidated,
  i.e. **rebuilt O(table) lazily on the next read** (the host-side build I confirmed earlier:
  `build_wave_resident_int4_index`, `engine_retained_read.rs:701` — D2H -> CPU hash -> H2D).
- The dual-store-tax benchmark (`r3_dual_store_tax.rs`) is unchanged: it still **never reads between writes**.
- The commit states the **sole remaining O(table)-per-commit cost is the `host_rows Arc::make_mut` clone**
  (skipping it makes the tax flat ~40us); correctness needs it because the host-materialization read path reads
  `host_rows`.

## Strengths (affirm — don't regress)

- **delta/undo over append-only**, for the right reason (latest reads stay predicate-free; index single-version).
- **Control-plane-first**: the host O(n^2) commit bug found + fixed by measurement before any GPU move.
- **Append wired into — and verified on — the production commit path**, with a non-vacuity counter
  (`open_shard_append_hits`) and a device-route test that asserts the count advances by exactly 50/50 and that
  an appended key resolves through the index, each gate sabotage-verified.
- **Durability stays WAL-on-NVMe; the GPU is reconstructible, never the durability record.**
- **Slice 7 (retire host store) gated on Slice 6 (GPU-side recovery proven).**

## Concerns — status after `3c1ec401`

### 1. The index is one of THREE O(table) terms; the append killed one, and the benchmark sees none of the rest  [OPEN — sharpened]
The dual-store tax was never a single O(table) cost. There are three:
- **(a) device column re-upload** — **ELIMINATED** by the in-place append. Good.
- **(b) the R1 index rebuild** — on append the index is *invalidated* (Finding A, correct) and therefore
  **rebuilt O(table) on the next read** (host D2H -> CPU -> H2D). Still O(table); still deferred (Slice 5).
- **(c) the `host_rows Arc::make_mut` clone** — O(table) **per commit**, still present (the 723us residual; the
  agent flags it for 1b-ii-d).

The benchmark measures **only (c)** — it never reads, so it cannot see (b) at all, and (a) is already gone. So
the "8x, heading to flat ~40us" number is the commit-side story with the index excluded. A real read-after-write
workload still pays (b). Two asks, unchanged and now sharper:
- **Add a read-after-write probe to the benchmark** so (b) is visible. Otherwise "flat" is a commit-only claim
  and (b) is silently O(table) — the vacuity class the proposal itself warns about.
- **For INSERT, append keys to the index (O(rows)) instead of invalidate+rebuild.** The index is a hash table;
  the appended keys are the same cheap append as the data. Correctness is already handled by invalidate; the
  *performance* fix (no rebuild) is the cheap part and need not wait for Slice 5. (The hard index work —
  tombstone/compaction under UPDATE/DELETE — genuinely belongs in Slice 5.)
- Related: (c) exists because the **host-materialization read path reads `host_rows`** — the same host-result
  coupling the read arc fought. Fully flat writes require the read to stop reading `host_rows` (read device
  directly); (b) and (c) are both downstream of "reads still touch host structures."

### 2. Index-vs-shard encoding contract  [DEFERRED — not yet triggered]
The append currently writes into **one capacity-padded buffer's headroom**, so rows stay flat-addressed and the
`(key<<32)|(row+1)` index encoding still holds. This means #2 is not live *yet* — but the current padded-buffer
implementation is the stepping stone toward (not yet) the design's segmented sealed+open multi-shard target.
When true multi-shard lands, "row" becomes `(shard, slot)` and the index value + cross-shard combine must carry
that dimension. Define the index<->shard contract before the multi-shard rollover, not after.

### 3. Wave-ring write delivery  [ADDRESSED IN IMPLEMENTATION — design note remains]
The implemented append is **host-driven** (`commit_mutation_at` -> `try_append`), not the wave ring — which is
the right call for Slice 1 and sidesteps R2.2c's unsolved at-most-one-kernel + multi-producer-ring problems. The
design's target still routes writes through the wave ring; keep that framed as a later, separately-gated
optimization so Slice 1 is not read as needing it.

### 4. Durability must not hard-depend on GDS  [FORWARD — Slice 6, unchanged]
Slice 6's GPU-state checkpoint via GPUDirect Storage is a fine optimization but a heavy dependency. Keep a
DtoH-then-write baseline checkpoint so RPO-0 / recovery-equivalence — and therefore retiring the host store
(Slice 7) — is not blocked on GDS availability. Make GDS the optimization, not the gate.

### 5. Apply->publish visibility window  [RESOLVED]
`try_append` runs **before `publish_committed_seq`**, and resident reads bound to the reader's MVCC
`snapshot.row_count` (Finding B), with the device header a liveness proof only. So a reader at the prior
`committed_seq` reads only its snapshot's rows and never observes the appended-but-unpublished rows — exactly the
atomically-published-count bound this concern asked for. Resolved.

### 6. Recovery RTO depends on apply/replay throughput  [FORWARD — unchanged]
WAL tail replay re-runs the incremental-write path, so checkpoint cadence must be sized against the **measured**
apply rate. Forward concern for Slice 6.

## New, from the latest work (independent observations)

- **The inert-append catch is the headline process win.** Wiring only `commit_dml_concurrent` (test-only) left
  the append inert; re-measuring the tax caught it. This both validates measure-don't-assume *and* underscores
  concern #1's benchmark point: a gate that doesn't exercise the production read-after-write path can pass while
  the cost is still there.
- **NULL inserts fall back to re-admit (Finding C) — a real scope limit, not just a fix.** An appended NULL int4
  would land in a bitmap-free open shard as a phantom 0 for the device aggregate/DISTINCT/GROUP BY routes, so
  `try_append` declines any NULL-bearing row. Correct — but it means **a table whose INSERTs carry NULLs gets no
  append speedup** (every such commit re-admits). State this scope limit; it narrows where the 8x applies.

## Bottom line

The direction is right and the first data-plane slice is real and verified on the production path (8x, device
re-upload gone), with the visibility window (#5) and the delivery mechanism (#3) both landing the right way. The
one thing not to call "flat" yet is the write path end-to-end: **two O(table) terms remain — the index rebuild
on the post-write read (b) and the `host_rows` clone on commit (c) — and the benchmark currently sees neither in
isolation.** Make the dual-store-tax benchmark read-after-write, and give INSERT an O(rows) index-append, before
the write path is described as O(rows-touched). #2 (index<->shard encoding) and the NULL-insert scope limit are
the contracts to pin down next.
