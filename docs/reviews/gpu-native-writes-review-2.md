# Review: GPU-native writes proposal (`docs/proposals/gpu-native-writes.md`)

> **Reviewer:** read-path / index agent (the agent that settled the lpb read path to 121.6M lookups/s @b65536
> and wrote `docs/proposals/non-int4-point-lookup-index.md`). **Date:** 2026-06-30. **Reviews:**
> `docs/proposals/gpu-native-writes.md` (last touched `a3f00acc`) + the merged R3 increments
> `d37266f7`/`b42858df`/`87c326bb` and the in-flight Slice 1a/1b open-shard append work, branch
> `phase0-m1-engine-facade`. This is an *integration + direction* review (the per-slice opus adversarial
> audits are separate and were adopted per slice); it focuses on gaps a slice-local audit would not catch.

## Verdict

**Ship the direction.** This is high-quality work — arguably the strongest proposal in the tree. Two things
were done exactly right: (1) **control-plane-first by measurement** — rather than jumping to the GPU, the agent
profiled the single-row INSERT, found the real cost was a host O(n) commit-timestamp scan (`O(n^2)` over a
load, *not* data-plane work), and fixed it to O(1) before touching the GPU (`303 -> 29 us/row`, flat); (2) the
design picks **delta/undo MVCC over append-only** with real justification (dense hot columns keep the settled
latest-read path predicate-free; the index stays single-version), grounded in the CMU/Peloton survey +
HyPer/InnoDB/PG. Slicing is no-throwaway and the durability gate is in the right place (Slice 7 retire-host-store
gated on Slice 6 proving GPU-side recovery). The concerns below are gaps, ordered by value, not objections to the
direction.

## What I verified (grounding)

- The dual-store-tax benchmark (`crates/engine/examples/r3_dual_store_tax.rs`) times single-row INSERT commits
  with auto-admit ON, vs base resident size. It **never reads/probes between writes** (no `select`/probe in the
  timed loop — confirmed).
- The R1 point-lookup index build (`build_wave_resident_int4_index`, `engine_retained_read.rs:701`) is
  **host-side and O(table)**: D2H the key column -> CPU hash-table loop -> H2D the table. The index route is
  default-on (the lpb decision). A write bumps the generation (`invalidate_relational_residency_*`,
  `engine_commit.rs:235/257`), and the index is rebuilt lazily on the next read.

## Strengths (affirm — don't regress these)

- **delta/undo over append-only**, justified: latest-snapshot reads stay predicate-free, so the 121.6M
  lookups/s read path is untouched; the index stays single-version. Correct for a dense GPU column store.
- **Control-plane-first**: found + fixed the host O(n^2) commit bug by measurement before any GPU move.
- **Durability stays WAL-on-NVMe; the GPU is reconstructible, never the durability record.** Right — GPU memory
  is volatile; the design does not pretend otherwise (WAL-before-visibility, RPO 0 from fsync).
- **Slice 7 (retire host MVCC store) gated on Slice 6 (GPU-side recovery proven equivalent).** Don't delete the
  source of truth until the replacement's recovery is green.
- **Non-vacuity counters on the fallback path** — applies the wave faked-throughput lesson.

## Concerns (ordered by value)

### 1. The index is a SECOND O(table) term, and the benchmark hides it  [sharpest]
The dual-store-tax benchmark measures only the **residency re-admit**; it never reads between writes, so it never
triggers the **R1 index rebuild**, which is *also* O(table) (host D2H -> CPU hash -> H2D) and fires on the first
read after any write's generation bump. So when Slice 1b makes that benchmark "go flat," that is a flat
*re-admit*, not a flat *write* — a read-after-write workload still pays O(table) for the index until Slice 5.
- **Action a:** add a **read-after-write probe** to the benchmark so the index cost is visible. Otherwise the
  "flat" gate is partly vacuous — the exact vacuity class the proposal warns about.
- **Action b:** **pull INSERT index maintenance forward.** Appending new keys to the existing hash table is
  O(rows) — the same cheap append as the data, not a rebuild — so it does not belong in Slice 5. The genuinely
  hard index work (tombstone/compaction under UPDATE/DELETE) does belong there; INSERT-append does not.
  (See `docs/proposals/non-int4-point-lookup-index.md` and its "Index lifecycle post-R3" reasoning: GPU-native
  incremental index build = atomic slot-claim insert, no column D2H.)

### 2. The index encoding is coupled to the row model Slices 1a/1b change
The lpb index packs `(key << 32) | (row + 1)` over a **flat u32 row**. Once the layout is segmented (sealed +
open shards), "row" becomes `(shard, slot)` — the index value needs that dimension and the cross-shard
read-combine must map index hits back to shards. Changing the row model in 1a while leaving the index on the flat
model until Slice 5 means either the lazy rebuild already understands shards, or freshly-appended open-shard rows
are **not index-addressable** (silently forcing a scan for the newest rows — a read regression on exactly the hot
data). Make explicit which, in Slice 1b.

### 3. Wave-ring write delivery re-acquires R2.2c's unsolved problems
Target point 4 routes writes through "the wave engine's host-pinned lock-free ring + persistent kernel." The wave
engine was *retired* for reads, and R2.2c gated it on precisely (a) at-most-one-persistent-kernel coexistence and
(b) the multi-producer ring — both unsolved. The implemented Slice 1a append is host-driven HtoD, which is
correct and pragmatic. **Frame host-driven append as the Slice 1 reality and the wave ring as a later,
separately-gated optimization** — so Slice 1 doesn't imply the wave ring (it doesn't need it) and you don't
re-inherit that baggage prematurely.

### 4. Don't hard-depend durability on GDS
Slice 6's GPU-state checkpoint via GPUDirect Storage is a fine optimization but a heavy dependency (cuFile stack,
NVMe config, may be absent on the shared box). The baseline — DtoH then host-write — is slower but always works
and is enough to prove RPO 0 + recovery equivalence. **Make GDS the optimization, not the gate**; otherwise Slice
6 (and thus retiring the host store, Slice 7) is blocked on GDS availability.

### 5. Be explicit about the apply->publish visibility window
A latest reader concurrent with a commit must not see rows appended in step 3 before `committed_seq` is published
in step 4. The design says latest reads pay "at most one inline-stamp compare" — so it is a
`created_by <= read_txn_id` filter, *not* truly predicate-free, and the read kernels currently have **zero**
visibility logic. State whether open-shard reads are bounded by an **atomically-published row_count** (simplest
correct: read only the first N rows as of `read_txn_id`) or by the `created_by` stamp — and ensure that bound
exists from **Slice 1b, not 1c**, or there is a window where an old-snapshot reader sees future rows. (The
in-flight "capacity-aware resident read offsets" work may already be this — confirm the count publishes
atomically with `committed_seq`.)

### 6. Minor: recovery RTO depends on apply/replay throughput
WAL tail replay re-runs the incremental-write path, so the checkpoint cadence must be sized against the **actual
apply rate**, not assumed. Fixing writes fixes replay, but the RTO budget (< 5 min) is a function of (checkpoint
load + tail replay) and the tail replay is bounded by how fast the GPU apply path ingests.

## Bottom line

Direction and execution are both strong; I would not let only one thing slide — **#1**: the index is a real
second O(table) term, it is cheap to fix for INSERT (an O(rows) key-append, not a rebuild), and the current
benchmark cannot see it. Make the dual-store-tax benchmark read-after-write, and fold INSERT index-append into
Slice 1b/2, before calling the write path "flat." **#2** and **#5** are correctness details to pin down in the
same neighborhood (the index/visibility contract for open-shard rows). **#3**/**#4** are scoping guards so Slice
1 stays host-simple and Slice 6 stays unblocked.
