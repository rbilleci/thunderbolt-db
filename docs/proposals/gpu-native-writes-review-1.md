# Independent design review (round 1): `gpu-native-writes.md`

> **What this is:** an independent adversarial review of the GPU-native writes proposal
> (`gpu-native-writes.md`), per charter ("never self-audit"). Round 1. Reviews the *design*, not the code
> (the proposal cites a separate code/residency audit). Evidence checked against `main`. No code changed.

## Verdict
**Strong, mature proposal — green-light Slice 1 (incremental INSERT).** The version-storage and durability
design is genuinely excellent. One structural gap matters: **this is a version-storage + durability proposal,
not a concurrency-control / write-throughput one — and deterministic CC + the SLO were the headline of R3**
(HANDOVER: "commit/durability **and deterministic CC**, ADR-009 MV-dependency-graph"). Close that before the
doc is accepted as *the* R3 plan.

## What's genuinely strong (don't relitigate)
- **Evidence-first, and the central benchmark is sound.** `r3_dual_store_tax.rs` loads the base untimed
  (auto-admit off → the base load doesn't pay the tax), populates residency once, then times single-row
  inserts vs a non-resident control — a clean, **non-vacuous** isolation of the re-admit tax. The
  260×@16k / O(table) motivation is real. The two O(n)→O(1) commit-path fixes (`b42858df`, `87c326bb`) are
  concrete progress.
- **Delta/undo over append-only is the right call, for the right reasons** — protects the predicate-free
  121.6M latest-read path, keeps the index single-version, avoids dead-version bloat. Well-sourced and
  correctly reasoned.
- **Production-grade durability:** WAL-before-visibility, RPO 0 from the fsync'd WAL (GPU-independent), GDS
  checkpoint for RTO, fuzzy MVCC checkpoint (no write stall), undo reconstructible from the WAL.
- **Target-first / no-throwaway slicing**, and strong validation (latest + old-snapshot +
  `incremental == full-rebuild` byte-identity, non-vacuity counter, no-read-regression gate, HAZARD).

## Substantive concerns (ranked)

### 1. Solves the dual-store *tax*, not the *concurrency-control / throughput* half of R3 (the headline)
The proposal assumes serial commit (`commit_seq` under the commit lock) and relegates concurrency to one risk
bullet. It never addresses whether the commit lock + serial `commit_seq` is a **throughput ceiling** (the same
single-coalescer trap the read path hit), nor engages ADR-009's deterministic spine / MV-dependency-graph. You
could build all 7 slices and still not know whether the write path hits 100k+ TPS concurrently.
**Recommendation:** either (a) re-scope the doc honestly as "write storage + durability (half of R3)" and open
a sibling proposal for deterministic CC / concurrent-commit throughput, or (b) add a commit-concurrency
section — does the lock serialize, or hold only for seq-assign + WAL-append while the GPU applies in
`commit_seq` order via the wave ring? what is the target concurrent-commit ceiling? Now that seq-assignment is
O(1), the architecture may support pipelined commits — but make that case explicitly, don't leave it implicit.

### 2. No concurrent-commit throughput benchmark — only serial single-row
The (sound) dual-store-tax bench measures serial single inserts; the SLO is concurrent TPS. Add the write-side
analog of `r2_wave_engine_ab` (concurrent commit throughput) to the plan, so the target's commit ceiling is
**measured**, not assumed. The current gates verify correctness + flat per-insert cost, never concurrent
throughput.

### 3. In-place UPDATE patch vs the lock-free latest-read path = a torn-row hazard (underweighted)
The read kernels read resident bytes directly with **no visibility check** (that is the 121.6M win), so they
cannot be protected by the creator stamp. An in-place **multi-column** UPDATE is multiple device stores → a
concurrent latest-reader can observe column A new + column B old. A single int4 column is one atomic 4B store
(fine); multi-column is a real hazard. **Recommendation:** for UPDATE, prefer
**append-new-version-to-the-open-shard + tombstone-old** (copy-on-write, consistent with the segmented model;
the old value naturally becomes the undo before-image) over in-place patch — it sidesteps torn rows entirely.
Related inconsistency: stamping `deleted_by` "in place" on a "sealed/immutable" shard *is* an in-place
mutation — sealed shards aren't truly immutable under that rule; reconcile.

### 4. Incremental index maintenance (Slice 5) is a hidden prerequisite for Slices 1b–4's O(rows) claim
Slice 1b uses "index: invalidate → lazy rebuild" = an O(table) index rebuild on the next read. So slice 1b's
"dual-store tax goes flat" gate is only true if you ignore the index, which silently reintroduces O(table).
**Recommendation:** measure the index-rebuild cost in slice 1b's gate (prove it does not reintroduce the tax),
or pull incremental index maintenance forward. As written, slices 1–4 may not deliver end-to-end
O(rows-touched) because the rebuild dominates.

### 5. Single-row commit latency is round-trip-bound — state it (the read-path lesson applies to writes)
"O(rows-touched)" is asymptotic; the constant for a single-row commit is the wave-ring round-trip + on-device
apply ≈ the same ~µs GPU round-trip floor reads have. So single-row-commit *latency* won't beat CPU; the win
is **batched/grouped-commit throughput** (which group-commit + the wave ring naturally provide). State this so
nobody expects sub-µs single-row commits — writes are a batched-throughput play, like reads.

## Minor notes
- **Scope:** incremental writes apply only to resident-supported types (int4, text-in-progress). Non-int4
  tables (bigint/uuid/numeric — see `non-int4-point-lookup-index.md`) stay on the host write path until
  residency covers them. State this scope.
- **Stamp wraparound:** `commit_seq` rebased to 32-bit per shard — bounded shard size bounds wraparound, but
  make the epoch handling concrete (the doc's "or an epoch" is the right instinct).
- **Point of no return:** retiring the host MVCC store (Slice 7) warrants an independent adversarial review of
  the *design* (not just the code/residency audit) before it proceeds. This note is round 1, not a substitute.

## Net
The storage + durability half is excellent and ready to execute — Slice 1 is the right, low-risk, no-throwaway
first step. The missing half is **concurrency control + concurrent-commit throughput**, the part R3 was
chartered to answer. Before this is accepted as *the* R3 plan: re-scope the title (or add the CC section), add
a concurrent-commit throughput benchmark, and reconsider in-place UPDATE patch vs append-new-version for the
lock-free read path (the one design choice most likely to bite correctness).
