# ARCHIVED — PLAN before task-ledger unification (2026-07-12)

> Historical sequencing record. It is not executable. The sole current work ledger is `docs/PLAN.md`.

What to build next, in order. The **rules** are in [CHARTER.md](../../CHARTER.md); the **why** in
[DECISIONS.md](../../DECISIONS.md); the **design** in [ARCHITECTURE.md](../../ARCHITECTURE.md); what's **done** in
[STATUS.md](../../STATUS.md); the **resume baton** in [HANDOVER.md](../../HANDOVER.md). This doc owns **sequencing**.

## 0. The fork that orders everything
The OLTP bet (DECISIONS ADR-008) is **unproven until measured**. Two near-term tracks compete for "first":
- **Prove the bet** — build the open-loop benchmark vs tuned Postgres (§1). *Recommended first: it tells you which
  architecture work actually matters.*
- **Unblock production** — STRATA auto-admission (§2), without which the entire GPU read path is dormant.

## 1. Benchmark mandate (prove or kill the OLTP bet)
**v1 landed (2026-06-27):** `engine/examples/oltp_auto_admit_ab` (per-op A/B) + `facade/examples/oltp_batched_read_scaling`
(concurrent batched-vs-host). First findings in DECISIONS ADR-008 — they already redirected priority: the point-read
gap is **host-side serial coalescer cost (~15µs/item)**, so §3's wave engine / coalescer fix is the measured critical
path, ahead of more resident-route breadth (S-C/S-D/S-E). Still owed below:
Before deep OLTP-engine investment:
- **Open-loop / offered-rate** harness measuring **p99 / p99.9 / p99.99 at a target TPS** (today's numbers are
  closed-loop / self-throttling — they cannot validate a latency bet).
- A **tuned CPU OLTP baseline on the same box** (Postgres; ideally an in-memory engine too), on a real OLTP
  workload (**TPC-C / sysbench-oltp / YCSB-A**).
- Report **split by transaction class** (CHARTER): the deterministic fast path vs the interactive slow class.
- A win = GPU beats the tuned baseline on p99 at TPS. The result tells us where GPU loses today (launch? PCIe?
  concurrency control? index?) and which of §3 to build first.

## 2. STRATA — make the GPU path the production default (DECISIONS ADR-010)
Spec: ARCHITECTURE §7 + §13.
- **S-A — vocabulary rename ✅ DONE (2026-06-27).** `RelationalResidentPartition → RelationalResidentShard`,
  `residency.partitions → shards`, `partition_device_memory → shard_device_memory`, `partition_id/count → shard_*`,
  route shapes `partitioned_* → sharded_*` (engine + observability; MVCC tuple-store "partition" left intact).
  Behavior-preserving, suite 729/0.
- **S-B — admission producer v1 (N=1 unified) behind `auto_admit_on_commit` ✅ DONE (2026-06-27).**
  Commit-triggered (all 3 commit paths), post-`publish_committed_seq`, best-effort (never fails the durable
  commit), via the `&self`+held-catalog-guard seam (`populate_relational_residency_snapshot_inner` /
  `admit_..._inner`). It originally landed default-OFF. Acceptance proves a CREATE+INSERT(with
  NULL) table is GPU-resident with **no explicit warm**, reads on the GPU route, results match the host path;
  HAZARD clean. The full pgwire-socket golden is also closed (2026-07-12).
- **S-C — same-GPU N>1 shards + partial-combine; incremental shard append ✅ DONE.** Fixed-width, bool, text,
  NULL/version sidecars, rollover, zone pruning, and device PK routes are live.
- **S-D — text/mixed-type shard recompaction/combine ✅ DONE.** Offset rebasing, bitmap gather, and 8-alignment
  are covered across read/write and streaming paths.
- **S-E — streaming executor + cross-shard/multi-GPU partial combine ✅ DONE (2026-07-12; ADR-012).** Scalar,
  projection, grouped/distinct, ordered top-N, JOIN/OUTER JOIN, and rank/window reads run in byte-bounded chunks.
  Pinned byte replay, spill, LRU eviction, one-chunk async lookahead, exact/Bloom keyed skipping, and chunk-primary
  maintenance are live. Scalar/projection/grouped chunks round-robin across healthy, fully-budgeted GPUs; only
  framed results/associative partials return to the coordinator. A secondary-GPU completed-chunk counter prevents
  vacuous routing claims; the two-device execution gate self-skips on this one-GPU workstation.
- **S-F — default GPU admission ✅ DONE (2026-07-12).** `auto_admit_on_commit` is ON in production; ordinary
  semantic tests use an explicit CPU-oracle constructor. Recovery suppresses per-record admission/elision, replays
  the durable store completely, then bulk-admits once. Absent an explicit budget, production uses 80% of physical
  VRAM. Replacement payload + row identity allocate before two-phase deterministic eviction; allocation/fit failure
  leaves the published resident set untouched. One allocation transaction serializes actual payload, version/identity
  region, rollover, and lazy device-index accounting; optional indexes decline to the resident GPU scan at the cap.
  The production mixed gate, with no explicit auto-admit enable, re-passed at 124.2k reads/s, p50 226us, zero host
  gathers/fallback groups, and 160/160 host-install-elided writes. Final STRATA re-audit: MERGE-SAFE, 0 Critical /
  0 High; the same allocate-first transaction also covers all three public benchmark chunk/shard installers.

- **S-E.P4 — CHUNK-AUTHORITATIVE TABLES ✅ SHIPPED 2026-07-11 (P4-1 `01520c2d`, P4-2a `7a9a5b5b`,
  P4-2b-i `4a22ef91`, P4-2b-ii `e5f17765`, P4-3 `144a7494`; P4-4 moot under freeze-not-drop; P4-5 =
  ledger closure — sidecar compaction + frozen-store reclamation REGISTERED behind the
  min-active-read-boundary fence; the arc balance sheet is in HANDOVER).** Original design (revised per
  the adversarial review):** P1/P2/P2b/P3 shipped the *durability and read* primitives; the review PROVED the
  write-side primitives are NOT reusable store-free (the P3 locate derives identity from a store scan; the
  P2 stamp is driven by the store-generation change-log diff and old/new chain classification) — P4 builds
  chunk-native twins first. THE CLASS (no flag — intrinsic, sticky, self-entered): cold entry exists + NO
  PK/UNIQUE + NO FK edge in either direction + the RUNTIME over-budget/not-admissible test (review H1: the
  catalog "elision-ineligible" predicate is self-contradictory — a no-constraint int4 heap IS
  elision-eligible; type-ineligibility ⟺ chunk-unencodability). ENTER at the commit hook under an EXPLICIT
  mutual-exclusion interlock with the elision ENTER (H1: a post-VACUUM budget change can trip both in one
  commit). RUNTIME-DERIVED: recovery replays the WAL into the store normally and the class re-enters
  (capture → drop rows); the artifact is the warm start. RECOVERY MATRIX INTERLOCK (H3): the store-row drop
  AND any WAL truncation covering class-table writes are gated on a DURABLE cold artifact covering the
  truncated suffix — else `store-dropped + WAL-truncated + artifact-missing` is unrecoverable.
  **SLICES (re-scoped):**
  - **P4-1 THE REVERSE GATHER — a NEW host columnar decoder (C2: not an elision twin; the elision
    rehydrate reads resident device shards which over-budget tables do not have).** Decode ColdPayload
    bytes (int4/int8/b128 sections, text blobs, bool bitmaps, null bitmaps) + sidecar masks back into
    store tuples. IDENTITY (C2): chunks carry no per-row ids — fresh row_ids are synthesized on rebuild
    (sound for the no-PK/no-FK class; `advance_row_id` discipline preserved). Charter (M4): this is NEW
    registered host debt — ledger row with deletion trigger = the device-index-over-chunks route. Gate:
    store → chunks → drop → rebuild → byte-identical reads differential.
  - **P4-2a CHUNK-NATIVE LOCATE + STAMP (C1, new primitives):** a device locate over the CHUNKS
    THEMSELVES yielding (chunk_idx, slot) — chunks gain a synthesized slot-identity at upload (the P3
    __row_id pattern applied to chunk sources instead of store scans) — and a locate-driven stamp path
    (no store diff, no chain classification). INSERT tail-append builds from the STATEMENT's rows.
  - **P4-2b THE CLASS + WRITE PATH:** apply skips the store install (elision early-return pattern;
    row-id allocator still advances); the chunk patch is the materialization: INSERT = tail from
    statement rows; DELETE = P4-2a locate+stamp; UPDATE = stamp-old + tail-append-new at a fresh row id.
    Patch failure de-authoritizes via P4-1 (never fails the acked commit). SERIAL-ROUTE GUARANTEE (M1):
    class tables are forced onto the serial apply (the lane pump's covered-insert wave assumes a resident
    open shard + PK by-key resolve — both absent here). COMMIT-LOCK BOUND (M2): bulk tails are chunked
    with the patch bounded per commit-lock hold (ledger row; the elision precedent is incremental device
    appends). RYW (M3): multi-statement transactions that write-then-read a class table de-authoritize
    (v1) — intra-txn tail staging is the later lift.
  - **P4-3 READ COMPLETENESS + THE MVCC GATES (C3, the hard one):** validity stops being generation-Arc
    equality (meaningless without store installs) — an is-chunk-authoritative check + boundary rules.
    PER-CHUNK BORN GATE: a reader skips chunks with `payload_copin_s > reader_copin_s` (tail appends are
    born at their boundary; the field exists, the gate must use it per-chunk). Deletes below the reader
    already serve via the sidecar (`deleted_by > rtx`). ENTRY QUIESCE RULE: class entry requires the
    global-min pinned read snapshot >= the entry boundary (a reader straddling the capture would need
    rows the scan never emitted — unrecoverable from chunks); a straddling reader post-entry
    de-authoritizes LOUDLY (counted; not the steady state). The CPU-pinned fallback must NEVER silently
    read the empty store (the elision-guard pattern at the dispatch seam, extended to this class).
  - **P4-4 RECOVERY SHORT-CIRCUIT:** at the P1 seam a restored class table drops its just-replayed store
    rows after the suffix patches land; the store-mode fallback stays; plus the H3 truncation gate.
  - **P4-5 THE DELETION SWEEP:** VACUUM sidecar compaction (device gather of surviving slots); the
    cold-tier/scan-build ledger row CLOSES (scan-build = bootstrap/de-auth import only); the M4 reverse-
    gather row OPENS; host-debt balance sheet.
  **NON-ISSUES (review-confirmed):** SI ledger reads (table,row_key)+unique slots only — a no-PK class
  records none, correct no-op; sequences/DEFAULTs read the catalog; no triggers. **DDL SEAM (H2):** the
  DDL preflight sweeps class tables through P4-1 BEFORE any `visible_relational_rows` read (ADD
  PK/UNIQUE/CHECK/FK would validate vacuously against an empty store) — the elision rehydrate-sweep seam,
  extended. **EXCLUSIONS:** uniqueness-constrained tables (device-index/fingerprint route later), FK-edged
  tables, elided tables, multi-GPU.

- **S-E.P5 — THE DEVICE INDEX OVER CHUNKS (keyed chunk-authoritative tables; designed 2026-07-11,
  REVISED per the adversarial design review — NEEDS-REVISION, all findings adopted).** Lifts the class's
  no-PK/UNIQUE exclusion for VRAM-RESIDENT-INDEX working sets (review H1: admitting ANY keyed table makes
  a point INSERT O(all-chunks) — probe-every-chunk + LRU thrash under the commit mutex — a REGRESSION vs
  the host value_index exactly on the over-VRAM tables the class targets; over-VRAM keyed tables need a
  chunk-skipping structure (per-chunk key bloom/zone-map pruning) — a LATER slice). RECON: every kernel is
  source-agnostic (`submit_compound_fold_fingerprints` raw base_ptr; write-locate/dense-probe take
  {index_ptr,mask,shift}; `build_int4_pk_hash_table_host_visible` + `retain_device_memory_copy` = the
  persistent off-budget buffer pattern); only the shard cache wrappers need chunk-shaped twins; the
  in-place INSERT kernel does not apply (chunks immutable — rebuild-per-payload, never insert-in-place).
  **DESIGN (revised):**
  - **CHUNK IDENTITY (review H3 — a correctness landmine):** ColdChunk gains a monotonic `chunk_id`,
    allocated ONLY at the two genuine-payload constructors (tail append; compaction survivors) and
    PRESERVED by every verbatim clone (re-pin, base-clone, untouched-clone) — fresh iff the payload bytes
    are new (a compacted chunk inheriting its source's id would serve the OLD index over NEW bytes:
    probe-slot misalignment = silent false-negative dup = C2). `chunk_id` (content identity, the cache
    key) COEXISTS with positional `chunk_idx` + `entry_epoch` (the stamp coordinate token) — never
    conflate; never cache a chunk_idx across entry Arcs (L3).
  - **THE INDEX:** per (table, chunk_id, key_id): device-fold fingerprints → all-visible hash table →
    one persistent retained buffer (~16B/row), with NEW explicit VRAM accounting + cap + LRU (eviction
    frees; rebuilt on demand). BUILD AT CLASS ENTRY, not lazily on the insert path (review H2: a lazy
    first-probe build from a SPILLED payload = an NVMe read under the ONE commit mutex stalling every
    table's commits); eligibility (H1) requires the full index set to fit the cap.
  - **THE RECHECK IS DEVICE-SIDE (P5-4 charter correction):** a probe hit selects candidate CHUNKS only;
    the full key tuple, sidecar visibility, and any residual predicate run through the device predicate VM.
    Only device-approved coordinates/row images cross back. The old per-slot DtoH + host compare was a charter
    breach and is deleted by P5-4. A masked (dead) hit is not a conflict.
  - **INSERT UNIQUENESS:** in-batch exact predicates run over one transient device relation (bounded at 256
    rows until the device exact tuple-hash/group replacement); existing rows use one multi-chunk candidate
    locate + exact device predicate. NULL uses the raw payload-placeholder fingerprint + exact device `IS NULL`.
    **UPDATE SELF-EXCLUSION (review C1):** the class UPDATE stamps its old coords in the COMMIT
    HOOK, so at probe time the old version is still live — the new-image probe must SKIP hits whose
    (chunk, slot) coordinates are in the update's own located set (the chunk-space `touched_keys`
    analog); probe/recheck at the statement snapshot, self-exclude by coordinate — a hit OUTSIDE the set
    is a genuine dup, INSIDE is the update-reinsert. Without this every same-key UPDATE false-rejects.
  - **DURABILITY INVARIANT (review C2):** a recheck FALSE-ACCEPT is RPO-VIOLATING, not merely wrong-now —
    the duplicate is durable in the WAL and replay's HOST validate rejects it: an acked commit becomes
    unreplayable. The P5-2 gate MUST include a REPLAY DIFFERENTIAL (adversarial fingerprint
    near-collisions through the chunk path → crash → replay through the host path → identical
    accept/reject sets), not just a live twin.
  - **BY-KEY DML:** Eq-on-key locates via the probe (range WHERE keeps the fold). Serial-route reliance
    is LOAD-BEARING (L1): the probe runs under the commit mutex; any future off-mutex prober must adopt
    the ensure_shard_pk_device_index lock discipline.
  - **P5-4 RE-SCOPED (review angle 7):** P5 does NOT delete the reverse gather — it survives for COLD
    de-auth/DDL sweeps; P5's honest contribution is keeping it OFF the hot path (the device recheck).
    The ledger row's trigger is re-worded accordingly.
  **SLICES:** P5-0 ✅ `8b1621d4` the device slot-addressed chunk recheck (decode differential vs the
  P4-1 decoder); P5-1 ✅ `cddce252` the chunk-index cache (chunk_id + build-at-entry +
  accounting/cap/LRU); P5-2 ✅ `629452c9` INSERT/UPDATE uniqueness at all four choke points + the C1
  self-exclusion + the C2 replay differentials (int4 + text) + the H1/H2-gated eligibility lift + the
  covered-route class refusal (audit MERGE-SAFE zero C/H); P5-3 ✅ `40f6cf89` by-key DML locate; **P5-4 ✅
  charter closure (uncommitted worktree)** (device exact predicate/visibility, device uniqueness threshold verdict,
  structural NULL, same-entry off-lock invariant, 256/257 deauth seam; third independent audit MERGE-SAFE 0 C/H/M,
  full serial GPU sweep 974/974); **P5-later ✅ COMPLETE (2026-07-12, uncommitted):** when the
  complete exact index set exceeds 256 MiB, compact per-chunk Bloom filters (8 bits/row, 3 hashes,
  64 MiB global cap) select candidate chunks in one GPU launch; the existing full device
  predicate + visibility pass remains authoritative. Cold capture primes spill-backed sets only
  after releasing the commit mutex; class entry never performs NVMe reads. Reservation/rollback,
  mode/deauth/eviction/compaction cleanup, and entry-epoch race retirement keep the cap exact.
  Gates force an all-positive Bloom, global-cap collision, spill-backed entry, compaction ID
  replacement, and E1-prime/E2-publication interleaving; hash sabotage bites; HAZARD 3+2 clean.

- **Production mixed GPU read/write non-vacuity gate ✅ (2026-07-12, uncommitted):** the real facade
  `PointLookupBatcher` runs against concurrent facade INSERTs on a sharded int4-PK table with an unprojected
  NULL. The gate requires zero per-query fallback groups, a nonzero GPU-probe delta inside the exact writer-active
  interval, every committed write to overlap readers and elide host install, and resident append-wave activity.
  Dense on-device created/deleted visibility then closed the append-window tail: three-run median 121.2k reads/s,
  p50 234us, p99 489us, p99.9 644us, writer-active p99.9 717us; zero host gathers/fallback groups and 160/160
  write elisions. All simple-OLTP SLOs pass. Final independent audit of the original gate:
  MERGE-SAFE 0 Critical / 0 High / 0 Medium / 0 Low.

**Golden wire test ✅ (2026-07-12, uncommitted):** drives SQL over the real pgwire socket
(`crates/server/tests/pgwire_roundtrip.rs` pattern), assert exact rows + that the GPU sharded route served them
(non-vacuous), differential vs 1/N shards/explicitly non-resident host, **with NULL data**, GPU-guarded.

## 3. OLTP execution engine (ARCHITECTURE §9 — the benchmark says this is the critical path for the bet)
- **Point-read coalescer (the measured bottleneck, DECISIONS ADR-008):**
  - **Tier 1 ✅ DONE (2026-06-27):** per-shape resident-read template (prepare once, reuse across needles) — removed the
    per-request plan/bind. 68k → 156k ops/s (2.3×), now scales; CPU gap 11× → ~4.5×. The template is the ingress the
    wave engine reuses. (Tier 2 = parallelize the single coalescer thread is **dropped** — the wave engine replaces it.)
  - **Residual:** ~6µs/item = result materialization + oneshot distribution, still single-coalescer.
- **Persistent-kernel wave engine + lock-free submission ring** (replace launch-per-batch); on-GPU result slots remove
  the remaining host per-item orchestration — the real path to the CPU ballpark (moves the bottleneck host→GPU, where
  it scales with hardware). **Started (2026-06-27): design + infra recon done.** Reuses Tier-1's template as the request
  descriptor + the existing CUDA FFI / context / module-cache / stream-pool / pinned-host pool / int4 `equal_any` kernel.
  Greenfield = (1) host-pinned **device-mapped** ring (`cuMemHostAlloc`+`DEVICEMAP`, add `cuMemHostGetDevicePointer`),
  (2) persistent kernel loop, (3) **clean-exit doorbell**, (4) result-slot layout. **Biggest risk = the exit** on this
  `--gpu-reset`-denied shared box (a hung kernel zombies the context). Mitigation: doorbell **plus a hard iteration-cap
  backstop** so the kernel ALWAYS self-terminates; single block (1 SM); lock-free atomics only. **Increment staging:**
  **1a ✅ DONE (2026-06-27)** = bare lifecycle proven: `crates/execution/examples/wave_lifecycle_probe.rs` — a persistent
  kernel polls a device-mapped doorbell, advances a heartbeat, and EXITS on the doorbell in **~3.5µs** (with a
  `%globaltimer` 30s wall-clock backstop as the zombie-prevention net). **9/9 clean lifecycles across 3 processes, no
  zombie context.** The biggest risk (clean exit on the `--gpu-reset`-denied box) is de-risked. Add
  `cuMemHostGetDevicePointer` to the engine FFI for 1b. **1b ✅ DONE (2026-06-27)** =
  `crates/execution/examples/wave_dataplane_probe.rs` — persistent-kernel threads lock-free-claim requests
  (`atom.add`, no barriers), scan a resident key column, gather a payload, and write `(value<<32)|done` as one atomic
  8-byte store; host enqueues needles + reads packed slots (no per-request host materialization). **The host-serial
  bottleneck is GONE — the bottleneck moved host→GPU (scales with hardware = the bet).** Independently audited: the
  number is REAL (reproduced ~9.5–10.1M across 20 runs), correctness SOUND (proven non-vacuous via sabotage variants);
  fixed a `membar.sys` ordering gap (held the number) + broadened not-found sampling. **Honest scan-knee curve (8192
  threads):** small tables are atomic-ceiling-bound (4k:10.0M, 20k:8.3M req/s), but the full-scan is O(rows) so
  100k:2.7M, 500k:755k, **1M-row table: 485k req/s** (~3× the 156k cap, ~CPU-ballpark, scan-bound). *Caveats:* bare
  data plane (no slot→wire mapping yet — parallelizable, not serial); vs the 156k batcher is apples-to-oranges
  (omits the full facade + neutral mapping).
- **1c ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_index_probe.rs` — a GPU hash index (host-built
  open-addressing, Fibonacci hash, in-kernel probe + gather, bounded probe count) **removes the O(rows) scan →
  ~10.5M point lookups/s FLAT across 1M/4M/16M-row tables (O(1)), ~13.6× the CPU's 770k, independent of table size.**
  Atomic-ceiling-bound now. **This validates the OLTP point-read bet at the data-plane level: the GPU does millions
  of lookups/s at realistic scale, residual bottleneck is GPU-architectural (scales with HW).**
- **1d-i ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_devatomic_probe.rs` — moved the claim/`completed`
  atomics to **device memory** (last completer sets a host-mapped `all_done` flag; device counter DtoH-verified):
  **read ceiling 10.5M → ~30M req/s (~2.9×), 3× stable, all correct.** New limiter = device-atomic contention on the
  single counter (peaks at LOW thread count, 512–1024; → batched/striped claiming next). Also quantified the
  **slot→wire mapping**: 200k packed slots → neutral rows in ~180µs–1ms single-threaded (~200M–1.1B rows/s), far below
  the ~7ms GPU drain + embarrassingly parallel → the bare-data-plane caveat is MINOR. (NB the `all_done` cross-thread
  ordering wants an independent audit before it's lifted into the engine.)
- **1d-ii ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_batchclaim_probe.rs` — **batched claiming**: each
  thread reserves K requests per `atom.add(claim, K)` + bumps `completed` once/batch under one `membar.sys`. Sweet spot
  **K=8 → ~45–53M req/s** (~1.5–1.75× over 1d-i's 30M, **~5× the original 10.5M, ~60–69× the CPU**), 3× stable. K is a
  balance (K≥32 collapses: fewer batches than threads → under-parallel + serial host-mapped writes). **Read ceiling is
  now firmly tens-of-millions; further gains need a structural lever (sharded per-block counters / cheaper result
  writes) — diminishing, tuning-sensitive. The read half of the bet is SETTLED.**

### Write path (the frontier) — probing
- **Write probe 1 ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_index_insert_probe.rs` — **concurrent
  lock-free index INSERT** (the novel, historically-hard piece: many threads `atom.cas.b64`-install (key,row) into a
  shared open-addressing table, no locks). All keys verified inserted exactly once (no lost/dup/torn). **~tens of
  BILLIONS of inserts/s** (1M: ~29G/s wall-clock, ~77G/s after removing the ~21µs launch floor; 16M/256MB L2-spill:
  4.2G/s). **Conclusion: GPU concurrent index maintenance is NOT a bottleneck.** The remaining write constraints
  (durability/WAL fsync, deterministic CC) are host-I/O + coordination problems CPU OLTP engines face too — the GPU
  isn't disadvantaged there. *Caveats:* low contention (sequential keys + Fibonacci spread); raw insert only (no
  commit/durability/MVCC/CC); synthetic keys. → **NEXT write probes:** contended inserts; the **commit/durability**
  path (group commit — the likely real write floor); deterministic CC for conflicts.
- **SM-coexistence gate ✅ DONE (2026-06-27)** = `crates/execution/examples/wave_coexist_probe.rs` — the R2 prerequisite
  (the recon's #1 unknown). A persistent kernel + concurrent engine scans on one shared context COEXIST cleanly (no
  deadlock/starvation/zombie, 188 SMs), but the SM-reservation cost is steeply non-linear (1 SM ~2%, 8 SMs ~60%, 32 SMs
  ~87% of concurrent scan throughput; busy-spin ≈ gentle ⇒ co-residency cost, not poll traffic). **⇒ the wave kernel is a
  ~1-SM sidecar OR replaces the per-batch path; never a fat co-resident** (DECISIONS ADR-008 "R2 SM-coexistence gate").
- **`all_done` ordering audit ✅ DONE (2026-06-27):** two independent adversarial auditors (split cumulativity GAP vs
  SOUND, converged) → **the sound completion gate is the host ACQUIRING the `completed` counter (DtoH `==requests` before
  reading slots = the proven 1b pattern); `all_done` is only a wake hint, never the correctness gate** (DECISIONS
  ADR-008 "R2 `all_done` ordering audit"). Carry this rule into (iv).
- **R2.1 wave read engine ✅ DONE (2026-06-27)** = `crates/execution/src/wave.rs` (new child module) — the proven 1d
  data-plane lifted in-crate as `WaveReadEngine`, running the persistent kernel on the engine's SHARED primary context.
  Multi-wave `submit` over a circular lock-free ring; completion GATED on the DtoH `completed` counter-acquire (audit
  rule), `all_done` = wake hint; doorbell + `%globaltimer` backstop + clean Drop. Kernel = the audited probe kernel with
  one change — `atom.cas` claim (bounded, no overshoot) vs `atom.add` — so cumulative `head` works across waves (the
  audited result/completion ordering path is unchanged). GPU test vs a CPU oracle (2 waves, clean exit) + ASCII guard
  pass; NOT yet wired into any query path.
- **R2.2a wave multi-projection ✅ DONE (2026-06-27):** `WaveReadEngine::submit` now gathers up to 4 int4 columns
  (R1's unrolled gather) over the ring and returns `CudaI32BatchProjectionRow`s byte-identical to the R1 index probe
  (GPU oracle test green); `atom.cas` bounded claim makes cumulative multi-wave work.
- **R2.2 freeze ROOT CAUSE PINNED ✅ (2026-06-27, `execution/examples/wave_freeze_probe`, DECISIONS ADR-008):** the
  freezer is **`cuMemAlloc`/`cuMemFree` (device-synchronizing), NOT the stream type.** The probe shows every interleave
  op keeps the persistent kernel ALIVE in µs except `cuMemAlloc+cuMemFree`, which blocks ~the backstop and kills it
  (it device-syncs, waiting for the never-ending kernel). Consequence: a co-resident wave kernel dies the instant the
  engine does a synchronizing alloc. The device-buffer **pool amortizes** alloc (steady-state reuses; syncs only on cold
  growth / overflow free), and the wave path is alloc-free → **R2.2 path: pre-warm the pool + suppress pool shrink while
  a wave kernel is resident** (pragmatic), or move engine device alloc to `cuMemAllocAsync` (robust). "Replace per-batch"
  alone is insufficient (other engine activity still allocs).
- **R2.2 PROPER PORT — verdict FLIPS: the wave WINS at small (OLTP) batches (2026-06-27, `5e6b2302`+`bcc12af5`,
  DECISIONS ADR-008 "R2.2 PROPER PORT").** The first "wave loses 118x, park" was a naive per-needle port measured in the
  wrong regime vs the wrong baseline (retracted). The proper port = optimized drain (clamped-batched CAS claim + amortized
  membar, 520k->2.27M/s) + async `submit_async`/`harvest` + host-mapped `completed` mirror on its own cacheline
  (fixed a false-sharing latency pathology). Single-flight, 1M rows, wave vs lpb: **batch1 124k/39k=3.20x, batch8
  448k/309k=1.45x, batch32 1.21M/1.19M=1.02x**, batch256 0.30x, batch65536 0.10x. The persistent kernel WINS at small
  batches (no per-batch launch: 8us/submit vs lpb ~25us) — exactly the OLTP point-lookup regime; lpb wins only at large
  batches (GPU-bound 23M vs the wave's ~2.27M drain ceiling, CAS-contention-limited). At batch 8-32 the wave is already
  3-8x the batcher's 156k. (Single-flight improved further with the device result ring + per-slot gate below.)
- **R2.2 device result ring + P2 per-slot gate / depth-K pipelining ✅ DONE (2026-06-28) — CONCURRENT PREMISE VALIDATED.**
  Device result ring + separate-stream DtoH harvest lifted single-flight drain to ~31.8M (exceeds lpb across the sweep);
  the follow-up-review correctness gates landed (C1 occupancy clamp, C2 u64 counters, C3 status-0 assert); **P1
  needles-to-device TRIED + REJECTED** (needle read is 4B/coalesced/L2-cacheable, not the cap — ~31.8M is the realistic
  in-crate ceiling for the row-materializing workload; 45M was a simpler single-u64 probe). **P2 per-slot status gate
  (sm_70 `st.release.sys`) replaces the single-flight cumulative gate → depth-K pipelining (harvest any order):** the
  offered-rate benchmark shows pipelining 1.15–2.0× over single-flight, **2.6–4.6× lpb / 9–32× the 156k batcher in the
  concurrent regime** — the regime the wave exists for, now directly benchmarked. Independent re-review concurs (qualitative
  result robust; reviews `docs/reviews/r2.2-wave-port-followup-review.md` + the P2 follow-up).
- **THROUGHPUT HEADROOM (≈5–10×) — realize IN R2.2b, do NOT chase in an isolated probe.** The offered rate **saturates at
  depth-4** because a SINGLE host thread's submit+harvest loop is the cap (3–6M mid/large batch), far below the GPU's ~30M
  single-flight drain ceiling. The lever is **multi-producer host submission** (concurrent ring with atomic head
  reservation, many producer threads) — which real OLTP connections provide for free. So realize it as part of the wiring
  and measure end-to-end, not as another synthetic multi-thread probe.
- **R2.2b = wire `WaveReadEngine` into the engine read path** (behind the existing default-OFF flag, with R1's lpb index
  probe as the fallback) and run the END-TO-END offered-rate A/B vs R1, real connection concurrency = the multi-producer
  load — the ship/no-ship decision. **Wiring BLOCKERS (must land in R2.2b):** (1) enforce the in-flight bound
  (un-harvested needles ≤ ring_capacity, else circular slots clobber → wrong rows); (2) a host-side harvest DEADLINE
  (a stalled wave must not spin forever); (3) crash/`SIGKILL`-safe shutdown (a killed process leaves the persistent
  kernel zombied until the 30s backstop, perturbing other GPU tenants — the `--gpu-reset`-denied risk in practice).
  **NEXT = R2.2b (above), or R3 writes (independent).**
- Deterministic spine + MV dependency-graph concurrency control (BOHM/PWV); host sequencing materializes
  non-deterministic inputs; the order is the replication log.
- GPU index + point-access path; resident **layout decided by measurement** (PAX vs columnar).
- GPU-side WAL-record generation + tighter group-commit batching (durability is already crash-safe — STATUS).
  **Group-commit prerequisite (latent bug):** before enabling multi-entry apply, fix the 4 DDL apply helpers
  (`database_exists`, `tablespace_exists`, `relational_view_depends_on_inner`/`has_dependents` in
  `engine_ddl_objects.rs`) that read the *published* catalog while the apply loop mutates the *working* copy (safe
  today only because `to_apply ≤ 1`); add a multi-entry-batch regression test.

## 4. Correctness gates (build alongside the relevant work)
- **Durability + recovery:** WAL-flush-failure injection; rejected commits never visible; restart/recover;
  `commit ≥ applied ≥ visible` monotonicity.
- **Replication consistency:** follower write-rejection, leader promotion/demotion, catch-up, snapshot install; no
  committed entry lost, no divergence after convergence.
- **Jepsen-style fault campaign (v1 gate):** kill / partition / reorder / flush-fail; linearizability; minimized
  reproducer discipline (seed/topology/fault-schedule/WAL+watermark).
- **Oracle hygiene:** production has no CPU oracle/fallback. The legacy oracle is now `cfg(test)` only; replacing its
  remaining semantic fixtures with GPU-native/closed-form oracles is the ADR-007 test-infrastructure cleanup.

### 4b. CPU relational-engine retirement (ADR-006)
- **Tier 3a/3b production read deletion ✅ DONE (2026-07-12).** Non-test builds do not compile
  `finalize_relational_select`, the CPU-pinned SELECT implementation, `CpuMvccExecutionBackend`, or the backend-chain
  CPU fallback. Resident, streaming, transient catalog/materialized-view, and CUDA-native MVCC routes execute on the
  GPU; a decline/device fault returns a GPU-required client error. A non-test integration gate forces nonresidency
  and proves no host result can escape. `pg_catalog`/`information_schema` single-relation reads now upload a transient
  relation and use the same GPU executor as catalog joins.
- **Tier 3c residency-shadow/result exceptions ✅ DONE (2026-07-12).** Published residency entries no longer retain
  decoded host rows, commit append maintenance no longer maintains a host row mirror, and the obsolete host
  snapshot-probe/finalizer fixtures are deleted. Bounded SQL-function literal results now execute as one-row
  transient GPU relations rather than returning a CPU-labelled relational result.
- **Parity-oracle source remains test-only.** `new_local_cpu_oracle`, `finalize_relational_select`,
  `CpuMvccExecutionBackend`, and `FirstCudaSliceParityBackend` exist only under `cfg(test)` to keep deterministic
  SQL/MVCC semantic fixtures hardware-independent. ADR-007's stricter “no CPU oracle even in tests” cleanup remains
  a test-infrastructure task, not a production data-path blocker.
- **Tier 2 — gated on R3 (GPU-native writes + deterministic CC):** delete the host write/commit/MVCC-store/CC
  (`engine_write_apply.rs`, `engine_dml_*`, `engine_commit.rs` apply paths, `storage` mutate/visibility,
  `write_path.rs` ledger). R3 not started. Effort **L** (largest tier).
- **Remaining host-debt boundary:** chunk reverse-gather/deauthorization and scan-build remain for DDL/recovery/import
  and failed post-commit repair, never as a steady-state SELECT execution tier. Deleting them requires device-native
  DDL validation/recovery repair with an RPO-preserving replacement; do not replace them with a lossy fail-loud arm.

## 5. Product backlog (verify status before starting)
Grouped, terse. Detail lives in ARCHITECTURE.
- **Unification:** consolidate the **three pgwire servers → one**; invert the `engine → protocol` dependency; flip
  the store so the engine is the single source of truth.
- **Engine core:** real MVCC (SI→**SSI**, write-write conflict detection); GPU write path (real device work on
  commit); de-monolith oversized source files (tests-first extraction).
- **GPU-resident catalog + function engine** (ARCHITECTURE §14).
- **Type-system breadth (Phase 8 review):** per-type **text AND binary wire codecs + OID/typmod** with no silent
  PG-compat gaps; typed param binding (binary + NULL); real queryable `pg_catalog`/`information_schema`; bignum
  NUMERIC >38-digit handling; SUM/AVG accumulator-overflow checked-arithmetic fix. **Exit gate: every type
  graduates to a GPU-resident route.**
- **OLTP route classes:** tenant/security-filtered page route; bounded two-table join route (package the join
  engine + fanout bound); computed-detail route with resident summaries + invalidation.
- **Charter debt:** retire remaining test-oracle-only host operators and the DDL/recovery reverse-gather repair after
  GPU-native replacements exist; adaptive sort dispatch and device-native DDL validation remain optimization/breadth work.
- **Durability / HA:** live streaming replication (network AppendEntries/RequestVote, heartbeats, leases/fencing,
  auto-failover); synchronous commit on quorum; crash-safety integration tests + object-store archival.
- **Scale:** connection scale toward 100k–1M (session admission); working-set > VRAM (STRATA spill / S-E);
  pinned-host D2H evaluation.
- **Hardening:** packaging (deb/rpm, Dockerfile, k8s/systemd); supply-chain (SBOM, cargo-audit/deny/geiger, GPU CI
  fatbins); `// SAFETY:` on every unsafe + panic reduction; observability (Prometheus/OTLP/tracing + audit log);
  mTLS + SCRAM channel binding + credential store.
- **Hot-path efficiency audit (Phase 7):** a deliberate **asymptotic** sweep of hot-path data structures —
  correctness review does NOT catch asymptotics (motivated by the live O(n²)-write `with_table_mut` deep-clone that
  passed two correctness audits yet nearly sank a milestone). A standing methodology gate, run under the Phase-5 harness.
- **Productization acceptance gate:** driver smokes (`tokio-postgres`, `sqlx`, `asyncpg`, `node-postgres`,
  `psycopg`, `pgx`, JDBC/R2DBC); `pg_dump`/`pg_restore`, `pg_dumpall --globals-only`; release-candidate preflight.

## 6. Parked / verify-before-starting
- **Two charter-debt correctness items** (STATUS "known gaps"): expression-overflow PG-divergence
  (gather-then-evaluate, `engine_expr.rs:~5757`); routing-gate case-sensitivity (`resident_route.rs:388`).
- **Modernization #21:** raise `.ptx` arch targets toward an sm_90 floor; numeric/uuid MIN/MAX two-pass →
  `atom.cas.b128`. *Verify against the tree — may be partly done.*
- **Perf residual:** async the fused `mixed_int_text` route's ~11 sync round-trips → ~2–3. Gated on the Phase-5
  open-loop harness.
- **Single-GPU multi-partition read-side decision:** the 8 partitioned probes are GPU-native; retiring them needs
  the general executor to iterate multi-partition resident tables, OR an explicit decision to keep them (the
  campaign's S10a was BLOCKED here). Decide alongside S-C/S-D.

## 7. Non-goals (until post-v1)
Cluster-reconfiguration automation; multi-region failover automation; broad extension surface; broad async-driver /
binary / COPY-streaming parity beyond the acceptance gate.
