# ARCHIVED — HANDOVER before task-ledger unification (2026-07-12)

> Historical resume record. Its `NEXT`, `OPEN`, and deferred language is not actionable. Consult
> `docs/PLAN.md` and the current `docs/HANDOVER.md`.

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Where we are,
> the open decision, and the rules. The **why** is in DECISIONS.md; the **how** in ARCHITECTURE.md; the
> **mandate** in CHARTER.md; the **plan** in PLAN.md. The E2.5c campaign detail + gate ledger is in
> HANDOVER_REMAINING_WORK.md; the WAL/conveyor research record is in WRITE_CONVEYOR.md.

## Current baton — STRATA plan closed in this worktree

The requested continuation is complete through multi-GPU partial scheduling, S-F, and production host-read
deletion. Highlights:

- Streaming relational breadth is device-native across scalar/projection/grouped/distinct/ordered, N-way
  INNER/OUTER JOIN, and rank/window paths, including composite/mixed-width keys, text partitions, aliases,
  ambiguity, NULL-pad/visibility, global OFFSET/LIMIT, and allocator-backed query budgets.
- Chunk-authoritative CHECK and non-self-FK eligibility, exact/Bloom keyed skipping, spill/LRU, async lookahead,
  compaction, and recovery artifacts are closed. Self-FK remains deliberately excluded.
- Scalar/projection/grouped chunks round-robin over physical, healthy GPUs with the full query budget; completed
  secondary-GPU chunks have a non-vacuity counter. This box has one GPU, so the real two-device gate self-skips.
- S-F is flipped: `auto_admit_on_commit` defaults ON. Recovery disables admission/elision during record replay and
  bulk-admits once afterward. Production's implicit budget is 80% of physical VRAM; accounting uses actual payload,
  side-region, and device-index allocations. Replacement resources allocate before two-phase eviction; fit/allocation
  failure preserves the old resident set. One allocation transaction caps admission, rollover, version regions, and
  lazy indexes; deterministic eviction includes shard tables.
- Production relational/MVCC host fallback is deleted from non-test builds. CPU-pinned SELECT,
  `finalize_relational_select`, `CpuMvccExecutionBackend`, and the fallback chain are `cfg(test)` only. Catalog and
  materialized-view rows execute through transient GPU relations; production decline/fault policy is fail-loud.
- The legacy residency host-row shadow is deleted: admission rows are discarded after upload, append maintenance no
  longer mirrors them, and the host snapshot-probe API/fixtures are gone. Bounded SQL-function literal results now
  execute through a one-row transient GPU relation.

Final gates: engine ordinary **505 passed / 0 failed / 485 ignored**; pgwire ordinary 3/0 plus the ignored
GPU-route golden 1/0; production catalog/function transient GPU gates pass; the full default-on mixed gate re-passes
at **116.2k reads/s, p50 246us, p99 501us, p99.9 671us**, zero host gathers, zero fallback groups, 160/160
host-install elisions. The read-kernel roofline remains at baseline ratios (`count_i32_compare` 0.91x in-L2;
grouped kernel 1,678 M elem/s). The canonical report card also passes: the 48M-row OUT-OF-L2 batched route reaches
**252.4M lookups/s at b65536, p50 131us**, and the indexed single-flight route is 3.23x the scan. Final STRATA
re-audit: **MERGE-SAFE, 0 Critical /
0 High**; it explicitly covered normal admission, rollover/sidecars/indexes, effective multi-GPU budgets, and all
three public benchmark installers.

What remains outside this completed STRATA/read slice: (1) a ≥2-GPU hardware run; (2) ADR-007 cleanup of the
test-only CPU parity oracle; (3) host write/commit/store deletion, gated on R3; (4) reverse-gather/deauthorization and
scan-build retirement after device-native DDL/recovery repair exists. These repair/import paths are not production
SELECT execution and must not be replaced by an RPO-losing post-commit error.

The older campaign narrative below is historical context; PLAN/STATUS are authoritative for current completion.

**Updated:** 2026-07-12. **Base:** `main` @ `d49f7a6d`; current worktree = **P5-4 + P5-LATER CHARTER CLOSURE + PRODUCTION
MIXED GPU-ROUTE NON-VACUITY GATE COMPLETE, UNCOMMITTED** (P5 third independent audit MERGE-SAFE: 0 Critical /
0 High / 0 Medium; mixed-gate final independent audit MERGE-SAFE: 0 Critical / 0 High / 0 Medium / 0 Low;
full serial GPU sweep 974/974). STRATA STREAMING EXECUTOR — S-E.1..S-E.4 SHIPPED: **the
FOLDABLE OPERATOR CLASSES of ADR-012 are ALL STREAMING** — scalar reductions, filter/project+window, GROUP
BY/DISTINCT, ORDER BY/top-N; user chose this track at the predicate-edges boundary). **>>> ACTIVE TRACK: STRATA
STREAMING EXECUTOR (ADR-012 / PLAN §2 S-E) <<<** — over-VRAM reads run ON THE DEVICE by folding over bounded chunks
(admit chunk → reduce on device → combine partial → evict → next), never all shards resident.
**S-E.6a DONE (`0a9b2fae`) — THE COLD TIER + THE S-E.5 RETURN: streaming replays DEVICE-FORMAT chunk bytes, 79×/33×.**
A streamed table's chunk payloads cache in host RAM on the first fold build and REPLAY byte-for-byte thereafter —
the ~68% host decode wall is GONE on hits (COUNT 181ms→2.3ms, SUM 184ms→5.5ms, 100k rows/25 chunks). The S-E.5
overlap pipeline MERGED BACK as the path in the same commit (its economics flipped exactly as predicted). MVCC
validity (audit HIGH adopted): generation-Arc ptr-equality alone can't carry SI (commits publish the generation
BEFORE bumping committed_seq) — installs run under the COMMIT LOCK proving `committed_seq == build_copin_s` with
the generation unchanged (no stamp above the build boundary), hits require `reader_copin_s >= build_copin_s`
(boundary-invariance). Cap policy + stale-entry eviction + docs (audit M/M/L/L) adopted. Cache = INTERIM
double-residency beside the tuple store; both retire with ADR-006. Gates: lib 501/501, streaming 13/13 (incl. the
write-invalidation gate, generation-sabotage-verified), sweep 432/434 (same 2 pre-existing), clippy clean.
**S-E.6b DONE (`078f59f6`): cold-tier NVMe SPILL** — captures >256MiB stream to an UNLINKED temp file during the scan
(create+unlink, OS-reclaimed, crash-safe; TMPDIR-honoring; positional read_exact_at replay into the async pinned
upload; IO errors poison-or-defer, never wrong; 128GiB disk-class cap beside the 4GiB RAM cap). Over-RAM tables —
previously refused installs — now cache. Audit MERGE-SAFE zero C/H/M (offset bookkeeping cursor-exact; async buffer
lifetime safe); both LOWs adopted (replay-failure eviction; nanos in spill names). **⛔ CHARTER-DRIFT RULING (user, BINDING 2026-07-11, memory `charter-drift-execution-discipline`):** past agents
built host engines via PRECEDENT-CHAINING + hallucinated charter glosses. ONLY charter text or a USER ruling
justifies host-side work; every interim host piece needs a ledger row with a NAMED deletion trigger; the deletion
ships in the SAME MERGE as its device replacement; audits judge drift against CHARTER.md TEXT only; host-debt
balance sheet at track boundaries. REGISTERED DEBT (deletion trigger = the S-E.6c arc, user-ruled sequencing):
the S-E.1 host scalar partial-combine, the S-E.2 LIMIT/OFFSET drain/truncate windowing, the S-E.3 renorm casts,
and the ~2k-LOC cold tier + scan-build. **6c-0 DONE (`34a00c8f`): the host scalar combine + windowing are DELETED** — one device aggregate pass folds
scalar partials (StreamAccum/compare_sql_values gone); the cross-chunk OFFSET/LIMIT window is one device pass;
the streaming module's host relational computation is ZERO outside two registered items (the grouped per-round
narrow — see the hazard below — and the cold tier/scan-build, deletion trigger 6c-1..3). Audit MERGE-SAFE zero
C/H with the standing charter-drift section; three LOWs adopted.
**✅ HAZARD FIXED (`0c96d1d3`): the MASKED-PASS2 phantom-group kernel bug.** Root cause (device-probed, NOT the
suspected un-memset accumulators — those were already filled): grouped numeric aggregation is TWO-PASS; pass 1
writes `row_slots` ONLY inside its mask-skippable MIN/MAX block; pass 2 (`numeric_minmax_lo`) launched
UNCONDITIONALLY and scattered slot_min/max through STALE POOLED row_slots — unbounded OOB writes corrupting
adjacent pool buffers (the compactor's out_count inflated -> a phantom group (Int4(0), Null) from never-scattered
stale bytes; order-dependent because fresh driver pages are zero). FIX = one line: pass2 gates on
`value_is_numeric && (agg_mask & 12) != 0`, matching its producer. REGRESSION GATE = the suite order
`cold_tier_spills` -> `distinct_over_budget` (deterministic under sabotage); a new shape test pins numeric
partials + SUM mask reaching the merge. **6c-0(c) RE-LANDED in the same commit** — the per-round host narrow
loop is DELETED (the last of the three glosses); the wrap/narrow are staging-encode/readback coercions.
**6c-1 DONE (`6687d5e0`): CHUNK-GRANULAR DELTA MAINTENANCE** — a write PATCHES the cold tier in O(delta): imbl
COW-chain diff (the cache's own generation pin FORCES the clone — refcount>=2 → make_mut can't keep the pointer →
diff complete BY CONSTRUCTION, audit-proven vs imbl source) → effective-range tiling → rebuild ONLY dirty chunks +
tail (INSERT = pure tail, ZERO rebuilds; 1-row DELETE = exactly ONE — both gated). Spill-aware rebuilds (audit F1),
tail-runt coalescing caps ping-pong fragmentation (F2, remaining O(chunks)/patch walk = ledgered), patch installs
excluded from the builds counter (F3), empty→insert covered (F6). The scan-build debt: O(table)/write →
O(delta)/write. **6c-2 (tombstone sidecars) DEFERRED as low-leverage post-6c-1** (deletes are already O(one-chunk);
sidecars pay off when cold bytes become PRIMARY — folded into the 6c-3+ arc).
**6c-3 DONE (`7fe0277e`): EAGER COMMIT-TIME MAINTENANCE + the F4 payload-only builder** — commits patch cold entries
in place (self-gating, best-effort, DELTA-BOUNDED at 4096 chains so bulk writes never stall the commit mutex;
oversized deltas defer to the lazy read-path patch = the correctness backstop); reads on maintained tables are
pure hits. Rebuilds no longer spend a throwaway DMA (payload-only builder). AUDIT LESSON (adopted into comments):
committed_seq is NOT frozen under the commit mutex (lanes publish lock-free) — the invariant is the
STRICT-EQUALITY install guard + generation ptr identity + per-read visibility; never weaken the generation check
on a frozen-seq assumption.
**FULL GPU SWEEP GREEN 438/438 (`e685a0b2`)** — the two long-failing gates (a1/a4c) root-caused: their HOST-STORE
ORACLE premise died when the plain-int4 shape became elision-eligible (device-authoritative commits leave the
store stale BY DESIGN); pinned with the sibling-gate pattern + a4c's obsolete "gather must decline NULLs"
modernized to positively gate the NULL-aware gather. Audited MERGE-SAFE.
**FULL WORKSPACE SUITE GREEN (2026-07-11):** `cargo test --workspace` exit 0 — all 25 test binaries ok
(196 host-side tests) on top of the 438/438 GPU sweep; the arc's storage-crate (changed_tuple_ids /
visible_versions_in_range) and execution-crate (async copy transport, masked-pass2 fix) surfaces verified.
NOTE: wider-type (int8-bearing) tables DO NOT elide under defaults — shard ADMISSION is on
(shard_int8_section_enabled=true, test-lever setter only) but elision ELIGIBILITY is int4-scoped (int8
sections decline in-place appends), so the "versioned wider-type elided reads de-elide" residue is
UNREACHABLE today; the real gap is wider-type elision eligibility (i64-section append kernels), a full
slice for a future session.
**P3 DONE: THE DML WHERE-LOCATE AS A STREAMING FOLD** — a DELETE/UPDATE on a NON-ADMITTED (over-budget)
table with a range-only WHERE previously fell to the PURE-HOST seq_scan + `select_filter_matches` loop (the
CPU relational engine's core; the device arm has no shards there, the value index no Eq bound). The locate
now runs ON-DEVICE: `try_streaming_dml_locate` stages the visible rows with a synthesized trailing
`__row_id` int8 column (S-E.3 synthesized-relation pattern; real columns keep their catalog indexes so the
DML predicate lowering binds unchanged), device-filters + gathers each bounded chunk, and maps survivors
back to `(row_id, row_key, row image)` — hooked in BOTH `prepare_delete`/`prepare_update` ladders after the
value-index decline (INSIDE the else-arm: the self-referencing-FK bypass still forces the host arm — its
index-arm FK validation is wrong for self-reference). NO host recheck (the read folds' precedent; the
executor path applies exact 3VL validity) — audit traced the sibling arms' rechecks to THEIR coarse
NULL-blind locate kernel, NOT the lowering. Any decline/failure falls to the host loop (never a wrong
answer); `Some(vec![])` = a valid 0-match resolve; counter `dml_streaming_resolve_hits`. Audit MERGE-SAFE
zero C/H; MEDIUM adopted as the TYPE-MATRIX DIFFERENTIAL gate (text/date/numeric/bigint-OR-bool/UPDATE over
a NULL-bearing table vs a host-arm twin — the recheck-free path is now differential-gated beyond int4);
LOWs ledgered (chunk-boundary match asserts; the pre-existing unbounded concurrent transient-residency
class). Gates: 5+1 GPU tests, 2 sabotages bite (identity off-by-one, predicate dropped), sweep 447/447,
lib 502/502, clippy Δ0. The host seq_scan loop REMAINS for the no-budget/unlowerable general case — it is
the interim store's operational path, deleted with the store at P4.
**⚠️ HAZARD FIXED (`c0ffd35a`): `imbl::OrdMap::diff` MISSES REAL CHANGES — a latent 6c-1-era WRONG ANSWER on
main.** P2's stamp counter caught it: three sequential single-row deletes, each patched against a freshly
pinned generation, and the THIRD delete's id vanished from `changed_tuple_ids` while the chains provably
differed (pure-CPU repro pinned in tests). A missed delta = a patched cold entry silently serving a deleted
row. FIX: the store's WRITE-SIDE CHANGE LOG (epoch + capped (epoch,id) ring; delta = the log slice — exact
O(delta) by construction; out-of-window = a full pointer-pruned key walk). **STRUCTURAL DIFFING IS BANNED
for correctness-bearing deltas.** GC pruning stays exempt (horizon/servability invariant, documented).
**P2 DONE: SV2 TOMBSTONE SIDECARS FOR COLD CHUNKS** — a pure DELETE now STAMPS its chunk's on-demand
`deleted_by` sidecar (dense i64/slot, 0x7F-live — the SV2 shard format; COW O(8B×rows), absent for
delete-free chunks) instead of the O(chunk) decode+rebuild; the replay uploads payload + 8-aligned sidecar
as ONE buffer and the executor's mask VM ANDs `deleted_by > read_txn_id` IN-KERNEL (the sanctioned
src=Some + vis=Some seam — first caller). Chunks carry `payload_copin_s` (the payload's OWN boundary,
preserved by stamps/reuse) — the slot is the id's RANK among payload-visible ids in the chunk range (a
metadata walk on the pinned generation; audit-proven boundary-invariant across patches). classify_pure_delete
tolerates EXACTLY one previously-live payload version gaining deleted_by; anything else (tail growth,
same-chain UPDATE appends, double deletes, vanished chains) keeps the rebuild arm. created_by NEVER
materialized (payload boundary = the D3 hwm); NO version columns on rows (SV1/SV2, settled). Sidecar-bearing
entries DECLINE the P1 v1 artifact (benign; artifact v2 = ledgered P2b). Audit MERGE-SAFE zero C/H (slot-rank
stability, classifier strictness, mask boundary, alignment all traced); MEDIUM adopted (the out-of-window
fallback walk now has its own storage-level gate); LOWs adopted (prune-site + lineage-contract docs).
Gates: 2 new GPU tests (value-sensitive SUM + bounded-window projection — a plain COUNT or an over-budget
ORDER-BY differential CANNOT catch a wrong-slot mask: the former is slot-blind, the latter honestly defers
to the CPU), 6c-1/6c-3 tests updated to stamp semantics, 3 sabotages bite (mask dropped, slot off-by-one,
capture-decline removed), sweep 450/450, workspace green, clippy Δ0.
**P2b DONE: COLD-CHECKPOINT ARTIFACT v2 (sidecar persistence)** — magic `GPUDBCOLDCKPT2`; per chunk the
artifact carries `payload_copin_s` + the optional deleted_by sidecar (v1 artifacts fail the magic = benign
skip, no migration); stamped entries now QUALIFY for the checkpoint and masked rows STAY MASKED across a
restart. Focused audit MERGE-SAFE; both LOWs adopted (restore adds sidecar bytes to the cap total — install
copies the builder total verbatim, no recompute; a POST-RESTORE delete gates `payload_copin_s` persistence —
a seam-defaulted boundary shifts the stamp's rank and masks the WRONG row, caught by the closed-form SUM,
sabotage-verified). Gates: round-trip + post-restore-stamp test, sweep 450/450.
**P4 DESIGN ADVERSARIALLY REVIEWED + REVISED (`84e76bd6`) — the AUTHORITATIVE program is PLAN.md §2
S-E.P4** (this HANDOVER sketch below is superseded where they differ). Review verdict NEEDS-REVISION, all
adopted; headline kills: the P3 locate + P2 stamp are STORE-DRIVEN (P4-2a builds chunk-native twins), the
reverse gather is a GREENFIELD host columnar decoder (new registered debt), the below-boundary reader needs
the per-chunk born gate + entry quiesce + a never-read-the-empty-store dispatch guard, plus the elision
mutual-exclusion, DDL-sweep, and WAL-truncation interlocks. **P4-1 SHIPPED: THE REVERSE GATHER** — the greenfield host columnar decoder (chunk device-format bytes ->
catalog-order rows: i32/i64/b128 sections via the capacity-derived offset helpers, bool + NULL-validity
bitmaps, text (n+1)-u64-offsets+blob, sidecar mask kernel-identical `deleted_by > rtx`, mis-sized sidecar =
loud error) + `reverse_gather_streamed_rows` (whole-entry, scan order). REGISTERED HOST DEBT (control-plane
de-auth only; deletion trigger = device-index-over-chunks); #[allow(dead_code)] until the P4-2b/P4-3
callers land. Focused audit MERGE-SAFE (field-for-field encoder symmetry confirmed; both LOWs adopted:
full NULL matrix in the gate, loud sidecar error). Gate: all-types round-trip differential + stamped-delete
exclusion; 2 sabotages bite (validity inverted, mask dropped). **P4-2a SHIPPED (`7a9a5b5b`): chunk-native locate + locate-driven stamp** — lower_resident_predicate over
staged chunks (sidecar vis composed; slots native, no __slot column) + coordinate-driven sidecar stamps at
the deleting boundary with the generation unchanged. TWO P4-2b OBLIGATIONS doc-contracted on the pair: the
coordinate token / single commit-lock critical section, and the store-divergence rebuild hazard (store must
be dropped/frozen for class tables first). **P4-2b-i SHIPPED: THE CHUNK-AUTHORITATIVE CLASS (the store deletion's pivot).** FREEZE-NOT-DROP closes
review-C3 without a reader tracker: class entry (commit hook, elision-arm ELSE = H1 mutual exclusion;
eligible = keyless + FK-free + budget + FRESH cold entry) FREEZES the store — the apply's Insert arm skips
the install (allocator advances), the commit hook appends the statement's rows as TAIL chunks
(payload_copin_s = the commit; install_streaming_cold_class — settledness from the HELD COMMIT LOCK +
serial-only class + frozen-generation ptr-check, because the general strict-equality proof cannot hold
pre-publish); below-boundary readers keep the frozen chains (exact MVCC). DE-AUTH (sticky exit) replays
tails into the store (SHARED-allocator tuple ids — the store-LOCAL next_tuple_id COLLIDED and replaced
live chains, found+fixed; born = chunk payload boundary) at: the CPU-pinned read seam, DML prepare, the
DDL sweep, multi-entry commits, the COPY path, append failure. AUDIT MERGE-BLOCKED→ALL SIX ADOPTED:
C1 the COPY-path de-auth RE-LOCKED the held commit mutex (explicit commit_lock_held param — the 6c-3
lesson again); H2 fold-failure evict DESTROYED the record-of-truth (evict is now a class no-op + the
de-auth None-entry arm HARD-ERRORS); M3 a None residency scope now de-auths ALL class tables
(conservative); M4 auto-admit skips class tables (a budget raise would publish a STALE resident snapshot
served BEFORE streaming dispatch); L5 lane-apply debug_assert; L6 tails build as DIRECT RAM chunks (the
builder retro-spill would poison on a spilled base; unbounded entry growth accepted-by-design + ledgered,
VACUUM compaction = P4-5). #[cfg(test)] CHUNK_CLASS_ENTRY_ENABLED_TEST keeps three store-driven gates on
their machinery. COVERAGE GAP (next slice): a live COPY-into-class-table test (the C1 scenario is fixed
structurally, untested end-to-end). **P4-2b-ii SHIPPED: CLASS DELETE/UPDATE VIA LOCATE+STAMP** — prepare resolves class DML FROM THE CHUNKS
(P4-2a locate with sidecar vis + P4-1 unmasked slot-aligned decode; matches = packed (chunk<<32|slot)
pseudo-ids + fabricated keys); the delta carries the ENTRY EPOCH (ColdTableChunks.entry_epoch, bumped at
every install — the P4-2a coordinate token as a u64); the apply skips the frozen store; the commit hook
stamps the coordinates iff the installed entry still carries the epoch (routed through the CLASS install —
the general strict-equality proof cannot hold pre-publish, the tail-append precedent) and UPDATE =
stamp-old + tail-append-new (U2 shape). DE-AUTH EXTENDED: sidecar stamps REPLAY into the store (base
chunks: slot→store-id via the rank enumeration at the payload boundary, stamps>freeze only; tail chunks:
insert-at-born + own-stamp tombstones) — the exited store is MVCC-WHOLE at every boundary (gated by the
DDL-sweep exit's closed-form SUM). Audit MERGE-SAFE zero C/H; the MEDIUM is a LATENT-UNREACHABLE
lost-delete on epoch drift (single-entry class DML runs prepare→apply→hook under ONE held commit mutex;
multi-entry de-auths up front) — doc-contracted at the fallback arm: THE OFF-LOCK-PREPARE FUTURE MUST
REPLACE IT with re-resolve+stamp under the lock, never a drop. LOWs noted: UPDATE stamp/append
non-atomicity on device error = the accepted de-auth-on-append-failure precedent (delete-shaped); pseudo-
key/SI-keyspace overlap latent-inert (serial-only). Unfiltered `DELETE FROM t` resolves 0 rows through the
prepare ladder ENGINE-WIDE (empty filter_groups match nothing — pre-existing, discovered here; the class
exit test uses the DDL sweep instead). **P4-3 SHIPPED: THE BORN GATE** — class hits require only `rtx >= FREEZE` (the entry boundary advances
per tail append — the old rule would MISS any reader pinned below the latest write = the C3 thrash cliff);
every replay surface (4 folds, the locate, the reverse gather) skips chunks `payload_copin_s > rtx`; with
the sidecar mask (`deleted_by > rtx`) the visibility algebra is EXACT per-reader MVCC (audit: "textbook" —
the UPDATE old/new transition atomic at D via matching strict compares; base-chunk union == visible-at-
freeze by the disjoint-range tiling; non-class arms inert). Audit MERGE-SAFE zero C/H/M; LOW-1 adopted
(the gather gained the same freeze floor — a below-freeze gather would silently drop freeze-rebuilt base
chunks); LOW-2 noted (the fold loops at old boundaries are single-threaded-untestable — the gather/locate
stand in; the gate predicate is textually identical across all six sites). Sweep 456/456.
**P4-4 RESOLVED AS A DESIGN NOTE (freeze-not-drop made it moot):** recovery replays the WAL into the store
normally; nothing is ever dropped (the freeze defers reclamation to P4-5's fenced VACUUM), so there is no
"drop just-replayed rows" step — the P1 artifact warm-starts the cache at the seam and the class RE-ENTERS
at its next eligible commit. Composition verified by the existing P1 + class-entry gates.
**P4-5 RESOLVED: LEDGER CLOSURE — COMPACTION IS FENCE-GATED BY DESIGN.** Working the de-auth interaction:
compacting a chunk moves its payload boundary ABOVE the freeze, flipping it into the de-auth's TAIL arm
(its rows would RE-INSERT beside their still-frozen store chains = double rows), and re-borning tail rows
at the compaction boundary loses them for readers pinned between the real born and the compaction — BOTH
are the reader-fence problem the frozen-store RAM reclamation was already deferred behind. An unfenced
compaction would be a plausible-but-wrong MVCC violation; it is NOT shipped. REGISTERED (one row, one
trigger): **sidecar compaction + frozen-store RAM reclamation, prerequisite = a MIN-ACTIVE-READ-BOUNDARY
tracker** (a fence proving no reader below the compaction/reclamation boundary) — with it, both become a
single quiesced maintenance pass (compact survivors via the device projection gather, drop sidecars,
reclaim the frozen chains) and the class's steady-state RAM cost drops to chunks-only.

**═══ THE SEALED-SHARDS-PRIMARY ARC — HOST-DEBT BALANCE SHEET (track boundary, 2026-07-11) ═══**
**DELETED from the host (the ADR-006 wins):** for CHUNK-AUTHORITATIVE tables the host tuple store's WRITE
PATH is GONE — no tuple installs, no value-index writes, no version-chain appends (the counters:
chunk_class_skipped_installs); DML locate for class tables runs ON-DEVICE (the chunk-native locate); the
per-write O(chunk) DELETE maintenance became an O(8B/row) sidecar stamp (P2) and class DML became
coordinate stamps (P4-2b-ii); range-DML WHERE-locates on ALL non-admitted tables run on-device (P3).
**RELOCATED DOWNWARD:** cold-tier maintenance moved from read-time patches to commit-time tail
appends/stamps for the class; recovery warm-starts from the durable artifact (P1/P2b) instead of
first-read scans.
**REGISTERED DEBT (open rows, each with a named trigger):** (1) the REVERSE GATHER host columnar decoder
(control-plane de-auth/import only; trigger = the device-index-over-chunks route lifting the no-uniqueness
class gate); (2) the SCAN-BUILD (build_cold_chunks/first-build staging — still the bootstrap + de-auth
import path; honestly OPEN, off the steady hot path since 6c-1/6c-3); (3) ~~the FROZEN-STORE RAM~~ **ROW
CLOSED: RECLAIMED AT CLASS ENTRY — NO FENCE NEEDED** (the audited soundness: readers only pin
current-committed at bind, the CPU-pinned guard de-auths BEFORE binding, in-flight readers hold COW
generation Arcs, auto-admit excludes class tables — every leg adversarially traced; class entry now
DELETES the table's host chains + value-index entries, de-auth v2 rebuilds chunk-only with base rows
born@1, and the slot→store-id rank enumeration is DELETED); (4) ~~the class tail growth~~ **ROW CLOSED: COMPACTION SHIPPED** — a chunk past the
dead-fraction threshold (>=1/4, >=8 rows) rebuilds from its SURVIVORS via the device projection gather
(sidecar + dead slots physically deleted; NULL/numeric round-trip gated). TIMING IS LOAD-BEARING (audit
HIGH, found+fixed in-flight): compaction runs at the commit hook AFTER `publish_committed_seq` — a
PRE-publish install would let a concurrent boundary-minus-one bind load the new entry and born-skip the
compacted chunk's still-visible SURVIVORS (compaction is the FIRST operation that born-gates
previously-visible rows; tail appends and stamps were safe pre-publish). Cap accounting subtracts the
replaced payload+sidecar (the audit LOW). **NET:** host RELATIONAL COMPUTATION on the class's steady path = ZERO (writes: statement-row
encode = staging; reads: device folds; DML: device locate + sidecar bookkeeping); the host's remaining
roles are the charter's own (WAL, orchestration, staging, boundary coercions) plus the four registered
rows above.
**>>> NEXT ARC: P4 — DELETE THE HOST TUPLE STORE FOR STREAMED TABLES <<<** (the ADR-006 endgame for the
streaming class; fresh-session-sized, decompose into audited slices):
(P4a) DURABLE VALIDITY: the runtime generation-Arc validity dies with the store — the (artifact boundary,
WAL position) token from P1 becomes the entry's identity; spill files become checkpoint artifacts with a
real lifecycle (no longer unlinked). (P4b) WRITE PATH: INSERT = tail chunk build from the statement's OWN
rows (no store roundtrip); DELETE/UPDATE = P3 locate → P2 sidecar stamp + tail append; the WAL record is
the durability, the chunk patch is the materialization. (P4c) Eq-LOCATE off the value index (a host
structure that dies with the store): the P3 fold already lowers Eq; measure, then route. (P4d) CONSTRAINT
validation via device scans (the ADR-006 elision machinery's patterns — uniqueness needs the compound-
fingerprint/device-index route, NOT a host index). (P4e) RECOVERY FLIP: for streamed tables the artifact
becomes LOAD-BEARING (mandatory, not benign-skip) + WAL-suffix replay patches chunks directly (no store
rebuild); the artifact needs retention/rotation discipline. (P4f) THE DELETION SWEEP: table_rows() loses
its role for streamed tables; the registered cold-tier/scan-build debt DELETES in the same merges (chunks
become the primary representation, the scan-build becomes the bootstrap-only import path). Interlocks:
P4a BEFORE P4e; P4b/P4c/P4d before P4f; VACUUM must learn sidecar compaction (rewrite a heavily-stamped
chunk) somewhere before P4f.
**HOST-DEBT BALANCE SHEET (the charter-drift ruling's boundary accounting, 2026-07-11):**
DELETED this arc: the host scalar combine (~130 LOC incl. all value comparisons/arithmetic), the host
LIMIT/OFFSET windowing (~30 LOC), the per-round grouped narrow loop (~25 LOC), the throwaway upload per rebuilt
chunk (F4). RELOCATED DOWNWARD: the O(table)-per-write scan-build became O(delta)-per-write (6c-1) and moved off
the read path for maintained tables (6c-3, delta-bounded). REMAINING REGISTERED (deletion trigger =
sealed-shards-primary): the cold tier + scan-build machinery (~2.5k LOC — grew this arc but its HOST-RELATIONAL
content is zero: staging, orchestration cardinality, boundary coercions only); the interim double-residency
(tuple store + cold bytes). NET: host RELATIONAL computation in the streaming path = ZERO.
**>>> JUST COMPLETED (uncommitted, 2026-07-12): S-E.P5-4 — CHARTER CLOSURE <<<**
(PLAN §2 S-E.P5, design adversarially reviewed + revised). SHIPPED SINCE THE P4 CLOSURE: store-row
RECLAMATION `de4a4f34` (class entry DELETES the host chains + value index — the no-fence soundness
audited leg-by-leg; de-auth chunk-only, the rank enumeration deleted), fence-free COMPACTION `dc7851bb`
(post-publish BY DESIGN — a pre-publish install lets a boundary-1 bind born-skip survivors; dead slots +
sidecars physically deleted), P5-0 `8b1621d4` (the device slot recheck — the M1 prerequisite). P5-1 SHIPPED
`cddce252` (chunk_id content identity; the accounted+capped index cache; the fold-path blob_offsets HIGH
fixed + gated; dup-tolerant builds; the chunk_key_needle parity contract). P5-2 SHIPPED `629452c9`
(audit MERGE-SAFE zero C/H): THE KEYED-CLASS LIFT — unique-keyed tables ENTER the class (host rows
reclaimed); uniqueness validates ON-DEVICE at ALL FOUR choke points (prepare_insert, prepare_update,
both txn-preflight arms — the txn preflight is the txn path's ONLY unique guard, commit-time de-auth
runs after it) via per-chunk key-index probe + P5-0 slot recheck at the statement snapshot. **P5-4 deletes
the host recheck:** the fingerprint index ONLY selects candidate chunks; exact key equality, sidecar visibility,
residual predicates, and structural NULL semantics run through the device predicate VM. In-batch uniqueness runs
over a transient device relation with a hard 256-row bound (larger batches conservatively de-authorize; deletion
trigger = a device exact tuple-hash/group operator, the registered scalability row). Device-approved coordinates
feed `unique_coordinate_threshold_reached`: the GPU applies UPDATE self-exclusions and returns the only conflict
status bit (threshold 2 in-batch, threshold 1 existing-row), so the host performs no uniqueness count/filter/EXISTS.
NULL keys use the raw-payload placeholder fingerprint and exact device `IS NULL`, never host comparison or de-auth.
C1 = packed-coordinate self-exclusion under the epoch token
(resolve + probe share the entry.chunks enumerate space); H1/H2 = the estimated index set must fit the
cap and BUILDS AT ENTRY under the commit lock; the covered lane route refuses class tables (its apply
has no class arm — a routed write would be lost). Adopted audit findings: TEXT-key gate + replay
differential; de-auth purges the table's key-index cache entries; saturating H1 sum. Gate craft: a
fingerprint COLLISION cannot be birthday-found (the final fold round is a BIJECTION of the last word —
fp(a1,b1)==fp(a2,b2) reduces to h1(a1)^h1(a2)==b1^b2), so the gate CONSTRUCTS it by bucketing
first-word states on their top 12 bits; verified against the real fingerprint before use. Txn-test
lesson: a txn's statements all carry the BEGIN's seq (the txn id). P5-3 SHIPPED `40f6cf89`; P5-4 supersedes its
recheck boundary. The first P5-4 audit found a CRITICAL double-load: off-lock fallback pinned E1, reloaded E2 for
locate, then decoded E2 coordinates against E1. Fixed by carrying one entry Arc through locate→image→epoch; a
deterministic concurrent compaction/SI test is sabotage-verified (restoring the double-load commits when it must
serialize). P5-3 original record (audit
MERGE-SAFE zero C/H): BY-KEY DML LOCATE — an Eq-on-unique-key WHERE resolves class DELETE/UPDATE via
the probe + P5-0 slot rechecks with images from the RECHECKED slots (the reverse-gather decoder off the
point-DML hot path); P5-4 moves the whole-group collision/residual/visibility decision on-device and performs only
the final approved-row value readback on host;
range/OR/NULL/partial-key/any-failure fall to the fold — never a decline. Audit MEDIUM adopted: the
probe mirrors the fold's rtx<freeze DECLINE (sub-freeze boundary -> the caller's DE-AUTH valve, never a
silent 0-row DML — every class chunk is born at-or-above the freeze so the born gate would mask ALL
hits). Compound-key probe-vs-fold DML twins gate the folded needle (a divergence = a silently LOST
delete). S-E.P5 P5-0..4 is COMPLETE in this worktree. The third independent audit is MERGE-SAFE (0 C/H/M;
one stale-comment LOW adopted). Gates: ordinary engine suite 503 passed / 471 GPU ignored; full serial GPU
suite 974/974; 256/257 authorization boundary; compound partial-NULL live/replay; deterministic off-lock
compaction/SI entry pin; exclusion sabotage bites. REMAINING AFTER P5-4:
chunk-skipping (bloom/zone pruning) for
over-VRAM keyed tables — **CLOSED by P5-later in this worktree:** a capped all-chunk GPU Bloom set
replaces the over-cap exact set for candidate routing, with exact device recheck, off-lock spill priming,
strict reservation/rollback, and stale-ID retirement across compaction and publication races. The forced
all-positive, global-cap, spill, compaction, and E1/E2 barrier gates pass; HAZARD 3+2 clean. NEXT FRONTIER
(the deletion directive): (a) the ADR-012 architectural program
(JOINs/views/windows on the streaming executor); (b) widening class ELIGIBILITY (CHECK-bearing tables
enter today? FK tables still refuse — inbound-FK validation needs cross-table device probes); (c) the
cold tier/scan-build registered debt (trigger: sealed-shards-primary — now largely paid by P1..P4).
**2026-07-12 WIP — relational breadth is NOT complete / NOT merge-safe yet.** A first implementation
streams bounded two-relation INNER/LEFT/RIGHT/FULL joins, layered views, and rank-family windows, but the
mandatory independent audit found load-bearing blockers that must be closed before this item can move to
DONE: rank's ordered run still crosses the cold-storage boundary before its final device pass (the key
arrays themselves are now read directly from the resident payload, including NULL validity, with rank input+
output budget-gated); rank/view recursion now threads one `copin_s` but still needs deterministic DDL-race
gates; the rank ordered run still needs a no-intermediate-readback final merge; and the implemented SQL
breadth still declines multi-step OUTER streaming and non-rank window-function families. Rank-family windows
now support device-side WHERE, NULL keys, multiple PARTITION/ORDER keys, and inherited named WINDOW clauses.
N-way INNER joins
now schedule byte-bounded Cartesian chunk/block combinations through the existing left-deep GPU executor,
with worst-intermediate cardinality in the budget proof. Join inputs, bitmap, packed
keys, hash/pair scratch, and fixed-width gathers are now conservatively charged to one peak-device-byte
budget with telemetry; explicit NULL placement and OUTER-WHERE post-filtering are streaming. The host
OUTER-complement loop has already been replaced by one bounded
device-resident match bitmap at a time plus GPU compaction; RIGHT/FULL uses a bounded second pass over one
right chunk, so N:N coordinates never accumulate on the host. Parser modifiers/`ORDER BY USING` now reject explicitly, rank
column aliases are preserved, and streaming chunks hard-check the u32 row-index bound. Do not ship or mark
S-E relational breadth complete until the fused pipeline, one-boundary recursion, full budget proof,
remaining semantic cases, new differential/DDL-race gates, HAZARD, sabotage, and a clean re-audit land.
**2026-07-12 REPORT CARD COMPLETE:** Section A completed cleanly (IN-L2 roof 1482 GB/s; OUT-OF-L2 roof
1446 GB/s; count-compare ratios 0.88 / 1.00). Section B's first attempt exposed harness/route drift
(`sharded_int4_equality_multi_column_projection` vs the unified retained template); the harness now explicitly
pins unified residency and the rerun completed (dense-batched: 39k/s @ b1 p50 25us, 8.53M/s @ b256 p50 29us,
125.5M/s @ b65536 p50 393us). Section C's 48M-row attempts proved the old 700s default insufficient: the first
measured build was 752.6s, and a 1000s run timed out in the final b65536 scan cell. The default is now 1200s and
the validating rerun COMPLETED with `executed_target=Gpu(0)` (build 767.5s = 734.4 insert + 33.1 residency).
OUT-OF-L2 dense-batched results: 39.8k/s @ b1 p50 24us, 9.00M/s @ b256 p50 27us, 104.1M/s @ b16384 p50
125us, 123.2M/s @ b65536 p50 404us; at b65536 the scan was 79.7k/s p50 822.9ms and lpb-index 8.86M/s p50
6.48ms (111.2x scan). The canonical 2-layer x 2-cache-regime report card is COMPLETE. The existing facade mixed read/write concurrency gate was run: with four continuous writers,
the concurrent snapshot path reached 80.6k→753.6k read QPS from 1→64 readers, p99 82→340us, and beat the
writer-excluding reference by 71.4x→16.3x. **The production device-route non-vacuity gate is now COMPLETE:**
the real facade `PointLookupBatcher` runs 32 readers + four facade writers over a sharded int4-PK table carrying
an unprojected NULL. Dense on-device created/deleted visibility closed the append-window tail: three-run median
121.2k reads/s, p50 234us, p99 489us, p99.9 644us; writer-active p99.9 717us. All 160 writes overlap live readers
and host-elide; every sharded batch remains dense-GPU with zero host-gather or per-query fallback groups, and
resident device append waves fire. All simple-OLTP SLOs pass. Final independent
audit of the original gate: MERGE-SAFE 0 C/H/M/L. Gates: facade 38/38 + concurrency 14/14; engine serial GPU 974/974; mixed gate
3× sequential + 2 simultaneous; NULL-route and fallback-counter sabotage controls bite.
The pgwire golden gate is complete: a real `tokio-postgres` client compares explicitly non-resident host parity,
one-shard GPU, and later-shard multi-GPU-route reads, including an unreferenced NULL plus a projected wire NULL;
both GPU arms advance the dense route counter (3× sequential + 2 simultaneous hazard-clean).
**THE PRIOR ARC (SEALED-SHARDS-PRIMARY P1..P4, COMPLETE):** — design in memory
`strata-streaming-executor`; P2 SV2 tombstone sidecars; P3 DML resolve via streaming folds; P4 the store
deletion for streamed tables + the registered cold-tier debt payoff. The ADR-006 store deletion follows.
**P1 DONE: THE DURABLE COLD CHECKPOINT** — the cold tier survives restarts via the checkpoint model (bulk
paths are checkpoint-only per the architecture; the WAL stays row-op): `checkpoint_intent_lanes` now also
writes `<base>.cold-checkpoint.<cut>` (magic + BOUNDARY + per-table signature/chunk-target + per-chunk
row_count/tuple_range/descriptor/payload bytes + FNV-1a trailer; tmp→fsync→rename→dir-fsync atomic; stale
cuts swept), and lanes recovery restores it at the SEAM (after checkpoint-records replay, before the lane
suffix) so the WAL SUFFIX IS THE DELTA STREAM — each suffix record patches the restored entries forward
through the 6c-1 patcher via the 6c-3 commit hooks. Durable validity = STRICT equality artifact-boundary ==
seam `committed_seq()` + column-signature guard + recovery determinism (an installed restore is trusted like
a live build — the store is not re-consulted on hits); every guard failure is a benign skip (first read
rebuilds). **AUDIT HIGH ADOPTED — THE BOUNDARY CONVENTION:** the live watermark has TWO conventions
(serial/replay publishes the INCLUSIVE last index; the lane pump publishes the EXCLUSIVE frontier
`visible_global_cut = base_seq + cut`) — the artifact must always carry the SEAM value `base_seq + cut - 1`,
accepting EITHER live watermark as the quiescence proof; the pre-audit code stamped the live watermark
verbatim, leaving production (pump-published) artifacts one high = restore silently inert, masked by
replay-derived tests (regression: `gpu_cold_checkpoint_restores_under_lane_pump_frontier_watermark`,
sabotage-verified). Audit MEDIUM adopted (docs state the real trust model: determinism + boundary equality,
not store re-verification) + LOWs (stale-artifact sweep on non-quiesced skips; the patch arm rides the
REGISTERED 6c-1 staging debt — no new host relational compute). Mid-capture commit races abort the artifact
(post-qualification watermark re-check). Gates: 5 GPU tests + CPU descriptor round-trip, FOUR sabotages bite
(restore-skip, boundary-guard, checksum, boundary-convention), lib 502/502, clippy Δ0.
**S-E.5 EXECUTED + REVERTED TO `feature/streaming-copy-overlap` (2026-07-10, no-losing-paths policy — RESOLVED:
merged back via S-E.6a above):** the
copy/compute-overlap pipeline (async pinned-staged uploads on a private copy stream + the stage-N/compute-N-1
lookahead in all four folds) was built, tested 11/11, and A/B'd NEUTRAL — {185,179,181}ms vs {185,178,177}ms
(COUNT over 100k rows / 25 chunks). ROOT CAUSE (measured): the fold is HOST-STAGING-BOUND at every chunk size —
host scan+decode+build ≈68%, "upload" ≈9% (itself mostly host payload assembly; the raw PCIe copy of a chunk is
~10µs), device compute ≈1%. The ~1.3µs/row MVCC decode is the ADR-006 interim host store (charter forbids
optimizing it); until S-E.6 makes the cold tier RAW DEVICE-FORMAT SHARD BYTES (no per-row decode), there is
nothing for the copy engine to overlap with. The branch returns AS the path with S-E.6 (upload becomes dominant
then). The `retain_device_memory_copy_async`/`PendingCudaResidentDeviceCopy` primitive lives on that branch.
**S-E.4 DONE (`318ca05f`):** single-key ORDER BY streams — TOP-N = per-chunk DEVICE sort + window (a chunk's local
top-(m+n) is its only possible global-window contribution, invariant audit-proven through compaction) → concat →
device compaction re-sort/re-window → ONE final device sort + the real window over a synthesized "__stream_runs"
relation; UNBOUNDED = plain per-chunk filter/project + one final device sort, honest defer when survivors outgrow
the budget. The sort NEVER runs on the host. Audit MERGE-SAFE zero C/H/M; one LOW adopted (decline the ORDER-BY-
expression empty-string sentinel up front). Gates: lib 501/501, streaming 11/11, sweep 430/432 (same 2 pre-existing
a1/a4c, proven on origin/main), clippy clean, sabotage (compaction window + the enforcing sort budget gate).
**S-E.3 DONE (`8036cfb7`):** GROUP BY (COUNT/SUM/MIN/MAX) + single-col DISTINCT over an over-budget table stream via
the TWO-LEVEL fold — per-chunk device grouped partials → concat (control plane) → ONE final device merge over a
synthesized `__stream_partials` relation (COUNT folds as SUM(count), SUM as SUM(sum), MIN/MAX as the extreme; the
host never groups). Mid-scan COMPACTION re-merges an over-target accumulator (the persistent accumulator as periodic
device re-merge); over-budget true cardinality DEFERS honestly (pre-upload BUDGET GATE — audit MEDIUM adopted:
partials can be WIDER than source rows (4B key→12B partial), the merge must never bust the budget it honors;
peak≤budget regression-gated). Grouped AVG / COUNT(DISTINCT) decline (not associatively decomposable). Audit: no
C/H — "never returns a wrong answer"; NULL-key round-trip + Sum(int8) scale-parity + synthesized-relation isolation
traced correct. Gates: lib 501/501, streaming 9/9, sweep 428/430 (same 2 pre-existing), clippy clean, triple
sabotage (merge kind / compaction drop / budget gate).
**S-E.2 DONE (`d26d41a7`):** a plain `All`/`Columns` PROJECTION over an over-budget table streams — per chunk the
WHERE + column gather run on the device (the same transient-source fold), survivors CONCAT (the §13 projection
combine); LIMIT/OFFSET = cross-chunk windowing of the survivor stream (device gather bounded to skip+take per
chunk; the executor's own control-plane-windowing precedent) and a satisfied LIMIT STOPS THE SCAN EARLY (tail never
staged). Ordering deterministic (both paths iterate ascending TupleId) so exact-order CPU differentials are sound.
Entry generalized to `try_streaming_select` (StreamShape: Reduction | Projection). Opus audit MERGE-SAFE, zero
C/H/M findings (windowing hand-traced; per-chunk re-bind proven single-filter; classifier misroute-free; charter
defensible; LOW WHERE-arith overflow parity unreachable — the lowered predicate grammar is col-op-literal only).
Gates: lib 501/501, streaming 6/6, sweep 425/427 (same 2 pre-existing), clippy clean, sabotage-verified
(early-exit + offset drain).
**S-E.1 DONE (`570c7fa4`):** a `COUNT(*)/SUM/MIN/MAX` over a table larger than the
configured per-GPU residency budget now streams on the device instead of de-eliding to the CPU host engine — new
`engine_streaming_exec.rs` folds the MVCC-visible rows into byte-bounded chunks, each uploaded as a transient
`build_transient_relation_residency` source + reduced via the existing `execute_resident_expr_select_with_binding`
(filter + reduce on the GPU), partials combined host-side (COUNT=Σ, SUM int4→Int8 i128-checked / int8+numeric→
`Decimal128::checked_add`, MIN/MAX via `compare_sql_values` skipping NULL); peak residency = one chunk (gauge
`streaming_fold_peak_chunk_bytes` ≤ budget). Activation gates on a CONFIGURED budget so default behavior is
byte-identical (no test sets one); any executor error defers to the authoritative CPU path (streaming only ADDS
reach); a genuine overflow surfaces. Opus audit CLEAN on MVCC/combine/gate/charter/defer; HIGH (chunk sizing counted
NULL/empty-text as 0 bytes → whole-table upload for null-heavy tables) fixed via device-byte accounting +
regression-gated; LOW (per-chunk overflow now surfaces uniformly) adopted. Gates: engine lib 501/501, GPU streaming
4/4, clippy clean, FULL GPU sweep 422/424 (the 2 fails = `capacity_payload_tests::{a1_device_row_identity,
a4c_device_gather}`, CONFIRMED PRE-EXISTING on clean origin/main, unrelated). REMAINING slices: S-E.6b+ NVMe spill + shard-granular evict/prefetch (S-E.6a cold tier + the S-E.5 return SHIPPED
`0a9b2fae`). Memory `strata-streaming-executor`. AVG deferred (needs the (sum,count) pair —
scalar AND grouped); grouped COUNT(DISTINCT) deferred (not associatively decomposable).

**Prior base** `7432ebf6` (CPU-ENGINE RETIREMENT — THIRTY merged wins, **THE
PREDICATE-EDGES ARC IS COMPLETE**: every scalar type × operator × operand-shape (literal, col-vs-col, mixed-width
group) × nullability now resolves ON-DEVICE for DML + reads, alone or in AND/OR; the remaining CPU-engine-deletion
work is the ADR-012 ARCHITECTURAL program (STRATA streaming executor, JOIN grammar, views, window functions):
TEXT/UUID COL-VS-COL on-device (NEW per-row two-column text byte-compare PTX kernel, audit-verified line-for-line;
uuid composes the existing b128 columns kernel as `UuidCmpColumnsMask`; BOTH-validity 3VL sabotage-proven; **PTX
LESSON: comments must be PURE ASCII — em-dashes 218 the whole module on the Blackwell driver JIT while ptxas
passes**, memory `ptx-ascii-comments-jit`) `7432ebf6`;
BOOL INEQUALITIES on-device via constant-fold (PG `false < true`: `<`/`<=`/`>`/`>=` vs a bool literal fold to
equality masks or ConstMask verdicts, NO new kernel; the const-TRUE shapes carry the validity AND = the ONLY 3VL
net, sabotage-proven; literal-on-left flips the op; both `compile_bool_leaf` + the non-null peephole; DML builder
bool arm Eq→all comparisons; audit SOUND zero defects, 16/16 fold rows verified) `034c5429`;
MIXED-WIDTH predicate groups on-device (int8/ts scalar leaves beside int4/int2/text/bool/date/uuid compile into ONE
I32 program via width-safe `LoadColumnI64` arms — the SV3b conjunct contract; LOCAL `mixed_width_i32_elem` gates at
4 sites, NOT a `predicate_vm_elem_type` widening; kills the "mixed int8/text" de-elide = the CHECK-bypass decline
recipe; audit MEDIUM int8+int2 I64 mis-read fixed + int2 AND/OR newly served; audit LOW uuid+int8-col-vs-col I64
path restored; int8 arith in a mix still hard-errors; double-sabotaged) `3445c6ac`;
NON-i32 FK COLUMNS ELIDE (uuid/int8/text/timestamp/numeric/bool fk children — NEW `device_eq_scan_literal` one
canonical Eq arm per device-scannable type feeding the elided scan arm; the `i32_section_needle` early-decline in
`device_visible_row_with_value` is gone; the per-fk-column eligibility check DROPPED since the main column gate is
exactly the helper's set; non-i32-PK parents stay non-elided/host-probed; audit SOUND zero defects; 5-type loop test
+ bool addendum, sabotage-verified) `4c1a830a`;
FK CHILD TABLES ELIDE (**the LAST structural elision class** — outbound-FK gate lifted to non-self-referencing +
all-fk-columns-i32-section; the inbound child-reference check on a parent DELETE runs ON-DEVICE via a new Eq
scan-locate fallback in `device_visible_row_with_value` when the dup-intolerant hash-index probe declines on the
duplicate-heavy fk column — same W0-guarded locate as range-DML, boundary RAISED to `committed_seq` per the
materialize contract, DATE fk needles round-trip the canonical `format_date` string since a raw-days Int4Literal is
a hard error and the decline would rehydrate on EVERY parent delete; opus audit SOUND, both residuals adopted;
sabotage-verified both arms) `aa75c11c`;
FK-REFERENCED PARENTS ELIDE (inbound-FK eligibility lifted for i32-PK-referenced tables — the FK validators were
already elision-aware via `visible_row_with_value`'s device arm; child INSERTs' parent-exists + parent DELETEs'
surviving-provider probes run ON-DEVICE, decline→rehydrate; + the DELETE-arm mid-preflight re-pin (the CHECK audit's
deferred twin) + PG MATCH-SIMPLE NULL-fk 3VL in both validator families) `451bb054`;
CHECK-on-NULL PG 3VL fix (NULL SATISFIES a CHECK — both engine evaluators guarded; legacy gpu-db-server emulator
divergence noted-not-patched per the charter ruling) `337454c7`;
CHECK-CONSTRAINED TABLES ELIDE (the first constraint class lifted — CHECK is row-local; ADD CHECK's existing-row
scan is elision-safe-by-construction; wave/lanes still route CHECK to the full off-lock prepare; FK stays blocked
both directions) + the audit-HIGH preflight stale-scan bypass FIXED (a mid-preflight rehydrate left the UPDATE
else-scan on a stale handle → vacuous CHECK pass → a DURABLE violating WAL entry that wedges all later commits;
fix = re-pin the outer view + raise the boundary, sabotage-verified) `6fba4ecb`; DATE RANGES
on-device for DML + reads (a DATE VM leaf, 4-byte LoadColumn I32-ONLY by audited width discipline; bounds are
CANONICAL `format_date` TextLiterals in the DML builder AND the &Select bridge — `date = 5` stays a HARD ERROR, a
sweep-caught PG regression fixed via the uuid format/parse pattern; placeholder-spanning read pin stays-elided +
sabotage-proven) `b18a018e`; NULLABLE-NUMERIC coverage pin (already worked via the I128 validity-aware AND arm)
`58e0110d`; NULLABLE-TIMESTAMP
ranges + COMPOUND TIMESTAMP READS on-device (the nullable local gate gains an I64 case for {Int8,Timestamp}; a NEW
timestamp scalar VM leaf accepts Int8Literal (DML micros) AND TextLiteral (read bounds, parsed) via LoadColumnI64 +
CompareScalarI64 + validity — arm-steal audited byte-for-byte equivalent; ts-vs-ts col-vs-col in AND newly works via
CompareBuffers@I64, test-covered) `bedde09e`; NULLABLE-UUID
ranges + IN resolve ON-DEVICE (a LOCAL I32-mask gate in the nullable branch — after the simple helpers, zero
diversion audited; per-leaf validity-AND is the 3VL net, READ-pinned with a zero-anchored range whose NULL
placeholder matches both bounds, sabotage-proven) `de289200`; UUID RANGES +
uuid IN resolve ON-DEVICE via `ExprStep::UuidCmpMask` (the b128 memcmp kernel composed into the mask VM —
`u>=A AND u<=B`, `u IN (A,B)`, mixed uuid+int4/text; uuid leaf dispatched BEFORE text since a uuid literal is a
TextLiteral; `try_lower_uuid_predicate` AND/OR falls through Ok(None), blast radius audited clean) `9184fab9`; TEXT RANGES +
nullable-text inequalities resolve ON-DEVICE via `ExprStep::TextCmpMask` (the text-compare kernel composed into the
mask VM — `name>='b' AND name<'d'`, nullable 3VL validity-AND, mixed text+int4; Eq/Ne restructure audited
behavior-identical; ABI verified vs the proven launcher) `a7a74935`; TEXT INEQUALITIES
`<`/`>`/`<=`/`>=` resolve ON-DEVICE for BOTH DML + reads via a NEW hand-written PTX kernel
`gpu_db_resident_text_compare_scalar_to_mask` (lexicographic unsigned byte compare, shorter-sorts-first,
byte-identical to `str::cmp` == the recheck; opus-audited instruction-by-instruction; scalar_on_left + nullable
validity) `7e06610e`; UUID INEQUALITIES
`<`/`>`/`<=`/`>=` DELETE/UPDATE resolve ON-DEVICE (uuid is byte-comparable — device kernel==recheck==PG MSB-first
byte-wise; builder Eq→all-comparisons) + `col IN (...)` confirmed on-device for int4/text (OR-of-Eq via mask VM, free
from the equality wins) `2170dee1`; LIKE-PREFIX
DELETE/UPDATE resolves ON-DEVICE (`text_col LIKE 'p%'` lowers to `Column Like TextLiteral(escaped-p%)` → existing
device text-LIKE kernel `expr_text_like_scalar_filter`; recheck `starts_with`; parser guarantees device==recheck;
CHARTER-PURE) `556e3c0b`; NULLABLE-COLUMN
DELETE/UPDATE resolves ON-DEVICE (`materialize_resident_row_via_hit` reads per-column validity bitmaps → SqlValue::Null
instead of declining wholesale on any null-bearing shard; NULL predicate operands double-excluded by locate+recheck
3VL; unblocks the common nullable-table case; CHARTER-PURE) `396977e8`; MULTI-BOUND
TIMESTAMP range DELETE/UPDATE resolves ON-DEVICE (`ts>=X AND ts<=Y` lowers on the i64 buffer VM — timestamp is i64
micros in the int8 section; `resident_device_int_column_offset`+`try_lower_timestamp_predicate` AND/OR path, LOCAL
i64-section gate so nullable-ts 3VL reads stay on their peephole; CHARTER-PURE) `28fe8df8`; UUID + BOOL
EQUALITY DELETE/UPDATE resolve ON-DEVICE (builder arms → existing `try_lower_uuid_predicate` (b128 byte compare) /
`try_lower_bool_predicate` (bitmap→mask); + a bool materialize bitmap arm; CHARTER-PURE, no host store, no new
kernel; `=` only) `ba02918b`; TEXT-EQUALITY
DELETE/UPDATE resolves ON-DEVICE (`text_col='lit'` lowers to `TextLiteral` → existing device byte-exact text kernel
`try_lower_text_predicate` + device text materialize + recheck; CHARTER-PURE, no host store, no new kernel; `=` only,
text has no device ordering) `be6bd91f`; MULTI-STATEMENT
INSERT-batch elision (a group-commit batch keeps its insert-only elided tables ELIDED via one incremental device
append per table, `InsertPerRow` stamps; was: `to_apply.len()>1` de-elided the whole scope every batched write)
`db29b8a8`; NUMERIC range DML (single+multi-bound) `bd55680a`; TIMESTAMP range DML `2fb42ab3`; INT8 range DML
(new `Int8Literal`→`CompareScalarI64`, >i32 bounds on-device) `9e62b882`; NULL coverage (nullable columns stay
elided, alignment-free `NULL_BITMAP_GATHER_PTX` kernel) `bb2a2c03`; range/non-point DELETE/UPDATE resolve on-device
`86a3ff6f`; point zero-match DML stays elided `6f7cad76`; declined resident reads → general GPU executor `ebd04717`;
TEXT compound-key uniqueness `df136262`; foundation `fe9af98d`).

**⛔ CHARTER RULING (user, EMPHATIC 2026-07-09): do NOT touch / repair / make-faithful / invest in the HOST STORE.**
The de-elide → `rehydrate_elided_table` → CPU tuple store is INTERIM debt to DELETE, not to improve. The only
charter-pure way to keep DML elided is to make the **DEVICE resolve the WHERE** so it never de-elides (MOVE WORK TO
THE GPU). A DELETE/UPDATE multi-entry-batch elision attempt that made the host-store de-elide reconcile faithful
(per-row `created_by`, gather-boundary seam) was REVERTED as charter-violating; the branch `feature/multi-entry-du-elision`
holds the dead approach. When a de-elide is unavoidable (device can't serve a shape), let it de-elide as-is — never
polish a path slated for deletion. See CHARTER.md, memory `stay-gpu-native-charter`.

**>>> ACTIVE: CPU-ENGINE DELETION (ADR-006) — closing the de-elide/host-fallback triggers <<<** Recon mapped the
deletion target (`finalize_relational_select` + `rel_exec_helpers.rs` host operators + `CpuMvccExecutionBackend`)
and the trigger taxonomy. FIRST LEVER SHIPPED (`ebd04717`): when the SPECIALIZED resident-route classifier declines a
SELECT shape on a GPU-RESIDENT table, `execute_relational_select_instrumented` now routes it to the GENERAL GPU Expr
executor (`execute_resident_select_via_general`: DISTINCT→distinct bridge, else grouped bridge, src=None so
`with_binding` resolves the whole-table or TYPE-COMPLETE unified shard source) instead of de-eliding to the CPU pinned
path. Wider-type filtered projections / scalar aggregates / DISTINCT / GROUP BY / single-key ORDER BY / `SELECT *`
over text / OFFSET / empty-aggregate (→NULL) all now stay ON THE DEVICE + ELIDED (`general_read_fallback_hits` proves
it). The general executor ERRORS (never mis-answers) on a shape it can't express → falls to CPU (honest partial
coverage). Opus audit MERGE-SAFE (7 angles; visibility rests on the single-buffer↔shard exclusivity invariant).
FOUR MERGED WINS so far: declined resident READS → general GPU executor (`ebd04717`); POINT zero-match DML stays
elided (`6f7cad76` — return HANDLED on an empty applied set + gate elision-ENTER on `applied_changed_rows`); RANGE/
non-point DELETE/UPDATE resolve ON-DEVICE (`86a3ff6f` — `try_resolve_dml_via_predicate_scan` lowers the WHERE to a
ResidentExpr + `locate_resident_delete_slots_detailed` single-snapshot+W0 + materialize/visibility/recheck; kills
O(table) de-elide churn for all int4 range DML). Debugging lesson: a backtrace at `rehydrate_elided_table` is the
definitive de-elide root-cause tool. **MULTI-STATEMENT batch de-elide PARTIALLY CLOSED (`db29b8a8`):** a
group-commit batch (`commit_mutation_batch` → `apply_and_publish_committed_inner`, `to_apply.len()>1`) used to
de-elide EVERY touched elided table up front (the banked per-type elision wins silently reverted under the batcher —
the SQL-text write path; tests survived only because they issue one statement at a time). Now a batch's insert-only
elided tables STAY ELIDED via one incremental `try_append` per table (`InsertPerRow` birth stamps, decline →
rehydrate-with-delta; maintained tables excluded from invalidate+auto-admit). Opus audit SOUND. Root cause: PK/unique
tables take the immediate single-entry commit (never batch) — only CONSTRAINT-FREE int4 tables batch, and they elide.
REMAINING de-elide/host triggers: multi-entry batches with a DELETE/UPDATE (or non-DML) still de-elide (insert-only
this slice); WIDER-TYPE range DML (int8/numeric/timestamp now on-device; text range still declines); NULL-bearing
shard `gather_resident_table_rows_from_device` edge (mostly closed by `bb2a2c03`); CHECK + FK (both directions) now
ELIDE (`6fba4ecb`/`451bb054`/`aa75c11c` — the structural classes are CLOSED; remaining edges are marginal predicate
shapes: bool inequalities, text/uuid col-vs-col in AND, mixed int8+text/bool widths). ARCHITECTURAL
GATES (a program, per ADR-012, user chose "shrink achievable surface"): non-resident/over-VRAM tables need the STRATA
STREAMING EXECUTOR (the documented terminal gate); views/matviews; JOINs beyond the 2-table `=` chain; window
functions (absent from the grammar). See memory `type-coverage-14`, `scalability-ledger`.
**ACTIVE lane:** TIER-1 TYPE/OP COVERAGE —
the covered lane write TRIAD is COMPLETE (INSERT + DELETE + UPDATE, all WAL-first), updates are
SUSTAINABLE (F3/U4 version-aware device PK index — dup-tolerant, mixed bench 1.4M TPS / 3 rebuilds),
**R-ver (read version resolution) COMPLETE — PART 1 + PART 2 MERGED** (reads over versioned elided
tables no longer de-elide/refuse), so MIXED OLTP is fully GPU-native for the int4-PK shapes.
**TYPE COVERAGE #14 per-type arc:** int4/date/int2 → int8/timestamp → **NUMERIC/UUID (b128) MERGED
(`b16f791f`)** → **BOOL MERGED (`13d8d645`)**. Bool is BIT-PACKED (1 bit/row, the terminal columnar
rep) via TWO new PTX kernels — `set_bool_bitmap_range` (incremental atomicOr append into pre-zeroed
headroom; makes bool ELIDE, since elision needs a handled device append) + `gather_bool_bitmap_from_
shard` (ALIGNMENT-FREE cross-shard bit repack, because shards seal at arbitrary non-32-aligned row
counts so a byte-copy can't concatenate bitmaps). Rehydration gather now materializes bool (filtered/
ORDER-BY-key shapes fall to CPU-pinned + rehydrate instead of hard-erroring). Seven adversarial audits
across U2/F3/U4/R-ver/numeric/bool all MERGE-SAFE; every CRITICAL/HIGH fixed + sabotage-verified +
regression-gated. **REHYDRATION CLOSED (`819efbf7`):** `gather_resident_table_rows_from_device` (the
device→host rebuild for read shapes the on-device routes can't serve — filtered value-column
projections, ORDER-BY-on-bool-key) now materializes NUMERIC/UUID (b128 reassembly) + BIGINT (verified),
so those shapes rehydrate instead of hard-erroring. **TEXT MERGED (`bcc8eed0`) — TYPE COVERAGE #14
PER-TYPE ARC COMPLETE: every scalar type (int2/4/8, date, timestamp, numeric, uuid, bool, text) is now
device-authoritative across write + on-device read + rehydration.** Text is ROLLOVER-ONLY (variable-
length has no headroom → each commit seals a dense text shard); the crux was a cross-shard gather that
byte-concats blobs + a NEW PTX kernel (`gpu_db_resident_text_offset_rebase`) that rebases each shard's
offsets by its running blob_base (offsets are blob-relative, can't byte-concat). Text shard-admission is
PK-GATED so legacy non-PK text tables stay single-buffer (blast-radius containment); surfaced + fixed a
real DROP shard-leak. Eight adversarial audits (…/numeric/bool/rehydration/text) all MERGE-SAFE.
**COMPOUND KEYS (Track 3) — DEVICE-NATIVE UNIQUENESS MERGED (`fe9af98d`; foundation
`3e3f520d`):** a compound `PRIMARY KEY` /
`UNIQUE` over i32-SECTION columns (Int4/Date/Int2) now ELIDES and validates uniqueness ON THE DEVICE — the
six DDL rejections are lifted for that subset (wider-typed compound stays honestly rejected, all pre-WAL).
THE ARCHITECTURE (best DELIVERED perf, CHARTER-PURE): the ordered key-column values fold into a 32-bit
SURROGATE FINGERPRINT (`compound_key_fingerprint`) that rides the ENTIRE existing single-column i32 device
index (build/insert/write-locate/visible-locate/coalescer/geometric-rebuild), inheriting every banked
optimization. CHARTER: the index REBUILD folds ON THE DEVICE (one new PTX kernel
`gpu_db_compound_fold_fingerprints` / `submit_compound_fold_fingerprints`, BYTE-IDENTICAL to the host
`compound_key_fingerprint`) so the raw resident key columns are NEVER read back to the host to be hashed —
the host reads only the derived fingerprint column, at charter parity with the single-column build; the
needle + incremental-append folds run on host-HELD values (the INSERT's own literals / the wave's rows =
control-plane, same posture as the single-column needle). EXACTNESS is free + device-native: the
write-locate probe is already non-authoritative (count>0 → recheck), so for a compound key the recheck
materializes the candidate row and compares the FULL TUPLE (`visible_row_with_tuple`), so a fingerprint
collision can never false-23505 or mis-locate. Compound tables take the CLASSIC covered path (the fused
INTENT lane rejects them — still device-native). Gates: engine 495/0, FULL GPU sweep 385/0, clippy
baseline. TWO adversarial opus audits (broad + a focused kernel audit): ONE real fix adopted (HIGH —
restored the lanes-mode LIVE-shard offset recompute in `ensure_shard_pk_device_index`) + ONE FALSE POSITIVE
disproved+documented (ordinal-based cache key is safe because every index-shape DDL triggers a GLOBAL
residency invalidation that purges the PK-index cache); the device fold kernel CLEARED (byte-identical to
the host fold, sabotage-verified via a device-fold-consistency dup test).
64-bit fingerprint. **This was the last charter-advancing step before CPU-engine deletion (ADR-006).**
**OPERATIONAL CASES (Stage 1) DEVICE-NATIVE:** a DELETE / UPDATE `WHERE a=? AND b=?` by a compound key
now resolves ON THE DEVICE — `resolve_dml_matches_via_device` -> `dml_device_probe_key` folds the
surrogate fingerprint from the key columns' Eq predicates and probes the compound index; the full
`filter_groups` recheck restores tuple exactness (a collision can never delete/update the wrong row), so
the table STAYS ELIDED instead of de-eliding (compound-keyed tables can't reach the visibility-blind
covered lane — it needs a covered-INSERT route, which rejects compound — so the SQL resolve is the only
path and it always rechecks). Point-reads already ran as device AND-scans (no de-elide). Focused audit
CLEARED (exactness-under-collision airtight). **WIDER KEY TYPES (Stage 2a) — i64/MIXED DONE for
INSERT-uniqueness + reads:** compound PK/UNIQUE over Int8/Timestamp columns (and MIXED int4+int8) now
elides + validates uniqueness ON THE DEVICE — each key column folds its i32-WORD decomposition (i32 -> 1
word; i64 -> 2 words [low32,high32] LE) into the fingerprint; the device fold kernel `COMPOUND_FOLD_PTX`
takes per-column WIDTHS. Eligibility is arity-aware: SINGLE-column keys stay i32-section (raw i32 key),
COMPOUND keys accept any foldable type (fingerprint). Focused audit CLEARED the host==device fold
byte-for-byte. b128/text keys stay rejected. **OPERATIONAL FOR i64 (Stage 2b) DONE:** DELETE/UPDATE BY
an i64 compound key now STAY ELIDED — a FINGERPRINT-based in-place tombstone-locate
(`try_tombstone_resident_delete_via_fingerprint`) replaces the Int4-scan `resident_int4_row_predicate`
for tables with an i64 key column: fold the row's key tuple -> fingerprint -> probe -> materialize each
hit on-device + TUPLE-VERIFY the key columns (so a collision can't tombstone the wrong slot) -> exact-1 ->
tombstone. UPDATE reuses it (tombstone-old). All-i32 tables keep the proven int4-predicate path. Focused
audit CLEARED (collision-safe, already-dead/SI-fix visibility sound, exact-1 sound). **b128 (Stage 2c) —
Numeric/Uuid INSERT-uniqueness + reads DONE:** a compound key over Numeric/Uuid columns folds 4 words
(16 LE bytes) via the widths kernel + `sql_value_key_words`; `materialize_resident_row_via_hit` now
reassembles b128 (i128 mantissa / raw uuid) so the recheck is device-native. LOAD-BEARING FIX: the needle
bind now uses `coerce_insert_value` (Text -> Uuid via parse_uuid; Numeric rescaled to the column scale) —
a uuid literal parses as Text and MUST coerce to bytes / a numeric to the column scale so host==device
fold agrees (audit scrutinized the numeric-scale invariant hardest — SOLID: stored value + needle share
the one rescale path, plus scale-independent Decimal128 recheck). Audit CLEARED. **b128 DELETE/UPDATE BY
KEY now DEVICE-NATIVE too:** the WHERE-literal coercion gap is CLOSED — `bind_delete_filter_groups` falls
back to `coerce_insert_value` (Text -> Uuid via parse_uuid; Text -> Timestamp) when `coerce_filter_literal`
leaves a type-mismatch, so `WHERE u='uuid-str'` matches the stored Uuid (also fixes uuid/timestamp WHERE
DELETEs generally). Audit CLEARED (fallback fires ONLY where the old code hard-errored -> no regression;
only Text->Uuid/Timestamp newly succeed). So the FIXED-WIDTH compound key types (int + numeric/uuid) are
now FULLY operational (INSERT-uniqueness + reads + DELETE/UPDATE). **TEXT (Stage 2d) — INSERT-uniqueness +
reads DONE (`df136262`):** a compound key over a TEXT column now ELIDES + validates uniqueness ON THE
DEVICE. A text column is variable-length, so it folds to ONE word = the FNV-1a hash of its UTF-8 bytes; the
device fold kernel `COMPOUND_FOLD_PTX` gains a TEXT SENTINEL branch (`widths[k]==0`) that reads the row's
`[start,end)` blob span (offsets array + blob, via a new `blob_offsets` kernel param) and hashes the bytes
BYTE-IDENTICALLY to the host `fnv1a_bytes` — so the device rebuild == the host probe needle. The byte loop
is UTF-8-exact (zero-extended `ld.global.u8`), not ASCII-only. `materialize_resident_row_via_hit` now
decodes a resident TEXT slot on-device (single-slot offsets+blob read) so the full-tuple recheck compares
the actual strings (a collision can't false-23505). `compound_key_type_supported`/`key_column_width_words`
accept Text (sentinel width 0); single-column keys stay i32-section (arity-aware); non-foldable types
(bool/float) stay rejected. Independent adversarial opus audit (PTX register liveness, host==device
byte-exactness, offset addressing, lanes recompute, recheck collision-separation, NULL text, multi-text
ordering) -> MERGE-SAFE, no defects. **So the ENTIRE compound-key type matrix (int2/4/8, date, timestamp,
numeric, uuid, text) is now device-native for INSERT-uniqueness + reads.**
GAPS: text/b128 DELETE/UPDATE BY KEY over a compound table with a text VALUE column or a NULL key column
de-elides (text-KEY DELETE/UPDATE not yet exercised — INSERT-scoped this slice); (c) reads over a VERSIONED
wider-type elided table de-elide (R-ver is int4-only). LEDGERED: fused-apply-for-compound, 64-bit
fingerprint, text-COMPACTION (rollover-only shard proliferation). See memory `type-coverage-14`,
`scalability-ledger`, `charter-governance-ruling`.

---

## ⛳ THE CHARTER IS THE CONSTRAINT — READ THIS FIRST

**The GPU is the execution substrate for the ENTIRE relational data path. The host is CONTROL PLANE ONLY.**
The end goal is to **REMOVE the CPU relational engine** — it exists today only as a parity oracle + a GPU-fault
safety net, both **interim debt to be deleted**, never product direction (ADR-006).

- **Host MAY:** wire I/O; SQL parse + plan; kernel orchestration/launch; txn coordination + sequencing;
  WAL/durability I/O; the staging upload (build + upload the next device generation); the single final
  device→wire result readback.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING, LIMIT/OFFSET,
  expression eval, NULL/3VL — and MUST NOT materialize results from `host_rows`.

**THE LOAD-BEARING DIRECTIVE (user, verbatim):** *"We need to stay GPU native with the solution. Follow the
charter."* When you MEASURE a host-side cost in a data-plane hot path, the fix is to **MOVE THE WORK ONTO THE
GPU**, not to optimize the host. **GOVERNANCE (user, 2026-07-03): agent-authored docs must never widen the
charter; exceptions exist only if written into CHARTER.md by the user.** Host-side addressing structures
(shard_pk_index et al.) are migration debt → wave-batched device probes, then DELETE
(memory: `charter-governance-ruling`, `stay-gpu-native-charter`).

**Success bar (trajectory bet):** same ORDER OF MAGNITUDE as a tuned CPU engine on today's hardware, with the
residual gap being GPU-ARCHITECTURAL so it closes as hardware advances. SLO: >100k TPS sustained, ≥400k burst,
p50/p99/p99.9 < 0.5/1/5 ms.

---

## WHERE WE ARE (all merged to main)

**The durable write engine is the default and the perf story is essentially won.** Arc this program:
32k → 414k (disruptor submit/poll) → 597k (fast lane) → 1.32M (FuaWalLaneSet intent lanes) → **1.68M
sustained / 2.78M burst durable TPS** (closed-loop per-request durable acks; steady ~1.5M with ±15% run
variance). E2.5c hardened + flipped it: lanes reopen/replay + crash-mid-wave orphan repair, atomic-sidecar
checkpoint/truncation + segment RECYCLE, and **THE DEFAULT FLIP** (`GPU_DB_WAL_DURABILITY=fua`,
`GPU_DB_INTENT_LANES=10`, fused-apply kernel — ALL default ON; champion knobs = defaults; deployment env:
`GPU_DB_INTENT_LANE_SEGMENT_BYTES=512MiB+`, `GPU_DB_OPEN_SHARD_FLOOR_ROWS` sizing).

**Latency:** the ≤512-client sync ack is DRIVE-BOUND (fence ~870µs/frame ≈ 90% of the ack; p50 1.08 p90
1.39ms @414k — consumer-NVMe FUA physics, not software). **Async commit (`3d1d7d55`, the pg
`synchronous_commit=off` model, opt-in per statement via `submit_covered_insert_intent_with_commit` /
engine default `GPU_DB_SYNCHRONOUS_COMMIT`):** acks at the APPLIED cut, WAL fences behind the ack; power
failure may lose a bounded suffix of async-acked intents, never consistency; visibility stays gated on
the STRICT cut (readers never see a revocable row; writer read-back lags ack ≤ ~one fence). A/B 512
clients: **async p50 0.40 p90 0.53 p99 0.74ms @968k vs sync 1.08/1.39 @414k — the sub-1ms p90 SLO is MET
on consumer NVMe under the relaxed contract**; 61k async 1.67M (high load is validate/apply-bound, not
durability-bound).

**Read path:** SETTLED at its architectural ceiling — 121.6M lookups/s @b65536, p99 476µs (banked; do not
re-litigate). Sharding = a SCALE play. The production facade combined read+write non-vacuity gate has now run;
see the current report-card paragraph above for the exact GPU-overlap and visibility-gather split.

**Merge-gate green baselines (preserve):** workspace 1174/1174 (default arm), engine 491/491
(explicit-serial arm), FULL GPU sweep 370/370 (`--test-threads=1`), intent suites lanes=2/6, wal crate
78/78, clippy clean.

---

## MEGA-FUSE: EXECUTED, then REVERTED TO `feature/mega-fuse` (2026-07-07, flag-reckoning policy)

Per the user's no-flag-proliferation ruling, the default-OFF mega-fuse arm was REMOVED from main and
preserved (code + audit + A/B record) on the `feature/mega-fuse` branch: A/B losers do not live on main.
It returns as a REPLACEMENT (not an alternative) once the cross-lane mega coalescer + WAL-first reorder
flip its economics. The section below is the executed record.

## MEGA-FUSE: EXECUTED (2026-07-07) — mechanism proven, ships DEFAULT-OFF, launch economics documented

The ratified next action ran end-to-end: recon → design → implementation → bug-find → fix → gates → A/B →
adversarial audit → ship. `GPU_DB_MEGA_FUSE=1` (default OFF) runs eligible covered-INSERT waves through ONE
probe-first device pass (`MEGA_FUSE_PTX`): host pre-resolves winner identity (ledger + dedup + seq/row-id
claims), the kernel probes the PK hash index per row (CAS), inserts winners, scatters values + stamps, and
returns per-row verdicts; verdict-1 rows get the authoritative snapshot recheck (visible = 23505,
tombstone-exonerated = classic-path requeue via the `no_mega` marker), and EVERY claimed seq is WAL-covered
(winners' records + EMPTY no-op records — replay-verified against the positional seq math and reopen
oracle seeding). **The separate validate launch is GONE on fused waves: 220 → 3µs/wave, 218k+ waves fired.**

**THE SUFFIX-TRIM LAW (found via the parity test's deliberate dup-PK probe):** a non-winner slot stamped
with the never-visible sentinel must NOT be published — tail losers are TRIMMED (row_count advances by the
winner prefix only; the device header word is re-corrected; hwm stays the real winner max), else the shard
pins permanently versioned and the ORDER-BY paths refuse forever. Interior losers (rare²) still pin
(`max_created_by = Index::MAX`, published atomically with the row-count advance) — correct, documented cost.

**HONEST A/B (why default-OFF): the per-wave blocking launch loses to the classic coalescers at BOTH ends.**
61k: mega {1.19, 1.26, 1.66}M vs classic {1.49, 1.54, 1.54}M — classic amortizes validate across lanes and
applies ~3 waves/launch asynchronously; mega serializes one launch per wave under the device lock. 512cl:
mega 290-300k @ p50 1.4ms vs classic 393-416k @ 1.10-1.26ms — mega launches BEFORE the WAL append,
serializing device time ahead of the fence and reintroducing the inline device wait no-reap removed.
**FOLLOW-UP that would flip the economics: (1) a CROSS-LANE MEGA COALESCER (one probe-insert launch over
all lanes' pending waves — the per-needle verdict design already supports it) + (2) WAL-FIRST reorder
(append winners optimistically, launch overlapping the fence, reconcile losers via the empty-record
mechanism).** Audit: MERGE-SAFE default-OFF; one MEDIUM adopted (mega WAL-append failure now poisons the
lanes loudly — the apply-before-append inversion must never silently serve phantom rows); interior-loser
index-decline documented as perf cost. Gates: MEGA suites 4/4 (default/lanes6/async arms), FULL GPU sweep
371/371, engine 491/491 both arms.

## >>> THE ONE NEXT ACTION: TYPE COVERAGE (#14) — the CPU-engine-deletion gate <<<

**The mixed-OLTP hot path is now fully GPU-native for the common int4-PK shapes: sustained writes
(F3/U4) AND reads (R-ver PART 1+2) over versioned tables no longer de-elide/refuse.** R-ver PART 2
(MERGED) threaded `ResidentVisibility` through the GROUP BY / DISTINCT / ORDER BY sharded sub-bridges
(the executor computes ONE survivor set `indices` = predicate AND visibility BEFORE group/sort/dedup,
and every reshaping kernel reads only `indices` — so no per-kernel change; both refusals deleted).
Gate `grouped_ordered_distinct_over_versioned_elided_hides_tombstones` + the flipped
`sharded_predicate_null_3vl_on_versioned_shard`; opus audit MERGE-SAFE.

**NEXT: TYPE COVERAGE (#14)** — the true CPU-engine-deletion gate (memory `type-coverage-14`). The
covered/elided fast path is int4-PK only. Order by leverage: numeric/uuid (i128) + bool (bitmap) are
incremental (reuse the per-type section machinery); **text LAST** (variable-length forces a new
append/rollover design); then compound PKs (rejected at parse today). Each type moved onto the GPU
path shrinks the CPU relational engine's surface until ADR-006 can DELETE it — the charter's finish
line. Small read residue also open: int8/date-bearing tables' UNFILTERED scans still de-elide (R-ver
is int4-only projection routing — extend the `int4_projection_all` classifier to int8/date).

**U2 lane UPDATE + F3/U4 COMPLETE + MERGED (2026-07-07):** a covered UPDATE = tombstone-OLD +
append-NEW at a FRESH new_row_id, both AT APPLY (WAL-first). F3/U4 = the version-aware dup-tolerant
device PK index (2 insert kernels + write-locate advance-past-match + `build_visible` dup_tolerant
gated on `deleted_stamps.is_some()`) so SUSTAINED updates no longer de-elide (mixed bench 1.4M TPS
/ 3 pk-rebuilds; was de-elide-every-update → panic). Two adversarial audits: U2 CRITICAL (replay
lock-step assert assumed rowid-order==seq-order under concurrent lanes — FALSE, removed) + F3/U4
CRITICALs 1/2 (write-locate first-match missed same-shard twins → point read empty + INSERT bypass
— fixed) all FIXED + sabotage-verified + regression-gated (`device_locate_same_shard_twin_*`,
`gpu_lane_update_sustained_stays_elided`, the high-water recovery gate). MEDIUM (elided_commit_delta
old-row removal) closed via `old_row_ids`. Gates: FULL GPU sweep 377/377, engine CPU 493, clippy
clean. Mixed I:U:D bench arm (`GPU_DB_BENCH_MIX_UPDATE`, bench-only). Memory `u2-lane-update-design`.

**TIER-1 DELETE PATH COMPLETE + MERGED (2026-07-07):** lane DELETE intents (U1) shipped, then made
WAL-FIRST — the delete locate + tombstone + rows-affected moved OFF the pump critical path to
apply-time, so a delete's ack is FENCE-BOUND (512-client mix=20% p50 1.58→1.19ms, at the
insert-only 1.13 floor). Merges: U1 core + rebuild fix, perf lever B (batched tombstone scatter
kernel — mix=20% 808K→1.34M), WAL-first (`6856ebcd`, supersedes the reverted lever A). All
opus-audited MERGE-SAFE. Two tracked non-blocking follow-ups (memory
`u1-lane-delete-implementation`): F1 duplicate same-key deletes double-count (likely unreachable),
F2 0-row deletes record a ledger slot → retryable same-key-insert abort. The mixed I/D bench arm
(`GPU_DB_BENCH_MIX_DELETE`, bench-only) is merged for sizing.

---

## >>> PREVIOUS NEXT ACTION (executed + reverted): THE MEGA-FUSE (user-ratified 2026-07-07) <<<

**Fuse validate+insert into ONE device launch over the merged cross-lane batch, by pre-resolving winner
identity host-side.** This is the only remaining structural 2M+ lever — config space is EXHAUSTED at ~1.5M
steady (the 1.66-1.68M records are favorable variance), and per-second stage attribution proved the dips are
drive stalls + closed-loop breathing, NOT stage inflation. The **validate stage (~210-320µs/wave, the
wave-batched device PK locate through the coalescer) is the LAST INLINE DEVICE WAIT** in the pump cycle.

What's known from the recon (2026-07-06):
- **Validate chain:** `wave_batch_locate_hit_counts_direct` (engine_retained_read.rs) →
  `submit_multi_shard_i32_write_locate` (execution/src/lib.rs, the proven M1 write-locate PTX).
- **Apply chain:** already fused — `FUSED_APPLY_PTX` (execution/src/lib.rs): one kernel for column scatter +
  created_by/row-id stamps + device row-count header + PK index CAS insert, default ON (+14-17%). **It is the
  template: validate's probe and the fused insert already share the per-shard device hash-index buffers.**
- **The blocker to clear (this is the actual work):** row-ids/commit-seqs are assigned post-conflict
  post-WAL-propose, and winner selection needs the host SI ledger + WAL commit order — so today a device
  probe-miss is not yet a commit. The redesign PRE-RESOLVES winner identity host-side (ledger/dedup/seq
  claim BEFORE the launch) and passes it in, so the kernel can probe→branch→CAS-insert in one pass.
  The count>0 host tombstone-recheck path must keep a correct fallback.
- **Known negatives (do not retry):** validate-overlap restructure (statistically neutral — the wait is
  coalescer-round queueing); passive wave coalescing; flusher-side sleeps; smaller GROUP_US; window/lane/
  fence/driver sweeps (all exhausted, see HANDOVER_REMAINING_WORK.md).

Expected shape: default-OFF flag, fired-counter, dup-key single-winner races re-gated under lanes, A/B
best-of-3 at the champion config + the 512-client floor. Slice gates per protocol below.

**THEN → TIER 1 PRODUCTION COVERAGE (user-set sequence, 2026-07-07):** the lane fast path is
covered-INSERT int4-PK only. Order: (1) **UPDATE/DELETE intents on the lane path** (largest functional
cliff; re-opens wave-batched locate in lane form), (2) crash/power-fail injection harness over the FUA
lane WAL + checkpoint sidecar (harden what's shipped before more surface lands), (3) type coverage
remainder — numeric/uuid (i128), bool (bitmap), **text LAST** (variable-length forces a new
append/rollover design), (4) compound PKs (rejected at parse today), (5) **the mixed read+write gate ✅ DONE
(2026-07-12; production facade, exact GPU/write overlap counters)**,
(6) CPU-engine deletion (ADR-006 — TC#14 is its gate). PLP-class media procurement runs in parallel
(would retire both the 0.9ms fence floor and the rare ~0.5s drive-stall tail with zero code changes).

---

## OPEN BOARD (beyond the one next action)

- **Drive-stall tail / PLP retest:** rare ~0.5s FUA stalls hold the contiguous cut (drive physics);
  re-run the low-client curve + a 30s attribution run on PLP-class media when available.
- **Lanes auto-checkpoint policy:** `checkpoint_intent_lanes()` is operator-explicit; needs the
  `maybe_auto_checkpoint_wal` twin once a cadence policy is chosen.
- **Archive/PITR in lanes mode:** correctly refused (F5) — lane commits carry no per-commit timestamps in
  v1; needs a lane-record timestamp story.
- **Multi-node/Raft:** assessed + deferred — lane seq claim ↦ Raft log-index block reservation; ack gates
  on QUORUM commit instead of the local FUA cut. A program, not a slice.
- **Scalability ledger rows:** #15 uncapped per-row locate loop; #26 populate-vs-commit re-admission race;
  concurrent-elision first-transition TOCTOU; dense-admission eviction-rollback residual. VACUUM V2s
  (key-clustered rebuild, background thread, park starvation). Multi-GPU. Ledger #18 32w PK'd inversion.
- **Bench honesty follow-up:** exclude the ~6s warm-up ramp from the sustained metric.
- **/tmp hygiene (operational hazard):** /tmp is QUOTA'd — leaked test WAL segments EDQUOT-wedge the whole
  box. `.cargo/config.toml` sets `TMPDIR=target/tmp` (dir must exist: `mkdir -p target/tmp`); never ship
  big segment-size DEFAULTS; `target/wal-intent-bench` accretes ~10GB/champion-run — rm periodically.

---

## The per-iteration protocol (NON-NEGOTIABLE)

1. **MEASURE first** — reproduce/confirm before changing code.
2. **Smallest correct slice on a BRANCH — NO feature flags (user mandate 2026-07-07):** a path
   merges only when correct AND complete, and it merges AS the path (the arm it replaces is deleted
   in the same merge); unfinished or losing paths stay on their branch (learnings in docs/memory).
   Product config stays ~5 documented settings (docs/CONFIG.md); bench knobs never read by the engine.
3. **GPU-native solution (the charter)** — data-plane hot path on the device, not the host.
4. **Differentials**: GPU == CPU == the SQL SPEC (memory `sql-spec-over-cpu-parity`). Prove the new path
   FIRED (a counter), not a silent fallback.
5. **NON-VACUOUS sabotage-verified asserts** — break the mechanism, watch the test FAIL, revert.
6. **INDEPENDENT ADVERSARIAL OPUS AUDIT before EVERY push** — never self-audit, never push unaudited;
   adopt findings → re-verify → push. Memory `audit-with-opus-subagents`. Main agent IMPLEMENTS directly
   (no delegation of implementation — memory `implementation-ownership-directive`).
7. **Pair LATENCY (p50/p99) with THROUGHPUT** on every benchmark line (memory `benchmark-report-card`).
8. **Update memory** after each slice; re-read the scalability ledger each loop (no new unscalable
   hot path without a ledger row).

**GPU discipline (hard rules):** GPU tests under `timeout`; **NEVER `--gpu-reset`**; **sweeps with
`--test-threads=1`** (parallel oversubscribes → spurious failures; re-run failures in isolation);
never a GPU test right after a timeout-killed one; **ASCII-only PTX**. Push to origin/main per standing
authorization (`git fetch && git rebase origin/main` first; no workspace-wide cargo fmt — scope
`-p gpu_db_engine`). User prefs: **BIGGER SLICES, FEWER CHECK-INS**; the USER sets the sequence at track
boundaries (memory `working-agreement-sequencing`).

---

## Pointers

- **Historical agent memory (not distributed with the project):** the private memory index used during this
  archived handover referenced:
  `e25c-hardening-default-flip` (the shipped state), `fua-pipelined-durable-wal` (the durable laws),
  `tmp-quota-test-wal-hygiene`, `scalability-ledger`, `type-coverage-14` (Tier 1 frame),
  `stay-gpu-native-charter`, `gpu-test-threads-serial`, `benchmark-report-card`.
- **Write path code:** intent lanes + pumps + waves = `engine_dml_concurrent.rs`; FUA lane WAL + cuts +
  pre-stager + recycle = `crates/wal/src/fua.rs`; fence pool = `crates/write_conveyor` +
  `fua_frame_log.rs`; validate chain = `engine_retained_read.rs`
  (`wave_batch_locate_hit_counts_direct`) → `execution/src/lib.rs`
  (`submit_multi_shard_i32_write_locate`, `FUSED_APPLY_PTX`); conflict ledger = `write_path.rs`;
  elision/rehydration/VACUUM = `engine_residency.rs`; shard reference = `docs/SHARD_STORAGE.md`.
- **Benchmarks:** `oltp_commit_slo_benchmark` — champion config: `GPU_DB_WAL_DURABILITY=fua` (default)
  `GPU_DB_BENCH_ARM=driver WRITERS=12 PUMPS=10 WINDOW=6144 GPU_DB_INTENT_LANE_MIN_WAVE=1024
  GPU_DB_INTENT_LANE_GROUP_US=4000 GPU_DB_OPEN_SHARD_FLOOR_ROWS=48000000 SHARD_TARGET=48000000
  GPU_DB_INTENT_LANE_SEGMENT_BYTES=536870912`; diagnostics: `GPU_DB_BENCH_TIMELINE=1`,
  `GPU_DB_BENCH_PIPEPHASE/HOSTPHASE=1`; low-load floor: 512 clients, defaults.
