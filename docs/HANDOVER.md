# HANDOVER — Resume Baton

> **This is a SINGLE ROLLING file. Overwrite it each session — never date it, never accrete.** Where we are,
> the open decision, and the rules. The **why** is in DECISIONS.md; the **how** in ARCHITECTURE.md; the
> **mandate** in CHARTER.md; the **plan** in PLAN.md. The E2.5c campaign detail + gate ledger is in
> HANDOVER_REMAINING_WORK.md; the WAL/conveyor research record is in WRITE_CONVEYOR.md.

**Updated:** 2026-07-09. **Base:** `main` @ `28fe8df8` (CPU-ENGINE RETIREMENT — TWELVE merged wins: MULTI-BOUND
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
shard `gather_resident_table_rows_from_device` edge (mostly closed by `bb2a2c03`); CHECK/FK block elision. ARCHITECTURAL
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
re-litigate). Sharding = a SCALE play. **Note:** the read path was settled BEFORE the lane architecture; a
combined read+write gate has never been run (Tier 3 below).

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
append/rollover design), (4) compound PKs (rejected at parse today), (5) the mixed read+write gate,
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

- **Memory (read first):** `MEMORY.md` index at
  `~/.claude/projects/-home-richard-projects-gpu-database-engine/memory/`. Key files:
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
