# Lane UPDATE/DELETE Intents — Tier 1 v1 Design (PROPOSAL, UNACCEPTED — review round 1 incorporated)

> Status: **design prep only** (user-directed 2026-07-07; no code). **Rev 3** — rev 2
> incorporated the mega-fuse author's review (against `4291cdb0`: pump-inline verdicts,
> single-shard probe, suffix-trim law); rev 3 applies the NO-FEATURE-FLAGS mandate — the mega
> arm now lives on `feature/mega-fuse` (removed from main), all flag/flip language replaced by
> branch discipline (§9). Decision points for the user are marked **[DECIDE]**;
> previously-open ones carry their ratified answers.
>
> **Review changelog (all points adopted):** §3 rewritten against the shipped contract — the
> CLASSIC lane pipeline is the base arm, mega op-codes deferred to U4; locate joins the existing
> coalesced validate launch (pump-time verdicts, no settle-side channel); 0-row ops filter
> PRE-CLAIM (no WAL record, no burned slot — the suffix-trim hazard is mega-arm-only); the
> visibility-aware duplicate-tolerant index rebuild moves INTO U1; §7's replay claim corrected
> (frozen serial prefix — flat refusal stays until U3 specs lane-seq-space entry); outcome
> plumbing = widen `CommitWaveOutcome` to `Result<u64>`; second-unique-column tables v1-excluded.

## 1. Problem and scope

**The production cliff:** after the first lane commit the engine is *intent-only* —
`intent_lanes_write_guard` refuses ALL classic DML/DDL ("v1 lanes contract"). The lane fast path
covers exactly one op: covered INSERT on int4-PK elided tables. A database that can never UPDATE,
DELETE, or run DDL after its first fast write is not a production engine. Tier 1 closes the DML
half of that cliff; the DDL/non-covered half is U3 (§7).

**v1 covered shapes** (mirroring the insert route's discipline — O(1) shape proof at prepare,
per-intent execution allocation-lean):

- `DELETE FROM t WHERE <pk_col> = $1`
- `UPDATE t SET <non-pk assignments, all INT4 constants/params> WHERE <pk_col> = $1`

Same table eligibility as `prepare_covered_insert_route`, plus (ratified): **tables with any
unique i32 index beyond the PK are v1-excluded at route prepare** — the simplest shape proof; a
partial-coverage rule invites drift bugs. Rows-affected is 0 or 1 by construction.

**v1 exclusions** (refused with a clear error until U3; never silently degraded):
- UPDATE of the PK column itself (cross-lane atomicity — v2, §9).
- Non-PK predicates, multi-row statements, RETURNING.

## 2. The op model

`LaneIntent` gains `op: LaneOpKind { Insert, Delete, Update }` + optional fields; the insert
fields stay as-is so the hot insert path is byte-identical when no U/D intents exist.

- **Delete** `{ slot (pk_slot_id, pk_value), read_snapshot, template (W5b delete record) }` —
  no values, no row-id/headroom reservation.
- **Update** `{ slot, new_values (full post-image, catalog order), read_snapshot, template }` —
  reserves ONE headroom slot + row_id for the new version ONLY after the locate confirms 1 row
  (§3.3). MVCC: update = tombstone old + append new (classic `PreparedMutation::Update`
  semantics; never in-place mutation — readers at older snapshots keep the old version).

**Outcome plumbing (ratified):** widen `CommitWaveOutcome` to `Result<u64, ExecuteError>`
(insert = 1) engine-wide, in U1 while the surface is smallest. `poll_intent` returns the count.

**Routes:** `prepare_covered_delete_route` / `prepare_covered_update_route`: pk slot id + column
index, catalog_seq, W5b record prefix + patch offsets, synthesized-SQL fallback prefix; update
routes pin the assignment column set (catalog order).

## 3. Pipeline integration — the CLASSIC lane pipeline is the base arm

**Shipped reality this rev binds to:** main runs the classic pipeline ONLY (inline coalesced
validate at the pump → optimistic publish → no-reap apply coalescer). The mega-fuse lost its A/B
(per-wave blocking launch loses to the coalescers at both load ends) and was moved to
`feature/mega-fuse` per the no-feature-flags mandate. U/D therefore ship on the classic arm;
mega op-codes are the U4 arm, developed on that branch and mergeable only as a REPLACEMENT if
the economics follow-up (cross-lane mega coalescer + WAL-first reorder) wins.

### 3.1 Routing and wave formation — unchanged
PK-hash lane routing serializes every op on a key through one lane in submit order. Mixed-op
waves allowed. Resize-barrier epoch argument unchanged.

### 3.2 Locate joins the existing validate launch (no new inline wait)
`lane_validate_unique` already runs ONE coalesced multi-shard write-locate launch per round over
the wave's insert needles (`submit_multi_shard_i32_write_locate` — whose kernel ALREADY outputs
`(shard, slot)` hit arrays; the counts-only wrapper discards them). U/D needles join the SAME
launch with a mixed needle set:

- **insert needle** → hit count (dup verdict, as today);
- **delete/update needle** → hit locations `(shard, slot)` + count.

Hits get the same authoritative host recheck at the item's snapshot (`visible_row_with_value`
semantics): a DEAD hit (tombstoned twin) resolves to **0 rows** for U/D — correct SQL semantics
(deleting a dead row affects nothing) — and to **insert-proceeds** for inserts (unchanged).
Zero additional launches; the locate rides the validate coalescer's existing amortization.

### 3.3 Conflict pass, 0-row filter, claim
Ledger rules (unchanged from rev 1 — first-updater-wins):

| op | ledger check (slot, snapshot) | on pass, record |
|---|---|---|
| INSERT | conflict if slot seq > snapshot | slot → my seq |
| DELETE | conflict if slot seq > snapshot | slot → my seq |
| UPDATE | conflict if slot seq > snapshot | slot → my seq |

Intra-wave same-slot: first op wins; later same-wave ops get retryable serialization errors.
**Differential expectation (pinned per review):** insert-after-delete of the same key within the
ledger window surfaces as a SERIALIZATION error on the lane path where the classic path may
raise 23505 — both SI-defensible; tests expect the serialization error and document the
divergence.

**The 0-row filter (new, and the load-bearing simplification):** locate verdicts exist BEFORE
the claim. A U/D whose needle resolved to 0 live rows completes at the conflict pass with
`Ok(0)` — **no seq claim, no WAL record, no headroom slot, no durability wait** (nothing was
written; PG-consistent). Consequences:
- No burned update slots in the base arm — **the suffix-trim/interior-sentinel hazard (review
  point 4) cannot occur on the classic arm at all**; it is confined to the U4 mega arm (§9).
- No 0-row records to replay; W5b records exist only for 1-row outcomes.

TOCTOU safety of locate-before-claim: all same-key ops are lane-serialized and intra-wave
same-slot ops are first-wins-rejected, so a located live row cannot be tombstoned by anyone else
between locate and apply; the apply-time `deleted_by` CAS is therefore guaranteed-win
(debug-asserted, not a verdict).

### 3.4 Claim + optimistic publish — unchanged shape
Seq block for wave winners (dense; 1-row U/D members included), row-id block only for
insert/update members. W5b records patched + frame-encoded in the fused patch pass, published at
pump time. Publish stays parallel on the pumps (never leader-side — the shipped mega audit's
saturation math). Safe without markers because U/D records are by-key + re-resolving at replay
(§5); the shipped INSERT loser mechanism (every claimed seq WAL-covered via EMPTY no-op records)
is orthogonal and unchanged.

### 3.5 Apply — tombstone work rides the apply coalescer
`ApplyRequest` gains parallel tombstone vectors: `(shard, slot, seq)` per 1-row U/D member.
`lane_apply_merged`'s leader pass adds ONE batched device stamp launch per merged pass (all
lanes' pending tombstones in one kernel: `atom.global.cas.b64 deleted_by[shard][slot] 0 → seq`),
alongside the existing merged append (update new-versions ride the append exactly like inserts).
No verdict DtoH — outcomes were resolved at the pump (§3.3); settle stays winners-only,
matching the shipped shape. `done`/cut advance semantics unchanged.

Stamp visibility correctness: monotonic u64 stores; a reader at snapshot < seq evaluates
`deleted_by > snapshot` → still visible (the SV6 stamping-under-readers argument). Stamps land
before the cut covers the wave, and visibility publication is cut-gated as ever.

**deleted_by sidecar (ratified: on-demand):** `shard_deleted_by_memory` allocates on a shard's
first delete. The apply LEADER materializes missing sidecars for the merged batch's target
shards before the stamp launch (one-time zeroed alloc per shard; every later wave is pure
kernel). Pre-materializing at admission (384MB/shard at the 48M floor) rejected.

### 3.6 The mega arm (U4, deferred)
When the mega economics flip (cross-lane mega coalescer + WAL-first reorder), U/D become per-item
op codes in `MEGA_FUSE_PTX` (probe-at-apply): delete = probe→CAS-stamp; update = probe→stamp +
append. That arm re-opens the 0-row-at-apply problem: burned update slots are INTERIOR sentinels
under the shipped SUFFIX-TRIM LAW (only tail losers trim; an interior sentinel pins
`max_created_by = Index::MAX` → ORDER-BY refused until rollover). Spec for U4, gated on U1/U2
measurements: (a) allocate update slots at the wave TAIL so burns trim; (b) the mixed bench must
show the 0-row-update rate before accepting any residual pin; (c) **[DECIDE at U4]** whether
VACUUM's dense rebuild resets `max_created_by` (today nothing unpins a shard). Consuming
verdicts stays wherever the insert arm consumes them (pump-inline today; deferred if the
coalesced-mega follow-up moves them) — U/D op semantics do not depend on the location.

## 4. Dead twins and the index — visibility-aware rebuild lands IN U1 (review point 5)

Delete→reinsert leaves a dead index entry; today the incremental insert CAS-hits it → decline →
host rebuild — and the rebuild indexes `[0, row_count)` INCLUDING dead rows' keys, so the rebuild
collides on the same key and **declines permanently**: any delete workload degrades the whole
locate path, not once but forever. Fix in U1 (host-side, cheap relative to kernel entry
replacement):

1. **Duplicate-tolerant build:** same-key entries occupy separate slots (open addressing already
   permits it); probes already return up to `max_hits` per needle + host visibility recheck —
   uniqueness semantics are unaffected. This alone removes the permanent decline.
2. **GC-boundary skip (refinement over "skip tombstoned"):** the PK index serves POINT READS as
   well as write-locate — a reader at an old snapshot must still find a dead version. The
   rebuild may omit only rows whose `deleted_by` < the active-snapshot GC boundary (the ledger
   prune boundary — no live reader can see them); twins above the boundary stay indexed and are
   handled by (1).

Kernel-level index-entry replacement (probe hit → dead check → CAS old→new packed entry) remains
the U4/v1.1 optimization if the mixed bench shows rebuild frequency still hurts.

## 5. W5b binary records + replay (unchanged from rev 1, now 1-row-only)

New op codes in `wal_binary.rs` (tag 0xFF, version bump):
- `OP_DELETE_BY_KEY { table, pk_col, pk_value }`
- `OP_UPDATE_BY_KEY { table, pk_col, pk_value, new_row_id, new_row_image }`

Replay (`apply_binary_wal_entry` arms): re-resolve the pk against the HOST store at replay time,
tombstone / tombstone+install. Determinism: replay runs in global seq order; all ops on a key
were lane-serialized in that order live; cross-key ops commute — replay reproduces the live
outcome. 0-row ops never reach the WAL (§3.3), so every U/D record replays to exactly 1 row
(assert LOUDLY at replay: a 0-row re-resolve of a durable U/D record is corruption or a
determinism bug, never silently skipped). No abort/no-op records needed for U/D.

Checkpoint/truncation: op-agnostic (checkpoints snapshot applied state). Archive/PITR stays
refused in lanes mode.

## 6. Read path, elision, VACUUM (unchanged from rev 1)

- Mask-VM visibility already evaluates sparse deleted_by — no reader changes.
- Elision: rehydration understands tombstones; `lane_apply_merged`'s rare rehydrate fallback arm
  grows tombstone equivalents (host store), same panic-on-invariant discipline.
- VACUUM #5 churn triggers now fire on lane tables; verify the deferred-tail auto-trigger against
  lane-applied tombstones (counter + regression); vacuum's lane interaction uses the existing
  drain/commit-lock/recheck loop.

## 7. Non-covered writes: flat refusal stays until U3 (corrected per review)

Rev 1 claimed the serial+lanes merge replay "already interleaves correctly by seq" — **wrong**:
reopen replays the serial log as a FROZEN PREFIX (`base_seq` = serial record count at
activation; the lanes own everything above; `next_seq == base_seq + lane_record_count` is
asserted). A post-activation serial append would replay out of position or trip the assert.
Quiesced classic writes must enter the LANE seq space. U3 spec sketch (design work, not assumed):
quiesce barrier (reuse the resize-barrier drain) → execute the classic statement under the
commit lock → encode its record(s) AS LANE RECORDS through a designated lane with claimed lane
seqs (the serial log stays frozen) → resume. Until U3 lands, the v1 contract's flat refusal
stands — correct over convenient.

## 8. Gates, counters, benchmarks

- Counters: per-op wave counts, locate verdict histogram (1-row / 0-row / dead-hit), 0-row
  pre-claim filters, sidecar materializations, duplicate-tolerant rebuild count, rebuild
  GC-boundary skips. Prove-the-path-fired discipline throughout.
- Differentials vs classic path on a non-activated twin: 0-row cases, snapshot conflicts,
  delete→reinsert (EXPECT serialization error in-window, 23505/insert-ok per visibility
  outside), update-then-read at old/new snapshots, GPU==CPU==spec.
- Sabotage: skip the deleted_by CAS → visibility test FAILS; break replay re-resolve → recovery
  parity FAILS; break the ledger delete rule → lost-update test FAILS; break duplicate-tolerant
  rebuild → delete→reinsert locate test FAILS.
- Recovery: durable replay parity incl. interleaved I/U/D + crash-mid-wave orphan repair +
  reopen-continues with mixed ops; the LOUD 0-row-replay assert sabotage-verified.
- Bench: `intent_fast_path_bench` mixed arm `GPU_DB_BENCH_MIX=I:U:D` (70:20:10 core-banking
  shape) at champion config + 512-client floor; rows-affected-weighted TPS with latency pairing.
  Insert-only baseline to preserve: {1.50, 1.41, 1.50}M. The mixed arm also SIZES: 0-row-update
  frequency (U4 gate), rebuild frequency post-U1-fix (kernel-replacement gate).
- Full suites: engine default+fua arms, GPU intent suites (serial + lanes=2/6), wal crate,
  clippy; TMPDIR hygiene. (MEGA suites live on `feature/mega-fuse`; rebasing that branch over
  U1/U2 must teach mega waves to refuse/route around U/D items until U4 adds the op codes.)

## 9. Slice plan (ratified order) + remaining decisions

- **U1 — DELETE end-to-end (classic arm):** LaneOpKind + routes, mixed-needle locate in the
  validate launch, ledger rule, 0-row pre-claim filter, `Result<u64>` outcome widening, apply-
  coalescer tombstone launch + leader sidecar materialization, W5b delete record + replay arm +
  LOUD assert, **visibility-aware duplicate-tolerant index rebuild**, full gate set.
- **U2 — UPDATE** (rides U1 + insert machinery: post-locate slot reservation, tombstone+append).
- **U3 — quiesce escape hatch** with the lane-seq-space entry protocol (§7).
- **U4 — mega-arm op codes** + tail-allocated update slots + VACUUM unpin decision + (if still
  needed) kernel index-entry replacement — gated on the mega economics follow-up and the U1/U2
  mixed-bench measurements.
**Branch discipline — NO feature flags (user mandate 2026-07-07, supersedes rev 2's flag
addendum):** there is no U/D flag and no "default flip" slice. Each slice develops on a BRANCH
and merges only when correct and complete — at which point lane U/D intents simply ARE the
engine's behavior (there is no old arm to keep: post-activation U/D was a refusal, so U1/U2 are
pure new capability; the refusal error for still-uncovered shapes remains until U3). The U4 mega
arm develops on `feature/mega-fuse` and returns to main only as a REPLACEMENT for the classic
arm if its economics win — the losing arm is deleted in the same merge. `GPU_DB_BENCH_MIX` is a
bench-only knob, never read by the engine.

Remaining **[DECIDE]**s: U4's VACUUM `max_created_by` unpin (§3.6); PK-update v2 (cross-lane
two-phase vs quiesce-only — defer until a workload demands it); who implements U1
(mega-fuse author offered; ownership is the user's call).

## 10. Contact surfaces

`engine_dml_concurrent.rs` (validate launch, conflict pass, apply coalescer, mega routing guard),
`engine_intent_lanes.rs` (LaneIntent/ApplyRequest), `engine_dml_intent.rs` (routes/API/outcome
widening), `wal_binary.rs` + `engine_commit.rs` (W5b + apply arms), `engine_residency.rs`
(sidecar materialization, index rebuild fix, rehydrate fallback), `engine_lifecycle.rs` (replay),
`execution/src/lib.rs` (locate wrapper returning locations; stamp kernel; U4 mega op codes).
