# Lane UPDATE/DELETE Intents — Tier 1 v1 Design (PROPOSAL, UNACCEPTED)

> Status: **design prep only** (user-directed 2026-07-07; no code). Depends on the mega-fuse
> slice (in flight, other session) — §3.4 states the exact contract this design consumes from it.
> Author lane: main session. Decision points for the user are marked **[DECIDE]**.

## 1. Problem and scope

**The production cliff:** after the first lane commit the engine is *intent-only* —
`intent_lanes_write_guard` refuses ALL classic DML/DDL ("v1 lanes contract"). The lane fast path
covers exactly one op: covered INSERT on int4-PK elided tables. A database that can never UPDATE,
DELETE, or run DDL after its first fast write is not a production engine. Tier 1 closes the DML
half of that cliff (DDL quiesce rides the same escape hatch, §7).

**v1 covered shapes** (mirroring the insert route's discipline — O(1) shape proof at prepare,
per-intent execution allocation-lean):

- `DELETE FROM t WHERE <pk_col> = $1`
- `UPDATE t SET <non-pk assignments, all INT4 constants/params> WHERE <pk_col> = $1`

Same table eligibility as `prepare_covered_insert_route`: strictly-INT4 columns, elided
(device-authoritative), FK/CHECK-free, no inbound FK, unique indexes all-i32, binary WAL records
enabled. Rows-affected result is 0 or 1 by construction.

**v1 exclusions** (fall back to the §7 escape hatch, never silently degraded):
- UPDATE of the PK column itself (delete+insert of a *different* key = two lanes; cross-lane
  atomicity is a v2 protocol — see §9).
- Non-PK predicates, multi-row statements, RETURNING, secondary unique columns beyond the PK'd
  shape's existing slots.

## 2. The op model

`LaneIntent` generalizes to carry an op kind (structurally: keep the flat struct, add
`op: LaneOpKind { Insert, Delete, Update }` + optional fields; the insert fields stay as-is so
the hot insert path is byte-identical when the flag is off):

- **Delete** `{ slot (pk_slot_id, pk_value), read_snapshot, template (W5b delete record), ... }`
  — no `values`, no row-id reservation, no headroom slot.
- **Update** `{ slot, new_values (full post-image, catalog order), read_snapshot, template }`
  — reserves ONE headroom row slot + row_id for the new version (MVCC: update = tombstone old +
  append new, exactly the classic `PreparedMutation::Update` semantics — never in-place value
  mutation, readers at older snapshots keep the old version).

**Outcome plumbing:** `poll_intent` today yields `Result<(), ExecuteError>`. Update/delete need
rows-affected. **[DECIDE]** (a) widen `CommitWaveOutcome` to `Result<u64>` engine-wide (insert = 1),
or (b) new `poll_intent_rows` beside it. (a) is cleaner; touches every settle site — do it in the
first slice while the surface is small.

**Routes:** `prepare_covered_delete_route` / `prepare_covered_update_route` returning route
structs that precompute: pk slot id + column index, catalog_seq, W5b record prefix + patch
offsets, and the synthesized-SQL fallback prefix. Update routes additionally pin the assignment
column set (catalog order) so a param vector maps positionally.

## 3. Pipeline integration (per pump stage)

The disruptor law holds: nothing blocks before the ack except the client's own poll; the pump
gains NO new inline device waits (that's what the mega-fuse just removed for inserts).

### 3.1 Routing and wave formation
Same PK-hash lane routing (`lane_for_pk(pk)`): every op on a key serializes through ONE lane in
submit order. Mixed-op waves are allowed (one drain = one wave, ops interleaved). The resize
barrier's epoch argument is unchanged (all ops drain at the flip; apply-time device state catches
cross-epoch conflicts).

### 3.2 Host conflict pass (the ledger)
The lane ledger stays the SI authority for the un-applied window; the invariant
(seq ≤ snapshot ⟹ within visible cut ⟹ applied ⟹ device-visible) already proven for inserts
covers update/delete probes at apply time too. Per-op rules:

| op | ledger check (slot, snapshot) | on pass, record |
|---|---|---|
| INSERT | conflict if slot seq > snapshot (ww) | slot → my seq |
| DELETE | conflict if slot seq > snapshot (first-updater-wins; the version I read moved) | slot → my seq |
| UPDATE | same as DELETE | slot → my seq |

Intra-wave same-slot: **first op wins, later ops in the same wave get retryable serialization
errors** (v1; matches the insert dup rule, avoids in-wave dependency chains). A client doing
delete→insert on the same key pipelined must poll the delete before submitting the insert
(document on the API).

### 3.3 Claim + optimistic publish (unchanged shape)
Seq block claimed for all wave winners (dense); row-id block claimed only for insert/update
members (update's new version). W5b binary records (§5) are patched + frame-encoded in the same
fused patch pass and published at pump time, BEFORE the verdict exists — deliberately, to keep
publish parallel on the pumps (mega-fuse audit trap #3). This is safe because update/delete
records are **by-key + re-resolving at replay** (§5), unlike insert records: a 0-row live outcome
replays as a 0-row outcome deterministically. No abort markers needed for update/delete.

### 3.4 Verdict-at-apply — the mega-fuse contract this design consumes
Required from the in-flight mega-fuse slice (audit checklist enforces these anyway):
1. per-item verdict array DtoH from the fused apply launch;
2. `ApplySlot` verdict publication (visible to settle before `done`);
3. settle-side per-item outcome dispatch (Err/count, cut-gated);
4. multi-shard probe descriptor table in the staging image;
5. leader-side validate fallback when the fused path can't run.

The kernel gains a per-item **op code**. Thread j:
- **INSERT** (as mega-fuse): sealed probe → open CAS-insert → scatter/stamp or sentinel+verdict.
- **DELETE**: probe ALL shards for pk (write-locate math). Miss → verdict `0 rows`. Hit
  (shard s, row r) → `atom.global.cas.b64 deleted_by[s][r] 0 → my_seq`; CAS-loss (already
  tombstoned by an earlier wave — ledger makes same-slot races impossible, so loss can only be a
  DEAD twin, i.e. tombstoned long ago) → verdict `0 rows`; win → verdict `1 row`.
- **UPDATE**: DELETE step; on `1 row` also do the INSERT step for the new version (headroom slot
  reserved at pump). On `0 rows` the reserved slot burns with the never-visible sentinel (same
  mechanism as a rejected insert; slot leak is bounded by 0-row-update rate, reclaimed by VACUUM).

Visibility correctness of device-side tombstoning: `deleted_by` stamps are monotonically
published u64 stores; a reader at snapshot < my seq evaluates `deleted_by > snapshot` → row still
visible — the same argument that admits SV6 created_by stamping concurrent with readers. The
stamp lands BEFORE the verdict DtoH (launch-completion order), and outcome/visibility publication
is cut-gated as ever.

**deleted_by sidecar availability (the one real allocation problem):** `shard_deleted_by_memory`
is ON-DEMAND — allocated at a shard's first delete. The kernel cannot allocate. Rule: the leader
materializes the sidecar (zeroed alloc, one HtoD-free memset) for any target shard lacking it
BEFORE the launch, driven by a cheap host check over the wave's op set. First-delete-per-shard
pays a one-time alloc on the leader (~amortized nil); every later wave is pure kernel.
**[DECIDE]** alternatively pre-materialize at lane-table admission (48M rows = 384MB/shard of
always-allocated sidecar) — simpler leader, fatter memory. Recommend on-demand + leader check.

### 3.5 Settle
`LaneSettle` items carry op kind; settle reads the slot verdicts: insert winners Ok(1),
update/delete Ok(verdict rows), kernel-rejected inserts Err(23505) behind the marker gate
(mega-fuse), catalog-drift/ledger rejects Err at pump (unchanged). Async-commit (`Off`) applies
to update/delete identically (ack at applied cut; bounded-loss contract covers "acked delete
undone by power failure" the same way it covers inserts — the WHOLE suffix vanishes together;
recovery replays the ordered durable prefix so no torn read-your-writes state).

## 4. SI semantics summary (must match the classic path + SQL spec)

- DELETE of a never-existing key → Ok(0). DELETE of a dead key → Ok(0). No error.
- DELETE/UPDATE where the key's version changed after my snapshot → serialization error (ledger).
- UPDATE post-image violating a unique slot other than the PK: v1 shape has PK as the only unique
  slot and PK-updates are excluded → structurally impossible; assert in route prepare (if a
  second unique i32 column exists, the UPDATE shape is only covered when it doesn't assign that
  column — enforced at route prepare **[DECIDE]** or v1-exclude such tables entirely).
- INSERT after DELETE of the same key (different intents, in order): delete tombstones; insert
  probe hits the DEAD index twin → mega-fuse dead-twin path. v1: that path is
  decline→cache-drop→rebuild — correct but O(shard) per occurrence. Under a delete-heavy OLTP mix
  this becomes COMMON: v1.1 should teach the kernel **index-entry replacement** (probe hit → load
  `deleted_by[row]` → nonzero ⟹ dead ⟹ CAS the index entry old→new packed) so reinsert is O(1).
  Gate v1 with a mixed-workload bench arm to size the cliff first (§8).

## 5. W5b binary records + replay

New op codes in `wal_binary.rs` (tag 0xFF, version bump):
- `OP_DELETE_BY_KEY { table, pk_col, pk_value }`
- `OP_UPDATE_BY_KEY { table, pk_col, pk_value, new_row_id, new_row_image }`

Replay (`apply_binary_wal_entry` arms): re-resolve the pk against the HOST store at replay time
(visible version at replay-now), tombstone / tombstone+install. Determinism: replay runs in
global seq order; all ops on a key were lane-serialized in that same order live; ops on different
keys commute — so replay's resolve reproduces the live verdict, including 0-row outcomes. This is
the property that lets update/delete keep optimistic publish with NO abort markers. The insert
marker mechanism (mega-fuse) is orthogonal and unchanged.

Checkpoint/truncation: records ride the existing lane frames; `checkpoint_intent_lanes` is
op-agnostic (it snapshots applied state). Archive/PITR stays refused in lanes mode (unchanged).

## 6. Read path, elision, VACUUM

- Mask-VM visibility already evaluates sparse deleted_by (SV0–SV6) — no reader changes.
- Elision: deletes/updates on elided tables keep the shards authoritative; rehydration
  (`visible_relational_rows`) already understands tombstones. The rehydrate fallback arm in
  `lane_apply_merged` must grow update/delete equivalents (tombstone via host store) — rare path,
  same panic-on-invariant discipline.
- VACUUM #5 churn triggers now actually fire on lane tables (deletes create the dead-row churn it
  was built for). The deferred-tail auto-trigger must be verified against lane-applied tombstones
  (counter + regression), and vacuum's drain interaction with lanes uses the existing
  drain/commit-lock/recheck loop (post-Lorentz fix).

## 7. The escape hatch: QUIESCE for non-covered writes

Reuse the resize-barrier machinery as a general **lane quiesce**: divert submits to the hold
queue → drain every lane to settlement → run the classic statement(s) under the commit lock
(serial WAL append — the serial+lanes merge replay already interleaves correctly by seq) →
resume routing. Exposed as `Engine::execute_dml_quiesced` (and the DDL twin), replacing the flat
refusal. Bounded cost = one barrier (measured 1.7–260ms depending on population). Non-covered
writes become CORRECT-but-slow instead of impossible — the right production posture.
**[DECIDE]**: v1 includes this, or stays refuse-only while UPDATE/DELETE intents land first.

## 8. Gates, counters, benchmarks

- Counters: per-op wave counts, verdict histograms (1-row/0-row/dead-twin/decline), sidecar
  materializations, quiesce count+duration. Prove-the-path-fired discipline throughout.
- Differentials: lane UPDATE/DELETE vs classic path on a non-activated twin engine (GPU==CPU==
  spec), including 0-row cases, snapshot-conflict cases, delete→reinsert, update-then-read at
  old/new snapshots.
- Sabotage: break deleted_by CAS (skip stamp) → visibility test FAILS; break replay re-resolve →
  recovery parity FAILS; break ledger delete rule → lost-update test FAILS.
- Recovery: durable WAL replay parity incl. interleaved insert/update/delete + crash-mid-wave
  orphan repair; reopen-continues with mixed ops.
- Bench: `intent_fast_path_bench` mixed arm `GPU_DB_BENCH_MIX=I:U:D` (e.g. 70:20:10 core-banking
  shape) at the champion config + 512-client floor; report rows-affected-weighted TPS with the
  standard latency pairing. Baseline to beat: insert-only {1.50,1.41,1.50}M.
- Full suites: engine default+fua arms, GPU intent suites (serial + lanes=2/6), wal crate,
  clippy; TMPDIR hygiene.

## 9. Open decisions for the user (besides inline [DECIDE]s)

1. **Slice order.** Proposed: U1 delete-only end-to-end (op model, ledger rule, kernel delete
   arm, W5b delete record, replay, gates) → U2 update (rides U1 + insert machinery) → U3 quiesce
   escape hatch → U4 dead-twin index replacement (sized by the U1/U2 mixed bench) → U5 default
   flip. Delete-first because it exercises every new mechanism with the smallest surface.
2. **PK-update v2**: cross-lane two-phase (tombstone in lane A gated on insert-claim in lane B)
   vs quiesce-only forever. Defer until a workload demands it.
3. **Whether Tier 1 waits for the mega-fuse to MERGE** or develops against its WIP contract in a
   worktree. Contact surface is severe (same kernel, same ApplySlot/settle code) — recommend:
   wait for merge, then U1.

## 10. Contact surfaces (for multi-agent coordination)

`crates/execution/src/lib.rs` (fused kernel + submit), `engine_dml_concurrent.rs` (pump, ledger
pass, settle), `engine_intent_lanes.rs` (LaneIntent/ApplyRequest/ApplySlot), `engine_dml_intent.rs`
(routes/API), `wal_binary.rs` (+ its `engine_commit.rs` apply arms), `engine_residency.rs`
(sidecar materialization, rehydrate fallback), `engine_lifecycle.rs` (replay arms). All overlap
the mega-fuse slice except wal_binary/lifecycle.
