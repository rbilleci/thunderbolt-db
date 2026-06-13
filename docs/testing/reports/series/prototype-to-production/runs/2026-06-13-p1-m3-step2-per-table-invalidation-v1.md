# P1-M3 step 2 (slice A) — Per-Table Residency Invalidation

Status: closed (independently audited; no blocker)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1 / §2.2 / Phase 2
"fix residency granularity"
Design: `docs/architecture/14-engine-snapshot-integration-design.md` (step 2)
Branch: `phase0-m1-engine-facade`

## Why this slice

Doc 14 step 2 is "make per-table residency a `SnapshotCell<Arc<owner>>`,
publish-on-commit, which also fixes the global stop-the-world invalidation." Mapping
the engine showed step 2 splits cleanly into two slices:

- **Slice A (this report): per-table invalidation** — fix the stop-the-world bug
  (`invalidate_relational_residency` at the old `engine/lib.rs:9017` invalidated
  *every* resident table on *any* write). Small, contained, correctness-sensitive, and
  immediately valuable for mixed read/write workloads. The "which table(s) did this
  commit mutate" logic it builds is reused by slice B.
- **Slice B (next): `SnapshotCell<Arc<owner>>` residency + `&self` reads** — the
  concurrency-safety machinery, naturally coupled with step 3 (the `&self` read flip),
  since a published generation only matters once reads are concurrent.

Slice A was chosen first because the `SnapshotCell` migration touches ~40 read sites
and only pays off with step 3, whereas per-table invalidation is a self-contained
correctness win that de-risks slice B.

## What changed (`crates/engine/src/lib.rs`)

The commit path now invalidates only the tables a committed batch actually mutated,
derived from the committed log entries — with a **conservative global fallback**:

- `residency_invalidation_scope(entries) -> Option<BTreeSet<String>>`: returns
  `Some(tables)` only when every command in the batch is one whose mutated table is
  unambiguous — `Insert`/`Update`/`Delete` (single table), `TruncateTable`,
  `DropTable` — plus `CreateTable` (a new table has no prior residency, contributes
  nothing). **Any** other command, or a payload that fails to decode/parse, returns
  `None`. Returning `None` falls back to the previous global invalidation.
- The design rule: **over-invalidation is safe (a perf cost); under-invalidation
  would serve wrong rows.** The scope therefore narrows only when certain and falls
  back to global for everything else — it can never leave a resident table stale.
- `invalidate_relational_residency_table(table, …)`: the per-table unit (snapshot +
  device memory + partitions). The refactored global `invalidate_relational_residency`
  is now "call the per-table helper for every resident table," reproducing the original
  two-loop fallback (the helper's device-memory/partition removals are unconditional,
  so in degenerate cache states it may clear a stray cross-map entry the old form left
  — strictly-safe extra cleanup, never stale).
- Both commit functions (`commit_mutation_at`, `commit_mutation_at_with_current_apply`)
  dispatch through `invalidate_relational_residency_for_commit`.

No read-path or `&mut self`→`&self` changes (that is slice B / step 3). The
`SnapshotCell` substrate is not yet wired (slice B).

## Tests (4 new, all passing without a GPU)

- `mutation_invalidates_only_the_mutated_table_residency` — make tables `a` and `b`
  resident; `INSERT INTO a`; assert `a` invalidated and **`b` still valid** (the core
  fix — previously `b` was globally invalidated).
- `create_table_does_not_invalidate_existing_residency` — `CREATE TABLE c` leaves an
  unrelated resident table `a` valid.
- `unscoped_ddl_conservatively_invalidates_unrelated_residency` — `ALTER TABLE a ADD
  COLUMN …` (not in the precise whitelist) conservatively invalidates the unrelated
  table `b` (the fallback).
- `residency_invalidation_scope_narrows_dml_and_falls_back_on_unknown` — unit test of
  the scope function: single-table DML → `Some({t})`; cross-table DML → union;
  `CREATE TABLE` → `Some({})`; an unscoped command in the batch → `None`; an
  unparseable payload → `None`.

## Validation

```text
cargo test -p gpu_db_engine            → 370 passed; 1 failed; 39 ignored
cargo fmt -p gpu_db_engine             → clean
cargo clippy -p gpu_db_engine          → 0 new warnings on the changed code
cargo build --workspace                → ok
```

The one failure, `p8_default_resident_route_executes_accepted_shapes`, is a
**pre-existing real-GPU failure on this host** (a `last_execution_kernel_event_elapsed_us`
vs metric assertion). Confirmed unrelated to this change by stashing the change and
re-running on a clean tree — it fails identically there. It is not introduced by, and
not in scope for, this slice. (Flagged for separate triage: a non-`#[ignore]` test
that depends on GPU kernel-event timing and is red on RTX PRO 6000 / driver
595.71.05.)

## Independent adversarial audit

A reviewer was tasked to **refute** the paramount claim — that the narrowing can never
under-invalidate (leave a resident table stale → wrong rows). Result: **no blocker, no
major; claim upheld.** The decisive structural facts it verified:

- The engine is **RESTRICT-only**: `RelationalForeignKey` has no `on_delete`/`on_update`;
  `apply_delete`/`apply_update`/`apply_insert` validate FK/unique/check **read-only and
  error** on violation — they never cascade writes to another table. There are **no
  triggers, no generated columns**.
- **No multi-table DML**: `Insert`/`Update`/`Delete` each carry one `.table`;
  `INSERT…SELECT` fails to parse (→ `None`/global); `DROP/TRUNCATE … CASCADE` and
  multi-table `TRUNCATE` are rejected by the parser.
- `CreateTable` **errors** on an existing name (no `IF NOT EXISTS`, no `AS SELECT`), so
  contributing the empty set is safe; a DROP-then-recreate clears residency via the
  `DropTable` scope entry.
- `residency_invalidation_scope` and the apply path call the **same `parse_command`** on
  the same payload, so they always agree on the mutated table.
- The two commit functions are the **only** invalidation call sites — unchanged from the
  pre-change code, so completeness is not reduced.

Only finding: a **NIT** — the global fallback is strictly-equivalent-or-safer, not
byte-"exactly" the original (unconditional cross-map cleanup in degenerate states).
Fixed by softening the code comment; no behavior change needed.

## Benchmark note (§5.7)

No benchmark re-run: the noise-controlled baseline is read-only (no writes during the
measured phase), so per-table invalidation does not change those numbers — its value
is for mixed read/write workloads (a write to one table no longer evicts others'
residency). Mechanistically isolated from the measured read path.

## Scope boundaries / next

- Slice B: `SnapshotCell<Arc<owner>>` residency + publish-on-commit + the `&self`
  read flip (step 3) — the concurrency unit; reuses this slice's mutated-table logic
  for publish-per-table.
- Schema-changing DDL (ADD/DROP COLUMN, etc.) is still globally invalidated
  (conservative). Extending the precise whitelist to single-table DDL is a future,
  optional narrowing.
