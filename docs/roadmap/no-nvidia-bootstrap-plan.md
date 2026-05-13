# No-NVIDIA Bootstrap Plan

This plan adapts implementation sequencing for environments without an NVIDIA GPU while preserving GPU-first architecture decisions.

## Goal

Start building immediately with CPU and simulation paths, so adding real CUDA execution later does not require architectural rewrites.

## Guiding principle

- **GPU-first architecture, CPU-first validation path**
- Implement interfaces, invariants, and data/commit boundaries now.
- Delay hardware-dependent execution and tuning until GPU access is available.

## Phase 0: GPU-ready core (start now)

### 0.1 Repository and module skeleton
Create clear subsystem boundaries:
- `protocol`
- `planner`
- `execution`
- `txn`
- `storage`
- `wal`
- `replication`
- `observability`

### 0.2 Protocol and session baseline
- PostgreSQL wire-protocol startup/auth skeleton
- simple query path
- session lifecycle and state machine baseline

### 0.3 Durability and recovery baseline
- WAL append/flush path
- checkpoint metadata
- crash recovery bootstrap flow

### 0.4 Transaction and MVCC baseline
- transaction lifecycle
- Read Committed semantics first
- visibility rules encoded as testable invariants

### 0.5 Replication-shaped commit path
- Implement `LogReplicator` abstraction now
- Implement `LocalReplicator` now (single-node durable commit)
- Keep leader/follower role model in API shape

### 0.6 Execution abstraction
- `Operator` interface with device annotation
- `CpuBackend` as reference semantics
- `MockGpuBackend` for deterministic simulation tests

### 0.7 CI and quality gates
- invariant tests (WAL-before-visibility, crash safety)
- deterministic replay tests
- parser/protocol fuzzing
- PR template enforcement of GPU-first checklist

## Phase 0.5: GPU simulation mode

### 0.5.1 Deterministic batching
- dual-trigger batching (count/time)
- ordered batch metadata persisted for replay validation

### 0.5.2 Simulated device routing
- planner emits device-targeted plans
- GPU-targeted operations route to `MockGpuBackend`
- explicit fallback reasons emitted as metrics

### 0.5.3 Placeholder telemetry
Track now so dashboards and alert contracts stabilize early:
- `gpu_fallback_rate`
- `batch_wait_ms`
- simulated `h2d_bytes`, `d2h_bytes`
- commit/applied lag metrics

## Phase 1: replication foundation before GPU hardware

### 1.1 Raft scaffolding
- `RaftReplicator` interface and state skeleton
- role gates (leader write acceptance)
- commit-index and applied-index counters

### 1.2 Snapshot hooks
- export/import snapshot interfaces
- compaction boundary APIs (even if no network transfer yet)

### 1.3 Failover/readiness behavior
- readiness semantics around role and recovery state
- write rejection behavior on non-leader roles

## Deferred until NVIDIA hardware is available

- real CUDA backend (`CudaBackend`) integration
- CUDA kernel implementation and correctness/perf validation
- GPUDirect Storage behavior
- NCCL/multi-GPU transport behavior
- occupancy/register/tuning workflows

## Hardware-onboarding checklist (future trigger)

When first NVIDIA environment becomes available:

1. Add CUDA build targets and CI runner with GPU.
2. Implement `CudaBackend` behind existing `GpuBackend` trait.
3. Port first operator subset to real kernels (scan/filter/point lookup).
4. Run CPU vs GPU parity harness in CI.
5. Enable hardware performance dashboards and regression thresholds.

## First two-week sprint (actionable)

1. Scaffold crate/modules and interface packages.
2. Implement `LogReplicator` + `LocalReplicator`.
3. Implement WAL append/flush and basic recovery sequence.
4. Implement CPU execution for minimal SQL subset.
5. Implement deterministic batch scheduler (CPU-only).
6. Add invariant and replay test suites.

## Queue additions for autonomous loop pickup

### Q1. Engine truth surface for snapshots, fallback, and replication health
Priority: highest
Status: completed on 2026-04-25; `Engine::status_snapshot()` is now the engine truth surface. Keep docs/tests aligned if the surface evolves.

Goal:
- Turn the recent MVCC/observability/replication helper work into a single trustworthy engine-level status surface that answers what state the engine is in and whether it is healthy/correct.

Acceptance criteria:
1. Expose one engine-level status/snapshot surface (API, struct, command, or metrics bundle) that includes at minimum:
   - latest in-memory snapshot identity/frontier
   - parity fallback rollups and active fallback reasons
   - replication lag and watermark state
   - blocker/readiness flags already present in telemetry where applicable
2. Define and enforce a small set of invariants on that surface (for example monotonic watermark movement, snapshot/frontier consistency, no impossible lag values).
3. Add focused tests that prove the surface is stable and semantically correct under normal progression plus at least one degraded/fallback case.
4. Document how an operator or developer should answer: “what snapshot served this?”, “why did this route to fallback?”, and “how far behind is replication?”
5. Update roadmap/docs as needed so this is treated as the truth surface for subsequent loop work.

Notes:
- Prefer one crisp trustworthy status surface over many loosely-related helper accessors.
- This is the immediate leverage point for the helper commits already landed.

### Q2. Vertical slice: execution over MVCC storage
Priority: highest
Status: completed for the bootstrap slice on 2026-04-25 and closed for the no-GPU phase on 2026-05-04 after bootstrap closeout review + full validation gate; current supported shape is full scan/key lookup + snapshot visibility + filter/order/project/limit with explicit `GpuMvccReadParityGap` fallback tracking. Reopen only if CUDA onboarding exposes a real contract gap.

Goal:
- Convert the new in-memory MVCC tuple store and reusable vec operator groundwork into a narrow but real end-to-end execution slice.

Acceptance criteria:
1. Implement a meaningful read path that runs against the MVCC store through execution abstractions, covering at least:
   - scan or key lookup
   - visibility filtering against a snapshot
   - projection/filtering through the execution layer
2. Keep device strategy explicit per guardrails:
   - CPU reference semantics implemented
   - GPU path declared or an explicit tracked fallback reason recorded
3. Add end-to-end tests that demonstrate the slice works through engine-facing entry points rather than subsystem-only unit tests.
4. Add at least one small benchmark or deterministic workload fixture so future loop runs can measure progress on this vertical slice.
5. Document the exact supported query shape and the next obvious extension boundary.

Notes:
- Favor a complete thin slice over broad unfinished operator scaffolding.
- This should make the engine visibly more capable, not just more internally prepared.
- Current bootstrap truth: the engine-facing MVCC slice now supports full scans + single-key lookups + key-batch fan-in lookups + explicit multi-source `Concat` / `ConcatDistinct` / `IntersectDistinct` / `IntersectAll` / `ExceptDistinct` / `ExceptAll` / `SymmetricDifferenceDistinct` / `SymmetricDifferenceAll` composition + generic `FollowValueChain { keys, plan, provenance }` linear nested-join expansion + `FollowValueChainBranches { keys, plans, fan_in, provenance }` plus `FollowValueChainLabeledBranches { keys, branches, fan_in, provenance }` branch expansion (including the existing named value→key, value→key→prefix, value→key→value→key, and deeper fan-out/terminal helper families as stable aliases) with snapshot visibility, prefix/value/composite/range filters, source-aware key/value plus branch-label filters for join-adjacent rows, explicit multi-frame provenance filters/projection/order controls (`Seed`, `TerminalInput`, `ValueHop(n)`), reusable provenance-frame bundles (`SeedThroughTerminalInput`, `FullPath`) across bundle-aware key/value exact-membership, counted bundle-membership thresholds, exact ordered path equality, ordered contiguous subpath matching, ordered repeated-subpath counting, exact-distance pair matching, whole-bundle prefix/suffix matching, anchored bundle-slice matching, exact/ranged first/last/nth occurrence matching, exact/ranged same-subpath ordinal plus adjacent and first/last-to-ordinal occurrence-distance helpers, exact/ranged first/last-to-ordinal same-subpath offset helpers, exact/ranged ordinal-pair same-subpath offset helpers, exact/ranged ordinal mixed-subpath occurrence-distance, exact/ranged first/last-to-ordinal mixed-subpath occurrence-distance helpers, exact/ranged nth-offset matching, exact/ranged first/last-to-ordinal mixed-subpath offset helpers, first/last exact/ranged mixed-occurrence offsets, first/last mixed-subpath occurrence-distance helpers, reusable occurrence-offset and occurrence-distance projection/order helpers, bundle-relative positional segment equality, exact whole-bundle cardinality checks, prefix filters, key/value path ordering, and summary projection helpers, mixed join-side projection controls (target-key+seed-value, source-key+target-value, source-value-only, branch-label+target-value, target-key+provenance-value, target-key+provenance-summary, target-key+bundle-summary), selectable single-frame source provenance on generic nested-join rows, and post-order limit under an explicit `GpuMvccReadParityGap` fallback contract.
- Deterministic workload fixtures now cover point-lookup/history replay, full-scan/simple-filter replay, and source-composition replay (`tests/fixtures/mvcc-read-workload.txt`, `tests/fixtures/mvcc-full-scan-workload.txt`, `tests/fixtures/mvcc-source-composition-workload.txt`).
- The supported first-CUDA subset now already has explicit engine-facing backend-swap parity regressions for both deterministic lookup and full-scan fixture shapes, so `CudaBackend` can be judged against CPU truth without changing the result contract.
- Next obvious extension boundary: the mixed-subpath offset/distance projection-order surface is now broad enough for bootstrap purposes; the remaining Q2 loop should prioritize semantic closeout, truth-surface consolidation, and proof that no interface rewrite is needed before `CudaBackend` lands.

### Q3. Replication semantics hardening under stress
Priority: highest
Status: completed on 2026-04-30 after closeout review + full validation gate. The replication semantics surface is now considered locked for the no-GPU bootstrap phase; only reopen Q3 if a genuinely new semantic category or contradiction is discovered.

Goal:
- Move from replication introspection to replication behavior that is predictable and trustworthy under skew, lag, replay, and resume conditions.

Closeout focus:
1. Prefer **gap-closing** work over frontier-expanding work.
2. Only add a new regression if it closes a clearly named semantic hole, resolves a contradiction, or is required to satisfy exit criteria.
3. Stop generating deeper combined-stack permutations unless a concrete unproven guarantee still depends on them.
4. Favor consolidation work now: tighten invariants, collapse duplicate coverage, document the semantic envelope, and prove the status/recovery surfaces match runtime truth.
5. If a candidate loop does not materially increase confidence that Q3 can be marked complete, do not do it.

Exit criteria for Q3:
1. **Core follower semantics locked**
   - ordering, rejection stability, commit/apply progression, resume/restart projection, and snapshot-install behavior are all covered by explicit tests.
2. **Status truth surfaces locked**
   - `status_snapshot()`, `recovery_state()`, `recovery_progress_gap()`, and `resume_as_follower(...)` are mutually consistent and reject impossible live-vs-durable states.
3. **Advanced snapshot identity semantics locked**
   - accepted advanced-frontier snapshots preserve exact durable identity,
   - same-frontier refreshes only change durable identity,
   - stale or incoherent installs remain no-ops,
   - newer-leader rejection/repair paths preserve or retire speculative tail correctly.
4. **Representative combined-stack coverage complete**
   - at least one explicit regression exists for each meaningful combined-stack family:
     - refresh -> rejection
     - refresh -> repair
     - repair-phase advanced replacement
     - post-rejection re-repair
     - role/term handoff before retirement
     - stale-install inertness during repair/commit/apply
   - additional permutations are out of scope unless they expose a distinct semantic rule.
5. **Documentation closeout complete**
   - the replication interface docs state the guarantees and non-guarantees clearly enough that a new contributor can tell what is intentionally supported.
6. **Validation gate green**
   - full `cargo fmt --all`
   - full `cargo clippy --all-targets --all-features -- -D warnings`
   - full `cargo test --all --all-features`

Definition of done:
- Q3 is complete when the team can name no remaining *semantic category* that lacks explicit proof. Missing permutations alone are not enough to keep Q3 open.

Clear exit strategy:
1. At the start of each loop, name the specific remaining semantic category being addressed.
2. If no such category can be named, do not invent a longer scenario; instead run the validation gate and perform a closeout review.
3. During closeout review, check the Q3 exit criteria one by one and record either:
   - satisfied,
   - blocked by a real gap, or
   - not yet proven by an explicit regression/doc invariant.
4. If every exit criterion is satisfied and the full validation gate passes, mark Q3 complete immediately and stop the Q3 loop.
5. If any item fails, the next loop must target that exact failed item and nothing broader.

Operational stop rule:
- Stop the Q3 loop when both conditions are true:
  1. there is no unproven semantic category left to name, and
  2. fmt + clippy + full tests are green.
- Do not continue looping just because another permutation could be written.

Q3 closeout checklist:
- [x] Core follower semantics locked
- [x] Status truth surfaces locked
- [x] Advanced snapshot identity semantics locked
- [x] Representative combined-stack coverage complete
- [x] Documentation closeout complete
- [x] Full validation gate green
- [x] No remaining named semantic gap

Closeout review recorded on 2026-04-30:
- satisfied: core follower semantics locked
- satisfied: status truth surfaces locked
- satisfied: advanced snapshot identity semantics locked
- satisfied: representative combined-stack coverage complete
- satisfied: documentation closeout complete
- satisfied: validation gate green (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`)
- satisfied: no remaining named semantic gap

Acceptance criteria:
1. Define explicit semantics/tests for ordering, apply progression, watermark movement, and recovery/resume behavior.
2. Add targeted tests for at least:
   - lagging follower/apply delay
   - resume after interruption or restart
   - stale or out-of-order entry handling
3. Ensure the exposed replication metrics/status surfaces remain consistent with actual state transitions in the hardening tests.
4. Document what guarantees the current engine does and does not make about replication correctness/readiness.
5. Reconcile any helper APIs that are too weak/ambiguous for these guarantees.
6. Mark Q3 complete once the exit criteria above are satisfied; do not leave it open for unbounded permutation growth.

Notes:
- Observability without semantics is not enough; this queue item is about trust.
- The burden of proof is now on finding a missing semantic category, not on inventing another longer scenario chain.

### Q4. Golden-wire `psql` compatibility suite
Priority: high
Status: completed on 2026-04-30; CI now installs `psql`, boots the repo-local compatibility endpoint, and runs the golden suite end-to-end.

Goal:
- Add a real-client compatibility suite that exercises the engine through the standard `psql`/libpq path rather than custom protocol fixtures only.

Acceptance criteria:
1. Add a scripted golden test harness that boots the engine and runs `psql` against it using standard libpq environment variables/flags.
2. Cover at least these flows end-to-end:
   - startup/auth/connect success
   - simple query (`SELECT 1` or closest supported bootstrap equivalent)
   - session reset/setup probes commonly emitted by `psql`/libpq
   - transaction begin/commit/rollback flow
   - one prepared/extended-query flow if supported, otherwise an explicit expected-failure golden case
   - one deterministic error-path golden case with asserted SQLSTATE/message contract if available
3. Store reproducible golden artifacts/expected outputs under a dedicated test directory.
4. Wire the suite into the standard CI/test entrypoint or document the exact temporary gate if CI wiring must land in a follow-up commit.
5. Document how to run/update the suite locally.

Notes:
- Prefer stable assertions over brittle byte-for-byte transcript checks where timestamps/noise vary.
- Use real `psql`/libpq behavior as the oracle for connection lifecycle compatibility.

### Q5. CI compatibility scorecard
Priority: high
Status: completed on 2026-04-30; CI now merges Rust test output plus real-client `psql` golden results into one published compatibility scorecard artifact.

Goal:
- Produce a hard compatibility scorecard in CI so protocol/SQL compatibility progress is measured, not inferred.

Acceptance criteria:
1. Define a machine-readable scorecard format (for example JSON or Markdown generated from test results).
2. Report, at minimum:
   - protocol/client flow coverage bucket counts
   - SQL/parser feature bucket counts
   - pass/fail totals
   - top failing compatibility categories
   - trend hook placeholder against previous baseline if full trend wiring is not yet implemented
3. Generate the scorecard in CI from real test outputs, not hand-written status.
4. Publish the scorecard as a CI artifact or checked-in generated example fixture for local inspection.
5. Document how future compatibility tests should register themselves with the scorecard.

Notes:
- Keep the first version simple and trustworthy.
- Prefer explicit bucket definitions over a fake single percentage.

Queue reconciliation note (2026-05-04):
- Q1 through Q5 are now complete.
- There is no remaining queued no-GPU implementation item to widen autonomously.
- Until an NVIDIA-capable environment exists or a concrete contradiction reopens a named semantic gap, autonomous loops should treat the queue as drained and stop after validation/doc reconciliation rather than inventing new bootstrap surface area.

## Post-CUDA PostgreSQL-compatible product loop

Status: active as of 2026-05-11. The no-GPU bootstrap queue and CUDA completion gates are closed for their defined scope. The autonomous loop now moves from internal MVCC/CUDA completion to the larger product goal: a PostgreSQL-compatible GPU-backed database engine.

Loop rules:
- Work the milestones below in priority order.
- Start each run by naming the active milestone and the exact exit criterion being advanced.
- Prefer real client-facing behavior over isolated parser or kernel breadth.
- Do not widen CUDA internals unless it directly supports a PostgreSQL-facing milestone or a measured benchmark target.
- Update `docs/compatibility/matrix.md`, scorecard docs, and roadmap wording whenever implementation truth changes.
- If a run cannot name a product milestone gap, run validation and stop instead of inventing work.

Validation gate for product-loop changes:
- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all --all-features`
- psql golden/compatibility scorecard generation when protocol or SQL behavior changes
- `scripts/run_cuda_parity.sh` when a change touches CUDA runtime, CUDA MVCC routing, or GPU fallback accounting

### P1. Relational SQL foundation

Goal:
- Move beyond bootstrap KV commands and hard-coded compatibility responses into a small relational SQL surface that can create, mutate, and query named tables through the engine.

Initial target:
- `CREATE TABLE` for a minimal typed table shape.
- `INSERT` into that table.
- `SELECT <columns> FROM <table> [WHERE ...] [ORDER BY ...] [LIMIT ...]` for a deliberately narrow predicate/order subset.
- Deterministic error reporting for unsupported relational syntax.

Exit criteria:
1. SQL parser/planner represents the supported relational forms as structured plans rather than string-matched server fixtures. First slice landed on 2026-05-11 for `CREATE TABLE`, `INSERT`, and narrow `SELECT`.
2. Supported relational reads execute through engine-facing storage/execution APIs, not only the compatibility stub. First slice landed on 2026-05-11 by lowering engine relational reads through `MvccReadQuery` with CPU reference execution and explicit GPU parity fallback.
3. At least one psql golden scenario creates a table, inserts rows, selects rows, and asserts stable output. Scenario `03_relational_create_insert_select.sql` was added on 2026-05-11; local execution still requires `psql` in the run shell.
4. Compatibility scorecard has explicit relational SQL buckets with passing/failing counts. First classifier bucket `sql.relational_foundation` landed on 2026-05-11.
5. Docs state the exact supported relational SQL subset and the next unsupported syntax boundary. README and compatibility matrix first-slice wording landed on 2026-05-11; P4 range bridge work later widened the narrow `WHERE` subset from equality-only to single-column literal comparisons (`=`, `<`, `<=`, `>`, `>=`), narrow `AND` conjunctions, and top-level `OR` groups over those literal predicates while keeping joins, broader expressions, nested boolean trees, NULLs, updates/deletes, constraints, and broad coercion unsupported.

### P2. Catalog, schema, and type spine

Goal:
- Add enough catalog/schema/type machinery for relational SQL and client introspection to have stable identities instead of ad hoc row labels.

Initial target:
- Table and column descriptors with stable ids/OIDs or an explicit OID allocation strategy.
- Basic scalar type registry for the first SQL subset.
- Minimal namespace handling, with `public` as the first supported schema.
- First `pg_catalog` compatibility views/functions required by common psql/libpq startup and introspection probes.

Exit criteria:
1. Created tables and columns are stored in a catalog structure used by planning/execution. First slice landed on 2026-05-11: created tables enter an engine-owned `public` schema catalog with stable user relation OIDs starting at `16384`, and columns carry stable ids, `attnum`, and table OID references used by execution-side insert validation, row decoding, and select projection. Follow-up on 2026-05-11 added catalog-bound select projection/filter/order planning before row execution.
2. Type metadata is used for row descriptions and basic coercion/validation in supported statements. First slice landed on 2026-05-11: `SqlType` exposes PostgreSQL-compatible `int4`/`text` OIDs, type widths, and catalog names; relational catalog columns store those values and the wire compatibility endpoint uses the same type metadata for row descriptions. Follow-up on 2026-05-11 exposed the supported `int4`/`text` registry through metadata-backed `pg_catalog.pg_type` introspection.
3. psql can inspect the first supported tables without relying on hard-coded fake responses for those objects. First compatibility-endpoint slice landed on 2026-05-11: exact `pg_catalog.pg_class` and `pg_catalog.pg_attribute` query shapes expose supported session tables and column type OIDs dynamically, with psql golden artifacts in `04_catalog_introspection.sql`. Follow-up on 2026-05-11 added stable relation OIDs plus column `attnum`/`atttypid`/`attlen` detail rows to the same metadata-backed catalog helpers, then added `pg_catalog.pg_type` rows for the supported SQL type registry. Follow-up on 2026-05-12 added real PostgreSQL 16 `psql \dt` and `\d <table>` coverage plus first-slice `information_schema.tables` / `information_schema.columns` coverage backed by session catalog metadata. Follow-up on 2026-05-12 added real `psql \dn` coverage and `information_schema.schemata` for the supported `public` namespace. Follow-up on 2026-05-12 added real `psql \dT pg_catalog.int4` / `\dT pg_catalog.text` coverage backed by the supported type registry. Follow-up on 2026-05-12 added real `psql \d+ <table>` verbose table display for supported plain session tables, backed by the same relation/column metadata plus explicit plain-table storage/access-method values. Follow-up on 2026-05-12 added schema-qualified `psql \d public.<table>` / `\d+ public.<table>` relation lookup plus `\dt public.*` and simple-prefix `\dt public.<prefix>*` table listing for supported `public` tables. Follow-up on 2026-05-12 added all-supported-public-table `information_schema.columns` enumeration backed by session catalog metadata plus supported column nullability/default/type-identity fields. Follow-up on 2026-05-12 added real PostgreSQL 16 `psql \dt+ <table>` verbose table-listing coverage for supported session tables, with catalog-backed persistence/access-method fields and intentionally blank size/description fields until the engine exposes PostgreSQL heap-size/description metadata. Follow-up on 2026-05-12 added real PostgreSQL 16 `psql \dp <table>` access-privilege listing plus `\z public.<pattern>` schema-qualified pattern coverage for supported plain `public` session tables, returning empty privilege/policy fields from catalog metadata until ACL and row-policy features exist. Follow-up on 2026-05-12 added richer `information_schema.tables` metadata for supported public tables, including table catalog, table type, insertability, typed-table flags, and null typed-table-only fields backed by the session catalog. Follow-up on 2026-05-12 added real PostgreSQL 16 `psql \di` / `\di public.*` empty index-listing coverage for the current no-user-visible-SQL-index subset. Follow-up on 2026-05-12 added `pg_catalog.pg_tables` discovery, joined `pg_catalog.pg_class` / `pg_catalog.pg_namespace` relation metadata, and joined `pg_catalog.pg_attribute` / `pg_catalog.pg_class` / `pg_catalog.pg_namespace` column metadata for supported `public` session tables backed by the same session catalog metadata. Follow-up on 2026-05-12 added a common `information_schema.columns` table-name `IN (...)` subset query backed by session catalog metadata. Follow-up on 2026-05-12 added a per-table `information_schema.columns` detail projection with column names, display types, nullability, and empty defaults for supported table columns. Follow-up on 2026-05-12 added a common `information_schema.tables` table-name `IN (...)` subset query backed by session catalog metadata. Follow-up on 2026-05-12 added a common joined `pg_catalog.pg_class` / `pg_catalog.pg_namespace` table-name `IN (...)` relation subset query backed by session catalog metadata. Follow-up on 2026-05-12 added a JDBC/ORM-style `information_schema.columns` projection with catalog/schema/table/column identity plus supported nullability/default/type identity and `int4` numeric precision/radix/scale metadata, then table-filtered and table-name `IN (...)` coverage for that same extended projection. Follow-up on 2026-05-12 added real PostgreSQL 16 `psql \d public.*` schema-wildcard table-description coverage backed by namespace-only `public` relation lookup and the existing per-table describe metadata. Follow-up on 2026-05-12 added real PostgreSQL 16 `psql \d <prefix>*` / `\d+ <prefix>*` table-description coverage backed by pattern-aware relation lookup over supported `public` tables. Follow-up on 2026-05-12 added real PostgreSQL 16 `psql \dt <prefix>*` / `\dt+ <prefix>*` table-listing coverage backed by pattern-aware table catalog rows. Follow-up on 2026-05-12 added schema-qualified verbose `psql \dt+ public.<table>` / `\dt+ public.<prefix>*` table-listing coverage backed by the same catalog rows without requiring `pg_table_is_visible(...)` for explicitly scoped namespaces. Follow-up on 2026-05-12 added schema-qualified prefix `psql \d public.<prefix>*` / `\d+ public.<prefix>*` table-description coverage backed by the same relation-pattern lookup and per-table describe metadata for supported `public` tables. Follow-up on 2026-05-12 added unqualified verbose `psql \dt+` table-listing coverage for all supported visible `public` session tables, backed by the same catalog rows with intentionally blank size/description fields. Follow-up on 2026-05-12 added exact table-name filtering for the richer `information_schema.tables` projection backed by supported session table metadata. Follow-up on 2026-05-12 added empty `information_schema.table_constraints` and `information_schema.key_column_usage` discovery for supported `public` tables, documenting that SQL constraints are not implemented yet. Follow-up on 2026-05-13 added empty `pg_catalog.pg_constraint` and `pg_catalog.pg_attrdef` discovery for common client probes, documenting that SQL constraints and column defaults are not implemented yet. Follow-up on 2026-05-13 added empty `pg_catalog.pg_description` discovery for common table/column comment probes, documenting that comments are not implemented yet. Follow-up on 2026-05-13 added real PostgreSQL 16 `psql \dn+ public` verbose schema introspection for the supported `public` namespace, returning owner metadata plus empty ACL/description/publication rows for the current no-ACL/no-comment/no-publication subset. Advanced PostgreSQL catalog query coverage remains open.
4. Catalog state survives the same durability/recovery boundary as user data or has an explicit documented bootstrap limitation. Bootstrap proof landed on 2026-05-11: `Engine::recover_from_durable_wal(...)` replays the durable WAL prefix and a regression proves relational catalog descriptors plus table rows recover together with stable relation OIDs and column ordinals. Remaining limitation: cross-process persisted WAL/checkpoint recovery is still P5 production storage work.
5. Docs and scorecard identify supported vs unsupported catalog/introspection surfaces. First docs/scorecard classifier update landed on 2026-05-11 with `sql.catalog_schema_types`; dynamic first-slice `pg_catalog` user-table introspection now covers table names, relation OIDs, column names, column ordinals, type OIDs, type names, type lengths, `pg_catalog.pg_tables` rows, joined `pg_catalog.pg_class` / `pg_catalog.pg_namespace` relation metadata including table-name `IN (...)` subset filtering and namespace-only `public` relation lookup, joined `pg_catalog.pg_attribute` / `pg_catalog.pg_class` / `pg_catalog.pg_namespace` column metadata with supported formatted type names and nullable flags for supported `public` session tables, and empty `pg_catalog.pg_constraint` / `pg_catalog.pg_attrdef` / `pg_catalog.pg_description` rows for the current no-SQL-constraint/no-column-default/no-comment/no-object-description subset; real-client `psql \dt`, `\dt+`, `\dt+ <table>`, `\dt <prefix>*`, `\dt+ <prefix>*`, schema-qualified `\dt public.*`, and simple-prefix `\dt public.<prefix>*` table listings, real-client `psql \d <table>`, `\d+ <table>`, `\d <prefix>*`, `\d+ <prefix>*`, `\d public.*`, and schema-qualified `\d public.<prefix>*` / `\d+ public.<prefix>*` column display including schema-qualified `public.<table>` describes, real-client `psql \dp <table>` / `\z public.<pattern>` access-privilege display with empty ACL/policy fields for supported plain public tables, real-client `psql \di` empty index-list display, `\dA` bootstrap access-method display, `\dv` / `\dv+` empty view-list display, `\dC` empty cast-list display, `\dx` empty extension-list display, and `\db` bootstrap tablespace-list display for the current no-user-visible-SQL-index/no-SQL-view/no-user-defined-cast/no-extension/no-tablespace-mutation subset, real-client `psql \dn` public-schema display, real-client `psql \dT pg_catalog.int4` / `\dT pg_catalog.text` type display, first-slice `information_schema.tables` / `information_schema.columns` rows for supported tables including richer table catalog/insertability/type-shape fields, table-name `IN (...)` subset table discovery, all-column, table-name `IN (...)` subset, and per-table detail projection enumeration across supported `public` table columns, supported nullability/default/type-identity and `int4` numeric precision/radix/scale fields including table-filtered and table-name `IN (...)` extended metadata, first-slice `information_schema.schemata` rows for the supported `public` schema, and empty `information_schema.table_constraints` / `information_schema.key_column_usage` rows for the current no-SQL-constraint subset, while user-visible SQL indexes, SQL views, SQL casts, SQL extensions, SQL constraints, column defaults, comments, user-defined types, PostgreSQL heap-size/description metadata, ACL/policy mutation, tablespace creation/location/options, and advanced PostgreSQL catalog coverage remain unsupported.

2026-05-13 follow-up: catalog-qualified `information_schema.columns` extended metadata filters using `table_catalog = current_database()` or the supported `postgres` catalog literal are covered by real psql golden scenario 44 and backed by session catalog column/type metadata. Catalog-qualified richer `information_schema.tables` filters using the same supported catalog predicates are covered by real psql golden scenario 45 and backed by session catalog table metadata. Direct `pg_catalog.pg_namespace` lookup for the supported `public` namespace is covered by real psql golden scenario 46 and returns the stable bootstrap namespace OID/name row while broader namespace catalog behavior remains out of scope. Real PostgreSQL 16 `psql \dn+ public` verbose schema introspection is covered by scenario 47 and returns the supported `public` namespace owner plus empty ACL/description/publication rows until those metadata surfaces exist. Real PostgreSQL 16 plain `psql \d` relation listing is covered by scenario 48 for supported `public` session tables and returns the same catalog-backed relation rows as table listing while unsupported relation kinds remain absent. Real PostgreSQL 16 `psql \dv` / `\dv+` view listing traffic is covered by scenario 49 and truthfully returns no rows for the current no-SQL-view subset. Real PostgreSQL 16 `psql \dm` / `\dm+` materialized-view listing and `\ds` / `\ds+` sequence listing traffic is covered by scenario 50 and truthfully returns no rows for the current no-materialized-view/no-sequence subset. Real PostgreSQL 16 `psql \df` function-listing traffic is covered by scenario 51 and truthfully returns no rows for the current no-user-defined-function subset. Real PostgreSQL 16 `psql \du` role listing is covered by scenario 52 and returns the bootstrap `postgres` role through a narrow `pg_catalog.pg_roles` compatibility slice while role mutation and broader role catalog behavior remain out of scope. Real PostgreSQL 16 `psql \l` database listing is covered by scenario 53 and returns the supported bootstrap `postgres` database through a narrow `pg_catalog.pg_database` compatibility slice while database creation, templates, ACL mutation, and broader database catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dx` extension listing is covered by scenario 54 and truthfully returns no rows through a narrow `pg_catalog.pg_extension` compatibility slice while extension install and broader extension catalog behavior remain out of scope. Real PostgreSQL 16 `psql \db` tablespace listing is covered by scenario 55 and returns bootstrap `pg_default` / `pg_global` rows through a narrow `pg_catalog.pg_tablespace` compatibility slice while tablespace creation, location management, options, and broader tablespace catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dL` procedural-language listing is covered by scenario 56 and truthfully returns no rows through a narrow `pg_catalog.pg_language` compatibility slice while language creation and broader language catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dA` access-method listing is covered by scenario 57 and returns the supported `heap` table access method through a narrow `pg_catalog.pg_am` compatibility slice while broader access-method creation or extension behavior remains out of scope. Real PostgreSQL 16 `psql \dT pg_catalog.*` and `\dT+ pg_catalog.*` type-listing traffic is covered by scenario 58 and returns the supported `int4`/`text` registry through a narrow `pg_catalog.pg_type` compatibility slice while user-defined types and broader type catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dD` and `\dD+` domain-listing traffic is covered by scenario 59 and truthfully returns no rows through a narrow `pg_catalog.pg_type` domain compatibility slice while domain creation, domain constraints, and broader domain catalog behavior remain out of scope. Real PostgreSQL 16 `psql \da` aggregate-listing traffic is covered by scenario 60 and truthfully returns no rows through a narrow `pg_catalog.pg_proc` aggregate compatibility slice while aggregate creation and broader function catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dc` conversion-listing traffic is covered by scenario 61 and truthfully returns no rows through a narrow `pg_catalog.pg_conversion` compatibility slice while encoding conversion creation and broader conversion catalog behavior remain out of scope. Real PostgreSQL 16 `psql \do` operator-listing traffic is covered by scenario 62 and truthfully returns no rows through a narrow `pg_catalog.pg_operator` compatibility slice while operator creation and broader operator catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dO` collation-listing traffic is covered by scenario 63 and truthfully returns no rows through a narrow `pg_catalog.pg_collation` compatibility slice while collation creation and broader collation catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dC` cast-listing traffic is covered by scenario 64 and truthfully returns no rows through a narrow `pg_catalog.pg_cast` compatibility slice while cast creation and broader cast catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dRp` publication-listing traffic is covered by scenario 65 and truthfully returns no rows through a narrow `pg_catalog.pg_publication` compatibility slice while publication creation and broader publication catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dRs` subscription-listing traffic is covered by scenario 66 and truthfully returns no rows through a narrow `pg_catalog.pg_subscription` compatibility slice while subscription creation and broader subscription catalog behavior remain out of scope. Real PostgreSQL 16 `psql \ddp` default-access-privilege listing traffic is covered by scenario 68 and truthfully returns no rows through a narrow `pg_catalog.pg_default_acl` compatibility slice while default privilege mutation and broader ACL catalog behavior remain out of scope. Real PostgreSQL 16 `psql \dd` object-description listing traffic is covered by scenario 70 and truthfully returns no rows for the current no-comment/no-user-defined-object-description subset while broader object comment, rule, trigger, operator-class, and operator-family catalog behavior remains out of scope.

2026-05-13 follow-up: common `information_schema.views` discovery for the supported `public` schema is covered by real psql golden scenario 71 and truthfully returns no rows for the current no-SQL-view subset while view definitions, view DDL, and broader view catalog behavior remain out of scope. Follow-up scenario 72 adds common `pg_catalog.pg_views` discovery for the same supported `public` schema and also truthfully returns no rows until SQL view metadata exists. Follow-up scenario 73 adds common `information_schema.tables` base-table discovery with system-schema exclusion, returning supported `public` session tables from catalog metadata.

### P3. PostgreSQL wire protocol execution path

Goal:
- Move the server from simple-query compatibility probes toward real libpq application compatibility.

Initial target:
- Implement a narrow but real extended-query path for `Parse`, `Bind`, `Describe`, `Execute`, `Sync`, and `Close`.
- Support text parameters and text result formats for the first relational SQL subset.
- Keep unsupported binary formats, copy, function call, and advanced portal behavior explicitly classified.

Exit criteria:
1. Extended protocol no longer returns the generic "unsupported by compatibility stub" error for the first supported prepared statement/query path. First slice landed on 2026-05-11: the compatibility endpoint accepts `Parse`, `Bind`, `Describe`, `Execute`, `Sync`, and `Close` for text-format `int4`/`text` parameters and text-format relational `SELECT` results.
2. Prepared statements and portals have session-local lifecycle tests. First helper-level coverage landed on 2026-05-11 for bound parameter substitution and catalog-backed row description. Follow-up on 2026-05-11 added session-local close lifecycle coverage proving statement close removes dependent portals while portal close leaves the prepared statement intact.
3. psql golden coverage includes at least one extended-query or prepared/parameterized flow that reaches engine execution. Scenario `05_extended_query_bind.sql` was added on 2026-05-11 and later tightened to a PostgreSQL 16-compatible `\bind` flow over a relational table; named parse/bind psql metacommands are client-version dependent, so session-local prepared/portal lifecycle remains covered by helper-level protocol tests. Follow-up on 2026-05-13 added scenario 67 for real PostgreSQL 16 `psql` `FETCH_COUNT` cursor flow over a supported relational `SELECT`, backed by session-local `DECLARE ... CURSOR FOR SELECT`, `FETCH FORWARD n`, and cursor `CLOSE` handling in the compatibility endpoint while broader cursor behavior remains out of scope. Scenario 69 covers real PostgreSQL 16 `psql \gdesc` result-description traffic for supported relational `SELECT` statements, including psql's follow-up `pg_catalog.format_type` query over described `int4`/`text` result OIDs.
4. Error responses include stable SQLSTATE/message contracts for unsupported protocol features. First slice landed on 2026-05-11 for unsupported parameter/result formats, missing prepared statements, missing portals, mismatched parameter counts, and limited portal fetches; follow-up tightened bind mismatch detection so extra parameters are rejected instead of ignored, then aligned described-portal execution with real `psql` row-description state. Copy/function-call flows remain explicitly unsupported.
5. Compatibility scorecard separates simple-query, extended-query, auth/startup, and error-path coverage. First `protocol.extended_query` classifier landed on 2026-05-11; follow-up added the `protocol.error_paths` bucket for protocol/psql compatibility tests whose names identify unsupported, missing, mismatch, invalid, SQLSTATE, or rejection behavior.

### P4. SQL-to-GPU execution bridge

Goal:
- Lower supported relational SQL plans into the existing MVCC/CUDA execution backend where the operation is GPU-eligible, while preserving CPU truth and fallback accounting.

Initial target:
- Map simple table scans, equality/range predicates, ordering, projection, and limit into `MvccReadQuery` or a successor contract only if the current contract becomes insufficient.
- Record planned vs executed device target per SQL query.

Exit criteria:
1. At least one relational `SELECT` over table data reports GPU execution on local NVIDIA hardware. First slice landed on 2026-05-11: `execute_relational_select_with_cuda_driver_probe(...)` can report `executed_target = gpu(0)` for `SELECT * FROM table` over stored relational rows.
2. CPU-vs-GPU parity tests compare SQL-level results, not just internal MVCC rows. First slice landed on 2026-05-11 with SQL-result parity coverage for relational table scans through an alternate GPU backend and an ignored local CUDA regression for the real driver path.
3. Unsupported SQL plan nodes fall back with explicit reason labels visible in `Engine::status_snapshot()` or equivalent telemetry. First slice landed on 2026-05-11: SQL-side filter/order/projection/limit finalization recorded `GpuMvccReadParityGap` while still allowing the underlying MVCC row fetch to execute on GPU. Follow-up on 2026-05-11 pushes supported equality predicates through the relational equality-index `KeyBatchLookup` source and unordered `LIMIT` into the MVCC/CUDA query. Later 2026-05-11 slices tighten fallback accounting so projection-only SQL result shaping no longer counts as GPU fallback after GPU row fetch, supported decoded-column `ORDER BY` uses an ordered key-batch access path with `LIMIT` pushdown, narrow literal range predicates (`<`, `<=`, `>`, `>=`) use a filtered key-batch bridge, narrow `AND` conjunctions over supported literal predicates use a conjunctive key-batch bridge, and top-level `OR` groups over supported literal predicates, including parenthesized groups, use a disjunctive key-batch bridge instead of recording `GpuMvccReadParityGap`.
4. Benchmark/scorecard output includes SQL-level GPU execution rate and CPU fallback rate for the supported relational mix. First slice landed on 2026-05-11 with `RelationalSqlGpuBridgeReport::from_results(...)` plus a `sql.gpu_bridge` scorecard bucket. Follow-up on 2026-05-11 publishes row-batch H2D transfer bytes and GPU execution timing samples from CUDA-probed MVCC reads into engine metrics, so the P7 relational report no longer records zero transfer/timing counters for GPU-executed SQL reads.
5. Docs explain the SQL plan shapes that are GPU-eligible and the next GPU bridge boundary. First slice docs landed on 2026-05-11: plain `SELECT * FROM table` row fetch is GPU-eligible. Follow-up on 2026-05-11 adds `SELECT * FROM table WHERE column = literal [LIMIT n]` for supported equality-index predicates plus unordered limits; projection-only result shaping is now classified as host formatting rather than GPU fallback. A later 2026-05-11 slice adds supported decoded-column `ORDER BY [ASC|DESC] [LIMIT n]` through an ordered key-batch bridge for full scans and equality-index predicates. Follow-up slices add single-column literal range comparisons, narrow `AND` conjunctions, and top-level `OR` groups, including parenthesized predicate groups, through filtered/conjunctive/disjunctive key-batch bridges for both unordered and ordered relational reads. Remaining bridge boundaries are broader expressions, nested boolean trees, and transfer-layout/batching work needed for performance.

### P5. Production storage, indexing, and recovery

Goal:
- Replace bootstrap in-memory assumptions with durable relational storage behavior suitable for restart, larger datasets, and GPU transfer planning.

Initial target:
- Durable table data/checkpoint/recovery path aligned with WAL-before-visibility.
- First index or access-path strategy for point/range predicates.
- Compaction/vacuum or documented retention boundary for MVCC versions.

Exit criteria:
1. A relational table survives restart/recovery in an automated test. First file-backed slice landed on 2026-05-11: `Engine::persist_durable_wal_to_file(...)` writes the flushed WAL prefix to a checksummed local segment and `Engine::recover_from_durable_wal_file(...)` recovers relational catalog plus table rows from that segment. Follow-up on 2026-05-11 added `Engine::persist_durable_wal_checkpoint(...)` / `Engine::recover_from_durable_wal_checkpoint(...)`, so a checkpoint-control file can identify the durable segment and validate durable record count plus last transaction id before replay.
2. WAL replay restores catalog plus table data to the committed boundary. The file-backed P5 slice reuses the existing durable-prefix replay path, so recovery applies only flushed records and rebuilds volatile relational indexes from committed WAL contents. The checkpoint-control slice now checks that the selected segment matches the committed boundary recorded by control metadata before applying it.
3. At least one indexed or access-path-backed predicate is used by planning/execution with a measurable fixture. First slice landed on 2026-05-11: WAL-applied inserts maintain a table/column/value equality index that is rebuilt by durable WAL recovery, and `WHERE column = literal` relational reads use a `KeyBatchLookup` access path with `RelationalSelectResult::access_path` reporting `EqualityIndex { table, column, matched_keys }`. Follow-up P4 bridge work can also report `OrderedKeyBatch { ... }` when supported `ORDER BY` shapes are lowered through an ordered key batch, and the file-backed P5 recovery test proves that indexed ordered key-batch access survives restart-shaped WAL replay.
4. MVCC version retention has a tested safe boundary or a documented operational limitation. First checkpoint-vacuum boundary landed on 2026-05-11: `Engine::checkpoint_vacuum_mvcc_versions(...)` prunes versions deleted at or before a durable safe transaction id, rejects boundaries that cross active transactions or the flushed WAL boundary, and keeps WAL replay as the source of truth for reconstructing pruned history. Relational equality-index entries remain volatile and rebuilt from WAL replay rather than manually pruned.
5. Storage/recovery docs and runbooks match the implemented behavior. First P5 docs reconciliation landed on 2026-05-11 in `docs/architecture/05-storage-and-recovery.md`, `docs/operations/runbooks.md`, README, compatibility matrix, and this roadmap; the checkpoint-control update now documents single-control-file segment discovery while leaving PITR selection, multi-segment archive replay, and automatic retention cleanup outside the current P5 proof.

### P6. Operational replication and deployment

Goal:
- Turn replication semantics and role gates into a deployable multi-node database story.

Initial target:
- 3-node raft deployment harness or reproducible local cluster script.
- Follower apply/catch-up and leader failover smoke path.
- Operator-facing readiness/failover evidence.

Exit criteria:
1. A scripted multi-node run demonstrates write on leader, catch-up on follower, and read-after-apply behavior. First slice landed on 2026-05-11 with `scripts/run_replication_cluster_smoke.sh`, a local in-process 3-node Raft smoke harness that prints catch-up/read-after-apply evidence. Follow-ups on 2026-05-11 route the smoke through typed `AppendEntriesRequest` / `AppendEntriesResponse` messages, add a tested binary frame codec, add reusable single-request TCP send/serve helpers, use that TCP helper for the failover append-entry replication step, add deterministic request-vote election evidence before failover, and add `scripts/run_replication_packaged_smoke.sh` to build and execute the packaged-local example binary.
2. A failover or leader-transition scenario is tested with explicit admission gates. First slice landed on 2026-05-11 with old-leader `NotLeader` rejection after role transition and new-leader continuation in `operational_replication_three_node_smoke_catches_up_reads_after_apply_and_gates_failover`. Follow-up on 2026-05-11 replaced manual promotion in the smoke path with `start_candidate_election(...)` plus request-vote quorum evidence before elected-leader continuation.
3. Replication lag/readiness/failure signals appear in the engine truth surface or operational output. First slice landed on 2026-05-11 with smoke output for leader/follower commit/apply/caught-up state plus assertions through `ReplicationProgress` and `ReplicationStatusSnapshot`. Follow-up on 2026-05-11 promoted that output into `OperationalClusterSmokeReport`, which emits stable operator-readable lines and treats the smoke as passed only when the follower is caught up, commit/apply indexes match the promoted leader, old-leader writes are rejected, and the promoted node is leader. Later P6 slices wrap this in `OperationalDeploymentPreflightReport`, add `OperationalTransportSmokeReport`, add typed append-entries response evidence, `OperationalElectionSmokeReport`, and `OperationalPackageSmokeReport`, so the scripts also emit deployment-scope, TCP append-entries transport evidence, deterministic election evidence, packaged-local entrypoint evidence, and explicit gap lines.
4. Backup/PITR/DR runbooks identify what is implemented, simulated, or still missing. First P6 runbook reconciliation landed on 2026-05-11 and labeled the local smoke path as in-process, with networked transport, automatic election, and packaged deployment still missing. Follow-ups on 2026-05-11 made that evidence executable: the smoke scripts now fail unless operator output includes `operational_deployment_preflight=passed`, `deployment_scope=packaged_local_three_node_raft_smoke`, `deployment_transport=single_request_tcp_append_entries ...`, `deployment_election=deterministic_request_vote ...`, `deployment_package=local_cargo_example_binary ...`, and explicit `deployment_gap_*` lines. Network transport, deterministic election, and packaged-local deployment now report implemented; long-running multi-process/container deployment remains outside the current P6 proof.
5. Compatibility scorecard or testing report includes the operational replication scenario. First slice landed on 2026-05-11 with the `replication.operational_cluster` scorecard bucket.

### P7. Real workload performance proof

Goal:
- Prove the design earns GPU complexity on realistic workloads, not only fixture-level parity.

Initial target:
- Define one app-shaped workload and one analytical/TPC-ish workload that fit the supported SQL subset.
- Measure CPU baseline, GPU path, fallback rate, H2D/D2H bytes, kernel time, and total latency/throughput.

Exit criteria:
1. Benchmarks run reproducibly from a checked-in command/script. First slice landed on 2026-05-11 with `scripts/run_p7_relational_benchmark.sh`.
2. Reports include dataset size, concurrency level, device info, fallback rate, and correctness validation. First report landed on 2026-05-11 in `docs/testing/reports/p7-relational-benchmark-2026-05-11.md` with dataset size, concurrency, NVIDIA device info, CPU/GPU-probe latency, qps, fallback rate, transfer counters, kernel-timing counters, and SQL-result correctness validation. Follow-up on 2026-05-11 adds an `app_batched_or_lookup` workload so the report compares repeated app point lookups against one supported batched `OR` shape. Later P7 report hardening adds per-workload p50/p95/max CPU and GPU-probe latency plus CPU-vs-GPU total ratios, making the "no workload-level GPU advantage yet" claim directly inspectable from the checked-in artifact. The CUDA probe runtime is now cached per engine and recorded in the report, normalized H2D-per-query plus D2H-per-result-row values separate transfer pressure from raw totals, and compact key-batch transfer now shows keyed workloads moving requested-row batches instead of full table row batches.
3. At least one benchmark demonstrates a clear GPU advantage or records why the current architecture does not yet achieve one. First report records no workload-level GPU advantage: analytical scans reach GPU execution but remain slower than CPU. Follow-up on 2026-05-11 removes SQL fallback from the indexed lookup workload by pushing supported equality predicates into the MVCC/CUDA bridge; later range/conjunction/disjunction slices add analytical predicate benchmarks, and the batched lookup slice shows one supported batched query reduces GPU-probe latency versus repeated point lookups while still not beating the CPU baseline.
4. Performance results drive a named follow-up decision, such as device-side set/multiset algebra, indexing, batching, or transfer layout work. First report names transfer layout, batching, SQL predicate/order/projection pushdown, and CUDA driver timing publication into engine metrics as the next performance decisions; after equality predicate pushdown, projection fallback-accounting cleanup, decoded-column ordering, range predicate bridging, conjunction bridging, disjunction bridging, SQL-visible CUDA probe transfer/timing publication, the batched lookup comparison, latency-ratio report hardening, per-engine CUDA driver probe runtime caching, normalized transfer-pressure reporting, and compact keyed H2D transfer, remaining SQL bridge performance decisions narrow to batching, richer expression/boolean pushdown, and kernel/transfer timing refinement beyond initial driver probing.
5. Docs record which performance claims are supported and which are not. First report limits the supported claim to reproducible routing/correctness measurement and explicitly rejects broad GPU advantage claims for the current relational workload mix.

## Exit criteria for no-GPU bootstrap phase

- Commit path is replication-shaped and invariant-tested.
- Planner/executor contracts are device-aware.
- CPU semantics stable enough to act as truth oracle.
- Mock GPU path runs deterministic replay tests.
- No major interface changes required before plugging in CUDA backend.

No-GPU bootstrap phase closeout recorded on 2026-05-04:
- [x] Commit path is replication-shaped and invariant-tested.
- [x] Planner/executor contracts are device-aware.
- [x] CPU semantics are stable enough to act as truth oracle.
- [x] Mock GPU path / fallback path uses deterministic replay fixtures for parity checks.
- [x] No major interface changes are required before plugging in `CudaBackend`.
- Evidence captured in `docs/roadmap/no-gpu-bootstrap-closeout-review.md`.

## Active loop policy until GPU transition

The no-GPU loop is no longer allowed to grow the execution surface just because another permutation is imaginable. From this point forward, each loop must choose one of only three justified modes:

1. **Named semantic gap closure**
   - Add or refine a read-path capability only if it closes a clearly named missing semantic category that matters for the eventual GPU executor contract.
   - The loop must state the gap explicitly before implementation.
2. **Closeout consolidation**
   - Tighten docs, invariants, fixtures, parity telemetry, or engine-facing tests so the current CPU truth surface is easier to port and verify on GPU.
   - Prefer this mode when no crisp missing semantic category can be named.
3. **GPU-transition preparation**
   - Strengthen the contract boundary the first CUDA slice will rely on: device routing, fallback accounting, deterministic parity fixtures, and operator-level shape stability.

Operational stop rule for the remaining no-GPU loop:
- If a proposed loop does not materially improve one of these three areas, do not do it.
- If no remaining semantic category can be named and the contract boundary already looks stable, run the full validation gate and perform a bootstrap closeout review instead of expanding Q2 further.
- Once Q1-Q5 are all marked complete, treat the autonomous no-GPU queue as closed unless hardware onboarding begins or a real contradiction reopens a named gap.

## Bootstrap closeout review (required before CUDA work starts)

Before starting real GPU execution, perform and record a closeout review that answers each item explicitly:

1. **Execution contract stability**
   - `Engine::execute_mvcc_query()` supported source/filter/order/projection semantics are documented and covered by engine-facing tests.
   - Remaining gaps are tracked as explicit future work, not implicit assumptions.
2. **Parity/fallback truth surface stability**
   - planned-vs-executed device reporting and fallback reasons are visible, deterministic, and already asserted in tests/fixtures where applicable.
3. **Deterministic replay coverage**
   - the deterministic point-lookup, full-scan/simple-filter, and source-composition workload fixtures remain sufficient to compare CPU and future GPU outputs for the first CUDA slice.
   - if not sufficient, add the smallest missing fixture before GPU work begins.
4. **No-rewrite check**
   - no pending design concern implies a major planner/executor/result-contract rewrite before `CudaBackend` can be attached.
   - `Engine::execute_mvcc_query()` now already routes through an internal execution-backend hook (`CpuMvccExecutionBackend` today), so backend swaps can be proven without changing `MvccReadQuery`, `MvccReadResult`, or `MvccReadRow { source_key, key, value }`.
   - the initial CUDA envelope is already regression-bounded: supported `FullScan` / `KeyLookup` + simple-filter shapes can be exercised through an alternate backend while unsupported shapes demonstrably fall back through the CPU reference path under the same tracked `GpuMvccReadParityGap` reason.
   - that first-slice boundary is also classified explicitly now (`unsupported_source`, `unsupported_order`, `unsupported_projection`, `unsupported_filter`, `empty_logical_filter_tree`), so future CUDA routing can explain the first missed contract edge without another query/result rewrite.
5. **Validation gate green**
   - `cargo fmt --all`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --all --all-features`

If every item above is satisfied, mark the no-GPU bootstrap phase closed and move to the hardware-onboarding checklist.

Closeout review recorded on 2026-05-04:
- satisfied: execution contract stability
- satisfied: parity/fallback truth surface stability
- satisfied: deterministic replay coverage
- satisfied: no-rewrite check (including explicit first-slice gap labels for future CUDA routing)
- satisfied: validation gate green (`cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all --all-features`)

## CUDA completion loop policy

Now that local NVIDIA hardware is available, the autonomous loop must stop treating CUDA work as an open-ended "next small primitive" queue. The loop should drive the project through named completion gates, in order, with each run either advancing the active gate, tightening the parity evidence for that gate, or stopping after validation if no material progress is available.

Loop rules:
- Start every CUDA loop by naming the active completion gate and the exact contract gap being closed.
- Do not widen API surface, query semantics, or CPU-only behavior unless it directly supports the active gate.
- Preserve the existing `Engine::execute_mvcc_query()` result contract unless a recorded design contradiction proves it cannot carry GPU execution safely.
- Keep CPU fallback live for unsupported shapes, but treat new fallback as temporary completion debt with an explicit `GpuMvccReadParityGap` reason and a milestone owner gate.
- Require CPU-vs-GPU parity coverage for every newly GPU-eligible shape before it can report `executed_target = gpu(...)`.
- Prefer milestone-sized changes over primitive-by-primitive churn: each committed slice should leave docs, tests, and telemetry pointing at the next remaining gap.
- If a loop cannot name a gate-aligned gap, run the validation gate and update the completion checklist instead of inventing work.

Validation gate for every completion-gate change:
- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all --all-features`
- relevant local NVIDIA ignored tests with `--include-ignored --nocapture` when the change touches CUDA runtime or CUDA MVCC routing

### CUDA completion gates

1. **GPU row format and transfer contract**
   - Define the device row layout for MVCC keys, values, visibility metadata, and provenance handles.
   - Record SoA/AoS choices, alignment/coalescing assumptions, null/empty value representation, and bounded allocation strategy.
   - Add host-to-device and device-to-host fixtures that compare encoded rows against the CPU truth rows.
   - Exit when supported first-slice queries can build and inspect a GPU row batch without changing `MvccReadRow`.
   - Progress on 2026-05-10: `CudaMvccRowBatch` defines the first SoA transfer contract: `u32` key/value offset arrays, flattened key/value byte buffers, `u64` begin/end transaction visibility bounds, and `u32` provenance handles. Empty keys/values are represented by equal adjacent offsets, transfer sizing is explicit, offsets are validated before launch, and `CudaDriverRuntime::mvcc_row_batch_lengths(...)` proves H2D -> CUDA kernel -> D2H inspection of row lengths without changing `MvccReadRow`.
2. **Native scan and snapshot visibility parity**
   - Move full-scan row selection and snapshot visibility filtering into CUDA kernels.
   - Keep the CPU storage contract as the truth oracle while proving visible/invisible row parity on deterministic fixtures.
   - Exit when supported filterless full-scan shapes no longer depend on CPU row selection for visibility decisions.
   - Progress on 2026-05-10: `CudaDriverRuntime::mvcc_visibility_mask(...)` now evaluates row-batch `created_by`/`deleted_by` MVCC bounds on-device and `CudaMvccExecutionBackend` composes that visibility mask with existing CUDA filter masks before projection. Supported CUDA full scans now feed all stored MVCC versions into the row batch before device visibility masking, so filterless full-scan row selection no longer depends on CPU-visible preselection; CPU fallback re-resolves the visible snapshot rows to preserve reference correctness. Gate 2 is closed for the supported full-scan envelope.
3. **Native point/key lookup parity**
   - Add GPU-side key lookup/index probing for supported point and batch lookup shapes.
   - Preserve request-order semantics and existing fallback accounting for unsupported lookup/source variants.
   - Exit when supported `KeyLookup` and first supported `KeyBatchLookup` shapes execute lookup work on CUDA with CPU parity tests.
   - Progress on 2026-05-10: supported `KeyLookup` now feeds all stored MVCC versions into CUDA and composes a device key-equality source mask with the device visibility/filter masks, so historical point lookup selection no longer depends on CPU-visible preselection. The first supported `KeyBatchLookup` path preserves request order by running one CUDA pass over a compact all-version row batch containing only requested keys instead of transferring unrelated table rows. Local hardware regressions cover historical key lookup and two-key compact batch lookup without fallback; CPU fallback still re-resolves visible rows when the driver is unavailable.
4. **Provenance and filter expansion parity**
   - Extend GPU predicates beyond the current filterless/value/key prefix/range mask primitives into the next provenance-aware filter categories.
   - Port only categories that can be named, fixture-backed, and compared against CPU output.
   - Exit when the remaining filter fallbacks are documented by unsupported semantic category rather than by missing primitive plumbing.
   - Progress on 2026-05-10: gate 4 has started with named provenance-aware predicate categories. `CudaMvccExecutionBackend` now supports single-frame `ProvenanceKeyPrefix` and `ProvenanceValueEquals` filters plus provenance bundle key/value/key-value equality, counted membership, key-prefix filters, stable bundle path predicates (`PathEquals`, `PathContains`, `PathCountAtLeast`, `PathPairAtDistance`, path prefix/suffix/slice/segment equality, and bundle length), and same-subpath plus mixed-subpath occurrence path predicate families, including nested `All` / `Any` composition with existing supported predicates, over CPU-resolved provenance rows from value-chain sources. This keeps source resolution and composition movement out of scope for gate 4 while proving CUDA predicate masks against CPU output. Provenance ordering/projection movement and native GPU source resolution for join/composition shapes remain open.
   - Closeout on 2026-05-10: gate 4 is closed for the current provenance filter enum surface. Provenance filter categories are CUDA-eligible over CPU-resolved provenance rows with CPU parity coverage; source-relative and branch-label filters move under gate 5 because they depend on broader source/composition movement rather than missing provenance predicate plumbing.
5. **Ordering, projection, and composition GPU coverage**
   - Move the first stable ordering and projection shapes behind the CUDA backend without changing external result rows.
   - Add GPU coverage for the first multi-source composition shape only after single-source scan/lookup/filter parity is stable.
   - Exit when at least one ordering, one projection, and one composition family has CUDA parity and unsupported families still fall back explicitly.
   - Progress on 2026-05-10: gate 5 has started with CUDA-backed key ordering for full-scan/key-lookup shapes plus CUDA-backed `Concat` over native child sources (`FullScan`, `KeyLookup`, `KeyBatchLookup`). The engine dispatches each child source through the same CUDA source/visibility/filter/projection path used by single-source reads and appends child outputs in source-list order, preserving existing `Concat` semantics without changing `MvccReadRow`. Broader composition families, non-key ordering, key-batch/concat ordering, and provenance/source-aware projection movement remain open.
   - Closeout on 2026-05-10: gate 5 is closed for the first coverage milestone. CUDA parity now exists for key/value/key-only/value-only projection, key asc/desc ordering on full-scan/key-lookup shapes, and `Concat` over native child sources. Unsupported ordering/projection/composition families are classified as explicit post-milestone `GpuMvccReadParityGap` work rather than accidental misses.
6. **Benchmark, telemetry, and fallback-rate regression gates**
   - Add benchmark fixtures that report GPU-executed workload percentage, CPU fallback rate, H2D/D2H bytes, kernel time, and batch wait time where available.
   - Treat rising fallback rate on the benchmark mix as a regression once a gate is closed.
   - Exit when local runs can distinguish correctness regressions from performance/fallback regressions.
   - Progress on 2026-05-10: `MvccBenchmarkReport::from_results(...)` now summarizes deterministic MVCC workload results into GPU-executed percentage, CPU fallback percentage, H2D/D2H bytes, kernel execution sample/time totals, and batch wait sample/time totals. The first regression fixture mixes CUDA-eligible key-ordered scan and native-source concat reads with an intentionally unsupported distinct composition read so local runs can distinguish coverage/fallback regressions from correctness regressions.
   - Closeout on 2026-05-10: gate 6 is closed for the first benchmark target. The mixed MVCC CUDA benchmark fixture originally asserted at least 66.66% GPU-executed workload coverage and at most 33.33% CPU fallback rate, then moved to 100% GPU execution and 0% CPU fallback for the current scan/concat/nested-composition mix on 2026-05-11 while carrying H2D/D2H, kernel time, and batch wait fields forward for real CUDA runner output.
7. **GPU CI or reproducible runner**
   - Provide either GPU-capable CI or a reproducible local runner script/profile that executes the CUDA parity suite and captures environment details.
   - Publish driver/device/runtime evidence with test output.
   - Exit when another developer or runner can repeat the CUDA validation gate without relying on ad hoc machine state.
   - Progress on 2026-05-10: `scripts/run_cuda_parity.sh` now records timestamp/host/kernel/Rust toolchain and `nvidia-smi` GPU inventory to `target/cuda-parity/environment.txt`, then runs the CUDA runtime and MVCC ignored hardware parity suites with `--include-ignored --nocapture`, teeing output to `target/cuda-parity/cuda-parity.log`.
   - Closeout on 2026-05-10: gate 7 is closed for a reproducible local runner. The runner passed on local hardware (`NVIDIA GeForce RTX 3090`, driver `595.58.03`) and captured repeatable environment/test evidence under `target/cuda-parity/`.

Completion rule:
- The project is not "CUDA complete" until gates 1-7 are closed, the fallback-rate benchmark target is recorded, and unsupported remaining shapes are deliberately classified as post-v1 scope rather than accidental gaps.

CUDA completion closeout recorded on 2026-05-10:
- [x] Gate 1: GPU row format and transfer contract
- [x] Gate 2: Native scan and snapshot visibility parity
- [x] Gate 3: Native point/key lookup parity
- [x] Gate 4: Provenance and filter expansion parity
- [x] Gate 5: Ordering, projection, and composition GPU coverage
- [x] Gate 6: Benchmark, telemetry, and fallback-rate regression gates
- [x] Gate 7: GPU CI or reproducible runner
- [x] Fallback-rate benchmark target recorded for the current mixed MVCC CUDA fixture
- [x] Unsupported remaining shapes classified as explicit post-milestone `GpuMvccReadParityGap` work

Post-closeout CUDA gap closure:
- Progress on 2026-05-10: CPU-resolved provenance sources can now keep CUDA execution when they request provenance ordering and provenance projection, provided they also contain a supported provenance predicate for CUDA mask execution. This closes the first post-milestone provenance order/projection routing gap without changing `MvccReadRow`: CUDA still performs source/visibility/filter masking, then the existing host row contract performs provenance sort/project materialization. Native GPU source resolution for join/composition shapes and composition beyond native-source `Concat` remain explicit post-milestone `GpuMvccReadParityGap` work.
- Progress on 2026-05-10: supported CUDA paths now apply post-filter/post-order `limit` handling without changing `MvccReadRow`. Single-source CUDA execution truncates the matched resolved rows after optional ordering and before projection; native `KeyBatchLookup` and `Concat` truncate after request-order/source-order fan-in. CPU parity coverage and local ignored CUDA regressions cover ordered full-scan limits plus concat/key-batch fan-in limits.
- Progress on 2026-05-10: source-relative and branch-label filters (`SourceKeyPrefix`, `SourceValueEquals`, `BranchLabelEquals`) are now CUDA-eligible over CPU-resolved rows via the existing CUDA mask bridge, with classifier, CPU parity, and local ignored CUDA coverage.
- Progress on 2026-05-10: source-relative and branch-label ordering (`SourceKeyAsc/Desc`, `SourceValueAsc/Desc`, `BranchLabelAsc/Desc`) is now CUDA-eligible over CPU-resolved rows through the existing post-mask sort path, with classifier, CPU parity, and local ignored CUDA coverage. Later slices closed CPU-resolved composition and key/value ordering gaps; remaining post-milestone work is native GPU source resolution for join/composition shapes.
- Progress on 2026-05-10: source-relative and branch-label projections (`BranchLabelTargetValue`, `SourceKeyTargetValue`, `SourceValueOnly`, `TargetKeySourceValue`) are now CUDA-eligible over CPU-resolved rows through the existing post-mask projection path, with classifier, CPU parity, and local ignored CUDA coverage. Later slices closed CPU-resolved composition and current enum-surface order/projection gaps; remaining post-milestone work is native GPU source resolution for join/composition shapes.
- Progress on 2026-05-10: `Concat` over CPU-resolved child sources is now CUDA-eligible after CPU source resolution, so mixed provenance/source-relative filter/order/projection work can still run through the CUDA mask/sort/project path without native join source resolution. Remaining composition gaps are distinct/intersect/except/symmetric-difference families and native GPU source resolution for joins/composition.
- Progress on 2026-05-10: distinct/intersect/except/symmetric-difference composition over CPU-resolved child sources is now CUDA-eligible after CPU source resolution when a supported source/provenance predicate drives CUDA mask execution. This preserves the existing CPU truth for set/multiset composition semantics while moving downstream mask/order/project work through the CUDA backend. Remaining composition gap is native GPU source resolution for joins/composition, plus native device-side set/multiset composition if future performance goals require it.
- Progress on 2026-05-10: value asc/desc ordering is now CUDA-eligible for native full-scan and key-lookup reads after CUDA source/visibility/filter masking, using the same stable post-mask sort path that already backs key ordering. Later slices extended key/value ordering to native fan-in and CPU-resolved sources.
- Progress on 2026-05-10: key/value ordering and post-order limit are now CUDA-eligible for native key-batch and native `Concat` fan-in. Fan-in wrappers preserve CUDA source/visibility/filter masking per child, defer final projection until after fan-in order/limit, and keep request/source order when no order is requested. Later classifier coverage closed the current enum-surface ordering gap; remaining post-milestone work is native GPU source resolution for joins/composition.
- Progress on 2026-05-10: key/value ordering is now CUDA-eligible over CPU-resolved sources, matching the existing post-mask sort semantics used for native sources. Classifier coverage now explicitly enumerates every current `MvccReadOrder` and `MvccProjection` variant over CPU-resolved CUDA sources. The stale "ordering/projection outside key/value/provenance/source/branch" gap is closed for the current enum surface; remaining ordering/projection work should be introduced only when new enum variants land or when native GPU source resolution makes a currently source-bound family meaningful for native sources.
- Progress on 2026-05-10: `FollowValueChain { terminal: CurrentRow }` has the first native CUDA join source-resolution slice. CUDA now selects visible seed rows and each value-key hop from all MVCC versions with device key masks plus the device MVCC visibility kernel, then reuses the existing host provenance/result-row assembly before downstream CUDA filter/order/projection. The remaining native source-resolution gaps are prefix-terminal value-chain expansion, branch fan-in/labeled branches, and native device-side set/multiset composition if future performance goals require it.
- Progress on 2026-05-10: prefix-terminal `FollowValueChain { terminal: CurrentValuePrefixes }` now has native CUDA source resolution as well. CUDA selects visible seeds and value-key hops with device key masks plus the device MVCC visibility kernel, then expands the terminal prefix through a CUDA key-prefix mask over all MVCC versions before the unchanged host provenance/result-row assembly and downstream CUDA filter/order/projection. The remaining native source-resolution gaps are branch fan-in/labeled branch shapes and native device-side set/multiset composition if future performance goals require it.
- Progress on 2026-05-10: `FollowValueChainBranches` and `FollowValueChainLabeledBranches` now reuse the native CUDA value-chain resolver per visible seed row. CUDA selects seed/hop/prefix rows from all MVCC versions with device key masks plus the device MVCC visibility kernel, then applies `AllBranches` or `FirstNonEmptyBranch` fan-in per seed while preserving branch labels and provenance assembly through the unchanged host row contract. The remaining named source-resolution gap is native device-side set/multiset composition if future performance goals require moving those already-CUDA-eligible CPU-resolved composition semantics onto the device.
- Progress on 2026-05-11: native distinct/intersect/except/symmetric-difference sources whose children are already CUDA-resolvable now select each child through CUDA source/visibility masks before preserving the existing host set/multiset semantics and downstream CUDA filter/order/projection path. Follow-up closeout coverage now enumerates every current native set/multiset variant (`ConcatDistinct`, `IntersectDistinct`, `IntersectAll`, `ExceptDistinct`, `ExceptAll`, `SymmetricDifferenceDistinct`, `SymmetricDifferenceAll`) through the classifier, backend-swap parity path, and local CUDA regression. This removes the correctness-routing gap for native composition children without pretending the set algebra itself is device-side; remaining work is performance-only native device-side set/multiset algebra if a future benchmark target requires it.
- Progress on 2026-05-11: native `Concat` now accepts any CUDA-resolvable child, including nested distinct/intersect/except/symmetric-difference composition, by resolving each child through CUDA source/visibility masks before applying the existing concat fan-in/order/limit/projection path. The mixed MVCC CUDA benchmark fixture now has 100% GPU execution and 0% CPU fallback for its current scan/concat/nested-composition mix; remaining set/multiset work is still performance-only device-side algebra if a future benchmark target requires it.

## First CUDA transition slice (once NVIDIA hardware is available)

This section records the already-started first slice. Future loop runs should treat it as gate-0/bootstrap evidence for the CUDA completion loop above, not as the full roadmap.

Status update on 2026-05-09:
- Local NVIDIA hardware is now visible to the loop (`NVIDIA GeForce RTX 3090`, driver `590.48.01`).
- `gpu_db_execution::CudaDriverRuntime` now probes `libcuda`, initializes the driver, records driver version, device count, device names, and device memory sizes, and plugs into the existing `GpuRuntime`/`DeviceRouter` contract.
- `CudaDriverRuntime::launch_smoke_add_one(...)` now proves the local driver path can load PTX, allocate device memory, launch a minimal kernel, synchronize, and copy the result back to the host; this is a launch-harness prerequisite, not MVCC execution parity.
- `CudaDriverRuntime::filter_all_mask(...)`, `filter_equal_u32_mask(...)`, `filter_equal_bytes_mask(...)`, and `filter_bytes_range_mask(...)` are now the first real CUDA mask primitives: they perform device-side unconditional mask generation or H2D input copy plus predicate mask generation, synchronization, and D2H mask copy for filterless reads, fixed-width equality, byte/string equality, and bytewise range batches with ignored local-hardware parity tests.
- `CudaMvccRowBatch` now establishes the first GPU row format/transfer contract for MVCC rows, including key/value SoA buffers, visibility metadata, and provenance handles; `CudaDriverRuntime::mvcc_row_batch_lengths(...)` validates device-side inspection of that batch shape on local hardware.
- `CudaDriverRuntime::mvcc_visibility_mask(...)` now launches a CUDA MVCC visibility kernel over row-batch transaction bounds and is composed into `CudaMvccExecutionBackend` before filter/projection output, with local hardware parity coverage for created/deleted snapshot edges.
- `CudaMvccExecutionBackend` is attached behind the existing MVCC backend boundary and can be reached through `Engine::execute_mvcc_query_with_cuda_driver_probe(...)`.
- The first MVCC CUDA integration is intentionally narrow: supported filterless full-scan/key-lookup/key-batch reads, key/value ascending/descending order for full-scan/key-lookup/key-batch/native-`Concat` reads and CPU-resolved sources, post-filter/post-order limits, CUDA-backed `Concat` over native child sources, native distinct/intersect/except/symmetric-difference composition over CUDA-resolved child sources, native `FollowValueChain` / `FollowValueChainBranches` / `FollowValueChainLabeledBranches` source resolution for current-row and prefix-terminal expansion, CPU-resolved `Concat`, distinct/intersect/except/symmetric-difference composition over CPU-resolved child sources, and reads with exact numeric or byte/string `ValueEquals` filters, key-prefix filters, general bytewise key ranges, single-frame provenance key-prefix/value-equality filters, provenance bundle key/value/key-value equality, counted membership, key-prefix filters, stable bundle path predicates (`PathEquals`, `PathContains`, `PathCountAtLeast`, `PathPairAtDistance`, path prefix/suffix/slice/segment equality, and bundle length), same-subpath occurrence path predicates, mixed-subpath occurrence path predicates, source-key/source-value/branch-label filters, source/branch ordering/projection, provenance ordering/projection, and nested `All`/`Any` combinations of supported predicates can now use CUDA mask primitives plus device-side MVCC source/visibility masking where applicable and report `executed_target = gpu(...)` on local NVIDIA hardware. Supported full scans, key lookups, first key-batch lookups, value-chain branch joins, and native composition children feed all stored MVCC versions into CUDA before source/visibility masking.
- Remaining MVCC CUDA gaps: performance-only native device-side set/multiset algebra if future benchmark goals require moving the currently host-preserved set semantics fully onto the device.

1. Add CUDA build targets plus at least one reproducible GPU-capable CI/dev environment.
   - In progress: local GPU-capable dev environment detected; driver-level runtime probing now exposes device inventory (`id`, name, total memory) and a validated minimal kernel launch/D2H smoke path for transition diagnostics.
2. Implement `CudaBackend` behind the existing backend trait boundary without changing engine-facing contracts.
   - In progress: backend attachment exists; filterless reads, exact numeric/string `ValueEquals`, key-prefix filters, general bytewise key ranges, key/value ordering, post-order limits, single-frame provenance key-prefix/value-equality filters plus provenance bundle key/value/key-value equality, counted membership, key-prefix filters, stable bundle path predicates, same-subpath occurrence path predicates, and mixed-subpath occurrence path predicates over CPU-resolved provenance rows, supported nested `All`/`Any` predicate trees, row-batch MVCC visibility, supported full-scan all-version row selection, supported point key lookup, request-order-preserving key-batch lookup, native value-chain source resolution, native `Concat`, and nested native distinct/intersect/except/symmetric-difference composition now run through real CUDA mask/source/visibility routing where applicable, while unsupported shapes still intentionally fall back under `GpuMvccReadParityGap`.
3. Port only the first operator subset:
   - scan
   - snapshot visibility filtering
   - simple filter predicates
     - In progress: unconditional mask generation, fixed-width equality, byte/string equality, bytewise range CUDA predicate primitives, row-batch MVCC visibility masking, all-version full-scan row selection, all-version point/key lookup source masking, request-order-preserving first key-batch source masking, native-source `Concat`, single-frame provenance key-prefix/value-equality filtering, provenance bundle key/value/key-value equality, counted membership, key-prefix filtering, stable bundle path predicates, same-subpath occurrence path predicates, mixed-subpath occurrence path predicates over CPU-resolved provenance rows, and the first `CudaMvccRowBatch` inspection kernel are integrated for filterless reads, MVCC `ValueEquals`, key-prefix filters, key ranges, snapshot visibility, and logical mask composition over supported predicates.
   - point lookup / key lookup
     - In progress: supported point lookup and first key-batch lookup source selection now run through CUDA key-equality masks over all MVCC versions with CPU-vs-GPU parity regressions on local hardware.
4. Run CPU-vs-GPU parity checks against the existing deterministic fixtures and any minimal new fixture added during closeout.
5. Keep fallback routing live so unsupported shapes still execute via CPU with explicit tracked reasons.
6. Only after parity is trustworthy, widen GPU coverage to ordering, projection, multi-source composition, nested joins, and provenance-aware execution.
   - In progress: key/value ordering, supported projections, post-order limits, provenance/source-aware filter/order/projection over CPU-resolved rows, native value-chain source resolution, native `Concat`, and native set/multiset composition over CUDA-resolvable children are now CUDA-routed under the existing backend boundary. Remaining set/multiset work is performance-only device-side algebra if future benchmark targets require moving currently host-preserved composition semantics onto the device.
