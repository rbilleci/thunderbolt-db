# gpu-database-engine

Bootstrap implementation workspace for the pre-NVIDIA phase.

## Current scope

- Replication-shaped local commit path
- WAL-before-visibility invariant tests
- WAL durability + queue watermarks (flushed, buffered, unflushed, pending depth/capacity, pending age/deadline), commit/apply/visibility lag gauges, explicit backlog/gap blocker flags, active transaction depth, and failover/admission readiness flags (`quiescent_for_failover`, `follower_promotion_ready`, `mutation_admission_saturated`)
- Engine truth surface via `Engine::status_snapshot()` for served snapshot identity/frontier, active fallback reasons + parity rollups, and replication/readiness health
- Engine telemetry snapshot + sink publication API for replication lag, durable WAL/snapshot frontier, write-path readiness/backlog state, and runtime metrics
- Thin MVCC execution slice via `Engine::execute_mvcc_query()` covering snapshot-bound full scan / key lookup / key-batch fan-in / explicit multi-source `Concat`, `ConcatDistinct`, `IntersectDistinct`, `IntersectAll`, `ExceptDistinct`, `ExceptAll`, `SymmetricDifferenceDistinct`, and `SymmetricDifferenceAll` composition / generic `FollowValueChain { keys, plan, provenance }` linear nested-join expansion / `FollowValueChainBranches { keys, plans, fan_in, provenance }` plus `FollowValueChainLabeledBranches { keys, branches, fan_in, provenance }` seed-grouped or first-non-empty branch expansion / legacy value→key reference helpers layered on that same chain mechanism / optional key-prefix/range or source-aware key/value or branch-label filtering, explicit multi-frame provenance filters/projection on nested-join rows, source-aware ordering, limit, and key/value or join-side source-value/branch-label/provenance-value projection through the execution layer with explicit CPU fallback parity tracking (`GPU-123`)
- CPU-first reference engine skeleton
- Device-aware execution abstractions

## MVCC vertical slice (current bootstrap shape)

- Engine entry point: `Engine::execute_mvcc_query(&MvccReadQuery)`
- Supported sources:
  - `MvccReadSource::FullScan`
  - `MvccReadSource::KeyLookup { key }`
  - `MvccReadSource::KeyBatchLookup { keys }` (fan-in multi-source lookup; preserves request order before downstream filter/order/limit)
  - `MvccReadSource::Concat { sources }` (explicit composition primitive that concatenates existing source shapes in source-list order before downstream filter/order/limit)
  - `MvccReadSource::ConcatDistinct { sources }` (ordered deduplicating composition primitive; keeps the first occurrence of each resolved row while preserving distinct source-provenance rows)
  - `MvccReadSource::IntersectDistinct { sources }` (ordered intersection primitive; emits first-source rows whose exact resolved-row identity appears in every subsource, with source provenance treated as part of identity)
  - `MvccReadSource::IntersectAll { sources }` (ordered multiset intersection primitive; emits first-source rows up to the minimum exact resolved-row identity count across subsources, with source provenance treated as part of identity)
  - `MvccReadSource::ExceptDistinct { sources }` (ordered subtraction primitive; emits first-source rows whose exact resolved-row identity does not appear in any remaining subsource, with source provenance treated as part of identity)
  - `MvccReadSource::ExceptAll { sources }` (ordered multiset subtraction primitive; emits first-source rows after subtracting exact resolved-row identity multiplicity contributed by remaining subsources, with source provenance treated as part of identity)
  - `MvccReadSource::SymmetricDifferenceDistinct { sources }` (ordered unique-presence primitive; emits rows whose exact resolved-row identity appears in exactly one subsource, preserving first appearance order and treating source provenance as part of identity)
  - `MvccReadSource::SymmetricDifferenceAll { sources }` (ordered multiset symmetric-difference primitive; iteratively cancels exact resolved-row identity multiplicity source-by-source and emits the remaining imbalance in first appearance order, with source provenance treated as part of identity)
  - `MvccReadSource::FollowValueChain { keys, plan, provenance }` (generic source-preserving linear nested-join helper; follows `plan.value_key_hops` visible value→key hops from each visible seed row, then either returns the current row or expands visible keys matching the current row's value as a prefix)
  - `MvccReadSource::FollowValueChainBranches { keys, plans, fan_in, provenance }` (generic source-preserving branch helper; resolves multiple linear value-chain plans per visible seed row under an explicit per-seed fan-in policy such as `AllBranches` or `FirstNonEmptyBranch`)
  - `MvccReadSource::FollowValueChainLabeledBranches { keys, branches, fan_in, provenance }` (branch-labeled nested-join helper; preserves the same branch fan-in semantics while attaching stable branch labels that downstream filters/order/projections can inspect without fabricating extra row state)
  - `MvccSourceProvenance::Seed` preserves the original seed row as join provenance for source-aware filters, ordering, and projections.
  - `MvccSourceProvenance::TerminalInput` retargets source-aware filters, ordering, and projections to the row that fed the terminal resolution step (for example the profile row whose value drove a prefix fan-out).
  - Nested-join rows now retain an internal provenance path, so explicit frame-aware filters/projection can inspect `Seed`, `TerminalInput`, or any `ValueHop(n)` frame without changing the external row contract.
  - `MvccReadSource::FollowValueKeyRefs { keys }` (join-adjacent foreign-key-style expansion from seed row values to referenced keys)
  - `MvccReadSource::FollowValueKeyPrefixes { keys }` (prefix-driven foreign-key-style expansion from seed row values to visible target-key ranges)
  - `MvccReadSource::FollowValueKeyRefPrefixes { keys }` (source-preserving two-hop expansion: seed value -> referenced row -> prefix-driven target fan-out)
  - `MvccReadSource::FollowValueKeyRefValueKeyRefs { keys }` (source-preserving three-hop expansion: seed value -> referenced row -> referenced row value -> final referenced row)
  - `MvccReadSource::FollowValueKeyRefValueKeyPrefixes { keys }` (source-preserving chained fan-out: seed value -> referenced row -> referenced row value -> visible prefix-driven target expansion)
  - `MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes { keys }` (source-preserving four-hop fan-out: seed value -> referenced row -> referenced row value -> referenced row -> visible prefix-driven target expansion)
  - `MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs { keys }` (source-preserving deeper terminal chain: seed value -> referenced row -> referenced row value -> referenced row -> referenced row value -> referenced row -> final visible target row)
  - `MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes { keys }` (source-preserving deeper fan-out chain: seed value -> referenced row -> referenced row value -> referenced row -> referenced row value -> visible referenced row -> visible prefix-driven target expansion)
  - `MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes { keys }` (source-preserving deeper nested fan-out chain: seed value -> referenced row -> referenced row value -> referenced row -> referenced row value -> visible referenced row -> visible referenced-row value -> visible prefix-driven target expansion)
- Supported snapshot rule:
  - `visibility.read_txn_id` selects the MVCC snapshot frontier
- Supported filters:
  - `MvccReadFilter::KeyPrefix(prefix)`
  - `MvccReadFilter::SourceKeyPrefix(prefix)`
  - `MvccReadFilter::ProvenanceKeyPrefix { frame, prefix }`
  - `MvccReadFilter::BranchLabelEquals(label)`
  - `MvccReadFilter::KeyRange { start_inclusive, end_exclusive }`
  - `MvccReadFilter::ValueEquals(value)`
  - `MvccReadFilter::SourceValueEquals(value)`
  - `MvccReadFilter::ProvenanceValueEquals { frame, expected }`
  - `MvccReadFilter::All([...])`
  - `MvccReadFilter::Any([...])`
- Supported projections:
  - `MvccProjection::KeyValue`
  - `MvccProjection::KeyOnly`
  - `MvccProjection::ValueOnly`
  - `MvccProjection::BranchLabelTargetValue` (branch-aware projection that mirrors the resolved branch label into `key` while surfacing the resolved target row's value in `value`)
  - `MvccProjection::SourceKeyTargetValue` (join-adjacent result shape that mirrors the original seed key into `key` while projecting the resolved target row's value)
  - `MvccProjection::SourceValueOnly` (join-adjacent result shape that projects only the original seed row's value)
  - `MvccProjection::TargetKeySourceValue` (join-adjacent result shape that keeps the target key while projecting the original seed row's value)
  - `MvccProjection::TargetKeyProvenanceValue { frame }` (join-adjacent result shape that keeps the resolved target key while projecting the value from an explicit provenance frame such as `Seed`, `TerminalInput`, or `ValueHop(n)`)
- Result row shape:
  - `MvccReadRow { source_key, key, value }`
  - `source_key` is populated for join-adjacent expansion sources so source-preserving joins can keep seed provenance visible while join-side projections reuse the same engine-facing result contract.
- Supported ordering:
  - `MvccReadOrder::KeyAsc`
  - `MvccReadOrder::KeyDesc`
  - `MvccReadOrder::ValueAsc`
  - `MvccReadOrder::ValueDesc`
  - `MvccReadOrder::BranchLabelAsc`
  - `MvccReadOrder::BranchLabelDesc`
  - `MvccReadOrder::SourceKeyAsc`
  - `MvccReadOrder::SourceKeyDesc`
  - `MvccReadOrder::SourceValueAsc`
  - `MvccReadOrder::SourceValueDesc`
  - `MvccReadOrder::ProvenanceKeyAsc { frame }`
  - `MvccReadOrder::ProvenanceKeyDesc { frame }`
  - `MvccReadOrder::ProvenanceValueAsc { frame }`
  - `MvccReadOrder::ProvenanceValueDesc { frame }`
- Optional row cap:
  - `limit: Some(n)` applies after visibility + filter + ordering stages
- Current device strategy:
  - planned target: GPU default device
  - executed target: CPU reference semantics via `ScanOperator` → `FilterOperator` → `SortOperator` → `LimitOperator` → `ProjectOperator`
  - fallback reason: `FallbackReason::GpuMvccReadParityGap` (`GPU-123`)
- Deterministic workload fixtures:
  - `tests/fixtures/mvcc-read-workload.txt`
  - `tests/fixtures/mvcc-source-composition-workload.txt`
- Next obvious extension boundary:
  - extend the multi-frame provenance surface into lightweight provenance-path summarization or reusable frame bundles now that filter/projection/ordering can all target explicit frames, without regressing the same engine-facing contract and explicit fallback accounting on the eventual GPU-backed path.

## Quickstart

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Command notes

- `SET key=value`, `SET key TO value`, `SET LOCAL key=value`, and `SET SESSION key=value` are equivalent.
- `DEL key`, `DELETE key`, and `DELETE FROM key` are equivalent.
- `BEGIN|COMMIT|ROLLBACK` also accept `WORK` and `TRANSACTION` aliases.
- `COMMIT|ROLLBACK|END ... AND [NO] CHAIN` forms are accepted; `AND CHAIN` reopens transaction context by allocating a fresh transaction id after the terminal transition.
- `END` maps to `COMMIT`; `ABORT` maps to `ROLLBACK`.
- `START TRANSACTION` and `START WORK` map to `BEGIN`; optional `READ ONLY` / `READ WRITE`, `[NOT] DEFERRABLE`, and `ISOLATION LEVEL {SERIALIZABLE|REPEATABLE READ|READ COMMITTED|READ UNCOMMITTED}` suffixes are accepted on `BEGIN`/`START` aliases (including comma-separated mode lists) and currently map to plain `BEGIN` behavior.
- `CHECKPOINT`, `FLUSH`, `FLUSH WAL`, `FLUSH LOG`, `FLUSH WRITE AHEAD`, `FLUSH WRITE AHEAD {LOG|WAL}`, `FLUSH WRITE-AHEAD`, `FLUSH WRITE-AHEAD {LOG|WAL}`, `FLUSH WRITEAHEAD`, `FLUSH WRITEAHEAD {LOG|WAL}`, `FLUSH WRITE_AHEAD`, `FLUSH WRITE_AHEAD {LOG|WAL}`, `FLUSH WRITE_AHEAD_LOG`, and `FLUSH WRITE_AHEAD_WAL` are equivalent admin/coordination commands and are tracked as CPU fallback metric events.
- Replication backlog blocker labels can be decoded from delimited text streams (CSV-like, multiline, and JSON-like arrays with quoted labels).
- `RESET {ALL|ROLE|AUTHORIZATION|AUTH|SESSION AUTHORIZATION|SESSION AUTH}`, `DISCARD {ALL|TEMP|TEMPORARY|TEMP TABLE[S]|TEMPORARY TABLE[S]|PLANS|SEQUENCES}`, `DEALLOCATE {ALL|name|PREPARE|PREPARED name}`, `SET ROLE {NONE|DEFAULT|name}` (including quoted names), `SET SESSION AUTHORIZATION value` / `SET SESSION AUTH value` (including quoted names), `SET TRANSACTION ...`, `SET SESSION CHARACTERISTICS AS TRANSACTION ...`, `CLOSE {ALL|name}`, `LISTEN channel`, `NOTIFY channel[, payload]`, and `UNLISTEN [*|ALL|channel]` are accepted as CPU-routed session-control no-ops in the bootstrap engine so PostgreSQL-style client reset probes do not fail parser validation.

## Safety invariant

The engine preserves **WAL-before-visibility**: a state transition is never visible to readers before its corresponding WAL record is durably flushed.
