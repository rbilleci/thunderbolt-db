# gpu-database-engine

Bootstrap implementation workspace for the pre-NVIDIA phase.

## Current scope

- Replication-shaped local commit path
- WAL-before-visibility invariant tests
- WAL durability + queue watermarks (flushed, buffered, unflushed, pending depth/capacity, pending age/deadline), commit/apply/visibility lag gauges, explicit backlog/gap blocker flags, active transaction depth, and failover/admission readiness flags (`quiescent_for_failover`, `follower_promotion_ready`, `mutation_admission_saturated`)
- Engine truth surface via `Engine::status_snapshot()` for served snapshot identity/frontier, active fallback reasons + parity rollups, and replication/readiness health
- Engine telemetry snapshot + sink publication API for replication lag, durable WAL/snapshot frontier, write-path readiness/backlog state, and runtime metrics
- Thin MVCC execution slice via `Engine::execute_mvcc_query()` covering snapshot-bound full scan / key lookup / key-batch fan-in / value→key reference expansion / value→key→prefix source-preserving expansion / value→key→value→key source-preserving chaining / value→key→value→prefix source-preserving chained fan-out, optional key-prefix/range or source-aware key/value filtering, source-aware ordering, limit, and key/value or join-side source-value projection through the execution layer with explicit CPU fallback parity tracking (`GPU-123`)
- CPU-first reference engine skeleton
- Device-aware execution abstractions

## MVCC vertical slice (current bootstrap shape)

- Engine entry point: `Engine::execute_mvcc_query(&MvccReadQuery)`
- Supported sources:
  - `MvccReadSource::FullScan`
  - `MvccReadSource::KeyLookup { key }`
  - `MvccReadSource::KeyBatchLookup { keys }` (fan-in multi-source lookup; preserves request order before downstream filter/order/limit)
  - `MvccReadSource::FollowValueKeyRefs { keys }` (join-adjacent foreign-key-style expansion from seed row values to referenced keys)
  - `MvccReadSource::FollowValueKeyPrefixes { keys }` (prefix-driven foreign-key-style expansion from seed row values to visible target-key ranges)
  - `MvccReadSource::FollowValueKeyRefPrefixes { keys }` (source-preserving two-hop expansion: seed value -> referenced row -> prefix-driven target fan-out)
  - `MvccReadSource::FollowValueKeyRefValueKeyRefs { keys }` (source-preserving three-hop expansion: seed value -> referenced row -> referenced row value -> final referenced row)
  - `MvccReadSource::FollowValueKeyRefValueKeyPrefixes { keys }` (source-preserving chained fan-out: seed value -> referenced row -> referenced row value -> visible prefix-driven target expansion)
- Supported snapshot rule:
  - `visibility.read_txn_id` selects the MVCC snapshot frontier
- Supported filters:
  - `MvccReadFilter::KeyPrefix(prefix)`
  - `MvccReadFilter::SourceKeyPrefix(prefix)`
  - `MvccReadFilter::KeyRange { start_inclusive, end_exclusive }`
  - `MvccReadFilter::ValueEquals(value)`
  - `MvccReadFilter::SourceValueEquals(value)`
  - `MvccReadFilter::All([...])`
  - `MvccReadFilter::Any([...])`
- Supported projections:
  - `MvccProjection::KeyValue`
  - `MvccProjection::KeyOnly`
  - `MvccProjection::ValueOnly`
  - `MvccProjection::SourceKeyTargetValue` (join-adjacent result shape that mirrors the original seed key into `key` while projecting the resolved target row's value)
  - `MvccProjection::SourceValueOnly` (join-adjacent result shape that projects only the original seed row's value)
  - `MvccProjection::TargetKeySourceValue` (join-adjacent result shape that keeps the target key while projecting the original seed row's value)
- Result row shape:
  - `MvccReadRow { source_key, key, value }`
  - `source_key` is populated for join-adjacent expansion sources so source-preserving joins can keep seed provenance visible while join-side projections reuse the same engine-facing result contract.
- Supported ordering:
  - `MvccReadOrder::KeyAsc`
  - `MvccReadOrder::KeyDesc`
  - `MvccReadOrder::ValueAsc`
  - `MvccReadOrder::ValueDesc`
  - `MvccReadOrder::SourceKeyAsc`
  - `MvccReadOrder::SourceKeyDesc`
  - `MvccReadOrder::SourceValueAsc`
  - `MvccReadOrder::SourceValueDesc`
- Optional row cap:
  - `limit: Some(n)` applies after visibility + filter + ordering stages
- Current device strategy:
  - planned target: GPU default device
  - executed target: CPU reference semantics via `ScanOperator` → `FilterOperator` → `SortOperator` → `LimitOperator` → `ProjectOperator`
  - fallback reason: `FallbackReason::GpuMvccReadParityGap` (`GPU-123`)
- Deterministic workload fixture:
  - `tests/fixtures/mvcc-read-workload.txt`
- Next obvious extension boundary:
  - widen the join-adjacent slice further from the current source-preserving multi-hop reads plus seed-side/target-side mixed projection, filter, and ordering controls into richer relational composition (for example more explicit join-shape composition primitives) while preserving the same engine-facing contract and explicit fallback accounting on the eventual GPU-backed path.

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
