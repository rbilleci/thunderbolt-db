# gpu-database-engine

Bootstrap implementation workspace for the pre-NVIDIA phase.

## Current scope

- Replication-shaped local commit path
- WAL-before-visibility invariant tests
- WAL durability + queue watermarks (flushed, buffered, unflushed, pending depth/capacity, pending age/deadline), commit/apply/visibility lag gauges, explicit backlog/gap blocker flags, active transaction depth, and failover/admission readiness flags (`quiescent_for_failover`, `follower_promotion_ready`, `mutation_admission_saturated`)
- Engine truth surface via `Engine::status_snapshot()` for served snapshot identity/frontier, active fallback reasons + parity rollups, and replication/readiness health
- Engine telemetry snapshot + sink publication API for replication lag, durable WAL/snapshot frontier, write-path readiness/backlog state, and runtime metrics
- Thin MVCC execution slice via `Engine::execute_mvcc_query()` covering snapshot-bound full scan / key lookup / key-batch fan-in / explicit multi-source `Concat`, `ConcatDistinct`, `IntersectDistinct`, `IntersectAll`, `ExceptDistinct`, `ExceptAll`, `SymmetricDifferenceDistinct`, and `SymmetricDifferenceAll` composition / generic `FollowValueChain { keys, plan, provenance }` linear nested-join expansion / `FollowValueChainBranches { keys, plans, fan_in, provenance }` plus `FollowValueChainLabeledBranches { keys, branches, fan_in, provenance }` seed-grouped or first-non-empty branch expansion / legacy value→key reference helpers layered on that same chain mechanism / optional key-prefix/range or source-aware key/value or branch-label filtering, explicit multi-frame provenance filters/projection plus reusable frame-bundle controls on nested-join rows including exact ordered bundle-path equality, ordered bundle-subpath matching, anchored bundle-slice matching, counted bundle-membership thresholds, exact/ranged first/last/nth occurrence matching, exact/ranged same-subpath ordinal plus adjacent first/last occurrence-distance helpers, exact/ranged ordinal mixed-subpath occurrence-distance helpers, exact/ranged nth-offset matching, and first/last exact/ranged mixed-occurrence offsets, first/last mixed-subpath occurrence-distance helpers, and bundle-relative positional segment equality, source-aware ordering, limit, and key/value or join-side source-value/branch-label/provenance-value/provenance-path-summary projection through the execution layer with explicit CPU fallback parity tracking (`GPU-123`)
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
  - `MvccReadFilter::ProvenanceBundleKeyEquals { bundle, expected }`
  - `MvccReadFilter::ProvenanceBundleKeyCountAtLeast { bundle, expected, min_count }`
  - `MvccReadFilter::ProvenanceBundleKeyPrefix { bundle, prefix }`
  - `MvccReadFilter::BranchLabelEquals(label)`
  - `MvccReadFilter::KeyRange { start_inclusive, end_exclusive }`
  - `MvccReadFilter::ValueEquals(value)`
  - `MvccReadFilter::SourceValueEquals(value)`
  - `MvccReadFilter::ProvenanceValueEquals { frame, expected }`
  - `MvccReadFilter::ProvenanceBundleValueEquals { bundle, expected }`
  - `MvccReadFilter::ProvenanceBundleValueCountAtLeast { bundle, expected, min_count }`
  - `MvccReadFilter::ProvenanceBundleKeyValueEquals { bundle, key, value }`
  - `MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast { bundle, key, value, min_count }`
  - `MvccReadFilter::ProvenanceBundlePathEquals { bundle, summary, expected }`
  - `MvccReadFilter::ProvenanceBundlePathContains { bundle, summary, expected }` (ordered contiguous subpath matching over named key/value/`key=value` provenance bundles)
  - `MvccReadFilter::ProvenanceBundlePathCountAtLeast { bundle, summary, expected, min_count }` (ordered repeated-subpath counting over named key/value/`key=value` provenance bundles)
  - `MvccReadFilter::ProvenanceBundlePathPairAtDistance { bundle, summary, left, right, distance }` (relative-position matching for bundle segments separated by an exact distance)
  - `MvccReadFilter::ProvenanceBundlePathSuffixEquals { bundle, summary, expected }` (whole-bundle suffix matching over named key/value/`key=value` provenance bundles)
  - `MvccReadFilter::ProvenanceBundlePathPrefixEquals { bundle, summary, expected }` (whole-bundle prefix matching over named key/value/`key=value` provenance bundles)
  - `MvccReadFilter::ProvenanceBundlePathSliceEquals { bundle, summary, start, expected }` (anchored contiguous subpath matching at a chosen bundle-relative offset over named key/value/`key=value` provenance bundles)
  - `MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt { bundle, summary, start, expected }` (require the first ordered contiguous occurrence of a key/value/`key=value` subpath to begin at an exact bundle-relative offset)
  - `MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin { bundle, summary, start_min, start_max, expected }` (require the first ordered contiguous occurrence of a key/value/`key=value` subpath to begin within an inclusive bundle-relative offset range)
  - `MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt { bundle, summary, start, expected }` (require the last ordered contiguous occurrence of a key/value/`key=value` subpath to begin at an exact bundle-relative offset)
  - `MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin { bundle, summary, start_min, start_max, expected }` (require the last ordered contiguous occurrence of a key/value/`key=value` subpath to begin within an inclusive bundle-relative offset range)
  - `MvccReadFilter::ProvenanceBundlePathOccurrenceAt { bundle, summary, occurrence_index, start, expected }` (require an exact zero-based ordered contiguous occurrence of a key/value/`key=value` subpath to begin at an exact bundle-relative offset)
  - `MvccReadFilter::ProvenanceBundlePathOccurrenceWithin { bundle, summary, occurrence_index, start_min, start_max, expected }` (require an exact zero-based ordered contiguous occurrence of a key/value/`key=value` subpath to begin within an inclusive bundle-relative offset range)
  - `MvccReadFilter::ProvenanceBundlePathOccurrenceDistance { bundle, summary, left_occurrence_index, right_occurrence_index, distance, expected }` (require two exact zero-based ordered contiguous occurrences of the same key/value/`key=value` subpath to begin an exact distance apart)
  - `MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin { bundle, summary, left_occurrence_index, right_occurrence_index, min_distance, max_distance, expected }` (require two exact zero-based ordered contiguous occurrences of the same key/value/`key=value` subpath to begin within an inclusive distance range)
  - `MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistance { bundle, summary, distance, expected }` (require the first two ordered contiguous occurrences of the same key/value/`key=value` subpath to begin an exact distance apart)
  - `MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin { bundle, summary, min_distance, max_distance, expected }` (require the first two ordered contiguous occurrences of the same key/value/`key=value` subpath to begin within an inclusive distance range)
  - `MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance { bundle, summary, distance, expected }` (require the last two ordered contiguous occurrences of the same key/value/`key=value` subpath to begin an exact distance apart)
  - `MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin { bundle, summary, min_distance, max_distance, expected }` (require the last two ordered contiguous occurrences of the same key/value/`key=value` subpath to begin within an inclusive distance range)
  - `MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance { bundle, summary, left_occurrence_index, left_expected, right_occurrence_index, right_expected, distance }` (require two exact zero-based ordered contiguous occurrences of different key/value/`key=value` subpaths to begin an exact distance apart)
  - `MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin { bundle, summary, left_occurrence_index, left_expected, right_occurrence_index, right_expected, min_distance, max_distance }` (require two exact zero-based ordered contiguous occurrences of different key/value/`key=value` subpaths to begin within an inclusive distance range)
  - `MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistance { bundle, summary, left_expected, right_expected, distance }` (require the first ordered contiguous occurrences of two different key/value/`key=value` subpaths to begin an exact distance apart)
  - `MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin { bundle, summary, left_expected, right_expected, min_distance, max_distance }` (require the first ordered contiguous occurrences of two different key/value/`key=value` subpaths to begin within an inclusive distance range)
  - `MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance { bundle, summary, left_expected, right_expected, distance }` (require the last ordered contiguous occurrences of two different key/value/`key=value` subpaths to begin an exact distance apart)
  - `MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin { bundle, summary, left_expected, right_expected, min_distance, max_distance }` (require the last ordered contiguous occurrences of two different key/value/`key=value` subpaths to begin within an inclusive distance range)
  - `MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt { bundle, summary, left_occurrence_index, left_expected, left_start, right_occurrence_index, right_expected, right_start }` (require exact zero-based ordered contiguous mixed subpath occurrences to begin at exact bundle-relative offsets)
  - `MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin { bundle, summary, left_occurrence_index, left_expected, left_start_min, left_start_max, right_occurrence_index, right_expected, right_start_min, right_start_max }` (require exact zero-based ordered contiguous mixed subpath occurrences to begin within inclusive bundle-relative offset ranges)
  - `MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceAt { bundle, summary, left_expected, left_start, right_expected, right_start }` (require the first ordered contiguous mixed subpath occurrences to begin at exact bundle-relative offsets)
  - `MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin { bundle, summary, left_expected, left_start_min, left_start_max, right_expected, right_start_min, right_start_max }` (require the first ordered contiguous mixed subpath occurrences to begin within inclusive bundle-relative offset ranges)
  - `MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt { bundle, summary, left_expected, left_start, right_expected, right_start }` (require the last ordered contiguous mixed subpath occurrences to begin at exact bundle-relative offsets)
  - `MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin { bundle, summary, left_expected, left_start_min, left_start_max, right_expected, right_start_min, right_start_max }` (require the last ordered contiguous mixed subpath occurrences to begin within inclusive bundle-relative offset ranges)
  - `MvccReadFilter::ProvenanceBundlePathSegmentEquals { bundle, summary, index, expected }` (bundle-relative positional equality over named key/value/`key=value` provenance bundles)
  - `MvccReadFilter::ProvenanceBundleLenEquals { bundle, expected_len }` (exact whole-bundle cardinality checks over named provenance bundles)
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
  - `MvccProjection::TargetKeyProvenanceSummary { summary }` (join-adjacent result shape that keeps the resolved target key while projecting a lightweight provenance-path summary such as joined keys, joined values, or joined `key=value` hops)
  - `MvccProjection::TargetKeyProvenanceBundleSummary { bundle, summary }` (join-adjacent result shape that keeps the resolved target key while projecting a lightweight summary of a named provenance-frame bundle such as `SeedThroughTerminalInput` or `FullPath`)
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
  - `MvccReadOrder::ProvenanceBundleKeyPathAsc { bundle }`
  - `MvccReadOrder::ProvenanceBundleKeyPathDesc { bundle }`
  - `MvccReadOrder::ProvenanceBundleValuePathAsc { bundle }`
  - `MvccReadOrder::ProvenanceBundleValuePathDesc { bundle }`
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
  - widen the same-subpath first/last relative-distance surface from adjacent-pair helpers into first/last-to-explicit-ordinal distance helpers without regressing the same engine-facing contract and explicit fallback accounting on the eventual GPU-backed path.

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
