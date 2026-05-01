# Execution Interfaces (Draft)

## Operator
- `open(ctx)`
- `next(ctx) -> RowBatch|End`
- `close(ctx)`

## Device annotation
Each physical plan node must include:
- `device = cpu | gpu(device_id)`
- fallback policy

## BatchScheduler
- `enqueue(txn)`
- `flush(reason=count|time|admin)`
- `dispatch(batch) -> batch_id`

## Bootstrap MVCC read slice

- Engine-facing entry point: `Engine::execute_mvcc_query(&MvccReadQuery)`
- Supported sources:
  - `FullScan`
  - `KeyLookup { key }`
  - `KeyBatchLookup { keys }` (fan-in multi-source lookup; preserves request order before downstream filter/order/limit)
  - `Concat { sources }` (explicit composition primitive; recursively concatenates existing source shapes in source-list order before downstream filter/order/limit)
  - `ConcatDistinct { sources }` (ordered deduplicating composition primitive; recursively keeps the first occurrence of each resolved row while preserving distinct source-provenance rows)
  - `IntersectDistinct { sources }` (ordered intersection primitive; emits first-source rows whose exact resolved-row identity appears in every subsource, with source provenance treated as part of identity)
  - `IntersectAll { sources }` (ordered multiset intersection primitive; emits first-source rows up to the minimum exact resolved-row identity count across subsources, with source provenance treated as part of identity)
  - `ExceptDistinct { sources }` (ordered subtraction primitive; emits first-source rows whose exact resolved-row identity does not appear in any remaining subsource, with source provenance treated as part of identity)
  - `ExceptAll { sources }` (ordered multiset subtraction primitive; emits first-source rows after subtracting exact resolved-row identity multiplicity contributed by remaining subsources, with source provenance treated as part of identity)
  - `SymmetricDifferenceDistinct { sources }` (ordered unique-presence primitive; emits rows whose exact resolved-row identity appears in exactly one subsource, preserving first appearance order and treating source provenance as part of identity)
  - `SymmetricDifferenceAll { sources }` (ordered multiset symmetric-difference primitive; iteratively cancels exact resolved-row identity multiplicity source-by-source and emits the remaining imbalance in first appearance order, with source provenance treated as part of identity)
  - `FollowValueChain { keys, plan }` (generic source-preserving linear nested-join helper; follows `plan.value_key_hops` visible value→key hops from each visible seed row, then either returns the current row or expands visible keys matching the current row's value as a prefix)
  - `FollowValueKeyRefs { keys }` (join-adjacent foreign-key-style expansion; for each visible seed key in request order, look up the visible target row whose key matches the seed row's value)
  - `FollowValueKeyPrefixes { keys }` (prefix-driven foreign-key-style expansion; for each visible seed key in request order, expand all visible target rows whose keys share the seed row's value as a prefix)
  - `FollowValueKeyRefPrefixes { keys }` (source-preserving two-hop expansion; for each visible seed key in request order, look up the visible intermediate row whose key matches the seed row's value, then expand all visible target rows whose keys share the intermediate row's value as a prefix)
  - `FollowValueKeyRefValueKeyRefs { keys }` (source-preserving three-hop expansion; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> final visible target row)
  - `FollowValueKeyRefValueKeyPrefixes { keys }` (source-preserving chained fan-out; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> visible prefix-driven target expansion)
  - `FollowValueKeyRefValueKeyRefPrefixes { keys }` (source-preserving four-hop fan-out; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> visible referenced row -> visible prefix-driven target expansion)
  - `FollowValueKeyRefValueKeyRefValueKeyRefs { keys }` (source-preserving deeper terminal chain; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> visible referenced row -> visible referenced-row value -> visible referenced row -> final visible target row)
  - `FollowValueKeyRefValueKeyRefValueKeyPrefixes { keys }` (source-preserving deeper fan-out chain; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> visible referenced row -> visible referenced-row value -> visible referenced row -> visible prefix-driven target expansion)
  - `FollowValueKeyRefValueKeyRefValueKeyRefPrefixes { keys }` (source-preserving deeper nested fan-out chain; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> visible referenced row -> visible referenced-row value -> visible referenced row -> visible referenced-row value -> visible prefix-driven target expansion)
- Snapshot binding:
  - `MvccReadQuery.visibility.read_txn_id` chooses the MVCC snapshot used by storage visibility checks
- Filter layer:
  - `KeyPrefix(prefix)`
  - `SourceKeyPrefix(prefix)`
  - `KeyRange { start_inclusive, end_exclusive }`
  - `ValueEquals(value)`
  - `SourceValueEquals(value)`
  - `All([filter...])`
  - `Any([filter...])`
  - source-aware filter variants only match join-adjacent rows; scan/lookup-only shapes leave them empty rather than fabricating source state
- Order layer:
  - `KeyAsc`
  - `KeyDesc`
  - `ValueAsc`
  - `ValueDesc`
  - `SourceKeyAsc`
  - `SourceKeyDesc`
  - `SourceValueAsc`
  - `SourceValueDesc`
  - source-aware ordering variants sort empty/non-join rows deterministically using empty source fields and target key tie-breaks
- Projection layer:
  - `KeyValue`
  - `KeyOnly`
  - `ValueOnly`
  - `SourceKeyTargetValue` (join-adjacent projection that mirrors the original seed key into `key` while surfacing the resolved target row's value in `value`)
  - `SourceValueOnly` (join-adjacent projection that emits only the original seed row's value)
  - `TargetKeySourceValue` (join-adjacent projection that keeps the resolved target key in `key` while surfacing the original seed row's value in `value`)
- Result row contract:
  - `MvccReadRow { source_key, key, value }`
  - `source_key` is `None` for scan/lookup shapes and populated for join-adjacent expansion rows so source provenance survives the current CPU reference pipeline while join-side projections still reuse the same result contract.
- Optional row cap:
  - `limit = Some(n)` applies after visibility + filter + ordering stages
- Current execution/device contract:
  - planned target = `gpu(default_gpu_id)`
  - executed target = `cpu`
  - fallback reason = `GpuMvccReadParityGap` (`GPU-123`, owner=`execution`, milestone=`m0-bootstrap`)
  - CPU path runs an explicit `ScanOperator` → `FilterOperator` → `SortOperator` → `LimitOperator` → `ProjectOperator` pipeline as the reference semantics for the slice

## Current extension boundary

- Deterministic workload fixtures now include both `tests/fixtures/mvcc-read-workload.txt` for point-lookup/history replay and `tests/fixtures/mvcc-source-composition-workload.txt` for source-composition replay.

- Supported ordering now covers key-aware and value-aware shapes (`KeyAsc` / `KeyDesc` / `ValueAsc` / `ValueDesc`), with value ordering using key order as the deterministic tie-breaker.
- Supported range semantics are lexicographic `KeyRange { start_inclusive, end_exclusive }` filters.
- `KeyBatchLookup { keys }` is the first multi-source bootstrap shape; it fans multiple point lookups into the same execution pipeline while preserving request order until an explicit order clause overrides it.
- `Concat { sources }` is the first explicit source-composition primitive; it lets the engine concatenate heterogeneous scan/lookup/join-adjacent source shapes recursively without changing the row contract or fallback semantics.
- `ConcatDistinct { sources }` widens that surface with ordered deduplication while still preserving rows that remain semantically distinct because their source provenance differs.
- `IntersectDistinct { sources }` adds the first set-style composition helper; it keeps the first source's order but only emits rows whose full resolved identity also appears in every remaining subsource.
- `IntersectAll { sources }` is the first ordered multiset helper; it keeps the first source's order and multiplicity, capped by the minimum matching count across the remaining subsources.
- `ExceptDistinct { sources }` complements that set-style surface with ordered subtraction; it keeps the first source's order while removing rows whose full resolved identity appears anywhere in the remaining subsources.
- `ExceptAll { sources }` extends subtraction into multiset territory; it keeps the first source's order and multiplicity after subtracting the total matching count contributed by the remaining subsources.
- `SymmetricDifferenceDistinct { sources }` widens the same surface with an ordered unique-presence helper; it emits rows whose full resolved identity appears in exactly one subsource while preserving first appearance order across the source list.
- `SymmetricDifferenceAll { sources }` extends that into multiset territory via iterative source-by-source cancellation, preserving the remaining multiplicity imbalance and first appearance order.
- `FollowValueChain { keys, plan }` is the new composable linear nested-join surface; the older one-off deep join helpers now map onto specific hop-count + terminal combinations instead of requiring another enum variant for every deeper chain.
- `FollowValueKeyRefs { keys }` is the first join-adjacent bootstrap shape; it performs a deterministic two-stage value→key expansion while preserving request order, skipping missing seed/target rows, and then composes through the same filter/order/limit pipeline.
- `FollowValueKeyPrefixes { keys }` widens that join-adjacent slice into prefix-driven fan-out expansion while still preserving seed request order, lexicographic target order within each seed, and clean skip behavior for missing seeds or empty expansions.
- `FollowValueKeyRefPrefixes { keys }` is the first source-preserving two-hop join shape; it preserves the original seed provenance across a value→key lookup and then a prefix fan-out from the intermediate row without changing the engine-facing row contract.
- `FollowValueKeyRefValueKeyRefs { keys }` widens that source-preserving join surface into a deterministic three-hop chain, proving the current row contract can carry deeper relational composition without losing seed provenance or changing fallback semantics.
- `FollowValueKeyRefValueKeyPrefixes { keys }` widens the same surface into a deterministic chained fan-out shape, proving the engine-facing contract still holds when the final hop expands to multiple visible rows.
- `FollowValueKeyRefValueKeyRefPrefixes { keys }` pushes that same source-preserving surface one hop deeper into a value→key→value→key→prefix chain while keeping the original seed provenance and the same fallback semantics intact.
- `FollowValueKeyRefValueKeyRefValueKeyRefs { keys }` adds the deeper terminal sibling, proving the source-preserving surface can keep extending through another referenced-row hop and still end in one visible target row without changing the contract.
- `FollowValueKeyRefValueKeyRefValueKeyPrefixes { keys }` adds the matching deeper fan-out sibling, proving the same deeper chain can also terminate in visible prefix expansion without changing the contract.
- `FollowValueKeyRefValueKeyRefValueKeyRefPrefixes { keys }` still exists as a stable named helper, but it now resolves through the generic linear chain mechanism instead of bespoke one-off nested logic.
- The row contract now carries optional `source_key` provenance, which enabled the first true source-preserving join shape to land without another result-surface rewrite.
- Next obvious Q2 extension is widening the new generic linear nested-join surface into branch/fan-in plans or richer provenance controls without weakening the explicit GPU fallback contract.
