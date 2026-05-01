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
  - `FollowValueKeyRefs { keys }` (join-adjacent foreign-key-style expansion; for each visible seed key in request order, look up the visible target row whose key matches the seed row's value)
  - `FollowValueKeyPrefixes { keys }` (prefix-driven foreign-key-style expansion; for each visible seed key in request order, expand all visible target rows whose keys share the seed row's value as a prefix)
  - `FollowValueKeyRefPrefixes { keys }` (source-preserving two-hop expansion; for each visible seed key in request order, look up the visible intermediate row whose key matches the seed row's value, then expand all visible target rows whose keys share the intermediate row's value as a prefix)
  - `FollowValueKeyRefValueKeyRefs { keys }` (source-preserving three-hop expansion; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> final visible target row)
  - `FollowValueKeyRefValueKeyPrefixes { keys }` (source-preserving chained fan-out; for each visible seed key in request order, follow seed value -> visible intermediate row -> visible intermediate value -> visible prefix-driven target expansion)
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

- Supported ordering now covers key-aware and value-aware shapes (`KeyAsc` / `KeyDesc` / `ValueAsc` / `ValueDesc`), with value ordering using key order as the deterministic tie-breaker.
- Supported range semantics are lexicographic `KeyRange { start_inclusive, end_exclusive }` filters.
- `KeyBatchLookup { keys }` is the first multi-source bootstrap shape; it fans multiple point lookups into the same execution pipeline while preserving request order until an explicit order clause overrides it.
- `FollowValueKeyRefs { keys }` is the first join-adjacent bootstrap shape; it performs a deterministic two-stage value→key expansion while preserving request order, skipping missing seed/target rows, and then composes through the same filter/order/limit pipeline.
- `FollowValueKeyPrefixes { keys }` widens that join-adjacent slice into prefix-driven fan-out expansion while still preserving seed request order, lexicographic target order within each seed, and clean skip behavior for missing seeds or empty expansions.
- `FollowValueKeyRefPrefixes { keys }` is the first source-preserving two-hop join shape; it preserves the original seed provenance across a value→key lookup and then a prefix fan-out from the intermediate row without changing the engine-facing row contract.
- `FollowValueKeyRefValueKeyRefs { keys }` widens that source-preserving join surface into a deterministic three-hop chain, proving the current row contract can carry deeper relational composition without losing seed provenance or changing fallback semantics.
- `FollowValueKeyRefValueKeyPrefixes { keys }` widens the same surface into a deterministic chained fan-out shape, proving the engine-facing contract still holds when the final hop expands to multiple visible rows.
- The row contract now carries optional `source_key` provenance, which enabled the first true source-preserving join shape to land without another result-surface rewrite.
- Next obvious Q2 extension is wider relational composition beyond the current chained fan-out + seed-side projection/filter/order surface (for example additional mixed result-shape controls or more explicit join-shape composition primitives) without weakening the explicit GPU fallback contract.
