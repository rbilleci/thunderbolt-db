# Non-int4 point-index design inputs

This non-authoritative reference records durable constraints for **READ-002** and **R3-002**. It does not prescribe
type order, implementation phases, branches, or current entry points.

## Objective

Provide O(1)-class equality lookup for bigint, text, UUID, numeric, and composite keys when measurement shows an
index beats the GPU scan, without creating a host index as the product path.

## Common representation

Every key index separates:

1. **Candidate hash:** fixed, deterministic device/host hash agreement used to select buckets.
2. **Exact verification:** full typed equality on-device; a hash match is never sufficient.
3. **Row/version identity:** a result addresses a row in one captured residency generation.
4. **Visibility:** created/deleted/version rules are applied before a hit becomes a result or uniqueness conflict.

Fixed-width keys may store canonical bytes inline. Text stores hash plus offset/length into generation-owned bytes.
Composite keys hash canonical typed components and verify each component, including collation/type semantics.

## Correctness constraints

- NULL never equals NULL for ordinary equality; uniqueness follows the catalog's NULL policy.
- Signed integers, UUID byte order, numeric scale/canonicalization, and text byte/collation rules must match SQL.
- Duplicate needles and duplicate candidate hashes cannot reorder or duplicate results.
- A dead/old version cannot satisfy a read or uniqueness check; update self-exclusion is coordinate/version based.
- Index identity includes the payload generation/content identity, so compaction or republish cannot reuse stale
  row coordinates.
- Decline routes to the authoritative GPU scan, never CPU relational execution.

## Lifecycle and budget

- Build, retained bytes, resize/headroom, and temporary scratch are explicitly accounted per GPU.
- Sealed immutable sources build once per content generation. Mutable/open sources require incremental maintenance
  or a measured bounded rebuild policy.
- Optional index allocation failure preserves correctness and residency, then uses the GPU scan.
- Eviction, compaction, deauthorization, and generation retirement purge every associated index resource.
- Write-capable indexes use the same representation and visibility contract as read indexes; do not build separate
  read/write truths.

## Measurement gate

For each type/workload, compare index build cost, retained VRAM, probe latency/throughput, update maintenance, and
scan cost across cardinality, key width, selectivity, cache regime, and batch size. Do not extrapolate int4 results
to text or wide composites.

## Acceptance evidence

- Index result is byte-identical to the GPU scan over present, absent, NULL, collision, duplicate-needle, and
  versioned cases.
- Hash sabotage/collision tests prove exact verification is load-bearing.
- A nonzero route counter proves the index fired.
- Recovery and post-write reads preserve the same result set.
- Report-card and type-specific measurements justify keeping the index rather than the scan.

The dated proposal and its obsolete file/branch guidance are archived at
`../archive/design/non-int4-point-lookup-index-2026-06-30.md`.
