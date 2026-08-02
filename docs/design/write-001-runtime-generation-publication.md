# WRITE-001 runtime logical generation publication

This is the stable runtime design prerequisite for the WRITE-001 replay builder. It implements the
sole immutable database-generation builder required by
[write-001-codec5-semantics-v2.md](write-001-codec5-semantics-v2.md); codec 5 neither supplies
these roots nor defines an alternate root allocator. This document does not own work sequencing or
acceptance. [`PLAN.md`](../PLAN.md) is the sole task ledger.

## Scope and authority

There is one logical publication authority. It owns a versioned database *data* root, immutable
catalog identity, stable-table map, table/index manifests, transaction-status view, and the exact
physical resources needed by a captured reader. A GPU builder derives final typed table and index
artifacts; the durable typed envelope records the exact resulting identity for recovery comparison.
The authority is installed only at the canonical contiguous publication boundary.

The following are not logical identities and must never be used as a root or substituted for one:

- `CatalogSnapshot::commit_seq`;
- `RelationalResidencySnapshot::generation`;
- route tokens, `Arc` addresses, device pointers, GPU IDs, cache/index-map identity, or admission
  accounting; and
- shard-map geometry, compaction, residency placement, pressure state, or retained byte counts.

Physical admission, re-admission, compaction, GPU replacement, cache rotation, and GC may replace
the pinned resources of a publication only by creating a publication with the *same* logical
generation. A current logical table root commits exactly its current visible entity set: one leaf
per stable row ID. Historical versions are committed by the older publication objects that retain
them, not duplicated in a newer root. Thus removing an already unreachable physical version after
the active-snapshot fence is root-invariant. Logical DML, table reset, create/drop, and an index
truth change create a new affected manifest before publication. There is no optional or zero
runtime root form.

The host may schedule, retain, and compare bounded commitments as transaction/durability
control-plane metadata. It never scans, evaluates, materializes, or hashes relational data to
construct a content commitment: GPU kernels produce every changed typed value, row, index-entry,
and data-map node root. The host may only link already-authenticated immutable data nodes by the
requested fixed path, retain the resulting `Arc`s, and compare their roots.

## Stable identity v1

Root format version one introduces separate, nonzero, database-local `u64` stable table and index
ID allocators. A table/index CREATE consumes one ID before any candidate build; its typed catalog
record and checkpoint manifest carry both the display OID and the stable ID. Rename preserves the
stable ID. DROP never refunds it, and recreate consumes a new ID. The allocator high-waters are
part of the same catalog WAL/checkpoint authority as the object descriptors; this is not a second
WAL or a launch-time input.

Existing catalog objects are migrated exactly once by a persisted catalog migration map plus the
post-migration high-waters. Relation-shaped objects use
`(object_kind, display_oid) -> stable_id`; a legacy column uses the distinct key
`(Column, owner_display_table_oid, legacy_column_id:u32, attnum:i16) -> stable_column_id:u32`.
Recovery reads that map; it must never infer an ID from an OID, name, order, or a fresh allocator.
An absent, duplicate, zero, or mismatched migration map is durability corruption.

The existing nonzero catalog-stable column ID is the v1 stable column ID, widened to `u64` only at
the root boundary. It is already allocated and durably carried with its table descriptor; a column
drop never recycles it. A legacy column lacking that durable identity is migrated in the same
catalog migration map before any root-format-v1 publication.

The data-root identity is the exact nonzero `database_id` plus root-format version. Cluster,
timeline, and format-epoch lineage remain mandatory WAL/checkpoint/publication validation metadata
but are not data-root inputs: an identical database restored into a follower or PITR timeline has
the same data root.

## Canonical commitments

`D(domain, fields...)` is SHA-256 over the ASCII `domain`, preceded by its little-endian `u64`
byte length, followed by the fields printed in that exact order. Integers are little-endian; every
digest is exactly 32 bytes; a byte string is preceded by its printed length; and every repeated
field has its explicitly printed `u32` or `u64` count. There is no implicit list count or sort.
Every domain in this document is distinct from codec-5 domains.

The catalog identity is exactly the existing canonical-envelope pair
`(catalog_epoch:u64, catalog_digest:[u8;32])`: it is the exact before/after pair in the canonical
header and the same pair verified by the WAL/checkpoint catalog authority. `catalog_digest` is not
a new catalog root, does not enter `database_root`, and must not be recomputed from a runtime
snapshot. A catalog snapshot is publishable only after its descriptor has been checked against that
one existing identity.

The sole builder consumes a `RootFreeGenerationInput`. It is constructed before the canonical
typed envelope is sealed, from (a) the pinned predecessor publication identity, (b) the existing
canonical before/after catalog identities and stable descriptors, (c) the stable transaction and
claimed commit sequence, and (d) GPU-observed final row changes and index-membership changes. Its
affected-table list is stable-table-ID ascending; each table's row mutations are stable-row-ID
ascending; each row's columns are stable-column-ID ascending; and each index list is stable-index-ID
ascending.
An input uses only insert/replace/remove of the *current* row entity, exact typed values, and exact
index-inclusion verdicts. It contains no final table/index/database root or generation, expected
S4/S6/S7/S8 fact, SQL text, host row vector, device pointer, or caller-supplied root.

For resolved/preflight records, live construction is strictly: reserve neutral typed facts and GPU
work; derive the root-free input from the observed GPU result and the pinned predecessor; build the
private final manifests; append the exact final identities to the same canonical envelope; then
seal and durably append that one envelope. The frozen codec-5 typed INSERT form retains its
existing S7/root descriptor and exact 92-byte marker. Every root-format-v1 resolved/preflight
operation without that frozen S7 carrier instead uses the generic terminal descriptor extension
defined below; it is part of the one envelope, not another WAL record. Fresh recovery pure-decodes
the same neutral facts, reconstructs the same root-free input, runs the builder, and compares its
output with the sealed final identities before it may publish.

The accepted direct deterministic WAL-first class has a distinct, equally canonical two-stage form.
It first appends/fences only the already complete root-free *intent/image* fragments in
`WAL_FIRST_TERMINAL_ROOTS` mode; these are sufficient to reproduce the deterministic apply but are
not yet a `RootFreeGenerationInput`, have no final generation identity, and cannot authorize apply
or publication. Hidden deterministic GPU apply then combines those fenced fragments with its
observed terminal outcome to derive the root-free input and private manifests. Its one required
terminal outcome marker carries the exact final generation descriptor below, is appended/fenced,
and is the only record completion which can authorize the candidate. Recovery treats an incomplete
fragment prefix as unpublished, and derives the same root-free input from the fenced fragment set
plus the terminal outcome before comparison. A record that cannot yield an exact input or whose
operation class and marker form disagree fails closed. Neither form creates a second WAL or
lets a builder read a final identity in order to create it.

`runtime_generation_input_digest` has one runtime-v1 preimage for every write class; it does not
reuse the INSERT-only codec-v2 formula. The root-free terminal-outcome input is exactly
`(outcome_kind:u8, affected_rows:u64, sqlstate_present:u8, sqlstate_or_zero:[u8;5],
constraint_id:u64)`, where successful outcomes have an absent/zero SQLSTATE and zero constraint.
The top-level digest is:

```text
runtime_generation_input_digest = D(
  "gpu-db/runtime-generation/input/v1",
  database_id:[u8;16],
  catalog_before_epoch:u64,
  catalog_before_digest:[u8;32],
  catalog_after_epoch:u64,
  catalog_after_digest:[u8;32],
  stable_transaction_id:u64,
  commit_sequence:u64,
  initial_database_root:[u8;32],
  terminal_outcome_input,
  table_delta_count:u32,
  each table_delta_digest in strictly ascending stable_table_id)
```

This is a distinct terminal-descriptor field, not a renamed codec field. A typed INSERT that
also uses the frozen semantics-v2 S7 generation group must independently compute and validate its
existing `generation_input_digest` using the fixed `gpu-db/write001/generation-input/v2` preimage;
that S7 value remains in its S7 root descriptor and is never substituted by, translated from, or
placed into `runtime_generation_input_digest`. The generic runtime digest instead binds the
cross-operation recovery/publication input described here.

Each table delta has one of exactly five tags: `1 RowSet`, `2 CreateEmpty`, `3 Drop`, `4
ResetEmpty`, or `5 Rebuild`. `RowSet` carries only changed current rows and preserves the
predecessor's complete enrolled index-ID/shape set byte-for-byte; it may change an existing index
root only through those changed rows' membership updates. `Rebuild` carries the complete final
current-row set and is the explicit (non-W1) nonempty DDL/rewrite form. `CreateEmpty` and
`Drop` carry no row records; `ResetEmpty` proves the exact nonempty predecessor manifest and emits
the canonical empty row/index maps. Its digest is:

```text
table_delta_digest = D(
  "gpu-db/runtime-generation/table-input/v1",
  table_delta_kind:u8,
  stable_table_id:u64,
  before_table_present:u8,
  before_data_generation_or_zero:u64,
  before_table_root_or_zero:[u8;32],
  before_logical_row_count:u64,
  final_index_shape_count:u32,
  each final index in strictly ascending stable_index_id:
    stable_index_id:u64,
    index_shape_root:[u8;32],
    before_index_present:u8,
    before_index_generation_or_zero:u64,
    before_index_root_or_zero:[u8;32],
  row_input_count:u32,
  each row_input_digest in strictly ascending stable_row_id)
```

The tags constrain presence and membership coverage exactly. `CreateEmpty` has no before table and
zero rows; `Drop` has a before table and zero rows; and `ResetEmpty` has a before table, zero
final rows, and may establish any complete final empty index set. `RowSet` has a before table and
requires its final index-shape vector to have exactly the same count, stable IDs, and
`index_shape_root` values as the before manifest; every final index has
`before_index_present=1`. Its row list is exactly the changed current rows, and each row carries
membership input for every unchanged-shape final index. `Rebuild` has a before table and carries
every final current row exactly once, except that it may have an absent before table only for
CREATE-with-rows. It is mandatory for an index CREATE, DROP, or logical shape change on a nonempty
final table. A final index absent from the before manifest has zero before generation/root. The
final index-shape vector is always the complete final catalog set, so index CREATE/DROP/rebuild
cannot be inferred from a row action or from a physical index build.

```text
row_input_digest = D(
  "gpu-db/runtime-generation/row-input/v1",
  row_action:u8,                 // 1 insert, 2 replace, 3 remove, 4 rebuild-current
  stable_table_id:u64,
  stable_row_id:u64,
  before_current_row_leaf_or_zero:[u8;32],
  after_present:u8,
  after_created_by_or_zero:u64,
  column_count:u32,
  each column in strictly ascending stable_column_id:
    stable_column_id:u64,
    column_shape_root:[u8;32],
    canonical_typed_value_root:[u8;32],
  index_membership_count:u32,
  each final index in strictly ascending stable_index_id:
    stable_index_id:u64,
    before_index_entry_leaf_or_zero:[u8;32],
    after_present:u8,
    key_column_count:u32,
    each key ordinal ascending:
      stable_column_id:u64,
      column_shape_root:[u8;32],
      canonical_typed_value_root:[u8;32])
```

`insert` requires a zero before row and present after row whose `created_by == commit_sequence`;
`replace` requires nonzero before and present after row with that same created-by value; and
`remove` requires nonzero before, absent after, zero columns, and zero key components. An absent
index membership has zero key count; a present one must exactly match the corresponding final
`index_shape_root` key descriptors. `rebuild-current` is permitted only inside `Rebuild`; it has a
present after row with an exact nonzero carried `created_by`, and may use either a matching nonzero
before leaf (rewrite of an existing entity) or zero before leaf with
`created_by == commit_sequence` (CREATE-with-rows). The `Rebuild` table action authorizes
replacement of the whole map; all other tags reject a row form outside these rules. These initial
row/index leaves are acquired only from the predecessor publication. No
table/index/database final root or final generation occurs anywhere in this preimage.

For every current catalog column, the builder derives this catalog-bound shape from the descriptor
already authenticated by `catalog_digest`:

```text
column_shape_root = D(
  "gpu-db/runtime-generation/column-shape/v1",
  stable_table_id:u64,
  stable_column_id:u64,
  attnum:i16,
  SQL_storage_type:[u8;4],
  declared_type_oid:u32,
  signed_type_size:i16)
```

`canonical_typed_value_root` is
`D("gpu-db/runtime-generation/typed-value/v1", column_shape_root:[u8;32], null_flag:u8,
logical_value_length:u32, exact_logical_value_bytes)`. `null_flag` is `0` for a value and `1` for
NULL; NULL has length zero. Non-NULL bytes are exactly the `s7-final-row/v2` typed-value grammar:
little-endian i32 for INT2/INT4/DATE, i64 for INT8/TIMESTAMP, two's-complement little-endian i128
for NUMERIC, raw 16 bytes for UUID, one byte `0`/`1` for BOOL, and raw UTF-8 for TEXT. The GPU
produces these values and roots from the device table; neither image names nor a physical vector
offset enters this root.

The one current entity leaf is:

```text
current_row_leaf = D(
  "gpu-db/runtime-generation/current-row/v1",
  stable_table_id:u64,
  stable_row_id:u64,
  created_by:u64,
  column_count:u32,
  each column in strictly ascending stable_column_id:
    stable_column_id:u64,
    column_shape_root:[u8;32],
    canonical_typed_value_root:[u8;32])
```

For one table, the current-entity map is a fixed-height 64-bit radix tree keyed by stable row ID.
The root has depth zero, leaves have depth 64, and at internal depth `d` bit `63-d` chooses left for
zero and right for one. Its exact nodes are:

```text
row_empty[64] = D("gpu-db/runtime-generation/row-empty-leaf/v1",
                  root_format_version:u16, stable_table_id:u64, subtree_count:u64=0)
row_empty[d] = D("gpu-db/runtime-generation/row-empty-node/v1",
                 root_format_version:u16, stable_table_id:u64, depth:u8,
                 subtree_count:u64=0, row_empty[d+1]:[u8;32], row_empty[d+1]:[u8;32])
row_leaf = D("gpu-db/runtime-generation/row-leaf/v1",
             root_format_version:u16, stable_table_id:u64, stable_row_id:u64,
             subtree_count:u64=1, current_row_leaf:[u8;32])
row_node = D("gpu-db/runtime-generation/row-node/v1",
             root_format_version:u16, stable_table_id:u64, depth:u8,
             subtree_count:u64, left_root:[u8;32], right_root:[u8;32])
```

For `row_node`, `subtree_count` is the checked sum of its two child counts; `row_empty[0]` is the
empty row-map root. An insert/replace/remove changes only the 64-node path and its GPU-produced
hashes. A zero ID, wrong depth, duplicate leaf, count mismatch, or removal of an absent leaf is
corruption. The table's logical row count is the root node's count. Previous published row maps,
not a deleted version in the current map, supply an old reader's MVCC view; therefore safe physical
GC does not modify a current row map.

Each enrolled index has an independently persistent 64-bit membership map keyed by the same stable
row ID. Its `index_shape_root` has this exact preimage, with keys in dense increasing key ordinal:

```text
index_shape_root = D(
  "gpu-db/runtime-generation/index-shape/v1",
  owner_stable_table_id:u64,
  stable_index_id:u64,
  index_flags:u32,
  null_equality_policy:u8,
  membership_predicate_root_or_zero:[u8;32],
  key_column_count:u32,
  each key ordinal 0..key_column_count:
    stable_column_id:u64,
    column_shape_root:[u8;32])
```

An unsupported or noncanonical membership predicate rejects before the builder; a supported
predicate's inclusion verdict is GPU-produced and is part of the root-free input. An entry leaf is:

```text
index_entry_leaf = D(
  "gpu-db/runtime-generation/index-entry/v1",
  owner_stable_table_id:u64,
  stable_index_id:u64,
  index_shape_root:[u8;32],
  stable_row_id:u64,
  created_by:u64,
  key_column_count:u32,
  each key ordinal 0..key_column_count:
    stable_column_id:u64,
    column_shape_root:[u8;32],
    canonical_typed_value_root:[u8;32])
```

The index membership radix grammar is identical in depth and bit order to the row map but uses
the following domains and binds the owner/index IDs in every preimage:

```text
index_empty[64] = D("gpu-db/runtime-generation/index-empty-leaf/v1",
                    root_format_version:u16, owner_stable_table_id:u64, stable_index_id:u64,
                    index_shape_root:[u8;32], subtree_count:u64=0)
index_empty[d] = D("gpu-db/runtime-generation/index-empty-node/v1",
                   root_format_version:u16, owner_stable_table_id:u64, stable_index_id:u64,
                   index_shape_root:[u8;32], depth:u8, subtree_count:u64=0,
                   index_empty[d+1]:[u8;32], index_empty[d+1]:[u8;32])
index_leaf = D("gpu-db/runtime-generation/index-leaf/v1",
               root_format_version:u16, owner_stable_table_id:u64, stable_index_id:u64,
               index_shape_root:[u8;32], stable_row_id:u64, subtree_count:u64=1,
               index_entry_leaf:[u8;32])
index_node = D("gpu-db/runtime-generation/index-node/v1",
               root_format_version:u16, owner_stable_table_id:u64, stable_index_id:u64,
               index_shape_root:[u8;32], depth:u8, subtree_count:u64,
               left_root:[u8;32], right_root:[u8;32])
```

The same checked-count, zero-ID, depth, duplicate, and absent-remove rules apply. This map commits
each row's canonical key tuple without sorting all keys; the physical GPU key index is a resource
inside the publication and must match this logical map, but its geometry never enters a root.
`index_content_root` is `index_empty[0]` or the index-map root, and its checked count is the index
entry count. An index root is:

```text
index_root = D(
  "gpu-db/runtime-generation/index-root/v1",
  owner_stable_table_id:u64,
  stable_index_id:u64,
  index_generation:u64,
  index_shape_root:[u8;32],
  index_entry_count:u64,
  index_content_root:[u8;32])
```

An index manifest vector is stable-index-ID ascending and contains every enrolled index. A
zero-effect inherited index reuses its exact prior map, count, manifest, and root; physical index
build, rebuild, eviction, and cache replacement do not alter it. A table manifest is:

```text
table_root = D(
  "gpu-db/runtime-generation/table-root/v1",
  stable_table_id:u64,
  data_generation:u64,
  logical_row_count:u64,
  current_row_map_root:[u8;32],
  index_count:u32,
  each stable-index-ID ascending:
    stable_index_id:u64, index_generation:u64, index_root:[u8;32])
```

`data_generation` is nonzero. It advances when its current-row map or an enrolled index root
changes, and is preserved by no-ops, catalog-only commits, admission, compaction, safe GC, memory
pressure, and route-cache rotation. The candidate carries explicit before/after table and index
generations/roots; it never infers a successor with `+1`. On a changed table or index, the final
generation is exactly the already-claimed nonzero `commit_sequence`; an unchanged table/index
retains its prior generation. A resolved/preflight envelope records those exact before/after
manifests and stable-ID allocator high-waters before durability; the WAL-first terminal descriptor
records the same identities before its complete-marker fence. Recovery recomputes the GPU artifacts
then compares every recorded root/generation before publication.

A catalog-only or name-only transition changes the separate canonical catalog identity but preserves
every table data root and the database data root. A table DROP removes its data manifest; its old
`Arc` remains pinned by older publications/readers until retirement.

## Persistent database map and publication

The database table map is a persistent, fixed-height binary radix tree keyed by the 64 bits of a
stable table ID, read most-significant bit first. The root has depth zero; a leaf is at depth 64.
For root-format version `1` and `database_id`, define:

```text
empty_leaf = D("gpu-db/runtime-generation/map-empty-leaf/v1",
                root_format_version:u16, database_id:[u8;16])
empty_node[64] = empty_leaf
empty_node[d] = D("gpu-db/runtime-generation/map-empty-node/v1",
                  root_format_version:u16, database_id:[u8;16], depth:u8,
                  empty_node[d+1]:[u8;32], empty_node[d+1]:[u8;32])
leaf = D("gpu-db/runtime-generation/map-leaf/v1",
         root_format_version:u16, database_id:[u8;16], stable_table_id:u64,
         table_root:[u8;32])
node = D("gpu-db/runtime-generation/map-node/v1",
         root_format_version:u16, database_id:[u8;16], depth:u8,
         left_root:[u8;32], right_root:[u8;32])
```

`empty_node[0]` is the exact empty-map root. A leaf path uses bit `63 - depth` at internal depth
`depth`; zero selects left and one selects right. Two leaves with the same stable table ID, any
zero stable ID, a duplicate manifest, an absent remove target, or a node at the wrong depth is
corruption. Updating, inserting, or deleting one table copies only its root-to-leaf path and
reuses every unchanged child `Arc`; the implementation must not rebuild an O(all-table) map merely
to change one table. The GPU builder produces the changed leaf/path roots and final database root;
the host only checks their requested paths and links their immutable node ownership.

The database root is:

```text
database_root = D(
  "gpu-db/runtime-generation/database-root/v1",
  root_format_version:u16,
  database_id:[u8;16],
  persistent_table_map_root:[u8;32]
)
```

### Terminal outcome and status-view closure

The existing canonical terminal outcome marker remains the one physical/logical record terminator.
Its format gains a versioned generation-descriptor extension; this is an extension of that marker,
not a fragment, a second terminal marker, or a second WAL. It does **not** consume a bit in
`CanonicalPreApplyHeader.flags`: the current codec owns both bits 31 and 30 and its content bits,
and this slice preserves those header bytes exactly.

The mode is carried only by the terminal marker's exact body grammar. It is either the legacy
`CanonicalOutcomeV1` body of exactly 92 bytes, or it is exactly:

```text
canonical_outcome_v1:[u8;92],
extension_magic:[u8;16] = b"GPUDBGENROOT1\0\0\0",
extension_version:u16 = 1,
extension_reserved:u16 = 0,
descriptor_bytes:u32,
generation_terminal_descriptor_v1:[u8;descriptor_bytes]
```

There is no trailing byte. The decoder first exact-decodes the 92-byte outcome prefix. A body of
exactly 92 bytes is the existing legacy form; an otherwise longer body must match this extension
grammar exactly, including its magic, version, zero reserved field, and declared length. The
existing exact `CanonicalOutcome` decoder consequently continues to reject an extension when
called on the complete marker body. The versioned envelope decoder recognizes the extended form
only for a root-format-v1 operation whose semantic fragment decoder permits the generic descriptor;
its descriptor `publication_form` below selects prebuilt-resolved or direct-WAL-first behavior.
The frozen codec-5 typed INSERT form requires its legacy 92-byte marker and S7 carrier; other
root-format-v1 operations require the extension. A form/body mismatch fails closed. An incomplete
fragment prefix has no marker and is unpublished.

Legacy markers retain the current final digest exactly:
`D("gpu-db/adr014/final-outcome/v1", exact_header_bytes, ordered_fragment_root,
canonical_outcome_v1)`. An extended marker has
`D("gpu-db/runtime-generation/terminal-envelope/v1", exact_header_bytes,
ordered_fragment_root, complete_exact_marker_body)`. The matching digest is the marker final leaf;
thus every extension byte, including the mode discriminator and descriptor length, is authenticated
without repurposing outer flags. `CanonicalEnvelope` and checkpoint recovery must expose which
format was decoded and calculate the matching formula; the legacy encoder/decoder stays byte-for-byte
compatible.

The generic extended marker has one descriptor and two exact forms:
`publication_form=1 PrebuiltResolved` for any root-format-v1 resolved/preflight operation that
does not have the frozen typed-INSERT S7 carrier, and
`publication_form=2 WAL_FIRST_TERMINAL_ROOTS` for the direct deterministic root-free operation.
Both carry this exact descriptor; only form 2 permits the fragment prefix to be fenced before the
final roots exist. The frozen typed INSERT keeps its presealed S7/root descriptor and exact legacy
92-byte terminal body, so it cannot accidentally acquire a second generic descriptor. The terminal
descriptor is:

```text
GenerationTerminalDescriptorV1 {
  root_descriptor_version:u16 = 1,
  publication_form:u8,          // 1 PrebuiltResolved, 2 WAL_FIRST_TERMINAL_ROOTS
  descriptor_reserved:[u8;5] = 0,
  initial_database_root:[u8;32],
  catalog_before_epoch:u64,
  catalog_before_digest:[u8;32],
  catalog_after_epoch:u64,
  catalog_after_digest:[u8;32],
  stable_transaction_id:u64,
  commit_sequence:u64,
  runtime_generation_input_digest:[u8;32],
  final_database_root:[u8;32],
  table_transition_count:u32,
  each stable-table-ID ascending:
    table_delta_kind:u8,
    stable_table_id:u64,
    table_before_present:u8,
    table_after_present:u8,
    table_reserved:[u8;5] = 0,
    data_generation_before_or_zero:u64,
    initial_table_root_or_zero:[u8;32],
    initial_logical_row_count_or_zero:u64,
    data_generation_after_or_zero:u64,
    final_table_root_or_zero:[u8;32],
    final_logical_row_count_or_zero:u64,
    before_index_count:u32,
    each stable-before-index-ID ascending:
      stable_index_id:u64,
      before_index_shape_root:[u8;32],
      index_generation_before_or_zero:u64,
      initial_index_root_or_zero:[u8;32],
    after_index_count:u32,
    each stable-after-index-ID ascending:
      stable_index_id:u64,
      after_index_shape_root:[u8;32],
      index_generation_after_or_zero:u64,
      final_index_root_or_zero:[u8;32]
}
```

The descriptor serialization is the literal little-endian concatenation of those fields, with the
printed `u32` counts and no alignment, alternate order, duplicate table/index, or trailing byte.
The presence convention is exact: a missing before/after table has every scalar on that side zero
and its corresponding index count zero. `CreateEmpty` is `before_present=0, after_present=1`
with an empty final row map; `Rebuild` is `1,1`, except that its explicitly admitted
CREATE-with-rows case is `0,1`; `Drop` is `1,0`; and `RowSet` and `ResetEmpty` are `1,1`.
`before_index_count` enumerates only the indexes owned by the table before the operation, while
`after_index_count` enumerates only its final owned indexes; neither is a union and their
independent shape roots make CREATE and DROP observable. An absent index generation/root is zero.
Present roots, generations, IDs, row counts, and shape roots obey their existing nonzero manifest
rules. The table/index vectors must agree with the descriptor's affected-object closure.

The root-free input derives `runtime_generation_input_digest` without final roots; recovery
recomputes it and every descriptor field. For a non-success terminal outcome, every final identity
equals its initial identity and the root-free mutation set is empty. The implementation must version
the canonical marker decoder/encoder and checkpoint compatibility at the same time; legacy marker
bytes remain legacy and never impersonate this form.

The publication-owned `PublishedStatusIndex` is itself an authenticated immutable COW map keyed by
the 64 bits of `stable_transaction_id`, with the same MSB-first/depth-zero-to-64 path rule. Its leaf
binds the terminal result rather than the earlier pending claim:

```text
status_entry_leaf = D(
  "gpu-db/runtime-generation/status-entry/v1",
  root_format_version:u16,
  database_id:[u8;16],
  stable_transaction_id:u64,
  request_digest:[u8;32],
  commit_sequence:u64,
  outcome_kind:u8,
  affected_rows:u64,
  sqlstate_present:u8,
  sqlstate_or_zero:[u8;5],
  constraint_id:u64,
  target_digest:[u8;32],
  returning_digest:[u8;32],
  terminal_envelope_digest:[u8;32])

status_empty[64] = D("gpu-db/runtime-generation/status-empty-leaf/v1",
                     root_format_version:u16, database_id:[u8;16], subtree_count:u64=0)
status_empty[d] = D("gpu-db/runtime-generation/status-empty-node/v1",
                    root_format_version:u16, database_id:[u8;16], depth:u8,
                    subtree_count:u64=0, status_empty[d+1]:[u8;32], status_empty[d+1]:[u8;32])
status_leaf = D("gpu-db/runtime-generation/status-leaf/v1",
                root_format_version:u16, database_id:[u8;16], stable_transaction_id:u64,
                subtree_count:u64=1, status_entry_leaf:[u8;32])
status_node = D("gpu-db/runtime-generation/status-node/v1",
                root_format_version:u16, database_id:[u8;16], depth:u8, subtree_count:u64,
                left_root:[u8;32], right_root:[u8;32])
```

`status_empty[0]` is the genesis status-view root; every node count is the checked child-count sum.
Duplicate transaction IDs, zero IDs, an invalid terminal outcome shape, wrong depth/count, or a
remove before the checkpoint/retry retention proof is corruption. A retention-prune candidate is a
normal successor with the checked checkpoint proof and new status-view root; it never changes a
database data root. Status is control-plane metadata, so the host may hash/link this map only from
the sealed terminal envelope; it cannot synthesize an outcome, retry result, or relational root.

The canonical publication object is an immutable `Arc<PublicationGeneration>` stored in one
`ArcSwap`. Its logical identity is:

```text
PublicationIdentity {
  visible_next: u64,
  database_root: [u8;32],
  catalog_epoch: u64,
  catalog_digest: [u8;32],
  status_covered_through: u64,
  status_view_root: [u8;32],
  last_terminal_envelope_digest_or_zero: [u8;32]
}
```

`visible_next` is the first uncovered commit index and
`status_covered_through == visible_next - 1`. The last terminal-envelope digest is zero only at
genesis; otherwise it is the complete canonical final-envelope digest for that exact terminal
outcome. `status_view_root` commits every retained terminal retry result through that boundary, not
the earlier pending `TransactionClaimStatus` claim.
The object contains the immutable COW `PublishedStatusIndex` authenticated by `status_view_root`,
the catalog snapshot checked against `catalog_epoch/digest`, all table/index manifests, and the
exact resident shard map, single-buffer entries, cold artifacts, sidecars, and index allocations
that realize those identities. The durable WAL remains the status authority; this pinned immutable
index is its publication-covered lookup view and may only be pruned under the existing
checkpoint/retry retention proof. It is neither a data-root input nor a second status WAL.

`publication_epoch` advances on every object install, including a placement-only resource
replacement, and is not a logical-root input. There is no separately published visibility scalar
with consistency authority. A legacy scalar may remain only as a post-install derived telemetry
mirror and no reader may pair it with independently loaded catalog, status, shard, sidecar, or
index state. Readers acquire and retain this one object for the full device submission. A
placement-only replacement revalidates the complete current `PublicationIdentity` under the
publication owner, installs a new resource epoch with unchanged logical identity, and cannot
replace a newer publication.

The coordinator retains `ReadyPublicationCandidate`s, not a bare `BTreeSet<Index>`. Each candidate
owns its exact `terminal_index`, predecessor `PublicationIdentity`, predecessor publication epoch
and retained `Arc`, final manifests/physical resources, catalog snapshot/identity, and exact
terminal outcome, envelope digest, and status-map root. Before WAL sealing, the one generation
sequencer links candidates in commit-index order: candidate `n` may use only the installed object
for `n - 1` or that immediate predecessor's private final identity. It cannot seal until that
predecessor identity is known. A failed or abandoned pre-durable candidate invalidates its entire
unsealed suffix; a candidate never silently rebases or receives a new root under the same sealed
envelope. If a placement-only install changes the retained predecessor instance, the candidate must
revalidate/re-materialize its resource layer under the publication owner; it may not overwrite a
newer instance blindly.

On durable-and-applied completion, `engine_commit_coordinator` stores the candidate by index. It
may install only the candidate at `visible_next` and only when the candidate's full predecessor
identity byte-equals the current `ArcSwap` object. A missing, stale, or mismatched predecessor
wedges the commit path for recovery; it must not skip, fold, or relabel a root. The single swap
installs data, catalog, indexes, status, resources, and the next visibility boundary together.
An abort/no-op therefore advances visibility, terminal status coverage, and publication epoch with
the prior data/catalog identity; a catalog-only transition changes only catalog identity; a
successful multi-table transaction privately constructs all changed paths and performs one
replacement. Readers observe either the old object or the complete new object.

## Recovery, checkpoints, and replay

Genesis creates the exact empty persistent table-map root and database root from `database_id`.
Recovery begins from an authenticated checkpoint *logical* catalog/data/status manifest and root,
not from a process-local runtime publication object. It rebuilds the immutable status lookup from
the checkpoint plus each complete ordered terminal envelope, derives each root-free input, and
rebuilds fresh GPU artifacts privately. Every candidate must match its sealed predecessor,
table/index/database roots, catalog pair, terminal envelope digest, and status-view root before
recovery installs one new runtime publication object. A checkpoint stores stable-ID
migration/allocator state plus the logical generation manifests, catalog identity, status retention
boundary, and roots; physical GPU allocations are reconstructible and never root inputs.

The recovery prefix cannot expose a publication root above the common durable-and-applied prefix.
Any mismatch among checkpoint manifest, replayed predecessor/root, catalog identity, status
coverage, table/index root, or logical row count is durability corruption and fails closed.

### Quiescent bootstrap and nonempty rebuild

The first runtime construction path for a nonempty v1 generation begins at a private sealed
`BootstrapPublicationSource -> BootstrapMaterializationLease` boundary. Its internal validated
replay owner is a `ValidatedBootstrapSource` held in a `BootstrapMaterializationSeed`; neither is
an uninstalled generation. It may run only during fresh recovery/startup before service exposure,
or after a complete reader, writer, and GPU drain.
It binds one common durable-and-applied cut `C`, `visible_next = C + 1`, database/root-format
identity, the exact WAL catalog epoch/digest pair, and the catalog snapshot produced by that same
replay. It also owns the persisted v1 stable-ID migration map and allocator high-waters, the
complete canonical terminal envelopes through `C`, and the physical-resource provenance required
to reconstruct every current table. Its sealed checkpoint-manifest companion carries the expected
database and status roots and, for each current stable table and index, the exact logical row
count, data/index generation, table/index root, index-shape root, and ordered key descriptors.
The materializer compares its GPU-produced artifacts against these facts; it may not derive a
generation from `C`, a cache generation, or a current residency object. A source is sealed before
materialization; it must not reload the catalog, status, MVCC, residency, shard, cold, sidecar, or
index state from independently published live maps.

The bootstrap materializer GPU-rehydrates every current table, including cold or nonresident
ones, and bulk-builds the canonical typed values, current-row leaves, index memberships, row and
index maps, table manifests, table map, status view, and database root. It returns a complete,
future-COW-capable but uninstalled generation: checked catalog identity and stable-ID state,
terminal status view, table/index proof store, and owned resident/cold/sidecar/index resources
are retained together. A collection of final roots without those persistent map paths is not a
valid bootstrap result, because a later `RowSet` could not authenticate its predecessor. Host code
may retain and compare authenticated commitments and resources, but may not hash relational rows
or relabel existing cache/residency identities as v1 roots. The source and uninstalled result have
no reader, `ArcSwap`, WAL, recovery callback, or publication-install API; the later single-swap
cutover consumes only a fully validated result.

## Required implementation boundary

The production owner is a private `engine_data_generation` module. It is the only constructor for
logical table/index/database generation manifests and the only place that derives their roots.
`engine_commit_coordinator` owns the atomic publication-object swap; `engine_state` exposes
acquisition only. Canonical live apply, transaction commit, concurrent lanes, DDL/reset/drop,
residency/index lifecycle, checkpoint, and recovery all feed this owner through their existing
sole WAL/apply/publication authority. No replay-specific or codec-specific root builder exists.

`ReplayBaseGenerationPin` is a later consumer only. It will acquire one published object, validate
the S7 initial table/database identities against it, and separately pin exact GPU resources. It
cannot create, modify, or publish a logical generation.

## Required evidence

This evidence is a dimension of the integrated WRITE-001 milestone candidate, not a separately accepted
publication slice. Focused tests may close it during implementation; WRITE-001 receives the single final
audit/HAZARD/report-card seal defined by `PLAN.md` and `AGENTS.md`.

The implementation must demonstrate deterministic live-versus-fresh-recovery roots; exact domain,
count, ordering, type/index-shape, and migration-map sabotage rejection; bounded
O(changed_rows * (64 + affected_indexes * 64)) device root work with no all-table scan for
`RowSet` (and O(final_rows * (64 + final_indexes * 64)) only for its explicitly complete
`Rebuild` DDL/rewrite form); changed data/index roots; no-op and catalog-only behavior;
placement/compaction/safe-GC invariance; create/drop/recreate separation; root/catalog/status/shard/sidecar/index/GPU substitution refusal;
predecessor-chain refusal for missing/stale/out-of-order candidates; multi-table atomic reader
visibility; retained old-generation lifetime; pending-claim versus terminal-outcome/status-root
substitution refusal; WAL-first incomplete-prefix refusal plus direct WAL-first live/recovery final
descriptor equality; crash-prefix refusal; actual-GPU NULL differential; and three serial plus two
concurrent HAZARD cohorts. The read-result/residency path changes, so the applicable quick screen
and frozen canonical full report card are required after independent audit.
