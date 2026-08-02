# WRITE-001 codec-5 semantics-v2 boundary

This is a stable ownership and semantic boundary beneath
[`write-001-general-insert-pipeline.md`](write-001-general-insert-pipeline.md). It is not a second
task ledger; [`../PLAN.md`](../PLAN.md) alone owns unfinished work and sequencing.

Only sections explicitly labelled **normative wire** below are byte-stable. S7 is now frozen for
one permanently closed typed-INSERT-only semantics-v2 profile. S8 and the final replay material
remain a requirements inventory, **not** a wire format. No implementation may infer S8/replay
tags, widths, domains, or payload bytes from that inventory.

Codec-5 format 1 / semantics 1 remains an inert historical/test form. No semantics-v2 writer is
eligible until its complete reader, replay IR, durable sequence boundary, GPU replay compiler,
and pre-WAL resource envelope are accepted together.

## Canonical scalar and digest rules

All integers are little-endian and unsigned unless marked otherwise. Every reserved byte is zero.
An absent `u32` reference is `0xffff_ffff`. A canonical digest written as
`D(domain, fields...)` is:

```text
SHA-256(
    little_endian_u64(byte_length(domain))
    || domain
    || fields in the stated order
)
```

There is no implicit field length, separator, terminator, or role byte. A field contributes its
exact fixed-width bytes or its explicitly length-delimited grammar. Empty variable data
contributes zero bytes after its declared length.

## Normative wire: semantics-v2 profile and S4 identity

Aggregate semantics value `2` selects the exact typed-INSERT-only S7 profile below, but remains
ineligible for any live writer, recovery, apply, or publication caller. The existing 96-byte
aggregate header and eight section headers remain its outer framing. The aggregate header's
`allocator_before` and `allocator_high_water`, plus the outer
`CanonicalPreApplyHeader.allocator_high_water`, are canonical zero sentinels. Stable row identity
is table-local under ADR-014. S7 table-block ranges are defensive transaction-local selection and
replay cross-checks only: every selected ID must already be covered by a separately
marker-committed, durable, published `AllocatorLease` witness under the same database/table
allocator identity. The lease remains the sole allocator authority and consumes unused or aborted
members. Semantics 1 retains its existing nonzero global allocator range and byte-for-byte
behavior.

The 96-byte stream header carries format `1`, semantics `2`, minimum reader `1`, maximum reader
`1`, and section count `8`; its magic, widths, field offsets, chunk format byte, section tags,
section headers, chunk/root trailer, and STATUS2 framing are otherwise unchanged.

Semantics 2 is permanently closed to typed INSERT:

- S3 has entry count and payload length zero.
- Every S1 family is typed INSERT (`1`); its family ordinal equals its statement ordinal.
- S1, S2, and S6 entry counts all equal the aggregate statement and INSERT-statement counts.
- Every S6 semantic class is typed INSERT (`1`).
- Aggregate catalog/reset/private-sequence flags and outer catalog/reset/rewrite/private-sequence
  content bits are zero. S5 contains only published sequence references.
- S7 has exactly one entry; S8 remains separately design-gated.
- Unknown, non-INSERT, COPY, opaque, replacement, deletion, private-sequence, catalog, reset, or
  rewrite tags reject. A later semantics version may add fully typed classes; semantics 2 is never
  widened.

The aggregate statement count is nonzero. The table-block count is nonzero and equals S7's table
count. As required by the unchanged canonical envelope, outer operation count equals the physical
fragment count (aggregate chunks plus STATUS2); the outer table-block count equals the
aggregate/S7 value, and the outer stable transaction ID equals the aggregate ID. Outer/status
request identity and the terminal aggregate root are separately closed below. Catalog echoes and
the zero allocator are additionally closed below.

S4 keeps its 64-byte width:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | statement ordinal |
| 4 | 4 | source-row ordinal |
| 8 | 8 | stable row ID |
| 16 | 1 | disposition: `1` survives, `2` applied then canceled, `3` suppressed at statement |
| 17 | 1 | flags, exactly zero |
| 18 | 2 | reserved, zero |
| 20 | 4 | S7 table-block reference |
| 24 | 4 | S7 transition reference |
| 28 | 4 | reserved, zero |
| 32 | 32 | statement digest |

Every disposition has a live table-block reference. A surviving row has a live transition
reference; canceled and suppressed rows use the absent-`u32` sentinel. Each S2 statement targets
the same stable table eventually named by its S7 table block. For each table, S4 row IDs in
statement/source order exactly cover `[row_allocator_before, row_allocator_high_water)`, whose
length equals that table's inserted-disposition count. That range must also pass the independent
durable-lease witness check below; neither S4 nor S7 authorizes or advances an allocator. The sum
of the table counts equals the aggregate original inserted-row count. Every S4 statement digest
equals the corresponding S1 statement digest. The outcome matrix below is the sole disposition
authority: a successful aggregate has only `Survives`; an aborted aggregate has
`AppliedThenCanceled` for every row of each prior successful statement and
`SuppressedAtStatement` for every row of its final failing statement. No other mixture is legal.

The numeric S4 form above is frozen. Its table/transition referential closure cannot be accepted
until the S7 wire is frozen and implemented in the same later checkpoint.

### Existing provisional S1--S6 scaffold

The accepted source currently named `executable_semantics_v2` is **not** a reader or owner of the
normative S4 form above. Its physical aggregate header remains format 1 / semantics 1, it has no
semantics-2 discriminator or writer, and it is reachable only from inert tests. It provisionally
validates one global `[allocator_before, allocator_high_water)` row-ID range and requires both
references to be absent for canceled or suppressed rows; its move-only S1--S6 draft retains those
provisional dispositions. Those rules are semantics-1 scaffold debt, not an alternate
semantics-v2 acceptance language.

After the exact S7 design gate, the S4 + S7 implementation checkpoint must make dispatch
semantics-version-specific, preserve semantics-1 bytes, and replace or rename that provisional
path so exactly one normative semantics-v2 S4/S7 authority remains. It may retain a historical
semantics-1 translator, but no shared validator may silently apply the global-range or
both-references-absent rules to semantics 2.

## Normative wire: shared typed image and vector

One `GPUDBTYPEDIMAGE2` columnar codec is shared by S7 final table images and future S8 retained
responses. It is not a second INSERT carrier.

The image header is exactly 112 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 16 | ASCII `GPUDBTYPEDIMAGE2` |
| 16 | 2 | version `2` |
| 18 | 2 | header bytes, `112` |
| 20 | 4 | exactly one role value: `1` final table image, `2` retained response |
| 24 | 4 | row count |
| 28 | 4 | column count |
| 32 | 8 | checked cell count, rows × columns |
| 40 | 8 | descriptor bytes, columns × 96 |
| 48 | 8 | packed name bytes |
| 56 | 8 | packed vector bytes |
| 64 | 32 | layout digest |
| 96 | 16 | reserved, zero |

The exact packing is header, descriptors, names in descriptor order, then vectors in descriptor
order, without gaps, overlaps, or trailing bytes. Each 96-byte descriptor is:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | dense descriptor ordinal |
| 4 | 4 | catalog-column ordinal, or `0xffff_ffff` for a derived projection |
| 8 | 4 | stable column ID, or zero for a derived projection |
| 12 | 4 | table reference, or `0xffff_ffff` for a derived projection |
| 16 | 2 | signed `attnum`, or `-32768` for a derived projection |
| 18 | 2 | flags, exactly zero |
| 20 | 4 | SQL storage type: tag, precision, scale, reserved zero |
| 24 | 4 | declared type OID, including domain OIDs |
| 28 | 2 | signed type size |
| 30 | 2 | result format: `0` text, `1` binary |
| 32 | 8 | absolute name offset |
| 40 | 8 | name length |
| 48 | 8 | absolute vector offset |
| 56 | 8 | vector length |
| 64 | 32 | vector digest |

SQL storage tags, canonical signed type sizes, and value domains are:

| Tag | SQL storage type | Type size | Accepted value domain |
|---:|---|---:|---|
| 1 | int2 | 2 | widened i32 in `[-32768, 32767]` |
| 2 | int4 | 4 | every i32 |
| 3 | int8 | 8 | every i64 |
| 4 | numeric(p,s) | -1 | i128 mantissa with `abs(mantissa) < 10^p` |
| 5 | bool | 1 | canonical bitmap below |
| 6 | text | -1 | canonical UTF-8/offset form below |
| 7 | date | 4 | i32 days in `[-2451545, 2145031949)` from 2000-01-01 |
| 8 | timestamp | 8 | i64 microseconds in `[-211813488000000000, 9223371331200000000)` from 2000-01-01 |
| 9 | UUID | 16 | every 16-byte value |

Non-numeric precision and scale are zero. Numeric precision is `1..=38`, scale is no greater than
precision, and the fourth SQL-type byte is always zero. The declared `type_oid` is nonzero; it may
be a domain OID different from the storage type's PostgreSQL OID. The descriptor's signed type
size equals the table above, and result format is exactly `0` or `1`. This inert codec performs no
catalog lookup: every nonzero `type_oid` is accepted by the image grammar.

Final-table descriptors have `name_offset == name_len == 0`, use result format zero, carry
non-sentinel stable table/column identity, and occur in catalog order (`catalog-column ordinal ==
descriptor ordinal`). Response descriptors carry nonempty canonical UTF-8 projection names
without NUL and use projection order, which may differ from catalog order and may repeat a source
column. A derived response projection uses all four identity sentinels together; if any derived
sentinel is present, all four are present.

The layout digest is:

```text
D(
  "gpu-db/write001/image-layout/v2",
  rows:u32,
  columns:u32,
  checked_cells:u64,
  each descriptor's bytes 0..48 in ordinal order,
  zero:u64, zero:u64, zero:[u8;32] for each descriptor's vector offset/length/digest,
  packed response names in descriptor order
)
```

Final-table names contribute no bytes. The role is intentionally absent, so byte-identical
geometry and identities have the same layout digest in either role.

Each vector is validity followed by values. Validity form `0` is `AllValid` and ends after its
one-byte tag. Form `1` contains an exact `u32` word count and LSB-first `u32` words; its count is
`ceil(rows/32)`, tail bits are zero, and an all-one bitmap (including the zero-row empty bitmap)
must use `AllValid`.

The values header is shape `u8`, logical count `u32` exactly equal to the image row count, and
payload length `u32` exactly equal to the following payload. Shapes are:

| Shape | Logical type/storage | Exact payload |
|---:|---|---|
| 1 | int2, int4, date / i32 | `4 × rows` bytes |
| 2 | int8, timestamp / i64 | `8 × rows` bytes |
| 3 | numeric / i128 mantissa | `16 × rows` bytes |
| 4 | UUID | `16 × rows` raw bytes |
| 5 | bool bitmap | `u32 word_count` then `4 × ceil(rows/32)` bytes |
| 6 | text | `u32(rows+1)`, that many `u64` offsets, `u32` byte length, UTF-8 bytes |

All scalar integers and offsets are little-endian. Fixed-width values satisfy the exact domains in
the table above. A bool payload repeats the exact `ceil(rows/32)` word count and has zero tail
bits. A text payload has exactly `rows + 1` offsets; the first is zero, they are monotonic, every
offset is within the UTF-8 blob and on a code-point boundary, and the final offset equals the blob
length. The blob itself is valid UTF-8. An invalid fixed-width/bool cell has a zero placeholder;
an invalid text cell repeats its preceding offset.

Zero-row vectors remain nonempty: validity is one byte; fixed-width values have a nine-byte header
and zero payload; bool values have a 13-byte body including zero word count; text values have a
25-byte body containing the sole zero offset and zero byte length. Each descriptor vector is
consumed exactly, without trailing bytes.

The vector digest is:

```text
D(
  "gpu-db/write001/typed-vector/v2",
  descriptor SQL storage-type bytes 20..24,
  rows:u32,
  exact validity bytes,
  exact values bytes
)
```

These tables and rules define the complete accepted image language. Header geometry, descriptor
bytes, name/vector regions, and total encoded length are recomputed with checked `u64` arithmetic;
the exact total equals the input length, every absolute range converts to the implementation
address space, descriptor/name/vector regions fill their declared areas without gaps or overlap,
and no trailing byte is accepted. Digest equality is checked only after the corresponding raw
header, directory, name, vector, placeholder, and scalar-domain rules pass.

## Normative wire: semantics-v2 S7 final overlay

S7 is one self-contained final-overlay container. It records only transaction-created `New`
transitions. `Replaced` and `Deleted` have no tag in this profile and any attempted encoding
rejects. Applied-then-canceled and statement-suppressed inputs remain S4 dispositions without a
transition. This is complete for the permanently typed-INSERT-only profile; admitting UPDATE,
DELETE, catalog, reset, rewrite, or opaque S3 bodies requires a later aggregate semantics version
with its own typed source and resolution grammar.

All S7 offsets are unsigned `u64` byte offsets from the first byte of the S7 payload. Every
directory reference is a dense zero-based `u32`; `0xffff_ffff` is the only absent reference.
Stable object and row identities are `u64`, nonzero, and not `0xffff_ffff_ffff_ffff`. Display OIDs
are independently carried `u32` values in `1..=0x7fff_ffff`; a stable identity and display OID are
different identity domains even when their numeric values happen to match. The only exceptions
are a synthesized table-column `NotNullGuard` and a `DomainConstraintGuard` classified as
synthesized domain NOT NULL; either may use display OID zero while retaining a nonzero stable
guard ID. Constraint display OID zero and stable ID `0xffff_ffff_ffff_ffff` are otherwise the
paired absence form in an index descriptor.

### S7 header

The header is exactly 640 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 16 | ASCII `GPUDBS7OVERLAY2\0` |
| 16 | 2 | S7 format version `1` |
| 18 | 2 | aggregate semantics version `2` |
| 20 | 4 | header bytes, `640` |
| 24 | 4 | flags, zero |
| 28 | 2 | directory count, `14` |
| 30 | 2 | root-descriptor version, `1` |
| 32 | 8 | exact total S7 payload bytes |
| 40 | 4 | table-block count |
| 44 | 4 | statement-resolution count |
| 48 | 4 | table-disposition-reference count |
| 52 | 4 | dependency-token count |
| 56 | 4 | statement-dependency-use count |
| 60 | 4 | index-descriptor count |
| 64 | 4 | index-key-column count |
| 68 | 4 | transition count |
| 72 | 4 | key-effect count |
| 76 | 4 | typed-key-component count |
| 80 | 4 | projection-binding count |
| 84 | 4 | image-descriptor count |
| 88 | 8 | typed-key value-arena bytes |
| 96 | 8 | image-arena bytes |
| 104 | 224 | fourteen `(offset:u64, byte_length:u64)` directory descriptors |
| 328 | 8 | outer catalog-before epoch echo |
| 336 | 8 | outer catalog-after epoch echo |
| 344 | 32 | outer catalog-before digest echo |
| 376 | 32 | outer catalog-after digest echo |
| 408 | 32 | initial database-data root |
| 440 | 32 | final database-data root |
| 472 | 32 | initial transaction-overlay root |
| 504 | 32 | final transaction-overlay root |
| 536 | 32 | root-descriptor digest |
| 568 | 32 | S7 payload digest |
| 600 | 40 | reserved, zero |

The fourteen descriptors at byte 104 are, in this exact order:

| Index | Header offset | Region | Entry width |
|---:|---:|---|---:|
| 0 | 104 | table blocks | 384 |
| 1 | 120 | table-disposition references | 32 |
| 2 | 136 | statement resolutions | 320 |
| 3 | 152 | dependency tokens | 224 |
| 4 | 168 | statement-dependency uses | 32 |
| 5 | 184 | index descriptors | 384 |
| 6 | 200 | index-key columns | 112 |
| 7 | 216 | transitions | 192 |
| 8 | 232 | key effects | 192 |
| 9 | 248 | typed-key components | 128 |
| 10 | 264 | projection bindings | 128 |
| 11 | 280 | image descriptors | 160 |
| 12 | 296 | typed-key value arena | byte arena |
| 13 | 312 | image arena | byte arena |

The table, statement, and image counts are nonzero; image count equals table count. The first
region starts at byte 640. Every fixed directory byte length is exactly `count × entry_width`,
the two arena lengths equal the header's arena lengths, every region begins at the preceding
region's checked end, and the final image-arena end equals `total_bytes`. Gaps, overlap, reordered
regions, surplus bytes, and an S7 section length different from `total_bytes` reject.

The two catalog epochs and digests exactly equal the outer `CanonicalPreApplyHeader` fields. Since
this profile cannot mutate the catalog, before and after epochs are equal and before and after
digests are equal. All four roots are nonzero. The initial overlay root equals the first S1
`overlay_before`; the final overlay root equals the last S1 `overlay_after`. The database roots
are governed by the root rules below rather than inferred from catalog or physical residency.

### Canonical directory order

Dense reference fields equal their physical directory ordinal. Directories use these exact
orders:

- table blocks: ascending stable table ID;
- table-disposition references: `(table_ref, stable_row_id)`;
- statement resolutions: ascending statement ordinal;
- dependency tokens: the dependency identity key defined below;
- statement-dependency uses:
  `(statement_ref, role, source_ordinal, transition_ref, key_effect_ref, dependency_ref)`;
- index descriptors: `(owner_table_id, stable_index_id)`;
- index-key columns: `(index_ref, key_ordinal)`;
- transitions: `(table_ref, stable_row_id)`;
- key effects: `(transition_ref, role, index_ref, source_catalog_ordinal)`;
- typed-key components: `(key_effect_ref, side, component_ordinal)`;
- projection bindings: `(statement_ref, projection_ordinal)`;
- images: ascending table reference.

All tuples are compared as unsigned values; the absent `u32` sentinel consequently sorts after
every live reference. Equal order keys, duplicate stable identities, duplicate physical
references, and a missing dense ordinal reject.

### Table blocks

Each table block is exactly 384 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | table reference, equal to directory ordinal |
| 4 | 4 | flags, zero |
| 8 | 8 | stable table ID |
| 16 | 4 | display table OID |
| 20 | 4 | target-table dependency-token reference |
| 24 | 8 | catalog epoch |
| 32 | 8 | data generation before |
| 40 | 8 | data generation after |
| 48 | 8 | table-local row allocator before |
| 56 | 8 | table-local row allocator high-water, exclusive |
| 64 | 8 | initial logical row count |
| 72 | 8 | final logical row count |
| 80 | 4 | table-disposition-reference start |
| 84 | 4 | table-disposition-reference count |
| 88 | 4 | transition start |
| 92 | 4 | transition count |
| 96 | 4 | owned index-descriptor start |
| 100 | 4 | owned index-descriptor count |
| 104 | 4 | key-effect start |
| 108 | 4 | key-effect count |
| 112 | 4 | image reference |
| 116 | 4 | catalog-column count |
| 120 | 8 | reserved, zero |
| 128 | 32 | table schema digest |
| 160 | 32 | initial table-data root |
| 192 | 32 | final table-data root |
| 224 | 32 | final-image layout digest |
| 256 | 32 | final-image content digest |
| 288 | 32 | table transition root |
| 320 | 32 | table index-effect root |
| 352 | 32 | table-manifest digest |

The catalog epoch equals both header catalog epochs. Generations are nonzero and not the maximum
`u64`. A block with at least one transition has
`data_generation_after > data_generation_before`; a block without transitions has equal
generations. Failed private candidates may consume generation identities, so adjacency is not a
wire invariant. Allocator bounds are nonzero, ordered, and their difference equals the
disposition count. `final_row_count == initial_row_count + transition_count`. Initial and final
table roots, schema digest, image digests, transition root, index-effect root, and manifest digest
are nonzero. Disposition count and catalog-column count are nonzero.

The disposition, transition, and key-effect `(start,count)` pairs name exact contiguous ranges
owned by this table; together, table blocks concatenate and exhaust those three directories.
Their zero-count ranges use the checked end of the preceding table's range or zero for the first
table. The index range is the exact contiguous run whose descriptor owner ID equals this table's
stable ID; parent-only FK index descriptors may occur between target-table runs. A zero index
range uses the lower-bound insertion point for `(stable_table_id, stable_index_id=0)`.

The image reference equals the table reference. The target dependency is kind `TargetTable`, has
`Write` access, and binds the same stable ID, display OID, schema digest, catalog epoch, initial
generation, and initial table root.

Table blocks biject the distinct target-table dependencies in S2: every distinct target occurs
once, every statement resolves to that one block, and no block exists without at least one S2
statement. The target block's qualified name, display OID, schema digest, stable ID, generation,
and initial root must also equal the pinned catalog witness below. Statements which share a block
must carry identical target dependency facts.

### Table-disposition references

The S4-to-table reorder directory has 32-byte entries:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | table reference |
| 4 | 4 | absolute S4 disposition reference |
| 8 | 8 | stable row ID |
| 16 | 4 | statement ordinal |
| 20 | 4 | source-row ordinal |
| 24 | 1 | S4 disposition tag |
| 25 | 1 | flags, zero |
| 26 | 6 | reserved, zero |

Every field duplicates and must equal the referenced S4 entry. The directory bijects all S4
entries exactly once. Within each table range, stable row IDs exactly cover
`[allocator_before, allocator_high_water)` in ascending order. Across table blocks, the sum of
disposition counts equals the aggregate original inserted-row count and the S4 entry count.

### Statement resolutions

Each 320-byte resolution closes one S1/S2/S4/S5/S6 statement:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | statement reference, equal to statement ordinal |
| 4 | 4 | S1 reference |
| 8 | 4 | S2 reference |
| 12 | 4 | S6 reference |
| 16 | 4 | target table reference |
| 20 | 4 | flags: bit 0 has `RETURNING`, bit 1 response retained |
| 24 | 4 | absolute S4 disposition start |
| 28 | 4 | S4 disposition count |
| 32 | 4 | absolute S5 sequence-effect start |
| 36 | 4 | S5 sequence-effect count |
| 40 | 4 | statement-dependency-use start |
| 44 | 4 | statement-dependency-use count |
| 48 | 4 | projection-binding start |
| 52 | 4 | projection-binding count |
| 56 | 4 | input-row count |
| 60 | 4 | surviving-row count |
| 64 | 8 | SQL affected-row count |
| 72 | 8 | dependency validation floor |
| 80 | 4 | exact S2 record bytes, excluding S2's length prefix |
| 84 | 4 | terminal-error dependency-token reference, or absent |
| 88 | 4 | terminal-error source-row ordinal, or absent |
| 92 | 4 | terminal-error source ordinal, or absent |
| 96 | 32 | request digest, equal to the referenced S1 request digest |
| 128 | 32 | typed statement digest |
| 160 | 32 | exact S2 record digest |
| 192 | 32 | S2 `RETURNING` layout digest |
| 224 | 32 | S1 overlay-before root |
| 256 | 32 | S1 overlay-after root |
| 288 | 32 | exact S6 entry digest |

Only flag bits 0 and 1 are known. S1, S2, and S6 references all equal the statement ordinal.
The resolution request digest equals the referenced S1 request digest. Target-table identity,
statement ordinal, row and column counts, typed statement digest, and `RETURNING`
geometry/layout are read from the strictly decoded S2 model and must equal the S1, S6, table,
image, and projection facts. For typed INSERT, the S1 request digest, S1 statement digest, and
resolution typed statement digest are all equal. Input-row count is nonzero. The S4 range is the
statement's exact
statement/source-order range; its count equals both
the S1 input-row count and S2 row count. Surviving count is the number of `Survives` tags. Affected
count is the number of `Survives` plus `AppliedThenCanceled` tags and must equal S6 outcome
semantics. S6 class is `1`, its statement/family ordinals and statement digest match, and its
outcome target digest equals `overlay_after`. On `CommitSuccess`, all three terminal-error fields
are absent. On `AbortError`, all three are live, the source row is less than input-row count, the
token has its terminal-error flag, and exactly one statement-dependency use for that statement
names the same token and source ordinal. The error binding rules and exact outcome relationship
are closed by the matrix below. `CommitNoOp` is invalid.

The S5, dependency-use, and projection ranges are exact canonical insertion-point ranges grouped
by statement. S5 effects biject S2 effects in effect order. Projection count and flag bit 0 are
nonzero exactly when S2 has `RETURNING`; bit 1 equals S6's retained-response flag and implies bit
0. S8 owns any retained bytes and remains independently gated. For a successful statement, S6
`returning_digest` equals the canonical logical result digest defined below; it is zero exactly
when projection count is zero. An abort has no result rows and its S6 `returning_digest` is zero.

The validation floor is nonzero. When the complete aggregate is bound to its outer envelope it is
strictly less than the outer commit sequence. Every dependency used by the statement has a token
floor no greater than this value, and each token's recorded floor is the minimum resolution floor
among all statements which use it.

The exact source and outcome digests are:

```text
s2_record_digest =
  D("gpu-db/write001/s7-s2-record/v2", s2_record_bytes:u32, exact S2 record bytes)

s6_entry_digest =
  D("gpu-db/write001/s7-s6-entry/v2", exact 136-byte S6 entry)
```

S2's four-byte section-entry length is not part of `s2_record_digest`; S6 has fixed width and no
outer length.

### Dependency tokens

Dependency tokens are global and deduplicated. Each is exactly 224 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | dependency reference, equal to directory ordinal |
| 4 | 1 | dependency kind |
| 5 | 1 | access strength |
| 6 | 2 | flags: bit 0 live key effect, bit 1 terminal-error identity |
| 8 | 8 | stable object ID |
| 16 | 4 | display object OID |
| 20 | 4 | INSERT target-table context reference |
| 24 | 8 | base generation/version |
| 32 | 8 | minimum validation floor |
| 40 | 4 | key-effect reference, or absent |
| 44 | 4 | object-descriptor reference, or absent |
| 48 | 8 | catalog epoch |
| 56 | 8 | reserved, zero |
| 64 | 32 | schema/shape digest |
| 96 | 32 | base object root |
| 128 | 32 | canonical object-name digest |
| 160 | 32 | kind-specific identity digest |
| 192 | 32 | dependency-token digest |

Kinds and their exact field rules are:

| Kind | Value | Access | Table context | Object descriptor | Key effect | Meaning |
|---|---:|---|---|---|---|---|
| `TargetTable` | 1 | `Write` | live target | absent | absent | target table/schema/base root |
| `ForeignParentTable` | 2 | `Validate` | live target | absent | absent | referenced FK table |
| `MaintainedIndex` | 3 | `Write` | live target | live index | absent | physical target-index maintenance |
| `UniqueKeyGuard` | 4 | `Validate` | live target | live index | success live / terminal absent | one unique-key guard |
| `ForeignParentIndex` | 5 | `Validate` | live target | live index | absent | FK supporting-index shape/root |
| `ForeignKeyGuard` | 6 | `Validate` | live target | live parent index | success live / terminal absent | one parent-key lookup |
| `Domain` | 7 | `Read` | live target | absent | absent | S2 domain binding |
| `PublishedSequence` | 8 | `Consumed` | live target | absent | absent | one already-durable S5 value transition |
| `NotNullGuard` | 9 | `Validate` | live target | absent | absent | one catalog-column NOT NULL guard |
| `CheckGuard` | 10 | `Validate` | live target | absent | absent | one table CHECK guard |
| `DomainConstraintGuard` | 11 | `Validate` | live target | absent | absent | one domain NOT NULL/CHECK guard |

Access values are `Read=1`, `Validate=2`, `Write=3`, and `Consumed=4`. `Read < Validate < Write`
is the only strength lattice; `Consumed` is legal only for kind 8 and never merges with another
access class. `Validate` is both the replay-validation dependency mode and the conflict
coordinator's guard-access mode; it does not itself grant visibility or substitute for the
statement snapshot. Bit 0 is set only for a participating successful kind-4 or kind-6 guard and
then its key-effect reference is live. Bit 1 occurs exactly once in an aborted aggregate, on the
token selected by the final statement resolution; that token has no key effect. The two bits are
mutually exclusive. All other tokens have zero flags and an absent key reference. Descriptor
references are live exactly for kinds 3 through 6. Every token's table context is the live S7
target table whose INSERT requires the dependency; an FK parent object's actual owner identity
remains in its table token or index descriptor. Base generation/version is nonzero and not the
maximum `u64`; minimum validation floor is nonzero except for `PublishedSequence`, which uses zero
because the referenced transition is already durable. Catalog epoch equals the S7/outer catalog
epoch.

For table kinds 1 and 2, schema digest and base root are nonzero and the kind-specific identity
digest is:

```text
D(
  "gpu-db/write001/s7-table-object/v2",
  kind:u8,
  stable_object_id:u64,
  display_oid:u32,
  catalog_epoch:u64,
  base_generation:u64,
  schema_digest:[u8;32],
  base_root:[u8;32],
  name_digest:[u8;32]
)
```

For index and index-guard kinds 3 through 6, schema digest is the owner-table schema digest, base
root equals the referenced index descriptor's base-index root, and the identity digest is:

```text
D(
  "gpu-db/write001/s7-index-object/v2",
  kind:u8,
  stable_object_id:u64,
  display_oid:u32,
  catalog_epoch:u64,
  base_generation:u64,
  schema_digest:[u8;32],
  base_root:[u8;32],
  name_digest:[u8;32],
  referenced_index_descriptor_digest:[u8;32],
  key_effect_digest_or_zero:[u8;32]
)
```

For a domain, `base_root` is zero and `schema_digest` is:

```text
D(
  "gpu-db/write001/s7-domain-shape/v2",
  SQL-storage-type:[u8;4],
  declared_type_oid:u32,
  signed_type_size:i16
)
```

The domain identity digest is:

```text
D(
  "gpu-db/write001/s7-domain-object/v2",
  stable_object_id:u64,
  display_oid:u32,
  catalog_epoch:u64,
  base_generation:u64,
  schema_digest:[u8;32],
  name_digest:[u8;32]
)
```

For constraint kinds 9 through 11, stable object ID is the stable guard/constraint ID from the
pinned catalog witness. Display OID is the catalog constraint OID; only a synthesized table-column
NOT NULL guard or synthesized domain-NOT-NULL guard may use zero because PostgreSQL has no
independently addressable `pg_constraint` row for those properties. Both still have nonzero stable
guard IDs. Schema digest is the typed evaluator/column shape digest and base root is its nonzero
catalog program/descriptor root. Their identity digest is:

```text
D(
  "gpu-db/write001/s7-constraint-object/v2",
  kind:u8,
  stable_object_id:u64,
  display_oid:u32,
  target_stable_table_id:u64,
  catalog_epoch:u64,
  base_generation:u64,
  schema_digest:[u8;32],
  base_root:[u8;32],
  name_digest:[u8;32]
)
```

For a published sequence, stable object ID and display OID identify the pinned catalog sequence
which S2/S5 name. Base generation/version is the published transition transaction ID, schema
digest is zero, base root is the exact S5 body digest, and name digest is its qualified catalog
name:

```text
D(
  "gpu-db/write001/s7-published-sequence/v2",
  stable_object_id:u64,
  display_oid:u32,
  catalog_epoch:u64,
  transition_txn_id:u64,
  name_digest:[u8;32],
  s5_body_digest:[u8;32],
  exact encoded BinarySequenceValueReference bytes
)
```

Identifier and qualified-name digests use distinct, length-delimited preimages:

```text
identifier_digest =
  D(
    "gpu-db/write001/s7-identifier/v2",
    utf8_byte_length:u32,
    exact UTF-8 bytes
  )

qualified_name_digest =
  D(
    "gpu-db/write001/s7-qualified-name/v2",
    schema_utf8_byte_length:u32,
    exact schema UTF-8 bytes,
    object_utf8_byte_length:u32,
    exact object UTF-8 bytes
  )

synthesized_not_null_name_digest =
  D(
    "gpu-db/write001/s7-synthesized-not-null-name/v2",
    owner_kind:u8,                 // 1 table column, 2 domain
    stable_owner_id:u64,
    source_ordinal:u32             // catalog column ordinal, or zero for domain
  )
```

Schema and object components are the already resolved catalog identifiers, not SQL text, a dotted
string, search-path input, or a case-folded reconstruction. Empty components, NUL, invalid UTF-8,
and surplus qualification reject. Dependency table/domain/index/named-constraint/sequence name
fields use `qualified_name_digest`; column, projection, and unqualified display-name fields use
`identifier_digest`. The two synthesized NOT NULL guard classes have no catalog constraint name
and use `synthesized_not_null_name_digest` instead; no fabricated SQL identifier is admitted.

The exact field sources and cross-equalities are:

| Token kind | Exact source and required equality |
|---|---|
| `TargetTable` | S2 dependency ordinal zero plus the matching pinned table row. Stable/display identity, qualified name, schema digest, catalog epoch, data generation, and data root equal both the table witness and its S7 table block. |
| `ForeignParentTable` | The S2 foreign-key `parent_dependency_ordinal` plus that exact pinned parent-table row. Its stable/display identity, qualified name, schema digest, data generation/root, and catalog epoch equal the owner-table fields of the same FK's supporting index descriptor. |
| `MaintainedIndex` | One S2 target-index raw ordinal and the matching pinned index row. Every stable/display/name/owner/schema/epoch/base-generation/base-root field equals its referenced descriptor. |
| `UniqueKeyGuard` | The same S2 target-index and descriptor as its maintenance effect. All index identity/generation fields equal the descriptor; a successful participating row also binds its exact key effect, while a terminal 23505 token has no effect and is selected by the failing statement. |
| `ForeignParentIndex` | The exact supporting index embedded in one S2 FK raw ordinal and its pinned index row. All index identity/generation fields equal the descriptor, whose owner fields equal that FK's parent-table dependency. |
| `ForeignKeyGuard` | The same S2 FK raw ordinal and supporting descriptor as `ForeignParentIndex`. All index identity/generation fields equal the descriptor; a successful participating row binds its exact FK key effect, while a terminal 23503 token has no effect. |
| `Domain` | One S2 domain ordinal plus its pinned domain row. Stable/display identity, qualified name, catalog generation, storage-shape digest, and epoch agree; base root is zero. |
| `PublishedSequence` | One exact S2/S5 published effect plus its pinned sequence row. Stable/display identity and qualified name agree with the catalog row; base generation is the S5 transition transaction ID and base root is the S5 body digest. |
| `NotNullGuard` | One catalog column in the target's pinned table descriptor. Stable guard ID, zero-or-catalog display OID, synthesized NOT NULL name digest, column-shape digest/root, constraint generation, and source catalog-column ordinal agree. |
| `CheckGuard` | One ordered CHECK descriptor in the target's pinned table descriptor. Stable/display identity, qualified name, typed evaluator shape/root, constraint generation, and raw CHECK ordinal agree. |
| `DomainConstraintGuard` | One ordered constraint descriptor of an S2-bound domain. Stable/display identity, typed evaluator shape/root, constraint generation, domain ordinal, and raw domain-constraint ordinal agree. A named domain CHECK uses its qualified name; synthesized domain NOT NULL uses the zero-display/synthesized-name form. |

The pinned catalog witness is defined below. It is the source of stable identities and
generations which S2 intentionally does not duplicate. Index token `stable_object_id`,
`display_oid`, `name_digest`, `base_generation`, `schema_digest`, and `base_root` therefore equal
the referenced descriptor's stable index ID, display index OID, index-name digest, base index
generation, owner schema digest, and base index root byte-for-byte. An FK parent token joins to
its supporting descriptor through the same S2 FK raw ordinal and parent dependency ordinal; name
or OID coincidence alone is never a join.

The per-kind `base_generation` source is table data generation for kinds 1--2, base index
generation for kinds 3--6, catalog object generation for kind 7 and kinds 9--11, and the durable
published transition transaction ID for kind 8. The token digest is:

```text
D(
  "gpu-db/write001/s7-dependency-token/v2",
  exact token bytes 0..192,
  zero:[u8;32],
  referenced_index_descriptor_digest_or_zero:[u8;32],
  referenced_key_effect_digest_or_zero:[u8;32]
)
```

This token digest is envelope-local because its preimage deliberately contains dense directory
references. It is never a cross-transaction lock or guard identity. The reference-neutral runtime
guard key is recomputed as:

```text
D(
  "gpu-db/write001/runtime-guard-key/v2",
  kind:u8,
  target_stable_table_id:u64,
  stable_object_id:u64,
  catalog_epoch:u64,
  base_generation:u64,
  schema_digest:[u8;32],
  base_root:[u8;32],
  name_digest:[u8;32],
  stable_index_id_or_zero:u64,
  key_arity_or_zero:u32,
  each canonical typed-value digest in key order
)
```

Static guards have zero arity. A successful participating unique/FK guard uses its component
typed-value digests, which contain no local references. A terminal 23505/23503 guard has no
transition or key-effect record, so its nonzero arity and typed-value digests are instead
recomputed directly from the S2 resolved source row selected by the statement resolution:
23505 traverses the target index key columns; 23503 traverses the FK child columns in the
supporting parent-index order. Type/size/declared-OID compatibility and logical scalar bytes obey
the same component rules, and every component must be non-NULL or that SQLSTATE/token pair
rejects. Thus the resolution's row/ordinal plus S2 and the base index shape bind one exact
reference-neutral terminal guard key even without a successful effect. The conflict coordinator
and replay validator order and merge only the derived runtime key plus access mode.

The dependency identity key used for sorting and deduplication is the exact tuple
`(kind, terminal_error_flag, stable_object_id, display_oid, target_table_ref, catalog_epoch, base_generation,
key_effect_ref, object_descriptor_ref, schema_digest, base_root, name_digest, identity_digest)`.
Access, bit-0 participation, floor, dense reference, and token digest are excluded. Equal keys collapse to one token
with the strongest legal access and minimum nonzero validation floor. Two physical entries with
the same identity key reject.

### Statement-dependency uses

Each 32-byte use binds one S2/S5 source fact to a deduplicated token:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | statement reference |
| 4 | 4 | dependency-token reference |
| 8 | 2 | role |
| 10 | 2 | flags, zero |
| 12 | 4 | kind-specific S2 source ordinal |
| 16 | 4 | transition reference, or absent |
| 20 | 4 | key-effect reference, or absent |
| 24 | 8 | reserved, zero |

Roles are `TargetTable=1`, `TargetIndex=2`, `UniqueGuard=3`, `ForeignParentTable=4`,
`ForeignParentIndex=5`, `ForeignGuard=6`, `Domain=7`, and `PublishedSequence=8`. The referenced
token kind must be the corresponding kind above. Additional roles are `NotNullGuard=9`,
`CheckGuard=10`, and `DomainConstraintGuard=11`. Target/index/domain/constraint static uses have
both effect references absent. Participating successful unique/FK guards have both transition and
key-effect references live. The one terminal-error unique/FK use has both absent and is instead
bound to the resolution's failing source row. Published sequences reference the survivor
transition when their disposition survives and otherwise use an absent transition; their key
effect is absent. Source ordinals are, respectively, zero, target-index raw ordinal, target-index
raw ordinal, S2 dependency ordinal, foreign-key raw ordinal, foreign-key raw ordinal, domain
ordinal, sequence-effect ordinal, catalog-column ordinal, raw CHECK ordinal, and the pinned
domain-constraint ordinal.

For a successful statement, uses biject every S2 target/index/domain/FK/sequence dependency,
every catalog-witness NOT NULL/CHECK/domain-constraint guard applicable to its resolved columns,
and every participating per-transition unique and FK guard; NULL-suppressed equality effects have
neither a token nor a use. For the final abort statement, static source dependencies and exactly
one terminal-error guard use are retained, while its statement-suppressed rows have no transition
or successful key-effect uses. Within that statement the bit-1 terminal use replaces, rather than
duplicates, the ordinary use of the same guard identity; other applicable static guards remain,
and an earlier successful statement may still use the separate bit-1-clear token for that
identity. No unreferenced token is permitted. Except for already-durable
`PublishedSequence` tokens whose floor is zero, the token floor is the minimum statement floor
among its uses; access is the maximum legal access demanded by its uses.

### Index descriptors and key columns

Every target-table index and every FK supporting index named by S2 has one deduplicated
384-byte descriptor:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | index reference, equal to directory ordinal |
| 4 | 4 | flags |
| 8 | 4 | owner table reference, or absent when parent-only |
| 12 | 4 | raw catalog index ordinal |
| 16 | 8 | stable index ID |
| 24 | 4 | display index OID |
| 28 | 4 | constraint display OID, or zero |
| 32 | 8 | stable constraint ID, or absent-`u64` |
| 40 | 4 | key-column start |
| 44 | 4 | key-column count |
| 48 | 1 | NULL equality policy, `1` (`NULLS DISTINCT`) |
| 49 | 15 | reserved, zero |
| 64 | 8 | stable owner-table ID |
| 72 | 8 | catalog epoch |
| 80 | 32 | owner-table schema digest |
| 112 | 32 | owner-table qualified-name digest |
| 144 | 32 | index qualified-name digest |
| 176 | 32 | constraint qualified-name digest, or zero |
| 208 | 32 | owner-table base root |
| 240 | 32 | base index root |
| 272 | 32 | final index root |
| 304 | 32 | index-descriptor digest |
| 336 | 4 | owner display table OID |
| 340 | 4 | reserved, zero |
| 344 | 8 | owner-table base generation |
| 352 | 8 | base index generation |
| 360 | 8 | final index generation |
| 368 | 16 | reserved, zero |

Flag bits are `UNIQUE=1<<0`, `PRIMARY_KEY=1<<1`, `UNIQUE_CONSTRAINT=1<<2`, and
`MAINTAINED_BY_OVERLAY=1<<3`; no others are known. Primary key implies the first three semantic
uniqueness/NOT-NULL properties but does not imply the `UNIQUE_CONSTRAINT` bit; unique constraint
implies unique. The first three bits otherwise equal S2's three booleans exactly.
Constraint OID/ID/name digest are all present exactly when `PRIMARY_KEY` or
`UNIQUE_CONSTRAINT` is set and are otherwise all absent. In the current catalog model a
constraint-backed index uses its index OID/ID/name for the constraint fields; separating the
fields prevents that implementation fact from becoming identity law. `MAINTAINED_BY_OVERLAY` is
set exactly when the owner table has an S7 block. Every index owned by a target table is
maintained; a parent-only FK index is not.

Stable/display index identity, owner stable/display identity, raw ordinal, qualified owner/index/
constraint names, schema, flags, catalog epoch, base generations/roots, and every key descriptor
must equal both the exact S2 index copy and the pinned catalog witness as applicable. S2 supplies
display/catalog/shape facts; the witness supplies stable IDs, generations, and roots. For an
owner represented by an S7 table block, the descriptor's owner stable/display identity,
qualified-name digest, schema digest, base generation, and base root equal that block's target
token and initial fields. For a parent-only supporting index, they equal the
`ForeignParentTable` token selected by the same S2 FK raw ordinal. This is the exact FK
parent-table-to-supporting-index join.

Base and final index generations are nonzero and not maximum `u64`; all three index roots are
nonzero. If an owned index has one or more maintenance effects, final generation is greater than
base generation and final root differs from base root. If it has no maintenance effects,
including every parent-only index and every aborted aggregate, final generation equals base
generation and final root equals base root. The sole generation builder prebuilds each changed
index and supplies its final generation/root through the required generation witness below.

Only NULL policy `1` is admitted. It means PostgreSQL default `NULLS DISTINCT`: a NULL component
suppresses uniqueness equality participation but never suppresses physical index maintenance.
`NULLS NOT DISTINCT` and unknown policies reject rather than being inferred.

The key range is nonempty and exact. Each 112-byte key-column descriptor is:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | key-column reference, equal to directory ordinal |
| 4 | 4 | index reference |
| 8 | 4 | key ordinal |
| 12 | 4 | owner catalog-column ordinal |
| 16 | 4 | stable column ID |
| 20 | 4 | owner display table OID |
| 24 | 2 | signed `attnum` |
| 26 | 2 | flags, zero |
| 28 | 4 | SQL storage type |
| 32 | 4 | declared type OID |
| 36 | 2 | signed type size |
| 38 | 2 | reserved, zero |
| 40 | 32 | column-name digest |
| 72 | 32 | key-column descriptor digest |
| 104 | 8 | reserved, zero |

SQL storage type, type OID, type size, column identity, key order, names, raw index ordinal, and
flags exactly equal the strict S2 catalog binding. Composite order is significant. Key ordinal is
dense from zero within each index. The key-column descriptor digest is:

```text
D(
  "gpu-db/write001/s7-index-key-column/v2",
  exact key-column bytes 0..72,
  zero:[u8;32],
  exact key-column bytes 104..112
)
```

The index descriptor digest is:

```text
D(
  "gpu-db/write001/s7-index-descriptor/v2",
  exact index bytes 0..304,
  zero:[u8;32],
  exact index bytes 336..384,
  each referenced key-column descriptor digest in key order
)
```

All index and key-column digests are nonzero. Index descriptors with the same stable identity but
different display/catalog/shape facts reject instead of being merged. The descriptor directory
bijects the union of every S2 target index and every S2 FK supporting-index copy after exact
stable-identity deduplication; an extra descriptor, omitted descriptor, or two conflicting copies
reject.

### Final transitions

Each 192-byte transition represents one final transaction-created row:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | transition reference, equal to directory ordinal |
| 4 | 4 | table reference |
| 8 | 8 | stable row ID |
| 16 | 1 | transition kind, `1` (`New`) |
| 17 | 1 | flags, zero |
| 18 | 2 | reserved, zero |
| 20 | 4 | source S4 disposition reference |
| 24 | 4 | source statement reference |
| 28 | 4 | source-row ordinal |
| 32 | 4 | image reference |
| 36 | 4 | image-row ordinal |
| 40 | 4 | key-effect start |
| 44 | 4 | key-effect count |
| 48 | 4 | final-writer statement reference |
| 52 | 12 | reserved, zero |
| 64 | 32 | source typed-statement digest |
| 96 | 32 | final logical-row digest |
| 128 | 32 | transition digest |
| 160 | 32 | reserved, zero |

Only kind 1 exists. Numeric values which might conventionally mean replace or delete are
reserved-invalid. The source S4 entry is `Survives`, has the same table, row, statement,
source-row, statement digest, and transition reference, and is referenced by no other transition.
Conversely every surviving S4 entry names exactly one transition; canceled/suppressed entries
name none. Therefore transition count equals the S4 survivor count and the aggregate final
row-transition count. Final writer equals the source statement in this typed-INSERT-only profile.

The image reference equals the table reference. Image rows are dense transition order within the
table, so `image_row == transition_ref - table.transition_start`. The key-effect range is exact
and grouped by transition. It contains one physical-maintenance effect for every maintained
target index, one unique-guard effect for every unique target index, and one FK-guard effect for
every S2 foreign key belonging to the source statement.

The final logical-row digest is computed by strict logical traversal of the decoded final image:

```text
D(
  "gpu-db/write001/s7-final-row/v2",
  stable_table_id:u64,
  stable_row_id:u64,
  image_ref:u32,
  image_row:u32,
  catalog_column_count:u32,
  for each catalog-order image column:
    catalog_column_ordinal:u32,
    column_id:u32,
    attnum:i16,
    SQL-storage-type:[u8;4],
    declared_type_oid:u32,
    signed_type_size:i16,
    validity:u8,                 // 0 non-NULL, 1 NULL
    logical_value_length:u32,
    exact logical value bytes
)
```

NULL has zero logical value length and no value bytes. Non-NULL logical value bytes are
little-endian i32 for int2/int4/date, little-endian i64 for int8/timestamp, little-endian
two's-complement i128 for numeric, raw 16 bytes for UUID, one byte `0` or `1` for bool, and raw
UTF-8 bytes for text. Image vector domain checks still apply before this digest is evaluated.

The transition digest is:

```text
D(
  "gpu-db/write001/s7-transition/v2",
  exact transition bytes 0..128,
  zero:[u8;32],
  exact transition bytes 160..192,
  each referenced key-effect digest in canonical effect order
)
```

### Typed new-key effects

Every key effect is 192 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | key-effect reference, equal to directory ordinal |
| 4 | 1 | role: `1` maintenance, `2` unique guard, `3` FK guard |
| 5 | 1 | action: `1` index insert, `2` unique validate, `3` FK validate |
| 6 | 2 | flags, zero |
| 8 | 4 | transition reference |
| 12 | 4 | index reference |
| 16 | 4 | dependency-token reference, or absent |
| 20 | 4 | old-component start, absent |
| 24 | 4 | old-component count, zero |
| 28 | 4 | new-component start |
| 32 | 4 | new-component count |
| 36 | 4 | key arity |
| 40 | 1 | old presence, zero |
| 41 | 1 | new presence, one |
| 42 | 1 | NULL policy, one |
| 43 | 1 | equality-guard participates, zero or one |
| 44 | 1 | key contains NULL, zero or one |
| 45 | 3 | reserved, zero |
| 48 | 4 | source catalog ordinal |
| 52 | 12 | reserved, zero |
| 64 | 32 | old-key digest, zero |
| 96 | 32 | new-key digest |
| 128 | 32 | key-effect digest |
| 160 | 32 | reserved, zero |

Role/action pairs are exactly `(1,1)`, `(2,2)`, and `(3,3)`. Old state is always the exact absent
form because this profile has only `New` transitions. New range is live, exact, and its count
equals both arity and the referenced index's key-column count.

Maintenance and unique effects use the target index's raw catalog ordinal as source ordinal. FK
effects use the S2 foreign-key raw ordinal. This distinguishes two constraints which legitimately
share one supporting parent index.

Maintenance always participates, including NULL keys, and references the `MaintainedIndex`
write token. A unique guard participates exactly when no component is NULL; a participating guard
references its `UniqueKeyGuard` token, while a NULL-suppressed guard uses an absent dependency.
An FK guard participates exactly when no child component is NULL; a participating guard
references its `ForeignKeyGuard` token, while a NULL-suppressed guard uses an absent dependency.
Suppression removes only equality validation, not the effect record or typed key.

The new-key digest is:

```text
D(
  "gpu-db/write001/s7-typed-key/v2",
  key_effect_ref:u32,
  side:u8,                       // 2 = new
  arity:u32,
  each component digest in component order
)
```

The effect digest intentionally normalizes the dependency reference to the absent sentinel.
Dependency identities for
unique/FK guards include this effect digest, so including the eventually assigned token reference
would be circular. The payload digest still covers the actual reference.

```text
D(
  "gpu-db/write001/s7-key-effect/v2",
  exact effect bytes 0..16,
  absent_u32:[u8;4],             // canonical normalization of dependency_ref
  exact effect bytes 20..128,
  zero:[u8;32],
  exact effect bytes 160..192,
  referenced index-descriptor digest:[u8;32],
  each new-component digest in component order
)
```

### Typed-key components and value arena

Each 128-byte component is:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | component reference, equal to directory ordinal |
| 4 | 4 | key-effect reference |
| 8 | 1 | side, `2` (`new`) |
| 9 | 1 | validity: `0` non-NULL, `1` NULL |
| 10 | 2 | flags, zero |
| 12 | 4 | component ordinal |
| 16 | 4 | index-key-column reference |
| 20 | 4 | source final-image catalog-column ordinal |
| 24 | 8 | offset relative to the typed-key value arena |
| 32 | 4 | value byte length |
| 36 | 4 | SQL storage type |
| 40 | 4 | declared type OID |
| 44 | 2 | signed type size |
| 46 | 2 | reserved, zero |
| 48 | 32 | canonical typed-value digest |
| 80 | 32 | component digest |
| 112 | 16 | reserved, zero |

Component ordinals are dense within an effect. Index type and ordinal equal the referenced
key-column descriptor. For maintenance/unique effects, the source image column is that same
target catalog column. For FK effects, the referenced key column belongs to the parent supporting
index while the source image column is the S2 child column; their resolved SQL storage type and
size are equal, while declared domain OIDs may differ exactly as S2 permits.

Value ranges occur in component-directory order, without gaps or overlap, and exactly fill the
typed-key arena. A NULL has zero length at the current arena cursor. A non-NULL value uses the
same logical scalar bytes defined for final-row hashing: exact lengths are 4, 8, 16, 16, or 1 for
the fixed representations, while text is its raw nonempty-or-empty UTF-8 byte sequence. Numeric,
date, timestamp, int2, bool, and UTF-8 domains match the shared image rules.

The typed-value and component digests are:

```text
D(
  "gpu-db/write001/s7-typed-key-value/v2",
  SQL-storage-type:[u8;4],
  declared_type_oid:u32,
  signed_type_size:i16,
  validity:u8,
  value_byte_length:u32,
  exact value bytes
)

D(
  "gpu-db/write001/s7-typed-key-component/v2",
  exact component bytes 0..80,
  zero:[u8;32],
  exact component bytes 112..128,
  referenced key-column descriptor digest:[u8;32]
)
```

Every component is compared against the referenced final-image cell after both have passed their
own scalar-domain checks. Validity and exact logical value bytes must match. This comparison, not
a host SQL value or a later catalog lookup, closes composite order and typed key contents.

### Projection bindings

S7 retains metadata only, never actual `RETURNING` rows. Each projection binding is 128 bytes:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | projection reference, equal to directory ordinal |
| 4 | 4 | statement reference |
| 8 | 4 | SQL-order projection ordinal |
| 12 | 4 | source final-image catalog-column ordinal |
| 16 | 4 | stable column ID |
| 20 | 4 | table reference |
| 24 | 2 | signed `attnum` |
| 26 | 2 | flags, zero |
| 28 | 4 | SQL storage type |
| 32 | 4 | declared type OID |
| 36 | 2 | signed type size |
| 38 | 2 | result format: `0` text, `1` binary |
| 40 | 4 | S2 projection ordinal |
| 44 | 20 | reserved, zero |
| 64 | 32 | projection-name digest |
| 96 | 32 | projection-binding digest |

Projection order, duplicates, wildcard expansion, name, source column, type, OID, and size exactly
equal S2. Current INSERT `RETURNING` has only bound target columns, so every source column is live;
derived sentinels are invalid in S7. Result format is the resolved protocol format and is not
inferred from S2. The name digest uses `identifier_digest` above. The binding digest is:

```text
D(
  "gpu-db/write001/s7-projection/v2",
  exact projection bytes 0..96,
  zero:[u8;32]
)
```

S7 projection digests close the statement-overlay commitment. They also define the logical S6
result identity before S8 exists. For a successful statement, result rows are exactly its S4
`Survives` and `AppliedThenCanceled` entries in source-row order. A
`SuppressedAtStatement` entry never produces a result row, and an abort produces no result
artifact. Each projected cell is read from the strictly decoded S2 resolved catalog-order vector
at that source row; it is not read from the final image because an earlier explicit-transaction
statement may later be canceled.

The canonical successful-statement result digest is:

```text
D(
  "gpu-db/write001/s7-statement-returning-result/v2",
  statement_ref:u32,
  result_row_count:u32,
  projection_count:u32,
  for each projection in SQL order:
    projection_binding_digest:[u8;32],
  for each selected result row in source order:
    source_row_ordinal:u32,
    for each projection in SQL order:
      projection_ordinal:u32,
      source_catalog_column_ordinal:u32,
      stable_column_id:u32,
      attnum:i16,
      SQL-storage-type:[u8;4],
      declared_type_oid:u32,
      signed_type_size:i16,
      result_format:u16,
      validity:u8,                 // 0 non-NULL, 1 NULL
      logical_value_length:u32,
      exact logical value bytes
)
```

Logical scalar bytes and NULL form are exactly the final-row rules above. S2 vector domain checks
run before this traversal. The digest is nonzero whenever projection count is nonzero, including
any future successful zero-row result; this permanently plain, nonempty INSERT profile currently
has one result row per input row on success. With no projections the canonical result identity is
the all-zero digest rather than a hash. S6 `returning_digest` must equal this exact value for
`CommitSuccess` and must be zero for `AbortError`. A coherent change to a value, row selection,
projection order/duplicate, name/type, or text/binary format therefore fails S6 closure even while
S8 is empty.

S8 will separately bind the same projection metadata, formats, row selection, and logical digest
to retained response-image bytes; it may not define a second logical-result digest.

### Final-overlay images

There is exactly one 160-byte image descriptor per table:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | image reference, equal to directory ordinal |
| 4 | 4 | table reference |
| 8 | 4 | role, `1` (`FinalTableImage`) |
| 12 | 4 | flags, zero |
| 16 | 4 | row count |
| 20 | 4 | column count |
| 24 | 8 | offset relative to image arena |
| 32 | 8 | exact encoded image bytes |
| 40 | 24 | reserved, zero |
| 64 | 32 | shared-image layout digest |
| 96 | 32 | full image-content digest |
| 128 | 32 | image-descriptor digest |

Image ranges occur in descriptor order without gaps or overlap and exactly fill the image arena.
The content digest is:

```text
D(
  "gpu-db/write001/s7-image-content/v2",
  encoded_image_bytes:u64,
  exact encoded GPUDBTYPEDIMAGE2 bytes
)
```

The descriptor digest is:

```text
D(
  "gpu-db/write001/s7-image-descriptor/v2",
  exact image-descriptor bytes 0..128,
  zero:[u8;32]
)
```

The nested bytes strictly decode under the shared `GPUDBTYPEDIMAGE2` grammar with role
`FinalTableImage`. Its table references all equal this descriptor's table reference, its columns
are the table block's complete catalog-order column set, and its layout digest equals both the
image descriptor and table block fields. The full content digest also equals both fields.

Image rows biject the table's `New` transitions in transition order. The image contains only the
complete final transaction-overlay contribution, not a copy of unchanged base rows: its row count
equals the table transition count. A table with no surviving transition still carries the
canonical nonempty zero-row image with every catalog column descriptor and zero-row vector. This
keeps schema/image closure exact without making WAL proportional to the base table.

For each source statement, S2 catalog columns exactly match the table image descriptors. For each
surviving source row, every S2 resolved typed value and validity bit equals its transition's image
row. Canceled/suppressed source rows have no image row. Multiple statements targeting the same
table must carry the same target stable binding, display OID, schema digest, dependency/index
catalog, and image column layout.

### Published sequence closure

Every semantics-v2 S2 sequence effect is published and has exactly one S5 entry, one
`PublishedSequence` token, and one statement-dependency use. Private S2/S5 effects, explicit
sequence calls, and unrooted private-sequence aggregate flags reject.

The exact decoded S5 `BinarySequenceValueReference` must bind the aggregate transaction mode and
parent transaction ID, the corresponding S1 statement request digest, statement and expression
ordinals, sequence OID, target display table OID, column ID, S4 disposition/stable row ID, returned
value, and S2 input digest. Published transition transaction IDs are strictly increasing in S5
source order.

For a surviving disposition, `final_value_overwritten` is false and the exact final-image cell is
non-NULL with S2's target storage type and the returned i64 value after S2's already-validated
target coercion. For a canceled or suppressed disposition,
`final_value_overwritten` is true and there is no transition/image cell. `default_expression` is
always true. These rules are complete because later UPDATE/DELETE statements cannot exist in this
semantics profile.

### Index-effect closure

For every transition:

- every target index has exactly one maintenance effect;
- every unique target index has exactly one additional unique-guard effect;
- every S2 foreign key of the source statement has exactly one FK-guard effect using its exact
  supporting parent index;
- no other effect exists.

The table's key-effect range is the concatenation of its transition ranges. Each maintained-index
effect's typed components equal the corresponding final-image columns. Each unique guard uses the
same component tuple as its maintenance effect. Each FK guard uses the S2 child columns in S2
foreign-key order and the supporting parent index's ordered key descriptors. Current S2 admits a
single-column FK; the wire grammar is composite-safe but may encode only the arity which S2
actually owns.

Malformed scalar values and an unsupported or unbound constraint shape reject before an aggregate
can exist. A supported NOT NULL, UNIQUE/primary-key, FK, CHECK, or domain failure instead produces
the one exact terminal-abort form below, so an already-durable published sequence effect remains
closed by S2/S5/status rather than disappearing. The abort carries no transitions or final index
effects; its final failing statement selects one catalog-bound error token and replay reruns the
same typed device validation to reproduce the exact error. A successful aggregate proves every
guard passed and carries all physical index effects. S7 never evaluates a SQL expression on the
host. It binds typed catalog programs/dependencies, admitted outcomes, physical insertions, and
final vectors into one fact set.

### Statement overlay-root chain

For each statement define:

```text
disposition_root =
  D(
    "gpu-db/write001/s7-statement-dispositions/v2",
    statement_ref:u32,
    disposition_count:u32,
    exact referenced S4 entries in source-row order
  )

sequence_root =
  D(
    "gpu-db/write001/s7-statement-sequences/v2",
    statement_ref:u32,
    sequence_effect_count:u32,
    exact referenced S5 entry prefixes and bodies in effect order
  )

dependency_root =
  D(
    "gpu-db/write001/s7-statement-dependencies/v2",
    statement_ref:u32,
    dependency_use_count:u32,
    for each use in canonical order:
      exact 32-byte use,
      referenced dependency-token digest
  )

projection_root =
  D(
    "gpu-db/write001/s7-statement-projections/v2",
    statement_ref:u32,
    projection_count:u32,
    each projection-binding digest in SQL order
  )

overlay_after =
  D(
    "gpu-db/write001/s7-statement-overlay-root/v2",
    overlay_before:[u8;32],
    statement_ref:u32,
    typed_statement_digest:[u8;32],
    s2_record_digest:[u8;32],
    disposition_root:[u8;32],
    sequence_root:[u8;32],
    dependency_root:[u8;32],
    projection_root:[u8;32]
  )
```

The S6 digest is deliberately not an input to `overlay_after`: S6 itself carries
`overlay_after` as its target digest, so including it would be circular. S6 exact bytes remain
covered by the statement resolution and S7 payload digest.

Statement zero's `overlay_before` equals the header initial-overlay root. Every later
`overlay_before` equals the prior statement's recomputed `overlay_after`. The last recomputed
value equals the header final-overlay root. Every S1 root and corresponding resolution field must
equal this chain.

### Data roots and their sole authority

Transition and index-effect roots for one table are:

```text
transition_root =
  D(
    "gpu-db/write001/s7-table-transition-root/v2",
    stable_table_id:u64,
    transition_count:u32,
    each transition digest in stable-row order
  )

index_effect_root =
  D(
    "gpu-db/write001/s7-table-index-effect-root/v2",
    stable_table_id:u64,
    key_effect_count:u32,
    each key-effect digest in canonical effect order
  )
```

The 32-byte initial/final table and database fields are **fixed root identities**, not a second
Merkle algorithm defined by codec 5. Root-descriptor version 1 means:

- the existing sole immutable database-generation builder owns the persistent/structurally
  shared map nodes, table/index/manifest roots, and their canonical 32-byte identities;
- preparation captures the initial root identities from one pinned publication object and
  prebuilds the complete private successor generation;
- S7 records the exact initial and prebuilt-final identities and binds them, plus every semantic
  transition/image commitment, in table manifests, the root-descriptor digest, and the full
  payload digest;
- every table-generation manifest owned by that builder contains the table data generation/root,
  logical row count, and the ordered `(stable_index_id, index_generation, index_root)` entry for
  every owned index; table-root equality therefore covers every maintained index result rather
  than merely its logical effect digest; and
- replay feeds the decoded transition/image/index facts to that same generation builder and
  compares the independently produced table, index, and database identities byte-for-byte before
  publication.

Codec 5 therefore neither fabricates a content root nor defines an alternate root allocator. A
nonempty transition set requires a different final table identity and the monotonic generation
advance already specified. An empty transition set uses equal initial/final table identities,
equal generations, and equal row counts; its transition and index roots still use their exact
zero-count forms. Consumed row IDs remain recorded by the allocator range without pretending
that an aborted/canceled row changed table contents.

If every table has zero transitions, initial and final database root identities are equal. If any
table changes, the final database identity is different and must be the single candidate
persistent-map root whose affected table entries equal the table-block final identities and whose
unaffected entries are structurally shared from the pinned initial database root. The builder,
not a codec-local hash fold, proves that map relationship.

The table manifest digest is:

```text
D(
  "gpu-db/write001/s7-table-manifest/v2",
  exact table-block bytes 0..352,
  zero:[u8;32],
  target dependency-token digest:[u8;32],
  exact table-disposition-reference entries in table order,
  each owned index-descriptor digest,
  each transition digest,
  each key-effect digest,
  image-descriptor digest:[u8;32]
)
```

The root-descriptor digest is:

```text
D(
  "gpu-db/write001/s7-root-descriptor/v2",
  root_descriptor_version:u16,
  catalog_before_epoch:u64,
  catalog_after_epoch:u64,
  catalog_before_digest:[u8;32],
  catalog_after_digest:[u8;32],
  initial_database_data_root:[u8;32],
  final_database_data_root:[u8;32],
  initial_overlay_root:[u8;32],
  final_overlay_root:[u8;32],
  table_count:u32,
  for each table block in stable-table order:
    table_ref:u32,
    stable_table_id:u64,
    data_generation_before:u64,
    data_generation_after:u64,
    initial_table_root:[u8;32],
    final_table_root:[u8;32],
    table_manifest_digest:[u8;32]
)
```

These are fixed content-addressed publication-root identities, not physical GPU coordinates and
not the test-only fabricated roots in the current indexed proof. When semantics 2 is eventually
promoted, the sole immutable snapshot/publication owner supplies the initial identities and the
sole generation builder supplies the prebuilt final identities; replay rebuilds and compares
them. The existing sole publication path may install only that verified final database root.
There is no second root allocator, catalog root, or residency-root authority. Until stable table
IDs and these root tokens are carried by that publication owner, semantics 2 remains
production-ineligible.

### Required validation witnesses — not wire

Opaque catalog, allocator, and publication identities cannot be authenticated by a context-free
codec. Semantics 2 therefore has one mandatory, borrowed
`SemanticsV2ValidationWitness`; it is validation input, not S7 bytes, a fallback lookup, or a
second carrier. A structural decoder may check raw grammar/digests and produce only a quarantined
host owner. Catalog and allocator groups may advance it once into a move-only
generation-pending owner; that owner is the sole input which recovery may give the GPU generation
builder. Only exact comparison with the resulting generation group advances it again into a
publication-eligible, fully witness-validated model. Test re-encoding requires that final state.
The live writer obtains all three groups from its pinned catalog/publication, durable allocator
index, and prebuilt private generation before WAL. Recovery necessarily produces the generation
group by running the reserved builder after pure decode, but still compares it before publication.
The inert checkpoint uses checked-in, independently constructed test witnesses and exposes no GPU
or publication transition.

The catalog, allocator, and generation groups' 16 database-ID bytes equal one another and the
outer `CanonicalPreApplyHeader.identity.database_id` byte-for-byte. No database-ID translation,
display OID, or stable object ID can substitute. The catalog group's epoch/digest and the
generation group's epoch/digest equal one another, both S7 catalog echoes, and the outer
before/after catalog epoch/digest; semantics 2 requires those before/after values to be equal.
The generation stable transaction ID and commit sequence equal the outer header fields.

The pinned catalog group has exact database ID, catalog epoch/digest, and these stable-identity
rows in canonical stable-ID order:

- table: stable/display table identity, resolved schema/name bytes, schema digest, data
  generation/root, catalog-order columns, ordered NOT NULL/CHECK descriptors, and ordered FK
  descriptors;
- index: stable/display index identity, owner stable/display table identity, resolved
  schema/index/table/constraint names, constraint stable/display identity when present, flags,
  raw ordinal, key descriptors, base generation/root, and catalog epoch;
- domain: stable/display domain identity, resolved schema/name, storage shape, catalog generation,
  and ordered domain-constraint descriptors;
- constraint/guard: kind, stable/display identity (zero display only for synthesized table-column
  or domain NOT NULL), qualified name for a named constraint or the exact synthesized-name owner
  fields for NOT NULL, owner stable table/domain identity, source ordinal, typed evaluator/column
  shape digest, program/descriptor root, and catalog generation; and
- sequence: stable/display identity, resolved schema/name, catalog generation, and the descriptor
  identity which S2/S5 use.

An FK descriptor additionally carries its stable constraint ID, raw FK ordinal, exact child and
parent column bindings, parent table stable/display identity, and supporting stable index ID.
These rows are the exclusive source for S7 stable IDs/generations/roots not present in S2. Counts
must exactly equal the deduplicated S2-plus-catalog dependency closure described above; a missing,
extra, reordered, or conflicting row rejects. The validator borrows this already pinned group and
does not allocate from attacker-declared counts.

The durable allocator group has exactly one row-allocator lease witness for each S7 table:

```text
database_id:[u8;16]
allocator_kind:u8 = 1 (TableRow)
stable_allocator_id:u64 = stable_table_id
lease_epoch:u64
lease_start:u64
lease_end:u64
prior_high_water:u64
new_high_water:u64
marker_system_txn_id:u64
marker_commit_sequence:u64
```

Epoch, IDs, bounds, and marker identities are nonzero and nonmaximum. The durable allocator index
must prove the marker is complete, durable, published in the same lineage before the user
transaction can use an ID, nonoverlapping at that epoch, and retained by the active checkpoint.
`prior_high_water <= lease_start < lease_end == new_high_water`. The table's complete
`[row_allocator_before,row_allocator_high_water)` range lies within
`[lease_start,lease_end)`; unused prefix/suffix and every canceled/suppressed member remain
consumed. The range itself never advances or reconstructs allocator state.

The generation group is one immutable builder result with:

```text
root_descriptor_version:u16 = 1
database_id:[u8;16]
catalog_epoch:u64
catalog_digest:[u8;32]
stable_transaction_id:u64
commit_sequence:u64
initial_database_root:[u8;32]
generation_input_digest:[u8;32]
final_database_root:[u8;32]
table_count:u32
for each S7 table in stable-table order:
  stable_table_id:u64
  final_data_generation:u64
  final_table_root:[u8;32]
  final_logical_row_count:u64
  owned_index_count:u32
  for each owned index in stable-index order:
    stable_index_id:u64
    final_index_generation:u64
    final_index_root:[u8;32]
```

`generation_input_digest` is independently computed by the generation builder and recomputed by
the witness validator without any final table/index/database identity in its preimage. It does
not reuse the ordinary transition, key-effect, or index-descriptor digests because those
intentionally bind final index identities. Instead it uses these three reference-neutral inputs:

```text
generation_row_input_digest =
  D(
    "gpu-db/write001/generation-row-input/v2",
    stable_table_id:u64,
    stable_row_id:u64,
    source_statement_ordinal:u32,
    source_row_ordinal:u32,
    catalog_column_count:u32,
    for each catalog-order resolved cell:
      catalog_column_ordinal:u32,
      stable_column_id:u32,
      attnum:i16,
      SQL-storage-type:[u8;4],
      declared_type_oid:u32,
      signed_type_size:i16,
      validity:u8,
      logical_value_length:u32,
      exact logical value bytes
  )

generation_index_shape_digest =
  D(
    "gpu-db/write001/generation-index-shape/v2",
    stable_owner_table_id:u64,
    stable_index_id:u64,
    index_flags:u32,
    null_equality_policy:u8,
    base_index_generation:u64,
    base_index_root:[u8;32],
    key_column_count:u32,
    for each key column in key order:
      key_ordinal:u32,
      owner_catalog_column_ordinal:u32,
      stable_column_id:u32,
      attnum:i16,
      SQL-storage-type:[u8;4],
      declared_type_oid:u32,
      signed_type_size:i16,
      column_name_digest:[u8;32]
  )

generation_index_effect_input_digest =
  D(
    "gpu-db/write001/generation-index-effect-input/v2",
    stable_table_id:u64,
    stable_row_id:u64,
    source_catalog_ordinal:u32,
    generation_index_shape_digest:[u8;32],
    key_arity:u32,
    each canonical typed-value digest in key order
  )
```

Only physical maintenance effects have a generation-index-effect input; UNIQUE/FK validation
guards do not mutate an index generation. The top-level input is:

```text
D(
  "gpu-db/write001/generation-input/v2",
  database_id:[u8;16],
  catalog_epoch:u64,
  catalog_digest:[u8;32],
  stable_transaction_id:u64,
  commit_sequence:u64,
  initial_database_root:[u8;32],
  table_count:u32,
  for each table in stable-table order:
    stable_table_id:u64,
    data_generation_before:u64,
    initial_table_root:[u8;32],
    allocator_before:u64,
    allocator_high_water:u64,
    initial_logical_row_count:u64,
    final_logical_row_count:u64,
    generation_row_input_count:u32,
    each generation_row_input_digest in stable-row order,
    image_layout_digest:[u8;32],
    image_content_digest:[u8;32],
    owned_index_count:u32,
    for each owned index in stable-index order:
      generation_index_shape_digest:[u8;32],
      maintenance_effect_count:u32,
      each generation_index_effect_input_digest in stable-row order
)
```

Generation-row inputs biject S7 transitions and are recomputed from their final-image cells.
Generation-index-effect inputs biject role-1 maintenance effects and are recomputed from their
typed components plus the referenced base index shape. None of these three digests traverses an
index descriptor, codec key-effect digest, final generation, or final root, so the generation
builder's inputs are acyclic. Preparation first seals these neutral facts, then prebuilds every
index/table/database output, then populates the final descriptor/root fields, and only then
computes the ordinary codec key-effect, transition, manifest, root-descriptor, payload, and
aggregate digests. No final identity is needed to derive the fact set which creates it.

The initial database/table/index fields equal the pinned publication and catalog groups. Every
builder output equals the corresponding S7 final field. Its table manifest necessarily contains
the complete ordered final-index tuple above; its database map contains each affected final table
entry and structurally shares every unaffected entry from the pinned initial root. On abort, all
output identities/generations/counts equal their initial values and the input digest has zero
transitions/effects. On success, the builder must have prebuilt all changed table and index
generations before WAL. Stable transaction ID and commit sequence equal the bound outer header;
the latter is the `created_by` value for every new version.

This external equality is the semantic root check. A hostile test may coherently replace an opaque
wire root and recompute every codec-owned manifest/root/payload/section digest; validation against
the unchanged independent witness must still reject. Conversely, a context-free structural
decoder claims only internal digest coverage and never claims it derived or authenticated an
opaque publication root.

### S7 payload digest

After every subordinate digest and root is populated, the payload digest is:

```text
D(
  "gpu-db/write001/s7-payload/v2",
  total_bytes:u64,
  exact header bytes 0..568,
  zero:[u8;32],
  exact header bytes 600..640,
  exact bytes 640..total_bytes
)
```

This digest covers the root-descriptor digest and every actual reference, including dependency
references deliberately normalized out of key-effect digests. It must be nonzero. The existing
aggregate section root additionally covers the exact S7 payload under the aggregate codec's
unchanged section/root domains.

### Whole-aggregate closure

Semantics version is dispatched before scalar or section validation. The semantics-1 measure,
encoder, decoder, roots, global allocator rules, provisional S4 reference rules, and accepted
bytes are unchanged. Semantics 2 uses a separate scalar validator and S4/S7 authority; a shared
validator may not reinterpret one version's allocator or sentinel rules as the other's.

For semantics 2:

- S1, S2, S4, S5, S6, S7, and aggregate/header counts close exactly as specified above; S3 is
  physically empty.
- The sum of per-table allocator lengths equals the S4 count and aggregate original row count.
  Aggregate and outer global allocator values are zero. Every table range is covered by its
  independently durable lease witness; S7 is only a cross-check.
- S4 survivors biject all transitions. Because only typed INSERT exists, no additional
  transition may appear and aggregate final-transition count equals the exact survivor count.
  The exact success/abort matrix below determines whether that count is all input rows or zero.
- Table blocks biject distinct S2 targets; index descriptors biject the deduplicated union of S2
  target and FK-supporting indexes; dependency tokens/uses biject their stated S2/catalog/effect
  sources. The pinned catalog witness rejects extra or missing semantic objects.
- Every initial and final table/index/database identity plus each generation-builder input matches
  the required external witness before the decoded model can escape quarantine.
- Aggregate catalog/reset/private-sequence flags and matching outer content bits are zero.
  Published-sequence flags are set exactly when S5 is nonempty. `RETURNING` flags are set exactly
  when at least one S2 projection exists.
- Exactly one aggregate mode bit, autocommit or explicit, is set. Outer row content is always set;
  outer published-sequence and `RETURNING` content mirror their aggregate flags, all other content
  bits are zero, and the existing codec-5/first-writer-epoch control bits retain their current
  meanings.
- During this S4/S7 checkpoint S8 entry count and payload length are zero, every S6
  response-retained bit is zero, and the retained-response aggregate flag is zero. The later
  independently frozen S8 contract may enable those already reserved retention facts without
  changing S7 or admitting another statement class.
- Outer operation count equals the exact physical fragment count, outer and aggregate table-block
  counts equal S7 table count, and stable transaction IDs match.
- Header catalog echoes match the outer envelope and remain unchanged. Database-local stable
  table/object identities and display OIDs are interpreted only under the outer database ID.
  Cluster/timeline/format/leader identities bind log lineage; they never substitute for an
  object ID and are not duplicated in S7.

The canonical aggregate request identity is:

```text
D(
  "gpu-db/write001/aggregate-request/v2",
  transaction_mode:u8,          // 1 autocommit, 2 explicit
  statement_count:u32,
  for each statement in order:
    statement_ordinal:u32,
    S1 request_digest:[u8;32],
    projection_count:u32,
    each projection result_format:u16 in SQL order
)
```

It equals the outer pre-apply request digest and STATUS2 request digest. STATUS2 aggregate root
equals the recomputed aggregate root; the terminal outer outcome target digest also equals that
aggregate root. Request identity is therefore retry identity, while the aggregate root remains
the exact encoded semantic target; neither substitutes for the other. The resolution request
digest is an echo, never another authority: it equals the referenced S1 request digest, which
equals the typed statement digest. The aggregate request additionally owns protocol result formats.
A retry using the same stable transaction ID and logical statement with a changed text/binary
Bind format therefore has a different request digest and is rejected against durable history; it
cannot silently receive or replace the originally claimed response contract.

An autocommit aggregate has exactly one statement. An explicit aggregate may have multiple typed
INSERT statements and uses their statement order and one overlay-root chain. Because S2 owns
plain nonempty INSERT and no `ON CONFLICT` action, `CommitNoOp` is invalid in every S6 entry and in
the outer outcome. Exactly these two matrices are legal:

| Terminal form | Per-statement S6 | S4 dispositions | S7 data/index state |
|---|---|---|---|
| `CommitSuccess` | Every statement is `CommitSuccess`; affected rows equal its nonzero input count; SQLSTATE absent; constraint ID zero; target is its overlay-after; RETURNING digest is the exact logical result digest. | Every row of every statement is `Survives`. | Every row has one `New` transition; all table/index final generations and roots match the prebuilt successor; publication is permitted. |
| `AbortError` | Exactly the final statement is `AbortError`; every earlier statement is the successful form above. The final statement has affected rows zero, RETURNING digest zero, and the exact error binding below. No later statement exists. | Every earlier statement row is `AppliedThenCanceled`; every final-statement row is `SuppressedAtStatement`. No other tag or mixture exists. | Transition and key-effect counts are zero; every table/index generation/root and database root remain equal to their initial values; publication of S7 data is forbidden. Durable S5 sequence transitions remain closed and are not rolled back. |

The final failing resolution has exactly one terminal-error token/use. Its source row and source
ordinal select the first failure produced by the semantics-v2 typed device validator under the
pinned catalog descriptor's canonical evaluator order. The S6 SQLSTATE/kind/source and stable
constraint identity are exactly:

| SQLSTATE | Required token | Source ordinal | S6 `constraint_id` |
|---|---|---|---|
| `23502` | `NotNullGuard`, or `DomainConstraintGuard` classified as domain NOT NULL | target catalog-column or domain-constraint ordinal | token stable guard/constraint ID |
| `23505` | `UniqueKeyGuard` | target-index raw ordinal | descriptor stable constraint ID when present, otherwise descriptor stable index ID |
| `23503` | `ForeignKeyGuard` | S2 foreign-key raw ordinal | stable FK constraint ID in the matching pinned catalog FK row |
| `23514` | `CheckGuard`, or `DomainConstraintGuard` classified as domain CHECK | raw table-check or domain-constraint ordinal | token stable constraint ID |

No other abort SQLSTATE is representable in semantics 2. Unsupported evaluator shapes,
coercion/parser errors, serialization failures, internal errors, and an error without the exact
catalog witness reject before this aggregate language is entered. The final S6 error token,
source row/ordinal, SQLSTATE, and constraint ID must reproduce byte-for-byte when the same typed
operator is rerun; mismatch is corruption or semantic-version skew.

The outer `CommitSuccess` has no SQLSTATE/constraint ID. For autocommit its affected-row count
equals the sole S6 count; for explicit mode it equals aggregate final-transition count, which here
also equals the sum of all statement input counts. The outer `AbortError` has zero affected rows
and copies the final S6 SQLSTATE and constraint ID exactly. In both forms its target digest is the
aggregate root.

STATUS2 statement count equals aggregate statement count. At this checkpoint response-artifact
count and S8 are zero. Statement-outcome root remains the S6 section root; response root remains
the unchanged formula over S6 and canonical empty S8 whenever any statement has `RETURNING`, and
is zero otherwise. The outer returning digest equals that status response root in both success and
abort forms. Thus an outer abort may have a nonzero response-contract digest even though the final
failing S6 has no result: it commits the exact prior statement outcomes, projection formats, and
absence of retained S8 bytes, not a fabricated result row.

### Checked decode and bounded allocation

All arithmetic is checked in `u64`. Before converting any count or offset to `usize`, the reader
verifies:

1. aggregate chunk/status/root framing and version dispatch;
2. the fixed 640-byte S7 header, known flags/tags/reserved bytes, total length, and catalog/root
   scalar minima;
3. all fourteen region multiplications, additions, canonical adjacency, and the exact final end;
4. fixed-entry scalar domains, canonical order, insertion-point ranges, and reference bounds;
5. arena adjacency, every nested image/value raw geometry, and exact persistent/scratch measures;
6. the aggregate/S7 payload digests needed to reject corrupted bytes before reservation.

Pass zero performs these checks by bounded streaming with fixed stack buffers and no heap
allocation. It may rescan chunk-backed sections to prove nonlocal bijections; it may not build an
offset table, section copy, hash map, or attacker-sized bitmap. Sorted directories and dense
references make duplicate/missing proofs allocation-free.

Only after pass zero succeeds may the reader make fallible exact reservations. Its measurement
owns:

- one exact persistent slot array for every retained S1/S2/S4/S5/S6/S7 logical directory;
- all strictly decoded move-only S2 models and final shared-image vector owners;
- exact name/text/value bytes retained by those models;
- the final compact table/statement/dependency/index/transition/key/projection ownership graph;
  and
- the maximum, not sum, of mutually exclusive cross-chunk scratch owners.

The reusable scratch maximum is the greatest of the largest S2 record (bounded by 16 MiB), largest
nested image copy plus the shared image decoder's measured scratch, largest other variable source
body, and any streaming encode/compare scratch. S3 contributes zero. Fixed S5/S6 stack buffers do
not become heap terms. Persistent owners already alive while scratch is used are counted
concurrently. Every reservation uses the exact measured layout and a failed allocation drains
already-created owners without exposing a partial semantic object.

After those exact reservations exist, later passes strictly decode S2 and nested images into the
reserved owners, verify every codec-computable subordinate digest/root formula, and close all
S1--S7 bijections. The borrowed catalog and allocator-lease witnesses are then checked exactly as
specified above before the quarantined decode can become the move-only generation-pending owner.
Only that owner may enter an already reserved GPU generation build. Exact generation-witness/root
comparison is required before publication eligibility. Opaque publication-root equality is
claimed only by that final phase. Structural or witness failure drops the entire reserved graph
and any launched builder work drains before its owners are released. No attacker-controlled
allocation is made from a count, length, or decoded collection size which pass zero did not
already measure and bound.

The aggregate's existing 64-MiB envelope ceiling remains the outer byte bound. Counts whose
minimum fixed bytes cannot fit their declared region reject before allocation. A directory count
or arena length which fits `u64` but not the implementation address space rejects before
conversion. Zero-row images remain nonempty, and zero-count ranges retain their canonical
insertion-point start.

Test-only re-encoding is permitted only from the fully validated retained model and must reproduce
every S1--S7 byte exactly. Production code exposes no reencoder, raw-body extractor, alternate
carrier, WAL constructor, recovery conversion, apply operation, or publication capability for
semantics 2 at this checkpoint.

### Required golden and hostile evidence

The implementation checkpoint must pin, as checked-in constants rather than regenerated expected
values:

- one minimal one-table/one-statement typed-abort vector with every row suppressed, its exact error
  binding, a canonical zero-row final image, unchanged table/index/database outputs, and a prior
  durable row-allocator lease witness;
- one explicit multi-statement abort whose earlier successful rows are canceled and whose final
  failing statement follows a published sequence default, proving that the sequence remains
  durable while data/index roots do not change;
- one successful multi-statement/two-table vector whose statement order interleaves the tables,
  with NULL and non-NULL composite keys, unique and FK equality-guard
  participation/NULL-suppression, `RETURNING` duplicates and mixed result formats, and a published
  sequence default;
- exact S7 bytes, total length, every directory offset/length, root-descriptor digest, S7 payload
  digest, aggregate section root, and witness-validated decode/re-encode equality for all three;
  and
- independent expected digests for one table, index descriptor, transition, typed key, dependency
  token, image content, statement overlay chain, logical RETURNING result, and generation input.
  Table/index/database root values are checked-in identities supplied by the independent test
  generation witness, not codec-derived digests.

Hostile cases must independently mutate/re-hash every magic/version/tag/flag/reserved field,
count/width multiplication, offset base, gap/overlap/end, dense reference, insertion-point range,
sort key, duplicate/missing bijection, table-local allocator boundary, S4 survivor/cancellation
reference, resolution/S1 request-digest equality, S2/S5/S6 binding, dependency
floor/access/dedup rule, stable-ID/display-OID distinction,
qualified-name component/length, catalog/index/FK owner cross-equality, base/final index generation,
composite key order/type/NULL participation, image layout/content/cell equality, projection
format/order/duplicate, logical RETURNING row/value identity, terminal outcome matrix/error
identity, every subordinate digest, root chain, and the full payload digest. Re-hashing only the
outer section must not conceal a forged inner fact. A coherent substitution of opaque
table/index/database roots with every dependent wire digest repaired must reject against the
unchanged generation witness. Missing/stale/overlapping allocator leases and a forged
generation-input witness reject before model release. One-byte-below persistent and scratch
bounds, injected allocation failure at each owner, cross-chunk split at every fixed header
boundary, and immediate clean retry prove bounded failure drain. Source guards prove S3 and
unsupported tags remain absent and semantics 2 has no live caller.

## Future S8 and replay requirements — not wire

No S8 magic, version, header width, entry width, enum/tag, digest domain, or arena grammar is
allocated here. Its later contract must biject retained S6 outcomes with typed response images,
including a nonempty zero-row `RETURNING` artifact, exact S7 projection layout and result formats,
result digest, and abort/no-artifact rules.

Pure decode must eventually destructively produce one move-only, generation-pending
`AggregateReplayTxn` with closed typed statements, dispositions, published sequence effects,
final overlay, and retained responses. It may not retain raw S2 bytes, current command/WAL
carriers, `WriteDelta`, SQL text, host row matrices, a reencoder, or an extracting escape hatch.
Published sequence references are validated against durable history after complete pure decode
and before GPU allocation. Replay uses the same typed GPU operators in statement order; only the
exact generation-witness comparison can advance the owner to publication eligibility, and
publication follows only after every statement, final-root, outcome, and response comparison
passes.

The later complete writer accounting includes the live prepared plan, exact aggregate/outer
buffers, retained response bodies, maximum interactive response scratch, encoder scratch, device
operators, result capacity, status, and publication capacity. After WAL, no fallback, reprepare,
alternate encoding, or alternate apply is legal.

## PLAN-owned wire-freeze gates

Before any writer could emit aggregate semantics 2, or any normative semantics-v2 S4/S7 reader,
digest, or replay owner could be added, WRITE-001 had to freeze and independently audit one
internally complete exact S7 contract. That historical gate is satisfied. Further implementation
and supported-class breadth remains WIP inside the one integrated WRITE-001 milestone candidate;
it is not a new accepted checkpoint. The frozen contract assigns every magic/version, numeric tag
and flag mask; proves every
width and offset sum; defines every offset base, ordering, absence, reference, and bijection rule;
pins the digest primitive, unique domain, and byte preimage for every digest; closes every
dependency-token, index-descriptor/key, projection, and supported statement-class payload
grammar; resolves database-local versus cluster-global catalog/root authority; states checked
decoding and allocation bounds; and provides golden vectors plus hostile decode/re-encode
evidence. Unsupported semantic classes reject and receive no opaque payload.

Only after that gate may the same checkpoint implement inert normative S4/S7, bind S4's frozen
numeric form to the exact S7 table/transition directory, and reconcile the explicitly
non-normative S1--S6 scaffold above. No checkpoint may leave two S4 acceptance authorities.

S8 and the final replay-class payloads receive the same design/audit gate in their later
PLAN-owned checkpoint before any writer or live historical translator can emit semantics-v2
bytes. Until all such gates and the final acceptance criteria pass, codec-5 semantics 2 remains
production-ineligible.
