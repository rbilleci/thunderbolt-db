# WRITE-001 codec-5 semantics-v2 boundary

This is a stable ownership and semantic boundary beneath
[`write-001-general-insert-pipeline.md`](write-001-general-insert-pipeline.md). It is not a second
task ledger; [`../PLAN.md`](../PLAN.md) alone owns unfinished work and sequencing.

Only sections explicitly labelled **normative wire** below are byte-stable. The S7, S8, and final
replay material is deliberately a requirements inventory, **not** a wire format. Current accepted
owners do not define catalog-object bodies, dependency tokens, final overlay roots, non-INSERT
class payloads, or result equality precisely enough to freeze those bytes without inventing a
second semantic authority. No implementation may infer tags, widths, domains, or payload bytes
from that inventory.

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

## Normative wire: semantics-v2 S4 identity

Aggregate semantics value `2` is reserved but is not eligible for serialization. S7, S8,
statement resolution, dependency/index bodies, and the final replay IR have no allocated wire
format yet. When semantics 2 becomes eligible, the existing 96-byte aggregate header and eight
section headers remain its outer framing; the aggregate header's `allocator_before` and
`allocator_high_water`, plus the outer `CanonicalPreApplyHeader.allocator_high_water`, are
canonical zero sentinels. Stable row identity is table-local under ADR-014, so the future S7 table
block is the only row-allocator authority. Semantics 1 retains its existing nonzero global
allocator range and byte-for-byte behavior.

S4 keeps its 64-byte width:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 4 | statement ordinal |
| 4 | 4 | source-row ordinal |
| 8 | 8 | stable row ID |
| 16 | 1 | disposition: `1` survives, `2` applied then canceled, `3` suppressed at statement |
| 17 | 1 | flags, exactly zero |
| 18 | 2 | reserved, zero |
| 20 | 4 | future S7 table-block reference |
| 24 | 4 | future S7 transition reference |
| 28 | 4 | reserved, zero |
| 32 | 32 | statement digest |

Every disposition has a live table-block reference. A surviving row has a live transition
reference; canceled and suppressed rows use the absent-`u32` sentinel. Each S2 statement targets
the same stable table eventually named by its S7 table block. For each table, S4 row IDs in
statement/source order exactly cover `[row_allocator_before, row_allocator_high_water)`, whose
length equals that table's inserted-disposition count. The sum of the table counts equals the
aggregate original inserted-row count. Every S4 statement digest equals the corresponding S1
statement digest.

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

One `GPUDBTYPEDIMAGE2` columnar codec is shared by future S7 final table images and S8 retained
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

## Future S7, S8, and replay requirements — not wire

No S7 or S8 magic, version, header width, entry width, enum/tag value, flag bit, digest domain,
digest preimage, arena grammar, or class-payload byte is stable today. The previously proposed
offset tables were removed because current accepted owners cannot justify them independently.
Historical or future code must not treat this section as a decoder specification.

The later S7 final-overlay contract must close, at minimum:

- stable-table ordering and table-local row allocator ranges;
- one final transition per surviving logical row and exact S4/transition bijection;
- new/replaced/deleted lifecycle, base identity, final writer, and final image position;
- table/object dependency floors and access strength without duplicate token authority;
- complete composite index identity, NULL policy, and old/new typed key effects;
- one closed typed statement resolution per S1/S6 entry;
- exact published/private sequence effect and overwrite/cancellation closure;
- initial/final data and catalog roots plus every table image digest.

The later S8 contract must biject retained S6 outcomes with typed response images, including a
nonempty zero-row `RETURNING` artifact, exact projection layout, result format, result digest, and
abort/no-artifact rules.

Pure decode must eventually destructively produce one move-only `AggregateReplayTxn` with closed
typed statements, dispositions, sequence effects, final overlay, and retained responses. It may
not retain raw S2/S3 bodies, current command/WAL carriers, `WriteDelta`, SQL text, host row
matrices, a reencoder, or an extracting escape hatch. Published sequence references are validated
after complete pure decode and before GPU allocation; private effects never perform an external
lookup. Replay uses the same typed GPU operators in statement order and publishes only after all
statement, final-root, outcome, and response comparisons pass.

Decoder pass zero verifies chunks, status, and roots without allocation. The next pass validates
all counts, minima, lengths, and offsets with checked `u64` before count-to-`usize` conversion.
Later passes use fallible exact reservations, at most one reusable maximum S2/S3 source scratch,
streaming canonical comparison, and destructive movement into the final IR. Writer accounting
must include the live prepared plan, exact aggregate/outer buffers, retained response bodies,
maximum interactive response scratch, and encoder scratch. After WAL, no fallback, reprepare,
alternate encoding, or alternate apply is legal.

## PLAN-owned wire-freeze gates

Before any writer can emit aggregate semantics 2, or any normative semantics-v2 S4/S7 reader,
digest, or replay owner is added, the next WRITE-001 checkpoint freezes and independently audits
one internally complete exact S7 contract. That gate covers only semantic classes implemented in
the same checkpoint and assigns every magic/version, numeric tag and flag mask; proves every
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
