# Proposal: non-int4 point-lookup indexes (bigint, text, uuid, numeric)

> **Status: PROPOSAL (not accepted).** Forward-looking design for review. Evidence below is grounded in the
> current kernel set on `main` (enumerated from `cached_function(c"...")` names in
> `crates/execution/src/lib.rs`) and the resident routing in `crates/engine/src/engine_retained_read.rs` /
> `engine_resident_probe.rs`. No code is changed by this doc.

## Problem
The fast, O(1) GPU **point-lookup index** path is **int4 only**. The only index-probe kernels that exist are:
- `gpu_db_resident_i32_index_probe`
- `gpu_db_resident_i32_index_probe_dense`

There is **no** `i64` / `text` / `uuid` / `numeric` index probe. The resident retained-read route confirms the
gate — it "currently support[s] only int4 equality projection routes" (`engine_retained_read.rs`), and the
router declines a non-int4 (or non-unique) key to the **scan** path.

Consequence: a primary-key point read — the canonical OLTP operation — on a **bigint**, **uuid**, **text**, or
**numeric** key runs as an **O(rows) full-column scan**, not an O(1) index probe. For modern PostgreSQL schemas
this is the *common* case, not an edge case (bigint `IDENTITY`/`bigserial` PKs, uuid PKs, natural text keys).
Correct, but the wrong complexity for OLTP on large tables.

## Evidence from the current code — the gap is narrow and the hard primitives exist
The engine already has broad non-int4 GPU support; the gap is *specifically* the resident point-lookup index.
Mapping the current kernel set by type × operation:

| Key type | Equality scan (O(rows)) | **Point-lookup index (O(1))** | Hash-table build+probe (joins) | Agg / sort / group |
|---|---|---|---|---|
| **int4 (i32)** | `equal_project`, `equal_any_project` | **`index_probe`, `index_probe_dense`** | `hash_join_build_i32`, `hash_join_probe_i32` | `i32_sum/minmax_at_indices`, radix+bitonic, `grouped_hash_*` |
| **bigint (i64)** | `i64_compare_scalar_to_mask` | **MISSING** | `hash_join_build_i64_nn`, `hash_join_emit_i64_nn` | `i64_minmax/sum_at_indices_i128`, `bitonic_sort_i64_step`, `widen_i32_to_i64` |
| **text** | `text_eq_scalar_to_mask` (+ `text_like_scalar_to_mask`) | **MISSING** | `hash_join_build_text`, `hash_join_probe_text`, `hash_join_emit_text_nn` | `bitonic_sort_text_step`, `mark_new_distinct_text` |
| **uuid (b128)** | `uuid_compare_scalar_to_mask`, `uuid_compare_columns_to_mask` | **MISSING** | none | min/max via the b128/i128 slot machinery |
| **numeric (i128)** | `i128_compare_scalar_to_mask`, `i128_compare_columns_to_mask` | **MISSING** | none | `i128_sum/minmax_partials_at_indices`, `group_by_numeric_minmax_lo`, `i128_mul` |

Two facts this table establishes:
1. **Non-int4 equality already runs on the GPU** (the `*_compare_scalar_to_mask` scan kernels) — these lookups
   are *not* falling to the host; they're just O(rows).
2. **The hardest primitive for an index — a GPU hash table with build + probe — already exists for i32, i64,
   and text** (the hash-join path). A point-lookup index is "hash key → probe hash table → compare on
   collision," which is exactly what `hash_join_build_*` / `hash_join_probe_*` do. So bigint and text indexes
   are an **adaptation of existing machinery**, not greenfield. Key-widening/packing primitives also already
   exist (`build_wide_key`, `widen_col_to_i64`, `pack_two_int4_cols`, `pack_two_cols_i128`).

## Why the int4 index doesn't simply "widen"
The int4 index packs `slot = (key << 32) | (row + 1)` — the 32-bit key is stored *inline* in the high half of
a 64-bit slot, so the probe needs no separate key storage and no collision-compare (check `slot >> 32 == key`).
This is an int4-specific optimization that **cannot widen**: a 64-bit (bigint), 128-bit (uuid/numeric), or
variable-length (text) key + a row id no longer fit one 64-bit slot. The general types require the standard
**hash-bucket index**: hash(key) → bucket; store key (or a key offset) + row id; compare the full key on
collision. So this is a *new index layout*, reusing the existing hash-table + per-type hash/compare kernels.

## Goals / non-goals
**Goals:** O(1) GPU point-lookup indexes for `bigint`, `text`, `uuid`, and `numeric` unique keys, byte-identical
to the scan path, routed automatically when a unique index exists; reuse the existing hash-join build/probe and
the lpb result-path wins.
**Non-goals (this proposal):** range indexes (B-tree-like ordered scans), non-unique secondary indexes, and the
write-path index-maintenance design beyond noting the dependency (that rides R3).

## Proposed design
1. **General resident hash-index layout** (replaces the int4 inline packing for non-int4): `hash(key) → bucket`,
   buckets store `(key | key_offset, row_id)`, open-addressing or chaining, collision-compare via the existing
   per-type compare primitive. The int4 path keeps its inline-packed fast index unchanged.
2. **Per-type probe kernels** — `gpu_db_resident_{i64,text,uuid,numeric}_index_probe[_dense]`, modeled on
   `index_probe_dense`:
   - **Dense single-pass compaction transfers directly** (it's about output slotting, type-agnostic): for a
     unique key, thread *i* writes `result[needle_i]` — no `atom.global.add`, no host scatter. Reuse the
     dense-emit structure proven on int4.
   - The **probe body** (hash + collision-compare) is type-specific; reuse `hash_join_build/probe_i64`/`_text`
     hashing for bigint/text, and the i128/b128 representation + `i128_compare`/`uuid_compare` for
     numeric/uuid (build the missing hash-probe for the 128-bit slot, shared by uuid and numeric).
3. **Result path** — reuse the lpb arc machinery (batched result + columnar drain + O(n) scatter), with a
   per-type *value* carry (the `Vec<i32>` carry is int4-specific → `i64` / bytes for the projected columns;
   text projection already exists via `equal_any_project_text`). See
   `docs/optimizations/cross-kernel-result-path-transfer.md`.
4. **Routing** — extend the resident route's index-vs-scan decision (today: int4-unique → index, else scan) to
   admit non-int4 unique keys to their per-type index probe, with the same decline-to-scan fallback for
   non-unique / un-buildable columns.

## Phasing (value × cost, evidence-based)
1. **bigint (i64) — first.** Highest value (the modern PK default) and lowest cost: fixed-width, and the GPU
   hash build+probe already exists for i64 in the join path (`hash_join_build_i64_nn`/`emit_i64_nn`). Mostly
   adaptation + the dense-emit wrapper.
2. **text — second.** Very common natural key; hash build+probe already exists (`hash_join_build_text` /
   `probe_text`). Needs the variable-length key storage (key heap / offsets) tied into the resident layout.
   (Bonus: this is the same index a Redis-style string-key KV tier needs — see `docs/future/redis/`.)
3. **uuid + numeric — third.** Both 128-bit; share the i128/b128 slot + compare machinery
   (`i128_compare_scalar_to_mask`, the b128 uuid slots). Needs the 128-bit hash-probe built (no join probe for
   these today). Numeric is the lowest priority as a *point-lookup* key (rarely an equality/PK column).
4. **composite (multi-column) — follow-on.** Partial primitives exist (`build_wide_key`, `pack_two_int4_cols`,
   `pack_two_cols_i128`); compose two keys into a wide key, then index the wide key.

## Risks / open questions
- **Collision handling + NULL semantics** must match the scan path exactly (3VL: NULL never equals a needle;
  note `uuid` already has the "no-VM memcmp + validity bit 0" pattern, and the grouped path remaps
  `hash i64::MIN → 0` and guards "a text key never aliases i128::MIN").
- **Variable-length text keys** need a device key heap + offset table in the resident layout (the join path
  stores text inline for its build side — confirm reuse vs. a resident-specific layout).
- **Index build + write-path maintenance** (insert/update/delete keep the index current) rides the unbuilt
  write path (R3); this proposal scopes the read/probe side and flags the maintenance dependency.
- **Index packing redesign** for the general bucket layout must not regress the int4 inline-packed fast path
  (keep int4 on its existing index).

## Validation
Byte-identical differentials per the lpb standard: `<type>_index_probe == <type>_compare_scalar_to_mask scan`
over NULL / NULL-as-0 / absent / duplicate-needle / multi-row cases, plus facade + protocol byte-identity.
Non-vacuous gates (a route-hit assertion so a silent scan-fallback can't pass — the lesson from the wave's
faked-throughput audit).

## Bottom line
The non-int4 point-lookup index is the one **specific** gap behind the int4-only fast path: equality already
runs on the GPU (as scans), and the GPU hash build+probe — the hard part of an index — already exists for
**bigint and text** (joins) and the compare/representation primitives exist for uuid/numeric. So this is mostly
**adapting existing machinery + the dense-emit/result-path wins**, phased bigint → text → uuid/numeric →
composite. It converts the canonical OLTP PK read from O(rows) to O(1) for the key types real schemas actually
use.

## Discipline (charter)
ASCII-only PTX (`ptxas -arch=sm_70` check before launch); GPU tests under `timeout`, never `--gpu-reset`;
independent adversarial audit on the new probe kernels (never self-audit); each per-type probe must hold the
`== scan` byte-identity differential green; commit/push/merge each verified increment.
