# NULL Representation & Three-Valued Logic

Status: **active design** (the critical-path unblock for unification; the first
milestone of M3). Short design — written before slicing, per the 2026-06-22
handover.

NULL is the storage-level gate that sits **below both the CPU and GPU executors**.
It is not an executor feature gap: the executors already interpret every operator
generally; what is missing is a *representation* for "no value" in the value model,
in storage, and (already present) on the wire. This document fixes that
representation and stages the three-valued-logic (3VL) semantics that follow.

Read this before adding `SqlValue::Null`, a null bitmap, an `IS NULL` operator, or
any host-side null check. The charter applies in full: the null bit is read **on
the GPU** by the kernels (comparisons / joins / aggregates honor NULL on-device),
never as a host pre/post-filter (`00-gpu-native-principles.md`).

---

## 1. Why this is the critical path (and why it is cheaper than it looks)

Unification (§9.1 — back the wire server with the engine) means the **engine's
store becomes the single source of truth**. That flip cannot precede NULL: the wire
protocol *requires* NULL (`DataRow` field length `-1`, bound NULL params) and the
engine's value model cannot represent it, so a flip today would regress every
NULL-bearing result. NULL is also the gate for the deferred join work (outer joins;
the latent NULL-key correctness requirement — see
`20-…`/`gpu-joins-m5`), `AVG`-over-empty, and the catalog/function engine (doc 20:
catalog columns and functions legitimately return NULL).

**Two findings from grounding the codebase make M3 smaller than the handover
assumed:**

1. **The wire NULL codec already exists.** `data_row_with_formats`
   (`crates/protocol/src/lib.rs:1133`) takes `&[Option<String>]` and writes a `-1`
   field length for `None` (`:1159`). Bind-parameter decoding already maps a `-1`
   length to `None` (`:495`). So "the wire NULL codec" is **done at the protocol
   layer**. The only wire-side gap is that the value model has nothing to *map to*
   `None`: the engine→wire path (`crates/server/src/lib.rs:264`) wraps every cell in
   `Some(db_value_text(value))` because `DbValue` has no `Null`.

2. **The legacy server does not store user-data NULL either.** It stores
   `Vec<Vec<SqlValue>>` (no `Null`), its 352 golden scenarios exercise NULL **only
   in catalog introspection** (`IS NULL` on system columns), and it *rejects* NULL
   bound params (`0A000`, "NULL … parameters are not supported"). So the parity bar
   for user-data NULL is low, and the store-flip risk the handover flagged is
   correspondingly lower. The hard work is the GPU-resident representation, not
   protocol parity.

---

## 2. The value-model decision: a typeless `SqlValue::Null`

NULL is added as a **single typeless variant** `SqlValue::Null` (mirrored by
`DbValue::Null` at the façade boundary, `crates/facade/src/lib.rs:60`), **not** as
`Option<SqlValue>` cells.

- *Why a variant, not `Option<SqlValue>`:* every cell, key, projection, and
  aggregate accumulator is `SqlValue` today; wrapping them all in `Option` is
  mechanical churn across the whole engine with no representational gain. A variant
  is local and the compiler drives every consumer to a decision (the same
  compiler-driven widening that the §9.6 decomposition used successfully).
- *Why typeless:* SQL's `NULL` literal is itself of unknown type until coerced. The
  column's declared `SqlType` is unchanged and is always available from the schema,
  so the value need not carry a type. There is **no** `SqlType::Null`.
- *Derived `Ord`/`Eq`:* `SqlValue` derives `Ord, PartialOrd, Eq, PartialEq`. These
  define an **internal total order** used for value-index keys and dedup — *not* SQL
  semantics. `Null` is placed **first** in the enum (discriminant 0, sorts lowest)
  so the internal order stays total and deterministic. SQL equality and ordering
  (where `NULL = NULL` is UNKNOWN and NULL placement follows `NULLS FIRST/LAST`)
  must **never** use the derived traits; they route through the explicit functions
  in §3. Two `Null`s being `Eq`/equal *for a HashMap/BTree key* is correct and
  desirable (it is a data-structure identity, not a SQL truth value).

### Wire boundary

Add `pg_adapter::db_value_text_opt(&DbValue) -> Option<String>` returning `None` for
`DbValue::Null` and `Some(db_value_text(v))` otherwise. The engine→wire loop
(`crates/server/src/lib.rs:267`) uses it; the existing `-1` codec does the rest.
`db_value_text` keeps returning `String` for non-null callers (three façade tests);
its `Null` arm is documented as wire-unreachable.

---

## 3. Three-valued logic is GPU-native; the host path is legacy

The charter is GPU-native (`00-gpu-native-principles.md` Rule 1): the relational data
path — including 3VL — runs on the GPU; the CPU host relational path is tracked **debt
to be retired by unification, not a place to invest.** So 3VL belongs on the GPU (§4),
and the host helpers get the **bare minimum** to stay total when the shared value model
gains a variant — *not* a 3VL feature build.

The load-bearing rule, evaluated **on the GPU**: *any comparison/arithmetic with a NULL
operand yields UNKNOWN/NULL, and a `WHERE` predicate that is UNKNOWN excludes the row.*

- **GPU (the real work, §4, slices 2–4).** The kernels read the per-column null bitmap
  and apply the rule on-device: a row whose operand bit is null does not satisfy a
  comparison; an equi-join excludes NULL keys (the latent join-key gate); aggregates
  skip NULL inputs (`COUNT(*)` excepted); ORDER BY honors `NULLS FIRST/LAST`. This is
  the 3VL of record.
- **Host helpers — minimum-to-stay-total, NOT a deliverable.** Adding `SqlValue::Null`
  forces arms on the legacy host helpers. `compare_sql_values` gets total `Null` arms
  purely so the internal value-index/dedup order never panics (Null = Null → Equal,
  Null < non-null). `select_filter_matches` short-circuits a NULL operand to `false`
  only so the legacy host filter is not *wrong* about NULL while it still exists; it is
  not an investment in the host path and carries no host-3VL test burden of its own.
  These helpers are slated for retirement with the legacy server.

---

## 4. Storage: a per-column, GPU-resident null bitmap

Mirror the existing **bool 1-bit-per-row bitmap** exactly — it is the proven,
self-describing pattern in the device payload.

- **Convention: 1 = valid (present), 0 = NULL** (Arrow / PostgreSQL tuple
  convention). A column with no nulls needs no bitmap at all (absence ⇒ all-valid),
  so non-nullable columns and existing data pay zero cost and stay byte-identical.
- **Device payload (`build_relational_device_payload`,
  `crates/engine/src/engine_residency.rs:19`).** Add a null-bitmap section
  (`ceil(row_count/32)` × `u32` per *nullable* column) after the bool section,
  before text. Each gets a self-describing layout
  `ResidentDeviceNullBitmapLayout { name: String, bitmap_byte_offset: u64 }` added to
  the residency descriptor (`RelationalResidencySnapshot`,
  `crates/engine/src/relational_model.rs:197`), with a
  `resident_device_null_column_offset()` lookup helper alongside the bool/text ones.
- **Kernel access.** Identical to the bool bitmap: load the `u32` word at
  `bitmap_byte_offset + (row/32)*4`, test bit `row%32`. A kernel that consults a
  column's null bit takes one extra `.u64` offset param (0 / sentinel ⇒ "column has
  no nulls, treat all valid"). New atomic-bearing kernels are **not** required for
  the bitmap itself — reads are plain loads — so the 700/716/717 hazard class is not
  reopened by the representation; only later operator kernels that *combine* null
  masks need the usual hazard re-runs.
- **Host storage.** The MVCC store already holds `Vec<Vec<SqlValue>>`; a null cell is
  `SqlValue::Null`. `decode_relational_row` must round-trip it, and
  `build_relational_device_payload` derives each nullable column's bitmap from its
  cells (bit = `!matches!(cell, SqlValue::Null)`), writing a *don't-care* value (0)
  into the typed section for null cells.

---

## 5. Slice ladder

Each slice is independently adversarial-audited before commit (non-negotiable), and
each gate is ptxas sm_90 + build + host (`-p gpu_db_engine -p gpu_db_sql`) + the GPU
`--ignored` suites + clippy `-D warnings`.

1. **Slice 1 — value-model foundation (this milestone's first commit).**
   `SqlValue::Null` + `DbValue::Null` + `map_value` + `db_value_text_opt` and the
   engine→wire `None`/`-1` path; `compare_sql_values` / `select_filter_matches` NULL
   arms (§3 host rules); every exhaustive `match` on `SqlValue` given a correct
   `Null` arm so the workspace compiles and stays **behavior-preserving** (no parse
   path or storage path yet *produces* a `Null`, so no existing query can regress).
   Tests prove non-vacuously that a constructed `DbValue::Null` row emits a `-1`
   DataRow field and that a NULL operand excludes a row in `select_filter_matches`.
2. **Slice 2 — nullable storage + the GPU null bitmap.** §4: host round-trip of
   `SqlValue::Null`; the device-payload bitmap section + descriptor + offset helper;
   residency admits null-bearing rows. Hazard re-runs if any atomic kernel changes.
3. **Slice 3 — ingest + `IS NULL`.** `NULL` literal parsing
   (`parse_inferred_unquoted_literal`), `INSERT … VALUES (…, NULL, …)`, COPY `\N` /
   empty-unquoted-CSV → `Null` (replacing `CopyParseError::NullNotSupported`),
   `DEFAULT NULL`, omitted-column → NULL for nullable columns, and the `IS NULL` /
   `IS NOT NULL` operators (host first, then GPU mask).
4. **Slice 4+ — GPU 3VL.** Kernels honor the bitmap in comparisons, then the
   **NULL-key join gate** (every key path — `key_i64`/`key_texts`/`key_b128`/N:N —
   must exclude NULL keys; `gpu-joins-m5` item 2), then **outer joins** (NULL-pad
   unmatched rows; the single biggest NULL-blocked item), then NULL in aggregates
   (`COUNT(*)` vs `COUNT(col)`, `AVG`-over-empty → NULL), then `NULLS FIRST/LAST` in
   ORDER BY.

Parallel, NULL-independent (the integration plumbing): grow `crates/server` —
engine-result→wire encoding for the **non-NULL** golden subset, transaction and
error-code mapping. This de-risks the store-flip and surfaces real protocol/driver
bugs early; it does not wait on any slice above.

---

## 6. Open policy decisions (deferred, not blocking slice 1)

- **NULL bound parameters.** The legacy server *rejects* them (`0A000`); real PG
  accepts them. Parity-with-legacy (reject) vs PG-correct (accept) is a genuine
  policy fork, surfaced at the extended-protocol slice, not now. Recommendation:
  accept (PG-correct) once storage can hold NULL, since the reject was only ever a
  representation limitation.
- **NULLS FIRST/LAST default in ORDER BY** (PG: NULLS LAST for ASC, FIRST for DESC)
  — a slice-4 concern; the internal derived `Ord` (§2) is independent of it.
- **Catalog/function NULLs** (doc 20) — `format_type`/`obj_description` etc. return
  NULL; lands with the function engine, which by then has `SqlValue::Null` to return.

---

## 7. Charter check

The representation keeps the relational data path on the GPU: the null bitmap is a
device-resident column companion read by the kernels, and 3VL is evaluated on-device
(§3, §4). No NEW CPU relational execution is introduced; the host only marshals the
bit the way it already marshals keys and offsets, and the legacy host helpers get only
the minimum to stay total (they are being retired by unification, not extended — the
investment is entirely GPU-side). The value model gains exactly one variant, and the
executor remains a general interpreter — no per-shape NULL code.
