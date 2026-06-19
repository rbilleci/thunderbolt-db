# Type matrix for the general GPU executor

Status: **in progress (2026-06-18).** The general GPU executor (docs 17, 18) runs int4
predicates / projections end to end. This doc is the plan to broaden it to the other
column types — the charter's "grow by type" axis (doc 17 §6: the type matrix is
*additive and orthogonal*, not multiplicative like query shapes).

Read `17-general-gpu-executor.md` first.

## The shape of the work (per type)

The executor is int4 end to end; each new type extends the same four layers:

1. **Residency** — the GPU snapshot must retain the column on-device. Today: int4 (fixed
   i32, row-major) and text (offsets + bytes). The device payload layout is `header (u64
   row_count) + int4 section + int8 section + … + text section`; a per-type offset
   resolver (`resident_device_<ty>_column_offset` in `relational_model.rs`) locates a
   column. **This is the gating prerequisite — a type the snapshot does not retain cannot
   be read on the GPU regardless of VM support.**
2. **Device VM** (`execution/src/expr_proto.ptx` + `execution/src/lib.rs`) — per-type
   load / arithmetic / compare kernels. The mask AND/OR and mask→indices compaction
   kernels are **type-agnostic** (they operate on 0/1 i32 masks) and are reused as-is. The
   VM is type-aware: each value buffer has an element type and kernels are selected by it.
3. **IR + mapper** (`engine_expr.rs`, `engine_sql_pg.rs`) — typed literals (`Int8Literal`,
   …) and type-directed literal coercion (PG coerces `int8col > 100`'s `100` to int8).
   Columns carry their catalog type.
4. **Executor** (`engine_expr.rs`) — per-type predicate lowering + projection
   materialization (`project_<ty>_rows_from_payload`).

## Decisions

- **Type-aware VM, not parallel clones.** One VM; the element type is a parameter on the
  load / arith / compare steps; the mask / compact stages are shared.
- **Per-type overflow semantics, PG-faithful.** int4 checks the int32 bounds (shipped, doc
  18 slice 2); int8 checks int64; numeric checks its precision. Always errors on overflow,
  never wraps, never CPU-fallback.
- **Reject mixed-type expressions for now.** PG numeric-tower promotion (`int4 + int8 ->
  int8`) is a follow-on; until then a mixed-type `Expr` is a hard error (never a silent
  wrong answer), and such a query simply stays on the existing hand-rolled path.

## Sequence

1. **int8** — **DONE** (the proof that the type matrix is additive): residency retention + offset
   resolver; comparison + projection (i64 resident-column compare peephole + projection); and checked
   arithmetic (the i64 buffer VM, int64-bounds overflow — add/sub sign-XOR, mul via `mul.hi.s64` vs
   the sign-extension of `mul.lo`). The arith VM is now **type-parameterized** (`ResidentElemType`,
   the per-type model this doc set out): one VM, the element type selects kernels + buffer sizes; the
   mask/compact stages are shared. int4 literals coerce to i64 (PG int4->int8); mixed int4/int8
   expressions are a hard error (promotion is a follow-on). int8 `AND`/`OR` and a fused i64
   compact-buffer fast-path are the remaining int8 follow-ons.
2. **numeric** (engine M1 binary i128, fixed 16-byte mantissa + per-column scale) — residency +
   **comparison + projection DONE end to end from SQL**: 16-byte mantissa section (after int8, offset
   resolver); signed i128 limb-compare kernels (`s64` high limb, tie-break on the `u64` low limb);
   `ResidentExpr::NumericLiteral(Decimal128)` + the `Fval` mapper; the executor rescales the literal to
   the column scale (UP only — a literal with more fractional digits than the column is rejected, since
   rounding would mis-answer; PG compares exactly) and runs the i128 compare filters; type-aware i128
   projection. The decimal SCALE is a per-column catalog constant (values rescaled on insert), so the
   snapshot stores only the mantissa. **Checked ARITHMETIC also DONE end to end from SQL** -- add/sub
   (`price + cost > 100`, `price - 5 > 100`) AND full MULTIPLY incl. **column*column and fractional**
   (`price * 2`, `price * 1.5`, `price * tax`): an i128 buffer VM (`ResidentElemType::I128`) with 2-limb
   add/sub (manual carry/borrow) + a signed 128x128->256-bit multiply (4-limb unsigned schoolbook + sign
   correction) in both scalar (`gpu_db_buffer_i128_mul_scalar`) and column*column
   (`gpu_db_buffer_i128_mul`) kernels; overflow via the high-limb sign rule (add/sub) / "high 128 !=
   sign-extension of low 128" (mul) -> PG `numeric field overflow`. `compile_numeric_arith` computes
   each subexpression's RESULT SCALE bottom-up: `+`/`-` keep the (equal) operand scale, `*` ADDS the
   operand scales (literal `*` adds its canonical scale); the terminal compare rescales the literal to
   that scale, and arith-vs-arith requires equal scales. The i32 `ExprStep` scalar bounds in-arith
   literals. **CROSS-SCALE is CLOSED** -- comparison AND add/sub (`p2 > p4`, `p2 + p4 > 100`,
   `price > 1.555`): when scales differ, load to i128 buffers and rescale the coarser side UP to the
   common = max scale (mantissa * 10^k via the mul kernel, k <= 9; engine-only, no new kernel) before
   the op; same-scale stays on the resident peephole. `numeric_arith_scale` predicts each side's scale
   statically (so cross-scale add/sub can rescale operands in the right stack order). **Numeric
   `AND`/`OR` also DONE** (`price > 1 AND price < 100`, incl. cross-scale + nested): each comparison
   compiles to a MASK via the shared `compile_numeric_compare`, `MaskBinary` combines, the i128 VM
   compacts -- mirroring the int path's `compile_predicate_program`. Mixed numeric/integer predicates,
   large in-arith literals, and >9-digit / overflowing rescales are hard errors. **SEVEN audits all
   SHIP** (compare found the alignment P0; add/sub 19k-fuzz; scalar-mul 310k-fuzz; col*col-mul 45k-fuzz;
   cross-scale-compare 4.6k-fuzz; cross-scale add/sub 60-iter fuzz + 4 fault-injections). NUMERIC is now
   essentially complete on the general executor. NEXT: the **text** type (compare + `LIKE`).

   NOTE (gotcha): PTX comments must be PURE ASCII — the runtime JIT's ptxas rejects a non-ASCII byte
   ("Unexpected non-ASCII character", INVALID_PTX 218) that the LOCAL ptxas tolerates. A guard test
   (`expr_proto_ptx_is_pure_ascii`) enforces it.
3. **text** (offsets + byte blob, already retained in residency) — **EQUALITY (Slice A) + LIKE
   (Slice B) DONE** on the general executor. Equality: byte-wise `=` / `<>` (`name = 'alice'`) via
   `gpu_db_resident_text_eq_scalar_to_mask` (one thread/row reads the offsets as 2x4-byte loads,
   byte-compares its slice to a H2D'd needle -> mask -> shared compactor); PG-exact on the default
   deterministic collations. LIKE: `name LIKE 'al%'` via `gpu_db_resident_text_like_scalar_to_mask`,
   a per-row iterative backtracking `%`/`_` matcher where `_`/`%` advance the text by a full UTF-8
   character (PG-correct on multi-byte); the host compiles the pattern (resolving `\` escapes) to a
   u32 token array (`(op<<8)|byte`) H2D'd to the kernel. Engine `try_lower_text_predicate` +
   `compile_like_pattern`; SQL->Expr `Sval`->`TextLiteral`, `AEXPR_LIKE`/`~~`->`Like`. Inequalities
   (need **collation sort keys**: precompute on CPU, byte-compare keys on GPU), `NOT LIKE` (`!~~`),
   text AND/OR, text col-vs-col, and mixed text/non-text are hard errors. NEXT for text = sort-key
   inequalities (or defer); the type is otherwise usable.
4. **date** — **COMPARISON DONE**: a `date` is i32 DAYS since 2000-01-01 (the PG epoch), so it
   REUSES the int4 residency section + the I32 compare VM (no new kernel). `SqlType::Date` +
   `SqlValue::Date(i32)`; a hand-rolled calendar module (`crates/sql/src/datetime.rs`, Howard
   Hinnant `days_from_civil`/`civil_from_days`, no `chrono` dep) parses/formats ISO `YYYY-MM-DD`.
   The residency builder + `resident_device_int4_column_offset` accept `Int4 | Date` (date columns
   ride the i32 section in catalog order); `try_lower_date_predicate` coerces the string literal to
   a day count (`parse_date`) at lowering and emits an I32 `CompareScalar`/`CompareBuffers`;
   projection tags the i32 as `Date`. `hire_date = '2024-01-15'`, `> / < / <>`, literal-on-left, and
   col-vs-col all run on the GPU. Date AND/OR / arithmetic, and date-vs-non-date, are hard errors.
5. **timestamp** — **COMPARISON DONE**: a `timestamp` is i64 MICROSECONDS since 2000-01-01 00:00:00,
   so it REUSES the int8 residency section + the i64 compare KERNELS (`expr_i64_compare_*` -- the i64
   micro literal exceeds the i32 `ExprStep` VM scalar, so it calls the kernels directly, not the VM).
   `parse_timestamp`/`format_timestamp` extend the calendar module (`HH:MM[:SS[.ffffff]]`, `T` or space
   separator, sub-second truncated to 6 digits). The int8 residency builder + `resident_device_int8_
   column_offset` accept `Int8 | Timestamp`; `try_lower_timestamp_predicate` coerces the string literal
   to micros at lowering; projection tags the i64 as Timestamp. `event_at = '2024-01-15 10:00:00'`,
   `>/<`, literal-on-left, and col-vs-col all run on the GPU. NEXT temporal = timestamptz / time /
   interval (follow-ons). **TEMPORAL CLUSTER (date + timestamp) DONE.**
6. **uuid** — **COMPARISON DONE**: a `uuid` is 16 raw bytes; PG compares two uuids by an unsigned
   big-endian `memcmp`. It REUSES the i128 (16-byte) residency section (`resident_device_numeric_
   column_offset` + the residency builder accept `Numeric | Uuid`; uuid stores its raw bytes where
   numeric stores its mantissa) but needs a NEW compare kernel: `gpu_db_resident_uuid_compare_scalar_
   to_mask` / `_columns_to_mask` do a per-row 16-byte byte-wise memcmp (read with `ld.global.u8`, so
   ALIGNMENT-SAFE regardless of the 16-byte section's offset) -> ordering -> the 6-way cmp result ->
   mask -> shared compactor; `scalar_on_left` negates the ordering. `crates/sql/src/uuid.rs` parses
   the canonical/bare/braced hex to `[u8; 16]` and formats canonical lowercase. `SqlType::Uuid`
   (oid 2950) + `SqlValue::Uuid([u8; 16])`; `try_lower_uuid_predicate` parses the string literal to
   16 bytes at lowering (dispatched before text); projection reuses the i128 projector (the raw bytes
   are the i128's LE form, recovered via `to_le_bytes`). `id = '...'`, all six comparators, literal-
   on-left, `<>`, and col-vs-col run on the GPU. Uuid `AND`/`OR`, and uuid-vs-non-uuid, hard-error.
7. **int2** (smallint) — **COMPARISON DONE**: a `smallint` is i16, stored WIDENED to i32 in the int4
   section, so it REUSES the int4 residency layout + the i32 compare VM (no new kernel). `SqlType::
   Int2` (oid 21) + `SqlValue::Int2(i16)`. The residency builder + `resident_device_int4_column_
   offset` accept `Int4 | Date | Int2`; `try_lower_int2_predicate` takes the `Int4Literal` as the i32
   scalar (an out-of-int16 literal is a VALID comparison that matches no rows -- PG widens both to
   int4 -- NOT a range error); projection narrows i32 back to i16. INSERT range-checks int4->int2
   ("smallint out of range"). `sz = 0`, `>/<` incl. negatives, literal-on-left, and col-vs-col run on
   the GPU. Smallint arithmetic (int16-bounds overflow) stays a hard error -- the int4 ARITH compiler
   does not pick up int2 -- a follow-on.
8. **bool** — **PREDICATE (`WHERE flag`) DONE, BIT-PACKED**: a bool column is retained as a 1-BIT-PER-ROW
   bitmap (`ceil(N/32)` LE u32 words) — 32x denser than the i32 sections, and itself a near-ready
   predicate mask. `gpu_db_resident_bool_to_mask` expands bit i -> the i32 row mask (with a `negate`
   flag for the future `NOT flag` / `= false`), then the shared compactor; the word load is 4-byte
   aligned so no fault. New self-describing residency section (`ResidentDeviceBoolColumnLayout`, stores
   the bitmap byte offset like text) + `resident_device_bool_column_offset`. `lower_resident_predicate`
   now accepts a BARE top-level `Column` (a bare non-bool column is invalid SQL — PG "argument of WHERE
   must be type boolean" — so it hard-errors). NULLs are a separate validity bitmap deferred to M3
   (the engine is non-null everywhere today), so only the value bit is stored; the layout is NULL-ready.
   Gate: engine GPU 25/0, execution GPU 49/0 (shared-PTX no-regression). **FOLLOW-ONS DONE**: `flag =
   true`/`= false`/`<>` (a `ResidentExpr::BoolLiteral` from the `Boolval` AST node lowers to the
   bitmap->mask kernel via `negate`), `NOT flag` (the mapper rewrites `NOT <bool col>` to `flag =
   false`), and `SELECT flag` PROJECTION (a host-side bitmap GATHER -- read each surviving row's word,
   extract its bit -- no kernel, mirroring the i32/i64 gather). Remaining: `COUNT(*) WHERE flag` via
   popcount over the bitmap (an aggregate / operator-axis item); general `NOT` (De Morgan over
   comparisons / AND-OR); bool in AND/OR (the bool path emits indices, not a composable mask).

Then: mixed-type promotion (the PG numeric tower) and the general-first routing flip (doc
17 §3.4, the charter end state).
