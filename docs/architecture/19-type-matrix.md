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
   statically (so cross-scale add/sub can rescale operands in the right stack order). Mixed
   numeric/integer, numeric `AND`/`OR`, large in-arith literals, and >9-digit / overflowing rescales are
   hard errors. **SIX audits all SHIP** (compare found the alignment P0; add/sub 19k-fuzz; scalar-mul
   310k-fuzz; col*col-mul 45k-fuzz; cross-scale-compare 4.6k-fuzz + max->min fault-injection). NEXT:
   numeric `AND`/`OR` (route numeric boolean combinators through the i128 mask VM).

   NOTE (gotcha): PTX comments must be PURE ASCII — the runtime JIT's ptxas rejects a non-ASCII byte
   ("Unexpected non-ASCII character", INVALID_PTX 218) that the LOCAL ptxas tolerates. A guard test
   (`expr_proto_ptx_is_pure_ascii`) enforces it.
3. **text** (already retained in residency) — equality / inequality + `LIKE`-prefix over
   offsets + bytes; folds the enumerated `text_prefix_like` route toward a general-path
   peephole.
4. **bool** (1-byte).

Then: mixed-type promotion (the PG numeric tower) and the general-first routing flip (doc
17 §3.4, the charter end state).
