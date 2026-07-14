//! Predicate dispatch plus boolean and standalone-NULL fast paths.
//!
//! The dispatcher orders visibility composition, typed peepholes, general VM lowering, and
//! arithmetic comparisons. Type-family lowering bodies remain with the parent executor owner.

use super::execution_source::ResidentVisibility;
use super::predicate_compiler::{
    arith_op_code, collect_expr_columns, compare_op_code, compile_arith_program,
    compile_predicate_program, flip_comparison_code, mixed_width_i32_elem,
    predicate_references_nullable_column, predicate_vm_elem_type,
};
use super::predicate_operands::expr_mentions_uuid;
use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::relational_model::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_null_column_offset, RelationalResidencySnapshot, RelationalTable,
};
use crate::{Engine, ExecuteError};
use gpu_db_execution::{CudaResidentDeviceMemory, ResidentElemType};
use gpu_db_sql::SqlType;
use gpu_db_types::EngineError;

impl Engine {
    /// Lower a bare bool-column predicate (`WHERE flag`) to surviving row indices (the type matrix,
    /// doc 19): the bool column's 1-bit-per-row bitmap expands directly to the row mask. A bare
    /// non-bool column is invalid SQL (PG: "argument of WHERE must be type boolean") -> hard error.
    fn lower_bool_column_predicate(
        &self,
        col: usize,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        let column = table.columns.get(col).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate column is outside the catalog table".to_string(),
            ))
        })?;
        if column.ty != SqlType::Bool {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a bare column predicate is valid only for a bool column (argument of WHERE must be \
                 type boolean)"
                    .to_string(),
            )));
        }
        let offset = resident_device_bool_column_offset(snapshot, table, col)?;
        device_memory
            .expr_bool_to_mask_filter(offset, false, row_count)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Lower a standalone `col IS NULL` / `IS NOT NULL` to surviving row indices (M3 -- doc 21). The
    /// column's NULL validity bitmap (1 = valid/present) feeds the same bitmap->mask kernel as a bool
    /// column: `IS NOT NULL` reads it as-is, `IS NULL` complements it (`negate = !is_not_null`). A column
    /// with NO validity bitmap holds no NULLs, so every row is valid -- IS NOT NULL is all rows, IS NULL
    /// none -- returned directly without a kernel.
    fn lower_is_null_predicate(
        &self,
        col: usize,
        is_not_null: bool,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        match resident_device_null_column_offset(snapshot, table, col)? {
            Some(offset) => device_memory
                .expr_bool_to_mask_filter(offset, !is_not_null, row_count)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))),
            None if is_not_null => {
                let n = u32::try_from(row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident row count exceeds u32".to_string(),
                    ))
                })?;
                Ok((0..n).collect())
            }
            None => Ok(Vec::new()),
        }
    }

    /// Try to lower `bool_col = true` / `bool_col = false` (and `<>`, either operand order) to
    /// surviving row indices via the bitmap->mask kernel (the type matrix, doc 19): `= true` selects
    /// the set bits, `= false` (and `NOT flag`, which the mapper rewrites to `= false`) the clear bits
    /// (negate). Returns None if not a `bool_col <op> bool_literal` shape. Ordering ops (`< > ...`) and
    /// bool in AND/OR are hard errors (follow-ons) so the executor never mis-answers.
    #[allow(clippy::too_many_arguments)]
    fn try_lower_bool_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        let is_bool_col =
            |idx: usize| table.columns.get(idx).map(|column| column.ty) == Some(SqlType::Bool);
        let (col, literal, column_on_left) = match (lhs, rhs) {
            (ResidentExpr::Column(col), ResidentExpr::BoolLiteral(b)) if is_bool_col(*col) => {
                (*col, *b, true)
            }
            (ResidentExpr::BoolLiteral(b), ResidentExpr::Column(col)) if is_bool_col(*col) => {
                (*col, *b, false)
            }
            _ => return Ok(None),
        };
        // ADR-006 (bool inequalities): the same constant-fold as `compile_bool_leaf` — PG orders
        // `false < true`, so `<`/`<=`/`>`/`>=` against a two-valued literal reduces to an
        // equality mask or a constant verdict. This peephole serves NON-NULL columns only (a
        // nullable-bool predicate routes through the VM block above), so constant-TRUE is ALL
        // rows — there is no UNKNOWN to exclude.
        let effective = if column_on_left {
            compare
        } else {
            match compare {
                ResidentBinaryOp::Lt => ResidentBinaryOp::Gt,
                ResidentBinaryOp::Le => ResidentBinaryOp::Ge,
                ResidentBinaryOp::Gt => ResidentBinaryOp::Lt,
                ResidentBinaryOp::Ge => ResidentBinaryOp::Le,
                other => other,
            }
        };
        let needle = match effective {
            ResidentBinaryOp::Eq => literal,
            ResidentBinaryOp::Ne => !literal,
            ResidentBinaryOp::Lt if literal => false,
            ResidentBinaryOp::Le if !literal => false,
            ResidentBinaryOp::Gt if !literal => true,
            ResidentBinaryOp::Ge if literal => true,
            ResidentBinaryOp::Lt | ResidentBinaryOp::Gt => {
                return Ok(Some(Vec::new())); // col < false / col > true: nothing qualifies
            }
            ResidentBinaryOp::Le | ResidentBinaryOp::Ge => {
                // col <= true / col >= false over a NON-NULL column: every row qualifies.
                let n = u32::try_from(row_count).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "bool predicate row count exceeds u32".to_string(),
                    ))
                })?;
                return Ok(Some((0..n).collect()));
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports only comparisons against a bool literal"
                        .to_string(),
                )));
            }
        };
        // `col == true` -> set bits (negate=false); `col == false` -> clear bits (negate=true).
        let offset = resident_device_bool_column_offset(snapshot, table, col)?;
        device_memory
            .expr_bool_to_mask_filter(offset, !needle, row_count)
            .map(Some)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Lower a predicate [`ResidentExpr`] to the surviving row indices, evaluated on the GPU.
    ///
    /// Coverage (grows by extending this method, per the design): the prototype shape
    /// `Compare(arith(Column a, Column b), Int4Literal k)` lowers to the composed buffer->buffer
    /// facade `expr_filter_two_col_compare_from_payload` (typed column loads and `a <arith> b` into
    /// an intermediate device buffer, then ordered compare-to-row-indices). Anything else is rejected with a
    /// pointer to the device bytecode VM that generalizes it — NOT by falling back to a shape method
    /// or to the CPU.
    pub(crate) fn lower_resident_predicate(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
        // SV3b/SV6 (MVCC visibility): `Some` on a VERSIONED shard — AND the on-device visibility mask
        // (`deleted_by > read_txn_id`, and/or `created_by <= read_txn_id`) onto the WHERE survivors.
        // `None` (version-free/cold shard, the majority) keeps the peephole fast paths below byte-identically.
        visibility: Option<ResidentVisibility>,
    ) -> Result<Vec<u32>, ExecuteError> {
        // SV3b: a versioned shard forces the mask VM (bypassing the typed peephole kernels) so the WHERE is a
        // program we can AND the i64 visibility mask onto — via the mixed-width VM (int4 WHERE at elem=I32 +
        // an i64 `deleted_by`/`created_by` compare). Only versioned shards reach here, so the peephole fast
        // paths below stay the common-case default.
        if let Some(vis) = visibility {
            // A predicate the VM can't lower at one elem width (mixed non-int8 widths / an unsupported shape)
            // can't compose the i64 visibility conjunct, so this HARD-ERRORS rather than risk leaking
            // tombstoned rows. NB: this is NOT the CPU fallback (that fires only on residency invalidation);
            // it surfaces as a query error. Inert until DELETE tombstoning is wired (SV4) -- at which point
            // extending visibility to these shapes (or routing them to the CPU-pinned path) is the follow-up.
            // ADR-006 (MIXED-WIDTH groups): a mixed int8/ts + i32-servable WHERE with width-safe
            // scalar i64 leaves composes with the (i64) visibility conjuncts at I32 — the same
            // LoadColumnI64-self-consumed contract the conjuncts themselves use. LOCAL fallback,
            // not a widening of the shared `predicate_vm_elem_type`.
            let elem = predicate_vm_elem_type(predicate, table)
                .or_else(|| mixed_width_i32_elem(predicate, table))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident visibility filter: this WHERE predicate is not VM-lowerable (mixed width / \
                         unsupported shape) so the MVCC visibility conjunct cannot be composed"
                            .to_string(),
                    ))
                })?;
            let mut program: Vec<gpu_db_execution::ExprStep> = Vec::new();
            let mut needles: Vec<Vec<u8>> = Vec::new();
            compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
            // AND the visibility conjunct(s) onto the WHERE mask. Each `CompareScalarI64` immediately
            // consumes its own `LoadColumnI64` buffer (the mixed-width VM caller contract).
            vis.push_conjuncts(&mut program, true);
            return device_memory
                .run_expr_predicate_filter_with_text(&program, &needles, row_count, elem)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }
        // bool-predicate (type matrix, doc 19): a bare `WHERE flag` is a bool COLUMN used directly as
        // a predicate -- a bool column is a 1-bit-per-row bitmap, so it expands straight to the row
        // mask (true rows). A bare NON-bool column is invalid SQL (PG: "argument of WHERE must be type
        // boolean"), so it hard-errors rather than mis-answering.
        if let ResidentExpr::Column(col) = predicate {
            return self.lower_bool_column_predicate(
                *col,
                table,
                snapshot,
                device_memory,
                row_count,
            );
        }

        // A standalone `col IS NULL` / `IS NOT NULL` (M3 -- doc 19/21): its NULL validity bitmap straight
        // to the row mask via the SAME bitmap->mask kernel as a bool column, pointed at the validity
        // bitmap. (Inside AND/OR it instead rides the general mask VM via `compile_predicate_program`.)
        if let ResidentExpr::IsNull { col, is_not_null } = predicate {
            return self.lower_is_null_predicate(
                *col,
                *is_not_null,
                table,
                snapshot,
                device_memory,
                row_count,
            );
        }

        // M3 (doc 21) WHERE 3VL: a predicate over a NULLABLE column routes through the general mask VM
        // (`compile_predicate_program`), which AND's each comparison leaf with the column's validity mask
        // so a NULL operand evaluates to UNKNOWN (the row is excluded). The typed peephole kernels below
        // read the column's placeholder bytes (0 / empty) and would mis-select NULL rows. The VM lowers an
        // int4/text/bool predicate (elem I32) or an int8 predicate (elem I64, the same i64 buffer VM the
        // non-null int8 AND/OR path uses — the validity AND is a type-independent i32 mask); a nullable
        // numeric/uuid/date/timestamp/int2 or MIXED-width predicate is a clean-error follow-up (no silent
        // mis-answer). A no-NULL table never enters here, so existing (non-nullable) plans keep their
        // peephole fast paths byte-identically.
        if predicate_references_nullable_column(predicate, table, snapshot)? {
            if let Some(elem) = predicate_vm_elem_type(predicate, table) {
                let mut program = Vec::new();
                let mut needles: Vec<Vec<u8>> = Vec::new();
                compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
                return device_memory
                    .run_expr_predicate_filter_with_text(&program, &needles, row_count, elem)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    });
            }
            // A nullable DATE/TIMESTAMP/NUMERIC simple comparison: the generic VM can't lower the literal
            // (temporal/numeric), so build the program directly (date I32 / timestamp+numeric I128/I64)
            // with the validity-AND appended.
            if let ResidentExpr::Binary { op, lhs, rhs } = predicate {
                if let Some(indices) = self.try_lower_nullable_temporal_predicate(
                    *op,
                    lhs,
                    rhs,
                    table,
                    snapshot,
                    device_memory,
                    row_count,
                )? {
                    return Ok(indices);
                }
                if let Some(indices) = self.try_lower_nullable_numeric_predicate(
                    *op,
                    lhs,
                    rhs,
                    table,
                    snapshot,
                    device_memory,
                    row_count,
                )? {
                    return Ok(indices);
                }
                // A nullable uuid SIMPLE comparison: the uuid memcmp peephole resolves the operand's
                // validity bitmap and AND's it with the compare mask (uuid has no VM step). Runs BEFORE
                // the uuid AND/OR block below so col-vs-col and every simple shape keep this proven path.
                if expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table) {
                    if let Some(indices) = self.try_lower_uuid_predicate(
                        *op,
                        lhs,
                        rhs,
                        table,
                        snapshot,
                        device_memory,
                        row_count,
                    )? {
                        return Ok(indices);
                    }
                }
                // NULLABLE uuid / timestamp AND/OR (ADR-006): uuid + timestamp are not in
                // `predicate_vm_elem_type`'s sets (deliberately — widening the SHARED helper diverts
                // working paths; the timestamp lesson), so a nullable uuid range/IN or a nullable
                // timestamp range missed the VM block above and used to hit the final error. Gate
                // LOCALLY by column set: {Uuid, Int4, Int2, Text, Bool} runs at I32 (uuid/text/bool
                // leaves are mask-only, int4/int2 load 4 bytes); {Int8, Timestamp} runs at I64 (a
                // timestamp is i64 micros in the int8 section; the `Int8Literal` leaf arm loads 8
                // bytes + CompareScalarI64). ZERO DIVERSION: all-non-uuid I32 combos and all-Int8
                // took the elem-Some block above, so these gates fire only with a uuid / timestamp
                // column present — shapes that previously ERRORED. Each leaf appends its own
                // validity-AND (3VL), exactly how the elem-Some VM block serves nullable int4/text.
                if matches!(op, ResidentBinaryOp::And | ResidentBinaryOp::Or) {
                    let mut cols = Vec::new();
                    collect_expr_columns(predicate, &mut cols);
                    let ty = |col: usize| table.columns.get(col).map(|column| column.ty);
                    let local_elem = if cols.is_empty() {
                        None
                    } else if cols.iter().all(|&col| {
                        // Date joins the I32 set (ADR-006 date compound): a date is i32 days in the
                        // int4 section, and its VM leaf emits a 4-byte LoadColumn — I32-only.
                        matches!(
                            ty(col),
                            Some(
                                SqlType::Uuid
                                    | SqlType::Int4
                                    | SqlType::Int2
                                    | SqlType::Text
                                    | SqlType::Bool
                                    | SqlType::Date
                            )
                        )
                    }) {
                        Some(ResidentElemType::I32)
                    } else if cols
                        .iter()
                        .all(|&col| matches!(ty(col), Some(SqlType::Int8 | SqlType::Timestamp)))
                    {
                        Some(ResidentElemType::I64)
                    } else {
                        // ADR-006 (MIXED-WIDTH groups): a nullable mixed int8/ts + i32-servable
                        // group runs at I32 when every i64 leaf is a width-safe scalar comparison.
                        // ZERO DIVERSION: the two branches above already served every all-I32-set
                        // and all-i64-section combination — this fires only for mixed sets, which
                        // previously fell to the final error. Each leaf appends its own
                        // validity-AND (3VL), including the int8 scalar arm.
                        mixed_width_i32_elem(predicate, table)
                    };
                    if let Some(elem) = local_elem {
                        let mut program = Vec::new();
                        let mut needles: Vec<Vec<u8>> = Vec::new();
                        compile_predicate_program(
                            predicate,
                            table,
                            snapshot,
                            &mut program,
                            &mut needles,
                        )?;
                        return device_memory
                            .run_expr_predicate_filter_with_text(
                                &program, &needles, row_count, elem,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            });
                    }
                }
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "WHERE over a nullable column of this type/shape is not yet supported on the GPU \
                 (M3 3VL follow-up: e.g. a cross-scale or arithmetic numeric, a compound or \
                 mixed-type temporal/numeric/uuid predicate)"
                    .to_string(),
            )));
        }

        let ResidentExpr::Binary {
            op: compare,
            lhs,
            rhs,
        } = predicate
        else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate must be a top-level comparison or AND/OR".to_string(),
            )));
        };

        // bool path (type matrix, doc 19): `flag = true` / `flag = false` / `<>` (and `NOT flag`,
        // which the mapper rewrites to `flag = false`) expand the bool bitmap to the mask via negate.
        // None for a non-bool-literal predicate; checked first since a BoolLiteral has no other home.
        if let Some(indices) = self.try_lower_bool_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // int8 path (type matrix, doc 19): a SIMPLE int8 comparison (`int8col <cmp> literal` /
        // `literal <cmp> int8col` / `int8col <cmp> int8col`) evaluates via the i64 compare kernels,
        // before the int4 paths below. `try_lower_int8_comparison` returns None for a pure-int4
        // predicate (fall through to the int4 path) and errors for an int8 shape not yet supported
        // (int8 arithmetic / mixed int4-int8) so the executor never silently mis-answers.
        if let Some(indices) = self.try_lower_int8_predicate(
            predicate,
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // numeric path (type matrix, doc 19): a SIMPLE numeric comparison (`numcol <cmp> literal` /
        // `literal <cmp> numcol` / equal-scale `numcol <cmp> numcol`) evaluates via the i128 compare
        // kernels. Returns None for a non-numeric predicate (fall through to int4); errors on a numeric
        // shape not yet supported (numeric arithmetic / AND-OR / mixed) so it never mis-answers.
        if let Some(indices) = self.try_lower_numeric_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // date path (type matrix, doc 19): a SIMPLE date comparison (`hire_date <cmp> '2024-01-15'`)
        // evaluates via the i32 compare path (a date is i32 days). BEFORE the text path, because the
        // date literal is a string literal the text path would otherwise grab. None for a non-date
        // predicate; errors on date AND/OR / arithmetic / mixed.
        if let Some(indices) = self.try_lower_date_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // timestamp path (type matrix, doc 19): a SIMPLE timestamp comparison (i64 microseconds ->
        // the int8 compare kernels). BEFORE the text path (the literal is a string). None for a
        // non-timestamp predicate; errors on timestamp AND/OR / arithmetic / mixed.
        if let Some(indices) = self.try_lower_timestamp_predicate(
            predicate,
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // uuid path (type matrix, doc 19): a SIMPLE uuid comparison via the byte-wise compare kernel.
        // BEFORE the text path (the literal is a string). None for a non-uuid predicate; errors on
        // uuid AND/OR / mixed.
        if let Some(indices) = self.try_lower_uuid_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // int2 path (type matrix, doc 19): a SIMPLE smallint comparison via the i32 compare VM
        // (a smallint is stored widened to i32). None for a non-smallint predicate; errors on
        // smallint AND/OR / arithmetic / a non-integer comparand.
        if let Some(indices) = self.try_lower_int2_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // text path (type matrix, doc 19): a SIMPLE text comparison (`textcol = 'lit'` / `<>`)
        // evaluates via the byte-wise text-equality kernel. Returns None for a non-text predicate
        // (fall through to int4); errors on a text shape not yet supported (LIKE / inequalities /
        // AND-OR / mixed) so it never silently mis-answers.
        if let Some(indices) = self.try_lower_text_predicate(
            *compare,
            lhs,
            rhs,
            table,
            snapshot,
            device_memory,
            row_count,
        )? {
            return Ok(indices);
        }

        // Boolean combinators (AND/OR) and `Ne` lower to the general mask-based predicate VM (each
        // comparison -> a mask, MaskBinary combines, the terminal compacts). The specialized 2-col /
        // arith / col-vs-col lowerings below handle the single eq/lt/le/gt/ge comparisons.
        if matches!(
            compare,
            ResidentBinaryOp::And | ResidentBinaryOp::Or | ResidentBinaryOp::Ne
        ) {
            let mut program = Vec::new();
            let mut needles: Vec<Vec<u8>> = Vec::new();
            compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
            return device_memory
                .run_expr_predicate_filter_with_text(
                    &program,
                    &needles,
                    row_count,
                    ResidentElemType::I32,
                )
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }

        let Some(comparison) = compare_op_code(*compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr predicate requires a comparison op (eq/lt/le/gt/ge/ne) or AND/OR"
                    .to_string(),
            )));
        };

        // Specialized facade for the exact `Compare(arith(Column, Column), Int4Literal)` shape. The
        // execution crate constructs the same typed postfix VM program used by general expressions,
        // then emits ascending indices through ordered compaction; this keeps one stable call surface
        // without a raw-offset shape kernel.
        if let (
            ResidentExpr::Binary {
                op: arith,
                lhs: a,
                rhs: b,
            },
            ResidentExpr::Int4Literal(needle),
        ) = (lhs.as_ref(), rhs.as_ref())
        {
            if let (Some(op_code), ResidentExpr::Column(col_a), ResidentExpr::Column(col_b)) =
                (arith_op_code(*arith), a.as_ref(), b.as_ref())
            {
                let a_offset = resident_device_int4_column_offset(snapshot, table, *col_a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, *col_b)?;
                return device_memory
                    .expr_filter_two_col_compare_from_payload(
                        a_offset, b_offset, op_code, row_count, *needle, comparison,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    });
            }
        }

        // PEEPHOLE fast-path (the `category = 0` shape): the exact predicate `int4col <cmp> literal`
        // (and the flipped `literal <cmp> int4col`, comparison flipped) lowers to the ORDERED parallel
        // compaction that emits surviving ROW INDICES ascending with NO host sort — replacing the
        // general `run_expr_arith_filter` path, which now uses the same ordered compactor. The
        // ordered compaction guarantees ascending output BY CONSTRUCTION (the contiguous block
        // partition + ordered intra-block prefix sum), which is exactly the contract the gather +
        // assembly depend on, so the result is byte-identical to the prior path. Only a PLAIN resident
        // int4 column qualifies: nullable columns / date / int2 / int8 / other types are routed by the
        // earlier type paths and never reach here, but the `SqlType::Int4` guard makes that explicit so
        // a future reordering can't silently feed a date/int2 column (same i32 section) into this path.
        let plain_int4_column = |expr: &ResidentExpr| -> Option<usize> {
            if let ResidentExpr::Column(col) = expr {
                if table.columns.get(*col).map(|column| column.ty) == Some(SqlType::Int4) {
                    return Some(*col);
                }
            }
            None
        };
        let ordered_indices = match (lhs.as_ref(), rhs.as_ref()) {
            (col_expr, ResidentExpr::Int4Literal(needle)) => {
                plain_int4_column(col_expr).map(|col| (col, *needle, comparison))
            }
            (ResidentExpr::Int4Literal(needle), col_expr) => plain_int4_column(col_expr)
                .map(|col| (col, *needle, flip_comparison_code(comparison))),
            _ => None,
        };
        if let Some((col, needle, cmp_code)) = ordered_indices {
            let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
            return device_memory
                .compare_indices_ordered_from_payload(byte_offset, row_count, needle, cmp_code)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())));
        }

        // GENERAL path: compile the arithmetic side(s) to bytecode and run the device VM.
        //   - `arith_tree <cmp> literal`        -> arith VM (one value buffer vs scalar)
        //   - `literal <cmp> arith_tree`        -> arith VM, comparison flipped
        //   - `arith_tree <cmp> arith_tree`     -> compile both, col-vs-col / expr-vs-expr VM
        // (AND/OR over masks is the next slice.)
        match (lhs.as_ref(), rhs.as_ref()) {
            (value, ResidentExpr::Int4Literal(needle)) => {
                let mut program = Vec::new();
                compile_arith_program(value, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_arith_filter(&program, row_count, comparison, *needle)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
            (ResidentExpr::Int4Literal(needle), value) => {
                let mut program = Vec::new();
                compile_arith_program(value, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_arith_filter(
                        &program,
                        row_count,
                        flip_comparison_code(comparison),
                        *needle,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
            (lhs_expr, rhs_expr) => {
                let mut program = Vec::new();
                compile_arith_program(lhs_expr, table, snapshot, &mut program)?;
                compile_arith_program(rhs_expr, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_compare_buffers_filter(&program, row_count, comparison)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
            }
        }
    }
}
