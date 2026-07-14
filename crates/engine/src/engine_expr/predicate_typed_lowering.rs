//! Complete typed predicate lowering for resident GPU columns.
//!
//! Each lowerer owns one SQL type family and returns `Ok(None)` only when that family does not
//! participate. The sibling dispatcher preserves ordering among these peepholes and the general VM.

use super::predicate_compiler::{
    boolean_op_code, collect_expr_columns, compile_numeric_compare,
    compile_numeric_predicate_program, compile_predicate_program, compile_text_eq_leaf,
    mixed_width_i32_elem, predicate_compare_code, push_column_validity_and, push_numeric_rescale,
    rescale_numeric_literal,
};
use super::predicate_operands::{
    column_numeric_scale, compile_like_pattern, date_column_index, date_literal_days,
    expr_mentions_bool_column, expr_mentions_date, expr_mentions_int2, expr_mentions_int4_column,
    expr_mentions_int8, expr_mentions_numeric, expr_mentions_text, expr_mentions_timestamp,
    expr_mentions_uuid, int2_column_index, int2_literal_value, int8_column_index,
    numeric_column_index, numeric_literal_value, text_column_index, text_literal_value,
    timestamp_column_index, timestamp_literal_micros, uuid_column_index, uuid_literal_bytes,
};
use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::relational_model::{
    resident_device_int4_column_offset, resident_device_int8_column_offset,
    resident_device_null_column_offset, resident_device_numeric_column_offset,
    resident_device_text_column_layout, RelationalResidencySnapshot, RelationalTable,
};
use crate::{Engine, ExecuteError};
use gpu_db_execution::{CudaResidentDeviceMemory, ExprStep, ResidentElemType};
use gpu_db_sql::{Decimal128, SqlType};
use gpu_db_types::EngineError;

impl Engine {
    /// Try to lower a SIMPLE int8 comparison to surviving row indices via the i64 compare kernels
    /// (the type matrix, doc 19). Supported shapes: `int8col <cmp> int4literal` (the int4 literal is
    /// widened to int8 — PG's int4->int8 coercion), `int4literal <cmp> int8col` (operand order
    /// preserved via `scalar_on_left`), and `int8col <cmp> int8col`.
    ///
    /// Returns `Ok(None)` for a pure-int4 predicate (the caller falls through to the int4 path) and
    /// `Err` when int8 appears in a shape not yet supported (int8 arithmetic, or a mixed int4/int8
    /// expression) — never a silent fall-through that would mis-answer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_int8_predicate(
        &self,
        predicate: &ResidentExpr,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        let lhs_int8 = int8_column_index(lhs, table);
        let rhs_int8 = int8_column_index(rhs, table);
        // Surface the device error verbatim (e.g. int64 overflow -> "bigint out of range"), matching
        // the int4 path's mapping.
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };

        // The simple int8 comparison PEEPHOLE (resident-column compare kernels) — only for an actual
        // comparison op. `predicate_compare_code` is None for AND/OR, so boolean combinators skip this
        // and fall through to the general int8 path below (NOT the int4 path — that was the silent
        // mis-answer the audit caught: int8 AND/OR read with the i32 4-byte stride).
        if let Some(cmp) = predicate_compare_code(compare) {
            match (lhs, rhs) {
                (ResidentExpr::Int4Literal(scalar), _) if rhs_int8.is_some() => {
                    let offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        rhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_scalar_filter(
                            offset,
                            i64::from(*scalar),
                            true,
                            cmp,
                            row_count,
                        )
                        .map(Some)
                        .map_err(map_err);
                }
                (_, ResidentExpr::Int4Literal(scalar)) if lhs_int8.is_some() => {
                    let offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        lhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_scalar_filter(
                            offset,
                            i64::from(*scalar),
                            false,
                            cmp,
                            row_count,
                        )
                        .map(Some)
                        .map_err(map_err);
                }
                _ if lhs_int8.is_some() && rhs_int8.is_some() => {
                    let a_offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        lhs_int8.expect("checked"),
                    )?;
                    let b_offset = resident_device_int8_column_offset(
                        snapshot,
                        table,
                        rhs_int8.expect("checked"),
                    )?;
                    return device_memory
                        .expr_i64_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                        .map(Some)
                        .map_err(map_err);
                }
                _ => {}
            }
        }

        // General int8 path: int8 ARITHMETIC / AND-OR / deeper predicates lower via the i64 buffer VM
        // (int4 LITERALS are widened to i64). A pure-int4 predicate returns None (the int4 paths handle
        // it); a MIXED int4/int8 expression is a hard error — never a silent mis-answer.
        let mentions_int8 = expr_mentions_int8(lhs, table) || expr_mentions_int8(rhs, table);
        if !mentions_int8 {
            return Ok(None);
        }
        // ADR-006 (MIXED-WIDTH groups): a mixed int8+{int4,int2,text,bool,date,uuid} predicate
        // now runs on the I32 mask VM when every int8 leaf is a WIDTH-SAFE scalar comparison (the
        // `LoadColumnI64`+`CompareScalarI64` arm — see `mixed_width_i32_elem`): fall through
        // (`Ok(None)`) to the general And/Or I32 branch. Otherwise: a mix bearing a 4-BYTE value
        // leaf (int4/int2/date — `LoadColumn` follows the program width and would 8-byte mis-read
        // in this I64 program; int2 was the audit's MEDIUM: it used to slip past this block into
        // the I64 compile) or a text/bool mask leaf keeps the HARD ERROR — never a silent
        // mis-answer at one element width. A UUID-ONLY mix with non-scalar int8 (col-vs-col /
        // arith) deliberately falls THROUGH to the I64 compile below — the uuid leaf is a
        // width-agnostic mask (`UuidCmpMask` reads b128 bytes, not elem-strided) and that path
        // pre-dates this slice (audit LOW: rejecting it here regressed a working shape).
        // int8-only predicates stay on this i64 VM below.
        let mentions_width_bound = expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int2(lhs, table)
            || expr_mentions_int2(rhs, table)
            || expr_mentions_text(lhs, table)
            || expr_mentions_text(rhs, table)
            || expr_mentions_bool_column(lhs, table)
            || expr_mentions_bool_column(rhs, table)
            || expr_mentions_date(lhs, table)
            || expr_mentions_date(rhs, table);
        if mentions_width_bound || expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)
        {
            if mixed_width_i32_elem(predicate, table).is_some() {
                return Ok(None);
            }
            if mentions_width_bound {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports a mixed int8 predicate only when every \
                     int8 leaf is a bare-column scalar comparison"
                        .to_string(),
                )));
            }
            // uuid-only mix with non-scalar int8: the pre-slice I64 path serves it correctly.
        }
        let mut program = Vec::new();
        let mut needles: Vec<Vec<u8>> = Vec::new();
        compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
        device_memory
            .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
            .map(Some)
            .map_err(map_err)
    }

    /// Cross-scale `numcol <cmp> literal` (or flipped): the literal is FINER than the column, so load
    /// the column, rescale it UP to the literal's scale (mantissa * 10^k, the buffer VM path), and
    /// compare to the literal's mantissa. `scalar_on_left` is the literal's side of the comparison.
    #[allow(clippy::too_many_arguments)]
    fn numeric_cross_scale_scalar(
        &self,
        column_offset: u64,
        column_scale: u8,
        literal: Decimal128,
        scalar_on_left: bool,
        cmp: u32,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        let needle = i32::try_from(literal.mantissa).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "numeric comparison literal is too large for the fast path yet".to_string(),
            ))
        })?;
        let mut program = vec![ExprStep::LoadColumn {
            byte_offset: column_offset,
        }];
        push_numeric_rescale(column_scale, literal.scale, &mut program)?;
        program.push(ExprStep::CompareScalar {
            cmp,
            scalar: needle,
            scalar_on_left,
        });
        device_memory
            .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Try to lower a SIMPLE date comparison (`datecol <cmp> 'YYYY-MM-DD'`, either operand order, or
    /// `datecol <cmp> datecol`) to surviving row indices via the i32 compare path (the type matrix,
    /// doc 19): a `date` is i32 days since 2000-01-01, so it reuses the int4 residency section + the
    /// I32 VM. The string literal is coerced to a day count (`parse_date`) at lowering, like PG. Returns
    /// None for a non-date predicate. Date `AND`/`OR` / arithmetic, and a date compared to a non-date
    /// value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_date_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            // AND/OR (ADR-006 date compound): fall through — the And/Or branch of
            // `lower_resident_predicate` runs `compile_predicate_program` at I32, whose DATE leaf
            // (LoadColumn 4-byte + CompareScalar days + validity) serves date ranges. A shape the
            // leaf can't express errors there and the caller declines — never a silent mis-answer.
            return Ok(None);
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err)
        };
        match (date_column_index(lhs, table), date_column_index(rhs, table)) {
            (Some(col), None) => {
                let days = date_literal_days(rhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: false,
                    },
                ])
            }
            (None, Some(col)) => {
                let days = date_literal_days(lhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: true,
                    },
                ])
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int4_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, b)?;
                run(&[
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ])
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a date predicate must involve a date column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE timestamp comparison (`ts_col <cmp> 'YYYY-MM-DD HH:MM:SS'`, either order,
    /// or `ts_col <cmp> ts_col`) to surviving row indices via the i64 compare KERNELS (the type matrix,
    /// doc 19): a `timestamp` is i64 microseconds, so it reuses the int8 residency section + the i64
    /// compare kernels. (The i64 microsecond literal exceeds the i32 `ExprStep` scalar, so this uses
    /// `expr_i64_compare_scalar_filter` directly, not the VM's i32-scalar step.) Returns None for a
    /// non-timestamp predicate. Timestamp `AND`/`OR` / arithmetic, and a timestamp compared to a
    /// non-timestamp value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_timestamp_predicate(
        &self,
        predicate: &ResidentExpr,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table)) {
            return Ok(None);
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let Some(cmp) = predicate_compare_code(compare) else {
            // AND/OR (ADR-006 multi-bound timestamp DML): a timestamp is i64 micros in the i64
            // section, so a TIMESTAMP-ONLY (i64-section) AND/OR lowers via the i64 buffer VM — the
            // SAME path int8 AND/OR uses (`compile_predicate_program` emits `CompareScalarI64` for
            // each `Int8Literal` bound the DML builder produced; `resident_device_int_column_offset`
            // resolves the timestamp column to the i64 section). Gate LOCALLY on every column being an
            // i64 section (Int8/Timestamp) — NOT via `predicate_vm_elem_type`, which the nullable-read
            // branch keys off (folding Timestamp in there would divert nullable-timestamp 3VL reads).
            // A MIXED timestamp/other-type predicate fails this check → clean error, never a silent
            // mis-answer at one element width. (`compile_predicate_program`'s TIMESTAMP leaf accepts
            // both an `Int8Literal` (raw micros — the DML builder) AND a `TextLiteral` bound (parsed
            // via `timestamp_literal_micros` — the read path); an unparseable literal errors cleanly,
            // so it can never mis-read a text needle against the i64 column.)
            let mut cols = Vec::new();
            collect_expr_columns(predicate, &mut cols);
            let all_i64_section = !cols.is_empty()
                && cols.iter().all(|&col| {
                    matches!(
                        table.columns.get(col).map(|column| column.ty),
                        Some(SqlType::Int8 | SqlType::Timestamp)
                    )
                });
            if all_i64_section {
                let mut program = Vec::new();
                let mut needles: Vec<Vec<u8>> = Vec::new();
                compile_predicate_program(predicate, table, snapshot, &mut program, &mut needles)?;
                return device_memory
                    .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
                    .map(Some)
                    .map_err(map_err);
            }
            // ADR-006 (MIXED-WIDTH groups): a mixed ts+{int4,text,bool,date,uuid} AND/OR whose
            // i64-section leaves are all WIDTH-SAFE scalar comparisons (the timestamp leaf emits
            // `LoadColumnI64` regardless of the program elem) falls through to the general And/Or
            // I32 branch instead of erroring.
            if mixed_width_i32_elem(predicate, table).is_some() {
                return Ok(None);
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports timestamp AND/OR only over a timestamp-only \
                 (i64) predicate or a mixed group of width-safe scalar leaves; timestamp \
                 arithmetic is a follow-on"
                    .to_string(),
            )));
        };
        match (
            timestamp_column_index(lhs, table),
            timestamp_column_index(rhs, table),
        ) {
            (Some(col), None) => {
                let micros = timestamp_literal_micros(rhs)?;
                let offset = resident_device_int8_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_i64_compare_scalar_filter(offset, micros, false, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, Some(col)) => {
                let micros = timestamp_literal_micros(lhs)?;
                let offset = resident_device_int8_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_i64_compare_scalar_filter(offset, micros, true, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int8_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int8_column_offset(snapshot, table, b)?;
                device_memory
                    .expr_i64_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                    .map(Some)
                    .map_err(map_err)
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a timestamp predicate must involve a timestamp column".to_string(),
            ))),
        }
    }

    /// M3 (doc 21) WHERE 3VL for a SIMPLE comparison over a nullable DATE or TIMESTAMP column (scalar,
    /// either operand order, or col-vs-col). The generic mask VM (`predicate_vm_elem_type`) cannot lower
    /// these: a date/timestamp literal is not an `Int4Literal`, and a timestamp's i64 microseconds exceed
    /// the VM's i32 `CompareScalar` scalar. So build the VM program directly — DATE: I32 + `CompareScalar`
    /// over the i32 days; TIMESTAMP: I64 + `CompareScalarI64` over the i64 micros; col-vs-col:
    /// `CompareBuffers` — then append the validity-AND (`push_column_validity_and` = `BoolMask` +
    /// `MaskBinary` AND) for each nullable operand, so a NULL operand is UNKNOWN ⇒ the row is excluded
    /// (correct at the top level / under no-NOT). Returns `None` for a non-date/timestamp predicate or a
    /// shape the peephole doesn't support (AND/OR / arithmetic), so the caller clean-errors. NO kernel
    /// change: the i64 compare-scalar / compare-buffers / bool-to-mask / mask-binary kernels already exist.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_nullable_temporal_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        // Only a simple comparison (eq/ne/lt/le/gt/ge); AND/OR yield None -> caller clean-errors.
        let Some(cmp) = predicate_compare_code(compare) else {
            return Ok(None);
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // DATE: i32 days in the int4 section -> the I32 VM (CompareScalar fits the i32 days literal).
        if expr_mentions_date(lhs, table) || expr_mentions_date(rhs, table) {
            let mut program = Vec::new();
            match (date_column_index(lhs, table), date_column_index(rhs, table)) {
                (Some(col), None) => {
                    let days = date_literal_days(rhs)?;
                    let off = resident_device_int4_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: false,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (None, Some(col)) => {
                    let days = date_literal_days(lhs)?;
                    let off = resident_device_int4_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalar {
                        cmp,
                        scalar: days,
                        scalar_on_left: true,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (Some(a), Some(b)) => {
                    let a_off = resident_device_int4_column_offset(snapshot, table, a)?;
                    let b_off = resident_device_int4_column_offset(snapshot, table, b)?;
                    program.push(ExprStep::LoadColumn { byte_offset: a_off });
                    program.push(ExprStep::LoadColumn { byte_offset: b_off });
                    program.push(ExprStep::CompareBuffers { cmp });
                    push_column_validity_and(a, table, snapshot, &mut program)?;
                    push_column_validity_and(b, table, snapshot, &mut program)?;
                }
                (None, None) => return Ok(None),
            }
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err);
        }
        // TIMESTAMP: i64 micros in the int8 section -> the I64 VM with CompareScalarI64 (the micros
        // literal exceeds the i32 CompareScalar scalar).
        if expr_mentions_timestamp(lhs, table) || expr_mentions_timestamp(rhs, table) {
            let mut program = Vec::new();
            match (
                timestamp_column_index(lhs, table),
                timestamp_column_index(rhs, table),
            ) {
                (Some(col), None) => {
                    let micros = timestamp_literal_micros(rhs)?;
                    let off = resident_device_int8_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalarI64 {
                        cmp,
                        scalar: micros,
                        scalar_on_left: false,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (None, Some(col)) => {
                    let micros = timestamp_literal_micros(lhs)?;
                    let off = resident_device_int8_column_offset(snapshot, table, col)?;
                    program.push(ExprStep::LoadColumn { byte_offset: off });
                    program.push(ExprStep::CompareScalarI64 {
                        cmp,
                        scalar: micros,
                        scalar_on_left: true,
                    });
                    push_column_validity_and(col, table, snapshot, &mut program)?;
                }
                (Some(a), Some(b)) => {
                    let a_off = resident_device_int8_column_offset(snapshot, table, a)?;
                    let b_off = resident_device_int8_column_offset(snapshot, table, b)?;
                    program.push(ExprStep::LoadColumn { byte_offset: a_off });
                    program.push(ExprStep::LoadColumn { byte_offset: b_off });
                    program.push(ExprStep::CompareBuffers { cmp });
                    push_column_validity_and(a, table, snapshot, &mut program)?;
                    push_column_validity_and(b, table, snapshot, &mut program)?;
                }
                (None, None) => return Ok(None),
            }
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I64)
                .map(Some)
                .map_err(map_err);
        }
        Ok(None)
    }

    /// M3 (doc 21) WHERE 3VL over a nullable NUMERIC column. The generic mask VM can't lower numeric (a
    /// numeric literal is not an Int4Literal; the i128 mantissa exceeds the i32 CompareScalar). Two routes,
    /// both on the I128 VM, each AND'ing the operand columns' validity (a NULL operand is UNKNOWN, excluded
    /// — correct for WHERE under AND/OR since there is no NOT). (a) A SIMPLE same-or-coarser-scale scalar /
    /// same-scale col-vs-col uses a direct `CompareScalarI128` / `CompareBuffers` program (handles a LARGE
    /// mantissa the i32-needle path can't). (b) AND/OR, numeric arithmetic, a FINER cross-scale literal, or
    /// a different-scale col-vs-col uses the shared `compile_numeric_predicate_program` /
    /// `compile_numeric_compare` VM path, now validity-aware (`push_leaf_validity_and` per leaf; same
    /// i32-needle limit as the non-null path). Returns `None` only for a non-numeric predicate; a mixed
    /// numeric/integer predicate is a clean error. NO kernel change.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_nullable_numeric_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)) {
            return Ok(None);
        }
        // Mixed numeric/integer is a clean error (compile_numeric_arith expects numeric operands) — mirror
        // the non-null path's guard so the message is clear rather than an opaque compile failure.
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed numeric/integer expressions yet"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I128)
                .map(Some)
                .map_err(map_err)
        };
        // numeric AND/OR -> the validity-aware compile path (each comparison leaf via compile_numeric_compare,
        // which now AND's the operand validity; MaskBinary combines).
        if let Some(bool_op) = boolean_op_code(compare) {
            let mut program = Vec::new();
            compile_numeric_predicate_program(lhs, table, snapshot, &mut program)?;
            compile_numeric_predicate_program(rhs, table, snapshot, &mut program)?;
            program.push(ExprStep::MaskBinary { op: bool_op });
            return run(&program);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Ok(None);
        };
        // SIMPLE same-or-coarser-scale scalar / same-scale col-vs-col -> the CompareScalarI128 fast path
        // (handles a large mantissa). `None` here means "not this fast shape" -> fall to compile_numeric_compare.
        let scalar_program = |col: usize,
                              literal: Decimal128,
                              scalar_on_left: bool|
         -> Result<Option<Vec<ExprStep>>, ExecuteError> {
            let col_scale = column_numeric_scale(table, col).expect("numeric column has a scale");
            let literal = literal.canonical();
            if literal.scale > col_scale {
                return Ok(None); // finer cross-scale literal -> the compile_numeric_compare fallback
            }
            let mantissa = rescale_numeric_literal(literal, col_scale)?;
            let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
            let mut program = vec![
                ExprStep::LoadColumn {
                    byte_offset: offset,
                },
                ExprStep::CompareScalarI128 {
                    cmp,
                    scalar: mantissa,
                    scalar_on_left,
                },
            ];
            push_column_validity_and(col, table, snapshot, &mut program)?;
            Ok(Some(program))
        };
        let fast: Option<Vec<ExprStep>> = match (
            numeric_column_index(lhs, table),
            numeric_column_index(rhs, table),
        ) {
            (Some(col), None) if numeric_literal_value(rhs).is_some() => {
                scalar_program(col, numeric_literal_value(rhs).expect("checked"), false)?
            }
            (None, Some(col)) if numeric_literal_value(lhs).is_some() => {
                scalar_program(col, numeric_literal_value(lhs).expect("checked"), true)?
            }
            (Some(a), Some(b))
                if column_numeric_scale(table, a) == column_numeric_scale(table, b) =>
            {
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                let mut program = vec![
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ];
                push_column_validity_and(a, table, snapshot, &mut program)?;
                push_column_validity_and(b, table, snapshot, &mut program)?;
                Some(program)
            }
            _ => None,
        };
        if let Some(program) = fast {
            return run(&program);
        }
        // Fallback: a FINER cross-scale literal / different-scale col-vs-col / numeric arithmetic -> the
        // validity-aware shared compile path (compile_numeric_compare appends push_leaf_validity_and).
        let mut program = Vec::new();
        compile_numeric_compare(lhs, rhs, cmp, table, snapshot, &mut program)?;
        run(&program)
    }

    /// Try to lower a SIMPLE uuid comparison (`id <cmp> 'uuid-literal'`, either operand order, or
    /// `id <cmp> id2`) to surviving row indices via the byte-wise uuid compare kernel (the type matrix,
    /// doc 19): a `uuid` is 16 raw bytes in the i128 section; PG compares uuids by an unsigned
    /// big-endian memcmp. The string literal is parsed to 16 bytes at lowering. Returns None for a
    /// non-uuid predicate. Uuid `AND`/`OR`, and a uuid compared to a non-uuid value, are hard errors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_uuid_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_uuid(lhs, table) || expr_mentions_uuid(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            // AND/OR (ADR-006): fall through — the And/Or branch of `lower_resident_predicate` runs
            // `compile_predicate_program`, whose uuid leaf compiles to a `UuidCmpMask` mask-VM step
            // (uuid IN / uuid ranges / mixed uuid+int4/text). Never a silent mis-answer: a shape the
            // leaf compiler can't express errors there and the caller declines.
            return Ok(None);
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // M3 (doc 21): a NULLABLE uuid operand's validity bitmap offset (None when the column holds no
        // NULL); the launcher AND's it with the compare mask so a NULL operand is UNKNOWN ⇒ excluded. A
        // non-nullable column contributes nothing (the no-NULL path stays byte-identical).
        let validity = |col: usize| -> Result<Vec<u64>, ExecuteError> {
            Ok(resident_device_null_column_offset(snapshot, table, col)?
                .into_iter()
                .collect())
        };
        match (uuid_column_index(lhs, table), uuid_column_index(rhs, table)) {
            (Some(col), None) => {
                let needle = uuid_literal_bytes(rhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(
                        offset,
                        &needle,
                        false,
                        cmp,
                        row_count,
                        &validity(col)?,
                    )
                    .map(Some)
                    .map_err(map_err)
            }
            (None, Some(col)) => {
                let needle = uuid_literal_bytes(lhs)?;
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                device_memory
                    .expr_uuid_compare_scalar_filter(
                        offset,
                        &needle,
                        true,
                        cmp,
                        row_count,
                        &validity(col)?,
                    )
                    .map(Some)
                    .map_err(map_err)
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                let mut validity_offsets = validity(a)?;
                validity_offsets.extend(validity(b)?);
                device_memory
                    .expr_uuid_compare_columns_filter(
                        a_offset,
                        b_offset,
                        cmp,
                        row_count,
                        &validity_offsets,
                    )
                    .map(Some)
                    .map_err(map_err)
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a uuid predicate must involve a uuid column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE smallint comparison (`sz <cmp> 5`, either operand order, or `sz <cmp>
    /// sz2`) to surviving row indices via the i32 compare path (the type matrix, doc 19): a `smallint`
    /// is stored widened to i32 in the int4 section, so it REUSES the int4 compare VM with the literal
    /// taken as an i32 scalar. Returns None for a non-smallint predicate. Smallint `AND`/`OR` /
    /// arithmetic (int16-bounds overflow is a follow-on), and a smallint compared to a non-integer
    /// value, are hard errors -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_int2_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_int2(lhs, table) || expr_mentions_int2(rhs, table)) {
            return Ok(None);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            // ADR-006 (MIXED-WIDTH groups): int2 AND/OR falls THROUGH to the general And/Or I32
            // branch — an int2 leaf is a 4-byte `LoadColumn` + `CompareScalar` in an I32 program
            // (the int4 section; `predicate_vm_elem_type` has always classed Int2 as I32), so
            // all-int2/int4 groups and int8+int2 mixes (the audit's MEDIUM: this error used to
            // block the int2 arm of the mixed fall-through) compile there. Non-boolean
            // non-comparison shapes keep the hard error.
            if boolean_op_code(compare).is_some() {
                return Ok(None);
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only simple smallint comparisons (smallint \
                 arithmetic is a follow-on)"
                    .to_string(),
            )));
        };
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let run = |program: &[ExprStep]| {
            device_memory
                .run_expr_predicate_filter(program, row_count, ResidentElemType::I32)
                .map(Some)
                .map_err(map_err)
        };
        match (int2_column_index(lhs, table), int2_column_index(rhs, table)) {
            (Some(col), None) => {
                let scalar = int2_literal_value(rhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar,
                        scalar_on_left: false,
                    },
                ])
            }
            (None, Some(col)) => {
                let scalar = int2_literal_value(lhs)?;
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                run(&[
                    ExprStep::LoadColumn { byte_offset },
                    ExprStep::CompareScalar {
                        cmp,
                        scalar,
                        scalar_on_left: true,
                    },
                ])
            }
            (Some(a), Some(b)) => {
                let a_offset = resident_device_int4_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_int4_column_offset(snapshot, table, b)?;
                run(&[
                    ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    },
                    ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    },
                    ExprStep::CompareBuffers { cmp },
                ])
            }
            (None, None) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a smallint predicate must involve a smallint column".to_string(),
            ))),
        }
    }

    /// Try to lower a SIMPLE text comparison (`textcol = 'lit'` / `textcol <> 'lit'`, either operand
    /// order) to surviving row indices via the byte-wise text-equality kernel (the type matrix, doc
    /// 19). Equality is byte identity -- PG deterministic-collation semantics. Returns None for a
    /// non-text predicate (the other type paths handle it). Inequalities, LIKE, and col-vs-col
    /// (ADR-006) lower via their kernels below; text `AND`/`OR` falls through to the mask VM; text
    /// mixed with another type in ONE comparison is a hard error -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_text_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_text(lhs, table) || expr_mentions_text(rhs, table)) {
            return Ok(None);
        }
        // AND/OR is not a single comparison: fall through (Ok(None)) so the mask VM compiles each leaf
        // (text -> TextEqMask, int4 -> arith+compare) and combines them. A text+int4 AND/OR is VALID
        // even though mixing the two types within ONE comparison (the guard below) is not. Mixed
        // text+int8/numeric AND/OR is rejected later (the int8/numeric paths) -- the i32 mask VM only.
        if matches!(compare, ResidentBinaryOp::And | ResidentBinaryOp::Or) {
            return Ok(None);
        }
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
            || expr_mentions_numeric(lhs, table)
            || expr_mentions_numeric(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed text/non-text expressions"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // COL-VS-COL single comparison (ADR-006): BOTH sides are text columns — run the per-row
        // two-column compare kernel via a one-leaf mask program (`compile_text_eq_leaf`'s
        // col-vs-col arm, which ANDs BOTH validity masks — either operand NULL is UNKNOWN). All
        // six ops; `a LIKE b` (col-vs-col LIKE) errors cleanly inside the leaf. The AND/OR shapes
        // already composed via the fall-through above.
        if text_column_index(lhs, table).is_some() && text_column_index(rhs, table).is_some() {
            let mut program = Vec::new();
            let mut needles: Vec<Vec<u8>> = Vec::new();
            compile_text_eq_leaf(
                compare,
                lhs,
                rhs,
                table,
                snapshot,
                &mut program,
                &mut needles,
            )?;
            return device_memory
                .run_expr_predicate_filter_with_text(
                    &program,
                    &needles,
                    row_count,
                    ResidentElemType::I32,
                )
                .map(Some)
                .map_err(map_err);
        }
        // textcol LIKE 'pattern' (the pattern is on the right; LIKE is NOT symmetric). The host
        // compiles the pattern (resolving `\` escapes) to the kernel's u32 token array.
        if matches!(compare, ResidentBinaryOp::Like) {
            let (col, pattern) = match (text_column_index(lhs, table), text_literal_value(rhs)) {
                (Some(col), Some(pattern)) => (col, pattern),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "LIKE must be a text column LIKE a literal pattern".to_string(),
                    )));
                }
            };
            let tokens = compile_like_pattern(pattern)?;
            let layout = resident_device_text_column_layout(snapshot, table, col)?;
            return device_memory
                .expr_text_like_scalar_filter(
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                    &tokens,
                    row_count,
                )
                .map(Some)
                .map_err(map_err);
        }
        // textcol <lt/le/gt/ge> 'literal' (ADR-006): LEXICOGRAPHIC unsigned byte compare on-device
        // (memcmp of the common prefix; shorter sorts first) — byte-identical to the host
        // `compare_sql_values` Text order (Rust `str::cmp`) that the DML recheck uses, so device ==
        // recheck. A column-on-RIGHT (`'lit' < col`) flips `scalar_on_left`; a nullable text column
        // AND's its validity mask (a NULL operand is UNKNOWN ⇒ excluded).
        if matches!(
            compare,
            ResidentBinaryOp::Lt
                | ResidentBinaryOp::Le
                | ResidentBinaryOp::Gt
                | ResidentBinaryOp::Ge
        ) {
            let (col, needle, scalar_on_left) =
                match (text_column_index(lhs, table), text_column_index(rhs, table)) {
                    (Some(col), None) if text_literal_value(rhs).is_some() => {
                        (col, text_literal_value(rhs).expect("checked"), false)
                    }
                    (None, Some(col)) if text_literal_value(lhs).is_some() => {
                        (col, text_literal_value(lhs).expect("checked"), true)
                    }
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "text comparison must be a column against a literal".to_string(),
                        )));
                    }
                };
            let cmp = predicate_compare_code(compare).expect("lt/le/gt/ge have compare codes");
            let layout = resident_device_text_column_layout(snapshot, table, col)?;
            let validity: Vec<u64> = resident_device_null_column_offset(snapshot, table, col)?
                .into_iter()
                .collect();
            return device_memory
                .expr_text_compare_scalar_filter(
                    layout.offsets_byte_offset,
                    layout.bytes_byte_offset,
                    layout.bytes_len,
                    needle.as_bytes(),
                    scalar_on_left,
                    cmp,
                    row_count,
                    &validity,
                )
                .map(Some)
                .map_err(map_err);
        }
        let negate = match compare {
            ResidentBinaryOp::Eq => false,
            ResidentBinaryOp::Ne => true,
            ResidentBinaryOp::Lt
            | ResidentBinaryOp::Le
            | ResidentBinaryOp::Gt
            | ResidentBinaryOp::Ge => {
                // Handled by the inequality block above; unreachable here.
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text inequality reached the eq/ne arm unexpectedly".to_string(),
                )));
            }
            // AND/OR returned early (above) -> the mask VM. Any other op here is unsupported.
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "the general GPU executor supports text = and <> only (LIKE is a follow-on)"
                        .to_string(),
                )));
            }
        };
        // textcol <eq/ne> 'literal' -- equality is symmetric, so operand order does not matter.
        let (col, literal) = match (text_column_index(lhs, table), text_column_index(rhs, table)) {
            (Some(col), None) if text_literal_value(rhs).is_some() => {
                (col, text_literal_value(rhs).expect("checked"))
            }
            (None, Some(col)) if text_literal_value(lhs).is_some() => {
                (col, text_literal_value(lhs).expect("checked"))
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "text comparison must be a column against a literal".to_string(),
                )));
            }
        };
        let layout = resident_device_text_column_layout(snapshot, table, col)?;
        device_memory
            .expr_text_eq_scalar_filter(
                layout.offsets_byte_offset,
                layout.bytes_byte_offset,
                layout.bytes_len,
                literal.as_bytes(),
                negate,
                row_count,
            )
            .map(Some)
            .map_err(map_err)
    }

    /// Try to lower a SIMPLE numeric comparison to surviving row indices via the i128 compare kernels
    /// (the type matrix, doc 19). Supported: `numcol <cmp> literal` / `literal <cmp> numcol` and
    /// `numcol <cmp> numcol`, at ANY scale -- same-scale (and coarser-or-equal literal) use the
    /// resident compare peephole; a SCALE MISMATCH (a finer literal, or two columns of different scale)
    /// loads to i128 buffers, rescales the coarser side UP to the common = max scale (mantissa * 10^k,
    /// k <= 9; a wider gap or a rescale overflow errors), then compares. Numeric ARITHMETIC comparisons
    /// and `AND`/`OR` of numeric comparisons also lower here (each comparison -> a mask via the i128
    /// VM, MaskBinary combines). Returns None for a non-numeric predicate (the int4/int8 paths handle
    /// it). A numeric value mixed with an int4/int8 COLUMN is a hard error -- never a silent mis-answer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_lower_numeric_predicate(
        &self,
        compare: ResidentBinaryOp,
        lhs: &ResidentExpr,
        rhs: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Option<Vec<u32>>, ExecuteError> {
        if !(expr_mentions_numeric(lhs, table) || expr_mentions_numeric(rhs, table)) {
            return Ok(None);
        }
        if expr_mentions_int4_column(lhs, table)
            || expr_mentions_int4_column(rhs, table)
            || expr_mentions_int8(lhs, table)
            || expr_mentions_int8(rhs, table)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor does not support mixed numeric/integer expressions yet"
                    .to_string(),
            )));
        }
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        // numeric AND/OR: compile each side's comparison(s) to a MASK (the buffer path), combine with
        // MaskBinary, and compact -- the same i128 mask VM the single-comparison arithmetic path uses.
        if let Some(bool_op) = boolean_op_code(compare) {
            let mut program = Vec::new();
            compile_numeric_predicate_program(lhs, table, snapshot, &mut program)?;
            compile_numeric_predicate_program(rhs, table, snapshot, &mut program)?;
            program.push(ExprStep::MaskBinary { op: bool_op });
            return device_memory
                .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                .map(Some)
                .map_err(map_err);
        }
        let Some(cmp) = predicate_compare_code(compare) else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "the general GPU executor supports only numeric comparisons and AND/OR".to_string(),
            )));
        };
        match (
            numeric_column_index(lhs, table),
            numeric_column_index(rhs, table),
        ) {
            (Some(col), None) if numeric_literal_value(rhs).is_some() => {
                let col_scale =
                    column_numeric_scale(table, col).expect("numeric column has a scale");
                let literal = numeric_literal_value(rhs).expect("checked").canonical();
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                if literal.scale <= col_scale {
                    // literal coarser-or-equal: rescale it UP to the column scale; resident peephole.
                    let mantissa = rescale_numeric_literal(literal, col_scale)?;
                    device_memory
                        .expr_i128_compare_scalar_filter(offset, mantissa, false, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    // CROSS-SCALE: the literal is FINER than the column -> rescale the column UP to the
                    // literal's scale (buffer VM path) and compare to the literal's mantissa.
                    self.numeric_cross_scale_scalar(
                        offset,
                        col_scale,
                        literal,
                        false,
                        cmp,
                        device_memory,
                        row_count,
                    )
                    .map(Some)
                }
            }
            (None, Some(col)) if numeric_literal_value(lhs).is_some() => {
                let col_scale =
                    column_numeric_scale(table, col).expect("numeric column has a scale");
                let literal = numeric_literal_value(lhs).expect("checked").canonical();
                let offset = resident_device_numeric_column_offset(snapshot, table, col)?;
                if literal.scale <= col_scale {
                    let mantissa = rescale_numeric_literal(literal, col_scale)?;
                    device_memory
                        .expr_i128_compare_scalar_filter(offset, mantissa, true, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    self.numeric_cross_scale_scalar(
                        offset,
                        col_scale,
                        literal,
                        true,
                        cmp,
                        device_memory,
                        row_count,
                    )
                    .map(Some)
                }
            }
            (Some(a), Some(b)) => {
                let a_scale = column_numeric_scale(table, a).expect("numeric column has a scale");
                let b_scale = column_numeric_scale(table, b).expect("numeric column has a scale");
                let a_offset = resident_device_numeric_column_offset(snapshot, table, a)?;
                let b_offset = resident_device_numeric_column_offset(snapshot, table, b)?;
                if a_scale == b_scale {
                    device_memory
                        .expr_i128_compare_columns_filter(a_offset, b_offset, cmp, row_count)
                        .map(Some)
                        .map_err(map_err)
                } else {
                    // CROSS-SCALE: load both, rescale the coarser column UP to the common (max) scale,
                    // then compare buffers (the buffer VM path).
                    let common = a_scale.max(b_scale);
                    let mut program = vec![ExprStep::LoadColumn {
                        byte_offset: a_offset,
                    }];
                    push_numeric_rescale(a_scale, common, &mut program)?;
                    program.push(ExprStep::LoadColumn {
                        byte_offset: b_offset,
                    });
                    push_numeric_rescale(b_scale, common, &mut program)?;
                    program.push(ExprStep::CompareBuffers { cmp });
                    device_memory
                        .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                        .map(Some)
                        .map_err(map_err)
                }
            }
            _ => {
                // Numeric ARITHMETIC comparison (`price * tax > 100`, `100 < price - fee`, ...):
                // compile both sides to a mask via the shared comparison core (each side's scale is
                // computed bottom-up and the coarser brought up to the common scale), then run the VM.
                let mut program = Vec::new();
                compile_numeric_compare(lhs, rhs, cmp, table, snapshot, &mut program)?;
                device_memory
                    .run_expr_predicate_filter(&program, row_count, ResidentElemType::I128)
                    .map(Some)
                    .map_err(map_err)
            }
        }
    }
}
