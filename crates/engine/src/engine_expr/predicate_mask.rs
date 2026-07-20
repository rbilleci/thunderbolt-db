//! Device predicate-mask construction for resident inputs and synthetic all-NULL OUTER pads.
//! Consumers retain orchestration; this leaf owns compiler selection, visibility conjunctions,
//! launch, and the pad mask/source/allocation lifetime bundle.

use super::execution_source::ResidentVisibility;
use super::join_source::JoinNullPadMask;
use super::predicate_compiler::{
    collect_expr_columns, compile_numeric_predicate_program, compile_predicate_program,
    mixed_width_i32_elem, predicate_vm_elem_type,
};
use super::predicate_operands::expr_mentions_numeric;
use crate::engine_expr_ir::ResidentExpr;
use crate::relational_model::{RelationalResidencySnapshot, RelationalTable};
use crate::{Engine, ExecuteError};
use gpu_db_execution::{
    CudaAllocationScope, CudaPredicateMaskI32, CudaResidentDeviceMemory, ResidentElemType,
};
use gpu_db_sql::{SqlType, SqlValue};
use gpu_db_types::EngineError;

fn validate_temporal_predicate_types(
    predicate: &ResidentExpr,
    table: &RelationalTable,
) -> Result<(), ExecuteError> {
    let ResidentExpr::Binary { op, lhs, rhs } = predicate else {
        return Ok(());
    };
    if matches!(
        op,
        crate::engine_expr_ir::ResidentBinaryOp::And | crate::engine_expr_ir::ResidentBinaryOp::Or
    ) {
        validate_temporal_predicate_types(lhs, table)?;
        return validate_temporal_predicate_types(rhs, table);
    }
    let column_type = |expr: &ResidentExpr| match expr {
        ResidentExpr::Column(column) => table.columns.get(*column).map(|column| column.ty),
        _ => None,
    };
    let mut columns = Vec::new();
    collect_expr_columns(predicate, &mut columns);
    let mentions_timestamp = columns.iter().any(|&column| {
        table.columns.get(column).map(|column| column.ty) == Some(SqlType::Timestamp)
    });
    if mentions_timestamp {
        let valid = match (lhs.as_ref(), rhs.as_ref()) {
            (ResidentExpr::Column(_), ResidentExpr::Column(_)) => {
                column_type(lhs) == Some(SqlType::Timestamp)
                    && column_type(rhs) == Some(SqlType::Timestamp)
            }
            (ResidentExpr::Column(_), ResidentExpr::TextLiteral(_))
            | (ResidentExpr::Column(_), ResidentExpr::Int8Literal(_)) => {
                column_type(lhs) == Some(SqlType::Timestamp)
            }
            (ResidentExpr::TextLiteral(_), ResidentExpr::Column(_))
            | (ResidentExpr::Int8Literal(_), ResidentExpr::Column(_)) => {
                column_type(rhs) == Some(SqlType::Timestamp)
            }
            _ => false,
        };
        if !valid {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a timestamp column compares only to a timestamp literal or another timestamp column"
                    .to_string(),
            )));
        }
    }
    let mentions_date = columns
        .iter()
        .any(|&column| table.columns.get(column).map(|column| column.ty) == Some(SqlType::Date));
    if mentions_date {
        let valid = match (lhs.as_ref(), rhs.as_ref()) {
            (ResidentExpr::Column(_), ResidentExpr::Column(_)) => {
                column_type(lhs) == Some(SqlType::Date) && column_type(rhs) == Some(SqlType::Date)
            }
            (ResidentExpr::Column(_), ResidentExpr::TextLiteral(_)) => {
                column_type(lhs) == Some(SqlType::Date)
            }
            (ResidentExpr::TextLiteral(_), ResidentExpr::Column(_)) => {
                column_type(rhs) == Some(SqlType::Date)
            }
            _ => false,
        };
        if !valid {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "a date column compares only to a date literal or another date column".to_string(),
            )));
        }
    }
    Ok(())
}

impl Engine {
    /// Compile a per-relation WHERE predicate over an OUTER join's synthetic all-NULL pad into a
    /// retained one-row DEVICE mask (S6 -- doc 22). The filter kernel consumes `mask[0]` directly;
    /// no host boolean verdict or semantic re-upload exists.
    /// Build a 1-row TRANSIENT relation whose every column is NULL and run the predicate through the SAME
    /// GPU WHERE-3VL mask VM as the real rows (`lower_resident_predicate`); the pad survives iff row 0
    /// survives. On the all-NULL row a comparison/arithmetic/bare-column leaf is AND'd with the (all-zero)
    /// validity mask -> UNKNOWN -> excluded; `col IS NULL` reads the 0 validity bit -> TRUE (the anti-join);
    /// AND/OR fold via the VM's Kleene masks. So the host no longer evaluates 3VL -- the GPU does. A
    /// predicate shape the VM cannot lower over a nullable column clean-errors there (the engine-wide WHERE
    /// 3VL limitation), exactly as the real-row survivor pass would for the same shape.
    pub(super) fn predicate_mask_on_null_pad(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
    ) -> Result<JoinNullPadMask, ExecuteError> {
        // One all-NULL row has no text bytes. Per column, 64 bytes strictly dominates the widest
        // 16-byte value/offset slot, its validity word, bool storage, and every <=8-byte section
        // alignment pad; the fixed 64-byte header allowance dominates the resident header/final pad.
        // Reserve this conservative raw-allocation bound BEFORE the transient builder allocates.
        let pad_source_bound = 64_usize.saturating_add(table.columns.len().saturating_mul(64));
        let allocation = CudaAllocationScope::reserve_external(pad_source_bound)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let null_row = vec![vec![SqlValue::Null; table.columns.len()]];
        let (snapshot, memory) = self.build_transient_relation_residency(table, &null_row)?;
        let mask = self.resident_predicate_device_mask(
            Some(predicate),
            table,
            &snapshot,
            &memory,
            1,
            None,
        )?;
        Ok(JoinNullPadMask {
            mask,
            _source: memory,
            _allocation: allocation,
        })
    }

    pub(crate) fn resident_predicate_device_mask(
        &self,
        predicate: Option<&ResidentExpr>,
        table: &RelationalTable,
        descriptor: &RelationalResidencySnapshot,
        memory: &CudaResidentDeviceMemory,
        row_count: u32,
        visibility: Option<ResidentVisibility>,
    ) -> Result<Option<CudaPredicateMaskI32>, ExecuteError> {
        if row_count == 0 || (predicate.is_none() && visibility.is_none()) {
            return Ok(None);
        }
        let (mut program, needles, elem) = match predicate {
            Some(predicate) => compile_resident_predicate(predicate, table, descriptor)?,
            None => (Vec::new(), Vec::new(), ResidentElemType::I64),
        };
        if let Some(visibility) = visibility {
            visibility.push_conjuncts(&mut program, predicate.is_some());
        }
        #[cfg(feature = "probe-timing")]
        eprintln!(
            "[probe] predicate_mask table={} rows={} elem={elem:?} steps={program:?} needles={needles:?}",
            table.name, row_count
        );
        memory
            .run_expr_predicate_mask_with_text(&program, &needles, row_count, elem)
            .map(Some)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    /// Exact predicate verdict over a bounded index candidate set. This is the collision-recheck
    /// terminal for prepared point routes: the index supplies addresses, while the fixed-width
    /// typed predicate runs at those addresses on the GPU.
    pub(crate) fn resident_predicate_device_filter_at_indices(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
        descriptor: &RelationalResidencySnapshot,
        memory: &CudaResidentDeviceMemory,
        indices: &[u32],
    ) -> Result<Vec<u32>, ExecuteError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let (program, needles, elem) = compile_resident_predicate(predicate, table, descriptor)?;
        if !needles.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "indexed candidate predicates currently require fixed-width device operands"
                    .to_string(),
            )));
        }
        let row_count = u32::try_from(descriptor.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident candidate source exceeds u32 device coordinates".to_string(),
            ))
        })?;
        memory
            .run_expr_predicate_filter_at_indices(&program, row_count, indices, elem)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }
}

type CompiledResidentPredicate = (
    Vec<gpu_db_execution::ExprStep>,
    Vec<Vec<u8>>,
    ResidentElemType,
);

fn compile_resident_predicate(
    predicate: &ResidentExpr,
    table: &RelationalTable,
    descriptor: &RelationalResidencySnapshot,
) -> Result<CompiledResidentPredicate, ExecuteError> {
    validate_temporal_predicate_types(predicate, table)?;
    let mut program = Vec::new();
    let mut needles = Vec::new();
    let mut columns = Vec::new();
    collect_expr_columns(predicate, &mut columns);
    let elem = if expr_mentions_numeric(predicate, table) {
        if columns.iter().any(|&column| {
            !matches!(
                table.columns.get(column).map(|column| column.ty),
                Some(SqlType::Numeric { .. })
            )
        }) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "mixed numeric/non-numeric resident predicate is not yet supported by device lowering"
                    .to_string(),
            )));
        }
        compile_numeric_predicate_program(predicate, table, descriptor, &mut program)?;
        ResidentElemType::I128
    } else {
        compile_predicate_program(predicate, table, descriptor, &mut program, &mut needles)?;
        predicate_vm_elem_type(predicate, table)
            .or_else(|| mixed_width_i32_elem(predicate, table))
            .or_else(|| {
                (!columns.is_empty()
                    && columns.iter().all(|&column| {
                        matches!(
                            table.columns.get(column).map(|column| column.ty),
                            Some(SqlType::Int8 | SqlType::Timestamp)
                        )
                    }))
                .then_some(ResidentElemType::I64)
            })
            .or_else(|| {
                (!columns.is_empty()
                    && columns.iter().all(|&column| {
                        matches!(
                            table.columns.get(column).map(|column| column.ty),
                            Some(
                                SqlType::Int2
                                    | SqlType::Int4
                                    | SqlType::Date
                                    | SqlType::Text
                                    | SqlType::Bool
                                    | SqlType::Uuid
                            )
                        )
                    }))
                .then_some(ResidentElemType::I32)
            })
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident predicate uses incompatible device element widths".to_string(),
                ))
            })?
    };
    Ok((program, needles, elem))
}
