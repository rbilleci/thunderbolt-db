//! Device predicate-mask construction for resident inputs and synthetic all-NULL OUTER pads.
//! Consumers retain orchestration; this leaf owns compiler selection, visibility conjunctions,
//! launch, and the pad mask/source/allocation lifetime bundle.

use super::execution_source::ResidentVisibility;
use super::join_source::JoinNullPadMask;
use super::predicate_compiler::{
    compile_numeric_predicate_program, compile_predicate_program, mixed_width_i32_elem,
    predicate_vm_elem_type,
};
use super::predicate_operands::expr_mentions_numeric;
use crate::engine_expr_ir::ResidentExpr;
use crate::relational_model::{RelationalResidencySnapshot, RelationalTable};
use crate::{Engine, ExecuteError};
use gpu_db_execution::{
    CudaAllocationScope, CudaPredicateMaskI32, CudaResidentDeviceMemory, ResidentElemType,
};
use gpu_db_sql::SqlValue;
use gpu_db_types::EngineError;

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
        let mut program = Vec::new();
        let mut needles = Vec::new();
        let elem = if let Some(predicate) = predicate {
            if expr_mentions_numeric(predicate, table) {
                compile_numeric_predicate_program(predicate, table, descriptor, &mut program)?;
                ResidentElemType::I128
            } else {
                compile_predicate_program(
                    predicate,
                    table,
                    descriptor,
                    &mut program,
                    &mut needles,
                )?;
                predicate_vm_elem_type(predicate, table)
                    .or_else(|| mixed_width_i32_elem(predicate, table))
                    .unwrap_or(ResidentElemType::I32)
            }
        } else {
            ResidentElemType::I64
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
}
