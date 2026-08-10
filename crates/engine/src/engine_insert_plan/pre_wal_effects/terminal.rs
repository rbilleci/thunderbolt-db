//! Inert WRITE-001 receipt terminal.
//!
//! The terminal consumes one pre-WAL plan and a test-only synthetic receipt bundle, validates the
//! entire bundle before seal, drops the sealed typed batch internally, and returns metadata-only
//! evidence. It has no live execution consumer.

use super::receipt::SequenceOutputEvidence;
#[cfg(test)]
use super::receipt::SequenceReceiptBundle;
#[cfg(test)]
use super::{
    baseline_drift, classification, Engine, ExecuteError, InsertEffectBaseline,
    PreparedInsertEffectPlan,
};
#[cfg(test)]
use std::sync::Arc;

/// Metadata-only result of an inert seal. No typed value vector, result container, or physical
/// plan can escape this type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct InertEffectSealEvidence {
    pub(super) txn_id: crate::TxnId,
    pub(super) autocommit: bool,
    pub(super) parent_request_digest: gpu_db_wal::CanonicalDigest,
    pub(super) statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal,
    pub(super) expression_ordinal_base: u32,
    pub(super) sequence_outputs: Box<[SequenceOutputEvidence]>,
    pub(super) row_count: u32,
    pub(super) returning: ReturningSealEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct ReturningSealEvidence {
    pub(super) row_count: u32,
    pub(super) column_count: u32,
    pub(super) cell_count: u64,
    pub(super) projections: Box<[ReturningProjectionSealEvidence]>,
    pub(super) projection_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct ReturningProjectionSealEvidence {
    pub(super) catalog_column_ordinal: u32,
    pub(super) column_id: u32,
    pub(super) attnum: i16,
    pub(super) name: Box<str>,
    pub(super) ty: crate::SqlType,
    pub(super) type_oid: u32,
    pub(super) type_size: i16,
}

/// The only inert terminal entrance. Its inputs are consumed exactly once. The autocommit branch
/// uses the canonical quiescent commit cut, while the explicit branch resolves and locks the exact
/// registered snapshot before it revalidates and seals.
#[cfg(test)]
pub(super) fn seal_for_test(
    plan: PreparedInsertEffectPlan,
    engine: &Engine,
    receipts: SequenceReceiptBundle,
) -> Result<InertEffectSealEvidence, ExecuteError> {
    match &plan.baseline {
        InsertEffectBaseline::Autocommit(_) => {
            if engine.current_transaction_read_snapshot().is_some() {
                return Err(baseline_drift(
                    "autocommit inert seal cannot run inside a transaction-read scope",
                ));
            }
            let _commit_guard = engine.commit_state_after_wave_quiescence()?;
            plan.validate_current(engine)?;
            seal_after_currentness(plan, receipts)
        }
        InsertEffectBaseline::Explicit(baseline) => {
            let snapshot = Arc::clone(&baseline.snapshot);
            let registered = engine
                .transaction_snapshot_handle(plan.parent.txn_id)
                .ok_or_else(|| baseline_drift("explicit seal snapshot is not registered"))?;
            if !Arc::ptr_eq(&registered, &snapshot) {
                return Err(baseline_drift(
                    "explicit seal snapshot registration differs from its captured owner",
                ));
            }
            let statement_lock = Arc::clone(&snapshot.statement_lock);
            let statement_guard = statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            plan.validate_current_statement_locked(engine, &statement_guard)?;
            seal_after_currentness(plan, receipts)
        }
    }
}

#[cfg(test)]
fn seal_after_currentness(
    plan: PreparedInsertEffectPlan,
    receipts: SequenceReceiptBundle,
) -> Result<InertEffectSealEvidence, ExecuteError> {
    let bindings = classification::build_seal_bindings_for_test(
        &plan.prepared,
        &plan.parent,
        &plan.sequence_effects,
        receipts,
    )?;
    let returning = returning_evidence(&plan.prepared)?;
    let evidence = InertEffectSealEvidence {
        txn_id: plan.parent.txn_id,
        autocommit: plan.parent.autocommit,
        parent_request_digest: plan.parent.request_digest,
        statement_ordinal: plan.parent.statement_ordinal,
        expression_ordinal_base: plan.parent.expression_ordinal_base,
        sequence_outputs: bindings.outputs,
        row_count: plan.prepared.effect_row_count(),
        returning,
    };
    let PreparedInsertEffectPlan { prepared, .. } = plan;
    let sealed = prepared.seal(bindings.bindings)?;
    drop(sealed);
    Ok(evidence)
}

#[cfg(test)]
fn returning_evidence(
    prepared: &crate::typed_insert_batch::PreparedTypedInsert,
) -> Result<ReturningSealEvidence, ExecuteError> {
    let shape = prepared.effect_returning_shape();
    let mut projections = Vec::with_capacity(
        usize::try_from(shape.column_count())
            .expect("u32 RETURNING projection count is addressable"),
    );
    for projection in prepared.effect_returning_projection_identities() {
        projections.push(ReturningProjectionSealEvidence {
            catalog_column_ordinal: projection.catalog_column_ordinal(),
            column_id: projection.column_id(),
            attnum: projection.attnum(),
            name: projection.name().into(),
            ty: projection.ty(),
            type_oid: projection.type_oid(),
            type_size: projection.type_size(),
        });
    }
    if projections.len() != usize::try_from(shape.column_count()).unwrap_or(usize::MAX)
        || u64::from(shape.row_count()).checked_mul(u64::from(shape.column_count()))
            != Some(shape.cell_count())
    {
        return Err(baseline_drift(
            "RETURNING projection geometry differs from typed semantic shape",
        ));
    }
    let projection_digest = crate::typed_insert_batch::canonical_returning_layout_digest(prepared)
        .map_err(ExecuteError::Engine)?;
    Ok(ReturningSealEvidence {
        row_count: shape.row_count(),
        column_count: shape.column_count(),
        cell_count: shape.cell_count(),
        projections: projections.into(),
        projection_digest,
    })
}

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod tests;
