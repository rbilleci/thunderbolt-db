//! Composite pre-WAL INSERT constraint owner.
//!
//! This is the sole owner of the short-lived source, CUDA allocation scope, sequential operator
//! schedule, and cross-class SQL diagnostic arbitration.  It completes before a WAL template,
//! allocator identity, residency append, or publication can exist.

use super::{batch_key_constraints, row_local_constraints, CatalogSnapshot, Engine, EngineError};
use crate::relational_model::RelationalTable;
use crate::typed_insert_batch::TypedInsertBatch;
use gpu_db_execution::CudaAllocationScope;

use super::constraint_arbitration::ConstraintCandidate;

/// Run the sole device-native local CHECK and batch-key validation for a transaction-private
/// typed stage. The proof is consumed here because canonical transaction commit independently
/// revalidates current-resident UNIQUE history from its final typed record.
pub(crate) fn validate_before_transaction_typed_stage(
    engine: &Engine,
    batch: &TypedInsertBatch,
    catalog: &CatalogSnapshot,
) -> Result<(), EngineError> {
    // This batch-local preflight runs before the candidate shard joins the transaction-private
    // generation, so it cannot yet decide FK provider membership. The shared overlay publishes
    // that candidate privately, performs the immediate device FK verdict, and restores it on a
    // rejection before the statement is accepted. COMMIT still owns final-image and concurrent
    // parent/child conflict revalidation over the canonical ordered record.
    if let Some(candidate) = validate_transaction_stage_inner(engine, batch, catalog)? {
        return Err(candidate.into_error());
    }
    Ok(())
}

fn validate_transaction_stage_inner(
    engine: &Engine,
    batch: &TypedInsertBatch,
    catalog: &CatalogSnapshot,
) -> Result<Option<ConstraintCandidate>, EngineError> {
    let table = bound_table(batch, catalog)?;
    let checks = row_local_constraints::compile(batch, table)?;
    let keys = batch_key_constraints::compile(batch, table)?;
    let check_scratch = row_local_constraints::scratch_bytes(batch, table)?;
    let key_scratch = batch_key_constraints::max_scratch_bytes(batch, &keys)?;
    let needs_source = !table.check_constraints.is_empty() || keys.requires_device_source();

    let mut candidates = Vec::new();
    if needs_source {
        let source_bytes = batch.row_local_constraint_device_payload_bytes()?;
        let operator_scratch = check_scratch.max(key_scratch);
        let peak_bytes = source_bytes.checked_add(operator_scratch).ok_or_else(|| {
            EngineError::ApplyFailed("device pre-WAL allocation peak overflows".to_string())
        })?;
        let gpu_id = engine.planner.default_gpu_id();
        let budget = match engine.relational_residency_budget_bytes(gpu_id) {
            Some(limit) => limit
                .checked_sub(engine.relational_resident_bytes_for_gpu(gpu_id))
                .ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "device pre-WAL allocation refused: resident data already exceeds its budget"
                            .to_string(),
                    )
                })?,
            None => peak_bytes,
        };
        let allocation_scope = CudaAllocationScope::with_budget(budget);
        CudaAllocationScope::ensure_available(peak_bytes).map_err(|error| {
            EngineError::ApplyFailed(format!("device pre-WAL allocation refused: {error}"))
        })?;
        let source = batch
            .row_local_constraint_device_source(engine, table)
            .map_err(row_local_constraints::as_engine_error)?;
        debug_assert_eq!(source.payload_bytes(), source_bytes);

        for candidate in row_local_constraints::evaluate(engine, batch, table, &source, &checks)? {
            let constraint = &table.check_constraints[candidate.check_ordinal];
            candidates.push(ConstraintCandidate::check(
                candidate.row,
                table.name.clone(),
                constraint.name.clone(),
                candidate.check_ordinal,
            ));
        }
        for candidate in batch_key_constraints::evaluate(batch, &source, &keys)? {
            match candidate {
                batch_key_constraints::BatchKeyCandidate::PrimaryKeyNull { row, column } => {
                    candidates.push(ConstraintCandidate::primary_key_null(
                        row,
                        table.name.clone(),
                        column.name,
                        column.attnum,
                    ))
                }
                batch_key_constraints::BatchKeyCandidate::Duplicate { row, index_ordinal } => {
                    candidates.push(ConstraintCandidate::unique(
                        row,
                        index_ordinal,
                        keys.index(index_ordinal).name.clone(),
                    ));
                }
            }
        }
        // Source leases disappear before the scope.  On every error path lexical Drop preserves
        // the same order, so the common payload cannot outlive the accounting scope.
        drop(source);
        drop(allocation_scope);
    }

    Ok(ConstraintCandidate::merge(candidates))
}

pub(super) fn bound_table<'a>(
    batch: &TypedInsertBatch,
    catalog: &'a CatalogSnapshot,
) -> Result<&'a RelationalTable, EngineError> {
    let (table_oid, schema_digest, catalog_seq) = batch.row_local_constraint_target();
    let table = catalog
        .relational_catalog
        .values()
        .find(|table| table.oid == table_oid)
        .ok_or_else(|| {
            EngineError::ApplyFailed("device pre-WAL target relation is absent".to_string())
        })?;
    if catalog.commit_seq != catalog_seq
        || crate::engine_transaction_reset::table_schema_digest(table).ok() != Some(schema_digest)
    {
        return Err(EngineError::ApplyFailed(
            "device pre-WAL target binding drifted before off-lock preparation".to_string(),
        ));
    }
    Ok(table)
}
