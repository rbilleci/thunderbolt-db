//! Resident-append eligibility adapter for already prepared typed INSERT semantics.
//!
//! This module owns no SQL interpretation, scalar/default evaluation, RETURNING binding, or
//! sequence effect. It only applies the existing physical compiler gates to the move-only draft.

use super::*;

pub(super) fn build(
    insert: &Insert,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
    capability: TypedInsertBuildCapability,
) -> Result<TypedInsertBuildResult, ExecuteError> {
    let Some(pre_semantics) = semantics::resolve_pre_semantic_insert(
        insert,
        catalog,
        prepared_catalog_seq,
        expected_catalog_version,
        InsertStatementOrdinal::FIRST,
    )?
    else {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::CatalogGeneration,
        ));
    };
    let table = pre_semantics.table;

    // Current catalog identity is an admission invariant, not an eligibility preference: a
    // stale/forged owner must decline before any physical capability can classify the shape.
    let requires_current_table_oid = capability == TypedInsertBuildCapability::ResidentAppend || {
        #[cfg(test)]
        {
            capability == TypedInsertBuildCapability::ProofOnly
                || capability == TypedInsertBuildCapability::ForeignKeyProofOnly
        }
        #[cfg(not(test))]
        {
            false
        }
    };
    if requires_current_table_oid
        && table
            .columns
            .iter()
            .any(|column| column.table_oid != table.oid)
    {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::CatalogGeneration,
        ));
    }

    // A physical builder must reject unsupported table constraints before semantic preparation.
    // In particular, ProofOnly intentionally declines an FK-bearing catalog/shape without
    // capturing its otherwise valid canonical parent closure. SemanticOnly remains below this
    // gate: it owns complete effect evidence, not a resident-append eligibility claim.
    let pre_semantic_constraints_supported = {
        #[cfg(test)]
        if capability == TypedInsertBuildCapability::SemanticOnly {
            true
        } else {
            if capability == TypedInsertBuildCapability::ProofOnly {
                table.foreign_keys.is_empty()
                    && crate::engine_insert_plan::row_local_constraints::checks_are_device_supported(
                        table,
                    )
                    && crate::engine_insert_plan::batch_key_constraints::table_has_supported_batch_key_constraints(table)
            } else if capability == TypedInsertBuildCapability::ForeignKeyProofOnly {
                !table.foreign_keys.is_empty()
                    && crate::engine_insert_plan::row_local_constraints::checks_are_device_supported(
                        table,
                    )
                    && crate::engine_insert_plan::batch_key_constraints::table_has_supported_batch_key_constraints(table)
            } else {
                crate::engine_insert_plan::row_local_constraints::table_has_supported_row_local_checks(
                    table,
                )
            }
        }
        #[cfg(not(test))]
        {
            crate::engine_insert_plan::row_local_constraints::table_has_supported_row_local_checks(
                table,
            )
        }
    };
    if !pre_semantic_constraints_supported {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::Constraints,
        ));
    }

    // The live resident-append route has no result-frame or sequence-effect owner.  Decline
    // before scalar-default lowering so legacy executes each scalar default exactly once.
    // SemanticOnly/effect callers bypass this physical gate and retain the complete semantics.
    if capability == TypedInsertBuildCapability::ResidentAppend
        && (!insert.returning.is_empty() || semantics::requests_sequence_default(&pre_semantics))
    {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::PhysicalUnsupported,
        ));
    }

    let prepared = semantics::prepare_resolved_typed_insert_semantics(
        pre_semantics,
        catalog,
        prepared_catalog_seq,
    )?;
    if prepared.has_returning() || prepared.has_sequence_requests() {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::PhysicalUnsupported,
        ));
    }
    let proof_only_indexed_constraints = {
        #[cfg(test)]
        {
            capability == TypedInsertBuildCapability::ProofOnly && !table.indexes.is_empty()
        }
        #[cfg(not(test))]
        {
            false
        }
    };
    let foreign_key_proof_only = {
        #[cfg(test)]
        {
            capability == TypedInsertBuildCapability::ForeignKeyProofOnly
        }
        #[cfg(not(test))]
        {
            false
        }
    };
    let batch = prepared.seal(
        sequence_defaults::SequenceDefaultBindings::empty(),
        proof_only_indexed_constraints,
        foreign_key_proof_only,
    )?;
    #[cfg(not(test))]
    let _ = capability;
    Ok(TypedInsertBuildResult::Ready(batch))
}
