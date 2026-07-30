//! Inert INSERT effect-plan capture and sequence classification.
//!
//! This leaf owns semantic and transaction-generation identity plus stable-OID sequence effect
//! classification, synthetic receipt validation, and local typed-default sealing. It deliberately
//! stops before live durable receipt lookup, session/result state, and every physical boundary.

mod classification;
mod receipt;
mod terminal;

use crate::{
    command_is_index_lifecycle, command_is_sequence_lifecycle, command_is_view_lifecycle,
    transaction_statement_digest, valid_index_lifecycle_operation_identity,
    valid_sequence_lifecycle_operation_identity, valid_sequence_reset_operation_identity,
    valid_view_lifecycle_operation_identity, BinaryCatalogRelationIdentity,
    BinaryCatalogRelationKind, BinarySequenceColumnDependencyIdentity,
    BinarySequenceValueReference, BinaryTransactionSequenceLifecycleOperationIdentity,
    BinaryTransactionSequenceLifecycleTargetIdentity,
    BinaryTransactionSequenceResetOperationIdentity, CatalogSnapshot, Engine, EngineError,
    ExecuteError, Index, Insert, PreparedMutation, RelationalResidentShard, TransactionOperation,
    TransactionSnapshot, TxnId,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

/// One exact parent statement identity. Both autocommit and explicit paths derive the digest from
/// the sealed pre-effect typed intent; they differ only in their transaction context.
#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
struct InsertEffectParentIdentity {
    txn_id: TxnId,
    autocommit: bool,
    request_digest: gpu_db_wal::CanonicalDigest,
    statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal,
    expression_ordinal_base: u32,
}

impl InsertEffectParentIdentity {
    fn from_prepared(
        txn_id: TxnId,
        autocommit: bool,
        prepared: &crate::typed_insert_batch::PreparedTypedInsert,
        statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal,
        expression_ordinal_base: u32,
    ) -> Result<Self, ExecuteError> {
        if txn_id == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "INSERT effect parent transaction id is zero".to_string(),
            )));
        }
        Ok(Self {
            txn_id,
            autocommit,
            request_digest: prepared.typed_statement_digest(),
            statement_ordinal,
            expression_ordinal_base,
        })
    }
}

#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
enum InsertEffectBaseline {
    Autocommit(AutocommitEffectBaseline),
    Explicit(ExplicitOverlayBaseline),
}

/// Exact autocommit catalog cut. A scalar sequence boundary and the catalog Arc must both remain
/// current; matching commit numbers alone are not an interchangeable catalog witness.
#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
struct AutocommitEffectBaseline {
    catalog: Arc<CatalogSnapshot>,
    catalog_seq: Index,
    boundary: Index,
}

/// One sorted stable-OID state observation. `None` proves that this effect-plan request had no
/// transaction-private sequence seed at capture; a later insertion is equally incompatible.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TouchedSequenceSeed {
    oid: u32,
    effective_name: String,
    oid_state: Option<(i64, bool)>,
    name_state: Option<(i64, bool)>,
}

/// Exact explicit-transaction witness. The retained snapshot keeps the owner tied to the active
/// transaction registration. GPU generation witnesses are deliberately non-owning: a semantic
/// plan must not pin a retired transaction-private allocation after its statement boundary.
#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
struct ExplicitOverlayBaseline {
    snapshot: Arc<TransactionSnapshot>,
    boundary: Index,
    transaction_catalog: Arc<CatalogSnapshot>,
    catalog_base: Option<Arc<CatalogSnapshot>>,
    delta_generation: u64,
    operation_count: u32,
    operation_fingerprint: gpu_db_wal::CanonicalDigest,
    next_row_id: u64,
    sequence_reference_count: u32,
    sequence_reference_fingerprint: gpu_db_wal::CanonicalDigest,
    resident_shards: Weak<BTreeMap<String, Vec<RelationalResidentShard>>>,
    streaming_cold_chunks:
        Weak<BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>>,
    catalog_overlay: Option<Arc<CatalogSnapshot>>,
    touched_sequence_seeds: Box<[TouchedSequenceSeed]>,
}

/// Move-only semantic/effect handoff. It has no physical, result, or state-update authority.
#[allow(dead_code)] // The production consumer is deliberately deferred to the next PLAN slice.
struct PreparedInsertEffectPlan {
    prepared: crate::typed_insert_batch::PreparedTypedInsert,
    parent: InsertEffectParentIdentity,
    sequence_effects: classification::PlannedSequenceEffects,
    baseline: InsertEffectBaseline,
}

impl PreparedInsertEffectPlan {
    /// Capture an exact current autocommit catalog cut before semantic lowering. The caller owns
    /// normal admission; this inert leaf only refuses a catalog Arc or boundary that has moved.
    #[allow(dead_code)] // The production consumer is deliberately deferred to the next PLAN slice.
    fn prepare_autocommit(
        engine: &Engine,
        txn_id: TxnId,
        insert: &Insert,
        catalog: Arc<CatalogSnapshot>,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
    ) -> Result<Option<Self>, ExecuteError> {
        let boundary = engine.committed_seq();
        let current_catalog = engine.catalog_snapshot();
        if !Arc::ptr_eq(&current_catalog, &catalog) || catalog.commit_seq != boundary {
            return Err(baseline_drift("autocommit catalog cut is not current"));
        }
        let statement_ordinal = crate::insert_semantic_ir::InsertStatementOrdinal::FIRST;
        let Some(prepared) = crate::typed_insert_batch::prepare_typed_insert_semantics_at(
            insert,
            &catalog,
            catalog.commit_seq,
            expected_catalog_version,
            statement_ordinal,
        )?
        else {
            return Ok(None);
        };
        let parent = InsertEffectParentIdentity::from_prepared(
            txn_id,
            true,
            &prepared,
            statement_ordinal,
            0,
        )?;
        let sequence_effects = classification::classify_autocommit(&prepared, &parent, &catalog)?;
        let plan = Self {
            prepared,
            parent,
            sequence_effects,
            baseline: InsertEffectBaseline::Autocommit(AutocommitEffectBaseline {
                catalog_seq: catalog.commit_seq,
                catalog,
                boundary,
            }),
        };
        plan.validate_current(engine)?;
        Ok(Some(plan))
    }

    /// Resolve the registered explicit snapshot and hold its own statement lock while capturing
    /// the generation. No external guard is accepted: an unrelated mutex cannot authorize this
    /// transaction's private GPU generation.
    #[allow(dead_code)] // The production consumer is deliberately deferred to the next PLAN slice.
    fn prepare_explicit(
        engine: &Engine,
        txn_id: TxnId,
        insert: &Insert,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
    ) -> Result<Option<Self>, ExecuteError> {
        let snapshot = engine
            .transaction_snapshot_handle(txn_id)
            .ok_or_else(|| baseline_drift("explicit transaction snapshot is not registered"))?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let statement_guard = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        engine.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        Self::prepare_explicit_statement_locked(
            engine,
            txn_id,
            snapshot,
            &statement_guard,
            insert,
            expected_catalog_version,
        )
    }

    /// The only statement-locked explicit capture entry. Its private caller obtains the guard
    /// from `snapshot.statement_lock`, so a borrowed guard cannot be supplied for another owner.
    fn prepare_explicit_statement_locked(
        engine: &Engine,
        txn_id: TxnId,
        snapshot: Arc<TransactionSnapshot>,
        _statement_guard: &std::sync::MutexGuard<'_, ()>,
        insert: &Insert,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
    ) -> Result<Option<Self>, ExecuteError> {
        engine.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let captured = ExplicitCapture::from_statement_locked(&snapshot)?;
        let statement_ordinal =
            crate::insert_semantic_ir::InsertStatementOrdinal::from_u32(captured.operation_count);
        let Some(prepared) = crate::typed_insert_batch::prepare_typed_insert_semantics_at(
            insert,
            &captured.transaction_catalog,
            captured.transaction_catalog.commit_seq,
            expected_catalog_version,
            statement_ordinal,
        )?
        else {
            return Ok(None);
        };
        let parent = InsertEffectParentIdentity::from_prepared(
            txn_id,
            false,
            &prepared,
            statement_ordinal,
            captured.expression_ordinal_base,
        )?;
        let touched_sequence_seeds = touched_sequence_seeds(&prepared, &captured)?;
        let operation_fingerprint = operation_identity_fingerprint(&captured.operations)?;
        let sequence_reference_fingerprint =
            sequence_reference_fingerprint(&captured.sequence_value_references)?;
        let sequence_effects = classification::classify_explicit(&prepared, &parent, &captured)?;
        let plan = Self {
            prepared,
            parent,
            sequence_effects,
            baseline: InsertEffectBaseline::Explicit(ExplicitOverlayBaseline {
                snapshot,
                boundary: captured.boundary,
                transaction_catalog: captured.transaction_catalog,
                catalog_base: captured.catalog_base,
                delta_generation: captured.delta_generation,
                operation_count: captured.operation_count,
                operation_fingerprint,
                next_row_id: captured.next_row_id,
                sequence_reference_count: captured.sequence_reference_count,
                resident_shards: captured.resident_shards,
                streaming_cold_chunks: captured.streaming_cold_chunks,
                catalog_overlay: captured.catalog_overlay,
                touched_sequence_seeds,
                sequence_reference_fingerprint,
            }),
        };
        plan.validate_current_statement_locked(engine, _statement_guard)?;
        Ok(Some(plan))
    }

    /// Recheck the one captured generation without mutating it. Explicit validation first proves
    /// that the retained snapshot is still the registered owner, then compares the complete
    /// baseline's scalar and pointer witnesses under a fresh delta lock.
    #[allow(dead_code)] // The production consumer is deliberately deferred to the next PLAN slice.
    fn validate_current(&self, engine: &Engine) -> Result<(), ExecuteError> {
        match &self.baseline {
            InsertEffectBaseline::Autocommit(baseline) => {
                let current = engine.catalog_snapshot();
                if !Arc::ptr_eq(&current, &baseline.catalog)
                    || current.commit_seq != baseline.catalog_seq
                    || engine.committed_seq() != baseline.boundary
                {
                    return Err(baseline_drift("autocommit catalog generation changed"));
                }
            }
            InsertEffectBaseline::Explicit(baseline) => baseline.validate_current(
                engine,
                self.parent.txn_id,
                self.parent.statement_ordinal,
                self.parent.expression_ordinal_base,
            )?,
        }
        Ok(())
    }

    /// Revalidate an explicit plan while the private caller still owns the exact registered
    /// snapshot statement lock. This avoids a non-reentrant relock between capture and seal.
    fn validate_current_statement_locked(
        &self,
        engine: &Engine,
        statement_guard: &std::sync::MutexGuard<'_, ()>,
    ) -> Result<(), ExecuteError> {
        match &self.baseline {
            InsertEffectBaseline::Explicit(baseline) => baseline.validate_current_statement_locked(
                engine,
                self.parent.txn_id,
                self.parent.statement_ordinal,
                self.parent.expression_ordinal_base,
                statement_guard,
            ),
            InsertEffectBaseline::Autocommit(_) => self.validate_current(engine),
        }
    }
}

struct ExplicitCapture {
    boundary: Index,
    snapshot_catalog: Arc<CatalogSnapshot>,
    transaction_catalog: Arc<CatalogSnapshot>,
    delta_generation: u64,
    operation_count: u32,
    expression_ordinal_base: u32,
    next_row_id: u64,
    sequence_reference_count: u32,
    resident_shards: Weak<BTreeMap<String, Vec<RelationalResidentShard>>>,
    streaming_cold_chunks:
        Weak<BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>>,
    catalog_base: Option<Arc<CatalogSnapshot>>,
    catalog_overlay: Option<Arc<CatalogSnapshot>>,
    operations: Box<[TransactionOperation]>,
    sequence_state_by_oid: BTreeMap<u32, (i64, bool)>,
    sequence_state: BTreeMap<String, (i64, bool)>,
    sequence_value_references: Box<[BinarySequenceValueReference]>,
}

impl ExplicitCapture {
    fn from_statement_locked(snapshot: &TransactionSnapshot) -> Result<Self, ExecuteError> {
        let delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !Arc::ptr_eq(&delta.resident_shards, &delta.resident_shards_authority)
            || !Arc::ptr_eq(
                &delta.streaming_cold_chunks,
                &delta.streaming_cold_chunks_authority,
            )
        {
            return Err(baseline_drift("transaction generation authority is torn"));
        }
        let operation_count = u32::try_from(delta.operations.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction operation count exceeds INSERT effect framing".to_string(),
            )
        })?;
        let expression_ordinal_base =
            statement_sequence_expression_count(&delta.sequence_value_references, operation_count)?;
        let sequence_reference_count = u32::try_from(delta.sequence_value_references.len())
            .map_err(|_| {
                ExecuteError::Unsupported(
                    "transaction sequence reference count exceeds INSERT effect framing"
                        .to_string(),
                )
            })?;
        let catalog_base = delta.catalog_base.clone();
        let catalog_overlay = delta.catalog_overlay.clone();
        let transaction_catalog = catalog_overlay
            .clone()
            .unwrap_or_else(|| Arc::clone(&snapshot.catalog));
        let operations = delta.operations.clone().into_boxed_slice();
        catalog_base_is_valid(&operations, &catalog_base, &snapshot.catalog)?;
        Ok(Self {
            boundary: snapshot.boundary,
            snapshot_catalog: Arc::clone(&snapshot.catalog),
            transaction_catalog,
            delta_generation: delta.generation,
            operation_count,
            expression_ordinal_base,
            next_row_id: delta.next_row_id,
            sequence_reference_count,
            resident_shards: Arc::downgrade(&delta.resident_shards),
            streaming_cold_chunks: Arc::downgrade(&delta.streaming_cold_chunks),
            catalog_base,
            catalog_overlay,
            operations,
            sequence_state_by_oid: delta.sequence_state_by_oid.clone(),
            sequence_state: delta.sequence_state.clone(),
            sequence_value_references: delta.sequence_value_references.clone().into(),
        })
    }
}

impl ExplicitOverlayBaseline {
    fn validate_current(
        &self,
        engine: &Engine,
        txn_id: TxnId,
        statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal,
        expression_ordinal_base: u32,
    ) -> Result<(), ExecuteError> {
        let statement_lock = Arc::clone(&self.snapshot.statement_lock);
        let _statement_guard = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.validate_current_statement_locked(
            engine,
            txn_id,
            statement_ordinal,
            expression_ordinal_base,
            &_statement_guard,
        )
    }

    fn validate_current_statement_locked(
        &self,
        engine: &Engine,
        txn_id: TxnId,
        statement_ordinal: crate::insert_semantic_ir::InsertStatementOrdinal,
        expression_ordinal_base: u32,
        _statement_guard: &std::sync::MutexGuard<'_, ()>,
    ) -> Result<(), ExecuteError> {
        engine.ensure_transaction_snapshot_current(txn_id, &self.snapshot)?;
        if self.snapshot.boundary != self.boundary {
            return Err(baseline_drift("explicit snapshot boundary changed"));
        }
        let delta = self
            .snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if statement_ordinal.as_u32() != self.operation_count
            || statement_sequence_expression_count(
                &delta.sequence_value_references,
                self.operation_count,
            )? != expression_ordinal_base
            || operation_identity_fingerprint(&delta.operations)? != self.operation_fingerprint
            || sequence_reference_fingerprint(&delta.sequence_value_references)?
                != self.sequence_reference_fingerprint
        {
            return Err(baseline_drift(
                "explicit transaction statement identity changed",
            ));
        }
        if !Arc::ptr_eq(&delta.resident_shards, &delta.resident_shards_authority)
            || !Arc::ptr_eq(
                &delta.streaming_cold_chunks,
                &delta.streaming_cold_chunks_authority,
            )
            || !self
                .resident_shards
                .upgrade()
                .is_some_and(|captured| Arc::ptr_eq(&delta.resident_shards, &captured))
            || !self
                .streaming_cold_chunks
                .upgrade()
                .is_some_and(|captured| Arc::ptr_eq(&delta.streaming_cold_chunks, &captured))
            || delta.generation != self.delta_generation
            || usize::try_from(self.operation_count).ok() != Some(delta.operations.len())
            || delta.next_row_id != self.next_row_id
            || usize::try_from(self.sequence_reference_count).ok()
                != Some(delta.sequence_value_references.len())
            || !option_arc_ptr_eq(&delta.catalog_base, &self.catalog_base)
            || !option_arc_ptr_eq(&delta.catalog_overlay, &self.catalog_overlay)
        {
            return Err(baseline_drift("explicit transaction generation changed"));
        }
        catalog_base_is_valid(
            &delta.operations,
            &delta.catalog_base,
            &self.snapshot.catalog,
        )?;
        let transaction_catalog = delta
            .catalog_overlay
            .clone()
            .unwrap_or_else(|| Arc::clone(&self.snapshot.catalog));
        if !Arc::ptr_eq(&transaction_catalog, &self.transaction_catalog)
            || self.touched_sequence_seeds.iter().any(|seed| {
                delta.sequence_state_by_oid.get(&seed.oid).copied() != seed.oid_state
                    || delta.sequence_state.get(&seed.effective_name).copied() != seed.name_state
            })
        {
            return Err(baseline_drift(
                "explicit transaction sequence or catalog state changed",
            ));
        }
        Ok(())
    }
}

fn option_arc_ptr_eq<T>(left: &Option<Arc<T>>, right: &Option<Arc<T>>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, Some(_)) | (Some(_), None) => false,
    }
}

fn catalog_base_is_valid(
    operations: &[TransactionOperation],
    catalog_base: &Option<Arc<CatalogSnapshot>>,
    snapshot_catalog: &Arc<CatalogSnapshot>,
) -> Result<(), ExecuteError> {
    let requires_catalog_base = operations.iter().any(|operation| {
        matches!(operation, TransactionOperation::Catalog(_))
            || matches!(
                operation,
                TransactionOperation::TableReset(reset)
                    if reset.sequence_reset_identity.is_some()
            )
    });
    match (requires_catalog_base, catalog_base) {
        (true, Some(base)) if Arc::ptr_eq(base, snapshot_catalog) => Ok(()),
        (true, Some(_)) => Err(baseline_drift(
            "catalog-bearing transaction stream changed its base generation",
        )),
        (true, None) => Err(baseline_drift(
            "catalog-bearing transaction stream has no base generation",
        )),
        (false, None) => Ok(()),
        (false, Some(_)) => Err(baseline_drift(
            "row-only transaction stream unexpectedly retained a catalog base",
        )),
    }
}

fn touched_sequence_seeds(
    prepared: &crate::typed_insert_batch::PreparedTypedInsert,
    captured: &ExplicitCapture,
) -> Result<Box<[TouchedSequenceSeed]>, ExecuteError> {
    let mut names_by_oid = BTreeMap::new();
    for request in prepared.effect_sequence_requests() {
        let effective_name = request.sequence_effective_name();
        let sequence = captured
            .transaction_catalog
            .relational_sequences
            .get(effective_name)
            .ok_or_else(|| {
                baseline_drift("prepared sequence target left the captured transaction catalog")
            })?;
        if sequence.oid != request.sequence_oid() {
            return Err(baseline_drift(
                "prepared sequence target stable identity changed in the captured catalog",
            ));
        }
        if names_by_oid
            .insert(request.sequence_oid(), effective_name.to_string())
            .is_some_and(|prior| prior != effective_name)
        {
            return Err(baseline_drift(
                "prepared sequence stable identity has conflicting effective names",
            ));
        }
    }
    Ok(names_by_oid
        .into_iter()
        .map(|(oid, effective_name)| TouchedSequenceSeed {
            oid,
            oid_state: captured.sequence_state_by_oid.get(&oid).copied(),
            name_state: captured.sequence_state.get(&effective_name).copied(),
            effective_name,
        })
        .collect())
}

fn operation_identity_fingerprint(
    operations: &[TransactionOperation],
) -> Result<gpu_db_wal::CanonicalDigest, ExecuteError> {
    let count = u32::try_from(operations.len()).map_err(|_| {
        ExecuteError::Unsupported(
            "transaction operation count exceeds INSERT effect witness framing".to_string(),
        )
    })?;
    let mut body = Vec::with_capacity(32 + operations.len().saturating_mul(40));
    body.extend_from_slice(b"GPUDBPREWALEFFECTOPERATIONS");
    body.push(1);
    body.extend_from_slice(&count.to_le_bytes());
    let mut catalog_index = 0u32;
    for (ordinal, operation) in operations.iter().enumerate() {
        let ordinal = u32::try_from(ordinal).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction operation ordinal exceeds INSERT effect witness framing".to_string(),
            )
        })?;
        body.extend_from_slice(&ordinal.to_le_bytes());
        match operation {
            TransactionOperation::Catalog(staged) => {
                if staged.ordinal != ordinal
                    || transaction_statement_digest(&staged.command)? != staged.statement_digest
                    || staged.view_identity.as_ref().is_some_and(|identity| {
                        !valid_view_lifecycle_operation_identity(&staged.command, identity)
                    })
                    || staged.index_identity.as_ref().is_some_and(|identity| {
                        !valid_index_lifecycle_operation_identity(&staged.command, identity)
                    })
                    || staged.sequence_identity.as_ref().is_some_and(|identity| {
                        !valid_sequence_lifecycle_operation_identity(&staged.command, identity)
                    })
                    || command_is_view_lifecycle(&staged.command) != staged.view_identity.is_some()
                    || command_is_index_lifecycle(&staged.command)
                        != staged.index_identity.is_some()
                    || command_is_sequence_lifecycle(&staged.command)
                        != staged.sequence_identity.is_some()
                {
                    return Err(baseline_drift(
                        "catalog operation lost its exact statement identity",
                    ));
                }
                body.push(1);
                body.extend_from_slice(&catalog_index.to_le_bytes());
                body.extend_from_slice(&staged.statement_digest);
                body.push(u8::from(staged.index_epoch_transition));
                append_sequence_input_oids(&mut body, &staged.sequence_input_oids)?;
                append_optional_sequence_lifecycle_identity(
                    &mut body,
                    staged.sequence_identity.as_ref(),
                )?;
                catalog_index = catalog_index.checked_add(1).ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "catalog operation count exceeds INSERT effect witness framing".to_string(),
                    )
                })?;
            }
            TransactionOperation::Row(staged) => {
                let (family, table) = match &staged.mutation {
                    PreparedMutation::Insert { table, .. } => (2, table),
                    PreparedMutation::Update { table, .. } => (3, table),
                    PreparedMutation::Delete { table, .. } => (4, table),
                };
                body.push(family);
                push_witness_string(&mut body, table)?;
                body.extend_from_slice(&staged.statement_digest);
                append_sequence_input_oids(&mut body, &staged.sequence_input_oids)?;
                match &staged.mutation {
                    PreparedMutation::Insert { seq_advances, .. } => {
                        append_sequence_advances(&mut body, seq_advances)?;
                    }
                    PreparedMutation::Update { .. } | PreparedMutation::Delete { .. } => {
                        append_sequence_advances(&mut body, &BTreeMap::new())?;
                    }
                }
            }
            TransactionOperation::TableReset(reset) => {
                if reset.ordinal != ordinal
                    || reset
                        .sequence_reset_identity
                        .as_ref()
                        .is_some_and(|identity| !valid_sequence_reset_operation_identity(identity))
                {
                    return Err(baseline_drift(
                        "table reset lost its exact statement ordinal",
                    ));
                }
                body.push(5);
                push_witness_string(&mut body, &reset.table)?;
                body.extend_from_slice(&reset.table_oid.to_le_bytes());
                body.extend_from_slice(&reset.schema_digest);
                body.extend_from_slice(&reset.statement_digest);
                append_optional_sequence_reset_identity(
                    &mut body,
                    reset.sequence_reset_identity.as_ref(),
                )?;
            }
        }
    }
    Ok(gpu_db_wal::canonical_request_digest(&body))
}

fn sequence_reference_fingerprint(
    references: &[BinarySequenceValueReference],
) -> Result<gpu_db_wal::CanonicalDigest, ExecuteError> {
    let count = u32::try_from(references.len()).map_err(|_| {
        ExecuteError::Unsupported(
            "transaction sequence reference count exceeds INSERT effect witness framing"
                .to_string(),
        )
    })?;
    let mut body = Vec::with_capacity(32 + references.len().saturating_mul(96));
    body.extend_from_slice(b"GPUDBPREWALEFFECTSEQUENCEREFS");
    body.push(1);
    body.extend_from_slice(&count.to_le_bytes());
    for reference in references {
        body.extend_from_slice(&reference.transition_txn_id.to_le_bytes());
        body.extend_from_slice(&reference.parent_txn_id.to_le_bytes());
        body.extend_from_slice(&reference.statement_ordinal.to_le_bytes());
        body.extend_from_slice(&reference.expression_ordinal.to_le_bytes());
        body.extend_from_slice(&reference.sequence_oid.to_le_bytes());
        body.extend_from_slice(&reference.returned_value.to_le_bytes());
        body.extend_from_slice(&reference.input_digest);
        body.extend_from_slice(&reference.table_oid.to_le_bytes());
        body.extend_from_slice(&reference.column_id.to_le_bytes());
        body.extend_from_slice(&reference.staging_row_ordinal.to_le_bytes());
        body.extend_from_slice(&reference.row_id.to_le_bytes());
        body.push(u8::from(reference.final_value_overwritten));
        body.push(u8::from(reference.default_expression));
    }
    Ok(gpu_db_wal::canonical_request_digest(&body))
}

fn push_witness_string(body: &mut Vec<u8>, value: &str) -> Result<(), ExecuteError> {
    let len = u32::try_from(value.len()).map_err(|_| {
        ExecuteError::Unsupported(
            "operation table name exceeds INSERT effect witness framing".to_string(),
        )
    })?;
    body.extend_from_slice(&len.to_le_bytes());
    body.extend_from_slice(value.as_bytes());
    Ok(())
}

fn append_sequence_input_oids(
    body: &mut Vec<u8>,
    sequence_input_oids: &BTreeMap<String, u32>,
) -> Result<(), ExecuteError> {
    append_witness_count(body, sequence_input_oids.len(), "sequence input count")?;
    for (name, oid) in sequence_input_oids {
        push_witness_string(body, name)?;
        body.extend_from_slice(&oid.to_le_bytes());
    }
    Ok(())
}

fn append_sequence_advances(
    body: &mut Vec<u8>,
    sequence_advances: &BTreeMap<String, (i64, bool)>,
) -> Result<(), ExecuteError> {
    append_witness_count(body, sequence_advances.len(), "sequence advancement count")?;
    for (name, (last_value, is_called)) in sequence_advances {
        push_witness_string(body, name)?;
        body.extend_from_slice(&last_value.to_le_bytes());
        body.push(u8::from(*is_called));
    }
    Ok(())
}

fn append_optional_sequence_lifecycle_identity(
    body: &mut Vec<u8>,
    identity: Option<&BinaryTransactionSequenceLifecycleOperationIdentity>,
) -> Result<(), ExecuteError> {
    let Some(identity) = identity else {
        body.push(0);
        return Ok(());
    };
    body.push(1);
    body.extend_from_slice(&identity.command_index.to_le_bytes());
    body.extend_from_slice(&identity.ordinal.to_le_bytes());
    append_sequence_lifecycle_targets(body, &identity.targets)
}

fn append_optional_sequence_reset_identity(
    body: &mut Vec<u8>,
    identity: Option<&BinaryTransactionSequenceResetOperationIdentity>,
) -> Result<(), ExecuteError> {
    let Some(identity) = identity else {
        body.push(0);
        return Ok(());
    };
    body.push(1);
    body.extend_from_slice(&identity.ordinal.to_le_bytes());
    push_witness_string(body, &identity.table)?;
    append_sequence_lifecycle_targets(body, &identity.targets)
}

fn append_sequence_lifecycle_targets(
    body: &mut Vec<u8>,
    targets: &[BinaryTransactionSequenceLifecycleTargetIdentity],
) -> Result<(), ExecuteError> {
    append_witness_count(body, targets.len(), "sequence lifecycle target count")?;
    for target in targets {
        push_witness_string(body, &target.before_name)?;
        append_optional_catalog_relation_identity(body, target.target_before.as_ref());
        append_sequence_dependencies(body, &target.dependencies_before)?;
        append_optional_witness_string(body, target.after_name.as_deref())?;
        append_optional_catalog_relation_identity(body, target.target_after.as_ref());
        append_sequence_dependencies(body, &target.dependencies_after)?;
    }
    Ok(())
}

fn append_sequence_dependencies(
    body: &mut Vec<u8>,
    dependencies: &BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
) -> Result<(), ExecuteError> {
    append_witness_count(
        body,
        dependencies.len(),
        "sequence lifecycle dependency count",
    )?;
    for (name, dependency) in dependencies {
        push_witness_string(body, name)?;
        append_catalog_relation_identity(body, &dependency.table);
        body.extend_from_slice(&dependency.column_id.to_le_bytes());
    }
    Ok(())
}

fn append_optional_catalog_relation_identity(
    body: &mut Vec<u8>,
    identity: Option<&BinaryCatalogRelationIdentity>,
) {
    match identity {
        Some(identity) => {
            body.push(1);
            append_catalog_relation_identity(body, identity);
        }
        None => body.push(0),
    }
}

fn append_catalog_relation_identity(body: &mut Vec<u8>, identity: &BinaryCatalogRelationIdentity) {
    body.push(match identity.kind {
        BinaryCatalogRelationKind::Table => 1,
        BinaryCatalogRelationKind::View => 2,
        BinaryCatalogRelationKind::Sequence => 3,
    });
    body.extend_from_slice(&identity.oid.to_le_bytes());
    body.extend_from_slice(&identity.digest);
}

fn append_optional_witness_string(
    body: &mut Vec<u8>,
    value: Option<&str>,
) -> Result<(), ExecuteError> {
    match value {
        Some(value) => {
            body.push(1);
            push_witness_string(body, value)
        }
        None => {
            body.push(0);
            Ok(())
        }
    }
}

fn append_witness_count(
    body: &mut Vec<u8>,
    count: usize,
    subject: &str,
) -> Result<(), ExecuteError> {
    let count = u32::try_from(count).map_err(|_| {
        ExecuteError::Unsupported(format!("{subject} exceeds INSERT effect witness framing"))
    })?;
    body.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

fn baseline_drift(message: &str) -> ExecuteError {
    ExecuteError::Serialization(format!("INSERT effect baseline drifted: {message}"))
}

fn statement_sequence_expression_count(
    references: &[BinarySequenceValueReference],
    statement_ordinal: u32,
) -> Result<u32, ExecuteError> {
    let mut expression_ordinals = references
        .iter()
        .filter(|reference| reference.statement_ordinal == statement_ordinal)
        .map(|reference| reference.expression_ordinal)
        .collect::<Vec<_>>();
    let count = u32::try_from(expression_ordinals.len()).map_err(|_| {
        ExecuteError::Unsupported(
            "transaction sequence reference count exceeds INSERT effect framing".to_string(),
        )
    })?;
    expression_ordinals.sort_unstable();
    if expression_ordinals
        .iter()
        .enumerate()
        .any(|(expected, actual)| u32::try_from(expected).ok() != Some(*actual))
    {
        return Err(baseline_drift(
            "transaction sequence references are not contiguous at statement ordinal",
        ));
    }
    Ok(count)
}

#[cfg(test)]
impl PreparedInsertEffectPlan {
    fn prepare_autocommit_for_test(
        engine: &Engine,
        txn_id: TxnId,
        insert: &Insert,
    ) -> Result<Self, ExecuteError> {
        Self::prepare_autocommit(engine, txn_id, insert, engine.catalog_snapshot(), None)?
            .ok_or_else(|| {
                baseline_drift("test autocommit semantic preparation unexpectedly deferred")
            })
    }

    fn prepare_explicit_for_test(
        engine: &Engine,
        txn_id: TxnId,
        insert: &Insert,
    ) -> Result<Self, ExecuteError> {
        Self::prepare_explicit(engine, txn_id, insert, None)?.ok_or_else(|| {
            baseline_drift("test explicit semantic preparation unexpectedly deferred")
        })
    }

    fn parent_for_test(
        &self,
    ) -> (
        TxnId,
        bool,
        gpu_db_wal::CanonicalDigest,
        crate::insert_semantic_ir::InsertStatementOrdinal,
        u32,
    ) {
        (
            self.parent.txn_id,
            self.parent.autocommit,
            self.parent.request_digest,
            self.parent.statement_ordinal,
            self.parent.expression_ordinal_base,
        )
    }

    fn explicit_baseline_for_test(&self) -> Option<&ExplicitOverlayBaseline> {
        let InsertEffectBaseline::Explicit(baseline) = &self.baseline else {
            return None;
        };
        Some(baseline)
    }

    fn sequence_effects_for_test(&self) -> &classification::PlannedSequenceEffects {
        &self.sequence_effects
    }

    fn published_receipt_input_digests_for_test(&self) -> Vec<gpu_db_wal::CanonicalDigest> {
        classification::published_input_digests_for_test(&self.sequence_effects)
    }

    fn seal_for_test(
        self,
        engine: &Engine,
        receipts: receipt::SequenceReceiptBundle,
    ) -> Result<terminal::InertEffectSealEvidence, ExecuteError> {
        terminal::seal_for_test(self, engine, receipts)
    }

    fn sabotage_parent_for_terminal_test(&mut self) {
        self.parent.request_digest[0] ^= 0x5a;
    }

    fn sabotage_classification_for_terminal_test(
        &mut self,
        sabotage: classification::TerminalPlanSabotage,
    ) {
        classification::sabotage_for_terminal_test(&mut self.sequence_effects, sabotage);
    }
}

#[cfg(test)]
#[path = "pre_wal_effects_tests.rs"]
mod tests;
