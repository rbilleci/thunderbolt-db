//! Move-only INSERT plan carriers between off-lock preparation and the commit gate.
//!
//! `PreparedDeviceInsertPlan` owns the sealed semantic batch and its zero-id WAL template while
//! it is safe to queue. `BoundDeviceInsertPlan` is deliberately lifetime-bound to the commit
//! gate: it owns the residency plan and the exact row-id-patched WAL token only after the caller
//! has revalidated the queued request against the current catalog and write set.

use crate::{CatalogSnapshot, Engine, EngineError, Index, WriteSet};

pub(crate) mod batch_key_constraints;
pub(crate) mod constraint_arbitration;
mod foreign_key_constraints;
pub(crate) mod host_retention;
mod pre_wal_capacity;
pub(crate) mod pre_wal_constraints;
mod pre_wal_effects;
pub(crate) mod pre_wal_footprint;
mod reserved_pre_wal;
#[cfg(test)]
pub(crate) use reserved_pre_wal::limit_next_test_indexed_host_scratch_to;
pub(crate) mod resident_constraint_generation;
pub(crate) mod resident_key_constraints;
pub(crate) mod row_local_constraints;

/// One off-lock, move-only INSERT semantic carrier.
///
/// The vectors remain owned by `batch`; the template is derived once from those sealed vectors.
/// Neither field is `Clone` or `Arc`-shared, so a queued candidate cannot split semantic and WAL
/// authority or be rebound to another request.
pub(crate) struct PreparedDeviceInsertPlan {
    batch: crate::typed_insert_batch::TypedInsertBatch,
    template: crate::wal_binary::PreparedBinaryInsertTemplate,
    domain_dependencies: Box<[crate::typed_insert_batch::TypedInsertDomainBinding]>,
    pre_wal_constraint_proof: pre_wal_constraints::PreWalConstraintProof,
    /// The resident/global UNIQUE proof is deliberately test-only until it owns the production
    /// current-commit/WAL handoff.  The live route continues to reject indexed typed batches in
    /// the builder's unchanged eligibility gate.
    #[cfg(test)]
    resident_key_proof: Option<resident_key_constraints::ResidentKeyConstraintProof>,
    /// Inert immediate-FK proof metadata. This is constructed only by the distinct test builder
    /// capability and has no live commit/WAL/apply path.
    #[cfg(test)]
    foreign_key_proof: Option<foreign_key_constraints::ForeignKeyConstraintProof>,
    #[cfg(test)]
    proof_only_local_candidate: Option<pre_wal_constraints::ConstraintCandidate>,
}

/// Exact proposed row identities derived for one current commit-gate attempt.
///
/// This private-by-fields proof keeps the residency input and WAL patch tied to the same original
/// `ProposedRowIdRange`; callers cannot pair arbitrary device identities with a separately bound
/// envelope.
pub(crate) struct DeviceInsertRowIdProposal {
    proposed_range: crate::wal_binary::ProposedRowIdRange,
    row_ids: Box<[u64]>,
}

/// Capability for the one deliberately inert indexed physical-materialization boundary.
///
/// The fields are private to this module so residency cannot mint a permit merely because the
/// type is visible across the crate.  There is intentionally no production constructor while the
/// indexed physical branches remain unreachable from the live INSERT route.  Test inspection
/// obtains this through [`issue_test_indexed_physical_materialization_permit`] immediately before
/// it crosses the same boundary a future bound carrier will own.
#[must_use]
pub(crate) struct IndexedPhysicalMaterializationPermit {
    _only_engine_insert_plan_may_issue: (),
}

#[cfg(test)]
fn issue_test_indexed_physical_materialization_permit() -> IndexedPhysicalMaterializationPermit {
    IndexedPhysicalMaterializationPermit {
        _only_engine_insert_plan_may_issue: (),
    }
}

/// A commit-gate-bound INSERT plan.
///
/// The residency plan retains the mutation gate for its apply interval; the WAL token contains
/// the same row identities in both payload and operation body. This token cannot cross the
/// off-lock queue because its lifetime is tied to the engine borrow that compiled residency.
pub(crate) struct BoundDeviceInsertPlan<'commit> {
    device_plan: crate::engine_residency::DeviceInsertPlan<'commit>,
    bound_wal: crate::wal_binary::BoundBinaryInsert,
    /// Captured while the serial wave owns the sole commit guard.  This must agree with both the
    /// canonical append and the post-WAL device apply; neither residency nor the bound carrier
    /// owns the guard itself.
    expected_commit_seq: Index,
}

/// Binding remains side-effect-free before canonical WAL. The caller preserves the established
/// retry policy, including the live bootstrap sentinel's retryable refusal.
pub(crate) enum BindDeviceInsertPlanError {
    Device(crate::engine_residency::DeviceInsertPlanPrepareError),
    Template { bound_bootstrap: bool },
}

impl PreparedDeviceInsertPlan {
    /// Compile the one binary template while the typed batch still owns its final semantic
    /// vectors. This is the only constructor, so the two authorities cannot be paired from
    /// independent off-lock values.
    pub(crate) fn from_typed_batch(
        mut batch: crate::typed_insert_batch::TypedInsertBatch,
        engine: &Engine,
        catalog: &CatalogSnapshot,
    ) -> Result<Self, EngineError> {
        #[cfg(test)]
        let (
            pre_wal_constraint_proof,
            resident_key_proof,
            foreign_key_proof,
            proof_only_local_candidate,
        ) = if batch.is_proof_only_indexed_constraints() {
            let prepared = pre_wal_constraints::prepare_before_queue(engine, &batch, catalog)?;
            let resident = resident_key_constraints::compile(engine);
            (prepared.proof, Some(resident), None, prepared.candidate)
        } else if batch.is_foreign_key_proof_only() {
            let prepared = pre_wal_constraints::prepare_before_queue_foreign_key_proof(
                engine, &batch, catalog,
            )?;
            let foreign = foreign_key_constraints::ForeignKeyConstraintProof::compile(
                engine, &batch, catalog,
            )?;
            (prepared.proof, None, Some(foreign), prepared.candidate)
        } else {
            (
                pre_wal_constraints::validate_before_queue(engine, &batch, catalog)?,
                None,
                None,
                None,
            )
        };
        #[cfg(not(test))]
        let pre_wal_constraint_proof =
            pre_wal_constraints::validate_before_queue(engine, &batch, catalog)?;
        let template = batch.binary_insert_template()?;
        let domain_dependencies = batch.take_domain_dependencies();
        Ok(Self {
            batch,
            template,
            domain_dependencies,
            pre_wal_constraint_proof,
            #[cfg(test)]
            resident_key_proof,
            #[cfg(test)]
            foreign_key_proof,
            #[cfg(test)]
            proof_only_local_candidate,
        })
    }

    pub(crate) fn table_name(&self) -> &str {
        self.batch.binary_insert_template_table_name()
    }

    pub(crate) fn table_oid(&self) -> u32 {
        self.batch.table_oid()
    }

    pub(crate) fn row_count(&self) -> u32 {
        self.batch.binary_insert_template_row_count()
    }

    /// Test-only current-generation UNIQUE/PK proof.  This remains outside every WAL/apply path:
    /// callers use it to prove the exact state-dependent rejection and SQL precedence before the
    /// product handoff is intentionally widened.
    #[cfg(test)]
    pub(crate) fn validate_current_resident_key_constraints(
        &self,
        engine: &Engine,
    ) -> Result<(), crate::ExecuteError> {
        let Some(proof) = self.resident_key_proof.as_ref() else {
            return Err(crate::ExecuteError::Unsupported(
                "resident INSERT key proof is not enabled for this batch".to_string(),
            ));
        };
        // This compatibility wrapper intentionally borrows the sole pre-WAL proof and discards
        // the returned seal.  The consuming index-delta seam below is the only path permitted to
        // transfer that proof into a later physical preparation.
        let _commit_guard = engine.commit_state();
        let mutation_gate = engine
            .read_state
            .residency
            .mutation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current_catalog = engine.catalog_snapshot();
        let predecessor_boundary = engine.committed_seq();
        resident_key_constraints::validate_current_generation(
            engine,
            &self.batch,
            proof,
            self.pre_wal_constraint_proof.batch_key_proof(),
            self.proof_only_local_candidate.clone(),
            &current_catalog,
            predecessor_boundary,
            &mutation_gate,
        )
        .map(|_| ())
    }

    /// Consume the proof-only indexed carrier into one inert, current-generation inspection.
    /// This deliberately stops before every canonical-operation, durability, device-write, and
    /// descriptor-publication boundary; the closure sees only scalar preparation evidence.
    #[cfg(test)]
    pub(crate) fn inspect_current_resident_index_delta<R>(
        self,
        engine: &Engine,
        proposal: DeviceInsertRowIdProposal,
        inspect: impl FnOnce(crate::engine_residency::index_delta::IndexedInPlaceProofReport) -> R,
    ) -> Result<R, crate::ExecuteError> {
        let commit_guard = engine.commit_state();
        let predecessor_boundary = engine.committed_seq();
        let current_catalog = engine.catalog_snapshot();
        let Self {
            batch,
            template,
            domain_dependencies,
            pre_wal_constraint_proof,
            resident_key_proof,
            foreign_key_proof: _,
            proof_only_local_candidate,
        } = self;
        if template.count() != batch.binary_insert_template_row_count()
            || !batch.domain_dependencies_match(&domain_dependencies, &current_catalog)
        {
            return Err(crate::ExecuteError::Serialization(
                "resident indexed proof semantic carrier drifted".to_string(),
            ));
        }
        let key_proof = pre_wal_constraint_proof
            .into_batch_key_proof_after_current_catalog(&current_catalog)
            .map_err(crate::ExecuteError::Engine)?;
        let resident_proof = resident_key_proof.ok_or_else(|| {
            crate::ExecuteError::Unsupported(
                "resident INSERT key proof is not enabled for this batch".to_string(),
            )
        })?;
        let table = pre_wal_constraints::bound_table_current_generation(&batch, &current_catalog)
            .map_err(crate::ExecuteError::Engine)?
            .clone();
        // The canonical order is commit -> named-index lifecycle -> residency mutation.  The
        // lifecycle guard defers cache retirement for the complete proof interval.
        let named_index_lifecycle = engine
            .read_state
            .residency
            .begin_transaction_named_index_publication(std::collections::BTreeSet::from([table
                .name
                .clone()]));
        let mutation_gate = engine
            .read_state
            .residency
            .mutation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let validation = resident_key_constraints::validate_current_generation(
            engine,
            &batch,
            &resident_proof,
            &key_proof,
            proof_only_local_candidate,
            &current_catalog,
            predecessor_boundary,
            &mutation_gate,
        )?;
        let DeviceInsertRowIdProposal {
            proposed_range,
            row_ids,
        } = proposal;
        if proposed_range.count() != batch.binary_insert_template_row_count()
            || row_ids.len() != batch.binary_insert_template_row_count() as usize
        {
            return Err(crate::ExecuteError::Serialization(
                "resident indexed proof row-id proposal drifted".to_string(),
            ));
        }
        crate::engine_residency::index_delta::inspect_prepared_indexed_in_place(
            engine,
            &table,
            predecessor_boundary,
            batch,
            crate::engine_residency::DeviceInsertRowIds::exact(row_ids),
            key_proof,
            validation,
            mutation_gate,
            named_index_lifecycle,
            &commit_guard,
            issue_test_indexed_physical_materialization_permit(),
            inspect,
        )
    }

    /// Test-only zero-CUDA inspection of the exact indexed in-place witnesses. Unlike the
    /// adjacent reservation seam, this stops before the append compiler and typed-index setup;
    /// it exists to prove the capacity-facing preview cannot allocate, launch, mutate, or publish.
    #[cfg(test)]
    pub(crate) fn inspect_current_resident_index_delta_preview<R>(
        self,
        engine: &Engine,
        proposal: DeviceInsertRowIdProposal,
        inspect: impl FnOnce(crate::engine_residency::index_delta::IndexedInPlacePreviewReport) -> R,
    ) -> Result<R, crate::ExecuteError> {
        let commit_guard = engine.commit_state();
        let predecessor_boundary = engine.committed_seq();
        let current_catalog = engine.catalog_snapshot();
        let Self {
            batch,
            template,
            domain_dependencies,
            pre_wal_constraint_proof,
            resident_key_proof,
            foreign_key_proof: _,
            proof_only_local_candidate,
        } = self;
        if template.count() != batch.binary_insert_template_row_count()
            || !batch.domain_dependencies_match(&domain_dependencies, &current_catalog)
        {
            return Err(crate::ExecuteError::Serialization(
                "resident indexed preview semantic carrier drifted".to_string(),
            ));
        }
        let key_proof = pre_wal_constraint_proof
            .into_batch_key_proof_after_current_catalog(&current_catalog)
            .map_err(crate::ExecuteError::Engine)?;
        let resident_proof = resident_key_proof.ok_or_else(|| {
            crate::ExecuteError::Unsupported(
                "resident INSERT key proof is not enabled for this batch".to_string(),
            )
        })?;
        let table = pre_wal_constraints::bound_table_current_generation(&batch, &current_catalog)
            .map_err(crate::ExecuteError::Engine)?
            .clone();
        let named_index_lifecycle = engine
            .read_state
            .residency
            .begin_transaction_named_index_publication(std::collections::BTreeSet::from([table
                .name
                .clone()]));
        let mutation_gate = engine
            .read_state
            .residency
            .mutation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let validation = resident_key_constraints::validate_current_generation(
            engine,
            &batch,
            &resident_proof,
            &key_proof,
            proof_only_local_candidate,
            &current_catalog,
            predecessor_boundary,
            &mutation_gate,
        )?;
        let DeviceInsertRowIdProposal {
            proposed_range,
            row_ids,
        } = proposal;
        if proposed_range.count() != batch.binary_insert_template_row_count()
            || row_ids.len() != batch.binary_insert_template_row_count() as usize
        {
            return Err(crate::ExecuteError::Serialization(
                "resident indexed preview row-id proposal drifted".to_string(),
            ));
        }
        crate::engine_residency::index_delta::inspect_prepared_indexed_in_place_preview(
            engine,
            &table,
            predecessor_boundary,
            batch,
            crate::engine_residency::DeviceInsertRowIds::exact(row_ids),
            key_proof,
            validation,
            mutation_gate,
            named_index_lifecycle,
            &commit_guard,
            inspect,
        )
    }

    /// Consume the proof-only indexed carrier into one inert fixed-rollover generation. The
    /// rollover payload, sidecars, and every distinct physical index are completed privately
    /// before this returns, while WAL, mutation, cache publication, and descriptor publication
    /// remain unreachable.
    #[cfg(test)]
    pub(crate) fn inspect_current_resident_index_rollover<R>(
        self,
        engine: &Engine,
        proposal: DeviceInsertRowIdProposal,
        inspect: impl FnOnce(
            crate::engine_residency::index_rollover::IndexedFixedRolloverProofReport,
        ) -> R,
    ) -> Result<R, crate::ExecuteError> {
        let commit_guard = engine.commit_state();
        let expected_commit_seq = commit_guard.repl.peek_next_index();
        let predecessor_boundary = engine.committed_seq();
        let current_catalog = engine.catalog_snapshot();
        let Self {
            batch,
            template,
            domain_dependencies,
            pre_wal_constraint_proof,
            resident_key_proof,
            foreign_key_proof: _,
            proof_only_local_candidate,
        } = self;
        if template.count() != batch.binary_insert_template_row_count()
            || !batch.domain_dependencies_match(&domain_dependencies, &current_catalog)
        {
            return Err(crate::ExecuteError::Serialization(
                "resident indexed rollover semantic carrier drifted".to_string(),
            ));
        }
        let key_proof = pre_wal_constraint_proof
            .into_batch_key_proof_after_current_catalog(&current_catalog)
            .map_err(crate::ExecuteError::Engine)?;
        let resident_proof = resident_key_proof.ok_or_else(|| {
            crate::ExecuteError::Unsupported(
                "resident INSERT key proof is not enabled for this batch".to_string(),
            )
        })?;
        let table = pre_wal_constraints::bound_table_current_generation(&batch, &current_catalog)
            .map_err(crate::ExecuteError::Engine)?
            .clone();
        let named_index_lifecycle = engine
            .read_state
            .residency
            .begin_transaction_named_index_publication(std::collections::BTreeSet::from([table
                .name
                .clone()]));
        let mutation_gate = engine
            .read_state
            .residency
            .mutation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let validation = resident_key_constraints::validate_current_generation(
            engine,
            &batch,
            &resident_proof,
            &key_proof,
            proof_only_local_candidate,
            &current_catalog,
            predecessor_boundary,
            &mutation_gate,
        )?;
        let DeviceInsertRowIdProposal {
            proposed_range,
            row_ids,
        } = proposal;
        if proposed_range.count() != batch.binary_insert_template_row_count()
            || row_ids.len() != batch.binary_insert_template_row_count() as usize
        {
            return Err(crate::ExecuteError::Serialization(
                "resident indexed rollover row-id proposal drifted".to_string(),
            ));
        }
        let allocator_before = proposed_range.first();
        let allocator_high_water = proposed_range.allocator_high_water();
        let inserted_rows = proposed_range.count();
        let forecast =
            crate::engine_residency::index_rollover::prepare_indexed_fixed_rollover_forecast(
                engine,
                &table,
                predecessor_boundary,
                batch,
                crate::engine_residency::DeviceInsertRowIds::exact(row_ids),
                key_proof,
                validation,
                mutation_gate,
                named_index_lifecycle,
                &commit_guard,
            )?;
        let physical_binding = forecast.pre_wal_binding();
        reserved_pre_wal::inspect_test_indexed_autocommit(
            physical_binding,
            allocator_before,
            allocator_high_water,
            inserted_rows,
            |permit| forecast.materialize(engine, &table, expected_commit_seq, permit),
            |prepared| prepared.inspect_fixed_rollover(inspect),
        )
    }

    /// Consume the FK-only semantic carrier into an inert current-generation proof.  Table
    /// access is retained before the commit gate, then the established commit -> mutation order
    /// pins one GPU shard-map generation.  No result can reach WAL, row-id allocation, apply, or
    /// resident publication; the callback receives a scalar test report only.
    #[cfg(test)]
    fn inspect_current_resident_foreign_key_constraints<R>(
        self,
        engine: &Engine,
        inspect: impl FnOnce(foreign_key_constraints::ForeignKeyProofReport) -> R,
    ) -> Result<R, crate::ExecuteError> {
        let Self {
            batch,
            template,
            domain_dependencies,
            pre_wal_constraint_proof,
            resident_key_proof: _,
            foreign_key_proof,
            proof_only_local_candidate,
        } = self;
        let foreign_key_proof = foreign_key_proof.ok_or_else(|| {
            crate::ExecuteError::Unsupported(
                "resident INSERT foreign-key proof is not enabled for this batch".to_string(),
            )
        })?;
        // This must occur before taking commit_state: table access owns the complete child/parent
        // dependency closure and follows the engine's access -> commit -> mutation ordering.
        let table_access =
            engine.acquire_autocommit_table_accesses(foreign_key_proof.access_table_names())?;
        let _commit_guard = engine.commit_state();
        let predecessor_boundary = engine.committed_seq();
        let current_catalog = engine.catalog_snapshot();
        if current_catalog.commit_seq != predecessor_boundary
            || template.count() != batch.binary_insert_template_row_count()
            || !batch.domain_dependencies_match(&domain_dependencies, &current_catalog)
            // This inert FK seam alone permits an unrelated monotonic DML catalog advance.  Its
            // target CHECK/unique witness remains exact, while the FK proof below revalidates
            // the child, every parent, and each backing index at the current boundary.
            || !pre_wal_constraint_proof.matches_current_target_binding(&current_catalog)
        {
            return Err(crate::ExecuteError::Serialization(
                "resident foreign-key proof semantic carrier drifted".to_string(),
            ));
        }
        let mutation_gate = engine
            .read_state
            .residency
            .mutation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let seal = foreign_key_constraints::validate_current_generation(
            engine,
            &batch,
            foreign_key_proof,
            proof_only_local_candidate,
            &current_catalog,
            predecessor_boundary,
            &mutation_gate,
            table_access,
        )?;
        Ok(inspect(seal.into_report()))
    }

    /// Revalidate the catalog-order semantic carrier and the template's exact row geometry under
    /// the current commit gate. Request identity and snapshot ownership remain with the queue
    /// item because that is where their allocation witness lives.
    pub(crate) fn matches_current_binding(
        &self,
        expected_write_set: &WriteSet,
        catalog: &CatalogSnapshot,
        prepared_catalog_seq: Index,
    ) -> bool {
        self.batch
            .matches_resident_append_insert(expected_write_set, catalog, prepared_catalog_seq)
            && self
                .batch
                .domain_dependencies_match(&self.domain_dependencies, catalog)
            && self
                .pre_wal_constraint_proof
                .matches_current_catalog(catalog)
            && self.template.count() == self.batch.binary_insert_template_row_count()
    }

    /// Derive the sole row-id proof before residency/WAL binding. It does not mutate the
    /// allocator; allocator consumption remains in the canonical post-WAL apply tail.
    pub(crate) fn prepare_row_id_proposal(
        &self,
        next_row_id: u64,
    ) -> Result<DeviceInsertRowIdProposal, EngineError> {
        let proposed_range =
            crate::wal_binary::ProposedRowIdRange::new(next_row_id, self.row_count())?;
        let row_ids = proposed_range.exact_row_ids()?;
        Ok(DeviceInsertRowIdProposal {
            proposed_range,
            row_ids,
        })
    }

    /// Consume the prepared semantic/WAL pair into one lifetime-bound plan at the current commit
    /// gate. The caller must first perform the queued request/snapshot/catalog/write-set proof
    /// through `matches_current_binding`; this method then makes residency and WAL share the one
    /// opaque `DeviceInsertRowIdProposal`.
    pub(crate) fn bind_for_current_commit<'commit>(
        self,
        engine: &'commit Engine,
        commit_proof: &std::sync::MutexGuard<'_, crate::CommitState>,
        proposal: DeviceInsertRowIdProposal,
    ) -> Result<BoundDeviceInsertPlan<'commit>, BindDeviceInsertPlanError> {
        let expected_commit_seq = commit_proof.repl.peek_next_index();
        let DeviceInsertRowIdProposal {
            proposed_range,
            row_ids,
        } = proposal;
        let device_plan = engine
            .compile_typed_insert_device_plan(
                self.batch,
                crate::engine_residency::DeviceInsertRowIds::exact(row_ids),
            )
            .map_err(BindDeviceInsertPlanError::Device)?;
        let bound_bootstrap = device_plan.is_bound_bootstrap_sentinel();
        let bound_wal = self
            .template
            .bind(proposed_range)
            .map_err(|_| BindDeviceInsertPlanError::Template { bound_bootstrap })?;
        Ok(BoundDeviceInsertPlan {
            device_plan,
            bound_wal,
            expected_commit_seq,
        })
    }
}

impl<'commit> BoundDeviceInsertPlan<'commit> {
    pub(crate) fn expected_commit_seq(&self) -> Index {
        self.expected_commit_seq
    }

    /// Split exactly once at the canonical operation/apply handoff. Both returned authorities
    /// originated in this bound token: canonical WAL consumes `BoundBinaryInsert`, while device
    /// apply consumes the residency plan before the existing group-durability/publication tail.
    pub(crate) fn into_parts(
        self,
    ) -> (
        crate::engine_residency::DeviceInsertPlan<'commit>,
        crate::wal_binary::BoundBinaryInsert,
        Index,
    ) {
        (self.device_plan, self.bound_wal, self.expected_commit_seq)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn insert_plan_spine_keeps_semantic_and_commit_bound_authority_move_only() {
        let source = include_str!("engine_insert_plan.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production insert plan precedes its tests");
        assert!(source.contains("struct PreparedDeviceInsertPlan"));
        assert!(source.contains("batch: crate::typed_insert_batch::TypedInsertBatch"));
        assert!(source.contains("template: crate::wal_binary::PreparedBinaryInsertTemplate"));
        assert!(source.contains(
            "domain_dependencies: Box<[crate::typed_insert_batch::TypedInsertDomainBinding]>"
        ));
        assert!(
            source.contains("pre_wal_constraint_proof: pre_wal_constraints::PreWalConstraintProof")
        );
        assert!(source.contains("struct BoundDeviceInsertPlan<'commit>"));
        assert!(source.contains("device_plan: crate::engine_residency::DeviceInsertPlan<'commit>"));
        assert!(source.contains("bound_wal: crate::wal_binary::BoundBinaryInsert"));
        assert!(source.contains("expected_commit_seq: Index"));
        assert!(source.contains("fn matches_current_binding"));
        assert!(source.contains("fn prepare_row_id_proposal"));
        assert!(source.contains("fn bind_for_current_commit<'commit>"));
        assert!(source.contains("commit_proof: &std::sync::MutexGuard<'_, crate::CommitState>"));
        assert!(source.contains("let expected_commit_seq = commit_proof.repl.peek_next_index();"));
        assert!(source.contains("domain_dependencies_match(&self.domain_dependencies, catalog)"));
        assert!(source.contains("matches_current_catalog(catalog)"));
        assert!(!source.contains("impl Clone for PreparedDeviceInsertPlan"));
        assert!(!source.contains("impl Clone for BoundDeviceInsertPlan"));
        assert!(!source.contains("Arc<crate::typed_insert_batch::TypedInsertBatch>"));
        assert!(!source.contains("Arc<crate::wal_binary::PreparedBinaryInsertTemplate>"));

        let offlock = include_str!("engine_dml_concurrent/state.rs")
            .split("\n#[cfg(test)]\nmod offlock_prepared_tests")
            .next()
            .expect("production off-lock state precedes its tests");
        assert!(
            offlock.contains("prepared_plan: crate::engine_insert_plan::PreparedDeviceInsertPlan")
        );
        assert!(!offlock.contains("struct OfflockTypedInsert {\n    batch:"));
        assert!(!offlock.contains("struct OfflockTypedInsert {\n    template:"));

        let pre_wal = include_str!("engine_dml_concurrent/fixed_insert.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production pre-WAL owner precedes its tests");
        assert!(pre_wal.contains("bound_plan: crate::engine_insert_plan::BoundDeviceInsertPlan"));
        assert!(
            pre_wal.contains("let (plan, bound, expected_commit_seq) = bound_plan.into_parts();")
        );
        assert!(pre_wal.contains("expected_commit_seq: Index"));
        assert!(pre_wal.contains("authority: ClaimedTypedInsertAuthority"));
        assert!(pre_wal.contains("authority.into_apply_authority("));
    }
}
