//! Move-only GPU root-publication proof.
//!
//! The constructors verify device-produced copy-on-write witnesses; the parent serialized
//! commit/apply owner remains the sole caller that may install the private candidate at its
//! visibility cut. This module owns no WAL, catalog mutation, allocator, or replay path.

use super::*;

/// One exact root-snapshot successor prepared by the typed terminal after it has completed the
/// device generation.  The predecessor `Arc` is retained so installation verifies the exact
/// publication object, rather than merely an equal collection of roots.  This is a temporary
/// type-neutral control-plane publication proof; it neither interprets codec-5 bytes nor grants
/// recovery any semantics-v2 authority.
#[must_use = "a prepared typed generation-root publication must be installed with its matching live apply or dropped"]
pub(crate) struct LiveTypedGenerationRootPublication {
    predecessor: Arc<TypedGenerationRootSnapshot>,
    candidate: Arc<TypedGenerationRootSnapshot>,
}

impl LiveTypedGenerationRootPublication {
    /// Whether this exact GPU-completed root candidate introduced `stable_table_id` from an
    /// absent predecessor.  The shared codec-5 finalizer uses this only to enroll a newly
    /// created table's already-applied named indexes in the existing residency publication
    /// lifecycle; it neither interprets the WAL nor selects a physical write path.
    pub(crate) fn introduced_table_from_absent_predecessor(&self, stable_table_id: u64) -> bool {
        self.predecessor.table(stable_table_id).is_none()
            && self.candidate.table(stable_table_id).is_some()
    }

    /// Construct one root-map successor from the exact snapshot and GPU-completed COW path.
    /// CREATE supplies an absent table predecessor (and the uninitialized database only for the
    /// first table); INSERT supplies the exact published table/database predecessors.  The
    /// persistent-map owner validates the full transition before this sole publication holder can
    /// install it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_exact_gpu_completed_table_map_predecessor(
        predecessor: Arc<TypedGenerationRootSnapshot>,
        stable_table_id: u64,
        expected_table_predecessor: Option<TypedTableGenerationRoot>,
        resets_existing_rows: bool,
        expected_database_root: Option<gpu_db_wal::CanonicalDigest>,
        successor: TypedTableGenerationRoot,
        successor_columns: &[TypedColumnGenerationRoot],
        expected_predecessor_index_roots: &[TypedIndexGenerationRoot],
        successor_index_roots: &[TypedIndexGenerationRoot],
        final_database_root: gpu_db_wal::CanonicalDigest,
        gpu_completion: &TypedTableMapGpuCompletion,
    ) -> Result<Self, EngineError> {
        Self::from_exact_gpu_completed_table_map_predecessor_with_created_indexes(
            predecessor,
            stable_table_id,
            expected_table_predecessor,
            resets_existing_rows,
            expected_database_root,
            successor,
            successor_columns,
            expected_predecessor_index_roots,
            successor_index_roots,
            final_database_root,
            gpu_completion,
            &[],
            &[],
        )
    }

    /// S3-authorized companion for a transaction-created index over an existing table.  It
    /// still constructs the same one table-map successor and has no publication method.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_exact_gpu_completed_table_map_predecessor_with_created_indexes(
        predecessor: Arc<TypedGenerationRootSnapshot>,
        stable_table_id: u64,
        expected_table_predecessor: Option<TypedTableGenerationRoot>,
        resets_existing_rows: bool,
        expected_database_root: Option<gpu_db_wal::CanonicalDigest>,
        successor: TypedTableGenerationRoot,
        successor_columns: &[TypedColumnGenerationRoot],
        expected_predecessor_index_roots: &[TypedIndexGenerationRoot],
        successor_index_roots: &[TypedIndexGenerationRoot],
        final_database_root: gpu_db_wal::CanonicalDigest,
        gpu_completion: &TypedTableMapGpuCompletion,
        created_index_ids: &[u64],
        retired_index_ids: &[u64],
    ) -> Result<Self, EngineError> {
        let candidate = predecessor
            .with_gpu_completed_table_map_substitution_with_created_indexes(
                stable_table_id,
                expected_table_predecessor,
                resets_existing_rows,
                expected_database_root,
                successor,
                successor_columns,
                expected_predecessor_index_roots,
                successor_index_roots,
                final_database_root,
                gpu_completion,
                created_index_ids,
                retired_index_ids,
            )?;
        Ok(Self {
            predecessor,
            candidate: Arc::new(candidate),
        })
    }

    /// Extend this still-private successor with one more GPU-completed table substitution. The
    /// original published `Arc` remains the sole installation predecessor, while each CUDA
    /// launch validates the exact candidate produced by the preceding launch. Only the final
    /// candidate can escape through `install_if_current`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn then_exact_gpu_completed_table_map_predecessor(
        self,
        stable_table_id: u64,
        expected_table_predecessor: Option<TypedTableGenerationRoot>,
        resets_existing_rows: bool,
        successor: TypedTableGenerationRoot,
        successor_columns: &[TypedColumnGenerationRoot],
        expected_predecessor_index_roots: &[TypedIndexGenerationRoot],
        successor_index_roots: &[TypedIndexGenerationRoot],
        final_database_root: gpu_db_wal::CanonicalDigest,
        gpu_completion: &TypedTableMapGpuCompletion,
    ) -> Result<Self, EngineError> {
        self.then_exact_gpu_completed_table_map_predecessor_with_created_indexes(
            stable_table_id,
            expected_table_predecessor,
            resets_existing_rows,
            successor,
            successor_columns,
            expected_predecessor_index_roots,
            successor_index_roots,
            final_database_root,
            gpu_completion,
            &[],
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn then_exact_gpu_completed_table_map_predecessor_with_created_indexes(
        mut self,
        stable_table_id: u64,
        expected_table_predecessor: Option<TypedTableGenerationRoot>,
        resets_existing_rows: bool,
        successor: TypedTableGenerationRoot,
        successor_columns: &[TypedColumnGenerationRoot],
        expected_predecessor_index_roots: &[TypedIndexGenerationRoot],
        successor_index_roots: &[TypedIndexGenerationRoot],
        final_database_root: gpu_db_wal::CanonicalDigest,
        gpu_completion: &TypedTableMapGpuCompletion,
        created_index_ids: &[u64],
        retired_index_ids: &[u64],
    ) -> Result<Self, EngineError> {
        let expected_database_root = self.candidate.database_root;
        let candidate = self
            .candidate
            .with_gpu_completed_table_map_substitution_with_created_indexes(
                stable_table_id,
                expected_table_predecessor,
                resets_existing_rows,
                expected_database_root,
                successor,
                successor_columns,
                expected_predecessor_index_roots,
                successor_index_roots,
                final_database_root,
                gpu_completion,
                created_index_ids,
                retired_index_ids,
            )?;
        self.candidate = Arc::new(candidate);
        Ok(self)
    }

    /// Borrow the exact private successor for the next CUDA table-map witness. This clone is not
    /// installable and carries no publication method; ownership remains with this publication.
    pub(crate) fn private_candidate(&self) -> Arc<TypedGenerationRootSnapshot> {
        Arc::clone(&self.candidate)
    }

    /// Construct the sole immutable-root successor for a GPU-completed `CREATE INDEX`
    /// enrollment.  This is a table-map relink, not a second descriptor/cache publication:
    /// the caller provides the exact existing table/index predecessor and the one device-root
    /// successor while `TypedGenerationRootSnapshot` verifies the COW witness.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_exact_gpu_completed_index_root_enrollment(
        predecessor: Arc<TypedGenerationRootSnapshot>,
        stable_table_id: u64,
        expected_table_predecessor: TypedTableGenerationRoot,
        successor_table: TypedTableGenerationRoot,
        expected_database_root: gpu_db_wal::CanonicalDigest,
        expected_predecessor_index_roots: &[TypedIndexGenerationRoot],
        successor_index_root: TypedIndexGenerationRoot,
        final_database_root: gpu_db_wal::CanonicalDigest,
        gpu_completion: &TypedTableMapGpuCompletion,
    ) -> Result<Self, EngineError> {
        let candidate = predecessor.with_gpu_completed_index_root_enrollment(
            stable_table_id,
            expected_table_predecessor,
            successor_table,
            expected_database_root,
            expected_predecessor_index_roots,
            successor_index_root,
            final_database_root,
            gpu_completion,
        )?;
        Ok(Self {
            predecessor,
            candidate: Arc::new(candidate),
        })
    }

    /// Install only while the exact captured predecessor remains current.  The serialized
    /// commit path is the sole writer today; pointer identity closes an ABA/equal-value gap if a
    /// future control-plane producer is introduced without joining that owner.
    pub(super) fn install_if_current(
        self,
        roots: &ArcSwap<TypedGenerationRootSnapshot>,
    ) -> Result<(), EngineError> {
        let current = roots.load_full();
        if !Arc::ptr_eq(&current, &self.predecessor) {
            return Err(EngineError::ApplyFailed(
                "typed generation root publication predecessor drifted before visibility"
                    .to_string(),
            ));
        }
        roots.store(self.candidate);
        Ok(())
    }
}
