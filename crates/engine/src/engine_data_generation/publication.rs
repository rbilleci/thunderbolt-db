//! The sole private builder for immutable logical data-generation manifests.
//!
//! It links completed GPU commitments along checked COW paths. It does not hash relation data,
//! install a live publication, or interact with engine state, WAL, recovery, or readers.

use std::sync::Arc;

use super::digest::{
    CatalogIdentity, CommitSequence, DataGeneration, DatabaseId, DatabaseRoot, IndexEntryLeafRoot,
    IndexGeneration, IndexRoot, PublicationEpoch, RootFormatVersion, RowMapRoot, StableIndexId,
    StableRowId, StableTableId, StatusViewRoot, TableMapRoot, TableRoot, VisibleNext,
};
use super::input::{
    FinalIndexShape, RootFreeGenerationInput, RowAction, RowSetInput, TableBefore, TableDelta,
    TerminalOutcome, TerminalOutcomeKind,
};
use super::manifest::{
    GpuRadixEmptyRoots, GpuRadixPathCompletion, TableManifest, TableMap, TableMapLeaf,
};
use super::status::{GpuStatusPathCompletion, PublishedStatusEntry, PublishedStatusIndex};
use super::DataGenerationError;

#[derive(Clone, Debug)]
pub(super) struct GpuIndexMembershipCompletion {
    pub(super) index_id: StableIndexId,
    pub(super) after_entry_leaf: Option<IndexEntryLeafRoot>,
    /// Present exactly when the membership leaf changes. A zero-effect membership inherits the
    /// predecessor map without requesting any GPU path work.
    pub(super) entry_map_path: Option<GpuRadixPathCompletion<super::digest::IndexMapRoot>>,
}

#[derive(Clone, Debug)]
pub(super) struct GpuRowMutationCompletion {
    pub(super) row_id: StableRowId,
    pub(super) after_current_row_leaf: Option<super::digest::CurrentRowLeafRoot>,
    pub(super) row_map_path: GpuRadixPathCompletion<RowMapRoot>,
    pub(super) index_memberships: Vec<GpuIndexMembershipCompletion>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GpuIndexManifestCompletion {
    pub(super) index_id: StableIndexId,
    pub(super) generation: IndexGeneration,
    pub(super) root: IndexRoot,
}

#[derive(Clone, Debug)]
pub(super) struct GpuTableRowSetCompletion {
    pub(super) table_id: StableTableId,
    pub(super) rows: Vec<GpuRowMutationCompletion>,
    pub(super) final_indexes: Vec<GpuIndexManifestCompletion>,
    pub(super) data_generation: DataGeneration,
    pub(super) table_root: TableRoot,
    pub(super) table_map_path: GpuRadixPathCompletion<TableMapRoot>,
}

#[derive(Clone, Debug)]
pub(super) struct GpuRowSetCompletion {
    pub(super) tables: Vec<GpuTableRowSetCompletion>,
    pub(super) status_path: GpuStatusPathCompletion,
    pub(super) final_database_root: DatabaseRoot,
}

#[derive(Clone, Debug)]
pub(super) struct GenesisGpuCompletion {
    pub(super) empty_table_map_roots: GpuRadixEmptyRoots<TableMapRoot>,
    pub(super) empty_status_map_roots: GpuRadixEmptyRoots<StatusViewRoot>,
    pub(super) database_root: DatabaseRoot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PublicationIdentity {
    pub(super) visible_next: VisibleNext,
    pub(super) database_root: DatabaseRoot,
    pub(super) catalog: CatalogIdentity,
    pub(super) status_covered_through: u64,
    pub(super) status_view_root: StatusViewRoot,
    pub(super) last_terminal_envelope_digest: Option<super::digest::TerminalEnvelopeDigest>,
}

impl PublicationIdentity {
    fn validate(&self) -> Result<(), DataGenerationError> {
        if self.visible_next.get().saturating_sub(1) != self.status_covered_through {
            return Err(DataGenerationError::Invalid("status coverage boundary"));
        }
        if (self.status_covered_through == 0) != self.last_terminal_envelope_digest.is_none() {
            return Err(DataGenerationError::Invalid(
                "last terminal envelope boundary",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct PublicationGeneration {
    pub(super) root_format: RootFormatVersion,
    pub(super) database_id: DatabaseId,
    pub(super) publication_epoch: PublicationEpoch,
    pub(super) identity: PublicationIdentity,
    pub(super) tables: TableMap,
    pub(super) status: PublishedStatusIndex,
}

impl PublicationGeneration {
    fn validate_hot_identity(&self) -> Result<(), DataGenerationError> {
        if self.root_format != RootFormatVersion::V1 {
            return Err(DataGenerationError::UnsupportedRootFormat(
                self.root_format.get(),
            ));
        }
        self.identity.validate()?;
        if self.status.root() != self.identity.status_view_root {
            return Err(DataGenerationError::Invalid("status map root"));
        }
        Ok(())
    }

    /// Recursive validation is for genesis, checkpoint/recovery reconstruction, and test/debug
    /// audit only. A hot RowSet successor validates just its touched paths.
    pub(super) fn validate_full_for_audit(&self) -> Result<(), DataGenerationError> {
        self.validate_hot_identity()?;
        self.tables.validate()?;
        self.tables.validate_manifest_closure()?;
        self.status.validate()
    }
}

/// A private sequencer payload. The coordinator will later own installing it after durable apply;
/// this type cannot install anything itself.
#[derive(Clone, Debug)]
pub(super) struct ReadyPublicationCandidate {
    pub(super) terminal_index: CommitSequence,
    pub(super) predecessor: PublicationIdentity,
    pub(super) predecessor_publication_epoch: PublicationEpoch,
    pub(super) predecessor_generation: Arc<PublicationGeneration>,
    pub(super) generation: Arc<PublicationGeneration>,
    pub(super) terminal_outcome: TerminalOutcome,
    pub(super) terminal_envelope_digest: super::digest::TerminalEnvelopeDigest,
}

impl ReadyPublicationCandidate {
    fn validate(&self) -> Result<(), DataGenerationError> {
        if self.terminal_index.get() != self.predecessor.visible_next.get() {
            return Err(DataGenerationError::PredecessorMismatch(
                "terminal sequence",
            ));
        }
        if self.predecessor != self.predecessor_generation.identity
            || self.predecessor_publication_epoch != self.predecessor_generation.publication_epoch
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "candidate retained predecessor",
            ));
        }
        let expected_visible_next = self
            .terminal_index
            .get()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if self.generation.identity.visible_next.get() != expected_visible_next
            || self.generation.identity.status_covered_through != self.terminal_index.get()
        {
            return Err(DataGenerationError::Invalid(
                "candidate visibility boundary",
            ));
        }
        if self.generation.identity.last_terminal_envelope_digest
            != Some(self.terminal_envelope_digest)
        {
            return Err(DataGenerationError::Invalid("candidate terminal envelope"));
        }
        self.terminal_outcome.validate()?;
        self.generation.validate_hot_identity()
    }
}

/// This builder has no raw-digest or host-hash API. Its inputs are typed roots that can originate
/// only from the future GPU completion adapter (or cfg(test) synthetic completions).
#[derive(Clone, Debug)]
pub(super) struct DataGenerationBuilder {
    root_format: RootFormatVersion,
    database_id: DatabaseId,
}

impl DataGenerationBuilder {
    pub(super) fn new(
        root_format: RootFormatVersion,
        database_id: DatabaseId,
    ) -> Result<Self, DataGenerationError> {
        if root_format != RootFormatVersion::V1 {
            return Err(DataGenerationError::UnsupportedRootFormat(
                root_format.get(),
            ));
        }
        Ok(Self {
            root_format,
            database_id,
        })
    }

    pub(super) fn genesis(
        &self,
        catalog: CatalogIdentity,
        gpu: GenesisGpuCompletion,
    ) -> Result<Arc<PublicationGeneration>, DataGenerationError> {
        let tables = TableMap::empty(gpu.empty_table_map_roots)?;
        let status = PublishedStatusIndex::genesis(self.database_id, gpu.empty_status_map_roots)?;
        let identity = PublicationIdentity {
            visible_next: VisibleNext::new(1)?,
            database_root: gpu.database_root,
            catalog,
            status_covered_through: 0,
            status_view_root: status.root(),
            last_terminal_envelope_digest: None,
        };
        let generation = Arc::new(PublicationGeneration {
            root_format: self.root_format,
            database_id: self.database_id,
            publication_epoch: PublicationEpoch::new(1)?,
            identity,
            tables,
            status,
        });
        generation.validate_full_for_audit()?;
        Ok(generation)
    }

    /// Build a private successful RowSet successor. All changed leaf/path/manifest/database roots
    /// are already completed GPU values; this method only verifies and relinks their bounded COW
    /// structure.
    pub(super) fn build_row_set(
        &self,
        predecessor: &Arc<PublicationGeneration>,
        input: &RootFreeGenerationInput,
        gpu: GpuRowSetCompletion,
    ) -> Result<ReadyPublicationCandidate, DataGenerationError> {
        predecessor.validate_hot_identity()?;
        input.validate()?;
        if input.database_id != self.database_id
            || predecessor.database_id != self.database_id
            || input.initial_database_root != predecessor.identity.database_root
            || input.catalog_before != predecessor.identity.catalog
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "generation identity",
            ));
        }
        if input.commit_sequence.get() != predecessor.identity.visible_next.get() {
            return Err(DataGenerationError::PredecessorMismatch("commit sequence"));
        }
        if input.terminal_outcome.kind != TerminalOutcomeKind::CommitSuccess {
            return Err(DataGenerationError::Unexpected(
                "non-success RowSet candidate",
            ));
        }
        if input.table_deltas.is_empty() {
            return Err(DataGenerationError::Unexpected("empty RowSet candidate"));
        }
        if gpu.tables.len() != input.table_deltas.len() {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "table completion coverage",
            ));
        }

        let mut tables = predecessor.tables.clone();
        for (delta, completion) in input.table_deltas.iter().zip(&gpu.tables) {
            let TableDelta::RowSet(row_set) = delta else {
                return Err(DataGenerationError::Unexpected("non-RowSet delta"));
            };
            if completion.table_id != row_set.table_id {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "table completion identity",
                ));
            }
            let before = tables
                .table(row_set.table_id)
                .cloned()
                .ok_or(DataGenerationError::Missing("RowSet predecessor table"))?;
            validate_row_set_predecessor(row_set, &before)?;
            let after = build_table_row_set(row_set, &before, completion, input.commit_sequence)?;
            let expected_leaf = TableMapLeaf {
                manifest: Arc::clone(&before),
            };
            let previous_table_map_root = tables.root();
            tables = tables.substitute(
                row_set.table_id,
                Some(&expected_leaf),
                Some(TableMapLeaf {
                    manifest: Arc::new(after),
                }),
                &completion.table_map_path,
            )?;
            if tables.root() == previous_table_map_root {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "changed table-map root",
                ));
            }
        }
        if tables.root() == predecessor.tables.root()
            || gpu.final_database_root == predecessor.identity.database_root
        {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "changed database root",
            ));
        }

        let status = predecessor.status.insert(
            PublishedStatusEntry {
                transaction_id: input.transaction_id,
                request_digest: input.status.request_digest,
                commit_sequence: input.commit_sequence,
                outcome: input.terminal_outcome.clone(),
                target_digest: input.status.target_digest,
                returning_digest: input.status.returning_digest,
                terminal_envelope_digest: input.status.terminal_envelope_digest,
            },
            &gpu.status_path,
        )?;
        let expected_status_count = predecessor
            .status
            .count()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if status.count() != expected_status_count || status.root() == predecessor.status.root() {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "changed status-map root",
            ));
        }
        let next_visible = input
            .commit_sequence
            .get()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        let identity = PublicationIdentity {
            visible_next: VisibleNext::new(next_visible)?,
            database_root: gpu.final_database_root,
            catalog: input.catalog_after,
            status_covered_through: input.commit_sequence.get(),
            status_view_root: status.root(),
            last_terminal_envelope_digest: Some(input.status.terminal_envelope_digest),
        };
        let epoch = predecessor
            .publication_epoch
            .get()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        let generation = Arc::new(PublicationGeneration {
            root_format: self.root_format,
            database_id: self.database_id,
            publication_epoch: PublicationEpoch::new(epoch)?,
            identity,
            tables,
            status,
        });
        let candidate = ReadyPublicationCandidate {
            terminal_index: input.commit_sequence,
            predecessor: predecessor.identity.clone(),
            predecessor_publication_epoch: predecessor.publication_epoch,
            predecessor_generation: Arc::clone(predecessor),
            generation,
            terminal_outcome: input.terminal_outcome.clone(),
            terminal_envelope_digest: input.status.terminal_envelope_digest,
        };
        candidate.validate()?;
        Ok(candidate)
    }
}

fn validate_row_set_predecessor(
    input: &RowSetInput,
    predecessor: &TableManifest,
) -> Result<(), DataGenerationError> {
    let TableBefore {
        data_generation,
        root,
        logical_row_count,
    } = input.before;
    if data_generation != predecessor.data_generation
        || root != predecessor.root
        || logical_row_count != predecessor.rows.count()
    {
        return Err(DataGenerationError::PredecessorMismatch(
            "RowSet table manifest",
        ));
    }
    if input.final_indexes.len() != predecessor.indexes.len() {
        return Err(DataGenerationError::PredecessorMismatch(
            "RowSet index count",
        ));
    }
    for (input_index, predecessor_index) in input.final_indexes.iter().zip(&predecessor.indexes) {
        let Some(before) = input_index.before else {
            return Err(DataGenerationError::PredecessorMismatch(
                "RowSet index before",
            ));
        };
        if input_index.index_id != predecessor_index.index_id
            || input_index.shape_root != predecessor_index.shape_root
            || before.generation != predecessor_index.generation
            || before.root != predecessor_index.root
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "RowSet index manifest",
            ));
        }
    }
    Ok(())
}

fn build_table_row_set(
    input: &RowSetInput,
    predecessor: &Arc<TableManifest>,
    completion: &GpuTableRowSetCompletion,
    commit_sequence: CommitSequence,
) -> Result<TableManifest, DataGenerationError> {
    if completion.rows.len() != input.rows.len()
        || completion.final_indexes.len() != predecessor.indexes.len()
    {
        return Err(DataGenerationError::GpuCompletionMismatch(
            "RowSet table coverage",
        ));
    }
    let mut rows = predecessor.rows.clone();
    let mut indexes = predecessor.indexes.clone();
    for (row_input, row_completion) in input.rows.iter().zip(&completion.rows) {
        if row_completion.row_id != row_input.row_id
            || row_completion.after_current_row_leaf.is_some() != row_input.after.is_some()
            || row_completion.index_memberships.len() != row_input.index_memberships.len()
        {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "RowSet row completion",
            ));
        }
        if row_input.before_current_row_leaf.is_some()
            && row_input.before_current_row_leaf == row_completion.after_current_row_leaf
        {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "replacement row leaf",
            ));
        }
        let before_row_map_root = rows.root();
        rows = rows.substitute(
            row_input.row_id,
            row_input.before_current_row_leaf.as_ref(),
            row_completion.after_current_row_leaf,
            &row_completion.row_map_path,
        )?;
        if rows.root() == before_row_map_root {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "changed row-map root",
            ));
        }
        for (membership, membership_completion) in row_input
            .index_memberships
            .iter()
            .zip(&row_completion.index_memberships)
        {
            if membership_completion.index_id != membership.index_id
                || membership_completion.after_entry_leaf.is_some() != membership.after_present
            {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "index membership completion",
                ));
            }
            // An index entry leaf includes the row's `created_by` commitment. A Replace creates
            // a new current row at this commit, so retaining a present predecessor entry would
            // replay the old creation commitment. RebuildCurrent intentionally has distinct
            // semantics and is not covered by this rule.
            if row_input.action == RowAction::Replace
                && membership.before_entry_leaf.is_some()
                && membership.before_entry_leaf == membership_completion.after_entry_leaf
            {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "replayed replacement index membership leaf",
                ));
            }
            let position = indexes
                .binary_search_by_key(&membership.index_id, |index| index.index_id)
                .map_err(|_| DataGenerationError::Missing("RowSet final index"))?;
            if indexes[position].entries.get(row_input.row_id)
                != membership.before_entry_leaf.as_ref()
            {
                return Err(DataGenerationError::PredecessorMismatch(
                    "RowSet index membership leaf",
                ));
            }
            let before_entry_map_root = indexes[position].entries.root();
            let membership_changed =
                membership.before_entry_leaf != membership_completion.after_entry_leaf;
            match (
                membership_changed,
                membership_completion.entry_map_path.as_ref(),
            ) {
                (true, Some(path)) => {
                    let mut index = indexes[position].as_ref().clone();
                    index.entries = index.entries.substitute(
                        row_input.row_id,
                        membership.before_entry_leaf.as_ref(),
                        membership_completion.after_entry_leaf,
                        path,
                    )?;
                    if index.entries.root() == before_entry_map_root {
                        return Err(DataGenerationError::GpuCompletionMismatch(
                            "changed index-map root",
                        ));
                    }
                    indexes[position] = Arc::new(index);
                }
                (true, None) => {
                    return Err(DataGenerationError::GpuCompletionMismatch(
                        "changed index membership path",
                    ));
                }
                (false, Some(_)) => {
                    return Err(DataGenerationError::GpuCompletionMismatch(
                        "zero-effect index membership path",
                    ));
                }
                (false, None) => {}
            }
        }
    }
    for (position, final_completion) in completion.final_indexes.iter().enumerate() {
        let index = indexes[position].as_ref();
        if final_completion.index_id != index.index_id {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "final index identity",
            ));
        }
        let changed = index.entries.root()
            != predecessor
                .index(index.index_id)
                .expect("validated predecessor index membership")
                .entries
                .root();
        if changed {
            if final_completion.generation.get() != commit_sequence.get()
                || final_completion.root
                    == predecessor
                        .index(index.index_id)
                        .expect("validated predecessor index membership")
                        .root
            {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "changed index manifest",
                ));
            }
            let index = Arc::make_mut(&mut indexes[position]);
            index.generation = final_completion.generation;
            index.root = final_completion.root;
        } else if final_completion.generation != index.generation
            || final_completion.root != index.root
        {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "unchanged index manifest",
            ));
        }
    }
    let changed = rows.root() != predecessor.rows.root()
        || indexes
            .iter()
            .zip(&predecessor.indexes)
            .any(|(after, before)| after.root != before.root);
    if changed {
        if completion.data_generation.get() != commit_sequence.get()
            || completion.table_root == predecessor.root
        {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "changed table manifest",
            ));
        }
    } else if completion.data_generation != predecessor.data_generation
        || completion.table_root != predecessor.root
    {
        return Err(DataGenerationError::GpuCompletionMismatch(
            "unchanged table manifest",
        ));
    }
    let result = TableManifest {
        table_id: input.table_id,
        data_generation: completion.data_generation,
        rows,
        indexes,
        root: completion.table_root,
    };
    Ok(result)
}

#[allow(dead_code)]
fn _final_index_shape_is_root_free(_: FinalIndexShape) {}
