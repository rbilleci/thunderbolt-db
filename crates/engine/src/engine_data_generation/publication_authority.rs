//! Private, inert ownership for one complete immutable publication tuple.
//!
//! This module deliberately does not construct logical roots, install a production generation,
//! or attach to the runtime state. It only keeps a fully formed logical generation, its checked
//! catalog witness, authenticated terminal-status view, and exact physical-resource owner under
//! one atomically acquired `Arc`.

use std::sync::Arc;

use arc_swap::ArcSwap;

use super::digest::{CatalogIdentity, CommitSequence, PublicationEpoch, RootFormatVersion};
use super::input::TerminalOutcomeKind;
use super::publication::{PublicationGeneration, PublicationIdentity};
use super::status::{PublishedStatusEntry, PublishedStatusIndex};
use super::DataGenerationError;

/// An opaque, immutable catalog snapshot witness. The future catalog authority must create this
/// only after checking its complete snapshot against the already-canonical catalog identity.
/// This kernel intentionally has no such production constructor yet.
#[derive(Debug)]
pub(super) struct CheckedCatalogSnapshot {
    identity: CatalogIdentity,
    sealed_snapshot: Arc<SealedCatalogSnapshot>,
}

#[derive(Debug)]
struct SealedCatalogSnapshot {
    identity: CatalogIdentity,
}

impl CheckedCatalogSnapshot {
    fn validate_for(&self, identity: CatalogIdentity) -> Result<(), DataGenerationError> {
        if self.identity != identity || self.sealed_snapshot.identity != identity {
            return Err(DataGenerationError::PredecessorMismatch(
                "publication catalog snapshot",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn synthetic_for_test(identity: CatalogIdentity) -> Arc<Self> {
        Arc::new(Self {
            identity,
            sealed_snapshot: Arc::new(SealedCatalogSnapshot { identity }),
        })
    }
}

/// A retained immutable status lookup, authenticated by the logical generation's status root.
/// The durable terminal stream remains outside this inert owner; this only pins its covered view.
#[derive(Debug)]
pub(super) struct AuthenticatedStatusView {
    root: super::digest::StatusViewRoot,
    entry_count: u64,
    index: Arc<PublishedStatusIndex>,
}

impl AuthenticatedStatusView {
    fn from_logical(logical: &PublicationGeneration) -> Result<Arc<Self>, DataGenerationError> {
        logical.status.validate()?;
        let root = logical.status.root();
        if root != logical.identity.status_view_root {
            return Err(DataGenerationError::Invalid("publication status root"));
        }
        Ok(Arc::new(Self {
            root,
            entry_count: logical.status.count(),
            index: Arc::new(logical.status.clone()),
        }))
    }

    fn validate_for(&self, logical: &PublicationGeneration) -> Result<(), DataGenerationError> {
        self.index.validate()?;
        if self.root != logical.identity.status_view_root
            || self.entry_count != logical.status.count()
            || self.index.root() != self.root
            || self.index.count() != self.entry_count
        {
            return Err(DataGenerationError::Invalid("publication status view"));
        }
        Ok(())
    }

    fn contains_exact(&self, entry: &PublishedStatusEntry) -> bool {
        self.index.entry(entry.transaction_id) == Some(entry)
    }
}

/// Binding that closes physical-resource ownership over one exact complete publication tuple.
/// Resource placement remains outside logical roots, but an owner may never be attached to a
/// different logical identity or publication epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PublicationResourceBinding {
    root_format: RootFormatVersion,
    database_id: super::digest::DatabaseId,
    identity: PublicationIdentity,
    publication_epoch: PublicationEpoch,
}

impl PublicationResourceBinding {
    fn from_logical(logical: &PublicationGeneration) -> Self {
        Self {
            root_format: logical.root_format,
            database_id: logical.database_id,
            identity: logical.identity.clone(),
            publication_epoch: logical.publication_epoch,
        }
    }

    fn placement_replacement_from(
        predecessor: &PublicationGeneration,
    ) -> Result<Self, DataGenerationError> {
        let next_epoch = predecessor
            .publication_epoch
            .get()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        Ok(Self {
            root_format: predecessor.root_format,
            database_id: predecessor.database_id,
            identity: predecessor.identity.clone(),
            publication_epoch: PublicationEpoch::new(next_epoch)?,
        })
    }
}

/// Exact retained resource owner. Its payload is purposefully opaque here: no allocation,
/// resource conversion, or bootstrap handoff is exposed by this first authority kernel.
#[derive(Debug)]
pub(super) struct SealedPublicationResources {
    binding: PublicationResourceBinding,
    owner: Arc<SealedPhysicalResourceOwner>,
}

#[derive(Debug)]
struct SealedPhysicalResourceOwner {
    binding: PublicationResourceBinding,
    #[cfg(test)]
    lifetime: Option<TestResourceLifetime>,
}

impl SealedPublicationResources {
    fn validate_for(&self, logical: &PublicationGeneration) -> Result<(), DataGenerationError> {
        let expected = PublicationResourceBinding::from_logical(logical);
        if self.binding != expected || self.owner.binding != expected {
            return Err(DataGenerationError::PredecessorMismatch(
                "publication resource binding",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn synthetic_for_test(
        logical: &PublicationGeneration,
        lifetime: Option<TestResourceLifetime>,
    ) -> Arc<Self> {
        let binding = PublicationResourceBinding::from_logical(logical);
        Arc::new(Self {
            binding: binding.clone(),
            owner: Arc::new(SealedPhysicalResourceOwner { binding, lifetime }),
        })
    }

    #[cfg(test)]
    fn synthetic_placement_replacement_for_test(
        predecessor: &PublicationAuthorityGeneration,
        lifetime: Option<TestResourceLifetime>,
    ) -> Result<Arc<Self>, DataGenerationError> {
        let binding = PublicationResourceBinding::placement_replacement_from(&predecessor.logical)?;
        Ok(Arc::new(Self {
            binding: binding.clone(),
            owner: Arc::new(SealedPhysicalResourceOwner { binding, lifetime }),
        }))
    }
}

/// The one indivisible publication tuple. It deliberately wraps the existing logical generation
/// rather than deriving, extracting, or comparing a new root format.
#[derive(Debug)]
pub(super) struct PublicationAuthorityGeneration {
    logical: Arc<PublicationGeneration>,
    catalog: Arc<CheckedCatalogSnapshot>,
    status: Arc<AuthenticatedStatusView>,
    resources: Arc<SealedPublicationResources>,
}

impl PublicationAuthorityGeneration {
    fn from_complete_parts(
        logical: Arc<PublicationGeneration>,
        catalog: Arc<CheckedCatalogSnapshot>,
        resources: Arc<SealedPublicationResources>,
    ) -> Result<Arc<Self>, DataGenerationError> {
        let status = AuthenticatedStatusView::from_logical(&logical)?;
        let generation = Arc::new(Self {
            logical,
            catalog,
            status,
            resources,
        });
        generation.validate_complete()?;
        Ok(generation)
    }

    fn validate_complete(&self) -> Result<(), DataGenerationError> {
        if self.logical.root_format != RootFormatVersion::V1 {
            return Err(DataGenerationError::UnsupportedRootFormat(
                self.logical.root_format.get(),
            ));
        }
        if self.logical.identity.visible_next.get().saturating_sub(1)
            != self.logical.identity.status_covered_through
        {
            return Err(DataGenerationError::Invalid("publication status boundary"));
        }
        if (self.logical.identity.status_covered_through == 0)
            != self
                .logical
                .identity
                .last_terminal_envelope_digest
                .is_none()
        {
            return Err(DataGenerationError::Invalid(
                "publication terminal boundary",
            ));
        }
        self.catalog.validate_for(self.logical.identity.catalog)?;
        self.status.validate_for(&self.logical)?;
        self.resources.validate_for(&self.logical)
    }

    fn capture_tuple(&self) -> PublicationTupleCapture {
        PublicationTupleCapture {
            identity: self.logical.identity.clone(),
            publication_epoch: self.logical.publication_epoch,
            catalog: Arc::clone(&self.catalog),
            status: Arc::clone(&self.status),
            resources: Arc::clone(&self.resources),
        }
    }
}

/// An exact captured tuple is held by a candidate in addition to its predecessor `Arc`. This
/// detects replacement of one component after a candidate has been sealed, including a component
/// with a superficially matching logical identity.
#[derive(Debug)]
struct PublicationTupleCapture {
    identity: PublicationIdentity,
    publication_epoch: PublicationEpoch,
    catalog: Arc<CheckedCatalogSnapshot>,
    status: Arc<AuthenticatedStatusView>,
    resources: Arc<SealedPublicationResources>,
}

impl PublicationTupleCapture {
    fn matches(&self, generation: &PublicationAuthorityGeneration) -> bool {
        self.identity == generation.logical.identity
            && self.publication_epoch == generation.logical.publication_epoch
            && Arc::ptr_eq(&self.catalog, &generation.catalog)
            && Arc::ptr_eq(&self.status, &generation.status)
            && Arc::ptr_eq(&self.resources, &generation.resources)
    }
}

/// Consumed by the private holder exactly once. A candidate retains its exact predecessor Arc and
/// snapshots every predecessor/successor component before the atomically checked replacement.
#[must_use = "a ready private publication candidate must be installed or intentionally dropped"]
pub(super) struct PublicationAuthorityCandidate {
    terminal_index: CommitSequence,
    terminal_entry: PublishedStatusEntry,
    predecessor: Arc<PublicationAuthorityGeneration>,
    predecessor_tuple: PublicationTupleCapture,
    successor: Arc<PublicationAuthorityGeneration>,
    successor_tuple: PublicationTupleCapture,
}

impl PublicationAuthorityCandidate {
    fn new(
        terminal_index: CommitSequence,
        terminal_entry: PublishedStatusEntry,
        predecessor: Arc<PublicationAuthorityGeneration>,
        successor: Arc<PublicationAuthorityGeneration>,
    ) -> Result<Self, DataGenerationError> {
        let candidate = Self {
            terminal_index,
            terminal_entry,
            predecessor_tuple: predecessor.capture_tuple(),
            successor_tuple: successor.capture_tuple(),
            predecessor,
            successor,
        };
        candidate.validate()?;
        Ok(candidate)
    }

    fn validate(&self) -> Result<(), DataGenerationError> {
        self.terminal_entry.validate()?;
        self.predecessor.validate_complete()?;
        self.successor.validate_complete()?;
        if self.terminal_entry.commit_sequence != self.terminal_index {
            return Err(DataGenerationError::Invalid(
                "candidate terminal status sequence",
            ));
        }
        if self
            .successor
            .logical
            .status
            .entry(self.terminal_entry.transaction_id)
            != Some(&self.terminal_entry)
            || !self.successor.status.contains_exact(&self.terminal_entry)
        {
            return Err(DataGenerationError::Invalid(
                "candidate successor status entry",
            ));
        }
        if self
            .successor
            .logical
            .identity
            .last_terminal_envelope_digest
            != Some(self.terminal_entry.terminal_envelope_digest)
        {
            return Err(DataGenerationError::Invalid("candidate terminal envelope"));
        }
        if self.terminal_entry.outcome.kind != TerminalOutcomeKind::CommitSuccess {
            if self.successor.logical.identity.catalog != self.predecessor.logical.identity.catalog
            {
                return Err(DataGenerationError::Invalid(
                    "candidate non-success catalog transition",
                ));
            }
            if self.successor.logical.identity.database_root
                != self.predecessor.logical.identity.database_root
                || self.successor.logical.tables.root() != self.predecessor.logical.tables.root()
            {
                return Err(DataGenerationError::Invalid(
                    "candidate non-success data closure",
                ));
            }
        }
        if !self.predecessor_tuple.matches(&self.predecessor) {
            return Err(DataGenerationError::PredecessorMismatch(
                "candidate retained predecessor tuple",
            ));
        }
        if !self.successor_tuple.matches(&self.successor) {
            return Err(DataGenerationError::Invalid("candidate successor tuple"));
        }
        if self.terminal_index.get() != self.predecessor_tuple.identity.visible_next.get() {
            return Err(DataGenerationError::PredecessorMismatch(
                "candidate terminal sequence",
            ));
        }
        let next_visible = self
            .terminal_index
            .get()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if self.successor_tuple.identity.visible_next.get() != next_visible
            || self.successor_tuple.identity.status_covered_through != self.terminal_index.get()
        {
            return Err(DataGenerationError::Invalid(
                "candidate successor status boundary",
            ));
        }
        let next_epoch = self
            .predecessor_tuple
            .publication_epoch
            .get()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if self.successor_tuple.publication_epoch.get() != next_epoch {
            return Err(DataGenerationError::PredecessorMismatch(
                "candidate publication epoch",
            ));
        }
        if self.successor.logical.root_format != self.predecessor.logical.root_format
            || self.successor.logical.database_id != self.predecessor.logical.database_id
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "candidate logical domain",
            ));
        }
        let expected_status_count = self
            .predecessor
            .status
            .entry_count
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if self.successor.status.entry_count != expected_status_count {
            return Err(DataGenerationError::Invalid("candidate status closure"));
        }
        Ok(())
    }
}

/// Consumed by the private holder exactly once to replace physical resources without changing a
/// logical publication. Its successor is derived solely from the retained predecessor and the
/// newly sealed resource owner; callers cannot supply a successor identity.
#[must_use = "a ready private placement replacement must be installed or intentionally dropped"]
struct PlacementReplacementCandidate {
    predecessor: Arc<PublicationAuthorityGeneration>,
    predecessor_tuple: PublicationTupleCapture,
    successor: Arc<PublicationAuthorityGeneration>,
    successor_tuple: PublicationTupleCapture,
}

impl PlacementReplacementCandidate {
    fn new(
        predecessor: Arc<PublicationAuthorityGeneration>,
        resources: Arc<SealedPublicationResources>,
    ) -> Result<Self, DataGenerationError> {
        predecessor.validate_complete()?;
        let replacement_binding =
            PublicationResourceBinding::placement_replacement_from(&predecessor.logical)?;
        let successor = Arc::new(PublicationAuthorityGeneration {
            logical: Arc::new(PublicationGeneration {
                root_format: predecessor.logical.root_format,
                database_id: predecessor.logical.database_id,
                publication_epoch: replacement_binding.publication_epoch,
                identity: predecessor.logical.identity.clone(),
                tables: predecessor.logical.tables.clone(),
                status: predecessor.logical.status.clone(),
            }),
            catalog: Arc::clone(&predecessor.catalog),
            status: Arc::clone(&predecessor.status),
            resources,
        });
        let candidate = Self {
            predecessor_tuple: predecessor.capture_tuple(),
            successor_tuple: successor.capture_tuple(),
            predecessor,
            successor,
        };
        candidate.validate()?;
        Ok(candidate)
    }

    fn validate(&self) -> Result<(), DataGenerationError> {
        self.predecessor.validate_complete()?;
        self.successor.validate_complete()?;
        if !self.predecessor_tuple.matches(&self.predecessor) {
            return Err(DataGenerationError::PredecessorMismatch(
                "placement retained predecessor tuple",
            ));
        }
        if !self.successor_tuple.matches(&self.successor) {
            return Err(DataGenerationError::Invalid("placement successor tuple"));
        }
        if self.successor.logical.root_format != self.predecessor.logical.root_format
            || self.successor.logical.database_id != self.predecessor.logical.database_id
            || self.successor.logical.identity != self.predecessor.logical.identity
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "placement logical identity",
            ));
        }
        if self.successor.logical.tables.root() != self.predecessor.logical.tables.root() {
            return Err(DataGenerationError::PredecessorMismatch(
                "placement table-map root",
            ));
        }
        if !Arc::ptr_eq(&self.successor.catalog, &self.predecessor.catalog) {
            return Err(DataGenerationError::PredecessorMismatch(
                "placement catalog witness",
            ));
        }
        if !Arc::ptr_eq(&self.successor.status, &self.predecessor.status) {
            return Err(DataGenerationError::PredecessorMismatch(
                "placement status witness",
            ));
        }
        let next_epoch = self
            .predecessor_tuple
            .publication_epoch
            .get()
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if self.successor_tuple.publication_epoch.get() != next_epoch {
            return Err(DataGenerationError::PredecessorMismatch(
                "placement publication epoch",
            ));
        }
        if Arc::ptr_eq(&self.successor.resources, &self.predecessor.resources)
            || Arc::ptr_eq(
                &self.successor.resources.owner,
                &self.predecessor.resources.owner,
            )
        {
            return Err(DataGenerationError::Invalid("placement resource reuse"));
        }
        Ok(())
    }
}

/// Private lock-free holder for exactly one complete immutable tuple. It has no connection to
/// production publication; a later coordinator may only use this shape after its own authority
/// boundary is accepted.
#[derive(Debug)]
pub(super) struct PublicationAuthorityHolder {
    current: ArcSwap<PublicationAuthorityGeneration>,
}

impl PublicationAuthorityHolder {
    fn new(genesis: Arc<PublicationAuthorityGeneration>) -> Result<Self, DataGenerationError> {
        genesis.validate_complete()?;
        if genesis.logical.identity.visible_next.get() != 1
            || genesis.logical.identity.status_covered_through != 0
            || genesis.status.entry_count != 0
        {
            return Err(DataGenerationError::Invalid("publication genesis boundary"));
        }
        Ok(Self {
            current: ArcSwap::new(genesis),
        })
    }

    fn acquire(&self) -> PublicationAuthorityPin {
        PublicationAuthorityPin {
            generation: self.current.load_full(),
        }
    }

    /// Consume a complete candidate. The compare-and-swap is the only replacement operation, so
    /// a stale candidate cannot overwrite a newer tuple between validation and installation.
    fn replace(&self, candidate: PublicationAuthorityCandidate) -> Result<(), DataGenerationError> {
        candidate.validate()?;
        let PublicationAuthorityCandidate {
            predecessor,
            successor,
            ..
        } = candidate;
        let observed = self.current.compare_and_swap(&predecessor, successor);
        if !Arc::ptr_eq(&observed, &predecessor) {
            return Err(DataGenerationError::PredecessorMismatch(
                "publication holder current tuple",
            ));
        }
        Ok(())
    }

    /// Placement replacement is deliberately a distinct exact-predecessor CAS. It cannot rebase
    /// on a terminal candidate or another placement replacement.
    fn replace_placement(
        &self,
        candidate: PlacementReplacementCandidate,
    ) -> Result<(), DataGenerationError> {
        candidate.validate()?;
        let PlacementReplacementCandidate {
            predecessor,
            successor,
            ..
        } = candidate;
        let observed = self.current.compare_and_swap(&predecessor, successor);
        if !Arc::ptr_eq(&observed, &predecessor) {
            return Err(DataGenerationError::PredecessorMismatch(
                "publication holder current tuple",
            ));
        }
        Ok(())
    }
}

/// A full-lifetime pin for one acquired publication tuple. It intentionally exposes no component
/// extraction API, preventing callers from re-pairing catalog, status, or resources later.
#[must_use = "a publication pin must be retained through the complete private operation"]
pub(super) struct PublicationAuthorityPin {
    generation: Arc<PublicationAuthorityGeneration>,
}

impl PublicationAuthorityPin {
    #[cfg(test)]
    fn is_exact(&self, generation: &Arc<PublicationAuthorityGeneration>) -> bool {
        Arc::ptr_eq(&self.generation, generation)
    }

    #[cfg(test)]
    fn validate_complete(&self) -> Result<(), DataGenerationError> {
        self.generation.validate_complete()
    }
}

#[cfg(test)]
#[derive(Debug)]
struct TestResourceLifetime {
    dropped: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
impl TestResourceLifetime {
    fn new(dropped: Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self { dropped }
    }
}

#[cfg(test)]
impl Drop for TestResourceLifetime {
    fn drop(&mut self) {
        self.dropped
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::engine_data_generation::digest::{
        synthetic_gpu_completion_for_test, CatalogEpoch, DatabaseId, RootFormatVersion,
        StableTransactionId, SyntheticRootForTest, VisibleNext,
    };
    use crate::engine_data_generation::manifest::{
        GpuRadixEmptyRoots, GpuRadixPathCompletion, GpuRadixPathNode, TableMap,
    };
    use crate::engine_data_generation::publication::{DataGenerationBuilder, GenesisGpuCompletion};
    use crate::engine_data_generation::status::{GpuStatusPathCompletion, PublishedStatusEntry};
    use crate::engine_data_generation::{
        input::{TerminalOutcome, TerminalOutcomeKind},
        DataGenerationError,
    };

    fn root<R: SyntheticRootForTest>(seed: u64) -> R {
        R::from_synthetic(synthetic_gpu_completion_for_test(seed))
    }

    fn empty_roots<R: SyntheticRootForTest>(seed: u64) -> GpuRadixEmptyRoots<R> {
        GpuRadixEmptyRoots::synthetic_for_test(
            (0..=64)
                .map(|depth| (depth as u8, root(seed + depth)))
                .collect(),
        )
    }

    fn status_path(seed: u64, prior_entries: u64) -> GpuStatusPathCompletion {
        GpuStatusPathCompletion {
            entry_leaf_root: root(seed),
            map_path: GpuRadixPathCompletion {
                leaf_root: root(seed + 1),
                nodes: (0..64)
                    .map(|depth| GpuRadixPathNode {
                        depth,
                        subtree_count: if prior_entries == 0 || depth > 62 {
                            1
                        } else {
                            2
                        },
                        root: root(seed + 2 + u64::from(depth)),
                    })
                    .collect(),
            },
        }
    }

    fn catalog(seed: u64) -> CatalogIdentity {
        CatalogIdentity::new(CatalogEpoch::new(seed), root(seed + 100))
    }

    fn terminal_noop() -> TerminalOutcome {
        TerminalOutcome {
            kind: TerminalOutcomeKind::CommitNoOp,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
        }
    }

    fn terminal_success() -> TerminalOutcome {
        TerminalOutcome {
            kind: TerminalOutcomeKind::CommitSuccess,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
        }
    }

    fn terminal_abort() -> TerminalOutcome {
        TerminalOutcome {
            kind: TerminalOutcomeKind::AbortError,
            affected_rows: 0,
            sqlstate: Some(*b"23505"),
            constraint_id: 9,
        }
    }

    fn genesis_logical() -> Arc<PublicationGeneration> {
        let builder = DataGenerationBuilder::new(
            RootFormatVersion::V1,
            DatabaseId::new([7; 16]).expect("database id"),
        )
        .expect("builder");
        builder
            .genesis(
                catalog(1),
                GenesisGpuCompletion {
                    empty_table_map_roots: empty_roots(200),
                    empty_status_map_roots: empty_roots(300),
                    database_root: root(400),
                },
            )
            .expect("genesis")
    }

    fn complete_generation(
        logical: Arc<PublicationGeneration>,
        lifetime: Option<TestResourceLifetime>,
    ) -> Arc<PublicationAuthorityGeneration> {
        let catalog = CheckedCatalogSnapshot::synthetic_for_test(logical.identity.catalog);
        let resources = SealedPublicationResources::synthetic_for_test(&logical, lifetime);
        PublicationAuthorityGeneration::from_complete_parts(logical, catalog, resources)
            .expect("complete generation")
    }

    fn successor_logical(
        predecessor: &PublicationAuthorityGeneration,
        catalog_after: CatalogIdentity,
        outcome: TerminalOutcome,
        seed: u64,
    ) -> Arc<PublicationGeneration> {
        let terminal_index = CommitSequence::new(predecessor.logical.identity.visible_next.get())
            .expect("terminal index");
        successor_logical_at(predecessor, catalog_after, terminal_index, outcome, seed)
    }

    fn successor_logical_at(
        predecessor: &PublicationAuthorityGeneration,
        catalog_after: CatalogIdentity,
        terminal_index: CommitSequence,
        outcome: TerminalOutcome,
        seed: u64,
    ) -> Arc<PublicationGeneration> {
        let status = predecessor
            .logical
            .status
            .insert(
                PublishedStatusEntry {
                    transaction_id: super::super::digest::StableTransactionId::new(
                        terminal_index.get(),
                    )
                    .expect("transaction id"),
                    request_digest: root(seed),
                    commit_sequence: terminal_index,
                    outcome,
                    target_digest: root(seed + 1),
                    returning_digest: root(seed + 2),
                    terminal_envelope_digest: root(seed + 3),
                },
                &status_path(seed + 10, predecessor.logical.status.count()),
            )
            .expect("status successor");
        Arc::new(PublicationGeneration {
            root_format: predecessor.logical.root_format,
            database_id: predecessor.logical.database_id,
            publication_epoch: PublicationEpoch::new(
                predecessor.logical.publication_epoch.get() + 1,
            )
            .expect("successor epoch"),
            identity: PublicationIdentity {
                visible_next: VisibleNext::new(terminal_index.get() + 1)
                    .expect("successor visibility"),
                database_root: predecessor.logical.identity.database_root,
                catalog: catalog_after,
                status_covered_through: terminal_index.get(),
                status_view_root: status.root(),
                last_terminal_envelope_digest: Some(root(seed + 3)),
            },
            tables: predecessor.logical.tables.clone(),
            status,
        })
    }

    fn successor_generation(
        predecessor: &Arc<PublicationAuthorityGeneration>,
        catalog_after: CatalogIdentity,
        outcome: TerminalOutcome,
        seed: u64,
    ) -> Arc<PublicationAuthorityGeneration> {
        complete_generation(
            successor_logical(predecessor, catalog_after, outcome, seed),
            None,
        )
    }

    fn ready_successor(
        predecessor: &Arc<PublicationAuthorityGeneration>,
        catalog_after: CatalogIdentity,
        outcome: TerminalOutcome,
        seed: u64,
    ) -> (
        Arc<PublicationAuthorityGeneration>,
        PublicationAuthorityCandidate,
    ) {
        let successor = successor_generation(predecessor, catalog_after, outcome, seed);
        let terminal_index = CommitSequence::new(predecessor.logical.identity.visible_next.get())
            .expect("terminal index");
        let terminal_entry = successor
            .logical
            .status
            .entry(StableTransactionId::new(terminal_index.get()).expect("transaction id"))
            .cloned()
            .expect("terminal status entry");
        let candidate = PublicationAuthorityCandidate::new(
            terminal_index,
            terminal_entry,
            Arc::clone(predecessor),
            Arc::clone(&successor),
        )
        .expect("candidate");
        (successor, candidate)
    }

    fn ready_placement_replacement(
        predecessor: &Arc<PublicationAuthorityGeneration>,
        lifetime: Option<TestResourceLifetime>,
    ) -> (
        Arc<PublicationAuthorityGeneration>,
        PlacementReplacementCandidate,
    ) {
        let resources = SealedPublicationResources::synthetic_placement_replacement_for_test(
            predecessor,
            lifetime,
        )
        .expect("placement resources");
        let candidate = PlacementReplacementCandidate::new(Arc::clone(predecessor), resources)
            .expect("placement candidate");
        (Arc::clone(&candidate.successor), candidate)
    }

    fn refresh_placement_successor_tuple_for_test(candidate: &mut PlacementReplacementCandidate) {
        candidate.successor_tuple = candidate.successor.capture_tuple();
    }

    fn retie_placement_logical_for_test(
        candidate: &mut PlacementReplacementCandidate,
        logical: Arc<PublicationGeneration>,
        resources: Arc<SealedPublicationResources>,
    ) {
        let successor_tuple = {
            let successor =
                Arc::get_mut(&mut candidate.successor).expect("unshared placement successor");
            successor.logical = logical;
            successor.resources = resources;
            successor.capture_tuple()
        };
        candidate.successor_tuple = successor_tuple;
    }

    fn successor_terminal_entry(
        successor: &PublicationAuthorityGeneration,
        terminal_index: CommitSequence,
    ) -> PublishedStatusEntry {
        successor
            .logical
            .status
            .entry(StableTransactionId::new(terminal_index.get()).expect("transaction id"))
            .cloned()
            .expect("terminal status entry")
    }

    fn retie_successor_for_test(
        successor: &mut PublicationAuthorityGeneration,
        identity: PublicationIdentity,
        tables: TableMap,
        status: PublishedStatusIndex,
    ) {
        let status_view = Arc::new(AuthenticatedStatusView {
            root: status.root(),
            entry_count: status.count(),
            index: Arc::new(status.clone()),
        });
        successor.logical = Arc::new(PublicationGeneration {
            root_format: successor.logical.root_format,
            database_id: successor.logical.database_id,
            publication_epoch: successor.logical.publication_epoch,
            identity,
            tables,
            status,
        });
        successor.status = status_view;
        successor.resources =
            SealedPublicationResources::synthetic_for_test(&successor.logical, None);
    }

    #[test]
    fn genesis_tuple_is_complete_and_pinnable() {
        let genesis = complete_generation(genesis_logical(), None);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");
        let pin = holder.acquire();

        assert!(pin.is_exact(&genesis));
        pin.validate_complete().expect("complete genesis pin");
        assert_eq!(genesis.logical.identity.visible_next.get(), 1);
        assert_eq!(genesis.status.entry_count, 0);
    }

    #[test]
    fn noop_candidate_advances_status_without_relabeling_data_or_catalog() {
        let genesis = complete_generation(genesis_logical(), None);
        let (successor, candidate) = ready_successor(
            &genesis,
            genesis.logical.identity.catalog,
            terminal_noop(),
            500,
        );
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");

        holder.replace(candidate).expect("replace noop");
        let pin = holder.acquire();
        assert!(pin.is_exact(&successor));
        assert_eq!(
            successor.logical.identity.database_root,
            genesis.logical.identity.database_root
        );
        assert_eq!(
            successor.logical.identity.catalog,
            genesis.logical.identity.catalog
        );
        assert_eq!(successor.status.entry_count, genesis.status.entry_count + 1);
        assert_eq!(successor.logical.identity.status_covered_through, 1);
        assert_eq!(
            successor
                .logical
                .status
                .entry(StableTransactionId::new(1).expect("transaction id"))
                .expect("terminal status entry")
                .outcome
                .kind,
            TerminalOutcomeKind::CommitNoOp
        );
    }

    #[test]
    fn catalog_only_candidate_advances_catalog_without_relabeling_data() {
        let genesis = complete_generation(genesis_logical(), None);
        let next_catalog = catalog(2);
        let (successor, candidate) =
            ready_successor(&genesis, next_catalog, terminal_success(), 600);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");

        holder.replace(candidate).expect("replace catalog-only");
        assert!(holder.acquire().is_exact(&successor));
        assert_ne!(
            successor.logical.identity.catalog,
            genesis.logical.identity.catalog
        );
        assert_eq!(
            successor.logical.identity.database_root,
            genesis.logical.identity.database_root
        );
        assert_eq!(
            successor.logical.tables.root(),
            genesis.logical.tables.root()
        );
        assert_eq!(
            successor
                .logical
                .status
                .entry(StableTransactionId::new(1).expect("transaction id"))
                .expect("terminal status entry")
                .outcome
                .kind,
            TerminalOutcomeKind::CommitSuccess
        );
    }

    #[test]
    fn placement_replacement_derives_only_epoch_and_new_resources() {
        let genesis = complete_generation(genesis_logical(), None);
        let (successor, candidate) = ready_placement_replacement(&genesis, None);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");

        assert_eq!(successor.logical.root_format, genesis.logical.root_format);
        assert_eq!(successor.logical.database_id, genesis.logical.database_id);
        assert_eq!(successor.logical.identity, genesis.logical.identity);
        assert_eq!(
            successor.logical.tables.root(),
            genesis.logical.tables.root()
        );
        assert!(Arc::ptr_eq(&successor.catalog, &genesis.catalog));
        assert!(Arc::ptr_eq(&successor.status, &genesis.status));
        assert_eq!(successor.status.entry_count, genesis.status.entry_count);
        assert_eq!(
            successor.logical.publication_epoch.get(),
            genesis.logical.publication_epoch.get() + 1
        );
        assert!(!Arc::ptr_eq(&successor.resources, &genesis.resources));
        assert!(!Arc::ptr_eq(
            &successor.resources.owner,
            &genesis.resources.owner
        ));

        holder
            .replace_placement(candidate)
            .expect("placement replacement");
        assert!(holder.acquire().is_exact(&successor));
    }

    #[test]
    fn placement_replacement_rejects_logical_witness_and_resource_substitution() {
        let genesis = complete_generation(genesis_logical(), None);

        let (_, mut identity) = ready_placement_replacement(&genesis, None);
        let logical = Arc::new(PublicationGeneration {
            root_format: identity.successor.logical.root_format,
            database_id: identity.successor.logical.database_id,
            publication_epoch: identity.successor.logical.publication_epoch,
            identity: PublicationIdentity {
                database_root: root(10_001),
                ..identity.successor.logical.identity.clone()
            },
            tables: identity.successor.logical.tables.clone(),
            status: identity.successor.logical.status.clone(),
        });
        let resources = SealedPublicationResources::synthetic_for_test(&logical, None);
        retie_placement_logical_for_test(&mut identity, logical, resources);
        assert!(matches!(
            identity.validate(),
            Err(DataGenerationError::PredecessorMismatch(
                "placement logical identity"
            ))
        ));

        let (_, mut catalog_substitution) = ready_placement_replacement(&genesis, None);
        let replacement_catalog = catalog(10_100);
        let logical = Arc::new(PublicationGeneration {
            root_format: catalog_substitution.successor.logical.root_format,
            database_id: catalog_substitution.successor.logical.database_id,
            publication_epoch: catalog_substitution.successor.logical.publication_epoch,
            identity: PublicationIdentity {
                catalog: replacement_catalog,
                ..catalog_substitution.successor.logical.identity.clone()
            },
            tables: catalog_substitution.successor.logical.tables.clone(),
            status: catalog_substitution.successor.logical.status.clone(),
        });
        let resources = SealedPublicationResources::synthetic_for_test(&logical, None);
        let successor_tuple = {
            let successor = Arc::get_mut(&mut catalog_substitution.successor)
                .expect("unshared placement successor");
            successor.logical = logical;
            successor.catalog = CheckedCatalogSnapshot::synthetic_for_test(replacement_catalog);
            successor.resources = resources;
            successor.capture_tuple()
        };
        catalog_substitution.successor_tuple = successor_tuple;
        assert!(matches!(
            catalog_substitution.validate(),
            Err(DataGenerationError::PredecessorMismatch(
                "placement logical identity"
            ))
        ));

        let (_, mut catalog_witness) = ready_placement_replacement(&genesis, None);
        let successor_tuple = {
            let successor =
                Arc::get_mut(&mut catalog_witness.successor).expect("unshared placement successor");
            successor.catalog =
                CheckedCatalogSnapshot::synthetic_for_test(successor.logical.identity.catalog);
            successor.capture_tuple()
        };
        catalog_witness.successor_tuple = successor_tuple;
        assert!(matches!(
            catalog_witness.validate(),
            Err(DataGenerationError::PredecessorMismatch(
                "placement catalog witness"
            ))
        ));

        let (_, mut status_substitution) = ready_placement_replacement(&genesis, None);
        let successor_tuple = {
            let successor = Arc::get_mut(&mut status_substitution.successor)
                .expect("unshared placement successor");
            successor.status = AuthenticatedStatusView::from_logical(&successor.logical)
                .expect("replacement status witness");
            successor.capture_tuple()
        };
        status_substitution.successor_tuple = successor_tuple;
        assert!(matches!(
            status_substitution.validate(),
            Err(DataGenerationError::PredecessorMismatch(
                "placement status witness"
            ))
        ));

        let (_, mut table_root) = ready_placement_replacement(&genesis, None);
        let tables = TableMap::empty(empty_roots(10_200)).expect("replacement table map");
        assert_ne!(tables.root(), genesis.logical.tables.root());
        let logical = Arc::new(PublicationGeneration {
            root_format: table_root.successor.logical.root_format,
            database_id: table_root.successor.logical.database_id,
            publication_epoch: table_root.successor.logical.publication_epoch,
            identity: table_root.successor.logical.identity.clone(),
            tables,
            status: table_root.successor.logical.status.clone(),
        });
        let resources = SealedPublicationResources::synthetic_for_test(&logical, None);
        retie_placement_logical_for_test(&mut table_root, logical, resources);
        assert!(matches!(
            table_root.validate(),
            Err(DataGenerationError::PredecessorMismatch(
                "placement table-map root"
            ))
        ));

        let (_, mut resource_substitution) = ready_placement_replacement(&genesis, None);
        let replacement =
            SealedPublicationResources::synthetic_placement_replacement_for_test(&genesis, None)
                .expect("replacement resources");
        Arc::get_mut(&mut resource_substitution.successor)
            .expect("unshared placement successor")
            .resources = replacement;
        assert!(matches!(
            resource_substitution.validate(),
            Err(DataGenerationError::Invalid("placement successor tuple"))
        ));
    }

    #[test]
    fn placement_replacement_rejects_visible_coverage_digest_and_epoch_substitution() {
        let genesis = complete_generation(genesis_logical(), None);

        for identity in [
            PublicationIdentity {
                visible_next: VisibleNext::new(2).expect("visible next"),
                ..genesis.logical.identity.clone()
            },
            PublicationIdentity {
                status_covered_through: 1,
                ..genesis.logical.identity.clone()
            },
            PublicationIdentity {
                last_terminal_envelope_digest: Some(root(10_300)),
                ..genesis.logical.identity.clone()
            },
        ] {
            let (_, mut candidate) = ready_placement_replacement(&genesis, None);
            let logical = Arc::new(PublicationGeneration {
                root_format: candidate.successor.logical.root_format,
                database_id: candidate.successor.logical.database_id,
                publication_epoch: candidate.successor.logical.publication_epoch,
                identity,
                tables: candidate.successor.logical.tables.clone(),
                status: candidate.successor.logical.status.clone(),
            });
            let resources = SealedPublicationResources::synthetic_for_test(&logical, None);
            retie_placement_logical_for_test(&mut candidate, logical, resources);
            assert!(matches!(
                candidate.validate(),
                Err(DataGenerationError::Invalid(_))
            ));
        }

        let (_, mut epoch) = ready_placement_replacement(&genesis, None);
        let logical = Arc::new(PublicationGeneration {
            root_format: epoch.successor.logical.root_format,
            database_id: epoch.successor.logical.database_id,
            publication_epoch: PublicationEpoch::new(3).expect("wrong epoch"),
            identity: epoch.successor.logical.identity.clone(),
            tables: epoch.successor.logical.tables.clone(),
            status: epoch.successor.logical.status.clone(),
        });
        let resources = SealedPublicationResources::synthetic_for_test(&logical, None);
        retie_placement_logical_for_test(&mut epoch, logical, resources);
        assert!(matches!(
            epoch.validate(),
            Err(DataGenerationError::PredecessorMismatch(
                "placement publication epoch"
            ))
        ));
    }

    #[test]
    fn terminal_and_placement_candidates_stale_each_other_without_rebase() {
        let genesis = complete_generation(genesis_logical(), None);
        let (terminal_successor, terminal) =
            ready_successor(&genesis, catalog(10_400), terminal_success(), 10_401);
        let (_, placement) = ready_placement_replacement(&genesis, None);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");

        holder.replace(terminal).expect("terminal replacement");
        assert!(matches!(
            holder.replace_placement(placement),
            Err(DataGenerationError::PredecessorMismatch(
                "publication holder current tuple"
            ))
        ));
        assert!(holder.acquire().is_exact(&terminal_successor));

        let genesis = complete_generation(genesis_logical(), None);
        let (placement_successor, placement) = ready_placement_replacement(&genesis, None);
        let (_, terminal) = ready_successor(&genesis, catalog(10_500), terminal_success(), 10_501);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");

        holder
            .replace_placement(placement)
            .expect("placement replacement");
        assert!(matches!(
            holder.replace(terminal),
            Err(DataGenerationError::PredecessorMismatch(
                "publication holder current tuple"
            ))
        ));
        assert!(holder.acquire().is_exact(&placement_successor));
    }

    #[test]
    fn placement_candidates_stale_each_other_without_rebase() {
        let genesis = complete_generation(genesis_logical(), None);
        let (first_successor, first) = ready_placement_replacement(&genesis, None);
        let (_, stale) = ready_placement_replacement(&genesis, None);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");

        holder
            .replace_placement(first)
            .expect("first placement replacement");
        assert!(matches!(
            holder.replace_placement(stale),
            Err(DataGenerationError::PredecessorMismatch(
                "publication holder current tuple"
            ))
        ));
        assert!(holder.acquire().is_exact(&first_successor));
    }

    #[test]
    fn placement_replacement_pins_observe_only_complete_old_or_new_tuples() {
        let genesis = complete_generation(genesis_logical(), None);
        let (successor, candidate) = ready_placement_replacement(&genesis, None);
        let holder =
            Arc::new(PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder"));
        let barrier = Arc::new(Barrier::new(2));
        let reader_holder = Arc::clone(&holder);
        let reader_barrier = Arc::clone(&barrier);
        let reader_genesis = Arc::clone(&genesis);
        let reader_successor = Arc::clone(&successor);
        let reader = std::thread::spawn(move || {
            reader_barrier.wait();
            for _ in 0..2_000 {
                let pin = reader_holder.acquire();
                assert!(pin.is_exact(&reader_genesis) || pin.is_exact(&reader_successor));
                pin.validate_complete().expect("complete pinned tuple");
            }
        });

        barrier.wait();
        holder
            .replace_placement(candidate)
            .expect("atomic placement replacement");
        reader.join().expect("reader thread");
        assert!(holder.acquire().is_exact(&successor));
    }

    #[test]
    fn old_placement_resources_live_until_the_final_pin_drops() {
        let dropped = Arc::new(AtomicBool::new(false));
        let genesis = complete_generation(genesis_logical(), None);
        let (first_successor, first) = ready_placement_replacement(
            &genesis,
            Some(TestResourceLifetime::new(Arc::clone(&dropped))),
        );
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");
        holder
            .replace_placement(first)
            .expect("first placement replacement");
        let pin = holder.acquire();
        let (_, second) = ready_placement_replacement(&first_successor, None);

        holder
            .replace_placement(second)
            .expect("second placement replacement");
        drop(first_successor);
        assert!(!dropped.load(Ordering::Acquire));
        drop(pin);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn placement_replacement_rejects_epoch_overflow() {
        let mut logical = genesis_logical();
        Arc::get_mut(&mut logical)
            .expect("unshared genesis logical")
            .publication_epoch = PublicationEpoch::new(u64::MAX).expect("maximum epoch");
        let predecessor = complete_generation(logical, None);
        let resources = SealedPublicationResources::synthetic_for_test(&predecessor.logical, None);

        assert!(matches!(
            PlacementReplacementCandidate::new(predecessor, resources),
            Err(DataGenerationError::CountOverflow)
        ));
    }

    #[test]
    fn candidate_rejects_terminal_entry_substitution_and_status_absence() {
        let genesis = complete_generation(genesis_logical(), None);

        let (_, mut substituted_entry) =
            ready_successor(&genesis, catalog(2), terminal_success(), 650);
        substituted_entry.terminal_entry.outcome = terminal_noop();
        assert!(matches!(
            substituted_entry.validate(),
            Err(DataGenerationError::Invalid(
                "candidate successor status entry"
            ))
        ));

        let (_, mut absent_status) = ready_successor(&genesis, catalog(2), terminal_success(), 660);
        let successor = Arc::get_mut(&mut absent_status.successor).expect("unshared successor");
        successor.logical = Arc::new(PublicationGeneration {
            root_format: successor.logical.root_format,
            database_id: successor.logical.database_id,
            publication_epoch: successor.logical.publication_epoch,
            identity: PublicationIdentity {
                status_view_root: genesis.logical.status.root(),
                ..successor.logical.identity.clone()
            },
            tables: successor.logical.tables.clone(),
            status: genesis.logical.status.clone(),
        });
        successor.status =
            AuthenticatedStatusView::from_logical(&successor.logical).expect("empty status view");
        successor.resources =
            SealedPublicationResources::synthetic_for_test(&successor.logical, None);
        assert!(matches!(
            absent_status.validate(),
            Err(DataGenerationError::Invalid(
                "candidate successor status entry"
            ))
        ));
    }

    #[test]
    fn candidate_rejects_terminal_digest_mismatch_and_non_success_catalog_transition() {
        let genesis = complete_generation(genesis_logical(), None);

        let (_, mut mismatched_digest) =
            ready_successor(&genesis, catalog(2), terminal_success(), 670);
        let successor = Arc::get_mut(&mut mismatched_digest.successor).expect("unshared successor");
        successor.logical = Arc::new(PublicationGeneration {
            root_format: successor.logical.root_format,
            database_id: successor.logical.database_id,
            publication_epoch: successor.logical.publication_epoch,
            identity: PublicationIdentity {
                last_terminal_envelope_digest: Some(root(679)),
                ..successor.logical.identity.clone()
            },
            tables: successor.logical.tables.clone(),
            status: successor.logical.status.clone(),
        });
        successor.resources =
            SealedPublicationResources::synthetic_for_test(&successor.logical, None);
        assert!(matches!(
            mismatched_digest.validate(),
            Err(DataGenerationError::Invalid("candidate terminal envelope"))
        ));

        for outcome in [terminal_noop(), terminal_abort()] {
            let successor = successor_generation(&genesis, catalog(3), outcome, 680);
            let terminal_index = CommitSequence::new(genesis.logical.identity.visible_next.get())
                .expect("terminal index");
            let terminal_entry = successor
                .logical
                .status
                .entry(StableTransactionId::new(terminal_index.get()).expect("transaction id"))
                .cloned()
                .expect("terminal status entry");
            assert!(matches!(
                PublicationAuthorityCandidate::new(
                    terminal_index,
                    terminal_entry,
                    Arc::clone(&genesis),
                    successor,
                ),
                Err(DataGenerationError::Invalid(
                    "candidate non-success catalog transition"
                ))
            ));
        }
    }

    #[test]
    fn non_success_candidates_reject_data_and_table_map_substitution() {
        let genesis = complete_generation(genesis_logical(), None);
        let terminal_index = CommitSequence::new(genesis.logical.identity.visible_next.get())
            .expect("terminal index");

        for (seed, outcome) in [(2_000, terminal_noop()), (2_200, terminal_abort())] {
            let mut database_root_substitution = successor_generation(
                &genesis,
                genesis.logical.identity.catalog,
                outcome.clone(),
                seed,
            );
            let terminal_entry =
                successor_terminal_entry(&database_root_substitution, terminal_index);
            let successor = Arc::get_mut(&mut database_root_substitution)
                .expect("unshared database-root successor");
            let identity = PublicationIdentity {
                database_root: root(seed + 90),
                ..successor.logical.identity.clone()
            };
            retie_successor_for_test(
                successor,
                identity,
                successor.logical.tables.clone(),
                successor.logical.status.clone(),
            );
            assert!(matches!(
                PublicationAuthorityCandidate::new(
                    terminal_index,
                    terminal_entry,
                    Arc::clone(&genesis),
                    database_root_substitution,
                ),
                Err(DataGenerationError::Invalid(
                    "candidate non-success data closure"
                ))
            ));

            let mut table_map_substitution = successor_generation(
                &genesis,
                genesis.logical.identity.catalog,
                outcome,
                seed + 100,
            );
            let terminal_entry = successor_terminal_entry(&table_map_substitution, terminal_index);
            let replacement_tables =
                TableMap::empty(empty_roots(seed + 190)).expect("replacement table map");
            assert_ne!(replacement_tables.root(), genesis.logical.tables.root());
            let successor =
                Arc::get_mut(&mut table_map_substitution).expect("unshared table-map successor");
            retie_successor_for_test(
                successor,
                successor.logical.identity.clone(),
                replacement_tables,
                successor.logical.status.clone(),
            );
            assert!(matches!(
                PublicationAuthorityCandidate::new(
                    terminal_index,
                    terminal_entry,
                    Arc::clone(&genesis),
                    table_map_substitution,
                ),
                Err(DataGenerationError::Invalid(
                    "candidate non-success data closure"
                ))
            ));
        }
    }

    #[test]
    fn candidate_rejects_invalid_terminal_outcome_and_status_sequence() {
        let genesis = complete_generation(genesis_logical(), None);
        let terminal_index = CommitSequence::new(genesis.logical.identity.visible_next.get())
            .expect("terminal index");

        let mut invalid_outcome_successor = successor_generation(
            &genesis,
            genesis.logical.identity.catalog,
            terminal_noop(),
            2_500,
        );
        let mut invalid_outcome =
            successor_terminal_entry(&invalid_outcome_successor, terminal_index);
        invalid_outcome.outcome.affected_rows = 1;
        let replacement_status = invalid_outcome_successor
            .logical
            .status
            .replace_entry_unchecked_for_test(invalid_outcome.clone(), &status_path(2_510, 0))
            .expect("coherent invalid-outcome status");
        let successor = Arc::get_mut(&mut invalid_outcome_successor)
            .expect("unshared invalid-outcome successor");
        let identity = PublicationIdentity {
            status_view_root: replacement_status.root(),
            ..successor.logical.identity.clone()
        };
        retie_successor_for_test(
            successor,
            identity,
            successor.logical.tables.clone(),
            replacement_status,
        );
        assert!(matches!(
            PublicationAuthorityCandidate::new(
                terminal_index,
                invalid_outcome,
                Arc::clone(&genesis),
                invalid_outcome_successor,
            ),
            Err(DataGenerationError::Invalid("commit outcome"))
        ));

        let mut wrong_sequence_successor =
            successor_generation(&genesis, catalog(2), terminal_success(), 2_600);
        let mut wrong_sequence =
            successor_terminal_entry(&wrong_sequence_successor, terminal_index);
        wrong_sequence.commit_sequence =
            CommitSequence::new(terminal_index.get() + 1).expect("wrong commit sequence");
        let replacement_status = wrong_sequence_successor
            .logical
            .status
            .replace_entry_unchecked_for_test(wrong_sequence.clone(), &status_path(2_610, 0))
            .expect("coherent wrong-sequence status");
        let successor =
            Arc::get_mut(&mut wrong_sequence_successor).expect("unshared wrong-sequence successor");
        let identity = PublicationIdentity {
            status_view_root: replacement_status.root(),
            ..successor.logical.identity.clone()
        };
        retie_successor_for_test(
            successor,
            identity,
            successor.logical.tables.clone(),
            replacement_status,
        );
        assert!(matches!(
            PublicationAuthorityCandidate::new(
                terminal_index,
                wrong_sequence,
                Arc::clone(&genesis),
                wrong_sequence_successor,
            ),
            Err(DataGenerationError::Invalid(
                "candidate terminal status sequence"
            ))
        ));
    }

    #[test]
    fn candidate_rejects_stale_and_gapped_predecessors_before_swap() {
        let genesis = complete_generation(genesis_logical(), None);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");
        let (first_successor, first) =
            ready_successor(&genesis, catalog(2), terminal_success(), 700);
        let (_, stale) = ready_successor(&genesis, catalog(3), terminal_success(), 800);

        holder.replace(first).expect("first replacement");
        assert!(matches!(
            holder.replace(stale),
            Err(DataGenerationError::PredecessorMismatch(
                "publication holder current tuple"
            ))
        ));
        assert!(holder.acquire().is_exact(&first_successor));

        let terminal_index = CommitSequence::new(3).expect("gapped terminal");
        let gapped_successor = complete_generation(
            successor_logical_at(
                &first_successor,
                catalog(4),
                terminal_index,
                terminal_success(),
                900,
            ),
            None,
        );
        let terminal_entry = gapped_successor
            .logical
            .status
            .entry(StableTransactionId::new(3).expect("transaction id"))
            .cloned()
            .expect("gapped terminal entry");
        let gapped = PublicationAuthorityCandidate::new(
            terminal_index,
            terminal_entry,
            Arc::clone(&first_successor),
            gapped_successor,
        );
        assert!(matches!(
            gapped,
            Err(DataGenerationError::PredecessorMismatch(
                "candidate terminal sequence"
            ))
        ));
    }

    #[test]
    fn candidate_rejects_epoch_catalog_status_root_and_resource_substitution() {
        let genesis = complete_generation(genesis_logical(), None);

        let (_, mut epoch) = ready_successor(&genesis, catalog(2), terminal_success(), 1_000);
        let successor = Arc::get_mut(&mut epoch.successor).expect("unshared successor");
        successor.logical = Arc::new(PublicationGeneration {
            publication_epoch: PublicationEpoch::new(9).expect("different epoch"),
            root_format: successor.logical.root_format,
            database_id: successor.logical.database_id,
            identity: successor.logical.identity.clone(),
            tables: successor.logical.tables.clone(),
            status: successor.logical.status.clone(),
        });
        successor.resources =
            SealedPublicationResources::synthetic_for_test(&successor.logical, None);
        assert!(matches!(
            epoch.validate(),
            Err(DataGenerationError::Invalid("candidate successor tuple"))
        ));

        let (_, mut catalog_substitution) =
            ready_successor(&genesis, catalog(2), terminal_success(), 1_100);
        let successor =
            Arc::get_mut(&mut catalog_substitution.successor).expect("unshared successor");
        successor.catalog =
            CheckedCatalogSnapshot::synthetic_for_test(successor.logical.identity.catalog);
        assert!(matches!(
            catalog_substitution.validate(),
            Err(DataGenerationError::Invalid("candidate successor tuple"))
        ));

        let (_, mut status_substitution) =
            ready_successor(&genesis, catalog(2), terminal_success(), 1_200);
        let successor =
            Arc::get_mut(&mut status_substitution.successor).expect("unshared successor");
        successor.status = AuthenticatedStatusView::from_logical(&successor.logical)
            .expect("replacement status view");
        assert!(matches!(
            status_substitution.validate(),
            Err(DataGenerationError::Invalid("candidate successor tuple"))
        ));

        let (_, mut root_substitution) =
            ready_successor(&genesis, catalog(2), terminal_success(), 1_300);
        let successor = Arc::get_mut(&mut root_substitution.successor).expect("unshared successor");
        successor.logical = Arc::new(PublicationGeneration {
            root_format: successor.logical.root_format,
            database_id: successor.logical.database_id,
            publication_epoch: successor.logical.publication_epoch,
            identity: PublicationIdentity {
                database_root: root(1_399),
                ..successor.logical.identity.clone()
            },
            tables: successor.logical.tables.clone(),
            status: successor.logical.status.clone(),
        });
        successor.resources =
            SealedPublicationResources::synthetic_for_test(&successor.logical, None);
        assert!(matches!(
            root_substitution.validate(),
            Err(DataGenerationError::Invalid("candidate successor tuple"))
        ));

        let (_, mut resource_substitution) =
            ready_successor(&genesis, catalog(2), terminal_success(), 1_400);
        let successor =
            Arc::get_mut(&mut resource_substitution.successor).expect("unshared successor");
        successor.resources =
            SealedPublicationResources::synthetic_for_test(&successor.logical, None);
        assert!(matches!(
            resource_substitution.validate(),
            Err(DataGenerationError::Invalid("candidate successor tuple"))
        ));
    }

    #[test]
    fn concurrent_pins_observe_only_complete_old_or_new_tuples() {
        let genesis = complete_generation(genesis_logical(), None);
        let (successor, candidate) =
            ready_successor(&genesis, catalog(2), terminal_success(), 1_500);
        let holder =
            Arc::new(PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder"));
        let barrier = Arc::new(Barrier::new(2));
        let reader_holder = Arc::clone(&holder);
        let reader_barrier = Arc::clone(&barrier);
        let reader_genesis = Arc::clone(&genesis);
        let reader_successor = Arc::clone(&successor);
        let reader = std::thread::spawn(move || {
            reader_barrier.wait();
            for _ in 0..2_000 {
                let pin = reader_holder.acquire();
                assert!(pin.is_exact(&reader_genesis) || pin.is_exact(&reader_successor));
                pin.validate_complete().expect("complete pinned tuple");
            }
        });

        barrier.wait();
        holder.replace(candidate).expect("atomic replacement");
        reader.join().expect("reader thread");
        assert!(holder.acquire().is_exact(&successor));
    }

    #[test]
    fn old_resources_live_until_the_last_pin_drops() {
        let dropped = Arc::new(AtomicBool::new(false));
        let genesis = complete_generation(
            genesis_logical(),
            Some(TestResourceLifetime::new(Arc::clone(&dropped))),
        );
        let (successor, candidate) =
            ready_successor(&genesis, catalog(2), terminal_success(), 1_600);
        let holder = PublicationAuthorityHolder::new(Arc::clone(&genesis)).expect("holder");
        let pin = holder.acquire();

        holder.replace(candidate).expect("replacement");
        drop(genesis);
        assert!(!dropped.load(Ordering::Acquire));
        drop(pin);
        assert!(dropped.load(Ordering::Acquire));
        assert!(holder.acquire().is_exact(&successor));
    }

    #[test]
    fn source_boundary_stays_private_and_unwired() {
        let source = include_str!("publication_authority.rs");
        for forbidden in [
            ["Sealed", "Int4PublicationGenerationV1"].concat(),
            ["Boot", "strapPublication"].concat(),
            ["Engine", "State"].concat(),
            ["engine_", "state"].concat(),
            ["engine_", "commit_coordinator"].concat(),
            ["committed", "_seq"].concat(),
            ["Generation", "Pending"].concat(),
            ["ReplayBase", "GenerationPin"].concat(),
            ["W", "AL"].concat(),
            ["re", "covery"].concat(),
            ["check", "point"].concat(),
            ["install", " hook"].concat(),
            ["com", "parator"].concat(),
            ["root ", "extraction"].concat(),
            ["type ", "grammar"].concat(),
            ["from_gpu_", "completion"].concat(),
        ] {
            assert!(
                !source.contains(&forbidden),
                "private publication kernel must not depend on {forbidden}"
            );
        }
        assert!(source.contains("struct PublicationAuthorityHolder"));
        assert!(source.contains("struct PlacementReplacementCandidate"));
        assert!(source.contains("ArcSwap<PublicationAuthorityGeneration>"));
        assert!(source.contains("compare_and_swap"));
        assert!(!source.contains(&["pub(", "crate)"].concat()));
    }
}
