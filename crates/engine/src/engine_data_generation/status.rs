//! Immutable terminal-status view shapes. This is a publication-owned lookup view, never a WAL.

use super::digest::{
    CommitSequence, DatabaseId, RequestDigest, ReturningDigest, StableTransactionId,
    StatusEntryLeafRoot, StatusViewRoot, TargetDigest, TerminalEnvelopeDigest,
};
use super::input::TerminalOutcome;
use super::manifest::{FixedRadixMap, GpuRadixEmptyRoots, GpuRadixPathCompletion, RadixLeafValue};
use super::DataGenerationError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PublishedStatusEntry {
    pub(super) transaction_id: StableTransactionId,
    pub(super) request_digest: RequestDigest,
    pub(super) commit_sequence: CommitSequence,
    pub(super) outcome: TerminalOutcome,
    pub(super) target_digest: TargetDigest,
    pub(super) returning_digest: ReturningDigest,
    pub(super) terminal_envelope_digest: TerminalEnvelopeDigest,
}

impl PublishedStatusEntry {
    pub(super) fn validate(&self) -> Result<(), DataGenerationError> {
        self.outcome.validate()
    }
}

#[derive(Clone, Debug)]
pub(super) struct StatusMapLeaf {
    pub(super) entry_leaf_root: StatusEntryLeafRoot,
    pub(super) entry: PublishedStatusEntry,
}

impl RadixLeafValue for StatusMapLeaf {
    fn same_commitment(&self, other: &Self) -> bool {
        self.entry_leaf_root == other.entry_leaf_root
            && self.entry.transaction_id == other.entry.transaction_id
    }
}

type StatusMap = FixedRadixMap<StableTransactionId, StatusViewRoot, StatusMapLeaf>;

#[derive(Clone, Debug)]
pub(super) struct GpuStatusPathCompletion {
    pub(super) entry_leaf_root: StatusEntryLeafRoot,
    pub(super) map_path: GpuRadixPathCompletion<StatusViewRoot>,
}

#[derive(Clone, Debug)]
pub(super) struct PublishedStatusIndex {
    database_id: DatabaseId,
    map: StatusMap,
}

impl PublishedStatusIndex {
    pub(super) fn genesis(
        database_id: DatabaseId,
        empty_roots: GpuRadixEmptyRoots<StatusViewRoot>,
    ) -> Result<Self, DataGenerationError> {
        Ok(Self {
            database_id,
            map: StatusMap::empty(empty_roots)?,
        })
    }

    pub(super) fn root(&self) -> StatusViewRoot {
        self.map.root()
    }

    pub(super) fn count(&self) -> u64 {
        self.map.count()
    }

    /// Private immutable lookup for publication-candidate closure. The durable stream remains
    /// authoritative; this only verifies that the captured publication view contains the exact
    /// terminal entry the candidate sealed.
    pub(super) fn entry(
        &self,
        transaction_id: StableTransactionId,
    ) -> Option<&PublishedStatusEntry> {
        self.map.get(transaction_id).map(|leaf| &leaf.entry)
    }

    pub(super) fn validate(&self) -> Result<(), DataGenerationError> {
        let _ = self.database_id;
        self.map.validate()?;
        self.map.validate_values(|transaction_id, leaf| {
            if transaction_id != leaf.entry.transaction_id {
                return Err(DataGenerationError::Invalid("status-map leaf identity"));
            }
            leaf.entry.validate()
        })
    }

    pub(super) fn insert(
        &self,
        entry: PublishedStatusEntry,
        completion: &GpuStatusPathCompletion,
    ) -> Result<Self, DataGenerationError> {
        entry.validate()?;
        let transaction_id = entry.transaction_id;
        let leaf = StatusMapLeaf {
            entry_leaf_root: completion.entry_leaf_root,
            entry,
        };
        let map = self
            .map
            .substitute(transaction_id, None, Some(leaf), &completion.map_path)?;
        Ok(Self {
            database_id: self.database_id,
            map,
        })
    }

    /// Test-only adversarial replacement that preserves the map's immutable shape while allowing
    /// candidate validation tests to carry a deliberately malformed terminal entry.
    #[cfg(test)]
    pub(super) fn replace_entry_unchecked_for_test(
        &self,
        entry: PublishedStatusEntry,
        completion: &GpuStatusPathCompletion,
    ) -> Result<Self, DataGenerationError> {
        let transaction_id = entry.transaction_id;
        let before = self
            .map
            .get(transaction_id)
            .cloned()
            .ok_or(DataGenerationError::Missing("test status entry"))?;
        let map = self.map.substitute(
            transaction_id,
            Some(&before),
            Some(StatusMapLeaf {
                entry_leaf_root: completion.entry_leaf_root,
                entry,
            }),
            &completion.map_path,
        )?;
        Ok(Self {
            database_id: self.database_id,
            map,
        })
    }
}
