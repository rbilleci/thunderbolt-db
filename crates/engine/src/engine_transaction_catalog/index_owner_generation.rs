//! Exact public owner-generation fence for transactional `CREATE UNIQUE INDEX`.
//!
//! Index lifecycle takes exclusive stable-table access for the rest of the explicit transaction.
//! Snapshot capture necessarily precedes that acquisition, however, so a concurrent DML owner can
//! publish and release its shared access in between. UNIQUE validation may consume transaction-
//! private DML, but its inherited public base must still be the exact live owner generation once
//! exclusive access linearizes. Otherwise validating the retained base could certify a key set
//! that no longer exists and COMMIT has no row write-set through which to discover that conflict.

use super::*;

#[cfg(test)]
type IndexOwnerPreAcquireHook = (usize, Arc<std::sync::Barrier>, Arc<std::sync::Barrier>);

#[cfg(test)]
fn index_owner_pre_acquire_hook() -> &'static std::sync::Mutex<Option<IndexOwnerPreAcquireHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<IndexOwnerPreAcquireHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

fn same_residency_entry(
    captured: Option<&RelationalResidencyEntry>,
    live: Option<&RelationalResidencyEntry>,
) -> bool {
    match (captured, live) {
        (None, None) => true,
        (Some(captured), Some(live)) => {
            let same_memory = match (&captured.device_memory, &live.device_memory) {
                (None, None) => true,
                (Some(captured), Some(live)) => Arc::ptr_eq(captured, live),
                _ => false,
            };
            captured.descriptor.as_ref() == live.descriptor.as_ref() && same_memory
        }
        _ => false,
    }
}

impl Engine {
    #[cfg(test)]
    pub(crate) fn set_index_owner_pre_acquire_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *index_owner_pre_acquire_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    pub(super) fn run_index_owner_pre_acquire_hook(&self) {
        let hook = {
            let mut hook = index_owner_pre_acquire_hook()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            hook.as_ref()
                .is_some_and(|(owner, _, _)| *owner == self as *const Self as usize)
                .then(|| hook.take())
                .flatten()
        };
        if let Some((_, reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    /// Prove that the retained PUBLIC base used by UNIQUE validation is still the exact live owner
    /// generation after this transaction has acquired exclusive stable-table access. The private
    /// shard/cold overlay is deliberately excluded: the validation path merges it over this base.
    ///
    /// Host MVCC alone is insufficient. Device-authoritative tables can freeze/elide that host
    /// generation, while chunk-authoritative tables publish their record of truth as cold-entry
    /// COW generations. Conversely a store-authoritative cold entry is only a rebuildable cache, so
    /// its Arc may churn without changing rows and is not part of this fence.
    pub(crate) fn validate_transaction_unique_index_owner_generation(
        &self,
        snapshot: &TransactionSnapshot,
        create: &CreateIndex,
    ) -> Result<(), ExecuteError> {
        if !create.unique
            || !snapshot
                .catalog
                .relational_catalog
                .contains_key(&create.table)
        {
            // A table introduced by this transaction has no concurrent public owner generation.
            return Ok(());
        }

        // Device tombstone contents can change in place after their first sidecar allocation:
        // descriptor fields and every resource Arc then remain identical. The non-pruned table
        // high-water is the logical content-generation token recorded by every committed DML
        // shape, including those in-place stamps. Exclusive owner access makes this sample stable.
        let table_oid = snapshot
            .catalog
            .relational_catalog
            .get(&create.table)
            .expect("existing CREATE UNIQUE target was checked above")
            .oid;
        if self
            .commit_state()
            .ledger
            .table_changed_after(table_oid, snapshot.boundary)
        {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{}\" changed after CREATE UNIQUE snapshot boundary {}",
                create.table, snapshot.boundary
            )));
        }

        let captured_table = snapshot.table_versions.get(&create.table);
        let live_table = self.read_state.mvcc.load_table(&create.table);
        let same_table = match (captured_table, live_table.as_ref()) {
            (None, None) => true,
            (Some(captured), Some(live)) => {
                captured.generation() == live.generation()
                    && Arc::ptr_eq(captured.get(), live.get())
            }
            _ => false,
        };

        let residency = &self.read_state.residency;
        let live_snapshots = residency.snapshots.load_full();
        let live_shards = residency.shards.load_full();
        let live_device_authority = residency.device_authoritative_tables.load_full();
        let live_chunk_authority = residency.chunk_authoritative_tables.load_full();
        let captured_chunk_boundary = snapshot
            .chunk_authoritative_tables
            .get(&create.table)
            .copied();
        let live_chunk_boundary = live_chunk_authority.get(&create.table).copied();
        let same_chunk_generation = match (captured_chunk_boundary, live_chunk_boundary) {
            (None, None) => true,
            (Some(captured_boundary), Some(live_boundary))
                if captured_boundary == live_boundary =>
            {
                let captured_cold = snapshot.base_streaming_cold_chunks.get(&create.table);
                let live_cold = residency.streaming_cold_chunks.load_full();
                match (captured_cold, live_cold.get(&create.table)) {
                    (Some(captured), Some(live)) => Arc::ptr_eq(captured, live),
                    _ => false,
                }
            }
            _ => false,
        };
        let same_generation = same_table
            && snapshot.device_authoritative_tables.contains(&create.table)
                == live_device_authority.contains(&create.table)
            && snapshot.resident_shards.get(&create.table) == live_shards.get(&create.table)
            && same_residency_entry(
                snapshot.resident_snapshots.get(&create.table),
                live_snapshots.get(&create.table),
            )
            && same_chunk_generation;
        if same_generation {
            return Ok(());
        }

        Err(ExecuteError::Serialization(format!(
            "relation \"{}\" changed its public storage generation between CREATE UNIQUE snapshot capture and exclusive access",
            create.table
        )))
    }
}
