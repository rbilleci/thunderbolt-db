//! Stable-identity table-access guards for the non-MVCC rewrite boundary.
//!
//! Ordinary table access is shared. A typed table reset upgrades the same transaction owner to
//! exclusive and retains that mode through terminal control. The registry never decides row or FK
//! semantics; it only closes the lifetime race between a retained root and a non-MVCC replacement.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableAccessMode {
    Shared,
    Exclusive,
}

#[derive(Debug, Default, Clone, Copy)]
struct TableAccessCounts {
    shared: usize,
    exclusive: usize,
}

impl TableAccessCounts {
    fn mode(self) -> Option<TableAccessMode> {
        if self.exclusive != 0 {
            Some(TableAccessMode::Exclusive)
        } else if self.shared != 0 {
            Some(TableAccessMode::Shared)
        } else {
            None
        }
    }

    fn add(&mut self, mode: TableAccessMode) {
        match mode {
            TableAccessMode::Shared => self.shared = self.shared.saturating_add(1),
            TableAccessMode::Exclusive => self.exclusive = self.exclusive.saturating_add(1),
        }
    }

    fn remove(&mut self, mode: TableAccessMode) {
        match mode {
            TableAccessMode::Shared => self.shared = self.shared.saturating_sub(1),
            TableAccessMode::Exclusive => self.exclusive = self.exclusive.saturating_sub(1),
        }
    }
}

#[derive(Debug, Default)]
struct TableAccessState {
    by_owner: BTreeMap<u64, BTreeMap<u32, TableAccessCounts>>,
    /// Synchronous one-table reads do not need an owner, upgrades, or a clonable token. Keeping
    /// their counts directly by OID avoids allocating the general retained-submission bookkeeping
    /// on every latency-oriented point-read batch.
    direct_shared: BTreeMap<u32, usize>,
}

#[derive(Debug, Default)]
pub(crate) struct TableAccessRegistry {
    next_owner: AtomicU64,
    state: Mutex<TableAccessState>,
}

impl TableAccessRegistry {
    pub(crate) fn lease(self: &Arc<Self>) -> Arc<TableAccessLease> {
        let owner = self
            .next_owner
            .fetch_add(1, AtomicOrdering::Relaxed)
            .saturating_add(1);
        Arc::new(TableAccessLease {
            registry: Arc::clone(self),
            owner,
            held: Mutex::new(BTreeMap::new()),
        })
    }

    fn acquire(
        &self,
        owner: u64,
        changes: &[(u32, Option<TableAccessMode>, TableAccessMode)],
        requested: TableAccessMode,
    ) -> Result<(), ExecuteError> {
        if changes.is_empty() {
            return Ok(());
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (table_oid, _, _) in changes {
            if requested == TableAccessMode::Exclusive
                && state.direct_shared.get(table_oid).copied().unwrap_or(0) != 0
            {
                return Err(ExecuteError::Serialization(format!(
                    "table identity {table_oid} has an incompatible retained access guard"
                )));
            }
            for (other_owner, held) in &state.by_owner {
                if *other_owner == owner {
                    continue;
                }
                let Some(held) = held.get(table_oid).and_then(|counts| counts.mode()) else {
                    continue;
                };
                if requested == TableAccessMode::Exclusive || held == TableAccessMode::Exclusive {
                    return Err(ExecuteError::Serialization(format!(
                        "table identity {table_oid} has an incompatible retained access guard"
                    )));
                }
            }
        }
        let held = state.by_owner.entry(owner).or_default();
        for (table_oid, previous, next) in changes {
            let counts = held.entry(*table_oid).or_default();
            if let Some(previous) = previous {
                counts.remove(*previous);
            }
            counts.add(*next);
        }
        Ok(())
    }

    pub(crate) fn acquire_direct_shared(
        self: &Arc<Self>,
        table_oid: u32,
    ) -> Result<DirectTableReadLease, ExecuteError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.by_owner.values().any(|held| {
            held.get(&table_oid).and_then(|counts| counts.mode())
                == Some(TableAccessMode::Exclusive)
        }) {
            return Err(ExecuteError::Serialization(format!(
                "table identity {table_oid} has an incompatible retained access guard"
            )));
        }
        let count = state.direct_shared.entry(table_oid).or_default();
        *count = count.saturating_add(1);
        Ok(DirectTableReadLease {
            registry: Arc::clone(self),
            table_oid,
        })
    }

    fn release_direct_shared(&self, table_oid: u32) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let remove = state
            .direct_shared
            .get_mut(&table_oid)
            .is_some_and(|count| {
                *count = count.saturating_sub(1);
                *count == 0
            });
        if remove {
            state.direct_shared.remove(&table_oid);
        }
    }

    fn release(&self, owner: u64, released: &BTreeMap<u32, TableAccessMode>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut remove_owner = false;
        if let Some(held) = state.by_owner.get_mut(&owner) {
            for (table_oid, mode) in released {
                if let Some(counts) = held.get_mut(table_oid) {
                    counts.remove(*mode);
                    if counts.mode().is_none() {
                        held.remove(table_oid);
                    }
                }
            }
            remove_owner = held.is_empty();
        }
        if remove_owner {
            state.by_owner.remove(&owner);
        }
    }
}

#[derive(Debug)]
pub(crate) struct DirectTableReadLease {
    registry: Arc<TableAccessRegistry>,
    table_oid: u32,
}

impl Drop for DirectTableReadLease {
    fn drop(&mut self) {
        self.registry.release_direct_shared(self.table_oid);
    }
}

#[derive(Debug)]
pub(crate) struct TableAccessLease {
    registry: Arc<TableAccessRegistry>,
    owner: u64,
    /// Identities owned by this particular token. Related tokens may share the logical owner, so
    /// a protocol proof can retain only its target closure after the parent transaction releases
    /// every other identity it accumulated.
    held: Mutex<BTreeMap<u32, TableAccessMode>>,
}

impl TableAccessLease {
    fn acquire(
        &self,
        table_oids: impl IntoIterator<Item = u32>,
        requested: TableAccessMode,
    ) -> Result<(), ExecuteError> {
        let table_oids = table_oids.into_iter().collect::<BTreeSet<_>>();
        let mut local = self
            .held
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let changes = table_oids
            .into_iter()
            .filter_map(|table_oid| {
                let previous = local.get(&table_oid).copied();
                let next = match (previous, requested) {
                    (Some(TableAccessMode::Exclusive), _) => return None,
                    (Some(TableAccessMode::Shared), TableAccessMode::Shared) => return None,
                    _ => requested,
                };
                Some((table_oid, previous, next))
            })
            .collect::<Vec<_>>();
        self.registry.acquire(self.owner, &changes, requested)?;
        for (table_oid, _, next) in changes {
            local.insert(table_oid, next);
        }
        Ok(())
    }

    pub(crate) fn acquire_shared(
        &self,
        table_oids: impl IntoIterator<Item = u32>,
    ) -> Result<(), ExecuteError> {
        self.acquire(table_oids, TableAccessMode::Shared)
    }

    pub(crate) fn acquire_exclusive(
        &self,
        table_oids: impl IntoIterator<Item = u32>,
    ) -> Result<(), ExecuteError> {
        self.acquire(table_oids, TableAccessMode::Exclusive)
    }

    /// Create a separately droppable token under the same logical owner. Same-owner compatibility
    /// lets this retain a shared target after the transaction owner upgraded that identity, while
    /// per-token accounting ensures terminal transaction cleanup releases unrelated/exclusive OIDs.
    pub(crate) fn retain_shared(
        &self,
        table_oids: impl IntoIterator<Item = u32>,
    ) -> Result<Arc<Self>, ExecuteError> {
        let retained = Arc::new(Self {
            registry: Arc::clone(&self.registry),
            owner: self.owner,
            held: Mutex::new(BTreeMap::new()),
        });
        retained.acquire_shared(table_oids)?;
        Ok(retained)
    }
}

impl Drop for TableAccessLease {
    fn drop(&mut self) {
        let held = self
            .held
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.registry.release(self.owner, held);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_guards_coexist_and_exclusive_upgrade_is_owner_reentrant() {
        let registry = Arc::new(TableAccessRegistry::default());
        let first = registry.lease();
        let second = registry.lease();
        first.acquire_shared([7]).unwrap();
        second.acquire_shared([7]).unwrap();
        assert!(matches!(
            first.acquire_exclusive([7]),
            Err(ExecuteError::Serialization(_))
        ));
        drop(second);
        first.acquire_exclusive([7]).unwrap();
        first.acquire_shared([7]).unwrap();
    }

    #[test]
    fn related_token_retains_only_its_shared_subset_after_parent_upgrade_and_drop() {
        let registry = Arc::new(TableAccessRegistry::default());
        let parent = registry.lease();
        parent.acquire_shared([7, 8]).unwrap();
        let proof = parent.retain_shared([7]).unwrap();
        parent.acquire_exclusive([7]).unwrap();
        drop(parent);

        let other = registry.lease();
        other.acquire_exclusive([8]).unwrap();
        assert!(matches!(
            other.acquire_exclusive([7]),
            Err(ExecuteError::Serialization(_))
        ));
        drop(proof);
        other.acquire_exclusive([7]).unwrap();
    }

    #[test]
    fn multi_table_acquisition_is_all_or_nothing() {
        let registry = Arc::new(TableAccessRegistry::default());
        let reset = registry.lease();
        let reader = registry.lease();
        reset.acquire_exclusive([11]).unwrap();
        assert!(reader.acquire_shared([10, 11]).is_err());
        drop(reset);
        reader.acquire_exclusive([10, 11]).unwrap();
    }

    #[test]
    fn direct_shared_guard_conflicts_with_exclusive_in_both_orders() {
        let registry = Arc::new(TableAccessRegistry::default());
        let direct = registry.acquire_direct_shared(17).unwrap();
        let reset = registry.lease();
        assert!(matches!(
            reset.acquire_exclusive([17]),
            Err(ExecuteError::Serialization(_))
        ));
        drop(direct);
        reset.acquire_exclusive([17]).unwrap();
        assert!(matches!(
            registry.acquire_direct_shared(17),
            Err(ExecuteError::Serialization(_))
        ));
    }
}
