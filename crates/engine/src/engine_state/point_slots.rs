//! Table-scoped retained point-route and mutation-epoch ownership.
//!
//! The outer directory is the single name-to-slot authority. A slot is stable for one catalog OID:
//! rename moves its Arc, DROP/recreate replaces it, and both point-route families plus the index
//! mutation epoch are retired together without rebuilding a global route map.

use super::*;

/// Stable per-table ownership for both retained point-route families and the in-place index
/// mutation epoch. Readers retain a route plan, never a mutable map borrow.
#[derive(Debug)]
pub(crate) struct TablePointSlot {
    pub(crate) table_oid: u32,
    /// A separately retained identity lets callers prove their captured slot was not replaced.
    pub(crate) slot_identity: Arc<()>,
    pub(crate) index_epoch: Arc<std::sync::atomic::AtomicU64>,
    /// One latest projection shape per table, preserving the bounded replacement policy.
    pub(crate) sharded_route: ArcSwapOption<CachedShardedPointRoute>,
    /// READ-002 remains a separate route family but shares table identity and retirement.
    pub(crate) compound_route: ArcSwapOption<CachedCompoundI32I64PointRoute>,
}

impl TablePointSlot {
    pub(crate) fn new(table_oid: u32) -> Self {
        Self {
            table_oid,
            slot_identity: Arc::new(()),
            index_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            sharded_route: ArcSwapOption::empty(),
            compound_route: ArcSwapOption::empty(),
        }
    }
}

fn table_point_slot_has_route(slot: &TablePointSlot) -> bool {
    slot.sharded_route.load().is_some() || slot.compound_route.load().is_some()
}

/// A table that already owns either route family consumes an existing retained slot. Only the
/// first route for a previously empty table can require retirement elsewhere.
fn table_point_route_requires_eviction(populated_slots: usize, target_populated: bool) -> bool {
    !target_populated
        && populated_slots >= crate::engine_retained_read::MAX_CACHED_SHARDED_POINT_ROUTES
}

/// BTreeMap iteration fixes the global victim order. Keeping this selection separate makes the
/// capacity decision testable without constructing GPU route plans.
fn first_other_populated_point_route_slot<'a, T>(
    table: &str,
    mut slots: impl Iterator<Item = (&'a String, &'a T)>,
    is_populated: impl Fn(&T) -> bool,
) -> Option<(&'a String, &'a T)> {
    slots.find(|(name, slot)| name.as_str() != table && is_populated(slot))
}

fn catalog_contains_exact_table(read_state: &ReadState, table: &RelationalTable) -> bool {
    read_state
        .latest_catalog()
        .relational_catalog
        .get(&table.name)
        .is_some_and(|current| current == table)
}

impl ResidencyReadState {
    /// Exact stable-identity epoch acquisition. The current published catalog must still name
    /// this OID before an empty directory may receive a new slot, closing stale-old-table creation
    /// after DROP/recreate (or rename) before the replacement has used a point route.
    pub(crate) fn point_index_mutation_epoch_for_table(
        &self,
        read_state: &ReadState,
        table: &RelationalTable,
    ) -> Option<Arc<std::sync::atomic::AtomicU64>> {
        self.ensure_table_point_slot(read_state, table)
            .map(|slot| Arc::clone(&slot.index_epoch))
    }

    /// Acquire the writer side of the table's exact epoch. A stale catalog relation never shares
    /// an epoch with a same-name replacement: currentness is checked before and after the CAS,
    /// and an ABA that wins that narrow interval is released before this returns `None`.
    pub(crate) fn begin_point_index_mutation_for_table(
        &self,
        read_state: &ReadState,
        table: &RelationalTable,
    ) -> Option<PointIndexMutationGuard> {
        let slot = self.ensure_table_point_slot(read_state, table)?;
        let slot_identity = Arc::clone(&slot.slot_identity);
        loop {
            if !catalog_contains_exact_table(read_state, table)
                || !self.table_point_slot_is_current(&table.name, table.oid, &slot, &slot_identity)
            {
                return None;
            }
            let current = slot.index_epoch.load(std::sync::atomic::Ordering::Acquire);
            if current == POINT_INDEX_MUTATION_POISON {
                return None;
            }
            if current & 1 != 0 {
                std::thread::yield_now();
                continue;
            }
            let writing = current.checked_add(1)?;
            current.checked_add(2)?;
            if slot
                .index_epoch
                .compare_exchange_weak(
                    current,
                    writing,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            let guard = PointIndexMutationGuard {
                epoch: Arc::clone(&slot.index_epoch),
            };
            if catalog_contains_exact_table(read_state, table)
                && self.table_point_slot_is_current(&table.name, table.oid, &slot, &slot_identity)
            {
                return Some(guard);
            }
            drop(guard);
            return None;
        }
    }

    /// Load the exact current table slot. OID plus outer Arc identity closes same-name ABA.
    pub(crate) fn table_point_slot(
        &self,
        table: &str,
        table_oid: u32,
    ) -> Option<Arc<TablePointSlot>> {
        self.table_point_slots
            .load_full()
            .get(table)
            .filter(|slot| slot.table_oid == table_oid)
            .cloned()
    }

    /// Install a slot only when the directory has no conflicting table identity. Expensive GPU
    /// preparation happens after this small structural publication; final publication rechecks.
    pub(crate) fn ensure_table_point_slot(
        &self,
        read_state: &ReadState,
        table: &RelationalTable,
    ) -> Option<Arc<TablePointSlot>> {
        if !catalog_contains_exact_table(read_state, table) {
            return None;
        }
        if let Some(slot) = self.table_point_slot(&table.name, table.oid) {
            return Some(slot);
        }
        if self.table_point_slots.load().contains_key(&table.name) {
            return None;
        }
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Recheck the current catalog while route publication is serialized. If DDL has replaced
        // this same name/OID since the first read, never install the stale slot into an empty map.
        if !catalog_contains_exact_table(read_state, table) {
            return None;
        }
        let slot = self.ensure_table_point_slot_under_publish_lock(&table.name, table.oid)?;
        if catalog_contains_exact_table(read_state, table) {
            return Some(slot);
        }
        // A DDL publisher raced the final catalog recheck. We still hold the same route lock it
        // needs for DROP/rename retirement, so retire only this exact slot before reporting a
        // stale capture. A replacement OID can never be removed by this cleanup.
        let current = self.table_point_slots.load();
        if current
            .get(&table.name)
            .is_some_and(|current| Arc::ptr_eq(current, &slot) && current.table_oid == table.oid)
        {
            let mut next = (**current).clone();
            next.remove(&table.name);
            slot.sharded_route.store(None);
            slot.compound_route.store(None);
            self.table_point_slots.store(Arc::new(next));
        }
        None
    }

    fn ensure_table_point_slot_under_publish_lock(
        &self,
        table: &str,
        table_oid: u32,
    ) -> Option<Arc<TablePointSlot>> {
        if let Some(slot) = self.table_point_slot(table, table_oid) {
            return Some(slot);
        }
        if self.table_point_slots.load().contains_key(table) {
            return None;
        }
        let slot = Arc::new(TablePointSlot::new(table_oid));
        let mut next = (**self.table_point_slots.load()).clone();
        next.insert(table.to_string(), Arc::clone(&slot));
        self.table_point_slots.store(Arc::new(next));
        Some(slot)
    }

    /// Reconcile the rare name-to-slot directory against the prior and next immutable catalogs.
    /// The caller holds
    /// `sharded_point_route_publish_lock` continuously through this reconciliation and the catalog
    /// history store. Slots are never created here: a table without an existing retained route or
    /// epoch owner remains absent until a current reader needs one.
    ///
    /// A rename is identified by its stable OID, not its old name. When the moved slot is the only
    /// matching owner, its Arc, identity, and epoch move to the catalog name, but both descriptor
    /// route families retire because their route shape still names the old table. Conflicting
    /// same-OID entries select the exact-name owner first and retire every loser, so one catalog
    /// relation can never retain two independently mutable point-slot epochs.
    pub(crate) fn reconcile_table_point_slots_under_publish_lock(
        &self,
        prior: &CatalogSnapshot,
        generation: &CatalogSnapshot,
    ) {
        let current = self.table_point_slots.load_full();
        if current.is_empty() {
            return;
        }

        let mut next = BTreeMap::new();
        let mut retained = BTreeSet::new();
        for (name, table) in &generation.relational_catalog {
            let exact = current
                .get(name)
                .filter(|slot| slot.table_oid == table.oid)
                .cloned();
            let moved = exact
                .is_none()
                .then(|| {
                    current
                        .iter()
                        .filter(|(_, slot)| slot.table_oid == table.oid)
                        .filter(|(_, slot)| !retained.contains(&Arc::as_ptr(slot)))
                        .min_by_key(|(candidate, _)| *candidate)
                        .map(|(candidate, slot)| (candidate.as_str(), Arc::clone(slot)))
                })
                .flatten();
            let Some((source_name, slot)) = exact.map(|slot| (name.as_str(), slot)).or(moved)
            else {
                continue;
            };
            let table_shape_changed = prior.relational_catalog.get(name) != Some(table);
            if source_name != name.as_str() || table_shape_changed {
                // A route embeds the table name/generation. Preserve this table's epoch owner on
                // rename or same-OID shape change but retire both stale route families before
                // publishing its new key.
                slot.sharded_route.store(None);
                slot.compound_route.store(None);
            }
            retained.insert(Arc::as_ptr(&slot));
            next.insert(name.clone(), slot);
        }

        for (name, slot) in current.iter() {
            let kept_here = next.get(name).is_some_and(|kept| Arc::ptr_eq(kept, slot));
            if !kept_here {
                // Includes absent DROPs, same-name OID replacement, stale pre-publish ensures,
                // and duplicate same-OID entries left by a raced early rename. Retained readers
                // keep their Arc, but no future lookup can publish either route family through it.
                slot.sharded_route.store(None);
                slot.compound_route.store(None);
            }
        }
        self.table_point_slots.store(Arc::new(next));
    }

    /// Final route publication calls this while holding
    /// `sharded_point_route_publish_lock`. That is the same lock held by catalog publication
    /// across slot reconciliation and history storage, so a route can neither publish for a
    /// dropped OID nor cross a rename/recreate catalog boundary after its expensive GPU build.
    pub(crate) fn published_catalog_contains_exact_table_under_publish_lock(
        &self,
        read_state: &ReadState,
        table: &RelationalTable,
    ) -> bool {
        catalog_contains_exact_table(read_state, table)
    }

    /// Recheck the table directory while route publication is serialized. Descriptor generation
    /// alone is insufficient because a dropped relation can be recreated under the same name.
    pub(crate) fn table_point_slot_is_current(
        &self,
        table: &str,
        table_oid: u32,
        expected_slot: &Arc<TablePointSlot>,
        expected_identity: &Arc<()>,
    ) -> bool {
        self.table_point_slots
            .load_full()
            .get(table)
            .is_some_and(|current| {
                current.table_oid == table_oid
                    && Arc::ptr_eq(current, expected_slot)
                    && Arc::ptr_eq(&current.slot_identity, expected_identity)
            })
    }

    /// Reserve one populated table slot under the existing route-publish lock. The lexical victim
    /// is deterministic; clearing both route families keeps ownership table-scoped and bounded.
    pub(crate) fn reserve_table_point_route_under_publish_lock(
        &self,
        table: &str,
        table_oid: u32,
        expected_slot: &Arc<TablePointSlot>,
        expected_identity: &Arc<()>,
    ) -> bool {
        if !self.table_point_slot_is_current(table, table_oid, expected_slot, expected_identity) {
            return false;
        }
        let target_populated = table_point_slot_has_route(expected_slot);
        if target_populated {
            return true;
        }
        let slots = self.table_point_slots.load_full();
        let populated = slots
            .values()
            .filter(|slot| table_point_slot_has_route(slot))
            .count();
        if !table_point_route_requires_eviction(populated, target_populated) {
            return true;
        }
        let Some((_, victim)) =
            first_other_populated_point_route_slot(table, slots.iter(), |slot| {
                table_point_slot_has_route(slot)
            })
        else {
            return false;
        };
        // Swapping `None` never clones a BTreeMap or allocates on the retirement path.
        victim.sharded_route.store(None);
        victim.compound_route.store(None);
        true
    }

    /// Allocation-free inner retirement. Readers that captured a plan retain that Arc; new hits miss.
    pub(crate) fn purge_table_point_routes_under_publish_lock(&self, table: &str) {
        if let Some(slot) = self.table_point_slots.load().get(table) {
            slot.sharded_route.store(None);
            slot.compound_route.store(None);
        }
    }

    pub(crate) fn sharded_point_route_count(&self) -> usize {
        self.table_point_slots
            .load_full()
            .values()
            .filter(|slot| slot.sharded_route.load().is_some())
            .count()
    }

    #[cfg(test)]
    pub(crate) fn compound_point_route_count(&self) -> usize {
        self.table_point_slots
            .load()
            .values()
            .filter(|slot| slot.compound_route.load().is_some())
            .count()
    }

    pub(crate) fn sharded_point_route_descriptor_bytes_for_gpu(&self, gpu_id: u16) -> u64 {
        self.table_point_slots
            .load()
            .values()
            .filter_map(|slot| slot.sharded_route.load_full())
            .filter(|route| route.gpu_id == gpu_id)
            .map(|route| route.plan.descriptor_allocated_bytes())
            .sum()
    }

    pub(crate) fn sharded_point_route_descriptor_bytes_for_gpu_excluding(
        &self,
        gpu_id: u16,
        table: &str,
    ) -> u64 {
        self.table_point_slots
            .load()
            .iter()
            .filter(|(name, _)| name.as_str() != table)
            .filter_map(|(_, slot)| slot.sharded_route.load_full())
            .filter(|route| route.gpu_id == gpu_id)
            .map(|route| route.plan.descriptor_allocated_bytes())
            .sum()
    }

    pub(crate) fn sharded_point_route_descriptor_bytes_for_table_gpu(
        &self,
        table: &str,
        gpu_id: u16,
    ) -> u64 {
        self.table_point_slots
            .load()
            .get(table)
            .and_then(|slot| slot.sharded_route.load_full())
            .filter(|route| route.gpu_id == gpu_id)
            .map_or(0, |route| route.plan.descriptor_allocated_bytes())
    }

    #[cfg(test)]
    pub(crate) fn reset_point_routes_for_test(&self) {
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for slot in self.table_point_slots.load().values() {
            slot.sharded_route.store(None);
            slot.compound_route.store(None);
        }
    }

    #[cfg(test)]
    pub(crate) fn clear_sharded_point_routes_for_test(&self) {
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for slot in self.table_point_slots.load().values() {
            slot.sharded_route.store(None);
        }
    }

    #[cfg(test)]
    pub(crate) fn sharded_point_route_for_table(
        &self,
        table: &str,
    ) -> Option<Arc<CachedShardedPointRoute>> {
        self.table_point_slots
            .load()
            .get(table)
            .and_then(|slot| slot.sharded_route.load_full())
    }

    #[cfg(test)]
    pub(crate) fn compound_point_route_for_table(
        &self,
        table: &str,
    ) -> Option<Arc<CachedCompoundI32I64PointRoute>> {
        self.table_point_slots
            .load()
            .get(table)
            .and_then(|slot| slot.compound_route.load_full())
    }

    /// Remove a dropped table's slot only while its stable catalog OID still agrees.
    pub(crate) fn remove_table_point_slot(&self, table: &str, table_oid: u32) {
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.table_point_slots.load();
        if !current
            .get(table)
            .is_some_and(|slot| slot.table_oid == table_oid)
        {
            return;
        }
        let mut next = (**current).clone();
        if let Some(slot) = next.remove(table) {
            slot.sharded_route.store(None);
            slot.compound_route.store(None);
        }
        self.table_point_slots.store(Arc::new(next));
    }

    /// Rename preserves the OID, slot identity, and mutation epoch while retiring route plans.
    pub(crate) fn rename_table_point_slot(&self, old: &str, new: &str, table_oid: u32) {
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.table_point_slots.load();
        let Some(slot) = current
            .get(old)
            .filter(|slot| slot.table_oid == table_oid)
            .cloned()
        else {
            return;
        };
        debug_assert!(
            !current.contains_key(new),
            "DDL validated that a table rename cannot replace another point slot"
        );
        slot.sharded_route.store(None);
        slot.compound_route.store(None);
        let mut next = (**current).clone();
        next.remove(old);
        next.insert(new.to_string(), slot);
        self.table_point_slots.store(Arc::new(next));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relational_model::RelationalIndex;
    use std::sync::atomic::Ordering;

    fn table(name: &str, oid: u32) -> RelationalTable {
        RelationalTable {
            schema: "public".to_string(),
            name: name.to_string(),
            oid,
            columns: Vec::new(),
            indexes: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
            acl: BTreeMap::new(),
        }
    }

    fn catalog(table: &RelationalTable) -> CatalogSnapshot {
        let mut catalog = CatalogSnapshot::default();
        catalog
            .relational_catalog
            .insert(table.name.clone(), table.clone());
        catalog
    }

    fn catalog_for_tables(commit_seq: Index, tables: &[RelationalTable]) -> Arc<CatalogSnapshot> {
        let mut catalog = CatalogSnapshot {
            commit_seq,
            ..CatalogSnapshot::default()
        };
        for table in tables {
            catalog
                .relational_catalog
                .insert(table.name.clone(), table.clone());
        }
        Arc::new(catalog)
    }

    fn read_state(table: &RelationalTable) -> ReadState {
        let state = ReadState::new();
        replace_catalog(&state, table);
        state
    }

    fn replace_catalog(state: &ReadState, table: &RelationalTable) {
        state.catalog_history.store(Arc::new(CatalogHistory {
            generations: vec![Arc::new(catalog(table))],
        }));
    }

    #[test]
    fn exact_epoch_capture_installs_one_oid_bound_slot() {
        let exact = table("inert", 41);
        let state = read_state(&exact);
        let residency = &state.residency;
        assert!(residency.table_point_slots.load().is_empty());
        let epoch = residency
            .point_index_mutation_epoch_for_table(&state, &exact)
            .expect("first exact capture installs the table slot");
        assert_eq!(epoch.load(Ordering::Acquire), 0);
        let slot = residency
            .table_point_slot("inert", 41)
            .expect("exact route publication can use the first exact capture");
        assert_eq!(slot.table_oid, 41);
        assert!(
            Arc::ptr_eq(&epoch, &slot.index_epoch),
            "the exact capture and route publication share one epoch owner"
        );
        let guard = residency
            .begin_point_index_mutation_for_table(&state, &exact)
            .expect("an exact current table acquires its own slot epoch");
        assert_eq!(epoch.load(Ordering::Acquire), 1);
        drop(guard);
        assert_eq!(epoch.load(Ordering::Acquire), 2);
    }

    #[test]
    fn same_oid_catalog_index_shape_change_rejects_a_stale_prebuilt_point_owner() {
        let before = table("same_oid_index_shape", 43);
        let state = read_state(&before);
        let residency = &state.residency;
        let prebuilt_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &before)
            .expect("the old catalog shape captures its point owner before DDL");
        let prebuilt_slot = residency
            .table_point_slot("same_oid_index_shape", 43)
            .expect("the old catalog shape owns a stable slot");
        let prebuilt_identity = Arc::clone(&prebuilt_slot.slot_identity);

        // ALTER-like catalog publication keeps the relation OID but adds an index.  The point
        // slot is intentionally retained for the current shape, so name+OID alone would let an
        // old expensive route or mutation publish through this exact same Arc.
        let mut after = before.clone();
        after.indexes.push(RelationalIndex {
            oid: 431,
            name: "same_oid_index_shape_id_key".to_string(),
            table: after.name.clone(),
            column: "id".to_string(),
            key_columns: vec!["id".to_string()],
            unique: true,
            primary_key: false,
            unique_constraint: true,
        });
        state.publish_catalog_generation_with_point_slot_fence(
            catalog_for_tables(1, &[after.clone()]),
            0,
        );
        let retained_slot = residency
            .table_point_slot("same_oid_index_shape", 43)
            .expect("same-OID catalog publication retains the structural slot");
        assert!(Arc::ptr_eq(&prebuilt_slot, &retained_slot));
        assert!(Arc::ptr_eq(
            &prebuilt_identity,
            &retained_slot.slot_identity
        ));
        assert!(Arc::ptr_eq(&prebuilt_epoch, &retained_slot.index_epoch));

        // The exact final-publish recheck runs under this same lock in both route families.
        let publish = residency
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            !residency.published_catalog_contains_exact_table_under_publish_lock(&state, &before),
            "a stale same-OID prebuilt route cannot pass its final catalog cut"
        );
        drop(publish);
        assert!(
            residency.ensure_table_point_slot(&state, &before).is_none(),
            "a stale same-OID point mutation cannot reacquire the retained slot"
        );
        assert!(
            residency
                .begin_point_index_mutation_for_table(&state, &before)
                .is_none(),
            "a stale same-OID point mutation cannot begin publication"
        );

        let current = residency
            .begin_point_index_mutation_for_table(&state, &after)
            .expect("the new full catalog shape still uses the retained current slot");
        assert_eq!(prebuilt_epoch.load(Ordering::Acquire), 1);
        drop(current);
        assert_eq!(prebuilt_epoch.load(Ordering::Acquire), 2);
    }

    #[test]
    fn terminal_point_epoch_fails_closed_without_spinning() {
        let exact = table("poisoned", 42);
        let state = read_state(&exact);
        let epoch = state
            .residency
            .point_index_mutation_epoch_for_table(&state, &exact)
            .expect("exact current table installs a point slot");
        epoch.store(POINT_INDEX_MUTATION_POISON, Ordering::Release);
        assert!(state
            .residency
            .begin_point_index_mutation_for_table(&state, &exact)
            .is_none());
    }

    #[test]
    fn rename_moves_slot_arc_and_drop_recreate_rejects_aba() {
        let old_table = table("before", 51);
        let state = read_state(&old_table);
        let residency = &state.residency;
        let old_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &old_table)
            .unwrap();
        let old_slot = residency.table_point_slot("before", 51).unwrap();
        let old_identity = Arc::clone(&old_slot.slot_identity);

        residency.rename_table_point_slot("before", "after", 51);
        let renamed_table = table("after", 51);
        replace_catalog(&state, &renamed_table);
        let renamed = residency.table_point_slot("after", 51).unwrap();
        assert!(Arc::ptr_eq(&old_slot, &renamed));
        assert!(Arc::ptr_eq(&old_epoch, &renamed.index_epoch));
        assert!(Arc::ptr_eq(&old_identity, &renamed.slot_identity));

        residency.remove_table_point_slot("after", 51);
        let recreated_table = table("after", 52);
        replace_catalog(&state, &recreated_table);
        assert!(residency
            .begin_point_index_mutation_for_table(&state, &renamed_table)
            .is_none());
        assert!(
            residency.table_point_slots.load().is_empty(),
            "a stale old OID may not install into an empty post-DROP directory"
        );
        let recreated_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &recreated_table)
            .unwrap();
        let recreated = residency.table_point_slot("after", 52).unwrap();
        assert!(!Arc::ptr_eq(&old_slot, &recreated));
        assert!(!Arc::ptr_eq(&old_epoch, &recreated_epoch));
        assert!(
            !residency.table_point_slot_is_current("after", 51, &old_slot, &old_identity),
            "a stale publication cannot target the recreated relation"
        );
        assert!(residency.table_point_slot_is_current(
            "after",
            52,
            &recreated,
            &recreated.slot_identity
        ));
        assert!(residency
            .begin_point_index_mutation_for_table(&state, &renamed_table)
            .is_none());
        let recreated_guard = residency
            .begin_point_index_mutation_for_table(&state, &recreated_table)
            .expect("the new OID acquires only its own fresh epoch");
        assert_eq!(recreated_epoch.load(Ordering::Acquire), 1);
        drop(recreated_guard);
        assert_eq!(recreated_epoch.load(Ordering::Acquire), 2);
    }

    #[test]
    fn catalog_publish_retires_stale_drop_slot_before_same_name_recreate() {
        let dropped = table("drop_slot_fence", 61);
        let state = read_state(&dropped);
        let residency = &state.residency;
        let original_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &dropped)
            .expect("the original table owns a slot before DROP");
        let original = residency
            .table_point_slot("drop_slot_fence", 61)
            .expect("original slot");
        let original_identity = Arc::clone(&original.slot_identity);

        // `apply_drop_table` retires the directory slot before its caller publishes the immutable
        // no-table catalog. Force an old-catalog reader through that gap.
        residency.remove_table_point_slot("drop_slot_fence", 61);
        let stale_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &dropped)
            .expect("the still-published old catalog admits the forced stale ensure");
        let stale = residency
            .table_point_slot("drop_slot_fence", 61)
            .expect("forced stale slot is installed before catalog publication");
        assert!(!Arc::ptr_eq(&original, &stale));

        // This is the production publication helper: reconciliation and immutable catalog storage
        // share the route-publish lock, so the stale old-OID entry cannot survive the DROP cut.
        state.publish_catalog_generation_with_point_slot_fence(catalog_for_tables(1, &[]), 0);
        assert!(residency.table_point_slots.load().is_empty());
        assert!(stale.sharded_route.load().is_none());
        assert!(stale.compound_route.load().is_none());

        let recreated_table = table("drop_slot_fence", 62);
        state.publish_catalog_generation_with_point_slot_fence(
            catalog_for_tables(2, std::slice::from_ref(&recreated_table)),
            0,
        );
        let recreated_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &recreated_table)
            .expect("the same-name replacement receives a current empty slot");
        let recreated = residency
            .table_point_slot("drop_slot_fence", 62)
            .expect("replacement slot");
        assert!(!Arc::ptr_eq(&original, &recreated));
        assert!(!Arc::ptr_eq(&stale, &recreated));
        assert!(!Arc::ptr_eq(&original_epoch, &recreated_epoch));
        assert!(!Arc::ptr_eq(&stale_epoch, &recreated_epoch));
        assert!(!Arc::ptr_eq(&original_identity, &recreated.slot_identity));
        assert!(residency.table_point_slot("drop_slot_fence", 61).is_none());
        assert!(residency.table_point_slot_is_current(
            "drop_slot_fence",
            62,
            &recreated,
            &recreated.slot_identity
        ));
    }

    #[test]
    fn catalog_publish_reconciles_stale_old_name_after_early_rename() {
        let before = table("rename_slot_before", 71);
        let state = read_state(&before);
        let residency = &state.residency;
        let original_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &before)
            .expect("the original table owns a slot before rename");
        let original = residency
            .table_point_slot("rename_slot_before", 71)
            .expect("original slot");
        let original_identity = Arc::clone(&original.slot_identity);

        // `apply_rename_table` moves the original Arc before publishing B. While A remains in the
        // old catalog, force a second stale A slot with the same OID into the directory.
        residency.rename_table_point_slot("rename_slot_before", "rename_slot_after", 71);
        let stale_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &before)
            .expect("the stale A ensure is admitted by the old catalog before publication");
        let stale = residency
            .table_point_slot("rename_slot_before", 71)
            .expect("forced stale A slot");
        assert!(!Arc::ptr_eq(&original, &stale));

        let after = table("rename_slot_after", 71);
        state.publish_catalog_generation_with_point_slot_fence(
            catalog_for_tables(1, std::slice::from_ref(&after)),
            0,
        );
        let retained = residency
            .table_point_slot("rename_slot_after", 71)
            .expect("B retains the original stable table slot");
        assert!(Arc::ptr_eq(&original, &retained));
        assert!(Arc::ptr_eq(&original_epoch, &retained.index_epoch));
        assert!(Arc::ptr_eq(&original_identity, &retained.slot_identity));
        assert!(residency
            .table_point_slot("rename_slot_before", 71)
            .is_none());
        assert!(stale.sharded_route.load().is_none());
        assert!(stale.compound_route.load().is_none());
        assert_eq!(
            residency
                .table_point_slots
                .load()
                .values()
                .filter(|slot| slot.table_oid == 71)
                .count(),
            1,
            "the catalog cut leaves no conflicting same-OID slot"
        );

        let replacement_a = table("rename_slot_before", 72);
        state.publish_catalog_generation_with_point_slot_fence(
            catalog_for_tables(2, &[after, replacement_a.clone()]),
            0,
        );
        let replacement_epoch = residency
            .point_index_mutation_epoch_for_table(&state, &replacement_a)
            .expect("the newly created A installs its own fresh slot");
        let replacement = residency
            .table_point_slot("rename_slot_before", 72)
            .expect("new A slot");
        assert!(!Arc::ptr_eq(&original, &replacement));
        assert!(!Arc::ptr_eq(&stale, &replacement));
        assert!(!Arc::ptr_eq(&original_epoch, &replacement_epoch));
        assert!(!Arc::ptr_eq(&stale_epoch, &replacement_epoch));
        assert!(!Arc::ptr_eq(&original_identity, &replacement.slot_identity));
    }

    #[test]
    fn unchanged_catalog_publication_keeps_the_existing_slot_directory() {
        let table = table("unchanged_slot_catalog", 81);
        let state = read_state(&table);
        let residency = &state.residency;
        let epoch = residency
            .point_index_mutation_epoch_for_table(&state, &table)
            .expect("current catalog installs the first slot");
        let slot = residency
            .table_point_slot("unchanged_slot_catalog", 81)
            .expect("first slot");
        let directory_before = residency.table_point_slots.load_full();

        // DML republishes catalog history with a new commit sequence but does not alter table
        // metadata. It must not take the slot-fence reconciliation path or COW-replace the
        // directory that retained route readers already pin.
        state.publish_catalog_generation_with_point_slot_fence(
            catalog_for_tables(1, std::slice::from_ref(&table)),
            0,
        );
        let directory_after = residency.table_point_slots.load_full();
        let retained = residency
            .table_point_slot("unchanged_slot_catalog", 81)
            .expect("unchanged publication retains the slot");
        assert!(Arc::ptr_eq(&directory_before, &directory_after));
        assert!(Arc::ptr_eq(&slot, &retained));
        assert!(Arc::ptr_eq(&epoch, &retained.index_epoch));
        assert!(Arc::ptr_eq(&slot.slot_identity, &retained.slot_identity));
    }

    #[test]
    fn global_route_slot_limit_evicts_the_first_other_populated_table() {
        let populated = (0..crate::engine_retained_read::MAX_CACHED_SHARDED_POINT_ROUTES)
            .map(|index| (format!("table-{index:02}"), true))
            .collect::<BTreeMap<_, _>>();
        assert!(table_point_route_requires_eviction(populated.len(), false));
        assert!(!table_point_route_requires_eviction(populated.len(), true));
        assert!(!table_point_route_requires_eviction(
            populated.len() - 1,
            false
        ));

        let (victim_name, victim_populated) =
            first_other_populated_point_route_slot("table-64", populated.iter(), |populated| {
                *populated
            })
            .expect("a full global route directory has a deterministic other victim");
        assert_eq!(victim_name, "table-00");
        assert!(*victim_populated);
    }

    #[test]
    fn route_purge_inner_is_allocation_free_store_only() {
        let source = include_str!("point_slots.rs");
        let body = source
            .split("pub(crate) fn purge_table_point_routes_under_publish_lock")
            .nth(1)
            .and_then(|tail| tail.split("pub(crate) fn sharded_point_route_count").next())
            .expect("point-route purge implementation");
        for forbidden in ["BTreeMap", ".clone()", ".insert(", ".remove("] {
            assert!(
                !body.contains(forbidden),
                "allocation-free route purge must not contain {forbidden}"
            );
        }
        assert!(body.contains("store(None)"));
    }
}
