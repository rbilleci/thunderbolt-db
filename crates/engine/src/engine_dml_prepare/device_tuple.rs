//! Exact typed device tuple-constraint probes with visibility and entity-exclusion checks.

use super::contracts::device_structural_tuple_predicates;
use super::*;

impl Engine {
    /// Probe compound keys through an exact typed resident predicate. Fingerprints remain addressing
    /// hints elsewhere; they are never constraint authority.
    pub(crate) fn device_visible_row_with_tuple(
        &self,
        table: &RelationalTable,
        mut visibility: StorageVisibility,
        _key_id: usize,
        _fingerprint: Option<i32>,
        key_cols: &[(usize, SqlValue)],
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Option<bool> {
        if let Some(snapshot) = self.current_transaction_read_snapshot() {
            if !snapshot
                .catalog
                .relational_catalog
                .contains_key(&table.name)
                && snapshot
                    .transaction_shards()
                    .get(&table.name)
                    .is_some_and(Vec::is_empty)
            {
                // The transaction catalog owns an exact empty device relation before the first
                // INSERT into a private CREATE. Compound keys have the same authoritative empty
                // verdict as their single-column twin; absence from the published caches is not
                // used as evidence.
                return Some(false);
            }
        }
        if self.table_chunk_authoritative(&table.name).is_some() {
            if self.current_transaction_read_snapshot().is_none() {
                visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
            }
            return self.chunk_class_visible_row_with_tuple(
                table,
                visibility.read_txn_id,
                key_cols,
                exclude_keys,
            );
        }
        if crate::engine_prepared_transaction::prepared_index_route_required() {
            let group = key_cols
                .iter()
                .map(|(column, value)| (*column, SelectFilterOp::Eq, value.clone()))
                .collect::<Vec<_>>();
            let hits = self
                .prepared_exact_index_hits(table, &group, visibility)
                .ok()?;
            return self.prepared_index_hit_exists(table, &hits, exclude_keys);
        }
        // A fingerprint is only an addressing accelerator and can collide. Constraint authority
        // therefore comes from the exact typed device predicate for every tuple shape.
        if self.current_transaction_read_snapshot().is_none() {
            visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
        }
        let predicates = device_structural_tuple_predicates(table, key_cols)?;
        let hits = self.locate_resident_conjunct_slots_detailed(table, &predicates)?;
        let mut answer = false;
        for hit in &hits {
            let region = hit.row_id.as_ref()?;
            let halves = region
                .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                .ok()?;
            let (lo, hi) = (*halves.first()?, *halves.get(1)?);
            let row_id = (lo as u32 as u64) | ((hi as u32 as u64) << 32);
            if row_id == u64::MAX {
                return None;
            }
            let key = relational_row_key(&table.name, row_id);
            if exclude_keys.is_some_and(|excluded| excluded.contains(&key)) {
                continue;
            }
            match self.materialize_resident_row_via_hit(table, hit, visibility.read_txn_id) {
                Some(Some(_)) => {
                    answer = true;
                    break;
                }
                Some(None) => continue,
                None => return None,
            }
        }
        self.read_state
            .residency
            .dml_device_validate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(answer)
    }
}
