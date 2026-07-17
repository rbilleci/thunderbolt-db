//! Exact device tuple-constraint probes: fingerprint-index candidates with a typed resident-scan
//! fallback, plus full materialized-value and entity-exclusion rechecks.

use super::contracts::device_structural_tuple_predicate;
use super::*;

impl Engine {
    /// COMPOUND KEYS: probe the fingerprint index, materialize each hit on-device, and confirm the
    /// full key tuple. Index decline stays device-native through an exact typed predicate scan.
    pub(crate) fn device_visible_row_with_tuple(
        &self,
        table: &RelationalTable,
        mut visibility: StorageVisibility,
        key_id: usize,
        fingerprint: Option<i32>,
        key_cols: &[(usize, SqlValue)],
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Option<bool> {
        if !self.dml_device_validate_enabled() {
            return None;
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
        let index_hits = fingerprint.and_then(|fingerprint| {
            self.locate_resident_pk_via_shard_index_detailed(table, key_id, fingerprint)
        });
        let hits = match index_hits {
            Some(hits) => hits,
            None => {
                // An index build/probe decline is not permission to consult the host value index.
                // Scan the exact typed tuple predicate on the same resident generation. This also
                // covers compound partial-NULL keys, which have no complete fingerprint.
                if self.current_transaction_read_snapshot().is_none() {
                    visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
                }
                let predicate = device_structural_tuple_predicate(table, key_cols)?;
                self.locate_resident_delete_slots_detailed(table, &predicate)?
            }
        };
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
            let row =
                match self.materialize_resident_row_via_hit(table, hit, visibility.read_txn_id) {
                    Some(Some(row)) => row,
                    Some(None) => continue,
                    None => return None,
                };
            if key_cols.iter().all(|(ci, v)| row.get(*ci) == Some(v)) {
                answer = true;
                break;
            }
        }
        self.read_state
            .residency
            .dml_device_validate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(answer)
    }
}
