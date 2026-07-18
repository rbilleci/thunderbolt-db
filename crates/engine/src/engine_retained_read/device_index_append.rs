use super::{Engine, SqlValue};

impl Engine {
    #[cfg(test)]
    pub(crate) fn set_shard_pk_index_append_post_launch_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *self
            .read_state
            .residency
            .shard_pk_index_append_post_launch_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((reached, resume));
    }

    /// M1 (ledger #24): INCREMENTALLY maintain a cached DEVICE PK index over an in-place append —
    /// insert only the k appended keys via the `index_insert` kernel (O(k)) instead of the O(rows)
    /// rebuild (`ensure_shard_pk_device_index`) the (ptr,row_count) validation would otherwise force
    /// every wave (the measured 305us/wave bottleneck). Called at the append chokepoint with the appended
    /// values in hand (no DtoH). Per entry: a different ptr (re-admit) or a basis != `base_row_count`
    /// (a prober rebuilt) is skipped; a DECLINED build entry stays declined (monotone); probe overflow or
    /// crossing the load rule (`2*new_count > table_size`) retires the table's cached routes/indexes so the
    /// next probe rebuilds at the grown, boundary-gated size without leaving a hidden route pin.
    pub(crate) fn extend_shard_pk_device_index_on_append(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        base_row_count: usize,
        column_values: &[Vec<i32>],
        // The appended rows' full SqlValues, so a fingerprint tail can fold wider/text key columns
        // whose values are absent from the i32 `column_values`.
        new_rows: &[Vec<SqlValue>],
    ) {
        // A table may contain no i32-section column at all (for example `k TEXT PRIMARY KEY`), so
        // `column_values` is not an authoritative row-count source for fingerprint maintenance.
        let appended = new_rows.len();
        if appended == 0 {
            return;
        }
        let new_count = base_row_count + appended;
        // Single-column keys: the cache is keyed by the catalog COLUMN INDEX, and the appended tail
        // is that column's values verbatim (`column_values[col_idx]`).
        for (col_idx, tail) in column_values.iter().enumerate() {
            self.extend_shard_pk_device_index_entry(
                (table_name.to_string(), shard_id, col_idx),
                device_ptr,
                base_row_count,
                new_count,
                tail,
            );
        }
        // Fingerprint-backed unique indexes (compound plus single wider/text) are keyed by
        // `FLAG | ordinal`; their appended tails fold the typed key values into per-row fingerprints.
        // Raw single-i32 indexes rode the loop above; catalog resolution is rare relative to append.
        let catalog = self.catalog_snapshot();
        if let Some(table) = catalog.relational_catalog.get(table_name) {
            for (ord, index) in table.indexes.iter().enumerate() {
                if !index.unique || !crate::engine_residency::index_uses_fingerprint(table, index) {
                    continue;
                }
                // Fold each appended row's key TUPLE into its fingerprint from the full SqlValues
                // (handles any supported key type, incl. i64). A row whose key can't fold (e.g. a NULL
                // key column) makes the whole tail unfoldable -> skip this index's incremental extend;
                // its cache entry stays at the old row_count and the next probe rebuilds it.
                let mut fp_tail: Vec<i32> = Vec::with_capacity(new_rows.len());
                let mut foldable = true;
                for row in new_rows {
                    match crate::engine_residency::compound_index_row_fingerprint(table, index, row)
                    {
                        Some(fp) => fp_tail.push(fp),
                        None => {
                            foldable = false;
                            break;
                        }
                    }
                }
                if !foldable {
                    continue;
                }
                let Some(key_id) = crate::engine_residency::index_probe_key_id(table, index, ord)
                else {
                    continue;
                };
                self.extend_shard_pk_device_index_entry(
                    (table_name.to_string(), shard_id, key_id),
                    device_ptr,
                    base_row_count,
                    new_count,
                    &fp_tail,
                );
            }
        }
    }

    /// M1 (ledger #24): maintain ONE cached device PK-index entry over an append — insert the k
    /// appended keys/fingerprints (`tail`) via the `index_insert` kernel. Shared by the single-column
    /// and fingerprint passes of `extend_shard_pk_device_index_on_append` (`tail` is a column's raw
    /// values or folded typed fingerprints; the device index treats both as opaque keys).
    fn extend_shard_pk_device_index_entry(
        &self,
        key: (String, u32, usize),
        device_ptr: u64,
        base_row_count: usize,
        new_count: usize,
        tail: &[i32],
    ) {
        {
            // Snapshot the entry basis under the lock (index Arc is cheap-cloned for the launch).
            let (index, table_mask, hash_shift) = {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let Some(entry) = cache.get(&key) else {
                    return; // not built on the device -> nothing to maintain
                };
                if entry.resident_device_ptr != device_ptr || entry.row_count != base_row_count {
                    return; // re-admit / a prober advanced it -> the prober path converges it
                }
                let Some(index) = entry.device_index.clone() else {
                    return; // DECLINED is monotone under appends
                };
                (index, entry.table_mask, entry.hash_shift)
            };
            let table_size = (table_mask as u64) + 1;
            if (new_count as u64).saturating_mul(2) > table_size {
                // Past the builder's load rule -> drop so the next probe rebuilds at the grown size.
                self.read_state
                    .residency
                    .purge_shard_pk_index_for_table(&key.0);
                return;
            }
            let Ok(base_row_u32) = u32::try_from(base_row_count) else {
                return;
            };
            // The kernel mutates the device index buffer IN PLACE (atom.cas). A launch failure ->
            // drop the entry (rebuild next probe); never a wrong index.
            match index.submit_i32_index_insert(table_mask, hash_shift, tail, base_row_u32) {
                Ok(dup) => {
                    if dup {
                        // Probe overflow makes this allocation unusable. Retire any prepared route before
                        // removing its accounted index-map owner; the next probe may rebuild at a wider size.
                        self.read_state
                            .residency
                            .purge_shard_pk_index_for_table(&key.0);
                        return;
                    }
                    #[cfg(test)]
                    self.read_state
                        .residency
                        .run_shard_pk_index_append_post_launch_hook();
                    let mut cache = self
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let Some(entry) = cache.get_mut(&key) else {
                        return;
                    };
                    // Re-validate the basis (a racing rebuild could have replaced it).
                    if entry.resident_device_ptr != device_ptr || entry.row_count != base_row_count
                    {
                        return;
                    }
                    let Some(current_index) = entry.device_index.as_ref() else {
                        return;
                    };
                    if !std::sync::Arc::ptr_eq(current_index, &index) {
                        // A boundary-aware rebuild replaced the map entry while the insertion was in
                        // flight. Only `index` received the appended keys; advancing the replacement's
                        // basis would make it look complete and create a false-negative point probe.
                        return;
                    }
                    entry.row_count = new_count;
                }
                Err(_) => {
                    self.read_state
                        .residency
                        .purge_shard_pk_index_for_table(&key.0);
                }
            }
        }
    }
}
