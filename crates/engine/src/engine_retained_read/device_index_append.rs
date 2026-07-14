use super::{Engine, SqlValue};

impl Engine {
    /// M1 (ledger #24): INCREMENTALLY maintain a cached DEVICE PK index over an in-place append —
    /// insert only the k appended keys via the `index_insert` kernel (O(k)) instead of the O(rows)
    /// rebuild (`ensure_shard_pk_device_index`) the (ptr,row_count) validation would otherwise force
    /// every wave (the measured 305us/wave bottleneck). Device analog of the host
    /// `extend_shard_pk_index_cache_on_append`. Called at the append chokepoint with the appended
    /// values in hand (no DtoH). Per entry: a different ptr (re-admit) or a basis != `base_row_count`
    /// (a prober rebuilt) is skipped; a DECLINED entry stays declined (monotone); a dup/overflow ->
    /// DECLINED; past the load rule (`2*new_count > table_size`) the entry is DROPPED (the next probe
    /// rebuilds at the grown size).
    pub(crate) fn extend_shard_pk_device_index_on_append(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        base_row_count: usize,
        column_values: &[Vec<i32>],
        // COMPOUND KEYS (wider types): the appended rows' full SqlValues, so a compound index's tail
        // fingerprint can fold WIDER key columns (i64) whose values are not in the i32 `column_values`.
        new_rows: &[Vec<SqlValue>],
    ) {
        let appended = column_values.first().map_or(0, Vec::len);
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
        // COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): each compound unique index's cache entry is keyed
        // by `FLAG | ordinal`, and its appended tail is the per-row FINGERPRINT folded from the key
        // columns' appended values. Only compound indexes need this second pass (single-column keys
        // rode the loop above); resolved from the catalog (rare relative to the append itself).
        let catalog = self.catalog_snapshot();
        if let Some(table) = catalog.relational_catalog.get(table_name) {
            for (ord, index) in table.indexes.iter().enumerate() {
                if !index.unique || !crate::engine_residency::index_is_compound(index) {
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
                let key_id = crate::engine_residency::COMPOUND_KEY_ID_FLAG | ord;
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
    /// and compound passes of `extend_shard_pk_device_index_on_append` (`tail` is a column's raw
    /// values or the folded compound fingerprints; the device index treats both as opaque keys).
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
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&key);
                return;
            }
            let Ok(base_row_u32) = u32::try_from(base_row_count) else {
                return;
            };
            // The kernel mutates the device index buffer IN PLACE (atom.cas). A launch failure ->
            // drop the entry (rebuild next probe); never a wrong index.
            match index.submit_i32_index_insert(table_mask, hash_shift, tail, base_row_u32) {
                Ok(dup) => {
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
                    if dup {
                        // F3/U4: the insert kernel now PLACES version twins, so `dup` no longer
                        // means "duplicate key" — it fires ONLY on a 256-probe OVERFLOW (a shard
                        // whose live+twin fan-out overran the probe cap). Drop the index so the
                        // next probe rebuilds at the grown, boundary-gated size (dead-below-GC
                        // twins are dropped there). A pathological hot-key with >256 un-GC'd
                        // versions stays declined until its readers release — a bounded transient.
                        entry.device_index = None;
                    }
                    entry.row_count = new_count;
                }
                Err(_) => {
                    self.read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&key);
                }
            }
        }
    }
}
