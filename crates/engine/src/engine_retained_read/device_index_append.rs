use super::{
    shard_fixed_width_key_offset, shard_key_column_blob_len, shard_key_column_blob_offset,
    shard_key_column_validity_offset, Engine, SqlValue,
};

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
    ) -> bool {
        // A table may contain no i32-section column at all (for example `k TEXT PRIMARY KEY`), so
        // `column_values` is not an authoritative row-count source for fingerprint maintenance.
        let appended = new_rows.len();
        if appended == 0 {
            return true;
        }
        let new_count = base_row_count + appended;
        let required_named = self.published_named_index_keys(
            table_name,
            shard_id,
            device_ptr,
            base_row_count,
            false,
        );
        if required_named.is_some() {
            let catalog = self.catalog_snapshot();
            if catalog
                .relational_catalog
                .get(table_name)
                .is_some_and(|table| {
                    table.indexes.iter().any(|index| {
                        crate::engine_residency::index_key_column_positions(table, index)
                            .is_none_or(|positions| {
                                new_rows.iter().any(|row| {
                                    positions.iter().any(|&position| {
                                        row.get(position)
                                            .is_none_or(|value| matches!(value, SqlValue::Null))
                                    })
                                })
                            })
                    })
                })
            {
                return false;
            }
        }
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
        let fingerprints_ok = self.extend_shard_fingerprint_device_indexes_on_append(
            table_name,
            shard_id,
            device_ptr,
            base_row_count,
            new_rows,
        );
        fingerprints_ok
            && required_named.is_none_or(|keys| {
                self.named_index_keys_cover(table_name, shard_id, device_ptr, new_count, &keys)
            })
    }

    /// Maintain only fingerprint-backed named indexes. The fused int4 append kernel already extends
    /// its raw PK index; this companion pass prevents it from silently skipping compound secondary
    /// indexes without inserting the raw key twice.
    pub(crate) fn extend_shard_fingerprint_device_indexes_on_append(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        base_row_count: usize,
        new_rows: &[Vec<SqlValue>],
    ) -> bool {
        if new_rows.is_empty() {
            return true;
        }
        let appended = new_rows.len();
        let new_count = base_row_count + appended;
        let required_named =
            self.published_named_index_keys(table_name, shard_id, device_ptr, base_row_count, true);
        let catalog = self.catalog_snapshot();
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return required_named.is_none();
        };
        let shards = self.read_residency_shards();
        let Some(shard) = shards
            .get(table_name)
            .and_then(|table_shards| table_shards.iter().find(|shard| shard.shard_id == shard_id))
        else {
            return required_named.is_none();
        };
        let Some(source) = shard
            .device_memory
            .as_ref()
            .filter(|memory| memory.device_ptr() == device_ptr)
            .cloned()
        else {
            return required_named.is_none();
        };

        // Build every fingerprint descriptor from the live resident layout, then mutate all matching
        // index allocations in one launch. Keys and fingerprints never materialize on the CPU.
        let mut requests = Vec::new();
        let mut bases = Vec::new();
        for (ordinal, index) in table.indexes.iter().enumerate() {
            if !crate::engine_residency::index_uses_fingerprint(table, index) {
                continue;
            }
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            if new_rows.iter().any(|row| {
                positions.iter().any(|&position| {
                    row.get(position)
                        .is_none_or(|value| matches!(value, SqlValue::Null))
                })
            }) {
                self.read_state
                    .residency
                    .purge_shard_pk_index_for_table(table_name);
                return required_named.is_none();
            }
            let Some(key_id) = crate::engine_residency::index_probe_key_id(table, index, ordinal)
            else {
                continue;
            };
            let Some(offsets) = positions
                .iter()
                .map(|&position| shard_fixed_width_key_offset(shard, table, position))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let Some(blob_offsets) = positions
                .iter()
                .map(|&position| shard_key_column_blob_offset(shard, table, position))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let Some(blob_lens) = positions
                .iter()
                .map(|&position| shard_key_column_blob_len(shard, table, position))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let Some(widths) = positions
                .iter()
                .map(|&position| {
                    crate::engine_residency::key_column_width_words(table.columns[position].ty)
                })
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let Some(validity_offsets) = positions
                .iter()
                .map(|&position| shard_key_column_validity_offset(shard, table, position))
                .collect::<Option<Vec<Option<u64>>>>()
            else {
                continue;
            };
            let mut columns = widths
                .iter()
                .enumerate()
                .map(|(idx, &width_words)| {
                    if width_words == u32::MAX {
                        gpu_db_execution::CudaCompoundFoldColumn::Bool {
                            bitmap_byte_offset: offsets[idx],
                        }
                    } else if width_words == 0 {
                        gpu_db_execution::CudaCompoundFoldColumn::Text {
                            offsets_byte_offset: offsets[idx],
                            bytes_byte_offset: blob_offsets[idx],
                            bytes_len: blob_lens[idx],
                        }
                    } else {
                        gpu_db_execution::CudaCompoundFoldColumn::Fixed {
                            byte_offset: offsets[idx],
                            width_words,
                        }
                    }
                })
                .collect::<Vec<_>>();
            let mut seen_validity = std::collections::BTreeSet::new();
            columns.extend(
                validity_offsets
                    .iter()
                    .flatten()
                    .filter_map(|&bitmap_byte_offset| {
                        seen_validity.insert(bitmap_byte_offset).then_some(
                            gpu_db_execution::CudaCompoundFoldColumn::Validity {
                                bitmap_byte_offset,
                            },
                        )
                    }),
            );
            let key = (table_name.to_string(), shard_id, key_id);
            let basis = {
                let cache = self
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let Some(entry) = cache.get(&key) else {
                    continue;
                };
                if entry.resident_device_ptr != device_ptr || entry.row_count != base_row_count {
                    continue;
                }
                let Some(device_index) = entry.device_index.clone() else {
                    continue;
                };
                (
                    device_index,
                    entry.table_mask,
                    entry.hash_shift,
                    key,
                    std::sync::Arc::clone(&entry.published_row_count),
                    std::sync::Arc::clone(&entry.published_has_postings),
                )
            };
            requests.push(gpu_db_execution::CudaResidentTypedIndexInsert {
                index: std::sync::Arc::clone(&basis.0),
                table_mask: basis.1,
                hash_shift: basis.2,
                columns,
            });
            bases.push(basis);
        }
        if !requests.is_empty() {
            let _index_mutation = self
                .read_state
                .residency
                .begin_point_index_mutation(table_name);
            match source.submit_resident_typed_indexes_insert_status(
                &requests,
                base_row_count,
                appended,
            ) {
                Ok(status) if !status.declined => {
                    #[cfg(test)]
                    self.read_state
                        .residency
                        .run_shard_pk_index_append_post_launch_hook();
                    // The allocation was mutated even if a racing lifecycle purge has removed its
                    // cache entry. Publish through the basis-owned Arcs first so every retained pin
                    // observes the new physical extent and posting mode independently of the map.
                    for (_, _, _, _, published_rows, published_postings) in &bases {
                        if status.created_posting {
                            published_postings.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        published_rows.store(new_count, std::sync::atomic::Ordering::Release);
                    }
                    let mut cache = self
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    for (device_index, _, _, key, _, _) in bases {
                        let Some(entry) = cache.get_mut(&key) else {
                            continue;
                        };
                        if entry.resident_device_ptr == device_ptr
                            && entry.row_count == base_row_count
                            && entry.device_index.as_ref().is_some_and(|current| {
                                std::sync::Arc::ptr_eq(current, &device_index)
                            })
                        {
                            entry.has_postings |= status.created_posting;
                            entry.row_count = new_count;
                        }
                    }
                }
                Ok(_) | Err(_) => {
                    self.read_state
                        .residency
                        .purge_shard_pk_index_for_table(table_name);
                    return required_named.is_none();
                }
            }
        }
        required_named.is_none_or(|keys| {
            self.named_index_keys_cover(table_name, shard_id, device_ptr, new_count, &keys)
        })
    }

    /// Return the named key ids only when every requested index is already a live publication over
    /// at least `row_count`. `None` means this shard was never enrolled, so legacy lazy indexes keep
    /// their best-effort behavior. Once enrolled, the post-append coverage check below is mandatory.
    fn published_named_index_keys(
        &self,
        table_name: &str,
        _shard_id: u32,
        _device_ptr: u64,
        _row_count: usize,
        fingerprint_only: bool,
    ) -> Option<Vec<usize>> {
        let catalog = self.catalog_snapshot();
        let table = catalog.relational_catalog.get(table_name)?;
        let keys = table
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, index)| {
                !fingerprint_only || crate::engine_residency::index_uses_fingerprint(table, index)
            })
            .map(|(ordinal, index)| {
                crate::engine_residency::index_probe_key_id(table, index, ordinal)
            })
            .collect::<Option<std::collections::BTreeSet<_>>>()?
            .into_iter()
            .collect::<Vec<_>>();
        self.read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&table.oid)
            .then_some(keys)
    }

    fn named_index_keys_cover(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        row_count: usize,
        keys: &[usize],
    ) -> bool {
        let _route_publish = self
            .read_state
            .residency
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cache = self
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let covered = keys.iter().all(|key_id| {
            cache
                .get(&(table_name.to_string(), shard_id, *key_id))
                .is_some_and(|entry| {
                    entry.resident_device_ptr == device_ptr
                        && entry.row_count >= row_count
                        && entry.device_index.is_some()
                })
        });
        if covered {
            let mut coverage = self
                .read_state
                .residency
                .named_index_coverage
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for key_id in keys {
                coverage.insert(
                    (table_name.to_string(), shard_id, *key_id),
                    (device_ptr, row_count),
                );
            }
        }
        covered
    }

    /// Fused append owns the raw index mutation while the companion path above owns fingerprints.
    /// Verify their union so a fused raw-index decline cannot be hidden by successful secondaries.
    pub(crate) fn named_indexes_cover_after_fused_append(
        &self,
        table_name: &str,
        shard_id: u32,
        device_ptr: u64,
        base_row_count: usize,
        new_count: usize,
    ) -> bool {
        self.published_named_index_keys(table_name, shard_id, device_ptr, base_row_count, false)
            .is_none_or(|keys| {
                self.named_index_keys_cover(table_name, shard_id, device_ptr, new_count, &keys)
            })
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
            let (index, table_mask, hash_shift, published_rows, published_postings) = {
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
                (
                    index,
                    entry.table_mask,
                    entry.hash_shift,
                    std::sync::Arc::clone(&entry.published_row_count),
                    std::sync::Arc::clone(&entry.published_has_postings),
                )
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
            let _index_mutation = self.read_state.residency.begin_point_index_mutation(&key.0);
            match index.submit_i32_index_insert_status(table_mask, hash_shift, tail, base_row_u32) {
                Ok(status) => {
                    if status.declined {
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
                    if status.created_posting {
                        published_postings.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    published_rows.store(new_count, std::sync::atomic::Ordering::Release);
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
                    entry.has_postings |= status.created_posting;
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
