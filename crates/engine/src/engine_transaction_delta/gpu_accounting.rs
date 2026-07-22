//! Pre-allocation GPU budget reservations and exact private-generation accounting.

use super::*;

#[derive(Default)]
struct TransactionPublicationImages {
    old_rows: Vec<Vec<SqlValue>>,
    old_row_ids: Vec<u64>,
    new_rows: Vec<Vec<SqlValue>>,
}

/// Exact pre-allocation charge for device objects created while building one transaction-private
/// generation. Reservations enter the engine-wide budget account before CUDA allocation. A failed
/// statement releases them on drop; a successful publish transfers them to the transaction's
/// current generation account. The transaction statement lock excludes same-transaction readers
/// while replacement charges are reconciled, and transaction teardown releases the final account.
pub(crate) struct TransactionGpuReservation<'a> {
    engine: &'a Engine,
    bytes_by_gpu: BTreeMap<u16, u64>,
    transferred: bool,
}

impl<'a> TransactionGpuReservation<'a> {
    pub(crate) fn new(engine: &'a Engine) -> Self {
        Self {
            engine,
            bytes_by_gpu: BTreeMap::new(),
            transferred: false,
        }
    }

    pub(crate) fn reserve(&mut self, gpu_id: u16, bytes: u64) -> Result<(), ExecuteError> {
        if bytes == 0 {
            return Ok(());
        }
        // Publish the reservation while holding the same allocation lock as global admission and
        // lazy indexes. Once charged, later global preflights include these bytes, so the lock can
        // be released before CUDA work without a check-then-allocate gap or a kernel-long hold.
        let _budget_guard = self
            .engine
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Includes every already-charged private generation. Publishing the charge under the
        // allocation lock prevents a global admission/index publication from changing the other
        // half of the total between this observation and reservation publication.
        let accounted = self.engine.relational_resident_bytes_for_gpu(gpu_id);
        if let Some(budget) = self.engine.relational_residency_budget_bytes(gpu_id) {
            if accounted.saturating_add(bytes) > budget {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction-private GPU allocation of {bytes} bytes on GPU {gpu_id} exceeds residency budget {budget}"
                ))));
            }
        }
        let mut account = self
            .engine
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let account_slot = account.entry(gpu_id).or_default();
        *account_slot = account_slot.saturating_add(bytes);
        let local_slot = self.bytes_by_gpu.entry(gpu_id).or_default();
        *local_slot = local_slot.saturating_add(bytes);
        Ok(())
    }

    /// Verify the CUDA allocation contract used by transaction COW: the amount admitted before
    /// allocation is the exact `allocated_bytes` later retained by the private generation. The
    /// current driver allocates exactly the requested region; failing closed here prevents a
    /// future padded/pooled allocator from silently retaining more device memory than was admitted.
    pub(crate) fn verify_allocation(
        &self,
        gpu_id: u16,
        reserved_bytes: u64,
        allocated_bytes: u64,
    ) -> Result<(), ExecuteError> {
        debug_assert_eq!(
            allocated_bytes, reserved_bytes,
            "transaction GPU allocation metadata must equal its pre-allocation reservation"
        );
        if allocated_bytes != reserved_bytes {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "transaction-private GPU allocation on GPU {gpu_id} retained {allocated_bytes} bytes after reserving {reserved_bytes}; the allocator requires an exact preflight estimator"
            ))));
        }
        Ok(())
    }

    /// Prove that the replacement graph cannot raise the retained private account above the
    /// prior generation plus this statement's admitted allocations. Call before publishing the
    /// replacement Arcs so an accounting-contract violation leaves the transaction unchanged.
    pub(crate) fn ensure_replacement_admitted(
        &self,
        charged: &BTreeMap<u16, u64>,
        replacement: &BTreeMap<u16, u64>,
    ) -> Result<(), ExecuteError> {
        for (gpu_id, replacement_bytes) in replacement {
            let admitted = charged
                .get(gpu_id)
                .copied()
                .unwrap_or(0)
                .saturating_add(self.bytes_by_gpu.get(gpu_id).copied().unwrap_or(0));
            if *replacement_bytes > admitted {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction-private replacement on GPU {gpu_id} retains {replacement_bytes} bytes but only {admitted} bytes were admitted"
                ))));
            }
        }
        Ok(())
    }

    /// Replace the prior generation's charge with the allocations actually reachable from the
    /// newly published private shard map. This releases superseded COW sidecars while retaining
    /// carried-forward private payloads exactly once. The caller holds the transaction statement
    /// lock, swaps and releases the only permitted old-map statement pin, then reconciles under the
    /// allocation lock so global admission cannot observe a transiently reduced account.
    pub(crate) fn replace_charges(
        &mut self,
        charged: &mut BTreeMap<u16, u64>,
        replacement: BTreeMap<u16, u64>,
    ) {
        debug_assert!(replacement.iter().all(|(gpu_id, replacement_bytes)| {
            *replacement_bytes
                <= charged
                    .get(gpu_id)
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(self.bytes_by_gpu.get(gpu_id).copied().unwrap_or(0))
        }));
        let _budget_guard = self
            .engine
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut account = self
            .engine
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (gpu_id, bytes) in charged.iter().chain(self.bytes_by_gpu.iter()) {
            let slot = account.entry(*gpu_id).or_default();
            *slot = slot.saturating_sub(*bytes);
        }
        for (gpu_id, bytes) in &replacement {
            let slot = account.entry(*gpu_id).or_default();
            *slot = slot.saturating_add(*bytes);
        }
        account.retain(|_, bytes| *bytes != 0);
        *charged = replacement;
        self.transferred = true;
    }
}

impl Drop for TransactionGpuReservation<'_> {
    fn drop(&mut self) {
        if self.transferred {
            return;
        }
        let mut account = self
            .engine
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (gpu_id, bytes) in &self.bytes_by_gpu {
            let slot = account.entry(*gpu_id).or_default();
            *slot = slot.saturating_sub(*bytes);
        }
        account.retain(|_, bytes| *bytes != 0);
    }
}

impl Engine {
    fn reconcile_transaction_commit_gpu_reservation(
        &self,
        current: &mut BTreeMap<u16, u64>,
        replacement: &BTreeMap<u16, u64>,
    ) -> Result<(), ExecuteError> {
        let _budget_guard = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for gpu_id in current
            .keys()
            .chain(replacement.keys())
            .copied()
            .collect::<BTreeSet<_>>()
        {
            let old = current.get(&gpu_id).copied().unwrap_or(0);
            let new = replacement.get(&gpu_id).copied().unwrap_or(0);
            let accounted_without_old = self
                .relational_resident_bytes_for_gpu(gpu_id)
                .saturating_sub(old);
            if self
                .relational_residency_budget_bytes(gpu_id)
                .is_some_and(|budget| accounted_without_old.saturating_add(new) > budget)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction commit requires {new} additional bytes for canonical GPU publication on GPU {gpu_id}, exceeding the residency budget"
                ))));
            }
        }
        let mut account = self
            .transaction_private_gpu_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (gpu_id, bytes) in current.iter() {
            let slot = account.entry(*gpu_id).or_default();
            *slot = slot.saturating_sub(*bytes);
        }
        for (gpu_id, bytes) in replacement {
            let slot = account.entry(*gpu_id).or_default();
            *slot = slot.saturating_add(*bytes);
        }
        account.retain(|_, bytes| *bytes != 0);
        *current = replacement.clone();
        Ok(())
    }

    /// Restore any previously enrolled existing-table indexes before WAL. The lifecycle guard
    /// acquired by the caller defers destructive purges until canonical apply completes, so this
    /// proof cannot disappear in the durability interval. Empty generations need no restoration:
    /// rollover publication will build the first non-empty shard from its exact reserved geometry.
    fn restore_transaction_named_indexes_before_wal(
        &self,
        transaction_catalog: &CatalogSnapshot,
        record: &BinaryTransactionRecord,
    ) -> Result<(), ExecuteError> {
        let created_tables = record
            .catalog_commands
            .iter()
            .filter_map(|command| match command {
                Command::CreateTable(create) => Some(create.table.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let reset_tables = record
            .table_resets
            .iter()
            .map(|reset| reset.table.as_str())
            .collect::<BTreeSet<_>>();
        let appended_tables = record
            .mutations
            .iter()
            .filter_map(|mutation| match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. } => Some(table.as_str()),
                BinaryTransactionMutation::Delete { .. } => None,
            })
            .filter(|table| !created_tables.contains(table) && !reset_tables.contains(table))
            .collect::<BTreeSet<_>>();
        let current_shards = self.read_residency_shards();
        for table_name in appended_tables {
            let table = transaction_catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "transaction index restoration targets unknown relation \"{table_name}\""
                    )))
                })?;
            if self.table_chunk_authoritative(table_name).is_some()
                || !self.relational_named_index_publication_required(table)
            {
                continue;
            }
            let shards = current_shards.get(table_name).ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{table_name}\" lost its resident shards before index restoration"
                ))
            })?;
            if shards.iter().all(|shard| shard.row_count == 0) {
                continue;
            }
            self.publish_relational_resident_indexes_for_generation(
                table,
                shards,
                self.committed_seq(),
                false,
                false,
                true,
            )?;
        }
        Ok(())
    }

    /// Compute the exact allocations retained by canonical publication of the final coalesced WAL
    /// record. Statement-private intermediate versions are deliberately absent: their allocation
    /// charges become temporary publication credit after durability, but they never define the
    /// final index or rollover geometry.
    fn transaction_canonical_publication_gpu_bytes(
        &self,
        snapshot: &TransactionSnapshot,
        transaction_catalog: &CatalogSnapshot,
        record: &BinaryTransactionRecord,
    ) -> Result<BTreeMap<u16, u64>, ExecuteError> {
        let created_tables = record
            .catalog_commands
            .iter()
            .filter_map(|command| match command {
                Command::CreateTable(create) => Some(create.table.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let reset_tables = record
            .table_resets
            .iter()
            .map(|reset| reset.table.clone())
            .collect::<BTreeSet<_>>();
        let mut by_table = BTreeMap::<String, TransactionPublicationImages>::new();
        for table in &reset_tables {
            by_table.entry(table.clone()).or_default();
        }
        for mutation in &record.mutations {
            let (table_name, row_id) = match mutation {
                BinaryTransactionMutation::Insert { table, row_id, .. }
                | BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => (table, *row_id),
            };
            let table = transaction_catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "transaction publication targets unknown relation \"{table_name}\""
                    )))
                })?;
            let images = by_table.entry(table_name.clone()).or_default();
            match mutation {
                BinaryTransactionMutation::Insert { row_encoded, .. } => {
                    images.new_rows.push(
                        decode_relational_row(row_encoded, &table.columns).map_err(|error| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction publication insert image decode failed for \"{table_name}\": {error}"
                            )))
                        })?,
                    );
                }
                BinaryTransactionMutation::Update {
                    old_row_encoded,
                    new_row_encoded,
                    ..
                } => {
                    images.old_rows.push(
                        decode_relational_row(old_row_encoded, &table.columns).map_err(|error| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction publication old update image decode failed for \"{table_name}\": {error}"
                            )))
                        })?,
                    );
                    images.old_row_ids.push(row_id);
                    images.new_rows.push(
                        decode_relational_row(new_row_encoded, &table.columns).map_err(|error| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction publication new update image decode failed for \"{table_name}\": {error}"
                            )))
                        })?,
                    );
                }
                BinaryTransactionMutation::Delete {
                    old_row_encoded, ..
                } => {
                    images.old_rows.push(
                        decode_relational_row(old_row_encoded, &table.columns).map_err(|error| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction publication delete image decode failed for \"{table_name}\": {error}"
                            )))
                        })?,
                    );
                    images.old_row_ids.push(row_id);
                }
            }
        }

        let private_shards = snapshot.transaction_shards();
        let current_shards = self.read_residency_shards();
        let mut bytes_by_gpu = BTreeMap::<u16, u64>::new();
        let mut add = |gpu_id: u16, bytes: u64| -> Result<(), ExecuteError> {
            let slot = bytes_by_gpu.entry(gpu_id).or_default();
            *slot = slot.checked_add(bytes).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction canonical publication reservation overflowed".to_string(),
                ))
            })?;
            Ok(())
        };
        for (table_name, images) in by_table {
            let table = transaction_catalog
                .relational_catalog
                .get(&table_name)
                .expect("publication image table was resolved above");
            if self.table_chunk_authoritative(&table_name).is_some()
                && !reset_tables.contains(&table_name)
            {
                continue;
            }
            if created_tables.contains(&table_name) || reset_tables.contains(&table_name) {
                if !images.old_rows.is_empty() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "fresh transaction relation \"{table_name}\" has a non-insert final mutation"
                    ))));
                }
                let gpu_id = private_shards
                    .get(&table_name)
                    .and_then(|shards| shards.first())
                    .map(|shard| shard.gpu_id)
                    .unwrap_or_else(|| self.planner.default_gpu_id());
                let names = table
                    .columns
                    .iter()
                    .map(|column| column.name.clone())
                    .collect::<Vec<_>>();
                let types = table
                    .columns
                    .iter()
                    .map(|column| column.ty)
                    .collect::<Vec<_>>();
                let (payload, ..) = crate::engine_residency::build_relational_device_payload(
                    &names,
                    &types,
                    &images.new_rows,
                )?;
                let row_count = images.new_rows.len();
                let index_bytes = crate::engine_residency::estimated_named_index_bytes_for_shard(
                    table, row_count, row_count,
                )
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{table_name}\" has unsupported mandatory index allocation geometry"
                    )))
                })?;
                let bytes = (payload.len() as u64)
                    .checked_add((row_count as u64).checked_mul(8).ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "fresh transaction row identity reservation overflowed".to_string(),
                        ))
                    })?)
                    .and_then(|bytes| bytes.checked_add(index_bytes))
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "fresh transaction publication reservation overflowed".to_string(),
                        ))
                    })?;
                add(gpu_id, bytes)?;
                continue;
            }

            if !images.old_rows.is_empty() {
                let locations = self
                    .locate_transaction_rows_by_identity(
                        table,
                        &images.old_rows,
                        &images.old_row_ids,
                        self.committed_seq(),
                    )
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "relation \"{table_name}\" changed before transaction publication preflight"
                        ))
                    })?;
                let table_shards = current_shards.get(&table_name).ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "relation \"{table_name}\" lost its resident shards before transaction publication"
                    ))
                })?;
                for shard_id in locations.keys() {
                    let shard = table_shards
                        .iter()
                        .find(|shard| shard.shard_id == *shard_id)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "relation \"{table_name}\" changed shard identity before transaction publication"
                            ))
                        })?;
                    if shard.deleted_by_region.is_none() {
                        add(
                            shard.gpu_id,
                            (shard.capacity as u64).checked_mul(8).ok_or_else(|| {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction deleted-by reservation overflowed".to_string(),
                                ))
                            })?,
                        )?;
                    }
                }
            }
            if !images.new_rows.is_empty() {
                let (gpu_id, bytes) =
                    self.transaction_resident_append_allocation_bytes(table, &images.new_rows)?;
                add(gpu_id, bytes)?;
            }
        }
        Ok(bytes_by_gpu)
    }

    pub(super) fn reserve_transaction_canonical_publication(
        &self,
        snapshot: &TransactionSnapshot,
        transaction_catalog: &CatalogSnapshot,
        record: &BinaryTransactionRecord,
    ) -> Result<(), ExecuteError> {
        self.restore_transaction_named_indexes_before_wal(transaction_catalog, record)?;
        let required = self.transaction_canonical_publication_gpu_bytes(
            snapshot,
            transaction_catalog,
            record,
        )?;
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let additional = required
            .into_iter()
            .filter_map(|(gpu_id, bytes)| {
                let private = delta
                    .private_gpu_bytes_by_gpu
                    .get(&gpu_id)
                    .copied()
                    .unwrap_or(0);
                let extra = bytes.saturating_sub(private);
                (extra != 0).then_some((gpu_id, extra))
            })
            .collect::<BTreeMap<_, _>>();
        self.reconcile_transaction_commit_gpu_reservation(
            &mut delta.commit_gpu_bytes_by_gpu,
            &additional,
        )
    }
}

/// Exact device allocations retained only by the transaction generation. The captured base
/// shard map defines globally-accounted identities; any distinct payload/sidecar/row-id pointer
/// reachable from the replacement is private. Pointer dedup prevents structural sharing from
/// being charged twice.
pub(crate) fn transaction_private_shard_bytes(
    base: &BTreeMap<String, Vec<RelationalResidentShard>>,
    replacement: &BTreeMap<String, Vec<RelationalResidentShard>>,
) -> BTreeMap<u16, u64> {
    let mut base_allocations = BTreeSet::new();
    for shard in base.values().flatten() {
        for memory in shard
            .device_memory
            .iter()
            .chain(shard.deleted_by_region.iter())
            .chain(shard.created_by_region.iter())
            .chain(shard.row_id_region.iter())
        {
            base_allocations.insert((memory.metadata().gpu_id, memory.device_ptr()));
        }
    }
    let mut seen = BTreeSet::new();
    let mut bytes = BTreeMap::new();
    for shard in replacement.values().flatten() {
        for memory in shard
            .device_memory
            .iter()
            .chain(shard.deleted_by_region.iter())
            .chain(shard.created_by_region.iter())
            .chain(shard.row_id_region.iter())
        {
            let identity = (memory.metadata().gpu_id, memory.device_ptr());
            if base_allocations.contains(&identity) || !seen.insert(identity) {
                continue;
            }
            let slot = bytes.entry(memory.metadata().gpu_id).or_insert(0u64);
            *slot = slot.saturating_add(memory.metadata().allocated_bytes);
        }
    }
    bytes
}

impl Engine {
    pub(super) fn clone_transaction_deleted_region(
        &self,
        shard: &RelationalResidentShard,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<Arc<CudaResidentDeviceMemory>, ExecuteError> {
        let bytes = (shard.capacity as u64).checked_mul(8).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "transaction tombstone region size overflow".to_string(),
            ))
        })?;
        gpu_reservation.reserve(shard.gpu_id, bytes)?;
        let fills = [gpu_db_execution::RecompactFill {
            byte_offset: 0,
            len: bytes,
            fill_byte: crate::engine_residency::DELETED_BY_LIVE_FILL_BYTE,
        }];
        let segments = shard
            .deleted_by_region
            .as_ref()
            .map(|region| {
                if region.metadata().allocated_bytes < bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "captured tombstone region is shorter than its shard capacity".to_string(),
                    )));
                }
                Ok(vec![gpu_db_execution::RecompactSegment {
                    src_device_ptr: region.device_ptr(),
                    src_byte_offset: 0,
                    dst_byte_offset: 0,
                    byte_len: bytes,
                }])
            })
            .transpose()?
            .unwrap_or_default();
        let memory = self
            .cuda_driver_probe_runtime()
            .retain_device_memory_recompacted(shard.gpu_id, bytes, &[], &fills, &segments)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction tombstone allocation failed: {err}"
                )))
            })?;
        gpu_reservation.verify_allocation(
            shard.gpu_id,
            bytes,
            memory.metadata().allocated_bytes,
        )?;
        Ok(Arc::new(memory))
    }

    pub(crate) fn append_transaction_delta_shard(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        if rows.is_empty() {
            return Ok(());
        }
        self.append_transaction_delta_shard_inner(table, rows, row_ids, shards, gpu_reservation)
    }

    /// Install an allocation-backed, typed zero-row root for a private reset/fence. An empty Vec is
    /// only a construction placeholder; it is never accepted as GPU authority or durable proof.
    pub(crate) fn append_transaction_empty_root(
        &self,
        table: &RelationalTable,
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        self.append_transaction_delta_shard_inner(table, &[], &[], shards, gpu_reservation)
    }

    fn append_transaction_delta_shard_inner(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        if rows.len() != row_ids.len() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "transaction delta rows and identities are not parallel".to_string(),
            )));
        }
        let table_shards = shards.get_mut(&table.name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" lost its transaction shard generation",
                table.name
            )))
        })?;
        let gpu_id = table_shards
            .first()
            .map(|shard| shard.gpu_id)
            .unwrap_or_else(|| self.planner.default_gpu_id());
        let names = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let (payload, text, bools, int4_stats, b128, nulls) =
            crate::engine_residency::build_relational_device_payload(&names, &types, rows)?;
        gpu_reservation.reserve(gpu_id, payload.len() as u64)?;
        let memory = self
            .relational_residency_device_memory(gpu_id, &payload)
            .map(Arc::new)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction delta payload allocation failed".to_string(),
                ))
            })?;
        gpu_reservation.verify_allocation(
            gpu_id,
            payload.len() as u64,
            memory.metadata().allocated_bytes,
        )?;
        let mut row_id_payload = Vec::with_capacity(row_ids.len() * 8);
        for row_id in row_ids {
            row_id_payload.extend_from_slice(&row_id.to_le_bytes());
        }
        let row_id_region = if row_id_payload.is_empty() {
            None
        } else {
            gpu_reservation.reserve(gpu_id, row_id_payload.len() as u64)?;
            let region = self
                .relational_residency_device_memory(gpu_id, &row_id_payload)
                .map(Arc::new)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction delta identity allocation failed".to_string(),
                    ))
                })?;
            gpu_reservation.verify_allocation(
                gpu_id,
                row_id_payload.len() as u64,
                region.metadata().allocated_bytes,
            )?;
            Some(region)
        };
        let int4 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
            .map(|column| column.name.clone())
            .collect();
        let int8 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect();
        let numeric = b128.into_iter().map(|(name, _)| name).collect();
        let shard_id = table_shards
            .iter()
            .map(|shard| shard.shard_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let row_start = table_shards
            .iter()
            .map(|shard| shard.row_start.saturating_add(shard.row_count))
            .max()
            .unwrap_or(0);
        // `allocated_bytes` describes the shard payload only; mandatory regions are accounted
        // separately everywhere else. Including row ids here would double-charge them.
        let allocated_bytes = memory.metadata().allocated_bytes;
        table_shards.push(RelationalResidentShard {
            shard_id,
            row_start,
            row_count: rows.len(),
            // Transaction-private shards are additive to the retained generation and discard no
            // globally visible physical history.
            history_floor_index: 0,
            capacity: rows.len(),
            // Private generations never mutate a shard in place, but the exact same descriptor
            // builder also installs a newly-created table during canonical replay. Mark supported
            // layouts append/rollover-capable so the next committed INSERT stays GPU-native.
            int4_appendable: table.columns.iter().all(|column| {
                matches!(
                    column.ty,
                    SqlType::Int2
                        | SqlType::Int4
                        | SqlType::Date
                        | SqlType::Int8
                        | SqlType::Timestamp
                        | SqlType::Numeric { .. }
                        | SqlType::Uuid
                        | SqlType::Bool
                        | SqlType::Text
                )
            }),
            resident_device_int4_column_stats: int4_stats,
            resident_bytes: payload.len() as u64,
            allocated_bytes,
            count_header_byte_offset: 0,
            resident_device_int4_columns: int4,
            resident_device_int8_columns: int8,
            resident_device_numeric_columns: numeric,
            resident_device_bool_columns: bools,
            resident_device_text_columns: text,
            resident_device_null_columns: nulls,
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            point_route_generation: Arc::new(()),
            device_memory_proof: Some(memory.metadata().clone()),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            device_memory: Some(memory),
            deleted_by_region: None,
            created_by_region: None,
            row_id_region,
            max_created_by: 0,
        });
        Ok(())
    }
}
