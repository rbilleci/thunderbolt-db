//! Commit auto-admission, transient relations, and benchmark installation ownership.

use super::*;

impl Engine {
    /// STRATA S-B: commit-triggered admission for non-authoritative bootstrap/repair tables. Runs
    /// after the publication-coordinator join while the commit mutex is held. Device-authoritative DML tables
    /// are already maintained in place and never depend on this best-effort helper; a missing route
    /// for them fails loudly instead of selecting a host execution tier.
    pub(crate) fn auto_admit_resident_tables(&self, tables: &std::collections::BTreeSet<String>) {
        // P4-2b (audit M4): NEVER admit a CHUNK-AUTHORITATIVE table — its store is FROZEN
        // (post-freeze writes live only in the chunks), so an admission (e.g. after a budget
        // raise) would publish a STALE resident snapshot that the resident route serves BEFORE
        // the streaming dispatch, with no de-auth guard in between.
        let class_map = self.read_state.residency.chunk_authoritative_tables.load();
        let tables: std::collections::BTreeSet<String> = tables
            .iter()
            .filter(|t| !class_map.contains_key(*t))
            .cloned()
            .collect();
        let tables = &tables;
        if tables.is_empty() {
            return;
        }
        // VACUUM #5: any rebuild resets the churn signal (the new generation is dense all-live).
        for table in tables {
            self.reset_tombstone_churn(table);
        }
        if tables.is_empty() {
            return;
        }
        let gpu_id = self.planner.default_gpu_id();
        let mut guard = self.ddl_catalog();
        let cat = &mut *guard;
        for table in tables {
            let _ = self.populate_relational_residency_snapshot_inner(cat, table, gpu_id);
        }
    }

    pub(crate) fn relational_residency_device_memory(
        &self,
        gpu_id: u16,
        payload: &[u8],
    ) -> Option<CudaResidentDeviceMemory> {
        let runtime = self.cuda_driver_probe_runtime();
        runtime.retain_device_memory_copy(gpu_id, payload).ok()
    }

    /// Build a TRANSIENT resident-like relation from already-materialized host `rows` -- a `RelationalTable`
    /// descriptor + an uploaded device payload that the GPU join path consumes EXACTLY like a published
    /// resident table (`lower_resident_predicate`, `project_*_rows_from_payload`,
    /// `join_fixed_payload_coordinates`),
    /// but WITHOUT publishing/admitting/evicting anything (the descriptor + device memory live only for the
    /// caller's query). This is the M5 J5 bridge for a SYNTHESIZED `pg_catalog`/`information_schema`
    /// relation, which has no residency snapshot: synthesize its rows -> this helper -> the existing int4
    /// inner join over the transient payload. Charter: the catalog join runs on the SAME GPU kernels as a
    /// user-table join. The input `rows` are control-plane staging values encoded and uploaded once;
    /// no host-row field is retained and the GPU join reads only the columnar device payload. `&self`:
    /// the upload only needs `cuda_driver_probe_runtime` (also `&self`).
    ///
    /// Mirrors `populate_relational_residency_snapshot_on_gpu`'s payload + descriptor build (the column
    /// lists feed `build_relational_device_payload`, whose offsets the descriptor's resident-column lists
    /// index), but SKIPS the MVCC tuple tail (the join reads only the columnar sections)
    /// and the admission machinery. A 0-row relation is fine: the payload is still a non-empty 8-byte
    /// row-count header (the upload's empty-payload guard never trips), and the inner join then yields an
    /// empty result via the empty-survivor / empty-key short-circuits (an empty side is the join's identity).
    pub(crate) fn build_transient_relation_residency(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, CudaResidentDeviceMemory), ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        let column_names: Vec<String> = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // The resident-column lists name (in catalog order) the columns living in each type-grouped
        // payload section; the descriptor's offset helpers index `build_relational_device_payload`'s
        // sections via these lists, so they MUST use the SAME type filters as the resident builder.
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_copy(gpu_id, &device_payload)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            capacity: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(device_memory.metadata().clone()),
        };
        Ok((snapshot, device_memory))
    }

    /// STRATA S-E.5: [`Engine::build_transient_relation_residency`] with the HtoD upload enqueued
    /// ASYNCHRONOUSLY (pinned staging + a private pooled copy stream), so the DMA overlaps the caller's
    /// host staging of the NEXT chunk and the previous chunk's kernels. The descriptor is complete
    /// immediately (proof metadata is allocation-time); the allocation must not be kernel-read until
    /// the returned pending copy's `wait()`. Falls back to the synchronous copy transparently when
    /// async staging is unavailable (`wait()` is then a no-op).
    /// 6c-3 (adopting the 6c-1 audit's F4): the payload + descriptor WITHOUT any upload — the cold
    /// tier's rebuild/eager-maintenance path constructs chunks for LATER replay (stage_cold_chunk
    /// stamps a fresh proof per upload), so building here must not spend a throwaway DMA — least of
    /// all on the COMMIT path. `device_memory_proof` is None until a replay stamps it.
    pub(crate) fn build_transient_relation_payload_only(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, Vec<u8>), ExecuteError> {
        let (snapshot, payload) = self.build_transient_parts(table, rows)?;
        Ok((snapshot, payload))
    }

    /// The shared payload+descriptor construction (no device work).
    fn build_transient_parts(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, Vec<u8>), ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        let column_names: Vec<String> = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            capacity: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: None,
        };
        Ok((snapshot, device_payload))
    }

    pub(crate) fn build_transient_relation_residency_async(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        gpu_id: u16,
    ) -> Result<
        (
            RelationalResidencySnapshot,
            gpu_db_execution::PendingCudaResidentDeviceCopy,
            // The built device payload BYTES, handed back so the streaming cold tier (S-E.6) can
            // cache them for byte-replay (the upload staged them into pinned memory already).
            Vec<u8>,
        ),
        ExecuteError,
    > {
        let column_names: Vec<String> = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let runtime = self.cuda_driver_probe_runtime();
        let pending = runtime
            .retain_device_memory_copy_async(gpu_id, &device_payload)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            capacity: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(pending.metadata().clone()),
        };
        Ok((snapshot, pending, device_payload))
    }

    pub fn install_benchmark_relational_residency_chunks(
        &mut self,
        install: BenchmarkRelationalResidencyChunkInstall<'_>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        let chunks = install.chunks;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        if chunks.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one retained chunk"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;
        let copied_bytes = chunks
            .iter()
            .try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident chunk ending at byte {end} exceeds allocation {allocated_bytes}"
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident copied byte count overflowed".to_string(),
                    ))
                })
            })?;
        if copied_bytes == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission copied no bytes".to_string(),
            )));
        }

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_chunks(gpu_id, allocated_bytes, chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let allocated_bytes = device_memory.metadata().allocated_bytes;
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, allocated_bytes)?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            capacity: row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: self.committed_seq(),
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_chunks<I>(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedChunkInstall<'_, I>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError>
    where
        I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
    {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_owned_chunks(gpu_id, allocated_bytes, install.chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let allocated_bytes = device_memory.metadata().allocated_bytes;
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, allocated_bytes)?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            capacity: row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: self.committed_seq(),
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_shards(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedShardInstall<'_>,
    ) -> Result<(), ExecuteError> {
        let table = install.table;
        if install.shards.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident shard admission requires at least one shard".to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();
        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident shard admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }

        let _total_resident_bytes = install.shards.iter().try_fold(0_u64, |total, shard| {
            if shard.row_count == 0 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} has no rows",
                    shard.shard_id
                ))));
            }
            if shard.chunks.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} has no retained chunks",
                    shard.shard_id
                ))));
            }
            total.checked_add(shard.resident_bytes).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "benchmark resident shard byte count overflowed".to_string(),
                ))
            })
        })?;
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&install.gpu_id);
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = self.cuda_driver_probe_runtime();
        let mut shards = Vec::new();
        let mut device_memory = BTreeMap::new();
        let mut total_allocated_bytes = 0_u64;
        for shard in install.shards {
            let copied_bytes = shard.chunks.iter().try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > shard.allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard {} chunk ending at byte {end} exceeds allocation {}",
                        shard.shard_id, shard.allocated_bytes
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard copied byte count overflowed".to_string(),
                    ))
                })
            })?;
            if copied_bytes == 0 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} copied no bytes",
                    shard.shard_id
                ))));
            }
            let retained = runtime
                .retain_device_memory_owned_chunks(
                    install.gpu_id,
                    shard.allocated_bytes,
                    shard.chunks,
                )
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard admission failed CUDA retained upload: {err}"
                    )))
                })?;
            let retained_allocated_bytes = retained.metadata().allocated_bytes;
            total_allocated_bytes = total_allocated_bytes
                .checked_add(retained_allocated_bytes)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard allocated byte count overflowed".to_string(),
                    ))
                })?;
            let device_memory_proof = Some(retained.metadata().clone());
            shards.push(RelationalResidentShard {
                shard_id: shard.shard_id,
                row_start: shard.row_start,
                row_count: shard.row_count,
                history_floor_index: self.committed_seq(),
                // Benchmark shards are read DENSE (explicit chunk layouts sized by row_count) and are not
                // append targets.
                capacity: shard.row_count,
                int4_appendable: false,
                // S-d3: benchmark shards carry no zone map -> never pruned (always gathered).
                resident_device_int4_column_stats: Vec::new(),
                // A1/SV2: benchmark shards carry no version metadata (no on-demand deleted_by region) -> the
                // visibility mask is skipped (they read as all-live, correct for latest-snapshot benchmarks).
                resident_bytes: shard.resident_bytes,
                allocated_bytes: retained_allocated_bytes,
                count_header_byte_offset: 0,
                resident_device_int4_columns: shard.resident_device_int4_columns,
                resident_device_int8_columns: Vec::new(), // benchmark chunks are int4-only
                resident_device_numeric_columns: Vec::new(),
                resident_device_bool_columns: Vec::new(),
                resident_device_text_columns: shard.resident_device_text_columns,
                // Benchmark installs carry no NULL metadata (dense, read-only, NULL-free chunks).
                resident_device_null_columns: Vec::new(),
                gpu_id: install.gpu_id,
                schema: catalog_table.schema.clone(),
                table: catalog_table.name.clone(),
                point_route_generation: Arc::new(()),
                device_memory_proof,
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: memory_pressure_active,
                memory_pressure_active,
                // D4: `install_shards` attaches `device_memory` from the map (the one enforcement
                // point); benchmark shards carry no version/identity regions (all-live, read-only).
                device_memory: None,
                deleted_by_region: None,
                created_by_region: None,
                row_id_region: None,
                max_created_by: 0,
            });
            device_memory.insert(shard.shard_id, Arc::new(retained));
        }
        let (_evicted_tables_on_admission, _resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, install.gpu_id, total_allocated_bytes)?;
        shards.sort_by_key(|shard| (shard.row_start, shard.shard_id));
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog().relational_resident_cache.install_shards(
            catalog_table.name,
            shards,
            device_memory,
            &read_state.residency,
        );
        Ok(())
    }
}
