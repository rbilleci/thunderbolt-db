//! Stable identity, dependency, and ABA closure for transaction-owned index lifecycle commands.

use super::*;
use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::engine_streaming_exec::{
    ColdIndexValidationWindow, ColdTableChunks, COLD_INDEX_MAX_HOST_STAGING_BYTES,
};
use crate::engine_transaction_reset::table_schema_digest;

#[cfg(test)]
type IndexValidationPauseHook = (usize, Arc<std::sync::Barrier>, Arc<std::sync::Barrier>);

#[cfg(test)]
fn index_validation_pause_hook() -> &'static std::sync::Mutex<Option<IndexValidationPauseHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<IndexValidationPauseHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

fn pooled_device_bucket(bytes: usize) -> Option<u64> {
    u64::try_from(bytes.max(256).checked_next_power_of_two()?).ok()
}

const COLD_INDEX_INITIAL_BATCH_ROWS: usize = 4 * 1024 * 1024;
const COLD_INDEX_MAX_PLANNED_WINDOWS: usize = 4096;
const COLD_INDEX_MAX_BATCHES: usize = 32;
const COLD_INDEX_MAX_SUBSET_PROOFS: usize = 512;

/// Complete simultaneously-live pooled geometry after predicate compaction has returned its host
/// survivor coordinates. The retained unified input is already charged by the enclosing scope.
/// This preflight is deliberately conservative for descriptor buffers, while allocator-backed
/// leases remain the exact high-water authority during execution.
fn unique_index_group_scratch_bytes(
    source_rows: usize,
    survivor_rows: usize,
    members: &[(usize, SqlType)],
) -> Option<u64> {
    let fixed_members = members
        .iter()
        .filter(|(_, ty)| !matches!(ty, SqlType::Text))
        .count();
    let text_members = members
        .iter()
        .filter(|(_, ty)| matches!(ty, SqlType::Text))
        .count();
    let fixed_width = members.iter().try_fold(0usize, |width, (_, ty)| {
        width.checked_add(match ty {
            SqlType::Numeric { .. } | SqlType::Uuid => 16,
            SqlType::Text => 0,
            _ => 8,
        })
    })?;
    let mut bytes = gpu_db_execution::group_by_duplicate_scratch_bytes(survivor_rows, true)?;
    if fixed_width != 0 {
        bytes = bytes
            .checked_add(pooled_device_bucket(source_rows.checked_mul(fixed_width)?)?)?
            .checked_add(pooled_device_bucket(
                fixed_members.checked_mul(3)?.checked_mul(8)?,
            )?)?;
    }
    if text_members != 0 {
        bytes = bytes.checked_add(pooled_device_bucket(
            text_members.checked_mul(3)?.checked_mul(8)?,
        )?)?;
    }
    Some(bytes)
}

fn is_cuda_allocation_budget_error(error: &ExecuteError) -> bool {
    matches!(
        error,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message.contains("CUDA allocation budget exceeded")
    )
}

fn is_cold_index_retile_error(error: &ExecuteError) -> bool {
    is_cuda_allocation_budget_error(error)
        || matches!(
            error,
            ExecuteError::ResourceExhausted(message)
                if message.contains("cold host staging")
        )
}

struct ColdIndexValidationRequest<'a> {
    table: &'a RelationalTable,
    cold: &'a ColdTableChunks,
    boundary: Index,
    predicate: &'a ResidentExpr,
    members: &'a [(usize, SqlType)],
    gpu_id: u16,
    host_staging_limit: usize,
}

enum TransactionUniqueIndexAuthority {
    Empty,
    Hot {
        shards: Vec<RelationalResidentShard>,
        rows: u64,
    },
    Cold {
        cold: Arc<ColdTableChunks>,
        rows: u64,
    },
}

impl TransactionUniqueIndexAuthority {
    fn rows(&self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Hot { rows, .. } | Self::Cold { rows, .. } => *rows,
        }
    }
}

pub(super) fn pinned_index_validation_input_bytes(
    shards: &[RelationalResidentShard],
    gpu_id: u16,
) -> Result<u64, ExecuteError> {
    let mut allocations = BTreeMap::new();
    for shard in shards {
        for memory in [
            shard.device_memory.as_ref(),
            shard.deleted_by_region.as_ref(),
            shard.created_by_region.as_ref(),
            shard.row_id_region.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if memory.metadata().gpu_id != gpu_id {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction-pinned index-validation shards span CUDA devices".to_string(),
                )));
            }
            allocations.insert(memory.device_ptr(), memory.metadata().allocated_bytes);
        }
    }
    Ok(allocations.values().copied().sum())
}

impl Engine {
    fn validate_transaction_hot_index_authority(
        &self,
        table: &RelationalTable,
        shards: &[RelationalResidentShard],
        boundary: Index,
    ) -> Result<u64, ExecuteError> {
        if shards.is_empty() {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{}\" lost its transaction-pinned GPU shard authority",
                table.name
            )));
        }
        let stale = || {
            ExecuteError::Serialization(format!(
                "relation \"{}\" has a stale or torn transaction-pinned GPU shard authority",
                table.name
            ))
        };
        let expected_int4 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_int8 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_b128 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_bool = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Bool))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_text = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Text))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        if expected_int4.len()
            + expected_int8.len()
            + expected_b128.len()
            + expected_bool.len()
            + expected_text.len()
            != table.columns.len()
        {
            return Err(stale());
        }
        let column_ordinals = table
            .columns
            .iter()
            .enumerate()
            .map(|(ordinal, column)| (column.name.as_str(), ordinal))
            .collect::<BTreeMap<_, _>>();
        let runtime_snapshot = self.router.runtime().snapshot();
        let expected_gpu = shards[0].gpu_id;
        let mut shard_ids = BTreeSet::new();
        let mut rows = 0u64;
        for shard in shards {
            let memory_pressure_active = runtime_snapshot
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            let memory = shard.device_memory.as_ref().ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{}\" lost retained device memory for shard {}",
                    table.name, shard.shard_id
                ))
            })?;
            if shard.schema != table.schema
                || shard.table != table.name
                || !shard.is_valid(memory_pressure_active)
                || shard.capacity < shard.row_count
                || shard.count_header_byte_offset != 0
                || shard.history_floor_index > boundary
                || shard.gpu_id != expected_gpu
                || memory.metadata().gpu_id != expected_gpu
                || !memory.metadata().retained
                || memory.metadata().copied_bytes > memory.metadata().allocated_bytes
                || shard.allocated_bytes != memory.metadata().allocated_bytes
                || shard.device_memory_proof.as_ref() != Some(memory.metadata())
                || !shard_ids.insert(shard.shard_id)
                || shard
                    .resident_device_int4_columns
                    .iter()
                    .map(String::as_str)
                    .ne(expected_int4.iter().copied())
                || shard
                    .resident_device_int4_column_stats
                    .iter()
                    .map(|stats| stats.name.as_str())
                    .ne(expected_int4.iter().copied())
                || shard
                    .resident_device_int8_columns
                    .iter()
                    .map(String::as_str)
                    .ne(expected_int8.iter().copied())
                || shard
                    .resident_device_numeric_columns
                    .iter()
                    .map(String::as_str)
                    .ne(expected_b128.iter().copied())
                || shard
                    .resident_device_bool_columns
                    .iter()
                    .map(|layout| layout.name.as_str())
                    .ne(expected_bool.iter().copied())
                || shard
                    .resident_device_text_columns
                    .iter()
                    .map(|layout| layout.name.as_str())
                    .ne(expected_text.iter().copied())
                || (!expected_text.is_empty() && shard.capacity != shard.row_count)
            {
                return Err(stale());
            }

            let capacity = u64::try_from(shard.capacity).map_err(|_| stale())?;
            let row_count = u64::try_from(shard.row_count).map_err(|_| stale())?;
            let mut cursor = 8u64
                .checked_add(
                    capacity
                        .checked_mul(expected_int4.len() as u64)
                        .and_then(|slots| slots.checked_mul(4))
                        .ok_or_else(stale)?,
                )
                .and_then(|cursor| {
                    capacity
                        .checked_mul(expected_int8.len() as u64)
                        .and_then(|slots| slots.checked_mul(8))
                        .and_then(|bytes| cursor.checked_add(bytes))
                })
                .and_then(|cursor| {
                    capacity
                        .checked_mul(expected_b128.len() as u64)
                        .and_then(|slots| slots.checked_mul(16))
                        .and_then(|bytes| cursor.checked_add(bytes))
                })
                .ok_or_else(stale)?;
            let bitmap_bytes = capacity
                .div_ceil(32)
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .ok_or_else(stale)?;
            for layout in &shard.resident_device_bool_columns {
                if layout.bitmap_byte_offset != cursor {
                    return Err(stale());
                }
                cursor = cursor.checked_add(bitmap_bytes).ok_or_else(stale)?;
            }
            let mut last_null_ordinal = None;
            for layout in &shard.resident_device_null_columns {
                let ordinal = column_ordinals
                    .get(layout.name.as_str())
                    .copied()
                    .ok_or_else(stale)?;
                if last_null_ordinal.is_some_and(|last| ordinal <= last)
                    || layout.bitmap_byte_offset != cursor
                {
                    return Err(stale());
                }
                last_null_ordinal = Some(ordinal);
                cursor = cursor.checked_add(bitmap_bytes).ok_or_else(stale)?;
            }
            for layout in &shard.resident_device_text_columns {
                cursor = cursor
                    .checked_add((8u64.wrapping_sub(cursor % 8)) % 8)
                    .ok_or_else(stale)?;
                if layout.offsets_byte_offset != cursor {
                    return Err(stale());
                }
                cursor = cursor
                    .checked_add(
                        row_count
                            .checked_add(1)
                            .and_then(|entries| entries.checked_mul(8))
                            .ok_or_else(stale)?,
                    )
                    .ok_or_else(stale)?;
                if layout.bytes_byte_offset != cursor {
                    return Err(stale());
                }
                cursor = cursor.checked_add(layout.bytes_len).ok_or_else(stale)?;
            }
            if cursor > memory.metadata().copied_bytes {
                return Err(stale());
            }

            let header = memory
                .read_resident_u64_column(shard.count_header_byte_offset, 1)
                .map_err(|_| stale())?;
            if header.as_slice() != [row_count] {
                // A READ COMMITTED transaction can retain this descriptor while a concurrent
                // writer advances its public open allocation in place. Keep that distinct from
                // every other torn-authority condition: statement staging may defer only this
                // narrow observation to the canonical commit-cut revalidation.
                return Err(ExecuteError::Serialization(format!(
                    "relation \"{}\" has transaction-pinned GPU shard count-header drift",
                    table.name
                )));
            }

            let sidecar_bytes = capacity.checked_mul(8).ok_or_else(stale)?;
            let valid_sidecar = |region: &Arc<gpu_db_execution::CudaResidentDeviceMemory>| -> bool {
                let proof = region.metadata();
                proof.gpu_id == expected_gpu
                    && proof.retained
                    && proof.copied_bytes >= sidecar_bytes
                    && proof.copied_bytes <= proof.allocated_bytes
            };
            if shard
                .deleted_by_region
                .as_ref()
                .is_some_and(|region| !valid_sidecar(region))
                || shard
                    .created_by_region
                    .as_ref()
                    .is_some_and(|region| !valid_sidecar(region))
                || shard
                    .row_id_region
                    .as_ref()
                    .is_some_and(|region| !valid_sidecar(region))
                || (shard.row_count != 0 && shard.row_id_region.is_none())
                || (shard.max_created_by > boundary && shard.created_by_region.is_none())
            {
                return Err(stale());
            }
            rows = rows.checked_add(shard.row_count as u64).ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{}\" transaction-pinned row count overflowed",
                    table.name
                ))
            })?;
        }
        Ok(rows)
    }

    fn transaction_unique_index_authority(
        &self,
        snapshot: &TransactionSnapshot,
        table: &RelationalTable,
    ) -> Result<TransactionUniqueIndexAuthority, ExecuteError> {
        let shards_by_table = snapshot
            .transaction_proven_resident_shards()
            .ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{}\" lost its proven transaction shard generation",
                    table.name
                ))
            })?;
        let expected_generation = snapshot
            .table_versions
            .get(&table.name)
            .map(|rows| rows.get());

        // A captured class/freeze token makes the private cold entry the sole retained relation.
        // Any shard entry is auxiliary or stale in this representation and must not compete with
        // the authority selected at BEGIN.
        if let Some(freeze) = snapshot
            .chunk_authoritative_tables
            .get(&table.name)
            .copied()
        {
            let cold_by_table = snapshot.transaction_proven_cold_chunks().ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{}\" lost its proven transaction cold generation",
                    table.name
                ))
            })?;
            let cold = cold_by_table.get(&table.name).cloned().ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{}\" lost its transaction-pinned cold authority",
                    table.name
                ))
            })?;
            let expected_generation = expected_generation.ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{}\" lost its transaction-pinned storage generation",
                    table.name
                ))
            })?;
            let rows = Self::validate_transaction_cold_index_authority(
                table,
                &cold,
                expected_generation,
                snapshot.boundary,
                Some(freeze),
            )?;
            return Ok(TransactionUniqueIndexAuthority::Cold { cold, rows });
        }

        // A non-empty descriptor vector includes the allocation-backed zero-row root. Once this
        // representation exists it is the transaction's complete private generation; an ordinary
        // cold cache may coexist, but it predates private DML and is never an alternate source.
        if let Some(shards) = shards_by_table
            .get(&table.name)
            .filter(|shards| !shards.is_empty())
            .cloned()
        {
            let rows =
                self.validate_transaction_hot_index_authority(table, &shards, snapshot.boundary)?;
            return Ok(TransactionUniqueIndexAuthority::Hot { shards, rows });
        }

        // An elided/device-authoritative table is not allowed to fall back to its incomplete host
        // generation or to an auxiliary store cache after losing the retained shard descriptor.
        if snapshot.device_authoritative_tables.contains(&table.name) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{}\" lost its transaction-pinned GPU shard authority",
                table.name
            )));
        }

        // A store-authoritative table may legitimately have no hot shards. Its captured cold cache
        // is then an immutable DEVICE-FORMAT staging authority only when both the entry Arc and the
        // underlying table-generation Arc are exactly those retained at snapshot acquisition.
        let cold_by_table = snapshot.transaction_proven_cold_chunks().ok_or_else(|| {
            ExecuteError::Serialization(format!(
                "relation \"{}\" lost its proven transaction cold generation",
                table.name
            ))
        })?;
        if let Some(cold) = cold_by_table.get(&table.name).cloned() {
            let base_cold = snapshot
                .base_streaming_cold_chunks
                .get(&table.name)
                .filter(|base| Arc::ptr_eq(base, &cold))
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "relation \"{}\" changed cold authority inside the transaction",
                        table.name
                    ))
                })?;
            let expected_generation = expected_generation.ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "relation \"{}\" lost its transaction-pinned storage generation",
                    table.name
                ))
            })?;
            let rows = Self::validate_transaction_cold_index_authority(
                table,
                base_cold,
                expected_generation,
                snapshot.boundary,
                None,
            )?;
            return Ok(TransactionUniqueIndexAuthority::Cold { cold, rows });
        }

        // CREATE TABLE is the only admitted transaction operation that can introduce a relation
        // absent from the base catalog. Its staging step creates an explicit empty shard-map slot;
        // with no cold entry and no later DML-produced shard, that typed fact proves cardinality 0.
        if !snapshot
            .catalog
            .relational_catalog
            .contains_key(&table.name)
            && shards_by_table.get(&table.name).is_some_and(Vec::is_empty)
        {
            return Ok(TransactionUniqueIndexAuthority::Empty);
        }

        Err(ExecuteError::Serialization(format!(
            "relation \"{}\" has no complete transaction-pinned GPU authority for index validation",
            table.name
        )))
    }

    #[cfg(test)]
    pub(crate) fn set_index_validation_pause_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *index_validation_pause_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    fn run_index_validation_pause_hook(&self) {
        let hook = {
            let mut hook = index_validation_pause_hook()
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

    pub(crate) fn transaction_created_table_implicit_index_identities(
        catalog: &CatalogSnapshot,
        table_name: &str,
    ) -> Result<Vec<BinaryCatalogIndexIdentity>, ExecuteError> {
        let table = required_table(catalog, table_name)?;
        table
            .indexes
            .iter()
            .filter(|index| index.primary_key || index.unique_constraint)
            .map(|index| index_identity(catalog, index))
            .collect()
    }

    pub(crate) fn validate_transaction_create_index_device(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        catalog: &CatalogSnapshot,
        create: &CreateIndex,
    ) -> Result<(), ExecuteError> {
        if !create.unique {
            return Ok(());
        }
        let table = required_table(catalog, &create.table)?;
        let members = create
            .columns
            .iter()
            .map(|name| {
                table
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, column)| &column.name == name)
                    .map(|(idx, column)| (idx, column.ty))
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "column \"{name}\" does not exist"
                        )))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.validate_transaction_unique_index_members_device(
            snapshot,
            table,
            &create.name,
            &members,
            None,
        )
    }

    /// Check only the transaction-private typed-INSERT shard suffix for an immediate constraint
    /// error.  The public-prefix shards intentionally stay out of this statement-time proof:
    /// their open allocation may be advanced by a concurrent writer after this transaction's
    /// READ COMMITTED snapshot.  The canonical terminal refreshes under the commit boundary and
    /// remains the sole authority for prefix-vs-newly-committed history.
    pub(crate) fn validate_transaction_private_typed_unique_indexes_device(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        table: &RelationalTable,
        private_shards: &[RelationalResidentShard],
    ) -> Result<(), ExecuteError> {
        if private_shards.is_empty() {
            return Ok(());
        }
        for index in table.indexes.iter().filter(|index| index.unique) {
            let members = index
                .key_columns
                .iter()
                .map(|name| {
                    table
                        .columns
                        .iter()
                        .enumerate()
                        .find(|(_, column)| &column.name == name)
                        .map(|(idx, column)| (idx, column.ty))
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::Durability(format!(
                                "unique index \"{}\" references missing column \"{name}\"",
                                index.name
                            )))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.validate_transaction_unique_index_members_device(
                snapshot,
                table,
                &index.name,
                &members,
                Some(private_shards),
            )?;
        }
        Ok(())
    }

    /// Prove an existing UNIQUE/PRIMARY KEY index against the complete transaction-private GPU
    /// generation.  This is deliberately shared by transactional CREATE INDEX and COMMIT: a
    /// statement-local typed batch can prove itself before WAL, but multiple private shards from
    /// separate statements must be grouped together before publication.
    fn validate_transaction_unique_index_members_device(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        table: &RelationalTable,
        index_name: &str,
        members: &[(usize, SqlType)],
        hot_shard_suffix: Option<&[RelationalResidentShard]>,
    ) -> Result<(), ExecuteError> {
        #[cfg(test)]
        self.run_index_validation_pause_hook();
        let authority = match hot_shard_suffix {
            Some(shards) => TransactionUniqueIndexAuthority::Hot {
                rows: self.validate_transaction_hot_index_authority(
                    table,
                    shards,
                    snapshot.boundary,
                )?,
                shards: shards.to_vec(),
            },
            None => self.transaction_unique_index_authority(snapshot, table)?,
        };
        let private_rows = authority.rows();
        if private_rows < 2 {
            return Ok(());
        }
        let predicate = members
            .iter()
            .map(|(column, _)| ResidentExpr::IsNull {
                col: *column,
                is_not_null: true,
            })
            .reduce(|left, right| ResidentExpr::Binary {
                op: ResidentBinaryOp::And,
                lhs: Box::new(left),
                rhs: Box::new(right),
            })
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "unique index has no key columns".to_string(),
                ))
            })?;
        let _read_scope = self.enter_transaction_read(Arc::clone(snapshot));
        let gpu_id = match &authority {
            TransactionUniqueIndexAuthority::Cold { cold, .. } => cold
                .chunks
                .iter()
                .find(|chunk| chunk.row_count != 0)
                .map(|chunk| chunk.snapshot.gpu_id)
                .unwrap_or_else(|| self.planner.default_gpu_id()),
            TransactionUniqueIndexAuthority::Hot { shards, .. } => shards[0].gpu_id,
            TransactionUniqueIndexAuthority::Empty => unreachable!(
                "empty UNIQUE-validation authority returned after a two-row cardinality gate"
            ),
        };
        // Query-local STRATA discipline: include only the exact transaction-pinned operand bytes,
        // deduplicated by device allocation identity. Unrelated caches cannot make this DDL fail,
        // and the global admission allocator lock is never held across a kernel launch.
        let scratch_limit = match (&authority, self.relational_residency_budget_bytes(gpu_id)) {
            // A chunk-authoritative relation has no device-resident input: its configured STRATA
            // budget is the complete transient working-set cap, exactly like the streaming folds.
            // Historical/global caches are not operands of this proof and are governed by canonical
            // residency admission separately.
            (TransactionUniqueIndexAuthority::Cold { .. }, Some(budget)) => budget,
            (TransactionUniqueIndexAuthority::Hot { shards, .. }, Some(budget)) => {
                let input_bytes = pinned_index_validation_input_bytes(shards, gpu_id)?;
                budget.checked_sub(input_bytes).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "transactional UNIQUE index validation has no GPU {gpu_id} residency \
                         headroom: its pinned input owns {input_bytes} bytes under budget {budget}"
                    )))
                })?
            }
            (TransactionUniqueIndexAuthority::Empty, _) => unreachable!(
                "empty UNIQUE-validation authority returned after a two-row cardinality gate"
            ),
            (_, None) => u64::MAX,
        };
        let _allocation_scope = gpu_db_execution::CudaAllocationScope::with_budget(scratch_limit);
        let outcome = match authority {
            TransactionUniqueIndexAuthority::Cold { cold, .. } => {
                let total_rows = usize::try_from(private_rows).map_err(|_| {
                    ExecuteError::ResourceExhausted(
                        "transactional UNIQUE index validation row count exceeds bounded host framing"
                            .to_string(),
                    )
                })?;
                let mut tile_rows = total_rows.clamp(1, COLD_INDEX_INITIAL_BATCH_ROWS);
                #[cfg(test)]
                {
                    let override_rows = self
                        .read_state
                        .residency
                        .cold_index_validation_batch_rows_override
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if override_rows != 0 {
                        tile_rows = total_rows.clamp(1, override_rows);
                    }
                }
                let validation = ColdIndexValidationRequest {
                    table,
                    cold: &cold,
                    boundary: snapshot.boundary,
                    predicate: &predicate,
                    members,
                    gpu_id,
                    host_staging_limit: usize::try_from(scratch_limit)
                        .unwrap_or(usize::MAX)
                        .min(COLD_INDEX_MAX_HOST_STAGING_BYTES),
                };
                let duplicate = loop {
                    match self.transaction_cold_index_windows_have_duplicate(&validation, tile_rows)
                    {
                        Ok(duplicate) => break Ok(duplicate),
                        Err(error) if is_cold_index_retile_error(&error) && tile_rows > 1 => {
                            // The exact allocator rejected this geometry before crossing the cap.
                            // Re-tile every immutable cold chunk into smaller DEVICE-FORMAT windows and
                            // restart the read-only proof. No catalog/WAL/publication state exists yet.
                            tile_rows = tile_rows.div_ceil(2);
                        }
                        Err(error) if is_cuda_allocation_budget_error(&error) => {
                            break Err(ExecuteError::ResourceExhausted(format!(
                            "transactional UNIQUE index validation cannot fit one-row GPU tiles \
                             within {scratch_limit} bytes of residency headroom; raise the GPU \
                             residency budget and retry"
                        )));
                        }
                        Err(error) => break Err(error),
                    }
                }?;
                if duplicate {
                    Err(ExecuteError::Engine(EngineError::UniqueViolation(format!(
                        "duplicate key value violates unique index \"{}\"",
                        index_name
                    ))))
                } else {
                    Ok(())
                }
            }
            TransactionUniqueIndexAuthority::Hot { shards, .. } => {
                let unified = self.build_scoped_sharded_unified_exec_source_from_shards(
                    table,
                    Some(&predicate),
                    snapshot.boundary,
                    shards,
                )?;
                self.transaction_unique_index_source_has_duplicate(
                    table, &unified, &predicate, members,
                )
                .and_then(|duplicate| {
                    if duplicate {
                        Err(ExecuteError::Engine(EngineError::UniqueViolation(format!(
                            "duplicate key value violates unique index \"{}\"",
                            index_name
                        ))))
                    } else {
                        Ok(())
                    }
                })
            }
            TransactionUniqueIndexAuthority::Empty => unreachable!(
                "empty UNIQUE-validation authority returned after a two-row cardinality gate"
            ),
        };
        #[cfg(test)]
        self.read_state
            .residency
            .cold_index_validation_peak_device_bytes
            .fetch_max(
                _allocation_scope.peak_bytes(),
                std::sync::atomic::Ordering::Relaxed,
            );
        outcome
    }

    /// Revalidate final existing unique indexes for surviving typed-insert tables. The selected
    /// authority is the transaction-pinned private shard generation, so this catches duplicate
    /// typed batches staged by distinct statements without decoding row images or staging a host
    /// relation back to the device. Legacy-only transactions keep their established validator and
    /// do not pay this typed-proof catalog walk.
    pub(crate) fn validate_transaction_final_typed_unique_indexes_device(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        catalog: &CatalogSnapshot,
        typed_tables: &BTreeSet<String>,
    ) -> Result<(), ExecuteError> {
        for table_name in typed_tables {
            let table = required_table(catalog, table_name)?;
            for index in table.indexes.iter().filter(|index| index.unique) {
                let members = index
                    .key_columns
                    .iter()
                    .map(|name| {
                        table
                            .columns
                            .iter()
                            .enumerate()
                            .find(|(_, column)| &column.name == name)
                            .map(|(idx, column)| (idx, column.ty))
                            .ok_or_else(|| {
                                ExecuteError::Engine(EngineError::Durability(format!(
                                    "unique index \"{}\" references missing column \"{name}\"",
                                    index.name
                                )))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.validate_transaction_unique_index_members_device(
                    snapshot,
                    table,
                    &index.name,
                    &members,
                    None,
                )?;
            }
        }
        Ok(())
    }

    fn transaction_cold_index_windows_have_duplicate(
        &self,
        validation: &ColdIndexValidationRequest<'_>,
        tile_rows: usize,
    ) -> Result<bool, ExecuteError> {
        let bounded = || {
            ExecuteError::ResourceExhausted(format!(
                "transactional UNIQUE index validation for relation \"{}\" exceeds the bounded \
                 out-of-core proof plan at {tile_rows} rows per batch; raise the GPU residency \
                 budget, compact/admit the relation, and retry",
                validation.table.name
            ))
        };
        let max_batches = {
            #[cfg(test)]
            {
                let limit = self
                    .read_state
                    .residency
                    .cold_index_validation_max_batches_override
                    .load(std::sync::atomic::Ordering::Relaxed);
                if limit != 0 {
                    COLD_INDEX_MAX_BATCHES.min(limit)
                } else {
                    COLD_INDEX_MAX_BATCHES
                }
            }
            #[cfg(not(test))]
            {
                COLD_INDEX_MAX_BATCHES
            }
        };
        let mut batches = Vec::<Vec<ColdIndexValidationWindow>>::new();
        let mut batch = Vec::<ColdIndexValidationWindow>::new();
        let mut batch_rows = 0usize;
        let mut planned_windows = 0usize;
        for (chunk_ordinal, chunk) in validation.cold.chunks.iter().enumerate() {
            let row_count = usize::try_from(chunk.row_count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "cold index-validation row count exceeds host framing".to_string(),
                ))
            })?;
            let mut row_start = 0usize;
            while row_start < row_count {
                let remaining_batch = tile_rows.checked_sub(batch_rows).ok_or_else(bounded)?;
                let count = remaining_batch.min(row_count - row_start);
                if count == 0 || planned_windows == COLD_INDEX_MAX_PLANNED_WINDOWS {
                    return Err(bounded());
                }
                batch.try_reserve(1).map_err(|_| bounded())?;
                batch.push(ColdIndexValidationWindow {
                    chunk_ordinal,
                    row_start,
                    row_count: count,
                });
                planned_windows += 1;
                batch_rows = batch_rows.checked_add(count).ok_or_else(bounded)?;
                row_start = row_start.checked_add(count).ok_or_else(bounded)?;
                #[cfg(test)]
                self.read_state
                    .residency
                    .cold_index_validation_peak_planned_windows
                    .fetch_max(planned_windows as u64, std::sync::atomic::Ordering::Relaxed);
                if batch_rows == tile_rows {
                    if batches.len() == max_batches {
                        return Err(bounded());
                    }
                    batches.try_reserve(1).map_err(|_| bounded())?;
                    batches.push(std::mem::take(&mut batch));
                    batch_rows = 0;
                }
            }
        }
        if !batch.is_empty() {
            if batches.len() == max_batches {
                return Err(bounded());
            }
            batches.try_reserve(1).map_err(|_| bounded())?;
            batches.push(batch);
        }
        if batches.is_empty() {
            return Ok(false);
        }
        let proof_count = if batches.len() == 1 {
            1
        } else {
            batches
                .len()
                .checked_mul(batches.len() - 1)
                .and_then(|pairs| pairs.checked_div(2))
                .ok_or_else(bounded)?
        };
        if proof_count > COLD_INDEX_MAX_SUBSET_PROOFS {
            return Err(bounded());
        }
        if batches.len() == 1 {
            return self.transaction_cold_index_subset_has_duplicate(validation, &batches[0]);
        }
        // Exact bounded out-of-core proof: a batch may combine many source chunks but owns at most
        // `tile_rows`. Every unordered batch pair is staged, filtered, and grouped by the ordinary
        // GPU operators. A within-batch duplicate appears in every pair containing that batch; a
        // cross-batch duplicate appears in its unique pair. Host descriptors and GPU launches are
        // both hard-capped before the first allocation, so lowering the budget cannot manufacture
        // an unbounded Vec or quadratic process-lifetime loop.
        for right in 1..batches.len() {
            for left in 0..right {
                let mut windows = Vec::new();
                windows
                    .try_reserve_exact(batches[left].len() + batches[right].len())
                    .map_err(|_| bounded())?;
                windows.extend_from_slice(&batches[left]);
                windows.extend_from_slice(&batches[right]);
                if self.transaction_cold_index_subset_has_duplicate(validation, &windows)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn transaction_cold_index_subset_has_duplicate(
        &self,
        validation: &ColdIndexValidationRequest<'_>,
        windows: &[ColdIndexValidationWindow],
    ) -> Result<bool, ExecuteError> {
        #[cfg(test)]
        self.read_state
            .residency
            .cold_index_validation_subset_proofs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let shards = self.stage_cold_index_validation_shards(
            validation.table,
            validation.cold,
            validation.boundary,
            validation.gpu_id,
            windows,
            validation.host_staging_limit,
        )?;
        let unified = self.build_scoped_sharded_unified_exec_source_from_shards(
            validation.table,
            Some(validation.predicate),
            validation.boundary,
            shards,
        )?;
        self.transaction_unique_index_source_has_duplicate(
            validation.table,
            &unified,
            validation.predicate,
            validation.members,
        )
    }

    fn transaction_unique_index_source_has_duplicate(
        &self,
        table: &RelationalTable,
        unified: &crate::engine_expr::ShardedUnifiedExecSource,
        predicate: &ResidentExpr,
        members: &[(usize, SqlType)],
    ) -> Result<bool, ExecuteError> {
        let indices = self.lower_resident_predicate(
            predicate,
            table,
            &unified.src.descriptor,
            &unified.src.device_memory,
            unified.src.row_count,
            unified.visibility,
        )?;
        if indices.is_empty() {
            // SQL UNIQUE treats every key containing NULL as distinct. Predicate compaction is
            // the device verdict that no fully-present key exists, so no wide key, descriptor, or
            // GROUP scratch is needed and a tight budget cannot manufacture a false rejection.
            return Ok(false);
        }
        let source_rows = usize::try_from(unified.src.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "UNIQUE index validation source exceeds host framing".to_string(),
            ))
        })?;
        let required = unique_index_group_scratch_bytes(source_rows, indices.len(), members)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "UNIQUE index validation scratch geometry overflowed".to_string(),
                ))
            })?;
        gpu_db_execution::CudaAllocationScope::ensure_available(required)
            .map_err(|error| ExecuteError::Engine(EngineError::ApplyFailed(error.to_string())))?;
        crate::engine_expr::composite_group_has_duplicate(
            &unified.src.descriptor,
            table,
            &unified.src.device_memory,
            members,
            &indices,
            unified.src.row_count,
        )
    }

    /// Revalidate only transaction-created UNIQUE indexes that remain in the final private
    /// catalog. Statement-time validation already proves the CREATE point; a later ordered DROP
    /// intentionally removes that constraint and must allow subsequent private duplicates.
    pub(crate) fn validate_transaction_surviving_created_unique_indexes_device(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        catalog: &CatalogSnapshot,
        commands: &[BinaryTransactionCatalogCommand],
        identities: &[BinaryTransactionIndexLifecycleOperationIdentity],
    ) -> Result<(), ExecuteError> {
        for identity in identities {
            let Some(operation) = usize::try_from(identity.command_index)
                .ok()
                .and_then(|command_index| commands.get(command_index))
            else {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "transactional index identity lost its catalog command".to_string(),
                )));
            };
            let Command::CreateIndex(create) = &operation.command else {
                continue;
            };
            if !create.unique {
                continue;
            }
            let target = identity.targets.first().ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "transactional CREATE INDEX identity lost its target".to_string(),
                ))
            })?;
            let created_oid = target
                .index_after
                .as_ref()
                .map(|index| index.oid)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "transactional CREATE INDEX identity lost its stable postimage".to_string(),
                    ))
                })?;
            let Some((table, index)) = catalog.relational_catalog.values().find_map(|table| {
                table
                    .indexes
                    .iter()
                    .find(|index| index.oid == created_oid)
                    .map(|index| (table, index))
            }) else {
                // An ordered DROP removed this exact allocation identity. A same-name recreate has
                // a different OID and is validated through its own CREATE identity.
                continue;
            };
            if !index.unique || index.table != table.name {
                return Err(ExecuteError::Engine(EngineError::Durability(format!(
                    "transaction-created unique index OID {created_oid} changed semantic owner"
                ))));
            }
            self.validate_transaction_create_index_device(
                snapshot,
                catalog,
                &CreateIndex {
                    name: index.name.clone(),
                    table: table.name.clone(),
                    column: index.column.clone(),
                    columns: index.key_columns.clone(),
                    unique: true,
                },
            )?;
        }
        Ok(())
    }

    pub(crate) fn transaction_index_table_identities(
        catalog: &CatalogSnapshot,
        command: &Command,
    ) -> Result<BTreeMap<String, u32>, ExecuteError> {
        let mut identities = BTreeMap::new();
        match command {
            Command::CreateIndex(create) => {
                let table = required_table(catalog, &create.table)?;
                identities.insert(table.name.clone(), table.oid);
            }
            Command::RenameIndex(rename) => {
                let (table, _) = required_index(catalog, &rename.old_name)?;
                identities.insert(table.name.clone(), table.oid);
            }
            Command::DropIndex(drop) => {
                for name in &drop.names {
                    if let Some((table, _)) = find_index(catalog, name)? {
                        identities.insert(table.name.clone(), table.oid);
                    } else if catalog
                        .pg_class_relation_kind(name)
                        .map_err(ExecuteError::Engine)?
                        .is_some()
                    {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "relation {name:?} is not an index"
                        ))));
                    } else if !drop.if_exists {
                        return Err(ExecuteError::UndefinedRelation(format!(
                            "transactional index target {name:?}"
                        )));
                    }
                }
            }
            _ => {}
        }
        Ok(identities)
    }

    pub(crate) fn transaction_index_operation_identity(
        command_index: u32,
        ordinal: u32,
        before: &CatalogSnapshot,
        after: &CatalogSnapshot,
        command: &Command,
    ) -> Result<BinaryTransactionIndexLifecycleOperationIdentity, ExecuteError> {
        let targets = match command {
            Command::CreateIndex(create) => {
                ensure_index_absent(before, &create.name, "CREATE INDEX target")?;
                let table_before = required_table(before, &create.table)?;
                let (table_after, index_after) = required_index(after, &create.name)?;
                vec![BinaryTransactionIndexLifecycleTargetIdentity {
                    before_name: create.name.clone(),
                    owner_name: Some(table_before.name.clone()),
                    table_before: Some(table_identity(before, table_before)?),
                    index_before: None,
                    after_name: Some(create.name.clone()),
                    table_after: Some(table_identity(after, table_after)?),
                    index_after: Some(index_identity(after, index_after)?),
                }]
            }
            Command::RenameIndex(rename) => {
                let (table_before, index_before) = required_index(before, &rename.old_name)?;
                ensure_index_absent(before, &rename.new_name, "ALTER INDEX destination")?;
                let (table_after, index_after) = required_index(after, &rename.new_name)?;
                vec![BinaryTransactionIndexLifecycleTargetIdentity {
                    before_name: rename.old_name.clone(),
                    owner_name: Some(table_before.name.clone()),
                    table_before: Some(table_identity(before, table_before)?),
                    index_before: Some(index_identity(before, index_before)?),
                    after_name: Some(rename.new_name.clone()),
                    table_after: Some(table_identity(after, table_after)?),
                    index_after: Some(index_identity(after, index_after)?),
                }]
            }
            Command::DropIndex(drop) => {
                let mut targets = Vec::with_capacity(drop.names.len());
                for name in &drop.names {
                    let (owner_name, table_before, index_before, table_after) =
                        if let Some((table, index)) = find_index(before, name)? {
                            (
                                Some(table.name.clone()),
                                Some(table_identity(before, table)?),
                                Some(index_identity(before, index)?),
                                Some(table_identity(
                                    after,
                                    after.relational_catalog.get(&table.name).ok_or_else(|| {
                                        ExecuteError::Engine(EngineError::Durability(format!(
                                            "DROP INDEX removed owning table {:?}",
                                            table.name
                                        )))
                                    })?,
                                )?),
                            )
                        } else {
                            ensure_index_absent(before, name, "DROP INDEX target")?;
                            (None, None, None, None)
                        };
                    ensure_index_absent(after, name, "DROP INDEX postimage")?;
                    targets.push(BinaryTransactionIndexLifecycleTargetIdentity {
                        before_name: name.clone(),
                        owner_name,
                        table_before,
                        index_before,
                        after_name: None,
                        table_after,
                        index_after: None,
                    });
                }
                targets
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional index identity names an unsupported command".to_string(),
                )))
            }
        };
        let identity = BinaryTransactionIndexLifecycleOperationIdentity {
            command_index,
            ordinal,
            targets,
        };
        if !valid_index_lifecycle_operation_identity(command, &identity) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "transactional index identity is not canonical".to_string(),
            )));
        }
        Ok(identity)
    }

    pub(crate) fn validate_transaction_index_before(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionIndexLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        if !valid_index_lifecycle_operation_identity(command, identity) {
            return Err(EngineError::Durability(
                "ordered index operation carries a noncanonical identity".to_string(),
            ));
        }
        match command {
            Command::CreateIndex(create) => {
                ensure_index_absent(catalog, &create.name, "CREATE INDEX target")
                    .map_err(execute_to_engine)?;
                let table = required_table(catalog, &create.table).map_err(execute_to_engine)?;
                validate_before(
                    &identity.targets[0],
                    Some(table.name.as_str()),
                    Some(table_identity(catalog, table).map_err(execute_to_engine)?),
                    None,
                )
            }
            Command::RenameIndex(rename) => {
                let (table, index) =
                    required_index(catalog, &rename.old_name).map_err(execute_to_engine)?;
                ensure_index_absent(catalog, &rename.new_name, "ALTER INDEX destination")
                    .map_err(execute_to_engine)?;
                validate_before(
                    &identity.targets[0],
                    Some(table.name.as_str()),
                    Some(table_identity(catalog, table).map_err(execute_to_engine)?),
                    Some(index_identity(catalog, index).map_err(execute_to_engine)?),
                )
            }
            Command::DropIndex(drop) => {
                for (name, target) in drop.names.iter().zip(&identity.targets) {
                    if let Some((table, index)) =
                        find_index(catalog, name).map_err(execute_to_engine)?
                    {
                        validate_before(
                            target,
                            Some(table.name.as_str()),
                            Some(table_identity(catalog, table).map_err(execute_to_engine)?),
                            Some(index_identity(catalog, index).map_err(execute_to_engine)?),
                        )?;
                    } else {
                        ensure_index_absent(catalog, name, "DROP INDEX target")
                            .map_err(execute_to_engine)?;
                        validate_before(target, None, None, None)?;
                    }
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "ordered index identity names an unsupported command".to_string(),
            )),
        }
    }

    pub(crate) fn validate_transaction_index_after(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionIndexLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        match command {
            Command::CreateIndex(create) => {
                let (table, index) =
                    required_index(catalog, &create.name).map_err(execute_to_engine)?;
                validate_after(catalog, &identity.targets[0], &create.name, table, index)
            }
            Command::RenameIndex(rename) => {
                ensure_index_absent(catalog, &rename.old_name, "ALTER INDEX old binding")
                    .map_err(execute_to_engine)?;
                let (table, index) =
                    required_index(catalog, &rename.new_name).map_err(execute_to_engine)?;
                validate_after(
                    catalog,
                    &identity.targets[0],
                    &rename.new_name,
                    table,
                    index,
                )
            }
            Command::DropIndex(drop) => {
                for (name, target) in drop.names.iter().zip(&identity.targets) {
                    ensure_index_absent(catalog, name, "DROP INDEX postimage")
                        .map_err(execute_to_engine)?;
                    match (
                        target.owner_name.as_deref(),
                        target.index_before.as_ref(),
                        target.table_after.as_ref(),
                    ) {
                        (Some(owner), Some(_), Some(expected)) => {
                            let table = catalog.relational_catalog.get(owner).ok_or_else(|| {
                                EngineError::Durability(format!(
                                    "ordered DROP INDEX removed owner relation {owner:?}"
                                ))
                            })?;
                            let observed =
                                table_identity(catalog, table).map_err(execute_to_engine)?;
                            if &observed != expected {
                                return Err(EngineError::Durability(format!(
                                    "ordered DROP INDEX target {name:?} changed owner-table postimage"
                                )));
                            }
                        }
                        (None, None, None) => {}
                        _ => {
                            return Err(EngineError::Durability(format!(
                                "ordered DROP INDEX target {name:?} carries an incomplete postimage"
                            )))
                        }
                    }
                }
                if identity
                    .targets
                    .iter()
                    .any(|target| target.after_name.is_some() || target.index_after.is_some())
                {
                    return Err(EngineError::Durability(
                        "ordered DROP INDEX carries a non-absent index postimage".to_string(),
                    ));
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "ordered index identity names an unsupported command".to_string(),
            )),
        }
    }
}

fn validate_before(
    target: &BinaryTransactionIndexLifecycleTargetIdentity,
    owner_name: Option<&str>,
    table: Option<BinaryCatalogRelationIdentity>,
    index: Option<BinaryCatalogIndexIdentity>,
) -> Result<(), EngineError> {
    if target.owner_name.as_deref() != owner_name
        || target.table_before != table
        || target.index_before != index
    {
        return Err(EngineError::Durability(format!(
            "ordered index target {:?} changed stable preimage or owner dependency",
            target.before_name
        )));
    }
    Ok(())
}

fn validate_after(
    catalog: &CatalogSnapshot,
    target: &BinaryTransactionIndexLifecycleTargetIdentity,
    name: &str,
    table: &RelationalTable,
    index: &RelationalIndex,
) -> Result<(), EngineError> {
    let observed_table = table_identity(catalog, table).map_err(execute_to_engine)?;
    let observed_index = index_identity(catalog, index).map_err(execute_to_engine)?;
    if target.after_name.as_deref() != Some(name)
        || target.owner_name.as_deref() != Some(table.name.as_str())
        || target.table_after.as_ref() != Some(&observed_table)
        || target.index_after.as_ref() != Some(&observed_index)
    {
        return Err(EngineError::Durability(format!(
            "ordered index target {name:?} changed stable postimage"
        )));
    }
    Ok(())
}

fn required_table<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<&'a RelationalTable, ExecuteError> {
    match catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
    {
        Some(PgClassRelationKind::Table) => catalog.relational_catalog.get(name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(format!(
                "catalog table binding {name:?} disappeared"
            )))
        }),
        Some(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "relation {name:?} is not a table"
        )))),
        None => Err(ExecuteError::UndefinedRelation(format!(
            "transactional index owner {name:?}"
        ))),
    }
}

fn find_index<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<Option<(&'a RelationalTable, &'a RelationalIndex)>, ExecuteError> {
    if catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
        != Some(PgClassRelationKind::Index)
    {
        return Ok(None);
    }
    let mut found = catalog.relational_catalog.values().filter_map(|table| {
        table
            .indexes
            .iter()
            .find(|index| index.name == name)
            .map(|index| (table, index))
    });
    let first = found.next();
    if found.next().is_some() {
        return Err(ExecuteError::Engine(EngineError::Durability(format!(
            "catalog contains duplicate index binding {name:?}"
        ))));
    }
    Ok(first)
}

fn required_index<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<(&'a RelationalTable, &'a RelationalIndex), ExecuteError> {
    if let Some(found) = find_index(catalog, name)? {
        return Ok(found);
    }
    if catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
        .is_some()
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "relation {name:?} is not an index"
        ))));
    }
    Err(ExecuteError::UndefinedRelation(format!(
        "transactional index target {name:?}"
    )))
}

fn ensure_index_absent(
    catalog: &CatalogSnapshot,
    name: &str,
    context: &str,
) -> Result<(), ExecuteError> {
    if find_index(catalog, name)?.is_some()
        || catalog
            .pg_class_relation_kind(name)
            .map_err(ExecuteError::Engine)?
            .is_some()
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "transactional {context} {name:?} is not absent"
        ))));
    }
    Ok(())
}

pub(super) fn table_identity(
    catalog: &CatalogSnapshot,
    table: &RelationalTable,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    let mut body = Vec::new();
    body.extend_from_slice(b"GPUDBTABLEINDEXIDENTITY2");
    body.extend_from_slice(&table_schema_digest(table)?);
    push_len(&mut body, table.indexes.len())?;
    for index in &table.indexes {
        body.extend_from_slice(&index.oid.to_le_bytes());
        push_identity_comment(
            &mut body,
            catalog
                .relational_comments
                .get(&RelationalCommentTarget::Index {
                    index: index.name.clone(),
                }),
        )?;
        if index.primary_key || index.unique_constraint {
            push_identity_comment(
                &mut body,
                catalog
                    .relational_comments
                    .get(&RelationalCommentTarget::Constraint {
                        table: table.name.clone(),
                        constraint: index.name.clone(),
                    }),
            )?;
        }
    }
    Ok(BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::Table,
        oid: table.oid,
        digest: gpu_db_wal::canonical_request_digest(&body),
    })
}

pub(super) fn index_identity(
    catalog: &CatalogSnapshot,
    index: &RelationalIndex,
) -> Result<BinaryCatalogIndexIdentity, ExecuteError> {
    let mut body = Vec::new();
    body.extend_from_slice(b"GPUDBINDEXIDENTITY2");
    body.extend_from_slice(&index.oid.to_le_bytes());
    push_string(&mut body, &index.name)?;
    push_string(&mut body, &index.table)?;
    push_string(&mut body, &index.column)?;
    push_len(&mut body, index.key_columns.len())?;
    for column in &index.key_columns {
        push_string(&mut body, column)?;
    }
    body.extend_from_slice(&[
        u8::from(index.unique),
        u8::from(index.primary_key),
        u8::from(index.unique_constraint),
    ]);
    push_identity_comment(
        &mut body,
        catalog
            .relational_comments
            .get(&RelationalCommentTarget::Index {
                index: index.name.clone(),
            }),
    )?;
    if index.primary_key || index.unique_constraint {
        push_identity_comment(
            &mut body,
            catalog
                .relational_comments
                .get(&RelationalCommentTarget::Constraint {
                    table: index.table.clone(),
                    constraint: index.name.clone(),
                }),
        )?;
    }
    let table_oid = catalog
        .relational_catalog
        .get(&index.table)
        .map(|table| table.oid)
        .ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(format!(
                "index {:?} lost owning table {:?}",
                index.name, index.table
            )))
        })?;
    Ok(BinaryCatalogIndexIdentity {
        oid: index.oid,
        table_oid,
        digest: gpu_db_wal::canonical_request_digest(&body),
    })
}

fn push_identity_comment(body: &mut Vec<u8>, comment: Option<&String>) -> Result<(), ExecuteError> {
    match comment {
        Some(comment) => {
            body.push(1);
            push_string(body, comment)
        }
        None => {
            body.push(0);
            Ok(())
        }
    }
}

fn push_len(body: &mut Vec<u8>, len: usize) -> Result<(), ExecuteError> {
    body.extend_from_slice(
        &u64::try_from(len)
            .map_err(|_| {
                ExecuteError::Unsupported(
                    "index identity count exceeds durable framing".to_string(),
                )
            })?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_string(body: &mut Vec<u8>, value: &str) -> Result<(), ExecuteError> {
    push_len(body, value.len())?;
    body.extend_from_slice(value.as_bytes());
    Ok(())
}

fn execute_to_engine(error: ExecuteError) -> EngineError {
    match error {
        ExecuteError::Engine(error) => error,
        other => EngineError::Durability(other.to_string()),
    }
}
