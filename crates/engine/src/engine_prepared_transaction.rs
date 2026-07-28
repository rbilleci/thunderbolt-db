//! Engine-issued prepared transaction routes.
//!
//! A route owns typed SQL templates plus catalog and device-index requirements. Callers can bind
//! values but cannot manufacture the proof that promotes work into W1/T8/T32. Submission verifies
//! current resident generations and mandatory named-index coverage before BEGIN, then requires the
//! indexed device route for every keyed read/update/delete; a decline aborts before WAL.

use super::*;
use crate::engine_mutation_admission::{
    predeclared_access_plan, validate_transaction_characteristics, PredeclaredSubmissionContext,
};

mod resources;
use resources::PreparedResourceEstimate;
mod service;
pub(crate) use service::PreparedTransactionServiceController;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreparedTransactionClassAdmissions {
    pub w1: u64,
    pub t8: u64,
    pub t32: u64,
    pub general: u64,
}

#[derive(Debug, Clone)]
struct PreparedTableRequirement {
    table: RelationalTable,
    probe_key_ids: BTreeSet<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRouteProof {
    engine_identity: std::sync::Weak<ReadState>,
    tables: BTreeMap<String, PreparedTableRequirement>,
    indexed_execution_eligible: bool,
}

impl PreparedRouteProof {
    pub(crate) fn table_names(&self) -> BTreeSet<String> {
        self.tables.keys().cloned().collect()
    }
}

#[derive(Debug, Clone)]
pub struct PreparedTransactionRoute {
    operations: Vec<PreparedCommand>,
    descriptions: Vec<PreparedCommandDescription>,
    characteristics: TransactionCharacteristics,
    proof: Arc<PreparedRouteProof>,
}

impl PreparedTransactionRoute {
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn parameter_types(&self, operation: usize) -> Option<&[SqlType]> {
        self.descriptions
            .get(operation)
            .map(|description| description.parameter_types.as_slice())
    }

    pub fn bind(
        &self,
        parameters: Vec<Vec<SqlValue>>,
    ) -> Result<BoundPreparedTransactionRoute, ExecuteError> {
        if parameters.len() != self.operations.len() {
            return Err(ExecuteError::Parse(ParseError::InvalidParameterCount {
                expected: self.operations.len(),
                actual: parameters.len(),
            }));
        }
        let operations = self
            .operations
            .iter()
            .zip(&self.descriptions)
            .zip(parameters)
            .map(|((operation, description), parameters)| {
                if parameters.len() != description.parameter_types.len() {
                    return Err(ExecuteError::Parse(ParseError::InvalidParameterCount {
                        expected: description.parameter_types.len(),
                        actual: parameters.len(),
                    }));
                }
                let parameters = parameters
                    .into_iter()
                    .zip(&description.parameter_types)
                    .map(|(value, expected)| prepared_parameter_value(value, *expected))
                    .collect::<Result<Vec<_>, _>>()?;
                operation
                    .bind(&parameters[..operation.parameter_count()])
                    .map_err(ExecuteError::Parse)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BoundPreparedTransactionRoute {
            operations,
            characteristics: self.characteristics,
            proof: Arc::clone(&self.proof),
        })
    }
}

#[derive(Debug, Clone)]
pub struct BoundPreparedTransactionRoute {
    operations: Vec<gpu_db_sql::ParsedCommand>,
    characteristics: TransactionCharacteristics,
    proof: Arc<PreparedRouteProof>,
}

#[derive(Clone)]
pub(crate) struct PreparedPhysicalPins {
    table_generations: BTreeMap<String, Arc<()>>,
    indexes: BTreeMap<(String, u32, usize), PreparedPinnedDeviceIndex>,
}

#[derive(Clone)]
pub(crate) struct PreparedPinnedDeviceIndex {
    pub(crate) resident_device_ptr: u64,
    /// Pins the exact source shard allocation. Numeric CUDA addresses are not ABA-safe after a
    /// cache purge; coverage and use therefore require Arc identity with the current descriptor.
    pub(crate) resident_guard: Arc<CudaResidentDeviceMemory>,
    pub(crate) published_row_count: Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) gc_boundary: Index,
    pub(crate) device_index: Arc<CudaResidentDeviceMemory>,
    pub(crate) table_mask: u32,
    pub(crate) hash_shift: u32,
    pub(crate) published_has_postings: Arc<std::sync::atomic::AtomicBool>,
}

pub(crate) type PreparedPinRegistry = Arc<Mutex<Vec<PreparedPhysicalPins>>>;

impl Engine {
    /// Compile typed templates into an engine-owned route proof. Index publication is device work
    /// over the authoritative generation; failure leaves no transaction/WAL claim.
    pub fn prepare_transaction_route(
        &self,
        operations: Vec<PreparedCommand>,
        parameter_type_hints: Vec<Vec<Option<SqlType>>>,
        characteristics: TransactionCharacteristics,
    ) -> Result<PreparedTransactionRoute, ExecuteError> {
        let characteristics = validate_transaction_characteristics(characteristics)?;
        if operations.is_empty() {
            return Err(ExecuteError::Unsupported(
                "prepared transaction route requires at least one operation".to_string(),
            ));
        }
        if parameter_type_hints.len() != operations.len() {
            return Err(ExecuteError::Unsupported(
                "prepared transaction route requires one parameter-hint vector per operation"
                    .to_string(),
            ));
        }
        let commit = self
            .commit_state_after_wave_quiescence()
            .map_err(ExecuteError::Engine)?;
        let prepared_snapshot = self.capture_statement_snapshot(self.committed_seq());
        drop(commit);
        let catalog = Arc::clone(&prepared_snapshot.catalog);
        let descriptions = operations
            .iter()
            .zip(&parameter_type_hints)
            .map(|(operation, hints)| {
                self.describe_prepared_command_in_catalog(operation, hints, &catalog)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shape = operations
            .iter()
            .map(|operation| {
                operation
                    .bind(&vec![SqlValue::Null; operation.parameter_count()])
                    .map_err(ExecuteError::Parse)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let access = predeclared_access_plan(&catalog, &shape)?;
        let mut tables = BTreeMap::new();
        for name in &access.touched_tables {
            let table = catalog
                .relational_catalog
                .get(name)
                .cloned()
                .ok_or_else(|| ExecuteError::UndefinedRelation(name.clone()))?;
            self.ensure_prepared_table_indexes(&table)?;
            tables.insert(
                name.clone(),
                PreparedTableRequirement {
                    table,
                    probe_key_ids: BTreeSet::new(),
                },
            );
        }
        let indexed_execution_eligible =
            populate_prepared_probe_requirements(&catalog, &shape, &mut tables)?;
        let proof = Arc::new(PreparedRouteProof {
            engine_identity: Arc::downgrade(&self.read_state),
            tables,
            indexed_execution_eligible,
        });
        // Do the same non-forgeable physical check submission performs. This returns generation and
        // allocation pins, proving that route creation itself cannot be a syntax-only promotion.
        let _pins = self.validate_prepared_route_physical(&proof)?;
        Ok(PreparedTransactionRoute {
            operations,
            descriptions,
            characteristics,
            proof,
        })
    }

    pub(crate) fn submit_bound_prepared_transaction(
        &self,
        txn_id: TxnId,
        transaction: BoundPreparedTransactionRoute,
    ) -> Result<PredeclaredTransactionResult, ExecuteError> {
        self.submit_bound_prepared_transaction_with_hooks(txn_id, transaction, || {}, |_| {})
    }

    #[cfg(test)]
    pub(crate) fn submit_bound_prepared_transaction_instrumented(
        &self,
        txn_id: TxnId,
        transaction: BoundPreparedTransactionRoute,
        on_operation_staged: impl FnMut(usize),
    ) -> Result<PredeclaredTransactionResult, ExecuteError> {
        self.submit_bound_prepared_transaction_with_hooks(
            txn_id,
            transaction,
            || {},
            on_operation_staged,
        )
    }

    #[cfg(test)]
    pub(crate) fn submit_bound_prepared_transaction_with_admission_hook(
        &self,
        txn_id: TxnId,
        transaction: BoundPreparedTransactionRoute,
        on_resources_admitted: impl FnOnce(),
    ) -> Result<PredeclaredTransactionResult, ExecuteError> {
        self.submit_bound_prepared_transaction_with_hooks(
            txn_id,
            transaction,
            on_resources_admitted,
            |_| {},
        )
    }

    fn submit_bound_prepared_transaction_with_hooks(
        &self,
        txn_id: TxnId,
        transaction: BoundPreparedTransactionRoute,
        on_resources_admitted: impl FnOnce(),
        on_operation_staged: impl FnMut(usize),
    ) -> Result<PredeclaredTransactionResult, ExecuteError> {
        let BoundPreparedTransactionRoute {
            operations,
            characteristics,
            proof,
        } = transaction;
        let initial_pins = self.validate_prepared_route_physical(&proof)?;
        let pin_registry = Arc::new(Mutex::new(vec![initial_pins]));
        let access = predeclared_access_plan(&self.catalog_snapshot(), &operations)?;
        let proof_tables = proof.tables.keys().cloned().collect::<BTreeSet<_>>();
        if access.touched_tables != proof_tables {
            return Err(prepared_route_error(
                "prepared transaction catalog dependency closure changed; re-prepare required",
            ));
        }
        let PreparedResourceEstimate {
            resources,
            fast_eligible,
        } = self.estimate_bound_prepared_resources(&operations, &proof)?;
        let class = TransactionClass::derive(resources, fast_eligible);
        let latency_class = class != TransactionClass::General;
        let indexed_route = fast_eligible;
        let declared = if latency_class {
            resources
        } else {
            TransactionResources {
                operations: resources.operations,
                mutations: resources.mutations,
                post_image_and_wal_bytes: u64::MAX,
                maintained_index_fanout: u32::MAX,
                touched_tables: resources.touched_tables,
                cold_accesses: 0,
                result_bytes: u64::MAX,
            }
        };
        // This permit is the class value's first operational consumer: W1/T8/T32 occupy separate
        // FIFO foreground lanes with distinct population/byte credits and queue deadlines. It is
        // acquired before BEGIN and held through publication-covered completion. General work is
        // deliberately supported outside these latency envelopes and cannot consume their slots.
        let _service = self.prepared_transaction_service.admit(class, resources)?;
        let _required =
            indexed_route.then(|| PreparedIndexRequirementGuard::enter(Arc::clone(&pin_registry)));
        on_resources_admitted();
        self.submit_predeclared_transaction_with_hooks(
            txn_id,
            PredeclaredTransaction::new(operations, declared, characteristics),
            PredeclaredSubmissionContext {
                engine_prepared_route: true,
                engine_prepared_fast_route: indexed_route,
                prepared_proof: Some(proof),
                prepared_pin_registry: Some(pin_registry),
                on_transaction_registered: || {},
                on_operation_staged,
            },
        )
    }

    pub fn prepared_transaction_class_admissions(&self) -> PreparedTransactionClassAdmissions {
        PreparedTransactionClassAdmissions {
            w1: self.prepared_transaction_class_admissions[0].load(AtomicOrdering::Relaxed),
            t8: self.prepared_transaction_class_admissions[1].load(AtomicOrdering::Relaxed),
            t32: self.prepared_transaction_class_admissions[2].load(AtomicOrdering::Relaxed),
            general: self.prepared_transaction_class_admissions[3].load(AtomicOrdering::Relaxed),
        }
    }

    pub(crate) fn record_prepared_transaction_admission(&self, class: TransactionClass) {
        let index = match class {
            TransactionClass::W1 => 0,
            TransactionClass::T8 => 1,
            TransactionClass::T32 => 2,
            TransactionClass::General => 3,
        };
        self.prepared_transaction_class_admissions[index].fetch_add(1, AtomicOrdering::Relaxed);
    }

    fn ensure_prepared_table_indexes(&self, table: &RelationalTable) -> Result<(), ExecuteError> {
        let shards = self.read_residency_shards();
        let resident_rows = shards
            .get(&table.name)
            .map(|shards| shards.iter().map(|shard| shard.row_count).sum::<usize>())
            .unwrap_or(0);
        drop(shards);
        if resident_rows > 0 && !table.indexes.is_empty() {
            self.publish_relational_resident_indexes(&table.name)?;
        }
        Ok(())
    }

    fn validate_prepared_route_physical(
        &self,
        proof: &PreparedRouteProof,
    ) -> Result<PreparedPhysicalPins, ExecuteError> {
        let commit = self
            .commit_state_after_wave_quiescence()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self.capture_statement_snapshot(self.committed_seq());
        drop(commit);
        self.validate_prepared_route_snapshot(proof, &snapshot)
    }

    pub(crate) fn validate_prepared_route_snapshot(
        &self,
        proof: &PreparedRouteProof,
        snapshot: &TransactionSnapshot,
    ) -> Result<PreparedPhysicalPins, ExecuteError> {
        if !proof
            .engine_identity
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, &self.read_state))
        {
            return Err(prepared_route_error(
                "prepared transaction route belongs to another engine instance",
            ));
        }
        // Match named-index publication/retirement's lock order. Besides avoiding ABBA, this makes
        // the catalog/generation/cache/coverage observations below one physical proof instant.
        let _route_publish = self
            .read_state
            .residency
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let catalog = &snapshot.catalog;
        let shards = &snapshot.resident_shards;
        let cache = self
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let coverage = self
            .read_state
            .residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let complete = self
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut table_generations = BTreeMap::new();
        let mut indexes = BTreeMap::new();
        for (name, requirement) in &proof.tables {
            if catalog.relational_catalog.get(name) != Some(&requirement.table) {
                return Err(prepared_route_error(format!(
                    "prepared transaction catalog dependency \"{name}\" changed; re-prepare required"
                )));
            }
            if snapshot.chunk_authoritative_tables.contains_key(name) {
                return Err(prepared_route_error(format!(
                    "prepared transaction relation \"{name}\" has cold accesses"
                )));
            }
            let table_shards = shards.get(name).ok_or_else(|| {
                prepared_route_error(format!(
                    "prepared transaction relation \"{name}\" has no resident shard generation"
                ))
            })?;
            let generation = table_shards
                .first()
                .map(|shard| Arc::clone(&shard.point_route_generation))
                .ok_or_else(|| {
                    prepared_route_error(format!(
                        "prepared transaction relation \"{name}\" has an empty shard set"
                    ))
                })?;
            if table_shards.iter().any(|shard| {
                !Arc::ptr_eq(&generation, &shard.point_route_generation)
                    || shard.device_memory.is_none()
                    || shard.device_memory_proof.is_none()
            }) {
                return Err(prepared_route_error(format!(
                    "prepared transaction relation \"{name}\" has a torn or non-device shard generation"
                )));
            }
            let resident_rows = table_shards
                .iter()
                .map(|shard| shard.row_count)
                .sum::<usize>();
            if resident_rows > 0
                && !requirement.table.indexes.is_empty()
                && complete.get(name)
                    != Some(&(requirement.table.oid, requirement.table.indexes.clone()))
            {
                return Err(prepared_route_error(format!(
                    "prepared transaction relation \"{name}\" lacks complete named-index coverage"
                )));
            }
            table_generations.insert(name.clone(), generation);
            let mut published_key_ids = BTreeSet::new();
            for (ordinal, index) in requirement.table.indexes.iter().enumerate() {
                let key_id =
                    crate::engine_residency::index_probe_key_id(&requirement.table, index, ordinal)
                        .ok_or_else(|| {
                            prepared_route_error(format!(
                        "prepared transaction index on relation \"{name}\" has no device key id"
                    ))
                        })?;
                published_key_ids.insert(key_id);
                for shard in table_shards.iter().filter(|shard| shard.row_count != 0) {
                    let resident_memory = shard.device_memory.as_ref().expect("device proof above");
                    let resident_ptr = resident_memory.device_ptr();
                    if !coverage
                        .get(&(name.clone(), shard.shard_id, key_id))
                        .is_some_and(|(covered_ptr, covered_rows)| {
                            *covered_ptr == resident_ptr && *covered_rows >= shard.row_count
                        })
                    {
                        return Err(prepared_route_error(format!(
                            "prepared transaction relation \"{name}\" lacks exact index coverage for shard {}",
                            shard.shard_id
                        )));
                    }
                    let allocation = cache
                        .get(&(name.clone(), shard.shard_id, key_id))
                        .filter(|entry| {
                            entry.resident_device_ptr == resident_ptr
                                && Arc::ptr_eq(&entry._resident_guard, resident_memory)
                                && entry.row_count >= shard.row_count
                                && entry.gc_boundary <= snapshot.boundary
                        })
                        .and_then(|entry| {
                            entry.device_index.as_ref().map(|allocation| {
                                PreparedPinnedDeviceIndex {
                                    resident_device_ptr: entry.resident_device_ptr,
                                    resident_guard: Arc::clone(&entry._resident_guard),
                                    published_row_count: Arc::clone(&entry.published_row_count),
                                    gc_boundary: entry.gc_boundary,
                                    device_index: Arc::clone(allocation),
                                    table_mask: entry.table_mask,
                                    hash_shift: entry.hash_shift,
                                    published_has_postings: Arc::clone(
                                        &entry.published_has_postings,
                                    ),
                                }
                            })
                        })
                        .ok_or_else(|| {
                            prepared_route_error(format!(
                                "prepared transaction relation \"{name}\" lacks an exact device index allocation for shard {}",
                                shard.shard_id
                            ))
                        })?;
                    indexes.insert((name.clone(), shard.shard_id, key_id), allocation);
                }
            }
            if !requirement.probe_key_ids.is_subset(&published_key_ids) {
                return Err(prepared_route_error(format!(
                    "prepared transaction relation \"{name}\" lacks an exact named index for every declared probe"
                )));
            }
        }
        Ok(PreparedPhysicalPins {
            table_generations,
            indexes,
        })
    }

    pub(crate) fn prepared_pins_cover_snapshot(
        &self,
        proof: &PreparedRouteProof,
        snapshot: &TransactionSnapshot,
        pins: &PreparedPhysicalPins,
    ) -> bool {
        if !proof
            .engine_identity
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, &self.read_state))
        {
            return false;
        }
        proof.tables.iter().all(|(name, requirement)| {
            if snapshot.catalog.relational_catalog.get(name) != Some(&requirement.table)
                || snapshot.chunk_authoritative_tables.contains_key(name)
            {
                return false;
            }
            let Some(shards) = snapshot.resident_shards.get(name) else {
                return false;
            };
            if !pins.table_generations.contains_key(name) {
                return false;
            }
            // Descriptor publication rotates the mutable-cache route token on every shard-map
            // update, including an in-place append whose device allocation is unchanged. A retained
            // prepared pin does not rely on that cache token: the exact current device pointer,
            // retained index allocation, published extent, and GC boundary are revalidated below.
            // A real replacement or rollover still fails because its pointer/shard has no matching
            // retained allocation.
            if shards
                .iter()
                .any(|shard| shard.device_memory.is_none() || shard.device_memory_proof.is_none())
            {
                return false;
            }
            requirement.probe_key_ids.iter().all(|key_id| {
                shards
                    .iter()
                    .filter(|shard| shard.row_count != 0)
                    .all(|shard| {
                        let Some(memory) = shard.device_memory.as_ref() else {
                            return false;
                        };
                        pins.indexes
                            .get(&(name.clone(), shard.shard_id, *key_id))
                            .is_some_and(|index| {
                                index.resident_device_ptr == memory.device_ptr()
                                    && Arc::ptr_eq(&index.resident_guard, memory)
                                    && index
                                        .published_row_count
                                        .load(std::sync::atomic::Ordering::Acquire)
                                        >= shard.row_count
                                    && index.gc_boundary <= snapshot.boundary
                            })
                    })
            })
        })
    }

    pub(crate) fn prepared_index_source_is_private_overlay(
        &self,
        table: &str,
        shard_id: u32,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        row_count: usize,
    ) -> bool {
        let Some(snapshot) = self.current_transaction_read_snapshot() else {
            return false;
        };
        let Some(base) = snapshot
            .resident_shards
            .get(table)
            .and_then(|shards| shards.iter().find(|shard| shard.shard_id == shard_id))
        else {
            return true;
        };
        base.device_memory.as_ref().is_none_or(|base_memory| {
            !Arc::ptr_eq(base_memory, device_memory) || row_count > base.row_count
        })
    }

    pub(crate) fn execute_prepared_index_select_in_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        snapshot: &Arc<TransactionSnapshot>,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        self.acquire_transaction_table_access(snapshot, [select.table.clone()])?;
        if select.distinct
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || !select.order_by.is_empty()
            || select.limit.is_some()
            || select.offset.is_some()
            || !matches!(
                select.projection,
                SelectProjection::All | SelectProjection::Columns(_)
            )
        {
            return Err(prepared_route_error(
                "prepared indexed SELECT requires a plain column projection without order/limit",
            ));
        }
        let table = snapshot
            .catalog
            .relational_catalog
            .get(&select.table)
            .cloned()
            .ok_or_else(|| ExecuteError::UndefinedRelation(select.table.clone()))?;
        if snapshot.table_has_typed_empty_root(&select.table) {
            let _scope = self.enter_transaction_read(Arc::clone(snapshot));
            return self.execute_transient_rows_via_general(
                select,
                table,
                Vec::new(),
                snapshot.boundary,
            );
        }
        let bound = bind_relational_select(&table, select)?;
        let groups = normalized_select_groups(&bound);
        let [group] = groups.as_slice() else {
            return Err(prepared_route_error(
                "prepared indexed SELECT requires one predicate group",
            ));
        };
        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
        let hits = self
            .prepared_exact_index_hits(
                &table,
                group,
                StorageVisibility {
                    read_txn_id: snapshot.boundary,
                },
            )
            .map_err(ExecuteError::Engine)?;
        let gpu_id = hits
            .first()
            .map_or(self.planner.default_gpu_id(), |hit| hit.descriptor.gpu_id);
        let mut values = Vec::new();
        let mut matched_rows = 0_usize;
        for hit in hits {
            let row = match self.materialize_resident_row_via_hit(&table, &hit, snapshot.boundary) {
                Some(Some(row)) => row,
                Some(None) => continue,
                None => {
                    return Err(prepared_route_error(
                        "prepared indexed SELECT could not materialize its device hit",
                    ));
                }
            };
            matched_rows = matched_rows.saturating_add(1);
            values.extend(
                bound
                    .selected_indexes
                    .iter()
                    .map(|column| row[*column].clone()),
            );
        }
        let ncols = bound.selected_columns.len();
        if matched_rows > 1 {
            return Err(prepared_route_error(
                "prepared unique-key SELECT returned more than one row",
            ));
        }
        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns),
            rows: RowBlock::flat(values, ncols),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::EqualityIndex {
                table: table.name,
                column: group
                    .iter()
                    .filter_map(|(column, _, _)| table.columns.get(*column))
                    .map(|column| column.name.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
                matched_keys: matched_rows,
            }),
        })
    }
}

fn normalized_select_groups(
    bound: &BoundRelationalSelect,
) -> Vec<Vec<(usize, SelectFilterOp, SqlValue)>> {
    if !bound.filter_groups.is_empty() {
        bound.filter_groups.clone()
    } else if !bound.filters.is_empty() {
        vec![bound.filters.clone()]
    } else {
        bound
            .filter
            .clone()
            .map_or_else(Vec::new, |filter| vec![vec![filter]])
    }
}

fn populate_prepared_probe_requirements(
    catalog: &CatalogSnapshot,
    operations: &[gpu_db_sql::ParsedCommand],
    requirements: &mut BTreeMap<String, PreparedTableRequirement>,
) -> Result<bool, ExecuteError> {
    let mut eligible = true;
    for operation in operations {
        match operation.command() {
            Command::Select(select) => {
                let table = catalog
                    .relational_catalog
                    .get(&select.table)
                    .ok_or_else(|| ExecuteError::UndefinedRelation(select.table.clone()))?;
                let bound = bind_relational_select(table, select)?;
                let groups = normalized_select_groups(&bound);
                eligible &= add_unique_group_probe(table, &groups, requirements);
            }
            Command::Insert(insert) => {
                let table = catalog
                    .relational_catalog
                    .get(&insert.table)
                    .ok_or_else(|| ExecuteError::UndefinedRelation(insert.table.clone()))?;
                eligible &= add_mutation_probe_closure(catalog, table, requirements);
            }
            Command::Update(update) => {
                let table = catalog
                    .relational_catalog
                    .get(&update.table)
                    .ok_or_else(|| ExecuteError::UndefinedRelation(update.table.clone()))?;
                let groups = bind_delete_filter_groups(
                    table,
                    &Delete {
                        table: update.table.clone(),
                        filter: update.filter.clone(),
                        filters: update.filters.clone(),
                        filter_groups: update.filter_groups.clone(),
                        returning: Vec::new(),
                    },
                )?;
                eligible &= add_unique_group_probe(table, &groups, requirements);
                eligible &= add_mutation_probe_closure(catalog, table, requirements);
            }
            Command::Delete(delete) => {
                let table = catalog
                    .relational_catalog
                    .get(&delete.table)
                    .ok_or_else(|| ExecuteError::UndefinedRelation(delete.table.clone()))?;
                let groups = bind_delete_filter_groups(table, delete)?;
                eligible &= add_unique_group_probe(table, &groups, requirements);
                eligible &= add_mutation_probe_closure(catalog, table, requirements);
            }
            _ => eligible = false,
        }
    }
    Ok(eligible)
}

fn add_unique_group_probe(
    table: &RelationalTable,
    groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
    requirements: &mut BTreeMap<String, PreparedTableRequirement>,
) -> bool {
    let [group] = groups else { return false };
    let Some((key_id, _)) = prepared_unique_probe_index(table, group) else {
        return false;
    };
    requirements
        .get_mut(&table.name)
        .expect("access closure contains operation target")
        .probe_key_ids
        .insert(key_id);
    true
}

fn add_mutation_probe_closure(
    catalog: &CatalogSnapshot,
    table: &RelationalTable,
    requirements: &mut BTreeMap<String, PreparedTableRequirement>,
) -> bool {
    let mut complete = true;
    for (ordinal, index) in table
        .indexes
        .iter()
        .enumerate()
        .filter(|(_, index)| index.unique)
    {
        complete &= add_exact_index_requirement(table, ordinal, index, requirements);
    }
    for foreign_key in &table.foreign_keys {
        let Some(parent) = catalog
            .relational_catalog
            .get(&foreign_key.referenced_table)
        else {
            complete = false;
            continue;
        };
        complete &= add_single_column_unique_requirement(
            parent,
            &foreign_key.referenced_column,
            requirements,
        );
    }
    for child in catalog.relational_catalog.values() {
        for foreign_key in child
            .foreign_keys
            .iter()
            .filter(|foreign_key| foreign_key.referenced_table == table.name)
        {
            complete &= add_single_column_unique_requirement(
                table,
                &foreign_key.referenced_column,
                requirements,
            );
            // A non-unique child index can have an unbounded posting list, while this route's
            // existence probe has a fixed candidate budget. Such PostgreSQL-valid schemas remain
            // supported by General execution but cannot borrow a bounded prepared class.
            complete &=
                add_single_column_unique_requirement(child, &foreign_key.column, requirements);
        }
    }
    complete
}

fn add_single_column_unique_requirement(
    table: &RelationalTable,
    column: &str,
    requirements: &mut BTreeMap<String, PreparedTableRequirement>,
) -> bool {
    let Some(position) = table
        .columns
        .iter()
        .position(|candidate| candidate.name == column)
    else {
        return false;
    };
    let Some((ordinal, index)) = table.indexes.iter().enumerate().find(|(_, index)| {
        index.unique
            && crate::engine_residency::index_key_column_positions(table, index).as_deref()
                == Some(std::slice::from_ref(&position))
    }) else {
        return false;
    };
    add_exact_index_requirement(table, ordinal, index, requirements)
}

fn add_exact_index_requirement(
    table: &RelationalTable,
    ordinal: usize,
    index: &RelationalIndex,
    requirements: &mut BTreeMap<String, PreparedTableRequirement>,
) -> bool {
    let Some(key_id) = crate::engine_residency::index_probe_key_id(table, index, ordinal) else {
        return false;
    };
    let Some(requirement) = requirements.get_mut(&table.name) else {
        return false;
    };
    requirement.probe_key_ids.insert(key_id);
    true
}

fn prepared_unique_probe_index<'a>(
    table: &'a RelationalTable,
    group: &[(usize, SelectFilterOp, SqlValue)],
) -> Option<(usize, &'a RelationalIndex)> {
    table
        .indexes
        .iter()
        .enumerate()
        .filter(|(_, index)| index.unique)
        .find_map(|(ordinal, index)| {
            let positions = crate::engine_residency::index_key_column_positions(table, index)?;
            positions
                .iter()
                .all(|position| {
                    group
                        .iter()
                        .any(|(column, op, _)| column == position && *op == SelectFilterOp::Eq)
                })
                .then(|| {
                    crate::engine_residency::index_probe_key_id(table, index, ordinal)
                        .map(|key_id| (key_id, index))
                })?
        })
}

pub(crate) enum PreparedIndexProbe {
    Empty,
    Key { key_id: usize, needle: i32 },
}

pub(crate) fn prepared_index_probe(
    table: &RelationalTable,
    group: &[(usize, SelectFilterOp, SqlValue)],
) -> Option<PreparedIndexProbe> {
    let (key_id, index) = prepared_unique_probe_index(table, group)?;
    let positions = crate::engine_residency::index_key_column_positions(table, index)?;
    let values = positions
        .iter()
        .map(|position| {
            group
                .iter()
                .find(|(column, op, _)| column == position && *op == SelectFilterOp::Eq)
                .map(|(_, _, value)| value)
        })
        .collect::<Option<Vec<_>>>()?;
    if values.iter().any(|value| matches!(value, SqlValue::Null)) {
        return Some(PreparedIndexProbe::Empty);
    }
    if crate::engine_residency::index_uses_fingerprint(table, index) {
        let mut words = Vec::new();
        for (&position, value) in positions.iter().zip(values) {
            let column = table.columns.get(position)?;
            let value = crate::rel_exec_helpers::coerce_insert_value(
                value.clone(),
                column.ty,
                &column.name,
            )
            .ok()?;
            words.extend(crate::engine_residency::sql_value_key_words(
                column.ty, &value,
            )?);
        }
        Some(PreparedIndexProbe::Key {
            key_id,
            needle: crate::engine_residency::compound_key_fingerprint(&words),
        })
    } else {
        let [position] = positions.as_slice() else {
            return None;
        };
        Some(PreparedIndexProbe::Key {
            key_id,
            needle: crate::engine_residency::i32_section_needle(
                table.columns.get(*position)?.ty,
                values[0],
            )?,
        })
    }
}

fn prepared_route_error(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Unsupported(format!(
        "engine-prepared transaction route unavailable: {}",
        message.into()
    ))
}

fn prepared_parameter_value(value: SqlValue, expected: SqlType) -> Result<SqlValue, ExecuteError> {
    if matches!(value, SqlValue::Null) {
        return Ok(value);
    }
    let compatible = matches!(
        (expected, &value),
        (SqlType::Int2, SqlValue::Int2(_))
            | (SqlType::Int4, SqlValue::Int4(_))
            | (SqlType::Int8, SqlValue::Int8(_))
            | (SqlType::Numeric { .. }, SqlValue::Numeric(_))
            | (SqlType::Bool, SqlValue::Bool(_))
            | (SqlType::Text, SqlValue::Text(_))
            | (SqlType::Date, SqlValue::Date(_))
            | (SqlType::Timestamp, SqlValue::Timestamp(_))
            | (SqlType::Uuid, SqlValue::Uuid(_))
    );
    if compatible {
        crate::rel_exec_helpers::validate_datetime_carrier(&value, expected)?;
        Ok(value)
    } else {
        let actual = match value {
            SqlValue::Null => "null",
            SqlValue::Int2(_) => "smallint",
            SqlValue::Int4(_) => "integer",
            SqlValue::Int8(_) => "bigint",
            SqlValue::Numeric(_) => "numeric",
            SqlValue::Bool(_) => "boolean",
            SqlValue::Text(_) => "text",
            SqlValue::Date(_) => "date",
            SqlValue::Timestamp(_) => "timestamp without time zone",
            SqlValue::Uuid(_) => "uuid",
            SqlValue::Parameter { .. } => "unbound parameter",
        };
        Err(ExecuteError::DatatypeMismatch(format!(
            "prepared transaction parameter requires {}, got {}",
            expected.catalog_name(),
            actual
        )))
    }
}

thread_local! {
    static PREPARED_INDEX_REQUIRED: std::cell::RefCell<Vec<PreparedPinRegistry>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

#[cfg(test)]
pub(crate) static PREPARED_PRIVATE_INDEX_REBUILD_ROWS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
pub(crate) static PREPARED_PRIVATE_INDEX_REBUILD_MAX_ROWS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
pub(crate) static PREPARED_PINNED_INDEX_HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
pub(crate) static PREPARED_BASE_INDEX_MISSES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn prepared_index_route_required() -> bool {
    PREPARED_INDEX_REQUIRED.with(|registries| !registries.borrow().is_empty())
}

pub(crate) fn prepared_pinned_device_index(
    table: &str,
    shard_id: u32,
    key_id: usize,
    resident_device: &Arc<CudaResidentDeviceMemory>,
    row_count: usize,
    gc_boundary: Index,
) -> Option<PreparedPinnedDeviceIndex> {
    let result = PREPARED_INDEX_REQUIRED.with(|registries| {
        registries.borrow().iter().rev().find_map(|registry| {
            registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .rev()
                .find_map(|pins| {
                    pins.indexes
                        .get(&(table.to_string(), shard_id, key_id))
                        .filter(|index| {
                            index.resident_device_ptr == resident_device.device_ptr()
                                && Arc::ptr_eq(&index.resident_guard, resident_device)
                                && index
                                    .published_row_count
                                    .load(std::sync::atomic::Ordering::Acquire)
                                    >= row_count
                                && index.gc_boundary <= gc_boundary
                        })
                        .cloned()
                })
        })
    });
    result
}

struct PreparedIndexRequirementGuard;

impl PreparedIndexRequirementGuard {
    fn enter(registry: PreparedPinRegistry) -> Self {
        PREPARED_INDEX_REQUIRED.with(|registries| registries.borrow_mut().push(registry));
        Self
    }
}

impl Drop for PreparedIndexRequirementGuard {
    fn drop(&mut self) {
        PREPARED_INDEX_REQUIRED.with(|registries| {
            registries
                .borrow_mut()
                .pop()
                .expect("prepared index requirement guards remain paired");
        });
    }
}

#[cfg(test)]
mod tests;
