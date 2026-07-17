//! Explicit-transaction private GPU generation: stage prepared DML into immutable shard overlays
//! without mutating the globally published relation. The transaction snapshot remains the owner;
//! SELECT and subsequent DML load the latest private generation through the ordinary residency
//! accessors.

use super::*;

mod gpu_accounting;
use gpu_accounting::{transaction_private_shard_bytes, TransactionGpuReservation};

type TargetRow<'a> = (u64, &'a [SqlValue]);

#[cfg(test)]
static FAIL_NEXT_TRANSACTION_POST_DURABLE_APPLY: AtomicBool = AtomicBool::new(false);

impl Engine {
    #[cfg(test)]
    pub(crate) fn fail_next_transaction_post_durable_apply(&self) {
        FAIL_NEXT_TRANSACTION_POST_DURABLE_APPLY.store(true, AtomicOrdering::Release);
    }

    /// Drive COMMIT and an optional successor as one engine-owned transaction-control operation.
    /// Returning the successor identity lets protocol façades adopt the transaction that was
    /// registered under the same commit/active-snapshot locks, preserving `AND CHAIN` atomicity.
    pub fn commit_explicit_transaction(
        &self,
        txn_id: TxnId,
        chain: bool,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if let Some(snapshot) = self.transaction_snapshot_handle(txn_id) {
            let _statement = snapshot
                .statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_commit_path_available()
                .map_err(ExecuteError::Engine)?;
            if !snapshot.transaction_delta_is_empty() {
                self.commit_transaction_delta(txn_id, chain, current_timestamp_micros())
            } else {
                self.finish_transaction_context(txn_id, true, chain)
                    .map_err(ExecuteError::Txn)
            }
        } else {
            self.finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    pub fn rollback_explicit_transaction(
        &self,
        txn_id: TxnId,
        chain: bool,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if let Some(snapshot) = self.transaction_snapshot_handle(txn_id) {
            let _statement = snapshot
                .statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_commit_path_available()
                .map_err(ExecuteError::Engine)?;
            self.finish_transaction_context(txn_id, false, chain)
                .map_err(ExecuteError::Txn)
        } else {
            self.finish_transaction_context(txn_id, false, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    /// A transaction-generation prepare may use only the resources captured at `BEGIN`. The
    /// autocommit compatibility ladders are allowed to rehydrate/de-authoritize and then rebind to
    /// current state; doing that while a retained scope is active would silently destroy snapshot
    /// isolation. Device/source declines therefore fail before any current-generation mutation.
    pub(crate) fn guard_transaction_dml_rebind(
        &self,
        table: &str,
        declined_source: &str,
    ) -> Result<(), EngineError> {
        if self.current_transaction_read_snapshot().is_some() {
            return Err(EngineError::ApplyFailed(format!(
                "transaction-generation DML validation for \"{table}\" declined its retained \
                 {declined_source}; refusing to rebind to current state"
            )));
        }
        Ok(())
    }

    /// Stage one DML statement in an explicit transaction. Preparation and every constraint/read
    /// bind to the retained generation plus prior private deltas. A successful statement publishes
    /// a new transaction-private shard map atomically; global WAL, MVCC, residency, and committed
    /// visibility remain untouched until COMMIT.
    pub fn execute_dml_in_transaction(
        &self,
        txn_id: TxnId,
        text: &str,
    ) -> Result<(), ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.intent_lanes_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        let _statement = snapshot
            .statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let command = parse_command(text)?;
        if !matches!(
            command,
            Command::Insert(_) | Command::Update(_) | Command::Delete(_)
        ) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "explicit transaction staging accepts INSERT, UPDATE, or DELETE".to_string(),
            )));
        }

        let table_name = match &command {
            Command::Insert(insert) => insert.table.as_str(),
            Command::Update(update) => update.table.as_str(),
            Command::Delete(delete) => delete.table.as_str(),
            _ => unreachable!("DML shape checked above"),
        };
        let table = snapshot
            .catalog
            .relational_catalog
            .get(table_name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table_name}\" does not exist in the transaction generation"
                )))
            })?;
        let _scope = self.enter_transaction_read(Arc::clone(&snapshot));

        let (generation, next_row_id) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (delta.generation, delta.next_row_id)
        };
        let prepared = self.prepare_dml(
            &command,
            DmlReadSnapshot {
                commit_seq: snapshot.boundary,
                next_row_id,
            },
            InsertPrepareValidation::Full,
        )?;
        let prepared_sequence_state = match &prepared.mutation {
            PreparedMutation::Insert { seq_advances, .. } => seq_advances.clone(),
            PreparedMutation::Update { .. } | PreparedMutation::Delete { .. } => BTreeMap::new(),
        };

        self.validate_transaction_delta_residency(&table)?;

        let current_shards = snapshot.transaction_shards();
        let current_cold_chunks = snapshot.transaction_cold_chunks();
        let mut next_shards = (*current_shards).clone();
        let mut next_cold_chunks = (*current_cold_chunks).clone();
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        self.apply_transaction_private_delta(
            &table,
            &prepared,
            snapshot.boundary,
            &mut next_shards,
            &mut next_cold_chunks,
            &mut gpu_reservation,
        )?;
        let next_private_gpu_bytes =
            transaction_private_shard_bytes(snapshot.resident_shards.as_ref(), &next_shards);

        // Compare-and-publish under the transaction mutex. Connection/session execution is ordered,
        // but the generation check also makes accidental same-transaction concurrent use fail loud
        // instead of losing one private statement.
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta.generation != generation {
            return Err(ExecuteError::Serialization(
                "concurrent statements attempted to publish the same transaction delta".to_string(),
            ));
        }
        gpu_reservation.ensure_replacement_admitted(
            &delta.private_gpu_bytes_by_gpu,
            &next_private_gpu_bytes,
        )?;
        delta.write_set.extend_deduplicated(&prepared.write_set);
        delta.next_row_id = delta.next_row_id.saturating_add(prepared.rows_consumed);
        delta.sequence_state.extend(prepared_sequence_state);
        delta.deltas.push(prepared);
        delta.resident_shards = Arc::new(next_shards);
        delta.streaming_cold_chunks = Arc::new(next_cold_chunks);
        delta.generation = delta.generation.saturating_add(1);
        // The transaction statement lock excludes every other same-transaction reader or writer.
        // Release this statement's old-map pin before dropping superseded allocation charges, then
        // atomically convert the temporary reservations into the exact replacement-generation account.
        drop(current_shards);
        gpu_reservation
            .replace_charges(&mut delta.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        Ok(())
    }

    /// Durably publish all staged statements as ONE resolved WAL record and ONE MVCC generation.
    /// The commit lock closes catalog/conflict races, final insert identities are claimed once, and
    /// recovery applies the same ordered row mutations without re-evaluating predicates.
    pub(crate) fn commit_transaction_delta(
        &self,
        txn_id: TxnId,
        chain: bool,
        timestamp_micros: u64,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.intent_lanes_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        // A classic wave installs its device/host generations before its durability tail publishes
        // `committed_seq`. Explicit-transaction row/unique/FK validation must not inspect that
        // applied-but-unpublished interval. Drain first, then prove under the commit lock that no
        // sequencer handed off a new tail in the acquisition gap; retry until the cut is settled.
        let mut commit = loop {
            if !self.wait_wave_tail_quiescence() {
                return Err(ExecuteError::Engine(self.commit_path_unavailable_error()));
            }
            let commit = self.commit_state();
            let applied = self.commit_wave.tails_applied.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            let maintenance_pending = self
                .commit_wave
                .tail_maintenance_pending
                .load(AtomicOrdering::Acquire);
            if applied == finished && maintenance_pending == 0 {
                // The tail that satisfied the barrier may have wedged while this COMMIT waited.
                // Recheck while holding the publication lock, before allocator identity claims,
                // WAL append, catalog mutation, or any transaction-visible state transition.
                if let Err(error) = self.ensure_commit_path_available() {
                    drop(commit);
                    return Err(ExecuteError::Engine(error));
                }
                break commit;
            }
            drop(commit);
            self.ensure_commit_path_available()
                .map_err(ExecuteError::Engine)?;
        };
        // The optimistic guard above can race first lane activation while this transaction waits
        // for wave tails/the commit lock. Activation uses the same lock; reject here before any WAL
        // or catalog/data mutation so serial and lane sequence domains never overlap.
        self.intent_lanes_write_guard()
            .map_err(ExecuteError::Engine)?;
        if commit.txn_manager.state(txn_id) != Some(TxnState::Active) {
            return Err(ExecuteError::Txn(TxnError::NotActive(txn_id)));
        }
        let current_catalog = self.catalog_snapshot();
        let mut current_catalog_contents = (*current_catalog).clone();
        let mut retained_catalog_contents = (*snapshot.catalog).clone();
        current_catalog_contents.commit_seq = 0;
        retained_catalog_contents.commit_seq = 0;
        if current_catalog_contents != retained_catalog_contents {
            return Err(ExecuteError::Serialization(format!(
                "catalog generation changed after transaction BEGIN ({} -> {})",
                snapshot.catalog.commit_seq, current_catalog.commit_seq
            )));
        }

        let deltas = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            delta.deltas.clone()
        };
        if deltas.is_empty() {
            drop(commit);
            return self
                .finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn);
        }
        let staged_tables = deltas
            .iter()
            .map(|delta| match &delta.mutation {
                PreparedMutation::Insert { table, .. }
                | PreparedMutation::Update { table, .. }
                | PreparedMutation::Delete { table, .. } => table,
            })
            .collect::<BTreeSet<_>>();
        if let Some(table) = staged_tables.iter().find(|table| {
            !snapshot
                .chunk_authoritative_tables
                .contains_key(table.as_str())
                && self.table_chunk_authoritative(table).is_some()
        }) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{table}\" became cold-chunk authoritative after transaction staging"
            )));
        }

        // A later statement may update/delete a row inserted earlier in this transaction. Its
        // provisional row key is not a real conflict point: nobody outside this transaction can
        // observe that entity, and COMMIT assigns it a fresh globally-claimed identity.
        let provisional_inserts = Self::transaction_insert_identities(&deltas)?;
        self.validate_transaction_device_row_conflicts(&snapshot, &deltas, &provisional_inserts)?;

        let insert_count = provisional_inserts.len() as u64;
        let final_base = self
            .read_state
            .mvcc
            .claim_row_id_block(insert_count)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction row identity space exhausted".to_string(),
                ))
            })?;
        let allocator_high_water = final_base.checked_add(insert_count).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "transaction row identity space exhausted".to_string(),
            ))
        })?;
        let record = Self::resolved_transaction_record(
            &deltas,
            &provisional_inserts,
            final_base,
            allocator_high_water,
        )?;
        self.validate_transaction_device_unique_conflicts(&snapshot, &record)?;
        self.validate_transaction_device_foreign_key_conflicts(&snapshot, &record)?;
        let payload = try_encode_binary_transaction(&record).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "resolved transaction WAL record exceeds binary framing limits".to_string(),
            ))
        })?;
        let payload: Arc<[u8]> = Arc::from(payload);

        let wal_len_before = commit.wal.len();
        let token = match commit.repl.propose(Arc::clone(&payload)) {
            Ok(token) => token,
            Err(err) => {
                return Err(ExecuteError::Engine(err));
            }
        };
        let record = match Self::canonical_wal_record(&commit, txn_id, token.index, 0, &payload) {
            Ok(record) => record,
            Err(err) => {
                commit.repl.rollback_unapplied_from(token.index);
                return Err(ExecuteError::Engine(err));
            }
        };
        commit.wal.append(record);
        if let Err(err) = commit.wal.flush_all() {
            commit.repl.rollback_unapplied_from(token.index);
            commit.wal.truncate(wal_len_before);
            return Err(ExecuteError::Engine(err));
        }
        commit
            .repl
            .wait_committed(token, Duration::from_millis(0))
            .map_err(ExecuteError::Engine)?;
        commit.record_transaction_status(txn_id, &payload, token.index);
        commit.record_commit_timestamp(txn_id, timestamp_micros);
        #[cfg(test)]
        let apply_result =
            if FAIL_NEXT_TRANSACTION_POST_DURABLE_APPLY.swap(false, AtomicOrdering::AcqRel) {
                Err(EngineError::ApplyFailed(
                    "injected post-durable explicit-transaction apply failure".to_string(),
                ))
            } else {
                self.apply_and_publish_committed(&mut commit, txn_id, token.index)
            };
        #[cfg(not(test))]
        let apply_result = self.apply_and_publish_committed(&mut commit, txn_id, token.index);
        if let Err(err) = apply_result {
            self.wedge_commit_path();
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "explicit transaction {txn_id} is durable but could not be fully installed: {err}; engine restart recovery required"
            ))));
        }

        commit
            .txn_manager
            .commit(txn_id)
            .expect("active transaction remained active under the commit lock");
        let successor = if chain {
            let next = commit.txn_manager.begin()?;
            Some((
                next.id,
                self.capture_transaction_snapshot(self.committed_seq()),
            ))
        } else {
            None
        };
        let mut active = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = active.deregister_transaction(txn_id);
        debug_assert!(removed.is_some(), "committed transaction lost its snapshot");
        let successor_id = successor.as_ref().map(|(next_id, _)| *next_id);
        if let Some((next_id, next_snapshot)) = successor {
            active.register_transaction(next_id, next_snapshot);
        }
        drop(active);
        self.gc_transaction_created_by_regions();
        self.metrics.inc_commit();
        Ok(successor_id)
    }

    fn transaction_insert_identities(
        deltas: &[WriteDelta],
    ) -> Result<BTreeSet<(String, u64)>, ExecuteError> {
        let mut identities = BTreeSet::new();
        for delta in deltas {
            let PreparedMutation::Insert {
                table,
                inserted_rows,
                ..
            } = &delta.mutation
            else {
                continue;
            };
            let prefix = relational_key_prefix(table);
            for (key, _) in inserted_rows {
                let row_id = crate::engine_residency::parse_relational_row_id(key, &prefix)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction INSERT lost provisional entity identity".to_string(),
                        ))
                    })?;
                if !identities.insert((table.clone(), row_id)) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction reused a provisional entity identity".to_string(),
                    )));
                }
            }
        }
        Ok(identities)
    }

    fn resolved_transaction_record(
        deltas: &[WriteDelta],
        provisional_inserts: &BTreeSet<(String, u64)>,
        final_base: u64,
        allocator_high_water: u64,
    ) -> Result<BinaryTransactionRecord, ExecuteError> {
        let final_ids = provisional_inserts
            .iter()
            .cloned()
            .zip(final_base..)
            .collect::<BTreeMap<_, _>>();
        let mut mutations = Vec::new();
        let mut sequence_advances = BTreeMap::new();
        for delta in deltas {
            match &delta.mutation {
                PreparedMutation::Insert {
                    table,
                    inserted_rows,
                    seq_advances,
                    ..
                } => {
                    sequence_advances.extend(
                        seq_advances
                            .iter()
                            .map(|(sequence, state)| (sequence.clone(), *state)),
                    );
                    let prefix = relational_key_prefix(table);
                    for (key, row) in inserted_rows {
                        let provisional =
                            crate::engine_residency::parse_relational_row_id(key, &prefix)
                                .ok_or_else(|| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transaction INSERT lost entity identity".to_string(),
                                    ))
                                })?;
                        let row_id = *final_ids
                            .get(&(table.clone(), provisional))
                            .expect("insert identity map covers every inserted row");
                        mutations.push(BinaryTransactionMutation::Insert {
                            table: table.clone(),
                            row_id,
                            row_encoded: encode_relational_row(row),
                        });
                    }
                }
                PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => {
                    if installs.len() != updated_old_rows.len() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction UPDATE is not a resolved resident mutation".to_string(),
                        )));
                    }
                    let prefix = relational_key_prefix(table);
                    for ((_, key, new_row), old_row) in installs.iter().zip(updated_old_rows) {
                        let provisional =
                            crate::engine_residency::parse_relational_row_id(key, &prefix)
                                .ok_or_else(|| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transaction UPDATE lost entity identity".to_string(),
                                    ))
                                })?;
                        let row_id = final_ids
                            .get(&(table.clone(), provisional))
                            .copied()
                            .unwrap_or(provisional);
                        mutations.push(BinaryTransactionMutation::Update {
                            table: table.clone(),
                            row_id,
                            old_row_encoded: encode_relational_row(old_row),
                            new_row_encoded: encode_relational_row(new_row),
                        });
                    }
                }
                PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => {
                    if delta.write_set.rows.len() != deleted_rows.len() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction DELETE is not a resolved resident mutation".to_string(),
                        )));
                    }
                    let prefix = relational_key_prefix(table);
                    for (key, old_row) in delta.write_set.rows.iter().zip(deleted_rows) {
                        let provisional =
                            crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                                .ok_or_else(|| {
                                    ExecuteError::Engine(EngineError::ApplyFailed(
                                        "transaction DELETE lost entity identity".to_string(),
                                    ))
                                })?;
                        let row_id = final_ids
                            .get(&(table.clone(), provisional))
                            .copied()
                            .unwrap_or(provisional);
                        mutations.push(BinaryTransactionMutation::Delete {
                            table: table.clone(),
                            row_id,
                            old_row_encoded: encode_relational_row(old_row),
                        });
                    }
                }
            }
        }
        let mutations = Self::coalesce_transaction_mutations(mutations)?;
        Ok(BinaryTransactionRecord {
            allocator_high_water,
            sequence_advances,
            mutations,
        })
    }

    fn validate_transaction_device_row_conflicts(
        &self,
        snapshot: &TransactionSnapshot,
        deltas: &[WriteDelta],
        provisional_inserts: &BTreeSet<(String, u64)>,
    ) -> Result<(), ExecuteError> {
        let current_boundary = self.committed_seq();
        let mut checked = BTreeSet::new();
        for delta in deltas {
            let (table_name, targets): (&str, Vec<(&str, &[SqlValue])>) = match &delta.mutation {
                PreparedMutation::Insert { .. } => continue,
                PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => (
                    table,
                    installs
                        .iter()
                        .zip(updated_old_rows)
                        .map(|((_, key, _), row)| (key.as_str(), row.as_slice()))
                        .collect(),
                ),
                PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => (
                    table,
                    delta
                        .write_set
                        .rows
                        .iter()
                        .zip(deleted_rows)
                        .map(|(key, row)| (key.row_key.as_str(), row.as_slice()))
                        .collect(),
                ),
            };
            let table = snapshot
                .catalog
                .relational_catalog
                .get(table_name)
                .expect("staged transaction table remains in its retained catalog");
            let prefix = relational_key_prefix(table_name);
            for (row_key, expected_row) in targets {
                let row_id = crate::engine_residency::parse_relational_row_id(row_key, &prefix)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction conflict target lost stable entity identity".to_string(),
                        ))
                    })?;
                let identity = (table_name.to_string(), row_id);
                if provisional_inserts.contains(&identity) || !checked.insert(identity) {
                    continue;
                }
                if snapshot.chunk_authoritative_tables.contains_key(table_name) {
                    let Some((observed, created_by, _, _)) =
                        self.class_row_by_entity_identity(table, row_id, current_boundary)
                    else {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            snapshot.boundary
                        )));
                    };
                    if created_by > snapshot.boundary || observed.as_slice() != expected_row {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            snapshot.boundary
                        )));
                    }
                    continue;
                }
                let filters = expected_row
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                    .collect::<Vec<_>>();
                let Some((key_id, needle)) = self.dml_device_probe_key(table, &filters) else {
                    return Err(ExecuteError::Serialization(format!(
                        "device write-conflict verdict unavailable for relation \"{table_name}\""
                    )));
                };
                let Some(hits) =
                    self.locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
                else {
                    return Err(ExecuteError::Serialization(format!(
                        "device write-conflict verdict unavailable for relation \"{table_name}\""
                    )));
                };
                let mut matched = 0usize;
                for hit in hits {
                    if self.hit_entity_id(&hit) != Some(row_id) {
                        continue;
                    }
                    let created_by = match &hit.created_by {
                        None => 0,
                        Some(region) => {
                            let halves = region
                                .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                                .map_err(|err| {
                                    ExecuteError::Serialization(format!(
                                        "device write-conflict stamp read failed for relation \"{table_name}\": {err}"
                                    ))
                                })?;
                            (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32)
                        }
                    };
                    if created_by > snapshot.boundary {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            snapshot.boundary
                        )));
                    }
                    match self.materialize_resident_row_via_hit(table, &hit, current_boundary) {
                        Some(Some(observed)) if observed.as_slice() == expected_row => matched += 1,
                        Some(Some(_)) | Some(None) => {}
                        None => {
                            return Err(ExecuteError::Serialization(format!(
                                "device write-conflict row verification failed for relation \"{table_name}\""
                            )))
                        }
                    }
                }
                if matched != 1 {
                    return Err(ExecuteError::Serialization(format!(
                        "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                        snapshot.boundary
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_transaction_device_unique_conflicts(
        &self,
        snapshot: &TransactionSnapshot,
        record: &BinaryTransactionRecord,
    ) -> Result<(), ExecuteError> {
        // Every UPDATE/DELETE identity retires its old unique-key ownership at this transaction's
        // single publish boundary. Candidate final rows (including INSERTs) must exclude the whole
        // retiring set, not merely an UPDATE's own identity: key release+reuse and key swaps are
        // valid when the transaction's final relation is unique.
        let mut retiring_ids = BTreeMap::<String, BTreeSet<u64>>::new();
        for mutation in &record.mutations {
            match mutation {
                BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => {
                    retiring_ids
                        .entry(table.clone())
                        .or_default()
                        .insert(*row_id);
                }
                BinaryTransactionMutation::Insert { .. } => {}
            }
        }
        // First-committer-wins includes released keys, not only current duplicates. Scan every old
        // and new exact key against physical device version history before the current-row check
        // below; an insert+delete or key-away after BEGIN must still serialize this transaction.
        for mutation in &record.mutations {
            let (table_name, encoded_rows): (&str, Vec<&str>) = match mutation {
                BinaryTransactionMutation::Insert {
                    table, row_encoded, ..
                } => (table, vec![row_encoded]),
                BinaryTransactionMutation::Update {
                    table,
                    old_row_encoded,
                    new_row_encoded,
                    ..
                } => (table, vec![old_row_encoded, new_row_encoded]),
                BinaryTransactionMutation::Delete {
                    table,
                    old_row_encoded,
                    ..
                } => (table, vec![old_row_encoded]),
            };
            let table = snapshot
                .catalog
                .relational_catalog
                .get(table_name)
                .expect("transaction record table remains in retained catalog");
            if !table.indexes.iter().any(|index| index.unique) {
                continue;
            }
            let rows = encoded_rows
                .into_iter()
                .map(|encoded| {
                    decode_relational_row(encoded, &table.columns).map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "transaction unique history image decode failed for relation \"{table_name}\": {err}"
                        )))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
            match self.device_unique_rows_conflict(table, &row_refs, snapshot.boundary) {
                Some(false) => {}
                Some(true) => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique conflict: write history changed in relation \"{table_name}\" after snapshot {}",
                        snapshot.boundary
                    )))
                }
                None => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique-history verdict unavailable for relation \"{table_name}\""
                    )))
                }
            }
        }
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        for mutation in &record.mutations {
            let (table_name, row_id, encoded) = match mutation {
                BinaryTransactionMutation::Insert {
                    table,
                    row_id,
                    row_encoded,
                } => (table, *row_id, row_encoded),
                BinaryTransactionMutation::Update {
                    table,
                    row_id,
                    new_row_encoded,
                    ..
                } => (table, *row_id, new_row_encoded),
                BinaryTransactionMutation::Delete { .. } => continue,
            };
            let table = snapshot
                .catalog
                .relational_catalog
                .get(table_name)
                .expect("transaction record table remains in retained catalog");
            if !table.indexes.iter().any(|index| index.unique) {
                continue;
            }
            let row = decode_relational_row(encoded, &table.columns).map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction unique image decode failed for relation \"{table_name}\": {err}"
                )))
            })?;
            if self.table_chunk_authoritative(table_name).is_some() {
                let exclusion = retiring_ids.get(table_name).map(|ids| {
                    let mut coordinates = BTreeSet::new();
                    let mut epoch = None;
                    for retiring_id in ids {
                        let (_, _, coordinate, observed_epoch) = self
                            .class_row_by_entity_identity(
                                table,
                                *retiring_id,
                                visibility.read_txn_id,
                            )
                            .ok_or_else(|| {
                                ExecuteError::Serialization(format!(
                                    "device retiring-identity verdict unavailable for entity {retiring_id} in relation \"{table_name}\""
                                ))
                            })?;
                        if epoch.is_some_and(|current| current != observed_epoch) {
                            return Err(ExecuteError::Serialization(format!(
                                "device retiring identities crossed class generations in relation \"{table_name}\""
                            )));
                        }
                        epoch = Some(observed_epoch);
                        coordinates.insert(coordinate);
                    }
                    Ok((coordinates, epoch.unwrap_or(0)))
                }).transpose()?;
                let exclusion_ref = exclusion
                    .as_ref()
                    .map(|(coordinates, epoch)| (coordinates, *epoch));
                match self.validate_class_insert_uniqueness(
                    table,
                    std::slice::from_ref(&row),
                    visibility.read_txn_id,
                    exclusion_ref,
                ) {
                    Some(Ok(())) => {}
                    Some(Err(_)) => {
                        return Err(ExecuteError::Serialization(format!(
                            "device unique conflict on entity {row_id} in relation \"{table_name}\""
                        )))
                    }
                    None => {
                        return Err(ExecuteError::Serialization(format!(
                        "device unique-conflict verdict unavailable for relation \"{table_name}\""
                    )))
                    }
                }
                continue;
            }

            let excluded = retiring_ids.get(table_name).map(|ids| {
                ids.iter()
                    .map(|retiring_id| relational_row_key(table_name, *retiring_id))
                    .collect::<BTreeSet<_>>()
            });
            for (ordinal, index) in table
                .indexes
                .iter()
                .enumerate()
                .filter(|(_, index)| index.unique)
            {
                let positions = crate::engine_residency::index_key_column_positions(table, index)
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "device unique-conflict key binding unavailable for relation \"{table_name}\""
                        ))
                    })?;
                let conflict = if crate::engine_residency::index_uses_fingerprint(table, index) {
                    let key_id = crate::engine_residency::index_probe_key_id(table, index, ordinal)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                            "device unique-conflict index unavailable for relation \"{table_name}\""
                        ))
                        })?;
                    let key_columns = positions
                        .iter()
                        .map(|position| (*position, row[*position].clone()))
                        .collect::<Vec<_>>();
                    match crate::engine_residency::compound_index_row_fingerprint(
                        table, index, &row,
                    ) {
                        Some(fingerprint) => self.device_visible_row_with_tuple(
                            table,
                            visibility,
                            key_id,
                            Some(fingerprint),
                            &key_columns,
                            excluded.as_ref(),
                        ),
                        None if positions.len() == 1 => self.device_visible_row_with_value(
                            table,
                            visibility,
                            positions[0],
                            &row[positions[0]],
                            excluded.as_ref(),
                        ),
                        None => self.device_visible_row_with_tuple(
                            table,
                            visibility,
                            key_id,
                            None,
                            &key_columns,
                            excluded.as_ref(),
                        ),
                    }
                } else {
                    self.device_visible_row_with_value(
                        table,
                        visibility,
                        positions[0],
                        &row[positions[0]],
                        excluded.as_ref(),
                    )
                };
                match conflict {
                    Some(false) => {}
                    Some(true) => {
                        return Err(ExecuteError::Serialization(format!(
                            "device unique conflict on entity {row_id} in relation \"{table_name}\""
                        )))
                    }
                    None => {
                        return Err(ExecuteError::Serialization(format!(
                        "device unique-conflict verdict unavailable for relation \"{table_name}\""
                    )))
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_transaction_device_foreign_key_conflicts(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        record: &BinaryTransactionRecord,
    ) -> Result<(), ExecuteError> {
        type IdentityRows = BTreeMap<String, Vec<(u64, Vec<SqlValue>)>>;
        let mut final_rows = IdentityRows::new();
        let mut removed_rows = IdentityRows::new();
        let mut mutated_ids = BTreeMap::<String, BTreeSet<u64>>::new();
        for mutation in &record.mutations {
            let (table_name, row_id) = match mutation {
                BinaryTransactionMutation::Insert { table, row_id, .. }
                | BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => (table, *row_id),
            };
            let table = snapshot
                .catalog
                .relational_catalog
                .get(table_name)
                .expect("transaction FK table remains in retained catalog");
            mutated_ids
                .entry(table_name.clone())
                .or_default()
                .insert(row_id);
            match mutation {
                BinaryTransactionMutation::Insert { row_encoded, .. } => {
                    let row = decode_relational_row(row_encoded, &table.columns).map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "transaction FK insert image decode failed for relation \"{table_name}\": {err}"
                        )))
                    })?;
                    final_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, row));
                }
                BinaryTransactionMutation::Update {
                    old_row_encoded,
                    new_row_encoded,
                    ..
                } => {
                    let old = decode_relational_row(old_row_encoded, &table.columns).map_err(
                        |err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction FK old image decode failed for relation \"{table_name}\": {err}"
                            )))
                        },
                    )?;
                    let new = decode_relational_row(new_row_encoded, &table.columns).map_err(
                        |err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction FK new image decode failed for relation \"{table_name}\": {err}"
                            )))
                        },
                    )?;
                    removed_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, old));
                    final_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, new));
                }
                BinaryTransactionMutation::Delete {
                    old_row_encoded, ..
                } => {
                    let old = decode_relational_row(old_row_encoded, &table.columns).map_err(
                        |err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction FK delete image decode failed for relation \"{table_name}\": {err}"
                            )))
                        },
                    )?;
                    removed_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, old));
                }
            }
        }
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };

        // Outbound stamps: untouched current providers are checked against the current device
        // generation. Providers written by this transaction must exist in its final private
        // device generation; the final-row stamp prevents an old BEGIN version from satisfying it.
        for (child_name, rows) in &final_rows {
            let child = snapshot
                .catalog
                .relational_catalog
                .get(child_name)
                .expect("transaction child remains cataloged");
            for foreign_key in &child.foreign_keys {
                let Some(parent) = snapshot
                    .catalog
                    .relational_catalog
                    .get(&foreign_key.referenced_table)
                else {
                    continue;
                };
                let child_idx =
                    relational_column_index(child, &foreign_key.column).map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                let parent_idx = relational_column_index(parent, &foreign_key.referenced_column)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                let parent_exclusions = mutated_ids.get(&parent.name).map(|ids| {
                    ids.iter()
                        .map(|row_id| relational_row_key(&parent.name, *row_id))
                        .collect::<BTreeSet<_>>()
                });
                for (_, row) in rows {
                    let value = &row[child_idx];
                    if matches!(value, SqlValue::Null) {
                        continue;
                    }
                    let current = self
                        .device_visible_row_with_value(
                            parent,
                            visibility,
                            parent_idx,
                            value,
                            parent_exclusions.as_ref(),
                        )
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "device foreign-key provider verdict unavailable for relation \"{}\"",
                                parent.name
                            ))
                        })?;
                    if current {
                        continue;
                    }
                    let final_provider_stamp =
                        final_rows.get(&parent.name).is_some_and(|providers| {
                            providers
                                .iter()
                                .any(|(_, provider)| provider[parent_idx] == *value)
                        });
                    let private_provider = if final_provider_stamp {
                        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
                        self.device_visible_row_with_value(
                            parent,
                            StorageVisibility {
                                read_txn_id: snapshot.boundary,
                            },
                            parent_idx,
                            value,
                            None,
                        )
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "private device foreign-key provider verdict unavailable for relation \"{}\"",
                                parent.name
                            ))
                        })?
                    } else {
                        false
                    };
                    if !private_provider {
                        return Err(ExecuteError::Serialization(format!(
                            "device foreign-key conflict on constraint \"{}\" in relation \"{}\"",
                            foreign_key.name, child.name
                        )));
                    }
                }
            }
        }

        // Inbound stamps: final private child images are explicit conflicts; current untouched
        // children are checked on-device with transaction-removed identities excluded.
        for (parent_name, rows) in &removed_rows {
            let parent = snapshot
                .catalog
                .relational_catalog
                .get(parent_name)
                .expect("transaction parent remains cataloged");
            for child in snapshot.catalog.relational_catalog.values() {
                for foreign_key in child
                    .foreign_keys
                    .iter()
                    .filter(|foreign_key| foreign_key.referenced_table == *parent_name)
                {
                    let parent_idx =
                        relational_column_index(parent, &foreign_key.referenced_column).map_err(
                            |err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())),
                        )?;
                    let child_idx =
                        relational_column_index(child, &foreign_key.column).map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    let child_exclusions = mutated_ids.get(&child.name).map(|ids| {
                        ids.iter()
                            .map(|row_id| relational_row_key(&child.name, *row_id))
                            .collect::<BTreeSet<_>>()
                    });
                    let final_children = final_rows.get(&child.name);
                    let final_providers = final_rows.get(parent_name);
                    let mut checked = BTreeSet::<SqlValue>::new();
                    for (_, old_parent) in rows {
                        let value = old_parent[parent_idx].clone();
                        if matches!(value, SqlValue::Null) || !checked.insert(value.clone()) {
                            continue;
                        }
                        if final_providers.is_some_and(|providers| {
                            providers
                                .iter()
                                .any(|(_, provider)| provider[parent_idx] == value)
                        }) {
                            continue;
                        }
                        if final_children.is_some_and(|children| {
                            children
                                .iter()
                                .any(|(_, child_row)| child_row[child_idx] == value)
                        }) {
                            return Err(ExecuteError::Serialization(format!(
                                "device foreign-key conflict on constraint \"{}\" in relation \"{}\"",
                                foreign_key.name, child.name
                            )));
                        }
                        let surviving_child = self
                            .device_visible_row_with_value(
                                child,
                                visibility,
                                child_idx,
                                &value,
                                child_exclusions.as_ref(),
                            )
                            .ok_or_else(|| {
                                ExecuteError::Serialization(format!(
                                    "device foreign-key child verdict unavailable for relation \"{}\"",
                                    child.name
                                ))
                            })?;
                        if surviving_child {
                            return Err(ExecuteError::Serialization(format!(
                                "device foreign-key conflict on constraint \"{}\" in relation \"{}\"",
                                foreign_key.name, child.name
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Keep BEGIN-generation preparation and its cheap conflict verdict available to the CPU
    /// parity oracle, but require an actual retained device generation before publishing a private
    /// delta. The CPU store is not a transaction-execution fallback for a conflict-free statement.
    fn validate_transaction_delta_residency(
        &self,
        table: &RelationalTable,
    ) -> Result<(), ExecuteError> {
        if self.table_chunk_authoritative(&table.name).is_some() {
            if self.read_streaming_cold_chunks().contains_key(&table.name) {
                return Ok(());
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no retained cold-chunk generation for transaction staging",
                table.name
            ))));
        }
        let shards = self.read_residency_shards();
        if shards
            .get(&table.name)
            .is_none_or(|table_shards| table_shards.is_empty())
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no retained GPU shard generation for transaction staging",
                table.name
            ))));
        }
        Ok(())
    }

    fn apply_transaction_private_delta(
        &self,
        table: &RelationalTable,
        delta: &WriteDelta,
        boundary: Index,
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        cold_chunks: &mut BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        if self.table_chunk_authoritative(&table.name).is_some() {
            let prefix = relational_key_prefix(&table.name);
            let entry = cold_chunks.get(&table.name).cloned().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" lost its retained cold-chunk generation",
                    table.name
                )))
            })?;
            let parse_ids = |keys: Vec<&str>| -> Result<Vec<u64>, ExecuteError> {
                keys.into_iter()
                    .map(|key| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction cold mutation lost stable entity identity"
                                        .to_string(),
                                ))
                            },
                        )
                    })
                    .collect()
            };
            let next = match &delta.mutation {
                PreparedMutation::Insert { inserted_rows, .. } => {
                    let rows = inserted_rows
                        .iter()
                        .map(|(_, row)| row.clone())
                        .collect::<Vec<_>>();
                    let row_ids =
                        parse_ids(inserted_rows.iter().map(|(key, _)| key.as_str()).collect())?;
                    self.append_transaction_cold_tail(table, &entry, &rows, &row_ids, boundary)
                }
                PreparedMutation::Update {
                    installs,
                    class_epoch: Some(epoch),
                    ..
                } => {
                    let coordinates = installs
                        .iter()
                        .map(|(coordinate, _, _)| *coordinate)
                        .collect::<Vec<_>>();
                    let row_ids =
                        parse_ids(installs.iter().map(|(_, key, _)| key.as_str()).collect())?;
                    let rows = installs
                        .iter()
                        .map(|(_, _, row)| row.clone())
                        .collect::<Vec<_>>();
                    self.stamp_transaction_cold_coordinates(&entry, &coordinates, *epoch, boundary)
                        .and_then(|stamped| {
                            self.append_transaction_cold_tail(
                                table, &stamped, &rows, &row_ids, boundary,
                            )
                        })
                }
                PreparedMutation::Delete {
                    tuple_ids,
                    class_epoch: Some(epoch),
                    ..
                } => self.stamp_transaction_cold_coordinates(&entry, tuple_ids, *epoch, boundary),
                PreparedMutation::Update {
                    class_epoch: None, ..
                }
                | PreparedMutation::Delete {
                    class_epoch: None, ..
                } => None,
            }
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" could not build a transaction-private cold delta",
                    table.name
                )))
            })?;
            cold_chunks.insert(table.name.clone(), next);
            return Ok(());
        }
        let prefix = relational_key_prefix(&table.name);
        match &delta.mutation {
            PreparedMutation::Insert { inserted_rows, .. } => {
                let row_ids = inserted_rows
                    .iter()
                    .map(|(key, _)| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction INSERT produced a non-canonical entity identity"
                                        .to_string(),
                                ))
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let rows = inserted_rows
                    .iter()
                    .map(|(_, row)| row.clone())
                    .collect::<Vec<_>>();
                self.append_transaction_delta_shard(table, &rows, &row_ids, shards, gpu_reservation)
            }
            PreparedMutation::Update {
                installs,
                updated_old_rows,
                ..
            } => {
                let row_ids = installs
                    .iter()
                    .map(|(_, key, _)| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction UPDATE lost stable entity identity".to_string(),
                                ))
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if row_ids.len() != updated_old_rows.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction UPDATE old/new identity vectors are not parallel".to_string(),
                    )));
                }
                let targets = row_ids
                    .iter()
                    .copied()
                    .zip(updated_old_rows.iter().map(Vec::as_slice))
                    .collect::<Vec<_>>();
                self.tombstone_transaction_rows(
                    table,
                    &targets,
                    boundary,
                    shards,
                    gpu_reservation,
                )?;
                let rows = installs
                    .iter()
                    .map(|(_, _, row)| row.clone())
                    .collect::<Vec<_>>();
                self.append_transaction_delta_shard(table, &rows, &row_ids, shards, gpu_reservation)
            }
            PreparedMutation::Delete { deleted_rows, .. } => {
                if deleted_rows.len() != delta.write_set.rows.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction DELETE row images lost their identity ordering".to_string(),
                    )));
                }
                let targets = delta
                    .write_set
                    .rows
                    .iter()
                    .zip(deleted_rows)
                    .map(|(key, row)| {
                        crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                            .map(|row_id| (row_id, row.as_slice()))
                            .ok_or_else(|| {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction DELETE lost stable entity identity".to_string(),
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.tombstone_transaction_rows(table, &targets, boundary, shards, gpu_reservation)
            }
        }
    }

    fn tombstone_transaction_rows(
        &self,
        table: &RelationalTable,
        targets: &[TargetRow<'_>],
        boundary: Index,
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        let mut slots_by_shard: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for &(row_id, old_row) in targets {
            let filters = old_row
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                .collect::<Vec<_>>();
            let (key_id, needle) = self.dml_device_probe_key(table, &filters).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no device key usable by the transaction delta",
                    table.name
                )))
            })?;
            let hits = self
                .locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" transaction target locate declined",
                        table.name
                    )))
                })?;
            let mut visible = Vec::new();
            let mut observed = Vec::new();
            for hit in hits {
                let Some(identity) = self.hit_entity_id(&hit) else {
                    observed.push((hit.shard_id, hit.slot, None, None));
                    continue;
                };
                let materialized = self.materialize_resident_row_via_hit(table, &hit, boundary);
                observed.push((hit.shard_id, hit.slot, Some(identity), materialized.clone()));
                if identity != row_id {
                    continue;
                }
                if materialized.is_some_and(|row| row.as_deref() == Some(old_row)) {
                    visible.push((hit.shard_id, hit.slot));
                }
            }
            if visible.len() != 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction target entity {row_id} resolved {} visible versions; expected {old_row:?}, observed {observed:?}",
                    visible.len(),
                ))));
            }
            let (shard_id, slot) = visible[0];
            slots_by_shard.entry(shard_id).or_default().push(slot);
        }

        let table_shards = shards.get_mut(&table.name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" lost its transaction shard generation",
                table.name
            )))
        })?;
        for (shard_id, mut slots) in slots_by_shard {
            slots.sort_unstable();
            slots.dedup();
            let shard = table_shards
                .iter_mut()
                .find(|shard| shard.shard_id == shard_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "transaction target shard {shard_id} disappeared"
                    )))
                })?;
            let private = self.clone_transaction_deleted_region(shard, gpu_reservation)?;
            let stamps = vec![boundary; slots.len()];
            private.scatter_u64_slots(&slots, &stamps).map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction tombstone scatter failed: {err}"
                )))
            })?;
            shard.deleted_by_region = Some(private);
        }
        Ok(())
    }

    pub(crate) fn hit_entity_id(
        &self,
        hit: &crate::engine_retained_read::ShardPkHit,
    ) -> Option<u64> {
        let words = hit
            .row_id
            .as_ref()?
            .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
            .ok()?;
        let lo = *words.first()? as u32 as u64;
        let hi = *words.get(1)? as u32 as u64;
        Some(lo | (hi << 32))
    }
}
