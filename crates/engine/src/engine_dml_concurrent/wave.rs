use super::canonical::WaveCanonicalFailure;
#[cfg(test)]
use super::Command;
use super::{
    current_timestamp_micros, relational_key_prefix, wave_host_phase_timing_enabled,
    CommitPathFailure, CommitWaveItem, CommitWaveTail, DmlReadSnapshot, Engine, EngineError,
    ExecuteError, Index, SqlValue, WAVE_HOST_STATS,
};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::Instant;

/// Fails every still-unset outcome in a wave batch if the sequencer dies mid-wave (the
/// apply-invariant panic path), and wedges the queue so waiters and future committers error out
/// instead of hanging. Forgotten (`std::mem::forget`) on the successful path.
struct CommitWaveBatchGuard<'a> {
    engine: &'a Engine,
    outcomes: &'a [super::CommitWaveOutcome],
}

impl Drop for CommitWaveBatchGuard<'_> {
    fn drop(&mut self) {
        let mut queue = self.engine.lock_commit_wave_queue();
        let failure = queue.wedged.clone().unwrap_or_else(|| {
            CommitPathFailure::compatibility("commit-wave sequencer died mid-wave".to_string())
        });
        queue.wedged = Some(failure.clone());
        queue.sequencer_active = false;
        // Fail everything still queued too — no sequencer will ever run it.
        let stranded: Vec<CommitWaveItem> = queue.items.drain(..).collect();
        drop(queue);
        for outcome in self.outcomes {
            if !outcome.done.load(AtomicOrdering::Acquire) {
                outcome.set_outcome(Err(failure.outcome_error()));
            }
        }
        for item in &stranded {
            if !item.outcome.done.load(AtomicOrdering::Acquire) {
                item.set_outcome(Err(failure.outcome_error()));
            }
        }
        self.engine.commit_wave.cv.notify_all();
        self.engine.wedge_commit_path();
    }
}

/// Preserve a fixed post-WAL durability identity when a locally held wave observes the service
/// gate after another tail failed. Compatibility errors retain their established text outcome.
fn settle_batch_after_commit_path_failure(batch: &[CommitWaveItem], error: EngineError) {
    match error {
        EngineError::DurabilityFault(fault) => {
            for item in batch {
                item.set_outcome(Err(ExecuteError::IndeterminateDurability(fault)));
            }
        }
        error => {
            let message = error.to_string();
            for item in batch {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    message.clone(),
                ))));
            }
        }
    }
}

impl Engine {
    /// Commit one WAVE: the per-item (3a)-(3e) steps of the old per-commit critical section, run
    /// back-to-back under ONE commit_mutex hold in wave order. W2: the durability tail (group
    /// fsync wait + `committed_seq` publish + acks) is RETURNED as a [`CommitWaveTail`] instead of
    /// running inline, so the caller can
    /// pipeline it against the next wave's sequencing. `None` = every item aborted pre-durable
    /// (outcomes already set). Every item's outcome slot is set exactly once; the
    /// `CommitWaveBatchGuard` fails any still-unset outcome (and wedges the queue) if this
    /// thread panics mid-wave (e.g. the apply-invariant panic, which also poisons the
    /// commit_mutex — the established wedge-don't-serve-torn-state policy); once the tail is
    /// built, its `armed` Drop carries that responsibility.
    pub(super) fn sequence_commit_wave(
        &self,
        batch: Vec<CommitWaveItem>,
    ) -> Option<CommitWaveTail> {
        // The entire wave (conflict-check, device re-resolve, flush, apply, publish) runs inside
        // one commit critical section. The internal-read marker preserves the existing lock
        // discipline for catalog/materialized-view work without enabling a DML repair fallback.
        self.skip_leader_check_during_internal_read(|engine| {
            engine.sequence_commit_wave_inner(batch)
        })
    }

    fn sequence_commit_wave_inner(&self, batch: Vec<CommitWaveItem>) -> Option<CommitWaveTail> {
        let guard_outcomes = batch
            .iter()
            .map(|item| std::sync::Arc::clone(&item.outcome))
            .collect::<Vec<_>>();
        let guard = CommitWaveBatchGuard {
            engine: self,
            outcomes: &guard_outcomes,
        };
        let wall_clock = current_timestamp_micros();
        let mut wave_tail: Option<(Index, usize)> = None;
        let mut committed: Vec<(usize, Index, u64)> = Vec::with_capacity(batch.len());

        let mut commit = self.commit_state();
        let next_row_id = self.read_state.mvcc.current_row_id();
        if let Err(error) = self.ensure_commit_path_available() {
            settle_batch_after_commit_path_failure(&batch, error);
            std::mem::forget(guard);
            return None;
        }
        if let Err(error) = self.legacy_lane_history_write_guard() {
            let message = error.to_string();
            for item in &batch {
                item.set_outcome(Err(ExecuteError::Engine(EngineError::Durability(
                    message.clone(),
                ))));
            }
            std::mem::forget(guard);
            return None;
        }
        let hostphase = wave_host_phase_timing_enabled();
        // E2.3 — the catalog generation is CONSTANT for the whole wave: DDL is the only publisher
        // and it commits under the very commit_mutex this sequencer holds, so no generation bump can
        // interleave a wave's items. Load the snapshot ONCE here instead of per item (the old
        // per-item `catalog_snapshot()` was an ArcSwap load + Arc clone on every commit — the
        // generation gate at re-resolve, the fast-run eligibility probe, and the intent fast-lane
        // gate all read it). `wave_catalog_seq` is the schema stamp every item compares against.
        let wave_catalog = self.catalog_snapshot();
        let wave_catalog_seq = wave_catalog.commit_seq;
        // Unique keys of successful earlier members whose device append/tombstone may still be
        // buffered until this wave's flush. This set is wave-bounded; committed history lives in
        // the resident version stamps queried below, not in the CPU commit ledger.
        let mut wave_unique_slots: std::collections::HashSet<
            crate::write_path::UniqueIndexSlotKey,
        > = std::collections::HashSet::new();
        let mut wave_unique_slots_i32: std::collections::HashSet<
            crate::write_path::IntUniqueSlotKey,
        > = std::collections::HashSet::new();
        // HOST-phase probe: `_hp` timestamps the running phase boundary; `hp!(k)` charges the elapsed
        // time since the last boundary to WAVE_HOST_STATS[k] and resets. Reset at each item's top.
        let mut _hp = hostphase.then(Instant::now);
        macro_rules! hp {
            ($k:expr) => {
                if let Some(ref mut t) = _hp {
                    let now = Instant::now();
                    WAVE_HOST_STATS[$k].fetch_add(
                        now.duration_since(*t).as_nanos() as u64,
                        AtomicOrdering::Relaxed,
                    );
                    *t = now;
                }
            };
        }
        // Index-based (not `iter().enumerate()`): the intent fast lane and the general path both
        // reach `batch[position]` while the `flush_*` closures also borrow `batch` — an index keeps
        // those borrows disjoint per statement without threading an iterator through the closures.
        #[allow(clippy::needless_range_loop)]
        for position in 0..batch.len() {
            if let Some(ref mut t) = _hp {
                *t = Instant::now();
            }
            let request_digest = batch[position].request.digest();
            match commit
                .resolve_transaction_retry_digest_outcome(batch[position].txn_id, request_digest)
            {
                Ok(Some((token, affected_rows))) if self.committed_seq() >= token.index => {
                    batch[position].set_outcome(Ok(affected_rows));
                    continue;
                }
                Ok(Some(_)) => {
                    batch[position].set_outcome(Err(ExecuteError::Indeterminate(format!(
                        "transaction id {} is committed but not yet publication-covered",
                        batch[position].txn_id
                    ))));
                    continue;
                }
                Err(error) => {
                    batch[position].set_outcome(Err(ExecuteError::Engine(error)));
                    continue;
                }
                Ok(None) => {}
            }
            match self.resolve_pending_transaction_claim(batch[position].txn_id, request_digest) {
                Ok(true) => {
                    batch[position].set_outcome(Err(ExecuteError::Indeterminate(format!(
                        "transaction id {} is pending in canonical mutation admission",
                        batch[position].txn_id
                    ))));
                    continue;
                }
                Err(error) => {
                    batch[position].set_outcome(Err(ExecuteError::Engine(error)));
                    continue;
                }
                Ok(false) => {}
            }
            if let Some(expectation) = batch[position].expected_catalog_version {
                if let Err(error) =
                    crate::engine_mutation_admission::validate_catalog_version_expectation(
                        expectation,
                        wave_catalog_seq,
                    )
                {
                    batch[position].set_outcome(Err(error));
                    continue;
                }
            }
            if let Some(state) = commit.txn_manager.state(batch[position].txn_id) {
                batch[position].set_outcome(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    format!(
                        "transaction id {} is already owned by transaction state {state:?}",
                        batch[position].txn_id
                    ),
                ))));
                continue;
            }
            // (3a) SI first-committer-wins. Row identity conflicts retain the bounded
            // oldest-active CPU map. Unique conflicts come from exact device version history
            // plus the wave-local bridge for earlier members in this UPDATE/DELETE wave.
            let item = &batch[position];
            let row_conflict = commit
                .ledger
                .conflicts_rows(&item.write_set, item.read_snapshot);
            let has_unique = !item.write_set.unique_slots.is_empty()
                || !item.write_set.unique_slots_i32.is_empty();
            let wave_unique_conflict = item
                .write_set
                .unique_slots
                .iter()
                .any(|slot| wave_unique_slots.contains(slot))
                || item
                    .write_set
                    .unique_slots_i32
                    .iter()
                    .any(|slot| wave_unique_slots_i32.contains(slot));
            let device_unique_conflict = if !has_unique || wave_unique_conflict {
                false
            } else {
                match item.offlock_prepared.as_ref().and_then(|prepared| {
                    prepared.legacy_delta().and_then(|delta| {
                        self.device_unique_write_conflicts(delta, item.read_snapshot)
                    })
                }) {
                    Some(conflict) => conflict,
                    None => {
                        // Host-neutral specification fixtures keep their parity ledger without
                        // claiming execution. Production has no host authority: a missing device
                        // history verdict fails closed. Test builds follow that same law once a
                        // table is device-authoritative, so an actual-GPU acceptance target cannot
                        // pass via the cfg(test) parity map.
                        #[cfg(test)]
                        {
                            let device_authoritative = match &item.cmd {
                                Command::Update(update) => Some(update.table.as_str()),
                                Command::Delete(delete) => Some(delete.table.as_str()),
                                _ => None,
                            }
                            .is_some_and(|table| {
                                self.table_device_authoritative(table)
                                    || self.table_chunk_authoritative(table).is_some()
                            });
                            device_authoritative
                                || commit
                                    .ledger
                                    .conflicts_unique(&item.write_set, item.read_snapshot)
                        }
                        #[cfg(not(test))]
                        {
                            true
                        }
                    }
                }
            };
            if row_conflict || wave_unique_conflict || device_unique_conflict {
                let read_snapshot = batch[position].read_snapshot;
                batch[position].set_outcome(Err(ExecuteError::Serialization(format!(
                    "write-write conflict on a key committed after read snapshot {read_snapshot}"
                ))));
                continue;
            }
            hp!(1);

            let item = &batch[position];

            // (3b) Re-resolve UPDATE/DELETE at the peeked sequence.
            let commit_seq = commit.repl.peek_next_index();
            let install_snapshot = DmlReadSnapshot {
                commit_seq,
                next_row_id,
            };
            let prepared = self.prepare_dml(&item.cmd, install_snapshot);
            let delta = match prepared {
                Ok(delta) => delta,
                Err(err) => {
                    item.set_outcome(Err(match err {
                        ExecuteError::Serialization(_)
                        | ExecuteError::IndeterminateDurability(_) => err,
                        other => ExecuteError::Serialization(format!(
                            "re-resolve at commit_seq {commit_seq} failed on a concurrent \
                             interleaving (retryable): {other}"
                        )),
                    }));
                    continue;
                }
            };
            let item_rows = delta.rows_affected();
            let returning = match self.project_dml_returning(&item.cmd, &delta, commit_seq) {
                Ok(returning) => returning,
                Err(error) => {
                    item.set_outcome(Err(error));
                    continue;
                }
            };
            item.outcome.set_returning(returning);
            hp!(2);
            let wal_payload = item.request.payload_arc();
            let canonical = match self.append_canonical_wave_operation(
                &mut commit,
                item.txn_id,
                commit_seq,
                wall_clock,
                item.request.digest(),
                &item.write_set,
                wal_payload,
                if item_rows == 0 {
                    gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
                } else {
                    gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                },
                item_rows,
            ) {
                Ok(canonical) => canonical,
                Err(WaveCanonicalFailure::PreDurable(error)) => {
                    item.set_outcome(Err(error));
                    continue;
                }
            };
            assert_eq!(
                canonical.commit_seq, commit_seq,
                "commit-path invariant violation: resolved canonical sequence drifted"
            );
            hp!(3);

            wave_unique_slots.extend(item.write_set.unique_slots.iter().cloned());
            wave_unique_slots_i32.extend(item.write_set.unique_slots_i32.iter().copied());
            hp!(4);

            // (3d) Apply failure after canonical WAL is fatal.
            enum DeviceMaintenance {
                Delete(String, Vec<Vec<SqlValue>>, Vec<u64>),
                Update(String, Vec<Vec<SqlValue>>, Vec<Vec<SqlValue>>, Vec<u64>),
            }
            let device_maintenance = match &delta.mutation {
                crate::write_path::PreparedMutation::Insert { .. } => panic!(
                    "commit-path invariant violation: INSERT reached the UPDATE/DELETE wave after codec-5 ingress cutover"
                ),
                crate::write_path::PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => {
                    let prefix = relational_key_prefix(table);
                    DeviceMaintenance::Delete(
                        table.clone(),
                        deleted_rows.clone(),
                        delta
                            .write_set
                            .rows
                            .iter()
                            .map(|key| {
                                crate::engine_residency::parse_relational_row_id(
                                    &key.row_key,
                                    &prefix,
                                )
                                .unwrap_or(u64::MAX)
                            })
                            .collect(),
                    )
                }
                crate::write_path::PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => {
                    let prefix = relational_key_prefix(table);
                    DeviceMaintenance::Update(
                        table.clone(),
                        updated_old_rows.clone(),
                        installs
                            .iter()
                            .map(|(_, _, values)| values.clone())
                            .collect(),
                        installs
                            .iter()
                            .map(|(_, key, _)| {
                                crate::engine_residency::parse_relational_row_id(key, &prefix)
                                    .unwrap_or(u64::MAX)
                            })
                            .collect(),
                    )
                }
            };
            self.apply_delta(delta, commit_seq, None)
                .unwrap_or_else(|err| {
                    panic!(
                        "commit-path invariant violation: apply at commit_seq {commit_seq} failed \
                     after re-validation at this seq succeeded: {err}"
                    )
                });
            commit.repl.mark_applied(commit_seq);
            hp!(5);

            // Residency, before publish: UPDATE and DELETE maintain the same authoritative
            // generation in place. A device decline
            // invalidates and schedules an exact re-admission before the next wave.
            wave_tail = Some((commit_seq, canonical.wal_position));
            match device_maintenance {
                DeviceMaintenance::Delete(table, rows, row_ids) => {
                    let handled =
                        wave_catalog
                            .relational_catalog
                            .get(&table)
                            .is_some_and(|table| {
                                self.try_tombstone_rows_by_identity(
                                    table, &rows, &row_ids, commit_seq,
                                )
                            });
                    if !handled {
                        panic!(
                            "commit-path invariant violation: durable wave DELETE for relation \"{table}\" \
                             declined device publication — refusing acknowledgement; WAL replay is required"
                        );
                    }
                    committed.push((position, commit_seq, item_rows));
                }
                DeviceMaintenance::Update(table, old_rows, new_rows, row_ids) => {
                    let handled =
                        wave_catalog
                            .relational_catalog
                            .get(&table)
                            .is_some_and(|table| {
                                self.try_update_resident_table(
                                    table,
                                    &old_rows,
                                    &new_rows,
                                    commit_seq,
                                    Some(&row_ids),
                                )
                            });
                    if !handled {
                        panic!(
                            "commit-path invariant violation: durable wave UPDATE for relation \"{table}\" \
                             declined device publication — refusing acknowledgement; WAL replay is required"
                        );
                    }
                    committed.push((position, commit_seq, item_rows));
                }
            }
            hp!(6);
        }
        // Prune the ledger once per wave (was per commit) below the oldest active snapshot.
        if let Some((last_seq, _)) = wave_tail {
            let prune_boundary = self
                .active_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .oldest()
                .map(|oldest| oldest.saturating_sub(1))
                .unwrap_or(last_seq);
            commit.ledger.prune_below(prune_boundary);
        }
        // Register the applied-but-unpublished tail while the commit lock still excludes explicit
        // transaction validation. The sequencer publishes it into the deque after returning; the
        // applied counter closes that handoff gap for the explicit-transaction settle proof.
        if wave_tail.is_some() {
            self.commit_wave
                .tails_applied
                .fetch_add(1, AtomicOrdering::Release);
        }
        // === leave the commit critical section BEFORE the fsync (D3b group commit) ===
        drop(commit);

        let Some((_last_seq, last_position)) = wave_tail else {
            // Every item aborted pre-durable; outcomes are already set.
            std::mem::forget(guard);
            return None;
        };

        // W2: the durability tail (fsync-wait → publish → acks) no longer runs on the
        // sequencer's critical path — it is handed back as a `CommitWaveTail` for the depth-1
        // pipeline, so the NEXT wave's sequencing overlaps THIS wave's fdatasync. The batch
        // guard's responsibility transfers to the tail's own `armed` Drop.
        std::mem::forget(guard);
        Some(CommitWaveTail {
            batch,
            committed,
            last_position,
            armed: true,
            durability_fault: None,
        })
    }
}
