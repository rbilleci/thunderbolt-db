//! Concurrent DML path + top-level text execution (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the snapshot-isolation
//! autocommit write path (is_concurrent_dml, register/deregister_active_snapshot,
//! execute_dml_concurrent[_instrumented], prepare_dml, dml_mutated_tables,
//! commit_dml_concurrent), the execute_text / execute_text_at_timestamp_micros /
//! execute_read_text entry points, and the read-pin helper.

use super::*;
use crate::engine_mutation_admission::validate_transaction_characteristics;
use crate::engine_transaction_reset::StableRetryOr;

mod canonical;
mod control_plane;
mod durability_failure;
mod lane;
mod lane_apply;
mod request;
mod state;
mod wave;

#[cfg(test)]
pub(crate) use canonical::issue_test_only_typed_insert_post_wal_apply_permit;
pub(crate) use canonical::issue_transaction_terminal_typed_insert_apply_permit;
pub(crate) use canonical::TypedInsertPostWalApplyPermit;
pub(crate) use durability_failure::CommitPathFailure;

use durability_failure::{execute_error_from_engine, TailCompletion};
pub(crate) use state::{
    commit_wave_done_payload_bytes, new_pending_outcome, CanonicalRequest, CommitWaveItem,
    CommitWaveOutcome, CommitWaveState, LaneIntent, LaneOpKind,
};
#[cfg(test)]
use state::{wave_tail_failure_publish_hook, wave_tail_handoff_hook};
use state::{CommitWaveDone, CommitWaveQueue, CommitWaveTail, OfflockPreparedDml};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DmlExecutionResult {
    pub rows_affected: u64,
    pub returning: Option<RelationalSelectResult>,
}

/// Upper bound on items per wave drain (keeps a single wave's worst-case commit latency bounded;
/// under saturation the NEXT wave picks the rest up immediately).
const COMMIT_WAVE_MAX_DEFAULT: usize = 1024;

/// Parallel vectors returned by one device launch form a single verdict. Every component must
/// cover exactly the submitted positions: accepting a short vector makes `zip` fail open, while
/// accepting a long vector hides an ABI/launch mismatch that can misassociate later results.
#[inline]
fn exact_device_verdict_cardinality(expected: usize, component_lengths: &[usize]) -> bool {
    component_lengths.iter().all(|length| *length == expected)
}

fn is_device_prepare_verdict_unavailable(error: &ExecuteError) -> bool {
    matches!(
        error,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message.starts_with("device DML verdict unavailable")
                || message.starts_with("device constraint verdict unavailable")
                || message.starts_with("device tuple-constraint verdict unavailable")
    )
}

pub(crate) fn command_has_returning(command: &Command) -> bool {
    match command {
        Command::Insert(insert) => !insert.returning.is_empty(),
        Command::Update(update) => !update.returning.is_empty(),
        Command::Delete(delete) => !delete.returning.is_empty(),
        _ => false,
    }
}

pub(crate) fn discarded_returning_error() -> ExecuteError {
    ExecuteError::Engine(discarded_returning_engine_error())
}

pub(crate) fn discarded_returning_engine_error() -> EngineError {
    EngineError::ApplyFailed("DML RETURNING requires a result-bearing execution API".to_string())
}

/// E2.2(d) — the wave-size / pipeline-depth knobs are env-overridable for the latency-knee sweep
/// (`GPU_DB_COMMIT_WAVE_MAX`, `GPU_DB_WAVE_TAIL_PIPELINE_DEPTH`), read once. The defaults are the
/// production values; the sweep finds the throughput/latency knee once (a)-(c) reshape the loop.
fn commit_wave_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_DB_COMMIT_WAVE_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(COMMIT_WAVE_MAX_DEFAULT)
    })
}

fn wave_tail_pipeline_depth() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GPU_DB_WAVE_TAIL_PIPELINE_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(WAVE_TAIL_PIPELINE_DEPTH_DEFAULT)
    })
}

/// W2b — max sequenced-but-unpublished wave tails outstanding in the durability pipeline. Depth 1
/// forced one fsync per wave (the sequencer stalled at the gate for ~the full fdatasync — a
/// measured 14% durable-arm REGRESSION); at depth N consecutive waves' records coalesce into
/// SHARED fsyncs (the group covers everything appended while the previous flush was in flight),
/// so the fsync count per commit drops toward 1/N. Correctness does not depend on finish order:
/// a tail's durability wait covers all EARLIER WAL positions (prefix durability) and
/// The publication coordinator joins exact ready indices, so concurrent claimers may finish tails
/// out of order without exposing a gap.
/// The bound caps applied-but-unpublished state at N waves (restart recovery replays the durable
/// prefix; nothing unpublished was ever acked).
const WAVE_TAIL_PIPELINE_DEPTH_DEFAULT: u64 = 8;

/// Commit-wave telemetry (waves sequenced / items committed through waves / total sequencing
/// nanos). W2 CHANGED [2]'s meaning: the timer now stops when `sequence_commit_wave` returns —
/// the group-durability tail (fsync-wait + publish + acks) is PIPELINED off the sequencer and
/// is NOT included (pre-W2 numbers included it; comparisons across d6d10f8e are
/// apples-to-oranges). Three relaxed adds PER WAVE — not per item — so it stays on permanently;
/// the phase-D SLO benchmark reads it to report wave amortization (`items/waves`) alongside TPS.
pub static WAVE_STATS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// FUSE-recon device-phase timing (nanos summed across the run): [0]=wave-batch LOCATE (validate),
/// [1]=device APPEND (HtoD chunks), [2]=device INDEX-INSERT (atom.cas kernel). Relaxed adds gated on
/// `GPU_DB_BENCH_DEVPHASE=1` so they never touch the steady-state hot path; the SLO benchmark reads
/// them to attribute the per-wave device round-trip cost and pick the fusion target.
pub static WAVE_DEVICE_STATS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// True when `GPU_DB_BENCH_DEVPHASE=1` — enables the [`WAVE_DEVICE_STATS`] per-phase timing.
pub(crate) fn wave_device_phase_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GPU_DB_BENCH_DEVPHASE").is_ok_and(|v| v == "1"))
}

/// HOST-sequencer per-item phase timing (nanos summed): the serial work under the commit_mutex,
/// which the driver-call probe showed is the PEAK-throughput wall (not device round-trips). Indices:
/// [0]=wave_batch_validate (design-B batched unique, whole call incl. its device locate),
/// [1]=SI conflict check, [2]=under-lock re-resolve (prepare_dml), [3]=sequence (WAL append +
/// propose + wait_committed), [4]=ledger record, [5]=apply (apply_delta / fast-run / append flush),
/// [6]=residency invalidate. Gated on `GPU_DB_BENCH_HOSTPHASE=1` so it never touches steady state.
pub static WAVE_HOST_STATS: [std::sync::atomic::AtomicU64; 7] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// True when `GPU_DB_BENCH_HOSTPHASE=1` — enables the [`WAVE_HOST_STATS`] per-phase timing.
pub(crate) fn wave_host_phase_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GPU_DB_BENCH_HOSTPHASE").is_ok_and(|v| v == "1"))
}

impl Engine {
    /// Translate only the exceptional public identities that arrived after a sequence child had
    /// already claimed their durable numeric slot. Normal callers retain their exact IDs.
    pub(crate) fn resolve_public_transaction_id(&self, txn_id: TxnId) -> TxnId {
        self.public_transaction_aliases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&txn_id)
            .copied()
            .unwrap_or(txn_id)
    }

    /// Resolve a caller identity, allocating a durable surrogate for a compatibility collision
    /// with an engine-owned sequence child or an earlier surrogate. The alias is retained so a
    /// retry of the caller's request resolves to the same canonical terminal identity. All other
    /// occupied durable IDs retain the normal fail-closed retry behavior at their terminal
    /// admission checks.
    pub(crate) fn resolve_or_allocate_public_transaction_id(
        &self,
        txn_id: TxnId,
    ) -> Result<TxnId, TxnError> {
        if let Some(effective_txn_id) = self
            .public_transaction_aliases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&txn_id)
            .copied()
        {
            return Ok(effective_txn_id);
        }

        let commit = self.commit_state();
        if !commit.transaction_status.contains_key(&txn_id) {
            return Ok(txn_id);
        }

        let sequence_child_claim = self
            .sequence_value_outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&txn_id)
            .is_some_and(|applied| applied.record.parent_txn_id != txn_id);
        let surrogate_claim = self
            .public_transaction_aliases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .any(|effective_txn_id| *effective_txn_id == txn_id);
        if !sequence_child_claim && !surrogate_claim {
            return Ok(txn_id);
        }

        let surrogate = self.allocate_unclaimed_transaction_id(&commit)?;
        let mut aliases = self
            .public_transaction_aliases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(*aliases.entry(txn_id).or_insert(surrogate))
    }

    /// Mutate aliases while the caller owns the alias map and, when transitioning a snapshot,
    /// the active-snapshot registry immediately afterwards.  That fixed lock order makes public
    /// identity resolution and snapshot lookup one atomic observation.
    pub(crate) fn complete_public_transaction_alias_in_map(
        aliases: &mut HashMap<TxnId, TxnId>,
        effective_txn_id: TxnId,
        successor: Option<TxnId>,
        terminal_is_durable: bool,
    ) {
        let public_ids = aliases
            .iter()
            .filter_map(|(public, effective)| (*effective == effective_txn_id).then_some(*public))
            .collect::<Vec<_>>();
        for public_id in public_ids {
            if let Some(successor) = successor {
                aliases.insert(public_id, successor);
            } else if !terminal_is_durable {
                aliases.remove(&public_id);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn set_wave_tail_handoff_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *wave_tail_handoff_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    pub(crate) fn set_wave_tail_failure_publish_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *wave_tail_failure_publish_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    /// Whether `text` is an UPDATE or DELETE on an existing base table that the **concurrent**
    /// commit path can execute via off-lock prepare plus the short commit critical section
    /// (write-half MVCC, Stage 4). INSERT intentionally always returns `false`: every INSERT
    /// ingress, including the public concurrent facade, enters the typed transaction overlay
    /// rather than advertising eligibility for the displaced wave path. Other commands return
    /// `false` as well. Conservative by construction: it never returns `true` for a statement
    /// the concurrent path cannot faithfully execute.
    pub fn is_concurrent_dml(&self, text: &str) -> bool {
        let Ok(cmd) = parse_command(text) else {
            return false;
        };
        self.is_concurrent_dml_command(&cmd)
    }

    pub(crate) fn is_concurrent_dml_command(&self, cmd: &Command) -> bool {
        let table_name = match &cmd {
            Command::Update(update) => &update.table,
            Command::Delete(delete) => &delete.table,
            _ => return false,
        };
        // Lock-free concurrent-DML classify (Stage 2 — blocker #1): probe the pinned catalog snapshot.
        let catalog = self.catalog_snapshot();
        let Some(_table) = catalog.relational_catalog.get(table_name) else {
            return false;
        };
        true
    }

    /// Register an autocommit statement's scalar read boundary for the oldest-active
    /// GC/ledger-prune boundary, returning a guard that deregisters on drop (write-half MVCC,
    /// Stage 4). Explicit transactions use the generation-owned keyed registration below.
    pub(crate) fn register_active_snapshot(&self, snapshot: Index) -> ActiveSnapshotGuard<'_> {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .register(snapshot);
        ActiveSnapshotGuard {
            engine: self,
            snapshot,
        }
    }

    pub(crate) fn deregister_active_snapshot(&self, snapshot: Index) {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .deregister(snapshot);
    }

    /// Capture the exact catalog, MVCC table generations, and resident GPU resource descriptors at
    /// `boundary`. The caller holds `commit`, so no DML/DDL publisher can advance the logical state
    /// while the bundle is assembled. The descriptor publisher lock makes the single-buffer and
    /// shard map loads one residency publication observation; both descriptor kinds own the device
    /// allocations they describe. Cached device indexes are lifetime-pinned as optional accelerators
    /// and remain subject to their existing buffer-identity validation before execution.
    pub(crate) fn capture_transaction_snapshot(
        &self,
        boundary: Index,
        characteristics: TransactionCharacteristics,
    ) -> Arc<TransactionSnapshot> {
        self.capture_read_snapshot(boundary, false, characteristics)
    }

    /// Register a transaction's GC boundary without retaining its device/index generation before
    /// the first data or catalog statement. The first statement replaces this shell with one full
    /// pinned snapshot under the existing statement-refresh protocol, so a `BEGIN` followed by
    /// `SET TRANSACTION`, rollback, or an idle connection cannot pay or hold the full GPU bundle.
    fn capture_transaction_context_shell(
        &self,
        boundary: Index,
        characteristics: TransactionCharacteristics,
    ) -> Arc<TransactionSnapshot> {
        let snapshot = self.capture_read_snapshot(boundary, true, characteristics);
        snapshot
            .data_snapshot_acquired
            .store(false, AtomicOrdering::Release);
        snapshot
    }

    /// Capture an autocommit statement generation. It has the same descriptor ownership guarantees
    /// as an explicit transaction but need not retain global index resources beyond the one read.
    pub(crate) fn capture_statement_snapshot(&self, boundary: Index) -> Arc<TransactionSnapshot> {
        self.capture_read_snapshot(
            boundary,
            true,
            TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
        )
    }

    fn capture_read_snapshot(
        &self,
        boundary: Index,
        statement_owned: bool,
        characteristics: TransactionCharacteristics,
    ) -> Arc<TransactionSnapshot> {
        let catalog = self.read_state.catalog_as_of(boundary);
        let table_versions = self.read_state.mvcc.capture_table_versions();
        let retained_gpu_account = Arc::clone(&self.transaction_retained_gpu_allocations);
        // Explicit snapshot capture and budget accounting both acquire lifetime-registry first.
        // Replacement/purge publishers do not need this lock, but no budget observer can pass it
        // between our descriptor/cache clone and exact allocation registration below.
        let mut retained_gpu_tracked = (!statement_owned).then(|| {
            retained_gpu_account
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        let (
            resident_snapshots,
            resident_shards,
            device_authoritative_tables,
            chunk_authoritative_tables,
            streaming_cold_chunks,
        ) = {
            let _publish = self
                .read_state
                .residency
                .descriptor_publish_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                self.read_state.residency.snapshots.load_full(),
                self.read_state.residency.shards.load_full(),
                self.read_state
                    .residency
                    .device_authoritative_tables
                    .load_full(),
                self.read_state
                    .residency
                    .chunk_authoritative_tables
                    .load_full(),
                self.read_state.residency.streaming_cold_chunks.load_full(),
            )
        };
        // Explicit transactions may execute many later statements, so retain every index allocation
        // that was valid at BEGIN. A statement-owned snapshot only needs the base descriptors here:
        // the one executing read clones its validated index Arc into the submission/job before launch.
        // Walking four global index maps on every autocommit point read added a fixed latency tax
        // without extending that already-bounded job lifetime.
        let mut resident_index_resources = Vec::new();
        if !statement_owned {
            resident_index_resources.extend(
                self.read_state
                    .residency
                    .wave_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .filter_map(|index| index.index_memory.as_ref().cloned()),
            );
            resident_index_resources.extend(
                self.read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .filter_map(|index| index.device_index.as_ref().cloned()),
            );
            resident_index_resources.extend(
                self.read_state
                    .residency
                    .chunk_key_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .map(|index| Arc::clone(&index.device)),
            );
            resident_index_resources.extend(
                self.read_state
                    .residency
                    .chunk_key_bloom
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .map(|bloom| Arc::clone(&bloom.device)),
            );
        }
        let mut resident_gpu_resources = resident_index_resources.clone();
        if !statement_owned {
            resident_gpu_resources.extend(
                resident_snapshots
                    .values()
                    .filter_map(|entry| entry.device_memory.as_ref().cloned()),
            );
            resident_gpu_resources.extend(resident_shards.values().flatten().flat_map(|shard| {
                [
                    shard.device_memory.as_ref(),
                    shard.deleted_by_region.as_ref(),
                    shard.created_by_region.as_ref(),
                    shard.row_id_region.as_ref(),
                ]
                .into_iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
            }));
        }
        let resident_gpu_charge = if let Some(tracked) = retained_gpu_tracked.as_deref_mut() {
            Arc::new(TransactionRetainedGpuCharge::register_locked(
                Arc::clone(&retained_gpu_account),
                &resident_gpu_resources,
                tracked,
            ))
        } else {
            Arc::new(TransactionRetainedGpuCharge::empty(Arc::clone(
                &retained_gpu_account,
            )))
        };
        drop(retained_gpu_tracked);
        Arc::new(TransactionSnapshot {
            characteristics,
            boundary,
            next_row_id: self.read_state.mvcc.current_row_id(),
            catalog,
            table_versions,
            resident_snapshots,
            resident_shards: Arc::clone(&resident_shards),
            device_authoritative_tables,
            chunk_authoritative_tables,
            delta: Arc::new(std::sync::Mutex::new(TransactionDeltaState {
                generation: 0,
                resident_shards: Arc::clone(&resident_shards),
                resident_shards_authority: Arc::clone(&resident_shards),
                streaming_cold_chunks: Arc::clone(&streaming_cold_chunks),
                streaming_cold_chunks_authority: Arc::clone(&streaming_cold_chunks),
                operations: Vec::new(),
                write_set: WriteSet::default(),
                next_row_id: self.read_state.mvcc.current_row_id(),
                sequence_state: BTreeMap::new(),
                sequence_state_by_oid: BTreeMap::new(),
                sequence_value_references: Vec::new(),
                catalog_base: None,
                catalog_overlay: None,
                private_gpu_bytes_by_gpu: BTreeMap::new(),
                commit_gpu_bytes_by_gpu: BTreeMap::new(),
            })),
            table_access: self.table_access.lease(),
            rewrite_fenced_tables: Arc::new(Mutex::new(BTreeSet::new())),
            statement_lock: Arc::new(std::sync::Mutex::new(())),
            program_owned: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            data_snapshot_acquired: Arc::new(std::sync::atomic::AtomicBool::new(statement_owned)),
            base_streaming_cold_chunks: streaming_cold_chunks,
            private_gpu_account: Arc::clone(&self.transaction_private_gpu_bytes),
            _resident_index_resources: resident_index_resources,
            _resident_gpu_charge: resident_gpu_charge,
        })
    }

    /// Open an explicit transaction and pin its one lifetime generation bundle: visibility boundary,
    /// catalog, table versions, and resident GPU resource descriptors. Its scalar boundary remains
    /// in the same space used by conflict-ledger pruning and MVCC GC. Lock order is `commit` ->
    /// residency publishers/caches -> `active_snapshots`.
    pub(crate) fn begin_transaction_context(
        &self,
        txn_id: TxnId,
        characteristics: TransactionCharacteristics,
    ) -> Result<(), ExecuteError> {
        let characteristics = validate_transaction_characteristics(characteristics)?;
        self.begin_transaction_context_with_owner(txn_id, false, characteristics, None)
            .map_err(ExecuteError::Txn)
    }

    pub(crate) fn begin_predeclared_transaction_context(
        &self,
        txn_id: TxnId,
        characteristics: TransactionCharacteristics,
    ) -> Result<(), ExecuteError> {
        let characteristics = validate_transaction_characteristics(characteristics)?;
        self.begin_transaction_context_with_owner(txn_id, true, characteristics, None)
            .map_err(ExecuteError::Txn)
    }

    pub(crate) fn begin_claimed_transaction_context(
        &self,
        txn_id: TxnId,
        characteristics: TransactionCharacteristics,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<(), ExecuteError> {
        let characteristics = validate_transaction_characteristics(characteristics)?;
        self.begin_transaction_context_with_owner(
            txn_id,
            false,
            characteristics,
            Some(request_digest),
        )
        .map_err(ExecuteError::Txn)
    }

    fn begin_transaction_context_with_owner(
        &self,
        txn_id: TxnId,
        program_owned: bool,
        characteristics: TransactionCharacteristics,
        allowed_pending: Option<gpu_db_wal::CanonicalDigest>,
    ) -> Result<(), TxnError> {
        let mut commit = self.commit_state();
        let pending = self
            .pending_transaction_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pending_rejected = match allowed_pending {
            Some(expected) => pending.get(&txn_id) != Some(&expected),
            None => pending.contains_key(&txn_id),
        };
        if pending_rejected {
            return Err(TxnError::AlreadyExists(txn_id));
        }
        drop(pending);
        let effective_txn_id = if commit.transaction_status.contains_key(&txn_id) {
            // A typed sequence DEFAULT commits a durable child before its parent DML terminal.
            // Compatibility callers may later use that child number as their own explicit
            // transaction ID. Preserve the public identity by assigning a private surrogate;
            // ordinary terminal/retry identities still fail closed as before.
            let sequence_child_claim = self
                .sequence_value_outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&txn_id)
                .is_some_and(|applied| applied.record.parent_txn_id != txn_id);
            if !sequence_child_claim || allowed_pending.is_some() {
                return Err(TxnError::AlreadyExists(txn_id));
            }
            self.begin_unclaimed_transaction(&mut commit)?
        } else {
            txn_id
        };
        #[cfg(feature = "probe-timing")]
        let probe_snapshot_capture_started = std::time::Instant::now();
        let snapshot =
            self.capture_transaction_context_shell(self.committed_seq(), characteristics);
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_transaction_begin_snapshot_capture_nanos(
            probe_snapshot_capture_started.elapsed().as_nanos() as u64,
        );
        snapshot
            .program_owned
            .store(program_owned, AtomicOrdering::Release);
        let active_txn_id = if effective_txn_id == txn_id {
            commit.txn_manager.begin_with_id(txn_id)?.id
        } else {
            // `begin_unclaimed_transaction` registered the surrogate while the same commit lock
            // was held; use that exact identity for the active snapshot below.
            effective_txn_id
        };
        if effective_txn_id == txn_id {
            self.active_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .register_transaction(active_txn_id, snapshot);
        } else {
            let mut aliases = self
                .public_transaction_aliases
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut active = self
                .active_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            active.register_transaction(active_txn_id, snapshot);
            aliases.insert(txn_id, active_txn_id);
        }
        Ok(())
    }

    /// Finish an explicit transaction, releasing its lifetime snapshot. `AND CHAIN` first allocates
    /// and registers the successor while the same locks are held, so exhaustion is a pre-effect
    /// error and there is no un-fenced gap between the two transaction contexts.
    pub(crate) fn finish_transaction_context(
        &self,
        txn_id: TxnId,
        committed: bool,
        chain: bool,
    ) -> Result<Option<TxnId>, TxnError> {
        let mut commit = self.commit_state();
        let characteristics = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .transaction_snapshot_handle(txn_id)
            .map_or(
                TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
                |snapshot| snapshot.characteristics,
            );
        let successor_id = if chain {
            Some(self.begin_unclaimed_transaction(&mut commit)?)
        } else {
            None
        };
        let terminal = if committed {
            commit.txn_manager.commit(txn_id)
        } else {
            commit.txn_manager.rollback(txn_id)
        };
        if let Err(error) = terminal {
            Self::cancel_chained_successor(&mut commit, successor_id);
            return Err(error);
        };
        let successor = successor_id.map(|next_id| {
            (
                next_id,
                self.capture_transaction_snapshot(self.committed_seq(), characteristics),
            )
        });
        let mut aliases = self
            .public_transaction_aliases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut active = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = active.deregister_transaction(txn_id);
        debug_assert!(
            removed.is_some(),
            "active transaction {txn_id} had no lifetime snapshot"
        );
        let successor_id = successor.as_ref().map(|(next_id, _)| *next_id);
        if let Some((next_id, next_snapshot)) = successor {
            active.register_transaction(next_id, next_snapshot);
        }
        Self::complete_public_transaction_alias_in_map(
            &mut aliases,
            txn_id,
            successor_id,
            committed,
        );
        drop(active);
        drop(aliases);
        self.gc_transaction_created_by_regions();
        Ok(successor_id)
    }

    /// Drop an internal autocommit transaction that failed before durability while leaving its
    /// caller-owned stable ID reusable. This is deliberately unavailable to user BEGIN/ROLLBACK.
    pub(crate) fn cancel_internal_transaction_context(
        &self,
        txn_id: TxnId,
    ) -> Result<(), TxnError> {
        let mut commit = self.commit_state();
        commit.txn_manager.cancel(txn_id)?;
        let removed = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .deregister_transaction(txn_id);
        debug_assert!(removed.is_some(), "cancelled transaction lost its snapshot");
        drop(commit);
        self.gc_transaction_created_by_regions();
        Ok(())
    }

    /// Off-lock prepare dispatch for the generic UPDATE/DELETE wave. INSERT admission is consumed
    /// by the transaction-overlay codec-5 route before this boundary.
    pub(crate) fn prepare_dml(
        &self,
        cmd: &Command,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, ExecuteError> {
        let delta = match cmd {
            Command::Insert(_) => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "generic concurrent wave cannot admit INSERT".to_string(),
                )))
            }
            Command::Update(update) => self.prepare_update(update, snapshot),
            Command::Delete(delete) => self.prepare_delete(delete, snapshot),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "execute_dml_concurrent received a non-DML command".to_string(),
                )));
            }
        }?;
        Ok(delta)
    }

    /// Enqueue one prepared concurrent DML commit into the deterministic commit WAVE and block
    /// until a sequencer completes it (ledger #6 / ADR-009 host spine — "the order IS the log").
    ///
    /// Committers no longer take the commit_mutex per statement. They push a fully-prepared wave
    /// item and either (a) become the SEQUENCER — the one thread that drains everything queued as
    /// a WAVE and commits it under a single commit_mutex hold — or (b) wait on the shared condvar
    /// until a sequencer completes their item. Any waiter can be PROMOTED to sequencer when the
    /// active one steps down (after its own item completes), so queued items are never
    /// leaderless. Wave order = queue arrival order = WAL/log order = apply order; within a wave
    /// there are NO ordering aborts (each item's re-resolve sees every earlier item's applied
    /// delta) — aborts remain only for genuine SI conflicts against state committed after an
    /// item's read snapshot, and for constraint phantoms at its commit seq (both retryable,
    /// pre-durable, side-effect-free).
    ///
    /// vs. the previous per-commit critical section: one mutex hand-off + one group-fsync wait +
    /// one publish PER WAVE instead of per commit, and the apply loop runs back-to-back on one
    /// core (the sequencer) instead of bouncing the MVCC structures across every writer's cache.
    #[allow(clippy::too_many_arguments)]
    fn commit_dml_concurrent(
        &self,
        txn_id: u64,
        cmd: Command,
        request: CanonicalRequest,
        write_set: WriteSet,
        read_snapshot: Index,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
        offlock_prepared: Option<OfflockPreparedDml>,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        let item = CommitWaveItem {
            txn_id,
            cmd,
            request,
            expected_catalog_version,
            offlock_prepared,
            write_set,
            read_snapshot,
            outcome: Arc::new(CommitWaveDone::default()),
        };
        let outcome = self.enqueue_commit_wave_item(item)?;
        // Blocking client: become the sequencer or spin/park on our own outcome, running the
        // pipeline's pending tails as a fallback claimer (the classic per-statement blocking arm).
        // Rows affected and RETURNING are carried through the common result surface.
        let rows_affected = if let Some(result) = self.pump_as_sequencer_if_idle(&outcome) {
            result?
        } else {
            self.await_commit_wave_outcome(&outcome)?
        };
        Ok(DmlExecutionResult {
            rows_affected,
            returning: outcome.take_returning(),
        })
    }

    #[cfg(test)]
    /// Construct an ordinary classic queue item for queue/wedge/durability failure tests. This
    /// fixture cannot carry a prepared INSERT delta or a pre-encoded INSERT WAL record.
    pub(crate) fn make_test_commit_wave_item(
        &self,
        txn_id: u64,
        cmd: Command,
        request: CanonicalRequest,
        write_set: WriteSet,
        read_snapshot: Index,
    ) -> CommitWaveItem {
        CommitWaveItem {
            txn_id,
            cmd,
            request,
            expected_catalog_version: None,
            offlock_prepared: None,
            write_set,
            read_snapshot,
            outcome: Arc::new(CommitWaveDone::default()),
        }
    }

    #[cfg(test)]
    /// Enqueue a classic fixture without blocking so queue and failure tests can observe its
    /// completion slot. Production callers enter through `commit_dml_concurrent`.
    pub(crate) fn submit_test_commit_wave_item(
        &self,
        item: CommitWaveItem,
    ) -> Result<CommitWaveOutcome, ExecuteError> {
        self.enqueue_commit_wave_item(item)
    }

    /// Enqueue one already-built classic wave item. Returns its completion slot, or the wedge error
    /// if the concurrent path is wedged pending recovery.
    fn enqueue_commit_wave_item(
        &self,
        item: CommitWaveItem,
    ) -> Result<CommitWaveOutcome, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(execute_error_from_engine)?;
        let outcome = Arc::clone(&item.outcome);
        let mut queue = self.lock_commit_wave_queue();
        self.ensure_commit_path_available()
            .map_err(execute_error_from_engine)?;
        if let Some(failure) = &queue.wedged {
            return Err(failure.outcome_error());
        }
        queue.items.push_back(item);
        Ok(outcome)
    }

    /// E2.2(c) — the driver-multiplexed pipeline pump: advance the commit wave by ONE unit of work
    /// without blocking on any particular outcome. If no sequencer is active and work is queued,
    /// promote this thread to sequence a drain; otherwise claim a pending durability tail. Returns
    /// whether it did work. N event-loop drivers call this in their poll loops so M logical clients
    /// share a few OS threads with no per-commit park/wake — the disruptor ingress the mandate
    /// calls for. `take_if_done` on a submitted ticket observes the result.
    pub fn drive_commit_wave(&self) -> bool {
        if self.ensure_commit_path_available().is_err() {
            return false;
        }
        // In lanes mode a pump call first advances one preparation lane (round-robin), then the same
        // call can advance canonical classic-wave work. Neither strategy owns sequence/WAL state.
        let mut did_work = false;
        if let Some(lanes) = &self.intent_lanes {
            let lanes = std::sync::Arc::clone(lanes);
            self.maybe_resize_lanes(&lanes);
            let lane = (lanes
                .pump_cursor
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % lanes.lane_count as u64) as usize;
            did_work |= self.drive_intent_lane(lane);
        }
        // Prefer claiming a pending tail (cheap, unblocks acks) before taking sequencer duty.
        if self.try_finish_pending_wave_tail() {
            return true;
        }
        let promote = {
            let mut queue = match self.commit_wave.queue.try_lock() {
                Ok(queue) => queue,
                Err(_) => return did_work,
            };
            if !queue.sequencer_active && !queue.items.is_empty() && queue.wedged.is_none() {
                queue.sequencer_active = true;
                true
            } else {
                false
            }
        };
        if promote {
            // Drive one full drain cycle. `own_outcome` is a never-completing sentinel: the
            // sequencer steps down on an empty queue (or when this dummy is "done", which it never
            // is), so this returns after draining everything currently queued.
            let sentinel: CommitWaveOutcome = Arc::new(CommitWaveDone::default());
            self.run_commit_wave_sequencer(&sentinel);
            return true;
        }
        did_work
    }

    /// If no sequencer is running, promote THIS thread to drain+sequence waves until `own_outcome`
    /// completes, then return its result. Returns `None` if a sequencer is already active (the
    /// caller should wait) or if the promotion ran but our outcome is not yet set (a pipelined
    /// tail we did not claim — fall through to the waiter loop).
    fn pump_as_sequencer_if_idle(
        &self,
        own_outcome: &CommitWaveOutcome,
    ) -> Option<Result<u64, ExecuteError>> {
        let promote = {
            let mut queue = self.lock_commit_wave_queue();
            if !queue.sequencer_active {
                queue.sequencer_active = true;
                true
            } else {
                false
            }
        };
        if promote {
            self.run_commit_wave_sequencer(own_outcome);
            return own_outcome.take_if_done();
        }
        None
    }

    /// Block until `own_outcome` completes: spin (claiming pending tails + promoting a leaderless
    /// queue), then fall back to the condvar. The tail of the classic blocking commit path.
    fn await_commit_wave_outcome(
        &self,
        own_outcome: &CommitWaveOutcome,
    ) -> Result<u64, ExecuteError> {
        let outcome: &CommitWaveOutcome = own_outcome;
        loop {
            // Spin first: under load a wave completes within tens of µs, far cheaper to poll than
            // to pay a futex sleep + wake per commit. The periodic promotion probe keeps queued
            // items from ever being leaderless (the previous sequencer may have stepped down
            // between our enqueue and our first probe). W2: the same probe CLAIMS the pipeline's
            // pending durability tail — a waiter whose own wave was pipelined is the natural
            // thread to run its fsync-wait + publish + acks (including its own).
            for spin in 0..4096_u32 {
                if let Some(result) = outcome.take_if_done() {
                    return result;
                }
                if spin % 64 == 63 {
                    self.try_finish_pending_wave_tail();
                    if let Some(result) = outcome.take_if_done() {
                        return result;
                    }
                    if let Ok(mut queue) = self.commit_wave.queue.try_lock() {
                        if !queue.sequencer_active && !queue.items.is_empty() {
                            queue.sequencer_active = true;
                            drop(queue);
                            self.run_commit_wave_sequencer(outcome);
                        }
                    }
                }
                std::hint::spin_loop();
            }
            // Fall back to the condvar (arrival lulls / oversubscribed cores).
            let mut queue = self.lock_commit_wave_queue();
            loop {
                if let Some(result) = outcome.take_if_done() {
                    return result;
                }
                // W2: claim the pending tail before sleeping — the sequencer's hand-off
                // notify_all wakes this loop, and the claim may complete our own outcome.
                drop(queue);
                if self.try_finish_pending_wave_tail() {
                    queue = self.lock_commit_wave_queue();
                    continue;
                }
                queue = self.lock_commit_wave_queue();
                if let Some(result) = outcome.take_if_done() {
                    return result;
                }
                if !queue.sequencer_active {
                    if queue.items.is_empty() {
                        break;
                    }
                    queue.sequencer_active = true;
                    drop(queue);
                    self.run_commit_wave_sequencer(outcome);
                    break;
                }
                queue = self
                    .commit_wave
                    .cv
                    .wait(queue)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    fn lock_commit_wave_queue(&self) -> std::sync::MutexGuard<'_, CommitWaveQueue> {
        self.commit_wave
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(test)]
    pub(crate) fn notify_simulated_wave_tail_change(&self) {
        let queue = self.lock_commit_wave_queue();
        self.commit_wave.cv.notify_all();
        drop(queue);
    }

    /// The promoted sequencer: drain-and-commit WAVES until this thread's own item is done (then
    /// hand leadership off) or the queue is empty. Wave size self-tunes to the arrival rate —
    /// everything queued while the previous wave was committing forms the next wave (bounded by
    /// `COMMIT_WAVE_MAX` per drain to keep worst-case wave latency bounded).
    ///
    /// W2/W2b — the durability pipeline: a sequenced wave's tail (fsync-wait → publish → acks)
    /// is handed to `pending_tails` instead of running inline, so this loop's next iteration
    /// sequences wave N+1 while claimers (the waves' own blocked waiters, or this thread at the
    /// capacity gate) run the fdatasyncs. Consecutive waves' records coalesce into shared
    /// fsyncs via the WAL's group-flush protocol (the sequencer keeps appending while a flush
    /// is in flight). The capacity gate bounds applied-but-unpublished state at
    /// [`WAVE_TAIL_PIPELINE_DEPTH`] waves; post-publish auto_admit re-admissions stay HERE
    /// (behind a clean full-drain barrier) so a stale re-admission can never interleave with a
    /// later wave's apply (the internal form of ledger #26).
    fn run_commit_wave_sequencer(&self, own_outcome: &CommitWaveOutcome) {
        loop {
            let batch: Vec<CommitWaveItem> = {
                let mut queue = self.lock_commit_wave_queue();
                let n = queue.items.len().min(commit_wave_max());
                if n == 0 {
                    queue.sequencer_active = false;
                    drop(queue);
                    // Liveness on the empty-queue step-down: never leave pipelined tails
                    // behind with no sequencer (their member waiters would still claim them,
                    // but finishing inline here is strictly sooner and keeps the single-writer
                    // case's latency identical to the pre-pipeline path). Tails already claimed
                    // by an in-flight waiter finish on that waiter.
                    while self.try_finish_pending_wave_tail() {}
                    // W1b: idle moment — the cheap auto-checkpoint bound probe (see the wave
                    // counter probe above for the sustained-load path).
                    self.maybe_auto_checkpoint_wal();
                    self.commit_wave.cv.notify_all();
                    return;
                }
                queue.items.drain(..n).collect()
            };
            if let Err(error) = self.ensure_commit_path_available() {
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
                let mut queue = self.lock_commit_wave_queue();
                queue.sequencer_active = false;
                drop(queue);
                self.commit_wave.cv.notify_all();
                return;
            }
            let wave_started = std::time::Instant::now();
            let wave_len = batch.len() as u64;
            let tail = self.sequence_commit_wave(batch);
            #[cfg(test)]
            if tail.is_some() {
                let hook = wave_tail_handoff_hook()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                if let Some((engine, reached, resume)) = hook {
                    if engine == self as *const Self as usize {
                        reached.wait();
                        resume.wait();
                    }
                }
            }
            let waves = WAVE_STATS[0].fetch_add(1, AtomicOrdering::Relaxed);
            // W1b: periodic auto-checkpoint probe OFF the critical section (the rotation takes
            // the commit_mutex itself). Every 256 waves keeps the under-bound check (one mutex
            // lock + field read) off the per-wave path; sustained load can never starve
            // rotation, and the empty-queue step-down below covers bursty loads.
            if waves % 256 == 255 {
                self.maybe_auto_checkpoint_wal();
            }
            WAVE_STATS[1].fetch_add(wave_len, AtomicOrdering::Relaxed);
            WAVE_STATS[2].fetch_add(
                wave_started.elapsed().as_nanos() as u64,
                AtomicOrdering::Relaxed,
            );
            if let Some(tail) = tail {
                // W2b capacity gate: at most WAVE_TAIL_PIPELINE_DEPTH tails outstanding —
                // consecutive waves' records coalesce into shared fsyncs while the bound caps
                // applied-but-unpublished state. Claim tails ourselves when the pipe is full
                // (under load the shared fsync already covered them and the finishes are
                // instant).
                self.wait_wave_tail_capacity(wave_tail_pipeline_depth() - 1);
                let mut tail = Some(tail);
                let handed = {
                    let mut tails = self
                        .commit_wave
                        .pending_tails
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if self.is_commit_path_poisoned() {
                        false
                    } else {
                        tails.push_back(tail.take().expect("tail is handed once"));
                        true
                    }
                };
                if !handed {
                    drop(tail); // armed Drop fails every member outcome
                    self.commit_wave
                        .tails_finished
                        .fetch_add(1, AtomicOrdering::Release);
                    let mut queue = self.lock_commit_wave_queue();
                    queue.sequencer_active = false;
                    drop(queue);
                    self.commit_wave.cv.notify_all();
                    return;
                }
                self.commit_wave
                    .tails_handed
                    .fetch_add(1, AtomicOrdering::Release);
                {
                    // Wake spinners/sleepers so a member waiter can claim the tail while we
                    // drain + sequence the next wave.
                    let _queue = self.lock_commit_wave_queue();
                    self.commit_wave.cv.notify_all();
                }
            } else {
                let _queue = self.lock_commit_wave_queue();
                self.commit_wave.cv.notify_all();
            }
            if own_outcome.done.load(AtomicOrdering::Acquire) {
                // Step down so this client thread can return; a waiter with a still-queued item
                // (or the next arrival) promotes itself. A pending tail may remain — its member
                // waiters claim it (their outcomes are unset, so they are still in the waiter
                // loops by definition).
                let mut queue = self.lock_commit_wave_queue();
                queue.sequencer_active = false;
                drop(queue);
                self.commit_wave.cv.notify_all();
                return;
            }
        }
    }

    /// W2/W2b — the sequencer's pipeline gate: block until at most `max_outstanding` handed
    /// tails remain unfinished, claiming pending ones ourselves first (the common single-writer
    /// / idle-waiter case finishes them inline; under load member waiters usually already did
    /// during our sequencing). `0` = full drain (the admit barrier). Returns `true` when the
    /// capacity condition genuinely holds; `false` on the WEDGED early-return, where in-flight
    /// tails may still complete (and even publish, via the durable-frontier fast path) after
    /// this returns — callers needing a REAL barrier (the admit arm) must treat `false` as
    /// "barrier not established".
    pub(crate) fn wait_wave_tail_capacity(&self, max_outstanding: u64) -> bool {
        loop {
            let handed = self.commit_wave.tails_handed.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if handed.saturating_sub(finished) <= max_outstanding {
                return true;
            }
            if self.try_finish_pending_wave_tail() {
                continue;
            }
            // Claimers are mid-flight (took the tails, still fsyncing): wait for their signal.
            let queue = self.lock_commit_wave_queue();
            if queue.wedged.is_some() {
                // Defensive: a wedged path never completes tails normally; the completion
                // guard's counter bump is the primary unblock, this is the belt-and-braces.
                return false;
            }
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            let handed = self.commit_wave.tails_handed.load(AtomicOrdering::Acquire);
            if handed.saturating_sub(finished) <= max_outstanding {
                return true;
            }
            let _queue = self
                .commit_wave
                .cv
                .wait(queue)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// W2 — finish one pipelined wave tail: ONE group-durability wait covering every record the
    /// wave appended, then ONE `committed_seq` publish at the wave's last seq (CAS max), then the
    /// outcome acks. WAL-before-visibility holds wave-wide: nothing in the wave is visible until
    /// its highest record is fsync-covered. Runs on WHICHEVER thread claimed the tail (a member
    /// waiter or the sequencer); requires no locks beyond what `wait_group_durable` takes
    /// internally, and never the commit_mutex the sequencer may be holding for the next wave.
    fn finish_wave_tail(&self, mut tail: CommitWaveTail) {
        #[cfg(feature = "probe-timing")]
        let probe_has_insert = tail
            .batch
            .iter()
            .any(|item| matches!(&item.cmd, Command::Insert(_)));
        #[cfg(not(feature = "probe-timing"))]
        let probe_has_insert = false;
        // `TailCompletion` owns the post-WAL failure boundary.  In particular, its fixed serial
        // branch drains queues and tails before it publishes a waiter wakeup.
        let mut completion = TailCompletion::new(self);
        #[cfg(feature = "probe-timing")]
        let probe_durability_started = probe_has_insert.then(Instant::now);
        let durability_result = self.wait_group_durable(tail.last_position, probe_has_insert);
        #[cfg(feature = "probe-timing")]
        if let Some(started) = probe_durability_started {
            self.record_insert_probe_durability_wait_nanos(started.elapsed().as_nanos() as u64);
        }
        if let Err(err) = durability_result {
            // The wave's deltas are applied-but-unpublishable; the flush protocol has recorded
            // its sticky failure (and the flusher wedge-panicked if we were the flusher — in
            // that case this line is unreachable and the completion guard runs on unwind). Fail
            // the wave's outcomes and wedge the queue.
            if let EngineError::DurabilityFault(fault) = err {
                let fault = self.fail_all_pending_commit_work_fixed(fault);
                tail.set_fixed_durability_failure(fault);
                completion.mark_fixed_failure(fault);
            } else {
                let mut queue = self.lock_commit_wave_queue();
                queue
                    .wedged
                    .get_or_insert_with(|| CommitPathFailure::compatibility(err.to_string()));
            }
            drop(tail); // armed: fails every still-unset member outcome with the wedge error
            return; // completion guard (clean=false) wedges idempotently + counts + notifies
        }
        if self.is_commit_path_poisoned() {
            if let Some(fault) = self.group_flush.fixed_poison.snapshot() {
                tail.set_fixed_durability_failure(fault);
                completion.mark_fixed_failure(fault);
            }
            drop(tail); // armed: fail outcomes; a wedged service publishes no later visibility
            return;
        }
        #[cfg(feature = "probe-timing")]
        let probe_publication_started = probe_has_insert.then(Instant::now);
        let last_committed_seq = tail.committed.last().map(|(_, seq, _)| *seq);
        if let Err(error) =
            self.publish_ready_indices(tail.committed.iter().map(|(_, commit_seq, _)| *commit_seq))
        {
            self.wedge_commit_path();
            let mut queue = self.lock_commit_wave_queue();
            queue
                .wedged
                .get_or_insert_with(|| CommitPathFailure::compatibility(error.to_string()));
            drop(queue);
            drop(tail);
            return;
        }
        // A later wave may finish its physical durability tail before an earlier wave. Reporting
        // readiness does not itself mean this tail is publication-covered: the coordinator holds
        // it behind the gap. Drive any lower pending tail, or wait for its owner, before resolving
        // this wave's SQL outcomes.
        if let Some(last_committed_seq) = last_committed_seq {
            if let Err(error) = self.wait_until_publication_covers(last_committed_seq) {
                self.wedge_commit_path();
                let mut queue = self.lock_commit_wave_queue();
                queue
                    .wedged
                    .get_or_insert_with(|| CommitPathFailure::compatibility(error.to_string()));
                drop(queue);
                drop(tail);
                return;
            }
        }
        if let Some(last_committed_seq) = last_committed_seq {
            let mut commit = self.commit_state();
            commit.ledger.mark_published_through(last_committed_seq);
        }
        for (position, _seq, rows) in &tail.committed {
            self.metrics.inc_commit();
            tail.batch[*position].set_outcome(Ok(*rows));
        }
        #[cfg(feature = "probe-timing")]
        if let Some(started) = probe_publication_started {
            self.record_insert_probe_publication_status_ack_nanos(
                started.elapsed().as_nanos() as u64
            );
        }
        tail.armed = false;
        completion.mark_clean();
    }

    fn wait_until_publication_covers(&self, commit_seq: Index) -> Result<(), EngineError> {
        loop {
            if self.committed_seq() >= commit_seq {
                return Ok(());
            }
            self.ensure_commit_path_available()?;
            // Pending tails are ordered by sequencing handoff. Helping here closes the liveness
            // case where this thread claimed a later tail while the lower tail has no scheduled
            // owner at this instant.
            if self.try_finish_pending_wave_tail() {
                continue;
            }
            let queue = self.lock_commit_wave_queue();
            if let Some(error) = &queue.wedged {
                return Err(error.engine_error());
            }
            if self.committed_seq() >= commit_seq {
                return Ok(());
            }
            let (_queue, _) = self
                .commit_wave
                .cv
                .wait_timeout(queue, std::time::Duration::from_millis(1))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// W2 — claim and finish the pending pipeline tail if one is waiting. Called from the waiter
    /// loops' probes (the tail's own members are the natural claimers — they are blocked on its
    /// outcomes) and by the sequencer as the fallback claimer at the depth gate. Returns whether
    /// a tail was finished.
    pub(crate) fn try_finish_pending_wave_tail(&self) -> bool {
        if self.is_commit_path_poisoned() {
            return false;
        }
        // AUDIT d6d10f8e F: every waiter probes this every 64 spin iterations — pre-check the
        // counters (two atomic loads) so the common nothing-pending case never touches the
        // shared mutex.
        {
            let handed = self.commit_wave.tails_handed.load(AtomicOrdering::Acquire);
            let finished = self
                .commit_wave
                .tails_finished
                .load(AtomicOrdering::Acquire);
            if handed <= finished {
                return false;
            }
        }
        let taken = {
            let mut tails = self
                .commit_wave
                .pending_tails
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tails.pop_front()
        };
        match taken {
            Some(tail) => {
                self.finish_wave_tail(tail);
                true
            }
            None => false,
        }
    }

    fn wait_group_durable(
        &self,
        wal_position: usize,
        _probe_insert: bool,
    ) -> Result<(), EngineError> {
        // The engine owns exactly one logical durability group for every WAL backend. Exact FUA
        // owns physical frame/fence execution beneath this coordinator; it does not create a
        // second engine-side scheduler or alternate failure path.
        loop {
            if self
                .group_flush
                .durable_records
                .load(AtomicOrdering::Acquire)
                >= wal_position
            {
                return Ok(());
            }
            let mut coord = self
                .group_flush
                .coord
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(failure) = &coord.failed {
                return Err(failure.engine_error());
            }
            // Re-check under the coordination lock (a flusher may have finished in between).
            if self
                .group_flush
                .durable_records
                .load(AtomicOrdering::Acquire)
                >= wal_position
            {
                return Ok(());
            }
            if coord.flusher_active {
                // A flush is in flight; wait for its result and re-evaluate.
                let _coord = self
                    .group_flush
                    .cv
                    .wait(coord)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                continue;
            }
            // Become the flusher. Lock order: coord is RELEASED before the commit_mutex is taken
            // (a flusher marks itself active first so no second fsync stacks up behind it).
            coord.flusher_active = true;
            drop(coord);
            // Two-phase group flush: snapshot the unflushed tail under a BRIEF commit_mutex hold,
            // then run the write + fsync with NO lock held — this is what lets other committers
            // validate/append/apply (and queue into the NEXT group) while this group's disk IO is
            // in flight. Completion takes only the WAL core's own lock, never the commit_mutex.
            #[cfg(feature = "probe-timing")]
            let probe_begin_started = _probe_insert.then(Instant::now);
            let begun = {
                let mut commit = self.commit_state();
                commit.wal.begin_group_flush()
            };
            #[cfg(feature = "probe-timing")]
            if let Some(started) = probe_begin_started {
                self.record_insert_probe_durability_begin_group_flush_nanos(
                    started.elapsed().as_nanos() as u64,
                );
            }
            let flush_result = match begun {
                Ok(gpu_db_wal::WalGroupFlushBegin::Clean { flushed_records }) => {
                    Ok(flushed_records)
                }
                Ok(gpu_db_wal::WalGroupFlushBegin::Job(job)) => {
                    #[cfg(feature = "probe-timing")]
                    let probe_job_started = _probe_insert.then(Instant::now);
                    let result = job.commit();
                    #[cfg(feature = "probe-timing")]
                    if let Some(started) = probe_job_started {
                        self.record_insert_probe_durability_job_wait_nanos(
                            started.elapsed().as_nanos() as u64,
                        );
                    }
                    result
                }
                Ok(gpu_db_wal::WalGroupFlushBegin::Busy) => Ok(self
                    .group_flush
                    .durable_records
                    .load(AtomicOrdering::Acquire)),
                Err(err) => Err(err),
            };
            #[cfg(test)]
            let flush_result = durability_failure::inject_fixed_group_completion_failure_for_test(
                self,
                flush_result,
            );
            let mut coord = self
                .group_flush
                .coord
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            coord.flusher_active = false;
            match flush_result {
                Ok(flushed_records) => {
                    self.group_flush
                        .durable_records
                        .store(flushed_records, AtomicOrdering::Release);
                    self.group_flush.cv.notify_all();
                    // Loop: our own record was appended before this flush began, so the frontier
                    // now covers it (or a concurrent truncate shrank nothing below it — see the
                    // rollback paths, which only ever drop UNFLUSHED records they own).
                }
                Err(EngineError::DurabilityFault(fault)) => {
                    let fault = self.group_flush.fixed_poison.install(fault);
                    let selected = match coord
                        .failed
                        .as_ref()
                        .and_then(CommitPathFailure::fixed_fault)
                    {
                        Some(existing) => existing,
                        None => {
                            coord.failed = Some(CommitPathFailure::Fixed(fault));
                            fault
                        }
                    };
                    // Publish the engine-wide admission gate before waking a group waiter.  A
                    // waiter that observes this completion must not claim new work while the
                    // owning tail is still draining already-applied members.
                    self.commit_path_wedged.store(true, AtomicOrdering::Release);
                    self.group_flush.cv.notify_all();
                    return Err(EngineError::DurabilityFault(selected));
                }
                Err(err) => {
                    coord.failed = Some(CommitPathFailure::compatibility(err.to_string()));
                    self.group_flush.cv.notify_all();
                    drop(coord);
                    // Wedge-don't-serve-torn-state: poison the commit_mutex so the façade refuses
                    // further statements (see the doc comment above for why no clean abort exists).
                    let _commit = self.commit_state();
                    panic!(
                        "group-commit fsync failed after member deltas were applied: {err} — \
                         wedging the commit path; restart recovery replays the durable WAL prefix"
                    );
                }
            }
        }
    }

    pub fn execute_text(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.execute_text_at_timestamp_micros(txn_id, text, timestamp_micros)
    }

    pub fn execute_text_at_timestamp_micros(
        &self,
        txn_id: u64,
        text: &str,
        timestamp_micros: u64,
    ) -> Result<(), ExecuteError> {
        let txn_id = self
            .resolve_or_allocate_public_transaction_id(txn_id)
            .map_err(ExecuteError::Txn)?;
        let command = parse_command(text)?;
        if matches!(&command, Command::Insert(_)) {
            // The public compatibility API must not retain a serialized INSERT authority. An
            // explicit transaction stages in its existing private generation; autocommit takes
            // the same claimed overlay used by mutation admission and preserves the caller's
            // supplied commit timestamp through its one canonical terminal.
            self.observe_transaction_id(txn_id);
            self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
            self.legacy_lane_history_write_guard()
                .map_err(ExecuteError::Engine)?;
            if self.transaction_snapshot_handle(txn_id).is_some() {
                return self
                    .execute_prepared_dml_in_transaction_with_result(txn_id, command, None, false)
                    .map(|_| ());
            }
            let (requests_published_sequence_default, route_catalog_version) =
                self.insert_sequence_default_route(&command);
            if requests_published_sequence_default {
                return self
                    .execute_sequence_default_autocommit_at_timestamp_micros(
                        txn_id,
                        command,
                        route_catalog_version.map(
                            crate::engine_mutation_admission::CatalogVersionExpectation::SequenceRoute,
                        ),
                        timestamp_micros,
                    )
                    .map(|_| ());
            }
            return self
                .execute_autocommit_insert_as_one_statement_overlay(
                    txn_id,
                    command,
                    CanonicalRequest::from_text(self, text).digest(),
                    None,
                    timestamp_micros,
                    || {},
                )
                .map(|_| ());
        }
        self.execute_parsed_text_at_timestamp_micros(txn_id, command, text, timestamp_micros, None)
    }

    pub(crate) fn execute_parsed_text(
        &self,
        txn_id: u64,
        command: Command,
        text: &str,
    ) -> Result<(), ExecuteError> {
        self.execute_parsed_text_with_catalog(txn_id, command, text, None)
    }

    pub(crate) fn execute_parsed_text_with_catalog(
        &self,
        txn_id: u64,
        command: Command,
        text: &str,
        expected_catalog_version: Option<Index>,
    ) -> Result<(), ExecuteError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.execute_parsed_text_at_timestamp_micros(
            txn_id,
            command,
            text,
            timestamp_micros,
            expected_catalog_version,
        )
    }

    fn execute_parsed_text_at_timestamp_micros(
        &self,
        txn_id: u64,
        command: Command,
        text: &str,
        timestamp_micros: u64,
        expected_catalog_version: Option<Index>,
    ) -> Result<(), ExecuteError> {
        // Compatibility callers supply their own canonical identity rather than borrowing the
        // facade allocator. Observe it before parsing/execution can publish an engine-owned
        // sequence transition, so the two claimants cannot alias.
        self.observe_transaction_id(txn_id);
        self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
        let result = self.execute_parsed_text_at_timestamp_micros_inner(
            txn_id,
            command,
            text,
            timestamp_micros,
            expected_catalog_version,
        );
        // VACUUM #5: run a commit-parked auto-vacuum now — the commit lock + catalog latch are
        // released, so `vacuum_table` can take them fresh (the in-commit trigger would deadlock).
        let parked = self
            .read_state
            .residency
            .pending_auto_vacuum
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(table) = parked {
            // Audit F1: the statement is already durable + published — a MAINTENANCE failure
            // must not turn its result into an error (a client retry would double-apply). The
            // scan still serves dead slots correctly; the churn counter was NOT reset, so the
            // trigger re-arms on the table's next tombstone. Count it and move on.
            if self.vacuum_table(&table).is_err() {
                self.read_state
                    .residency
                    .auto_vacuum_failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // W1b: the serialized path's maintenance point — commit lock + catalog latch are
        // released here, so the rotation can take them fresh (cheap under the size bound).
        self.maybe_auto_checkpoint_wal();
        result
    }

    fn execute_parsed_text_at_timestamp_micros_inner(
        &self,
        txn_id: u64,
        cmd: Command,
        text: &str,
        timestamp_micros: u64,
        expected_catalog_version: Option<Index>,
    ) -> Result<(), ExecuteError> {
        if self.is_commit_path_poisoned() {
            return Err(execute_error_from_engine(
                self.commit_path_unavailable_error(),
            ));
        }
        if command_has_returning(&cmd) {
            return Err(discarded_returning_error());
        }
        if let Command::TruncateTable(truncate) = &cmd {
            if self.transaction_snapshot_handle(txn_id).is_some() {
                self.execute_truncate_in_transaction(
                    txn_id,
                    truncate.clone(),
                    expected_catalog_version,
                )?;
            } else {
                self.execute_truncate_autocommit(
                    txn_id,
                    truncate.clone(),
                    expected_catalog_version,
                )?;
            }
            return Ok(());
        }
        if let Some(snapshot) = self.transaction_snapshot_handle(txn_id) {
            self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
            if matches!(
                &cmd,
                Command::Insert(_) | Command::Update(_) | Command::Delete(_)
            ) {
                return self
                    .execute_parsed_dml_in_transaction_with_result(txn_id, cmd)
                    .map(|_| ());
            }
            if matches!(&cmd, Command::CreateTable(_) | Command::CreateIndex(_)) {
                return self.execute_catalog_in_transaction(
                    txn_id,
                    cmd,
                    std::sync::Arc::<str>::from(text),
                    expected_catalog_version,
                );
            }
            if !matches!(
                &cmd,
                Command::Begin { .. } | Command::Commit { .. } | Command::Rollback { .. }
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "command is not supported inside an active transaction; it was not executed"
                        .to_string(),
                )));
            }
        }
        // `execute_text` and mutation admission divert every live autocommit INSERT to the
        // claimed typed-overlay terminal before reaching this compatibility dispatcher. Keep
        // that boundary explicit: accepting it here would recreate the retired serialized
        // INSERT claimant below, while explicit transactions above still use the typed overlay.
        if matches!(&cmd, Command::Insert(_)) {
            return Err(ExecuteError::Unsupported(
                "direct parsed autocommit INSERT must enter typed transaction admission"
                    .to_string(),
            ));
        }
        let canonical_payload = Self::command_claims_canonical_mutation(&cmd)
            .then(|| std::sync::Arc::<[u8]>::from(text.as_bytes()));
        // Resolve stable request identity before claiming any new relation. Exact retries have no
        // new table access and must remain idempotent while a later reset retains exclusivity.
        // Fresh work retains the claimant through reverse-gather, apply, and publication.
        let _table_access = if let Some(payload) = canonical_payload.as_ref() {
            let request_digest = gpu_db_wal::canonical_request_digest(payload);
            match self.acquire_autocommit_command_table_access_after_retry(
                &cmd,
                txn_id,
                request_digest,
            )? {
                StableRetryOr::Terminal(_) => return Ok(()),
                StableRetryOr::Fresh(access) => access,
            }
        } else {
            self.acquire_autocommit_command_table_access(&cmd)?
        };
        // RETIRE-002 boundary: representation-changing commands reverse-gather every
        // device-authoritative table before DDL repair/validation. Normal DML and SELECT never
        // enter this sweep, and no device decline dispatches here.
        let representation_neutral = matches!(
            &cmd,
            Command::Update(_)
                | Command::Delete(_)
                | Command::Select(_)
                | Command::SelectLiteral(_)
                | Command::CreateTable(_)
                | Command::Begin { .. }
                | Command::Commit { .. }
                | Command::Rollback { .. }
        );
        if !representation_neutral {
            let elided: Vec<String> = self
                .read_state
                .residency
                .device_authoritative_tables
                .load()
                .iter()
                .cloned()
                .collect();
            for table_name in elided {
                self.rehydrate_elided_serialized(&table_name)
                    .map_err(ExecuteError::Engine)?;
            }
        }
        // P4-2b (S-E.P4, design review H2): the SAME sweep for CHUNK-AUTHORITATIVE tables — a
        // DDL preflight (ADD PK/UNIQUE/CHECK/FK) reading visible_relational_rows against a
        // FROZEN store would validate vacuously; de-authoritize every class table first.
        if !representation_neutral {
            let class_tables: Vec<String> = self
                .read_state
                .residency
                .chunk_authoritative_tables
                .load()
                .keys()
                .cloned()
                .collect();
            for table_name in class_tables {
                self.deauthoritize_chunk_table(&table_name, false)
                    .map_err(ExecuteError::Engine)?;
            }
        }

        match cmd {
            // The pre-dispatch guard above rejects live autocommit INSERT. This arm only makes
            // that invariant explicit to the compiler; it has no encoder, apply, or publication
            // behavior.
            Command::Insert(_) => unreachable!("autocommit INSERT was rejected before fallback"),
            Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::CreateSchema(_)
            | Command::DropSchema(_)
            | Command::CreateDatabase(_)
            | Command::DropDatabase(_)
            | Command::RenameDatabase(_)
            | Command::CreateTablespace(_)
            | Command::DropTablespace(_)
            | Command::RenameTablespace(_)
            | Command::CreateTable(_)
            | Command::AddPrimaryKey(_)
            | Command::AddUniqueConstraint(_)
            | Command::AddCheckConstraint(_)
            | Command::AddForeignKey(_)
            | Command::AddColumn(_)
            | Command::RenameTable(_)
            | Command::RenameColumn(_)
            | Command::RenameConstraint(_)
            | Command::DropColumn(_)
            | Command::DropConstraint(_)
            | Command::CreateIndex(_)
            | Command::RenameIndex(_)
            | Command::CreateView(_)
            | Command::RenameView(_)
            | Command::CreateMaterializedView(_)
            | Command::RefreshMaterializedView(_)
            | Command::RenameMaterializedView(_)
            | Command::CreateFunction(_)
            | Command::RenameFunction(_)
            | Command::DropFunction(_)
            | Command::CreateSequence(_)
            | Command::CreateDomain(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceSetVal(_)
            | Command::SequenceRestart(_)
            | Command::RenameSequence(_)
            | Command::DropTable(_)
            | Command::TruncateTable(_)
            | Command::DropIndex(_)
            | Command::DropView(_)
            | Command::DropMaterializedView(_)
            | Command::DropSequence(_)
            | Command::DropDomain(_)
            | Command::GrantTable(_)
            | Command::RevokeTable(_)
            | Command::GrantSchema(_)
            | Command::RevokeSchema(_)
            | Command::GrantDatabase(_)
            | Command::RevokeDatabase(_)
            | Command::GrantTablespace(_)
            | Command::RevokeTablespace(_)
            | Command::GrantFunction(_)
            | Command::RevokeFunction(_)
            | Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_)
            | Command::CreateRole(_)
            | Command::DropRole(_)
            | Command::RenameRole(_)
            | Command::AlterRoleLogin(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_)
            | Command::AlterColumnDefault(_)
            | Command::CommentOn(_)
            | Command::Delete(_)
            | Command::Update(_) => {
                self.preflight_unique_index_constraints(&cmd, txn_id)?;
                match self.route_command(&cmd) {
                    RouteDecision::Gpu(_) | RouteDecision::Cpu => {
                        self.commit_mutation_at_with_catalog_table_access(
                            txn_id,
                            std::sync::Arc::clone(
                                canonical_payload
                                    .as_ref()
                                    .expect("serialized mutation carries canonical payload"),
                            ),
                            timestamp_micros,
                            expected_catalog_version,
                            _table_access.as_ref(),
                        )?;
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation_at_with_catalog_table_access(
                            txn_id,
                            std::sync::Arc::clone(
                                canonical_payload
                                    .as_ref()
                                    .expect("serialized mutation carries canonical payload"),
                            ),
                            timestamp_micros,
                            expected_catalog_version,
                            _table_access.as_ref(),
                        )?;
                    }
                }
            }
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll | Command::SetRole { .. } | Command::SessionControl { .. } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ShowTransactionIsolation => {
                return Err(ExecuteError::NonReadCommand(
                    "SHOW TRANSACTION ISOLATION LEVEL",
                ));
            }
            Command::CreateExtension(create) => {
                validate_bootstrap_create_extension(&create).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::DropExtension(drop) => {
                validate_bootstrap_drop_extension(&drop).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin { characteristics } => {
                self.begin_transaction_context(txn_id, characteristics)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                let snapshot = self
                    .transaction_snapshot_handle(txn_id)
                    .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
                self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
                let _statement = snapshot
                    .statement_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
                self.ensure_commit_path_available()
                    .map_err(execute_error_from_engine)?;
                if snapshot.transaction_delta_is_empty() {
                    self.finish_transaction_context(txn_id, true, chain)?;
                } else {
                    self.commit_transaction_delta(txn_id, chain, timestamp_micros)?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                let snapshot = self
                    .transaction_snapshot_handle(txn_id)
                    .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
                self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
                let _statement = snapshot
                    .statement_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
                self.ensure_commit_path_available()
                    .map_err(execute_error_from_engine)?;
                self.finish_transaction_context(txn_id, false, chain)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
            Command::Select(_)
            | Command::SelectFunction(_)
            | Command::SelectLiteral(_)
            | Command::PreparedCatalog(_)
            | Command::SequenceCurrVal(_) => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }

        Ok(())
    }

    pub fn execute_read_text(&mut self, text: &str) -> Result<Option<String>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(execute_error_from_engine)?;
        let cmd = parse_command(text)?;

        match cmd {
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state_mut().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
                Ok(self.get(&key))
            }
            Command::Begin { .. } => Err(ExecuteError::NonReadCommand("BEGIN")),
            Command::Commit { .. } => Err(ExecuteError::NonReadCommand("COMMIT")),
            Command::Rollback { .. } => Err(ExecuteError::NonReadCommand("ROLLBACK")),
            Command::Flush => Err(ExecuteError::NonReadCommand("FLUSH")),
            Command::ResetAll => Err(ExecuteError::NonReadCommand("RESET ALL")),
            Command::ShowTransactionIsolation => Err(ExecuteError::NonReadCommand(
                "SHOW TRANSACTION ISOLATION LEVEL",
            )),
            Command::SetRole { .. } => Err(ExecuteError::NonReadCommand("SET ROLE")),
            Command::SessionControl { .. } => Err(ExecuteError::NonReadCommand("SET")),
            Command::SetKv { .. } => Err(ExecuteError::NonReadCommand("SET")),
            Command::DeleteKv { .. } => Err(ExecuteError::NonReadCommand("DEL/DELETE")),
            Command::CreateSchema(_) => Err(ExecuteError::NonReadCommand("CREATE SCHEMA")),
            Command::DropSchema(_) => Err(ExecuteError::NonReadCommand("DROP SCHEMA")),
            Command::CreateDatabase(_) => Err(ExecuteError::NonReadCommand("CREATE DATABASE")),
            Command::DropDatabase(_) => Err(ExecuteError::NonReadCommand("DROP DATABASE")),
            Command::RenameDatabase(_) => Err(ExecuteError::NonReadCommand("ALTER DATABASE")),
            Command::CreateTablespace(_) => Err(ExecuteError::NonReadCommand("CREATE TABLESPACE")),
            Command::DropTablespace(_) => Err(ExecuteError::NonReadCommand("DROP TABLESPACE")),
            Command::RenameTablespace(_) => Err(ExecuteError::NonReadCommand("ALTER TABLESPACE")),
            Command::CreateTable(_) => Err(ExecuteError::NonReadCommand("CREATE TABLE")),
            Command::AddPrimaryKey(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddUniqueConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddCheckConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddForeignKey(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameTable(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::DropColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::DropConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::CreateIndex(_) => Err(ExecuteError::NonReadCommand("CREATE INDEX")),
            Command::RenameIndex(_) => Err(ExecuteError::NonReadCommand("ALTER INDEX")),
            Command::CreateView(_) => Err(ExecuteError::NonReadCommand("CREATE VIEW")),
            Command::RenameView(_) => Err(ExecuteError::NonReadCommand("ALTER VIEW")),
            Command::CreateMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("CREATE MATERIALIZED VIEW"))
            }
            Command::CreateExtension(_) => Err(ExecuteError::NonReadCommand("CREATE EXTENSION")),
            Command::DropExtension(_) => Err(ExecuteError::NonReadCommand("DROP EXTENSION")),
            Command::RefreshMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("REFRESH MATERIALIZED VIEW"))
            }
            Command::RenameMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("ALTER MATERIALIZED VIEW"))
            }
            Command::CreateFunction(_) => Err(ExecuteError::NonReadCommand("CREATE FUNCTION")),
            Command::RenameFunction(_) => Err(ExecuteError::NonReadCommand("ALTER FUNCTION")),
            Command::DropFunction(_) => Err(ExecuteError::NonReadCommand("DROP FUNCTION")),
            Command::CreateSequence(_) => Err(ExecuteError::NonReadCommand("CREATE SEQUENCE")),
            Command::CreateDomain(_) => Err(ExecuteError::NonReadCommand("CREATE DOMAIN")),
            Command::SequenceNextVal(_) => Err(ExecuteError::NonReadCommand("SELECT nextval")),
            Command::SequenceSetVal(_) => Err(ExecuteError::NonReadCommand("SELECT setval")),
            Command::SequenceRestart(_) => Err(ExecuteError::NonReadCommand("ALTER SEQUENCE")),
            Command::RenameSequence(_) => Err(ExecuteError::NonReadCommand("ALTER SEQUENCE")),
            Command::DropTable(_) => Err(ExecuteError::NonReadCommand("DROP TABLE")),
            Command::TruncateTable(_) => Err(ExecuteError::NonReadCommand("TRUNCATE TABLE")),
            Command::DropIndex(_) => Err(ExecuteError::NonReadCommand("DROP INDEX")),
            Command::DropView(_) => Err(ExecuteError::NonReadCommand("DROP VIEW")),
            Command::DropMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("DROP MATERIALIZED VIEW"))
            }
            Command::DropSequence(_) => Err(ExecuteError::NonReadCommand("DROP SEQUENCE")),
            Command::DropDomain(_) => Err(ExecuteError::NonReadCommand("DROP DOMAIN")),
            Command::GrantTable(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeTable(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantSchema(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeSchema(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantDatabase(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeDatabase(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantTablespace(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeTablespace(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantFunction(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeFunction(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::CreatePublication(_) => {
                Err(ExecuteError::NonReadCommand("CREATE PUBLICATION"))
            }
            Command::DropPublication(_) => Err(ExecuteError::NonReadCommand("DROP PUBLICATION")),
            Command::CreateSubscription(_) => {
                Err(ExecuteError::NonReadCommand("CREATE SUBSCRIPTION"))
            }
            Command::DropSubscription(_) => Err(ExecuteError::NonReadCommand("DROP SUBSCRIPTION")),
            Command::CreateRole(_) => Err(ExecuteError::NonReadCommand("CREATE ROLE")),
            Command::DropRole(_) => Err(ExecuteError::NonReadCommand("DROP ROLE")),
            Command::RenameRole(_) => Err(ExecuteError::NonReadCommand("ALTER ROLE")),
            Command::AlterRoleLogin(_) => Err(ExecuteError::NonReadCommand("ALTER ROLE")),
            Command::GrantDefaultTablePrivileges(_) => {
                Err(ExecuteError::NonReadCommand("ALTER DEFAULT PRIVILEGES"))
            }
            Command::RevokeDefaultTablePrivileges(_) => {
                Err(ExecuteError::NonReadCommand("ALTER DEFAULT PRIVILEGES"))
            }
            Command::AlterColumnDefault(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::CommentOn(_) => Err(ExecuteError::NonReadCommand("COMMENT")),
            Command::Insert(_) => Err(ExecuteError::NonReadCommand("INSERT")),
            Command::Delete(_) => Err(ExecuteError::NonReadCommand("DELETE")),
            Command::Update(_) => Err(ExecuteError::NonReadCommand("UPDATE")),
            Command::Select(_)
            | Command::SelectFunction(_)
            | Command::SelectLiteral(_)
            | Command::PreparedCatalog(_)
            | Command::SequenceCurrVal(_) => Err(ExecuteError::NonReadCommand("SELECT")),
        }
    }

    /// Pin one statement-stable relational read snapshot at the ALREADY-CHOSEN boundary `s` (PART B
    /// catalog↔data co-pinning). The boundary `s` was loaded ONCE per statement (by
    /// [`Engine::bind_relational_select_for_execution`], which also selected the catalog as-of `s`), so
    /// the data this pins and the catalog the statement bound against are the SAME generation — a
    /// concurrent shape-changing DDL can never split the reader's (catalog, data) pair. Pins ONE
    /// generation of `table` (its rows + value-index together) at `s`.
    pub(crate) fn pin_relational_read_at(&self, table: &str, s: Index) -> RelationalReadPin {
        RelationalReadPin {
            visibility: StorageVisibility { read_txn_id: s },
            table_rows: self.read_table_rows_at(table, s),
        }
    }
}

#[cfg(test)]
mod device_verdict_cardinality_tests {
    use super::exact_device_verdict_cardinality;

    #[test]
    fn exact_cardinality_rejects_short_and_long_device_components() {
        assert!(exact_device_verdict_cardinality(3, &[3, 3, 3]));
        assert!(!exact_device_verdict_cardinality(3, &[2, 3, 3]));
        assert!(!exact_device_verdict_cardinality(3, &[3, 4, 3]));
    }
}
