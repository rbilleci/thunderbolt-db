use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

use gpu_db_batching::{BatchItem, DualTriggerBatcher, FlushReason};
use gpu_db_execution::{
    DeviceRouter, DeviceTarget, FilterOperator, LimitOperator, MockGpuRuntime, Operator,
    ProjectOperator, RouteDecision, ScanOperator, SortOperator,
};
use gpu_db_metrics::{BatchFlushReason, FallbackReason, RuntimeMetrics};
use gpu_db_observability::{
    ActiveFallbackReason, EngineStatusSnapshot, EngineTelemetrySnapshot, FallbackStatus,
    ReadinessStatus, ReplicationLagSnapshot, SnapshotStatus, TelemetrySink,
};
use gpu_db_planner::{ExecutionPlan, Planner, PlannerConfig};
use gpu_db_protocol::{parse_command, Command, ParseError};
use gpu_db_replication::{LocalReplicator, LogReplicator, ReplicatedStateMachine};
use gpu_db_storage::{
    InMemoryTupleStore, NewTuple, StorageError, TupleStore, TupleVersion,
    Visibility as StorageVisibility,
};
use gpu_db_txn::{TxnError, TxnManager};
use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term, TxnId};
use gpu_db_wal::{WalBuffer, WalRecord};

#[derive(Debug, Default)]
pub struct KvStateMachine {
    pub applied: Vec<Vec<u8>>,
    pub kv: BTreeMap<String, String>,
}

impl ReplicatedStateMachine for KvStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        self.applied.push(entry.payload.clone());
        if let Ok(s) = std::str::from_utf8(&entry.payload) {
            if let Ok(cmd) = parse_command(s) {
                match cmd {
                    Command::SetKv { key, value } => {
                        self.kv.insert(key, value);
                    }
                    Command::DeleteKv { key } => {
                        self.kv.remove(&key);
                    }
                    Command::Begin
                    | Command::Commit { .. }
                    | Command::Rollback { .. }
                    | Command::Flush
                    | Command::ResetAll
                    | Command::GetKv { .. } => {}
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecuteError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Txn(#[from] TxnError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("command is not readable via execute_read_text: {0}")]
    NonReadCommand(&'static str),
}

#[derive(Debug, Clone)]
struct PendingMutation {
    txn_id: u64,
    payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MvccReadSource {
    FullScan,
    KeyLookup { key: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MvccReadFilter {
    KeyPrefix(String),
    KeyRange {
        start_inclusive: String,
        end_exclusive: String,
    },
    ValueEquals(String),
    All(Vec<MvccReadFilter>),
    Any(Vec<MvccReadFilter>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccReadOrder {
    KeyAsc,
    KeyDesc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccProjection {
    KeyValue,
    KeyOnly,
    ValueOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccReadQuery {
    pub source: MvccReadSource,
    pub visibility: StorageVisibility,
    pub filter: Option<MvccReadFilter>,
    pub order: Option<MvccReadOrder>,
    pub projection: MvccProjection,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccReadRow {
    pub key: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccReadResult {
    pub planned_target: DeviceTarget,
    pub executed_target: DeviceTarget,
    pub fallback_reason: Option<FallbackReason>,
    pub rows: Vec<MvccReadRow>,
}

fn decode_mvcc_row(bytes: &[u8], projection: MvccProjection) -> MvccReadRow {
    let text = String::from_utf8_lossy(bytes).into_owned();
    match projection {
        MvccProjection::KeyValue => {
            let mut parts = text.splitn(2, '\t');
            MvccReadRow {
                key: parts.next().map(str::to_owned),
                value: parts.next().map(str::to_owned),
            }
        }
        MvccProjection::KeyOnly => MvccReadRow {
            key: Some(text),
            value: None,
        },
        MvccProjection::ValueOnly => MvccReadRow {
            key: None,
            value: Some(text),
        },
    }
}

fn mvcc_row_matches_filter(row: &TupleVersion, filter: &MvccReadFilter) -> bool {
    match filter {
        MvccReadFilter::KeyPrefix(prefix) => row.key.starts_with(prefix),
        MvccReadFilter::KeyRange {
            start_inclusive,
            end_exclusive,
        } => {
            row.key.as_str() >= start_inclusive.as_str()
                && row.key.as_str() < end_exclusive.as_str()
        }
        MvccReadFilter::ValueEquals(expected) => row.value == *expected,
        MvccReadFilter::All(filters) => filters
            .iter()
            .all(|filter| mvcc_row_matches_filter(row, filter)),
        MvccReadFilter::Any(filters) => filters
            .iter()
            .any(|filter| mvcc_row_matches_filter(row, filter)),
    }
}

fn encode_mvcc_projection(row: TupleVersion, projection: MvccProjection) -> Vec<u8> {
    match projection {
        MvccProjection::KeyValue => format!("{}\t{}", row.key, row.value).into_bytes(),
        MvccProjection::KeyOnly => row.key.into_bytes(),
        MvccProjection::ValueOnly => row.value.into_bytes(),
    }
}

fn mvcc_row_cmp(
    left: &TupleVersion,
    right: &TupleVersion,
    order: MvccReadOrder,
) -> std::cmp::Ordering {
    match order {
        MvccReadOrder::KeyAsc => left.key.cmp(&right.key),
        MvccReadOrder::KeyDesc => right.key.cmp(&left.key),
    }
}

fn collect_operator_rows<Row, Op>(mut operator: Op) -> Vec<Row>
where
    Op: Operator<Row>,
{
    operator.open();
    let mut rows = Vec::new();
    while let Some(row) = operator.next() {
        rows.push(row);
    }
    operator.close();
    rows
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacklogBlocker {
    Wal,
    PendingBatch,
    ActiveTxn,
    CommitApplyGap,
    ApplyVisibleGap,
}

impl BacklogBlocker {
    pub const ALL: [Self; 5] = [
        Self::Wal,
        Self::PendingBatch,
        Self::ActiveTxn,
        Self::CommitApplyGap,
        Self::ApplyVisibleGap,
    ];

    pub const fn bit(self) -> u8 {
        match self {
            Self::Wal => ReplicationWatermarks::BACKLOG_BLOCKER_WAL,
            Self::PendingBatch => ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH,
            Self::ActiveTxn => ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN,
            Self::CommitApplyGap => ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP,
            Self::ApplyVisibleGap => ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP,
        }
    }

    pub const fn from_bit(bit: u8) -> Option<Self> {
        match bit {
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL => Some(Self::Wal),
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH => Some(Self::PendingBatch),
            ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN => Some(Self::ActiveTxn),
            ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP => Some(Self::CommitApplyGap),
            ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP => Some(Self::ApplyVisibleGap),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wal => "wal",
            Self::PendingBatch => "pending_batch",
            Self::ActiveTxn => "active_txn",
            Self::CommitApplyGap => "commit_apply_gap",
            Self::ApplyVisibleGap => "apply_visible_gap",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        let mut normalized = [0_u8; 32];
        let mut len = 0usize;

        for b in label.trim().bytes() {
            let folded = match b {
                b'A'..=b'Z' => b + 32,
                b'-' | b' ' | b'.' => b'_',
                _ => b,
            };

            if folded == b'_' && len > 0 && normalized[len - 1] == b'_' {
                continue;
            }

            if len == normalized.len() {
                return None;
            }

            normalized[len] = folded;
            len += 1;
        }

        let mut start = 0usize;
        while start < len && normalized[start] == b'_' {
            start += 1;
        }

        let mut end = len;
        while end > start && normalized[end - 1] == b'_' {
            end -= 1;
        }

        match &normalized[start..end] {
            b"wal" => Some(Self::Wal),
            b"pending_batch" => Some(Self::PendingBatch),
            b"active_txn" => Some(Self::ActiveTxn),
            b"commit_apply_gap" => Some(Self::CommitApplyGap),
            b"apply_visible_gap" => Some(Self::ApplyVisibleGap),
            _ => None,
        }
    }
}

impl fmt::Display for BacklogBlocker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown backlog blocker label: {label}")]
pub struct ParseBacklogBlockerError {
    label: String,
}

impl ParseBacklogBlockerError {
    pub fn label(&self) -> &str {
        &self.label
    }

    fn unknown(label: &str) -> Self {
        Self {
            label: label.trim().to_owned(),
        }
    }
}

impl FromStr for BacklogBlocker {
    type Err = ParseBacklogBlockerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_label(s).ok_or_else(|| ParseBacklogBlockerError::unknown(s))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicationWatermarks {
    pub role: Role,
    pub term: Term,
    pub commit_index: Index,
    pub applied_index: Index,
    pub visible_index: Index,
    pub commit_apply_gap: Index,
    pub apply_visible_gap: Index,
    pub snapshot_id: u64,
    pub wal_flushed_count: usize,
    pub wal_last_durable_txn_id: Option<TxnId>,
    pub wal_buffered_count: usize,
    pub wal_unflushed_count: usize,
    pub pending_batch_len: usize,
    pub pending_batch_cap: usize,
    pub pending_batch_remaining_capacity: usize,
    pub pending_batch_utilization_permyriad: u16,
    pub pending_batch_remaining_capacity_permyriad: u16,
    pub pending_batch_oldest_age_ms: Option<u64>,
    pub pending_batch_time_until_deadline_ms: Option<u64>,
    pub active_txn_count: usize,
    pub oldest_active_txn_id: Option<TxnId>,
    pub newest_active_txn_id: Option<TxnId>,
    pub has_wal_backlog: bool,
    pub has_pending_batch_backlog: bool,
    pub has_active_txn_backlog: bool,
    pub has_commit_apply_gap: bool,
    pub has_apply_visible_gap: bool,
    pub has_backlog_blockers: bool,
    pub backlog_blocker_count: u8,
    pub backlog_blocker_mask: u8,
    pub mutation_admission_saturated: bool,
    pub quiescent_for_failover: bool,
    pub follower_promotion_ready: bool,
}

impl ReplicationWatermarks {
    pub const BACKLOG_BLOCKER_WAL: u8 = 1 << 0;
    pub const BACKLOG_BLOCKER_PENDING_BATCH: u8 = 1 << 1;
    pub const BACKLOG_BLOCKER_ACTIVE_TXN: u8 = 1 << 2;
    pub const BACKLOG_BLOCKER_COMMIT_APPLY_GAP: u8 = 1 << 3;
    pub const BACKLOG_BLOCKER_APPLY_VISIBLE_GAP: u8 = 1 << 4;
    pub const KNOWN_BACKLOG_BLOCKER_MASK: u8 = Self::BACKLOG_BLOCKER_WAL
        | Self::BACKLOG_BLOCKER_PENDING_BATCH
        | Self::BACKLOG_BLOCKER_ACTIVE_TXN
        | Self::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        | Self::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;

    pub const fn known_backlog_blocker_mask() -> u8 {
        Self::KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub const fn unknown_backlog_blocker_mask(mask: u8) -> u8 {
        mask & !Self::KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub const fn sanitize_backlog_blocker_mask(mask: u8) -> u8 {
        mask & Self::KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub fn backlog_blocker_count_from_mask(mask: u8) -> u8 {
        Self::sanitize_backlog_blocker_mask(mask).count_ones() as u8
    }

    pub const fn has_backlog_blockers_in_mask(mask: u8) -> bool {
        Self::sanitize_backlog_blocker_mask(mask) != 0
    }

    pub fn has_backlog_blocker(&self, blocker_bit: u8) -> bool {
        debug_assert!(blocker_bit.is_power_of_two());
        let known_bit = Self::sanitize_backlog_blocker_mask(blocker_bit);
        known_bit != 0 && (self.backlog_blocker_mask & known_bit != 0)
    }

    pub fn has_blocker_kind(&self, blocker: BacklogBlocker) -> bool {
        self.has_backlog_blocker(blocker.bit())
    }

    pub fn backlog_blockers(&self) -> impl Iterator<Item = BacklogBlocker> + '_ {
        BacklogBlocker::ALL
            .into_iter()
            .filter(|blocker| self.has_blocker_kind(*blocker))
    }

    pub fn backlog_blocker_labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.backlog_blockers().map(|blocker| blocker.as_str())
    }

    pub fn backlog_blocker_bits(&self) -> impl Iterator<Item = u8> + '_ {
        self.backlog_blockers().map(|blocker| blocker.bit())
    }

    pub fn backlog_blockers_from_mask(mask: u8) -> impl Iterator<Item = BacklogBlocker> {
        let known_mask = Self::sanitize_backlog_blocker_mask(mask);
        BacklogBlocker::ALL
            .into_iter()
            .filter(move |blocker| known_mask & blocker.bit() != 0)
    }

    pub fn backlog_blocker_mask_from_labels<'a>(labels: impl IntoIterator<Item = &'a str>) -> u8 {
        labels
            .into_iter()
            .filter_map(BacklogBlocker::from_label)
            .fold(0_u8, |mask, blocker| mask | blocker.bit())
    }

    pub fn backlog_blocker_mask_from_delimited_labels(labels: &str) -> u8 {
        percent_decode_lossy(labels)
            .split([
                ',', ';', '|', '/', '\\', ':', '+', '&', '=', '\n', '\r', '\t', '[', ']', '{', '}',
                '(', ')', '<', '>', '"', '\'', '`',
            ])
            .filter_map(BacklogBlocker::from_label)
            .fold(0_u8, |mask, blocker| mask | blocker.bit())
    }

    pub fn backlog_blocker_labels_from_mask(mask: u8) -> impl Iterator<Item = &'static str> {
        Self::backlog_blockers_from_mask(mask).map(BacklogBlocker::as_str)
    }

    pub fn backlog_blocker_delimited_labels_from_mask(mask: u8, delimiter: &str) -> String {
        Self::backlog_blocker_labels_from_mask(mask)
            .collect::<Vec<_>>()
            .join(delimiter)
    }

    pub fn max_replication_gap(&self) -> Index {
        self.commit_apply_gap.max(self.apply_visible_gap)
    }

    pub fn total_backlog_items(&self) -> usize {
        self.wal_unflushed_count + self.pending_batch_len + self.active_txn_count
    }

    pub fn is_fully_caught_up(&self) -> bool {
        !self.has_wal_backlog
            && !self.has_pending_batch_backlog
            && !self.has_active_txn_backlog
            && !self.has_commit_apply_gap
            && !self.has_apply_visible_gap
    }
}

fn percent_decode_lossy(input: &str) -> String {
    fn hex_value(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let bytes = input.as_bytes();
    let mut idx = 0;
    let mut decoded = String::with_capacity(input.len());
    while idx < bytes.len() {
        if bytes[idx] == b'%' && idx + 2 < bytes.len() {
            if let (Some(high), Some(low)) = (hex_value(bytes[idx + 1]), hex_value(bytes[idx + 2]))
            {
                decoded.push((high << 4 | low) as char);
                idx += 3;
                continue;
            }
        }
        decoded.push(bytes[idx] as char);
        idx += 1;
    }
    decoded
}

pub struct Engine {
    repl: LocalReplicator,
    wal: WalBuffer,
    sm: KvStateMachine,
    mvcc_store: InMemoryTupleStore,
    txn_ids_by_index: BTreeMap<Index, TxnId>,
    txn_manager: TxnManager,
    visible_up_to: Index,
    metrics: RuntimeMetrics,
    batcher: DualTriggerBatcher<PendingMutation>,
    planner: Planner,
    router: DeviceRouter<MockGpuRuntime>,
}

impl Engine {
    pub fn new_local() -> Self {
        Self::with_planner_config(PlannerConfig::default())
    }

    pub fn with_planner_config(planner_cfg: PlannerConfig) -> Self {
        Self {
            repl: LocalReplicator::leader(),
            wal: WalBuffer::default(),
            sm: KvStateMachine::default(),
            mvcc_store: InMemoryTupleStore::new(),
            txn_ids_by_index: BTreeMap::new(),
            txn_manager: TxnManager::default(),
            visible_up_to: 0,
            metrics: RuntimeMetrics::default(),
            batcher: DualTriggerBatcher::new(64, Duration::from_millis(1)),
            planner: Planner::new(planner_cfg),
            router: DeviceRouter::new(MockGpuRuntime::default()),
        }
    }

    pub fn with_batching(max_items: usize, max_wait: Duration) -> Self {
        Self::with_batching_and_planner_config(max_items, max_wait, PlannerConfig::default())
    }

    pub fn with_batching_and_planner_config(
        max_items: usize,
        max_wait: Duration,
        planner_cfg: PlannerConfig,
    ) -> Self {
        let mut s = Self::with_planner_config(planner_cfg);
        s.batcher = DualTriggerBatcher::new(max_items, max_wait);
        s
    }

    pub fn simulate_next_wal_flush_failure(&mut self) {
        self.wal.fail_next_flush();
    }

    pub fn mark_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_unavailable(gpu_id);
    }

    pub fn clear_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_unavailable(gpu_id);
    }

    pub fn mark_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_memory_pressured(gpu_id);
    }

    pub fn clear_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_memory_pressured(gpu_id);
    }

    pub fn set_gpu_runtime_saturated(&mut self, saturated: bool) {
        self.router.runtime_mut().set_saturated(saturated);
    }

    pub fn become_follower(&mut self, term: Term) {
        self.repl.become_follower(term);
    }

    pub fn become_leader(&mut self, term: Term) {
        self.repl.become_leader(term);
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.repl.become_candidate(term);
    }

    pub fn commit_mutation(
        &mut self,
        txn_id: u64,
        payload: Vec<u8>,
    ) -> Result<CommitToken, EngineError> {
        if self.repl.role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let wal_len_before = self.wal.len();
        self.wal.append(WalRecord {
            txn_id,
            payload: payload.clone(),
        });

        let token = match self.repl.propose(payload) {
            Ok(token) => token,
            Err(err) => {
                self.wal.truncate(wal_len_before);
                return Err(err);
            }
        };
        if let Err(err) = self.wal.flush_all() {
            self.repl.rollback_unapplied_from(token.index);
            self.wal.truncate(wal_len_before);
            return Err(err);
        }

        self.repl.wait_committed(token, Duration::from_millis(0))?;
        self.txn_ids_by_index.insert(token.index, txn_id);

        let to_apply: Vec<LogEntry> = self
            .repl
            .drain_committed_from(self.repl.applied_index())
            .cloned()
            .collect();

        for e in &to_apply {
            self.sm.apply(e)?;
            self.apply_mvcc_entry(e)?;
            self.repl.mark_applied(e.index);
        }

        self.visible_up_to = self.visible_up_to.max(token.index);
        self.metrics.inc_commit();

        Ok(token)
    }

    fn apply_mvcc_entry(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        let Ok(text) = std::str::from_utf8(&entry.payload) else {
            return Ok(());
        };
        let Ok(cmd) = parse_command(text) else {
            return Ok(());
        };

        let txn_id = self
            .txn_ids_by_index
            .get(&entry.index)
            .copied()
            .unwrap_or(entry.index);
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };

        match cmd {
            Command::SetKv { key, value } => {
                if let Some(tuple) = self
                    .mvcc_store
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?
                {
                    self.mvcc_store
                        .tuple_update(tuple.tuple_id, value, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                } else {
                    self.mvcc_store
                        .tuple_insert(NewTuple { key, value }, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
            }
            Command::DeleteKv { key } => {
                if let Some(tuple) = self
                    .mvcc_store
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?
                {
                    self.mvcc_store
                        .tuple_delete(tuple.tuple_id, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    pub fn enqueue_set_text(
        &mut self,
        txn_id: u64,
        text: &str,
        now: Instant,
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        match cmd {
            Command::SetKv { .. } | Command::DeleteKv { .. } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                match self.route_command(&cmd) {
                    RouteDecision::Gpu(_) => {
                        let queue_cap = self.batcher.max_items();
                        let pending = self.batcher.len();
                        if pending >= queue_cap {
                            self.metrics.inc_fallback(FallbackReason::GpuQueueSaturated);
                            return Err(ExecuteError::Engine(
                                EngineError::MutationQueueOverloaded {
                                    pending,
                                    cap: queue_cap,
                                },
                            ));
                        }

                        let maybe_batch = self.batcher.enqueue(
                            PendingMutation {
                                txn_id,
                                payload: text.as_bytes().to_vec(),
                            },
                            now,
                        );
                        if let Some(batch) = maybe_batch {
                            self.metrics.observe_pending_batch_len(batch.items.len());
                            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
                        } else {
                            self.metrics.observe_pending_batch_len(self.batcher.len());
                        }
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                    RouteDecision::Cpu => {
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                }
            }
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.txn_manager.commit(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.txn_manager.rollback(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
        }
        Ok(())
    }

    pub fn tick_batching(&mut self, now: Instant) -> Result<(), EngineError> {
        if self.repl.role() != Role::Leader {
            if self.has_pending_batch() {
                return Err(EngineError::NotLeader);
            }
            return Ok(());
        }

        if let Some(batch) = self.batcher.maybe_flush_due_to_time(now) {
            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
        }
        Ok(())
    }

    pub fn flush_admin(&mut self) -> Result<(), EngineError> {
        if self.repl.role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        if let Some(batch) = self.batcher.flush_admin() {
            self.apply_batch(batch.reason, batch.items.into_iter(), Instant::now())?;
        }
        Ok(())
    }

    fn apply_batch<I>(
        &mut self,
        reason: FlushReason,
        items: I,
        flushed_at: Instant,
    ) -> Result<(), EngineError>
    where
        I: Iterator<Item = BatchItem<PendingMutation>>,
    {
        let metric_reason = match reason {
            FlushReason::Count => BatchFlushReason::Count,
            FlushReason::Time => BatchFlushReason::Time,
            FlushReason::Admin => BatchFlushReason::Admin,
        };

        let mut remaining = items.peekable();
        while let Some(p) = remaining.next() {
            let wait = flushed_at
                .saturating_duration_since(p.enqueued_at)
                .as_millis() as u64;
            let txn_id = p.item.txn_id;
            let payload = p.item.payload.clone();

            // In no-GPU bootstrap mode, batched mutations represent the simulated
            // GPU-eligible write path. Track transfer and kernel timing envelopes
            // so telemetry contracts are stable before CUDA is wired in.
            self.metrics.observe_h2d_bytes(payload.len() as u64);
            let simulated_kernel_ms = ((payload.len() as u64) / 1024).max(1);
            self.metrics.observe_kernel_exec_ms(simulated_kernel_ms);
            let simulated_occupancy = Self::simulate_kernel_occupancy_permyriad(payload.len());
            self.metrics
                .observe_kernel_occupancy_permyriad(simulated_occupancy);

            if let Err(err) = self.commit_mutation(txn_id, payload) {
                let tail: Vec<_> = std::iter::once(p).chain(remaining).collect();
                self.batcher.requeue_front(tail);
                self.metrics.observe_pending_batch_len(self.batcher.len());
                return Err(err);
            }

            self.metrics.observe_batch_wait_ms(wait);
        }

        self.metrics.observe_pending_batch_len(self.batcher.len());
        self.metrics.inc_batch_flush(metric_reason);
        Ok(())
    }

    pub fn plan_text(&self, text: &str) -> Result<ExecutionPlan, ParseError> {
        let cmd = parse_command(text)?;
        Ok(self.planner.plan_command(&cmd))
    }

    pub fn execute_text(&mut self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::SetKv { .. } | Command::DeleteKv { .. } => match self.route_command(&cmd) {
                RouteDecision::Gpu(_) | RouteDecision::Cpu => {
                    self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                }
                RouteDecision::CpuFallback { reason, .. } => {
                    self.metrics.inc_gpu_fallback(reason);
                    self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                }
            },
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.txn_manager.commit(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.txn_manager.rollback(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
        }

        Ok(())
    }

    pub fn execute_read_text(&mut self, text: &str) -> Result<Option<&str>, ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::GetKv { key } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
                Ok(self.get(&key))
            }
            Command::Begin => Err(ExecuteError::NonReadCommand("BEGIN")),
            Command::Commit { .. } => Err(ExecuteError::NonReadCommand("COMMIT")),
            Command::Rollback { .. } => Err(ExecuteError::NonReadCommand("ROLLBACK")),
            Command::Flush => Err(ExecuteError::NonReadCommand("FLUSH")),
            Command::ResetAll => Err(ExecuteError::NonReadCommand("RESET ALL")),
            Command::SetKv { .. } => Err(ExecuteError::NonReadCommand("SET")),
            Command::DeleteKv { .. } => Err(ExecuteError::NonReadCommand("DEL/DELETE")),
        }
    }

    pub fn execute_mvcc_query(
        &mut self,
        query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        if self.repl.role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }

        let planned_target = DeviceTarget::Gpu(self.planner.default_gpu_id());
        let executed_target = DeviceTarget::Cpu;
        let fallback_reason = Some(FallbackReason::GpuMvccReadParityGap);
        self.metrics
            .inc_fallback(FallbackReason::GpuMvccReadParityGap);

        let rows = match &query.source {
            MvccReadSource::FullScan => {
                let mut cursor = self.mvcc_store.seq_scan_open(query.visibility)?;
                let mut rows = Vec::new();
                while let Some(tuple) = cursor.next() {
                    rows.push(tuple);
                }
                rows
            }
            MvccReadSource::KeyLookup { key } => self
                .mvcc_store
                .tuple_fetch_by_key(key, query.visibility)?
                .into_iter()
                .collect(),
        };

        let projection = query.projection;
        let encoded_rows = match (query.filter.clone(), query.order, query.limit) {
            (Some(filter), Some(order), Some(limit)) => {
                collect_operator_rows(ProjectOperator::new(
                    LimitOperator::new(
                        SortOperator::new(
                            FilterOperator::new(
                                ScanOperator::new(rows),
                                move |row: &TupleVersion| mvcc_row_matches_filter(row, &filter),
                            ),
                            move |left: &TupleVersion, right: &TupleVersion| {
                                mvcc_row_cmp(left, right, order)
                            },
                        ),
                        limit,
                    ),
                    move |row| encode_mvcc_projection(row, projection),
                ))
            }
            (Some(filter), Some(order), None) => collect_operator_rows(ProjectOperator::new(
                SortOperator::new(
                    FilterOperator::new(ScanOperator::new(rows), move |row: &TupleVersion| {
                        mvcc_row_matches_filter(row, &filter)
                    }),
                    move |left: &TupleVersion, right: &TupleVersion| {
                        mvcc_row_cmp(left, right, order)
                    },
                ),
                move |row| encode_mvcc_projection(row, projection),
            )),
            (Some(filter), None, Some(limit)) => collect_operator_rows(ProjectOperator::new(
                LimitOperator::new(
                    FilterOperator::new(ScanOperator::new(rows), move |row: &TupleVersion| {
                        mvcc_row_matches_filter(row, &filter)
                    }),
                    limit,
                ),
                move |row| encode_mvcc_projection(row, projection),
            )),
            (Some(filter), None, None) => collect_operator_rows(ProjectOperator::new(
                FilterOperator::new(ScanOperator::new(rows), move |row: &TupleVersion| {
                    mvcc_row_matches_filter(row, &filter)
                }),
                move |row| encode_mvcc_projection(row, projection),
            )),
            (None, Some(order), Some(limit)) => collect_operator_rows(ProjectOperator::new(
                LimitOperator::new(
                    SortOperator::new(
                        ScanOperator::new(rows),
                        move |left: &TupleVersion, right: &TupleVersion| {
                            mvcc_row_cmp(left, right, order)
                        },
                    ),
                    limit,
                ),
                move |row| encode_mvcc_projection(row, projection),
            )),
            (None, Some(order), None) => collect_operator_rows(ProjectOperator::new(
                SortOperator::new(
                    ScanOperator::new(rows),
                    move |left: &TupleVersion, right: &TupleVersion| {
                        mvcc_row_cmp(left, right, order)
                    },
                ),
                move |row| encode_mvcc_projection(row, projection),
            )),
            (None, None, Some(limit)) => collect_operator_rows(ProjectOperator::new(
                LimitOperator::new(ScanOperator::new(rows), limit),
                move |row| encode_mvcc_projection(row, projection),
            )),
            (None, None, None) => {
                collect_operator_rows(ProjectOperator::new(ScanOperator::new(rows), move |row| {
                    encode_mvcc_projection(row, projection)
                }))
            }
        };

        let total_d2h_bytes: u64 = encoded_rows.iter().map(|row| row.len() as u64).sum();
        if total_d2h_bytes > 0 {
            self.metrics.observe_d2h_bytes(total_d2h_bytes);
        }

        let projected = encoded_rows
            .iter()
            .map(|row| decode_mvcc_row(row, query.projection))
            .collect();

        Ok(MvccReadResult {
            planned_target,
            executed_target,
            fallback_reason,
            rows: projected,
        })
    }

    pub fn visible_up_to(&self) -> Index {
        self.visible_up_to
    }

    pub fn applied_len(&self) -> usize {
        self.sm.applied.len()
    }

    pub fn wal_flushed_count(&self) -> usize {
        self.wal.flushed_count()
    }

    pub fn wal_buffered_count(&self) -> usize {
        self.wal.len()
    }

    pub fn wal_unflushed_count(&self) -> usize {
        self.wal.unflushed_count()
    }

    pub fn durable_wal_records(&self) -> &[WalRecord] {
        self.wal.flushed_records()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.sm.kv.get(key).map(|s| s.as_str())
    }

    pub fn visible_state_fingerprint(&self) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x00000100000001B3;

        fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
            for b in bytes {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hash
        }

        let mut hash = FNV_OFFSET_BASIS;
        for (k, v) in &self.sm.kv {
            hash = hash_bytes(hash, k.as_bytes());
            hash = hash_bytes(hash, &[0xFF]);
            hash = hash_bytes(hash, v.as_bytes());
            hash = hash_bytes(hash, &[0x00]);
        }
        hash
    }

    pub fn active_txn_count(&self) -> usize {
        self.txn_manager.active_count()
    }

    pub fn replication_watermarks(&self) -> ReplicationWatermarks {
        let now = Instant::now();
        let pending_batch_len = self.batcher.len();
        let pending_batch_cap = self.batcher.max_items();
        let wal_unflushed_count = self.wal.unflushed_count();
        let active_txn_count = self.txn_manager.active_count();
        let role = self.repl.role();
        let pending_batch_remaining_capacity = pending_batch_cap.saturating_sub(pending_batch_len);
        let pending_batch_utilization_permyriad = if pending_batch_cap == 0 {
            0
        } else {
            let utilization =
                (pending_batch_len as u128).saturating_mul(10_000) / (pending_batch_cap as u128);
            utilization.min(10_000) as u16
        };
        let pending_batch_remaining_capacity_permyriad =
            10_000u16.saturating_sub(pending_batch_utilization_permyriad);

        let commit_index = self.repl.commit_index();
        let applied_index = self.repl.applied_index();
        let visible_index = self.visible_up_to;

        let commit_apply_gap = commit_index.saturating_sub(applied_index);
        let apply_visible_gap = applied_index.saturating_sub(visible_index);

        let oldest_active_txn_id = self.txn_manager.oldest_active_txn_id();
        let newest_active_txn_id = self.txn_manager.newest_active_txn_id();
        let has_wal_backlog = wal_unflushed_count > 0;
        let has_pending_batch_backlog = pending_batch_len > 0;
        let has_active_txn_backlog = active_txn_count > 0;
        let has_commit_apply_gap = commit_apply_gap > 0;
        let has_apply_visible_gap = apply_visible_gap > 0;
        let backlog_blocker_mask = (u8::from(has_wal_backlog)
            * ReplicationWatermarks::BACKLOG_BLOCKER_WAL)
            | (u8::from(has_pending_batch_backlog)
                * ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH)
            | (u8::from(has_active_txn_backlog)
                * ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN)
            | (u8::from(has_commit_apply_gap)
                * ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP)
            | (u8::from(has_apply_visible_gap)
                * ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP);
        let backlog_blocker_count =
            ReplicationWatermarks::backlog_blocker_count_from_mask(backlog_blocker_mask);
        let has_backlog_blockers =
            ReplicationWatermarks::has_backlog_blockers_in_mask(backlog_blocker_mask);

        let wal_checkpoint = self.wal.checkpoint_meta();

        ReplicationWatermarks {
            role,
            term: self.repl.current_term(),
            commit_index,
            applied_index,
            visible_index,
            commit_apply_gap,
            apply_visible_gap,
            snapshot_id: self.repl.snapshot_meta().snapshot_id,
            wal_flushed_count: wal_checkpoint.durable_record_count,
            wal_last_durable_txn_id: wal_checkpoint.last_durable_txn_id,
            wal_buffered_count: self.wal.len(),
            wal_unflushed_count,
            pending_batch_len,
            pending_batch_cap,
            pending_batch_remaining_capacity,
            pending_batch_utilization_permyriad,
            pending_batch_remaining_capacity_permyriad,
            pending_batch_oldest_age_ms: self
                .pending_batch_oldest_age(now)
                .map(|age| age.as_millis() as u64),
            pending_batch_time_until_deadline_ms: self
                .pending_batch_time_until_deadline(now)
                .map(|remaining| remaining.as_millis() as u64),
            active_txn_count,
            oldest_active_txn_id,
            newest_active_txn_id,
            has_wal_backlog,
            has_pending_batch_backlog,
            has_active_txn_backlog,
            has_commit_apply_gap,
            has_apply_visible_gap,
            has_backlog_blockers,
            backlog_blocker_count,
            backlog_blocker_mask,
            mutation_admission_saturated: pending_batch_len >= pending_batch_cap,
            quiescent_for_failover: role == Role::Leader
                && !has_wal_backlog
                && !has_pending_batch_backlog
                && !has_active_txn_backlog,
            follower_promotion_ready: role == Role::Follower
                && !has_commit_apply_gap
                && !has_apply_visible_gap
                && !has_wal_backlog
                && !has_pending_batch_backlog
                && !has_active_txn_backlog,
        }
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.repl.export_snapshot_meta()
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        self.repl.install_snapshot(meta);
        self.visible_up_to = self
            .visible_up_to
            .max(self.repl.snapshot_meta().last_included_index);
    }

    pub fn snapshot_meta(&self) -> SnapshotMeta {
        self.repl.snapshot_meta()
    }

    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.metrics
    }

    pub fn telemetry_snapshot(&self) -> EngineTelemetrySnapshot {
        let marks = self.replication_watermarks();
        EngineTelemetrySnapshot {
            role: marks.role,
            replication_lag: ReplicationLagSnapshot {
                commit_index: marks.commit_index,
                applied_index: marks.applied_index,
                visible_index: marks.visible_index,
                commit_apply_gap: marks.commit_apply_gap,
                apply_visible_gap: marks.apply_visible_gap,
            },
            runtime_metrics: self.metrics.snapshot(),
            snapshot_id: marks.snapshot_id,
            wal_flushed_count: marks.wal_flushed_count,
            wal_last_durable_txn_id: marks.wal_last_durable_txn_id,
            wal_buffered_count: marks.wal_buffered_count,
            wal_unflushed_count: marks.wal_unflushed_count,
            pending_batch_len: marks.pending_batch_len,
            pending_batch_cap: marks.pending_batch_cap,
            active_txn_count: marks.active_txn_count,
            backlog_blocker_count: marks.backlog_blocker_count,
            backlog_blocker_mask: marks.backlog_blocker_mask,
            mutation_admission_saturated: marks.mutation_admission_saturated,
            quiescent_for_failover: marks.quiescent_for_failover,
            follower_promotion_ready: marks.follower_promotion_ready,
            gpu_parity_fallbacks: self.metrics.fallback_counts_by_gpu_parity_issue(),
            gpu_runtime: self.router.runtime().snapshot(),
        }
    }

    pub fn status_snapshot(&self) -> EngineStatusSnapshot {
        let marks = self.replication_watermarks();
        let snapshot_meta = self.snapshot_meta();
        let runtime_metrics = self.metrics.snapshot();
        let gpu_runtime = self.router.runtime().snapshot();

        let mut active_reasons = Vec::new();
        if !gpu_runtime.unavailable_gpu_ids.is_empty() {
            active_reasons.push(ActiveFallbackReason::GpuUnavailable {
                gpu_ids: gpu_runtime.unavailable_gpu_ids.clone(),
            });
        }
        if !gpu_runtime.memory_pressured_gpu_ids.is_empty() {
            active_reasons.push(ActiveFallbackReason::GpuMemoryPressure {
                gpu_ids: gpu_runtime.memory_pressured_gpu_ids.clone(),
            });
        }
        if gpu_runtime.saturated {
            active_reasons.push(ActiveFallbackReason::GpuQueueSaturated);
        }

        EngineStatusSnapshot::new(
            marks.role,
            marks.term,
            SnapshotStatus {
                snapshot_id: snapshot_meta.snapshot_id,
                last_included_index: snapshot_meta.last_included_index,
                last_included_term: snapshot_meta.last_included_term,
                visible_index: marks.visible_index,
            },
            ReplicationLagSnapshot {
                commit_index: marks.commit_index,
                applied_index: marks.applied_index,
                visible_index: marks.visible_index,
                commit_apply_gap: marks.commit_apply_gap,
                apply_visible_gap: marks.apply_visible_gap,
            },
            ReadinessStatus {
                pending_batch_len: marks.pending_batch_len,
                pending_batch_cap: marks.pending_batch_cap,
                active_txn_count: marks.active_txn_count,
                wal_unflushed_count: marks.wal_unflushed_count,
                backlog_blocker_count: marks.backlog_blocker_count,
                backlog_blocker_mask: marks.backlog_blocker_mask,
                mutation_admission_saturated: marks.mutation_admission_saturated,
                quiescent_for_failover: marks.quiescent_for_failover,
                follower_promotion_ready: marks.follower_promotion_ready,
            },
            FallbackStatus {
                last_reason: runtime_metrics.last_fallback_reason,
                gpu_parity_fallbacks: self.metrics.fallback_counts_by_gpu_parity_issue(),
                active_reasons,
                gpu_runtime,
            },
            runtime_metrics,
        )
        .expect("engine status snapshot invariants should hold")
    }

    pub fn publish_telemetry<S: TelemetrySink>(&self, sink: &mut S) {
        sink.publish(&self.telemetry_snapshot());
    }

    pub fn pending_batch_len(&self) -> usize {
        self.batcher.len()
    }

    pub fn has_pending_batch(&self) -> bool {
        !self.batcher.is_empty()
    }

    pub fn pending_batch_oldest_age(&self, now: Instant) -> Option<Duration> {
        self.batcher
            .first_enqueued_at()
            .map(|head| now.saturating_duration_since(head))
    }

    pub fn pending_batch_time_until_deadline(&self, now: Instant) -> Option<Duration> {
        self.batcher.time_until_flush_deadline(now)
    }

    pub fn batching_config(&self) -> (usize, Duration) {
        (self.batcher.max_items(), self.batcher.max_wait())
    }

    fn route_command(&self, cmd: &Command) -> RouteDecision {
        let plan = self.planner.plan_command(cmd);
        let Some(node) = plan.nodes().first() else {
            return RouteDecision::Cpu;
        };
        self.router.route(&node.op)
    }

    fn simulate_kernel_occupancy_permyriad(payload_len: usize) -> u16 {
        // Bootstrap heuristic for no-GPU mode: scale occupancy with payload size
        // while capping at 100% to keep telemetry realistic.
        let permyriad = 2_500u64.saturating_add((payload_len as u64).saturating_mul(100));
        permyriad.min(10_000) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_db_execution::DeviceTarget;
    use gpu_db_metrics::GpuParityIssue;
    use gpu_db_observability::InMemoryTelemetrySink;

    #[test]
    fn planner_targets_mutations_to_gpu() {
        let e = Engine::new_local();
        let plan = e.plan_text("SET a=1").unwrap();

        assert_eq!(plan.nodes().len(), 1);
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(0));
    }

    #[test]
    fn planner_targets_get_to_cpu_fallback_path() {
        let e = Engine::new_local();
        let plan = e.plan_text("GET a").unwrap();

        assert_eq!(plan.nodes().len(), 1);
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Cpu);
    }

    #[test]
    fn planner_config_can_override_default_gpu_target() {
        let e = Engine::with_planner_config(PlannerConfig { default_gpu_id: 3 });
        let plan = e.plan_text("SET a=1").unwrap();

        assert_eq!(plan.nodes().len(), 1);
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(3));
    }

    #[test]
    fn wal_before_visibility_holds() {
        let mut e = Engine::new_local();
        let t = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        assert!(e.wal_flushed_count() >= 1);
        assert!(e.visible_up_to() >= t.index);
        assert!(e.applied_len() >= 1);
    }

    #[test]
    fn commit_indices_monotonic() {
        let mut e = Engine::new_local();
        let a = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let b = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
        assert!(b.index > a.index);
        assert!(e.visible_up_to() >= b.index);
    }

    #[test]
    fn execute_set_updates_state_machine() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
    }

    #[test]
    fn execute_set_accepts_session_and_local_scope_aliases() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET SESSION balance=100").unwrap();
        e.execute_text(2, "SET LOCAL balance TO 101").unwrap();

        assert_eq!(e.get("balance"), Some("101"));
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn execute_del_removes_existing_key() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.execute_text(2, "DEL balance").unwrap();

        assert_eq!(e.get("balance"), None);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn execute_delete_alias_removes_existing_key() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.execute_text(2, "DELETE balance").unwrap();

        assert_eq!(e.get("balance"), None);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn execute_read_text_get_returns_current_value_without_committing() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();

        let value = e.execute_read_text("GET balance").unwrap();
        assert_eq!(value, Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 1);
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);
        assert_eq!(
            e.metrics().last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
    }

    #[test]
    fn execute_read_text_get_missing_key_does_not_track_d2h_bytes() {
        let mut e = Engine::new_local();
        let value = e.execute_read_text("GET absent").unwrap();

        assert_eq!(value, None);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn execute_read_text_rejects_non_read_commands() {
        let mut e = Engine::new_local();

        let begin_err = e.execute_read_text("BEGIN").unwrap_err();
        assert!(matches!(begin_err, ExecuteError::NonReadCommand("BEGIN")));

        let set_err = e.execute_read_text("SET balance=100").unwrap_err();
        assert!(matches!(set_err, ExecuteError::NonReadCommand("SET")));

        let reset_err = e.execute_read_text("RESET ALL").unwrap_err();
        assert!(matches!(
            reset_err,
            ExecuteError::NonReadCommand("RESET ALL")
        ));

        let discard_err = e.execute_read_text("DISCARD TEMP").unwrap_err();
        assert!(matches!(
            discard_err,
            ExecuteError::NonReadCommand("RESET ALL")
        ));

        let del_err = e.execute_read_text("DELETE balance").unwrap_err();
        assert!(matches!(
            del_err,
            ExecuteError::NonReadCommand("DEL/DELETE")
        ));

        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn execute_read_text_rejects_get_when_not_leader() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.become_follower(2);

        let err = e.execute_read_text("GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn execute_text_get_rejects_when_not_leader() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.execute_text(1, "GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn execute_text_get_rejects_when_candidate() {
        let mut e = Engine::new_local();
        e.become_candidate(2);

        let err = e.execute_text(1, "GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn execute_text_get_tracks_d2h_bytes_for_hits_only() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();

        e.execute_text(2, "GET balance").unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);

        e.execute_text(3, "GET missing").unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);
    }

    #[test]
    fn execute_read_text_rejects_get_when_candidate() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.become_candidate(2);

        let err = e.execute_read_text("GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn batching_flushes_on_count_and_updates_metric() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.enqueue_set_text(2, "SET b=2", t0).unwrap();
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.metrics().batch_flush_count, 1);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(
            e.metrics().last_batch_flush_reason(),
            Some(BatchFlushReason::Count)
        );
        assert_eq!(e.metrics().batch_wait_samples, 2);
        assert_eq!(e.metrics().batch_wait_total_ms, 0);
        assert_eq!(e.metrics().last_batch_wait_ms(), Some(0));
        assert_eq!(
            e.metrics().h2d_bytes_total,
            "SET a=1".len() as u64 + "SET b=2".len() as u64
        );
        assert_eq!(e.metrics().kernel_exec_samples, 2);
        assert_eq!(e.metrics().kernel_exec_total_ms, 2);
        assert_eq!(e.metrics().last_kernel_exec_ms(), Some(1));
        assert_eq!(e.metrics().kernel_occupancy_samples, 2);
        assert_eq!(e.metrics().kernel_occupancy_total_permyriad, 6400);
        assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(3200));
        assert_eq!(e.metrics().pending_batch_peak, 2);
        assert_eq!(e.metrics().last_pending_batch_len(), Some(0));
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn batching_flushes_on_time() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=7", t0).unwrap();
        assert!(e.has_pending_batch());
        assert_eq!(e.pending_batch_len(), 1);
        e.tick_batching(t0 + Duration::from_millis(3)).unwrap();
        assert_eq!(e.get("a"), Some("7"));
        assert!(!e.has_pending_batch());
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 1);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Time), 1);
        assert_eq!(e.metrics().batch_wait_samples, 1);
        assert_eq!(e.metrics().batch_wait_total_ms, 3);
        assert_eq!(e.metrics().last_batch_wait_ms(), Some(3));
    }

    #[test]
    fn batching_kernel_occupancy_caps_at_full_utilization() {
        let mut e = Engine::with_batching(1, Duration::from_secs(999));
        let t0 = Instant::now();
        let payload = format!("SET a={}", "x".repeat(512));

        e.enqueue_set_text(1, &payload, t0).unwrap();

        assert_eq!(e.metrics().kernel_occupancy_samples, 1);
        assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(10_000));
        assert_eq!(e.metrics().kernel_occupancy_total_permyriad, 10_000);
    }

    #[test]
    fn admin_flush_tracks_reason() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=9", t0).unwrap();
        assert_eq!(e.pending_batch_len(), 1);
        e.flush_admin().unwrap();

        assert_eq!(e.get("a"), Some("9"));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    }

    #[test]
    fn admin_flush_without_pending_queue_is_noop() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.flush_admin().unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.pending_batch_oldest_age(t0), None);
        assert_eq!(e.pending_batch_time_until_deadline(t0), None);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn batching_config_reflects_engine_settings() {
        let e = Engine::with_batching(7, Duration::from_millis(42));
        assert_eq!(e.batching_config(), (7, Duration::from_millis(42)));
    }

    #[test]
    fn batching_and_planner_config_can_be_combined() {
        let e = Engine::with_batching_and_planner_config(
            3,
            Duration::from_millis(9),
            PlannerConfig { default_gpu_id: 5 },
        );

        assert_eq!(e.batching_config(), (3, Duration::from_millis(9)));
        let plan = e.plan_text("SET a=1").unwrap();
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(5));
    }

    #[test]
    fn pending_batch_deadline_counts_down_and_clears_after_flush() {
        let mut e = Engine::with_batching(10, Duration::from_millis(10));
        let t0 = Instant::now();

        assert_eq!(e.pending_batch_time_until_deadline(t0), None);

        e.enqueue_set_text(1, "SET a=9", t0).unwrap();
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(4)),
            Some(Duration::from_millis(6))
        );
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(12)),
            Some(Duration::ZERO)
        );

        e.flush_admin().unwrap();
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(13)),
            None
        );
    }

    #[test]
    fn pending_batch_oldest_age_tracks_then_clears_after_flush() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=9", t0).unwrap();

        let age = e
            .pending_batch_oldest_age(t0 + Duration::from_millis(5))
            .expect("pending batch age should exist");
        assert!(age >= Duration::from_millis(5));

        e.flush_admin().unwrap();
        assert_eq!(
            e.pending_batch_oldest_age(t0 + Duration::from_millis(6)),
            None
        );
    }

    #[test]
    fn batching_can_apply_set_then_del_in_order() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.enqueue_set_text(2, "DEL a", t0).unwrap();

        assert_eq!(e.get("a"), None);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn deterministic_replay_matches_between_immediate_and_batched_mutation_paths() {
        let trace = [
            (1, "SET acct_a=10"),
            (2, "SET acct_b=25"),
            (3, "DEL acct_a"),
            (4, "SET acct_c=77"),
            (5, "DELETE acct_b"),
            (6, "SET acct_a=99"),
        ];

        let mut immediate = Engine::new_local();
        for (txn_id, cmd) in trace {
            immediate.execute_text(txn_id, cmd).unwrap();
        }

        let mut batched = Engine::with_batching(64, Duration::from_secs(999));
        let t0 = Instant::now();
        for (txn_id, cmd) in trace {
            batched.enqueue_set_text(txn_id, cmd, t0).unwrap();
        }
        batched.flush_admin().unwrap();

        assert_eq!(immediate.sm.applied, batched.sm.applied);
        assert_eq!(immediate.sm.kv, batched.sm.kv);
        assert_eq!(immediate.visible_up_to(), batched.visible_up_to());
        assert_eq!(
            immediate.visible_state_fingerprint(),
            batched.visible_state_fingerprint()
        );
        assert_eq!(immediate.wal_flushed_count(), trace.len());
        assert_eq!(batched.wal_flushed_count(), trace.len());
    }

    #[test]
    fn visible_state_fingerprint_changes_with_visible_kv_state() {
        let mut e = Engine::new_local();
        let empty = e.visible_state_fingerprint();

        e.execute_text(1, "SET a=1").unwrap();
        let after_set = e.visible_state_fingerprint();
        assert_ne!(after_set, empty);

        e.execute_text(2, "DELETE a").unwrap();
        let after_delete = e.visible_state_fingerprint();
        assert_eq!(after_delete, empty);
    }

    #[test]
    fn flush_command_drains_pending_batch() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=5", t0).unwrap();
        e.execute_text(2, "FLUSH").unwrap();

        assert_eq!(e.get("a"), Some("5"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    }

    #[test]
    fn flush_aliases_drain_pending_batch() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=5", t0).unwrap();
        e.execute_text(2, "FLUSH WAL").unwrap();

        e.enqueue_set_text(3, "SET b=7", t0).unwrap();
        e.execute_text(4, "FLUSH LOG").unwrap();

        assert_eq!(e.get("a"), Some("5"));
        assert_eq!(e.get("b"), Some("7"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 2);
    }

    #[test]
    fn wal_flush_failure_prevents_visibility_advance() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();
        let res = e.commit_mutation(1, b"SET a=1".to_vec());
        assert!(matches!(res, Err(EngineError::Durability(_))));
        assert_eq!(e.visible_up_to(), 0);
    }

    #[test]
    fn wal_flush_failure_does_not_leak_into_later_successful_commit() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();
        let _ = e.commit_mutation(1, b"SET a=1".to_vec());

        e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

        assert_eq!(e.get("a"), None);
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.applied_len(), 1);
    }

    #[test]
    fn wal_flush_failure_discards_unflushed_record_from_buffer() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

        assert!(matches!(err, EngineError::Durability(_)));
        assert_eq!(e.wal_flushed_count(), 0);
        assert_eq!(e.wal_buffered_count(), 0);
        assert_eq!(e.wal_unflushed_count(), 0);
    }

    #[test]
    fn durable_wal_records_exclude_failed_commit_attempts() {
        let mut e = Engine::new_local();

        e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        e.simulate_next_wal_flush_failure();
        let _ = e.commit_mutation(2, b"SET b=2".to_vec());

        let durable = e.durable_wal_records();
        assert_eq!(durable.len(), 1);
        assert_eq!(durable[0].txn_id, 1);
        assert_eq!(durable[0].payload, b"SET a=1".to_vec());
    }

    #[test]
    fn follower_rejects_commit_without_visibility_or_wal_flush() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.visible_up_to(), 0);
        assert_eq!(e.wal_flushed_count(), 0);
        assert_eq!(e.applied_len(), 0);
    }

    #[test]
    fn follower_rejects_batched_enqueue_without_mutating_queue_or_metrics() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_follower(2);

        let t0 = Instant::now();
        let err = e.enqueue_set_text(1, "SET a=1", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_get_rejects_when_not_leader_without_queue_side_effects() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_follower(2);

        let t0 = Instant::now();
        let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_get_rejects_when_candidate_without_queue_side_effects() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_candidate(2);

        let t0 = Instant::now();
        let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_get_tracks_d2h_bytes_for_hits_only() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET balance=100", t0).unwrap();
        e.flush_admin().unwrap();

        e.enqueue_set_text(2, "GET balance", t0).unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);

        e.enqueue_set_text(3, "GET missing", t0).unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);
    }

    #[test]
    fn execute_text_mutation_falls_back_to_cpu_when_gpu_is_unavailable() {
        let mut e = Engine::new_local();
        e.mark_gpu_unavailable(0);

        e.execute_text(1, "SET balance=100").unwrap();

        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);

        let snapshot = e.telemetry_snapshot();
        assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
        assert!(snapshot.has_gpu_parity_fallbacks());
        assert!(snapshot.has_gpu_runtime_pressure());
        assert_eq!(snapshot.blocked_gpu_ids(), vec![0]);
        assert_eq!(
            snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
                id: "GPU-120",
                owner: "runtime",
                milestone: "m0-bootstrap",
            }),
            Some(&1)
        );
    }

    #[test]
    fn enqueue_mutation_falls_back_to_cpu_when_gpu_is_memory_pressured() {
        let mut e = Engine::with_batching(8, Duration::from_secs(60));
        let t0 = Instant::now();
        e.mark_gpu_memory_pressured(0);

        e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(
            e.metrics().fallback_for(FallbackReason::GpuMemoryPressure),
            1
        );
    }

    #[test]
    fn enqueue_mutation_runtime_saturation_falls_back_before_queueing() {
        let mut e = Engine::with_batching(8, Duration::from_secs(60));
        let t0 = Instant::now();
        e.set_gpu_runtime_saturated(true);

        e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(
            e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
            1
        );

        let snapshot = e.telemetry_snapshot();
        assert!(snapshot.has_gpu_runtime_pressure());
        assert!(snapshot.blocked_gpu_ids().is_empty());
        assert!(snapshot.gpu_runtime.saturated);
    }

    #[test]
    fn candidate_rejects_commit_and_batched_enqueue() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_candidate(2);

        let commit_err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
        assert!(matches!(commit_err, EngineError::NotLeader));

        let enqueue_err = e
            .enqueue_set_text(1, "SET a=1", Instant::now())
            .unwrap_err();
        assert!(matches!(
            enqueue_err,
            ExecuteError::Engine(EngineError::NotLeader)
        ));

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 0);
        assert_eq!(e.visible_up_to(), 0);
    }

    #[test]
    fn failed_admin_flush_does_not_increment_flush_metrics_or_drop_pending_queue() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let err = e.flush_admin().unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
        assert_eq!(e.pending_batch_len(), 1);
    }

    #[test]
    fn batch_flush_wal_failure_requeues_items_for_retry() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.simulate_next_wal_flush_failure();
        let err = e
            .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
            .unwrap_err();

        assert!(matches!(
            err,
            ExecuteError::Engine(EngineError::Durability(_))
        ));
        assert_eq!(e.pending_batch_len(), 2);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_wait_samples, 0);
        assert_eq!(e.metrics().pending_batch_peak, 2);
        assert_eq!(e.metrics().last_pending_batch_len(), Some(2));
        assert_eq!(e.metrics().commits_total, 0);

        e.flush_admin().unwrap();
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn enqueue_rejects_new_mutation_when_retry_queue_is_saturated() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.simulate_next_wal_flush_failure();
        let flush_err = e
            .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
            .unwrap_err();
        assert!(matches!(
            flush_err,
            ExecuteError::Engine(EngineError::Durability(_))
        ));
        assert_eq!(e.pending_batch_len(), 2);

        let saturated_err = e
            .enqueue_set_text(3, "SET c=3", t0 + Duration::from_millis(2))
            .unwrap_err();
        assert!(matches!(
            saturated_err,
            ExecuteError::Engine(EngineError::MutationQueueOverloaded { pending: 2, cap: 2 })
        ));
        assert_eq!(e.pending_batch_len(), 2);
        assert_eq!(
            e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
            1
        );
        assert_eq!(e.metrics().commits_total, 0);

        let snapshot = e.telemetry_snapshot();
        assert!(snapshot.has_gpu_parity_fallbacks());
        assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
        assert_eq!(
            snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
                id: "GPU-121",
                owner: "runtime",
                milestone: "m0-bootstrap",
            }),
            Some(&1)
        );
    }

    #[test]
    fn failed_time_flush_does_not_drop_pending_queue() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let err = e.tick_batching(t0 + Duration::from_millis(3)).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.pending_batch_len(), 1);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn follower_tick_without_pending_batch_is_noop() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        e.become_follower(2);

        e.tick_batching(Instant::now()).unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn pending_batch_can_be_flushed_after_follower_is_promoted_back_to_leader() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let tick_err = e.tick_batching(t0 + Duration::from_secs(1)).unwrap_err();
        assert!(matches!(tick_err, EngineError::NotLeader));
        assert_eq!(e.pending_batch_len(), 1);
        assert_eq!(e.get("a"), None);

        e.become_leader(3);
        e.flush_admin().unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
        assert_eq!(e.metrics().commits_total, 1);
    }

    #[test]
    fn execute_text_non_mutations_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::new_local();

        e.execute_text(1, "BEGIN").unwrap();
        e.execute_text(1, "COMMIT").unwrap();
        e.execute_text(2, "BEGIN").unwrap();
        e.execute_text(2, "ROLLBACK").unwrap();
        e.execute_text(3, "GET missing").unwrap();
        e.execute_text(4, "FLUSH").unwrap();
        e.execute_text(5, "RESET ALL").unwrap();
        e.execute_text(6, "DISCARD TEMP").unwrap();

        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 8);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 8);
        assert_eq!(
            e.metrics().last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_non_mutations_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "BEGIN", t0).unwrap();
        e.enqueue_set_text(1, "COMMIT", t0).unwrap();
        e.enqueue_set_text(2, "BEGIN", t0).unwrap();
        e.enqueue_set_text(2, "ROLLBACK", t0).unwrap();
        e.enqueue_set_text(3, "GET missing", t0).unwrap();
        e.enqueue_set_text(4, "FLUSH", t0).unwrap();
        e.enqueue_set_text(5, "RESET ALL", t0).unwrap();
        e.enqueue_set_text(6, "DISCARD TEMP", t0).unwrap();

        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 8);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 8);
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn commit_and_rollback_require_active_transaction_context() {
        let mut e = Engine::new_local();

        let commit_err = e.execute_text(10, "COMMIT").unwrap_err();
        assert!(matches!(
            commit_err,
            ExecuteError::Txn(TxnError::NotFound(10))
        ));

        let rollback_err = e.execute_text(11, "ROLLBACK").unwrap_err();
        assert!(matches!(
            rollback_err,
            ExecuteError::Txn(TxnError::NotFound(11))
        ));

        e.execute_text(12, "BEGIN").unwrap();
        let duplicate_begin_err = e.execute_text(12, "BEGIN").unwrap_err();
        assert!(matches!(
            duplicate_begin_err,
            ExecuteError::Txn(TxnError::AlreadyExists(12))
        ));

        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.active_txn_count(), 1);
    }

    #[test]
    fn and_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();

        e.execute_text(21, "BEGIN").unwrap();
        e.execute_text(21, "COMMIT AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(22, "COMMIT").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(31, "BEGIN").unwrap();
        e.execute_text(31, "ROLLBACK AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(32, "ROLLBACK").unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn enqueue_non_mutation_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();
        let t0 = Instant::now();

        e.enqueue_set_text(41, "BEGIN", t0).unwrap();
        e.enqueue_set_text(41, "COMMIT AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(42, "COMMIT", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(51, "BEGIN", t0).unwrap();
        e.enqueue_set_text(51, "ROLLBACK AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(52, "ROLLBACK", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn transaction_control_alias_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();

        e.execute_text(61, "BEGIN").unwrap();
        e.execute_text(61, "END AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(62, "END").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(71, "BEGIN").unwrap();
        e.execute_text(71, "ABORT AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(72, "ABORT").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(75, "BEGIN").unwrap();
        e.execute_text(75, "END WORK AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(76, "COMMIT").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(77, "BEGIN").unwrap();
        e.execute_text(77, "ABORT WORK AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(78, "ROLLBACK").unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn start_alias_and_work_aliases_drive_transaction_state_transitions() {
        let mut e = Engine::new_local();

        e.execute_text(73, "START TRANSACTION READ ONLY").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(73, "COMMIT WORK").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(74, "START WORK, READ WRITE, DEFERRABLE")
            .unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(74, "ROLLBACK TRANSACTION").unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn enqueue_transaction_control_alias_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();
        let t0 = Instant::now();

        e.enqueue_set_text(81, "BEGIN", t0).unwrap();
        e.enqueue_set_text(81, "END AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(82, "END", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(91, "BEGIN", t0).unwrap();
        e.enqueue_set_text(91, "ABORT AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(92, "ABORT", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(95, "BEGIN", t0).unwrap();
        e.enqueue_set_text(95, "END WORK AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(96, "COMMIT", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(97, "BEGIN", t0).unwrap();
        e.enqueue_set_text(97, "ABORT WORK AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(98, "ROLLBACK", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn enqueue_start_alias_and_work_aliases_drive_transaction_state_transitions() {
        let mut e = Engine::new_local();
        let t0 = Instant::now();

        e.enqueue_set_text(93, "START TRANSACTION READ ONLY", t0)
            .unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(93, "COMMIT WORK", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(94, "START WORK, READ WRITE, DEFERRABLE", t0)
            .unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(94, "ROLLBACK TRANSACTION", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn commit_and_chain_propagates_txn_id_exhaustion() {
        let mut e = Engine::new_local();

        e.execute_text(u64::MAX, "BEGIN").unwrap();
        let err = e.execute_text(u64::MAX, "COMMIT AND CHAIN").unwrap_err();

        assert!(matches!(err, ExecuteError::Txn(TxnError::IdExhausted)));
        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 1);
    }

    #[test]
    fn replication_watermarks_track_commit_apply_visibility_and_durability() {
        let mut e = Engine::new_local();

        let before = e.replication_watermarks();
        assert_eq!(before.role, Role::Leader);
        assert_eq!(before.commit_index, 0);
        assert_eq!(before.applied_index, 0);
        assert_eq!(before.visible_index, 0);
        assert_eq!(before.commit_apply_gap, 0);
        assert_eq!(before.apply_visible_gap, 0);
        assert_eq!(before.snapshot_id, 0);
        assert_eq!(before.wal_flushed_count, 0);
        assert_eq!(before.wal_buffered_count, 0);
        assert_eq!(before.wal_unflushed_count, 0);
        assert_eq!(before.pending_batch_len, 0);
        assert_eq!(before.pending_batch_cap, 64);
        assert_eq!(before.pending_batch_remaining_capacity, 64);
        assert_eq!(before.pending_batch_utilization_permyriad, 0);
        assert_eq!(before.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(before.pending_batch_oldest_age_ms, None);
        assert_eq!(before.pending_batch_time_until_deadline_ms, None);
        assert_eq!(before.active_txn_count, 0);
        assert_eq!(before.oldest_active_txn_id, None);
        assert_eq!(before.newest_active_txn_id, None);
        assert!(!before.has_wal_backlog);
        assert!(!before.has_pending_batch_backlog);
        assert!(!before.has_active_txn_backlog);
        assert!(!before.has_commit_apply_gap);
        assert!(!before.has_apply_visible_gap);
        assert!(!before.has_backlog_blockers);
        assert_eq!(before.backlog_blocker_count, 0);
        assert_eq!(before.backlog_blocker_mask, 0);
        assert!(!before.mutation_admission_saturated);
        assert!(before.quiescent_for_failover);
        assert!(!before.follower_promotion_ready);

        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let after = e.replication_watermarks();

        assert_eq!(after.role, Role::Leader);
        assert!(after.term >= before.term);
        assert_eq!(after.commit_index, token.index);
        assert_eq!(after.applied_index, token.index);
        assert_eq!(after.visible_index, token.index);
        assert_eq!(after.commit_apply_gap, 0);
        assert_eq!(after.apply_visible_gap, 0);
        assert_eq!(after.snapshot_id, 0);
        assert!(after.wal_flushed_count >= 1);
        assert_eq!(after.wal_buffered_count, e.wal_buffered_count());
        assert_eq!(after.wal_unflushed_count, e.wal_unflushed_count());
        assert_eq!(after.pending_batch_len, 0);
        assert_eq!(after.pending_batch_cap, 64);
        assert_eq!(after.pending_batch_remaining_capacity, 64);
        assert_eq!(after.pending_batch_utilization_permyriad, 0);
        assert_eq!(after.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(after.pending_batch_oldest_age_ms, None);
        assert_eq!(after.pending_batch_time_until_deadline_ms, None);
        assert_eq!(after.active_txn_count, 0);
        assert_eq!(after.oldest_active_txn_id, None);
        assert_eq!(after.newest_active_txn_id, None);
        assert!(!after.has_wal_backlog);
        assert!(!after.has_pending_batch_backlog);
        assert!(!after.has_active_txn_backlog);
        assert!(!after.has_commit_apply_gap);
        assert!(!after.has_apply_visible_gap);
        assert!(!after.has_backlog_blockers);
        assert_eq!(after.backlog_blocker_count, 0);
        assert_eq!(after.backlog_blocker_mask, 0);
        assert!(!after.mutation_admission_saturated);
        assert!(after.quiescent_for_failover);
        assert!(!after.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_do_not_advance_on_rejected_follower_commit() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));

        let marks = e.replication_watermarks();
        assert_eq!(marks.role, Role::Follower);
        assert_eq!(marks.term, 2);
        assert_eq!(marks.commit_index, 0);
        assert_eq!(marks.applied_index, 0);
        assert_eq!(marks.visible_index, 0);
        assert_eq!(marks.commit_apply_gap, 0);
        assert_eq!(marks.apply_visible_gap, 0);
        assert_eq!(marks.wal_flushed_count, 0);
        assert_eq!(marks.wal_last_durable_txn_id, None);
        assert_eq!(marks.wal_buffered_count, 0);
        assert_eq!(marks.wal_unflushed_count, 0);
        assert_eq!(marks.pending_batch_len, 0);
        assert_eq!(marks.pending_batch_cap, 64);
        assert_eq!(marks.pending_batch_remaining_capacity, 64);
        assert_eq!(marks.pending_batch_utilization_permyriad, 0);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(marks.pending_batch_oldest_age_ms, None);
        assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
        assert_eq!(marks.active_txn_count, 0);
        assert!(!marks.has_wal_backlog);
        assert!(!marks.has_pending_batch_backlog);
        assert!(!marks.has_active_txn_backlog);
        assert!(!marks.has_commit_apply_gap);
        assert!(!marks.has_apply_visible_gap);
        assert!(!marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_mask, 0);
        assert!(!marks.mutation_admission_saturated);
        assert!(!marks.quiescent_for_failover);
        assert!(marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_include_buffered_wal_records() {
        let mut e = Engine::new_local();

        e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.wal_buffered_count, 2);
        assert_eq!(marks.wal_flushed_count, 2);
        assert_eq!(marks.wal_last_durable_txn_id, Some(2));
        assert_eq!(marks.wal_unflushed_count, 0);
        assert_eq!(marks.pending_batch_len, 0);
        assert_eq!(marks.pending_batch_cap, 64);
        assert_eq!(marks.pending_batch_remaining_capacity, 64);
        assert_eq!(marks.pending_batch_utilization_permyriad, 0);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(marks.pending_batch_oldest_age_ms, None);
        assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
        assert_eq!(marks.active_txn_count, 0);
        assert!(!marks.mutation_admission_saturated);
        assert!(marks.quiescent_for_failover);
    }

    #[test]
    fn replication_watermarks_pending_batch_time_fields_clear_after_flush() {
        let mut e = Engine::with_batching(3, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        let before_flush = e.replication_watermarks();
        assert_eq!(before_flush.pending_batch_len, 1);
        assert_eq!(before_flush.pending_batch_cap, 3);
        assert_eq!(before_flush.pending_batch_remaining_capacity, 2);
        assert_eq!(before_flush.pending_batch_utilization_permyriad, 3_333);
        assert_eq!(
            before_flush.pending_batch_remaining_capacity_permyriad,
            6_667
        );
        assert!(before_flush.pending_batch_oldest_age_ms.is_some());
        assert!(before_flush.pending_batch_time_until_deadline_ms.is_some());

        e.flush_admin().unwrap();
        let after_flush = e.replication_watermarks();
        assert_eq!(after_flush.pending_batch_len, 0);
        assert_eq!(after_flush.pending_batch_cap, 3);
        assert_eq!(after_flush.pending_batch_remaining_capacity, 3);
        assert_eq!(after_flush.pending_batch_utilization_permyriad, 0);
        assert_eq!(
            after_flush.pending_batch_remaining_capacity_permyriad,
            10_000
        );
        assert_eq!(after_flush.pending_batch_oldest_age_ms, None);
        assert_eq!(after_flush.pending_batch_time_until_deadline_ms, None);
    }

    #[test]
    fn replication_watermarks_include_pending_batch_depth() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.pending_batch_len, 1);
        assert_eq!(marks.pending_batch_cap, 2);
        assert_eq!(marks.pending_batch_remaining_capacity, 1);
        assert_eq!(marks.pending_batch_utilization_permyriad, 5_000);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 5_000);
        assert!(marks.pending_batch_oldest_age_ms.is_some());
        assert!(marks.pending_batch_time_until_deadline_ms.is_some());
        assert!(marks.has_pending_batch_backlog);
        assert!(marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_count, 1);
        assert_eq!(
            marks.backlog_blocker_mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
        );
        assert!(!marks.has_wal_backlog);
        assert!(!marks.has_active_txn_backlog);
        assert!(!marks.has_commit_apply_gap);
        assert!(!marks.has_apply_visible_gap);
        assert_eq!(marks.wal_buffered_count, 0);
        assert_eq!(marks.wal_unflushed_count, 0);
        assert_eq!(marks.active_txn_count, 0);
        assert!(!marks.quiescent_for_failover);
    }

    #[test]
    fn replication_watermarks_flag_mutation_admission_saturation() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.simulate_next_wal_flush_failure();
        let err = e
            .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
            .unwrap_err();
        assert!(matches!(
            err,
            ExecuteError::Engine(EngineError::Durability(_))
        ));

        let marks = e.replication_watermarks();
        assert_eq!(marks.pending_batch_len, 2);
        assert_eq!(marks.pending_batch_cap, 2);
        assert_eq!(marks.pending_batch_remaining_capacity, 0);
        assert_eq!(marks.pending_batch_utilization_permyriad, 10_000);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 0);
        assert!(marks.has_pending_batch_backlog);
        assert!(!marks.has_wal_backlog);
        assert!(!marks.has_active_txn_backlog);
        assert!(marks.mutation_admission_saturated);
        assert!(!marks.quiescent_for_failover);
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_follower_promotion_ready_requires_no_backlog() {
        let mut e = Engine::with_batching(8, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(3);

        let marks = e.replication_watermarks();
        assert_eq!(marks.role, Role::Follower);
        assert_eq!(marks.pending_batch_len, 1);
        assert!(!marks.quiescent_for_failover);
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_follower_promotion_ready_requires_zero_active_txns() {
        let mut e = Engine::new_local();
        e.become_follower(5);
        e.execute_text(9, "BEGIN").unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.role, Role::Follower);
        assert_eq!(marks.active_txn_count, 1);
        assert_eq!(marks.oldest_active_txn_id, Some(9));
        assert_eq!(marks.newest_active_txn_id, Some(9));
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_include_active_transaction_count() {
        let mut e = Engine::new_local();

        e.execute_text(42, "BEGIN").unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.active_txn_count, 1);
        assert_eq!(marks.oldest_active_txn_id, Some(42));
        assert_eq!(marks.newest_active_txn_id, Some(42));
        assert_eq!(marks.pending_batch_len, 0);
        assert_eq!(marks.pending_batch_oldest_age_ms, None);
        assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
        assert!(marks.has_active_txn_backlog);
        assert!(marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_count, 1);
        assert_eq!(
            marks.backlog_blocker_mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        );
        assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
        assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
        assert!(!marks.has_pending_batch_backlog);
        assert!(!marks.has_wal_backlog);
        assert_eq!(marks.wal_buffered_count, 0);
        assert!(!marks.mutation_admission_saturated);
        assert!(!marks.quiescent_for_failover);
    }

    #[test]
    fn backlog_blocker_enum_roundtrips_through_bits_and_labels() {
        for blocker in BacklogBlocker::ALL {
            assert!(blocker.bit().is_power_of_two());
            assert!(!blocker.as_str().is_empty());
            assert_eq!(BacklogBlocker::from_bit(blocker.bit()), Some(blocker));
            assert_eq!(BacklogBlocker::from_label(blocker.as_str()), Some(blocker));

            let mut marks = Engine::new_local().replication_watermarks();
            marks.backlog_blocker_mask = blocker.bit();
            assert!(marks.has_blocker_kind(blocker));
            assert_eq!(marks.backlog_blockers().collect::<Vec<_>>(), vec![blocker]);
            assert_eq!(
                marks.backlog_blocker_labels().collect::<Vec<_>>(),
                vec![blocker.as_str()]
            );
            assert_eq!(
                marks.backlog_blocker_bits().collect::<Vec<_>>(),
                vec![blocker.bit()]
            );
            assert_eq!(
                ReplicationWatermarks::backlog_blockers_from_mask(marks.backlog_blocker_mask)
                    .collect::<Vec<_>>(),
                vec![blocker]
            );
        }
    }

    #[test]
    fn backlog_blocker_display_and_from_str_roundtrip() {
        for blocker in BacklogBlocker::ALL {
            let label = blocker.to_string();
            assert_eq!(label, blocker.as_str());
            assert_eq!(label.parse::<BacklogBlocker>(), Ok(blocker));
        }
    }

    #[test]
    fn backlog_blocker_from_str_reports_unknown_label() {
        let err = "  not-a-real-blocker  "
            .parse::<BacklogBlocker>()
            .expect_err("unknown blocker labels should fail to parse");

        assert_eq!(err.label(), "not-a-real-blocker");
        assert_eq!(
            err.to_string(),
            "unknown backlog blocker label: not-a-real-blocker"
        );
    }

    #[test]
    fn replication_watermarks_backlog_blockers_from_mask_ignores_unknown_bits() {
        let known_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN;
        let unknown_mask = 1 << 7;

        assert_eq!(BacklogBlocker::from_bit(unknown_mask), None);
        assert_eq!(
            ReplicationWatermarks::backlog_blockers_from_mask(known_mask | unknown_mask)
                .collect::<Vec<_>>(),
            vec![BacklogBlocker::Wal, BacklogBlocker::ActiveTxn]
        );
    }

    #[test]
    fn backlog_blocker_mask_helpers_strip_unknown_bits() {
        let unknown_mask = (1 << 5) | (1 << 7);
        let mixed_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | unknown_mask;

        assert_eq!(
            ReplicationWatermarks::known_backlog_blocker_mask(),
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
        assert_eq!(
            ReplicationWatermarks::unknown_backlog_blocker_mask(mixed_mask),
            unknown_mask
        );
        assert_eq!(
            ReplicationWatermarks::sanitize_backlog_blocker_mask(mixed_mask),
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blocker_count_from_mask(mixed_mask),
            2
        );
        assert!(ReplicationWatermarks::has_backlog_blockers_in_mask(
            mixed_mask
        ));
        assert!(!ReplicationWatermarks::has_backlog_blockers_in_mask(
            unknown_mask
        ));
    }

    #[test]
    fn replication_watermarks_backlog_blocker_mask_from_labels_ignores_unknowns() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
            "pending_batch",
            "unknown",
            "active_txn",
            "pending_batch",
        ]);

        assert_eq!(BacklogBlocker::from_label("unknown"), None);
        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blockers_from_mask(mask).collect::<Vec<_>>(),
            vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blocker_labels_from_mask(mask).collect::<Vec<_>>(),
            vec!["pending_batch", "active_txn"]
        );
    }

    #[test]
    fn backlog_blocker_label_decode_normalizes_case_spacing_and_hyphenation() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
            " WAL ",
            "pending-batch",
            "ACTIVE TXN",
            "commit-apply-gap",
            "apply visible gap",
            "pending.batch",
            "commit--apply  gap",
        ]);

        assert_eq!(
            BacklogBlocker::from_label("PENDING-BATCH"),
            Some(BacklogBlocker::PendingBatch)
        );
        assert_eq!(
            BacklogBlocker::from_label("apply visible gap"),
            Some(BacklogBlocker::ApplyVisibleGap)
        );
        assert_eq!(
            BacklogBlocker::from_label("pending.batch"),
            Some(BacklogBlocker::PendingBatch)
        );
        assert_eq!(
            BacklogBlocker::from_label("commit--apply  gap"),
            Some(BacklogBlocker::CommitApplyGap)
        );
        assert_eq!(
            BacklogBlocker::from_label("__wal__"),
            Some(BacklogBlocker::Wal)
        );
        assert_eq!(
            BacklogBlocker::from_label("___active.txn___"),
            Some(BacklogBlocker::ActiveTxn)
        );
        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_decodes_csv_like_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal, pending-batch; ACTIVE TXN | unknown / apply visible gap : commit apply gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_ignores_empty_segments() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            " , ; | pending_batch || wal ,, ",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_multiline_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal\n pending_batch\r\nACTIVE TXN\t| commit apply gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_jsonish_arrays() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "[\"wal\",\"active_txn\",\"apply visible gap\"]",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_braced_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "{'wal';'active_txn';'apply visible gap'}",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_parenthesized_and_angle_bracket_streams()
    {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "<(wal|active_txn|apply visible gap)>",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_single_quoted_arrays() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "['pending-batch','commit_apply_gap']",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_backtick_quoted_labels() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "`wal`,`active_txn`,`apply_visible_gap`",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_plus_delimiter() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal+active_txn+apply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_windows_style_backslash_delimiter() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal\\active_txn\\apply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_assignment_and_ampersand_delimiters() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "backlog_blockers=wal&active_txn&apply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_percent_encoded_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal%2Cpending-batch%7CACTIVE%20TXN%2Fapply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_delimited_labels_from_mask_emits_canonical_order() {
        let mask = ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;

        assert_eq!(
            ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ","),
            "wal,active_txn,apply_visible_gap"
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, " | "),
            "wal | active_txn | apply_visible_gap"
        );
    }

    #[test]
    fn backlog_blocker_delimited_mask_roundtrip_is_stable() {
        let mask = ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP;
        let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ";");

        assert_eq!(
            ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
            mask
        );
    }

    #[test]
    fn backlog_blocker_delimited_mask_roundtrip_is_stable_with_colon_delimiter() {
        let mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;
        let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ":");

        assert_eq!(
            ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
            mask
        );
    }

    #[test]
    fn replication_watermarks_aggregate_multiple_backlog_blockers() {
        let mut e = Engine::with_batching(8, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(7, "SET a=1", t0).unwrap();
        e.execute_text(8, "BEGIN").unwrap();

        let marks = e.replication_watermarks();
        assert!(marks.has_pending_batch_backlog);
        assert!(marks.has_active_txn_backlog);
        assert!(marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_count, 2);
        assert_eq!(
            marks.backlog_blocker_mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        );
        assert_eq!(
            marks.backlog_blocker_count,
            marks.backlog_blocker_mask.count_ones() as u8
        );
        assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH));
        assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
        assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
        assert!(!marks.has_backlog_blocker(1 << 7));
        assert_eq!(
            marks.backlog_blockers().collect::<Vec<_>>(),
            vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
        );
        assert_eq!(
            marks.backlog_blocker_labels().collect::<Vec<_>>(),
            vec!["pending_batch", "active_txn"]
        );
        assert_eq!(
            marks.backlog_blocker_bits().collect::<Vec<_>>(),
            vec![
                ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH,
                ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            ]
        );
        assert_eq!(marks.max_replication_gap(), 0);
        assert_eq!(marks.total_backlog_items(), 2);
        assert!(!marks.is_fully_caught_up());
        assert!(!marks.quiescent_for_failover);
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn snapshot_export_tracks_last_applied_index() {
        let mut e = Engine::new_local();
        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

        let exported = e.export_snapshot_meta();
        let current = e.snapshot_meta();

        assert_eq!(exported.last_included_index, token.index);
        assert_eq!(current.last_included_index, token.index);
        assert_eq!(exported.snapshot_id, 1);
        assert_eq!(current.snapshot_id, 1);
    }

    #[test]
    fn install_snapshot_advances_visible_and_replication_watermarks() {
        let mut e = Engine::new_local();
        e.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 3,
            snapshot_id: 11,
        });

        let marks = e.replication_watermarks();
        assert_eq!(marks.term, 3);
        assert_eq!(marks.max_replication_gap(), 0);
        assert_eq!(marks.total_backlog_items(), 0);
        assert!(marks.is_fully_caught_up());
        assert_eq!(marks.commit_index, 7);
        assert_eq!(marks.applied_index, 7);
        assert_eq!(marks.visible_index, 7);
        assert_eq!(marks.commit_apply_gap, 0);
        assert_eq!(marks.apply_visible_gap, 0);
        assert_eq!(marks.snapshot_id, 11);

        let next = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
        assert_eq!(next.index, 8);
        assert_eq!(e.get("b"), Some("2"));
    }

    #[test]
    fn export_snapshot_meta_is_reflected_in_replication_watermarks() {
        let mut e = Engine::new_local();

        assert_eq!(e.replication_watermarks().snapshot_id, 0);

        e.export_snapshot_meta();
        assert_eq!(e.replication_watermarks().snapshot_id, 1);

        e.export_snapshot_meta();
        assert_eq!(e.replication_watermarks().snapshot_id, 2);
    }

    #[test]
    fn telemetry_snapshot_reflects_replication_lag_and_runtime_metrics() {
        let mut e = Engine::with_batching(8, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();

        let snapshot = e.telemetry_snapshot();

        assert_eq!(snapshot.role, Role::Leader);
        assert_eq!(snapshot.replication_lag.commit_index, 0);
        assert_eq!(snapshot.replication_lag.applied_index, 0);
        assert_eq!(snapshot.replication_lag.visible_index, 0);
        assert_eq!(snapshot.replication_lag.commit_apply_gap, 0);
        assert_eq!(snapshot.replication_lag.apply_visible_gap, 0);
        assert_eq!(snapshot.runtime_metrics.pending_batch_peak, 1);
        assert_eq!(snapshot.runtime_metrics.last_pending_batch_len, Some(1));
        assert_eq!(snapshot.runtime_metrics.commits_total, 0);
        assert_eq!(snapshot.snapshot_id, 0);
        assert_eq!(snapshot.wal_flushed_count, 0);
        assert_eq!(snapshot.wal_last_durable_txn_id, None);
        assert_eq!(snapshot.wal_buffered_count, 0);
        assert_eq!(snapshot.wal_unflushed_count, 0);
        assert_eq!(snapshot.pending_batch_len, 1);
        assert_eq!(snapshot.pending_batch_cap, 8);
        assert_eq!(snapshot.active_txn_count, 0);
        assert_eq!(snapshot.backlog_blocker_count, 1);
        assert!(snapshot.has_backlog_blockers());
        assert!(!snapshot.quiescent_for_failover);
        assert!(!snapshot.mutation_admission_saturated);
        assert!(snapshot.gpu_parity_fallbacks.is_empty());
    }

    #[test]
    fn status_snapshot_answers_snapshot_and_replication_health_questions() {
        let mut e = Engine::new_local();
        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let exported = e.export_snapshot_meta();

        let status = e.status_snapshot();

        assert_eq!(status.role, Role::Leader);
        assert_eq!(status.term, 1);
        assert_eq!(status.snapshot.snapshot_id, exported.snapshot_id);
        assert_eq!(status.snapshot.last_included_index, token.index);
        assert_eq!(status.snapshot.last_included_term, 1);
        assert_eq!(status.snapshot.visible_index, token.index);
        assert_eq!(status.served_snapshot_frontier(), token.index);
        assert_eq!(status.replication_lag.commit_index, token.index);
        assert_eq!(status.replication_lag.applied_index, token.index);
        assert_eq!(status.replication_distance(), 0);
        assert!(status.why_routed_to_fallback_labels().is_empty());
        assert_eq!(status.latest_fallback_reason(), None);
        assert_eq!(status.backlog_blocker_labels(), Vec::<&'static str>::new());
        status.validate().unwrap();
    }

    #[test]
    fn status_snapshot_surfaces_active_fallback_reasons_and_rollups() {
        let mut e = Engine::new_local();
        e.mark_gpu_unavailable(0);
        e.set_gpu_runtime_saturated(true);

        e.execute_text(1, "SET a=1").unwrap();

        let status = e.status_snapshot();

        assert_eq!(
            status.latest_fallback_reason(),
            Some(FallbackReason::GpuUnavailable)
        );
        assert_eq!(
            status.why_routed_to_fallback_labels(),
            vec!["gpu_unavailable", "gpu_queue_saturated"]
        );
        assert!(status.fallback.is_actively_degraded());
        assert!(status.fallback.has_gpu_parity_fallbacks());
        assert_eq!(status.fallback.gpu_parity_fallback_total(), 1);
        assert_eq!(
            status.fallback.active_reasons,
            vec![
                ActiveFallbackReason::GpuUnavailable { gpu_ids: vec![0] },
                ActiveFallbackReason::GpuQueueSaturated,
            ]
        );
        status.validate().unwrap();
    }

    #[test]
    fn execute_mvcc_query_runs_visibility_filtered_scan_through_execution_layer() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=open").unwrap();
        e.execute_text(2, "SET acct:2=pending").unwrap();
        e.execute_text(3, "SET acct:1=closed").unwrap();
        e.execute_text(4, "DELETE acct:2").unwrap();

        let result = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            })
            .unwrap();

        assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(result.executed_target, DeviceTarget::Cpu);
        assert_eq!(
            result.fallback_reason,
            Some(FallbackReason::GpuMvccReadParityGap)
        );
        assert_eq!(
            result.rows,
            vec![
                MvccReadRow {
                    key: Some("acct:1".to_string()),
                    value: Some("closed".to_string()),
                },
                MvccReadRow {
                    key: Some("acct:2".to_string()),
                    value: Some("pending".to_string()),
                },
            ]
        );
        assert_eq!(
            e.metrics()
                .fallback_for(FallbackReason::GpuMvccReadParityGap),
            1
        );
    }

    #[test]
    fn execute_mvcc_query_replays_deterministic_workload_fixture_for_point_lookup() {
        let mut e = Engine::new_local();
        for (txn_id, command) in include_str!("../../../tests/fixtures/mvcc-read-workload.txt")
            .lines()
            .enumerate()
        {
            e.execute_text((txn_id + 1) as u64, command).unwrap();
        }

        let historical = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::ValueOnly,
                limit: None,
            })
            .unwrap();

        assert_eq!(
            historical.rows,
            vec![MvccReadRow {
                key: None,
                value: Some("open".to_string()),
            }]
        );

        let current = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "user:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 5 },
                filter: Some(MvccReadFilter::ValueEquals("active".to_string())),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert_eq!(
            current.rows,
            vec![MvccReadRow {
                key: Some("user:1".to_string()),
                value: None,
            }]
        );
    }

    #[test]
    fn execute_mvcc_query_supports_composite_filter_shapes() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=open").unwrap();
        e.execute_text(2, "SET acct:2=locked").unwrap();
        e.execute_text(3, "SET user:1=active").unwrap();
        e.execute_text(4, "SET user:2=locked").unwrap();

        let all_filter = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::All(vec![
                    MvccReadFilter::KeyPrefix("acct:".to_string()),
                    MvccReadFilter::ValueEquals("locked".to_string()),
                ])),
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            })
            .unwrap();
        assert_eq!(
            all_filter.rows,
            vec![MvccReadRow {
                key: Some("acct:2".to_string()),
                value: Some("locked".to_string()),
            }]
        );

        let any_filter = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::Any(vec![
                    MvccReadFilter::KeyPrefix("acct:".to_string()),
                    MvccReadFilter::ValueEquals("active".to_string()),
                ])),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();
        assert_eq!(
            any_filter.rows,
            vec![
                MvccReadRow {
                    key: Some("acct:1".to_string()),
                    value: None,
                },
                MvccReadRow {
                    key: Some("acct:2".to_string()),
                    value: None,
                },
                MvccReadRow {
                    key: Some("user:1".to_string()),
                    value: None,
                },
            ]
        );
    }

    #[test]
    fn execute_mvcc_query_supports_key_range_filter_shapes() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=open").unwrap();
        e.execute_text(2, "SET acct:2=locked").unwrap();
        e.execute_text(3, "SET acct:3=closed").unwrap();
        e.execute_text(4, "SET acct:4=suspended").unwrap();

        let ranged = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::KeyRange {
                    start_inclusive: "acct:2".to_string(),
                    end_exclusive: "acct:4".to_string(),
                }),
                order: Some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::KeyValue,
                limit: None,
            })
            .unwrap();

        assert_eq!(
            ranged.rows,
            vec![
                MvccReadRow {
                    key: Some("acct:2".to_string()),
                    value: Some("locked".to_string()),
                },
                MvccReadRow {
                    key: Some("acct:3".to_string()),
                    value: Some("closed".to_string()),
                },
            ]
        );
    }

    #[test]
    fn execute_mvcc_query_supports_limit_after_filtering() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=open").unwrap();
        e.execute_text(2, "SET acct:2=locked").unwrap();
        e.execute_text(3, "SET acct:3=locked").unwrap();

        let limited = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: Some(2),
            })
            .unwrap();

        assert_eq!(
            limited.rows,
            vec![
                MvccReadRow {
                    key: Some("acct:1".to_string()),
                    value: None,
                },
                MvccReadRow {
                    key: Some("acct:2".to_string()),
                    value: None,
                },
            ]
        );
    }

    #[test]
    fn execute_mvcc_query_supports_key_ordering_before_limit() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:2=locked").unwrap();
        e.execute_text(2, "SET acct:1=open").unwrap();
        e.execute_text(3, "SET acct:3=closed").unwrap();

        let descending = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
                order: Some(MvccReadOrder::KeyDesc),
                projection: MvccProjection::KeyOnly,
                limit: Some(2),
            })
            .unwrap();

        assert_eq!(
            descending.rows,
            vec![
                MvccReadRow {
                    key: Some("acct:3".to_string()),
                    value: None,
                },
                MvccReadRow {
                    key: Some("acct:2".to_string()),
                    value: None,
                },
            ]
        );
    }

    #[test]
    fn publish_telemetry_emits_snapshot_to_sink() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET a=1").unwrap();

        let mut sink = InMemoryTelemetrySink::default();
        e.publish_telemetry(&mut sink);

        assert_eq!(sink.snapshots().len(), 1);
        let snapshot = &sink.snapshots()[0];
        assert_eq!(snapshot.role, Role::Leader);
        assert_eq!(snapshot.replication_lag.commit_index, 1);
        assert_eq!(snapshot.replication_lag.applied_index, 1);
        assert_eq!(snapshot.replication_lag.visible_index, 1);
        assert_eq!(snapshot.replication_lag.commit_apply_gap, 0);
        assert_eq!(snapshot.replication_lag.apply_visible_gap, 0);
        assert_eq!(snapshot.runtime_metrics.commits_total, 1);
        assert_eq!(snapshot.snapshot_id, 0);
        assert_eq!(snapshot.wal_flushed_count, 1);
        assert_eq!(snapshot.wal_last_durable_txn_id, Some(1));
        assert_eq!(snapshot.wal_buffered_count, 1);
        assert_eq!(snapshot.wal_unflushed_count, 0);
        assert_eq!(snapshot.pending_batch_len, 0);
        assert_eq!(snapshot.active_txn_count, 0);
        assert_eq!(snapshot.backlog_blocker_count, 0);
        assert!(!snapshot.has_backlog_blockers());
        assert!(snapshot.quiescent_for_failover);
        assert!(snapshot.gpu_parity_fallbacks.is_empty());
    }

    #[test]
    fn installing_older_snapshot_is_a_status_no_op() {
        let mut e = Engine::new_local();
        let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let baseline = e.status_snapshot();

        e.install_snapshot(SnapshotMeta {
            last_included_index: committed.index.saturating_sub(1),
            last_included_term: 1,
            snapshot_id: 99,
        });

        let marks = e.replication_watermarks();
        assert_eq!(marks.commit_index, committed.index);
        assert_eq!(marks.applied_index, committed.index);
        assert_eq!(marks.visible_index, committed.index);
        assert_eq!(marks.snapshot_id, baseline.snapshot.snapshot_id);
        assert_eq!(e.status_snapshot(), baseline);
        assert_eq!(e.visible_up_to(), committed.index);
        assert_eq!(e.get("a"), Some("1"));
    }

    #[test]
    fn installing_higher_index_lower_term_snapshot_is_a_status_no_op() {
        let mut e = Engine::new_local();
        let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

        e.install_snapshot(SnapshotMeta {
            last_included_index: committed.index + 2,
            last_included_term: 3,
            snapshot_id: 11,
        });
        let baseline = e.status_snapshot();
        let baseline_marks = e.replication_watermarks();

        e.install_snapshot(SnapshotMeta {
            last_included_index: committed.index + 3,
            last_included_term: 2,
            snapshot_id: 99,
        });

        let marks = e.replication_watermarks();
        assert_eq!(marks, baseline_marks);
        assert_eq!(e.status_snapshot(), baseline);
        assert_eq!(e.visible_up_to(), baseline.snapshot.visible_index);
        assert_eq!(e.get("a"), Some("1"));
    }
}
