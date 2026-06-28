//! Engine control-plane state types (P0 §9.6 decomposition, behavior-preserving):
//! replication backlog/watermarks (BacklogBlocker, ReplicationWatermarks), the
//! opaque read-state handle (ReadState) and the bounded catalog-generation ring
//! (CatalogHistory/CatalogSnapshot), plus the residency read-state and route
//! telemetry. Small state aggregates the Engine owns; not the commit-critical core.

use super::*;

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
    pub(crate) label: String,
}

impl ParseBacklogBlockerError {
    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn unknown(label: &str) -> Self {
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

pub(crate) fn percent_decode_lossy(input: &str) -> String {
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

/// The engine's lock-free read-path state, shared by value-Arc between the owning [`Engine`] and the
/// concurrent-dispatch façade (`SharedEngine`) so reads and the concurrent-DML path reach it WITHOUT
/// taking the engine `RwLock` (lock-free read path, write-half MVCC).
///
/// Every field is interior-mutable (atomic / [`SnapshotCell`] / [`arc_swap::ArcSwap`] / [`Mutex`]),
/// so a holder of `&Engine` (a lock-free reader) and a holder of `&mut Engine` (serialized DDL) only
/// ever obtain `&ReadState` through the shared `Arc` and never alias the same byte mutably. The
/// engine `RwLock`'s remaining job is purely the **catalog latch** (DDL / KV / sequential-INSERT
/// mutual exclusion); the read state below is published to lock-free readers via the
/// `committed_seq`-last release-store discipline (see `publish_committed_seq`).
///
/// Stage 1 holds the already-lock-free fields (`mvcc`, `committed_seq`, the resident device-memory
/// maps, route telemetry). The catalog (Stage 2) and the resident snapshot/shard metadata
/// (Stage 3) move in behind `ArcSwap` later.
///
/// Public as an **opaque** type (all fields private): the concurrent-dispatch façade holds an
/// `Arc<ReadState>` (Stage 4) to drive the lock-free read path without naming the engine's internals.
pub struct ReadState {
    // A bounded RING of recent catalog generations ordered by `commit_seq` (PART B: catalog↔data
    // co-pinning). DDL edits a *working* copy of the catalog maps under the catalog latch and then
    // pushes a fresh stamped generation here; lock-free readers + the off-latch concurrent-DML preflight
    // select the generation AS-OF their pinned `committed_seq` via [`ReadState::catalog_as_of`], so a
    // shape-changing DDL committed concurrently with a reader can never split the reader's (catalog,
    // data) pair. Behind an `ArcSwap<Arc<CatalogHistory>>` so the push (COW: clone the small Vec of
    // `Arc` pointers, append, prune) is wait-free for readers. DDL is rare + serialized so the ring is
    // tiny (pruned below the oldest active read snapshot in the commit critical section).
    pub(crate) catalog_history: ArcSwap<CatalogHistory>,
    // Versioned, publish-on-commit MVCC data: per-table `SnapshotCell<Arc<TableVersionData>>`
    // (rows + value-index) + a KV partition, `&self`-readable / lock-free (write-half Stage 3).
    pub(crate) mvcc: MvccData,
    // The single MVCC visibility/publish boundary (the highest committed `commit_seq`), as an atomic
    // so the concurrent commit critical section can bump it via `&self` (release-store, LAST — the
    // publish point) while lock-free readers acquire-load it once per statement (write-half Stage 4).
    pub(crate) committed_seq: AtomicU64,
    // GPU-resident read-route metadata (device memory + — from Stage 3 — snapshot/shard maps).
    pub(crate) residency: ResidencyReadState,
    // Per-execution route telemetry the read path records through `&self` (Mutex + a test counter).
    pub(crate) route_telemetry: RouteTelemetry,
}

impl ReadState {
    pub(crate) fn new() -> Self {
        Self {
            catalog_history: ArcSwap::new(Arc::new(CatalogHistory::initial())),
            mvcc: MvccData::new(),
            committed_seq: AtomicU64::new(0),
            residency: ResidencyReadState::default(),
            route_telemetry: RouteTelemetry::default(),
        }
    }

    /// The catalog generation visible AS-OF the boundary `s` (the greatest `commit_seq <= s`), as an
    /// owned `Arc` pinned for the statement (PART B). A reader loads its `committed_seq = s` ONCE, then
    /// binds + pins data at THAT same `s`, so the catalog it binds against and the data it reads are
    /// the same generation — a concurrent shape-changing DDL can never split them. Falls back to the
    /// oldest retained generation if `s` predates the ring (only possible if a generation a reader
    /// could still need was pruned, which the oldest-active-snapshot prune boundary prevents).
    pub(crate) fn catalog_as_of(&self, s: Index) -> Arc<CatalogSnapshot> {
        self.catalog_history.load().as_of(s)
    }

    /// The most-recently-published catalog generation (greatest `commit_seq`), owned. Used by paths
    /// that legitimately want "latest" rather than a pinned boundary (e.g. a freshly-built engine's
    /// constructor-time reads, or admin introspection that is not snapshot-pinned).
    pub(crate) fn latest_catalog(&self) -> Arc<CatalogSnapshot> {
        self.catalog_history.load().latest()
    }
}

/// The minimum number of recent catalog generations the ring ALWAYS retains, so a lock-free reader
/// that pinned a `committed_seq` boundary without registering an active snapshot still finds the
/// generation as-of that boundary (a single statement cannot straddle this many serialized DDLs). DDL
/// is rare and the per-generation payload is a clone of the small working catalog maps, so a generous
/// floor is cheap; pruning by the oldest active snapshot still applies on top (whichever keeps more).
pub(crate) const MIN_RETAINED_CATALOG_GENERATIONS: usize = 256;

/// A bounded ring of recent catalog generations ordered by ascending `commit_seq` (PART B). The
/// lock-free read path + off-latch DML preflight select a generation as-of a pinned boundary; the
/// single serialized DDL publisher pushes a new generation and prunes (below the oldest active read
/// snapshot, but always keeping the last [`MIN_RETAINED_CATALOG_GENERATIONS`]). Immutable once
/// published (COW-replaced wholesale via the `ArcSwap`).
#[derive(Debug, Default)]
pub(crate) struct CatalogHistory {
    /// Generations in ascending `commit_seq` order; never empty after `initial()`.
    pub(crate) generations: Vec<Arc<CatalogSnapshot>>,
}

impl CatalogHistory {
    /// The initial history: one empty generation at `commit_seq = 0` whose `public_schema_exists` /
    /// `public_schema_implicit` match a fresh engine's working state (so a reader/preflight before any
    /// DDL sees the correct bootstrap catalog, not the all-`false` `Default`).
    pub(crate) fn initial() -> Self {
        Self {
            generations: vec![Arc::new(CatalogSnapshot {
                commit_seq: 0,
                relational_public_schema_exists: true,
                relational_public_schema_implicit: true,
                ..CatalogSnapshot::default()
            })],
        }
    }

    /// The generation with the greatest `commit_seq <= s`, or the oldest retained generation if `s`
    /// predates the ring. Linear scan from the newest end; the ring is tiny (DDL is rare + serialized).
    pub(crate) fn as_of(&self, s: Index) -> Arc<CatalogSnapshot> {
        for generation in self.generations.iter().rev() {
            if generation.commit_seq <= s {
                return Arc::clone(generation);
            }
        }
        // `s` predates every retained generation (its generation was pruned). Return the oldest we
        // still hold — the prune boundary (oldest active read snapshot) guarantees no in-flight reader
        // pinned at `s` actually needs an older one.
        Arc::clone(self.generations.first().expect("history is never empty"))
    }

    pub(crate) fn latest(&self) -> Arc<CatalogSnapshot> {
        Arc::clone(self.generations.last().expect("history is never empty"))
    }

    /// A new history with `generation` appended (newest), then generations strictly older than the
    /// newest one that is still `<= prune_below` dropped — i.e. keep the single generation a reader
    /// pinned at `prune_below` would select, plus everything newer. `prune_below` is the oldest active
    /// read snapshot's boundary; a `None`/`0` keeps everything but the redundant prefix.
    pub(crate) fn pushed(&self, generation: Arc<CatalogSnapshot>, prune_below: Index) -> Self {
        let mut generations = self.generations.clone();
        generations.push(generation);
        // Two prune bounds; take the one that drops the FEWEST (keeps the most history):
        //   (1) below the oldest active read snapshot — the newest generation still `<= prune_below` is
        //       the one a reader registered at `prune_below` would select; everything strictly before it
        //       is unreachable by any registered snapshot (the concurrent-DML write path registers).
        //   (2) a COUNT floor — ALWAYS retain the last `MIN_RETAINED_CATALOG_GENERATIONS` generations,
        //       so a LOCK-FREE *read* that pinned a boundary WITHOUT registering an active snapshot
        //       (reads stay OFF the `active_snapshots` mutex — the whole point of the lock-free path)
        //       still finds its generation: a single statement cannot straddle that many serialized
        //       DDLs. DDL is rare, so the ring stays tiny under either bound.
        let keep_from_snapshot = generations
            .iter()
            .rposition(|g| g.commit_seq <= prune_below)
            .unwrap_or(0);
        let keep_from_count = generations
            .len()
            .saturating_sub(MIN_RETAINED_CATALOG_GENERATIONS);
        let keep_from = keep_from_snapshot.min(keep_from_count);
        if keep_from > 0 {
            generations.drain(0..keep_from);
        }
        Self { generations }
    }
}

/// The immutable, published slice of the catalog, published as one `Arc` per DDL commit (Stage 2 —
/// blocker #1; lock-free read path, write-half MVCC). The lock-free read path itself consults only the
/// first four maps (tables / views / materialized views / functions); the rest are here so the
/// **off-latch** paths — the concurrent-DML preflight (`preflight_unique_index_constraints` + its
/// helper tree) and the test-only catalog introspection accessors — can read the catalog WITHOUT
/// taking the catalog latch (which would serialize DML behind DDL and re-enter the latch). Because the
/// preflight runs strictly BEFORE any apply and DDL is single-writer, the published snapshot a
/// preflight reads is byte-identical to the working maps it used to read directly. Only the
/// oid/column-id allocators and the resident-cache admission accounting are NOT here — those are
/// touched solely by the under-latch apply path / residency admin. `commit_seq` stamps the commit
/// `Index` this generation was published at (PART B: catalog↔data co-pinning); the read path selects
/// the generation as-of its pinned `committed_seq` so a shape-changing DDL can never split a reader's
/// (catalog, data) pair. The field names mirror the `Engine`/`DdlCatalogState` working maps so the
/// publish is a straight clone-and-store.
#[derive(Debug, Clone, Default)]
pub(crate) struct CatalogSnapshot {
    /// The commit `Index` this catalog generation was published at (PART B). `0` for the initial
    /// empty generation. A reader pinned at `committed_seq = s` selects the generation with the
    /// greatest `commit_seq <= s` (`catalog_as_of`).
    pub(crate) commit_seq: Index,
    pub(crate) relational_catalog: BTreeMap<String, RelationalTable>,
    pub(crate) relational_views: BTreeMap<String, RelationalView>,
    pub(crate) relational_materialized_views: BTreeMap<String, RelationalMaterializedView>,
    pub(crate) relational_functions: BTreeMap<String, RelationalFunction>,
    pub(crate) relational_sequences: BTreeMap<String, RelationalSequence>,
    pub(crate) relational_domains: BTreeMap<String, RelationalDomain>,
    pub(crate) relational_publications: BTreeMap<String, RelationalPublication>,
    pub(crate) relational_subscriptions: BTreeMap<String, RelationalSubscription>,
    pub(crate) relational_roles: BTreeMap<String, RelationalRole>,
    pub(crate) relational_databases: BTreeMap<String, RelationalDatabase>,
    pub(crate) relational_tablespaces: BTreeMap<String, RelationalTablespace>,
    pub(crate) relational_public_schema_exists: bool,
    pub(crate) relational_public_schema_implicit: bool,
    pub(crate) relational_schema_acl: BTreeMap<String, BTreeSet<SchemaPrivilege>>,
    pub(crate) relational_default_table_acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
    pub(crate) relational_comments: BTreeMap<RelationalCommentTarget, String>,
}

/// The GPU-resident read-route metadata reached by the lock-free read path. The device-memory maps
/// are the authoritative residency tombstone gate the concurrent commit path flips via `&self`; the
/// snapshot/shard metadata (Stage 3 — blocker #2) is published behind `ArcSwap` so the resident
/// route can `load()` a guard whose pinned `Arc` outlives the across-kernel-launch read, and the
/// serialized catalog-latch path (warm-up / DDL drop / invalidate / memory-pressure — NEVER the
/// concurrent commit path) mutates it copy-on-write.
#[derive(Debug, Default)]
pub(crate) struct ResidencyReadState {
    pub(crate) device_memory: ResidentDeviceMemoryMap,
    pub(crate) shard_device_memory: ShardResidentDeviceMemoryMap,
    /// ADR-009 R1: per-table GPU hash-index reuse cache for the index-probe point-lookup route (built
    /// lazily, behind the default-OFF `wave_engine_enabled` flag). A plain `Mutex` (not the lock-free
    /// `ArcSwap` the hot path uses) because the index route is opt-in + the lock is taken only off the
    /// fast cache-hit path; staleness is handled by the per-entry `generation` tag, not by eviction.
    pub(crate) wave_index: Mutex<BTreeMap<String, WaveResidentIndex>>,
    /// ADR-009 R2.2b: per-table PERSISTENT wave read engine cache for the persistent-kernel point-lookup
    /// route (built lazily, behind the default-OFF `wave_persistent_engine_enabled` flag nested under
    /// `wave_engine_enabled`). A plain `Mutex` like `wave_index`, BUT each entry owns a live GPU kernel +
    /// watchdog petter, so unlike the passive index — which self-invalidates by ptr on the next read — an
    /// entry MUST be dropped to reclaim its SM/buffers: the serialized-invalidation (catalog-latch) path
    /// removes + drops it on DDL/drop/memory-pressure; a re-admission rebuild overwrites + drops the stale
    /// entry. Staleness on the concurrent commit path is handled by the residency tombstone (which makes the
    /// route unreachable) + ptr-keyed overwrite-on-next-read, NOT by eviction in the commit critical section
    /// (Drop joins the petter ~watchdog window — too costly to hold there).
    pub(crate) wave_read_engine: Mutex<BTreeMap<String, WaveResidentReadEngine>>,
    /// ADR-009 R2.2b: serializes wave-engine BUILDS (not submits) so two concurrent cache misses cannot
    /// both launch a persistent kernel. Critical for the AT-MOST-ONE-RESIDENT invariant: two full-occupancy
    /// persistent spin-kernels in the shared context mutually starve (neither yields its SMs), so the second
    /// launch + the first's later teardown (`cuStreamSynchronize`, infinite backstop) would DEADLOCK. Held
    /// across the drain-existing + launch-new + publish sequence; cache HITS never take it (hot path).
    pub(crate) wave_build_latch: Mutex<()>,
    /// ADR-009 R2.2b: count of batches actually SERVED by the persistent wave route (one per successful
    /// `WaveReadEngine::submit`), as opposed to falling through to the lpb index probe / scan. Real telemetry
    /// for the R2.2b-3 A/B (wave-hit vs fallback rate) AND the test signal that proves the ROUTE (not just the
    /// engine in isolation) produced the rows — output equality alone can't, since all routes are
    /// byte-identical by design. `Relaxed` (a monotonic counter, no ordering dependency).
    pub(crate) wave_route_hits: std::sync::atomic::AtomicU64,
    /// DECISIONS "lpb read levers" #1: count of batches served by the DENSE-emit index probe (vs the atomic
    /// kernel). The test signal that proves the dense route actually ran (output equality alone can't, since
    /// dense and atomic are byte-identical by design). `Relaxed` monotonic counter.
    pub(crate) dense_index_probe_hits: std::sync::atomic::AtomicU64,
    // The per-table resident snapshot metadata + shard metadata, each an immutable published map
    // (Stage 3 — blocker #2). Readers `load()` (wait-free) and pin the `Arc` across the kernel launch;
    // the single serialized publisher COW-stores a fresh map on warm-up / DDL drop / invalidate /
    // memory-pressure.
    pub(crate) snapshots: ArcSwap<BTreeMap<String, RelationalResidencyEntry>>,
    pub(crate) shards: ArcSwap<BTreeMap<String, Vec<RelationalResidentShard>>>,
}

impl ResidencyReadState {
    /// COW-mutate the resident snapshot map under the serialized catalog latch: clone the published
    /// map, apply `mutate`, then atomically store it. In-flight readers keep the generation they
    /// loaded. One publisher (the catalog-latch path), so the load→clone→store is race-free.
    pub(crate) fn with_snapshots_mut<R>(
        &self,
        mutate: impl FnOnce(&mut BTreeMap<String, RelationalResidencyEntry>) -> R,
    ) -> R {
        let mut next = (**self.snapshots.load()).clone();
        let result = mutate(&mut next);
        self.snapshots.store(Arc::new(next));
        result
    }

    /// COW-mutate the resident shard map under the serialized catalog latch (see
    /// [`ResidencyReadState::with_snapshots_mut`]).
    pub(crate) fn with_shards_mut<R>(
        &self,
        mutate: impl FnOnce(&mut BTreeMap<String, Vec<RelationalResidentShard>>) -> R,
    ) -> R {
        let mut next = (**self.shards.load()).clone();
        let result = mutate(&mut next);
        self.shards.store(Arc::new(next));
        result
    }
}

/// Per-execution route telemetry, recorded by the read path through `&self` (the read path writes the
/// latest route decision + measured timings after each query). The `Mutex` recovers from poison so a
/// panicking reader never wedges route recording for everyone.
#[derive(Debug, Default)]
pub(crate) struct RouteTelemetry {
    pub(crate) latest_route_decisions:
        Mutex<BTreeMap<String, RelationalResidentRouteDecisionStatus>>,
    // Test-only counter of `record_route_execution_observation` calls. Lets tests assert a
    // route's telemetry observation is recorded exactly once per execution (guards against the
    // single-predicate mixed int4+text delegation double-counting it).
    #[cfg(test)]
    pub(crate) route_execution_observation_count: std::sync::atomic::AtomicU64,
}

impl RouteTelemetry {
    /// Lock the route-decision map, recovering from poison (a panicking reader must not
    /// wedge route recording for everyone).
    pub(crate) fn route_decisions(
        &self,
    ) -> std::sync::MutexGuard<'_, BTreeMap<String, RelationalResidentRouteDecisionStatus>> {
        self.latest_route_decisions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn record_route_decision(&self, decision: RelationalResidentRouteDecisionStatus) {
        self.route_decisions()
            .insert(decision.table.clone(), decision);
    }

    pub(crate) fn record_route_execution_observation(
        &self,
        table: &str,
        observation: RelationalResidentRouteExecutionObservation,
    ) {
        #[cfg(test)]
        self.route_execution_observation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut decisions = self.route_decisions();
        if let Some(decision) = decisions.get_mut(table) {
            decision.last_execution_h2d_bytes = Some(observation.h2d_bytes);
            decision.last_execution_d2h_bytes = Some(observation.d2h_bytes);
            decision.last_execution_kernel_samples = Some(observation.kernel_samples);
            decision.last_execution_kernel_ms = Some(observation.kernel_ms);
            decision.last_execution_kernel_event_elapsed_us = observation.kernel_event_elapsed_us;
            decision.last_execution_rows = Some(observation.rows);
            decision.last_execution_wall_micros = Some(observation.wall_micros);
        }
    }

    pub(crate) fn record_route_device_lookup_micros(
        &self,
        table: &str,
        elapsed_micros: u64,
        matched_rows: usize,
    ) {
        let mut decisions = self.route_decisions();
        if let Some(decision) = decisions.get_mut(table) {
            decision.last_execution_device_lookup_micros = Some(elapsed_micros);
            decision.last_execution_matched_rows = Some(matched_rows);
        }
    }

    pub(crate) fn record_route_selected_projection_micros(
        &self,
        table: &str,
        match_index_micros: u64,
        selected_projection_micros: u64,
        result_materialization_micros: u64,
        matched_rows: usize,
    ) {
        let mut decisions = self.route_decisions();
        if let Some(decision) = decisions.get_mut(table) {
            decision.last_execution_match_index_micros = Some(match_index_micros);
            decision.last_execution_selected_projection_micros = Some(selected_projection_micros);
            decision.last_execution_result_materialization_micros =
                Some(result_materialization_micros);
            decision.last_execution_matched_rows = Some(matched_rows);
        }
    }

    pub(crate) fn remove_table(&self, table: &str) {
        self.route_decisions().remove(table);
    }
}
