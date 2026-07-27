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
/// `committed_seq`-last release-store discipline (owned by the commit publication coordinator).
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
    /// Monotonic PostgreSQL-compatible non-MVCC rewrite fence keyed by stable table OID.
    /// Readers acquire the table guard before sampling this map; an older retained boundary then
    /// binds the typed empty root instead of traversing the retired generation.
    pub(crate) table_rewrite_fences: ArcSwap<BTreeMap<u32, Index>>,
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
            table_rewrite_fences: ArcSwap::new(Arc::new(BTreeMap::new())),
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

    pub(crate) fn publish_table_rewrite_fences(
        &self,
        table_oids: impl IntoIterator<Item = u32>,
        commit_seq: Index,
    ) {
        let current = self.table_rewrite_fences.load();
        let mut next = (**current).clone();
        for table_oid in table_oids {
            next.entry(table_oid)
                .and_modify(|fence| *fence = (*fence).max(commit_seq))
                .or_insert(commit_seq);
        }
        self.table_rewrite_fences.store(Arc::new(next));
    }

    pub(crate) fn prune_table_rewrite_fences(
        &self,
        live_table_oids: &BTreeSet<u32>,
        safe_boundary: Index,
    ) {
        let current = self.table_rewrite_fences.load();
        if !current
            .iter()
            .any(|(oid, fence)| !live_table_oids.contains(oid) && *fence <= safe_boundary)
        {
            return;
        }
        let mut next = (**current).clone();
        next.retain(|oid, fence| live_table_oids.contains(oid) || *fence > safe_boundary);
        self.table_rewrite_fences.store(Arc::new(next));
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
                relational_next_oid: FIRST_USER_RELATION_OID,
                index_oid_epoch_current: true,
                legacy_recovery_index_oids_assigned: false,
                legacy_recovery_next_index_oid: FIRST_LEGACY_RECOVERY_INDEX_OID,
                relational_next_column_id: FIRST_USER_COLUMN_ID,
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

/// The immutable, published slice of the catalog, published as one stamped `Arc` at each canonical
/// commit (Stage 2 — blocker #1; lock-free read path, write-half MVCC). DDL writes schema/object
/// metadata and allocator high-waters, while typed sequence-value transitions update sequence
/// state. Commits with neither effect republish unchanged contents at a new sequence so data and
/// catalog remain co-pinned. The lock-free read path itself consults only the
/// first four maps (tables / views / materialized views / functions); the rest are here so the
/// **off-latch** paths — the concurrent-DML preflight (`preflight_unique_index_constraints` + its
/// helper tree) and the test-only catalog introspection accessors — can read the catalog WITHOUT
/// taking the catalog latch (which would serialize DML behind DDL and re-enter the latch). Because the
/// preflight runs strictly BEFORE any apply and DDL is single-writer, the published snapshot a
/// preflight reads is byte-identical to the working maps it used to read directly. The OID and
/// column-ID allocators are included as immutable high-water proofs; only resident-cache admission
/// accounting remains outside this snapshot. `commit_seq` stamps the commit `Index` this generation
/// was published at (PART B: catalog↔data co-pinning); the read path selects the generation as-of its
/// pinned `committed_seq` so a shape-changing DDL can never split a reader's (catalog, data) pair.
/// The field names mirror the `Engine`/`DdlCatalogState` working maps so publication is a straight
/// clone-and-store.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
    /// Catalog-stable object identity allocation is part of the immutable generation proof.
    /// Transaction-private DDL may survive a later non-catalog commit only when these high-waters
    /// are unchanged; otherwise replaying the staged command could silently assign different OIDs
    /// or column identities than the private catalog used by later statements.
    pub(crate) relational_next_oid: u32,
    /// One-way stable-index identity migration state is part of the allocator proof. A transaction
    /// may not rebase across this boundary or forget recovery-assigned identities while retaining
    /// an otherwise byte-identical catalog.
    pub(crate) index_oid_epoch_current: bool,
    pub(crate) legacy_recovery_index_oids_assigned: bool,
    pub(crate) legacy_recovery_next_index_oid: u32,
    pub(crate) relational_next_column_id: u32,
}

impl CatalogSnapshot {
    pub(crate) fn pg_class_relation_kind(
        &self,
        name: &str,
    ) -> Result<Option<PgClassRelationKind>, EngineError> {
        resolve_pg_class_relation_kind(
            name,
            self.relational_catalog.contains_key(name),
            self.relational_catalog
                .values()
                .map(|table| {
                    table
                        .indexes
                        .iter()
                        .filter(|index| index.name == name)
                        .count()
                })
                .sum(),
            self.relational_views.contains_key(name),
            self.relational_materialized_views.contains_key(name),
            self.relational_sequences.contains_key(name),
        )
    }

    /// Compare the complete catalog authority while deliberately ignoring only its publication
    /// sequence. Every commit republishes the unchanged working catalog at a newer sequence, so
    /// exact equality would spuriously serialize transaction-private DDL after ordinary DML/KV.
    /// Cloning keeps this proof automatically closed over future snapshot fields through derived
    /// `PartialEq`, rather than maintaining a second field-by-field authority list.
    pub(crate) fn same_contents(&self, other: &Self) -> bool {
        if self.commit_seq == other.commit_seq {
            return self == other;
        }
        let mut normalized = other.clone();
        normalized.commit_seq = self.commit_seq;
        self == &normalized
    }
}

pub(crate) type NamedIndexCoverage = BTreeMap<(String, u32, usize), (u64, usize)>;

/// Host seqlock bracketing in-place device-index mutation. Point readers sample the shared epoch
/// once before and once after a singleton launch; overlap or an older prepared epoch retries through
/// the capacity-bounded posting kernel. Writers are serialized per table by the odd epoch.
pub(crate) struct PointIndexMutationGuard {
    epoch: Arc<std::sync::atomic::AtomicU64>,
}

#[derive(Debug, Default)]
struct TransactionNamedIndexLifecycle {
    publication_active: bool,
    protected_tables: BTreeSet<String>,
    deferred_purges: BTreeSet<String>,
    final_publication_tables: BTreeSet<String>,
    post_publication_deferred_purges: BTreeSet<String>,
}

std::thread_local! {
    static TRANSACTION_NAMED_INDEX_PUBLICATION_OWNER_ACTIVE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Marks the sole canonical apply thread so its own generation-replacement purges bypass the
/// external-retirement deferral window. The marker is thread-local: pressure/retirement callers on
/// any other thread remain deferred and are replayed after final publication.
pub(crate) struct TransactionNamedIndexPublicationOwnerGuard;

impl TransactionNamedIndexPublicationOwnerGuard {
    pub(crate) fn enter() -> Self {
        TRANSACTION_NAMED_INDEX_PUBLICATION_OWNER_ACTIVE.with(|active| {
            debug_assert!(
                !active.replace(true),
                "transaction named-index publication owner cannot nest"
            );
        });
        Self
    }
}

impl Drop for TransactionNamedIndexPublicationOwnerGuard {
    fn drop(&mut self) {
        TRANSACTION_NAMED_INDEX_PUBLICATION_OWNER_ACTIVE.with(|active| {
            debug_assert!(
                active.replace(false),
                "transaction named-index publication owner marker was lost"
            );
        });
    }
}

/// A composite transaction has restored every enrolled index before WAL and is carrying that
/// exact coverage through canonical apply. Cache retirement that races this interval is deferred;
/// the final guard releases and performs those purges only after device publication finishes.
pub(crate) struct TransactionNamedIndexPublicationGuard<'a> {
    residency: &'a ResidencyReadState,
    final_publication_entered: bool,
    finished: bool,
}

impl TransactionNamedIndexPublicationGuard<'_> {
    /// Enter canonical apply's final-publication phase. A purge deferred before this point is
    /// superseded by the pending publication. Every external purge from this point through guard
    /// completion is conservatively replayed afterward, even when it overlaps an intermediate
    /// rebuild; canonical owner's own retirements use the explicit during-transaction bypass.
    pub(crate) fn enter_final_publication(&mut self) {
        self.residency
            .enter_transaction_named_index_final_publication();
        self.final_publication_entered = true;
    }

    /// Complete a successful canonical generation publication. Purges deferred before its final
    /// phase are superseded; final-phase purges retire the published generation. Dropping without
    /// completion applies every request to clean up an aborted preflight or failed apply.
    pub(crate) fn complete(mut self) {
        debug_assert!(
            self.final_publication_entered,
            "successful transaction index publication must enter its final phase"
        );
        self.residency
            .finish_transaction_named_index_publication(true);
        self.finished = true;
    }
}

impl Drop for TransactionNamedIndexPublicationGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.residency
                .finish_transaction_named_index_publication(false);
        }
    }
}

impl Drop for PointIndexMutationGuard {
    fn drop(&mut self) {
        let prior = self
            .epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        debug_assert_eq!(prior & 1, 1, "point-index writer epoch must be odd");
    }
}

/// The GPU-resident read-route metadata reached by the lock-free read path. The device-memory maps
/// are the authoritative residency tombstone gate the concurrent commit path flips via `&self`; the
/// snapshot/shard metadata (Stage 3 — blocker #2) is published behind `ArcSwap` so the resident
/// route can `load()` a guard whose pinned `Arc` outlives the across-kernel-launch read, and the
/// publishers (the serialized catalog-latch paths — warm-up / DDL drop / invalidate /
/// memory-pressure — plus, since W0, the CONCURRENT commit path's invalidation FLAGGING) mutate it
/// copy-on-write, serialized by `descriptor_publish_lock`.
#[derive(Debug, Default)]
pub(crate) struct ResidencyReadState {
    pub(crate) device_memory: ResidentDeviceMemoryMap,
    pub(crate) shard_device_memory: ShardResidentDeviceMemoryMap,
    /// SV2 (sparse-versioning): the per-shard on-demand `deleted_by` tombstone region, keyed `(table,
    /// shard_id)`, PARALLEL to `shard_device_memory`. A delete-free shard has NO entry here (the HyPer
    /// "un-versioned rows pay nothing" property); a shard's region — a `capacity`-sized u64 buffer born
    /// all-live (`u64::MAX`) — is allocated on its FIRST DELETE (`tombstone_resident_shard_slots`). Presence
    /// in this map IS the shard's "has tombstones" flag; a DELETE stamps `deleted_by[slot] = commit_seq` here
    /// (out-of-line — the immutable column payload is never touched). The read filter (SV3) gathers it into
    /// the recompaction's unified buffer for the `deleted_by > read_txn_id` mask.
    pub(crate) shard_deleted_by_memory: ShardResidentDeviceMemoryMap,
    /// SV6 (`created_by` SI lower bound — the SV5 flip-gate): the per-shard on-demand `created_by` region,
    /// keyed `(table, shard_id)`, PARALLEL to `shard_deleted_by_memory` and under the SAME lifecycle
    /// discipline (cleaned at every site the buffer it annotates is retired). A shard whose rows are all
    /// born-visible has NO entry (the sparse-versioning property); the region — a `capacity`-sized i64
    /// buffer born all-visible (fill `0x00` = created_by 0 <= every read snapshot) — is allocated the first
    /// time an incremental UPDATE commit (SV5) APPENDS a new row version, which stamps
    /// `created_by[slot] = commit_seq` there. The read filter ANDs `created_by <= read_txn_id` so a reader
    /// bound to an OLDER snapshot cannot see the appended version (the SV5 P2 double-read window). Plain
    /// INSERT appends stay unstamped (born-visible, today's semantics).
    pub(crate) shard_created_by_memory: ShardResidentDeviceMemoryMap,
    /// RETIREMENT A1 (ledger #2, option A — device-authoritative): the per-shard u64 ROW-IDENTITY
    /// region, keyed `(table, shard_id)`, same lifecycle discipline as the version regions. Slot `s`
    /// holds the row's durable `row_id` (the repair key is derivable: `rel/{table}/{row_id:020}`), stamped at
    /// admission (parsed from the scanned tuple keys) and on every append (parsed from the commit's
    /// write-set keys; an UPDATE-appended version carries the ORIGINAL row's id — same key). The
    /// UNSTAMPED sentinel is `u64::MAX` (a benchmark/synthetic install has no durable identity; a
    /// device resolve finding the sentinel fails closed). 8 B/row device cost — ledgered;
    /// range-compression is a later optimization. This lets device locate return the WriteDelta's
    /// stable identity without consulting a host tuple store.
    pub(crate) shard_row_id_memory: ShardResidentDeviceMemoryMap,
    /// ADR-009 R1: per-table GPU hash-index reuse cache for the index-probe point-lookup route (built
    /// lazily, behind the default-ON `index_probe_enabled` flag). A plain `Mutex` (not the lock-free
    /// `ArcSwap` the hot path uses) because the lock is taken only off the
    /// fast cache-hit path; staleness is handled by the per-entry `generation` tag, not by eviction.
    pub(crate) wave_index: Mutex<BTreeMap<String, WaveResidentIndex>>,
    /// Per-shard PK hash index resident on the device. Keyed `(table, shard_id, col_idx)`,
    /// validated by `(ptr, row_count)` plus the ABA resident guard, and purged at every
    /// residency-retire site.
    pub(crate) shard_pk_device_index:
        Mutex<BTreeMap<(String, u32, usize), CachedShardPkDeviceIndex>>,
    /// Stable catalog OIDs for which every named index passed explicit device publication. Enrollment
    /// survives cache purge, table rename, and shape-changing DDL so a later mutation cannot silently
    /// downgrade mandatory indexes to legacy best-effort lazy-cache behavior. A dropped/recreated table
    /// receives a new OID and therefore cannot inherit the stale requirement.
    pub(crate) named_index_publications: Mutex<BTreeMap<u32, Vec<RelationalIndex>>>,
    /// O(indexes) coverage manifest for mandatory named indexes. It mirrors only successfully
    /// linearized cache entries and is cleared with cache retirement, letting rollover publication
    /// add one shard without re-enumerating every historical shard.
    pub(crate) named_index_coverage: Mutex<NamedIndexCoverage>,
    /// O(1) proof that `named_index_coverage` is complete for the current table/index shape. Cache
    /// retirement clears this marker while preserving mandatory enrollment by OID.
    pub(crate) named_index_coverage_complete: Mutex<BTreeMap<String, (u32, Vec<RelationalIndex>)>>,
    /// Test-only proof that incremental rollover publication visits only the new shard per index.
    #[cfg(test)]
    pub(crate) named_index_publication_shard_visits: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    pub(crate) named_index_publication_pre_linearize_hook:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Test-only interleaving seam after a named-index manifest is published but before its caller
    /// resumes canonical apply.
    #[cfg(test)]
    pub(crate) named_index_publication_post_publish_hook:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Generation-owned, GPU-resident multi-shard point route keyed by table/filter/projection shape.
    /// Exact Arc publication identity makes a route reusable without re-enumerating every shard per batch;
    /// a new publication misses and replaces it while in-flight readers keep the old plan pinned.
    pub(crate) sharded_point_routes: ArcSwap<ShardedPointRouteMap>,
    /// Per-table seqlock for in-place device-index mutation versus retained singleton plans. The
    /// map is touched only on route preparation or mutation; cache-hit reads retain the entry Arc.
    pub(crate) point_index_mutation_epochs:
        Mutex<BTreeMap<String, Arc<std::sync::atomic::AtomicU64>>>,
    /// READ-002's compound generation routes are isolated from the established int4 latency cache so
    /// adding a route family cannot change the production cache-hit plan type or branch shape.
    pub(crate) compound_point_routes: ArcSwap<CompoundPointRouteMap>,
    /// Dedicated compound-route bytes remain charged until the last plan owner drains, including
    /// readers and old ArcSwap map guards that outlive cache retirement. Current-map accounting alone
    /// would otherwise admit a replacement while the retired table-scale directory is still live.
    pub(crate) live_compound_point_route_bytes: Arc<Mutex<BTreeMap<(u16, String), u64>>>,
    /// Serializes rare route-cache COW publications and retirement purges; cache-hit reads stay lock-free.
    pub(crate) sharded_point_route_publish_lock: Mutex<()>,
    /// Linearizes destructive named/shard-index cache retirement with the pre-WAL-to-apply interval
    /// of a composite transaction. It is a short state latch, not a lock held across WAL: active
    /// publication marks one serialized writer and purgers defer only its affected table names,
    /// avoiding budget/route lock inversion while preserving mandatory coverage through the durable cut.
    transaction_named_index_lifecycle: Mutex<TransactionNamedIndexLifecycle>,
    /// Test-only interleaving seam after canonical apply returns and before the lifecycle guard
    /// drains final-publication-phase purge requests.
    #[cfg(test)]
    transaction_named_index_post_apply_hook:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// PERF-001: count of batches that reused an exact-generation GPU-resident shard descriptor plan.
    /// A nonzero value proves the hot route avoided per-batch shard enumeration and descriptor upload;
    /// generation replacement/purge still forces a miss and rebuild.
    pub(crate) sharded_point_route_cache_hits: std::sync::atomic::AtomicU64,
    /// Test-only one-shot fault seam: 1=prepare, 2=submit, 3=completion. Production has no branch/state.
    #[cfg(test)]
    pub(crate) sharded_point_forced_cuda_failure: std::sync::atomic::AtomicU8,
    #[cfg(test)]
    pub(crate) sharded_point_route_pre_publish_hook:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Test-only interleaving seam after an index build completes but before it can publish.
    #[cfg(test)]
    pub(crate) shard_pk_index_pre_publish_hook:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Test-only interleaving seam after append mutates its captured index but before basis publication.
    #[cfg(test)]
    pub(crate) shard_pk_index_append_post_launch_hook:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// Test-only interleaving seam after NULL eligibility is checked on the captured shard snapshot.
    #[cfg(test)]
    pub(crate) sharded_point_after_eligibility_hook:
        Mutex<Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>>,
    /// S-F/R-1 hard-cap serialization. Every allocation that becomes part of the durable
    /// relational resident set (admission payloads/mandatory regions and lazy device indexes)
    /// holds this lock from its budget preflight through publication. That closes the otherwise
    /// unavoidable check-then-allocate race between concurrent readers building indexes and an
    /// admission replacing/evicting tables. The lock is deliberately allocation-scoped, never
    /// held across a kernel launch or query execution.
    pub(crate) budget_allocation_lock: Mutex<()>,
    /// DECISIONS "lpb read levers" #1: count of batches served by the DENSE-emit index probe (vs the atomic
    /// kernel). The test signal that proves the dense route actually ran (output equality alone can't, since
    /// dense and atomic are byte-identical by design). `Relaxed` monotonic counter.
    /// M1: count of PK locates served by the device write-locate kernel.
    pub(crate) device_write_locate_hits: std::sync::atomic::AtomicU64,
    /// U1: count of coalesced VISIBLE-LOCATE launches (lane DELETE target resolution with
    /// on-device MVCC visibility — the fired-counter for the delete-intent device path).
    pub(crate) device_visible_locate_hits: std::sync::atomic::AtomicU64,
    /// U1: count of lane DELETE tombstones stamped IN PLACE by the apply coalescer (the
    /// fired-counter for the device tombstone path — a silent rehydrate fallback would pass
    /// output equality while abandoning the in-place design).
    pub(crate) lane_tombstone_applies: std::sync::atomic::AtomicU64,
    /// E2.5b-2 diagnostics: PK device-index rebuilds.
    pub(crate) lane_diag_rebuilds: std::sync::atomic::AtomicU64,
    pub(crate) dense_index_probe_hits: std::sync::atomic::AtomicU64,
    /// Slice 1b-ii-c: count of commits served by the IN-PLACE open-shard APPEND (vs a whole-table
    /// re-admit). The test signal that the append actually fired — output equality can't prove it
    /// (append and re-admit are byte-identical), and device-ptr stability can't either (a same-size
    /// re-admit reuses the just-freed address). `Relaxed` monotonic counter.
    pub(crate) open_shard_append_hits: std::sync::atomic::AtomicU64,
    /// S-F/R-1: open-shard rollovers declined before allocation because the new payload plus
    /// version/identity regions would exceed the GPU residency budget.
    pub(crate) rollover_budget_declines: std::sync::atomic::AtomicU64,
    /// Build-only INSERT qualification evidence. Every `with_shards_mut*` publication counts the
    /// descriptors structurally cloned from the current immutable map plus the descriptors whose
    /// route token is retokened in the next map. This is intentionally owned here rather than by
    /// one mutation caller: in-place appends, rollover, admission, and maintenance all publish
    /// through these helpers.
    #[cfg(feature = "probe-timing")]
    pub(crate) insert_probe_descriptor_clone_retoken_visits: std::sync::atomic::AtomicU64,
    /// E2.5c 2M+ push (b): merged applies served by the FUSED device pass.
    pub(crate) fused_apply_hits: std::sync::atomic::AtomicU64,
    /// S-d3: count of shards actually GATHERED (recompacted) by the sharded read after zone-map pruning.
    /// The non-vacuity signal that pruning fired — output equality can't prove a shard was skipped
    /// (a pruned shard holds no matching rows, so the result is identical either way). `Relaxed` monotonic.
    pub(crate) sharded_shards_gathered: std::sync::atomic::AtomicU64,
    /// Sub-slice 3b: count of sharded point-lookup reads served by the CROSS-SHARD PK INDEX route (the
    /// cached hash+bloom `locate` restricted the gathered shard set to the located shard(s), instead of
    /// gathering every zone-map-non-excluded shard). The non-vacuity signal that the index route actually
    /// fired — output equality can't prove it (the index route and the full scan return byte-identical
    /// rows by construction; only the SET of shards gathered differs). `Relaxed` monotonic counter.
    pub(crate) shard_index_route_hits: std::sync::atomic::AtomicU64,
    /// Step 1 (lpb-for-shards): count of BATCHES served by the batched cross-shard point-lookup path
    /// (`gather_sharded_int4_point_lookups_batched` — one batched locate + one kernel-gather per
    /// (shard, projected column) instead of a launch per needle). The non-vacuity signal that the batched
    /// path fired (vs a per-needle fallback). `Relaxed` monotonic counter.
    pub(crate) sharded_point_batch_hits: std::sync::atomic::AtomicU64,
    /// Sub-slice 8 (GPU-native probe): count of batches served by the fully-GPU dense-emit path
    /// (`gather_sharded_int4_point_lookups_batched_gpu` — device-resident per-shard index + the
    /// `gpu_db_resident_i32_index_probe_dense` kernel probes+gathers+emits on the GPU, no host per-needle
    /// probe). The non-vacuity signal that the fully device-probed path served the batch.
    pub(crate) sharded_point_gpu_probe_hits: std::sync::atomic::AtomicU64,
    /// Sub-slice 8 v3 (O(1) routing): count of GPU-native batches where the multi-shard kernel took the
    /// BINARY-SEARCH path (the shards were host-proven ascending-disjoint, so each needle routes to its one
    /// shard in O(log shards) instead of the O(shards) linear scan). The non-vacuity signal that binary routing
    /// (vs the linear fallback) actually fired. `Relaxed` monotonic counter.
    pub(crate) sharded_point_binary_route_hits: std::sync::atomic::AtomicU64,
    /// RETIREMENT A2: count of DML statements whose matches the DEVICE resolve served (locate ->
    /// row-identity -> keyed fetch). The non-vacuity signal — output equality cannot prove which
    /// resolver ran. `Relaxed` monotonic counter.
    pub(crate) dml_device_resolve_hits: std::sync::atomic::AtomicU64,
    /// RETIREMENT A3: count of constraint probes ANSWERED authoritatively by the device index
    /// (unique/FK validators). Both answers count — FALSE (no visible row carries the value) is the
    /// load-bearing one. `Relaxed` monotonic counter.
    pub(crate) dml_device_validate_hits: std::sync::atomic::AtomicU64,
    /// CPU-ENGINE RETIREMENT (ADR-006): count of relational SELECTs that the SPECIALIZED resident route
    /// DECLINED but the GENERAL GPU Expr executor then served on-device. The non-vacuity signal that a
    /// wider-type / non-enumerated read shape stayed on the GPU — output equality with a rows-only
    /// specification cannot prove WHICH engine ran. `Relaxed` monotonic counter.
    pub(crate) general_read_fallback_hits: std::sync::atomic::AtomicU64,
    /// STRATA S-E.1 (streaming executor, ADR-012): count of relational SELECTs whose scalar reduction
    /// (COUNT(*) / SUM / MIN / MAX) was served OUT-OF-CORE by the streaming fold — the table's visible
    /// rows chunked to a per-GPU byte budget, each chunk uploaded + reduced ON THE DEVICE, partials
    /// combined by a final device reduction, never all shards resident at once. The non-vacuity signal
    /// that an over-VRAM aggregate stayed on the GPU. `Relaxed`.
    pub(crate) streaming_fold_hits: std::sync::atomic::AtomicU64,
    /// STRATA S-E.1: total streaming chunks reduced across all folds (a fold over an over-budget table
    /// runs >1). Proves bounded-residency chunking actually fired (a single-chunk fold == 1).
    pub(crate) streaming_fold_chunks: std::sync::atomic::AtomicU64,
    /// STRATA S-E multi-GPU partial combine: chunks whose device work completed on a GPU other
    /// than the query's coordinator/default GPU. A non-zero delta proves round-robin routing
    /// actually executed remotely; merely discovering or configuring a second GPU is insufficient.
    pub(crate) streaming_fold_secondary_gpu_chunks: std::sync::atomic::AtomicU64,
    /// STRATA S-E.1: the high-water device bytes of any single streaming chunk (the peak transient
    /// residency of the fold). The out-of-core proof: this stays <= the configured budget even when the
    /// whole table's bytes dwarf it. `fetch_max`, monotonic across the process.
    pub(crate) streaming_fold_peak_chunk_bytes: std::sync::atomic::AtomicU64,
    /// Test-only high-water of source windows staged by one bounded cold-index proof.
    #[cfg(test)]
    pub(crate) cold_index_validation_peak_staged_chunks: std::sync::atomic::AtomicU64,
    /// Exact allocator-backed high-water for one transactional cold UNIQUE validation attempt.
    /// Includes sliced cold payloads/tombstone sidecars, unified D2D source, and pooled operator
    /// scratch; tests compare it directly with the configured residency headroom.
    #[cfg(test)]
    pub(crate) cold_index_validation_peak_device_bytes: std::sync::atomic::AtomicU64,
    /// Exact fallible host payload/sidecar lease high-water for one cold validation window.
    #[cfg(test)]
    pub(crate) cold_index_validation_peak_host_staging_bytes: std::sync::atomic::AtomicU64,
    /// Bounded-planner evidence: high-water window descriptors retained by one plan and exact
    /// number of GPU subset proofs launched. A refusal test can force tiny batches without
    /// manufacturing a table large enough to exhaust host memory.
    #[cfg(test)]
    pub(crate) cold_index_validation_peak_planned_windows: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    pub(crate) cold_index_validation_subset_proofs: std::sync::atomic::AtomicU64,
    /// Engine-local test overrides (zero = production limit), isolated across concurrent engines.
    #[cfg(test)]
    pub(crate) cold_index_validation_batch_rows_override: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    pub(crate) cold_index_validation_max_batches_override: std::sync::atomic::AtomicUsize,
    /// ADR-012 streaming two-relation joins served as bounded chunk/block pairs.
    pub(crate) streaming_join_hits: std::sync::atomic::AtomicU64,
    pub(crate) streaming_join_block_pairs: std::sync::atomic::AtomicU64,
    pub(crate) streaming_join_peak_device_bytes: std::sync::atomic::AtomicU64,
    pub(crate) streaming_window_hits: std::sync::atomic::AtomicU64,
    /// STRATA S-E.6: the streaming COLD TIER — per-table DEVICE-FORMAT chunk payloads cached in host
    /// RAM after a fold's first (MVCC-scan) build, replayed byte-for-byte on later streaming reads
    /// (no per-row decode, no payload assembly — the measured ~68% host wall). Validity = the pinned
    /// tuple-store generation-payload Arc (pointer equality; any write COW-publishes a fresh Arc ->
    /// miss -> rebuild) + the chunk target. COW map: readers `load()` wait-free; the (rare) install
    /// and cap-eviction publishers serialize on `streaming_cold_lock`.
    pub(crate) streaming_cold_chunks:
        ArcSwap<BTreeMap<String, std::sync::Arc<crate::engine_streaming_exec::ColdTableChunks>>>,
    /// Serializes `streaming_cold_chunks` publishers (install / cap eviction).
    pub(crate) streaming_cold_lock: Mutex<()>,
    /// STRATA S-E.6: streaming reads served from the cold tier (byte-replay, no MVCC decode).
    pub(crate) streaming_cold_hits: std::sync::atomic::AtomicU64,
    /// STRATA S-E.6: cold-tier builds installed (a fold's scan captured its chunks for reuse).
    pub(crate) streaming_cold_builds: std::sync::atomic::AtomicU64,
    /// STRATA S-E.6b: cold-tier installs spilled to the unlinked temp file (over the RAM threshold).
    pub(crate) streaming_cold_spills: std::sync::atomic::AtomicU64,
    /// STRATA 6c-1: cold-tier entries PATCHED in place of a full rebuild after a write (the O(delta)
    /// maintenance win — the non-vacuity signal that a write no longer costs O(table)).
    pub(crate) streaming_cold_patches: std::sync::atomic::AtomicU64,
    /// STRATA 6c-1: dirty chunks rebuilt across all patches (a one-row write should rebuild ONE).
    pub(crate) streaming_cold_chunks_rebuilt: std::sync::atomic::AtomicU64,
    /// P1 (sealed-shards-primary): cold-tier tables restored from the checkpoint artifact at reopen
    /// (the warm-start signal — the first streaming read after recovery replays bytes, no scan).
    pub(crate) streaming_cold_restored: std::sync::atomic::AtomicU64,
    /// P2 (sealed-shards-primary): cold-chunk rows tombstone-STAMPED into deleted_by sidecars in
    /// place of an O(chunk) rebuild (the SV2 win; chunks_rebuilt stays flat for pure deletes).
    pub(crate) streaming_cold_stamps: std::sync::atomic::AtomicU64,
    /// P3 (sealed-shards-primary): DELETE/UPDATE WHERE-locates resolved ON-DEVICE via the streaming
    /// fold (a non-admitted table whose predicate the value index could not bound — previously the
    /// pure-host seq_scan+filter loop, the reachable CPU-relational-engine residue).
    pub(crate) dml_streaming_resolve_hits: std::sync::atomic::AtomicU64,
    /// R3-004: tables whose live relational image is device-authoritative. Normal DML enters this
    /// set after publishing a maintained generation and never clears it through a fallback; only
    /// explicit RETIRE-002 DDL/vacuum/recovery repair may reverse-gather and de-authorize. COW set;
    /// readers load without waiting on the apply path.
    pub(crate) device_authoritative_tables: ArcSwap<std::collections::BTreeSet<String>>,
    /// R3-004: commits that published device-authoritative relational state.
    pub(crate) device_authoritative_commits: std::sync::atomic::AtomicU64,
    /// P4-2b (S-E.P4): CHUNK-AUTHORITATIVE tables — name -> the entry boundary. Cold chunks and
    /// their sidecars are the live materialization; the boundary fences retained snapshots and
    /// explicit RETIRE-002 repair. COW map, publishers serialize on the commit path.
    pub(crate) chunk_authoritative_tables: ArcSwap<std::collections::BTreeMap<String, Index>>,
    /// Device-authoritative commits maintained through the chunk-tail path.
    pub(crate) chunk_class_device_commits: std::sync::atomic::AtomicU64,
    /// P4 compactions and physically removed tombstoned slots.
    #[cfg(test)]
    pub(crate) chunk_class_compactions: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    pub(crate) chunk_class_compacted_slots: std::sync::atomic::AtomicU64,
    /// RETIRE-002 repair exits that reverse-gather a chunk generation before deauthorization.
    pub(crate) chunk_class_deauths: std::sync::atomic::AtomicU64,
    /// P5-1: the per-chunk device KEY-INDEX cache — (table, chunk_id, key_id) -> a persistent
    /// retained hash-index buffer over the chunk's key fingerprints (all-visible build; the
    /// sidecar applies at the probe's recheck). chunk_id is the validity token (fresh iff the
    /// payload bytes are new), so stamps and tail appends never invalidate existing entries.
    /// VRAM-accounted + capped with LRU (a touch counter; eviction frees; rebuilt on demand).
    pub(crate) chunk_key_index:
        Mutex<BTreeMap<(String, u64, usize), crate::engine_streaming_exec::ChunkKeyIndex>>,
    /// P5-1: retained chunk-index VRAM bytes (the shard twin has NO accounting — this one does).
    pub(crate) chunk_key_index_bytes: std::sync::atomic::AtomicU64,
    /// P5-1: the LRU touch clock.
    pub(crate) chunk_key_index_clock: std::sync::atomic::AtomicU64,
    /// P5-later: compact all-chunk Bloom filters used when the full retained index set exceeds its cap.
    pub(crate) chunk_key_bloom:
        Mutex<BTreeMap<(String, u64, usize), crate::engine_streaming_exec::ChunkKeyBloom>>,
    pub(crate) chunk_key_bloom_bytes: std::sync::atomic::AtomicU64,
    /// Candidate-routing launches served by the Bloom path (non-vacuity for over-cap keyed admission).
    pub(crate) chunk_key_bloom_probes: std::sync::atomic::AtomicU64,
    /// P5-2: keyed-class uniqueness preflights served ON-DEVICE (probe + slot recheck) — the
    /// non-vacuity signal for the keyed eligibility lift.
    pub(crate) chunk_class_unique_probes: std::sync::atomic::AtomicU64,
    /// P5-2: duplicates the device probe REJECTED (a recheck-confirmed conflict).
    pub(crate) chunk_class_unique_probe_conflicts: std::sync::atomic::AtomicU64,
    /// P5-3: class DML locates served by the KEY-INDEX PROBE (vs the full fold scan).
    pub(crate) chunk_class_dml_key_locates: std::sync::atomic::AtomicU64,
    /// P5 charter closure: candidate-index hits whose visibility + exact key/residual
    /// predicate were decided by the device predicate VM. This proves the host comparator is
    /// not silently serving the authoritative recheck.
    pub(crate) chunk_class_device_exact_rechecks: std::sync::atomic::AtomicU64,
    /// VACUUM #5: per-table count of incremental tombstone stamps since the last rebuild —
    /// the CHURN signal (each SV4b/SV5/A4b tombstone adds a dead slot; enough of them degrade
    /// the PK index to dup-declines and bloat scans). Reset by vacuum/re-admit. Serialized-path
    /// writers only; COW map, readers load() wait-free.
    pub(crate) resident_tombstone_churn: ArcSwap<std::collections::BTreeMap<String, u64>>,
    /// VACUUM #5: the auto-trigger's DEFERRED handoff — the commit arm detects the threshold
    /// while holding the commit lock + catalog latch (running the vacuum there self-deadlocks:
    /// its re-admit needs the latch), so it parks the table name here and the execute_text tail
    /// runs `vacuum_table` after the commit releases both.
    pub(crate) pending_auto_vacuum: std::sync::Mutex<Option<String>>,
    /// VACUUM #5 (audit F1): deferred auto-vacuums that FAILED — maintenance errors must never
    /// fail the (already durable) statement that drained them; they land here as telemetry and
    /// the churn counter re-arms the trigger.
    pub(crate) auto_vacuum_failures: std::sync::atomic::AtomicU64,
    // The per-table resident snapshot metadata + shard metadata, each an immutable published map
    // (Stage 3 — blocker #2). Readers `load()` (wait-free) and pin the `Arc` across the kernel launch;
    // the single serialized publisher COW-stores a fresh map on warm-up / DDL drop / invalidate /
    // memory-pressure.
    pub(crate) snapshots: ArcSwap<BTreeMap<String, RelationalResidencyEntry>>,
    pub(crate) shards: ArcSwap<BTreeMap<String, Vec<RelationalResidentShard>>>,
    /// Always-present mutation boundary for every residency publisher, sidecar lifecycle change,
    /// and device-index rebuild. It is independent of optional intent lanes, so local/DDL engines
    /// cannot accidentally plan a pre-WAL append without the exclusion that protects its sealed
    /// descriptor and allocation geometry. Lock order is commit -> residency mutation -> budget.
    /// Readers remain lock-free.
    pub(crate) mutation_gate: Mutex<()>,
    /// W0: serializes the load→clone→store publishers of `snapshots`/`shards` (see
    /// [`ResidencyReadState::with_snapshots_mut`]). Readers never touch it.
    pub(crate) descriptor_publish_lock: Mutex<()>,
    /// THE FLIP (deadlock fix, caught by the burn-in): a LOCK-FREE mirror of the catalog's
    /// `relational_resident_cache.budget_bytes_by_gpu`, updated by the (rare, `&mut self`) budget
    /// setters. The resident-route PLANNER reads budgets from HERE — reading them through
    /// `ddl_catalog()` self-deadlocked when a route was planned INSIDE the commit critical section
    /// (a materialized-view create/refresh internal read on a shard-resident table holds the catalog
    /// latch). The catalog copy stays authoritative for ADMISSION (which already holds the latch).
    pub(crate) admission_budget_bytes_by_gpu: ArcSwap<BTreeMap<u16, u64>>,
}

impl ResidencyReadState {
    pub(crate) fn begin_transaction_named_index_publication(
        &self,
        protected_tables: BTreeSet<String>,
    ) -> TransactionNamedIndexPublicationGuard<'_> {
        let mut lifecycle = self
            .transaction_named_index_lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert!(
            !lifecycle.publication_active,
            "the serialized commit authority permits one named-index publication"
        );
        lifecycle.publication_active = true;
        lifecycle.protected_tables = protected_tables;
        lifecycle.deferred_purges.clear();
        lifecycle.final_publication_tables.clear();
        lifecycle.post_publication_deferred_purges.clear();
        drop(lifecycle);
        TransactionNamedIndexPublicationGuard {
            residency: self,
            final_publication_entered: false,
            finished: false,
        }
    }

    fn enter_transaction_named_index_final_publication(&self) {
        let mut lifecycle = self
            .transaction_named_index_lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert!(lifecycle.publication_active);
        lifecycle.final_publication_tables = lifecycle.protected_tables.clone();
        // Canonical apply is one atomic publication. Requests recorded before its final phase are
        // ordered before that publication. Once apply begins, external pressure/retirement must be
        // replayed after success so no request that observed a newly allocated generation is lost.
        lifecycle.post_publication_deferred_purges.clear();
    }

    #[cfg(test)]
    pub(crate) fn set_transaction_named_index_post_apply_hook(
        &self,
        reached: Arc<std::sync::Barrier>,
        resume: Arc<std::sync::Barrier>,
    ) {
        *self
            .transaction_named_index_post_apply_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((reached, resume));
    }

    #[cfg(test)]
    pub(crate) fn run_transaction_named_index_post_apply_hook(&self) {
        let hook = self
            .transaction_named_index_post_apply_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    fn finish_transaction_named_index_publication(&self, publication_succeeded: bool) {
        let mut lifecycle = self
            .transaction_named_index_lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert!(lifecycle.publication_active);
        lifecycle.publication_active = false;
        lifecycle.protected_tables.clear();
        let deferred = std::mem::take(&mut lifecycle.deferred_purges);
        lifecycle.final_publication_tables.clear();
        let post_publication = std::mem::take(&mut lifecycle.post_publication_deferred_purges);
        if publication_succeeded {
            // Requests before the final cut observed a generation canonical apply superseded.
            // Requests after it observed the final generation and must win even though the guard
            // was still active while transaction acknowledgement was being completed.
            for table in post_publication {
                self.purge_shard_pk_index_for_table_inner(&table);
            }
        } else {
            // Keep the lifecycle latch until cleanup completes. A new transaction therefore begins
            // either before a purge request (and defers it) or after its full route/cache retirement;
            // it can never capture the half-purged interval.
            for table in deferred {
                self.purge_shard_pk_index_for_table_inner(&table);
            }
        }
    }

    pub(crate) fn point_index_mutation_epoch(
        &self,
        table: &str,
    ) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::clone(
            self.point_index_mutation_epochs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(table.to_string())
                .or_default(),
        )
    }

    pub(crate) fn begin_point_index_mutation(&self, table: &str) -> PointIndexMutationGuard {
        let epoch = self.point_index_mutation_epoch(table);
        loop {
            let current = epoch.load(std::sync::atomic::Ordering::Acquire);
            if current & 1 == 0 {
                let writing = current
                    .checked_add(1)
                    .expect("point-index mutation epoch exhausted");
                if epoch
                    .compare_exchange_weak(
                        current,
                        writing,
                        std::sync::atomic::Ordering::AcqRel,
                        std::sync::atomic::Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return PointIndexMutationGuard { epoch };
                }
            }
            std::thread::yield_now();
        }
    }

    #[cfg(test)]
    pub(crate) fn run_shard_pk_index_append_post_launch_hook(&self) {
        let hook = self
            .shard_pk_index_append_post_launch_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some((reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    /// Drop every cached device PK index for `table` at residency-retire sites so the cache does
    /// not retain a stale shard buffer or device allocation across a generation replacement.
    pub(crate) fn purge_shard_pk_index_for_table(&self, table: &str) {
        let mut lifecycle = self
            .transaction_named_index_lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if lifecycle.publication_active && lifecycle.protected_tables.contains(table) {
            let canonical_owner =
                TRANSACTION_NAMED_INDEX_PUBLICATION_OWNER_ACTIVE.with(std::cell::Cell::get);
            if canonical_owner {
                drop(lifecycle);
                self.purge_shard_pk_index_for_table_inner(table);
                return;
            }
            lifecycle.deferred_purges.insert(table.to_string());
            if lifecycle.final_publication_tables.contains(table) {
                lifecycle
                    .post_publication_deferred_purges
                    .insert(table.to_string());
            }
            return;
        }
        // Keep the state latch across the physical purge so a transaction cannot increment the
        // active counter between the zero observation and cache removal.
        self.purge_shard_pk_index_for_table_inner(table);
    }

    /// Canonical transaction publisher's in-window replacement path. The lifecycle guard remains
    /// active, so external purge requests still defer; this owner may retire the protected old
    /// manifest immediately before installing an empty/fresh final manifest.
    pub(crate) fn purge_shard_pk_index_for_table_during_transaction(&self, table: &str) {
        let lifecycle = self
            .transaction_named_index_lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert!(
            !lifecycle.publication_active || lifecycle.protected_tables.contains(table),
            "an active transaction index publisher may retire only its protected tables"
        );
        drop(lifecycle);
        self.purge_shard_pk_index_for_table_inner(table);
    }

    /// Retire only cache identities absent from the transaction's final catalog after replacement
    /// coverage has linearized. Surviving raw-column aliases and stable index-OID keys keep their
    /// allocations; dropped/recreated OIDs cannot remain a hidden residency owner.
    pub(crate) fn retain_shard_pk_indexes_for_table_during_transaction(
        &self,
        table: &str,
        retained_key_ids: &BTreeSet<usize>,
    ) {
        let lifecycle = self
            .transaction_named_index_lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert!(
            !lifecycle.publication_active || lifecycle.protected_tables.contains(table),
            "an active transaction index publisher may retain only its protected tables"
        );
        drop(lifecycle);

        // Route -> index is the cache ownership order. Every prepared route is catalog-shaped, so
        // retire the table's routes even when its underlying stable key allocation survives rename.
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        {
            let current = self.sharded_point_routes.load();
            if current
                .keys()
                .any(|(cached_table, _, _)| cached_table == table)
            {
                let mut next = (**current).clone();
                next.retain(|(cached_table, _, _), _| cached_table != table);
                self.sharded_point_routes.store(Arc::new(next));
            }
        }
        {
            let current = self.compound_point_routes.load();
            if current
                .keys()
                .any(|(cached_table, _, _)| cached_table == table)
            {
                let mut next = (**current).clone();
                next.retain(|(cached_table, _, _), _| cached_table != table);
                self.compound_point_routes.store(Arc::new(next));
            }
        }
        self.shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|(cached_table, _, key_id), _| {
                cached_table != table || retained_key_ids.contains(key_id)
            });
    }

    fn purge_shard_pk_index_for_table_inner(&self, table: &str) {
        // Route -> index is the global ownership order. Retire cached plans while their indexes remain
        // map-accounted, drop this method's old-map guard, and only then remove the index entries. Prepared
        // route publication holds the same route lock through its under-lock index-identity validation, so a
        // purge cannot slip between validation and cache store.
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        {
            let current = self.sharded_point_routes.load();
            if current
                .keys()
                .any(|(cached_table, _, _)| cached_table == table)
            {
                let mut next = (**current).clone();
                next.retain(|(cached_table, _, _), _| cached_table != table);
                self.sharded_point_routes.store(Arc::new(next));
            }
        }
        {
            let current = self.compound_point_routes.load();
            if current
                .keys()
                .any(|(cached_table, _, _)| cached_table == table)
            {
                let mut next = (**current).clone();
                next.retain(|(cached_table, _, _), _| cached_table != table);
                self.compound_point_routes.store(Arc::new(next));
            }
        }
        self.shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|(cached_table, _, _), _| cached_table != table);
        self.named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|(cached_table, _, _), _| cached_table != table);
        self.named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(table);
    }

    /// Retire cached prepared routes for the named tables. An already-submitted reader retains its
    /// own plan Arc through completion, but the global cache stops pinning the generation before return.
    fn purge_sharded_point_routes_for_tables<'a>(&self, tables: impl IntoIterator<Item = &'a str>) {
        let tables = tables
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        if tables.is_empty() {
            return;
        }
        let _publish = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.sharded_point_routes.load();
        if current
            .keys()
            .any(|(cached_table, _, _)| tables.contains(cached_table.as_str()))
        {
            let mut next = (**current).clone();
            next.retain(|(cached_table, _, _), _| !tables.contains(cached_table.as_str()));
            self.sharded_point_routes.store(Arc::new(next));
        }
        let current = self.compound_point_routes.load();
        if current
            .keys()
            .any(|(cached_table, _, _)| tables.contains(cached_table.as_str()))
        {
            let mut next = (**current).clone();
            next.retain(|(cached_table, _, _), _| !tables.contains(cached_table.as_str()));
            self.compound_point_routes.store(Arc::new(next));
        }
    }

    /// W0: flag `table`'s residency descriptors (snapshot + every shard) invalidated at
    /// `(txn_id, index)` — ONE `descriptor_publish_lock` hold covering the already-flagged check
    /// AND both map publishes. The check makes repeated invalidations of the same table (the
    /// concurrent wave invalidates PER ITEM) O(load + compare) instead of two full COW clones —
    /// measured −13% sustained on the default (invalidate-per-item) benchmark arm without it.
    /// Skipping is safe ONLY because the check runs under the same lock every publisher takes:
    /// a generation observed flagged here can only be replaced by a LATER lock-holder (e.g. a
    /// re-admission installing fresh descriptors), which would equally have overwritten a
    /// redundant re-flag.
    pub(crate) fn flag_table_descriptors_invalidated(&self, table: &str, txn_id: u64, index: u64) {
        let _publish = self
            .descriptor_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let snapshot_flagged = self.snapshots.load().get(table).is_none_or(|entry| {
            entry.descriptor.invalidated_by_txn_id.is_some()
                && entry
                    .descriptor
                    .device_memory_proof
                    .as_ref()
                    .is_none_or(|proof| !proof.retained)
        });
        let shards_flagged = self.shards.load().get(table).is_none_or(|shards| {
            shards.iter().all(|shard| {
                shard.invalidated_by_txn_id.is_some()
                    && shard
                        .device_memory_proof
                        .as_ref()
                        .is_none_or(|proof| !proof.retained)
            })
        });
        if snapshot_flagged && shards_flagged {
            return;
        }
        if !snapshot_flagged {
            let mut next = (**self.snapshots.load()).clone();
            if let Some(entry) = next.get_mut(table) {
                // make_mut COWs the shared descriptor into a fresh version.
                let snapshot = Arc::make_mut(&mut entry.descriptor);
                if snapshot.invalidated_by_txn_id.is_none() {
                    snapshot.invalidated_by_txn_id = Some(txn_id);
                    snapshot.invalidated_at_index = Some(index);
                }
                if let Some(proof) = snapshot.device_memory_proof.as_mut() {
                    proof.retained = false;
                }
            }
            self.snapshots.store(Arc::new(next));
        }
        if !shards_flagged {
            let mut next = (**self.shards.load()).clone();
            if let Some(shards) = next.get_mut(table) {
                let generation = Arc::new(());
                for shard in shards.iter_mut() {
                    if shard.invalidated_by_txn_id.is_none() {
                        shard.invalidated_by_txn_id = Some(txn_id);
                        shard.invalidated_at_index = Some(index);
                    }
                    if let Some(proof) = shard.device_memory_proof.as_mut() {
                        proof.retained = false;
                    }
                    shard.point_route_generation = Arc::clone(&generation);
                }
            }
            self.shards.store(Arc::new(next));
            // The invalidated publication is a new table generation even though its payload Arc is retained
            // for already-captured readers. Rotate before purging while descriptor publication stays locked,
            // so a G0 preparer cannot pass its route-lock recheck and republish after retirement.
            self.purge_sharded_point_routes_for_tables(std::iter::once(table));
        }
    }

    /// COW-mutate the resident snapshot map: clone the published map, apply `mutate`, then
    /// atomically store it. In-flight readers keep the generation they loaded. W0: publishers
    /// serialize on `descriptor_publish_lock` — the catalog-latch paths were the single publisher
    /// historically, but the CONCURRENT commit path now flags invalidations here too (it holds the
    /// commit_mutex, NOT the latch), and two racing load→clone→store publishers would lose one
    /// side's update. Readers stay lock-free (`ArcSwap::load`).
    pub(crate) fn with_snapshots_mut<R>(
        &self,
        mutate: impl FnOnce(&mut BTreeMap<String, RelationalResidencyEntry>) -> R,
    ) -> R {
        let _publish = self
            .descriptor_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut next = (**self.snapshots.load()).clone();
        let result = mutate(&mut next);
        self.snapshots.store(Arc::new(next));
        result
    }

    /// COW-mutate the resident shard map (see [`ResidencyReadState::with_snapshots_mut`] for the
    /// W0 publisher-serialization contract).
    pub(crate) fn with_shards_mut<R>(
        &self,
        mutate: impl FnOnce(&mut BTreeMap<String, Vec<RelationalResidentShard>>) -> R,
    ) -> R {
        let _publish = self
            .descriptor_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.shards.load_full();
        #[cfg(feature = "probe-timing")]
        let descriptor_clones = current
            .values()
            .map(|shards| shards.len() as u64)
            .sum::<u64>();
        let mut next = (*current).clone();
        let result = mutate(&mut next);
        // Derive the exact changed-table set while publication is serialized, so no caller can forget
        // to rotate the per-table token or retire its plan. RelationalResidentShard equality deliberately
        // excludes `point_route_generation`; the comparison therefore reflects real descriptor changes.
        let table_names = current
            .keys()
            .chain(next.keys())
            .collect::<std::collections::BTreeSet<_>>();
        let changed_tables = table_names
            .into_iter()
            .filter(|table| current.get(*table) != next.get(*table))
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        #[cfg(feature = "probe-timing")]
        let mut descriptor_retoken_visits = 0_u64;
        for table in &changed_tables {
            if let Some(shards) = next.get_mut(table) {
                let generation = Arc::new(());
                #[cfg(feature = "probe-timing")]
                {
                    descriptor_retoken_visits =
                        descriptor_retoken_visits.saturating_add(shards.len() as u64);
                }
                for shard in shards {
                    shard.point_route_generation = Arc::clone(&generation);
                }
            }
        }
        #[cfg(feature = "probe-timing")]
        self.insert_probe_descriptor_clone_retoken_visits.fetch_add(
            descriptor_clones.saturating_add(descriptor_retoken_visits),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shards.store(Arc::new(next));
        // Store-before-purge is deliberate. New readers immediately see a new table token; old readers
        // fail their under-lock global-token recheck. Retaining descriptor_publish_lock through this purge
        // closes the prior purge-before-publication re-insertion window.
        self.purge_sharded_point_routes_for_tables(changed_tables.iter().map(String::as_str));
        result
    }

    /// Publish a mutation whose owner already knows the one affected table. Hot append/admission paths use
    /// this form so rotating the point-route token stays O(1) beyond the shard-map clone they already pay;
    /// the generic form above remains for rare multi-table maintenance and derives its changed set exactly.
    pub(crate) fn with_shards_mut_for_table<R>(
        &self,
        table: &str,
        mutate: impl FnOnce(&mut BTreeMap<String, Vec<RelationalResidentShard>>) -> R,
    ) -> R {
        let _publish = self
            .descriptor_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.shards.load_full();
        #[cfg(feature = "probe-timing")]
        let descriptor_clones = current
            .values()
            .map(|shards| shards.len() as u64)
            .sum::<u64>();
        let mut next = (*current).clone();
        let result = mutate(&mut next);
        #[cfg(feature = "probe-timing")]
        let mut descriptor_retoken_visits = 0_u64;
        if let Some(shards) = next.get_mut(table) {
            let generation = Arc::new(());
            #[cfg(feature = "probe-timing")]
            {
                descriptor_retoken_visits = shards.len() as u64;
            }
            for shard in shards {
                shard.point_route_generation = Arc::clone(&generation);
            }
        }
        #[cfg(feature = "probe-timing")]
        self.insert_probe_descriptor_clone_retoken_visits.fetch_add(
            descriptor_clones.saturating_add(descriptor_retoken_visits),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shards.store(Arc::new(next));
        self.purge_sharded_point_routes_for_tables(std::iter::once(table));
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
