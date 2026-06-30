//! Relational data-model + residency types (P0 §9.6 decomposition, behavior-
//! preserving): the catalog schema structs (RelationalTable/Column/Index/...),
//! the select result, residency snapshot + its operations, retained-read
//! handles, resident-device column stats/layout, benchmark chunk installs, and
//! the residency warmup/maintenance policy/report + access-path types, plus the
//! resident-device offset/stats free helpers the read path calls.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalTable {
    pub schema: String,
    pub name: String,
    pub oid: u32,
    pub columns: Vec<RelationalColumn>,
    pub indexes: Vec<RelationalIndex>,
    pub check_constraints: Vec<RelationalCheckConstraint>,
    pub foreign_keys: Vec<RelationalForeignKey>,
    pub acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalColumn {
    pub id: u32,
    pub table_oid: u32,
    pub attnum: i16,
    pub name: String,
    pub ty: SqlType,
    pub domain: Option<String>,
    pub default: Option<ColumnDefault>,
    pub type_oid: u32,
    pub type_size: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalIndex {
    pub name: String,
    pub table: String,
    pub column: String,
    pub unique: bool,
    pub primary_key: bool,
    pub unique_constraint: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalCheckConstraint {
    pub name: String,
    pub column: String,
    pub op: SelectFilterOp,
    pub value: SqlValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalForeignKey {
    pub name: String,
    pub column: String,
    pub referenced_table: String,
    pub referenced_column: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalView {
    pub schema: String,
    pub name: String,
    pub oid: u32,
    pub query: Select,
    pub definition: String,
    pub acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalMaterializedView {
    pub schema: String,
    pub name: String,
    pub oid: u32,
    pub query: Select,
    pub definition: String,
    pub columns: Vec<RelationalColumn>,
    pub rows: Vec<Vec<SqlValue>>,
    pub acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalFunction {
    pub schema: String,
    pub name: String,
    pub oid: u32,
    pub return_type: SqlType,
    pub body: String,
    pub acl: BTreeMap<String, BTreeSet<FunctionPrivilege>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalSequence {
    pub schema: String,
    pub name: String,
    pub oid: u32,
    pub last_value: i64,
    pub is_called: bool,
    pub acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalDomain {
    pub schema: String,
    pub name: String,
    pub oid: u32,
    pub base_type: SqlType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalPublication {
    pub name: String,
    pub oid: u32,
    pub all_tables: bool,
    pub tables: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalSubscription {
    pub name: String,
    pub oid: u32,
    pub connection: String,
    pub publications: Vec<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalRole {
    pub name: String,
    pub oid: u32,
    pub login: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalDatabase {
    pub name: String,
    pub oid: u32,
    pub acl: BTreeMap<String, BTreeSet<DatabasePrivilege>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalTablespace {
    pub name: String,
    pub oid: u32,
    pub location: String,
    pub acl: BTreeMap<String, BTreeSet<TablespacePrivilege>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RelationalCommentTarget {
    Database { database: String },
    Role { role: String },
    Schema { schema: String },
    Tablespace { tablespace: String },
    Table { table: String },
    Column { table: String, attnum: i16 },
    Index { index: String },
    View { view: String },
    MaterializedView { materialized_view: String },
    Extension { extension: String },
    Function { function: String },
    Sequence { sequence: String },
    Domain { domain: String },
    Publication { publication: String },
    Subscription { subscription: String },
    Constraint { table: String, constraint: String },
}

/// A flat, row-major block of result values: `values` holds `ncols`-wide rows back-to-back, so row `i` is
/// `values[i*ncols .. (i+1)*ncols]`. Replaces the per-row `Vec<Vec<SqlValue>>` boxing that capped end-to-end
/// read throughput at ~7.8M (the per-row heap Vec + grouping was 153ns/row; flat is ~2.3ns/row -- DECISIONS
/// "Result-path optimization" step-1: 66x, drain-bound). Keeps `SqlValue` so the wire value-type + encoder
/// are unchanged. Transparent-ish: `iter()`/`Index` yield row slices `&[SqlValue]`, `PartialEq<Vec<Vec<..>>>`
/// + `From<Vec<Vec<..>>>` keep existing construction/asserts working with minimal churn.
#[derive(Debug, Clone, Default)]
pub struct RowBlock {
    values: Vec<SqlValue>,
    ncols: usize,
}

// Row-based equality: two blocks are equal iff they hold the same ROWS, regardless of how `ncols` was
// recorded for an empty block (an empty result may carry ncols=0 from `From<vec![]>` or ncols=N from the
// schema width). Field-wise derive would wrongly distinguish those, breaking byte-identity asserts.
impl PartialEq for RowBlock {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
impl Eq for RowBlock {}

impl RowBlock {
    /// Build directly from a flat row-major value buffer (the fast path: no per-row Vec). `ncols` must be
    /// > 0 unless `values` is empty.
    pub fn flat(values: Vec<SqlValue>, ncols: usize) -> Self {
        debug_assert!(ncols > 0 || values.is_empty(), "RowBlock: ncols=0 with non-empty values");
        Self { values, ncols }
    }
    pub fn len(&self) -> usize {
        if self.ncols == 0 {
            0
        } else {
            self.values.len() / self.ncols
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// Row `i` as a `&[SqlValue]` slice (no alloc).
    pub fn row(&self, i: usize) -> &[SqlValue] {
        let n = self.ncols;
        &self.values[i * n..i * n + n]
    }
    /// Iterate rows as `&[SqlValue]` slices.
    pub fn iter(&self) -> std::slice::Chunks<'_, SqlValue> {
        self.values.chunks(self.ncols.max(1))
    }
    /// Convert back to the boxed `Vec<Vec<SqlValue>>` (for the few paths/tests that still want it).
    pub fn into_boxed(self) -> Vec<Vec<SqlValue>> {
        if self.ncols == 0 {
            return Vec::new();
        }
        self.values.chunks(self.ncols).map(<[SqlValue]>::to_vec).collect()
    }
    /// The `count` rows starting at row `start` as a flat `&[SqlValue]` slice (`count*ncols` values) — for
    /// slicing a multi-needle batched block into one needle's rows, zero-copy.
    pub fn rows_slice(&self, start: usize, count: usize) -> &[SqlValue] {
        let nc = self.ncols;
        &self.values[start * nc..(start + count) * nc]
    }
}

/// A BATCHED retained-read result: ONE shared schema + ONE flat i32 buffer over ALL needles' rows (in needle
/// order) + per-needle row ranges. Replaces the N-per-needle `RelationalSelectResult` structs + by-needle
/// grouping + 2N `Arc` clones + N column re-maps — the residual that capped end-to-end point reads below the
/// GPU drain (DECISIONS "Result-path optimization"). The int4 point-read route is ALWAYS i32 (NULL already
/// encoded as 0), so `values` holds raw `i32` rather than a fat `Vec<SqlValue>` (~24-32B/entry, ~3MB/65536-
/// batch) — the batcher maps `i32 -> DbValue::Int4` directly, skipping the SqlValue intermediate entirely.
/// The batcher slices `values` by `needle_ranges[i]` to answer needle `i`'s request, mapping the schema ONCE.
#[derive(Debug, Clone)]
pub struct RelationalRetainedBatchResult {
    pub columns: Arc<Vec<RelationalColumn>>,
    pub access_path: Arc<RelationalAccessPath>,
    pub gpu_id: u16,
    /// All needles' projected i32 values, flat + row-major, concatenated in needle order.
    pub values: Vec<i32>,
    pub ncols: usize,
    /// Per needle (needle order): `(start_row, row_count)` into `values`.
    pub needle_ranges: Vec<(u32, u32)>,
}

impl RelationalRetainedBatchResult {
    pub fn needle_count(&self) -> usize {
        self.needle_ranges.len()
    }
    /// Needle `i`'s projected i32 values as a flat slice (`row_count*ncols`).
    pub fn needle_values(&self, i: usize) -> &[i32] {
        let (start, count) = self.needle_ranges[i];
        let nc = self.ncols;
        &self.values[start as usize * nc..(start as usize + count as usize) * nc]
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

impl From<Vec<Vec<SqlValue>>> for RowBlock {
    fn from(rows: Vec<Vec<SqlValue>>) -> Self {
        let ncols = rows.first().map(Vec::len).unwrap_or(0);
        // `ncols` is inferred from the FIRST row, so a ragged input would silently reshape (and an
        // empty-first-then-nonempty input would violate the ncols==0-implies-empty invariant). Every live
        // result path feeds uniform-width rows; assert it so a future ragged producer trips here in tests
        // (audit P3). Release builds skip the scan.
        debug_assert!(
            rows.iter().all(|r| r.len() == ncols),
            "RowBlock::from requires uniform row width (ragged rows would misreshape the flat buffer)"
        );
        Self {
            values: rows.into_iter().flatten().collect(),
            ncols,
        }
    }
}

impl PartialEq<Vec<Vec<SqlValue>>> for RowBlock {
    fn eq(&self, other: &Vec<Vec<SqlValue>>) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a == b.as_slice())
    }
}

impl std::ops::Index<usize> for RowBlock {
    type Output = [SqlValue];
    fn index(&self, i: usize) -> &[SqlValue] {
        self.row(i)
    }
}

impl<'a> IntoIterator for &'a RowBlock {
    type Item = &'a [SqlValue];
    type IntoIter = std::slice::Chunks<'a, SqlValue>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalSelectResult {
    /// `Arc`-shared so a BATCHED point-read (one result per needle, all sharing the same projected schema)
    /// refcount-clones the columns instead of DEEP-cloning the `String`-bearing `RelationalColumn`s per
    /// needle — that per-needle schema deep-clone was the entire ~3M end-to-end read cap (DECISIONS
    /// "Result-path optimization"; Arc-share = 15x, above the GPU drain). Reads deref transparently.
    pub columns: Arc<Vec<RelationalColumn>>,
    pub rows: RowBlock,
    pub planned_target: DeviceTarget,
    pub executed_target: DeviceTarget,
    pub fallback_reason: Option<FallbackReason>,
    /// `Arc`-shared for the same reason as `columns` (a batched point-read shares one access path).
    pub access_path: Arc<RelationalAccessPath>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RelationalCopyAdmissionProfile {
    pub rows: usize,
    pub render_sql_wal_payload_micros: u128,
    pub commit_total_micros: u128,
    pub wal_commit_flush_boundary_micros: u128,
    pub current_apply_total_micros: u128,
    pub row_prepare_micros: u128,
    pub unique_preflight_micros: u128,
    pub check_preflight_micros: u128,
    pub foreign_key_preflight_micros: u128,
    pub mvcc_insert_micros: u128,
    pub value_index_append_micros: u128,
    pub residency_invalidation_micros: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencySnapshot {
    pub gpu_id: u16,
    pub schema: String,
    pub table: String,
    pub generation: u64,
    pub row_count: usize,
    /// Number of row SLOTS the fixed-width device sections are sized for (`>= row_count`). Dense /
    /// sealed payloads have `capacity == row_count`; an OPEN shard (Slice 1b) exceeds it by the reserved
    /// headroom. Section OFFSETS derive from `capacity` (the `resident_device_*_column_offset` helpers
    /// read it); `row_count` bounds the live rows. `capacity == row_count` reproduces the dense layout.
    pub capacity: usize,
    pub column_count: usize,
    pub resident_bytes: u64,
    // NOTE: the heavy host-side row materialization (`resident_rows`) is NOT here -- it lives in the
    // separate, Arc-shared `RelationalResidencyEntry.host_rows`, so this GPU/catalog DESCRIPTOR stays
    // lightweight and is never deep-copied on the GPU read path (it is the device executor's contract;
    // only the CPU / enumerated paths read host rows). See `RelationalResidencyEntry`.
    pub resident_device_int4_columns: Vec<String>,
    pub resident_device_int4_column_stats: Vec<ResidentDeviceInt4ColumnStats>,
    /// int8 columns retained in the device payload (fixed 8-byte row-major, after the int4 section,
    /// before the text section), in catalog order — the general GPU executor reads int8 predicates /
    /// projections from here (the type matrix, doc 19). Empty for sharded / benchmark installs
    /// that have not adopted int8 retention yet.
    pub resident_device_int8_columns: Vec<String>,
    /// numeric columns retained in the device payload (fixed 16-byte i128 mantissa, row-major, after
    /// the int8 section, before the text section), in catalog order — the general GPU executor reads
    /// numeric predicates / projections from here (the type matrix, doc 19). The decimal SCALE is a
    /// per-column catalog constant (values are rescaled on insert), so only the mantissa is stored.
    /// Empty for sharded / benchmark installs that have not adopted numeric retention yet.
    pub resident_device_numeric_columns: Vec<String>,
    /// bool columns retained as 1-bit-per-row bitmaps (the type matrix, doc 19), in catalog order.
    /// Self-describing: each carries its bitmap's byte offset. Empty for installs not retaining bool.
    pub resident_device_bool_columns: Vec<ResidentDeviceBoolColumnLayout>,
    pub resident_device_text_columns: Vec<ResidentDeviceTextColumnLayout>,
    /// Per-column NULL validity bitmaps (M3 — doc 21), one entry per column that contains a NULL
    /// (1 = valid, 0 = NULL). Empty when no column has a NULL — the common case today, since ingest of
    /// NULL is a later slice — so existing payloads/descriptors are unchanged. See
    /// [`ResidentDeviceNullBitmapLayout`].
    pub resident_device_null_columns: Vec<ResidentDeviceNullBitmapLayout>,
    pub valid_through_index: Index,
    pub invalidated_by_txn_id: Option<TxnId>,
    pub invalidated_at_index: Option<Index>,
    pub invalidated_by_memory_pressure: bool,
    pub memory_pressure_active: bool,
    pub last_refresh_cost: Option<RelationalResidencyRefreshCost>,
    pub admission_budget_bytes: Option<u64>,
    pub resident_bytes_after_admission: u64,
    pub evicted_tables_on_admission: Vec<String>,
    pub device_memory_proof: Option<CudaDeviceMemoryProof>,
}

/// A published, immutable per-table residency entry (the value stored in the residency snapshot map).
/// Splits the lightweight GPU/catalog DESCRIPTOR ([`RelationalResidencySnapshot`]) from the heavy
/// host-side row materialization, each `Arc`-shared so a reader -- or a `with_snapshots_mut` map clone
/// (every invalidation / DDL) -- bumps a refcount instead of deep-copying the rows. The GPU executor
/// reads only `descriptor`; the CPU / enumerated paths read `host_rows`. Keeping the host copy (the
/// charter's "CPU materialization = debt") off the device read path AND independently shareable is the
/// architectural point: `relational_residency_snapshot_ref` returns the descriptor;
/// `relational_residency_host_rows` returns the rows.
#[derive(Debug, Clone)]
pub struct RelationalResidencyEntry {
    pub descriptor: std::sync::Arc<RelationalResidencySnapshot>,
    pub host_rows: std::sync::Arc<Vec<Vec<SqlValue>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalRetainedSnapshotHandle {
    pub schema: String,
    pub table: String,
    pub gpu_id: u16,
    pub generation: u64,
    pub row_count: usize,
    pub column_count: usize,
    pub resident_bytes: u64,
    pub valid_through_index: Index,
    pub valid: bool,
    pub has_retained_device_memory: bool,
    pub resident_device_int4_columns: Vec<String>,
    pub resident_device_text_columns: Vec<ResidentDeviceTextColumnLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationalRetainedReadParam {
    Int4Eq { column: String, value: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalRetainedReadJob {
    pub route_id: String,
    pub schema: String,
    pub table: String,
    pub snapshot_generation: u64,
    pub params: Vec<RelationalRetainedReadParam>,
    pub(crate) select: Select,
}

/// The **needle-invariant** plan for a batchable int4-equality point lookup, prepared ONCE per shape
/// (`prepare_relational_retained_read_template`) and reused across every needle in a batch by
/// `submit_relational_retained_template_point_lookups`. It is the route descriptor the point-lookup
/// batcher (and, later, the persistent-kernel wave engine) enqueues: everything *except* the
/// per-request `i32` needle. Splitting it out removes the per-request host re-plan/re-bind that
/// bottlenecked the single coalescer thread at ~15µs/item (DECISIONS ADR-008 "First measurement").
/// `result_columns` + `access_path` are the fields the completion stamps onto every per-needle
/// result — verified needle-invariant (the same for all needles of one shape).
#[derive(Debug, Clone)]
pub struct RelationalRetainedReadTemplate {
    pub route_id: String,
    pub schema: String,
    pub table: RelationalTable,
    pub snapshot_generation: u64,
    pub(crate) selected_indexes: Vec<usize>,
    pub(crate) result_columns: Vec<RelationalColumn>,
    pub(crate) filter_idx: usize,
    pub(crate) access_path: RelationalAccessPath,
}

impl RelationalRetainedReadTemplate {
    /// True iff every projected column is int4 — the only projection the fast resident `equal_any`
    /// kernel can materialize (route shapes `int4_equality_projection` / `_multi_column_`). A mixed
    /// int4+text projection (`int4_equality_mixed_column_projection`) returns false: the int4 kernel
    /// cannot project text, so the batcher must route those to the text-capable general batch path
    /// (mirrors the jobs path's int4-only gate in `try_submit_relational_retained_int4_projection_jobs`).
    pub fn is_int4_only_projection(&self) -> bool {
        self.result_columns
            .iter()
            .all(|column| column.ty == SqlType::Int4)
    }
}

pub struct RelationalRetainedReadSubmission {
    pub route_id: String,
    pub table: String,
    pub snapshot_generation: u64,
    pub job_count: usize,
    pub submit_wall_micros: u64,
    pub(crate) inner: RelationalRetainedReadSubmissionInner,
}

impl RelationalRetainedReadSubmission {
    pub fn is_pending(&self) -> bool {
        matches!(
            self.inner,
            RelationalRetainedReadSubmissionInner::PendingInt4Projection(_)
        )
    }

    pub fn complete_detached(self) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        match self.inner {
            RelationalRetainedReadSubmissionInner::Ready(results) => Ok(results),
            RelationalRetainedReadSubmissionInner::PendingInt4Projection(pending) => Ok(
                Engine::complete_relational_retained_int4_projection_submission_detached(*pending)?
                    .results,
            ),
        }
    }
}

pub(crate) enum RelationalRetainedReadSubmissionInner {
    Ready(Vec<RelationalSelectResult>),
    // Boxed: this variant's payload is a large struct (table + several Vecs + a CUDA submission),
    // dwarfing the sibling `Ready(Vec<..>)`; boxing keeps the enum small to move (clippy
    // large_enum_variant). The submission is heap-heavy and created once per batch, so the box
    // alloc is negligible.
    PendingInt4Projection(Box<RelationalRetainedInt4ProjectionSubmission>),
}

pub(crate) struct RelationalRetainedInt4ProjectionSubmission {
    pub(crate) table: RelationalTable,
    pub(crate) snapshot_gpu_id: u16,
    pub(crate) selected_indexes: Vec<usize>,
    /// The batch's needle count + the SHARED projected schema (needle-invariant). Earlier this was a
    /// `Vec<(Arc, Arc, i32)>` with one entry per needle, but every entry held the SAME two `Arc`s and the
    /// needle value was unused by both completions — so building N tuples (2N `Arc` clones + a 1.5MB Vec at
    /// b65536) was ~358us of pure per-batch waste (DECISIONS "Result-path optimization"). The hot batched
    /// completion needs only `needle_count` + the shared schema; the cold per-needle completion refcount-
    /// clones the schema `needle_count` times at completion instead.
    pub(crate) needle_count: usize,
    pub(crate) shared_columns: Arc<Vec<RelationalColumn>>,
    pub(crate) shared_access_path: Arc<RelationalAccessPath>,
    pub(crate) before_metrics: RuntimeMetricsSnapshot,
    pub(crate) batch_started: Instant,
    pub(crate) payload: DeferredProbe,
}

/// The deferred (lpb) GPU submission, either the atomic-compaction kernel (the scan + the original unique
/// index probe) or the DENSE-emit unique index probe (DECISIONS "lpb read levers" #1). Both drain to the
/// SAME `CudaI32BatchProjectionColumns` via `complete_detached_columnar`, so the choice never changes results.
pub(crate) enum DeferredProbe {
    Atomic(CudaI32EqualAnyProjectSubmission),
    Dense(CudaI32IndexProbeDenseSubmission),
}

pub(crate) struct RelationalRetainedInt4ProjectionCompletion {
    pub(crate) table_name: String,
    pub(crate) before_metrics: RuntimeMetricsSnapshot,
    pub(crate) batch_micros: u64,
    pub(crate) wall_micros: u64,
    pub(crate) materialization_micros: u64,
    pub(crate) total_rows: usize,
    pub(crate) int4_result_columns: usize,
    pub(crate) kernel_event_elapsed_us: Option<u64>,
    pub(crate) results: Vec<RelationalSelectResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentDeviceInt4ColumnStats {
    pub name: String,
    pub min: i32,
    pub max: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentDeviceTextColumnLayout {
    pub name: String,
    pub offsets_byte_offset: u64,
    pub bytes_byte_offset: u64,
    pub bytes_len: u64,
}

/// A `bool` column retained in the device payload as a 1-bit-per-row BITMAP (the type matrix, doc 19):
/// `ceil(capacity / 32)` little-endian u32 words, bit `i` (LSB-first within its word) = row `i`'s
/// value. 1 bit/row -- 32x denser than the i32 sections, and a near-ready predicate mask. NULLs are a
/// separate VALIDITY bitmap ([`ResidentDeviceNullBitmapLayout`], M3 — doc 21), so this stores only the
/// value bit. The section is self-describing (like text): the bitmap's byte offset is recorded at build
/// time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentDeviceBoolColumnLayout {
    pub name: String,
    pub bitmap_byte_offset: u64,
}

/// A column's per-row NULL VALIDITY bitmap in the device payload (M3 — doc 21):
/// `ceil(capacity / 32)` little-endian u32 words, bit `i` (LSB-first) = row `i`, where **1 = valid
/// (present), 0 = NULL** (Arrow / PostgreSQL convention). One layout is emitted ONLY for a column that
/// actually contains a NULL; a column with no NULLs has NO bitmap (absence ⇒ all-valid), so non-nullable
/// columns and pre-M3 payloads stay byte-identical. The kernels read it on-device to honor three-valued
/// logic. Self-describing like the bool/text layouts: the byte offset is recorded at build time, and the
/// section starts 4-aligned (every preceding section is a multiple of 4 bytes) so the u32 words load
/// safely. Applies to any column type (the value lives in its own typed section).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentDeviceNullBitmapLayout {
    pub name: String,
    pub bitmap_byte_offset: u64,
}

pub struct BenchmarkRelationalResidencyChunkInstall<'a> {
    pub table: &'a str,
    pub gpu_id: u16,
    pub row_count: usize,
    pub resident_bytes: u64,
    pub resident_device_int4_columns: Vec<String>,
    pub resident_device_int4_column_stats: Vec<ResidentDeviceInt4ColumnStats>,
    pub resident_device_text_columns: Vec<ResidentDeviceTextColumnLayout>,
    pub allocated_bytes: u64,
    pub chunks: &'a [CudaDeviceMemoryChunk<'a>],
}

pub struct BenchmarkRelationalResidencyOwnedChunkInstall<'a, I>
where
    I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
{
    pub table: &'a str,
    pub gpu_id: u16,
    pub row_count: usize,
    pub resident_bytes: u64,
    pub resident_device_int4_columns: Vec<String>,
    pub resident_device_int4_column_stats: Vec<ResidentDeviceInt4ColumnStats>,
    pub resident_device_text_columns: Vec<ResidentDeviceTextColumnLayout>,
    pub allocated_bytes: u64,
    pub chunks: I,
}

pub struct BenchmarkRelationalResidencyOwnedShard {
    pub shard_id: u32,
    pub row_start: usize,
    pub row_count: usize,
    pub resident_bytes: u64,
    pub allocated_bytes: u64,
    pub resident_device_int4_columns: Vec<String>,
    pub resident_device_text_columns: Vec<ResidentDeviceTextColumnLayout>,
    pub chunks: Vec<CudaOwnedDeviceMemoryChunk>,
}

pub struct BenchmarkRelationalResidencyOwnedShardInstall<'a> {
    pub table: &'a str,
    pub gpu_id: u16,
    pub shards: Vec<BenchmarkRelationalResidencyOwnedShard>,
}

impl RelationalResidencySnapshot {
    pub fn is_valid(&self) -> bool {
        self.invalidated_by_txn_id.is_none()
            && self.invalidated_at_index.is_none()
            && !self.invalidated_by_memory_pressure
            && !self.memory_pressure_active
    }

    pub(crate) fn next_generation(previous: Option<&Self>) -> u64 {
        previous
            .map(|snapshot| snapshot.generation.saturating_add(1))
            .unwrap_or(1)
    }
}

pub(crate) fn validate_bootstrap_create_extension(
    create: &CreateExtension,
) -> Result<(), EngineError> {
    if create.name != "plpgsql" {
        return Err(EngineError::ApplyFailed(
            "only the bootstrap plpgsql extension is supported".to_string(),
        ));
    }
    if create
        .schema
        .as_deref()
        .is_some_and(|schema| schema != "pg_catalog")
    {
        return Err(EngineError::ApplyFailed(
            "plpgsql extension creation is only supported in pg_catalog".to_string(),
        ));
    }
    if !create.if_not_exists {
        return Err(EngineError::ApplyFailed(
            "extension \"plpgsql\" already exists".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_bootstrap_drop_extension(drop: &DropExtension) -> Result<(), EngineError> {
    if drop.name != "plpgsql" {
        return Err(EngineError::ApplyFailed(format!(
            "extension \"{}\" does not exist",
            drop.name
        )));
    }
    if !drop.if_exists {
        return Err(EngineError::ApplyFailed(
            "cannot drop bootstrap extension \"plpgsql\"".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn parse_bounded_sql_function_body(
    body: &str,
    return_type: SqlType,
) -> Result<SqlValue, ExecuteError> {
    let Some(rest) = strip_keyword_prefix_case_insensitive(body.trim(), "SELECT") else {
        return Err(unsupported_function_body_error());
    };
    let literal = rest.trim();
    if literal.is_empty()
        || find_keyword_outside_quotes(literal, "FROM").is_some()
        || find_keyword_outside_quotes(literal, "WHERE").is_some()
        || find_keyword_outside_quotes(literal, "ORDER").is_some()
        || find_keyword_outside_quotes(literal, "GROUP").is_some()
        || find_keyword_outside_quotes(literal, "LIMIT").is_some()
        || find_keyword_outside_quotes(literal, "OFFSET").is_some()
        || literal.contains(',')
    {
        return Err(unsupported_function_body_error());
    }
    match return_type {
        SqlType::Int2 => literal
            .parse::<i16>()
            .map(SqlValue::Int2)
            .map_err(|_| unsupported_function_body_error()),
        SqlType::Int4 => literal
            .parse::<i32>()
            .map(SqlValue::Int4)
            .map_err(|_| unsupported_function_body_error()),
        SqlType::Int8 => literal
            .parse::<i64>()
            .map(SqlValue::Int8)
            .map_err(|_| unsupported_function_body_error()),
        SqlType::Numeric { scale, .. } => Decimal128::parse_at_scale(literal, scale)
            .map(SqlValue::Numeric)
            .ok_or_else(unsupported_function_body_error),
        SqlType::Bool => match literal.to_ascii_lowercase().as_str() {
            "true" | "t" => Ok(SqlValue::Bool(true)),
            "false" | "f" => Ok(SqlValue::Bool(false)),
            _ => Err(unsupported_function_body_error()),
        },
        SqlType::Text => parse_bounded_text_literal(literal)
            .map(SqlValue::Text)
            .ok_or_else(unsupported_function_body_error),
        SqlType::Date => gpu_db_sql::datetime::parse_date(literal)
            .map(SqlValue::Date)
            .ok_or_else(unsupported_function_body_error),
        SqlType::Timestamp => gpu_db_sql::datetime::parse_timestamp(literal)
            .map(SqlValue::Timestamp)
            .ok_or_else(unsupported_function_body_error),
        SqlType::Uuid => gpu_db_sql::uuid::parse_uuid(literal)
            .map(SqlValue::Uuid)
            .ok_or_else(unsupported_function_body_error),
    }
}

pub(crate) fn strip_keyword_prefix_case_insensitive<'a>(
    input: &'a str,
    keyword: &str,
) -> Option<&'a str> {
    if input.len() < keyword.len() {
        return None;
    }
    let (head, tail) = input.split_at(keyword.len());
    if !head.eq_ignore_ascii_case(keyword) {
        return None;
    }
    if tail
        .chars()
        .next()
        .is_some_and(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())
    {
        return None;
    }
    Some(tail)
}

pub(crate) fn find_keyword_outside_quotes(input: &str, keyword: &str) -> Option<usize> {
    let bytes = input.as_bytes();
    let keyword_bytes = keyword.as_bytes();
    let mut idx = 0;
    let mut in_quote = false;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            if in_quote && idx + 1 < bytes.len() && bytes[idx + 1] == b'\'' {
                idx += 2;
                continue;
            }
            in_quote = !in_quote;
            idx += 1;
            continue;
        }
        if !in_quote
            && idx + keyword_bytes.len() <= bytes.len()
            && input[idx..idx + keyword_bytes.len()].eq_ignore_ascii_case(keyword)
        {
            let before_ok = idx == 0
                || !bytes[idx - 1].is_ascii_alphanumeric()
                    && bytes[idx - 1] != b'_'
                    && bytes[idx - 1] != b'$';
            let after_idx = idx + keyword_bytes.len();
            let after_ok = after_idx == bytes.len()
                || !bytes[after_idx].is_ascii_alphanumeric()
                    && bytes[after_idx] != b'_'
                    && bytes[after_idx] != b'$';
            if before_ok && after_ok {
                return Some(idx);
            }
        }
        idx += 1;
    }
    None
}

pub(crate) fn parse_bounded_text_literal(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    if bytes.len() < 2 || bytes.first() != Some(&b'\'') || bytes.last() != Some(&b'\'') {
        return None;
    }
    let inner = &input[1..input.len() - 1];
    let mut result = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            if chars.peek() == Some(&'\'') {
                chars.next();
                result.push('\'');
            } else {
                return None;
            }
        } else {
            result.push(ch);
        }
    }
    Some(result)
}

pub(crate) fn unsupported_function_body_error() -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(
        "only literal SELECT bodies are supported for SQL function execution".to_string(),
    ))
}

pub(crate) fn resident_device_int4_column_offset(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Result<u64, ExecuteError> {
    let column = table.columns.get(column_idx).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is outside the catalog table".to_string(),
        ))
    })?;
    // int4, date and int2 share the i32 section (a date is i32 days; a smallint widens to i32).
    if !matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is not int4/date/int2".to_string(),
        )));
    }
    let int4_ordinal = table
        .columns
        .iter()
        .take(column_idx)
        .filter(|candidate| matches!(candidate.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
        .count();
    if snapshot
        .resident_device_int4_columns
        .get(int4_ordinal)
        .is_none_or(|name| name != &column.name)
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident snapshot device payload has no int4 column \"{}\"",
            column.name
        ))));
    }
    // Section offsets use the fixed-width CAPACITY the sections were sized for (== row_count for a dense
    // payload; > row_count for an open shard with reserved headroom), so a padded shard addresses right.
    let capacity = u64::try_from(snapshot.capacity).map_err(|_| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident snapshot capacity exceeds retained device-memory proof range".to_string(),
        ))
    })?;
    let int4_width = std::mem::size_of::<i32>() as u64;
    let offset = capacity
        .checked_mul(int4_width)
        .and_then(|column_bytes| {
            (int4_ordinal as u64)
                .checked_mul(column_bytes)
                .and_then(|prefix_bytes| {
                    (std::mem::size_of::<u64>() as u64).checked_add(prefix_bytes)
                })
        })
        .ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot int4 payload offset overflowed".to_string(),
            ))
        })?;
    Ok(offset)
}

/// Byte offset of bool column `column_idx`'s 1-bit-per-row bitmap within the retained device payload
/// (the type matrix, doc 19). Self-describing like text: the offset was recorded at build time, so
/// this is a lookup by column name (no section-size math). Validates the column is bool and present.
pub(crate) fn resident_device_bool_column_offset(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Result<u64, ExecuteError> {
    let column = table.columns.get(column_idx).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is outside the catalog table".to_string(),
        ))
    })?;
    if column.ty != SqlType::Bool {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is not bool".to_string(),
        )));
    }
    snapshot
        .resident_device_bool_columns
        .iter()
        .find(|layout| layout.name == column.name)
        .map(|layout| layout.bitmap_byte_offset)
        .ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident snapshot device payload has no bool column \"{}\"",
                column.name
            )))
        })
}

/// Byte offset of column `column_idx`'s NULL VALIDITY bitmap within the retained device payload (M3 —
/// doc 21), or `None` if the column has NO bitmap, which means it contains no NULLs ⇒ **all rows valid**.
/// Unlike the bool/int4 helpers, a missing entry is the NORMAL all-valid case (not an error): a column
/// only gets a bitmap when it actually holds a NULL. A kernel reads `Some(offset)` as the validity
/// bitmap (1 = valid, 0 = NULL) and `None` as "every row valid". Applies to any column type. Errors only
/// if `column_idx` is outside the catalog table.
pub(crate) fn resident_device_null_column_offset(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Result<Option<u64>, ExecuteError> {
    let column = table.columns.get(column_idx).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is outside the catalog table".to_string(),
        ))
    })?;
    Ok(snapshot
        .resident_device_null_columns
        .iter()
        .find(|layout| layout.name == column.name)
        .map(|layout| layout.bitmap_byte_offset))
}

/// Byte offset of int8 column `column_idx` within the retained device payload (the type matrix, doc
/// 19). Layout: header (u64) + the WHOLE int4 section (`int4_columns * capacity * 4`) + the int8
/// columns before this one (`int8_ordinal * capacity * 8`). Validates the column is int8 and present
/// in `snapshot.resident_device_int8_columns`. Mirrors [`resident_device_int4_column_offset`].
// ALIGNMENT INVARIANT: this offset is 4-mod-8 (NOT 8-aligned) exactly when `(#int4 columns ×
// row_count)` is odd, because the int8 section follows the int4 section. Any device kernel that reads
// a 64-bit value here MUST do it as two 4-byte loads, never a single `ld.u64` -- a misaligned 64-bit
// load faults (CUDA 716 / illegal address) and poisons the context. This is reachable, not theoretical.
pub(crate) fn resident_device_int8_column_offset(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Result<u64, ExecuteError> {
    let column = table.columns.get(column_idx).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is outside the catalog table".to_string(),
        ))
    })?;
    // int8 and timestamp share the i64 section (a timestamp is i64 microseconds), so resolve either.
    if !matches!(column.ty, SqlType::Int8 | SqlType::Timestamp) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is not int8/timestamp".to_string(),
        )));
    }
    let int8_ordinal = table
        .columns
        .iter()
        .take(column_idx)
        .filter(|candidate| matches!(candidate.ty, SqlType::Int8 | SqlType::Timestamp))
        .count();
    if snapshot
        .resident_device_int8_columns
        .get(int8_ordinal)
        .is_none_or(|name| name != &column.name)
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident snapshot device payload has no int8 column \"{}\"",
            column.name
        ))));
    }
    // Section offsets use the fixed-width CAPACITY (== row_count dense; > row_count for an open shard).
    let capacity = u64::try_from(snapshot.capacity).map_err(|_| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident snapshot capacity exceeds retained device-memory proof range".to_string(),
        ))
    })?;
    let payload_offset_overflow = || {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident snapshot int8 payload offset overflowed".to_string(),
        ))
    };
    let int4_section_bytes = capacity
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|int4_col_bytes| {
            (snapshot.resident_device_int4_columns.len() as u64).checked_mul(int4_col_bytes)
        })
        .ok_or_else(payload_offset_overflow)?;
    let int8_prefix_bytes = capacity
        .checked_mul(std::mem::size_of::<i64>() as u64)
        .and_then(|int8_col_bytes| (int8_ordinal as u64).checked_mul(int8_col_bytes))
        .ok_or_else(payload_offset_overflow)?;
    (std::mem::size_of::<u64>() as u64)
        .checked_add(int4_section_bytes)
        .and_then(|after_int4| after_int4.checked_add(int8_prefix_bytes))
        .ok_or_else(payload_offset_overflow)
}

/// Byte offset of numeric column `column_idx` within the retained device payload (the type matrix,
/// doc 19). Layout: header (u64) + the WHOLE int4 section + the WHOLE int8 section + the numeric
/// columns before this one (`numeric_ordinal * capacity * 16`). Validates the column is numeric and
/// present in `snapshot.resident_device_numeric_columns`. A numeric mantissa is a fixed 16-byte i128;
/// the decimal scale is the column's catalog scale (values are rescaled on insert), not stored here.
pub(crate) fn resident_device_numeric_column_offset(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Result<u64, ExecuteError> {
    let column = table.columns.get(column_idx).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is outside the catalog table".to_string(),
        ))
    })?;
    // numeric and uuid share the 16-byte section (a uuid is 16 raw bytes), so resolve either here.
    if !matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is not numeric/uuid".to_string(),
        )));
    }
    let numeric_ordinal = table
        .columns
        .iter()
        .take(column_idx)
        .filter(|candidate| matches!(candidate.ty, SqlType::Numeric { .. } | SqlType::Uuid))
        .count();
    if snapshot
        .resident_device_numeric_columns
        .get(numeric_ordinal)
        .is_none_or(|name| name != &column.name)
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident snapshot device payload has no numeric column \"{}\"",
            column.name
        ))));
    }
    // Section offsets use the fixed-width CAPACITY (== row_count dense; > row_count for an open shard).
    let capacity = u64::try_from(snapshot.capacity).map_err(|_| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident snapshot capacity exceeds retained device-memory proof range".to_string(),
        ))
    })?;
    let payload_offset_overflow = || {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident snapshot numeric payload offset overflowed".to_string(),
        ))
    };
    let int4_section_bytes = capacity
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|int4_col_bytes| {
            (snapshot.resident_device_int4_columns.len() as u64).checked_mul(int4_col_bytes)
        })
        .ok_or_else(payload_offset_overflow)?;
    let int8_section_bytes = capacity
        .checked_mul(std::mem::size_of::<i64>() as u64)
        .and_then(|int8_col_bytes| {
            (snapshot.resident_device_int8_columns.len() as u64).checked_mul(int8_col_bytes)
        })
        .ok_or_else(payload_offset_overflow)?;
    let numeric_prefix_bytes = capacity
        .checked_mul(std::mem::size_of::<i128>() as u64)
        .and_then(|numeric_col_bytes| (numeric_ordinal as u64).checked_mul(numeric_col_bytes))
        .ok_or_else(payload_offset_overflow)?;
    (std::mem::size_of::<u64>() as u64)
        .checked_add(int4_section_bytes)
        .and_then(|after_int4| after_int4.checked_add(int8_section_bytes))
        .and_then(|after_int8| after_int8.checked_add(numeric_prefix_bytes))
        .ok_or_else(payload_offset_overflow)
}

/// Resolve the device payload byte offset of an int4 OR int8 column by its catalog type (the type
/// matrix, doc 19) — the type-dispatching resolver the arith VM lowering uses for `Column` leaves.
pub(crate) fn resident_device_int_column_offset(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Result<u64, ExecuteError> {
    match table.columns.get(column_idx).map(|column| column.ty) {
        // int2 (smallint) is stored WIDENED to i32 in the int4 section, so it reads as an int4 column.
        Some(SqlType::Int4 | SqlType::Int2) => {
            resident_device_int4_column_offset(snapshot, table, column_idx)
        }
        Some(SqlType::Int8) => resident_device_int8_column_offset(snapshot, table, column_idx),
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory column is neither int2/int4 nor int8".to_string(),
        ))),
    }
}

pub(crate) fn resident_device_int4_column_stats<'a>(
    snapshot: &'a RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Option<&'a ResidentDeviceInt4ColumnStats> {
    let column = table.columns.get(column_idx)?;
    if column.ty != SqlType::Int4 {
        return None;
    }
    snapshot
        .resident_device_int4_column_stats
        .iter()
        .find(|stats| stats.name == column.name)
}

pub(crate) fn resident_i32_comparison_domain_is_empty(
    stats: &ResidentDeviceInt4ColumnStats,
    needle: i32,
    comparison: CudaI32Comparison,
) -> bool {
    if stats.min > stats.max {
        return true;
    }
    match comparison {
        CudaI32Comparison::Lt => stats.min >= needle,
        CudaI32Comparison::Lte => stats.min > needle,
        CudaI32Comparison::Gt => stats.max <= needle,
        CudaI32Comparison::Gte => stats.max < needle,
    }
}

pub(crate) fn acl_relation_kind_label(kind: AclRelationKind) -> &'static str {
    match kind {
        AclRelationKind::Relation => "relation",
        AclRelationKind::Table => "table",
        AclRelationKind::View => "view",
        AclRelationKind::MaterializedView => "materialized view",
        AclRelationKind::Sequence => "sequence",
    }
}

pub(crate) fn resident_device_text_column_layout<'a>(
    snapshot: &'a RelationalResidencySnapshot,
    table: &RelationalTable,
    column_idx: usize,
) -> Result<&'a ResidentDeviceTextColumnLayout, ExecuteError> {
    let column = table.columns.get(column_idx).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory predicate column is outside the catalog table".to_string(),
        ))
    })?;
    if column.ty != SqlType::Text {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident device-memory text-prefix count proof currently supports only text predicates"
                .to_string(),
        )));
    }
    snapshot
        .resident_device_text_columns
        .iter()
        .find(|layout| layout.name == column.name)
        .ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident snapshot device payload has no text column \"{}\"",
                column.name
            )))
        })
}

pub(crate) fn resident_device_i32_comparison(op: SelectFilterOp) -> Option<CudaI32Comparison> {
    match op {
        SelectFilterOp::Lt => Some(CudaI32Comparison::Lt),
        SelectFilterOp::Lte => Some(CudaI32Comparison::Lte),
        SelectFilterOp::Gt => Some(CudaI32Comparison::Gt),
        SelectFilterOp::Gte => Some(CudaI32Comparison::Gte),
        SelectFilterOp::Eq | SelectFilterOp::LikePrefix => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencyRefreshCost {
    pub previous_row_count: usize,
    pub refreshed_row_count: usize,
    pub row_delta: i128,
    pub previous_resident_bytes: u64,
    pub refreshed_resident_bytes: u64,
    pub resident_byte_delta: i128,
    pub refreshed_from_index: Index,
    pub refreshed_through_index: Index,
    pub invalidated_by_txn_id: Option<TxnId>,
    pub invalidated_at_index: Option<Index>,
    pub invalidated_by_memory_pressure: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RelationalResidencyWarmupPolicy {
    pub gpu_id: Option<u16>,
    pub tables: Vec<String>,
    pub max_table_count: Option<usize>,
    pub budget_bytes: Option<u64>,
    pub refresh_invalidated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencyWarmupReport {
    pub gpu_id: u16,
    pub budget_bytes: Option<u64>,
    pub requested_tables: Vec<String>,
    pub entries: Vec<RelationalResidencyWarmupEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencyWarmupEntry {
    pub table: String,
    pub action: RelationalResidencyWarmupAction,
    pub reason: String,
    pub resident_bytes: u64,
    pub evicted_tables: Vec<String>,
    pub route_decision: Option<RelationalResidentRouteDecisionStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationalResidencyWarmupAction {
    Warmed,
    Refreshed,
    AlreadyResident,
    Skipped,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencyMaintenancePolicy {
    pub gpu_id: Option<u16>,
    pub tables: Vec<String>,
    pub max_table_count: Option<usize>,
    pub budget_bytes: Option<u64>,
    pub refresh_invalidated: bool,
}

impl Default for RelationalResidencyMaintenancePolicy {
    fn default() -> Self {
        Self {
            gpu_id: None,
            tables: Vec::new(),
            max_table_count: None,
            budget_bytes: None,
            refresh_invalidated: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencyMaintenanceReport {
    pub gpu_id: u16,
    pub budget_bytes: Option<u64>,
    pub requested_tables: Vec<String>,
    pub entry_count: usize,
    pub warmed_count: usize,
    pub refreshed_count: usize,
    pub already_resident_count: usize,
    pub skipped_count: usize,
    pub error_count: usize,
    pub route_ready_count: usize,
    pub route_blocked_count: usize,
    pub route_ready_tables: Vec<String>,
    pub route_blockers: Vec<RelationalResidencyMaintenanceBlocker>,
    pub entries: Vec<RelationalResidencyWarmupEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalResidencyMaintenanceBlocker {
    pub table: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationalAccessPath {
    FullTableScan,
    EqualityIndex {
        table: String,
        column: String,
        matched_keys: usize,
    },
    FilteredKeyBatch {
        table: String,
        predicate_column: String,
        predicate_op: SelectFilterOp,
        matched_keys: usize,
    },
    ConjunctiveFilteredKeyBatch {
        table: String,
        predicate_count: usize,
        matched_keys: usize,
    },
    DisjunctiveFilteredKeyBatch {
        table: String,
        predicate_group_count: usize,
        matched_keys: usize,
    },
    OrderedKeyBatch {
        table: String,
        predicate_column: Option<String>,
        predicate_op: Option<SelectFilterOp>,
        order_column: String,
        descending: bool,
        matched_keys: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalSqlGpuBridgeReport {
    pub query_count: usize,
    pub gpu_executed_count: usize,
    pub cpu_fallback_count: usize,
    pub gpu_executed_permyriad: u16,
    pub cpu_fallback_permyriad: u16,
}

impl RelationalSqlGpuBridgeReport {
    pub fn from_results(results: &[RelationalSelectResult]) -> Self {
        let query_count = results.len();
        let gpu_executed_count = results
            .iter()
            .filter(|result| matches!(result.executed_target, DeviceTarget::Gpu(_)))
            .count();
        let cpu_fallback_count = results
            .iter()
            .filter(|result| result.fallback_reason.is_some())
            .count();

        Self {
            query_count,
            gpu_executed_count,
            cpu_fallback_count,
            gpu_executed_permyriad: permyriad(gpu_executed_count, query_count),
            cpu_fallback_permyriad: permyriad(cpu_fallback_count, query_count),
        }
    }
}
