//! Write-half MVCC write-path internals (P0 §9.6 decomposition, behavior-
//! preserving). The value types a `prepare_*` produces and `apply_delta`
//! installs: read-boundary snapshot, write-set + conflict keys, prepared
//! mutations, the recent-commits ledger (SI first-committer-wins), and the
//! active-snapshot multiset + RAII guard (oldest-active GC boundary). The
//! commit/apply LOGIC stays on `impl Engine`; these are its collaborators.

use super::*;

/// The off-lock read boundary a `prepare_*` reads against (write-half MVCC, Stage 2).
///
/// `commit_seq` is both the read visibility (`is_visible` boundary) AND the version stamp the
/// matching `apply_delta` writes — the Stage 0 unification. `next_row_id` snapshots
/// `relational_next_row_id` so `prepare_insert` can compute deterministic row keys without
/// mutating the engine. Under the still-serialized commit these equal the values the old direct
/// apply used; Stage 4 will take this snapshot at statement start instead.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DmlReadSnapshot {
    pub(crate) commit_seq: TxnId,
    pub(crate) next_row_id: u64,
}

/// A single row slot a write touches: `(table, row_key)`. Inserts claim a fresh slot; updates
/// rewrite an existing slot in place (old version tombstoned + new version at the same key);
/// deletes tombstone a slot. This is the per-row conflict point for Stage 4's first-committer-wins
/// SI validation (a recent-commits ledger keyed by `(table, row_key)`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RowWriteKey {
    pub(crate) table: String,
    pub(crate) row_key: String,
}

/// A unique-index slot a write claims or releases: `(table, column, value)` for a column carrying
/// a unique index. Two transactions writing the same unique slot conflict (first-committer-wins),
/// so this is the second conflict dimension Stage 4 validates. Non-unique value-index appends are
/// NOT conflict points and are carried in [`PreparedMutation`], not here.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct UniqueIndexSlotKey {
    pub(crate) table: String,
    pub(crate) column: String,
    pub(crate) value: String,
}

/// E2.2(a) — the INTEGER-keyed unique-slot conflict identity for the covered-INSERT INTENT fast
/// path: `(slot_id, value)` where `slot_id` packs the stable `(table_oid, column_id)` catalog
/// identity ([`pack_unique_slot_id`]) and `value` is the raw `i32` the row wrote into that
/// unique i32 column. Two writes to the same i32 unique slot conflict exactly as the String-keyed
/// [`UniqueIndexSlotKey`] would — but the hot path builds/records/checks them with ZERO
/// allocations (no `table`/`column` clone, no decimal-string format of the value).
///
/// CROSS-PATH INTEROP is exact and automatic: [`WriteSet::add_unique_slots`] (the single chokepoint
/// every path funnels through) emits the integer slot ALONGSIDE the String slot for every i32
/// unique column, so a classic-path write and an intent-path write to the same slot record into
/// the SAME integer map at the same `commit_seq` and see each other's conflicts. The intent path
/// builds ONLY the integer slot (skipping the String allocation entirely); the classic path keeps
/// both (the String slot stays authoritative for non-i32 columns and is a harmless redundancy for
/// i32 ones).
pub(crate) type IntUniqueSlotKey = (u64, i32);

/// Pack the stable catalog identity of a unique i32 column into the [`IntUniqueSlotKey`] slot id:
/// `(table_oid << 32) | column_id`. Both are catalog-assigned and monotonic, so a dropped+recreated
/// table (fresh oid) never aliases the old — strictly MORE precise than the name-keyed String slot,
/// and a hash-free exact identity every path can compute from the `RelationalTable`/`RelationalColumn`
/// already in hand.
pub(crate) fn pack_unique_slot_id(table_oid: u32, column_id: u32) -> u64 {
    ((table_oid as u64) << 32) | (column_id as u64)
}

/// The complete, reusable write-set a `prepare_*` computes: every row slot and every
/// unique-index slot the matching `apply_delta` will touch — no more, no less. Stage 4's
/// conflict detector consumes exactly this shape to validate a prepared txn against commits since
/// its snapshot. Kept as ordered `Vec`s (small per statement); callers that need set semantics can
/// collect into a `BTreeSet` (the keys are `Ord`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct WriteSet {
    pub(crate) rows: Vec<RowWriteKey>,
    pub(crate) unique_slots: Vec<UniqueIndexSlotKey>,
    /// E2.2(a) — the integer-keyed projection of the i32 unique slots (see [`IntUniqueSlotKey`]).
    /// The classic path fills this ALONGSIDE `unique_slots` (for i32 columns); the intent fast
    /// path fills ONLY this. Both feed the ledger's integer map, so cross-path conflicts are exact.
    pub(crate) unique_slots_i32: Vec<IntUniqueSlotKey>,
}

impl WriteSet {
    /// Append the unique-index slots `values` occupies for `table`. Mirrors
    /// `validate_unique_indexes_for_rows`: for each unique index, resolve the column position
    /// (skip if the column is absent, as the validator does) and record `(table, column, value)`.
    /// E2.2(a): for a unique i32 column also emit the allocation-free integer slot so classic-path
    /// writes share the integer conflict map with the intent fast path.
    pub(crate) fn add_unique_slots(&mut self, table: &RelationalTable, values: &[SqlValue]) {
        for index in table.indexes.iter().filter(|index| index.unique) {
            let Some(column_idx) = table
                .columns
                .iter()
                .position(|column| column.name == index.column)
            else {
                continue;
            };
            self.unique_slots.push(UniqueIndexSlotKey {
                table: table.name.clone(),
                column: index.column.clone(),
                value: relational_index_value(&values[column_idx]),
            });
            if let SqlValue::Int4(v) = &values[column_idx] {
                let column = &table.columns[column_idx];
                self.unique_slots_i32
                    .push((pack_unique_slot_id(table.oid, column.id), *v));
            }
        }
    }
}

/// The concrete, installable mutation a `prepare_*` produced, paired with its [`WriteSet`] in a
/// [`WriteDelta`]. Holds everything `apply_delta` needs to mutate engine state and nothing it must
/// recompute. The value-index entries (all columns, unique or not) are precomputed here so apply
/// is a pure install.
#[derive(Debug, Clone)]
pub(crate) enum PreparedMutation {
    /// New rows to install at reserved keys; `inserted_rows` is `(row_key, values)` and
    /// `value_index_entries` is the per-(table,column,value) row-key appends. `seq_advances`
    /// is the post-state (`last_value`, `is_called`) for each sequence consumed by `nextval`
    /// column defaults: `prepare_insert` reads the sequence state and computes the values
    /// purely (into a local scratch), recording the final advancement here for `apply_delta`
    /// to install — keeping prepare free of the sequence mutation `nextval` would otherwise do.
    Insert {
        table: String,
        inserted_rows: Vec<(String, Vec<SqlValue>)>,
        value_index_entries: BTreeMap<ColumnValueKey, Vec<String>>,
        seq_advances: BTreeMap<String, (i64, bool)>,
    },
    /// In-place version rewrites; `installs` is `(tuple_id, row_key, new_values)` (tuple_id is the
    /// existing version chain to tombstone+append onto), plus the new images' value-index appends.
    Update {
        table: String,
        installs: Vec<(u64, String, Vec<SqlValue>)>,
        value_index_entries: BTreeMap<ColumnValueKey, Vec<String>>,
        /// SV5: the OLD row images (catalog order), PARALLEL to `installs` (same order), captured BEFORE the
        /// assignments overwrote them. The commit path tombstones the old resident slot + appends the new
        /// image (from `installs`) in place instead of the O(table) re-admit.
        updated_old_rows: Vec<Vec<SqlValue>>,
    },
    /// Existing versions to tombstone, by tuple_id, in `table`'s partition. `deleted_rows` carries the
    /// resolved row images (catalog order) SV4b surfaces to the commit path so a single-entry DELETE can
    /// LOCATE + tombstone them on the resident GPU shard IN PLACE instead of the O(table) invalidate+re-admit.
    Delete {
        table: String,
        tuple_ids: Vec<u64>,
        deleted_rows: Vec<Vec<SqlValue>>,
    },
}

/// The row-level mutation a single committed log entry applied, surfaced by `apply_mvcc_entry` so the
/// commit path can maintain GPU residency INCREMENTALLY for a single-entry commit (INSERT=append,
/// DELETE=tombstone, UPDATE=tombstone old + append new) instead of the O(table) invalidate + re-admit,
/// and record the applied [`WriteSet`] into the SI recent-commits ledger (C2, write-path assessment:
/// the ledger must see SERIALIZED-path writes too, or a concurrent committer validating against an
/// older snapshot silently misses them — a lost update). `None` for every other command.
#[derive(Debug, Clone)]
pub(crate) enum AppliedRowMutation {
    Insert {
        table: String,
        rows: Vec<Vec<SqlValue>>,
        write_set: WriteSet,
        /// RETIREMENT A1: the installed rows' host identities (parsed from the delta's
        /// `inserted_rows` keys — INSERT write-sets deliberately carry no row keys, so the delta is
        /// the identity source). Parallel to `rows`.
        row_ids: Vec<u64>,
    },
    Delete {
        table: String,
        rows: Vec<Vec<SqlValue>>,
        write_set: WriteSet,
    },
    Update {
        table: String,
        old_rows: Vec<Vec<SqlValue>>,
        new_rows: Vec<Vec<SqlValue>>,
        /// RETIREMENT A4b: the updated rows' IDENTITIES (parsed from the installs' row keys, EXACT
        /// parallel to `old_rows`/`new_rows` by construction) — the appended new versions keep
        /// them (A1). `None` for any unparseable key -> the commit arm declines the incremental
        /// path (re-admit, always correct).
        row_ids: Option<Vec<u64>>,
        write_set: WriteSet,
    },
}

impl AppliedRowMutation {
    /// The `(table, row-key)` + unique-slot conflict footprint the apply installed — exactly the
    /// prepare-computed [`WriteSet`] of the underlying delta, for SI ledger recording.
    pub(crate) fn write_set(&self) -> &WriteSet {
        match self {
            Self::Insert { write_set, .. }
            | Self::Delete { write_set, .. }
            | Self::Update { write_set, .. } => write_set,
        }
    }
}

/// A prepared (but not yet installed) write: the [`WriteSet`] for conflict detection plus the
/// [`PreparedMutation`] to install (write-half MVCC, Stage 2). Produced PURELY by `prepare_*`
/// from a [`DmlReadSnapshot`] (no engine mutation); installed by [`Engine::apply_delta`] under
/// the commit lock, stamped with `commit_seq`. `rows_consumed` is how many row-ids the install
/// advances `relational_next_row_id` by (inserts only).
#[derive(Debug, Clone)]
pub(crate) struct WriteDelta {
    // Read by Stage 4's SI conflict detector (`RecentCommitsLedger::conflicts`) to validate the
    // prepared write-set against commits since the snapshot, AND recorded into the ledger after a
    // successful commit.
    pub(crate) write_set: WriteSet,
    pub(crate) rows_consumed: u64,
    pub(crate) mutation: PreparedMutation,
}

/// The recent-commits ledger: every committed write's `(table, row-key)` and unique-index slot →
/// the highest `commit_seq` that wrote it (write-half MVCC, Stage 4, conflict-detection §3.3). The
/// SI write-write check (first-committer-wins) is: a prepared txn with `read_snapshot = S` conflicts
/// iff ANY key in its write-set has a recorded `commit_seq > S` — i.e. some other transaction wrote
/// the same key AFTER this one took its snapshot. Pruned below the oldest active read snapshot (no
/// active transaction can still be reading before that boundary, so older ledger entries can never
/// be the "winner" of a future conflict) — which is also the safe MVCC GC boundary.
#[derive(Debug, Default)]
pub(crate) struct RecentCommitsLedger {
    pub(crate) rows: BTreeMap<RowWriteKey, Index>,
    pub(crate) unique_slots: BTreeMap<UniqueIndexSlotKey, Index>,
    /// E2.2(a) — the integer-keyed unique-slot map (see [`IntUniqueSlotKey`]). A strict projection
    /// of the i32 unique slots that BOTH paths maintain, so the intent fast path's allocation-free
    /// conflict check/record sees classic-path writes and vice versa. `HashMap`, not `BTreeMap`:
    /// this map is never range-scanned (prune walks it) and the O(1) probe is the hot-path win.
    pub(crate) unique_slots_i32: std::collections::HashMap<IntUniqueSlotKey, Index>,
}

impl RecentCommitsLedger {
    /// First-committer-wins SI validation: does any key in `write_set` carry a recorded commit
    /// strictly newer than `read_snapshot`? If so the preparing txn read a now-stale snapshot of
    /// that key and must abort (retryable). Equality on `read_snapshot` does NOT conflict — that is
    /// a commit this txn's snapshot already saw.
    pub(crate) fn conflicts(&self, write_set: &WriteSet, read_snapshot: Index) -> bool {
        write_set
            .rows
            .iter()
            .any(|key| self.rows.get(key).is_some_and(|&seq| seq > read_snapshot))
            || write_set.unique_slots.iter().any(|key| {
                self.unique_slots
                    .get(key)
                    .is_some_and(|&seq| seq > read_snapshot)
            })
            || write_set.unique_slots_i32.iter().any(|key| {
                self.unique_slots_i32
                    .get(key)
                    .is_some_and(|&seq| seq > read_snapshot)
            })
    }

    /// Record a committed write-set at `commit_seq` (the highest writer of each key wins — commits
    /// are assigned monotonically increasing `commit_seq` under the commit_mutex, so a later commit
    /// always overwrites with a larger value).
    /// E2.5b-2 — conflict check for ONE integer slot (the lean lane form).
    pub(crate) fn conflicts_int_slot(&self, slot: IntUniqueSlotKey, read_snapshot: Index) -> bool {
        self.unique_slots_i32
            .get(&slot)
            .is_some_and(|&seq| seq > read_snapshot)
    }

    /// E2.5b-2 — record ONE integer unique slot (the lane pump's fused-pass
    /// form; equivalent to `record` for a write-set holding exactly this slot).
    pub(crate) fn record_int_slot(&mut self, slot: IntUniqueSlotKey, commit_seq: Index) {
        self.unique_slots_i32.insert(slot, commit_seq);
    }

    pub(crate) fn record(&mut self, write_set: &WriteSet, commit_seq: Index) {
        for key in &write_set.rows {
            self.rows.insert(key.clone(), commit_seq);
        }
        for key in &write_set.unique_slots {
            self.unique_slots.insert(key.clone(), commit_seq);
        }
        for &key in &write_set.unique_slots_i32 {
            self.unique_slots_i32.insert(key, commit_seq);
        }
    }

    /// Drop entries written at or before `boundary` (no active snapshot reads before it, so they can
    /// never win a future conflict). Keeps the ledger bounded by the active-snapshot window.
    pub(crate) fn prune_below(&mut self, boundary: Index) {
        self.rows.retain(|_, &mut seq| seq > boundary);
        self.unique_slots.retain(|_, &mut seq| seq > boundary);
        self.unique_slots_i32.retain(|_, &mut seq| seq > boundary);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.rows.len() + self.unique_slots.len() + self.unique_slots_i32.len()
    }
}

/// Tracks the read snapshots of in-flight transactions (write-half MVCC, Stage 4) as an ordered
/// multiset of `read_snapshot` `commit_seq` values. A transaction registers its snapshot at
/// prepare-begin (off-lock) and deregisters at commit/abort. The MINIMUM registered snapshot is the
/// oldest-active boundary: ledger entries and MVCC versions older than it can be reclaimed because no
/// active transaction can still observe (or conflict against) them. This is the concrete
/// oldest-active-`commit_seq` the Stage-0 GC-boundary debt needed.
#[derive(Debug, Default)]
pub(crate) struct ActiveSnapshots {
    pub(crate) counts: BTreeMap<Index, usize>,
}

impl ActiveSnapshots {
    pub(crate) fn register(&mut self, snapshot: Index) {
        *self.counts.entry(snapshot).or_insert(0) += 1;
    }

    pub(crate) fn deregister(&mut self, snapshot: Index) {
        if let Some(count) = self.counts.get_mut(&snapshot) {
            *count -= 1;
            if *count == 0 {
                self.counts.remove(&snapshot);
            }
        }
    }

    /// The oldest active read snapshot, or `None` when no transaction is in flight.
    pub(crate) fn oldest(&self) -> Option<Index> {
        self.counts.keys().next().copied()
    }
}

/// RAII guard that deregisters a transaction's read snapshot from [`ActiveSnapshots`] on drop
/// (write-half MVCC, Stage 4), so a snapshot is released even if prepare/commit returns early (a
/// serialization abort, a constraint error) — keeping the oldest-active GC boundary from getting
/// stuck behind an aborted transaction.
pub(crate) struct ActiveSnapshotGuard<'a> {
    pub(crate) engine: &'a Engine,
    pub(crate) snapshot: Index,
}

impl Drop for ActiveSnapshotGuard<'_> {
    fn drop(&mut self) {
        self.engine.deregister_active_snapshot(self.snapshot);
    }
}
