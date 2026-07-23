//! Write-half MVCC write-path internals (P0 §9.6 decomposition, behavior-
//! preserving). The value types a `prepare_*` produces and `apply_delta`
//! installs: read-boundary snapshot, write-set + conflict keys, prepared
//! mutations, the recent-commits ledger (SI first-committer-wins), and the
//! active-snapshot registry + RAII guard (oldest-active GC boundary). The
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// Relations changed by this write. Row/unique keys may both be empty (for example an INSERT
    /// into a heap without indexes), so the relation footprint must be carried independently for
    /// FK dependency history and publication auditing.
    pub(crate) tables: BTreeSet<String>,
    pub(crate) rows: Vec<RowWriteKey>,
    pub(crate) unique_slots: Vec<UniqueIndexSlotKey>,
    /// E2.2(a) — the integer-keyed projection of the i32 unique slots (see [`IntUniqueSlotKey`]).
    /// The classic path fills this ALONGSIDE `unique_slots` (for i32 columns); the intent fast
    /// path fills ONLY this. Both feed the ledger's integer map, so cross-path conflicts are exact.
    pub(crate) unique_slots_i32: Vec<IntUniqueSlotKey>,
}

impl WriteSet {
    pub(crate) fn extend_deduplicated(&mut self, other: &Self) {
        self.tables.extend(other.tables.iter().cloned());
        self.rows.extend(other.rows.iter().cloned());
        self.rows.sort();
        self.rows.dedup();
        self.unique_slots.extend(other.unique_slots.iter().cloned());
        self.unique_slots.sort();
        self.unique_slots.dedup();
        self.unique_slots_i32
            .extend(other.unique_slots_i32.iter().copied());
        self.unique_slots_i32.sort_unstable();
        self.unique_slots_i32.dedup();
    }

    /// Append the unique-index slots `values` occupies for `table`. Mirrors
    /// `validate_unique_indexes_for_rows`: for each unique index, resolve the column position
    /// (skip if the column is absent, as the validator does) and record `(table, column, value)`.
    /// E2.2(a): for a unique i32 column also emit the allocation-free integer slot so classic-path
    /// writes share the integer conflict map with the intent fast path.
    pub(crate) fn add_unique_slots(&mut self, table: &RelationalTable, values: &[SqlValue]) {
        for index in table.indexes.iter().filter(|index| index.unique) {
            // COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): a compound unique index contributes ONE
            // conflict slot per row keyed by the whole ordered tuple — a string slot (the
            // index-name-qualified tuple, for the classic conflict map) and, when every key column
            // is an i32-section value, the integer fingerprint slot the intent fast path also emits
            // (so cross-path conflicts stay exact). Same-tuple concurrent writers share the
            // fingerprint -> a real conflict is never missed; a fingerprint collision between two
            // DISTINCT tuples only over-conflicts (safe, retryable).
            if crate::engine_residency::index_is_compound(index) {
                let Some(positions) =
                    crate::engine_residency::index_key_column_positions(table, index)
                else {
                    continue;
                };
                // Join the per-column index strings with a control-char separator so distinct tuples
                // can never concatenate to the same key (`(a="1|2", b="3")` vs `(a="1", b="2|3")`).
                let tuple_value: String = positions
                    .iter()
                    .map(|&i| relational_index_value(&values[i]))
                    .collect::<Vec<_>>()
                    .join("\u{1}");
                self.unique_slots.push(UniqueIndexSlotKey {
                    table: table.name.clone(),
                    column: index.name.clone(),
                    value: tuple_value,
                });
                if let Some(fp) =
                    crate::engine_residency::compound_index_row_fingerprint(table, index, values)
                {
                    self.unique_slots_i32.push((
                        crate::engine_residency::compound_unique_slot_id(table, index),
                        fp,
                    ));
                }
                continue;
            }
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
/// recompute.
#[derive(Debug, Clone)]
pub(crate) enum PreparedMutation {
    /// New rows to publish on the device; `inserted_rows` is `(row_key, values)`. `seq_advances`
    /// is the post-state (`last_value`, `is_called`) for each sequence consumed by `nextval`
    /// column defaults: `prepare_insert` reads the sequence state and computes the values
    /// purely (into a local scratch), recording the final advancement here for `apply_delta`
    /// to install — keeping prepare free of the sequence mutation `nextval` would otherwise do.
    Insert {
        table: String,
        inserted_rows: Vec<(String, Vec<SqlValue>)>,
        seq_advances: BTreeMap<String, (i64, bool)>,
    },
    /// In-place version rewrites; `installs` is `(tuple_id, row_key, new_values)` (tuple_id is the
    /// existing device entity to tombstone+append onto).
    Update {
        table: String,
        installs: Vec<(u64, String, Vec<SqlValue>)>,
        /// SV5: the OLD row images (catalog order), PARALLEL to `installs` (same order), captured BEFORE the
        /// assignments overwrote them. The commit path tombstones the old resident slot + appends the new
        /// image (from `installs`) in place in the authoritative device generation.
        updated_old_rows: Vec<Vec<SqlValue>>,
        /// P4-2b-ii: the class coordinate token (see `Delete::class_epoch`); `installs` ids are
        /// then packed coordinates of the OLD versions.
        class_epoch: Option<u64>,
    },
    /// Existing versions to tombstone, by tuple_id, in `table`'s partition. `deleted_rows` carries the
    /// resolved row images (catalog order) SV4b surfaces to the commit path so a single-entry DELETE can
    /// LOCATE + tombstone them on the authoritative resident GPU shard in place.
    Delete {
        table: String,
        tuple_ids: Vec<u64>,
        deleted_rows: Vec<Vec<SqlValue>>,
        /// P4-2b-ii: `Some(entry_epoch)` on a CHUNK-AUTHORITATIVE table — `tuple_ids` are then
        /// PACKED (chunk_idx << 32 | slot) coordinates from the chunk-native resolve, the apply
        /// skips the (frozen) store, and the commit hook stamps them iff the installed entry
        /// still carries this epoch (the P4-2a coordinate token).
        class_epoch: Option<u64>,
    },
}

/// The row-level mutation a single committed log entry applied, surfaced by `apply_mvcc_entry` so the
/// commit path can maintain GPU residency INCREMENTALLY for a single-entry commit (INSERT=append,
/// DELETE=tombstone, UPDATE=tombstone old + append new),
/// and record the applied [`WriteSet`] into the SI recent-commits ledger (C2, write-path assessment:
/// the ledger must see SERIALIZED-path writes too, or a concurrent committer validating against an
/// older snapshot silently misses them — a lost update). `None` for every other command.
#[derive(Debug, Clone)]
pub(crate) enum AppliedRowMutation {
    /// Typed non-MVCC table-root replacement. It participates in the same atomic transaction
    /// publication batch as post-reset row images but never masquerades as a row DELETE.
    TableReset {
        reset: BinaryTransactionTableReset,
        write_set: WriteSet,
    },
    Insert {
        table: String,
        rows: Vec<Vec<SqlValue>>,
        write_set: WriteSet,
        /// RETIREMENT A1: the installed rows' stable entity identities (parsed from the delta's
        /// `inserted_rows` keys — INSERT write-sets deliberately carry no row keys, so the delta is
        /// the identity source). Parallel to `rows`.
        row_ids: Vec<u64>,
    },
    Delete {
        table: String,
        rows: Vec<Vec<SqlValue>>,
        write_set: WriteSet,
        /// P4-2b-ii: the class stamp inputs — packed coordinates + the entry epoch (`None` on a
        /// non-class delete).
        class_stamp: Option<(Vec<u64>, u64)>,
    },
    Update {
        table: String,
        old_rows: Vec<Vec<SqlValue>>,
        new_rows: Vec<Vec<SqlValue>>,
        /// RETIREMENT A4b: the updated rows' IDENTITIES (parsed from the installs' row keys, EXACT
        /// parallel to `old_rows`/`new_rows` by construction) — the appended new versions keep
        /// them (A1). `None` for any unparseable key makes publication fail closed.
        row_ids: Option<Vec<u64>>,
        /// P4-2b-ii: the class stamp inputs — packed `(chunk_idx, slot)` coordinates plus the
        /// entry epoch. Stable logical identities remain in `row_ids`; never overload one as the
        /// other. The atomic-transaction maintainer uses epoch zero as its dynamic-resolve marker.
        class_stamp: Option<(Vec<u64>, u64)>,
        write_set: WriteSet,
    },
}

impl AppliedRowMutation {
    /// The `(table, row-key)` + unique-slot conflict footprint the apply installed — exactly the
    /// prepare-computed [`WriteSet`] of the underlying delta, for SI ledger recording.
    pub(crate) fn write_set(&self) -> &WriteSet {
        match self {
            Self::TableReset { write_set, .. }
            | Self::Insert { write_set, .. }
            | Self::Delete { write_set, .. }
            | Self::Update { write_set, .. } => write_set,
        }
    }

    /// Exact relational row count produced by replay/apply. This is compared with the canonical
    /// terminal marker before a recovered engine is returned to service.
    pub(crate) fn rows_affected(&self) -> u64 {
        match self {
            Self::TableReset { .. } => 0,
            Self::Insert { rows, .. } | Self::Delete { rows, .. } => rows.len() as u64,
            Self::Update { old_rows, .. } => old_rows.len() as u64,
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
    /// Exact publication visible when this mutation was prepared. READ COMMITTED may rebase the
    /// transaction overlay onto later publications, but doing so must never advance an earlier
    /// mutation's first-committer-wins validation floor.
    pub(crate) read_snapshot: Index,
    /// Exact catalog objects used to bind and validate this statement. READ COMMITTED may advance
    /// past unrelated DDL, but it must reject ALTER or same-name DROP/recreate of any object that
    /// an already-staged mutation depended on.
    pub(crate) catalog_dependencies: BTreeMap<String, RelationalTable>,
    /// Related parent/child relations read by FK preflight. Their mutation history is validated at
    /// this delta's exact `read_snapshot`, preventing a later RC statement from hiding an ABA race.
    pub(crate) foreign_key_dependencies: BTreeSet<String>,
    pub(crate) rows_consumed: u64,
    pub(crate) mutation: PreparedMutation,
}

/// One resolved table-root barrier retained in the transaction-private operation stream. The
/// durable before-root proof is constant-size: the last canonical table publication plus a
/// GPU-produced visible-row count. Transaction-private rows shadowed by the reset are deliberately
/// excluded from that globally replayable proof.
#[derive(Debug, Clone)]
pub(crate) struct StagedTableReset {
    pub(crate) ordinal: u32,
    /// Canonical identity of the fully typed/bound statement. This survives even when a later
    /// operation shadows every relational effect of this reset.
    pub(crate) statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) table: String,
    pub(crate) table_oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) source_commit_seq: Index,
    pub(crate) before_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) after_empty_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) expected_rows: u64,
    pub(crate) read_snapshot: Index,
    pub(crate) catalog_dependencies: BTreeMap<String, RelationalTable>,
    pub(crate) foreign_key_dependencies: BTreeSet<String>,
    pub(crate) dependency_identities: BTreeMap<String, u32>,
}

/// Bind exact-retry identity to the admitted typed command rather than to its final relational
/// effects. In particular, two zero-row predicates or two values later shadowed by TRUNCATE must
/// remain different requests even though their resolved mutation sets are identical.
pub(crate) fn transaction_statement_digest(
    command: &Command,
) -> Result<gpu_db_wal::CanonicalDigest, ExecuteError> {
    let encoded = serde_json::to_vec(command).map_err(|error| {
        ExecuteError::Engine(EngineError::Durability(format!(
            "typed transaction statement cannot be encoded canonically: {error}"
        )))
    })?;
    let encoded_len = u64::try_from(encoded.len()).map_err(|_| {
        ExecuteError::Unsupported(
            "typed transaction statement exceeds canonical digest framing".to_string(),
        )
    })?;
    let mut body = Vec::with_capacity(40 + encoded.len());
    body.extend_from_slice(b"GPUDBTXNSTATEMENT1");
    body.extend_from_slice(&encoded_len.to_le_bytes());
    body.extend_from_slice(&encoded);
    Ok(gpu_db_wal::canonical_request_digest(&body))
}

/// One staged DML statement plus its request identity. Keeping this wrapper transaction-local
/// avoids adding cold exact-retry metadata to the latency-sensitive ordinary `WriteDelta` path.
#[derive(Debug, Clone)]
pub(crate) struct StagedRowOperation {
    pub(crate) statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) delta: WriteDelta,
}

impl std::ops::Deref for StagedRowOperation {
    type Target = WriteDelta;

    fn deref(&self) -> &Self::Target {
        &self.delta
    }
}

impl std::ops::DerefMut for StagedRowOperation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.delta
    }
}

/// The statement-ordered private transaction stream. Catalog operations, resets, and row
/// operations share one ordinal authority. Keeping resets beside row operations makes shadowing
/// structural: replay starts from the retained base, applies a reset as an empty-root replacement,
/// then applies only the row operations that follow it.
#[derive(Debug, Clone)]
pub(crate) enum TransactionOperation {
    Row(Arc<StagedRowOperation>),
    TableReset(Arc<StagedTableReset>),
    // Keep new variants append-only: established discriminants feed large engine match tables,
    // and changing them has previously moved latency-sensitive linked code measurably.
    Catalog(Arc<StagedCatalogCommand>),
}

impl WriteDelta {
    /// U1 rows-affected surface: the exact row count this prepared mutation touches — the value
    /// a committed item's `Ok(_)` outcome reports (INSERT = rows installed, UPDATE = versions
    /// rewritten, DELETE = versions tombstoned).
    pub(crate) fn rows_affected(&self) -> u64 {
        match &self.mutation {
            PreparedMutation::Insert { inserted_rows, .. } => inserted_rows.len() as u64,
            PreparedMutation::Update { installs, .. } => installs.len() as u64,
            PreparedMutation::Delete { tuple_ids, .. } => tuple_ids.len() as u64,
        }
    }
}

/// The recent row-identity ledger: every committed `(table, row-key)` carries the highest
/// `commit_seq` that wrote it. Unique-slot history is device-authoritative in production; the two
/// unique maps below exist only in `cfg(test)` as a host-neutral parity ledger. Row entries
/// are pruned below the oldest active read snapshot, which is also the safe MVCC GC boundary.
#[derive(Debug, Default)]
pub(crate) struct RecentCommitsLedger {
    pub(crate) rows: BTreeMap<RowWriteKey, Index>,
    pub(crate) tables: BTreeMap<String, Index>,
    #[cfg(test)]
    pub(crate) unique_slots: BTreeMap<UniqueIndexSlotKey, Index>,
    /// Driverless parity twin for the allocation-free i32 slot projection.
    #[cfg(test)]
    pub(crate) unique_slots_i32: std::collections::HashMap<IntUniqueSlotKey, Index>,
}

impl RecentCommitsLedger {
    /// First-committer-wins SI validation: does any key in `write_set` carry a recorded commit
    /// strictly newer than `read_snapshot`? If so the preparing txn read a now-stale snapshot of
    /// that key and must abort (retryable). Equality on `read_snapshot` does NOT conflict — that is
    /// a commit this txn's snapshot already saw.
    #[cfg(test)]
    pub(crate) fn conflicts(&self, write_set: &WriteSet, read_snapshot: Index) -> bool {
        self.conflicts_rows(write_set, read_snapshot)
            || self.conflicts_unique(write_set, read_snapshot)
    }

    pub(crate) fn conflicts_rows(&self, write_set: &WriteSet, read_snapshot: Index) -> bool {
        write_set
            .rows
            .iter()
            .any(|key| self.rows.get(key).is_some_and(|&seq| seq > read_snapshot))
    }

    pub(crate) fn table_changed_after(&self, table: &str, read_snapshot: Index) -> bool {
        self.tables
            .get(table)
            .is_some_and(|&commit_seq| commit_seq > read_snapshot)
    }

    /// Stable logical identity of the currently-published table root. Catalog publication rekeys
    /// this high-water across rename and removes dropped identities, keeping it bounded by current
    /// catalog cardinality; typed reset WAL uses it as its replay-stable source-root descriptor.
    pub(crate) fn table_root_index(&self, table: &str) -> Index {
        self.tables.get(table).copied().unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn conflicts_unique(&self, write_set: &WriteSet, read_snapshot: Index) -> bool {
        write_set.unique_slots.iter().any(|key| {
            self.unique_slots
                .get(key)
                .is_some_and(|&seq| seq > read_snapshot)
        }) || write_set.unique_slots_i32.iter().any(|key| {
            self.unique_slots_i32
                .get(key)
                .is_some_and(|&seq| seq > read_snapshot)
        })
    }

    /// Record committed row identities. Test builds additionally maintain the driverless unique
    /// parity maps; production unique conflict history is read from resident version stamps.
    pub(crate) fn record(&mut self, write_set: &WriteSet, commit_seq: Index) {
        for table in &write_set.tables {
            self.tables.insert(table.clone(), commit_seq);
        }
        for key in &write_set.rows {
            self.rows.insert(key.clone(), commit_seq);
        }
        #[cfg(test)]
        {
            for key in &write_set.unique_slots {
                self.unique_slots.insert(key.clone(), commit_seq);
            }
            for &key in &write_set.unique_slots_i32 {
                self.unique_slots_i32.insert(key, commit_seq);
            }
        }
    }

    pub(crate) fn reconcile_table_roots(
        &mut self,
        prior_identities: &BTreeMap<String, u32>,
        current_identities: &BTreeMap<String, u32>,
    ) {
        let current_names_by_oid = current_identities
            .iter()
            .map(|(name, oid)| (*oid, name.as_str()))
            .collect::<BTreeMap<_, _>>();
        let mut next = BTreeMap::new();
        for (name, commit_seq) in std::mem::take(&mut self.tables) {
            let destination = match prior_identities.get(&name) {
                Some(oid) => current_names_by_oid.get(oid).copied(),
                None => current_identities
                    .contains_key(&name)
                    .then_some(name.as_str()),
            };
            if let Some(destination) = destination {
                next.entry(destination.to_string())
                    .and_modify(|prior: &mut Index| *prior = (*prior).max(commit_seq))
                    .or_insert(commit_seq);
            }
        }
        self.tables = next;
    }

    /// Drop entries written at or before `boundary` (no active snapshot reads before it, so they can
    /// never win a future conflict). Keeps the ledger bounded by the active-snapshot window.
    pub(crate) fn prune_below(&mut self, boundary: Index) {
        // `tables` is the durable logical root oracle for typed non-MVCC rewrites. Catalog
        // publication bounds it by live stable identities; temporal pruning here would erase the
        // source-root high-water of a still-live relation.
        self.rows.retain(|_, &mut seq| seq > boundary);
        #[cfg(test)]
        {
            self.unique_slots.retain(|_, &mut seq| seq > boundary);
            self.unique_slots_i32.retain(|_, &mut seq| seq > boundary);
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.tables.len() + self.rows.len() + self.unique_slots.len() + self.unique_slots_i32.len()
    }
}

/// One transaction or autocommit statement's generation-owned read world. `boundary` is the visibility stamp, but
/// correctness does not rely on that scalar alone: the exact catalog, per-table MVCC generations,
/// single-buffer entries, shard descriptors, and already-built device index allocations are retained
/// together until terminal transaction control. Later commits may publish replacement descriptors;
/// these owned generations remain alive and cannot be paired with newer state.
#[derive(Debug)]
pub(crate) struct TransactionSnapshot {
    pub(crate) characteristics: TransactionCharacteristics,
    pub(crate) boundary: Index,
    /// Row-identity allocator boundary captured with the transaction generation. INSERT prepare
    /// must not derive provisional identities from a newer allocator observation; the private
    /// transaction delta will apply its own deterministic offset before COMMIT assigns final slots.
    pub(crate) next_row_id: u64,
    pub(crate) catalog: Arc<CatalogSnapshot>,
    pub(crate) table_versions: BTreeMap<String, SnapshotHandle<Arc<TableVersionData>>>,
    pub(crate) resident_snapshots: Arc<BTreeMap<String, RelationalResidencyEntry>>,
    pub(crate) resident_shards: Arc<BTreeMap<String, Vec<RelationalResidentShard>>>,
    /// Representation authority is part of the generation, not current global policy. An elided
    /// table's captured host generation is intentionally incomplete; a class table's captured cold
    /// entry is its post-freeze authority. DML prepare must retain and consult these exact maps.
    pub(crate) device_authoritative_tables: Arc<BTreeSet<String>>,
    pub(crate) chunk_authoritative_tables: Arc<BTreeMap<String, Index>>,
    /// Transaction-private, device-resident write generation. Every successful DML statement
    /// replaces this state atomically with a new immutable shard map: retained base shards carry
    /// private tombstone sidecars and INSERT/UPDATE post-images live in dense private shards.
    /// SELECT and later DML clone the published map, so no reader can observe half a statement.
    pub(crate) delta: Arc<std::sync::Mutex<TransactionDeltaState>>,
    /// Shared/exclusive stable-OID guards retained for this statement or explicit transaction.
    pub(crate) table_access: Arc<TableAccessLease>,
    /// Stable table names whose current rewrite fence is newer than this retained boundary.
    /// Membership grows within one retained statement/snapshot; READ COMMITTED replaces the set
    /// when it rebases to a publication that already contains those prior fences.
    pub(crate) rewrite_fenced_tables: Arc<Mutex<BTreeSet<String>>>,
    /// One explicit transaction is a sequential statement stream. Holding this guard for the full
    /// SELECT/DML/terminal-control operation prevents a DML replacement from retiring the exact
    /// GPU charge of a superseded private generation while a same-transaction reader still pins
    /// that generation's device allocations.
    pub(crate) statement_lock: Arc<std::sync::Mutex<()>>,
    /// Set before a predeclared program's snapshot becomes discoverable. Ordinary same-id entries
    /// reject this token; only the program's private statement-locked helpers may proceed.
    pub(crate) program_owned: Arc<std::sync::atomic::AtomicBool>,
    /// False until the first data/catalog statement chooses the transaction snapshot. PostgreSQL
    /// REPEATABLE READ is lazy at BEGIN; READ COMMITTED replaces the captured base every statement.
    pub(crate) data_snapshot_acquired: Arc<std::sync::atomic::AtomicBool>,
    /// The globally published cold generation captured with the base. The private delta owns a
    /// derived replacement; READ COMMITTED rebase restarts from this exact immutable map.
    pub(crate) base_streaming_cold_chunks:
        Arc<BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>>,
    /// Engine-wide charge table shared with every transaction snapshot. Sequential statement
    /// ownership makes the delta state's current-generation charge exact; `Drop` releases that
    /// final charge when the transaction snapshot's last retained handle disappears.
    pub(crate) private_gpu_account: Arc<std::sync::Mutex<BTreeMap<u16, u64>>>,
    /// Lifetime pins for device index allocations that existed at capture. Execution validates an
    /// index against the retained data-buffer identity before use and may rebuild lazily when absent;
    /// this vector prevents a captured valid index allocation from being reclaimed underneath the
    /// transaction merely because the current-generation cache was purged.
    pub(crate) _resident_index_resources: Vec<Arc<CudaResidentDeviceMemory>>,
    /// Accounting twin for every payload, sidecar, and index allocation retained by this snapshot.
    /// The shared registry is keyed by exact device allocation identity, so a current allocation is
    /// charged once and remains charged after same-table replacement/cache purge until the final
    /// transaction generation releases its guard.
    pub(crate) _resident_gpu_charge: Arc<TransactionRetainedGpuCharge>,
}

pub(crate) type TransactionRetainedGpuAccount = BTreeMap<(u16, u64), (u64, usize)>;

#[derive(Debug)]
pub(crate) struct TransactionRetainedGpuCharge {
    account: Arc<std::sync::Mutex<TransactionRetainedGpuAccount>>,
    allocations: Vec<(u16, u64)>,
}

impl TransactionRetainedGpuCharge {
    pub(crate) fn empty(account: Arc<std::sync::Mutex<TransactionRetainedGpuAccount>>) -> Self {
        Self {
            account,
            allocations: Vec::new(),
        }
    }

    pub(crate) fn register_locked(
        account: Arc<std::sync::Mutex<TransactionRetainedGpuAccount>>,
        resources: &[Arc<CudaResidentDeviceMemory>],
        tracked: &mut TransactionRetainedGpuAccount,
    ) -> Self {
        let allocations = resources
            .iter()
            .map(|memory| {
                (
                    (memory.metadata().gpu_id, memory.device_ptr()),
                    memory.metadata().allocated_bytes,
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (identity, bytes) in &allocations {
            let entry = tracked.entry(*identity).or_insert((*bytes, 0));
            debug_assert_eq!(entry.0, *bytes);
            entry.1 = entry.1.saturating_add(1);
        }
        Self {
            account,
            allocations: allocations.into_keys().collect(),
        }
    }
}

impl Drop for TransactionRetainedGpuCharge {
    fn drop(&mut self) {
        let mut tracked = self
            .account
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for identity in &self.allocations {
            let remove = if let Some((_, owners)) = tracked.get_mut(identity) {
                *owners = owners.saturating_sub(1);
                *owners == 0
            } else {
                false
            };
            if remove {
                tracked.remove(identity);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct TransactionDeltaState {
    pub(crate) generation: u64,
    pub(crate) resident_shards: Arc<BTreeMap<String, Vec<RelationalResidentShard>>>,
    /// Transaction-private cold authority. Chunk-class INSERTs append device-format tail chunks
    /// here with a birth boundary visible to this transaction, never to the globally published
    /// cold map. SELECT and later DML therefore consume the same immutable private generation.
    pub(crate) streaming_cold_chunks:
        Arc<BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>>,
    pub(crate) operations: Vec<TransactionOperation>,
    pub(crate) write_set: WriteSet,
    pub(crate) next_row_id: u64,
    /// Transaction-local post-state for every sequence consumed by a column default. A later
    /// statement seeds its pure `nextval` scratch here, so values advance across private statements
    /// without mutating the published catalog before COMMIT.
    pub(crate) sequence_state: BTreeMap<String, (i64, bool)>,
    /// Exact published catalog generation beneath every ordered private catalog operation.
    /// Catalog operations themselves live in `operations`, so mixed DDL/DML/reset programs retain
    /// one statement-order authority rather than maintaining a parallel command stream.
    pub(crate) catalog_base: Option<Arc<CatalogSnapshot>>,
    pub(crate) catalog_overlay: Option<Arc<CatalogSnapshot>>,
    pub(crate) private_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
    /// Capacity held for mandatory named indexes that canonical apply must build when a
    /// transaction-created table first becomes globally resident. Unlike `private_gpu_bytes`,
    /// these bytes have no private allocation yet; they close deterministic post-WAL budget
    /// failure without publishing transaction-private indexes globally.
    pub(crate) commit_gpu_bytes_by_gpu: BTreeMap<u16, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StagedCatalogCommand {
    pub(crate) ordinal: u32,
    pub(crate) statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) command: Command,
    pub(crate) view_identity: Option<BinaryTransactionViewOperationIdentity>,
}

impl Drop for TransactionSnapshot {
    fn drop(&mut self) {
        if Arc::strong_count(&self.delta) != 1 {
            return;
        }
        let charged = self
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .private_gpu_bytes_by_gpu
            .clone();
        let commit_charged = self
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .commit_gpu_bytes_by_gpu
            .clone();
        let mut account = self
            .private_gpu_account
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (gpu_id, bytes) in charged.into_iter().chain(commit_charged) {
            let slot = account.entry(gpu_id).or_default();
            *slot = slot.saturating_sub(bytes);
            if *slot == 0 {
                account.remove(&gpu_id);
            }
        }
    }
}

impl TransactionSnapshot {
    pub(crate) fn transaction_shards(&self) -> Arc<BTreeMap<String, Vec<RelationalResidentShard>>> {
        Arc::clone(
            &self
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .resident_shards,
        )
    }

    pub(crate) fn transaction_cold_chunks(
        &self,
    ) -> Arc<BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>> {
        Arc::clone(
            &self
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .streaming_cold_chunks,
        )
    }

    pub(crate) fn transaction_delta_is_empty(&self) -> bool {
        let delta = self
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        delta.operations.is_empty()
    }

    pub(crate) fn transaction_table_is_reset(&self, table: &str) -> bool {
        self.delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .operations
            .iter()
            .rev()
            .any(|operation| {
                matches!(operation, TransactionOperation::TableReset(reset) if reset.table == table)
            })
    }

    pub(crate) fn table_is_rewrite_fenced(&self, table: &str) -> bool {
        self.rewrite_fenced_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(table)
    }

    pub(crate) fn table_has_typed_empty_root(&self, table: &str) -> bool {
        let rewrite_fenced = self.table_is_rewrite_fenced(table);
        let delta = self
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let reset = delta
            .operations
            .iter()
            .rev()
            .any(|operation| {
                matches!(operation, TransactionOperation::TableReset(reset) if reset.table == table)
            });
        (rewrite_fenced || reset)
            && delta.resident_shards.get(table).is_some_and(|shards| {
                !shards.is_empty()
                    && shards
                        .iter()
                        .all(|shard| shard.row_count == 0 && shard.device_memory.is_some())
            })
            && delta
                .streaming_cold_chunks
                .get(table)
                .is_none_or(|chunks| chunks.chunks.is_empty())
    }

    pub(crate) fn transaction_catalog(&self) -> Arc<CatalogSnapshot> {
        self.delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .catalog_overlay
            .clone()
            .unwrap_or_else(|| Arc::clone(&self.catalog))
    }
}

/// Tracks the read snapshots of in-flight statements and explicit transactions (write-half MVCC,
/// Stage 4) as an ordered multiset of `read_snapshot` `commit_seq` values. An autocommit statement
/// registers for its prepare/commit window; an explicit transaction registers once at `BEGIN` and
/// keeps that reference until `COMMIT`/`ROLLBACK`. The MINIMUM registered snapshot is the
/// oldest-active boundary: ledger entries and MVCC versions older than it can be reclaimed because no
/// active transaction can still observe (or conflict against) them. This is the concrete
/// oldest-active-`commit_seq` the Stage-0 GC-boundary debt needed.
#[derive(Debug, Default)]
pub(crate) struct ActiveSnapshots {
    pub(crate) counts: BTreeMap<Index, usize>,
    /// Explicit transaction identity -> its one generation-owned lifetime snapshot. Keeping this
    /// map beside the ordered multiset makes duplicate registration/removal exact while `counts`
    /// continues to fold statement-local and transaction-held references into one GC boundary.
    pub(crate) transactions: BTreeMap<TxnId, Arc<TransactionSnapshot>>,
    /// GC/ledger-retention floor owned by each explicit transaction. It is moved to the lazy first
    /// statement snapshot, then retained at the minimum statement boundary for the transaction's
    /// remaining lifetime even when READ COMMITTED installs a newer statement base.
    pub(crate) transaction_floors: BTreeMap<TxnId, Index>,
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

    pub(crate) fn register_transaction(
        &mut self,
        txn_id: TxnId,
        snapshot: Arc<TransactionSnapshot>,
    ) {
        let boundary = snapshot.boundary;
        let previous = self.transactions.insert(txn_id, snapshot);
        let previous_floor = self.transaction_floors.insert(txn_id, boundary);
        debug_assert!(
            previous.is_none(),
            "transaction {txn_id} registered more than one active snapshot"
        );
        debug_assert!(previous_floor.is_none());
        if let Some(previous) = previous {
            self.deregister(previous_floor.unwrap_or(previous.boundary));
        }
        self.register(boundary);
    }

    pub(crate) fn deregister_transaction(
        &mut self,
        txn_id: TxnId,
    ) -> Option<Arc<TransactionSnapshot>> {
        let snapshot = self.transactions.remove(&txn_id)?;
        let floor = self
            .transaction_floors
            .remove(&txn_id)
            .unwrap_or(snapshot.boundary);
        self.deregister(floor);
        Some(snapshot)
    }

    #[cfg(test)]
    pub(crate) fn transaction_snapshot(&self, txn_id: TxnId) -> Option<Index> {
        self.transactions
            .get(&txn_id)
            .map(|snapshot| snapshot.boundary)
    }

    pub(crate) fn transaction_snapshot_handle(
        &self,
        txn_id: TxnId,
    ) -> Option<Arc<TransactionSnapshot>> {
        self.transactions.get(&txn_id).cloned()
    }

    pub(crate) fn replace_transaction_snapshot(
        &mut self,
        txn_id: TxnId,
        expected: &Arc<TransactionSnapshot>,
        replacement: Arc<TransactionSnapshot>,
        advance_lazy_floor: bool,
    ) -> Result<(), ExecuteError> {
        let current = self
            .transactions
            .get(&txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotActive(txn_id)))?;
        if !Arc::ptr_eq(current, expected) {
            return Err(ExecuteError::Txn(TxnError::NotActive(txn_id)));
        }
        let old_floor = self
            .transaction_floors
            .get(&txn_id)
            .copied()
            .unwrap_or(current.boundary);
        let new_floor = if advance_lazy_floor {
            replacement.boundary
        } else {
            old_floor.min(replacement.boundary)
        };
        self.transactions.insert(txn_id, replacement);
        self.transaction_floors.insert(txn_id, new_floor);
        if old_floor != new_floor {
            self.deregister(old_floor);
            self.register(new_floor);
        }
        Ok(())
    }

    /// The oldest active read snapshot, or `None` when no transaction is in flight.
    pub(crate) fn oldest(&self) -> Option<Index> {
        self.counts.keys().next().copied()
    }
}

/// RAII guard that deregisters an autocommit statement's read snapshot from [`ActiveSnapshots`] on
/// drop (write-half MVCC, Stage 4), so a snapshot is released even if prepare/commit returns early (a
/// serialization abort, a constraint error). Explicit transactions use the registry's keyed
/// lifetime entry instead, released by transaction control.
pub(crate) struct ActiveSnapshotGuard<'a> {
    pub(crate) engine: &'a Engine,
    pub(crate) snapshot: Index,
}

impl Drop for ActiveSnapshotGuard<'_> {
    fn drop(&mut self) {
        self.engine.deregister_active_snapshot(self.snapshot);
    }
}
