//! E2.1 — the covered-INSERT INTENT fast path (GPU-native write dataplane).
//!
//! The classic concurrent-DML path pays SQL text (`parse_command`) plus the pure
//! off-lock `prepare_insert` (catalog bind, per-value coercion, constraint
//! dispatch) for EVERY statement, even for the flagship OLTP shape: a prepared,
//! covered, single-row INSERT into an elided (device-authoritative) PK'd int4
//! table. For that shape everything the prepare computes is a pure function of
//! (route, params):
//!
//! - values are `Int4(param)` (identity coercion; never NULL, so the PK
//!   not-null gate passes by construction);
//! - the unique check is wave-DEFERRED (`insert_unique_wave_batchable`, one
//!   batched device locate per wave — M1 design B);
//! - the value-index map is EMPTY (elided tables skip the host value index);
//! - the write-set is exactly the unique slots of the row's key columns.
//!
//! So a prepared ROUTE ([`CoveredInsertRoute`]) captures the shape checks once,
//! and each intent execution builds the wave item DIRECTLY — no parse, no
//! `prepare_insert` — and rides the UNMODIFIED commit-wave machinery: the
//! sequencer's batched device PK validation, the delta-reuse re-key, the W5a
//! binary WAL record (whole wave coalesced into one FUA frame by the group
//! flush), the wave-batched device open-shard append (one HtoD + created_by /
//! row-id stamps + device PK-index insert per flush), and the pipelined
//! durability tail (ack = durable cut covering the wave + publish).
//!
//! SAFETY ENVELOPE: eligibility is re-checked at EXECUTE time against the live
//! catalog generation. Any drift (DDL, de-elision, table dropped) falls back to
//! the classic `execute_dml_concurrent` path over the synthesized SQL text —
//! the intent path never introduces a validation skip the classic deferral
//! doesn't already have, and the synthesized text keeps the WAL replayable on
//! every fallback arm (the sequencer's binary-encode fallback writes
//! `item.payload` verbatim).

use super::*;

/// A prepared covered-INSERT route: the immutable per-statement shape checks,
/// captured once so per-intent execution is allocation-lean and validation is
/// O(1). Obtain via [`Engine::prepare_covered_insert_route`]; invalidated by
/// any DDL (execution falls back to the classic path and the caller should
/// re-prepare).
#[derive(Debug, Clone)]
pub struct CoveredInsertRoute {
    table: String,
    /// Shared table name for lean lane intents (cheap Arc clone per submit).
    table_arc: std::sync::Arc<str>,
    /// Catalog generation at prepare. Execute compares the LIVE generation:
    /// a mismatch means a DDL committed since — the shape proof is stale.
    catalog_seq: Index,
    column_count: usize,
    /// `"INSERT INTO <table> VALUES ("` — the synthesized-SQL prefix for the
    /// WAL-fallback payload and the classic-path fallback.
    sql_prefix: String,
    /// E2.2(a) — the INTEGER conflict slots this route's inserts claim, precomputed once:
    /// `(packed_slot_id, column_index)` for every unique i32 index (all of them, since the
    /// route requires every column INT4). Execute reads `params[column_index]` to build the
    /// allocation-free [`crate::write_path::IntUniqueSlotKey`]s — no per-item String format,
    /// no `(table, column)` clone. The stable `(table_oid, column_id)` identity is shared with
    /// the classic path (see [`crate::write_path::add_unique_slots`]), so cross-path SI conflicts
    /// against the same slot are exact.
    unique_i32_slots: Vec<(u64, usize)>,
    /// E2.2(b) — the fixed byte offset of the single row's `u64` row id inside the pre-encoded
    /// W5a binary WAL record for this route's table: `3 (tag/ver/op) + 2 (table_len) +
    /// table_len + 4 (row_count)`. The sequencer patches 8 bytes here with the wave-assigned id.
    binary_row_id_offset: u32,
}

impl CoveredInsertRoute {
    pub fn table(&self) -> &str {
        &self.table
    }

    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// Synthesize the canonical SQL text for `params` (the classic-path /
    /// WAL-record fallback payload; parse round-trips to the same `Insert`).
    fn synthesize_text(&self, params: &[i32]) -> String {
        use std::fmt::Write as _;
        // prefix + per-param worst case "-2147483648, " + ")"
        let mut text = String::with_capacity(self.sql_prefix.len() + params.len() * 13 + 1);
        text.push_str(&self.sql_prefix);
        for (position, param) in params.iter().enumerate() {
            if position > 0 {
                text.push_str(", ");
            }
            let _ = write!(text, "{param}");
        }
        text.push(')');
        text
    }
}

impl Engine {
    /// Prepare a covered-INSERT route for `table`: the E2.1 intent fast path's
    /// per-statement handle. Errors unless the table is CURRENTLY the covered
    /// flagship shape:
    ///
    /// - every column strictly `INT4` with no column defaults (an intent
    ///   provides every column positionally);
    /// - wave-batchable unique validation (`insert_unique_wave_batchable`):
    ///   table ELIDED (device-authoritative), FK/CHECK-free, no inbound FK,
    ///   at least one unique index and every unique index on an i32 column,
    ///   with the device write-locate + wave-batch flags enabled;
    /// - binary WAL records enabled (covered inserts log W5a row-op records).
    ///
    /// Elision is entered lazily (first successful wave-batched device append
    /// with elision enabled), so warm the table with a few classic inserts
    /// before preparing the route.
    pub fn prepare_covered_insert_route(
        &self,
        table_name: &str,
    ) -> Result<CoveredInsertRoute, ExecuteError> {
        let catalog = self.catalog_snapshot();
        let table = catalog.relational_catalog.get(table_name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{table_name}\" does not exist"
            )))
        })?;
        let route_err = |reason: &str| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "table \"{table_name}\" is not covered-INSERT routable: {reason}"
            )))
        };
        if !table
            .columns
            .iter()
            .all(|column| matches!(column.ty, SqlType::Int4))
        {
            return Err(route_err("every column must be INT4"));
        }
        if table.columns.iter().any(|column| column.default.is_some()) {
            return Err(route_err(
                "column defaults are not intent-coverable (an intent provides every column)",
            ));
        }
        if !self.binary_wal_records_enabled() {
            return Err(route_err("binary WAL records are disabled"));
        }
        if !self.insert_unique_wave_batchable(&catalog, table) {
            return Err(route_err(
                "wave-batched device PK validation is unavailable (table not elided yet, \
                 constraints present, or device write-locate flags disabled)",
            ));
        }
        // E2.2(a): precompute the integer conflict slots — one per unique index, keyed by the
        // stable (table_oid, column_id) identity + the row's i32 value at wave time. Every column
        // is INT4 here, so every unique index qualifies for the allocation-free integer slot.
        let unique_i32_slots: Vec<(u64, usize)> = table
            .indexes
            .iter()
            .filter(|index| index.unique)
            .filter_map(|index| {
                table
                    .columns
                    .iter()
                    .position(|column| column.name == index.column)
                    .map(|column_idx| {
                        let column = &table.columns[column_idx];
                        (
                            crate::write_path::pack_unique_slot_id(table.oid, column.id),
                            column_idx,
                        )
                    })
            })
            .collect();
        // E2.2(b): the W5a single-row record lays out the row id at a fixed offset after the
        // header (tag/ver/op) + table-len prefix + row-count. Encoding uses the bare `table.name`
        // (the delta mutation's table string), matching the sequencer's binary-record input.
        let binary_row_id_offset = (3 + 2 + table.name.len() + 4) as u32;
        Ok(CoveredInsertRoute {
            table: table.name.clone(),
            catalog_seq: catalog.commit_seq,
            column_count: table.columns.len(),
            sql_prefix: format!("INSERT INTO {} VALUES (", table.name),
            table_arc: std::sync::Arc::from(table_name),
            unique_i32_slots,
            binary_row_id_offset,
        })
    }

    /// Execute one covered-INSERT intent: `params` are the row's INT4 values in
    /// catalog column order. Semantics are identical to executing the
    /// synthesized `INSERT INTO t VALUES (...)` through
    /// [`Engine::execute_dml_concurrent`] — duplicate keys raise the same
    /// 23505 `ApplyFailed`, SI conflicts the same retryable `Serialization` —
    /// but the hot path skips SQL parse and `prepare_insert` entirely and
    /// enqueues the wave item directly. Returns once the intent's wave is
    /// DURABLE (WAL frame under the durable cut) and published.
    ///
    /// On any eligibility drift (DDL since the route's prepare, de-elision,
    /// column-count mismatch against a re-created table) this transparently
    /// falls back to the classic text path, which re-validates everything.
    pub fn execute_covered_insert_intent(
        &self,
        txn_id: u64,
        route: &CoveredInsertRoute,
        params: &[i32],
    ) -> Result<(), ExecuteError> {
        self.check_intent_params(route, params)?;
        // Pin + register the read snapshot exactly like the classic off-lock
        // prepare (the guard holds the MVCC GC boundary; conflicts() validates
        // against this snapshot).
        let read_snapshot = self.committed_seq();
        let snapshot_guard = self.register_active_snapshot(read_snapshot);
        match self.build_covered_insert_intent(txn_id, route, params, read_snapshot) {
            IntentBuild::Item(item) => self.commit_wave_item_blocking(item),
            IntentBuild::Fallback(text) => {
                drop(snapshot_guard);
                self.execute_dml_concurrent(txn_id, &text)
            }
        }
    }

    /// E2.2(c) — SUBMIT a covered-INSERT intent WITHOUT blocking, returning an [`IntentTicket`].
    /// A driver thread advances the pipeline via [`Engine::drive_commit_wave`] and reaps the
    /// ticket with [`Engine::poll_intent`]; N drivers thereby carry M logical clients with no
    /// per-commit thread park/wake. The read snapshot stays registered (GC/prune boundary) until
    /// the ticket is polled to completion or dropped (the ticket owns the release: Drop is the
    /// audit-F2 safety net, so an abandoned ticket cannot pin the GC/prune boundary forever).
    ///
    /// On eligibility drift the classic text path is run INLINE (rare) and the ticket returns its
    /// resolved outcome on the first poll — the async surface never silently skips a validation.
    /// E2.5b-2 lean build: everything the lane pump needs, nothing more — no
    /// SQL text, no row-key String, no delta/AST clones, no residency set.
    /// `None` = eligibility drift (caller falls back to the classic path).
    fn build_lane_intent(
        &self,
        txn_id: u64,
        route: &CoveredInsertRoute,
        params: &[i32],
        read_snapshot: Index,
    ) -> Option<crate::engine_dml_concurrent::LaneIntent> {
        let catalog = self.catalog_snapshot();
        let prepared_catalog_seq = catalog.commit_seq;
        let table = catalog.relational_catalog.get(&route.table)?;
        let eligible = prepared_catalog_seq == route.catalog_seq
            && self.binary_wal_records_enabled()
            && route.unique_i32_slots.len() == 1
            && table.columns.len() == route.column_count
            && self.insert_unique_wave_batchable(&catalog, table);
        if !eligible {
            return None;
        }
        let values: Vec<SqlValue> = params.iter().map(|&param| SqlValue::Int4(param)).collect();
        let (slot_id, column_idx) = route.unique_i32_slots[0];
        let (template, row_id_offset) =
            crate::wal_binary::try_encode_binary_insert(&route.table, &[(0u64, values.as_slice())])
                .map(|bytes| {
                    (
                        std::sync::Arc::<[u8]>::from(bytes.as_slice()),
                        route.binary_row_id_offset,
                    )
                })?;
        Some(crate::engine_dml_concurrent::LaneIntent {
            txn_id,
            slot: (slot_id, params[column_idx]),
            read_snapshot,
            prepared_catalog_seq,
            filter_idx: column_idx as u32,
            row_id_offset,
            table: std::sync::Arc::clone(&route.table_arc),
            template,
            values,
            outcome: crate::engine_dml_concurrent::new_pending_outcome(),
            outstanding: None,
        })
    }

    pub fn submit_covered_insert_intent(
        &self,
        txn_id: u64,
        route: &CoveredInsertRoute,
        params: &[i32],
    ) -> Result<IntentTicket, ExecuteError> {
        self.check_intent_params(route, params)?;
        // LEAN LANE PATH: in lanes mode, build the compact LaneIntent and push
        // straight to its PK lane — CommitWaveItem is never constructed here.
        if let Some(lanes) = &self.intent_lanes {
            let read_snapshot = self.committed_seq();
            std::mem::forget(self.register_active_snapshot(read_snapshot));
            let snapshot_hold =
                Some((std::sync::Arc::clone(&self.active_snapshots), read_snapshot));
            if let Some(intent) = self.build_lane_intent(txn_id, route, params, read_snapshot) {
                let outcome = std::sync::Arc::clone(&intent.outcome);
                // ACTIVE-LANE RESIZE barrier: while a resize leader drains the
                // in-flight population, new intents divert to the hold queue
                // (not counted as outstanding) and re-route after the flip.
                if lanes
                    .resize_holding
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    lanes
                        .resize_hold
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(intent);
                } else {
                    crate::Engine::enqueue_lane_intent(lanes, intent);
                }
                return Ok(IntentTicket {
                    outcome: Some(outcome),
                    snapshot_hold,
                    resolved: None,
                });
            }
            // eligibility drift: classic inline fallback (rare)
            let resolved = self.execute_dml_concurrent(txn_id, &route.synthesize_text(params));
            self.deregister_active_snapshot(read_snapshot);
            return Ok(IntentTicket {
                outcome: None,
                snapshot_hold: None,
                resolved: Some(resolved),
            });
        }
        let read_snapshot = self.committed_seq();
        // Register WITHOUT the RAII guard — the ticket takes an OWNED hold on the registry, so
        // the boundary is released by the completing poll or the ticket's Drop (audit F2), never
        // leaked by a dropped-unpolled ticket.
        std::mem::forget(self.register_active_snapshot(read_snapshot));
        let snapshot_hold = Some((std::sync::Arc::clone(&self.active_snapshots), read_snapshot));
        match self.build_covered_insert_intent(txn_id, route, params, read_snapshot) {
            IntentBuild::Item(item) => {
                let outcome = match self.submit_commit_wave_item(item) {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        // Enqueue refused (wedged): release the boundary and surface the error.
                        self.deregister_active_snapshot(read_snapshot);
                        return Err(err);
                    }
                };
                Ok(IntentTicket {
                    outcome: Some(outcome),
                    snapshot_hold,
                    resolved: None,
                })
            }
            IntentBuild::Fallback(text) => {
                // Rare drift: run the classic path inline, release the boundary, return a
                // pre-resolved ticket (poll yields the result once).
                let resolved = self.execute_dml_concurrent(txn_id, &text);
                self.deregister_active_snapshot(read_snapshot);
                Ok(IntentTicket {
                    outcome: None,
                    snapshot_hold: None,
                    resolved: Some(resolved),
                })
            }
        }
    }

    /// E2.2(c) — poll a submitted intent. Returns `None` while in flight, `Some(result)` exactly
    /// once on completion (releasing the read-snapshot GC boundary), and `None` thereafter.
    pub fn poll_intent(&self, ticket: &mut IntentTicket) -> Option<Result<(), ExecuteError>> {
        if let Some(resolved) = ticket.resolved.take() {
            return Some(resolved);
        }
        let outcome = ticket.outcome.as_ref()?;
        let result = outcome.take_if_done()?;
        ticket.outcome = None;
        ticket.release_snapshot();
        Some(result)
    }

    fn check_intent_params(
        &self,
        route: &CoveredInsertRoute,
        params: &[i32],
    ) -> Result<(), ExecuteError> {
        if params.len() != route.column_count {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "covered-INSERT intent expects {} params for table \"{}\", got {}",
                route.column_count,
                route.table,
                params.len()
            ))));
        }
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        Ok(())
    }

    /// Build the wave item for a covered-INSERT intent (shared by the blocking + async arms), or
    /// fall back to synthesized SQL text on eligibility drift. The pure-function equivalent of
    /// parse + `prepare_insert` for this shape: identity coercion, PK not-null vacuous for i32
    /// params, unique check wave-deferred + integer-slotted (a), value index elided-empty, and the
    /// W5a binary record PRE-ENCODED with a placeholder row id (b).
    fn build_covered_insert_intent(
        &self,
        txn_id: u64,
        route: &CoveredInsertRoute,
        params: &[i32],
        read_snapshot: Index,
    ) -> IntentBuild {
        // Re-derive eligibility against the LIVE catalog generation. The gate must match what the
        // wave sequencer will re-derive under the same generation stamp: a stale route (DDL) or a
        // de-elided table falls back to the classic path — never a validation skip.
        let catalog = self.catalog_snapshot();
        let prepared_catalog_seq = catalog.commit_seq;
        let table = catalog.relational_catalog.get(&route.table);
        let eligible = prepared_catalog_seq == route.catalog_seq
            && self.binary_wal_records_enabled()
            && table.is_some_and(|table| {
                table.columns.len() == route.column_count
                    && self.insert_unique_wave_batchable(&catalog, table)
            });
        if !eligible {
            return IntentBuild::Fallback(route.synthesize_text(params));
        }

        let values: Vec<SqlValue> = params.iter().map(|&param| SqlValue::Int4(param)).collect();
        // E2.2(a): build the ALLOCATION-FREE integer conflict slots from the route's precomputed
        // (slot_id, column_index) list — no String format, no (table, column) clone.
        let mut write_set = WriteSet::default();
        for &(slot_id, column_idx) in &route.unique_i32_slots {
            write_set
                .unique_slots_i32
                .push((slot_id, params[column_idx]));
        }
        let snapshot = self.dml_read_snapshot(read_snapshot);
        let row_key = relational_row_key(&route.table, snapshot.next_row_id);
        let delta = WriteDelta {
            write_set: write_set.clone(),
            rows_consumed: 1,
            mutation: PreparedMutation::Insert {
                table: route.table.clone(),
                inserted_rows: vec![(row_key, values.clone())],
                value_index_entries: BTreeMap::new(),
                seq_advances: BTreeMap::new(),
            },
        };
        // The Insert AST is the wave's validation currency (batched device locate needles; the
        // Full re-prepare on mid-wave catalog drift) — empty column list = catalog order.
        let cmd = Command::Insert(Insert {
            table: route.table.clone(),
            columns: Vec::new(),
            rows: vec![values.clone()],
        });
        // E2.2(b): pre-encode the W5a binary record with a PLACEHOLDER row id (0). The sequencer
        // patches the real id at `binary_row_id_offset`. Encoding is a pure function of the row
        // image, so this runs OFF the sequencer. `None` (width-exceeding; never for this shape)
        // leaves the sequencer's per-item encode path in charge.
        let binary_wal_template =
            crate::wal_binary::try_encode_binary_insert(&route.table, &[(0u64, values.as_slice())])
                .map(|bytes| {
                    (
                        std::sync::Arc::<[u8]>::from(bytes.as_slice()),
                        route.binary_row_id_offset,
                    )
                });
        // WAL fallback payload: the sequencer logs the W5a BINARY record for this reuse-eligible
        // delta; the text payload is written verbatim only on its fallback arms (catalog drift
        // mid-wave, binary encode decline), so it must stay valid replayable SQL.
        let text = route.synthesize_text(params);
        let mut residency_tables = BTreeSet::new();
        residency_tables.insert(route.table.clone());
        IntentBuild::Item(self.make_covered_insert_wave_item(
            txn_id,
            cmd,
            &text,
            write_set,
            read_snapshot,
            residency_tables,
            prepared_catalog_seq,
            Some(delta),
            binary_wal_template,
        ))
    }
}

/// The outcome of [`Engine::build_covered_insert_intent`]: a ready-to-enqueue wave item, or a
/// synthesized-SQL fallback for the classic path (eligibility drift).
enum IntentBuild {
    Item(crate::engine_dml_concurrent::CommitWaveItem),
    Fallback(String),
}

/// E2.2(c) — a handle to a submitted covered-INSERT intent. Poll it with [`Engine::poll_intent`];
/// the read-snapshot GC boundary it pins is released on the completing poll.
pub struct IntentTicket {
    /// The wave-item completion slot (async path); `None` once reaped or for the inline fallback.
    outcome: Option<crate::engine_dml_concurrent::CommitWaveOutcome>,
    /// The read-snapshot GC boundary this ticket still holds: an OWNED handle to the engine's
    /// active-snapshot registry plus the pinned snapshot. Released exactly once — by the
    /// completing poll, or by `Drop` (audit F2: a submitted-but-never-polled ticket must not pin
    /// `ledger.prune_below`/MVCC GC forever).
    snapshot_hold: Option<(
        std::sync::Arc<std::sync::Mutex<crate::ActiveSnapshots>>,
        Index,
    )>,
    /// A pre-resolved result (the inline classic fallback arm), yielded on the first poll.
    resolved: Option<Result<(), ExecuteError>>,
}

impl IntentTicket {
    /// Release the read-snapshot GC boundary (idempotent; `take` makes double-release a no-op).
    fn release_snapshot(&mut self) {
        if let Some((registry, snapshot)) = self.snapshot_hold.take() {
            registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .deregister(snapshot);
        }
    }
}

impl Drop for IntentTicket {
    fn drop(&mut self) {
        self.release_snapshot();
    }
}
