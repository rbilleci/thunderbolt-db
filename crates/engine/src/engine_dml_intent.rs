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
    /// Catalog generation at prepare. Execute compares the LIVE generation:
    /// a mismatch means a DDL committed since — the shape proof is stale.
    catalog_seq: Index,
    column_count: usize,
    /// `"INSERT INTO <table> VALUES ("` — the synthesized-SQL prefix for the
    /// WAL-fallback payload and the classic-path fallback.
    sql_prefix: String,
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
        Ok(CoveredInsertRoute {
            table: table.name.clone(),
            catalog_seq: catalog.commit_seq,
            column_count: table.columns.len(),
            sql_prefix: format!("INSERT INTO {} VALUES (", table.name),
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
        // Pin + register the read snapshot exactly like the classic off-lock
        // prepare (the guard holds the MVCC GC boundary; conflicts() validates
        // against this snapshot).
        let read_snapshot = self.committed_seq();
        let snapshot_guard = self.register_active_snapshot(read_snapshot);

        // Re-derive eligibility against the LIVE catalog generation. The gate
        // must match what the wave sequencer will re-derive under the same
        // generation stamp: a stale route (DDL) or a de-elided table falls
        // back to the classic path — never a validation skip.
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
            drop(snapshot_guard);
            let text = route.synthesize_text(params);
            return self.execute_dml_concurrent(txn_id, &text);
        }
        let table = table.expect("eligibility checked table presence");

        // Build the wave item directly — the pure-function equivalent of
        // parse + `prepare_insert` for this shape (identity coercion, PK
        // not-null vacuous for i32 params, unique check wave-deferred, value
        // index elided-empty). The row key is snapshot-relative exactly like
        // `prepare_insert`'s encode; the sequencer re-keys it at the wave's
        // row-id cursor (`rekey_offlock_insert_delta`).
        let values: Vec<SqlValue> = params.iter().map(|&param| SqlValue::Int4(param)).collect();
        let mut write_set = WriteSet::default();
        write_set.add_unique_slots(table, &values);
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
        // The Insert AST is the wave's validation currency (batched device
        // locate needles; the Full re-prepare on mid-wave catalog drift) —
        // empty column list = catalog order, matching `params`.
        let cmd = Command::Insert(Insert {
            table: route.table.clone(),
            columns: Vec::new(),
            rows: vec![values],
        });
        // WAL fallback payload: the sequencer logs the W5a BINARY record for
        // this reuse-eligible delta; the text payload is written verbatim only
        // on its fallback arms (catalog drift mid-wave, binary encode decline),
        // so it must stay valid replayable SQL.
        let text = route.synthesize_text(params);
        let mut residency_tables = BTreeSet::new();
        residency_tables.insert(route.table.clone());
        self.commit_dml_concurrent(
            txn_id,
            cmd,
            &text,
            write_set,
            read_snapshot,
            residency_tables,
            prepared_catalog_seq,
            Some(delta),
        )
    }
}
