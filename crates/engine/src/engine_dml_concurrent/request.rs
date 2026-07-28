//! Concurrent DML request admission: snapshot capture, off-lock prepare, and wave submission.

use super::*;

impl Engine {
    /// Execute one autocommit DML statement on the concurrent snapshot-isolation path. The
    /// result-bearing twin is the only API that accepts `RETURNING`; unit APIs fail before work
    /// rather than silently discard rows.
    pub fn execute_dml_concurrent(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let command = parse_command(text)?;
        if command_has_returning(&command) {
            return Err(discarded_returning_error());
        }
        self.execute_parsed_dml_concurrent_with_result(txn_id, command, text)
            .map(|_| ())
    }

    pub fn execute_dml_concurrent_with_result(
        &self,
        txn_id: u64,
        text: &str,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        let command = parse_command(text)?;
        self.execute_parsed_dml_concurrent_with_result(txn_id, command, text)
    }

    pub(crate) fn execute_parsed_dml_concurrent_with_result(
        &self,
        txn_id: u64,
        command: Command,
        text: &str,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.execute_parsed_dml_concurrent_with_catalog(txn_id, command, text, None)
    }

    pub(crate) fn execute_parsed_dml_concurrent_with_catalog(
        &self,
        txn_id: u64,
        command: Command,
        text: &str,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        // This compatibility entry accepts caller-assigned identities. An ordinary sequence
        // default may claim its own transaction before the user DML record, so reserve the outer
        // identity in the shared allocator first.
        self.observe_transaction_id(txn_id);
        self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.transaction_snapshot_handle(txn_id).is_some() {
            if let Some(expectation) = expected_catalog_version {
                let snapshot = self
                    .transaction_snapshot_handle(txn_id)
                    .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
                crate::engine_mutation_admission::validate_catalog_version_expectation(
                    expectation,
                    snapshot.catalog.commit_seq,
                )?;
            }
            return self.execute_parsed_dml_in_transaction_with_result(txn_id, command);
        }
        self.execute_parsed_dml_concurrent_instrumented_with_catalog(
            txn_id,
            command,
            text,
            expected_catalog_version,
            || {},
        )
    }

    /// Hooked unit-result compatibility API used by deterministic SI conflict tests.
    pub fn execute_dml_concurrent_instrumented(
        &self,
        txn_id: u64,
        text: &str,
        on_prepared: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        let command = parse_command(text)?;
        if command_has_returning(&command) {
            return Err(discarded_returning_error());
        }
        self.execute_parsed_dml_concurrent_instrumented_with_result(
            txn_id,
            command,
            text,
            on_prepared,
        )
        .map(|_| ())
    }

    pub(crate) fn execute_parsed_dml_concurrent_instrumented_with_result(
        &self,
        txn_id: u64,
        cmd: Command,
        text: &str,
        on_prepared: impl FnOnce(),
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.execute_parsed_dml_concurrent_instrumented_with_catalog(
            txn_id,
            cmd,
            text,
            None,
            on_prepared,
        )
    }

    pub(crate) fn execute_parsed_dml_concurrent_instrumented_with_catalog(
        &self,
        txn_id: u64,
        cmd: Command,
        text: &str,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
        on_prepared: impl FnOnce(),
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.observe_transaction_id(txn_id);
        self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.is_commit_path_poisoned() {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "commit path is wedged; restart recovery required".to_string(),
            )));
        }
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "instrumented concurrent DML is autocommit-only; an active transaction must use the serialized private-generation entry"
                    .to_string(),
            )));
        }
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let table_name = match &cmd {
            Command::Insert(insert) => insert.table.as_str(),
            Command::Update(update) => update.table.as_str(),
            Command::Delete(delete) => delete.table.as_str(),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "concurrent DML admission received a non-DML command".to_string(),
                )))
            }
        };
        // Stable transaction identity resolves before fresh table access or state-sensitive
        // device preparation. The shared helper also re-resolves after a raced lease failure: an
        // identical request can become terminal between the first lookup and an intervening reset
        // taking exclusivity.
        let request = CanonicalRequest::from_text(self, text);
        let _table_access = match self.acquire_autocommit_table_access_after_retry(
            table_name,
            txn_id,
            request.digest(),
        )? {
            StableRetryOr::Terminal(affected_rows) => {
                if command_has_returning(&cmd) {
                    return Err(ExecuteError::Unsupported(
                            "terminal retry of DML RETURNING is fail-closed until canonical status persists the returned frame"
                                .to_string(),
                        ));
                }
                return Ok(DmlExecutionResult {
                    rows_affected: affected_rows,
                    returning: None,
                });
            }
            StableRetryOr::Fresh(access) => access,
        };
        self.ensure_dml_device_generation(&cmd)?;
        let transaction_snapshot = self.transaction_snapshot_handle(txn_id);
        let read_snapshot = transaction_snapshot
            .as_ref()
            .map_or_else(|| self.committed_seq(), |snapshot| snapshot.boundary);
        let _snapshot_guard = transaction_snapshot
            .is_none()
            .then(|| self.register_active_snapshot(read_snapshot));
        let pinned_catalog = transaction_snapshot.as_ref().map_or_else(
            || self.catalog_snapshot(),
            |snapshot| Arc::clone(&snapshot.catalog),
        );
        let prepared_catalog_seq = pinned_catalog.commit_seq;
        let snapshot = transaction_snapshot.as_ref().map_or_else(
            || self.dml_read_snapshot(read_snapshot),
            |generation| DmlReadSnapshot {
                commit_seq: generation.boundary,
                next_row_id: generation.next_row_id,
            },
        );
        #[cfg(feature = "probe-timing")]
        let probe_insert = matches!(&cmd, Command::Insert(_));
        #[cfg(feature = "probe-timing")]
        let probe_prepare_started = probe_insert.then(Instant::now);
        let direct_typed = if self.binary_wal_records_enabled() {
            crate::typed_insert_batch::try_prepare_typed_insert_batch(
                &cmd,
                &pinned_catalog,
                prepared_catalog_seq,
                expected_catalog_version,
            )?
        } else {
            None
        };
        let (write_set, offlock_prepared) = if let Some(batch) = direct_typed {
            let prepared = OfflockPreparedDml::typed_insert(
                batch,
                self,
                &pinned_catalog,
                &request,
                read_snapshot,
            )
            .map_err(ExecuteError::Engine)?;
            let write_set = prepared.write_set().clone();
            debug_assert_eq!(prepared.read_snapshot(), read_snapshot);
            #[cfg(feature = "probe-timing")]
            self.record_insert_probe_direct_fixed_insert_carrier();
            (write_set, Some(prepared))
        } else {
            let prepared = {
                const GENERATION_RETRIES: usize = 64;
                let mut attempts = 0;
                loop {
                    let result = if let Some(generation) = transaction_snapshot.as_ref() {
                        let _scope = self.enter_transaction_read(Arc::clone(generation));
                        self.prepare_dml(&cmd, snapshot, InsertPrepareValidation::WaveOffLock)
                    } else {
                        self.prepare_dml(&cmd, snapshot, InsertPrepareValidation::WaveOffLock)
                    };
                    match result {
                        Ok(prepared) => break prepared,
                        Err(error)
                            if attempts < GENERATION_RETRIES
                                && is_device_prepare_verdict_unavailable(&error) =>
                        {
                            attempts += 1;
                            self.ensure_dml_device_generation(&cmd)?;
                            std::thread::yield_now();
                        }
                        Err(error) => return Err(error),
                    }
                }
            };
            #[cfg(feature = "probe-timing")]
            if matches!(&cmd, Command::Insert(_)) {
                self.record_insert_probe_legacy_insert_delta_build(prepared.rows_consumed);
            }
            let prepared = OfflockPreparedDml::legacy(prepared, read_snapshot);
            let write_set = prepared.write_set().clone();
            debug_assert_eq!(prepared.read_snapshot(), read_snapshot);
            (write_set, Some(prepared))
        };
        #[cfg(feature = "probe-timing")]
        if let Some(started) = probe_prepare_started {
            self.record_insert_probe_offlock_prepare_nanos(started.elapsed().as_nanos() as u64);
            self.record_insert_probe_peak_statement(
                text.len() as u64,
                insert_device_statement_bytes_estimate(&cmd),
            );
        }
        on_prepared();
        self.commit_dml_concurrent(
            txn_id,
            cmd,
            request,
            write_set,
            read_snapshot,
            prepared_catalog_seq,
            expected_catalog_version,
            offlock_prepared,
        )
    }
}

/// Conservative logical payload estimate for the device append staging seam. This is deliberately
/// labelled an estimate: its owner is the typed request before table-specific residency layout is
/// selected, while the exact H2D/apply wall time is measured at the common wave flush seam.
#[cfg(feature = "probe-timing")]
fn insert_device_statement_bytes_estimate(command: &Command) -> u64 {
    let Command::Insert(insert) = command else {
        return 0;
    };
    insert
        .rows
        .iter()
        .map(|row| {
            let values = row.iter().fold(0_u64, |bytes, cell| {
                bytes.saturating_add(match cell {
                    InsertCell::Default { .. } => 0,
                    InsertCell::Value { value, .. } => match value {
                        // This estimator runs before semantic lowering. An unresolved parameter
                        // cannot have staging bytes yet, just like an explicit DEFAULT request.
                        SqlValue::Null | SqlValue::Parameter { .. } => 0,
                        SqlValue::Bool(_) => 1,
                        SqlValue::Int2(_) => 2,
                        SqlValue::Int4(_) | SqlValue::Date(_) => 4,
                        SqlValue::Int8(_) | SqlValue::Timestamp(_) => 8,
                        SqlValue::Numeric(_) | SqlValue::Uuid(_) => 16,
                        SqlValue::Text(text) => text.len() as u64,
                    },
                })
            });
            // The append seam retains row identity and MVCC birth metadata alongside values.
            values.saturating_add(16)
        })
        .sum()
}

#[cfg(all(test, feature = "probe-timing"))]
mod probe_timing_tests {
    use super::*;

    #[test]
    fn insert_payload_estimate_keeps_default_and_unbound_parameter_at_zero_bytes() {
        let command = Command::Insert(Insert {
            table: "probe_cells".to_string(),
            columns: vec![
                "id".to_string(),
                "defaulted".to_string(),
                "parameter".to_string(),
                "note".to_string(),
                "nullable".to_string(),
            ],
            rows: vec![vec![
                InsertCell::literal(SqlValue::Int4(7)),
                InsertCell::sql_default(),
                InsertCell::Value {
                    value: SqlValue::Parameter {
                        index: 1,
                        cast: None,
                    },
                    provenance: gpu_db_sql::InsertValueProvenance::Parameter { index: 1 },
                },
                InsertCell::programmatic(SqlValue::Text("gpu".to_string())),
                InsertCell::literal(SqlValue::Null),
            ]],
            returning: Vec::new(),
        });

        assert_eq!(insert_device_statement_bytes_estimate(&command), 16 + 4 + 3);
    }
}
