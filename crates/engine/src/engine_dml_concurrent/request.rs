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
        expected_catalog_version: Option<u64>,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.transaction_snapshot_handle(txn_id).is_some() {
            if let Some(expected) = expected_catalog_version {
                let snapshot = self
                    .transaction_snapshot_handle(txn_id)
                    .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
                crate::engine_mutation_admission::validate_prepared_catalog_version(
                    expected,
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
        expected_catalog_version: Option<u64>,
        on_prepared: impl FnOnce(),
    ) -> Result<DmlExecutionResult, ExecuteError> {
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
        let request_digest = gpu_db_wal::canonical_request_digest(text.as_bytes());
        let _table_access = match self.acquire_autocommit_table_access_after_retry(
            table_name,
            txn_id,
            request_digest,
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
        let prepared_catalog_seq = transaction_snapshot.as_ref().map_or_else(
            || self.catalog_snapshot().commit_seq,
            |snapshot| snapshot.catalog.commit_seq,
        );
        let snapshot = transaction_snapshot.as_ref().map_or_else(
            || self.dml_read_snapshot(read_snapshot),
            |generation| DmlReadSnapshot {
                commit_seq: generation.boundary,
                next_row_id: generation.next_row_id,
            },
        );
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
        on_prepared();
        let write_set = prepared.write_set.clone();
        self.commit_dml_concurrent(
            txn_id,
            cmd,
            text,
            write_set,
            read_snapshot,
            prepared_catalog_seq,
            expected_catalog_version,
            Some(prepared),
        )
    }
}
