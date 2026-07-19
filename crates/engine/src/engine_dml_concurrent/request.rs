//! Concurrent DML request admission: snapshot capture, off-lock prepare, and wave submission.

use super::*;

impl Engine {
    /// Execute one autocommit DML statement on the concurrent snapshot-isolation path. The
    /// result-bearing twin is the only API that accepts `RETURNING`; unit APIs fail before work
    /// rather than silently discard rows.
    pub fn execute_dml_concurrent(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        if parse_command(text).is_ok_and(|command| command_has_returning(&command)) {
            return Err(discarded_returning_error());
        }
        self.execute_dml_concurrent_with_result(txn_id, text)
            .map(|_| ())
    }

    pub fn execute_dml_concurrent_with_result(
        &self,
        txn_id: u64,
        text: &str,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.intent_lanes_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return self.execute_dml_in_transaction_with_result(txn_id, text);
        }
        self.execute_dml_concurrent_instrumented_with_result(txn_id, text, || {})
    }

    /// Hooked unit-result compatibility API used by deterministic SI conflict tests.
    pub fn execute_dml_concurrent_instrumented(
        &self,
        txn_id: u64,
        text: &str,
        on_prepared: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        if parse_command(text).is_ok_and(|command| command_has_returning(&command)) {
            return Err(discarded_returning_error());
        }
        self.execute_dml_concurrent_instrumented_with_result(txn_id, text, on_prepared)
            .map(|_| ())
    }

    fn execute_dml_concurrent_instrumented_with_result(
        &self,
        txn_id: u64,
        text: &str,
        on_prepared: impl FnOnce(),
    ) -> Result<DmlExecutionResult, ExecuteError> {
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
        let cmd = parse_command(text)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
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
            Some(prepared),
        )
    }
}
