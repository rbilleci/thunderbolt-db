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
        let txn_id = self.resolve_public_transaction_id(txn_id);
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
        if matches!(&command, Command::Insert(_)) {
            let (requests_published_sequence_default, route_catalog_version) =
                self.insert_sequence_default_route(&command);
            let catalog_expectation = expected_catalog_version.or_else(|| {
                requests_published_sequence_default.then(|| {
                    crate::engine_mutation_admission::CatalogVersionExpectation::SequenceRoute(
                        route_catalog_version
                            .expect("published sequence-default route reports its catalog cut"),
                    )
                })
            });
            if requests_published_sequence_default {
                return self.execute_sequence_default_autocommit(
                    txn_id,
                    command,
                    catalog_expectation,
                );
            }
            return self.execute_autocommit_insert_as_one_statement_overlay(
                txn_id,
                command,
                CanonicalRequest::from_text(self, text).digest(),
                catalog_expectation,
                current_timestamp_micros(),
                || {},
            );
        }
        self.execute_parsed_dml_concurrent_instrumented_with_catalog(
            txn_id,
            command,
            text,
            expected_catalog_version,
            || {},
        )
    }

    /// Every autocommit `INSERT` is one private overlay followed immediately by the existing
    /// canonical transaction terminal. This deliberately reuses explicit staging, device result
    /// materialization, rollback, WAL, apply, and publication rather than letting ordinary
    /// autocommit and `RETURNING` own different INSERT lifecycles.
    pub(crate) fn execute_autocommit_insert_as_one_statement_overlay<OnStaged>(
        &self,
        txn_id: u64,
        command: Command,
        request_digest: gpu_db_wal::CanonicalDigest,
        expectation: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
        timestamp_micros: u64,
        on_staged: OnStaged,
    ) -> Result<DmlExecutionResult, ExecuteError>
    where
        OnStaged: FnOnce(),
    {
        if self.is_commit_path_poisoned() {
            return Err(super::durability_failure::execute_error_from_engine(
                self.commit_path_unavailable_error(),
            ));
        }
        if let Some(rows_affected) =
            self.resolve_stable_retry_before_table_access(txn_id, request_digest)?
        {
            if command_has_returning(&command) {
                return Err(ExecuteError::Unsupported(
                    "terminal retry of INSERT RETURNING is fail-closed until canonical status persists the returned frame"
                        .to_string(),
                ));
            }
            return Ok(DmlExecutionResult {
                rows_affected,
                returning: None,
            });
        }
        if let Some(expectation) = expectation {
            crate::engine_mutation_admission::validate_catalog_version_expectation(
                expectation,
                self.catalog_snapshot().commit_seq,
            )?;
        }
        // Establish the complete target/FK device generation set before capturing the
        // one-statement transaction snapshot. This is the same GPU-native admission boundary
        // used by the concurrent DML route: a missing generation is uploaded under the canonical
        // commit/catalog lock order, never reconstructed from host rows inside the private
        // transaction or after WAL.
        self.ensure_dml_device_generation(&command)?;
        match self.reserve_pending_transaction_claim(txn_id, request_digest) {
            Ok(true) => {}
            Ok(false) => {
                return Err(ExecuteError::Indeterminate(format!(
                    "autocommit INSERT transaction {txn_id} is pending in canonical mutation admission"
                )));
            }
            Err(error) => return Err(ExecuteError::Engine(error)),
        }
        if let Err(begin_error) = self.begin_claimed_transaction_context(
            txn_id,
            TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
            request_digest,
        ) {
            self.release_pending_transaction_claim(txn_id, request_digest);
            return match self.resolve_stable_retry_before_table_access(txn_id, request_digest) {
                Ok(Some(rows_affected)) if !command_has_returning(&command) => Ok(DmlExecutionResult {
                    rows_affected,
                    returning: None,
                }),
                Ok(Some(_)) => Err(ExecuteError::Unsupported(
                    "terminal retry of INSERT RETURNING is fail-closed until canonical status persists the returned frame"
                        .to_string(),
                )),
                Ok(None) => Err(begin_error),
                Err(retry_error) => Err(retry_error),
            };
        }
        #[cfg(feature = "probe-timing")]
        let probe_statement_stage_started = std::time::Instant::now();
        let result = match self.execute_prepared_dml_in_transaction_with_result(
            txn_id,
            command,
            expectation,
            true,
        ) {
            Ok(result) => {
                on_staged();
                result
            }
            Err(statement_error) => match self.cancel_internal_transaction_context(txn_id) {
                Ok(()) => {
                    self.release_pending_transaction_claim(txn_id, request_digest);
                    return Err(statement_error);
                }
                Err(cancel_error) => {
                    return Err(ExecuteError::Indeterminate(format!(
                        "autocommit INSERT failed before its private overlay could cancel: \
                         statement error: {statement_error}; cancellation error: {cancel_error}"
                    )));
                }
            },
        };
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_transaction_statement_stage_nanos(
            probe_statement_stage_started.elapsed().as_nanos() as u64,
        );
        #[cfg(feature = "probe-timing")]
        let probe_terminal_started = std::time::Instant::now();
        match self.commit_claimed_transaction_delta(txn_id, request_digest, timestamp_micros) {
            Ok(()) => {
                // This shared terminal also serves public `execute_text` INSERTs. Keep W1
                // checkpoint rotation here, after the commit lock has been released, so neither
                // typed autocommit ingress can retain an unbounded live WAL segment.
                self.maybe_auto_checkpoint_wal();
                #[cfg(feature = "probe-timing")]
                self.record_insert_probe_transaction_terminal_nanos(
                    probe_terminal_started.elapsed().as_nanos() as u64,
                );
                Ok(result)
            }
            Err(commit_error) => {
                // A pre-WAL terminal rejection still owns only the internal overlay. Release it
                // and its pending claim so the caller's stable request ID remains retryable. A
                // wedged/post-durable terminal deliberately remains for restart recovery.
                if !self.is_commit_path_poisoned()
                    && self.transaction_snapshot_handle(txn_id).is_some()
                {
                    if let Err(cancel_error) = self.cancel_internal_transaction_context(txn_id) {
                        return Err(ExecuteError::Indeterminate(format!(
                            "autocommit INSERT commit rejected before durability but its private \
                             overlay could not cancel: commit error: {commit_error}; cancellation \
                             error: {cancel_error}"
                        )));
                    }
                    self.release_pending_transaction_claim(txn_id, request_digest);
                }
                Err(commit_error)
            }
        }
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
        let txn_id = self.resolve_public_transaction_id(txn_id);
        self.observe_transaction_id(txn_id);
        self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        if self.is_commit_path_poisoned() {
            return Err(super::durability_failure::execute_error_from_engine(
                self.commit_path_unavailable_error(),
            ));
        }
        if self.transaction_snapshot_handle(txn_id).is_some() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "instrumented concurrent DML is autocommit-only; an active transaction must use the serialized private-generation entry"
                    .to_string(),
            )));
        }
        if matches!(&cmd, Command::Insert(_)) {
            let (requests_published_sequence_default, route_catalog_version) =
                self.insert_sequence_default_route(&cmd);
            let catalog_expectation = expected_catalog_version.or_else(|| {
                requests_published_sequence_default.then(|| {
                    crate::engine_mutation_admission::CatalogVersionExpectation::SequenceRoute(
                        route_catalog_version
                            .expect("published sequence-default route reports its catalog cut"),
                    )
                })
            });
            if requests_published_sequence_default {
                // The instrumented facade exposes only a test synchronization hook. Sequence
                // defaults still take their dedicated parent lifecycle rather than a generic
                // overlay; signal the hook at the equivalent admission boundary.
                on_prepared();
                return self.execute_sequence_default_autocommit(txn_id, cmd, catalog_expectation);
            }
            return self.execute_autocommit_insert_as_one_statement_overlay(
                txn_id,
                cmd,
                CanonicalRequest::from_text(self, text).digest(),
                catalog_expectation,
                current_timestamp_micros(),
                on_prepared,
            );
        }
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let table_name = match &cmd {
            Command::Update(update) => update.table.as_str(),
            Command::Delete(delete) => delete.table.as_str(),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "concurrent DML admission received a non-DML command".to_string(),
                )));
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
                    self.prepare_dml(&cmd, snapshot)
                } else {
                    self.prepare_dml(&cmd, snapshot)
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
        let prepared = OfflockPreparedDml::legacy_update_or_delete(prepared, read_snapshot)
            .map_err(ExecuteError::Engine)?;
        let write_set = prepared.write_set().clone();
        debug_assert_eq!(prepared.read_snapshot(), read_snapshot);
        on_prepared();
        self.commit_dml_concurrent(
            txn_id,
            cmd,
            request,
            write_set,
            read_snapshot,
            expected_catalog_version,
            Some(prepared),
        )
    }
}
