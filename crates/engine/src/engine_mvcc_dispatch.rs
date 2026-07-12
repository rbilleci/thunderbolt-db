//! MVCC read-query dispatch (P0 §9.6 decomposition, behavior-preserving): a
//! focused `impl Engine` block that drives an MvccReadQuery through the resident
//! CUDA-native source path / the Cpu+Cuda execution backends with fallback, and
//! records the probe + read-result metrics. The backends/model live in
//! mvcc_read_exec / mvcc_read_model; this is the Engine-side orchestration.

use super::*;

impl Engine {
    /// Resolve a `MvccReadQuery` against the **KV partition** (the non-relational namespace).
    /// This is the table-agnostic entry the KV `MvccReadSource` machinery uses; relational reads
    /// go through [`Engine::execute_mvcc_query_on_pin`] so they resolve against the SAME pinned
    /// generation their value-index lookup used (write-half Stage 4, prereq #1).
    pub fn execute_mvcc_query(
        &self,
        query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        #[cfg(not(test))]
        return self.execute_mvcc_query_with_cuda_driver_probe(query);
        #[cfg(test)]
        {
        let backend = CpuMvccExecutionBackend;
        let kv = self.read_state.mvcc.load_kv();
        self.execute_mvcc_query_with_fallback_reason(
            kv.get(),
            query,
            &backend,
            Some(FallbackReason::GpuMvccReadParityGap),
            false,
        )
        }
    }

    /// Resolve a `MvccReadQuery` against the SAME pinned generation the query was built from
    /// (prereq #1, Stage 4) — the rows come from the exact `commit_seq` whose value-index produced
    /// the keys, so no concurrent publish can interleave index and rows.
    #[cfg(test)]
    pub(crate) fn execute_mvcc_query_on_pin(
        &self,
        pin: &RelationalReadPin,
        query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        let backend = CpuMvccExecutionBackend;
        self.execute_mvcc_query_with_fallback_reason(
            pin.store(),
            query,
            &backend,
            Some(FallbackReason::GpuMvccReadParityGap),
            false,
        )
    }

    pub fn execute_mvcc_query_with_cuda_driver_probe(
        &self,
        query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        let kv = self.read_state.mvcc.load_kv();
        self.execute_mvcc_query_with_cuda_driver_probe_on_store(kv.get(), query)
    }

    pub(crate) fn execute_mvcc_query_with_cuda_driver_probe_on_store(
        &self,
        read_store: &InMemoryTupleStore,
        query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        let runtime = self.cuda_driver_probe_runtime();
        let backend = CudaMvccExecutionBackend::new(runtime, self.planner.default_gpu_id());
        if is_cuda_native_source_query(query) {
            return self.execute_cuda_native_source_query(read_store, query, &backend);
        }

        self.execute_mvcc_query_with_fallback_reason(read_store, query, &backend, None, true)
    }

    pub(crate) fn cuda_driver_probe_runtime(&self) -> CudaDriverRuntime {
        self.cached_cuda_probe_runtime
            .get_or_init(|| {
                CudaDriverRuntime::probe().unwrap_or_else(|_| CudaDriverRuntime::unavailable())
            })
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn execute_mvcc_query_with_backend<B: MvccExecutionBackend>(
        &mut self,
        query: &MvccReadQuery,
        backend: &B,
    ) -> Result<MvccReadResult, ExecuteError> {
        let kv = self.read_state.mvcc.load_kv();
        self.execute_mvcc_query_with_fallback_reason(kv.get(), query, backend, None, false)
    }

    #[cfg(test)]
    pub(crate) fn execute_mvcc_query_with_backend_fallback<B: MvccExecutionBackend>(
        &mut self,
        query: &MvccReadQuery,
        backend: &B,
    ) -> Result<MvccReadResult, ExecuteError> {
        let kv = self.read_state.mvcc.load_kv();
        self.execute_mvcc_query_with_fallback_reason(kv.get(), query, backend, None, false)
    }

    /// Test-only: `execute_cuda_native_source_query` resolving against the KV partition (the
    /// KV-namespace CUDA-native tests use this).
    #[cfg(test)]
    pub(crate) fn execute_cuda_native_source_query_kv(
        &self,
        query: &MvccReadQuery,
        backend: &CudaMvccExecutionBackend,
    ) -> Result<MvccReadResult, ExecuteError> {
        let kv = self.read_state.mvcc.load_kv();
        self.execute_cuda_native_source_query(kv.get(), query, backend)
    }

    pub(crate) fn execute_mvcc_query_with_fallback_reason<B: MvccExecutionBackend>(
        &self,
        read_store: &InMemoryTupleStore,
        query: &MvccReadQuery,
        backend: &B,
        fallback_reason: Option<FallbackReason>,
        observe_cuda_probe_metrics: bool,
    ) -> Result<MvccReadResult, ExecuteError> {
        // The leader gate re-reads `repl_role()` (which locks the commit_mutex). A relational SELECT
        // executed as part of *applying* a committed materialized-view entry runs INSIDE the commit
        // critical section (which already holds the commit_mutex) and is on a guaranteed leader, so
        // re-locking here would self-deadlock the non-reentrant commit_mutex. `mvcc_read_skips_leader_
        // check()` is set only on that internal apply-path read; every client read leaves it false and
        // is gated exactly as before.
        if !self.mvcc_read_skips_leader_check() && self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }

        let planned_target = DeviceTarget::Gpu(self.planner.default_gpu_id());
        let rows = resolve_mvcc_source(read_store, &query.source, query.visibility)?;
        let cuda_h2d_bytes = if observe_cuda_probe_metrics {
            cuda_mvcc_row_batch_transfer_bytes(&rows)
        } else {
            0
        };

        let cuda_start = observe_cuda_probe_metrics.then(Instant::now);
        #[cfg(test)]
        let backend_result =
            execute_mvcc_backend_chain(query, rows, backend, &CpuMvccExecutionBackend);
        #[cfg(not(test))]
        let backend_result = match backend.execute(query, rows) {
            MvccBackendDispatch::Executed(executed) => FinalizedMvccBackendExecution {
                executed_target: executed.executed_target,
                fallback_reason: None,
                rows: executed.rows,
            },
            MvccBackendDispatch::Fallback { reason, .. } => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "GPU execution is required for MVCC reads: {reason:?}"
                ))))
            }
        };
        let result = MvccReadResult {
            planned_target,
            executed_target: backend_result.executed_target,
            fallback_reason: backend_result.fallback_reason.or(fallback_reason),
            rows: backend_result.rows,
        };
        if let Some(start) = cuda_start {
            self.observe_cuda_probe_execution_metrics(&result, cuda_h2d_bytes, start.elapsed());
        }
        self.observe_mvcc_read_result_metrics(&result);
        Ok(result)
    }

    fn execute_cuda_native_source_query(
        &self,
        read_store: &InMemoryTupleStore,
        query: &MvccReadQuery,
        backend: &CudaMvccExecutionBackend,
    ) -> Result<MvccReadResult, ExecuteError> {
        // Same mid-commit reentrancy guard as `execute_mvcc_query_with_fallback_reason` (see there).
        if !self.mvcc_read_skips_leader_check() && self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }

        let planned_target = DeviceTarget::Gpu(self.planner.default_gpu_id());
        let rows = resolve_mvcc_all_versions(read_store, query.visibility)?;
        let mut cuda_h2d_bytes = cuda_mvcc_row_batch_transfer_bytes(&rows);

        let cuda_start = Instant::now();
        let backend_attempt = match &query.source {
            MvccReadSource::KeyBatchLookup { keys } => {
                match execute_cuda_native_key_batch_query(query, keys, rows, backend) {
                    Ok((execution, key_batch_h2d_bytes)) => {
                        cuda_h2d_bytes = key_batch_h2d_bytes;
                        Ok(execution)
                    }
                    Err(reason) => Err(reason),
                }
            }
            MvccReadSource::Concat { sources } => {
                execute_cuda_native_concat_query(query, sources, rows, backend)
            }
            MvccReadSource::FollowValueChain {
                keys,
                plan,
                provenance,
            } => execute_cuda_native_follow_value_chain_query(
                query,
                keys,
                *plan,
                *provenance,
                rows,
                backend,
            ),
            MvccReadSource::FollowValueChainBranches {
                keys,
                plans,
                fan_in,
                provenance,
            } => execute_cuda_native_follow_value_chain_branches_query(
                query,
                keys,
                plans,
                *fan_in,
                *provenance,
                rows,
                backend,
            ),
            MvccReadSource::FollowValueChainLabeledBranches {
                keys,
                branches,
                fan_in,
                provenance,
            } => execute_cuda_native_follow_value_chain_labeled_branches_query(
                query,
                keys,
                branches,
                *fan_in,
                *provenance,
                rows,
                backend,
            ),
            source if is_cuda_native_composition_source(source) => {
                execute_cuda_native_composition_query(query, source, rows, backend)
            }
            _ => execute_cuda_native_single_source_query(query, rows, backend),
        };
        #[cfg(test)]
        let backend_result = backend_attempt.unwrap_or_else(|reason| {
            let visible_rows = resolve_mvcc_source(read_store, &query.source, query.visibility)
                .expect("visibility was validated before native CUDA dispatch");
            let cpu_execution = match CpuMvccExecutionBackend.execute(query, visible_rows) {
                MvccBackendDispatch::Executed(executed) => executed,
                MvccBackendDispatch::Fallback { .. } => {
                    unreachable!("CPU fallback backend must execute")
                }
            };
            FinalizedMvccBackendExecution {
                executed_target: cpu_execution.executed_target,
                fallback_reason: Some(reason),
                rows: cpu_execution.rows,
            }
        });
        #[cfg(not(test))]
        let backend_result = backend_attempt.map_err(|reason| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "GPU execution is required for MVCC reads: {reason:?}"
            )))
        })?;

        let result = MvccReadResult {
            planned_target,
            executed_target: backend_result.executed_target,
            fallback_reason: backend_result.fallback_reason,
            rows: backend_result.rows,
        };
        self.observe_cuda_probe_execution_metrics(&result, cuda_h2d_bytes, cuda_start.elapsed());
        self.observe_mvcc_read_result_metrics(&result);
        Ok(result)
    }

    fn observe_cuda_probe_execution_metrics(
        &self,
        result: &MvccReadResult,
        h2d_bytes: u64,
        elapsed: Duration,
    ) {
        if !matches!(result.executed_target, DeviceTarget::Gpu(_)) {
            return;
        }

        if h2d_bytes > 0 {
            self.metrics.observe_h2d_bytes(h2d_bytes);
        }
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
    }

    fn observe_mvcc_read_result_metrics(&self, result: &MvccReadResult) {
        if let Some(reason) = result.fallback_reason {
            self.metrics.inc_fallback(reason);
        }

        let total_d2h_bytes: u64 = result.rows.iter().map(mvcc_read_row_size).sum();
        if total_d2h_bytes > 0 {
            self.metrics.observe_d2h_bytes(total_d2h_bytes);
        }
    }
}
