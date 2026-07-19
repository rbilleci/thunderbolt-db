//! Generic KV/MVCC query boundary and the shared cached CUDA-runtime probe.

use super::*;

const GENERIC_MVCC_UNSUPPORTED: &str =
    "generic KV MVCC queries require a device-resident result pipeline; host-staged execution is retired";

impl Engine {
    /// The generic table-agnostic KV query surface is intentionally unsupported.
    ///
    /// Relational reads use pinned resident GPU operators. The former implementation resolved host
    /// tuples before launching mask kernels and then performed selection, ordering, LIMIT, projection,
    /// and result assembly on the CPU. Failing here, before loading or resolving the KV store, keeps
    /// that compatibility surface from masquerading as GPU execution while SIDE-001 remains parked.
    pub fn execute_mvcc_query(
        &self,
        _query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if !self.mvcc_read_skips_leader_check() && self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        Err(ExecuteError::Engine(EngineError::ApplyFailed(
            GENERIC_MVCC_UNSUPPORTED.to_string(),
        )))
    }

    /// Compatibility name retained for callers that explicitly requested the old CUDA probe.
    /// It has the same fail-loud boundary and performs no source resolution, transfer, or metrics.
    pub fn execute_mvcc_query_with_cuda_driver_probe(
        &self,
        query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        self.execute_mvcc_query(query)
    }

    /// Evaluate an MVCC fixture against the closed-form, rows-only test specification.
    #[cfg(test)]
    pub(crate) fn evaluate_mvcc_query_specification(
        &self,
        query: &MvccReadQuery,
    ) -> Result<MvccSpecificationResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let kv = self.read_state.mvcc.load_kv();
        self.evaluate_mvcc_query_specification_on_store(kv.get(), query)
    }

    #[cfg(test)]
    pub(crate) fn evaluate_mvcc_query_specification_on_pin(
        &self,
        pin: &RelationalReadPin,
        query: &MvccReadQuery,
    ) -> Result<MvccSpecificationResult, ExecuteError> {
        self.evaluate_mvcc_query_specification_on_store(pin.store(), query)
    }

    #[cfg(test)]
    fn evaluate_mvcc_query_specification_on_store(
        &self,
        read_store: &InMemoryTupleStore,
        query: &MvccReadQuery,
    ) -> Result<MvccSpecificationResult, ExecuteError> {
        if !self.mvcc_read_skips_leader_check() && self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let rows = resolve_mvcc_source(read_store, &query.source, query.visibility)?;
        Ok(evaluate_mvcc_specification(query, rows))
    }

    pub(crate) fn cuda_driver_probe_runtime(&self) -> CudaDriverRuntime {
        self.cached_cuda_probe_runtime
            .get_or_init(|| {
                CudaDriverRuntime::probe().unwrap_or_else(|_| CudaDriverRuntime::unavailable())
            })
            .clone()
    }
}
