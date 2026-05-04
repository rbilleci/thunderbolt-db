# No-GPU Bootstrap Closeout Review

Date: 2026-05-04
Status: closed
Mode: closeout consolidation / CUDA-transition preparation

## Decision

The no-GPU bootstrap phase is closed. The current engine-facing MVCC contract is broad enough for the first CUDA slice, and future loops should only reopen this phase if hardware onboarding exposes a real contract or parity gap.

## Review

### 1. Execution contract stability
- `Engine::execute_mvcc_query(&MvccReadQuery)` remains the single engine-facing MVCC read surface.
- Supported source/filter/order/projection semantics are documented in:
  - `README.md`
  - `docs/interfaces/execution-interfaces.md`
  - `docs/roadmap/no-nvidia-bootstrap-plan.md`
- Engine-facing regression coverage already exercises the documented surface directly through `execute_mvcc_query(...)`, including workload-fixture replay and provenance-aware composition.
- Remaining work is explicitly about CUDA attachment and parity, not another engine-facing result-contract rewrite.

### 2. Parity / fallback truth surface stability
- Every MVCC read still reports:
  - planned target = `gpu(default_gpu_id)`
  - executed target = `cpu`
  - fallback reason = `GpuMvccReadParityGap` (`GPU-123`)
- Deterministic fixture regressions now assert that device/fallback contract directly for all current workload fixtures:
  - `execute_mvcc_query_replays_deterministic_workload_fixture_for_point_lookup`
  - `execute_mvcc_query_replays_deterministic_full_scan_workload_fixture`
  - `execute_mvcc_query_replays_deterministic_source_composition_workload_fixture`
- The source-composition fixture also proves `status_snapshot()` rolls those parity fallbacks up consistently.

### 3. Deterministic replay coverage
- Existing fixtures are sufficient for the first CUDA slice:
  - `tests/fixtures/mvcc-read-workload.txt` covers point lookup + history replay.
  - `tests/fixtures/mvcc-full-scan-workload.txt` covers full scan + simple-filter replay across historical and current snapshots.
  - `tests/fixtures/mvcc-source-composition-workload.txt` covers source composition and join-adjacent behavior.
- Together they provide stable CPU truth for scan/visibility/filter/point-lookup parity checks while keeping fallback accounting observable.

### 4. No-rewrite check
- The first CUDA slice can attach behind the existing backend boundary and preserve:
  - `MvccReadQuery`
  - `MvccReadResult`
  - `MvccReadRow { source_key, key, value }`
  - `status_snapshot()` fallback / replication truth surfaces
- That backend boundary is now explicit in code: `Engine::execute_mvcc_query()` routes through `CpuMvccExecutionBackend` today, and a backend-swap regression proves the result contract stays stable when the execution path changes.
- A follow-on first-slice parity regression now proves the intended initial CUDA envelope can execute through an alternate backend for `FullScan` / `KeyLookup` + simple-filter shapes while unsupported composition still falls back through the CPU reference backend with the same tracked `GpuMvccReadParityGap` reason.
- Unsupported CUDA shapes can continue to route through the explicit CPU fallback path without weakening the GPU-first contract.

### 5. Validation gate
- `cargo fmt --all`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all --all-features`

Result: all green on 2026-05-04.

## Next step when NVIDIA hardware is available
1. Add CUDA build targets / reproducible GPU-capable environment.
2. Implement `CudaBackend` behind the existing backend boundary.
3. Port only the first parity-checkable slice:
   - scan
   - snapshot visibility filtering
   - simple filter predicates
   - point lookup / key lookup
4. Run CPU-vs-GPU parity against the deterministic point-lookup, full-scan/simple-filter, and source-composition fixtures before widening coverage.
