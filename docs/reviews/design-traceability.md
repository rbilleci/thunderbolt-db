# Design Traceability Matrix

Date: 2026-03-15  
Source baseline: `DESIGN.md`

This matrix maps major design intents from the original design document to implementation-facing docs created in this repository.

## Traceability Table

| DESIGN.md area | Intent summary | Primary mapped docs | Status |
|---|---|---|---|
| 1. Introduction & requirements | PG-compatible, ACID, hybrid CPU/GPU, multi-GPU direction | `docs/architecture/01-system-invariants.md`, `docs/roadmap/v0-v1.md` | Covered |
| 1.1 Performance targets | Latency/TPS/SLO framing | `docs/architecture/06-observability-slos.md` | Covered (framework; exact thresholds iterated) |
| 1.2 Technology choices | Rust + CUDA path, parser strategy | `docs/interfaces/*`, `docs/architecture/04-execution-model-cpu-gpu.md` | Covered |
| 1.3 Layered architecture | Clear subsystem boundaries | `docs/architecture/01-system-invariants.md`, `docs/interfaces/*` | Covered |
| 2. Protocol & SQL compatibility | PG wire semantics and compatibility discipline | `docs/architecture/01-system-invariants.md`, `docs/roadmap/v0-v1.md` | Covered (phased depth) |
| 2.2 Session management | Connection/session model and limits | `docs/architecture/06-observability-slos.md`, `docs/architecture/09-session-management-and-admission.md` | Covered (bootstrap limits; adaptive controls deferred) |
| 2.3 SQL dialect/features | Broad SQL + transactional semantics | `docs/roadmap/v0-v1.md`, `docs/interfaces/transaction-interfaces.md` | Covered (phased) |
| 2.4 System catalog compatibility | Tooling compatibility posture | `docs/architecture/01-system-invariants.md` | Covered (policy-level) |
| 2.5 Error taxonomy | Structured failure propagation | `docs/architecture/02-commit-path.md`, `docs/architecture/06-observability-slos.md`, `docs/interfaces/error-interfaces.md` | Covered |
| 3. Storage layout & buffering | Hybrid storage and memory management | `docs/architecture/05-storage-and-recovery.md`, `docs/architecture/04-execution-model-cpu-gpu.md` | Covered (high-level) |
| 3.7 WAL/checkpoints | Durability model and recovery boundaries | `docs/architecture/02-commit-path.md`, `docs/architecture/05-storage-and-recovery.md` | Covered |
| 4. GPU integration | Batched GPU execution, error handling, abstractions | `docs/architecture/04-execution-model-cpu-gpu.md`, `docs/interfaces/execution-interfaces.md` | Covered |
| 5. Hybrid + multi-GPU scaling | Routing and replication-aware scaling | `docs/architecture/03-replication-model.md`, `docs/roadmap/v0-v1.md` | Covered (phased) |
| 5.1 Cost model | CPU/GPU routing economics | `docs/GPU_GUARDRAILS.md`, `docs/architecture/04-execution-model-cpu-gpu.md` | Covered (policy-level) |
| 6. Transaction management | MVCC/isolation and visibility rules | `docs/interfaces/transaction-interfaces.md`, `docs/architecture/01-system-invariants.md` | Covered |
| 7. Fault tolerance/replication/recovery | HA, replication, failover, recovery | `docs/architecture/03-replication-model.md`, `docs/architecture/08-backup-pitr-dr.md` | Covered |
| 8. Security & compliance | Banking-grade security/control posture | `docs/architecture/07-security-and-compliance.md` | Covered |
| 9. Observability | Metrics/logging/admin visibility | `docs/architecture/06-observability-slos.md` | Covered |
| 10. Testing strategy | Correctness, parity, chaos, regressions | `docs/GPU_GUARDRAILS.md`, `docs/reviews/design-alignment-check.md`, future testing plan doc | Partially covered |
| 11. Implementation plan | Phased delivery and risk reduction | `docs/roadmap/v0-v1.md`, ADR set | Covered |

## Notes on Intentional Compression

- The current docs intentionally compress execution scope while preserving architectural intent.
- Areas marked *Partially covered* are primarily execution-detail depth gaps, not directional contradictions.

## Next traceability hardening steps

1. ✅ Added `docs/compatibility/matrix.md` for explicit PostgreSQL feature status by phase.
2. ✅ Added `docs/testing/parity-and-jepsen-plan.md` for deterministic replay and consistency validation criteria.
3. ✅ Added `docs/operations/runbooks.md` to link DR/security controls to exact procedures.
4. ✅ Expanded 2.2 session-management traceability with concrete runtime admission-state contracts and signal-to-action mapping in `docs/architecture/09-session-management-and-admission.md`.
5. ✅ Added `docs/interfaces/error-interfaces.md` with crate-level error contracts, composition boundaries, and operator response mapping.
