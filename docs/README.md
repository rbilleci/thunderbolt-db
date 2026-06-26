# Documentation Guide

This project uses a layered documentation model so implementation can proceed quickly without losing architectural rigor.

> **Start here for current work:** [`docs/PLAN.md`](PLAN.md) — the single charter-based, STRATA-first **forward
> plan** (what to build next, in order). The reading order below is the architectural baseline; `PLAN.md` owns
> sequencing and supersedes the old session handovers (removed) and the roadmap docs' ordering.

## Recommended Reading Order

1. **Design baseline**
   - `DESIGN.md`
   - `docs/architecture/00-gpu-native-principles.md`

2. **Non-negotiable constraints**
   - `docs/architecture/01-system-invariants.md`

3. **Core execution and durability path**
   - `docs/architecture/02-commit-path.md`
   - `docs/architecture/05-storage-and-recovery.md`

4. **Scalability and replication model**
   - `docs/architecture/03-replication-model.md`

5. **CPU/GPU execution model**
   - `docs/architecture/04-execution-model-cpu-gpu.md`

6. **Operational expectations**
   - `docs/architecture/06-observability-slos.md`

7. **Security and resilience operations**
   - `docs/architecture/07-security-and-compliance.md`
   - `docs/architecture/08-backup-pitr-dr.md`

8. **Session management and admission control**
   - `docs/architecture/09-session-management-and-admission.md`

9. **P8 GPU-optimized storage design**
   - `docs/architecture/10-p8-gpu-optimized-storage-engine.md`

10. **High-throughput query runtime**
   - `docs/architecture/11-high-throughput-query-runtime.md`

11. **ACID, isolation, and GPU memory**
   - `docs/architecture/12-acid-isolation-and-gpu-memory.md`

12. **Interfaces and implementation contracts**
   - `docs/interfaces/replication-interfaces.md`
   - `docs/interfaces/execution-interfaces.md`
   - `docs/interfaces/transaction-interfaces.md`
   - `docs/interfaces/error-interfaces.md`

13. **Research journal and candidate techniques** (concluded architecture search — reference only)
   - `docs/research/gpu-db-paper-candidates.md`
   - `docs/research/gpu-db-literature-journal.md`
   - `docs/research/architecture-compatibility.md`
   - `docs/research/architecture-compatibility/paper-mechanism-coverage.md`
   - `docs/research/architecture-compatibility/benchmark-backlog.md`
   - `docs/research/end-to-end-architecture-design-space.md`
   - `docs/research/end-to-end-architecture-candidates.md`
   - `docs/research/end-to-end-architecture-final-dossier.md`
   - `docs/research/end-to-end-architecture-diagrams.html`
   - `docs/research/end-to-end-architecture-diagrams.png`

14. **Decision history (ADRs)**
   - `docs/adr/README.md`
   - `docs/adr/ADR-001-log-boundary-is-wal.md`
   - `docs/adr/ADR-002-deterministic-batch-ordering.md`
   - `docs/adr/ADR-003-cpu-fallback-policy.md`
   - `docs/adr/ADR-004-replicator-interface.md`
   - `docs/adr/ADR-005-snapshot-install-snapshot-strategy.md`

15. **Delivery scope and sequencing**
   - `docs/PLAN.md` — the unified forward plan (authoritative for order)
   - `docs/roadmap/prototype-to-production-plan.md` — P0–P8 milestone detail (sequencing superseded by PLAN)
   - `docs/roadmap/gpu-native-oltp-roadmap.md` — OLTP route classes (sequencing superseded by PLAN)
   - `docs/roadmap/v0-v1.md` — early v0→v1 baseline (superseded; requirements reference)
   - `docs/roadmap/implementation-log.md` — history
   - `docs/archive/` — archived dead-premise plans (no-nvidia-bootstrap, no-gpu-bootstrap-closeout-review)

16. **Compatibility and validation gates**
   - `docs/compatibility/matrix.md`
   - `docs/testing/parity-and-jepsen-plan.md`

17. **Operations runbooks**
   - `docs/operations/runbooks.md`

18. **Process guardrails**
   - `docs/GPU_GUARDRAILS.md`
   - `.github/pull_request_template.md`

## How to Use This Set

- Use `DESIGN.md` for full-system intent and long-range targets.
- Use architecture docs for implementation constraints and sequencing.
- Use interface docs while coding.
- Use research docs to convert papers into benchmark candidates before
  changing the runtime or storage design.
- Use ADRs when making changes that are expensive to reverse.
- Use guardrails and PR checklist to prevent CPU-first drift.
