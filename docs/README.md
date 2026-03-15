# Documentation Guide

This project uses a layered documentation model so implementation can proceed quickly without losing architectural rigor.

## Recommended Reading Order

1. **Design baseline**
   - `DESIGN.md`

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

8. **Interfaces and implementation contracts**
   - `docs/interfaces/replication-interfaces.md`
   - `docs/interfaces/execution-interfaces.md`
   - `docs/interfaces/transaction-interfaces.md`

9. **Decision history (ADRs)**
   - `docs/adr/README.md`
   - `docs/adr/ADR-001-log-boundary-is-wal.md`
   - `docs/adr/ADR-002-deterministic-batch-ordering.md`
   - `docs/adr/ADR-003-cpu-fallback-policy.md`
   - `docs/adr/ADR-004-replicator-interface.md`
   - `docs/adr/ADR-005-snapshot-install-snapshot-strategy.md`

10. **Delivery scope and sequencing**
   - `docs/roadmap/v0-v1.md`
   - `docs/roadmap/no-nvidia-bootstrap-plan.md`

11. **Process guardrails**
   - `docs/GPU_GUARDRAILS.md`
   - `.github/pull_request_template.md`

## How to Use This Set

- Use `DESIGN.md` for full-system intent and long-range targets.
- Use architecture docs for implementation constraints and sequencing.
- Use interface docs while coding.
- Use ADRs when making changes that are expensive to reverse.
- Use guardrails and PR checklist to prevent CPU-first drift.
