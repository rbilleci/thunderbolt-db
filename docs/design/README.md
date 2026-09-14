# Design references

Documents here record accepted detailed designs, current decision inputs, and durable mechanism constraints. They
are **not plans** and do not own priority, status, or sequencing. The only executable ledger is
[`../PLAN.md`](../PLAN.md); implementation facts live in [`../STATUS.md`](../STATUS.md).

- [`write-path-adr-014.md`](write-path-adr-014.md) — accepted canonical GPU-native write/MVCC/transaction design.
- [`write-path-adr-014-compatibility.md`](write-path-adr-014-compatibility.md) — compatibility deviations and
  rule-by-rule implementation graduation ownership.
- [`write-path-adr-014-traces.md`](write-path-adr-014-traces.md) — accepted decision-level ACID and failure traces.
- [`write-path-adr-014-recovery-profile.md`](write-path-adr-014-recovery-profile.md) — five-minute standalone
  recovery capacity profile and refusal rules.
- [`oltp-benchmark-workload-v1.md`](oltp-benchmark-workload-v1.md) — immutable BENCH-001 workload and acceptance
  accounting.
- [`non-int4-index-design-inputs.md`](non-int4-index-design-inputs.md) — accepted index constraints and remaining
  point-lookup breadth under **READ-002**.

Dated proposals, superseded inputs, probes, packet manifests, and detailed reviews are available from Git history
and the `v0.1.0-alpha.1` tag rather than the current source tree.
