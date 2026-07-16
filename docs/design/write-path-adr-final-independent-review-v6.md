# Focused R3-001 independent review v6 — target remediation

**Date:** 2026-07-16

**Packet:** [`write-path-adr-review-packet-v6.md`](write-path-adr-review-packet-v6.md)

**Verdict:** **REVISE**

All 18 frozen hashes matched. Formatting, strict Clippy, controller execution, and diff checks passed. The reviewer
confirmed complete-envelope class admission, derived wave budgeting, all nine strict percentile boundaries, the
unchanged Candidate-A W1 failure, and no ACID, durability, publication, recovery, STRATA, RPO/RTO, or physical-
selection semantic change.

## Blocking findings

1. **The reference workload was not yet deterministic.** The 200-transaction class mix and 650-operation arithmetic
   were exact, but T8/T32 SQL, route manifests, numeric resource bounds, data shape, and access distribution were
   still left for BENCH-001 to choose later. Calling those routes frozen before the manifest existed left a material
   workload-selection escape.
2. **Peak accounting was ambiguous.** A one-second 400,000-arrival cohort could mean eventual cohort completions or
   completions timestamped inside the same second. The contract also did not fix the ten burst identifiers/order, so
   it did not fully prevent omitting a failed burst.
3. **One active consumer retained a conflicting standalone throughput label.**
   `gpu_mixed_read_write_gate.rs` called its >100,000 read-QPS threshold a mixed OLTP SLO even though the revised
   charter reserves >100,000 TPS for the canonical aggregate transaction mix. That active file was absent from v6.

The percentile-margin composition was not a separate blocker because the binding gate directly measures end-to-end
open-loop latency. Component arithmetic may only act as a conservative scheduler/profile input and cannot replace
that measurement.

## Required correction boundary

The replacement packet must freeze the schema, data cardinality, seed/access distribution, exact R1/W1/T8/T32 SQL
and order, numeric resource envelopes, sustained offered/measurement protocol, and named burst schedule before any
benchmark run. It must distinguish cohort TPS from wall-clock completion throughput and require all ten bursts. The
active mixed read/write gate must label >100,000 read QPS as a local non-vacuity floor rather than charter system TPS.
This verdict does not reopen the accepted v4 physical/ACID/durability design.
