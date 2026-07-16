# Focused R3-001 independent review v7 — workload remediation

> Archived acceptance-process verdict. Non-actionable; current work lives only in `docs/PLAN.md`.

**Date:** 2026-07-16

**Packet:** [`write-path-adr-review-packet-v7.md`](write-path-adr-review-packet-v7.md)

**Verdict:** **REVISE**

All 22 frozen hashes matched. Formatting, both strict Clippy commands, the 13-family controller executable, and
diff checks passed. The reviewer confirmed the exact 200-transaction/650-operation mix and pending-pool capacity;
the sustained completion-window and fixed `B01`–`B10` cohort accounting; complete-envelope admission and all nine
strict percentile boundaries; the unchanged Candidate-A W1 failure; and no ACID, durability, publication, recovery,
STRATA, RPO/RTO, or physical-selection semantic change.

## Blocking findings

1. **The sustained arrival process was not exact.** The manifest fixed 110,000 scheduled transactions/s but did not
   fix each arrival timestamp. Uniformly spaced, clumped, and Poisson inputs all met that wording while producing
   materially different queueing and latency.
2. **Generated transaction parameters were not fully executable.** The ledger ID's `operation_ordinal` did not
   define its base or scope; the W1 DELETE ordinal lacked an explicit domain/order; and T32 did not define its four
   account pairs, amount-output consumption, or exact amount assignment.

## Required correction boundary

The replacement must freeze a zero-based sustained-arrival timestamp formula and the zero-based ledger/DELETE
ordinal domains. It must also define the parameter-generator consumption order, T8/T32 debit-credit pairing, and
amount assignment without leaving BENCH-001 a contention, latency, value, or key-validity choice. This verdict does
not reopen the accepted v4 physical/ACID/durability design.
