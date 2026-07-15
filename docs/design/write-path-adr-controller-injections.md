# R3-001 bounded adaptation and pressure injections

**Date:** 2026-07-15  
**Artifact:** [`write_path_adaptation_injections.rs`](../../crates/engine/examples/write_path_adaptation_injections.rs)  
**Disposition:** **PASS — 12/12 bounded decision families**

This build-only executable closes the design-selection evidence gap for the proposed write controller. It is a
deterministic state/pressure model, not the production controller, a product benchmark, or authority to serve or
mutate durable state. Its purpose is to inject the exact conditions named by the ADR and prove that each policy has
one bounded, non-semantic-changing response before implementation begins under R3-002/003 and RETIRE-002.

## Injected matrix

| Injection | Bound/action asserted | Result |
|---|---|---|
| Sparse fast lane beside a young high-population lane and an older slow-class item | The W1 example reserves 700 us of its 1,500-us p99 for downstream work; the sparse item ships as a one-intent partial wave exactly at its 800-us residual oldest-age budget. Global population and slow-class age cannot hold it. | PASS |
| Wave byte/predicted-service overflow | Independent byte-only and service-only two-item candidates ship one-item partial waves before the W1 example's 800-us residual age deadline when the next item would cross 1,000 bytes or 500 us. A single item that exceeds either cap is classified and rejected before claim rather than waiting forever. | PASS |
| Cold staging incomplete | Cold/repair work remains `HoldColdBeforeClaim`; it cannot acquire a sequence or WAL position. A separately prepared resident-fast item remains claimable. | PASS |
| Latest-head/index unavailable | The fast route becomes explicitly unready before claim; no scan or host lookup is selected. Once ready, the same resident-fast class may claim. | PASS |
| Apply leads durability by the hard gap | Visibility stays at `min(durable_next, applied_next)`, durability gets priority, and new work is throttled before WAL. | PASS |
| Durability leads apply by the hard gap | Visibility stays at the applied prefix, apply gets priority, and new work is throttled before WAL. | PASS |
| Durability floor exceeds residual p99 budget | The measured 1,723-us fence plus 200-us margin is unqualified for W1's 1,500-us p99 but qualified for T8's 3,000-us p99. Exact equality with T32's strict 6,000-us target fails while one microsecond below qualifies. No outcome contains an asynchronous-mode action. | PASS |
| Held snapshot at resident high watermark | Reclaim remains false, device-format STRATA demotion becomes true, and admission throttles before WAL while cold quota fits. | PASS |
| Pressure hysteresis, cold-quota exhaustion, and disabled-maintenance sabotage | Resident and cold budgets each have soft/high/hard/lower thresholds: soft starts maintenance while admitting, high throttles, hard rejects, and lower resumes. `Maintaining` stays armed above either lower; both `Throttling` and `Rejecting` recover to throttling above lower and ordinary admission only at/below both lowers. Exhausted cold quota or disabled required maintenance rejects before WAL. | PASS |
| Intent/byte credits including internally retained work | A request that would cross either hard cap rejects before claim; an exact-bound request is admitted. Client-ticket release does not enter the decision. | PASS |
| Compaction old+new+scratch overlap and foreground age | Insufficient maximum-overlap capacity rejects before maintenance starts; sufficient capacity runs only a bounded quantum and yields at the foreground age ceiling. | PASS |
| Reclaim yield/service ranking under starvation | Yield/service ratio wins ordinarily; a candidate crossing the starvation-age threshold takes the next bounded quantum. | PASS |
| Proposed stop-the-world lane-count resize | The action vocabulary retains fixed lanes and refuses the global-drain transition. | PASS |

The executable prints 12 named family passes plus a final aggregate pass. The cold and index rows share one
preparation family, both lag-direction rows share one first-gap family, and the combined hysteresis/quota table row
prints two pressure families; with the three maintenance/resize rows this totals 12.

## Invariants demonstrated

- Every overload, unavailable-index, cold-stall, or pressure failure acts before the sequence/WAL boundary.
- `visible_next` never passes the lesser of durable/applied prefixes.
- Fast and slow classes do not share an unbounded coalescer or global-age decision.
- W1/T8/T32 synchronous qualification uses the admitted class's strict p99 target; a request cannot borrow a larger
  class budget without first satisfying that class's operation, mutation, and resource envelope.
- Oldest age, bytes, predicted service, intent credits, and byte credits are independent caps. Reaching a wave cap
  ships before the age deadline; an individually oversized item rejects before claim.
- Soft/high/hard/lower-resume actions are distinct for both resident and cold budgets, and prior rejecting/
  throttling state cannot bypass either lower-resume hysteresis boundary.
- Held snapshots are never canceled and their history is never reclaimed; STRATA demotion is the only modeled
  relief while cold quota fits.
- Required maintenance cannot be disabled and then replaced by silent scan, host execution, discarded history, or
  post-ack repair.
- The controller never changes MVCC representation, synchronous acknowledgement, RPO, or transaction outcome.

## Reproduction

```text
rustfmt --edition 2021 crates/engine/examples/write_path_adaptation_injections.rs
cargo clippy -p gpu_db_engine --example write_path_adaptation_injections -- -D warnings
cargo run --quiet -p gpu_db_engine --example write_path_adaptation_injections
```

The strict Clippy build and all assertions pass. Production implementation must repeat these scenarios with real
queues, device indexes, STRATA artifacts, maintenance, stage telemetry, and latency distributions. Disabled
maintenance and constrained-pressure sabotage remain mandatory production-graduation gates; this model closes only
the pre-acceptance controller decision, not those implementation results.
