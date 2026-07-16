# Independent adversarial performance audit — proposed write-path ADR

**Review date:** 2026-07-15

**Scope:** proposed canonical append/tombstone design, pinned source commit `f701d8b6`, live intent-lane/FUA
conveyor, STRATA placement, index maintenance, and recovery/migration service bounds.

**Reviewer independence:** performed by a separate review agent that did not author or edit the proposal.

**Verdict:** **REVISE before acceptance.**

This is review evidence for **R3-001**, not an ADR or work ledger. Open evidence and sequencing remain in
[`PLAN.md`](../PLAN.md). The revised proposal incorporates every finding below; incorporation does not turn the
verdict into acceptance.

## Findings and adopted dispositions

### CRITICAL — the selected physical representation lacks its performance gate

At audit time the charter required more than 100,000 sustained TPS, a 400,000 TPS burst, and a uniform simple-OLTP
p99 below 1 ms. The accepted 2026-07-15 target refinement now keeps 1 ms for R1 reads, uses 1.5 ms for W1 single
mutations, 3 ms for T8, and 6 ms for T32 ([`CHARTER.md`](../CHARTER.md)); this historical finding and its adopted
bounded-controller disposition remain valid. The later throughput clarification binds 100,000/400,000 aggregate
committed TPS to immutable `oltp-benchmark-workload-v1.md`, including the deterministic 60/25/10/5 R1/W1/T8/T32
system mix, exact route/data/access envelopes, sustained window, and named peak cohorts; standalone class TPS is
diagnostic.
The pre-review evidence had focused correctness tests and a static two-int4
64-byte/version estimate, but no update-heavy synchronous-commit TPS/tail distribution. It therefore cannot claim
that append/tombstone meets the binding latency, throughput, footprint, or write-amplification requirements.

**Adopted:** the proposal now makes physical selection conditional on an internal Candidate-A SLO matrix, removes
the evidence-complete claim, defines the dimensions and sabotage cases, and reopens physical encoding only if a
binding metric fails. The tuned PostgreSQL comparison remains separately owned by **BENCH-001**.

### HIGH — latest lookup and index service are unbounded under churn

The prior proposal allowed historical candidates to accumulate and treated a GPU scan as the generic answer when
an optional index could not fit. A hot key plus a pinned snapshot could therefore lengthen fast-route work without
bound. The live path already drops/rebuilds a PK index after a dead-twin collision in
[`lane_apply.rs`](../../crates/engine/src/engine_dml_concurrent/lane_apply.rs), and current vacuum uses a churn
threshold rather than a latest-route service bound.

**Adopted:** prepared fast routes now require capacity-reserved latest-head indexes physically separated from
prunable history; load, probe, candidate, and fanout bounds are mandatory. Scan remains a GPU-native slow-class
path, not a silent prepared-route fallback. Maintenance or pre-WAL refusal occurs before a bound is crossed.

### HIGH — existing conveyor automation can violate the SLO

The live conveyor has useful population-scaled wave formation, fence-slot-driven subframing, and hysteretic active-
lane resizing. They do not form a tail-safe contract:

- the grouping deadline can reach 2,000 microseconds in
  [`lane.rs`](../../crates/engine/src/engine_dml_concurrent/lane.rs), already above both the former uniform target and
  the current complete W1 p99 target;
- validation/apply coalescers can drain an unbounded matching/pending queue; and
- active-lane resize performs a global drain whose source records 0.6–1.2 second spikes and permits a five-second
  drain window.

**Adopted:** every stage is count/byte bounded; waves and coalescers ship on size, predicted service, or oldest-item
deadline; age-aware fairness protects sparse lanes and prepared traffic; deadlines derive from the residual end-to-
end budget. Automatic batching and credit admission are mandatory. Drain-barrier lane resizing is explicitly
excluded until a barrier-free or p99.9-bounded replacement exists.

### HIGH — cold or repair work could stall the first-gap publication cut

The joined durable/applied cut correctly stops at its first gap. That makes post-claim NVMe staging, index rebuild,
rollover allocation, compaction, or repair a global head-of-line risk.

**Adopted:** direct WAL-first work requires resident source/index state and a bounded reserved apply path. All cold
staging and unbounded repair completes before claim; slow work revalidates conflict/generation tokens immediately
before sequencing. Prepared resident traffic owns an explicit credit reservation.

### HIGH — GC, index maintenance, and STRATA pressure lacked an automatic controller

Hard budgets and the reclaim/compact/demote/reject order were correct, but trigger combination alone permits both a
hard-quota latency cliff and synchronous maintenance stalls.

**Adopted:** automatic maintenance is mandatory. The proposal defines the required signals, soft/high/hard/lower-
resume watermarks, hysteresis, bounded time/byte quanta, foreground-age yield, replacement/scratch reservation, and
benefit-per-cost candidate ranking. Snapshot cancellation remains explicit-only.

### HIGH — acknowledged async population is not resource pressure

Live async tickets can release at the applied cut while durability and visibility still lag. Using only client
`outstanding` can therefore make hidden state appear idle.

**Adopted:** the design separately accounts for client, prepared, sequenced, applied-not-durable, durable-not-
applied, and unpublished bytes/intents. Async release keeps sequence/resource credits charged until publication or
recovery disposal. Async mode is never an automatic SLO response.

**Subsequent ACID refinement:** the pre-publication response is an explicit non-commit asynchronous ticket, not SQL
commit success, and it blocks dependent session work until publication/failure. The credit and telemetry finding
still applies unchanged; see [`write-path-adr-acid-audit.md`](write-path-adr-acid-audit.md).

**Subsequent consistency refinement:** the internal transaction/conveyor record owns validation floors, snapshot/
status pins, and resource credits until terminal publication; the client ticket is only an observation handle.
This corrects any inference that `IntentTicket::Drop` safely ends queued-work ownership. The publication root is
also required to use bounded persistent structural sharing so atomic multi-table publication does not become
O(all tables). See [`write-path-adr-consistency-audit.md`](write-path-adr-consistency-audit.md).

### MEDIUM — full-image and WAL amplification lacked a fast-class bound

Appending an entire wide post-image for a narrow update can consume payload, WAL, replication, and index bandwidth
well beyond the two-int4 model.

**Adopted:** fast admission is bounded by post-image, WAL, varlen, maintained-index fanout, and predicted device
service. Wide work uses the measured slow class. A gate failure attributable to full-image amplification reopens
immutable column-group sharing or device-native physical deltas while preserving logical append/tombstone MVCC.

### MEDIUM — recovery and migration lacked time/space service bounds

Rebuilding every index before one publication and draining/converting the whole database had no capacity estimate
or bound against the five-minute recovery target.

**Adopted:** recovery and migration require byte-bounded STRATA streaming and durable/device/scratch preflight.
Only indexes needed by admitted routes block service. Migration reports time/space estimates and aborts a bounded
drain back to the legacy pointer. DUR-001's checkpoint cadence must enforce R3-001's replay-work/RTO ceiling.

## Required automated adaptation

The design did not already cover enough automation. The proposed target now requires:

1. deadline-and-size-aware wave and coalescer formation;
2. per-stage intent/byte credits and pre-WAL overload admission;
3. resident-fast versus cold/repair-slow classification before sequencing;
4. bounded latest-head degradation detection and automatic maintenance;
5. snapshot-aware GC, compaction, and STRATA demotion with watermarks/hysteresis; and
6. durable/apply/publication-lag feedback independent of client acknowledgement.

It does **not** require stop-the-world runtime lane-count resizing, automatic synchronous-commit changes, or
automatic switching between MVCC representations.

## Evidence required by this audit

The exact SLO matrix and sabotage conditions are normative in
[`write-path-adr-proposal.md`](write-path-adr-proposal.md). In particular, acceptance still needs hot-key churn,
cold-staging/index-rebuild delay, sparse-lane/global-skew and hysteresis stability, constrained long-snapshot
pressure, both stalled-FUA and delayed-GPU-apply cut imbalances, and disabled-maintenance sabotage. Until those
results exist, the independent verdict remains **REVISE**.

Subsequent integration review separated ADR design acceptance from implementation graduation: R3-001 still needs
the Candidate-A SLO/canonical-footprint decision matrix, bounded non-production evidence probes, and a final design
re-review. The canonical controller/maintenance sabotage matrix is repeated after acceptance before the new path can
become production authority. This removes circular dependence on implementation tasks without weakening the audit's
binding SLO requirement.

Subsequent R3-001 remediation preserves the failed current end-to-end matrix, measures the common
synchronous-durability envelope with the actual engine-facing frame-log rate plus a labeled same-physics percentile
harness, and supplies the bounded Candidate-A/B width/fanout/batch/footprint comparison in
[`write-path-adr-physical-selection.md`](write-path-adr-physical-selection.md). Compact append/tombstone wins that
physical comparison. The corrected bounded 13-family decision-policy sabotage in
[`write-path-adr-controller-injections.md`](write-path-adr-controller-injections.md) now covers the required
cold/index preclaim, full-envelope class admission, class-escalation/count/resource sabotage, all-percentile strict
qualification, sparse/global skew, lag-direction, pressure/hysteresis, credit, maintenance, starvation, and drain-
resize transitions. The strict end-to-end SLO and canonical controller/maintenance fault campaign remain
mandatory production graduation; a durability
floor that consumes the budget now makes the advertised low-latency profile explicitly unqualified rather than
causing an async or semantic switch. This historical audit verdict remains **REVISE**; only the later fresh final
acceptance review may judge the remediated packet.

Final review v3 then found that the first bounded model's wave-cap case had already expired its age deadline and
that its pressure state omitted soft and Rejecting-state hysteresis. The corrected model independently exercises
byte- and service-triggered shipment before the deadline, rejects an individually oversized item before claim, and
implements soft/high/hard/lower recovery from Maintaining, Throttling, and Rejecting. Those corrections still
required a newly frozen independent review; packet v4 subsequently returned **ACCEPT** with no pre-acceptance
blocker. This historical audit remains evidence rather than the user's acceptance decision.

## Design elements retained without objection

- stable logical row, version, and physical-coordinate separation;
- one visibility law across resident and cold placement;
- generation-atomic append/tombstone publication;
- overlapped durability and hidden device apply joined at publication;
- worst-case capacity reservation before sequence claim;
- first-gap fail-closed cuts and direct recovery semantics;
- explicit async semantics outside RPO-0 evidence;
- no silent snapshot cancellation; and
- rejection of the measured blocking mega-fuse.
