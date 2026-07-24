# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** has a verified transactional-sequence candidate. `CREATE SEQUENCE`,
  `ALTER SEQUENCE ... RESTART`, rename, ordered multi-target drop, and owned-sequence
  `TRUNCATE ... RESTART IDENTITY` share the existing private `TransactionOperation` stream and sole
  WAL/publication owner. Stable OIDs bind lifecycle, dependent defaults, DML advances, restart/reset barriers, and
  recovery; additive opcodes 18/19 preserve opcode 4–17 bytes. Materialized views remain pre-effect refusal, and the
  legacy host-backed server refuses sequence restart with `0A000`.
- Ordinary/static gates are green: engine ordinary is **630/630** with **602** ignored. The complete
  include-ignored engine sweep passes **1,229/1,232** and retains only
  the three explicitly PLAN-owned base failures. Sequence NULL/lifecycle plus exact-budget publication each pass
  three serial and two simultaneous HAZARD executions without CUDA 700/716/717; reset-before-rename and
  generated-creator binding pass the same matrix. Non-vacuous restart sabotage fails `(24,false)` versus
  `(23,false)`; stale reset dependencies/root proof and both generated-OID swap forms also fail before their
  restored candidates and recovery pass. The repaired quick screen is flat:
  out-of-L2 roofline/grouped **1440.7 GB/s/1675.9 M-elem/s** and in-L2 production point reads
  **262.858M/s, p50 118us** versus the accepted **264.394M/s, p50 118us** baseline.
- The first exact-tree audit rejected legacy replay drift, generated-SERIAL rename binding by final name, and
  stable-value-only opcode-18 ownership. Each regression failed before repair and now passes: historical raw SQL
  retains acknowledged defaults, generated sequences close by captured stable OID across rename, and opcodes 18/19
  require a lifecycle/reset owner. The audit follow-up replaced a vacuous V2 post-boundary fixture with a genuine
  V1 canonical record; complete/split rename/drop dependency coverage passes and current-policy sabotage fails. Two
  later findings are repaired: a reset staged before owned-sequence rename now rebinds only its final output proof
  under unchanged stable OID/dependency closure while preserving its reset ordinal, and generated SERIAL output is
  cross-linked to each CREATE ordinal at encode/decode/apply plus creator-local stepwise replay.
  Exact-tree implementation re-audit returned **ACCEPT**. The one canonical full A/B/C report card also completed
  with final marker `report_card_execution_status=complete mode=full sections=A,B,C canonical=true`: Layer 1 is
  **1319.9/1442.8 GB/s** in/out of L2 with **1674.2 M-elem/s** grouped, and Layer 2 is
  **260.853M/s, p50 117us** in-L2 plus **236.360M/s, p50 140us** out-of-L2 after the 48M-row fixture built in
  **2243.6s**. Same-auditor post-card provenance/performance verification returned **ACCEPT** with no finding and
  no rerun required on docs-closeout tree `a0d6a83a…` / diff `9c79c501…`.
  Bounded roots include `wal_binary.rs` at **1,959**, `engine_transaction_catalog.rs` at **1,932**, and
  `engine_durability.rs` at **1,962**; reset rebind is a **155**-line private owner.
  Existing PLAN-owned source outliers are `engine_dml_concurrent.rs` at **2,175**,
  `engine_commit.rs` at **2,244**, and `engine_mutation_admission.rs` at **2,010**; none has an exception.

## Resume here

After this candidate lands, resume **PRODUCT-001** at the PLAN-owned ordinary published-sequence boundary:
`nextval`, sequence-backed defaults, and both `setval` forms on an unchanged published identity need one typed,
exactly-once `SequenceValueTransition` that survives enclosing user rollback and preserves stable-identity retry,
recovery, and `currval` semantics. Do not fold private CREATE/RESTART children into that transition. Materialized
views remain a pre-effect refusal. Then close the PLAN-ordered SQLSTATE/type-codec, named-client, and mixed-recovery
proofs before the legacy/P8 compatibility-and-deletion slice.
