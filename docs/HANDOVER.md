# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** has a verified ordinary published-sequence candidate. Existing published `nextval`, both
  `setval` forms, and omitted INSERT defaults use a separate typed `SequenceValueTransition` in the sole
  commit/WAL/status/publication order. Additive opcodes 20/21 bind stable OID, prior/new state, exact retry identity,
  and later user-envelope references while preserving earlier bytes. A transition survives user rollback;
  recovery and exact retry cannot consume it twice.
- Default references bind table/column/final-row identity, materialized value, and later update/delete disposition.
  Positive and negative route pins prevent ADD/DROP DEFAULT races from changing transition semantics. Facade
  `currval` is stable-OID session state with PostgreSQL operation-specific behavior. A shared checked ID allocator
  prevents facade, compatibility, parent-envelope, and transition aliases. Private CREATE/RESTART value children
  remain outside this ordinary transition and materialized views remain pre-effect refusal.
- Gates are green: engine **645/645** active, facade **83/83**, concurrency **13/13**, and the complete serialized
  GPU sweep **1,244/1,247** with only the same three PLAN-owned failures. The transactional NULL differential and
  published-default rollback/recovery test each pass three serial plus two simultaneous GPU runs. Workspace,
  strict Clippy, dependency, format/diff, and source-size disposition gates pass. New production owners are
  **1,346/575** lines; `engine_mutation_admission.rs` remains PLAN-owned at **2,087** lines, and the newly crossed
  **2,002**-line transaction-catalog mixed root now has its analyzed no-exception core-test extraction in PLAN.
  The final quick screen completed A/B at **1471.0/1428.5 GB/s** raw in/out of L2,
  **1673.2 M-elem/s** grouped, and **268.255M/s, p50 117us** in-L2 production point reads.
- The first audit withheld the full card and found the chained-transaction second allocator, corrupt
  reference-count preallocation, and missing direct sabotage. A second audit withheld it for post-terminal chain
  exhaustion/snapshot leakage and the missing catalog size disposition. All findings are repaired and fully
  re-gated; chained successors now register pre-effect and cancel on later failure, with engine/facade/GC/WAL retry
  sabotage. The third exact-tree audit returned **ACCEPT** on base `5d1fc1890544453767e2f7ba0941acbfb63c0e21`,
  staged tree `91ca526ee67a781d7889a23c479d65106292fffa`, and cached binary-diff SHA-256
  `00a84dd60586de5d06db50658c27cde66d3a821485c5a5652b5edcb4c309296c` across 41 paths without drift.
- The candidate's single canonical full A/B/C card completed with the exact canonical marker. Raw rooflines are
  **1391.7/1433.8 GB/s**, grouped is **1674.2 M-elem/s**, and production point reads are
  **271.032M/s, p50 116us** in-L2 and **249.569M/s, p50 139us** out-of-L2. The 48M-row fixture built in
  **2328.7s** with zero final-residency work; point throughput is **+3.9%/+5.6%** against the accepted card, and the
  isolated target was removed. Same-auditor post-card audit returned **ACCEPT** with no High or Medium finding on
  closeout tree `915254c7bb9c92f6a2b2067d10dd04e417e6d389` / cached binary-diff SHA-256
  `3e989cd4219edc409e51943563d43d0d7d693a977e483b9d53c56e5f69dcc1a2`; it matched full-card log
  SHA-256 `214fd92342f05c720d0fbac26172cf8eb96bce619f4d75d2cb5f6a6f01bef82b`, provenance, cleanup, and
  baseline deltas. The ordinary published-sequence slice is accepted without a card rerun.

## Resume here

Resume the PLAN-owned **PRODUCT-001** SQLSTATE/type-codec, named-client, and mixed-recovery proofs before the
legacy/P8 compatibility-and-deletion slice. The 2,002-line transaction-catalog core-test extraction remains
PRODUCT-001-owned before the parent closes.
