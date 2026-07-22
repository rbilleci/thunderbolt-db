# HANDOVER — Resume Baton

This file is only the short resume pointer. [`PLAN.md`](PLAN.md) owns all open, deferred, blocked, and sequenced
work; [`STATUS.md`](STATUS.md) owns accepted evidence and current facts; the
[`PRODUCT-001 inventory`](design/product-001-serving-write-inventory.md) owns the factual source/consumer map.

## Current boundary

- **PRODUCT-001** has an independently accepted PostgreSQL 16 pg_dump/pg_restore and pg_dumpall-globals slice on
  `gpu-db-engine-server`, in addition to SQLx/simple query, COPY, TLS/SCRAM, cancellation, prepared/portal/session
  compatibility, and psql/GPU-catalog/R2DBC. All eight application-driver gates remain canonical; fresh architecture,
  semantics, and evidence panels accepted the exact frozen candidate with no blockers. No later PRODUCT-001 boundary
  was started in this slice.
- All 18 plain/archive/parallel/clean/insert/split/metadata/privilege dump cases and pg_dumpall role-login/
  tablespace/comment/ACL restore pass. Exact quote-aware program recognition, pinned catalog/sequence snapshots,
  complete candidate relations plus typed GPU filter/join/projection/order plans and a block-reduced device COUNT,
  transaction/failed-state controls, exact source/restored sequence ACL equivalence, isolated default-ACL
  inheritance, and safe child/port ownership pass focused gates. The accepted immutable review
  target is base `744d2e1113f403afff1e88bc175f11f6cda2dc7e`, code/test index tree
  `91efd8ba72c666e1b51881257930f778a8c71d1d`, and cached binary-diff SHA-256
  `5671761b1fd4a62eab7457f9a8e731a3b807c3c2f58c6d0477bd826c5c8a96d8` across 62 code/test paths.
- Current evidence is SQL **57**, engine **525**, facade **75** plus concurrency **13**, server **76**, protocol
  **71 + 127**, both dump harnesses, psql 04/06/07, the eight-driver aggregate, workspace check, strict Clippy, and
  static gates. The clean isolated full final-repair card measures **230.154M/s at p50 156us** in-L2 and
  **198.870M/s at p50 201us** out-of-L2 after a **2,129.4s + 0.0s** build/residency phase, **+0.3%/-1.0%** versus
  the prior accepted **229.397M/s/200.834M/s** baseline. The apparent **201.959M/s/174.844M/s** regression came from
  a shared target containing a GCC-13.3-built AWS-LC archive on the GCC 15.2 host, not from the point-read source;
  two clean exact-source builds were byte-identical and restored 227–230M/s targeted throughput. The canonical card
  now builds in a fresh isolated target, records compiler/binary/native-archive identities, and invokes the exact
  hashed binaries directly. No facade, WAL/sequence/commit/publication, or CPU relational owner was added.
- The remaining legacy listener/P8 adapters and consumers, broader transactional/recovery compatibility,
  multiple transactional catalog commands, typed table-reset/rewrite-fence coverage, and PLAN-owned **2,012**-line
  `engine_mutation_admission.rs` plus **2,083**-line `engine_dml_concurrent.rs` are the current PRODUCT-001 facts.
- The transaction-private catalog-generation proof is independently accepted at base
  `ae222b5809bca620bd2e483e80cd4c8a3ed7945c`, code/test tree
  `7a41b1a86e277b612110d0ca0def609b9ad14e2a`, and cached binary-diff SHA-256
  `f3db8416f31024edfda84d9d0ea28c4f060c75129782ef270e5f63d333ad2bae`. Unrelated DML/KV publication no longer
  aborts a staged CREATE when every catalog field and allocator high-water is unchanged; real catalog/allocator
  drift remains pre-WAL `40001`. The GPU NULL/rekey/recovery proof passes 3 sequential plus 2 concurrent rounds.

## Resume here

Resume **PRODUCT-001** only at the PLAN current-focus boundary: replace active-transaction `TRUNCATE ... CONTINUE
IDENTITY` row-delete emulation with the typed table-reset, ordered overlay, WAL/replay, and non-MVCC rewrite-fence
contract. After that slice is independently accepted, continue the PLAN-ordered multiple-DDL, SQLSTATE/type-codec,
named-client, and mixed-recovery proofs. Only then re-inventory and migrate or disposition the remaining legacy
psql/preflight/benchmark consumers and delete the legacy listener plus independently callable P8 protocol adapters
in the same frozen, independently audited slice. Follow the complete sequence and deletion gates only from
[`PLAN.md`](PLAN.md).
