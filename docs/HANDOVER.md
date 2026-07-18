# HANDOVER — Resume Baton

This file records only the current boundary and where the next agent resumes. `PLAN.md` owns all open work;
`STATUS.md` owns completed evidence. Do not turn this file into another backlog.

## Current boundary

- **R3-001 through R3-005 and DUR-002 are complete.** R3-004 removed normal host write/commit/MVCC tuple authority,
  `CachedShardPkIndex`, host DML/constraint probes, and their fallback dispatch. Supported writes, constraints,
  transactions, COPY, and replay publish device generations or fail-stop before acknowledgement.
- Explicit DDL/recovery/import/VACUUM reverse-gather and the bounded hot-to-cold representation transition remain
  isolated under **RETIRE-002**; neither is a host relational fallback or result path. Generic CUDA result
  post-processing remains **RETIRE-003**.
- The closeout HAZARD run found and fixed a lane strict-ack cut race: local and global visibility boundaries now
  derive from one captured durable/applied cut. `STATUS.md` owns the complete implementation and gate evidence.
- **RETIRE-001 is the sole NOW task.** Replace test-only CPU relational oracles/finalization with GPU-native or
  closed-form specification oracles before deleting each source boundary.
- **STRUCT-001 is closed.** The only production size exception is the registered 2,421-line `engine_expr.rs`.
  Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- The R3-004 closeout tree is the accepted baseline. High-value authority boundaries are
  `engine_commit.rs`/`engine_commit_residency.rs`, `engine_write_apply.rs`,
  `engine_dml_concurrent/{wave,lane,lane_apply}.rs`, `engine_transaction_commit.rs`,
  `engine_retained_read{,/template.rs}`, and `engine_residency/maintenance.rs` (RETIRE-002 repair only).
- `PLAN.md` is reconciled to one active path and `STATUS.md` records completed R3 evidence. Preserve that ownership
  split: only PLAN may own open work; HANDOVER remains a short pointer.

## Resume here

1. Read the required project documents, then start **RETIRE-001** with one test-oracle/finalization boundary.
2. Preserve every semantic fixture with a non-vacuous GPU or closed-form specification oracle before deleting its
   CPU implementation. Production behavior must remain unchanged.
3. Keep **RETIRE-002** repair and **RETIRE-003** result-path work outside this slice unless PLAN explicitly promotes
   them. Do not restore any host write/index/probe authority removed by R3-004.
4. Use the full applicable actual-GPU, NULL, and HAZARD gates. Run the canonical report card for any read-kernel,
   residency-layout, or result-path change.

## Last green evidence — 2026-07-17

- Engine library: **1,016/1,016** with ignored actual-GPU tests included; facade library: **47/47**, with its
  serialized concurrency integration suite also **14/14**.
- Nine high-risk families pass **27 sequential + 18 concurrent** HAZARD invocations with no CUDA 700/716/717.
- Workspace all-target/all-feature check and strict Clippy pass; formatting, diff whitespace, source-size, and
  independent host-authority/reference audits are clean.
- The canonical report card completed both layers/cache regimes. Authoritative-shard batched point reads reached
  **89.6M/s at p50 605us** in-L2 and **3.29M/s at p50 19.784ms** out-of-L2; raw roofline and representation context
  are in `STATUS.md`, and **PERF-001** owns the measured gap to the retired unified-snapshot card.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and use workspace-local `target/tmp` rather
  than `/tmp` for large generated artifacts.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
