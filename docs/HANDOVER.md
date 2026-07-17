# HANDOVER — Resume Baton

This file records only the current boundary and where the next agent resumes. `PLAN.md` owns all open work;
`STATUS.md` owns completed evidence. Do not turn this file into another backlog.

## Current boundary

- **R3-001/R3-002/R3-003/R3-005 are complete.** The independent adversarial audit returned **ACCEPT** after every
  finding was adopted. Typed/nullable/compound authoritative writes, device-current unique history,
  transaction-private hot/cold generations, retained snapshots, atomic replay, and safe-horizon reclamation are
  accepted facts in `STATUS.md`.
- The final audit correction rechecks inbound FKs for class-authoritative parent DELETEs under `commit_mutex`
  immediately before WAL. Its deterministic race gate pauses after provisional validation, commits the child, and
  proves the definitive check rejects the parent without WAL or commit-path poisoning. The pinned-history gate
  proves compaction defers for an old generation and becomes eligible after retirement.
- **DUR-002 is complete.** The accepted ADR-014 envelope is typed, non-circular, checksummed, lineage-bound, and
  physically maps each FUA lane fragment to one checked global commit range. Durable status claims resolve retries
  and discarded orphans from stable authority; strict acknowledgements require durable-and-applied publication;
  working-catalog multi-entry DDL/DML applies atomically; recovery is repeatable and has one bounded fresh
  Engine/runtime retry for recognized CUDA context loss. `STATUS.md` owns the complete fault and gate evidence.
- **R3-004 is the sole NOW task.** Delete bootstrap/recovery host write/store authority as small independently
  gated slices under that one ID; do not add a temporary CPU authority or combine unrelated deletion boundaries.
- **READ-001** owns the previously observed broad facade shape-changing-DDL/read and elided-rehydration stress
  debt. It is not part of the DUR-002 acceptance path and must not displace the active correctness task.
- **STRUCT-001 is closed.** The only production size exception is the registered 2,423-line `engine_expr.rs`.
  Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- Local `main` contains canonical ADR commit `84cbab44`, accepted R3 integration commit `7e9e1568`, and qualification
  commit `397b8165`. The working tree carries the complete uncommitted DUR-002 implementation and evidence. The
  exact pre-integration accepted tree remains recoverable at `preserve/r3-audited-20260717` (`eb6f0319`). Nothing
  has been pushed; `origin/main` remains at `84cbab44`.
- High-value R3 entry points are execution `write_locate.rs`/`resident_index_build.rs`; engine
  `engine_dml_concurrent/{state,wave,lane,lane_apply}.rs`, `engine_dml_prepare/{device_tuple,unique_conflict}.rs`,
  `engine_transaction_{delta,commit}.rs`, `engine_streaming_exec/streaming_{dml_class,transaction_cow}.rs`, and
  `write_path.rs`. Read the full durability diff before deleting shared commit/recovery bootstrap code under
  R3-004.
- `PLAN.md` is reconciled to one active path; `STATUS.md` contains the accepted R3 evidence. Preserve that ownership
  split: only PLAN may own open work.

## Resume here

1. Read the required project documents and the complete DUR-002 diff, then start **R3-004**. Preserve the accepted
   R3 and durability boundaries and their fail-closed GPU-native authority.
2. Inventory one coherent host write/store authority boundary, identify its device-native and recovery consumers,
   and delete it as one independently reviewable slice. Keep behavior changes separate from structural deletion.
3. Run the affected recovery, transaction, device-history, and actual-GPU gates after every slice. Never replace a
   deleted host authority with a temporary CPU cache, probe, or replay executor.
4. Leave the known facade shape-changing-DDL/read and elided-rehydration stress findings under **READ-001** unless
   that task is explicitly promoted; they do not reopen DUR-002.
5. Run **BENCH-001** only in a quiet reproducible PostgreSQL window after fixing its stale mutation-boundary wrapper.
   Apply **CFG-001** only when its owning subsystem is already being changed.

## Last green evidence — 2026-07-17

- Execution library: **56 ordinary + 79 actual-GPU = 135/135** passed.
- Engine library: the current live-GPU default suite passes **533**, ignores **514**, and fails none. Focused DUR-002
  actual-GPU FUA/intent, strict-ack compatibility, lane DELETE replay, and lane UPDATE replay gates pass.
- WAL passes **96/96**, replication **189/189**, and SQL **23/23**. Workspace all-target/all-feature check and strict
  workspace Clippy pass; global formatting, diff whitespace, and source-size gates pass.
- Facade library: **39 ordinary + 8 actual-GPU = 47/47** passed.
- Nine high-risk device-history, transaction, tail, lane, publication-lock, and retained-completion families passed
  three sequential plus two concurrent HAZARD invocations. The corrected FK race and pinned-history epoch tests
  additionally passed ten consecutive GPU invocations each. No CUDA 700/716/717 occurred.
- The broader facade concurrency binary reproduces its ledgered **READ-001** state: 9 pass, 4 fail, 1 ignored on
  shape-changing-DDL/read route declines and the elided-rehydration race. This is not DUR-002 acceptance evidence.
- Current touched source sizes remain inside the standard: `wal/lib.rs` 1,943, `engine_commit.rs` 1,987,
  `engine_durability.rs` 1,284, `engine_lifecycle.rs` 1,345, and `tests/recovery.rs` 2,801 lines.
- The canonical report card completed both layers and cache regimes. Batched point reads reached **264.2M/s at
  p50 118us** in-L2 and **275.7M/s at p50 110us** out-of-L2; exact figures and raw-kernel ratios are in `STATUS.md`.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and use workspace-local `target/tmp` rather
  than `/tmp` for large generated artifacts.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
