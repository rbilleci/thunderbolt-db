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
- **DUR-002 is the sole NOW task.** Bind multi-entry DDL existence/dependency checks to the working catalog, then
  execute the full crash/power-fail campaign across FUA lanes, checkpoint sidecar, WAL truncation, cold artifacts,
  recovery, and post-durable apply failure.
- **R3-004 remains blocked only by DUR-002.** Do not delete bootstrap/recovery host write/store sources until the
  durability campaign is accepted; then delete them as small independently gated slices under the same ID.
- **READ-001** owns the previously observed broad facade shape-changing-DDL/read and elided-rehydration stress
  debt. It is not part of the DUR-002 acceptance path and must not displace the active correctness task.
- **STRUCT-001 is closed.** The only production size exception is the registered 2,423-line `engine_expr.rs`.
  Physical multi-GPU work remains user-parked under **MULTI-001/002/003**.

## Integrated baseline — preserve it

- Local `main` is clean and contains canonical ADR commit `84cbab44`, accepted R3 integration commit `7e9e1568`,
  and this qualification/handover update. The exact pre-integration accepted tree remains recoverable at
  `preserve/r3-audited-20260717` (`eb6f0319`). Nothing has been pushed; `origin/main` remains at `84cbab44`.
- High-value R3 entry points are execution `write_locate.rs`/`resident_index_build.rs`; engine
  `engine_dml_concurrent/{state,wave,lane,lane_apply}.rs`, `engine_dml_prepare/{device_tuple,unique_conflict}.rs`,
  `engine_transaction_{delta,commit}.rs`, `engine_streaming_exec/streaming_{dml_class,transaction_cow}.rs`, and
  `write_path.rs`. Read the full diff before changing shared commit/recovery code for DUR-002.
- `PLAN.md` is reconciled to one active path; `STATUS.md` contains the accepted R3 evidence. Preserve that ownership
  split: only PLAN may own open work.

## Resume here

1. Read the required project documents and the integrated R3 diff from `84cbab44..main`, then start **DUR-002**.
   Preserve the accepted R3 boundary and its fail-closed GPU-native authority.
2. Fix multi-entry DDL preflight so existence and dependency decisions use the transaction's working catalog before
   apply. Add a regression that would fail if any entry consults only the published catalog.
3. Run the bounded crash/power-fail campaign at every DUR-002 boundary. Prove no acknowledged commit is lost, no
   rejected commit becomes visible, restart is deterministic, and a post-durable failure cannot permit unsafe
   continued service.
4. Record accepted evidence in `STATUS.md`; remove DUR-002 from `PLAN.md` only when its full gate passes. Then
   unblock **R3-004** and execute one independently reviewable deletion slice at a time.
5. Run **BENCH-001** only in a quiet reproducible PostgreSQL window after fixing its stale mutation-boundary wrapper.
   Apply **CFG-001** only when its owning subsystem is already being changed.

## Last green evidence — 2026-07-17

- Execution library: **56 ordinary + 79 actual-GPU = 135/135** passed.
- Engine library: **523 ordinary + 514 actual-GPU = 1,037/1,037** passed.
- Facade library: **39 ordinary + 8 actual-GPU = 47/47** passed.
- Nine high-risk device-history, transaction, tail, lane, publication-lock, and retained-completion families passed
  three sequential plus two concurrent HAZARD invocations. The corrected FK race and pinned-history epoch tests
  additionally passed ten consecutive GPU invocations each. No CUDA 700/716/717 occurred.
- Workspace all-target/all-feature `cargo check` and strict workspace Clippy passed. Changed Rust sources pass
  global rustfmt; `git diff --check` and source-size gates pass. `engine_dml_concurrent.rs` is 1,999 lines,
  `engine_commit.rs` is 1,992, and `tests/streaming_exec.rs` is 2,950.
- The canonical report card completed both layers and cache regimes. Batched point reads reached **264.2M/s at
  p50 118us** in-L2 and **275.7M/s at p50 110us** out-of-L2; exact figures and raw-kernel ratios are in `STATUS.md`.

## Required operations

- Read `AGENTS.md`, `docs/CHARTER.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/PLAN.md`,
  `docs/STATUS.md`, and `docs/CODE_SIZE.md` before changing runtime, storage, or scheduling.
- Never use `--gpu-reset`. Serialize ordinary GPU sweeps, use timeouts, and use workspace-local `target/tmp` rather
  than `/tmp` for large generated artifacts.
- Run `scripts/benchmark_report_card.sh` after any read-kernel, residency-layout, or result-path change; compare
  ratios rather than absolute bandwidth.
