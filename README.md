# gpu-database-engine

A GPU-native PostgreSQL-compatible database engine under active development. GPU-resident execution is the
product direction: the host handles protocol, planning, sequencing, durability, replication, staging, and final
readback, while relational decisions and result values execute on the device.

## Start here

- [`docs/CHARTER.md`](docs/CHARTER.md) — mandate and non-negotiable execution boundary.
- [`docs/PLAN.md`](docs/PLAN.md) — the **only** open/deferred work ledger.
- [`docs/STATUS.md`](docs/STATUS.md) — current implementation facts and verification snapshot.
- [`docs/HANDOVER.md`](docs/HANDOVER.md) — short current resume baton.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — accepted rationale and ADRs.
- [`docs/CODE_SIZE.md`](docs/CODE_SIZE.md) — source-size and safe decomposition standard.

Design references under `docs/design/` are non-authoritative. Material under `docs/archive/` is historical and
must never be interpreted as current work.

## Current envelope

The engine has production-default GPU admission, sharded and streaming relational execution, GPU-native
catalog/transient relations, PostgreSQL-facing compatibility for the documented bounded surface, WAL-before-
visibility durability, covered intent-lane writes, and mixed GPU read/write gates. See `docs/STATUS.md` for the
exact built surface and measurements; see `docs/PLAN.md` for every remaining obligation.

## Common validation

```bash
cargo check --workspace --all-targets --all-features
cargo test --workspace --all-features
scripts/run_psql_golden.sh
scripts/run_application_driver_smokes.sh
```

Read/residency/result-path changes use the canonical report card:

```bash
scripts/benchmark_report_card.sh --quick  # development screen: Sections A+B, never acceptance evidence
scripts/benchmark_report_card.sh --full   # frozen-candidate acceptance card: Sections A+B+C
```

GPU tests must use bounded timeouts and serial sweeps. Never use `--gpu-reset`. Follow `AGENTS.md` for the full
repository discipline and environment-specific gates.

## Safety invariant

A state transition is never visible before its WAL record is durably committed. Recovery, fallback, and repair
changes must preserve acknowledged commits; fail-loud behavior is not an acceptable substitute for an
RPO-preserving repair path.
