# gpu-database-engine

An experimental GPU-native PostgreSQL-compatible OLTP database engine. Relational decisions and result values run
on the GPU; the host owns the PostgreSQL wire protocol, planning, sequencing, durability I/O, and GPU orchestration.

This is an early source release for evaluation and development. It is not production-ready, does not implement the
full PostgreSQL surface, and currently requires NVIDIA Blackwell-class hardware. The exact built surface and open
work are recorded in [`docs/STATUS.md`](docs/STATUS.md) and [`docs/PLAN.md`](docs/PLAN.md).

## Tested platform and prerequisites

The release candidate is tested on Ubuntu 26.04 with Rust 1.97.1, an RTX PRO 6000 Blackwell GPU, and NVIDIA driver
595.84. The supported production floor is Blackwell / compute capability 12.0 (`sm_120`). Other Linux versions,
drivers, GPUs, and architectures have not completed the release gate.

A source build needs:

- the Rust 1.97.1 toolchain selected by `rust-toolchain.toml` and Cargo;
- an NVIDIA display driver exposing `libcuda.so.1` and a visible supported GPU;
- a C/C++ build toolchain, CMake, Clang and libclang, `pkg-config`, Perl, Python 3, and ripgrep;
- PostgreSQL `psql` for the quickstart, compatibility checks, and release smoke.

On Ubuntu, the native build tools and client can be installed with:

```bash
sudo apt-get update
sudo apt-get install build-essential clang libclang-dev cmake pkg-config perl python3 ripgrep postgresql-client
```

Install Rust through [rustup](https://rustup.rs/) and the NVIDIA driver through the distribution or NVIDIA's
documented driver channel. The proprietary driver is an external system dependency licensed separately by NVIDIA;
it is not included in this repository.

The normal build uses the checked-in PTX. The CUDA Toolkit and `nvcc` are only needed to regenerate PTX after
editing a `.cu` or `.cuh` file. Each CUDA source names its exact regeneration command next to the corresponding PTX.

## Build and run a durable local server

Build the sole product server from a clean checkout:

```bash
mkdir -p target/tmp
export TMPDIR="$PWD/target/tmp"
cargo build --locked --release -p gpu_db_server --bin gpu-db-engine-server
```

Start it on loopback with a persistent WAL path. The explicit serial durability mode is the most portable evaluation
profile; one intent lane keeps this first run on the simple fdatasync-backed path.

```bash
mkdir -p target/oss-demo
GPU_DB_WAL_DURABILITY=serial \
GPU_DB_INTENT_LANES=1 \
GPU_DB_WAL_SEGMENT="$PWD/target/oss-demo/server.wal" \
target/release/gpu-db-engine-server --listen 127.0.0.1:55432
```

The default `local-dev` security profile is loopback-only trust authentication without TLS. Leave the server in the
foreground, open another terminal in the checkout, and run:

```bash
export PGHOST=127.0.0.1 PGPORT=55432 PGUSER=postgres PGDATABASE=postgres PGSSLMODE=disable

psql -X -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, note TEXT);
INSERT INTO accounts VALUES (1, 10, 'kept'), (2, 20, NULL);
BEGIN;
UPDATE accounts SET balance = balance + 5 WHERE id = 1;
COMMIT;
BEGIN;
DELETE FROM accounts WHERE id = 2;
ROLLBACK;
SELECT id, balance, note FROM accounts ORDER BY id;
SELECT SUM(balance) AS total_balance FROM accounts;
SQL
```

The result contains both rows, balances `15` and `20`, and a total of `35`. Stop the server with Ctrl-C, run the same
server command again, and verify recovery:

```bash
psql -X -v ON_ERROR_STOP=1 -c 'SELECT id, balance, note FROM accounts ORDER BY id'
```

The recovered result must still contain balances `15` and `20`. Reusing the same WAL path is what makes this a
restart test. Starting the server without `GPU_DB_WAL_SEGMENT` selects an in-memory WAL and is not crash-durable.

For an automated version of the durable SQL/restart route, run:

```bash
scripts/run_oss_release_smoke.sh
```

## Security profiles

Use `local-dev` only for the loopback quickstart with non-sensitive data. The server has an explicit production
transport profile requiring TLS plus SCRAM-SHA-256 credentials:

```bash
GPU_DB_WAL_DURABILITY=serial \
GPU_DB_WAL_SEGMENT="$PWD/target/oss-demo/production.wal" \
target/release/gpu-db-engine-server \
  --listen 127.0.0.1:55432 \
  --security-profile production \
  --tls-cert path/to/server.crt \
  --tls-key path/to/server.key \
  --auth-user gpudb \
  --auth-scram-verifier-file path/to/scram-verifier.txt
```

That profile is implemented and tested, but this experimental release does not claim production operational
readiness. See [`SECURITY.md`](SECURITY.md) before any deployment or vulnerability report.

## Current envelope

The engine exposes a PostgreSQL 16-style wire protocol and a bounded SQL/type surface including transactional DDL
and DML, prepared parameters, NULL handling, COPY, indexes and constraints, joins, grouping, ordering, aggregates,
and GPU-resident point/scan paths documented in `docs/STATUS.md`. WAL-before-visibility and crash recovery are
release invariants.

Current boundaries include:

- one supported product server and single-node evaluation scope;
- NVIDIA Blackwell GPU required, with no CPU relational fallback;
- partial PostgreSQL SQL/catalog/type/driver compatibility rather than drop-in PostgreSQL equivalence;
- no accepted multi-node HA, automatic checkpoint/PITR policy, 100k-connection scale, or comparative OLTP result;
- the production TLS/SCRAM profile does not by itself make the system production-ready.

These boundaries are tracked under existing IDs in `docs/PLAN.md`; they do not prevent the durable local workflow
above from functioning.

## Validation

Host-neutral checks and the trusted GPU release matrix are intentionally separate. Common developer gates are:

```bash
cargo check --locked --workspace --all-targets --all-features
cargo test --locked --workspace --all-features
scripts/run_psql_golden.sh
scripts/run_application_driver_smokes.sh
```

GPU tests use bounded timeouts and serial sweeps. Never use `--gpu-reset`. Changes to a read kernel, residency
layout, or result path use the benchmark report card described in `AGENTS.md`; its quick mode is only a development
screen. Follow [`AGENTS.md`](AGENTS.md) for the complete gate order.

## Project documentation

- [`docs/CHARTER.md`](docs/CHARTER.md) — mandate and execution boundary.
- [`docs/PLAN.md`](docs/PLAN.md) — the only open/deferred work ledger.
- [`docs/STATUS.md`](docs/STATUS.md) — current implementation facts and evidence.
- [`docs/HANDOVER.md`](docs/HANDOVER.md) — current resume baton.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — accepted rationale.
- [`docs/CODE_SIZE.md`](docs/CODE_SIZE.md) — source-size and decomposition standard.

Material under `docs/archive/` is historical and non-actionable. It may describe superseded behavior or local
benchmark environments.

## License and contributions

Project-authored work is licensed under GNU GPL version 3 only (`GPL-3.0-only`) with a narrow section-7 permission
for the separately installed CUDA Driver API; see [`LICENSE`](LICENSE), [`CUDA_EXCEPTION`](CUDA_EXCEPTION), and
[`COPYRIGHT`](COPYRIGHT). Dependencies and external runtime tools retain their own terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Contributions require a Developer Certificate of Origin sign-off. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
