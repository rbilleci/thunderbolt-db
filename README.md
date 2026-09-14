# gpu-database-engine

`gpu-database-engine` is an experimental GPU-native, PostgreSQL-compatible OLTP database engine. It speaks a
PostgreSQL wire protocol and keeps relational execution on the GPU: the host handles connection I/O, SQL parsing and
planning, transaction sequencing, WAL I/O, and device orchestration.

This is source release `0.1.0-alpha.1`. It is for evaluation and development with non-sensitive data, not a
production database or a drop-in replacement for PostgreSQL.

## Requirements

The release candidate was tested on Ubuntu 26.04 with Rust 1.97.1, an RTX PRO 6000 Blackwell GPU, and NVIDIA driver
595.84. A supported NVIDIA Blackwell GPU (compute capability 12.0 / `sm_120`) must be visible through `libcuda.so.1`.
There is no CPU relational fallback: unavailable or failed GPU work fails rather than running relational operators on
the host.

The checkout selects Rust 1.97.1 through [`rust-toolchain.toml`](rust-toolchain.toml). Install Rust with
[rustup](https://rustup.rs/), an NVIDIA display driver through the distribution or NVIDIA's documented driver
channel, and these Ubuntu build and validation tools:

```bash
sudo apt-get update
sudo apt-get install build-essential clang libclang-dev cmake pkg-config perl python3 ripgrep postgresql-client
```

The normal build uses checked-in PTX. A CUDA Toolkit and `nvcc` are needed only when regenerating PTX after editing
CUDA source.

## Build and run

Build the only product server from a clean checkout. Keeping temporary build files in the checkout avoids relying on
a small system `/tmp`.

```bash
mkdir -p target/tmp
export TMPDIR="$PWD/target/tmp"
cargo build --locked --release -p gpu_db_server --bin gpu-db-engine-server
```

Start a loopback server with a durable WAL. The first-run profile below uses serial WAL durability and one intent
lane.

```bash
mkdir -p target/oss-demo
GPU_DB_WAL_DURABILITY=serial \
GPU_DB_INTENT_LANES=1 \
GPU_DB_WAL_SEGMENT="$PWD/target/oss-demo/server.wal" \
target/release/gpu-db-engine-server --listen 127.0.0.1:55432
```

The default `local-dev` profile is loopback-only trust authentication without TLS. In another terminal, exercise a
durable transaction and read it through `psql`:

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

The rows have balances `15` and `20`, and the total is `35`. Stop the server, start it again with the same WAL path,
then repeat the read to verify recovery. Starting without `GPU_DB_WAL_SEGMENT` selects an in-memory WAL and does not
provide crash durability.

## Security profile

Use `local-dev` only for local evaluation with non-sensitive data. Any non-loopback deployment must explicitly use
the production profile with TLS and SCRAM-SHA-256 credentials:

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

The profile is covered by TLS/SCRAM checks, but it does not make this experimental engine production-ready. Read
[`SECURITY.md`](SECURITY.md) before deployment or reporting a vulnerability.

## Validate a checkout

Run the host-visible build and ownership checks first:

```bash
cargo check --locked --workspace --all-targets --all-features
scripts/check_product_ownership.sh
```

On a host with a visible NVIDIA GPU and `psql`, the durable release smoke builds the release server unless
`OSS_SERVER_BIN` names an existing executable. It creates its own temporary WAL and loopback port, exercises typed
and NULL values, commit and rollback, kills and restarts the server, and verifies the recovered ordered rows and
aggregate.

```bash
scripts/run_oss_release_smoke.sh
```

For an archive of a committed release tree, run:

```bash
RELEASE_REF=HEAD scripts/build_source_release.sh
```

It writes a reproducible `gzip -n` archive under `target/releases` by default, reports its commit, version, path,
SHA-256, and size, and checks the required license and third-party notice files. Set `RELEASE_OUT_DIR` to an existing
writable directory to choose another destination.

## Scope and limits

The server has one supported product binary and a single-node evaluation scope. It implements a bounded SQL, type,
catalog, and driver surface with transactional DDL/DML, prepared parameters, NULL handling, COPY, indexes and
constraints, joins, grouping, ordering, aggregates, and GPU-resident point and scan paths. The exact implementation
surface is recorded in [`docs/STATUS.md`](docs/STATUS.md).

It does not claim complete PostgreSQL compatibility, multi-node HA, automatic checkpoint/PITR policy,
100k-connection scale, or a comparative OLTP result. Current, deferred, and blocked work is owned only by
[`docs/PLAN.md`](docs/PLAN.md).

## Documentation

- [`docs/CHARTER.md`](docs/CHARTER.md) — GPU-native mandate and execution boundary.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — runtime, storage, durability, and execution design.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — accepted design decisions.
- [`docs/STATUS.md`](docs/STATUS.md) — implementation facts and evidence.
- [`docs/PLAN.md`](docs/PLAN.md) — the open-work ledger.
- [`docs/HANDOVER.md`](docs/HANDOVER.md) — current engineering baton.
- [`AGENTS.md`](AGENTS.md) — development gates, benchmark rules, and source-size guidance.

## License and contributing

Project-authored work is licensed under GNU GPL version 3 only (`GPL-3.0-only`) with a narrow section-7 permission
for the separately installed CUDA Driver API. See [`LICENSE`](LICENSE), [`CUDA_EXCEPTION`](CUDA_EXCEPTION),
[`COPYRIGHT`](COPYRIGHT), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Dependencies and external runtime
tools retain their own terms.

Contributions require Developer Certificate of Origin sign-off. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
