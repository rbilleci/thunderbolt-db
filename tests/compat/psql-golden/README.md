# psql golden compatibility suite

This suite exercises a real `psql`/libpq path and compares normalized output against golden artifacts.

## Run

```bash
export PGHOST=127.0.0.1
export PGPORT=5432
export PGDATABASE=postgres
export PGUSER=postgres
export PGPASSWORD=postgres

scripts/run_psql_golden.sh
```

Optional:
- `PSQL_BIN=/path/to/psql`
- `PSQL_GOLDEN_OUT_DIR=/tmp/psql-golden`
- `PSQL_GOLDEN_BOOT_CMD='cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen 127.0.0.1:5432'`
- `PSQL_GOLDEN_BOOT_CWD=/path/to/repo-or-service`
- `PSQL_GOLDEN_STOP_CMD='pkill -f gpu-db-server'`
- `PSQL_GOLDEN_WAIT_HOST=127.0.0.1`
- `PSQL_GOLDEN_WAIT_PORT=5432`
- `PSQL_GOLDEN_WAIT_TIMEOUT_SEC=30`
- `PSQL_GOLDEN_REPORT=target/compat/psql-golden-report.json`

When `PSQL_GOLDEN_BOOT_CMD` is set, the harness starts the target service itself, waits for the configured TCP endpoint to accept connections, captures boot logs in `target/psql-golden/boot.log`, and writes a machine-readable scenario report to `PSQL_GOLDEN_REPORT` for CI scorecards.

## Update expected artifacts

1. Run the suite against the target endpoint.
2. Inspect `target/psql-golden/*.txt`.
3. If the new output is correct and deterministic, copy it into `tests/compat/psql-golden/expected/`.
4. Optionally add `tests/compat/psql-golden/expected/<scenario>.rc` when a scenario expects non-zero `psql` exit status (defaults to `0` when omitted).
5. Optionally add `tests/compat/psql-golden/scenarios/<scenario>.psqlargs` with one extra `psql` CLI argument per line when a scenario needs custom flags.

## Notes

- The harness intentionally strips volatile lines (timing/version/SSL banner noise).
- Keep scenario assertions stable and semantic, avoid transient text where possible.
- Current scenario coverage includes connect/simple-query, session reset probes, SQL prepare/execute/deallocate flow, transaction begin/commit/rollback flow, relational create/insert/select flows, metadata-backed catalog/type introspection, `pg_catalog.pg_tables` discovery, empty `pg_catalog.pg_indexes` discovery for the current no-user-visible-SQL-index subset, joined `pg_catalog.pg_class` / `pg_catalog.pg_namespace` relation metadata and table-name `IN (...)` plus namespace-only `public` relation subsets for supported `public` tables, joined `pg_catalog.pg_attribute` / `pg_catalog.pg_class` / `pg_catalog.pg_namespace` column metadata with formatted supported type names, real `psql \dt`, `\dt+`, `\dt+ <table>`, `\dt <prefix>*`, `\dt+ <prefix>*`, `\dt public.<prefix>*`, and `\dt+ public.<prefix>*` table listing, real `psql \d <table>` / `\d+ <table>` / `\d <prefix>*` / `\d+ <prefix>*` / `\d public.*` / `\d public.<prefix>*` / `\d+ public.<prefix>*` column display, real `psql \di` empty index listing for the current subset, real `psql \dn` schema listing, real `psql \dT pg_catalog.int4` / `\dT pg_catalog.text` type display, first-slice `information_schema` table/column/schema introspection including richer supported table metadata, table-name `IN (...)` table subsets, all supported `public` table columns, table-name `IN (...)` column subsets, per-table column detail projection, rich column metadata, and extended column numeric precision/radix/scale metadata for the supported subset, including table-filtered extended column metadata, extended-query bind execution, and a deterministic error path.
- CI now boots the repo-local compatibility endpoint with `cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen 127.0.0.1:55432` before running the suite, so Q4 has a real service endpoint instead of a manual-only placeholder.
- If a wire-level extended-query flow is intentionally unsupported, encode that as an explicit expected failure in a dedicated scenario and set the expected `.rc` artifact.
