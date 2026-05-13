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
- Current scenario coverage includes connect/simple-query, session reset probes, SQL prepare/execute/deallocate flow, transaction begin/commit/rollback flow, relational create/insert/select flows, metadata-backed catalog/type introspection, `pg_catalog.pg_tables` discovery, direct `pg_catalog.pg_namespace` lookup for the supported `public` namespace, empty `pg_catalog.pg_indexes` discovery for the current no-user-visible-SQL-index subset, empty `pg_catalog.pg_constraint` discovery for the current no-SQL-constraint subset, empty `pg_catalog.pg_attrdef` discovery for the current no-column-default subset, empty `pg_catalog.pg_description` discovery for the current no-comment subset, joined `pg_catalog.pg_class` / `pg_catalog.pg_namespace` relation metadata and table-name `IN (...)` plus namespace-only `public` relation subsets for supported `public` tables, joined `pg_catalog.pg_attribute` / `pg_catalog.pg_class` / `pg_catalog.pg_namespace` column metadata with formatted supported type names, real `psql \dt`, `\dt+`, `\dt+ <table>`, `\dt <prefix>*`, `\dt+ <prefix>*`, `\dt public.<prefix>*`, and `\dt+ public.<prefix>*` table listing, real `psql \d <table>` / `\d+ <table>` / `\d <prefix>*` / `\d+ <prefix>*` / `\d public.*` / `\d public.<prefix>*` / `\d+ public.<prefix>*` column display, real `psql \di` empty index listing, `\dv` / `\dv+` empty view listing, `\dm` / `\dm+` plus `\ds` / `\ds+` empty materialized-view/sequence listing, `\df` empty function listing, `\da` empty aggregate listing, `\dc` empty conversion listing, `\do` empty operator listing, `\dO` empty collation listing, `\dC` empty cast listing, `\dRp` empty publication listing, `\dRs` empty subscription listing, `\ddp` empty default-access-privilege listing, `\du` bootstrap role listing, `\l` bootstrap database listing, `\db` bootstrap tablespace listing, `\dx` empty extension listing, `\dL` empty procedural-language listing, `\dD` / `\dD+` empty domain listing, and `\dA` bootstrap access-method listing for the current subset, real `psql \dn` and `\dn+ public` schema listing, real `psql \dT pg_catalog.int4` / `\dT pg_catalog.text` type display, `\dT pg_catalog.*` and `\dT+ pg_catalog.*` supported built-in type listing, first-slice `information_schema` table/column/schema introspection including richer supported table metadata, table-name `IN (...)` and exact table-name table subsets, all supported `public` table columns, table-name `IN (...)` column subsets, per-table column detail projection, rich column metadata, extended column numeric precision/radix/scale metadata for the supported subset, including exact table-filtered and table-name `IN (...)` extended column metadata, and empty `information_schema.table_constraints` / `information_schema.key_column_usage` discovery for the current no-SQL-constraint subset, extended-query bind execution, real `psql` `FETCH_COUNT` cursor-style SELECT retrieval, real `psql \gdesc` result-description display over supported relational `SELECT` statements, and a deterministic error path.
- Scenario 44 covers catalog-qualified `information_schema.columns` extended metadata filters for `table_catalog = current_database()` backed by supported `public` session catalog metadata.
- Scenario 45 covers catalog-qualified `information_schema.tables` richer table metadata filters for `table_catalog = current_database()` backed by supported `public` session catalog metadata.
- Scenario 46 covers direct `pg_catalog.pg_namespace` lookup for the supported `public` namespace OID/name metadata.
- Scenario 47 covers real `psql \dn+ public` verbose schema introspection for the supported `public` namespace, returning owner metadata plus empty ACL/description fields for the current no-ACL/no-comment subset.
- Scenario 48 covers real plain `psql \d` relation listing for supported `public` session tables, backed by the session catalog relation metadata.
- Scenario 49 covers real `psql \dv` / `\dv+` view listing traffic, returning no rows for the current no-SQL-view subset while preserving supported table metadata.
- Scenario 50 covers real `psql \dm` / `\dm+` materialized-view and `\ds` / `\ds+` sequence listing traffic, returning no rows for the current no-materialized-view/no-sequence subset while preserving supported table metadata.
- Scenario 51 covers real `psql \df` function listing traffic, returning no rows for the current no-user-defined-function subset.
- Scenario 52 covers real `psql \du` role listing traffic, returning the bootstrap `postgres` role from a narrow `pg_catalog.pg_roles` compatibility slice while keeping role mutation out of scope.
- Scenario 53 covers real `psql \l` database listing traffic, returning the supported bootstrap `postgres` database from a narrow `pg_catalog.pg_database` compatibility slice while keeping database creation, templates, ACL mutation, and broader database catalog behavior out of scope.
- Scenario 54 covers real `psql \dx` extension listing traffic, returning no rows for the current no-extension subset while keeping extension install and extension catalog state out of scope.
- Scenario 55 covers real `psql \db` tablespace listing traffic, returning the bootstrap `pg_default` and `pg_global` metadata through a narrow `pg_catalog.pg_tablespace` compatibility slice while keeping tablespace creation, location management, options, and broader tablespace catalog behavior out of scope.
- Scenario 56 covers real `psql \dL` procedural-language listing traffic, returning no rows for the current no-procedural-language subset while keeping language creation and broader `pg_catalog.pg_language` behavior out of scope.
- Scenario 57 covers real `psql \dA` access-method listing traffic, returning the supported bootstrap `heap` table access method through a narrow `pg_catalog.pg_am` compatibility slice while keeping access-method creation, extension, and broader access-method catalog behavior out of scope.
- Scenario 58 covers real `psql \dT pg_catalog.*` and `\dT+ pg_catalog.*` type-listing traffic, returning the supported `int4`/`text` registry from a narrow `pg_catalog.pg_type` compatibility slice while keeping user-defined types and broader type catalog behavior out of scope.
- Scenario 59 covers real `psql \dD` and `\dD+` domain-listing traffic, returning no rows for the current no-domain subset while keeping domain creation, domain constraints, and broader domain catalog behavior out of scope.
- Scenario 60 covers real `psql \da` aggregate-listing traffic, returning no rows for the current no-user-defined-aggregate subset while keeping aggregate creation and broader function catalog behavior out of scope.
- Scenario 61 covers real `psql \dc` conversion-listing traffic, returning no rows for the current no-conversion subset while keeping encoding conversion creation and broader conversion catalog behavior out of scope.
- Scenario 62 covers real `psql \do` operator-listing traffic, returning no rows for the current no-user-defined-operator subset while keeping operator creation and broader operator catalog behavior out of scope.
- Scenario 63 covers real `psql \dO` collation-listing traffic, returning no rows for the current no-user-defined-collation subset while keeping collation creation and broader collation catalog behavior out of scope.
- Scenario 64 covers real `psql \dC` cast-listing traffic, returning no rows for the current no-user-defined-cast subset while keeping cast creation and broader cast catalog behavior out of scope.
- Scenario 65 covers real `psql \dRp` publication-listing traffic, returning no rows for the current no-publication subset while keeping publication creation and broader publication catalog behavior out of scope.
- Scenario 66 covers real `psql \dRs` subscription-listing traffic, returning no rows for the current no-subscription subset while keeping subscription creation and broader subscription catalog behavior out of scope.
- Scenario 67 covers real PostgreSQL 16 `psql` `FETCH_COUNT` cursor flow over a supported relational `SELECT`, proving session-local `DECLARE ... CURSOR FOR SELECT`, repeated `FETCH FORWARD n`, and cursor `CLOSE` handling return the same ordered rows as the normal SELECT path.
- Scenario 68 covers real PostgreSQL 16 `psql \ddp` default-access-privilege listing traffic, returning no rows for the current no-default-ACL subset while keeping default privilege mutation and broader ACL catalog behavior out of scope.
- Scenario 69 covers real PostgreSQL 16 `psql \gdesc` result-description traffic over a supported relational `SELECT`, including the follow-up `pg_catalog.format_type` formatter query over described `int4`/`text` result OIDs.
- CI now boots the repo-local compatibility endpoint with `cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen 127.0.0.1:55432` before running the suite, so Q4 has a real service endpoint instead of a manual-only placeholder.
- If a wire-level extended-query flow is intentionally unsupported, encode that as an explicit expected failure in a dedicated scenario and set the expected `.rc` artifact.
