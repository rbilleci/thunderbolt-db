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

## Update expected artifacts

1. Run the suite against the target endpoint.
2. Inspect `target/psql-golden/*.txt`.
3. If the new output is correct and deterministic, copy it into `tests/compat/psql-golden/expected/`.

## Notes

- The harness intentionally strips volatile lines (timing/version/SSL banner noise).
- Keep scenario assertions stable and semantic, avoid transient text where possible.
- If an extended-query/prepared flow is intentionally unsupported, encode that as an explicit expected failure in a dedicated scenario.
