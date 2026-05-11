# Compatibility Scorecard

The compatibility scorecard turns test output into a machine-readable summary so progress is measurable in CI.

## Inputs

- Source of truth: `cargo test --workspace` output (`target/compat/cargo-test.log` in CI)
- Source of truth: `psql` golden suite report (`target/compat/psql-golden-report.json` in CI)
- Generator: `scripts/generate_compat_scorecard.py`

## Outputs

- JSON: `docs/compatibility/scorecard.latest.json`
- Markdown: `docs/compatibility/scorecard.latest.md`
- CI artifact: `compatibility-scorecard`

## Local run

```bash
mkdir -p target/compat
cargo test --workspace -- --color never | tee target/compat/cargo-test.log
python3 scripts/generate_compat_scorecard.py \
  --input target/compat/cargo-test.log \
  --psql-report target/compat/psql-golden-report.json \
  --output docs/compatibility/scorecard.latest.json \
  --markdown docs/compatibility/scorecard.latest.md \
  --baseline docs/compatibility/scorecard.baseline.json
```

## Real-client psql golden suite

The `psql` compatibility golden suite lives at `tests/compat/psql-golden/` and is executed with:

```bash
scripts/run_psql_golden.sh
```

This suite is intentionally libpq/`psql`-driven, not fixture-only protocol parsing, so client lifecycle behavior is validated end-to-end.

Current CI gate status: wired. The CI workflow installs `psql`, boots the repo-local compatibility endpoint via `PSQL_GOLDEN_BOOT_CMD='cargo run -p gpu_db_protocol --bin gpu-db-server -- --listen 127.0.0.1:55432'`, waits for the TCP endpoint, and then runs the same golden scenarios used locally.
The harness also emits `target/compat/psql-golden-report.json`, and the scorecard merges that real-client result stream with the Rust test log so protocol progress is measured in one place.

## Bucket registration for new tests

The scorecard classifies tests by test id patterns (crate + test name for Rust tests, `psql_golden::<scenario>` for real-client scenarios).
To make new compatibility tests visible in the right bucket:

1. Use test names that include the target behavior keyword, for example:
   - `startup`, `frontend`, `session_lifecycle` for protocol/client flows
   - `parses_`, `rejects_` for SQL/parser coverage
   - `relational`, `create_table`, `insert`, or `select` for the P1 relational SQL foundation bucket
   - `transaction`, `commit`, `rollback` for transaction flows
2. If a new category is needed, add or refine matching rules in `classify()` inside `scripts/generate_compat_scorecard.py`.
3. Regenerate scorecard outputs and include them in the same PR.

For `psql` golden scenarios specifically:
- use descriptive scenario filenames such as `03_extended_query_rejection.sql`;
- keep the behavior keyword in the filename (`startup`, `session_reset`, `prepare`, `transaction`, `relational`, `create_table`, `insert`, `select`, etc.);
- if the scenario belongs in a new compatibility bucket, extend `classify()` in `scripts/generate_compat_scorecard.py`.

## Trend hook

`docs/compatibility/scorecard.baseline.json` is the baseline placeholder used for failed-test deltas,
including both total failed-count drift and per-bucket failed-count drift.
Later CI can replace this with previous-run or main-branch baselines without changing the scorecard schema.
