# Compatibility Scorecard

The compatibility scorecard turns test output into a machine-readable summary so progress is measurable in CI.

## Inputs

- Source of truth: `cargo test --workspace` output (`target/compat/cargo-test.log` in CI)
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
  --output docs/compatibility/scorecard.latest.json \
  --markdown docs/compatibility/scorecard.latest.md \
  --baseline docs/compatibility/scorecard.baseline.json
```

## Bucket registration for new tests

The scorecard classifies tests by test id patterns (crate + test name).
To make new compatibility tests visible in the right bucket:

1. Use test names that include the target behavior keyword, for example:
   - `startup`, `frontend`, `session_lifecycle` for protocol/client flows
   - `parses_`, `rejects_` for SQL/parser coverage
   - `transaction`, `commit`, `rollback` for transaction flows
2. If a new category is needed, add or refine matching rules in `classify()` inside `scripts/generate_compat_scorecard.py`.
3. Regenerate scorecard outputs and include them in the same PR.

## Trend hook

`docs/compatibility/scorecard.baseline.json` is the baseline placeholder used for simple failed-test deltas.
Later CI can replace this with previous-run or main-branch baselines without changing the scorecard schema.
