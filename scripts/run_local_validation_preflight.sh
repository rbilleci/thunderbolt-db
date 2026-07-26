#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p target/compat

printf 'local_validation_preflight_step=fmt status=running\n'
cargo fmt --all -- --check
printf 'local_validation_preflight_step=fmt status=passed\n'

printf 'local_validation_preflight_step=clippy status=running\n'
cargo clippy --all-targets --all-features -- -D warnings
printf 'local_validation_preflight_step=clippy status=passed\n'

printf 'local_validation_preflight_step=cargo_test status=running\n'
cargo test --all --all-features -- --color never 2>&1 | tee target/compat/cargo-test.log
printf 'local_validation_preflight_step=cargo_test status=passed\n'

printf 'local_validation_preflight_step=product_ownership status=running\n'
scripts/check_product_ownership.sh
printf 'local_validation_preflight_step=product_ownership status=passed\n'

printf 'local_validation_preflight_step=psql_golden status=running\n'
PGHOST="${PGHOST:-127.0.0.1}" \
PGPORT="${PGPORT:-55432}" \
PGDATABASE="${PGDATABASE:-postgres}" \
PGUSER="${PGUSER:-postgres}" \
PSQL_GOLDEN_BOOT_CMD="${PSQL_GOLDEN_BOOT_CMD:-cargo run -p gpu_db_server --bin gpu-db-engine-server -- --listen 127.0.0.1:${PGPORT:-55432}}" \
PSQL_GOLDEN_WAIT_HOST="${PSQL_GOLDEN_WAIT_HOST:-127.0.0.1}" \
PSQL_GOLDEN_WAIT_PORT="${PSQL_GOLDEN_WAIT_PORT:-${PGPORT:-55432}}" \
PSQL_GOLDEN_WAIT_TIMEOUT_SEC="${PSQL_GOLDEN_WAIT_TIMEOUT_SEC:-30}" \
PSQL_GOLDEN_RESTART_EACH_SCENARIO="${PSQL_GOLDEN_RESTART_EACH_SCENARIO:-1}" \
PSQL_GOLDEN_REPORT="${PSQL_GOLDEN_REPORT:-target/compat/psql-golden-report.json}" \
  scripts/run_psql_golden.sh
printf 'local_validation_preflight_step=psql_golden status=passed\n'

printf 'local_validation_preflight_step=scorecard status=running\n'
python3 scripts/generate_compat_scorecard.py \
  --input target/compat/cargo-test.log \
  --psql-report "${PSQL_GOLDEN_REPORT:-target/compat/psql-golden-report.json}" \
  --output docs/compatibility/scorecard.latest.json \
  --markdown docs/compatibility/scorecard.latest.md \
  --baseline docs/compatibility/scorecard.baseline.json
git diff --exit-code -- \
  docs/compatibility/scorecard.latest.json \
  docs/compatibility/scorecard.latest.md
printf 'local_validation_preflight_step=scorecard status=passed\n'

printf 'local_validation_preflight=passed\n'
printf 'local_validation_preflight_scope=fmt_clippy_all_features_single_product_server_psql_golden_scorecard_freshness\n'
printf 'local_validation_preflight_cargo_all_features=passed\n'
printf 'local_validation_preflight_psql_golden=passed\n'
printf 'local_validation_preflight_scorecard_freshness=checked_in\n'
