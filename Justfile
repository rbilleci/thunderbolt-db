set shell := ["bash", "-cu"]

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace --all-features

compat-scorecard:
	mkdir -p target/compat
	cargo test --workspace -- --color never 2>&1 | tee target/compat/cargo-test.log
	python3 scripts/generate_compat_scorecard.py \
	  --input target/compat/cargo-test.log \
	  --output docs/compatibility/scorecard.latest.json \
	  --markdown docs/compatibility/scorecard.latest.md \
	  --baseline docs/compatibility/scorecard.baseline.json

compat-psql-golden:
	scripts/run_psql_golden.sh

check: fmt clippy test
