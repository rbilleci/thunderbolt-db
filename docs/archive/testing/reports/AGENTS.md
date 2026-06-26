# Reports Directory Guide

This directory is intentionally organized as a small top-level index plus
curated report series. Keep it that way.

## Top-Level Rules

- Keep `docs/testing/reports/` mostly empty.
- The only top-level markdown file should normally be `README.md`, plus this
  `AGENTS.md`.
- Do not add dated report files directly under `docs/testing/reports/`.
- Do not reintroduce top-level compatibility pointer files for moved reports.
- Stable generated-asset symlinks may remain at the top level when older graph
  links need them.

## Where Reports Go

Put report markdown files under:

```text
docs/testing/reports/series/<series-name>/runs/<report-file>.md
```

Each series should have a `README.md` that describes the stream and links to
its run reports.

Current report series:

- `local-release-candidate-readiness/` - local/dev readiness, production
  security posture, replication-channel security, and release-candidate
  evidence bundles.
- `p7-p8-residency-baselines/` - early GPU relational benchmark and
  resident-cache baseline reports.
- `p8-25pct-full-run/` - P8 25%/10% identical benchmark execution, streaming,
  aggregate refresh, and over-resident readiness.
- `p8-ch-residency-setup/` - P8 CH benchmark setup, residency baseline,
  generator, resident-cache, and early CUDA route probes.
- `p8-concurrency-steady-state/` - retained-read concurrency,
  persistent-client measurement, and steady-state response optimization.
- `p8-copy-admission/` - COPY/session, SQL-visible admission, value-index,
  WAL, and 30k rows/sec recheck reports.
- `p8-partitioned-resident-routes/` - partitioned retained-route
  implementation reports for over-resident readiness.
- `p8-pgwire-endpoint/` - pgwire endpoint, protocol/session adapters, client
  harnesses, fairness, cleanup, and endpoint curves.
- `p8-retained-concurrency/` - retained-route concurrency graph history and
  generated CSV/SVG assets.
- `p8-retained-route-primitives/` - retained-route primitive implementation
  reports before partitioned retained routes.

If a new report does not fit any existing series, create a new
`series/<series-name>/README.md` and put the report under that series'
`runs/` directory. Prefer a focused series name over a generic bucket.

## Moving Or Adding Reports

When moving or adding report files:

1. Keep the original report filename unless there is a strong reason to rename
   it.
2. Update all internal links to use the canonical series path.
3. Use relative links from series README files to `runs/<file>.md`.
4. Fix image/asset links after moves; run reports nested under
   `series/<name>/runs/` usually need `../../../series/...` for shared report
   assets under `docs/testing/reports/series/`.
5. Keep generated benchmark artifacts under `target/` out of link-existence
   checks unless they are intentionally checked into the repo.

## Validation Before Commit

Before committing report organization changes, run at minimum:

```bash
git diff --check
git diff --cached --check
```

Also run a local markdown link check or equivalent targeted check that verifies
repo-local links still resolve after moves. A successful final state should not
have stale links to old top-level dated reports such as:

```text
docs/testing/reports/2026-...
../reports/2026-...
```

## Commit Scope

Keep report reorganization commits scoped to:

- `docs/testing/reports/`
- report references in docs such as `README.md`, `docs/testing/benchmarks/`,
  or compatibility/roadmap docs

Do not stage unrelated research-loop files, lock files, benchmark artifacts, or
implementation changes as part of a reports reorganization commit.
