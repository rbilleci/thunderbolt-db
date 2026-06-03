# Test Report Artifacts

This directory stores reproducible artifacts for parity, durability, replication, benchmark, and fault-injection runs.

## Layout

- `series/` stores all curated report streams and their immutable run reports.
- `runs/` is reserved for future one-off run reports that do not yet belong to a series.
- The top level intentionally stays small: this `README.md`, stable compatibility asset symlinks, and the `runs/` / `series/` directories.
- Older top-level report pointers were removed after internal references were updated to canonical series paths.

Migrated series:

- [`series/local-release-candidate-readiness/`](series/local-release-candidate-readiness/) owns local/dev readiness, production-security posture, replication-channel security, and release-candidate evidence-bundle reports.
- [`series/p7-p8-residency-baselines/`](series/p7-p8-residency-baselines/) owns early GPU relational benchmark and resident-cache baseline reports.
- [`series/p8-25pct-full-run/`](series/p8-25pct-full-run/) owns P8 25%/10% identical benchmark execution, streaming, aggregate refresh, and over-resident readiness reports.
- [`series/p8-ch-residency-setup/`](series/p8-ch-residency-setup/) owns P8 CH benchmark setup, residency baseline, generator, resident-cache, and early CUDA route probes.
- [`series/p8-concurrency-steady-state/`](series/p8-concurrency-steady-state/) owns P8 retained-read concurrency, persistent-client measurement, and steady-state response optimization reports.
- [`series/p8-copy-admission/`](series/p8-copy-admission/) owns P8 COPY/session, SQL-visible admission, value-index, WAL, and 30k rows/sec recheck reports.
- [`series/p8-partitioned-resident-routes/`](series/p8-partitioned-resident-routes/) owns partitioned retained-route implementation reports for over-resident readiness.
- [`series/p8-pgwire-endpoint/`](series/p8-pgwire-endpoint/) owns P8 pgwire endpoint, protocol/session adapters, client harnesses, fairness, cleanup, and endpoint curve reports.
- [`series/p8-retained-concurrency/`](series/p8-retained-concurrency/) owns retained-route concurrency graph history and generated CSV/SVG assets.
- [`series/p8-retained-route-primitives/`](series/p8-retained-route-primitives/) owns P8 retained-route primitive implementation reports before partitioned retained routes.

## Naming

Use UTC timestamps and stream identifiers:

- `YYYYMMDDTHHMMSSZ-parity-<seed>.md`
- `YYYYMMDDTHHMMSSZ-durability-<scenario>.md`
- `YYYYMMDDTHHMMSSZ-replication-<scenario>.md`
- `YYYY-MM-DD-<milestone>-benchmark-<scenario>.md`
- `YYYY-MM-DD-<milestone>-<scenario>.md`
- `YYYYMMDDTHHMMSSZ-jepsen-<scenario>.md`

When a report belongs to a migrated stream, keep the same filename under the
stream's `runs/` or `series/` location and update internal references to the
canonical series path before committing.

## Minimum report fields

Each report should include:

1. **Context**
   - git commit SHA
   - test stream (`parity`, `durability`, `replication`, `benchmark`, `jepsen`)
   - seed/workload profile
2. **Topology + fault schedule**
   - node layout
   - injected faults and timing
3. **Observed outcome**
   - pass/fail
   - invariant check status
   - command/error excerpt (if any)
4. **Watermark snapshot**
   - commit/applied/visible indices
   - WAL flushed/buffered/unflushed counts
   - pending batch depth + age/deadline
5. **Reproduction**
   - exact command(s) used
   - deterministic rerun notes

## Stub template

Copy this block when creating a new report:

```markdown
# <stream> report: <short title>

- timestamp_utc: <YYYY-MM-DDTHH:MM:SSZ>
- git_sha: <sha>
- stream: <parity|durability|replication|jepsen>
- seed_or_scenario: <value>
- topology: <value>

## Fault schedule
- <fault 1>
- <fault 2>

## Outcome
- result: <pass|fail>
- invariant_status:
  - wal_before_visibility: <pass|fail>
  - role_gating: <pass|fail>
  - deterministic_replay: <pass|fail|n/a>
- error_excerpt: <none|snippet>

## Watermarks
- commit_index: <n>
- applied_index: <n>
- visible_index: <n>
- wal_flushed_count: <n>
- wal_buffered_count: <n>
- wal_unflushed_count: <n>
- pending_batch_len: <n>
- pending_batch_oldest_age_ms: <n|none>
- pending_batch_time_until_deadline_ms: <n|none>

## Reproduction
```bash
<exact commands>
```

## Notes
- <follow-up actions>
```
