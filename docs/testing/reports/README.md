# Test Report Artifacts

This directory stores reproducible artifacts for parity, durability,
replication, benchmark, and fault-injection runs.

## Layout

- `runs/` stores individual immutable run reports when a stream has enough
  volume to leave the top-level directory.
- `series/` stores curated indexes and long-lived visual histories that span
  multiple runs.
- Top-level dated reports are legacy-compatible and may remain here until their
  stream is migrated in a bounded slice.
- Top-level compatibility pointers or symlinks may be kept when older report,
  PR, or Discord links are likely to exist.

Migrated series:

- `series/local-release-candidate-readiness/` owns local/dev readiness,
  production-security posture, replication-channel security, and
  release-candidate evidence-bundle reports.
- `series/p7-p8-residency-baselines/` owns the early GPU relational benchmark
  and resident-cache baseline reports.
- `series/p8-copy-admission/` owns the P8 COPY/session, SQL-visible admission,
  value-index, WAL, and 30k rows/sec recheck reports.
- `series/p8-partitioned-resident-routes/` owns the partitioned retained-route
  implementation reports for over-resident readiness.
- `series/p8-retained-concurrency/` owns the retained-route concurrency graph
  history and its generated CSV/SVG assets.

## Naming

Use UTC timestamps and stream identifiers:

- `YYYYMMDDTHHMMSSZ-parity-<seed>.md`
- `YYYYMMDDTHHMMSSZ-durability-<scenario>.md`
- `YYYYMMDDTHHMMSSZ-replication-<scenario>.md`
- `YYYY-MM-DD-<milestone>-benchmark-<scenario>.md`
- `YYYY-MM-DD-<milestone>-<scenario>.md`
- `YYYYMMDDTHHMMSSZ-jepsen-<scenario>.md`

When a report belongs to a migrated stream, keep the same filename under the
stream's `runs/` or `series/` location and leave a compatibility pointer from
the previous path if external references are expected.

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
