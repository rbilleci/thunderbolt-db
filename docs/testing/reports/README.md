# Test Report Artifacts

This directory stores reproducible artifacts for parity, durability, replication, benchmark, and fault-injection runs.

## Naming

Use UTC timestamps and stream identifiers:

- `YYYYMMDDTHHMMSSZ-parity-<seed>.md`
- `YYYYMMDDTHHMMSSZ-durability-<scenario>.md`
- `YYYYMMDDTHHMMSSZ-replication-<scenario>.md`
- `YYYY-MM-DD-<milestone>-benchmark-<scenario>.md`
- `YYYYMMDDTHHMMSSZ-jepsen-<scenario>.md`

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
