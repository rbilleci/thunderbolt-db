# P8 COPY Admission

This series preserves the P8 COPY/load/admission reports, from reusable
protocol COPY parsing through SQL-visible engine admission, value-index/WAL
optimization, and 10% copy-throughput rechecks.

## Reports

| report | focus |
| --- | --- |
| `runs/2026-05-31-p8-protocol-wire-session-copy-adapter-v1.md` | reusable protocol wire/session COPY adapter boundary |
| `runs/2026-05-31-p8-engine-copy-wal-mvcc-adapter-v1.md` | engine COPY-to-WAL/MVCC adapter boundary |
| `runs/2026-05-31-p8-sql-visible-retained-admission-v1.md` | SQL-visible retained admission proof |
| `runs/2026-05-31-p8-benchmark-chunked-admission-v1.md` | benchmark-only chunked resident-cache admission |
| `runs/2026-06-01-p8-engine-pgwire-full-copy-session-v1.md` | engine pgwire COPY/session readiness |
| `runs/2026-06-01-p8-engine-pgwire-full-copy-throughput-v1.md` | full-COPY throughput blocker narrowing |
| `runs/2026-06-01-p8-10pct-copy-path-single-load-curves-v1.md` | 10% copy-path single-load curve blocker |
| `runs/2026-06-01-p8-engine-sql-visible-bulk-copy-admission-v1.md` | SQL-visible MVCC bulk COPY admission |
| `runs/2026-06-01-p8-engine-sql-visible-value-index-bulk-admission-v1.md` | value-index bulk admission |
| `runs/2026-06-01-p8-engine-sql-visible-copy-admission-phase-profile-v1.md` | COPY admission phase profiling |
| `runs/2026-06-02-p8-10pct-copy-admission-30000-recheck-v1.md` | 30k rows/sec COPY admission recheck |
| `runs/2026-06-02-p8-copy-admission-wal-value-index-path-v1.md` | WAL/value-index path optimization recheck |
| `runs/2026-06-02-p8-copy-admission-wal-value-index-architecture-v1.md` | WAL/value-index architecture slice |

Top-level report files with the original names are kept as compatibility
pointers for older links.
