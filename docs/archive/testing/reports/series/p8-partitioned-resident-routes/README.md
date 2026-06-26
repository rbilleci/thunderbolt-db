# P8 Partitioned Resident Routes

This series preserves the bounded partitioned retained-route implementation
reports that narrowed the over-resident execution blocker into concrete route
primitives.

## Reports

| report | route primitive |
| --- | --- |
| `runs/2026-06-02-p8-partitioned-resident-count-route-v1.md` | partitioned retained `COUNT(*)` |
| `runs/2026-06-02-p8-partitioned-resident-key-lookup-route-v1.md` | partitioned retained int4 key lookup |
| `runs/2026-06-02-p8-partitioned-resident-multi-column-lookup-route-v1.md` | partitioned retained multi-column int4 lookup |
| `runs/2026-06-02-p8-partitioned-resident-sum-route-v1.md` | partitioned retained equality `SUM` |
| `runs/2026-06-02-p8-partitioned-resident-between-avg-route-v1.md` | partitioned retained `BETWEEN`/`AVG` |
| `runs/2026-06-02-p8-partitioned-resident-filtered-max-route-v1.md` | partitioned retained filtered `MAX` |
| `runs/2026-06-02-p8-partitioned-resident-filtered-min-route-v1.md` | partitioned retained filtered `MIN` |
| `runs/2026-06-02-p8-partitioned-resident-filtered-avg-route-v1.md` | partitioned retained filtered `AVG` |

Top-level report files with the original names are kept as compatibility
pointers for older links.
