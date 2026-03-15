# Observability and SLOs

## Required metrics

- End-to-end commit latency (p50/p95/p99)
- WAL append and flush latency
- Replication lag (leader->follower)
- Apply lag (commit-index -> applied-index)
- GPU queue depth and kernel time
- CPU fallback rate
- Batch wait time

## Baseline SLO tracking

- Throughput (TPS)
- Tail latency budget (P99/P99.9)
- Recovery time objectives

## Alerts

- Sustained replication lag above threshold
- Fallback rate spikes
- Commit latency regression
- Apply lag growth
