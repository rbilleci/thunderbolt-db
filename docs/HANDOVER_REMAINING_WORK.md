# HANDOVER — Remaining Work on the Durable Write Path (post-E2.5c)

Written 2026-07-06, end of the E2.5c hardening + default-flip campaign (commits
`5bd9c3a0..a994ad85` on `codex/durable-write-throughput`, all pushed, every slice
independently opus-audited). Companion to `docs/HANDOVER.md` and `docs/WRITE_CONVEYOR.md`.

## What this campaign CLOSED (was sections 1-3 of the previous handover)

- **E2.5c-1 REOPEN/REPLAY (`5bd9c3a0`)**: lanes-mode reopen is LIVE — serial-then-lanes
  merge replay, crash-mid-wave orphan repair (never-acked frames above the cut durably
  discarded; wal-level `repair_lane_orphans`), reopen-continues-appending (oracle/base_seq/
  cuts pre-seeded; the intent-only contract survives reopen), ELISION RE-ENTRY at route
  prepare (a recovered engine can accept intents again). Disk-authoritative lane count +
  segment geometry.
- **E2.5c-2 TRUNCATION + RECYCLE (`e41ca889`)**: `Engine::checkpoint_intent_lanes()` bounds
  the lane logs — generation-pathed checkpoint segment committed by ONE atomic sidecar
  (`<base>.lanes-checkpoint`; segment fsync -> sidecar rename -> prune; a crash at any point
  leaves old or new checkpoint fully intact), baseline-aware recovery/repair/reopen, and
  segment RECYCLE: retired lane segments feed the pre-stager (epoch-stamped header rewrite,
  no extent prewrite — the prewrite fsync was a device-wide NVMe FLUSH during live fencing).
- **E2.5c-3 THE DEFAULT FLIP (`cb38bf80`)**: the new write engine IS the default —
  `GPU_DB_WAL_DURABILITY` defaults to fua (opt-out `serial`), `GPU_DB_INTENT_LANES` defaults
  to 10 (opt-out `0`/`1`), `GROUP_US` default 2000 (a population-scaled cap). Lane WAL backing
  is LAZY (created at route prepare / first wave), so default-ON costs nothing for engines
  that never take the intent path. Champion knobs = defaults; only deployment env remains
  (`GPU_DB_INTENT_LANE_SEGMENT_BYTES=512MiB+`, `GPU_DB_OPEN_SHARD_FLOOR_ROWS` sizing).
- **2M+ (a) STAGE ATTRIBUTION (`a994ad85`)**: `GPU_DB_BENCH_TIMELINE=1` now prints per-second
  per-wave stage costs. VERDICT: TPS dips are NOT stage inflation — they are (1) rare ~0.5s
  drive stalls holding the contiguous cut (acklag 66ms with normal completed-fence mean =
  one stalled FUA fence; drive physics, not software), and (2) closed-loop wave-size
  breathing. Validate (~210-320us/wave) is the constant structural wall.
- **2M+ (b) FUSED APPLY (`a994ad85`)**: one kernel for column scatter + created_by/row-id
  stamps + device row-count header + PK index CAS insert (was ~8 driver calls, now 1 HtoD +
  1 launch + 1 DtoH). DEFAULT ON. A/B best-of-3: **FUSED {1.65, 1.42, 1.43}M vs unfused
  {1.41, 1.22, 1.31}M sustained** (+14% mean, +17% best, p50 21.6 vs 26.3ms); 512-client
  floor unchanged (p50 1.22 p90 1.33ms @404k).
- **Raft-compatible seq oracle: ASSESSED + DEFERRED.** The engine's `CommitState.repl` is
  hard-typed `LocalReplicator`; `RaftReplicator` exists only in the replication crate and is
  not wired into any engine commit path, so lanes-under-Raft is structurally unreachable
  today — there is nothing to guard. Integration design (when the engine goes multi-node):
  lane seq claims must become replicated-log appends (the oracle's fetch_add block claim maps
  to a Raft log-index block reservation) and the client ack must gate on QUORUM commit of the
  lane frame rather than the local FUA cut — i.e. the cross-lane contiguous cut generalizes
  from "durable on this drive" to "committed by the quorum". That is the multi-node program,
  not a write-path slice.

## What REMAINS open

1. **The true 2M+ jump is structural (unchanged verdict, now better evidenced):** the
   validate stage (wave-batched device PK locate through the coalescer) is the last inline
   device wait. The recon's full PROBE-INSERT fusion is still blocked by host ordering
   (row-ids/commit seqs assigned post-conflict post-WAL) — the deeper redesign pre-resolves
   winner identity host-side so validate+insert can be ONE kernel. The fused-apply kernel
   (execution/src/lib.rs `FUSED_APPLY_PTX`) is the template: validate's probe and the fused
   insert already share hash buffers.
2. **Drive-stall tail**: the per-second attribution showed rare ~0.5s FUA stalls holding the
   cut (second-28 signature: acklag 66ms, normal completed-fence mean). Un-fixable in
   software on this consumer drive; on PLP-class media both the stall and the 0.9ms fence
   floor should vanish — re-run the low-client curve AND a 30s attribution run when such
   hardware is available (unchanged action from the previous handover).
3. **Type coverage #14** remains the true CPU-engine-deletion gate (lanes v1 is
   covered-INSERT int4-PK only; UPDATE/DELETE intents and wider types unstarted).
4. **Lanes auto-checkpoint policy**: `checkpoint_intent_lanes()` is an explicit operator
   call; a size-bound auto-trigger (the serial WAL's `maybe_auto_checkpoint_wal` twin) is a
   small follow-up once a cadence policy is chosen.
5. **Archive/PITR in lanes mode** stays refused (F5): lane commits carry no per-commit
   timestamps in v1. The checkpoint chain now bounds disk; timestamped archival needs a
   lane-record timestamp story first.

## Gate ledger (this campaign's exit state)

engine CPU 489/489 (new-default AND explicit-serial arms); FULL GPU sweep 370/370
single-threaded under all new defaults (fua + lanes 10 + fused apply); intent suites green
at default/serial/lanes2/lanes6, fused on/off; wal 82/82; write_conveyor 59/59; workspace
17/17 crates.

**Gate-hygiene lesson (bisected this campaign):** the full GPU sweep had not been run since
the capacity-floor commit `6c3fe683` landed — it silently broke
`sv6_created_by_gate_reader_at_prior_snapshot_never_sees_updated_key_twice` (admit-shape
calibration; recalibrated to 256 rows in `a994ad85`). Intent-suite-only gating is NOT enough
for slices that touch admission/append shapes: run the full sweep for those.

## Records ledger (updated)

| Metric | Value | Where |
|---|---|---|
| Peak sustained durable TPS | 1,677,904 (pre-campaign record) | 12dr/10P/6144, post-park |
| Best sustained, FUSED defaults | 1,650,718 (best-of-3 {1.65,1.42,1.43}M) | fused-apply A/B |
| Burst (100ms x 10 metric) | 2,782,540 | settle-fix round |
| 512-client ack (all defaults) | p50 1.22ms p90 1.33ms @ 404k TPS | drive-bound floor |
| WAL-lane standalone | 6.3M writes/s @32k clients | write_conveyor bench |

Operational cautions carry forward: /tmp is quota'd (TMPDIR=target/tmp); prune
target/wal-intent-bench (~10GB/champion run) and target/tmp after fua-mode suite runs;
bench declarations best-of-3 on a quiet box.
