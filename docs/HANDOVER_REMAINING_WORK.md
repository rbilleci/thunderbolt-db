# HANDOVER — Remaining Work on the Durable Write Path (E2.5b-2 → E2.5c and beyond)

Written 2026-07-06, end of the `codex/durable-write-throughput` campaign session.
Companion to `docs/HANDOVER.md` (the full evidence chain / session log — read its 2026-07-06
entries for the measurements behind every claim here) and `docs/WRITE_CONVEYOR.md` (the FUA WAL
design). Branch: `codex/durable-write-throughput`, all work committed and pushed; gates green
(engine 488/488 in default AND fua modes; wal 78/78; write_conveyor 58/58; GPU intent suites at
lanes off/2/6, `--ignored --test-threads=1 tests::intent_fast_path`).

## Where the system stands

- **Architecture (all shipped, default-safe):** N-lane intent commit pipeline (PK-hash routed,
  lock-free seq oracle, cross-lane validate/apply coalescers, no-reap apply handoff, cut-gated
  settlement), FUA fence-pool WAL lanes (pre-staged segments, sub-frame splitting, parked idle
  fence threads), and WORKLOAD ADAPTIVITY: one configuration serves 512 clients (p50 1.10-1.13ms,
  p90 1.36-1.40ms @ ~405k TPS) through 73k clients (peak sustained 1,677,904 TPS, mean ~1.5M ±15%,
  p90 ~31ms). Population-scaled wave formation, AUTO sub-frames, dynamic active-lane subset with a
  drain-barrier resize (full-width start; down-flips need a 500ms sustained-low streak).
- **Champion bench env:** `GPU_DB_WAL_DURABILITY=fua GPU_DB_INTENT_LANES=10
  GPU_DB_INTENT_LANE_SEGMENT_BYTES=536870912 GPU_DB_INTENT_LANE_MIN_WAVE=512
  GPU_DB_INTENT_LANE_GROUP_US=2000 GPU_DB_OPEN_SHARD_FLOOR_ROWS=48000000` + bench
  `ARM=driver WRITERS=12 PUMPS=10 WINDOW=6144 SECONDS=15 SHARD_TARGET=48000000`
  (`crates/engine/examples/intent_fast_path_bench.rs`; diagnostics: `[lanes:]`, `[pump host:]`,
  `[publish->settle:]`, `[fence:]`, `[adaptive:]`, `GPU_DB_BENCH_TIMELINE=1`).
- **Correctness state:** all five findings of the full-arc adversarial audit adopted (`9073c816`,
  incl. the CRITICAL oracle activation double-seed); latent apply-failure cut-hole livelock fixed
  (`782213d9`); crash-recovery parity tested; lanes REOPEN is refused by design (see E2.5c).

## 1. The 2M+ sustained push (task #7, still open)

Sustained mean is ~1.5M with ±15% run variance; single seconds already touch 2.08M. Two routes,
best pursued in this order:

a. **Variance hunt (cheapest).** The gap between the ~1.5M mean and the 1.68M peak is
   run-to-run/second-to-second breathing, not a fixed cost. `GPU_DB_BENCH_TIMELINE=1` gives
   per-second TPS; what's missing is per-second STAGE attribution (correlate dips with
   validate/apply/fence/settle). Suspects: XFS journal bursts, drive thermal/SLC behavior,
   jemalloc purge, scheduler migrations. A found-and-fixed 15% closes most of the 2M gap.

b. **The mega-fuse (recon done, sized, in HANDOVER.md).** Fusing validate+apply into one device
   pass. Full probe-insert fusion is BLOCKED by host ordering (row-ids/commit seqs are assigned
   post-conflict post-WAL; a device miss is not a commit). The realistic bounded slice:
   - Fuse the APPLY side into ONE launch: column append + created_by stamp + row_id stamp +
     `submit_i32_index_insert` (~7 driver calls → ~2-3). Entry points:
     `lane_apply_merged` (engine_dml_concurrent.rs), `try_append_to_resident_open_shard`
     (engine_residency.rs:7914 in-place branch), `extend_shard_pk_device_index_on_append`
     (engine_retained_read.rs:885), kernels in `crates/execution/src/lib.rs` (`:11566` insert,
     `:11706` write-locate — SAME hash-index buffers, so one kernel can probe→branch→insert).
   - Expected ~10-20% (apply is off the pump's critical path since no-reap); the bigger prize
     needs winner identity pre-resolved host-side so probe-insert can fuse — a design slice.

## 2. Sub-millisecond client acks (task #9 closed with a hardware verdict)

The ≤512-client ack is DRIVE-BOUND: fence mean 869-894µs/frame ≈ 90% of publish→settle, flat
across 128-512 clients; engine-side overhead totals ~0.25ms. **On PLP-class NVMe (fence
10-20µs) the code as-is should deliver p90 ~0.3-0.4ms.** Action when hardware is available:
re-run the low-client curve; re-probe `fence_qd`/sub-frame knee per device (the bimodal-FUA
behavior is drive-specific). No engine work pending.

## 3. E2.5c hardening (the productionization debt — required before default flips)

- **Lanes reopen/replay.** Reopen of a lanes-mode database is REFUSED today
  (`engine_lifecycle.rs` `intent_lane_files_exist` + `FuaWalLaneSet::reopen` fail-closed rules).
  Needed: serial-then-lanes merge replay (serial WAL covers pre-activation, lane logs carry
  explicit global seqs above `base_seq`; `recover_lanes` merge exists and is property-tested),
  then reopen-continues-appending. Gate with crash-mid-wave fault injection.
- **Lane-log truncation/archive.** Lane segments are never pruned; archive/PITR REFUSES in lanes
  mode (audit F5). Needs checkpoint + prefix truncation across N lanes (lock-step with the
  cross-lane cut), then FuaWalSegment-style RECYCLE wiring in the engine backend (recycle exists
  in `write_conveyor`, epoch-stamped; engine still create-only rolls with the pre-stager).
- **Raft-compatible seq oracle.** The lock-free oracle seeds from `repl.peek_next_index` once and
  the repl log intentionally does not carry lane payloads (single-node v1). Multi-node needs the
  claim integrated with the replication log.
- **Default flips.** `GPU_DB_WAL_DURABILITY=fua` and lanes-mode ON by default — only after
  reopen/replay + truncation land. Segment-size default stays 64MiB (a 512MiB DEFAULT quota-bombed
  test tempdirs — see the /tmp incident in HANDOVER.md; big segments are a deployment env:
  `GPU_DB_INTENT_LANE_SEGMENT_BYTES=536870912` is part of the champion config).
- **Adaptivity follow-ups (nice-to-have):** resize thresholds (UP 4096 / DOWN 1024 / 500ms
  streak / low=4) are hand-picked — consider making them env knobs; an incremental epoch handoff
  could replace the drain barrier entirely (design sketch: dual-epoch ledger union is UNSAFE
  without ordering — see the barrier safety argument in HANDOVER.md before attempting).

## 4. The wider program (context from the campaign)

- **Type coverage #14** remains the true CPU-engine-deletion gate (lanes v1 is covered-INSERT,
  int4-PK only; UPDATE/DELETE intents and wider types are unstarted on the lanes path).
- **Charter discipline:** data-plane work moves to the GPU; host data-plane costs are deletion
  targets. Every slice: implement → test → adversarial audit (opus) → adopt findings → push.
  GPU test sweeps single-threaded (`--test-threads=1`); grouped changes gate on BOTH the ignored
  GPU suites and the normal `tests::resident_probe::` suite.
- **Operational cautions:** `/tmp` is quota'd — `TMPDIR=target/tmp` is set in `.cargo/config.toml`
  (run `mkdir -p target/tmp` in fresh clones; the EDQUOT wedge signature is "every shell command
  exits 1 with no output"). `target/wal-intent-bench` accumulates ~10GB per champion run — prune.
  Bench runs must not share the box with builds/tests (±15% variance is real; best-of-3 anything
  you intend to declare).

## Records ledger (for regression checks)

| Metric | Value | Where |
|---|---|---|
| Peak sustained durable TPS | 1,677,904 (best-of-3 {1.23,1.60,1.68}M) | 12dr/10P/6144, post-park |
| Burst (100ms×10 metric) | 2,782,540 | settle-fix round |
| High-load tail | p90 ~28-31ms, p99 41-53ms | post-park adaptive |
| 512-client ack | p50 1.10-1.13ms, p90 1.36-1.40ms @ ~405k TPS | drive-bound floor |
| Resize barrier cost | 1 down-flip = 1.7ms; 0 resizes at steady load; max 259ms at transitions | barrier-softening |
| WAL-lane standalone | 6.3M writes/s @32k clients; 2.65M/s @ p90 927µs | write_conveyor bench |
