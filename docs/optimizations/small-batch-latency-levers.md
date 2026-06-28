# Small-batch / single-batch (batch=1) latency levers

**Scope:** minimizing **single-request / small-batch latency** on the lpb GPU int4 unique-key index-probe
read path. This is the latency-focused deep-dive; the broader read-path backlog (incl. the p99 tail and
throughput-relevant items) is in [`read-path-lpb-levers.md`](read-path-lpb-levers.md). Levers #1, #4, #5 here
overlap that doc — **do them once**, not twice.

**Status when written:** 2026-06-28, post-`ec4b1a8c`. **Code-only analysis** (no benchmarks — another agent is
optimizing concurrently). Key files: `crates/execution/src/lib.rs` (submit `~9790-9919`, complete
`2706-2860`), `crates/execution/src/wave.rs` (host-mapped zero-copy infra to reuse).

---

## Why batch=1 is special (the model)

At batch=1 the GPU does one thread probing a hash table (~ns). **All the cost is host↔GPU round-trips + the
launch + fixed host setup.** The phase-split measured lpb batch=1 = **9us submit + 16us complete = 25us**
(vs the wave's 10us submit + ~0us complete). So the governing equation is:

> batch=1 latency ≈ (number of `cuStreamSynchronize` round-trips) × (~5-7us each) + launch + fixed host work

The critical path, as written today:

- **Submit** (`lib.rs:9790-9919`, no sync — deferred): 5 device-buffer pool leases (`9790`…) →
  `cached_function` lookup (`9795`) → HtoD needles (`9856`) + memset count (`9865`) → 2x `cuEventRecord`
  (`9881`,`9901`) → `cuLaunchKernel` (`9884`).
- **Complete** (`lib.rs:2706-2860`): **sync #1** kernel-drain (`2730`) → `cuEventElapsedTime` (`2734`) →
  DtoH count + **sync #2** (`2778`,`2787`) → size Vecs from count (`2814`) → 3x DtoH results +
  **sync #3** (`2822-2846`).

**Three full round-trips + one launch.** Lowering the round-trip count is the dominant lever.

---

## Levers (ranked by batch=1 impact)

### 1. Collapse 3 syncs → 1 (dominant)
- **sync #1 (`lib.rs:2730`) exists only for the timing read** (`2734`). The count DtoH (`2778`) is
  stream-ordered after the kernel, so sync #2 already covers kernel completion. With timing gated (lever #4),
  sync #1 is pure redundancy → remove → 2 syncs.
- **sync #2 (`lib.rs:2787`) exists only to size the result Vecs from the count** (`2814-2816`). For the unique
  index (<=1 match/needle), over-allocate to `needle_count` and fold count+results into the single covering
  sync #3 → remove → **1 sync.**
- **Payoff:** 3 round-trips → 1 ≈ complete 16us → ~6us. **Catch:** over-fetch is valid only for the unique
  index; gate behind a submission-type flag so the `equal_any` scan path (>1 match/needle) keeps its count
  sync. (Same lever as `read-path-lpb-levers.md` #2.)

### 2. Zero-copy host-mapped output + spin-poll (beat the 1-sync floor — the wave's trick, no persistent kernel)
- **Change:** the kernel writes results into a **device-mapped pinned** buffer + sets a host-mapped `done`
  flag (`st.release.sys`); complete = spin-poll the flag, then read results straight from the mapped buffer.
  **No DtoH, no `cuStreamSynchronize`.** The infra already exists in `crates/execution/src/wave.rs`
  (`cuMemHostAlloc` DEVICEMAP `0x02` + `cuMemHostGetDevicePointer`); this is exactly why the wave's complete
  is ~0us — applied here per-launch with no persistent kernel.
- **Make it size-adaptive:** small batch → mapped output + poll; large batch → device buffer + bulk DtoH
  (host-mapped PCIe writes are slow for big results — the wave proved this; that's the whole reason it moved to
  a device ring + bulk DtoH for large batches). Pick a byte threshold to route.
- **Payoff:** gets small-batch complete toward ~0us — dodges the DtoH and the driver's blocking-wait wakeup
  latency. This is the structural way below the 1-sync floor without going persistent.
- **Catch:** memory ordering (host observes result writes only after the flag) — reuse the wave's validated
  `.sys`-release + acquire-poll. **Independent audit required.** Also fixes the same p99 tail as
  `read-path-lpb-levers.md` #3 (a non-blocking wait), so coordinate — one mechanism covers both.

### 3. Needles as a kernel parameter, not an HtoD buffer (submit side)
- **Current code:** submit leases a needles buffer (`9790` region) + HtoDs it (`9856`); the kernel reads it via
  the `needles_arg` pointer in the args array (`9824`).
- **Change:** cuLaunchKernel's arg space (`lib.rs:9813-9829`) holds ~4KB → up to ~1000 int4 needles inline.
  For small batches, pass needles as a launch parameter → removes one device-buffer lease + one HtoD enqueue.
- **Payoff:** meaningful at batch=1 where that HtoD is fixed driver overhead. **Catch:** keep the buffer path
  for batches above the param-space threshold.

### 4. Gate the event timing (cheap; PREREQUISITE for lever #1's sync #1 removal)
- **Current code:** `timed` always-on when the stream has events (`lib.rs:9842`); 2x `cuEventRecord`
  (`9881`,`9901`) + `cuEventElapsedTime` (`2734`). The elapsed read is *why* sync #1 must stay.
- **Change:** gate behind a flag / sample. **Payoff:** removes the records + the stalling elapsed read AND
  unlocks lever #1's sync #1 removal. (Same lever as `read-path-lpb-levers.md` #4 — do once.)

### 5. Dense per-needle emit in the kernel (enables lever #1's over-fetch)
- **Current code:** `atom.global.add` compaction (`lib.rs:9678-9688`).
- **Change:** unique index → thread *i* writes `result[i]` densely (presence flag), no atomic. Makes the count
  unnecessary (→ enables #1's over-fetch with no count sync), drops the `needle_indices` buffer (slot==needle),
  removes the host scatter. (Same lever as `read-path-lpb-levers.md` #1 — do once; gate vs the scan path.)

### 6. Trim / pre-stage the buffer leases (submit side)
- **Current code:** submit takes 5 device-buffer pool leases (`lib.rs:9790`…), each a pool mutex + bucket
  lookup.
- **Change:** with dense emit (#5) you need fewer (drop `needle_indices`, maybe `row_indices`/`count`). Add a
  dedicated pre-allocated batch=1 buffer set, or a fast-path that skips bucket logic for tiny sizes.
- **Payoff:** cuts per-lease overhead on the batch=1 critical path. Minor but real.

### 7. The irreducible launch floor (be honest)
- `cuLaunchKernel` (`lib.rs:9884`) is ~1-2us fixed overhead lpb pays **every** batch and cannot remove. This
  is the one place the persistent wave structurally wins batch=1 (no launch at all). For *concurrent* singles,
  multi-stream pipelining overlaps the launches (the A/B showed lpb scaling 840k→1.83M across threads), so
  small-batch *throughput under concurrency* is already handled; only a genuinely isolated single request hits
  this floor. After all host-side levers, lpb's batch=1 floor ≈ **launch + one poll**.

### 8. Coalescing is the production answer for throughput
- `DECISIONS.md` already established batch=1 single-flight is the only sub-SLO regime, and the coalescer
  batches concurrent point reads into one launch under load. So levers #1-#6 target single-request **latency**
  (a lightly-loaded shard's p99), not throughput. Don't position them as throughput-vs-SLO work.

---

## How they compose (recommended order for the implementing agent)

1. **#4 (gate timing)** → **#5 (dense unique emit)** → **#1 (collapse to 1 sync)** = the "untimed dense-unique
   single-sync" path. This alone takes complete from 3 round-trips to 1. (Requires the submission-type flag;
   independent audit on the kernel change.)
2. **#2 (zero-copy mapped output + poll)** on top → small-batch complete toward ~0us, and it doubles as the
   p99-tail fix (`read-path-lpb-levers.md` #3). Size-adaptive vs the large-batch DtoH path.
3. **#3 (needle-as-param)** + **#6 (fewer leases)** → trim the submit side.

After 1-3, lpb batch=1 is essentially *launch + one poll*; the only remaining structural gap to the wave is
the launch itself (lever #7).

**Strategic note:** these are latency wins; batch=1 *throughput* is sub-SLO only when un-coalesced, and the
coalescer already covers that under load. So this is p99-latency / lightly-loaded-shard polish — valuable, but
weigh against R3 (writes, the unmeasured SLO gate).

## Discipline (charter, non-negotiable)
GPU tests under `timeout`, NEVER `--gpu-reset`; ASCII-only PTX (`ptxas -arch=sm_70` check before launch);
**independent adversarial audit** on kernel/protocol changes (#1, #2, #5) — never self-audit; hazard-test
(repeated runs, zero CUDA 700/716/717) any host-mapped/polling change; commit/push/merge each verified
increment with trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; do NOT run a
GPU test right after a `timeout`-killed one. Coordinate with the active agent before touching shared
submit/complete code.
