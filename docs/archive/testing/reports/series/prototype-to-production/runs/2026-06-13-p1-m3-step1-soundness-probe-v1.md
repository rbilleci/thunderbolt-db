# P1-M3 step 1 — Real-GPU Snapshot Soundness Probe

Status: closed (acceptance gate 1 retired)
Date: 2026-06-13
Plan: `docs/roadmap/prototype-to-production-plan.md` Phase 1 / §8 / §9.3
Design: `docs/architecture/14-engine-snapshot-integration-design.md` (gate 1)
Branch: `phase0-m1-engine-facade`

## Goal

Retire the soundness crux of P1-M3 **before** touching any engine read-path
signatures: prove that GPU-resident device memory published in a snapshot generation
is **not** freed while a reader still holds that generation, and **is** freed after
the reader drains. This is the device-memory-lifetime risk the P1-M2 spike explicitly
could not exercise (it used a leaked-`'static` buffer). Doc 14 gate 1.

## What was built (all in `crates/execution`, additive)

1. **`unsafe impl Send + Sync for CudaResidentDeviceMemory`** with a precise
   `// SAFETY:` comment (`crates/execution/src/lib.rs`, just below the existing
   `CudaResidentDeviceMemoryReadView` impls). The owner is `!Send`/`!Sync` for exactly
   one reason — the raw `context: *mut c_void`; every other field is already
   thread-safe. Soundness rests on the publish-don't-mutate / free-only-at-`Drop`-
   after-refcount-drain discipline (see *Soundness argument* below). This is the
   load-bearing `unsafe` of the whole milestone.
2. **The probe**:
   `published_resident_generation_survives_a_replacement_publish_and_is_freed_after_drain`
   — `#[ignore = "requires a local NVIDIA driver and GPU"]`, added to
   `scripts/run_cuda_parity.sh`. It:
   - allocates a real `CudaResidentDeviceMemory` (g1) with a known payload and wraps it
     in a Drop-observing `ObservableResident`;
   - publishes it through `gpu_db_snapshot::SnapshotCell` (dev-dependency added);
   - on a **reader thread** pins g1, while the **main thread** publishes a replacement
     generation g2, and asserts g1 is *not* freed while held;
   - then, still holding g1 after g2 is current, does a **real cross-thread GPU read**
     of g1 (`submit_match_project_i32_equal_any_from_payload` → `complete_detached`)
     and asserts the correct rows come back — proving "not freed while held" at the
     **GPU level**, not merely by Rust refcount;
   - joins the reader and asserts g1 **is** freed once its last handle drops, while the
     current generation g2 stays alive.

The probe **cannot compile** unless `CudaResidentDeviceMemory: Send + Sync` (the
`Arc<SnapshotCell<Arc<ObservableResident>>>` must cross the thread boundary), so it is
both a witness and an exercise of change 1.

## Validation

```text
cargo fmt --check -p gpu_db_execution        → clean
cargo clippy -p gpu_db_execution --tests     → 0 new warnings (3 pre-existing
                                               "too many arguments" at lib.rs
                                               :202/:528/:3308, unrelated)
cargo test -p gpu_db_execution               → 22 passed; 0 failed; 10 ignored
probe (debug)   ×5  --include-ignored         → 5/5 passed
probe (release) ×3  --include-ignored         → 3/3 passed
```

Host: RTX PRO 6000 Blackwell Max-Q (97,887 MiB), driver 595.71.05, libcuda present.
The probe is **deterministic** — all ordering is enforced by a 2-party `Barrier` plus
`thread::join` (happens-before), not by timing; the GPU read is a synchronous CUDA
call sequence with no timing assertion.

## Soundness argument (what is and is not proven)

**Proven (gate 1):** under `SnapshotCell` publish-on-commit, a generation a reader
holds survives a concurrent writer publish, stays GPU-valid, and is reclaimed (real
`cu_mem_free`) only after its last reader drains — including the cross-thread case
where the owner is created on the main thread and dropped on the reader thread.

**NOT proven here (deferred, by design):**
- **Concurrent reads** (≥2 readers over one generation with no `&mut self`
  bottleneck) — that is gate 2, addressed when the read path flips to `&self`
  (step 3). The probe uses a **single** reader.
- **Context-currency safety of non-context-setting launch paths** — asserted in the
  SAFETY comment **by inspection**, not by the probe (the probe uses only the
  `submit`/`complete_detached` paths, which `cuCtxSetCurrent` themselves). On a thread
  with *no* context current, a non-setting launch returns `INVALID_CONTEXT` (safe);
  on a thread with a *foreign* context current it would misdirect the launch
  (unsafe). That hazard is a property of the per-allocation-context API — present with
  or without this `unsafe impl` — and is exactly what the shared-context milestone
  (§9.3) removes.

## Independent adversarial audit

An independent reviewer was tasked to **refute** (a) the `unsafe impl` soundness,
(b) that the probe is a real (non-vacuous) proof, (c) determinism, (d) that the docs
don't overclaim. Result: **no blocker**; all four claims upheld after genuine attack
(verified g1/g2 are distinct allocations/contexts, the kernel reads g1's `device_ptr`,
a freed g1 would panic at `cuCtxSetCurrent`, no stray `Arc` keeps g1 alive, 17/17
extra reruns). Findings fixed in this commit:

- **MAJOR (comment accuracy):** the SAFETY comment claimed un-current-context launches
  always return a safe `INVALID_CONTEXT`; corrected to distinguish *no* current context
  (safe error) from a *foreign* current context (misdirected launch), and to note the
  hazard predates and is independent of the `Send`/`Sync` impl. Also marked the
  sub-claim as by-inspection.
- **MINOR:** added this run report (it was referenced before it existed); reaffirmed in
  doc 14 gate 1 that the result is single-reader liveness, not concurrency.
- **NIT:** reworded the Drop-observer comment (the flag flips as `Drop` is *entered*,
  immediately before the real free on the same synchronous path); annotated the
  intentional `SnapshotCell<Arc<owner>>` shape (mirrors the engine target).

## Benchmark note (§5.7)

No performance number — this milestone is additive and **mechanistically isolated**
from the M0 measured path (the engine is not yet wired to the snapshot substrate; the
new code is a `!`-gated test plus a trait-impl that changes no runtime behavior). The
latency payoff (collapsing M0's queue-wait term via concurrent `&self` reads) is
claimed at **step 4**, when the read path adopts the substrate and M0 is re-run with
the Phase-5 noise controls.

## Next (P1-M3 continued)

2. Make per-table residency a `SnapshotCell<Arc<owner>>`; publish-on-commit instead of
   in-place free (also fixes the global stop-the-world invalidation at
   `engine/lib.rs:9017`). Target the shared-context end-state (§9.3), not deeper
   per-allocation-context coupling.
3. Flip `execute_relational_select` + the resident-route methods from `&mut self` to
   `&self` over a loaded generation (gate 2: concurrent reads execute).
4. Re-run M0 with median-of-N + CI and show the queue-wait term drop (first milestone
   that may claim a real latency improvement).
