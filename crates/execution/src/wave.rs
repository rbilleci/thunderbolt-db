//! R2 wave read engine (ADR-009) — the persistent-kernel point-lookup read path, lifted into the crate
//! from the proven standalone probes (`examples/wave_devatomic_probe.rs` 1d-i) so it runs on the engine's
//! REAL shared primary context instead of a throwaway one.
//!
//! A single persistent kernel is launched once over a resident int4 table's GPU hash index (the SAME R1
//! index format `(key<<32)|(row+1)`, Fibonacci hash, 256-probe cap). Its worker threads drain a host-
//! pinned **device-mapped ring** of needle requests: each thread statically owns a GRID-STRIDE slice of the
//! index space (no claim counter, no block barriers — a barrier deadlock would evade the backstop and zombie
//! the shared context), hash-probes the index, **gathers up to `MAX_PROJECTIONS` int4 columns** at the
//! matched row (exactly like the R1 `gpu_db_resident_i32_index_probe` kernel), and writes a per-needle
//! result record: the BODY (row_index + values) into a **DEVICE result ring** (full device-memory bandwidth
//! — per-needle host-mapped record writes capped the drain ~7.5M), then a per-slot STATUS into a host-mapped
//! status ring. `submit_async` enqueues a wave and returns a `WaveTicket` immediately; `harvest`, once every
//! slot of the wave is marked done, bulk-`DtoH`s its bodies (on a separate stream) and returns
//! `CudaI32BatchProjectionRow`s — byte-identical to the R1 index probe, one row per found needle. This
//! device-result path lifts the large-batch drain to ~30M (1.25-1.35x the launch-per-batch index probe);
//! smaller batches beat it ~1.9-2.5x. The blocking `submit` is `submit_async` + spin-`harvest`.
//!
//! ## Completion gate (P2): PER-SLOT status ring, sound for depth-K pipelining
//! Per processed index the kernel writes the record body to `res_dev`, then `st.release.sys` writes the slot
//! STATUS (1=found, 2=not-found) into a host-mapped STATUS RING (`idx & ring_mask`). The release orders the
//! body BEFORE the status, so a host that observes `status[slot] != 0` may safely DtoH that slot's body.
//! `harvest(ticket)` is ready iff EVERY slot of `[base, base+len)` is non-zero — a PER-SLOT signal, sound for
//! ANY number of in-flight waves harvested in ANY order (depth-K pipelining), unlike a cumulative counter
//! (which a later wave's indices can push past an earlier wave's range while a slot there is still unwritten).
//! `submit_async` CLEARS a wave's status slots to 0 before publishing `head`, so a reused ring slot's prior
//! status can't be read as a false-ready. CROSS-ENGINE ORDERING NOTE: the body lives in device memory and is
//! read by the DtoH copy engine on a separate stream (no event edge to the worker kernel); body visibility
//! rests on `.sys`-scope release flushing the body ahead of the host-visible status, plus `harvest`'s
//! `cuStreamSynchronize` — the same cross-engine cumulativity property the device-result design relies on,
//! EMPIRICALLY VALIDATED by the byte-identical stale-DtoH stress gate + the depth-K reused-slot test (a stale
//! or torn body would mismatch). CALLER CONTRACT: bound total un-harvested needles to `ring_capacity` (else
//! the circular ring overwrites an un-read wave) and harvest each ticket exactly once.
//!
//! ## Grid-stride claim (no claim counter, no CAS) + thread-0 coordinator
//! Each thread STATICALLY owns indices `tid, tid+T, tid+2T, ...` (T = total launched threads); index `i` is
//! processed by exactly thread `i mod T`, exactly once. There is **no claim counter and no CAS**, so claim
//! contention is eliminated and the drain no longer collapses past ~512 threads (the prior clamped-`atom.cas`
//! variant capped ~3.2M at 128 threads and DEGRADED/timed out beyond that). **Only thread 0 touches the
//! host-mapped ctrl block**: it mirrors host `doorbell`+`head` into DEVICE memory (which every other worker
//! polls — device reads, no PCIe). Reason: thousands of workers polling the host-mapped ctrl over PCIe
//! congest the bus and starve the host's `head` write -> waves never start -> timeouts. The kernel targets
//! `sm_70` for `st.release.sys` (also drops the bare-`membar.sys` cumulativity assumption an earlier audit
//! flagged); the `claim_batch`/`all_done` params are vestigial (unused by the per-slot kernel).
//!
//! ## Safety net (the `--gpu-reset`-denied-box rule, proven in probe 1a)
//! The kernel ALWAYS self-terminates by ANY of: (1) the host doorbell (clean `shutdown`/`Drop`); (2) a
//! crash-safe WATCHDOG (`watchdog_ns` > 0) — a host petter thread increments a heartbeat ~1/s; if it goes
//! stale (the host died/was SIGKILLed) thread 0 rings the doorbell so the kernel exits within ~`watchdog_ns`
//! instead of zombie-ing on the shared box; (3) a `%globaltimer` absolute backstop (`backstop_ns`) as the
//! final net. A LONG-LIVED engine-owned kernel uses the watchdog (so it never dies mid-operation yet exits
//! fast on crash); standalone tests pass `watchdog_ns=0` and rely on the fixed backstop + explicit shutdown.
//! SM-coexistence rule (DECISIONS ADR-008): keep the persistent footprint small; `threads` is a modest knob.
//! ASCII-only PTX (the driver JIT rejects non-ASCII even where the local ptxas tolerates it).
//!
//! R2.2a = the data plane with multi-column projection, verified in ISOLATION vs the R1 index-probe
//! oracle; NOT wired into any engine query path yet (that is R2.2b).

// TODO(R2.2b): drop this once the engine wiring consumes `WaveReadEngine`. Until then the type is used
// only by its `#[cfg(test)]` GPU test, so a non-test build sees it as dead.
#![allow(dead_code)]

use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use libloading::Library;

use crate::{
    check_cuda, CudaI32BatchProjectionColumns, CudaI32BatchProjectionRow, CudaResidentDeviceMemory,
    CudaResidentReadSource, CudaRuntimeProbeError, GpuPrimaryContext,
};

const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CU_STREAM_NON_BLOCKING: u32 = 0x01;
/// Control block bytes (device-mapped pinned): [doorbell@0 u32, head@8 u64]. The host publishes `head` (u64,
/// cumulative; u32 would wrap at ~2^32 lookups) at `HEAD_OFFSET`; thread 0 mirrors doorbell+head into device
/// memory for the workers. Completion is the PER-SLOT status ring (P2), not a counter in this block.
const CTRL_BYTES: usize = 128;
/// Byte offset of the host-mapped cumulative `head` (u64, 8-aligned).
const HEAD_OFFSET: usize = 8;
/// Byte offset of the host-mapped crash-safe WATCHDOG heartbeat (u32). The host petter increments it; thread
/// 0 exits the kernel if it goes stale > `watchdog_ns` (host died/hung). 0 elsewhere in the block.
const HEARTBEAT_OFFSET: usize = 16;
/// Max projected int4 columns per needle (matches the R1 index probe's `MAX_PROJECTIONS`).
const MAX_PROJECTIONS: usize = 4;
/// Per-needle result record (device-mapped): [status@0 u32, pad@4, row@8 u64, v0@16, v1@20, v2@24, v3@28].
/// status: 0 = incomplete (bug), 1 = found, 2 = not found.
const RES_SLOT_BYTES: usize = 32;
/// Grid-stride indices a thread processes between completion flushes. Amortizes the `membar.sys` + the
/// `atom.add(completed)` over up to K indices (the probe's sweet spot was K=8). Overridable for tuning via
/// `GPU_DB_WAVE_CLAIM_BATCH`.
const CLAIM_BATCH: u32 = 8;

/// Handle to an in-flight `submit_async` wave: its cumulative ring range `[base, base+len)`. `harvest`
/// turns it into rows once every slot of the range is marked done in the status ring. `Copy` so the host can
/// queue many in flight (depth-K) and harvest them in any order.
#[derive(Clone, Copy, Debug)]
pub struct WaveTicket {
    /// Cumulative u64 (never wraps for the engine's lifetime; u32 would wrap at ~2^32 lookups). `head`/`idx`
    /// stay u64 in the kernel for the same reason (the `idx < head` compare).
    base: u64,
    len: u32,
}
/// Host-side timeout waiting for a wave to drain (the kernel's own backstop is the device-side net).
const DRAIN_TIMEOUT: Duration = Duration::from_secs(20);

// FFI signatures resolved ad-hoc from the primary context's `Library` (the crate idiom).
// `cuMemHostGetDevicePointer` is not bound as a `GpuPrimaryContext` field; the device-mapped buffers need
// it to hand the kernel a device pointer.
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
// Async DtoH on a SEPARATE stream so it never blocks on (or device-syncs against) the persistent kernel's
// stream. Used to bulk-copy a drained wave's records from a DEVICE result buffer to host.
type CuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32;
// Persistent-grid occupancy invariant (C1): EVERY launched block of a grid-stride persistent kernel must be
// co-resident, else its indices are never drained -> `completed` stalls -> 30s-backstop hang + empty rows.
// Used at construction to clamp `threads` to `maxActiveBlocksPerSM * SM_count` blocks.
type CuCtxGetDevice = unsafe extern "C" fn(*mut i32) -> i32;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
type CuOccupancyMaxActiveBlocksPerMultiprocessor =
    unsafe extern "C" fn(*mut i32, *mut c_void, i32, usize) -> i32;
/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type CuMemHostGetDevicePointer = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> i32;
type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> i32;
type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
type CuStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
type CuStreamDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
#[allow(clippy::type_complexity)]
type CuLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

/// The persistent multi-projection data-plane kernel (P2 per-slot status gate). Each thread processes a
/// GRID-STRIDE slice of the index space (`idx = tid, tid+T, ...`; no claim counter, no CAS), per index doing
/// R1's 4-way-unrolled multi-column gather, writing the record body (row, v0..v3) to the DEVICE result ring,
/// then a `st.release.sys` of the per-slot STATUS (1=found, 2=not-found) into a host-mapped STATUS RING. The
/// release orders the body BEFORE the status, so when the host observes a slot's status != 0 it may safely
/// DtoH that slot's body -- a PER-SLOT completion signal sound for ANY number of in-flight waves (depth-K
/// pipelining), unlike a cumulative counter. Thread 0 is the coordinator: it mirrors host doorbell+head into
/// device memory so the other workers never poll the host-mapped ctrl over PCIe. `.target sm_70` for
/// release/acquire (also drops the bare-`membar.sys` cumulativity assumption an earlier audit flagged). ASCII.
const WAVE_DATAPLANE_PTX: &[u8] = br#"
.version 6.0
.target sm_70
.address_size 64

.visible .entry gpu_db_wave_read_dataplane(
    .param .u64 ctrl,
    .param .u64 counters,
    .param .u64 req,
    .param .u64 res,
    .param .u64 status,
    .param .u64 resident,
    .param .u64 index,
    .param .u32 hash_shift,
    .param .u32 table_mask,
    .param .u32 ring_mask,
    .param .u32 proj_count,
    .param .u64 proj_off0,
    .param .u64 proj_off1,
    .param .u64 proj_off2,
    .param .u64 proj_off3,
    .param .u32 claim_batch,
    .param .u64 max_ns,
    .param .u64 watchdog_ns
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<64>;

    ld.param.u64 %rd1, [ctrl];
    ld.param.u64 %rd2, [counters];
    ld.param.u64 %rd3, [req];
    ld.param.u64 %rd4, [res];
    ld.param.u64 %rd37, [status];
    ld.param.u64 %rd5, [resident];
    ld.param.u64 %rd6, [index];
    ld.param.u32 %r1, [hash_shift];
    ld.param.u32 %r2, [table_mask];
    ld.param.u32 %r3, [ring_mask];
    ld.param.u32 %r4, [proj_count];
    ld.param.u64 %rd7, [proj_off0];
    ld.param.u64 %rd30, [proj_off1];
    ld.param.u64 %rd31, [proj_off2];
    ld.param.u64 %rd32, [proj_off3];
    ld.param.u32 %r5, [claim_batch];
    ld.param.u64 %rd8, [max_ns];
    ld.param.u64 %rd46, [watchdog_ns];          // 0 = disabled; else exit if the host heartbeat goes stale
    mov.u64 %rd9, %globaltimer;
    mov.u32 %r29, 0;                            // thread-0 watchdog: last heartbeat value seen
    mov.u64 %rd45, %rd9;                        // thread-0 watchdog: %globaltimer at the last heartbeat change

    // Grid-stride claim (ONCE): each thread statically owns indices tid, tid+T, tid+2T, ... where T is the
    // TOTAL launched threads. There is NO claim counter and NO CAS -> zero claim contention -> throughput
    // scales with threads. (The clamped-CAS variant capped ~3.2M at 128 threads and DEGRADED past that on
    // CAS contention; 2048 threads timed out.) Indices are global/cumulative so sequential waves are
    // covered seamlessly: index i is processed by exactly thread (i mod T), exactly once.
    mov.u32 %r26, %ntid.x;
    mov.u32 %r23, %ctaid.x;
    mov.u32 %r27, %tid.x;                        // special regs must be moved before use in mad/mul
    mad.lo.u32 %r23, %r23, %r26, %r27;           // r23 = global tid = ctaid*ntid + tid
    mov.u32 %r24, %nctaid.x;
    mul.lo.u32 %r24, %r24, %r26;                // r24 = T = nctaid * ntid (actual total threads)
    cvt.u64.u32 %rd40, %r23;                     // rd40 = idx (u64 CUMULATIVE; u32 wraps at ~2^32 lookups)
    cvt.u64.u32 %rd41, %r24;                     // rd41 = T (u64 grid stride)

$L_loop:
    // Thread 0 is the SOLE accessor of the host-mapped ctrl block: it mirrors host doorbell+head into DEVICE
    // memory (counters+16, counters+8) for the other workers to poll. Every other thread polls ONLY device
    // memory. Reason: 1000s of threads polling the host-mapped ctrl over PCIe congest the bus and starve the
    // host's head write -> waves never start -> timeouts (the failure an audit traced; the limiter that
    // capped earlier variants at ~512 threads). Completion is now per-slot (the status ring), not a counter.
    setp.ne.u32 %p1, %r23, 0;
    @%p1 bra $L_poll;
    // Layout: ctrl [doorbell@0 u32, head@8 u64]; counters [dev_head@8 u64, dev_doorbell@16 u32].
    ld.volatile.global.u32 %r28, [%rd1];        // host doorbell (u32)
    st.volatile.global.u32 [%rd2+16], %r28;     // -> device doorbell mirror (counters+16)
    ld.volatile.global.u64 %rd36, [%rd1+8];     // host head (u64, ctrl+8)
    st.volatile.global.u64 [%rd2+8], %rd36;     // -> device head mirror (u64, counters+8)
    // CRASH-SAFE WATCHDOG (thread 0 only): the host petter increments the heartbeat (ctrl+16) ~1/s. If it
    // goes stale for > watchdog_ns the host died/hung (e.g. SIGKILL) -> set the device doorbell so ALL threads
    // self-terminate within ~watchdog_ns, instead of zombie-ing until max_ns on the --gpu-reset-denied box.
    setp.eq.u64 %p2, %rd46, 0;
    @%p2 bra $L_poll;                            // watchdog disabled (watchdog_ns == 0)
    ld.volatile.global.u32 %r30, [%rd1+16];      // host heartbeat
    setp.ne.u32 %p2, %r30, %r29;
    @%p2 bra $L_wd_pet;                           // heartbeat changed -> record + reset the staleness timer
    setp.eq.u32 %p2, %r29, 0;
    @%p2 bra $L_poll;                            // NOT PRIMED yet (no pet seen) -> never fire before the first
    mov.u64 %rd47, %globaltimer;                 //   pet (the staleness clock is armed by the first pet, so a
    sub.u64 %rd48, %rd47, %rd45;                 //   slow petter start can't spuriously kill a healthy kernel)
    setp.lt.u64 %p2, %rd48, %rd46;
    @%p2 bra $L_poll;                            // not yet stale
    mov.u32 %r28, 1;
    st.volatile.global.u32 [%rd2+16], %r28;      // STALE -> ring the device doorbell (all threads exit)
    bra $L_poll;
$L_wd_pet:
    mov.u32 %r29, %r30;                          // record the pet (>= 1, so r29 != 0 == primed)
    mov.u64 %rd45, %globaltimer;
$L_poll:
    ld.volatile.global.u32 %r6, [%rd2+16];      // device doorbell mirror (counters+16; device read, no PCIe)
    setp.ne.s32 %p1, %r6, 0;
    @%p1 bra $L_flush_exit;
    mov.u64 %rd10, %globaltimer;
    sub.u64 %rd11, %rd10, %rd9;
    setp.ge.u64 %p1, %rd11, %rd8;
    @%p1 bra $L_flush_exit;                      // wall-clock backstop -> flush pending + exit
    ld.volatile.global.u64 %rd42, [%rd2+8];     // device head mirror (u64, counters+8; device read, no PCIe)
    setp.lt.u64 %p1, %rd40, %rd42;              // idx(u64) < head(u64) -> needle ready
    @%p1 bra $L_have;
    // No needle for idx yet: back off on the wall clock (NOT a tight head re-poll: many idle threads polling
    // head over PCIe starve the host's head write). Per-slot statuses are released per index -> nothing to flush.
$L_backoff:
    mov.u64 %rd33, %globaltimer;
$L_backoff_spin:
    mov.u64 %rd34, %globaltimer;
    sub.u64 %rd35, %rd34, %rd33;
    setp.lt.u64 %p1, %rd35, 1024;
    @%p1 bra $L_backoff_spin;
    bra $L_loop;

$L_have:
    cvt.u32.u64 %r12, %rd40;                     // low 32 bits of idx(u64) -> ring slot addressing
    and.b32 %r13, %r12, %r3;
    mul.wide.u32 %rd12, %r13, 4;
    add.u64 %rd13, %rd3, %rd12;
    ld.volatile.global.u32 %r14, [%rd13];
    mul.wide.u32 %rd14, %r13, 32;
    add.u64 %rd15, %rd4, %rd14;
    mul.lo.u32 %r15, %r14, 2654435761;
    shr.u32 %r15, %r15, %r1;
    mov.u32 %r16, 0;
    mov.u32 %r17, 2;

$L_probe:
    and.b32 %r15, %r15, %r2;
    mul.wide.u32 %rd16, %r15, 8;
    add.u64 %rd17, %rd6, %rd16;
    ld.global.u64 %rd18, [%rd17];
    setp.eq.u64 %p3, %rd18, 0;
    @%p3 bra $L_write;
    shr.u64 %rd19, %rd18, 32;
    cvt.u32.u64 %r18, %rd19;
    setp.ne.s32 %p3, %r18, %r14;
    @%p3 bra $L_probe_next;
    cvt.u32.u64 %r19, %rd18;
    sub.u32 %r19, %r19, 1;
    cvt.u64.u32 %rd20, %r19;
    mov.u32 %r17, 1;
    st.global.u64 [%rd15+8], %rd20;
    mul.lo.u64 %rd21, %rd20, 4;
    add.u64 %rd22, %rd5, %rd7;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+16], %r20;
    setp.le.u32 %p4, %r4, 1;
    @%p4 bra $L_write;
    add.u64 %rd22, %rd5, %rd30;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+20], %r20;
    setp.le.u32 %p4, %r4, 2;
    @%p4 bra $L_write;
    add.u64 %rd22, %rd5, %rd31;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+24], %r20;
    setp.le.u32 %p4, %r4, 3;
    @%p4 bra $L_write;
    add.u64 %rd22, %rd5, %rd32;
    add.u64 %rd22, %rd22, %rd21;
    ld.global.s32 %r20, [%rd22];
    st.global.s32 [%rd15+28], %r20;
    bra $L_write;

$L_probe_next:
    add.u32 %r15, %r15, 1;
    add.u32 %r16, %r16, 1;
    setp.ge.u32 %p4, %r16, 256;
    @%p4 bra $L_write;
    bra $L_probe;

$L_write:
    // RELEASE the per-slot status into the host-mapped STATUS RING. release.sys orders the record body
    // (row/v0.. -> res_dev, stored above) BEFORE this store, so the host observing status[slot]!=0 may safely
    // DtoH that slot's body. rd12 = slot*4 (computed at $L_have; u32 status, same stride as the u32 needle).
    // Per-slot signal -> sound for any number of in-flight waves (depth-K pipelining), no cumulative counter.
    add.u64 %rd23, %rd37, %rd12;
    st.release.sys.global.u32 [%rd23], %r17;
    add.u64 %rd40, %rd40, %rd41;                 // grid-stride advance: idx(u64) += T(u64)
    bra $L_loop;

$L_flush_exit:
$L_done:
    ret;
}
"#;

/// Resolve a CUDA driver symbol from `lib` (trying each name in order) and COPY OUT the bare fn pointer.
///
/// # Safety
/// The returned fn pointer is valid only while `lib` stays loaded. The caller stores it inside a
/// `WaveReadEngine` that also holds the `Arc<GpuPrimaryContext>` (hence the `Arc<Library>`), so the
/// library outlives every stored pointer.
unsafe fn sym<T: Copy>(lib: &Library, names: &[&[u8]]) -> Result<T, CudaRuntimeProbeError> {
    for name in names {
        if let Ok(symbol) = lib.get::<T>(name) {
            return Ok(*symbol);
        }
    }
    Err(CudaRuntimeProbeError::DriverLibraryUnavailable)
}

/// A persistent GPU "wave" read kernel resident on the engine's shared primary context, draining a
/// circular needle ring against one resident int4 table's hash index, projecting up to `MAX_PROJECTIONS`
/// columns. `submit` runs one wave at a time (serialized) and reads its rows back.
pub struct WaveReadEngine {
    /// The shared primary context the kernel + the index/table device buffers live in. Kept alive for the
    /// engine's lifetime AND re-set-current in `shutdown` so a cross-thread `Drop` (this type is `Send` for
    /// engine ownership) frees its context-bound CUDA resources against the right context.
    primary: Arc<GpuPrimaryContext>,
    _index: Arc<CudaResidentDeviceMemory>,
    _resident: Arc<CudaResidentDeviceMemory>,
    stream: *mut c_void,
    /// Device-mapped pinned control block (host view): [doorbell@0 u32, head@8 u64, completed_mirror@64 u64].
    ctrl_host: *mut c_void,
    /// Device-mapped pinned needle ring (host view): i32[ring_capacity].
    req_host: *mut c_void,
    /// Device-mapped pinned per-slot STATUS RING (host view): u32[ring_capacity]. The kernel `st.release.sys`
    /// writes status[slot] (1=found, 2=not-found) per index; `harvest` reads it locally (the per-slot
    /// completion gate, sound for depth-K pipelining); `submit_async` clears a wave's slots before publishing.
    status_host: *mut c_void,
    /// PINNED host staging buffer for the DtoH of a drained wave's records (RES_SLOT_BYTES * ring_capacity).
    /// No longer kernel-written (the kernel writes the DEVICE `res_dev` buffer); this is the DtoH target.
    res_host: *mut c_void,
    /// DEVICE result ring (the kernel writes records here at full device-memory bandwidth, vs per-needle
    /// host-mapped PCIe writes which capped the drain ~7.5M). `harvest` bulk-DtoHs the drained region.
    res_dev: u64,
    /// Separate stream for the result DtoH so it never queues behind the never-ending persistent kernel.
    copy_stream: *mut c_void,
    cu_memcpy_dtoh_async: CuMemcpyDtoHAsync,
    /// Device counters [completed@0 u64, dev_head@8 u64, dev_doorbell@16 u32] (monotonic; never reset).
    counters: u64,
    ring_capacity: usize,
    ring_mask: u32,
    proj_count: u32,
    /// Cumulative count of needles ever published (host-tracked; mirrors ctrl.head). Monotonic u64.
    head: u64,
    cu_stream_synchronize: CuStreamSynchronize,
    cu_stream_destroy: CuStreamDestroy,
    cu_mem_free: CuMemFree,
    cu_mem_free_host: CuMemFreeHost,
    /// Crash-safe watchdog: a background thread increments the host-mapped heartbeat ~1/s so the kernel keeps
    /// running; `shutdown` flips the stop flag + joins it. If the host process is SIGKILLed (no `shutdown`),
    /// the heartbeat stops, goes stale, and thread 0 self-terminates the kernel within ~`watchdog_ns`.
    heartbeat_stop: Arc<AtomicBool>,
    heartbeat_thread: Option<std::thread::JoinHandle<()>>,
    shut: bool,
}

// SAFETY: the raw pointers (host-mapped/device buffers, fn pointers) and `head` are owned by the engine and
// only accessed through `&self`/`&mut self`, which the engine crate serializes behind an `Arc<Mutex<_>>`
// (one connection thread at a time). The petter thread only touches its own captured ctrl address + the
// `Arc<AtomicBool>` stop flag, never the struct. So the engine can be MOVED across threads (Send); concurrent
// access is prevented by the Mutex (which provides Sync). It is deliberately NOT `Sync`.
unsafe impl Send for WaveReadEngine {}

impl WaveReadEngine {
    /// Launch the persistent kernel over `index` (built with `table_mask`/`hash_shift`) and `resident`
    /// (the table whose columns are gathered at `projection_offsets[j] + row*4`), on the SAME shared
    /// context the `index` lives in. `projection_offsets` must be 1..=MAX_PROJECTIONS. `ring_capacity` is
    /// rounded up to a power of two and bounds a single wave. `threads` is the persistent grid (kept
    /// modest per the SM-coexistence rule). Runs until `shutdown`/`Drop`, or `backstop_ns` elapses, or (if
    /// `watchdog_ns` > 0) the host heartbeat goes stale for `watchdog_ns` (crash-safe self-termination;
    /// spawns a petter thread). Pass `watchdog_ns = 0` to disable the watchdog (fixed-backstop only).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: Arc<CudaResidentDeviceMemory>,
        resident: Arc<CudaResidentDeviceMemory>,
        projection_offsets: &[u64],
        table_mask: u32,
        hash_shift: u32,
        ring_capacity: usize,
        threads: u32,
        backstop_ns: u64,
        watchdog_ns: u64,
    ) -> Result<Self, CudaRuntimeProbeError> {
        if projection_offsets.is_empty() || projection_offsets.len() > MAX_PROJECTIONS {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                projection_offsets.len(),
            ));
        }
        let proj_count = projection_offsets.len() as u32;
        let mut proj = [0u64; MAX_PROJECTIONS];
        proj[..projection_offsets.len()].copy_from_slice(projection_offsets);

        let ring_capacity = ring_capacity.next_power_of_two().max(8);
        let ring_mask = (ring_capacity - 1) as u32;
        let threads = threads.max(1);

        let primary = index.primary_arc();
        primary.set_current()?;
        let lib = primary.lib();

        let cu_mem_alloc: CuMemAlloc = unsafe { sym(lib, &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"]) }?;
        let cu_mem_free: CuMemFree = unsafe { sym(lib, &[b"cuMemFree_v2\0", b"cuMemFree\0"]) }?;
        let cu_memcpy_htod: CuMemcpyHtoD =
            unsafe { sym(lib, &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"]) }?;
        let cu_memcpy_dtoh_async: CuMemcpyDtoHAsync =
            unsafe { sym(lib, &[b"cuMemcpyDtoHAsync_v2\0", b"cuMemcpyDtoHAsync\0"]) }?;
        let cu_mem_host_alloc: CuMemHostAlloc = unsafe { sym(lib, &[b"cuMemHostAlloc\0"]) }?;
        let cu_mem_host_get_device_pointer: CuMemHostGetDevicePointer = unsafe {
            sym(
                lib,
                &[
                    b"cuMemHostGetDevicePointer_v2\0",
                    b"cuMemHostGetDevicePointer\0",
                ],
            )
        }?;
        let cu_mem_free_host: CuMemFreeHost = unsafe { sym(lib, &[b"cuMemFreeHost\0"]) }?;
        let cu_stream_create: CuStreamCreate = unsafe { sym(lib, &[b"cuStreamCreate\0"]) }?;
        let cu_stream_synchronize: CuStreamSynchronize =
            unsafe { sym(lib, &[b"cuStreamSynchronize\0"]) }?;
        let cu_stream_destroy: CuStreamDestroy =
            unsafe { sym(lib, &[b"cuStreamDestroy_v2\0", b"cuStreamDestroy\0"]) }?;
        let cu_launch_kernel: CuLaunchKernel = unsafe { sym(lib, &[b"cuLaunchKernel\0"]) }?;
        let cu_ctx_get_device: CuCtxGetDevice = unsafe { sym(lib, &[b"cuCtxGetDevice\0"]) }?;
        let cu_device_get_attribute: CuDeviceGetAttribute =
            unsafe { sym(lib, &[b"cuDeviceGetAttribute\0"]) }?;
        let cu_occupancy: CuOccupancyMaxActiveBlocksPerMultiprocessor =
            unsafe { sym(lib, &[b"cuOccupancyMaxActiveBlocksPerMultiprocessor\0"]) }?;

        let mut ptx = Vec::with_capacity(WAVE_DATAPLANE_PTX.len() + 1);
        ptx.extend_from_slice(WAVE_DATAPLANE_PTX);
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_wave_read_dataplane", &ptx)?;

        let resident_ptr = resident.device_ptr();
        let index_ptr = index.device_ptr();
        let mut threads_per_block = threads.min(256);
        // C1 (persistent-grid occupancy invariant, audit): a grid-stride persistent kernel needs EVERY
        // launched block co-resident -- a block that doesn't fit at launch is never scheduled, so the
        // indices it statically owns are never drained and `completed` never reaches `head` (30s-backstop
        // hang + empty rows). Clamp `threads` so the grid fits: blocks <= maxActiveBlocksPerSM * SM_count.
        let mut threads = threads;
        {
            let mut device: i32 = 0;
            let mut sm_count: i32 = 0;
            let mut max_blocks_per_sm: i32 = 0;
            check_cuda(unsafe { cu_ctx_get_device(&mut device) })?;
            check_cuda(unsafe {
                cu_device_get_attribute(
                    &mut sm_count,
                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    device,
                )
            })?;
            check_cuda(unsafe {
                cu_occupancy(&mut max_blocks_per_sm, function, threads_per_block as i32, 0)
            })?;
            if sm_count > 0 && max_blocks_per_sm > 0 {
                let capacity_blocks = (max_blocks_per_sm as u32).saturating_mul(sm_count as u32);
                let capacity_threads = capacity_blocks.saturating_mul(threads_per_block);
                if threads > capacity_threads {
                    threads = capacity_threads;
                    threads_per_block = threads.min(256);
                }
            }
        }
        let blocks = threads.div_ceil(256);

        let mut counters: u64 = 0;
        let mut ctrl_host: *mut c_void = ptr::null_mut();
        let mut req_host: *mut c_void = ptr::null_mut();
        let mut status_host: *mut c_void = ptr::null_mut();
        let mut res_host: *mut c_void = ptr::null_mut();
        let mut res_dev: u64 = 0;
        let mut stream: *mut c_void = ptr::null_mut();
        let mut copy_stream: *mut c_void = ptr::null_mut();
        let setup = (|| -> Result<(), CudaRuntimeProbeError> {
            // [completed@0 u64, dev_head@8 u64, dev_doorbell@16 u32] — dev_head/dev_doorbell are DEVICE-memory
            // mirrors of the host-mapped ctrl that thread 0 republishes each iter, so the other (1000s of)
            // worker threads poll DEVICE memory instead of hammering the host-mapped ctrl over PCIe (which
            // starves the host's head write -> timeouts). u64 completed/head: see CTRL_BYTES (no 2^32 wrap).
            check_cuda(unsafe { cu_mem_alloc(&mut counters, 32) })?;
            let zero = [0u64; 4];
            check_cuda(unsafe { cu_memcpy_htod(counters, zero.as_ptr().cast::<c_void>(), 32) })?;

            check_cuda(unsafe {
                cu_mem_host_alloc(&mut ctrl_host, CTRL_BYTES, CU_MEMHOSTALLOC_DEVICEMAP)
            })?;
            unsafe {
                for i in 0..(CTRL_BYTES / 4) {
                    ptr::write_volatile((ctrl_host as *mut u32).add(i), 0);
                }
            }
            check_cuda(unsafe {
                cu_mem_host_alloc(&mut req_host, ring_capacity * 4, CU_MEMHOSTALLOC_DEVICEMAP)
            })?;
            // Host-mapped per-slot STATUS RING (u32/slot). Kernel release-writes it; harvest reads it locally.
            check_cuda(unsafe {
                cu_mem_host_alloc(&mut status_host, ring_capacity * 4, CU_MEMHOSTALLOC_DEVICEMAP)
            })?;
            unsafe {
                ptr::write_bytes(status_host as *mut u8, 0, ring_capacity * 4);
            }
            check_cuda(unsafe {
                cu_mem_host_alloc(
                    &mut res_host,
                    ring_capacity * RES_SLOT_BYTES,
                    CU_MEMHOSTALLOC_DEVICEMAP,
                )
            })?;
            unsafe {
                ptr::write_bytes(res_host as *mut u8, 0, ring_capacity * RES_SLOT_BYTES);
            }
            // DEVICE result ring the kernel writes (full device bandwidth). harvest DtoHs the drained region.
            check_cuda(unsafe { cu_mem_alloc(&mut res_dev, ring_capacity * RES_SLOT_BYTES) })?;
            fence(Ordering::SeqCst);

            let mut ctrl_dptr: u64 = 0;
            let mut req_dptr: u64 = 0;
            let mut status_dptr: u64 = 0;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut ctrl_dptr, ctrl_host, 0) })?;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut req_dptr, req_host, 0) })?;
            check_cuda(unsafe { cu_mem_host_get_device_pointer(&mut status_dptr, status_host, 0) })?;

            check_cuda(unsafe { cu_stream_create(&mut stream, CU_STREAM_NON_BLOCKING) })?;
            check_cuda(unsafe { cu_stream_create(&mut copy_stream, CU_STREAM_NON_BLOCKING) })?;

            let mut a_ctrl = ctrl_dptr;
            let mut a_counters = counters;
            let mut a_req = req_dptr;
            let mut a_res = res_dev;
            let mut a_status = status_dptr;
            let mut a_resident = resident_ptr;
            let mut a_index = index_ptr;
            let mut a_shift = hash_shift;
            let mut a_tmask = table_mask;
            let mut a_rmask = ring_mask;
            let mut a_pcount = proj_count;
            let mut a_off0 = proj[0];
            let mut a_off1 = proj[1];
            let mut a_off2 = proj[2];
            let mut a_off3 = proj[3];
            let mut a_claim_batch = std::env::var("GPU_DB_WAVE_CLAIM_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(CLAIM_BATCH);
            let mut a_max = backstop_ns;
            let mut a_watchdog = watchdog_ns;
            let mut params = [
                (&mut a_ctrl as *mut u64).cast::<c_void>(),
                (&mut a_counters as *mut u64).cast::<c_void>(),
                (&mut a_req as *mut u64).cast::<c_void>(),
                (&mut a_res as *mut u64).cast::<c_void>(),
                (&mut a_status as *mut u64).cast::<c_void>(),
                (&mut a_resident as *mut u64).cast::<c_void>(),
                (&mut a_index as *mut u64).cast::<c_void>(),
                (&mut a_shift as *mut u32).cast::<c_void>(),
                (&mut a_tmask as *mut u32).cast::<c_void>(),
                (&mut a_rmask as *mut u32).cast::<c_void>(),
                (&mut a_pcount as *mut u32).cast::<c_void>(),
                (&mut a_off0 as *mut u64).cast::<c_void>(),
                (&mut a_off1 as *mut u64).cast::<c_void>(),
                (&mut a_off2 as *mut u64).cast::<c_void>(),
                (&mut a_off3 as *mut u64).cast::<c_void>(),
                (&mut a_claim_batch as *mut u32).cast::<c_void>(),
                (&mut a_max as *mut u64).cast::<c_void>(),
                (&mut a_watchdog as *mut u64).cast::<c_void>(),
            ];
            check_cuda(unsafe {
                cu_launch_kernel(
                    function,
                    blocks,
                    1,
                    1,
                    threads_per_block,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })?;
            Ok(())
        })();

        if let Err(err) = setup {
            unsafe {
                if !copy_stream.is_null() {
                    cu_stream_destroy(copy_stream);
                }
                if !stream.is_null() {
                    cu_stream_destroy(stream);
                }
                if res_dev != 0 {
                    cu_mem_free(res_dev);
                }
                if !res_host.is_null() {
                    cu_mem_free_host(res_host);
                }
                if !status_host.is_null() {
                    cu_mem_free_host(status_host);
                }
                if !req_host.is_null() {
                    cu_mem_free_host(req_host);
                }
                if !ctrl_host.is_null() {
                    cu_mem_free_host(ctrl_host);
                }
                if counters != 0 {
                    cu_mem_free(counters);
                }
            }
            return Err(err);
        }

        // Crash-safe watchdog petter: increment the host-mapped heartbeat ~3x per `watchdog_ns` window (cap
        // 1s) so the live kernel keeps running; `shutdown` stops + joins it. If this process is SIGKILLed the
        // petter dies with it, the heartbeat goes stale, and thread 0 self-terminates the kernel within
        // ~watchdog_ns. Pass the ctrl address as a usize (raw pointers are not Send); ctrl_host outlives the
        // thread (shutdown joins it before freeing). watchdog_ns == 0 disables the watchdog (no petter).
        let heartbeat_stop = Arc::new(AtomicBool::new(false));
        let heartbeat_thread = if watchdog_ns > 0 {
            let stop = Arc::clone(&heartbeat_stop);
            let hb_addr = (ctrl_host as usize) + HEARTBEAT_OFFSET;
            let interval = Duration::from_nanos((watchdog_ns / 3).max(1)).min(Duration::from_secs(1));
            Some(std::thread::spawn(move || {
                let hb = hb_addr as *mut u32;
                let mut counter: u32 = 1;
                while !stop.load(Ordering::Relaxed) {
                    unsafe { ptr::write_volatile(hb, counter) };
                    fence(Ordering::SeqCst); // make the heartbeat visible to the GPU over PCIe (as the head publish does)
                    // Never write 0: 0 is the "no pet seen yet" sentinel the kernel uses to stay un-armed
                    // before the first pet (skip it on the ~136-year u32 wrap).
                    counter = counter.wrapping_add(1);
                    if counter == 0 {
                        counter = 1;
                    }
                    std::thread::sleep(interval);
                }
            }))
        } else {
            None
        };

        Ok(Self {
            primary,
            _index: index,
            _resident: resident,
            stream,
            ctrl_host,
            req_host,
            status_host,
            res_host,
            res_dev,
            copy_stream,
            cu_memcpy_dtoh_async,
            counters,
            ring_capacity,
            ring_mask,
            proj_count,
            head: 0,
            cu_stream_synchronize,
            cu_stream_destroy,
            cu_mem_free,
            cu_mem_free_host,
            heartbeat_stop,
            heartbeat_thread,
            shut: false,
        })
    }

    /// Enqueue a wave of `needles` (each a key value) and return IMMEDIATELY (non-blocking) with a
    /// `WaveTicket`. Many tickets may be in flight (depth-K) and harvested in ANY order — the PER-SLOT status
    /// gate makes that sound (see `harvest`). `needle_index` in the harvested rows is the position within THIS
    /// `needles` slice.
    ///
    /// CALLER CONTRACT (currently UNENFORCED — debug-assert/track when this is wired into a query path, R2.2b):
    /// bound the total un-harvested needles across all in-flight tickets to `ring_capacity`, else the circular
    /// ring overwrites an un-read wave's slots; and harvest each ticket exactly once. A wave the kernel never
    /// finishes (it hit the doorbell/backstop mid-wave) leaves its slots unwritten forever, so a depth-K
    /// caller's `harvest` spin loop must carry its own deadline (the blocking `submit` already bounds this via
    /// `DRAIN_TIMEOUT`).
    pub fn submit_async(
        &mut self,
        needles: &[i32],
    ) -> Result<WaveTicket, CudaRuntimeProbeError> {
        if needles.is_empty() {
            return Ok(WaveTicket {
                base: self.head,
                len: 0,
            });
        }
        if needles.len() > self.ring_capacity {
            return Err(CudaRuntimeProbeError::InvalidInputLength(needles.len()));
        }
        let base = self.head;
        // Bulk-copy needles into the (contiguous modulo ring wraparound) slots, AND CLEAR this wave's status
        // slots to 0, BEFORE publishing head. The clear is mandatory for the per-slot gate: a slot reused from
        // an earlier wave still holds that wave's status (1/2); clearing to 0 makes `harvest` see "not yet
        // written" until the kernel release-writes this wave's status. (Host-local writes to the mapped rings.)
        let req = self.req_host as *mut i32;
        let status = self.status_host as *mut u32;
        let cap = self.ring_capacity;
        let start = (base as usize) & (self.ring_mask as usize);
        let n = needles.len();
        unsafe {
            if start + n <= cap {
                ptr::copy_nonoverlapping(needles.as_ptr(), req.add(start), n);
                ptr::write_bytes(status.add(start), 0, n);
            } else {
                let first = cap - start;
                ptr::copy_nonoverlapping(needles.as_ptr(), req.add(start), first);
                ptr::copy_nonoverlapping(needles.as_ptr().add(first), req, n - first);
                ptr::write_bytes(status.add(start), 0, first);
                ptr::write_bytes(status, 0, n - first);
            }
        }
        fence(Ordering::SeqCst);
        let new_head = base + needles.len() as u64;
        self.head = new_head;
        // publish cumulative head (u64) at ctrl+HEAD_OFFSET, after the needle writes are fenced
        unsafe {
            ptr::write_volatile(
                (self.ctrl_host as *mut u8).add(HEAD_OFFSET).cast::<u64>(),
                new_head,
            )
        };
        fence(Ordering::SeqCst);
        Ok(WaveTicket {
            base,
            len: needles.len() as u32,
        })
    }

    /// Non-blocking completion check + read for a `submit_async` ticket. The completion gate is PER-SLOT: the
    /// wave is done iff EVERY one of its slots' status (the kernel `st.release.sys`-writes 1=found/2=not-found
    /// into the host-mapped status ring per index) is non-zero. Host-local reads (~ns), no DtoH. This is sound
    /// for ANY number of in-flight waves (depth-K pipelining, harvest in any order): a per-slot signal, unlike
    /// a cumulative counter which a LATER wave's indices can push past this wave's range while a slot here is
    /// still unwritten. The kernel's release orders each slot's record body (-> `res_dev`) BEFORE its status,
    /// so observing status != 0 means the body is committed; the `Acquire` fence + `read_records`' DtoH then
    /// read it. If ready, returns the rows (found needles only, byte-identical to the R1 index probe); else
    /// `None` (a not-yet-written slot only delays; the caller retries). LIVENESS: thread 0's block must stay
    /// co-resident (the modest grids the occupancy clamp allows); eviction would stall, caught by the backstop.
    /// Per-row form (correctness tests / request-response callers). The HOT engine path calls
    /// `harvest_columnar` to skip the per-row `Vec` allocation (DECISIONS "Tail latency").
    pub fn harvest(
        &self,
        ticket: WaveTicket,
    ) -> Result<Option<Vec<CudaI32BatchProjectionRow>>, CudaRuntimeProbeError> {
        Ok(self.harvest_columnar(ticket)?.map(CudaI32BatchProjectionColumns::into_rows))
    }

    pub fn harvest_columnar(
        &self,
        ticket: WaveTicket,
    ) -> Result<Option<CudaI32BatchProjectionColumns>, CudaRuntimeProbeError> {
        if ticket.len == 0 {
            return Ok(Some(CudaI32BatchProjectionColumns {
                projection_count: self.proj_count as usize,
                ..Default::default()
            }));
        }
        let status_ring = self.status_host as *const u32;
        let mask = self.ring_mask as usize;
        for i in 0..ticket.len as usize {
            let slot = (ticket.base as usize).wrapping_add(i) & mask;
            if unsafe { ptr::read_volatile(status_ring.add(slot)) } == 0 {
                return Ok(None); // this slot not yet released by the kernel -> wave not complete
            }
        }
        fence(Ordering::Acquire); // observed-status -> read-body ordering (pairs with the kernel's release.sys)
        Ok(Some(self.read_records(ticket.base, ticket.len)))
    }

    /// Read the per-needle records for a fully-drained wave `[base, base+len)` into the COLUMNAR form (found
    /// needles only). Caller MUST have confirmed completion (the `harvest` gate) before calling.
    fn read_records(&self, base: u64, len: u32) -> CudaI32BatchProjectionColumns {
        // This is the only driver-touching step of the harvest path (the cuMemcpyDtoHAsync + sync below).
        // `WaveReadEngine` is `Send` and the engine drives `submit`/`harvest` from CONNECTION threads, which
        // are not the builder thread `new` set the context current on. cuMemcpy*/cuStreamSynchronize target
        // the calling thread's CURRENT context, so re-establish the primary context here (matching the
        // per-query path + `complete_detached`, which do the same). Best-effort (a failed set_current would
        // surface as the DtoH erroring). Harmless on drivers that auto-bind the primary for an unbound thread;
        // load-bearing for portability / multi-GPU / the multi-producer routing R2.2b-3 will exercise.
        let _ = self.primary.set_current();
        let cap = self.ring_capacity;
        let start = (base as usize) & (self.ring_mask as usize);
        let n = len as usize;
        let dst = self.res_host as *mut u8;
        let bytes = RES_SLOT_BYTES;
        // Bulk-DtoH the drained wave's bodies from the DEVICE result ring into the pinned host staging, on the
        // copy stream (so it never queues behind the never-ending persistent kernel). Two copies if the wave
        // wraps the ring; the staging lands the bodies contiguously at [0, n*32). The kernel's `st.release.sys`
        // of each slot's status (which this wave's gate already observed != 0) ordered that slot's body ahead
        // of it; `cu_stream_synchronize` then orders the copy before the host parse.
        unsafe {
            if start + n <= cap {
                (self.cu_memcpy_dtoh_async)(
                    dst.cast::<c_void>(),
                    self.res_dev + (start * bytes) as u64,
                    n * bytes,
                    self.copy_stream,
                );
            } else {
                let first = (cap - start) * bytes;
                (self.cu_memcpy_dtoh_async)(
                    dst.cast::<c_void>(),
                    self.res_dev + (start * bytes) as u64,
                    first,
                    self.copy_stream,
                );
                (self.cu_memcpy_dtoh_async)(
                    dst.add(first).cast::<c_void>(),
                    self.res_dev,
                    n * bytes - first,
                    self.copy_stream,
                );
            }
            (self.cu_stream_synchronize)(self.copy_stream);
        }
        // Found/not-found comes from the host-mapped STATUS RING (the kernel no longer writes status into the
        // device record); row/values come from the DtoH'd body. The gate already confirmed every slot != 0.
        let status_ring = self.status_host as *const u32;
        let proj = self.proj_count as usize;
        // COLUMNAR: append directly into the three flat arrays (no per-row `Vec`). `needle_index` here is the
        // slot position `i` (the wave writes one slot per submitted needle, in order).
        let mut values = Vec::new();
        let mut needle_indices = Vec::new();
        let mut row_indices = Vec::new();
        for i in 0..n {
            let slot = (base as usize).wrapping_add(i) & (self.ring_mask as usize);
            let status = unsafe { ptr::read_volatile(status_ring.add(slot)) };
            // C3 (audit, defense-in-depth): status 0 = slot the kernel never wrote. The gate must never admit
            // one; if it does, fail LOUDLY in tests instead of silently dropping the row (short results).
            debug_assert!(
                status != 0,
                "harvest read an UNWRITTEN slot (status=0) at i={i} -- completion gate admitted an undrained wave",
            );
            if status != 1 {
                continue; // 2 = not found (no row, like the R1 index probe)
            }
            let rec = unsafe { (dst as *const u8).add(i * RES_SLOT_BYTES) };
            let row_index = unsafe { ptr::read_unaligned(rec.add(8).cast::<u64>()) };
            needle_indices.push(i as u32);
            row_indices.push(row_index);
            for j in 0..proj {
                values.push(unsafe { ptr::read_unaligned(rec.add(16 + j * 4).cast::<i32>()) });
            }
        }
        CudaI32BatchProjectionColumns {
            values,
            needle_indices,
            row_indices,
            projection_count: proj,
            status: Vec::new(),
        }
    }

    /// Blocking single-wave submit (`submit_async` + spin-`harvest`). Per-row form for correctness tests and
    /// request-response callers; the HOT engine path calls `submit_columnar`. For THROUGHPUT use
    /// `submit_async`/`harvest` pipelined (depth K).
    pub fn submit(
        &mut self,
        needles: &[i32],
    ) -> Result<Vec<CudaI32BatchProjectionRow>, CudaRuntimeProbeError> {
        Ok(self.submit_columnar(needles)?.into_rows())
    }

    /// Blocking single-wave submit returning the COLUMNAR form (no per-row `Vec` — DECISIONS "Tail latency").
    pub fn submit_columnar(
        &mut self,
        needles: &[i32],
    ) -> Result<CudaI32BatchProjectionColumns, CudaRuntimeProbeError> {
        let ticket = self.submit_async(needles)?;
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        loop {
            if let Some(columns) = self.harvest_columnar(ticket)? {
                return Ok(columns);
            }
            if Instant::now() >= deadline {
                return Err(CudaRuntimeProbeError::KernelLaunchFailed(-1));
            }
            std::hint::spin_loop();
        }
    }

    /// Ring the doorbell, wait for the kernel to exit, then free the stream + buffers. Idempotent.
    pub fn shutdown(&mut self) {
        if self.shut {
            return;
        }
        self.shut = true;
        // `WaveReadEngine` is `Send` (engine ownership across connection threads), so `shutdown`/`Drop` can
        // run on a thread other than the one that built it (e.g. an eviction on the catalog-latch thread, or
        // an Arc overwrite on a read thread). cuStreamDestroy/cuMemFree/cuMemFreeHost are bound to the calling
        // thread's CURRENT context, which only `new` set (on the builder thread). Re-establish the primary
        // context current here so the frees below target the right context regardless of dropping thread;
        // best-effort (a failed set_current means the frees would fail anyway -- a teardown-path leak, not UB).
        let _ = self.primary.set_current();
        // Stop + join the watchdog petter BEFORE freeing ctrl_host (the petter writes the heartbeat there).
        self.heartbeat_stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.heartbeat_thread.take() {
            let _ = t.join();
        }
        unsafe { ptr::write_volatile(self.ctrl_host as *mut u32, 1) }; // doorbell
        fence(Ordering::SeqCst);
        unsafe {
            (self.cu_stream_synchronize)(self.stream); // kernel drained (doorbell) before any free
            (self.cu_stream_synchronize)(self.copy_stream);
            (self.cu_stream_destroy)(self.copy_stream);
            (self.cu_stream_destroy)(self.stream);
            (self.cu_mem_free)(self.res_dev);
            (self.cu_mem_free_host)(self.res_host);
            (self.cu_mem_free_host)(self.status_host);
            (self.cu_mem_free_host)(self.req_host);
            (self.cu_mem_free_host)(self.ctrl_host);
            (self.cu_mem_free)(self.counters);
        }
    }

    /// Test hook: simulate host death (SIGKILL) -- stop+join the watchdog petter WITHOUT a clean `shutdown`
    /// (no doorbell), then block on the kernel's stream and return how long it took to drain. If the watchdog
    /// works, the stale heartbeat makes thread 0 ring the doorbell within ~`watchdog_ns` -> fast drain; if it
    /// does not, the stream blocks until the fixed backstop. The kernel is dead afterward; `Drop` still cleans
    /// up (doorbell is idempotent, the petter is already joined).
    #[cfg(test)]
    pub(crate) fn simulate_host_death_then_wait_exit(&mut self) -> Duration {
        self.heartbeat_stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.heartbeat_thread.take() {
            let _ = t.join();
        }
        let t = Instant::now();
        unsafe {
            (self.cu_stream_synchronize)(self.stream);
        }
        t.elapsed()
    }
}

impl Drop for WaveReadEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDriverRuntime;

    #[test]
    fn wave_dataplane_ptx_is_pure_ascii() {
        // The driver JIT's ptxas rejects a non-ASCII byte (INVALID_PTX 218) even where the local ptxas
        // tolerates it, so a stray em-dash / smart-quote in a comment fails every launch. Keep it ASCII.
        if let Some(pos) = WAVE_DATAPLANE_PTX.iter().position(|&byte| !byte.is_ascii()) {
            let line = WAVE_DATAPLANE_PTX[..pos]
                .iter()
                .filter(|&&byte| byte == b'\n')
                .count()
                + 1;
            panic!("WAVE_DATAPLANE_PTX has a non-ASCII byte at offset {pos} (line {line})");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_point_lookups_match_index_probe_oracle() {
        // Build a tiny resident int4 table + its R1-format hash index on the SHARED primary context, run
        // the persistent wave engine over it with a 2-column projection, and assert every wave's rows are
        // BYTE-IDENTICAL to the R1 index probe (`submit_match_project_i32_index_probe_from_payload`). Two
        // waves exercise the cumulative ring.
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return; // no GPU -> skip (also #[ignore]d by default)
        };
        let rows: u64 = 1000;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &buf)
                .expect("resident device memory"),
        );
        let key_offset = 0_u64;
        let payload_offset = rows * 4;
        let projections = [key_offset, payload_offset];

        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &index_bytes)
                .expect("index device memory"),
        );

        // The R1 index probe is the oracle.
        let oracle = |needles: &[i32]| -> Vec<CudaI32BatchProjectionRow> {
            let submission = resident
                .submit_match_project_i32_index_probe_from_payload(
                    &index_resident,
                    table_mask,
                    hash_shift,
                    needles,
                    &projections,
                    rows,
                )
                .expect("index submit");
            let mut r = submission.complete(&resident).expect("index complete");
            r.sort_by_key(|row| (row.needle_index, row.row_index));
            r
        };

        let mut engine = WaveReadEngine::new(
            Arc::clone(&index_resident),
            Arc::clone(&resident),
            &projections,
            table_mask,
            hash_shift,
            1024,
            1024,
            30_000_000_000,
            0, // watchdog disabled (test relies on the fixed backstop + explicit shutdown)
        )
        .expect("wave engine launches on the shared context");

        let waves = [
            vec![keys[5], keys[100], keys[999], -12345, keys[0]],
            vec![keys[10], keys[20], 2, keys[30]],
        ];
        // Run all waves on the persistent kernel FIRST (no interleaved GPU launches), then validate vs
        // the R1 oracle — isolates multi-wave persistence from any concurrent-launch interaction.
        let mut results = Vec::new();
        for needles in &waves {
            let mut got = engine.submit(needles).expect("wave submit");
            got.sort_by_key(|row| (row.needle_index, row.row_index));
            results.push(got);
        }
        // Non-vacuous spot-check gather (also a 3rd wave) — collected BEFORE any GPU oracle launch.
        let spot = engine.submit(&[keys[5]]).expect("spot submit");
        engine.shutdown();

        for (got, needles) in results.iter().zip(waves.iter()) {
            let want = oracle(needles);
            assert_eq!(
                *got, want,
                "wave rows must be byte-identical to the R1 index probe for {needles:?}"
            );
        }
        assert_eq!(spot.len(), 1, "present key -> exactly one row");
        assert_eq!(
            spot[0].values,
            vec![keys[5], payload[5]],
            "2-column gather (key, payload) at row 5"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_depth_k_pipelined_out_of_order_matches_oracle() {
        // P2b: keep K waves IN FLIGHT (submit_async x K, no harvest between), then harvest them in REVERSE
        // order, asserting each is byte-identical to a CPU oracle, across many rounds that REUSE ring slots.
        // A cumulative-counter gate (a later wave's indices pushing it past an earlier wave's range) OR a
        // stale slot would mismatch here. Proves the per-slot status gate is sound for out-of-order depth-K
        // pipelining. (CPU oracle, not the R1 probe -> no cuMemAlloc while the kernel is live.)
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = 1000;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &buf)
                .expect("resident device memory"),
        );
        let projections = [0_u64, rows * 4]; // [key, payload]
        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &index_bytes)
                .expect("index device memory"),
        );

        let mut engine = WaveReadEngine::new(
            Arc::clone(&index_resident),
            Arc::clone(&resident),
            &projections,
            table_mask,
            hash_shift,
            1024,
            1024,
            30_000_000_000,
            0, // watchdog disabled (test relies on the fixed backstop + explicit shutdown)
        )
        .expect("wave engine");

        let k = 8usize; // waves in flight
        let wlen = 5usize; // needles per wave
        let rounds = 64usize; // > ring_capacity/(k*wlen) -> slots are reused many times
        let mut counter = 0usize;
        for round in 0..rounds {
            // Submit K waves; keep all in flight. needles map slot->row deterministically; distinct per wave.
            let mut inflight: Vec<(WaveTicket, usize)> = Vec::with_capacity(k);
            for _w in 0..k {
                let base_idx = counter;
                counter += wlen;
                let needles: Vec<i32> =
                    (0..wlen).map(|i| keys[(base_idx + i) % rows as usize]).collect();
                let ticket = engine.submit_async(&needles).expect("submit_async");
                inflight.push((ticket, base_idx));
            }
            // Harvest in REVERSE submission order (exercises out-of-order completion).
            for (ticket, base_idx) in inflight.into_iter().rev() {
                let deadline = Instant::now() + Duration::from_secs(10);
                let got = loop {
                    if let Some(r) = engine.harvest(ticket).expect("harvest") {
                        break r;
                    }
                    assert!(Instant::now() < deadline, "harvest timed out (round {round})");
                };
                assert_eq!(got.len(), wlen, "wave row count (round {round})");
                for (j, row) in got.iter().enumerate() {
                    let want_row = (base_idx + j) % rows as usize;
                    assert_eq!(row.needle_index, j, "needle_index (round {round})");
                    assert_eq!(row.row_index, want_row as u64, "row_index (round {round})");
                    assert_eq!(
                        row.values,
                        vec![keys[want_row], payload[want_row]],
                        "[key,payload] gather (round {round}, j {j}) -- stale/out-of-order slot?"
                    );
                }
            }
        }
        engine.shutdown();
    }

    #[test]
    fn ctrl_layout_offsets_match_the_kernel_contract() {
        // Pin the host<->PTX offset contract (audit suggestion): the kernel HARD-CODES ctrl head@8. If this
        // host constant drifts, the u64 head load misaligns or reads the wrong field. head must be 8-aligned
        // (u64) and its 8 bytes must fit in the ctrl block.
        assert_eq!(HEAD_OFFSET, 8, "kernel reads host head at ctrl+8");
        assert_eq!(HEAD_OFFSET % 8, 0, "u64 head must be 8-aligned");
        assert!(HEAD_OFFSET + 8 <= CTRL_BYTES, "head (8B) must fit in the ctrl block");
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_clamps_oversized_threads_no_hang() {
        // C1 (audit): a `threads` value far beyond the GPU's resident capacity must NOT silently hang. A
        // persistent grid-stride kernel needs EVERY launched block co-resident (an un-resident block's
        // statically-owned indices are never drained -> `completed` stalls below `head` -> 30s-backstop hang
        // + empty rows). `new()` clamps `threads` to the occupancy capacity, so construct + submit succeed.
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = 256;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &buf)
                .expect("resident device memory"),
        );
        let projections = [0_u64, rows * 4];
        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &index_bytes)
                .expect("index device memory"),
        );

        // 50M threads = ~195k blocks, vastly beyond any GPU's resident capacity. Without the C1 clamp the
        // un-resident blocks' indices never drain and this hangs to the 30s backstop; with it, it just works.
        let mut engine = WaveReadEngine::new(
            Arc::clone(&index_resident),
            Arc::clone(&resident),
            &projections,
            table_mask,
            hash_shift,
            1024,
            50_000_000,
            30_000_000_000,
            0, // watchdog disabled (test relies on the fixed backstop + explicit shutdown)
        )
        .expect("wave engine constructs with threads clamped to occupancy");

        let needles = vec![keys[1], keys[100], keys[200]];
        let mut got = engine
            .submit(&needles)
            .expect("submit must not hang with oversized threads (clamped grid stays resident)");
        engine.shutdown();
        got.sort_by_key(|row| (row.needle_index, row.row_index));
        assert_eq!(
            got.len(),
            3,
            "all present keys found -> the clamped grid drained every index"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_watchdog_self_terminates_on_host_death() {
        // R2.2b blocker #3: a long-lived engine-owned kernel can't use a fixed backstop (it would die mid-
        // operation) and must NOT zombie ~30s if the host is SIGKILLed (it perturbs other tenants on the
        // --gpu-reset-denied box). With a watchdog (500ms) + the host petter running, the kernel stays alive
        // and serves a submit; when we simulate host death (stop the petter, no clean shutdown), thread 0's
        // stale-heartbeat watchdog must self-terminate the kernel FAST (~watchdog_ns), not at the 30s backstop.
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = 256;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &buf)
                .expect("resident device memory"),
        );
        let projections = [0_u64, rows * 4];
        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(
            runtime
                .retain_device_memory_copy(0, &index_bytes)
                .expect("index device memory"),
        );

        // backstop 30s (the fixed net), watchdog 500ms (the responsive net) + the petter running.
        let mut engine = WaveReadEngine::new(
            Arc::clone(&index_resident),
            Arc::clone(&resident),
            &projections,
            table_mask,
            hash_shift,
            1024,
            1024,
            30_000_000_000,
            500_000_000,
        )
        .expect("wave engine");

        // Kernel is alive (petter keeps it so): a submit succeeds.
        let got = engine.submit(&[keys[7]]).expect("submit while watchdog-petted");
        assert_eq!(got.len(), 1, "present key found while the kernel is petted");

        // Let the petter run (interval ~watchdog_ns/3 = 167ms) so thread 0 sees >= 1 real pet and ARMS the
        // watchdog -- mirrors real use (the kernel runs petted for a long time before any crash). Without a
        // real pet the watchdog stays un-armed (by design: a never-petted kernel must not self-kill from
        // launch). The kernel must NOT have fired during this healthy window.
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            !engine.submit(&[keys[9]]).expect("kernel still alive while petted").is_empty(),
            "watchdog must NOT fire while the petter is alive"
        );

        // Simulate SIGKILL: stop the petter (no clean shutdown) and time the kernel's exit.
        let exit = engine.simulate_host_death_then_wait_exit();
        assert!(
            exit < Duration::from_secs(5),
            "watchdog must self-terminate the kernel within ~watchdog_ns (took {exit:?}); a fixed-backstop \
             kernel would block ~30s and zombie the shared box"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_arc_mutex_send_ownership() {
        // R2.2b-1: the engine owns the WaveReadEngine as `Arc<Mutex<WaveReadEngine>>` shared across connection
        // threads (raw pointers -> `unsafe impl Send`, sound because the Mutex serializes access). Validate the
        // model: build it, MOVE the Arc<Mutex> into another thread, submit there, get correct rows, drop clean.
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = 256;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(runtime.retain_device_memory_copy(0, &buf).expect("resident"));
        let projections = [0_u64, rows * 4];
        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(runtime.retain_device_memory_copy(0, &index_bytes).expect("index"));

        // backstop near-infinite + a live watchdog (2s) -- the engine-ownership configuration.
        let engine = Arc::new(std::sync::Mutex::new(
            WaveReadEngine::new(
                Arc::clone(&index_resident),
                Arc::clone(&resident),
                &projections,
                table_mask,
                hash_shift,
                1024,
                1024,
                u64::MAX,
                2_000_000_000,
            )
            .expect("wave engine"),
        ));

        // Move the shared handle into another thread (compile-checks + runtime-proves Send) and submit there.
        let worker = Arc::clone(&engine);
        let needles = vec![keys[5], keys[100], keys[200]];
        let mut got = std::thread::spawn(move || {
            let mut guard = worker.lock().unwrap_or_else(|p| p.into_inner());
            guard.submit(&needles).expect("submit from another thread")
        })
        .join()
        .expect("worker thread");
        got.sort_by_key(|row| (row.needle_index, row.row_index));
        assert_eq!(got.len(), 3, "present keys found via the shared Arc<Mutex> handle");
        for row in &got {
            assert_eq!(
                row.values,
                vec![keys[row.row_index as usize], payload[row.row_index as usize]]
            );
        }
        engine.lock().unwrap_or_else(|p| p.into_inner()).shutdown(); // clean teardown (joins the petter)
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_engine_drops_cleanly_on_a_foreign_thread() {
        // R2.2b Slice 2: `WaveReadEngine` is `Send` so the engine can DROP it (evict on DROP TABLE /
        // memory-pressure, or overwrite on re-admission) on a thread other than the builder. shutdown()/Drop
        // free context-bound CUDA resources (cuStreamDestroy/cuMemFree/cuMemFreeHost), which target the
        // DROPPING thread's CURRENT context -- only `new` set that, on the builder (here, main). Build on main,
        // then MOVE the bare engine into a fresh thread that NEVER set the context current and let it Drop
        // there. Reaching the end without a fault/hang IS the assertion. NOTE: this is a REGRESSION GUARD for
        // the cross-thread-Drop path; it is vacuous FOR the shutdown set_current fix on a driver that
        // auto-binds the primary context for an unbound thread (as this box's does -- teardown also succeeds
        // without the fix). The fix is still required on strict / multi-GPU drivers where an unbound thread
        // has no current context, so the frees would hit CUDA_ERROR_INVALID_CONTEXT.
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = 256;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(runtime.retain_device_memory_copy(0, &buf).expect("resident"));
        let projections = [0_u64, rows * 4];
        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(runtime.retain_device_memory_copy(0, &index_bytes).expect("index"));

        // engine-ownership config: near-infinite backstop + a live 2s watchdog.
        let mut engine = WaveReadEngine::new(
            Arc::clone(&index_resident),
            Arc::clone(&resident),
            &projections,
            table_mask,
            hash_shift,
            1024,
            1024,
            u64::MAX,
            2_000_000_000,
        )
        .expect("wave engine");
        // Prove it is live on the builder thread before handing it off.
        let got = engine.submit(&[keys[5]]).expect("submit on builder");
        assert_eq!(got.len(), 1, "engine live before foreign-thread drop");

        // Move the bare engine (Send) into a fresh thread that never set the context current; drop it there.
        std::thread::spawn(move || {
            drop(engine); // -> Drop -> shutdown -> set_current(primary) + context-bound frees
        })
        .join()
        .expect("foreign drop thread joined without panic");
        // No CUDA fault / hang above == the cross-thread teardown set the context current and freed cleanly.
    }

    /// R2.2 diagnostic — persistent wave vs launch-per-batch R1 index probe, swept over batch size.
    /// CAVEAT (DECISIONS ADR-008): this is SINGLE-FLIGHT submit-and-wait (blocking) — the wave's WORST case,
    /// NOT its intended pipelined/concurrent regime. Each timed wave is verified to return `batch` rows
    /// (in-loop adversarial check — a no-op/early harvest trips it instead of inflating throughput, the bug
    /// an independent audit found in the old gate). Verified picture (grid-stride drain, thread-0
    /// coordinator, DEVICE result ring + bulk DtoH harvest): the wave EXCEEDS lpb at EVERY batch size —
    /// ~1.7x at 1/8/256, ~1.85x at 32, and ~1.35x (~31.8M vs ~23.5M) at 65536, peaking around 8-16k threads.
    /// The earlier ~7.6M large-batch cap was the per-needle host-mapped record WRITE (not the host loops,
    /// not atom.add, not claim — K/thread/bulk-host-I/O sweeps were all flat); moving records to device
    /// memory + one DtoH lifted it 4.2x. Small batches pay ~6us of DtoH latency vs the old host-mapped read
    /// but still beat lpb. Wave is measured FIRST then shut down, THEN the index probe — so the index probe's
    /// `cuMemAlloc` (which device-syncs) never runs while the wave kernel is live (the freeze root cause).
    ///
    /// Run: `GPU_DB_WAVE_BENCH_ROWS=1000000 cargo test -p gpu_db_execution \
    ///   wave::tests::wave_vs_launch_per_batch_throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_vs_launch_per_batch_throughput() {
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = std::env::var("GPU_DB_WAVE_BENCH_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);
        // Persistent-kernel thread count (PCIe-polling congestion suspect — sweep it).
        // NOTE: this naive port is CONGESTION-bound (per-needle host-mapped record write + `membar.sys`
        // per needle + per-needle CAS) — MORE threads make it WORSE (256+ times out on large batches),
        // unlike the optimized probe (batched claiming + device atomics) which scaled to 8192. 128 is the
        // sweet spot here. The probe hit ~45M/s; this port plateaus ~520k/s (~85x slower) — see the
        // "R2.2 evidence gate CORRECTION" in DECISIONS: the wave engine was NOT fairly tested.
        let wave_threads: u32 = std::env::var("GPU_DB_WAVE_BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        // Adaptive iters so total work per batch size is ~constant (large single waves = continuous fill).
        let target_lookups: usize = 2_000_000;
        let keys: Vec<i32> = (0..rows as i32).map(|r| r.wrapping_mul(3).wrapping_add(1)).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r.wrapping_mul(1000).wrapping_add(7)).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(runtime.retain_device_memory_copy(0, &buf).expect("resident"));
        let payload_offset = rows * 4;
        let projections = [payload_offset]; // "SELECT payload WHERE key = ?"

        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(runtime.retain_device_memory_copy(0, &index_bytes).expect("index"));

        // SMALL realistic point-lookup batches (where lpb is launch-overhead-bound). The wave is drain-
        // bound (persistent kernel, no per-batch launch), so it sustains throughput as the batch shrinks
        // while lpb falls off; plus 65536 for the wave drain ceiling.
        let batch_sizes = [1usize, 8, 32, 256, 65536];
        let ring_capacity = 131_072usize;
        let iters_for = |batch: usize| (target_lookups / batch).clamp(20, 50_000);
        let needles_for = |batch: usize, iter: usize| -> Vec<i32> {
            (0..batch)
                .map(|i| keys[(iter.wrapping_mul(batch).wrapping_add(i)) % rows as usize])
                .collect()
        };

        // --- Persistent wave engine (alloc-free submits) ---
        let mut wave_lps = [0f64; 5];
        {
            let mut engine = WaveReadEngine::new(
                Arc::clone(&index_resident),
                Arc::clone(&resident),
                &projections,
                table_mask,
                hash_shift,
                ring_capacity,
                wave_threads,
                30_000_000_000,
                0, // watchdog disabled (benchmark relies on the fixed backstop + explicit shutdown)
            )
            .expect("wave engine");

            // CORRECTNESS + STALE-DtoH STRESS GATE (charter: adversarial verification, audit-driven).
            // The device-result ring is DtoH'd on a separate stream with no event edge to the worker kernel,
            // so a stale/torn copy would yield the right ROW COUNT but WRONG bytes (a prior wave's slot
            // contents). Two independent audits flagged that a count-only check can't catch this. So here we
            // run MANY waves at the FULL thread count with byte-identical CPU-oracle checks, REUSING ring
            // slots (needles rotate per `it`, so a stale read of a reused slot from an earlier wave carries
            // DIFFERENT values and mismatches). needles_for(b,it)=keys[(it*b+i)%rows], all present, so
            // row=(it*b+i)%rows and the projection ([payload_offset]) yields payload[row]. This is also the
            // discriminating experiment the auditors prescribed. CPU oracle (no cuMemAlloc -> no freeze).
            let verify = |it: usize, cb: usize, got: &[CudaI32BatchProjectionRow]| {
                assert_eq!(
                    got.len(),
                    cb,
                    "wave returned {} rows, expected {cb} (it={it} threads={wave_threads}) -- no-op/short harvest",
                    got.len()
                );
                for (k, row) in got.iter().enumerate() {
                    let want_row = (it.wrapping_mul(cb).wrapping_add(k)) % rows as usize;
                    assert!(
                        row.needle_index == k
                            && row.row_index == want_row as u64
                            && row.values.len() == 1
                            && row.values[0] == payload[want_row],
                        "STALE/WRONG DtoH record it={it} k={k} cb={cb} threads={wave_threads}: \
                         got (ni={}, row={}, vals={:?}), want (ni={k}, row={want_row}, val={})",
                        row.needle_index,
                        row.row_index,
                        row.values,
                        payload[want_row],
                    );
                }
            };
            // Heavy slot reuse (small batch, many waves) + the fast-drain race window (large batch).
            for it in 0..2000 {
                let got = engine.submit(&needles_for(256, it)).expect("verify submit 256");
                verify(it, 256, &got);
            }
            let big = 65536usize.min(ring_capacity);
            for it in 0..40 {
                let got = engine.submit(&needles_for(big, it)).expect("verify submit big");
                verify(it, big, &got);
            }

            for (bi, &batch) in batch_sizes.iter().enumerate() {
                let iters = iters_for(batch);
                // A timeout (Err) records 0 for this config rather than crashing (high thread counts can
                // congest the persistent kernel past the drain deadline — itself a finding).
                if (0..4).any(|w| engine.submit(&needles_for(batch, w)).is_err()) {
                    wave_lps[bi] = 0.0;
                    continue;
                }
                let t = Instant::now();
                let mut ok = true;
                for it in 0..iters {
                    match engine.submit(&needles_for(batch, it)) {
                        // EVERY timed wave MUST return `batch` rows (needles are all present keys). This
                        // bakes the charter's adversarial verification INTO the measurement: a no-op/early
                        // harvest (the gate-underflow bug the audit found) returns empty/short and trips
                        // here instead of silently inflating throughput. `black_box` stops the compiler
                        // eliding the `read_records` work.
                        Ok(rows) => {
                            assert_eq!(
                                rows.len(),
                                batch,
                                "TIMED wave returned {} rows, expected {batch} (no-op/early harvest) \
                                 threads={wave_threads}",
                                rows.len()
                            );
                            std::hint::black_box(&rows);
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                let secs = t.elapsed().as_secs_f64();
                wave_lps[bi] = if ok { (iters * batch) as f64 / secs } else { 0.0 };
            }
            engine.shutdown(); // MUST shut down before the index probe's cuMemAlloc (freeze root cause)
        }

        // --- Launch-per-batch R1 index probe (cuMemAlloc safe now: no wave kernel alive) ---
        let mut lpb_lps = [0f64; 5];
        for (bi, &batch) in batch_sizes.iter().enumerate() {
            let iters = iters_for(batch);
            for w in 0..4 {
                let s = resident
                    .submit_match_project_i32_index_probe_from_payload(
                        &index_resident, table_mask, hash_shift, &needles_for(batch, w), &projections, rows,
                    )
                    .expect("warmup submit");
                let _ = s.complete(&resident).expect("warmup complete");
            }
            let t = Instant::now();
            for it in 0..iters {
                let s = resident
                    .submit_match_project_i32_index_probe_from_payload(
                        &index_resident, table_mask, hash_shift, &needles_for(batch, it), &projections, rows,
                    )
                    .expect("lpb submit");
                let _ = s.complete(&resident).expect("lpb complete");
            }
            let secs = t.elapsed().as_secs_f64();
            lpb_lps[bi] = (iters * batch) as f64 / secs;
        }

        println!("\n# R2.2 gate: persistent WAVE vs LAUNCH-PER-BATCH index probe  rows={rows} wave_threads={wave_threads}");
        println!(
            "  {:>6}  {:>16}  {:>16}  {:>9}  {:>12}  {:>12}",
            "batch", "wave lookups/s", "lpb lookups/s", "speedup", "wave us/sub", "lpb us/call"
        );
        for (bi, &batch) in batch_sizes.iter().enumerate() {
            let speedup = if lpb_lps[bi] > 0.0 { wave_lps[bi] / lpb_lps[bi] } else { 0.0 };
            let wave_us = 1.0e6 / (wave_lps[bi] / batch as f64);
            let lpb_us = 1.0e6 / (lpb_lps[bi] / batch as f64);
            println!(
                "  {batch:>6}  {:>16.0}  {:>16.0}  {speedup:>8.2}x  {wave_us:>11.2}  {lpb_us:>11.2}",
                wave_lps[bi], lpb_lps[bi]
            );
        }
        println!("# large batch = continuous-fill (the wave's intended regime); batcher ~156k single-coalescer cap.");
    }

    /// P2c — the PREMISE GATE for the concurrent regime the wave engine exists for. Single-flight only proves
    /// wave > lpb across the sweep; the wave's real advantage is keeping the persistent kernel CONTINUOUSLY
    /// fed by overlapping host submit/harvest with the GPU drain (depth-K pipelining), which the per-slot gate
    /// (P2) now allows. This sweeps (batch, depth) and reports the SUSTAINED throughput; depth=1 is the
    /// single-flight baseline, so the depth-K/depth-1 ratio is the pipelining win. Every harvested wave is
    /// black_box'd + asserted to return `batch` rows (no-op guard). Then lpb (after shutdown; cuMemAlloc-safe).
    ///   `GPU_DB_WAVE_BENCH_ROWS=1000000 GPU_DB_WAVE_BENCH_THREADS=8192 cargo test -r -p gpu_db_execution \
    ///    wave::tests::wave_pipelined_offered_rate -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn wave_pipelined_offered_rate() {
        use std::collections::VecDeque;
        use std::hint::black_box;
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        let rows: u64 = std::env::var("GPU_DB_WAVE_BENCH_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);
        let threads: u32 = std::env::var("GPU_DB_WAVE_BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let keys: Vec<i32> = (0..rows as i32).map(|r| r.wrapping_mul(3).wrapping_add(1)).collect();
        let payload: Vec<i32> = (0..rows as i32).map(|r| r.wrapping_mul(1000).wrapping_add(7)).collect();
        let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
        for &k in &keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &v in &payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let resident = Arc::new(runtime.retain_device_memory_copy(0, &buf).expect("resident"));
        let payload_offset = rows * 4;
        let projections = [payload_offset]; // 1-col gather, like the single-flight benchmark
        let table_size = ((rows * 2) as u32).next_power_of_two();
        let table_mask = table_size - 1;
        let hash_shift = 32 - table_size.trailing_zeros();
        let mut index = vec![0_u64; table_size as usize];
        for (r, &k) in keys.iter().enumerate() {
            let key = k as u32;
            let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
            while index[h as usize] != 0 {
                h = (h + 1) & table_mask;
            }
            index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
        }
        let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
        let index_resident = Arc::new(runtime.retain_device_memory_copy(0, &index_bytes).expect("index"));

        let needles_for = |batch: usize, iter: usize| -> Vec<i32> {
            (0..batch)
                .map(|i| keys[(iter.wrapping_mul(batch).wrapping_add(i)) % rows as usize])
                .collect()
        };

        let ring_capacity = 1usize << 17; // 131072: bounds (max depth * max batch) in flight
        let batch_sizes = [1usize, 8, 32, 256];
        let depths = [1usize, 4, 16, 64];

        println!(
            "\n# R2.2 P2c: PIPELINED (depth-K) offered-rate  rows={rows} threads={threads} ring={ring_capacity}"
        );
        println!("  batch  depth        lookups/s   x(vs depth=1)      us/wave");

        let mut engine = WaveReadEngine::new(
            Arc::clone(&index_resident),
            Arc::clone(&resident),
            &projections,
            table_mask,
            hash_shift,
            ring_capacity,
            threads,
            30_000_000_000,
            0, // watchdog disabled (test relies on the fixed backstop + explicit shutdown)
        )
        .expect("wave engine");

        for &batch in &batch_sizes {
            // target ~1M lookups per (batch,depth) point, clamped to a reasonable wave count
            let waves: usize = (1_000_000 / batch).clamp(200, 200_000);
            let mut base_thru = 0f64;
            for &depth in &depths {
                // depth*batch must fit the ring (bound on in-flight, the caller contract).
                if depth * batch > ring_capacity {
                    continue;
                }
                // warm
                for w in 0..depth.min(waves) {
                    let _ = engine.submit(&needles_for(batch, w));
                }
                let mut inflight: VecDeque<WaveTicket> = VecDeque::with_capacity(depth);
                let mut submitted = 0usize;
                let mut harvested = 0usize;
                let mut ok = true;
                let t = Instant::now();
                while harvested < waves {
                    while inflight.len() < depth && submitted < waves {
                        match engine.submit_async(&needles_for(batch, submitted)) {
                            Ok(tk) => {
                                inflight.push_back(tk);
                                submitted += 1;
                            }
                            Err(_) => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        break;
                    }
                    let front = *inflight.front().unwrap();
                    let deadline = Instant::now() + Duration::from_secs(20);
                    loop {
                        match engine.harvest(front) {
                            Ok(Some(r)) => {
                                assert_eq!(r.len(), batch, "pipelined wave row count (batch={batch} depth={depth})");
                                black_box(&r);
                                inflight.pop_front();
                                harvested += 1;
                                break;
                            }
                            Ok(None) => {
                                if Instant::now() > deadline {
                                    ok = false;
                                    break;
                                }
                            }
                            Err(_) => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        break;
                    }
                }
                let secs = t.elapsed().as_secs_f64();
                let thru = if ok { (harvested * batch) as f64 / secs } else { 0.0 };
                if depth == 1 {
                    base_thru = thru;
                }
                let speedup = if base_thru > 0.0 { thru / base_thru } else { 0.0 };
                let us_wave = if thru > 0.0 { 1.0e6 / (thru / batch as f64) } else { f64::INFINITY };
                println!("  {batch:>5}  {depth:>5}  {thru:>15.0}   {speedup:>13.2}x  {us_wave:>11.2}");
            }
        }
        engine.shutdown();
        println!("# depth=1 is single-flight; depth-K/depth-1 ratio = the pipelining (concurrent-regime) win.");
    }
}
