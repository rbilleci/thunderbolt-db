use std::os::raw::c_void;
use std::sync::Arc;

use crate::cuda_context::{
    check_cuda, GpuPrimaryContext, PooledDeviceBufferOwned, PooledStreamOwned,
};
use crate::resident_memory::CudaResidentDeviceMemory;
use crate::{copy_pinned_into, stage_result_dtoh_async, CudaRuntimeProbeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaI32BatchProjectionRow {
    pub needle_index: usize,
    pub row_index: u64,
    pub values: Vec<i32>,
}

/// COLUMNAR form of a matched-row batch — the three flat arrays the GPU D2H already produces, kept flat
/// instead of split into one heap `Vec<i32>` per row. The per-row `CudaI32BatchProjectionRow` split cost
/// ~1585us/65536-batch (and spiked the p99 tail) for nothing — the engine re-flattens the rows anyway
/// (DECISIONS "Tail latency"). `values` is row-major: row `i`'s projection is `values[i*projection_count..]`,
/// matched against needle `needle_indices[i]` at table row `row_indices[i]`. Index arrays are validated
/// (`needle_index < needles_len`, `row_index < row_count`) when produced, so consumers may trust them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaI32BatchProjectionColumns {
    pub values: Vec<i32>,
    pub needle_indices: Vec<u32>,
    pub row_indices: Vec<u64>,
    pub projection_count: usize,
    /// DENSE LAYOUT marker (DECISIONS "lpb read levers" #1): EMPTY for the compacted atomic/wave form above.
    /// When NON-empty, `values` is the DENSE form — one slot per needle (`status.len()` needles, gaps for
    /// absent), so `values[i*projection_count..]` is NEEDLE `i`'s projection and `status[i]` is 1 (found) /
    /// 2 (not-found); `needle_indices`/`row_indices` are empty. The engine assemble compacts it in ONE
    /// sequential pass (no host scatter), instead of re-compacting an already-compacted result.
    pub status: Vec<u32>,
}

impl CudaI32BatchProjectionColumns {
    pub fn nrows(&self) -> usize {
        self.needle_indices.len()
    }
    /// Row `i`'s projected values (`projection_count` wide).
    pub fn row_values(&self, i: usize) -> &[i32] {
        let p = self.projection_count;
        &self.values[i * p..i * p + p]
    }
    /// Build the columnar form from the per-row form (the inverse of `into_rows`) — used by the wave/lpb
    /// differential + materialized-arm tests, which author per-row fixtures. `projection_count` from the
    /// first row's width (uniform by construction).
    pub fn from_rows(rows: Vec<CudaI32BatchProjectionRow>) -> Self {
        let projection_count = rows.first().map(|row| row.values.len()).unwrap_or(0);
        // `projection_count` is inferred from the FIRST row, so a ragged or zero-width-with-rows input would
        // misreshape (and `into_rows` would silently drop rows at width 0). Every live shape is uniform and
        // >=1 wide; assert it so a future zero/ragged-projection producer trips here in tests (audit P3).
        debug_assert!(
            rows.is_empty()
                || (projection_count > 0
                    && rows.iter().all(|r| r.values.len() == projection_count)),
            "from_rows requires uniform, non-zero-width rows (got width {projection_count})"
        );
        let mut values = Vec::with_capacity(rows.len() * projection_count);
        let mut needle_indices = Vec::with_capacity(rows.len());
        let mut row_indices = Vec::with_capacity(rows.len());
        for row in rows {
            needle_indices.push(row.needle_index as u32);
            row_indices.push(row.row_index);
            values.extend(row.values);
        }
        Self {
            values,
            needle_indices,
            row_indices,
            projection_count,
            status: Vec::new(),
        }
    }
    /// Bridge back to the per-row form for the cold per-needle completion + tests (re-introduces the per-row
    /// `Vec`, but only off the hot batched path). The arrays were validated when produced.
    pub fn into_rows(self) -> Vec<CudaI32BatchProjectionRow> {
        // Width 0 with rows present would silently drop them (`chunks(0.max(1))` mis-chunks). Unreachable on
        // live shapes (projections are >=1 wide); fail loud in tests rather than drop silently (audit P3).
        debug_assert!(
            self.projection_count > 0 || self.values.is_empty(),
            "into_rows: projection_count==0 with non-empty values would silently drop rows"
        );
        // DENSE layout (status set): slot `i` is needle `i`; keep status==1. Cold per-needle path; row_index
        // synthesized 0 (unique => never read).
        if !self.status.is_empty() {
            let p = self.projection_count.max(1);
            let mut rows = Vec::new();
            for i in 0..self.status.len() {
                if self.status[i] == 1 {
                    rows.push(CudaI32BatchProjectionRow {
                        needle_index: i,
                        row_index: 0,
                        values: self.values[i * p..i * p + p].to_vec(),
                    });
                }
            }
            return rows;
        }
        // The DENSE index-probe produces no `row_indices` (unique => no within-needle sort needs them); the
        // cold per-needle path that calls `into_rows` synthesizes 0 (the value is never read — a 1-row needle
        // is trivially ordered, and the differentials compare projected values, not row_index).
        let has_row_indices = !self.row_indices.is_empty();
        let p = self.projection_count.max(1);
        self.values
            .chunks(p)
            .zip(self.needle_indices)
            .enumerate()
            .map(|(i, (row, needle_index))| CudaI32BatchProjectionRow {
                needle_index: needle_index as usize,
                row_index: if has_row_indices {
                    self.row_indices[i]
                } else {
                    0
                },
                values: row.to_vec(),
            })
            .collect()
    }
}

pub struct CudaI32EqualAnyProjectSubmission {
    pub(super) projection_count: usize,
    pub(super) needles_len: usize,
    pub(super) row_count: u64,
    // Shared primary context: gives the deferred `complete` re-entry to the buffer/pinned/stream
    // pools and the (optional) async transfer symbols, exactly like the synchronous routes get
    // via `resident.primary()`. Cheap `Arc` clone; already `Send + Sync`.
    pub(super) primary: Arc<GpuPrimaryContext>,
    // P2-M2 (equal_any split-route migration): the device buffers are now leased from the shared
    // `OutputBufferPool` (no per-call `cuMemAlloc`/`cuMemFree`) and the kernel runs on a pooled
    // private stream (no NULL-stream context-wide barrier), mirroring the text route. Because this
    // route defers `complete` (possibly to another thread), the pooled buffers + stream are
    // carried as owned (`Arc`-holding) guards that return to their pools on Drop — the `'static`
    // analogue of the synchronous routes' borrow-scoped leases. The kernel's HtoD(needles) +
    // memset(count) + launch are enqueued (not synced) on `stream` in `submit`; `complete` syncs
    // on `stop_event`, reads the count, then reads the result arrays — using the async (pinned,
    // stream-ordered) copies when the optional symbols are present and the blocking ones otherwise.
    pub(super) values_guard: PooledDeviceBufferOwned,
    pub(super) indices_guard: PooledDeviceBufferOwned,
    pub(super) row_indices_guard: PooledDeviceBufferOwned,
    pub(super) count_guard: PooledDeviceBufferOwned,
    pub(super) _needles_guard: PooledDeviceBufferOwned,
    // Held (not released in `submit`) so its timing events stay valid for `complete` and no other
    // reader leases this stream while our kernel is still enqueued on it. Released on Drop.
    //
    // `Option` is the success/drop coordination lever for the `Drop` impl below: a successful
    // `complete_detached` `take()`s this stream out (after its own covering sync), leaving `None`,
    // so the `Drop` impl sees `None` and does NOT redundantly re-sync. If the submission is dropped
    // WITHOUT `complete` (early `Err` in the engine, or any `?`/cancel/panic in the submit→complete
    // window), it is still `Some`, so `Drop` drains it before the field guards release the pooled
    // device buffers + stream back to the SHARED pools — without that drain a concurrent
    // `lease_device_buffer` could re-lease memory the in-flight kernel/HtoD/memset still writes
    // (cross-thread use-after-free).
    pub(super) stream: Option<PooledStreamOwned>,
    // Whether the pooled stream's start/stop events were available (best-effort timing).
    pub(super) timed: bool,
    // R1b: for the GPU index-probe route ONLY (`None` for the scan route), a refcount on the device
    // index buffer the kernel reads. Unlike the resident allocation (owned by the residency map and
    // kept alive by the caller per the contract above), the index lives in an engine-side reuse cache
    // that a concurrent re-admission may evict mid-flight; pinning the `Arc` here makes the index
    // buffer outlive THIS kernel's submit->complete regardless of cache eviction. Released on Drop.
    pub(super) _wave_index_guard: Option<Arc<CudaResidentDeviceMemory>>,
}

// Pending read submissions own their temporary CUDA allocations/events/module.
// The resident allocation itself remains owned elsewhere and must outlive
// completion.
unsafe impl Send for CudaI32EqualAnyProjectSubmission {}
impl CudaI32EqualAnyProjectSubmission {
    pub fn complete(
        self,
        resident: &CudaResidentDeviceMemory,
    ) -> Result<Vec<CudaI32BatchProjectionRow>, CudaRuntimeProbeError> {
        let (rows, elapsed_us) = self.complete_detached()?;
        resident.record_kernel_event_elapsed_us(elapsed_us);
        Ok(rows)
    }

    /// Per-row form (cold per-needle completion + tests): the columnar drain then `into_rows`. The HOT batched
    /// path calls `complete_detached_columnar` directly to skip the 65536 per-row allocations (DECISIONS
    /// "Tail latency").
    pub fn complete_detached(
        self,
    ) -> Result<(Vec<CudaI32BatchProjectionRow>, Option<u64>), CudaRuntimeProbeError> {
        let (columns, elapsed_us) = self.complete_detached_columnar()?;
        Ok((columns.into_rows(), elapsed_us))
    }

    pub fn complete_detached_columnar(
        mut self,
    ) -> Result<(CudaI32BatchProjectionColumns, Option<u64>), CudaRuntimeProbeError> {
        let primary = Arc::clone(&self.primary);
        primary.set_current()?;
        // Take ownership of the pooled stream out of the submission. This local guard keeps the
        // stream alive (and pool-bound only on its own Drop at the end of this function) for the
        // whole completion, AND leaves `self.stream == None` so the submission's `Drop` does NOT
        // re-sync on this success path (no double-drain). The covering sync below is the single
        // drain for the success path.
        let stream_owned = self
            .stream
            .take()
            .expect("pooled stream held until complete");
        let pooled = stream_owned
            .pooled
            .as_ref()
            .expect("pooled stream held until complete");
        let stream = pooled.stream;

        // The kernel + its HtoD/memset were enqueued (not synced) on this private stream in
        // `submit`. One stream sync here drains all of that: it is the deferred counterpart of the
        // synchronous routes' post-launch sync. (Syncing the stream also completes the recorded
        // stop event, so the elapsed-time read below is valid.)
        check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) })?;

        let elapsed_us = if self.timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (primary.cu_event_elapsed_time)(
                    &mut elapsed_ms,
                    pooled.start_event,
                    pooled.stop_event,
                )
            })?;
            Some((f64::from(elapsed_ms) * 1_000.0).ceil() as u64)
        } else {
            None
        };

        // Optional async transfer symbols (present on a modern driver): the result D2H then stage
        // through pooled pinned host buffers, stream-ordered, behind a single covering sync — the
        // same lever the text route uses. On an old driver lacking them, fall back to the blocking
        // `cuMemcpyDtoH` (the stream is already idle after the sync above, so a blocking copy on
        // the NULL stream observes the kernel's writes correctly). Correctness is identical on
        // both paths; only the transfer is accelerated when the symbols exist.
        let async_dtoh = primary.cu_memcpy_dtoh_async;

        // Error-path stream drain: once a result D2H is enqueued on `stream`, an early `?` would
        // unwind the owned buffer/stream guards (returning them to their pools) while the copy may
        // still be in flight — a use-after-free for the next leaser. `drain_err` does a best-effort
        // blocking sync FIRST (at the error site, before any guard Drop), then yields the original
        // error. Zero success-path cost (`map_err` skips the closure on `Ok`).
        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            // SAFETY: `stream` is the live held pooled stream; a blocking sync on it is valid here
            // (the primary context is current). The result is intentionally ignored — best-effort
            // drain on an already-failing path.
            unsafe {
                let _ = (primary.cu_stream_synchronize)(stream);
            }
            err
        };

        // Read the match count first (sizes every result array). Stage through a pooled pinned
        // buffer on the async path; a stack u32 otherwise.
        let mut match_count_host = 0_u32;
        if let Some(dtoh_async) = async_dtoh {
            let count_pinned = primary.lease_pinned_host_buffer(std::mem::size_of::<u32>());
            let count_dst: *mut c_void = count_pinned
                .as_ref()
                .map(|p| p.ptr)
                .unwrap_or_else(|| (&mut match_count_host as *mut u32).cast::<c_void>());
            check_cuda(unsafe {
                dtoh_async(
                    count_dst,
                    self.count_guard.ptr,
                    std::mem::size_of::<u32>(),
                    stream,
                )
            })
            .map_err(drain_err)?;
            check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
            if let Some(pinned) = &count_pinned {
                // SAFETY: the sync completed the 4-byte D2H into the page-aligned pinned region.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pinned.ptr.cast::<u32>(),
                        &mut match_count_host as *mut u32,
                        1,
                    );
                }
            }
        } else {
            check_cuda(unsafe {
                (primary.cu_memcpy_dtoh)(
                    (&mut match_count_host as *mut u32).cast::<c_void>(),
                    self.count_guard.ptr,
                    std::mem::size_of::<u32>(),
                )
            })?;
        }
        let match_count = u64::from(match_count_host);
        if match_count > self.row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let match_count_usize = usize::try_from(match_count)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

        let mut values = vec![0_i32; match_count_usize.saturating_mul(self.projection_count)];
        let mut needle_indices = vec![0_u32; match_count_usize];
        let mut row_indices = vec![0_u64; match_count_usize];

        if let Some(dtoh_async) = async_dtoh {
            // Stream-ordered result D2H into pooled pinned buffers, each only over the populated
            // [0, count) prefix, behind ONE covering sync — so the three copies overlap on the
            // copy engine instead of serializing as blocking barriers.
            let values_pinned = stage_result_dtoh_async(
                primary.as_ref(),
                dtoh_async,
                stream,
                self.values_guard.ptr,
                &mut values,
            )
            .map_err(drain_err)?;
            let needle_indices_pinned = stage_result_dtoh_async(
                primary.as_ref(),
                dtoh_async,
                stream,
                self.indices_guard.ptr,
                &mut needle_indices,
            )
            .map_err(drain_err)?;
            let row_indices_pinned = stage_result_dtoh_async(
                primary.as_ref(),
                dtoh_async,
                stream,
                self.row_indices_guard.ptr,
                &mut row_indices,
            )
            .map_err(drain_err)?;
            check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
            copy_pinned_into(&values_pinned, &mut values);
            copy_pinned_into(&needle_indices_pinned, &mut needle_indices);
            copy_pinned_into(&row_indices_pinned, &mut row_indices);
        } else {
            if !values.is_empty() {
                check_cuda(unsafe {
                    (primary.cu_memcpy_dtoh)(
                        values.as_mut_ptr().cast::<c_void>(),
                        self.values_guard.ptr,
                        values.len() * std::mem::size_of::<i32>(),
                    )
                })?;
            }
            if !needle_indices.is_empty() {
                check_cuda(unsafe {
                    (primary.cu_memcpy_dtoh)(
                        needle_indices.as_mut_ptr().cast::<c_void>(),
                        self.indices_guard.ptr,
                        needle_indices.len() * std::mem::size_of::<u32>(),
                    )
                })?;
            }
            if !row_indices.is_empty() {
                check_cuda(unsafe {
                    (primary.cu_memcpy_dtoh)(
                        row_indices.as_mut_ptr().cast::<c_void>(),
                        self.row_indices_guard.ptr,
                        row_indices.len() * std::mem::size_of::<u64>(),
                    )
                })?;
            }
        }

        // Validate the index arrays in a tight, ALLOCATION-FREE loop (the old per-row path validated while
        // building 65536 owned `CudaI32BatchProjectionRow`s — that allocation storm was ~1585us/batch + the
        // p99 tail, DECISIONS "Tail latency"). Same invariants: `needle_index < needles_len`,
        // `row_index < row_count`. The columnar arrays are returned as-is for the engine to scatter flat.
        for &needle_index in &needle_indices {
            if needle_index as usize >= self.needles_len {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    needle_index as usize,
                ));
            }
        }
        for &row_index in &row_indices {
            if row_index >= self.row_count {
                return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
            }
        }
        let columns = CudaI32BatchProjectionColumns {
            values,
            needle_indices,
            row_indices,
            projection_count: self.projection_count,
            status: Vec::new(),
        };
        Ok((columns, elapsed_us))
    }
}

impl Drop for CudaI32EqualAnyProjectSubmission {
    fn drop(&mut self) {
        // Drop-WITHOUT-`complete` safety drain. `submit` enqueues HtoD(needles) + memset(count) +
        // the kernel on the held pooled private stream WITHOUT syncing; the covering
        // `cuStreamSynchronize` lives only in `complete_detached`. If this submission is dropped
        // before `complete` runs (the early `Err` in the engine's
        // `complete_relational_retained_int4_projection_submission`, or any `?`/cancel/panic in the
        // submit→complete window), the field guards below — `PooledDeviceBufferOwned` ×5 then
        // `PooledStreamOwned` — would otherwise return the device buffers + stream to the SHARED
        // pools WHILE that work is still in flight, so a concurrent `lease_device_buffer` on another
        // thread could re-lease the memory the kernel still writes (cross-thread use-after-free).
        //
        // Drain-BEFORE-release ordering: Rust runs this explicit `Drop::drop` body in full BEFORE
        // dropping the struct's fields (which release the pools), so syncing the held stream here
        // guarantees the in-flight work has finished before any guard releases.
        //
        // No double-drain on the success path: a successful `complete_detached` `take()`s `stream`
        // out (after its own covering sync), leaving `None`, so the `if let Some` below is skipped
        // and this drop does no redundant sync — only the genuine drop-without-complete path (where
        // `stream` is still `Some`) drains.
        //
        // Best-effort, NO PANIC (a panic in `Drop` during unwinding aborts the process): the
        // context is bound first because this drop may run on a thread that never bound it, then the
        // stream is synced — every result is ignored, mirroring the `drain_err` best-effort style.
        if let Some(stream_owned) = self.stream.as_ref() {
            if let Some(pooled) = stream_owned.pooled.as_ref() {
                // Bind the primary context FIRST (idempotent; may run unbound here), exactly like
                // `complete_detached`'s `set_current` and `CudaResidentDeviceMemory::drop`'s bind
                // before its `cuMemFree`. Ignore the result — cleanup path.
                let _ = self.primary.set_current();
                // SAFETY: `pooled.stream` is the live held pooled stream; a blocking sync on it is
                // valid with the primary context bound above. The result is intentionally ignored —
                // best-effort drain on a teardown path that must never panic.
                unsafe {
                    let _ = (self.primary.cu_stream_synchronize)(pooled.stream);
                }
            }
        }
        // Fields drop after this body: the buffer/stream guards now return to their pools AFTER the
        // drain above, so the next leaser never observes memory still under an in-flight kernel.
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaI32TextBatchProjectionRow {
    pub needle_index: usize,
    pub row_index: u64,
    pub values: Vec<i32>,
    pub text: String,
    pub text_is_null: bool,
}
