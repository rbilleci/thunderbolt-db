//! Test-only independent GPU argsort oracles.
//!
//! These resident-pointer implementations are deliberately excluded from product code. They provide
//! algorithmically independent bitonic and single-thread radix parity for the typed production radix path.

use std::os::raw::c_void;

use super::{
    check_cuda, CudaResidentDeviceMemory, CudaRuntimeProbeError, GpuPrimaryContext, PooledStream,
};

// Stable bitonic argsort GPU parity oracle. This test-only implementation sorts N i64 keys
// (resident, at `keys_device_ptr`) by (key, original index) and returns the permutation indices,
// ascending or descending. Two kernels run on one pooled stream:
//   - `..._init`: grid-stride fill keys_work[i] = (i<N) ? key[i] : i64::MAX (pad sorts to the end),
//     idx_work[i] = i.
//   - `..._step`: one compare-exchange stage of the bitonic network; the host loops k (subsequence
//     size) and j (compare distance) over O(log²N) launches.
// STABLE: equal keys break the tie by ASCENDING original index regardless of direction, matching the
// engine's stable CPU ORDER BY. Host-looped steps handle any N; the per-step launch overhead is
// exactly why the production typed radix path wins above its measured crossover.
pub(super) fn launch_cuda_resident_i64_argsort_bitonic(
    resident: &CudaResidentDeviceMemory,
    keys_device_ptr: u64,
    n: u64,
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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

    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_bitonic_argsort_init(
    .param .u64 keys_src_ptr,
    .param .u64 n,
    .param .u64 n_pad,
    .param .u64 keys_work_ptr,
    .param .u64 idx_work_ptr,
    .param .u32 descending
)
{
    .reg .pred %p_done;
    .reg .pred %p_real;
    .reg .pred %p_pdesc;
    .reg .u32 %desc;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %gdim;
    .reg .u32 %tmp32;
    .reg .u32 %idxv;
    .reg .u64 %src;
    .reg .u64 %n;
    .reg .u64 %npad;
    .reg .u64 %kw;
    .reg .u64 %iw;
    .reg .u64 %i;
    .reg .u64 %stride;
    .reg .u64 %off8;
    .reg .u64 %off4;
    .reg .u64 %addr;
    .reg .s64 %keyv;

    ld.param.u64 %src, [keys_src_ptr];
    ld.param.u64 %n, [n];
    ld.param.u64 %npad, [n_pad];
    ld.param.u64 %kw, [keys_work_ptr];
    ld.param.u64 %iw, [idx_work_ptr];
    ld.param.u32 %desc, [descending];
    setp.eq.u32 %p_pdesc, %desc, 1;

    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %tmp32, %bid, %bdim, %lane;
    cvt.u64.u32 %i, %tmp32;
    mul.lo.u32 %tmp32, %gdim, %bdim;
    cvt.u64.u32 %stride, %tmp32;

iloop:
    setp.ge.u64 %p_done, %i, %npad;
    @%p_done bra idone;
    mul.lo.u64 %off8, %i, 8;
    mul.lo.u64 %off4, %i, 4;
    cvt.u32.u64 %idxv, %i;
    add.u64 %addr, %iw, %off4;
    st.global.u32 [%addr], %idxv;
    setp.lt.u64 %p_real, %i, %n;
    @!%p_real bra ipad;
    add.u64 %addr, %src, %off8;
    ld.global.s64 %keyv, [%addr];
    bra istore;
ipad:
    // Padding sentinel must sink to the UNREAD end so idx[0..n) holds only real rows.
    // Ascending: pad = i64::MAX (sorts to the high end). Descending: pad = i64::MIN
    // (sorts to the low end). MIN is computed as MAX+1 (two's-complement wrap) to avoid
    // the most-negative-literal PTX parse pitfall.
    mov.s64 %keyv, 9223372036854775807;
    @%p_pdesc add.s64 %keyv, %keyv, 1;
istore:
    add.u64 %addr, %kw, %off8;
    st.global.s64 [%addr], %keyv;
    add.u64 %i, %i, %stride;
    bra iloop;
idone:
    ret;
}

.visible .entry gpu_db_bitonic_argsort_step(
    .param .u64 keys_work_ptr,
    .param .u64 idx_work_ptr,
    .param .u64 n_pad,
    .param .u64 j,
    .param .u64 k,
    .param .u32 descending
)
{
    .reg .pred %p_done;
    .reg .pred %p_pair;
    .reg .pred %p_asc;
    .reg .pred %p_nasc;
    .reg .pred %p_desc;
    .reg .pred %p_ndesc;
    .reg .pred %p_kgt;
    .reg .pred %p_klt;
    .reg .pred %p_keq;
    .reg .pred %p_igt;
    .reg .pred %p_ilt;
    .reg .pred %p_ka;
    .reg .pred %p_kb;
    .reg .pred %p_ta;
    .reg .pred %p_tb;
    .reg .pred %p_iaft;
    .reg .pred %p_ibef;
    .reg .pred %p_s1;
    .reg .pred %p_s2;
    .reg .pred %p_swap;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %gdim;
    .reg .u32 %tmp32;
    .reg .u32 %desc;
    .reg .u32 %idxi;
    .reg .u32 %idxl;
    .reg .u64 %kw;
    .reg .u64 %iw;
    .reg .u64 %npad;
    .reg .u64 %j;
    .reg .u64 %k;
    .reg .u64 %i;
    .reg .u64 %l;
    .reg .u64 %stride;
    .reg .u64 %band;
    .reg .u64 %ai8;
    .reg .u64 %al8;
    .reg .u64 %ai4;
    .reg .u64 %al4;
    .reg .u64 %addr;
    .reg .s64 %ki;
    .reg .s64 %kl;

    ld.param.u64 %kw, [keys_work_ptr];
    ld.param.u64 %iw, [idx_work_ptr];
    ld.param.u64 %npad, [n_pad];
    ld.param.u64 %j, [j];
    ld.param.u64 %k, [k];
    ld.param.u32 %desc, [descending];

    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %tmp32, %bid, %bdim, %lane;
    cvt.u64.u32 %i, %tmp32;
    mul.lo.u32 %tmp32, %gdim, %bdim;
    cvt.u64.u32 %stride, %tmp32;

    setp.eq.u32 %p_desc, %desc, 1;
    not.pred %p_ndesc, %p_desc;

sloop:
    setp.ge.u64 %p_done, %i, %npad;
    @%p_done bra sdone;
    xor.b64 %l, %i, %j;
    setp.gt.u64 %p_pair, %l, %i;
    @!%p_pair bra snext;

    mul.lo.u64 %ai8, %i, 8;
    mul.lo.u64 %al8, %l, 8;
    mul.lo.u64 %ai4, %i, 4;
    mul.lo.u64 %al4, %l, 4;
    add.u64 %addr, %kw, %ai8;
    ld.global.s64 %ki, [%addr];
    add.u64 %addr, %kw, %al8;
    ld.global.s64 %kl, [%addr];
    add.u64 %addr, %iw, %ai4;
    ld.global.u32 %idxi, [%addr];
    add.u64 %addr, %iw, %al4;
    ld.global.u32 %idxl, [%addr];

    setp.gt.s64 %p_kgt, %ki, %kl;
    setp.lt.s64 %p_klt, %ki, %kl;
    setp.eq.s64 %p_keq, %ki, %kl;
    setp.gt.u32 %p_igt, %idxi, %idxl;
    setp.lt.u32 %p_ilt, %idxi, %idxl;
    and.pred %p_s1, %p_desc, %p_klt;
    and.pred %p_s2, %p_ndesc, %p_kgt;
    or.pred %p_ka, %p_s1, %p_s2;
    and.pred %p_s1, %p_desc, %p_kgt;
    and.pred %p_s2, %p_ndesc, %p_klt;
    or.pred %p_kb, %p_s1, %p_s2;
    and.pred %p_ta, %p_keq, %p_igt;
    or.pred %p_iaft, %p_ka, %p_ta;
    and.pred %p_tb, %p_keq, %p_ilt;
    or.pred %p_ibef, %p_kb, %p_tb;
    and.b64 %band, %i, %k;
    setp.eq.u64 %p_asc, %band, 0;
    not.pred %p_nasc, %p_asc;
    and.pred %p_s1, %p_asc, %p_iaft;
    and.pred %p_s2, %p_nasc, %p_ibef;
    or.pred %p_swap, %p_s1, %p_s2;
    @!%p_swap bra snext;
    add.u64 %addr, %kw, %ai8;
    st.global.s64 [%addr], %kl;
    add.u64 %addr, %kw, %al8;
    st.global.s64 [%addr], %ki;
    add.u64 %addr, %iw, %ai4;
    st.global.u32 [%addr], %idxl;
    add.u64 %addr, %iw, %al4;
    st.global.u32 [%addr], %idxi;

snext:
    add.u64 %i, %i, %stride;
    bra sloop;
sdone:
    ret;
}
"#;

    if n == 0 {
        return Ok(Vec::new());
    }
    // Argsort indices are materialized as u32 on-device (idx buffer = n_pad*4, `cvt.u32.u64`).
    // Guard the precondition so >4B rows fail loud instead of silently truncating the index.
    if n > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    // `keys_device_ptr` is an absolute device address; the caller owns its [ptr, ptr+n*8) sizing.
    let n_pad = n
        .checked_next_power_of_two()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let n_usize =
        usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let n_pad_usize = usize::try_from(n_pad)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let init_fn = resident
        .primary()
        .cached_function(c"gpu_db_bitonic_argsort_init", &ptx)?;
    let step_fn = resident
        .primary()
        .cached_function(c"gpu_db_bitonic_argsort_step", &ptx)?;

    let primary = resident.primary();
    let keys_work = primary.lease_device_buffer(
        n_pad_usize
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )?;
    let idx_work = primary.lease_device_buffer(
        n_pad_usize
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )?;

    primary.set_current()?;
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = lease
        .pooled
        .as_ref()
        .expect("pooled stream just set")
        .stream;
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        // SAFETY: best-effort drain on an already-failing path before the leases unwind.
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    const BLOCK: u32 = 256;
    let grid = n_pad.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;

    let mut src_arg = keys_device_ptr;
    let mut n_arg = n;
    let mut n_pad_arg = n_pad;
    let mut kw_arg = keys_work.ptr;
    let mut iw_arg = idx_work.ptr;
    let mut desc_arg: u32 = u32::from(descending);
    let mut init_args = [
        (&mut src_arg as *mut u64).cast::<c_void>(),
        (&mut n_arg as *mut u64).cast::<c_void>(),
        (&mut n_pad_arg as *mut u64).cast::<c_void>(),
        (&mut kw_arg as *mut u64).cast::<c_void>(),
        (&mut iw_arg as *mut u64).cast::<c_void>(),
        (&mut desc_arg as *mut u32).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            init_fn,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            init_args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
    .map_err(drain_err)?;

    let mut k = 2_u64;
    while k <= n_pad {
        let mut j = k >> 1;
        while j > 0 {
            let mut j_arg = j;
            let mut k_arg = k;
            let mut step_args = [
                (&mut kw_arg as *mut u64).cast::<c_void>(),
                (&mut iw_arg as *mut u64).cast::<c_void>(),
                (&mut n_pad_arg as *mut u64).cast::<c_void>(),
                (&mut j_arg as *mut u64).cast::<c_void>(),
                (&mut k_arg as *mut u64).cast::<c_void>(),
                (&mut desc_arg as *mut u32).cast::<c_void>(),
            ];
            check_cuda(unsafe {
                cu_launch_kernel(
                    step_fn,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    step_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })
            .map_err(drain_err)?;
            j >>= 1;
        }
        k <<= 1;
    }

    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
    resident.record_kernel_event_elapsed_us(None);

    let mut indices = vec![0_u32; n_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            indices.as_mut_ptr().cast::<c_void>(),
            idx_work.ptr,
            n_usize * std::mem::size_of::<u32>(),
        )
    })
    .map_err(drain_err)?;
    drop(lease);
    Ok(indices)
}

/// Serial single-thread GPU LSD-radix argsort — the GPU-native parity ORACLE + benchmark
/// baseline for the parallel `launch_cuda_resident_i64_argsort_radix` (S3). One device thread
/// runs a textbook stable counting sort: 16 LSD passes of 4 bits each over a signed→unsigned
/// key transform (XOR `mask`: 0x8000…0 ascending so i64 order == u64 order; 0x7FFF…F descending
/// = the complement, so one ascending radix yields descending keys with the SAME ascending-index
/// tie-break — stable in both directions). Obviously correct + stable (serial in-order scatter),
/// so the parallel version is validated against THIS (same algorithm → isolates parallelization
/// bugs) as well as the independently-verified bitonic arm. Test-only; never on the hot path.
#[cfg(test)]
pub(super) fn launch_cuda_resident_i64_argsort_radix_serial(
    resident: &CudaResidentDeviceMemory,
    keys_device_ptr: u64,
    n: u64,
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
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

    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_radix_argsort_serial(
    .param .u64 keys_src_ptr,
    .param .u64 n,
    .param .u64 mask,
    .param .u64 keys_a_ptr,
    .param .u64 keys_b_ptr,
    .param .u64 idx_a_ptr,
    .param .u64 idx_b_ptr,
    .param .u64 hist_ptr
)
{
    .reg .pred %p;
    .reg .u32 %thr;
    .reg .u32 %bid;
    .reg .u32 %lane32;
    .reg .u32 %pass;
    .reg .u32 %shift;
    .reg .u32 %d;
    .reg .u32 %cnt;
    .reg .u32 %run;
    .reg .u32 %tmp;
    .reg .u32 %idxv;
    .reg .u32 %pos;
    .reg .u64 %src;
    .reg .u64 %n;
    .reg .u64 %mask;
    .reg .u64 %ka;
    .reg .u64 %kb;
    .reg .u64 %ia;
    .reg .u64 %ib;
    .reg .u64 %hist;
    .reg .u64 %i;
    .reg .u64 %o8;
    .reg .u64 %o4;
    .reg .u64 %addr;
    .reg .u64 %key;
    .reg .u64 %dig;

    // Single-thread oracle: only (block 0, thread 0) executes; the rest return.
    mov.u32 %thr, %tid.x;
    mov.u32 %bid, %ctaid.x;
    or.b32 %lane32, %thr, %bid;
    setp.ne.u32 %p, %lane32, 0;
    @%p bra done;

    ld.param.u64 %src, [keys_src_ptr];
    ld.param.u64 %n, [n];
    ld.param.u64 %mask, [mask];
    ld.param.u64 %ka, [keys_a_ptr];
    ld.param.u64 %kb, [keys_b_ptr];
    ld.param.u64 %ia, [idx_a_ptr];
    ld.param.u64 %ib, [idx_b_ptr];
    ld.param.u64 %hist, [hist_ptr];

    // init: keys_a[i] = key[i] XOR mask ; idx_a[i] = i
    mov.u64 %i, 0;
init_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra init_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %src, %o8;
    ld.global.u64 %key, [%addr];
    xor.b64 %key, %key, %mask;
    add.u64 %addr, %ka, %o8;
    st.global.u64 [%addr], %key;
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %ia, %o4;
    cvt.u32.u64 %idxv, %i;
    st.global.u32 [%addr], %idxv;
    add.u64 %i, %i, 1;
    bra init_loop;
init_done:

    mov.u32 %pass, 0;
pass_loop:
    setp.ge.u32 %p, %pass, 16;
    @%p bra pass_done;
    mul.lo.u32 %shift, %pass, 4;

    // zero hist[0..16)
    mov.u32 %d, 0;
hz_loop:
    setp.ge.u32 %p, %d, 16;
    @%p bra hz_done;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    mov.u32 %tmp, 0;
    st.global.u32 [%addr], %tmp;
    add.u32 %d, %d, 1;
    bra hz_loop;
hz_done:

    // count: hist[digit(keys_a[i])]++
    mov.u64 %i, 0;
count_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra count_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %ka, %o8;
    ld.global.u64 %key, [%addr];
    shr.u64 %dig, %key, %shift;
    and.b64 %dig, %dig, 15;
    cvt.u32.u64 %d, %dig;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    ld.global.u32 %cnt, [%addr];
    add.u32 %cnt, %cnt, 1;
    st.global.u32 [%addr], %cnt;
    add.u64 %i, %i, 1;
    bra count_loop;
count_done:

    // exclusive scan: run=0; for d: t=hist[d]; hist[d]=run; run+=t
    mov.u32 %run, 0;
    mov.u32 %d, 0;
scan_loop:
    setp.ge.u32 %p, %d, 16;
    @%p bra scan_done;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    ld.global.u32 %cnt, [%addr];
    st.global.u32 [%addr], %run;
    add.u32 %run, %run, %cnt;
    add.u32 %d, %d, 1;
    bra scan_loop;
scan_done:

    // stable scatter (in input order): pos = hist[d]++ ; keys_b[pos]=key ; idx_b[pos]=idx_a[i]
    mov.u64 %i, 0;
scatter_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra scatter_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %ka, %o8;
    ld.global.u64 %key, [%addr];
    shr.u64 %dig, %key, %shift;
    and.b64 %dig, %dig, 15;
    cvt.u32.u64 %d, %dig;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    ld.global.u32 %pos, [%addr];
    add.u32 %tmp, %pos, 1;
    st.global.u32 [%addr], %tmp;
    // keys_b[pos] = key
    mul.wide.u32 %o8, %pos, 8;
    add.u64 %addr, %kb, %o8;
    st.global.u64 [%addr], %key;
    // idx_b[pos] = idx_a[i]
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %ia, %o4;
    ld.global.u32 %idxv, [%addr];
    mul.wide.u32 %o4, %pos, 4;
    add.u64 %addr, %ib, %o4;
    st.global.u32 [%addr], %idxv;
    add.u64 %i, %i, 1;
    bra scatter_loop;
scatter_done:

    // copy b -> a (keys + idx) so the next pass reads from a
    mov.u64 %i, 0;
copy_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra copy_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %kb, %o8;
    ld.global.u64 %key, [%addr];
    add.u64 %addr, %ka, %o8;
    st.global.u64 [%addr], %key;
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %ib, %o4;
    ld.global.u32 %idxv, [%addr];
    add.u64 %addr, %ia, %o4;
    st.global.u32 [%addr], %idxv;
    add.u64 %i, %i, 1;
    bra copy_loop;
copy_done:

    add.u32 %pass, %pass, 1;
    bra pass_loop;
pass_done:
done:
    ret;
}
"#;

    if n == 0 {
        return Ok(Vec::new());
    }
    if n > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    // `keys_device_ptr` is an absolute device address; the caller owns its [ptr, ptr+n*8) sizing.
    let n_usize =
        usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_argsort_serial", &ptx)?;

    let primary = resident.primary();
    let bytes8 = n_usize
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes4 = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let keys_a = primary.lease_device_buffer(bytes8)?;
    let keys_b = primary.lease_device_buffer(bytes8)?;
    let idx_a = primary.lease_device_buffer(bytes4)?;
    let idx_b = primary.lease_device_buffer(bytes4)?;
    let hist = primary.lease_device_buffer(16 * std::mem::size_of::<u32>())?;

    primary.set_current()?;
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = lease
        .pooled
        .as_ref()
        .expect("pooled stream just set")
        .stream;
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    let mut src_arg = keys_device_ptr;
    let mut n_arg = n;
    let mut mask_arg: u64 = if descending {
        0x7FFF_FFFF_FFFF_FFFF
    } else {
        0x8000_0000_0000_0000
    };
    let mut ka_arg = keys_a.ptr;
    let mut kb_arg = keys_b.ptr;
    let mut ia_arg = idx_a.ptr;
    let mut ib_arg = idx_b.ptr;
    let mut hist_arg = hist.ptr;
    let mut args = [
        (&mut src_arg as *mut u64).cast::<c_void>(),
        (&mut n_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut ka_arg as *mut u64).cast::<c_void>(),
        (&mut kb_arg as *mut u64).cast::<c_void>(),
        (&mut ia_arg as *mut u64).cast::<c_void>(),
        (&mut ib_arg as *mut u64).cast::<c_void>(),
        (&mut hist_arg as *mut u64).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            kernel_fn,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
    .map_err(drain_err)?;
    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
    resident.record_kernel_event_elapsed_us(None);

    let mut indices = vec![0_u32; n_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            indices.as_mut_ptr().cast::<c_void>(),
            idx_a.ptr,
            n_usize * std::mem::size_of::<u32>(),
        )
    })
    .map_err(drain_err)?;
    drop(lease);
    Ok(indices)
}
