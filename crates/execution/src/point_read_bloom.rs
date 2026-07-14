use std::ffi::c_void;
use std::sync::Arc;

use super::{check_cuda, CudaResidentDeviceMemory, CudaResidentReadSource, CudaRuntimeProbeError};

/// P5-later: one compact per-chunk Bloom filter. `bit_mask + 1` is a power-of-two bit count.
pub struct ChunkBloomProbeShard {
    pub bloom: Arc<CudaResidentDeviceMemory>,
    pub bit_mask: u32,
}

pub(super) fn probe_cuda_chunk_blooms(
    ctx: &CudaResidentDeviceMemory,
    blooms: &[ChunkBloomProbeShard],
    needles: &[i32],
) -> Result<Vec<Vec<u32>>, CudaRuntimeProbeError> {
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
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
.target sm_30
.address_size 64
.visible .entry gpu_db_chunk_bloom_probe(
    .param .u64 desc_ptr, .param .u32 chunk_count, .param .u64 needles_ptr,
    .param .u32 needle_count, .param .u64 out_ptr)
{
    .reg .pred %p<5>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<18>;
    ld.param.u64 %rd1, [desc_ptr];
    ld.param.u32 %r1, [chunk_count];
    ld.param.u64 %rd2, [needles_ptr];
    ld.param.u32 %r2, [needle_count];
    ld.param.u64 %rd3, [out_ptr];
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    mul.lo.u32 %r7, %r1, %r2;
    setp.ge.u32 %p1, %r6, %r7;
    @%p1 bra DONE;
    div.u32 %r8, %r6, %r1;
    rem.u32 %r9, %r6, %r1;
    mul.wide.u32 %rd4, %r8, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u32 %r10, [%rd5];
    mul.wide.u32 %rd6, %r9, 16;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u64 %rd8, [%rd7];
    ld.global.u64 %rd9, [%rd7+8];
    cvt.u32.u64 %r11, %rd9;
    mul.lo.u32 %r12, %r10, 2654435761;
    shr.u32 %r13, %r10, 16;
    xor.b32 %r13, %r13, %r10;
    mul.lo.u32 %r13, %r13, 2246822519;
    or.b32 %r13, %r13, 1;
    mov.u32 %r14, 0;
LOOP:
    mad.lo.u32 %r15, %r14, %r13, %r12;
    and.b32 %r15, %r15, %r11;
    shr.u32 %r16, %r15, 5;
    and.b32 %r17, %r15, 31;
    mov.u32 %r18, 1;
    shl.b32 %r18, %r18, %r17;
    mul.wide.u32 %rd10, %r16, 4;
    add.u64 %rd11, %rd8, %rd10;
    ld.global.u32 %r19, [%rd11];
    and.b32 %r19, %r19, %r18;
    setp.eq.u32 %p2, %r19, 0;
    @%p2 bra MISS;
    add.u32 %r14, %r14, 1;
    setp.lt.u32 %p3, %r14, 3;
    @%p3 bra LOOP;
    mov.u32 %r20, 1;
    bra WRITE;
MISS:
    mov.u32 %r20, 0;
WRITE:
    cvt.u64.u32 %rd12, %r6;
    add.u64 %rd13, %rd3, %rd12;
    st.global.u8 [%rd13], %r20;
DONE:
    ret;
}
"#;
    if blooms.is_empty() || needles.is_empty() {
        return Ok(vec![Vec::new(); needles.len()]);
    }
    let chunk_count = u32::try_from(blooms.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(blooms.len()))?;
    let needle_count = u32::try_from(needles.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needles.len()))?;
    let total = blooms
        .len()
        .checked_mul(needles.len())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut desc = Vec::with_capacity(blooms.len() * 2);
    let mut guards = Vec::with_capacity(blooms.len());
    for bloom in blooms {
        if bloom.bloom.metadata().gpu_id != ctx.metadata().gpu_id {
            return Err(CudaRuntimeProbeError::InvalidDeviceCount(i32::from(
                bloom.bloom.metadata().gpu_id,
            )));
        }
        if bloom.bit_mask == 0 || !bloom.bit_mask.wrapping_add(1).is_power_of_two() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                bloom.bit_mask as usize,
            ));
        }
        let required = (u64::from(bloom.bit_mask) + 1).div_ceil(8);
        if required > bloom.bloom.metadata().allocated_bytes {
            return Err(CudaRuntimeProbeError::InvalidInputLength(required as usize));
        }
        desc.push(bloom.bloom.device_ptr());
        desc.push(u64::from(bloom.bit_mask));
        guards.push(Arc::clone(&bloom.bloom));
    }
    let primary = ctx.primary_arc();
    primary.set_current()?;
    let htod = unsafe {
        primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let dtoh = unsafe {
        primary
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let launch = unsafe {
        primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let desc_bytes = std::mem::size_of_val(desc.as_slice());
    let needle_bytes = std::mem::size_of_val(needles);
    let desc_guard = primary.lease_device_buffer_owned(desc_bytes)?;
    let needle_guard = primary.lease_device_buffer_owned(needle_bytes)?;
    let out_guard = primary.lease_device_buffer_owned(total)?;
    check_cuda(unsafe { htod(desc_guard.ptr, desc.as_ptr().cast(), desc_bytes) })?;
    check_cuda(unsafe { htod(needle_guard.ptr, needles.as_ptr().cast(), needle_bytes) })?;
    let mut ptx = PTX.to_vec();
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_chunk_bloom_probe", &ptx)?;
    let mut desc_arg = desc_guard.ptr;
    let mut chunks_arg = chunk_count;
    let mut needles_arg = needle_guard.ptr;
    let mut needle_count_arg = needle_count;
    let mut out_arg = out_guard.ptr;
    let mut args = [
        (&mut desc_arg as *mut u64).cast(),
        (&mut chunks_arg as *mut u32).cast(),
        (&mut needles_arg as *mut u64).cast(),
        (&mut needle_count_arg as *mut u32).cast(),
        (&mut out_arg as *mut u64).cast(),
    ];
    let threads = 128u32;
    let blocks = u32::try_from(total)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(total))?
        .div_ceil(threads);
    check_cuda(unsafe {
        launch(
            function,
            blocks,
            1,
            1,
            threads,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    let mut raw = vec![0u8; total];
    check_cuda(unsafe { dtoh(raw.as_mut_ptr().cast(), out_guard.ptr, total) })?;
    drop(guards);
    let mut out = vec![Vec::new(); needles.len()];
    for (needle, row) in raw.chunks_exact(blooms.len()).enumerate() {
        for (chunk, &hit) in row.iter().enumerate() {
            if hit != 0 {
                out[needle].push(chunk as u32);
            }
        }
    }
    Ok(out)
}
