//! Bounded SHA-256 over canonical byte ranges that already reside on one CUDA device.
//!
//! The host supplies only `(device pointer, byte length)` descriptors; source bytes and the
//! contiguous 32-byte outputs remain resident.  This is intentionally an execution primitive,
//! not a logical-generation, WAL, recovery, or publication authority.

use std::ffi::c_void;

use crate::{
    launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError, GpuPrimaryContext,
};

pub(crate) const SHA256_PTX: &[u8] = include_bytes!("sha256_kernel.ptx");
pub(crate) const SHA256_DESCRIPTOR_WORDS: usize = 2;
pub(crate) const SHA256_DESCRIPTOR_BYTES: usize =
    SHA256_DESCRIPTOR_WORDS * std::mem::size_of::<u64>();
pub(crate) const SHA256_THREADS_PER_BLOCK: u32 = 128;
pub(crate) const SHA256_MAX_GRID_X: u32 = 65_535;

/// Bytes in one SHA-256 digest.
pub const CUDA_SHA256_DIGEST_BYTES: u64 = 32;

/// Largest byte range SHA-256 can encode: the SHA-256 final length field is an unsigned 64-bit
/// bit count, so inputs larger than this have no canonical SHA-256 representation.
pub const CUDA_SHA256_MAX_INPUT_BYTES: u64 = u64::MAX / 8;

/// Canonical upper bound on independently described SHA-256 messages in one launch.  It keeps
/// host descriptor staging bounded at [`CUDA_SHA256_MAX_DESCRIPTOR_BYTES`]; callers that have
/// more inputs batch them explicitly while the kernel's grid-stride loop remains valid.
pub const CUDA_SHA256_MAX_BATCH_BUFFERS: usize = 1_048_576;

/// Canonical maximum host/device descriptor staging for one SHA-256 launch: 16 bytes per input
/// `(device_ptr, byte_len)` descriptor and no source payload bytes.
pub const CUDA_SHA256_MAX_DESCRIPTOR_BYTES: usize =
    CUDA_SHA256_MAX_BATCH_BUFFERS * SHA256_DESCRIPTOR_BYTES;

/// One canonical byte range already resident on the GPU.
///
/// `byte_offset..byte_offset + byte_len` is validated before the descriptor reaches the device.
/// Empty ranges are valid and hash as the standard SHA-256 empty message.
#[derive(Debug, Clone, Copy)]
pub struct CudaSha256DeviceBuffer<'a> {
    pub memory: &'a CudaResidentDeviceMemory,
    pub byte_offset: u64,
    pub byte_len: u64,
}

impl CudaSha256DeviceBuffer<'_> {
    /// Describe one explicit canonical byte range without copying it through the host.
    pub fn new(
        memory: &CudaResidentDeviceMemory,
        byte_offset: u64,
        byte_len: u64,
    ) -> CudaSha256DeviceBuffer<'_> {
        CudaSha256DeviceBuffer {
            memory,
            byte_offset,
            byte_len,
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) type CuLaunchKernel = unsafe extern "C" fn(
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

fn invalid_input_length(value: u64) -> CudaRuntimeProbeError {
    CudaRuntimeProbeError::InvalidInputLength(usize::try_from(value).unwrap_or(usize::MAX))
}

fn checked_descriptor_bytes(input_count: usize) -> Result<usize, CudaRuntimeProbeError> {
    if input_count == 0 || input_count > CUDA_SHA256_MAX_BATCH_BUFFERS {
        return Err(CudaRuntimeProbeError::InvalidInputLength(input_count));
    }
    let bytes = input_count
        .checked_mul(SHA256_DESCRIPTOR_BYTES)
        .ok_or_else(|| invalid_input_length(u64::MAX))?;
    if bytes > CUDA_SHA256_MAX_DESCRIPTOR_BYTES {
        return Err(CudaRuntimeProbeError::InvalidInputLength(input_count));
    }
    Ok(bytes)
}

pub(crate) fn checked_completion_descriptor_bytes(
    input_count: usize,
) -> Result<usize, CudaRuntimeProbeError> {
    checked_descriptor_bytes(input_count)
}

fn checked_span(
    memory: &CudaResidentDeviceMemory,
    byte_offset: u64,
    byte_len: u64,
) -> Result<u64, CudaRuntimeProbeError> {
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| invalid_input_length(byte_len))?;
    if end > memory.metadata().allocated_bytes {
        return Err(invalid_input_length(byte_len));
    }
    memory
        .device_ptr()
        .checked_add(byte_offset)
        .ok_or_else(|| invalid_input_length(byte_offset))
}

pub(crate) fn checked_completion_span(
    memory: &CudaResidentDeviceMemory,
    byte_offset: u64,
    byte_len: u64,
) -> Result<u64, CudaRuntimeProbeError> {
    checked_span(memory, byte_offset, byte_len)
}

/// Both spans have already passed [`checked_span`], so their end offsets cannot overflow.  Empty
/// SHA-256 messages do not dereference their source pointer and therefore do not conflict with an
/// output range in the same allocation.
fn spans_overlap(left_offset: u64, left_len: u64, right_offset: u64, right_len: u64) -> bool {
    left_len != 0
        && right_len != 0
        && left_offset < right_offset + right_len
        && right_offset < left_offset + left_len
}

pub(crate) fn resolve_sha256_launch(
    primary: &GpuPrimaryContext,
) -> Result<(*mut c_void, CuLaunchKernel), CudaRuntimeProbeError> {
    let mut ptx = Vec::with_capacity(SHA256_PTX.len() + 1);
    ptx.extend_from_slice(SHA256_PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_sha256_buffers", &ptx)?;
    let launch = unsafe {
        *primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    Ok((function, launch))
}

/// Resolve the one closed runtime-generation-v1 genesis-chain kernel.  Its domain grammar and
/// output order are compiled into PTX; callers can provide only the device-resident database ID
/// configuration owned by the completion transport.
pub(crate) fn resolve_runtime_generation_v1_genesis_launch(
    primary: &GpuPrimaryContext,
) -> Result<(*mut c_void, CuLaunchKernel), CudaRuntimeProbeError> {
    let mut ptx = Vec::with_capacity(SHA256_PTX.len() + 1);
    ptx.extend_from_slice(SHA256_PTX);
    ptx.push(0);
    let function = primary.cached_function(c"gpu_db_runtime_generation_v1_genesis_roots", &ptx)?;
    let launch = unsafe {
        *primary
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    Ok((function, launch))
}

pub(crate) unsafe fn launch_sha256_kernel(
    launch: CuLaunchKernel,
    function: *mut c_void,
    input_count: u32,
    descriptor_device_ptr: u64,
    output_ptr: u64,
    stream: *mut c_void,
) -> i32 {
    let mut descriptor_arg = descriptor_device_ptr;
    let mut count_arg = input_count;
    let mut output_arg = output_ptr;
    let mut args = [
        (&mut descriptor_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u32).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    let blocks = count_arg
        .div_ceil(SHA256_THREADS_PER_BLOCK)
        .clamp(1, SHA256_MAX_GRID_X);
    launch(
        function,
        blocks,
        1,
        1,
        SHA256_THREADS_PER_BLOCK,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        std::ptr::null_mut(),
    )
}

/// Launch the closed v1 genesis-chain program.  The configuration is a 16-byte device record:
/// the database-ID device pointer followed by the fixed root-format version.  The kernel itself
/// owns every ASCII domain, field order, child link, and semantic output slot.
pub(crate) unsafe fn launch_runtime_generation_v1_genesis_kernel(
    launch: CuLaunchKernel,
    function: *mut c_void,
    configuration_device_ptr: u64,
    output_ptr: u64,
    stream: *mut c_void,
) -> i32 {
    let mut configuration_arg = configuration_device_ptr;
    let mut output_arg = output_ptr;
    let mut args = [
        (&mut configuration_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
    ];
    launch(
        function,
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
}

impl CudaResidentDeviceMemory {
    /// Hash one or more canonical device byte ranges into this device allocation.
    ///
    /// The digest for `inputs[n]` is written at
    /// `output_byte_offset + n * CUDA_SHA256_DIGEST_BYTES`.  Inputs and output must share the
    /// exact primary CUDA context; the only HtoD traffic is the fixed 16-byte-per-input descriptor
    /// list.  Completion synchronizes the private execution stream, so the caller can submit a
    /// dependent device kernel immediately without a host digest copy.
    pub fn sha256_canonical_buffers(
        &self,
        inputs: &[CudaSha256DeviceBuffer<'_>],
        output_byte_offset: u64,
    ) -> Result<(), CudaRuntimeProbeError> {
        let descriptor_bytes = checked_descriptor_bytes(inputs.len())?;
        let output_bytes = u64::try_from(inputs.len())
            .ok()
            .and_then(|count| count.checked_mul(CUDA_SHA256_DIGEST_BYTES))
            .ok_or_else(|| invalid_input_length(u64::MAX))?;
        let output_ptr = checked_span(self, output_byte_offset, output_bytes)?;

        let descriptor_capacity = descriptor_bytes / std::mem::size_of::<u64>();
        let mut descriptors = Vec::with_capacity(descriptor_capacity);
        for input in inputs {
            if input.byte_len > CUDA_SHA256_MAX_INPUT_BYTES
                || input.memory.metadata().gpu_id != self.metadata().gpu_id
                || !std::ptr::eq(input.memory.primary(), self.primary())
            {
                return Err(invalid_input_length(input.byte_len));
            }
            let input_ptr = checked_span(input.memory, input.byte_offset, input.byte_len)?;
            if input.memory.allocation_identity() == self.allocation_identity()
                && spans_overlap(
                    input.byte_offset,
                    input.byte_len,
                    output_byte_offset,
                    output_bytes,
                )
            {
                return Err(invalid_input_length(input.byte_len));
            }
            descriptors.extend_from_slice(&[input_ptr, input.byte_len]);
        }
        debug_assert_eq!(
            descriptors.len() * std::mem::size_of::<u64>(),
            descriptor_bytes
        );

        let primary = self.primary();
        primary.set_current()?;
        let descriptor_device = primary.lease_device_buffer(descriptor_bytes)?;
        let hto_d = primary
            .cu_memcpy_htod_async
            .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
        let (function, launch) = resolve_sha256_launch(primary)?;
        launch_on_pooled_stream(self, None, |stream, _| {
            let copy = unsafe {
                hto_d(
                    descriptor_device.ptr,
                    descriptors.as_ptr().cast::<c_void>(),
                    descriptor_bytes,
                    stream,
                )
            };
            if copy != 0 {
                return copy;
            }
            unsafe {
                launch_sha256_kernel(
                    launch,
                    function,
                    inputs.len() as u32,
                    descriptor_device.ptr,
                    output_ptr,
                    stream,
                )
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDriverRuntime;
    use sha2::{Digest, Sha256};

    const EMPTY_SHA256: [u8; 32] = [
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55,
    ];
    const ABC_SHA256: [u8; 32] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    const MULTI_BLOCK_SHA256: [u8; 32] = [
        0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8, 0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60,
        0x39, 0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67, 0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb,
        0x06, 0xc1,
    ];

    fn cuda_runtime() -> Option<CudaDriverRuntime> {
        let runtime = CudaDriverRuntime::probe().ok()?;
        let snapshot = runtime.snapshot();
        (snapshot.driver_available && snapshot.device_count > 0).then_some(runtime)
    }

    #[test]
    fn sha256_descriptor_batch_ceiling_rejects_before_cuda_or_allocation() {
        assert_eq!(
            checked_descriptor_bytes(CUDA_SHA256_MAX_BATCH_BUFFERS),
            Ok(CUDA_SHA256_MAX_DESCRIPTOR_BYTES),
        );
        assert!(matches!(
            checked_descriptor_bytes(CUDA_SHA256_MAX_BATCH_BUFFERS + 1),
            Err(CudaRuntimeProbeError::InvalidInputLength(count))
                if count == CUDA_SHA256_MAX_BATCH_BUFFERS + 1
        ));
    }

    #[test]
    fn sha256_canonical_buffers_match_known_digests_on_device() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let abc = runtime
            .retain_device_memory_copy(0, b"prefixabc")
            .expect("resident abc source");
        let empty = runtime
            .retain_device_memory_copy(0, b"empty anchor")
            .expect("resident empty source");
        let multi_bytes = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let mut multi_payload = b"start:".to_vec();
        multi_payload.extend_from_slice(multi_bytes);
        multi_payload.extend_from_slice(b":end");
        let multi = runtime
            .retain_device_memory_copy(0, &multi_payload)
            .expect("resident multi-block source");
        let output = runtime
            .retain_device_memory_zeroed(0, CUDA_SHA256_DIGEST_BYTES * 4)
            .expect("resident digest output");

        output
            .sha256_canonical_buffers(
                &[
                    CudaSha256DeviceBuffer::new(&abc, 6, 3),
                    CudaSha256DeviceBuffer::new(&empty, 0, 0),
                    CudaSha256DeviceBuffer::new(&multi, 6, multi_bytes.len() as u64),
                ],
                CUDA_SHA256_DIGEST_BYTES,
            )
            .expect("device SHA-256 launch");
        assert_eq!(
            output
                .read_resident_bytes(CUDA_SHA256_DIGEST_BYTES, 3 * 32)
                .expect("test-only digest readback"),
            [ABC_SHA256, EMPTY_SHA256, MULTI_BLOCK_SHA256].concat(),
        );
    }

    #[test]
    fn sha256_canonical_buffers_read_every_message_byte_across_block_boundaries() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        // Each range begins at an intentionally nonzero resident offset.  The 65-byte case must
        // read source byte 64 rather than wrap to byte 0, while the five lengths cover both SHA-256
        // padding branches around the 56-byte final-length field boundary.
        const BOUNDARY_LENGTHS: [usize; 5] = [55, 56, 63, 64, 65];
        let canonical_bytes = (0_u8..65)
            .map(|byte| byte.wrapping_mul(37))
            .collect::<Vec<_>>();
        let mut payload = b"offset:".to_vec();
        payload.extend_from_slice(&canonical_bytes);
        payload.extend_from_slice(b":suffix");
        let source = runtime
            .retain_device_memory_copy(0, &payload)
            .expect("resident boundary source");
        let output = runtime
            .retain_device_memory_zeroed(
                0,
                CUDA_SHA256_DIGEST_BYTES * BOUNDARY_LENGTHS.len() as u64,
            )
            .expect("resident boundary output");
        let inputs = BOUNDARY_LENGTHS
            .iter()
            .copied()
            .map(|length| CudaSha256DeviceBuffer::new(&source, 7, length as u64))
            .collect::<Vec<_>>();

        output
            .sha256_canonical_buffers(&inputs, 0)
            .expect("device boundary SHA-256 launch");

        // Test-only reference: production code never invokes SHA-256 on the CPU or reads these
        // device digests back.  This detects an accidental `input[position & 63]` source read.
        let expected = BOUNDARY_LENGTHS
            .iter()
            .flat_map(|length| Sha256::digest(&canonical_bytes[..*length]))
            .collect::<Vec<_>>();
        assert_eq!(
            output
                .read_resident_bytes(0, expected.len())
                .expect("test-only digest readback"),
            expected,
        );
    }

    #[test]
    fn sha256_canonical_buffers_reject_out_of_bounds_ranges_before_launch() {
        let Some(runtime) = cuda_runtime() else {
            return;
        };
        let source = runtime
            .retain_device_memory_copy(0, b"abc")
            .expect("resident source");
        let output = runtime
            .retain_device_memory_zeroed(0, CUDA_SHA256_DIGEST_BYTES)
            .expect("resident output");
        assert!(matches!(
            output.sha256_canonical_buffers(&[CudaSha256DeviceBuffer::new(&source, 2, 2)], 0,),
            Err(CudaRuntimeProbeError::InvalidInputLength(2))
        ));
        let wrong_context = source.clone_with_primary_for_test(
            crate::cuda_context::distinct_gpu_primary_context_for_test(0)
                .expect("distinct source context"),
        );
        assert!(matches!(
            output
                .sha256_canonical_buffers(&[CudaSha256DeviceBuffer::new(&wrong_context, 0, 3)], 0,),
            Err(CudaRuntimeProbeError::InvalidInputLength(3))
        ));

        let overlapping = runtime
            .retain_device_memory_copy(0, b"0123456789abcdef0123456789abcdef")
            .expect("resident overlapping source/output");
        assert!(matches!(
            overlapping
                .sha256_canonical_buffers(&[CudaSha256DeviceBuffer::new(&overlapping, 4, 3)], 0,),
            Err(CudaRuntimeProbeError::InvalidInputLength(3))
        ));
    }
}
