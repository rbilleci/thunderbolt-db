//! Host-staged CUDA hash-join benchmark and reference ownership.
//!
//! Production resident relations use the device-resident coordinate join path. These public APIs
//! remain for the standard read-kernel roofline and focused execution tests; their input columns
//! are uploaded on every call and their coordinate pairs are read back to the host.

use std::os::raw::c_void;

use super::{CudaResidentDeviceMemory, CudaRuntimeProbeError, check_cuda, launch_on_pooled_stream};

/// Outcome of [`CudaResidentDeviceMemory::hash_join_inner_i64`]: the matched row-index pairs, or a
/// signal that the build-side join key is not unique (N:N many-to-many fan-out is a follow-up).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HashJoinOutcome {
    /// `build_idxs[i]` joins `probe_idxs[i]` (parallel vectors of absolute resident row indices).
    Pairs {
        build_idxs: Vec<u32>,
        probe_idxs: Vec<u32>,
    },
    /// A build-side join key repeated -> the unique-build-key inner join can't represent it yet.
    DuplicateBuildKey,
}

impl CudaResidentDeviceMemory {
    /// GPU inner equi-join (M5) on an int key with a UNIQUE build-side key. `build_keys`/`probe_keys`
    /// are the join-key columns sign-extended to i64 (each relation's full-scan rows). Builds an
    /// open-addressing b128 hash table on the build side (lock-free atom.cas.b128 claim) and probes;
    /// returns the matched `(build_row_idx, probe_row_idx)` pairs (output <= probe_n, since the unique
    /// build key gives each probe row <=1 match). `DuplicateBuildKey` if a build key repeats (N:N is a
    /// follow-up). The join itself is on the GPU; only the resulting index pairs come to the host.
    ///
    /// `build_validity`/`probe_validity` are optional dense LSB-first u32 validity bitmaps (bit `i` in
    /// `word[i>>5]` at `i & 31`, `1 = valid`) sized to `ceil(n/32)` words. A `0` bit marks a NULL join key,
    /// which the kernel SKIPS on-device (an equi-join `NULL = x` is UNKNOWN -> matches nothing, and a
    /// skipped build key never claims a slot, so it is invisible to the duplicate check). `None` means every
    /// key is valid; `None` on both sides keeps the no-NULL join byte-identical to the pre-V1b sentinel path.
    pub fn hash_join_inner_i64(
        &self,
        build_keys: &[i64],
        probe_keys: &[i64],
        build_validity: Option<&[u32]>,
        probe_validity: Option<&[u32]>,
    ) -> Result<HashJoinOutcome, CudaRuntimeProbeError> {
        launch_cuda_hash_join_inner_i64(
            self,
            build_keys,
            probe_keys,
            build_validity,
            probe_validity,
        )
    }

    /// GPU inner equi-join (M5 J4b) on a TEXT key with a UNIQUE build-side key. `build_texts`/`probe_texts`
    /// are staged join-key bytes uploaded per call by benchmark/reference callers. Production resident joins
    /// use device-resident payload descriptors instead. Builds an open-addressing
    /// b128 hash table keyed on a FNV-1a-64 hash of
    /// the build bytes (lock-free atom.cas.b128 claim) and VERIFIES the full bytes on a hash match, so a
    /// 64-bit hash collision between distinct texts never mis-joins or spuriously reports a duplicate.
    /// Returns the matched `(build_row_idx, probe_row_idx)` pairs (output <= probe_n); `DuplicateBuildKey`
    /// if a build text repeats (N:N is a follow-up). The MATCH (hash + byte-verify) is on the GPU; only the
    /// resulting index pairs come back.
    ///
    /// `build_validity`/`probe_validity` are optional dense LSB-first u32 validity bitmaps (`1 = valid`,
    /// `ceil(n/32)` words) marking each side's NULL keys, skipped on-device (see [`Self::hash_join_inner_i64`]
    /// for the layout + semantics). `None` on both sides keeps the no-NULL join byte-identical.
    pub fn hash_join_inner_text(
        &self,
        build_texts: &[&[u8]],
        probe_texts: &[&[u8]],
        build_validity: Option<&[u32]>,
        probe_validity: Option<&[u32]>,
    ) -> Result<HashJoinOutcome, CudaRuntimeProbeError> {
        launch_cuda_hash_join_inner_text(
            self,
            build_texts,
            probe_texts,
            build_validity,
            probe_validity,
        )
    }

    /// GPU inner equi-join (M5 N:N) on an int key where BOTH sides may have DUPLICATE keys -- the general
    /// many-to-many join (each key's build rows × probe rows). `build_keys`/`probe_keys` are sign-extended
    /// to i64. Builds a per-bucket CHAIN of all build rows with each key (lock-free atom.cas.b64 claim +
    /// atom.exch.b64 prepend), then for each probe row emits one pair per chained build row. Returns the
    /// matched `(build_row_idx, probe_row_idx)` pairs (output = sum over probes of its key's build count,
    /// up to build_n×probe_n). The join is on the GPU; only the index pairs come back. No DuplicateBuildKey
    /// (duplicates are the point).
    ///
    /// `build_validity`/`probe_validity` are optional dense LSB-first u32 validity bitmaps (`1 = valid`,
    /// `ceil(n/32)` words): a `0` build bit skips chaining that build row, a `0` probe bit skips emitting
    /// for that probe row (an equi-join NULL key matches nothing). `None` on both sides is byte-identical.
    pub fn hash_join_inner_i64_nn(
        &self,
        build_keys: &[i64],
        probe_keys: &[i64],
        build_validity: Option<&[u32]>,
        probe_validity: Option<&[u32]>,
    ) -> Result<(Vec<u32>, Vec<u32>), CudaRuntimeProbeError> {
        launch_cuda_hash_join_inner_i64_nn(
            self,
            build_keys,
            probe_keys,
            build_validity,
            probe_validity,
        )
    }

    /// GPU inner equi-join (M5 N:N) on a TEXT (or 16-byte numeric/uuid) key where BOTH sides may have
    /// DUPLICATE keys -- the many-to-many text join (each key's build rows × probe rows). Combines the J4b
    /// text hash (FNV + byte-verify over the dense build/probe buffer) with the i64 N:N chaining: each
    /// build row prepends to its bucket's chain, and each probe emits one pair per chained build row.
    /// Returns the matched `(build_row_idx, probe_row_idx)` pairs (output = sum over probes of its key's
    /// build count). The MATCH is on the GPU; only the index pairs come back. No DuplicateBuildKey.
    ///
    /// `build_validity`/`probe_validity` are optional dense LSB-first u32 validity bitmaps (`1 = valid`,
    /// `ceil(n/32)` words) marking each side's NULL keys, skipped on-device as in
    /// [`Self::hash_join_inner_i64_nn`]. `None` on both sides is byte-identical.
    pub fn hash_join_inner_text_nn(
        &self,
        build_texts: &[&[u8]],
        probe_texts: &[&[u8]],
        build_validity: Option<&[u32]>,
        probe_validity: Option<&[u32]>,
    ) -> Result<(Vec<u32>, Vec<u32>), CudaRuntimeProbeError> {
        launch_cuda_hash_join_inner_text_nn(
            self,
            build_texts,
            probe_texts,
            build_validity,
            probe_validity,
        )
    }
}

fn validate_validity_words(
    words: Option<&[u32]>,
    row_count: usize,
) -> Result<Option<&[u32]>, CudaRuntimeProbeError> {
    let expected = row_count.div_ceil(32);
    if let Some(words) = words {
        if words.len() != expected {
            return Err(CudaRuntimeProbeError::InvalidInputLength(words.len()));
        }
    }
    Ok(words)
}

/// GPU inner equi-join (M5), int key, UNIQUE build key (see
/// [`CudaResidentDeviceMemory::hash_join_inner_i64`]). Fills a b128 hash table to EMPTY128, uploads the
/// two key columns, runs the build (atom.cas.b128 claim + dup detect) then the probe (lookup +
/// atom.add append), and returns the matched (build_idx, probe_idx) pairs -- or DuplicateBuildKey.
fn launch_cuda_hash_join_inner_i64(
    resident: &CudaResidentDeviceMemory,
    build_keys: &[i64],
    probe_keys: &[i64],
    build_validity: Option<&[u32]>,
    probe_validity: Option<&[u32]>,
) -> Result<HashJoinOutcome, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let build_n = build_keys.len();
    let probe_n = probe_keys.len();
    let build_valid_words = validate_validity_words(build_validity, build_n)?;
    let probe_valid_words = validate_validity_words(probe_validity, probe_n)?;
    // i64::MIN is the EMPTY128 hi sentinel -- a key of that value would read back as an empty slot
    // (a silent missed match). int4 join keys sign-extend into [-2^31, 2^31) and never reach it; an
    // int8 join key that IS i64::MIN needs the dedicated-slot route (like the GROUP BY i64::MIN key) --
    // a follow-up when int8 join keys land. Reject it here rather than return a wrong answer.
    if build_keys.iter().chain(probe_keys).any(|&k| k == i64::MIN) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    // Empty inputs -> no matches (an inner join with an empty side yields nothing).
    if build_n == 0 || probe_n == 0 {
        return Ok(HashJoinOutcome::Pairs {
            build_idxs: Vec::new(),
            probe_idxs: Vec::new(),
        });
    }
    let build_n_u64 = build_n as u64;
    let probe_n_u64 = probe_n as u64;
    // Hash table sized to the next pow2 > 2*build_n so open-addressing probing always terminates.
    let npot = build_n
        .checked_mul(2)
        .and_then(|x| x.checked_next_power_of_two())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(build_n))?
        .max(2);
    let mask = (npot - 1) as u64;
    let slot_bytes = npot
        .checked_mul(16)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let build_bytes = build_n
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(build_n))?;
    let probe_bytes = probe_n
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(probe_n))?;
    let pairs_bytes = probe_n
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(probe_n))?; // probe_n pairs * 2 u32
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
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
    let fill_fn = primary.cached_function(c"gpu_db_fill_i128", &ptx)?;
    let build_fn = primary.cached_function(c"gpu_db_hash_join_build_i32", &ptx)?;
    let probe_fn = primary.cached_function(c"gpu_db_hash_join_probe_i32", &ptx)?;
    let slot_keys = primary.lease_device_buffer(slot_bytes)?;
    let build_dev = primary.lease_device_buffer(build_bytes)?;
    let probe_dev = primary.lease_device_buffer(probe_bytes)?;
    let dup_flag = primary.lease_device_buffer(8)?;
    let cursor = primary.lease_device_buffer(8)?;
    let pairs = primary.lease_device_buffer(pairs_bytes)?;
    // Optional validity bitmaps (V1b-wire): lease + (below) upload a dense LSB-first u32 bitmap per side;
    // the kernel skips a NULL key (bit 0). A missing bitmap passes the u64::MAX sentinel => every key valid
    // (byte-identical). The buffers outlive both launches (build reads build_valid in phase
    // 1, probe reads probe_valid in phase 2 -- both are uploaded in phase 1, before the BUILD).
    let build_valid_dev = match build_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let probe_valid_dev = match probe_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let build_valid_arg = build_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    let probe_valid_arg = probe_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    const BLOCK: u32 = 256;
    let grid = |n: usize| (n.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    let zero8 = [0u64];
    // Phase 1: upload keys + validity bitmaps, zero dup_flag/cursor, fill the slot table to EMPTY128, BUILD.
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        for (dst, src, bytes) in [
            (
                build_dev.ptr,
                build_keys.as_ptr().cast::<c_void>(),
                build_bytes,
            ),
            (
                probe_dev.ptr,
                probe_keys.as_ptr().cast::<c_void>(),
                probe_bytes,
            ),
            (dup_flag.ptr, zero8.as_ptr().cast::<c_void>(), 8usize),
            (cursor.ptr, zero8.as_ptr().cast::<c_void>(), 8usize),
        ] {
            let rc = unsafe { htod_async(dst, src, bytes, stream) };
            if rc != 0 {
                return rc;
            }
        }
        for (dev, words) in [
            (&build_valid_dev, build_valid_words),
            (&probe_valid_dev, probe_valid_words),
        ] {
            if let (Some(d), Some(w)) = (dev, words) {
                let rc =
                    unsafe { htod_async(d.ptr, w.as_ptr().cast::<c_void>(), w.len() * 4, stream) };
                if rc != 0 {
                    return rc;
                }
            }
        }
        // fill_i128(slot_keys, npot, lo=0, hi=i64::MIN) -> EMPTY128
        let mut f0 = slot_keys.ptr;
        let mut f1 = npot as u64;
        let mut f2 = 0u64;
        let mut f3 = i64::MIN as u64;
        let mut fargs = [
            (&mut f0 as *mut u64).cast::<c_void>(),
            (&mut f1 as *mut u64).cast::<c_void>(),
            (&mut f2 as *mut u64).cast::<c_void>(),
            (&mut f3 as *mut u64).cast::<c_void>(),
        ];
        let rc = unsafe {
            cu_launch_kernel(
                fill_fn,
                grid(npot),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // build(build_keys, build_n, slot_keys, mask, dup_flag, build_validity). build_valid_arg = u64::MAX
        // (no bitmap => every key valid, byte-identical) or a real bitmap ptr (skip NULL build keys).
        let mut b0 = build_dev.ptr;
        let mut b1 = build_n_u64;
        let mut b2 = slot_keys.ptr;
        let mut b3 = mask;
        let mut b4 = dup_flag.ptr;
        let mut b5 = build_valid_arg;
        let mut bargs = [
            (&mut b0 as *mut u64).cast::<c_void>(),
            (&mut b1 as *mut u64).cast::<c_void>(),
            (&mut b2 as *mut u64).cast::<c_void>(),
            (&mut b3 as *mut u64).cast::<c_void>(),
            (&mut b4 as *mut u64).cast::<c_void>(),
            (&mut b5 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                build_fn,
                grid(build_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                bargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    // Reject a non-unique build key (N:N is a follow-up).
    let mut dup_host = [0u64];
    check_cuda(unsafe { cu_memcpy_dtoh(dup_host.as_mut_ptr().cast::<c_void>(), dup_flag.ptr, 8) })?;
    if dup_host[0] != 0 {
        return Ok(HashJoinOutcome::DuplicateBuildKey);
    }
    // Phase 2: PROBE -> append matched (build_idx, probe_idx) at the atomic cursor.
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        // probe(probe_keys, probe_n, slot_keys, mask, out_pairs, out_cursor, probe_validity). probe_valid_arg
        // = u64::MAX (no bitmap => every key valid, byte-identical) or a real bitmap ptr (skip NULL probes).
        let mut p0 = probe_dev.ptr;
        let mut p1 = probe_n_u64;
        let mut p2 = slot_keys.ptr;
        let mut p3 = mask;
        let mut p4 = pairs.ptr;
        let mut p5 = cursor.ptr;
        let mut p6 = probe_valid_arg;
        let mut pargs = [
            (&mut p0 as *mut u64).cast::<c_void>(),
            (&mut p1 as *mut u64).cast::<c_void>(),
            (&mut p2 as *mut u64).cast::<c_void>(),
            (&mut p3 as *mut u64).cast::<c_void>(),
            (&mut p4 as *mut u64).cast::<c_void>(),
            (&mut p5 as *mut u64).cast::<c_void>(),
            (&mut p6 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                probe_fn,
                grid(probe_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                pargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let mut cur_host = [0u64];
    check_cuda(unsafe { cu_memcpy_dtoh(cur_host.as_mut_ptr().cast::<c_void>(), cursor.ptr, 8) })?;
    let n_pairs =
        usize::try_from(cur_host[0]).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n_pairs > probe_n {
        // The unique-build-key invariant bounds matches by probe_n; more means a kernel bug.
        return Err(CudaRuntimeProbeError::InvalidInputLength(n_pairs));
    }
    let mut flat = vec![0u32; n_pairs * 2];
    if n_pairs > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(flat.as_mut_ptr().cast::<c_void>(), pairs.ptr, n_pairs * 8)
        })?;
    }
    let mut build_idxs = Vec::with_capacity(n_pairs);
    let mut probe_idxs = Vec::with_capacity(n_pairs);
    for pair in flat.chunks_exact(2) {
        build_idxs.push(pair[0]);
        probe_idxs.push(pair[1]);
    }
    Ok(HashJoinOutcome::Pairs {
        build_idxs,
        probe_idxs,
    })
}

/// Backs [`CudaResidentDeviceMemory::hash_join_inner_text`] (M5 J4b). Packs the build + probe key bytes
/// into ONE dense device buffer `[build_offsets][build_bytes][probe_offsets(8-aligned)][probe_bytes]`
/// (offsets are 8-byte entries the kernels read as 2x ld.u32), fills the b128 slot table to EMPTY128, runs
/// the FNV-hash BUILD (lock-free atom.cas.b128 + byte-verify on collision), reads dup_flag, then the PROBE
/// (verify + atom.add append). Output <= probe_n. cuCtxSynchronize between phases (a fully-drained launch,
/// off the bool-GROUP-BY hazard); the upload/lease buffers outlive every launch that reads them.
fn launch_cuda_hash_join_inner_text(
    resident: &CudaResidentDeviceMemory,
    build_texts: &[&[u8]],
    probe_texts: &[&[u8]],
    build_validity: Option<&[u32]>,
    probe_validity: Option<&[u32]>,
) -> Result<HashJoinOutcome, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let build_n = build_texts.len();
    let probe_n = probe_texts.len();
    let build_valid_words = validate_validity_words(build_validity, build_n)?;
    let probe_valid_words = validate_validity_words(probe_validity, probe_n)?;
    // Empty inputs -> no matches (an inner join with an empty side yields nothing).
    if build_n == 0 || probe_n == 0 {
        return Ok(HashJoinOutcome::Pairs {
            build_idxs: Vec::new(),
            probe_idxs: Vec::new(),
        });
    }
    // Dense per-side (offsets, bytes); offsets[i] is the byte offset of text i WITHIN its bytes section.
    let mut build_offsets: Vec<u64> = Vec::with_capacity(build_n + 1);
    let mut build_bytes: Vec<u8> = Vec::new();
    build_offsets.push(0);
    for t in build_texts {
        build_bytes.extend_from_slice(t);
        build_offsets.push(build_bytes.len() as u64);
    }
    let mut probe_offsets: Vec<u64> = Vec::with_capacity(probe_n + 1);
    let mut probe_bytes: Vec<u8> = Vec::new();
    probe_offsets.push(0);
    for t in probe_texts {
        probe_bytes.extend_from_slice(t);
        probe_offsets.push(probe_bytes.len() as u64);
    }
    // One buffer: [build_offsets][build_bytes][pad to 8][probe_offsets][probe_bytes]. The offsets sections
    // land 8-aligned (the kernels' 2x ld.u32 reads are 716-safe regardless).
    let mut payload: Vec<u8> = Vec::new();
    for &o in &build_offsets {
        payload.extend_from_slice(&o.to_le_bytes());
    }
    let build_bytes_off = payload.len() as u64;
    payload.extend_from_slice(&build_bytes);
    while !payload.len().is_multiple_of(8) {
        payload.push(0);
    }
    let probe_offsets_off = payload.len() as u64;
    for &o in &probe_offsets {
        payload.extend_from_slice(&o.to_le_bytes());
    }
    let probe_bytes_off = payload.len() as u64;
    payload.extend_from_slice(&probe_bytes);
    let build_offsets_off = 0u64;
    // Hash table sized to the next pow2 > 2*build_n so open-addressing probing always terminates.
    let npot = build_n
        .checked_mul(2)
        .and_then(|x| x.checked_next_power_of_two())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(build_n))?
        .max(2);
    let mask = (npot - 1) as u64;
    let slot_bytes = npot
        .checked_mul(16)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let pairs_bytes = probe_n
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(probe_n))?;
    let build_n_u64 = build_n as u64;
    let probe_n_u64 = probe_n as u64;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
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
    let fill_fn = primary.cached_function(c"gpu_db_fill_i128", &ptx)?;
    let build_fn = primary.cached_function(c"gpu_db_hash_join_build_text", &ptx)?;
    let probe_fn = primary.cached_function(c"gpu_db_hash_join_probe_text", &ptx)?;
    let slot_keys = primary.lease_device_buffer(slot_bytes)?;
    let payload_dev = primary.lease_device_buffer(payload.len())?;
    let dup_flag = primary.lease_device_buffer(8)?;
    let cursor = primary.lease_device_buffer(8)?;
    let pairs = primary.lease_device_buffer(pairs_bytes)?;
    // Optional validity bitmaps (V1b-wire): dense LSB-first u32 per side; a 0 bit = a NULL key the kernel
    // skips. None => u64::MAX sentinel => no bitmap => byte-identical. Both buffers upload in phase 1.
    let build_valid_dev = match build_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let probe_valid_dev = match probe_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let build_valid_arg = build_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    let probe_valid_arg = probe_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    const BLOCK: u32 = 256;
    let grid = |n: usize| (n.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    let zero8 = [0u64];
    // Phase 1: upload the dense buffer + validity bitmaps, zero dup_flag/cursor, fill EMPTY128, run BUILD.
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        for (dst, src, bytes) in [
            (
                payload_dev.ptr,
                payload.as_ptr().cast::<c_void>(),
                payload.len(),
            ),
            (dup_flag.ptr, zero8.as_ptr().cast::<c_void>(), 8usize),
            (cursor.ptr, zero8.as_ptr().cast::<c_void>(), 8usize),
        ] {
            let rc = unsafe { htod_async(dst, src, bytes, stream) };
            if rc != 0 {
                return rc;
            }
        }
        for (dev, words) in [
            (&build_valid_dev, build_valid_words),
            (&probe_valid_dev, probe_valid_words),
        ] {
            if let (Some(d), Some(w)) = (dev, words) {
                let rc =
                    unsafe { htod_async(d.ptr, w.as_ptr().cast::<c_void>(), w.len() * 4, stream) };
                if rc != 0 {
                    return rc;
                }
            }
        }
        let mut f0 = slot_keys.ptr;
        let mut f1 = npot as u64;
        let mut f2 = 0u64;
        let mut f3 = i64::MIN as u64;
        let mut fargs = [
            (&mut f0 as *mut u64).cast::<c_void>(),
            (&mut f1 as *mut u64).cast::<c_void>(),
            (&mut f2 as *mut u64).cast::<c_void>(),
            (&mut f3 as *mut u64).cast::<c_void>(),
        ];
        let rc = unsafe {
            cu_launch_kernel(
                fill_fn,
                grid(npot),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // build(payload, build_offsets_off, build_bytes_off, build_n, slots, mask, dup_flag, build_validity).
        // build_valid_arg = u64::MAX (no bitmap => valid, byte-identical) or a real bitmap ptr (skip NULLs).
        let mut b0 = payload_dev.ptr;
        let mut b1 = build_offsets_off;
        let mut b2 = build_bytes_off;
        let mut b3 = build_n_u64;
        let mut b4 = slot_keys.ptr;
        let mut b5 = mask;
        let mut b6 = dup_flag.ptr;
        let mut b7 = build_valid_arg;
        let mut bargs = [
            (&mut b0 as *mut u64).cast::<c_void>(),
            (&mut b1 as *mut u64).cast::<c_void>(),
            (&mut b2 as *mut u64).cast::<c_void>(),
            (&mut b3 as *mut u64).cast::<c_void>(),
            (&mut b4 as *mut u64).cast::<c_void>(),
            (&mut b5 as *mut u64).cast::<c_void>(),
            (&mut b6 as *mut u64).cast::<c_void>(),
            (&mut b7 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                build_fn,
                grid(build_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                bargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let mut dup_host = [0u64];
    check_cuda(unsafe { cu_memcpy_dtoh(dup_host.as_mut_ptr().cast::<c_void>(), dup_flag.ptr, 8) })?;
    if dup_host[0] != 0 {
        return Ok(HashJoinOutcome::DuplicateBuildKey);
    }
    // Phase 2: PROBE -> append matched (build_idx, probe_idx) at the atomic cursor.
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        // probe(payload, probe_off, probe_bytes, probe_n, build_off, build_bytes, slots, mask, out_pairs,
        // out_cursor, probe_validity). probe_valid_arg = u64::MAX (no bitmap, byte-identical) or a real ptr.
        let mut p0 = payload_dev.ptr;
        let mut p1 = probe_offsets_off;
        let mut p2 = probe_bytes_off;
        let mut p3 = probe_n_u64;
        let mut p4 = build_offsets_off;
        let mut p5 = build_bytes_off;
        let mut p6 = slot_keys.ptr;
        let mut p7 = mask;
        let mut p8 = pairs.ptr;
        let mut p9 = cursor.ptr;
        let mut p10 = probe_valid_arg;
        let mut pargs = [
            (&mut p0 as *mut u64).cast::<c_void>(),
            (&mut p1 as *mut u64).cast::<c_void>(),
            (&mut p2 as *mut u64).cast::<c_void>(),
            (&mut p3 as *mut u64).cast::<c_void>(),
            (&mut p4 as *mut u64).cast::<c_void>(),
            (&mut p5 as *mut u64).cast::<c_void>(),
            (&mut p6 as *mut u64).cast::<c_void>(),
            (&mut p7 as *mut u64).cast::<c_void>(),
            (&mut p8 as *mut u64).cast::<c_void>(),
            (&mut p9 as *mut u64).cast::<c_void>(),
            (&mut p10 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                probe_fn,
                grid(probe_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                pargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let mut cur_host = [0u64];
    check_cuda(unsafe { cu_memcpy_dtoh(cur_host.as_mut_ptr().cast::<c_void>(), cursor.ptr, 8) })?;
    let n_pairs =
        usize::try_from(cur_host[0]).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n_pairs > probe_n {
        return Err(CudaRuntimeProbeError::InvalidInputLength(n_pairs));
    }
    let mut flat = vec![0u32; n_pairs * 2];
    if n_pairs > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(flat.as_mut_ptr().cast::<c_void>(), pairs.ptr, n_pairs * 8)
        })?;
    }
    let mut build_idxs = Vec::with_capacity(n_pairs);
    let mut probe_idxs = Vec::with_capacity(n_pairs);
    for pair in flat.chunks_exact(2) {
        build_idxs.push(pair[0]);
        probe_idxs.push(pair[1]);
    }
    Ok(HashJoinOutcome::Pairs {
        build_idxs,
        probe_idxs,
    })
}

/// Backs [`CudaResidentDeviceMemory::hash_join_inner_i64_nn`] (M5 N:N). Builds a per-bucket CHAIN of all
/// build rows per key (`gpu_db_hash_join_build_i64_nn`: cas.b64 claim + exch.b64 prepend + next[]), then
/// EMITS one pair per (chained build row × matching probe row). Output is unbounded (up to build×probe),
/// so the emit runs TWICE: a COUNT pass (cap=0 -> the cursor counts every match, no write), then the real
/// emit (cap=total) into an exactly-sized buffer. Within phase 1 the BUILD + COUNT-emit are STREAM-ORDERED
/// (build completes before the count-emit walks `next[]`); a cuCtxSynchronize separates phase 1 from phase
/// 2 (read the cursor, then the real emit). The upload/lease buffers outlive every launch.
fn launch_cuda_hash_join_inner_i64_nn(
    resident: &CudaResidentDeviceMemory,
    build_keys: &[i64],
    probe_keys: &[i64],
    build_validity: Option<&[u32]>,
    probe_validity: Option<&[u32]>,
) -> Result<(Vec<u32>, Vec<u32>), CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let build_n = build_keys.len();
    let probe_n = probe_keys.len();
    let build_valid_words = validate_validity_words(build_validity, build_n)?;
    let probe_valid_words = validate_validity_words(probe_validity, probe_n)?;
    if build_n == 0 || probe_n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // i64::MIN aliases the EMPTY slot sentinel (a missed match); reject it (int4 keys never reach it; an
    // int8 key that IS i64::MIN is the dedicated-slot follow-up, as for the unique-build join).
    if build_keys.iter().chain(probe_keys).any(|&k| k == i64::MIN) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let build_n_u64 = build_n as u64;
    let probe_n_u64 = probe_n as u64;
    let npot = build_n
        .checked_mul(2)
        .and_then(|x| x.checked_next_power_of_two())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(build_n))?
        .max(2);
    let mask = (npot - 1) as u64;
    let slot_bytes = npot
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let build_bytes = build_n
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(build_n))?;
    let probe_bytes = probe_n
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(probe_n))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
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
    let fill_fn = primary.cached_function(c"gpu_db_fill_i64", &ptx)?;
    let build_fn = primary.cached_function(c"gpu_db_hash_join_build_i64_nn", &ptx)?;
    let emit_fn = primary.cached_function(c"gpu_db_hash_join_emit_i64_nn", &ptx)?;
    let slot_keys = primary.lease_device_buffer(slot_bytes)?;
    let slot_head = primary.lease_device_buffer(slot_bytes)?;
    let next = primary.lease_device_buffer(build_bytes)?;
    let build_dev = primary.lease_device_buffer(build_bytes)?;
    let probe_dev = primary.lease_device_buffer(probe_bytes)?;
    let cursor = primary.lease_device_buffer(8)?;
    // Optional validity bitmaps (V1b-wire): dense LSB-first u32 per side; a 0 build bit skips chaining that
    // row, a 0 probe bit skips emitting for it. None => u64::MAX sentinel (byte-identical). Both upload
    // in phase 1; the probe bitmap is read by BOTH the count-emit (phase 1) and the real emit (phase 2).
    let build_valid_dev = match build_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let probe_valid_dev = match probe_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let build_valid_arg = build_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    let probe_valid_arg = probe_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    const BLOCK: u32 = 256;
    let grid = |n: usize| (n.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    let zero8 = [0u64];
    // Phase 1: upload keys + validity, zero cursor, fill slot_keys -> EMPTY (i64::MIN) + slot_head -> END
    // (u64::MAX), BUILD the chains, then a COUNT emit (cap=0) so the cursor holds the exact output size.
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        for (dst, src, bytes) in [
            (
                build_dev.ptr,
                build_keys.as_ptr().cast::<c_void>(),
                build_bytes,
            ),
            (
                probe_dev.ptr,
                probe_keys.as_ptr().cast::<c_void>(),
                probe_bytes,
            ),
            (cursor.ptr, zero8.as_ptr().cast::<c_void>(), 8usize),
        ] {
            let rc = unsafe { htod_async(dst, src, bytes, stream) };
            if rc != 0 {
                return rc;
            }
        }
        for (dev, words) in [
            (&build_valid_dev, build_valid_words),
            (&probe_valid_dev, probe_valid_words),
        ] {
            if let (Some(d), Some(w)) = (dev, words) {
                let rc =
                    unsafe { htod_async(d.ptr, w.as_ptr().cast::<c_void>(), w.len() * 4, stream) };
                if rc != 0 {
                    return rc;
                }
            }
        }
        for (ptr, value) in [(slot_keys.ptr, i64::MIN as u64), (slot_head.ptr, u64::MAX)] {
            let mut f0 = ptr;
            let mut f1 = npot as u64;
            let mut f2 = value;
            let mut fargs = [
                (&mut f0 as *mut u64).cast::<c_void>(),
                (&mut f1 as *mut u64).cast::<c_void>(),
                (&mut f2 as *mut u64).cast::<c_void>(),
            ];
            let rc = unsafe {
                cu_launch_kernel(
                    fill_fn,
                    grid(npot),
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    fargs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        // build(..., mask, build_validity). build_valid_arg = u64::MAX (no bitmap => valid, byte-identical)
        // or a real bitmap ptr (skip NULL build keys before chaining).
        let mut b0 = build_dev.ptr;
        let mut b1 = build_n_u64;
        let mut b2 = slot_keys.ptr;
        let mut b3 = slot_head.ptr;
        let mut b4 = next.ptr;
        let mut b5 = mask;
        let mut b6 = build_valid_arg;
        let mut bargs = [
            (&mut b0 as *mut u64).cast::<c_void>(),
            (&mut b1 as *mut u64).cast::<c_void>(),
            (&mut b2 as *mut u64).cast::<c_void>(),
            (&mut b3 as *mut u64).cast::<c_void>(),
            (&mut b4 as *mut u64).cast::<c_void>(),
            (&mut b5 as *mut u64).cast::<c_void>(),
            (&mut b6 as *mut u64).cast::<c_void>(),
        ];
        let rc = unsafe {
            cu_launch_kernel(
                build_fn,
                grid(build_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                bargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // COUNT emit: cap = 0 -> every match increments the cursor, none writes. probe_valid_arg = u64::MAX
        // (no bitmap => every probe key valid, byte-identical) or a real bitmap ptr (skip NULL probe keys).
        let mut e0 = probe_dev.ptr;
        let mut e1 = probe_n_u64;
        let mut e2 = slot_keys.ptr;
        let mut e3 = slot_head.ptr;
        let mut e4 = next.ptr;
        let mut e5 = mask;
        let mut e6 = 0u64;
        let mut e7 = 0u64; // out_pairs (unused at cap=0)
        let mut e8 = cursor.ptr;
        let mut e9 = probe_valid_arg;
        let mut eargs = [
            (&mut e0 as *mut u64).cast::<c_void>(),
            (&mut e1 as *mut u64).cast::<c_void>(),
            (&mut e2 as *mut u64).cast::<c_void>(),
            (&mut e3 as *mut u64).cast::<c_void>(),
            (&mut e4 as *mut u64).cast::<c_void>(),
            (&mut e5 as *mut u64).cast::<c_void>(),
            (&mut e6 as *mut u64).cast::<c_void>(),
            (&mut e7 as *mut u64).cast::<c_void>(),
            (&mut e8 as *mut u64).cast::<c_void>(),
            (&mut e9 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                emit_fn,
                grid(probe_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                eargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let mut cur_host = [0u64];
    check_cuda(unsafe { cu_memcpy_dtoh(cur_host.as_mut_ptr().cast::<c_void>(), cursor.ptr, 8) })?;
    let total =
        usize::try_from(cur_host[0]).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let pairs_bytes = total
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(total))?;
    let pairs = primary.lease_device_buffer(pairs_bytes)?;
    let total_u64 = total as u64;
    // Phase 2: zero the cursor, EMIT for real (cap = total -> every match writes).
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe { htod_async(cursor.ptr, zero8.as_ptr().cast::<c_void>(), 8usize, stream) };
        if rc != 0 {
            return rc;
        }
        // Real emit: cap = total -> every match writes. probe_valid_arg = u64::MAX (no bitmap => valid,
        // byte-identical) or a real bitmap ptr (skip NULL probe keys); same value as the COUNT pass.
        let mut e0 = probe_dev.ptr;
        let mut e1 = probe_n_u64;
        let mut e2 = slot_keys.ptr;
        let mut e3 = slot_head.ptr;
        let mut e4 = next.ptr;
        let mut e5 = mask;
        let mut e6 = total_u64;
        let mut e7 = pairs.ptr;
        let mut e8 = cursor.ptr;
        let mut e9 = probe_valid_arg;
        let mut eargs = [
            (&mut e0 as *mut u64).cast::<c_void>(),
            (&mut e1 as *mut u64).cast::<c_void>(),
            (&mut e2 as *mut u64).cast::<c_void>(),
            (&mut e3 as *mut u64).cast::<c_void>(),
            (&mut e4 as *mut u64).cast::<c_void>(),
            (&mut e5 as *mut u64).cast::<c_void>(),
            (&mut e6 as *mut u64).cast::<c_void>(),
            (&mut e7 as *mut u64).cast::<c_void>(),
            (&mut e8 as *mut u64).cast::<c_void>(),
            (&mut e9 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                emit_fn,
                grid(probe_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                eargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let mut flat = vec![0u32; total * 2];
    check_cuda(unsafe {
        cu_memcpy_dtoh(flat.as_mut_ptr().cast::<c_void>(), pairs.ptr, pairs_bytes)
    })?;
    let mut build_idxs = Vec::with_capacity(total);
    let mut probe_idxs = Vec::with_capacity(total);
    for pair in flat.chunks_exact(2) {
        build_idxs.push(pair[0]);
        probe_idxs.push(pair[1]);
    }
    Ok((build_idxs, probe_idxs))
}

/// Backs [`CudaResidentDeviceMemory::hash_join_inner_text_nn`] (M5 N:N text). Packs the build+probe key
/// bytes into ONE dense `[build_offsets][build_bytes][probe_offsets(8-aligned)][probe_bytes]` buffer (as
/// J4b), fills the b128 slot table to EMPTY128 + slot_head to END=u64::MAX, BUILDs the per-bucket chains
/// (FNV+verify claim + exch.b64 prepend), then a 2-pass emit handles the unbounded output: a COUNT pass
/// (cap=0 -> cursor=total) sizes an exact buffer, then the REAL emit (cap=total). Build+count are
/// stream-ordered in phase 1; a cuCtxSynchronize separates phase 2. The upload/lease buffers outlive all.
fn launch_cuda_hash_join_inner_text_nn(
    resident: &CudaResidentDeviceMemory,
    build_texts: &[&[u8]],
    probe_texts: &[&[u8]],
    build_validity: Option<&[u32]>,
    probe_validity: Option<&[u32]>,
) -> Result<(Vec<u32>, Vec<u32>), CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let build_n = build_texts.len();
    let probe_n = probe_texts.len();
    let build_valid_words = validate_validity_words(build_validity, build_n)?;
    let probe_valid_words = validate_validity_words(probe_validity, probe_n)?;
    if build_n == 0 || probe_n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // Dense per-side (offsets, bytes) -> one buffer (offsets sections 8-aligned), as in the J4b launcher.
    let mut build_offsets: Vec<u64> = Vec::with_capacity(build_n + 1);
    let mut build_bytes: Vec<u8> = Vec::new();
    build_offsets.push(0);
    for t in build_texts {
        build_bytes.extend_from_slice(t);
        build_offsets.push(build_bytes.len() as u64);
    }
    let mut probe_offsets: Vec<u64> = Vec::with_capacity(probe_n + 1);
    let mut probe_bytes: Vec<u8> = Vec::new();
    probe_offsets.push(0);
    for t in probe_texts {
        probe_bytes.extend_from_slice(t);
        probe_offsets.push(probe_bytes.len() as u64);
    }
    let mut payload: Vec<u8> = Vec::new();
    for &o in &build_offsets {
        payload.extend_from_slice(&o.to_le_bytes());
    }
    let build_bytes_off = payload.len() as u64;
    payload.extend_from_slice(&build_bytes);
    while !payload.len().is_multiple_of(8) {
        payload.push(0);
    }
    let probe_offsets_off = payload.len() as u64;
    for &o in &probe_offsets {
        payload.extend_from_slice(&o.to_le_bytes());
    }
    let probe_bytes_off = payload.len() as u64;
    payload.extend_from_slice(&probe_bytes);
    let build_offsets_off = 0u64;
    let npot = build_n
        .checked_mul(2)
        .and_then(|x| x.checked_next_power_of_two())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(build_n))?
        .max(2);
    let mask = (npot - 1) as u64;
    let slot_bytes = npot
        .checked_mul(16)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let head_bytes = npot
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(npot))?;
    let next_bytes = build_n
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(build_n))?;
    let build_n_u64 = build_n as u64;
    let probe_n_u64 = probe_n as u64;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
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
    let fill128_fn = primary.cached_function(c"gpu_db_fill_i128", &ptx)?;
    let fill64_fn = primary.cached_function(c"gpu_db_fill_i64", &ptx)?;
    let build_fn = primary.cached_function(c"gpu_db_hash_join_build_text_nn", &ptx)?;
    let emit_fn = primary.cached_function(c"gpu_db_hash_join_emit_text_nn", &ptx)?;
    let slot_keys = primary.lease_device_buffer(slot_bytes)?;
    let slot_head = primary.lease_device_buffer(head_bytes)?;
    let next = primary.lease_device_buffer(next_bytes)?;
    let payload_dev = primary.lease_device_buffer(payload.len())?;
    let cursor = primary.lease_device_buffer(8)?;
    // Optional validity bitmaps (V1b-wire): dense LSB-first u32 per side; a 0 build bit skips chaining, a 0
    // probe bit skips emitting. None => u64::MAX sentinel (byte-identical). Both upload in phase 1; the
    // probe bitmap is read by BOTH the count emit (phase 1) and the real emit (phase 2) via emit_launch.
    let build_valid_dev = match build_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let probe_valid_dev = match probe_valid_words {
        Some(w) => Some(primary.lease_device_buffer(w.len() * 4)?),
        None => None,
    };
    let build_valid_arg = build_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    let probe_valid_arg = probe_valid_dev.as_ref().map_or(u64::MAX, |d| d.ptr);
    const BLOCK: u32 = 256;
    let grid = |n: usize| (n.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    let zero8 = [0u64];
    // Build the 13-arg emit kernel argument array (cap + out_pairs are mutated between the count / real
    // passes). `out` is the pairs device pointer (0 in the count pass, where cap=0 prevents any write).
    let emit_launch = |stream: *mut c_void, cap: u64, out: u64| -> i32 {
        let mut e0 = payload_dev.ptr;
        let mut e1 = probe_offsets_off;
        let mut e2 = probe_bytes_off;
        let mut e3 = probe_n_u64;
        let mut e4 = build_offsets_off;
        let mut e5 = build_bytes_off;
        let mut e6 = slot_keys.ptr;
        let mut e7 = slot_head.ptr;
        let mut e8 = next.ptr;
        let mut e9 = mask;
        let mut e10 = cap;
        let mut e11 = out;
        let mut e12 = cursor.ptr;
        // probe_valid_arg = u64::MAX (no bitmap => every probe key valid, byte-identical) or a real bitmap
        // ptr (skip NULL probe keys); the same value drives both the count and the real emit pass.
        let mut e13 = probe_valid_arg;
        let mut eargs = [
            (&mut e0 as *mut u64).cast::<c_void>(),
            (&mut e1 as *mut u64).cast::<c_void>(),
            (&mut e2 as *mut u64).cast::<c_void>(),
            (&mut e3 as *mut u64).cast::<c_void>(),
            (&mut e4 as *mut u64).cast::<c_void>(),
            (&mut e5 as *mut u64).cast::<c_void>(),
            (&mut e6 as *mut u64).cast::<c_void>(),
            (&mut e7 as *mut u64).cast::<c_void>(),
            (&mut e8 as *mut u64).cast::<c_void>(),
            (&mut e9 as *mut u64).cast::<c_void>(),
            (&mut e10 as *mut u64).cast::<c_void>(),
            (&mut e11 as *mut u64).cast::<c_void>(),
            (&mut e12 as *mut u64).cast::<c_void>(),
            (&mut e13 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                emit_fn,
                grid(probe_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                eargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    };
    // Phase 1: upload payload + validity, zero cursor, fill slot_keys EMPTY128 + slot_head END, BUILD,
    // COUNT (cap=0).
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        for (dst, src, bytes) in [
            (
                payload_dev.ptr,
                payload.as_ptr().cast::<c_void>(),
                payload.len(),
            ),
            (cursor.ptr, zero8.as_ptr().cast::<c_void>(), 8usize),
        ] {
            let rc = unsafe { htod_async(dst, src, bytes, stream) };
            if rc != 0 {
                return rc;
            }
        }
        for (dev, words) in [
            (&build_valid_dev, build_valid_words),
            (&probe_valid_dev, probe_valid_words),
        ] {
            if let (Some(d), Some(w)) = (dev, words) {
                let rc =
                    unsafe { htod_async(d.ptr, w.as_ptr().cast::<c_void>(), w.len() * 4, stream) };
                if rc != 0 {
                    return rc;
                }
            }
        }
        let mut f0 = slot_keys.ptr;
        let mut f1 = npot as u64;
        let mut f2 = 0u64;
        let mut f3 = i64::MIN as u64;
        let mut fargs = [
            (&mut f0 as *mut u64).cast::<c_void>(),
            (&mut f1 as *mut u64).cast::<c_void>(),
            (&mut f2 as *mut u64).cast::<c_void>(),
            (&mut f3 as *mut u64).cast::<c_void>(),
        ];
        let rc = unsafe {
            cu_launch_kernel(
                fill128_fn,
                grid(npot),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        let mut g0 = slot_head.ptr;
        let mut g1 = npot as u64;
        let mut g2 = u64::MAX;
        let mut gargs = [
            (&mut g0 as *mut u64).cast::<c_void>(),
            (&mut g1 as *mut u64).cast::<c_void>(),
            (&mut g2 as *mut u64).cast::<c_void>(),
        ];
        let rc = unsafe {
            cu_launch_kernel(
                fill64_fn,
                grid(npot),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                gargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // build(payload, build_off, build_bytes, build_n, slots, slot_head, next, mask, build_validity).
        // build_valid_arg = u64::MAX (no bitmap => valid, byte-identical) or a real bitmap ptr (skip NULLs).
        let mut b0 = payload_dev.ptr;
        let mut b1 = build_offsets_off;
        let mut b2 = build_bytes_off;
        let mut b3 = build_n_u64;
        let mut b4 = slot_keys.ptr;
        let mut b5 = slot_head.ptr;
        let mut b6 = next.ptr;
        let mut b7 = mask;
        let mut b8 = build_valid_arg;
        let mut bargs = [
            (&mut b0 as *mut u64).cast::<c_void>(),
            (&mut b1 as *mut u64).cast::<c_void>(),
            (&mut b2 as *mut u64).cast::<c_void>(),
            (&mut b3 as *mut u64).cast::<c_void>(),
            (&mut b4 as *mut u64).cast::<c_void>(),
            (&mut b5 as *mut u64).cast::<c_void>(),
            (&mut b6 as *mut u64).cast::<c_void>(),
            (&mut b7 as *mut u64).cast::<c_void>(),
            (&mut b8 as *mut u64).cast::<c_void>(),
        ];
        let rc = unsafe {
            cu_launch_kernel(
                build_fn,
                grid(build_n),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                bargs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        emit_launch(stream, 0, 0)
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let mut cur_host = [0u64];
    check_cuda(unsafe { cu_memcpy_dtoh(cur_host.as_mut_ptr().cast::<c_void>(), cursor.ptr, 8) })?;
    let total =
        usize::try_from(cur_host[0]).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let pairs_bytes = total
        .checked_mul(8)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(total))?;
    let pairs = primary.lease_device_buffer(pairs_bytes)?;
    let total_u64 = total as u64;
    // Phase 2: zero the cursor, EMIT for real (cap = total).
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe { htod_async(cursor.ptr, zero8.as_ptr().cast::<c_void>(), 8usize, stream) };
        if rc != 0 {
            return rc;
        }
        emit_launch(stream, total_u64, pairs.ptr)
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let mut flat = vec![0u32; total * 2];
    check_cuda(unsafe {
        cu_memcpy_dtoh(flat.as_mut_ptr().cast::<c_void>(), pairs.ptr, pairs_bytes)
    })?;
    let mut build_idxs = Vec::with_capacity(total);
    let mut probe_idxs = Vec::with_capacity(total);
    for pair in flat.chunks_exact(2) {
        build_idxs.push(pair[0]);
        probe_idxs.push(pair[1]);
    }
    Ok((build_idxs, probe_idxs))
}
