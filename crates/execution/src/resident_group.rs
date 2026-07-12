use std::ffi::CStr;
use std::os::raw::c_void;

use super::{CudaResidentDeviceMemory, CudaRuntimeProbeError, check_cuda, launch_on_pooled_stream};

/// One GROUP BY output group: the int4 key, the row COUNT, and the SUM of the value column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupByI32Row {
    /// Group key, widened to i64 (int4 keys are sign-extended; int8 keys are exact). The engine
    /// narrows back to Int4 for an int4 GROUP BY column and keeps Int8 for an int8 one.
    pub key: i64,
    pub count: u64,
    pub sum: i64,
    /// High 64 bits of the per-group SUM when the value is int8 (the sum is accumulated as i128:
    /// `sum_hi:sum`). Zero for int4 sums (which fit `sum` alone). The engine combines them into an
    /// i128 -> numeric for `SUM(bigint)` / `AVG(bigint)`.
    pub sum_hi: i64,
    /// MIN / MAX of the value column per group (sign-extended into i64 slots; the engine narrows back
    /// to int4 / keeps int8). For COUNT(*) (value == key) they are ignored.
    pub min: i64,
    pub max: i64,
    /// High 64 bits of the per-group i128 MIN / MAX when the value is NUMERIC (stored as
    /// `min_hi:min` / `max_hi:max`). Zero for int4 / int8 MIN/MAX (which use `min` / `max` alone).
    pub min_hi: i64,
    pub max_hi: i64,
    /// MIN / MAX of a UUID value column per group, in canonical (memcmp / big-endian) byte order
    /// (`min_uuid[0]` is the first textual uuid byte). Populated only for a `MIN`/`MAX` over a uuid
    /// value (the b128 CAS-loop kernel); `[0; 16]` for every other aggregate/type.
    pub min_uuid: [u8; 16],
    pub max_uuid: [u8; 16],
    /// GROUP BY key when the key column is i128-wide (NUMERIC mantissa or UUID bytes, claimed via
    /// `atom.cas.b128`). The engine reconstructs `SqlValue::Numeric` (mantissa = this) or
    /// `SqlValue::Uuid` (`this.to_le_bytes()` = canonical bytes). `0` for i64-key paths.
    pub key_i128: i128,
    /// M3 (doc 21): this group is the NULL-KEY group — every row whose GROUP BY key is NULL forms ONE
    /// group (3VL: NULLs group together, distinct from any real value). The kernel routes NULL-key rows
    /// to a dedicated reserved slot; the engine renders this group's key as `SqlValue::Null`. `key`/
    /// `key_i128` are a don't-care placeholder for it. `false` for every real-keyed group.
    pub key_is_null: bool,
}

/// GROUP BY an int4 `key` column, aggregating COUNT(*) and SUM(int4 `sum`) over a filtered set of row
/// indices, via GPU hash aggregation (the operator axis, doc 19). Allocates an open-addressing hash
/// table sized > the row count (so probing terminates), inits slot_keys to EMPTY and the count/sum
/// accumulators to 0, runs the group-by kernel, D2Hs the slots, and host-compacts the occupied groups.
/// For COUNT(*) pass `sum_byte_offset = key_byte_offset` (the summed value is then ignored).
/// `indices` may be empty (-> no groups). NB: this is the correct baseline; a shared-memory two-level
/// kernel (far less atomic contention at low cardinality) and a GPU slot-compaction are follow-ons.
#[allow(unused_assignments)] // f0/f2 are re-read via the raw fill-arg pointers
#[allow(clippy::too_many_arguments)] // kernel launcher: offsets + per-column-type layout flags
pub(super) fn launch_cuda_group_by_i32_count_sum(
    resident: &CudaResidentDeviceMemory,
    key_byte_offset: u64,
    sum_byte_offset: u64,
    indices: &[u32],
    kernel: &'static CStr,
    value_is_int8: bool,
    key_is_int8: bool,
    value_is_numeric: bool,
    value_is_uuid: bool,
    key_is_i128: bool,
    key_is_text: bool,
    key_offsets_off: u64,
    key_bytes_off: u64,
    value_is_text: bool,
    value_offsets_off: u64,
    value_bytes_off: u64,
    key_base_override: u64,
    value_base_override: u64,
    // General all-fixed COMPOSITE key: comp_w > 0 -> the key is a `comp_w`-byte wide key per row in
    // key_base_override (built by gpu_db_build_wide_key); grouped via a (rep_idx, hash) b128 claim.
    comp_w: u64,
    // General COMPOSITE with TEXT members: n_text members, each (offsets_off, bytes_off) [16 bytes] in
    // text_desc_ptr (a device buffer the CALLER holds alive). 0 = no text members.
    n_text: u64,
    text_desc_ptr: u64,
    // M3 (doc 21) 3VL: `Some(off)` = the VALUE column's NULL validity bitmap byte offset (1 = valid) —
    // a NULL value is skipped from this pass's count/sum/min/max (the slot is still claimed by the key,
    // so the group still appears) ⇒ non-NULL aggregate semantics. `None` = no bitmap ⇒ every value valid
    // (the COUNT(*) pass + every non-nullable pass). Only the single-level kernel honors it; the engine
    // forces single-level for a nullable value, and the twolevel kernel ignores the (extra) arg.
    value_null_off: Option<u64>,
    // M3 (doc 21) 3VL: `Some(off)` = the KEY column's NULL validity bitmap byte offset (1 = valid) — a
    // NULL key routes to the dedicated NULL-KEY slot, forming one group rendered as SqlValue::Null.
    // `None` = no bitmap ⇒ every key valid. Only the single-level kernel honors it; the engine passes
    // it only for a plain int4/int8 COLUMN key (sentinel for expr/composite/text/i128 keys).
    key_null_off: Option<u64>,
    // Query-aware aggregate-selection MASK (`grouped_agg_mask`: 1=COUNT, 2=SUM, 4=MIN, 8=MAX; ALL=15).
    // Only the masked-in fields' per-row update atomics run in BOTH kernels; the slot claim + group-key
    // emit always run (group identity is mask-independent). Masked-out fields stay at their init
    // sentinels; the executor reads only what it requested, so a reduced mask is byte-identical for it.
    agg_mask: u32,
) -> Result<Vec<GroupByI32Row>, CudaRuntimeProbeError> {
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
    type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    const EMPTY: i64 = i64::MIN;

    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let count = indices.len();
    let idx_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let count_u64 = count as u64;
    // Power-of-two table sized > 2x the rows so linear probing always finds a free slot / the key.
    let nslots = count
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .map(|n| n.max(16))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    // TWO extra slots beyond the power-of-two hash range: index nslots is the dedicated i64::MIN-key slot
    // (i64::MIN collides with the EMPTY sentinel, so it cannot live in the hash table), and index nslots+1
    // is the M3 (doc 21) dedicated NULL-KEY group slot (the single-level kernel routes a NULL key here via
    // `key_null_off`, forming one group). Both stay empty when unused (int4 keys / no key bitmap). Always
    // allocated so the init/D2H cover them.
    let alloc_slots = nslots + 2;
    let alloc_slots_u64 = alloc_slots as u64;
    let slot_bytes = alloc_slots
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(nslots))?;
    let mask = (nslots - 1) as u64;

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
    let cu_memset_d8_async = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
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
    // Two-level (shared-mem local aggregation -> global merge): far less global-atomic contention at
    // low cardinality. The single-level `gpu_db_group_by_i32_count_sum` stays as the reference kernel.
    let group_fn = primary.cached_function(kernel, &ptx)?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let slot_keys = primary.lease_device_buffer(slot_bytes)?;
    let slot_count = primary.lease_device_buffer(slot_bytes)?;
    let slot_sum = primary.lease_device_buffer(slot_bytes)?;
    // MIN/MAX slots: the single-level kernel fills these per group; the two-level kernel ignores them
    // (they stay at the i64::MAX/MIN identity). Always allocated so the launch signature is uniform.
    let slot_min = primary.lease_device_buffer(slot_bytes)?;
    let slot_max = primary.lease_device_buffer(slot_bytes)?;
    // High 64 bits of the i128 per-group SUM for int8 values (zeroed; the int4 path leaves it 0).
    let slot_sum_hi = primary.lease_device_buffer(slot_bytes)?;
    // A single global flag the kernel `red.global.or`s when a numeric SUM overflows i128 (PG numeric
    // field overflow). Zeroed before launch; only the numeric path ever writes it.
    let overflow_flag = primary.lease_device_buffer(std::mem::size_of::<u64>())?;
    // Numeric MIN/MAX i128 high limbs (pass 1) + a row_slots scratch: row_slots[i] = the slot the
    // i-th row claimed, written by pass 1's numeric branch so the lock-free pass-2 kernel can resolve
    // the i128 LOW limb among the high-limb ties. int4/int8/two-level paths leave these unused.
    let slot_min_hi = primary.lease_device_buffer(slot_bytes)?;
    let slot_max_hi = primary.lease_device_buffer(slot_bytes)?;
    let row_slots_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(count))?;
    let row_slots = primary.lease_device_buffer(row_slots_bytes)?;
    // UUID MIN/MAX slots: one b128 (16 bytes) per slot. The single-level kernel's uuid branch CAS-loops
    // into these; every other path leaves them untouched. Always allocated (uniform launch signature),
    // but only memset + D2H'd when the value is a uuid. Identities: MIN = 0xFF..FF (the largest uuid),
    // MAX = 0x00..00 -- so any real uuid replaces the identity on its first CAS.
    let uuid_slot_bytes = alloc_slots
        .checked_mul(16)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(alloc_slots))?;
    let slot_min_uuid = primary.lease_device_buffer(uuid_slot_bytes)?;
    let slot_max_uuid = primary.lease_device_buffer(uuid_slot_bytes)?;
    // i128 GROUP BY key slots: one b128 (16 bytes) per slot, claimed via atom.cas.b128. Only the
    // numeric/uuid-key path reads/writes these (filled to EMPTY128 = i128::MIN then); leased always for
    // a uniform launch signature. Shares the alloc_slots*16 size with the uuid value slots.
    let slot_keys_i128 = primary.lease_device_buffer(uuid_slot_bytes)?;
    // Text GROUP-BY keys also live in slot_keys_i128 (b128 = (rep_row_idx, text_hash), claimed via
    // atom.cas.b128 with a full-text verify-on-lost-CAS), so the EMPTY128 fill is needed for them too.
    let fill_i128_fn = if key_is_i128 || key_is_text || comp_w > 0 || n_text > 0 {
        Some(primary.cached_function(c"gpu_db_fill_i128", &ptx)?)
    } else {
        None
    };

    // GPU stream-compaction (result-path O(groups), not O(row_count)). After the aggregate kernel a
    // single-block kernel scans the alloc_slots slot table, applies the SAME occupancy predicate the
    // host compact loop used, and (ascending-slot order, two-level prefix sum -> byte-identical order)
    // scatters every occupied group's fields into DENSE arrays + writes out_count. We then D2H ONLY the
    // out_count dense rows, not the full ~2*row_count slot table. Dense arrays are worst-case sized to
    // alloc_slots (every slot occupied) and leased like the slot buffers; the D2H slices to out_count.
    let compact_fn = primary.cached_function(c"gpu_db_group_by_slot_compact", &ptx)?;
    let out_key = primary.lease_device_buffer(slot_bytes)?;
    let out_count_arr = primary.lease_device_buffer(slot_bytes)?;
    let out_sum = primary.lease_device_buffer(slot_bytes)?;
    let out_sum_hi = primary.lease_device_buffer(slot_bytes)?;
    let out_min = primary.lease_device_buffer(slot_bytes)?;
    let out_max = primary.lease_device_buffer(slot_bytes)?;
    let out_min_hi = primary.lease_device_buffer(slot_bytes)?;
    let out_max_hi = primary.lease_device_buffer(slot_bytes)?;
    let out_min_uuid = primary.lease_device_buffer(uuid_slot_bytes)?;
    let out_max_uuid = primary.lease_device_buffer(uuid_slot_bytes)?;
    let out_keyi128 = primary.lease_device_buffer(uuid_slot_bytes)?;
    let out_isnull_bytes = alloc_slots
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(alloc_slots))?;
    let out_isnull = primary.lease_device_buffer(out_isnull_bytes)?;
    let out_count_dev = primary.lease_device_buffer(std::mem::size_of::<u64>())?;

    const BLOCK: u32 = 256;
    // Fills cover alloc_slots (nslots + the dedicated i64::MIN slot) so that slot is initialized too.
    let fill_grid = alloc_slots_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let group_grid = count_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;

    // Numeric MIN/MAX stores an i128, so its identities are the i128 extremes split into limbs:
    // MIN = i128::MAX (hi i64::MAX, lo u64::MAX), MAX = i128::MIN (hi i64::MIN, lo 0). int4/int8 keep
    // the s64 identities in the LOW limb (hi unused -> 0).
    // Text MIN/MAX stores a ROW INDEX (not a value); both slots start at the EMPTY sentinel u64::MAX
    // (-1) so the first row of each group claims via CAS and later rows compare lexicographically.
    let min_lo_id: u64 = if value_is_text || value_is_numeric {
        u64::MAX
    } else {
        i64::MAX as u64
    };
    let max_lo_id: u64 = if value_is_text {
        u64::MAX
    } else if value_is_numeric {
        0
    } else {
        i64::MIN as u64
    };
    let min_hi_id: u64 = if value_is_numeric { i64::MAX as u64 } else { 0 };
    let max_hi_id: u64 = if value_is_numeric { i64::MIN as u64 } else { 0 };

    // One fill-arg set, mutated between the 3 slot_keys / slot_min / slot_max fills (cuLaunchKernel
    // copies the arg values at call time, so re-launching after mutating f0/f2 is safe).
    let mut f0 = slot_keys.ptr;
    let mut f1 = alloc_slots_u64;
    let mut f2 = EMPTY as u64;
    let mut fill_args = [
        (&mut f0 as *mut u64).cast::<c_void>(),
        (&mut f1 as *mut u64).cast::<c_void>(),
        (&mut f2 as *mut u64).cast::<c_void>(),
    ];
    let mut a0 = resident.device_ptr();
    let mut a1 = key_byte_offset;
    let mut a2 = sum_byte_offset;
    let mut a3 = indices_dev.ptr;
    let mut a4 = count_u64;
    let mut a5 = mask;
    let mut a6 = slot_keys.ptr;
    let mut a7 = slot_count.ptr;
    let mut a8 = slot_sum.ptr;
    let mut a9 = slot_min.ptr;
    let mut a10 = slot_max.ptr;
    let mut a11 = u64::from(value_is_int8);
    let mut a12 = slot_sum_hi.ptr;
    let mut a13 = u64::from(key_is_int8);
    let mut a14 = u64::from(value_is_numeric);
    let mut a15 = overflow_flag.ptr;
    let mut a16 = slot_min_hi.ptr;
    let mut a17 = slot_max_hi.ptr;
    let mut a18 = row_slots.ptr;
    let mut a19 = u64::from(value_is_uuid);
    let mut a20 = slot_min_uuid.ptr;
    let mut a21 = slot_max_uuid.ptr;
    let mut a22 = u64::from(key_is_i128);
    let mut a23 = slot_keys_i128.ptr;
    let mut a24 = u64::from(key_is_text);
    let mut a25 = key_offsets_off;
    let mut a26 = key_bytes_off;
    let mut a27 = u64::from(value_is_text);
    let mut a28 = value_offsets_off;
    let mut a29 = value_bytes_off;
    let mut a30 = key_base_override;
    let mut a31 = value_base_override;
    let mut a32 = comp_w;
    let mut a33 = n_text;
    let mut a34 = text_desc_ptr;
    // M3 3VL: u64::MAX sentinel when there is no value validity bitmap; otherwise the bounds-checked
    // offset. The kernel reads bit `idx` for each idx in `indices`, so the bitmap must cover the max
    // index (< the table's resident row count). Appended LAST so the twolevel kernel (fewer params)
    // ignores it.
    let mut a35 = match value_null_off {
        None => u64::MAX,
        Some(off) => {
            let max_idx = indices.iter().copied().max().unwrap_or(0) as u64;
            let words_bytes = (max_idx / 32 + 1)
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            let end = off
                .checked_add(words_bytes)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > resident.metadata().allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
            }
            off
        }
    };
    // M3 (doc 21): the KEY column's NULL validity bitmap arg, same sentinel + bounds-check as the value
    // bitmap (the kernel reads bit `idx` for each idx in `indices`).
    let mut a36 = match key_null_off {
        None => u64::MAX,
        Some(off) => {
            let max_idx = indices.iter().copied().max().unwrap_or(0) as u64;
            let words_bytes = (max_idx / 32 + 1)
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            let end = off
                .checked_add(words_bytes)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if end > resident.metadata().allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
            }
            off
        }
    };
    // Query-aware aggregate-selection mask (grouped_agg_mask). Passed as the LAST kernel arg (a37) to
    // BOTH the single-level and two-level kernels; widened to u64 for the uniform u64 launch-arg array.
    let mut a37 = u64::from(agg_mask);
    // gpu_db_fill_i128(slot_keys_i128, alloc_slots, lo=0, hi=i64::MIN) -> EMPTY128 = i128::MIN.
    let mut g0 = slot_keys_i128.ptr;
    let mut g1 = alloc_slots_u64;
    let mut g2 = 0u64;
    let mut g3 = i64::MIN as u64;
    let mut fill128_args = [
        (&mut g0 as *mut u64).cast::<c_void>(),
        (&mut g1 as *mut u64).cast::<c_void>(),
        (&mut g2 as *mut u64).cast::<c_void>(),
        (&mut g3 as *mut u64).cast::<c_void>(),
    ];
    let mut group_args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u64).cast::<c_void>(),
        (&mut a7 as *mut u64).cast::<c_void>(),
        (&mut a8 as *mut u64).cast::<c_void>(),
        (&mut a9 as *mut u64).cast::<c_void>(),
        (&mut a10 as *mut u64).cast::<c_void>(),
        (&mut a11 as *mut u64).cast::<c_void>(),
        (&mut a12 as *mut u64).cast::<c_void>(),
        (&mut a13 as *mut u64).cast::<c_void>(),
        (&mut a14 as *mut u64).cast::<c_void>(),
        (&mut a15 as *mut u64).cast::<c_void>(),
        (&mut a16 as *mut u64).cast::<c_void>(),
        (&mut a17 as *mut u64).cast::<c_void>(),
        (&mut a18 as *mut u64).cast::<c_void>(),
        (&mut a19 as *mut u64).cast::<c_void>(),
        (&mut a20 as *mut u64).cast::<c_void>(),
        (&mut a21 as *mut u64).cast::<c_void>(),
        (&mut a22 as *mut u64).cast::<c_void>(),
        (&mut a23 as *mut u64).cast::<c_void>(),
        (&mut a24 as *mut u64).cast::<c_void>(),
        (&mut a25 as *mut u64).cast::<c_void>(),
        (&mut a26 as *mut u64).cast::<c_void>(),
        (&mut a27 as *mut u64).cast::<c_void>(),
        (&mut a28 as *mut u64).cast::<c_void>(),
        (&mut a29 as *mut u64).cast::<c_void>(),
        (&mut a30 as *mut u64).cast::<c_void>(),
        (&mut a31 as *mut u64).cast::<c_void>(),
        (&mut a32 as *mut u64).cast::<c_void>(),
        (&mut a33 as *mut u64).cast::<c_void>(),
        (&mut a34 as *mut u64).cast::<c_void>(),
        (&mut a35 as *mut u64).cast::<c_void>(),
        (&mut a36 as *mut u64).cast::<c_void>(),
        (&mut a37 as *mut u64).cast::<c_void>(),
    ];
    // Pass 2 (numeric MIN/MAX only): a second, LOCK-FREE kernel that resolves the i128 low limb after
    // pass 1 (the main kernel) finalized the high limbs. Cached + its args built only for numeric.
    // PHANTOM-GROUP FIX: pass 2 resolves the numeric MIN/MAX LOW limbs by re-visiting each row's
    // claimed slot via row_slots -- but pass 1 writes row_slots ONLY inside its min/max block, which
    // the aggregate mask can skip entirely (mask & 12 == 0, e.g. a grouped SUM(numeric)). Launching
    // pass 2 then scatters slot_min/slot_max through STALE POOLED row_slots values -- u32 garbage slot
    // indices = unbounded OOB writes that corrupt adjacent pool allocations (observed: the compactor's
    // buffers -> phantom groups, order-dependent on pool reuse). Gate pass 2 exactly as pass 1 gates
    // the block that feeds it: MIN or MAX actually requested.
    let pass2_fn = if value_is_numeric && (agg_mask & 12) != 0 {
        Some(primary.cached_function(c"gpu_db_group_by_numeric_minmax_lo", &ptx)?)
    } else {
        None
    };
    let mut q0 = resident.device_ptr();
    let mut q1 = sum_byte_offset;
    let mut q2 = indices_dev.ptr;
    let mut q3 = count_u64;
    let mut q4 = row_slots.ptr;
    let mut q5 = slot_min.ptr;
    let mut q6 = slot_max.ptr;
    let mut q7 = slot_min_hi.ptr;
    let mut q8 = slot_max_hi.ptr;
    // M3 (doc 21) 3VL: pass 2 reads the SAME value validity bitmap as pass 1 (a35, already bounds-checked)
    // and skips NULL rows -- so it never folds a stale pooled row_slots slot for a NULL value.
    let mut q9 = a35;
    let mut pass2_args = [
        (&mut q0 as *mut u64).cast::<c_void>(),
        (&mut q1 as *mut u64).cast::<c_void>(),
        (&mut q2 as *mut u64).cast::<c_void>(),
        (&mut q3 as *mut u64).cast::<c_void>(),
        (&mut q4 as *mut u64).cast::<c_void>(),
        (&mut q5 as *mut u64).cast::<c_void>(),
        (&mut q6 as *mut u64).cast::<c_void>(),
        (&mut q7 as *mut u64).cast::<c_void>(),
        (&mut q8 as *mut u64).cast::<c_void>(),
        (&mut q9 as *mut u64).cast::<c_void>(),
    ];
    // Compaction-kernel args. `use_i128` mirrors the host hash-range occupancy condition (i128/text/
    // composite keys live in slot_keys_i128); `key_is_i128_strict` mirrors the dedicated i64::MIN slot's
    // `key_i128: if key_is_i128 { i128::MIN }` (ONLY the plain-i128-key path, not text/composite).
    let use_i128_occ: u64 = u64::from(key_is_i128 || key_is_text || comp_w > 0 || n_text > 0);
    let key_is_i128_strict: u64 = u64::from(key_is_i128);
    let mut c0 = slot_keys.ptr;
    let mut c1 = slot_count.ptr;
    let mut c2 = slot_sum.ptr;
    let mut c3 = slot_sum_hi.ptr;
    let mut c4 = slot_min.ptr;
    let mut c5 = slot_max.ptr;
    let mut c6 = slot_min_hi.ptr;
    let mut c7 = slot_max_hi.ptr;
    let mut c8 = slot_min_uuid.ptr;
    let mut c9 = slot_max_uuid.ptr;
    let mut c10 = slot_keys_i128.ptr;
    let mut c11 = nslots as u64;
    let mut c12 = alloc_slots_u64;
    let mut c13 = use_i128_occ;
    let mut c14 = key_is_i128_strict;
    let mut c15 = out_key.ptr;
    let mut c16 = out_count_arr.ptr;
    let mut c17 = out_sum.ptr;
    let mut c18 = out_sum_hi.ptr;
    let mut c19 = out_min.ptr;
    let mut c20 = out_max.ptr;
    let mut c21 = out_min_hi.ptr;
    let mut c22 = out_max_hi.ptr;
    let mut c23 = out_min_uuid.ptr;
    let mut c24 = out_max_uuid.ptr;
    let mut c25 = out_keyi128.ptr;
    let mut c26 = out_isnull.ptr;
    let mut c27 = out_count_dev.ptr;
    let mut compact_args = [
        (&mut c0 as *mut u64).cast::<c_void>(),
        (&mut c1 as *mut u64).cast::<c_void>(),
        (&mut c2 as *mut u64).cast::<c_void>(),
        (&mut c3 as *mut u64).cast::<c_void>(),
        (&mut c4 as *mut u64).cast::<c_void>(),
        (&mut c5 as *mut u64).cast::<c_void>(),
        (&mut c6 as *mut u64).cast::<c_void>(),
        (&mut c7 as *mut u64).cast::<c_void>(),
        (&mut c8 as *mut u64).cast::<c_void>(),
        (&mut c9 as *mut u64).cast::<c_void>(),
        (&mut c10 as *mut u64).cast::<c_void>(),
        (&mut c11 as *mut u64).cast::<c_void>(),
        (&mut c12 as *mut u64).cast::<c_void>(),
        (&mut c13 as *mut u64).cast::<c_void>(),
        (&mut c14 as *mut u64).cast::<c_void>(),
        (&mut c15 as *mut u64).cast::<c_void>(),
        (&mut c16 as *mut u64).cast::<c_void>(),
        (&mut c17 as *mut u64).cast::<c_void>(),
        (&mut c18 as *mut u64).cast::<c_void>(),
        (&mut c19 as *mut u64).cast::<c_void>(),
        (&mut c20 as *mut u64).cast::<c_void>(),
        (&mut c21 as *mut u64).cast::<c_void>(),
        (&mut c22 as *mut u64).cast::<c_void>(),
        (&mut c23 as *mut u64).cast::<c_void>(),
        (&mut c24 as *mut u64).cast::<c_void>(),
        (&mut c25 as *mut u64).cast::<c_void>(),
        (&mut c26 as *mut u64).cast::<c_void>(),
        (&mut c27 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                idx_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe { cu_memset_d8_async(slot_count.ptr, 0, slot_bytes, stream) };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe { cu_memset_d8_async(slot_sum.ptr, 0, slot_bytes, stream) };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe { cu_memset_d8_async(slot_sum_hi.ptr, 0, slot_bytes, stream) };
        if rc != 0 {
            return rc;
        }
        let rc =
            unsafe { cu_memset_d8_async(overflow_flag.ptr, 0, std::mem::size_of::<u64>(), stream) };
        if rc != 0 {
            return rc;
        }
        // UUID MIN/MAX identities: MIN slot = 0xFF..FF (largest uuid), MAX slot = 0x00..00 (smallest).
        // Only the uuid path reads/writes these, so only init them then (the buffers are always leased
        // for a uniform launch signature but stay untouched otherwise).
        if value_is_uuid {
            let rc =
                unsafe { cu_memset_d8_async(slot_min_uuid.ptr, 0xFF, uuid_slot_bytes, stream) };
            if rc != 0 {
                return rc;
            }
            let rc =
                unsafe { cu_memset_d8_async(slot_max_uuid.ptr, 0x00, uuid_slot_bytes, stream) };
            if rc != 0 {
                return rc;
            }
        }
        // i128 GROUP BY keys: fill slot_keys_i128 = EMPTY128 (i128::MIN) before the claim runs.
        if let Some(f) = fill_i128_fn {
            let rc = unsafe {
                cu_launch_kernel(
                    f,
                    fill_grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    fill128_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        // (row_slots needs no init: pass 1 writes every NON-NULL row's slot before pass 2 reads it; a NULL
        // row's slot is left stale but pass 2 skips NULL rows via the value validity bitmap, never reading it.)
        // fill slot_keys = EMPTY
        let rc = unsafe {
            cu_launch_kernel(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fill_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // fill slot_min (low limb) = the MIN identity (s64 i64::MAX, or numeric i128::MAX low = u64::MAX)
        f0 = slot_min.ptr;
        f2 = min_lo_id;
        let rc = unsafe {
            cu_launch_kernel(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fill_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // fill slot_max (low limb) = the MAX identity (s64 i64::MIN, or numeric i128::MIN low = 0)
        f0 = slot_max.ptr;
        f2 = max_lo_id;
        let rc = unsafe {
            cu_launch_kernel(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fill_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // fill slot_min_hi / slot_max_hi (numeric MIN/MAX high limbs; 0 for int4/int8).
        f0 = slot_min_hi.ptr;
        f2 = min_hi_id;
        let rc = unsafe {
            cu_launch_kernel(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fill_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        f0 = slot_max_hi.ptr;
        f2 = max_hi_id;
        let rc = unsafe {
            cu_launch_kernel(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                fill_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            cu_launch_kernel(
                group_fn,
                group_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                group_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return rc;
        }
        // Pass 2 (numeric MIN/MAX): resolve the i128 low limb lock-free, on the same stream.
        if let Some(f) = pass2_fn {
            let rc = unsafe {
                cu_launch_kernel(
                    f,
                    group_grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    pass2_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        // GPU stream-compaction: single block (the two-level prefix-sum scatter is block-local; one
        // block grid-strides the whole slot table with a running offset, like the HAVING compactor). On
        // the same stream, after every aggregate pass has finalized the slots.
        unsafe {
            cu_launch_kernel(
                compact_fn,
                1,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                compact_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    // RESULT PATH (O(groups), not O(row_count)): the compaction kernel above already scanned the slot
    // table and wrote the occupied groups DENSELY in ascending-slot order + the group count. D2H only
    // `out_count` (one u64) and then the dense arrays SLICED to out_count entries -- not the full
    // ~2*row_count slot table. The dense arrays carry EXACTLY the fields the old host `for i in 0..nslots`
    // loop produced (key/count/sum/sum_hi/min/max/min_hi/max_hi + the raw b128 uuid + b128 key_i128 +
    // the key_is_null flag), in the SAME order, so the built groups are byte-identical.
    let mut out_count: u64 = 0;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut out_count as *mut u64).cast::<c_void>(),
            out_count_dev.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    // out_count <= alloc_slots (every slot occupied is the worst case) -- a larger value means the kernel
    // miscounted (corrupt) and would over-read the dense buffers; treat as a hard error.
    let groups_len = usize::try_from(out_count)
        .ok()
        .filter(|&n| n <= alloc_slots)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut groups = Vec::with_capacity(groups_len);
    if groups_len > 0 {
        let dense_i64_bytes = groups_len
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(groups_len))?;
        let dense_b128_bytes = groups_len
            .checked_mul(16)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(groups_len))?;
        let dense_u32_bytes = groups_len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(groups_len))?;
        let mut keys = vec![0i64; groups_len];
        let mut counts = vec![0u64; groups_len];
        let mut sums = vec![0i64; groups_len];
        let mut sum_his = vec![0i64; groups_len];
        let mut mins = vec![0i64; groups_len];
        let mut maxs = vec![0i64; groups_len];
        let mut min_his = vec![0i64; groups_len];
        let mut max_his = vec![0i64; groups_len];
        let mut min_uuid_bytes = vec![0u8; dense_b128_bytes];
        let mut max_uuid_bytes = vec![0u8; dense_b128_bytes];
        let mut key_i128s = vec![0i128; groups_len];
        let mut is_nulls = vec![0u32; groups_len];
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                keys.as_mut_ptr().cast::<c_void>(),
                out_key.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                counts.as_mut_ptr().cast::<c_void>(),
                out_count_arr.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                sums.as_mut_ptr().cast::<c_void>(),
                out_sum.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                sum_his.as_mut_ptr().cast::<c_void>(),
                out_sum_hi.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                mins.as_mut_ptr().cast::<c_void>(),
                out_min.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                maxs.as_mut_ptr().cast::<c_void>(),
                out_max.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                min_his.as_mut_ptr().cast::<c_void>(),
                out_min_hi.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                max_his.as_mut_ptr().cast::<c_void>(),
                out_max_hi.ptr,
                dense_i64_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                min_uuid_bytes.as_mut_ptr().cast::<c_void>(),
                out_min_uuid.ptr,
                dense_b128_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                max_uuid_bytes.as_mut_ptr().cast::<c_void>(),
                out_max_uuid.ptr,
                dense_b128_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                key_i128s.as_mut_ptr().cast::<c_void>(),
                out_keyi128.ptr,
                dense_b128_bytes,
            )
        })?;
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                is_nulls.as_mut_ptr().cast::<c_void>(),
                out_isnull.ptr,
                dense_u32_bytes,
            )
        })?;
        // Rebuild a canonical-order uuid from a dense row's 16 device bytes -- the SAME byte-reversal the
        // old host loop applied (the kernel stored the b128 as `(uhi << 64) | ulo` with uhi = uuid bytes
        // 0-7 big-endian and ulo = bytes 8-15 big-endian; in LE device memory that is the uuid fully
        // byte-reversed). Only run for a uuid value aggregate; [0; 16] otherwise (matches the old path,
        // which left the b128 slots at their identity / unread for non-uuid and emitted [0; 16]).
        let uuid_at = |buf: &[u8], row: usize| -> [u8; 16] {
            if !value_is_uuid {
                return [0u8; 16];
            }
            let b = &buf[row * 16..row * 16 + 16];
            let ulo = u64::from_le_bytes(b[0..8].try_into().unwrap());
            let uhi = u64::from_le_bytes(b[8..16].try_into().unwrap());
            let mut out = [0u8; 16];
            out[0..8].copy_from_slice(&uhi.to_be_bytes());
            out[8..16].copy_from_slice(&ulo.to_be_bytes());
            out
        };
        for i in 0..groups_len {
            groups.push(GroupByI32Row {
                key: keys[i],
                count: counts[i],
                sum: sums[i],
                sum_hi: sum_his[i],
                min: mins[i],
                max: maxs[i],
                min_hi: min_his[i],
                max_hi: max_his[i],
                min_uuid: uuid_at(&min_uuid_bytes, i),
                max_uuid: uuid_at(&max_uuid_bytes, i),
                key_i128: key_i128s[i],
                key_is_null: is_nulls[i] != 0,
            });
        }
    }
    // A numeric SUM that overflowed i128 in any group/thread set this flag on-device -> PG numeric
    // field overflow (never silently wrapped). Checked after the kernel like the scalar numeric SUM.
    let mut overflow = 0u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut overflow as *mut u64).cast::<c_void>(),
            overflow_flag.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    if overflow != 0 {
        return Err(CudaRuntimeProbeError::NumericFieldOverflow);
    }
    Ok(groups)
}

/// Benchmark-only: time JUST the GROUP BY KERNEL (excluding the per-call alloc / H2D / D2H / host
/// compaction that dominate the end-to-end latency) via CUDA events, returning the MIN over `runs`
/// kernel launches plus the result rows (for a correctness check). Sets up once; per run resets the
/// global table (fill + memset) UNTIMED, then events bracket only the group kernel. Null stream.
#[allow(unused_assignments)] // f0/f2 are re-read via the raw fill-arg pointers
pub(super) fn launch_cuda_group_by_kernel_timed(
    resident: &CudaResidentDeviceMemory,
    key_byte_offset: u64,
    sum_byte_offset: u64,
    indices: &[u32],
    kernel: &'static CStr,
    runs: u32,
    // Query-aware aggregate-selection mask (`grouped_agg_mask`), passed as the LAST kernel arg.
    agg_mask: u32,
) -> Result<(Vec<GroupByI32Row>, f32), CudaRuntimeProbeError> {
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
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuStreamSync = unsafe extern "C" fn(*mut c_void) -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    const EMPTY: i64 = i64::MIN;

    if indices.is_empty() {
        return Ok((Vec::new(), 0.0));
    }
    let count = indices.len();
    let idx_bytes = std::mem::size_of_val(indices);
    let count_u64 = count as u64;
    let nslots = (count * 2).next_power_of_two().max(16);
    let slot_bytes = nslots * std::mem::size_of::<i64>();
    let mask = (nslots - 1) as u64;
    let nslots_u64 = nslots as u64;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch = unsafe {
        *resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset = unsafe {
        *resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_htod = unsafe {
        *resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_dtoh = unsafe {
        *resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_stream_sync = unsafe {
        *resident
            .lib()
            .get::<CuStreamSync>(b"cuStreamSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let fill_fn = primary.cached_function(c"gpu_db_fill_i64", &ptx)?;
    let group_fn = primary.cached_function(kernel, &ptx)?;
    let indices_dev = primary.lease_device_buffer(idx_bytes)?;
    let slot_keys = primary.lease_device_buffer(slot_bytes)?;
    let slot_count = primary.lease_device_buffer(slot_bytes)?;
    let slot_sum = primary.lease_device_buffer(slot_bytes)?;
    // min/max slots: the single-level kernel writes them (it dereferences the params); the two-level
    // ignores them. Real buffers required so the single-level kernel does not write to a stray pointer.
    let slot_min = primary.lease_device_buffer(slot_bytes)?;
    let slot_max = primary.lease_device_buffer(slot_bytes)?;
    // slot_sum_hi + overflow_flag + slot_min_hi/max_hi/row_slots: required so the 19-param kernel has
    // valid pointers; unused by the int4 bench (it never takes the int8/numeric branches that write them).
    let slot_sum_hi = primary.lease_device_buffer(slot_bytes)?;
    let overflow_flag = primary.lease_device_buffer(std::mem::size_of::<u64>())?;
    let slot_min_hi = primary.lease_device_buffer(slot_bytes)?;
    let slot_max_hi = primary.lease_device_buffer(slot_bytes)?;
    let row_slots = primary.lease_device_buffer(count.max(1) * std::mem::size_of::<u32>())?;
    // UUID MIN/MAX slots (b128, 16 bytes/slot): valid pointers so the 27-param kernel has no stray arg.
    // The int4 bench never takes the uuid branch (value_is_uuid = 0), so they are never dereferenced.
    let slot_min_uuid = primary.lease_device_buffer(nslots * 16)?;
    let slot_max_uuid = primary.lease_device_buffer(nslots * 16)?;
    // i128 / text GROUP BY key slots (b128): a valid pointer so the 27-param kernel has no stray arg.
    // The int4 bench never takes the i128/text-key branch (key_is_i128 = key_is_text = 0), so it is
    // never dereferenced.
    let slot_keys_i128 = primary.lease_device_buffer(nslots * 16)?;
    check_cuda(unsafe {
        cu_htod(
            indices_dev.ptr,
            indices.as_ptr().cast::<c_void>(),
            idx_bytes,
        )
    })?;

    let null = std::ptr::null_mut::<c_void>();
    let mut start = std::ptr::null_mut::<c_void>();
    let mut stop = std::ptr::null_mut::<c_void>();
    check_cuda(unsafe { (primary.cu_event_create)(&mut start, 0) })?;
    check_cuda(unsafe { (primary.cu_event_create)(&mut stop, 0) })?;

    const BLOCK: u32 = 256;
    let fill_grid = nslots_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let group_grid = count_u64.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut f0 = slot_keys.ptr;
    let mut f1 = nslots_u64;
    let mut f2 = EMPTY as u64;
    let mut fill_args = [
        (&mut f0 as *mut u64).cast::<c_void>(),
        (&mut f1 as *mut u64).cast::<c_void>(),
        (&mut f2 as *mut u64).cast::<c_void>(),
    ];
    let mut a = [
        resident.device_ptr(),
        key_byte_offset,
        sum_byte_offset,
        indices_dev.ptr,
        count_u64,
        mask,
        slot_keys.ptr,
        slot_count.ptr,
        slot_sum.ptr,
        slot_min.ptr,
        slot_max.ptr,
        0, // value_is_int8 = false: the timed bench always aggregates an int4 value column
        slot_sum_hi.ptr,
        0, // key_is_int8 = false: the timed bench always groups by an int4 key
        0, // value_is_numeric = false
        overflow_flag.ptr,
        slot_min_hi.ptr,
        slot_max_hi.ptr,
        row_slots.ptr,
        0, // value_is_uuid = false: the timed bench never aggregates a uuid value
        slot_min_uuid.ptr,
        slot_max_uuid.ptr,
        0, // key_is_i128 = false: the timed bench always groups by an int4 key
        slot_keys_i128.ptr,
        0,                   // key_is_text = false: the timed bench always groups by an int4 key
        0,                   // key_offsets_off (unused)
        0,                   // key_bytes_off (unused)
        0,        // value_is_text = false: the timed bench always aggregates an int4 value
        0,        // value_offsets_off (unused)
        0,        // value_bytes_off (unused)
        0, // key_base_override = 0: the timed bench uses column keys (no derived-buffer override)
        0, // value_base_override = 0: the timed bench uses column values
        0, // comp_w = 0: the timed bench is not a wide-key composite
        0, // n_text = 0: the timed bench has no text members
        0, // text_desc_ptr = 0
        u64::MAX, // M3 value_null_off = sentinel (the timed bench is non-nullable -> no value skip)
        u64::MAX, // M3 key_null_off = sentinel (the timed bench is non-nullable -> no NULL-key group)
        u64::from(agg_mask), // query-aware aggregate-selection mask (a37, LAST kernel arg)
    ];
    let mut group_args: Vec<*mut c_void> = a
        .iter_mut()
        .map(|x| (x as *mut u64).cast::<c_void>())
        .collect();

    let mut best = f32::MAX;
    for _ in 0..runs {
        check_cuda(unsafe { cu_memset(slot_count.ptr, 0, slot_bytes) })?;
        check_cuda(unsafe { cu_memset(slot_sum.ptr, 0, slot_bytes) })?;
        // fill keys=EMPTY, min=i64::MAX, max=i64::MIN (reset f0/f2 each iteration).
        f0 = slot_keys.ptr;
        f2 = EMPTY as u64;
        check_cuda(unsafe {
            cu_launch(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                null,
                fill_args.as_mut_ptr(),
                null.cast(),
            )
        })?;
        f0 = slot_min.ptr;
        f2 = i64::MAX as u64;
        check_cuda(unsafe {
            cu_launch(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                null,
                fill_args.as_mut_ptr(),
                null.cast(),
            )
        })?;
        f0 = slot_max.ptr;
        f2 = i64::MIN as u64;
        check_cuda(unsafe {
            cu_launch(
                fill_fn,
                fill_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                null,
                fill_args.as_mut_ptr(),
                null.cast(),
            )
        })?;
        check_cuda(unsafe { (primary.cu_event_record)(start, null) })?;
        check_cuda(unsafe {
            cu_launch(
                group_fn,
                group_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                null,
                group_args.as_mut_ptr(),
                null.cast(),
            )
        })?;
        check_cuda(unsafe { (primary.cu_event_record)(stop, null) })?;
        check_cuda(unsafe { cu_stream_sync(null) })?;
        let mut ms = 0f32;
        check_cuda(unsafe { (primary.cu_event_elapsed_time)(&mut ms, start, stop) })?;
        best = best.min(ms);
    }

    let mut keys = vec![0i64; nslots];
    let mut counts = vec![0u64; nslots];
    let mut sums = vec![0i64; nslots];
    check_cuda(unsafe {
        cu_dtoh(
            keys.as_mut_ptr().cast::<c_void>(),
            slot_keys.ptr,
            slot_bytes,
        )
    })?;
    check_cuda(unsafe {
        cu_dtoh(
            counts.as_mut_ptr().cast::<c_void>(),
            slot_count.ptr,
            slot_bytes,
        )
    })?;
    check_cuda(unsafe { cu_dtoh(sums.as_mut_ptr().cast::<c_void>(), slot_sum.ptr, slot_bytes) })?;
    unsafe {
        (primary.cu_event_destroy)(start);
        (primary.cu_event_destroy)(stop);
    }
    let mut groups = Vec::new();
    for i in 0..nslots {
        if keys[i] != EMPTY {
            // min/max are placeholders here -- the timed bench only validates COUNT/SUM (the two
            // kernels intentionally differ on min/max: single-level computes them, two-level doesn't).
            groups.push(GroupByI32Row {
                key: keys[i],
                count: counts[i],
                sum: sums[i],
                sum_hi: 0,
                min: 0,
                max: 0,
                min_hi: 0,
                max_hi: 0,
                min_uuid: [0u8; 16],
                max_uuid: [0u8; 16],
                key_i128: 0,
                key_is_null: false,
            });
        }
    }
    Ok((groups, best))
}
