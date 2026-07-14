//! GPU ordering for already-materialized grouped result rows.
//!
//! The relational ordering decision remains on-device. Host construction of key matrices/payload rows,
//! key H2D, and permutation D2H are explicit RETIRE-003 result-path debt, not target architecture.

use crate::ExecuteError;
use gpu_db_sql::{SqlType, SqlValue};
use gpu_db_types::EngineError;

/// The ON-DEVICE sort PERMUTATION for `rows` by `order` ((result-column index, descending)) -- the index
/// vector `perm` such that `rows[perm[0]], rows[perm[1]], ...` is sorted. Grouped final ORDER BY
/// reorders/windows result rows by this index vector, and GROUP BY multi-pass alignment (S2.3) reorders
/// per-pass group arrays on-device (a TOTAL key order aligns every pass by index -- no host sort).
pub(crate) fn gpu_sort_permutation(
    rows: &[Vec<SqlValue>],
    order: &[(usize, bool)],
    // Parallel to `order`: the explicit NULLS FIRST/LAST override per key (None = PG default). Empty =
    // every key default. Honored ON-DEVICE for EVERY key type via the comparator's per-key validity bitmap
    // (built from the result rows into the payload) + the nulls_first bitmask -- no host NULL sentinel.
    nulls_first: &[Option<bool>],
    col_types: &[SqlType],
    device_memory: &gpu_db_execution::CudaResidentDeviceMemory,
) -> Result<Vec<u32>, ExecuteError> {
    if order.is_empty() || rows.len() <= 1 {
        return Ok((0..rows.len() as u32).collect());
    }
    let n = rows.len();
    let map_sort_err = |e: gpu_db_execution::CudaRuntimeProbeError| {
        ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
    };
    // (result-column index, kind 0=int / 1=text / 2=numeric / 3=uuid) per ORDER BY key.
    let mut classified: Vec<(usize, u8)> = Vec::with_capacity(order.len());
    for &(idx, _desc) in order {
        let kind = match col_types[idx] {
            // Bool sorts as its 0/1 ordinal in the int matrix (false < true), like the host key_cmp did.
            SqlType::Int4
            | SqlType::Int8
            | SqlType::Int2
            | SqlType::Date
            | SqlType::Timestamp
            | SqlType::Bool => 0u8,
            SqlType::Text => 1,
            SqlType::Numeric { .. } => 2,
            SqlType::Uuid => 3,
            // Exhaustive over SqlType: every result-column type is now sortable on-device. A new SqlType
            // must be classified here (compile error otherwise) rather than silently falling through.
        };
        classified.push((idx, kind));
    }
    // M3 (doc 21): per-key effective NULLS FIRST bit (explicit override, else the key's desc = PG default).
    // The NULL placement is decided ON-DEVICE by the sort comparator reading each key's validity bitmap --
    // INT keys included (their bitmap rides the payload, their value the int matrix), so there is NO host
    // NULL sentinel and int8/timestamp need no special-casing.
    let mut nulls_first_mask: u64 = 0;
    for (ki, &(_, desc)) in order.iter().enumerate() {
        if nulls_first.get(ki).copied().flatten().unwrap_or(desc) {
            nulls_first_mask |= 1u64 << ki;
        }
    }
    let num_int = classified.iter().filter(|&&(_, k)| k == 0).count();
    // INT key matrix, row-major by row position. A NULL writes a 0 PLACEHOLDER; the comparator detects the
    // NULL from the validity bitmap (built below) BEFORE the value compare, so the placeholder is never
    // compared -- the NULL placement decision is on-device, not a host sentinel value.
    let mut int_keys: Vec<i64> = Vec::with_capacity(n * num_int);
    for row in rows {
        for &(idx, kind) in &classified {
            if kind == 0 {
                int_keys.push(match row[idx] {
                    SqlValue::Int4(v) | SqlValue::Date(v) => i64::from(v),
                    SqlValue::Int2(v) => i64::from(v),
                    SqlValue::Int8(v) | SqlValue::Timestamp(v) => v,
                    SqlValue::Bool(b) => i64::from(b),
                    SqlValue::Null => 0,
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "ORDER BY int key encountered a non-int value".to_string(),
                        )));
                    }
                });
            }
        }
    }
    let mut desc_mask = 0u64;
    for (ki, &(_, desc)) in order.iter().enumerate() {
        if desc {
            desc_mask |= 1u64 << ki;
        }
    }
    // A NULL in ANY key column routes to the hetero comparator (it reads each key's validity bitmap from the
    // payload ON-DEVICE). An all-int, NULL-FREE sort keeps the fast int-matrix path (no bitmaps needed).
    let has_null_key = classified
        .iter()
        .any(|&(idx, _)| rows.iter().any(|r| matches!(r[idx], SqlValue::Null)));
    let non_int: Vec<(usize, u8)> = classified
        .iter()
        .copied()
        .filter(|&(_, k)| k != 0)
        .collect();
    let perm: Vec<u32> = if non_int.is_empty() && !has_null_key {
        device_memory
            .bitonic_sort_multikey(&int_keys, n, num_int, desc_mask)
            .map_err(map_sort_err)?
    } else {
        // A resident-like payload over EVERY key column (M3 -- doc 21): build_relational_device_payload emits
        // a validity bitmap for any column holding a NULL, so an INT key's NULL placement is read ON-DEVICE
        // from its payload bitmap (the int VALUE still rides `int_keys`; the int value section in the payload
        // is unused, but its presence keeps every later section's offset correct). Non-int keys read both
        // value + validity from the payload. No host NULL sentinel on any key.
        let names: Vec<String> = (0..classified.len()).map(|i| format!("__gsk{i}")).collect();
        let types: Vec<SqlType> = classified.iter().map(|&(idx, _)| col_types[idx]).collect();
        let payload_rows: Vec<Vec<SqlValue>> = rows
            .iter()
            .map(|r| classified.iter().map(|&(idx, _)| r[idx].clone()).collect())
            .collect();
        let (payload, text_layouts, _bool, _int4, b128_layouts, null_layouts) =
            crate::engine_residency::build_relational_device_payload(
                &names,
                &types,
                &payload_rows,
            )?;
        // null_offs[ki] = key ki's validity-bitmap byte offset in the payload (u64::MAX = the column has no
        // NULL ⇒ pure value compare). Found by the key's own name, so an int key gets its bitmap too.
        let null_off_of = |ki: usize| -> u64 {
            let name = format!("__gsk{ki}");
            null_layouts
                .iter()
                .find(|l| l.name == name)
                .map_or(u64::MAX, |l| l.bitmap_byte_offset)
        };
        // Walk ORDER BY order: int -> the next int-matrix slot; text/numeric/uuid -> the next section in its
        // type group (the helper lays each group out in passed-column order). The validity bitmap (all key
        // types) is read on-device via null_offs.
        let mut int_slot = 0u32;
        let mut text_idx = 0usize;
        let mut b128_idx = 0usize;
        let mut text_cols: Vec<(u64, u64)> = Vec::new();
        let mut b128_cols: Vec<u64> = Vec::new();
        let mut key_plan: Vec<u32> = Vec::with_capacity(order.len());
        let mut null_offs: Vec<u64> = Vec::with_capacity(order.len());
        for (ki, &(_, kind)) in classified.iter().enumerate() {
            match kind {
                0 => {
                    key_plan.push(int_slot);
                    int_slot += 1;
                }
                1 => {
                    let tl = &text_layouts[text_idx];
                    key_plan.push(0x4000_0000_u32 | text_cols.len() as u32);
                    text_cols.push((tl.offsets_byte_offset, tl.bytes_byte_offset));
                    text_idx += 1;
                }
                k => {
                    let off = b128_layouts[b128_idx].1;
                    let tag = if k == 2 {
                        0x8000_0000_u32
                    } else {
                        0xC000_0000_u32
                    };
                    key_plan.push(tag | b128_cols.len() as u32);
                    b128_cols.push(off);
                    b128_idx += 1;
                }
            }
            null_offs.push(null_off_of(ki));
        }
        let indices: Vec<u64> = (0..n as u64).collect();
        // NULL placement is honored ON-DEVICE for EVERY key: the comparator reads the key's validity bitmap
        // (null_offs) and places NULL per the per-key nulls_first bit (nulls_first_mask). An int key reads
        // its value from `int_keys` but its NULL-ness from the payload bitmap -- no host sentinel.
        device_memory
            .bitonic_sort_hetero_on_payload(
                &payload,
                &indices,
                &int_keys,
                num_int,
                &text_cols,
                &b128_cols,
                &key_plan,
                desc_mask,
                &null_offs,
                nulls_first_mask,
            )
            .map_err(map_sort_err)?
    };
    Ok(perm)
}
