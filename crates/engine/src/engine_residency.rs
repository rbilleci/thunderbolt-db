//! GPU residency management + resident-route planning (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for populating/admitting
//! resident snapshots (incl. on-GPU), the benchmark chunk/shard installs,
//! resident device-memory + bytes accounting, retained-read snapshot handles,
//! warmup/maintenance policy execution, and the resident-route planners
//! (plan_relational_resident_route + sharded variant) + residency status.

use super::*;

/// Build the GPU device payload for typed columns + their row values -- the columnar
/// `[8-byte row_count header][int4/date/int2 i32][int8/timestamp i64][numeric/uuid 16B][bool bitmap]
/// [text 8-aligned offsets + bytes]` layout (type-grouped, catalog order within each type; varlen text
/// offsets 8-aligned per the CUDA-716 lesson). Returns the payload (row_count header filled, NO MVCC
/// tail) + the text/bool column layouts + per-int4-column min/max -- the SAME bytes/offsets the
/// resident-table builder produces, so a non-table caller (e.g. the grouped-sort) can build a
/// resident-like buffer without re-implementing the byte mappings. `column_names` / `column_types` /
/// each row in `rows` are parallel by column index.
#[allow(clippy::type_complexity)]
pub(crate) fn build_relational_device_payload(
    column_names: &[String],
    column_types: &[SqlType],
    rows: &[Vec<SqlValue>],
) -> Result<
    (
        Vec<u8>,
        Vec<ResidentDeviceTextColumnLayout>,
        Vec<ResidentDeviceBoolColumnLayout>,
        Vec<ResidentDeviceInt4ColumnStats>,
        // (column name, byte-offset) of each numeric/uuid 16-byte section -- so a non-table caller
        // (the GPU grouped-sort) can address them without recomputing the layout by formula.
        Vec<(String, u64)>,
        // Per-column NULL validity bitmaps (M3 — doc 21), one per column that contains a NULL.
        Vec<ResidentDeviceNullBitmapLayout>,
    ),
    ExecuteError,
> {
    // The unified-buffer / sealed-shard path is exactly `capacity == row_count` — every padding loop
    // in the capacity-aware builder is then zero-iteration, so the bytes are identical to before.
    build_relational_device_payload_with_capacity(column_names, column_types, rows, rows.len())
}

/// Slice 1a (GPU-native writes): build the columnar payload with each FIXED-WIDTH section sized for
/// `capacity >= row_count` rows (headroom = `capacity - row_count` zero-padded slots), so an OPEN shard
/// can have new rows appended into the headroom in place (via `append_owned_chunks`) instead of being
/// rebuilt + re-uploaded. The 8-byte header still records `row_count` (the live row count); section
/// OFFSETS derive from `capacity` (callers/read-helpers pass it). `capacity == row_count` reproduces the
/// dense build byte-for-byte. Headroom is only definable for fixed-width sections, so an open
/// (`capacity > row_count`) payload rejects variable-length text columns; the open-shard route declines
/// text tables to the full re-admit.
pub(crate) fn build_relational_device_payload_with_capacity(
    column_names: &[String],
    column_types: &[SqlType],
    rows: &[Vec<SqlValue>],
    capacity: usize,
) -> Result<
    (
        Vec<u8>,
        Vec<ResidentDeviceTextColumnLayout>,
        Vec<ResidentDeviceBoolColumnLayout>,
        Vec<ResidentDeviceInt4ColumnStats>,
        Vec<(String, u64)>,
        Vec<ResidentDeviceNullBitmapLayout>,
    ),
    ExecuteError,
> {
    let row_count = rows.len();
    if capacity < row_count {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident payload capacity {capacity} is below the row count {row_count}"
        ))));
    }
    // Defensive bound: `capacity` is engine-chosen (an open-shard size), never user input, but guard an
    // absurd value that would pad/allocate a multi-GB payload (an allocation panic) rather than return a
    // clean error — the read-offset helpers already use checked arithmetic.
    if capacity > (1_usize << 31) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident payload capacity {capacity} is implausibly large"
        ))));
    }
    if capacity > row_count && column_types.iter().any(|ty| matches!(ty, SqlType::Text)) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "open-shard (capacity-padded) resident payload does not support variable-length text columns"
                .to_string(),
        )));
    }
    let mut device_payload = vec![0u8; std::mem::size_of::<u64>()];
    let mut resident_device_text_columns = Vec::new();
    let mut resident_device_bool_columns = Vec::new();
    let mut resident_device_int4_column_stats = Vec::new();
    let mut resident_device_b128_columns: Vec<(String, u64)> = Vec::new();
    let mut resident_device_null_columns: Vec<ResidentDeviceNullBitmapLayout> = Vec::new();

    // int4 / date / int2 share the i32 section (a date is i32 days; a smallint widens to i32).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
        .map(|(i, _)| i)
    {
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        for row in rows {
            // A NULL writes a don't-care 0 placeholder (the validity bitmap marks the row; the kernels
            // skip it) and is EXCLUDED from min/max so it can't pull the stats toward 0.
            let is_null = matches!(row[col_idx], SqlValue::Null);
            let value: i32 = match row[col_idx] {
                SqlValue::Int4(value) | SqlValue::Date(value) => value,
                SqlValue::Int2(value) => i32::from(value),
                SqlValue::Null => 0,
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot int4/date/int2 payload encountered a non-i32 value"
                            .to_string(),
                    )));
                }
            };
            if !is_null {
                min = min.min(value);
                max = max.max(value);
            }
            device_payload.extend_from_slice(&value.to_le_bytes());
        }
        for _ in row_count..capacity {
            device_payload.extend_from_slice(&0_i32.to_le_bytes());
        }
        resident_device_int4_column_stats.push(ResidentDeviceInt4ColumnStats {
            name: column_names[col_idx].clone(),
            min,
            max,
        });
    }
    // int8 / timestamp share the i64 section (a timestamp is i64 microseconds).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Int8 | SqlType::Timestamp))
        .map(|(i, _)| i)
    {
        for row in rows {
            // NULL → a don't-care 0 placeholder (the validity bitmap marks the row).
            let value: i64 = match row[col_idx] {
                SqlValue::Int8(value) | SqlValue::Timestamp(value) => value,
                SqlValue::Null => 0,
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot int8/timestamp payload encountered a non-i64 value"
                            .to_string(),
                    )));
                }
            };
            device_payload.extend_from_slice(&value.to_le_bytes());
        }
        for _ in row_count..capacity {
            device_payload.extend_from_slice(&0_i64.to_le_bytes());
        }
    }
    // numeric / uuid share the 16-byte section (numeric = i128 mantissa LE; uuid = raw 16 bytes).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Numeric { .. } | SqlType::Uuid))
        .map(|(i, _)| i)
    {
        let section_byte_offset = device_payload.len() as u64;
        for row in rows {
            match &row[col_idx] {
                SqlValue::Numeric(value) => {
                    device_payload.extend_from_slice(&value.mantissa.to_le_bytes());
                }
                SqlValue::Uuid(bytes) => device_payload.extend_from_slice(bytes),
                // NULL → 16 don't-care zero bytes (the validity bitmap marks the row).
                SqlValue::Null => device_payload.extend_from_slice(&[0u8; 16]),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot numeric/uuid payload encountered a wrong-typed value"
                            .to_string(),
                    )));
                }
            }
        }
        for _ in row_count..capacity {
            device_payload.extend_from_slice(&[0u8; 16]);
        }
        resident_device_b128_columns.push((column_names[col_idx].clone(), section_byte_offset));
    }
    // bool -> a 1-bit-per-row bitmap (ceil(row_count/32) LE u32 words, bit i = row i, LSB-first).
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Bool))
        .map(|(i, _)| i)
    {
        let bitmap_byte_offset = device_payload.len() as u64;
        let mut words = vec![0u32; capacity.div_ceil(32)];
        for (i, row) in rows.iter().enumerate() {
            match row[col_idx] {
                // NULL leaves the value bit 0 (don't-care; the validity bitmap marks the row).
                SqlValue::Bool(true) => words[i / 32] |= 1u32 << (i % 32),
                SqlValue::Bool(false) | SqlValue::Null => {}
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot bool payload encountered a non-bool value".to_string(),
                    )));
                }
            }
        }
        for word in &words {
            device_payload.extend_from_slice(&word.to_le_bytes());
        }
        resident_device_bool_columns.push(ResidentDeviceBoolColumnLayout {
            name: column_names[col_idx].clone(),
            bitmap_byte_offset,
        });
    }
    // NULL validity bitmaps (M3 — doc 21): ONE 1-bit-per-row bitmap (1 = valid/present, 0 = NULL,
    // LSB-first u32 words like bool) per column that actually contains a NULL. A no-NULL column emits
    // nothing — absence ⇒ all-valid — so existing non-null payloads stay byte-identical. Placed after
    // the bool section (every preceding section is a multiple of 4 bytes ⇒ this section start is
    // 4-aligned, so the u32 words load safely) and before text (text records its own offset, so it just
    // starts later). Iterates ALL columns in catalog order — a NULL can appear in any type, its value
    // riding the don't-care placeholder its own typed section wrote above.
    for (col_idx, name) in column_names.iter().enumerate() {
        if !rows.iter().any(|row| matches!(row[col_idx], SqlValue::Null)) {
            continue;
        }
        let bitmap_byte_offset = device_payload.len() as u64;
        let mut words = vec![0u32; capacity.div_ceil(32)];
        for (i, row) in rows.iter().enumerate() {
            if !matches!(row[col_idx], SqlValue::Null) {
                words[i / 32] |= 1u32 << (i % 32); // 1 = valid/present
            }
        }
        for word in &words {
            device_payload.extend_from_slice(&word.to_le_bytes());
        }
        resident_device_null_columns.push(ResidentDeviceNullBitmapLayout {
            name: name.clone(),
            bitmap_byte_offset,
        });
    }
    // text -> an 8-ALIGNED offsets section (n+1 i64 LE; read as 2x ld.u32 -> 716-safe) + a bytes blob.
    for col_idx in column_types
        .iter()
        .enumerate()
        .filter(|&(_i, ty)| matches!(ty, SqlType::Text))
        .map(|(i, _)| i)
    {
        while !device_payload.len().is_multiple_of(8) {
            device_payload.push(0);
        }
        let offsets_byte_offset = device_payload.len() as u64;
        let mut text_offsets = Vec::with_capacity(row_count + 1);
        let mut text_bytes = Vec::new();
        text_offsets.push(0_u64);
        for row in rows {
            match &row[col_idx] {
                SqlValue::Text(value) => text_bytes.extend_from_slice(value.as_bytes()),
                // NULL → an empty (zero-length) placeholder span; the validity bitmap marks the row.
                SqlValue::Null => {}
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident snapshot text payload encountered non-text value".to_string(),
                    )));
                }
            }
            text_offsets.push(text_bytes.len() as u64);
        }
        for offset in &text_offsets {
            device_payload.extend_from_slice(&offset.to_le_bytes());
        }
        let bytes_byte_offset = device_payload.len() as u64;
        device_payload.extend_from_slice(&text_bytes);
        resident_device_text_columns.push(ResidentDeviceTextColumnLayout {
            name: column_names[col_idx].clone(),
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len: text_bytes.len() as u64,
        });
    }
    device_payload[..std::mem::size_of::<u64>()]
        .copy_from_slice(&(row_count as u64).to_le_bytes());
    Ok((
        device_payload,
        resident_device_text_columns,
        resident_device_bool_columns,
        resident_device_int4_column_stats,
        resident_device_b128_columns,
        resident_device_null_columns,
    ))
}

/// Decode a fixed-width int4 SqlValue to its i32 residency encoding, matching
/// `compute_open_shard_int4_append_chunks` byte-for-byte (Int4/Date pass through, Int2 widens,
/// NULL materializes as 0). Used to extend a shard's zone map (min/max) on in-place append.
pub(crate) fn sql_value_as_int4(value: &SqlValue) -> i32 {
    match value {
        SqlValue::Int4(v) | SqlValue::Date(v) => *v,
        SqlValue::Int2(v) => i32::from(*v),
        _ => 0,
    }
}

/// The uniform memset byte whose repetition is the `deleted_by` LIVE sentinel `0x7F7F_7F7F_7F7F_7F7F` — a
/// large POSITIVE signed i64 (the device visibility compare `deleted_by > read_txn_id` is a signed s64
/// kernel; `u64::MAX` would be -1 signed and a live row would wrongly fail the compare) that exceeds every
/// real commit `Index`, and is memset-friendly (uniform byte) for both the on-demand region and the SV3a
/// recompaction fill.
pub(crate) const DELETED_BY_LIVE_FILL_BYTE: u8 = 0x7F;

/// Slice 1b-ii: compute the per-section append chunks that write `new_rows` into an OPEN shard's
/// reserved headroom starting at slot `row_start`, for a capacity-padded INT4 layout of `capacity`
/// slots. Each chunk lands EXACTLY where the capacity-aware read offsets expect it (column `c` at
/// `8 + c*capacity*4`, row `r` at `+ r*4`), so feeding these to `append_owned_chunks` makes the open
/// shard byte-identical to a full rebuild. The header (live row count) chunk is emitted LAST so a
/// partial append (a mid-list CUDA failure) can never advertise rows whose column bytes are missing
/// (the `append_owned_chunks` partial-failure contract).
///
/// Returns `Err` — the caller must fall back to a full re-admit — when the table is not all
/// int4/date/int2 (this slice's open-shard append is fixed-width-i32 only; text/int8/numeric ride the
/// re-admit path until later slices), and when `row_start + new_rows.len() > capacity` (the headroom is
/// exhausted — the caller must seal this shard and roll a fresh open one).
pub(crate) fn compute_open_shard_int4_append_chunks(
    column_types: &[SqlType],
    capacity: usize,
    row_start: usize,
    new_rows: &[Vec<SqlValue>],
) -> Result<Vec<CudaOwnedDeviceMemoryChunk>, ExecuteError> {
    let appended = new_rows.len();
    let end = row_start.checked_add(appended).ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "open-shard append row index overflowed".to_string(),
        ))
    })?;
    if end > capacity {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "open-shard append of {appended} rows at {row_start} exceeds capacity {capacity}"
        ))));
    }
    // Mirror the builder's defensive capacity bound so the unchecked offset multiplies below cannot
    // overflow usize (the read helpers + append_owned_chunks are likewise checked/guarded).
    if capacity > (1_usize << 31) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "open-shard append capacity {capacity} is implausibly large"
        ))));
    }
    if !column_types
        .iter()
        .all(|ty| matches!(ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "open-shard append supports int4/date/int2 columns only (this slice)".to_string(),
        )));
    }
    // Row-arity guard: a malformed (short/long) row must return the fallback Err, never panic on the
    // unchecked `row[col_idx]` indexing below.
    if new_rows.iter().any(|row| row.len() != column_types.len()) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "open-shard append row has the wrong column count".to_string(),
        )));
    }
    let header_bytes = std::mem::size_of::<u64>();
    let width = std::mem::size_of::<i32>();
    // Column chunks FIRST, header LAST (the partial-failure contract: never advertise un-written rows).
    let mut chunks = Vec::with_capacity(column_types.len() + 1);
    for (col_idx, _ty) in column_types.iter().enumerate() {
        let section_start = header_bytes + col_idx * capacity * width;
        let byte_offset = (section_start + row_start * width) as u64;
        let mut bytes = Vec::with_capacity(appended * width);
        for row in new_rows {
            let value: i32 = match row[col_idx] {
                SqlValue::Int4(value) | SqlValue::Date(value) => value,
                SqlValue::Int2(value) => i32::from(value),
                // A NULL int4 materializes as 0 (the validity bitmap, a later slice, marks the row).
                SqlValue::Null => 0,
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "open-shard int4 append encountered a non-i32 value".to_string(),
                    )))
                }
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        chunks.push(CudaOwnedDeviceMemoryChunk { byte_offset, bytes });
    }
    chunks.push(CudaOwnedDeviceMemoryChunk {
        byte_offset: 0,
        bytes: (end as u64).to_le_bytes().to_vec(),
    });
    Ok(chunks)
}

#[cfg(test)]
mod capacity_payload_tests {
    use super::*;

    fn int4_cols() -> (Vec<String>, Vec<SqlType>) {
        (
            vec!["id".to_string(), "balance".to_string()],
            vec![SqlType::Int4, SqlType::Int4],
        )
    }
    fn int4_rows(n: i32) -> Vec<Vec<SqlValue>> {
        (0..n)
            .map(|i| vec![SqlValue::Int4(i), SqlValue::Int4(i * 10)])
            .collect()
    }

    /// NON-VACUOUS layout check across ALL section types (opus audit finding): the prior test compared
    /// the delegating wrapper to `..._with_capacity(.., row_count)` — the SAME call — so it could never
    /// fail, and it only used int4. This asserts the EXACT dense layout of a multi-type schema at 32 rows
    /// (a 32-word bitmap boundary), so a wrong section size — e.g. a bitmap `div_ceil` regression — shifts
    /// the total length / section offsets and is caught. (Text is omitted: it is data-dependent and
    /// self-describing, and is covered by the null/text suites.)
    #[test]
    fn dense_multitype_payload_layout_is_exact() {
        // Catalog order: id i32, maybe i32 (nullable), big i64, amt numeric (16B), flag bool.
        let names: Vec<String> = ["id", "maybe", "big", "amt", "flag"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let types = vec![
            SqlType::Int4,
            SqlType::Int4,
            SqlType::Int8,
            SqlType::Numeric {
                precision: 18,
                scale: 0,
            },
            SqlType::Bool,
        ];
        let n = 32_usize; // capacity == row_count (dense / production path); crosses a 32-word boundary
        let rows: Vec<Vec<SqlValue>> = (0..n as i64)
            .map(|i| {
                vec![
                    SqlValue::Int4(i as i32),
                    if i % 4 == 0 {
                        SqlValue::Null
                    } else {
                        SqlValue::Int4(i as i32 * 2)
                    },
                    SqlValue::Int8(i * 1000),
                    SqlValue::Numeric(gpu_db_sql::Decimal128::new(i as i128, 0)),
                    SqlValue::Bool(i % 2 == 0),
                ]
            })
            .collect();
        let p = build_relational_device_payload_with_capacity(&names, &types, &rows, n)
            .unwrap()
            .0;
        // Exact layout: header 8 | i32 x2 (n*4 each) | i64 (n*8) | 16B (n*16) | bool 1 word | null 1 word.
        let words = n.div_ceil(32); // 1 at n=32; a div_ceil regression makes this 2 -> length changes
        let i64_off = 8 + 2 * n * 4;
        let b128_off = i64_off + n * 8;
        let bool_off = b128_off + n * 16;
        let null_off = bool_off + words * 4;
        let expected_len = null_off + words * 4;
        assert_eq!(p.len(), expected_len, "exact dense multi-type payload length");
        assert_eq!(
            u64::from_le_bytes(p[0..8].try_into().unwrap()),
            n as u64,
            "header = live row count"
        );
        // i64 section spot-check: row 5 = 5000.
        let off = i64_off + 5 * 8;
        assert_eq!(i64::from_le_bytes(p[off..off + 8].try_into().unwrap()), 5000);
        // bool bitmap: row0 flag=true -> bit0 set; row1 flag=false -> bit1 clear.
        let bool_word = u32::from_le_bytes(p[bool_off..bool_off + 4].try_into().unwrap());
        assert_eq!(bool_word & 0b11, 0b01, "flag bits: row0 set, row1 clear");
        // NULL validity bitmap (1 = present): row0=NULL -> bit0 clear; row1=present -> bit1 set.
        let null_word = u32::from_le_bytes(p[null_off..null_off + 4].try_into().unwrap());
        assert_eq!(null_word & 0b11, 0b10, "validity bits: row0 NULL, row1 present");
    }

    /// Proves the offset helpers are CAPACITY-aware (opus audit #5: the only thing that actually
    /// exercises the capacity stride through the helpers — the regression only covers capacity ==
    /// row_count). With capacity > row_count the int8 section must start AFTER the capacity-padded int4
    /// sections, not the row_count-sized ones.
    #[test]
    fn offset_helpers_use_capacity_not_row_count() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE t (a INT, b INT, c BIGINT)")
            .expect("create");
        let table = engine.relational_catalog_table("t").expect("table");

        let snapshot = |capacity: usize| RelationalResidencySnapshot {
            gpu_id: 0,
            schema: "public".to_string(),
            table: "t".to_string(),
            generation: 0,
            row_count: 3,
            capacity,
            column_count: 3,
            resident_bytes: 0,
            resident_device_int4_columns: vec!["a".to_string(), "b".to_string()],
            resident_device_int4_column_stats: vec![],
            resident_device_int8_columns: vec!["c".to_string()],
            resident_device_numeric_columns: vec![],
            resident_device_bool_columns: vec![],
            resident_device_text_columns: vec![],
            resident_device_null_columns: vec![],
            valid_through_index: 0,
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: vec![],
            device_memory_proof: None,
        };

        // capacity = 8 > row_count = 3: each int4 section is capacity*4 = 32 bytes.
        let s8 = snapshot(8);
        assert_eq!(resident_device_int4_column_offset(&s8, &table, 0).unwrap(), 8);
        assert_eq!(resident_device_int4_column_offset(&s8, &table, 1).unwrap(), 8 + 8 * 4);
        // int8 col `c` starts AFTER both capacity-padded int4 sections: 8 + 2*(8*4) = 72.
        assert_eq!(resident_device_int8_column_offset(&s8, &table, 2).unwrap(), 8 + 2 * 8 * 4);

        // Dense (capacity == row_count == 3): int8 col `c` at 8 + 2*(3*4) = 32 — proving capacity, not
        // row_count, drives the stride (a row_count stride would give 32 for BOTH cases).
        let s3 = snapshot(3);
        assert_eq!(resident_device_int8_column_offset(&s3, &table, 2).unwrap(), 8 + 2 * 3 * 4);
    }

    /// 1b-ii: int4 open-shard append chunks land at the capacity-aware read offsets (column c at
    /// `8 + c*capacity*4`, row r at `+ r*4`), header LAST (partial-failure contract), with eligibility
    /// + capacity-overflow rejection (the full-rebuild fallback).
    #[test]
    fn open_shard_int4_append_chunks_match_capacity_layout() {
        let types = vec![SqlType::Int4, SqlType::Int4]; // id, balance
        let capacity = 8;
        let row_start = 3;
        let new_rows = vec![
            vec![SqlValue::Int4(3), SqlValue::Int4(30)],
            vec![SqlValue::Int4(4), SqlValue::Int4(40)],
        ];
        let chunks =
            compute_open_shard_int4_append_chunks(&types, capacity, row_start, &new_rows).unwrap();
        assert_eq!(chunks.len(), 3, "2 column chunks + 1 header chunk");
        let le = |vals: &[i32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
        // col0 (id): section 8 + 0*8*4 = 8; row 3 -> 8 + 3*4 = 20; bytes [3,4].
        assert_eq!(chunks[0].byte_offset, 8 + 3 * 4);
        assert_eq!(chunks[0].bytes, le(&[3, 4]));
        // col1 (balance): section 8 + 1*8*4 = 40; row 3 -> 52; bytes [30,40].
        assert_eq!(chunks[1].byte_offset, 8 + 8 * 4 + 3 * 4);
        assert_eq!(chunks[1].bytes, le(&[30, 40]));
        // header LAST: offset 0, live row count = row_start + 2 = 5.
        assert_eq!(chunks[2].byte_offset, 0);
        assert_eq!(chunks[2].bytes, 5_u64.to_le_bytes().to_vec());

        // ineligible (text) -> Err (caller falls back to re-admit).
        assert!(
            compute_open_shard_int4_append_chunks(&[SqlType::Int4, SqlType::Text], capacity, 0, &[])
                .is_err()
        );
        // capacity overflow -> Err (caller seals + rolls a new shard).
        let two = vec![vec![SqlValue::Int4(0)], vec![SqlValue::Int4(1)]];
        assert!(compute_open_shard_int4_append_chunks(&[SqlType::Int4], 4, 3, &two).is_err());

        // value-encoding arms (opus coverage note): Int2 widens, Date passes, NULL -> 0.
        let mixed = compute_open_shard_int4_append_chunks(
            &[SqlType::Int2, SqlType::Date],
            4,
            0,
            &[
                vec![SqlValue::Int2(7), SqlValue::Date(100)],
                vec![SqlValue::Null, SqlValue::Null],
            ],
        )
        .unwrap();
        assert_eq!(mixed[0].bytes, le(&[7, 0]), "int2 widened + null->0");
        assert_eq!(mixed[1].byte_offset, 8 + 4 * 4, "date section after the int2 section");
        assert_eq!(mixed[1].bytes, le(&[100, 0]), "date pass-through + null->0");
        // a wrong-typed value in an eligible column -> Err (clean, no panic).
        assert!(compute_open_shard_int4_append_chunks(
            &[SqlType::Int4],
            4,
            0,
            &[vec![SqlValue::Int8(1)]]
        )
        .is_err());
        // a malformed (short) row -> Err, never a panic (the row-arity guard).
        assert!(compute_open_shard_int4_append_chunks(
            &[SqlType::Int4, SqlType::Int4],
            4,
            0,
            &[vec![SqlValue::Int4(1)]]
        )
        .is_err());
    }

    /// Slice 1b-ii-c END-TO-END: committed INSERTs on a GPU-resident int4 table APPEND in place to the
    /// open shard via the SERIALIZED commit path (`commit_mutation_at`, the path `execute_text` — hence
    /// the façade — actually uses) instead of re-uploading the whole table. All gates read through the
    /// DEVICE resident route (the retained-template point-lookup path + the wave index), NOT the MVCC
    /// store, so they actually exercise the appended device bytes + the index (an earlier version read
    /// the store and was vacuous — append==re-admit there by construction). Gates:
    ///  1. NON-VACUITY — the append FIRED: across 50 in-headroom inserts the open_shard_append_hits
    ///     counter advances by EXACTLY 50 (sabotage the commit hook → 0 → fail). Output equality / device
    ///     ptr-stability cannot prove it (append==re-admit byte-identical; a same-size re-admit reuses
    ///     the freed address).
    ///  2. DEVICE CORRECTNESS + Finding A — the GPU index probe over the appended shard equals the scan,
    ///     resolves an APPENDED key to its bytes, and misses an absent key. A generation-blind stale
    ///     index (cache keyed on the unchanged device ptr) would miss the appended key.
    ///  3. NULL GUARD (audit DO-NOT-SHIP fix) — a committed INSERT carrying a NULL must NOT append in
    ///     place (no validity bitmap on the open shard → an appended NULL reads as a phantom 0 on the
    ///     device aggregate/DISTINCT routes); it must re-admit, so the counter does NOT advance.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn open_shard_append_fires_in_place_and_matches_device_route() {
        use gpu_db_sql::{parse_command, Command, Select};
        let select_cmd = |sql: &str| -> Select {
            match parse_command(sql).unwrap() {
                Command::Select(s) => s,
                other => panic!("expected SELECT, got {other:?}"),
            }
        };
        // DEVICE route, per needle: the resident retained-template point-lookup path (wave index when
        // enabled, else the device scan) — reads the resident buffer, NOT the MVCC store.
        let run = |e: &Engine, select: &Select, needles: &[i32]| -> Vec<RowBlock> {
            let template = e.prepare_relational_retained_read_template(select).unwrap();
            let submission = e
                .submit_relational_retained_template_point_lookups(&template, needles)
                .unwrap();
            e.complete_relational_retained_read_submission(submission)
                .unwrap()
                .iter()
                .map(|r| r.rows.clone())
                .collect()
        };

        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        // Settle at 300 rows: the open shard's last headroom-overflow re-admit (at row 129) set capacity
        // 512, so rows 130..512 — incl. the next 50 appends — fit without a further re-admit.
        for i in 0..300_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let select_unique = select_cmd("SELECT id, balance FROM accounts WHERE id = 1");
        if !e.plan_relational_resident_route(&select_unique).accepted {
            return; // resident device route unavailable (no GPU / not accepted) -> nothing to exercise
        }

        // (1) NON-VACUITY: all 50 in-headroom commits take the IN-PLACE append.
        let hits_before = e.open_shard_append_hits();
        for i in 300..350_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        assert_eq!(
            e.open_shard_append_hits() - hits_before,
            50,
            "all 50 in-headroom commits must take the IN-PLACE open-shard append (sabotaging the commit \
             hook drops this to 0); output equality alone cannot prove the append fired"
        );

        // (2) DEVICE CORRECTNESS + Finding A: index-probe vs scan over needles incl. APPENDED keys
        // (342, 349) and an absent key (999).
        let needles = vec![5_i32, 200, 342, 349, 999];
        e.set_index_probe_enabled(false);
        let scan = run(&e, &select_unique, &needles);
        e.set_index_probe_enabled(true);
        let index = run(&e, &select_unique, &needles);
        assert_eq!(
            index, scan,
            "GPU index-probe rows over the appended shard must equal the device scan rows (guards \
             Finding A: an in-place append keeps the device ptr, so a generation-blind cached index \
             would be stale and miss appended keys)"
        );
        // Non-vacuous: the flag-on run BUILT a real GPU index over the `id` filter column (col 0).
        {
            let cache = e
                .read_state
                .residency
                .wave_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let entry = cache
                .get("accounts")
                .expect("wave index cached after a flag-on run");
            assert!(
                entry.index_memory.is_some() && entry.column_idx == 0,
                "the index-probe run must have built a real GPU index over the id column"
            );
        }
        // ABSOLUTE: the in-place-appended key 342 resolves to its appended bytes; 999 is absent.
        let at = |n: i32| index[needles.iter().position(|&x| x == n).unwrap()].clone();
        let row342 = at(342);
        assert_eq!(row342.len(), 1, "appended key 342 present via the device route");
        assert_eq!(
            row342.row(0),
            &[SqlValue::Int4(342), SqlValue::Int4(3420)],
            "the device route returns the in-place-appended row's bytes"
        );
        assert!(at(999).is_empty(), "absent key 999 returns no row");

        // (2b) Finding A SPECIFICALLY: the flag-on run above BUILT + cached the GPU index. Now append a
        // new key IN PLACE (same device ptr — capacity 512 holds row 351) and look it up via the index.
        // The append must have INVALIDATED the cached index so the rebuild sees the new key; a stale,
        // generation-blind HIT (cache keyed only on the unchanged ptr) would miss it. Remove the
        // wave_index invalidation in try_append -> this lookup returns empty -> fail.
        e.execute_text(20_000, "INSERT INTO accounts (id, balance) VALUES (360, 3600)")
            .unwrap();
        let after = run(&e, &select_unique, &[360]);
        assert_eq!(
            after[0].len(),
            1,
            "a key appended in place AFTER the index was built must be found — try_append must \
             invalidate the cached GPU index (audit Finding A); a generation-blind stale HIT misses it"
        );
        assert_eq!(
            after[0].row(0),
            &[SqlValue::Int4(360), SqlValue::Int4(3600)],
            "the post-build in-place-appended row resolves to its bytes via the rebuilt index"
        );

        // (3) NULL-GUARD NON-VACUITY (audit DO-NOT-SHIP fix): a committed INSERT carrying a NULL must
        // fall back to re-admit (build the validity bitmap), NOT append in place. Remove the NULL guard
        // in try_append -> this insert appends -> the counter advances -> this assertion fails.
        let hits_pre_null = e.open_shard_append_hits();
        e.execute_text(10_000, "INSERT INTO accounts (id, balance) VALUES (5000, NULL)")
            .unwrap();
        assert_eq!(
            e.open_shard_append_hits(),
            hits_pre_null,
            "an appended NULL must force re-admit (which builds the validity bitmap), NOT an in-place \
             append — an appended NULL has no bitmap and would read as a phantom 0 on the device"
        );
    }

    /// Slice 1b-ii-d: committed INSERTs append host rows as immutable SEGMENTS (so the commit is O(rows
    /// appended), not the O(table) `host_rows` deep-clone — the dual-store tax's last residual; benchmark
    /// shows the tax now flat ~40us at every base size). This gate proves the segmented `host_rows` reads
    /// back CORRECTLY through the HOST-MATERIALIZATION path
    /// (`execute_relational_select_with_resident_snapshot_probe`, which iterates `host_rows` segments via
    /// `host_rows_iter`): after many appends produce MULTIPLE segments (non-vacuity assert), a scan +
    /// point lookup over them must match the non-resident store baseline — same rows, same order. A
    /// cross-segment ordering bug (or a dropped segment) in `host_rows_iter` fails the scan equality.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn host_rows_segmented_append_reads_match_baseline() {
        use gpu_db_sql::{parse_command, Command};
        let load = |e: &Engine| {
            e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
                .unwrap();
            for i in 0..200_i64 {
                e.execute_text(
                    (i as u64) + 2,
                    &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
                )
                .unwrap();
            }
        };

        // Resident path: auto-admit -> the open-shard append fires repeatedly, accumulating host_rows
        // SEGMENTS between headroom-overflow re-admits.
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        load(&e);
        // NON-VACUITY: the appends produced MULTIPLE host_rows segments — else cross-segment iteration
        // order is untested (a single dense segment reads correctly trivially).
        let entry = e.relational_residency_entry("accounts").unwrap();
        assert!(
            entry.host_rows.len() > 1,
            "appends must produce multiple host_rows segments (got {}) or this gate is vacuous",
            entry.host_rows.len()
        );
        assert_eq!(entry.host_row_count(), 200, "all 200 rows across segments");
        drop(entry);

        // Non-resident store baseline (single canonical materialization).
        let base = Engine::new_local();
        load(&base);

        let resident = |sql: &str| match parse_command(sql).unwrap() {
            Command::Select(s) => e
                .execute_relational_select_with_resident_snapshot_probe(&s) // iterates host_rows segments
                .unwrap()
                .rows,
            _ => panic!("not a SELECT"),
        };
        let baseline = |sql: &str| match parse_command(sql).unwrap() {
            Command::Select(s) => base
                .execute_relational_select_with_cuda_driver_probe(&s)
                .unwrap()
                .rows,
            _ => panic!("not a SELECT"),
        };
        // No ORDER BY: the result order IS the host_rows_iter (segment) order, so this differential is
        // sensitive to cross-segment ordering (a re-sort would mask a wrong-order bug). Both paths yield
        // insertion / TupleId order, so they must match iff the segments iterate in append order.
        let scan = "SELECT id, balance FROM accounts";
        let point = "SELECT id, balance FROM accounts WHERE id = 137";
        let r_scan = resident(scan);
        assert_eq!(
            r_scan,
            baseline(scan),
            "segmented host_rows scan must match the baseline (cross-segment ORDER + content)"
        );
        assert_eq!(
            resident(point),
            baseline(point),
            "segmented host_rows point lookup must match the baseline"
        );
        assert_eq!(r_scan.len(), 200, "all 200 rows present via the segmented host path");
    }

    /// Billions-of-rows S-d1: admitting a table as a (single dense) SEGMENTED shard — `shard_residency`
    /// flag ON, routed through the sharded resident read path — must produce byte-identical reads to the
    /// single capacity-padded unified buffer (flag OFF). NON-VACUITY: with the flag ON the table lands in
    /// `residency.shards` (the sharded route), and NOT with the flag OFF (the single-buffer route).
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_residency_admit_reads_match_single_buffer() {
        let run = |shard: bool| -> (RowBlock, RowBlock, RowBlock, bool) {
            let mut e = Engine::new_local();
            e.set_shard_residency_enabled(shard);
            e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
                .unwrap();
            let mut txn = 2u64;
            let mut id = 0i64;
            while id < 1000 {
                let mut vals = String::new();
                for _ in 0..200 {
                    if id >= 1000 {
                        break;
                    }
                    if !vals.is_empty() {
                        vals.push(',');
                    }
                    vals.push_str(&format!("({id}, {})", id * 10));
                    id += 1;
                }
                e.execute_text(txn, &format!("INSERT INTO accounts (id, balance) VALUES {vals}"))
                    .unwrap();
                txn += 1;
            }
            e.populate_relational_residency_snapshot("accounts").unwrap();
            let in_shards = e
                .read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .is_some_and(|s| !s.is_empty());
            let sel = |sql: &str| -> RowBlock {
                e.execute_relational_select_text(sql).unwrap().rows
            };
            (
                sel("SELECT id, balance FROM accounts WHERE id = 137"),
                sel("SELECT id, balance FROM accounts ORDER BY id"),
                sel("SELECT COUNT(*) FROM accounts"),
                in_shards,
            )
        };
        let (on_pt, on_scan, on_cnt, on_in_shards) = run(true);
        let (off_pt, off_scan, off_cnt, off_in_shards) = run(false);
        // NON-VACUITY: flag ON took the sharded route; flag OFF took the single buffer.
        assert!(
            on_in_shards,
            "flag ON must admit the table as a shard (the sharded read route)"
        );
        assert!(
            !off_in_shards,
            "flag OFF must use the single buffer (no shard)"
        );
        // point-lookup + COUNT(*) ARE served by the sharded route (the non-vacuous sharded gates); the
        // ORDER BY scan on a shard table takes the CPU fallback (a shard has no `snapshots` entry, so the
        // gpu-sortable gate declines) — kept as a correctness check (CPU shard path == GPU single buffer).
        assert_eq!(on_pt, off_pt, "sharded point lookup == single-buffer baseline");
        assert_eq!(on_scan, off_scan, "scan (CPU fallback) == single-buffer baseline");
        assert_eq!(on_cnt, off_cnt, "sharded COUNT(*) == single-buffer baseline");
        assert_eq!(on_scan.len(), 1000, "all 1000 rows present");

        // S-d2a non-vacuity: the OPEN shard carries capacity HEADROOM (capacity > row_count), and the
        // capacity-aware sharded recompaction above addressed it correctly (the reads matched the
        // baseline — a dense-stride recompaction would have copied the wrong column slice and diverged).
        {
            let mut e = Engine::new_local();
            e.set_shard_residency_enabled(true);
            e.execute_text(1, "CREATE TABLE t (id INT, balance INT)").unwrap();
            e.execute_text(2, "INSERT INTO t (id, balance) VALUES (1,10),(2,20),(3,30)")
                .unwrap();
            e.populate_relational_residency_snapshot("t").unwrap();
            let shards = e.read_state.residency.shards.load();
            let shard = &shards.get("t").expect("table admitted as a shard")[0];
            assert_eq!(shard.row_count, 3, "live row count");
            assert!(
                shard.capacity > shard.row_count,
                "open shard must carry headroom: capacity {} > row_count {}",
                shard.capacity,
                shard.row_count
            );
        }
    }

    /// Audit (S-d1) fix: a runtime `shard_residency` flag FLIP + re-admit must clear the OPPOSITE residency
    /// representation, so a stale shard can't shadow a fresh snapshot (the read route checks shards FIRST →
    /// it would serve wrong rows) and a stale snapshot can't shadow fresh shards. Verifies the map state
    /// after each flip — WITHOUT the clears the stale cell persists, so the `!in_*` asserts fail.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_residency_flag_flip_clears_opposite_representation() {
        let mut e = Engine::new_local();
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        e.execute_text(
            2,
            "INSERT INTO accounts (id, balance) VALUES (1, 100), (2, 200), (3, 300)",
        )
        .unwrap();
        let in_shards = |e: &Engine| {
            e.read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .is_some_and(|s| !s.is_empty())
        };
        let in_snaps =
            |e: &Engine| e.read_state.residency.snapshots.load().get("accounts").is_some();

        // OFF -> single buffer.
        e.set_shard_residency_enabled(false);
        e.populate_relational_residency_snapshot("accounts").unwrap();
        assert!(in_snaps(&e) && !in_shards(&e), "OFF admits the single buffer");
        // Flip ON -> shard; the stale snapshot must be cleared.
        e.set_shard_residency_enabled(true);
        e.populate_relational_residency_snapshot("accounts").unwrap();
        assert!(
            in_shards(&e) && !in_snaps(&e),
            "OFF->ON re-admit must clear the stale snapshot"
        );
        // Flip OFF -> single buffer; the stale shard must be cleared (the wrong-rows footgun).
        e.set_shard_residency_enabled(false);
        e.populate_relational_residency_snapshot("accounts").unwrap();
        assert!(
            in_snaps(&e) && !in_shards(&e),
            "ON->OFF re-admit must clear the stale shard (else it shadows the fresh snapshot)"
        );
        // Reads are correct after the flips.
        let r = e
            .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 2")
            .unwrap()
            .rows;
        assert_eq!(r.len(), 1, "id=2 present");
        assert_eq!(r.row(0), &[SqlValue::Int4(2), SqlValue::Int4(200)]);
    }

    /// S-d2b: with the shard flag ON, a committed INSERT appends IN PLACE into the resident OPEN shard's
    /// headroom (no whole-table re-admit) — the shard-path analog of 1b-ii-c/d. Across 50 in-headroom
    /// commits the open_shard_append_hits counter advances by exactly 50, the table stays ONE shard (in
    /// place, not rollover/re-admit), the open shard's row_count grows, and a point lookup + COUNT over the
    /// sharded route return the appended rows. (Counter +50 vs 0 distinguishes append from re-admit.)
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_open_append_in_place_reads_correct() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        // Batch-admit 300 rows as ONE shard with headroom (default 4M target -> the admit builds a
        // capacity-1024 open shard holding 300 rows), so the next 50 single-row commits append IN PLACE
        // without a rollover (rollover is exercised by shard_open_append_rolls_over_and_reads_correct).
        let base: Vec<String> = (0..300_i64).map(|i| format!("({i}, {})", i * 10)).collect();
        e.execute_text(
            2,
            &format!("INSERT INTO accounts (id, balance) VALUES {}", base.join(",")),
        )
        .unwrap();
        let shard_state = |e: &Engine| -> (usize, usize) {
            let shards = e.read_state.residency.shards.load();
            let s = shards.get("accounts").expect("shard-resident");
            (s.len(), s.last().unwrap().row_count)
        };
        let count_before = shard_state(&e).0;
        let hits_before = e.open_shard_append_hits();
        for i in 300..350_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let (count_after, open_row_count) = shard_state(&e);
        // NON-VACUITY: the 50 in-headroom commits appended IN PLACE to the open shard (counter +50, vs 0
        // for a re-admit), the shard count is unchanged (no rollover/re-admit), and row_count grew.
        assert_eq!(
            e.open_shard_append_hits() - hits_before,
            50,
            "all 50 in-headroom commits must append in place to the open shard (re-admit would give 0)"
        );
        assert_eq!(count_after, count_before, "no rollover/re-admit -> shard count unchanged");
        assert_eq!(
            open_row_count, 350,
            "the open shard's row_count grew to 350 via in-place append"
        );

        // Reads over the sharded route include the in-place-appended rows.
        let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
        let pt = sel("SELECT id, balance FROM accounts WHERE id = 342");
        assert_eq!(pt.len(), 1, "appended key 342 present via the sharded route");
        assert_eq!(pt.row(0), &[SqlValue::Int4(342), SqlValue::Int4(3420)]);
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(350)],
            "COUNT(*) sees the appended rows"
        );
    }

    /// S-d2c: with the shard flag ON and a small shard-size target, a table built via committed INSERTs
    /// SEALS the full open shard and ROLLS OVER into a fresh one — growing as MULTIPLE bounded shards
    /// instead of ever re-admitting the whole table (this is what removes the ~536M single-buffer cap). The
    /// recompaction reads correctly across the sealed + open shards. NON-VACUITY: rollover produced multiple
    /// shards (a broken rollover -> re-admit -> ONE dense shard -> the assert fails); the per-shard row_counts
    /// sum to the table total; point lookups across different shards + COUNT(*) are correct.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_open_append_rolls_over_and_reads_correct() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // small -> roll over every ~64 rows
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let (shard_count, total_rows) = {
            let shards = e.read_state.residency.shards.load();
            let s = shards.get("accounts").expect("shard-resident");
            (s.len(), s.iter().map(|sh| sh.row_count).sum::<usize>())
        };
        // NON-VACUITY: rollover produced MULTIPLE bounded shards (200 rows / 64 target -> >= 3); a broken
        // rollover would re-admit into ONE dense shard.
        assert!(
            shard_count >= 3,
            "rollover must grow the table as multiple shards (got {shard_count})"
        );
        assert_eq!(total_rows, 200, "per-shard row_counts sum to the table total");

        // Reads recompact across all shards: point lookups landing in different shards + COUNT(*).
        let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
        for id in [5_i64, 70, 137, 199] {
            let r = sel(&format!("SELECT id, balance FROM accounts WHERE id = {id}"));
            assert_eq!(r.len(), 1, "id={id} present across shards");
            assert_eq!(
                r.row(0),
                &[SqlValue::Int4(id as i32), SqlValue::Int4((id * 10) as i32)],
                "id={id} value correct across shards"
            );
        }
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(200)],
            "COUNT(*) across all shards"
        );
        assert_eq!(
            sel("SELECT id, balance FROM accounts").len(),
            200,
            "scan recompacts all shards"
        );
    }

    /// S-d3: per-shard zone maps (min/max) PRUNE the sharded read. A table grown as several bounded shards
    /// by ORDERED committed INSERTs gives each shard a disjoint key range, so a point lookup gathers ~1
    /// shard instead of recompacting all of them — O(1), not O(num_shards).
    ///
    /// NON-VACUITY / anti-sabotage, all keyed off the `sharded_shards_gathered` telemetry (output equality
    /// alone can't see a skipped shard — a pruned shard holds no matching rows either way):
    ///  1. the table really is MULTIPLE shards (>= 3), so pruning to 1 is a real reduction;
    ///  2. a point lookup gathers EXACTLY 1 shard (a broken prune keeping all would read `shard_count`);
    ///  3. a full scan (no predicate) gathers ALL shards (proves the counter isn't hardwired to 1, and that
    ///     no-filter never prunes);
    ///  4. EVERY key across shard boundaries — including the MAX key, which was appended IN PLACE into the
    ///     open shard AFTER its rollover-admit — still reads its row (a broken append-time zone-map
    ///     extension would leave the open shard's max stale and WRONGLY prune the just-appended key -> 0 rows);
    ///  5. an absent key (beyond every range) prunes to the keep-one fallback (1 shard) and returns empty.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_zone_map_prunes_point_lookup() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // small -> several shards with disjoint ascending key ranges
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let shard_count = {
            let shards = e.read_state.residency.shards.load();
            shards.get("accounts").expect("shard-resident").len()
        };
        // (1) multiple shards, so a prune to 1 is meaningful.
        assert!(
            shard_count >= 3,
            "need several shards for pruning to matter (got {shard_count})"
        );

        let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
        let gathered = |e: &Engine| e.sharded_shards_gathered();

        // (2) a point lookup landing inside one shard's range gathers EXACTLY one shard.
        let before = gathered(&e);
        let pt = sel("SELECT id, balance FROM accounts WHERE id = 137");
        let one = gathered(&e) - before;
        assert_eq!(pt.len(), 1, "id=137 present");
        assert_eq!(pt.row(0), &[SqlValue::Int4(137), SqlValue::Int4(1370)]);
        assert_eq!(
            one, 1,
            "zone-map prune must gather exactly ONE shard for a point lookup (got {one} of {shard_count})"
        );

        // (3) COUNT(*) (shape `sharded_count_all`) routes through the SAME recompaction function but carries
        //     NO predicate at all, so it never prunes and gathers ALL shards — proving the counter reaches
        //     `shard_count` (it is not pinned to 1) and that pruning is precisely what cut the equality lookup
        //     to one shard.
        let before = gathered(&e);
        let cnt = sel("SELECT COUNT(*) FROM accounts");
        let scanned = gathered(&e) - before;
        assert_eq!(cnt.row(0), &[SqlValue::Int8(200)], "COUNT(*) across all shards");
        assert_eq!(
            scanned, shard_count as u64,
            "an unpruned COUNT(*) must gather ALL shards, proving the counter isn't pinned to 1"
        );

        // (4) every boundary + the MAX key (appended in place into the open shard) still reads correctly,
        //     each gathering exactly one shard -> no key is ever wrongly pruned.
        for id in [0_i64, 63, 64, 127, 128, 191, 199] {
            let before = gathered(&e);
            let r = sel(&format!("SELECT id, balance FROM accounts WHERE id = {id}"));
            let g = gathered(&e) - before;
            assert_eq!(r.len(), 1, "id={id} must be found (never wrongly pruned)");
            assert_eq!(
                r.row(0),
                &[SqlValue::Int4(id as i32), SqlValue::Int4((id * 10) as i32)],
                "id={id} value correct after pruning"
            );
            assert_eq!(g, 1, "id={id} prunes to exactly one shard (got {g})");
        }

        // (5) a key beyond every shard's range prunes to the keep-one fallback and returns empty.
        let before = gathered(&e);
        let miss = sel("SELECT id, balance FROM accounts WHERE id = 100000");
        let g = gathered(&e) - before;
        assert_eq!(miss.len(), 0, "absent key returns no rows");
        assert_eq!(g, 1, "an out-of-range needle prunes to the single keep-one fallback shard");
    }

    /// Read a shard's ON-DEMAND `deleted_by` region back from device (DtoH), first `count` slots. Returns
    /// `None` when the shard has NO region (delete-free). u64 reconstructed from i32 LE (lo, hi) pairs.
    fn read_shard_deleted_by_region(
        e: &Engine,
        table: &str,
        shard_id: u32,
        count: usize,
    ) -> Option<Vec<u64>> {
        let region = e
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), shard_id))?;
        let halves = region
            .read_resident_i32_column(0, count * 2)
            .expect("read deleted_by region");
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let lo = halves[2 * i] as u32 as u64;
            let hi = halves[2 * i + 1] as u32 as u64;
            out.push((hi << 32) | lo);
        }
        Some(out)
    }

    /// Test helper: does ANY shard of `table` currently hold a LIVE `deleted_by` region (cell present AND
    /// `Some`)? False after either `invalidate_table` (publishes `None`, device buffer freed, cell kept) or
    /// `remove_table` (cell dropped). Use to prove a region was RELEASED. Reads the published cell map
    /// directly (in-crate).
    fn table_has_any_deleted_by_cell(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_deleted_by_memory
            .cells
            .load()
            .iter()
            .any(|((cell_table, _), cell)| cell_table == table && cell.load().get().is_some())
    }

    /// Test helper: does ANY cell KEY for `table` still exist (regardless of `Some`/`None`)? Distinguishes
    /// `invalidate_table` (key KEPT as a `None` tombstone) from `remove_table` (key ERASED). Use to prove
    /// DROP fully removes the entry -- invalidate alone would leak a dangling `None` key per dropped table.
    fn table_has_any_deleted_by_key(e: &Engine, table: &str) -> bool {
        e.read_state
            .residency
            .shard_deleted_by_memory
            .cells
            .load()
            .keys()
            .any(|(cell_table, _)| cell_table == table)
    }

    /// SV2 (sparse-versioning): a shard is born DELETE-FREE and carries NO `deleted_by` region — the HyPer
    /// "un-versioned rows pay nothing" property. Across admission + in-place append + rollover, NO shard has a
    /// tombstone region until a DELETE touches it (SV4). NON-VACUITY: the table is really multiple shards
    /// (rollover) and EVERY one has no region (a regression that eagerly allocated would fail this), while the
    /// reads are still correct (all 200 rows live).
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_born_delete_free_carries_no_region() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // small -> admission + in-place append + rollover all exercised
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let shard_ids: Vec<(u32, usize)> = {
            let shards = e.read_state.residency.shards.load();
            let table_shards = shards.get("accounts").unwrap();
            assert!(
                table_shards.len() >= 3,
                "need rollover into multiple shards (got {})",
                table_shards.len()
            );
            table_shards.iter().map(|s| (s.shard_id, s.row_count)).collect()
        };
        for (shard_id, row_count) in shard_ids {
            assert!(
                read_shard_deleted_by_region(&e, "accounts", shard_id, row_count).is_none(),
                "a delete-free shard {shard_id} must carry NO deleted_by region (zero version overhead)"
            );
        }
        assert_eq!(
            e.execute_relational_select_text("SELECT COUNT(*) FROM accounts")
                .unwrap()
                .rows
                .row(0),
            &[SqlValue::Int8(200)],
            "all rows live (no tombstones)"
        );
    }

    /// SV2 (incremental DELETE write): the tombstone primitive ALLOCATES the shard's `deleted_by` region on
    /// its FIRST delete (delete-free shards pay zero) + stamps `deleted_by[slot] = commit_seq` there,
    /// OUT-OF-LINE (row column bytes untouched — the read visibility filter is SV3, so at SV2 a tombstoned row
    /// STILL reads). NON-VACUITY: no region before the first delete; only the targeted slots flip; the region
    /// is REUSED (not re-allocated) on a second delete; a full scan still returns all 5 rows byte-intact.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_tombstone_allocates_region_and_stamps_out_of_line() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        e.execute_text(
            2,
            "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30),(4,40),(5,50)",
        )
        .unwrap();
        let shard_id = e.read_state.residency.shards.load().get("accounts").unwrap()[0].shard_id;

        // Before any delete: NO region (the zero-cost property).
        assert!(
            read_shard_deleted_by_region(&e, "accounts", shard_id, 5).is_none(),
            "a delete-free shard has no deleted_by region"
        );

        // First delete: allocates the region + stamps slots 1 and 3 with seq 777.
        assert!(
            e.tombstone_resident_shard_slots("accounts", shard_id, &[1, 3], 777),
            "first tombstone allocates the region + stamps"
        );
        let db = read_shard_deleted_by_region(&e, "accounts", shard_id, 5)
            .expect("region is allocated on the first delete");
        assert_eq!(
            db,
            vec![0x7F7F_7F7F_7F7F_7F7F, 777, 0x7F7F_7F7F_7F7F_7F7F, 777, 0x7F7F_7F7F_7F7F_7F7F],
            "only the targeted slots are stamped; other rows stay the live sentinel (0x7F7F.. = signed-safe)"
        );

        // Out-of-line: column bytes untouched -> (visibility unwired at SV2) a full scan still returns 5 rows.
        let rows = e
            .execute_relational_select_text("SELECT id, balance FROM accounts")
            .unwrap()
            .rows;
        assert_eq!(rows.len(), 5, "columns intact: all rows still read (visibility is SV3)");
        let mut seen: Vec<(i32, i32)> = (0..rows.len())
            .map(|i| match (&rows.row(i)[0], &rows.row(i)[1]) {
                (SqlValue::Int4(id), SqlValue::Int4(bal)) => (*id, *bal),
                other => panic!("unexpected row shape {other:?}"),
            })
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);

        // Second delete on the SAME shard REUSES the region (no re-alloc) and stamps another slot.
        assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[0], 888));
        assert_eq!(
            read_shard_deleted_by_region(&e, "accounts", shard_id, 5).unwrap(),
            vec![888, 777, 0x7F7F_7F7F_7F7F_7F7F, 777, 0x7F7F_7F7F_7F7F_7F7F],
            "the second delete reuses the region + preserves the earlier stamps"
        );

        // Bounds: an out-of-range slot is rejected.
        assert!(
            !e.tombstone_resident_shard_slots("accounts", shard_id, &[5], 777),
            "slot == row_count (headroom) must be rejected"
        );
    }

    /// SV4 prerequisite #1 (lifecycle/leak, audit-flagged): the on-demand `deleted_by` region a tombstone
    /// allocates MUST be released whenever the resident buffer it annotates is invalidated (an invalidating
    /// commit -> O(table) re-admit) or dropped -- otherwise a re-admit rebuilds the shard ALL-LIVE from the
    /// host store yet inherits a STALE tombstone region (wrong-results: rows wrongly hidden), and DROP TABLE
    /// leaks the tombstone device buffers. NON-VACUITY: the region is proven PRESENT first, then proven GONE
    /// after each lifecycle event, with an end-to-end read confirming the buffer really is fresh + all-live.
    /// Path B (DROP) is specific to the `apply_drop_table` `remove_table` erase (KEY absence). NOTE: Path A's
    /// DELETE fires BOTH the commit invalidate mirror AND the auto-admit re-admit `remove_table`, so the
    /// re-admit MASKS the invalidate mirror here -- `shard_deleted_by_region_released_by_invalidate_alone`
    /// (auto-admit OFF) isolates the serialized invalidate mirror; the sharded re-admit + eviction-cleanup
    /// mirrors have their own isolated gates.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_deleted_by_region_released_on_invalidate_and_drop() {
        // --- Path A: an invalidating commit (SQL DELETE -> invalidate + re-admit) releases the region ---
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30)")
            .unwrap();
        let shard_id = e.read_state.residency.shards.load().get("accounts").unwrap()[0].shard_id;
        // The SV2 primitive allocates the region on this first tombstone.
        assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[1], 777));
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "precondition: the tombstone allocated a live deleted_by region"
        );
        // A DELETE goes through invalidate + the O(table) re-admit today (the path SV4 will replace).
        e.execute_text(3, "DELETE FROM accounts WHERE id = 2").unwrap();
        assert!(
            !table_has_any_deleted_by_cell(&e, "accounts"),
            "invalidate/re-admit must release the stale deleted_by region (leak + wrong-results guard)"
        );
        // End-to-end: the DELETE really removed id=2, and the rebuilt buffer reads ALL-LIVE (no stale hide
        // from the released tombstone region) -- id=1 was tombstoned resident-only, so it must reappear.
        let rows = e
            .execute_relational_select_text("SELECT id FROM accounts")
            .unwrap()
            .rows;
        let mut ids: Vec<i32> = (0..rows.len())
            .map(|i| match &rows.row(i)[0] {
                SqlValue::Int4(id) => *id,
                other => panic!("unexpected row shape {other:?}"),
            })
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 3], "id=2 deleted; id=1 all-live again (stale tombstone released)");

        // --- Path B: DROP TABLE releases the region ---
        e.execute_text(4, "INSERT INTO accounts (id, balance) VALUES (7,70)")
            .unwrap();
        let shard_id2 = e.read_state.residency.shards.load().get("accounts").unwrap()[0].shard_id;
        assert!(e.tombstone_resident_shard_slots("accounts", shard_id2, &[0], 888));
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "precondition: the region is re-allocated on the post-re-admit shard"
        );
        e.execute_text(5, "DROP TABLE accounts").unwrap();
        // DROP fully ERASES the cell entries (invalidate alone would leave a dangling `None` key per dropped
        // table). Asserting KEY absence (not just Some absence) makes this specific to `remove_table`.
        assert!(
            !table_has_any_deleted_by_key(&e, "accounts"),
            "DROP TABLE must erase the deleted_by cell entries (no leaked per-table keys / device memory)"
        );
    }

    /// SV4 prereq #1 (round-2 audit P3 hardening): ISOLATE the serialized-commit `invalidate_table` mirror.
    /// With AUTO-ADMIT OFF, a DELETE invalidates residency but triggers NO re-admit, so the region is released
    /// SOLELY by the commit-path `invalidate_relational_residency_table` deleted_by mirror -- nothing masks it
    /// (unlike `..._on_invalidate_and_drop`, where the re-admit's `remove_table` would hide a deleted mirror).
    /// Sabotage: delete ONLY the serialized `shard_deleted_by_memory.invalidate_table` (engine_commit.rs) and
    /// this FAILS. This is the exact production DELETE path SV4 will build on.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_deleted_by_region_released_by_invalidate_alone() {
        let mut e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(false); // NO re-admit after the DELETE -> isolates the invalidate mirror
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30)")
            .unwrap();
        // Explicit shard-resident admit (auto-admit is off).
        e.populate_relational_residency_snapshot("accounts").unwrap();
        let shard_id = e.read_state.residency.shards.load().get("accounts").unwrap()[0].shard_id;
        assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[1], 777));
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "precondition: the tombstone allocated a live deleted_by region"
        );
        // DELETE invalidates residency; with auto-admit OFF nothing re-admits -> the serialized-commit
        // invalidate mirror is the ONLY thing that can release the region.
        e.execute_text(3, "DELETE FROM accounts WHERE id = 2").unwrap();
        assert!(
            !table_has_any_deleted_by_cell(&e, "accounts"),
            "the serialized-commit invalidate mirror must release the region even with no re-admit"
        );
    }

    /// SV4 prereq #1 (audit Finding 2): a WARMUP/REFRESH re-admit (`populate_relational_residency_snapshot`)
    /// reaches the SHARDED re-admit branch with NO preceding invalidate, so it must itself erase stale
    /// `deleted_by` regions -- else the fresh all-live shard 0 (reused shard_id) inherits the tombstone region
    /// and wrongly hides rows at SV4. NON-VACUITY: region proven present, then KEY-absent after the refresh.
    /// Sabotage: remove the sharded-branch `shard_deleted_by_memory.remove_table` and this FAILS.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_deleted_by_region_released_on_warmup_readmit() {
        let mut e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20),(3,30)")
            .unwrap();
        e.populate_relational_residency_snapshot("accounts").unwrap();
        let shard_id = e.read_state.residency.shards.load().get("accounts").unwrap()[0].shard_id;
        assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[1], 777));
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "precondition: the tombstone allocated a live deleted_by region"
        );
        // Warmup/refresh re-admit -- NO commit, so NO invalidate precedes it (the path Finding 2 patched).
        e.populate_relational_residency_snapshot("accounts").unwrap();
        assert!(
            !table_has_any_deleted_by_key(&e, "accounts"),
            "warmup re-admit (no preceding invalidate) must erase the stale deleted_by region"
        );
    }

    /// SV4 prereq #1 (audit Finding 1): `RelationalResidentCache::remove_table` (the BUDGET-EVICTION cleanup)
    /// must release the table's `deleted_by` regions. This is DEFENSIVE today: the eviction loop draws its
    /// candidates ONLY from the single-buffer `snapshots` map (engine_residency.rs, the `candidates` filter),
    /// and a region-bearing table is by construction SHARD-resident (removed from `snapshots`), so `remove_table`
    /// is currently only ever called on region-free tables. But it is the exact call a future shard-eviction
    /// will make, so we test the METHOD CONTRACT directly: a shard-resident, tombstoned table passed to
    /// `remove_table` has its region ERASED. NON-VACUITY: region present, then KEY-absent. Sabotage: remove the
    /// `shard_deleted_by_memory.remove_table` in `RelationalResidentCache::remove_table` and this FAILS.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn resident_cache_remove_table_releases_deleted_by_region() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1,10),(2,20)")
            .unwrap();
        let shard_id = e.read_state.residency.shards.load().get("accounts").unwrap()[0].shard_id;
        assert!(e.tombstone_resident_shard_slots("accounts", shard_id, &[0], 5));
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "precondition: the tombstone allocated a live deleted_by region"
        );
        // Invoke the cache eviction-cleanup method DIRECTLY -- the exact call the budget-eviction loop makes
        // (`cat.relational_resident_cache.remove_table(map_key, &residency, &route_telemetry)`).
        {
            let guard = e.ddl_catalog();
            guard.relational_resident_cache.remove_table(
                "accounts",
                &e.read_state.residency,
                &e.read_state.route_telemetry,
            );
        }
        assert!(
            !table_has_any_deleted_by_key(&e, "accounts"),
            "RelationalResidentCache::remove_table must erase the table's deleted_by regions (eviction cleanup)"
        );
    }

    /// SV3b (MVCC read visibility): once a shard is tombstoned (the SV2 primitive), the SHARDED read path
    /// HIDES the tombstoned rows -- the on-device predicate ANDs `deleted_by > read_txn_id` over a co-resident
    /// i64 `deleted_by` column gathered into the unified buffer (memset to the all-live sentinel, then the
    /// versioned shard's live prefix DtoD-copied over). Gate: a point lookup for a tombstoned key returns
    /// EMPTY; COUNT(*) drops by exactly the tombstone count; a LIVE neighbor in the SAME now-versioned shard
    /// still reads (the live sentinel passes the SIGNED compare -- guards the fill-vs-stamp boundary); a point
    /// lookup pruned to a DIFFERENT, un-tombstoned shard is byte-identical (no deleted_by region -> the `None`
    /// visibility path). This is the host-MVCC visibility semantics enforced ENTIRELY on the GPU.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shard_visibility_filter_hides_tombstoned_rows() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8; id=k sits in shard (k/64) at slot (k%64)
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;

        // Baseline (delete-free): every shard is un-versioned -> the read takes the `None` visibility path.
        assert_eq!(sel("SELECT id FROM accounts WHERE id = 0").len(), 1, "id=0 present pre-delete");
        assert_eq!(sel("SELECT COUNT(*) FROM accounts").row(0), &[SqlValue::Int8(200)]);

        // Tombstone id=0 (shard 0, slot 0) at a commit seq well below the read snapshot so
        // `deleted_by(=5) > read_txn_id` is FALSE and the row is hidden.
        let shard0 = e.read_state.residency.shards.load().get("accounts").unwrap()[0].shard_id;
        assert!(
            e.tombstone_resident_shard_slots("accounts", shard0, &[0], 5),
            "tombstone id=0 at slot 0"
        );

        // (1) the tombstoned key is now INVISIBLE to a point lookup (gathers the versioned shard 0).
        assert_eq!(
            sel("SELECT id FROM accounts WHERE id = 0").len(),
            0,
            "tombstoned id=0 hidden by the on-device visibility filter"
        );

        // (2) COUNT(*) over ALL shards drops by exactly one (deleted_by built for the whole unified buffer;
        //     the un-versioned shards' rows are the all-live memset fill).
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(199)],
            "COUNT reflects the single tombstone"
        );

        // (3) a LIVE neighbor in the SAME now-versioned shard still reads -- the live sentinel passes the
        //     visibility compare (guards the fill-vs-tombstone boundary + the signed-safe sentinel).
        let n1 = sel("SELECT id, balance FROM accounts WHERE id = 1");
        assert_eq!(n1.len(), 1, "live neighbor id=1 in the versioned shard still visible");
        assert_eq!(n1.row(0), &[SqlValue::Int4(1), SqlValue::Int4(10)]);

        // (4) a point lookup pruned to a DIFFERENT, un-tombstoned shard is unaffected (no region -> `None`).
        let far = sel("SELECT id, balance FROM accounts WHERE id = 137");
        assert_eq!(far.len(), 1, "un-tombstoned shard unaffected");
        assert_eq!(far.row(0), &[SqlValue::Int4(137), SqlValue::Int4(1370)]);
    }

    /// SV4 (GPU-native DELETE, locate+tombstone data-plane primitive): `try_tombstone_resident_delete`
    /// LOCATES a row by an int4-equality predicate (zone-map-pruned, per-shard) and stamps its `deleted_by`
    /// -- so the SV3b read HIDES exactly that row, with NO O(table) re-admit. This is the mechanism SV4b wires
    /// into the DELETE commit. NON-VACUITY / correctness of LOCATE: a wrong slot would hide the WRONG row, so
    /// the neighbor-still-visible + COUNT-drops-by-exactly-one + other-shard-untouched asserts fail unless
    /// locate returns the EXACT slot. Multi-shard (size 64) exercises zone-map pruning + cross-shard locate.
    /// Delete-free byte-identical is proven by the pre-delete COUNT + the untouched rows post-delete.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn resident_delete_locate_and_tombstone_hides_exactly_the_matched_row() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8; id=k in shard k/64 at slot k%64
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
        // Pre-delete: delete-free reads are byte-identical (all 200 rows live, none hidden).
        assert_eq!(sel("SELECT COUNT(*) FROM accounts").row(0), &[SqlValue::Int8(200)]);
        assert_eq!(sel("SELECT id FROM accounts WHERE id = 130").len(), 1, "id=130 present pre-delete");

        // Build the point predicate `id = 130` (id is catalog column 0) and DELETE it via the GPU primitive.
        let table = e.relational_catalog_table("accounts").unwrap();
        let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
        let pred = crate::engine_expr::ResidentExpr::Binary {
            op: crate::engine_expr::ResidentBinaryOp::Eq,
            lhs: Box::new(crate::engine_expr::ResidentExpr::Column(id_col)),
            rhs: Box::new(crate::engine_expr::ResidentExpr::Int4Literal(130)),
        };
        // commit_seq 5 (well below the read snapshot) -> `deleted_by(=5) > read_txn_id` is FALSE -> hidden.
        let n = e
            .try_tombstone_resident_delete(&table, &pred, 5)
            .expect("resident, prunable point delete succeeds");
        assert_eq!(n, 1, "exactly one resident row matched id=130");

        // The matched row is now hidden; its neighbors + other shards are UNTOUCHED (proves the RIGHT slot).
        assert_eq!(sel("SELECT id FROM accounts WHERE id = 130").len(), 0, "id=130 tombstoned -> hidden");
        assert_eq!(sel("SELECT id FROM accounts WHERE id = 129").len(), 1, "same-shard neighbor 129 still visible");
        assert_eq!(sel("SELECT id FROM accounts WHERE id = 131").len(), 1, "same-shard neighbor 131 still visible");
        assert_eq!(sel("SELECT id FROM accounts WHERE id = 5").len(), 1, "a row in a DIFFERENT shard untouched");
        assert_eq!(
            sel("SELECT COUNT(*) FROM accounts").row(0),
            &[SqlValue::Int8(199)],
            "COUNT drops by exactly the one tombstoned row"
        );
    }

    /// SV4b (GPU-native incremental DELETE, commit WIRING): with `resident_delete_tombstone_enabled` ON, a
    /// single-row SQL DELETE on a shard-resident table LOCATES + tombstones the row's slot IN PLACE (no
    /// O(table) re-admit) and the GPU read == host MVCC. NON-VACUITY: the deleted_by region EXISTING after
    /// the DELETE proves the tombstone route ran (a re-admit fallback rebuilds ALL-LIVE => NO region), while
    /// the flag-OFF control gives the IDENTICAL result via re-admit (NO region). A MULTI-ROW DELETE falls
    /// back to re-admit (region cleared) and is still correct -- the exact-count safety net.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn sv4b_sql_delete_tombstones_in_place_and_matches_host_mvcc() {
        let load = |e: &Engine| {
            e.set_shard_residency_enabled(true);
            e.set_auto_admit_on_commit(true);
            e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8
            e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)").unwrap();
            for i in 0..200_i64 {
                e.execute_text(
                    (i as u64) + 2,
                    &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
                )
                .unwrap();
            }
        };
        let count = |e: &Engine| match e
            .execute_relational_select_text("SELECT COUNT(*) FROM accounts")
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("unexpected COUNT shape {other:?}"),
        };
        let present = |e: &Engine, id: i64| {
            !e.execute_relational_select_text(&format!("SELECT id FROM accounts WHERE id = {id}"))
                .unwrap()
                .rows
                .is_empty()
        };

        // --- flag ON: the single-row DELETE routes through the in-place tombstone ---
        let e = Engine::new_local();
        e.set_resident_delete_tombstone_enabled(true);
        load(&e);
        assert!(!table_has_any_deleted_by_cell(&e, "accounts"), "delete-free: no region");
        assert_eq!(count(&e), 200);

        e.execute_text(202, "DELETE FROM accounts WHERE id = 130").unwrap();
        // NON-VACUITY: the tombstone path ran (region allocated). A re-admit fallback would leave NO region.
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "single-row DELETE routed through the in-place tombstone (region allocated)"
        );
        assert!(!present(&e, 130), "id=130 deleted -> hidden on the GPU route");
        assert!(present(&e, 129) && present(&e, 131), "same-shard neighbors still visible");
        assert!(present(&e, 5), "a row in a different shard untouched");
        assert_eq!(count(&e), 199, "COUNT drops by exactly one (== host MVCC)");

        // A MULTI-ROW DELETE (2 rows) is NOT yet incremental -> exact-count gate returns false -> re-admit,
        // which rebuilds all-live (region CLEARED by prereq #1) and is still correct.
        e.execute_text(203, "DELETE FROM accounts WHERE id = 50 OR id = 51").unwrap();
        assert!(
            !table_has_any_deleted_by_cell(&e, "accounts"),
            "multi-row DELETE fell back to re-admit (all-live rebuild -> no region)"
        );
        assert!(!present(&e, 50) && !present(&e, 51), "multi-row DELETE removed both rows");
        assert!(!present(&e, 130), "the earlier single-row delete stays deleted (host store)");
        assert_eq!(count(&e), 197, "COUNT == host MVCC after 3 total deletes");

        // --- flag OFF control: the SAME single-row DELETE via re-admit -> identical result, NO region ---
        let c = Engine::new_local(); // resident_delete_tombstone_enabled stays default OFF
        load(&c);
        c.execute_text(202, "DELETE FROM accounts WHERE id = 130").unwrap();
        assert!(
            !table_has_any_deleted_by_cell(&c, "accounts"),
            "flag OFF: DELETE re-admits (all-live) -> no region"
        );
        assert!(!present(&c, 130), "control: id=130 deleted");
        assert_eq!(count(&c), 199, "control: COUNT 199 == the flag-ON result (byte-identical semantics)");
    }

    /// SV5 (GPU-native incremental UPDATE, commit WIRING): with `resident_update_tombstone_enabled` ON, a
    /// single-row SQL UPDATE on a shard-resident table TOMBSTONES the old version's slot + APPENDS the new
    /// image IN PLACE (no O(table) re-admit) and the GPU read == host MVCC. NON-VACUITY: the deleted_by region
    /// EXISTING after the UPDATE proves the tombstone-old route ran (re-admit fallback rebuilds ALL-LIVE => NO
    /// region); the read returns the NEW value; COUNT is unchanged (old hidden + new visible); the OLD value is
    /// hidden; a MULTI-ROW UPDATE falls back to re-admit (correct); flag-OFF control identical.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn sv5_sql_update_tombstones_old_appends_new_matches_host_mvcc() {
        let load = |e: &Engine| {
            e.set_shard_residency_enabled(true);
            e.set_auto_admit_on_commit(true);
            e.set_shard_size_target(64);
            e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)").unwrap();
            for i in 0..200_i64 {
                e.execute_text(
                    (i as u64) + 2,
                    &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
                )
                .unwrap();
            }
        };
        let count = |e: &Engine| match e
            .execute_relational_select_text("SELECT COUNT(*) FROM accounts")
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("unexpected COUNT shape {other:?}"),
        };
        let balance_of = |e: &Engine, id: i64| -> Option<i32> {
            let rows = e
                .execute_relational_select_text(&format!(
                    "SELECT id, balance FROM accounts WHERE id = {id}"
                ))
                .unwrap()
                .rows;
            if rows.is_empty() {
                return None;
            }
            match &rows.row(0)[1] {
                SqlValue::Int4(b) => Some(*b),
                other => panic!("unexpected row shape {other:?}"),
            }
        };
        let old_balance_visible = |e: &Engine| {
            // The OLD (id=130, balance=1300) image must be HIDDEN: a lookup by the old balance finds nothing.
            !e.execute_relational_select_text("SELECT id FROM accounts WHERE balance = 1300")
                .unwrap()
                .rows
                .is_empty()
        };

        // --- flag ON: the single-row UPDATE routes through tombstone-old + append-new ---
        let e = Engine::new_local();
        e.set_resident_update_tombstone_enabled(true);
        load(&e);
        assert!(!table_has_any_deleted_by_cell(&e, "accounts"), "no region pre-update");
        assert_eq!(count(&e), 200);
        assert_eq!(balance_of(&e, 130), Some(1300), "pre-update balance");

        e.execute_text(202, "UPDATE accounts SET balance = 9999 WHERE id = 130").unwrap();
        // NON-VACUITY: the tombstone-old path ran (region allocated). Re-admit fallback would leave NO region.
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "single-row UPDATE routed through tombstone-old + append-new (region allocated)"
        );
        assert_eq!(balance_of(&e, 130), Some(9999), "id=130 reads the NEW balance (appended version)");
        assert!(!old_balance_visible(&e), "the OLD (id=130,balance=1300) image is hidden");
        assert_eq!(balance_of(&e, 131), Some(1310), "same-shard neighbor untouched");
        assert_eq!(balance_of(&e, 5), Some(50), "a row in a different shard untouched");
        assert_eq!(count(&e), 200, "COUNT unchanged (old hidden + new visible) == host MVCC");

        // An int4-UNCHANGED update (same-value: id=5 already has balance 5*10=50) still routes: tombstone-OLD
        // FIRST locates the old slot on the buffer BEFORE the identical-int4 new row is appended (count 1), so
        // it tombstones the OLD slot, not the new. Exercises the order-sensitivity the value-changing case can't.
        e.execute_text(203, "UPDATE accounts SET balance = 50 WHERE id = 5").unwrap();
        assert!(
            table_has_any_deleted_by_cell(&e, "accounts"),
            "same-value UPDATE still routes through tombstone-old + append-new"
        );
        assert_eq!(balance_of(&e, 5), Some(50), "id=5 still reads 50 (old hidden, new appended, same value)");
        assert_eq!(count(&e), 200, "COUNT unchanged after the int4-unchanged update");

        // A MULTI-ROW UPDATE (2 rows) is NOT yet incremental -> re-admit (all-live rebuild -> no region),
        // still correct; the earlier single-row update persists (host store rebuilt).
        e.execute_text(204, "UPDATE accounts SET balance = 0 WHERE id = 10 OR id = 11").unwrap();
        assert!(
            !table_has_any_deleted_by_cell(&e, "accounts"),
            "multi-row UPDATE fell back to re-admit (no region)"
        );
        assert_eq!(balance_of(&e, 10), Some(0));
        assert_eq!(balance_of(&e, 11), Some(0));
        assert_eq!(balance_of(&e, 130), Some(9999), "single-row update persists across the re-admit");
        assert_eq!(count(&e), 200);

        // --- flag OFF control: the SAME single-row UPDATE via re-admit -> identical result, NO region ---
        let c = Engine::new_local(); // resident_update_tombstone_enabled stays default OFF
        load(&c);
        c.execute_text(202, "UPDATE accounts SET balance = 9999 WHERE id = 130").unwrap();
        assert!(
            !table_has_any_deleted_by_cell(&c, "accounts"),
            "flag OFF: UPDATE re-admits (all-live) -> no region"
        );
        assert_eq!(balance_of(&c, 130), Some(9999), "control: new balance");
        assert_eq!(count(&c), 200, "control: COUNT 200 == the flag-ON result (byte-identical semantics)");
    }

    /// CROSS-SHARD PK INDEX sub-slice 1: the per-shard hash-index locate returns the IDENTICAL physical
    /// (shard, LOCAL slot) the scan-based locate finds -- present keys across shards, absent keys (empty),
    /// NULL-as-0 (id 0), and it DECLINES (None -> scan fallback) on a duplicate key. ORACLE = the proven SV4a
    /// scan-based `locate_resident_delete_slots` (an INDEPENDENT mechanism: hash-probe vs scan-predicate, so
    /// agreement is strong). NON-VACUITY: a wrong slot / missed shard / wrong decline diverges from the oracle;
    /// a cross-check confirms exactly one hit per unique key. Multi-shard (size 64) exercises per-shard build.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn cross_shard_pk_index_locate_matches_scan_locate() {
        let e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.set_auto_admit_on_commit(true);
        e.set_shard_size_target(64); // 200 rows -> shards 64,64,64,8
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)").unwrap();
        for i in 0..200_i64 {
            e.execute_text(
                (i as u64) + 2,
                &format!("INSERT INTO accounts (id, balance) VALUES ({i}, {})", i * 10),
            )
            .unwrap();
        }
        let table = e.relational_catalog_table("accounts").unwrap();
        let id_col = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
        // Oracle: SV4a scan-based locate for `id = k`, flattened + sorted to (shard, slot).
        let scan_locate = |k: i32| -> Vec<(u32, u32)> {
            let pred = crate::engine_expr::ResidentExpr::Binary {
                op: crate::engine_expr::ResidentBinaryOp::Eq,
                lhs: Box::new(crate::engine_expr::ResidentExpr::Column(id_col)),
                rhs: Box::new(crate::engine_expr::ResidentExpr::Int4Literal(k)),
            };
            let mut v: Vec<(u32, u32)> = e
                .locate_resident_delete_slots(&table, &pred)
                .unwrap()
                .into_iter()
                .flat_map(|(shard, slots)| slots.into_iter().map(move |s| (shard, s)))
                .collect();
            v.sort_unstable();
            v
        };
        // Present keys across multiple shards: index locate == scan locate, exactly one hit each.
        for k in [0_i32, 5, 63, 64, 130, 199] {
            let mut idx = e
                .locate_resident_pk_via_shard_index(&table, id_col, k)
                .expect("resident + unique -> Some");
            idx.sort_unstable();
            assert_eq!(idx, scan_locate(k), "index locate == scan locate for id={k}");
            assert_eq!(idx.len(), 1, "unique key id={k} -> exactly one (shard,slot) hit");
        }
        // Absent key: both empty.
        let mut absent = e
            .locate_resident_pk_via_shard_index(&table, id_col, 999)
            .expect("resident -> Some(empty)");
        absent.sort_unstable();
        assert_eq!(absent, scan_locate(999));
        assert!(absent.is_empty(), "absent key -> no hit");

        // DUP-DECLINE: a table with a duplicate int4 key -> the hash build declines -> None (scan fallback),
        // because a hash holds one row/key but the scan returns EVERY match.
        let d = Engine::new_local();
        d.set_shard_residency_enabled(true);
        d.set_auto_admit_on_commit(true);
        d.execute_text(1, "CREATE TABLE dup (id INT, balance INT)").unwrap();
        d.execute_text(2, "INSERT INTO dup (id, balance) VALUES (1,10),(1,20),(2,30)")
            .unwrap();
        let dtable = d.relational_catalog_table("dup").unwrap();
        let did = crate::rel_exec_helpers::relational_column_index(&dtable, "id").unwrap();
        assert!(
            d.locate_resident_pk_via_shard_index(&dtable, did, 1).is_none(),
            "duplicate key -> hash declines -> None (caller falls back to the scan)"
        );
    }

    /// `capacity > row_count` pads each i32 section to `capacity` (real values then zero headroom);
    /// the header still records `row_count`; section offsets derive from `capacity`.
    #[test]
    fn capacity_padding_reserves_headroom_with_capacity_offsets() {
        let (names, types) = int4_cols();
        let rows = int4_rows(3);
        let capacity = 8;
        let payload = build_relational_device_payload_with_capacity(&names, &types, &rows, capacity)
            .unwrap()
            .0;
        // Layout: 8-byte header, int4 col0 (capacity*4), int4 col1 (capacity*4).
        assert_eq!(payload.len(), 8 + 2 * capacity * 4);
        let header = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        assert_eq!(header, 3, "header records the live row count, not capacity");
        let at = |sec: usize, row: usize| -> i32 {
            let off = 8 + sec * capacity * 4 + row * 4;
            i32::from_le_bytes(payload[off..off + 4].try_into().unwrap())
        };
        // col0 (id): 0,1,2 then zero headroom.
        assert_eq!((at(0, 0), at(0, 2), at(0, 3), at(0, 7)), (0, 2, 0, 0));
        // col1 (balance): 0,10,20 then zero headroom.
        assert_eq!((at(1, 0), at(1, 2), at(1, 3)), (0, 20, 0));
    }

    /// An OPEN (capacity > row_count) payload rejects variable-length text; `capacity == row_count`
    /// (dense) still accepts it. And `capacity < row_count` is always rejected.
    #[test]
    fn open_payload_rejects_text_and_undersize() {
        let names = vec!["id".to_string(), "name".to_string()];
        let types = vec![SqlType::Int4, SqlType::Text];
        let rows = vec![vec![SqlValue::Int4(1), SqlValue::Text("a".to_string())]];
        assert!(build_relational_device_payload_with_capacity(&names, &types, &rows, 1).is_ok());
        assert!(build_relational_device_payload_with_capacity(&names, &types, &rows, 4).is_err());
        let (names, types) = int4_cols();
        let rows = int4_rows(5);
        assert!(build_relational_device_payload_with_capacity(&names, &types, &rows, 3).is_err());
    }
}

impl Engine {
    pub fn populate_relational_residency_snapshot(
        &mut self,
        table: &str,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        self.populate_relational_residency_snapshot_on_gpu(table, gpu_id)
    }

    fn populate_relational_residency_snapshot_inner(
        &self,
        cat: &mut DdlCatalogState,
        table: &str,
        gpu_id: u16,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let catalog_table = cat
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        let prefix = relational_key_prefix(table);
        let mut row_count = 0usize;
        let mut resident_bytes = 0u64;
        let mut resident_rows = Vec::new();
        let mut raw_device_tail = Vec::new();
        {
            let table_rows = self.read_state.mvcc.table_rows(table);
            let mut cursor = table_rows.store().seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                raw_device_tail.extend_from_slice(tuple.key.as_bytes());
                raw_device_tail.extend_from_slice(tuple.value.as_bytes());
                let decoded = decode_relational_row(&tuple.value, &catalog_table.columns)?;
                row_count += 1;
                resident_bytes = resident_bytes
                    .saturating_add(tuple.key.len() as u64)
                    .saturating_add(
                        decoded
                            .iter()
                            .map(relational_resident_value_bytes)
                            .sum::<u64>(),
                    );
                resident_rows.push(decoded);
            }
        }
        // int4 AND date columns share the i32 section: a `date` is physically an i32 (days since
        // 2000-01-01), so it rides the int4 residency layout + the i32 compare kernels (the type
        // matrix, doc 19). The catalog type distinguishes them for lowering/projection.
        // int4, date AND int2 share the i32 section: a `date` is i32 days and a `smallint` widens to
        // i32, so both ride the int4 residency layout + compare path (the type matrix, doc 19).
        let resident_device_int4_columns = catalog_table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        // int8 AND timestamp columns share the i64 section: a `timestamp` is i64 microseconds, so it
        // rides the int8 residency layout + the i64 compare kernels (the type matrix, doc 19).
        let resident_device_int8_columns = catalog_table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = catalog_table
            .columns
            .iter()
            // numeric AND uuid share the 16-byte section: a `uuid` is 16 raw bytes (a byte-wise
            // compare kernel reads them; numeric stores its i128 mantissa). The type matrix, doc 19.
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let column_names: Vec<String> = catalog_table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_types: Vec<SqlType> =
            catalog_table.columns.iter().map(|column| column.ty).collect();
        // Slice 1b-ii: a PURELY-int4 table is laid down as an OPEN shard with capacity headroom (~2x
        // rows, power-of-two) so committed INSERTs append in place (amortized O(1)/row) instead of
        // re-uploading the whole table every commit. Other shapes (and huge / empty tables) stay dense.
        let purely_int4 = row_count > 0
            && row_count < (1usize << 29)
            && !column_types.is_empty()
            && column_types
                .iter()
                .all(|ty| matches!(ty, SqlType::Int4 | SqlType::Date | SqlType::Int2));
        // A PURELY-int4 table is laid down with capacity HEADROOM (~2x rows, power-of-two) so committed
        // INSERTs append in place (1b-ii). S-d2: the sharded read is now capacity-aware (the recompaction
        // gather + `resident_snapshot_for_shard` stride by `shard.capacity`), so the OPEN shard gets the
        // same headroom as the single buffer. Other shapes (and huge/empty tables) stay dense.
        let capacity = if purely_int4 {
            let doubled = row_count.saturating_mul(2).next_power_of_two();
            // S-d2c: on the shard path, CAP the open shard at the target size (`row_count` if it already
            // exceeds it — a large admit is one dense shard) so it seals + rolls over at the target rather
            // than growing unbounded. The single buffer is uncapped (its cap is the 536M guard above).
            if self.shard_residency_enabled() {
                doubled.min(self.shard_size_target()).max(row_count)
            } else {
                doubled
            }
        } else {
            row_count
        };
        let (
            mut device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload_with_capacity(
            &column_names,
            &column_types,
            &resident_rows,
            capacity,
        )?;
        // The dead MVCC tail rides only the dense (sealed) payload; an OPEN shard (capacity > row_count)
        // omits it (no kernel reads it) so the section headroom an append writes into stays clean.
        if capacity == row_count {
            device_payload.extend_from_slice(&raw_device_tail);
        }
        // SV1/SV2 (sparse-versioning): NO version metadata rides the shard payload. `created_by` is gone
        // (SV1 — returns as a zone map + boundary under SI), and `deleted_by` is now ON-DEMAND: a shard is
        // born delete-free with NO tombstone region; its `deleted_by` region (a separate device buffer in
        // `shard_deleted_by_memory`) is allocated on the shard's FIRST delete. So a delete-free / cold shard
        // pays ZERO version overhead (the HyPer property). See docs/proposals/sparse-mvcc-version-metadata.md.

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let admission_budget_bytes = cat.relational_resident_cache.budget_bytes_by_gpu.get(&gpu_id).copied();
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot_inner(cat, table, gpu_id, resident_bytes)?;
        let device_memory = self.relational_residency_device_memory(gpu_id, &device_payload);
        let device_memory_proof = device_memory
            .as_ref()
            .map(|device_memory| device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            // headroom for an open shard (== row_count for the dense/sealed path); Slice 1b-ii.
            capacity,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: self.committed_seq(),
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        // Billions-of-rows segmented layout (S-d1, default-OFF `shard_residency_enabled`): admit as a
        // SEGMENTED shard list — ONE dense shard for now; seal/rollover into many shards is S-d2 — routed
        // through the sharded resident read path, instead of the single capacity-padded unified buffer
        // (which caps at ~536M rows and re-admits O(table)). The single dense shard reuses the SAME columnar
        // payload + layout the single buffer uses (header at offset 0, dense columns), so the (already
        // tested) sharded read path reads it identically. Requires GPU device memory; without it (no GPU)
        // we fall through to the single-buffer/host path. Append/seal/per-shard-index land in S-d2/S-d3.
        if self.shard_residency_enabled() && device_memory.is_some() {
            // Audit (S-d1) fix: this re-admit makes the SHARD representation authoritative — clear any prior
            // single-buffer cell for the table so a runtime flag flip (OFF->ON) cannot leave a stale
            // snapshot/device_memory shadowing the shards. Idempotent (a no-op when none exists).
            read_state.residency.with_snapshots_mut(|snapshots| {
                snapshots.remove(table);
            });
            read_state.residency.device_memory.remove(table);
            // SV4 prereq #1 (lifecycle): this SHARDED re-admit installs a FRESH all-live shard 0, but a
            // warmup/refresh (`populate_relational_residency_snapshot_on_gpu`) reaches here with NO preceding
            // invalidate -- so erase any stale `deleted_by` regions for the table (keyed by the reused
            // shard_id) or the fresh shard would inherit them (SV4 wrong-results). Symmetric to the
            // single-buffer path below. INERT until SV4 (no region exists today).
            read_state.residency.shard_deleted_by_memory.remove_table(table);
            let dm = device_memory.expect("device_memory.is_some() checked");
            let shard = RelationalResidentShard {
                shard_id: 0,
                row_start: 0,
                row_count,
                // S-d2: the OPEN shard carries headroom (capacity > row_count for int4); the recompaction
                // gather + offset helpers stride by this capacity. (Dead MVCC tail omitted when padded.)
                capacity,
                // S-d2b: append-eligible iff purely int4 (no text / int8 / numeric / bool / NULL sections)
                // AND it has headroom — the SAME eligibility the single-buffer append checks.
                int4_appendable: snapshot.resident_device_int8_columns.is_empty()
                    && snapshot.resident_device_numeric_columns.is_empty()
                    && snapshot.resident_device_bool_columns.is_empty()
                    && snapshot.resident_device_text_columns.is_empty()
                    && snapshot.resident_device_null_columns.is_empty()
                    && snapshot.column_count == snapshot.resident_device_int4_columns.len()
                    && capacity > row_count,
                // S-d3: the zone map (min/max per int4 column) for shard pruning.
                resident_device_int4_column_stats: snapshot
                    .resident_device_int4_column_stats
                    .clone(),
                resident_bytes,
                allocated_bytes: device_payload.len() as u64,
                count_header_byte_offset: 0,
                resident_device_int4_columns: snapshot.resident_device_int4_columns.clone(),
                resident_device_text_columns: snapshot.resident_device_text_columns.clone(),
                gpu_id,
                schema: snapshot.schema.clone(),
                table: snapshot.table.clone(),
                device_memory_proof: snapshot.device_memory_proof.clone(),
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: memory_pressure_active,
                memory_pressure_active,
            };
            let mut shard_memory = BTreeMap::new();
            shard_memory.insert(0_u32, dm);
            cat.relational_resident_cache.install_shards(
                catalog_table.name,
                vec![shard],
                shard_memory,
                &read_state.residency,
            );
            return Ok(snapshot);
        }
        // Audit (S-d1) fix: the single-buffer path is authoritative here — clear any prior SHARD cell for
        // the table so a flag flip (ON->OFF) cannot leave a stale shard shadowing the fresh snapshot (the
        // read route checks shards FIRST, so a still-valid stale shard would serve wrong rows). Idempotent.
        read_state.residency.with_shards_mut(|shards| {
            shards.remove(table);
        });
        read_state.residency.shard_device_memory.remove_table(table);
        // SV4 prereq #1 (lifecycle): the single-buffer path replaces the table's shards, so clear any stale
        // `deleted_by` regions -- a flag flip / re-admit must not leave a tombstone region shadowing the fresh
        // all-live buffer (wrong-results guard). INERT until SV4 (no region exists today).
        read_state.residency.shard_deleted_by_memory.remove_table(table);
        cat
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                resident_rows,
                device_memory,
                &read_state.residency,
            );
        Ok(snapshot)
    }

    fn admit_relational_residency_snapshot_inner(
        &self,
        cat: &mut DdlCatalogState,
        table: &str,
        gpu_id: u16,
        resident_bytes: u64,
    ) -> Result<(Vec<String>, u64), ExecuteError> {
        let Some(budget_bytes) = cat.relational_resident_cache.budget_bytes_by_gpu.get(&gpu_id).copied() else {
            let resident_bytes_after_admission = self
                .relational_resident_bytes_for_gpu_excluding(gpu_id, table)
                .saturating_add(resident_bytes);
            cat
                .relational_resident_cache
                .record_decision(RelationalResidentCacheDecision {
                    table: table.to_string(),
                    gpu_id,
                    accepted: true,
                    reason: "admitted without budget limit".to_string(),
                    resident_bytes,
                    budget_bytes: None,
                    current_bytes_before: resident_bytes_after_admission
                        .saturating_sub(resident_bytes),
                    current_bytes_after: resident_bytes_after_admission,
                    evicted_tables: Vec::new(),
                });
            return Ok((Vec::new(), resident_bytes_after_admission));
        };
        if resident_bytes > budget_bytes {
            let current_bytes = self.relational_resident_bytes_for_gpu(gpu_id);
            cat
                .relational_resident_cache
                .record_decision(RelationalResidentCacheDecision {
                    table: table.to_string(),
                    gpu_id,
                    accepted: false,
                    reason: "resident snapshot exceeds GPU budget".to_string(),
                    resident_bytes,
                    budget_bytes: Some(budget_bytes),
                    current_bytes_before: current_bytes,
                    current_bytes_after: current_bytes,
                    evicted_tables: Vec::new(),
                });
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{table}\" resident snapshot requires {resident_bytes} bytes, exceeding GPU {gpu_id} residency budget {budget_bytes} bytes"
            ))));
        }

        let mut current_bytes = self.relational_resident_bytes_for_gpu_excluding(gpu_id, table);
        let current_bytes_before = current_bytes;
        let mut evicted_tables = Vec::new();
        if current_bytes.saturating_add(resident_bytes) <= budget_bytes {
            cat
                .relational_resident_cache
                .record_decision(RelationalResidentCacheDecision {
                    table: table.to_string(),
                    gpu_id,
                    accepted: true,
                    reason: "admitted within budget".to_string(),
                    resident_bytes,
                    budget_bytes: Some(budget_bytes),
                    current_bytes_before,
                    current_bytes_after: current_bytes + resident_bytes,
                    evicted_tables: Vec::new(),
                });
            return Ok((evicted_tables, current_bytes + resident_bytes));
        }

        let mut candidates = self
            .read_state
            .residency
            .snapshots
            .load()
            .iter()
            .filter(|(name, entry)| name.as_str() != table && entry.descriptor.gpu_id == gpu_id)
            .map(|(name, entry)| {
                (
                    entry.descriptor.valid_through_index,
                    entry.descriptor.table.clone(),
                    name.clone(),
                    entry.descriptor.resident_bytes,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort();
        for (_valid_through_index, _snapshot_table, map_key, bytes) in candidates {
            if current_bytes.saturating_add(resident_bytes) <= budget_bytes {
                break;
            }
            let read_state = Arc::clone(&self.read_state);
            cat
                .relational_resident_cache
                .remove_table(&map_key, &read_state.residency, &read_state.route_telemetry);
            current_bytes = current_bytes.saturating_sub(bytes);
            evicted_tables.push(map_key);
        }

        cat
            .relational_resident_cache
            .record_decision(RelationalResidentCacheDecision {
                table: table.to_string(),
                gpu_id,
                accepted: true,
                reason: if evicted_tables.is_empty() {
                    "admitted within budget".to_string()
                } else {
                    "admitted after deterministic eviction".to_string()
                },
                resident_bytes,
                budget_bytes: Some(budget_bytes),
                current_bytes_before,
                current_bytes_after: current_bytes + resident_bytes,
                evicted_tables: evicted_tables.clone(),
            });
        Ok((evicted_tables, current_bytes + resident_bytes))
    }

    /// `&mut self` entry for the operator warm path: acquire the catalog latch, then run the
    /// `&self`+held-guard producer (the STRATA S-B seam — also reachable from the `&self` commit path).
    fn populate_relational_residency_snapshot_on_gpu(
        &mut self,
        table: &str,
        gpu_id: u16,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let mut guard = self.ddl_catalog();
        self.populate_relational_residency_snapshot_inner(&mut guard, table, gpu_id)
    }

    /// `&mut self` entry for the benchmark residency installers.
    fn admit_relational_residency_snapshot(
        &mut self,
        table: &str,
        gpu_id: u16,
        resident_bytes: u64,
    ) -> Result<(Vec<String>, u64), ExecuteError> {
        let mut guard = self.ddl_catalog();
        self.admit_relational_residency_snapshot_inner(&mut guard, table, gpu_id, resident_bytes)
    }

    /// STRATA S-B: enable/disable automatic GPU-residency admission on commit. Default OFF — turning it
    /// on makes a committed table GPU-resident so subsequent reads take the GPU-native route instead of
    /// the host path. `&self` (an interior-mutable flag the commit path reads).
    pub fn set_auto_admit_on_commit(&self, on: bool) {
        self.auto_admit_on_commit
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn auto_admit_on_commit_enabled(&self) -> bool {
        self.auto_admit_on_commit
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// ADR-009 R1: enable/disable the GPU index-probe point-lookup route (default OFF). When on, a
    /// resident int4 unique-key equality batch probes a cached GPU hash index instead of full-scanning;
    /// non-unique columns / un-buildable indexes transparently fall back to the scan. `&self` (an
    /// interior-mutable flag the read path reads).
    pub fn set_index_probe_enabled(&self, on: bool) {
        self.index_probe_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn index_probe_enabled(&self) -> bool {
        self.index_probe_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// DECISIONS "lpb read levers" #1: route the lpb unique index probe through the DENSE-emit kernel. Default
    /// off; byte-identical to the atomic kernel. `&self` (interior-mutable flag the read path reads).
    pub fn set_dense_index_probe_enabled(&self, on: bool) {
        self.dense_index_probe_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn dense_index_probe_enabled(&self) -> bool {
        self.dense_index_probe_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Billions-of-rows segmented layout (S-d1): enable/disable admitting a table as a SEGMENTED shard list
    /// (routed through the sharded resident read path) instead of one capacity-padded unified buffer.
    /// DEFAULT OFF — the A/B lever to validate the shard path before flipping the default. Interior-mutable.
    pub fn set_shard_residency_enabled(&self, on: bool) {
        self.shard_residency_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn shard_residency_enabled(&self) -> bool {
        self.shard_residency_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// SV4b: enable GPU-native incremental DELETE — a single-entry DELETE commit locates + tombstones the
    /// deleted rows' resident slots IN PLACE (O(rows)) instead of the O(table) invalidate + re-admit. DEFAULT
    /// OFF (nested under the shard path); OFF => a DELETE re-admits exactly as before (byte-identical). The
    /// A/B lever for the incremental-DELETE win. Interior-mutable (the commit path reads it).
    pub fn set_resident_delete_tombstone_enabled(&self, on: bool) {
        self.resident_delete_tombstone_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn resident_delete_tombstone_enabled(&self) -> bool {
        self.resident_delete_tombstone_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// SV5: enable GPU-native incremental UPDATE — a single-entry UPDATE commit tombstones the old version's
    /// resident slot + appends the new image to the open shard IN PLACE (O(rows)) instead of the O(table)
    /// invalidate + re-admit. DEFAULT OFF (nested under the shard path); OFF => an UPDATE re-admits exactly as
    /// before (byte-identical). The A/B lever for the incremental-UPDATE win. Interior-mutable.
    ///
    /// **DO NOT FLIP ON in production yet (audit P2):** the appended new version has no `created_by`
    /// lower-bound gate, so a concurrent reader at `committed_seq = C-1` (pre-publish torn read) sees the
    /// updated key TWICE. Gated on the `created_by` boundary / consistent-snapshot fix + a concurrent-reader
    /// test. See `Engine::try_update_resident_commit`'s SI CAVEAT.
    pub fn set_resident_update_tombstone_enabled(&self, on: bool) {
        self.resident_update_tombstone_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn resident_update_tombstone_enabled(&self) -> bool {
        self.resident_update_tombstone_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// S-d2c: set the target row count per shard (the rollover/seal threshold). Settable small in tests.
    pub fn set_shard_size_target(&self, rows: usize) {
        self.shard_size_target
            .store(rows.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn shard_size_target(&self) -> usize {
        self.shard_size_target
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// DECISIONS "lpb read levers" #1: count of batches served by the DENSE-emit index probe. The test signal
    /// that the dense route actually ran (dense and atomic are byte-identical, so output equality can't prove
    /// which kernel produced the rows).
    pub fn dense_index_probe_hits(&self) -> u64 {
        self.read_state
            .residency
            .dense_index_probe_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Slice 1b-ii-c: count of commits served by the IN-PLACE open-shard append (vs a whole-table
    /// re-admit). The test/telemetry signal that the append actually fired — output equality and even
    /// device-ptr stability can't prove it (a same-size re-admit reuses the freed address).
    pub fn open_shard_append_hits(&self) -> u64 {
        self.read_state
            .residency
            .open_shard_append_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// S-d3: count of shards actually GATHERED (recompacted) by the sharded read after zone-map pruning.
    /// The non-vacuity signal that pruning fired — output equality can't prove a shard was skipped, since
    /// a pruned shard holds no matching rows and the result is identical either way.
    pub fn sharded_shards_gathered(&self) -> u64 {
        self.read_state
            .residency
            .sharded_shards_gathered
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The number of resident shards a table currently holds (0 if not shard-resident). Read-only
    /// telemetry for benchmarks/tests that assert a table really grew into N bounded shards (else a
    /// "flat latency vs shard count" claim could be vacuously true on a silently single-shard table).
    pub fn resident_shard_count(&self, table: &str) -> usize {
        self.read_state
            .residency
            .shards
            .load()
            .get(table)
            .map_or(0, |shards| shards.len())
    }

    /// Slice 1b-ii: append an INSERT's APPLIED rows IN PLACE into the table's resident OPEN shard's
    /// capacity headroom, instead of a full re-admit. `new_rows` MUST be the post-coercion/post-default
    /// applied images (the WriteDelta's `PreparedMutation::Insert.inserted_rows`), in catalog order, so
    /// the appended bytes match what a full rebuild would store. Returns `true` iff it appended +
    /// republished; `false` (the caller MUST fall back to invalidate + re-admit) when the table is not
    /// purely-int4-resident, is invalidated, lacks headroom, or the device append fails. MUST run BEFORE
    /// `committed_seq` is published, so a reader at the new commit observes the advanced `row_count`
    /// (visibility ordering — same placement as the invalidation it replaces).
    pub(crate) fn try_append_resident_int4_open_shard(
        &self,
        table: &str,
        new_rows: &[Vec<SqlValue>],
    ) -> bool {
        if new_rows.is_empty() {
            return false;
        }
        // Audit DO-NOT-SHIP fix: a NULL in an appended row would need a validity bitmap, but the open
        // shard this path appends into is bitmap-free by construction (the eligibility check below
        // requires `resident_device_null_columns.is_empty()`) and this append writes a NULL int4 as a
        // placeholder 0 WITHOUT creating/maintaining a bitmap. The device aggregate / DISTINCT / GROUP BY
        // routes derive NULL-ness solely from the bitmap, so an appended NULL would read as a phantom 0
        // (wrong results vs the re-admit baseline). Decline -> the caller re-admits, which BUILDS the
        // correct bitmap; thereafter the table has a null column so this path always declines it. (Bitmap
        // maintenance on append is a later slice.)
        if new_rows
            .iter()
            .any(|row| row.iter().any(|v| matches!(v, SqlValue::Null)))
        {
            return false;
        }
        // S-d2b: a SHARD-resident table (the segmented layout, default-OFF flag) appends to its OPEN shard's
        // headroom instead of the single buffer. (Admission publishes a table to shards XOR snapshots, so
        // the two paths never overlap for one table.)
        if self
            .read_state
            .residency
            .shards
            .load()
            .get(table)
            .is_some_and(|shards| !shards.is_empty())
        {
            return self.try_append_to_resident_open_shard(table, new_rows);
        }
        let (capacity, row_start, column_count) = {
            let snapshots = self.read_state.residency.snapshots.load();
            let Some(entry) = snapshots.get(table) else {
                return false;
            };
            let s = &entry.descriptor;
            // Purely int4-resident: every column rides the i32 section (no other typed sections), so the
            // int4 append op covers the whole row. (Date/Int2 ride i32 too — handled by the op.)
            let purely_int4 = s.resident_device_int8_columns.is_empty()
                && s.resident_device_numeric_columns.is_empty()
                && s.resident_device_bool_columns.is_empty()
                && s.resident_device_text_columns.is_empty()
                && s.resident_device_null_columns.is_empty()
                && s.column_count == s.resident_device_int4_columns.len();
            if !s.is_valid() || !purely_int4 {
                return false;
            }
            match s.row_count.checked_add(new_rows.len()) {
                Some(end) if end <= s.capacity => (s.capacity, s.row_count, s.column_count),
                _ => return false, // no headroom (or overflow) -> caller re-admits (with fresh headroom)
            }
        };
        let Some(device_memory) = self.read_state.residency.device_memory.get(table) else {
            return false;
        };
        // The append op reads each value's SqlValue variant (Int4/Date/Int2) for encoding; the column
        // TYPES only gate eligibility + count, and a purely-int4 table is all-i32-section by definition.
        let column_types = vec![SqlType::Int4; column_count];
        let chunks =
            match compute_open_shard_int4_append_chunks(&column_types, capacity, row_start, new_rows) {
                Ok(chunks) => chunks,
                Err(_) => return false,
            };
        if device_memory.append_owned_chunks(chunks).is_err() {
            // A partial/failed append leaves bytes only in the (still-invisible) headroom beyond
            // row_count; returning false makes the caller invalidate + re-admit, discarding them.
            return false;
        }
        let k = new_rows.len();
        let appended_bytes = (k * column_count * std::mem::size_of::<i32>()) as u64;
        self.read_state.residency.with_snapshots_mut(|snapshots| {
            if let Some(entry) = snapshots.get_mut(table) {
                let desc = std::sync::Arc::make_mut(&mut entry.descriptor);
                desc.generation = desc.generation.saturating_add(1);
                desc.row_count += k;
                desc.resident_bytes = desc.resident_bytes.saturating_add(appended_bytes);
                // Keep host_rows (the CPU-path materialization) consistent: new INSERTs get monotonic
                // tuple_ids, so they sort to the END of the seq-scan order — the same place the device
                // append wrote them. Slice 1b-ii-d: APPEND a new immutable segment (the appended rows)
                // instead of deep-cloning the whole row vec. make_mut here clones only the small
                // Vec<HostRowSegment> pointer list (one element per append since the last re-admit folds it
                // back to one), so the host side is O(rows appended) + O(num_segments) — no longer the
                // O(table) dual-store residual. Prior segments' row data is shared (refcounted), not copied.
                std::sync::Arc::make_mut(&mut entry.host_rows)
                    .push(std::sync::Arc::new(new_rows.to_vec()));
            }
        });
        // Slice 1b-ii (audit Finding A): the wave/lpb GPU index cache (engine_retained_read.rs) validates a
        // cached entry by (column, resident_device_ptr) ONLY — it is BLIND to generation/row_count. An
        // in-place append keeps the SAME device_ptr, so a cached index built over [0, old_row_count) would
        // be a stale HIT that reports the just-appended keys as not-found (a lost-from-reads committed
        // INSERT). Drop the table's entry so the next probe rebuilds over the new row_count. This runs
        // before the caller's publish_committed_seq, so a reader that observes the new committed_seq can
        // never bind the stale index. (In-flight probes pinned the prior index's own Arc; removing the
        // map entry only prevents NEW binds — the buffer frees once no submission holds it.)
        self.read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(table);
        self.read_state
            .residency
            .open_shard_append_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// S-d2b: append a committed INSERT's applied rows IN PLACE into the resident table's OPEN shard's
    /// headroom (the last shard in `residency.shards`) — the shard-path analog of the single-buffer append.
    /// Empty + NULL-bearing rows are already rejected by the caller. Returns false (caller invalidates +
    /// re-admits) when the open shard isn't int4-appendable, is invalid, or has no headroom (seal + a fresh
    /// open shard on overflow is S-d2c), or the device append fails. No host_rows (shard tables read via the
    /// device recompaction, not the host-materialization path) and no wave_index (the shard read recompacts
    /// a fresh unified buffer per query, so there is no ptr-keyed cached index to invalidate).
    fn try_append_to_resident_open_shard(&self, table: &str, new_rows: &[Vec<SqlValue>]) -> bool {
        let pressured_gpus = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .clone();
        let k = new_rows.len();
        // Read the OPEN (last) shard's state once.
        let (shard_id, capacity, row_count, row_start, column_count, column_names, gpu_id, schema, max_shard_id) = {
            let shards = self.read_state.residency.shards.load();
            let Some(table_shards) = shards.get(table) else {
                return false;
            };
            let Some(open) = table_shards.last() else {
                return false;
            };
            if !open.int4_appendable || !open.is_valid(pressured_gpus.contains(&open.gpu_id)) {
                return false;
            }
            (
                open.shard_id,
                open.capacity,
                open.row_count,
                open.row_start,
                open.resident_device_int4_columns.len(),
                open.resident_device_int4_columns.clone(),
                open.gpu_id,
                open.schema.clone(),
                table_shards.iter().map(|s| s.shard_id).max().unwrap_or(0),
            )
        };
        let column_types = vec![SqlType::Int4; column_count];

        // FITS the open shard's headroom -> append IN PLACE (1b-ii on the shard path).
        if row_count.checked_add(k).is_some_and(|end| end <= capacity) {
            let Some(shard_device_memory) = self
                .read_state
                .residency
                .shard_device_memory
                .get(&(table.to_string(), shard_id))
            else {
                return false;
            };
            // The append position within THIS shard's buffer is its LOCAL row_count (rows [0, row_count)
            // are live; the new rows go at [row_count, row_count+k)), NOT the shard's global `row_start`.
            let chunks =
                match compute_open_shard_int4_append_chunks(&column_types, capacity, row_count, new_rows) {
                    Ok(chunks) => chunks,
                    Err(_) => return false,
                };
            // SV1: no `created_by` stamp on append (per-row created_by is gone). `deleted_by` needs no write
            // either — the headroom was pre-filled with the live sentinel at admission, so appended rows are
            // born live. Only the value-column + header chunks are written.
            if shard_device_memory.append_owned_chunks(chunks).is_err() {
                // Partial/failed append leaves bytes only in invisible headroom beyond row_count;
                // returning false makes the caller invalidate + re-admit, discarding them.
                return false;
            }
            let appended_bytes = (k * column_count * std::mem::size_of::<i32>()) as u64;
            // S-d3: extend the open shard's zone map (min/max per int4 column) to cover the appended rows,
            // so a point-lookup prune never wrongly skips a shard holding a just-appended needle. O(k*cols).
            let new_min_max: Vec<(i32, i32)> = (0..column_count)
                .map(|c| {
                    new_rows.iter().fold((i32::MAX, i32::MIN), |(lo, hi), row| {
                        let v = sql_value_as_int4(&row[c]);
                        (lo.min(v), hi.max(v))
                    })
                })
                .collect();
            self.read_state.residency.with_shards_mut(|shards| {
                if let Some(table_shards) = shards.get_mut(table) {
                    if let Some(open) = table_shards.last_mut() {
                        open.row_count += k;
                        open.resident_bytes = open.resident_bytes.saturating_add(appended_bytes);
                        for (stat, (lo, hi)) in open
                            .resident_device_int4_column_stats
                            .iter_mut()
                            .zip(new_min_max.iter())
                        {
                            stat.min = stat.min.min(*lo);
                            stat.max = stat.max.max(*hi);
                        }
                    }
                }
            });
            self.read_state
                .residency
                .open_shard_append_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }

        // S-d2c ROLLOVER: the open shard is full -> SEAL it (leave it in place, immutable) and build + install
        // a NEW open shard holding the k rows (capacity = the target, so it grows to the target before the
        // next rollover). O(rows appended), NOT the O(table) re-admit -> this is what removes the 536M cap.
        let new_capacity = self
            .shard_size_target()
            .max(k.saturating_mul(2).next_power_of_two());
        let (device_payload, int4_stats) = match build_relational_device_payload_with_capacity(
            &column_names,
            &column_types,
            new_rows,
            new_capacity,
        ) {
            // Pure int4 + NULL-free (the caller rejects NULLs) -> text/bool/b128/null outputs are empty; keep
            // the columnar payload + the int4 zone-map stats (min/max over the k rows) for pruning (S-d3).
            Ok((payload, _text, _bool, stats, _b128, _null)) => (payload, stats),
            Err(_) => return false,
        };
        // SV1/SV2: the rolled shard carries NO version metadata in its payload — `created_by` is gone and
        // `deleted_by` is on-demand (allocated in `shard_deleted_by_memory` on the shard's first delete).
        let Some(new_device_memory) = self.relational_residency_device_memory(gpu_id, &device_payload)
        else {
            return false;
        };
        let new_shard_id = max_shard_id.saturating_add(1);
        let pressured = pressured_gpus.contains(&gpu_id);
        let new_shard = RelationalResidentShard {
            shard_id: new_shard_id,
            row_start: row_start.saturating_add(row_count),
            row_count: k,
            capacity: new_capacity,
            int4_appendable: true,
            resident_device_int4_column_stats: int4_stats,
            resident_bytes: (8 + k * column_count * std::mem::size_of::<i32>()) as u64,
            allocated_bytes: device_payload.len() as u64,
            count_header_byte_offset: 0,
            resident_device_int4_columns: column_names,
            resident_device_text_columns: Vec::new(),
            gpu_id,
            schema,
            table: table.to_string(),
            device_memory_proof: Some(new_device_memory.metadata().clone()),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: pressured,
            memory_pressure_active: pressured,
        };
        // Publish the new shard's device memory BEFORE its metadata, so a reader that observes the new shard
        // in the shards list always finds its device memory (the recompaction loads the list then the memory).
        self.read_state
            .residency
            .shard_device_memory
            .insert_shard(table, new_shard_id, new_device_memory);
        self.read_state.residency.with_shards_mut(|shards| {
            if let Some(table_shards) = shards.get_mut(table) {
                table_shards.push(new_shard);
            }
        });
        self.read_state
            .residency
            .open_shard_append_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Slice A2 (incremental DELETE): stamp `deleted_by[slot] = commit_seq` for each of `slots` in a
    /// resident shard's tombstone section, via ONE targeted device write (`append_owned_chunks`, which
    /// bounds-checks each chunk against the allocation). This is an OUT-OF-LINE tombstone: it writes only
    /// the `deleted_by` metadata word, NEVER the row's column bytes — so a lock-free, predicate-free reader
    /// can never observe a torn row (review Finding 3), and the change is a single aligned u64 store.
    ///
    /// Returns `false` (caller must fall back to invalidate + re-admit) if the shard is missing / any slot is
    /// out of `[0, row_count)` / the region allocation or device write fails. `slots` are LOCAL indices.
    ///
    /// UNWIRED (`#[allow(dead_code)]`) — wired into the DELETE-only commit path in SV4 (slot-finding via the
    /// pruned-shard predicate). **SV4 PREREQUISITES (audit-flagged, out of scope until wired — the primitive
    /// creates NO region in production today, so they are inert now):**
    ///  1. **Lifecycle/leak — DONE (SV4 prereq #1):** `shard_deleted_by_memory` cleanup is now wired at every
    ///     site the resident buffer it annotates is retired. Two categories, distinguished by whether an
    ///     invalidate precedes the retire:
    ///       - `invalidate_table` (device buffer freed, cell kept) on the three invalidate paths (serialized
    ///         `invalidate_relational_residency_table` + concurrent-commit + memory-pressure variants).
    ///       - full `remove_table` (keys erased) on the paths that retire a buffer WITHOUT a preceding
    ///         invalidate: the single-buffer AND sharded re-admit branches in `populate_..._snapshot_inner`
    ///         (a warmup/refresh has no invalidate), the BUDGET-EVICTION path (`RelationalResidentCache::
    ///         remove_table`, evicting a different table during admission), and `apply_drop_table` (a DROPped
    ///         table is gone for good — stronger than `shard_device_memory`, which leaves `None` cells on DROP).
    ///     Gates (all sabotage-verified non-vacuous): `shard_deleted_by_region_released_on_invalidate_and_drop`
    ///     (invalidate + DROP), `shard_deleted_by_region_released_on_warmup_readmit` (sharded re-admit with no
    ///     preceding invalidate), and `resident_cache_remove_table_releases_deleted_by_region` (the eviction-
    ///     cleanup method contract, currently defensive). This keeps a re-admit (rebuilt all-live from the host
    ///     store) from inheriting a stale tombstone region and stops evicted/dropped tables leaking regions.
    ///  2. **Concurrency:** hold the COMMIT LOCK across the get-or-allocate below, else two concurrent
    ///     first-deletes to the same shard both allocate + the losing region's `Arc` leaks (writes still land
    ///     safely; only the buffer leaks). SV4 runs this under the serialized commit lock, which is the fix.
    #[allow(dead_code)]
    pub(crate) fn tombstone_resident_shard_slots(
        &self,
        table: &str,
        shard_id: u32,
        slots: &[u32],
        commit_seq: Index,
    ) -> bool {
        if slots.is_empty() {
            return true;
        }
        // Read the shard's shape once. The commit lock (when wired) makes this + the region allocation atomic;
        // even without it, `append_owned_chunks` re-bounds-checks every chunk vs the region's allocated_bytes,
        // so a torn read can only produce a rejected write (-> `false` -> re-admit), never an OOB.
        let (capacity, row_count, gpu_id) = {
            let shards = self.read_state.residency.shards.load();
            let Some(table_shards) = shards.get(table) else {
                return false;
            };
            let Some(shard) = table_shards.iter().find(|s| s.shard_id == shard_id) else {
                return false;
            };
            (shard.capacity, shard.row_count, shard.gpu_id)
        };
        // Bounds: every slot must be a live row of THIS shard (never headroom / out of range).
        if slots.iter().any(|&slot| (slot as usize) >= row_count) {
            return false;
        }
        // SV2: get-or-allocate the shard's ON-DEMAND `deleted_by` region (a separate `capacity`-sized u64
        // device buffer born all-live). A delete-free shard has NO entry -> the FIRST delete allocates it, so
        // the un-versioned majority pays zero. The region is `capacity` (not `row_count`) u64s so later
        // in-place appends into the open shard's headroom are already live without extending it.
        let width = std::mem::size_of::<u64>() as u64;
        let region = match self
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), shard_id))
        {
            Some(region) => region,
            None => {
                // Born all-live: every u64 = the LIVE sentinel `0x7F7F_7F7F_7F7F_7F7F` (memset byte 0x7F).
                // It must be a LARGE POSITIVE SIGNED i64 (the read visibility compare `deleted_by >
                // read_txn_id` uses the SIGNED s64 kernel) — `u64::MAX` would be -1 as signed and a live row
                // would wrongly FAIL `> read_txn_id`. 0x7F7F... ≈ 9.1e18 > every real commit `Index`; byte
                // 0x7F is also uniform so the same value is producible by the SV3a recompaction memset-fill.
                let live_payload = vec![DELETED_BY_LIVE_FILL_BYTE; capacity * std::mem::size_of::<u64>()];
                let Some(region) = self.relational_residency_device_memory(gpu_id, &live_payload) else {
                    return false;
                };
                self.read_state.residency.shard_deleted_by_memory.insert_shard(
                    table,
                    shard_id,
                    region,
                );
                match self
                    .read_state
                    .residency
                    .shard_deleted_by_memory
                    .get(&(table.to_string(), shard_id))
                {
                    Some(region) => region,
                    None => return false,
                }
            }
        };
        let chunks: Vec<CudaOwnedDeviceMemoryChunk> = slots
            .iter()
            .map(|&slot| CudaOwnedDeviceMemoryChunk {
                // The region is JUST deleted_by (0-based): slot `s`'s stamp is at byte `s * 8`.
                byte_offset: u64::from(slot) * width,
                bytes: commit_seq.to_le_bytes().to_vec(),
            })
            .collect();
        region.append_owned_chunks(chunks).is_ok()
    }

    /// STRATA S-B: commit-triggered, best-effort GPU-residency admission for the tables a commit
    /// mutated. Runs AFTER `publish_committed_seq` (so it snapshots the new generation) while the
    /// commit_mutex is held; it can NEVER fail the commit — over-budget / memory-pressure / GPU-absent /
    /// dropped-table simply leaves the table non-resident (reads fall back to the host path). N=1
    /// unified buffer per table (single-GPU); shard/spill is S-C/S-E.
    pub(crate) fn auto_admit_resident_tables(&self, tables: &std::collections::BTreeSet<String>) {
        if tables.is_empty() {
            return;
        }
        let gpu_id = self.planner.default_gpu_id();
        let mut guard = self.ddl_catalog();
        let cat = &mut *guard;
        for table in tables {
            let _ = self.populate_relational_residency_snapshot_inner(cat, table, gpu_id);
        }
    }

    fn relational_residency_device_memory(
        &self,
        gpu_id: u16,
        payload: &[u8],
    ) -> Option<CudaResidentDeviceMemory> {
        let runtime = self.cuda_driver_probe_runtime();
        runtime.retain_device_memory_copy(gpu_id, payload).ok()
    }

    /// Build a TRANSIENT resident-like relation from already-materialized host `rows` -- a `RelationalTable`
    /// descriptor + an uploaded device payload that the GPU join path consumes EXACTLY like a published
    /// resident table (`lower_resident_predicate`, `project_*_rows_from_payload`, `hash_join_inner_i64`),
    /// but WITHOUT publishing/admitting/evicting anything (the descriptor + device memory live only for the
    /// caller's query). This is the M5 J5 bridge for a SYNTHESIZED `pg_catalog`/`information_schema`
    /// relation, which has no residency snapshot: synthesize its rows -> this helper -> the existing int4
    /// inner join over the transient payload. Charter: the catalog join runs on the SAME GPU kernels as a
    /// user-table join (no CPU relational join; only the host-rows gather crosses to the host, as for a
    /// resident table). `&self`: the upload only needs `cuda_driver_probe_runtime` (also `&self`).
    ///
    /// Mirrors `populate_relational_residency_snapshot_on_gpu`'s payload + descriptor build (the column
    /// lists feed `build_relational_device_payload`, whose offsets the descriptor's resident-column lists
    /// index), but SKIPS the MVCC tuple tail (the join reads columnar sections + host rows, never the tail)
    /// and the admission machinery. A 0-row relation is fine: the payload is still a non-empty 8-byte
    /// row-count header (the upload's empty-payload guard never trips), and the inner join then yields an
    /// empty result via the empty-survivor / empty-key short-circuits (an empty side is the join's identity).
    pub(crate) fn build_transient_relation_residency(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, CudaResidentDeviceMemory), ExecuteError> {
        let gpu_id = self.planner.default_gpu_id();
        let column_names: Vec<String> =
            table.columns.iter().map(|column| column.name.clone()).collect();
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        // The resident-column lists name (in catalog order) the columns living in each type-grouped
        // payload section; the descriptor's offset helpers index `build_relational_device_payload`'s
        // sections via these lists, so they MUST use the SAME type filters as the resident builder.
        let resident_device_int4_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_int8_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let resident_device_numeric_columns = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let (
            device_payload,
            resident_device_text_columns,
            resident_device_bool_columns,
            resident_device_int4_column_stats,
            _resident_device_b128_columns,
            resident_device_null_columns,
        ) = build_relational_device_payload(&column_names, &column_types, rows)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_copy(gpu_id, &device_payload)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 1,
            row_count: rows.len(),
            capacity: rows.len(),
            column_count: table.columns.len(),
            resident_bytes: device_payload.len() as u64,
            resident_device_int4_columns,
            resident_device_int4_column_stats,
            resident_device_int8_columns,
            resident_device_numeric_columns,
            resident_device_bool_columns,
            resident_device_text_columns,
            resident_device_null_columns,
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(device_memory.metadata().clone()),
        };
        Ok((snapshot, device_memory))
    }

    pub fn install_benchmark_relational_residency_chunks(
        &mut self,
        install: BenchmarkRelationalResidencyChunkInstall<'_>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError> {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        let chunks = install.chunks;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        if chunks.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one retained chunk"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;
        let copied_bytes = chunks
            .iter()
            .try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident chunk ending at byte {end} exceeds allocation {allocated_bytes}"
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident copied byte count overflowed".to_string(),
                    ))
                })
            })?;
        if copied_bytes == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission copied no bytes".to_string(),
            )));
        }

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, resident_bytes)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_chunks(gpu_id, allocated_bytes, chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            capacity: row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: self.committed_seq(),
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog_mut()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Vec::new(), // benchmark install path: no host-row materialization
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_chunks<I>(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedChunkInstall<'_, I>,
    ) -> Result<RelationalResidencySnapshot, ExecuteError>
    where
        I: IntoIterator<Item = CudaOwnedDeviceMemoryChunk>,
    {
        let table = install.table;
        let gpu_id = install.gpu_id;
        let row_count = install.row_count;
        let resident_bytes = install.resident_bytes;
        let allocated_bytes = install.allocated_bytes;
        if row_count == 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident chunk admission requires at least one generated row"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();

        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }
        Self::validate_benchmark_resident_chunk_columns(
            &catalog_table,
            &install.resident_device_int4_columns,
            &install.resident_device_int4_column_stats,
            &install.resident_device_text_columns,
        )?;

        let previous_snapshot = self
            .read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone());
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let admission_budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let (evicted_tables_on_admission, resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, gpu_id, resident_bytes)?;
        let runtime = self.cuda_driver_probe_runtime();
        let device_memory = runtime
            .retain_device_memory_owned_chunks(gpu_id, allocated_bytes, install.chunks)
            .map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident chunk admission failed CUDA retained upload: {err}"
                )))
            })?;
        let device_memory_proof = Some(device_memory.metadata().clone());
        let snapshot = RelationalResidencySnapshot {
            gpu_id,
            schema: catalog_table.schema,
            table: catalog_table.name.clone(),
            generation: RelationalResidencySnapshot::next_generation(previous_snapshot.as_deref()),
            row_count,
            capacity: row_count,
            column_count: catalog_table.columns.len(),
            resident_bytes,
            resident_device_int4_columns: install.resident_device_int4_columns,
            resident_device_int4_column_stats: install.resident_device_int4_column_stats,
            // Benchmark install path: int8 device retention is not wired here yet (doc 19 — the
            // general executor reads int8 only from the standard populate path).
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: install.resident_device_text_columns,
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: memory_pressure_active,
            memory_pressure_active,
            last_refresh_cost: previous_snapshot.as_ref().map(|previous| {
                RelationalResidencyRefreshCost {
                    previous_row_count: previous.row_count,
                    refreshed_row_count: row_count,
                    row_delta: row_count as i128 - previous.row_count as i128,
                    previous_resident_bytes: previous.resident_bytes,
                    refreshed_resident_bytes: resident_bytes,
                    resident_byte_delta: resident_bytes as i128 - previous.resident_bytes as i128,
                    refreshed_from_index: previous.valid_through_index,
                    refreshed_through_index: self.committed_seq(),
                    invalidated_by_txn_id: previous.invalidated_by_txn_id,
                    invalidated_at_index: previous.invalidated_at_index,
                    invalidated_by_memory_pressure: previous.invalidated_by_memory_pressure,
                }
            }),
            admission_budget_bytes,
            resident_bytes_after_admission,
            evicted_tables_on_admission,
            device_memory_proof,
        };
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog_mut()
            .relational_resident_cache
            .install_snapshot(
                catalog_table.name,
                snapshot.clone(),
                Vec::new(), // benchmark install path: no host-row materialization
                Some(device_memory),
                &read_state.residency,
            );
        Ok(snapshot)
    }

    pub fn install_benchmark_relational_residency_owned_shards(
        &mut self,
        install: BenchmarkRelationalResidencyOwnedShardInstall<'_>,
    ) -> Result<(), ExecuteError> {
        let table = install.table;
        if install.shards.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "benchmark resident shard admission requires at least one shard"
                    .to_string(),
            )));
        }
        let catalog_table = self
            .ddl_catalog_mut()
            .relational_catalog
            .get(table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table}\" does not exist"
                )))
            })?
            .clone();
        let visible_rows = self.visible_relational_row_count(table)?;
        if visible_rows != 0 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident shard admission requires relation \"{table}\" to have no SQL-visible rows; found {visible_rows}"
            ))));
        }

        let total_resident_bytes =
            install
                .shards
                .iter()
                .try_fold(0_u64, |total, shard| {
                    if shard.row_count == 0 {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "benchmark resident shard {} has no rows",
                            shard.shard_id
                        ))));
                    }
                    if shard.chunks.is_empty() {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "benchmark resident shard {} has no retained chunks",
                            shard.shard_id
                        ))));
                    }
                    total.checked_add(shard.resident_bytes).ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "benchmark resident shard byte count overflowed".to_string(),
                        ))
                    })
                })?;
        let (_evicted_tables_on_admission, _resident_bytes_after_admission) =
            self.admit_relational_residency_snapshot(table, install.gpu_id, total_resident_bytes)?;

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&install.gpu_id);
        let runtime = self.cuda_driver_probe_runtime();
        let mut shards = Vec::new();
        let mut device_memory = BTreeMap::new();
        for shard in install.shards {
            let copied_bytes = shard.chunks.iter().try_fold(0_u64, |total, chunk| {
                let len = u64::try_from(chunk.bytes.len()).map_err(|_| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk length exceeds u64".to_string(),
                    ))
                })?;
                let end = chunk.byte_offset.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard chunk offset overflowed".to_string(),
                    ))
                })?;
                if end > shard.allocated_bytes {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard {} chunk ending at byte {end} exceeds allocation {}",
                        shard.shard_id, shard.allocated_bytes
                    ))));
                }
                total.checked_add(len).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "benchmark resident shard copied byte count overflowed".to_string(),
                    ))
                })
            })?;
            if copied_bytes == 0 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "benchmark resident shard {} copied no bytes",
                    shard.shard_id
                ))));
            }
            let retained = runtime
                .retain_device_memory_owned_chunks(
                    install.gpu_id,
                    shard.allocated_bytes,
                    shard.chunks,
                )
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "benchmark resident shard admission failed CUDA retained upload: {err}"
                    )))
                })?;
            let device_memory_proof = Some(retained.metadata().clone());
            shards.push(RelationalResidentShard {
                shard_id: shard.shard_id,
                row_start: shard.row_start,
                row_count: shard.row_count,
                // Benchmark shards are read DENSE (explicit chunk layouts sized by row_count) and are not
                // append targets.
                capacity: shard.row_count,
                int4_appendable: false,
                // S-d3: benchmark shards carry no zone map -> never pruned (always gathered).
                resident_device_int4_column_stats: Vec::new(),
                // A1/SV2: benchmark shards carry no version metadata (no on-demand deleted_by region) -> the
                // visibility mask is skipped (they read as all-live, correct for latest-snapshot benchmarks).
                resident_bytes: shard.resident_bytes,
                allocated_bytes: shard.allocated_bytes,
                count_header_byte_offset: 0,
                resident_device_int4_columns: shard.resident_device_int4_columns,
                resident_device_text_columns: shard.resident_device_text_columns,
                gpu_id: install.gpu_id,
                schema: catalog_table.schema.clone(),
                table: catalog_table.name.clone(),
                device_memory_proof,
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: memory_pressure_active,
                memory_pressure_active,
            });
            device_memory.insert(shard.shard_id, retained);
        }
        shards.sort_by_key(|shard| (shard.row_start, shard.shard_id));
        let read_state = Arc::clone(&self.read_state);
        self.ddl_catalog_mut()
            .relational_resident_cache
            .install_shards(
                catalog_table.name,
                shards,
                device_memory,
                &read_state.residency,
            );
        Ok(())
    }

    /// Synthesize a single-store-shaped [`RelationalResidencySnapshot`] DESCRIPTOR for ONE shard
    /// (S10c slice 1). A shard's SoA payload is self-contained (`count_header_byte_offset == 0`,
    /// sized by `shard.row_count`), so a descriptor whose `row_count == shard.row_count` plus the
    /// shard's `resident_device_{int4,text}_columns` makes the SINGLE-store offset helpers address the
    /// shard buffer BYTE-IDENTICALLY — letting the general resident-Expr executor serve one shard
    /// slice when handed it via `ResidentExecSource`. Mirrors the benchmark snapshot constructor (the
    /// per-table install path) field-for-field; the fields the offset helpers DON'T read (generation,
    /// stats, int8/numeric/bool/null columns, refresh cost, admission accounting) take inert defaults.
    /// The identity guard (`schema`/`table` == catalog) and `is_valid()` are satisfied for a valid
    /// shard, so the executor's per-source identity/validity prechecks pass.
    pub(crate) fn resident_snapshot_for_shard(
        &self,
        shard: &RelationalResidentShard,
        table: &RelationalTable,
    ) -> RelationalResidencySnapshot {
        RelationalResidencySnapshot {
            gpu_id: shard.gpu_id,
            schema: shard.schema.clone(),
            table: shard.table.clone(),
            generation: 0,
            // CRITICAL: the shard's own row count sizes the SoA the single-store offset helpers
            // read, so they address THIS shard's buffer (not the whole table). S-d2: an OPEN shard is
            // capacity-padded (headroom for appends), so the column STRIDE is `shard.capacity` while the
            // live row count is `shard.row_count` — exactly the single buffer's capacity/row_count split.
            row_count: shard.row_count,
            capacity: shard.capacity,
            column_count: table.columns.len(),
            resident_bytes: shard.resident_bytes,
            resident_device_int4_columns: shard.resident_device_int4_columns.clone(),
            resident_device_int4_column_stats: Vec::new(),
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: shard.resident_device_text_columns.clone(),
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: shard.invalidated_by_txn_id,
            invalidated_at_index: shard.invalidated_at_index,
            invalidated_by_memory_pressure: shard.invalidated_by_memory_pressure,
            memory_pressure_active: shard.memory_pressure_active,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: shard.device_memory_proof.clone(),
        }
    }

    /// S10c slice 2a: synthesize the single-store-shaped DESCRIPTOR for the ONE UNIFIED int4-only
    /// buffer recompacted from all of a table's shards. Like [`Self::resident_snapshot_for_shard`]
    /// but sized by the WHOLE table (`row_count == total_row_count`) so the single-store offset helpers
    /// address the unified SoA byte-identically. `int4_columns` is the shards' OWN (uniform)
    /// `resident_device_int4_columns` -- i.e. the list the unified buffer was physically recompacted from,
    /// NOT a catalog re-derivation. Labelling the descriptor with the actual buffer layout keeps the
    /// offset helper's per-read name-check load-bearing (a read of a column whose name does not sit at the
    /// labelled int4 ordinal errors instead of silently returning another column's bytes) -- audit F1.
    /// Text is deferred in this slice, so `resident_device_text_columns` is empty. The
    /// `device_memory_proof` is the unified buffer's freshly-built proof.
    pub(crate) fn resident_snapshot_for_unified(
        &self,
        table: &RelationalTable,
        total_row_count: usize,
        gpu_id: u16,
        resident_bytes: u64,
        proof: CudaDeviceMemoryProof,
        int4_columns: Vec<String>,
    ) -> RelationalResidencySnapshot {
        let resident_device_int4_columns = int4_columns;
        RelationalResidencySnapshot {
            gpu_id,
            schema: table.schema.clone(),
            table: table.name.clone(),
            generation: 0,
            row_count: total_row_count,
            capacity: total_row_count,
            column_count: table.columns.len(),
            resident_bytes,
            resident_device_int4_columns,
            resident_device_int4_column_stats: Vec::new(),
            resident_device_int8_columns: Vec::new(),
            resident_device_numeric_columns: Vec::new(),
            resident_device_bool_columns: Vec::new(),
            resident_device_text_columns: Vec::new(),
            resident_device_null_columns: Vec::new(),
            valid_through_index: self.committed_seq(),
            invalidated_by_txn_id: None,
            invalidated_at_index: None,
            invalidated_by_memory_pressure: false,
            memory_pressure_active: false,
            last_refresh_cost: None,
            admission_budget_bytes: None,
            resident_bytes_after_admission: 0,
            evicted_tables_on_admission: Vec::new(),
            device_memory_proof: Some(proof),
        }
    }

    fn validate_benchmark_resident_chunk_columns(
        table: &RelationalTable,
        int4_columns: &[String],
        int4_stats: &[ResidentDeviceInt4ColumnStats],
        text_columns: &[ResidentDeviceTextColumnLayout],
    ) -> Result<(), ExecuteError> {
        let expected_int4 = table
            .columns
            .iter()
            .filter(|column| column.ty == SqlType::Int4)
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        if int4_columns != expected_int4.as_slice() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 column layout {:?} does not match catalog int4 columns {:?}",
                int4_columns, expected_int4
            ))));
        }
        let actual_int4_stats = int4_stats
            .iter()
            .map(|stats| stats.name.clone())
            .collect::<Vec<_>>();
        if actual_int4_stats != expected_int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 stats layout {:?} does not match catalog int4 columns {:?}",
                actual_int4_stats, expected_int4
            ))));
        }
        if let Some(stats) = int4_stats.iter().find(|stats| stats.min > stats.max) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk int4 stats for column \"{}\" have min greater than max",
                stats.name
            ))));
        }
        let expected_text = table
            .columns
            .iter()
            .filter(|column| column.ty == SqlType::Text)
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let actual_text = text_columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        if actual_text != expected_text {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "benchmark resident chunk text column layout {:?} does not match catalog text columns {:?}",
                actual_text, expected_text
            ))));
        }
        Ok(())
    }

    fn visible_relational_row_count(&self, table: &str) -> Result<usize, ExecuteError> {
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        let prefix = relational_key_prefix(table);
        let table_rows = self.read_state.mvcc.table_rows(table);
        let mut cursor = table_rows.store().seq_scan_open(visibility)?;
        let mut row_count = 0usize;
        while let Some(tuple) = cursor.next() {
            if tuple.key.starts_with(&prefix) {
                row_count = row_count.checked_add(1).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "visible relational row count overflowed".to_string(),
                    ))
                })?;
            }
        }
        Ok(row_count)
    }

    fn relational_resident_bytes_for_gpu_excluding(&self, gpu_id: u16, table: &str) -> u64 {
        let snapshot_bytes: u64 = self
            .read_state
            .residency
            .snapshots
            .load()
            .iter()
            .filter(|(name, entry)| name.as_str() != table && entry.descriptor.gpu_id == gpu_id)
            .map(|(_name, entry)| entry.descriptor.resident_bytes)
            .sum();
        let shard_bytes: u64 = self
            .read_state
            .residency
            .shards
            .load()
            .iter()
            .filter(|(name, _shards)| name.as_str() != table)
            .flat_map(|(_name, shards)| shards)
            .filter(|shard| shard.gpu_id == gpu_id)
            .map(|shard| shard.resident_bytes)
            .sum();
        snapshot_bytes.saturating_add(shard_bytes)
    }

    pub fn relational_residency_snapshot(
        &self,
        table: &str,
    ) -> Option<RelationalResidencySnapshot> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| {
                let mut snapshot = (*entry.descriptor).clone();
                snapshot.memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                snapshot
            })
    }

    pub fn relational_retained_snapshot_handle(
        &self,
        table: &str,
    ) -> Option<RelationalRetainedSnapshotHandle> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| {
                let snapshot = &entry.descriptor;
                let memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                RelationalRetainedSnapshotHandle {
                    schema: snapshot.schema.clone(),
                    table: snapshot.table.clone(),
                    gpu_id: snapshot.gpu_id,
                    generation: snapshot.generation,
                    row_count: snapshot.row_count,
                    column_count: snapshot.column_count,
                    resident_bytes: snapshot.resident_bytes,
                    valid_through_index: snapshot.valid_through_index,
                    valid: snapshot.invalidated_by_txn_id.is_none()
                        && snapshot.invalidated_at_index.is_none()
                        && !snapshot.invalidated_by_memory_pressure
                        && !memory_pressure_active,
                    has_retained_device_memory: self
                        .read_state
                        .residency
                        .device_memory
                        .contains_key(table),
                    resident_device_int4_columns: snapshot.resident_device_int4_columns.clone(),
                    resident_device_text_columns: snapshot.resident_device_text_columns.clone(),
                }
            })
    }

    pub fn relational_retained_device_read_view(
        &self,
        table: &str,
    ) -> Option<CudaResidentDeviceMemoryReadView> {
        let handle = self.relational_retained_snapshot_handle(table)?;
        if !handle.valid || !handle.has_retained_device_memory {
            return None;
        }
        self.read_state
            .residency
            .device_memory
            .get(table)
            .map(|device_memory| device_memory.read_view())
    }

    /// Pin the resident snapshot metadata for `table` as an OWNED clone (Stage 3 — blocker #2). The
    /// resident-route consumers used to hold a `&` borrow of the snapshot map across the kernel launch;
    /// now the map is published behind `ArcSwap`, so this loads the published generation and clones the
    /// table's entry out. The clone is owned (no map/guard borrow held across the submission), and the
    /// consumers only read scalar fields + column layouts off it before submitting — so an owned clone
    /// is a drop-in for the former borrow with no lifetime entanglement. Cloning a single snapshot's
    /// metadata once per resident-route statement is negligible against the GPU kernel it precedes.
    /// The lightweight, Arc-shared GPU/catalog DESCRIPTOR for a resident table (no host rows). Readers
    /// clone the `Arc` -- a refcount bump, never the row data. (Was an owned deep-clone that copied the
    /// table's host rows on every general-executor query; the split moved those to `host_rows`.)
    pub(crate) fn relational_residency_snapshot_ref(
        &self,
        table: &str,
    ) -> Option<Arc<RelationalResidencySnapshot>> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .map(|entry| entry.descriptor.clone())
    }

    /// The WHOLE residency entry (descriptor + host rows) from ONE atomic `load()`, so a reader that
    /// needs BOTH halves sees a single consistent generation. Use this instead of calling
    /// `relational_residency_snapshot_ref` + `relational_residency_host_rows` separately -- two
    /// `load()`s could straddle a concurrent publish and pair a descriptor with mismatched rows.
    pub(crate) fn relational_residency_entry(
        &self,
        table: &str,
    ) -> Option<RelationalResidencyEntry> {
        self.read_state
            .residency
            .snapshots
            .load()
            .get(table)
            .cloned()
    }

    pub fn warm_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyWarmupPolicy,
    ) -> RelationalResidencyWarmupReport {
        let policy_sets_gpu = policy.gpu_id.is_some();
        let gpu_id = policy
            .gpu_id
            .unwrap_or_else(|| self.planner.default_gpu_id());
        let policy_sets_budget = policy.budget_bytes.is_some();
        if let Some(budget_bytes) = policy.budget_bytes {
            self.set_relational_residency_budget_bytes(gpu_id, budget_bytes);
        }
        let budget_bytes = self.relational_residency_budget_bytes(gpu_id);
        let requested_tables = if policy.tables.is_empty() {
            self.ddl_catalog_mut()
                .relational_catalog
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        } else {
            policy.tables.clone()
        };
        let mut selected_tables = requested_tables.clone();
        selected_tables.sort();
        selected_tables.dedup();
        if let Some(max_table_count) = policy.max_table_count {
            selected_tables.truncate(max_table_count);
        }

        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&gpu_id);
        let mut entries = Vec::new();
        for table in selected_tables {
            if memory_pressure_active {
                entries.push(RelationalResidencyWarmupEntry {
                    table,
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: format!("GPU {gpu_id} is memory pressured"),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }
            if !self
                .ddl_catalog_mut()
                .relational_catalog
                .contains_key(&table)
            {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "only supported public base tables can be warmed".to_string(),
                    resident_bytes: 0,
                    evicted_tables: Vec::new(),
                    route_decision: None,
                });
                continue;
            }

            let existing = self.relational_residency_snapshot(&table);
            let existing_valid = existing
                .as_ref()
                .is_some_and(|snapshot| snapshot.is_valid());
            let existing_retained = self.read_state.residency.device_memory.contains_key(&table);
            if existing_valid && existing_retained && !policy_sets_budget && !policy_sets_gpu {
                let route_decision = self.warmup_route_readiness_decision(&table);
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::AlreadyResident,
                    reason: "resident snapshot is already valid and retained".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision,
                });
                continue;
            }
            if existing.is_some() && !policy.refresh_invalidated && !existing_valid {
                entries.push(RelationalResidencyWarmupEntry {
                    table: table.clone(),
                    action: RelationalResidencyWarmupAction::Skipped,
                    reason: "resident snapshot is invalidated and refresh is disabled".to_string(),
                    resident_bytes: existing
                        .as_ref()
                        .map(|snapshot| snapshot.resident_bytes)
                        .unwrap_or(0),
                    evicted_tables: Vec::new(),
                    route_decision: self.warmup_route_readiness_decision(&table),
                });
                continue;
            }

            let refreshing = existing.is_some();
            match self.populate_relational_residency_snapshot_on_gpu(&table, gpu_id) {
                Ok(snapshot) => {
                    let route_decision = self.warmup_route_readiness_decision(&table);
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: if refreshing {
                            RelationalResidencyWarmupAction::Refreshed
                        } else {
                            RelationalResidencyWarmupAction::Warmed
                        },
                        reason: self
                            .ddl_catalog_mut()
                            .relational_resident_cache
                            .last_decision(&table)
                            .map(|decision| decision.reason.clone())
                            .unwrap_or_else(|| "resident snapshot warmed".to_string()),
                        resident_bytes: snapshot.resident_bytes,
                        evicted_tables: snapshot.evicted_tables_on_admission,
                        route_decision,
                    });
                }
                Err(err) => {
                    entries.push(RelationalResidencyWarmupEntry {
                        table: table.clone(),
                        action: RelationalResidencyWarmupAction::Error,
                        reason: err.to_string(),
                        resident_bytes: 0,
                        evicted_tables: Vec::new(),
                        route_decision: self.warmup_route_readiness_decision(&table),
                    });
                }
            }
        }

        RelationalResidencyWarmupReport {
            gpu_id,
            budget_bytes,
            requested_tables,
            entries,
        }
    }

    pub fn maintain_relational_residency_with_policy(
        &mut self,
        policy: RelationalResidencyMaintenancePolicy,
    ) -> RelationalResidencyMaintenanceReport {
        let warmup = self.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
            gpu_id: policy.gpu_id,
            tables: policy.tables,
            max_table_count: policy.max_table_count,
            budget_bytes: policy.budget_bytes,
            refresh_invalidated: policy.refresh_invalidated,
        });
        let mut warmed_count = 0;
        let mut refreshed_count = 0;
        let mut already_resident_count = 0;
        let mut skipped_count = 0;
        let mut error_count = 0;
        let mut route_ready_tables = Vec::new();
        let mut route_blockers = Vec::new();

        for entry in &warmup.entries {
            match entry.action {
                RelationalResidencyWarmupAction::Warmed => warmed_count += 1,
                RelationalResidencyWarmupAction::Refreshed => refreshed_count += 1,
                RelationalResidencyWarmupAction::AlreadyResident => already_resident_count += 1,
                RelationalResidencyWarmupAction::Skipped => skipped_count += 1,
                RelationalResidencyWarmupAction::Error => error_count += 1,
            }

            match entry.route_decision.as_ref() {
                Some(route) if route.accepted => route_ready_tables.push(entry.table.clone()),
                Some(route) => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: if matches!(
                        entry.action,
                        RelationalResidencyWarmupAction::Skipped
                            | RelationalResidencyWarmupAction::Error
                    ) {
                        entry.reason.clone()
                    } else {
                        route.reason.clone()
                    },
                }),
                None => route_blockers.push(RelationalResidencyMaintenanceBlocker {
                    table: entry.table.clone(),
                    reason: entry.reason.clone(),
                }),
            }
        }

        let entry_count = warmup.entries.len();
        RelationalResidencyMaintenanceReport {
            gpu_id: warmup.gpu_id,
            budget_bytes: warmup.budget_bytes,
            requested_tables: warmup.requested_tables,
            entry_count,
            warmed_count,
            refreshed_count,
            already_resident_count,
            skipped_count,
            error_count,
            route_ready_count: route_ready_tables.len(),
            route_blocked_count: route_blockers.len(),
            route_ready_tables,
            route_blockers,
            entries: warmup.entries,
        }
    }

    fn warmup_route_readiness_decision(
        &mut self,
        table: &str,
    ) -> Option<RelationalResidentRouteDecisionStatus> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        let Ok(Command::Select(select)) = parse_command(&sql) else {
            return None;
        };
        Some(self.plan_relational_resident_route(&select))
    }

    fn relational_snapshot_cache_state(
        snapshot: &RelationalResidencySnapshot,
        memory_pressure_active: bool,
    ) -> &'static str {
        if memory_pressure_active || snapshot.invalidated_by_memory_pressure {
            "InvalidatedByMemoryPressure"
        } else if snapshot.invalidated_by_txn_id.is_some()
            || snapshot.invalidated_at_index.is_some()
        {
            "Invalidated"
        } else {
            "Valid"
        }
    }

    fn resident_route_reject(
        table: &str,
        reason: impl Into<String>,
        query_shape: impl Into<String>,
    ) -> RelationalResidentRouteDecisionStatus {
        RelationalResidentRouteDecisionStatus {
            table: table.to_string(),
            gpu_id: None,
            snapshot_generation: None,
            shard_count: 0,
            accepted: false,
            reason: reason.into(),
            query_shape: query_shape.into(),
            cache_state: "Absent".to_string(),
            valid: false,
            has_retained_device_memory: false,
            estimated_rows: 0,
            resident_bytes: 0,
            budget_bytes: None,
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: 0,
            d2h_bytes_estimate: 0,
            d2h_rows_estimate: 0,
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        }
    }

    pub fn plan_relational_resident_route(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        let decision = self.plan_relational_resident_route_inner(select);
        self.read_state
            .route_telemetry
            .record_route_decision(decision.clone());
        decision
    }

    fn plan_relational_resident_route_inner(
        &self,
        select: &Select,
    ) -> RelationalResidentRouteDecisionStatus {
        // Lock-free read path (Stage 2 — blocker #1): pin the catalog snapshot for the relation-kind
        // check (the subsequent table bind pins its own; both are immutable published snapshots).
        let catalog = self.catalog_snapshot();
        if catalog.relational_views.contains_key(&select.table)
            || catalog
                .relational_materialized_views
                .contains_key(&select.table)
        {
            return Self::resident_route_reject(
                &select.table,
                "resident routing currently supports only public base tables",
                "unsupported_relation_kind",
            );
        }

        let (table, bound, _copin_s) = match self.bind_relational_select_for_execution(select) {
            Ok(bound) => bound,
            Err(err) => {
                return Self::resident_route_reject(
                    &select.table,
                    format!("unsupported select shape: {err}"),
                    "unsupported_select",
                );
            }
        };

        // Stage 3 — blocker #2: pin the published resident shard + snapshot maps for the rest of
        // the planning decision (the shard slice is passed by reference into the sharded-route
        // planner, and the snapshot is read field-by-field below — both must outlive those uses, so the
        // guards are bound here and held to the end of the function).
        let shards_guard = self.read_state.residency.shards.load();
        let snapshots_guard = self.read_state.residency.snapshots.load();

        let query_shape = match resident_route_query_shape(select, &table, &bound) {
            Some(shape) => shape,
            None => {
                if let Some(shards) = shards_guard.get(&table.name) {
                    if let Some(shape) =
                        sharded_resident_route_query_shape(select, &table, &bound)
                    {
                        return self.plan_relational_sharded_resident_route(
                            select, &table, shape, shards,
                        );
                    }
                }
                return Self::resident_route_reject(
                    &table.name,
                    "resident routing has no retained-kernel proof for this SELECT shape",
                    "unsupported_select",
                );
            }
        };

        if let Some(shards) = shards_guard.get(&table.name) {
            return self.plan_relational_sharded_resident_route(
                select,
                &table,
                query_shape,
                shards,
            );
        }

        let Some(entry) = snapshots_guard.get(&table.name) else {
            return Self::resident_route_reject(
                &table.name,
                "relation has no resident snapshot",
                query_shape,
            );
        };
        let snapshot = &entry.descriptor;
        let memory_pressure_active = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .contains(&snapshot.gpu_id);
        let cache_state = Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
        let valid = snapshot.invalidated_by_txn_id.is_none()
            && snapshot.invalidated_at_index.is_none()
            && !snapshot.invalidated_by_memory_pressure
            && !memory_pressure_active;
        let has_retained_device_memory = self
            .read_state
            .residency
            .device_memory
            .contains_key(&table.name);
        let d2h_bytes_estimate = resident_route_d2h_bytes_estimate(select, &query_shape, snapshot);
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id: Some(snapshot.gpu_id),
            snapshot_generation: Some(snapshot.generation),
            shard_count: 1,
            accepted: false,
            reason: String::new(),
            query_shape,
            cache_state: cache_state.to_string(),
            valid,
            has_retained_device_memory,
            estimated_rows: snapshot.row_count,
            resident_bytes: snapshot.resident_bytes,
            budget_bytes: snapshot.admission_budget_bytes,
            refresh_resident_bytes: snapshot
                .last_refresh_cost
                .as_ref()
                .map(|cost| cost.refreshed_resident_bytes),
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: snapshot.resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, snapshot.row_count),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if snapshot.schema != table.schema || snapshot.table != table.name {
            decision.reason =
                "resident snapshot no longer matches catalog table identity".to_string();
        } else if !valid {
            decision.reason = format!("resident snapshot is {cache_state}");
        } else if !has_retained_device_memory {
            decision.reason = "resident snapshot has no retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "resident route accepted".to_string();
        }
        decision
    }

    fn plan_relational_sharded_resident_route(
        &self,
        select: &Select,
        table: &RelationalTable,
        query_shape: String,
        shards: &[RelationalResidentShard],
    ) -> RelationalResidentRouteDecisionStatus {
        let total_rows = shards
            .iter()
            .map(|shard| shard.row_count)
            .sum::<usize>();
        let total_resident_bytes = shards
            .iter()
            .map(|shard| shard.resident_bytes)
            .sum::<u64>();
        let gpu_id = shards.first().map(|shard| shard.gpu_id);
        let sharded_query_shape = if query_shape == "count_all" {
            "sharded_count_all".to_string()
        } else if query_shape == "int4_equality_projection" {
            "sharded_int4_equality_projection".to_string()
        } else if query_shape == "int4_equality_multi_column_projection" {
            "sharded_int4_equality_multi_column_projection".to_string()
        } else if matches!(
            query_shape.as_str(),
            "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            query_shape
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Avg { .. })
        {
            "sharded_int4_filtered_avg".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Min { .. })
        {
            "sharded_int4_filtered_min".to_string()
        } else if query_shape == "int4_filtered_scalar_aggregate"
            && matches!(select.projection, SelectProjection::Max { .. })
        {
            "sharded_int4_filtered_max".to_string()
        } else if query_shape == "int4_distinct_projection" {
            "sharded_int4_distinct_projection".to_string()
        } else if query_shape == "int4_filtered_distinct_projection" {
            "sharded_int4_filtered_distinct_projection".to_string()
        } else if query_shape == "int4_grouped_aggregate" {
            "sharded_int4_grouped_aggregate".to_string()
        } else if query_shape == "int4_filtered_grouped_aggregate" {
            "sharded_int4_filtered_grouped_aggregate".to_string()
        } else if query_shape == "int4_ordered_projection" {
            "sharded_int4_ordered_projection".to_string()
        } else {
            query_shape
        };
        let d2h_bytes_estimate = if matches!(
            sharded_query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
        ) {
            shards
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX)
        } else if sharded_query_shape == "sharded_int4_equality_multi_column_projection" {
            let SelectProjection::Columns(columns) = &select.projection else {
                return Self::resident_route_reject(
                    &table.name,
                    "sharded resident routing has no retained-kernel proof for this SELECT shape",
                    sharded_query_shape,
                );
            };
            u64::try_from(total_rows)
                .unwrap_or(u64::MAX)
                .saturating_mul(
                    u64::try_from(columns.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<i32>() as u64)
                        .saturating_add(std::mem::size_of::<u64>() as u64),
                )
                .saturating_add(
                    u64::try_from(shards.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(std::mem::size_of::<u64>() as u64),
                )
        } else {
            0
        };
        let mut decision = RelationalResidentRouteDecisionStatus {
            table: table.name.clone(),
            gpu_id,
            snapshot_generation: None,
            shard_count: shards.len(),
            accepted: false,
            reason: String::new(),
            query_shape: sharded_query_shape,
            cache_state: "Valid".to_string(),
            valid: true,
            has_retained_device_memory: false,
            estimated_rows: total_rows,
            resident_bytes: total_resident_bytes,
            budget_bytes: gpu_id.and_then(|gpu_id| self.relational_residency_budget_bytes(gpu_id)),
            refresh_resident_bytes: None,
            h2d_bytes_if_resident: 0,
            h2d_bytes_if_cold: total_resident_bytes,
            d2h_bytes_estimate,
            d2h_rows_estimate: resident_route_d2h_rows_estimate(select, total_rows),
            last_execution_h2d_bytes: None,
            last_execution_d2h_bytes: None,
            last_execution_kernel_samples: None,
            last_execution_kernel_ms: None,
            last_execution_kernel_event_elapsed_us: None,
            last_execution_rows: None,
            last_execution_wall_micros: None,
            last_execution_device_lookup_micros: None,
            last_execution_match_index_micros: None,
            last_execution_selected_projection_micros: None,
            last_execution_result_materialization_micros: None,
            last_execution_matched_rows: None,
        };

        if !matches!(
            decision.query_shape.as_str(),
            "sharded_count_all"
                | "sharded_int4_equality_projection"
                | "sharded_int4_equality_multi_column_projection"
                | "sharded_int4_equality_sum"
                | "sharded_int4_between_avg"
                | "sharded_int4_filtered_avg"
                | "sharded_int4_filtered_min"
                | "sharded_int4_filtered_max"
                | "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_grouped_aggregate"
                | "sharded_int4_filtered_grouped_aggregate"
                | "sharded_int4_ordered_projection"
        ) {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason =
                "sharded resident routing currently supports only unfiltered COUNT(*), same-column int4 equality projection, int4 equality multi-column projection, int4 equality SUM, int4 BETWEEN AVG, int4 filtered AVG, int4 filtered MIN, int4 filtered MAX, int4 [filtered] DISTINCT projection, int4 [filtered] grouped aggregate, and int4 ordered projection"
                    .to_string();
            return decision;
        }
        let mut required_int4_columns = BTreeSet::new();
        if decision.query_shape == "sharded_int4_equality_multi_column_projection" {
            let SelectProjection::Columns(columns) = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires projected columns".to_string();
                return decision;
            };
            for column in columns {
                required_int4_columns.insert(column.clone());
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_equality_sum" {
            let SelectProjection::Sum { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires SUM(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_between_avg" | "sharded_int4_filtered_avg"
        ) {
            let SelectProjection::Avg { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires AVG(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_min" {
            let SelectProjection::Min { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires MIN(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if decision.query_shape == "sharded_int4_filtered_max" {
            let SelectProjection::Max { column } = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires MAX(int4_column)".to_string();
                return decision;
            };
            required_int4_columns.insert(column.clone());
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_grouped_aggregate" | "sharded_int4_filtered_grouped_aggregate"
        ) {
            // Grouped int4 aggregate (S10c slice 2b): the per-shard layout check must cover both the
            // GROUP BY key column AND the aggregated value column, plus every filter column. Mirror the
            // route classifier's group/value extraction (resident_route.rs `resident_route_query_shape`):
            // GroupedCount groups by `column` and counts it; the other grouped projections carry an
            // explicit group/value column pair.
            match &select.projection {
                SelectProjection::GroupedCount { column } => {
                    required_int4_columns.insert(column.clone());
                }
                SelectProjection::GroupedSum {
                    group_column,
                    sum_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(sum_column.clone());
                }
                SelectProjection::GroupedAvg {
                    group_column,
                    avg_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(avg_column.clone());
                }
                SelectProjection::GroupedMin {
                    group_column,
                    min_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(min_column.clone());
                }
                SelectProjection::GroupedMax {
                    group_column,
                    max_column,
                } => {
                    required_int4_columns.insert(group_column.clone());
                    required_int4_columns.insert(max_column.clone());
                }
                _ => {
                    decision.cache_state = "Absent".to_string();
                    decision.valid = false;
                    decision.reason =
                        "sharded resident routing requires a grouped int4 aggregate projection"
                            .to_string();
                    return decision;
                }
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        } else if matches!(
            decision.query_shape.as_str(),
            "sharded_int4_distinct_projection"
                | "sharded_int4_filtered_distinct_projection"
                | "sharded_int4_ordered_projection"
        ) {
            // Single-column DISTINCT / ordered int4 projection (S10c slice 2b): the per-shard layout
            // check must cover the single projected/distinct column plus every filter column. The route
            // classifier accepts only a single projected column for these shapes.
            let SelectProjection::Columns(columns) = &select.projection else {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason =
                    "sharded resident routing requires a single projected int4 column".to_string();
                return decision;
            };
            for column in columns {
                required_int4_columns.insert(column.clone());
            }
            if let Some(filter) = &select.filter {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in &select.filters {
                required_int4_columns.insert(filter.column.clone());
            }
            for filter in select.filter_groups.iter().flatten() {
                required_int4_columns.insert(filter.column.clone());
            }
        }
        if shards.is_empty() {
            decision.cache_state = "Absent".to_string();
            decision.valid = false;
            decision.reason = "relation has no resident shards".to_string();
            return decision;
        }

        let mut has_all_device_memory = true;
        for shard in shards {
            let memory_pressure_active = self
                .router
                .runtime()
                .snapshot()
                .memory_pressured_gpu_ids
                .contains(&shard.gpu_id);
            let valid = shard.is_valid(memory_pressure_active);
            decision.valid &= valid;
            if memory_pressure_active || shard.invalidated_by_memory_pressure {
                decision.cache_state = "InvalidatedByMemoryPressure".to_string();
            } else if shard.invalidated_by_txn_id.is_some()
                || shard.invalidated_at_index.is_some()
            {
                decision.cache_state = "Invalidated".to_string();
            }
            if shard.schema != table.schema || shard.table != table.name {
                decision.reason =
                    "resident shard no longer matches catalog table identity".to_string();
                return decision;
            }
            if !self
                .read_state
                .residency
                .shard_device_memory
                .contains_key(&(table.name.clone(), shard.shard_id))
            {
                has_all_device_memory = false;
            }
            if !required_int4_columns.is_empty()
                && required_int4_columns
                    .iter()
                    .any(|column| !shard.resident_device_int4_columns.contains(column))
            {
                decision.cache_state = "Absent".to_string();
                decision.valid = false;
                decision.reason = format!(
                    "resident shard {} lacks required int4 projection layout",
                    shard.shard_id
                );
                return decision;
            }
        }
        decision.has_retained_device_memory = has_all_device_memory;
        if !decision.valid {
            decision.reason = format!("resident shard set is {}", decision.cache_state);
        } else if !has_all_device_memory {
            decision.reason =
                "resident shard set has missing retained device memory".to_string();
        } else {
            decision.accepted = true;
            decision.reason = "sharded resident route accepted".to_string();
        }
        decision
    }

    pub(crate) fn relational_residency_status(&self) -> RelationalResidencyStatus {
        // Stage 3 — blocker #2: iterate a pinned snapshot generation (the per-table `last_decision` it
        // joins to still lives on the resident cache and is read via `&self` inside the closure).
        let snapshots_guard = self.read_state.residency.snapshots.load();
        let mut tables = snapshots_guard
            .values()
            .map(|entry| {
                let snapshot = &entry.descriptor;
                let memory_pressure_active = self
                    .router
                    .runtime()
                    .snapshot()
                    .memory_pressured_gpu_ids
                    .contains(&snapshot.gpu_id);
                let last_decision = self
                    .ddl_catalog()
                    .relational_resident_cache
                    .last_decision(&snapshot.table)
                    .cloned();
                let last_decision = last_decision.as_ref();
                let cache_state =
                    Self::relational_snapshot_cache_state(snapshot, memory_pressure_active);
                RelationalResidencyTableStatus {
                    schema: snapshot.schema.clone(),
                    table: snapshot.table.clone(),
                    gpu_id: snapshot.gpu_id,
                    snapshot_generation: snapshot.generation,
                    cache_state: cache_state.to_string(),
                    row_count: snapshot.row_count,
                    column_count: snapshot.column_count,
                    resident_bytes: snapshot.resident_bytes,
                    valid_through_index: snapshot.valid_through_index,
                    valid: snapshot.invalidated_by_txn_id.is_none()
                        && snapshot.invalidated_at_index.is_none()
                        && !snapshot.invalidated_by_memory_pressure
                        && !memory_pressure_active,
                    invalidated_by_txn_id: snapshot.invalidated_by_txn_id,
                    invalidated_at_index: snapshot.invalidated_at_index,
                    invalidated_by_memory_pressure: snapshot.invalidated_by_memory_pressure,
                    memory_pressure_active,
                    admission_budget_bytes: snapshot.admission_budget_bytes,
                    resident_bytes_after_admission: snapshot.resident_bytes_after_admission,
                    evicted_tables_on_admission: snapshot.evicted_tables_on_admission.clone(),
                    last_decision_accepted: last_decision.map(|decision| decision.accepted),
                    last_decision_reason: last_decision.map(|decision| decision.reason.clone()),
                    last_decision_current_bytes_before: last_decision
                        .map(|decision| decision.current_bytes_before),
                    last_decision_current_bytes_after: last_decision
                        .map(|decision| decision.current_bytes_after),
                    device_memory_proof: snapshot.device_memory_proof.clone(),
                }
            })
            .collect::<Vec<_>>();
        tables.sort_by(|left, right| {
            left.gpu_id
                .cmp(&right.gpu_id)
                .then_with(|| left.schema.cmp(&right.schema))
                .then_with(|| left.table.cmp(&right.table))
        });

        let mut resident_bytes_by_gpu = BTreeMap::new();
        for table in &tables {
            *resident_bytes_by_gpu.entry(table.gpu_id).or_insert(0) += table.resident_bytes;
        }

        RelationalResidencyStatus {
            tables,
            latest_route_decisions: self
                .read_state
                .route_telemetry
                .route_decisions()
                .values()
                .cloned()
                .collect(),
            resident_bytes_by_gpu,
            budget_bytes_by_gpu: self
                .ddl_catalog()
                .relational_resident_cache
                .budget_bytes_by_gpu
                .clone(),
        }
    }
}
