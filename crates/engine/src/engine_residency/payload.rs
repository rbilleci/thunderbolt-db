//! Typed resident payload/key encoding and open-shard append construction.

use super::*;

pub(crate) type RelationalDevicePayload = (
    Vec<u8>,
    Vec<ResidentDeviceTextColumnLayout>,
    Vec<ResidentDeviceBoolColumnLayout>,
    Vec<ResidentDeviceInt4ColumnStats>,
    Vec<(String, u64)>,
    Vec<ResidentDeviceNullBitmapLayout>,
);

/// Cohesive descriptor inputs for a device-recompacted unified shard snapshot.
pub(crate) struct UnifiedResidentSnapshotParts {
    pub(crate) total_row_count: usize,
    pub(crate) gpu_id: u16,
    pub(crate) resident_bytes: u64,
    pub(crate) proof: CudaDeviceMemoryProof,
    pub(crate) int4_columns: Vec<String>,
    pub(crate) int8_columns: Vec<String>,
    pub(crate) numeric_columns: Vec<String>,
    pub(crate) bool_columns: Vec<ResidentDeviceBoolColumnLayout>,
    pub(crate) text_columns: Vec<ResidentDeviceTextColumnLayout>,
    pub(crate) null_columns: Vec<ResidentDeviceNullBitmapLayout>,
}

/// Build the GPU device payload for typed columns + their row values -- the columnar
/// `[8-byte row_count header][int4/date/int2 i32][int8/timestamp i64][numeric/uuid 16B][bool bitmap]
/// [text 8-aligned offsets + bytes]` layout (type-grouped, catalog order within each type; varlen text
/// offsets 8-aligned per the CUDA-716 lesson). Returns the payload (row_count header filled, NO MVCC
/// tail) + the text/bool column layouts + per-int4-column min/max -- the SAME bytes/offsets the
/// resident-table builder produces, so a non-table caller (e.g. the grouped-sort) can build a
/// resident-like buffer without re-implementing the byte mappings. `column_names` / `column_types` /
/// each row in `rows` are parallel by column index.
pub(crate) fn build_relational_device_payload(
    column_names: &[String],
    column_types: &[SqlType],
    rows: &[Vec<SqlValue>],
) -> Result<RelationalDevicePayload, ExecuteError> {
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
) -> Result<RelationalDevicePayload, ExecuteError> {
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
        if !rows
            .iter()
            .any(|row| matches!(row[col_idx], SqlValue::Null))
        {
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
    device_payload[..std::mem::size_of::<u64>()].copy_from_slice(&(row_count as u64).to_le_bytes());
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

/// TYPE-COVERAGE track 2 (Date/Int2): the typed DECODE inverse of `sql_value_as_int4` for the
/// i32-section types — the A4a materializer / A4c gather previously DECLINED any non-strictly-
/// Int4 table because they typed every value `SqlValue::Int4` (the A4a audit-F1 mistype
/// discipline); deriving the variant from the CATALOG column type lifts that. `None` for any
/// non-i32-section type (the caller declines to the host path) and for an out-of-range Int2
/// payload (corrupt section bytes must DECLINE, never silently truncate).
pub(crate) fn sql_value_from_i32_section(ty: gpu_db_sql::SqlType, v: i32) -> Option<SqlValue> {
    match ty {
        gpu_db_sql::SqlType::Int4 => Some(SqlValue::Int4(v)),
        gpu_db_sql::SqlType::Date => Some(SqlValue::Date(v)),
        gpu_db_sql::SqlType::Int2 => i16::try_from(v).ok().map(SqlValue::Int2),
        _ => None,
    }
}

/// TYPE-COVERAGE track 2 slice 2 stage (iii): the typed DECODE for the i64 section
/// (Int8/Timestamp) — the A4a materializer / A4c gather read these columns as two u32 halves
/// (the 4-mod-8 discipline) and type the value from the CATALOG. `None` for any other type.
pub(crate) fn sql_value_from_i64_section(ty: gpu_db_sql::SqlType, v: i64) -> Option<SqlValue> {
    match ty {
        gpu_db_sql::SqlType::Int8 => Some(SqlValue::Int8(v)),
        gpu_db_sql::SqlType::Timestamp => Some(SqlValue::Timestamp(v)),
        _ => None,
    }
}

/// TYPE-COVERAGE track 2: encode an equality NEEDLE for the device i32-section probe, requiring
/// the value VARIANT to agree with the column's catalog type (Int4->Int4, Date->Date,
/// Int2->Int2). Strict agreement is load-bearing for the A3 probe's authoritative FALSE (a
/// loosely-coerced needle could encode to bytes the section never stores -> false MISS =
/// constraint hole); the section bytes were written by `sql_value_as_int4` from values of the
/// column's own type, so the exact-variant encode is exact.
pub(crate) fn i32_section_needle(ty: gpu_db_sql::SqlType, value: &SqlValue) -> Option<i32> {
    match (ty, value) {
        (gpu_db_sql::SqlType::Int4, SqlValue::Int4(v)) => Some(*v),
        (gpu_db_sql::SqlType::Date, SqlValue::Date(v)) => Some(*v),
        (gpu_db_sql::SqlType::Int2, SqlValue::Int2(v)) => Some(i32::from(*v)),
        _ => None,
    }
}

/// COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): the device i32 PK index stores/probes a single 32-bit
/// key per row. A compound `PRIMARY KEY (a, b, ...)` (or compound `UNIQUE`) over i32-SECTION columns
/// (Int4/Date/Int2) is served by folding the ordered key-column values into ONE 32-bit SURROGATE key —
/// the fingerprint below — which then rides the EXACT single-column index machinery (host builder,
/// device insert/write-locate/visible-locate kernels, coalescer, geometric rebuild) unchanged: to the
/// device the fingerprint is just an opaque 32-bit key.
///
/// EXACTNESS is not a property of the fingerprint (distinct tuples CAN collide on 32 bits): it is
/// restored by the authoritative recheck. The device write-locate probe is already NON-authoritative —
/// any `count > 0` triggers `visible_row_with_value`, which materializes the candidate row on-device and
/// compares. For a compound key that recheck compares the FULL key TUPLE (see
/// `tuple_matches_key_columns`), so a fingerprint collision can never produce a false 23505 or a wrong
/// locate — only a rare, bounded extra recheck. Collision frequency is bounded per shard by the shard
/// floor (a probe is already O(shards); the recheck adds a constant factor, not a new scalability class).
///
/// The fold is ORDER-SENSITIVE (the key-column order is part of the tuple identity). It runs on the HOST
/// for probe needles + the incremental append, and ON THE DEVICE for the index rebuild
/// (`COMPOUND_FOLD_PTX` / `sql_value_key_words`, which is BYTE-IDENTICAL to this function — that
/// host/device agreement is the load-bearing invariant a divergence would break). `key == 0` is fine: the
/// slot packs `(key<<32)|(row+1)` and `row+1 >= 1`, so a packed slot is never the empty-slot sentinel 0.
pub(crate) fn compound_key_fingerprint(vals: &[i32]) -> i32 {
    let mut h: u32 = 0x811C_9DC5; // FNV-1a offset basis
    for &v in vals {
        h ^= v as u32;
        h = h.wrapping_mul(0x0100_0193); // FNV prime
        h = h.rotate_left(13).wrapping_add(0x9E37_79B1); // extra avalanche + Fibonacci constant
    }
    h as i32
}

/// FINGERPRINT INDEXES: the device-probe "key id" that identifies WHICH unique index a probe/gather
/// targets. A raw single-column i32-section index keeps its catalog COLUMN INDEX verbatim
/// (byte-compatible with every existing cache entry and offset computation). A fingerprint-backed index
/// encodes `FLAG | ordinal` where `ordinal` is the index's position in `table.indexes` — a small,
/// per-index-UNIQUE, collision-free discriminator. This includes compound indexes and single-column
/// wider/text indexes. A probabilistic hash of the column set could alias two indexes and silently serve
/// the wrong index buffer = a MISSED-duplicate correctness bug, so the ordinal is used, not a hash. The
/// flag bit (the top usize bit) can never collide with a real column index (`< usize::MAX >> 1`).
pub(crate) const COMPOUND_KEY_ID_FLAG: usize = 1_usize << (usize::BITS - 1);

/// COMPOUND KEYS: `true` when `index` spans more than one key column.
pub(crate) fn index_is_compound(index: &RelationalIndex) -> bool {
    index.key_columns.len() > 1
}

/// R3-002: `true` when the device index stores the canonical 32-bit fingerprint rather than one
/// raw i32-section value. Compound keys always use the fingerprint contract. A single wider/text key
/// joins that contract only when its resident type has an exact fold encoding; unsupported keys remain
/// outside device-index eligibility.
pub(crate) fn index_uses_fingerprint(table: &RelationalTable, index: &RelationalIndex) -> bool {
    if index_is_compound(index) {
        return true;
    }
    let Some(position) = index_key_column_positions(table, index)
        .and_then(|positions| positions.first().copied().filter(|_| positions.len() == 1))
    else {
        return false;
    };
    let ty = table.columns[position].ty;
    compound_key_type_supported(ty)
        && !matches!(
            ty,
            gpu_db_sql::SqlType::Int4 | gpu_db_sql::SqlType::Date | gpu_db_sql::SqlType::Int2
        )
}

/// COMPOUND KEYS: resolve `index.key_columns` (ordered) to their catalog column positions. `None` if any
/// named key column is absent from the table (a malformed catalog — the caller declines the fast path).
pub(crate) fn index_key_column_positions(
    table: &RelationalTable,
    index: &RelationalIndex,
) -> Option<Vec<usize>> {
    index
        .key_columns
        .iter()
        .map(|name| table.columns.iter().position(|c| &c.name == name))
        .collect()
}

/// Can `index` be validated by the DEVICE index probe? Every key column must have a canonical resident
/// word/text fold. Single i32-section keys keep the raw-key layout; compound and single wider/text keys
/// store the folded fingerprint. A key column of an unsupported type keeps the index OFF the elision path
/// (honest partial coverage).
pub(crate) fn index_all_key_columns_foldable(
    table: &RelationalTable,
    index: &RelationalIndex,
) -> bool {
    !index.key_columns.is_empty()
        && index.key_columns.iter().all(|name| {
            table
                .columns
                .iter()
                .find(|c| &c.name == name)
                .is_some_and(|c| compound_key_type_supported(c.ty))
        })
}

/// The device-probe key id for `index` (see [`COMPOUND_KEY_ID_FLAG`]). `ordinal` is the index's
/// position in `table.indexes`. Raw single-i32 -> the key column's catalog index; fingerprint-backed ->
/// `FLAG | ordinal`. Returns `None` if the raw single key column can't be resolved.
///
/// CACHE SAFETY (audit): the ordinal is also the discriminator for the per-shard PK device-index
/// cache `(table, shard_id, key_id)`, and DROP CONSTRAINT / DROP INDEX SHIFT ordinals. This is sound
/// because EVERY index-shape DDL (CREATE/DROP INDEX, ADD/DROP PRIMARY KEY / UNIQUE) is an "other DDL"
/// in `residency_invalidation_scope` -> the conservative GLOBAL `invalidate_relational_residency`,
/// which runs `invalidate_relational_residency_table` for every resident table and thereby
/// `purge_shard_pk_index_for_table` (engine_commit.rs). So no cache entry survives an ordinal shift;
/// the next probe rebuilds against the current ordinals. Regression:
/// `gpu_compound_drop_constraint_shifts_ordinal_without_aliasing_the_device_index`.
pub(crate) fn index_probe_key_id(
    table: &RelationalTable,
    index: &RelationalIndex,
    ordinal: usize,
) -> Option<usize> {
    if index_uses_fingerprint(table, index) {
        Some(COMPOUND_KEY_ID_FLAG | ordinal)
    } else {
        table.columns.iter().position(|c| c.name == index.column)
    }
}

/// Decode a device-probe `key_id` (see [`index_probe_key_id`]) back to the ordered catalog positions
/// of its key column(s). A raw single-i32 key id IS the column index (`[key_id]`); a fingerprint key id
/// (`COMPOUND_KEY_ID_FLAG | ordinal`) resolves `table.indexes[ordinal].key_columns`.
/// `None` if the ordinal / a named key column is out of range (a torn catalog -> the caller declines).
pub(crate) fn probe_key_id_positions(table: &RelationalTable, key_id: usize) -> Option<Vec<usize>> {
    if key_id & COMPOUND_KEY_ID_FLAG != 0 {
        let ordinal = key_id & !COMPOUND_KEY_ID_FLAG;
        index_key_column_positions(table, table.indexes.get(ordinal)?)
    } else {
        Some(vec![key_id])
    }
}

/// COMPOUND KEYS: fold a ROW's key-column values (catalog order in `values`) into the surrogate
/// fingerprint needle. `None` if any key column is absent or not an i32-SECTION value (the caller then
/// declines the device fast path and falls to the host validate ladder). NULL is not an i32-section
/// value, so a row with a NULL key column returns `None` — its uniqueness rides the host path (which is
/// where PK-NOT-NULL / NULL-tuple semantics live anyway).
pub(crate) fn compound_index_row_fingerprint(
    table: &RelationalTable,
    index: &RelationalIndex,
    values: &[SqlValue],
) -> Option<i32> {
    let mut words: Vec<i32> = Vec::with_capacity(index.key_columns.len());
    for name in &index.key_columns {
        let pos = table.columns.iter().position(|c| &c.name == name)?;
        words.extend(sql_value_key_words(
            table.columns[pos].ty,
            values.get(pos)?,
        )?);
    }
    Some(compound_key_fingerprint(&words))
}

/// COMPOUND KEYS (wider types, TYPE-COVERAGE #14 Track 3): the ORDERED i32 WORDS of a key column's
/// value, matching the on-device section's LITTLE-ENDIAN byte layout EXACTLY so the host fold (needle /
/// append / SI slot) and the device fold (`gpu_db_compound_fold_fingerprints`, which reads the raw
/// section words) agree byte-for-byte. Int4/Date -> `[v]`; Int2 -> `[widened v]`; Int8/Timestamp ->
/// `[low32, high32]` (the i64 section stores `value.to_le_bytes()`, read as two LE i32 words); b128
/// Numeric/Uuid -> 4 LE i32 words (the b128 section stores the i128 mantissa `to_le_bytes()` /
/// the raw uuid bytes). `None` for NULL or an unsupported key type -> the caller declines the device
/// fast path (host validates). NUMERIC scale: the section stores the mantissa RESCALED to the column's
/// scale (values are rescaled on insert), and the needle/WHERE value is coerced to the same column type
/// before folding, so the mantissa words agree; the full-tuple recheck is the exactness backstop.
pub(crate) fn sql_value_key_words(ty: gpu_db_sql::SqlType, value: &SqlValue) -> Option<Vec<i32>> {
    match (ty, value) {
        (gpu_db_sql::SqlType::Int4, SqlValue::Int4(v)) => Some(vec![*v]),
        (gpu_db_sql::SqlType::Date, SqlValue::Date(v)) => Some(vec![*v]),
        (gpu_db_sql::SqlType::Int2, SqlValue::Int2(v)) => Some(vec![i32::from(*v)]),
        (gpu_db_sql::SqlType::Int8, SqlValue::Int8(v))
        | (gpu_db_sql::SqlType::Timestamp, SqlValue::Timestamp(v)) => {
            let bits = *v as u64;
            Some(vec![bits as u32 as i32, (bits >> 32) as u32 as i32])
        }
        (gpu_db_sql::SqlType::Numeric { .. }, SqlValue::Numeric(dec)) => {
            let bits = dec.mantissa as u128;
            Some((0..4).map(|i| (bits >> (32 * i)) as u32 as i32).collect())
        }
        (gpu_db_sql::SqlType::Uuid, SqlValue::Uuid(bytes)) => Some(
            (0..4)
                .map(|i| {
                    i32::from_le_bytes([
                        bytes[4 * i],
                        bytes[4 * i + 1],
                        bytes[4 * i + 2],
                        bytes[4 * i + 3],
                    ])
                })
                .collect(),
        ),
        // TEXT (variable-length): no fixed section width, so the column folds to ONE word = the FNV-1a
        // hash of its UTF-8 bytes (`fnv1a_bytes`), which the device fold kernel computes over the resident
        // text blob byte-for-byte identically. A 32-bit collision is separated by the full-tuple recheck.
        (gpu_db_sql::SqlType::Text, SqlValue::Text(s)) => Some(vec![fnv1a_bytes(s.as_bytes())]),
        (gpu_db_sql::SqlType::Bool, SqlValue::Bool(v)) => Some(vec![if *v { 1 } else { 0 }]),
        _ => None,
    }
}

/// COMPOUND KEYS (text): the FNV-1a hash of a byte string, BYTE-IDENTICAL to the device fold kernel's
/// text branch (`h = 0x811C9DC5; per byte h ^= b; h *= 0x01000193`). A text key column folds to this
/// single word; the outer `compound_key_fingerprint` then mixes it with the other columns' words.
pub(crate) fn fnv1a_bytes(bytes: &[u8]) -> i32 {
    let mut h: u32 = 0x811C_9DC5;
    for &b in bytes {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h as i32
}

/// COMPOUND KEYS (wider types): the device fold kernel's per-column WIDTH — the number of fixed-width i32
/// WORDS (Int4/Date/Int2 -> 1 in the i32 section; Int8/Timestamp -> 2 in the i64 section; Numeric/Uuid ->
/// 4 in the b128 section), the TEXT SENTINEL 0, or the BOOL bitmap sentinel `u32::MAX`. `None` for
/// a type not supported as a fingerprint key column. A type is valid IFF this returns `Some`.
pub(crate) fn key_column_width_words(ty: gpu_db_sql::SqlType) -> Option<u32> {
    match ty {
        gpu_db_sql::SqlType::Int4 | gpu_db_sql::SqlType::Date | gpu_db_sql::SqlType::Int2 => {
            Some(1)
        }
        gpu_db_sql::SqlType::Int8 | gpu_db_sql::SqlType::Timestamp => Some(2),
        gpu_db_sql::SqlType::Numeric { .. } | gpu_db_sql::SqlType::Uuid => Some(4),
        gpu_db_sql::SqlType::Text => Some(0), // text sentinel (variable-length, hashed on-device)
        gpu_db_sql::SqlType::Bool => Some(u32::MAX), // one-bit bitmap sentinel
    }
}

/// COMPOUND KEYS: `true` when `ty` is supported as a compound key column (see [`key_column_width_words`]).
pub(crate) fn compound_key_type_supported(ty: gpu_db_sql::SqlType) -> bool {
    key_column_width_words(ty).is_some()
}

/// COMPOUND KEYS: the SI-ledger integer conflict slot id for a compound unique index. The top bit is
/// SET so it can never alias a single-column slot's `(oid << 32) | column_id` on the same table (real
/// column ids are small positive u32s). The low 31 bits fold the ordered key-column ids for per-index
/// stability; a fold collision between two compound indexes only OVER-conflicts (a safe, retryable
/// false abort), never MISSES a real conflict (same tuple -> same fingerprint -> same (slot,value) key).
pub(crate) fn compound_unique_slot_id(table: &RelationalTable, index: &RelationalIndex) -> u64 {
    let mut fold: u32 = 0x811C_9DC5;
    for name in &index.key_columns {
        if let Some(col) = table.columns.iter().find(|c| &c.name == name) {
            fold ^= col.id;
            fold = fold.wrapping_mul(0x0100_0193);
        }
    }
    crate::write_path::pack_unique_slot_id(table.oid, 0x8000_0000 | (fold & 0x7FFF_FFFF))
}

/// The uniform memset byte whose repetition is the `deleted_by` LIVE sentinel `0x7F7F_7F7F_7F7F_7F7F` — a
/// large POSITIVE signed i64 (the device visibility compare `deleted_by > read_txn_id` is a signed s64
/// kernel; `u64::MAX` would be -1 signed and a live row would wrongly fail the compare) that exceeds every
/// real commit `Index`, and is memset-friendly (uniform byte) for both the on-demand region and the SV3a
/// recompaction fill.
pub(crate) const DELETED_BY_LIVE_FILL_BYTE: u8 = 0x7F;

/// SV6: the uniform memset byte whose repetition is the `created_by` BORN-VISIBLE sentinel `0` — the device
/// lower-bound compare is `created_by <= read_txn_id` (signed s64) and every read snapshot is `>= 0`, so a
/// row without a stamp (admission-built rows, plain INSERT appends, un-versioned shards in the recompaction
/// fill) is visible to every reader. Memset-friendly (uniform `0x00`) for both the on-demand region and the
/// recompaction fill, mirroring [`DELETED_BY_LIVE_FILL_BYTE`].
/// D3 (ADR-013 pre1): the birth stamp(s) an append carries. Every append is stamped on the sharded
/// (default) layout; the variants distinguish WHY, because the single-buffer kill-switch layout —
/// which has no region machinery and sits outside the A5 gate — may append an INSERT unstamped
/// (documented born-visible semantics) but MUST decline an UPDATE's new version (SV5 P2).
pub(crate) enum AppendCreatedBy<'a> {
    /// A plain INSERT whose rows are all born at one commit seq.
    InsertUniform(Index),
    /// The wave-batched INSERT flush: one birth seq PER ROW (the batch spans multiple commits).
    InsertPerRow(&'a [Index]),
    /// An incremental UPDATE's appended new version (SV5/SV6): must stamp or decline.
    UpdateNewVersion(Index),
}

impl AppendCreatedBy<'_> {
    /// One stamp per appended row; `None` = a malformed per-row slice (caller bug -> decline).
    pub(super) fn stamps_for(&self, rows: usize) -> Option<Vec<Index>> {
        match self {
            AppendCreatedBy::InsertUniform(seq) | AppendCreatedBy::UpdateNewVersion(seq) => {
                Some(vec![*seq; rows])
            }
            AppendCreatedBy::InsertPerRow(seqs) => (seqs.len() == rows).then(|| seqs.to_vec()),
        }
    }
}

pub(crate) const CREATED_BY_VISIBLE_FILL_BYTE: u8 = 0x00;

/// RETIREMENT A1: the row-identity region's UNSTAMPED sentinel — every byte 0xFF makes the u64
/// `u64::MAX`, which no real `row_id` reaches (ids allocate monotonically from 1). A live slot
/// reading the sentinel (or a shard with NO region — benchmark/synthetic installs) means "identity
/// unknown": the device resolve declines to the host path. Headroom is born-sentinel so a skipped
/// append stamp is DETECTABLE, never a wrong identity.
pub(crate) const ROW_ID_UNSTAMPED_FILL_BYTE: u8 = 0xFF;

/// RETIREMENT A1: parse the `row_id` out of a relational row key (`rel/{table}/{row_id:020}`).
/// `None` on any malformed key -> the caller stamps the sentinel (identity unknown, never wrong).
pub(crate) fn parse_relational_row_id(key: &str, prefix: &str) -> Option<u64> {
    key.strip_prefix(prefix)?.parse::<u64>().ok()
}

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
    if !column_types.iter().all(|ty| {
        matches!(
            ty,
            SqlType::Int4
                | SqlType::Date
                | SqlType::Int2
                | SqlType::Int8
                | SqlType::Timestamp
                // TYPE-COVERAGE #14 (numeric): the b128 (Numeric/Uuid) 16-byte section.
                | SqlType::Numeric { .. }
                | SqlType::Uuid
                // TYPE-COVERAGE #14 (bool): the 1-bit/row bitmap — emits NO chunk here (the caller's
                // device atomicOr set-range op writes its bits); the encoder just skips the column.
                | SqlType::Bool
        )
    }) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "open-shard append supports fixed-width (i32/i64/b128) + bool-bitmap sections only"
                .to_string(),
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
    // TYPE-COVERAGE track 2 slice 2 stage (ii): MIXED fixed-width sections. Catalog order no
    // longer equals section ordinal - each column maps to (section, within-section ordinal):
    // i32 sections first (header + c32*capacity*4), then i64 sections
    // (header + num_i32*capacity*4 + c64*capacity*8) - the shared offset-helper formula.
    let num_i32_cols = column_types
        .iter()
        .filter(|ty| matches!(ty, SqlType::Int4 | SqlType::Date | SqlType::Int2))
        .count();
    let i64_section_base = header_bytes + num_i32_cols * capacity * std::mem::size_of::<i32>();
    // TYPE-COVERAGE #14 (numeric): the b128 section follows the i64 sections — base = i64_base +
    // num_i64*capacity*8; each b128 column strides capacity*16 (matches the payload builder).
    let num_i64_cols = column_types
        .iter()
        .filter(|ty| matches!(ty, SqlType::Int8 | SqlType::Timestamp))
        .count();
    let numeric_section_base =
        i64_section_base + num_i64_cols * capacity * std::mem::size_of::<i64>();
    // Column chunks FIRST, header LAST (the partial-failure contract: never advertise un-written rows).
    let mut chunks = Vec::with_capacity(column_types.len() + 1);
    let mut i32_ordinal = 0_usize;
    let mut i64_ordinal = 0_usize;
    let mut numeric_ordinal = 0_usize;
    for (col_idx, ty) in column_types.iter().enumerate() {
        match ty {
            SqlType::Int4 | SqlType::Date | SqlType::Int2 => {
                let width = std::mem::size_of::<i32>();
                let section_start = header_bytes + i32_ordinal * capacity * width;
                let byte_offset = (section_start + row_start * width) as u64;
                let mut bytes = Vec::with_capacity(appended * width);
                for row in new_rows {
                    let value: i32 = match row[col_idx] {
                        SqlValue::Int4(value) | SqlValue::Date(value) => value,
                        SqlValue::Int2(value) => i32::from(value),
                        // A NULL materializes as 0 (the validity bitmap, a later slice, marks the row).
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
                i32_ordinal += 1;
            }
            SqlType::Int8 | SqlType::Timestamp => {
                let width = std::mem::size_of::<i64>();
                let section_start = i64_section_base + i64_ordinal * capacity * width;
                let byte_offset = (section_start + row_start * width) as u64;
                let mut bytes = Vec::with_capacity(appended * width);
                for row in new_rows {
                    let value: i64 = match row[col_idx] {
                        SqlValue::Int8(value) | SqlValue::Timestamp(value) => value,
                        SqlValue::Null => 0,
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "open-shard i64 append encountered a non-i64 value".to_string(),
                            )))
                        }
                    };
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                chunks.push(CudaOwnedDeviceMemoryChunk { byte_offset, bytes });
                i64_ordinal += 1;
            }
            SqlType::Numeric { .. } | SqlType::Uuid => {
                // TYPE-COVERAGE #14 (numeric): the b128 (16-byte) section — numeric = i128 mantissa
                // LE, uuid = raw 16 bytes, NULL = 16 zero bytes (validity bitmap marks the row).
                // Byte-identical to the payload builder's numeric/uuid section encoding.
                let width = 16_usize;
                let section_start = numeric_section_base + numeric_ordinal * capacity * width;
                let byte_offset = (section_start + row_start * width) as u64;
                let mut bytes = Vec::with_capacity(appended * width);
                for row in new_rows {
                    match &row[col_idx] {
                        SqlValue::Numeric(value) => {
                            bytes.extend_from_slice(&value.mantissa.to_le_bytes())
                        }
                        SqlValue::Uuid(uuid) => bytes.extend_from_slice(uuid),
                        SqlValue::Null => bytes.extend_from_slice(&[0u8; 16]),
                        _ => {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                                "open-shard b128 append encountered a non-numeric/uuid value"
                                    .to_string(),
                            )))
                        }
                    }
                }
                chunks.push(CudaOwnedDeviceMemoryChunk { byte_offset, bytes });
                numeric_ordinal += 1;
            }
            // TYPE-COVERAGE #14 (bool): the bitmap is NOT a capacity-strided fixed-width chunk — its
            // bits are set by the caller's device atomicOr op (`set_bool_bitmap_range`) into the
            // pre-zeroed headroom. Emit no chunk and touch no fixed-width ordinal.
            SqlType::Bool => {}
            _ => unreachable!("the section guard above rejects non-fixed-width/bool columns"),
        }
    }
    chunks.push(CudaOwnedDeviceMemoryChunk {
        byte_offset: 0,
        bytes: (end as u64).to_le_bytes().to_vec(),
    });
    Ok(chunks)
}
