use super::*;

/// The decoded form of a v1 binary INSERT record.
pub(crate) struct BinaryInsertRecord {
    pub(crate) table: String,
    /// `(row_id, encoded_row)` — the row image in `encode_relational_row`'s cell encoding
    /// (the tuple store's canonical format; decoded against the catalog at apply time).
    pub(crate) rows: Vec<(u64, String)>,
}

/// The decoded form of a W5b by-key DELETE record (U1).
pub(crate) struct BinaryDeleteByKeyRecord {
    pub(crate) table: String,
    /// The unique key column NAME (catalog identity survives replays; the covered route
    /// guarantees it is the single unique i32 index column).
    pub(crate) pk_column: String,
    pub(crate) pk_value: i32,
}

/// The decoded form of a W5b by-key UPDATE record (U2).
pub(crate) struct BinaryUpdateByKeyRecord {
    pub(crate) table: String,
    pub(crate) pk_column: String,
    pub(crate) pk_value: i32,
    /// Legacy v1 allocator reservation. ADR-014 replay derives replacement identity from the
    /// visible old version; this value is retained for framing and allocator high-water parity.
    pub(crate) new_row_id: u64,
    /// The new row image in `encode_relational_row`'s cell encoding (all columns, new values).
    pub(crate) new_row_encoded: String,
}

/// Encode a W5b by-key DELETE record. `None` on width-exceeding names (caller falls back to the
/// SQL-text record, same contract as the insert encoder).
pub(crate) fn encode_binary_delete_by_key(
    table: &str,
    pk_column: &str,
    pk_value: i32,
) -> Option<Vec<u8>> {
    if table.len() > u16::MAX as usize || pk_column.len() > u16::MAX as usize {
        return None;
    }
    let mut out = Vec::with_capacity(3 + 2 + table.len() + 2 + pk_column.len() + 4);
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    out.push(OP_DELETE_BY_KEY);
    out.extend_from_slice(&(table.len() as u16).to_le_bytes());
    out.extend_from_slice(table.as_bytes());
    out.extend_from_slice(&(pk_column.len() as u16).to_le_bytes());
    out.extend_from_slice(pk_column.as_bytes());
    out.extend_from_slice(&pk_value.to_le_bytes());
    Some(out)
}

/// Encode a W5b by-key UPDATE record (U2): table + pk column/value + the new version's row id +
/// the new row image (all columns). `None` on width-exceeding shapes (caller falls back to SQL
/// text). `new_row` is the full post-image in catalog order.
pub(crate) fn encode_binary_update_by_key(
    table: &str,
    pk_column: &str,
    pk_value: i32,
    new_row_id: u64,
    new_row: &[SqlValue],
) -> Option<Vec<u8>> {
    if table.len() > u16::MAX as usize || pk_column.len() > u16::MAX as usize {
        return None;
    }
    let encoded = encode_relational_row(new_row);
    let encoded_bytes = encoded.as_bytes();
    if encoded_bytes.len() > u32::MAX as usize {
        return None;
    }
    let mut out = Vec::with_capacity(
        3 + 2 + table.len() + 2 + pk_column.len() + 4 + 8 + 4 + encoded_bytes.len(),
    );
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    out.push(OP_UPDATE_BY_KEY);
    out.extend_from_slice(&(table.len() as u16).to_le_bytes());
    out.extend_from_slice(table.as_bytes());
    out.extend_from_slice(&(pk_column.len() as u16).to_le_bytes());
    out.extend_from_slice(pk_column.as_bytes());
    out.extend_from_slice(&pk_value.to_le_bytes());
    out.extend_from_slice(&new_row_id.to_le_bytes());
    out.extend_from_slice(&(encoded_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(encoded_bytes);
    Some(out)
}

/// U2: the byte offset of the 8-byte `new_row_id` field inside an `OP_UPDATE_BY_KEY` record for
/// `(table, pk_column)` — the pump patches the row id it claims at wave formation here. Layout:
/// tag(1) + version(1) + op(1) + table_len(2) + table + pkcol_len(2) + pkcol + pk_value(4), then
/// the `new_row_id`.
pub(crate) fn binary_update_new_row_id_offset(table: &str, pk_column: &str) -> usize {
    3 + 2 + table.len() + 2 + pk_column.len() + 4
}

/// Encode a covered INSERT delta as a v1 binary record. `rows` are `(row_id, values)`.
pub(crate) fn try_encode_binary_insert(
    table: &str,
    rows: &[(u64, &[SqlValue])],
) -> Option<Vec<u8>> {
    if table.len() > u16::MAX as usize || rows.len() > u32::MAX as usize {
        return None;
    }
    let mut out = encode_binary_insert_unchecked(table, rows)?;
    out.shrink_to_fit();
    Some(out)
}

fn encode_binary_insert_unchecked(table: &str, rows: &[(u64, &[SqlValue])]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(64 + rows.len() * 48);
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    out.push(OP_INSERT);
    let table_bytes = table.as_bytes();
    out.extend_from_slice(&(table_bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(table_bytes);
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for (row_id, values) in rows {
        out.extend_from_slice(&row_id.to_le_bytes());
        let encoded = encode_relational_row(values);
        let encoded_bytes = encoded.as_bytes();
        if encoded_bytes.len() > u32::MAX as usize {
            return None;
        }
        out.extend_from_slice(&(encoded_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(encoded_bytes);
    }
    Some(out)
}

/// Whether `payload` is a binary WAL record (any version).
pub(crate) fn is_binary_wal_record(payload: &[u8]) -> bool {
    payload.first() == Some(&WAL_BINARY_TAG)
}

/// Decode a v1 binary INSERT. Tagged corruption or version skew fails loudly.
pub(crate) fn decode_binary_insert(payload: &[u8]) -> Result<BinaryInsertRecord, EngineError> {
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], EngineError> {
        let end = at.checked_add(n).ok_or_else(|| fail("length overflow"))?;
        let slice = payload.get(at..end).ok_or_else(|| fail("truncated"))?;
        at = end;
        Ok(slice)
    };
    if take(1)?[0] != WAL_BINARY_TAG {
        return Err(fail("missing tag"));
    }
    let version = take(1)?[0];
    if version != WAL_BINARY_VERSION {
        return Err(fail(&format!("unsupported version {version}")));
    }
    if take(1)?[0] != OP_INSERT {
        return Err(fail("unsupported op"));
    }
    let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
    let table = std::str::from_utf8(take(table_len)?)
        .map_err(|_| fail("non-utf8 table name"))?
        .to_string();
    let row_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut rows = Vec::with_capacity(row_count.min(64 * 1024));
    for _ in 0..row_count {
        let row_id = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let encoded_len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let encoded = std::str::from_utf8(take(encoded_len)?)
            .map_err(|_| fail("non-utf8 row encoding"))?
            .to_string();
        rows.push((row_id, encoded));
    }
    if at != payload.len() {
        return Err(fail("trailing bytes"));
    }
    Ok(BinaryInsertRecord { table, rows })
}
