//! W5a — BINARY row-op WAL records (the resolved-change-record format, assessment D5/R3).
//!
//! The WAL was a logical SQL-text log: replay re-parsed and re-executed every statement
//! (~30µs/record), the sequencer's covered INSERTs re-serialized their statement text, and every
//! checkpoint carried the full SQL. A binary record carries the RESOLVED mutation instead — the
//! table, the row ids, and the row images in the tuple store's canonical cell encoding — so
//! replay is decode + install (no parse, no re-resolve, no validation re-run: the record exists
//! only because the original commit validated it).
//!
//! FRAMING: the record rides the existing WAL framing (txn_id / len / FNV checksum) as an opaque
//! payload. `payload[0] == 0xFF` marks a binary record — 0xFF is an invalid UTF-8 leading byte,
//! so every SQL-text consumer's defensive `from_utf8 -> skip` arm (KvStateMachine, telemetry)
//! ignores binary records WITHOUT modification; the consumers that must apply them
//! (`apply_mvcc_entry`, `residency_invalidation_scope`) dispatch on the tag explicitly BEFORE
//! their UTF-8 checks. SQL text can never collide (it starts with printable ASCII).
//!
//! v1 (W5a) scope: the COVERED single-table INSERT class (the delta-reuse class: FK/CHECK-free,
//! no sequence defaults, unique-slots claimed via the ledger) — exactly the records whose apply
//! is a pure install. Everything else stays SQL text. UPDATE/DELETE follow in W5b.

use super::*;

/// Binary-record lead byte (invalid UTF-8 on purpose — see the module doc).
pub(crate) const WAL_BINARY_TAG: u8 = 0xFF;
/// Format version for forward evolution; bump on layout change.
const WAL_BINARY_VERSION: u8 = 1;
/// Op codes.
const OP_INSERT: u8 = 1;
/// U1 (W5b): a covered lane DELETE, logged BY KEY — replay re-resolves the key against the
/// replayed state (deterministic: all ops on a key are lane-serialized in seq order, cross-key
/// ops commute), so the record needs no resolved tuple/row ids. 0-row deletes never reach the
/// WAL (the pump's pre-claim filter completes them without a seq), so a durable delete record
/// re-resolving to 0 rows at replay is corruption-or-determinism-bug — the apply arm is LOUD.
const OP_DELETE_BY_KEY: u8 = 2;

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

/// A decoded binary WAL record of any op (the tag dispatch for apply/replay consumers).
pub(crate) enum BinaryWalRecord {
    Insert(BinaryInsertRecord),
    DeleteByKey(BinaryDeleteByKeyRecord),
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

/// Decode ANY binary record (op dispatch). Errors are LOUD (`Durability`) — a tagged record
/// that fails to decode is corruption-or-version-skew, never silently skipped.
pub(crate) fn decode_binary_record(payload: &[u8]) -> Result<BinaryWalRecord, EngineError> {
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    match payload.get(2) {
        Some(&OP_INSERT) => decode_binary_insert(payload).map(BinaryWalRecord::Insert),
        Some(&OP_DELETE_BY_KEY) => {
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
            if take(1)?[0] != OP_DELETE_BY_KEY {
                return Err(fail("op dispatch mismatch"));
            }
            let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(table_len)?)
                .map_err(|_| fail("non-utf8 table name"))?
                .to_string();
            let column_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let pk_column = std::str::from_utf8(take(column_len)?)
                .map_err(|_| fail("non-utf8 column name"))?
                .to_string();
            let pk_value = i32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if at != payload.len() {
                return Err(fail("trailing bytes"));
            }
            Ok(BinaryWalRecord::DeleteByKey(BinaryDeleteByKeyRecord {
                table,
                pk_column,
                pk_value,
            }))
        }
        Some(op) => Err(fail(&format!("unsupported op {op}"))),
        None => Err(fail("truncated header")),
    }
}

/// Encode a covered INSERT delta as a v1 binary record. `rows` are `(row_id, values)`.
/// Returns `None` on width-exceeding shapes (table name > u16, cell encoding > u32 — no
/// realistic statement, but the durability path must not silently wrap; the caller falls back
/// to the SQL-text record).
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

/// Decode a v1 binary record. Errors are LOUD (`Durability`): a tagged record that fails to
/// decode is corruption-or-version-skew — never silently skipped (the silent-skip discipline is
/// only for the UNTAGGED text consumers).
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
    // Capacity clamp (audit 21eddaa7 C): don't trust the record's count for the allocation —
    // a corrupt count fails loudly at `take` anyway; the clamp keeps the allocation bounded.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w5a_binary_insert_round_trips() {
        let rows_owned = [
            (
                7_u64,
                vec![SqlValue::Int4(1), SqlValue::Text("a|b\\c".into())],
            ),
            (8_u64, vec![SqlValue::Int4(2), SqlValue::Null]),
        ];
        let rows: Vec<(u64, &[SqlValue])> = rows_owned
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        let payload = try_encode_binary_insert("public_t", &rows).unwrap();
        assert!(is_binary_wal_record(&payload));
        assert!(
            std::str::from_utf8(&payload).is_err(),
            "0xFF tag must break UTF-8"
        );
        let decoded = decode_binary_insert(&payload).unwrap();
        assert_eq!(decoded.table, "public_t");
        assert_eq!(decoded.rows.len(), 2);
        assert_eq!(decoded.rows[0].0, 7);
        assert_eq!(decoded.rows[0].1, encode_relational_row(&rows_owned[0].1));
    }

    #[test]
    fn w5a_truncated_and_skewed_records_fail_loudly() {
        let payload = try_encode_binary_insert("t", &[(1, &[SqlValue::Int4(5)])]).unwrap();
        assert!(decode_binary_insert(&payload[..payload.len() - 1]).is_err());
        let mut skewed = payload.clone();
        skewed[1] = 99; // version
        assert!(decode_binary_insert(&skewed).is_err());
    }
}
