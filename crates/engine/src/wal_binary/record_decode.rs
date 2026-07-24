//! Binary WAL envelope dispatch and legacy point-mutation decoding.

use super::*;

/// Decode any binary record. Tagged decode failures are durability errors, never silent skips.
pub(crate) fn decode_binary_record(payload: &[u8]) -> Result<BinaryWalRecord, EngineError> {
    if let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(payload)? {
        let operation = envelope
            .fragments
            .iter()
            .find(|fragment| {
                fragment.kind != gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus
            })
            .ok_or_else(|| {
                EngineError::Durability(
                    "canonical binary WAL record has no operation fragment".to_string(),
                )
            })?;
        let inner = Engine::decode_engine_operation(&operation.body)?;
        return decode_binary_record(&inner);
    }
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    match payload.get(2) {
        Some(&OP_INSERT) => decode_binary_insert(payload).map(BinaryWalRecord::Insert),
        Some(&OP_TRANSACTION)
        | Some(&OP_COMPOSITE_TRANSACTION)
        | Some(&OP_TABLE_RESET_TRANSACTION)
        | Some(&OP_IDENTITY_TRANSACTION)
        | Some(&OP_IDENTITY_COMPOSITE_TRANSACTION)
        | Some(&OP_IDENTITY_TABLE_RESET_TRANSACTION)
        | Some(&OP_ORDERED_CATALOG_TRANSACTION)
        | Some(&OP_IDENTITY_ORDERED_CATALOG_TRANSACTION)
        | Some(&OP_ORDERED_CATALOG_VIEW_TRANSACTION)
        | Some(&OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION)
        | Some(&OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION)
        | Some(&OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION)
        | Some(&OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION)
        | Some(&OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION)
        | Some(&OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION)
        | Some(&OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION) => {
            decode_binary_transaction(payload).map(BinaryWalRecord::Transaction)
        }
        Some(&OP_UPDATE_BY_KEY) => {
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
            if take(1)?[0] != WAL_BINARY_VERSION {
                return Err(fail("unsupported version"));
            }
            if take(1)?[0] != OP_UPDATE_BY_KEY {
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
            let new_row_id = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
            let enc_len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let new_row_encoded = std::str::from_utf8(take(enc_len)?)
                .map_err(|_| fail("non-utf8 row encoding"))?
                .to_string();
            if at != payload.len() {
                return Err(fail("trailing bytes"));
            }
            Ok(BinaryWalRecord::UpdateByKey(BinaryUpdateByKeyRecord {
                table,
                pk_column,
                pk_value,
                new_row_id,
                new_row_encoded,
            }))
        }
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
