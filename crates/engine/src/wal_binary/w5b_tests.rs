use super::*;

/// U1: the by-key DELETE record round-trips through the op-dispatch decoder, and a
/// truncated/trailing-bytes record fails LOUDLY (never a silent skip).
#[test]
fn w5b_delete_by_key_round_trips_and_fails_loud() {
    let payload = encode_binary_delete_by_key("public_accounts", "id", -73).unwrap();
    assert!(is_binary_wal_record(&payload));
    match decode_binary_record(&payload).unwrap() {
        BinaryWalRecord::DeleteByKey(record) => {
            assert_eq!(record.table, "public_accounts");
            assert_eq!(record.pk_column, "id");
            assert_eq!(record.pk_value, -73);
        }
        _ => panic!("decoded the wrong op"),
    }
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());
    let mut trailing = payload.clone();
    trailing.push(0);
    assert!(decode_binary_record(&trailing).is_err());
    // Unknown op byte is a loud version-skew error.
    let mut skewed = payload;
    skewed[2] = 99;
    assert!(decode_binary_record(&skewed).is_err());
}

#[test]
fn w5b_decoded_delete_reencodes_the_exact_current_bytes() {
    let payload = encode_binary_delete_by_key("public_accounts", "id", -73).unwrap();
    let BinaryWalRecord::DeleteByKey(record) = decode_binary_record(&payload).unwrap() else {
        panic!("decoded the wrong op");
    };
    assert_eq!(reencode_binary_delete_by_key(&record).unwrap(), payload);
}

/// U2 (W5b): the by-key UPDATE record round-trips (table + pk + new_row_id + new image)
/// through the op-dispatch decoder; truncation/trailing bytes fail LOUDLY.
#[test]
fn w5b_update_by_key_round_trips_and_fails_loud() {
    let new_row = [SqlValue::Int4(42), SqlValue::Int4(999)];
    let payload = encode_binary_update_by_key("public_t", "id", 42, 7_000_001, &new_row).unwrap();
    assert!(is_binary_wal_record(&payload));
    assert!(std::str::from_utf8(&payload).is_err()); // 0xFF tag -> invalid UTF-8
    match decode_binary_record(&payload).unwrap() {
        BinaryWalRecord::UpdateByKey(record) => {
            assert_eq!(record.table, "public_t");
            assert_eq!(record.pk_column, "id");
            assert_eq!(record.pk_value, 42);
            assert_eq!(record.new_row_id, 7_000_001);
            assert_eq!(record.new_row_encoded, encode_relational_row(&new_row));
        }
        _ => panic!("decoded the wrong op"),
    }
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());
    let mut trailing = payload.clone();
    trailing.push(0);
    assert!(decode_binary_record(&trailing).is_err());

    // PUMP PATCH OFFSET (non-vacuous): the pump stamps its claimed row id at
    // `binary_update_new_row_id_offset` into a PLACEHOLDER-0 record; patching there must land
    // EXACTLY on the decoded new_row_id (a wrong offset = silent identity corruption).
    let placeholder = encode_binary_update_by_key("public_t", "id", 42, 0, &new_row).unwrap();
    assert_eq!(
        placeholder.len(),
        payload.len(),
        "placeholder is byte-width identical"
    );
    let off = binary_update_new_row_id_offset("public_t", "id");
    let mut patched = placeholder.clone();
    patched[off..off + 8].copy_from_slice(&7_000_001u64.to_le_bytes());
    assert_eq!(
        patched, payload,
        "patching the placeholder at the offset reproduces the fully-encoded record"
    );
    match decode_binary_record(&patched).unwrap() {
        BinaryWalRecord::UpdateByKey(record) => assert_eq!(record.new_row_id, 7_000_001),
        _ => panic!("decoded the wrong op"),
    }
}

#[test]
fn w5b_decoded_update_reencodes_the_exact_current_bytes() {
    let values = [
        SqlValue::Int4(42),
        SqlValue::Text("current codec".to_string()),
    ];
    let payload = encode_binary_update_by_key("public_t", "id", 42, 7_000_001, &values).unwrap();
    let BinaryWalRecord::UpdateByKey(record) = decode_binary_record(&payload).unwrap() else {
        panic!("decoded the wrong op");
    };
    assert_eq!(reencode_binary_update_by_key(&record).unwrap(), payload);
}
