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

#[test]
fn row_only_transaction_keeps_v1_opcode_and_composite_typed_catalog_round_trips() {
    let mut sequence_advances = BTreeMap::new();
    sequence_advances.insert("s".to_string(), (5, true));
    let row_only = BinaryTransactionRecord {
        allocator_high_water: 9,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        catalog_output: None,
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        table_resets: Vec::new(),
        sequence_advances,
        table_identities: BTreeMap::new(),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: "t".to_string(),
            row_id: 8,
            row_encoded: "i:42".to_string(),
        }],
    };
    let old_payload = try_encode_binary_transaction(&row_only).unwrap();
    // Literal bytes captured from the pre-composite v1 row-transaction framing. Deliberately
    // do not use codec constants here: the fixture must detect an opcode/tag/version drift as
    // well as sequence and row-mutation layout drift.
    let pre_composite_fixture = vec![
        255, 1, 4, // tag, version, row-only transaction opcode
        9, 0, 0, 0, 0, 0, 0, 0, // allocator high-water
        1, 0, 0, 0, // one sequence advance
        1, 0, b's', // sequence name
        5, 0, 0, 0, 0, 0, 0, 0, 1, // sequence post-state + is_called
        1, 0, 0, 0, // one mutation
        1, // INSERT
        1, 0, b't', // table name
        8, 0, 0, 0, 0, 0, 0, 0, // stable row identity
        4, 0, 0, 0, b'i', b':', b'4', b'2', // encoded row image
    ];
    assert_eq!(old_payload, pre_composite_fixture);
    assert_eq!(
        old_payload[..3],
        [WAL_BINARY_TAG, WAL_BINARY_VERSION, OP_TRANSACTION]
    );
    assert!(matches!(
        decode_binary_record(&old_payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == row_only
    ));

    let command = parse_command("CREATE TABLE composite_codec (id int4)").unwrap();
    let composite = BinaryTransactionRecord {
        allocator_high_water: 8,
        catalog_commands: vec![BinaryTransactionCatalogCommand {
            ordinal: 0,
            command,
        }],
        created_table_identities: BTreeMap::new(),
        catalog_output: None,
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: "composite_codec".to_string(),
            row_id: 7,
            row_encoded: encode_relational_row(&[SqlValue::Int4(1)]),
        }],
    };
    let payload = try_encode_binary_transaction(&composite).unwrap();
    assert_eq!(
        payload[..3],
        [WAL_BINARY_TAG, WAL_BINARY_VERSION, OP_COMPOSITE_TRANSACTION]
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == composite
    ));
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());

    let two_commands = BinaryTransactionRecord {
        catalog_commands: vec![
            BinaryTransactionCatalogCommand {
                ordinal: 0,
                command: parse_command("CREATE TABLE composite_codec_a (id int4)").unwrap(),
            },
            BinaryTransactionCatalogCommand {
                ordinal: 2,
                command: parse_command("CREATE TABLE composite_codec_b (id int4)").unwrap(),
            },
        ],
        created_table_identities: BTreeMap::from([
            (
                "composite_codec_a".to_string(),
                BinaryTransactionTableIdentity {
                    table_oid: 41,
                    schema_digest: [1; 32],
                },
            ),
            (
                "composite_codec_b".to_string(),
                BinaryTransactionTableIdentity {
                    table_oid: 42,
                    schema_digest: [2; 32],
                },
            ),
        ]),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: 43,
            relational_next_column_id: 3,
            created_sequence_oids: BTreeMap::new(),
        }),
        operation_order: vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Insert {
                table: "composite_codec_a".to_string(),
            },
            BinaryTransactionOperationIdentity::Catalog { command_index: 1 },
        ],
        statement_digests: vec![
            transaction_statement_digest(
                &parse_command("CREATE TABLE composite_codec_a (id int4)").unwrap(),
            )
            .unwrap(),
            transaction_statement_digest(
                &parse_command("INSERT INTO composite_codec_a VALUES (1)").unwrap(),
            )
            .unwrap(),
            transaction_statement_digest(
                &parse_command("CREATE TABLE composite_codec_b (id int4)").unwrap(),
            )
            .unwrap(),
        ],
        sequence_input_oids: BTreeMap::new(),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: "composite_codec_a".to_string(),
            row_id: 7,
            row_encoded: encode_relational_row(&[SqlValue::Int4(1)]),
        }],
        ..composite
    };
    let ordered = try_encode_binary_transaction(&two_commands).unwrap();
    assert_eq!(
        ordered[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_ORDERED_CATALOG_TRANSACTION
        ]
    );
    assert!(matches!(
        decode_binary_record(&ordered).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == two_commands
    ));

    // Exact retry identity is the complete typed statement stream, not merely the surviving
    // resolved mutations. Distinct zero-row UPDATEs and INSERTs shadowed by a later reset must
    // therefore produce distinct transaction payloads.
    let mut zero_row_update = two_commands.clone();
    zero_row_update.catalog_commands.truncate(1);
    zero_row_update
        .created_table_identities
        .retain(|table, _| table == "composite_codec_a");
    zero_row_update.operation_order = vec![
        BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
        BinaryTransactionOperationIdentity::Update {
            table: "composite_codec_a".to_string(),
        },
    ];
    zero_row_update.statement_digests = vec![
        transaction_statement_digest(&zero_row_update.catalog_commands[0].command).unwrap(),
        transaction_statement_digest(
            &parse_command("UPDATE composite_codec_a SET id = 1 WHERE id = 999").unwrap(),
        )
        .unwrap(),
    ];
    zero_row_update.mutations.clear();
    let zero_row_one = try_encode_binary_transaction(&zero_row_update).unwrap();
    zero_row_update.statement_digests[1] = transaction_statement_digest(
        &parse_command("UPDATE composite_codec_a SET id = 2 WHERE id = 999").unwrap(),
    )
    .unwrap();
    let zero_row_two = try_encode_binary_transaction(&zero_row_update).unwrap();
    assert_ne!(zero_row_one, zero_row_two);

    let mut reset_shadowed_insert = zero_row_update.clone();
    reset_shadowed_insert.operation_order = vec![
        BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
        BinaryTransactionOperationIdentity::Insert {
            table: "composite_codec_a".to_string(),
        },
        BinaryTransactionOperationIdentity::TableReset {
            table: "composite_codec_a".to_string(),
        },
    ];
    reset_shadowed_insert.statement_digests = vec![
        transaction_statement_digest(&reset_shadowed_insert.catalog_commands[0].command).unwrap(),
        transaction_statement_digest(
            &parse_command("INSERT INTO composite_codec_a VALUES (1)").unwrap(),
        )
        .unwrap(),
        transaction_statement_digest(&parse_command("TRUNCATE composite_codec_a").unwrap())
            .unwrap(),
    ];
    reset_shadowed_insert.table_resets = vec![BinaryTransactionTableReset {
        ordinal: 2,
        table: "composite_codec_a".to_string(),
        table_oid: 41,
        schema_digest: [1; 32],
        source_commit_seq: 0,
        before_digest: [2; 32],
        expected_rows: 0,
        after_empty_digest: [3; 32],
        dependency_identities: BTreeMap::from([("composite_codec_a".to_string(), 41)]),
    }];
    let shadowed_one = try_encode_binary_transaction(&reset_shadowed_insert).unwrap();
    reset_shadowed_insert.statement_digests[1] = transaction_statement_digest(
        &parse_command("INSERT INTO composite_codec_a VALUES (2)").unwrap(),
    )
    .unwrap();
    let shadowed_two = try_encode_binary_transaction(&reset_shadowed_insert).unwrap();
    assert_ne!(shadowed_one, shadowed_two);

    // A final INSERT mutation cannot be justified by a DELETE-only statement identity.
    let mut wrong_family = two_commands.clone();
    wrong_family.operation_order[1] = BinaryTransactionOperationIdentity::Delete {
        table: "composite_codec_a".to_string(),
    };
    wrong_family.statement_digests[1] = transaction_statement_digest(
        &parse_command("DELETE FROM composite_codec_a WHERE id = 1").unwrap(),
    )
    .unwrap();
    assert!(try_encode_binary_transaction(&wrong_family).is_none());

    let operation_marker = [TXN_OPERATION_INSERT, "composite_codec_a".len() as u8, 0];
    let operation_offset = ordered
        .windows(operation_marker.len())
        .position(|window| window == operation_marker)
        .expect("ordered INSERT operation marker");
    let mut forged_family = ordered.clone();
    forged_family[operation_offset] = TXN_OPERATION_DELETE;
    assert!(decode_binary_record(&forged_family).is_err());

    let mut missing_statement = two_commands.clone();
    missing_statement.operation_order.pop();
    assert!(try_encode_binary_transaction(&missing_statement).is_none());
    let mut missing_output = two_commands.clone();
    missing_output.catalog_output = None;
    assert!(try_encode_binary_transaction(&missing_output).is_none());
    let mut rogue_sequence = two_commands.clone();
    rogue_sequence
        .catalog_output
        .as_mut()
        .unwrap()
        .created_sequence_oids
        .insert("not_created".to_string(), 99);
    assert!(try_encode_binary_transaction(&rogue_sequence).is_none());
    let mut malformed = payload;
    malformed[3..7].copy_from_slice(&2_u32.to_le_bytes());
    assert!(decode_binary_record(&malformed).is_err());
}

#[test]
fn identity_bound_transaction_round_trips_and_covers_the_exact_mutation_set() {
    let record = BinaryTransactionRecord {
        allocator_high_water: 9,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        catalog_output: None,
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::from([(
            "identity_rows".to_string(),
            BinaryTransactionTableIdentity {
                table_oid: 42,
                schema_digest: [7; 32],
            },
        )]),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: "identity_rows".to_string(),
            row_id: 8,
            row_encoded: "i:42".to_string(),
        }],
    };
    let payload = try_encode_binary_transaction(&record).unwrap();
    assert_eq!(
        payload[..3],
        [WAL_BINARY_TAG, WAL_BINARY_VERSION, OP_IDENTITY_TRANSACTION]
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == record
    ));
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());

    let mut missing = record.clone();
    missing.table_identities.clear();
    missing.table_identities.insert(
        "other_rows".to_string(),
        BinaryTransactionTableIdentity {
            table_oid: 43,
            schema_digest: [8; 32],
        },
    );
    assert!(try_encode_binary_transaction(&missing).is_none());
}

#[test]
fn typed_table_reset_round_trips_and_rejects_noncanonical_composition() {
    let reset = BinaryTransactionTableReset {
        ordinal: 3,
        table: "accounts".to_string(),
        table_oid: 42,
        schema_digest: [1; 32],
        source_commit_seq: 6,
        before_digest: [2; 32],
        expected_rows: 7,
        after_empty_digest: [3; 32],
        dependency_identities: BTreeMap::from([("accounts".to_string(), 42)]),
    };
    let record = BinaryTransactionRecord {
        allocator_high_water: 11,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        catalog_output: None,
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        table_resets: vec![reset],
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: "accounts".to_string(),
            row_id: 10,
            row_encoded: "i:9".to_string(),
        }],
    };
    let payload = try_encode_binary_transaction(&record).unwrap();
    assert_eq!(
        payload[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_TABLE_RESET_TRANSACTION
        ]
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == record
    ));
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());

    let mut noncanonical = record.clone();
    noncanonical.mutations = vec![BinaryTransactionMutation::Delete {
        table: "accounts".to_string(),
        row_id: 1,
        old_row_encoded: "i:1".to_string(),
    }];
    assert!(try_encode_binary_transaction(&noncanonical).is_none());

    let mut missing_target = record;
    missing_target.table_resets[0].dependency_identities.clear();
    assert!(try_encode_binary_transaction(&missing_target).is_none());
}
