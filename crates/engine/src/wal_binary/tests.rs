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
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 9,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
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
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 8,
        catalog_commands: vec![BinaryTransactionCatalogCommand {
            ordinal: 0,
            command,
        }],
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
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
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
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
        created_table_index_identities: BTreeMap::new(),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: 43,
            relational_next_column_id: 3,
            created_sequence_oids: BTreeMap::new(),
        }),
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
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
        sequence_value_references: Vec::new(),
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
    zero_row_update
        .created_table_index_identities
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
fn transactional_view_uses_additive_opcode_and_exact_identity_closure() {
    let command = parse_command("CREATE VIEW codec_view AS SELECT id FROM codec_source").unwrap();
    let record = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 0,
        catalog_commands: vec![BinaryTransactionCatalogCommand {
            ordinal: 0,
            command: command.clone(),
        }],
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: 43,
            relational_next_column_id: 7,
            created_sequence_oids: BTreeMap::new(),
        }),
        view_operations: vec![BinaryTransactionViewOperationIdentity {
            command_index: 0,
            ordinal: 0,
            target_before: None,
            dependencies: BTreeMap::from([(
                "codec_source".to_string(),
                BinaryCatalogRelationIdentity {
                    kind: BinaryCatalogRelationKind::Table,
                    oid: 41,
                    digest: [1; 32],
                },
            )]),
            target_after: BinaryCatalogRelationIdentity {
                kind: BinaryCatalogRelationKind::View,
                oid: 42,
                digest: [2; 32],
            },
        }],
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: vec![BinaryTransactionOperationIdentity::Catalog { command_index: 0 }],
        statement_digests: vec![transaction_statement_digest(&command).unwrap()],
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: Vec::new(),
    };
    let payload = try_encode_binary_transaction(&record).unwrap();
    assert_eq!(
        payload[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_ORDERED_CATALOG_VIEW_TRANSACTION
        ]
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == record
    ));
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());

    let mut forged_legacy_opcode = payload.clone();
    forged_legacy_opcode[2] = OP_ORDERED_CATALOG_TRANSACTION;
    assert!(decode_binary_record(&forged_legacy_opcode).is_err());

    let mut wrong_position = record.clone();
    wrong_position.view_operations[0].ordinal = 1;
    assert!(try_encode_binary_transaction(&wrong_position).is_none());
    let mut missing_source = record.clone();
    missing_source.view_operations[0].dependencies.clear();
    assert!(try_encode_binary_transaction(&missing_source).is_none());
    let mut wrong_target_kind = record.clone();
    wrong_target_kind.view_operations[0].target_after.kind = BinaryCatalogRelationKind::Table;
    assert!(try_encode_binary_transaction(&wrong_target_kind).is_none());

    let mut replacement = record;
    replacement.view_operations[0].target_before =
        Some(replacement.view_operations[0].target_after.clone());
    replacement.view_operations[0].target_after.digest = [3; 32];
    replacement.catalog_commands[0].command =
        parse_command("CREATE OR REPLACE VIEW codec_view AS SELECT id FROM codec_source").unwrap();
    replacement.statement_digests[0] =
        transaction_statement_digest(&replacement.catalog_commands[0].command).unwrap();
    assert_ne!(
        payload,
        try_encode_binary_transaction(&replacement).unwrap(),
        "replace request and postimage identity must change retry bytes"
    );
}

#[test]
fn transactional_view_lifecycle_uses_additive_opcode_and_canonical_targets() {
    let create = parse_command("CREATE VIEW codec_view AS SELECT id FROM codec_source").unwrap();
    let rename = parse_command("ALTER VIEW codec_view RENAME TO codec_renamed").unwrap();
    let drop = parse_command("DROP VIEW IF EXISTS codec_renamed, missing_codec_view").unwrap();
    let source = BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::Table,
        oid: 41,
        digest: [1; 32],
    };
    let created = BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::View,
        oid: 42,
        digest: [2; 32],
    };
    let renamed = BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::View,
        oid: 42,
        digest: [3; 32],
    };
    let record = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 0,
        catalog_commands: vec![
            BinaryTransactionCatalogCommand {
                ordinal: 0,
                command: create.clone(),
            },
            BinaryTransactionCatalogCommand {
                ordinal: 1,
                command: rename.clone(),
            },
            BinaryTransactionCatalogCommand {
                ordinal: 2,
                command: drop.clone(),
            },
        ],
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: 43,
            relational_next_column_id: 7,
            created_sequence_oids: BTreeMap::new(),
        }),
        view_operations: Vec::new(),
        view_lifecycle_operations: vec![
            BinaryTransactionViewLifecycleOperationIdentity {
                command_index: 0,
                ordinal: 0,
                targets: vec![BinaryTransactionViewLifecycleTargetIdentity {
                    before_name: "codec_view".to_string(),
                    target_before: None,
                    dependencies: BTreeMap::from([("codec_source".to_string(), source.clone())]),
                    after_name: Some("codec_view".to_string()),
                    target_after: Some(created.clone()),
                }],
            },
            BinaryTransactionViewLifecycleOperationIdentity {
                command_index: 1,
                ordinal: 1,
                targets: vec![BinaryTransactionViewLifecycleTargetIdentity {
                    before_name: "codec_view".to_string(),
                    target_before: Some(created),
                    dependencies: BTreeMap::from([("codec_source".to_string(), source.clone())]),
                    after_name: Some("codec_renamed".to_string()),
                    target_after: Some(renamed.clone()),
                }],
            },
            BinaryTransactionViewLifecycleOperationIdentity {
                command_index: 2,
                ordinal: 2,
                targets: vec![
                    BinaryTransactionViewLifecycleTargetIdentity {
                        before_name: "codec_renamed".to_string(),
                        target_before: Some(renamed),
                        dependencies: BTreeMap::from([("codec_source".to_string(), source)]),
                        after_name: None,
                        target_after: None,
                    },
                    BinaryTransactionViewLifecycleTargetIdentity {
                        before_name: "missing_codec_view".to_string(),
                        target_before: None,
                        dependencies: BTreeMap::new(),
                        after_name: None,
                        target_after: None,
                    },
                ],
            },
        ],
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 1 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 2 },
        ],
        statement_digests: [&create, &rename, &drop]
            .into_iter()
            .map(|command| transaction_statement_digest(command).unwrap())
            .collect(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: Vec::new(),
    };

    let payload = try_encode_binary_transaction(&record).unwrap();
    assert_eq!(
        payload[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION,
        ]
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == record
    ));

    let mut forged_legacy_opcode = payload.clone();
    forged_legacy_opcode[2] = OP_ORDERED_CATALOG_VIEW_TRANSACTION;
    assert!(decode_binary_record(&forged_legacy_opcode).is_err());

    for mut tampered in [
        {
            let mut tampered = record.clone();
            tampered.view_lifecycle_operations[1].targets[0].after_name =
                Some("wrong_name".to_string());
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_lifecycle_operations[2].targets.swap(0, 1);
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_lifecycle_operations[2].targets[1]
                .dependencies
                .insert(
                    "smuggled".to_string(),
                    BinaryCatalogRelationIdentity {
                        kind: BinaryCatalogRelationKind::Table,
                        oid: 99,
                        digest: [9; 32],
                    },
                );
            tampered
        },
    ] {
        assert!(try_encode_binary_transaction(&tampered).is_none());
        tampered.view_lifecycle_operations.clear();
        assert!(try_encode_binary_transaction(&tampered).is_none());
    }
}

#[test]
fn identity_bound_transaction_round_trips_and_covers_the_exact_mutation_set() {
    let record = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 9,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
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
fn transactional_sequence_lifecycle_uses_additive_stable_oid_opcodes() {
    let create = parse_command("CREATE SEQUENCE codec_sequence").unwrap();
    let sequence_identity = BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::Sequence,
        oid: 41,
        digest: [1; 32],
    };
    let lifecycle = BinaryTransactionSequenceLifecycleOperationIdentity {
        command_index: 0,
        ordinal: 0,
        targets: vec![BinaryTransactionSequenceLifecycleTargetIdentity {
            before_name: "codec_sequence".to_string(),
            target_before: None,
            dependencies_before: BTreeMap::new(),
            after_name: Some("codec_sequence".to_string()),
            target_after: Some(sequence_identity),
            dependencies_after: BTreeMap::new(),
        }],
    };
    let catalog_only = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::IndexIdentityV1,
        allocator_high_water: 0,
        catalog_commands: vec![BinaryTransactionCatalogCommand {
            ordinal: 0,
            command: create.clone(),
        }],
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: 42,
            relational_next_column_id: 1,
            created_sequence_oids: BTreeMap::new(),
        }),
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: vec![lifecycle],
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: vec![BinaryTransactionOperationIdentity::Catalog { command_index: 0 }],
        statement_digests: vec![transaction_statement_digest(&create).unwrap()],
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: Vec::new(),
    };
    let payload = try_encode_binary_transaction(&catalog_only).unwrap();
    assert_eq!(
        payload[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION,
        ]
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == catalog_only
    ));

    let insert = parse_command("INSERT INTO codec_sequence_owner (payload) VALUES (7)").unwrap();
    let identity_bound = BinaryTransactionRecord {
        allocator_high_water: 8,
        sequence_advances_by_oid: BTreeMap::from([(41, (1, true))]),
        operation_order: vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Insert {
                table: "codec_sequence_owner".to_string(),
            },
        ],
        statement_digests: vec![
            transaction_statement_digest(&create).unwrap(),
            transaction_statement_digest(&insert).unwrap(),
        ],
        sequence_input_oids: BTreeMap::from([((1, "codec_sequence".to_string()), 41)]),
        sequence_value_references: Vec::new(),
        table_identities: BTreeMap::from([(
            "codec_sequence_owner".to_string(),
            BinaryTransactionTableIdentity {
                table_oid: 50,
                schema_digest: [5; 32],
            },
        )]),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: "codec_sequence_owner".to_string(),
            row_id: 7,
            row_encoded: "i:1".to_string(),
        }],
        ..catalog_only.clone()
    };
    let identity_payload = try_encode_binary_transaction(&identity_bound).unwrap();
    assert_eq!(
        identity_payload[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION,
        ]
    );
    assert!(matches!(
        decode_binary_record(&identity_payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == identity_bound
    ));
    assert!(decode_binary_record(&identity_payload[..identity_payload.len() - 1]).is_err());

    let mut forged_family = identity_payload;
    forged_family[2] = OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION;
    assert!(decode_binary_record(&forged_family).is_err());

    let mut missing_stable_state = identity_bound.clone();
    missing_stable_state.sequence_advances_by_oid.clear();
    assert!(try_encode_binary_transaction(&missing_stable_state).is_none());
    let mut wrong_input_oid = identity_bound.clone();
    *wrong_input_oid
        .sequence_input_oids
        .values_mut()
        .next()
        .unwrap() = 42;
    assert!(try_encode_binary_transaction(&wrong_input_oid).is_none());
    let mut smuggled_legacy_state = identity_bound;
    smuggled_legacy_state
        .sequence_advances
        .insert("codec_sequence".to_string(), (1, true));
    assert!(try_encode_binary_transaction(&smuggled_legacy_state).is_none());
}

#[test]
fn sequence_lifecycle_opcode_requires_a_lifecycle_or_reset_owner() {
    let create =
        parse_command("CREATE TABLE stable_only_owner (payload int4)").expect("valid CREATE TABLE");
    let insert =
        parse_command("INSERT INTO stable_only_owner (payload) VALUES (1)").expect("valid INSERT");
    let stable_only = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::IndexIdentityV1,
        allocator_high_water: 2,
        catalog_commands: vec![BinaryTransactionCatalogCommand {
            ordinal: 0,
            command: create.clone(),
        }],
        created_table_identities: BTreeMap::from([(
            "stable_only_owner".to_string(),
            BinaryTransactionTableIdentity {
                table_oid: 50,
                schema_digest: [1; 32],
            },
        )]),
        created_table_index_identities: BTreeMap::from([(
            "stable_only_owner".to_string(),
            Vec::new(),
        )]),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: 51,
            relational_next_column_id: 2,
            created_sequence_oids: BTreeMap::new(),
        }),
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::from([(41, (1, true))]),
        operation_order: vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Insert {
                table: "stable_only_owner".to_string(),
            },
        ],
        statement_digests: vec![
            transaction_statement_digest(&create).unwrap(),
            transaction_statement_digest(&insert).unwrap(),
        ],
        sequence_input_oids: BTreeMap::from([((1, "unowned_sequence".to_string()), 41)]),
        sequence_value_references: Vec::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: "stable_only_owner".to_string(),
            row_id: 1,
            row_encoded: "i:1".to_string(),
        }],
    };
    assert!(
        try_encode_binary_transaction(&stable_only).is_none(),
        "stable sequence values may accompany opcode 18, but cannot select that family"
    );

    // This is the formerly decoder-valid zero-catalog-command shape. It is deliberately assembled
    // below the encoder so the decoder's own canonicality boundary is covered independently.
    let mut forged = vec![
        WAL_BINARY_TAG,
        WAL_BINARY_VERSION,
        OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION,
    ];
    forged.extend_from_slice(&0_u32.to_le_bytes()); // catalog commands
    forged.extend_from_slice(&0_u32.to_le_bytes()); // created table identities
    forged.extend_from_slice(&0_u32.to_le_bytes()); // created table index identities
    forged.extend_from_slice(&51_u32.to_le_bytes()); // catalog next OID
    forged.extend_from_slice(&2_u32.to_le_bytes()); // catalog next column id
    forged.extend_from_slice(&0_u32.to_le_bytes()); // generated sequence OIDs
    forged.extend_from_slice(&0_u32.to_le_bytes()); // view lifecycle identities
    forged.extend_from_slice(&0_u32.to_le_bytes()); // index lifecycle identities
    forged.extend_from_slice(&0_u32.to_le_bytes()); // sequence lifecycle identities
    forged.extend_from_slice(&0_u32.to_le_bytes()); // sequence reset identities
    forged.extend_from_slice(&1_u32.to_le_bytes()); // stable sequence advances
    forged.extend_from_slice(&41_u32.to_le_bytes());
    forged.extend_from_slice(&1_i64.to_le_bytes());
    forged.push(1);
    forged.extend_from_slice(&1_u32.to_le_bytes()); // operation order
    forged.push(TXN_OPERATION_INSERT);
    forged.extend_from_slice(&1_u16.to_le_bytes());
    forged.extend_from_slice(b"t");
    forged.extend_from_slice(&1_u32.to_le_bytes()); // statement digests
    forged.extend_from_slice(&[1; 32]);
    forged.extend_from_slice(&1_u32.to_le_bytes()); // sequence inputs
    forged.extend_from_slice(&0_u32.to_le_bytes());
    forged.extend_from_slice(&1_u16.to_le_bytes());
    forged.extend_from_slice(b"s");
    forged.extend_from_slice(&41_u32.to_le_bytes());
    forged.extend_from_slice(&0_u32.to_le_bytes()); // table resets
    forged.extend_from_slice(&1_u32.to_le_bytes()); // table identities
    forged.extend_from_slice(&1_u16.to_le_bytes());
    forged.extend_from_slice(b"t");
    forged.extend_from_slice(&50_u32.to_le_bytes());
    forged.extend_from_slice(&[2; 32]);
    forged.extend_from_slice(&2_u64.to_le_bytes()); // allocator high water
    forged.extend_from_slice(&0_u32.to_le_bytes()); // legacy sequence advances
    forged.extend_from_slice(&1_u32.to_le_bytes()); // mutations
    forged.push(TXN_INSERT);
    forged.extend_from_slice(&1_u16.to_le_bytes());
    forged.extend_from_slice(b"t");
    forged.extend_from_slice(&1_u64.to_le_bytes());
    forged.extend_from_slice(&3_u32.to_le_bytes());
    forged.extend_from_slice(b"i:1");

    let error = match decode_binary_record(&forged) {
        Ok(_) => panic!("stable-value-only opcode 18 must be rejected by the decoder"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("requires a lifecycle or reset identity"),
        "{error}"
    );
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
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 11,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
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

#[test]
fn sequence_value_transition_codec_is_strict_and_identity_bound() {
    let operation = BinarySequenceValueOperation::NextVal;
    let parent_request_digest = [0x42; 32];
    let input_digest = sequence_value_input_digest(SequenceValueInput {
        parent_txn_id: 11,
        parent_autocommit: false,
        statement_ordinal: 2,
        expression_ordinal: 3,
        parent_request_digest,
        source_name: "codec_value",
        operation,
        set_value: None,
    });
    let transition = BinarySequenceValueTransitionRecord {
        transition_txn_id: 12,
        parent_txn_id: 11,
        parent_autocommit: false,
        statement_ordinal: 2,
        expression_ordinal: 3,
        parent_request_digest,
        input_digest,
        sequence_oid: 41,
        source_name: "codec_value".to_string(),
        effective_name: "codec_value".to_string(),
        published_name: "codec_value".to_string(),
        base_catalog_generation: 7,
        prior_last_value: 9,
        prior_is_called: true,
        new_last_value: 10,
        new_is_called: true,
        returned_value: 10,
        private_descriptor_digest: None,
        operation,
    };
    let payload = encode_sequence_value_transition(&transition).unwrap();
    assert_eq!(
        payload[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_SEQUENCE_VALUE_TRANSITION
        ]
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::SequenceValueTransition(decoded) if decoded == transition
    ));
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());
    let mut trailing = payload.clone();
    trailing.push(0);
    assert!(decode_binary_record(&trailing).is_err());
    assert!(validate_sequence_envelope_transaction_id(&payload, 12).is_ok());
    assert!(validate_sequence_envelope_transaction_id(&payload, 13).is_err());

    let mut bad_prior = transition.clone();
    bad_prior.prior_last_value = i64::MAX;
    assert!(encode_sequence_value_transition(&bad_prior).is_none());
    let mut bad_digest = transition;
    bad_digest.input_digest[0] ^= 1;
    assert!(encode_sequence_value_transition(&bad_digest).is_none());
}

#[test]
fn sequence_reference_wrapper_round_trips_and_rejects_noncanonical_references() {
    let input_digest = sequence_value_input_digest(SequenceValueInput {
        parent_txn_id: 21,
        parent_autocommit: false,
        statement_ordinal: 0,
        expression_ordinal: 0,
        parent_request_digest: [0x24; 32],
        source_name: "wrapped_value",
        operation: BinarySequenceValueOperation::NextVal,
        set_value: None,
    });
    let reference = BinarySequenceValueReference {
        transition_txn_id: 22,
        parent_txn_id: 21,
        statement_ordinal: 0,
        expression_ordinal: 0,
        sequence_oid: 51,
        returned_value: 1,
        input_digest,
        table_oid: 0,
        column_id: 0,
        staging_row_ordinal: 0,
        row_id: 0,
        final_value_overwritten: false,
        default_expression: false,
    };
    assert_eq!(ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES, 86);
    let mut exact_reference = [0xa5; ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
    encode_sequence_value_reference_into_exact(&reference, &mut exact_reference).unwrap();
    let mut golden_reference = [0; ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
    golden_reference[0..8].copy_from_slice(&22_u64.to_le_bytes());
    golden_reference[8..16].copy_from_slice(&21_u64.to_le_bytes());
    golden_reference[16..20].copy_from_slice(&0_u32.to_le_bytes());
    golden_reference[20..24].copy_from_slice(&0_u32.to_le_bytes());
    golden_reference[24..28].copy_from_slice(&51_u32.to_le_bytes());
    golden_reference[28..36].copy_from_slice(&1_i64.to_le_bytes());
    golden_reference[36..68].copy_from_slice(&input_digest);
    assert_eq!(
        exact_reference, golden_reference,
        "standalone reference layout"
    );
    assert_eq!(
        decode_sequence_value_reference_exact(&exact_reference).unwrap(),
        reference
    );
    for retained in 0..ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES {
        assert!(decode_sequence_value_reference_exact(&exact_reference[..retained]).is_err());
    }
    let mut surplus_reference = exact_reference.to_vec();
    surplus_reference.push(0);
    assert!(decode_sequence_value_reference_exact(&surplus_reference).is_err());
    let mut zero_transition = exact_reference;
    zero_transition[0..8].fill(0);
    assert!(decode_sequence_value_reference_exact(&zero_transition).is_err());
    let mut zero_digest = exact_reference;
    zero_digest[36..68].fill(0);
    assert!(decode_sequence_value_reference_exact(&zero_digest).is_err());
    for flag in [84, 85] {
        let mut invalid_boolean = exact_reference;
        invalid_boolean[flag] = 2;
        assert!(decode_sequence_value_reference_exact(&invalid_boolean).is_err());
    }
    let mut invalid_staging = reference.clone();
    invalid_staging.staging_row_ordinal = 1;
    assert!(
        encode_sequence_value_reference_into_exact(&invalid_staging, &mut exact_reference).is_err()
    );

    let mut first_default = reference.clone();
    first_default.transition_txn_id = 30;
    first_default.expression_ordinal = 5;
    first_default.table_oid = 71;
    first_default.column_id = 81;
    first_default.row_id = 91;
    first_default.default_expression = true;
    first_default.final_value_overwritten = true;
    let mut second_default = first_default.clone();
    second_default.transition_txn_id = 31;
    second_default.expression_ordinal = 6;
    second_default.column_id = 82;
    assert_eq!(second_default.sequence_oid, first_default.sequence_oid);
    assert!(valid_sequence_value_reference_closure(&[
        first_default.clone(),
        second_default.clone(),
    ]));
    let mut duplicate_expression = second_default.clone();
    duplicate_expression.transition_txn_id = 32;
    duplicate_expression.expression_ordinal = first_default.expression_ordinal;
    duplicate_expression.column_id = 83;
    assert!(!valid_sequence_value_reference_closure(&[
        first_default.clone(),
        duplicate_expression,
    ]));
    let mut duplicate_default_binding = second_default;
    duplicate_default_binding.transition_txn_id = 32;
    duplicate_default_binding.expression_ordinal = 7;
    duplicate_default_binding.column_id = first_default.column_id;
    assert!(!valid_sequence_value_reference_closure(&[
        first_default,
        duplicate_default_binding,
    ]));

    let base = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 1,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: vec![reference.clone()],
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: Vec::new(),
    };
    let payload = try_encode_binary_transaction(&base).unwrap();
    assert_eq!(
        payload[..3],
        [
            WAL_BINARY_TAG,
            WAL_BINARY_VERSION,
            OP_SEQUENCE_REFERENCED_TRANSACTION
        ]
    );
    let standalone_offset =
        15 + usize::try_from(u64::from_le_bytes(payload[3..11].try_into().unwrap())).unwrap();
    assert_eq!(
        &payload[standalone_offset..standalone_offset + ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES],
        &exact_reference,
        "opcode-21 framing must reuse the standalone reference bytes"
    );
    assert!(matches!(
        decode_binary_record(&payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == base
    ));
    assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());
    let wrapped_base_len =
        usize::try_from(u64::from_le_bytes(payload[3..11].try_into().unwrap())).unwrap();
    let reference_count_at = 11 + wrapped_base_len;
    let mut huge_count = payload.clone();
    huge_count[reference_count_at..reference_count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let huge_count_error = match decode_binary_record(&huge_count) {
        Ok(_) => panic!("an oversized sequence-reference count must fail before allocation"),
        Err(error) => error,
    };
    assert!(
        huge_count_error
            .to_string()
            .contains("reference count does not match remaining bytes"),
        "{huge_count_error}"
    );
    let mut trailing = payload;
    trailing.push(0);
    assert!(decode_binary_record(&trailing).is_err());

    let mut invalid_default = base.clone();
    invalid_default.sequence_value_references[0].default_expression = true;
    assert!(try_encode_binary_transaction(&invalid_default).is_none());
    let mut wrong_parent = base.clone();
    wrong_parent.sequence_value_references[0].parent_txn_id = 23;
    let wrong_parent_payload = try_encode_binary_transaction(&wrong_parent).unwrap();
    assert!(validate_sequence_envelope_transaction_id(&wrong_parent_payload, 21).is_err());
    let mut out_of_order = base;
    let mut earlier = reference;
    earlier.transition_txn_id = 20;
    out_of_order.sequence_value_references.push(earlier);
    assert!(try_encode_binary_transaction(&out_of_order).is_none());
}
