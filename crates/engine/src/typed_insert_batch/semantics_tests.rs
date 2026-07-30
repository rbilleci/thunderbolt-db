use super::*;

fn prepared_sequence_insert(insert: &Insert, catalog: &CatalogSnapshot) -> PreparedTypedInsert {
    prepare_typed_insert_semantics(insert, catalog, catalog.commit_seq, None)
        .expect("sequence INSERT semantic preparation succeeds")
        .expect("current catalog generation prepares")
}

fn sequence_bindings(
    prepared: &PreparedTypedInsert,
    parent: &sequence_defaults::effects::SequenceDefaultParentContext,
    first_value: i64,
) -> Vec<sequence_defaults::SequenceDefaultBinding> {
    prepared
        .sequence_requests()
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, request)| {
            sequence_defaults::SequenceDefaultBinding::published(
                request,
                parent.clone(),
                first_value + i64::try_from(index).unwrap(),
            )
        })
        .collect()
}

fn sequence_parent(
    txn_id: TxnId,
    autocommit: bool,
    expression_ordinal_base: u32,
) -> sequence_defaults::effects::SequenceDefaultParentContext {
    sequence_defaults::effects::SequenceDefaultParentContext::for_test(
        txn_id,
        autocommit,
        gpu_db_wal::canonical_request_digest(b"typed-insert-sequence-parent"),
        InsertStatementOrdinal::FIRST,
        expression_ordinal_base,
    )
}

#[test]
fn semantic_prepare_binds_returning_identities_duplicates_wildcards_and_result_geometry() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE returning_semantics (small int2, id int4, large int8, amount numeric(10,2), day date, happened timestamp, uid uuid, enabled bool, note text)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let insert = Insert {
        table: "returning_semantics".to_string(),
        columns: vec!["id".to_string()],
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Int4(7)]]),
        returning: vec![
            "note".to_string(),
            "id".to_string(),
            "note".to_string(),
            PROJECTION_WILDCARD_SENTINEL.to_string(),
        ],
    };
    let prepared = prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .unwrap()
        .unwrap();
    let table = &catalog.relational_catalog["returning_semantics"];
    let bound = prepared.returning();
    let expected_ids = [
        table.columns[8].id,
        table.columns[1].id,
        table.columns[8].id,
    ]
    .into_iter()
    .chain(table.columns.iter().map(|column| column.id))
    .collect::<Vec<_>>();
    assert_eq!(
        bound
            .columns()
            .iter()
            .map(|column| column.column_id())
            .collect::<Vec<_>>(),
        expected_ids,
        "RETURNING order and duplicates are semantic output authority"
    );
    assert_eq!(
        bound
            .columns()
            .iter()
            .map(|column| column.catalog_column_ordinal())
            .collect::<Vec<_>>(),
        vec![8, 1, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        bound
            .columns()
            .iter()
            .map(|column| column.ty())
            .collect::<Vec<_>>(),
        vec![
            SqlType::Text,
            SqlType::Int4,
            SqlType::Text,
            SqlType::Int2,
            SqlType::Int4,
            SqlType::Int8,
            SqlType::Numeric {
                precision: 10,
                scale: 2,
            },
            SqlType::Date,
            SqlType::Timestamp,
            SqlType::Uuid,
            SqlType::Bool,
            SqlType::Text,
        ]
    );
    let geometry = bound.geometry();
    assert_eq!(
        (
            geometry.row_count,
            geometry.column_count,
            geometry.cell_count
        ),
        (1, 12, 12)
    );
    let effect_shape = prepared.effect_returning_shape();
    assert_eq!(
        (
            effect_shape.row_count(),
            effect_shape.column_count(),
            effect_shape.cell_count(),
        ),
        (1, 12, 12),
        "the scalar effect handoff retains RETURNING geometry without a result container"
    );
    assert_eq!(
        prepared
            .effect_returning_projection_identities()
            .map(|identity| (
                identity.catalog_column_ordinal(),
                identity.column_id(),
                identity.attnum(),
                identity.name().to_string(),
                identity.ty(),
                identity.type_oid(),
                identity.type_size(),
            ))
            .collect::<Vec<_>>(),
        [8_usize, 1, 8]
            .into_iter()
            .chain(0..table.columns.len())
            .map(|catalog_column_ordinal| {
                let column = &table.columns[catalog_column_ordinal];
                (
                    u32::try_from(catalog_column_ordinal).unwrap(),
                    column.id,
                    column.attnum,
                    column.name.clone(),
                    column.ty,
                    column.type_oid,
                    column.type_size,
                )
            })
            .collect::<Vec<_>>(),
        "the scalar effect handoff retains SQL order, duplicate projections, and wildcard expansion"
    );
    assert!(try_prepare_typed_insert_batch(
        &Command::Insert(insert),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .is_none());
}

#[test]
fn semantic_effect_shape_keeps_first_wrapper_and_explicit_statement_ordinal() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE effect_statement_shape (first serial, second serial, note int4 DEFAULT 9)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let Command::Insert(insert) = parse_command(
        "INSERT INTO effect_statement_shape (second, note) VALUES (DEFAULT, 7), (22, DEFAULT), (DEFAULT, DEFAULT)",
    )
    .unwrap()
    else {
        panic!("parser must produce INSERT");
    };

    let first = prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .unwrap()
        .unwrap();
    let seventh_ordinal = InsertStatementOrdinal::from_u32(7);
    let seventh = prepare_typed_insert_semantics_at(
        &insert,
        &catalog,
        catalog.commit_seq,
        None,
        seventh_ordinal,
    )
    .unwrap()
    .unwrap();
    let table = &catalog.relational_catalog["effect_statement_shape"];

    assert_eq!(
        first.effect_statement_ordinal(),
        InsertStatementOrdinal::FIRST
    );
    assert_eq!(seventh.effect_statement_ordinal(), seventh_ordinal);
    assert_eq!(seventh.effect_row_count(), 3);
    let target = seventh.effect_target();
    assert_eq!(target.table_oid(), table.oid);
    assert_eq!(
        target.prepared_catalog_seq(),
        catalog.commit_seq,
        "the effect target retains the exact prepared catalog generation"
    );
    assert_eq!(
        target.schema_digest(),
        crate::engine_transaction_reset::table_schema_digest(table).unwrap()
    );

    let first_requests = first
        .effect_sequence_requests()
        .map(|request| {
            (
                request.target_table_oid(),
                request.row_ordinal(),
                request.catalog_column_ordinal(),
                request.column_id(),
                request.sequence_oid(),
                request.sequence_source_name().to_string(),
                request.sequence_effective_name().to_string(),
                request.statement_ordinal(),
                request.expression_ordinal(),
            )
        })
        .collect::<Vec<_>>();
    let seventh_requests = seventh
        .effect_sequence_requests()
        .map(|request| {
            (
                request.target_table_oid(),
                request.row_ordinal(),
                request.catalog_column_ordinal(),
                request.column_id(),
                request.sequence_oid(),
                request.sequence_source_name().to_string(),
                request.sequence_effective_name().to_string(),
                request.statement_ordinal(),
                request.expression_ordinal(),
            )
        })
        .collect::<Vec<_>>();
    assert!(first_requests
        .iter()
        .all(|request| request.7 == InsertStatementOrdinal::FIRST));
    assert!(seventh_requests
        .iter()
        .all(|request| request.7 == seventh_ordinal));
    assert_eq!(
        first_requests
            .iter()
            .map(|request| (
                &request.0, &request.1, &request.2, &request.3, &request.4, &request.5, &request.6,
                &request.8
            ))
            .collect::<Vec<_>>(),
        seventh_requests
            .iter()
            .map(|request| (
                &request.0, &request.1, &request.2, &request.3, &request.4, &request.5, &request.6,
                &request.8
            ))
            .collect::<Vec<_>>(),
        "only statement identity changes; row-major/catalog-column request geometry is stable"
    );
    assert_eq!(
        seventh_requests
            .iter()
            .map(|request| (request.1, request.2, request.8))
            .collect::<Vec<_>>(),
        vec![(0, 0, 0), (0, 1, 1), (1, 0, 2), (2, 0, 4), (2, 1, 5)],
        "inactive default cells keep their established expression gaps"
    );
}

#[test]
fn semantic_prepare_keeps_coercion_before_returning_bind_errors() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE returning_precedence (id int4)")
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let malformed = Insert {
        table: "returning_precedence".to_string(),
        columns: vec!["id".to_string()],
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Text("bad".to_string())]]),
        returning: vec!["missing".to_string()],
    };
    assert!(matches!(
        prepare_typed_insert_semantics(&malformed, &catalog, catalog.commit_seq, None),
        Err(ExecuteError::Engine(EngineError::ApplyFailed(message)))
            if message == "invalid value for column \"id\""
    ));
    let returning_missing = Insert {
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Int4(1)]]),
        ..malformed
    };
    assert!(matches!(
        prepare_typed_insert_semantics(&returning_missing, &catalog, catalog.commit_seq, None),
        Err(ExecuteError::Engine(EngineError::UndefinedColumn(name))) if name == "missing"
    ));
}

#[test]
fn semantic_prepare_fails_closed_when_an_fk_parent_is_absent() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE absent_fk_child (id int4, parent_id int4)")
        .unwrap();
    let mut catalog = (*engine.catalog_snapshot()).clone();
    catalog
        .relational_catalog
        .get_mut("absent_fk_child")
        .unwrap()
        .foreign_keys
        .push(crate::relational_model::RelationalForeignKey {
            name: "absent_fk_child_parent_fkey".to_string(),
            column: "parent_id".to_string(),
            referenced_table: "absent_fk_parent".to_string(),
            referenced_column: "id".to_string(),
        });
    let Command::Insert(insert) =
        parse_command("INSERT INTO absent_fk_child (id, parent_id) VALUES (1, 7)").unwrap()
    else {
        panic!("test command is an INSERT");
    };

    assert!(matches!(
        prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None),
        Err(ExecuteError::Engine(EngineError::ApplyFailed(message)))
            if message == "typed INSERT canonical foreign-key parent is absent"
    ));
}

#[test]
fn sequence_defaults_discover_row_major_requests_without_execution_and_seal_exactly_once() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE sequence_semantics (first serial, second serial, note int4 DEFAULT 9)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let Command::Insert(insert) = parse_command(
        "INSERT INTO sequence_semantics (second, note) VALUES (DEFAULT, 7), (22, DEFAULT), (DEFAULT, DEFAULT)",
    )
    .unwrap() else {
        panic!("parser must produce INSERT");
    };
    let table = &catalog.relational_catalog["sequence_semantics"];
    let first_sequence = match table.columns[0].default.as_ref() {
        Some(ColumnDefault::SequenceNextVal { sequence, .. }) => sequence,
        _ => panic!("first serial has a sequence default"),
    };
    let second_sequence = match table.columns[1].default.as_ref() {
        Some(ColumnDefault::SequenceNextVal { sequence, .. }) => sequence,
        _ => panic!("second serial has a sequence default"),
    };
    let first_oid = catalog.relational_sequences[first_sequence].oid;
    let second_oid = catalog.relational_sequences[second_sequence].oid;
    let sequence_state_before = catalog.relational_sequences.clone();
    let prepared = prepared_sequence_insert(&insert, &catalog);
    let parent = sequence_parent(701, false, 40);
    assert_eq!(
        prepared
            .sequence_requests()
            .iter()
            .map(|request| (
                request.target_table_oid,
                request.row_ordinal,
                request.catalog_column_ordinal,
                request.column_id,
                request.sequence_oid,
                request.statement_ordinal,
                request.expression_ordinal,
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                table.oid,
                0,
                0,
                table.columns[0].id,
                first_oid,
                InsertStatementOrdinal::FIRST,
                0
            ),
            (
                table.oid,
                0,
                1,
                table.columns[1].id,
                second_oid,
                InsertStatementOrdinal::FIRST,
                1
            ),
            (
                table.oid,
                1,
                0,
                table.columns[0].id,
                first_oid,
                InsertStatementOrdinal::FIRST,
                2
            ),
            (
                table.oid,
                2,
                0,
                table.columns[0].id,
                first_oid,
                InsertStatementOrdinal::FIRST,
                4
            ),
            (
                table.oid,
                2,
                1,
                table.columns[1].id,
                second_oid,
                InsertStatementOrdinal::FIRST,
                5
            ),
        ]
    );
    assert_eq!(catalog.relational_sequences, sequence_state_before);
    assert!(try_prepare_typed_insert_batch(
        &Command::Insert(insert.clone()),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .is_none());

    let mut bindings = sequence_bindings(&prepared, &parent, 100);
    bindings[0] = sequence_defaults::SequenceDefaultBinding::private(
        bindings[0].request_mut_for_test().clone(),
        parent.clone(),
        100,
    );
    let batch = prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::from_bindings(parent, bindings),
            false,
            false,
        )
        .unwrap();
    assert_eq!(batch.sequence_bindings.len(), 5);
    assert!(matches!(
        &batch.columns[0].values,
        TypedInsertColumnValues::I32(values) if values.as_ref() == [100, 102, 103]
    ));
    assert!(matches!(
        &batch.columns[1].values,
        TypedInsertColumnValues::I32(values) if values.as_ref() == [101, 22, 104]
    ));
    assert!(matches!(
        &batch.columns[0].default_resolution,
        TypedInsertDefaultResolution::Bitmap(words) if words.as_ref() == [0b111]
    ));
    assert!(matches!(
        &batch.columns[1].default_resolution,
        TypedInsertDefaultResolution::Bitmap(words) if words.as_ref() == [0b101]
    ));
    assert_eq!(catalog.relational_sequences, sequence_state_before);
}

#[test]
fn inactive_sequence_columns_do_not_shift_active_slots_or_canonical_input_digests() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE inactive_leading (supplied_leading serial, active serial)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE inactive_middle (first serial, supplied_middle serial, tail serial)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();

    let Command::Insert(leading) =
        parse_command("INSERT INTO inactive_leading (supplied_leading) VALUES (41), (42)").unwrap()
    else {
        panic!("parser must produce INSERT");
    };
    let leading_prepared = prepared_sequence_insert(&leading, &catalog);
    assert_eq!(
        leading_prepared
            .sequence_requests()
            .iter()
            .map(|request| (
                request.row_ordinal,
                request.catalog_column_ordinal,
                request.expression_ordinal
            ))
            .collect::<Vec<_>>(),
        vec![(0, 1, 0), (1, 1, 1)],
        "the fully supplied leading serial is not an active default expression"
    );
    let leading_parent = sequence_parent(901, false, 60);
    let leading_bindings = sequence_bindings(&leading_prepared, &leading_parent, 100);
    let leading_sequence = match catalog.relational_catalog["inactive_leading"].columns[1]
        .default
        .as_ref()
    {
        Some(ColumnDefault::SequenceNextVal { sequence, .. }) => sequence,
        _ => panic!("active serial owns a sequence"),
    };
    let (leading_expression, leading_digest) = leading_bindings[0]
        .published_input_identity_for_test()
        .expect("test binding is published");
    assert_eq!(leading_expression, 60);
    assert_eq!(
        leading_digest,
        crate::sequence_value_input_digest(crate::SequenceValueInput {
            parent_txn_id: 901,
            parent_autocommit: false,
            statement_ordinal: InsertStatementOrdinal::FIRST.as_u32(),
            expression_ordinal: 60,
            parent_request_digest: gpu_db_wal::canonical_request_digest(
                b"typed-insert-sequence-parent",
            ),
            source_name: leading_sequence,
            operation: crate::BinarySequenceValueOperation::Default,
            set_value: None,
        })
    );

    let Command::Insert(middle) =
        parse_command("INSERT INTO inactive_middle (first, supplied_middle) VALUES (DEFAULT, 9)")
            .unwrap()
    else {
        panic!("parser must produce INSERT");
    };
    let middle_prepared = prepared_sequence_insert(&middle, &catalog);
    assert_eq!(
        middle_prepared
            .sequence_requests()
            .iter()
            .map(|request| (
                request.row_ordinal,
                request.catalog_column_ordinal,
                request.expression_ordinal
            ))
            .collect::<Vec<_>>(),
        vec![(0, 0, 0), (0, 2, 1)],
        "the fully supplied middle serial does not reserve a later active slot"
    );
    let middle_parent = sequence_parent(902, false, 80);
    let middle_bindings = sequence_bindings(&middle_prepared, &middle_parent, 200);
    let tail_sequence = match catalog.relational_catalog["inactive_middle"].columns[2]
        .default
        .as_ref()
    {
        Some(ColumnDefault::SequenceNextVal { sequence, .. }) => sequence,
        _ => panic!("tail serial owns a sequence"),
    };
    let (tail_expression, tail_digest) = middle_bindings[1]
        .published_input_identity_for_test()
        .expect("test binding is published");
    assert_eq!(tail_expression, 81);
    let expected_tail_digest = crate::sequence_value_input_digest(crate::SequenceValueInput {
        parent_txn_id: 902,
        parent_autocommit: false,
        statement_ordinal: InsertStatementOrdinal::FIRST.as_u32(),
        expression_ordinal: 81,
        parent_request_digest: gpu_db_wal::canonical_request_digest(
            b"typed-insert-sequence-parent",
        ),
        source_name: tail_sequence,
        operation: crate::BinarySequenceValueOperation::Default,
        set_value: None,
    });
    assert_eq!(tail_digest, expected_tail_digest);
    assert_ne!(
        tail_digest,
        crate::sequence_value_input_digest(crate::SequenceValueInput {
            parent_txn_id: 902,
            parent_autocommit: false,
            statement_ordinal: InsertStatementOrdinal::FIRST.as_u32(),
            expression_ordinal: 82,
            parent_request_digest: gpu_db_wal::canonical_request_digest(
                b"typed-insert-sequence-parent",
            ),
            source_name: tail_sequence,
            operation: crate::BinarySequenceValueOperation::Default,
            set_value: None,
        })
    );
}

#[test]
fn sequence_default_seal_rejects_missing_duplicate_order_identity_and_value_drift() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE sequence_seal (first serial, second serial)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let Command::Insert(insert) =
        parse_command("INSERT INTO sequence_seal (second) VALUES (DEFAULT), (DEFAULT)").unwrap()
    else {
        panic!("parser must produce INSERT");
    };
    let parent = sequence_parent(801, false, 0);
    let other_parent = sequence_parent(802, false, 0);
    let seal_rejects = |mutate: &dyn Fn(&mut Vec<sequence_defaults::SequenceDefaultBinding>)| {
        let prepared = prepared_sequence_insert(&insert, &catalog);
        let mut bindings = sequence_bindings(&prepared, &parent, 10);
        mutate(&mut bindings);
        sequence_defaults::reset_materialization_write_count();
        assert!(prepared
            .seal(
                sequence_defaults::SequenceDefaultBindings::from_bindings(parent.clone(), bindings),
                false,
                false,
            )
            .is_err());
        assert_eq!(
            sequence_defaults::materialization_write_count(),
            0,
            "whole-bundle preflight rejects late binding sabotage before any earlier default cell mutates"
        );
    };
    let prepared = prepared_sequence_insert(&insert, &catalog);
    assert!(prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false
        )
        .is_err());
    seal_rejects(&|bindings| {
        bindings[1] = sequence_defaults::SequenceDefaultBinding::published(
            bindings[0].request_mut_for_test().clone(),
            parent.clone(),
            10,
        )
    });
    seal_rejects(&|bindings| bindings.swap(0, 1));
    seal_rejects(&|bindings| bindings[0].request_mut_for_test().sequence_oid += 1);
    seal_rejects(&|bindings| bindings[0].request_mut_for_test().column_id += 1);
    seal_rejects(&|bindings| bindings[0].set_effect_value_for_test(99));
    seal_rejects(&|bindings| bindings[0].clear_published_receipt_identity_for_test());
    seal_rejects(&|bindings| {
        bindings[0] = sequence_defaults::SequenceDefaultBinding::published(
            bindings[0].request_mut_for_test().clone(),
            other_parent.clone(),
            10,
        )
    });
    seal_rejects(&|bindings| {
        bindings[0] = sequence_defaults::SequenceDefaultBinding::private(
            bindings[2].request_mut_for_test().clone(),
            parent.clone(),
            12,
        )
    });
    seal_rejects(&|bindings| {
        bindings[0] = sequence_defaults::SequenceDefaultBinding::published(
            bindings[0].request_mut_for_test().clone(),
            parent.clone(),
            i64::from(i32::MAX) + 1,
        )
    });
}

#[test]
fn semantic_lowering_leaves_result_wal_apply_and_route_authority_outside_new_leaves() {
    let returning = include_str!("returning.rs");
    let semantics = include_str!("semantics.rs");
    let sequence = include_str!("sequence_defaults.rs");
    let sequence_effects = include_str!("sequence_defaults/effects.rs");
    let builder = include_str!("builder.rs");
    for source in [returning, semantics, sequence, sequence_effects] {
        assert!(!source.contains("project_dml_returning"));
        assert!(!source.contains("WriteDelta"));
        assert!(!source.contains("Vec<Vec<SqlValue>>"));
        assert!(!source.contains("into_resident_append_source"));
    }
    assert!(!returning.contains("DmlExecutionResult"));
    assert!(!sequence.contains("execute_sequence"));
    assert!(!sequence.contains("apply_and_publish"));
    assert!(!sequence_effects.contains("apply_and_publish"));
    assert!(!builder.contains("coerce_insert_value"));
    assert!(!builder.contains("defaults::resolve"));
}
