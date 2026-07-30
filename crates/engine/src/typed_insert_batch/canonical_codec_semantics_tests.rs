//! Semantic identity coverage for the inert typed-INSERT codec.

use super::*;

fn seal(
    insert: &crate::Insert,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
) -> TypedInsertBatch {
    prepare_typed_insert_semantics(insert, catalog, prepared_catalog_seq, None)
        .expect("semantic fixture preparation succeeds")
        .expect("semantic fixture is current")
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("semantic fixture seals")
}

fn rich_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            41,
            "CREATE TABLE codec_rich (small int2, id int4, big int8, amount numeric(10,2), enabled bool, note text, day date, happened timestamp, uid uuid)",
        )
        .expect("rich fixture table creates");
    let insert = crate::Insert {
        table: "codec_rich".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(vec![
            vec![
                crate::SqlValue::Int2(-7),
                crate::SqlValue::Int4(8),
                crate::SqlValue::Int8(-9),
                crate::SqlValue::Numeric(crate::Decimal128::new(1_234_567_890, 2)),
                crate::SqlValue::Bool(true),
                crate::SqlValue::Text(String::new()),
                crate::SqlValue::Date(20_000),
                crate::SqlValue::Timestamp(1_785_168_000_000_000),
                crate::SqlValue::Uuid([0x5a; 16]),
            ],
            vec![
                crate::SqlValue::Null,
                crate::SqlValue::Null,
                crate::SqlValue::Null,
                crate::SqlValue::Null,
                crate::SqlValue::Null,
                crate::SqlValue::Null,
                crate::SqlValue::Null,
                crate::SqlValue::Null,
                crate::SqlValue::Null,
            ],
        ]),
        returning: vec![
            "note".to_string(),
            "id".to_string(),
            "note".to_string(),
            crate::PROJECTION_WILDCARD_SENTINEL.to_string(),
        ],
    };
    let catalog = engine.catalog_snapshot();
    seal(&insert, &catalog, catalog.commit_seq)
}

fn golden_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(44, "CREATE TABLE codec_golden (id int4)")
        .expect("golden fixture creates");
    let crate::Command::Insert(insert) =
        crate::parse_command("INSERT INTO codec_golden VALUES (7)").expect("golden literal parses")
    else {
        panic!("golden fixture is INSERT");
    };
    let catalog = engine.catalog_snapshot();
    seal(&insert, &catalog, catalog.commit_seq)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn canonical_codec_golden_bytes_and_header_digests() {
    const GOLDEN_HEX: &str = "47505544425459504544494e5331000001000100010001000000000008000000540100004b8312552549ca641423ccd4b573c9c62fe63fc47d2a34c9a1acb64971cbb469929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b010000004a000000060000007075626c69630c000000636f6465635f676f6c64656e00400000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c1461000000000100000001000000020000003e00000001000000000000000200000069640100000001000200000017000000040001000000000000000001000000010200000000010100000004000000070000000300000047000000010000000000000001060000007075626c69630c000000636f6465635f676f6c64656e00400000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c1461040000000400000000000000050000000400000000000000060000000400000000000000070000003400000001000000000000000000000000000000929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b0000000008000000050000000000000000";
    const STATEMENT_DIGEST: &str =
        "4b8312552549ca641423ccd4b573c9c62fe63fc47d2a34c9a1acb64971cbb469";
    const RETURNING_DIGEST: &str =
        "929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b";
    let bytes = encode(&golden_batch()).expect("golden record encodes");
    assert_eq!(hex(&bytes), GOLDEN_HEX);
    assert_eq!(hex(&bytes[36..68]), STATEMENT_DIGEST);
    assert_eq!(hex(&bytes[68..100]), RETURNING_DIGEST);
}

#[test]
fn canonical_codec_round_trips_all_nine_types_and_null_empty_text_distinction() {
    let batch = rich_batch();
    let bytes = encode(&batch).expect("rich record encodes");
    assert_eq!(
        decode(&bytes).expect("rich record decodes").reencode(),
        bytes
    );
    assert!(matches!(
        &batch.columns[5].values,
        TypedInsertColumnValues::Text { offsets, bytes }
            if offsets.as_ref() == [0, 0, 0] && bytes.is_empty()
    ));
    assert!(batch
        .columns
        .iter()
        .all(|column| { column.validity.is_valid(0) && !column.validity.is_valid(1) }));
    assert_eq!(
        batch
            .returning
            .effect_projection_identities()
            .map(|projection| projection.catalog_column_ordinal())
            .collect::<Vec<_>>(),
        vec![5, 1, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "RETURNING duplicates and wildcard expansion retain exact order"
    );
}

#[test]
fn canonical_codec_refuses_forged_index_and_foreign_key_closure_before_hashing() {
    let mut index_forgery = super::tests::catalog_closure_batch();
    index_forgery.canonical_catalog.indexes[0].raw_ordinal = 9;
    assert!(encode(&index_forgery).is_err());

    let mut foreign_key_forgery = super::tests::catalog_closure_batch();
    foreign_key_forgery.canonical_catalog.foreign_keys[0]
        .parent_column
        .column_id ^= 0x40;
    assert!(encode(&foreign_key_forgery).is_err());
}

#[test]
fn canonical_codec_carries_literal_bound_programmatic_omitted_and_default_provenance() {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            42,
            "CREATE TABLE codec_provenance (id int4, bound int4, omitted int4 DEFAULT 19, explicit_default int4 DEFAULT 23)",
        )
        .expect("provenance fixture creates");
    let crate::Command::Insert(literal) = crate::parse_command(
        "INSERT INTO codec_provenance (id, explicit_default) VALUES (7, DEFAULT)",
    )
    .expect("literal fixture parses") else {
        panic!("fixture is INSERT");
    };
    let catalog = engine.catalog_snapshot();
    let literal_batch = seal(&literal, &catalog, catalog.commit_seq);
    assert!(matches!(
        literal_batch.columns[0].input_provenance[0],
        TypedInsertInputProvenance::Literal
    ));
    assert!(matches!(
        literal_batch.columns[1].input_provenance[0],
        TypedInsertInputProvenance::Omitted
    ));
    assert!(matches!(
        literal_batch.columns[2].input_provenance[0],
        TypedInsertInputProvenance::Omitted
    ));
    assert!(matches!(
        literal_batch.columns[3].input_provenance[0],
        TypedInsertInputProvenance::SqlDefault
    ));
    decode(&encode(&literal_batch).expect("literal record encodes"))
        .expect("literal record decodes");

    let bound = crate::PreparedCommand::parse(
        "INSERT INTO codec_provenance (bound, explicit_default, id) VALUES ($1, DEFAULT, $2)",
    )
    .expect("bound fixture parses")
    .bind(&[crate::SqlValue::Null, crate::SqlValue::Int4(8)])
    .expect("bound fixture binds");
    let crate::Command::Insert(bound_insert) = bound.command() else {
        panic!("bound fixture is INSERT");
    };
    let bound_batch = seal(bound_insert, &catalog, catalog.commit_seq);
    assert!(matches!(
        bound_batch.columns[0].input_provenance[0],
        TypedInsertInputProvenance::BoundParameter { index: 2 }
    ));
    assert!(matches!(
        bound_batch.columns[1].input_provenance[0],
        TypedInsertInputProvenance::BoundParameter { index: 1 }
    ));
    decode(&encode(&bound_batch).expect("bound record encodes")).expect("bound record decodes");

    let programmatic = crate::Insert {
        table: "codec_provenance".to_string(),
        columns: vec!["id".to_string(), "explicit_default".to_string()],
        rows: vec![vec![
            crate::InsertCell::programmatic(crate::SqlValue::Int4(9)),
            crate::InsertCell::programmatic_default(),
        ]],
        returning: Vec::new(),
    };
    let programmatic_batch = seal(&programmatic, &catalog, catalog.commit_seq);
    assert!(matches!(
        programmatic_batch.columns[0].input_provenance[0],
        TypedInsertInputProvenance::ProgrammaticValue
    ));
    assert!(matches!(
        programmatic_batch.columns[3].input_provenance[0],
        TypedInsertInputProvenance::ProgrammaticDefault
    ));
    decode(&encode(&programmatic_batch).expect("programmatic record encodes"))
        .expect("programmatic record decodes");
}

#[test]
fn typed_statement_digest_is_value_and_order_sensitive_but_not_catalog_handle_or_prepare_seq_sensitive(
) {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(43, "CREATE TABLE codec_identity (left int4, right int4)")
        .expect("identity fixture creates");
    let catalog = engine.catalog_snapshot();
    let independent_catalog = std::sync::Arc::new((*catalog).clone());
    assert_ne!(
        std::sync::Arc::as_ptr(&catalog),
        std::sync::Arc::as_ptr(&independent_catalog),
        "the same immutable catalog contents use distinct Arc allocations"
    );
    let ordered = crate::Insert {
        table: "codec_identity".to_string(),
        columns: vec!["left".to_string(), "right".to_string()],
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(1),
            crate::SqlValue::Int4(2),
        ]]),
        returning: Vec::new(),
    };
    let reordered = crate::Insert {
        table: "codec_identity".to_string(),
        columns: vec!["right".to_string(), "left".to_string()],
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(2),
            crate::SqlValue::Int4(1),
        ]]),
        returning: Vec::new(),
    };
    let changed = crate::Insert {
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(1),
            crate::SqlValue::Int4(3),
        ]]),
        ..ordered.clone()
    };
    let first = seal(&ordered, &catalog, catalog.commit_seq);
    let mut same_handle_new_prepare_seq = seal(&ordered, &independent_catalog, catalog.commit_seq);
    same_handle_new_prepare_seq.table.prepared_catalog_seq = catalog.commit_seq + 99;
    let reordered = seal(&reordered, &catalog, catalog.commit_seq);
    let changed = seal(&changed, &catalog, catalog.commit_seq);
    assert_eq!(
        first.typed_statement_digest,
        same_handle_new_prepare_seq.typed_statement_digest
    );
    assert_eq!(
        encode(&first).expect("first record"),
        encode(&same_handle_new_prepare_seq).expect("same semantic record")
    );
    assert_ne!(
        first.typed_statement_digest,
        reordered.typed_statement_digest
    );
    assert_ne!(first.typed_statement_digest, changed.typed_statement_digest);
}
