use super::*;

fn prepare(engine: &Engine, sql: &str) -> TypedInsertBatch {
    let command = parse_command(sql).expect("test INSERT parses");
    let catalog = engine.catalog_snapshot();
    try_prepare_typed_insert_batch(&command, &catalog, catalog.commit_seq, None)
        .expect("default lowering has no semantic error")
        .expect("literal and absent defaults are directly eligible")
}

fn i32_values(column: &TypedInsertColumn) -> &[i32] {
    let TypedInsertColumnValues::I32(values) = &column.values else {
        panic!("expected int4 vector");
    };
    values
}

fn bool_bits(column: &TypedInsertColumn) -> &[u32] {
    let TypedInsertColumnValues::BoolBits(words) = &column.values else {
        panic!("expected bool bitmap vector");
    };
    words
}

fn text_parts(column: &TypedInsertColumn) -> (&[u64], &[u8]) {
    let TypedInsertColumnValues::Text { offsets, bytes } = &column.values else {
        panic!("expected text vector");
    };
    (offsets, bytes)
}

fn bitmap(column: &TypedInsertColumn) -> &[u32] {
    let TypedInsertDefaultResolution::Bitmap(words) = &column.default_resolution else {
        panic!("default-bearing vector needs a compact resolution bitmap");
    };
    words
}

#[test]
fn literal_absent_and_explicit_defaults_rebuild_mixed_text_bool_rows_in_catalog_order() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE default_vectors (id int4, note text DEFAULT '', enabled bool DEFAULT true, spare int4)",
        )
        .unwrap();
    let batch = prepare(
        &engine,
        "INSERT INTO default_vectors (enabled, id, note) VALUES \
         (DEFAULT, 1, DEFAULT), (DEFAULT, 2, NULL), (false, 3, 'direct')",
    );

    assert_eq!(i32_values(&batch.columns[0]), &[1, 2, 3]);
    assert_eq!(
        batch.columns[1].input_states.as_ref(),
        &[
            TypedInsertInputState::ExplicitDefault,
            TypedInsertInputState::ProvidedNull,
            TypedInsertInputState::Provided,
        ]
    );
    let (offsets, bytes) = text_parts(&batch.columns[1]);
    assert_eq!(offsets, &[0, 0, 0, 6]);
    assert_eq!(bytes, b"direct");
    let TypedInsertColumnValidity::Bitmap(note_validity) = &batch.columns[1].validity else {
        panic!("empty-string default and direct text are valid while explicit NULL is not");
    };
    assert_eq!(note_validity.as_ref(), &[0b101]);
    assert_eq!(bitmap(&batch.columns[1]), &[0b001]);

    assert_eq!(
        batch.columns[2].input_states.as_ref(),
        &[
            TypedInsertInputState::ExplicitDefault,
            TypedInsertInputState::ExplicitDefault,
            TypedInsertInputState::Provided,
        ]
    );
    assert_eq!(bool_bits(&batch.columns[2]), &[0b011]);
    assert!(matches!(
        batch.columns[2].validity,
        TypedInsertColumnValidity::AllValid
    ));
    assert_eq!(bitmap(&batch.columns[2]), &[0b011]);

    assert_eq!(
        batch.columns[3].input_states.as_ref(),
        &[
            TypedInsertInputState::Omitted,
            TypedInsertInputState::Omitted,
            TypedInsertInputState::Omitted,
        ]
    );
    let TypedInsertColumnValidity::Bitmap(spare_validity) = &batch.columns[3].validity else {
        panic!("no declared default must materialize as SQL NULL");
    };
    assert_eq!(spare_validity.as_ref(), &[0]);
    assert_eq!(bitmap(&batch.columns[3]), &[0b111]);
    assert!(batch.columns.iter().all(|column| {
        matches!(column.presence, TypedInsertColumnPresence::AllProvided)
            && column.all_inputs_are_resolved(3)
    }));

    let mut encoded = Vec::new();
    batch
        .append_binary_insert_template_row(0, &mut encoded)
        .unwrap();
    assert_eq!(encoded, b"i:1|t:|b:t|null");
    encoded.clear();
    batch
        .append_binary_insert_template_row(1, &mut encoded)
        .unwrap();
    assert_eq!(encoded, b"i:2|null|b:t|null");
}

#[test]
fn default_resolution_bitmaps_zero_tail_bits_at_row_thirty_three() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE default_tail (id int4, flag bool DEFAULT true, absent int4)",
        )
        .unwrap();
    let values = (0..33)
        .map(|row| format!("({row})"))
        .collect::<Vec<_>>()
        .join(",");
    let batch = prepare(
        &engine,
        &format!("INSERT INTO default_tail (id) VALUES {values}"),
    );

    assert_eq!(bool_bits(&batch.columns[1]), &[u32::MAX, 1]);
    assert!(matches!(
        batch.columns[1].validity,
        TypedInsertColumnValidity::AllValid
    ));
    assert_eq!(bitmap(&batch.columns[1]), &[u32::MAX, 1]);
    let TypedInsertColumnValidity::Bitmap(absent_validity) = &batch.columns[2].validity else {
        panic!("no default remains SQL NULL");
    };
    assert_eq!(absent_validity.as_ref(), &[0, 0]);
    assert_eq!(bitmap(&batch.columns[2]), &[u32::MAX, 1]);
    assert!(batch
        .columns
        .iter()
        .all(|column| column.full_invariants_hold(33)));
}

#[test]
fn sequence_default_is_deferred_only_when_requested_and_all_supplied_rows_are_typed_eligible() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE SEQUENCE direct_sequence")
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE sequence_defaults (id int4 DEFAULT nextval('direct_sequence'::regclass), payload int4)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let supplied =
        parse_command("INSERT INTO sequence_defaults (payload, id) VALUES (10, 7), (11, 8)")
            .unwrap();
    assert!(
        try_prepare_typed_insert_batch(&supplied, &catalog, catalog.commit_seq, None)
            .unwrap()
            .is_some()
    );
    assert!(
        !engine.is_concurrent_dml_command(&supplied),
        "the current concurrent admission conservatively reserves all sequence-owning tables"
    );

    let requested = parse_command("INSERT INTO sequence_defaults (payload) VALUES (12)").unwrap();
    assert!(
        try_prepare_typed_insert_batch(&requested, &catalog, catalog.commit_seq, None)
            .unwrap()
            .is_none()
    );
    assert!(!engine.is_concurrent_dml_command(&requested));
}

#[test]
fn resident_append_returning_defers_before_scalar_default_lowering() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE returning_default_gate (id int4, value int4 DEFAULT 1.5::numeric(10,1))",
        )
        .unwrap();
    crate::column_default::reset_scalar_default_evaluation_count("value");
    let command = parse_command(
        "INSERT INTO returning_default_gate (id) VALUES (1), (2) RETURNING id, value",
    )
    .unwrap();
    let catalog = engine.catalog_snapshot();

    assert!(
        try_prepare_typed_insert_batch(&command, &catalog, catalog.commit_seq, None)
            .unwrap()
            .is_none(),
        "the live resident-append adapter must defer RETURNING before default lowering"
    );
    assert_eq!(
        crate::column_default::scalar_default_evaluation_count("value"),
        0,
        "the deferred live adapter must not evaluate a scalar default the legacy route owns"
    );
}

#[test]
fn resident_append_sequence_decline_precedes_scalar_default_lowering() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE SEQUENCE sequence_gate_seq")
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE sequence_scalar_gate (serial_value int4 DEFAULT nextval('sequence_gate_seq'::regclass), scalar_value int4 DEFAULT 1.5::numeric(10,1))",
        )
        .unwrap();
    crate::column_default::reset_scalar_default_evaluation_count("scalar_value");
    let command =
        parse_command("INSERT INTO sequence_scalar_gate (serial_value) VALUES (DEFAULT)").unwrap();
    let catalog = engine.catalog_snapshot();

    assert!(
        try_prepare_typed_insert_batch(&command, &catalog, catalog.commit_seq, None)
            .unwrap()
            .is_none(),
        "a requested sequence default has no resident-append effect owner"
    );
    assert_eq!(
        crate::column_default::scalar_default_evaluation_count("scalar_value"),
        0,
        "the declined live adapter must not evaluate an unrelated scalar default"
    );

    let supplied_sequence =
        parse_command("INSERT INTO sequence_scalar_gate (serial_value) VALUES (17)").unwrap();
    assert!(
        try_prepare_typed_insert_batch(
            &supplied_sequence,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .is_some(),
        "a supplied sequence column must not over-decline solely because another scalar default is omitted"
    );
    assert_eq!(
        crate::column_default::scalar_default_evaluation_count("scalar_value"),
        1,
        "the eligible typed route owns its one scalar broadcast evaluation"
    );

    crate::column_default::reset_scalar_default_evaluation_count("scalar_value");
    engine.execute_text(3, "BEGIN").unwrap();
    engine
        .execute_text(
            3,
            "INSERT INTO sequence_scalar_gate (serial_value) VALUES (DEFAULT)",
        )
        .unwrap();
    assert_eq!(
        crate::column_default::scalar_default_evaluation_count("scalar_value"),
        1,
        "the serialized sequence owner must evaluate the scalar default once"
    );
    engine.execute_text(3, "COMMIT").unwrap();
    let sequence = engine
        .relational_catalog_sequence("sequence_gate_seq")
        .unwrap();
    assert_eq!(
        (sequence.last_value, sequence.is_called),
        (1, true),
        "the one requested sequence default advances exactly once"
    );
}
