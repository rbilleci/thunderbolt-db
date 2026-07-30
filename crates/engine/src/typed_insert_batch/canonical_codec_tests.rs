use super::*;

struct BytewiseCanonicalSource<'a>(&'a [u8]);

impl CanonicalTypedInsertReadAt for BytewiseCanonicalSource<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.0.len()).expect("fixture source length fits u64")
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let start = usize::try_from(offset).map_err(|_| codec_error("fixture offset overflows"))?;
        let end = start
            .checked_add(out.len())
            .ok_or_else(|| codec_error("fixture source range overflows"))?;
        let source = self
            .0
            .get(start..end)
            .ok_or_else(|| codec_error("fixture source is truncated"))?;
        for (destination, source) in out.iter_mut().zip(source) {
            *destination = *source;
        }
        Ok(())
    }
}

pub(super) fn prepared_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE codec_probe (id int4, note text, amount numeric(4,2))",
        )
        .expect("codec fixture table creates");
    let insert = crate::Insert {
        table: "codec_probe".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(vec![
            vec![
                crate::SqlValue::Int4(7),
                crate::SqlValue::Text("alpha".to_string()),
                crate::SqlValue::Numeric(crate::Decimal128::new(1_234, 2)),
            ],
            vec![
                crate::SqlValue::Int4(8),
                crate::SqlValue::Null,
                crate::SqlValue::Numeric(crate::Decimal128::new(-99, 2)),
            ],
        ]),
        returning: vec!["note".to_string(), "id".to_string(), "note".to_string()],
    };
    let catalog = engine.catalog_snapshot();
    let prepared = prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("codec fixture semantic preparation succeeds")
        .expect("current catalog generation prepares");
    prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("codec fixture seals without sequence effects")
}

#[test]
fn canonical_codec_encoded_len_runs_the_exact_encoder_traversal_without_materializing() {
    for batch in [prepared_batch(), serial_batch(41), catalog_closure_batch()] {
        let encoded = encode(&batch).expect("fixture encodes");
        assert_eq!(
            encoded_len(&batch).expect("counting traversal succeeds"),
            encoded.len()
        );
    }
}

#[test]
fn canonical_decoder_uses_fallible_exact_vector_reservations_and_retries_cleanly() {
    let bytes = encode(&prepared_batch()).expect("canonical record encodes");
    let (_, stats) = decode::observe_stats_for_test(|| decode(&bytes).expect("record decodes"));
    let owners = stats.attempts;
    assert!(owners != 0, "decoded fixture must retain owned vectors");
    assert!(
        stats.persistent_bytes != 0,
        "decoded fixture must reserve retained owner bytes"
    );
    assert!(
        stats.scratch_bytes != 0,
        "canonical reencode scratch must be reserved/injected"
    );
    for owner in 1..=owners {
        let failed = decode::fail_at_for_test(owner, || decode(&bytes));
        assert!(failed.is_err(), "decoded owner {owner} must fail closed");
        assert!(
            decode(&bytes).is_ok(),
            "decoded owner {owner} must drain for retry"
        );
    }
}

#[test]
fn canonical_source_measure_binds_bytewise_copy_before_model_decode() {
    let bytes = encode(&prepared_batch()).expect("canonical source fixture encodes");
    let source = BytewiseCanonicalSource(&bytes);
    let (measure, stats) = decode::observe_stats_for_test(|| {
        measure_decoded_canonical_typed_insert_from_source(&source)
            .expect("bytewise source raw pass succeeds")
    });
    assert_eq!(
        stats.attempts, 0,
        "successful raw S2 measure must not reserve decoded owners"
    );
    assert_eq!(
        measure.record_bytes(),
        u64::try_from(bytes.len()).expect("fixture fits u64")
    );
    assert!(measure.persistent_bytes() != 0);
    assert_eq!(
        measure
            .maximum_with_record_copy_bytes()
            .expect("fixture scratch peak fits u64"),
        u64::try_from(bytes.len())
            .expect("fixture fits u64")
            .checked_mul(2)
            .expect("fixture peak fits u64"),
        "the raw copy and final exact reencode are concurrently live"
    );
    assert_eq!(
        measure
            .maximum_with_record_copy_allocation_slots()
            .expect("fixture scratch slots fit u64"),
        2,
        "the raw copy and final exact reencode are two concurrent allocations"
    );

    let mut copy = vec![0_u8; bytes.len()];
    copy_decoded_canonical_typed_insert_after_measure(&source, measure, &mut copy)
        .expect("copy remeasures and preserves bytewise source evidence");
    let (decoded, observed) = decode::observe_stats_for_test(|| {
        decode_decoded_canonical_typed_insert_after_measure(&copy, measure)
            .expect("opaque measure authorizes only its copied bytes")
    });
    assert_eq!(decoded.facts().row_count, 2);
    assert_eq!(
        observed.persistent_bytes,
        measure.persistent_bytes(),
        "S2 raw measure must equal all retained decoder owners"
    );
    assert_eq!(
        observed.persistent_slots,
        measure.persistent_allocation_slots(),
        "S2 raw measure must equal every retained allocation slot"
    );
    assert_eq!(
        observed.scratch_bytes,
        measure.maximum_scratch_bytes(),
        "S2 raw measure must equal peak reencode scratch"
    );
    assert_eq!(
        observed.scratch_slots,
        measure.maximum_scratch_allocation_slots(),
        "S2 raw measure must equal peak scratch allocation slots"
    );
    assert_eq!(
        observed.attempts,
        measure
            .persistent_allocation_slots()
            .checked_add(measure.maximum_scratch_allocation_slots())
            .expect("fixture reservation attempts fit u64"),
        "every measured persistent/scratch allocation slot must have one exact attempt; \
         persistent slots observed={} measured={}, scratch slots observed={} measured={}",
        observed.persistent_slots,
        measure.persistent_allocation_slots(),
        observed.scratch_slots,
        measure.maximum_scratch_allocation_slots(),
    );
    for owner in 1..=observed.attempts {
        let rejected = decode::fail_at_for_test(owner, || {
            decode_decoded_canonical_typed_insert_after_measure(&copy, measure)
        });
        assert!(rejected.is_err(), "S2 owner {owner} must reject cleanly");
        assert!(
            decode_decoded_canonical_typed_insert_after_measure(&copy, measure).is_ok(),
            "S2 owner {owner} failure must drain for a retry"
        );
    }
    let last_owner = observed
        .attempts
        .checked_add(1)
        .expect("fixture owner count fits injection domain");
    for (persistent_bytes, persistent_slots, scratch_bytes, scratch_slots) in [
        (Some(measure.persistent_bytes() - 1), None, None, None),
        (
            None,
            Some(measure.persistent_allocation_slots() - 1),
            None,
            None,
        ),
        (None, None, Some(measure.maximum_scratch_bytes() - 1), None),
        (
            None,
            None,
            None,
            Some(measure.maximum_scratch_allocation_slots() - 1),
        ),
    ] {
        let rejected = decode::fail_with_limits_for_test(
            last_owner,
            persistent_bytes,
            persistent_slots,
            scratch_bytes,
            scratch_slots,
            || decode_decoded_canonical_typed_insert_after_measure(&copy, measure),
        );
        assert!(rejected.is_err(), "one-unit-below S2 budget must refuse");
        assert!(
            decode_decoded_canonical_typed_insert_after_measure(&copy, measure).is_ok(),
            "one-unit-below S2 refusal must drain for retry"
        );
    }

    let mut stale = copy.clone();
    stale[0] ^= 1;
    let stale_source = BytewiseCanonicalSource(&stale);
    let mut stale_destination = vec![0_u8; stale.len()];
    assert!(
        copy_decoded_canonical_typed_insert_after_measure(
            &stale_source,
            measure,
            &mut stale_destination,
        )
        .is_err(),
        "same-length source drift must fail before copy"
    );
    assert!(
        decode_decoded_canonical_typed_insert_after_measure(&stale, measure).is_err(),
        "same-length source drift must fail before model-owner allocation"
    );

    let first_serial = encode(&serial_batch(41)).expect("first same-length serial fixture encodes");
    let second_serial =
        encode(&serial_batch(42)).expect("second same-length serial fixture encodes");
    assert_eq!(
        first_serial.len(),
        second_serial.len(),
        "serial values must give a same-length content-drift fixture"
    );
    let first_measure =
        measure_decoded_canonical_typed_insert_from_source(&BytewiseCanonicalSource(&first_serial))
            .expect("first serial source measures");
    let second_source = BytewiseCanonicalSource(&second_serial);
    let mut same_length_destination = vec![0_u8; second_serial.len()];
    assert!(
        copy_decoded_canonical_typed_insert_after_measure(
            &second_source,
            first_measure,
            &mut same_length_destination,
        )
        .is_err(),
        "the source fingerprint must refuse a distinct valid same-length S2 record"
    );
}

#[test]
fn canonical_decoder_source_guard_has_no_arc_or_implicit_byte_owner() {
    let decoder = include_str!("canonical_codec_decode.rs");
    let reservation = include_str!("canonical_codec_decode_reservation.rs");
    assert!(!decoder.contains("Arc::from"));
    assert!(!decoder.contains("String::from_utf8"));
    assert!(!decoder.contains(".to_vec()"));
    assert!(!decoder.contains("Vec::with_capacity"));
    assert!(decoder.contains("reserve_string"));
    assert!(reservation.contains("reserve_reencode_scratch"));
    assert!(reservation.contains("values.len() != values.capacity()"));
}

pub(super) fn serial_batch(value: i64) -> TypedInsertBatch {
    serial_batch_values(&[value])
}

pub(super) fn serial_batch_values(values: &[i64]) -> TypedInsertBatch {
    assert!(
        !values.is_empty(),
        "sequence fixture needs at least one row"
    );
    let engine = crate::Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE codec_serial (id serial, payload int4)")
        .expect("serial codec fixture table creates");
    let values_sql = values
        .iter()
        .enumerate()
        .map(|(row, _)| format!("(DEFAULT, {})", row + 7))
        .collect::<Vec<_>>()
        .join(", ");
    let crate::Command::Insert(insert) = crate::parse_command(&format!(
        "INSERT INTO codec_serial (id, payload) VALUES {values_sql}"
    ))
    .expect("serial fixture INSERT parses") else {
        panic!("fixture must parse as INSERT");
    };
    let catalog = engine.catalog_snapshot();
    let prepared = prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("serial fixture semantic preparation succeeds")
        .expect("serial fixture prepares");
    let typed_statement_digest = prepared.typed_statement_digest();
    let parent = sequence_defaults::effects::SequenceDefaultParentContext::for_test(
        71,
        true,
        typed_statement_digest,
        InsertStatementOrdinal::FIRST,
        0,
    );
    let bindings = prepared
        .sequence_requests()
        .iter()
        .cloned()
        .zip(values.iter().copied())
        .map(|(request, value)| {
            sequence_defaults::SequenceDefaultBinding::published(request, parent.clone(), value)
        })
        .collect();
    prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::from_bindings(parent, bindings),
            false,
            false,
        )
        .expect("serial fixture seals")
}

pub(super) fn serial_index_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE codec_serial_index (id serial PRIMARY KEY, payload int4)",
        )
        .expect("serial/index codec fixture table creates");
    let crate::Command::Insert(insert) =
        crate::parse_command("INSERT INTO codec_serial_index (id, payload) VALUES (DEFAULT, 7)")
            .expect("serial/index codec fixture INSERT parses")
    else {
        panic!("fixture must parse as INSERT");
    };
    let catalog = engine.catalog_snapshot();
    let prepared = prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("serial/index fixture semantic preparation succeeds")
        .expect("serial/index fixture prepares");
    let typed_statement_digest = prepared.typed_statement_digest();
    let parent = sequence_defaults::effects::SequenceDefaultParentContext::for_test(
        72,
        true,
        typed_statement_digest,
        InsertStatementOrdinal::FIRST,
        0,
    );
    let bindings = prepared
        .sequence_requests()
        .iter()
        .cloned()
        .map(|request| {
            sequence_defaults::SequenceDefaultBinding::published(request, parent.clone(), 1)
        })
        .collect();
    prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::from_bindings(parent, bindings),
            false,
            false,
        )
        .expect("serial/index fixture seals")
}

pub(super) fn external_parent_domain_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(1, "CREATE DOMAIN codec_external_amount AS int4")
        .expect("external domain fixture creates");
    engine
        .execute_text(
            2,
            "CREATE TABLE codec_external_parent (id codec_external_amount PRIMARY KEY)",
        )
        .expect("external domain parent fixture creates");
    engine
        .execute_text(
            3,
            "CREATE TABLE codec_external_child (id int4, parent_id int4)",
        )
        .expect("external domain child fixture creates");
    engine
        .execute_text(
            4,
            "ALTER TABLE ONLY codec_external_child ADD CONSTRAINT codec_external_child_parent_fkey FOREIGN KEY (parent_id) REFERENCES codec_external_parent(id)",
        )
        .expect("external domain foreign key fixture creates");
    let insert = crate::Insert {
        table: "codec_external_child".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(7),
            crate::SqlValue::Int4(11),
        ]]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("external domain fixture semantic preparation succeeds")
        .expect("external domain fixture prepares")
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("external domain fixture seals")
}

pub(super) fn catalog_closure_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(1, "CREATE DOMAIN codec_amount AS int4")
        .expect("domain fixture creates");
    engine
        .execute_text(2, "CREATE TABLE codec_parent (id int4 PRIMARY KEY)")
        .expect("parent fixture creates");
    engine
        .execute_text(
            3,
            "CREATE TABLE codec_child (id int4 UNIQUE, pid int4, amount codec_amount)",
        )
        .expect("child fixture creates");
    engine
        .execute_text(
            4,
            "CREATE INDEX codec_child_pid_amount_idx ON codec_child (pid, amount)",
        )
        .expect("compound target index fixture creates");
    engine
        .execute_text(
            5,
            "ALTER TABLE ONLY codec_child ADD CONSTRAINT codec_child_pid_fkey FOREIGN KEY (pid) REFERENCES codec_parent(id)",
        )
        .expect("foreign-key fixture creates");
    let insert = crate::Insert {
        table: "codec_child".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(7),
            crate::SqlValue::Int4(11),
            crate::SqlValue::Int4(123),
        ]]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let prepared = prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("closure fixture semantic preparation succeeds")
        .expect("closure fixture prepares");
    prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("closure fixture seals")
}

pub(super) fn external_shared_supporting_index_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE codec_fk_parent (id int4 PRIMARY KEY)")
        .expect("shared-support parent table creates");
    engine
        .execute_text(
            2,
            "CREATE TABLE codec_fk_child (id int4, left_id int4, right_id int4)",
        )
        .expect("shared-support child table creates");
    for (txn_id, sql) in [
        (
            3,
            "ALTER TABLE ONLY codec_fk_child ADD CONSTRAINT codec_fk_child_left_fkey FOREIGN KEY (left_id) REFERENCES codec_fk_parent(id)",
        ),
        (
            4,
            "ALTER TABLE ONLY codec_fk_child ADD CONSTRAINT codec_fk_child_right_fkey FOREIGN KEY (right_id) REFERENCES codec_fk_parent(id)",
        ),
    ] {
        engine
            .execute_text(txn_id, sql)
            .expect("shared-support foreign key creates");
    }
    let insert = crate::Insert {
        table: "codec_fk_child".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(7),
            crate::SqlValue::Int4(11),
            crate::SqlValue::Int4(11),
        ]]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("shared-support semantic preparation succeeds")
        .expect("shared-support fixture prepares")
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("shared-support fixture seals")
}

pub(super) fn distinct_external_supporting_indexes_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    for (txn_id, sql) in [
        (
            1,
            "CREATE TABLE codec_left_parent (id int4 PRIMARY KEY)",
        ),
        (
            2,
            "CREATE TABLE codec_right_parent (id int4 PRIMARY KEY)",
        ),
        (
            3,
            "CREATE TABLE codec_two_parent_child (id int4, left_id int4, right_id int4)",
        ),
        (
            4,
            "ALTER TABLE ONLY codec_two_parent_child ADD CONSTRAINT codec_two_parent_child_left_fkey FOREIGN KEY (left_id) REFERENCES codec_left_parent(id)",
        ),
        (
            5,
            "ALTER TABLE ONLY codec_two_parent_child ADD CONSTRAINT codec_two_parent_child_right_fkey FOREIGN KEY (right_id) REFERENCES codec_right_parent(id)",
        ),
    ] {
        engine
            .execute_text(txn_id, sql)
            .expect("distinct-supporting-index fixture catalog setup succeeds");
    }
    let insert = crate::Insert {
        table: "codec_two_parent_child".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(7),
            crate::SqlValue::Int4(11),
            crate::SqlValue::Int4(13),
        ]]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("distinct-supporting-index semantic preparation succeeds")
        .expect("distinct-supporting-index fixture prepares")
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("distinct-supporting-index fixture seals")
}

pub(super) fn self_referencing_catalog_closure_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE codec_self_fk (id int4 PRIMARY KEY, parent_id int4)",
        )
        .expect("self-referencing fixture table creates");
    let mut catalog = (*engine.catalog_snapshot()).clone();
    catalog
        .relational_catalog
        .get_mut("codec_self_fk")
        .expect("self-referencing fixture target exists")
        .foreign_keys
        .push(crate::relational_model::RelationalForeignKey {
            name: "codec_self_fk_parent_fkey".to_string(),
            column: "parent_id".to_string(),
            referenced_table: "codec_self_fk".to_string(),
            referenced_column: "id".to_string(),
        });
    let insert = crate::Insert {
        table: "codec_self_fk".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(vec![vec![
            crate::SqlValue::Int4(7),
            crate::SqlValue::Int4(7),
        ]]),
        returning: Vec::new(),
    };
    let prepared = prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("self-referencing fixture semantic preparation succeeds")
        .expect("self-referencing fixture prepares");
    prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("self-referencing fixture seals")
}

pub(super) fn section_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut offset = HEADER_LEN;
    let mut offsets = Vec::new();
    for _ in 0..SECTION_COUNT {
        offsets.push(offset);
        let length = usize::try_from(u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .expect("section length width"),
        ))
        .expect("section length addressable");
        offset += 8 + length;
    }
    assert_eq!(offset, bytes.len());
    offsets
}

fn reject(mutated: &[u8]) {
    assert!(
        decode(mutated).is_err(),
        "mutated record unexpectedly decoded"
    );
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 width"))
}

fn first_column_offsets(bytes: &[u8]) -> (usize, usize, usize) {
    let columns = section_offsets(bytes)[1] + 8;
    assert_eq!(
        u32_at(bytes, columns),
        3,
        "fixture has three target columns"
    );
    let mut cursor = columns + 4 + 4;
    cursor += 4 + usize::try_from(u32_at(bytes, cursor)).expect("identifier length");
    cursor += 4 + 2 + 4 + 4 + 2;
    for _ in 0..2 {
        let tag = bytes[cursor];
        cursor += 1 + if tag == 1 { 4 } else { 0 };
    }
    let validity = cursor;
    assert_eq!(bytes[validity], 0, "fixture first column is all-valid");
    let presence = validity + 1;
    let defaults = presence + 1;
    assert_eq!(bytes[defaults], 0, "fixture first column has no defaults");
    let values = defaults + 1 + 4 + 2 * 6;
    (presence, values + 1, section_offsets(bytes)[0] + 8)
}

/// Keep every published receipt witness valid while substituting a parent from another typed
/// statement. The header digest is intentionally not touched: parent identity is excluded from
/// the statement intent, so this exercises the separate binding invariant rather than a digest
/// mismatch shortcut.
fn foreign_parent_with_rehashed_receipt(mut bytes: Vec<u8>) -> Vec<u8> {
    let sequence_section = section_offsets(&bytes)[7];
    let payload = sequence_section + 8;
    assert_eq!(bytes[payload], 1, "fixture has a shared sequence parent");
    let parent_txn_id = u64::from_le_bytes(
        bytes[payload + 1..payload + 9]
            .try_into()
            .expect("parent txn width"),
    );
    let parent_autocommit = bytes[payload + 9] == 1;
    let statement_ordinal = u32_at(&bytes, payload + 42);
    let request = payload + 58;
    let source_length = usize::try_from(u32_at(&bytes, request + 20)).expect("source length");
    let source_start = request + 24;
    let source_end = source_start + source_length;
    let source_name = std::str::from_utf8(&bytes[source_start..source_end])
        .expect("fixture sequence name is UTF-8")
        .to_owned();
    let effective_length_offset = source_end;
    let effective_length =
        usize::try_from(u32_at(&bytes, effective_length_offset)).expect("effective length");
    let request_tail = effective_length_offset + 4 + effective_length + 8;
    let absolute_expression_ordinal = u32_at(&bytes, request_tail);
    let effect_tag = request_tail + 4 + 32 + 8;
    assert_eq!(bytes[effect_tag], 1, "fixture uses a published receipt");
    let input_digest_offset = effect_tag + 1 + 8;
    let foreign_digest = [0x4b; 32];
    bytes[payload + 10..payload + 42].copy_from_slice(&foreign_digest);
    let input_digest = crate::sequence_value_input_digest(crate::SequenceValueInput {
        parent_txn_id,
        parent_autocommit,
        statement_ordinal,
        expression_ordinal: absolute_expression_ordinal,
        parent_request_digest: foreign_digest,
        source_name: &source_name,
        operation: crate::BinarySequenceValueOperation::Default,
        set_value: None,
    });
    bytes[input_digest_offset..input_digest_offset + input_digest.len()]
        .copy_from_slice(&input_digest);
    bytes
}

#[test]
fn canonical_codec_round_trips_returning_duplicates_and_nullable_text() {
    let batch = prepared_batch();
    let bytes = encode(&batch).expect("canonical record encodes");
    let decoded = crate::typed_insert_batch::decode_canonical_typed_insert_record(&bytes)
        .expect("canonical record decodes through the narrow facade");
    assert_eq!(decoded.reencode(), bytes);
    assert_eq!(
        decoded.typed_statement_digest(),
        batch.typed_statement_digest
    );
    assert_eq!(
        decoded.returning_digest(),
        returning_layout_digest(&batch.returning).expect("RETURNING layout hashes")
    );
}

#[test]
fn decoded_record_retains_model_facts_without_retaining_raw_record_bytes() {
    let batch = prepared_batch();
    let bytes = encode(&batch).expect("canonical record encodes");
    let decoded = decode(&bytes).expect("canonical record decodes");
    let facts = decoded.facts();
    assert_eq!(facts.typed_statement_digest, batch.typed_statement_digest);
    assert_eq!(facts.statement_ordinal, InsertStatementOrdinal::FIRST);
    assert_eq!(facts.row_count, 2);
    assert_eq!(facts.column_count, 3);
    assert_ne!(facts.target.oid, 0);
    assert_ne!(facts.target.schema_digest, [0; 32]);
    assert_eq!(facts.returning.digest, decoded.returning_digest());
    assert_eq!(facts.returning.row_count, 2);
    assert_eq!(facts.returning.column_count, 3);
    assert_eq!(facts.returning.cell_count, 6);
    assert_eq!(facts.sequence_effect_count, 0);
    assert!(decoded.sequence_parent().is_none());
    assert!(decoded.sequence_effects().next().is_none());

    // Reencoding allocates only when this test asks for it; decode retained the validated model,
    // and the rebuilt canonical bytes must stay exact.
    assert_eq!(decoded.reencode(), bytes);

    let codec_source = include_str!("canonical_codec.rs");
    let decoder_source = include_str!("canonical_codec_decode.rs");
    assert!(codec_source.contains("pub(crate) struct DecodedTypedInsertRecord"));
    assert!(codec_source.contains("model: decode::DecodedModel"));
    assert!(
        !codec_source.contains("bytes: Box<[u8]>")
            && !codec_source
                .contains("derive(Clone, Copy)\npub(crate) struct DecodedTypedInsertRecord")
            && !decoder_source.contains("canonical.into()"),
        "decoded records must stay move-only and model-owned, without a raw byte authority"
    );
}

#[test]
fn decoded_record_exposes_published_sequence_facts_in_canonical_order() {
    let batch = serial_batch_values(&[41, 42]);
    let decoded =
        decode(&encode(&batch).expect("serial record encodes")).expect("serial record decodes");
    let facts = decoded.facts();
    assert_eq!(facts.sequence_effect_count, 2);
    let parent = decoded
        .sequence_parent()
        .expect("nonempty effect vector has one admitted parent");
    assert_eq!(parent.txn_id, 71);
    assert!(parent.autocommit);
    assert_eq!(parent.request_digest, facts.typed_statement_digest);
    assert_eq!(parent.statement_ordinal, InsertStatementOrdinal::FIRST);
    assert_eq!(parent.expression_ordinal_base, 0);

    let effects = decoded.sequence_effects().collect::<Vec<_>>();
    assert_eq!(effects.len(), 2);
    for (expected, effect) in effects.iter().enumerate() {
        assert_eq!(effect.request.effect_ordinal, expected as u32);
        assert_eq!(effect.request.target_table_oid, facts.target.oid);
        assert_eq!(effect.request.row_ordinal, expected as u32);
        assert_eq!(effect.request.catalog_column_ordinal, 0);
        assert_ne!(effect.request.column_id, 0);
        assert_ne!(effect.request.sequence_oid, 0);
        assert_eq!(effect.request.statement_ordinal, facts.statement_ordinal);
        assert_eq!(effect.request.expression_ordinal, expected as u32);
        assert_eq!(effect.request.absolute_expression_ordinal, expected as u32);
        match effect.kind {
            DecodedSequenceEffectKindFacts::Published {
                transition_txn_id,
                input_digest,
                returned_value,
            } => {
                assert_ne!(transition_txn_id, 0);
                assert_ne!(input_digest, [0; 32]);
                assert_eq!(returned_value, effect.resolved_value);
                assert_eq!(returned_value, 41 + expected as i64);
            }
            DecodedSequenceEffectKindFacts::Private { input_digest: _ } => {
                panic!("published fixture must not expose private sequence facts")
            }
        }
    }
}

#[test]
fn canonical_codec_round_trips_domains_indexes_and_foreign_key_closure() {
    let batch = catalog_closure_batch();
    let bytes = encode(&batch).expect("closure record encodes");
    assert_eq!(
        decode(&bytes).expect("closure record decodes").reencode(),
        bytes
    );
}

#[test]
fn canonical_codec_round_trips_self_referencing_foreign_key_closure() {
    let batch = self_referencing_catalog_closure_batch();
    let bytes = encode(&batch).expect("self-referencing closure record encodes");
    assert_eq!(
        decode(&bytes)
            .expect("self-referencing closure record decodes")
            .reencode(),
        bytes
    );
}

#[test]
fn typed_statement_digest_excludes_sequence_outcome_but_record_does_not() {
    let first = serial_batch(41);
    let second = serial_batch(42);
    assert_eq!(first.typed_statement_digest, second.typed_statement_digest);

    let first_bytes = encode(&first).expect("first sequence record encodes");
    let second_bytes = encode(&second).expect("second sequence record encodes");
    assert_ne!(
        first_bytes, second_bytes,
        "resolved sequence outputs stay in the record"
    );
    assert_eq!(&first_bytes[36..68], &second_bytes[36..68]);
    decode(&first_bytes).expect("first sequence record decodes");
    decode(&second_bytes).expect("second sequence record decodes");
}

#[test]
fn canonical_codec_rejects_a_foreign_sequence_parent_after_rehashing_receipt_evidence() {
    let bytes = encode(&serial_batch(41)).expect("sequence record encodes");
    let foreign_parent = foreign_parent_with_rehashed_receipt(bytes);
    reject(&foreign_parent);
}

#[test]
fn canonical_codec_rejects_header_section_and_length_sabotage() {
    let bytes = encode(&prepared_batch()).expect("canonical record encodes");
    let offsets = section_offsets(&bytes);
    let mut sabotage = Vec::new();
    for offset in [0, 16, 18, 20, 22, 24, 28, 30, 32, 36, 68] {
        let mut mutated = bytes.clone();
        mutated[offset] ^= 0x5a;
        sabotage.push(mutated);
    }
    for offset in offsets {
        for field in [offset, offset + 2, offset + 4] {
            let mut mutated = bytes.clone();
            mutated[field] ^= 0x01;
            sabotage.push(mutated);
        }
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    sabotage.push(trailing);
    assert!(
        sabotage.len() >= 35,
        "sabotage matrix must cover every section header"
    );
    for mutated in sabotage {
        reject(&mutated);
    }
}

#[test]
fn canonical_record_prefix_is_the_shared_strict_format_and_bound_authority() {
    let batch = prepared_batch();
    let bytes = encode(&batch).expect("canonical record encodes");
    let header: &[u8; HEADER_LEN] = bytes[..HEADER_LEN]
        .try_into()
        .expect("canonical record has its fixed header");
    let prefix = parse_canonical_typed_insert_record_prefix(header, bytes.len())
        .expect("canonical prefix validates");
    assert_eq!(prefix.typed_statement_digest, batch.typed_statement_digest);
    assert_eq!(
        prefix.returning_digest,
        returning_layout_digest(&batch.returning).expect("RETURNING digest computes")
    );

    assert!(parse_canonical_typed_insert_record_prefix(header, bytes.len() + 1).is_err());
    for offset in [0, 16, 18, 20, 22, 24, 28, 30, 32] {
        let mut sabotaged = bytes.clone();
        sabotaged[offset] ^= 0x5a;
        let header: &[u8; HEADER_LEN] = sabotaged[..HEADER_LEN]
            .try_into()
            .expect("sabotaged record retains fixed header width");
        assert!(
            parse_canonical_typed_insert_record_prefix(header, sabotaged.len()).is_err(),
            "prefix accepted header sabotage at byte {offset}"
        );
    }
}

#[test]
fn canonical_codec_rejects_noncanonical_target_and_vector_forms() {
    let bytes = encode(&prepared_batch()).expect("canonical record encodes");
    let (presence, logical_count, target) = first_column_offsets(&bytes);

    let mut noncanonical_presence = bytes.clone();
    noncanonical_presence[presence] = 1;
    reject(&noncanonical_presence);

    let mut wrong_vector_count = bytes.clone();
    wrong_vector_count[logical_count..logical_count + 4].copy_from_slice(&3_u32.to_le_bytes());
    reject(&wrong_vector_count);

    let mut wrong_target_column_count = bytes;
    let target_column_count = target
        + 4
        + usize::try_from(u32_at(&wrong_target_column_count, target)).expect("schema length")
        + 4
        + usize::try_from(u32_at(
            &wrong_target_column_count,
            target + 4 + usize::try_from(u32_at(&wrong_target_column_count, target)).unwrap(),
        ))
        .expect("table length")
        + 4
        + 32
        + 4
        + 4;
    wrong_target_column_count[target_column_count..target_column_count + 4]
        .copy_from_slice(&4_u32.to_le_bytes());
    reject(&wrong_target_column_count);
}

#[test]
fn canonical_codec_rejects_numeric_precision_drift_before_digest_comparison() {
    let bytes = encode(&prepared_batch()).expect("canonical record encodes");
    let encoded = 1_234_i128.to_le_bytes();
    let value_offset = bytes
        .windows(encoded.len())
        .position(|window| window == encoded)
        .expect("fixture numeric payload is present");
    let mut mutated = bytes;
    mutated[value_offset..value_offset + encoded.len()].copy_from_slice(&10_000_i128.to_le_bytes());
    reject(&mutated);
}

#[test]
fn canonical_codec_stays_inert_and_has_no_legacy_binary_or_json_path() {
    let source = [
        include_str!("canonical_codec.rs"),
        include_str!("canonical_codec_decode.rs"),
        include_str!("canonical_codec_decode_reencode.rs"),
        include_str!("canonical_codec_sequence.rs"),
    ]
    .into_iter()
    .flat_map(|source| source.lines())
    .filter(|line| !line.trim_start().starts_with("//"))
    .collect::<String>();
    for forbidden in [
        "PreparedBinaryInsertTemplate",
        "try_encode_binary_insert",
        "encode_relational_row",
        "decode_binary_insert",
        "BinaryWalRecord",
        "transaction_statement_digest",
        "WalRecord",
        "Cuda",
        "Device",
        "RowId",
        "apply_",
        "recover_",
    ] {
        assert!(
            !source.contains(forbidden),
            "canonical codec must remain separate from legacy/WAL authority: {forbidden}"
        );
    }
}
