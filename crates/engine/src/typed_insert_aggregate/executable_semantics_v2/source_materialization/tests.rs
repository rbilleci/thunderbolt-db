use super::*;
use crate::insert_semantic_ir::InsertStatementOrdinal;
use crate::typed_insert_aggregate::{
    decode_typed_insert_aggregate_bodies, encode_typed_insert_aggregate_bodies,
    reserve_typed_insert_aggregate_bodies, typed_insert_aggregate_status_roots,
    TypedInsertAggregateSectionView, TypedInsertAggregateStatusRoots, TypedInsertAggregateView,
    TypedInsertStatusV2, AGGREGATE_CHUNK_PAYLOAD_BYTES, AGGREGATE_FLAG_CATALOG,
    AGGREGATE_FLAG_EXPLICIT, AGGREGATE_FLAG_PRIVATE_SEQUENCE, AGGREGATE_FLAG_PUBLISHED_SEQUENCE,
    AGGREGATE_FLAG_RESET, AGGREGATE_FLAG_RETAINED_RESPONSE, AGGREGATE_FLAG_RETURNING,
    AGGREGATE_SECTION_COUNT, OUTER_CONTENT_CATALOG, OUTER_CONTENT_PRIVATE_SEQUENCE,
    OUTER_CONTENT_PUBLISHED_SEQUENCE, OUTER_CONTENT_RESET, OUTER_CONTENT_RETURNING,
    OUTER_CONTENT_REWRITE, OUTER_CONTENT_ROW, OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
};
use crate::typed_insert_batch::{
    encode_canonical_typed_insert_record_for_test, prepare_typed_insert_semantics_at,
    sequence_defaults, DecodedSequenceEffectKindFacts,
};
use crate::{encode_sequence_value_reference_into_exact, BinarySequenceValueReference};

#[path = "tests/sabotage.rs"]
mod sabotage;

#[derive(Clone)]
struct DirectorySpec {
    statement_ordinal: u32,
    family_ordinal: u32,
    family: u8,
    input_rows: u32,
    request_digest: gpu_db_wal::CanonicalDigest,
    statement_digest: gpu_db_wal::CanonicalDigest,
    overlay_before: gpu_db_wal::CanonicalDigest,
    overlay_after: gpu_db_wal::CanonicalDigest,
}

#[derive(Clone)]
struct RowSpec {
    statement_ordinal: u32,
    source_row: u32,
    stable_row_id: u64,
    kind: u8,
    block_ref: u32,
    transition_ref: u32,
    statement_digest: gpu_db_wal::CanonicalDigest,
}

fn typed_record(statement_ordinal: u32, values: &[i32]) -> Vec<u8> {
    typed_record_rows(
        statement_ordinal,
        values
            .iter()
            .map(|value| (*value, format!("row-{value}")))
            .collect(),
    )
}

fn typed_record_with_text(statement_ordinal: u32, text: String) -> Vec<u8> {
    typed_record_rows(statement_ordinal, vec![(7, text)])
}

fn typed_record_rows(statement_ordinal: u32, rows: Vec<(i32, String)>) -> Vec<u8> {
    typed_record_rows_with_returning(statement_ordinal, rows, true)
}

fn typed_record_without_returning(statement_ordinal: u32, value: i32) -> Vec<u8> {
    typed_record_rows_with_returning(
        statement_ordinal,
        vec![(value, format!("without-returning-{value}"))],
        false,
    )
}

fn typed_record_rows_with_returning(
    statement_ordinal: u32,
    rows: Vec<(i32, String)>,
    with_returning: bool,
) -> Vec<u8> {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE source_materialize_typed (id int4, note text)",
        )
        .unwrap();
    let insert = crate::Insert {
        table: "source_materialize_typed".to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(
            rows.into_iter()
                .map(|(id, note)| vec![crate::SqlValue::Int4(id), crate::SqlValue::Text(note)])
                .collect(),
        ),
        returning: if with_returning {
            vec!["note".to_string(), "id".to_string()]
        } else {
            Vec::new()
        },
    };
    let catalog = engine.catalog_snapshot();
    let prepared = prepare_typed_insert_semantics_at(
        &insert,
        &catalog,
        catalog.commit_seq,
        None,
        InsertStatementOrdinal::from_u32(statement_ordinal),
    )
    .unwrap()
    .unwrap();
    let batch = prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .unwrap();
    encode_canonical_typed_insert_record_for_test(&batch).unwrap()
}

fn serial_typed_record(
    statement_ordinal: u32,
    parent_txn_id: u64,
    parent_autocommit: bool,
) -> Vec<u8> {
    serial_typed_record_with_effect(statement_ordinal, parent_txn_id, parent_autocommit, false)
}

fn private_serial_typed_record(statement_ordinal: u32, parent_txn_id: u64) -> Vec<u8> {
    crate::typed_insert_batch::private_sequence_chain_record_for_test(
        InsertStatementOrdinal::from_u32(statement_ordinal),
        parent_txn_id,
    )
    .unwrap()
}

fn serial_typed_record_with_effect(
    statement_ordinal: u32,
    parent_txn_id: u64,
    parent_autocommit: bool,
    private: bool,
) -> Vec<u8> {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE source_materialize_serial (id serial, payload int4)",
        )
        .unwrap();
    let crate::Command::Insert(insert) = crate::parse_command(
        "INSERT INTO source_materialize_serial (id, payload) VALUES (DEFAULT, 7)",
    )
    .unwrap() else {
        panic!("serial fixture must parse as INSERT");
    };
    let catalog = engine.catalog_snapshot();
    let ordinal = InsertStatementOrdinal::from_u32(statement_ordinal);
    let prepared =
        prepare_typed_insert_semantics_at(&insert, &catalog, catalog.commit_seq, None, ordinal)
            .unwrap()
            .unwrap();
    let typed_digest = prepared.typed_statement_digest();
    let parent = sequence_defaults::effects::SequenceDefaultParentContext::for_test(
        parent_txn_id,
        parent_autocommit,
        typed_digest,
        ordinal,
        0,
    );
    let bindings = prepared
        .sequence_requests()
        .iter()
        .cloned()
        .map(|request| {
            if private {
                sequence_defaults::SequenceDefaultBinding::private(request, parent.clone(), 41)
            } else {
                sequence_defaults::SequenceDefaultBinding::published(request, parent.clone(), 41)
            }
        })
        .collect();
    let batch = prepared
        .seal(
            sequence_defaults::SequenceDefaultBindings::from_bindings(parent, bindings),
            false,
            false,
        )
        .unwrap();
    encode_canonical_typed_insert_record_for_test(&batch).unwrap()
}

fn current_set_body(value: &str) -> Vec<u8> {
    let payload = serde_json::to_vec(&crate::Command::SetKv {
        key: "source_materialize_key".to_string(),
        value: value.to_string(),
    })
    .unwrap();
    let mut body = Vec::with_capacity(20 + payload.len());
    body.extend_from_slice(b"GPUDBOP1");
    body.push(4); // current typed-command-v2 operation codec
    body.extend_from_slice(&[0; 3]);
    body.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    body.extend_from_slice(&payload);
    body
}

fn current_command_body(sql: &str) -> Vec<u8> {
    let command = crate::parse_command(sql).unwrap();
    let payload = serde_json::to_vec(&command).unwrap();
    let mut body = Vec::with_capacity(20 + payload.len());
    body.extend_from_slice(b"GPUDBOP1");
    body.push(4);
    body.extend_from_slice(&[0; 3]);
    body.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    body.extend_from_slice(&payload);
    body
}

fn strict_s3_facts(
    body: &[u8],
) -> crate::engine_canonical_operation::CurrentNonInsertCanonicalOperationFacts {
    strict_s3_facts_with_kind(body, gpu_db_wal::CanonicalFragmentKind::RowMutation)
}

fn strict_s3_facts_with_kind(
    body: &[u8],
    kind: gpu_db_wal::CanonicalFragmentKind,
) -> crate::engine_canonical_operation::CurrentNonInsertCanonicalOperationFacts {
    let digest = gpu_db_wal::canonical_request_digest(body);
    let operation = decode_current_non_insert_canonical_operation(
        kind,
        4,
        u32::try_from(body.len()).unwrap(),
        digest,
        body,
    )
    .unwrap();
    operation.facts()
}

fn append_directory(bytes: &mut Vec<u8>, entries: &[DirectorySpec]) {
    for entry in entries {
        bytes.extend_from_slice(&entry.statement_ordinal.to_le_bytes());
        bytes.extend_from_slice(&entry.family_ordinal.to_le_bytes());
        bytes.push(entry.family);
        bytes.extend_from_slice(&[0; 3]);
        bytes.extend_from_slice(&entry.input_rows.to_le_bytes());
        bytes.extend_from_slice(&entry.request_digest);
        bytes.extend_from_slice(&entry.statement_digest);
        bytes.extend_from_slice(&entry.overlay_before);
        bytes.extend_from_slice(&entry.overlay_after);
    }
}

fn append_typed_sources(bytes: &mut Vec<u8>, records: &[Vec<u8>]) {
    for record in records {
        bytes.extend_from_slice(&u32::try_from(record.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(record);
    }
}

fn append_non_insert_source(
    bytes: &mut Vec<u8>,
    body: &[u8],
    statement_ordinal: u32,
    family_ordinal: u32,
    kind: gpu_db_wal::CanonicalFragmentKind,
) {
    let facts = strict_s3_facts_with_kind(body, kind);
    bytes.extend_from_slice(&statement_ordinal.to_le_bytes());
    bytes.extend_from_slice(&family_ordinal.to_le_bytes());
    bytes.extend_from_slice(&(facts.fragment_kind() as u16).to_le_bytes());
    bytes.push(facts.operation_codec());
    bytes.push(0);
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&facts.body_digest());
    bytes.extend_from_slice(body);
}

fn append_dispositions(bytes: &mut Vec<u8>, rows: &[RowSpec]) {
    for row in rows {
        bytes.extend_from_slice(&row.statement_ordinal.to_le_bytes());
        bytes.extend_from_slice(&row.source_row.to_le_bytes());
        bytes.extend_from_slice(&row.stable_row_id.to_le_bytes());
        bytes.push(row.kind);
        bytes.extend_from_slice(&[0; 3]);
        bytes.extend_from_slice(&row.block_ref.to_le_bytes());
        bytes.extend_from_slice(&row.transition_ref.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&row.statement_digest);
    }
}

fn append_s5_prefix(
    bytes: &mut Vec<u8>,
    statement_ordinal: u32,
    effect_ordinal: u32,
    kind: u8,
    flags: u8,
    disposition_ordinal: u32,
    body: &[u8],
) {
    bytes.extend_from_slice(&statement_ordinal.to_le_bytes());
    bytes.extend_from_slice(&effect_ordinal.to_le_bytes());
    bytes.push(kind);
    bytes.push(flags);
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes.extend_from_slice(&disposition_ordinal.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
    let body_digest = if kind == 2 {
        [0; 32]
    } else {
        gpu_db_wal::canonical_request_digest(body)
    };
    bytes.extend_from_slice(&body_digest);
    bytes.extend_from_slice(body);
}

fn append_typed_sequence_closure(
    bytes: &mut Vec<u8>,
    statement_ordinal: u32,
    record: &DecodedTypedInsertRecord,
    rows: &[RowSpec],
    row_offset: usize,
) {
    let parent = record.sequence_parent();
    for effect in record.sequence_effects() {
        let disposition_ordinal = row_offset + effect.request.row_ordinal as usize;
        let row = &rows[disposition_ordinal];
        match effect.kind {
            DecodedSequenceEffectKindFacts::Published {
                transition_txn_id,
                input_digest,
                returned_value,
            } => {
                let parent = parent.expect("published test effect has parent");
                let final_value_overwritten = row.kind != DISPOSITION_SURVIVES;
                let reference = BinarySequenceValueReference {
                    transition_txn_id,
                    parent_txn_id: parent.txn_id,
                    statement_ordinal,
                    expression_ordinal: effect.request.absolute_expression_ordinal,
                    sequence_oid: effect.request.sequence_oid,
                    returned_value,
                    input_digest,
                    table_oid: effect.request.target_table_oid,
                    column_id: effect.request.column_id,
                    staging_row_ordinal: 0,
                    row_id: row.stable_row_id,
                    final_value_overwritten,
                    default_expression: true,
                };
                let mut body = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
                encode_sequence_value_reference_into_exact(&reference, &mut body).unwrap();
                append_s5_prefix(
                    bytes,
                    statement_ordinal,
                    effect.request.effect_ordinal,
                    1,
                    1 | if final_value_overwritten { 2 } else { 0 },
                    u32::try_from(disposition_ordinal).unwrap(),
                    &body,
                );
            }
            DecodedSequenceEffectKindFacts::Private { .. } => {
                let final_value_overwritten = row.kind != DISPOSITION_SURVIVES;
                append_s5_prefix(
                    bytes,
                    statement_ordinal,
                    effect.request.effect_ordinal,
                    2,
                    1 | if final_value_overwritten { 2 } else { 0 },
                    u32::try_from(disposition_ordinal).unwrap(),
                    &[],
                );
            }
        }
    }
}

fn append_explicit_s3_sequence_closure(
    bytes: &mut Vec<u8>,
    body: &[u8],
    fragment_kind: gpu_db_wal::CanonicalFragmentKind,
    request_digest: gpu_db_wal::CanonicalDigest,
    parent_txn_id: u64,
) {
    let operation = decode_current_non_insert_canonical_operation(
        fragment_kind,
        4,
        u32::try_from(body.len()).unwrap(),
        gpu_db_wal::canonical_request_digest(body),
        body,
    )
    .unwrap();
    let Some(explicit) = operation.explicit_sequence_facts() else {
        return;
    };
    let reference = BinarySequenceValueReference {
        transition_txn_id: 900,
        parent_txn_id,
        statement_ordinal: 1,
        expression_ordinal: 0,
        sequence_oid: 77,
        returned_value: explicit.requested_set_value().unwrap_or(42),
        input_digest: crate::sequence_value_input_digest(crate::SequenceValueInput {
            parent_txn_id,
            parent_autocommit: false,
            statement_ordinal: 1,
            expression_ordinal: 0,
            parent_request_digest: request_digest,
            source_name: explicit.source_name(),
            operation: explicit.operation(),
            set_value: explicit.requested_set_value(),
        }),
        table_oid: 0,
        column_id: 0,
        staging_row_ordinal: 0,
        row_id: 0,
        final_value_overwritten: false,
        default_expression: false,
    };
    let mut encoded = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
    encode_sequence_value_reference_into_exact(&reference, &mut encoded).unwrap();
    append_s5_prefix(bytes, 1, 0, 1, 0, u32::MAX, &encoded);
}

fn s6_class_for_s3(class: crate::engine_canonical_operation::CurrentNonInsertSemanticClass) -> u16 {
    match class {
        crate::engine_canonical_operation::CurrentNonInsertSemanticClass::Update => 3,
        crate::engine_canonical_operation::CurrentNonInsertSemanticClass::Delete => 4,
        crate::engine_canonical_operation::CurrentNonInsertSemanticClass::Catalog => 5,
        crate::engine_canonical_operation::CurrentNonInsertSemanticClass::TableReset => 6,
        crate::engine_canonical_operation::CurrentNonInsertSemanticClass::TableRewrite => 7,
        crate::engine_canonical_operation::CurrentNonInsertSemanticClass::Sequence => 8,
        crate::engine_canonical_operation::CurrentNonInsertSemanticClass::KeyValueMutation => 9,
    }
}

fn append_s6_entry(
    bytes: &mut Vec<u8>,
    statement_ordinal: u32,
    family_ordinal: u32,
    semantic_class: u16,
    flags: u16,
    statement_digest: gpu_db_wal::CanonicalDigest,
    outcome: gpu_db_wal::CanonicalOutcome,
) {
    bytes.extend_from_slice(&statement_ordinal.to_le_bytes());
    bytes.extend_from_slice(&family_ordinal.to_le_bytes());
    bytes.extend_from_slice(&semantic_class.to_le_bytes());
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes.extend_from_slice(&statement_digest);
    let mut outcome_bytes = [0_u8; gpu_db_wal::CANONICAL_OUTCOME_BYTES];
    gpu_db_wal::encode_canonical_outcome_into_exact(&outcome, &mut outcome_bytes).unwrap();
    bytes.extend_from_slice(&outcome_bytes);
}

fn replace_s6_outcome(payload: &mut [u8], statement: usize, outcome: gpu_db_wal::CanonicalOutcome) {
    let base = statement * 136;
    let mut bytes = [0_u8; gpu_db_wal::CANONICAL_OUTCOME_BYTES];
    gpu_db_wal::encode_canonical_outcome_into_exact(&outcome, &mut bytes).unwrap();
    payload[base + 44..base + 136].copy_from_slice(&bytes);
}

fn append_typed_outcome(
    bytes: &mut Vec<u8>,
    statement_ordinal: u32,
    family_ordinal: u32,
    facts: crate::typed_insert_batch::DecodedTypedInsertRecordFacts,
    target_digest: gpu_db_wal::CanonicalDigest,
    rows: &[RowSpec],
    row_offset: usize,
) {
    let affected_rows = rows[row_offset..row_offset + facts.row_count as usize]
        .iter()
        .filter(|row| {
            matches!(
                row.kind,
                DISPOSITION_SURVIVES | DISPOSITION_APPLIED_THEN_CANCELED
            )
        })
        .count() as u64;
    let has_returning = facts.returning.column_count != 0;
    append_s6_entry(
        bytes,
        statement_ordinal,
        family_ordinal,
        1,
        u16::from(has_returning),
        facts.typed_statement_digest,
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows,
            sqlstate: None,
            constraint_id: 0,
            target_digest,
            returning_digest: if has_returning {
                // S2 owns the typed projection layout. S6 owns a logical result identity
                // which S8 will later recompute, so the two domains stay distinct here.
                let mut logical_result = Vec::from(&b"source-materialization-s6-returning"[..]);
                logical_result.extend_from_slice(&facts.returning.digest);
                gpu_db_wal::canonical_request_digest(&logical_result)
            } else {
                [0; 32]
            },
        },
    );
}

fn append_s3_outcome(
    bytes: &mut Vec<u8>,
    statement_ordinal: u32,
    family_ordinal: u32,
    target_digest: gpu_db_wal::CanonicalDigest,
    facts: crate::engine_canonical_operation::CurrentNonInsertCanonicalOperationFacts,
) {
    let has_returning = facts.has_returning();
    append_s6_entry(
        bytes,
        statement_ordinal,
        family_ordinal,
        s6_class_for_s3(facts.semantic_class()),
        u16::from(has_returning),
        facts.statement_digest(),
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
            target_digest,
            // S6 records only presence for a strict S3 typed projection. S7 owns exact
            // projection resolution, so this is intentionally an opaque nonzero identity.
            returning_digest: if has_returning {
                s3_returning_digest(facts.statement_digest())
            } else {
                [0; 32]
            },
        },
    );
}

fn s3_returning_digest(
    statement_digest: gpu_db_wal::CanonicalDigest,
) -> gpu_db_wal::CanonicalDigest {
    let mut bytes = Vec::from(&b"source-materialization-s3-returning"[..]);
    bytes.extend_from_slice(&statement_digest);
    gpu_db_wal::canonical_request_digest(&bytes)
}

fn source_payloads(
    first_typed: Vec<u8>,
    second_typed: Vec<u8>,
    non_insert_body: Vec<u8>,
) -> [Vec<u8>; AGGREGATE_SECTION_COUNT] {
    source_payloads_with_s3(
        first_typed,
        second_typed,
        non_insert_body,
        gpu_db_wal::CanonicalFragmentKind::RowMutation,
        41,
    )
}

fn source_payloads_with_s3(
    first_typed: Vec<u8>,
    second_typed: Vec<u8>,
    non_insert_body: Vec<u8>,
    non_insert_kind: gpu_db_wal::CanonicalFragmentKind,
    explicit_sequence_parent_txn: u64,
) -> [Vec<u8>; AGGREGATE_SECTION_COUNT] {
    let first_record =
        crate::typed_insert_batch::decode_canonical_typed_insert_record(&first_typed).unwrap();
    let second_record =
        crate::typed_insert_batch::decode_canonical_typed_insert_record(&second_typed).unwrap();
    let first = first_record.facts();
    let second = second_record.facts();
    assert_eq!(first.statement_ordinal.as_u32(), 0);
    assert_eq!(second.statement_ordinal.as_u32(), 2);
    let non_insert = strict_s3_facts_with_kind(&non_insert_body, non_insert_kind);
    let directory = [
        DirectorySpec {
            statement_ordinal: 0,
            family_ordinal: 0,
            family: FAMILY_TYPED_INSERT,
            input_rows: first.row_count,
            request_digest: first.typed_statement_digest,
            statement_digest: first.typed_statement_digest,
            overlay_before: [1; 32],
            overlay_after: [2; 32],
        },
        DirectorySpec {
            statement_ordinal: 1,
            family_ordinal: 0,
            family: FAMILY_CANONICAL_OPERATION,
            input_rows: 0,
            request_digest: non_insert.request_digest(),
            statement_digest: non_insert.statement_digest(),
            overlay_before: [2; 32],
            overlay_after: [3; 32],
        },
        DirectorySpec {
            statement_ordinal: 2,
            family_ordinal: 1,
            family: FAMILY_TYPED_INSERT,
            input_rows: second.row_count,
            request_digest: second.typed_statement_digest,
            statement_digest: second.typed_statement_digest,
            overlay_before: [3; 32],
            overlay_after: [4; 32],
        },
    ];
    let mut rows = Vec::new();
    let mut stable_row_id = 100_u64;
    for (statement_ordinal, facts) in [(0, first), (2, second)] {
        for source_row in 0..facts.row_count {
            let global = rows.len();
            let kind = match global {
                0 => DISPOSITION_SURVIVES,
                1 => DISPOSITION_APPLIED_THEN_CANCELED,
                _ => DISPOSITION_SUPPRESSED_AT_STATEMENT,
            };
            rows.push(RowSpec {
                statement_ordinal,
                source_row,
                stable_row_id,
                kind,
                block_ref: if kind == DISPOSITION_SURVIVES {
                    global as u32
                } else {
                    u32::MAX
                },
                transition_ref: if kind == DISPOSITION_SURVIVES {
                    global as u32
                } else {
                    u32::MAX
                },
                statement_digest: facts.typed_statement_digest,
            });
            stable_row_id += 1;
        }
    }
    let mut sections: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|section| vec![section as u8 + 1]);
    sections[0].clear();
    append_directory(&mut sections[0], &directory);
    sections[1].clear();
    append_typed_sources(&mut sections[1], &[first_typed, second_typed]);
    sections[2].clear();
    append_non_insert_source(&mut sections[2], &non_insert_body, 1, 0, non_insert_kind);
    sections[3].clear();
    append_dispositions(&mut sections[3], &rows);
    sections[4].clear();
    append_typed_sequence_closure(&mut sections[4], 0, &first_record, &rows, 0);
    append_typed_sequence_closure(
        &mut sections[4],
        2,
        &second_record,
        &rows,
        first.row_count as usize,
    );
    append_explicit_s3_sequence_closure(
        &mut sections[4],
        &non_insert_body,
        non_insert_kind,
        non_insert.statement_digest(),
        explicit_sequence_parent_txn,
    );
    sections[5].clear();
    append_typed_outcome(&mut sections[5], 0, 0, first, [2; 32], &rows, 0);
    append_s3_outcome(&mut sections[5], 1, 0, [3; 32], non_insert);
    append_typed_outcome(
        &mut sections[5],
        2,
        1,
        second,
        [4; 32],
        &rows,
        first.row_count as usize,
    );
    sections[7].clear();
    sections
}

fn entry_counts(payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT]) -> [u32; AGGREGATE_SECTION_COUNT] {
    [
        3,
        2,
        1,
        u32::try_from(payloads[3].len() / ROW_DISPOSITION_ENTRY_BYTES as usize).unwrap(),
        s5_entry_count(&payloads[4]),
        3,
        1,
        u32::from(!payloads[7].is_empty()),
    ]
}

fn s5_entry_count(payload: &[u8]) -> u32 {
    let mut cursor = 0_usize;
    let mut count = 0_u32;
    while payload.len().saturating_sub(cursor) >= 52 {
        let body_len =
            u32::from_le_bytes(payload[cursor + 16..cursor + 20].try_into().unwrap()) as usize;
        cursor = cursor.saturating_add(52 + body_len);
        count += 1;
    }
    count
}

fn s5_has_published(payload: &[u8]) -> bool {
    let mut cursor = 0_usize;
    while payload.len().saturating_sub(cursor) >= 52 {
        if payload[cursor + 8] == 1 {
            return true;
        }
        let body_len =
            u32::from_le_bytes(payload[cursor + 16..cursor + 20].try_into().unwrap()) as usize;
        cursor += 52 + body_len;
    }
    false
}

fn s5_has_private(payload: &[u8]) -> bool {
    let mut cursor = 0_usize;
    while payload.len().saturating_sub(cursor) >= 52 {
        if payload[cursor + 8] == 2 {
            return true;
        }
        let body_len =
            u32::from_le_bytes(payload[cursor + 16..cursor + 20].try_into().unwrap()) as usize;
        cursor += 52 + body_len;
    }
    false
}

#[derive(Default)]
struct S6FixtureFacts {
    has_returning: bool,
    retained_response_count: u32,
    has_catalog: bool,
    has_reset: bool,
    has_rewrite: bool,
}

fn s6_fixture_facts(payload: &[u8]) -> S6FixtureFacts {
    let mut facts = S6FixtureFacts::default();
    for entry in payload.chunks_exact(136) {
        facts.has_returning |= u16::from_le_bytes(entry[10..12].try_into().unwrap()) & 1 != 0;
        facts.retained_response_count +=
            u32::from(u16::from_le_bytes(entry[10..12].try_into().unwrap()) & 2 != 0);
        match u16::from_le_bytes(entry[8..10].try_into().unwrap()) {
            5 => facts.has_catalog = true,
            6 => facts.has_reset = true,
            7 => facts.has_rewrite = true,
            _ => {}
        }
    }
    facts
}

fn source_view<'a>(
    payloads: &'a [Vec<u8>; AGGREGATE_SECTION_COUNT],
    transaction_id: u64,
) -> TypedInsertAggregateView<'a> {
    let original_rows =
        u64::try_from(payloads[3].len() / ROW_DISPOSITION_ENTRY_BYTES as usize).unwrap();
    let final_row_transition_count = payloads[3]
        .chunks_exact(ROW_DISPOSITION_ENTRY_BYTES as usize)
        .filter(|entry| entry[16] == DISPOSITION_SURVIVES)
        .count()
        .try_into()
        .unwrap();
    let has_published_sequence = s5_has_published(&payloads[4]);
    let has_private_sequence = s5_has_private(&payloads[4]);
    let s6 = s6_fixture_facts(&payloads[5]);
    TypedInsertAggregateView {
        flags: AGGREGATE_FLAG_EXPLICIT
            | if s6.has_returning {
                AGGREGATE_FLAG_RETURNING
            } else {
                0
            }
            | if s6.retained_response_count != 0 {
                AGGREGATE_FLAG_RETAINED_RESPONSE
            } else {
                0
            }
            | if has_published_sequence {
                AGGREGATE_FLAG_PUBLISHED_SEQUENCE
            } else {
                0
            }
            | if has_private_sequence {
                AGGREGATE_FLAG_PRIVATE_SEQUENCE
            } else {
                0
            }
            | if s6.has_catalog {
                AGGREGATE_FLAG_CATALOG
            } else {
                0
            }
            | if s6.has_reset {
                AGGREGATE_FLAG_RESET
            } else {
                0
            },
        outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1
            | OUTER_CONTENT_ROW
            | if s6.has_returning {
                OUTER_CONTENT_RETURNING
            } else {
                0
            }
            | if has_published_sequence {
                OUTER_CONTENT_PUBLISHED_SEQUENCE
            } else {
                0
            }
            | if has_private_sequence {
                OUTER_CONTENT_PRIVATE_SEQUENCE
            } else {
                0
            }
            | if s6.has_catalog {
                OUTER_CONTENT_CATALOG
            } else {
                0
            }
            | if s6.has_reset { OUTER_CONTENT_RESET } else { 0 }
            | if s6.has_rewrite {
                OUTER_CONTENT_REWRITE
            } else {
                0
            },
        stable_transaction_id: transaction_id,
        statement_count: 3,
        insert_statement_count: 2,
        original_inserted_row_count: original_rows,
        final_row_transition_count,
        allocator_before: 100,
        allocator_high_water: 100 + original_rows,
        table_block_count: 1,
        sections: std::array::from_fn(|index| TypedInsertAggregateSectionView {
            entry_count: entry_counts(payloads)[index],
            payload: &payloads[index],
        }),
    }
}

fn status(
    roots: TypedInsertAggregateStatusRoots,
    transaction_id: u64,
    response_artifact_count: u32,
) -> TypedInsertStatusV2 {
    TypedInsertStatusV2 {
        database_id: [1; 16],
        timeline_id: [2; 16],
        txn_id: transaction_id,
        request_digest: [3; 32],
        isolation: 1,
        flags: 0,
        retention_deadline: 0,
        statement_count: 3,
        response_artifact_count,
        statement_outcome_root: roots.statement_outcome_root,
        response_root: roots.response_root,
        aggregate_root: roots.aggregate_root,
    }
}

fn materialize(
    payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
    transaction_id: u64,
) -> Result<TypedInsertAggregateSourceDraft, EngineError> {
    materialize_with_view_mutation(payloads, transaction_id, |_| {})
}

fn materialize_with_view_mutation(
    payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
    transaction_id: u64,
    mutate: impl FnOnce(&mut TypedInsertAggregateView<'_>),
) -> Result<TypedInsertAggregateSourceDraft, EngineError> {
    let mut view = source_view(payloads, transaction_id);
    mutate(&mut view);
    let layout = view
        .measure()
        .map_err(|error| violation(&format!("test aggregate measurement failed: {error}")))?;
    let roots = typed_insert_aggregate_status_roots(&view, &layout)?;
    let reserved = reserve_typed_insert_aggregate_bodies(layout)?;
    let encoded = encode_typed_insert_aggregate_bodies(
        &view,
        &status(roots, transaction_id, view.sections[7].entry_count),
        reserved,
    )?;
    let bodies: Vec<Vec<u8>> = encoded.fragment_bodies().map(<[u8]>::to_vec).collect();
    let references: Vec<&[u8]> = bodies.iter().map(Vec::as_slice).collect();
    let decoded = decode_typed_insert_aggregate_bodies(view.outer_flags, &references)?;
    assert_eq!(
        decoded.sections()[ROW_DISPOSITIONS_SECTION].entry_count,
        entry_counts(payloads)[ROW_DISPOSITIONS_SECTION],
        "fixture preserves S4 outer count"
    );
    assert_eq!(
        decoded.sections()[ROW_DISPOSITIONS_SECTION].payload_bytes,
        payloads[ROW_DISPOSITIONS_SECTION].len() as u64,
        "fixture preserves S4 payload bytes"
    );
    let s4_remaining = decoded.with_section_reader(ROW_DISPOSITIONS_SECTION, |reader| {
        let remaining = reader.remaining();
        reader.skip(remaining)?;
        Ok(remaining)
    })?;
    assert_eq!(
        s4_remaining,
        payloads[ROW_DISPOSITIONS_SECTION].len() as u64,
        "fixture S4 reader is bounded to the direct S4 payload"
    );
    materialize_source_draft(&decoded)
}

fn mixed_payloads() -> [Vec<u8>; AGGREGATE_SECTION_COUNT] {
    source_payloads(
        typed_record(0, &[7, 8]),
        typed_record(2, &[9]),
        current_set_body("mixed"),
    )
}

fn padding_to_straddle_section(
    make_payloads: impl Fn(Vec<u8>) -> [Vec<u8>; AGGREGATE_SECTION_COUNT],
    section: usize,
    entry_bytes: u64,
) -> [Vec<u8>; AGGREGATE_SECTION_COUNT] {
    let empty = current_set_body("");
    let baseline = make_payloads(empty.clone());
    let baseline_offset =
        source_view(&baseline, 41).measure().unwrap().sections[section].payload_offset;
    let chunk = AGGREGATE_CHUNK_PAYLOAD_BYTES;
    let target = chunk - entry_bytes + 1;
    let delta = (target + chunk - baseline_offset % chunk) % chunk;
    let body_len = empty.len() + usize::try_from(delta).unwrap();
    assert!(body_len <= MAX_SOURCE_BODY_BYTES);
    let payloads = make_payloads(current_set_body(
        &"x".repeat(usize::try_from(delta).unwrap()),
    ));
    let layout = source_view(&payloads, 41).measure().unwrap();
    assert!(
        layout.sections[section].payload_offset % chunk + entry_bytes > chunk,
        "selected S{section} entry must straddle one aggregate chunk"
    );
    payloads
}

#[test]
fn materializes_fixed_s5_and_s6_entries_across_codec5_chunk_boundaries() {
    let s5 = padding_to_straddle_section(
        |s3| source_payloads(typed_record(0, &[7]), serial_typed_record(2, 41, false), s3),
        4,
        52 + crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES as u64,
    );
    assert_eq!(
        s5[4].len(),
        52 + crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES
    );
    assert_eq!(materialize(&s5, 41).unwrap().sequence_effects.len(), 1);

    let s6 = padding_to_straddle_section(
        |s3| source_payloads(typed_record(0, &[7]), typed_record(2, &[8]), s3),
        5,
        136,
    );
    assert_eq!(s6[5].len(), 3 * 136);
    assert!(materialize(&s6, 41).is_ok());
}

fn assert_rejected(payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT], needle: &str) {
    let error = match materialize(payloads, 41) {
        Ok(_) => panic!("source sabotage must reject"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains(needle),
        "expected {needle:?}, got {error}"
    );
}

fn s2_entry_ranges(bytes: &[u8]) -> Vec<Range<usize>> {
    let mut cursor = 0;
    let mut ranges = Vec::new();
    while cursor < bytes.len() {
        let start = cursor;
        let len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4 + len;
        ranges.push(start..cursor);
    }
    ranges
}

#[test]
fn materializes_interleaved_s2_s3_sources_with_compact_global_s4_ranges() {
    let draft = materialize(&mixed_payloads(), 41).unwrap();
    assert_eq!(draft.statements.len(), 3);
    assert_eq!(draft.dispositions.len(), 3);
    assert_eq!(draft.statements[0].input_rows, 0..2);
    assert_eq!(draft.statements[1].input_rows, 2..2);
    assert_eq!(draft.statements[2].input_rows, 2..3);
    assert_eq!(draft.statements[0].statement_ordinal, 0);
    assert_eq!(draft.statements[1].family_ordinal, 0);
    assert_eq!(draft.statements[2].statement_ordinal, 2);
    assert_eq!(draft.statements[0].overlay_before, [1; 32]);
    assert_eq!(draft.statements[1].overlay_after, [3; 32]);
    assert_eq!(
        draft.statements[0].request_digest,
        draft.statements[0].statement_digest
    );
    assert!(matches!(
        draft.statements[0].source,
        SourceStatementKind::TypedInsert(_)
    ));
    assert!(matches!(
        draft.statements[1].source,
        SourceStatementKind::CanonicalOperation(_)
    ));
    assert!(matches!(
        draft.statements[2].source,
        SourceStatementKind::TypedInsert(_)
    ));
    assert!(matches!(
        draft.dispositions[0].kind,
        RowDispositionKind::Survives
    ));
    assert!(matches!(
        draft.dispositions[1].kind,
        RowDispositionKind::AppliedThenCanceled
    ));
    assert!(matches!(
        draft.dispositions[2].kind,
        RowDispositionKind::SuppressedAtStatement
    ));
    assert_eq!(draft.dispositions[0].stable_row_id, 100);
    assert_eq!(draft.dispositions[0].block_ref, 0);
    assert_eq!(draft.dispositions[0].transition_ref, 0);
    assert_eq!(draft.dispositions[1].block_ref, u32::MAX);
    let SourceStatementKind::CanonicalOperation(operation) = &draft.statements[1].source else {
        panic!("second source remains the strict non-INSERT operation");
    };
    assert_eq!(
        operation.facts().statement_digest(),
        draft.statements[1].statement_digest
    );
}

#[test]
fn materializes_published_and_private_s2_s5_effects_without_a_second_source_owner() {
    let payloads = source_payloads(
        serial_typed_record(0, 41, false),
        private_serial_typed_record(2, 41),
        current_set_body("sequence-closure"),
    );
    let draft = materialize(&payloads, 41).unwrap();
    assert_eq!(draft.sequence_effects.len(), 3);
    assert_eq!(draft.statements[0].sequence_effects, 0..1);
    assert_eq!(draft.statements[1].sequence_effects, 1..1);
    assert_eq!(draft.statements[2].sequence_effects, 1..3);
    match &draft.sequence_effects[0].kind {
        CompactSequenceEffectKind::Published(reference) => {
            assert!(reference.default_expression);
            assert!(!reference.final_value_overwritten);
            assert_eq!(reference.row_id, 100);
        }
        CompactSequenceEffectKind::Private { .. } => {
            panic!("first S2 effect must retain its published S5 reference")
        }
    }
    match &draft.sequence_effects[1].kind {
        CompactSequenceEffectKind::Private {
            input_digest,
            final_value_overwritten,
        } => {
            assert_ne!(*input_digest, [0; 32]);
            assert!(*final_value_overwritten);
            assert_eq!(draft.sequence_effects[1].disposition_ordinal, 1);
        }
        CompactSequenceEffectKind::Published(_) => {
            panic!("second S2 effect must retain private binding evidence")
        }
    }
    assert!(draft
        .statements
        .iter()
        .all(|statement| statement.outcome.is_some()));
}

#[test]
fn materializes_strict_s3_nextval_and_setval_with_statement_digest_parent_identity() {
    for sql in [
        "SELECT nextval('source_materialize_s5_nextval')",
        "SELECT setval('source_materialize_s5_setval', 73, false)",
    ] {
        let payloads = source_payloads_with_s3(
            typed_record(0, &[7]),
            typed_record(2, &[8]),
            current_command_body(sql),
            gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
            41,
        );
        let draft = materialize(&payloads, 41).unwrap();
        assert_eq!(draft.statements[1].sequence_effects, 0..1);
        let CompactSequenceEffectKind::Published(reference) = &draft.sequence_effects[0].kind
        else {
            panic!("strict S3 sequence must own one published S5 reference");
        };
        assert!(!reference.default_expression);
        assert_eq!(reference.parent_txn_id, 41);
        assert_eq!(reference.statement_ordinal, 1);
        assert_eq!(reference.expression_ordinal, 0);
        if sql.contains("setval") {
            assert_eq!(reference.returned_value, 73);
        }
    }
}

#[test]
fn s6_binds_every_reachable_strict_s3_semantic_class() {
    let cases = vec![
        (
            current_command_body("UPDATE source_s6 SET id = 8 WHERE id = 7"),
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            StatementSemanticClass::Update,
        ),
        (
            current_command_body("DELETE FROM source_s6 WHERE id = 7"),
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            StatementSemanticClass::Delete,
        ),
        (
            current_command_body("CREATE SEQUENCE source_s6_catalog"),
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            StatementSemanticClass::Catalog,
        ),
        (
            current_command_body("TRUNCATE source_s6"),
            gpu_db_wal::CanonicalFragmentKind::TableReset,
            StatementSemanticClass::TableReset,
        ),
        (
            current_command_body("REFRESH MATERIALIZED VIEW source_s6_view"),
            gpu_db_wal::CanonicalFragmentKind::TableRewrite,
            StatementSemanticClass::TableRewrite,
        ),
        (
            current_command_body("SELECT nextval('source_s6_sequence')"),
            gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
            StatementSemanticClass::Sequence,
        ),
        (
            current_set_body("source-s6-kv"),
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            StatementSemanticClass::KeyValueMutation,
        ),
    ];
    for (body, kind, expected) in cases {
        let payloads =
            source_payloads_with_s3(typed_record(0, &[7]), typed_record(2, &[8]), body, kind, 41);
        let draft = materialize(&payloads, 41).unwrap();
        assert_eq!(
            draft.statements[1].outcome.as_ref().unwrap().semantic_class,
            expected
        );
    }
}

#[test]
fn s6_binds_strict_s3_update_delete_returning_presence_without_projection_resolution() {
    for (sql, expected_class) in [
        (
            "UPDATE source_s6 SET id = 8 WHERE id = 7 RETURNING id",
            StatementSemanticClass::Update,
        ),
        (
            "DELETE FROM source_s6 WHERE id = 7 RETURNING id",
            StatementSemanticClass::Delete,
        ),
    ] {
        let body = current_command_body(sql);
        let payloads = source_payloads_with_s3(
            typed_record(0, &[7]),
            typed_record(2, &[8]),
            body,
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            41,
        );
        let draft = materialize(&payloads, 41).unwrap();
        let outcome = draft.statements[1].outcome.as_ref().unwrap();
        assert_eq!(outcome.semantic_class, expected_class);
        assert!(outcome.has_returning);
        assert!(!outcome.response_retained);
        assert_ne!(outcome.outcome.returning_digest, [0; 32]);
    }
}

#[test]
fn s6_accepts_zero_row_strict_s3_returning_noop_with_a_nonzero_digest() {
    let body = current_command_body("UPDATE source_s6 SET id = 8 WHERE id = 7 RETURNING id");
    let facts = strict_s3_facts(&body);
    assert!(facts.has_returning());
    let mut payloads = source_payloads_with_s3(
        typed_record(0, &[7]),
        typed_record(2, &[8]),
        body,
        gpu_db_wal::CanonicalFragmentKind::RowMutation,
        41,
    );
    replace_s6_outcome(
        &mut payloads[5],
        1,
        gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitNoOp,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
            target_digest: [3; 32],
            returning_digest: s3_returning_digest(facts.statement_digest()),
        },
    );
    assert!(materialize(&payloads, 41).is_ok());
}

#[test]
fn s6_strict_s3_retained_returning_closes_aggregate_outer_and_s8_counts() {
    let mut payloads = source_payloads_with_s3(
        typed_record(0, &[7]),
        typed_record(2, &[8]),
        current_command_body("DELETE FROM source_s6 WHERE id = 7 RETURNING id"),
        gpu_db_wal::CanonicalFragmentKind::RowMutation,
        41,
    );
    // S8 has no typed projection parser in this inert S1--S6 seam. It nevertheless owns one
    // counted retained artifact so S6's presence/retention closure is already exact.
    payloads[5][136 + 10] |= 2;
    payloads[7].push(8);
    assert!(materialize(&payloads, 41).is_ok());

    assert!(materialize_with_view_mutation(&payloads, 41, |view| {
        view.flags &= !AGGREGATE_FLAG_RETURNING;
    })
    .is_err());
    assert!(materialize_with_view_mutation(&payloads, 41, |view| {
        view.outer_flags &= !OUTER_CONTENT_RETURNING;
    })
    .is_err());
    assert!(materialize_with_view_mutation(&payloads, 41, |view| {
        view.sections[7].entry_count = 0;
    })
    .is_err());
}

#[test]
fn materializes_a_strict_s3_body_across_a_codec5_chunk_boundary() {
    let empty = current_set_body("");
    let value_len = MAX_SOURCE_BODY_BYTES - empty.len();
    let body = current_set_body(&"x".repeat(value_len));
    assert_eq!(body.len(), MAX_SOURCE_BODY_BYTES);
    let payloads = source_payloads(typed_record(0, &[7]), typed_record(2, &[8]), body);
    let view = source_view(&payloads, 41);
    assert!(view.measure().unwrap().chunk_count >= 2);
    let draft = materialize(&payloads, 41).unwrap();
    assert_eq!(draft.statements.len(), 3);
}

#[test]
fn materializes_a_strict_s2_record_across_a_codec5_chunk_boundary() {
    let empty = typed_record_with_text(0, String::new());
    let text_len = MAX_SOURCE_BODY_BYTES - empty.len();
    let first = typed_record_with_text(0, "x".repeat(text_len));
    assert_eq!(first.len(), MAX_SOURCE_BODY_BYTES);
    let payloads = source_payloads(first, typed_record(2, &[8]), current_set_body("s2"));
    let view = source_view(&payloads, 41);
    assert!(view.measure().unwrap().chunk_count >= 2);
    let draft = materialize(&payloads, 41).unwrap();
    assert_eq!(draft.statements.len(), 3);
}

#[test]
fn source_sabotage_rejects_swaps_duplicates_omissions_lengths_and_trailing_bytes() {
    let mut swapped = mixed_payloads();
    let ranges = s2_entry_ranges(&swapped[1]);
    let first = swapped[1][ranges[0].clone()].to_vec();
    let second = swapped[1][ranges[1].clone()].to_vec();
    swapped[1].clear();
    swapped[1].extend_from_slice(&second);
    swapped[1].extend_from_slice(&first);
    assert_rejected(&swapped, "S1/S2 typed record ordinal");

    let mut duplicate = mixed_payloads();
    let ranges = s2_entry_ranges(&duplicate[1]);
    let first = duplicate[1][ranges[0].clone()].to_vec();
    duplicate[1].truncate(ranges[0].end);
    duplicate[1].extend_from_slice(&first);
    assert_rejected(&duplicate, "S1/S2 typed record ordinal");

    let mut omitted = mixed_payloads();
    let ranges = s2_entry_ranges(&omitted[1]);
    omitted[1].truncate(ranges[0].end);
    assert_rejected(&omitted, "section reader is truncated");

    let mut wrong_length = mixed_payloads();
    let len = u32::from_le_bytes(wrong_length[1][0..4].try_into().unwrap());
    wrong_length[1][0..4].copy_from_slice(&(len + 1).to_le_bytes());
    assert_rejected(&wrong_length, "S2 strict decode");

    let mut trailing = mixed_payloads();
    trailing[1].push(0);
    assert_rejected(&trailing, "section reader leaves trailing bytes");

    let mut omitted_s3 = mixed_payloads();
    omitted_s3[2].clear();
    assert_rejected(&omitted_s3, "S3 source prefix is truncated");

    let mut duplicate_s3 = mixed_payloads();
    let first_s3 = duplicate_s3[2].clone();
    duplicate_s3[2].extend_from_slice(&first_s3);
    assert_rejected(&duplicate_s3, "section reader leaves trailing bytes");

    let mut truncated_s3_length = mixed_payloads();
    let body_len = u32::from_le_bytes(truncated_s3_length[2][12..16].try_into().unwrap());
    truncated_s3_length[2][12..16].copy_from_slice(&(body_len + 1).to_le_bytes());
    assert_rejected(&truncated_s3_length, "section reader is truncated");
}

#[test]
fn source_sabotage_rejects_family_ordinal_digest_row_parent_and_s4_binding_drift() {
    let mut family = mixed_payloads();
    family[2][4..8].copy_from_slice(&1_u32.to_le_bytes());
    assert_rejected(&family, "S1/S3 ordinal");

    let mut statement_ordinal = mixed_payloads();
    statement_ordinal[2][0..4].copy_from_slice(&2_u32.to_le_bytes());
    assert_rejected(&statement_ordinal, "S1/S3 ordinal");

    let mut digest = mixed_payloads();
    digest[2][16] ^= 1;
    assert_rejected(&digest, "body digest");

    let mut typed_row_count = mixed_payloads();
    let one_row = typed_record(0, &[77]);
    let ranges = s2_entry_ranges(&typed_row_count[1]);
    let mut replacement = Vec::with_capacity(4 + one_row.len());
    replacement.extend_from_slice(&u32::try_from(one_row.len()).unwrap().to_le_bytes());
    replacement.extend_from_slice(&one_row);
    typed_row_count[1].splice(ranges[0].clone(), replacement);
    assert_rejected(
        &typed_row_count,
        "S1/S2 typed record ordinal, row count, or digest",
    );

    let serial = serial_typed_record(0, 71, false);
    let parent_payloads =
        source_payloads(serial, typed_record(2, &[8]), current_set_body("parent"));
    assert!(materialize(&parent_payloads, 71).is_ok());
    let error = match materialize(&parent_payloads, 72) {
        Ok(_) => panic!("foreign sequence parent must reject"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("sequence parent"));

    let mismatched_mode = source_payloads(
        serial_typed_record(0, 41, true),
        typed_record(2, &[8]),
        current_set_body("mode"),
    );
    assert_rejected(
        &mismatched_mode,
        "parent autocommit does not bind aggregate transaction mode",
    );

    let mut s4 = mixed_payloads();
    s4[3][32] ^= 1;
    assert_rejected(&s4, "canonical statement/source/digest/allocator order");
}

#[test]
fn source_draft_has_no_raw_source_carrier_or_live_conversion() {
    let source = include_str!("../source_materialization.rs");
    let owner = source
        .split("pub(crate) struct TypedInsertAggregateSourceDraft")
        .nth(1)
        .unwrap()
        .split("/// Materialize")
        .next()
        .unwrap();
    assert!(
        owner.contains("Box<[SourceStatement]>") && owner.contains("Box<[CompactRowDisposition]>")
    );
    for forbidden in [
        "Vec<u8>",
        "Box<[u8]>",
        "impl Clone",
        "DeviceInsertPlan",
        "into_recovery",
    ] {
        assert!(
            !owner.contains(forbidden),
            "draft owner retained or exposed {forbidden}"
        );
    }
    let sequence_outcomes = include_str!("sequence_outcomes.rs");
    assert!(
        sequence_outcomes.contains("try_reserve_exact"),
        "S5 must reserve its one compact owner before chunk decoding"
    );
    for forbidden in [
        "Vec<u8>",
        "Box<[u8]>",
        "impl Clone",
        "DeviceInsertPlan",
        "into_recovery",
        "into_device",
    ] {
        assert!(
            !sequence_outcomes.contains(forbidden),
            "S5/S6 source closure retained or exposed {forbidden}"
        );
    }
}
