//! Focused semantics-v2 live-writer regression tests.
//!
//! These test the unchanged writer facade and its private grammar-format leaf.

use super::*;
use crate::insert_semantic_ir::InsertStatementOrdinal;
use crate::typed_insert_batch::{
    prepare_typed_insert_semantics_at, sequence_defaults::SequenceDefaultBindings,
};
use crate::{parse_command, Command, Engine};

#[test]
fn transition_digest_streaming_preserves_the_frozen_contiguous_form() {
    let typed_statement_digest = [0x41; 32];
    let mut transitions = Vec::new();
    for row in 0..3_u32 {
        transitions.push(transition_bytes(
            row,
            0,
            100 + u64::from(row),
            row,
            0,
            row,
            0,
            row,
            0,
            0,
            0,
            typed_statement_digest,
            [row as u8; 32],
            transition_digest(
                row,
                0,
                100 + u64::from(row),
                row,
                0,
                row,
                0,
                row,
                0,
                0,
                0,
                typed_statement_digest,
                [row as u8; 32],
                [0; 32],
            ),
            [0; 32],
        ));
    }
    let contiguous: Vec<[u8; 32]> = transitions
        .iter()
        .map(|transition| transition[128..160].try_into().unwrap())
        .collect();
    assert_eq!(
        digest_transition_records(
            b"gpu-db/write001/s7-table-transition-root/v2",
            17,
            &transitions,
        )
        .unwrap(),
        digest_list(
            b"gpu-db/write001/s7-table-transition-root/v2",
            17,
            &contiguous,
        ),
    );

    let table = [0x19; 384];
    let target = [0x29; 32];
    let dispositions = [0x39; 64];
    let image = [0x49; 32];
    let old_manifest = v2_digest(
        b"gpu-db/write001/s7-table-manifest/v2",
        &[
            &table[..352],
            &[0; 32],
            &target,
            &dispositions,
            &contiguous.concat(),
            &image,
        ],
    );
    assert_eq!(
        table_manifest_digest(&table, target, &dispositions, &[], &transitions, &[], image)
            .unwrap(),
        old_manifest,
    );
    let mut changed = transitions.clone();
    changed[0][128] ^= 1;
    assert_ne!(
        table_manifest_digest(&table, target, &dispositions, &[], &changed, &[], image).unwrap(),
        old_manifest,
    );
}

#[test]
fn mixed_type_null_writer_strictly_closes_semantics_two() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE writer_people (id INT4, note TEXT, active BOOL, score INT8)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let table = catalog.relational_catalog["writer_people"].clone();
    let Command::Insert(insert) = parse_command(
        "INSERT INTO writer_people (score, active, note, id) VALUES \
         (9000000000, TRUE, 'alpha', 1), (NULL, FALSE, NULL, 2) \
         RETURNING note, score, active, id, note",
    )
    .unwrap() else {
        unreachable!("writer test SQL is INSERT")
    };
    let prepared = prepare_typed_insert_semantics_at(
        &insert,
        &catalog,
        catalog.commit_seq,
        None,
        InsertStatementOrdinal::from_u32(0),
    )
    .unwrap()
    .expect("mixed typed INSERT prepares");
    let typed_statement_digest = prepared.typed_statement_digest();
    let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
    let sources = batch.seal_codec5_sources().unwrap();
    let table_schema_digest = crate::engine_transaction_reset::table_schema_digest(&table).unwrap();
    let source =
        crate::typed_insert_batch::PreparedResidentAppendSource::from_decoded_final_table_image(
            crate::typed_insert_batch::decode_typed_image(sources.final_image()).unwrap(),
            &table,
            table_schema_digest,
            catalog.commit_seq,
        )
        .expect("mixed typed image has one resident source");
    let row_sources = [
        crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
            stable_row_id: 100,
            statement_ordinal: 0,
            source_row_ordinal: 0,
        },
        crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
            stable_row_id: 101,
            statement_ordinal: 0,
            source_row_ordinal: 1,
        },
    ];
    let runtime = source
        .prepare_runtime_generation_view(&row_sources)
        .unwrap();
    assert_eq!(runtime.geometry().value_bytes, 23);
    let final_writers = [
        LiveTypedInsertFinalWriter {
            stable_row_id: 100,
            source_statement_ordinal: 0,
            source_row_ordinal: 0,
            final_writer_statement_ordinal: 0,
            final_writer_statement_digest: [0; 32],
            survives: true,
        },
        LiveTypedInsertFinalWriter {
            stable_row_id: 101,
            source_statement_ordinal: 0,
            source_row_ordinal: 1,
            final_writer_statement_ordinal: 0,
            final_writer_statement_digest: [0; 32],
            survives: true,
        },
    ];
    let encoded = encode_live_typed_insert(&LiveTypedInsertView {
        identity: LiveTypedInsertIdentity {
            physical: gpu_db_wal::CanonicalPhysicalRange {
                log_epoch: 1,
                lane_id: 0,
                segment_id: 17,
                first_frame_ordinal: 0,
            },
            canonical: gpu_db_wal::CanonicalIdentity {
                database_id: [1; 16],
                cluster_id: [2; 16],
                timeline_id: [3; 16],
                format_epoch: 1,
            },
            leader_epoch: 1,
            commit_sequence: 17,
            stable_transaction_id: 77,
            mode: LiveTypedInsertMode::Autocommit,
            request_digest: live_autocommit_request_digest(typed_statement_digest),
            isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
            catalog_epoch: catalog.commit_seq,
            catalog_digest: [4; 32],
            catalog_after_epoch: catalog.commit_seq,
            catalog_after_digest: [4; 32],
            dependency_validation_floor: catalog.commit_seq,
        },
        table: LiveTypedInsertTableGeneration {
            schema: &table.schema,
            name: &table.name,
            stable_table_id: table.stable_table_id,
            display_oid: table.oid,
            schema_digest: table_schema_digest,
            image_content_digest: v2_digest(
                b"gpu-db/write001/s7-image-content/v2",
                &[
                    &(sources.final_image().len() as u64).to_le_bytes(),
                    sources.final_image(),
                ],
            ),
            data_generation_before: 11,
            data_generation_after: 17,
            row_allocator_before: 100,
            row_allocator_high_water: 102,
            initial_logical_row_count: 0,
            final_logical_row_count: 2,
            initial_table_root: [5; 32],
            final_table_root: [6; 32],
            initial_database_root: [7; 32],
            final_database_root: [8; 32],
            resets_existing_rows: false,
            initial_table_absent: false,
            indexes: &[],
            created_indexes_on_existing_table: &[],
        },
        statements: &[LiveTypedInsertStatementView {
            statement_ordinal: 0,
            operation_ordinal: 0,
            table_ref: 0,
            table_schema_digest,
            typed_statement_digest,
            record: sources.record(),
            sealed_source: Some(&sources),
        }],
        final_image: sources.final_image(),
        foreign_indexes: &[],
        published_sequence_references: &[],
        final_row_digests: Some(&runtime),
        final_writers: &final_writers,
        source_geometry: Some(runtime.geometry()),
    })
    .unwrap();
    let fragments = encoded.envelope.bodies().canonical_fragment_refs();
    super::super::close_canonical_semantics_v2_for_test(
        encoded.envelope.header(),
        encoded.envelope.outcome(),
        fragments.as_slice(),
    )
    .expect("mixed type and NULL codec-5 writer closes strict S1-S8 semantics");
}

#[test]
fn provenance_sealed_feature_free_record_rejects_a_live_ordinal_mismatch() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE writer_provenance (id INT4)")
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let table = catalog.relational_catalog["writer_provenance"].clone();
    let Command::Insert(insert) =
        parse_command("INSERT INTO writer_provenance VALUES (1)").unwrap()
    else {
        unreachable!("writer test SQL is INSERT")
    };
    // Seal S2 for ordinal one, then deliberately present its exact immutable record at ordinal
    // zero. The fast provenance check must not hide the strict decoder's ordinal rejection.
    let prepared = prepare_typed_insert_semantics_at(
        &insert,
        &catalog,
        catalog.commit_seq,
        None,
        InsertStatementOrdinal::from_u32(1),
    )
    .unwrap()
    .expect("feature-free typed INSERT prepares");
    let typed_statement_digest = prepared.typed_statement_digest();
    let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
    let sources = batch.seal_codec5_sources().unwrap();
    assert_eq!(sources.feature_free_live_row_count_for(0), None);
    assert_eq!(sources.feature_free_live_row_count_for(1), Some(1));
    let table_schema_digest = crate::engine_transaction_reset::table_schema_digest(&table).unwrap();
    let source =
        crate::typed_insert_batch::PreparedResidentAppendSource::from_decoded_final_table_image(
            crate::typed_insert_batch::decode_typed_image(sources.final_image()).unwrap(),
            &table,
            table_schema_digest,
            catalog.commit_seq,
        )
        .expect("feature-free typed image has one resident source");
    let row_sources = [
        crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
            stable_row_id: 100,
            statement_ordinal: 0,
            source_row_ordinal: 0,
        },
    ];
    let runtime = source
        .prepare_runtime_generation_view(&row_sources)
        .unwrap();
    let final_writers = [LiveTypedInsertFinalWriter {
        stable_row_id: 100,
        source_statement_ordinal: 0,
        source_row_ordinal: 0,
        final_writer_statement_ordinal: 0,
        final_writer_statement_digest: [0; 32],
        survives: true,
    }];
    let result = encode_live_typed_insert(&LiveTypedInsertView {
        identity: LiveTypedInsertIdentity {
            physical: gpu_db_wal::CanonicalPhysicalRange {
                log_epoch: 1,
                lane_id: 0,
                segment_id: 17,
                first_frame_ordinal: 0,
            },
            canonical: gpu_db_wal::CanonicalIdentity {
                database_id: [1; 16],
                cluster_id: [2; 16],
                timeline_id: [3; 16],
                format_epoch: 1,
            },
            leader_epoch: 1,
            commit_sequence: 17,
            stable_transaction_id: 77,
            mode: LiveTypedInsertMode::Autocommit,
            request_digest: live_autocommit_request_digest(typed_statement_digest),
            isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
            catalog_epoch: catalog.commit_seq,
            catalog_digest: [4; 32],
            catalog_after_epoch: catalog.commit_seq,
            catalog_after_digest: [4; 32],
            dependency_validation_floor: catalog.commit_seq,
        },
        table: LiveTypedInsertTableGeneration {
            schema: &table.schema,
            name: &table.name,
            stable_table_id: table.stable_table_id,
            display_oid: table.oid,
            schema_digest: table_schema_digest,
            image_content_digest: v2_digest(
                b"gpu-db/write001/s7-image-content/v2",
                &[
                    &(sources.final_image().len() as u64).to_le_bytes(),
                    sources.final_image(),
                ],
            ),
            data_generation_before: 11,
            data_generation_after: 17,
            row_allocator_before: 100,
            row_allocator_high_water: 101,
            initial_logical_row_count: 0,
            final_logical_row_count: 1,
            initial_table_root: [5; 32],
            final_table_root: [6; 32],
            initial_database_root: [7; 32],
            final_database_root: [8; 32],
            resets_existing_rows: false,
            initial_table_absent: false,
            indexes: &[],
            created_indexes_on_existing_table: &[],
        },
        statements: &[LiveTypedInsertStatementView {
            statement_ordinal: 0,
            operation_ordinal: 0,
            table_ref: 0,
            table_schema_digest,
            typed_statement_digest,
            record: sources.record(),
            sealed_source: Some(&sources),
        }],
        final_image: sources.final_image(),
        foreign_indexes: &[],
        published_sequence_references: &[],
        final_row_digests: Some(&runtime),
        final_writers: &final_writers,
        source_geometry: Some(runtime.geometry()),
    });
    assert!(result.is_err(), "mismatched sealed S2 ordinal must reject");
    let error = result.err().expect("rejection was checked above");
    assert!(
        error
            .to_string()
            .contains("S2 statement ordinal, digest, or row geometry differs"),
        "unexpected ordinal-sabotage error: {error}"
    );
}

#[test]
fn compound_nonunique_index_writer_strictly_closes_semantics_two() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            900_103,
            "CREATE TABLE writer_indexed (tenant_id INT4, status INT4)",
        )
        .unwrap();
    engine
        .execute_text(
            900_104,
            "CREATE INDEX writer_indexed_by_tenant_status ON writer_indexed (tenant_id, status)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let table = catalog.relational_catalog["writer_indexed"].clone();
    assert_eq!(table.indexes.len(), 1, "fixture has one owned named index");
    let Command::Insert(insert) =
        parse_command("INSERT INTO writer_indexed VALUES (7, 11), (7, 13)").unwrap()
    else {
        unreachable!("writer test SQL is INSERT")
    };
    let prepared = prepare_typed_insert_semantics_at(
        &insert,
        &catalog,
        catalog.commit_seq,
        None,
        InsertStatementOrdinal::from_u32(0),
    )
    .unwrap()
    .expect("compound indexed typed INSERT prepares");
    let typed_statement_digest = prepared.typed_statement_digest();
    let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
    let sources = batch.seal_codec5_sources().unwrap();
    let source = batch
        .into_codec5_resident_append_source_for_test(&catalog)
        .expect("compound indexed codec-5 image has one resident source");
    let row_sources = [
        crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
            stable_row_id: 100,
            statement_ordinal: 0,
            source_row_ordinal: 0,
        },
        crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
            stable_row_id: 101,
            statement_ordinal: 0,
            source_row_ordinal: 1,
        },
    ];
    let runtime = source
        .prepare_runtime_generation_view(&row_sources)
        .unwrap();
    let final_writers = [
        LiveTypedInsertFinalWriter {
            stable_row_id: 100,
            source_statement_ordinal: 0,
            source_row_ordinal: 0,
            final_writer_statement_ordinal: 0,
            final_writer_statement_digest: [0; 32],
            survives: true,
        },
        LiveTypedInsertFinalWriter {
            stable_row_id: 101,
            source_statement_ordinal: 0,
            source_row_ordinal: 1,
            final_writer_statement_ordinal: 0,
            final_writer_statement_digest: [0; 32],
            survives: true,
        },
    ];
    let index_generations = [LiveTypedInsertIndexGeneration {
        catalog: &table.indexes[0],
        base_generation: 11,
        base_root: [0x19; 32],
        final_generation: 17,
        final_root: [0x1a; 32],
    }];
    let encoded = encode_live_typed_insert(&LiveTypedInsertView {
        identity: LiveTypedInsertIdentity {
            physical: gpu_db_wal::CanonicalPhysicalRange {
                log_epoch: 1,
                lane_id: 0,
                segment_id: 17,
                first_frame_ordinal: 0,
            },
            canonical: gpu_db_wal::CanonicalIdentity {
                database_id: [1; 16],
                cluster_id: [2; 16],
                timeline_id: [3; 16],
                format_epoch: 1,
            },
            leader_epoch: 1,
            commit_sequence: 17,
            stable_transaction_id: 77,
            mode: LiveTypedInsertMode::Autocommit,
            request_digest: live_autocommit_request_digest(typed_statement_digest),
            isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
            catalog_epoch: catalog.commit_seq,
            catalog_digest: [4; 32],
            catalog_after_epoch: catalog.commit_seq,
            catalog_after_digest: [4; 32],
            dependency_validation_floor: catalog.commit_seq,
        },
        table: LiveTypedInsertTableGeneration {
            schema: &table.schema,
            name: &table.name,
            stable_table_id: table.stable_table_id,
            display_oid: table.oid,
            schema_digest: crate::engine_transaction_reset::table_schema_digest(&table).unwrap(),
            image_content_digest: v2_digest(
                b"gpu-db/write001/s7-image-content/v2",
                &[
                    &(sources.final_image().len() as u64).to_le_bytes(),
                    sources.final_image(),
                ],
            ),
            data_generation_before: 11,
            data_generation_after: 17,
            row_allocator_before: 100,
            row_allocator_high_water: 102,
            initial_logical_row_count: 0,
            final_logical_row_count: 2,
            initial_table_root: [5; 32],
            final_table_root: [6; 32],
            initial_database_root: [7; 32],
            final_database_root: [8; 32],
            resets_existing_rows: false,
            initial_table_absent: false,
            indexes: &index_generations,
            created_indexes_on_existing_table: &[],
        },
        statements: &[LiveTypedInsertStatementView {
            statement_ordinal: 0,
            operation_ordinal: 0,
            table_ref: 0,
            table_schema_digest: crate::engine_transaction_reset::table_schema_digest(&table)
                .unwrap(),
            typed_statement_digest,
            record: sources.record(),
            sealed_source: None,
        }],
        final_image: sources.final_image(),
        foreign_indexes: &[],
        published_sequence_references: &[],
        // The test-only source supplies the canonical final-row digest. Indexed S7
        // transition/effect closure is deliberately serialized by this writer, so no
        // second CPU index or successor-root authority is introduced.
        final_row_digests: Some(&runtime),
        final_writers: &final_writers,
        source_geometry: Some(runtime.geometry()),
    })
    .expect("indexed writer serializes one canonical S7 closure");
    let fragments = encoded.envelope.bodies().canonical_fragment_refs();
    super::super::close_canonical_semantics_v2_for_test(
        encoded.envelope.header(),
        encoded.envelope.outcome(),
        fragments.as_slice(),
    )
    .expect("compound indexed codec-5 writer strictly closes retained S1-S8 semantics");
}
