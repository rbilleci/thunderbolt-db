//! Focused nonempty neutral-input fixture; it exercises no logical reencoder or live path.

use super::{input, sealed};
use crate::typed_insert_aggregate::semantics_v2::{
    pass_zero::SemanticsV2S7HeaderIdentity,
    retained::{
        graph::{
            ReservedSemanticsV2Graph, RetainedIndexDescriptor, RetainedIndexKeyColumn,
            RetainedKeyComponent, RetainedKeyEffect, RetainedTable, RetainedTransition,
        },
        SemanticsV2BoundIdentity, SemanticsV2CatalogColumnWitness,
        SemanticsV2CatalogIndexKeyWitness, SemanticsV2CatalogIndexWitness,
        SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness,
    },
};
use crate::typed_insert_batch::{
    decode_typed_image, encode_typed_image, TypedImageColumnView, TypedImageRole, TypedImageView,
    TypedInsertColumnValidity, TypedInsertColumnValues,
};
use crate::SqlType;
use sha2::{Digest, Sha256};

const TABLE_ID: u64 = 500;
const INDEX_ID: u64 = 600;
const CATALOG_EPOCH: u64 = 4;
const I32_STORAGE: [u8; 4] = [2, 0, 0, 0];
const TEXT_STORAGE: [u8; 4] = [6, 0, 0, 0];

struct FixtureCandidate;
struct FixtureWork;

impl sealed::Candidate for FixtureCandidate {}
impl sealed::Work for FixtureWork {}

#[test]
fn nonempty_neutral_fixture_measures_fills_and_matches_independent_digest() {
    let identity = SemanticsV2BoundIdentity {
        database_id: [1; 16],
        cluster_id: [2; 16],
        timeline_id: [3; 16],
        format_epoch: 1,
        leader_epoch: 1,
        catalog_epoch: CATALOG_EPOCH,
        catalog_digest: [4; 32],
        stable_transaction_id: 90,
        autocommit: true,
        commit_sequence: 91,
        initial_database_root: [5; 32],
    };
    let int_values = TypedInsertColumnValues::I32(vec![42].into_boxed_slice());
    let null_text_values = TypedInsertColumnValues::Text {
        offsets: vec![0, 0].into_boxed_slice(),
        bytes: Vec::new().into_boxed_slice(),
    };
    let all_valid = TypedInsertColumnValidity::AllValid;
    let null_row = TypedInsertColumnValidity::Bitmap(vec![0].into_boxed_slice());
    let image_columns = [
        TypedImageColumnView {
            catalog_column_ordinal: 0,
            stable_column_id: 701,
            table_ref: 0,
            attnum: 1,
            ty: SqlType::Int4,
            type_oid: 23,
            type_size: 4,
            result_format: 0,
            name: "",
            validity: &all_valid,
            values: &int_values,
        },
        TypedImageColumnView {
            catalog_column_ordinal: 1,
            stable_column_id: 702,
            table_ref: 0,
            attnum: 2,
            ty: SqlType::Text,
            type_oid: 25,
            type_size: -1,
            result_format: 0,
            name: "",
            validity: &null_row,
            values: &null_text_values,
        },
    ];
    let image_bytes = encode_typed_image(&TypedImageView {
        role: TypedImageRole::FinalTableImage,
        rows: 1,
        columns: &image_columns,
    })
    .expect("nonempty image fixture encodes through the checked typed-image codec");
    let image = decode_typed_image(&image_bytes).expect("nonempty image fixture decodes");
    let layout = image.facts().layout_digest;
    let key_value_digest = typed_i32_value_digest(42);

    let columns = [
        catalog_column(0, 701, 1, "id", I32_STORAGE, 23, 4),
        catalog_column(1, 702, 2, "note", TEXT_STORAGE, 25, -1),
    ];
    let keys = [SemanticsV2CatalogIndexKeyWitness {
        key_ordinal: 0,
        owner_catalog_column_ordinal: 0,
        stable_column_id: 701,
        attnum: 1,
        name: "id",
        storage: I32_STORAGE,
        declared_type_oid: 23,
        signed_type_size: 4,
        column_name_digest: [11; 32],
    }];
    let tables = [SemanticsV2CatalogTableWitness {
        stable_table_id: TABLE_ID,
        display_oid: 50,
        schema: "public",
        name: "fixture",
        schema_digest: [6; 32],
        data_generation: 8,
        data_root: [7; 32],
        catalog_columns: &columns,
        not_null_guards: &[],
        check_guards: &[],
        foreign_keys: &[],
    }];
    let indexes = [SemanticsV2CatalogIndexWitness {
        stable_index_id: INDEX_ID,
        display_oid: 60,
        owner_stable_table_id: TABLE_ID,
        owner_display_oid: 50,
        schema: "public",
        name: "fixture_id_key",
        owner_schema: "public",
        owner_name: "fixture",
        constraint_stable_id: 0,
        constraint_display_oid: 0,
        constraint_schema: "",
        constraint_name: "",
        schema_digest: [6; 32],
        index_flags: 3,
        null_equality_policy: 0,
        raw_catalog_ordinal: 0,
        key_columns: &keys,
        catalog_epoch: CATALOG_EPOCH,
        base_generation: 8,
        base_root: [8; 32],
    }];
    let catalog = SemanticsV2CatalogWitness {
        database_id: identity.database_id,
        catalog_epoch: identity.catalog_epoch,
        catalog_digest: identity.catalog_digest,
        tables: &tables,
        indexes: &indexes,
        domains: &[],
        guards: &[],
        sequences: &[],
    };
    let graph = fixture_graph(image, layout, key_value_digest);

    let independent = input::generation_input_digest(&graph, &catalog, identity)
        .expect("independent retained traversal accepts nonempty fixture");
    let measure = input::measure(&graph, &catalog, identity)
        .expect("neutral traversal measures nonempty fixture");
    let reservation = measure.builder_reservation();
    assert_eq!(reservation.tables(), 1);
    assert_eq!(reservation.rows(), 1);
    assert_eq!(reservation.cells(), 2);
    assert_eq!(reservation.value_bytes(), 4);
    assert_eq!(reservation.indexes(), 1);
    assert_eq!(reservation.keys(), 1);
    assert_eq!(reservation.effects(), 1);
    assert_eq!(reservation.effect_values(), 1);
    assert_eq!(reservation.effect_value_bytes(), 4);
    assert_eq!(reservation.table_outputs(), 1);
    assert_eq!(reservation.index_outputs(), 1);

    let registry = input::GenerationQuarantineRegistry::<FixtureCandidate, FixtureWork>::new();
    let launch = input::reserve_and_fill(
        measure,
        &graph,
        &catalog,
        identity,
        FixtureCandidate,
        &registry,
    )
    .expect("nonempty fixture fills the exact sealed neutral owners");
    let neutral = launch.input().neutral_view();
    assert_eq!(neutral.tables().count(), 1);
    assert_eq!(neutral.rows().count(), 1);
    assert_eq!(neutral.cells().count(), 2);
    assert!(neutral.cells().nth(1).expect("text cell").is_null);
    assert_eq!(neutral.indexes().count(), 1);
    assert_eq!(neutral.keys().count(), 1);
    assert_eq!(neutral.effects().count(), 1);
    assert_eq!(neutral.effect_values().count(), 1);
    assert_eq!(
        launch
            .input()
            .builder_input_digest()
            .expect("sealed neutral traversal remains exact"),
        independent,
    );
    drop(launch);
}

fn catalog_column(
    ordinal: u32,
    stable_column_id: u32,
    attnum: i16,
    name: &'static str,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
) -> SemanticsV2CatalogColumnWitness<'static> {
    SemanticsV2CatalogColumnWitness {
        catalog_column_ordinal: ordinal,
        stable_column_id,
        attnum,
        name,
        storage,
        declared_type_oid,
        signed_type_size,
        column_shape_digest: [9; 32],
        column_root: [10; 32],
    }
}

fn fixture_graph(
    image: crate::typed_insert_batch::DecodedTypedImage,
    layout: [u8; 32],
    key_value_digest: [u8; 32],
) -> ReservedSemanticsV2Graph {
    ReservedSemanticsV2Graph {
        header: SemanticsV2S7HeaderIdentity {
            total_bytes: 0,
            root_descriptor_version: 1,
            catalog_before_epoch: CATALOG_EPOCH,
            catalog_after_epoch: CATALOG_EPOCH,
            catalog_before_digest: [4; 32],
            catalog_after_digest: [4; 32],
            initial_database_root: [5; 32],
            final_database_root: [12; 32],
            initial_overlay_root: [13; 32],
            final_overlay_root: [14; 32],
            root_descriptor: [15; 32],
            payload_digest: [16; 32],
        },
        statements: Vec::new(),
        records: Vec::new(),
        dispositions: Vec::new(),
        sequence_effects: Vec::new(),
        outcomes: Vec::new(),
        tables: vec![RetainedTable {
            table_ref: 0,
            stable_table_id: TABLE_ID,
            display_oid: 50,
            target_dependency_ref: 0,
            catalog_epoch: CATALOG_EPOCH,
            data_generation_before: 8,
            data_generation_after: 9,
            row_allocator_before: 100,
            row_allocator_high_water: 101,
            initial_logical_row_count: 1,
            final_logical_row_count: 2,
            disposition_start: 0,
            disposition_count: 0,
            transition_start: 0,
            transition_count: 1,
            key_effect_start: 0,
            key_effect_count: 1,
            owned_index_start: 0,
            owned_index_count: 1,
            image_ref: 0,
            catalog_column_count: 2,
            schema_digest: [6; 32],
            initial_table_root: [7; 32],
            final_table_root: [17; 32],
            transition_root: [18; 32],
            index_effect_root: [19; 32],
            image_layout_digest: layout,
            image_content_digest: [20; 32],
            image_arena_offset: 0,
            image_encoded_bytes: 0,
            image_descriptor_digest: [21; 32],
            manifest_digest: [22; 32],
        }],
        table_dispositions: Vec::new(),
        resolutions: Vec::new(),
        dependencies: Vec::new(),
        dependency_uses: Vec::new(),
        indexes: vec![RetainedIndexDescriptor {
            index_ref: 0,
            owner_table_ref: 0,
            raw_catalog_ordinal: 0,
            stable_index_id: INDEX_ID,
            display_oid: 60,
            stable_constraint_id: 0,
            constraint_display_oid: 0,
            flags: 3,
            null_equality_policy: 0,
            key_start: 0,
            key_count: 1,
            catalog_epoch: CATALOG_EPOCH,
            owner_stable_table_id: TABLE_ID,
            owner_display_oid: 50,
            owner_schema_digest: [6; 32],
            owner_name_digest: [23; 32],
            index_name_digest: [24; 32],
            constraint_name_digest: [0; 32],
            owner_table_base_root: [7; 32],
            base_index_root: [8; 32],
            final_index_root: [25; 32],
            descriptor_digest: [26; 32],
            owner_data_generation: 8,
            base_index_generation: 8,
            final_index_generation: 9,
        }],
        index_key_columns: vec![RetainedIndexKeyColumn {
            key_column_ref: 0,
            index_ref: 0,
            key_ordinal: 0,
            owner_catalog_column_ordinal: 0,
            stable_column_id: 701,
            owner_display_table_oid: 50,
            attnum: 1,
            storage: I32_STORAGE,
            declared_type_oid: 23,
            signed_type_size: 4,
            column_name_digest: [11; 32],
            key_digest: [27; 32],
        }],
        transitions: vec![RetainedTransition {
            transition_ref: 0,
            table_ref: 0,
            stable_row_id: 900,
            source_disposition_ref: 0,
            source_statement_ordinal: 0,
            source_row_ordinal: 0,
            image_ref: 0,
            image_row_ordinal: 0,
            key_effect_start: 0,
            key_effect_count: 1,
            final_writer_statement_ordinal: 0,
            typed_statement_digest: [28; 32],
            final_row_digest: [29; 32],
            transition_digest: [30; 32],
        }],
        key_effects: vec![RetainedKeyEffect {
            effect_ref: 0,
            role: 1,
            action: 1,
            transition_ref: 0,
            index_ref: 0,
            dependency_ref: 0,
            new_component_start: 0,
            new_component_count: 1,
            key_arity: 1,
            participates: true,
            contains_null: false,
            source_catalog_ordinal: 0,
            typed_key_digest: [31; 32],
            effect_digest: [32; 32],
        }],
        key_components: vec![RetainedKeyComponent {
            component_ref: 0,
            effect_ref: 0,
            side: 1,
            validity: 1,
            component_ordinal: 0,
            key_column_ref: 0,
            source_catalog_ordinal: 0,
            value_arena_offset: 0,
            value_bytes: 4,
            storage: I32_STORAGE,
            declared_type_oid: 23,
            signed_type_size: 4,
            typed_value_digest: key_value_digest,
            component_digest: [33; 32],
        }],
        projections: Vec::new(),
        images: vec![image],
    }
}

fn typed_i32_value_digest(value: i32) -> [u8; 32] {
    let domain = b"gpu-db/write001/s7-typed-key-value/v2";
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(I32_STORAGE);
    digest.update(23_u32.to_le_bytes());
    digest.update(4_i16.to_le_bytes());
    digest.update([0]);
    digest.update(4_u32.to_le_bytes());
    digest.update(value.to_le_bytes());
    digest.finalize().into()
}
