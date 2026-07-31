//! Q2 catalog-guard hostile evidence over the real codec-closed A/B/A owner.
//!
//! The catalog below is fixed witness material. Unchanged S2 is decoded only to prove it still
//! agrees with those literals; the hostile S7 mutation never supplies catalog facts.

use super::{
    close_canonical_semantics_v2_for_test, measure_canonical_semantics_v2, q1_sabotage, q1_vectors,
    validate_canonical_semantics_v2_guards_for_test,
};
use crate::typed_insert_aggregate::semantics_v2::retained::{
    SemanticsV2CatalogColumnWitness, SemanticsV2CatalogGuardWitness,
    SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness,
};
use crate::SqlType;

const ABSENT_U32: u32 = u32::MAX;
const CATALOG_EPOCH: u64 = 17;
const PARENT_OID: u32 = 16_384;
const CHILD_OID: u32 = 16_387;
const PARENT_SCHEMA_DIGEST: [u8; 32] = [
    0x64, 0xb2, 0xbb, 0xb1, 0xf7, 0xba, 0x21, 0xaf, 0x78, 0xd7, 0xc0, 0x3d, 0x61, 0x7e, 0xe3, 0xf9,
    0x68, 0x28, 0x0e, 0xcc, 0xec, 0xf6, 0xe9, 0x32, 0x9a, 0xe6, 0x25, 0xba, 0x85, 0xe3, 0x27, 0x52,
];
const CHILD_SCHEMA_DIGEST: [u8; 32] = [
    0xc6, 0x3d, 0xf5, 0xde, 0x09, 0x40, 0xac, 0xd6, 0xa8, 0xa3, 0x27, 0x1c, 0x4b, 0xa0, 0x04, 0x8c,
    0xac, 0x9c, 0x82, 0x38, 0x0a, 0xdc, 0x81, 0x98, 0x96, 0x03, 0xc2, 0xca, 0x89, 0x37, 0xf1, 0xad,
];

#[test]
fn q2_owner_swapped_a_b_a_guard_rejects_only_after_real_codec_close() {
    let unchanged = q1_vectors::successful_a_b_a_fixture();
    assert_a_b_a_target_columns(&unchanged);
    let parent_columns = [catalog_column(1, [0x61; 32], [0x62; 32])];
    let child_columns = [catalog_column(5, [0x63; 32], [0x64; 32])];
    let parent_guards = [not_null_guard(
        1_901, 1_101, PARENT_OID, [0x61; 32], [0x62; 32], 11,
    )];
    let child_guards = [not_null_guard(
        1_902, 1_102, CHILD_OID, [0x63; 32], [0x64; 32], 21,
    )];
    let global_guards = [
        not_null_guard(1_901, 1_101, PARENT_OID, [0x61; 32], [0x62; 32], 11),
        not_null_guard(1_902, 1_102, CHILD_OID, [0x63; 32], [0x64; 32], 21),
    ];
    let tables = [
        SemanticsV2CatalogTableWitness {
            stable_table_id: 1_101,
            display_oid: PARENT_OID,
            schema: "public",
            name: "q1_parent",
            schema_digest: PARENT_SCHEMA_DIGEST,
            data_generation: 11,
            data_root: [0x22; 32],
            catalog_columns: &parent_columns,
            not_null_guards: &parent_guards,
            check_guards: &[],
            foreign_keys: &[],
        },
        SemanticsV2CatalogTableWitness {
            stable_table_id: 1_102,
            display_oid: CHILD_OID,
            schema: "public",
            name: "q1_child",
            schema_digest: CHILD_SCHEMA_DIGEST,
            data_generation: 21,
            data_root: [0x32; 32],
            catalog_columns: &child_columns,
            not_null_guards: &child_guards,
            check_guards: &[],
            foreign_keys: &[],
        },
    ];
    let catalog = SemanticsV2CatalogWitness {
        // Matches the independently pinned A/B/A status identity.  The guard-only leaf does
        // not consume it, but the catalog remains plausible to the enclosing identity gate.
        database_id: [0xa1; 16],
        catalog_epoch: CATALOG_EPOCH,
        catalog_digest: [0x33; 32],
        tables: &tables,
        indexes: &[],
        domains: &[],
        guards: &global_guards,
        sequences: &[],
    };

    let unchanged_fragments = fragments(&unchanged);
    measure_canonical_semantics_v2(&unchanged.outer, &unchanged.outcome, &unchanged_fragments)
        .expect("unchanged A/B/A passes pass zero");
    close_canonical_semantics_v2_for_test(
        &unchanged.outer,
        &unchanged.outcome,
        &unchanged_fragments,
    )
    .expect("unchanged A/B/A closes codec evidence before guard validation");
    validate_canonical_semantics_v2_guards_for_test(
        &unchanged.outer,
        &unchanged.outcome,
        &unchanged_fragments,
        &catalog,
    )
    .expect("unchanged A/B/A accepts the independently assembled guard catalog");

    let swapped = q1_sabotage::q2_owner_swapped_guard_fixture();
    let swapped_fragments = fragments(&swapped);
    measure_canonical_semantics_v2(&swapped.outer, &swapped.outcome, &swapped_fragments)
        .expect("coherently swapped A/B/A still passes pass zero");
    close_canonical_semantics_v2_for_test(&swapped.outer, &swapped.outcome, &swapped_fragments)
        .expect("coherently swapped A/B/A remains codec-closed");
    let error = validate_canonical_semantics_v2_guards_for_test(
        &swapped.outer,
        &swapped.outcome,
        &swapped_fragments,
        &catalog,
    )
    .expect_err("unchanged catalog rejects the swapped static guard owner");
    assert!(
        error
            .to_string()
            .contains("pinned catalog guard is outside the exact S2/catalog guard closure"),
        "unexpected Q2 owner-swap rejection: {error}"
    );

    let retried = q1_vectors::successful_a_b_a_fixture();
    let retried_fragments = fragments(&retried);
    validate_canonical_semantics_v2_guards_for_test(
        &retried.outer,
        &retried.outcome,
        &retried_fragments,
        &catalog,
    )
    .expect("fresh unchanged codec owner accepts after hostile consuming rejection");
}

fn fragments(fixture: &q1_vectors::Q1Fixture) -> [gpu_db_wal::CanonicalFragmentRef<'_>; 2] {
    [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &fixture.fragment_body,
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        },
    ]
}

fn assert_a_b_a_target_columns(fixture: &q1_vectors::Q1Fixture) {
    let expected = [
        (PARENT_OID, PARENT_SCHEMA_DIGEST, 1_u32),
        (CHILD_OID, CHILD_SCHEMA_DIGEST, 5_u32),
        (PARENT_OID, PARENT_SCHEMA_DIGEST, 1_u32),
    ];
    let mut offset = 0_usize;
    for (record_ordinal, (oid, schema_digest, stable_column_id)) in expected.into_iter().enumerate()
    {
        let record_bytes = u32::from_le_bytes(
            fixture.sections[1][offset..offset + 4]
                .try_into()
                .expect("Q1 S2 record length"),
        ) as usize;
        let end = offset + 4 + record_bytes;
        let record = crate::typed_insert_batch::decode_canonical_typed_insert_record(
            &fixture.sections[1][offset + 4..end],
        )
        .expect("Q1 S2 source remains strict");
        let target = record.target_identity();
        let column = record
            .catalog_columns()
            .next()
            .expect("Q1 target has a first catalog column");
        assert_eq!(target.oid, oid, "Q1 record {record_ordinal} target OID");
        assert_eq!(
            target.schema_digest, schema_digest,
            "Q1 record {record_ordinal} schema"
        );
        assert_eq!(
            column.catalog_column_ordinal, 0,
            "Q1 guard source is column zero"
        );
        assert_eq!(
            column.column_id, stable_column_id,
            "Q1 guard column identity"
        );
        assert_eq!(column.attnum, 1, "Q1 guard column attnum");
        assert_eq!(column.name, "id", "Q1 guard source remains id");
        assert_eq!(storage(column.ty), [2, 0, 0, 0], "Q1 guard column storage");
        assert_eq!(column.type_oid, 23, "Q1 guard column type OID");
        assert_eq!(column.type_size, 4, "Q1 guard column type size");
        offset = end;
    }
    assert_eq!(
        offset,
        fixture.sections[1].len(),
        "A/B/A exhausts S2 records"
    );
}

fn catalog_column(
    stable_column_id: u32,
    column_shape_digest: [u8; 32],
    column_root: [u8; 32],
) -> SemanticsV2CatalogColumnWitness<'static> {
    SemanticsV2CatalogColumnWitness {
        catalog_column_ordinal: 0,
        stable_column_id,
        attnum: 1,
        name: "id",
        storage: [2, 0, 0, 0],
        declared_type_oid: 23,
        signed_type_size: 4,
        column_shape_digest,
        column_root,
    }
}

fn not_null_guard(
    stable_guard_id: u64,
    owner_stable_id: u64,
    owner_display_oid: u32,
    shape_digest: [u8; 32],
    program_or_descriptor_root: [u8; 32],
    catalog_generation: u64,
) -> SemanticsV2CatalogGuardWitness<'static> {
    SemanticsV2CatalogGuardWitness {
        kind: 9,
        stable_guard_id,
        display_oid: 0,
        schema: "",
        name: "",
        synthesized_not_null: true,
        owner_kind: 1,
        owner_stable_id,
        owner_display_oid,
        owner_catalog_column_ordinal: 0,
        domain_ordinal: ABSENT_U32,
        raw_constraint_ordinal: 0,
        source_ordinal: 0,
        shape_digest,
        program_or_descriptor_root,
        catalog_generation,
    }
}

fn storage(ty: SqlType) -> [u8; 4] {
    match ty {
        SqlType::Int2 => [1, 0, 0, 0],
        SqlType::Int4 => [2, 0, 0, 0],
        SqlType::Int8 => [3, 0, 0, 0],
        SqlType::Numeric { precision, scale } => [4, precision, scale, 0],
        SqlType::Bool => [5, 0, 0, 0],
        SqlType::Text => [6, 0, 0, 0],
        SqlType::Date => [7, 0, 0, 0],
        SqlType::Timestamp => [8, 0, 0, 0],
        SqlType::Uuid => [9, 0, 0, 0],
    }
}
