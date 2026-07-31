//! Independently pinned Q2 catalog and durable-allocator witness literals.
//!
//! These rows deliberately spell the reviewed catalog/lease authority rather than inspecting a
//! decoded S2 record.  The retained validation path is responsible for proving the literals
//! match the actual codec-closed owner.

use crate::typed_insert_aggregate::semantics_v2::retained::{
    AllocatorLeaseSpecForTest, SemanticsV2CatalogColumnWitness,
    SemanticsV2CatalogForeignKeyWitness, SemanticsV2CatalogGuardWitness,
    SemanticsV2CatalogIndexKeyWitness, SemanticsV2CatalogIndexWitness,
    SemanticsV2CatalogSequenceWitness, SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness,
};

const ABSENT_U32: u32 = u32::MAX;
const INT4_STORAGE: [u8; 4] = [2, 0, 0, 0];
const TEXT_STORAGE: [u8; 4] = [6, 0, 0, 0];
const DATABASE_ID: [u8; 16] = [0xa1; 16];
const CATALOG_DIGEST: [u8; 32] = [0x33; 32];

const MINIMAL_SCHEMA: [u8; 32] = [
    0xe3, 0x0d, 0xae, 0x61, 0x6b, 0xe5, 0xc3, 0xb6, 0x9d, 0x3a, 0x30, 0xe1, 0xf3, 0x15, 0x2d, 0xed,
    0x1b, 0x23, 0xf1, 0xe6, 0x9c, 0x85, 0x99, 0x4a, 0x18, 0xee, 0x54, 0x00, 0xeb, 0x8c, 0x14, 0x61,
];
const ABORT_SCHEMA: [u8; 32] = [
    0x94, 0x80, 0xde, 0xba, 0x8c, 0x96, 0x45, 0x6c, 0x16, 0xb7, 0x4d, 0x31, 0xbf, 0x1d, 0xe9, 0x52,
    0x3e, 0xc6, 0x20, 0xaa, 0x7f, 0x95, 0x98, 0xa2, 0x5c, 0xe7, 0xa7, 0xb0, 0xd6, 0x78, 0xa9, 0x39,
];
const PARENT_SCHEMA: [u8; 32] = [
    0x64, 0xb2, 0xbb, 0xb1, 0xf7, 0xba, 0x21, 0xaf, 0x78, 0xd7, 0xc0, 0x3d, 0x61, 0x7e, 0xe3, 0xf9,
    0x68, 0x28, 0x0e, 0xcc, 0xec, 0xf6, 0xe9, 0x32, 0x9a, 0xe6, 0x25, 0xba, 0x85, 0xe3, 0x27, 0x52,
];
const CHILD_SCHEMA: [u8; 32] = [
    0xc6, 0x3d, 0xf5, 0xde, 0x09, 0x40, 0xac, 0xd6, 0xa8, 0xa3, 0x27, 0x1c, 0x4b, 0xa0, 0x04, 0x8c,
    0xac, 0x9c, 0x82, 0x38, 0x0a, 0xdc, 0x81, 0x98, 0x96, 0x03, 0xc2, 0xca, 0x89, 0x37, 0xf1, 0xad,
];
const ID_NAME_DIGEST: [u8; 32] = [
    0xac, 0xc6, 0x30, 0x59, 0xd4, 0xd8, 0xcb, 0xac, 0x28, 0x7a, 0x45, 0xa5, 0x5d, 0x0f, 0xac, 0xf5,
    0x99, 0xeb, 0xba, 0x9c, 0x12, 0x8c, 0x7c, 0x60, 0xd4, 0x41, 0xcf, 0x69, 0xf9, 0x1b, 0x19, 0x7a,
];
const KEY_A_NAME_DIGEST: [u8; 32] = [
    0x63, 0x16, 0xea, 0x1e, 0xef, 0x95, 0xa6, 0xcc, 0x34, 0x58, 0xbd, 0x3a, 0x56, 0x8d, 0xb5, 0xa0,
    0x4f, 0x6d, 0xf1, 0x9d, 0x63, 0x5d, 0x44, 0xac, 0xeb, 0x82, 0x8b, 0xc8, 0x6d, 0x27, 0xa8, 0x98,
];
const KEY_B_NAME_DIGEST: [u8; 32] = [
    0x61, 0x1e, 0x46, 0x04, 0x9f, 0xb4, 0xe3, 0xe6, 0x7f, 0x17, 0x4b, 0x36, 0xa7, 0xe5, 0x8a, 0x7c,
    0x2f, 0xc6, 0x45, 0x67, 0xed, 0x3e, 0x99, 0x0a, 0x5d, 0x0a, 0xfd, 0xb2, 0x7b, 0x63, 0xf2, 0xef,
];
const EXPLICIT_SEQUENCE_DESCRIPTOR: [u8; 32] = [
    0xa2, 0x4d, 0x22, 0xea, 0x7b, 0xa6, 0x17, 0xd9, 0x5d, 0x65, 0x35, 0xeb, 0xee, 0x51, 0xbc, 0xdd,
    0x77, 0x52, 0x41, 0x45, 0x79, 0xc2, 0x58, 0x63, 0x49, 0xd4, 0x67, 0xea, 0xf2, 0xd9, 0xcf, 0x4f,
];
const SUCCESS_SEQUENCE_DESCRIPTOR: [u8; 32] = [
    0xaa, 0xed, 0x37, 0xab, 0xd2, 0x92, 0x4c, 0x6c, 0xa7, 0x96, 0x81, 0x39, 0xcd, 0xbe, 0x1d, 0x99,
    0x1e, 0xbd, 0x1d, 0x34, 0x49, 0xe8, 0xa0, 0xc8, 0x0f, 0xbf, 0x51, 0xe9, 0xab, 0xa9, 0x48, 0x79,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Q2WitnessCase {
    MinimalAbort,
    ExplicitAbort,
    SuccessfulInterleaved,
}

pub(super) fn with_witness<T>(
    case: Q2WitnessCase,
    operation: impl FnOnce(SemanticsV2CatalogWitness<'_>, &[AllocatorLeaseSpecForTest]) -> T,
) -> T {
    match case {
        Q2WitnessCase::MinimalAbort => with_minimal_abort(operation),
        Q2WitnessCase::ExplicitAbort => with_explicit_abort(operation),
        Q2WitnessCase::SuccessfulInterleaved => with_successful_interleaved(operation),
    }
}

fn with_minimal_abort<T>(
    operation: impl FnOnce(SemanticsV2CatalogWitness<'_>, &[AllocatorLeaseSpecForTest]) -> T,
) -> T {
    let columns = [column(
        0,
        1,
        1,
        "id",
        INT4_STORAGE,
        23,
        4,
        [0x23; 32],
        [0x24; 32],
    )];
    let guards = [not_null_guard(
        301, 0, 101, 16_384, 0, [0x23; 32], [0x24; 32], 12,
    )];
    let tables = [table(
        101,
        16_384,
        "codec_golden",
        MINIMAL_SCHEMA,
        11,
        [0x22; 32],
        &columns,
        &guards,
        &[],
    )];
    let leases = [lease(101, 100, 101)];
    operation(catalog(7, &tables, &[], &guards, &[]), &leases)
}

fn with_explicit_abort<T>(
    operation: impl FnOnce(SemanticsV2CatalogWitness<'_>, &[AllocatorLeaseSpecForTest]) -> T,
) -> T {
    let columns = [
        column(0, 1, 1, "id", INT4_STORAGE, 23, 4, [0x31; 32], [0x41; 32]),
        column(
            1,
            2,
            2,
            "required",
            INT4_STORAGE,
            23,
            4,
            [0x23; 32],
            [0x24; 32],
        ),
    ];
    let guards = [not_null_guard(
        1_901, 301, 1_101, 16_384, 1, [0x23; 32], [0x24; 32], 9,
    )];
    let tables = [table(
        1_101,
        16_384,
        "q1_abort",
        ABORT_SCHEMA,
        11,
        [0x22; 32],
        &columns,
        &guards,
        &[],
    )];
    let sequences = [sequence(
        1_701,
        16_385,
        "q1_abort_id_seq",
        EXPLICIT_SEQUENCE_DESCRIPTOR,
    )];
    let leases = [lease(1_101, 100, 102)];
    operation(catalog(17, &tables, &[], &guards, &sequences), &leases)
}

fn with_successful_interleaved<T>(
    operation: impl FnOnce(SemanticsV2CatalogWitness<'_>, &[AllocatorLeaseSpecForTest]) -> T,
) -> T {
    let parent_columns = [
        column(0, 1, 1, "id", INT4_STORAGE, 23, 4, [0x61; 32], [0x62; 32]),
        column(
            1,
            2,
            2,
            "key_a",
            INT4_STORAGE,
            23,
            4,
            [0x32; 32],
            [0x42; 32],
        ),
        column(
            2,
            3,
            3,
            "key_b",
            INT4_STORAGE,
            23,
            4,
            [0x33; 32],
            [0x43; 32],
        ),
        column(
            3,
            4,
            4,
            "note",
            TEXT_STORAGE,
            25,
            -1,
            [0x34; 32],
            [0x44; 32],
        ),
    ];
    let child_columns = [
        column(0, 5, 1, "id", INT4_STORAGE, 23, 4, [0x63; 32], [0x64; 32]),
        column(
            1,
            6,
            2,
            "parent_id",
            INT4_STORAGE,
            23,
            4,
            [0x36; 32],
            [0x46; 32],
        ),
        column(
            2,
            7,
            3,
            "key_a",
            INT4_STORAGE,
            23,
            4,
            [0x37; 32],
            [0x47; 32],
        ),
        column(
            3,
            8,
            4,
            "key_b",
            INT4_STORAGE,
            23,
            4,
            [0x38; 32],
            [0x48; 32],
        ),
        column(
            4,
            9,
            5,
            "note",
            TEXT_STORAGE,
            25,
            -1,
            [0x39; 32],
            [0x49; 32],
        ),
    ];
    let parent_guards = [not_null_guard(
        1_901, 0, 1_101, 16_384, 0, [0x61; 32], [0x62; 32], 11,
    )];
    let child_guards = [not_null_guard(
        1_902, 0, 1_102, 16_387, 0, [0x63; 32], [0x64; 32], 21,
    )];
    let all_guards = [
        not_null_guard(1_901, 0, 1_101, 16_384, 0, [0x61; 32], [0x62; 32], 11),
        not_null_guard(1_902, 0, 1_102, 16_387, 0, [0x63; 32], [0x64; 32], 21),
    ];
    let parent_pkey_keys = [index_key(0, 0, 1, 1, "id", ID_NAME_DIGEST)];
    let parent_unique_keys = [
        index_key(0, 1, 2, 2, "key_a", KEY_A_NAME_DIGEST),
        index_key(1, 2, 3, 3, "key_b", KEY_B_NAME_DIGEST),
    ];
    let child_pkey_keys = [index_key(0, 0, 5, 1, "id", ID_NAME_DIGEST)];
    let child_unique_keys = [
        index_key(0, 2, 7, 3, "key_a", KEY_A_NAME_DIGEST),
        index_key(1, 3, 8, 4, "key_b", KEY_B_NAME_DIGEST),
    ];
    let child_foreign_keys = [SemanticsV2CatalogForeignKeyWitness {
        stable_constraint_id: 6_101,
        display_oid: 16_391,
        raw_foreign_key_ordinal: 0,
        schema: "public",
        name: "q1_child_parent_fk",
        child_catalog_column_ordinal: 1,
        child_stable_column_id: 6,
        parent_stable_table_id: 1_101,
        parent_display_oid: 16_384,
        parent_catalog_column_ordinal: 0,
        parent_stable_column_id: 1,
        supporting_stable_index_id: 2_101,
    }];
    let tables = [
        table(
            1_101,
            16_384,
            "q1_parent",
            PARENT_SCHEMA,
            11,
            [0x22; 32],
            &parent_columns,
            &parent_guards,
            &[],
        ),
        table(
            1_102,
            16_387,
            "q1_child",
            CHILD_SCHEMA,
            21,
            [0x32; 32],
            &child_columns,
            &child_guards,
            &child_foreign_keys,
        ),
    ];
    let indexes = [
        index(
            2_101,
            16_385,
            1_101,
            16_384,
            "q1_parent",
            PARENT_SCHEMA,
            "q1_parent_pkey",
            2_101,
            16_385,
            11,
            0,
            11,
            [0x22; 32],
            1,
            [0x70; 32],
            &parent_pkey_keys,
        ),
        index(
            2_102,
            16_386,
            1_101,
            16_384,
            "q1_parent",
            PARENT_SCHEMA,
            "q1_parent_key_a_key",
            2_102,
            16_386,
            13,
            1,
            11,
            [0x22; 32],
            1,
            [0x71; 32],
            &parent_unique_keys,
        ),
        index(
            3_101,
            16_388,
            1_102,
            16_387,
            "q1_child",
            CHILD_SCHEMA,
            "q1_child_pkey",
            3_101,
            16_388,
            11,
            0,
            21,
            [0x32; 32],
            1,
            [0x72; 32],
            &child_pkey_keys,
        ),
        index(
            3_102,
            16_389,
            1_102,
            16_387,
            "q1_child",
            CHILD_SCHEMA,
            "q1_child_key_a_key",
            3_102,
            16_389,
            13,
            1,
            21,
            [0x32; 32],
            1,
            [0x73; 32],
            &child_unique_keys,
        ),
    ];
    let sequences = [sequence(
        1_701,
        16_390,
        "q1_child_id_seq",
        SUCCESS_SEQUENCE_DESCRIPTOR,
    )];
    let leases = [lease(1_101, 100, 102), lease(1_102, 200, 201)];
    operation(
        catalog(17, &tables, &indexes, &all_guards, &sequences),
        &leases,
    )
}

fn catalog<'a>(
    epoch: u64,
    tables: &'a [SemanticsV2CatalogTableWitness<'a>],
    indexes: &'a [SemanticsV2CatalogIndexWitness<'a>],
    guards: &'a [SemanticsV2CatalogGuardWitness<'a>],
    sequences: &'a [SemanticsV2CatalogSequenceWitness<'a>],
) -> SemanticsV2CatalogWitness<'a> {
    SemanticsV2CatalogWitness {
        database_id: DATABASE_ID,
        catalog_epoch: epoch,
        catalog_digest: CATALOG_DIGEST,
        tables,
        indexes,
        domains: &[],
        guards,
        sequences,
    }
}

#[allow(clippy::too_many_arguments)]
fn table<'a>(
    stable_table_id: u64,
    display_oid: u32,
    name: &'a str,
    schema_digest: [u8; 32],
    data_generation: u64,
    data_root: [u8; 32],
    catalog_columns: &'a [SemanticsV2CatalogColumnWitness<'a>],
    not_null_guards: &'a [SemanticsV2CatalogGuardWitness<'a>],
    foreign_keys: &'a [SemanticsV2CatalogForeignKeyWitness<'a>],
) -> SemanticsV2CatalogTableWitness<'a> {
    SemanticsV2CatalogTableWitness {
        stable_table_id,
        display_oid,
        schema: "public",
        name,
        schema_digest,
        data_generation,
        data_root,
        catalog_columns,
        not_null_guards,
        check_guards: &[],
        foreign_keys,
    }
}

#[allow(clippy::too_many_arguments)]
fn column<'a>(
    catalog_column_ordinal: u32,
    stable_column_id: u32,
    attnum: i16,
    name: &'a str,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
    column_shape_digest: [u8; 32],
    column_root: [u8; 32],
) -> SemanticsV2CatalogColumnWitness<'a> {
    SemanticsV2CatalogColumnWitness {
        catalog_column_ordinal,
        stable_column_id,
        attnum,
        name,
        storage,
        declared_type_oid,
        signed_type_size,
        column_shape_digest,
        column_root,
    }
}

#[allow(clippy::too_many_arguments)]
fn not_null_guard<'a>(
    stable_guard_id: u64,
    display_oid: u32,
    owner_stable_id: u64,
    owner_display_oid: u32,
    owner_catalog_column_ordinal: u32,
    shape_digest: [u8; 32],
    program_or_descriptor_root: [u8; 32],
    catalog_generation: u64,
) -> SemanticsV2CatalogGuardWitness<'a> {
    SemanticsV2CatalogGuardWitness {
        kind: 9,
        stable_guard_id,
        display_oid,
        schema: "",
        name: "",
        synthesized_not_null: true,
        owner_kind: 1,
        owner_stable_id,
        owner_display_oid,
        owner_catalog_column_ordinal,
        domain_ordinal: ABSENT_U32,
        raw_constraint_ordinal: 0,
        source_ordinal: owner_catalog_column_ordinal,
        shape_digest,
        program_or_descriptor_root,
        catalog_generation,
    }
}

fn index_key<'a>(
    key_ordinal: u32,
    owner_catalog_column_ordinal: u32,
    stable_column_id: u32,
    attnum: i16,
    name: &'a str,
    column_name_digest: [u8; 32],
) -> SemanticsV2CatalogIndexKeyWitness<'a> {
    SemanticsV2CatalogIndexKeyWitness {
        key_ordinal,
        owner_catalog_column_ordinal,
        stable_column_id,
        attnum,
        name,
        storage: INT4_STORAGE,
        declared_type_oid: 23,
        signed_type_size: 4,
        column_name_digest,
    }
}

#[allow(clippy::too_many_arguments)]
fn index<'a>(
    stable_index_id: u64,
    display_oid: u32,
    owner_stable_table_id: u64,
    owner_display_oid: u32,
    owner_name: &'a str,
    schema_digest: [u8; 32],
    name: &'a str,
    constraint_stable_id: u64,
    constraint_display_oid: u32,
    index_flags: u32,
    raw_catalog_ordinal: u32,
    _owner_data_generation: u64,
    _owner_table_root: [u8; 32],
    base_generation: u64,
    base_root: [u8; 32],
    key_columns: &'a [SemanticsV2CatalogIndexKeyWitness<'a>],
) -> SemanticsV2CatalogIndexWitness<'a> {
    SemanticsV2CatalogIndexWitness {
        stable_index_id,
        display_oid,
        owner_stable_table_id,
        owner_display_oid,
        schema: "public",
        name,
        owner_schema: "public",
        owner_name,
        constraint_stable_id,
        constraint_display_oid,
        constraint_schema: "public",
        constraint_name: name,
        schema_digest,
        index_flags,
        null_equality_policy: 1,
        raw_catalog_ordinal,
        key_columns,
        catalog_epoch: 17,
        base_generation,
        base_root,
    }
}

fn sequence<'a>(
    stable_sequence_id: u64,
    display_oid: u32,
    name: &'a str,
    descriptor_digest: [u8; 32],
) -> SemanticsV2CatalogSequenceWitness<'a> {
    SemanticsV2CatalogSequenceWitness {
        stable_sequence_id,
        display_oid,
        schema: "public",
        name,
        catalog_generation: 1,
        descriptor_digest,
    }
}

fn lease(stable_allocator_id: u64, lease_start: u64, lease_end: u64) -> AllocatorLeaseSpecForTest {
    AllocatorLeaseSpecForTest {
        stable_allocator_id,
        lease_start,
        lease_end,
    }
}
