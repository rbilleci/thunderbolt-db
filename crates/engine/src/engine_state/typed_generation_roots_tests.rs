use super::*;

type Digest = gpu_db_wal::CanonicalDigest;

fn gpu_root(seed: u64) -> Digest {
    let mut root = [0; 32];
    root[..8].copy_from_slice(&seed.to_le_bytes());
    root[31] = 0xa5;
    root
}

fn empty_roots(seed: u64) -> [Digest; 65] {
    std::array::from_fn(|depth| gpu_root(seed + depth as u64))
}

fn path_roots(seed: u64) -> [Digest; 64] {
    std::array::from_fn(|depth| gpu_root(seed + depth as u64))
}

fn completion(
    empty: [Digest; 65],
    initial_leaf: Digest,
    initial_path: [Digest; 64],
    final_leaf: Digest,
    final_path: [Digest; 64],
) -> TypedTableMapGpuCompletion {
    TypedTableMapGpuCompletion::from_completed_roots(
        empty,
        initial_leaf,
        initial_path,
        final_leaf,
        final_path,
    )
    .expect("synthetic GPU roots are nonzero")
}

fn root(generation: u64, seed: u64, rows: u64) -> TypedTableGenerationRoot {
    TypedTableGenerationRoot {
        data_generation: generation,
        table_root: gpu_root(seed),
        logical_row_count: rows,
    }
}

fn columns(seed: u64) -> Vec<TypedColumnGenerationRoot> {
    vec![
        TypedColumnGenerationRoot {
            catalog_column_ordinal: 0,
            stable_column_id: 101,
            attnum: 1,
            column_shape_root: gpu_root(seed),
            column_root: gpu_root(seed + 1),
        },
        TypedColumnGenerationRoot {
            catalog_column_ordinal: 1,
            stable_column_id: 102,
            attnum: 2,
            column_shape_root: gpu_root(seed + 2),
            column_root: gpu_root(seed + 3),
        },
    ]
}

fn index(stable_index_id: u64, generation: u64, seed: u64) -> TypedIndexGenerationRoot {
    TypedIndexGenerationRoot {
        stable_index_id,
        index_generation: generation,
        index_root: gpu_root(seed),
    }
}

struct TwoTableFixture {
    snapshot: TypedGenerationRootSnapshot,
    empty: [Digest; 65],
    first_table: TypedTableGenerationRoot,
    first_columns: Vec<TypedColumnGenerationRoot>,
    first_final_leaf: Digest,
    first_final_path: [Digest; 64],
    second_table: TypedTableGenerationRoot,
    second_final_path: [Digest; 64],
    second_database_root: Digest,
}

fn two_table_fixture() -> TwoTableFixture {
    let empty = empty_roots(10);
    let first_final_leaf = gpu_root(1_000);
    let first_final_path = path_roots(1_100);
    let first_table = root(1, 1_200, 0);
    let first_columns = columns(1_300);
    let first_database_root = gpu_root(1_400);
    let first = TypedGenerationRootSnapshot::default()
        .with_gpu_completed_table_map_substitution(
            1,
            None,
            false,
            None,
            first_table,
            &first_columns,
            &[],
            &[],
            first_database_root,
            &completion(
                empty,
                empty[64],
                std::array::from_fn(|depth| empty[depth]),
                first_final_leaf,
                first_final_path,
            ),
        )
        .expect("first CREATE accepts the GPU empty-map witness");

    let second_table = root(1, 1_500, 0);
    let second_columns = columns(1_600);
    let second_final_path = path_roots(1_700);
    let second_database_root = gpu_root(1_800);
    let second_initial_path = std::array::from_fn(|depth| {
        if depth == 0 {
            first_final_path[0]
        } else {
            empty[depth]
        }
    });
    let snapshot = first
        .with_gpu_completed_table_map_substitution(
            1_u64 << 63,
            None,
            false,
            Some(first_database_root),
            second_table,
            &second_columns,
            &[],
            &[],
            second_database_root,
            &completion(
                empty,
                empty[64],
                second_initial_path,
                gpu_root(1_900),
                second_final_path,
            ),
        )
        .expect("second CREATE preserves the first table-map leaf");

    TwoTableFixture {
        snapshot,
        empty,
        first_table,
        first_columns,
        first_final_leaf,
        first_final_path,
        second_table,
        second_final_path,
        second_database_root,
    }
}

#[test]
fn typed_root_map_first_create_uses_gpu_empty_map_and_initializes_once() {
    let empty = empty_roots(3);
    let table = root(1, 100, 0);
    let table_columns = columns(200);
    let final_path = path_roots(300);
    let database_root = gpu_root(400);
    let snapshot = TypedGenerationRootSnapshot::default()
        .with_gpu_completed_table_map_substitution(
            1,
            None,
            false,
            None,
            table,
            &table_columns,
            &[],
            &[],
            database_root,
            &completion(
                empty,
                empty[64],
                std::array::from_fn(|depth| empty[depth]),
                gpu_root(500),
                final_path,
            ),
        )
        .expect("first CREATE accepts the sole uninitialized GPU map");

    assert_eq!(snapshot.database_root, Some(database_root));
    assert_eq!(snapshot.table(1), Some(table));
    assert_eq!(snapshot.table_columns(1).collect::<Vec<_>>(), table_columns);
    assert_eq!(snapshot.table_map_root(), Some(final_path[0]));
    assert_eq!(
        snapshot
            .table_map
            .as_ref()
            .expect("CREATE initialized the map")
            .count(),
        1,
        "the count is derived from retained child structure, never supplied by the GPU",
    );
    assert_eq!(
        snapshot
            .table_map_sibling_roots(1)
            .expect("well-formed map")
            .expect("initialized map"),
        std::array::from_fn(|depth| empty[depth + 1]),
    );
}

#[test]
fn typed_root_map_first_create_accepts_all_catalog_ordered_same_generation_index_roots() {
    let empty = empty_roots(600);
    let table = root(7, 700, 0);
    let table_columns = columns(800);
    let inline_index = index(17, table.data_generation, 900);
    let final_path = path_roots(1_000);
    let database_root = gpu_root(1_100);
    let gpu_completion = completion(
        empty,
        empty[64],
        std::array::from_fn(|depth| empty[depth]),
        gpu_root(1_200),
        final_path,
    );
    let snapshot = TypedGenerationRootSnapshot::default()
        .with_gpu_completed_table_map_substitution(
            1,
            None,
            false,
            None,
            table,
            &table_columns,
            &[],
            std::slice::from_ref(&inline_index),
            database_root,
            &gpu_completion,
        )
        .expect("CREATE accepts its one GPU-produced inline index root");

    assert_eq!(
        snapshot.table_index_roots(1).collect::<Vec<_>>(),
        vec![inline_index]
    );

    let wrong_generation = index(17, table.data_generation - 1, 1_300);
    assert!(TypedGenerationRootSnapshot::default()
        .with_gpu_completed_table_map_substitution(
            1,
            None,
            false,
            None,
            table,
            &table_columns,
            &[],
            std::slice::from_ref(&wrong_generation),
            database_root,
            &gpu_completion,
        )
        .is_err());

    let extra_index = index(18, table.data_generation, 1_400);
    let multiple = TypedGenerationRootSnapshot::default()
        .with_gpu_completed_table_map_substitution(
            1,
            None,
            false,
            None,
            table,
            &table_columns,
            &[],
            &[inline_index, extra_index],
            database_root,
            &gpu_completion,
        )
        .expect("CREATE accepts every same-generation catalog-ordered index root");
    assert_eq!(
        multiple.table_index_roots(1).collect::<Vec<_>>(),
        vec![inline_index, extra_index]
    );
}

#[test]
fn typed_root_map_second_create_preserves_first_leaf_and_requires_current_database_root() {
    let fixture = two_table_fixture();
    assert_eq!(
        fixture.snapshot.table(1_u64 << 63),
        Some(fixture.second_table)
    );
    assert_eq!(fixture.snapshot.table(1), Some(fixture.first_table));

    let missing_database_predecessor = fixture.snapshot.with_gpu_completed_table_map_substitution(
        1_u64 << 62,
        None,
        false,
        None,
        root(1, 2_000, 0),
        &columns(2_010),
        &[],
        &[],
        gpu_root(2_020),
        &completion(
            fixture.empty,
            fixture.empty[64],
            std::array::from_fn(|depth| {
                if depth == 0 {
                    fixture.second_final_path[0]
                } else {
                    fixture.empty[depth]
                }
            }),
            gpu_root(2_030),
            path_roots(2_040),
        ),
    );
    assert!(missing_database_predecessor.is_err());
}

#[test]
fn typed_root_map_a_to_b_to_a_relinks_only_the_changed_path() {
    let fixture = two_table_fixture();
    let successor = root(2, 2_100, 1);
    let mut successor_columns = fixture.first_columns.clone();
    for (ordinal, column) in successor_columns.iter_mut().enumerate() {
        column.column_root = gpu_root(2_200 + ordinal as u64);
    }
    let initial_path = std::array::from_fn(|depth| {
        if depth == 0 {
            fixture.second_final_path[0]
        } else {
            fixture.first_final_path[depth]
        }
    });
    let final_path = path_roots(2_300);
    let final_database_root = gpu_root(2_400);
    let after = fixture
        .snapshot
        .with_gpu_completed_table_map_substitution(
            1,
            Some(fixture.first_table),
            false,
            Some(fixture.second_database_root),
            successor,
            &successor_columns,
            &[],
            &[],
            final_database_root,
            &completion(
                fixture.empty,
                fixture.first_final_leaf,
                initial_path,
                gpu_root(2_500),
                final_path,
            ),
        )
        .expect("A -> B -> A follows the current root and preserves B");

    assert_eq!(after.database_root, Some(final_database_root));
    assert_eq!(after.table(1), Some(successor));
    assert_eq!(after.table(1_u64 << 63), Some(fixture.second_table));
    assert_eq!(
        after.table_columns(1).collect::<Vec<_>>(),
        successor_columns
    );
    assert_eq!(
        after
            .table_map
            .as_ref()
            .expect("table map remains initialized")
            .count(),
        2,
    );
    let expected_siblings = std::array::from_fn(|depth| {
        if depth == 0 {
            fixture.second_final_path[1]
        } else {
            fixture.empty[depth + 1]
        }
    });
    assert_eq!(
        after
            .table_map_sibling_roots(1)
            .expect("well-formed map")
            .expect("initialized map"),
        expected_siblings,
        "the GPU predecessor witness is root-to-leaf ordered and retains B at depth zero",
    );
}

#[test]
fn typed_root_map_rejects_wrong_initial_path_and_nonempty_empty_root() {
    let fixture = two_table_fixture();
    let wrong_path = std::array::from_fn(|depth| {
        if depth == 0 {
            gpu_root(9_000)
        } else {
            fixture.empty[depth]
        }
    });
    let wrong_path_result = fixture.snapshot.with_gpu_completed_table_map_substitution(
        1_u64 << 62,
        None,
        false,
        Some(fixture.second_database_root),
        root(1, 9_010, 0),
        &columns(9_020),
        &[],
        &[],
        gpu_root(9_030),
        &completion(
            fixture.empty,
            fixture.empty[64],
            wrong_path,
            gpu_root(9_040),
            path_roots(9_050),
        ),
    );
    assert!(wrong_path_result.is_err());

    let correct_initial_path = std::array::from_fn(|depth| {
        if depth == 0 {
            fixture.second_final_path[0]
        } else {
            fixture.empty[depth]
        }
    });
    let mut impossible_final_path = path_roots(9_100);
    impossible_final_path[0] = fixture.empty[0];
    let wrong_count_root_result = fixture.snapshot.with_gpu_completed_table_map_substitution(
        1_u64 << 62,
        None,
        false,
        Some(fixture.second_database_root),
        root(1, 9_110, 0),
        &columns(9_120),
        &[],
        &[],
        gpu_root(9_130),
        &completion(
            fixture.empty,
            fixture.empty[64],
            correct_initial_path,
            gpu_root(9_140),
            impossible_final_path,
        ),
    );
    assert!(wrong_count_root_result.is_err());
}

#[test]
fn typed_root_map_rejects_duplicate_or_reordered_column_leafs() {
    let empty = empty_roots(10_000);
    let valid_completion = completion(
        empty,
        empty[64],
        std::array::from_fn(|depth| empty[depth]),
        gpu_root(10_100),
        path_roots(10_110),
    );
    let mut reordered = columns(10_200);
    reordered.swap(0, 1);
    assert!(TypedGenerationRootSnapshot::default()
        .with_gpu_completed_table_map_substitution(
            1,
            None,
            false,
            None,
            root(1, 10_210, 0),
            &reordered,
            &[],
            &[],
            gpu_root(10_220),
            &valid_completion,
        )
        .is_err());

    let mut duplicate = columns(10_300);
    duplicate[1].stable_column_id = duplicate[0].stable_column_id;
    assert!(TypedGenerationRootSnapshot::default()
        .with_gpu_completed_table_map_substitution(
            1,
            None,
            false,
            None,
            root(1, 10_310, 0),
            &duplicate,
            &[],
            &[],
            gpu_root(10_320),
            &valid_completion,
        )
        .is_err());
}

#[test]
fn typed_root_map_enrolls_an_index_then_evolves_it_without_changing_other_tables() {
    let fixture = two_table_fixture();
    let base_index = index(17, 1, 11_000);
    let enrollment_successor = root(2, 11_005, 0);
    let enrollment_final_path = path_roots(11_010);
    let enrollment_initial_path = std::array::from_fn(|depth| {
        if depth == 0 {
            fixture.second_final_path[0]
        } else {
            fixture.first_final_path[depth]
        }
    });
    let enrollment_database_root = gpu_root(11_020);
    let enrolled = fixture
        .snapshot
        .with_gpu_completed_index_root_enrollment(
            1,
            fixture.first_table,
            enrollment_successor,
            fixture.second_database_root,
            &[],
            base_index,
            enrollment_database_root,
            &completion(
                fixture.empty,
                fixture.first_final_leaf,
                enrollment_initial_path,
                gpu_root(11_030),
                enrollment_final_path,
            ),
        )
        .expect("CREATE INDEX enrolls the first GPU base root");

    assert_eq!(enrolled.table(1), Some(enrollment_successor));
    assert_eq!(
        enrolled.table_columns(1).collect::<Vec<_>>(),
        fixture.first_columns
    );
    assert_eq!(
        enrolled.table_index_roots(1).collect::<Vec<_>>(),
        vec![base_index]
    );
    assert_eq!(enrolled.table_index_root(1, 17), Some(base_index));
    assert_eq!(enrolled.table_index_root(1, 18), None);
    assert_eq!(
        enrolled.table(1_u64 << 63),
        Some(fixture.second_table),
        "index enrollment relinks only the target table-map path",
    );
    assert!(enrolled.table_index_roots(1_u64 << 63).next().is_none());

    let successor = root(3, 11_100, 1);
    let mut successor_columns = fixture.first_columns.clone();
    for (ordinal, column) in successor_columns.iter_mut().enumerate() {
        column.column_root = gpu_root(11_110 + ordinal as u64);
    }
    let successor_index = index(17, 2, 11_120);
    let after = enrolled
        .with_gpu_completed_table_map_substitution(
            1,
            Some(enrollment_successor),
            false,
            Some(enrollment_database_root),
            successor,
            &successor_columns,
            &[base_index],
            &[successor_index],
            gpu_root(11_130),
            &completion(
                fixture.empty,
                gpu_root(11_030),
                enrollment_final_path,
                gpu_root(11_140),
                path_roots(11_150),
            ),
        )
        .expect("indexed INSERT evolves the exact enrolled index root");

    assert_eq!(after.table(1), Some(successor));
    assert_eq!(
        after.table_index_roots(1).collect::<Vec<_>>(),
        vec![successor_index]
    );
    assert_eq!(after.table_index_root(1, 17), Some(successor_index));
    assert_eq!(after.table(1_u64 << 63), Some(fixture.second_table));
    assert!(after.table_index_roots(1_u64 << 63).next().is_none());
}

#[test]
fn typed_root_map_rejects_duplicate_reordered_or_stale_index_roots() {
    let fixture = two_table_fixture();
    let first_index = index(17, 1, 12_000);
    let first_enrollment_successor = root(2, 12_005, 0);
    let first_final_path = path_roots(12_010);
    let first_initial_path = std::array::from_fn(|depth| {
        if depth == 0 {
            fixture.second_final_path[0]
        } else {
            fixture.first_final_path[depth]
        }
    });
    let first_database_root = gpu_root(12_020);
    let once = fixture
        .snapshot
        .with_gpu_completed_index_root_enrollment(
            1,
            fixture.first_table,
            first_enrollment_successor,
            fixture.second_database_root,
            &[],
            first_index,
            first_database_root,
            &completion(
                fixture.empty,
                fixture.first_final_leaf,
                first_initial_path,
                gpu_root(12_030),
                first_final_path,
            ),
        )
        .expect("first index enrollment succeeds");
    let second_index = index(19, 1, 12_040);
    let second_enrollment_successor = root(3, 12_045, 0);
    let second_final_path = path_roots(12_050);
    let second_database_root = gpu_root(12_060);
    let twice = once
        .with_gpu_completed_index_root_enrollment(
            1,
            first_enrollment_successor,
            second_enrollment_successor,
            first_database_root,
            &[first_index],
            second_index,
            second_database_root,
            &completion(
                fixture.empty,
                gpu_root(12_030),
                first_final_path,
                gpu_root(12_070),
                second_final_path,
            ),
        )
        .expect("catalog-later second index enrollment succeeds");

    let successor = root(4, 12_080, 1);
    let mut successor_columns = fixture.first_columns.clone();
    for (ordinal, column) in successor_columns.iter_mut().enumerate() {
        column.column_root = gpu_root(12_090 + ordinal as u64);
    }
    let predecessor_index_roots = [first_index, second_index];
    let completion = TypedTableMapGpuCompletion::synthetic_first_create_for_test(12_100);
    let stale = twice.with_gpu_completed_table_map_substitution(
        1,
        Some(second_enrollment_successor),
        false,
        Some(second_database_root),
        successor,
        &successor_columns,
        &predecessor_index_roots,
        &[first_index, index(19, 2, 12_110)],
        gpu_root(12_120),
        &completion,
    );
    assert!(
        stale.is_err(),
        "an indexed INSERT cannot retain a stale root"
    );

    let duplicate = twice.with_gpu_completed_table_map_substitution(
        1,
        Some(second_enrollment_successor),
        false,
        Some(second_database_root),
        successor,
        &successor_columns,
        &predecessor_index_roots,
        &[index(17, 2, 12_130), index(17, 3, 12_140)],
        gpu_root(12_150),
        &completion,
    );
    assert!(
        duplicate.is_err(),
        "index stable identities must remain unique"
    );

    let reordered = twice.with_gpu_completed_table_map_substitution(
        1,
        Some(second_enrollment_successor),
        false,
        Some(second_database_root),
        successor,
        &successor_columns,
        &predecessor_index_roots,
        &[index(19, 2, 12_160), index(17, 2, 12_170)],
        gpu_root(12_180),
        &completion,
    );
    assert!(
        reordered.is_err(),
        "index roots must retain catalog raw-ID order"
    );
}
