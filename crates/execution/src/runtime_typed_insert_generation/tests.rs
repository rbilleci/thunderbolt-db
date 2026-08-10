use super::*;
use crate::CudaDriverRuntime;
use sha2::{Digest, Sha256};

#[derive(Clone)]
struct Cell {
    ordinal: u32,
    column: u32,
    attnum: i16,
    storage: [u8; 4],
    oid: u32,
    size: i16,
    null: bool,
    value: Vec<u8>,
}

fn begin(domain: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest
}

fn finish(digest: Sha256) -> [u8; 32] {
    digest.finalize().into()
}

fn shape(table: u64, cell: &Cell) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/column-shape/v1");
    digest.update(table.to_le_bytes());
    digest.update((cell.column as u64).to_le_bytes());
    digest.update(cell.attnum.to_le_bytes());
    digest.update(cell.storage);
    digest.update(cell.oid.to_le_bytes());
    digest.update(cell.size.to_le_bytes());
    finish(digest)
}

fn typed(shape: [u8; 32], cell: &Cell) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/typed-value/v1");
    digest.update(shape);
    digest.update([u8::from(cell.null)]);
    digest.update((cell.value.len() as u32).to_le_bytes());
    digest.update(&cell.value);
    finish(digest)
}

fn column_manifest(table: u64, columns: &[Cell]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/column-manifest/v1");
    digest.update(table.to_le_bytes());
    digest.update((columns.len() as u32).to_le_bytes());
    for column in columns {
        let shape = shape(table, column);
        digest.update(shape);
        digest.update(empty_column(table, shape));
    }
    finish(digest)
}

fn empty_column(table: u64, shape: [u8; 32]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/column-empty/v1");
    digest.update(table.to_le_bytes());
    digest.update(shape);
    digest.update(0_u64.to_le_bytes());
    finish(digest)
}

// The live kernel deliberately uses its v2 domains below. These independent host-side vectors
// prove the general descriptor grammar and its data-parallel batch reduction.
fn v2_empty_row(table: u64) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/row-empty/v2");
    digest.update(1_u16.to_le_bytes());
    digest.update(table.to_le_bytes());
    finish(digest)
}

fn v2_current_row(
    table: u64,
    row: u64,
    generation: u64,
    source_row: u32,
    cells: &[Cell],
) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/current-row/v2");
    digest.update(table.to_le_bytes());
    digest.update(row.to_le_bytes());
    digest.update(generation.to_le_bytes());
    digest.update(0_u32.to_le_bytes());
    digest.update(source_row.to_le_bytes());
    digest.update((cells.len() as u32).to_le_bytes());
    for cell in cells {
        let shape = shape(table, cell);
        digest.update((cell.column as u64).to_le_bytes());
        digest.update(shape);
        digest.update(typed(shape, cell));
    }
    finish(digest)
}

fn s7_final_row(table: u64, row: u64, row_ordinal: u32, cells: &[Cell]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/write001/s7-final-row/v2");
    digest.update(table.to_le_bytes());
    digest.update(row.to_le_bytes());
    digest.update(0_u32.to_le_bytes());
    digest.update(row_ordinal.to_le_bytes());
    digest.update((cells.len() as u32).to_le_bytes());
    for (ordinal, cell) in cells.iter().enumerate() {
        digest.update((ordinal as u32).to_le_bytes());
        digest.update(cell.column.to_le_bytes());
        digest.update(cell.attnum.to_le_bytes());
        digest.update(cell.storage);
        digest.update(cell.oid.to_le_bytes());
        digest.update(cell.size.to_le_bytes());
        digest.update([u8::from(cell.null)]);
        digest.update((cell.value.len() as u32).to_le_bytes());
        digest.update(&cell.value);
    }
    finish(digest)
}

fn s7_transition(
    row_id: u64,
    row_ordinal: u32,
    typed_statement_digest: [u8; 32],
    final_row_digest: [u8; 32],
) -> [u8; 32] {
    let mut raw = [0_u8; 192];
    raw[0..4].copy_from_slice(&row_ordinal.to_le_bytes());
    raw[8..16].copy_from_slice(&row_id.to_le_bytes());
    raw[16] = 1;
    raw[20..24].copy_from_slice(&row_ordinal.to_le_bytes());
    raw[28..32].copy_from_slice(&row_ordinal.to_le_bytes());
    raw[36..40].copy_from_slice(&row_ordinal.to_le_bytes());
    raw[64..96].copy_from_slice(&typed_statement_digest);
    raw[96..128].copy_from_slice(&final_row_digest);
    let mut digest = begin(b"gpu-db/write001/s7-transition/v2");
    digest.update(raw);
    finish(digest)
}

fn v2_batch_row_root(table: u64, generation: u64, rows: &[(u64, Vec<Cell>)]) -> [u8; 32] {
    let empty = v2_empty_row(table);
    let mut current: Vec<(u64, [u8; 32])> = rows
        .iter()
        .enumerate()
        .map(|(source_row, (id, cells))| {
            let mut leaf = begin(b"gpu-db/runtime-generation/row-leaf/v2");
            leaf.update(1_u16.to_le_bytes());
            leaf.update(table.to_le_bytes());
            leaf.update(id.to_le_bytes());
            leaf.update(1_u64.to_le_bytes());
            leaf.update(v2_current_row(
                table,
                *id,
                generation,
                source_row as u32,
                cells,
            ));
            (1, finish(leaf))
        })
        .collect();
    let mut level = 0_u32;
    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len().div_ceil(2));
        for pair in current.chunks(2) {
            let (left_count, left_digest) = pair[0];
            let (right_count, right_digest) = pair.get(1).copied().unwrap_or((0, empty));
            let mut node = begin(b"gpu-db/runtime-generation/batch-row-node/v2");
            node.update(1_u16.to_le_bytes());
            node.update(table.to_le_bytes());
            node.update(generation.to_le_bytes());
            node.update(level.to_le_bytes());
            node.update(left_count.to_le_bytes());
            node.update(right_count.to_le_bytes());
            node.update(left_digest);
            node.update(right_digest);
            next.push((left_count + right_count, finish(node)));
        }
        current = next;
        level += 1;
    }
    current[0].1
}

fn v2_table_genesis(table: u64, base_generation: u64, columns: &[Cell]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/table-root/v2");
    digest.update(1_u16.to_le_bytes());
    digest.update(table.to_le_bytes());
    digest.update(base_generation.to_le_bytes());
    digest.update(0_u64.to_le_bytes());
    digest.update(v2_empty_row(table));
    digest.update((columns.len() as u32).to_le_bytes());
    digest.update(column_manifest(table, columns));
    finish(digest)
}

// The protocol preimage is intentionally written as its exact ordered eight-field form.
#[allow(clippy::too_many_arguments)]
fn v2_table_successor(
    table: u64,
    base_generation: u64,
    commit_sequence: u64,
    initial_rows: u64,
    final_rows: u64,
    initial_root: [u8; 32],
    batch_root: [u8; 32],
    columns: &[Cell],
) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/table-successor/v2");
    digest.update(1_u16.to_le_bytes());
    digest.update(table.to_le_bytes());
    digest.update(base_generation.to_le_bytes());
    digest.update(commit_sequence.to_le_bytes());
    digest.update(initial_rows.to_le_bytes());
    digest.update(final_rows.to_le_bytes());
    digest.update(initial_root);
    digest.update(batch_root);
    digest.update((columns.len() as u32).to_le_bytes());
    digest.update(column_manifest(table, columns));
    finish(digest)
}

fn table_map_empty_roots(database: [u8; 16]) -> [[u8; 32]; 65] {
    let mut roots = [[0_u8; 32]; 65];
    let mut leaf = begin(b"gpu-db/runtime-generation/map-empty-leaf/v1");
    leaf.update(1_u16.to_le_bytes());
    leaf.update(database);
    roots[64] = finish(leaf);
    for depth in (0..64).rev() {
        let mut node = begin(b"gpu-db/runtime-generation/map-empty-node/v1");
        node.update(1_u16.to_le_bytes());
        node.update(database);
        node.update([depth as u8]);
        node.update(roots[depth + 1]);
        node.update(roots[depth + 1]);
        roots[depth] = finish(node);
    }
    roots
}

fn table_map_leaf(database: [u8; 16], table: u64, table_root: [u8; 32]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/map-leaf/v1");
    digest.update(1_u16.to_le_bytes());
    digest.update(database);
    digest.update(table.to_le_bytes());
    digest.update(table_root);
    finish(digest)
}

fn table_map_path(
    database: [u8; 16],
    table: u64,
    leaf: [u8; 32],
    siblings: [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH],
) -> [[u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH] {
    let mut path = [[0_u8; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH];
    let empty = table_map_empty_roots(database);
    let mut child = leaf;
    for depth in (0..RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH).rev() {
        if child == empty[depth + 1] && siblings[depth] == empty[depth + 1] {
            path[depth] = empty[depth];
            child = path[depth];
            continue;
        }
        let mut node = begin(b"gpu-db/runtime-generation/map-node/v1");
        node.update(1_u16.to_le_bytes());
        node.update(database);
        node.update([depth as u8]);
        if ((table >> (63 - depth)) & 1) == 0 {
            node.update(child);
            node.update(siblings[depth]);
        } else {
            node.update(siblings[depth]);
            node.update(child);
        }
        path[depth] = finish(node);
        child = path[depth];
    }
    path
}

fn database_root(database: [u8; 16], table_map_root: [u8; 32]) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/runtime-generation/database-root/v1");
    digest.update(1_u16.to_le_bytes());
    digest.update(database);
    digest.update(table_map_root);
    finish(digest)
}

fn one_table_map_predecessor(
    database: [u8; 16],
    table: u64,
    table_root: [u8; 32],
) -> RuntimeTypedInsertGenerationTableMapPredecessor {
    let empty = table_map_empty_roots(database);
    let siblings = std::array::from_fn(|depth| empty[depth + 1]);
    let path = table_map_path(
        database,
        table,
        table_map_leaf(database, table, table_root),
        siblings,
    );
    RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
        initial_database_root: database_root(database, path[0]),
        sibling_roots: siblings,
    }
}

fn one_table_map_retained_predecessor(
    database: [u8; 16],
    table: u64,
    table_root: [u8; 32],
) -> RuntimeTypedInsertGenerationTableMapPredecessor {
    let empty_roots = table_map_empty_roots(database);
    let sibling_roots = std::array::from_fn(|depth| empty_roots[depth + 1]);
    let initial_leaf_root = table_map_leaf(database, table, table_root);
    let initial_path_roots = table_map_path(database, table, initial_leaf_root, sibling_roots);
    RuntimeTypedInsertGenerationTableMapPredecessor::PinnedRetained {
        initial_database_root: database_root(database, initial_path_roots[0]),
        sibling_roots,
        empty_roots,
        initial_leaf_root,
        initial_path_roots,
    }
}

fn v2_generation_input(
    identity: RuntimeTypedInsertGenerationIdentity,
    table: RuntimeTypedInsertGenerationTable,
    rows: usize,
    batch_root: [u8; 32],
) -> [u8; 32] {
    let mut digest = begin(b"gpu-db/write001/generation-input/v4");
    digest.update(identity.database_id);
    digest.update(identity.catalog_epoch.to_le_bytes());
    digest.update(identity.catalog_digest);
    digest.update(identity.stable_transaction_id.to_le_bytes());
    digest.update(identity.commit_sequence.to_le_bytes());
    let initial_database_root = match table.table_map_predecessor {
        RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase => {
            database_root(
                identity.database_id,
                table_map_empty_roots(identity.database_id)[0],
            )
        }
        RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
            initial_database_root,
            ..
        }
        | RuntimeTypedInsertGenerationTableMapPredecessor::PinnedRetained {
            initial_database_root,
            ..
        } => initial_database_root,
    };
    digest.update(initial_database_root);
    digest.update([table.action.encode()]);
    digest.update(table.stable_table_id.to_le_bytes());
    digest.update(table.base_data_generation.to_le_bytes());
    digest.update(table.base_table_root);
    digest.update(table.row_allocator_before.to_le_bytes());
    digest.update(table.row_allocator_high_water.to_le_bytes());
    digest.update(table.initial_logical_row_count.to_le_bytes());
    digest.update(table.final_logical_row_count.to_le_bytes());
    digest.update((rows as u32).to_le_bytes());
    digest.update(batch_root);
    digest.update(table.image_layout_digest);
    digest.update(table.image_content_digest);
    digest.update(0_u32.to_le_bytes());
    finish(digest)
}

#[test]
fn mixed_typed_insert_generation_matches_independent_digest_grammar() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count == 0 {
        return;
    }
    let target = runtime
        .runtime_typed_insert_generation_target(0)
        .expect("target");
    let database = *b"typed-gen-db-001";
    let table_id = 17;
    let base_generation = 5;
    let commit_sequence = 91;
    let rows = [
        (
            100,
            vec![
                Cell {
                    ordinal: 0,
                    column: 10,
                    attnum: 1,
                    storage: [2, 0, 0, 0],
                    oid: 23,
                    size: 4,
                    null: false,
                    value: 42_i32.to_le_bytes().to_vec(),
                },
                Cell {
                    ordinal: 1,
                    column: 11,
                    attnum: 2,
                    storage: [3, 0, 0, 0],
                    oid: 20,
                    size: 8,
                    null: false,
                    value: (-9_i64).to_le_bytes().to_vec(),
                },
                Cell {
                    ordinal: 2,
                    column: 12,
                    attnum: 3,
                    storage: [5, 0, 0, 0],
                    oid: 16,
                    size: 1,
                    null: false,
                    value: vec![1],
                },
                Cell {
                    ordinal: 3,
                    column: 13,
                    attnum: 4,
                    storage: [6, 0, 0, 0],
                    oid: 25,
                    size: -1,
                    null: false,
                    value: b"gpu".to_vec(),
                },
            ],
        ),
        (
            101,
            vec![
                Cell {
                    ordinal: 0,
                    column: 10,
                    attnum: 1,
                    storage: [2, 0, 0, 0],
                    oid: 23,
                    size: 4,
                    null: false,
                    value: (-7_i32).to_le_bytes().to_vec(),
                },
                Cell {
                    ordinal: 1,
                    column: 11,
                    attnum: 2,
                    storage: [3, 0, 0, 0],
                    oid: 20,
                    size: 8,
                    null: false,
                    value: 1000_i64.to_le_bytes().to_vec(),
                },
                Cell {
                    ordinal: 2,
                    column: 12,
                    attnum: 3,
                    storage: [5, 0, 0, 0],
                    oid: 16,
                    size: 1,
                    null: true,
                    value: vec![],
                },
                Cell {
                    ordinal: 3,
                    column: 13,
                    attnum: 4,
                    storage: [6, 0, 0, 0],
                    oid: 25,
                    size: -1,
                    null: false,
                    value: b"db".to_vec(),
                },
            ],
        ),
    ];
    let base_table_root = v2_table_genesis(table_id, base_generation, &rows[0].1);
    let table_map_predecessor = one_table_map_predecessor(database, table_id, base_table_root);
    let initial_database_root = match table_map_predecessor {
        RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
            initial_database_root,
            ..
        }
        | RuntimeTypedInsertGenerationTableMapPredecessor::PinnedRetained {
            initial_database_root,
            ..
        } => initial_database_root,
        RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase => {
            unreachable!()
        }
    };
    let identity = RuntimeTypedInsertGenerationIdentity {
        database_id: database,
        catalog_epoch: 7,
        catalog_digest: [0x31; 32],
        stable_transaction_id: 88,
        commit_sequence,
        write001_typed_statement_digest: [0x91; 32],
    };
    let table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::RowSetInsert,
        table_map_predecessor,
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: base_generation,
        base_table_root,
        row_allocator_before: 99,
        row_allocator_high_water: 101,
        initial_logical_row_count: 0,
        final_logical_row_count: 2,
        image_layout_digest: [0x41; 32],
        image_content_digest: [0x51; 32],
    };
    let cells = rows.iter().map(|(_, cells)| cells.len()).sum();
    let value_bytes = rows
        .iter()
        .flat_map(|(_, cells)| cells)
        .map(|cell| cell.value.len())
        .sum();
    let prepared = PreparedRuntimeTypedInsertGeneration::reserve(
        target,
        RuntimeTypedInsertGenerationAttempt::new(1).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: rows.len(),
            cells,
            value_bytes,
            indexes: 0,
            index_keys: 0,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .expect("reserve generic typed generation");
    let submission = prepared.launch(|encoder| {
        encoder.write_identity(identity);
        encoder.write_table(table);
        for (source_row, (row_id, row_cells)) in rows.iter().enumerate() {
            encoder.write_row(RuntimeTypedInsertGenerationRow {
                stable_table_id: table_id,
                stable_row_id: *row_id,
                source_statement_ordinal: 0,
                source_row_ordinal: source_row as u32,
                cell_count: row_cells.len() as u32,
            });
        }
        for (_, row_cells) in &rows {
            for cell in row_cells {
                encoder.write_cell(RuntimeTypedInsertGenerationCell {
                    catalog_column_ordinal: cell.ordinal,
                    stable_column_id: cell.column,
                    attnum: cell.attnum,
                    storage: cell.storage,
                    declared_type_oid: cell.oid,
                    signed_type_size: cell.size,
                    is_null: cell.null,
                    value: &cell.value,
                });
            }
        }
    });
    let proof = match submission.complete() {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("generation failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!("unknown quiescence: {:?}", unknown.error())
        }
    };
    let final_row_root = v2_batch_row_root(table_id, commit_sequence, &rows);
    let expected_table_root = v2_table_successor(
        table_id,
        base_generation,
        commit_sequence,
        0,
        2,
        base_table_root,
        final_row_root,
        &rows[0].1,
    );
    let empty_table_map = table_map_empty_roots(database);
    let siblings = std::array::from_fn(|depth| empty_table_map[depth + 1]);
    let expected_initial_path = table_map_path(
        database,
        table_id,
        table_map_leaf(database, table_id, base_table_root),
        siblings,
    );
    let expected_final_path = table_map_path(
        database,
        table_id,
        table_map_leaf(database, table_id, expected_table_root),
        siblings,
    );
    let expected_database_root = database_root(database, expected_final_path[0]);
    let expected_input = v2_generation_input(identity, table, rows.len(), final_row_root);
    proof.consume(|attempt, commitments, logical| {
        assert_eq!(attempt.get(), 1);
        assert_eq!(
            &logical.slots[logical.layout.current_row_leaves..logical.layout.s7_final_row_digests],
            rows.iter()
                .enumerate()
                .map(|(source_row, (row_id, cells))| {
                    v2_current_row(table_id, *row_id, commit_sequence, source_row as u32, cells)
                })
                .collect::<Vec<_>>(),
        );
        for (ordinal, (row_id, cells)) in rows.iter().enumerate() {
            let expected_final = s7_final_row(table_id, *row_id, ordinal as u32, cells);
            let mut actual_final = [0_u8; 32];
            let mut actual_transition = [0_u8; 32];
            logical
                .copy_write001_s7_transition_digests_into(
                    ordinal,
                    *row_id,
                    identity.write001_typed_statement_digest,
                    &mut actual_final,
                    &mut actual_transition,
                )
                .expect("device owns every S7 transition digest");
            assert_eq!(actual_final, expected_final);
            assert_eq!(
                actual_transition,
                s7_transition(
                    *row_id,
                    ordinal as u32,
                    identity.write001_typed_statement_digest,
                    expected_final,
                ),
            );
        }
        let mut final_row = [0_u8; 32];
        let mut transition = [0_u8; 32];
        assert!(
            logical
                .copy_write001_s7_transition_digests_into(
                    0,
                    rows[0].0,
                    [0xfe; 32],
                    &mut final_row,
                    &mut transition,
                )
                .is_err(),
            "a different typed statement cannot relabel a device transition digest"
        );
        assert!(
            logical
                .copy_write001_s7_transition_digests_into(
                    0,
                    rows[0].0 + 1,
                    identity.write001_typed_statement_digest,
                    &mut final_row,
                    &mut transition,
                )
                .is_err(),
            "a different stable row identity cannot relabel a device transition digest"
        );
        assert_eq!(
            &logical.slots[logical.layout.row_nodes..logical.layout.row_nodes + 1],
            &[final_row_root],
        );
        let mut bytes = [0_u8; RUNTIME_TYPED_INSERT_GENERATION_COMMITMENT_BYTES];
        commitments.encode_durable_into(&mut bytes).unwrap();
        assert_eq!(&bytes[32..64], &base_table_root);
        assert_eq!(&bytes[64..96], &expected_table_root);
        assert_eq!(&bytes[96..128], &initial_database_root);
        assert_eq!(&bytes[128..160], &expected_database_root);
        assert_eq!(&bytes[..32], &expected_input);
        let mut observed_empty = [[0_u8; 32]; 65];
        let mut observed_initial_leaf = [0_u8; 32];
        let mut observed_initial_path = [[0_u8; 32]; 64];
        let mut observed_final_leaf = [0_u8; 32];
        let mut observed_final_path = [[0_u8; 32]; 64];
        logical
            .copy_table_map_transition_into(
                &mut observed_empty,
                &mut observed_initial_leaf,
                &mut observed_initial_path,
                &mut observed_final_leaf,
                &mut observed_final_path,
            )
            .expect("device owns the exact table-map COW path");
        assert_eq!(observed_empty, empty_table_map);
        assert_eq!(
            observed_initial_leaf,
            table_map_leaf(database, table_id, base_table_root)
        );
        assert_eq!(observed_initial_path, expected_initial_path);
        assert_eq!(
            observed_final_leaf,
            table_map_leaf(database, table_id, expected_table_root)
        );
        assert_eq!(observed_final_path, expected_final_path);
        assert_eq!(logical.column_count, rows[0].1.len());
        assert_eq!(
            logical.layout.column_roots - logical.layout.typed_roots,
            cells
        );
        assert_eq!(logical.rows, rows.len());
        assert_eq!(
            compact_active_level_counts(logical.first_row_id, logical.rows)
                .into_iter()
                .sum::<usize>(),
            1
        );
    });
}

#[test]
fn s7_final_row_digest_parallel_arena_covers_single_variable_width_column() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count == 0 {
        return;
    }
    let target = runtime
        .runtime_typed_insert_generation_target(0)
        .expect("target");
    let database = *b"typed-s7-arena01";
    let table_id = 41;
    let base_generation = 7;
    let commit_sequence = 103;
    let rows = [
        (
            800,
            vec![Cell {
                ordinal: 0,
                column: 91,
                attnum: 1,
                storage: [6, 0, 0, 0],
                oid: 25,
                size: -1,
                null: false,
                value: b"parallel-generic-s7-".repeat(97),
            }],
        ),
        (
            801,
            vec![Cell {
                ordinal: 0,
                column: 91,
                attnum: 1,
                storage: [6, 0, 0, 0],
                oid: 25,
                size: -1,
                null: true,
                value: Vec::new(),
            }],
        ),
        (
            802,
            vec![Cell {
                ordinal: 0,
                column: 91,
                attnum: 1,
                storage: [6, 0, 0, 0],
                oid: 25,
                size: -1,
                null: false,
                value: b"generic-value-tail".repeat(211),
            }],
        ),
    ];
    let base_table_root = v2_table_genesis(table_id, base_generation, &rows[0].1);
    let identity = RuntimeTypedInsertGenerationIdentity {
        database_id: database,
        catalog_epoch: 8,
        catalog_digest: [0x71; 32],
        stable_transaction_id: 102,
        commit_sequence,
        write001_typed_statement_digest: [0xa1; 32],
    };
    let table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::RowSetInsert,
        table_map_predecessor: one_table_map_predecessor(database, table_id, base_table_root),
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: base_generation,
        base_table_root,
        row_allocator_before: 799,
        row_allocator_high_water: 802,
        initial_logical_row_count: 0,
        final_logical_row_count: rows.len() as u64,
        image_layout_digest: [0x81; 32],
        image_content_digest: [0x91; 32],
    };
    let value_bytes = rows
        .iter()
        .flat_map(|(_, cells)| cells)
        .map(|cell| cell.value.len())
        .sum();
    let prepared = PreparedRuntimeTypedInsertGeneration::reserve(
        target,
        RuntimeTypedInsertGenerationAttempt::new(3).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: rows.len(),
            cells: rows.len(),
            value_bytes,
            indexes: 0,
            index_keys: 0,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .expect("reserve one-column variable-width generation");
    let completion = prepared
        .launch(|encoder| {
            encoder.write_identity(identity);
            encoder.write_table(table);
            for (source_row, (row_id, row_cells)) in rows.iter().enumerate() {
                encoder.write_row(RuntimeTypedInsertGenerationRow {
                    stable_table_id: table_id,
                    stable_row_id: *row_id,
                    source_statement_ordinal: 0,
                    source_row_ordinal: source_row as u32,
                    cell_count: row_cells.len() as u32,
                });
                for cell in row_cells {
                    encoder.write_cell(RuntimeTypedInsertGenerationCell {
                        catalog_column_ordinal: cell.ordinal,
                        stable_column_id: cell.column,
                        attnum: cell.attnum,
                        storage: cell.storage,
                        declared_type_oid: cell.oid,
                        signed_type_size: cell.size,
                        is_null: cell.null,
                        value: &cell.value,
                    });
                }
            }
        })
        .complete();
    let proof = match completion {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("generation failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!("unknown quiescence: {:?}", unknown.error())
        }
    };
    proof.consume(|attempt, _, logical| {
        assert_eq!(attempt.get(), 3);
        for (ordinal, (row_id, cells)) in rows.iter().enumerate() {
            let expected_final = s7_final_row(table_id, *row_id, ordinal as u32, cells);
            let mut actual_final = [0_u8; 32];
            let mut actual_transition = [0_u8; 32];
            logical
                .copy_write001_s7_transition_digests_into(
                    ordinal,
                    *row_id,
                    identity.write001_typed_statement_digest,
                    &mut actual_final,
                    &mut actual_transition,
                )
                .expect("device owns every variable-width S7 transition digest");
            assert_eq!(actual_final, expected_final);
            assert_eq!(
                actual_transition,
                s7_transition(
                    *row_id,
                    ordinal as u32,
                    identity.write001_typed_statement_digest,
                    expected_final,
                ),
            );
        }
    });
}

#[test]
fn empty_create_generation_authenticates_ordered_generic_column_shapes() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count == 0 {
        return;
    }
    let target = runtime
        .runtime_typed_insert_generation_target(0)
        .expect("target");
    let database = *b"typed-create-001";
    let table_id = 29;
    let commit_sequence = 41;
    let columns = vec![
        Cell {
            ordinal: 0,
            column: 71,
            attnum: 1,
            storage: [2, 0, 0, 0],
            oid: 23,
            size: 4,
            null: true,
            value: Vec::new(),
        },
        Cell {
            ordinal: 1,
            column: 72,
            attnum: 2,
            storage: [6, 0, 0, 0],
            oid: 25,
            size: -1,
            null: true,
            value: Vec::new(),
        },
        Cell {
            ordinal: 2,
            column: 73,
            attnum: 3,
            storage: [5, 0, 0, 0],
            oid: 16,
            size: 1,
            null: true,
            value: Vec::new(),
        },
    ];
    let identity = RuntimeTypedInsertGenerationIdentity {
        database_id: database,
        catalog_epoch: 3,
        catalog_digest: [0x61; 32],
        stable_transaction_id: 40,
        commit_sequence,
        write001_typed_statement_digest: [0; 32],
    };
    let table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::CreateEmpty,
        table_map_predecessor:
            RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase,
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: commit_sequence,
        base_table_root: [0; 32],
        row_allocator_before: 1,
        row_allocator_high_water: 1,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        image_layout_digest: [0x62; 32],
        // Empty CREATE has no final image; zero is the explicit absence sentinel.
        image_content_digest: [0; 32],
    };
    let prepared = PreparedRuntimeTypedInsertGeneration::reserve(
        target,
        RuntimeTypedInsertGenerationAttempt::new(2).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: 0,
            cells: columns.len(),
            value_bytes: 0,
            indexes: 0,
            index_keys: 0,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .expect("reserve empty CREATE generation");
    let completion = prepared
        .launch(|encoder| {
            encoder.write_identity(identity);
            encoder.write_table(table);
            for column in &columns {
                encoder.write_cell(RuntimeTypedInsertGenerationCell {
                    catalog_column_ordinal: column.ordinal,
                    stable_column_id: column.column,
                    attnum: column.attnum,
                    storage: column.storage,
                    declared_type_oid: column.oid,
                    signed_type_size: column.size,
                    is_null: true,
                    value: &[],
                });
            }
        })
        .complete();
    let proof = match completion {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("empty CREATE generation failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!("unknown CREATE quiescence: {:?}", unknown.error())
        }
    };
    let expected_table = v2_table_genesis(table_id, commit_sequence, &columns);
    let empty_table_map = table_map_empty_roots(database);
    let empty_siblings = std::array::from_fn(|depth| empty_table_map[depth + 1]);
    let expected_initial_path =
        table_map_path(database, table_id, empty_table_map[64], empty_siblings);
    let expected_final_path = table_map_path(
        database,
        table_id,
        table_map_leaf(database, table_id, expected_table),
        empty_siblings,
    );
    let expected_initial_database = database_root(database, expected_initial_path[0]);
    let expected_database = database_root(database, expected_final_path[0]);
    proof.consume(|attempt, commitments, logical| {
        assert_eq!(attempt.get(), 2);
        let mut bytes = [0_u8; RUNTIME_TYPED_INSERT_GENERATION_COMMITMENT_BYTES];
        commitments.encode_durable_into(&mut bytes).unwrap();
        assert_eq!(&bytes[32..64], &[0; 32]);
        assert_eq!(&bytes[64..96], &expected_table);
        assert_eq!(&bytes[96..128], &expected_initial_database);
        assert_eq!(&bytes[128..160], &expected_database);
        let mut observed_shapes = vec![[0_u8; 32]; columns.len()];
        let mut observed_columns = vec![[0_u8; 32]; columns.len()];
        logical
            .copy_column_roots_into(&mut observed_shapes, &mut observed_columns)
            .unwrap();
        assert_eq!(
            observed_shapes,
            columns
                .iter()
                .map(|column| shape(table_id, column))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            observed_columns,
            columns
                .iter()
                .map(|column| empty_column(table_id, shape(table_id, column)))
                .collect::<Vec<_>>()
        );
        assert_eq!(logical.rows, 0);
        assert_eq!(logical.layout.row_node_capacity, 0);
        let mut observed_empty = [[0_u8; 32]; 65];
        let mut observed_initial_leaf = [0_u8; 32];
        let mut observed_initial_path = [[0_u8; 32]; 64];
        let mut observed_final_leaf = [0_u8; 32];
        let mut observed_final_path = [[0_u8; 32]; 64];
        logical
            .copy_table_map_transition_into(
                &mut observed_empty,
                &mut observed_initial_leaf,
                &mut observed_initial_path,
                &mut observed_final_leaf,
                &mut observed_final_path,
            )
            .expect("first CREATE returns the canonical empty predecessor and successor path");
        assert_eq!(observed_empty, empty_table_map);
        assert_eq!(observed_initial_leaf, empty_table_map[64]);
        assert_eq!(observed_initial_path, expected_initial_path);
        assert_eq!(
            observed_final_leaf,
            table_map_leaf(database, table_id, expected_table)
        );
        assert_eq!(observed_final_path, expected_final_path);
    });
}

#[test]
fn empty_create_generation_authenticates_one_inline_primary_index() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count == 0 {
        return;
    }
    let target = runtime
        .runtime_typed_insert_generation_target(0)
        .expect("target");
    let database = *b"typed-create-pk1";
    let table_id = 31;
    let commit_sequence = 43;
    let column = Cell {
        ordinal: 0,
        column: 74,
        attnum: 1,
        storage: [2, 0, 0, 0],
        oid: 23,
        size: 4,
        null: true,
        value: Vec::new(),
    };
    let identity = RuntimeTypedInsertGenerationIdentity {
        database_id: database,
        catalog_epoch: 3,
        catalog_digest: [0x63; 32],
        stable_transaction_id: 42,
        commit_sequence,
        write001_typed_statement_digest: [0; 32],
    };
    let table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::CreateEmpty,
        table_map_predecessor:
            RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase,
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: commit_sequence,
        base_table_root: [0; 32],
        row_allocator_before: 1,
        row_allocator_high_water: 1,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        image_layout_digest: [0x64; 32],
        image_content_digest: [0; 32],
    };
    let index = RuntimeTypedInsertGenerationIndex {
        stable_index_id: 802,
        raw_catalog_index_ordinal: 0,
        // UNIQUE | PRIMARY | MAINTAINED.
        index_flags: 11,
        null_equality_policy: 1,
        base_generation: 0,
        base_root: [0; 32],
        key_start: 0,
        key_count: 1,
        effect_start: 0,
        effect_count: 0,
    };
    let key = RuntimeTypedInsertGenerationIndexKeyColumn {
        key_ordinal: 0,
        catalog_column_ordinal: column.ordinal,
        stable_column_id: column.column,
        attnum: column.attnum,
        storage: column.storage,
        declared_type_oid: column.oid,
        signed_type_size: column.size,
        column_name_digest: [0x65; 32],
    };
    let proof = match PreparedRuntimeTypedInsertGeneration::reserve(
        target,
        RuntimeTypedInsertGenerationAttempt::new(commit_sequence).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: 0,
            cells: 1,
            value_bytes: 0,
            indexes: 1,
            index_keys: 1,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .expect("reserve indexed empty CREATE generation")
    .launch(|encoder| {
        encoder.write_identity(identity);
        encoder.write_table(table);
        encoder.write_cell(RuntimeTypedInsertGenerationCell {
            catalog_column_ordinal: column.ordinal,
            stable_column_id: column.column,
            attnum: column.attnum,
            storage: column.storage,
            declared_type_oid: column.oid,
            signed_type_size: column.size,
            is_null: true,
            value: &[],
        });
        encoder.write_index(index);
        encoder.write_index_key(key);
    })
    .complete()
    {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("GPU indexed CREATE failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!(
                "GPU indexed CREATE quiescence unknown: {:?}",
                unknown.error()
            )
        }
    };
    proof.consume(|_attempt, commitments, logical| {
        let mut final_table_root = [0; 32];
        commitments.copy_final_table_root_into(&mut final_table_root);
        assert_ne!(final_table_root, [0; 32]);
        let mut roots = [RuntimeTypedInsertGenerationIndexRoot {
            stable_index_id: 0,
            initial_generation: 0,
            initial_root: [0; 32],
            final_generation: 0,
            final_root: [0; 32],
        }];
        logical
            .copy_index_generation_roots_into(&mut roots)
            .unwrap();
        assert_eq!(roots[0].stable_index_id, index.stable_index_id);
        assert_eq!(roots[0].initial_generation, 0);
        assert_eq!(roots[0].initial_root, [0; 32]);
        assert_eq!(roots[0].final_generation, commit_sequence);
        assert_ne!(roots[0].final_root, [0; 32]);
    });
}

#[test]
fn second_empty_create_reuses_a_nonempty_table_map_predecessor_path() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count == 0 {
        return;
    }
    let target = runtime
        .runtime_typed_insert_generation_target(0)
        .expect("target");
    let database = *b"typed-create-002";
    let existing_table_id = 1;
    let table_id = (1_u64 << 63) | 29;
    let commit_sequence = 47;
    let columns = vec![Cell {
        ordinal: 0,
        column: 81,
        attnum: 1,
        storage: [2, 0, 0, 0],
        oid: 23,
        size: 4,
        null: true,
        value: Vec::new(),
    }];
    let existing_root = v2_table_genesis(existing_table_id, 5, &columns);
    let empty = table_map_empty_roots(database);
    let existing_siblings = std::array::from_fn(|depth| empty[depth + 1]);
    let existing_path = table_map_path(
        database,
        existing_table_id,
        table_map_leaf(database, existing_table_id, existing_root),
        existing_siblings,
    );
    let mut create_siblings = std::array::from_fn(|depth| empty[depth + 1]);
    create_siblings[0] = existing_path[1];
    let initial_database_root = database_root(database, existing_path[0]);
    let identity = RuntimeTypedInsertGenerationIdentity {
        database_id: database,
        catalog_epoch: 4,
        catalog_digest: [0x72; 32],
        stable_transaction_id: 46,
        commit_sequence,
        write001_typed_statement_digest: [0; 32],
    };
    let table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::CreateEmpty,
        table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
            initial_database_root,
            sibling_roots: create_siblings,
        },
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: commit_sequence,
        base_table_root: [0; 32],
        row_allocator_before: 1,
        row_allocator_high_water: 1,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        image_layout_digest: [0x73; 32],
        image_content_digest: [0; 32],
    };
    let proof = match PreparedRuntimeTypedInsertGeneration::reserve(
        target,
        RuntimeTypedInsertGenerationAttempt::new(4).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: 0,
            cells: columns.len(),
            value_bytes: 0,
            indexes: 0,
            index_keys: 0,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .expect("reserve second CREATE")
    .launch(|encoder| {
        encoder.write_identity(identity);
        encoder.write_table(table);
        for column in &columns {
            encoder.write_cell(RuntimeTypedInsertGenerationCell {
                catalog_column_ordinal: column.ordinal,
                stable_column_id: column.column,
                attnum: column.attnum,
                storage: column.storage,
                declared_type_oid: column.oid,
                signed_type_size: column.size,
                is_null: true,
                value: &[],
            });
        }
    })
    .complete()
    {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("second CREATE generation failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!("unknown second CREATE quiescence: {:?}", unknown.error())
        }
    };
    let expected_table = v2_table_genesis(table_id, commit_sequence, &columns);
    let expected_initial_path = table_map_path(database, table_id, empty[64], create_siblings);
    let expected_final_path = table_map_path(
        database,
        table_id,
        table_map_leaf(database, table_id, expected_table),
        create_siblings,
    );
    proof.consume(|_, commitments, logical| {
        let mut observed_initial_database = [0_u8; 32];
        let mut observed_final_database = [0_u8; 32];
        commitments.copy_initial_database_root_into(&mut observed_initial_database);
        commitments.copy_final_database_root_into(&mut observed_final_database);
        assert_eq!(observed_initial_database, initial_database_root);
        assert_eq!(
            observed_final_database,
            database_root(database, expected_final_path[0])
        );
        let mut observed_empty = [[0_u8; 32]; 65];
        let mut observed_initial_leaf = [0_u8; 32];
        let mut observed_initial_path = [[0_u8; 32]; 64];
        let mut observed_final_leaf = [0_u8; 32];
        let mut observed_final_path = [[0_u8; 32]; 64];
        logical
            .copy_table_map_transition_into(
                &mut observed_empty,
                &mut observed_initial_leaf,
                &mut observed_initial_path,
                &mut observed_final_leaf,
                &mut observed_final_path,
            )
            .expect("second CREATE exposes an exact COW path");
        assert_eq!(observed_empty, empty);
        assert_eq!(observed_initial_leaf, empty[64]);
        assert_eq!(observed_initial_path, expected_initial_path);
        assert_eq!(
            observed_final_leaf,
            table_map_leaf(database, table_id, expected_table)
        );
        assert_eq!(observed_final_path, expected_final_path);
    });
}

#[test]
fn table_map_predecessor_rejects_wrong_sibling_root_and_target_presence() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count == 0 {
        return;
    }
    let target = runtime
        .runtime_typed_insert_generation_target(0)
        .expect("target");
    let database = *b"typed-map-reject";
    let table_id = 53;
    let commit_sequence = 63;
    let column = Cell {
        ordinal: 0,
        column: 91,
        attnum: 1,
        storage: [2, 0, 0, 0],
        oid: 23,
        size: 4,
        null: false,
        value: 7_i32.to_le_bytes().to_vec(),
    };
    let base_root = v2_table_genesis(table_id, 7, std::slice::from_ref(&column));
    let pinned = one_table_map_predecessor(database, table_id, base_root);
    let identity = RuntimeTypedInsertGenerationIdentity {
        database_id: database,
        catalog_epoch: 6,
        catalog_digest: [0x83; 32],
        stable_transaction_id: 62,
        commit_sequence,
        write001_typed_statement_digest: [0; 32],
    };
    let row_set_table = |predecessor| RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::RowSetInsert,
        table_map_predecessor: predecessor,
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: 7,
        base_table_root: base_root,
        row_allocator_before: 100,
        row_allocator_high_water: 100,
        initial_logical_row_count: 0,
        final_logical_row_count: 1,
        image_layout_digest: [0x84; 32],
        image_content_digest: [0x85; 32],
    };
    let complete_row_set = |attempt, table| {
        PreparedRuntimeTypedInsertGeneration::reserve(
            target.clone(),
            RuntimeTypedInsertGenerationAttempt::new(attempt).unwrap(),
            RuntimeTypedInsertGenerationGeometry {
                rows: 1,
                cells: 1,
                value_bytes: column.value.len(),
                indexes: 0,
                index_keys: 0,
                index_effects: 0,
                index_effect_components: 0,
            },
        )
        .expect("reserve rejected row set")
        .launch(|encoder| {
            encoder.write_identity(identity);
            encoder.write_table(table);
            encoder.write_row(RuntimeTypedInsertGenerationRow {
                stable_table_id: table_id,
                stable_row_id: 100,
                source_statement_ordinal: 0,
                source_row_ordinal: 0,
                cell_count: 1,
            });
            encoder.write_cell(RuntimeTypedInsertGenerationCell {
                catalog_column_ordinal: column.ordinal,
                stable_column_id: column.column,
                attnum: column.attnum,
                storage: column.storage,
                declared_type_oid: column.oid,
                signed_type_size: column.size,
                is_null: column.null,
                value: &column.value,
            });
        })
        .complete()
    };
    let retained = one_table_map_retained_predecessor(database, table_id, base_root);
    let retained_completion = complete_row_set(4, row_set_table(retained));
    let RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(retained_proof)) = retained_completion
    else {
        panic!("GPU retained table-map predecessor generation failed");
    };
    retained_proof.consume(|_, _, logical| {
        let mut observed_empty = [[0_u8; 32]; 65];
        let mut observed_initial_leaf = [0_u8; 32];
        let mut observed_initial_path = [[0_u8; 32]; 64];
        let mut observed_final_leaf = [0_u8; 32];
        let mut observed_final_path = [[0_u8; 32]; 64];
        logical
            .copy_table_map_transition_into(
                &mut observed_empty,
                &mut observed_initial_leaf,
                &mut observed_initial_path,
                &mut observed_final_leaf,
                &mut observed_final_path,
            )
            .expect("retained predecessor completion has the fixed map shape");
        let expected_empty = table_map_empty_roots(database);
        let expected_initial_leaf = table_map_leaf(database, table_id, base_root);
        let expected_initial_path = table_map_path(
            database,
            table_id,
            expected_initial_leaf,
            std::array::from_fn(|depth| expected_empty[depth + 1]),
        );
        assert_eq!(observed_empty, expected_empty);
        assert_eq!(observed_initial_leaf, expected_initial_leaf);
        assert_eq!(observed_initial_path, expected_initial_path);
        assert_ne!(observed_final_leaf, expected_initial_leaf);
        assert_ne!(observed_final_path, expected_initial_path);
    });
    let RuntimeTypedInsertGenerationTableMapPredecessor::PinnedRetained {
        initial_database_root: retained_initial_database_root,
        sibling_roots: retained_sibling_roots,
        empty_roots: retained_empty_roots,
        initial_leaf_root: retained_initial_leaf_root,
        initial_path_roots,
    } = retained
    else {
        unreachable!();
    };
    let mut wrong_initial_path = initial_path_roots;
    wrong_initial_path[0][0] ^= 1;
    assert!(matches!(
        complete_row_set(
            5,
            row_set_table(
                RuntimeTypedInsertGenerationTableMapPredecessor::PinnedRetained {
                    initial_database_root: retained_initial_database_root,
                    sibling_roots: retained_sibling_roots,
                    empty_roots: retained_empty_roots,
                    initial_leaf_root: retained_initial_leaf_root,
                    initial_path_roots: wrong_initial_path,
                }
            )
        ),
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(
            RuntimeTypedInsertGenerationError::DeviceRejected(_)
        ))
    ));
    let RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
        initial_database_root,
        sibling_roots,
    } = pinned
    else {
        unreachable!();
    };
    let mut wrong_siblings = sibling_roots;
    wrong_siblings[0][0] ^= 1;
    assert!(matches!(
        complete_row_set(
            6,
            row_set_table(RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
                initial_database_root,
                sibling_roots: wrong_siblings,
            })
        ),
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(
            RuntimeTypedInsertGenerationError::DeviceRejected(_)
        ))
    ));
    let mut wrong_root = initial_database_root;
    wrong_root[0] ^= 1;
    assert!(matches!(
        complete_row_set(
            7,
            row_set_table(RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
                initial_database_root: wrong_root,
                sibling_roots,
            })
        ),
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(
            RuntimeTypedInsertGenerationError::DeviceRejected(_)
        ))
    ));
    let target_present_table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::CreateEmpty,
        table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
            initial_database_root,
            sibling_roots,
        },
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: commit_sequence,
        base_table_root: [0; 32],
        row_allocator_before: 1,
        row_allocator_high_water: 1,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        image_layout_digest: [0x86; 32],
        image_content_digest: [0; 32],
    };
    let target_presence = PreparedRuntimeTypedInsertGeneration::reserve(
        target,
        RuntimeTypedInsertGenerationAttempt::new(8).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: 0,
            cells: 1,
            value_bytes: 0,
            indexes: 0,
            index_keys: 0,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .expect("reserve rejected CREATE")
    .launch(|encoder| {
        encoder.write_identity(identity);
        encoder.write_table(target_present_table);
        encoder.write_cell(RuntimeTypedInsertGenerationCell {
            catalog_column_ordinal: column.ordinal,
            stable_column_id: column.column,
            attnum: column.attnum,
            storage: column.storage,
            declared_type_oid: column.oid,
            signed_type_size: column.size,
            is_null: true,
            value: &[],
        });
    })
    .complete();
    assert!(matches!(
        target_presence,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(
            RuntimeTypedInsertGenerationError::DeviceRejected(_)
        ))
    ));
}

#[test]
fn gpu_derives_compound_unique_index_enrollment_and_append_generation_roots() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    if runtime.snapshot().device_count == 0 {
        return;
    }
    let target = runtime
        .runtime_typed_insert_generation_target(0)
        .expect("target");
    let database = *b"typed-index-can1";
    let table_id = 41;
    let columns = [
        Cell {
            ordinal: 0,
            column: 701,
            attnum: 1,
            storage: [2, 0, 0, 0],
            oid: 23,
            size: 4,
            null: true,
            value: Vec::new(),
        },
        Cell {
            ordinal: 1,
            column: 702,
            attnum: 2,
            storage: [2, 0, 0, 0],
            oid: 23,
            size: 4,
            null: true,
            value: Vec::new(),
        },
    ];
    let create_identity = RuntimeTypedInsertGenerationIdentity {
        database_id: database,
        catalog_epoch: 11,
        catalog_digest: [0x10; 32],
        stable_transaction_id: 900_001,
        commit_sequence: 101,
        write001_typed_statement_digest: [0; 32],
    };
    let create_table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::CreateEmpty,
        table_map_predecessor:
            RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase,
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: create_identity.commit_sequence,
        base_table_root: [0; 32],
        row_allocator_before: 1,
        row_allocator_high_water: 1,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        image_layout_digest: [0x22; 32],
        image_content_digest: [0; 32],
    };
    let create_proof = match PreparedRuntimeTypedInsertGeneration::reserve(
        target.clone(),
        RuntimeTypedInsertGenerationAttempt::new(101).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: 0,
            cells: columns.len(),
            value_bytes: 0,
            indexes: 0,
            index_keys: 0,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .unwrap()
    .launch(|encoder| {
        encoder.write_identity(create_identity);
        encoder.write_table(create_table);
        for column in &columns {
            encoder.write_cell(RuntimeTypedInsertGenerationCell {
                catalog_column_ordinal: column.ordinal,
                stable_column_id: column.column,
                attnum: column.attnum,
                storage: column.storage,
                declared_type_oid: column.oid,
                signed_type_size: column.size,
                is_null: true,
                value: &[],
            });
        }
    })
    .complete()
    {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("GPU CREATE failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!("GPU CREATE quiescence unknown: {:?}", unknown.error())
        }
    };
    let (created_table_root, created_database_root, empty_map) =
        create_proof.consume(|_attempt, commitments, logical| {
            let mut table_root = [0; 32];
            let mut database_root = [0; 32];
            commitments.copy_final_table_root_into(&mut table_root);
            commitments.copy_final_database_root_into(&mut database_root);
            let mut empty = [[0; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH + 1];
            let mut initial_leaf = [0; 32];
            let mut initial_path = [[0; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH];
            let mut final_leaf = [0; 32];
            let mut final_path = [[0; 32]; RUNTIME_TYPED_INSERT_GENERATION_TABLE_MAP_DEPTH];
            logical
                .copy_table_map_transition_into(
                    &mut empty,
                    &mut initial_leaf,
                    &mut initial_path,
                    &mut final_leaf,
                    &mut final_path,
                )
                .unwrap();
            (table_root, database_root, empty)
        });
    let siblings = std::array::from_fn(|depth| empty_map[depth + 1]);
    let index = RuntimeTypedInsertGenerationIndex {
        stable_index_id: 801,
        raw_catalog_index_ordinal: 0,
        index_flags: 9,
        null_equality_policy: 1,
        base_generation: 0,
        base_root: [0; 32],
        key_start: 0,
        key_count: 2,
        effect_start: 0,
        effect_count: 0,
    };
    let keys = [
        RuntimeTypedInsertGenerationIndexKeyColumn {
            key_ordinal: 0,
            catalog_column_ordinal: 0,
            stable_column_id: 701,
            attnum: 1,
            storage: [2, 0, 0, 0],
            declared_type_oid: 23,
            signed_type_size: 4,
            column_name_digest: [0x71; 32],
        },
        RuntimeTypedInsertGenerationIndexKeyColumn {
            key_ordinal: 1,
            catalog_column_ordinal: 1,
            stable_column_id: 702,
            attnum: 2,
            storage: [2, 0, 0, 0],
            declared_type_oid: 23,
            signed_type_size: 4,
            column_name_digest: [0x72; 32],
        },
    ];
    let enroll_identity = RuntimeTypedInsertGenerationIdentity {
        commit_sequence: 102,
        stable_transaction_id: 900_002,
        ..create_identity
    };
    let enroll_table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::EnrollIndex,
        table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
            initial_database_root: created_database_root,
            sibling_roots: siblings,
        },
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: 101,
        base_table_root: created_table_root,
        row_allocator_before: 1,
        row_allocator_high_water: 1,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        image_layout_digest: [0x22; 32],
        image_content_digest: [0; 32],
    };
    let enroll_proof = match PreparedRuntimeTypedInsertGeneration::reserve(
        target.clone(),
        RuntimeTypedInsertGenerationAttempt::new(102).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: 0,
            cells: columns.len(),
            value_bytes: 0,
            indexes: 1,
            index_keys: 2,
            index_effects: 0,
            index_effect_components: 0,
        },
    )
    .unwrap()
    .launch(|encoder| {
        encoder.write_identity(enroll_identity);
        encoder.write_table(enroll_table);
        for column in &columns {
            encoder.write_cell(RuntimeTypedInsertGenerationCell {
                catalog_column_ordinal: column.ordinal,
                stable_column_id: column.column,
                attnum: column.attnum,
                storage: column.storage,
                declared_type_oid: column.oid,
                signed_type_size: column.size,
                is_null: true,
                value: &[],
            });
        }
        encoder.write_index(index);
        for key in keys {
            encoder.write_index_key(key);
        }
    })
    .complete()
    {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("GPU index enrollment failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!(
                "GPU index enrollment quiescence unknown: {:?}",
                unknown.error()
            )
        }
    };
    let (enrolled_table_root, enrolled_database_root, enrolled_index_root) =
        enroll_proof.consume(|_attempt, commitments, logical| {
            let mut table_root = [0; 32];
            let mut database_root = [0; 32];
            commitments.copy_final_table_root_into(&mut table_root);
            commitments.copy_final_database_root_into(&mut database_root);
            let mut roots = [RuntimeTypedInsertGenerationIndexRoot {
                stable_index_id: 0,
                initial_generation: 0,
                initial_root: [0; 32],
                final_generation: 0,
                final_root: [0; 32],
            }];
            logical
                .copy_index_generation_roots_into(&mut roots)
                .unwrap();
            assert_eq!(roots[0].stable_index_id, index.stable_index_id);
            assert_eq!(roots[0].initial_root, [0; 32]);
            assert_eq!(roots[0].final_generation, enroll_identity.commit_sequence);
            assert_ne!(roots[0].final_root, [0; 32]);
            (table_root, database_root, roots[0])
        });
    let indexed_identity = RuntimeTypedInsertGenerationIdentity {
        commit_sequence: 103,
        stable_transaction_id: 900_003,
        write001_typed_statement_digest: [0x33; 32],
        ..create_identity
    };
    let indexed_table = RuntimeTypedInsertGenerationTable {
        action: RuntimeTypedInsertGenerationTableAction::RowSetInsert,
        table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
            initial_database_root: enrolled_database_root,
            sibling_roots: siblings,
        },
        stable_table_id: table_id,
        write001_final_image_ref: 0,
        base_data_generation: enroll_identity.commit_sequence,
        base_table_root: enrolled_table_root,
        row_allocator_before: 1,
        row_allocator_high_water: 2,
        initial_logical_row_count: 0,
        final_logical_row_count: 1,
        image_layout_digest: [0x22; 32],
        image_content_digest: [0x44; 32],
    };
    let indexed_descriptor = RuntimeTypedInsertGenerationIndex {
        base_generation: enrolled_index_root.final_generation,
        base_root: enrolled_index_root.final_root,
        effect_count: 1,
        ..index
    };
    let run_indexed = |bad_component: bool| {
        PreparedRuntimeTypedInsertGeneration::reserve(
            target.clone(),
            RuntimeTypedInsertGenerationAttempt::new(indexed_identity.commit_sequence).unwrap(),
            RuntimeTypedInsertGenerationGeometry {
                rows: 1,
                cells: 2,
                value_bytes: 8,
                indexes: 1,
                index_keys: 2,
                index_effects: 1,
                index_effect_components: 2,
            },
        )
        .unwrap()
        .launch(|encoder| {
            encoder.write_identity(indexed_identity);
            encoder.write_table(indexed_table);
            encoder.write_row(RuntimeTypedInsertGenerationRow {
                stable_table_id: table_id,
                stable_row_id: 1,
                source_statement_ordinal: 0,
                source_row_ordinal: 0,
                cell_count: 2,
            });
            for (ordinal, column) in columns.iter().enumerate() {
                encoder.write_cell(RuntimeTypedInsertGenerationCell {
                    catalog_column_ordinal: column.ordinal,
                    stable_column_id: column.column,
                    attnum: column.attnum,
                    storage: column.storage,
                    declared_type_oid: column.oid,
                    signed_type_size: column.size,
                    is_null: false,
                    value: &(ordinal as i32 + 10).to_le_bytes(),
                });
            }
            encoder.write_index(indexed_descriptor);
            for key in keys {
                encoder.write_index_key(key);
            }
            encoder.write_index_effect(RuntimeTypedInsertGenerationIndexEffect {
                stable_table_id: table_id,
                stable_index_id: index.stable_index_id,
                stable_row_id: 1,
                source_catalog_ordinal: 0,
                component_start: 0,
                component_count: 2,
            });
            encoder.write_index_effect_component(
                RuntimeTypedInsertGenerationIndexEffectComponent {
                    catalog_column_ordinal: 0,
                    stable_column_id: 701,
                },
            );
            encoder.write_index_effect_component(
                RuntimeTypedInsertGenerationIndexEffectComponent {
                    catalog_column_ordinal: 1,
                    stable_column_id: if bad_component { 999 } else { 702 },
                },
            );
        })
        .complete()
    };
    let indexed_proof = match run_indexed(false) {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("GPU indexed row generation failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!(
                "GPU indexed row generation quiescence unknown: {:?}",
                unknown.error()
            )
        }
    };
    let (first_indexed_table_root, first_indexed_database_root, first_indexed_root) = indexed_proof
        .consume(|_attempt, commitments, logical| {
            let mut input = [0; 32];
            let mut initial_table = [0; 32];
            let mut final_table = [0; 32];
            let mut final_database = [0; 32];
            commitments.copy_generation_input_into(&mut input);
            commitments.copy_initial_table_root_into(&mut initial_table);
            commitments.copy_final_table_root_into(&mut final_table);
            commitments.copy_final_database_root_into(&mut final_database);
            assert_ne!(input, [0; 32], "the GPU closes generation-input/v2");
            assert_eq!(initial_table, enrolled_table_root);
            assert_ne!(final_table, initial_table);
            let mut roots = [RuntimeTypedInsertGenerationIndexRoot {
                stable_index_id: 0,
                initial_generation: 0,
                initial_root: [0; 32],
                final_generation: 0,
                final_root: [0; 32],
            }];
            logical
                .copy_index_generation_roots_into(&mut roots)
                .unwrap();
            assert_eq!(
                roots[0].initial_generation,
                enrolled_index_root.final_generation
            );
            assert_eq!(roots[0].initial_root, enrolled_index_root.final_root);
            assert_eq!(roots[0].final_generation, indexed_identity.commit_sequence);
            assert_ne!(roots[0].final_root, roots[0].initial_root);
            (final_table, final_database, roots[0])
        });
    assert!(matches!(
        run_indexed(true),
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(
            RuntimeTypedInsertGenerationError::DeviceRejected(17)
        ))
    ));
    let nonempty_identity = RuntimeTypedInsertGenerationIdentity {
        commit_sequence: 104,
        stable_transaction_id: 900_004,
        ..indexed_identity
    };
    let nonempty_table = RuntimeTypedInsertGenerationTable {
        table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor::Pinned {
            initial_database_root: first_indexed_database_root,
            sibling_roots: siblings,
        },
        base_data_generation: indexed_identity.commit_sequence,
        base_table_root: first_indexed_table_root,
        row_allocator_before: 2,
        row_allocator_high_water: 3,
        initial_logical_row_count: 1,
        final_logical_row_count: 2,
        ..indexed_table
    };
    let nonempty_descriptor = RuntimeTypedInsertGenerationIndex {
        base_generation: first_indexed_root.final_generation,
        base_root: first_indexed_root.final_root,
        ..indexed_descriptor
    };
    let second_values = [21_i32.to_le_bytes(), 22_i32.to_le_bytes()];
    let nonempty_predecessor = PreparedRuntimeTypedInsertGeneration::reserve(
        target,
        RuntimeTypedInsertGenerationAttempt::new(nonempty_identity.commit_sequence).unwrap(),
        RuntimeTypedInsertGenerationGeometry {
            rows: 1,
            cells: 2,
            value_bytes: 8,
            indexes: 1,
            index_keys: 2,
            index_effects: 1,
            index_effect_components: 2,
        },
    )
    .unwrap()
    .launch(|encoder| {
        encoder.write_identity(nonempty_identity);
        encoder.write_table(nonempty_table);
        encoder.write_row(RuntimeTypedInsertGenerationRow {
            stable_table_id: table_id,
            stable_row_id: 2,
            source_statement_ordinal: 0,
            source_row_ordinal: 0,
            cell_count: 2,
        });
        for (ordinal, column) in columns.iter().enumerate() {
            encoder.write_cell(RuntimeTypedInsertGenerationCell {
                catalog_column_ordinal: column.ordinal,
                stable_column_id: column.column,
                attnum: column.attnum,
                storage: column.storage,
                declared_type_oid: column.oid,
                signed_type_size: column.size,
                is_null: false,
                value: &second_values[ordinal],
            });
        }
        encoder.write_index(nonempty_descriptor);
        for key in keys {
            encoder.write_index_key(key);
        }
        encoder.write_index_effect(RuntimeTypedInsertGenerationIndexEffect {
            stable_table_id: table_id,
            stable_index_id: index.stable_index_id,
            stable_row_id: 2,
            source_catalog_ordinal: 0,
            component_start: 0,
            component_count: 2,
        });
        for key in keys {
            encoder.write_index_effect_component(
                RuntimeTypedInsertGenerationIndexEffectComponent {
                    catalog_column_ordinal: key.catalog_column_ordinal,
                    stable_column_id: key.stable_column_id,
                },
            );
        }
    })
    .complete();
    let nonempty_proof = match nonempty_predecessor {
        RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
        RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
            panic!("GPU nonempty indexed successor generation failed: {error:?}")
        }
        RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
            panic!(
                "GPU nonempty indexed successor quiescence unknown: {:?}",
                unknown.error()
            )
        }
    };
    nonempty_proof.consume(|_attempt, commitments, logical| {
        let mut initial_table = [0; 32];
        let mut final_table = [0; 32];
        commitments.copy_initial_table_root_into(&mut initial_table);
        commitments.copy_final_table_root_into(&mut final_table);
        assert_eq!(initial_table, first_indexed_table_root);
        assert_ne!(final_table, initial_table);
        let mut roots = [RuntimeTypedInsertGenerationIndexRoot {
            stable_index_id: 0,
            initial_generation: 0,
            initial_root: [0; 32],
            final_generation: 0,
            final_root: [0; 32],
        }];
        logical
            .copy_index_generation_roots_into(&mut roots)
            .unwrap();
        assert_eq!(roots[0].stable_index_id, index.stable_index_id);
        assert_eq!(
            roots[0].initial_generation,
            first_indexed_root.final_generation
        );
        assert_eq!(roots[0].initial_root, first_indexed_root.final_root);
        assert_eq!(roots[0].final_generation, nonempty_identity.commit_sequence);
        assert_ne!(roots[0].final_root, roots[0].initial_root);
    });
}
