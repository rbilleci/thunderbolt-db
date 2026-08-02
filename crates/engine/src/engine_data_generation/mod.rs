//! Production-compiled WRITE-001 logical-generation foundations and the one sealed recovery
//! spine. The live module below owns no independent WAL/apply/publication authority: the commit
//! coordinator remains the sole visibility boundary.

mod bootstrap_publication;
mod bootstrap_rebuild;
mod bootstrap_rebuild_gpu;
mod digest;
mod gpu_completion;
mod input;
mod live_spine;
mod manifest;
mod publication;
mod resources;
mod status;

pub(crate) use live_spine::{SealedInt4PublicationGenerationV1, SealedInt4RebuildMetadataV1};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(super) enum DataGenerationError {
    #[error("unsupported root format {0}")]
    UnsupportedRootFormat(u16),
    #[error("zero {0}")]
    ZeroIdentity(&'static str),
    #[error("zero digest is not a logical commitment")]
    ZeroDigest,
    #[error("noncanonical ordering for {0}")]
    NonCanonicalOrder(&'static str),
    #[error("invalid {0}")]
    Invalid(&'static str),
    #[error("missing {0}")]
    Missing(&'static str),
    #[error("unexpected {0}")]
    Unexpected(&'static str),
    #[error("predecessor mismatch for {0}")]
    PredecessorMismatch(&'static str),
    #[error("GPU completion mismatch for {0}")]
    GpuCompletionMismatch(&'static str),
    #[error("checked count overflow")]
    CountOverflow,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::digest::*;
    use super::input::*;
    use super::manifest::*;
    use super::publication::*;
    use super::status::*;
    use super::DataGenerationError;

    fn gpu_root<R: SyntheticRootForTest>(seed: u64) -> R {
        R::from_synthetic(synthetic_gpu_completion_for_test(seed))
    }

    fn empty_root_pairs<R: SyntheticRootForTest>(seed: u64) -> Vec<(u8, R)> {
        (0..=64)
            .map(|depth| (depth as u8, gpu_root(seed + depth)))
            .collect()
    }

    fn empty_roots<R: SyntheticRootForTest>(seed: u64) -> GpuRadixEmptyRoots<R> {
        GpuRadixEmptyRoots::synthetic_for_test(empty_root_pairs(seed))
    }

    fn path<R: SyntheticRootForTest>(
        seed: u64,
        counts: impl Fn(u8) -> u64,
    ) -> GpuRadixPathCompletion<R> {
        GpuRadixPathCompletion {
            leaf_root: gpu_root(seed + 1_000),
            nodes: (0..64)
                .map(|depth| GpuRadixPathNode {
                    depth,
                    subtree_count: counts(depth),
                    root: gpu_root(seed + u64::from(depth)),
                })
                .collect(),
        }
    }

    fn terminal(kind: TerminalOutcomeKind) -> TerminalOutcome {
        match kind {
            TerminalOutcomeKind::CommitSuccess => TerminalOutcome {
                kind,
                affected_rows: 1,
                sqlstate: None,
                constraint_id: 0,
            },
            TerminalOutcomeKind::CommitNoOp => TerminalOutcome {
                kind,
                affected_rows: 0,
                sqlstate: None,
                constraint_id: 0,
            },
            TerminalOutcomeKind::AbortError => TerminalOutcome {
                kind,
                affected_rows: 0,
                sqlstate: Some(*b"23505"),
                constraint_id: 9,
            },
        }
    }

    fn status_input(seed: u64) -> StatusInput {
        StatusInput {
            request_digest: gpu_root(seed),
            target_digest: gpu_root(seed + 1),
            returning_digest: gpu_root(seed + 2),
            terminal_envelope_digest: gpu_root(seed + 3),
        }
    }

    fn status_path(seed: u64, counts: impl Fn(u8) -> u64) -> GpuStatusPathCompletion {
        GpuStatusPathCompletion {
            entry_leaf_root: gpu_root(seed),
            map_path: path(seed + 1, counts),
        }
    }

    struct RowSetFixture {
        builder: DataGenerationBuilder,
        predecessor: Arc<PublicationGeneration>,
        input: RootFreeGenerationInput,
        gpu: GpuRowSetCompletion,
        changed_table: StableTableId,
        retained_table: StableTableId,
        retained_table_two: StableTableId,
    }

    fn row_set_fixture() -> RowSetFixture {
        let database_id = DatabaseId::new([7; 16]).expect("database id");
        let catalog = CatalogIdentity::new(CatalogEpoch::new(3), gpu_root(10));
        let builder =
            DataGenerationBuilder::new(RootFormatVersion::V1, database_id).expect("builder");
        let genesis = builder
            .genesis(
                catalog,
                GenesisGpuCompletion {
                    empty_table_map_roots: empty_roots(100),
                    empty_status_map_roots: empty_roots(200),
                    database_root: gpu_root(300),
                },
            )
            .expect("genesis");

        let changed_table = StableTableId::new(1).expect("table id");
        let retained_table = StableTableId::new(1_u64 << 63).expect("table id");
        let retained_table_two = StableTableId::new(1_u64 << 62).expect("table id");
        let changed_index = StableIndexId::new(11).expect("index id");
        let changed_manifest = Arc::new(TableManifest {
            table_id: changed_table,
            data_generation: DataGeneration::new(1).expect("generation"),
            rows: RowMap::empty(empty_roots(400)).expect("empty row map"),
            indexes: vec![Arc::new(IndexManifest {
                owner_table_id: changed_table,
                index_id: changed_index,
                generation: IndexGeneration::new(1).expect("generation"),
                shape_root: gpu_root(450),
                entries: IndexMap::empty(empty_roots(460)).expect("empty index map"),
                root: gpu_root(470),
            })],
            root: gpu_root(500),
        });
        let retained_manifest = Arc::new(TableManifest {
            table_id: retained_table,
            data_generation: DataGeneration::new(1).expect("generation"),
            rows: RowMap::empty(empty_roots(600)).expect("empty row map"),
            indexes: Vec::new(),
            root: gpu_root(700),
        });
        let retained_manifest_two = Arc::new(TableManifest {
            table_id: retained_table_two,
            data_generation: DataGeneration::new(1).expect("generation"),
            rows: RowMap::empty(empty_roots(710)).expect("empty row map"),
            indexes: Vec::new(),
            root: gpu_root(720),
        });
        let tables = genesis
            .tables
            .substitute(
                changed_table,
                None,
                Some(TableMapLeaf {
                    manifest: Arc::clone(&changed_manifest),
                }),
                &path(800, |_| 1),
            )
            .expect("first table map path")
            .substitute(
                retained_table,
                None,
                Some(TableMapLeaf {
                    manifest: Arc::clone(&retained_manifest),
                }),
                &path(900, |depth| if depth == 0 { 2 } else { 1 }),
            )
            .expect("second table map path");
        let tables = tables
            .substitute(
                retained_table_two,
                None,
                Some(TableMapLeaf {
                    manifest: Arc::clone(&retained_manifest_two),
                }),
                &path(730, |depth| {
                    if depth == 0 {
                        3
                    } else if depth == 1 {
                        2
                    } else {
                        1
                    }
                }),
            )
            .expect("third table map path");
        let first_status = status_input(1_000);
        let status = genesis
            .status
            .insert(
                PublishedStatusEntry {
                    transaction_id: StableTransactionId::new(1).expect("transaction"),
                    request_digest: first_status.request_digest,
                    commit_sequence: CommitSequence::new(1).expect("commit"),
                    outcome: terminal(TerminalOutcomeKind::CommitSuccess),
                    target_digest: first_status.target_digest,
                    returning_digest: first_status.returning_digest,
                    terminal_envelope_digest: first_status.terminal_envelope_digest,
                },
                &status_path(1_100, |_| 1),
            )
            .expect("first status path");
        let predecessor = Arc::new(PublicationGeneration {
            root_format: RootFormatVersion::V1,
            database_id,
            publication_epoch: PublicationEpoch::new(1).expect("epoch"),
            identity: PublicationIdentity {
                visible_next: VisibleNext::new(2).expect("visible next"),
                database_root: gpu_root(1_200),
                catalog,
                status_covered_through: 1,
                status_view_root: status.root(),
                last_terminal_envelope_digest: Some(first_status.terminal_envelope_digest),
            },
            tables,
            status,
        });
        predecessor
            .validate_full_for_audit()
            .expect("synthetic predecessor");

        let row_id = StableRowId::new(42).expect("row id");
        let status = status_input(1_300);
        let input = RootFreeGenerationInput {
            database_id,
            initial_database_root: predecessor.identity.database_root,
            catalog_before: catalog,
            catalog_after: catalog,
            transaction_id: StableTransactionId::new(2).expect("transaction"),
            commit_sequence: CommitSequence::new(2).expect("commit"),
            terminal_outcome: terminal(TerminalOutcomeKind::CommitSuccess),
            status,
            table_deltas: vec![TableDelta::RowSet(RowSetInput {
                table_id: changed_table,
                before: TableBefore {
                    data_generation: changed_manifest.data_generation,
                    root: changed_manifest.root,
                    logical_row_count: 0,
                },
                final_indexes: vec![FinalIndexShape {
                    index_id: changed_index,
                    shape_root: changed_manifest.indexes[0].shape_root,
                    key_descriptors: Vec::new(),
                    before: Some(IndexBefore {
                        generation: changed_manifest.indexes[0].generation,
                        root: changed_manifest.indexes[0].root,
                    }),
                }],
                rows: vec![RowInput {
                    row_id,
                    action: RowAction::Insert,
                    before_current_row_leaf: None,
                    after: Some(AfterRow {
                        created_by: CommitSequence::new(2).expect("commit"),
                    }),
                    columns: Vec::new(),
                    index_memberships: vec![IndexMembershipInput {
                        index_id: changed_index,
                        before_entry_leaf: None,
                        after_present: true,
                        key_columns: Vec::new(),
                    }],
                }],
            })],
        };
        let gpu = GpuRowSetCompletion {
            tables: vec![GpuTableRowSetCompletion {
                table_id: changed_table,
                rows: vec![GpuRowMutationCompletion {
                    row_id,
                    after_current_row_leaf: Some(gpu_root(1_400)),
                    row_map_path: path(1_500, |_| 1),
                    index_memberships: vec![GpuIndexMembershipCompletion {
                        index_id: changed_index,
                        after_entry_leaf: Some(gpu_root(1_450)),
                        entry_map_path: Some(path(1_460, |_| 1)),
                    }],
                }],
                final_indexes: vec![GpuIndexManifestCompletion {
                    index_id: changed_index,
                    generation: IndexGeneration::new(2).expect("generation"),
                    root: gpu_root(1_470),
                }],
                data_generation: DataGeneration::new(2).expect("generation"),
                table_root: gpu_root(1_600),
                table_map_path: path(1_700, |depth| {
                    if depth == 0 {
                        3
                    } else if depth == 1 {
                        2
                    } else {
                        1
                    }
                }),
            }],
            status_path: status_path(1_800, |depth| if depth <= 62 { 2 } else { 1 }),
            final_database_root: gpu_root(1_900),
        };
        RowSetFixture {
            builder,
            predecessor,
            input,
            gpu,
            changed_table,
            retained_table,
            retained_table_two,
        }
    }

    fn row_set_fixture_with_existing_replace(
        entry_leaf: Option<IndexEntryLeafRoot>,
    ) -> (RowSetFixture, CurrentRowLeafRoot) {
        let mut fixture = row_set_fixture();
        let row_id = StableRowId::new(42).expect("fixture row id");
        let current_row_leaf: CurrentRowLeafRoot = gpu_root(1_955);
        let predecessor_table = fixture
            .predecessor
            .tables
            .table(fixture.changed_table)
            .expect("changed predecessor table")
            .clone();
        let rows = predecessor_table
            .rows
            .substitute(row_id, None, Some(current_row_leaf), &path(1_960, |_| 1))
            .expect("populate predecessor row");
        let mut index = predecessor_table.indexes[0].as_ref().clone();
        if let Some(entry_leaf) = entry_leaf {
            index.entries = index
                .entries
                .substitute(row_id, None, Some(entry_leaf), &path(2_030, |_| 1))
                .expect("populate predecessor membership");
        }
        let mut table = predecessor_table.as_ref().clone();
        table.rows = rows;
        table.indexes[0] = Arc::new(index);
        let table = Arc::new(table);
        let tables = fixture
            .predecessor
            .tables
            .substitute(
                fixture.changed_table,
                Some(&TableMapLeaf {
                    manifest: predecessor_table,
                }),
                Some(TableMapLeaf {
                    manifest: Arc::clone(&table),
                }),
                &path(2_100, |depth| {
                    if depth == 0 {
                        3
                    } else if depth == 1 {
                        2
                    } else {
                        1
                    }
                }),
            )
            .expect("replace predecessor table");
        let previous_generation = fixture.predecessor.as_ref();
        fixture.predecessor = Arc::new(PublicationGeneration {
            root_format: previous_generation.root_format,
            database_id: previous_generation.database_id,
            publication_epoch: previous_generation.publication_epoch,
            identity: previous_generation.identity.clone(),
            tables,
            status: previous_generation.status.clone(),
        });
        fixture
            .predecessor
            .validate_full_for_audit()
            .expect("populated predecessor");

        let TableDelta::RowSet(row_set) = fixture
            .input
            .table_deltas
            .first_mut()
            .expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        row_set.before.logical_row_count = 1;
        row_set.rows[0].action = RowAction::Replace;
        row_set.rows[0].before_current_row_leaf = Some(current_row_leaf);
        row_set.rows[0].index_memberships[0].before_entry_leaf = entry_leaf;
        row_set.rows[0].index_memberships[0].after_present = entry_leaf.is_some();
        let index_completion = &mut fixture.gpu.tables[0];
        index_completion.rows[0].index_memberships[0].after_entry_leaf = entry_leaf;
        index_completion.rows[0].index_memberships[0].entry_map_path = None;
        index_completion.final_indexes[0].generation = table.indexes[0].generation;
        index_completion.final_indexes[0].root = table.indexes[0].root;
        (fixture, current_row_leaf)
    }

    #[test]
    fn genesis_and_row_set_relink_only_completed_gpu_roots() {
        let fixture = row_set_fixture();
        let retained_before = fixture
            .predecessor
            .tables
            .table(fixture.retained_table)
            .expect("retained table")
            .clone();
        let retained_two_before = fixture
            .predecessor
            .tables
            .table(fixture.retained_table_two)
            .expect("second retained table")
            .clone();
        let candidate = fixture
            .builder
            .build_row_set(&fixture.predecessor, &fixture.input, fixture.gpu.clone())
            .expect("RowSet candidate");
        let changed = candidate
            .generation
            .tables
            .table(fixture.changed_table)
            .expect("changed table");
        assert_eq!(changed.rows.count(), 1);
        assert_eq!(changed.indexes[0].entries.count(), 1);
        assert_eq!(
            fixture
                .predecessor
                .tables
                .table(fixture.changed_table)
                .expect("old table")
                .rows
                .count(),
            0
        );
        assert!(Arc::ptr_eq(
            &retained_before,
            candidate
                .generation
                .tables
                .table(fixture.retained_table)
                .expect("retained successor table")
        ));
        assert!(Arc::ptr_eq(
            &retained_two_before,
            candidate
                .generation
                .tables
                .table(fixture.retained_table_two)
                .expect("second retained successor table")
        ));
        assert_eq!(candidate.generation.status.count(), 2);
        assert_eq!(candidate.generation.identity.visible_next.get(), 3);
    }

    #[test]
    fn zero_effect_index_membership_reuses_the_inherited_map_arc_and_rejects_a_path() {
        let (fixture, _current_row_leaf) = row_set_fixture_with_existing_replace(None);
        let predecessor_index = fixture
            .predecessor
            .tables
            .table(fixture.changed_table)
            .expect("old table")
            .indexes[0]
            .clone();
        let candidate = fixture
            .builder
            .build_row_set(&fixture.predecessor, &fixture.input, fixture.gpu.clone())
            .expect("zero-effect index membership candidate");
        let successor_index = &candidate
            .generation
            .tables
            .table(fixture.changed_table)
            .expect("successor table")
            .indexes[0];
        assert_eq!(successor_index.generation, predecessor_index.generation);
        assert_eq!(successor_index.root, predecessor_index.root);
        assert_eq!(
            successor_index.entries.root_arc_address_for_test(),
            predecessor_index.entries.root_arc_address_for_test()
        );
        assert!(Arc::ptr_eq(successor_index, &predecessor_index));

        let mut malformed_gpu = fixture.gpu;
        malformed_gpu.tables[0].rows[0].index_memberships[0].entry_map_path =
            Some(path(1_950, |_| 1));
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, malformed_gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "zero-effect index membership path"
            ))
        ));
    }

    #[test]
    fn replace_membership_requires_new_leaf_and_actual_predecessor_leaf() {
        let entry_leaf: IndexEntryLeafRoot = gpu_root(2_190);
        let (fixture, _current_row_leaf) = row_set_fixture_with_existing_replace(Some(entry_leaf));
        assert!(matches!(
            fixture.builder.build_row_set(
                &fixture.predecessor,
                &fixture.input,
                fixture.gpu.clone()
            ),
            Err(DataGenerationError::GpuCompletionMismatch(
                "replayed replacement index membership leaf"
            ))
        ));

        let mut missing_path_gpu = fixture.gpu.clone();
        missing_path_gpu.tables[0].rows[0].index_memberships[0].after_entry_leaf =
            Some(gpu_root(2_195));
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, missing_path_gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed index membership path"
            ))
        ));

        let mut forged_present_input = fixture.input.clone();
        let TableDelta::RowSet(row_set) = forged_present_input
            .table_deltas
            .first_mut()
            .expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        let forged_leaf: IndexEntryLeafRoot = gpu_root(2_200);
        assert_ne!(forged_leaf, entry_leaf);
        row_set.rows[0].index_memberships[0].before_entry_leaf = Some(forged_leaf);
        let mut forged_present_gpu = fixture.gpu.clone();
        forged_present_gpu.tables[0].rows[0].index_memberships[0].after_entry_leaf =
            Some(gpu_root(2_201));
        forged_present_gpu.tables[0].rows[0].index_memberships[0].entry_map_path =
            Some(path(2_202, |_| 1));
        assert!(matches!(
            fixture.builder.build_row_set(
                &fixture.predecessor,
                &forged_present_input,
                forged_present_gpu
            ),
            Err(DataGenerationError::PredecessorMismatch(
                "RowSet index membership leaf"
            ))
        ));

        let mut forged_absent_input = fixture.input.clone();
        let TableDelta::RowSet(row_set) = forged_absent_input
            .table_deltas
            .first_mut()
            .expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        row_set.rows[0].index_memberships[0].before_entry_leaf = None;
        row_set.rows[0].index_memberships[0].after_present = false;
        let mut forged_absent_gpu = fixture.gpu;
        forged_absent_gpu.tables[0].rows[0].index_memberships[0].after_entry_leaf = None;
        assert!(matches!(
            fixture.builder.build_row_set(
                &fixture.predecessor,
                &forged_absent_input,
                forged_absent_gpu
            ),
            Err(DataGenerationError::PredecessorMismatch(
                "RowSet index membership leaf"
            ))
        ));
    }

    #[test]
    fn row_set_hot_validation_visits_only_touched_radix_paths() {
        let fixture = row_set_fixture();
        // The fixture has two untouched table leaves in addition to the changed table. Reset
        // after fixture construction so this measures only the private RowSet successor build.
        reset_hot_path_node_visits_for_test();
        let candidate = fixture
            .builder
            .build_row_set(&fixture.predecessor, &fixture.input, fixture.gpu)
            .expect("bounded RowSet candidate");
        let visits = hot_path_node_visits_for_test();
        // One affected row map, index map, table map, and status map each traverse fixed 64-bit
        // paths. The small amount of predecessor lookup is likewise bounded by that fixed depth;
        // the untouched tables do not add recursive validation work.
        assert!(visits <= 512, "unexpected hot path visits: {visits}");
        candidate
            .generation
            .validate_full_for_audit()
            .expect("full audit validation");
        assert!(
            hot_path_node_visits_for_test() > visits + 100,
            "full audit did not traverse retained generations"
        );
    }

    #[test]
    fn row_set_refuses_replayed_root_substitution() {
        let fixture = row_set_fixture();
        let mut gpu = fixture.gpu.clone();
        gpu.tables[0].rows[0].row_map_path.nodes[0].root = fixture
            .predecessor
            .tables
            .table(fixture.changed_table)
            .expect("old table")
            .rows
            .root();
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed row-map root"
            ))
        ));

        let fixture = row_set_fixture();
        let mut gpu = fixture.gpu.clone();
        gpu.tables[0].rows[0].index_memberships[0]
            .entry_map_path
            .as_mut()
            .expect("changed membership path")
            .nodes[0]
            .root = fixture
            .predecessor
            .tables
            .table(fixture.changed_table)
            .expect("old table")
            .indexes[0]
            .entries
            .root();
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed index-map root"
            ))
        ));

        let fixture = row_set_fixture();
        let mut gpu = fixture.gpu.clone();
        gpu.tables[0].final_indexes[0].root = fixture
            .predecessor
            .tables
            .table(fixture.changed_table)
            .expect("old table")
            .indexes[0]
            .root;
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed index manifest"
            ))
        ));

        let fixture = row_set_fixture();
        let mut gpu = fixture.gpu.clone();
        gpu.tables[0].table_root = fixture
            .predecessor
            .tables
            .table(fixture.changed_table)
            .expect("old table")
            .root;
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed table manifest"
            ))
        ));

        let fixture = row_set_fixture();
        let mut gpu = fixture.gpu.clone();
        gpu.tables[0].table_map_path.nodes[0].root = fixture.predecessor.tables.root();
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed table-map root"
            ))
        ));

        let fixture = row_set_fixture();
        let mut gpu = fixture.gpu.clone();
        gpu.final_database_root = fixture.predecessor.identity.database_root;
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed database root"
            ))
        ));

        let fixture = row_set_fixture();
        let mut gpu = fixture.gpu.clone();
        gpu.status_path.map_path.nodes[0].root = fixture.predecessor.status.root();
        assert!(matches!(
            fixture
                .builder
                .build_row_set(&fixture.predecessor, &fixture.input, gpu),
            Err(DataGenerationError::GpuCompletionMismatch(
                "changed status-map root"
            ))
        ));
    }

    #[test]
    fn radix_rejects_substituted_wrong_depth_empty_root() {
        let row_one = StableRowId::new(1).expect("row");
        let row_three = StableRowId::new(3).expect("row");
        let map = RowMap::empty(empty_roots(2_000)).expect("empty map");
        let map = map
            .substitute(row_one, None, Some(gpu_root(2_100)), &path(2_200, |_| 1))
            .expect("initial path");
        let corrupt = map
            .corrupt_first_empty_root_for_test(1, gpu_root(2_300))
            .expect("corrupt test map");
        let substituted = corrupt
            .substitute(
                row_three,
                None,
                Some(gpu_root(2_400)),
                &path(2_500, |depth| if depth <= 62 { 2 } else { 1 }),
            )
            .expect("hot path does not recursively audit an untouched sibling");
        assert!(matches!(
            substituted.validate(),
            Err(DataGenerationError::Invalid("radix empty-node depth"))
        ));
    }

    #[test]
    fn radix_removal_requires_completed_canonical_empty_roots() {
        let row = StableRowId::new(5).expect("row");
        let row_leaf: CurrentRowLeafRoot = gpu_root(2_510);
        let empty_pairs = empty_root_pairs::<RowMapRoot>(2_520);
        let empty_leaf_root = empty_pairs[64].1;
        let map = RowMap::empty(GpuRadixEmptyRoots::synthetic_for_test(empty_pairs))
            .expect("empty row map");
        let map = map
            .substitute(row, None, Some(row_leaf), &path(2_600, |_| 1))
            .expect("insert row");
        let bad_leaf_completion = GpuRadixPathCompletion {
            leaf_root: gpu_root(2_700),
            nodes: (0..64)
                .map(|depth| GpuRadixPathNode {
                    depth,
                    subtree_count: 0,
                    root: gpu_root(2_710 + u64::from(depth)),
                })
                .collect(),
        };
        assert!(matches!(
            map.substitute(row, Some(&row_leaf), None, &bad_leaf_completion),
            Err(DataGenerationError::GpuCompletionMismatch(
                "empty radix leaf root"
            ))
        ));

        let bad_node_completion = GpuRadixPathCompletion {
            leaf_root: empty_leaf_root,
            nodes: (0..64)
                .map(|depth| GpuRadixPathNode {
                    depth,
                    subtree_count: 0,
                    root: gpu_root(2_800 + u64::from(depth)),
                })
                .collect(),
        };
        assert!(matches!(
            map.substitute(row, Some(&row_leaf), None, &bad_node_completion),
            Err(DataGenerationError::GpuCompletionMismatch(
                "empty radix node root"
            ))
        ));
    }

    #[test]
    fn gpu_commitment_debug_is_redacted() {
        let completed = synthetic_gpu_completion_for_test(0xfeed_beef);
        assert_eq!(format!("{completed:?}"), "GpuCompletedDigest(<opaque>)");
        let database_root = <DatabaseRoot as SyntheticRootForTest>::from_synthetic(completed);
        assert_eq!(format!("{database_root:?}"), "DatabaseRoot(<opaque>)");
    }

    #[test]
    fn genesis_rejects_poisoned_empty_root_depth_sets_before_map_construction() {
        let database_id = DatabaseId::new([8; 16]).expect("database id");
        let builder =
            DataGenerationBuilder::new(RootFormatVersion::V1, database_id).expect("builder");
        let catalog = CatalogIdentity::new(CatalogEpoch::new(1), gpu_root(2_510));

        let mut permuted = empty_root_pairs::<TableMapRoot>(2_520);
        permuted.swap(0, 1);
        assert!(matches!(
            builder.genesis(
                catalog,
                GenesisGpuCompletion {
                    empty_table_map_roots: GpuRadixEmptyRoots::synthetic_for_test(permuted),
                    empty_status_map_roots: empty_roots(2_600),
                    database_root: gpu_root(2_700),
                },
            ),
            Err(DataGenerationError::GpuCompletionMismatch(
                "empty radix root depth"
            ))
        ));

        let mut duplicated = empty_root_pairs::<TableMapRoot>(2_710);
        duplicated[1].1 = duplicated[0].1;
        assert!(matches!(
            builder.genesis(
                catalog,
                GenesisGpuCompletion {
                    empty_table_map_roots: GpuRadixEmptyRoots::synthetic_for_test(duplicated),
                    empty_status_map_roots: empty_roots(2_800),
                    database_root: gpu_root(2_900),
                },
            ),
            Err(DataGenerationError::GpuCompletionMismatch(
                "duplicate empty radix root"
            ))
        ));
    }

    #[test]
    fn index_key_ordinal_preserves_composite_key_grammar() {
        let mut valid = row_set_fixture().input;
        let TableDelta::RowSet(row_set) = valid.table_deltas.first_mut().expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        // The composite index grammar is `(b, a)`: b intentionally has the larger stable ID.
        // Ordinals, rather than stable identities, bind the key component order.
        let b = IndexKeyDescriptor {
            key_ordinal: 0,
            column_id: StableColumnId::new(9).expect("column b"),
            shape_root: gpu_root(2_910),
        };
        let a = IndexKeyDescriptor {
            key_ordinal: 1,
            column_id: StableColumnId::new(3).expect("column a"),
            shape_root: gpu_root(2_912),
        };
        row_set.final_indexes[0].key_descriptors = vec![b, a];
        row_set.rows[0].index_memberships[0].key_columns.extend([
            IndexKeyInput {
                key_ordinal: b.key_ordinal,
                column_id: b.column_id,
                shape_root: b.shape_root,
                typed_value_root: gpu_root(2_911),
            },
            IndexKeyInput {
                key_ordinal: a.key_ordinal,
                column_id: a.column_id,
                shape_root: a.shape_root,
                typed_value_root: gpu_root(2_913),
            },
        ]);
        assert!(valid.validate().is_ok());

        let mut swapped_components = valid.clone();
        let TableDelta::RowSet(row_set) = swapped_components
            .table_deltas
            .first_mut()
            .expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        row_set.rows[0].index_memberships[0].key_columns.swap(0, 1);
        assert!(matches!(
            swapped_components.validate(),
            Err(DataGenerationError::Invalid("index key ordinal"))
        ));

        let mut wrong_column = valid.clone();
        let TableDelta::RowSet(row_set) =
            wrong_column.table_deltas.first_mut().expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        row_set.rows[0].index_memberships[0].key_columns[0].column_id =
            StableColumnId::new(8).expect("wrong column");
        assert!(matches!(
            wrong_column.validate(),
            Err(DataGenerationError::Invalid("index key descriptor column"))
        ));

        let mut wrong_shape = valid.clone();
        let TableDelta::RowSet(row_set) =
            wrong_shape.table_deltas.first_mut().expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        row_set.rows[0].index_memberships[0].key_columns[1].shape_root = gpu_root(2_914);
        assert!(matches!(
            wrong_shape.validate(),
            Err(DataGenerationError::Invalid("index key descriptor shape"))
        ));

        let mut swapped_ordinals = valid;
        let TableDelta::RowSet(row_set) = swapped_ordinals
            .table_deltas
            .first_mut()
            .expect("RowSet input")
        else {
            panic!("fixture has a RowSet");
        };
        let keys = &mut row_set.rows[0].index_memberships[0].key_columns;
        keys[0].key_ordinal = 1;
        keys[1].key_ordinal = 0;
        assert!(matches!(
            swapped_ordinals.validate(),
            Err(DataGenerationError::Invalid("index key ordinal"))
        ));
    }

    #[test]
    fn root_free_validation_keeps_catalog_only_noop_and_rejects_bad_delta_forms() {
        let fixture = row_set_fixture();
        let mut no_op = fixture.input.clone();
        no_op.terminal_outcome = terminal(TerminalOutcomeKind::CommitNoOp);
        no_op.table_deltas.clear();
        assert!(no_op.validate().is_ok());

        let mut no_op_catalog_substitution = no_op.clone();
        no_op_catalog_substitution.catalog_after =
            CatalogIdentity::new(CatalogEpoch::new(4), gpu_root(2_550));
        assert!(matches!(
            no_op_catalog_substitution.validate(),
            Err(DataGenerationError::Invalid(
                "non-success catalog transition"
            ))
        ));

        let mut no_op_affected_rows = no_op.clone();
        no_op_affected_rows.terminal_outcome.affected_rows = 1;
        assert!(matches!(
            no_op_affected_rows.validate(),
            Err(DataGenerationError::Invalid("commit outcome"))
        ));

        let mut aborted = fixture.input.clone();
        aborted.terminal_outcome = terminal(TerminalOutcomeKind::AbortError);
        assert!(matches!(
            aborted.validate(),
            Err(DataGenerationError::Invalid("abort generation mutations"))
        ));

        aborted.table_deltas.clear();
        aborted.catalog_after = CatalogIdentity::new(CatalogEpoch::new(4), gpu_root(2_551));
        assert!(matches!(
            aborted.validate(),
            Err(DataGenerationError::Invalid(
                "non-success catalog transition"
            ))
        ));

        aborted.catalog_after = aborted.catalog_before;
        aborted.terminal_outcome.affected_rows = 1;
        assert!(matches!(
            aborted.validate(),
            Err(DataGenerationError::Invalid("abort affected rows"))
        ));

        let before_index = IndexBefore {
            generation: IndexGeneration::new(1).expect("generation"),
            root: gpu_root(2_600),
        };
        let mut create = no_op.clone();
        create.terminal_outcome = terminal(TerminalOutcomeKind::CommitSuccess);
        create.table_deltas = vec![TableDelta::CreateEmpty(CreateEmptyInput {
            table_id: fixture.changed_table,
            final_indexes: vec![FinalIndexShape {
                index_id: StableIndexId::new(1).expect("index"),
                shape_root: gpu_root(2_601),
                key_descriptors: Vec::new(),
                before: Some(before_index),
            }],
        })];
        assert!(matches!(
            create.validate(),
            Err(DataGenerationError::Invalid("new final index predecessor"))
        ));

        let mut rebuild = no_op;
        rebuild.terminal_outcome = terminal(TerminalOutcomeKind::CommitSuccess);
        rebuild.table_deltas = vec![TableDelta::Rebuild(RebuildInput {
            table_id: fixture.changed_table,
            before: None,
            final_indexes: Vec::new(),
            rows: Vec::new(),
        })];
        assert!(matches!(
            rebuild.validate(),
            Err(DataGenerationError::Invalid("empty Rebuild"))
        ));

        let mut rebuild_remove = fixture.input.clone();
        rebuild_remove.table_deltas = vec![TableDelta::Rebuild(RebuildInput {
            table_id: fixture.changed_table,
            before: Some(TableBefore {
                data_generation: DataGeneration::new(1).expect("generation"),
                root: gpu_root(2_602),
                logical_row_count: 1,
            }),
            final_indexes: Vec::new(),
            rows: vec![RowInput {
                row_id: StableRowId::new(1).expect("row"),
                action: RowAction::Remove,
                before_current_row_leaf: Some(gpu_root(2_603)),
                after: None,
                columns: Vec::new(),
                index_memberships: Vec::new(),
            }],
        })];
        assert!(matches!(
            rebuild_remove.validate(),
            Err(DataGenerationError::Invalid("Rebuild row action"))
        ));

        let mut create_rebuild_with_prior_row = fixture.input.clone();
        create_rebuild_with_prior_row.table_deltas = vec![TableDelta::Rebuild(RebuildInput {
            table_id: fixture.changed_table,
            before: None,
            final_indexes: Vec::new(),
            rows: vec![RowInput {
                row_id: StableRowId::new(2).expect("row"),
                action: RowAction::RebuildCurrent,
                before_current_row_leaf: Some(gpu_root(2_604)),
                after: Some(AfterRow {
                    created_by: CommitSequence::new(2).expect("commit"),
                }),
                columns: Vec::new(),
                index_memberships: Vec::new(),
            }],
        })];
        assert!(matches!(
            create_rebuild_with_prior_row.validate(),
            Err(DataGenerationError::Invalid("new Rebuild row predecessor"))
        ));

        let mut rebuild_with_new_index_prior_leaf = fixture.input.clone();
        rebuild_with_new_index_prior_leaf.table_deltas = vec![TableDelta::Rebuild(RebuildInput {
            table_id: fixture.changed_table,
            before: Some(TableBefore {
                data_generation: DataGeneration::new(1).expect("generation"),
                root: gpu_root(2_605),
                logical_row_count: 1,
            }),
            final_indexes: vec![FinalIndexShape {
                index_id: StableIndexId::new(2).expect("index"),
                shape_root: gpu_root(2_606),
                key_descriptors: Vec::new(),
                before: None,
            }],
            rows: vec![RowInput {
                row_id: StableRowId::new(3).expect("row"),
                action: RowAction::RebuildCurrent,
                before_current_row_leaf: Some(gpu_root(2_607)),
                after: Some(AfterRow {
                    created_by: CommitSequence::new(2).expect("commit"),
                }),
                columns: Vec::new(),
                index_memberships: vec![IndexMembershipInput {
                    index_id: StableIndexId::new(2).expect("index"),
                    before_entry_leaf: Some(gpu_root(2_608)),
                    after_present: true,
                    key_columns: Vec::new(),
                }],
            }],
        })];
        assert!(matches!(
            rebuild_with_new_index_prior_leaf.validate(),
            Err(DataGenerationError::Invalid(
                "new Rebuild index membership predecessor"
            ))
        ));

        let mut rebuild_new_row_with_old_index_leaf = fixture.input.clone();
        rebuild_new_row_with_old_index_leaf.table_deltas =
            vec![TableDelta::Rebuild(RebuildInput {
                table_id: fixture.changed_table,
                before: Some(TableBefore {
                    data_generation: DataGeneration::new(1).expect("generation"),
                    root: gpu_root(2_609),
                    logical_row_count: 1,
                }),
                final_indexes: vec![FinalIndexShape {
                    index_id: StableIndexId::new(3).expect("index"),
                    shape_root: gpu_root(2_610),
                    key_descriptors: Vec::new(),
                    before: Some(IndexBefore {
                        generation: IndexGeneration::new(1).expect("generation"),
                        root: gpu_root(2_611),
                    }),
                }],
                rows: vec![RowInput {
                    row_id: StableRowId::new(4).expect("row"),
                    action: RowAction::RebuildCurrent,
                    before_current_row_leaf: None,
                    after: Some(AfterRow {
                        created_by: CommitSequence::new(2).expect("commit"),
                    }),
                    columns: Vec::new(),
                    index_memberships: vec![IndexMembershipInput {
                        index_id: StableIndexId::new(3).expect("index"),
                        before_entry_leaf: Some(gpu_root(2_612)),
                        after_present: true,
                        key_columns: Vec::new(),
                    }],
                }],
            })];
        assert!(matches!(
            rebuild_new_row_with_old_index_leaf.validate(),
            Err(DataGenerationError::Invalid(
                "new row index membership predecessor"
            ))
        ));

        let mut reset = fixture.input;
        reset.table_deltas = vec![TableDelta::ResetEmpty(ResetEmptyInput {
            table_id: fixture.changed_table,
            before: TableBefore {
                data_generation: DataGeneration::new(1).expect("generation"),
                root: gpu_root(2_613),
                logical_row_count: 0,
            },
            final_indexes: Vec::new(),
        })];
        assert!(matches!(
            reset.validate(),
            Err(DataGenerationError::Invalid("empty ResetEmpty predecessor"))
        ));
    }
}
