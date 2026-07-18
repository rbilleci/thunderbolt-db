use crate::{
    Engine, MvccProjection, MvccProvenanceFrameBundle, MvccProvenanceSummary, MvccReadFilter,
    MvccReadOrder, MvccReadQuery, MvccReadRow, MvccReadSource, MvccSourceProvenance,
    MvccValueChainPlan, MvccValueChainTerminal, StorageVisibility,
};

#[test]
fn execute_mvcc_query_supports_whole_bundle_cardinality_filters() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let full_path_len = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 3,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        full_path_len.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop".to_string()),
        }]
    );

    let truncated_bundle_len = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected_len: 2,
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        truncated_bundle_len.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop".to_string()),
        }]
    );

    let len_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(len_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_and_nth_occurrence_filters() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_occurrence_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let first_occurrence_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 0,
                start_max: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_range_match.rows,
        first_occurrence_match.rows
    );

    let last_occurrence_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop".to_string()),
        }]
    );

    let last_occurrence_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 0,
                start_max: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(last_occurrence_range_match.rows, last_occurrence_match.rows);

    let nth_occurrence_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        nth_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let wrong_first_occurrence_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_occurrence_miss.rows.is_empty());

    let wrong_first_occurrence_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 1,
                start_max: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_occurrence_range_miss.rows.is_empty());

    let wrong_last_occurrence_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["profile:loop".to_string(), "acct:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_last_occurrence_miss.rows.is_empty());

    let wrong_last_occurrence_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 0,
                start_max: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_last_occurrence_range_miss.rows.is_empty());

    let inverted_first_occurrence_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 1,
                start_max: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_first_occurrence_range_miss.rows.is_empty());

    let wrong_nth_occurrence_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 2,
                start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_nth_occurrence_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_range_filters() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 2,
                start_max: 3,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        occurrence_range_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let truncated_occurrence_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 0,
                start_max: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_occurrence_range_miss.rows.is_empty());

    let wrong_occurrence_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 3,
                start_max: 3,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_occurrence_range_miss.rows.is_empty());

    let inverted_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 3,
                start_max: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_range_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_distance_filters() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_distance_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let truncated_occurrence_distance_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_occurrence_distance_miss.rows.is_empty());

    let wrong_occurrence_distance_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_occurrence_distance_miss.rows.is_empty());

    let reversed_occurrence_distance_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                right_occurrence_index: 0,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_occurrence_distance_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_distance_range_filters() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_distance_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 2,
                    max_distance: 3,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        occurrence_distance_range_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let truncated_occurrence_distance_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 2,
                    max_distance: 3,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_occurrence_distance_range_miss.rows.is_empty());

    let wrong_occurrence_distance_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 3,
                    max_distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_occurrence_distance_range_miss.rows.is_empty());

    let inverted_occurrence_distance_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 3,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_occurrence_distance_range_miss.rows.is_empty());

    let reversed_occurrence_distance_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    right_occurrence_index: 0,
                    min_distance: 2,
                    max_distance: 3,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_occurrence_distance_range_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_occurrence_distance_filters() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_occurrence_distance_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let first_occurrence_distance_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 2,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_distance_range_match.rows,
        first_occurrence_distance_match.rows
    );

    let last_occurrence_distance_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_occurrence_distance_match.rows,
        first_occurrence_distance_match.rows
    );

    let last_occurrence_distance_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 2,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_occurrence_distance_range_match.rows,
        first_occurrence_distance_match.rows
    );

    let truncated_last_occurrence_distance_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_last_occurrence_distance_miss.rows.is_empty());

    let wrong_first_occurrence_distance_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 3,
                    max_distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_occurrence_distance_range_miss.rows.is_empty());

    let inverted_last_occurrence_distance_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 3,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_last_occurrence_distance_range_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_occurrence_distance_filters() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_to_ordinal_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
            first_to_ordinal_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some(
                    "acct:loop -> profile:loop -> acct:loop -> profile:loop -> acct:loop -> profile:loop"
                        .to_string(),
                ),
            }]
        );

    let first_to_ordinal_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    min_distance: 4,
                    max_distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_to_ordinal_range_match.rows,
        first_to_ordinal_match.rows
    );

    let ordinal_to_last_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 1,
                    distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(ordinal_to_last_match.rows, first_to_ordinal_match.rows);

    let ordinal_to_last_range_match = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 1,
                    min_distance: 2,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_to_last_range_match.rows,
        first_to_ordinal_match.rows
    );

    let truncated_first_to_ordinal_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_first_to_ordinal_miss.rows.is_empty());

    let wrong_first_to_ordinal_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    min_distance: 5,
                    max_distance: 6,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_to_ordinal_range_miss.rows.is_empty());

    let inverted_ordinal_to_last_range_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 1,
                    min_distance: 3,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_index_miss = e
        .evaluate_mvcc_query_specification(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 3,
                    distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_index_miss.rows.is_empty());
}
