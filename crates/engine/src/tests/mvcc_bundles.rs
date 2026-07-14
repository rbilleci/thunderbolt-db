use super::*;
mod provenance_bundle_paths;

#[test]
fn execute_mvcc_query_supports_whole_bundle_cardinality_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let full_path_len = e
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_to_ordinal_match = e
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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
        .execute_mvcc_query(&MvccReadQuery {
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

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_occurrence_offset_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_to_ordinal_match = e
        .execute_mvcc_query(&MvccReadQuery {
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
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start: 0,
                    occurrence_start: 4,
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
        .execute_mvcc_query(&MvccReadQuery {
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
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start_min: 0,
                    first_start_max: 0,
                    occurrence_start_min: 4,
                    occurrence_start_max: 4,
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
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                occurrence_start: 2,
                last_start: 4,
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

    assert_eq!(ordinal_to_last_match.rows, first_to_ordinal_match.rows);

    let ordinal_to_last_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                occurrence_start_min: 2,
                occurrence_start_max: 2,
                last_start_min: 4,
                last_start_max: 4,
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
        ordinal_to_last_range_match.rows,
        first_to_ordinal_match.rows
    );

    let truncated_first_to_ordinal_miss = e
        .execute_mvcc_query(&MvccReadQuery {
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
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start: 0,
                    occurrence_start: 4,
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
        .execute_mvcc_query(&MvccReadQuery {
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
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start_min: 1,
                    first_start_max: 1,
                    occurrence_start_min: 5,
                    occurrence_start_max: 6,
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
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                occurrence_start_min: 3,
                occurrence_start_max: 2,
                last_start_min: 5,
                last_start_max: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 3,
                occurrence_start: 2,
                last_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_offset_projection_and_ordering() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:noop=profile:noop").unwrap();
    e.execute_text(4, "SET profile:noop=acct:noop").unwrap();

    let source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:noop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
        ],
    };

    let first_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("0".to_string()),
            },
        ]
    );

    let last_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("4".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let ordinal_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Nth(1),
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Nth(1),
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let missing_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["missing:key".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["missing:key".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        missing_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_distance_projection_and_ordering() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:noop=profile:noop").unwrap();
    e.execute_text(4, "SET profile:noop=acct:noop").unwrap();

    let source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:noop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
        ],
    };

    let same_subpath_distances = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        same_subpath_distances.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("4".to_string()),
            },
        ]
    );

    let mixed_distances = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceDesc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        mixed_distances.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("5".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let missing_distances = e
        .execute_mvcc_query(&MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["missing:left".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["missing:right".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["missing:left".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["missing:right".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        missing_distances.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_bundle_mixed_occurrence_offset_pair_projection_and_ordering() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:noop=profile:noop").unwrap();
    e.execute_text(4, "SET profile:noop=acct:noop").unwrap();

    let source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:noop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
        ],
    };

    let ascending_pairs = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(
                MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:loop".to_string()],
                    left_occurrence: MvccProvenanceOccurrence::First,
                    right_expected: vec!["profile:loop".to_string()],
                    right_occurrence: MvccProvenanceOccurrence::Last,
                },
            ),
            projection: MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ascending_pairs.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("0,5".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("1,4".to_string()),
            },
        ]
    );

    let descending_pairs = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(
                MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairDesc {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:loop".to_string()],
                    left_occurrence: MvccProvenanceOccurrence::First,
                    right_expected: vec!["profile:loop".to_string()],
                    right_occurrence: MvccProvenanceOccurrence::Last,
                },
            ),
            projection: MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        descending_pairs.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("1,4".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("0,5".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let missing_pairs = e
        .execute_mvcc_query(&MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(
                MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["missing:left".to_string()],
                    left_occurrence: MvccProvenanceOccurrence::First,
                    right_expected: vec!["missing:right".to_string()],
                    right_occurrence: MvccProvenanceOccurrence::Last,
                },
            ),
            projection: MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["missing:left".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["missing:right".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        missing_pairs.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_bundle_ordinal_pair_occurrence_offset_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let ordinal_pair_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start: 2,
                right_occurrence_index: 2,
                right_start: 4,
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
            ordinal_pair_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some(
                    "acct:loop -> profile:loop -> acct:loop -> profile:loop -> acct:loop -> profile:loop"
                        .to_string(),
                ),
            }]
        );

    let ordinal_pair_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start_min: 2,
                left_start_max: 2,
                right_occurrence_index: 2,
                right_start_min: 4,
                right_start_max: 4,
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

    assert_eq!(ordinal_pair_range_match.rows, ordinal_pair_match.rows);

    let truncated_ordinal_pair_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start: 2,
                right_occurrence_index: 2,
                right_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_ordinal_pair_miss.rows.is_empty());

    let wrong_ordinal_pair_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start_min: 3,
                left_start_max: 3,
                right_occurrence_index: 2,
                right_start_min: 5,
                right_start_max: 6,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_pair_range_miss.rows.is_empty());

    let inverted_ordinal_pair_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start_min: 3,
                left_start_max: 2,
                right_occurrence_index: 2,
                right_start_min: 5,
                right_start_max: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_pair_range_miss.rows.is_empty());

    let wrong_ordinal_pair_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start: 2,
                right_occurrence_index: 3,
                right_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_pair_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_mixed_occurrence_distance_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();

    let mixed_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    distance: 2,
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
        mixed_occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("acct:1 -> profile:1 -> team:alpha -> acct:1".to_string()),
        }]
    );

    let mixed_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 1,
                    max_distance: 2,
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
        mixed_occurrence_distance_range_match.rows,
        mixed_occurrence_distance_match.rows
    );

    let truncated_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_mixed_occurrence_distance_miss.rows.is_empty());

    let truncated_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 1,
                    max_distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());

    let wrong_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_mixed_occurrence_distance_miss.rows.is_empty());

    let wrong_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 0,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_mixed_occurrence_distance_range_miss.rows.is_empty());

    let inverted_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 3,
                    max_distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());

    let reversed_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_mixed_occurrence_distance_miss.rows.is_empty());

    let reversed_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    min_distance: 1,
                    max_distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_mixed_occurrence_distance_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_mixed_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
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
        first_mixed_occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_mixed_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 1,
                    max_distance: 1,
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
        first_mixed_occurrence_distance_range_match.rows,
        first_mixed_occurrence_distance_match.rows
    );

    let last_mixed_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
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
        last_mixed_occurrence_distance_match.rows,
        first_mixed_occurrence_distance_match.rows
    );

    let last_mixed_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 1,
                    max_distance: 1,
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
        last_mixed_occurrence_distance_range_match.rows,
        first_mixed_occurrence_distance_match.rows
    );

    let truncated_last_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_last_mixed_occurrence_distance_miss
        .rows
        .is_empty());

    let wrong_first_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 2,
                    max_distance: 3,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());

    let reversed_last_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_last_mixed_occurrence_distance_miss.rows.is_empty());

    let inverted_last_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 2,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_last_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_mixed_occurrence_distance_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_to_ordinal_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 4,
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
        first_to_ordinal_mixed_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_to_ordinal_mixed_range_match = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 5,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: Some(
                    MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
                        bundle: MvccProvenanceFrameBundle::FullPath,
                        summary: MvccProvenanceSummary::KeyPath,
                        left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                        right_occurrence_index: 1,
                        right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                        min_distance: 4,
                        max_distance: 4,
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
        first_to_ordinal_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
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
        ordinal_to_last_mixed_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 1,
                    max_distance: 1,
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
        ordinal_to_last_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let truncated_first_to_ordinal_mixed_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_first_to_ordinal_mixed_miss.rows.is_empty());

    let wrong_first_to_ordinal_mixed_range_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 5,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: Some(
                    MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
                        bundle: MvccProvenanceFrameBundle::FullPath,
                        summary: MvccProvenanceSummary::KeyPath,
                        left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                        right_occurrence_index: 1,
                        right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                        min_distance: 5,
                        max_distance: 6,
                    },
                ),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

    assert!(wrong_first_to_ordinal_mixed_range_miss.rows.is_empty());

    let inverted_ordinal_to_last_mixed_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 2,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_mixed_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_mixed_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 2,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_mixed_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_mixed_occurrence_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_to_ordinal_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 0,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
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
        first_to_ordinal_mixed_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_to_ordinal_mixed_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 0,
                    left_start_max: 0,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
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
        first_to_ordinal_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
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
        ordinal_to_last_mixed_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 3,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
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
        ordinal_to_last_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let truncated_first_to_ordinal_mixed_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 0,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_first_to_ordinal_mixed_miss.rows.is_empty());

    let wrong_first_to_ordinal_mixed_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 1,
                    left_start_max: 1,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 5,
                    right_start_max: 6,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_to_ordinal_mixed_range_miss.rows.is_empty());

    let inverted_ordinal_to_last_mixed_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 4,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 5,
                    right_start_max: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_mixed_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_mixed_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 2,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_mixed_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_ordinal_mixed_occurrence_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let ordinal_mixed_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
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
        ordinal_mixed_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let ordinal_mixed_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start_min: 3,
                left_start_max: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start_min: 4,
                right_start_max: 4,
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
        ordinal_mixed_occurrence_range_match.rows,
        ordinal_mixed_occurrence_match.rows
    );

    let truncated_ordinal_mixed_occurrence_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_ordinal_mixed_occurrence_miss.rows.is_empty());

    let wrong_ordinal_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start_min: 0,
                left_start_max: 2,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start_min: 4,
                right_start_max: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_mixed_occurrence_range_miss.rows.is_empty());

    let inverted_ordinal_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start_min: 4,
                left_start_max: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start_min: 4,
                right_start_max: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_mixed_occurrence_range_miss.rows.is_empty());

    let wrong_ordinal_mixed_occurrence_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_mixed_occurrence_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_mixed_occurrence_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_mixed_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 0,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 1,
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
        first_mixed_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_mixed_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 0,
                    left_start_max: 0,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 1,
                    right_start_max: 1,
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
        first_mixed_occurrence_range_match.rows,
        first_mixed_occurrence_match.rows
    );

    let last_mixed_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
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
        last_mixed_occurrence_match.rows,
        first_mixed_occurrence_match.rows
    );

    let last_mixed_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 3,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
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
        last_mixed_occurrence_range_match.rows,
        first_mixed_occurrence_match.rows
    );

    let truncated_last_mixed_occurrence_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_last_mixed_occurrence_miss.rows.is_empty());

    let wrong_first_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 1,
                    left_start_max: 2,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 1,
                    right_start_max: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_mixed_occurrence_range_miss.rows.is_empty());

    let inverted_last_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 4,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_last_mixed_occurrence_range_miss.rows.is_empty());
}
