use super::*;

#[test]
fn execute_mvcc_query_sorts_missing_provenance_frames_deterministically() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(4, "SET standalone:1=Loose").unwrap();
    e.execute_text(5, "SET standalone:2=Leaf").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FullScan,
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::KeyPrefix("standalone:".to_string()),
                MvccReadFilter::KeyPrefix("team:alpha:".to_string()),
            ])),
            order: Some(MvccReadOrder::ProvenanceKeyDesc {
                frame: MvccProvenanceFrame::TerminalInput,
            }),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("standalone:1".to_string()),
                value: Some("Loose".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("standalone:2".to_string()),
                value: Some("Leaf".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_labeled_branch_projection_and_ordering() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                branches: vec![
                    MvccLabeledValueChainBranch {
                        label: "members".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                    },
                    MvccLabeledValueChainBranch {
                        label: "team".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::BranchLabelAsc),
            projection: MvccProjection::BranchLabelTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alpha Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Beta Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team".to_string()),
                value: Some("Alpha Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team".to_string()),
                value: Some("Beta Team".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_preserves_labeled_branch_identity_and_first_match_filtering() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Shared Team").unwrap();
    e.execute_text(6, "SET team:beta=Shared Team").unwrap();

    let distinct = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::FollowValueChainLabeledBranches {
                        keys: vec!["acct:1".to_string()],
                        branches: vec![MvccLabeledValueChainBranch {
                            label: "primary".to_string(),
                            plan: MvccValueChainPlan {
                                value_key_hops: 2,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                        }],
                        fan_in: MvccValueChainBranchFanIn::AllBranches,
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChainLabeledBranches {
                        keys: vec!["acct:1".to_string()],
                        branches: vec![MvccLabeledValueChainBranch {
                            label: "fallback".to_string(),
                            plan: MvccValueChainPlan {
                                value_key_hops: 2,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                        }],
                        fan_in: MvccValueChainBranchFanIn::AllBranches,
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: None,
            order: Some(MvccReadOrder::BranchLabelAsc),
            projection: MvccProjection::BranchLabelTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        distinct.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("fallback".to_string()),
                value: Some("Shared Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("primary".to_string()),
                value: Some("Shared Team".to_string()),
            },
        ]
    );

    let first_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                branches: vec![
                    MvccLabeledValueChainBranch {
                        label: "team".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                    },
                    MvccLabeledValueChainBranch {
                        label: "members".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::BranchLabelEquals("team".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_match.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_symmetric_difference_all_source_composition() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET user:1=active").unwrap();
    e.execute_text(9, "SET user:2=locked").unwrap();

    let exact_imbalance = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceAll {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:1".to_string(),
                            "user:2".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_imbalance.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:2".to_string()),
                value: None,
            },
        ]
    );

    let join_imbalance = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceAll {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        join_imbalance.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_refs_join_adjacent_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=active").unwrap();
    e.execute_text(4, "SET profile:2=suspended").unwrap();
    e.execute_text(5, "SET acct:3=missing").unwrap();
    e.execute_text(6, "SET acct:1=profile:3").unwrap();
    e.execute_text(7, "SET profile:3=closed").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefs {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("active".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:3".to_string()),
                value: Some("closed".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefs {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("closed".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:3".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_prefixes_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=order:1:").unwrap();
    e.execute_text(2, "SET acct:2=order:2:").unwrap();
    e.execute_text(3, "SET order:1:a=paid").unwrap();
    e.execute_text(4, "SET order:1:b=packed").unwrap();
    e.execute_text(5, "SET order:2:a=queued").unwrap();
    e.execute_text(6, "SET order:3:a=orphan").unwrap();
    e.execute_text(7, "SET acct:3=missing:").unwrap();
    e.execute_text(8, "SET acct:1=order:1b:").unwrap();
    e.execute_text(9, "SET order:1b:a=shipped").unwrap();
    e.execute_text(10, "SET order:1b:b=delivered").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("queued".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: Some("shipped".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: Some("delivered".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("delivered".to_string()),
                MvccReadFilter::ValueEquals("queued".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_prefixes_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=order:1:").unwrap();
    e.execute_text(5, "SET profile:2=order:2:").unwrap();
    e.execute_text(6, "SET order:1:a=paid").unwrap();
    e.execute_text(7, "SET order:1:b=packed").unwrap();
    e.execute_text(8, "SET order:2:a=queued").unwrap();
    e.execute_text(9, "SET order:2:b=delivered").unwrap();
    e.execute_text(10, "SET order:3:a=orphan").unwrap();
    e.execute_text(11, "SET profile:1=order:1b:").unwrap();
    e.execute_text(12, "SET order:1b:a=shipped").unwrap();
    e.execute_text(13, "SET order:1b:b=cancelled").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 13 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("queued".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:b".to_string()),
                value: Some("delivered".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: Some("shipped".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: Some("cancelled".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 13 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("queued".to_string()),
                MvccReadFilter::ValueEquals("shipped".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_refs_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET acct:4=profile:4").unwrap();
    e.execute_text(5, "SET profile:1=team:1").unwrap();
    e.execute_text(6, "SET profile:2=team:2").unwrap();
    e.execute_text(7, "SET profile:4=missing-team").unwrap();
    e.execute_text(8, "SET team:1=gold").unwrap();
    e.execute_text(9, "SET team:2=silver").unwrap();
    e.execute_text(10, "SET team:3=bronze").unwrap();
    e.execute_text(11, "SET profile:1=team:3").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                    "acct:4".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:2".to_string()),
                value: Some("silver".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:3".to_string()),
                value: Some("bronze".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("silver".to_string()),
                MvccReadFilter::ValueEquals("bronze".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:3".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_prefixes_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET acct:4=profile:4").unwrap();
    e.execute_text(5, "SET profile:1=team:1").unwrap();
    e.execute_text(6, "SET profile:2=team:2").unwrap();
    e.execute_text(7, "SET profile:4=missing-team").unwrap();
    e.execute_text(8, "SET team:1=order:1:").unwrap();
    e.execute_text(9, "SET team:2=order:2:").unwrap();
    e.execute_text(10, "SET order:1:a=paid").unwrap();
    e.execute_text(11, "SET order:1:b=packed").unwrap();
    e.execute_text(12, "SET order:2:a=queued").unwrap();
    e.execute_text(13, "SET order:2:b=shipped").unwrap();
    e.execute_text(14, "SET order:3:a=orphan").unwrap();
    e.execute_text(15, "SET profile:1=team:3").unwrap();
    e.execute_text(16, "SET team:3=order:3:").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                    "acct:4".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("queued".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:b".to_string()),
                value: Some("shipped".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:3:a".to_string()),
                value: Some("orphan".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:2".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("orphan".to_string()),
                MvccReadFilter::ValueEquals("queued".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:3:a".to_string()),
                value: None,
            },
        ]
    );

    let join_side_projection = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: None,
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        join_side_projection.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:b".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:3:a".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_join_side_projection_keeps_non_join_shapes_stable() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_join_side_source_filters() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::SourceKeyPrefix("acct:1".to_string()),
                MvccReadFilter::SourceValueEquals("profile:1".to_string()),
                MvccReadFilter::KeyPrefix("team:alpha".to_string()),
            ])),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(1),
        })
        .unwrap();

    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha:2".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
}

#[test]
fn execute_mvcc_query_source_filters_are_empty_for_non_join_shapes() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: Some(MvccReadFilter::SourceValueEquals("open".to_string())),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert!(result.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_join_side_source_ordering() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:2=profile:2").unwrap();
    e.execute_text(2, "SET acct:1=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();
    e.execute_text(7, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

    let source_key_ordered = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        source_key_ordered.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );

    let source_value_ordered = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::SourceValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        source_value_ordered.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_source_ordering_keeps_non_join_shapes_stable() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:2=locked").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyDesc),
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_mixed_join_side_projection_controls() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:2=profile:2").unwrap();
    e.execute_text(2, "SET acct:1=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();

    let source_key_target_value = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_key_target_value.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("Bob".to_string()),
            },
        ]
    );

    let source_value_only = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:2".to_string())),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::SourceValueOnly,
            limit: Some(1),
        })
        .unwrap();

    assert_eq!(
        source_value_only.rows,
        vec![MvccReadRow {
            source_key: Some("acct:2".to_string()),
            key: None,
            value: Some("profile:2".to_string()),
        }]
    );
}

#[test]
fn execute_mvcc_query_mixed_join_projection_keeps_non_join_shapes_stable() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();

    let source_key_target_value = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 1 },
            filter: None,
            order: None,
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_key_target_value.rows,
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: Some("open".to_string()),
        }]
    );

    let source_value_only = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 1 },
            filter: None,
            order: None,
            projection: MvccProjection::SourceValueOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_value_only.rows,
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: None,
        }]
    );
}

#[test]
fn execute_mvcc_query_supports_composite_filter_shapes() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();
    e.execute_text(4, "SET user:2=locked").unwrap();

    let all_filter = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::ValueEquals("locked".to_string()),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();
    assert_eq!(
        all_filter.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:2".to_string()),
            value: Some("locked".to_string()),
        }]
    );

    let any_filter = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();
    assert_eq!(
        any_filter.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_key_range_filter_shapes() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();
    e.execute_text(4, "SET acct:4=suspended").unwrap();

    let ranged = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyRange {
                start_inclusive: "acct:2".to_string(),
                end_exclusive: "acct:4".to_string(),
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ranged.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("locked".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_limit_after_filtering() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET acct:3=locked").unwrap();

    let limited = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        limited.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_key_ordering_before_limit() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:2=locked").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();

    let descending = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        descending.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_value_ordering_before_limit() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:2=locked").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:4=closed").unwrap();
    e.execute_text(4, "SET acct:3=closed").unwrap();

    let ascending = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        ascending.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("locked".to_string()),
            },
        ]
    );

    let descending = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        descending.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn publish_telemetry_emits_snapshot_to_sink() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET a=1").unwrap();

    let mut sink = InMemoryTelemetrySink::default();
    e.publish_telemetry(&mut sink);

    assert_eq!(sink.snapshots().len(), 1);
    let snapshot = &sink.snapshots()[0];
    assert_eq!(snapshot.role, Role::Leader);
    assert_eq!(snapshot.replication_lag.commit_index, 1);
    assert_eq!(snapshot.replication_lag.applied_index, 1);
    assert_eq!(snapshot.replication_lag.visible_index, 1);
    assert_eq!(snapshot.replication_lag.commit_apply_gap, 0);
    assert_eq!(snapshot.replication_lag.apply_visible_gap, 0);
    assert_eq!(snapshot.runtime_metrics.commits_total, 1);
    assert_eq!(snapshot.snapshot_id, 0);
    assert_eq!(snapshot.wal_flushed_count, 1);
    assert_eq!(snapshot.wal_last_durable_txn_id, Some(1));
    assert_eq!(snapshot.wal_buffered_count, 1);
    assert_eq!(snapshot.wal_unflushed_count, 0);
    assert_eq!(snapshot.pending_batch_len, 0);
    assert_eq!(snapshot.active_txn_count, 0);
    assert_eq!(snapshot.backlog_blocker_count, 0);
    assert!(!snapshot.has_backlog_blockers());
    assert!(snapshot.quiescent_for_failover);
    assert!(snapshot.gpu_parity_fallbacks.is_empty());
}

#[test]
fn installing_older_snapshot_is_a_status_no_op() {
    let mut e = Engine::new_local_cpu_oracle();
    let committed = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();
    let baseline = e.status_snapshot();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index.saturating_sub(1),
        last_included_term: 1,
        snapshot_id: 99,
    });

    let marks = e.replication_watermarks();
    assert_eq!(marks.commit_index, committed.index);
    assert_eq!(marks.applied_index, committed.index);
    assert_eq!(marks.visible_index, committed.index);
    assert_eq!(marks.snapshot_id, baseline.snapshot.snapshot_id);
    assert_eq!(e.status_snapshot(), baseline);
    assert_eq!(e.visible_up_to(), committed.index);
    assert_eq!(e.get("a").as_deref(), Some("1"));
}

#[test]
fn installing_higher_index_lower_term_snapshot_is_a_status_no_op() {
    let mut e = Engine::new_local_cpu_oracle();
    let committed = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index + 2,
        last_included_term: 3,
        snapshot_id: 11,
    });
    let baseline = e.status_snapshot();
    let baseline_marks = e.replication_watermarks();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index + 3,
        last_included_term: 2,
        snapshot_id: 99,
    });

    let marks = e.replication_watermarks();
    assert_eq!(marks, baseline_marks);
    assert_eq!(e.status_snapshot(), baseline);
    assert_eq!(e.visible_up_to(), baseline.snapshot.visible_index);
    assert_eq!(e.get("a").as_deref(), Some("1"));
}

#[test]
fn installing_advanced_snapshot_replaces_snapshot_identity_exactly() {
    let mut e = Engine::new_local_cpu_oracle();
    let committed = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index,
        last_included_term: 1,
        snapshot_id: 11,
    });
    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index + 2,
        last_included_term: 2,
        snapshot_id: 4,
    });

    let marks = e.replication_watermarks();
    let status = e.status_snapshot();
    assert_eq!(marks.snapshot_id, 4);
    assert_eq!(status.snapshot.snapshot_id, 4);
    assert_eq!(status.snapshot.last_included_index, committed.index + 2);
    assert_eq!(status.snapshot.last_included_term, 2);
}
