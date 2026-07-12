use super::*;

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_filters() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();
    e.execute_text(7, "SET team:beta:2=Bianca").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 7 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceKeyPrefix {
                frame: MvccProvenanceFrame::Seed,
                prefix: "acct:".to_string(),
            },
            MvccReadFilter::ProvenanceValueEquals {
                frame: MvccProvenanceFrame::TerminalInput,
                expected: "team:beta".to_string(),
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_source_relative_filters() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChainLabeledBranches {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            branches: vec![MvccLabeledValueChainBranch {
                label: "team".to_string(),
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
            }],
            fan_in: MvccValueChainBranchFanIn::AllBranches,
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::SourceKeyPrefix("acct:".to_string()),
            MvccReadFilter::SourceValueEquals("profile:1".to_string()),
            MvccReadFilter::BranchLabelEquals("team".to_string()),
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("Alpha Team".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_source_relative_order() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=member:1").unwrap();
    e.execute_text(4, "SET member:1=Alice").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChainLabeledBranches {
            keys: vec!["acct:1".to_string()],
            branches: vec![
                MvccLabeledValueChainBranch {
                    label: "member".to_string(),
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
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
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::BranchLabelDesc),
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("member:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("member:1".to_string()),
                value: Some("Alice".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_source_relative_projection() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=Alpha Team").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChainLabeledBranches {
            keys: vec!["acct:1".to_string()],
            branches: vec![MvccLabeledValueChainBranch {
                label: "team".to_string(),
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
            }],
            fan_in: MvccValueChainBranchFanIn::AllBranches,
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: None,
        projection: MvccProjection::TargetKeySourceValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_concat_cpu_resolved_sources() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_bundle_filters() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 8 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceBundleKeyEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected: "profile:2".to_string(),
            },
            MvccReadFilter::ProvenanceBundleKeyPrefix {
                bundle: MvccProvenanceFrameBundle::FullPath,
                prefix: "profile:".to_string(),
            },
            MvccReadFilter::ProvenanceBundleValueEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected: "team:beta".to_string(),
            },
            MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                key: "profile:2".to_string(),
                value: "team:beta".to_string(),
                min_count: 1,
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_bundle_path_filters() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 8 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceBundlePathContains {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["profile:2".to_string(), "team:beta".to_string()],
            },
            MvccReadFilter::ProvenanceBundlePathSegmentEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
                index: 1,
                expected: "team:beta".to_string(),
            },
            MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::ValuePath,
                expected: vec!["team:beta".to_string(), "member:2".to_string()],
            },
            MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 3,
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_bundle_occurrence_path_filters(
) {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 3,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 5 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 3,
                expected: vec!["acct:1".to_string()],
            },
            MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                right_occurrence_index: 0,
                right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                distance: 2,
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_runs_nested_native_composition() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::Concat {
                    sources: vec![MvccReadSource::ConcatDistinct {
                        sources: vec![
                            MvccReadSource::KeyLookup {
                                key: "acct:1".to_string(),
                            },
                            MvccReadSource::KeyLookup {
                                key: "acct:2".to_string(),
                            },
                        ],
                    }],
                },
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            },
            &FirstCudaSliceParityBackend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
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
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_replays_deterministic_workload_fixture_for_point_lookup() {
    let e = Engine::new_local_cpu_oracle();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-read-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let historical = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::ValueOnly,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &historical, 1);

    assert_eq!(
        historical.rows,
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: Some("open".to_string()),
        }]
    );

    let current = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "user:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::ValueEquals("active".to_string())),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &current, 2);

    assert_eq!(
        current.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:1".to_string()),
            value: None,
        }]
    );
}

#[test]
fn execute_mvcc_query_replays_deterministic_full_scan_workload_fixture() {
    let e = Engine::new_local_cpu_oracle();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let historical = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::Any(vec![
                    MvccReadFilter::ValueEquals("open".to_string()),
                    MvccReadFilter::ValueEquals("hold".to_string()),
                ]),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &historical, 1);
    assert_eq!(
        historical.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );

    let current = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::Any(vec![
                    MvccReadFilter::ValueEquals("closed".to_string()),
                    MvccReadFilter::ValueEquals("archived".to_string()),
                ]),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &current, 2);
    assert_eq!(
        current.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("archived".to_string()),
            },
        ]
    );

    let status = e.status_snapshot();
    assert_eq!(status.fallback.gpu_parity_fallback_total(), 2);
    assert_eq!(
        status.latest_fallback_reason(),
        Some(FallbackReason::GpuMvccReadParityGap)
    );
}

#[test]
fn execute_mvcc_query_replays_deterministic_source_composition_workload_fixture() {
    let e = Engine::new_local_cpu_oracle();
    for (txn_id, command) in
        include_str!("../../../../tests/fixtures/mvcc-source-composition-workload.txt")
            .lines()
            .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let multiset_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectAll {
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
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &multiset_overlap, 1);

    assert_eq!(
        multiset_overlap.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:1".to_string()),
            value: None,
        }]
    );

    let multiset_imbalance = e
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
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &multiset_imbalance, 2);

    assert_eq!(
        multiset_imbalance.rows,
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

    let status = e.status_snapshot();
    assert_eq!(status.fallback.gpu_parity_fallback_total(), 2);
    assert_eq!(
        status.latest_fallback_reason(),
        Some(FallbackReason::GpuMvccReadParityGap)
    );
}

#[test]
fn execute_mvcc_query_supports_multi_key_lookup_fan_in_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();
    e.execute_text(4, "SET acct:1=closed").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec![
                    "user:1".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:2".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
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

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "user:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("closed".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: Some("active".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_concat_source_composition() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(9, "SET user:1=active").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "user:1".to_string(),
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:2".to_string(), "missing".to_string()],
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
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    },
                    MvccReadSource::KeyLookup {
                        key: "user:1".to_string(),
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::SourceKeyPrefix("acct:1".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
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
fn execute_mvcc_query_supports_concat_distinct_source_composition() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(9, "SET user:1=active").unwrap();

    let deduped = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "user:1".to_string(),
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "acct:2".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
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
        deduped.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: None,
            },
        ]
    );

    let source_distinction = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_distinction.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_intersect_distinct_source_composition() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:alpha").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(9, "SET team:beta:1=Bob").unwrap();
    e.execute_text(10, "SET user:1=active").unwrap();

    let exact_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectDistinct {
                sources: vec![
                    MvccReadSource::Concat {
                        sources: vec![
                            MvccReadSource::KeyLookup {
                                key: "user:1".to_string(),
                            },
                            MvccReadSource::KeyBatchLookup {
                                keys: vec!["acct:2".to_string(), "user:1".to_string()],
                            },
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_overlap.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );

    let source_sensitive_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:3".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:3".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: Some(MvccReadFilter::ValueEquals("Alice".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_sensitive_overlap.rows,
        vec![MvccReadRow {
            source_key: Some("acct:3".to_string()),
            key: Some("team:alpha:1".to_string()),
            value: Some("profile:3".to_string()),
        }]
    );
}

#[test]
fn execute_mvcc_query_supports_except_distinct_source_composition() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:alpha").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(9, "SET team:beta:1=Bob").unwrap();
    e.execute_text(10, "SET user:1=active").unwrap();
    e.execute_text(11, "SET user:2=locked").unwrap();

    let exact_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptDistinct {
                sources: vec![
                    MvccReadSource::Concat {
                        sources: vec![
                            MvccReadSource::KeyBatchLookup {
                                keys: vec![
                                    "user:1".to_string(),
                                    "user:2".to_string(),
                                    "acct:2".to_string(),
                                ],
                            },
                            MvccReadSource::KeyLookup {
                                key: "user:1".to_string(),
                            },
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_difference.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:2".to_string()),
            value: None,
        }]
    );

    let source_sensitive_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:3".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:3".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
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
        source_sensitive_difference.rows,
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
fn execute_mvcc_query_supports_symmetric_difference_distinct_source_composition() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:alpha").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(9, "SET team:beta:1=Bob").unwrap();
    e.execute_text(10, "SET user:1=active").unwrap();
    e.execute_text(11, "SET user:2=locked").unwrap();
    e.execute_text(12, "SET user:3=standby").unwrap();

    let exact_uniques = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceDistinct {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:2".to_string(),
                            "user:2".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "user:3".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 12 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_uniques.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:3".to_string()),
                value: None,
            },
        ]
    );

    let source_sensitive_uniques = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:3".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:3".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 12 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Ally".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_sensitive_uniques.rows,
        vec![
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
}

#[test]
fn execute_mvcc_query_supports_intersect_all_source_composition() {
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

    let exact_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectAll {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:2".to_string(),
                            "user:1".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "user:1".to_string()],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "user:2".to_string()],
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
        exact_overlap.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:1".to_string()),
            value: None,
        }]
    );

    let join_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectAll {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
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
        join_overlap.rows,
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
fn execute_mvcc_query_supports_except_all_source_composition() {
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

    let exact_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptAll {
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
        exact_difference.rows,
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

    let join_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptAll {
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
        join_difference.rows,
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
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_prefixes_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team:alpha:").unwrap();
    e.execute_text(9, "SET prefix:beta:=team:beta:").unwrap();
    e.execute_text(10, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(11, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(12, "SET team:beta:1=Bob").unwrap();
    e.execute_text(13, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(14, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(15, "SET prefix:alpha:v2=team:alpha:v2:")
        .unwrap();
    e.execute_text(16, "SET team:alpha:v2:1=Astra").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
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
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_value_key_refs_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=person:1").unwrap();
    e.execute_text(11, "SET team-lead:2=person:2").unwrap();
    e.execute_text(12, "SET person:1=Alice").unwrap();
    e.execute_text(13, "SET person:2=Bob").unwrap();
    e.execute_text(14, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(15, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(16, "SET team-lead:3=person:3").unwrap();
    e.execute_text(17, "SET person:3=Astra").unwrap();
    e.execute_text(18, "SET team-root:3=prefix:ghost:").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 18 },
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
                key: Some("person:2".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("person:3".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 18 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("person:2".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("person:3".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("person:3".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_value_key_prefixes_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=team:alpha:").unwrap();
    e.execute_text(11, "SET team-lead:2=team:beta:").unwrap();
    e.execute_text(12, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(13, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(14, "SET team:beta:1=Bob").unwrap();
    e.execute_text(15, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(16, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(17, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(18, "SET team-lead:3=team:alpha:v2:")
        .unwrap();
    e.execute_text(19, "SET team:alpha:v2:1=Astra").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 19 },
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
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 19 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_value_key_ref_prefixes_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=group-prefix:alpha:")
        .unwrap();
    e.execute_text(11, "SET team-lead:2=group-prefix:beta:")
        .unwrap();
    e.execute_text(12, "SET group-prefix:alpha:=squad:alpha:")
        .unwrap();
    e.execute_text(13, "SET group-prefix:beta:=squad:beta:")
        .unwrap();
    e.execute_text(14, "SET squad:alpha:1=Alice").unwrap();
    e.execute_text(15, "SET squad:alpha:2=Ally").unwrap();
    e.execute_text(16, "SET squad:beta:1=Bob").unwrap();
    e.execute_text(17, "SET squad:beta:2=Bianca").unwrap();
    e.execute_text(18, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(19, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(20, "SET team-lead:3=group-prefix:alpha:v2:")
        .unwrap();
    e.execute_text(21, "SET group-prefix:alpha:v2:=squad:alpha:v2:")
        .unwrap();
    e.execute_text(22, "SET squad:alpha:v2:1=Astra").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 22 },
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
                key: Some("squad:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("squad:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("squad:alpha:v2:1".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 22 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("squad:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("squad:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("squad:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_generic_follow_value_chain_plan() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=group-prefix:alpha:")
        .unwrap();
    e.execute_text(11, "SET team-lead:2=group-prefix:beta:")
        .unwrap();
    e.execute_text(12, "SET group-prefix:alpha:=squad:alpha:")
        .unwrap();
    e.execute_text(13, "SET group-prefix:beta:=squad:beta:")
        .unwrap();
    e.execute_text(14, "SET squad:alpha:1=Alice").unwrap();
    e.execute_text(15, "SET squad:alpha:2=Ally").unwrap();
    e.execute_text(16, "SET squad:beta:1=Bob").unwrap();
    e.execute_text(17, "SET squad:beta:2=Bianca").unwrap();
    e.execute_text(18, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(19, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(20, "SET team-lead:3=group-prefix:alpha:v2:")
        .unwrap();
    e.execute_text(21, "SET group-prefix:alpha:v2:=squad:alpha:v2:")
        .unwrap();
    e.execute_text(22, "SET squad:alpha:v2:1=Astra").unwrap();
    e.execute_text(23, "SET squad:alpha:v2:1=talent:1").unwrap();
    e.execute_text(24, "SET squad:beta:1=talent:2").unwrap();
    e.execute_text(25, "SET talent:1=Architect").unwrap();
    e.execute_text(26, "SET talent:2=Builder").unwrap();

    let specialized_prefix = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    let generic_prefix = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(generic_prefix.rows, specialized_prefix.rows);

    let specialized_terminal = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    let generic_terminal = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(generic_terminal.rows, specialized_terminal.rows);
}

#[test]
fn execute_mvcc_query_supports_follow_value_chain_branches_source() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(7, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();
    e.execute_text(9, "SET team:beta:2=Bianca").unwrap();

    let branch_grouped = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "acct:3".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        branch_grouped.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("Ally".to_string()),
            },
        ]
    );

    let branch_concat = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec![
                            "acct:2".to_string(),
                            "acct:1".to_string(),
                            "acct:3".to_string(),
                        ],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec![
                            "acct:2".to_string(),
                            "acct:1".to_string(),
                            "acct:3".to_string(),
                        ],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_ne!(branch_concat.rows, branch_grouped.rows);
    assert_eq!(
        branch_concat.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("Ally".to_string()),
            },
        ]
    );

    let ordered_projection = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Ally".to_string()),
                MvccReadFilter::ValueEquals("team:beta".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyDesc),
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordered_projection.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_chain_branch_first_non_empty_fan_in() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET acct:4=profile:4").unwrap();
    e.execute_text(5, "SET profile:1=team:alpha").unwrap();
    e.execute_text(6, "SET profile:2=team:beta").unwrap();
    e.execute_text(7, "SET profile:4=team:delta").unwrap();
    e.execute_text(8, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(9, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(10, "SET team:beta:1=Bob").unwrap();
    e.execute_text(11, "SET team:delta:1=Dora").unwrap();

    let first_non_empty = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:3".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "acct:4".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: None,
            order: None,
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_non_empty.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:4".to_string()),
                key: Some("acct:4".to_string()),
                value: Some("Dora".to_string()),
            },
        ]
    );

    let all_branches = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:3".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "acct:4".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("team:alpha".to_string()),
                MvccReadFilter::ValueEquals("team:beta".to_string()),
                MvccReadFilter::ValueEquals("team:delta".to_string()),
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
                MvccReadFilter::ValueEquals("Dora".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        all_branches.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:4".to_string()),
                key: Some("acct:4".to_string()),
                value: Some("team:delta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:4".to_string()),
                key: Some("acct:4".to_string()),
                value: Some("Dora".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_chain_terminal_input_provenance() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::SourceValueEquals("team:beta".to_string())),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_branch_fan_in_with_terminal_input_provenance() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:gamma").unwrap();
    e.execute_text(7, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();
    e.execute_text(9, "SET team:beta:2=Bianca").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:3".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("profile:".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}
