use super::*;

#[test]
fn execute_mvcc_query_runs_visibility_filtered_scan_through_execution_layer() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=pending").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(
        result.fallback_reason,
        Some(FallbackReason::GpuMvccReadParityGap)
    );
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(
        e.metrics()
            .fallback_for(FallbackReason::GpuMvccReadParityGap),
        1
    );
}

#[derive(Debug, Clone)]
struct RecordingMvccBackend {
    executed_target: DeviceTarget,
    rows: Vec<MvccReadRow>,
}

impl MvccExecutionBackend for RecordingMvccBackend {
    fn execute(&self, _query: &MvccReadQuery, _rows: Vec<ResolvedMvccRow>) -> MvccBackendDispatch {
        MvccBackendDispatch::Executed(MvccBackendExecution {
            executed_target: self.executed_target,
            rows: self.rows.clone(),
        })
    }
}

fn first_cuda_slice_support_query() -> MvccReadQuery {
    MvccReadQuery {
        source: MvccReadSource::KeyLookup {
            key: "acct:1".to_string(),
        },
        visibility: StorageVisibility { read_txn_id: 1 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::KeyPrefix("acct:".to_string()),
            MvccReadFilter::ValueEquals("open".to_string()),
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    }
}

#[test]
fn first_cuda_slice_query_gap_accepts_supported_shape() {
    let query = first_cuda_slice_support_query();

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_first_cuda_slice_query(&query));
}

#[test]
fn cuda_native_full_scan_query_requires_full_scan_first_slice_shape() {
    let mut query = first_cuda_slice_support_query();
    assert!(is_cuda_native_source_query(&query));
    assert!(!is_cuda_native_full_scan_query(&query));

    query.source = MvccReadSource::FullScan;
    assert!(is_cuda_native_source_query(&query));
    assert!(is_cuda_native_full_scan_query(&query));

    query.order = Some(MvccReadOrder::ValueAsc);
    assert!(is_cuda_native_source_query(&query));
    assert!(is_cuda_native_full_scan_query(&query));
}

#[test]
fn first_cuda_slice_query_gap_accepts_key_order_for_native_single_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FullScan;
    query.order = Some(MvccReadOrder::KeyDesc);

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::KeyBatchLookup {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query.source = MvccReadSource::FullScan;
    query.order = Some(MvccReadOrder::ValueAsc);
    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_concat_of_native_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:2".to_string(), "acct:3".to_string()],
            },
        ],
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::Concat {
        sources: vec![MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        }],
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));
}

#[test]
fn first_cuda_slice_query_gap_accepts_native_follow_value_chain_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 1,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = None;
    query.order = Some(MvccReadOrder::SourceKeyAsc);
    query.projection = MvccProjection::TargetKeySourceValue;

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 1,
            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::FollowValueChainBranches {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
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
        fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
        provenance: MvccSourceProvenance::Seed,
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::FollowValueChainLabeledBranches {
        keys: vec!["acct:1".to_string()],
        branches: vec![MvccLabeledValueChainBranch {
            label: "members".to_string(),
            plan: MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
        }],
        fan_in: MvccValueChainBranchFanIn::AllBranches,
        provenance: MvccSourceProvenance::TerminalInput,
    };
    query.filter = Some(MvccReadFilter::BranchLabelEquals("members".to_string()));
    query.order = Some(MvccReadOrder::BranchLabelAsc);
    query.projection = MvccProjection::BranchLabelTargetValue;
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));
}

#[test]
fn first_cuda_slice_query_gap_reports_first_unsupported_boundary() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::Concat {
        sources: vec![MvccReadSource::ConcatDistinct {
            sources: vec![MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            }],
        }],
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query = first_cuda_slice_support_query();
    query.order = Some(MvccReadOrder::BranchLabelAsc);
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::UnsupportedOrder)
    );

    query = first_cuda_slice_support_query();
    query.projection = MvccProjection::SourceValueOnly;
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::UnsupportedProjection)
    );

    query = first_cuda_slice_support_query();
    query.limit = Some(1);
    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_reports_filter_shape_gaps() {
    let mut query = first_cuda_slice_support_query();
    query.filter = Some(MvccReadFilter::All(vec![]));
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::EmptyLogicalFilterTree)
    );

    query = first_cuda_slice_support_query();
    query.filter = Some(MvccReadFilter::Any(vec![MvccReadFilter::All(vec![
        MvccReadFilter::KeyPrefix("acct:".to_string()),
        MvccReadFilter::SourceKeyPrefix("seed:".to_string()),
    ])]));
    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query = first_cuda_slice_support_query();
    query.filter = Some(MvccReadFilter::BranchLabelEquals("fallback".to_string()));
    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_filter_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 1,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::ProvenanceKeyPrefix {
            frame: MvccProvenanceFrame::Seed,
            prefix: "acct:".to_string(),
        },
        MvccReadFilter::ProvenanceValueEquals {
            frame: MvccProvenanceFrame::TerminalInput,
            expected: "team:alpha".to_string(),
        },
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query.filter = Some(MvccReadFilter::KeyPrefix("team:".to_string()));
    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_bundle_path_filters() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 2,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::ProvenanceBundlePathContains {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            expected: vec!["profile:2".to_string(), "team:beta".to_string()],
        },
        MvccReadFilter::ProvenanceBundlePathSegmentEquals {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            index: 1,
            expected: "profile:2".to_string(),
        },
        MvccReadFilter::ProvenanceBundleLenEquals {
            bundle: MvccProvenanceFrameBundle::FullPath,
            expected_len: 3,
        },
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_bundle_occurrence_path_filters() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 3,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::All(vec![
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
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_projection_and_order_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 2,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::ProvenanceBundleLenEquals {
        bundle: MvccProvenanceFrameBundle::FullPath,
        expected_len: 3,
    });
    query.order = Some(MvccReadOrder::ProvenanceValueDesc {
        frame: MvccProvenanceFrame::TerminalInput,
    });
    query.projection = MvccProjection::TargetKeyProvenanceBundleSummary {
        bundle: MvccProvenanceFrameBundle::FullPath,
        summary: MvccProvenanceSummary::KeyPath,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query.source = MvccReadSource::FullScan;
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::UnsupportedOrder)
    );
}

#[test]
fn first_cuda_slice_query_gap_accepts_all_order_projection_variants_over_resolved_source() {
    let resolved_source = MvccReadSource::FollowValueChainLabeledBranches {
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
    };
    let supported_cpu_resolved_filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::SourceKeyPrefix("acct:".to_string()),
        MvccReadFilter::ProvenanceBundleLenEquals {
            bundle: MvccProvenanceFrameBundle::FullPath,
            expected_len: 3,
        },
    ]));
    let occurrence_order_expected = vec!["acct:1".to_string()];
    let mixed_order_left_expected = vec!["acct:1".to_string(), "profile:1".to_string()];
    let mixed_order_right_expected = vec!["team:alpha".to_string()];
    let orders = vec![
        MvccReadOrder::KeyAsc,
        MvccReadOrder::KeyDesc,
        MvccReadOrder::ValueAsc,
        MvccReadOrder::ValueDesc,
        MvccReadOrder::BranchLabelAsc,
        MvccReadOrder::BranchLabelDesc,
        MvccReadOrder::SourceKeyAsc,
        MvccReadOrder::SourceKeyDesc,
        MvccReadOrder::SourceValueAsc,
        MvccReadOrder::SourceValueDesc,
        MvccReadOrder::ProvenanceKeyAsc {
            frame: MvccProvenanceFrame::Seed,
        },
        MvccReadOrder::ProvenanceKeyDesc {
            frame: MvccProvenanceFrame::TerminalInput,
        },
        MvccReadOrder::ProvenanceValueAsc {
            frame: MvccProvenanceFrame::Seed,
        },
        MvccReadOrder::ProvenanceValueDesc {
            frame: MvccProvenanceFrame::TerminalInput,
        },
        MvccReadOrder::ProvenanceBundleKeyPathAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
        },
        MvccReadOrder::ProvenanceBundleKeyPathDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
        },
        MvccReadOrder::ProvenanceBundleValuePathAsc {
            bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
        },
        MvccReadOrder::ProvenanceBundleValuePathDesc {
            bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            expected: occurrence_order_expected.clone(),
            occurrence: MvccProvenanceOccurrence::First,
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::ValuePath,
            expected: occurrence_order_expected.clone(),
            occurrence: MvccProvenanceOccurrence::Last,
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            left_expected: occurrence_order_expected.clone(),
            left_occurrence: MvccProvenanceOccurrence::First,
            right_expected: mixed_order_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::Nth(0),
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyValuePath,
            left_expected: occurrence_order_expected.clone(),
            left_occurrence: MvccProvenanceOccurrence::Last,
            right_expected: mixed_order_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::First,
        },
        MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            left_expected: mixed_order_left_expected.clone(),
            left_occurrence: MvccProvenanceOccurrence::First,
            right_expected: mixed_order_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::Last,
        },
        MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::ValuePath,
            left_expected: mixed_order_left_expected,
            left_occurrence: MvccProvenanceOccurrence::Nth(1),
            right_expected: mixed_order_right_expected,
            right_occurrence: MvccProvenanceOccurrence::First,
        },
    ];

    for order in orders {
        let mut query = first_cuda_slice_support_query();
        query.source = resolved_source.clone();
        query.filter = supported_cpu_resolved_filter.clone();
        query.order = Some(order);

        assert_eq!(
            first_cuda_slice_query_gap(&query),
            None,
            "{:?}",
            query.order
        );
    }

    let occurrence_projection_expected = vec!["acct:1".to_string()];
    let mixed_projection_left_expected = vec!["acct:1".to_string(), "profile:1".to_string()];
    let mixed_projection_right_expected = vec!["team:alpha".to_string()];
    let projections = vec![
        MvccProjection::KeyValue,
        MvccProjection::KeyOnly,
        MvccProjection::ValueOnly,
        MvccProjection::BranchLabelTargetValue,
        MvccProjection::SourceKeyTargetValue,
        MvccProjection::SourceValueOnly,
        MvccProjection::TargetKeySourceValue,
        MvccProjection::TargetKeyProvenanceValue {
            frame: MvccProvenanceFrame::TerminalInput,
        },
        MvccProjection::TargetKeyProvenanceSummary {
            summary: MvccProvenanceSummary::KeyValuePath,
        },
        MvccProjection::TargetKeyProvenanceBundleSummary {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
        },
        MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            expected: occurrence_projection_expected.clone(),
            occurrence: MvccProvenanceOccurrence::First,
        },
        MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::ValuePath,
            left_expected: occurrence_projection_expected,
            left_occurrence: MvccProvenanceOccurrence::First,
            right_expected: mixed_projection_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::Last,
        },
        MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyValuePath,
            left_expected: mixed_projection_left_expected,
            left_occurrence: MvccProvenanceOccurrence::Nth(1),
            right_expected: mixed_projection_right_expected,
            right_occurrence: MvccProvenanceOccurrence::First,
        },
    ];

    for projection in projections {
        let mut query = first_cuda_slice_support_query();
        query.source = resolved_source.clone();
        query.filter = supported_cpu_resolved_filter.clone();
        query.projection = projection;

        assert_eq!(
            first_cuda_slice_query_gap(&query),
            None,
            "{:?}",
            query.projection
        );
    }
}

#[test]
fn first_cuda_slice_query_gap_accepts_source_relative_filters_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChainLabeledBranches {
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
    };
    query.filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::SourceKeyPrefix("acct:".to_string()),
        MvccReadFilter::SourceValueEquals("profile:1".to_string()),
        MvccReadFilter::BranchLabelEquals("team".to_string()),
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_source_relative_order_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChainLabeledBranches {
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
    };
    query.filter = Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string()));
    query.order = Some(MvccReadOrder::BranchLabelDesc);

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_source_relative_projection_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChainLabeledBranches {
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
    };
    query.filter = Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string()));
    query.projection = MvccProjection::TargetKeySourceValue;

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_concat_of_cpu_resolved_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::Concat {
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
    };
    query.filter = Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string()));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_gap_labels_are_stable_for_docs_and_future_routing() {
    assert_eq!(
        FirstCudaSliceGap::UnsupportedSource.label(),
        "unsupported_source"
    );
    assert_eq!(
        FirstCudaSliceGap::UnsupportedOrder.label(),
        "unsupported_order"
    );
    assert_eq!(
        FirstCudaSliceGap::UnsupportedProjection.label(),
        "unsupported_projection"
    );
    assert_eq!(
        FirstCudaSliceGap::UnsupportedFilter.label(),
        "unsupported_filter"
    );
    assert_eq!(
        FirstCudaSliceGap::EmptyLogicalFilterTree.label(),
        "empty_logical_filter_tree"
    );
}

#[test]
fn execute_mvcc_query_keeps_result_contract_stable_across_backend_swap() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();

    let backend = RecordingMvccBackend {
        executed_target: DeviceTarget::Gpu(0),
        rows: vec![MvccReadRow {
            source_key: Some("seed:acct".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("open".to_string()),
        }],
    };

    let result = e
        .execute_mvcc_query_with_backend(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 1 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &backend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(result.rows, backend.rows);
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn cuda_native_full_scan_resolution_feeds_all_versions_to_visibility_kernel() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let kv = e.read_state.mvcc.load_kv();
    let rows = resolve_mvcc_all_versions(kv.get(), StorageVisibility { read_txn_id: 2 }).unwrap();
    let identities = rows
        .iter()
        .map(|row| {
            (
                row.tuple.key.as_str(),
                row.tuple.value.as_str(),
                row.tuple.created_by,
                row.tuple.deleted_by,
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        identities,
        vec![
            ("acct:1", "open", 1, Some(3)),
            ("acct:1", "closed", 3, None),
            ("acct:2", "hold", 2, Some(4)),
        ]
    );
}

#[test]
fn cuda_native_full_scan_fallback_re_resolves_cpu_visible_rows() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
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
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_key_lookup_fallback_re_resolves_cpu_visible_row() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:1".to_string()),
            value: Some("open".to_string()),
        }]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_key_batch_fallback_preserves_request_order() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_composition_fallback_re_resolves_cpu_visible_rows() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::IntersectAll {
                    sources: vec![
                        MvccReadSource::KeyBatchLookup {
                            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                        },
                        MvccReadSource::KeyBatchLookup {
                            keys: vec!["acct:2".to_string(), "acct:3".to_string()],
                        },
                    ],
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:2".to_string()),
            value: None,
        }]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_follow_value_chain_fallback_re_resolves_cpu_visible_rows() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET profile:1=team:beta").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::TargetKeySourceValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("profile:1".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_runs_supported_lookup_without_fallback() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 1 },
                filter: Some(MvccReadFilter::ValueEquals("open".to_string())),
                order: None,
                projection: MvccProjection::ValueOnly,
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
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: Some("open".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_runs_supported_full_scan_without_fallback() {
    let mut e = Engine::new_local_cpu_oracle();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
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
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("archived".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_orders_cpu_resolved_rows_by_value_without_fallback()
{
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
                order: Some(MvccReadOrder::ValueDesc),
                projection: MvccProjection::KeyValue,
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
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn cuda_mvcc_backend_falls_back_when_driver_is_unavailable() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    let backend = CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0);

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 1 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &backend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(result.rows.len(), 1);
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_historical_key_lookup_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:1".to_string()),
            value: Some("open".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_key_batch_lookup_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:3".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    let kv = e.read_state.mvcc.load_kv();
    let all_version_rows =
        resolve_mvcc_all_versions(kv.get(), StorageVisibility { read_txn_id: 3 }).unwrap();
    let compact_rows = ["acct:3", "acct:1"]
        .iter()
        .flat_map(|key| {
            all_version_rows
                .iter()
                .filter(move |row| row.tuple.key == *key)
                .cloned()
        })
        .collect::<Vec<_>>();
    let compact_h2d_bytes = cuda_mvcc_row_batch_transfer_bytes(&compact_rows);

    assert_eq!(e.metrics().snapshot().h2d_bytes_total, compact_h2d_bytes);
    assert!(compact_h2d_bytes < cuda_mvcc_row_batch_transfer_bytes(&all_version_rows));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_follow_value_chain_source_resolution_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET profile:1=team:alpha-v2").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_concat_native_sources_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();
    e.execute_text(4, "SET acct:4=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "acct:2".to_string(),
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:4".to_string(), "acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_concat_limit_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();
    e.execute_text(4, "SET acct:4=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "acct:2".to_string(),
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:4".to_string(), "acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_key_order_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_value_order_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_fan_in_order_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=pending").unwrap();
    e.execute_text(4, "SET acct:4=archived").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::KeyLookup {
                        key: "acct:4".to_string(),
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyOnly,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_ordered_limit_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::KeyValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_general_key_range_filter_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:0=cold").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:7=hold").unwrap();
    e.execute_text(4, "SET acct:9=closed").unwrap();
    e.execute_text(5, "SET user:1=active").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::KeyRange {
                start_inclusive: "acct:1".to_string(),
                end_exclusive: "acct:9".to_string(),
            }),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
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
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:7".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_prefix_equivalent_key_range_filter_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyRange {
                start_inclusive: "acct:".to_string(),
                end_exclusive: "acct;".to_string(),
            }),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
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
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_key_prefix_filter_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
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
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_string_value_equals_filter_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ValueEquals("open".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
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
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_numeric_value_equals_filter_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=7").unwrap();
    e.execute_text(2, "SET acct:2=8").unwrap();
    e.execute_text(3, "SET acct:3=7").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ValueEquals("7".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
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
                value: Some("7".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("7".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_filterless_full_scan_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
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
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_historical_visibility_mask_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
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
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_logical_supported_filters_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:0=cold").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:2=hold").unwrap();
    e.execute_text(4, "SET acct:3=closed").unwrap();
    e.execute_text(5, "SET user:1=open").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyRange {
                    start_inclusive: "acct:1".to_string(),
                    end_exclusive: "acct:4".to_string(),
                },
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

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
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
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_filters_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();
    e.execute_text(7, "SET team:beta:2=Bianca").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
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
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_source_relative_filters_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("Alpha Team".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_source_relative_order_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=member:1").unwrap();
    e.execute_text(4, "SET member:1=Alice").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
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
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_cpu_resolved_key_value_order_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
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
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_source_relative_projection_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=Alpha Team").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_concat_cpu_resolved_sources_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
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
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_distinct_cpu_resolved_sources_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

fn seed_native_composition_rows(e: &mut Engine) {
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();
}

fn native_set_composition_cases() -> Vec<(&'static str, MvccReadSource, Vec<&'static str>)> {
    let left = MvccReadSource::KeyBatchLookup {
        keys: vec![
            "acct:1".to_string(),
            "acct:2".to_string(),
            "acct:2".to_string(),
        ],
    };
    let right = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::KeyLookup {
                key: "acct:2".to_string(),
            },
            MvccReadSource::KeyLookup {
                key: "acct:3".to_string(),
            },
        ],
    };
    let sources = || vec![left.clone(), right.clone()];

    vec![
        (
            "concat_distinct",
            MvccReadSource::ConcatDistinct { sources: sources() },
            vec!["acct:1", "acct:2", "acct:3"],
        ),
        (
            "intersect_distinct",
            MvccReadSource::IntersectDistinct { sources: sources() },
            vec!["acct:2"],
        ),
        (
            "intersect_all",
            MvccReadSource::IntersectAll { sources: sources() },
            vec!["acct:2"],
        ),
        (
            "except_distinct",
            MvccReadSource::ExceptDistinct { sources: sources() },
            vec!["acct:1"],
        ),
        (
            "except_all",
            MvccReadSource::ExceptAll { sources: sources() },
            vec!["acct:1", "acct:2"],
        ),
        (
            "symmetric_difference_distinct",
            MvccReadSource::SymmetricDifferenceDistinct { sources: sources() },
            vec!["acct:1", "acct:3"],
        ),
        (
            "symmetric_difference_all",
            MvccReadSource::SymmetricDifferenceAll { sources: sources() },
            vec!["acct:1", "acct:2", "acct:3"],
        ),
    ]
}

fn key_only_rows(keys: &[&str]) -> Vec<MvccReadRow> {
    keys.iter()
        .map(|key| MvccReadRow {
            source_key: None,
            key: Some((*key).to_string()),
            value: None,
        })
        .collect()
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_native_set_composition_without_fallback() {
    let mut e = Engine::new_local_cpu_oracle();
    seed_native_composition_rows(&mut e);

    for (name, source, expected_keys) in native_set_composition_cases() {
        let result = e
            .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
                source,
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: None,
                order: Some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert_eq!(result.planned_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(result.fallback_reason, None, "{name}");
        assert_eq!(result.rows, key_only_rows(&expected_keys), "{name}");
    }

    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_nested_native_composition_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
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
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_bundle_filters_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
                MvccReadFilter::ProvenanceBundleKeyPrefix {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    prefix: "profile:".to_string(),
                },
                MvccReadFilter::ProvenanceBundleValueCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    expected: "team:beta".to_string(),
                    min_count: 1,
                },
                MvccReadFilter::ProvenanceBundleKeyValueEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    key: "profile:2".to_string(),
                    value: "team:beta".to_string(),
                },
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_bundle_path_filters_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_bundle_occurrence_path_filters_without_fallback()
{
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
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
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_projection_order_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=member:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 3,
            }),
            order: Some(MvccReadOrder::ProvenanceValueDesc {
                frame: MvccProvenanceFrame::TerminalInput,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("acct:2 -> profile:2 -> team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("acct:1 -> profile:1 -> team:alpha".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_prefix_terminal_value_chain_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: Some(MvccReadFilter::ProvenanceKeyPrefix {
                frame: MvccProvenanceFrame::TerminalInput,
                prefix: "profile:".to_string(),
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_labeled_branch_source_resolution_without_fallback() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                branches: vec![
                    MvccLabeledValueChainBranch {
                        label: "missing".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
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
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::BranchLabelEquals("members".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::BranchLabelTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alpha Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Beta Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Bob".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_supported_lookup_fixture() {
    let mut e = Engine::new_local_cpu_oracle();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-read-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let query = MvccReadQuery {
        source: MvccReadSource::KeyLookup {
            key: "acct:1".to_string(),
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::ValueEquals("closed".to_string())),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_supported_full_scan_fixture() {
    let mut e = Engine::new_local_cpu_oracle();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let query = MvccReadQuery {
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
    };

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_supported_key_range_filter() {
    let mut e = Engine::new_local_cpu_oracle();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::KeyRange {
            start_inclusive: "acct:1".to_string(),
            end_exclusive: "acct:4".to_string(),
        }),
        order: None,
        projection: MvccProjection::KeyOnly,
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
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_concat_native_sources() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();
    e.execute_text(4, "SET acct:4=closed").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyLookup {
                    key: "acct:2".to_string(),
                },
                MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:4".to_string(), "acct:1".to_string()],
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
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
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn first_cuda_slice_query_gap_accepts_distinct_cpu_resolved_composition() {
    let query = MvccReadQuery {
        source: MvccReadSource::ConcatDistinct {
            sources: vec![
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::SourceKeyAsc),
        projection: MvccProjection::TargetKeySourceValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_native_set_composition() {
    for (name, source, _) in native_set_composition_cases() {
        let query = MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: None,
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyOnly,
            limit: None,
        };

        assert_eq!(first_cuda_slice_query_gap(&query), None, "{name}");
    }
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_native_set_composition_variants() {
    for (name, source, expected_keys) in native_set_composition_cases() {
        let mut cpu_engine = Engine::new_local_cpu_oracle();
        seed_native_composition_rows(&mut cpu_engine);
        let query = MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: None,
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyOnly,
            limit: None,
        };

        let cpu = cpu_engine.execute_mvcc_query(&query).unwrap();
        assert_mvcc_query_uses_tracked_cpu_fallback(&cpu_engine, &cpu, 1);

        let mut backend_engine = Engine::new_local_cpu_oracle();
        seed_native_composition_rows(&mut backend_engine);
        let backend = backend_engine
            .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
            .unwrap();

        assert_eq!(backend.planned_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(backend.executed_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(backend.fallback_reason, None, "{name}");
        assert_eq!(backend.rows, cpu.rows, "{name}");
        assert_eq!(backend.rows, key_only_rows(&expected_keys), "{name}");
        assert_eq!(
            backend_engine.metrics().snapshot().fallback_total,
            0,
            "{name}"
        );
    }
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_distinct_cpu_resolved_sources() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::ConcatDistinct {
            sources: vec![
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::SourceKeyAsc),
        projection: MvccProjection::TargetKeySourceValue,
        limit: None,
    };

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
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_key_order() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::KeyDesc),
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
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn first_cuda_slice_query_gap_accepts_value_order_for_native_single_sources() {
    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_value_order() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyValue,
        limit: None,
    };

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
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn first_cuda_slice_query_gap_accepts_fan_in_order_for_native_sources() {
    let key_batch = MvccReadQuery {
        source: MvccReadSource::KeyBatchLookup {
            keys: vec!["acct:3".to_string(), "acct:1".to_string()],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyOnly,
        limit: Some(2),
    };
    assert_eq!(first_cuda_slice_query_gap(&key_batch), None);

    let concat = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                },
                MvccReadSource::KeyLookup {
                    key: "acct:4".to_string(),
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyOnly,
        limit: Some(3),
    };
    assert_eq!(first_cuda_slice_query_gap(&concat), None);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_fan_in_order() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=pending").unwrap();
    e.execute_text(4, "SET acct:4=archived").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                },
                MvccReadSource::KeyLookup {
                    key: "acct:4".to_string(),
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyOnly,
        limit: Some(3),
    };

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
                source_key: None,
                key: Some("acct:4".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_limit_after_order() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::KeyDesc),
        projection: MvccProjection::KeyOnly,
        limit: Some(2),
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
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn mvcc_benchmark_report_summarizes_gpu_coverage_and_fallback_rate() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=pending").unwrap();

    let supported_scan = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::KeyAsc),
        projection: MvccProjection::KeyOnly,
        limit: None,
    };
    let supported_concat = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyLookup {
                    key: "acct:2".to_string(),
                },
                MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: None,
        order: None,
        projection: MvccProjection::ValueOnly,
        limit: None,
    };
    let supported_nested_distinct = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "acct:1".to_string(),
                    },
                    MvccReadSource::KeyLookup {
                        key: "acct:1".to_string(),
                    },
                ],
            }],
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: None,
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    let results = vec![
        e.execute_mvcc_query_with_backend_fallback(&supported_scan, &FirstCudaSliceParityBackend)
            .unwrap(),
        e.execute_mvcc_query_with_backend_fallback(&supported_concat, &FirstCudaSliceParityBackend)
            .unwrap(),
        e.execute_mvcc_query_with_backend_fallback(
            &supported_nested_distinct,
            &FirstCudaSliceParityBackend,
        )
        .unwrap(),
    ];

    let report = MvccBenchmarkReport::from_results(&results, &e.metrics().snapshot());

    assert_eq!(report.workload_count, 3);
    assert_eq!(report.gpu_executed_count, 3);
    assert_eq!(report.cpu_fallback_count, 0);
    assert_eq!(report.gpu_executed_permyriad, 10_000);
    assert_eq!(report.cpu_fallback_permyriad, 0);
    assert!(report.d2h_bytes_total > 0);
    assert_eq!(report.h2d_bytes_total, 0);
    assert_eq!(report.kernel_exec_samples, 0);
    assert_eq!(report.batch_wait_samples, 0);
}
