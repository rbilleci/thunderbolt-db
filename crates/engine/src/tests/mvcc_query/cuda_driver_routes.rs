use crate::{
    cuda_mvcc_row_batch_transfer_bytes, resolve_mvcc_all_versions, Engine,
    MvccLabeledValueChainBranch, MvccProjection, MvccProvenanceFrame, MvccReadFilter,
    MvccReadOrder, MvccReadQuery, MvccReadRow, MvccReadSource, MvccSourceProvenance,
    MvccValueChainBranchFanIn, MvccValueChainPlan, MvccValueChainTerminal, StorageVisibility,
};
use gpu_db_execution::DeviceTarget;

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_historical_key_lookup_without_fallback() {
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
    let e = Engine::new_local_test_engine();
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
