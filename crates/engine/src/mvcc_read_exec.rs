//! MVCC read-execution subsystem (P0 §9.6 decomposition, behavior-preserving):
//! the CPU/CUDA execution backends + dispatch types, value-chain / provenance
//! resolution, row projection/filter/compare, and the source-resolution read
//! path. Operates on the mvcc_read_model types; the Engine drives it.

use super::*;

mod row_ops;
pub(crate) use row_ops::{
    cuda_mvcc_row_batch_transfer_bytes, mvcc_read_row_size, mvcc_row_cmp, mvcc_row_matches_filter,
    project_mvcc_row, resolved_mvcc_row_key,
};
mod source_resolution;
#[allow(unused_imports)]
pub(crate) use source_resolution::{
    resolve_follow_value_chain, resolve_follow_value_chain_branches,
    resolve_follow_value_chain_from_seed, resolve_follow_value_chain_labeled_branches,
    resolve_mvcc_all_versions, resolve_mvcc_source,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MvccBackendExecution {
    pub(crate) executed_target: DeviceTarget,
    pub(crate) rows: Vec<MvccReadRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum MvccBackendDispatch {
    Executed(MvccBackendExecution),
    Fallback {
        reason: FallbackReason,
        rows: Vec<ResolvedMvccRow>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedMvccRow {
    pub(crate) branch_label: Option<String>,
    pub(crate) source_key: Option<String>,
    pub(crate) source_tuple: Option<TupleVersion>,
    pub(crate) provenance_path: Option<Vec<TupleVersion>>,
    pub(crate) terminal_input_index: Option<usize>,
    pub(crate) tuple: TupleVersion,
}

pub(crate) trait MvccExecutionBackend {
    fn execute(&self, query: &MvccReadQuery, rows: Vec<ResolvedMvccRow>) -> MvccBackendDispatch;
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg(test)]
pub(crate) struct CpuMvccExecutionBackend;

#[cfg(test)]
impl MvccExecutionBackend for CpuMvccExecutionBackend {
    fn execute(&self, query: &MvccReadQuery, rows: Vec<ResolvedMvccRow>) -> MvccBackendDispatch {
        let projection = &query.projection;
        let projected = match (query.filter.clone(), query.order.clone(), query.limit) {
            (Some(filter), Some(order), Some(limit)) => {
                collect_operator_rows(ProjectOperator::new(
                    LimitOperator::new(
                        SortOperator::new(
                            FilterOperator::new(
                                ScanOperator::new(rows),
                                move |row: &ResolvedMvccRow| mvcc_row_matches_filter(row, &filter),
                            ),
                            move |left: &ResolvedMvccRow, right: &ResolvedMvccRow| {
                                mvcc_row_cmp(left, right, &order)
                            },
                        ),
                        limit,
                    ),
                    move |row| project_mvcc_row(row, projection),
                ))
            }
            (Some(filter), Some(order), None) => collect_operator_rows(ProjectOperator::new(
                SortOperator::new(
                    FilterOperator::new(ScanOperator::new(rows), move |row: &ResolvedMvccRow| {
                        mvcc_row_matches_filter(row, &filter)
                    }),
                    move |left: &ResolvedMvccRow, right: &ResolvedMvccRow| {
                        mvcc_row_cmp(left, right, &order)
                    },
                ),
                move |row| project_mvcc_row(row, projection),
            )),
            (Some(filter), None, Some(limit)) => collect_operator_rows(ProjectOperator::new(
                LimitOperator::new(
                    FilterOperator::new(ScanOperator::new(rows), move |row: &ResolvedMvccRow| {
                        mvcc_row_matches_filter(row, &filter)
                    }),
                    limit,
                ),
                move |row| project_mvcc_row(row, projection),
            )),
            (Some(filter), None, None) => collect_operator_rows(ProjectOperator::new(
                FilterOperator::new(ScanOperator::new(rows), move |row: &ResolvedMvccRow| {
                    mvcc_row_matches_filter(row, &filter)
                }),
                move |row| project_mvcc_row(row, projection),
            )),
            (None, Some(order), Some(limit)) => collect_operator_rows(ProjectOperator::new(
                LimitOperator::new(
                    SortOperator::new(
                        ScanOperator::new(rows),
                        move |left: &ResolvedMvccRow, right: &ResolvedMvccRow| {
                            mvcc_row_cmp(left, right, &order)
                        },
                    ),
                    limit,
                ),
                move |row| project_mvcc_row(row, projection),
            )),
            (None, Some(order), None) => collect_operator_rows(ProjectOperator::new(
                SortOperator::new(
                    ScanOperator::new(rows),
                    move |left: &ResolvedMvccRow, right: &ResolvedMvccRow| {
                        mvcc_row_cmp(left, right, &order)
                    },
                ),
                move |row| project_mvcc_row(row, projection),
            )),
            (None, None, Some(limit)) => collect_operator_rows(ProjectOperator::new(
                LimitOperator::new(ScanOperator::new(rows), limit),
                move |row| project_mvcc_row(row, projection),
            )),
            (None, None, None) => {
                collect_operator_rows(ProjectOperator::new(ScanOperator::new(rows), move |row| {
                    project_mvcc_row(row, projection)
                }))
            }
        };

        MvccBackendDispatch::Executed(MvccBackendExecution {
            executed_target: DeviceTarget::Cpu,
            rows: projected,
        })
    }
}

pub(crate) struct CudaMvccExecutionBackend {
    pub(crate) router: DeviceRouter<CudaDriverRuntime>,
    pub(crate) gpu_id: u16,
}

impl CudaMvccExecutionBackend {
    pub(crate) fn new(runtime: CudaDriverRuntime, gpu_id: u16) -> Self {
        Self {
            router: DeviceRouter::new(runtime),
            gpu_id,
        }
    }
}

impl MvccExecutionBackend for CudaMvccExecutionBackend {
    fn execute(&self, query: &MvccReadQuery, rows: Vec<ResolvedMvccRow>) -> MvccBackendDispatch {
        let op = PlannedOp {
            name: "mvcc_read_first_cuda_slice".to_string(),
            target: DeviceTarget::Gpu(self.gpu_id),
        };

        match self.router.route(&op) {
            RouteDecision::Gpu(_) => {}
            RouteDecision::CpuFallback { reason, .. } => {
                return MvccBackendDispatch::Fallback {
                    reason: FallbackReason::from(reason),
                    rows,
                };
            }
            RouteDecision::Cpu => {
                unreachable!("CUDA MVCC backend always plans GPU execution")
            }
        }

        if first_cuda_slice_query_gap(query).is_some() {
            return MvccBackendDispatch::Fallback {
                reason: FallbackReason::GpuMvccReadParityGap,
                rows,
            };
        }

        match execute_cuda_supported_filter(query, rows, self.router.runtime(), self.gpu_id) {
            Ok(executed) => MvccBackendDispatch::Executed(executed),
            Err(rows) => MvccBackendDispatch::Fallback {
                reason: FallbackReason::GpuMvccReadParityGap,
                rows,
            },
        }
    }
}

pub(crate) fn execute_cuda_supported_filter(
    query: &MvccReadQuery,
    rows: Vec<ResolvedMvccRow>,
    runtime: &CudaDriverRuntime,
    gpu_id: u16,
) -> Result<MvccBackendExecution, Vec<ResolvedMvccRow>> {
    let mut mask = cuda_source_mask(&query.source, &rows, runtime).map_err(|_| rows.clone())?;
    let visibility_mask = cuda_visibility_mask(&rows, query.visibility.read_txn_id, runtime)
        .map_err(|_| rows.clone())?;
    for (matched_source, visible) in mask.iter_mut().zip(visibility_mask) {
        *matched_source &= visible;
    }

    let filter_mask = if let Some(filter) = query.filter.as_ref() {
        cuda_filter_mask(filter, &rows, runtime).map_err(|_| rows.clone())?
    } else {
        runtime
            .filter_all_mask(rows.len())
            .map_err(|_| rows.clone())?
    };

    for (visible, matched) in mask.iter_mut().zip(filter_mask) {
        *visible &= matched;
    }

    let mut matched_rows = rows
        .into_iter()
        .zip(mask)
        .filter_map(|(row, matched)| matched.then_some(row))
        .collect::<Vec<_>>();

    if let Some(order) = query.order.as_ref() {
        matched_rows.sort_by(|left, right| mvcc_row_cmp(left, right, order));
    }

    if let Some(limit) = query.limit {
        matched_rows.truncate(limit);
    }

    let projection = &query.projection;
    let rows = matched_rows
        .into_iter()
        .map(|row| project_mvcc_row(row, projection))
        .collect();

    Ok(MvccBackendExecution {
        executed_target: DeviceTarget::Gpu(gpu_id),
        rows,
    })
}

pub(crate) fn cuda_visibility_mask(
    rows: &[ResolvedMvccRow],
    read_txn_id: u64,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<bool>, ()> {
    let batch = CudaMvccRowBatch::from_key_values_with_metadata(rows.iter().map(|row| {
        (
            row.tuple.key.as_bytes(),
            row.tuple.value.as_bytes(),
            row.tuple.created_by,
            row.tuple.deleted_by.unwrap_or(u64::MAX),
            None,
        )
    }))
    .map_err(|_| ())?;
    runtime
        .mvcc_visibility_mask(&batch, read_txn_id)
        .map_err(|_| ())
}

pub(crate) fn cuda_source_mask(
    source: &MvccReadSource,
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
) -> Result<Vec<bool>, ()> {
    match source {
        MvccReadSource::FullScan => runtime.filter_all_mask(rows.len()).map_err(|_| ()),
        MvccReadSource::KeyLookup { key } => cuda_key_exact_mask(rows, runtime, key.as_bytes()),
        MvccReadSource::KeyBatchLookup { keys } => {
            let mut combined = vec![false; rows.len()];
            for mask in keys
                .iter()
                .map(|key| cuda_key_exact_mask(rows, runtime, key.as_bytes()))
            {
                for (combined, matched) in combined.iter_mut().zip(mask?) {
                    *combined |= matched;
                }
            }
            Ok(combined)
        }
        source if is_cuda_native_composition_source(source) => {
            runtime.filter_all_mask(rows.len()).map_err(|_| ())
        }
        source if is_cuda_cpu_resolved_source(source) => {
            runtime.filter_all_mask(rows.len()).map_err(|_| ())
        }
        _ => Err(()),
    }
}

pub(crate) fn cuda_filter_mask(
    filter: &MvccReadFilter,
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
) -> Result<Vec<bool>, ()> {
    match filter {
        MvccReadFilter::KeyPrefix(prefix) => cuda_key_prefix_mask(rows, runtime, prefix.as_bytes()),
        MvccReadFilter::ProvenanceKeyPrefix { frame, prefix } => {
            cuda_provenance_key_prefix_mask(rows, runtime, *frame, prefix.as_bytes())
        }
        MvccReadFilter::ProvenanceBundleKeyEquals { bundle, expected } => {
            cuda_provenance_bundle_key_count_mask(rows, runtime, *bundle, expected.as_bytes(), 1)
        }
        MvccReadFilter::ProvenanceBundleKeyCountAtLeast {
            bundle,
            expected,
            min_count,
        } => cuda_provenance_bundle_key_count_mask(
            rows,
            runtime,
            *bundle,
            expected.as_bytes(),
            *min_count,
        ),
        MvccReadFilter::ProvenanceBundleKeyPrefix { bundle, prefix } => {
            cuda_provenance_bundle_key_prefix_mask(rows, runtime, *bundle, prefix.as_bytes())
        }
        MvccReadFilter::KeyRange {
            start_inclusive,
            end_exclusive,
        } => cuda_key_range_mask(rows, runtime, start_inclusive, end_exclusive),
        MvccReadFilter::ValueEquals(expected) => cuda_value_equals_mask(rows, runtime, expected),
        MvccReadFilter::ProvenanceValueEquals { frame, expected } => {
            cuda_provenance_value_equals_mask(rows, runtime, *frame, expected)
        }
        MvccReadFilter::ProvenanceBundleValueEquals { bundle, expected } => {
            cuda_provenance_bundle_value_count_mask(rows, runtime, *bundle, expected.as_bytes(), 1)
        }
        MvccReadFilter::ProvenanceBundleValueCountAtLeast {
            bundle,
            expected,
            min_count,
        } => cuda_provenance_bundle_value_count_mask(
            rows,
            runtime,
            *bundle,
            expected.as_bytes(),
            *min_count,
        ),
        MvccReadFilter::ProvenanceBundleKeyValueEquals { bundle, key, value } => {
            cuda_provenance_bundle_key_value_count_mask(
                rows,
                runtime,
                *bundle,
                key.as_bytes(),
                value.as_bytes(),
                1,
            )
        }
        MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast {
            bundle,
            key,
            value,
            min_count,
        } => cuda_provenance_bundle_key_value_count_mask(
            rows,
            runtime,
            *bundle,
            key.as_bytes(),
            value.as_bytes(),
            *min_count,
        ),
        MvccReadFilter::SourceKeyPrefix(_)
        | MvccReadFilter::BranchLabelEquals(_)
        | MvccReadFilter::SourceValueEquals(_)
        | MvccReadFilter::ProvenanceBundlePathEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathContains { .. }
        | MvccReadFilter::ProvenanceBundlePathCountAtLeast { .. }
        | MvccReadFilter::ProvenanceBundlePathPairAtDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathSuffixEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathPrefixEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathSliceEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
            ..
        }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrencePairAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
            ..
        }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathSegmentEquals { .. }
        | MvccReadFilter::ProvenanceBundleLenEquals { .. } => {
            cuda_cpu_resolved_filter_mask(rows, runtime, filter)
        }
        MvccReadFilter::All(filters) => {
            let mut combined = vec![true; rows.len()];
            for mask in filters
                .iter()
                .map(|filter| cuda_filter_mask(filter, rows, runtime))
            {
                for (combined, matched) in combined.iter_mut().zip(mask?) {
                    *combined &= matched;
                }
            }
            Ok(combined)
        }
        MvccReadFilter::Any(filters) => {
            let mut combined = vec![false; rows.len()];
            for mask in filters
                .iter()
                .map(|filter| cuda_filter_mask(filter, rows, runtime))
            {
                for (combined, matched) in combined.iter_mut().zip(mask?) {
                    *combined |= matched;
                }
            }
            Ok(combined)
        }
    }
}

pub(crate) fn cuda_cpu_resolved_filter_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    filter: &MvccReadFilter,
) -> Result<Vec<bool>, ()> {
    let values = rows
        .iter()
        .map(|row| {
            if mvcc_row_matches_filter(row, filter) {
                b"1".as_slice()
            } else {
                b"0".as_slice()
            }
        })
        .collect::<Vec<_>>();
    runtime
        .filter_equal_bytes_mask(&values, b"1")
        .map_err(|_| ())
}

pub(crate) fn cuda_key_range_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    start_inclusive: &str,
    end_exclusive: &str,
) -> Result<Vec<bool>, ()> {
    let Some(prefix) = prefix_equivalent_key_range(start_inclusive, end_exclusive) else {
        let values = rows
            .iter()
            .map(|row| row.tuple.key.as_bytes())
            .collect::<Vec<_>>();
        return runtime
            .filter_bytes_range_mask(
                &values,
                start_inclusive.as_bytes(),
                end_exclusive.as_bytes(),
            )
            .map_err(|_| ());
    };

    cuda_key_prefix_mask(rows, runtime, prefix.as_bytes())
}

pub(crate) fn cuda_key_exact_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    expected: &[u8],
) -> Result<Vec<bool>, ()> {
    let values = rows
        .iter()
        .map(|row| row.tuple.key.as_bytes())
        .collect::<Vec<_>>();
    runtime
        .filter_equal_bytes_mask(&values, expected)
        .map_err(|_| ())
}

pub(crate) fn cuda_key_prefix_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    prefix: &[u8],
) -> Result<Vec<bool>, ()> {
    let prefix_len = prefix.len();
    let values = rows
        .iter()
        .map(|row| {
            let key = row.tuple.key.as_bytes();
            &key[..key.len().min(prefix_len)]
        })
        .collect::<Vec<_>>();
    runtime
        .filter_equal_bytes_mask(&values, prefix)
        .map_err(|_| ())
}

pub(crate) fn cuda_value_equals_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    expected: &str,
) -> Result<Vec<bool>, ()> {
    if let Ok(needle) = expected.parse::<u32>() {
        if let Ok(values) = rows
            .iter()
            .map(|row| row.tuple.value.parse::<u32>())
            .collect::<Result<Vec<_>, _>>()
        {
            return runtime
                .filter_equal_u32_mask(&values, needle)
                .map_err(|_| ());
        }
    }

    let values = rows
        .iter()
        .map(|row| row.tuple.value.as_bytes())
        .collect::<Vec<_>>();
    runtime
        .filter_equal_bytes_mask(&values, expected.as_bytes())
        .map_err(|_| ())
}

pub(crate) fn cuda_provenance_key_prefix_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    frame: MvccProvenanceFrame,
    prefix: &[u8],
) -> Result<Vec<bool>, ()> {
    let prefix_len = prefix.len();
    let mut present = Vec::with_capacity(rows.len());
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        if let Some(tuple) = resolved_mvcc_row_provenance_tuple(row, frame) {
            let key = tuple.key.as_bytes();
            present.push(true);
            values.push(&key[..key.len().min(prefix_len)]);
        } else {
            present.push(false);
            values.push(&[][..]);
        }
    }

    let mut mask = runtime
        .filter_equal_bytes_mask(&values, prefix)
        .map_err(|_| ())?;
    for (matched, present) in mask.iter_mut().zip(present) {
        *matched &= present;
    }
    Ok(mask)
}

pub(crate) fn cuda_provenance_value_equals_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    frame: MvccProvenanceFrame,
    expected: &str,
) -> Result<Vec<bool>, ()> {
    let mut present = Vec::with_capacity(rows.len());
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        if let Some(tuple) = resolved_mvcc_row_provenance_tuple(row, frame) {
            present.push(true);
            values.push(tuple.value.as_bytes());
        } else {
            present.push(false);
            values.push(&[][..]);
        }
    }

    let mut mask = runtime
        .filter_equal_bytes_mask(&values, expected.as_bytes())
        .map_err(|_| ())?;
    for (matched, present) in mask.iter_mut().zip(present) {
        *matched &= present;
    }
    Ok(mask)
}

pub(crate) fn cuda_provenance_bundle_key_count_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    bundle: MvccProvenanceFrameBundle,
    expected: &[u8],
    min_count: usize,
) -> Result<Vec<bool>, ()> {
    cuda_provenance_bundle_count_mask(rows, bundle, min_count, |tuples, index| {
        let values = tuples
            .iter()
            .map(|bundle| {
                bundle
                    .as_ref()
                    .and_then(|tuples| tuples.get(index))
                    .map_or(&[][..], |tuple| tuple.key.as_bytes())
            })
            .collect::<Vec<_>>();
        runtime
            .filter_equal_bytes_mask(&values, expected)
            .map_err(|_| ())
    })
}

pub(crate) fn cuda_provenance_bundle_key_prefix_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    bundle: MvccProvenanceFrameBundle,
    prefix: &[u8],
) -> Result<Vec<bool>, ()> {
    let prefix_len = prefix.len();
    cuda_provenance_bundle_any_mask(rows, bundle, |tuples, index| {
        let values = tuples
            .iter()
            .map(|bundle| {
                bundle
                    .as_ref()
                    .and_then(|tuples| tuples.get(index))
                    .map_or(&[][..], |tuple| {
                        let key = tuple.key.as_bytes();
                        &key[..key.len().min(prefix_len)]
                    })
            })
            .collect::<Vec<_>>();
        runtime
            .filter_equal_bytes_mask(&values, prefix)
            .map_err(|_| ())
    })
}

pub(crate) fn cuda_provenance_bundle_value_count_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    bundle: MvccProvenanceFrameBundle,
    expected: &[u8],
    min_count: usize,
) -> Result<Vec<bool>, ()> {
    cuda_provenance_bundle_count_mask(rows, bundle, min_count, |tuples, index| {
        let values = tuples
            .iter()
            .map(|bundle| {
                bundle
                    .as_ref()
                    .and_then(|tuples| tuples.get(index))
                    .map_or(&[][..], |tuple| tuple.value.as_bytes())
            })
            .collect::<Vec<_>>();
        runtime
            .filter_equal_bytes_mask(&values, expected)
            .map_err(|_| ())
    })
}

pub(crate) fn cuda_provenance_bundle_key_value_count_mask(
    rows: &[ResolvedMvccRow],
    runtime: &CudaDriverRuntime,
    bundle: MvccProvenanceFrameBundle,
    expected_key: &[u8],
    expected_value: &[u8],
    min_count: usize,
) -> Result<Vec<bool>, ()> {
    cuda_provenance_bundle_count_mask(rows, bundle, min_count, |tuples, index| {
        let keys = tuples
            .iter()
            .map(|bundle| {
                bundle
                    .as_ref()
                    .and_then(|tuples| tuples.get(index))
                    .map_or(&[][..], |tuple| tuple.key.as_bytes())
            })
            .collect::<Vec<_>>();
        let values = tuples
            .iter()
            .map(|bundle| {
                bundle
                    .as_ref()
                    .and_then(|tuples| tuples.get(index))
                    .map_or(&[][..], |tuple| tuple.value.as_bytes())
            })
            .collect::<Vec<_>>();
        let mut key_mask = runtime
            .filter_equal_bytes_mask(&keys, expected_key)
            .map_err(|_| ())?;
        let value_mask = runtime
            .filter_equal_bytes_mask(&values, expected_value)
            .map_err(|_| ())?;
        for (matched_key, matched_value) in key_mask.iter_mut().zip(value_mask) {
            *matched_key &= matched_value;
        }
        Ok(key_mask)
    })
}

pub(crate) fn cuda_provenance_bundle_any_mask(
    rows: &[ResolvedMvccRow],
    bundle: MvccProvenanceFrameBundle,
    mut positional_mask: impl FnMut(&[Option<Vec<&TupleVersion>>], usize) -> Result<Vec<bool>, ()>,
) -> Result<Vec<bool>, ()> {
    cuda_provenance_bundle_count_mask(rows, bundle, 1, |tuples, index| {
        positional_mask(tuples, index)
    })
}

pub(crate) fn cuda_provenance_bundle_count_mask(
    rows: &[ResolvedMvccRow],
    bundle: MvccProvenanceFrameBundle,
    min_count: usize,
    mut positional_mask: impl FnMut(&[Option<Vec<&TupleVersion>>], usize) -> Result<Vec<bool>, ()>,
) -> Result<Vec<bool>, ()> {
    let bundles = rows
        .iter()
        .map(|row| resolved_mvcc_row_provenance_bundle(row, bundle))
        .collect::<Vec<_>>();
    let max_len = bundles
        .iter()
        .filter_map(|bundle| bundle.as_ref().map(Vec::len))
        .max()
        .unwrap_or(0);
    let present = bundles.iter().map(Option::is_some).collect::<Vec<_>>();
    let mut counts = vec![0usize; rows.len()];

    for index in 0..max_len {
        let mask = positional_mask(&bundles, index)?;
        for (count, matched) in counts.iter_mut().zip(mask) {
            if matched {
                *count += 1;
            }
        }
    }

    Ok(present
        .into_iter()
        .zip(counts)
        .map(|(present, count)| present && count >= min_count)
        .collect())
}

pub(crate) fn prefix_equivalent_key_range<'a>(
    start_inclusive: &'a str,
    end_exclusive: &str,
) -> Option<&'a str> {
    let mut successor = start_inclusive.as_bytes().to_vec();
    let last = successor.last_mut()?;
    if *last == u8::MAX {
        return None;
    }
    *last += 1;

    (successor == end_exclusive.as_bytes()).then_some(start_inclusive)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FinalizedMvccBackendExecution {
    pub(crate) executed_target: DeviceTarget,
    pub(crate) fallback_reason: Option<FallbackReason>,
    pub(crate) rows: Vec<MvccReadRow>,
}

#[cfg(test)]
pub(crate) fn execute_mvcc_backend_chain<B: MvccExecutionBackend, F: MvccExecutionBackend>(
    query: &MvccReadQuery,
    rows: Vec<ResolvedMvccRow>,
    backend: &B,
    cpu_fallback: &F,
) -> FinalizedMvccBackendExecution {
    match backend.execute(query, rows) {
        MvccBackendDispatch::Executed(executed) => FinalizedMvccBackendExecution {
            executed_target: executed.executed_target,
            fallback_reason: None,
            rows: executed.rows,
        },
        MvccBackendDispatch::Fallback { reason, rows } => match cpu_fallback.execute(query, rows) {
            MvccBackendDispatch::Executed(executed) => FinalizedMvccBackendExecution {
                executed_target: executed.executed_target,
                fallback_reason: Some(reason),
                rows: executed.rows,
            },
            MvccBackendDispatch::Fallback { .. } => {
                unreachable!("CPU fallback backend must execute")
            }
        },
    }
}

pub(crate) fn execute_cuda_native_single_source_query(
    query: &MvccReadQuery,
    rows: Vec<ResolvedMvccRow>,
    backend: &CudaMvccExecutionBackend,
) -> Result<FinalizedMvccBackendExecution, FallbackReason> {
    match backend.execute(query, rows) {
        MvccBackendDispatch::Executed(executed) => Ok(FinalizedMvccBackendExecution {
            executed_target: executed.executed_target,
            fallback_reason: None,
            rows: executed.rows,
        }),
        MvccBackendDispatch::Fallback { reason, .. } => Err(reason),
    }
}

pub(crate) fn execute_cuda_native_key_batch_query(
    query: &MvccReadQuery,
    keys: &[String],
    all_version_rows: Vec<ResolvedMvccRow>,
    backend: &CudaMvccExecutionBackend,
) -> Result<(FinalizedMvccBackendExecution, u64), FallbackReason> {
    let mut compact_rows = Vec::new();
    for key in keys {
        compact_rows.extend(
            all_version_rows
                .iter()
                .filter(|row| row.tuple.key == *key)
                .cloned(),
        );
    }
    let h2d_bytes = cuda_mvcc_row_batch_transfer_bytes(&compact_rows);

    let compact_query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: query.visibility,
        filter: query.filter.clone(),
        order: query.order.clone(),
        projection: query.projection.clone(),
        limit: query.limit,
    };
    match backend.execute(&compact_query, compact_rows) {
        MvccBackendDispatch::Executed(executed) => Ok((
            FinalizedMvccBackendExecution {
                executed_target: executed.executed_target,
                fallback_reason: None,
                rows: executed.rows,
            },
            h2d_bytes,
        )),
        MvccBackendDispatch::Fallback { reason, .. } => Err(reason),
    }
}

pub(crate) fn execute_cuda_native_concat_query(
    query: &MvccReadQuery,
    sources: &[MvccReadSource],
    all_version_rows: Vec<ResolvedMvccRow>,
    backend: &CudaMvccExecutionBackend,
) -> Result<FinalizedMvccBackendExecution, FallbackReason> {
    let runtime = ensure_cuda_backend_available(backend)?;
    let mut rows = Vec::new();
    for source in sources {
        let source_rows =
            resolve_cuda_native_source_rows(source, &all_version_rows, query.visibility, runtime)?;
        let source_query = MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: query.visibility,
            filter: query.filter.clone(),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        };
        let execution =
            execute_cuda_native_single_source_query(&source_query, source_rows, backend)?;
        rows.extend(execution.rows);
    }

    if let Some(order) = query.order.as_ref() {
        sort_projected_key_value_rows(&mut rows, order);
    }

    if let Some(limit) = query.limit {
        rows.truncate(limit);
    }

    let rows = rows
        .into_iter()
        .map(|row| project_key_value_read_row(row, &query.projection))
        .collect();

    Ok(FinalizedMvccBackendExecution {
        executed_target: DeviceTarget::Gpu(backend.gpu_id),
        fallback_reason: None,
        rows,
    })
}

pub(crate) fn execute_cuda_native_composition_query(
    query: &MvccReadQuery,
    source: &MvccReadSource,
    all_version_rows: Vec<ResolvedMvccRow>,
    backend: &CudaMvccExecutionBackend,
) -> Result<FinalizedMvccBackendExecution, FallbackReason> {
    let runtime = ensure_cuda_backend_available(backend)?;
    let rows =
        resolve_cuda_native_composition_rows(source, &all_version_rows, query.visibility, runtime)?;

    match backend.execute(query, rows) {
        MvccBackendDispatch::Executed(executed) => Ok(FinalizedMvccBackendExecution {
            executed_target: executed.executed_target,
            fallback_reason: None,
            rows: executed.rows,
        }),
        MvccBackendDispatch::Fallback { reason, .. } => Err(reason),
    }
}

pub(crate) fn resolve_cuda_native_source_rows(
    source: &MvccReadSource,
    all_version_rows: &[ResolvedMvccRow],
    visibility: StorageVisibility,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<ResolvedMvccRow>, FallbackReason> {
    match source {
        MvccReadSource::FullScan => {
            let visibility_mask =
                cuda_visibility_mask(all_version_rows, visibility.read_txn_id, runtime)
                    .map_err(|_| FallbackReason::GpuMvccReadParityGap)?;
            Ok(all_version_rows
                .iter()
                .zip(visibility_mask)
                .filter(|(_, visible)| *visible)
                .map(|(row, _)| row.clone())
                .collect())
        }
        MvccReadSource::KeyLookup { key } => {
            Ok(
                cuda_visible_key_rows(all_version_rows, key, visibility, runtime)?
                    .into_iter()
                    .map(resolved_row_from_tuple)
                    .collect(),
            )
        }
        MvccReadSource::KeyBatchLookup { keys } => {
            let mut rows = Vec::new();
            for key in keys {
                rows.extend(
                    cuda_visible_key_rows(all_version_rows, key, visibility, runtime)?
                        .into_iter()
                        .map(resolved_row_from_tuple),
                );
            }
            Ok(rows)
        }
        MvccReadSource::Concat { sources } => {
            let mut rows = Vec::new();
            for source in sources {
                rows.extend(resolve_cuda_native_source_rows(
                    source,
                    all_version_rows,
                    visibility,
                    runtime,
                )?);
            }
            Ok(rows)
        }
        MvccReadSource::FollowValueChain {
            keys,
            plan,
            provenance,
        } => resolve_cuda_native_follow_value_chain_rows(
            all_version_rows,
            keys,
            visibility,
            *plan,
            *provenance,
            runtime,
        ),
        MvccReadSource::FollowValueChainBranches {
            keys,
            plans,
            fan_in,
            provenance,
        } => resolve_cuda_native_follow_value_chain_branches_rows(
            all_version_rows,
            keys,
            visibility,
            plans,
            *fan_in,
            *provenance,
            runtime,
        ),
        MvccReadSource::FollowValueChainLabeledBranches {
            keys,
            branches,
            fan_in,
            provenance,
        } => resolve_cuda_native_follow_value_chain_labeled_branches_rows(
            all_version_rows,
            keys,
            visibility,
            branches,
            *fan_in,
            *provenance,
            runtime,
        ),
        source if is_cuda_native_composition_source(source) => {
            resolve_cuda_native_composition_rows(source, all_version_rows, visibility, runtime)
        }
        _ => Err(FallbackReason::GpuMvccReadParityGap),
    }
}

pub(crate) fn resolved_row_from_tuple(tuple: TupleVersion) -> ResolvedMvccRow {
    ResolvedMvccRow {
        branch_label: None,
        source_key: None,
        source_tuple: None,
        provenance_path: None,
        terminal_input_index: None,
        tuple,
    }
}

pub(crate) fn resolve_cuda_native_composition_rows(
    source: &MvccReadSource,
    all_version_rows: &[ResolvedMvccRow],
    visibility: StorageVisibility,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<ResolvedMvccRow>, FallbackReason> {
    let sources = match source {
        MvccReadSource::ConcatDistinct { sources }
        | MvccReadSource::IntersectDistinct { sources }
        | MvccReadSource::IntersectAll { sources }
        | MvccReadSource::ExceptDistinct { sources }
        | MvccReadSource::ExceptAll { sources }
        | MvccReadSource::SymmetricDifferenceDistinct { sources }
        | MvccReadSource::SymmetricDifferenceAll { sources } => sources,
        _ => return Err(FallbackReason::GpuMvccReadParityGap),
    };

    let resolved_sources = sources
        .iter()
        .map(|source| {
            resolve_cuda_native_source_rows(source, all_version_rows, visibility, runtime)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(compose_resolved_mvcc_rows(source, resolved_sources))
}

pub(crate) fn compose_resolved_mvcc_rows(
    source: &MvccReadSource,
    resolved_sources: Vec<Vec<ResolvedMvccRow>>,
) -> Vec<ResolvedMvccRow> {
    match source {
        MvccReadSource::ConcatDistinct { .. } => {
            let mut rows = Vec::new();
            let mut seen = BTreeSet::new();
            for source_rows in resolved_sources {
                for row in source_rows {
                    if seen.insert(resolved_mvcc_row_key(&row)) {
                        rows.push(row);
                    }
                }
            }
            rows
        }
        MvccReadSource::IntersectDistinct { .. } => {
            let mut sources_iter = resolved_sources.into_iter();
            let Some(first_rows) = sources_iter.next() else {
                return Vec::new();
            };
            let remaining_sets = sources_iter
                .map(|rows| {
                    rows.into_iter()
                        .map(|row| resolved_mvcc_row_key(&row))
                        .collect::<BTreeSet<_>>()
                })
                .collect::<Vec<_>>();

            let mut intersection = Vec::new();
            let mut emitted = BTreeSet::new();
            'rows: for row in first_rows {
                let key = resolved_mvcc_row_key(&row);
                if !emitted.insert(key.clone()) {
                    continue;
                }
                for other in &remaining_sets {
                    if !other.contains(&key) {
                        continue 'rows;
                    }
                }
                intersection.push(row);
            }
            intersection
        }
        MvccReadSource::IntersectAll { .. } => {
            let mut sources_iter = resolved_sources.into_iter();
            let Some(first_rows) = sources_iter.next() else {
                return Vec::new();
            };
            let remaining_counts = sources_iter
                .map(|rows| {
                    let mut counts = BTreeMap::new();
                    for key in rows.into_iter().map(|row| resolved_mvcc_row_key(&row)) {
                        *counts.entry(key).or_insert(0usize) += 1;
                    }
                    counts
                })
                .collect::<Vec<_>>();

            let mut consumed = BTreeMap::new();
            let mut intersection = Vec::new();
            'rows: for row in first_rows {
                let key = resolved_mvcc_row_key(&row);
                let next_count = consumed.get(&key).copied().unwrap_or(0) + 1;
                for counts in &remaining_counts {
                    if counts.get(&key).copied().unwrap_or(0) < next_count {
                        continue 'rows;
                    }
                }
                consumed.insert(key, next_count);
                intersection.push(row);
            }
            intersection
        }
        MvccReadSource::ExceptDistinct { .. } => {
            let mut sources_iter = resolved_sources.into_iter();
            let Some(first_rows) = sources_iter.next() else {
                return Vec::new();
            };
            let exclusion_set = sources_iter
                .flat_map(|rows| rows.into_iter().map(|row| resolved_mvcc_row_key(&row)))
                .collect::<BTreeSet<_>>();

            let mut difference = Vec::new();
            let mut emitted = BTreeSet::new();
            for row in first_rows {
                let key = resolved_mvcc_row_key(&row);
                if !emitted.insert(key.clone()) || exclusion_set.contains(&key) {
                    continue;
                }
                difference.push(row);
            }
            difference
        }
        MvccReadSource::ExceptAll { .. } => {
            let mut sources_iter = resolved_sources.into_iter();
            let Some(first_rows) = sources_iter.next() else {
                return Vec::new();
            };
            let mut exclusion_counts = BTreeMap::new();
            for rows in sources_iter {
                for key in rows.into_iter().map(|row| resolved_mvcc_row_key(&row)) {
                    *exclusion_counts.entry(key).or_insert(0usize) += 1;
                }
            }

            let mut consumed = BTreeMap::new();
            let mut difference = Vec::new();
            for row in first_rows {
                let key = resolved_mvcc_row_key(&row);
                let next_count = consumed.get(&key).copied().unwrap_or(0) + 1;
                consumed.insert(key.clone(), next_count);
                if exclusion_counts.get(&key).copied().unwrap_or(0) >= next_count {
                    continue;
                }
                difference.push(row);
            }
            difference
        }
        MvccReadSource::SymmetricDifferenceDistinct { .. } => {
            let mut presence_counts = BTreeMap::new();
            for rows in &resolved_sources {
                let per_source = rows
                    .iter()
                    .map(resolved_mvcc_row_key)
                    .collect::<BTreeSet<_>>();
                for key in per_source {
                    *presence_counts.entry(key).or_insert(0usize) += 1;
                }
            }

            let mut output = Vec::new();
            let mut emitted = BTreeSet::new();
            for rows in resolved_sources {
                for row in rows {
                    let key = resolved_mvcc_row_key(&row);
                    if presence_counts.get(&key) == Some(&1) && emitted.insert(key) {
                        output.push(row);
                    }
                }
            }
            output
        }
        MvccReadSource::SymmetricDifferenceAll { .. } => {
            let mut remaining = BTreeMap::new();
            for rows in &resolved_sources {
                let mut source_counts = BTreeMap::new();
                for key in rows.iter().map(resolved_mvcc_row_key) {
                    *source_counts.entry(key).or_insert(0usize) += 1;
                }
                for (key, count) in source_counts {
                    let current = remaining.remove(&key).unwrap_or(0);
                    if current >= count {
                        let next = current - count;
                        if next > 0 {
                            remaining.insert(key, next);
                        }
                    } else {
                        remaining.insert(key, count - current);
                    }
                }
            }

            let mut output = Vec::new();
            for rows in resolved_sources {
                for row in rows {
                    let key = resolved_mvcc_row_key(&row);
                    if let Some(count) = remaining.get_mut(&key) {
                        if *count > 0 {
                            output.push(row);
                            *count -= 1;
                        }
                    }
                }
            }
            output
        }
        _ => Vec::new(),
    }
}

pub(crate) fn ensure_cuda_backend_available(
    backend: &CudaMvccExecutionBackend,
) -> Result<&CudaDriverRuntime, FallbackReason> {
    let op = PlannedOp {
        name: "mvcc_read_native_cuda_source_resolution".to_string(),
        target: DeviceTarget::Gpu(backend.gpu_id),
    };
    match backend.router.route(&op) {
        RouteDecision::Gpu(_) => Ok(backend.router.runtime()),
        RouteDecision::CpuFallback { reason, .. } => Err(FallbackReason::from(reason)),
        RouteDecision::Cpu => unreachable!("CUDA source resolution always plans GPU execution"),
    }
}

pub(crate) fn cuda_visible_key_rows(
    rows: &[ResolvedMvccRow],
    key: &str,
    visibility: StorageVisibility,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<TupleVersion>, FallbackReason> {
    let key_mask = cuda_key_exact_mask(rows, runtime, key.as_bytes())
        .map_err(|_| FallbackReason::GpuMvccReadParityGap)?;
    let visibility_mask = cuda_visibility_mask(rows, visibility.read_txn_id, runtime)
        .map_err(|_| FallbackReason::GpuMvccReadParityGap)?;

    Ok(rows
        .iter()
        .zip(key_mask.into_iter().zip(visibility_mask))
        .filter(|(_, (key_matched, visible))| *key_matched && *visible)
        .map(|(row, _)| row.tuple.clone())
        .collect())
}

pub(crate) fn cuda_visible_key_prefix_rows(
    rows: &[ResolvedMvccRow],
    prefix: &str,
    visibility: StorageVisibility,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<TupleVersion>, FallbackReason> {
    let prefix_mask = cuda_key_prefix_mask(rows, runtime, prefix.as_bytes())
        .map_err(|_| FallbackReason::GpuMvccReadParityGap)?;
    let visibility_mask = cuda_visibility_mask(rows, visibility.read_txn_id, runtime)
        .map_err(|_| FallbackReason::GpuMvccReadParityGap)?;

    Ok(rows
        .iter()
        .zip(prefix_mask.into_iter().zip(visibility_mask))
        .filter(|(_, (prefix_matched, visible))| *prefix_matched && *visible)
        .map(|(row, _)| row.tuple.clone())
        .collect())
}

pub(crate) fn resolve_cuda_native_follow_value_chain_rows(
    all_version_rows: &[ResolvedMvccRow],
    keys: &[String],
    visibility: StorageVisibility,
    plan: MvccValueChainPlan,
    provenance: MvccSourceProvenance,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<ResolvedMvccRow>, FallbackReason> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = cuda_visible_key_rows(all_version_rows, key, visibility, runtime)?
            .into_iter()
            .next()
        else {
            continue;
        };

        rows.extend(resolve_cuda_native_follow_value_chain_from_seed(
            all_version_rows,
            &seed,
            visibility,
            plan,
            provenance,
            None,
            runtime,
        )?);
    }

    Ok(rows)
}

pub(crate) fn resolve_cuda_native_follow_value_chain_from_seed(
    all_version_rows: &[ResolvedMvccRow],
    seed: &TupleVersion,
    visibility: StorageVisibility,
    plan: MvccValueChainPlan,
    provenance: MvccSourceProvenance,
    branch_label: Option<&str>,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<ResolvedMvccRow>, FallbackReason> {
    let mut current = seed.clone();
    let mut provenance_path = vec![seed.clone()];
    let mut previous = None;
    for _ in 0..plan.value_key_hops {
        previous = Some(current.clone());
        let next_key = current.value.clone();
        let Some(next) = cuda_visible_key_rows(all_version_rows, &next_key, visibility, runtime)?
            .into_iter()
            .next()
        else {
            return Ok(Vec::new());
        };
        current = next;
        provenance_path.push(current.clone());
    }

    let terminal_input_index = match plan.terminal {
        MvccValueChainTerminal::CurrentRow => {
            if plan.value_key_hops == 0 {
                0
            } else {
                provenance_path.len().saturating_sub(2)
            }
        }
        MvccValueChainTerminal::CurrentValuePrefixes => provenance_path.len().saturating_sub(1),
    };
    let provenance_tuple = match provenance {
        MvccSourceProvenance::Seed => seed.clone(),
        MvccSourceProvenance::TerminalInput => match plan.terminal {
            MvccValueChainTerminal::CurrentRow => previous.unwrap_or_else(|| seed.clone()),
            MvccValueChainTerminal::CurrentValuePrefixes => current.clone(),
        },
    };

    let mut rows = Vec::new();
    match plan.terminal {
        MvccValueChainTerminal::CurrentRow => rows.push(ResolvedMvccRow {
            branch_label: branch_label.map(ToOwned::to_owned),
            source_key: Some(provenance_tuple.key.clone()),
            source_tuple: Some(provenance_tuple),
            provenance_path: Some(provenance_path),
            terminal_input_index: Some(terminal_input_index),
            tuple: current,
        }),
        MvccValueChainTerminal::CurrentValuePrefixes => {
            for tuple in
                cuda_visible_key_prefix_rows(all_version_rows, &current.value, visibility, runtime)?
            {
                rows.push(ResolvedMvccRow {
                    branch_label: branch_label.map(ToOwned::to_owned),
                    source_key: Some(provenance_tuple.key.clone()),
                    source_tuple: Some(provenance_tuple.clone()),
                    provenance_path: Some(provenance_path.clone()),
                    terminal_input_index: Some(terminal_input_index),
                    tuple,
                });
            }
        }
    }
    Ok(rows)
}

pub(crate) fn resolve_cuda_native_follow_value_chain_branches_rows(
    all_version_rows: &[ResolvedMvccRow],
    keys: &[String],
    visibility: StorageVisibility,
    plans: &[MvccValueChainPlan],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<ResolvedMvccRow>, FallbackReason> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = cuda_visible_key_rows(all_version_rows, key, visibility, runtime)?
            .into_iter()
            .next()
        else {
            continue;
        };

        match fan_in {
            MvccValueChainBranchFanIn::AllBranches => {
                for plan in plans {
                    rows.extend(resolve_cuda_native_follow_value_chain_from_seed(
                        all_version_rows,
                        &seed,
                        visibility,
                        *plan,
                        provenance,
                        None,
                        runtime,
                    )?);
                }
            }
            MvccValueChainBranchFanIn::FirstNonEmptyBranch => {
                for plan in plans {
                    let branch_rows = resolve_cuda_native_follow_value_chain_from_seed(
                        all_version_rows,
                        &seed,
                        visibility,
                        *plan,
                        provenance,
                        None,
                        runtime,
                    )?;
                    if !branch_rows.is_empty() {
                        rows.extend(branch_rows);
                        break;
                    }
                }
            }
        }
    }
    Ok(rows)
}

pub(crate) fn resolve_cuda_native_follow_value_chain_labeled_branches_rows(
    all_version_rows: &[ResolvedMvccRow],
    keys: &[String],
    visibility: StorageVisibility,
    branches: &[MvccLabeledValueChainBranch],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
    runtime: &CudaDriverRuntime,
) -> Result<Vec<ResolvedMvccRow>, FallbackReason> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = cuda_visible_key_rows(all_version_rows, key, visibility, runtime)?
            .into_iter()
            .next()
        else {
            continue;
        };

        match fan_in {
            MvccValueChainBranchFanIn::AllBranches => {
                for branch in branches {
                    rows.extend(resolve_cuda_native_follow_value_chain_from_seed(
                        all_version_rows,
                        &seed,
                        visibility,
                        branch.plan,
                        provenance,
                        Some(branch.label.as_str()),
                        runtime,
                    )?);
                }
            }
            MvccValueChainBranchFanIn::FirstNonEmptyBranch => {
                for branch in branches {
                    let branch_rows = resolve_cuda_native_follow_value_chain_from_seed(
                        all_version_rows,
                        &seed,
                        visibility,
                        branch.plan,
                        provenance,
                        Some(branch.label.as_str()),
                        runtime,
                    )?;
                    if !branch_rows.is_empty() {
                        rows.extend(branch_rows);
                        break;
                    }
                }
            }
        }
    }
    Ok(rows)
}

pub(crate) fn execute_cuda_native_follow_value_chain_query(
    query: &MvccReadQuery,
    keys: &[String],
    plan: MvccValueChainPlan,
    provenance: MvccSourceProvenance,
    all_version_rows: Vec<ResolvedMvccRow>,
    backend: &CudaMvccExecutionBackend,
) -> Result<FinalizedMvccBackendExecution, FallbackReason> {
    let runtime = ensure_cuda_backend_available(backend)?;
    let rows = resolve_cuda_native_follow_value_chain_rows(
        &all_version_rows,
        keys,
        query.visibility,
        plan,
        provenance,
        runtime,
    )?;

    match backend.execute(query, rows) {
        MvccBackendDispatch::Executed(executed) => Ok(FinalizedMvccBackendExecution {
            executed_target: executed.executed_target,
            fallback_reason: None,
            rows: executed.rows,
        }),
        MvccBackendDispatch::Fallback { reason, .. } => Err(reason),
    }
}

pub(crate) fn execute_cuda_native_follow_value_chain_branches_query(
    query: &MvccReadQuery,
    keys: &[String],
    plans: &[MvccValueChainPlan],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
    all_version_rows: Vec<ResolvedMvccRow>,
    backend: &CudaMvccExecutionBackend,
) -> Result<FinalizedMvccBackendExecution, FallbackReason> {
    let runtime = ensure_cuda_backend_available(backend)?;
    let rows = resolve_cuda_native_follow_value_chain_branches_rows(
        &all_version_rows,
        keys,
        query.visibility,
        plans,
        fan_in,
        provenance,
        runtime,
    )?;

    match backend.execute(query, rows) {
        MvccBackendDispatch::Executed(executed) => Ok(FinalizedMvccBackendExecution {
            executed_target: executed.executed_target,
            fallback_reason: None,
            rows: executed.rows,
        }),
        MvccBackendDispatch::Fallback { reason, .. } => Err(reason),
    }
}

pub(crate) fn execute_cuda_native_follow_value_chain_labeled_branches_query(
    query: &MvccReadQuery,
    keys: &[String],
    branches: &[MvccLabeledValueChainBranch],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
    all_version_rows: Vec<ResolvedMvccRow>,
    backend: &CudaMvccExecutionBackend,
) -> Result<FinalizedMvccBackendExecution, FallbackReason> {
    let runtime = ensure_cuda_backend_available(backend)?;
    let rows = resolve_cuda_native_follow_value_chain_labeled_branches_rows(
        &all_version_rows,
        keys,
        query.visibility,
        branches,
        fan_in,
        provenance,
        runtime,
    )?;

    match backend.execute(query, rows) {
        MvccBackendDispatch::Executed(executed) => Ok(FinalizedMvccBackendExecution {
            executed_target: executed.executed_target,
            fallback_reason: None,
            rows: executed.rows,
        }),
        MvccBackendDispatch::Fallback { reason, .. } => Err(reason),
    }
}

pub(crate) fn sort_projected_key_value_rows(rows: &mut [MvccReadRow], order: &MvccReadOrder) {
    rows.sort_by(|left, right| match order {
        MvccReadOrder::KeyAsc => left.key.cmp(&right.key),
        MvccReadOrder::KeyDesc => right.key.cmp(&left.key),
        MvccReadOrder::ValueAsc => left.value.cmp(&right.value),
        MvccReadOrder::ValueDesc => right.value.cmp(&left.value),
        _ => Ordering::Equal,
    });
}

pub(crate) fn project_key_value_read_row(
    row: MvccReadRow,
    projection: &MvccProjection,
) -> MvccReadRow {
    match projection {
        MvccProjection::KeyValue => row,
        MvccProjection::KeyOnly => MvccReadRow {
            source_key: row.source_key,
            key: row.key,
            value: None,
        },
        MvccProjection::ValueOnly => MvccReadRow {
            source_key: row.source_key,
            key: None,
            value: row.value,
        },
        _ => row,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum FirstCudaSliceGap {
    UnsupportedSource,
    UnsupportedOrder,
    UnsupportedProjection,
    UnsupportedFilter,
    EmptyLogicalFilterTree,
}

#[cfg_attr(not(test), allow(dead_code))]
impl FirstCudaSliceGap {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::UnsupportedSource => "unsupported_source",
            Self::UnsupportedOrder => "unsupported_order",
            Self::UnsupportedProjection => "unsupported_projection",
            Self::UnsupportedFilter => "unsupported_filter",
            Self::EmptyLogicalFilterTree => "empty_logical_filter_tree",
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn first_cuda_slice_filter_gap(filter: &MvccReadFilter) -> Option<FirstCudaSliceGap> {
    match filter {
        MvccReadFilter::All(filters) | MvccReadFilter::Any(filters) if filters.is_empty() => {
            Some(FirstCudaSliceGap::EmptyLogicalFilterTree)
        }
        MvccReadFilter::All(filters) | MvccReadFilter::Any(filters) => {
            filters.iter().find_map(first_cuda_slice_filter_gap)
        }
        _ => None,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn cuda_filter_contains_provenance_predicate(filter: &MvccReadFilter) -> bool {
    match filter {
        MvccReadFilter::ProvenanceKeyPrefix { .. }
        | MvccReadFilter::ProvenanceBundleKeyEquals { .. }
        | MvccReadFilter::ProvenanceBundleKeyCountAtLeast { .. }
        | MvccReadFilter::ProvenanceBundleKeyPrefix { .. }
        | MvccReadFilter::ProvenanceValueEquals { .. }
        | MvccReadFilter::ProvenanceBundleValueEquals { .. }
        | MvccReadFilter::ProvenanceBundleValueCountAtLeast { .. }
        | MvccReadFilter::ProvenanceBundleKeyValueEquals { .. }
        | MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast { .. }
        | MvccReadFilter::ProvenanceBundlePathEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathContains { .. }
        | MvccReadFilter::ProvenanceBundlePathCountAtLeast { .. }
        | MvccReadFilter::ProvenanceBundlePathPairAtDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathSuffixEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathPrefixEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathSliceEquals { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
            ..
        }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrencePairAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
            ..
        }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistance { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedAt { .. }
        | MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt { .. }
        | MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin { .. }
        | MvccReadFilter::ProvenanceBundlePathSegmentEquals { .. }
        | MvccReadFilter::ProvenanceBundleLenEquals { .. } => true,
        MvccReadFilter::All(filters) | MvccReadFilter::Any(filters) => filters
            .iter()
            .any(cuda_filter_contains_provenance_predicate),
        _ => false,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn cuda_filter_contains_cpu_resolved_predicate(filter: &MvccReadFilter) -> bool {
    match filter {
        MvccReadFilter::SourceKeyPrefix(_)
        | MvccReadFilter::SourceValueEquals(_)
        | MvccReadFilter::BranchLabelEquals(_) => true,
        MvccReadFilter::All(filters) | MvccReadFilter::Any(filters) => filters
            .iter()
            .any(cuda_filter_contains_cpu_resolved_predicate),
        _ => cuda_filter_contains_provenance_predicate(filter),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_cuda_cpu_resolved_source(source: &MvccReadSource) -> bool {
    match source {
        MvccReadSource::Concat { sources }
        | MvccReadSource::ConcatDistinct { sources }
        | MvccReadSource::IntersectDistinct { sources }
        | MvccReadSource::IntersectAll { sources }
        | MvccReadSource::ExceptDistinct { sources }
        | MvccReadSource::ExceptAll { sources }
        | MvccReadSource::SymmetricDifferenceDistinct { sources }
        | MvccReadSource::SymmetricDifferenceAll { sources } => {
            sources.iter().all(is_cuda_cpu_resolved_source)
        }
        MvccReadSource::FollowValueChain { .. }
        | MvccReadSource::FollowValueChainBranches { .. }
        | MvccReadSource::FollowValueChainLabeledBranches { .. }
        | MvccReadSource::FollowValueKeyRefs { .. }
        | MvccReadSource::FollowValueKeyPrefixes { .. }
        | MvccReadSource::FollowValueKeyRefPrefixes { .. }
        | MvccReadSource::FollowValueKeyRefValueKeyRefs { .. }
        | MvccReadSource::FollowValueKeyRefValueKeyPrefixes { .. }
        | MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes { .. }
        | MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs { .. }
        | MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes { .. }
        | MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes { .. } => true,
        _ => false,
    }
}

pub(crate) fn is_cuda_native_single_source(source: &MvccReadSource) -> bool {
    matches!(
        source,
        MvccReadSource::FullScan
            | MvccReadSource::KeyLookup { .. }
            | MvccReadSource::KeyBatchLookup { .. }
    )
}

pub(crate) fn is_cuda_native_concat_source(source: &MvccReadSource) -> bool {
    matches!(source, MvccReadSource::Concat { sources } if sources.iter().all(is_cuda_native_source_resolvable))
}

pub(crate) fn is_cuda_native_follow_value_chain_source(source: &MvccReadSource) -> bool {
    matches!(
        source,
        MvccReadSource::FollowValueChain { .. }
            | MvccReadSource::FollowValueChainBranches { .. }
            | MvccReadSource::FollowValueChainLabeledBranches { .. }
    )
}

pub(crate) fn is_cuda_native_source_resolvable(source: &MvccReadSource) -> bool {
    match source {
        source
            if is_cuda_native_single_source(source)
                || is_cuda_native_concat_source(source)
                || is_cuda_native_follow_value_chain_source(source) =>
        {
            true
        }
        MvccReadSource::ConcatDistinct { sources }
        | MvccReadSource::IntersectDistinct { sources }
        | MvccReadSource::IntersectAll { sources }
        | MvccReadSource::ExceptDistinct { sources }
        | MvccReadSource::ExceptAll { sources }
        | MvccReadSource::SymmetricDifferenceDistinct { sources }
        | MvccReadSource::SymmetricDifferenceAll { sources } => {
            sources.iter().all(is_cuda_native_source_resolvable)
        }
        _ => false,
    }
}

pub(crate) fn is_cuda_native_composition_source(source: &MvccReadSource) -> bool {
    matches!(
        source,
        MvccReadSource::ConcatDistinct { sources }
            | MvccReadSource::IntersectDistinct { sources }
            | MvccReadSource::IntersectAll { sources }
            | MvccReadSource::ExceptDistinct { sources }
            | MvccReadSource::ExceptAll { sources }
            | MvccReadSource::SymmetricDifferenceDistinct { sources }
            | MvccReadSource::SymmetricDifferenceAll { sources }
            if sources.iter().all(is_cuda_native_source_resolvable)
    )
}

pub(crate) fn is_cuda_order_supported(query: &MvccReadQuery, order: &MvccReadOrder) -> bool {
    if matches!(
        order,
        MvccReadOrder::KeyAsc
            | MvccReadOrder::KeyDesc
            | MvccReadOrder::ValueAsc
            | MvccReadOrder::ValueDesc
    ) && matches!(
        &query.source,
        MvccReadSource::FullScan
            | MvccReadSource::KeyLookup { .. }
            | MvccReadSource::KeyBatchLookup { .. }
    ) {
        return true;
    }

    if matches!(
        order,
        MvccReadOrder::KeyAsc
            | MvccReadOrder::KeyDesc
            | MvccReadOrder::ValueAsc
            | MvccReadOrder::ValueDesc
    ) && (is_cuda_native_concat_source(&query.source)
        || is_cuda_native_composition_source(&query.source))
    {
        return true;
    }

    is_cuda_cpu_resolved_source(&query.source)
        && matches!(
            order,
            MvccReadOrder::KeyAsc
                | MvccReadOrder::KeyDesc
                | MvccReadOrder::ValueAsc
                | MvccReadOrder::ValueDesc
                | MvccReadOrder::BranchLabelAsc
                | MvccReadOrder::BranchLabelDesc
                | MvccReadOrder::SourceKeyAsc
                | MvccReadOrder::SourceKeyDesc
                | MvccReadOrder::SourceValueAsc
                | MvccReadOrder::SourceValueDesc
                | MvccReadOrder::ProvenanceKeyAsc { .. }
                | MvccReadOrder::ProvenanceKeyDesc { .. }
                | MvccReadOrder::ProvenanceValueAsc { .. }
                | MvccReadOrder::ProvenanceValueDesc { .. }
                | MvccReadOrder::ProvenanceBundleKeyPathAsc { .. }
                | MvccReadOrder::ProvenanceBundleKeyPathDesc { .. }
                | MvccReadOrder::ProvenanceBundleValuePathAsc { .. }
                | MvccReadOrder::ProvenanceBundleValuePathDesc { .. }
                | MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc { .. }
                | MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc { .. }
                | MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc { .. }
                | MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceDesc { .. }
                | MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc { .. }
                | MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairDesc { .. }
        )
}

pub(crate) fn is_cuda_projection_supported(query: &MvccReadQuery) -> bool {
    if matches!(
        query.projection,
        MvccProjection::KeyValue | MvccProjection::KeyOnly | MvccProjection::ValueOnly
    ) {
        return true;
    }

    is_cuda_cpu_resolved_source(&query.source)
        && matches!(
            query.projection,
            MvccProjection::BranchLabelTargetValue
                | MvccProjection::SourceKeyTargetValue
                | MvccProjection::SourceValueOnly
                | MvccProjection::TargetKeySourceValue
                | MvccProjection::TargetKeyProvenanceValue { .. }
                | MvccProjection::TargetKeyProvenanceSummary { .. }
                | MvccProjection::TargetKeyProvenanceBundleSummary { .. }
                | MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset { .. }
                | MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance { .. }
                | MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair { .. }
        )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn first_cuda_slice_query_gap(query: &MvccReadQuery) -> Option<FirstCudaSliceGap> {
    let native_source = is_cuda_native_single_source(&query.source)
        || is_cuda_native_concat_source(&query.source)
        || is_cuda_native_follow_value_chain_source(&query.source)
        || is_cuda_native_composition_source(&query.source);
    let cpu_resolved_source = is_cuda_cpu_resolved_source(&query.source);
    if !native_source && !cpu_resolved_source {
        return Some(FirstCudaSliceGap::UnsupportedSource);
    }

    if query
        .order
        .as_ref()
        .is_some_and(|order| !is_cuda_order_supported(query, order))
    {
        return Some(FirstCudaSliceGap::UnsupportedOrder);
    }

    if !is_cuda_projection_supported(query) {
        return Some(FirstCudaSliceGap::UnsupportedProjection);
    }

    if !native_source
        && cpu_resolved_source
        && !query
            .filter
            .as_ref()
            .is_some_and(cuda_filter_contains_cpu_resolved_predicate)
    {
        return Some(FirstCudaSliceGap::UnsupportedFilter);
    }

    query.filter.as_ref().and_then(first_cuda_slice_filter_gap)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_first_cuda_slice_query(query: &MvccReadQuery) -> bool {
    first_cuda_slice_query_gap(query).is_none()
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_cuda_native_full_scan_query(query: &MvccReadQuery) -> bool {
    matches!(query.source, MvccReadSource::FullScan) && is_first_cuda_slice_query(query)
}

pub(crate) fn is_cuda_native_source_query(query: &MvccReadQuery) -> bool {
    (is_cuda_native_single_source(&query.source)
        || is_cuda_native_concat_source(&query.source)
        || is_cuda_native_follow_value_chain_source(&query.source)
        || is_cuda_native_composition_source(&query.source))
        && is_first_cuda_slice_query(query)
}

pub(crate) type ResolvedTupleIdentity = (u64, String, String, u64, Option<u64>);
pub(crate) type ResolvedMvccRowIdentity = (
    Option<String>,
    Option<String>,
    Option<ResolvedTupleIdentity>,
    Option<Vec<ResolvedTupleIdentity>>,
    Option<usize>,
    ResolvedTupleIdentity,
);

pub(crate) fn resolved_tuple_identity(tuple: &TupleVersion) -> ResolvedTupleIdentity {
    (
        tuple.tuple_id,
        tuple.key.clone(),
        tuple.value.clone(),
        tuple.created_by,
        tuple.deleted_by,
    )
}

pub(crate) fn resolved_mvcc_row_provenance_tuple(
    row: &ResolvedMvccRow,
    frame: MvccProvenanceFrame,
) -> Option<&TupleVersion> {
    let path = row.provenance_path.as_ref()?;
    match frame {
        MvccProvenanceFrame::Seed => path.first(),
        MvccProvenanceFrame::TerminalInput => {
            row.terminal_input_index.and_then(|index| path.get(index))
        }
        MvccProvenanceFrame::ValueHop(index) => path.get(index),
    }
}

pub(crate) fn resolved_mvcc_row_provenance_bundle(
    row: &ResolvedMvccRow,
    bundle: MvccProvenanceFrameBundle,
) -> Option<Vec<&TupleVersion>> {
    let path = row.provenance_path.as_ref()?;
    match bundle {
        MvccProvenanceFrameBundle::SeedThroughTerminalInput => {
            let end = row.terminal_input_index?;
            Some(path.iter().take(end + 1).collect())
        }
        MvccProvenanceFrameBundle::FullPath => Some(path.iter().collect()),
    }
}

pub(crate) fn summarize_mvcc_row_provenance_path(
    row: &ResolvedMvccRow,
    summary: MvccProvenanceSummary,
) -> Option<String> {
    summarize_mvcc_provenance_tuples(row.provenance_path.as_ref()?.iter(), summary)
}

pub(crate) fn summarize_mvcc_row_provenance_bundle(
    row: &ResolvedMvccRow,
    bundle: MvccProvenanceFrameBundle,
    summary: MvccProvenanceSummary,
) -> Option<String> {
    let segments = resolved_mvcc_row_provenance_bundle_segments(row, bundle, summary)?;
    Some(segments.join(" -> "))
}

pub(crate) fn resolved_mvcc_row_provenance_bundle_segments(
    row: &ResolvedMvccRow,
    bundle: MvccProvenanceFrameBundle,
    summary: MvccProvenanceSummary,
) -> Option<Vec<String>> {
    let tuples = resolved_mvcc_row_provenance_bundle(row, bundle)?;
    Some(collect_mvcc_provenance_segments(
        tuples.into_iter(),
        summary,
    ))
}

pub(crate) fn mvcc_provenance_segments_occurrence_start(
    segments: &[String],
    expected: &[String],
    occurrence: MvccProvenanceOccurrence,
) -> Option<usize> {
    match occurrence {
        MvccProvenanceOccurrence::First => {
            mvcc_provenance_segments_first_subpath_start(segments, expected)
        }
        MvccProvenanceOccurrence::Last => {
            mvcc_provenance_segments_last_subpath_start(segments, expected)
        }
        MvccProvenanceOccurrence::Nth(index) => {
            mvcc_provenance_segments_nth_subpath_start(segments, expected, index)
        }
    }
}

pub(crate) fn resolved_mvcc_row_provenance_bundle_occurrence_offset(
    row: &ResolvedMvccRow,
    bundle: MvccProvenanceFrameBundle,
    summary: MvccProvenanceSummary,
    expected: &[String],
    occurrence: MvccProvenanceOccurrence,
) -> Option<usize> {
    let segments = resolved_mvcc_row_provenance_bundle_segments(row, bundle, summary)?;
    mvcc_provenance_segments_occurrence_start(&segments, expected, occurrence)
}

pub(crate) fn resolved_mvcc_row_provenance_bundle_occurrence_distance(
    row: &ResolvedMvccRow,
    bundle: MvccProvenanceFrameBundle,
    summary: MvccProvenanceSummary,
    left_expected: &[String],
    left_occurrence: MvccProvenanceOccurrence,
    right_expected: &[String],
    right_occurrence: MvccProvenanceOccurrence,
) -> Option<usize> {
    let segments = resolved_mvcc_row_provenance_bundle_segments(row, bundle, summary)?;
    let left_start =
        mvcc_provenance_segments_occurrence_start(&segments, left_expected, left_occurrence)?;
    let right_start =
        mvcc_provenance_segments_occurrence_start(&segments, right_expected, right_occurrence)?;
    right_start.checked_sub(left_start)
}

pub(crate) fn resolved_mvcc_row_provenance_bundle_mixed_occurrence_offset_pair(
    row: &ResolvedMvccRow,
    bundle: MvccProvenanceFrameBundle,
    summary: MvccProvenanceSummary,
    left_expected: &[String],
    left_occurrence: MvccProvenanceOccurrence,
    right_expected: &[String],
    right_occurrence: MvccProvenanceOccurrence,
) -> Option<(usize, usize)> {
    let segments = resolved_mvcc_row_provenance_bundle_segments(row, bundle, summary)?;
    let left_start =
        mvcc_provenance_segments_occurrence_start(&segments, left_expected, left_occurrence)?;
    let right_start =
        mvcc_provenance_segments_occurrence_start(&segments, right_expected, right_occurrence)?;
    Some((left_start, right_start))
}

pub(crate) fn summarize_mvcc_provenance_tuples<'a>(
    tuples: impl Iterator<Item = &'a TupleVersion>,
    summary: MvccProvenanceSummary,
) -> Option<String> {
    let segments = collect_mvcc_provenance_segments(tuples, summary);
    Some(segments.join(" -> "))
}

pub(crate) fn mvcc_provenance_segments_contain_ordered_subpath(
    segments: &[String],
    expected: &[String],
) -> bool {
    mvcc_provenance_segments_ordered_subpath_count(segments, expected) > 0
}

pub(crate) fn mvcc_provenance_segments_ordered_subpath_count(
    segments: &[String],
    expected: &[String],
) -> usize {
    if expected.is_empty() || expected.len() > segments.len() {
        return 0;
    }

    segments
        .windows(expected.len())
        .filter(|window| *window == expected)
        .count()
}

pub(crate) fn mvcc_provenance_segments_have_pair_at_distance(
    segments: &[String],
    left: &str,
    right: &str,
    distance: usize,
) -> bool {
    segments.iter().enumerate().any(|(index, segment)| {
        segment == left
            && segments
                .get(index.saturating_add(distance))
                .is_some_and(|candidate| candidate == right)
    })
}

pub(crate) fn mvcc_provenance_segments_have_suffix(
    segments: &[String],
    expected: &[String],
) -> bool {
    !expected.is_empty()
        && expected.len() <= segments.len()
        && segments[segments.len() - expected.len()..] == *expected
}

pub(crate) fn mvcc_provenance_segments_have_prefix(
    segments: &[String],
    expected: &[String],
) -> bool {
    !expected.is_empty()
        && expected.len() <= segments.len()
        && segments[..expected.len()] == *expected
}

pub(crate) fn mvcc_provenance_segments_have_slice_at(
    segments: &[String],
    start: usize,
    expected: &[String],
) -> bool {
    !expected.is_empty()
        && start
            .checked_add(expected.len())
            .is_some_and(|end| end <= segments.len() && segments[start..end] == *expected)
}

pub(crate) fn mvcc_provenance_segments_first_subpath_start(
    segments: &[String],
    expected: &[String],
) -> Option<usize> {
    if expected.is_empty() || expected.len() > segments.len() {
        return None;
    }

    segments
        .windows(expected.len())
        .position(|window| window == expected)
}

pub(crate) fn mvcc_provenance_segments_last_subpath_start(
    segments: &[String],
    expected: &[String],
) -> Option<usize> {
    if expected.is_empty() || expected.len() > segments.len() {
        return None;
    }

    segments
        .windows(expected.len())
        .rposition(|window| window == expected)
}

pub(crate) fn mvcc_provenance_segments_nth_subpath_start(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
) -> Option<usize> {
    if expected.is_empty() || expected.len() > segments.len() {
        return None;
    }

    segments
        .windows(expected.len())
        .enumerate()
        .filter_map(|(index, window)| (window == expected).then_some(index))
        .nth(occurrence_index)
}

pub(crate) fn mvcc_provenance_segments_start_within(
    actual_start: Option<usize>,
    start_min: usize,
    start_max: usize,
) -> bool {
    if start_min > start_max {
        return false;
    }

    actual_start.is_some_and(|start| (start_min..=start_max).contains(&start))
}

pub(crate) fn mvcc_provenance_segments_nth_subpath_start_within(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
    start_min: usize,
    start_max: usize,
) -> bool {
    mvcc_provenance_segments_start_within(
        mvcc_provenance_segments_nth_subpath_start(segments, expected, occurrence_index),
        start_min,
        start_max,
    )
}

pub(crate) fn mvcc_provenance_segments_occurrence_distance(
    segments: &[String],
    expected: &[String],
    left_occurrence_index: usize,
    right_occurrence_index: usize,
) -> Option<usize> {
    mvcc_provenance_segments_mixed_occurrence_distance(
        segments,
        expected,
        left_occurrence_index,
        expected,
        right_occurrence_index,
    )
}

pub(crate) fn mvcc_provenance_segments_first_occurrence_distance(
    segments: &[String],
    expected: &[String],
) -> Option<usize> {
    mvcc_provenance_segments_occurrence_distance(segments, expected, 0, 1)
}

pub(crate) fn mvcc_provenance_segments_first_occurrence_distance_within(
    segments: &[String],
    expected: &[String],
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_first_occurrence_distance(segments, expected)
        .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_last_occurrence_distance(
    segments: &[String],
    expected: &[String],
) -> Option<usize> {
    let starts = if expected.is_empty() || expected.len() > segments.len() {
        return None;
    } else {
        segments
            .windows(expected.len())
            .enumerate()
            .filter_map(|(index, window)| (window == expected).then_some(index))
            .collect::<Vec<_>>()
    };

    starts
        .get(starts.len().checked_sub(2)?)
        .zip(starts.last())
        .and_then(|(left_start, right_start)| right_start.checked_sub(*left_start))
}

pub(crate) fn mvcc_provenance_segments_last_occurrence_distance_within(
    segments: &[String],
    expected: &[String],
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_last_occurrence_distance(segments, expected)
        .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_first_occurrence_to_occurrence_distance(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
) -> Option<usize> {
    mvcc_provenance_segments_occurrence_distance(segments, expected, 0, occurrence_index)
}

pub(crate) fn mvcc_provenance_segments_first_occurrence_to_occurrence_distance_within(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_first_occurrence_to_occurrence_distance(
        segments,
        expected,
        occurrence_index,
    )
    .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_first_occurrence_to_occurrence_at(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
    first_start: usize,
    occurrence_start: usize,
) -> bool {
    mvcc_provenance_segments_first_subpath_start(segments, expected) == Some(first_start)
        && mvcc_provenance_segments_nth_subpath_start(segments, expected, occurrence_index)
            == Some(occurrence_start)
}

pub(crate) fn mvcc_provenance_segments_first_occurrence_to_occurrence_within(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
    first_start_range: std::ops::RangeInclusive<usize>,
    occurrence_start_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if first_start_range.is_empty() || occurrence_start_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_first_subpath_start(segments, expected)
        .is_some_and(|start| first_start_range.contains(&start))
        && mvcc_provenance_segments_nth_subpath_start(segments, expected, occurrence_index)
            .is_some_and(|start| occurrence_start_range.contains(&start))
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_distance(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
) -> Option<usize> {
    let starts = if expected.is_empty() || expected.len() > segments.len() {
        return None;
    } else {
        segments
            .windows(expected.len())
            .enumerate()
            .filter_map(|(index, window)| (window == expected).then_some(index))
            .collect::<Vec<_>>()
    };

    starts
        .get(occurrence_index)
        .zip(starts.last())
        .and_then(|(left_start, right_start)| right_start.checked_sub(*left_start))
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_distance_within(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_occurrence_to_last_distance(segments, expected, occurrence_index)
        .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_at(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
    occurrence_start: usize,
    last_start: usize,
) -> bool {
    mvcc_provenance_segments_nth_subpath_start(segments, expected, occurrence_index)
        == Some(occurrence_start)
        && mvcc_provenance_segments_last_subpath_start(segments, expected) == Some(last_start)
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_within(
    segments: &[String],
    expected: &[String],
    occurrence_index: usize,
    occurrence_start_range: std::ops::RangeInclusive<usize>,
    last_start_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if occurrence_start_range.is_empty() || last_start_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_nth_subpath_start(segments, expected, occurrence_index)
        .is_some_and(|start| occurrence_start_range.contains(&start))
        && mvcc_provenance_segments_last_subpath_start(segments, expected)
            .is_some_and(|start| last_start_range.contains(&start))
}

pub(crate) fn mvcc_provenance_segments_occurrence_pair_at(
    segments: &[String],
    expected: &[String],
    left_occurrence_index: usize,
    left_start: usize,
    right_occurrence_index: usize,
    right_start: usize,
) -> bool {
    mvcc_provenance_segments_nth_subpath_start(segments, expected, left_occurrence_index)
        == Some(left_start)
        && mvcc_provenance_segments_nth_subpath_start(segments, expected, right_occurrence_index)
            == Some(right_start)
}

pub(crate) fn mvcc_provenance_segments_occurrence_pair_within(
    segments: &[String],
    expected: &[String],
    left_occurrence_index: usize,
    left_start_range: std::ops::RangeInclusive<usize>,
    right_occurrence_index: usize,
    right_start_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if left_start_range.is_empty() || right_start_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_nth_subpath_start(segments, expected, left_occurrence_index)
        .is_some_and(|start| left_start_range.contains(&start))
        && mvcc_provenance_segments_nth_subpath_start(segments, expected, right_occurrence_index)
            .is_some_and(|start| right_start_range.contains(&start))
}

pub(crate) fn mvcc_provenance_segments_mixed_occurrence_distance(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    right_expected: &[String],
    right_occurrence_index: usize,
) -> Option<usize> {
    let left_start =
        mvcc_provenance_segments_nth_subpath_start(segments, left_expected, left_occurrence_index)?;
    let right_start = mvcc_provenance_segments_nth_subpath_start(
        segments,
        right_expected,
        right_occurrence_index,
    )?;
    right_start.checked_sub(left_start)
}

pub(crate) fn mvcc_provenance_segments_occurrence_distance_within(
    segments: &[String],
    expected: &[String],
    left_occurrence_index: usize,
    right_occurrence_index: usize,
    min_distance: usize,
    max_distance: usize,
) -> bool {
    mvcc_provenance_segments_mixed_occurrence_distance_within(
        segments,
        expected,
        left_occurrence_index,
        expected,
        right_occurrence_index,
        min_distance,
        max_distance,
    )
}

pub(crate) fn mvcc_provenance_segments_mixed_occurrence_distance_within(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    right_expected: &[String],
    right_occurrence_index: usize,
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_mixed_occurrence_distance(
        segments,
        left_expected,
        left_occurrence_index,
        right_expected,
        right_occurrence_index,
    )
    .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_distance(
    segments: &[String],
    left_expected: &[String],
    right_expected: &[String],
) -> Option<usize> {
    let left_start = mvcc_provenance_segments_first_subpath_start(segments, left_expected)?;
    let right_start = mvcc_provenance_segments_first_subpath_start(segments, right_expected)?;
    right_start.checked_sub(left_start)
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_distance_within(
    segments: &[String],
    left_expected: &[String],
    right_expected: &[String],
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_first_mixed_occurrence_distance(
        segments,
        left_expected,
        right_expected,
    )
    .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_last_mixed_occurrence_distance(
    segments: &[String],
    left_expected: &[String],
    right_expected: &[String],
) -> Option<usize> {
    let left_start = mvcc_provenance_segments_last_subpath_start(segments, left_expected)?;
    let right_start = mvcc_provenance_segments_last_subpath_start(segments, right_expected)?;
    right_start.checked_sub(left_start)
}

pub(crate) fn mvcc_provenance_segments_last_mixed_occurrence_distance_within(
    segments: &[String],
    left_expected: &[String],
    right_expected: &[String],
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_last_mixed_occurrence_distance(segments, left_expected, right_expected)
        .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_distance(
    segments: &[String],
    left_expected: &[String],
    right_expected: &[String],
    right_occurrence_index: usize,
) -> Option<usize> {
    let left_start = mvcc_provenance_segments_first_subpath_start(segments, left_expected)?;
    let right_start = mvcc_provenance_segments_nth_subpath_start(
        segments,
        right_expected,
        right_occurrence_index,
    )?;
    right_start.checked_sub(left_start)
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_distance_within(
    segments: &[String],
    left_expected: &[String],
    right_expected: &[String],
    right_occurrence_index: usize,
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_distance(
        segments,
        left_expected,
        right_expected,
        right_occurrence_index,
    )
    .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_mixed_distance(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    right_expected: &[String],
) -> Option<usize> {
    let left_start =
        mvcc_provenance_segments_nth_subpath_start(segments, left_expected, left_occurrence_index)?;
    let right_start = mvcc_provenance_segments_last_subpath_start(segments, right_expected)?;
    right_start.checked_sub(left_start)
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_mixed_distance_within(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    right_expected: &[String],
    min_distance: usize,
    max_distance: usize,
) -> bool {
    if min_distance > max_distance {
        return false;
    }

    mvcc_provenance_segments_occurrence_to_last_mixed_distance(
        segments,
        left_expected,
        left_occurrence_index,
        right_expected,
    )
    .is_some_and(|distance| (min_distance..=max_distance).contains(&distance))
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_at(
    segments: &[String],
    left_expected: &[String],
    left_start: usize,
    right_expected: &[String],
    right_occurrence_index: usize,
    right_start: usize,
) -> bool {
    mvcc_provenance_segments_first_subpath_start(segments, left_expected) == Some(left_start)
        && mvcc_provenance_segments_nth_subpath_start(
            segments,
            right_expected,
            right_occurrence_index,
        ) == Some(right_start)
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_within(
    segments: &[String],
    left_expected: &[String],
    left_start_range: std::ops::RangeInclusive<usize>,
    right_expected: &[String],
    right_occurrence_index: usize,
    right_start_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if left_start_range.is_empty() || right_start_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_first_subpath_start(segments, left_expected)
        .is_some_and(|start| left_start_range.contains(&start))
        && mvcc_provenance_segments_nth_subpath_start(
            segments,
            right_expected,
            right_occurrence_index,
        )
        .is_some_and(|start| right_start_range.contains(&start))
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_mixed_at(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    left_start: usize,
    right_expected: &[String],
    right_start: usize,
) -> bool {
    mvcc_provenance_segments_nth_subpath_start(segments, left_expected, left_occurrence_index)
        == Some(left_start)
        && mvcc_provenance_segments_last_subpath_start(segments, right_expected)
            == Some(right_start)
}

pub(crate) fn mvcc_provenance_segments_occurrence_to_last_mixed_within(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    left_start_range: std::ops::RangeInclusive<usize>,
    right_expected: &[String],
    right_start_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if left_start_range.is_empty() || right_start_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_nth_subpath_start(segments, left_expected, left_occurrence_index)
        .is_some_and(|start| left_start_range.contains(&start))
        && mvcc_provenance_segments_last_subpath_start(segments, right_expected)
            .is_some_and(|start| right_start_range.contains(&start))
}

pub(crate) fn mvcc_provenance_segments_mixed_occurrence_at(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    left_start: usize,
    right_expected: &[String],
    right_occurrence_index: usize,
    right_start: usize,
) -> bool {
    mvcc_provenance_segments_nth_subpath_start(segments, left_expected, left_occurrence_index)
        == Some(left_start)
        && mvcc_provenance_segments_nth_subpath_start(
            segments,
            right_expected,
            right_occurrence_index,
        ) == Some(right_start)
}

pub(crate) fn mvcc_provenance_segments_mixed_occurrence_within(
    segments: &[String],
    left_expected: &[String],
    left_occurrence_index: usize,
    left_range: std::ops::RangeInclusive<usize>,
    right_expected: &[String],
    right_occurrence_index: usize,
    right_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if left_range.is_empty() || right_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_nth_subpath_start(segments, left_expected, left_occurrence_index)
        .is_some_and(|start| left_range.contains(&start))
        && mvcc_provenance_segments_nth_subpath_start(
            segments,
            right_expected,
            right_occurrence_index,
        )
        .is_some_and(|start| right_range.contains(&start))
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_at(
    segments: &[String],
    left_expected: &[String],
    left_start: usize,
    right_expected: &[String],
    right_start: usize,
) -> bool {
    mvcc_provenance_segments_first_subpath_start(segments, left_expected) == Some(left_start)
        && mvcc_provenance_segments_first_subpath_start(segments, right_expected)
            == Some(right_start)
}

pub(crate) fn mvcc_provenance_segments_first_mixed_occurrence_within(
    segments: &[String],
    left_expected: &[String],
    left_range: std::ops::RangeInclusive<usize>,
    right_expected: &[String],
    right_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if left_range.is_empty() || right_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_first_subpath_start(segments, left_expected)
        .is_some_and(|start| left_range.contains(&start))
        && mvcc_provenance_segments_first_subpath_start(segments, right_expected)
            .is_some_and(|start| right_range.contains(&start))
}

pub(crate) fn mvcc_provenance_segments_last_mixed_occurrence_at(
    segments: &[String],
    left_expected: &[String],
    left_start: usize,
    right_expected: &[String],
    right_start: usize,
) -> bool {
    mvcc_provenance_segments_last_subpath_start(segments, left_expected) == Some(left_start)
        && mvcc_provenance_segments_last_subpath_start(segments, right_expected)
            == Some(right_start)
}

pub(crate) fn mvcc_provenance_segments_last_mixed_occurrence_within(
    segments: &[String],
    left_expected: &[String],
    left_range: std::ops::RangeInclusive<usize>,
    right_expected: &[String],
    right_range: std::ops::RangeInclusive<usize>,
) -> bool {
    if left_range.is_empty() || right_range.is_empty() {
        return false;
    }

    mvcc_provenance_segments_last_subpath_start(segments, left_expected)
        .is_some_and(|start| left_range.contains(&start))
        && mvcc_provenance_segments_last_subpath_start(segments, right_expected)
            .is_some_and(|start| right_range.contains(&start))
}

pub(crate) fn mvcc_provenance_tuple_count_at_least<'a>(
    tuples: impl Iterator<Item = &'a TupleVersion>,
    min_count: usize,
    mut predicate: impl FnMut(&TupleVersion) -> bool,
) -> bool {
    tuples
        .filter(|tuple| predicate(tuple))
        .take(min_count)
        .count()
        >= min_count
}

pub(crate) fn collect_mvcc_provenance_segments<'a>(
    tuples: impl Iterator<Item = &'a TupleVersion>,
    summary: MvccProvenanceSummary,
) -> Vec<String> {
    tuples
        .map(|tuple| match summary {
            MvccProvenanceSummary::KeyPath => tuple.key.clone(),
            MvccProvenanceSummary::ValuePath => tuple.value.clone(),
            MvccProvenanceSummary::KeyValuePath => format!("{}={}", tuple.key, tuple.value),
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn collect_operator_rows<Row, Op>(mut operator: Op) -> Vec<Row>
where
    Op: Operator<Row>,
{
    operator.open();
    let mut rows = Vec::new();
    while let Some(row) = operator.next() {
        rows.push(row);
    }
    operator.close();
    rows
}
