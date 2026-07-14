use super::{
    is_cuda_cpu_resolved_source, is_cuda_native_composition_source, mvcc_row_cmp,
    mvcc_row_matches_filter, project_mvcc_row, resolved_mvcc_row_provenance_bundle,
    resolved_mvcc_row_provenance_tuple, CudaDriverRuntime, CudaMvccRowBatch, DeviceTarget,
    MvccBackendExecution, MvccProvenanceFrame, MvccProvenanceFrameBundle, MvccReadFilter,
    MvccReadQuery, MvccReadSource, ResolvedMvccRow, TupleVersion,
};

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
