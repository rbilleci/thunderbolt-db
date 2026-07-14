use super::{
    mvcc_provenance_segments_contain_ordered_subpath,
    mvcc_provenance_segments_first_mixed_occurrence_at,
    mvcc_provenance_segments_first_mixed_occurrence_distance,
    mvcc_provenance_segments_first_mixed_occurrence_distance_within,
    mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_at,
    mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_distance,
    mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_distance_within,
    mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_within,
    mvcc_provenance_segments_first_mixed_occurrence_within,
    mvcc_provenance_segments_first_occurrence_distance,
    mvcc_provenance_segments_first_occurrence_distance_within,
    mvcc_provenance_segments_first_occurrence_to_occurrence_at,
    mvcc_provenance_segments_first_occurrence_to_occurrence_distance,
    mvcc_provenance_segments_first_occurrence_to_occurrence_distance_within,
    mvcc_provenance_segments_first_occurrence_to_occurrence_within,
    mvcc_provenance_segments_first_subpath_start, mvcc_provenance_segments_have_pair_at_distance,
    mvcc_provenance_segments_have_prefix, mvcc_provenance_segments_have_slice_at,
    mvcc_provenance_segments_have_suffix, mvcc_provenance_segments_last_mixed_occurrence_at,
    mvcc_provenance_segments_last_mixed_occurrence_distance,
    mvcc_provenance_segments_last_mixed_occurrence_distance_within,
    mvcc_provenance_segments_last_mixed_occurrence_within,
    mvcc_provenance_segments_last_occurrence_distance,
    mvcc_provenance_segments_last_occurrence_distance_within,
    mvcc_provenance_segments_last_subpath_start, mvcc_provenance_segments_mixed_occurrence_at,
    mvcc_provenance_segments_mixed_occurrence_distance,
    mvcc_provenance_segments_mixed_occurrence_distance_within,
    mvcc_provenance_segments_mixed_occurrence_within, mvcc_provenance_segments_nth_subpath_start,
    mvcc_provenance_segments_nth_subpath_start_within,
    mvcc_provenance_segments_occurrence_distance,
    mvcc_provenance_segments_occurrence_distance_within,
    mvcc_provenance_segments_occurrence_pair_at, mvcc_provenance_segments_occurrence_pair_within,
    mvcc_provenance_segments_occurrence_to_last_at,
    mvcc_provenance_segments_occurrence_to_last_distance,
    mvcc_provenance_segments_occurrence_to_last_distance_within,
    mvcc_provenance_segments_occurrence_to_last_mixed_at,
    mvcc_provenance_segments_occurrence_to_last_mixed_distance,
    mvcc_provenance_segments_occurrence_to_last_mixed_distance_within,
    mvcc_provenance_segments_occurrence_to_last_mixed_within,
    mvcc_provenance_segments_occurrence_to_last_within,
    mvcc_provenance_segments_ordered_subpath_count, mvcc_provenance_segments_start_within,
    mvcc_provenance_tuple_count_at_least, resolved_mvcc_row_provenance_bundle,
    resolved_mvcc_row_provenance_bundle_mixed_occurrence_offset_pair,
    resolved_mvcc_row_provenance_bundle_occurrence_distance,
    resolved_mvcc_row_provenance_bundle_occurrence_offset,
    resolved_mvcc_row_provenance_bundle_segments, resolved_mvcc_row_provenance_tuple,
    resolved_tuple_identity, summarize_mvcc_row_provenance_bundle,
    summarize_mvcc_row_provenance_path, CudaMvccRowBatch, MvccProjection, MvccProvenanceSummary,
    MvccReadFilter, MvccReadOrder, MvccReadRow, ResolvedMvccRow, ResolvedMvccRowIdentity,
};

pub(crate) fn project_mvcc_row(row: ResolvedMvccRow, projection: &MvccProjection) -> MvccReadRow {
    let provenance_value = match projection {
        MvccProjection::TargetKeyProvenanceValue { frame } => {
            resolved_mvcc_row_provenance_tuple(&row, *frame).map(|tuple| tuple.value.clone())
        }
        MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
            bundle,
            summary,
            expected,
            occurrence,
        } => resolved_mvcc_row_provenance_bundle_occurrence_offset(
            &row,
            *bundle,
            *summary,
            expected,
            *occurrence,
        )
        .map(|offset| offset.to_string()),
        MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
            bundle,
            summary,
            left_expected,
            left_occurrence,
            right_expected,
            right_occurrence,
        } => resolved_mvcc_row_provenance_bundle_occurrence_distance(
            &row,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        )
        .map(|distance| distance.to_string()),
        MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
            bundle,
            summary,
            left_expected,
            left_occurrence,
            right_expected,
            right_occurrence,
        } => resolved_mvcc_row_provenance_bundle_mixed_occurrence_offset_pair(
            &row,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        )
        .map(|(left, right)| format!("{left},{right}")),
        _ => None,
    };
    let provenance_summary = match projection {
        MvccProjection::TargetKeyProvenanceSummary { summary } => {
            summarize_mvcc_row_provenance_path(&row, *summary)
        }
        MvccProjection::TargetKeyProvenanceBundleSummary { bundle, summary } => {
            summarize_mvcc_row_provenance_bundle(&row, *bundle, *summary)
        }
        _ => None,
    };

    let ResolvedMvccRow {
        branch_label,
        source_key,
        source_tuple,
        provenance_path: _,
        terminal_input_index: _,
        tuple,
    } = row;

    match projection {
        MvccProjection::KeyValue => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: Some(tuple.value),
        },
        MvccProjection::KeyOnly => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: None,
        },
        MvccProjection::ValueOnly => MvccReadRow {
            source_key,
            key: None,
            value: Some(tuple.value),
        },
        MvccProjection::BranchLabelTargetValue => MvccReadRow {
            key: branch_label,
            source_key,
            value: Some(tuple.value),
        },
        MvccProjection::SourceKeyTargetValue => MvccReadRow {
            key: source_key.clone(),
            source_key,
            value: Some(tuple.value),
        },
        MvccProjection::SourceValueOnly => MvccReadRow {
            source_key,
            key: None,
            value: source_tuple.map(|tuple| tuple.value),
        },
        MvccProjection::TargetKeySourceValue => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: source_tuple.map(|tuple| tuple.value),
        },
        MvccProjection::TargetKeyProvenanceValue { .. } => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: provenance_value,
        },
        MvccProjection::TargetKeyProvenanceSummary { .. } => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: provenance_summary,
        },
        MvccProjection::TargetKeyProvenanceBundleSummary { .. } => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: provenance_summary,
        },
        MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset { .. } => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: provenance_value,
        },
        MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance { .. } => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: provenance_value,
        },
        MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair { .. } => MvccReadRow {
            source_key,
            key: Some(tuple.key),
            value: provenance_value,
        },
    }
}

pub(crate) fn mvcc_read_row_size(row: &MvccReadRow) -> u64 {
    row.source_key
        .as_ref()
        .map_or(0, |value| value.len() as u64)
        + row.key.as_ref().map_or(0, |value| value.len() as u64)
        + row.value.as_ref().map_or(0, |value| value.len() as u64)
}

pub(crate) fn cuda_mvcc_row_batch_transfer_bytes(rows: &[ResolvedMvccRow]) -> u64 {
    CudaMvccRowBatch::from_key_values_with_metadata(rows.iter().map(|row| {
        (
            row.tuple.key.as_bytes(),
            row.tuple.value.as_bytes(),
            row.tuple.created_by,
            row.tuple.deleted_by.unwrap_or(u64::MAX),
            None,
        )
    }))
    .map(|batch| batch.transfer_bytes() as u64)
    .unwrap_or(u64::MAX)
}

pub(crate) fn mvcc_row_matches_filter(row: &ResolvedMvccRow, filter: &MvccReadFilter) -> bool {
    match filter {
        MvccReadFilter::KeyPrefix(prefix) => row.tuple.key.starts_with(prefix),
        MvccReadFilter::SourceKeyPrefix(prefix) => row
            .source_key
            .as_ref()
            .is_some_and(|source_key| source_key.starts_with(prefix)),
        MvccReadFilter::ProvenanceKeyPrefix { frame, prefix } => {
            resolved_mvcc_row_provenance_tuple(row, *frame)
                .is_some_and(|tuple| tuple.key.starts_with(prefix))
        }
        MvccReadFilter::ProvenanceBundleKeyEquals { bundle, expected } => {
            resolved_mvcc_row_provenance_bundle(row, *bundle)
                .is_some_and(|tuples| tuples.into_iter().any(|tuple| tuple.key == *expected))
        }
        MvccReadFilter::ProvenanceBundleKeyCountAtLeast {
            bundle,
            expected,
            min_count,
        } => resolved_mvcc_row_provenance_bundle(row, *bundle).is_some_and(|tuples| {
            mvcc_provenance_tuple_count_at_least(tuples.into_iter(), *min_count, |tuple| {
                tuple.key == *expected
            })
        }),
        MvccReadFilter::ProvenanceBundleKeyPrefix { bundle, prefix } => {
            resolved_mvcc_row_provenance_bundle(row, *bundle).is_some_and(|tuples| {
                tuples
                    .into_iter()
                    .any(|tuple| tuple.key.starts_with(prefix))
            })
        }
        MvccReadFilter::BranchLabelEquals(expected) => row.branch_label.as_ref() == Some(expected),
        MvccReadFilter::KeyRange {
            start_inclusive,
            end_exclusive,
        } => {
            row.tuple.key.as_str() >= start_inclusive.as_str()
                && row.tuple.key.as_str() < end_exclusive.as_str()
        }
        MvccReadFilter::ValueEquals(expected) => row.tuple.value == *expected,
        MvccReadFilter::SourceValueEquals(expected) => row
            .source_tuple
            .as_ref()
            .is_some_and(|source_tuple| source_tuple.value == *expected),
        MvccReadFilter::ProvenanceValueEquals { frame, expected } => {
            resolved_mvcc_row_provenance_tuple(row, *frame)
                .is_some_and(|tuple| tuple.value == *expected)
        }
        MvccReadFilter::ProvenanceBundleValueEquals { bundle, expected } => {
            resolved_mvcc_row_provenance_bundle(row, *bundle)
                .is_some_and(|tuples| tuples.into_iter().any(|tuple| tuple.value == *expected))
        }
        MvccReadFilter::ProvenanceBundleValueCountAtLeast {
            bundle,
            expected,
            min_count,
        } => resolved_mvcc_row_provenance_bundle(row, *bundle).is_some_and(|tuples| {
            mvcc_provenance_tuple_count_at_least(tuples.into_iter(), *min_count, |tuple| {
                tuple.value == *expected
            })
        }),
        MvccReadFilter::ProvenanceBundleKeyValueEquals { bundle, key, value } => {
            resolved_mvcc_row_provenance_bundle(row, *bundle).is_some_and(|tuples| {
                tuples
                    .into_iter()
                    .any(|tuple| tuple.key == *key && tuple.value == *value)
            })
        }
        MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast {
            bundle,
            key,
            value,
            min_count,
        } => resolved_mvcc_row_provenance_bundle(row, *bundle).is_some_and(|tuples| {
            mvcc_provenance_tuple_count_at_least(tuples.into_iter(), *min_count, |tuple| {
                tuple.key == *key && tuple.value == *value
            })
        }),
        MvccReadFilter::ProvenanceBundlePathEquals {
            bundle,
            summary,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary)
            .is_some_and(|segments| segments == *expected),
        MvccReadFilter::ProvenanceBundlePathContains {
            bundle,
            summary,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| mvcc_provenance_segments_contain_ordered_subpath(&segments, expected),
        ),
        MvccReadFilter::ProvenanceBundlePathCountAtLeast {
            bundle,
            summary,
            expected,
            min_count,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_ordered_subpath_count(&segments, expected) >= *min_count
            },
        ),
        MvccReadFilter::ProvenanceBundlePathPairAtDistance {
            bundle,
            summary,
            left,
            right,
            distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_have_pair_at_distance(
                    &segments,
                    left.as_str(),
                    right.as_str(),
                    *distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathSuffixEquals {
            bundle,
            summary,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary)
            .is_some_and(|segments| mvcc_provenance_segments_have_suffix(&segments, expected)),
        MvccReadFilter::ProvenanceBundlePathPrefixEquals {
            bundle,
            summary,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary)
            .is_some_and(|segments| mvcc_provenance_segments_have_prefix(&segments, expected)),
        MvccReadFilter::ProvenanceBundlePathSliceEquals {
            bundle,
            summary,
            start,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| mvcc_provenance_segments_have_slice_at(&segments, *start, expected),
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt {
            bundle,
            summary,
            start,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_subpath_start(&segments, expected) == Some(*start)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin {
            bundle,
            summary,
            start_min,
            start_max,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_start_within(
                    mvcc_provenance_segments_first_subpath_start(&segments, expected),
                    *start_min,
                    *start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt {
            bundle,
            summary,
            start,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_last_subpath_start(&segments, expected) == Some(*start)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin {
            bundle,
            summary,
            start_min,
            start_max,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_start_within(
                    mvcc_provenance_segments_last_subpath_start(&segments, expected),
                    *start_min,
                    *start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceAt {
            bundle,
            summary,
            occurrence_index,
            start,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_nth_subpath_start(&segments, expected, *occurrence_index)
                    == Some(*start)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
            bundle,
            summary,
            occurrence_index,
            start_min,
            start_max,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_nth_subpath_start_within(
                    &segments,
                    expected,
                    *occurrence_index,
                    *start_min,
                    *start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
            bundle,
            summary,
            left_occurrence_index,
            right_occurrence_index,
            distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_distance(
                    &segments,
                    expected,
                    *left_occurrence_index,
                    *right_occurrence_index,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
            bundle,
            summary,
            left_occurrence_index,
            right_occurrence_index,
            min_distance,
            max_distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_distance_within(
                    &segments,
                    expected,
                    *left_occurrence_index,
                    *right_occurrence_index,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistance {
            bundle,
            summary,
            distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_occurrence_distance(&segments, expected)
                    == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin {
            bundle,
            summary,
            min_distance,
            max_distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_occurrence_distance_within(
                    &segments,
                    expected,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance {
            bundle,
            summary,
            distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_last_occurrence_distance(&segments, expected)
                    == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin {
            bundle,
            summary,
            min_distance,
            max_distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_last_occurrence_distance_within(
                    &segments,
                    expected,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance {
            bundle,
            summary,
            occurrence_index,
            distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_occurrence_to_occurrence_distance(
                    &segments,
                    expected,
                    *occurrence_index,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
            bundle,
            summary,
            occurrence_index,
            min_distance,
            max_distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_occurrence_to_occurrence_distance_within(
                    &segments,
                    expected,
                    *occurrence_index,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceAt {
            bundle,
            summary,
            occurrence_index,
            first_start,
            occurrence_start,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_occurrence_to_occurrence_at(
                    &segments,
                    expected,
                    *occurrence_index,
                    *first_start,
                    *occurrence_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin {
            bundle,
            summary,
            occurrence_index,
            first_start_min,
            first_start_max,
            occurrence_start_min,
            occurrence_start_max,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_occurrence_to_occurrence_within(
                    &segments,
                    expected,
                    *occurrence_index,
                    *first_start_min..=*first_start_max,
                    *occurrence_start_min..=*occurrence_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistance {
            bundle,
            summary,
            occurrence_index,
            distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_distance(
                    &segments,
                    expected,
                    *occurrence_index,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistanceWithin {
            bundle,
            summary,
            occurrence_index,
            min_distance,
            max_distance,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_distance_within(
                    &segments,
                    expected,
                    *occurrence_index,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastAt {
            bundle,
            summary,
            occurrence_index,
            occurrence_start,
            last_start,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_at(
                    &segments,
                    expected,
                    *occurrence_index,
                    *occurrence_start,
                    *last_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastWithin {
            bundle,
            summary,
            occurrence_index,
            occurrence_start_min,
            occurrence_start_max,
            last_start_min,
            last_start_max,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_within(
                    &segments,
                    expected,
                    *occurrence_index,
                    *occurrence_start_min..=*occurrence_start_max,
                    *last_start_min..=*last_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrencePairAt {
            bundle,
            summary,
            left_occurrence_index,
            left_start,
            right_occurrence_index,
            right_start,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_pair_at(
                    &segments,
                    expected,
                    *left_occurrence_index,
                    *left_start,
                    *right_occurrence_index,
                    *right_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin {
            bundle,
            summary,
            left_occurrence_index,
            left_start_min,
            left_start_max,
            right_occurrence_index,
            right_start_min,
            right_start_max,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_pair_within(
                    &segments,
                    expected,
                    *left_occurrence_index,
                    *left_start_min..=*left_start_max,
                    *right_occurrence_index,
                    *right_start_min..=*right_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            right_occurrence_index,
            right_expected,
            distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_mixed_occurrence_distance(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    right_expected,
                    *right_occurrence_index,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            right_occurrence_index,
            right_expected,
            min_distance,
            max_distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_mixed_occurrence_distance_within(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    right_expected,
                    *right_occurrence_index,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistance {
            bundle,
            summary,
            left_expected,
            right_expected,
            distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_distance(
                    &segments,
                    left_expected,
                    right_expected,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin {
            bundle,
            summary,
            left_expected,
            right_expected,
            min_distance,
            max_distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_distance_within(
                    &segments,
                    left_expected,
                    right_expected,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance {
            bundle,
            summary,
            left_expected,
            right_expected,
            distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_last_mixed_occurrence_distance(
                    &segments,
                    left_expected,
                    right_expected,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin {
            bundle,
            summary,
            left_expected,
            right_expected,
            min_distance,
            max_distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_last_mixed_occurrence_distance_within(
                    &segments,
                    left_expected,
                    right_expected,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance {
            bundle,
            summary,
            left_expected,
            right_occurrence_index,
            right_expected,
            distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_distance(
                    &segments,
                    left_expected,
                    right_expected,
                    *right_occurrence_index,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
            bundle,
            summary,
            left_expected,
            right_occurrence_index,
            right_expected,
            min_distance,
            max_distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_distance_within(
                    &segments,
                    left_expected,
                    right_expected,
                    *right_occurrence_index,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistance {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            right_expected,
            distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_mixed_distance(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    right_expected,
                ) == Some(*distance)
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            right_expected,
            min_distance,
            max_distance,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_mixed_distance_within(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    right_expected,
                    *min_distance,
                    *max_distance,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt {
            bundle,
            summary,
            left_expected,
            left_start,
            right_occurrence_index,
            right_expected,
            right_start,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_at(
                    &segments,
                    left_expected,
                    *left_start,
                    right_expected,
                    *right_occurrence_index,
                    *right_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin {
            bundle,
            summary,
            left_expected,
            left_start_min,
            left_start_max,
            right_occurrence_index,
            right_expected,
            right_start_min,
            right_start_max,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_to_occurrence_within(
                    &segments,
                    left_expected,
                    *left_start_min..=*left_start_max,
                    right_expected,
                    *right_occurrence_index,
                    *right_start_min..=*right_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedAt {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            left_start,
            right_expected,
            right_start,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_mixed_at(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    *left_start,
                    right_expected,
                    *right_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedWithin {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            left_start_min,
            left_start_max,
            right_expected,
            right_start_min,
            right_start_max,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_occurrence_to_last_mixed_within(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    *left_start_min..=*left_start_max,
                    right_expected,
                    *right_start_min..=*right_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            left_start,
            right_occurrence_index,
            right_expected,
            right_start,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_mixed_occurrence_at(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    *left_start,
                    right_expected,
                    *right_occurrence_index,
                    *right_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin {
            bundle,
            summary,
            left_occurrence_index,
            left_expected,
            left_start_min,
            left_start_max,
            right_occurrence_index,
            right_expected,
            right_start_min,
            right_start_max,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_mixed_occurrence_within(
                    &segments,
                    left_expected,
                    *left_occurrence_index,
                    *left_start_min..=*left_start_max,
                    right_expected,
                    *right_occurrence_index,
                    *right_start_min..=*right_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceAt {
            bundle,
            summary,
            left_expected,
            left_start,
            right_expected,
            right_start,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_at(
                    &segments,
                    left_expected,
                    *left_start,
                    right_expected,
                    *right_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin {
            bundle,
            summary,
            left_expected,
            left_start_min,
            left_start_max,
            right_expected,
            right_start_min,
            right_start_max,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_first_mixed_occurrence_within(
                    &segments,
                    left_expected,
                    *left_start_min..=*left_start_max,
                    right_expected,
                    *right_start_min..=*right_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt {
            bundle,
            summary,
            left_expected,
            left_start,
            right_expected,
            right_start,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_last_mixed_occurrence_at(
                    &segments,
                    left_expected,
                    *left_start,
                    right_expected,
                    *right_start,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin {
            bundle,
            summary,
            left_expected,
            left_start_min,
            left_start_max,
            right_expected,
            right_start_min,
            right_start_max,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary).is_some_and(
            |segments| {
                mvcc_provenance_segments_last_mixed_occurrence_within(
                    &segments,
                    left_expected,
                    *left_start_min..=*left_start_max,
                    right_expected,
                    *right_start_min..=*right_start_max,
                )
            },
        ),
        MvccReadFilter::ProvenanceBundlePathSegmentEquals {
            bundle,
            summary,
            index,
            expected,
        } => resolved_mvcc_row_provenance_bundle_segments(row, *bundle, *summary)
            .is_some_and(|segments| segments.get(*index) == Some(expected)),
        MvccReadFilter::ProvenanceBundleLenEquals {
            bundle,
            expected_len,
        } => resolved_mvcc_row_provenance_bundle(row, *bundle)
            .is_some_and(|tuples| tuples.len() == *expected_len),
        MvccReadFilter::All(filters) => filters
            .iter()
            .all(|filter| mvcc_row_matches_filter(row, filter)),
        MvccReadFilter::Any(filters) => filters
            .iter()
            .any(|filter| mvcc_row_matches_filter(row, filter)),
    }
}

pub(crate) fn mvcc_row_cmp(
    left: &ResolvedMvccRow,
    right: &ResolvedMvccRow,
    order: &MvccReadOrder,
) -> std::cmp::Ordering {
    match order {
        MvccReadOrder::KeyAsc => left.tuple.key.cmp(&right.tuple.key),
        MvccReadOrder::KeyDesc => right.tuple.key.cmp(&left.tuple.key),
        MvccReadOrder::ValueAsc => left
            .tuple
            .value
            .cmp(&right.tuple.value)
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::ValueDesc => right
            .tuple
            .value
            .cmp(&left.tuple.value)
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::BranchLabelAsc => left
            .branch_label
            .as_deref()
            .unwrap_or("")
            .cmp(right.branch_label.as_deref().unwrap_or(""))
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::BranchLabelDesc => right
            .branch_label
            .as_deref()
            .unwrap_or("")
            .cmp(left.branch_label.as_deref().unwrap_or(""))
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::SourceKeyAsc => left
            .source_key
            .as_deref()
            .unwrap_or("")
            .cmp(right.source_key.as_deref().unwrap_or(""))
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::SourceKeyDesc => right
            .source_key
            .as_deref()
            .unwrap_or("")
            .cmp(left.source_key.as_deref().unwrap_or(""))
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::SourceValueAsc => left
            .source_tuple
            .as_ref()
            .map(|tuple| tuple.value.as_str())
            .unwrap_or("")
            .cmp(
                right
                    .source_tuple
                    .as_ref()
                    .map(|tuple| tuple.value.as_str())
                    .unwrap_or(""),
            )
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::SourceValueDesc => right
            .source_tuple
            .as_ref()
            .map(|tuple| tuple.value.as_str())
            .unwrap_or("")
            .cmp(
                left.source_tuple
                    .as_ref()
                    .map(|tuple| tuple.value.as_str())
                    .unwrap_or(""),
            )
            .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::ProvenanceKeyAsc { frame } => {
            resolved_mvcc_row_provenance_tuple(left, *frame)
                .map(|tuple| tuple.key.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(right, *frame)
                        .map(|tuple| tuple.key.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceKeyDesc { frame } => {
            resolved_mvcc_row_provenance_tuple(right, *frame)
                .map(|tuple| tuple.key.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(left, *frame)
                        .map(|tuple| tuple.key.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceValueAsc { frame } => {
            resolved_mvcc_row_provenance_tuple(left, *frame)
                .map(|tuple| tuple.value.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(right, *frame)
                        .map(|tuple| tuple.value.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceValueDesc { frame } => {
            resolved_mvcc_row_provenance_tuple(right, *frame)
                .map(|tuple| tuple.value.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(left, *frame)
                        .map(|tuple| tuple.value.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleKeyPathAsc { bundle } => {
            summarize_mvcc_row_provenance_bundle(left, *bundle, MvccProvenanceSummary::KeyPath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        right,
                        *bundle,
                        MvccProvenanceSummary::KeyPath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleKeyPathDesc { bundle } => {
            summarize_mvcc_row_provenance_bundle(right, *bundle, MvccProvenanceSummary::KeyPath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        left,
                        *bundle,
                        MvccProvenanceSummary::KeyPath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleValuePathAsc { bundle } => {
            summarize_mvcc_row_provenance_bundle(left, *bundle, MvccProvenanceSummary::ValuePath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        right,
                        *bundle,
                        MvccProvenanceSummary::ValuePath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleValuePathDesc { bundle } => {
            summarize_mvcc_row_provenance_bundle(right, *bundle, MvccProvenanceSummary::ValuePath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        left,
                        *bundle,
                        MvccProvenanceSummary::ValuePath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc {
            bundle,
            summary,
            expected,
            occurrence,
        } => resolved_mvcc_row_provenance_bundle_occurrence_offset(
            left,
            *bundle,
            *summary,
            expected,
            *occurrence,
        )
        .cmp(&resolved_mvcc_row_provenance_bundle_occurrence_offset(
            right,
            *bundle,
            *summary,
            expected,
            *occurrence,
        ))
        .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc {
            bundle,
            summary,
            expected,
            occurrence,
        } => resolved_mvcc_row_provenance_bundle_occurrence_offset(
            right,
            *bundle,
            *summary,
            expected,
            *occurrence,
        )
        .cmp(&resolved_mvcc_row_provenance_bundle_occurrence_offset(
            left,
            *bundle,
            *summary,
            expected,
            *occurrence,
        ))
        .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc {
            bundle,
            summary,
            left_expected,
            left_occurrence,
            right_expected,
            right_occurrence,
        } => resolved_mvcc_row_provenance_bundle_occurrence_distance(
            left,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        )
        .cmp(&resolved_mvcc_row_provenance_bundle_occurrence_distance(
            right,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        ))
        .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceDesc {
            bundle,
            summary,
            left_expected,
            left_occurrence,
            right_expected,
            right_occurrence,
        } => resolved_mvcc_row_provenance_bundle_occurrence_distance(
            right,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        )
        .cmp(&resolved_mvcc_row_provenance_bundle_occurrence_distance(
            left,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        ))
        .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
            bundle,
            summary,
            left_expected,
            left_occurrence,
            right_expected,
            right_occurrence,
        } => resolved_mvcc_row_provenance_bundle_mixed_occurrence_offset_pair(
            left,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        )
        .cmp(
            &resolved_mvcc_row_provenance_bundle_mixed_occurrence_offset_pair(
                right,
                *bundle,
                *summary,
                left_expected,
                *left_occurrence,
                right_expected,
                *right_occurrence,
            ),
        )
        .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
        MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairDesc {
            bundle,
            summary,
            left_expected,
            left_occurrence,
            right_expected,
            right_occurrence,
        } => resolved_mvcc_row_provenance_bundle_mixed_occurrence_offset_pair(
            right,
            *bundle,
            *summary,
            left_expected,
            *left_occurrence,
            right_expected,
            *right_occurrence,
        )
        .cmp(
            &resolved_mvcc_row_provenance_bundle_mixed_occurrence_offset_pair(
                left,
                *bundle,
                *summary,
                left_expected,
                *left_occurrence,
                right_expected,
                *right_occurrence,
            ),
        )
        .then_with(|| left.tuple.key.cmp(&right.tuple.key)),
    }
}

pub(crate) fn resolved_mvcc_row_key(row: &ResolvedMvccRow) -> ResolvedMvccRowIdentity {
    (
        row.branch_label.clone(),
        row.source_key.clone(),
        row.source_tuple.as_ref().map(resolved_tuple_identity),
        row.provenance_path
            .as_ref()
            .map(|path| path.iter().map(resolved_tuple_identity).collect()),
        row.terminal_input_index,
        resolved_tuple_identity(&row.tuple),
    )
}
