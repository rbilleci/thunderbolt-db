//! Test-only closed-form MVCC semantic specifications.
//!
//! Production relational reads execute through resident GPU operators. The former generic
//! host-staged CUDA-MVCC compatibility backend was retired by RETIRE-003 because it downloaded
//! predicate masks and then compacted, ordered, limited, projected, and assembled rows on the host.

use super::*;

mod row_ops;
pub(crate) use row_ops::{
    mvcc_row_cmp, mvcc_row_matches_filter, project_mvcc_row, resolved_mvcc_row_key,
};
mod source_resolution;
pub(crate) use source_resolution::resolve_mvcc_source;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedMvccRow {
    pub(crate) branch_label: Option<String>,
    pub(crate) source_key: Option<String>,
    pub(crate) source_tuple: Option<TupleVersion>,
    pub(crate) provenance_path: Option<Vec<TupleVersion>>,
    pub(crate) terminal_input_index: Option<usize>,
    pub(crate) tuple: TupleVersion,
}
pub(crate) type ResolvedTupleIdentity = (u64, String, String, u64, Option<u64>);

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

/// Closed-form semantic specification for host-neutral unit fixtures.
///
/// This is not an execution backend: it has no target, fallback, routing, transfer, or metrics
/// contract and cannot be installed in product dispatch.
pub(crate) fn evaluate_mvcc_specification(
    query: &MvccReadQuery,
    mut rows: Vec<ResolvedMvccRow>,
) -> MvccSpecificationResult {
    if let Some(filter) = &query.filter {
        rows.retain(|row| mvcc_row_matches_filter(row, filter));
    }
    if let Some(order) = &query.order {
        rows.sort_by(|left, right| mvcc_row_cmp(left, right, order));
    }
    if let Some(limit) = query.limit {
        rows.truncate(limit);
    }

    MvccSpecificationResult {
        rows: rows
            .into_iter()
            .map(|row| project_mvcc_row(row, &query.projection))
            .collect(),
    }
}
