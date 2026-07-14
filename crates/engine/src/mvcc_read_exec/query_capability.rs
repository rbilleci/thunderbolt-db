use super::{MvccProjection, MvccReadFilter, MvccReadOrder, MvccReadQuery, MvccReadSource};

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
