//! MVCC read-query model (P0 §9.6 decomposition, behavior-preserving): the
//! source/filter/order/projection enums, provenance value-chain types, the
//! read query/row/result structs, and the benchmark report. Pure data + its
//! constructors; the execution backends and read-path logic stay on lib.rs.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MvccReadSource {
    FullScan,
    KeyLookup {
        key: String,
    },
    KeyBatchLookup {
        keys: Vec<String>,
    },
    Concat {
        sources: Vec<MvccReadSource>,
    },
    ConcatDistinct {
        sources: Vec<MvccReadSource>,
    },
    IntersectDistinct {
        sources: Vec<MvccReadSource>,
    },
    IntersectAll {
        sources: Vec<MvccReadSource>,
    },
    ExceptDistinct {
        sources: Vec<MvccReadSource>,
    },
    ExceptAll {
        sources: Vec<MvccReadSource>,
    },
    SymmetricDifferenceDistinct {
        sources: Vec<MvccReadSource>,
    },
    SymmetricDifferenceAll {
        sources: Vec<MvccReadSource>,
    },
    FollowValueChain {
        keys: Vec<String>,
        plan: MvccValueChainPlan,
        provenance: MvccSourceProvenance,
    },
    FollowValueChainBranches {
        keys: Vec<String>,
        plans: Vec<MvccValueChainPlan>,
        fan_in: MvccValueChainBranchFanIn,
        provenance: MvccSourceProvenance,
    },
    FollowValueChainLabeledBranches {
        keys: Vec<String>,
        branches: Vec<MvccLabeledValueChainBranch>,
        fan_in: MvccValueChainBranchFanIn,
        provenance: MvccSourceProvenance,
    },
    FollowValueKeyRefs {
        keys: Vec<String>,
    },
    FollowValueKeyPrefixes {
        keys: Vec<String>,
    },
    FollowValueKeyRefPrefixes {
        keys: Vec<String>,
    },
    FollowValueKeyRefValueKeyRefs {
        keys: Vec<String>,
    },
    FollowValueKeyRefValueKeyPrefixes {
        keys: Vec<String>,
    },
    FollowValueKeyRefValueKeyRefPrefixes {
        keys: Vec<String>,
    },
    FollowValueKeyRefValueKeyRefValueKeyRefs {
        keys: Vec<String>,
    },
    FollowValueKeyRefValueKeyRefValueKeyPrefixes {
        keys: Vec<String>,
    },
    FollowValueKeyRefValueKeyRefValueKeyRefPrefixes {
        keys: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccValueChainTerminal {
    CurrentRow,
    CurrentValuePrefixes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvccValueChainPlan {
    pub value_key_hops: usize,
    pub terminal: MvccValueChainTerminal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccLabeledValueChainBranch {
    pub label: String,
    pub plan: MvccValueChainPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccValueChainBranchFanIn {
    AllBranches,
    FirstNonEmptyBranch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccSourceProvenance {
    Seed,
    TerminalInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MvccProvenanceFrame {
    Seed,
    TerminalInput,
    ValueHop(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccProvenanceSummary {
    KeyPath,
    ValuePath,
    KeyValuePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccProvenanceFrameBundle {
    SeedThroughTerminalInput,
    FullPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccProvenanceOccurrence {
    First,
    Last,
    Nth(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MvccReadFilter {
    KeyPrefix(String),
    SourceKeyPrefix(String),
    ProvenanceKeyPrefix {
        frame: MvccProvenanceFrame,
        prefix: String,
    },
    ProvenanceBundleKeyEquals {
        bundle: MvccProvenanceFrameBundle,
        expected: String,
    },
    ProvenanceBundleKeyCountAtLeast {
        bundle: MvccProvenanceFrameBundle,
        expected: String,
        min_count: usize,
    },
    ProvenanceBundleKeyPrefix {
        bundle: MvccProvenanceFrameBundle,
        prefix: String,
    },
    BranchLabelEquals(String),
    KeyRange {
        start_inclusive: String,
        end_exclusive: String,
    },
    ValueEquals(String),
    SourceValueEquals(String),
    ProvenanceValueEquals {
        frame: MvccProvenanceFrame,
        expected: String,
    },
    ProvenanceBundleValueEquals {
        bundle: MvccProvenanceFrameBundle,
        expected: String,
    },
    ProvenanceBundleValueCountAtLeast {
        bundle: MvccProvenanceFrameBundle,
        expected: String,
        min_count: usize,
    },
    ProvenanceBundleKeyValueEquals {
        bundle: MvccProvenanceFrameBundle,
        key: String,
        value: String,
    },
    ProvenanceBundleKeyValueCountAtLeast {
        bundle: MvccProvenanceFrameBundle,
        key: String,
        value: String,
        min_count: usize,
    },
    ProvenanceBundlePathEquals {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
    },
    ProvenanceBundlePathContains {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
    },
    ProvenanceBundlePathCountAtLeast {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
        min_count: usize,
    },
    ProvenanceBundlePathPairAtDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left: String,
        right: String,
        distance: usize,
    },
    ProvenanceBundlePathSuffixEquals {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
    },
    ProvenanceBundlePathPrefixEquals {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
    },
    ProvenanceBundlePathSliceEquals {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        start: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        start: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        start_min: usize,
        start_max: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathLastOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        start: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathLastOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        start_min: usize,
        start_max: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        start: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        start_min: usize,
        start_max: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        right_occurrence_index: usize,
        distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        right_occurrence_index: usize,
        min_distance: usize,
        max_distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        min_distance: usize,
        max_distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathLastOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathLastOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        min_distance: usize,
        max_distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        min_distance: usize,
        max_distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceToOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        first_start: usize,
        occurrence_start: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        first_start_min: usize,
        first_start_max: usize,
        occurrence_start_min: usize,
        occurrence_start_max: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceToLastDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceToLastDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        min_distance: usize,
        max_distance: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceToLastAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        occurrence_start: usize,
        last_start: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrenceToLastWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        occurrence_index: usize,
        occurrence_start_min: usize,
        occurrence_start_max: usize,
        last_start_min: usize,
        last_start_max: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrencePairAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_start: usize,
        right_occurrence_index: usize,
        right_start: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathOccurrencePairWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_start_min: usize,
        left_start_max: usize,
        right_occurrence_index: usize,
        right_start_min: usize,
        right_start_max: usize,
        expected: Vec<String>,
    },
    ProvenanceBundlePathMixedOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        distance: usize,
    },
    ProvenanceBundlePathMixedOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        min_distance: usize,
        max_distance: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        right_expected: Vec<String>,
        distance: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        right_expected: Vec<String>,
        min_distance: usize,
        max_distance: usize,
    },
    ProvenanceBundlePathLastMixedOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        right_expected: Vec<String>,
        distance: usize,
    },
    ProvenanceBundlePathLastMixedOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        right_expected: Vec<String>,
        min_distance: usize,
        max_distance: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        distance: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        min_distance: usize,
        max_distance: usize,
    },
    ProvenanceBundlePathOccurrenceToLastMixedDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        right_expected: Vec<String>,
        distance: usize,
    },
    ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        right_expected: Vec<String>,
        min_distance: usize,
        max_distance: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_start: usize,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        right_start: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_start_min: usize,
        left_start_max: usize,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        right_start_min: usize,
        right_start_max: usize,
    },
    ProvenanceBundlePathOccurrenceToLastMixedAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        left_start: usize,
        right_expected: Vec<String>,
        right_start: usize,
    },
    ProvenanceBundlePathOccurrenceToLastMixedWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        left_start_min: usize,
        left_start_max: usize,
        right_expected: Vec<String>,
        right_start_min: usize,
        right_start_max: usize,
    },
    ProvenanceBundlePathMixedOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        left_start: usize,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        right_start: usize,
    },
    ProvenanceBundlePathMixedOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_occurrence_index: usize,
        left_expected: Vec<String>,
        left_start_min: usize,
        left_start_max: usize,
        right_occurrence_index: usize,
        right_expected: Vec<String>,
        right_start_min: usize,
        right_start_max: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_start: usize,
        right_expected: Vec<String>,
        right_start: usize,
    },
    ProvenanceBundlePathFirstMixedOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_start_min: usize,
        left_start_max: usize,
        right_expected: Vec<String>,
        right_start_min: usize,
        right_start_max: usize,
    },
    ProvenanceBundlePathLastMixedOccurrenceAt {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_start: usize,
        right_expected: Vec<String>,
        right_start: usize,
    },
    ProvenanceBundlePathLastMixedOccurrenceWithin {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_start_min: usize,
        left_start_max: usize,
        right_expected: Vec<String>,
        right_start_min: usize,
        right_start_max: usize,
    },
    ProvenanceBundlePathSegmentEquals {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        index: usize,
        expected: String,
    },
    ProvenanceBundleLenEquals {
        bundle: MvccProvenanceFrameBundle,
        expected_len: usize,
    },
    All(Vec<MvccReadFilter>),
    Any(Vec<MvccReadFilter>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MvccReadOrder {
    KeyAsc,
    KeyDesc,
    ValueAsc,
    ValueDesc,
    BranchLabelAsc,
    BranchLabelDesc,
    SourceKeyAsc,
    SourceKeyDesc,
    SourceValueAsc,
    SourceValueDesc,
    ProvenanceKeyAsc {
        frame: MvccProvenanceFrame,
    },
    ProvenanceKeyDesc {
        frame: MvccProvenanceFrame,
    },
    ProvenanceValueAsc {
        frame: MvccProvenanceFrame,
    },
    ProvenanceValueDesc {
        frame: MvccProvenanceFrame,
    },
    ProvenanceBundleKeyPathAsc {
        bundle: MvccProvenanceFrameBundle,
    },
    ProvenanceBundleKeyPathDesc {
        bundle: MvccProvenanceFrameBundle,
    },
    ProvenanceBundleValuePathAsc {
        bundle: MvccProvenanceFrameBundle,
    },
    ProvenanceBundleValuePathDesc {
        bundle: MvccProvenanceFrameBundle,
    },
    ProvenanceBundlePathOccurrenceOffsetAsc {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
        occurrence: MvccProvenanceOccurrence,
    },
    ProvenanceBundlePathOccurrenceOffsetDesc {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
        occurrence: MvccProvenanceOccurrence,
    },
    ProvenanceBundlePathOccurrenceDistanceAsc {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_occurrence: MvccProvenanceOccurrence,
        right_expected: Vec<String>,
        right_occurrence: MvccProvenanceOccurrence,
    },
    ProvenanceBundlePathOccurrenceDistanceDesc {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_occurrence: MvccProvenanceOccurrence,
        right_expected: Vec<String>,
        right_occurrence: MvccProvenanceOccurrence,
    },
    ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_occurrence: MvccProvenanceOccurrence,
        right_expected: Vec<String>,
        right_occurrence: MvccProvenanceOccurrence,
    },
    ProvenanceBundlePathMixedOccurrenceOffsetPairDesc {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_occurrence: MvccProvenanceOccurrence,
        right_expected: Vec<String>,
        right_occurrence: MvccProvenanceOccurrence,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MvccProjection {
    KeyValue,
    KeyOnly,
    ValueOnly,
    BranchLabelTargetValue,
    SourceKeyTargetValue,
    SourceValueOnly,
    TargetKeySourceValue,
    TargetKeyProvenanceValue {
        frame: MvccProvenanceFrame,
    },
    TargetKeyProvenanceSummary {
        summary: MvccProvenanceSummary,
    },
    TargetKeyProvenanceBundleSummary {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
    },
    TargetKeyProvenanceBundleOccurrenceOffset {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        expected: Vec<String>,
        occurrence: MvccProvenanceOccurrence,
    },
    TargetKeyProvenanceBundleOccurrenceDistance {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_occurrence: MvccProvenanceOccurrence,
        right_expected: Vec<String>,
        right_occurrence: MvccProvenanceOccurrence,
    },
    TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
        bundle: MvccProvenanceFrameBundle,
        summary: MvccProvenanceSummary,
        left_expected: Vec<String>,
        left_occurrence: MvccProvenanceOccurrence,
        right_expected: Vec<String>,
        right_occurrence: MvccProvenanceOccurrence,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccReadQuery {
    pub source: MvccReadSource,
    pub visibility: StorageVisibility,
    pub filter: Option<MvccReadFilter>,
    pub order: Option<MvccReadOrder>,
    pub projection: MvccProjection,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccReadRow {
    pub source_key: Option<String>,
    pub key: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccReadResult {
    pub planned_target: DeviceTarget,
    pub executed_target: DeviceTarget,
    pub fallback_reason: Option<FallbackReason>,
    pub rows: Vec<MvccReadRow>,
}

/// Test-only semantic outcome produced by the closed-form MVCC specification oracle.
///
/// Deliberately carries rows only: specification fixtures must not masquerade as an executed
/// device/host route or manufacture production fallback telemetry.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MvccSpecificationResult {
    pub(crate) rows: Vec<MvccReadRow>,
}

/// Test-only evidence captured from an actual MVCC execution backend.
///
/// Kept separate from semantic specification rows so a fixture cannot present host-finalized
/// rows as though they were the backend's production result.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MvccExecutionEvidence {
    pub(crate) planned_target: DeviceTarget,
    pub(crate) executed_target: DeviceTarget,
    pub(crate) fallback_reason: Option<FallbackReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvccBenchmarkReport {
    pub workload_count: usize,
    pub gpu_executed_count: usize,
    pub cpu_fallback_count: usize,
    pub gpu_executed_permyriad: u16,
    pub cpu_fallback_permyriad: u16,
    pub h2d_bytes_total: u64,
    pub d2h_bytes_total: u64,
    pub kernel_exec_samples: u64,
    pub kernel_exec_total_ms: u64,
    pub batch_wait_samples: u64,
    pub batch_wait_total_ms: u64,
}

impl MvccBenchmarkReport {
    pub fn from_results(
        results: &[MvccReadResult],
        metrics: &RuntimeMetricsSnapshot,
    ) -> MvccBenchmarkReport {
        let workload_count = results.len();
        let gpu_executed_count = results
            .iter()
            .filter(|result| matches!(result.executed_target, DeviceTarget::Gpu(_)))
            .count();
        let cpu_fallback_count = results
            .iter()
            .filter(|result| result.fallback_reason.is_some())
            .count();

        MvccBenchmarkReport {
            workload_count,
            gpu_executed_count,
            cpu_fallback_count,
            gpu_executed_permyriad: permyriad(gpu_executed_count, workload_count),
            cpu_fallback_permyriad: permyriad(cpu_fallback_count, workload_count),
            h2d_bytes_total: metrics.h2d_bytes_total,
            d2h_bytes_total: metrics.d2h_bytes_total,
            kernel_exec_samples: metrics.kernel_exec_samples,
            kernel_exec_total_ms: metrics.kernel_exec_total_ms,
            batch_wait_samples: metrics.batch_wait_samples,
            batch_wait_total_ms: metrics.batch_wait_total_ms,
        }
    }
}

pub(crate) fn permyriad(numerator: usize, denominator: usize) -> u16 {
    if denominator == 0 {
        return 0;
    }
    ((numerator as u128 * 10_000) / denominator as u128)
        .try_into()
        .unwrap_or(u16::MAX)
}
