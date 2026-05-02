use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

use gpu_db_batching::{BatchItem, DualTriggerBatcher, FlushReason};
use gpu_db_execution::{
    DeviceRouter, DeviceTarget, FilterOperator, LimitOperator, MockGpuRuntime, Operator,
    ProjectOperator, RouteDecision, ScanOperator, SortOperator,
};
use gpu_db_metrics::{BatchFlushReason, FallbackReason, RuntimeMetrics};
use gpu_db_observability::{
    ActiveFallbackReason, EngineStatusSnapshot, EngineTelemetrySnapshot, FallbackStatus,
    ReadinessStatus, ReplicationLagSnapshot, SnapshotStatus, TelemetrySink,
};
use gpu_db_planner::{ExecutionPlan, Planner, PlannerConfig};
use gpu_db_protocol::{parse_command, Command, ParseError};
use gpu_db_replication::{LocalReplicator, LogReplicator, ReplicatedStateMachine};
use gpu_db_storage::{
    InMemoryTupleStore, NewTuple, StorageError, TupleStore, TupleVersion,
    Visibility as StorageVisibility,
};
use gpu_db_txn::{TxnError, TxnManager};
use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term, TxnId};
use gpu_db_wal::{WalBuffer, WalRecord};

#[derive(Debug, Default)]
pub struct KvStateMachine {
    pub applied: Vec<Vec<u8>>,
    pub kv: BTreeMap<String, String>,
}

impl ReplicatedStateMachine for KvStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        self.applied.push(entry.payload.clone());
        if let Ok(s) = std::str::from_utf8(&entry.payload) {
            if let Ok(cmd) = parse_command(s) {
                match cmd {
                    Command::SetKv { key, value } => {
                        self.kv.insert(key, value);
                    }
                    Command::DeleteKv { key } => {
                        self.kv.remove(&key);
                    }
                    Command::Begin
                    | Command::Commit { .. }
                    | Command::Rollback { .. }
                    | Command::Flush
                    | Command::ResetAll
                    | Command::GetKv { .. } => {}
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecuteError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Txn(#[from] TxnError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("command is not readable via execute_read_text: {0}")]
    NonReadCommand(&'static str),
}

#[derive(Debug, Clone)]
struct PendingMutation {
    txn_id: u64,
    payload: Vec<u8>,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    ProvenanceKeyAsc { frame: MvccProvenanceFrame },
    ProvenanceKeyDesc { frame: MvccProvenanceFrame },
    ProvenanceValueAsc { frame: MvccProvenanceFrame },
    ProvenanceValueDesc { frame: MvccProvenanceFrame },
    ProvenanceBundleKeyPathAsc { bundle: MvccProvenanceFrameBundle },
    ProvenanceBundleKeyPathDesc { bundle: MvccProvenanceFrameBundle },
    ProvenanceBundleValuePathAsc { bundle: MvccProvenanceFrameBundle },
    ProvenanceBundleValuePathDesc { bundle: MvccProvenanceFrameBundle },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedMvccRow {
    branch_label: Option<String>,
    source_key: Option<String>,
    source_tuple: Option<TupleVersion>,
    provenance_path: Option<Vec<TupleVersion>>,
    terminal_input_index: Option<usize>,
    tuple: TupleVersion,
}

type ResolvedTupleIdentity = (u64, String, String, u64, Option<u64>);
type ResolvedMvccRowIdentity = (
    Option<String>,
    Option<String>,
    Option<ResolvedTupleIdentity>,
    Option<Vec<ResolvedTupleIdentity>>,
    Option<usize>,
    ResolvedTupleIdentity,
);

fn resolved_tuple_identity(tuple: &TupleVersion) -> ResolvedTupleIdentity {
    (
        tuple.tuple_id,
        tuple.key.clone(),
        tuple.value.clone(),
        tuple.created_by,
        tuple.deleted_by,
    )
}

fn resolved_mvcc_row_provenance_tuple(
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

fn resolved_mvcc_row_provenance_bundle(
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

fn summarize_mvcc_row_provenance_path(
    row: &ResolvedMvccRow,
    summary: MvccProvenanceSummary,
) -> Option<String> {
    summarize_mvcc_provenance_tuples(row.provenance_path.as_ref()?.iter(), summary)
}

fn summarize_mvcc_row_provenance_bundle(
    row: &ResolvedMvccRow,
    bundle: MvccProvenanceFrameBundle,
    summary: MvccProvenanceSummary,
) -> Option<String> {
    let segments = resolved_mvcc_row_provenance_bundle_segments(row, bundle, summary)?;
    Some(segments.join(" -> "))
}

fn resolved_mvcc_row_provenance_bundle_segments(
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

fn summarize_mvcc_provenance_tuples<'a>(
    tuples: impl Iterator<Item = &'a TupleVersion>,
    summary: MvccProvenanceSummary,
) -> Option<String> {
    let segments = collect_mvcc_provenance_segments(tuples, summary);
    Some(segments.join(" -> "))
}

fn mvcc_provenance_segments_contain_ordered_subpath(
    segments: &[String],
    expected: &[String],
) -> bool {
    mvcc_provenance_segments_ordered_subpath_count(segments, expected) > 0
}

fn mvcc_provenance_segments_ordered_subpath_count(
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

fn mvcc_provenance_segments_have_pair_at_distance(
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

fn mvcc_provenance_segments_have_suffix(segments: &[String], expected: &[String]) -> bool {
    !expected.is_empty()
        && expected.len() <= segments.len()
        && segments[segments.len() - expected.len()..] == *expected
}

fn mvcc_provenance_tuple_count_at_least<'a>(
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

fn collect_mvcc_provenance_segments<'a>(
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

fn project_mvcc_row(row: ResolvedMvccRow, projection: MvccProjection) -> MvccReadRow {
    let provenance_value = match projection {
        MvccProjection::TargetKeyProvenanceValue { frame } => {
            resolved_mvcc_row_provenance_tuple(&row, frame).map(|tuple| tuple.value.clone())
        }
        _ => None,
    };
    let provenance_summary = match projection {
        MvccProjection::TargetKeyProvenanceSummary { summary } => {
            summarize_mvcc_row_provenance_path(&row, summary)
        }
        MvccProjection::TargetKeyProvenanceBundleSummary { bundle, summary } => {
            summarize_mvcc_row_provenance_bundle(&row, bundle, summary)
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
    }
}

fn mvcc_read_row_size(row: &MvccReadRow) -> u64 {
    row.source_key
        .as_ref()
        .map_or(0, |value| value.len() as u64)
        + row.key.as_ref().map_or(0, |value| value.len() as u64)
        + row.value.as_ref().map_or(0, |value| value.len() as u64)
}

fn mvcc_row_matches_filter(row: &ResolvedMvccRow, filter: &MvccReadFilter) -> bool {
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

fn mvcc_row_cmp(
    left: &ResolvedMvccRow,
    right: &ResolvedMvccRow,
    order: MvccReadOrder,
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
            resolved_mvcc_row_provenance_tuple(left, frame)
                .map(|tuple| tuple.key.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(right, frame)
                        .map(|tuple| tuple.key.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceKeyDesc { frame } => {
            resolved_mvcc_row_provenance_tuple(right, frame)
                .map(|tuple| tuple.key.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(left, frame)
                        .map(|tuple| tuple.key.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceValueAsc { frame } => {
            resolved_mvcc_row_provenance_tuple(left, frame)
                .map(|tuple| tuple.value.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(right, frame)
                        .map(|tuple| tuple.value.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceValueDesc { frame } => {
            resolved_mvcc_row_provenance_tuple(right, frame)
                .map(|tuple| tuple.value.as_str())
                .unwrap_or("")
                .cmp(
                    resolved_mvcc_row_provenance_tuple(left, frame)
                        .map(|tuple| tuple.value.as_str())
                        .unwrap_or(""),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleKeyPathAsc { bundle } => {
            summarize_mvcc_row_provenance_bundle(left, bundle, MvccProvenanceSummary::KeyPath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        right,
                        bundle,
                        MvccProvenanceSummary::KeyPath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleKeyPathDesc { bundle } => {
            summarize_mvcc_row_provenance_bundle(right, bundle, MvccProvenanceSummary::KeyPath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        left,
                        bundle,
                        MvccProvenanceSummary::KeyPath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleValuePathAsc { bundle } => {
            summarize_mvcc_row_provenance_bundle(left, bundle, MvccProvenanceSummary::ValuePath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        right,
                        bundle,
                        MvccProvenanceSummary::ValuePath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
        MvccReadOrder::ProvenanceBundleValuePathDesc { bundle } => {
            summarize_mvcc_row_provenance_bundle(right, bundle, MvccProvenanceSummary::ValuePath)
                .unwrap_or_default()
                .cmp(
                    &summarize_mvcc_row_provenance_bundle(
                        left,
                        bundle,
                        MvccProvenanceSummary::ValuePath,
                    )
                    .unwrap_or_default(),
                )
                .then_with(|| left.tuple.key.cmp(&right.tuple.key))
        }
    }
}

fn resolved_mvcc_row_key(row: &ResolvedMvccRow) -> ResolvedMvccRowIdentity {
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

fn collect_operator_rows<Row, Op>(mut operator: Op) -> Vec<Row>
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

fn resolve_follow_value_chain_from_seed(
    store: &InMemoryTupleStore,
    seed: &TupleVersion,
    visibility: StorageVisibility,
    plan: MvccValueChainPlan,
    provenance: MvccSourceProvenance,
    branch_label: Option<&str>,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut current = seed.clone();
    let mut provenance_path = vec![seed.clone()];
    let mut previous = None;
    for _ in 0..plan.value_key_hops {
        previous = Some(current.clone());
        let Some(next) = store.tuple_fetch_by_key(&current.value, visibility)? else {
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
    let source_key = provenance_tuple.key.clone();

    let mut rows = Vec::new();
    match plan.terminal {
        MvccValueChainTerminal::CurrentRow => rows.push(ResolvedMvccRow {
            branch_label: branch_label.map(ToOwned::to_owned),
            source_key: Some(source_key.clone()),
            source_tuple: Some(provenance_tuple.clone()),
            provenance_path: Some(provenance_path.clone()),
            terminal_input_index: Some(terminal_input_index),
            tuple: current,
        }),
        MvccValueChainTerminal::CurrentValuePrefixes => {
            let mut cursor = store.seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&current.value) {
                    rows.push(ResolvedMvccRow {
                        branch_label: branch_label.map(ToOwned::to_owned),
                        source_key: Some(source_key.clone()),
                        source_tuple: Some(provenance_tuple.clone()),
                        provenance_path: Some(provenance_path.clone()),
                        terminal_input_index: Some(terminal_input_index),
                        tuple,
                    });
                }
            }
        }
    }
    Ok(rows)
}

fn resolve_follow_value_chain(
    store: &InMemoryTupleStore,
    keys: &[String],
    visibility: StorageVisibility,
    plan: MvccValueChainPlan,
    provenance: MvccSourceProvenance,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = store.tuple_fetch_by_key(key, visibility)? else {
            continue;
        };
        rows.extend(resolve_follow_value_chain_from_seed(
            store, &seed, visibility, plan, provenance, None,
        )?);
    }
    Ok(rows)
}

fn resolve_follow_value_chain_branches(
    store: &InMemoryTupleStore,
    keys: &[String],
    visibility: StorageVisibility,
    plans: &[MvccValueChainPlan],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = store.tuple_fetch_by_key(key, visibility)? else {
            continue;
        };
        match fan_in {
            MvccValueChainBranchFanIn::AllBranches => {
                for plan in plans {
                    rows.extend(resolve_follow_value_chain_from_seed(
                        store, &seed, visibility, *plan, provenance, None,
                    )?);
                }
            }
            MvccValueChainBranchFanIn::FirstNonEmptyBranch => {
                for plan in plans {
                    let branch_rows = resolve_follow_value_chain_from_seed(
                        store, &seed, visibility, *plan, provenance, None,
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

fn resolve_follow_value_chain_labeled_branches(
    store: &InMemoryTupleStore,
    keys: &[String],
    visibility: StorageVisibility,
    branches: &[MvccLabeledValueChainBranch],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = store.tuple_fetch_by_key(key, visibility)? else {
            continue;
        };
        match fan_in {
            MvccValueChainBranchFanIn::AllBranches => {
                for branch in branches {
                    rows.extend(resolve_follow_value_chain_from_seed(
                        store,
                        &seed,
                        visibility,
                        branch.plan,
                        provenance,
                        Some(branch.label.as_str()),
                    )?);
                }
            }
            MvccValueChainBranchFanIn::FirstNonEmptyBranch => {
                for branch in branches {
                    let branch_rows = resolve_follow_value_chain_from_seed(
                        store,
                        &seed,
                        visibility,
                        branch.plan,
                        provenance,
                        Some(branch.label.as_str()),
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

fn resolve_mvcc_source(
    store: &InMemoryTupleStore,
    source: &MvccReadSource,
    visibility: StorageVisibility,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    match source {
        MvccReadSource::FullScan => {
            let mut cursor = store.seq_scan_open(visibility)?;
            let mut rows = Vec::new();
            while let Some(tuple) = cursor.next() {
                rows.push(ResolvedMvccRow {
                    branch_label: None,
                    source_key: None,
                    source_tuple: None,
                    provenance_path: None,
                    terminal_input_index: None,
                    tuple,
                });
            }
            Ok(rows)
        }
        MvccReadSource::KeyLookup { key } => Ok(store
            .tuple_fetch_by_key(key, visibility)?
            .into_iter()
            .map(|tuple| ResolvedMvccRow {
                branch_label: None,
                source_key: None,
                source_tuple: None,
                provenance_path: None,
                terminal_input_index: None,
                tuple,
            })
            .collect()),
        MvccReadSource::KeyBatchLookup { keys } => {
            let mut rows = Vec::new();
            for key in keys {
                if let Some(tuple) = store.tuple_fetch_by_key(key, visibility)? {
                    rows.push(ResolvedMvccRow {
                        branch_label: None,
                        source_key: None,
                        source_tuple: None,
                        provenance_path: None,
                        terminal_input_index: None,
                        tuple,
                    });
                }
            }
            Ok(rows)
        }
        MvccReadSource::Concat { sources } => {
            let mut rows = Vec::new();
            for source in sources {
                rows.extend(resolve_mvcc_source(store, source, visibility)?);
            }
            Ok(rows)
        }
        MvccReadSource::ConcatDistinct { sources } => {
            let mut rows = Vec::new();
            let mut seen = BTreeSet::new();
            for source in sources {
                for row in resolve_mvcc_source(store, source, visibility)? {
                    if seen.insert(resolved_mvcc_row_key(&row)) {
                        rows.push(row);
                    }
                }
            }
            Ok(rows)
        }
        MvccReadSource::IntersectDistinct { sources } => {
            let mut sources_iter = sources.iter();
            let Some(first_source) = sources_iter.next() else {
                return Ok(Vec::new());
            };

            let first_rows = resolve_mvcc_source(store, first_source, visibility)?;
            let mut intersection = Vec::new();
            let mut emitted = BTreeSet::new();
            let remaining_sets = sources_iter
                .map(|source| {
                    resolve_mvcc_source(store, source, visibility).map(|rows| {
                        rows.into_iter()
                            .map(|row| resolved_mvcc_row_key(&row))
                            .collect::<BTreeSet<_>>()
                    })
                })
                .collect::<Result<Vec<_>, StorageError>>()?;

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
            Ok(intersection)
        }
        MvccReadSource::IntersectAll { sources } => {
            let mut sources_iter = sources.iter();
            let Some(first_source) = sources_iter.next() else {
                return Ok(Vec::new());
            };

            let first_rows = resolve_mvcc_source(store, first_source, visibility)?;
            let remaining_counts = sources_iter
                .map(|source| {
                    resolve_mvcc_source(store, source, visibility).map(|rows| {
                        let mut counts = BTreeMap::new();
                        for key in rows.into_iter().map(|row| resolved_mvcc_row_key(&row)) {
                            *counts.entry(key).or_insert(0usize) += 1;
                        }
                        counts
                    })
                })
                .collect::<Result<Vec<_>, StorageError>>()?;

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
            Ok(intersection)
        }
        MvccReadSource::ExceptDistinct { sources } => {
            let mut sources_iter = sources.iter();
            let Some(first_source) = sources_iter.next() else {
                return Ok(Vec::new());
            };

            let first_rows = resolve_mvcc_source(store, first_source, visibility)?;
            let exclusion_set = sources_iter
                .map(|source| {
                    resolve_mvcc_source(store, source, visibility).map(|rows| {
                        rows.into_iter()
                            .map(|row| resolved_mvcc_row_key(&row))
                            .collect::<BTreeSet<_>>()
                    })
                })
                .collect::<Result<Vec<_>, StorageError>>()?
                .into_iter()
                .flatten()
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
            Ok(difference)
        }
        MvccReadSource::ExceptAll { sources } => {
            let mut sources_iter = sources.iter();
            let Some(first_source) = sources_iter.next() else {
                return Ok(Vec::new());
            };

            let first_rows = resolve_mvcc_source(store, first_source, visibility)?;
            let mut exclusion_counts = BTreeMap::new();
            for source in sources_iter {
                for key in resolve_mvcc_source(store, source, visibility)?
                    .into_iter()
                    .map(|row| resolved_mvcc_row_key(&row))
                {
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
            Ok(difference)
        }
        MvccReadSource::SymmetricDifferenceDistinct { sources } => {
            let resolved_sources = sources
                .iter()
                .map(|source| resolve_mvcc_source(store, source, visibility))
                .collect::<Result<Vec<_>, StorageError>>()?;

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
            Ok(output)
        }
        MvccReadSource::SymmetricDifferenceAll { sources } => {
            let resolved_sources = sources
                .iter()
                .map(|source| resolve_mvcc_source(store, source, visibility))
                .collect::<Result<Vec<_>, StorageError>>()?;

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
            Ok(output)
        }
        MvccReadSource::FollowValueChain {
            keys,
            plan,
            provenance,
        } => resolve_follow_value_chain(store, keys, visibility, *plan, *provenance),
        MvccReadSource::FollowValueChainBranches {
            keys,
            plans,
            fan_in,
            provenance,
        } => resolve_follow_value_chain_branches(
            store,
            keys,
            visibility,
            plans,
            *fan_in,
            *provenance,
        ),
        MvccReadSource::FollowValueChainLabeledBranches {
            keys,
            branches,
            fan_in,
            provenance,
        } => resolve_follow_value_chain_labeled_branches(
            store,
            keys,
            visibility,
            branches,
            *fan_in,
            *provenance,
        ),
        MvccReadSource::FollowValueKeyRefs { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyPrefixes { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 0,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefPrefixes { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefValueKeyRefs { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefValueKeyPrefixes { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                MvccSourceProvenance::Seed,
            )
        }
        MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                MvccSourceProvenance::Seed,
            )
        }
        MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 4,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                MvccSourceProvenance::Seed,
            )
        }
        MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                MvccSourceProvenance::Seed,
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacklogBlocker {
    Wal,
    PendingBatch,
    ActiveTxn,
    CommitApplyGap,
    ApplyVisibleGap,
}

impl BacklogBlocker {
    pub const ALL: [Self; 5] = [
        Self::Wal,
        Self::PendingBatch,
        Self::ActiveTxn,
        Self::CommitApplyGap,
        Self::ApplyVisibleGap,
    ];

    pub const fn bit(self) -> u8 {
        match self {
            Self::Wal => ReplicationWatermarks::BACKLOG_BLOCKER_WAL,
            Self::PendingBatch => ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH,
            Self::ActiveTxn => ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN,
            Self::CommitApplyGap => ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP,
            Self::ApplyVisibleGap => ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP,
        }
    }

    pub const fn from_bit(bit: u8) -> Option<Self> {
        match bit {
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL => Some(Self::Wal),
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH => Some(Self::PendingBatch),
            ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN => Some(Self::ActiveTxn),
            ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP => Some(Self::CommitApplyGap),
            ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP => Some(Self::ApplyVisibleGap),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wal => "wal",
            Self::PendingBatch => "pending_batch",
            Self::ActiveTxn => "active_txn",
            Self::CommitApplyGap => "commit_apply_gap",
            Self::ApplyVisibleGap => "apply_visible_gap",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        let mut normalized = [0_u8; 32];
        let mut len = 0usize;

        for b in label.trim().bytes() {
            let folded = match b {
                b'A'..=b'Z' => b + 32,
                b'-' | b' ' | b'.' => b'_',
                _ => b,
            };

            if folded == b'_' && len > 0 && normalized[len - 1] == b'_' {
                continue;
            }

            if len == normalized.len() {
                return None;
            }

            normalized[len] = folded;
            len += 1;
        }

        let mut start = 0usize;
        while start < len && normalized[start] == b'_' {
            start += 1;
        }

        let mut end = len;
        while end > start && normalized[end - 1] == b'_' {
            end -= 1;
        }

        match &normalized[start..end] {
            b"wal" => Some(Self::Wal),
            b"pending_batch" => Some(Self::PendingBatch),
            b"active_txn" => Some(Self::ActiveTxn),
            b"commit_apply_gap" => Some(Self::CommitApplyGap),
            b"apply_visible_gap" => Some(Self::ApplyVisibleGap),
            _ => None,
        }
    }
}

impl fmt::Display for BacklogBlocker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown backlog blocker label: {label}")]
pub struct ParseBacklogBlockerError {
    label: String,
}

impl ParseBacklogBlockerError {
    pub fn label(&self) -> &str {
        &self.label
    }

    fn unknown(label: &str) -> Self {
        Self {
            label: label.trim().to_owned(),
        }
    }
}

impl FromStr for BacklogBlocker {
    type Err = ParseBacklogBlockerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_label(s).ok_or_else(|| ParseBacklogBlockerError::unknown(s))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicationWatermarks {
    pub role: Role,
    pub term: Term,
    pub commit_index: Index,
    pub applied_index: Index,
    pub visible_index: Index,
    pub commit_apply_gap: Index,
    pub apply_visible_gap: Index,
    pub snapshot_id: u64,
    pub wal_flushed_count: usize,
    pub wal_last_durable_txn_id: Option<TxnId>,
    pub wal_buffered_count: usize,
    pub wal_unflushed_count: usize,
    pub pending_batch_len: usize,
    pub pending_batch_cap: usize,
    pub pending_batch_remaining_capacity: usize,
    pub pending_batch_utilization_permyriad: u16,
    pub pending_batch_remaining_capacity_permyriad: u16,
    pub pending_batch_oldest_age_ms: Option<u64>,
    pub pending_batch_time_until_deadline_ms: Option<u64>,
    pub active_txn_count: usize,
    pub oldest_active_txn_id: Option<TxnId>,
    pub newest_active_txn_id: Option<TxnId>,
    pub has_wal_backlog: bool,
    pub has_pending_batch_backlog: bool,
    pub has_active_txn_backlog: bool,
    pub has_commit_apply_gap: bool,
    pub has_apply_visible_gap: bool,
    pub has_backlog_blockers: bool,
    pub backlog_blocker_count: u8,
    pub backlog_blocker_mask: u8,
    pub mutation_admission_saturated: bool,
    pub quiescent_for_failover: bool,
    pub follower_promotion_ready: bool,
}

impl ReplicationWatermarks {
    pub const BACKLOG_BLOCKER_WAL: u8 = 1 << 0;
    pub const BACKLOG_BLOCKER_PENDING_BATCH: u8 = 1 << 1;
    pub const BACKLOG_BLOCKER_ACTIVE_TXN: u8 = 1 << 2;
    pub const BACKLOG_BLOCKER_COMMIT_APPLY_GAP: u8 = 1 << 3;
    pub const BACKLOG_BLOCKER_APPLY_VISIBLE_GAP: u8 = 1 << 4;
    pub const KNOWN_BACKLOG_BLOCKER_MASK: u8 = Self::BACKLOG_BLOCKER_WAL
        | Self::BACKLOG_BLOCKER_PENDING_BATCH
        | Self::BACKLOG_BLOCKER_ACTIVE_TXN
        | Self::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        | Self::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;

    pub const fn known_backlog_blocker_mask() -> u8 {
        Self::KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub const fn unknown_backlog_blocker_mask(mask: u8) -> u8 {
        mask & !Self::KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub const fn sanitize_backlog_blocker_mask(mask: u8) -> u8 {
        mask & Self::KNOWN_BACKLOG_BLOCKER_MASK
    }

    pub fn backlog_blocker_count_from_mask(mask: u8) -> u8 {
        Self::sanitize_backlog_blocker_mask(mask).count_ones() as u8
    }

    pub const fn has_backlog_blockers_in_mask(mask: u8) -> bool {
        Self::sanitize_backlog_blocker_mask(mask) != 0
    }

    pub fn has_backlog_blocker(&self, blocker_bit: u8) -> bool {
        debug_assert!(blocker_bit.is_power_of_two());
        let known_bit = Self::sanitize_backlog_blocker_mask(blocker_bit);
        known_bit != 0 && (self.backlog_blocker_mask & known_bit != 0)
    }

    pub fn has_blocker_kind(&self, blocker: BacklogBlocker) -> bool {
        self.has_backlog_blocker(blocker.bit())
    }

    pub fn backlog_blockers(&self) -> impl Iterator<Item = BacklogBlocker> + '_ {
        BacklogBlocker::ALL
            .into_iter()
            .filter(|blocker| self.has_blocker_kind(*blocker))
    }

    pub fn backlog_blocker_labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.backlog_blockers().map(|blocker| blocker.as_str())
    }

    pub fn backlog_blocker_bits(&self) -> impl Iterator<Item = u8> + '_ {
        self.backlog_blockers().map(|blocker| blocker.bit())
    }

    pub fn backlog_blockers_from_mask(mask: u8) -> impl Iterator<Item = BacklogBlocker> {
        let known_mask = Self::sanitize_backlog_blocker_mask(mask);
        BacklogBlocker::ALL
            .into_iter()
            .filter(move |blocker| known_mask & blocker.bit() != 0)
    }

    pub fn backlog_blocker_mask_from_labels<'a>(labels: impl IntoIterator<Item = &'a str>) -> u8 {
        labels
            .into_iter()
            .filter_map(BacklogBlocker::from_label)
            .fold(0_u8, |mask, blocker| mask | blocker.bit())
    }

    pub fn backlog_blocker_mask_from_delimited_labels(labels: &str) -> u8 {
        percent_decode_lossy(labels)
            .split([
                ',', ';', '|', '/', '\\', ':', '+', '&', '=', '\n', '\r', '\t', '[', ']', '{', '}',
                '(', ')', '<', '>', '"', '\'', '`',
            ])
            .filter_map(BacklogBlocker::from_label)
            .fold(0_u8, |mask, blocker| mask | blocker.bit())
    }

    pub fn backlog_blocker_labels_from_mask(mask: u8) -> impl Iterator<Item = &'static str> {
        Self::backlog_blockers_from_mask(mask).map(BacklogBlocker::as_str)
    }

    pub fn backlog_blocker_delimited_labels_from_mask(mask: u8, delimiter: &str) -> String {
        Self::backlog_blocker_labels_from_mask(mask)
            .collect::<Vec<_>>()
            .join(delimiter)
    }

    pub fn max_replication_gap(&self) -> Index {
        self.commit_apply_gap.max(self.apply_visible_gap)
    }

    pub fn total_backlog_items(&self) -> usize {
        self.wal_unflushed_count + self.pending_batch_len + self.active_txn_count
    }

    pub fn is_fully_caught_up(&self) -> bool {
        !self.has_wal_backlog
            && !self.has_pending_batch_backlog
            && !self.has_active_txn_backlog
            && !self.has_commit_apply_gap
            && !self.has_apply_visible_gap
    }
}

fn percent_decode_lossy(input: &str) -> String {
    fn hex_value(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let bytes = input.as_bytes();
    let mut idx = 0;
    let mut decoded = String::with_capacity(input.len());
    while idx < bytes.len() {
        if bytes[idx] == b'%' && idx + 2 < bytes.len() {
            if let (Some(high), Some(low)) = (hex_value(bytes[idx + 1]), hex_value(bytes[idx + 2]))
            {
                decoded.push((high << 4 | low) as char);
                idx += 3;
                continue;
            }
        }
        decoded.push(bytes[idx] as char);
        idx += 1;
    }
    decoded
}

pub struct Engine {
    repl: LocalReplicator,
    wal: WalBuffer,
    sm: KvStateMachine,
    mvcc_store: InMemoryTupleStore,
    txn_ids_by_index: BTreeMap<Index, TxnId>,
    txn_manager: TxnManager,
    visible_up_to: Index,
    metrics: RuntimeMetrics,
    batcher: DualTriggerBatcher<PendingMutation>,
    planner: Planner,
    router: DeviceRouter<MockGpuRuntime>,
}

impl Engine {
    pub fn new_local() -> Self {
        Self::with_planner_config(PlannerConfig::default())
    }

    pub fn with_planner_config(planner_cfg: PlannerConfig) -> Self {
        Self {
            repl: LocalReplicator::leader(),
            wal: WalBuffer::default(),
            sm: KvStateMachine::default(),
            mvcc_store: InMemoryTupleStore::new(),
            txn_ids_by_index: BTreeMap::new(),
            txn_manager: TxnManager::default(),
            visible_up_to: 0,
            metrics: RuntimeMetrics::default(),
            batcher: DualTriggerBatcher::new(64, Duration::from_millis(1)),
            planner: Planner::new(planner_cfg),
            router: DeviceRouter::new(MockGpuRuntime::default()),
        }
    }

    pub fn with_batching(max_items: usize, max_wait: Duration) -> Self {
        Self::with_batching_and_planner_config(max_items, max_wait, PlannerConfig::default())
    }

    pub fn with_batching_and_planner_config(
        max_items: usize,
        max_wait: Duration,
        planner_cfg: PlannerConfig,
    ) -> Self {
        let mut s = Self::with_planner_config(planner_cfg);
        s.batcher = DualTriggerBatcher::new(max_items, max_wait);
        s
    }

    pub fn simulate_next_wal_flush_failure(&mut self) {
        self.wal.fail_next_flush();
    }

    pub fn mark_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_unavailable(gpu_id);
    }

    pub fn clear_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_unavailable(gpu_id);
    }

    pub fn mark_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_memory_pressured(gpu_id);
    }

    pub fn clear_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_memory_pressured(gpu_id);
    }

    pub fn set_gpu_runtime_saturated(&mut self, saturated: bool) {
        self.router.runtime_mut().set_saturated(saturated);
    }

    pub fn become_follower(&mut self, term: Term) {
        self.repl.become_follower(term);
    }

    pub fn become_leader(&mut self, term: Term) {
        self.repl.become_leader(term);
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.repl.become_candidate(term);
    }

    pub fn commit_mutation(
        &mut self,
        txn_id: u64,
        payload: Vec<u8>,
    ) -> Result<CommitToken, EngineError> {
        if self.repl.role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let wal_len_before = self.wal.len();
        self.wal.append(WalRecord {
            txn_id,
            payload: payload.clone(),
        });

        let token = match self.repl.propose(payload) {
            Ok(token) => token,
            Err(err) => {
                self.wal.truncate(wal_len_before);
                return Err(err);
            }
        };
        if let Err(err) = self.wal.flush_all() {
            self.repl.rollback_unapplied_from(token.index);
            self.wal.truncate(wal_len_before);
            return Err(err);
        }

        self.repl.wait_committed(token, Duration::from_millis(0))?;
        self.txn_ids_by_index.insert(token.index, txn_id);

        let to_apply: Vec<LogEntry> = self
            .repl
            .drain_committed_from(self.repl.applied_index())
            .cloned()
            .collect();

        for e in &to_apply {
            self.sm.apply(e)?;
            self.apply_mvcc_entry(e)?;
            self.repl.mark_applied(e.index);
        }

        self.visible_up_to = self.visible_up_to.max(token.index);
        self.metrics.inc_commit();

        Ok(token)
    }

    fn apply_mvcc_entry(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        let Ok(text) = std::str::from_utf8(&entry.payload) else {
            return Ok(());
        };
        let Ok(cmd) = parse_command(text) else {
            return Ok(());
        };

        let txn_id = self
            .txn_ids_by_index
            .get(&entry.index)
            .copied()
            .unwrap_or(entry.index);
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };

        match cmd {
            Command::SetKv { key, value } => {
                if let Some(tuple) = self
                    .mvcc_store
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?
                {
                    self.mvcc_store
                        .tuple_update(tuple.tuple_id, value, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                } else {
                    self.mvcc_store
                        .tuple_insert(NewTuple { key, value }, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
            }
            Command::DeleteKv { key } => {
                if let Some(tuple) = self
                    .mvcc_store
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?
                {
                    self.mvcc_store
                        .tuple_delete(tuple.tuple_id, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    pub fn enqueue_set_text(
        &mut self,
        txn_id: u64,
        text: &str,
        now: Instant,
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        match cmd {
            Command::SetKv { .. } | Command::DeleteKv { .. } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                match self.route_command(&cmd) {
                    RouteDecision::Gpu(_) => {
                        let queue_cap = self.batcher.max_items();
                        let pending = self.batcher.len();
                        if pending >= queue_cap {
                            self.metrics.inc_fallback(FallbackReason::GpuQueueSaturated);
                            return Err(ExecuteError::Engine(
                                EngineError::MutationQueueOverloaded {
                                    pending,
                                    cap: queue_cap,
                                },
                            ));
                        }

                        let maybe_batch = self.batcher.enqueue(
                            PendingMutation {
                                txn_id,
                                payload: text.as_bytes().to_vec(),
                            },
                            now,
                        );
                        if let Some(batch) = maybe_batch {
                            self.metrics.observe_pending_batch_len(batch.items.len());
                            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
                        } else {
                            self.metrics.observe_pending_batch_len(self.batcher.len());
                        }
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                    RouteDecision::Cpu => {
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                }
            }
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.txn_manager.commit(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.txn_manager.rollback(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
        }
        Ok(())
    }

    pub fn tick_batching(&mut self, now: Instant) -> Result<(), EngineError> {
        if self.repl.role() != Role::Leader {
            if self.has_pending_batch() {
                return Err(EngineError::NotLeader);
            }
            return Ok(());
        }

        if let Some(batch) = self.batcher.maybe_flush_due_to_time(now) {
            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
        }
        Ok(())
    }

    pub fn flush_admin(&mut self) -> Result<(), EngineError> {
        if self.repl.role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        if let Some(batch) = self.batcher.flush_admin() {
            self.apply_batch(batch.reason, batch.items.into_iter(), Instant::now())?;
        }
        Ok(())
    }

    fn apply_batch<I>(
        &mut self,
        reason: FlushReason,
        items: I,
        flushed_at: Instant,
    ) -> Result<(), EngineError>
    where
        I: Iterator<Item = BatchItem<PendingMutation>>,
    {
        let metric_reason = match reason {
            FlushReason::Count => BatchFlushReason::Count,
            FlushReason::Time => BatchFlushReason::Time,
            FlushReason::Admin => BatchFlushReason::Admin,
        };

        let mut remaining = items.peekable();
        while let Some(p) = remaining.next() {
            let wait = flushed_at
                .saturating_duration_since(p.enqueued_at)
                .as_millis() as u64;
            let txn_id = p.item.txn_id;
            let payload = p.item.payload.clone();

            // In no-GPU bootstrap mode, batched mutations represent the simulated
            // GPU-eligible write path. Track transfer and kernel timing envelopes
            // so telemetry contracts are stable before CUDA is wired in.
            self.metrics.observe_h2d_bytes(payload.len() as u64);
            let simulated_kernel_ms = ((payload.len() as u64) / 1024).max(1);
            self.metrics.observe_kernel_exec_ms(simulated_kernel_ms);
            let simulated_occupancy = Self::simulate_kernel_occupancy_permyriad(payload.len());
            self.metrics
                .observe_kernel_occupancy_permyriad(simulated_occupancy);

            if let Err(err) = self.commit_mutation(txn_id, payload) {
                let tail: Vec<_> = std::iter::once(p).chain(remaining).collect();
                self.batcher.requeue_front(tail);
                self.metrics.observe_pending_batch_len(self.batcher.len());
                return Err(err);
            }

            self.metrics.observe_batch_wait_ms(wait);
        }

        self.metrics.observe_pending_batch_len(self.batcher.len());
        self.metrics.inc_batch_flush(metric_reason);
        Ok(())
    }

    pub fn plan_text(&self, text: &str) -> Result<ExecutionPlan, ParseError> {
        let cmd = parse_command(text)?;
        Ok(self.planner.plan_command(&cmd))
    }

    pub fn execute_text(&mut self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::SetKv { .. } | Command::DeleteKv { .. } => match self.route_command(&cmd) {
                RouteDecision::Gpu(_) | RouteDecision::Cpu => {
                    self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                }
                RouteDecision::CpuFallback { reason, .. } => {
                    self.metrics.inc_gpu_fallback(reason);
                    self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                }
            },
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.txn_manager.commit(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.txn_manager.rollback(txn_id)?;
                if chain {
                    self.txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
        }

        Ok(())
    }

    pub fn execute_read_text(&mut self, text: &str) -> Result<Option<&str>, ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::GetKv { key } => {
                if self.repl.role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
                Ok(self.get(&key))
            }
            Command::Begin => Err(ExecuteError::NonReadCommand("BEGIN")),
            Command::Commit { .. } => Err(ExecuteError::NonReadCommand("COMMIT")),
            Command::Rollback { .. } => Err(ExecuteError::NonReadCommand("ROLLBACK")),
            Command::Flush => Err(ExecuteError::NonReadCommand("FLUSH")),
            Command::ResetAll => Err(ExecuteError::NonReadCommand("RESET ALL")),
            Command::SetKv { .. } => Err(ExecuteError::NonReadCommand("SET")),
            Command::DeleteKv { .. } => Err(ExecuteError::NonReadCommand("DEL/DELETE")),
        }
    }

    pub fn execute_mvcc_query(
        &mut self,
        query: &MvccReadQuery,
    ) -> Result<MvccReadResult, ExecuteError> {
        if self.repl.role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }

        let planned_target = DeviceTarget::Gpu(self.planner.default_gpu_id());
        let executed_target = DeviceTarget::Cpu;
        let fallback_reason = Some(FallbackReason::GpuMvccReadParityGap);
        self.metrics
            .inc_fallback(FallbackReason::GpuMvccReadParityGap);

        let rows = resolve_mvcc_source(&self.mvcc_store, &query.source, query.visibility)?;

        let projection = query.projection;
        let projected = match (query.filter.clone(), query.order, query.limit) {
            (Some(filter), Some(order), Some(limit)) => {
                collect_operator_rows(ProjectOperator::new(
                    LimitOperator::new(
                        SortOperator::new(
                            FilterOperator::new(
                                ScanOperator::new(rows),
                                move |row: &ResolvedMvccRow| mvcc_row_matches_filter(row, &filter),
                            ),
                            move |left: &ResolvedMvccRow, right: &ResolvedMvccRow| {
                                mvcc_row_cmp(left, right, order)
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
                        mvcc_row_cmp(left, right, order)
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
                            mvcc_row_cmp(left, right, order)
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
                        mvcc_row_cmp(left, right, order)
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

        let total_d2h_bytes: u64 = projected.iter().map(mvcc_read_row_size).sum();
        if total_d2h_bytes > 0 {
            self.metrics.observe_d2h_bytes(total_d2h_bytes);
        }

        Ok(MvccReadResult {
            planned_target,
            executed_target,
            fallback_reason,
            rows: projected,
        })
    }

    pub fn visible_up_to(&self) -> Index {
        self.visible_up_to
    }

    pub fn applied_len(&self) -> usize {
        self.sm.applied.len()
    }

    pub fn wal_flushed_count(&self) -> usize {
        self.wal.flushed_count()
    }

    pub fn wal_buffered_count(&self) -> usize {
        self.wal.len()
    }

    pub fn wal_unflushed_count(&self) -> usize {
        self.wal.unflushed_count()
    }

    pub fn durable_wal_records(&self) -> &[WalRecord] {
        self.wal.flushed_records()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.sm.kv.get(key).map(|s| s.as_str())
    }

    pub fn visible_state_fingerprint(&self) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x00000100000001B3;

        fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
            for b in bytes {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hash
        }

        let mut hash = FNV_OFFSET_BASIS;
        for (k, v) in &self.sm.kv {
            hash = hash_bytes(hash, k.as_bytes());
            hash = hash_bytes(hash, &[0xFF]);
            hash = hash_bytes(hash, v.as_bytes());
            hash = hash_bytes(hash, &[0x00]);
        }
        hash
    }

    pub fn active_txn_count(&self) -> usize {
        self.txn_manager.active_count()
    }

    pub fn replication_watermarks(&self) -> ReplicationWatermarks {
        let now = Instant::now();
        let pending_batch_len = self.batcher.len();
        let pending_batch_cap = self.batcher.max_items();
        let wal_unflushed_count = self.wal.unflushed_count();
        let active_txn_count = self.txn_manager.active_count();
        let role = self.repl.role();
        let pending_batch_remaining_capacity = pending_batch_cap.saturating_sub(pending_batch_len);
        let pending_batch_utilization_permyriad = if pending_batch_cap == 0 {
            0
        } else {
            let utilization =
                (pending_batch_len as u128).saturating_mul(10_000) / (pending_batch_cap as u128);
            utilization.min(10_000) as u16
        };
        let pending_batch_remaining_capacity_permyriad =
            10_000u16.saturating_sub(pending_batch_utilization_permyriad);

        let commit_index = self.repl.commit_index();
        let applied_index = self.repl.applied_index();
        let visible_index = self.visible_up_to;

        let commit_apply_gap = commit_index.saturating_sub(applied_index);
        let apply_visible_gap = applied_index.saturating_sub(visible_index);

        let oldest_active_txn_id = self.txn_manager.oldest_active_txn_id();
        let newest_active_txn_id = self.txn_manager.newest_active_txn_id();
        let has_wal_backlog = wal_unflushed_count > 0;
        let has_pending_batch_backlog = pending_batch_len > 0;
        let has_active_txn_backlog = active_txn_count > 0;
        let has_commit_apply_gap = commit_apply_gap > 0;
        let has_apply_visible_gap = apply_visible_gap > 0;
        let backlog_blocker_mask = (u8::from(has_wal_backlog)
            * ReplicationWatermarks::BACKLOG_BLOCKER_WAL)
            | (u8::from(has_pending_batch_backlog)
                * ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH)
            | (u8::from(has_active_txn_backlog)
                * ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN)
            | (u8::from(has_commit_apply_gap)
                * ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP)
            | (u8::from(has_apply_visible_gap)
                * ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP);
        let backlog_blocker_count =
            ReplicationWatermarks::backlog_blocker_count_from_mask(backlog_blocker_mask);
        let has_backlog_blockers =
            ReplicationWatermarks::has_backlog_blockers_in_mask(backlog_blocker_mask);

        let wal_checkpoint = self.wal.checkpoint_meta();

        ReplicationWatermarks {
            role,
            term: self.repl.current_term(),
            commit_index,
            applied_index,
            visible_index,
            commit_apply_gap,
            apply_visible_gap,
            snapshot_id: self.repl.snapshot_meta().snapshot_id,
            wal_flushed_count: wal_checkpoint.durable_record_count,
            wal_last_durable_txn_id: wal_checkpoint.last_durable_txn_id,
            wal_buffered_count: self.wal.len(),
            wal_unflushed_count,
            pending_batch_len,
            pending_batch_cap,
            pending_batch_remaining_capacity,
            pending_batch_utilization_permyriad,
            pending_batch_remaining_capacity_permyriad,
            pending_batch_oldest_age_ms: self
                .pending_batch_oldest_age(now)
                .map(|age| age.as_millis() as u64),
            pending_batch_time_until_deadline_ms: self
                .pending_batch_time_until_deadline(now)
                .map(|remaining| remaining.as_millis() as u64),
            active_txn_count,
            oldest_active_txn_id,
            newest_active_txn_id,
            has_wal_backlog,
            has_pending_batch_backlog,
            has_active_txn_backlog,
            has_commit_apply_gap,
            has_apply_visible_gap,
            has_backlog_blockers,
            backlog_blocker_count,
            backlog_blocker_mask,
            mutation_admission_saturated: pending_batch_len >= pending_batch_cap,
            quiescent_for_failover: role == Role::Leader
                && !has_wal_backlog
                && !has_pending_batch_backlog
                && !has_active_txn_backlog,
            follower_promotion_ready: role == Role::Follower
                && !has_commit_apply_gap
                && !has_apply_visible_gap
                && !has_wal_backlog
                && !has_pending_batch_backlog
                && !has_active_txn_backlog,
        }
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.repl.export_snapshot_meta()
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        self.repl.install_snapshot(meta);
        self.visible_up_to = self
            .visible_up_to
            .max(self.repl.snapshot_meta().last_included_index);
    }

    pub fn snapshot_meta(&self) -> SnapshotMeta {
        self.repl.snapshot_meta()
    }

    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.metrics
    }

    pub fn telemetry_snapshot(&self) -> EngineTelemetrySnapshot {
        let marks = self.replication_watermarks();
        EngineTelemetrySnapshot {
            role: marks.role,
            replication_lag: ReplicationLagSnapshot {
                commit_index: marks.commit_index,
                applied_index: marks.applied_index,
                visible_index: marks.visible_index,
                commit_apply_gap: marks.commit_apply_gap,
                apply_visible_gap: marks.apply_visible_gap,
            },
            runtime_metrics: self.metrics.snapshot(),
            snapshot_id: marks.snapshot_id,
            wal_flushed_count: marks.wal_flushed_count,
            wal_last_durable_txn_id: marks.wal_last_durable_txn_id,
            wal_buffered_count: marks.wal_buffered_count,
            wal_unflushed_count: marks.wal_unflushed_count,
            pending_batch_len: marks.pending_batch_len,
            pending_batch_cap: marks.pending_batch_cap,
            active_txn_count: marks.active_txn_count,
            backlog_blocker_count: marks.backlog_blocker_count,
            backlog_blocker_mask: marks.backlog_blocker_mask,
            mutation_admission_saturated: marks.mutation_admission_saturated,
            quiescent_for_failover: marks.quiescent_for_failover,
            follower_promotion_ready: marks.follower_promotion_ready,
            gpu_parity_fallbacks: self.metrics.fallback_counts_by_gpu_parity_issue(),
            gpu_runtime: self.router.runtime().snapshot(),
        }
    }

    pub fn status_snapshot(&self) -> EngineStatusSnapshot {
        let marks = self.replication_watermarks();
        let snapshot_meta = self.snapshot_meta();
        let runtime_metrics = self.metrics.snapshot();
        let gpu_runtime = self.router.runtime().snapshot();

        let mut active_reasons = Vec::new();
        if !gpu_runtime.unavailable_gpu_ids.is_empty() {
            active_reasons.push(ActiveFallbackReason::GpuUnavailable {
                gpu_ids: gpu_runtime.unavailable_gpu_ids.clone(),
            });
        }
        if !gpu_runtime.memory_pressured_gpu_ids.is_empty() {
            active_reasons.push(ActiveFallbackReason::GpuMemoryPressure {
                gpu_ids: gpu_runtime.memory_pressured_gpu_ids.clone(),
            });
        }
        if gpu_runtime.saturated {
            active_reasons.push(ActiveFallbackReason::GpuQueueSaturated);
        }

        EngineStatusSnapshot::new(
            marks.role,
            marks.term,
            SnapshotStatus {
                snapshot_id: snapshot_meta.snapshot_id,
                last_included_index: snapshot_meta.last_included_index,
                last_included_term: snapshot_meta.last_included_term,
                visible_index: marks.visible_index,
            },
            ReplicationLagSnapshot {
                commit_index: marks.commit_index,
                applied_index: marks.applied_index,
                visible_index: marks.visible_index,
                commit_apply_gap: marks.commit_apply_gap,
                apply_visible_gap: marks.apply_visible_gap,
            },
            ReadinessStatus {
                pending_batch_len: marks.pending_batch_len,
                pending_batch_cap: marks.pending_batch_cap,
                active_txn_count: marks.active_txn_count,
                wal_unflushed_count: marks.wal_unflushed_count,
                backlog_blocker_count: marks.backlog_blocker_count,
                backlog_blocker_mask: marks.backlog_blocker_mask,
                mutation_admission_saturated: marks.mutation_admission_saturated,
                quiescent_for_failover: marks.quiescent_for_failover,
                follower_promotion_ready: marks.follower_promotion_ready,
            },
            FallbackStatus {
                last_reason: runtime_metrics.last_fallback_reason,
                gpu_parity_fallbacks: self.metrics.fallback_counts_by_gpu_parity_issue(),
                active_reasons,
                gpu_runtime,
            },
            runtime_metrics,
        )
        .expect("engine status snapshot invariants should hold")
    }

    pub fn publish_telemetry<S: TelemetrySink>(&self, sink: &mut S) {
        sink.publish(&self.telemetry_snapshot());
    }

    pub fn pending_batch_len(&self) -> usize {
        self.batcher.len()
    }

    pub fn has_pending_batch(&self) -> bool {
        !self.batcher.is_empty()
    }

    pub fn pending_batch_oldest_age(&self, now: Instant) -> Option<Duration> {
        self.batcher
            .first_enqueued_at()
            .map(|head| now.saturating_duration_since(head))
    }

    pub fn pending_batch_time_until_deadline(&self, now: Instant) -> Option<Duration> {
        self.batcher.time_until_flush_deadline(now)
    }

    pub fn batching_config(&self) -> (usize, Duration) {
        (self.batcher.max_items(), self.batcher.max_wait())
    }

    fn route_command(&self, cmd: &Command) -> RouteDecision {
        let plan = self.planner.plan_command(cmd);
        let Some(node) = plan.nodes().first() else {
            return RouteDecision::Cpu;
        };
        self.router.route(&node.op)
    }

    fn simulate_kernel_occupancy_permyriad(payload_len: usize) -> u16 {
        // Bootstrap heuristic for no-GPU mode: scale occupancy with payload size
        // while capping at 100% to keep telemetry realistic.
        let permyriad = 2_500u64.saturating_add((payload_len as u64).saturating_mul(100));
        permyriad.min(10_000) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_db_execution::DeviceTarget;
    use gpu_db_metrics::GpuParityIssue;
    use gpu_db_observability::InMemoryTelemetrySink;

    #[test]
    fn planner_targets_mutations_to_gpu() {
        let e = Engine::new_local();
        let plan = e.plan_text("SET a=1").unwrap();

        assert_eq!(plan.nodes().len(), 1);
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(0));
    }

    #[test]
    fn planner_targets_get_to_cpu_fallback_path() {
        let e = Engine::new_local();
        let plan = e.plan_text("GET a").unwrap();

        assert_eq!(plan.nodes().len(), 1);
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Cpu);
    }

    #[test]
    fn planner_config_can_override_default_gpu_target() {
        let e = Engine::with_planner_config(PlannerConfig { default_gpu_id: 3 });
        let plan = e.plan_text("SET a=1").unwrap();

        assert_eq!(plan.nodes().len(), 1);
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(3));
    }

    #[test]
    fn wal_before_visibility_holds() {
        let mut e = Engine::new_local();
        let t = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        assert!(e.wal_flushed_count() >= 1);
        assert!(e.visible_up_to() >= t.index);
        assert!(e.applied_len() >= 1);
    }

    #[test]
    fn commit_indices_monotonic() {
        let mut e = Engine::new_local();
        let a = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let b = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
        assert!(b.index > a.index);
        assert!(e.visible_up_to() >= b.index);
    }

    #[test]
    fn execute_set_updates_state_machine() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
    }

    #[test]
    fn execute_set_accepts_session_and_local_scope_aliases() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET SESSION balance=100").unwrap();
        e.execute_text(2, "SET LOCAL balance TO 101").unwrap();

        assert_eq!(e.get("balance"), Some("101"));
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn execute_del_removes_existing_key() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.execute_text(2, "DEL balance").unwrap();

        assert_eq!(e.get("balance"), None);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn execute_delete_alias_removes_existing_key() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.execute_text(2, "DELETE balance").unwrap();

        assert_eq!(e.get("balance"), None);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn execute_read_text_get_returns_current_value_without_committing() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();

        let value = e.execute_read_text("GET balance").unwrap();
        assert_eq!(value, Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 1);
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);
        assert_eq!(
            e.metrics().last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
    }

    #[test]
    fn execute_read_text_get_missing_key_does_not_track_d2h_bytes() {
        let mut e = Engine::new_local();
        let value = e.execute_read_text("GET absent").unwrap();

        assert_eq!(value, None);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn execute_read_text_rejects_non_read_commands() {
        let mut e = Engine::new_local();

        let begin_err = e.execute_read_text("BEGIN").unwrap_err();
        assert!(matches!(begin_err, ExecuteError::NonReadCommand("BEGIN")));

        let set_err = e.execute_read_text("SET balance=100").unwrap_err();
        assert!(matches!(set_err, ExecuteError::NonReadCommand("SET")));

        let reset_err = e.execute_read_text("RESET ALL").unwrap_err();
        assert!(matches!(
            reset_err,
            ExecuteError::NonReadCommand("RESET ALL")
        ));

        let discard_err = e.execute_read_text("DISCARD TEMP").unwrap_err();
        assert!(matches!(
            discard_err,
            ExecuteError::NonReadCommand("RESET ALL")
        ));

        let del_err = e.execute_read_text("DELETE balance").unwrap_err();
        assert!(matches!(
            del_err,
            ExecuteError::NonReadCommand("DEL/DELETE")
        ));

        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn execute_read_text_rejects_get_when_not_leader() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.become_follower(2);

        let err = e.execute_read_text("GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn execute_text_get_rejects_when_not_leader() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.execute_text(1, "GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn execute_text_get_rejects_when_candidate() {
        let mut e = Engine::new_local();
        e.become_candidate(2);

        let err = e.execute_text(1, "GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn execute_text_get_tracks_d2h_bytes_for_hits_only() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();

        e.execute_text(2, "GET balance").unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);

        e.execute_text(3, "GET missing").unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);
    }

    #[test]
    fn execute_read_text_rejects_get_when_candidate() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET balance=100").unwrap();
        e.become_candidate(2);

        let err = e.execute_read_text("GET balance").unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
    }

    #[test]
    fn batching_flushes_on_count_and_updates_metric() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.enqueue_set_text(2, "SET b=2", t0).unwrap();
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.metrics().batch_flush_count, 1);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(
            e.metrics().last_batch_flush_reason(),
            Some(BatchFlushReason::Count)
        );
        assert_eq!(e.metrics().batch_wait_samples, 2);
        assert_eq!(e.metrics().batch_wait_total_ms, 0);
        assert_eq!(e.metrics().last_batch_wait_ms(), Some(0));
        assert_eq!(
            e.metrics().h2d_bytes_total,
            "SET a=1".len() as u64 + "SET b=2".len() as u64
        );
        assert_eq!(e.metrics().kernel_exec_samples, 2);
        assert_eq!(e.metrics().kernel_exec_total_ms, 2);
        assert_eq!(e.metrics().last_kernel_exec_ms(), Some(1));
        assert_eq!(e.metrics().kernel_occupancy_samples, 2);
        assert_eq!(e.metrics().kernel_occupancy_total_permyriad, 6400);
        assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(3200));
        assert_eq!(e.metrics().pending_batch_peak, 2);
        assert_eq!(e.metrics().last_pending_batch_len(), Some(0));
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn batching_flushes_on_time() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=7", t0).unwrap();
        assert!(e.has_pending_batch());
        assert_eq!(e.pending_batch_len(), 1);
        e.tick_batching(t0 + Duration::from_millis(3)).unwrap();
        assert_eq!(e.get("a"), Some("7"));
        assert!(!e.has_pending_batch());
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 1);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Time), 1);
        assert_eq!(e.metrics().batch_wait_samples, 1);
        assert_eq!(e.metrics().batch_wait_total_ms, 3);
        assert_eq!(e.metrics().last_batch_wait_ms(), Some(3));
    }

    #[test]
    fn batching_kernel_occupancy_caps_at_full_utilization() {
        let mut e = Engine::with_batching(1, Duration::from_secs(999));
        let t0 = Instant::now();
        let payload = format!("SET a={}", "x".repeat(512));

        e.enqueue_set_text(1, &payload, t0).unwrap();

        assert_eq!(e.metrics().kernel_occupancy_samples, 1);
        assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(10_000));
        assert_eq!(e.metrics().kernel_occupancy_total_permyriad, 10_000);
    }

    #[test]
    fn admin_flush_tracks_reason() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=9", t0).unwrap();
        assert_eq!(e.pending_batch_len(), 1);
        e.flush_admin().unwrap();

        assert_eq!(e.get("a"), Some("9"));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    }

    #[test]
    fn admin_flush_without_pending_queue_is_noop() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.flush_admin().unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.pending_batch_oldest_age(t0), None);
        assert_eq!(e.pending_batch_time_until_deadline(t0), None);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn batching_config_reflects_engine_settings() {
        let e = Engine::with_batching(7, Duration::from_millis(42));
        assert_eq!(e.batching_config(), (7, Duration::from_millis(42)));
    }

    #[test]
    fn batching_and_planner_config_can_be_combined() {
        let e = Engine::with_batching_and_planner_config(
            3,
            Duration::from_millis(9),
            PlannerConfig { default_gpu_id: 5 },
        );

        assert_eq!(e.batching_config(), (3, Duration::from_millis(9)));
        let plan = e.plan_text("SET a=1").unwrap();
        assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(5));
    }

    #[test]
    fn pending_batch_deadline_counts_down_and_clears_after_flush() {
        let mut e = Engine::with_batching(10, Duration::from_millis(10));
        let t0 = Instant::now();

        assert_eq!(e.pending_batch_time_until_deadline(t0), None);

        e.enqueue_set_text(1, "SET a=9", t0).unwrap();
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(4)),
            Some(Duration::from_millis(6))
        );
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(12)),
            Some(Duration::ZERO)
        );

        e.flush_admin().unwrap();
        assert_eq!(
            e.pending_batch_time_until_deadline(t0 + Duration::from_millis(13)),
            None
        );
    }

    #[test]
    fn pending_batch_oldest_age_tracks_then_clears_after_flush() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=9", t0).unwrap();

        let age = e
            .pending_batch_oldest_age(t0 + Duration::from_millis(5))
            .expect("pending batch age should exist");
        assert!(age >= Duration::from_millis(5));

        e.flush_admin().unwrap();
        assert_eq!(
            e.pending_batch_oldest_age(t0 + Duration::from_millis(6)),
            None
        );
    }

    #[test]
    fn batching_can_apply_set_then_del_in_order() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.enqueue_set_text(2, "DEL a", t0).unwrap();

        assert_eq!(e.get("a"), None);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn deterministic_replay_matches_between_immediate_and_batched_mutation_paths() {
        let trace = [
            (1, "SET acct_a=10"),
            (2, "SET acct_b=25"),
            (3, "DEL acct_a"),
            (4, "SET acct_c=77"),
            (5, "DELETE acct_b"),
            (6, "SET acct_a=99"),
        ];

        let mut immediate = Engine::new_local();
        for (txn_id, cmd) in trace {
            immediate.execute_text(txn_id, cmd).unwrap();
        }

        let mut batched = Engine::with_batching(64, Duration::from_secs(999));
        let t0 = Instant::now();
        for (txn_id, cmd) in trace {
            batched.enqueue_set_text(txn_id, cmd, t0).unwrap();
        }
        batched.flush_admin().unwrap();

        assert_eq!(immediate.sm.applied, batched.sm.applied);
        assert_eq!(immediate.sm.kv, batched.sm.kv);
        assert_eq!(immediate.visible_up_to(), batched.visible_up_to());
        assert_eq!(
            immediate.visible_state_fingerprint(),
            batched.visible_state_fingerprint()
        );
        assert_eq!(immediate.wal_flushed_count(), trace.len());
        assert_eq!(batched.wal_flushed_count(), trace.len());
    }

    #[test]
    fn visible_state_fingerprint_changes_with_visible_kv_state() {
        let mut e = Engine::new_local();
        let empty = e.visible_state_fingerprint();

        e.execute_text(1, "SET a=1").unwrap();
        let after_set = e.visible_state_fingerprint();
        assert_ne!(after_set, empty);

        e.execute_text(2, "DELETE a").unwrap();
        let after_delete = e.visible_state_fingerprint();
        assert_eq!(after_delete, empty);
    }

    #[test]
    fn flush_command_drains_pending_batch() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=5", t0).unwrap();
        e.execute_text(2, "FLUSH").unwrap();

        assert_eq!(e.get("a"), Some("5"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    }

    #[test]
    fn flush_aliases_drain_pending_batch() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=5", t0).unwrap();
        e.execute_text(2, "FLUSH WAL").unwrap();

        e.enqueue_set_text(3, "SET b=7", t0).unwrap();
        e.execute_text(4, "FLUSH LOG").unwrap();

        assert_eq!(e.get("a"), Some("5"));
        assert_eq!(e.get("b"), Some("7"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 2);
    }

    #[test]
    fn wal_flush_failure_prevents_visibility_advance() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();
        let res = e.commit_mutation(1, b"SET a=1".to_vec());
        assert!(matches!(res, Err(EngineError::Durability(_))));
        assert_eq!(e.visible_up_to(), 0);
    }

    #[test]
    fn wal_flush_failure_does_not_leak_into_later_successful_commit() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();
        let _ = e.commit_mutation(1, b"SET a=1".to_vec());

        e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

        assert_eq!(e.get("a"), None);
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.applied_len(), 1);
    }

    #[test]
    fn wal_flush_failure_discards_unflushed_record_from_buffer() {
        let mut e = Engine::new_local();
        e.simulate_next_wal_flush_failure();

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

        assert!(matches!(err, EngineError::Durability(_)));
        assert_eq!(e.wal_flushed_count(), 0);
        assert_eq!(e.wal_buffered_count(), 0);
        assert_eq!(e.wal_unflushed_count(), 0);
    }

    #[test]
    fn durable_wal_records_exclude_failed_commit_attempts() {
        let mut e = Engine::new_local();

        e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        e.simulate_next_wal_flush_failure();
        let _ = e.commit_mutation(2, b"SET b=2".to_vec());

        let durable = e.durable_wal_records();
        assert_eq!(durable.len(), 1);
        assert_eq!(durable[0].txn_id, 1);
        assert_eq!(durable[0].payload, b"SET a=1".to_vec());
    }

    #[test]
    fn follower_rejects_commit_without_visibility_or_wal_flush() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.visible_up_to(), 0);
        assert_eq!(e.wal_flushed_count(), 0);
        assert_eq!(e.applied_len(), 0);
    }

    #[test]
    fn follower_rejects_batched_enqueue_without_mutating_queue_or_metrics() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_follower(2);

        let t0 = Instant::now();
        let err = e.enqueue_set_text(1, "SET a=1", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_get_rejects_when_not_leader_without_queue_side_effects() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_follower(2);

        let t0 = Instant::now();
        let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_get_rejects_when_candidate_without_queue_side_effects() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_candidate(2);

        let t0 = Instant::now();
        let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

        assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().fallback_total, 0);
        assert_eq!(e.metrics().d2h_bytes_total, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_get_tracks_d2h_bytes_for_hits_only() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET balance=100", t0).unwrap();
        e.flush_admin().unwrap();

        e.enqueue_set_text(2, "GET balance", t0).unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);

        e.enqueue_set_text(3, "GET missing", t0).unwrap();
        assert_eq!(e.metrics().d2h_bytes_total, "100".len() as u64);
    }

    #[test]
    fn execute_text_mutation_falls_back_to_cpu_when_gpu_is_unavailable() {
        let mut e = Engine::new_local();
        e.mark_gpu_unavailable(0);

        e.execute_text(1, "SET balance=100").unwrap();

        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);

        let snapshot = e.telemetry_snapshot();
        assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
        assert!(snapshot.has_gpu_parity_fallbacks());
        assert!(snapshot.has_gpu_runtime_pressure());
        assert_eq!(snapshot.blocked_gpu_ids(), vec![0]);
        assert_eq!(
            snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
                id: "GPU-120",
                owner: "runtime",
                milestone: "m0-bootstrap",
            }),
            Some(&1)
        );
    }

    #[test]
    fn enqueue_mutation_falls_back_to_cpu_when_gpu_is_memory_pressured() {
        let mut e = Engine::with_batching(8, Duration::from_secs(60));
        let t0 = Instant::now();
        e.mark_gpu_memory_pressured(0);

        e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(
            e.metrics().fallback_for(FallbackReason::GpuMemoryPressure),
            1
        );
    }

    #[test]
    fn enqueue_mutation_runtime_saturation_falls_back_before_queueing() {
        let mut e = Engine::with_batching(8, Duration::from_secs(60));
        let t0 = Instant::now();
        e.set_gpu_runtime_saturated(true);

        e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

        assert_eq!(e.get("balance"), Some("100"));
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 1);
        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(
            e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
            1
        );

        let snapshot = e.telemetry_snapshot();
        assert!(snapshot.has_gpu_runtime_pressure());
        assert!(snapshot.blocked_gpu_ids().is_empty());
        assert!(snapshot.gpu_runtime.saturated);
    }

    #[test]
    fn candidate_rejects_commit_and_batched_enqueue() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        e.become_candidate(2);

        let commit_err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
        assert!(matches!(commit_err, EngineError::NotLeader));

        let enqueue_err = e
            .enqueue_set_text(1, "SET a=1", Instant::now())
            .unwrap_err();
        assert!(matches!(
            enqueue_err,
            ExecuteError::Engine(EngineError::NotLeader)
        ));

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 0);
        assert_eq!(e.visible_up_to(), 0);
    }

    #[test]
    fn failed_admin_flush_does_not_increment_flush_metrics_or_drop_pending_queue() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let err = e.flush_admin().unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
        assert_eq!(e.metrics().last_batch_flush_reason(), None);
        assert_eq!(e.metrics().commits_total, 0);
        assert_eq!(e.pending_batch_len(), 1);
    }

    #[test]
    fn batch_flush_wal_failure_requeues_items_for_retry() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.simulate_next_wal_flush_failure();
        let err = e
            .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
            .unwrap_err();

        assert!(matches!(
            err,
            ExecuteError::Engine(EngineError::Durability(_))
        ));
        assert_eq!(e.pending_batch_len(), 2);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().batch_wait_samples, 0);
        assert_eq!(e.metrics().pending_batch_peak, 2);
        assert_eq!(e.metrics().last_pending_batch_len(), Some(2));
        assert_eq!(e.metrics().commits_total, 0);

        e.flush_admin().unwrap();
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.get("b"), Some("2"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
        assert_eq!(e.metrics().commits_total, 2);
    }

    #[test]
    fn enqueue_rejects_new_mutation_when_retry_queue_is_saturated() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.simulate_next_wal_flush_failure();
        let flush_err = e
            .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
            .unwrap_err();
        assert!(matches!(
            flush_err,
            ExecuteError::Engine(EngineError::Durability(_))
        ));
        assert_eq!(e.pending_batch_len(), 2);

        let saturated_err = e
            .enqueue_set_text(3, "SET c=3", t0 + Duration::from_millis(2))
            .unwrap_err();
        assert!(matches!(
            saturated_err,
            ExecuteError::Engine(EngineError::MutationQueueOverloaded { pending: 2, cap: 2 })
        ));
        assert_eq!(e.pending_batch_len(), 2);
        assert_eq!(
            e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
            1
        );
        assert_eq!(e.metrics().commits_total, 0);

        let snapshot = e.telemetry_snapshot();
        assert!(snapshot.has_gpu_parity_fallbacks());
        assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
        assert_eq!(
            snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
                id: "GPU-121",
                owner: "runtime",
                milestone: "m0-bootstrap",
            }),
            Some(&1)
        );
    }

    #[test]
    fn failed_time_flush_does_not_drop_pending_queue() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        let t0 = Instant::now();
        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let err = e.tick_batching(t0 + Duration::from_millis(3)).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(e.pending_batch_len(), 1);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn follower_tick_without_pending_batch_is_noop() {
        let mut e = Engine::with_batching(10, Duration::from_millis(2));
        e.become_follower(2);

        e.tick_batching(Instant::now()).unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().batch_flush_count, 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn pending_batch_can_be_flushed_after_follower_is_promoted_back_to_leader() {
        let mut e = Engine::with_batching(10, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(2);

        let tick_err = e.tick_batching(t0 + Duration::from_secs(1)).unwrap_err();
        assert!(matches!(tick_err, EngineError::NotLeader));
        assert_eq!(e.pending_batch_len(), 1);
        assert_eq!(e.get("a"), None);

        e.become_leader(3);
        e.flush_admin().unwrap();

        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.get("a"), Some("1"));
        assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
        assert_eq!(e.metrics().commits_total, 1);
    }

    #[test]
    fn execute_text_non_mutations_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::new_local();

        e.execute_text(1, "BEGIN").unwrap();
        e.execute_text(1, "COMMIT").unwrap();
        e.execute_text(2, "BEGIN").unwrap();
        e.execute_text(2, "ROLLBACK").unwrap();
        e.execute_text(3, "GET missing").unwrap();
        e.execute_text(4, "FLUSH").unwrap();
        e.execute_text(5, "RESET ALL").unwrap();
        e.execute_text(6, "DISCARD TEMP").unwrap();

        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 8);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 8);
        assert_eq!(
            e.metrics().last_fallback_reason(),
            Some(FallbackReason::NotGpuEligible)
        );
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn enqueue_non_mutations_count_as_not_gpu_eligible_fallbacks() {
        let mut e = Engine::with_batching(2, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "BEGIN", t0).unwrap();
        e.enqueue_set_text(1, "COMMIT", t0).unwrap();
        e.enqueue_set_text(2, "BEGIN", t0).unwrap();
        e.enqueue_set_text(2, "ROLLBACK", t0).unwrap();
        e.enqueue_set_text(3, "GET missing", t0).unwrap();
        e.enqueue_set_text(4, "FLUSH", t0).unwrap();
        e.enqueue_set_text(5, "RESET ALL", t0).unwrap();
        e.enqueue_set_text(6, "DISCARD TEMP", t0).unwrap();

        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 8);
        assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 8);
        assert_eq!(e.pending_batch_len(), 0);
        assert_eq!(e.metrics().commits_total, 0);
    }

    #[test]
    fn commit_and_rollback_require_active_transaction_context() {
        let mut e = Engine::new_local();

        let commit_err = e.execute_text(10, "COMMIT").unwrap_err();
        assert!(matches!(
            commit_err,
            ExecuteError::Txn(TxnError::NotFound(10))
        ));

        let rollback_err = e.execute_text(11, "ROLLBACK").unwrap_err();
        assert!(matches!(
            rollback_err,
            ExecuteError::Txn(TxnError::NotFound(11))
        ));

        e.execute_text(12, "BEGIN").unwrap();
        let duplicate_begin_err = e.execute_text(12, "BEGIN").unwrap_err();
        assert!(matches!(
            duplicate_begin_err,
            ExecuteError::Txn(TxnError::AlreadyExists(12))
        ));

        assert_eq!(e.metrics().fallback_total, 1);
        assert_eq!(e.active_txn_count(), 1);
    }

    #[test]
    fn and_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();

        e.execute_text(21, "BEGIN").unwrap();
        e.execute_text(21, "COMMIT AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(22, "COMMIT").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(31, "BEGIN").unwrap();
        e.execute_text(31, "ROLLBACK AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(32, "ROLLBACK").unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn enqueue_non_mutation_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();
        let t0 = Instant::now();

        e.enqueue_set_text(41, "BEGIN", t0).unwrap();
        e.enqueue_set_text(41, "COMMIT AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(42, "COMMIT", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(51, "BEGIN", t0).unwrap();
        e.enqueue_set_text(51, "ROLLBACK AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(52, "ROLLBACK", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn transaction_control_alias_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();

        e.execute_text(61, "BEGIN").unwrap();
        e.execute_text(61, "END AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(62, "END").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(71, "BEGIN").unwrap();
        e.execute_text(71, "ABORT AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(72, "ABORT").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(75, "BEGIN").unwrap();
        e.execute_text(75, "END WORK AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(76, "COMMIT").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(77, "BEGIN").unwrap();
        e.execute_text(77, "ABORT WORK AND CHAIN").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(78, "ROLLBACK").unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn start_alias_and_work_aliases_drive_transaction_state_transitions() {
        let mut e = Engine::new_local();

        e.execute_text(73, "START TRANSACTION READ ONLY").unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(73, "COMMIT WORK").unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.execute_text(74, "START WORK, READ WRITE, DEFERRABLE")
            .unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.execute_text(74, "ROLLBACK TRANSACTION").unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn enqueue_transaction_control_alias_chain_forms_reopen_transaction_context() {
        let mut e = Engine::new_local();
        let t0 = Instant::now();

        e.enqueue_set_text(81, "BEGIN", t0).unwrap();
        e.enqueue_set_text(81, "END AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(82, "END", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(91, "BEGIN", t0).unwrap();
        e.enqueue_set_text(91, "ABORT AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(92, "ABORT", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(95, "BEGIN", t0).unwrap();
        e.enqueue_set_text(95, "END WORK AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(96, "COMMIT", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(97, "BEGIN", t0).unwrap();
        e.enqueue_set_text(97, "ABORT WORK AND CHAIN", t0).unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(98, "ROLLBACK", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn enqueue_start_alias_and_work_aliases_drive_transaction_state_transitions() {
        let mut e = Engine::new_local();
        let t0 = Instant::now();

        e.enqueue_set_text(93, "START TRANSACTION READ ONLY", t0)
            .unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(93, "COMMIT WORK", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);

        e.enqueue_set_text(94, "START WORK, READ WRITE, DEFERRABLE", t0)
            .unwrap();
        assert_eq!(e.active_txn_count(), 1);
        e.enqueue_set_text(94, "ROLLBACK TRANSACTION", t0).unwrap();
        assert_eq!(e.active_txn_count(), 0);
    }

    #[test]
    fn commit_and_chain_propagates_txn_id_exhaustion() {
        let mut e = Engine::new_local();

        e.execute_text(u64::MAX, "BEGIN").unwrap();
        let err = e.execute_text(u64::MAX, "COMMIT AND CHAIN").unwrap_err();

        assert!(matches!(err, ExecuteError::Txn(TxnError::IdExhausted)));
        assert_eq!(e.active_txn_count(), 0);
        assert_eq!(e.metrics().fallback_total, 1);
    }

    #[test]
    fn replication_watermarks_track_commit_apply_visibility_and_durability() {
        let mut e = Engine::new_local();

        let before = e.replication_watermarks();
        assert_eq!(before.role, Role::Leader);
        assert_eq!(before.commit_index, 0);
        assert_eq!(before.applied_index, 0);
        assert_eq!(before.visible_index, 0);
        assert_eq!(before.commit_apply_gap, 0);
        assert_eq!(before.apply_visible_gap, 0);
        assert_eq!(before.snapshot_id, 0);
        assert_eq!(before.wal_flushed_count, 0);
        assert_eq!(before.wal_buffered_count, 0);
        assert_eq!(before.wal_unflushed_count, 0);
        assert_eq!(before.pending_batch_len, 0);
        assert_eq!(before.pending_batch_cap, 64);
        assert_eq!(before.pending_batch_remaining_capacity, 64);
        assert_eq!(before.pending_batch_utilization_permyriad, 0);
        assert_eq!(before.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(before.pending_batch_oldest_age_ms, None);
        assert_eq!(before.pending_batch_time_until_deadline_ms, None);
        assert_eq!(before.active_txn_count, 0);
        assert_eq!(before.oldest_active_txn_id, None);
        assert_eq!(before.newest_active_txn_id, None);
        assert!(!before.has_wal_backlog);
        assert!(!before.has_pending_batch_backlog);
        assert!(!before.has_active_txn_backlog);
        assert!(!before.has_commit_apply_gap);
        assert!(!before.has_apply_visible_gap);
        assert!(!before.has_backlog_blockers);
        assert_eq!(before.backlog_blocker_count, 0);
        assert_eq!(before.backlog_blocker_mask, 0);
        assert!(!before.mutation_admission_saturated);
        assert!(before.quiescent_for_failover);
        assert!(!before.follower_promotion_ready);

        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let after = e.replication_watermarks();

        assert_eq!(after.role, Role::Leader);
        assert!(after.term >= before.term);
        assert_eq!(after.commit_index, token.index);
        assert_eq!(after.applied_index, token.index);
        assert_eq!(after.visible_index, token.index);
        assert_eq!(after.commit_apply_gap, 0);
        assert_eq!(after.apply_visible_gap, 0);
        assert_eq!(after.snapshot_id, 0);
        assert!(after.wal_flushed_count >= 1);
        assert_eq!(after.wal_buffered_count, e.wal_buffered_count());
        assert_eq!(after.wal_unflushed_count, e.wal_unflushed_count());
        assert_eq!(after.pending_batch_len, 0);
        assert_eq!(after.pending_batch_cap, 64);
        assert_eq!(after.pending_batch_remaining_capacity, 64);
        assert_eq!(after.pending_batch_utilization_permyriad, 0);
        assert_eq!(after.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(after.pending_batch_oldest_age_ms, None);
        assert_eq!(after.pending_batch_time_until_deadline_ms, None);
        assert_eq!(after.active_txn_count, 0);
        assert_eq!(after.oldest_active_txn_id, None);
        assert_eq!(after.newest_active_txn_id, None);
        assert!(!after.has_wal_backlog);
        assert!(!after.has_pending_batch_backlog);
        assert!(!after.has_active_txn_backlog);
        assert!(!after.has_commit_apply_gap);
        assert!(!after.has_apply_visible_gap);
        assert!(!after.has_backlog_blockers);
        assert_eq!(after.backlog_blocker_count, 0);
        assert_eq!(after.backlog_blocker_mask, 0);
        assert!(!after.mutation_admission_saturated);
        assert!(after.quiescent_for_failover);
        assert!(!after.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_do_not_advance_on_rejected_follower_commit() {
        let mut e = Engine::new_local();
        e.become_follower(2);

        let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));

        let marks = e.replication_watermarks();
        assert_eq!(marks.role, Role::Follower);
        assert_eq!(marks.term, 2);
        assert_eq!(marks.commit_index, 0);
        assert_eq!(marks.applied_index, 0);
        assert_eq!(marks.visible_index, 0);
        assert_eq!(marks.commit_apply_gap, 0);
        assert_eq!(marks.apply_visible_gap, 0);
        assert_eq!(marks.wal_flushed_count, 0);
        assert_eq!(marks.wal_last_durable_txn_id, None);
        assert_eq!(marks.wal_buffered_count, 0);
        assert_eq!(marks.wal_unflushed_count, 0);
        assert_eq!(marks.pending_batch_len, 0);
        assert_eq!(marks.pending_batch_cap, 64);
        assert_eq!(marks.pending_batch_remaining_capacity, 64);
        assert_eq!(marks.pending_batch_utilization_permyriad, 0);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(marks.pending_batch_oldest_age_ms, None);
        assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
        assert_eq!(marks.active_txn_count, 0);
        assert!(!marks.has_wal_backlog);
        assert!(!marks.has_pending_batch_backlog);
        assert!(!marks.has_active_txn_backlog);
        assert!(!marks.has_commit_apply_gap);
        assert!(!marks.has_apply_visible_gap);
        assert!(!marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_mask, 0);
        assert!(!marks.mutation_admission_saturated);
        assert!(!marks.quiescent_for_failover);
        assert!(marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_include_buffered_wal_records() {
        let mut e = Engine::new_local();

        e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.wal_buffered_count, 2);
        assert_eq!(marks.wal_flushed_count, 2);
        assert_eq!(marks.wal_last_durable_txn_id, Some(2));
        assert_eq!(marks.wal_unflushed_count, 0);
        assert_eq!(marks.pending_batch_len, 0);
        assert_eq!(marks.pending_batch_cap, 64);
        assert_eq!(marks.pending_batch_remaining_capacity, 64);
        assert_eq!(marks.pending_batch_utilization_permyriad, 0);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
        assert_eq!(marks.pending_batch_oldest_age_ms, None);
        assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
        assert_eq!(marks.active_txn_count, 0);
        assert!(!marks.mutation_admission_saturated);
        assert!(marks.quiescent_for_failover);
    }

    #[test]
    fn replication_watermarks_pending_batch_time_fields_clear_after_flush() {
        let mut e = Engine::with_batching(3, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        let before_flush = e.replication_watermarks();
        assert_eq!(before_flush.pending_batch_len, 1);
        assert_eq!(before_flush.pending_batch_cap, 3);
        assert_eq!(before_flush.pending_batch_remaining_capacity, 2);
        assert_eq!(before_flush.pending_batch_utilization_permyriad, 3_333);
        assert_eq!(
            before_flush.pending_batch_remaining_capacity_permyriad,
            6_667
        );
        assert!(before_flush.pending_batch_oldest_age_ms.is_some());
        assert!(before_flush.pending_batch_time_until_deadline_ms.is_some());

        e.flush_admin().unwrap();
        let after_flush = e.replication_watermarks();
        assert_eq!(after_flush.pending_batch_len, 0);
        assert_eq!(after_flush.pending_batch_cap, 3);
        assert_eq!(after_flush.pending_batch_remaining_capacity, 3);
        assert_eq!(after_flush.pending_batch_utilization_permyriad, 0);
        assert_eq!(
            after_flush.pending_batch_remaining_capacity_permyriad,
            10_000
        );
        assert_eq!(after_flush.pending_batch_oldest_age_ms, None);
        assert_eq!(after_flush.pending_batch_time_until_deadline_ms, None);
    }

    #[test]
    fn replication_watermarks_include_pending_batch_depth() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.pending_batch_len, 1);
        assert_eq!(marks.pending_batch_cap, 2);
        assert_eq!(marks.pending_batch_remaining_capacity, 1);
        assert_eq!(marks.pending_batch_utilization_permyriad, 5_000);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 5_000);
        assert!(marks.pending_batch_oldest_age_ms.is_some());
        assert!(marks.pending_batch_time_until_deadline_ms.is_some());
        assert!(marks.has_pending_batch_backlog);
        assert!(marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_count, 1);
        assert_eq!(
            marks.backlog_blocker_mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
        );
        assert!(!marks.has_wal_backlog);
        assert!(!marks.has_active_txn_backlog);
        assert!(!marks.has_commit_apply_gap);
        assert!(!marks.has_apply_visible_gap);
        assert_eq!(marks.wal_buffered_count, 0);
        assert_eq!(marks.wal_unflushed_count, 0);
        assert_eq!(marks.active_txn_count, 0);
        assert!(!marks.quiescent_for_failover);
    }

    #[test]
    fn replication_watermarks_flag_mutation_admission_saturation() {
        let mut e = Engine::with_batching(2, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.simulate_next_wal_flush_failure();
        let err = e
            .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
            .unwrap_err();
        assert!(matches!(
            err,
            ExecuteError::Engine(EngineError::Durability(_))
        ));

        let marks = e.replication_watermarks();
        assert_eq!(marks.pending_batch_len, 2);
        assert_eq!(marks.pending_batch_cap, 2);
        assert_eq!(marks.pending_batch_remaining_capacity, 0);
        assert_eq!(marks.pending_batch_utilization_permyriad, 10_000);
        assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 0);
        assert!(marks.has_pending_batch_backlog);
        assert!(!marks.has_wal_backlog);
        assert!(!marks.has_active_txn_backlog);
        assert!(marks.mutation_admission_saturated);
        assert!(!marks.quiescent_for_failover);
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_follower_promotion_ready_requires_no_backlog() {
        let mut e = Engine::with_batching(8, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();
        e.become_follower(3);

        let marks = e.replication_watermarks();
        assert_eq!(marks.role, Role::Follower);
        assert_eq!(marks.pending_batch_len, 1);
        assert!(!marks.quiescent_for_failover);
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_follower_promotion_ready_requires_zero_active_txns() {
        let mut e = Engine::new_local();
        e.become_follower(5);
        e.execute_text(9, "BEGIN").unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.role, Role::Follower);
        assert_eq!(marks.active_txn_count, 1);
        assert_eq!(marks.oldest_active_txn_id, Some(9));
        assert_eq!(marks.newest_active_txn_id, Some(9));
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn replication_watermarks_include_active_transaction_count() {
        let mut e = Engine::new_local();

        e.execute_text(42, "BEGIN").unwrap();

        let marks = e.replication_watermarks();
        assert_eq!(marks.active_txn_count, 1);
        assert_eq!(marks.oldest_active_txn_id, Some(42));
        assert_eq!(marks.newest_active_txn_id, Some(42));
        assert_eq!(marks.pending_batch_len, 0);
        assert_eq!(marks.pending_batch_oldest_age_ms, None);
        assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
        assert!(marks.has_active_txn_backlog);
        assert!(marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_count, 1);
        assert_eq!(
            marks.backlog_blocker_mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        );
        assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
        assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
        assert!(!marks.has_pending_batch_backlog);
        assert!(!marks.has_wal_backlog);
        assert_eq!(marks.wal_buffered_count, 0);
        assert!(!marks.mutation_admission_saturated);
        assert!(!marks.quiescent_for_failover);
    }

    #[test]
    fn backlog_blocker_enum_roundtrips_through_bits_and_labels() {
        for blocker in BacklogBlocker::ALL {
            assert!(blocker.bit().is_power_of_two());
            assert!(!blocker.as_str().is_empty());
            assert_eq!(BacklogBlocker::from_bit(blocker.bit()), Some(blocker));
            assert_eq!(BacklogBlocker::from_label(blocker.as_str()), Some(blocker));

            let mut marks = Engine::new_local().replication_watermarks();
            marks.backlog_blocker_mask = blocker.bit();
            assert!(marks.has_blocker_kind(blocker));
            assert_eq!(marks.backlog_blockers().collect::<Vec<_>>(), vec![blocker]);
            assert_eq!(
                marks.backlog_blocker_labels().collect::<Vec<_>>(),
                vec![blocker.as_str()]
            );
            assert_eq!(
                marks.backlog_blocker_bits().collect::<Vec<_>>(),
                vec![blocker.bit()]
            );
            assert_eq!(
                ReplicationWatermarks::backlog_blockers_from_mask(marks.backlog_blocker_mask)
                    .collect::<Vec<_>>(),
                vec![blocker]
            );
        }
    }

    #[test]
    fn backlog_blocker_display_and_from_str_roundtrip() {
        for blocker in BacklogBlocker::ALL {
            let label = blocker.to_string();
            assert_eq!(label, blocker.as_str());
            assert_eq!(label.parse::<BacklogBlocker>(), Ok(blocker));
        }
    }

    #[test]
    fn backlog_blocker_from_str_reports_unknown_label() {
        let err = "  not-a-real-blocker  "
            .parse::<BacklogBlocker>()
            .expect_err("unknown blocker labels should fail to parse");

        assert_eq!(err.label(), "not-a-real-blocker");
        assert_eq!(
            err.to_string(),
            "unknown backlog blocker label: not-a-real-blocker"
        );
    }

    #[test]
    fn replication_watermarks_backlog_blockers_from_mask_ignores_unknown_bits() {
        let known_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN;
        let unknown_mask = 1 << 7;

        assert_eq!(BacklogBlocker::from_bit(unknown_mask), None);
        assert_eq!(
            ReplicationWatermarks::backlog_blockers_from_mask(known_mask | unknown_mask)
                .collect::<Vec<_>>(),
            vec![BacklogBlocker::Wal, BacklogBlocker::ActiveTxn]
        );
    }

    #[test]
    fn backlog_blocker_mask_helpers_strip_unknown_bits() {
        let unknown_mask = (1 << 5) | (1 << 7);
        let mixed_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | unknown_mask;

        assert_eq!(
            ReplicationWatermarks::known_backlog_blocker_mask(),
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
        assert_eq!(
            ReplicationWatermarks::unknown_backlog_blocker_mask(mixed_mask),
            unknown_mask
        );
        assert_eq!(
            ReplicationWatermarks::sanitize_backlog_blocker_mask(mixed_mask),
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blocker_count_from_mask(mixed_mask),
            2
        );
        assert!(ReplicationWatermarks::has_backlog_blockers_in_mask(
            mixed_mask
        ));
        assert!(!ReplicationWatermarks::has_backlog_blockers_in_mask(
            unknown_mask
        ));
    }

    #[test]
    fn replication_watermarks_backlog_blocker_mask_from_labels_ignores_unknowns() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
            "pending_batch",
            "unknown",
            "active_txn",
            "pending_batch",
        ]);

        assert_eq!(BacklogBlocker::from_label("unknown"), None);
        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blockers_from_mask(mask).collect::<Vec<_>>(),
            vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blocker_labels_from_mask(mask).collect::<Vec<_>>(),
            vec!["pending_batch", "active_txn"]
        );
    }

    #[test]
    fn backlog_blocker_label_decode_normalizes_case_spacing_and_hyphenation() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
            " WAL ",
            "pending-batch",
            "ACTIVE TXN",
            "commit-apply-gap",
            "apply visible gap",
            "pending.batch",
            "commit--apply  gap",
        ]);

        assert_eq!(
            BacklogBlocker::from_label("PENDING-BATCH"),
            Some(BacklogBlocker::PendingBatch)
        );
        assert_eq!(
            BacklogBlocker::from_label("apply visible gap"),
            Some(BacklogBlocker::ApplyVisibleGap)
        );
        assert_eq!(
            BacklogBlocker::from_label("pending.batch"),
            Some(BacklogBlocker::PendingBatch)
        );
        assert_eq!(
            BacklogBlocker::from_label("commit--apply  gap"),
            Some(BacklogBlocker::CommitApplyGap)
        );
        assert_eq!(
            BacklogBlocker::from_label("__wal__"),
            Some(BacklogBlocker::Wal)
        );
        assert_eq!(
            BacklogBlocker::from_label("___active.txn___"),
            Some(BacklogBlocker::ActiveTxn)
        );
        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_decodes_csv_like_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal, pending-batch; ACTIVE TXN | unknown / apply visible gap : commit apply gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_ignores_empty_segments() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            " , ; | pending_batch || wal ,, ",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_multiline_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal\n pending_batch\r\nACTIVE TXN\t| commit apply gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_jsonish_arrays() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "[\"wal\",\"active_txn\",\"apply visible gap\"]",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_braced_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "{'wal';'active_txn';'apply visible gap'}",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_parenthesized_and_angle_bracket_streams()
    {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "<(wal|active_txn|apply visible gap)>",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_single_quoted_arrays() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "['pending-batch','commit_apply_gap']",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_backtick_quoted_labels() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "`wal`,`active_txn`,`apply_visible_gap`",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_plus_delimiter() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal+active_txn+apply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_windows_style_backslash_delimiter() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal\\active_txn\\apply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_assignment_and_ampersand_delimiters() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "backlog_blockers=wal&active_txn&apply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_mask_from_delimited_labels_accepts_percent_encoded_streams() {
        let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
            "wal%2Cpending-batch%7CACTIVE%20TXN%2Fapply_visible_gap",
        );

        assert_eq!(
            mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_WAL
                | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
                | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
        );
    }

    #[test]
    fn backlog_blocker_delimited_labels_from_mask_emits_canonical_order() {
        let mask = ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;

        assert_eq!(
            ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ","),
            "wal,active_txn,apply_visible_gap"
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, " | "),
            "wal | active_txn | apply_visible_gap"
        );
    }

    #[test]
    fn backlog_blocker_delimited_mask_roundtrip_is_stable() {
        let mask = ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP;
        let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ";");

        assert_eq!(
            ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
            mask
        );
    }

    #[test]
    fn backlog_blocker_delimited_mask_roundtrip_is_stable_with_colon_delimiter() {
        let mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;
        let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ":");

        assert_eq!(
            ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
            mask
        );
    }

    #[test]
    fn replication_watermarks_aggregate_multiple_backlog_blockers() {
        let mut e = Engine::with_batching(8, Duration::from_secs(999));
        let t0 = Instant::now();

        e.enqueue_set_text(7, "SET a=1", t0).unwrap();
        e.execute_text(8, "BEGIN").unwrap();

        let marks = e.replication_watermarks();
        assert!(marks.has_pending_batch_backlog);
        assert!(marks.has_active_txn_backlog);
        assert!(marks.has_backlog_blockers);
        assert_eq!(marks.backlog_blocker_count, 2);
        assert_eq!(
            marks.backlog_blocker_mask,
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
                | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        );
        assert_eq!(
            marks.backlog_blocker_count,
            marks.backlog_blocker_mask.count_ones() as u8
        );
        assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH));
        assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
        assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
        assert!(!marks.has_backlog_blocker(1 << 7));
        assert_eq!(
            marks.backlog_blockers().collect::<Vec<_>>(),
            vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
        );
        assert_eq!(
            marks.backlog_blocker_labels().collect::<Vec<_>>(),
            vec!["pending_batch", "active_txn"]
        );
        assert_eq!(
            marks.backlog_blocker_bits().collect::<Vec<_>>(),
            vec![
                ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH,
                ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            ]
        );
        assert_eq!(marks.max_replication_gap(), 0);
        assert_eq!(marks.total_backlog_items(), 2);
        assert!(!marks.is_fully_caught_up());
        assert!(!marks.quiescent_for_failover);
        assert!(!marks.follower_promotion_ready);
    }

    #[test]
    fn snapshot_export_tracks_last_applied_index() {
        let mut e = Engine::new_local();
        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

        let exported = e.export_snapshot_meta();
        let current = e.snapshot_meta();

        assert_eq!(exported.last_included_index, token.index);
        assert_eq!(current.last_included_index, token.index);
        assert_eq!(exported.snapshot_id, 1);
        assert_eq!(current.snapshot_id, 1);
    }

    #[test]
    fn install_snapshot_advances_visible_and_replication_watermarks() {
        let mut e = Engine::new_local();
        e.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 3,
            snapshot_id: 11,
        });

        let marks = e.replication_watermarks();
        assert_eq!(marks.term, 3);
        assert_eq!(marks.max_replication_gap(), 0);
        assert_eq!(marks.total_backlog_items(), 0);
        assert!(marks.is_fully_caught_up());
        assert_eq!(marks.commit_index, 7);
        assert_eq!(marks.applied_index, 7);
        assert_eq!(marks.visible_index, 7);
        assert_eq!(marks.commit_apply_gap, 0);
        assert_eq!(marks.apply_visible_gap, 0);
        assert_eq!(marks.snapshot_id, 11);

        let next = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
        assert_eq!(next.index, 8);
        assert_eq!(e.get("b"), Some("2"));
    }

    #[test]
    fn export_snapshot_meta_is_reflected_in_replication_watermarks() {
        let mut e = Engine::new_local();

        assert_eq!(e.replication_watermarks().snapshot_id, 0);

        e.export_snapshot_meta();
        assert_eq!(e.replication_watermarks().snapshot_id, 1);

        e.export_snapshot_meta();
        assert_eq!(e.replication_watermarks().snapshot_id, 2);
    }

    #[test]
    fn telemetry_snapshot_reflects_replication_lag_and_runtime_metrics() {
        let mut e = Engine::with_batching(8, Duration::from_secs(60));
        let t0 = Instant::now();

        e.enqueue_set_text(1, "SET a=1", t0).unwrap();

        let snapshot = e.telemetry_snapshot();

        assert_eq!(snapshot.role, Role::Leader);
        assert_eq!(snapshot.replication_lag.commit_index, 0);
        assert_eq!(snapshot.replication_lag.applied_index, 0);
        assert_eq!(snapshot.replication_lag.visible_index, 0);
        assert_eq!(snapshot.replication_lag.commit_apply_gap, 0);
        assert_eq!(snapshot.replication_lag.apply_visible_gap, 0);
        assert_eq!(snapshot.runtime_metrics.pending_batch_peak, 1);
        assert_eq!(snapshot.runtime_metrics.last_pending_batch_len, Some(1));
        assert_eq!(snapshot.runtime_metrics.commits_total, 0);
        assert_eq!(snapshot.snapshot_id, 0);
        assert_eq!(snapshot.wal_flushed_count, 0);
        assert_eq!(snapshot.wal_last_durable_txn_id, None);
        assert_eq!(snapshot.wal_buffered_count, 0);
        assert_eq!(snapshot.wal_unflushed_count, 0);
        assert_eq!(snapshot.pending_batch_len, 1);
        assert_eq!(snapshot.pending_batch_cap, 8);
        assert_eq!(snapshot.active_txn_count, 0);
        assert_eq!(snapshot.backlog_blocker_count, 1);
        assert!(snapshot.has_backlog_blockers());
        assert!(!snapshot.quiescent_for_failover);
        assert!(!snapshot.mutation_admission_saturated);
        assert!(snapshot.gpu_parity_fallbacks.is_empty());
    }

    #[test]
    fn status_snapshot_answers_snapshot_and_replication_health_questions() {
        let mut e = Engine::new_local();
        let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
        let exported = e.export_snapshot_meta();

        let status = e.status_snapshot();

        assert_eq!(status.role, Role::Leader);
        assert_eq!(status.term, 1);
        assert_eq!(status.snapshot.snapshot_id, exported.snapshot_id);
        assert_eq!(status.snapshot.last_included_index, token.index);
        assert_eq!(status.snapshot.last_included_term, 1);
        assert_eq!(status.snapshot.visible_index, token.index);
        assert_eq!(status.served_snapshot_frontier(), token.index);
        assert_eq!(status.replication_lag.commit_index, token.index);
        assert_eq!(status.replication_lag.applied_index, token.index);
        assert_eq!(status.replication_distance(), 0);
        assert!(status.why_routed_to_fallback_labels().is_empty());
        assert_eq!(status.latest_fallback_reason(), None);
        assert_eq!(status.backlog_blocker_labels(), Vec::<&'static str>::new());
        status.validate().unwrap();
    }

    #[test]
    fn status_snapshot_surfaces_active_fallback_reasons_and_rollups() {
        let mut e = Engine::new_local();
        e.mark_gpu_unavailable(0);
        e.set_gpu_runtime_saturated(true);

        e.execute_text(1, "SET a=1").unwrap();

        let status = e.status_snapshot();

        assert_eq!(
            status.latest_fallback_reason(),
            Some(FallbackReason::GpuUnavailable)
        );
        assert_eq!(
            status.why_routed_to_fallback_labels(),
            vec!["gpu_unavailable", "gpu_queue_saturated"]
        );
        assert!(status.fallback.is_actively_degraded());
        assert!(status.fallback.has_gpu_parity_fallbacks());
        assert_eq!(status.fallback.gpu_parity_fallback_total(), 1);
        assert_eq!(
            status.fallback.active_reasons,
            vec![
                ActiveFallbackReason::GpuUnavailable { gpu_ids: vec![0] },
                ActiveFallbackReason::GpuQueueSaturated,
            ]
        );
        status.validate().unwrap();
    }

    #[test]
    fn execute_mvcc_query_runs_visibility_filtered_scan_through_execution_layer() {
        let mut e = Engine::new_local();
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

    #[test]
    fn execute_mvcc_query_replays_deterministic_workload_fixture_for_point_lookup() {
        let mut e = Engine::new_local();
        for (txn_id, command) in include_str!("../../../tests/fixtures/mvcc-read-workload.txt")
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
    fn execute_mvcc_query_replays_deterministic_source_composition_workload_fixture() {
        let mut e = Engine::new_local();
        for (txn_id, command) in
            include_str!("../../../tests/fixtures/mvcc-source-composition-workload.txt")
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
    }

    #[test]
    fn execute_mvcc_query_supports_multi_key_lookup_fan_in_source() {
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
    fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_value_key_ref_prefixes_source(
    ) {
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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

    #[test]
    fn execute_mvcc_query_supports_multi_frame_provenance_filters_and_projection() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=profile:1").unwrap();
        e.execute_text(2, "SET acct:2=profile:2").unwrap();
        e.execute_text(3, "SET profile:1=team:alpha").unwrap();
        e.execute_text(4, "SET profile:2=team:beta").unwrap();
        e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
        e.execute_text(6, "SET team:beta:1=Bob").unwrap();
        e.execute_text(7, "SET team:beta:2=Bianca").unwrap();

        let query = e
            .execute_mvcc_query(&MvccReadQuery {
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
                        prefix: "acct:2".to_string(),
                    },
                    MvccReadFilter::ProvenanceValueEquals {
                        frame: MvccProvenanceFrame::TerminalInput,
                        expected: "team:beta".to_string(),
                    },
                ])),
                order: Some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::TargetKeyProvenanceValue {
                    frame: MvccProvenanceFrame::TerminalInput,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            query.rows,
            vec![
                MvccReadRow {
                    source_key: Some("acct:2".to_string()),
                    key: Some("team:beta:1".to_string()),
                    value: Some("team:beta".to_string()),
                },
                MvccReadRow {
                    source_key: Some("acct:2".to_string()),
                    key: Some("team:beta:2".to_string()),
                    value: Some("team:beta".to_string()),
                },
            ]
        );
    }

    #[test]
    fn execute_mvcc_query_preserves_multi_frame_provenance_identity_under_concat_distinct() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=profile:1").unwrap();
        e.execute_text(2, "SET profile:1=team:shared").unwrap();
        e.execute_text(3, "SET team:shared=member:1").unwrap();
        e.execute_text(4, "SET member:1=Alice").unwrap();

        let distinct = e
            .execute_mvcc_query(&MvccReadQuery {
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
                            keys: vec!["acct:1".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 1,
                                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                    ],
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::KeyPrefix("team:shared".to_string())),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceValue {
                    frame: MvccProvenanceFrame::ValueHop(2),
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            distinct.rows,
            vec![
                MvccReadRow {
                    source_key: Some("acct:1".to_string()),
                    key: Some("team:shared".to_string()),
                    value: Some("member:1".to_string()),
                },
                MvccReadRow {
                    source_key: Some("acct:1".to_string()),
                    key: Some("team:shared".to_string()),
                    value: None,
                },
            ]
        );
    }

    #[test]
    fn execute_mvcc_query_supports_provenance_path_summary_projection() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=profile:1").unwrap();
        e.execute_text(2, "SET acct:2=profile:2").unwrap();
        e.execute_text(3, "SET profile:1=team:alpha").unwrap();
        e.execute_text(4, "SET profile:2=team:beta").unwrap();
        e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
        e.execute_text(6, "SET team:alpha:2=Aria").unwrap();
        e.execute_text(7, "SET team:beta:1=Bob").unwrap();

        let query = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 7 },
                filter: None,
                order: Some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::TargetKeyProvenanceSummary {
                    summary: MvccProvenanceSummary::KeyValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            query.rows,
            vec![
                MvccReadRow {
                    source_key: Some("acct:1".to_string()),
                    key: Some("team:alpha:1".to_string()),
                    value: Some("acct:1=profile:1 -> profile:1=team:alpha".to_string(),),
                },
                MvccReadRow {
                    source_key: Some("acct:1".to_string()),
                    key: Some("team:alpha:2".to_string()),
                    value: Some("acct:1=profile:1 -> profile:1=team:alpha".to_string(),),
                },
                MvccReadRow {
                    source_key: Some("acct:2".to_string()),
                    key: Some("team:beta:1".to_string()),
                    value: Some("acct:2=profile:2 -> profile:2=team:beta".to_string(),),
                },
            ]
        );
    }

    #[test]
    fn execute_mvcc_query_provenance_summary_projection_keeps_non_join_shapes_stable() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET standalone:1=Loose").unwrap();

        let query = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 1 },
                filter: Some(MvccReadFilter::KeyPrefix("standalone:".to_string())),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceSummary {
                    summary: MvccProvenanceSummary::ValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            query.rows,
            vec![MvccReadRow {
                source_key: None,
                key: Some("standalone:1".to_string()),
                value: None,
            }]
        );
    }

    #[test]
    fn execute_mvcc_query_supports_frame_aware_provenance_ordering() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=profile:2").unwrap();
        e.execute_text(2, "SET acct:2=profile:1").unwrap();
        e.execute_text(3, "SET profile:1=team:alpha").unwrap();
        e.execute_text(4, "SET profile:2=team:beta").unwrap();
        e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
        e.execute_text(6, "SET team:alpha:2=Aria").unwrap();
        e.execute_text(7, "SET team:beta:1=Bob").unwrap();
        e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

        let query = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: None,
                order: Some(MvccReadOrder::ProvenanceValueAsc {
                    frame: MvccProvenanceFrame::TerminalInput,
                }),
                projection: MvccProjection::TargetKeyProvenanceValue {
                    frame: MvccProvenanceFrame::TerminalInput,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            query.rows,
            vec![
                MvccReadRow {
                    source_key: Some("acct:2".to_string()),
                    key: Some("team:alpha:1".to_string()),
                    value: Some("team:alpha".to_string()),
                },
                MvccReadRow {
                    source_key: Some("acct:2".to_string()),
                    key: Some("team:alpha:2".to_string()),
                    value: Some("team:alpha".to_string()),
                },
                MvccReadRow {
                    source_key: Some("acct:1".to_string()),
                    key: Some("team:beta:1".to_string()),
                    value: Some("team:beta".to_string()),
                },
                MvccReadRow {
                    source_key: Some("acct:1".to_string()),
                    key: Some("team:beta:2".to_string()),
                    value: Some("team:beta".to_string()),
                },
            ]
        );
    }

    #[test]
    fn execute_mvcc_query_supports_provenance_frame_bundle_controls() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:1=profile:2").unwrap();
        e.execute_text(2, "SET acct:2=profile:1").unwrap();
        e.execute_text(3, "SET profile:1=team:alpha").unwrap();
        e.execute_text(4, "SET profile:2=team:beta").unwrap();
        e.execute_text(5, "SET team:alpha=member:1").unwrap();
        e.execute_text(6, "SET team:beta=member:2").unwrap();
        e.execute_text(7, "SET member:1=Alice").unwrap();
        e.execute_text(8, "SET member:2=Bob").unwrap();

        let ordered = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: None,
                order: Some(MvccReadOrder::ProvenanceBundleValuePathAsc {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                }),
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::ValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            ordered.rows,
            vec![
                MvccReadRow {
                    source_key: Some("acct:2".to_string()),
                    key: Some("team:alpha".to_string()),
                    value: Some("profile:1 -> team:alpha".to_string()),
                },
                MvccReadRow {
                    source_key: Some("acct:1".to_string()),
                    key: Some("team:beta".to_string()),
                    value: Some("profile:2 -> team:beta".to_string()),
                },
            ]
        );

        let filtered = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundleValueEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    expected: "team:beta".to_string(),
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::ValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            filtered.rows,
            vec![MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("profile:2 -> team:beta -> member:2".to_string()),
            }]
        );

        let key_ordered = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundleKeyPrefix {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    prefix: "profile:2".to_string(),
                }),
                order: Some(MvccReadOrder::ProvenanceBundleKeyPathAsc {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                }),
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            key_ordered.rows,
            vec![MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("acct:1 -> profile:2 -> team:beta".to_string()),
            }]
        );

        let key_exact = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundleKeyEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    expected: "profile:2".to_string(),
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            key_exact.rows,
            vec![MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("acct:1 -> profile:2".to_string()),
            }]
        );

        let key_value_exact = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundleKeyValueEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    key: "profile:2".to_string(),
                    value: "team:beta".to_string(),
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            key_value_exact.rows,
            vec![MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some(
                    "acct:1=profile:2 -> profile:2=team:beta -> team:beta=member:2".to_string(),
                ),
            }]
        );

        let cross_frame_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundleKeyValueEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    key: "acct:1".to_string(),
                    value: "team:beta".to_string(),
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert!(cross_frame_miss.rows.is_empty());

        let key_path_exact = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["acct:1".to_string(), "profile:2".to_string()],
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            key_path_exact.rows,
            vec![MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("acct:1 -> profile:2".to_string()),
            }]
        );

        let value_path_exact = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::ValuePath,
                    expected: vec![
                        "profile:2".to_string(),
                        "team:beta".to_string(),
                        "member:2".to_string(),
                    ],
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::ValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            value_path_exact.rows,
            vec![MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("profile:2 -> team:beta -> member:2".to_string()),
            }]
        );

        let ordered_path_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["profile:2".to_string(), "acct:1".to_string()],
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert!(ordered_path_miss.rows.is_empty());

        let key_path_contains = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathContains {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["profile:2".to_string(), "team:beta".to_string()],
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            key_path_contains.rows,
            vec![MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("acct:1 -> profile:2 -> team:beta".to_string()),
            }]
        );

        let truncated_bundle_contains_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathContains {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::ValuePath,
                    expected: vec!["team:beta".to_string(), "member:2".to_string()],
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::ValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert!(truncated_bundle_contains_miss.rows.is_empty());

        let ordered_subpath_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 8 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathContains {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyValuePath,
                    expected: vec![
                        "profile:2=team:beta".to_string(),
                        "acct:1=profile:2".to_string(),
                    ],
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert!(ordered_subpath_miss.rows.is_empty());
    }

    #[test]
    fn execute_mvcc_query_supports_quantified_and_positional_provenance_bundle_filters() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
        e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
        e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
        e.execute_text(4, "SET profile:solo=team:solo").unwrap();

        let key_count = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundleKeyCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    expected: "acct:loop".to_string(),
                    min_count: 2,
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            key_count.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("acct:loop -> profile:loop -> acct:loop".to_string()),
            }]
        );

        let value_count = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundleValueCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    expected: "profile:loop".to_string(),
                    min_count: 2,
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::ValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            value_count.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("profile:loop -> acct:loop -> profile:loop".to_string()),
            }]
        );

        let key_value_count = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    key: "acct:loop".to_string(),
                    value: "profile:loop".to_string(),
                    min_count: 2,
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            key_value_count.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some(
                    "acct:loop=profile:loop -> profile:loop=acct:loop -> acct:loop=profile:loop"
                        .to_string(),
                ),
            }]
        );

        let position_match = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathSegmentEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::ValuePath,
                    index: 1,
                    expected: "acct:loop".to_string(),
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::ValuePath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            position_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("profile:loop -> acct:loop".to_string()),
            }]
        );

        let position_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathSegmentEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::ValuePath,
                    index: 2,
                    expected: "profile:loop".to_string(),
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(position_miss.rows.is_empty());

        let threshold_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundleKeyCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    expected: "acct:loop".to_string(),
                    min_count: 3,
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(threshold_miss.rows.is_empty());
    }

    #[test]
    fn execute_mvcc_query_supports_repeated_provenance_bundle_subpath_filters() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
        e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
        e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
        e.execute_text(4, "SET profile:solo=team:solo").unwrap();

        let repeated_subpath = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::Concat {
                    sources: vec![
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:loop".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 3,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:solo".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 1,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                    ],
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                    min_count: 2,
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            repeated_subpath.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
            }]
        );

        let truncated_bundle_repeat_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                    min_count: 2,
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(truncated_bundle_repeat_miss.rows.is_empty());

        let impossible_repeat_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["profile:loop".to_string(), "acct:loop".to_string()],
                    min_count: 2,
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(impossible_repeat_miss.rows.is_empty());
    }

    #[test]
    fn execute_mvcc_query_supports_relative_provenance_bundle_distance_filters() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
        e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
        e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
        e.execute_text(4, "SET profile:solo=team:solo").unwrap();

        let distance_match = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::Concat {
                    sources: vec![
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:loop".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 3,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:solo".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 1,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                    ],
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathPairAtDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left: "profile:loop".to_string(),
                    right: "profile:loop".to_string(),
                    distance: 2,
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            distance_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
            }]
        );

        let truncated_distance_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathPairAtDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left: "profile:loop".to_string(),
                    right: "profile:loop".to_string(),
                    distance: 2,
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(truncated_distance_miss.rows.is_empty());

        let mismatch_distance_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathPairAtDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left: "profile:loop".to_string(),
                    right: "profile:loop".to_string(),
                    distance: 1,
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(mismatch_distance_miss.rows.is_empty());
    }

    #[test]
    fn execute_mvcc_query_supports_provenance_bundle_suffix_filters() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
        e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
        e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
        e.execute_text(4, "SET profile:solo=team:solo").unwrap();

        let suffix_match = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::Concat {
                    sources: vec![
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:loop".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 3,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:solo".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 1,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                    ],
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec![
                        "profile:loop".to_string(),
                        "acct:loop".to_string(),
                        "profile:loop".to_string(),
                    ],
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            suffix_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
            }]
        );

        let truncated_suffix_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec![
                        "profile:loop".to_string(),
                        "acct:loop".to_string(),
                        "profile:loop".to_string(),
                    ],
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(truncated_suffix_miss.rows.is_empty());

        let mismatch_suffix_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["profile:loop".to_string(), "profile:loop".to_string()],
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(mismatch_suffix_miss.rows.is_empty());
    }

    #[test]
    fn execute_mvcc_query_supports_whole_bundle_cardinality_filters() {
        let mut e = Engine::new_local();
        e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
        e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
        e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
        e.execute_text(4, "SET profile:solo=team:solo").unwrap();

        let full_path_len = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::Concat {
                    sources: vec![
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:loop".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 2,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                        MvccReadSource::FollowValueChain {
                            keys: vec!["acct:solo".to_string()],
                            plan: MvccValueChainPlan {
                                value_key_hops: 1,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                            provenance: MvccSourceProvenance::Seed,
                        },
                    ],
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    expected_len: 3,
                }),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            full_path_len.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("acct:loop -> profile:loop -> acct:loop".to_string()),
            }]
        );

        let truncated_bundle_len = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    expected_len: 2,
                }),
                order: Some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

        assert_eq!(
            truncated_bundle_len.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("acct:loop -> profile:loop".to_string()),
            }]
        );

        let len_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    expected_len: 4,
                }),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert!(len_miss.rows.is_empty());
    }

    #[test]
    fn execute_mvcc_query_sorts_missing_provenance_frames_deterministically() {
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
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
        let mut e = Engine::new_local();
        let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
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
        assert_eq!(e.get("a"), Some("1"));
    }

    #[test]
    fn installing_higher_index_lower_term_snapshot_is_a_status_no_op() {
        let mut e = Engine::new_local();
        let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

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
        assert_eq!(e.get("a"), Some("1"));
    }

    #[test]
    fn installing_advanced_snapshot_replaces_snapshot_identity_exactly() {
        let mut e = Engine::new_local();
        let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

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
}
