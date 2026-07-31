//! Allocation-free S8 retained-response proof.
//!
//! S8 is deliberately separate from the frozen S7 proof: it can bind S6/S7/S4/S2 facts, but
//! cannot manufacture an eligibility decision.  In particular, a locally coherent omitted
//! eligible artifact remains a later retention-authority question.

#[path = "s8/artifacts.rs"]
mod artifacts;
#[path = "s8/source.rs"]
mod source;

use super::super::super::codec::DecodedAggregateFraming;
use super::s7::S7Measure;
#[cfg(test)]
use crate::typed_insert_aggregate::AGGREGATE_FLAG_RETAINED_RESPONSE;
use crate::EngineError;

pub(in crate::typed_insert_aggregate::semantics_v2) use artifacts::S8PassZeroMeasure;

#[cfg(test)]
pub(super) fn empty_measure_for_test() -> S8PassZeroMeasure {
    S8PassZeroMeasure {
        identity: artifacts::S8ResponseIdentity {
            present: false,
            aggregate_flags: 0,
            stable_transaction_id: 0,
            request_digest: [0; 32],
            s6_section_root: [0; 32],
            s7_section_root: [0; 32],
            s8_section_root: [0; 32],
            response_root: [0; 32],
            status_artifact_count: 0,
            retention_deadline: 0,
            total_bytes: 0,
            artifact_count: 0,
            selection_count: 0,
            image_arena_bytes: 0,
            payload_digest: [0; 32],
        },
        image_persistent_bytes: 0,
        image_persistent_slots: 0,
        maximum_scratch_bytes: 0,
        maximum_scratch_slots: 0,
    }
}

#[cfg(test)]
pub(super) fn nonempty_measure_for_test() -> S8PassZeroMeasure {
    S8PassZeroMeasure {
        identity: artifacts::S8ResponseIdentity {
            present: true,
            aggregate_flags: AGGREGATE_FLAG_RETAINED_RESPONSE,
            stable_transaction_id: 17,
            request_digest: [1; 32],
            s6_section_root: [2; 32],
            s7_section_root: [3; 32],
            s8_section_root: [4; 32],
            response_root: [5; 32],
            status_artifact_count: 2,
            retention_deadline: 900,
            total_bytes: 1024,
            artifact_count: 2,
            selection_count: 2,
            image_arena_bytes: 512,
            payload_digest: [6; 32],
        },
        image_persistent_bytes: 3072,
        image_persistent_slots: 11,
        maximum_scratch_bytes: 1536,
        maximum_scratch_slots: 5,
    }
}

pub(super) fn measure(
    framing: &DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    s7: &S7Measure,
    terminal_abort: Option<u32>,
    aggregate_retained_response: bool,
) -> Result<S8PassZeroMeasure, EngineError> {
    artifacts::measure(
        framing,
        outer,
        &s7.directory_counts,
        terminal_abort,
        aggregate_retained_response,
    )
}

#[cfg(test)]
pub(super) fn measure_with_known_s7_counts_for_test(
    framing: &DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    s7_counts: &[u32; 12],
) -> Result<S8PassZeroMeasure, EngineError> {
    artifacts::measure(framing, outer, s7_counts, None, true)
}
