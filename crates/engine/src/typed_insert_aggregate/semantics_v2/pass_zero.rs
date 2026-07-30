//! Semantics-v2 pass-zero facade.
//!
//! The substantial S1--S7 streaming proof lives in `pass_zero/s7.rs`; keeping this facade small
//! makes the allocation-phase owner distinct from the fixed-directory grammar it measures.

#[path = "pass_zero/s7.rs"]
mod s7;

pub(crate) use s7::SemanticsV2PassZero;

/// Every variable identity field from the fixed S7 header.  Pass zero has already checked these
/// against the canonical outer/S1 closure; carrying the scalar header identity forward lets the
/// retained graph recheck roots and, only after full witness validation, reproduce the frozen
/// logical header without retaining raw aggregate bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2S7HeaderIdentity {
    pub(super) total_bytes: u64,
    pub(super) root_descriptor_version: u16,
    pub(super) catalog_before_epoch: u64,
    pub(super) catalog_after_epoch: u64,
    pub(super) catalog_before_digest: [u8; 32],
    pub(super) catalog_after_digest: [u8; 32],
    pub(super) initial_database_root: [u8; 32],
    pub(super) final_database_root: [u8; 32],
    pub(super) initial_overlay_root: [u8; 32],
    pub(super) final_overlay_root: [u8; 32],
    pub(super) root_descriptor: [u8; 32],
    pub(super) payload_digest: [u8; 32],
}

/// Exact structural terms which the retained owner receives from pass zero.  This is deliberately
/// a value-only handoff: it cannot carry fragment bytes or authorize a model before the strict
/// S2 source measurement supplies its own retained-owner terms.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2StructuralMeasure {
    pub(super) s1_statement_count: u32,
    pub(super) s2_record_count: u32,
    pub(super) s4_disposition_count: u64,
    pub(super) s5_effect_count: u32,
    pub(super) s6_outcome_count: u32,
    pub(super) s7_directory_counts: [u32; 12],
    pub(super) s7_header: SemanticsV2S7HeaderIdentity,
    /// Exact retained owners below this layer, measured by their strict source decoders.  The
    /// retained graph must add only its own typed directory vectors; wire widths and arenas are
    /// not a persistent allocation ABI.
    pub(super) s2_decoded_persistent_bytes: u64,
    pub(super) s2_decoded_persistent_slots: u64,
    pub(super) image_decoded_persistent_bytes: u64,
    pub(super) image_decoded_persistent_slots: u64,
    pub(super) raw_maximum_scratch_bytes: u64,
    pub(super) raw_maximum_scratch_slots: u64,
}

impl SemanticsV2PassZero {
    /// The retained decoder may use only this sealed output for aggregate/S7 sizing.  The
    /// subsequently-added S2 source measurement contributes the strict decoded-record owners
    /// and its concurrently-live scratch; no retained pass may derive those terms from a second
    /// traversal of attacker-declared counts.
    pub(super) fn into_structural_measure(self) -> SemanticsV2StructuralMeasure {
        SemanticsV2StructuralMeasure {
            s1_statement_count: self.s1_statement_count,
            s2_record_count: self.s2_record_count,
            s4_disposition_count: self.s4_disposition_count,
            s5_effect_count: self.s5_effect_count,
            s6_outcome_count: self.s6_outcome_count,
            s7_directory_counts: self.s7_directory_counts,
            s7_header: self.s7_header,
            s2_decoded_persistent_bytes: self.s2_decoded_persistent_bytes,
            s2_decoded_persistent_slots: self.s2_decoded_persistent_slots,
            image_decoded_persistent_bytes: self.image_decoded_persistent_bytes,
            image_decoded_persistent_slots: self.image_decoded_persistent_slots,
            raw_maximum_scratch_bytes: self.raw_maximum_scratch_bytes,
            raw_maximum_scratch_slots: self.raw_maximum_scratch_slots,
        }
    }
}

pub(super) fn measure(
    framing: &super::super::codec::DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
) -> Result<SemanticsV2PassZero, crate::EngineError> {
    s7::measure(framing, outer)
}

#[cfg(test)]
pub(super) fn validate_dependency_token_digests_for_test(
    framing: &super::super::codec::DecodedAggregateFraming<'_>,
    raw: &[u8; 224],
) -> Result<(), crate::EngineError> {
    s7::validate_dependency_token_digests_for_test(framing, raw)
}
