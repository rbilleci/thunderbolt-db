//! Strict post-reservation retained fill.
//!
//! This child of the retained boundary is intentionally the only code allowed to consume the
//! opaque exact graph owner.  It rereads source-backed S2 records and S7 images without exposing
//! aggregate bytes, then seals the full typed graph into the inert quarantine state.

#[path = "fill/fixed.rs"]
mod fixed;
#[path = "fill/source.rs"]
mod source;

#[path = "fill/s2.rs"]
mod s2;
#[path = "fill/s7.rs"]
mod s7;

use super::reservation::{
    ReservedSemanticsV2GraphOwner, RetainedSemanticsV2Reservation, RetainedSemanticsV2SourceMeasure,
};
use super::{quarantine_after_strict_fill, QuarantinedSemanticsV2, SemanticsV2BoundIdentity};
use crate::typed_insert_aggregate::codec::DecodedAggregateFraming;
use crate::typed_insert_aggregate::semantics_v2::pass_zero::SemanticsV2PassZero;
use crate::EngineError;

#[derive(Clone, Copy, Default)]
pub(super) struct ObservedSourceMeasure {
    s2_persistent_bytes: u64,
    s2_persistent_slots: u64,
    image_persistent_bytes: u64,
    image_persistent_slots: u64,
    maximum_scratch_bytes: u64,
    maximum_scratch_slots: u64,
}

impl ObservedSourceMeasure {
    fn checked_add_s2(
        &mut self,
        persistent_bytes: u64,
        persistent_slots: u64,
        scratch_bytes: u64,
        scratch_slots: u64,
    ) -> Result<(), EngineError> {
        self.s2_persistent_bytes = self
            .s2_persistent_bytes
            .checked_add(persistent_bytes)
            .ok_or_else(|| source::fill_error("decoded S2 persistent bytes overflow"))?;
        self.s2_persistent_slots = self
            .s2_persistent_slots
            .checked_add(persistent_slots)
            .ok_or_else(|| source::fill_error("decoded S2 persistent slots overflow"))?;
        self.maximum_scratch_bytes = self.maximum_scratch_bytes.max(scratch_bytes);
        self.maximum_scratch_slots = self.maximum_scratch_slots.max(scratch_slots);
        Ok(())
    }

    fn checked_add_image(
        &mut self,
        persistent_bytes: u64,
        persistent_slots: u64,
        scratch_bytes: u64,
        scratch_slots: u64,
    ) -> Result<(), EngineError> {
        self.image_persistent_bytes = self
            .image_persistent_bytes
            .checked_add(persistent_bytes)
            .ok_or_else(|| source::fill_error("decoded image persistent bytes overflow"))?;
        self.image_persistent_slots = self
            .image_persistent_slots
            .checked_add(persistent_slots)
            .ok_or_else(|| source::fill_error("decoded image persistent slots overflow"))?;
        self.maximum_scratch_bytes = self.maximum_scratch_bytes.max(scratch_bytes);
        self.maximum_scratch_slots = self.maximum_scratch_slots.max(scratch_slots);
        Ok(())
    }

    fn require_exact(self, expected: RetainedSemanticsV2SourceMeasure) -> Result<(), EngineError> {
        if self.s2_persistent_bytes != expected.s2_persistent_bytes
            || self.s2_persistent_slots != expected.s2_persistent_slots
            || self.image_persistent_bytes != expected.image_persistent_bytes
            || self.image_persistent_slots != expected.image_persistent_slots
            || self.maximum_scratch_bytes != expected.maximum_scratch_bytes
            || self.maximum_scratch_slots != expected.maximum_scratch_slots
        {
            return Err(source::fill_error(
                "strict source/image fill drifted from the pass-zero reservation measure",
            ));
        }
        Ok(())
    }
}

/// Consume a successful raw proof only after reserving every direct and nested retained owner.
/// This has no transition to WAL, recovery, execution, GPU, result, or publication state.
pub(super) fn fill_after_pass_zero(
    framing: &DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    pass_zero: SemanticsV2PassZero,
) -> Result<QuarantinedSemanticsV2, EngineError> {
    let reservation =
        RetainedSemanticsV2Reservation::from_structural(pass_zero.into_structural_measure())?;
    let mut owner = reservation.reserve()?;
    let expected_source = owner.source_measure();
    let mut observed = ObservedSourceMeasure::default();

    fixed::fill_fixed_sections(framing, owner.graph_mut())?;
    s2::fill_s2_records(framing, owner.graph_mut(), &mut observed)?;
    s7::fill_s7(framing, owner.graph_mut(), &mut observed)?;
    observed.require_exact(expected_source)?;
    validate_filled_bindings(&owner)?;

    let identity = SemanticsV2BoundIdentity {
        database_id: outer.identity.database_id,
        catalog_epoch: outer.catalog_before_epoch,
        catalog_digest: outer.catalog_before_digest,
        stable_transaction_id: outer.stable_transaction_id,
        autocommit: framing.header_scalars().flags
            & crate::typed_insert_aggregate::AGGREGATE_FLAG_AUTOCOMMIT
            != 0,
        commit_sequence: outer.commit_seq,
        initial_database_root: owner.graph_mut().header.initial_database_root,
    };
    let graph = owner.into_exact_graph()?;
    Ok(quarantine_after_strict_fill(identity, graph))
}

fn validate_filled_bindings(owner: &ReservedSemanticsV2GraphOwner) -> Result<(), EngineError> {
    // S7's complete nonlocal grammar was proved before any allocation.  This local replay binds
    // the move-only S2/image owners just constructed from their independently remeasured source.
    // It does not create a second raw or reencoding authority.
    let graph = owner.graph();
    if graph.statements.len() != graph.records.len()
        || graph.statements.len() != graph.outcomes.len()
        || graph.tables.len() != graph.images.len()
    {
        return Err(source::fill_error(
            "strict retained source owners have divergent lengths",
        ));
    }
    for (ordinal, (statement, record)) in graph.statements.iter().zip(&graph.records).enumerate() {
        let facts = record.facts();
        if statement.statement_ordinal != ordinal as u32
            || statement.input_row_count != facts.row_count
            || statement.typed_statement_digest != facts.typed_statement_digest
            || statement.record_bytes == 0
            || statement.record_digest == [0; 32]
        {
            return Err(source::fill_error(
                "decoded S2 record does not bind its S1 statement",
            ));
        }
    }
    for (ordinal, (table, image)) in graph.tables.iter().zip(&graph.images).enumerate() {
        let facts = image.facts();
        if table.table_ref != ordinal as u32
            || table.image_ref != ordinal as u32
            || facts.rows != table.transition_count
            || facts.columns != table.catalog_column_count
            || facts.layout_digest != table.image_layout_digest
        {
            return Err(source::fill_error(
                "decoded S7 image does not bind its table descriptor",
            ));
        }
    }
    for (ordinal, resolution) in graph.resolutions.iter().enumerate() {
        let statement = graph
            .statements
            .get(ordinal)
            .ok_or_else(|| source::fill_error("S7 resolution has no S1 statement"))?;
        let record = graph
            .records
            .get(ordinal)
            .ok_or_else(|| source::fill_error("S7 resolution has no decoded S2 record"))?;
        let table = graph
            .tables
            .get(usize::try_from(resolution.table_ref).map_err(|_| {
                source::fill_error("S7 resolution table reference exceeds host addressability")
            })?)
            .ok_or_else(|| source::fill_error("S7 resolution has no retained target table"))?;
        let facts = record.facts();
        let bound = resolution.statement_ordinal == ordinal as u32
            && resolution.record_ref == ordinal as u32
            && resolution.outcome_ref == ordinal as u32
            && resolution.record_bytes == statement.record_bytes
            && resolution.record_digest == statement.record_digest
            && resolution.request_digest == statement.request_digest
            && resolution.typed_statement_digest == statement.typed_statement_digest
            && resolution.overlay_before == statement.overlay_before
            && resolution.overlay_after == statement.overlay_after
            && resolution.input_row_count == facts.row_count
            && resolution.returning_digest == facts.returning.digest
            && table.display_oid == facts.target.oid;
        if !bound {
            return Err(source::fill_error(
                "S7 resolution does not bind the retained S1/S2 owner",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn fail_copy_at_for_test<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    source::fail_copy_at_for_test(attempt, operation)
}
