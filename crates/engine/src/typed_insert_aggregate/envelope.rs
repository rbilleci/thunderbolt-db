//! Exact outer canonical-WAL buffers for one encoded codec-5 aggregate.

use super::*;
use crate::EngineError;

/// Exact caller-owned WAL buffers after capacity admission but before they carry record authority.
///
/// This move-only state owns already encoded aggregate bodies, so an outer-envelope failure drops
/// the whole candidate rather than exposing a partial record.
pub(crate) struct ReservedTypedInsertCanonicalEnvelope {
    bodies: EncodedTypedInsertAggregateBodies,
    physical: gpu_db_wal::CanonicalPhysicalRange,
    header: gpu_db_wal::CanonicalPreApplyHeader,
    outcome: gpu_db_wal::CanonicalOutcome,
    packed_payload: Box<[u8]>,
    serialized_record: Box<[u8]>,
}

/// Fully root-closed codec-5 aggregate plus its exact immutable outer WAL bytes.
pub(crate) struct EncodedTypedInsertCanonicalEnvelope {
    bodies: EncodedTypedInsertAggregateBodies,
    physical: gpu_db_wal::CanonicalPhysicalRange,
    header: gpu_db_wal::CanonicalPreApplyHeader,
    outcome: gpu_db_wal::CanonicalOutcome,
    encoding: gpu_db_wal::ExactCanonicalRecordEncoding,
    packed_payload: Box<[u8]>,
    serialized_record: Box<[u8]>,
}

impl ReservedTypedInsertCanonicalEnvelope {
    pub(crate) fn packed_payload_len(&self) -> usize {
        self.packed_payload.len()
    }

    pub(crate) fn serialized_record_len(&self) -> usize {
        self.serialized_record.len()
    }
}

impl EncodedTypedInsertCanonicalEnvelope {
    pub(crate) fn bodies(&self) -> &EncodedTypedInsertAggregateBodies {
        &self.bodies
    }

    pub(crate) fn physical(&self) -> gpu_db_wal::CanonicalPhysicalRange {
        self.physical
    }

    pub(crate) fn header(&self) -> &gpu_db_wal::CanonicalPreApplyHeader {
        &self.header
    }

    pub(crate) fn outcome(&self) -> &gpu_db_wal::CanonicalOutcome {
        &self.outcome
    }

    pub(crate) fn encoding(&self) -> gpu_db_wal::ExactCanonicalRecordEncoding {
        self.encoding
    }

    pub(crate) fn packed_payload(&self) -> &[u8] {
        &self.packed_payload
    }

    pub(crate) fn serialized_record(&self) -> &[u8] {
        &self.serialized_record
    }
}

/// Allocate the two exact outer WAL buffers measured by the same codec-5 layout.
///
/// The future terminal carrier is the only production caller and must hold its all-resource
/// capacity lease before invoking this constructor.
pub(crate) fn reserve_typed_insert_canonical_envelope(
    bodies: EncodedTypedInsertAggregateBodies,
    physical: gpu_db_wal::CanonicalPhysicalRange,
    header: gpu_db_wal::CanonicalPreApplyHeader,
    outcome: gpu_db_wal::CanonicalOutcome,
) -> Result<ReservedTypedInsertCanonicalEnvelope, EngineError> {
    validate_outer_binding(&bodies, physical, &header, &outcome)?;
    let refs = bodies.canonical_fragment_refs();
    let body_lengths = bodies.layout().live_fragment_body_bytes();
    let measured =
        gpu_db_wal::measure_canonical_exact_buffers(physical, &header, body_lengths, &outcome)?;
    if measured != bodies.layout().wal || refs.as_slice().len() != body_lengths.len() {
        return Err(error("outer WAL measurement differs from codec-5 layout"));
    }
    let packed_len = usize::try_from(measured.packed_record_bytes)
        .map_err(|_| error("packed WAL record exceeds addressable host memory"))?;
    let serialized_len = usize::try_from(measured.serialized_record_bytes)
        .map_err(|_| error("serialized WAL record exceeds addressable host memory"))?;
    Ok(ReservedTypedInsertCanonicalEnvelope {
        bodies,
        physical,
        header,
        outcome,
        packed_payload: zeroed_box(packed_len),
        serialized_record: zeroed_box(serialized_len),
    })
}

/// Fill both reserved outer buffers and consume them into immutable record authority.
pub(crate) fn encode_reserved_typed_insert_canonical_envelope(
    mut reserved: ReservedTypedInsertCanonicalEnvelope,
) -> Result<EncodedTypedInsertCanonicalEnvelope, EngineError> {
    validate_outer_binding(
        &reserved.bodies,
        reserved.physical,
        &reserved.header,
        &reserved.outcome,
    )?;
    let refs = reserved.bodies.canonical_fragment_refs();
    let encoding = gpu_db_wal::encode_canonical_record_exact_from_borrowed(
        reserved.header.stable_transaction_id,
        reserved.physical,
        &reserved.header,
        refs.as_slice(),
        &reserved.outcome,
        &mut reserved.packed_payload,
        &mut reserved.serialized_record,
    )?;
    if encoding.footprint != reserved.bodies.layout().wal {
        return Err(error(
            "encoded outer WAL footprint drifted from reservation",
        ));
    }
    Ok(EncodedTypedInsertCanonicalEnvelope {
        bodies: reserved.bodies,
        physical: reserved.physical,
        header: reserved.header,
        outcome: reserved.outcome,
        encoding,
        packed_payload: reserved.packed_payload,
        serialized_record: reserved.serialized_record,
    })
}

fn validate_outer_binding(
    bodies: &EncodedTypedInsertAggregateBodies,
    physical: gpu_db_wal::CanonicalPhysicalRange,
    header: &gpu_db_wal::CanonicalPreApplyHeader,
    outcome: &gpu_db_wal::CanonicalOutcome,
) -> Result<(), EngineError> {
    let measure = &bodies.layout().measure;
    let status = bodies.status();
    let expected_affected_rows = if measure.flags & AGGREGATE_FLAG_AUTOCOMMIT != 0 {
        // Autocommit carries exactly one INSERT statement. Its terminal retry/status outcome is
        // the rows that statement inserted, not the cardinality of its statement list.
        measure.original_inserted_row_count
    } else {
        // An explicit aggregate can compose several statement outcomes into fewer surviving
        // physical changes, so its enclosing terminal outcome binds the final transition count.
        measure.final_row_transition_count
    };
    let outcome_rows_are_valid = match outcome.kind {
        gpu_db_wal::CanonicalOutcomeKind::CommitSuccess => {
            outcome.affected_rows == expected_affected_rows
        }
        gpu_db_wal::CanonicalOutcomeKind::CommitNoOp => {
            outcome.affected_rows == 0 && measure.final_row_transition_count == 0
        }
        gpu_db_wal::CanonicalOutcomeKind::AbortError => outcome.affected_rows == 0,
    };
    if physical.log_epoch == 0
        || physical.segment_id == 0
        || header.stable_transaction_id != measure.stable_transaction_id
        || header.stable_transaction_id != status.txn_id
        || header.identity.database_id != status.database_id
        || header.identity.timeline_id != status.timeline_id
        || header.request_digest != status.request_digest
        || header.isolation as u8 != status.isolation
        || header.flags != measure.outer_flags
        || header.operation_count != bodies.layout().fragment_count
        || header.table_block_count != measure.table_block_count
        || header.allocator_high_water != measure.allocator_high_water
        || outcome.target_digest != bodies.aggregate_root()
        || outcome.returning_digest != status.response_root
        || !outcome_rows_are_valid
    {
        return Err(error(
            "outer header, outcome, status, or aggregate identity is inconsistent",
        ));
    }
    Ok(())
}

fn zeroed_box(len: usize) -> Box<[u8]> {
    let mut bytes = Box::<[u8]>::new_uninit_slice(len);
    for byte in bytes.iter_mut() {
        byte.write(0);
    }
    // SAFETY: every u8 slot is initialized above.
    unsafe { bytes.assume_init() }
}

fn error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate canonical envelope: {message}"
    ))
}
