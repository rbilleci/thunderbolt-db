//! Inert provisional semantics-1 S1--S6 source materialization for the future aggregate semantic
//! IR.
//!
//! This leaf consumes no WAL envelope, creates no replay transaction, and has no recovery,
//! apply, plan, or publication caller.  It first reuses the allocation-free S1/S4 validation
//! traversal, then reads the four direct sections again to retain exactly one joined source per
//! statement and one compact disposition per original INSERT row. Its parent is not normative
//! semantics-v2 dispatch; the accepted physical aggregate remains semantics 1. Raw S2/S3 bytes
//! exist only in one bounded scratch owner while their strict current
//! codec is decoded.

use super::super::TypedInsertAggregateMeasure;
use super::*;
use crate::engine_canonical_operation::{
    decode_current_non_insert_canonical_operation, CurrentNonInsertCanonicalOperation,
};
use crate::typed_insert_aggregate::AGGREGATE_FLAG_AUTOCOMMIT;
use crate::typed_insert_batch::{
    decode_canonical_typed_insert_record, DecodedTypedInsertRecord,
    CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES,
};
use std::ops::Range;

#[path = "source_materialization/sequence_outcomes.rs"]
mod sequence_outcomes;

const TYPED_INSERT_SOURCES_SECTION: usize = 1;
const NON_INSERT_SOURCES_SECTION: usize = 2;
const TYPED_INSERT_SOURCES_TAG: u16 = 2;
const NON_INSERT_SOURCES_TAG: u16 = 3;
const S3_PREFIX_BYTES: u64 = 48;
const MAX_SOURCE_BODY_BYTES: usize = 16 * 1024 * 1024;

/// The incomplete, move-only S1--S6 semantic source owner.
///
/// It deliberately stops before S7 final overlay and S8 responses.  S5 sequence closure and S6
/// per-statement outcomes are retained only as exact, inert source facts; this is not a replay
/// transaction and has no conversion into recovery or a device INSERT plan.
#[allow(dead_code)] // Inert source-materialization seam; later replay IR owns its adoption.
pub(crate) struct TypedInsertAggregateSourceDraft {
    statements: Box<[SourceStatement]>,
    dispositions: Box<[CompactRowDisposition]>,
    /// Exactly one compact S5 owner.  Published entries retain their decoded fixed-width
    /// reference; private entries retain only their binding digest while S2 owns private state.
    sequence_effects: Box<[CompactSequenceEffect]>,
}

/// S1 scalars fused with exactly one moved S2 or S3 semantic source.
struct SourceStatement {
    statement_ordinal: u32,
    family_ordinal: u32,
    request_digest: gpu_db_wal::CanonicalDigest,
    statement_digest: gpu_db_wal::CanonicalDigest,
    overlay_before: gpu_db_wal::CanonicalDigest,
    overlay_after: gpu_db_wal::CanonicalDigest,
    input_rows: Range<u32>,
    sequence_effects: Range<u32>,
    outcome: Option<CompactStatementOutcome>,
    source: SourceStatementKind,
}

/// The one semantic source belonging to a statement-directory entry.
///
/// Neither variant retains S2/S3 wire bytes after strict decoding.  Both source models are
/// already non-Clone, so this enum and its enclosing draft are move-only by construction.
enum SourceStatementKind {
    TypedInsert(DecodedTypedInsertRecord),
    CanonicalOperation(CurrentNonInsertCanonicalOperation),
}

/// S4 after statement/source/digest order has been validated.  Its enclosing statement range
/// determines the original statement and source ordinals, so this compact owner omits both as
/// well as the duplicate statement digest.
struct CompactRowDisposition {
    stable_row_id: u64,
    kind: RowDispositionKind,
    block_ref: u32,
    transition_ref: u32,
}

/// S5 facts compacted after their exact body has been decoded and bound to S2/S3.
///
/// The enclosing statement range supplies statement and effect ordinals, so no duplicate ordinal
/// fields are retained.  This is deliberately not a sequence-state carrier.
struct CompactSequenceEffect {
    disposition_ordinal: u32,
    kind: CompactSequenceEffectKind,
}

enum CompactSequenceEffectKind {
    Published(crate::BinarySequenceValueReference),
    Private {
        input_digest: gpu_db_wal::CanonicalDigest,
        /// S7 must later prove the final image/value closure for a surviving row. For a canceled
        /// or suppressed row, S5 already requires this fact to be true.
        final_value_overwritten: bool,
    },
}

/// S6's one decoded result for an existing S1 source statement.
struct CompactStatementOutcome {
    semantic_class: StatementSemanticClass,
    has_returning: bool,
    response_retained: bool,
    outcome: gpu_db_wal::CanonicalOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatementSemanticClass {
    TypedInsert,
    Update,
    Delete,
    Catalog,
    TableReset,
    TableRewrite,
    Sequence,
    KeyValueMutation,
}

/// Materialize the incomplete S1--S6 source owner after the established allocation-free S1/S4
/// first pass. The returned value is intentionally private to this aggregate semantics module.
#[allow(dead_code)]
pub(super) fn materialize_source_draft(
    decoded: &DecodedTypedInsertAggregate<'_>,
) -> Result<TypedInsertAggregateSourceDraft, EngineError> {
    let summary = validate_semantics_v1_scaffold(decoded)?;
    let statement_capacity = usize::try_from(summary.statement_count)
        .map_err(|_| violation("S1 statement count exceeds host address space"))?;
    let disposition_capacity = usize::try_from(summary.original_row_count)
        .map_err(|_| violation("S4 disposition count exceeds host address space"))?;
    let mut statements = Vec::with_capacity(statement_capacity);
    let mut dispositions = Vec::with_capacity(disposition_capacity);
    let measure = &decoded.layout().measure;
    let sections = decoded.sections();
    let typed_section = sections
        .get(TYPED_INSERT_SOURCES_SECTION)
        .ok_or_else(|| violation("S2 section is absent"))?;
    let non_insert_section = sections
        .get(NON_INSERT_SOURCES_SECTION)
        .ok_or_else(|| violation("S3 section is absent"))?;
    if typed_section.tag != TYPED_INSERT_SOURCES_TAG
        || typed_section.entry_count != summary.typed_insert_count
        || non_insert_section.tag != NON_INSERT_SOURCES_TAG
        || non_insert_section.entry_count != summary.canonical_operation_count
    {
        return Err(violation(
            "S2/S3 physical source counts do not match S1 families",
        ));
    }

    decoded.with_section_reader(STATEMENT_DIRECTORY_SECTION, |directory| {
        decoded.with_section_reader(TYPED_INSERT_SOURCES_SECTION, |typed_sources| {
            decoded.with_section_reader(NON_INSERT_SOURCES_SECTION, |non_insert_sources| {
                decoded.with_section_reader(ROW_DISPOSITIONS_SECTION, |row_dispositions| {
                    validate_payload_length(
                        directory,
                        summary.statement_count,
                        STATEMENT_DIRECTORY_ENTRY_BYTES,
                        "S1",
                    )?;
                    for expected_statement_ordinal in 0..summary.statement_count {
                        let entry = read_statement_directory_entry(directory)?;
                        if entry.statement_ordinal != expected_statement_ordinal {
                            return Err(violation(
                                "S1 statement ordinals changed before source materialization",
                            ));
                        }
                        let source = match entry.family {
                            StatementFamily::TypedInsert => SourceStatementKind::TypedInsert(
                                read_typed_insert_source(typed_sources, entry, measure)?,
                            ),
                            StatementFamily::CanonicalOperation => {
                                SourceStatementKind::CanonicalOperation(read_non_insert_source(
                                    non_insert_sources,
                                    entry,
                                )?)
                            }
                        };
                        let input_rows_start = u32::try_from(dispositions.len()).map_err(|_| {
                            violation("S4 disposition start exceeds u32 source ordinal")
                        })?;
                        if entry.family == StatementFamily::TypedInsert {
                            for expected_source_row in 0..entry.input_row_count {
                                let row = read_row_disposition(row_dispositions)?;
                                validate_row_disposition(
                                    row,
                                    entry.statement_ordinal,
                                    expected_source_row,
                                    entry.statement_digest,
                                    measure
                                        .allocator_before
                                        .checked_add(dispositions.len() as u64)
                                        .ok_or_else(|| {
                                            violation("S4 allocator cursor overflows")
                                        })?,
                                )?;
                                dispositions.push(CompactRowDisposition {
                                    stable_row_id: row.stable_row_id,
                                    kind: row.kind,
                                    block_ref: row.block_ref,
                                    transition_ref: row.transition_ref,
                                });
                            }
                        }
                        let input_rows_end = u32::try_from(dispositions.len()).map_err(|_| {
                            violation("S4 disposition end exceeds u32 source ordinal")
                        })?;
                        statements.push(SourceStatement {
                            statement_ordinal: entry.statement_ordinal,
                            family_ordinal: entry.family_ordinal,
                            request_digest: entry.request_digest,
                            statement_digest: entry.statement_digest,
                            overlay_before: entry.overlay_before,
                            overlay_after: entry.overlay_after,
                            input_rows: input_rows_start..input_rows_end,
                            sequence_effects: 0..0,
                            outcome: None,
                            source,
                        });
                    }
                    Ok(())
                })
            })
        })
    })?;

    if statements.len() != statement_capacity || dispositions.len() != disposition_capacity {
        return Err(violation(
            "S1--S4 materialization did not consume its exact admitted capacities",
        ));
    }
    let mut draft = TypedInsertAggregateSourceDraft {
        statements: statements.into_boxed_slice(),
        dispositions: dispositions.into_boxed_slice(),
        sequence_effects: Box::default(),
    };
    sequence_outcomes::materialize_sequence_effects_and_outcomes(decoded, &mut draft)?;
    Ok(draft)
}

fn read_typed_insert_source(
    reader: &mut DecodedAggregateSectionReader<'_>,
    entry: StatementDirectoryEntry,
    measure: &TypedInsertAggregateMeasure,
) -> Result<DecodedTypedInsertRecord, EngineError> {
    let record_len = usize::try_from(reader.u32()?)
        .map_err(|_| violation("S2 typed record length exceeds host address space"))?;
    if !(CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES..=MAX_SOURCE_BODY_BYTES).contains(&record_len) {
        return Err(violation(
            "S2 typed record length is outside its bounded canonical range",
        ));
    }
    let bytes = copy_bounded_source_body(reader, record_len, "S2 typed record")?;
    let record = decode_canonical_typed_insert_record(&bytes).map_err(|error| {
        EngineError::Durability(format!("typed INSERT aggregate S2 strict decode: {error}"))
    })?;
    let facts = record.facts();
    if entry.family_ordinal >= measure.insert_statement_count
        || facts.statement_ordinal.as_u32() != entry.statement_ordinal
        || facts.row_count != entry.input_row_count
        || facts.typed_statement_digest != entry.statement_digest
        || entry.request_digest != entry.statement_digest
    {
        return Err(violation(
            "S1/S2 typed record ordinal, row count, or digest closure is invalid",
        ));
    }
    let parent = record.sequence_parent();
    if facts.sequence_effect_count == 0 {
        if parent.is_some() || record.sequence_effects().len() != 0 {
            return Err(violation(
                "S2 empty sequence-effect closure is inconsistent",
            ));
        }
    } else {
        let parent = parent.ok_or_else(|| violation("S2 sequence effects lack their parent"))?;
        if parent.autocommit != (measure.flags & AGGREGATE_FLAG_AUTOCOMMIT != 0) {
            return Err(violation(
                "S1/S2 sequence parent autocommit does not bind aggregate transaction mode",
            ));
        }
        if parent.txn_id != measure.stable_transaction_id
            || parent.request_digest != entry.request_digest
            || parent.statement_ordinal.as_u32() != entry.statement_ordinal
            || record.sequence_effects().len() != facts.sequence_effect_count as usize
        {
            return Err(violation(
                "S1/S2 sequence parent or effect count is invalid",
            ));
        }
    }
    Ok(record)
}

fn read_non_insert_source(
    reader: &mut DecodedAggregateSectionReader<'_>,
    entry: StatementDirectoryEntry,
) -> Result<CurrentNonInsertCanonicalOperation, EngineError> {
    if reader.remaining() < S3_PREFIX_BYTES {
        return Err(violation("S3 source prefix is truncated"));
    }
    let statement_ordinal = reader.u32()?;
    let family_ordinal = reader.u32()?;
    let fragment_kind = gpu_db_wal::CanonicalFragmentKind::decode(reader.u16()?)?;
    let operation_codec = reader.u8()?;
    if reader.u8()? != 0 {
        return Err(violation("S3 source flags are nonzero"));
    }
    let body_len = reader.u32()?;
    let body_digest = reader.digest()?;
    let bytes = copy_bounded_source_body(
        reader,
        usize::try_from(body_len)
            .map_err(|_| violation("S3 operation body length exceeds host address space"))?,
        "S3 operation body",
    )?;
    let operation = decode_current_non_insert_canonical_operation(
        fragment_kind,
        operation_codec,
        body_len,
        body_digest,
        &bytes,
    )?;
    let facts = operation.facts();
    if entry.input_row_count != 0
        || statement_ordinal != entry.statement_ordinal
        || family_ordinal != entry.family_ordinal
        || fragment_kind != facts.fragment_kind()
        || operation_codec != facts.operation_codec()
        || body_digest != facts.body_digest()
        || entry.request_digest != body_digest
        || entry.statement_digest != facts.statement_digest()
    {
        return Err(violation(
            "S1/S3 ordinal, kind, body digest, or statement digest closure is invalid",
        ));
    }
    Ok(operation)
}

fn copy_bounded_source_body(
    reader: &mut DecodedAggregateSectionReader<'_>,
    bytes: usize,
    source: &'static str,
) -> Result<Box<[u8]>, EngineError> {
    if bytes > MAX_SOURCE_BODY_BYTES {
        return Err(violation(match source {
            "S2 typed record" => "S2 typed record exceeds the 16MiB source copy bound",
            "S3 operation body" => "S3 operation body exceeds the 16MiB source copy bound",
            _ => "source body exceeds the 16MiB copy bound",
        }));
    }
    let mut body = vec![0_u8; bytes].into_boxed_slice();
    reader.copy_exact(&mut body)?;
    Ok(body)
}

#[cfg(test)]
#[path = "source_materialization/tests.rs"]
mod tests;
