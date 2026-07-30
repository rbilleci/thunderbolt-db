//! Inert S5 sequence-closure and S6 statement-outcome source materialization.
//!
//! S5 is read straight from the chunk-bounded aggregate reader into a compact owner. Published
//! references use only one fixed 86-byte stack buffer; private effects retain no raw body because
//! their complete state remains exclusively in the decoded S2 model. S6 similarly uses its fixed
//! 92-byte outcome stack buffer. Neither section adds a second raw-source owner or a live replay
//! conversion. The existing production strict/canonical S2 re-encode peak remains
//! future-footprint debt; the 16 MiB cap applies only to raw S2/S3 source copies.

use super::*;
use crate::engine_canonical_operation::CurrentNonInsertSemanticClass;
use crate::typed_insert_aggregate::{
    AGGREGATE_FLAG_AUTOCOMMIT, AGGREGATE_FLAG_CATALOG, AGGREGATE_FLAG_PRIVATE_SEQUENCE,
    AGGREGATE_FLAG_PUBLISHED_SEQUENCE, AGGREGATE_FLAG_RESET, AGGREGATE_FLAG_RETAINED_RESPONSE,
    AGGREGATE_FLAG_RETURNING, OUTER_CONTENT_CATALOG, OUTER_CONTENT_PRIVATE_SEQUENCE,
    OUTER_CONTENT_PUBLISHED_SEQUENCE, OUTER_CONTENT_RESET, OUTER_CONTENT_RETURNING,
    OUTER_CONTENT_REWRITE,
};
use crate::typed_insert_batch::DecodedSequenceEffectKindFacts;

const SEQUENCE_CLOSURE_SECTION: usize = 4;
const STATEMENT_OUTCOMES_SECTION: usize = 5;
const SEQUENCE_CLOSURE_TAG: u16 = 5;
const STATEMENT_OUTCOMES_TAG: u16 = 6;
const S5_PREFIX_BYTES: u64 = 52;
const S6_ENTRY_BYTES: u64 = 136;
const S5_KIND_PUBLISHED: u8 = 1;
const S5_KIND_PRIVATE: u8 = 2;
const S5_FLAG_DEFAULT_EXPRESSION: u8 = 1 << 0;
const S5_FLAG_FINAL_OVERWRITTEN: u8 = 1 << 1;
const S5_KNOWN_FLAGS: u8 = S5_FLAG_DEFAULT_EXPRESSION | S5_FLAG_FINAL_OVERWRITTEN;
const S5_NO_DISPOSITION: u32 = u32::MAX;
const S6_FLAG_HAS_RETURNING: u16 = 1 << 0;
const S6_FLAG_RESPONSE_RETAINED: u16 = 1 << 1;
const S6_KNOWN_FLAGS: u16 = S6_FLAG_HAS_RETURNING | S6_FLAG_RESPONSE_RETAINED;

/// Add S5 and S6 facts to the already materialized S1--S4 source statements.
///
/// The two sections are deliberately separate passes. S5 must finish its owned compact effect
/// vector before S6 attaches the one terminal fact to each existing statement; neither pass
/// allocates a second statement carrier.
pub(super) fn materialize_sequence_effects_and_outcomes(
    decoded: &DecodedTypedInsertAggregate<'_>,
    draft: &mut TypedInsertAggregateSourceDraft,
) -> Result<(), EngineError> {
    materialize_s5(decoded, draft)?;
    materialize_s6(decoded, draft)
}

fn materialize_s5(
    decoded: &DecodedTypedInsertAggregate<'_>,
    draft: &mut TypedInsertAggregateSourceDraft,
) -> Result<(), EngineError> {
    let section = decoded
        .sections()
        .get(SEQUENCE_CLOSURE_SECTION)
        .ok_or_else(|| violation("S5 section is absent"))?;
    if section.tag != SEQUENCE_CLOSURE_TAG {
        return Err(violation("S5 section tag is invalid"));
    }
    let required_count = required_s5_entry_count(&draft.statements)?;
    if u64::from(section.entry_count) != required_count {
        return Err(violation(
            "S5 physical entry count does not match decoded S2/S3 source effects",
        ));
    }
    let minimum_bytes = required_count
        .checked_mul(S5_PREFIX_BYTES)
        .ok_or_else(|| violation("S5 minimum payload byte count overflows"))?;
    if section.payload_bytes < minimum_bytes {
        return Err(violation(
            "S5 payload cannot contain its decoded source-required entry prefixes",
        ));
    }
    let capacity = usize::try_from(required_count)
        .map_err(|_| violation("S5 source-required entry count exceeds host address space"))?;
    let mut effects = Vec::new();
    effects
        .try_reserve_exact(capacity)
        .map_err(|_| violation("S5 compact sequence-effect reservation failed"))?;
    let mut prior_published_transition = None;
    let measure = &decoded.layout().measure;
    let aggregate_autocommit = measure.flags & AGGREGATE_FLAG_AUTOCOMMIT != 0;
    let dispositions = &draft.dispositions;
    let statements = &mut draft.statements;

    decoded.with_section_reader(SEQUENCE_CLOSURE_SECTION, |reader| {
        for statement in statements.iter_mut() {
            let start = u32::try_from(effects.len())
                .map_err(|_| violation("S5 effect start exceeds u32 source ordinal"))?;
            match &statement.source {
                SourceStatementKind::TypedInsert(record) => {
                    let facts = record.facts();
                    let parent = record.sequence_parent();
                    for effect in record.sequence_effects() {
                        let entry = read_s5_entry(
                            reader,
                            statement.statement_ordinal,
                            effect.request.effect_ordinal,
                        )?;
                        let row = effect_disposition(
                            dispositions,
                            statement,
                            effect.request.row_ordinal,
                        )?;
                        validate_typed_s5_effect(
                            entry,
                            statement,
                            facts,
                            parent,
                            effect,
                            row,
                            aggregate_autocommit,
                            &mut prior_published_transition,
                            &mut effects,
                        )?;
                    }
                }
                SourceStatementKind::CanonicalOperation(operation) => {
                    if operation.facts().semantic_class() == CurrentNonInsertSemanticClass::Sequence
                    {
                        let entry = read_s5_entry(reader, statement.statement_ordinal, 0)?;
                        validate_explicit_sequence_s5_effect(
                            entry,
                            statement,
                            measure.stable_transaction_id,
                            aggregate_autocommit,
                            operation,
                            &mut prior_published_transition,
                            &mut effects,
                        )?;
                    }
                }
            }
            let end = u32::try_from(effects.len())
                .map_err(|_| violation("S5 effect end exceeds u32 source ordinal"))?;
            statement.sequence_effects = start..end;
        }
        Ok(())
    })?;

    if effects.len() != capacity {
        return Err(violation(
            "S5 physical entry count does not close over canonical S1 statement order",
        ));
    }
    draft.sequence_effects = effects.into_boxed_slice();
    validate_s5_aggregate_flags(decoded, draft)?;
    Ok(())
}

fn validate_s5_aggregate_flags(
    decoded: &DecodedTypedInsertAggregate<'_>,
    draft: &TypedInsertAggregateSourceDraft,
) -> Result<(), EngineError> {
    let has_published = draft
        .sequence_effects
        .iter()
        .any(|effect| matches!(&effect.kind, CompactSequenceEffectKind::Published(_)));
    let has_private = draft
        .sequence_effects
        .iter()
        .any(|effect| matches!(&effect.kind, CompactSequenceEffectKind::Private { .. }));
    let flags = decoded.layout().measure.flags;
    let outer = decoded.layout().measure.outer_flags;
    if (flags & AGGREGATE_FLAG_PUBLISHED_SEQUENCE != 0) != has_published
        || (outer & OUTER_CONTENT_PUBLISHED_SEQUENCE != 0) != has_published
        || (flags & AGGREGATE_FLAG_PRIVATE_SEQUENCE != 0) != has_private
        || (outer & OUTER_CONTENT_PRIVATE_SEQUENCE != 0) != has_private
    {
        return Err(violation(
            "S5 sequence ownership does not close over aggregate and outer content flags",
        ));
    }
    Ok(())
}

fn required_s5_entry_count(statements: &[SourceStatement]) -> Result<u64, EngineError> {
    statements.iter().try_fold(0_u64, |count, statement| {
        let required = match &statement.source {
            SourceStatementKind::TypedInsert(record) => {
                u64::from(record.facts().sequence_effect_count)
            }
            SourceStatementKind::CanonicalOperation(operation)
                if operation.facts().semantic_class()
                    == CurrentNonInsertSemanticClass::Sequence =>
            {
                1
            }
            SourceStatementKind::CanonicalOperation(_) => 0,
        };
        count
            .checked_add(required)
            .ok_or_else(|| violation("S5 source-required entry count overflows"))
    })
}

struct S5Entry {
    effect_ordinal: u32,
    flags: u8,
    disposition_ordinal: u32,
    body: S5Body,
}

enum S5Body {
    Published(crate::BinarySequenceValueReference),
    Private,
}

fn read_s5_entry(
    reader: &mut DecodedAggregateSectionReader<'_>,
    expected_statement_ordinal: u32,
    expected_effect_ordinal: u32,
) -> Result<S5Entry, EngineError> {
    if reader.remaining() < S5_PREFIX_BYTES {
        return Err(violation("S5 entry prefix is truncated"));
    }
    let statement_ordinal = reader.u32()?;
    let effect_ordinal = reader.u32()?;
    let kind = reader.u8()?;
    let flags = reader.u8()?;
    if flags & !S5_KNOWN_FLAGS != 0 || reader.u16()? != 0 {
        return Err(violation("S5 flags or reserved bytes are noncanonical"));
    }
    let disposition_ordinal = reader.u32()?;
    let body_len = reader.u32()?;
    let body_digest = reader.digest()?;
    if statement_ordinal != expected_statement_ordinal || effect_ordinal != expected_effect_ordinal
    {
        return Err(violation("S5 statement/effect order is not canonical"));
    }
    let body = match kind {
        S5_KIND_PUBLISHED => {
            if body_len as usize != crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES {
                return Err(violation("S5 published body width is not exact"));
            }
            let mut bytes = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
            reader.copy_exact(&mut bytes)?;
            if gpu_db_wal::canonical_request_digest(&bytes) != body_digest {
                return Err(violation(
                    "S5 published body digest does not match exact bytes",
                ));
            }
            let reference = crate::decode_sequence_value_reference_exact(&bytes).map_err(|error| {
                EngineError::Durability(format!(
                    "typed INSERT aggregate executable semantics v2: S5 published reference strict decode: {error}"
                ))
            })?;
            S5Body::Published(reference)
        }
        S5_KIND_PRIVATE => {
            if body_len != 0 || body_digest != [0; 32] {
                return Err(violation(
                    "S5 private entry must have an empty zero-digest body",
                ));
            }
            S5Body::Private
        }
        _ => return Err(violation("S5 sequence effect kind is invalid")),
    };
    Ok(S5Entry {
        effect_ordinal,
        flags,
        disposition_ordinal,
        body,
    })
}

fn effect_disposition<'a>(
    dispositions: &'a [CompactRowDisposition],
    statement: &SourceStatement,
    row_ordinal: u32,
) -> Result<&'a CompactRowDisposition, EngineError> {
    let absolute = statement
        .input_rows
        .start
        .checked_add(row_ordinal)
        .filter(|absolute| *absolute < statement.input_rows.end)
        .ok_or_else(|| violation("S2 sequence request row is outside its S4 statement range"))?;
    dispositions
        .get(
            usize::try_from(absolute)
                .map_err(|_| violation("S4 disposition ordinal is not addressable"))?,
        )
        .ok_or_else(|| violation("S4 disposition is absent for an S2 sequence request"))
}

#[allow(clippy::too_many_arguments)]
fn validate_typed_s5_effect(
    entry: S5Entry,
    statement: &SourceStatement,
    facts: crate::typed_insert_batch::DecodedTypedInsertRecordFacts,
    parent: Option<crate::typed_insert_batch::DecodedSequenceParentFacts>,
    effect: crate::typed_insert_batch::DecodedSequenceEffectFacts,
    disposition: &CompactRowDisposition,
    aggregate_autocommit: bool,
    prior_published_transition: &mut Option<u64>,
    out: &mut Vec<CompactSequenceEffect>,
) -> Result<(), EngineError> {
    let parent = parent.ok_or_else(|| violation("S2 sequence effect lacks its decoded parent"))?;
    let expected_disposition = statement
        .input_rows
        .start
        .checked_add(effect.request.row_ordinal)
        .ok_or_else(|| violation("S5 disposition ordinal overflows"))?;
    if entry.disposition_ordinal != expected_disposition
        || entry.disposition_ordinal == S5_NO_DISPOSITION
        || parent.autocommit != aggregate_autocommit
        || parent.txn_id == 0
        || parent.request_digest != statement.request_digest
        || parent.statement_ordinal.as_u32() != statement.statement_ordinal
        || facts.statement_ordinal.as_u32() != statement.statement_ordinal
        || effect.request.statement_ordinal.as_u32() != statement.statement_ordinal
        || effect.request.target_table_oid != facts.target.oid
        || effect.request.expression_ordinal
            != effect
                .request
                .absolute_expression_ordinal
                .checked_sub(parent.expression_ordinal_base)
                .ok_or_else(|| violation("S2 absolute sequence expression underflows its parent"))?
    {
        return Err(violation(
            "S5 typed effect does not close over S1/S2 parent/request facts",
        ));
    }
    let non_surviving = !matches!(disposition.kind, RowDispositionKind::Survives);
    match (entry.body, effect.kind) {
        (
            S5Body::Published(reference),
            DecodedSequenceEffectKindFacts::Published {
                transition_txn_id,
                input_digest,
                returned_value,
            },
        ) => {
            let expected_flags = S5_FLAG_DEFAULT_EXPRESSION
                | if reference.final_value_overwritten {
                    S5_FLAG_FINAL_OVERWRITTEN
                } else {
                    0
                };
            if entry.flags != expected_flags
                || !reference.default_expression
                || (non_surviving && !reference.final_value_overwritten)
                || reference.transition_txn_id != transition_txn_id
                || reference.parent_txn_id != parent.txn_id
                || reference.statement_ordinal != statement.statement_ordinal
                || reference.expression_ordinal != effect.request.absolute_expression_ordinal
                || reference.sequence_oid != effect.request.sequence_oid
                || reference.returned_value != returned_value
                || reference.returned_value != effect.resolved_value
                || reference.input_digest != input_digest
                || reference.table_oid != effect.request.target_table_oid
                || reference.column_id != effect.request.column_id
                || reference.row_id != disposition.stable_row_id
            {
                return Err(violation(
                    "S5 published default reference does not close over S1/S2/S4 facts",
                ));
            }
            if prior_published_transition
                .replace(reference.transition_txn_id)
                .is_some_and(|prior| prior >= reference.transition_txn_id)
            {
                return Err(violation(
                    "S5 published transition ids are not strictly canonical in source order",
                ));
            }
            out.push(CompactSequenceEffect {
                disposition_ordinal: entry.disposition_ordinal,
                kind: CompactSequenceEffectKind::Published(reference),
            });
        }
        (S5Body::Private, DecodedSequenceEffectKindFacts::Private { input_digest }) => {
            let final_value_overwritten = entry.flags & S5_FLAG_FINAL_OVERWRITTEN != 0;
            let expected_flags = S5_FLAG_DEFAULT_EXPRESSION
                | if final_value_overwritten {
                    S5_FLAG_FINAL_OVERWRITTEN
                } else {
                    0
                };
            if entry.flags != expected_flags || (non_surviving && !final_value_overwritten) {
                return Err(violation(
                    "S5 private default flags do not bind the decoded S4 disposition",
                ));
            }
            out.push(CompactSequenceEffect {
                disposition_ordinal: entry.disposition_ordinal,
                kind: CompactSequenceEffectKind::Private {
                    input_digest,
                    final_value_overwritten,
                },
            });
        }
        (S5Body::Private, DecodedSequenceEffectKindFacts::Published { .. })
        | (S5Body::Published(_), DecodedSequenceEffectKindFacts::Private { .. }) => {
            return Err(violation(
                "S5 effect kind does not match decoded S2 effect ownership",
            ));
        }
    }
    Ok(())
}

fn validate_explicit_sequence_s5_effect(
    entry: S5Entry,
    statement: &SourceStatement,
    aggregate_txn_id: u64,
    aggregate_autocommit: bool,
    operation: &CurrentNonInsertCanonicalOperation,
    prior_published_transition: &mut Option<u64>,
    out: &mut Vec<CompactSequenceEffect>,
) -> Result<(), EngineError> {
    let S5Body::Published(reference) = entry.body else {
        return Err(violation(
            "S5 private explicit sequence state is not representable",
        ));
    };
    let explicit = operation
        .explicit_sequence_facts()
        .ok_or_else(|| violation("S3 sequence class does not expose typed nextval/setval facts"))?;
    let expected_digest = crate::sequence_value_input_digest(crate::SequenceValueInput {
        parent_txn_id: aggregate_txn_id,
        parent_autocommit: aggregate_autocommit,
        statement_ordinal: statement.statement_ordinal,
        expression_ordinal: 0,
        // Codec-4 S3 keeps its body digest as S1 request identity, while the current sequence
        // lifecycle binds its parent request to the typed command's statement digest.
        parent_request_digest: statement.statement_digest,
        source_name: explicit.source_name(),
        operation: explicit.operation(),
        set_value: explicit.requested_set_value(),
    });
    if entry.effect_ordinal != 0
        || entry.flags != 0
        || entry.disposition_ordinal != S5_NO_DISPOSITION
        || reference.parent_txn_id != aggregate_txn_id
        || reference.statement_ordinal != statement.statement_ordinal
        || reference.expression_ordinal != 0
        || reference.sequence_oid == 0
        || reference.input_digest != expected_digest
        || reference.default_expression
        || reference.final_value_overwritten
        || reference.table_oid != 0
        || reference.column_id != 0
        || reference.row_id != 0
        || explicit
            .requested_set_value()
            .is_some_and(|requested| reference.returned_value != requested)
    {
        return Err(violation(
            "S5 explicit sequence reference does not close over S1/S3 aggregate facts",
        ));
    }
    if prior_published_transition
        .replace(reference.transition_txn_id)
        .is_some_and(|prior| prior >= reference.transition_txn_id)
    {
        return Err(violation(
            "S5 published transition ids are not strictly canonical in source order",
        ));
    }
    out.push(CompactSequenceEffect {
        disposition_ordinal: entry.disposition_ordinal,
        kind: CompactSequenceEffectKind::Published(reference),
    });
    Ok(())
}

fn materialize_s6(
    decoded: &DecodedTypedInsertAggregate<'_>,
    draft: &mut TypedInsertAggregateSourceDraft,
) -> Result<(), EngineError> {
    let section = decoded
        .sections()
        .get(STATEMENT_OUTCOMES_SECTION)
        .ok_or_else(|| violation("S6 section is absent"))?;
    if section.tag != STATEMENT_OUTCOMES_TAG
        || section.entry_count
            != u32::try_from(draft.statements.len())
                .map_err(|_| violation("S1 statement count exceeds S6 ordinal range"))?
    {
        return Err(violation("S6 physical count does not match S1 statements"));
    }
    decoded.with_section_reader(STATEMENT_OUTCOMES_SECTION, |reader| {
        let expected = u64::from(section.entry_count)
            .checked_mul(S6_ENTRY_BYTES)
            .ok_or_else(|| violation("S6 payload byte count overflows"))?;
        if reader.remaining() != expected {
            return Err(violation(
                "S6 payload length is not exact for its outer statement count",
            ));
        }
        for statement in draft.statements.iter_mut() {
            let entry = read_s6_entry(reader)?;
            validate_s6_entry(entry, statement, &draft.dispositions)?;
        }
        Ok(())
    })?;
    validate_s6_aggregate_flags(decoded, draft)
}

fn validate_s6_aggregate_flags(
    decoded: &DecodedTypedInsertAggregate<'_>,
    draft: &TypedInsertAggregateSourceDraft,
) -> Result<(), EngineError> {
    let mut has_returning = false;
    let mut retained_count = 0_u32;
    let mut has_catalog = false;
    let mut has_reset = false;
    let mut has_rewrite = false;
    for statement in draft.statements.iter() {
        let outcome = statement
            .outcome
            .as_ref()
            .ok_or_else(|| violation("S6 did not attach one outcome to every S1 statement"))?;
        has_returning |= outcome.has_returning;
        retained_count = retained_count
            .checked_add(u32::from(outcome.response_retained))
            .ok_or_else(|| violation("S6 retained response count overflows"))?;
        has_catalog |= outcome.semantic_class == StatementSemanticClass::Catalog;
        has_reset |= outcome.semantic_class == StatementSemanticClass::TableReset;
        has_rewrite |= outcome.semantic_class == StatementSemanticClass::TableRewrite;
    }
    let flags = decoded.layout().measure.flags;
    let outer = decoded.layout().measure.outer_flags;
    let s8_entries = decoded.sections()[7].entry_count;
    if (flags & AGGREGATE_FLAG_RETURNING != 0) != has_returning
        || (outer & OUTER_CONTENT_RETURNING != 0) != has_returning
        || (flags & AGGREGATE_FLAG_RETAINED_RESPONSE != 0) != (retained_count != 0)
        || retained_count != s8_entries
        || (flags & AGGREGATE_FLAG_CATALOG != 0) != has_catalog
        || (outer & OUTER_CONTENT_CATALOG != 0) != has_catalog
        || (flags & AGGREGATE_FLAG_RESET != 0) != has_reset
        || (outer & OUTER_CONTENT_RESET != 0) != has_reset
        || (outer & OUTER_CONTENT_REWRITE != 0) != has_rewrite
    {
        return Err(violation(
            "S6 outcomes do not close over aggregate flags, outer content, or S8 retention count",
        ));
    }
    Ok(())
}

struct S6Entry {
    statement_ordinal: u32,
    family_ordinal: u32,
    semantic_class: StatementSemanticClass,
    has_returning: bool,
    response_retained: bool,
    statement_digest: gpu_db_wal::CanonicalDigest,
    outcome: gpu_db_wal::CanonicalOutcome,
}

fn read_s6_entry(reader: &mut DecodedAggregateSectionReader<'_>) -> Result<S6Entry, EngineError> {
    let statement_ordinal = reader.u32()?;
    let family_ordinal = reader.u32()?;
    let semantic_class = decode_s6_semantic_class(reader.u16()?)?;
    let flags = reader.u16()?;
    if flags & !S6_KNOWN_FLAGS != 0 {
        return Err(violation("S6 outcome flags are noncanonical"));
    }
    let statement_digest = reader.digest()?;
    let mut outcome_bytes = [0_u8; gpu_db_wal::CANONICAL_OUTCOME_BYTES];
    reader.copy_exact(&mut outcome_bytes)?;
    let outcome = gpu_db_wal::decode_canonical_outcome_exact(&outcome_bytes).map_err(|error| {
        EngineError::Durability(format!(
            "typed INSERT aggregate executable semantics v2: S6 outcome strict decode: {error}"
        ))
    })?;
    Ok(S6Entry {
        statement_ordinal,
        family_ordinal,
        semantic_class,
        has_returning: flags & S6_FLAG_HAS_RETURNING != 0,
        response_retained: flags & S6_FLAG_RESPONSE_RETAINED != 0,
        statement_digest,
        outcome,
    })
}

fn decode_s6_semantic_class(value: u16) -> Result<StatementSemanticClass, EngineError> {
    match value {
        1 => Ok(StatementSemanticClass::TypedInsert),
        2 => Err(violation(
            "S6 CopyInsert has no owned source in the current draft",
        )),
        3 => Ok(StatementSemanticClass::Update),
        4 => Ok(StatementSemanticClass::Delete),
        5 => Ok(StatementSemanticClass::Catalog),
        6 => Ok(StatementSemanticClass::TableReset),
        7 => Ok(StatementSemanticClass::TableRewrite),
        8 => Ok(StatementSemanticClass::Sequence),
        9 => Ok(StatementSemanticClass::KeyValueMutation),
        _ => Err(violation("S6 semantic class is invalid")),
    }
}

fn validate_s6_entry(
    entry: S6Entry,
    statement: &mut SourceStatement,
    dispositions: &[CompactRowDisposition],
) -> Result<(), EngineError> {
    let expected_class = expected_s6_class(&statement.source);
    if entry.statement_ordinal != statement.statement_ordinal
        || entry.family_ordinal != statement.family_ordinal
        || entry.semantic_class != expected_class
        || entry.statement_digest != statement.statement_digest
        || entry.outcome.target_digest != statement.overlay_after
        || (entry.response_retained && !entry.has_returning)
        || (entry.response_retained
            && entry.outcome.kind == gpu_db_wal::CanonicalOutcomeKind::AbortError)
    {
        return Err(violation(
            "S6 statement identity, class, target, or retention binding is invalid",
        ));
    }
    match &statement.source {
        SourceStatementKind::TypedInsert(record) => {
            let facts = record.facts();
            let has_returning = facts.returning.column_count != 0;
            if entry.has_returning != has_returning {
                return Err(violation(
                    "S6 typed INSERT RETURNING flag does not match S2 layout",
                ));
            }
            let affected_rows =
                typed_statement_affected_rows(dispositions, statement.input_rows.clone())?;
            validate_typed_outcome(&entry.outcome, affected_rows, has_returning)?;
        }
        SourceStatementKind::CanonicalOperation(operation) => {
            let has_returning = operation.facts().has_returning();
            if entry.has_returning != has_returning {
                return Err(violation(
                    "S6 current S3 RETURNING flag does not match the decoded operation",
                ));
            }
            // S6 proves only the presence of a typed UPDATE/DELETE RETURNING result. The
            // operation remains move-owned here; S7 will bind its exact typed projections.
            validate_s3_outcome(&entry.outcome, expected_class, has_returning)?;
        }
    }
    statement.outcome = Some(CompactStatementOutcome {
        semantic_class: entry.semantic_class,
        has_returning: entry.has_returning,
        response_retained: entry.response_retained,
        outcome: entry.outcome,
    });
    Ok(())
}

fn expected_s6_class(source: &SourceStatementKind) -> StatementSemanticClass {
    match source {
        SourceStatementKind::TypedInsert(_) => StatementSemanticClass::TypedInsert,
        SourceStatementKind::CanonicalOperation(operation) => {
            match operation.facts().semantic_class() {
                CurrentNonInsertSemanticClass::Update => StatementSemanticClass::Update,
                CurrentNonInsertSemanticClass::Delete => StatementSemanticClass::Delete,
                CurrentNonInsertSemanticClass::Catalog => StatementSemanticClass::Catalog,
                CurrentNonInsertSemanticClass::TableReset => StatementSemanticClass::TableReset,
                CurrentNonInsertSemanticClass::TableRewrite => StatementSemanticClass::TableRewrite,
                CurrentNonInsertSemanticClass::Sequence => StatementSemanticClass::Sequence,
                CurrentNonInsertSemanticClass::KeyValueMutation => {
                    StatementSemanticClass::KeyValueMutation
                }
            }
        }
    }
}

fn typed_statement_affected_rows(
    dispositions: &[CompactRowDisposition],
    range: Range<u32>,
) -> Result<u64, EngineError> {
    let mut affected = 0_u64;
    for disposition in &dispositions[usize::try_from(range.start)
        .map_err(|_| violation("S4 typed range start is not addressable"))?
        ..usize::try_from(range.end)
            .map_err(|_| violation("S4 typed range end is not addressable"))?]
    {
        if matches!(
            disposition.kind,
            RowDispositionKind::Survives | RowDispositionKind::AppliedThenCanceled
        ) {
            affected = affected
                .checked_add(1)
                .ok_or_else(|| violation("S6 typed affected-row count overflows"))?;
        }
    }
    Ok(affected)
}

fn validate_typed_outcome(
    outcome: &gpu_db_wal::CanonicalOutcome,
    affected_rows: u64,
    has_returning: bool,
) -> Result<(), EngineError> {
    match outcome.kind {
        gpu_db_wal::CanonicalOutcomeKind::CommitSuccess => {
            if outcome.affected_rows != affected_rows {
                return Err(violation(
                    "S6 typed INSERT success affected rows do not match S4",
                ));
            }
            validate_success_or_noop_constraint(outcome)?;
            if has_returning != (outcome.returning_digest != [0; 32]) {
                return Err(violation(
                    "S6 typed INSERT success RETURNING digest presence is invalid",
                ));
            }
        }
        gpu_db_wal::CanonicalOutcomeKind::CommitNoOp => {
            if outcome.affected_rows != 0 || affected_rows != 0 {
                return Err(violation("S6 CommitNoOp must have zero affected rows"));
            }
            validate_success_or_noop_constraint(outcome)?;
            if has_returning != (outcome.returning_digest != [0; 32]) {
                return Err(violation(
                    "S6 typed INSERT no-op RETURNING digest presence is invalid",
                ));
            }
        }
        gpu_db_wal::CanonicalOutcomeKind::AbortError => {
            if affected_rows != 0 {
                return Err(violation(
                    "S6 typed INSERT abort cannot hide S4 affected-row dispositions",
                ));
            }
            validate_abort_outcome(outcome)?;
        }
    }
    Ok(())
}

fn validate_s3_outcome(
    outcome: &gpu_db_wal::CanonicalOutcome,
    class: StatementSemanticClass,
    has_returning: bool,
) -> Result<(), EngineError> {
    match outcome.kind {
        gpu_db_wal::CanonicalOutcomeKind::CommitSuccess => {
            validate_success_or_noop_constraint(outcome)?;
            // Current UPDATE/DELETE retain their own resolved count semantics. The remaining
            // strict S3 classes are statement effects, not row producers, and therefore report 0.
            if !matches!(
                class,
                StatementSemanticClass::Update | StatementSemanticClass::Delete
            ) && outcome.affected_rows != 0
            {
                return Err(violation(
                    "S6 non-DML S3 success must have zero affected rows",
                ));
            }
            validate_s3_returning_digest_presence(outcome, has_returning, "success")?;
        }
        gpu_db_wal::CanonicalOutcomeKind::CommitNoOp => {
            if outcome.affected_rows != 0 {
                return Err(violation("S6 CommitNoOp must have zero affected rows"));
            }
            validate_success_or_noop_constraint(outcome)?;
            validate_s3_returning_digest_presence(outcome, has_returning, "no-op")?;
        }
        gpu_db_wal::CanonicalOutcomeKind::AbortError => validate_abort_outcome(outcome)?,
    }
    Ok(())
}

fn validate_s3_returning_digest_presence(
    outcome: &gpu_db_wal::CanonicalOutcome,
    has_returning: bool,
    outcome_name: &'static str,
) -> Result<(), EngineError> {
    if has_returning != (outcome.returning_digest != [0; 32]) {
        return Err(violation(match outcome_name {
            "success" => "S6 current S3 success RETURNING digest presence is invalid",
            "no-op" => "S6 current S3 no-op RETURNING digest presence is invalid",
            _ => unreachable!("S3 outcome names are closed"),
        }));
    }
    Ok(())
}

fn validate_success_or_noop_constraint(
    outcome: &gpu_db_wal::CanonicalOutcome,
) -> Result<(), EngineError> {
    if outcome.sqlstate.is_some() || outcome.constraint_id != 0 {
        return Err(violation(
            "S6 successful or no-op outcome carries SQLSTATE/constraint id",
        ));
    }
    Ok(())
}

fn validate_abort_outcome(outcome: &gpu_db_wal::CanonicalOutcome) -> Result<(), EngineError> {
    let state = outcome
        .sqlstate
        .ok_or_else(|| violation("S6 abort outcome lacks its SQLSTATE"))?;
    if outcome.affected_rows != 0 || outcome.returning_digest != [0; 32] {
        return Err(violation(
            "S6 abort outcome carries rows or RETURNING digest",
        ));
    }
    let supports_constraint = matches!(&state, b"23502" | b"23503" | b"23505" | b"23514");
    if supports_constraint != (outcome.constraint_id != 0) {
        return Err(violation(
            "S6 abort constraint id does not match its supported SQLSTATE class",
        ));
    }
    Ok(())
}

/// Source owner after one Engine-held durable sequence-reference validation.
///
/// This deliberately has no constructor or live conversion. A future replay/GPU adopter may
/// only receive it from the Engine-owned validation boundary below.
#[allow(dead_code)]
pub(crate) struct SequenceValidatedTypedInsertAggregateSourceDraft {
    draft: TypedInsertAggregateSourceDraft,
}

/// Inert Engine-owned seam for the complete S1--S6 source owner.
///
/// This intentionally borrows and filters the one S5 effect box once. It makes no allocation,
/// no replay plan, and no route selection; a future replay/GPU adopter must invoke it after pure
/// codec materialization and before consuming the owner.
impl crate::Engine {
    #[allow(dead_code)]
    pub(crate) fn validate_typed_insert_aggregate_source_draft_sequence_references(
        &self,
        draft: TypedInsertAggregateSourceDraft,
    ) -> Result<SequenceValidatedTypedInsertAggregateSourceDraft, EngineError> {
        self.validate_sequence_value_reference_iter(draft.sequence_effects.iter().filter_map(
            |effect| match &effect.kind {
                CompactSequenceEffectKind::Published(reference) => Some(reference),
                CompactSequenceEffectKind::Private { .. } => None,
            },
        ))?;
        Ok(SequenceValidatedTypedInsertAggregateSourceDraft { draft })
    }
}
