//! Private provisional semantics-v1 S1/S4 scaffold for the future codec-5 replay owner.
//!
//! This is not the normative semantics-v2 S4 reader. Codec-5's physical aggregate header remains
//! semantics v1, and the provisional S4 uses
//! that form's global allocator range and both-sentinel canceled/suppressed references, and every
//! owner here is deliberately unreachable from the live writer, recovery, and apply paths. The
//! PLAN-owned S4/S7 checkpoint must version, replace, or rename this scaffold after freezing the
//! exact S7 contract; it may not leave a second S4 acceptance authority.
//!
//! The validator does *not* claim S2 typed-record table/schema closure or S7 final-image/block
//! closure.  In particular, proving survivor-reference uniqueness or a complete image bijection
//! here would require unbounded state and would duplicate S7's eventual authority.  This slice
//! checks only the S4 structural reference contract; the full replay transaction validator will
//! bind each typed statement to S2 and each surviving reference to S7.  S4 survivors are only an
//! INSERT-origin subset of the aggregate's final transitions: S7 may additionally contain
//! replacement or delete transitions for pre-existing rows.

use super::codec::{DecodedAggregateSectionReader, DecodedTypedInsertAggregate};
use crate::EngineError;

#[path = "semantics_v1_scaffold/source_materialization.rs"]
mod source_materialization;

const STATEMENT_DIRECTORY_SECTION: usize = 0;
const ROW_DISPOSITIONS_SECTION: usize = 3;
const STATEMENT_DIRECTORY_TAG: u16 = 1;
const ROW_DISPOSITIONS_TAG: u16 = 4;
const STATEMENT_DIRECTORY_ENTRY_BYTES: u64 = 144;
const ROW_DISPOSITION_ENTRY_BYTES: u64 = 64;

const FAMILY_TYPED_INSERT: u8 = 1;
const FAMILY_CANONICAL_OPERATION: u8 = 2;
const DISPOSITION_SURVIVES: u8 = 1;
const DISPOSITION_APPLIED_THEN_CANCELED: u8 = 2;
const DISPOSITION_SUPPRESSED_AT_STATEMENT: u8 = 3;

/// Stack-only facts established by the S1/S4 traversal.  This carries no replay, mutation, or
/// publication capability; a later owner must still complete S2/S7 and the remaining sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticsV1ScaffoldSummary {
    statement_count: u32,
    typed_insert_count: u32,
    canonical_operation_count: u32,
    original_row_count: u64,
    insert_affected_row_count: u64,
    final_survivor_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatementFamily {
    TypedInsert,
    CanonicalOperation,
}

/// Validate the provisional semantics-1 S1 and S4 payloads directly over the decoded chunk owners.
///
/// This traversal is allocation-free.  It keeps one current S1 entry while consuming exactly its
/// S4 source-row range, which proves directory/source ordering without ever assembling a section
/// or building a statement map. It is a semantics-v1-only scaffold and is private until the
/// final replay transaction owns the full S1--S8 closure.
fn validate_semantics_v1_scaffold(
    decoded: &DecodedTypedInsertAggregate<'_>,
) -> Result<SemanticsV1ScaffoldSummary, EngineError> {
    let measure = &decoded.layout().measure;
    let sections = decoded.sections();
    let directory_section = sections
        .get(STATEMENT_DIRECTORY_SECTION)
        .ok_or_else(|| violation("S1 section is absent"))?;
    let disposition_section = sections
        .get(ROW_DISPOSITIONS_SECTION)
        .ok_or_else(|| violation("S4 section is absent"))?;
    if directory_section.tag != STATEMENT_DIRECTORY_TAG
        || directory_section.entry_count != measure.statement_count
    {
        return Err(violation(
            "S1 physical count does not match aggregate statement count",
        ));
    }
    if disposition_section.tag != ROW_DISPOSITIONS_TAG
        || u64::from(disposition_section.entry_count) != measure.original_inserted_row_count
    {
        return Err(violation(
            "S4 physical count does not match aggregate original row count",
        ));
    }

    decoded.with_section_reader(STATEMENT_DIRECTORY_SECTION, |directory| {
        decoded.with_section_reader(ROW_DISPOSITIONS_SECTION, |dispositions| {
            validate_payload_length(
                directory,
                directory_section.entry_count,
                STATEMENT_DIRECTORY_ENTRY_BYTES,
                "S1",
            )?;
            validate_payload_length(
                dispositions,
                disposition_section.entry_count,
                ROW_DISPOSITION_ENTRY_BYTES,
                "S4",
            )?;

            let mut typed_insert_count = 0_u32;
            let mut canonical_operation_count = 0_u32;
            let mut original_row_count = 0_u64;
            let mut consumed_rows = 0_u64;
            let mut insert_affected_row_count = 0_u64;
            let mut final_survivor_count = 0_u64;
            let mut next_stable_row_id = measure.allocator_before;
            let mut prior_overlay_after = None;

            for expected_statement_ordinal in 0..measure.statement_count {
                let entry = read_statement_directory_entry(directory)?;
                if entry.statement_ordinal != expected_statement_ordinal {
                    return Err(violation(
                        "S1 statement ordinals are not globally canonical",
                    ));
                }
                if entry.request_digest == [0; 32]
                    || entry.statement_digest == [0; 32]
                    || entry.overlay_before == [0; 32]
                    || entry.overlay_after == [0; 32]
                {
                    return Err(violation("S1 requires nonzero digests and overlay roots"));
                }
                if prior_overlay_after
                    .replace(entry.overlay_after)
                    .is_some_and(|prior| prior != entry.overlay_before)
                {
                    return Err(violation(
                        "S1 overlay roots do not chain between statements",
                    ));
                }

                match entry.family {
                    StatementFamily::TypedInsert => {
                        if entry.family_ordinal != typed_insert_count {
                            return Err(violation(
                                "S1 typed INSERT family ordinals are not contiguous",
                            ));
                        }
                        typed_insert_count = typed_insert_count
                            .checked_add(1)
                            .ok_or_else(|| violation("S1 typed INSERT count overflows"))?;
                        original_row_count = original_row_count
                            .checked_add(u64::from(entry.input_row_count))
                            .ok_or_else(|| violation("S1 original row count overflows"))?;
                        for expected_source_row in 0..entry.input_row_count {
                            let disposition = read_row_disposition(dispositions)?;
                            validate_row_disposition(
                                disposition,
                                entry.statement_ordinal,
                                expected_source_row,
                                entry.statement_digest,
                                next_stable_row_id,
                            )?;
                            next_stable_row_id = next_stable_row_id
                                .checked_add(1)
                                .ok_or_else(|| violation("S4 stable row-id range overflows"))?;
                            consumed_rows = consumed_rows
                                .checked_add(1)
                                .ok_or_else(|| violation("S4 row count overflows"))?;
                            match disposition.kind {
                                RowDispositionKind::Survives => {
                                    final_survivor_count =
                                        final_survivor_count.checked_add(1).ok_or_else(|| {
                                            violation("S4 final survivor count overflows")
                                        })?;
                                    insert_affected_row_count =
                                        insert_affected_row_count.checked_add(1).ok_or_else(
                                            || violation("S4 affected-row count overflows"),
                                        )?;
                                }
                                RowDispositionKind::AppliedThenCanceled => {
                                    insert_affected_row_count =
                                        insert_affected_row_count.checked_add(1).ok_or_else(
                                            || violation("S4 affected-row count overflows"),
                                        )?;
                                }
                                RowDispositionKind::SuppressedAtStatement => {}
                            }
                        }
                    }
                    StatementFamily::CanonicalOperation => {
                        if entry.family_ordinal != canonical_operation_count {
                            return Err(violation(
                                "S1 canonical-operation family ordinals are not contiguous",
                            ));
                        }
                        if entry.input_row_count != 0 {
                            return Err(violation(
                                "S1 canonical operation has a nonzero INSERT row count",
                            ));
                        }
                        canonical_operation_count = canonical_operation_count
                            .checked_add(1)
                            .ok_or_else(|| violation("S1 canonical-operation count overflows"))?;
                    }
                }
            }

            if typed_insert_count != measure.insert_statement_count
                || typed_insert_count != sections[1].entry_count
                || canonical_operation_count != sections[2].entry_count
            {
                return Err(violation(
                    "S1 family counts do not index the S2/S3 physical families",
                ));
            }
            if original_row_count != measure.original_inserted_row_count
                || consumed_rows != measure.original_inserted_row_count
                || consumed_rows != u64::from(disposition_section.entry_count)
            {
                return Err(violation("S1/S4 original INSERT row counts do not close"));
            }
            if next_stable_row_id != measure.allocator_high_water {
                return Err(violation(
                    "S4 stable row ids are not the contiguous aggregate allocator range",
                ));
            }
            if final_survivor_count > measure.final_row_transition_count {
                return Err(violation(
                    "S4 surviving rows exceed aggregate final transition count",
                ));
            }
            Ok(SemanticsV1ScaffoldSummary {
                statement_count: measure.statement_count,
                typed_insert_count,
                canonical_operation_count,
                original_row_count,
                insert_affected_row_count,
                final_survivor_count,
            })
        })
    })
}

fn validate_payload_length(
    reader: &mut DecodedAggregateSectionReader<'_>,
    expected_count: u32,
    entry_bytes: u64,
    section: &'static str,
) -> Result<(), EngineError> {
    let expected_bytes = u64::from(expected_count)
        .checked_mul(entry_bytes)
        .ok_or_else(|| violation("semantics-v1 scaffold payload size overflows"))?;
    if reader.remaining() != expected_bytes {
        return Err(violation(match section {
            "S1" => "S1 payload length is not exact for its outer entry count",
            "S4" => "S4 payload length is not exact for its outer entry count",
            _ => "semantics-v1 scaffold payload length is not exact",
        }));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct StatementDirectoryEntry {
    statement_ordinal: u32,
    family_ordinal: u32,
    family: StatementFamily,
    input_row_count: u32,
    request_digest: [u8; 32],
    statement_digest: [u8; 32],
    overlay_before: [u8; 32],
    overlay_after: [u8; 32],
}

fn read_statement_directory_entry(
    reader: &mut DecodedAggregateSectionReader<'_>,
) -> Result<StatementDirectoryEntry, EngineError> {
    let statement_ordinal = reader.u32()?;
    let family_ordinal = reader.u32()?;
    let family = match reader.u8()? {
        FAMILY_TYPED_INSERT => StatementFamily::TypedInsert,
        FAMILY_CANONICAL_OPERATION => StatementFamily::CanonicalOperation,
        _ => return Err(violation("S1 statement family is invalid")),
    };
    if reader.u8()? != 0 || reader.u16()? != 0 {
        return Err(violation("S1 family flags or reserved bytes are nonzero"));
    }
    Ok(StatementDirectoryEntry {
        statement_ordinal,
        family_ordinal,
        family,
        input_row_count: reader.u32()?,
        request_digest: reader.digest()?,
        statement_digest: reader.digest()?,
        overlay_before: reader.digest()?,
        overlay_after: reader.digest()?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowDispositionKind {
    Survives,
    AppliedThenCanceled,
    SuppressedAtStatement,
}

#[derive(Debug, Clone, Copy)]
struct RowDispositionEntry {
    statement_ordinal: u32,
    source_row: u32,
    stable_row_id: u64,
    kind: RowDispositionKind,
    block_ref: u32,
    transition_ref: u32,
    statement_digest: [u8; 32],
}

fn read_row_disposition(
    reader: &mut DecodedAggregateSectionReader<'_>,
) -> Result<RowDispositionEntry, EngineError> {
    let statement_ordinal = reader.u32()?;
    let source_row = reader.u32()?;
    let stable_row_id = reader.u64()?;
    let kind = match reader.u8()? {
        DISPOSITION_SURVIVES => RowDispositionKind::Survives,
        DISPOSITION_APPLIED_THEN_CANCELED => RowDispositionKind::AppliedThenCanceled,
        DISPOSITION_SUPPRESSED_AT_STATEMENT => RowDispositionKind::SuppressedAtStatement,
        _ => return Err(violation("S4 row disposition is invalid")),
    };
    if reader.u8()? != 0 || reader.u16()? != 0 {
        return Err(violation(
            "S4 row-disposition flags or reserved bytes are nonzero",
        ));
    }
    let block_ref = reader.u32()?;
    let transition_ref = reader.u32()?;
    if reader.u32()? != 0 {
        return Err(violation(
            "S4 row-disposition trailing reserved field is nonzero",
        ));
    }
    Ok(RowDispositionEntry {
        statement_ordinal,
        source_row,
        stable_row_id,
        kind,
        block_ref,
        transition_ref,
        statement_digest: reader.digest()?,
    })
}

fn validate_row_disposition(
    entry: RowDispositionEntry,
    expected_statement_ordinal: u32,
    expected_source_row: u32,
    expected_statement_digest: [u8; 32],
    expected_stable_row_id: u64,
) -> Result<(), EngineError> {
    if entry.statement_ordinal != expected_statement_ordinal
        || entry.source_row != expected_source_row
        || entry.stable_row_id != expected_stable_row_id
        || entry.statement_digest != expected_statement_digest
    {
        return Err(violation(
            "S4 row does not bind to canonical statement/source/digest/allocator order",
        ));
    }
    let refs_are_sentinel = entry.block_ref == u32::MAX && entry.transition_ref == u32::MAX;
    let refs_are_live = entry.block_ref != u32::MAX && entry.transition_ref != u32::MAX;
    match entry.kind {
        RowDispositionKind::Survives if refs_are_live => Ok(()),
        RowDispositionKind::Survives => Err(violation(
            "S4 surviving row must carry non-sentinel block and transition references",
        )),
        RowDispositionKind::AppliedThenCanceled | RowDispositionKind::SuppressedAtStatement
            if refs_are_sentinel =>
        {
            Ok(())
        }
        RowDispositionKind::AppliedThenCanceled | RowDispositionKind::SuppressedAtStatement => Err(
            violation("S4 canceled or suppressed row must carry only sentinel references"),
        ),
    }
}

fn violation(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v1 scaffold: {message}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_insert_aggregate::{
        decode_typed_insert_aggregate_bodies, encode_typed_insert_aggregate_bodies,
        reserve_typed_insert_aggregate_bodies, typed_insert_aggregate_status_roots,
        EncodedTypedInsertAggregateBodies, TypedInsertAggregateSectionView,
        TypedInsertAggregateStatusRoots, TypedInsertAggregateView, TypedInsertStatusV2,
        AGGREGATE_CHUNK_PAYLOAD_BYTES, AGGREGATE_FLAG_EXPLICIT, AGGREGATE_SECTION_COUNT,
        OUTER_CONTENT_ROW, OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
    };

    const ENTRY_COUNTS: [u32; AGGREGATE_SECTION_COUNT] = [3, 2, 1, 4, 0, 3, 1, 0];

    #[derive(Clone, Copy)]
    struct DirectorySpec {
        ordinal: u32,
        family_ordinal: u32,
        family: u8,
        rows: u32,
        request: u8,
        statement: u8,
        before: u8,
        after: u8,
    }

    #[derive(Clone, Copy)]
    struct DispositionSpec {
        statement_ordinal: u32,
        source_row: u32,
        stable_row_id: u64,
        kind: u8,
        block_ref: u32,
        transition_ref: u32,
        statement: u8,
    }

    fn directory_payload(entries: &[DirectorySpec]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(entries.len() * 144);
        for entry in entries {
            bytes.extend_from_slice(&entry.ordinal.to_le_bytes());
            bytes.extend_from_slice(&entry.family_ordinal.to_le_bytes());
            bytes.push(entry.family);
            bytes.extend_from_slice(&[0; 3]);
            bytes.extend_from_slice(&entry.rows.to_le_bytes());
            bytes.extend_from_slice(&[entry.request; 32]);
            bytes.extend_from_slice(&[entry.statement; 32]);
            bytes.extend_from_slice(&[entry.before; 32]);
            bytes.extend_from_slice(&[entry.after; 32]);
        }
        bytes
    }

    fn dispositions_payload(entries: &[DispositionSpec]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(entries.len() * 64);
        for entry in entries {
            bytes.extend_from_slice(&entry.statement_ordinal.to_le_bytes());
            bytes.extend_from_slice(&entry.source_row.to_le_bytes());
            bytes.extend_from_slice(&entry.stable_row_id.to_le_bytes());
            bytes.push(entry.kind);
            bytes.extend_from_slice(&[0; 3]);
            bytes.extend_from_slice(&entry.block_ref.to_le_bytes());
            bytes.extend_from_slice(&entry.transition_ref.to_le_bytes());
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&[entry.statement; 32]);
        }
        bytes
    }

    fn valid_payloads() -> [Vec<u8>; AGGREGATE_SECTION_COUNT] {
        let directory = [
            DirectorySpec {
                ordinal: 0,
                family_ordinal: 0,
                family: FAMILY_TYPED_INSERT,
                rows: 2,
                request: 10,
                statement: 11,
                before: 20,
                after: 21,
            },
            DirectorySpec {
                ordinal: 1,
                family_ordinal: 0,
                family: FAMILY_CANONICAL_OPERATION,
                rows: 0,
                request: 12,
                statement: 13,
                before: 21,
                after: 22,
            },
            DirectorySpec {
                ordinal: 2,
                family_ordinal: 1,
                family: FAMILY_TYPED_INSERT,
                rows: 2,
                request: 14,
                statement: 15,
                before: 22,
                after: 23,
            },
        ];
        let dispositions = [
            DispositionSpec {
                statement_ordinal: 0,
                source_row: 0,
                stable_row_id: 100,
                kind: DISPOSITION_SURVIVES,
                block_ref: 7,
                transition_ref: 9,
                statement: 11,
            },
            DispositionSpec {
                statement_ordinal: 0,
                source_row: 1,
                stable_row_id: 101,
                kind: DISPOSITION_APPLIED_THEN_CANCELED,
                block_ref: u32::MAX,
                transition_ref: u32::MAX,
                statement: 11,
            },
            DispositionSpec {
                statement_ordinal: 2,
                source_row: 0,
                stable_row_id: 102,
                kind: DISPOSITION_SUPPRESSED_AT_STATEMENT,
                block_ref: u32::MAX,
                transition_ref: u32::MAX,
                statement: 15,
            },
            DispositionSpec {
                statement_ordinal: 2,
                source_row: 1,
                stable_row_id: 103,
                kind: DISPOSITION_SURVIVES,
                block_ref: 8,
                transition_ref: 10,
                statement: 15,
            },
        ];
        std::array::from_fn(|section| match section {
            0 => directory_payload(&directory),
            3 => dispositions_payload(&dispositions),
            _ => vec![section as u8 + 1],
        })
    }

    fn view<'a>(payloads: &'a [Vec<u8>; AGGREGATE_SECTION_COUNT]) -> TypedInsertAggregateView<'a> {
        TypedInsertAggregateView {
            flags: AGGREGATE_FLAG_EXPLICIT,
            outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            stable_transaction_id: 41,
            statement_count: 3,
            insert_statement_count: 2,
            original_inserted_row_count: 4,
            // S4 has two surviving INSERT images.  The remaining three transitions belong to
            // S7's future replace/delete closure for pre-existing rows.
            final_row_transition_count: 5,
            allocator_before: 100,
            allocator_high_water: 104,
            table_block_count: 1,
            sections: std::array::from_fn(|index| TypedInsertAggregateSectionView {
                entry_count: ENTRY_COUNTS[index],
                payload: &payloads[index],
            }),
        }
    }

    fn status(roots: TypedInsertAggregateStatusRoots) -> TypedInsertStatusV2 {
        TypedInsertStatusV2 {
            database_id: [1; 16],
            timeline_id: [2; 16],
            txn_id: 41,
            request_digest: [3; 32],
            isolation: 1,
            flags: 0,
            retention_deadline: 0,
            statement_count: 3,
            response_artifact_count: 0,
            statement_outcome_root: roots.statement_outcome_root,
            response_root: roots.response_root,
            aggregate_root: roots.aggregate_root,
        }
    }

    fn encode(payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT]) -> EncodedTypedInsertAggregateBodies {
        let view = view(payloads);
        let layout = view.measure().expect("valid aggregate fixture layout");
        let roots =
            typed_insert_aggregate_status_roots(&view, &layout).expect("valid fixture roots");
        let reserved = reserve_typed_insert_aggregate_bodies(layout).expect("reserve fixture");
        encode_typed_insert_aggregate_bodies(&view, &status(roots), reserved)
            .expect("encode fixture")
    }

    fn validate(
        payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
    ) -> Result<SemanticsV1ScaffoldSummary, EngineError> {
        let encoded = encode(payloads);
        let bodies: Vec<Vec<u8>> = encoded.fragment_bodies().map(<[u8]>::to_vec).collect();
        let refs: Vec<&[u8]> = bodies.iter().map(Vec::as_slice).collect();
        let decoded = decode_typed_insert_aggregate_bodies(
            OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            &refs,
        )?;
        validate_semantics_v1_scaffold(&decoded)
    }

    fn assert_rejected(payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT], needle: &str) {
        let error = validate(payloads).expect_err("sabotaged v1 scaffold must reject");
        assert!(
            error.to_string().contains(needle),
            "expected {needle:?}, got {error}"
        );
    }

    #[test]
    fn golden_mixed_families_and_all_row_dispositions_close_without_owned_replay_state() {
        let summary = validate(&valid_payloads()).expect("valid v1 scaffold closure");
        assert_eq!(
            summary,
            SemanticsV1ScaffoldSummary {
                statement_count: 3,
                typed_insert_count: 2,
                canonical_operation_count: 1,
                original_row_count: 4,
                insert_affected_row_count: 3,
                final_survivor_count: 2,
            }
        );
    }

    #[test]
    fn s1_rejects_per_family_ordinal_overlay_chain_and_entry_field_sabotage() {
        let mut family_ordinal = valid_payloads();
        let third_entry = 2 * STATEMENT_DIRECTORY_ENTRY_BYTES as usize;
        family_ordinal[0][third_entry + 4..third_entry + 8].copy_from_slice(&9_u32.to_le_bytes());
        assert_rejected(&family_ordinal, "typed INSERT family ordinals");

        let mut root_chain = valid_payloads();
        let second_entry = STATEMENT_DIRECTORY_ENTRY_BYTES as usize;
        root_chain[0][second_entry + 80..second_entry + 112].fill(99);
        assert_rejected(&root_chain, "overlay roots do not chain");

        let mut zero_digest = valid_payloads();
        let first_entry = 0;
        zero_digest[0][first_entry + 16..first_entry + 48].fill(0);
        assert_rejected(&zero_digest, "requires nonzero digests and overlay roots");

        let mut zero_overlay_root = valid_payloads();
        zero_overlay_root[0][first_entry + 80..first_entry + 112].fill(0);
        assert_rejected(
            &zero_overlay_root,
            "requires nonzero digests and overlay roots",
        );

        let mut family_flag = valid_payloads();
        family_flag[0][first_entry + 9] = 1;
        assert_rejected(&family_flag, "family flags or reserved bytes are nonzero");

        let mut family_reserved = valid_payloads();
        family_reserved[0][first_entry + 10..first_entry + 12]
            .copy_from_slice(&1_u16.to_le_bytes());
        assert_rejected(
            &family_reserved,
            "family flags or reserved bytes are nonzero",
        );
    }

    #[test]
    fn s1_accepts_equal_nonzero_request_and_statement_digests() {
        let mut equal_digests = valid_payloads();
        equal_digests[0][16..48].fill(11);
        assert_eq!(
            validate(&equal_digests).unwrap().insert_affected_row_count,
            3,
            "equal nonzero request and statement digests are a valid value collision"
        );
    }

    #[test]
    fn s4_rejects_allocator_source_reference_and_entry_field_sabotage() {
        let mut allocator_gap = valid_payloads();
        let third_row = 2 * ROW_DISPOSITION_ENTRY_BYTES as usize;
        allocator_gap[3][third_row + 8..third_row + 16].copy_from_slice(&104_u64.to_le_bytes());
        assert_rejected(
            &allocator_gap,
            "canonical statement/source/digest/allocator order",
        );

        let mut source_reorder = valid_payloads();
        let first_row = 0;
        source_reorder[3][first_row + 4..first_row + 8].copy_from_slice(&1_u32.to_le_bytes());
        assert_rejected(
            &source_reorder,
            "canonical statement/source/digest/allocator order",
        );

        let mut canceled_refs = valid_payloads();
        let second_row = ROW_DISPOSITION_ENTRY_BYTES as usize;
        canceled_refs[3][second_row + 20..second_row + 24].copy_from_slice(&7_u32.to_le_bytes());
        assert_rejected(&canceled_refs, "canceled or suppressed row");

        let mut partial_survivor_refs = valid_payloads();
        partial_survivor_refs[3][first_row + 24..first_row + 28]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert_rejected(
            &partial_survivor_refs,
            "surviving row must carry non-sentinel",
        );

        let mut leading_flag = valid_payloads();
        leading_flag[3][first_row + 17] = 1;
        assert_rejected(
            &leading_flag,
            "row-disposition flags or reserved bytes are nonzero",
        );

        let mut leading_reserved = valid_payloads();
        leading_reserved[3][first_row + 18..first_row + 20].copy_from_slice(&1_u16.to_le_bytes());
        assert_rejected(
            &leading_reserved,
            "row-disposition flags or reserved bytes are nonzero",
        );

        let mut trailing_reserved = valid_payloads();
        trailing_reserved[3][first_row + 28..first_row + 32].copy_from_slice(&1_u32.to_le_bytes());
        assert_rejected(&trailing_reserved, "trailing reserved field is nonzero");

        let mut statement_digest = valid_payloads();
        statement_digest[3][first_row + 32..first_row + 64].fill(12);
        assert_rejected(
            &statement_digest,
            "canonical statement/source/digest/allocator order",
        );
    }

    #[test]
    fn payloads_require_exact_length_for_their_outer_entry_counts() {
        // `validate` remeasures and re-encodes each altered section, so its physical aggregate
        // geometry/root remains valid and the direct outer-count length guard is reached.
        let mut short_s1 = valid_payloads();
        short_s1[0].pop();
        assert_rejected(
            &short_s1,
            "S1 payload length is not exact for its outer entry count",
        );

        let mut surplus_s1 = valid_payloads();
        surplus_s1[0].push(0);
        assert_rejected(
            &surplus_s1,
            "S1 payload length is not exact for its outer entry count",
        );

        let mut short_s4 = valid_payloads();
        short_s4[3].pop();
        assert_rejected(
            &short_s4,
            "S4 payload length is not exact for its outer entry count",
        );

        let mut surplus_s4 = valid_payloads();
        surplus_s4[3].push(0);
        assert_rejected(
            &surplus_s4,
            "S4 payload length is not exact for its outer entry count",
        );
    }

    #[test]
    fn s4_duplicate_or_sparse_survivor_references_remain_explicitly_deferred_to_s7() {
        let mut payloads = valid_payloads();
        let fourth_row = 3 * ROW_DISPOSITION_ENTRY_BYTES as usize;
        // Reuse the first survivor's references.  S4 can prove neither the target image set nor a
        // unique reference bijection without owning S7; the full replay validator must reject it.
        payloads[3][fourth_row + 20..fourth_row + 24].copy_from_slice(&7_u32.to_le_bytes());
        payloads[3][fourth_row + 24..fourth_row + 28].copy_from_slice(&9_u32.to_le_bytes());
        assert_eq!(validate(&payloads).unwrap().final_survivor_count, 2);

        let mut sparse = valid_payloads();
        sparse[3][fourth_row + 20..fourth_row + 24].copy_from_slice(&999_u32.to_le_bytes());
        sparse[3][fourth_row + 24..fourth_row + 28].copy_from_slice(&1000_u32.to_le_bytes());
        assert_eq!(validate(&sparse).unwrap().final_survivor_count, 2);
    }

    #[test]
    fn v1_scaffold_validation_crosses_a_chunk_boundary_without_assembling_s1_or_s4() {
        let mut payloads = valid_payloads();
        let base = view(&payloads).measure().unwrap();
        let target = AGGREGATE_CHUNK_PAYLOAD_BYTES - 12;
        assert!(base.sections[3].payload_offset < target);
        let extra = usize::try_from(target - base.sections[3].payload_offset).unwrap();
        payloads[1].resize(payloads[1].len() + extra, 0xa5);
        let encoded = encode(&payloads);
        assert!(encoded.layout().chunk_count >= 2);
        assert_eq!(validate(&payloads).unwrap().final_survivor_count, 2);
    }

    #[test]
    fn validator_source_uses_the_bounded_reader_and_no_owned_replay_collection() {
        let source = include_str!("semantics_v1_scaffold.rs");
        let validator = source
            .split("fn validate_semantics_v1_scaffold")
            .nth(1)
            .expect("validator exists")
            .split("fn validate_payload_length")
            .next()
            .expect("validator precedes payload-length helper");
        assert!(validator.contains("with_section_reader"));
        for forbidden in [
            "Vec<", "Vec::", "HashMap", "BTreeMap", "collect(", "to_vec(",
        ] {
            assert!(
                !validator.contains(forbidden),
                "validator built owned replay state through {forbidden}"
            );
        }
        assert!(source.contains("S7's eventual authority"));
        assert!(source.contains("bind each typed statement to S2"));
    }
}
