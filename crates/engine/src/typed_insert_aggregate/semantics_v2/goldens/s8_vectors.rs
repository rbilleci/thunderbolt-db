//! S8 retained-response evidence built from the frozen Q1 source shapes.
//!
//! This is a test-only wire constructor.  It never decodes an accepted retained owner back into
//! bytes and has no live writer, WAL recovery, apply, GPU, or publication edge.

use super::s8_literals::{
    S8EnvelopeLiteral, B_ENVELOPE_LITERAL, C_ENVELOPE_LITERAL, D_ENVELOPE_LITERAL,
};
use super::{fill_canonical_semantics_v2_for_test, measure_canonical_semantics_v2, q1_vectors};
use crate::insert_semantic_ir::InsertStatementOrdinal;
use crate::typed_insert_aggregate::{
    encode_status_v2, TypedInsertStatusV2, AGGREGATE_CHUNK_FLAG_FIRST, AGGREGATE_CHUNK_FLAG_LAST,
    AGGREGATE_CHUNK_HEADER_BYTES, AGGREGATE_CHUNK_MAGIC, AGGREGATE_CHUNK_PAYLOAD_BYTES,
    AGGREGATE_FLAG_RETAINED_RESPONSE, AGGREGATE_FORMAT_VERSION, AGGREGATE_SECTION_COUNT,
    AGGREGATE_SECTION_HEADER_BYTES, AGGREGATE_STATUS_V2_BYTES,
    ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE,
};
use crate::typed_insert_batch::{
    decode_canonical_typed_insert_record, decode_typed_image,
    encode_canonical_typed_insert_record_for_test, encode_typed_image,
    prepare_typed_insert_semantics_at, sequence_defaults, DecodedSequenceEffectKindFacts,
    TypedImageColumnView, TypedImageRole, TypedImageView, TypedInsertColumnValidity,
    TypedInsertColumnValues,
};
use crate::{encode_sequence_value_reference_into_exact, BinarySequenceValueReference};
use sha2::{Digest, Sha256};

const S6_BYTES: usize = 136;
const S7_HEADER_BYTES: usize = 640;
const S7_RESOLUTION_BYTES: usize = 320;
const S7_PROJECTION_BYTES: usize = 128;
const S8_HEADER_BYTES: usize = 256;
const S8_ARTIFACT_BYTES: usize = 288;
const S8_SELECTION_BYTES: usize = 32;

#[test]
fn b_c_d_response_envelopes_match_independently_captured_literals() {
    assert_envelope_literal(
        "B",
        &retained_explicit_abort_final_returning_fixture(),
        &B_ENVELOPE_LITERAL,
    );
    assert_envelope_literal("C", &selective_a_b_a_fixture(), &C_ENVELOPE_LITERAL);
    assert_envelope_literal("D", &zero_selection_fixture(), &D_ENVELOPE_LITERAL);
}

fn assert_envelope_literal(
    label: &str,
    fixture: &q1_vectors::Q1Fixture,
    expected: &S8EnvelopeLiteral,
) {
    fn hex(bytes: &[u8]) -> String {
        use std::fmt::Write as _;

        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }

    let s8 = &fixture.sections[7];
    assert_eq!(s8.len(), expected.s8_bytes, "{label} S8 length");
    assert_eq!(
        hex(&Sha256::digest(s8)),
        expected.s8_sha256,
        "{label} S8 SHA-256"
    );
    assert_eq!(
        hex(&s8[208..240]),
        expected.s8_payload_digest,
        "{label} S8 payload digest"
    );
    assert_eq!(
        hex(&fixture.section_roots[7]),
        expected.s8_section_root,
        "{label} S8 section root"
    );
    assert_eq!(
        hex(&fixture.outcome.returning_digest),
        expected.response_root,
        "{label} response root"
    );
    assert_eq!(
        hex(&fixture.aggregate_root),
        expected.aggregate_root,
        "{label} aggregate root"
    );
    assert_eq!(
        hex(&fixture.status),
        expected.status_hex,
        "{label} exact STATUS2 bytes"
    );
    assert_eq!(
        read_u64(&fixture.status, 92),
        expected.status_deadline,
        "{label} retention deadline"
    );
    assert_eq!(
        read_u32(&fixture.status, 104),
        expected.status_artifact_count,
        "{label} artifact count"
    );
}

#[test]
fn c_selective_a_b_a_retention_reaches_pending_with_exact_s8_owners() {
    let fixture = selective_a_b_a_fixture();
    let fragments = fragments(&fixture);
    measure_canonical_semantics_v2(&fixture.outer, &fixture.outcome, &fragments)
        .expect("selective retained A/B/A fixture passes the complete raw proof");

    // This is the production shell, not Q2's deliberately empty-S8 adapter.  It proves that a
    // coherent successful statement omitted from S8 remains locally accepted and advances only
    // to RetentionAuthorityPending, where a future claim proof owns eligibility completeness.
    fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
        .and_then(|quarantined| quarantined.close_codec())
        .expect("selective locally coherent omission reaches retention-authority pending");

    let s6_flags = [
        read_u16(&fixture.sections[5], 10),
        read_u16(&fixture.sections[5], S6_BYTES + 10),
        read_u16(&fixture.sections[5], 2 * S6_BYTES + 10),
    ];
    assert_eq!(s6_flags, [3, 1, 3]);
    let (resolution_start, _) = s7_region(&fixture.sections[6], 2);
    for (statement, expected) in [3_u32, 1, 3].into_iter().enumerate() {
        assert_eq!(
            read_u32(
                &fixture.sections[6],
                resolution_start + statement * S7_RESOLUTION_BYTES + 20,
            ),
            expected,
            "S7 statement {statement} has the exact retained-response flags"
        );
    }
    assert_eq!(read_u32(&fixture.status, 104), 2, "STATUS2 artifact count");
    assert_eq!(
        read_u64(&fixture.status, 92),
        900,
        "STATUS2 retention deadline"
    );
}

#[test]
fn c_selective_a_b_a_source_copy_failures_drop_every_partial_s8_owner() {
    let fixture = selective_a_b_a_fixture();
    let fragments = fragments(&fixture);
    let (initial, attempts) = super::super::observe_retained_source_copy_attempts_for_test(|| {
        fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
    });
    initial.expect("the C fixture establishes its source-copy attempt count");
    assert!(attempts != 0, "C includes strict S2/S7/S8 source copies");

    for attempt in 1..=attempts {
        let failed = super::super::fail_retained_source_copy_at_for_test(attempt, || {
            fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
        });
        assert!(failed.is_err(), "source-copy attempt {attempt} must fail");
        fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
            .and_then(|quarantined| quarantined.close_codec())
            .unwrap_or_else(|error| {
                panic!("source-copy attempt {attempt} leaves a partial C owner: {error}")
            });
    }
}

#[test]
fn b_explicit_abort_retains_earlier_applied_then_canceled_returning() {
    let fixture = retained_explicit_abort_final_returning_fixture();
    let fragments = fragments(&fixture);
    let measure = measure_canonical_semantics_v2(&fixture.outer, &fixture.outcome, &fragments)
        .expect("explicit abort with an earlier retained response passes raw proof");
    assert_eq!(
        measure.terminal_kind,
        gpu_db_wal::CanonicalOutcomeKind::AbortError,
        "the final statement remains the terminal abort"
    );
    fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
        .and_then(|quarantined| quarantined.close_codec())
        .expect("the earlier statement-visible response reaches retention-authority pending");

    assert_eq!(read_u16(&fixture.sections[5], 10), 3);
    assert_eq!(read_u16(&fixture.sections[5], S6_BYTES + 10), 1);
    let (resolution_start, _) = s7_region(&fixture.sections[6], 2);
    assert_eq!(
        read_u32(
            &fixture.sections[6],
            resolution_start + S7_RESOLUTION_BYTES + 20
        ),
        1,
        "the final failing RETURNING statement keeps only S7 bit 0"
    );
    assert_eq!(
        read_u32(
            &fixture.sections[6],
            resolution_start + S7_RESOLUTION_BYTES + 52
        ),
        1,
        "the final failing statement retains its one S7 projection"
    );
    assert_eq!(
        &fixture.sections[5][S6_BYTES + 104..S6_BYTES + 136],
        &[0; 32],
        "the final failing RETURNING has no S6 logical result"
    );
    assert_eq!(
        fixture.sections[3][16], 2,
        "the retained response selects the earlier AppliedThenCanceled S4 row"
    );
    assert_eq!(read_u32(&fixture.status, 104), 1);
    assert_eq!(read_u64(&fixture.status, 92), 900);
    assert_eq!(
        read_u32(&fixture.sections[7], S8_HEADER_BYTES + 4),
        0,
        "the only S8 artifact names the earlier success, never the final failure"
    );
}

#[test]
fn b_explicit_abort_cross_chunk_s8_owners_pass_full_inert_close() {
    let base = large_explicit_abort_fixture(0);
    let plus_one = large_explicit_abort_fixture(1);
    let base_s8 = s8_stream_offset(&base.stream);
    assert_eq!(
        s8_stream_offset(&plus_one.stream),
        base_s8 + 1,
        "the abort-only S2 padding moves the retained S8 start one byte at a time"
    );

    for (label, relative) in s8_cross_chunk_ranges(&base) {
        let pad = usize::try_from(AGGREGATE_CHUNK_PAYLOAD_BYTES)
            .expect("canonical chunk payload fits usize")
            .checked_sub(base_s8 + relative.start + 1)
            .expect("baseline retained S8 starts below the canonical chunk boundary");
        let fixture = large_explicit_abort_fixture(pad);
        let chunks = canonical_chunks(&fixture);
        assert_eq!(chunks.len(), 2, "{label} has exactly two canonical chunks");
        assert_eq!(
            chunks[0].len(),
            usize::try_from(gpu_db_wal::canonical_fragment_body_limit())
                .expect("fragment limit fits usize"),
            "{label} fills the first canonical fragment exactly"
        );
        let boundary = usize::try_from(AGGREGATE_CHUNK_PAYLOAD_BYTES)
            .expect("canonical chunk payload fits usize");
        let start = s8_stream_offset(&fixture.stream) + relative.start;
        let end = s8_stream_offset(&fixture.stream) + relative.end;
        assert!(
            start < boundary && boundary < end,
            "canonical chunk boundary crosses the real S8 {label} owner"
        );
        let mut fragments: Vec<_> = chunks
            .iter()
            .map(|body| gpu_db_wal::CanonicalFragmentRef {
                kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
                body,
            })
            .collect();
        fragments.push(gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        });
        measure_canonical_semantics_v2(&fixture.outer, &fixture.outcome, &fragments)
            .expect("cross-chunk B source passes production pass zero");
        fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
            .and_then(|quarantined| quarantined.close_codec())
            .unwrap_or_else(|error| panic!("cross-chunk B {label} reaches inert pending: {error}"));
    }
}

#[test]
fn d_zero_selection_zero_row_response_is_standalone_valid_but_aggregate_rejected() {
    let fixture = zero_selection_fixture();
    let reference = q1_vectors::successful_a_b_a_fixture();
    let reference_fragments = fragments(&reference);
    let candidate_fragments = fragments(&fixture);
    let s8 = &fixture.sections[7];
    let measure = super::super::measure_s8_against_reference_for_test(
        &reference.outer,
        &reference_fragments,
        &fixture.outer,
        &candidate_fragments,
    )
    .expect("the production S8 artifact proof accepts the real zero-selection framing");
    assert!(measure.identity.present);
    assert_eq!(measure.identity.selection_count, 0);
    let image_offset = usize::try_from(read_u64(s8, 88)).expect("D image offset fits usize");
    let image = decode_typed_image(&s8[image_offset..]).expect("D zero-row image decodes");
    assert_eq!(image.facts().role, TypedImageRole::RetainedResponse);
    assert_eq!(image.facts().rows, 0);
    assert_eq!(read_u32(s8, 44), 0, "D has no row-selection directory");
    assert_eq!(read_u32(s8, S8_HEADER_BYTES + 20), 0);
    assert_eq!(read_u32(s8, S8_HEADER_BYTES + 32), 0);

    let rejected =
        measure_canonical_semantics_v2(&fixture.outer, &fixture.outcome, &candidate_fragments)
            .expect_err("the complete aggregate rejects S4/S7 mismatch outside the S8-only seam");
    assert!(
        rejected
            .to_string()
            .contains("S4/S1 statement/source/digest/reference closure is invalid"),
        "D's complete proof rejects the deliberately inconsistent S4 aggregate: {rejected}"
    );
}

pub(super) fn selective_a_b_a_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let (resolution_start, resolution_bytes) = s7_region(&sections[6], 2);
    assert_eq!(resolution_bytes, 3 * S7_RESOLUTION_BYTES);

    for statement in [0_u32, 2] {
        mark_response_retained(&mut sections, resolution_start, statement);
    }
    refresh_s7_payload_digest(&mut sections[6]);

    let section_counts = [3, 3, 0, 3, 1, 3, 1, 2];
    let s6_root = section_root(5, section_counts[5], &sections[5]);
    let s7_root = section_root(6, section_counts[6], &sections[6]);
    sections[7] = encode_s8(
        &sections,
        s6_root,
        s7_root,
        base.outer.stable_transaction_id,
        base.outer.request_digest,
        &[0, 2],
        ResponseRows::SelectedOne,
    );

    reframe_retained_fixture(base, sections, section_counts, 900)
}

fn zero_selection_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    sections[3][16] = 3; // failed-noop: no S4 row in statement zero qualifies for S8 selection.
    let (resolution_start, resolution_bytes) = s7_region(&sections[6], 2);
    assert_eq!(resolution_bytes, 3 * S7_RESOLUTION_BYTES);
    mark_response_retained(&mut sections, resolution_start, 0);
    refresh_s7_payload_digest(&mut sections[6]);

    let section_counts = [3, 3, 0, 3, 1, 3, 1, 1];
    let s6_root = section_root(5, section_counts[5], &sections[5]);
    let s7_root = section_root(6, section_counts[6], &sections[6]);
    sections[7] = encode_s8(
        &sections,
        s6_root,
        s7_root,
        base.outer.stable_transaction_id,
        base.outer.request_digest,
        &[0],
        ResponseRows::Zero,
    );
    reframe_retained_fixture(base, sections, section_counts, 900)
}

fn retained_explicit_abort_final_returning_fixture() -> q1_vectors::Q1Fixture {
    retained_explicit_abort_final_returning_fixture_from(
        q1_vectors::explicit_abort_fixture(),
        final_returning_record(),
    )
}

fn retained_explicit_abort_final_returning_fixture_from(
    mut base: q1_vectors::Q1Fixture,
    final_bytes: Vec<u8>,
) -> q1_vectors::Q1Fixture {
    let mut sections = base.sections.clone();
    let final_record = decode_canonical_typed_insert_record(&final_bytes)
        .expect("final failing RETURNING S2 record decodes");
    let final_facts = final_record.facts();

    let first_record_bytes = s2_record_bytes(&sections[1], 0).to_vec();
    sections[1].clear();
    append_s2_record(&mut sections[1], &first_record_bytes);
    append_s2_record(&mut sections[1], &final_bytes);

    let final_s1 = &mut sections[0][144..288];
    final_s1[16..48].copy_from_slice(&final_facts.typed_statement_digest);
    final_s1[48..80].copy_from_slice(&final_facts.typed_statement_digest);
    sections[3][64 + 32..64 + 64].copy_from_slice(&final_facts.typed_statement_digest);

    let (resolution_start, resolution_bytes) = s7_region(&sections[6], 2);
    assert_eq!(resolution_bytes, 2 * S7_RESOLUTION_BYTES);
    let (sequence_entry, sequence_token) =
        final_sequence_entry_and_token(&final_record, read_u64(&sections[3], 64 + 8), &sections[6]);
    sections[4] = sequence_entry;

    let dependency_start = s7_region(&sections[6], 3).0;
    sections[6][dependency_start + 224..dependency_start + 448].copy_from_slice(&sequence_token);
    let dependency_digests = std::array::from_fn(|ordinal| {
        let start = dependency_start + ordinal * 224 + 192;
        sections[6][start..start + 32]
            .try_into()
            .expect("explicit-abort dependency digest width")
    });
    let use_start = s7_region(&sections[6], 4).0;
    let final_uses = &sections[6][use_start + 2 * 32..use_start + 5 * 32];
    let final_dependency_root = statement_dependency_root(1, final_uses, &dependency_digests);
    let final_disposition_root = statement_disposition_root(1, &sections[3][64..128]);
    let final_sequence_root = statement_sequence_root(1, &sections[4]);
    let final_projection = final_projection_bytes(&final_record, 1);
    let final_projection_digest: [u8; 32] = final_projection[96..128]
        .try_into()
        .expect("final RETURNING projection digest width");
    let final_projection_root = projection_root_from_digests(1, &[final_projection_digest]);
    let final_before: [u8; 32] = sections[0][144 + 80..144 + 112]
        .try_into()
        .expect("final S1 overlay-before width");
    let final_record_digest = s2_record_digest(&final_bytes);
    let final_overlay = overlay_root(
        final_before,
        1,
        final_facts.typed_statement_digest,
        final_record_digest,
        [
            final_disposition_root,
            final_sequence_root,
            final_dependency_root,
            final_projection_root,
        ],
    );
    sections[0][144 + 112..144 + 144].copy_from_slice(&final_overlay);

    let final_s6 = &mut sections[5][S6_BYTES..2 * S6_BYTES];
    write_u16(final_s6, 10, 1);
    final_s6[12..44].copy_from_slice(&final_facts.typed_statement_digest);
    let mut final_outcome = gpu_db_wal::decode_canonical_outcome_exact(&final_s6[44..136])
        .expect("existing final abort outcome decodes");
    final_outcome.target_digest = final_overlay;
    final_outcome.returning_digest = [0; 32];
    let mut final_outcome_bytes = [0; gpu_db_wal::CANONICAL_OUTCOME_BYTES];
    gpu_db_wal::encode_canonical_outcome_into_exact(&final_outcome, &mut final_outcome_bytes)
        .expect("final abort outcome reencodes");
    final_s6[44..136].copy_from_slice(&final_outcome_bytes);
    let final_s6_digest = digest(b"gpu-db/write001/s7-s6-entry/v2", &[final_s6]);

    let final_resolution = resolution_start + S7_RESOLUTION_BYTES;
    write_u32(&mut sections[6], final_resolution + 20, 1);
    write_u32(&mut sections[6], final_resolution + 52, 1);
    write_u32(
        &mut sections[6],
        final_resolution + 80,
        u32::try_from(final_bytes.len()).expect("final S2 byte length fits u32"),
    );
    sections[6][final_resolution + 96..final_resolution + 128]
        .copy_from_slice(&final_facts.typed_statement_digest);
    sections[6][final_resolution + 128..final_resolution + 160]
        .copy_from_slice(&final_facts.typed_statement_digest);
    sections[6][final_resolution + 160..final_resolution + 192]
        .copy_from_slice(&final_record_digest);
    sections[6][final_resolution + 192..final_resolution + 224]
        .copy_from_slice(&final_facts.returning.digest);
    sections[6][final_resolution + 256..final_resolution + 288].copy_from_slice(&final_overlay);
    sections[6][final_resolution + 288..final_resolution + 320].copy_from_slice(&final_s6_digest);

    append_final_projection_and_reframe_s7(&mut sections[6], &final_projection, final_overlay);
    let (resolution_start, resolution_bytes) = s7_region(&sections[6], 2);
    assert_eq!(resolution_bytes, 2 * S7_RESOLUTION_BYTES);
    mark_response_retained(&mut sections, resolution_start, 0);
    refresh_s7_payload_digest(&mut sections[6]);

    let section_counts = [2, 2, 0, 2, 1, 2, 1, 1];
    let s6_root = section_root(5, section_counts[5], &sections[5]);
    let s7_root = section_root(6, section_counts[6], &sections[6]);
    sections[7] = encode_s8(
        &sections,
        s6_root,
        s7_root,
        base.outer.stable_transaction_id,
        aggregate_request_digest(&sections[0]),
        &[0],
        ResponseRows::SelectedOne,
    );
    base.outer.request_digest = aggregate_request_digest(&sections[0]);
    reframe_retained_fixture(base, sections, section_counts, 900)
}

fn mark_response_retained(
    sections: &mut [Vec<u8>; AGGREGATE_SECTION_COUNT],
    resolution_start: usize,
    statement: u32,
) {
    let statement = usize::try_from(statement).expect("fixture statement ordinal fits usize");
    let s6 = &mut sections[5][statement * S6_BYTES..(statement + 1) * S6_BYTES];
    assert_ne!(
        read_u16(s6, 10) & 1,
        0,
        "retained response requires RETURNING"
    );
    write_u16(s6, 10, 3);
    let entry_digest = digest(b"gpu-db/write001/s7-s6-entry/v2", &[s6]);
    let resolution = resolution_start + statement * S7_RESOLUTION_BYTES;
    write_u32(&mut sections[6], resolution + 20, 3);
    sections[6][resolution + 288..resolution + 320].copy_from_slice(&entry_digest);
}

fn final_returning_record() -> Vec<u8> {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE q1_abort (id serial, required int4)")
        .expect("B fixture table creates");
    canonical_insert_record(
        &engine,
        "INSERT INTO q1_abort (required) VALUES (NULL) RETURNING id",
        1,
        true,
    )
}

fn canonical_insert_record(
    engine: &crate::Engine,
    sql: &str,
    statement: u32,
    published_sequence: bool,
) -> Vec<u8> {
    let crate::Command::Insert(insert) =
        crate::parse_command(sql).expect("B fixture INSERT parses")
    else {
        panic!("B fixture final command must be INSERT");
    };
    canonical_insert_from_ast(engine, &insert, statement, published_sequence)
}

fn canonical_insert_from_ast(
    engine: &crate::Engine,
    insert: &crate::Insert,
    statement: u32,
    published_sequence: bool,
) -> Vec<u8> {
    let catalog = engine.catalog_snapshot();
    let ordinal = InsertStatementOrdinal::from_u32(statement);
    let prepared =
        prepare_typed_insert_semantics_at(insert, &catalog, catalog.commit_seq, None, ordinal)
            .expect("B fixture final semantic preparation succeeds")
            .expect("B fixture final target is current");
    let typed_digest = prepared.typed_statement_digest();
    let bindings = if published_sequence {
        let parent = sequence_defaults::effects::SequenceDefaultParentContext::for_test(
            177,
            false,
            typed_digest,
            ordinal,
            0,
        );
        let bindings = prepared
            .sequence_requests()
            .iter()
            .cloned()
            .map(|request| {
                sequence_defaults::SequenceDefaultBinding::published(request, parent.clone(), 41)
            })
            .collect();
        sequence_defaults::SequenceDefaultBindings::from_bindings(parent, bindings)
    } else {
        sequence_defaults::SequenceDefaultBindings::empty()
    };
    let batch = prepared
        .seal(bindings, false, false)
        .expect("B fixture final typed batch seals");
    encode_canonical_typed_insert_record_for_test(&batch)
        .expect("B fixture final canonical S2 record encodes")
}

fn large_explicit_abort_fixture(pad_bytes: usize) -> q1_vectors::Q1Fixture {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE q1_abort (id serial, required int4, padding text)",
        )
        .expect("large B fixture table creates");
    let template = crate::PreparedCommand::parse(
        "INSERT INTO q1_abort (id, required, padding) VALUES (1, 7, $1) RETURNING id",
    )
    .expect("large B first prepared INSERT parses");
    let bound = template
        .bind(&[crate::SqlValue::Text("x".repeat(pad_bytes))])
        .expect("large B first parameter binds into the parsed INSERT AST");
    let crate::Command::Insert(first_insert) = bound.command() else {
        panic!("large B first bound command must remain INSERT");
    };
    let first = canonical_insert_from_ast(&engine, first_insert, 0, false);
    let aborted = canonical_insert_record(
        &engine,
        "INSERT INTO q1_abort (required) VALUES (NULL)",
        1,
        true,
    );
    let final_returning = canonical_insert_record(
        &engine,
        "INSERT INTO q1_abort (required) VALUES (NULL) RETURNING id",
        1,
        true,
    );
    let mut fixture = retained_explicit_abort_final_returning_fixture_from(
        q1_vectors::explicit_abort_fixture_from_records_for_test(first, aborted),
        final_returning,
    );
    fixture.outer.operation_count = 3;
    fixture
}

fn s8_cross_chunk_ranges(
    fixture: &q1_vectors::Q1Fixture,
) -> [(&'static str, std::ops::Range<usize>); 6] {
    let s8 = &fixture.sections[7];
    let artifact = S8_HEADER_BYTES..S8_HEADER_BYTES + S8_ARTIFACT_BYTES;
    let selection = usize::try_from(read_u64(s8, 72)).expect("S8 selection offset fits usize");
    let image = usize::try_from(read_u64(s8, 88)).expect("S8 image offset fits usize");
    let descriptors =
        usize::try_from(read_u64(s8, image + 40)).expect("typed image descriptor bytes fit usize");
    let names =
        usize::try_from(read_u64(s8, image + 48)).expect("typed image name bytes fit usize");
    let vectors =
        usize::try_from(read_u64(s8, image + 56)).expect("typed image vector bytes fit usize");
    let packed_names = image + 112 + descriptors;
    let value_vectors = packed_names + names;
    assert_ne!(names, 0, "retained id response keeps a packed column name");
    assert_ne!(vectors, 0, "retained id response keeps a value vector");
    [
        ("header", 0..S8_HEADER_BYTES),
        ("artifact descriptor", artifact),
        ("selection", selection..selection + S8_SELECTION_BYTES),
        ("role-2 image header", image..image + 112),
        ("packed name", packed_names..packed_names + names),
        ("value vector", value_vectors..value_vectors + vectors),
    ]
}

fn s8_stream_offset(stream: &[u8]) -> usize {
    let mut cursor = 96_usize;
    for section in 0..AGGREGATE_SECTION_COUNT {
        let header = &stream[cursor..cursor + AGGREGATE_SECTION_HEADER_BYTES as usize];
        let bytes = usize::try_from(read_u64(header, 8)).expect("section bytes fit usize");
        cursor += AGGREGATE_SECTION_HEADER_BYTES as usize;
        if section == 7 {
            return cursor;
        }
        cursor += bytes;
    }
    unreachable!("aggregate has S8 section")
}

fn canonical_chunks(fixture: &q1_vectors::Q1Fixture) -> Vec<Vec<u8>> {
    let stream = &fixture.stream;
    let payload =
        usize::try_from(AGGREGATE_CHUNK_PAYLOAD_BYTES).expect("canonical chunk payload fits usize");
    let chunk_count = stream.len().div_ceil(payload);
    assert_eq!(
        chunk_count, 2,
        "cross-chunk fixture stays within two fragments"
    );
    let mut chunks = Vec::with_capacity(chunk_count);
    for ordinal in 0..chunk_count {
        let start = ordinal * payload;
        let end = (start + payload).min(stream.len());
        let mut body = vec![0; AGGREGATE_CHUNK_HEADER_BYTES as usize];
        body[..8].copy_from_slice(AGGREGATE_CHUNK_MAGIC);
        body[8] = ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE;
        body[9] = AGGREGATE_FORMAT_VERSION as u8;
        let mut flags = 0_u16;
        if ordinal == 0 {
            flags |= AGGREGATE_CHUNK_FLAG_FIRST;
        }
        if ordinal + 1 == chunk_count {
            flags |= AGGREGATE_CHUNK_FLAG_LAST;
        }
        write_u16(&mut body, 10, flags);
        write_u64(&mut body, 12, stream.len() as u64);
        write_u32(&mut body, 20, ordinal as u32);
        write_u32(&mut body, 24, chunk_count as u32);
        write_u64(&mut body, 28, start as u64);
        write_u32(&mut body, 36, (end - start) as u32);
        body[44..76].copy_from_slice(&fixture.aggregate_root);
        body.extend_from_slice(&stream[start..end]);
        chunks.push(body);
    }
    chunks
}

fn s2_record_bytes(s2: &[u8], ordinal: usize) -> &[u8] {
    let mut offset = 0_usize;
    for current in 0..=ordinal {
        let bytes = usize::try_from(read_u32(s2, offset)).expect("S2 record length fits usize");
        offset = offset.checked_add(4).expect("S2 record prefix offset fits");
        let end = offset.checked_add(bytes).expect("S2 record end fits");
        let record = &s2[offset..end];
        if current == ordinal {
            return record;
        }
        offset = end;
    }
    unreachable!("bounded S2 record lookup returns its requested ordinal")
}

fn append_s2_record(out: &mut Vec<u8>, record: &[u8]) {
    let bytes = u32::try_from(record.len()).expect("B S2 record length fits u32");
    out.extend_from_slice(&bytes.to_le_bytes());
    out.extend_from_slice(record);
}

fn final_sequence_entry_and_token(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    stable_row_id: u64,
    s7: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let facts = record.facts();
    let parent = record
        .sequence_parent()
        .expect("B final published sequence has a parent");
    let effect = record
        .sequence_effects()
        .next()
        .expect("B final published sequence has one effect");
    let DecodedSequenceEffectKindFacts::Published {
        transition_txn_id,
        input_digest,
        returned_value,
    } = effect.kind
    else {
        panic!("B final sequence effect is published");
    };
    record
        .sequence_bindings()
        .next()
        .expect("B final sequence binding exists");
    let reference = BinarySequenceValueReference {
        transition_txn_id,
        parent_txn_id: parent.txn_id,
        statement_ordinal: 1,
        expression_ordinal: effect.request.absolute_expression_ordinal,
        sequence_oid: effect.request.sequence_oid,
        returned_value,
        input_digest,
        table_oid: effect.request.target_table_oid,
        column_id: effect.request.column_id,
        staging_row_ordinal: 0,
        row_id: stable_row_id,
        final_value_overwritten: true,
        default_expression: true,
    };
    let mut body = [0; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
    encode_sequence_value_reference_into_exact(&reference, &mut body)
        .expect("B sequence reference encodes");
    let body_digest = gpu_db_wal::canonical_request_digest(&body);
    let mut entry = Vec::with_capacity(52 + body.len());
    entry.extend_from_slice(&1_u32.to_le_bytes());
    entry.extend_from_slice(&0_u32.to_le_bytes());
    entry.push(1);
    entry.push(3);
    entry.extend_from_slice(&0_u16.to_le_bytes());
    entry.extend_from_slice(&1_u32.to_le_bytes());
    entry.extend_from_slice(&(body.len() as u32).to_le_bytes());
    entry.extend_from_slice(&body_digest);
    entry.extend_from_slice(&body);

    let dependency_start = s7_region(s7, 3).0;
    let sequence_offset = dependency_start + 224;
    let mut token = s7[sequence_offset..sequence_offset + 224].to_vec();
    let stable_id = read_u64(&token, 8);
    let epoch = read_u64(&token, 48);
    let name_digest: [u8; 32] = token[128..160]
        .try_into()
        .expect("B sequence name digest width");
    token[96..128].copy_from_slice(&body_digest);
    let identity = digest(
        b"gpu-db/write001/s7-published-sequence/v2",
        &[
            &stable_id.to_le_bytes(),
            &effect.request.sequence_oid.to_le_bytes(),
            &epoch.to_le_bytes(),
            &transition_txn_id.to_le_bytes(),
            &name_digest,
            &body_digest,
            &body,
        ],
    );
    token[160..192].copy_from_slice(&identity);
    let token_digest = digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&token[..192], &[0; 32], &[0; 32], &[0; 32]],
    );
    token[192..224].copy_from_slice(&token_digest);
    assert_eq!(facts.row_count, 1, "B final source remains one row");
    (entry, token)
}

fn final_projection_bytes(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    projection_ref: u32,
) -> [u8; S7_PROJECTION_BYTES] {
    let source = record
        .returning_projections()
        .next()
        .expect("B final failing record has one RETURNING projection");
    assert!(
        record.returning_projections().nth(1).is_none(),
        "B final failing record has exactly one RETURNING projection"
    );
    let mut raw = [0_u8; S7_PROJECTION_BYTES];
    write_u32(&mut raw, 0, projection_ref);
    write_u32(&mut raw, 4, 1);
    write_u32(&mut raw, 8, 0);
    write_u32(&mut raw, 12, source.catalog_column_ordinal);
    write_u32(&mut raw, 16, source.column_id);
    write_u32(&mut raw, 20, 0);
    raw[24..26].copy_from_slice(&source.attnum.to_le_bytes());
    raw[28..32].copy_from_slice(&storage(source.ty));
    write_u32(&mut raw, 32, source.type_oid);
    raw[36..38].copy_from_slice(&source.type_size.to_le_bytes());
    write_u16(&mut raw, 38, 0);
    write_u32(&mut raw, 40, 0);
    raw[64..96].copy_from_slice(&identifier_digest(source.name));
    let projection_digest = digest(b"gpu-db/write001/s7-projection/v2", &[&raw[..96], &[0; 32]]);
    raw[96..128].copy_from_slice(&projection_digest);
    raw
}

fn append_final_projection_and_reframe_s7(
    s7: &mut Vec<u8>,
    projection: &[u8; S7_PROJECTION_BYTES],
    final_overlay: [u8; 32],
) {
    let (projection_start, projection_bytes) = s7_region(s7, 10);
    assert_eq!(projection_bytes, S7_PROJECTION_BYTES);
    let insertion = projection_start
        .checked_add(projection_bytes)
        .expect("B projection insertion fits");
    s7.splice(insertion..insertion, projection.iter().copied());
    write_u64(s7, 112 + 10 * 16, 2 * S7_PROJECTION_BYTES as u64);
    for directory in 11..14 {
        let offset = read_u64(s7, 104 + directory * 16);
        write_u64(
            s7,
            104 + directory * 16,
            offset
                .checked_add(S7_PROJECTION_BYTES as u64)
                .expect("B shifted S7 directory offset fits"),
        );
    }
    write_u32(s7, 80, 2);
    let total_bytes = u64::try_from(s7.len()).expect("B S7 total length fits u64");
    write_u64(s7, 32, total_bytes);
    s7[504..536].copy_from_slice(&final_overlay);
    let root_descriptor = s7_root_descriptor(s7);
    s7[536..568].copy_from_slice(&root_descriptor);
}

fn statement_disposition_root(statement: u32, s4: &[u8]) -> [u8; 32] {
    digest(
        b"gpu-db/write001/s7-statement-dispositions/v2",
        &[&statement.to_le_bytes(), &1_u32.to_le_bytes(), s4],
    )
}

fn statement_sequence_root(statement: u32, entry: &[u8]) -> [u8; 32] {
    digest(
        b"gpu-db/write001/s7-statement-sequences/v2",
        &[&statement.to_le_bytes(), &1_u32.to_le_bytes(), entry],
    )
}

fn statement_dependency_root(
    statement: u32,
    uses: &[u8],
    dependency_digests: &[[u8; 32]; 4],
) -> [u8; 32] {
    assert_eq!(
        uses.len() % 32,
        0,
        "B dependency uses retain their fixed width"
    );
    let mut hasher = begin(b"gpu-db/write001/s7-statement-dependencies/v2");
    hasher.update(statement.to_le_bytes());
    hasher.update(
        u32::try_from(uses.len() / 32)
            .expect("B use count fits u32")
            .to_le_bytes(),
    );
    for usage in uses.chunks_exact(32) {
        let dependency = usize::try_from(read_u32(usage, 4)).expect("B dependency ref fits usize");
        hasher.update(usage);
        hasher.update(
            dependency_digests
                .get(dependency)
                .expect("B dependency use refers to its token"),
        );
    }
    hasher.finalize().into()
}

fn projection_root_from_digests(statement: u32, projections: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = begin(b"gpu-db/write001/s7-statement-projections/v2");
    hasher.update(statement.to_le_bytes());
    hasher.update(
        u32::try_from(projections.len())
            .expect("B projection count fits u32")
            .to_le_bytes(),
    );
    for projection in projections {
        hasher.update(projection);
    }
    hasher.finalize().into()
}

fn s2_record_digest(record: &[u8]) -> [u8; 32] {
    digest(
        b"gpu-db/write001/s7-s2-record/v2",
        &[
            &u32::try_from(record.len())
                .expect("B S2 record length fits u32")
                .to_le_bytes(),
            record,
        ],
    )
}

fn overlay_root(
    before: [u8; 32],
    statement: u32,
    typed_statement_digest: [u8; 32],
    record_digest: [u8; 32],
    statement_roots: [[u8; 32]; 4],
) -> [u8; 32] {
    digest(
        b"gpu-db/write001/s7-statement-overlay-root/v2",
        &[
            &before,
            &statement.to_le_bytes(),
            &typed_statement_digest,
            &record_digest,
            &statement_roots[0],
            &statement_roots[1],
            &statement_roots[2],
            &statement_roots[3],
        ],
    )
}

fn s7_root_descriptor(s7: &[u8]) -> [u8; 32] {
    let table_start = s7_region(s7, 0).0;
    let table = &s7[table_start..table_start + 384];
    let mut hasher = begin(b"gpu-db/write001/s7-root-descriptor/v2");
    hasher.update(1_u16.to_le_bytes());
    hasher.update(read_u64(s7, 328).to_le_bytes());
    hasher.update(read_u64(s7, 336).to_le_bytes());
    hasher.update(&s7[344..376]);
    hasher.update(&s7[376..408]);
    hasher.update(&s7[408..440]);
    hasher.update(&s7[440..472]);
    hasher.update(&s7[472..504]);
    hasher.update(&s7[504..536]);
    hasher.update(read_u32(s7, 40).to_le_bytes());
    hasher.update(0_u32.to_le_bytes());
    hasher.update(&table[8..16]);
    hasher.update(&table[32..48]);
    hasher.update(&table[160..224]);
    hasher.update(&table[352..384]);
    hasher.finalize().into()
}

fn aggregate_request_digest(s1: &[u8]) -> [u8; 32] {
    let mut hasher = begin(b"gpu-db/write001/aggregate-request/v2");
    hasher.update([2]);
    hasher.update(2_u32.to_le_bytes());
    for statement in 0..2_usize {
        hasher.update(
            u32::try_from(statement)
                .expect("B statement ordinal fits u32")
                .to_le_bytes(),
        );
        let entry = &s1[statement * 144..(statement + 1) * 144];
        hasher.update(&entry[16..48]);
        hasher.update(1_u32.to_le_bytes());
        hasher.update(0_u16.to_le_bytes());
    }
    hasher.finalize().into()
}

fn identifier_digest(identifier: &str) -> [u8; 32] {
    digest(
        b"gpu-db/write001/s7-identifier/v2",
        &[
            &u32::try_from(identifier.len())
                .expect("B identifier length fits u32")
                .to_le_bytes(),
            identifier.as_bytes(),
        ],
    )
}

fn storage(ty: crate::SqlType) -> [u8; 4] {
    match ty {
        crate::SqlType::Int2 => [1, 0, 0, 0],
        crate::SqlType::Int4 => [2, 0, 0, 0],
        crate::SqlType::Int8 => [3, 0, 0, 0],
        crate::SqlType::Numeric { precision, scale } => [4, precision, scale, 0],
        crate::SqlType::Bool => [5, 0, 0, 0],
        crate::SqlType::Text => [6, 0, 0, 0],
        crate::SqlType::Date => [7, 0, 0, 0],
        crate::SqlType::Timestamp => [8, 0, 0, 0],
        crate::SqlType::Uuid => [9, 0, 0, 0],
    }
}

fn reframe_retained_fixture(
    base: q1_vectors::Q1Fixture,
    sections: [Vec<u8>; AGGREGATE_SECTION_COUNT],
    entry_counts: [u32; AGGREGATE_SECTION_COUNT],
    retention_deadline: u64,
) -> q1_vectors::Q1Fixture {
    assert!(entry_counts[7] != 0 && retention_deadline != 0);
    let mut header: [u8; 96] = base.stream[..96]
        .try_into()
        .expect("Q1 aggregate header width");
    let aggregate_flags = read_u32(&header, 24) | AGGREGATE_FLAG_RETAINED_RESPONSE;
    write_u32(&mut header, 24, aggregate_flags);
    let section_headers: [[u8; 16]; AGGREGATE_SECTION_COUNT] = std::array::from_fn(|index| {
        let mut section_header = [0; 16];
        write_u16(&mut section_header, 0, index as u16 + 1);
        write_u32(&mut section_header, 4, entry_counts[index]);
        write_u64(&mut section_header, 8, sections[index].len() as u64);
        section_header
    });
    let section_bytes =
        section_headers
            .iter()
            .zip(&sections)
            .fold(0_u64, |total, (section_header, section)| {
                total
                    .checked_add(section_header.len() as u64)
                    .and_then(|total| total.checked_add(section.len() as u64))
                    .expect("bounded S8 fixture section bytes fit u64")
            });
    write_u64(&mut header, 32, section_bytes);
    let section_roots = std::array::from_fn(|index| {
        v1_digest(
            b"gpu-db/write001/aggregate-section/v1",
            &[&section_headers[index], &sections[index]],
        )
    });
    let aggregate_root = v1_digest(
        b"gpu-db/write001/aggregate-root/v1",
        &[
            &header,
            &section_roots[0],
            &section_roots[1],
            &section_roots[2],
            &section_roots[3],
            &section_roots[4],
            &section_roots[5],
            &section_roots[6],
            &section_roots[7],
        ],
    );
    let response_root = v1_digest(
        b"gpu-db/write001/response-root/v1",
        &[&section_roots[5], &section_roots[7]],
    );
    let mut stream = header.to_vec();
    for (section_header, section) in section_headers.iter().zip(&sections) {
        stream.extend_from_slice(section_header);
        stream.extend_from_slice(section);
    }
    stream.extend_from_slice(&aggregate_root);
    let mut fragment_body = base.fragment_body[..76].to_vec();
    write_u64(&mut fragment_body, 12, stream.len() as u64);
    write_u32(&mut fragment_body, 36, stream.len() as u32);
    fragment_body[44..76].copy_from_slice(&aggregate_root);
    fragment_body.extend_from_slice(&stream);
    let status = TypedInsertStatusV2 {
        database_id: base.outer.identity.database_id,
        timeline_id: base.outer.identity.timeline_id,
        txn_id: base.outer.stable_transaction_id,
        request_digest: base.outer.request_digest,
        isolation: base.outer.isolation as u8,
        flags: 0,
        retention_deadline,
        statement_count: read_u32(&header, 48),
        response_artifact_count: entry_counts[7],
        statement_outcome_root: section_roots[5],
        response_root,
        aggregate_root,
    };
    let mut status_bytes = vec![0; AGGREGATE_STATUS_V2_BYTES as usize];
    encode_status_v2(&status, &mut status_bytes).expect("S8 fixture STATUS2 encodes");
    let mut outcome = base.outcome;
    outcome.target_digest = aggregate_root;
    outcome.returning_digest = response_root;
    q1_vectors::Q1Fixture {
        sections,
        stream,
        fragment_body,
        status: status_bytes,
        outer: base.outer,
        outcome,
        aggregate_root,
        section_roots,
        root_descriptor: base.root_descriptor,
        payload_digest: base.payload_digest,
        table_manifest: base.table_manifest,
        overlay_after: base.overlay_after,
    }
}

/// Test-only hostile-vector framing helper.  It refreshes the S8 payload digest and every
/// aggregate/STATUS root after a byte mutation, without granting a production encoding edge.
pub(super) fn reframe_s8_after_mutation_for_sabotage(
    base: q1_vectors::Q1Fixture,
    mutate: impl FnOnce(&mut [u8]),
) -> q1_vectors::Q1Fixture {
    let mut sections = base.sections.clone();
    mutate(&mut sections[7]);
    refresh_s8_payload_digest(&mut sections[7]);
    reframe_retained_fixture(base, sections, [3, 3, 0, 3, 1, 3, 1, 2], 900)
}

fn encode_s8(
    sections: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
    s6_root: [u8; 32],
    s7_root: [u8; 32],
    stable_transaction_id: u64,
    request_digest: [u8; 32],
    retained_statements: &[u32],
    rows: ResponseRows,
) -> Vec<u8> {
    assert!(
        !retained_statements.is_empty(),
        "present S8 fixture has at least one artifact"
    );
    let mut materials: Vec<_> = retained_statements
        .iter()
        .copied()
        .map(|statement| artifact_material(sections, statement, rows))
        .collect();
    let mut image_bytes = 0_u64;
    for material in &mut materials {
        material.image_offset = image_bytes;
        image_bytes = image_bytes
            .checked_add(u64::try_from(material.image.len()).expect("fixture image fits u64"))
            .expect("fixture image arena fits u64");
    }
    let artifact_bytes = u64::try_from(materials.len())
        .expect("fixture artifact count fits u64")
        .checked_mul(S8_ARTIFACT_BYTES as u64)
        .expect("fixture artifact bytes fit u64");
    let selection_count = materials.iter().fold(0_u32, |total, material| {
        total
            .checked_add(material.selection_count)
            .expect("fixture selection count fits u32")
    });
    let selection_bytes = u64::from(selection_count)
        .checked_mul(S8_SELECTION_BYTES as u64)
        .expect("fixture selection bytes fit u64");
    let artifact_offset = S8_HEADER_BYTES as u64;
    let selection_offset = artifact_offset
        .checked_add(artifact_bytes)
        .expect("fixture selection offset fits u64");
    let image_offset = selection_offset
        .checked_add(selection_bytes)
        .expect("fixture image offset fits u64");
    let total_bytes = image_offset
        .checked_add(image_bytes)
        .expect("fixture total S8 bytes fit u64");

    let mut selections = Vec::with_capacity(materials.len() * S8_SELECTION_BYTES);
    let mut artifacts = Vec::with_capacity(materials.len() * S8_ARTIFACT_BYTES);
    let mut selection_start = 0_u32;
    for (ordinal, material) in materials.iter().enumerate() {
        let ordinal = u32::try_from(ordinal).expect("fixture artifact ordinal fits u32");
        let selection = (material.selection_count != 0)
            .then(|| selection_bytes_for(sections, ordinal, material.statement));
        if let Some(selection) = selection.as_ref() {
            selections.extend_from_slice(selection);
        }
        let descriptor = artifact_descriptor(
            sections,
            ordinal,
            material,
            selection.as_ref(),
            selection_start,
        );
        artifacts.extend_from_slice(&descriptor);
        selection_start = selection_start
            .checked_add(material.selection_count)
            .expect("fixture selection cursor fits u32");
    }

    let mut header = [0_u8; S8_HEADER_BYTES];
    header[..16].copy_from_slice(b"GPUDBS8RESPONSE2");
    write_u16(&mut header, 16, 1);
    write_u16(&mut header, 18, 2);
    write_u32(&mut header, 20, S8_HEADER_BYTES as u32);
    write_u16(&mut header, 28, 3);
    write_u16(&mut header, 30, 2);
    write_u64(&mut header, 32, total_bytes);
    write_u32(&mut header, 40, materials.len() as u32);
    write_u32(&mut header, 44, selection_count);
    write_u64(&mut header, 48, image_bytes);
    write_u64(&mut header, 56, artifact_offset);
    write_u64(&mut header, 64, artifact_bytes);
    write_u64(&mut header, 72, selection_offset);
    write_u64(&mut header, 80, selection_bytes);
    write_u64(&mut header, 88, image_offset);
    write_u64(&mut header, 96, image_bytes);
    write_u64(&mut header, 104, stable_transaction_id);
    header[112..144].copy_from_slice(&request_digest);
    header[144..176].copy_from_slice(&s6_root);
    header[176..208].copy_from_slice(&s7_root);

    let mut payload = header.to_vec();
    payload.extend_from_slice(&artifacts);
    payload.extend_from_slice(&selections);
    for material in &materials {
        payload.extend_from_slice(&material.image);
    }
    let payload_digest = digest(
        b"gpu-db/write001/s8-payload/v2",
        &[
            &total_bytes.to_le_bytes(),
            &payload[..208],
            &[0; 32],
            &payload[240..],
        ],
    );
    payload[208..240].copy_from_slice(&payload_digest);
    payload
}

struct ArtifactMaterial {
    statement: u32,
    image: Vec<u8>,
    image_offset: u64,
    selection_count: u32,
}

#[derive(Clone, Copy)]
enum ResponseRows {
    SelectedOne,
    Zero,
}

fn artifact_material(
    sections: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
    statement: u32,
    rows: ResponseRows,
) -> ArtifactMaterial {
    let record = record_for_statement(&sections[1], statement);
    let (resolution_start, _) = s7_region(&sections[6], 2);
    let resolution = &sections[6][resolution_start + statement as usize * S7_RESOLUTION_BYTES
        ..resolution_start + (statement as usize + 1) * S7_RESOLUTION_BYTES];
    let projection_start = read_u32(resolution, 48);
    let projection_count = read_u32(resolution, 52);
    let (projection_directory, _) = s7_region(&sections[6], 10);
    let mut values = Vec::with_capacity(projection_count as usize);
    let mut validity = Vec::with_capacity(projection_count as usize);
    let mut columns = Vec::with_capacity(projection_count as usize);
    for projection in projection_start..projection_start + projection_count {
        let raw = &sections[6][projection_directory + projection as usize * S7_PROJECTION_BYTES
            ..projection_directory + (projection as usize + 1) * S7_PROJECTION_BYTES];
        let source_ordinal = read_u32(raw, 12);
        let source = record
            .catalog_columns()
            .find(|column| column.catalog_column_ordinal == source_ordinal)
            .expect("Q1 projection source is present in its S2 record");
        values.push(match rows {
            ResponseRows::SelectedOne => {
                let (_, value) = record
                    .column_value_at(source_ordinal, 0)
                    .expect("Q1 projection row is present in S2");
                value_owner(value)
            }
            ResponseRows::Zero => empty_value_owner(source.ty),
        });
        validity.push(TypedInsertColumnValidity::AllValid);
        columns.push((source, read_u32(raw, 20), read_u16(raw, 38)));
    }
    let views: Vec<_> = columns
        .iter()
        .zip(validity.iter().zip(values.iter()))
        .map(
            |((column, table_ref, result_format), (validity, values))| TypedImageColumnView {
                catalog_column_ordinal: column.catalog_column_ordinal,
                stable_column_id: column.column_id,
                table_ref: *table_ref,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                result_format: *result_format,
                name: column.name,
                validity,
                values,
            },
        )
        .collect();
    let image = encode_typed_image(&TypedImageView {
        role: TypedImageRole::RetainedResponse,
        rows: match rows {
            ResponseRows::SelectedOne => 1,
            ResponseRows::Zero => 0,
        },
        columns: &views,
    })
    .expect("Q1 retained response image encodes");
    ArtifactMaterial {
        statement,
        image,
        image_offset: 0,
        selection_count: match rows {
            ResponseRows::SelectedOne => 1,
            ResponseRows::Zero => 0,
        },
    }
}

fn value_owner(
    value: crate::typed_insert_batch::DecodedTypedValueFacts<'_>,
) -> TypedInsertColumnValues {
    match value {
        crate::typed_insert_batch::DecodedTypedValueFacts::I32(value) => {
            TypedInsertColumnValues::I32(vec![value].into_boxed_slice())
        }
        crate::typed_insert_batch::DecodedTypedValueFacts::Text(value) => {
            TypedInsertColumnValues::Text {
                offsets: vec![0, value.len() as u64].into_boxed_slice(),
                bytes: value.as_bytes().to_vec().into_boxed_slice(),
            }
        }
        other => panic!("Q1 retained response fixture uses only i32/text projections: {other:?}"),
    }
}

fn empty_value_owner(ty: crate::SqlType) -> TypedInsertColumnValues {
    match ty {
        crate::SqlType::Int4 => TypedInsertColumnValues::I32(Vec::new().into_boxed_slice()),
        crate::SqlType::Text => TypedInsertColumnValues::Text {
            offsets: vec![0].into_boxed_slice(),
            bytes: Vec::new().into_boxed_slice(),
        },
        other => panic!("Q1 zero-row response fixture uses only i32/text projections: {other:?}"),
    }
}

fn artifact_descriptor(
    sections: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
    artifact_ref: u32,
    material: &ArtifactMaterial,
    selection: Option<&[u8; S8_SELECTION_BYTES]>,
    selection_start: u32,
) -> [u8; S8_ARTIFACT_BYTES] {
    let statement = material.statement as usize;
    let s6 = &sections[5][statement * S6_BYTES..(statement + 1) * S6_BYTES];
    let outcome = gpu_db_wal::decode_canonical_outcome_exact(&s6[44..])
        .expect("Q1 retained statement has canonical S6 outcome");
    let (resolution_start, _) = s7_region(&sections[6], 2);
    let resolution = &sections[6][resolution_start + statement * S7_RESOLUTION_BYTES
        ..resolution_start + (statement + 1) * S7_RESOLUTION_BYTES];
    let projection_start = read_u32(resolution, 48);
    let projection_count = read_u32(resolution, 52);
    let projection_root = projection_root(
        &sections[6],
        material.statement,
        projection_start,
        projection_count,
    );
    let mut selection_hasher = begin(b"gpu-db/write001/s8-row-selection-root/v2");
    selection_hasher.update(artifact_ref.to_le_bytes());
    selection_hasher.update(material.statement.to_le_bytes());
    selection_hasher.update(material.selection_count.to_le_bytes());
    if let Some(selection) = selection {
        selection_hasher.update(selection);
    }
    let selection_root: [u8; 32] = selection_hasher.finalize().into();
    let image_layout: [u8; 32] = material.image[64..96]
        .try_into()
        .expect("typed image layout width");
    let image_content = digest(
        b"gpu-db/write001/s8-image-content/v2",
        &[
            &(material.image.len() as u64).to_le_bytes(),
            &material.image,
        ],
    );
    let mut artifact = [0_u8; S8_ARTIFACT_BYTES];
    write_u32(&mut artifact, 0, artifact_ref);
    write_u32(&mut artifact, 4, material.statement);
    write_u32(&mut artifact, 8, material.statement);
    write_u32(&mut artifact, 12, material.statement);
    write_u32(&mut artifact, 16, selection_start);
    write_u32(&mut artifact, 20, material.selection_count);
    write_u32(&mut artifact, 24, projection_start);
    write_u32(&mut artifact, 28, projection_count);
    write_u32(&mut artifact, 32, material.selection_count);
    write_u32(&mut artifact, 36, projection_count);
    write_u64(&mut artifact, 40, material.image_offset);
    write_u64(&mut artifact, 48, material.image.len() as u64);
    write_u16(&mut artifact, 56, 1);
    write_u32(&mut artifact, 60, 2);
    artifact[64..96].copy_from_slice(&resolution[128..160]);
    artifact[96..128].copy_from_slice(&outcome.returning_digest);
    artifact[128..160].copy_from_slice(&projection_root);
    artifact[160..192].copy_from_slice(&selection_root);
    artifact[192..224].copy_from_slice(&image_layout);
    artifact[224..256].copy_from_slice(&image_content);
    let s6_entry = digest(b"gpu-db/write001/s7-s6-entry/v2", &[s6]);
    let (projection_directory, _) = s7_region(&sections[6], 10);
    let mut hasher = begin(b"gpu-db/write001/s8-artifact/v2");
    hasher.update(&artifact[..256]);
    hasher.update([0; 32]);
    hasher.update(s6_entry);
    hasher.update(resolution);
    for projection in projection_start..projection_start + projection_count {
        let offset = projection_directory + projection as usize * S7_PROJECTION_BYTES;
        hasher.update(&sections[6][offset + 96..offset + 128]);
    }
    if let Some(selection) = selection {
        hasher.update(selection);
    }
    artifact[256..288].copy_from_slice(&hasher.finalize());
    artifact
}

fn selection_bytes_for(
    sections: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
    artifact_ref: u32,
    statement: u32,
) -> [u8; S8_SELECTION_BYTES] {
    let s4 = &sections[3][statement as usize * 64..(statement as usize + 1) * 64];
    assert!(
        matches!(s4[16], 1 | 2),
        "selected Q1 S4 row is visible to RETURNING"
    );
    let mut selection = [0_u8; S8_SELECTION_BYTES];
    write_u32(&mut selection, 0, artifact_ref);
    write_u32(&mut selection, 4, 0);
    write_u32(&mut selection, 8, statement);
    write_u32(&mut selection, 12, statement);
    selection[16..20].copy_from_slice(&s4[4..8]);
    selection[20..24].copy_from_slice(&s4[20..24]);
    selection[24..32].copy_from_slice(&s4[8..16]);
    selection
}

fn projection_root(s7: &[u8], statement: u32, start: u32, count: u32) -> [u8; 32] {
    let (directory, _) = s7_region(s7, 10);
    let mut hasher = begin(b"gpu-db/write001/s7-statement-projections/v2");
    hasher.update(statement.to_le_bytes());
    hasher.update(count.to_le_bytes());
    for ordinal in start..start + count {
        let offset = directory + ordinal as usize * S7_PROJECTION_BYTES;
        hasher.update(&s7[offset + 96..offset + 128]);
    }
    hasher.finalize().into()
}

fn record_for_statement(
    s2: &[u8],
    statement: u32,
) -> crate::typed_insert_batch::DecodedTypedInsertRecord {
    let mut cursor = 0_usize;
    for ordinal in 0..=statement {
        let bytes = read_u32(s2, cursor) as usize;
        cursor += 4;
        let record = &s2[cursor..cursor + bytes];
        if ordinal == statement {
            return decode_canonical_typed_insert_record(record)
                .expect("Q1 retained response S2 record decodes");
        }
        cursor += bytes;
    }
    unreachable!("bounded statement loop returns at its requested ordinal")
}

fn refresh_s7_payload_digest(s7: &mut [u8]) {
    let total = s7.len() as u64;
    let payload = digest(
        b"gpu-db/write001/s7-payload/v2",
        &[&total.to_le_bytes(), &s7[..568], &[0; 32], &s7[600..]],
    );
    s7[568..600].copy_from_slice(&payload);
}

fn refresh_s8_payload_digest(s8: &mut [u8]) {
    let total = read_u64(s8, 32);
    assert_eq!(total, s8.len() as u64, "S8 test payload length is exact");
    let payload = digest(
        b"gpu-db/write001/s8-payload/v2",
        &[&total.to_le_bytes(), &s8[..208], &[0; 32], &s8[240..]],
    );
    s8[208..240].copy_from_slice(&payload);
}

fn section_root(index: usize, entries: u32, section: &[u8]) -> [u8; 32] {
    let mut header = [0_u8; AGGREGATE_SECTION_HEADER_BYTES as usize];
    write_u16(&mut header, 0, index as u16 + 1);
    write_u32(&mut header, 4, entries);
    write_u64(&mut header, 8, section.len() as u64);
    v1_digest(b"gpu-db/write001/aggregate-section/v1", &[&header, section])
}

fn s7_region(s7: &[u8], directory: usize) -> (usize, usize) {
    let offset = read_u64(s7, 104 + directory * 16) as usize;
    let bytes = read_u64(s7, 112 + directory * 16) as usize;
    (offset, bytes)
}

fn fragments(fixture: &q1_vectors::Q1Fixture) -> [gpu_db_wal::CanonicalFragmentRef<'_>; 2] {
    [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &fixture.fragment_body,
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        },
    ]
}

fn begin(domain: &[u8]) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher
}

fn digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = begin(domain);
    for field in fields {
        hasher.update(field);
    }
    hasher.finalize().into()
}

fn v1_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = begin(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("fixture u16"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixture u32"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixture u64"))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
