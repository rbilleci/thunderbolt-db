//! Decoder adversarial-input coverage lives here as the matrix grows.

use super::tests::{prepared_batch, section_offsets};
use super::*;

fn baseline() -> Vec<u8> {
    encode(&prepared_batch()).expect("sabotage fixture encodes")
}

fn reject(bytes: &[u8]) {
    assert!(decode(bytes).is_err(), "forged record unexpectedly decodes");
}

fn reject_with(bytes: &[u8], expected: &str) {
    let Err(error) = decode(bytes) else {
        panic!("forged record unexpectedly decodes");
    };
    assert!(
        error.to_string().contains(expected),
        "forged record reached the wrong rejection: {error}"
    );
}

fn single_column_batch(
    txn_id: TxnId,
    table: &str,
    ty: &str,
    rows: Vec<crate::SqlValue>,
) -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(txn_id, &format!("CREATE TABLE {table} (value {ty})"))
        .expect("single-column fixture creates");
    let insert = crate::Insert {
        table: table.to_string(),
        columns: Vec::new(),
        rows: crate::Insert::programmatic_rows(rows.into_iter().map(|value| vec![value]).collect()),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    prepare_typed_insert_semantics(&insert, &catalog, catalog.commit_seq, None)
        .expect("single-column fixture prepares")
        .expect("single-column fixture is current")
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("single-column fixture seals")
}

fn text_batch_at_size(text_len: usize) -> TypedInsertBatch {
    single_column_batch(
        63,
        "codec_exact_size",
        "text",
        vec![crate::SqlValue::Text("x".repeat(text_len))],
    )
}

fn many_empty_text_batch(rows: usize) -> TypedInsertBatch {
    single_column_batch(
        65,
        "codec_many_empty_text",
        "text",
        std::iter::repeat_with(|| crate::SqlValue::Text(String::new()))
            .take(rows)
            .collect(),
    )
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 width"))
}

fn first_column_value_and_state_offsets(bytes: &[u8]) -> (usize, usize) {
    let mut cursor = section_offsets(bytes)[1] + 8;
    assert_eq!(u32_at(bytes, cursor), 1, "fixture has one target column");
    cursor += 4 + 4;
    cursor += 4 + usize::try_from(u32_at(bytes, cursor)).expect("column name length");
    cursor += 4 + 2 + 4 + 4 + 2;
    for _ in 0..2 {
        let tag = bytes[cursor];
        cursor += 1 + if tag == 1 { 4 } else { 0 };
    }
    match bytes[cursor] {
        0 => cursor += 1,
        1 => {
            let word_count = usize::try_from(u32_at(bytes, cursor + 1)).expect("bitmap count");
            cursor += 1 + 4 + word_count * 4;
        }
        other => panic!("unexpected fixture validity form {other}"),
    }
    assert_eq!(bytes[cursor], 0, "fixture presence is all-provided");
    cursor += 1;
    assert_eq!(bytes[cursor], 0, "fixture defaults are direct");
    cursor += 1;
    let rows = usize::try_from(u32_at(bytes, cursor)).expect("row count");
    let state = cursor + 4;
    (state + rows * 6, state)
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> WireCursor<'a> {
    fn new(bytes: &'a [u8], cursor: usize) -> Self {
        Self { bytes, cursor }
    }

    fn byte(&mut self) -> (usize, u8) {
        let offset = self.cursor;
        self.cursor += 1;
        (offset, self.bytes[offset])
    }

    fn u32(&mut self) -> (usize, u32) {
        let offset = self.cursor;
        self.cursor += 4;
        (offset, u32_at(self.bytes, offset))
    }

    fn skip(&mut self, width: usize) {
        self.cursor += width;
    }

    fn identifier(&mut self) {
        let (_, length) = self.u32();
        self.skip(usize::try_from(length).expect("fixture identifier length fits"));
    }

    fn sql_type(&mut self) -> (usize, usize) {
        let tag = self.byte().0;
        self.skip(2);
        let reserved = self.byte().0;
        (tag, reserved)
    }

    fn option_u32(&mut self) -> (usize, Option<usize>) {
        let (tag, value) = self.byte();
        match value {
            0 => (tag, None),
            1 => (tag, Some(self.u32().0)),
            other => panic!("fixture option tag {other} is invalid"),
        }
    }

    fn bitmap_form(&mut self) -> usize {
        let (tag, value) = self.byte();
        if value == 1 {
            let (_, words) = self.u32();
            self.skip(usize::try_from(words).expect("bitmap word count fits") * 4);
        } else {
            assert_eq!(value, 0, "fixture bitmap tag is valid");
        }
        tag
    }
}

#[derive(Clone, Copy)]
struct ColumnOffsets {
    ordinal: usize,
    column_id: usize,
    sql_type: usize,
    sql_reserved: usize,
    source_option: usize,
    domain_option: usize,
    validity_form: usize,
    default_form: usize,
    provenance: usize,
    provenance_index: usize,
    vector_tag: usize,
    vector_payload: usize,
}

fn column_offsets(bytes: &[u8]) -> Vec<ColumnOffsets> {
    let mut wire = WireCursor::new(bytes, section_offsets(bytes)[1] + 8);
    let (_, count) = wire.u32();
    let mut columns = Vec::new();
    for _ in 0..count {
        let ordinal = wire.u32().0;
        wire.identifier();
        let column_id = wire.u32().0;
        wire.skip(2);
        let (sql_type, sql_reserved) = wire.sql_type();
        wire.skip(4 + 2);
        let (source_option, _) = wire.option_u32();
        let (domain_option, _) = wire.option_u32();
        let validity_form = wire.bitmap_form();
        wire.skip(1);
        let default_form = wire.bitmap_form();
        let (_, rows) = wire.u32();
        let mut provenance = None;
        for _ in 0..rows {
            wire.skip(1);
            let tag = wire.byte().0;
            let index = wire.u32().0;
            provenance.get_or_insert((tag, index));
        }
        let vector_tag = wire.byte().0;
        wire.skip(4);
        let (vector_payload, payload) = wire.u32();
        wire.skip(usize::try_from(payload).expect("vector payload fits fixture"));
        let (provenance, provenance_index) = provenance.expect("column has at least one row");
        columns.push(ColumnOffsets {
            ordinal,
            column_id,
            sql_type,
            sql_reserved,
            source_option,
            domain_option,
            validity_form,
            default_form,
            provenance,
            provenance_index,
            vector_tag,
            vector_payload,
        });
    }
    columns
}

#[derive(Clone, Copy)]
struct DependencyOffsets {
    ordinal: usize,
    role: usize,
}

fn dependency_offsets(bytes: &[u8]) -> Vec<DependencyOffsets> {
    let mut wire = WireCursor::new(bytes, section_offsets(bytes)[2] + 8);
    let (_, count) = wire.u32();
    let mut dependencies = Vec::new();
    for _ in 0..count {
        let ordinal = wire.u32().0;
        let role = wire.byte().0;
        wire.identifier();
        wire.identifier();
        wire.skip(4 + 32);
        dependencies.push(DependencyOffsets { ordinal, role });
    }
    dependencies
}

#[derive(Clone, Copy)]
struct IndexOffsets {
    raw_ordinal: usize,
    oid: usize,
    flags: [usize; 3],
}

fn index_offsets(bytes: &[u8]) -> Vec<IndexOffsets> {
    let mut wire = WireCursor::new(bytes, section_offsets(bytes)[4] + 8);
    let (_, count) = wire.u32();
    let mut indexes = Vec::new();
    for _ in 0..count {
        wire.skip(4);
        let raw_ordinal = wire.u32().0;
        let oid = wire.u32().0;
        wire.identifier();
        wire.identifier();
        wire.identifier();
        let flags = [wire.byte().0, wire.byte().0, wire.byte().0];
        let (_, keys) = wire.u32();
        for _ in 0..keys {
            wire.skip(4 + 4 + 4 + 2);
            wire.identifier();
            wire.skip(4 + 4 + 2);
        }
        indexes.push(IndexOffsets {
            raw_ordinal,
            oid,
            flags,
        });
    }
    indexes
}

#[derive(Clone, Copy)]
struct PrivateSequenceOffsets {
    parent: usize,
    effect: usize,
    lifetime: usize,
    owner: usize,
    predecessor: usize,
}

fn first_private_sequence_offsets(bytes: &[u8]) -> PrivateSequenceOffsets {
    let mut wire = WireCursor::new(bytes, section_offsets(bytes)[7] + 8);
    let parent = wire.byte().0;
    assert_eq!(
        bytes[parent], 1,
        "private fixture carries a sequence parent"
    );
    wire.skip(8 + 1 + 32 + 4 + 4);
    assert!(wire.u32().1 > 0, "private fixture has effects");
    wire.skip(4);
    wire.skip(5 * 4);
    wire.identifier();
    wire.identifier();
    wire.skip(4 + 4 + 4 + 32 + 8);
    let effect = wire.byte().0;
    assert_eq!(bytes[effect], 2, "private fixture uses private evidence");
    wire.skip(8 + 1 + 8 + 1);
    let lifetime = wire.byte().0;
    let owner = wire.byte().0;
    wire.skip(4 + 32);
    let (_, creator) = wire.byte();
    if creator == 1 {
        wire.skip(4);
    }
    let predecessor = wire.byte().0;
    PrivateSequenceOffsets {
        parent,
        effect,
        lifetime,
        owner,
        predecessor,
    }
}

fn bound_parameter_batch() -> TypedInsertBatch {
    let engine = crate::Engine::new_local();
    engine
        .execute_text(64, "CREATE TABLE codec_bound_parameter (value int4)")
        .expect("bound fixture creates");
    let bound =
        crate::PreparedCommand::parse("INSERT INTO codec_bound_parameter (value) VALUES ($1)")
            .expect("bound fixture parses")
            .bind(&[crate::SqlValue::Int4(7)])
            .expect("bound fixture binds");
    let crate::Command::Insert(insert) = bound.command() else {
        panic!("bound fixture is INSERT");
    };
    let catalog = engine.catalog_snapshot();
    prepare_typed_insert_semantics(insert, &catalog, catalog.commit_seq, None)
        .expect("bound fixture prepares")
        .expect("bound fixture is current")
        .seal(
            sequence_defaults::SequenceDefaultBindings::empty(),
            false,
            false,
        )
        .expect("bound fixture seals")
}

fn with_u32(mut bytes: Vec<u8>, offset: usize, value: u32) -> Vec<u8> {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    bytes
}

#[test]
fn canonical_codec_writer_enforces_exact_record_limit() {
    let mut writer = Writer::new(MAX_RECORD_BYTES);
    writer
        .bytes(&vec![0; MAX_RECORD_BYTES])
        .expect("exact 16 MiB record is admitted");
    assert!(writer.u8(0).is_err(), "one byte beyond 16 MiB is rejected");
}

#[test]
fn canonical_codec_accepts_an_actual_exact_16mib_batch_and_rejects_one_byte_more() {
    let base = encode(&text_batch_at_size(0))
        .expect("empty text fixture encodes")
        .len();
    let text_len = MAX_RECORD_BYTES
        .checked_sub(base)
        .expect("fixed record overhead fits below 16 MiB");
    let exact = encode(&text_batch_at_size(text_len)).expect("exact 16 MiB record encodes");
    assert_eq!(exact.len(), MAX_RECORD_BYTES);
    assert_eq!(
        encoded_len(&text_batch_at_size(text_len)).expect("exact record counts"),
        MAX_RECORD_BYTES
    );
    assert!(
        encode(&text_batch_at_size(text_len + 1)).is_err(),
        "one payload byte above the fragment ceiling is rejected"
    );
    assert!(
        encoded_len(&text_batch_at_size(text_len + 1)).is_err(),
        "counting keeps the same inclusive one-fragment ceiling"
    );
}

#[test]
fn canonical_codec_decodes_encoder_output_just_above_the_legacy_generic_count_cap() {
    let batch = many_empty_text_batch((1 << 20) + 1);
    let bytes = encode(&batch).expect("just-over-cap empty text record encodes");
    assert!(
        bytes.len() < MAX_RECORD_BYTES,
        "fixture remains below 16 MiB"
    );
    assert_eq!(
        decode(&bytes)
            .expect("canonical encoder output must decode")
            .reencode(),
        bytes
    );
}

#[test]
fn canonical_codec_rejects_every_header_and_section_header_byte() {
    let bytes = baseline();
    for byte in 0..HEADER_LEN {
        let mut forged = bytes.clone();
        forged[byte] ^= 0x80;
        reject(&forged);
    }
    for section in section_offsets(&bytes) {
        for byte in section..section + 8 {
            let mut forged = bytes.clone();
            forged[byte] ^= 0x40;
            reject(&forged);
        }
    }
}

#[test]
fn canonical_codec_rejects_all_truncations_trailing_bytes_and_absurd_counts_without_panicking() {
    let bytes = baseline();
    for length in 0..bytes.len() {
        let result = std::panic::catch_unwind(|| decode(&bytes[..length]));
        assert!(result.is_ok(), "truncation {length} panicked");
        assert!(result.expect("truncation is unwind-safe").is_err());
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    reject(&trailing);

    for section in section_offsets(&bytes) {
        let mut forged = bytes.clone();
        forged[section + 8..section + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        let result = std::panic::catch_unwind(|| decode(&forged));
        assert!(result.is_ok(), "huge count at section {section} panicked");
        assert!(result.expect("huge count is unwind-safe").is_err());
    }
}

#[test]
fn canonical_codec_rejects_identifier_utf8_empty_name_and_length_forgeries() {
    let bytes = baseline();
    let target = section_offsets(&bytes)[0] + 8;
    let schema_length = u32::from_le_bytes(bytes[target..target + 4].try_into().unwrap()) as usize;
    assert!(schema_length > 0);
    let mut invalid_utf8 = bytes.clone();
    invalid_utf8[target + 4] = 0xff;
    reject(&invalid_utf8);

    let mut empty_identifier = bytes.clone();
    empty_identifier[target..target + 4].copy_from_slice(&0_u32.to_le_bytes());
    reject(&empty_identifier);

    for section in section_offsets(&bytes) {
        let mut too_long = bytes.clone();
        too_long[section + 4..section + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        reject(&too_long);
    }
}

#[test]
fn canonical_codec_rejects_bitmap_tails_text_offsets_and_input_state_tags() {
    let bool_bytes = encode(&single_column_batch(
        61,
        "codec_bool_tail",
        "bool",
        vec![crate::SqlValue::Bool(true), crate::SqlValue::Bool(false)],
    ))
    .expect("bool record encodes");
    let (bool_values, _) = first_column_value_and_state_offsets(&bool_bytes);
    assert_eq!(bool_bytes[bool_values], 5, "bool vector shape");
    let bool_word = bool_values + 1 + 4 + 4 + 4;
    let mut bad_tail = bool_bytes.clone();
    let tail_word = u32_at(&bad_tail, bool_word) | (1 << 31);
    bad_tail[bool_word..bool_word + 4].copy_from_slice(&tail_word.to_le_bytes());
    reject(&bad_tail);

    let text_bytes = encode(&single_column_batch(
        62,
        "codec_text_offset",
        "text",
        vec![crate::SqlValue::Text(String::new()), crate::SqlValue::Null],
    ))
    .expect("text record encodes");
    let (text_values, text_state) = first_column_value_and_state_offsets(&text_bytes);
    assert_eq!(text_bytes[text_values], 6, "text vector shape");
    let text_offsets = text_values + 1 + 4 + 4 + 4;
    let mut bad_offset = text_bytes.clone();
    bad_offset[text_offsets + 8..text_offsets + 16].copy_from_slice(&1_u64.to_le_bytes());
    reject(&bad_offset);

    let mut bad_state = text_bytes.clone();
    bad_state[text_state] = 0;
    reject(&bad_state);

    let multibyte = encode(&single_column_batch(
        94,
        "codec_text_boundary",
        "text",
        vec![
            crate::SqlValue::Text("é".to_string()),
            crate::SqlValue::Text("x".to_string()),
        ],
    ))
    .expect("multibyte text fixture encodes");
    let (multibyte_values, _) = first_column_value_and_state_offsets(&multibyte);
    let multibyte_offsets = multibyte_values + 1 + 4 + 4 + 4;
    let mut split_codepoint = multibyte;
    split_codepoint[multibyte_offsets + 8..multibyte_offsets + 16]
        .copy_from_slice(&1_u64.to_le_bytes());
    reject(&split_codepoint);
}

#[test]
fn canonical_codec_rejects_order_and_mirrored_digest_forgeries() {
    let bytes = baseline();
    let columns = section_offsets(&bytes)[1] + 8;
    let mut bad_order = bytes.clone();
    bad_order[columns + 4..columns + 8].copy_from_slice(&1_u32.to_le_bytes());
    reject(&bad_order);

    let returning = section_offsets(&bytes)[6] + 8;
    let mut mirrored_returning_digest = bytes;
    mirrored_returning_digest[returning + 16..returning + 48].copy_from_slice(&[0xa7; 32]);
    mirrored_returning_digest[68..100].copy_from_slice(&[0xa7; 32]);
    reject(&mirrored_returning_digest);
}

#[test]
fn canonical_codec_rejects_true_rehashed_returning_index_and_foreign_key_forgeries() {
    let returning = decode::rehashed_forgery_for_test(
        &baseline(),
        decode::RehashedForgery::ReturningProjection,
    )
    .expect("RETURNING forgery rehashes");
    reject(&returning);

    let closure =
        encode(&super::tests::catalog_closure_batch()).expect("catalog-closure fixture encodes");
    for kind in [
        decode::RehashedForgery::TargetIndex,
        decode::RehashedForgery::ForeignKey,
        decode::RehashedForgery::ForeignKeySupportingIndexMetadata,
    ] {
        let forged =
            decode::rehashed_forgery_for_test(&closure, kind).expect("catalog forgery rehashes");
        reject(&forged);
    }

    let self_referencing = encode(&super::tests::self_referencing_catalog_closure_batch())
        .expect("self-referencing catalog-closure fixture encodes");
    let forged = decode::rehashed_forgery_for_test(
        &self_referencing,
        decode::RehashedForgery::SelfForeignKeySupportingIndexIdentity,
    )
    .expect("self-referencing supporting-index forgery rehashes");
    reject(&forged);

    let shared_external = encode(&super::tests::external_shared_supporting_index_batch())
        .expect("shared external supporting-index fixture encodes");
    let forged = decode::rehashed_forgery_for_test(
        &shared_external,
        decode::RehashedForgery::ExternalSupportingIndexConflict,
    )
    .expect("external supporting-index conflict rehashes");
    reject(&forged);
}

#[test]
fn canonical_codec_rejects_true_rehashed_sequence_geometry_and_mode_forgeries() {
    let serial = encode(&super::tests::serial_batch(41)).expect("serial fixture encodes");
    let transition_parent = decode::rehashed_forgery_for_test(
        &serial,
        decode::RehashedForgery::SequenceTransitionMatchesParent,
    )
    .expect("published-transition forgery rehashes");
    reject(&transition_parent);
    let int8_target =
        decode::rehashed_forgery_for_test(&serial, decode::RehashedForgery::SequenceInt8Target)
            .expect("Int8 target forgery rehashes");
    reject_with(
        &int8_target,
        "sequence effect does not match resolved typed vector output",
    );

    let serial_two = encode(&super::tests::serial_batch_values(&[41, 42]))
        .expect("two-row serial fixture encodes");
    let duplicate_local = decode::rehashed_forgery_for_test(
        &serial_two,
        decode::RehashedForgery::SequenceDuplicateLocal,
    )
    .expect("duplicate-local sequence forgery rehashes");
    reject_with(
        &duplicate_local,
        "sequence expression geometry is noncanonical",
    );
    let effective_identity = decode::rehashed_forgery_for_test(
        &serial_two,
        decode::RehashedForgery::SequenceEffectiveIdentity,
    )
    .expect("effective-name sequence forgery rehashes");
    reject_with(&effective_identity, "sequence column identity drifted");
    let reversed_transition = decode::rehashed_forgery_for_test(
        &serial_two,
        decode::RehashedForgery::SequenceTransitionReverse,
    )
    .expect("reversed-transition sequence forgery rehashes");
    reject(&reversed_transition);

    let private =
        encode(&super::sequence_tests::private_chain_batch()).expect("private fixture encodes");
    for kind in [
        decode::RehashedForgery::SequenceMixedMode,
        decode::RehashedForgery::SequencePrivateOwnerFuture,
    ] {
        let forged = decode::rehashed_forgery_for_test(&private, kind)
            .expect("private sequence law forgery rehashes");
        reject(&forged);
    }
    let autocommit_private = decode::rehashed_forgery_for_test(
        &private,
        decode::RehashedForgery::SequenceAutocommitPrivate,
    )
    .expect("isolated autocommit/private forgery rehashes");
    let Err(error) = decode(&autocommit_private) else {
        panic!("autocommit/private forgery unexpectedly decodes");
    };
    assert!(
        error
            .to_string()
            .contains("autocommit sequence section has private effect"),
        "autocommit/private forgery must reach the isolated mode check: {error}"
    );
}

#[test]
fn canonical_codec_rejects_true_rehashed_global_identity_and_value_domain_forgeries() {
    let closure =
        encode(&super::tests::catalog_closure_batch()).expect("catalog-closure fixture encodes");
    for kind in [
        decode::RehashedForgery::IndexClassNameCollision,
        decode::RehashedForgery::IndexOidOutOfRange,
        decode::RehashedForgery::ExternalColumnIdCollision,
        decode::RehashedForgery::ExternalColumnHugeOrdinal,
        decode::RehashedForgery::ForeignKeyTypeOidRelationCollision,
    ] {
        let forged = decode::rehashed_forgery_for_test(&closure, kind)
            .expect("catalog identity forgery rehashes");
        reject(&forged);
    }

    let serial = encode(&super::tests::serial_batch(41)).expect("serial fixture encodes");
    let forged = decode::rehashed_forgery_for_test(
        &serial,
        decode::RehashedForgery::SequenceClassNameCollision,
    )
    .expect("sequence class-name forgery rehashes");
    reject_with(&forged, "qualified class-name identity drifted");

    let date = encode(&single_column_batch(
        92,
        "codec_date_carrier",
        "date",
        vec![crate::SqlValue::Date(0)],
    ))
    .expect("date fixture encodes");
    let timestamp = encode(&single_column_batch(
        93,
        "codec_timestamp_carrier",
        "timestamp",
        vec![crate::SqlValue::Timestamp(0)],
    ))
    .expect("timestamp fixture encodes");
    for (bytes, kind) in [
        (date.clone(), decode::RehashedForgery::DateCarrierUnderflow),
        (date, decode::RehashedForgery::DateCarrierOverflow),
        (
            timestamp.clone(),
            decode::RehashedForgery::TimestampCarrierUnderflow,
        ),
        (timestamp, decode::RehashedForgery::TimestampCarrierOverflow),
    ] {
        let forged = decode::rehashed_forgery_for_test(&bytes, kind)
            .expect("temporal carrier forgery rehashes");
        reject(&forged);
    }
}

#[test]
fn canonical_codec_rejects_stable_tag_form_and_identity_corruption_matrix() {
    let direct = baseline();
    let direct_columns = column_offsets(&direct);
    let direct_validity = direct_columns
        .iter()
        .find(|column| direct[column.validity_form] == 1)
        .expect("nullable fixture carries a validity bitmap");
    let bound = encode(&bound_parameter_batch()).expect("bound fixture encodes");
    let bound_column = column_offsets(&bound)[0];
    assert_eq!(bound[bound_column.provenance], 3, "fixture is bound");
    let serial = encode(&super::tests::serial_batch(31)).expect("serial fixture encodes");
    let serial_default = column_offsets(&serial)
        .into_iter()
        .find(|column| serial[column.default_form] == 1)
        .expect("serial fixture carries a default bitmap");
    let closure =
        encode(&super::tests::catalog_closure_batch()).expect("catalog closure fixture encodes");
    let closure_columns = column_offsets(&closure);
    let closure_domain = closure_columns
        .iter()
        .find(|column| closure[column.domain_option] == 1)
        .expect("domain fixture has a domain option");
    let dependencies = dependency_offsets(&closure);
    let indexes = index_offsets(&closure);
    assert!(
        indexes.len() >= 2,
        "closure fixture has multiple target indexes"
    );
    let private = encode(&super::sequence_tests::private_chain_batch())
        .expect("private sequence fixture encodes");
    let private_offsets = first_private_sequence_offsets(&private);

    let tag_cases = vec![
        (
            "SQL type tag",
            direct.clone(),
            direct_columns[0].sql_type,
            0,
        ),
        (
            "SQL type reserved byte",
            direct.clone(),
            direct_columns[0].sql_reserved,
            1,
        ),
        (
            "source option tag",
            direct.clone(),
            direct_columns[0].source_option,
            2,
        ),
        (
            "domain option tag",
            closure.clone(),
            closure_domain.domain_option,
            2,
        ),
        (
            "validity bitmap form",
            direct.clone(),
            direct_validity.validity_form,
            2,
        ),
        (
            "default bitmap form",
            serial.clone(),
            serial_default.default_form,
            2,
        ),
        (
            "input provenance tag",
            direct.clone(),
            direct_columns[0].provenance,
            0,
        ),
        (
            "sequence parent tag",
            private.clone(),
            private_offsets.parent,
            2,
        ),
        (
            "sequence effect tag",
            private.clone(),
            private_offsets.effect,
            3,
        ),
        (
            "private lifetime tag",
            private.clone(),
            private_offsets.lifetime,
            0,
        ),
        (
            "private owner tag",
            private.clone(),
            private_offsets.owner,
            0,
        ),
        (
            "private predecessor tag",
            private.clone(),
            private_offsets.predecessor,
            3,
        ),
    ];
    for (name, mut forged, offset, value) in tag_cases {
        forged[offset] = value;
        assert!(decode(&forged).is_err(), "{name} forgery decoded");
    }

    let scalar_cases = vec![
        (
            "bound parameter index zero",
            with_u32(bound, bound_column.provenance_index, 0),
        ),
        ("vector arm tag", {
            let mut forged = direct.clone();
            forged[direct_columns[0].vector_tag] = 0;
            forged
        }),
        (
            "vector payload length",
            with_u32(direct.clone(), direct_columns[0].vector_payload, 0),
        ),
        ("dependency role", {
            let mut forged = closure.clone();
            forged[dependencies[0].role] = 0;
            forged
        }),
    ];
    for (name, forged) in scalar_cases {
        assert!(decode(&forged).is_err(), "{name} forgery decoded");
    }
    for (flag, offset) in indexes[0].flags.into_iter().enumerate() {
        let mut forged = closure.clone();
        forged[offset] = 2;
        assert!(
            decode(&forged).is_err(),
            "index boolean flag {flag} forgery decoded"
        );
    }

    let duplicate_and_order_cases = vec![
        (
            "column count",
            with_u32(
                direct.clone(),
                section_offsets(&direct)[1] + 8,
                u32_at(&direct, section_offsets(&direct)[1] + 8) + 1,
            ),
        ),
        (
            "dependency count",
            with_u32(
                closure.clone(),
                section_offsets(&closure)[2] + 8,
                u32_at(&closure, section_offsets(&closure)[2] + 8) + 1,
            ),
        ),
        (
            "duplicate target column identity",
            with_u32(
                direct.clone(),
                direct_columns[1].column_id,
                u32_at(&direct, direct_columns[0].column_id),
            ),
        ),
        (
            "column ordinal order",
            with_u32(direct.clone(), direct_columns[1].ordinal, 0),
        ),
        (
            "dependency ordinal order",
            with_u32(closure.clone(), dependencies[1].ordinal, 0),
        ),
        (
            "duplicate index identity",
            with_u32(
                closure.clone(),
                indexes[1].oid,
                u32_at(&closure, indexes[0].oid),
            ),
        ),
        (
            "index ordinal order",
            with_u32(closure, indexes[1].raw_ordinal, 0),
        ),
    ];
    for (name, forged) in duplicate_and_order_cases {
        assert!(decode(&forged).is_err(), "{name} forgery decoded");
    }
}
