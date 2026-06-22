//! M3 NULL representation tests.
//!
//! Slice 1 (below) covers the value model — host comparison/predicate semantics, the
//! storage encode/decode round-trip, and the value-index key. Slice 2 (at the bottom)
//! covers the GPU-resident null **validity bitmap** built into the device payload. Both
//! are exercised before any parse/GPU path can *produce* a NULL. See
//! `docs/architecture/21-null-representation-and-three-valued-logic.md`.

use super::*;

use std::cmp::Ordering;

use gpu_db_sql::{SelectFilterOp, SqlType, SqlValue};

use crate::engine_residency::build_relational_device_payload;

fn column(name: &str, ty: SqlType) -> RelationalColumn {
    RelationalColumn {
        id: 1,
        table_oid: 1,
        attnum: 1,
        name: name.to_string(),
        ty,
        domain: None,
        default: None,
        type_oid: ty.postgres_oid(),
        type_size: ty.type_size(),
    }
}

const ALL_OPS: [SelectFilterOp; 6] = [
    SelectFilterOp::Eq,
    SelectFilterOp::Lt,
    SelectFilterOp::Lte,
    SelectFilterOp::Gt,
    SelectFilterOp::Gte,
    SelectFilterOp::LikePrefix,
];

#[test]
fn null_operand_excludes_the_row_for_every_comparison_op() {
    // SQL three-valued logic: a comparison with a NULL operand is UNKNOWN, and an
    // UNKNOWN WHERE predicate excludes the row (never TRUE). This must hold for EVERY
    // op and for NULL on either side (and both).
    for op in ALL_OPS {
        assert!(
            !select_filter_matches(&SqlValue::Null, op, &SqlValue::Int4(5)),
            "NULL on the left must not match for {op:?}"
        );
        assert!(
            !select_filter_matches(&SqlValue::Int4(5), op, &SqlValue::Null),
            "NULL on the right must not match for {op:?}"
        );
        assert!(
            !select_filter_matches(&SqlValue::Null, op, &SqlValue::Null),
            "NULL = NULL is UNKNOWN, must not match for {op:?}"
        );
        // Text operands too (the LikePrefix path and text equality).
        assert!(
            !select_filter_matches(&SqlValue::Null, op, &SqlValue::Text("a".to_string())),
            "NULL vs text must not match for {op:?}"
        );
    }
}

#[test]
fn non_null_comparisons_are_unchanged_by_the_null_short_circuit() {
    // Regression guard: the NULL short-circuit must not perturb the existing
    // non-null comparison behavior.
    assert!(select_filter_matches(
        &SqlValue::Int4(5),
        SelectFilterOp::Eq,
        &SqlValue::Int4(5)
    ));
    assert!(select_filter_matches(
        &SqlValue::Int4(4),
        SelectFilterOp::Lt,
        &SqlValue::Int4(5)
    ));
    assert!(!select_filter_matches(
        &SqlValue::Int4(6),
        SelectFilterOp::Lt,
        &SqlValue::Int4(5)
    ));
    assert!(select_filter_matches(
        &SqlValue::Text("abc".to_string()),
        SelectFilterOp::LikePrefix,
        &SqlValue::Text("ab".to_string())
    ));
}

#[test]
fn compare_sql_values_is_total_with_null_sorting_lowest() {
    // The INTERNAL total order (value-index/dedup only) must stay total: NULL equals
    // itself and sorts below every typed value, regardless of the other side's type.
    assert_eq!(
        compare_sql_values(&SqlValue::Null, &SqlValue::Null),
        Ordering::Equal
    );
    for other in [
        SqlValue::Int4(0),
        SqlValue::Int8(-1),
        SqlValue::Bool(false),
        SqlValue::Text(String::new()),
        SqlValue::Uuid([0u8; 16]),
    ] {
        assert_eq!(
            compare_sql_values(&SqlValue::Null, &other),
            Ordering::Less,
            "NULL must sort below {other:?}"
        );
        assert_eq!(
            compare_sql_values(&other, &SqlValue::Null),
            Ordering::Greater,
            "{other:?} must sort above NULL"
        );
    }
}

#[test]
fn null_round_trips_through_storage_encode_decode_for_any_column_type() {
    // A stored NULL encodes type-independently and decodes back to NULL for ANY
    // column type. Non-null neighbours in the same row are unaffected.
    let row = vec![
        SqlValue::Int4(7),
        SqlValue::Null,
        SqlValue::Text("hi".to_string()),
    ];
    let encoded = encode_relational_row(&row);
    // The NULL field is the reserved prefix-free `null` token.
    assert_eq!(encoded, "i:7|null|t:hi");

    let columns = [
        column("a", SqlType::Int4),
        column("b", SqlType::Text),
        column("c", SqlType::Text),
    ];
    let decoded = decode_relational_row(&encoded, &columns).unwrap();
    assert_eq!(decoded, row);

    // NULL decodes to NULL irrespective of the declared column type.
    for ty in [
        SqlType::Int4,
        SqlType::Int8,
        SqlType::Bool,
        SqlType::Text,
        SqlType::Date,
        SqlType::Timestamp,
        SqlType::Uuid,
        SqlType::Numeric {
            precision: 10,
            scale: 2,
        },
    ] {
        assert_eq!(
            decode_relational_value("null", &column("x", ty)).unwrap(),
            SqlValue::Null,
            "the `null` token must decode to NULL for {ty:?}"
        );
    }
}

#[test]
fn the_text_value_null_does_not_collide_with_the_null_sentinel() {
    // A genuine text value "null" is stored WITH the `t:` prefix, so it must decode
    // back to Text("null"), never to SqlValue::Null. This is what keeps the sentinel
    // unambiguous.
    let encoded = encode_relational_row(&[SqlValue::Text("null".to_string())]);
    assert_eq!(encoded, "t:null");
    let decoded = decode_relational_value("t:null", &column("c", SqlType::Text)).unwrap();
    assert_eq!(decoded, SqlValue::Text("null".to_string()));
}

#[test]
fn null_value_index_key_is_distinct_from_every_typed_key() {
    // The NULL value-index key is the bare `null` token; no typed value (all of which
    // are type-prefixed) can produce it — in particular the text "null" keys to
    // `t:null`, not `null`.
    assert_eq!(relational_index_value(&SqlValue::Null), "null");
    assert_ne!(
        relational_index_value(&SqlValue::Null),
        relational_index_value(&SqlValue::Text("null".to_string()))
    );
}

// ============================================================================
// Slice 2 — the GPU-resident null VALIDITY bitmap in the device payload.
// `build_relational_device_payload` is what gets memcpy'd to the device verbatim, so
// verifying its bytes + the returned layouts verifies what is resident on the GPU.
// ============================================================================

/// Read row `i`'s validity bit (1 = valid/present, 0 = NULL) from the LE u32 bitmap at `offset`.
fn validity_bit(payload: &[u8], offset: u64, i: usize) -> bool {
    let word_off = offset as usize + (i / 32) * 4;
    let word = u32::from_le_bytes(payload[word_off..word_off + 4].try_into().unwrap());
    (word >> (i % 32)) & 1 == 1
}

#[test]
fn a_payload_with_no_nulls_emits_no_bitmap_and_stays_byte_identical() {
    // The whole point of "absence ⇒ all-valid": a no-NULL payload must add ZERO bytes and carry an
    // empty null-column list, so pre-M3 residency is unchanged.
    let names = vec!["a".to_string(), "b".to_string()];
    let types = vec![SqlType::Int4, SqlType::Text];
    let rows = vec![
        vec![SqlValue::Int4(1), SqlValue::Text("x".to_string())],
        vec![SqlValue::Int4(2), SqlValue::Text("yy".to_string())],
    ];
    let (payload, _text, _bool, _int4, _b128, null_cols) =
        build_relational_device_payload(&names, &types, &rows).unwrap();
    assert!(null_cols.is_empty(), "no NULLs => no validity bitmap");
    // header(8) + int4(2*4=8) -> 16 (8-aligned, no text pad) + offsets(3*8=24) + bytes("x"+"yy"=3).
    assert_eq!(payload.len(), 8 + 8 + 24 + 3, "no extra null-bitmap bytes were added");
}

#[test]
fn a_nullable_int4_column_gets_a_validity_bitmap_zero_placeholder_and_null_excluded_stats() {
    let names = vec!["a".to_string()];
    let types = vec![SqlType::Int4];
    let rows = vec![
        vec![SqlValue::Int4(10)],
        vec![SqlValue::Null],
        vec![SqlValue::Int4(30)],
    ];
    let (payload, _text, _bool, int4_stats, _b128, null_cols) =
        build_relational_device_payload(&names, &types, &rows).unwrap();

    assert_eq!(null_cols.len(), 1);
    assert_eq!(null_cols[0].name, "a");
    let off = null_cols[0].bitmap_byte_offset;
    assert_eq!(off % 4, 0, "the u32 validity bitmap must start 4-aligned");
    assert!(validity_bit(&payload, off, 0), "row 0 (10) is valid");
    assert!(!validity_bit(&payload, off, 1), "row 1 (NULL) is invalid");
    assert!(validity_bit(&payload, off, 2), "row 2 (30) is valid");

    // The int4 section wrote a don't-care 0 at the NULL row (row 1 occupies bytes [12..16]).
    let placeholder = i32::from_le_bytes(payload[12..16].try_into().unwrap());
    assert_eq!(placeholder, 0, "NULL int4 cell is a 0 placeholder");

    // Stats EXCLUDE the NULL: a NULL must not pull min toward 0.
    assert_eq!(int4_stats[0].min, 10);
    assert_eq!(int4_stats[0].max, 30);
}

#[test]
fn null_in_a_text_column_emits_an_empty_span_and_a_validity_bitmap() {
    let names = vec!["t".to_string()];
    let types = vec![SqlType::Text];
    let rows = vec![
        vec![SqlValue::Text("ab".to_string())],
        vec![SqlValue::Null],
        vec![SqlValue::Text("c".to_string())],
    ];
    let (payload, text_layouts, _bool, _int4, _b128, null_cols) =
        build_relational_device_payload(&names, &types, &rows).unwrap();

    assert_eq!(null_cols.len(), 1);
    assert_eq!(null_cols[0].name, "t");
    let off = null_cols[0].bitmap_byte_offset;
    assert!(validity_bit(&payload, off, 0));
    assert!(!validity_bit(&payload, off, 1));
    assert!(validity_bit(&payload, off, 2));

    // The NULL row contributes a zero-length text span: offsets[1] == offsets[2].
    let tl = &text_layouts[0];
    let read_off = |k: usize| {
        let p = tl.offsets_byte_offset as usize + k * 8;
        u64::from_le_bytes(payload[p..p + 8].try_into().unwrap())
    };
    assert_eq!(read_off(1), read_off(2), "NULL text row spans zero bytes");
    assert_eq!(tl.bytes_len, 3, "only \"ab\" + \"c\" contribute bytes");
}

#[test]
fn only_columns_that_contain_a_null_get_a_bitmap_in_catalog_order() {
    let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let types = vec![SqlType::Int4, SqlType::Int8, SqlType::Int4];
    let rows = vec![
        vec![SqlValue::Int4(1), SqlValue::Int8(100), SqlValue::Null],
        vec![SqlValue::Null, SqlValue::Int8(200), SqlValue::Int4(9)],
    ];
    let (payload, _text, _bool, _int4, _b128, null_cols) =
        build_relational_device_payload(&names, &types, &rows).unwrap();

    // `a` and `c` contain a NULL → bitmaps, in catalog order; `b` has none → no bitmap.
    let got: Vec<&str> = null_cols.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(got, vec!["a", "c"]);

    // The two bitmaps are at distinct, increasing, 4-aligned offsets.
    assert!(null_cols[0].bitmap_byte_offset < null_cols[1].bitmap_byte_offset);
    for layout in &null_cols {
        assert_eq!(layout.bitmap_byte_offset % 4, 0);
    }
    // `a`'s validity: row 0 valid, row 1 NULL.
    assert!(validity_bit(&payload, null_cols[0].bitmap_byte_offset, 0));
    assert!(!validity_bit(&payload, null_cols[0].bitmap_byte_offset, 1));
    // `c`'s validity: row 0 NULL, row 1 valid.
    assert!(!validity_bit(&payload, null_cols[1].bitmap_byte_offset, 0));
    assert!(validity_bit(&payload, null_cols[1].bitmap_byte_offset, 1));
}
