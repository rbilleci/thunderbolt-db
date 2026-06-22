//! Slice-1 (M3 NULL) value-model foundation tests.
//!
//! These cover the *representation* of `SqlValue::Null` below the executors — the
//! host comparison/predicate semantics, the storage encode/decode round-trip, and
//! the value-index key — before any parse or GPU path can *produce* a NULL. See
//! `docs/architecture/21-null-representation-and-three-valued-logic.md`.

use super::*;

use std::cmp::Ordering;

use gpu_db_sql::{SelectFilterOp, SqlType, SqlValue};

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
