/// NON-VACUOUS layout check across ALL section types (opus audit finding): the prior test compared
/// the delegating wrapper to `..._with_capacity(.., row_count)` — the SAME call — so it could never
/// fail, and it only used int4. This asserts the EXACT dense layout of a multi-type schema at 32 rows
/// (a 32-word bitmap boundary), so a wrong section size — e.g. a bitmap `div_ceil` regression — shifts
/// the total length / section offsets and is caught. (Text is omitted: it is data-dependent and
/// self-describing, and is covered by the null/text suites.)
#[test]
fn dense_multitype_payload_layout_is_exact() {
    // Catalog order: id i32, maybe i32 (nullable), big i64, amt numeric (16B), flag bool.
    let names: Vec<String> = ["id", "maybe", "big", "amt", "flag"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let types = vec![
        SqlType::Int4,
        SqlType::Int4,
        SqlType::Int8,
        SqlType::Numeric {
            precision: 18,
            scale: 0,
        },
        SqlType::Bool,
    ];
    let n = 32_usize; // capacity == row_count (dense / production path); crosses a 32-word boundary
    let rows: Vec<Vec<SqlValue>> = (0..n as i64)
        .map(|i| {
            vec![
                SqlValue::Int4(i as i32),
                if i % 4 == 0 {
                    SqlValue::Null
                } else {
                    SqlValue::Int4(i as i32 * 2)
                },
                SqlValue::Int8(i * 1000),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(i as i128, 0)),
                SqlValue::Bool(i % 2 == 0),
            ]
        })
        .collect();
    let p = build_relational_device_payload_with_capacity(&names, &types, &rows, n)
        .unwrap()
        .0;
    // Exact layout: header 8 | i32 x2 (n*4 each) | i64 (n*8) | 16B (n*16) | bool 1 word | null 1 word.
    let words = n.div_ceil(32); // 1 at n=32; a div_ceil regression makes this 2 -> length changes
    let i64_off = 8 + 2 * n * 4;
    let b128_off = i64_off + n * 8;
    let bool_off = b128_off + n * 16;
    let null_off = bool_off + words * 4;
    let expected_len = null_off + words * 4;
    assert_eq!(
        p.len(),
        expected_len,
        "exact dense multi-type payload length"
    );
    assert_eq!(
        u64::from_le_bytes(p[0..8].try_into().unwrap()),
        n as u64,
        "header = live row count"
    );
    // i64 section spot-check: row 5 = 5000.
    let off = i64_off + 5 * 8;
    assert_eq!(
        i64::from_le_bytes(p[off..off + 8].try_into().unwrap()),
        5000
    );
    // bool bitmap: row0 flag=true -> bit0 set; row1 flag=false -> bit1 clear.
    let bool_word = u32::from_le_bytes(p[bool_off..bool_off + 4].try_into().unwrap());
    assert_eq!(bool_word & 0b11, 0b01, "flag bits: row0 set, row1 clear");
    // NULL validity bitmap (1 = present): row0=NULL -> bit0 clear; row1=present -> bit1 set.
    let null_word = u32::from_le_bytes(p[null_off..null_off + 4].try_into().unwrap());
    assert_eq!(
        null_word & 0b11,
        0b10,
        "validity bits: row0 NULL, row1 present"
    );
}

/// Proves the offset helpers are CAPACITY-aware (opus audit #5: the only thing that actually
/// exercises the capacity stride through the helpers — the regression only covers capacity ==
/// row_count). With capacity > row_count the int8 section must start AFTER the capacity-padded int4
/// sections, not the row_count-sized ones.
#[test]
fn offset_helpers_use_capacity_not_row_count() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE t (a INT, b INT, c BIGINT)")
        .expect("create");
    let table = engine.relational_catalog_table("t").expect("table");

    let snapshot = |capacity: usize| RelationalResidencySnapshot {
        gpu_id: 0,
        schema: "public".to_string(),
        table: "t".to_string(),
        generation: 0,
        row_count: 3,
        capacity,
        column_count: 3,
        resident_bytes: 0,
        resident_device_int4_columns: vec!["a".to_string(), "b".to_string()],
        resident_device_int4_column_stats: vec![],
        resident_device_int8_columns: vec!["c".to_string()],
        resident_device_numeric_columns: vec![],
        resident_device_bool_columns: vec![],
        resident_device_text_columns: vec![],
        resident_device_null_columns: vec![],
        valid_through_index: 0,
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        last_refresh_cost: None,
        admission_budget_bytes: None,
        resident_bytes_after_admission: 0,
        evicted_tables_on_admission: vec![],
        device_memory_proof: None,
    };

    // capacity = 8 > row_count = 3: each int4 section is capacity*4 = 32 bytes.
    let s8 = snapshot(8);
    assert_eq!(
        resident_device_int4_column_offset(&s8, &table, 0).unwrap(),
        8
    );
    assert_eq!(
        resident_device_int4_column_offset(&s8, &table, 1).unwrap(),
        8 + 8 * 4
    );
    // int8 col `c` starts AFTER both capacity-padded int4 sections: 8 + 2*(8*4) = 72.
    assert_eq!(
        resident_device_int8_column_offset(&s8, &table, 2).unwrap(),
        8 + 2 * 8 * 4
    );

    // Dense (capacity == row_count == 3): int8 col `c` at 8 + 2*(3*4) = 32 — proving capacity, not
    // row_count, drives the stride (a row_count stride would give 32 for BOTH cases).
    let s3 = snapshot(3);
    assert_eq!(
        resident_device_int8_column_offset(&s3, &table, 2).unwrap(),
        8 + 2 * 3 * 4
    );
}

/// 1b-ii: int4 open-shard append chunks land at the capacity-aware read offsets (column c at
/// `8 + c*capacity*4`, row r at `+ r*4`), header LAST (partial-failure contract), with eligibility
/// + capacity-overflow rejection (the full-rebuild fallback).
#[test]
fn open_shard_int4_append_chunks_match_capacity_layout() {
    let types = vec![SqlType::Int4, SqlType::Int4]; // id, balance
    let capacity = 8;
    let row_start = 3;
    let new_rows = vec![
        vec![SqlValue::Int4(3), SqlValue::Int4(30)],
        vec![SqlValue::Int4(4), SqlValue::Int4(40)],
    ];
    let chunks =
        compute_open_shard_int4_append_chunks(&types, capacity, row_start, &new_rows).unwrap();
    assert_eq!(chunks.len(), 3, "2 column chunks + 1 header chunk");
    let le = |vals: &[i32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    // col0 (id): section 8 + 0*8*4 = 8; row 3 -> 8 + 3*4 = 20; bytes [3,4].
    assert_eq!(chunks[0].byte_offset, 8 + 3 * 4);
    assert_eq!(chunks[0].bytes, le(&[3, 4]));
    // col1 (balance): section 8 + 1*8*4 = 40; row 3 -> 52; bytes [30,40].
    assert_eq!(chunks[1].byte_offset, 8 + 8 * 4 + 3 * 4);
    assert_eq!(chunks[1].bytes, le(&[30, 40]));
    // header LAST: offset 0, live row count = row_start + 2 = 5.
    assert_eq!(chunks[2].byte_offset, 0);
    assert_eq!(chunks[2].bytes, 5_u64.to_le_bytes().to_vec());

    // ineligible (text) -> Err (caller falls back to re-admit).
    assert!(compute_open_shard_int4_append_chunks(
        &[SqlType::Int4, SqlType::Text],
        capacity,
        0,
        &[]
    )
    .is_err());
    // capacity overflow -> Err (caller seals + rolls a new shard).
    let two = vec![vec![SqlValue::Int4(0)], vec![SqlValue::Int4(1)]];
    assert!(compute_open_shard_int4_append_chunks(&[SqlType::Int4], 4, 3, &two).is_err());

    // value-encoding arms (opus coverage note): Int2 widens, Date passes, NULL -> 0.
    let mixed = compute_open_shard_int4_append_chunks(
        &[SqlType::Int2, SqlType::Date],
        4,
        0,
        &[
            vec![SqlValue::Int2(7), SqlValue::Date(100)],
            vec![SqlValue::Null, SqlValue::Null],
        ],
    )
    .unwrap();
    assert_eq!(mixed[0].bytes, le(&[7, 0]), "int2 widened + null->0");
    assert_eq!(
        mixed[1].byte_offset,
        8 + 4 * 4,
        "date section after the int2 section"
    );
    assert_eq!(mixed[1].bytes, le(&[100, 0]), "date pass-through + null->0");
    // a wrong-typed value in an eligible column -> Err (clean, no panic).
    assert!(compute_open_shard_int4_append_chunks(
        &[SqlType::Int4],
        4,
        0,
        &[vec![SqlValue::Int8(1)]]
    )
    .is_err());
    // a malformed (short) row -> Err, never a panic (the row-arity guard).
    assert!(compute_open_shard_int4_append_chunks(
        &[SqlType::Int4, SqlType::Int4],
        4,
        0,
        &[vec![SqlValue::Int4(1)]]
    )
    .is_err());
}
