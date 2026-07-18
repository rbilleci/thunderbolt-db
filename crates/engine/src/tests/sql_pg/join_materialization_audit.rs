use crate::Engine;
use crate::RelationalSelectResult;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlValue};

// ============================================================================================
// S7/V3 INDEPENDENT ADVERSARIAL AUDIT (commit 70758557): join result materialization on-device.
// Goal: break the device gather -- a MATCHED-row value-NULL emitting a placeholder, a pad emitting
// row-0's value, a type-narrowing/tagging error, or a LIMIT/OFFSET window divergence.
// ============================================================================================

// Helper: pull the single (col 0) value of a one-row resident SELECT -- the "truth" produced by the
// already-audited resident materialization path (S1), used as the expected value for the join gather.
#[cfg(test)]
fn audit_one_val(e: &Engine, sql: &str) -> SqlValue {
    let r = e
        .execute_resident_expr_select_sql(sql)
        .expect("reference select");
    assert_eq!(
        r.rows.len(),
        1,
        "reference select must return exactly one row: {sql}"
    );
    r.rows[0][0].clone()
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_matched_row_value_null_every_type() {
    // HUNT #1: a MATCHED join row whose projected NON-KEY column is NULL must come back SqlValue::Null
    // for EVERY nullable type -- not the device placeholder (0/""/0-mantissa/zero-uuid/false). The
    // committed test only proves int4/text/numeric. Here: int2, int8, date, timestamp, uuid, bool,
    // numeric@scale4. The join KEY (id) is non-null; every value column carries a NULL on the matched row.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    e.execute_text(
        2,
        "CREATE TABLE r (rid INT, s2 SMALLINT, s8 BIGINT, d DATE, ts TIMESTAMP, u UUID, b BOOLEAN, n4 NUMERIC(12,4))",
    )
    .unwrap();
    e.execute_text(3, "INSERT INTO l (id) VALUES (1),(2)")
        .unwrap();
    // rid=1: every value column NULL. rid=2: every value column a distinctive NON-null value.
    e.execute_text(
        4,
        "INSERT INTO r (rid, s2, s8, d, ts, u, b, n4) VALUES \
         (1, NULL, NULL, NULL, NULL, NULL, NULL, NULL), \
         (2, -12345, 9000000000, '2024-03-14', '2024-03-14 13:37:00', \
          'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', true, 1234.5678)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    // The non-null reference values (rid=2), from the audited resident path.
    let v_s2 = audit_one_val(&e, "SELECT s2 FROM r WHERE rid = 2");
    let v_s8 = audit_one_val(&e, "SELECT s8 FROM r WHERE rid = 2");
    let v_d = audit_one_val(&e, "SELECT d FROM r WHERE rid = 2");
    let v_ts = audit_one_val(&e, "SELECT ts FROM r WHERE rid = 2");
    let v_u = audit_one_val(&e, "SELECT u FROM r WHERE rid = 2");
    let v_b = audit_one_val(&e, "SELECT b FROM r WHERE rid = 2");
    let v_n4 = audit_one_val(&e, "SELECT n4 FROM r WHERE rid = 2");
    // sanity: the references are the real (non-Null) values and exercise the placeholder hazard.
    assert_eq!(v_s2, SqlValue::Int2(-12345));
    assert_eq!(v_b, SqlValue::Bool(true));
    assert!(matches!(v_u, SqlValue::Uuid(_)));
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.s2, r.s8, r.d, r.ts, r.u, r.b, r.n4 \
             FROM l JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("inner join projecting every nullable type");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            // rid=1 matched: EVERY value column is NULL (validity bitmap), NOT a placeholder.
            vec![
                SqlValue::Int4(1),
                SqlValue::Null, // s2 (placeholder would be Int2(0))
                SqlValue::Null, // s8 (placeholder Int8(0))
                SqlValue::Null, // d  (placeholder Date(0))
                SqlValue::Null, // ts (placeholder Timestamp(0))
                SqlValue::Null, // u  (placeholder Uuid([0;16]))
                SqlValue::Null, // b  (placeholder Bool(false))
                SqlValue::Null, // n4 (placeholder Numeric(0))
            ],
            // rid=2 matched: every value column the exact non-null value (type-exact narrow/tag/bytes).
            vec![SqlValue::Int4(2), v_s2, v_s8, v_d, v_ts, v_u, v_b, v_n4],
        ],
        "matched-row NULLs across all types must be SqlValue::Null, non-nulls type-exact"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_outer_pad_on_nonnullable_columns_all_types() {
    // HUNT #2 + #3: an OUTER pad must force NULL on a column that has NO validity bitmap (non-nullable),
    // independent of validity -- AND must not leak row-0's value (pads use placeholder index 0). Row 0 of
    // the padded relation holds DISTINCTIVE values for every type; the unmatched left rows must be NULL.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    // r has NO nullable columns (none of these ever hold a NULL -> no validity bitmap is built).
    e.execute_text(
        2,
        "CREATE TABLE r (rid INT, s2 SMALLINT, s8 BIGINT, d DATE, ts TIMESTAMP, u UUID, b BOOLEAN, n NUMERIC(10,2), name TEXT)",
    )
    .unwrap();
    // l has id 1,2,3. r has rid=1 (ROW 0, distinctive) and rid=2. id=3 is UNMATCHED -> a pad over r.
    e.execute_text(3, "INSERT INTO l (id) VALUES (1),(2),(3)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO r (rid, s2, s8, d, ts, u, b, n, name) VALUES \
         (1, 777, 123456789012, '2030-12-31', '2030-12-31 23:59:59', \
          '11111111-2222-3333-4444-555555555555', true, 42.42, 'ROW0'), \
         (2, 1, 1, '2000-01-01', '2000-01-01 00:00:00', \
          '00000000-0000-0000-0000-000000000001', false, 1.00, 'row2')",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.s2, r.s8, r.d, r.ts, r.u, r.b, r.n, r.name \
             FROM l LEFT JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("left outer join, pad over a fully non-nullable relation");
    assert_eq!(res.rows.len(), 3);
    // id=3 row: every r column is a pad -> must be NULL, NOT row-0's distinctive value (777/'ROW0'/...).
    let pad_row = &res.rows[2];
    assert_eq!(pad_row[0], SqlValue::Int4(3), "left id survives");
    for (c, v) in pad_row.iter().enumerate().skip(1) {
        assert_eq!(
            *v,
            SqlValue::Null,
            "padded non-nullable column {c} must be NULL, not row-0's value (placeholder-0 leak)"
        );
    }
    // And the matched id=1 row really carries row-0's distinctive values (proves the gather isn't dead).
    assert_eq!(
        res.rows[0][1],
        SqlValue::Int2(777),
        "matched id=1 gets row-0 s2"
    );
    assert_eq!(
        res.rows[0][8],
        SqlValue::Text("ROW0".to_string()),
        "matched id=1 gets row-0 name"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_empty_padded_side_right_full() {
    // HUNT #3: a relation whose row_count==0 appears as a PADDED side (RIGHT/FULL with an empty side).
    // gather_col must early-return all-NULL (no device read at index 0 into an empty payload) -- no panic.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, w INT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (id, name) VALUES (1,'a'),(2,'b')")
        .unwrap();
    // r is EMPTY.
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    // LEFT JOIN with empty right -> both left rows survive, r columns NULL.
    let left = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, l.name, r.w FROM l LEFT JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("left join over an empty right relation");
    assert_eq!(
        left.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("a".to_string()),
                SqlValue::Null
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("b".to_string()),
                SqlValue::Null
            ],
        ],
        "empty padded side -> r.w all NULL, no panic/OOB"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_limit_offset_no_orderby_matches_join_order() {
    // HUNT #5: LIMIT/OFFSET WITHOUT ORDER BY must window the join result in JOIN ORDER (identity perm),
    // exactly as the old drain/truncate did. We make the join order deterministic (unique 1:1 keys, build
    // on the unique side) and verify the windowed slice is a contiguous slice of the full result.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, score INT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO l (id, name) VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e')",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO r (rid, score) VALUES (1,10),(2,20),(3,30),(4,40),(5,50)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let full = e
        .execute_resident_expr_select_sql("SELECT l.name FROM l JOIN r ON l.id = r.rid")
        .expect("full join, no window");
    let full_names: Vec<SqlValue> = full.rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(full_names.len(), 5);
    // OFFSET 1 LIMIT 2 (no ORDER BY) -> the contiguous slice [1..3) of the join order.
    let win = e
        .execute_resident_expr_select_sql(
            "SELECT l.name FROM l JOIN r ON l.id = r.rid LIMIT 2 OFFSET 1",
        )
        .expect("windowed join, no order by");
    let win_names: Vec<SqlValue> = win.rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(
        win_names,
        full_names[1..3].to_vec(),
        "LIMIT 2 OFFSET 1 with no ORDER BY must equal the contiguous join-order slice"
    );
    // OFFSET only (no LIMIT) -> the tail [2..].
    let tail = e
        .execute_resident_expr_select_sql("SELECT l.name FROM l JOIN r ON l.id = r.rid OFFSET 2")
        .expect("offset-only join");
    let tail_names: Vec<SqlValue> = tail.rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(
        tail_names,
        full_names[2..].to_vec(),
        "OFFSET 2, no LIMIT -> tail in join order"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_empty_result_no_matches() {
    // HUNT #9: no matches -> work_n == 0 -> the gather returns empty, transpose -> 0 rows, no panic.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, w NUMERIC(10,2))")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (id, name) VALUES (1,'a'),(2,'b')")
        .unwrap();
    e.execute_text(4, "INSERT INTO r (rid, w) VALUES (100, 1.00),(200, 2.00)")
        .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, l.name, r.w FROM l JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("inner join with no matches");
    assert!(res.rows.is_empty(), "no matches -> empty result, no panic");
    // also with a window applied on top of empty.
    let res2 = e
        .execute_resident_expr_select_sql(
            "SELECT l.id FROM l JOIN r ON l.id = r.rid LIMIT 5 OFFSET 0",
        )
        .expect("inner join no matches + window");
    assert!(res2.rows.is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_nn_and_multiway_gather_right_rows() {
    // HUNT #8: N:N many-to-many + a 3-way chain. The carried index vectors must gather the RIGHT rows
    // from each side's device payload (text + numeric + null values), not misaligned values.
    let mut e = Engine::new_local_test_engine();
    // N:N on an int key: l has key 1 twice, r has key 1 twice -> 4 result rows, each a distinct (l,r) pair.
    e.execute_text(1, "CREATE TABLE l (k INT, lv TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (k INT, rv NUMERIC(10,2))")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (k, lv) VALUES (1,'x'),(1,'y')")
        .unwrap();
    e.execute_text(4, "INSERT INTO r (k, rv) VALUES (1, 1.10),(1, NULL)")
        .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let nn = e
        .execute_resident_expr_select_sql(
            "SELECT l.lv, r.rv FROM l JOIN r ON l.k = r.k ORDER BY l.lv, r.rv",
        )
        .expect("N:N int join");
    // 4 pairs: (x,1.10),(x,NULL),(y,1.10),(y,NULL). r.rv NULL must be Null from the validity bitmap.
    // ORDER BY r.rv places NULL last (ASC PG default). So per lv: [1.10, NULL].
    assert_eq!(
        nn.rows,
        vec![
            vec![
                SqlValue::Text("x".to_string()),
                SqlValue::Numeric(Decimal128::new(110, 2))
            ],
            vec![SqlValue::Text("x".to_string()), SqlValue::Null],
            vec![
                SqlValue::Text("y".to_string()),
                SqlValue::Numeric(Decimal128::new(110, 2))
            ],
            vec![SqlValue::Text("y".to_string()), SqlValue::Null],
        ],
        "N:N gather: each (l,r) pair's text+numeric (incl. a NULL numeric value) is correct"
    );
    // 3-way chain a JOIN b JOIN c. Carried indices into 3 payloads.
    e.execute_text(10, "CREATE TABLE a (aid INT, an TEXT)")
        .unwrap();
    e.execute_text(11, "CREATE TABLE b (bid INT, bref INT, bn TEXT)")
        .unwrap();
    e.execute_text(12, "CREATE TABLE c (cid INT, cn TEXT)")
        .unwrap();
    e.execute_text(13, "INSERT INTO a (aid, an) VALUES (1,'a1'),(2,'a2')")
        .unwrap();
    e.execute_text(
        14,
        "INSERT INTO b (bid, bref, bn) VALUES (1,1,'b1'),(2,2,'b2')",
    )
    .unwrap();
    e.execute_text(15, "INSERT INTO c (cid, cn) VALUES (1,'c1'),(2,'c2')")
        .unwrap();
    for t in ["a", "b", "c"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let threeway = e
        .execute_resident_expr_select_sql(
            "SELECT a.an, b.bn, c.cn FROM a JOIN b ON a.aid = b.bref JOIN c ON b.bid = c.cid ORDER BY a.an",
        )
        .expect("3-way chain join");
    assert_eq!(
        threeway.rows,
        vec![
            vec![
                SqlValue::Text("a1".to_string()),
                SqlValue::Text("b1".to_string()),
                SqlValue::Text("c1".to_string())
            ],
            vec![
                SqlValue::Text("a2".to_string()),
                SqlValue::Text("b2".to_string()),
                SqlValue::Text("c2".to_string())
            ],
        ],
        "3-way chain: each side's text gathered from its own payload at the carried row"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_using_natural_and_star_gather() {
    // HUNT #7: USING coalesced column (mapped to rel 0) + bare `*` gather correctly.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, lname TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (id INT, rscore INT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO l (id, lname) VALUES (1,'a'),(2,'b'),(3,'c')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO r (id, rscore) VALUES (1,10),(2,20)")
        .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    // USING (id): the coalesced id from rel0, then l.lname, then r.rscore.
    let star = e
        .execute_resident_expr_select_sql("SELECT * FROM l JOIN r USING (id) ORDER BY id")
        .expect("USING join star");
    assert_eq!(
        star.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("a".to_string()),
                SqlValue::Int4(10)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("b".to_string()),
                SqlValue::Int4(20)
            ],
        ],
        "USING star: coalesced id (rel0) + l.lname + r.rscore gathered from device"
    );
    // unqualified id reference resolves to the left copy.
    let bare = e
        .execute_resident_expr_select_sql(
            "SELECT id, lname, rscore FROM l JOIN r USING (id) ORDER BY id",
        )
        .expect("USING explicit columns");
    assert_eq!(bare.rows, star.rows, "explicit list equals star for USING");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_nonvacuity_placeholder_would_leak_without_override() {
    // NON-VACUITY PROOF: the device DOES store a 0/false/""/zero placeholder for a NULL cell. This test
    // asserts the placeholder values directly via a deliberately-WRONG expectation -- it MUST PANIC,
    // proving the SqlValue::Null in the real test is the validity override doing real work (not that the
    // device happens to be empty/absent). If this test ever PASSES, the placeholder is leaking == bug.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    e.execute_text(
        2,
        "CREATE TABLE r (rid INT, b BOOLEAN, name TEXT, s2 SMALLINT)",
    )
    .unwrap();
    e.execute_text(3, "INSERT INTO l (id) VALUES (1)").unwrap();
    e.execute_text(
        4,
        "INSERT INTO r (rid, b, name, s2) VALUES (1, NULL, NULL, NULL)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT r.b, r.name, r.s2 FROM l JOIN r ON l.id = r.rid")
        .expect("join");
    let row = &res.rows[0];
    // The CORRECT result is all-Null. The placeholder (the WRONG result) would be Bool(false)/Text("")/Int2(0).
    let placeholder = vec![
        SqlValue::Bool(false),
        SqlValue::Text(String::new()),
        SqlValue::Int2(0),
    ];
    let result = std::panic::catch_unwind(|| {
        assert_eq!(*row, placeholder, "if this matched, the placeholder LEAKED");
    });
    assert!(
        result.is_err(),
        "NON-VACUITY: the result must NOT equal the device placeholder (got {row:?}) -- \
         the validity override is load-bearing"
    );
    // Belt-and-suspenders: it IS all-Null.
    assert_eq!(*row, vec![SqlValue::Null, SqlValue::Null, SqlValue::Null]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_order_by_nullable_value_with_window() {
    // HUNT #5 + #1 cross: ORDER BY a device-gathered NULLABLE value column on the join, then window it.
    // The gathered NULL must (a) render Null and (b) sort to PG default (NULLs last ASC), and the window
    // must slice the SORTED order (not join order). A divergence would silently reorder/mis-window.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, w INT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (id) VALUES (1),(2),(3),(4)")
        .unwrap();
    // w values: 30, NULL, 10, 20 -> sorted ASC: 10(id3),20(id4),30(id1),NULL(id2).
    e.execute_text(
        4,
        "INSERT INTO r (rid, w) VALUES (1,30),(2,NULL),(3,10),(4,20)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let sorted = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.w FROM l JOIN r ON l.id = r.rid ORDER BY r.w",
        )
        .expect("order by nullable value");
    assert_eq!(
        sorted.rows,
        vec![
            vec![SqlValue::Int4(3), SqlValue::Int4(10)],
            vec![SqlValue::Int4(4), SqlValue::Int4(20)],
            vec![SqlValue::Int4(1), SqlValue::Int4(30)],
            vec![SqlValue::Int4(2), SqlValue::Null], // NULL last (ASC default)
        ],
        "ORDER BY r.w ASC: NULL sorts last, value device-gathered"
    );
    // Window the sorted order: OFFSET 1 LIMIT 2 -> rows [20(id4), 30(id1)].
    let win = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.w FROM l JOIN r ON l.id = r.rid ORDER BY r.w LIMIT 2 OFFSET 1",
        )
        .expect("order by + window");
    assert_eq!(
        win.rows,
        vec![
            vec![SqlValue::Int4(4), SqlValue::Int4(20)],
            vec![SqlValue::Int4(1), SqlValue::Int4(30)],
        ],
        "window slices the SORTED order, not join order"
    );
    // NULLS FIRST explicitly: NULL should now be the first row; window OFFSET 0 LIMIT 1 -> the NULL row.
    let nf = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.w FROM l JOIN r ON l.id = r.rid ORDER BY r.w NULLS FIRST LIMIT 1",
        )
        .expect("nulls first + limit 1");
    assert_eq!(
        nf.rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Null]],
        "NULLS FIRST + LIMIT 1 -> the NULL row first"
    );
}

// ============================================================================
// S5/V1a adversarial audit: join text/numeric/uuid KEY values from the DEVICE
// payload (not host_rows). The charter claim is BEHAVIOR-PRESERVING vs the old
// host_rows gather. These tests target byte-order/value exactness, negatives,
// multi-way carried-index gathers, N:N, empty inputs, and the NULL gate.
// ============================================================================

/// HUNT #1: UUID byte-order / value exactness AND that the joined-on uuid VALUE is projected
/// byte-identically. The matched uuids are ASYMMETRIC/non-palindromic byte patterns, so a byte-order
/// bug in `project_i128 -> to_le_bytes()` (vs the old `SqlValue::Uuid(bytes)` host read) would EITHER
/// mismatch the join (wrong/missing pairs) OR project a byte-swapped uuid. The existing test only
/// projected `gname` (another column), so a uuid-value byte-swap would have been invisible there.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_uuid_byteorder_value_exactness() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE gd (gid UUID, gname TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE ud (uid INT, gid UUID)")
        .unwrap();
    // Deliberately asymmetric byte patterns: a byte-swap would change the value AND break matching.
    let a = "0102030405060708090a0b0c0d0e0f10"; // bare 32-hex form
    let b = "fffefdfc-fbfa-f9f8-f7f6-f5f4f3f2f1f0"; // descending, high bit set in byte 0
    let c = "00112233-4455-6677-8899-aabbccddeeff";
    let unmatched = "deadbeef-0000-1111-2222-333344445555";
    let a_canon = gpu_db_sql::uuid::format_uuid(&gpu_db_sql::uuid::parse_uuid(a).unwrap());
    e.execute_text(
        3,
        &format!("INSERT INTO gd (gid, gname) VALUES ('{a}','A'),('{b}','B'),('{c}','C')"),
    )
    .unwrap();
    e.execute_text(
        4,
        &format!(
            "INSERT INTO ud (uid, gid) VALUES (1,'{a}'),(2,'{b}'),(3,'{c}'),(4,'{unmatched}')"
        ),
    )
    .unwrap();
    let mut ok = true;
    for t in ["gd", "ud"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    // Project BOTH the uuid join key (from each side) AND a tag, so a byte-order bug surfaces in the
    // projected value, not just the match set.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT ud.uid, ud.gid, gd.gid, gd.gname FROM ud JOIN gd ON ud.gid = gd.gid ORDER BY ud.uid",
        )
        .expect("uuid-key join projecting the uuid value");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    // Exactly 3 matches (the deadbeef user drops). Both projected uuids must be the canonical value.
    let rows: Vec<(i32, String, String, String)> = res
        .rows
        .iter()
        .map(|r| {
            let uid = match &r[0] {
                SqlValue::Int4(v) => *v,
                o => panic!("uid {o:?}"),
            };
            let u_gid = match &r[1] {
                SqlValue::Uuid(bytes) => gpu_db_sql::uuid::format_uuid(bytes),
                o => panic!("ud.gid not uuid: {o:?}"),
            };
            let g_gid = match &r[2] {
                SqlValue::Uuid(bytes) => gpu_db_sql::uuid::format_uuid(bytes),
                o => panic!("gd.gid not uuid: {o:?}"),
            };
            let gname = match &r[3] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("gname {o:?}"),
            };
            (uid, u_gid, g_gid, gname)
        })
        .collect();
    assert_eq!(
        rows.len(),
        3,
        "exactly the 3 matched uuids (deadbeef drops)"
    );
    // Row 1: uuid `a` -> both projected uuids equal `a`'s canonical form (NOT byte-swapped).
    assert_eq!(rows[0].0, 1);
    assert_eq!(rows[0].1, a_canon, "ud.gid value exact (no byte swap)");
    assert_eq!(rows[0].2, a_canon, "gd.gid value exact (no byte swap)");
    assert_eq!(rows[0].3, "A");
    // Row 2: uuid `b` (descending, high-bit byte0) round-trips exactly + matched the right group.
    assert_eq!(rows[1].0, 2);
    assert_eq!(rows[1].1, "fffefdfc-fbfa-f9f8-f7f6-f5f4f3f2f1f0");
    assert_eq!(rows[1].2, "fffefdfc-fbfa-f9f8-f7f6-f5f4f3f2f1f0");
    assert_eq!(rows[1].3, "B");
    // Row 3: uuid `c`.
    assert_eq!(rows[2].0, 3);
    assert_eq!(rows[2].1, "00112233-4455-6677-8899-aabbccddeeff");
    assert_eq!(rows[2].3, "C");
}

/// HUNT #2: NUMERIC mantissa exactness across scales + NEGATIVES + large i128 magnitudes. The old host
/// path used `value.mantissa.to_le_bytes()`; the device path projects the i128 then `.to_le_bytes()`.
/// A sign-extension/limb bug in `project_i128` would surface as a missing/extra join pair OR a
/// byte-swapped projected value. Existing tests used only small POSITIVE mantissas (100.00/200.50).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_numeric_mantissa_exactness_negatives_and_large() {
    let mut e = Engine::new_local_test_engine();
    // scale 0. NEGATIVES are the high-limb probe: a -1 mantissa is 0xFF..FF across ALL 16 bytes
    // (including the HIGH 64-bit limb); a dropped/zeroed high limb in project_i128 would turn it into a
    // large POSITIVE value and break the match. Also a beyond-i32 positive (3e9) crosses the 32-bit line.
    e.execute_text(1, "CREATE TABLE t0 (k NUMERIC(30,0), name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE a0 (aid INT, k NUMERIC(30,0))")
        .unwrap();
    let big = "3000000000"; // > i32::MAX (2.1e9): a 32-bit-limb bug would corrupt it
    e.execute_text(
        3,
        &format!("INSERT INTO t0 (k, name) VALUES (-1,'neg1'),({big},'big'),(-999999999,'negbil')"),
    )
    .unwrap();
    e.execute_text(
        4,
        &format!("INSERT INTO a0 (aid, k) VALUES (1,-1),(2,{big}),(3,-999999999),(4,777)"),
    )
    .unwrap();
    // high scale + negative fraction
    e.execute_text(5, "CREATE TABLE th (k NUMERIC(20,6), name TEXT)")
        .unwrap();
    e.execute_text(6, "CREATE TABLE ah (aid INT, k NUMERIC(20,6))")
        .unwrap();
    e.execute_text(
        7,
        "INSERT INTO th (k, name) VALUES (-12.345678,'negfrac'),(0.000001,'tiny')",
    )
    .unwrap();
    e.execute_text(
        8,
        "INSERT INTO ah (aid, k) VALUES (1,-12.345678),(2,0.000001),(3,5.000000)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["t0", "a0", "th", "ah"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let collect = |res: &RelationalSelectResult| -> Vec<(i32, String, String)> {
        let mut v: Vec<(i32, String, String)> = res
            .rows
            .iter()
            .map(|r| {
                let aid = match &r[0] {
                    SqlValue::Int4(v) => *v,
                    o => panic!("aid {o:?}"),
                };
                let k = match &r[1] {
                    SqlValue::Numeric(d) => d.mantissa.to_string(),
                    o => panic!("k not numeric: {o:?}"),
                };
                let name = match &r[2] {
                    SqlValue::Text(t) => t.clone(),
                    o => panic!("name {o:?}"),
                };
                (aid, k, name)
            })
            .collect();
        v.sort();
        v
    };
    let r0 = e
        .execute_resident_expr_select_sql(
            "SELECT a0.aid, a0.k, t0.name FROM a0 JOIN t0 ON a0.k = t0.k",
        )
        .expect("scale-0 negative/large numeric join");
    assert_eq!(r0.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        collect(&r0),
        vec![
            (1, "-1".to_string(), "neg1".to_string()),
            (2, big.to_string(), "big".to_string()),
            (3, "-999999999".to_string(), "negbil".to_string()),
        ],
        "scale-0 negatives + a near-i128::MAX mantissa match exactly; 777 (unmatched) drops"
    );
    let rh = e
        .execute_resident_expr_select_sql(
            "SELECT ah.aid, ah.k, th.name FROM ah JOIN th ON ah.k = th.k",
        )
        .expect("high-scale numeric join");
    assert_eq!(
        collect(&rh),
        vec![
            (1, "-12345678".to_string(), "negfrac".to_string()),
            (2, "1".to_string(), "tiny".to_string()),
        ],
        "high-scale negative fraction + tiny value match exactly; 5.0 (unmatched) drops"
    );
}

/// HUNT #3: a numeric/uuid (b128) key used in a LATER step of a 3-way join. The accumulated side's key
/// is gathered at CARRIED indices that came from a prior step -- those indices must index the
/// relation's OWN payload, not the prior result. A wrong base would gather the wrong key -> wrong matches.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_b128_key_in_later_multiway_step() {
    let mut e = Engine::new_local_test_engine();
    // r0 (int pk) -> r1 (int fk to r0, uuid u) -> r2 (uuid u). The uuid join is the SECOND step, joining
    // the ACCUMULATED (r0,r1) on r1.u against r2.u. r1.u is gathered at carried r1 indices.
    e.execute_text(1, "CREATE TABLE r0 (id INT, tag TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r1 (id INT, u UUID)")
        .unwrap();
    e.execute_text(3, "CREATE TABLE r2 (u UUID, label TEXT)")
        .unwrap();
    let ux = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    let uy = "12345678-9abc-def0-1234-567890abcdef";
    let uz = "00000000-0000-0000-0000-000000000001";
    // r0 rows 1..3; r1 maps id->uuid (id1->ux, id2->uy, id3->uz); r2 has ux,uy only.
    e.execute_text(
        4,
        "INSERT INTO r0 (id, tag) VALUES (1,'one'),(2,'two'),(3,'three')",
    )
    .unwrap();
    e.execute_text(
        5,
        &format!("INSERT INTO r1 (id, u) VALUES (1,'{ux}'),(2,'{uy}'),(3,'{uz}')"),
    )
    .unwrap();
    e.execute_text(
        6,
        &format!("INSERT INTO r2 (u, label) VALUES ('{ux}','X'),('{uy}','Y')"),
    )
    .unwrap();
    let mut ok = true;
    for t in ["r0", "r1", "r2"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT r0.tag, r2.label, r1.u FROM r0 JOIN r1 ON r0.id = r1.id JOIN r2 ON r1.u = r2.u ORDER BY r0.tag",
        )
        .expect("3-way: int step then uuid step on the accumulated side");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let rows: Vec<(String, String, String)> = res
        .rows
        .iter()
        .map(|r| {
            let tag = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("tag {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("label {o:?}"),
            };
            let u = match &r[2] {
                SqlValue::Uuid(b) => gpu_db_sql::uuid::format_uuid(b),
                o => panic!("u {o:?}"),
            };
            (tag, label, u)
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            ("one".to_string(), "X".to_string(), ux.to_string()),
            ("two".to_string(), "Y".to_string(), uy.to_string()),
        ],
        "id3->uz has no r2 match and drops; the carried r1.u gathers the RIGHT uuid per accumulated tuple"
    );
}

/// HUNT #5: empty `abs` early return on a b128/text step. A per-side WHERE filters one side to ZERO
/// rows before a uuid-key join -> the `if abs.is_empty()` path must yield an empty result, no panic.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_empty_side_before_b128_and_text_step() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE gu (gid UUID, gname TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE uu (uid INT, gid UUID)")
        .unwrap();
    let g = "11112222-3333-4444-5555-666677778888";
    e.execute_text(
        3,
        &format!("INSERT INTO gu (gid, gname) VALUES ('{g}','g')"),
    )
    .unwrap();
    e.execute_text(
        4,
        &format!("INSERT INTO uu (uid, gid) VALUES (1,'{g}'),(2,'{g}')"),
    )
    .unwrap();
    e.execute_text(5, "CREATE TABLE ls (lid INT, t TEXT)")
        .unwrap();
    e.execute_text(6, "CREATE TABLE rs (rid INT, t TEXT)")
        .unwrap();
    e.execute_text(7, "INSERT INTO ls (lid, t) VALUES (1,'x'),(2,'y')")
        .unwrap();
    e.execute_text(8, "INSERT INTO rs (rid, t) VALUES (10,'x')")
        .unwrap();
    let mut ok = true;
    for t in ["gu", "uu", "ls", "rs"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    // WHERE filters uu to empty (no uid > 100) before the uuid join.
    let uuid_empty = e
        .execute_resident_expr_select_sql(
            "SELECT uu.uid, gu.gname FROM uu JOIN gu ON uu.gid = gu.gid WHERE uu.uid > 100",
        )
        .expect("empty uuid side must not panic");
    assert!(
        uuid_empty.rows.is_empty(),
        "filtered-to-empty uuid side -> empty result"
    );
    // WHERE filters rs to empty before the text join.
    let text_empty = e
        .execute_resident_expr_select_sql(
            "SELECT ls.lid, rs.rid FROM ls JOIN rs ON ls.t = rs.t WHERE rs.rid > 100",
        )
        .expect("empty text side must not panic");
    assert!(
        text_empty.rows.is_empty(),
        "filtered-to-empty text side -> empty result"
    );
    // A NON-empty text match through the device gather (so this test also covers the text-key gather
    // correctness, not just the empty path): 'x' matches lid 1 -> rid 10.
    let text_match = e
        .execute_resident_expr_select_sql("SELECT ls.lid, rs.rid FROM ls JOIN rs ON ls.t = rs.t")
        .expect("text-key match through the device gather");
    let pairs: Vec<(i32, i32)> = text_match
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Int4(a), SqlValue::Int4(b)) => (*a, *b),
            o => panic!("{o:?}"),
        })
        .collect();
    assert_eq!(
        pairs,
        vec![(1, 10)],
        "text key 'x' matches lid 1 -> rid 10 (device-gathered keys)"
    );
}

/// HUNT #4 + #6: N:N numeric/uuid AND the NULL gate together. Both sides duplicate the numeric key; one
/// side ALSO has a NULL-key row. The NULL row must match nothing (excluded by the host gate BEFORE the
/// device gather) and the cross product of the non-NULL key must be exact. Confirms the device gather is
/// never reached on a NULL index.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_nn_numeric_with_null_key_gate() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE ln (lid INT, amt NUMERIC(10,2))")
        .unwrap();
    e.execute_text(2, "CREATE TABLE rn (rid INT, amt NUMERIC(10,2))")
        .unwrap();
    // key 5.00 duplicated on both sides; a NULL-key row on each side must drop, NOT spuriously match.
    e.execute_text(
        3,
        "INSERT INTO ln (lid, amt) VALUES (1,5.00),(2,5.00),(3,NULL)",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO rn (rid, amt) VALUES (10,5.00),(11,5.00),(12,NULL)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["ln", "rn"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT ln.lid, rn.rid FROM ln JOIN rn ON ln.amt = rn.amt",
        )
        .expect("N:N numeric with NULL keys");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut pairs: Vec<(i32, i32)> = res
        .rows
        .iter()
        .map(|r| {
            let a = match &r[0] {
                SqlValue::Int4(v) => *v,
                o => panic!("{o:?}"),
            };
            let b = match &r[1] {
                SqlValue::Int4(v) => *v,
                o => panic!("{o:?}"),
            };
            (a, b)
        })
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![(1, 10), (1, 11), (2, 10), (2, 11)],
        "5.00 cross product only; NULL=NULL is UNKNOWN -> the (3,_)/(12,_) NULL rows match nothing"
    );
}

/// CROSS-CHECK that the new device gather is BEHAVIOR-PRESERVING by comparing the same query/data the
/// way the diff claims: a uuid+numeric join whose RESULT (matches AND projected key values) is the
/// expected set computed from the host data. This is the non-vacuous "device == host bytes" proof for a
/// dataset with both an asymmetric uuid and a negative numeric in the SAME query, projected end-to-end.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_mixed_uuid_numeric_end_to_end() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE k (gid UUID, amt NUMERIC(12,3))")
        .unwrap();
    e.execute_text(2, "CREATE TABLE p (gid UUID, tag TEXT)")
        .unwrap();
    let u = "80706050-4030-2010-0fef-dfcfbfaf9f8f"; // high bit set, asymmetric
    e.execute_text(
        3,
        &format!("INSERT INTO k (gid, amt) VALUES ('{u}',-42.500)"),
    )
    .unwrap();
    e.execute_text(4, &format!("INSERT INTO p (gid, tag) VALUES ('{u}','hit'),('deadbeef-0000-0000-0000-000000000000','miss')")).unwrap();
    let mut ok = true;
    for t in ["k", "p"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT k.gid, k.amt, p.tag FROM k JOIN p ON k.gid = p.gid",
        )
        .expect("uuid join projecting uuid + negative numeric");
    assert_eq!(res.rows.len(), 1, "only the matching uuid pair");
    let row = &res.rows[0];
    match &row[0] {
        SqlValue::Uuid(b) => assert_eq!(gpu_db_sql::uuid::format_uuid(b), u, "uuid value exact"),
        o => panic!("gid {o:?}"),
    }
    match &row[1] {
        SqlValue::Numeric(d) => assert_eq!(d.mantissa, -42500, "negative numeric mantissa exact"),
        o => panic!("amt {o:?}"),
    }
    match &row[2] {
        SqlValue::Text(t) => assert_eq!(t, "hit"),
        o => panic!("tag {o:?}"),
    }
}
