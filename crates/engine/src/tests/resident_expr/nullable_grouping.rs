use crate::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlValue};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_skips_null_values_and_groups_null_keys() {
    // M3 (doc 21): GROUP BY full 3VL on the GPU — a NULL aggregate VALUE is SKIPPED from SUM/AVG/MIN/MAX
    // (the single-level kernel's value-skip) while COUNT(*) still counts the row (a dedicated total-count
    // pass), an all-NULL group's aggregate is SQL NULL, AND a NULL group KEY forms its OWN group (the
    // kernel's reserved NULL-key slot) rendered SqlValue::Null. A NULL-free nullable column is unchanged.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT, h INT)")
        .unwrap();
    // g has a NULL (row 2); v has a NULL (rows 4 and 6); h has none.
    // h=7 -> v{10, 20}; h=8 -> v{30, NULL}; h=9 -> v{NULL} (an all-NULL group).
    e.execute_text(
        2,
        "INSERT INTO t (g,v,h) VALUES (1,10,7),(NULL,20,7),(2,30,8),(1,NULL,8),(2,NULL,9)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // GROUP BY a NULL-bearing KEY: the NULL keys form their OWN group (rendered SqlValue::Null), which
    // sorts first. g: 1,NULL,2,1,2 -> g=1 count 2, g=2 count 2, g=NULL count 1.
    let gc = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("GROUP BY a nullable key forms a NULL group");
    assert_eq!(
        gc.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(1)],
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(2)],
        ],
        "NULL keys form their own group (COUNT(*) counts them); it sorts first"
    );

    // SUM over a nullable VALUE: NULLs skipped. h=7 -> 30, h=8 -> 30 (NULL skipped), h=9 -> NULL (all-NULL).
    let s = e
        .execute_resident_expr_select_sql("SELECT h, SUM(v) FROM t GROUP BY h")
        .expect("SUM over a nullable value runs");
    assert_eq!(
        s.rows,
        vec![
            vec![SqlValue::Int4(7), SqlValue::Int8(30)],
            vec![SqlValue::Int4(8), SqlValue::Int8(30)],
            vec![SqlValue::Int4(9), SqlValue::Null],
        ],
        "SUM skips NULL values; an all-NULL group is NULL"
    );
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));

    // COUNT(*) + SUM together: COUNT(*) counts EVERY row (incl. NULL-v), SUM skips NULLs. h=8 -> (2, 30);
    // h=9 -> (1, NULL). Exercises the dedicated total-count pass alongside the value-skip pass.
    let cs = e
        .execute_resident_expr_select_sql("SELECT h, COUNT(*), SUM(v) FROM t GROUP BY h")
        .expect("COUNT(*) + SUM over a nullable value");
    assert_eq!(
        cs.rows,
        vec![
            vec![SqlValue::Int4(7), SqlValue::Int8(2), SqlValue::Int8(30)],
            vec![SqlValue::Int4(8), SqlValue::Int8(2), SqlValue::Int8(30)],
            vec![SqlValue::Int4(9), SqlValue::Int8(1), SqlValue::Null],
        ],
        "COUNT(*) counts NULL-valued rows; SUM skips them"
    );

    // MIN / MAX / AVG over the nullable value: NULLs skipped; all-NULL group -> NULL.
    let mn = e
        .execute_resident_expr_select_sql("SELECT h, MIN(v) FROM t GROUP BY h")
        .expect("MIN over a nullable value");
    assert_eq!(
        mn.rows,
        vec![
            vec![SqlValue::Int4(7), SqlValue::Int4(10)],
            vec![SqlValue::Int4(8), SqlValue::Int4(30)],
            vec![SqlValue::Int4(9), SqlValue::Null],
        ],
        "MIN skips NULLs (not the 0 placeholder); all-NULL group is NULL"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_null_key_group_with_null_values() {
    // M3 (doc 21): the NULL-KEY group + the value-skip + the total-count pass interact correctly. Every
    // NULL-key row groups together (distinct from real key 0); COUNT(*) counts ALL of them (incl. a NULL-
    // value one); SUM skips the NULL value AMONG the null-key rows. k: 1,NULL,2,NULL,1,NULL.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE tk (k INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO tk (k,v) VALUES (1,10),(NULL,20),(2,30),(NULL,40),(1,50),(NULL,NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tk").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k=1 -> v{10,50}: count 2, sum 60. k=2 -> v{30}: count 1, sum 30.
    // k=NULL -> rows (NULL,20),(NULL,40),(NULL,NULL): count 3 (ALL), sum 60 (the NULL value skipped).
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tk GROUP BY k")
        .expect("GROUP BY a nullable key with nullable values");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(3), SqlValue::Int8(60)],
            vec![SqlValue::Int4(1), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "NULL keys group together; COUNT(*) counts all (incl. the NULL-value row); SUM skips the NULL"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_null_key_group_with_all_null_values() {
    // M3 (doc 21) regression: a NULL-key group whose aggregate values are ALL NULL. The value pass's
    // reserved null slot then has count 0 — but the group MUST still appear (COUNT(*) counts the rows;
    // SUM is NULL). The reserved slot is emitted on its CLAIMED MARKER (slot_keys != EMPTY), not count,
    // so it appears CONSISTENTLY in every pass and the by-index merge stays aligned (else: panic / the
    // null row silently vanishes).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE tp (k INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO tp (k,v) VALUES (1,10),(NULL,NULL),(2,30),(NULL,NULL),(1,50),(NULL,NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tp").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k=NULL -> 3 rows, all v NULL: COUNT(*) 3, SUM NULL. The multi-pass query is the panic case.
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tp GROUP BY k")
        .expect("COUNT(*)+SUM with an all-NULL-value NULL-key group");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(3), SqlValue::Null],
            vec![SqlValue::Int4(1), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "the all-NULL-value NULL-key group still appears (COUNT(*)=3, SUM=NULL), aligned across passes"
    );
    // SUM only (no COUNT*): the null group must NOT silently vanish.
    let s = e
        .execute_resident_expr_select_sql("SELECT k, SUM(v) FROM tp GROUP BY k")
        .expect("SUM with an all-NULL-value NULL-key group");
    assert_eq!(
        s.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Null],
            vec![SqlValue::Int4(1), SqlValue::Int8(60)],
            vec![SqlValue::Int4(2), SqlValue::Int8(30)],
        ],
        "the all-NULL-value NULL-key group still appears with SUM NULL (not dropped)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_skips_null_int8_values() {
    // M3 (doc 21): the value-skip is type-agnostic (it gates the accumulate before the per-type sum), so
    // a nullable BIGINT value also skips NULLs on the GPU (the i64 value / i128-carry sum path). MIN(v)
    // returns int8; an all-NULL group is NULL.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t8 (g INT, v BIGINT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t8 (g,v) VALUES (1,100),(1,NULL),(2,9999999999),(3,NULL)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t8").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // g=1 -> {100, NULL} -> MIN 100; g=2 -> {9999999999} -> MIN 9999999999; g=3 -> {NULL} -> NULL.
    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t8 GROUP BY g")
        .expect("MIN over a nullable bigint");
    assert_eq!(
        mn.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(100)],
            vec![SqlValue::Int4(2), SqlValue::Int8(9999999999)],
            vec![SqlValue::Int4(3), SqlValue::Null],
        ],
        "MIN(bigint) skips NULLs; all-NULL group is NULL"
    );
    // COUNT(*) counts every row (incl. the NULL-v rows).
    let c = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t8 GROUP BY g")
        .expect("COUNT(*) over a nullable bigint table");
    assert_eq!(
        c.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(1)],
            vec![SqlValue::Int4(3), SqlValue::Int8(1)],
        ],
        "COUNT(*) counts NULL-valued rows"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_key_with_count_distinct_clean_errors() {
    // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE group key is a clean-error follow-up. The
    // COUNT(DISTINCT) sub-passes don't route the NULL key to the reserved slot, so they'd merge NULL-key
    // rows into the placeholder group -> fewer groups than the reference pass -> by-index merge panic.
    // Reject cleanly rather than panic / mis-answer. (Pre-existing for int keys; this guard fixes that
    // too.) A NON-nullable key with COUNT(DISTINCT) is unaffected.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE tcd (g INT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tcd (g,v) VALUES (1,10),(NULL,20),(0,30),(NULL,20),(1,10),(0,40)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tcd").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM tcd GROUP BY g")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("nullable key with COUNT(DISTINCT)") || err.contains("not yet supported"),
        "COUNT(DISTINCT) over a nullable key must clean-error (not panic), got: {err}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_text_key_forms_null_group() {
    // M3 (doc 21): GROUP BY a nullable TEXT key — a NULL key forms its OWN group (rendered SqlValue::Null,
    // sorts first), distinct from real keys, via the kernel's hoisted NULL-key check routing to the
    // reserved slot BEFORE the text claim. A NULL text key is NOT folded into the empty-string group.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE tgt (k TEXT, v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tgt (k,v) VALUES ('a',10),(NULL,20),('b',30),(NULL,40),('a',50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tgt").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k: 'a'->{10,50}=2/60, 'b'->{30}=1/30, NULL->{20,40}=2/60. NULL sorts first.
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tgt GROUP BY k")
        .expect("GROUP BY a nullable text key runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Text("a".to_string()), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![SqlValue::Text("b".to_string()), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "NULL text key forms its own group (sorts first), not folded into a real/empty-string group"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_numeric_key_forms_null_group() {
    // M3 (doc 21): GROUP BY a nullable NUMERIC key — a NULL key forms its own group (the i128 claim path
    // now sees only non-NULL keys; NULLs route to the reserved slot). A NULL is NOT folded into 0.00.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE tgn (k NUMERIC(10,2), v INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO tgn (k,v) VALUES (1.50,10),(NULL,20),(2.50,30),(NULL,40),(1.50,50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tgn").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k: 1.50->2/60, 2.50->1/30, NULL->2/60. NULL sorts first.
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tgn GROUP BY k")
        .expect("GROUP BY a nullable numeric key runs on the GPU");
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![
                SqlValue::Numeric(Decimal128::new(150, 2)),
                SqlValue::Int8(2),
                SqlValue::Int8(60)
            ],
            vec![
                SqlValue::Numeric(Decimal128::new(250, 2)),
                SqlValue::Int8(1),
                SqlValue::Int8(30)
            ],
        ],
        "NULL numeric key forms its own group (sorts first), not folded into 0.00"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_uuid_key_forms_null_group() {
    // M3 (doc 21): GROUP BY a nullable UUID key — a NULL key forms its own group (the i128/b128 claim sees
    // only non-NULL keys). A NULL is NOT folded into the all-zero uuid.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE tgu (k UUID, v INT)")
        .unwrap();
    let uuid_for = |i: i64| format!("00000000-0000-0000-0000-0000000000{i:02x}");
    e.execute_text(
        2,
        &format!(
            "INSERT INTO tgu (k,v) VALUES ('{}',10),(NULL,20),('{}',30),(NULL,40),('{}',50)",
            uuid_for(5),
            uuid_for(15),
            uuid_for(5),
        ),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tgu").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // k: uuid5->2/60, uuid15->1/30, NULL->2/60. NULL sorts first, then uuid5, uuid15 (byte order).
    let r = e
        .execute_resident_expr_select_sql("SELECT k, COUNT(*), SUM(v) FROM tgu GROUP BY k")
        .expect("GROUP BY a nullable uuid key runs on the GPU");
    let uuid =
        |i: i64| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(&uuid_for(i)).expect("valid uuid"));
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Null, SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![uuid(5), SqlValue::Int8(2), SqlValue::Int8(60)],
            vec![uuid(15), SqlValue::Int8(1), SqlValue::Int8(30)],
        ],
        "NULL uuid key forms its own group (sorts first), not folded into the all-zero uuid"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_numeric_value_skips_nulls() {
    // M3 (doc 21): GROUP BY over a nullable NUMERIC value now runs full 3VL on the GPU. The numeric
    // MIN/MAX is a TWO-PASS kernel: pass 1 finalizes the i128 HIGH limb + records each NON-NULL row's
    // claimed slot into a pooled row_slots scratch; pass 2 (gpu_db_group_by_numeric_minmax_lo) resolves
    // the LOW limb. BOTH passes now read the value validity bitmap and skip NULL rows — so a NULL row's
    // STALE pooled row_slots slot is never folded (the prior 700/OOB hazard). SUM/MIN/MAX skip NULLs;
    // COUNT(*) counts every row; an all-NULL group's aggregate is SQL NULL.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE tn (g INT, v NUMERIC(10,2), w NUMERIC(10,2))",
    )
    .unwrap();
    // g=1 -> v{10.50, 30.25, NULL}: real MIN/MAX distinction (10.50 vs 30.25) + a NULL skip, SUM 40.75.
    // g=2 -> v{5.00, NULL}: one non-NULL + a NULL skip. g=3 -> v{NULL}: an all-NULL group -> NULL. w: none.
    e.execute_text(
        2,
        "INSERT INTO tn (g,v,w) VALUES \
         (1,10.50,1.00),(1,30.25,2.00),(1,NULL,3.00),(2,5.00,4.00),(2,NULL,5.00),(3,NULL,6.00)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("tn").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Dirty the pooled row_slots buffer FIRST with a non-NULL numeric GROUP BY of a DIFFERENT shape, so a
    // stale-slot read in pass 2 (the bug this slice fixes) would surface as a wrong answer below.
    let _ = e
        .execute_resident_expr_select_sql("SELECT g, MIN(w), MAX(w) FROM tn GROUP BY g")
        .expect("non-nullable numeric MIN/MAX runs (dirties row_slots)");

    // SUM(v): NULLs skipped. g=1 -> 40.75 (10.50+30.25), g=2 -> 5.00, g=3 -> NULL (all-NULL).
    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM tn GROUP BY g")
        .expect("SUM over a nullable numeric value runs");
    assert_eq!(
        s.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Numeric(Decimal128::new(4075, 2))
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Numeric(Decimal128::new(500, 2))
            ],
            vec![SqlValue::Int4(3), SqlValue::Null],
        ],
        "SUM(numeric) skips NULLs; an all-NULL group is NULL"
    );
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));

    // MIN(v) + MAX(v): the TWO-PASS path. g=1 -> MIN 10.50 / MAX 30.25 (NULL skipped, not folded as 0);
    // g=2 -> 5.00 / 5.00; g=3 -> NULL / NULL.
    let mm = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v), MAX(v) FROM tn GROUP BY g")
        .expect("MIN/MAX over a nullable numeric value runs");
    assert_eq!(
        mm.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Numeric(Decimal128::new(1050, 2)),
                SqlValue::Numeric(Decimal128::new(3025, 2)),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Numeric(Decimal128::new(500, 2)),
                SqlValue::Numeric(Decimal128::new(500, 2)),
            ],
            vec![SqlValue::Int4(3), SqlValue::Null, SqlValue::Null],
        ],
        "MIN/MAX(numeric) skip NULLs (never the 0 placeholder); all-NULL group is NULL"
    );

    // COUNT(*) + MIN(v): COUNT(*) counts EVERY row (incl. NULL-v), MIN skips NULLs. g=1 -> (3, 10.50);
    // g=3 -> (1, NULL). Exercises the total-count pass alongside the two-pass numeric MIN.
    let cm = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*), MIN(v) FROM tn GROUP BY g")
        .expect("COUNT(*) + MIN over a nullable numeric value");
    assert_eq!(
        cm.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int8(3),
                SqlValue::Numeric(Decimal128::new(1050, 2))
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int8(2),
                SqlValue::Numeric(Decimal128::new(500, 2))
            ],
            vec![SqlValue::Int4(3), SqlValue::Int8(1), SqlValue::Null],
        ],
        "COUNT(*) counts NULL-valued numeric rows; MIN skips them"
    );

    // A NON-nullable numeric value (w has no NULLs) is unaffected — no validity bitmap, byte-identical path.
    let ok = e
        .execute_resident_expr_select_sql("SELECT g, SUM(w) FROM tn GROUP BY g")
        .expect("SUM over a NON-nullable numeric value still runs");
    assert_eq!(
        ok.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Numeric(Decimal128::new(600, 2))
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Numeric(Decimal128::new(900, 2))
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Numeric(Decimal128::new(600, 2))
            ],
        ],
        "non-nullable numeric SUM is unchanged"
    );
}
