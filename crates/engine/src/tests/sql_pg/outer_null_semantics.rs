use crate::Engine;
use crate::{RelationalSelectResult, RowBlock};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_left_outer_join_null_pads_unmatched_left_rows() {
    // 2-relation LEFT OUTER join (M3 -- doc 21): every LEFT row appears; an unmatched left row -- the
    // CHILDLESS parent 3, AND the NULL-key left row 'nokey' (which matches nothing, 3VL) -- is kept with
    // the right relation's columns NULL-padded.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE lp (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE lc (pid INT, label TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO lp (id, name) VALUES (1,'a'),(2,'b'),(3,'c'),(NULL,'nokey')",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO lc (pid, label) VALUES (1,'x'),(1,'y'),(2,'z')",
    )
    .unwrap();
    let ps = e.populate_relational_residency_snapshot("lp").unwrap();
    let cs = e.populate_relational_residency_snapshot("lc").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM lp LEFT JOIN lc ON lp.id = lc.pid",
        )
        .expect("left outer join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, Option<String>)> = res
        .rows
        .iter()
        .map(|r| {
            let name = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("name: {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => Some(t.clone()),
                SqlValue::Null => None,
                o => panic!("label: {o:?}"),
            };
            (name, label)
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("a".to_string(), Some("x".to_string())),
            ("a".to_string(), Some("y".to_string())),
            ("b".to_string(), Some("z".to_string())),
            ("c".to_string(), None), // childless parent 3 -> NULL-padded right columns
            ("nokey".to_string(), None), // NULL-key left row matches nothing -> NULL-padded
        ],
        "LEFT JOIN keeps every left row; unmatched (incl. NULL-key) rows are NULL-padded"
    );

    // TEXT-key LEFT join: the NULL pad maps over the GPU text hash join too (a different build/probe
    // orientation than int). 'z' has no match -> NULL-padded.
    e.execute_text(5, "CREATE TABLE tp (k TEXT, name TEXT)")
        .unwrap();
    e.execute_text(6, "CREATE TABLE tc (k TEXT, label TEXT)")
        .unwrap();
    e.execute_text(
        7,
        "INSERT INTO tp (k, name) VALUES ('a','pa'),('b','pb'),('z','pz')",
    )
    .unwrap();
    e.execute_text(8, "INSERT INTO tc (k, label) VALUES ('a','ca'),('b','cb')")
        .unwrap();
    let tps = e.populate_relational_residency_snapshot("tp").unwrap();
    let tcs = e.populate_relational_residency_snapshot("tc").unwrap();
    if tps.device_memory_proof.is_none() || tcs.device_memory_proof.is_none() {
        return;
    }
    let tres = e
        .execute_resident_expr_select_sql("SELECT name, label FROM tp LEFT JOIN tc ON tp.k = tc.k")
        .expect("text-key left join");
    let mut tgot: Vec<(String, Option<String>)> = tres
        .rows
        .iter()
        .map(|r| {
            let name = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("name: {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => Some(t.clone()),
                SqlValue::Null => None,
                o => panic!("label: {o:?}"),
            };
            (name, label)
        })
        .collect();
    tgot.sort();
    assert_eq!(
        tgot,
        vec![
            ("pa".to_string(), Some("ca".to_string())),
            ("pb".to_string(), Some("cb".to_string())),
            ("pz".to_string(), None), // 'z' has no match -> NULL-padded
        ],
        "text-key LEFT join NULL-pads the unmatched left row"
    );

    // A WHERE on a LEFT join filters the JOINED RESULT (PG semantics), not a per-side pushdown. The
    // predicate runs on the GPU; only lc row (1,'x') passes `lc.label = 'x'`, so every other tuple --
    // including the NULL-padded (c) / (nokey) rows whose lc.label is NULL (UNKNOWN) -- is dropped.
    let wres = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM lp LEFT JOIN lc ON lp.id = lc.pid WHERE lc.label = 'x'",
        )
        .expect("LEFT JOIN with WHERE on the inner side filters the result");
    let mut wgot: Vec<(String, Option<String>)> = wres
        .rows
        .iter()
        .map(|r| {
            let name = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("name: {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => Some(t.clone()),
                SqlValue::Null => None,
                o => panic!("label: {o:?}"),
            };
            (name, label)
        })
        .collect();
    wgot.sort();
    assert_eq!(
        wgot,
        vec![("a".to_string(), Some("x".to_string()))],
        "WHERE on the inner side of a LEFT join drops the non-matching + NULL-padded tuples"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_right_and_full_outer_join_null_pad_the_correct_side() {
    // RIGHT keeps every RIGHT (new) row (unmatched -> the left columns NULL-padded); FULL keeps both
    // sides' unmatched rows (M3 -- doc 21). rl 2 'b' is left-only; rr 3 'z' is right-only.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE rl (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE rr (rid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO rl (id, name) VALUES (1,'a'),(2,'b')")
        .unwrap();
    e.execute_text(4, "INSERT INTO rr (rid, label) VALUES (1,'x'),(3,'z')")
        .unwrap();
    let ls = e.populate_relational_residency_snapshot("rl").unwrap();
    let rs = e.populate_relational_residency_snapshot("rr").unwrap();
    if ls.device_memory_proof.is_none() || rs.device_memory_proof.is_none() {
        return;
    }
    let opt_pairs = |res: &RelationalSelectResult| -> Vec<(Option<String>, Option<String>)> {
        let opt = |c: &SqlValue| match c {
            SqlValue::Text(t) => Some(t.clone()),
            SqlValue::Null => None,
            o => panic!("expected text/null, got {o:?}"),
        };
        let mut v: Vec<(Option<String>, Option<String>)> =
            res.rows.iter().map(|r| (opt(&r[0]), opt(&r[1]))).collect();
        v.sort();
        v
    };

    // RIGHT: every rr row appears; (3,'z') is right-only -> name NULL. rl's left-only 'b' is DROPPED.
    let right = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM rl RIGHT JOIN rr ON rl.id = rr.rid",
        )
        .expect("right outer join");
    assert_eq!(right.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        opt_pairs(&right),
        vec![
            (None, Some("z".to_string())), // right-only -> left columns NULL
            (Some("a".to_string()), Some("x".to_string())),
        ],
        "RIGHT JOIN keeps every right row; left-only 'b' dropped"
    );

    // FULL: the match + BOTH unmatched sides.
    let full = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM rl FULL JOIN rr ON rl.id = rr.rid",
        )
        .expect("full outer join");
    assert_eq!(
        opt_pairs(&full),
        vec![
            (None, Some("z".to_string())),                  // right-only
            (Some("a".to_string()), Some("x".to_string())), // match
            (Some("b".to_string()), None),                  // left-only
        ],
        "FULL JOIN keeps the match + both unmatched sides"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nway_outer_join_null_pads_through_the_pipeline() {
    // N-way (multi-step) OUTER join (M3 -- doc 21): a prior step's NULL pad carries a JOIN_NULL_ROW
    // sentinel; the next step reads it as a NULL key (matches nothing; a LEFT step re-pads it) instead of
    // treating the sentinel as a device row index. Two cases: (1) the carried NULL is in a relation NOT used as the next key (the
    // tuple still participates via a non-padded key); (2) the carried NULL IS the next key (the tuple is
    // re-padded). All on the GPU join pipeline.
    let mut e = Engine::new_local_test_engine();
    // Case 1: A LEFT JOIN B (on A.id) LEFT JOIN C (on A.id). A=3 has no B (B NULL-padded) but matches C=3.
    e.execute_text(1, "CREATE TABLE a3 (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE b3 (aid INT, bl TEXT)")
        .unwrap();
    e.execute_text(3, "CREATE TABLE c3 (aid INT, cl TEXT)")
        .unwrap();
    e.execute_text(4, "INSERT INTO a3 (id,name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(5, "INSERT INTO b3 (aid,bl) VALUES (1,'b1'),(2,'b2')")
        .unwrap();
    e.execute_text(6, "INSERT INTO c3 (aid,cl) VALUES (1,'c1'),(3,'c3')")
        .unwrap();
    for t in ["a3", "b3", "c3"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let opt = |v: &SqlValue| match v {
        SqlValue::Text(t) => Some(t.clone()),
        SqlValue::Null => None,
        o => panic!("unexpected {o:?}"),
    };
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, bl, cl FROM a3 LEFT JOIN b3 ON a3.id = b3.aid LEFT JOIN c3 ON a3.id = c3.aid",
        )
        .expect("N-way LEFT-LEFT join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, Option<String>, Option<String>)> = res
        .rows
        .iter()
        .map(|r| (opt(&r[0]).unwrap(), opt(&r[1]), opt(&r[2])))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("a".into(), Some("b1".into()), Some("c1".into())),
            ("b".into(), Some("b2".into()), None), // B match, no C
            ("c".into(), None, Some("c3".into())), // B NULL-padded (step 1), still matches C on A.id
        ],
        "N-way LEFT-LEFT: a carried NULL in a non-key relation still joins on a non-padded key"
    );

    // Case 2: A LEFT JOIN B (on A.id) LEFT JOIN C (on B.cid). A=3's B is NULL-padded, so B.cid is a carried
    // NULL key at step 2 -> A=3 matches no C -> C re-padded (the sentinel must never become a device row index).
    e.execute_text(7, "CREATE TABLE a4 (id INT, name TEXT)")
        .unwrap();
    e.execute_text(8, "CREATE TABLE b4 (aid INT, cid INT, bl TEXT)")
        .unwrap();
    e.execute_text(9, "CREATE TABLE c4 (id INT, cl TEXT)")
        .unwrap();
    e.execute_text(
        10,
        "INSERT INTO a4 (id,name) VALUES (1,'a'),(2,'b'),(3,'c')",
    )
    .unwrap();
    e.execute_text(
        11,
        "INSERT INTO b4 (aid,cid,bl) VALUES (1,100,'b1'),(2,200,'b2')",
    )
    .unwrap();
    e.execute_text(
        12,
        "INSERT INTO c4 (id,cl) VALUES (100,'c1'),(200,'c2'),(300,'c3')",
    )
    .unwrap();
    for t in ["a4", "b4", "c4"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, bl, cl FROM a4 LEFT JOIN b4 ON a4.id = b4.aid LEFT JOIN c4 ON b4.cid = c4.id",
        )
        .expect("N-way LEFT-LEFT join keyed on a previously-padded relation");
    let mut got: Vec<(String, Option<String>, Option<String>)> = res
        .rows
        .iter()
        .map(|r| (opt(&r[0]).unwrap(), opt(&r[1]), opt(&r[2])))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("a".into(), Some("b1".into()), Some("c1".into())),
            ("b".into(), Some("b2".into()), Some("c2".into())),
            ("c".into(), None, None), // B NULL-padded -> B.cid NULL key at step 2 -> C re-padded
        ],
        "N-way: a carried NULL used AS the next join key matches nothing and is re-padded (no OOB)"
    );

    // Case 3: a RIGHT step over a MULTI-relation accumulated side -- A JOIN B (inner) RIGHT JOIN C. An
    // unmatched C row pads the ENTIRE accumulated side (BOTH A and B), exercising the RIGHT pad's
    // `take(new_rel)` over >1 accumulated relations.
    e.execute_text(13, "CREATE TABLE a5 (id INT, an TEXT)")
        .unwrap();
    e.execute_text(14, "CREATE TABLE b5 (aid INT, bn TEXT)")
        .unwrap();
    e.execute_text(15, "CREATE TABLE c5 (cx INT, cn TEXT)")
        .unwrap();
    e.execute_text(16, "INSERT INTO a5 (id,an) VALUES (1,'a1'),(2,'a2')")
        .unwrap();
    e.execute_text(17, "INSERT INTO b5 (aid,bn) VALUES (1,'b1'),(2,'b2')")
        .unwrap();
    e.execute_text(18, "INSERT INTO c5 (cx,cn) VALUES (1,'c1'),(3,'c3')")
        .unwrap();
    for t in ["a5", "b5", "c5"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT an, bn, cn FROM a5 JOIN b5 ON a5.id = b5.aid RIGHT JOIN c5 ON b5.aid = c5.cx",
        )
        .expect("RIGHT JOIN over a multi-relation accumulated side");
    let mut got: Vec<(Option<String>, Option<String>, Option<String>)> = res
        .rows
        .iter()
        .map(|r| (opt(&r[0]), opt(&r[1]), opt(&r[2])))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            (None, None, Some("c3".into())), // right-only C row pads BOTH accumulated relations (A and B)
            (Some("a1".into()), Some("b1".into()), Some("c1".into())),
        ],
        "RIGHT step over a 2-relation accumulated side pads the WHOLE accumulated tuple (A and B)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_explicit_nulls_first_last_honored_on_device() {
    // M3 (doc 21): explicit NULLS FIRST / NULLS LAST OVERRIDES PG's default placement, honored ON-DEVICE
    // in the GPU sort comparator (the per-key nulls_first bitmask), DECOUPLED from ASC/DESC. Without an
    // override ASC = NULLS LAST and DESC = NULLS FIRST (the default, covered elsewhere).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES (3,100),(NULL,100),(1,100),(NULL,100),(2,100)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("t")
        .unwrap()
        .device_memory_proof
        .is_none()
    {
        return;
    }
    let ints = |rows: &RowBlock| -> Vec<Option<i32>> {
        rows.iter()
            .map(|row| match row[0] {
                SqlValue::Int4(v) => Some(v),
                SqlValue::Null => None,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect()
    };
    // ASC NULLS FIRST: NULLs first, then ascending (overrides the ASC default of NULLS LAST).
    let r = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE b >= 0 ORDER BY a ASC NULLS FIRST")
        .unwrap();
    assert_eq!(
        ints(&r.rows),
        vec![None, None, Some(1), Some(2), Some(3)],
        "ASC NULLS FIRST"
    );
    // DESC NULLS LAST: descending, then NULLs last (overrides the DESC default of NULLS FIRST).
    let r = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE b >= 0 ORDER BY a DESC NULLS LAST")
        .unwrap();
    assert_eq!(
        ints(&r.rows),
        vec![Some(3), Some(2), Some(1), None, None],
        "DESC NULLS LAST"
    );
    // Sanity: the default is unchanged (ASC => NULLS LAST) when no override is given.
    let r = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE b >= 0 ORDER BY a")
        .unwrap();
    assert_eq!(
        ints(&r.rows),
        vec![Some(1), Some(2), Some(3), None, None],
        "ASC default = NULLS LAST"
    );
    // Explicit NULLS FIRST/LAST is now ALSO honored on the GROUP BY result path. a=[3,NULL,1,NULL,2] ->
    // groups {1,2,3,NULL}; ORDER BY a NULLS FIRST -> NULL first, then ascending.
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT a, COUNT(*) FROM t WHERE b >= 0 GROUP BY a ORDER BY a NULLS FIRST",
        )
        .expect("grouped ORDER BY a NULLS FIRST");
    assert_eq!(
        ints(&r.rows),
        vec![None, Some(1), Some(2), Some(3)],
        "explicit NULLS FIRST on a GROUP BY result places the NULL group first"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_composite_key_per_member_null_on_device() {
    // M3 (doc 21): GROUP BY a COMPOSITE key (`a, b`) with nullable members. Per-member NULL is encoded in
    // the wide key (a trailing validity word written ON-DEVICE by gpu_db_build_wide_key), so (NULL,5),
    // (1,5), (NULL,6), (NULL,NULL), (1,NULL) are all DISTINCT groups, each member rendered SqlValue::Null
    // from the representative row. A nullable composite routes to the wide-key path (the i64 pack has no
    // room for validity).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES (1,5),(NULL,5),(1,5),(NULL,6),(NULL,NULL),(1,NULL)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("t")
        .unwrap()
        .device_memory_proof
        .is_none()
    {
        return;
    }
    let groups = |rows: &RowBlock| -> Vec<(Option<i32>, Option<i32>, i64)> {
        let opt = |v: &SqlValue| match v {
            SqlValue::Int4(x) => Some(*x),
            SqlValue::Null => None,
            o => panic!("unexpected key {o:?}"),
        };
        let mut v: Vec<(Option<i32>, Option<i32>, i64)> = rows
            .iter()
            .map(|r| {
                let c = match r[2] {
                    SqlValue::Int8(x) => x,
                    SqlValue::Int4(x) => i64::from(x),
                    ref o => panic!("unexpected count {o:?}"),
                };
                (opt(&r[0]), opt(&r[1]), c)
            })
            .collect();
        v.sort();
        v
    };
    let r = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("GROUP BY a nullable composite key");
    assert_eq!(
        groups(&r.rows),
        vec![
            (None, None, 1),       // (NULL, NULL)
            (None, Some(5), 1),    // (NULL, 5)
            (None, Some(6), 1),    // (NULL, 6)
            (Some(1), None, 1),    // (1, NULL)
            (Some(1), Some(5), 2), // (1, 5) x2
        ],
        "composite NULL members form distinct groups: (NULL,5) != (1,5) != (NULL,6) != (NULL,NULL) != (1,NULL)"
    );
    // COUNT(DISTINCT) over a nullable composite key clean-errors (its dedup sub-pass builds the wide key
    // without validity, so it would merge NULL with a real value) -- a clean error, not a wrong answer.
    assert!(
        e.execute_resident_expr_select_sql("SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a, b")
            .is_err(),
        "COUNT(DISTINCT) over a nullable composite key clean-errors"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_count_distinct_over_a_nullable_value_clean_errors() {
    // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE value must EXCLUDE NULLs (PG). The SCALAR form
    // now COMPUTES this (the aggregate validity conjunct ANDs `v IS NOT NULL` into the predicate, so
    // the distinct pass sees no NULLs — the former clean-error is resolved); the GROUPED paths still
    // have no value validity in the sort-based reps pass, so they keep the clean error rather than
    // silently over-count. (A non-null COUNT(DISTINCT) is unaffected -- the count_distinct suite.)
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10),(1,NULL),(1,10),(2,20)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("t")
        .unwrap()
        .device_memory_proof
        .is_none()
    {
        return;
    }
    assert!(
        e.execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
            .is_err(),
        "grouped COUNT(DISTINCT) over a nullable value clean-errors (no silent over-count)"
    );
    let scalar = e
        .execute_resident_expr_select_sql("SELECT COUNT(DISTINCT v) FROM t")
        .unwrap();
    assert_eq!(
        scalar.rows,
        vec![vec![SqlValue::Int8(2)]],
        "scalar COUNT(DISTINCT v) counts distinct NON-NULL values (10 and 20 = 2), excluding NULL"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_expression_key_forms_a_null_group_on_device() {
    // M3 (doc 21): GROUP BY a NULLABLE int4 EXPRESSION (`a + b`, b non-null) -- the rows where a is NULL
    // (so a+b is NULL) form their OWN group, rendered SqlValue::Null. Reuses the single-column NULL-key
    // reserved slot via the one nullable operand's validity bitmap (ZERO kernel change).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES (1,10),(NULL,10),(1,10),(2,10),(NULL,10)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("t")
        .unwrap()
        .device_memory_proof
        .is_none()
    {
        return;
    }
    let groups = |rows: &RowBlock| -> Vec<(Option<i32>, i64)> {
        let mut v: Vec<(Option<i32>, i64)> = rows
            .iter()
            .map(|r| {
                let k = match r[0] {
                    SqlValue::Int4(x) => Some(x),
                    SqlValue::Null => None,
                    ref o => panic!("unexpected key {o:?}"),
                };
                let c = match r[1] {
                    SqlValue::Int8(x) => x,
                    SqlValue::Int4(x) => i64::from(x),
                    ref o => panic!("unexpected count {o:?}"),
                };
                (k, c)
            })
            .collect();
        v.sort();
        v
    };
    // a+b: 11, NULL, 11, 12, NULL -> groups {11:2, 12:1, NULL:2}.
    let r = e
        .execute_resident_expr_select_sql("SELECT a + b, COUNT(*) FROM t GROUP BY a + b")
        .expect("GROUP BY nullable int4 expression");
    assert_eq!(
        groups(&r.rows),
        vec![(None, 2), (Some(11), 2), (Some(12), 1)],
        "GROUP BY a+b: the NULL-result rows form their own group (rendered NULL)"
    );
    // GROUP BY over an expression with TWO nullable operands clean-errors (needs a derived validity AND).
    e.execute_text(3, "CREATE TABLE t2 (a INT, b INT)").unwrap();
    e.execute_text(4, "INSERT INTO t2 (a,b) VALUES (1,2),(NULL,NULL)")
        .unwrap();
    if e.populate_relational_residency_snapshot("t2")
        .unwrap()
        .device_memory_proof
        .is_some()
    {
        assert!(
            e.execute_resident_expr_select_sql("SELECT a + b, COUNT(*) FROM t2 GROUP BY a + b")
                .is_err(),
            "GROUP BY an expression over two nullable operands clean-errors"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_nullable_expression_places_null_results_on_device() {
    // M3 (doc 21): ORDER BY a NULLABLE int4 EXPRESSION (`a + b`). A NULL result (any operand NULL) becomes
    // the i64::MAX default-end sentinel, blended ON-DEVICE (a validity-mask VM run + the blend kernel), so
    // NULL-expression rows sort to PG's default end. No host NULL decision.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (id INT, a INT, b INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (id,a,b) VALUES (1,5,1),(2,NULL,1),(3,2,1),(4,NULL,1)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("t")
        .unwrap()
        .device_memory_proof
        .is_none()
    {
        return;
    }
    let ids = |rows: &RowBlock| -> Vec<i32> {
        rows.iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect()
    };
    // a+b: id1=6, id2=NULL, id3=3, id4=NULL. ASC default = NULLS LAST; the secondary `id` orders the
    // (tied) NULL-result group deterministically -> [3 (=3), 1 (=6), 2 (NULL), 4 (NULL)].
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM t ORDER BY a + b, id")
        .expect("nullable int4 expression ORDER BY");
    assert_eq!(
        ids(&r.rows),
        vec![3, 1, 2, 4],
        "nullable a+b: non-NULL ascending, then NULL results last (PG default), on-device"
    );
    // A nullable int8 expression clean-errors (the i64::MAX NULL sentinel could collide with a real bigint).
    e.execute_text(3, "CREATE TABLE t8 (id INT, a BIGINT, b BIGINT)")
        .unwrap();
    e.execute_text(4, "INSERT INTO t8 (id,a,b) VALUES (1,5,1),(2,NULL,1)")
        .unwrap();
    if e.populate_relational_residency_snapshot("t8")
        .unwrap()
        .device_memory_proof
        .is_some()
    {
        assert!(
            e.execute_resident_expr_select_sql("SELECT id FROM t8 ORDER BY a + b")
                .is_err(),
            "nullable int8 expression ORDER BY clean-errors (sentinel collision)"
        );
    }
    // Explicit NULLS FIRST/LAST on a nullable expression clean-errors (value-sentinel only does default).
    assert!(
        e.execute_resident_expr_select_sql("SELECT id FROM t ORDER BY a + b NULLS FIRST")
            .is_err(),
        "explicit NULLS FIRST on a nullable expression clean-errors"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_result_order_by_explicit_nulls_first_last_on_device() {
    // M3 (doc 21): explicit NULLS FIRST/LAST on a GROUP BY result ORDER BY is honored ON-DEVICE by
    // `gpu_sort_permutation`; the validity bitmap and per-key nulls-first mask place the NULL group.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10),(NULL,10),(2,10),(NULL,10)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("t")
        .unwrap()
        .device_memory_proof
        .is_none()
    {
        return;
    }
    let keys = |rows: &RowBlock| -> Vec<Option<i32>> {
        rows.iter()
            .map(|r| match r[0] {
                SqlValue::Int4(x) => Some(x),
                SqlValue::Null => None,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect()
    };
    // groups {1, 2, NULL}. ORDER BY g NULLS FIRST overrides the ASC default (NULLS LAST) -> NULL first.
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g NULLS FIRST",
        )
        .expect("grouped ORDER BY NULLS FIRST");
    assert_eq!(
        keys(&r.rows),
        vec![None, Some(1), Some(2)],
        "grouped ORDER BY g NULLS FIRST"
    );
    // Default ASC = NULLS LAST.
    let r = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g")
        .expect("grouped ORDER BY default");
    assert_eq!(
        keys(&r.rows),
        vec![Some(1), Some(2), None],
        "grouped ORDER BY g default = NULLS LAST"
    );
    // DESC NULLS LAST overrides the DESC default (NULLS FIRST) -> descending then NULL last.
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g DESC NULLS LAST",
        )
        .expect("grouped ORDER BY DESC NULLS LAST");
    assert_eq!(
        keys(&r.rows),
        vec![Some(2), Some(1), None],
        "grouped ORDER BY g DESC NULLS LAST"
    );
    // A BIGINT result key is now exact too: its NULL is marked by an on-device validity bitmap (not a
    // value sentinel), so explicit NULLS FIRST is honored with no collision risk. g8=[1,NULL] -> NULL first.
    e.execute_text(3, "CREATE TABLE t8 (g BIGINT, v INT)")
        .unwrap();
    e.execute_text(4, "INSERT INTO t8 (g,v) VALUES (1,10),(NULL,10)")
        .unwrap();
    if e.populate_relational_residency_snapshot("t8")
        .unwrap()
        .device_memory_proof
        .is_some()
    {
        let r8 = e
            .execute_resident_expr_select_sql(
                "SELECT g, COUNT(*) FROM t8 GROUP BY g ORDER BY g NULLS FIRST",
            )
            .expect("bigint grouped ORDER BY g NULLS FIRST");
        let g8: Vec<Option<i64>> = r8
            .rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int8(x) => Some(x),
                SqlValue::Null => None,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect();
        assert_eq!(
            g8,
            vec![None, Some(1)],
            "bigint grouped ORDER BY g NULLS FIRST: NULL group first"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_order_by_explicit_nulls_first_last_on_device() {
    // M3 (doc 21): explicit NULLS FIRST/LAST on a JOIN-result ORDER BY is honored ON-DEVICE by
    // `sort_join_coordinates` (the override is threaded through `JoinPlan::order_by_nulls_first`). A LEFT
    // join pads the unmatched row's x to NULL; the override places it.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, n TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, x INT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (id,n) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(4, "INSERT INTO r (rid,x) VALUES (1,5),(2,7)")
        .unwrap();
    for t in ["l", "r"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let rows = |res: &RelationalSelectResult| -> Vec<(String, Option<i32>)> {
        res.rows
            .iter()
            .map(|r| {
                let n = match &r[0] {
                    SqlValue::Text(t) => t.clone(),
                    o => panic!("unexpected {o:?}"),
                };
                let x = match r[1] {
                    SqlValue::Int4(v) => Some(v),
                    SqlValue::Null => None,
                    ref o => panic!("unexpected {o:?}"),
                };
                (n, x)
            })
            .collect()
    };
    // result: (a,5),(b,7),(c,NULL). ORDER BY x NULLS FIRST overrides the ASC default -> NULL (c) first.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT n, x FROM l LEFT JOIN r ON l.id = r.rid ORDER BY x NULLS FIRST",
        )
        .expect("join ORDER BY x NULLS FIRST");
    assert_eq!(
        rows(&res),
        vec![
            ("c".into(), None),
            ("a".into(), Some(5)),
            ("b".into(), Some(7))
        ],
        "join ORDER BY x NULLS FIRST places the NULL-padded row first"
    );
    // Default ASC = NULLS LAST.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT n, x FROM l LEFT JOIN r ON l.id = r.rid ORDER BY x",
        )
        .expect("join ORDER BY x default");
    assert_eq!(
        rows(&res),
        vec![
            ("a".into(), Some(5)),
            ("b".into(), Some(7)),
            ("c".into(), None)
        ],
        "join ORDER BY x default = NULLS LAST"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_outer_join_with_where_filters_the_result_not_the_inputs() {
    // M3 (doc 21): a WHERE on an OUTER join filters the JOINED RESULT (PG semantics), NOT a per-side
    // pushdown (which is not filter-commutative for an outer join). The predicate runs on the GPU
    // (lower_resident_predicate); its survivor set post-filters the padded result: a JOIN_NULL_ROW pad
    // means the relation's columns are NULL -> the predicate is UNKNOWN -> the tuple is dropped.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, x INT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (id,name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(4, "INSERT INTO r (rid,x) VALUES (1,5),(2,7)")
        .unwrap();
    for t in ["l", "r"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let rows = |res: &RelationalSelectResult| -> Vec<(String, Option<i32>)> {
        let mut v: Vec<(String, Option<i32>)> = res
            .rows
            .iter()
            .map(|row| {
                let name = match &row[0] {
                    SqlValue::Text(t) => t.clone(),
                    o => panic!("unexpected {o:?}"),
                };
                let x = match row[1] {
                    SqlValue::Int4(v) => Some(v),
                    SqlValue::Null => None,
                    ref o => panic!("unexpected {o:?}"),
                };
                (name, x)
            })
            .collect();
        v.sort();
        v
    };

    // WHERE on the INNER (padded) side: `r.x = 5` drops the NULL-padded row (c) AND the non-matching
    // matched row (b, x=7) -- effectively inner on that condition. Only (a, 5) survives.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, x FROM l LEFT JOIN r ON l.id = r.rid WHERE r.x = 5",
        )
        .expect("LEFT JOIN with WHERE on the inner side");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        rows(&res),
        vec![("a".into(), Some(5))],
        "WHERE on the padded side filters the result (UNKNOWN on the NULL pad drops it)"
    );

    // WHERE on the PRESERVED (left) side: `l.id >= 2` keeps id 2 (matched, x=7) and id 3 (padded, NULL),
    // dropping id 1 -- the padding is preserved for the surviving left rows.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, x FROM l LEFT JOIN r ON l.id = r.rid WHERE l.id >= 2",
        )
        .expect("LEFT JOIN with WHERE on the preserved side");
    assert_eq!(
        rows(&res),
        vec![("b".into(), Some(7)), ("c".into(), None)],
        "WHERE on the preserved side keeps the NULL pad for surviving left rows"
    );

    // ANTI-JOIN: `WHERE r.x IS NULL` is TRUE on the NULL pad, so the post-filter must KEEP the padded
    // (unmatched-left) rows -- and drop every matched row (whose r.x is non-NULL). l=3 has no r match.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, x FROM l LEFT JOIN r ON l.id = r.rid WHERE r.x IS NULL",
        )
        .expect("LEFT JOIN anti-join (WHERE inner IS NULL)");
    assert_eq!(
        rows(&res),
        vec![("c".into(), None)],
        "anti-join: WHERE inner.col IS NULL keeps the NULL-padded (unmatched) left rows"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_outer_join_pad_where_3vl_on_device_v2() {
    // S6/V2 (doc 22): the OUTER-join NULL-pad's WHERE 3VL truth is decided ON-DEVICE (the all-NULL pad is
    // run through the SAME GPU WHERE-3VL mask VM as the real rows), replacing the host Kleene
    // `predicate_truth_on_null_pad`. Covers pad-SURVIVES (IS NULL, IS NULL OR cmp, IS NULL AND IS NULL) and
    // pad-DROPS (IS NOT NULL, comparison compound) across the SAME query so a wrong pad decision shows up as
    // a missing/extra row. l=3 ('c') is the unmatched (padded) left row; r1.x small, r2.x large.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, x INT, y INT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (id,name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(4, "INSERT INTO r (rid,x,y) VALUES (1,5,5),(2,200,50)")
        .unwrap();
    for t in ["l", "r"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let names = |res: &RelationalSelectResult| -> Vec<String> {
        let mut v: Vec<String> = res
            .rows
            .iter()
            .map(|row| match &row[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("expected text, got {o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    let q = |e: &mut Engine, whr: &str| -> Vec<String> {
        names(
            &e.execute_resident_expr_select_sql(&format!(
                "SELECT name FROM l LEFT JOIN r ON l.id = r.rid WHERE {whr}"
            ))
            .unwrap_or_else(|err| panic!("query `{whr}` failed: {err}")),
        )
    };
    // pad SURVIVES: `r.x IS NULL` is TRUE on the all-NULL pad -> keep 'c'; matched rows (x non-null) drop.
    assert_eq!(
        q(&mut e, "r.x IS NULL"),
        vec!["c".to_string()],
        "IS NULL: pad survives, matched drop"
    );
    // pad SURVIVES via the OR's IS NULL branch; matched r2 (x=200>100) also survives, r1 (x=5) drops.
    assert_eq!(
        q(&mut e, "r.x IS NULL OR r.x > 100"),
        vec!["b".to_string(), "c".to_string()],
        "IS NULL OR cmp: pad survives via IS NULL, a matched row survives via the comparison"
    );
    // pad SURVIVES: both leaves IS NULL -> TRUE AND TRUE on the pad; matched rows (both non-null) drop.
    assert_eq!(
        q(&mut e, "r.x IS NULL AND r.y IS NULL"),
        vec!["c".to_string()],
        "IS NULL AND IS NULL: pad survives, matched drop"
    );
    // pad DROPS: `IS NOT NULL` is FALSE on the all-NULL pad; matched rows (x non-null) survive.
    assert_eq!(
        q(&mut e, "r.x IS NOT NULL"),
        vec!["a".to_string(), "b".to_string()],
        "IS NOT NULL: pad drops, matched survive"
    );
    // pad DROPS: a comparison compound is UNKNOWN on the all-NULL pad; both matched rows satisfy it.
    assert_eq!(
        q(&mut e, "r.x > 1 AND r.y < 100"),
        vec!["a".to_string(), "b".to_string()],
        "comparison compound: pad drops (UNKNOWN), matched rows evaluated normally"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_outer_join_pad_where_3vl_types_and_full_join_v2() {
    // S6/V2: the on-device pad eval works for a NON-int pad column (numeric / text IS NULL) and for a FULL
    // join (both sides can be padded). A numeric/text `IS NULL` on the all-NULL pad must read the validity
    // bit on-device (0 -> NULL -> IS NULL TRUE), not depend on a host Kleene.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, amt NUMERIC(10,2), tag TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO l (id,name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO r (rid,amt,tag) VALUES (1,10.50,'p'),(2,20.00,'q')",
    )
    .unwrap();
    for t in ["l", "r"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let names = |res: &RelationalSelectResult| -> Vec<Option<String>> {
        let mut v: Vec<Option<String>> = res
            .rows
            .iter()
            .map(|row| match &row[0] {
                SqlValue::Text(t) => Some(t.clone()),
                SqlValue::Null => None,
                o => panic!("expected text/null, got {o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    // numeric pad column IS NULL -> pad 'c' survives (the device reads the numeric column's 0 validity bit).
    let num = e
        .execute_resident_expr_select_sql(
            "SELECT name FROM l LEFT JOIN r ON l.id = r.rid WHERE r.amt IS NULL",
        )
        .expect("numeric IS NULL anti-join");
    assert_eq!(
        names(&num),
        vec![Some("c".to_string())],
        "numeric pad column IS NULL: pad survives"
    );
    // text pad column IS NULL -> pad 'c' survives.
    let txt = e
        .execute_resident_expr_select_sql(
            "SELECT name FROM l LEFT JOIN r ON l.id = r.rid WHERE r.tag IS NULL",
        )
        .expect("text IS NULL anti-join");
    assert_eq!(
        names(&txt),
        vec![Some("c".to_string())],
        "text pad column IS NULL: pad survives"
    );
    // FULL join: the LEFT-only pad ('c', r columns NULL) survives `r.amt IS NULL`; matched rows drop; no
    // right-only row exists (every r matched). The pad decision is the same on-device path.
    let full = e
        .execute_resident_expr_select_sql(
            "SELECT name FROM l FULL JOIN r ON l.id = r.rid WHERE r.amt IS NULL",
        )
        .expect("full join with IS NULL");
    assert_eq!(
        names(&full),
        vec![Some("c".to_string())],
        "FULL join: the left-only pad survives IS NULL"
    );
}

// ── S6/V2 adversarial regression net (adopted from the independent audit of `76315706`) ──────────────
// Differential-grade (parent==child across 62 shapes) + non-vacuity-proven (sabotaging the pad eval to
// Ok(false)/Ok(true) makes these fail with the predicted wrong rows). They isolate the subtle interaction
// of a REAL NULL in the matched data with the synthetic all-NULL pad, an N-way OUTER carried JOIN_NULL_ROW,
// and the Kleene corner folds on the pad.

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s6_outer_where_real_null_mixed_with_pad() {
    // A real-NULL matched row AND the synthetic pad both satisfy `IS NULL` (the survivor pass evaluates the
    // real NULL on-device; the pad eval evaluates the synthetic NULL on-device -- both must agree). An N-way
    // OUTER then carries a JOIN_NULL_ROW into a SECOND pad.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, x INT)")
        .unwrap();
    e.execute_text(3, "CREATE TABLE s (sid INT, w INT, lbl TEXT)")
        .unwrap();
    e.execute_text(4, "INSERT INTO l (id,name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(5, "INSERT INTO r (rid,x) VALUES (1,5),(2,NULL)")
        .unwrap(); // r.x real NULL
    e.execute_text(6, "INSERT INTO s (sid,w,lbl) VALUES (1,9,'x'),(2,9,'y')")
        .unwrap();
    for t in ["l", "r", "s"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let names = |res: &RelationalSelectResult| -> Vec<String> {
        let mut v: Vec<String> = res
            .rows
            .iter()
            .map(|row| match &row[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("expected text, got {o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    let q = |e: &mut Engine, sql: &str| names(&e.execute_resident_expr_select_sql(sql).unwrap());
    // real-NULL matched row (b) AND pad (c) both satisfy IS NULL.
    assert_eq!(
        q(
            &mut e,
            "SELECT name FROM l LEFT JOIN r ON l.id=r.rid WHERE r.x IS NULL"
        ),
        vec!["b".to_string(), "c".to_string()]
    );
    // only the real non-null matched row (a) survives IS NOT NULL; real-null b and pad c drop.
    assert_eq!(
        q(
            &mut e,
            "SELECT name FROM l LEFT JOIN r ON l.id=r.rid WHERE r.x IS NOT NULL"
        ),
        vec!["a".to_string()]
    );
    // IS NULL OR cmp: a via 5>3, b via IS NULL, c via IS NULL -> all three.
    assert_eq!(
        q(
            &mut e,
            "SELECT name FROM l LEFT JOIN r ON l.id=r.rid WHERE r.x IS NULL OR r.x > 3"
        ),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    // N-way: c's r-pad carries a JOIN_NULL_ROW into s -> s.lbl NULL only for c.
    assert_eq!(
        q(
            &mut e,
            "SELECT name FROM l LEFT JOIN r ON l.id=r.rid LEFT JOIN s ON r.rid=s.sid WHERE s.lbl IS NULL"
        ),
        vec!["c".to_string()]
    );
    assert_eq!(
        q(
            &mut e,
            "SELECT name FROM l LEFT JOIN r ON l.id=r.rid LEFT JOIN s ON r.rid=s.sid WHERE s.w IS NOT NULL"
        ),
        vec!["a".to_string(), "b".to_string()]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s6_pad_where_kleene_corners() {
    // Kleene corners on the all-NULL pad (real data fully non-null so the survivor pass uses the
    // non-nullable peephole -> the pad eval is the only 3VL difference). Each fold is checked against PG.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "CREATE TABLE r (rid INT, xi INT, yi INT, fl BOOL, tag TEXT)",
    )
    .unwrap();
    e.execute_text(3, "INSERT INTO l (id,name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO r (rid,xi,yi,fl,tag) VALUES (1,5,5,true,'p'),(2,200,50,false,'q')",
    )
    .unwrap();
    for t in ["l", "r"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let q = |e: &mut Engine, whr: &str| -> Vec<String> {
        let res = e
            .execute_resident_expr_select_sql(&format!(
                "SELECT name FROM l LEFT JOIN r ON l.id=r.rid WHERE {whr}"
            ))
            .unwrap();
        let mut v: Vec<String> = res
            .rows
            .iter()
            .map(|row| match &row[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("expected text, got {o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    // IS NULL(T) AND cmp(U) -> U -> pad drops; matched rows fail the cmp -> none.
    assert!(q(&mut e, "r.xi IS NULL AND r.xi > 5").is_empty());
    // IS NULL(T) OR cmp -> T -> pad survives (c); matched b via 200>5.
    assert_eq!(
        q(&mut e, "r.xi IS NULL OR r.xi > 5"),
        vec!["b".to_string(), "c".to_string()]
    );
    // IS NOT NULL(F) OR IS NULL(T) -> T -> all (a,b matched non-null; c pad).
    assert_eq!(
        q(&mut e, "r.xi IS NOT NULL OR r.yi IS NULL"),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    // IS NOT NULL(F) AND IS NULL(T) -> F -> none.
    assert!(q(&mut e, "r.xi IS NOT NULL AND r.yi IS NULL").is_empty());
    // bool col on pad is UNKNOWN; `r.fl = false OR r.xi IS NULL` -> matched b (fl=false), pad c (IS NULL).
    assert_eq!(
        q(&mut e, "r.fl = false OR r.xi IS NULL"),
        vec!["b".to_string(), "c".to_string()]
    );
    // (IS NOT NULL AND cmp) OR IS NULL -> (F)OR(T) on pad -> survive c; b via (T AND 200>5).
    assert_eq!(
        q(&mut e, "(r.xi IS NOT NULL AND r.xi > 5) OR r.yi IS NULL"),
        vec!["b".to_string(), "c".to_string()]
    );
}
