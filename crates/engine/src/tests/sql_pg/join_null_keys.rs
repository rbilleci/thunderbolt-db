use crate::Engine;
use crate::RelationalSelectResult;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_two_relations_int_key() {
    // M5: a 2-relation INNER equi-join on an int key, via the general path's GPU hash join. parent.id
    // is UNIQUE (the build side); child.parent_id is the FK (1:N + an orphan + a childless parent).
    //   parent: (1,a),(2,b),(3,c)   child: (1,x),(1,y),(2,z),(99,orphan)
    //   parent JOIN child ON parent.id = child.parent_id -> (a,x),(a,y),(b,z); 99 + parent 3 dropped.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE parent (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE child (parent_id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO parent (id, name) VALUES (1,'a'),(2,'b'),(3,'c')",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO child (parent_id, label) VALUES (1,'x'),(1,'y'),(2,'z'),(99,'orphan')",
    )
    .unwrap();
    let ps = e.populate_relational_residency_snapshot("parent").unwrap();
    let cs = e.populate_relational_residency_snapshot("child").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    // Extract (name, label) text pairs + sort (the join emit order is unspecified without ORDER BY).
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = res
            .rows
            .iter()
            .map(|r| {
                let s = |c: &SqlValue| match c {
                    SqlValue::Text(t) => t.clone(),
                    other => panic!("expected text, got {other:?}"),
                };
                (s(&r[0]), s(&r[1]))
            })
            .collect();
        v.sort();
        v
    };
    let expected = vec![
        ("a".to_string(), "x".to_string()),
        ("a".to_string(), "y".to_string()),
        ("b".to_string(), "z".to_string()),
    ];
    // Via the general path directly + via the production wire/text dispatch (the hand-rolled parser
    // rejects JOIN -> the Err arm -> the general path).
    let direct = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM parent JOIN child ON parent.id = child.parent_id",
        )
        .expect("inner join (general path)");
    assert_eq!(direct.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(direct.columns.len(), 2);
    assert!(direct.columns[0].name.eq_ignore_ascii_case("name"));
    assert!(direct.columns[1].name.eq_ignore_ascii_case("label"));
    assert_eq!(
        pairs(&direct),
        expected,
        "1:N inner join, orphan + childless dropped"
    );
    let wire = e
        .execute_relational_select_text(
            "SELECT name, label FROM parent JOIN child ON parent.id = child.parent_id",
        )
        .expect("inner join (text/wire dispatch)");
    assert_eq!(wire.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        pairs(&wire),
        expected,
        "the wire dispatch routes JOIN to the general path"
    );
    // Qualified projection (parent.name, child.label) resolves each column to its relation.
    let qualified = e
        .execute_resident_expr_select_sql(
            "SELECT parent.name, child.label FROM parent JOIN child ON parent.id = child.parent_id",
        )
        .expect("inner join, qualified projection");
    assert_eq!(
        pairs(&qualified),
        expected,
        "qualified column refs resolve per-relation"
    );
    // The ON written the other way around (child.parent_id = parent.id) is the same join.
    let swapped = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM parent JOIN child ON child.parent_id = parent.id",
        )
        .expect("inner join, ON operands swapped");
    assert_eq!(
        pairs(&swapped),
        expected,
        "ON operand order does not matter"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_excludes_null_keys_three_valued_logic() {
    // M3 NULL-key gate (doc 21): in an equi-join `NULL = x` is UNKNOWN, so a row whose join key is NULL
    // matches NOTHING -- on BOTH sides. INT keys would otherwise share the 0 placeholder and SPURIOUSLY
    // match each other; TEXT/b128 keys would otherwise ERROR in the key gather. Both NULL-key rows drop.
    let mut e = Engine::new_local_test_engine();
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = res
            .rows
            .iter()
            .map(|r| {
                let s = |c: &SqlValue| match c {
                    SqlValue::Text(t) => t.clone(),
                    other => panic!("expected text, got {other:?}"),
                };
                (s(&r[0]), s(&r[1]))
            })
            .collect();
        v.sort();
        v
    };

    // --- INT key: without the gate, the two NULL ids share the 0 placeholder and spuriously pair. ---
    e.execute_text(1, "CREATE TABLE p (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE c (pid INT, label TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO p (id, name) VALUES (1,'a'),(2,'b'),(NULL,'pnull')",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO c (pid, label) VALUES (1,'x'),(2,'z'),(NULL,'cnull')",
    )
    .unwrap();
    let ps = e.populate_relational_residency_snapshot("p").unwrap();
    let cs = e.populate_relational_residency_snapshot("c").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let int_join = e
        .execute_resident_expr_select_sql("SELECT name, label FROM p JOIN c ON p.id = c.pid")
        .expect("int join with NULL keys");
    assert_eq!(int_join.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        pairs(&int_join),
        vec![
            ("a".to_string(), "x".to_string()),
            ("b".to_string(), "z".to_string())
        ],
        "NULL int keys match nothing -- no spurious ('pnull','cnull') pair"
    );

    // --- TEXT key: without the gate, the key-bytes gather would ERROR on a NULL cell. ---
    e.execute_text(5, "CREATE TABLE s (k TEXT, x TEXT)")
        .unwrap();
    e.execute_text(6, "CREATE TABLE u (k TEXT, y TEXT)")
        .unwrap();
    e.execute_text(7, "INSERT INTO s (k, x) VALUES ('m','sm'),(NULL,'snull')")
        .unwrap();
    e.execute_text(8, "INSERT INTO u (k, y) VALUES ('m','tm'),(NULL,'tnull')")
        .unwrap();
    let ss = e.populate_relational_residency_snapshot("s").unwrap();
    let us = e.populate_relational_residency_snapshot("u").unwrap();
    if ss.device_memory_proof.is_none() || us.device_memory_proof.is_none() {
        return;
    }
    let text_join = e
        .execute_resident_expr_select_sql("SELECT x, y FROM s JOIN u ON s.k = u.k")
        .expect("text join with NULL keys must not error");
    assert_eq!(
        pairs(&text_join),
        vec![("sm".to_string(), "tm".to_string())],
        "NULL text keys match nothing (and the gather does not error)"
    );

    // --- COMPOSITE int key (a.k1=b.k1 AND a.k2=b.k2): a row with ANY member NULL is excluded (else the
    // NULL member's 0 placeholder would pack to the same composite key and spuriously match). ---
    e.execute_text(9, "CREATE TABLE ca (k1 INT, k2 INT, x TEXT)")
        .unwrap();
    e.execute_text(10, "CREATE TABLE cb (k1 INT, k2 INT, y TEXT)")
        .unwrap();
    e.execute_text(
        11,
        "INSERT INTO ca (k1, k2, x) VALUES (1,1,'ca1'),(2,NULL,'canull')",
    )
    .unwrap();
    e.execute_text(
        12,
        "INSERT INTO cb (k1, k2, y) VALUES (1,1,'cb1'),(2,NULL,'cbnull')",
    )
    .unwrap();
    let cas = e.populate_relational_residency_snapshot("ca").unwrap();
    let cbs = e.populate_relational_residency_snapshot("cb").unwrap();
    if cas.device_memory_proof.is_none() || cbs.device_memory_proof.is_none() {
        return;
    }
    let comp_join = e
        .execute_resident_expr_select_sql(
            "SELECT x, y FROM ca JOIN cb ON ca.k1 = cb.k1 AND ca.k2 = cb.k2",
        )
        .expect("composite int join with a NULL member");
    assert_eq!(
        pairs(&comp_join),
        vec![("ca1".to_string(), "cb1".to_string())],
        "a NULL composite-key member excludes the row -- no ('canull','cbnull')"
    );

    // --- NUMERIC (b128) key: a NULL numeric key is excluded (and the 16-byte gather does not error). ---
    e.execute_text(13, "CREATE TABLE na (k NUMERIC(10,2), x TEXT)")
        .unwrap();
    e.execute_text(14, "CREATE TABLE nb (k NUMERIC(10,2), y TEXT)")
        .unwrap();
    e.execute_text(
        15,
        "INSERT INTO na (k, x) VALUES (1.50,'na1'),(NULL,'nanull')",
    )
    .unwrap();
    e.execute_text(
        16,
        "INSERT INTO nb (k, y) VALUES (1.50,'nb1'),(NULL,'nbnull')",
    )
    .unwrap();
    let nas = e.populate_relational_residency_snapshot("na").unwrap();
    let nbs = e.populate_relational_residency_snapshot("nb").unwrap();
    if nas.device_memory_proof.is_none() || nbs.device_memory_proof.is_none() {
        return;
    }
    let num_join = e
        .execute_resident_expr_select_sql("SELECT x, y FROM na JOIN nb ON na.k = nb.k")
        .expect("numeric join with NULL keys must not error");
    assert_eq!(
        pairs(&num_join),
        vec![("na1".to_string(), "nb1".to_string())],
        "NULL numeric keys match nothing (the b128 gather does not error)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_at_scale_grid_stride_v1b() {
    // V1b WIRE: the NULL-key skip is now ON-DEVICE (the hash-join kernel reads a per-side validity bitmap),
    // not a host pre-filter. At SCALE (>256 rows => multiple 256-thread blocks, grid-stride) a NULL key on
    // either side must (a) match nothing and (b) NOT spuriously match a REAL key 0 -- every NULL fixed-width
    // cell stores a 0 placeholder on-device, so the on-device skip is the ONLY thing preventing the NULL rows
    // from colliding with the real key 0 (which DOES exist here). bigp.id is unique on its non-NULL rows, so
    // the unique-build kernel runs; if the skip were broken, a NULL build row (placeholder 0) would collide
    // with real key 0 -> a spurious DuplicateBuildKey N:N fallback + wrong pairs, caught by the exact set.
    let mut e = Engine::new_local_test_engine();
    const N: i32 = 600;
    let mut pv = String::new();
    let mut cv = String::new();
    for i in 0..N {
        if i > 0 {
            pv.push(',');
            cv.push(',');
        }
        // NULL conditions chosen so id 0 stays a REAL key on both sides.
        if i % 13 == 5 {
            pv.push_str(&format!("(NULL,'p{i}')"));
        } else {
            pv.push_str(&format!("({i},'p{i}')"));
        }
        if i % 17 == 3 {
            cv.push_str(&format!("(NULL,'c{i}')"));
        } else {
            cv.push_str(&format!("({i},'c{i}')"));
        }
    }
    e.execute_text(1, "CREATE TABLE bigp (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE bigc (pid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, &format!("INSERT INTO bigp (id, name) VALUES {pv}"))
        .unwrap();
    e.execute_text(4, &format!("INSERT INTO bigc (pid, label) VALUES {cv}"))
        .unwrap();
    let ps = e.populate_relational_residency_snapshot("bigp").unwrap();
    let cs = e.populate_relational_residency_snapshot("bigc").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM bigp JOIN bigc ON bigp.id = bigc.pid",
        )
        .expect("scale int join with NULL keys");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| {
            let s = |c: &SqlValue| match c {
                SqlValue::Text(t) => t.clone(),
                o => panic!("expected text, got {o:?}"),
            };
            (s(&r[0]), s(&r[1]))
        })
        .collect();
    got.sort();
    let mut expected: Vec<(String, String)> = (0..N)
        .filter(|&i| i % 13 != 5 && i % 17 != 3)
        .map(|i| (format!("p{i}"), format!("c{i}")))
        .collect();
    expected.sort();
    assert_eq!(
        got, expected,
        "every non-NULL key matches 1:1; NULL rows (incl. the placeholder-0 rows) match nothing"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_int2_int8_uuid_v1b() {
    // V1b: the on-device NULL-key skip works for the int2 + int8 (i64 section) widths and the uuid (b128)
    // type (int4/text/numeric/composite are covered by gpu_inner_join_excludes_null_keys_*). int8 keys
    // exercise the i64-section gather; key 0 is a real key on both sides, so the NULL row's 0 placeholder
    // (a NULL int8 cell stores 0, NOT i64::MIN, so the launcher's i64::MIN reject is not tripped) must not
    // collide with it.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE i2a (k INT2, x TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE i2b (k INT2, y TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO i2a (k,x) VALUES (0,'a0'),(7,'a7'),(NULL,'anull')",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO i2b (k,y) VALUES (0,'b0'),(7,'b7'),(NULL,'bnull')",
    )
    .unwrap();
    e.execute_text(5, "CREATE TABLE i8a (k INT8, x TEXT)")
        .unwrap();
    e.execute_text(6, "CREATE TABLE i8b (k INT8, y TEXT)")
        .unwrap();
    e.execute_text(
        7,
        "INSERT INTO i8a (k,x) VALUES (0,'a0'),(9000000000,'abig'),(NULL,'anull')",
    )
    .unwrap();
    e.execute_text(
        8,
        "INSERT INTO i8b (k,y) VALUES (0,'b0'),(9000000000,'bbig'),(NULL,'bnull')",
    )
    .unwrap();
    let u1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    e.execute_text(9, "CREATE TABLE uxa (k UUID, x TEXT)")
        .unwrap();
    e.execute_text(10, "CREATE TABLE uxb (k UUID, y TEXT)")
        .unwrap();
    e.execute_text(
        11,
        &format!("INSERT INTO uxa (k,x) VALUES ('{u1}','a1'),(NULL,'anull')"),
    )
    .unwrap();
    e.execute_text(
        12,
        &format!("INSERT INTO uxb (k,y) VALUES ('{u1}','b1'),(NULL,'bnull')"),
    )
    .unwrap();
    let mut ok = true;
    for t in ["i2a", "i2b", "i8a", "i8b", "uxa", "uxb"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            o => panic!("expected text, got {o:?}"),
        };
        let mut v: Vec<(String, String)> = res.rows.iter().map(|r| (s(&r[0]), s(&r[1]))).collect();
        v.sort();
        v
    };
    let i2 = e
        .execute_resident_expr_select_sql("SELECT x, y FROM i2a JOIN i2b ON i2a.k = i2b.k")
        .expect("int2 NULL-key join");
    assert_eq!(
        pairs(&i2),
        vec![
            ("a0".to_string(), "b0".to_string()),
            ("a7".to_string(), "b7".to_string())
        ],
        "int2 NULL key skipped; key 0 still matches"
    );
    let i8 = e
        .execute_resident_expr_select_sql("SELECT x, y FROM i8a JOIN i8b ON i8a.k = i8b.k")
        .expect("int8 NULL-key join");
    assert_eq!(
        pairs(&i8),
        vec![
            ("a0".to_string(), "b0".to_string()),
            ("abig".to_string(), "bbig".to_string())
        ],
        "int8 NULL key skipped; real key 0 + the big key match (placeholder 0 != i64::MIN)"
    );
    let ux = e
        .execute_resident_expr_select_sql("SELECT x, y FROM uxa JOIN uxb ON uxa.k = uxb.k")
        .expect("uuid NULL-key join");
    assert_eq!(
        pairs(&ux),
        vec![("a1".to_string(), "b1".to_string())],
        "uuid NULL key skipped on the b128 path"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_n_to_n_int_and_text_v1b() {
    // V1b: the N:N (many-to-many chaining) kernels skip a NULL key too -- a NULL BUILD key is never chained,
    // a NULL PROBE key emits nothing. Both sides carry DUPLICATE keys (forcing the N:N fallback) PLUS NULLs;
    // key 0 (int) / 'm' (text) is a real DUPLICATED key, so a broken skip would chain the placeholder-0 /
    // empty-string NULL rows into that key's cross-product and emit spurious pairs.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE nna (k INT, x TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE nnb (k INT, y TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO nna (k,x) VALUES (0,'a0a'),(0,'a0b'),(1,'a1'),(NULL,'anull')",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO nnb (k,y) VALUES (0,'b0a'),(0,'b0b'),(1,'b1'),(NULL,'bnull')",
    )
    .unwrap();
    e.execute_text(5, "CREATE TABLE tta (k TEXT, x TEXT)")
        .unwrap();
    e.execute_text(6, "CREATE TABLE ttb (k TEXT, y TEXT)")
        .unwrap();
    e.execute_text(
        7,
        "INSERT INTO tta (k,x) VALUES ('m','a1'),('m','a2'),(NULL,'anull')",
    )
    .unwrap();
    e.execute_text(
        8,
        "INSERT INTO ttb (k,y) VALUES ('m','b1'),('m','b2'),(NULL,'bnull')",
    )
    .unwrap();
    let mut ok = true;
    for t in ["nna", "nnb", "tta", "ttb"] {
        ok &= e
            .populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_some();
    }
    if !ok {
        return;
    }
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            o => panic!("expected text, got {o:?}"),
        };
        let mut v: Vec<(String, String)> = res.rows.iter().map(|r| (s(&r[0]), s(&r[1]))).collect();
        v.sort();
        v
    };
    let int_nn = e
        .execute_resident_expr_select_sql("SELECT x, y FROM nna JOIN nnb ON nna.k = nnb.k")
        .expect("int N:N NULL-key join");
    assert_eq!(
        pairs(&int_nn),
        vec![
            ("a0a".to_string(), "b0a".to_string()),
            ("a0a".to_string(), "b0b".to_string()),
            ("a0b".to_string(), "b0a".to_string()),
            ("a0b".to_string(), "b0b".to_string()),
            ("a1".to_string(), "b1".to_string()),
        ],
        "int N:N: key 0's 2x2 cross product + key 1's 1x1; NULL rows never chain/emit"
    );
    let text_nn = e
        .execute_resident_expr_select_sql("SELECT x, y FROM tta JOIN ttb ON tta.k = ttb.k")
        .expect("text N:N NULL-key join");
    assert_eq!(
        pairs(&text_nn),
        vec![
            ("a1".to_string(), "b1".to_string()),
            ("a1".to_string(), "b2".to_string()),
            ("a2".to_string(), "b1".to_string()),
            ("a2".to_string(), "b2".to_string()),
        ],
        "text N:N: 'm's 2x2 cross product; NULL rows never chain/emit"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_right_and_full_outer_padded_v1b() {
    // V1b: a NULL-key row in an OUTER join matches nothing (skipped on-device) and so must be PADDED on the
    // outer side, never dropped or spuriously matched against the OTHER side's NULL-key row. RIGHT keeps
    // every right row (its NULL-key row left-padded; the left NULL-key row is left-only -> dropped); FULL
    // keeps both sides' unmatched rows (incl. both NULL-key rows).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE ol (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE orr (rid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO ol (id, name) VALUES (1,'a'),(NULL,'lnull')")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO orr (rid, label) VALUES (1,'x'),(NULL,'rnull')",
    )
    .unwrap();
    let ls = e.populate_relational_residency_snapshot("ol").unwrap();
    let rs = e.populate_relational_residency_snapshot("orr").unwrap();
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
    let right = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM ol RIGHT JOIN orr ON ol.id = orr.rid",
        )
        .expect("right outer join with NULL keys");
    assert_eq!(right.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        opt_pairs(&right),
        vec![
            (None, Some("rnull".to_string())), // right NULL-key row -> left padded (NOT matched to 'lnull')
            (Some("a".to_string()), Some("x".to_string())),
        ],
        "RIGHT: the right NULL-key row is left-padded; the left NULL-key row is dropped (left-only)"
    );
    let full = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM ol FULL JOIN orr ON ol.id = orr.rid",
        )
        .expect("full outer join with NULL keys");
    assert_eq!(
        opt_pairs(&full),
        vec![
            (None, Some("rnull".to_string())), // right-only NULL-key row
            (Some("a".to_string()), Some("x".to_string())), // the lone match
            (Some("lnull".to_string()), None), // left-only NULL-key row
        ],
        "FULL: both NULL-key rows are kept and padded; they do NOT match each other"
    );
}

// ── V1b WIRE adversarial regression net (adopted from the independent audit of `5724bf55`) ───────────
// Each was fault-injection-proven non-vacuous by the auditor: with the on-device skip disabled (bitmaps
// forced to None), each FAILS with the exact spurious match noted. They isolate cases the author's 4 v1b
// tests don't: build-side swap, an all-NULL build column, composite member-AND, the anti-join silent
// data-loss, and the 32-bit bitmap word boundary.

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_all_null_build_column_empty_effective_build() {
    // An ENTIRELY-NULL build key column: every build key is skipped on-device => the effective build is
    // empty => nothing matches, EVEN against a real probe key 0. The build's placeholder index 0 is itself
    // a NULL row. Skip-disabled: the two NULL build rows (placeholder key 0) collide => DuplicateBuildKey =>
    // N:N => both spuriously match the probe's real key 0.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE ba (k INT, x TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE bb (k INT, y TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO ba (k,x) VALUES (NULL,'a0'),(NULL,'a1')")
        .unwrap();
    e.execute_text(4, "INSERT INTO bb (k,y) VALUES (0,'b0'),(5,'b5')")
        .unwrap();
    let bas = e.populate_relational_residency_snapshot("ba").unwrap();
    let bbs = e.populate_relational_residency_snapshot("bb").unwrap();
    if bas.device_memory_proof.is_none() || bbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT x, y FROM ba JOIN bb ON ba.k = bb.k")
        .expect("all-NULL build column join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert!(
        res.rows.is_empty(),
        "an all-NULL build key column matches nothing -- not even the probe's real key 0; got {:?}",
        res.rows
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_null_on_build_side_when_smaller_side_swaps() {
    // The bitmaps must SWAP with the keys when the smaller side becomes the build. Here the NEW (right) side
    // is smaller, so the hash join builds on it (build/probe swap); the right bitmap must follow to the build
    // slot. NULLs on BOTH sides. Skip-disabled or a mis-swapped bitmap => a NULL row leaks into the result.
    let mut e = Engine::new_local_test_engine();
    // acc (left) = 3 rows incl a NULL; new (right) = 2 rows incl a NULL => smaller_is_left = false => the
    // build swaps to the right side.
    e.execute_text(1, "CREATE TABLE sa (k INT, x TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE sb (k INT, y TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO sa (k,x) VALUES (1,'a1'),(2,'a2'),(NULL,'anull')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO sb (k,y) VALUES (1,'b1'),(NULL,'bnull')")
        .unwrap();
    let sas = e.populate_relational_residency_snapshot("sa").unwrap();
    let sbs = e.populate_relational_residency_snapshot("sb").unwrap();
    if sas.device_memory_proof.is_none() || sbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT x, y FROM sa JOIN sb ON sa.k = sb.k")
        .expect("build-side-swap NULL-key join");
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            o => panic!("expected (text,text), got {o:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![("a1".to_string(), "b1".to_string())],
        "only key 1 matches; the NULL rows on both sides are skipped even though build/probe swapped"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_composite_one_member_null_other_equals_real_row() {
    // A composite key where ONE member is NULL but the other equals a real row's member. Validity must be
    // AND'd across BOTH members, so `(5,NULL)` is skipped. The NULL member's 0 placeholder makes `(5,NULL)`
    // pack identically to a real `(5,0)` row on the other side -- if validity were checked on member0 only,
    // `(5,NULL)` would spuriously match `(5,0)`.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE ca (k1 INT, k2 INT, x TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE cb (k1 INT, k2 INT, y TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO ca (k1,k2,x) VALUES (5,7,'match'),(5,NULL,'pnull')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO cb (k1,k2,y) VALUES (5,7,'cb7'),(5,0,'cb0')")
        .unwrap();
    let cas = e.populate_relational_residency_snapshot("ca").unwrap();
    let cbs = e.populate_relational_residency_snapshot("cb").unwrap();
    if cas.device_memory_proof.is_none() || cbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT x, y FROM ca JOIN cb ON ca.k1 = cb.k1 AND ca.k2 = cb.k2",
        )
        .expect("composite NULL-member join");
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            o => panic!("expected (text,text), got {o:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![("match".to_string(), "cb7".to_string())],
        "(5,NULL) is skipped (validity AND'd across members); no spurious ('pnull','cb0')"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_anti_join_left_where_inner_is_null_with_null_keys() {
    // The anti-join `LEFT JOIN ... WHERE inner IS NULL` with NULL keys on BOTH sides. The left NULL-key row
    // matches nothing => it is NULL-padded => WHERE inner IS NULL KEEPS it. If the two NULL rows spuriously
    // matched, the left NULL-key row would be a MATCH (inner not NULL) and silently DROPPED -- data loss.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE la (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE lb (rid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO la (id,name) VALUES (1,'a'),(NULL,'lnull')")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO lb (rid,label) VALUES (1,'x'),(NULL,'rnull')",
    )
    .unwrap();
    let las = e.populate_relational_residency_snapshot("la").unwrap();
    let lbs = e.populate_relational_residency_snapshot("lb").unwrap();
    if las.device_memory_proof.is_none() || lbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name FROM la LEFT JOIN lb ON la.id = lb.rid WHERE lb.label IS NULL",
        )
        .expect("anti-join with NULL keys");
    let mut got: Vec<String> = res
        .rows
        .iter()
        .map(|r| match &r[0] {
            SqlValue::Text(s) => s.clone(),
            o => panic!("expected text, got {o:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec!["lnull".to_string()],
        "the unmatched NULL-key left row is padded + kept by WHERE inner IS NULL (not spuriously matched)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_null_at_word_boundary_index_32() {
    // The ONLY NULL key is at row index 32 -- bitmap word 1 (32>>5), bit 0 (32&31). Catches any off-by-one
    // in the kernel's bitmap word/bit math: exactly row 32 must be skipped, every other row matches 1:1.
    let mut e = Engine::new_local_test_engine();
    const N: i32 = 40;
    let mut wa = String::new();
    let mut wb = String::new();
    for i in 0..N {
        if i > 0 {
            wa.push(',');
            wb.push(',');
        }
        if i == 32 {
            wa.push_str(&format!("(NULL,'a{i}')"));
        } else {
            wa.push_str(&format!("({i},'a{i}')"));
        }
        wb.push_str(&format!("({i},'b{i}')"));
    }
    e.execute_text(1, "CREATE TABLE wa (k INT, x TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE wb (k INT, y TEXT)")
        .unwrap();
    e.execute_text(3, &format!("INSERT INTO wa (k,x) VALUES {wa}"))
        .unwrap();
    e.execute_text(4, &format!("INSERT INTO wb (k,y) VALUES {wb}"))
        .unwrap();
    let was = e.populate_relational_residency_snapshot("wa").unwrap();
    let wbs = e.populate_relational_residency_snapshot("wb").unwrap();
    if was.device_memory_proof.is_none() || wbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT x, y FROM wa JOIN wb ON wa.k = wb.k")
        .expect("word-boundary NULL-key join");
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            o => panic!("expected (text,text), got {o:?}"),
        })
        .collect();
    got.sort();
    let mut expected: Vec<(String, String)> = (0..N)
        .filter(|&i| i != 32)
        .map(|i| (format!("a{i}"), format!("b{i}")))
        .collect();
    expected.sort();
    assert_eq!(
        got, expected,
        "exactly row 32 (word 1, bit 0) is skipped; all others match 1:1"
    );
}
