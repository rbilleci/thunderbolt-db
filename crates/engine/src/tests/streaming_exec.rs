use super::*;
mod chunk_class_lifecycle;
mod chunk_locate;
mod grouped_distinct;
mod ordered;
mod projection;
mod rank_windows;
mod reverse_gather;
mod scalar_reductions;
mod views;

/// Probe for a usable GPU by populating a throwaway table and checking the retained device proof, then
/// drop it (so it leaves no residency / budget footprint). Returns false to SKIP a GPU test off-box.
fn gpu_available(e: &mut Engine, seq: &mut u64) -> bool {
    *seq += 1;
    e.execute_text(*seq, "CREATE TABLE __se_probe (x INT)")
        .unwrap();
    *seq += 1;
    e.execute_text(*seq, "INSERT INTO __se_probe VALUES (1)")
        .unwrap();
    let snapshot = e
        .populate_relational_residency_snapshot("__se_probe")
        .unwrap();
    let available = snapshot.device_memory_proof.is_some();
    *seq += 1;
    e.execute_text(*seq, "DROP TABLE __se_probe").unwrap();
    available
}

fn select(sql: &str) -> Select {
    let Command::Select(select) = parse_command(sql).unwrap() else {
        panic!("not a SELECT: {sql}");
    };
    select
}

/// ADR-012 JOIN: two non-resident relations are captured at budget/4, joined as bounded logical
/// block pairs on the GPU, and concatenated. NULL join keys never match; NULL projected values survive.
/// Differential oracle is the same GPU join after explicit whole-table residency, never a CPU join.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_inner_join_two_over_budget_relations() {
    let _entry_disabled = ClassEntryDisabled::new();
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jl (k INT, lv INT, note TEXT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jr (k INT, rv INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jna (k INT, x INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jnb (k INT, y INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jnc (k INT, z INT, note TEXT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jnd (k INT, q INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE joa (k INT, x INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE job (k INT, y INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jxa (k INT, x INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jxb (k INT, y INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jxc (x INT, y INT, z INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jm4 (k INT, v INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE jm8 (k BIGINT, v INT)")
        .unwrap();
    let left = (0..600)
        .map(|i| {
            let key = if i % 97 == 0 {
                "NULL".to_string()
            } else {
                i.to_string()
            };
            let note = if i % 17 == 0 {
                "NULL".to_string()
            } else {
                format!("'n{i:04}'")
            };
            format!("({key}, {i}, {note})")
        })
        .collect::<Vec<_>>()
        .join(",");
    let right = (300..900)
        .map(|i| {
            let key = if i % 89 == 0 {
                "NULL".to_string()
            } else {
                i.to_string()
            };
            format!("({key}, {i})")
        })
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO jl VALUES {left}"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO jr VALUES {right}"))
        .unwrap();
    let nna = (0..60)
        .map(|i| format!("({}, {i})", i % 3))
        .collect::<Vec<_>>()
        .join(",");
    let nnb = (0..45)
        .map(|i| format!("({}, {i})", i % 3))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO jna VALUES {nna}"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO jna VALUES (9, 999)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO jnb VALUES {nnb}"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO jnb VALUES (7, 777)")
        .unwrap();
    let nnc = (0..30)
        .map(|i| {
            let key = if i == 29 {
                "NULL".to_string()
            } else {
                (i % 3).to_string()
            };
            let note = if i % 5 == 0 {
                "NULL".to_string()
            } else {
                format!("'c{i:02}'")
            };
            format!("({key}, {i}, {note})")
        })
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO jnc VALUES {nnc}"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO jnd VALUES (0, 7), (0, 8), (7, 9)")
        .unwrap();
    let outer_a = (0..400)
        .map(|i| format!("({i}, {})", i * 10))
        .collect::<Vec<_>>()
        .join(",");
    let outer_b = (200..600)
        .map(|i| format!("({i}, {})", i * 100))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO joa VALUES {outer_a}"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO job VALUES {outer_b}"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO jxa VALUES (1, 10), (2, 20)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO jxb VALUES (1, 100), (2, 200)")
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO jxc VALUES (10, 100, 111), (20, 200, 222), (10, 200, 999)",
    )
    .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO jm4 VALUES (-1, 10), (2, 20)")
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO jm8 VALUES (-1, 100), (2, 200), (2147483648, 999)",
    )
    .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM jl WHERE k = 400").unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    // Prime both cold entries after the device-native delete. This test keeps class entry disabled
    // because its differential oracle later re-admits the same store generation whole-resident.
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM jl"))
        .unwrap();
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM jr"))
        .unwrap();
    let sql = "SELECT l.k, l.lv, r.rv, l.note FROM jl l JOIN jr r ON l.k = r.k \
               WHERE l.lv >= 350 AND r.rv < 550";
    let streamed = e
        .execute_resident_expr_select_sql(sql)
        .expect("streaming join");
    assert_eq!(streamed.executed_target, DeviceTarget::Gpu(0));
    assert!(e.streaming_join_hits() > 0, "streaming join route fired");
    assert!(
        e.streaming_join_block_pairs() > 1,
        "genuine multi-block fold"
    );
    assert!(
        e.streaming_join_peak_device_bytes() <= 4096,
        "all simultaneously live join payload/scratch allocations stay within budget"
    );
    let ordered_sql = "SELECT l.k, l.lv, r.rv, l.note FROM jl l JOIN jr r ON l.k = r.k \
                       WHERE l.lv >= 350 AND r.rv < 550 \
                       ORDER BY l.lv DESC LIMIT 31 OFFSET 7";
    let streamed_ordered = e
        .execute_resident_expr_select_sql(ordered_sql)
        .expect("streaming ordered top-N join");
    assert_eq!(streamed_ordered.rows.len(), 31);
    let hidden_order_sql = "SELECT l.lv FROM jl l JOIN jr r ON l.k = r.k \
                            ORDER BY r.rv DESC LIMIT 23 OFFSET 4";
    let streamed_hidden_order = e
        .execute_resident_expr_select_sql(hidden_order_sql)
        .expect("qualified non-projected streaming JOIN order key");
    let aliased_order = e
        .execute_resident_expr_select_sql(
            "SELECT l.lv AS left_value FROM jl l JOIN jr r ON l.k = r.k \
             ORDER BY left_value DESC LIMIT 7",
        )
        .expect("streaming JOIN output alias and alias ORDER BY");
    assert_eq!(aliased_order.columns[0].name, "left_value");
    let aliased_values = aliased_order
        .rows
        .iter()
        .map(|row| match row[0] {
            SqlValue::Int4(value) => value,
            ref other => panic!("aliased value: {other:?}"),
        })
        .collect::<Vec<_>>();
    assert!(aliased_values
        .windows(2)
        .all(|values| values[0] >= values[1]));
    let ambiguous_where = e
        .execute_resident_expr_select_sql(
            "SELECT l.lv FROM jl l JOIN jr r ON l.k = r.k WHERE k > 350",
        )
        .expect_err("unqualified duplicate WHERE column must be ambiguous");
    assert!(ambiguous_where.to_string().contains("ambiguous"));
    let mixed_width = e
        .execute_resident_expr_select_sql(
            "SELECT a.v, b.v FROM jm4 a JOIN jm8 b ON a.k = b.k ORDER BY a.v",
        )
        .expect("mixed int4/int8 streaming join");
    assert_eq!(
        mixed_width.rows,
        vec![
            vec![SqlValue::Int4(10), SqlValue::Int4(100)],
            vec![SqlValue::Int4(20), SqlValue::Int4(200)],
        ]
    );
    let duplicate_name_order_sql = "SELECT l.k, r.k FROM jl l JOIN jr r ON l.k = r.k \
                                    ORDER BY r.k DESC LIMIT 19";
    let streamed_duplicate_name_order = e
        .execute_resident_expr_select_sql(duplicate_name_order_sql)
        .expect("qualified streaming JOIN order key with duplicate output names");
    let nn_sql = "SELECT a.x, b.y FROM jna a JOIN jnb b ON a.k = b.k";
    let streamed_nn = e
        .execute_resident_expr_select_sql(nn_sql)
        .expect("bounded streaming N:N join");
    assert_eq!(streamed_nn.rows.len(), 900);
    // The bounded 4-column run includes a variable-width text column and 78 retained candidates.
    // Allocator-backed accounting charges the real power-of-two pool buckets, so this distinct shape
    // gets an honest 8 KiB query budget rather than a logical-width estimate.
    e.set_relational_residency_budget_bytes(0, 8192);
    let nway_sql = "SELECT a.x, b.y, c.z, c.note \
                    FROM jna a JOIN jnb b ON a.k = b.k \
                               JOIN jnc c ON b.k = c.k \
                    ORDER BY a.x, b.y, c.z LIMIT 73 OFFSET 5";
    let streamed_nway = e
        .execute_resident_expr_select_sql(nway_sql)
        .expect("bounded three-relation streaming join");
    assert_eq!(streamed_nway.rows.len(), 73);
    assert!(streamed_nway
        .rows
        .iter()
        .any(|row| row[3] == SqlValue::Null));
    let cross_relation_composite = e
        .execute_resident_expr_select_sql(
            "SELECT c.z FROM jxa a JOIN jxb b ON a.k = b.k \
             JOIN jxc c ON a.x = c.x AND b.y = c.y",
        )
        .expect("composite N-way keys from different accumulated relations");
    let mut cross_relation_rows = cross_relation_composite.rows.clone().into_boxed();
    cross_relation_rows.sort();
    assert_eq!(
        cross_relation_rows,
        vec![vec![SqlValue::Int4(111)], vec![SqlValue::Int4(222)]],
        "each composite component must gather from its own accumulated relation; z=999 is a trap"
    );
    let nway_left_sql = "SELECT a.x, b.y, d.q \
                         FROM jna a LEFT JOIN jnb b ON a.k = b.k \
                                    LEFT JOIN jnd d ON b.k = d.k \
                         ORDER BY a.x, b.y, d.q LIMIT 41";
    let streamed_nway_left = e
        .execute_resident_expr_select_sql(nway_left_sql)
        .expect("streaming multi-step LEFT OUTER join");
    assert_eq!(streamed_nway_left.rows.len(), 41);
    assert!(streamed_nway_left
        .rows
        .iter()
        .any(|row| row[2] == SqlValue::Null));
    let nway_right_sql = "SELECT a.x, b.y, d.q \
                          FROM jna a JOIN jnb b ON a.k = b.k \
                                     RIGHT JOIN jnd d ON b.k = d.k \
                          ORDER BY d.q DESC, a.x, b.y LIMIT 5";
    let streamed_nway_right = e
        .execute_resident_expr_select_sql(nway_right_sql)
        .expect("streaming RIGHT step over a multi-relation accumulated side");
    assert_eq!(
        streamed_nway_right.rows.row(0),
        &[SqlValue::Null, SqlValue::Null, SqlValue::Int4(9)]
    );
    let nway_two_right_sql = "SELECT a.x, b.y, d.q \
                              FROM jna a RIGHT JOIN jnb b ON a.k = b.k \
                                         FULL JOIN jnd d ON b.k = d.k \
                              ORDER BY b.y DESC NULLS LAST, d.q, a.x LIMIT 5";
    e.set_relational_residency_budget_bytes(0, 4608);
    let streamed_nway_two_right = e
        .execute_resident_expr_select_sql(nway_two_right_sql)
        .expect("streaming prefix replay across two RIGHT/FULL steps");
    assert!(streamed_nway_two_right
        .rows
        .iter()
        .any(|row| { row == [SqlValue::Null, SqlValue::Int4(777), SqlValue::Int4(9)] }));
    e.set_relational_residency_budget_bytes(0, 4096);
    let outer_sql = "SELECT a.k, a.x, b.k, b.y FROM joa a FULL JOIN job b ON a.k = b.k";
    let streamed_outer = e
        .execute_resident_expr_select_sql(outer_sql)
        .expect("streaming FULL OUTER join");
    assert_eq!(streamed_outer.rows.len(), 600);
    assert!(streamed_outer
        .rows
        .iter()
        .any(|row| row[0] == SqlValue::Null));
    assert!(streamed_outer
        .rows
        .iter()
        .any(|row| row[2] == SqlValue::Null));
    let outer_where_sql = "SELECT a.k, a.x, b.y FROM joa a LEFT JOIN job b ON a.k = b.k \
                           WHERE b.y IS NULL";
    let streamed_outer_where = e
        .execute_resident_expr_select_sql(outer_where_sql)
        .expect("streaming OUTER WHERE anti-join");
    assert_eq!(streamed_outer_where.rows.len(), 200);
    assert!(streamed_outer_where
        .rows
        .iter()
        .all(|row| row[2] == SqlValue::Null));
    // Keep offset+limit itself within the budget while forcing the final window to cross from
    // matched rows into the globally appended right-unmatched tail.
    e.set_relational_residency_budget_bytes(0, 32 * 1024);
    let outer_ordered_sql = "SELECT a.k, a.x, b.k, b.y FROM joa a FULL JOIN job b ON a.k = b.k \
                             ORDER BY b.y ASC NULLS LAST LIMIT 37 OFFSET 390";
    let streamed_outer_ordered = e
        .execute_resident_expr_select_sql(outer_ordered_sql)
        .expect("streaming ordered FULL OUTER join");
    assert_eq!(streamed_outer_ordered.rows.len(), 37);
    let outer_limit_sql =
        "SELECT a.k, a.x, b.k, b.y FROM joa a FULL JOIN job b ON a.k = b.k LIMIT 29 OFFSET 13";
    let streamed_outer_limit = e
        .execute_resident_expr_select_sql(outer_limit_sql)
        .expect("bounded unordered FULL OUTER join window");
    assert_eq!(streamed_outer_limit.rows.len(), 29);

    // GPU-native whole-resident oracle.
    e.clear_relational_residency_budget_bytes(0);
    e.populate_relational_residency_snapshot("jl").unwrap();
    e.populate_relational_residency_snapshot("jr").unwrap();
    e.populate_relational_residency_snapshot("jna").unwrap();
    e.populate_relational_residency_snapshot("jnb").unwrap();
    e.populate_relational_residency_snapshot("jnc").unwrap();
    e.populate_relational_residency_snapshot("jnd").unwrap();
    e.populate_relational_residency_snapshot("joa").unwrap();
    e.populate_relational_residency_snapshot("job").unwrap();
    let resident = e
        .execute_resident_expr_select_sql(sql)
        .expect("resident join oracle");
    let resident_ordered = e
        .execute_resident_expr_select_sql(ordered_sql)
        .expect("resident ordered join oracle");
    let resident_hidden_order = e
        .execute_resident_expr_select_sql(hidden_order_sql)
        .expect("resident hidden-order join oracle");
    let resident_duplicate_name_order = e
        .execute_resident_expr_select_sql(duplicate_name_order_sql)
        .expect("resident duplicate-name order oracle");
    let resident_nn = e
        .execute_resident_expr_select_sql(nn_sql)
        .expect("resident N:N join oracle");
    let resident_nway = e
        .execute_resident_expr_select_sql(nway_sql)
        .expect("resident three-relation join oracle");
    let resident_nway_left = e
        .execute_resident_expr_select_sql(nway_left_sql)
        .expect("resident multi-step LEFT oracle");
    let resident_nway_right = e
        .execute_resident_expr_select_sql(nway_right_sql)
        .expect("resident multi-step RIGHT oracle");
    let resident_nway_two_right = e
        .execute_resident_expr_select_sql(nway_two_right_sql)
        .expect("resident two-step RIGHT/FULL oracle");
    let resident_outer = e
        .execute_resident_expr_select_sql(outer_sql)
        .expect("resident FULL OUTER join oracle");
    let resident_outer_where = e
        .execute_resident_expr_select_sql(outer_where_sql)
        .expect("resident OUTER WHERE oracle");
    let resident_outer_ordered = e
        .execute_resident_expr_select_sql(outer_ordered_sql)
        .expect("resident ordered FULL OUTER join oracle");
    let mut streamed_rows = streamed.rows.clone().into_boxed();
    let mut resident_rows = resident.rows.clone().into_boxed();
    streamed_rows.sort_by(|a, b| crate::rel_exec_helpers::compare_sql_values(&a[1], &b[1]));
    resident_rows.sort_by(|a, b| crate::rel_exec_helpers::compare_sql_values(&a[1], &b[1]));
    assert_eq!(streamed_rows, resident_rows);
    assert_eq!(
        streamed_ordered.rows.clone().into_boxed(),
        resident_ordered.rows.clone().into_boxed(),
        "streaming local-top-N compaction + final device merge matches resident GPU ordering"
    );
    assert_eq!(
        streamed_hidden_order.rows.clone().into_boxed(),
        resident_hidden_order.rows.clone().into_boxed(),
        "qualified non-projected ORDER BY remains device-resident"
    );
    assert_eq!(
        streamed_duplicate_name_order.rows.clone().into_boxed(),
        resident_duplicate_name_order.rows.clone().into_boxed(),
        "qualified ORDER BY resolves provenance despite duplicate projected names"
    );
    let sort_pair_rows = |result: &RelationalSelectResult| {
        let mut rows = result.rows.clone().into_boxed();
        rows.sort_by(|a, b| {
            crate::rel_exec_helpers::compare_sql_values(&a[0], &b[0])
                .then_with(|| crate::rel_exec_helpers::compare_sql_values(&a[1], &b[1]))
        });
        rows
    };
    assert_eq!(sort_pair_rows(&streamed_nn), sort_pair_rows(&resident_nn));
    assert_eq!(
        streamed_nway.rows.clone().into_boxed(),
        resident_nway.rows.clone().into_boxed(),
        "N-way chunk/block Cartesian scheduling matches the resident left-deep GPU join"
    );
    assert_eq!(
        streamed_nway_left.rows.clone().into_boxed(),
        resident_nway_left.rows.clone().into_boxed(),
        "multi-step LEFT scheduling emits each globally unmatched tuple exactly once"
    );
    assert_eq!(
        streamed_nway_right.rows.clone().into_boxed(),
        resident_nway_right.rows.clone().into_boxed(),
        "a RIGHT step globally completes after every accumulated chunk/block"
    );
    assert_eq!(
        streamed_nway_two_right.rows.clone().into_boxed(),
        resident_nway_two_right.rows.clone().into_boxed(),
        "recursive prefix replay preserves an earlier right complement through a later FULL step"
    );
    let sort_outer_rows = |result: &RelationalSelectResult| {
        let mut rows = result.rows.clone().into_boxed();
        rows.sort_by(|a, b| {
            crate::rel_exec_helpers::compare_sql_values(&a[0], &b[0])
                .then_with(|| crate::rel_exec_helpers::compare_sql_values(&a[2], &b[2]))
        });
        rows
    };
    assert_eq!(
        sort_outer_rows(&streamed_outer),
        sort_outer_rows(&resident_outer),
        "global unmatched-coordinate completion matches resident FULL OUTER GPU join"
    );
    assert_eq!(
        sort_outer_rows(&streamed_outer_where),
        sort_outer_rows(&resident_outer_where),
        "OUTER WHERE membership is post-join while unmatched membership remains ON-only"
    );
    assert_eq!(
        streamed_outer_ordered
            .rows
            .iter()
            .map(|row| row[3].clone())
            .collect::<Vec<_>>(),
        resident_outer_ordered
            .rows
            .iter()
            .map(|row| row[3].clone())
            .collect::<Vec<_>>(),
        "outer unmatched tails participate in bounded top-N compaction"
    );
    assert!(streamed_outer_ordered
        .rows
        .iter()
        .all(|row| { resident_outer.rows.iter().any(|candidate| candidate == row) }));
    for row in streamed_outer_limit.rows.iter() {
        assert!(
            resident_outer.rows.iter().any(|candidate| candidate == row),
            "unordered early-window output must be a valid FULL OUTER row: {row:?}"
        );
    }
    assert!(
        streamed_rows.iter().any(|row| row[3] == SqlValue::Null),
        "projected NULL data survives the streaming join"
    );
    assert!(
        streamed_rows
            .iter()
            .all(|row| row[0] != SqlValue::Null && row[1] == row[2]),
        "NULL keys never match and every emitted key/value pair is exact"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_join_mixed_int4_int8_keys() {
    let _entry_disabled = ClassEntryDisabled::new();
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE mix4 (k INT, v INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE mix8 (k BIGINT, v INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO mix4 VALUES (-1, 10), (2, 20)")
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO mix8 VALUES (-1, 100), (2, 200), (2147483648, 999)",
    )
    .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let result = e
        .execute_resident_expr_select_sql(
            "SELECT a.v, b.v FROM mix4 a JOIN mix8 b ON a.k = b.k ORDER BY a.v",
        )
        .expect("mixed-width streaming join");
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(10), SqlValue::Int4(100)],
            vec![SqlValue::Int4(20), SqlValue::Int4(200)],
        ]
    );
}

/// Disable chunk-class ENTRY for a store-driven gate test (6c-1/6c-3/P2 machinery), restoring
/// on drop (the GPU suite is --test-threads=1, so set/restore is race-free).
struct ClassEntryDisabled;
impl ClassEntryDisabled {
    fn new() -> Self {
        crate::engine_streaming_exec::CHUNK_CLASS_ENTRY_ENABLED_TEST
            .store(false, std::sync::atomic::Ordering::Relaxed);
        ClassEntryDisabled
    }
}
impl Drop for ClassEntryDisabled {
    fn drop(&mut self) {
        crate::engine_streaming_exec::CHUNK_CLASS_ENTRY_ENABLED_TEST
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Force the retained exact key-index set over-cap so small GPU fixtures exercise the compact
/// all-chunk Bloom route. The ignored GPU suite is run with `--test-threads=1`.
struct ChunkKeyIndexCapOverride;
impl ChunkKeyIndexCapOverride {
    fn tiny() -> Self {
        crate::engine_streaming_exec::CHUNK_KEY_INDEX_CAP_BYTES_TEST
            .store(1, std::sync::atomic::Ordering::Relaxed);
        crate::engine_streaming_exec::CHUNK_KEY_BLOOM_ALL_POSITIVE_TEST
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}
impl Drop for ChunkKeyIndexCapOverride {
    fn drop(&mut self) {
        crate::engine_streaming_exec::CHUNK_KEY_INDEX_CAP_BYTES_TEST
            .store(0, std::sync::atomic::Ordering::Relaxed);
        crate::engine_streaming_exec::CHUNK_KEY_BLOOM_CAP_BYTES_TEST
            .store(0, std::sync::atomic::Ordering::Relaxed);
        crate::engine_streaming_exec::CHUNK_KEY_BLOOM_ALL_POSITIVE_TEST
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// P5-0 — THE DEVICE SLOT RECHECK differential: for every slot of a staged mixed-type
/// NULL-bearing chunk, the single-slot device materialization must equal the P4-1 host
/// decoder's row exactly, and the sidecar/born masks must agree (a stamped slot returns
/// Some(None) at-or-above its stamp and the live row below it).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_device_slot_recheck_matches_host_decoder() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE mix (a INT, s SMALLINT, big BIGINT, d DATE, ts TIMESTAMP, \
         n NUMERIC(10,2), flag BOOL, t TEXT, u UUID)",
    )
    .unwrap();
    const N: i32 = 300;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let t = if i % 7 == 0 {
            "NULL".into()
        } else {
            format!("'txt{:04}'", i)
        };
        let n = if i % 5 == 0 {
            "NULL".into()
        } else {
            format!("{}.{:02}", i, i % 100)
        };
        let big = if i % 11 == 0 {
            "NULL".into()
        } else {
            format!("{}", i64::from(i) * 999_983)
        };
        let flag = if i % 17 == 0 {
            "NULL".into()
        } else if i % 2 == 0 {
            "true".into()
        } else {
            "false".to_string()
        };
        let u = if i % 19 == 0 {
            "NULL".into()
        } else {
            format!("'00000000-0000-0000-0000-{:012x}'", i)
        };
        values.push_str(&format!(
            "({i}, {}, {big}, '2024-{:02}-{:02}', '2024-01-01 00:{:02}:{:02}', {n}, {flag}, {t}, {u})",
            if i % 13 == 0 { "NULL".to_string() } else { format!("{}", i % 300 - 150) },
            1 + (i % 12),
            1 + (i % 28),
            i % 60,
            (i * 7) % 60,
        ));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO mix VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM mix"))
        .unwrap();
    // Stamp one row so the mask path is exercised (class or store-driven — either stamps).
    seq += 1;
    e.execute_text(seq, "DELETE FROM mix WHERE a = 42").unwrap();
    let rtx = e.committed_seq();

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("mix")
        .cloned()
        .unwrap();
    let map = e.read_state.residency.streaming_cold_chunks.load();
    let entry = map.get("mix").expect("cold entry");
    let mut checked = 0usize;
    for chunk in &entry.chunks {
        // Stage once per chunk; final-readback every slot against the construction oracle.
        let (staged, _vis) = e
            .stage_cold_chunk(chunk, rtx)
            .expect("stage")
            .ready()
            .expect("ready");
        let host_unmasked =
            crate::engine_streaming_exec::decode_cold_chunk_rows(&table, chunk, 0).unwrap();
        for (slot, expected) in host_unmasked.iter().enumerate() {
            let got = e
                .read_cold_chunk_slot_values(&table, chunk, &staged, slot)
                .expect("no decline");
            assert_eq!(&got, expected, "slot {slot} value mismatch");
            checked += 1;
        }
    }
    assert_eq!(checked, N as usize, "every slot rechecked");
    // Visibility is a DEVICE decision: the deleted key no longer locates, while an adjacent
    // live key does. No host sidecar inspection participates in the oracle.
    let dead = crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
        &table,
        &[vec![(0, SelectFilterOp::Eq, SqlValue::Int4(42))]],
    )
    .unwrap();
    assert!(
        e.locate_streaming_cold_slots(&table, &dead, rtx)
            .unwrap()
            .is_empty(),
        "the device visibility mask excludes the stamped row"
    );
    let live = crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
        &table,
        &[vec![(0, SelectFilterOp::Eq, SqlValue::Int4(41))]],
    )
    .unwrap();
    assert_eq!(
        e.locate_streaming_cold_slots(&table, &live, rtx)
            .unwrap()
            .iter()
            .map(|(_, slots)| slots.len())
            .sum::<usize>(),
        1,
        "the neighboring live row survives on-device"
    );
}

/// P5-1 — THE CHUNK KEY-INDEX CACHE: build per-chunk device hash indexes over a key column,
/// probe present/absent needles in ONE multi-chunk launch, recheck each hit's value via the
/// P5-0 device slot read, and verify the all-visible-index + recheck-mask contract (a stamped
/// row still HITS the index; the recheck masks it — the P5-2 uniqueness semantics).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_key_index_builds_probes_and_rechecks() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT)")
        .unwrap();
    const N: i32 = 900;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO facts (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM facts"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);
    // A stamped delete: the index is ALL-VISIBLE, so the probe must still hit and the P5-0
    // recheck must mask.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 7")
        .unwrap();
    let rtx = e.committed_seq();

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("facts")
        .cloned()
        .unwrap();
    let entry = e
        .read_state
        .residency
        .streaming_cold_chunks
        .load()
        .get("facts")
        .cloned()
        .unwrap();
    let indexes = e
        .ensure_chunk_key_indexes(&table, &entry, &[0], 0)
        .expect("indexes build");
    assert!(!indexes.is_empty());
    assert!(
        e.read_state
            .residency
            .chunk_key_index_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0,
        "the retained buffers are accounted"
    );

    // Needles: three present (5, 500, 100000 — the tail row), the stamped one (7), one absent.
    let needles: Vec<i32> = vec![5, 500, 100000, 7, 424242];
    let hits = e
        .probe_chunk_key_indexes(&indexes, &needles)
        .expect("probe");
    assert_eq!(hits.len(), 5);
    // Present keys: the raw index hits, and the exact DEVICE predicate yields one live slot.
    for (n, expect_a) in [(0usize, 5i32), (1, 500), (2, 100000)] {
        assert!(!hits[n].is_empty(), "needle {n}: raw index hit");
        let predicate = crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
            &table,
            &[vec![(0, SelectFilterOp::Eq, SqlValue::Int4(expect_a))]],
        )
        .unwrap();
        let exact = e
            .locate_streaming_cold_slots(&table, &predicate, rtx)
            .unwrap();
        assert_eq!(
            exact.iter().map(|(_, slots)| slots.len()).sum::<usize>(),
            1,
            "needle {n}: exactly one device-approved hit"
        );
    }
    // The STAMPED key: the index hits, the recheck masks — no live hit (the P5-2 not-a-conflict).
    assert!(
        !hits[3].is_empty(),
        "the all-visible index still hits the stamped key"
    );
    let stamped = crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
        &table,
        &[vec![(0, SelectFilterOp::Eq, SqlValue::Int4(7))]],
    )
    .unwrap();
    assert!(
        e.locate_streaming_cold_slots(&table, &stamped, rtx)
            .unwrap()
            .is_empty(),
        "the stamped hit is masked by the device recheck"
    );
    // The absent key: no hits at all.
    assert!(hits[4].is_empty(), "an absent key misses every chunk index");

    // THE FOLD PATH (audit HIGH regression: per-column-parallel blob_offsets — a single int8 key
    // folds on-device; the needle is the host fingerprint via the shared helper): build indexes
    // over a BIGINT column and probe present/absent keys through fingerprints.
    e.clear_relational_residency_budget_bytes(0);
    seq += 1;
    e.execute_text(seq, "CREATE TABLE keyed8 (k BIGINT, v INT)")
        .unwrap();
    let mut v8 = String::new();
    for i in 0..600i64 {
        if i > 0 {
            v8.push(',');
        }
        v8.push_str(&format!("({}, {})", i * 1_000_000_007, i));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO keyed8 (k, v) VALUES {v8}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM keyed8"))
        .unwrap();
    let table8 = e
        .catalog_snapshot()
        .relational_catalog
        .get("keyed8")
        .cloned()
        .unwrap();
    let entry8 = e
        .read_state
        .residency
        .streaming_cold_chunks
        .load()
        .get("keyed8")
        .cloned()
        .expect("keyed8 cold entry");
    let indexes8 = e
        .ensure_chunk_key_indexes(&table8, &entry8, &[0], 0)
        .expect("int8-key indexes build (the fold path)");
    let present = Engine::chunk_key_needle(
        &table8,
        &[0],
        &[SqlValue::Int8(5 * 1_000_000_007), SqlValue::Int4(5)],
    )
    .expect("needle");
    let absent = Engine::chunk_key_needle(
        &table8,
        &[0],
        &[SqlValue::Int8(999_999_999_999), SqlValue::Int4(0)],
    )
    .expect("needle");
    let hits8 = e
        .probe_chunk_key_indexes(&indexes8, &[present, absent])
        .expect("probe");
    assert!(!hits8[0].is_empty(), "the present folded fingerprint hits");
    let rtx8 = e.committed_seq();
    let present_predicate = crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
        &table8,
        &[vec![(
            0,
            SelectFilterOp::Eq,
            SqlValue::Int8(5 * 1_000_000_007),
        )]],
    )
    .unwrap();
    assert_eq!(
        e.locate_streaming_cold_slots(&table8, &present_predicate, rtx8)
            .unwrap()
            .iter()
            .map(|(_, slots)| slots.len())
            .sum::<usize>(),
        1,
        "the folded int8 key locates its exact row on-device"
    );
    let absent_predicate = crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(
        &table8,
        &[vec![(
            0,
            SelectFilterOp::Eq,
            SqlValue::Int8(999_999_999_999),
        )]],
    )
    .unwrap();
    assert!(
        e.locate_streaming_cold_slots(&table8, &absent_predicate, rtx8)
            .unwrap()
            .is_empty(),
        "an absent fingerprint collision cannot survive the exact device predicate"
    );
}

/// P5-2 — THE KEYED-CLASS LIFT (INSERT): a PK'd over-budget table ENTERS the class (eligibility
/// no longer refuses unique indexes); its host rows are RECLAIMED; INSERT uniqueness is then
/// validated ON-DEVICE (per-chunk key-index probe + P5-0 slot recheck at the statement
/// snapshot): a genuine dup rejects WITHOUT de-auth, an in-batch dup rejects on-device, a
/// tombstoned key re-inserts (a masked hit is NOT a conflict), and fresh keys append as tails.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_lift_insert_uniqueness() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE ku (a INT PRIMARY KEY, t TEXT)")
        .unwrap();
    const N: i32 = 1200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO ku (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM ku");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    let _ = count(&e);
    seq += 1;
    e.execute_text(seq, "INSERT INTO ku (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(
        e.chunk_class_entries(),
        1,
        "the KEYED table must ENTER the class (the P5-2 lift)"
    );
    assert!(
        e.read_state
            .mvcc
            .table_rows("ku")
            .store()
            .all_versions()
            .is_empty(),
        "entry reclaims the host rows"
    );

    // A genuine duplicate vs a BASE chunk: rejected on-device, class INTACT.
    let exact_before = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO ku (a, t) VALUES (500, 'dup')")
        .expect_err("dup key 500 must reject");
    assert!(
        format!("{err:?}").contains("duplicate key value"),
        "unique violation, got {err:?}"
    );
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "a dup rejection must NOT de-auth"
    );
    assert!(
        e.chunk_class_unique_probe_conflicts() >= 1,
        "the conflict came from the device probe's recheck"
    );
    assert!(
        e.chunk_class_device_exact_rechecks() > exact_before,
        "the exact key verdict ran through the device predicate VM"
    );

    // A duplicate vs a TAIL chunk (the enter row): the tail's index builds lazily and probes.
    let exact_before = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO ku (a, t) VALUES (100000, 'dup-tail')")
        .expect_err("dup key 100000 must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);
    assert!(
        e.chunk_class_device_exact_rechecks() > exact_before,
        "the tail-key exact equality ran on-device"
    );

    // An IN-BATCH duplicate: exact comparison over the transient device relation.
    let exact_before = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let err = e
        .execute_text(
            seq,
            "INSERT INTO ku (a, t) VALUES (777001, 'x'), (777001, 'y')",
        )
        .expect_err("in-batch dup must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);
    assert!(
        e.chunk_class_device_exact_rechecks() > exact_before,
        "within-statement exact equality ran on-device"
    );
    assert_eq!(count(&e), i64::from(N) + 1, "no rejected row ever landed");

    // Fresh keys append as tails; the probe VALIDATED (non-vacuity) and the class held.
    let probes_before = e.chunk_class_unique_probes();
    let skipped_before = e.chunk_class_device_commits();
    seq += 1;
    e.execute_text(seq, "INSERT INTO ku (a, t) VALUES (600000, 'fresh')")
        .unwrap();
    assert!(
        e.chunk_class_unique_probes() > probes_before,
        "the accept path went through the device probe"
    );
    assert!(
        e.chunk_class_device_commits() > skipped_before,
        "still classed"
    );
    assert_eq!(count(&e), i64::from(N) + 2);

    // Tombstone-then-reinsert: the probe HITS the dead slot, the recheck masks it at the
    // statement snapshot — NOT a conflict.
    seq += 1;
    e.execute_text(seq, "DELETE FROM ku WHERE a = 500").unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "class DELETE stays classed");
    seq += 1;
    e.execute_text(seq, "INSERT INTO ku (a, t) VALUES (500, 'reborn')")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "a masked hit is NOT a conflict");
    assert_eq!(count(&e), i64::from(N) + 2);
}

/// Wider class eligibility: CHECK remains row-local, while non-self outbound and inbound FK
/// validation probes chunk-authoritative parent/child rows with exact device predicates. Genuine
/// violations reject without de-authorizing; deleting the child then its provider succeeds.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_check_and_foreign_keys_stay_device_native() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE cp (id INT PRIMARY KEY, note TEXT, CHECK (id > 0))",
    )
    .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE cc (id INT PRIMARY KEY, parent_id INT, v INT, note TEXT, \
                          CONSTRAINT cc_v_pos CHECK (v > 0))",
    )
    .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "ALTER TABLE ONLY cc ADD CONSTRAINT cc_parent_fk FOREIGN KEY (parent_id) \
         REFERENCES cp(id)",
    )
    .unwrap();
    let parents = (1..=1200)
        .map(|id| format!("({id}, 'p{id:04}')"))
        .collect::<Vec<_>>()
        .join(",");
    let children = (1..=1200)
        .map(|id| format!("({id}, {id}, 1, 'c{id:04}')"))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO cp VALUES {parents}"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO cc VALUES {children}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM cp"))
        .unwrap();
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM cc"))
        .unwrap();

    seq += 1;
    e.execute_text(seq, "INSERT INTO cp VALUES (100000, 'enter')")
        .unwrap();
    assert!(e.table_chunk_authoritative("cp").is_some());
    let exact_before = e.chunk_class_device_exact_rechecks();
    seq += 1;
    e.execute_text(seq, "INSERT INTO cc VALUES (100000, 100000, 1, 'enter')")
        .unwrap();
    assert!(e.table_chunk_authoritative("cc").is_some());
    assert!(
        e.chunk_class_device_exact_rechecks() > exact_before,
        "child entry validated its provider against parent chunks"
    );

    seq += 1;
    let fk_error = e
        .execute_text(seq, "INSERT INTO cc VALUES (100001, 999999, 1, 'bad-fk')")
        .expect_err("missing provider must reject");
    assert!(format!("{fk_error:?}").contains("foreign key constraint"));
    seq += 1;
    let check_error = e
        .execute_text(
            seq,
            "INSERT INTO cc VALUES (100002, 100000, -1, 'bad-check')",
        )
        .expect_err("CHECK violation must reject");
    assert!(format!("{check_error:?}").contains("check constraint"));
    assert_eq!(e.chunk_class_deauths(), 0);

    seq += 1;
    e.execute_text(seq, "DELETE FROM cc WHERE id = 100000")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM cp WHERE id = 100000")
        .unwrap();

    // Use a distinct pair for the rejected inbound check so the successful child->provider delete
    // sequence above also proves both classed DELETE arms independently.
    seq += 1;
    e.execute_text(seq, "INSERT INTO cp VALUES (100003, 'guarded')")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO cc VALUES (100003, 100003, 1, 'guarded')")
        .unwrap();
    let exact_before = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let wal_before_reject = e.durable_wal_records().len();
    let inbound_error = e
        .execute_text(seq, "DELETE FROM cp WHERE id = 100003")
        .expect_err("referenced provider delete must reject");
    assert!(format!("{inbound_error:?}").contains("foreign key constraint"));
    assert!(e.chunk_class_device_exact_rechecks() > exact_before);
    assert_eq!(
        e.durable_wal_records().len(),
        wal_before_reject,
        "the class inbound-FK rejection must happen before WAL"
    );
    assert!(!e.is_commit_path_poisoned());
    seq += 1;
    e.execute_text(seq, "INSERT INTO cp VALUES (100004, 'after-reject')")
        .expect("a pre-WAL constraint rejection must not wedge later writes");
    let provider_after_reject = e
        .execute_relational_select(&select("SELECT id FROM cp WHERE id = 100003"))
        .unwrap();
    assert_eq!(
        provider_after_reject.rows.len(),
        1,
        "a rejected parent DELETE must not hide its provider"
    );
    assert_eq!(e.chunk_class_deauths(), 0);
    assert!(e.table_chunk_authoritative("cp").is_some());
    assert!(e.table_chunk_authoritative("cc").is_some());

    // Facade transaction identities are not MVCC indices. Rebuild from WAL, re-enter both cold
    // classes, then use deliberately low/reused facade ids: the parent must still resolve at the
    // recovered committed boundary and reject before adding a durable record.
    let durable = e.durable_wal_records();
    let mut recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    recovered.set_relational_residency_budget_bytes(0, 8192);
    let _ = recovered
        .execute_relational_select(&select("SELECT COUNT(*) FROM cp"))
        .unwrap();
    let _ = recovered
        .execute_relational_select(&select("SELECT COUNT(*) FROM cc"))
        .unwrap();
    let _ = recovered
        .execute_relational_select(&select("SELECT SUM(id) FROM cp"))
        .unwrap();
    let _ = recovered
        .execute_relational_select(&select("SELECT SUM(id) FROM cc"))
        .unwrap();
    recovered
        .execute_text(10_001, "INSERT INTO cp VALUES (100005, 'recovered')")
        .unwrap();
    recovered
        .execute_text(
            10_002,
            "INSERT INTO cc VALUES (100005, 100005, 1, 'recovered')",
        )
        .unwrap();
    assert!(recovered.table_chunk_authoritative("cp").is_some());
    assert!(recovered.table_chunk_authoritative("cc").is_some());
    let recovered_wal_before = recovered.durable_wal_records().len();
    let recovered_error = recovered
        .execute_text(1, "DELETE FROM cp WHERE id = 100005")
        .expect_err("decoupled facade id must not hide the current provider");
    assert!(
        format!("{recovered_error:?}").contains("foreign key constraint"),
        "unexpected recovered FK error: {recovered_error:?}"
    );
    assert_eq!(recovered.durable_wal_records().len(), recovered_wal_before);
    assert!(!recovered.is_commit_path_poisoned());

    // Deterministic TOCTOU: pause after the parent's provisional class preflight and immediately
    // before commit_mutex acquisition, commit a child in the gap, then resume. The definitive
    // check under commit_mutex must see the child and reject the parent DELETE without appending
    // its WAL record.
    recovered
        .execute_text(10_003, "INSERT INTO cp VALUES (100006, 'racing')")
        .unwrap();
    let recovered = std::sync::Arc::new(recovered);
    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    recovered.set_commit_prelock_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let deleting = std::sync::Arc::clone(&recovered);
    let delete = std::thread::spawn(move || {
        deleting.execute_text(10_004, "DELETE FROM cp WHERE id = 100006")
    });
    reached.wait();
    let race_wal_before = recovered.durable_wal_records().len();
    recovered
        .execute_text(
            10_005,
            "INSERT INTO cc VALUES (100006, 100006, 1, 'racing')",
        )
        .unwrap();
    resume.wait();
    let race_error = delete
        .join()
        .expect("parent delete thread")
        .expect_err("child committed in the preflight gap must block the parent delete");
    assert!(format!("{race_error:?}").contains("foreign key constraint"));
    assert_eq!(
        recovered.durable_wal_records().len(),
        race_wal_before + 1,
        "only the racing child INSERT may reach WAL"
    );
    assert!(!recovered.is_commit_path_poisoned());
    recovered
        .execute_text(10_006, "INSERT INTO cp VALUES (100007, 'after-race')")
        .expect("the raced pre-WAL rejection must not wedge later writes");
}

/// P5-later — OVER-CAP KEYED CLASS: when the complete retained exact-index set cannot co-reside,
/// compact per-chunk Bloom filters still admit the class. The GPU Bloom probe only chooses
/// chunks; exact predicate + visibility remains authoritative for duplicate checks and point DML.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_over_cap_bloom_candidates_stay_exact() {
    let _forced_over_cap = ChunkKeyIndexCapOverride::tiny();
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE kb (a INT PRIMARY KEY, u INT UNIQUE, v INT)",
    )
    .unwrap();
    const N: i32 = 1200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {}, {})", i + 50_000, i * 10));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kb (a, u, v) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kb"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (100000, 150000, -1)")
        .unwrap();
    assert_eq!(
        e.chunk_class_entries(),
        1,
        "over-cap keyed table enters via Bloom set"
    );
    assert!(e.table_chunk_authoritative("kb").is_some());
    assert!(
        e.read_state
            .mvcc
            .table_rows("kb")
            .store()
            .all_versions()
            .is_empty(),
        "host row chains were reclaimed"
    );

    let bloom_0 = e.chunk_key_bloom_probes();
    let exact_0 = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (1100, 999999, 1)")
        .expect_err("base-chunk duplicate must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert!(
        e.chunk_key_bloom_probes() > bloom_0,
        "candidate decision ran on GPU Bloom"
    );
    assert!(
        e.chunk_class_device_exact_rechecks() > exact_0,
        "conflict was exactly rechecked"
    );
    assert_eq!(e.chunk_class_deauths(), 0);

    // The entry-triggering row is a later tail chunk. Its Bloom is built on demand and must not
    // be omitted merely because it was not part of the original base capture.
    let bloom_1 = e.chunk_key_bloom_probes();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (100000, 777777, 2)")
        .expect_err("tail duplicate must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert!(e.chunk_key_bloom_probes() > bloom_1);

    // PostgreSQL UNIQUE treats every NULL-bearing key as distinct, including in the chunk class.
    seq += 1;
    e.execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (200000, NULL, 3)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (200001, NULL, 4)")
        .expect("a second NULL unique key remains distinct");
    assert_eq!(e.chunk_class_deauths(), 0);

    // Every test Bloom is deliberately ALL-POSITIVE, so the absent key is a guaranteed false
    // positive in every chunk. Point UPDATE and the miss must still resolve exactly on-device;
    // no host-store restoration or de-authorization is allowed.
    let bloom_2 = e.chunk_key_bloom_probes();
    let locates_2 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "UPDATE kb SET v = 4242 WHERE a = 1100")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kb WHERE a = 987654")
        .unwrap();
    assert!(e.chunk_key_bloom_probes() >= bloom_2 + 2);
    assert!(e.chunk_class_dml_key_locates() >= locates_2 + 2);
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "Bloom route stays chunk-authoritative"
    );
    assert!(e.table_chunk_authoritative("kb").is_some());
    let bloom_ids_before: std::collections::BTreeSet<u64> = e
        .read_state
        .residency
        .chunk_key_bloom
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .keys()
        .filter_map(|(table, chunk_id, _)| (table == "kb").then_some(*chunk_id))
        .collect();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kb WHERE a < 400").unwrap();
    e.maybe_compact_chunk_class("kb");
    assert!(
        e.chunk_class_compactions() > 0,
        "range delete compacts a keyed chunk"
    );
    let bloom_ids_after: std::collections::BTreeSet<u64> = e
        .read_state
        .residency
        .chunk_key_bloom
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .keys()
        .filter_map(|(table, chunk_id, _)| (table == "kb").then_some(*chunk_id))
        .collect();
    assert!(
        bloom_ids_before
            .difference(&bloom_ids_after)
            .next()
            .is_some(),
        "compaction publication purges the replaced chunk-id Bloom"
    );
    let row = e
        .execute_relational_select(&select("SELECT v FROM kb WHERE a = 1100"))
        .unwrap();
    assert_eq!(
        row.rows.clone().into_boxed(),
        vec![vec![SqlValue::Int4(4242)]]
    );
}

/// The Bloom cap is global. A second table that cannot reserve a complete set must roll back every
/// partial buffer, leave the already-authoritative first table intact, and refuse class entry.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_bloom_global_cap_rolls_back_failed_admission() {
    let _forced_over_cap = ChunkKeyIndexCapOverride::tiny();
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let rows = |base: i32| {
        (0..800)
            .map(|i| format!("({}, {})", base + i, i))
            .collect::<Vec<_>>()
            .join(",")
    };
    seq += 1;
    e.execute_text(seq, "CREATE TABLE bca (a INT PRIMARY KEY, v INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE bcb (a INT PRIMARY KEY, v INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO bca (a, v) VALUES {}", rows(0)))
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        &format!("INSERT INTO bcb (a, v) VALUES {}", rows(200000)),
    )
    .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM bca"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO bca (a, v) VALUES (100000, 1)")
        .unwrap();
    assert!(e.table_chunk_authoritative("bca").is_some());
    let first_bytes = e.chunk_key_bloom_bytes();
    assert!(first_bytes > 0);
    let forced_cap = first_bytes + first_bytes / 2;
    crate::engine_streaming_exec::CHUNK_KEY_BLOOM_CAP_BYTES_TEST
        .store(forced_cap, std::sync::atomic::Ordering::Relaxed);

    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM bcb"))
        .unwrap();
    assert!(
        e.chunk_key_bloom_bytes() <= first_bytes,
        "failed priming cannot retain any additional VRAM"
    );
    assert!(
        !e.read_state
            .residency
            .chunk_key_bloom
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .any(|(table, _, _)| table == "bcb"),
        "failed priming must roll back every partial bcb Bloom"
    );
    assert!(e.table_chunk_authoritative("bcb").is_none());
    assert!(e.table_chunk_authoritative("bca").is_some());
    assert!(e.chunk_key_bloom_bytes() <= forced_cap);
}

/// Spill-backed keyed captures prime their Bloom set only after the capture installer releases the
/// commit mutex. The later entry may build the fresh RAM tail, but must not reread old spill chunks.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_bloom_spill_is_primed_before_class_entry() {
    let _forced_over_cap = ChunkKeyIndexCapOverride::tiny();
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(1024, std::sync::atomic::Ordering::Relaxed);
    let result = std::panic::catch_unwind(|| {
        let mut e = Engine::new_local_test_engine();
        let mut seq = 0_u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE kbs (a INT PRIMARY KEY, v INT)")
            .unwrap();
        let values = (0..1200)
            .map(|i| format!("({i}, {})", i * 10))
            .collect::<Vec<_>>()
            .join(",");
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO kbs (a, v) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 4096);
        let _ = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kbs"))
            .unwrap();
        assert!(
            e.streaming_cold_spills() > 0,
            "fixture must be spill-backed"
        );
        assert!(e.chunk_key_bloom_bytes() > 0, "Bloom set primed off-lock");
        seq += 1;
        e.execute_text(seq, "INSERT INTO kbs (a, v) VALUES (100000, -1)")
            .unwrap();
        assert!(e.table_chunk_authoritative("kbs").is_some());
        assert_eq!(e.chunk_class_deauths(), 0);
    });
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(0, std::sync::atomic::Ordering::Relaxed);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

/// Audit M2 — the temporary exact-predicate batch bound is fail-closed, not a host fallback.
/// Exactly 256 fresh rows stay classed and are device-validated; 257 fresh rows are rejected
/// without deauthorizing or landing a partial statement.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_unique_batch_bound_fails_closed_at_257() {
    fn populate_and_enter(e: &mut Engine, seq: &mut u64, table: &str) {
        let entries_before = e.chunk_class_entries();
        e.clear_relational_residency_budget_bytes(0);
        *seq += 1;
        e.execute_text(*seq, &format!("CREATE TABLE {table} (a INT PRIMARY KEY)"))
            .unwrap();
        let values = (0..600)
            .map(|i| format!("({i})"))
            .collect::<Vec<_>>()
            .join(",");
        *seq += 1;
        e.execute_text(*seq, &format!("INSERT INTO {table} VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 8192);
        let _ = e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        assert!(
            e.table_chunk_authoritative(table).is_some(),
            "{table} must class immediately after its forced streaming capture"
        );
        *seq += 1;
        e.execute_text(*seq, &format!("INSERT INTO {table} VALUES (100000)"))
            .unwrap();
        assert_eq!(
            e.chunk_class_entries(),
            entries_before + 1,
            "premise: {table} entered the class"
        );
    }

    fn count(e: &Engine, table: &str) -> i64 {
        match e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(value) => value,
            ref other => panic!("count: {other:?}"),
        }
    }

    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }

    populate_and_enter(&mut e, &mut seq, "bound_ok");
    let deauth_before = e.chunk_class_deauths();
    let probes_before = e.chunk_class_unique_probes();
    let values_256 = (0..256)
        .map(|i| format!("({})", 200000 + i))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO bound_ok VALUES {values_256}"))
        .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        deauth_before,
        "the inclusive 256-row bound remains device-authoritative"
    );
    assert!(e.chunk_class_unique_probes() > probes_before);

    let values_257 = (0..257)
        .map(|i| format!("({})", 300000 + i))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    let err = e
        .execute_text(seq, &format!("INSERT INTO bound_ok VALUES {values_257}"))
        .expect_err("257 rows exceed the bounded exact device validation");
    assert!(format!("{err:?}").contains("device exact unique batch limit"));
    assert_eq!(
        e.chunk_class_deauths(),
        deauth_before,
        "the oversized batch must not cross into host authority"
    );
    assert_eq!(count(&e, "bound_ok"), 600 + 1 + 256);
    seq += 1;
    e.execute_text(seq, "DROP TABLE bound_ok").unwrap();

    populate_and_enter(&mut e, &mut seq, "bound_dup");
    let deauth_before = e.chunk_class_deauths();
    let mut duplicate_values = (0..256)
        .map(|i| format!("({})", 400000 + i))
        .collect::<Vec<_>>();
    duplicate_values.push("(400000)".to_owned());
    seq += 1;
    let err = e
        .execute_text(
            seq,
            &format!(
                "INSERT INTO bound_dup VALUES {}",
                duplicate_values.join(",")
            ),
        )
        .expect_err("the oversized batch must fail before any host duplicate validation");
    assert!(format!("{err:?}").contains("device exact unique batch limit"));
    assert_eq!(
        e.chunk_class_deauths(),
        deauth_before,
        "the rejecting oversized batch remains device-authoritative"
    );
    assert_eq!(count(&e, "bound_dup"), 601, "no rejected row landed");
}

/// P5-2 — C1 SELF-EXCLUSION: an UPDATE's own located coordinates are SELF, not conflicts (the
/// old versions are live at probe time — stamps land in the commit hook). Key-preserving
/// multi-row updates pass; a key change INTO an existing key rejects; a key change to a fresh
/// key frees the old key for re-insert.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_update_self_exclusion() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kv (a INT PRIMARY KEY, t TEXT)")
        .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kv (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kv"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kv (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // Key-preserving multi-row UPDATE: every new image's key HITS its own located slot — all
    // self-excluded, zero conflicts, class intact.
    let conflicts_before = e.chunk_class_unique_probe_conflicts();
    seq += 1;
    e.execute_text(seq, "UPDATE kv SET t = 'self' WHERE a < 50")
        .unwrap();
    assert_eq!(
        e.chunk_class_unique_probe_conflicts(),
        conflicts_before,
        "C1: self-hits are NOT conflicts"
    );
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "key-preserving UPDATE stays classed"
    );

    // A key change INTO an existing key: a genuine conflict (the hit is NOT self).
    seq += 1;
    let err = e
        .execute_text(seq, "UPDATE kv SET a = 43 WHERE a = 44")
        .expect_err("44 -> 43 collides with the live 43");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);

    // A key change to a FRESH key: passes; the old key is then free for re-insert and the new
    // key is taken.
    seq += 1;
    e.execute_text(seq, "UPDATE kv SET a = 999999 WHERE a = 45")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kv (a, t) VALUES (45, 'reused')")
        .unwrap();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kv (a, t) VALUES (999999, 'taken')")
        .expect_err("the moved-to key is live");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0, "the whole arc stayed classed");
}

/// P5-2 — FOLD-PATH NEEDLE PARITY + FINGERPRINT COLLISION: a COMPOUND key (two int4 columns)
/// builds its chunk indexes over device-folded fingerprints and probes with the HOST-derived
/// twin (`chunk_key_needle`) — a derivation mismatch is a silent all-miss (dup accepted), so the
/// dup rejection here IS the parity proof. Then the adversarial case: two DISTINCT keys with
/// COLLIDING 32-bit fingerprints — the colliding insert must be ACCEPTED (the full-tuple
/// recheck distinguishes), the true dup still rejects.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_compound_fold_parity_and_collision() {
    // Host-side birthday search for a fingerprint collision on (5, j): two j values whose key
    // word-vectors [w(5), w(j)] fold to the SAME fingerprint.
    let word = |v: i32| {
        crate::engine_residency::sql_value_key_words(SqlType::Int4, &SqlValue::Int4(v)).unwrap()
    };
    let fp = |a: i32, b: i32| {
        let mut words = word(a);
        words.extend(word(b));
        crate::engine_residency::compound_key_fingerprint(&words)
    };
    // CONSTRUCT the collision (a birthday search cannot find one here: the fingerprint's final
    // per-word round is a BIJECTION of the last word, so fp(a1,b1) == fp(a2,b2) reduces to
    // h1(a1) ^ h1(a2) == b1 ^ b2 — vanishingly rare over a small grid). Bucket first-word
    // states by their TOP 12 BITS; two same-bucket states differ by x < 2^20, and b2 = b1 ^ x
    // completes the pair. `step` replicates the fingerprint's per-word round FOR THE SEARCH
    // ONLY — the REAL `compound_key_fingerprint` verifies the constructed pair below (drift in
    // the round fails that assert loudly, never a silent mis-gate).
    let step = |h: u32, w: i32| -> u32 {
        let h = (h ^ (w as u32)).wrapping_mul(0x0100_0193);
        h.rotate_left(13).wrapping_add(0x9E37_79B1)
    };
    let mut buckets: std::collections::HashMap<u32, (i32, u32)> = std::collections::HashMap::new();
    let mut found: Option<((i32, i32), (i32, i32))> = None;
    // Outside the filler key space (filler a < 1000, b = 3i < 3000; skip the enter row's
    // a = 100000; b values sit at 2^20 +- x, far above every filler b).
    let mut a: i32 = 10_000;
    while found.is_none() {
        assert!(a < 2_000_000, "no same-bucket first-word pair found");
        if a != 100_000 {
            let h1 = step(0x811C_9DC5, word(a)[0]);
            if let Some((a_prev, h_prev)) = buckets.insert(h1 >> 20, (a, h1)) {
                let x = (h_prev ^ h1) as i32;
                let b1 = 1_i32 << 20;
                found = Some(((a_prev, b1), (a, b1 ^ x)));
            }
        }
        a += 1;
    }
    let ((a1, b1), (a2, b2)) = found.unwrap();
    assert_eq!(
        fp(a1, b1),
        fp(a2, b2),
        "the constructed pair must collide under the REAL fingerprint"
    );
    assert!((a1, b1) != (a2, b2), "distinct tuples");

    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE kc (a INT, b INT, t TEXT, PRIMARY KEY (a, b))",
    )
    .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {}, 'txt{:04}')", i * 3, i % 500));
    }
    // The first collision twin rides the base data.
    values.push_str(&format!(",({a1}, {b1}, 'twin1')"));
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kc (a, b, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kc"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kc (a, b, t) VALUES (100000, 0, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1, "compound-keyed table enters");

    // Parity: a true compound dup (10, 30) rejects via the folded probe.
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kc (a, b, t) VALUES (10, 30, 'dup')")
        .expect_err("compound dup must reject (needle parity)");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);

    // Same first column, different second: NOT a dup.
    seq += 1;
    e.execute_text(seq, "INSERT INTO kc (a, b, t) VALUES (10, 31, 'ok')")
        .unwrap();

    // The COLLIDING key: same fingerprint as (5, j1), different tuple — the recheck must ACCEPT.
    let conflicts_before = e.chunk_class_unique_probe_conflicts();
    seq += 1;
    e.execute_text(
        seq,
        &format!("INSERT INTO kc (a, b, t) VALUES ({a2}, {b2}, 'twin2')"),
    )
    .unwrap_or_else(|err| panic!("fingerprint collision must NOT reject a distinct key: {err:?}"));
    assert_eq!(
        e.chunk_class_unique_probe_conflicts(),
        conflicts_before,
        "no conflict was recorded for the collision"
    );
    // And the true dup of the twin still rejects.
    seq += 1;
    let err = e
        .execute_text(
            seq,
            &format!("INSERT INTO kc (a, b, t) VALUES ({a1}, {b1}, 'dup-twin')"),
        )
        .expect_err("the twin's true dup must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0, "the whole arc stayed classed");
}

/// P5 charter closure — NULL KEY: PostgreSQL UNIQUE treats every NULL-bearing tuple as distinct.
/// Both NULL inserts succeed while the class stays device-authoritative; the adjacent non-NULL
/// duplicate proves that ordinary unique conflicts still use the device probe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_null_unique_stays_device_native() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kn (a INT, u INT UNIQUE, t TEXT)")
        .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {}, 'txt{:04}')", i + 50_000, i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kn (a, u, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kn"))
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO kn (a, u, t) VALUES (100000, 99000, 'enter')",
    )
    .unwrap();
    assert_eq!(e.chunk_class_entries(), 1, "UNIQUE-column table enters");

    // Sanity: the device probe is live for non-NULL keys.
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kn (a, u, t) VALUES (1, 50001, 'dup')")
        .expect_err("dup u must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0);

    // NULL-bearing keys are distinct under PostgreSQL UNIQUE semantics and remain class-authoritative.
    let probes_before = e.chunk_class_unique_probes();
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO kn (a, u, t) VALUES (2000000, NULL, 'null1')",
    )
    .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "the NULL key must stay chunk-authoritative"
    );
    assert!(
        e.chunk_class_unique_probes() > probes_before,
        "the first NULL insert still traversed the device-native class validation path"
    );
    // The second NULL is distinct too; accepting it must not deauthorize the class.
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO kn (a, u, t) VALUES (2000001, NULL, 'null2')",
    )
    .expect("the second NULL unique key remains distinct");
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "the class remains authoritative"
    );
}

/// Audit M1 — compound partial-NULL uniqueness through every class seam: all NULL-bearing tuples
/// remain distinct within a batch and against existing rows, a key-preserving UPDATE remains
/// device-authoritative, tombstone/reinsert succeeds, and WAL replay lands the identical accepted
/// history.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_compound_partial_null_unique_device_and_replay() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-p5-null-compound-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    let expected = {
        let mut e = Engine::new_local_test_engine();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(
            seq,
            "CREATE TABLE null_k (id INT, u INT, v INT, payload INT, UNIQUE (u, v))",
        )
        .unwrap();
        let values = (0..600)
            .map(|i| format!("({i}, {}, {}, {i})", i + 1000, i + 2000))
            .collect::<Vec<_>>()
            .join(",");
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO null_k VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 8192);
        let _ = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM null_k"))
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO null_k VALUES (90000, 90000, 90000, 0)")
            .unwrap();
        assert_eq!(e.chunk_class_entries(), 1, "premise: compound classed");

        let conflicts_before = e.chunk_class_unique_probe_conflicts();
        let exact_before = e.chunk_class_device_exact_rechecks();
        seq += 1;
        let duplicate = e
            .execute_text(seq, "INSERT INTO null_k VALUES (90001, 1000, 2000, 0)")
            .expect_err("ordinary non-NULL compound uniqueness remains enforced");
        assert!(format!("{duplicate:?}").contains("duplicate key value"));
        assert!(e.chunk_class_unique_probe_conflicts() > conflicts_before);
        assert!(e.chunk_class_device_exact_rechecks() > exact_before);
        assert_eq!(e.chunk_class_deauths(), 0);

        // Same NULL component, distinct second key: legal and device-validated in one batch.
        seq += 1;
        e.execute_text(
            seq,
            "INSERT INTO null_k VALUES (10000, NULL, 7, 1), (10001, NULL, 8, 2)",
        )
        .unwrap();
        // Even equal non-NULL members remain distinct when one UNIQUE member is NULL.
        seq += 1;
        e.execute_text(
            seq,
            "INSERT INTO null_k VALUES (10002, NULL, 9, 3), (10003, NULL, 9, 4)",
        )
        .expect("partial-NULL tuples are distinct within one batch");

        // A tuple equal to an existing partial-NULL tuple is also distinct.
        seq += 1;
        e.execute_text(seq, "INSERT INTO null_k VALUES (10004, NULL, 7, 5)")
            .expect("existing partial-NULL tuple remains distinct");

        // Key-preserving UPDATE: its own NULL-key coordinate is self, not a conflict.
        seq += 1;
        e.execute_text(seq, "UPDATE null_k SET payload = 77 WHERE id = 10000")
            .unwrap();
        // Tombstone the old tuple by a non-key locator; the same NULL key is then reusable.
        seq += 1;
        e.execute_text(seq, "DELETE FROM null_k WHERE id = 10000")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO null_k VALUES (10005, NULL, 7, 88)")
            .unwrap();
        assert_eq!(e.chunk_class_deauths(), 0, "all NULL seams stayed classed");

        let count = match e
            .execute_relational_select(&select("SELECT COUNT(*) FROM null_k"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(v) => v,
            ref other => panic!("count: {other:?}"),
        };
        let sum = match e
            .execute_relational_select(&select("SELECT SUM(payload) FROM null_k"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(v) => v,
            ref other => panic!("sum: {other:?}"),
        };
        (count, sum)
    };
    let e = Engine::open_durable_wal_segment(&base).expect("partial-NULL history replays");
    let count = match e
        .execute_relational_select(&select("SELECT COUNT(*) FROM null_k"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(v) => v,
        ref other => panic!("count: {other:?}"),
    };
    let sum = match e
        .execute_relational_select(&select("SELECT SUM(payload) FROM null_k"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(v) => v,
        ref other => panic!("sum: {other:?}"),
    };
    assert_eq!((count, sum), expected, "live/replay partial-NULL parity");
}

/// P5-2 — THE C2 REPLAY DIFFERENTIAL: every keyed-class verdict must MATCH what recovery's
/// host-path replay would decide — a probe false-accept is a WAL-durable duplicate the replay
/// then REJECTS, i.e. an UNREPLAYABLE acked commit (an RPO violation, strictly worse than a
/// wrong answer). Drive the full keyed history through the class (accepts + rejects +
/// tombstone-reinsert + a key-moving update), crash WITHOUT a checkpoint, reopen: recovery must
/// succeed and the replayed state must be value-identical.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_replay_differential() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-p52-replay-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    const N: i32 = 1000;
    let (count_live, sum_live) = {
        let mut e = Engine::new_local_test_engine();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE kr (a INT PRIMARY KEY, t TEXT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO kr (a, t) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 8192);
        let _ = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kr"))
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (100000, 'enter')")
            .unwrap();
        assert_eq!(e.chunk_class_entries(), 1, "premise: classed");

        // The adversarial history: device-accepted commits interleaved with device-rejected
        // statements (the rejects must NOT be in the WAL), a tombstone re-insert, and a
        // key-moving update.
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (600000, 'fresh')")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (500, 'dup')")
            .expect_err("dup rejected live");
        seq += 1;
        e.execute_text(seq, "DELETE FROM kr WHERE a = 500").unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (500, 'reborn')")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "UPDATE kr SET a = 999999 WHERE a = 45")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (45, 'reused')")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kr (a, t) VALUES (999999, 'taken')")
            .expect_err("moved-to key rejected live");
        assert_eq!(
            e.chunk_class_deauths(),
            0,
            "the whole history stayed classed"
        );

        let count = match e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kr"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        };
        let sum = match e
            .execute_relational_select(&select("SELECT SUM(a) FROM kr"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        };
        (count, sum)
        // DROP = the crash: no checkpoint, the WAL is the only truth.
    };

    // RETIRE-002 recovery repair replays the acked history and must accept every acked commit
    // (C2), then publish a value-identical device generation.
    let e = Engine::open_durable_wal_segment(&base).expect("recovery must replay cleanly (C2)");
    let count = match e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kr"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("count: {other:?}"),
    };
    let sum = match e
        .execute_relational_select(&select("SELECT SUM(a) FROM kr"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("sum: {other:?}"),
    };
    assert_eq!(count, count_live, "replayed cardinality differs (C2)");
    assert_eq!(sum, sum_live, "replayed values differ (C2)");
    // Spot checks on the interesting keys.
    for (key, expect) in [(500, 1i64), (45, 1), (999999, 1), (600000, 1), (44, 1)] {
        let q = select(&format!("SELECT COUNT(*) FROM kr WHERE a = {key}"));
        let got = match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("spot: {other:?}"),
        };
        assert_eq!(got, expect, "key {key}");
    }
}

/// R3-003 / P5-2 — a class INSERT publishes only a transaction-private device-format tail. SELECT
/// and later unique validation consume it, rollback discards it, and COMMIT emits one resolved
/// transaction record before appending the row at the real commit boundary.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_txn_insert_is_private_atomic_and_recoverable() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kt (a INT PRIMARY KEY, t TEXT)")
        .unwrap();
    const N: i32 = 1000;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kt (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kt"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kt (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // A claim followed by a release after BEGIN is absent from the current visible chunks but
    // remains a physical version history conflict. This is the cold-class twin of the resident
    // key-away regression and proves COMMIT no longer depends on the CPU unique-slot ledger.
    seq += 1;
    let history_txn = seq;
    e.execute_text(history_txn, "BEGIN").unwrap();
    e.execute_text(
        history_txn,
        "INSERT INTO kt (a, t) VALUES (800000, 'stale-private')",
    )
    .unwrap();
    seq += 1;
    let compactions_before_history = e.chunk_class_compactions();
    e.execute_text(
        seq,
        "INSERT INTO kt (a, t) VALUES \
         (800000, 'external'), (800001, 'x1'), (800002, 'x2'), (800003, 'x3'), \
         (800004, 'x4'), (800005, 'x5'), (800006, 'x6'), (800007, 'x7')",
    )
    .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kt WHERE a >= 800000 AND a <= 800002")
        .unwrap();
    assert_eq!(
        e.chunk_class_compactions(),
        compactions_before_history,
        "the oldest active writer must fence a >=25%-dead current-only chunk rebuild"
    );
    let exact_before = e.chunk_class_device_exact_rechecks();
    let history_error = e
        .execute_text(history_txn, "COMMIT")
        .expect_err("cold claim/release history must serialize the stale transaction");
    assert!(
        matches!(&history_error, ExecuteError::Serialization(message)
            if message.contains("device unique conflict") && message.contains("write history")),
        "unexpected cold history verdict: {history_error:?}"
    );
    assert!(e.chunk_class_device_exact_rechecks() > exact_before);
    e.execute_text(history_txn, "ROLLBACK").unwrap();
    e.maybe_compact_chunk_class("kt");
    assert!(
        e.chunk_class_compactions() > compactions_before_history,
        "non-vacuity: the same >=25%-dead chunk compacts after the old snapshot retires"
    );
    assert!(e
        .execute_relational_select(&select("SELECT a FROM kt WHERE a = 800000"))
        .unwrap()
        .rows
        .is_empty());
    assert_eq!(e.chunk_class_entries(), 1, "history check stays classed");

    // The dup rejects INSIDE the transaction (the preflight probe). A txn's statements all
    // carry the BEGIN's seq — the txn id.
    seq += 1;
    e.execute_text(seq, "BEGIN").unwrap();
    let err = e
        .execute_text(seq, "INSERT INTO kt (a, t) VALUES (500, 'dup')")
        .expect_err("txn dup must reject at preflight");
    assert!(format!("{err:?}").contains("duplicate key value"));
    e.execute_text(seq, "ROLLBACK").unwrap();

    let fresh = select("SELECT a, t FROM kt WHERE a = 700000");

    // A fresh row is visible only through the retained transaction generation. A later statement
    // probes that private tail too, proving constraint read-your-writes without a CPU index.
    seq += 1;
    e.execute_text(seq, "BEGIN").unwrap();
    e.execute_text(seq, "INSERT INTO kt (a, t) VALUES (700000, NULL)")
        .unwrap();
    assert_eq!(
        e.execute_relational_select_in_transaction(seq, &fresh)
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Int4(700000), SqlValue::Null]
    );
    assert!(e.execute_relational_select(&fresh).unwrap().rows.is_empty());
    let own_dup = e
        .execute_text(seq, "INSERT INTO kt (a, t) VALUES (700000, 'dup-own')")
        .expect_err("private tail must participate in unique validation");
    assert!(format!("{own_dup:?}").contains("duplicate key value"));
    e.execute_text(seq, "ROLLBACK").unwrap();
    assert!(e.execute_relational_select(&fresh).unwrap().rows.is_empty());

    seq += 1;
    e.execute_text(seq, "BEGIN").unwrap();
    let wal_before = e.durable_wal_records().len();
    e.execute_text(seq, "INSERT INTO kt (a, t) VALUES (700000, NULL)")
        .unwrap();
    e.execute_text(seq, "UPDATE kt SET t = 'final' WHERE a = 700000")
        .unwrap();
    e.execute_text(seq, "UPDATE kt SET t = NULL WHERE a = 45")
        .unwrap();
    e.execute_text(seq, "DELETE FROM kt WHERE a = 46").unwrap();
    assert_eq!(
        e.execute_relational_select_in_transaction(seq, &fresh)
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Int4(700000), SqlValue::Text("final".to_string())]
    );
    e.execute_text(seq, "COMMIT").unwrap();
    assert_eq!(e.durable_wal_records().len(), wal_before + 1);
    assert_eq!(e.chunk_class_deauths(), 0, "atomic insert stays classed");
    assert_eq!(
        e.execute_relational_select(&fresh).unwrap().rows.row(0),
        &[SqlValue::Int4(700000), SqlValue::Text("final".to_string())]
    );
    assert_eq!(
        e.execute_relational_select(&select("SELECT t FROM kt WHERE a = 45"))
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Null]
    );
    assert!(e
        .execute_relational_select(&select("SELECT a FROM kt WHERE a = 46"))
        .unwrap()
        .rows
        .is_empty());
    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select(&fresh)
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Int4(700000), SqlValue::Text("final".to_string())]
    );
    assert_eq!(
        recovered
            .execute_relational_select(&select("SELECT t FROM kt WHERE a = 45"))
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Null]
    );
    assert!(recovered
        .execute_relational_select(&select("SELECT a FROM kt WHERE a = 46"))
        .unwrap()
        .rows
        .is_empty());
}

/// A transaction touching one cold table more than once builds one private table entry. Failure
/// on the second operation is post-durable fail-stop, but the globally installed cold Arc remains
/// the exact pre-transaction entry; restart replay installs both resolved mutations exactly once.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_transaction_second_cow_failure_publishes_nothing_and_recovers_exactly() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kt_fail (a INT PRIMARY KEY, t TEXT)")
        .unwrap();
    let values = (0..1000)
        .map(|i| format!("({i}, 'v{i:04}')"))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kt_fail VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kt_fail"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO kt_fail VALUES (100000, 'enter')")
        .unwrap();
    assert!(e.table_chunk_authoritative("kt_fail").is_some());

    let before = e.read_streaming_cold_chunks()["kt_fail"].clone();
    seq += 1;
    let txn = seq;
    e.execute_text(txn, "BEGIN").unwrap();
    e.execute_text(txn, "UPDATE kt_fail SET t = 'changed' WHERE a = 40")
        .unwrap();
    e.execute_text(txn, "DELETE FROM kt_fail WHERE a = 41")
        .unwrap();
    e.fail_transaction_cold_mutation_at(2);
    let error = e.execute_text(txn, "COMMIT").unwrap_err();
    assert!(error.to_string().contains("restart recovery required"));
    let after = e.read_state.residency.streaming_cold_chunks.load_full();
    assert!(
        Arc::ptr_eq(&before, &after["kt_fail"]),
        "a second-operation decline must leave the original cold table entry installed"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select(&select("SELECT t FROM kt_fail WHERE a = 40"))
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Text("changed".to_string())]
    );
    assert!(
        recovered
            .execute_relational_select(&select("SELECT a FROM kt_fail WHERE a = 41"))
            .unwrap()
            .rows
            .is_empty(),
        "recovery must replay the complete resolved transaction, not the first cold COW only"
    );
}

/// P5-2 (audit MEDIUM) — TEXT UNIQUE KEY through the class probe: a text key folds to ONE
/// FNV-1a word over its UTF-8 bytes via the fold kernel's TEXT SENTINEL branch (widths[k]==0,
/// blob span read), and the host needle derives the SAME word — a divergence is a silent
/// all-miss dup accept, so the dup rejection is the parity proof. The tombstone re-insert and
/// the C2 replay differential ride the same history.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_text_key_probe_and_replay() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-p52-text-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    const N: i32 = 1000;
    let (count_live, sum_live) = {
        let mut e = Engine::new_local_test_engine();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE kx (k TEXT PRIMARY KEY, v INT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("('key-{i:05}', {i})"));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO kx (k, v) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 8192);
        let _ = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kx"))
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kx (k, v) VALUES ('enter', -1)")
            .unwrap();
        assert_eq!(e.chunk_class_entries(), 1, "the TEXT-keyed table enters");

        // Parity: the true text dup rejects via the folded probe (base chunk + tail chunk).
        seq += 1;
        let err = e
            .execute_text(seq, "INSERT INTO kx (k, v) VALUES ('key-00500', 0)")
            .expect_err("text dup must reject (fold/needle parity)");
        assert!(format!("{err:?}").contains("duplicate key value"));
        seq += 1;
        let err = e
            .execute_text(seq, "INSERT INTO kx (k, v) VALUES ('enter', 0)")
            .expect_err("tail text dup must reject");
        assert!(format!("{err:?}").contains("duplicate key value"));
        assert_eq!(e.chunk_class_deauths(), 0, "rejections stay classed");

        // Near-miss shapes: prefix / suffix / case variants are DISTINCT keys and must accept.
        seq += 1;
        e.execute_text(
            seq,
            "INSERT INTO kx (k, v) VALUES ('key-0050', 1), ('key-005000', 2), ('KEY-00500', 3)",
        )
        .unwrap();

        // Tombstone re-insert: masked hit is not a conflict.
        seq += 1;
        e.execute_text(seq, "DELETE FROM kx WHERE k = 'key-00007'")
            .unwrap();
        seq += 1;
        e.execute_text(seq, "INSERT INTO kx (k, v) VALUES ('key-00007', 700)")
            .unwrap();
        assert_eq!(
            e.chunk_class_deauths(),
            0,
            "the whole history stayed classed"
        );

        let count = match e
            .execute_relational_select(&select("SELECT COUNT(*) FROM kx"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        };
        let sum = match e
            .execute_relational_select(&select("SELECT SUM(v) FROM kx"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        };
        (count, sum)
        // DROP = the crash.
    };

    // C2: RETIRE-002 recovery repair must replay the acked text-key history cleanly.
    let e = Engine::open_durable_wal_segment(&base).expect("recovery must replay cleanly (C2)");
    let count = match e
        .execute_relational_select(&select("SELECT COUNT(*) FROM kx"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("count: {other:?}"),
    };
    let sum = match e
        .execute_relational_select(&select("SELECT SUM(v) FROM kx"))
        .unwrap()
        .rows
        .row(0)[0]
    {
        SqlValue::Int8(n) => n,
        ref other => panic!("sum: {other:?}"),
    };
    assert_eq!(count, count_live, "replayed cardinality differs (C2)");
    assert_eq!(sum, sum_live, "replayed values differ (C2)");
}

/// P5-3 — BY-KEY DML LOCATE: an Eq-on-unique-key WHERE resolves through the chunk key-index
/// probe (one device locate + slot rechecks) instead of the full fold scan, with matches
/// materialized from the rechecked slots — the reverse-gather decoder stays off the point-DML
/// hot path. Residual non-key predicates and visibility re-filter on-device; misses and dead
/// keys yield 0-row DML; range WHERE keeps the fold. The differential twin: the same history driven
/// through range predicates (the fold path) must land the identical state.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_dml_key_locate() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    for table in ["kp", "kf"] {
        seq += 1;
        e.execute_text(
            seq,
            &format!("CREATE TABLE {table} (a INT PRIMARY KEY, v INT)"),
        )
        .unwrap();
        const N: i32 = 1000;
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {})", i * 10));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO {table} (a, v) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 8192);
    for table in ["kp", "kf"] {
        e.transition_device_table_to_streaming_repair_above(table, 1)
            .unwrap();
        let _ = e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        assert!(
            e.table_chunk_authoritative(table).is_some(),
            "{table} must class immediately after its forced streaming capture"
        );
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, v) VALUES (100000, -1)"),
        )
        .unwrap();
        assert!(
            e.table_chunk_authoritative(table).is_some(),
            "{table} must remain classed after its entry insert"
        );
    }
    assert_eq!(e.chunk_class_entries(), 2, "both twins classed");

    // THE PROBE TWIN (kp): point DML by key. THE FOLD TWIN (kf): the same logical ops through
    // range predicates (`a >= k AND a <= k` is 2 non-Eq filters -> the fold locate).
    let key_locates_0 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 500").unwrap();
    assert!(
        e.chunk_class_dml_key_locates() > key_locates_0,
        "the point DELETE rode the key probe"
    );
    seq += 1;
    e.execute_text(seq, "DELETE FROM kf WHERE a >= 500 AND a <= 500")
        .unwrap();

    seq += 1;
    e.execute_text(seq, "UPDATE kp SET v = 12345 WHERE a = 700")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "UPDATE kf SET v = 12345 WHERE a >= 700 AND a <= 700")
        .unwrap();

    // Residual predicate: key matches, non-key predicate does NOT -> 0-row DML.
    let key_locates_1 = e.chunk_class_dml_key_locates();
    let exact_rechecks_1 = e.chunk_class_device_exact_rechecks();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 701 AND v = -999")
        .unwrap();
    assert!(
        e.chunk_class_dml_key_locates() > key_locates_1,
        "the residual-predicate DELETE still rode the probe"
    );
    assert!(
        e.chunk_class_device_exact_rechecks() > exact_rechecks_1,
        "the residual predicate was decided by the device VM"
    );
    seq += 1;
    e.execute_text(
        seq,
        "DELETE FROM kf WHERE a >= 701 AND a <= 701 AND v = -999",
    )
    .unwrap();

    // A missing key and a DEAD key: 0-row DML, no error, still classed.
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 987654")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kp WHERE a = 500").unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kf WHERE a >= 987654 AND a <= 987654")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM kf WHERE a >= 500 AND a <= 500")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "every shape stayed classed");

    // THE DIFFERENTIAL: identical final state on both twins.
    for q in [
        "SELECT COUNT(*) FROM {T}",
        "SELECT SUM(v) FROM {T}",
        "SELECT SUM(a) FROM {T}",
        "SELECT COUNT(*) FROM {T} WHERE v = 12345",
    ] {
        let probe = e
            .execute_relational_select(&select(&q.replace("{T}", "kp")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        let fold = e
            .execute_relational_select(&select(&q.replace("{T}", "kf")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        assert_eq!(probe, fold, "probe/fold divergence on {q}");
    }
    // And the closed form: 1001 rows - 1 deleted; v updated on one row.
    let q = select("SELECT COUNT(*) FROM kp");
    match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
        SqlValue::Int8(n) => assert_eq!(n, 1000),
        ref other => panic!("count: {other:?}"),
    }
}

/// Audit C1 — off-lock class prepare must keep one entry Arc from coordinate locate through
/// row-image materialization and the epoch token. Pause a DELETE after it pins E1, publish E2 by
/// tombstone-republishing the chunk (including the paused DELETE's key), then resume. The E1
/// write-set must conflict with E2. The active old snapshot must also defer physical compaction:
/// reclaiming that history while the writer is pinned would make the conflict unverifiable.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_offlock_prepare_pins_one_entry_across_sidecar_republish() {
    let mut engine = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut engine, &mut seq) {
        return;
    }
    seq += 1;
    engine
        .execute_text(seq, "CREATE TABLE epoch_t (a INT PRIMARY KEY, v INT)")
        .unwrap();
    let values = (0..1000)
        .map(|i| format!("({i}, {i})"))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    engine
        .execute_text(seq, &format!("INSERT INTO epoch_t VALUES {values}"))
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    let _ = engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM epoch_t"))
        .unwrap();
    seq += 1;
    engine
        .execute_text(seq, "INSERT INTO epoch_t VALUES (100000, -1)")
        .unwrap();
    assert_eq!(engine.chunk_class_entries(), 1, "premise: classed");

    let engine = std::sync::Arc::new(engine);
    let (pinned, resume) = crate::engine_streaming_exec::install_class_resolve_pin_hook();
    let deleting = std::sync::Arc::clone(&engine);
    let delete_seq = seq + 1;
    let delete = std::thread::spawn(move || {
        deleting.execute_dml_concurrent(delete_seq, "DELETE FROM epoch_t WHERE v = 100")
    });
    pinned.wait(); // DELETE pinned E1 and its read snapshot; it has not located yet.

    let stamps_before = engine.streaming_cold_stamps();
    let compactions_before = engine.chunk_class_compactions();
    let update_seq = seq + 2;
    engine
        .execute_text(update_seq, "DELETE FROM epoch_t WHERE a < 200 OR a = 900")
        .unwrap();
    assert!(
        engine.streaming_cold_stamps() > stamps_before,
        "the interposed write must publish a tombstone-sidecar E2"
    );
    assert_eq!(
        engine.chunk_class_compactions(),
        compactions_before,
        "the pinned old writer must defer physical history reclamation"
    );
    resume.wait();
    let err = delete
        .join()
        .expect("delete thread")
        .expect_err("the stale off-lock write-set must conflict");
    assert!(
        matches!(err, ExecuteError::Serialization(_)),
        "expected SI conflict, got {err:?}"
    );
    let row = engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM epoch_t WHERE a = 100"))
        .unwrap();
    assert_eq!(
        row.rows.iter().collect::<Vec<_>>(),
        vec![&[SqlValue::Int8(0)][..]],
        "the interposed delete remains the sole committed writer"
    );
    engine.maybe_compact_chunk_class("epoch_t");
    assert!(
        engine.chunk_class_compactions() > compactions_before,
        "the same tombstones must compact after the old writer retires"
    );
}

/// P5-3 (audit LOW) — COMPOUND-KEY DML through the probe: the by-key locate on a (a, b) PK
/// rides the FOLDED fingerprint needle (not the raw-i32 fast path) — a needle/build divergence
/// here is a silently MISSED DML match (lost delete/update), so the probe twin (Eq on both key
/// columns) differentials against the fold twin (range predicates) over the identical history.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_compound_dml_key_locate() {
    let mut e = Engine::new_local_test_engine();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    for table in ["cp", "cf"] {
        seq += 1;
        e.execute_text(
            seq,
            &format!("CREATE TABLE {table} (a INT, b INT, v INT, PRIMARY KEY (a, b))"),
        )
        .unwrap();
        const N: i32 = 1000;
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {}, {})", i * 3, i * 10));
        }
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, b, v) VALUES {values}"),
        )
        .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 8192);
    for table in ["cp", "cf"] {
        e.transition_device_table_to_streaming_repair_above(table, 1)
            .unwrap();
        let _ = e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        assert!(
            e.table_chunk_authoritative(table).is_some(),
            "{table} must class immediately after its forced streaming capture"
        );
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, b, v) VALUES (100000, 0, -1)"),
        )
        .unwrap();
        assert!(
            e.table_chunk_authoritative(table).is_some(),
            "{table} must remain classed after its entry insert"
        );
    }
    assert_eq!(e.chunk_class_entries(), 2, "both compound twins classed");

    // Point DELETE + UPDATE by the FULL compound key (probe twin) vs range (fold twin).
    let key_locates_0 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "DELETE FROM cp WHERE a = 500 AND b = 1500")
        .unwrap();
    assert!(
        e.chunk_class_dml_key_locates() > key_locates_0,
        "the compound point DELETE rode the folded-needle probe"
    );
    seq += 1;
    e.execute_text(
        seq,
        "DELETE FROM cf WHERE a >= 500 AND a <= 500 AND b >= 1500 AND b <= 1500",
    )
    .unwrap();
    seq += 1;
    e.execute_text(seq, "UPDATE cp SET v = 777 WHERE a = 700 AND b = 2100")
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "UPDATE cf SET v = 777 WHERE a >= 700 AND a <= 700 AND b >= 2100 AND b <= 2100",
    )
    .unwrap();
    // A partial-key Eq (only `a`) does NOT cover the compound index -> the fold serves it.
    let key_locates_1 = e.chunk_class_dml_key_locates();
    seq += 1;
    e.execute_text(seq, "DELETE FROM cp WHERE a = 600").unwrap();
    assert_eq!(
        e.chunk_class_dml_key_locates(),
        key_locates_1,
        "a partial key must NOT ride the probe"
    );
    seq += 1;
    e.execute_text(seq, "DELETE FROM cf WHERE a >= 600 AND a <= 600")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "every shape stayed classed");

    for q in [
        "SELECT COUNT(*) FROM {T}",
        "SELECT SUM(v) FROM {T}",
        "SELECT SUM(b) FROM {T}",
        "SELECT COUNT(*) FROM {T} WHERE v = 777",
    ] {
        let probe = e
            .execute_relational_select(&select(&q.replace("{T}", "cp")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        let fold = e
            .execute_relational_select(&select(&q.replace("{T}", "cf")))
            .unwrap()
            .rows
            .row(0)
            .to_vec();
        assert_eq!(probe, fold, "compound probe/fold divergence on {q}");
    }
}
