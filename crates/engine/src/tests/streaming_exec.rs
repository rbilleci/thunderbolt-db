use super::*;
mod cold_tier;
mod grouped_distinct;
mod ordered;
mod projection;
mod rank_windows;
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
    let mut e = Engine::new_local_cpu_oracle();
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
            let key = if i % 97 == 0 { "NULL".to_string() } else { i.to_string() };
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
            let key = if i % 89 == 0 { "NULL".to_string() } else { i.to_string() };
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
            let key = if i == 29 { "NULL".to_string() } else { (i % 3).to_string() };
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
    e.set_relational_residency_budget_bytes(0, 4096);
    // Prime both cold entries, then stamp one left row dead. The join must compose the chunk
    // sidecar visibility mask before matching; a leaked tombstone would survive the resident oracle.
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM jl"))
        .unwrap();
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM jr"))
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM jl WHERE k = 400").unwrap();
    let sql = "SELECT l.k, l.lv, r.rv, l.note FROM jl l JOIN jr r ON l.k = r.k \
               WHERE l.lv >= 350 AND r.rv < 550";
    let streamed = e
        .execute_resident_expr_select_sql(sql)
        .expect("streaming join");
    assert_eq!(streamed.executed_target, DeviceTarget::Gpu(0));
    assert!(e.streaming_join_hits() > 0, "streaming join route fired");
    assert!(e.streaming_join_block_pairs() > 1, "genuine multi-block fold");
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
    assert!(aliased_values.windows(2).all(|values| values[0] >= values[1]));
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
    assert_eq!(streamed_nway_right.rows.row(0), &[SqlValue::Null, SqlValue::Null, SqlValue::Int4(9)]);
    let nway_two_right_sql = "SELECT a.x, b.y, d.q \
                              FROM jna a RIGHT JOIN jnb b ON a.k = b.k \
                                         FULL JOIN jnd d ON b.k = d.k \
                              ORDER BY b.y DESC NULLS LAST, d.q, a.x LIMIT 5";
    e.set_relational_residency_budget_bytes(0, 4608);
    let streamed_nway_two_right = e
        .execute_resident_expr_select_sql(nway_two_right_sql)
        .expect("streaming prefix replay across two RIGHT/FULL steps");
    assert!(streamed_nway_two_right.rows.iter().any(|row| {
        row == [SqlValue::Null, SqlValue::Int4(777), SqlValue::Int4(9)]
    }));
    e.set_relational_residency_budget_bytes(0, 4096);
    let outer_sql =
        "SELECT a.k, a.x, b.k, b.y FROM joa a FULL JOIN job b ON a.k = b.k";
    let streamed_outer = e
        .execute_resident_expr_select_sql(outer_sql)
        .expect("streaming FULL OUTER join");
    assert_eq!(streamed_outer.rows.len(), 600);
    assert!(streamed_outer.rows.iter().any(|row| row[0] == SqlValue::Null));
    assert!(streamed_outer.rows.iter().any(|row| row[2] == SqlValue::Null));
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
            crate::rel_exec_helpers::compare_sql_values(&a[0], &b[0]).then_with(|| {
                crate::rel_exec_helpers::compare_sql_values(&a[1], &b[1])
            })
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
    assert!(streamed_outer_ordered.rows.iter().all(|row| {
        resident_outer.rows.iter().any(|candidate| candidate == row)
    }));
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
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE mix4 (k INT, v INT)").unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE mix8 (k BIGINT, v INT)").unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO mix4 VALUES (-1, 10), (2, 20)").unwrap();
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

// ===================== P1 (sealed-shards-primary): the DURABLE cold checkpoint =====================

/// Pure encode/decode round-trip of the chunk descriptor (no GPU): every persisted field survives;
/// the transient bookkeeping (memory proof, refresh cost, invalidation) restores to its defaults.
#[test]
fn cold_checkpoint_descriptor_round_trips() {
    let descriptor = RelationalResidencySnapshot {
        gpu_id: 3,
        schema: "public".into(),
        table: "t".into(),
        generation: 41,
        row_count: 12,
        capacity: 12,
        column_count: 4,
        resident_bytes: 4096,
        resident_device_int4_columns: vec!["a".into(), "b".into()],
        resident_device_int4_column_stats: vec![
            crate::relational_model::ResidentDeviceInt4ColumnStats {
                name: "a".into(),
                min: -7,
                max: 900,
            },
        ],
        resident_device_int8_columns: vec!["big".into()],
        resident_device_numeric_columns: vec!["price".into()],
        resident_device_bool_columns: vec![
            crate::relational_model::ResidentDeviceBoolColumnLayout {
                name: "flag".into(),
                bitmap_byte_offset: 128,
            },
        ],
        resident_device_text_columns: vec![
            crate::relational_model::ResidentDeviceTextColumnLayout {
                name: "name".into(),
                offsets_byte_offset: 256,
                bytes_byte_offset: 304,
                bytes_len: 77,
            },
        ],
        resident_device_null_columns: vec![
            crate::relational_model::ResidentDeviceNullBitmapLayout {
                name: "b".into(),
                bitmap_byte_offset: 512,
            },
        ],
        valid_through_index: 99,
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        last_refresh_cost: None,
        admission_budget_bytes: None,
        resident_bytes_after_admission: 4096,
        evicted_tables_on_admission: Vec::new(),
        device_memory_proof: None,
    };
    let mut w = crate::engine_streaming_exec::ColdCkptWriter {
        inner: Vec::<u8>::new(),
        hash: crate::engine_streaming_exec::FNV_OFFSET,
    };
    crate::engine_streaming_exec::encode_cold_descriptor(&mut w, &descriptor).unwrap();
    let mut r = crate::engine_streaming_exec::ColdCkptReader {
        inner: std::io::Cursor::new(w.inner),
    };
    let decoded = crate::engine_streaming_exec::decode_cold_descriptor(&mut r).unwrap();
    assert_eq!(decoded, descriptor);
}

/// Shared P1 harness: a lanes-mode durable database whose table `t` (plain 2-col int4, NO PK —
/// keeps the shape elision-ineligible so the streaming scan's store premise holds) has
/// `serial_rows` rows from the serial (pre-activation) phase and 24 fabricated lane-commit rows.
/// Returns (wal base path, expected row count, next fabricated row id base, next lane seq).
fn p1_lanes_streaming_fixture(
    tag: &str,
    serial_rows: i32,
) -> Option<(std::path::PathBuf, i64, u64)> {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-cold-ckpt-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    let row_base = {
        let mut e = Engine::new_local_cpu_oracle();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return None; // off-box: skip
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE t (a INT, b INT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..serial_rows {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {})", i * 2));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO t (a, b) VALUES {values}"))
            .unwrap();
        e.read_state.mvcc.current_row_id()
    };
    // Fabricated 2-lane history: 24 one-row binary INSERT commits (the recovery-suite pattern —
    // no live post-activation writes are needed, so the test never trips the classic-write guard).
    let tiny = 16 << 10;
    {
        let set = gpu_db_wal::FuaWalLaneSet::create(&base, 2, 2, tiny).expect("create lanes");
        for seq in 0..24u64 {
            let values = vec![SqlValue::Int4(10_000 + seq as i32), SqlValue::Int4(0)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(row_base + seq, values.as_slice())],
            )
            .expect("binary encode");
            set.append(
                (seq % 2) as usize,
                seq,
                &[gpu_db_wal::WalRecord {
                    txn_id: 500 + seq,
                    payload: payload.into(),
                }],
            )
            .expect("append");
        }
        set.wait_durable(24).expect("durable");
    }
    Some((base, i64::from(serial_rows) + 24, row_base + 24))
}

fn p1_count(e: &Engine) -> i64 {
    let q = select("SELECT COUNT(*) FROM t");
    match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
        SqlValue::Int8(n) => n,
        ref other => panic!("COUNT returned {other:?}"),
    }
}

/// Recovery now eagerly installs a GPU-resident snapshot. These tests exercise the distinct over-budget
/// streaming/cold-checkpoint path, so explicitly evict that snapshot after setting the tiny budget. The cold
/// tier is separate ownership and deliberately survives this resident-cache eviction across reopen.
fn p1_force_streaming(e: &mut Engine, budget: u64) {
    e.set_relational_residency_budget_bytes(0, budget);
    let catalog = e.ddl_catalog();
    catalog.relational_resident_cache.remove_table(
        "t",
        &e.read_state.residency,
        &e.read_state.route_telemetry,
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_restores_streaming_cold_across_reopen() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("restore", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        assert!(
            e.streaming_fold_hits() >= 1 && e.streaming_cold_builds() >= 1,
            "premise: the read streamed and captured the cold tier (fold {}, builds {})",
            e.streaming_fold_hits(),
            e.streaming_cold_builds()
        );
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(
            e.streaming_cold_checkpointed() >= 1,
            "the lanes checkpoint must persist the quiesced cold tier"
        );
        cut
    };
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    assert!(
        artifact.exists(),
        "artifact {} must exist",
        artifact.display()
    );

    // REOPEN: the seam install restores the cold tier; the first streaming read is a byte REPLAY
    // (a HIT with zero fresh scan-builds), and the answer matches.
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after checkpoint");
    assert!(
        e.streaming_cold_restored() >= 1,
        "reopen must restore the cold tier from the checkpoint artifact"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(p1_count(&e), expected);
    assert!(
        e.streaming_cold_hits() >= 1,
        "the restored entry must serve the first streaming read"
    );
    assert_eq!(
        e.streaming_cold_builds(),
        0,
        "no fresh scan-build: the restore IS the build"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_patches_forward_post_checkpoint_wal_suffix() {
    let Some((base, expected, next_row)) = p1_lanes_streaming_fixture("suffix", 300) else {
        return;
    };
    let budget = 512u64;
    {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(e.streaming_cold_checkpointed() >= 1);
    }
    // Post-checkpoint WAL SUFFIX: 6 more fabricated lane commits ABOVE the checkpoint cut.
    let tiny = 16 << 10;
    {
        let set = gpu_db_wal::FuaWalLaneSet::reopen_from(&base, 2, 2, tiny, 24)
            .expect("wal-level reopen from baseline");
        for seq in 24..30u64 {
            let values = vec![SqlValue::Int4(20_000 + seq as i32), SqlValue::Int4(1)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(next_row + (seq - 24), values.as_slice())],
            )
            .expect("binary encode");
            set.append(
                (seq % 2) as usize,
                seq,
                &[gpu_db_wal::WalRecord {
                    txn_id: 900 + seq,
                    payload: payload.into(),
                }],
            )
            .expect("append");
        }
        set.wait_durable(30).expect("durable");
    }
    // REOPEN: restore at the seam, then the 6-record suffix replays THROUGH the restored entry —
    // the 6c-3 commit hooks patch it forward (the WAL suffix IS the delta stream). The first
    // streaming read replays patched bytes and sees the suffix rows.
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after suffix");
    assert!(
        e.streaming_cold_restored() >= 1,
        "the seam restore must land before the suffix replays"
    );
    assert!(
        e.streaming_cold_patches() >= 1,
        "suffix replay must patch the restored entry via the commit hooks"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(
        p1_count(&e),
        expected + 6,
        "the suffix rows must be visible"
    );
    assert!(e.streaming_cold_hits() >= 1);
    assert_eq!(
        e.streaming_cold_builds(),
        0,
        "restore + patches carried the entry — no fresh scan-build"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_corrupt_artifact_is_skipped_never_wrong() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("corrupt", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert!(e.streaming_cold_checkpointed() >= 1);
        cut
    };
    // Flip one byte in the artifact BODY (past the magic): the FNV trailer must reject it.
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    let mut bytes = std::fs::read(&artifact).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    std::fs::write(&artifact, &bytes).unwrap();

    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen with corrupt artifact");
    assert_eq!(
        e.streaming_cold_restored(),
        0,
        "a checksum-failed artifact must restore NOTHING"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(
        p1_count(&e),
        expected,
        "the read rebuilds from the store — never wrong"
    );
    assert!(
        e.streaming_cold_builds() >= 1,
        "the skipped restore leaves the first read to scan + capture"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_boundary_mismatch_is_skipped() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("boundary", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert!(e.streaming_cold_checkpointed() >= 1);
        cut
    };
    // Tamper the artifact's BOUNDARY field (u64 right after the magic) and RECOMPUTE the FNV
    // trailer — a checksum-valid artifact whose boundary does not match the replay seam. The
    // strict-equality guard must skip it (installing would replay bytes from the WRONG commit
    // index — the one guard corruption cannot exercise).
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    let mut bytes = std::fs::read(&artifact).unwrap();
    let magic_len = b"GPUDBCOLDCKPT1\n".len();
    let boundary = u64::from_le_bytes(bytes[magic_len..magic_len + 8].try_into().unwrap());
    bytes[magic_len..magic_len + 8].copy_from_slice(&(boundary + 1).to_le_bytes());
    let body_len = bytes.len() - 8;
    let mut hash = crate::engine_streaming_exec::FNV_OFFSET;
    for b in &bytes[..body_len] {
        hash = (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3);
    }
    bytes[body_len..].copy_from_slice(&hash.to_le_bytes());
    std::fs::write(&artifact, &bytes).unwrap();

    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen with tampered boundary");
    assert_eq!(
        e.streaming_cold_restored(),
        0,
        "a boundary-mismatched artifact must restore NOTHING (checksum alone cannot catch it)"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(
        p1_count(&e),
        expected,
        "the read rebuilds from the store — never wrong"
    );
    assert!(e.streaming_cold_builds() >= 1);
}

/// AUDIT HIGH regression (the boundary convention): the LIVE lane pump publishes committed_seq
/// as the EXCLUSIVE frontier (`visible_global_cut = base_seq + cut`), while the recovery seam's
/// replay publishes the INCLUSIVE last record index (`base_seq + cut - 1`) — one less. The
/// pre-fix code stamped the artifact with the live watermark verbatim, so every artifact captured
/// from a pump-published engine carried a boundary ONE HIGH and the restore silently never fired
/// in production (the four sibling tests replay-derive their watermark on BOTH sides, so they
/// cannot see it). This test emulates the pump's convention exactly — it re-publishes the
/// watermark at the frontier before checkpointing — and requires the restore to land anyway.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_restores_under_lane_pump_frontier_watermark() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("frontier", 300) else {
        return;
    };
    let budget = 512u64;
    {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        assert!(e.streaming_cold_builds() >= 1);
        // Emulate the pump: publish the EXCLUSIVE frontier (base_seq + cut), the value
        // engine_dml_concurrent's settle path publishes after a quiesced wave. The visible set
        // is unchanged (no stamp exists at the frontier).
        let lanes = e.intent_lanes.as_ref().expect("lanes installed");
        let frontier = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire) + 24;
        e.publish_committed_seq(frontier);
        assert_eq!(
            e.committed_seq(),
            frontier,
            "premise: frontier-convention watermark"
        );
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(
            e.streaming_cold_checkpointed() >= 1,
            "the frontier watermark must be ACCEPTED as the quiescence proof"
        );
    }
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after frontier checkpoint");
    assert!(
        e.streaming_cold_restored() >= 1,
        "the artifact must carry the SEAM boundary (inclusive last index), not the live \
         frontier — a frontier-stamped artifact never restores"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(p1_count(&e), expected);
    assert!(e.streaming_cold_hits() >= 1);
    assert_eq!(e.streaming_cold_builds(), 0);
}

// ============ P3 (sealed-shards-primary): the DML WHERE-locate as a streaming fold ============

/// Range-WHERE DELETE on a NON-ADMITTED (over-budget) table: the value index cannot bound it (no
/// Eq leaf) and the device arm has no shards — previously the pure-host seq_scan+filter loop. The
/// locate must now run ON-DEVICE via the streaming fold (counter-gated) and produce exactly the
/// host arm's result (differential: a twin engine with no budget runs the identical statement
/// through the host loop).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_range_delete_on_device() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local_cpu_oracle(); // host-arm oracle (no budget -> host seq_scan locate)
    let mut twin_seq = 0u64;

    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(*s, "CREATE TABLE big (a INT, b INT)")
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO big (a, b) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 4096);

    assert_eq!(e.dml_streaming_resolve_hits(), 0);
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a > 1200")
        .unwrap();
    assert_eq!(
        e.dml_streaming_resolve_hits(),
        1,
        "the range DELETE locate must resolve via the streaming fold"
    );
    twin_seq += 1;
    twin.execute_text(twin_seq, "DELETE FROM big WHERE a > 1200")
        .unwrap();

    // Differential: identical surviving rows (read both through the same CPU-pinned path).
    e.clear_relational_residency_budget_bytes(0);
    let q = select("SELECT a, b FROM big ORDER BY a");
    let got = e
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let want = twin
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got.len(), 1201, "rows 0..=1200 survive");
    assert_eq!(got, want, "device locate == host locate");
}

/// The UPDATE twin: range-WHERE assignments through the streaming locate, differential-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_range_update_on_device() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local_cpu_oracle();
    let mut twin_seq = 0u64;

    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(*s, "CREATE TABLE big (a INT, b INT)")
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO big (a, b) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 4096);

    seq += 1;
    e.execute_text(seq, "UPDATE big SET b = -5 WHERE a >= 1400")
        .unwrap();
    assert_eq!(
        e.dml_streaming_resolve_hits(),
        1,
        "the range UPDATE locate must resolve via the streaming fold"
    );
    twin_seq += 1;
    twin.execute_text(twin_seq, "UPDATE big SET b = -5 WHERE a >= 1400")
        .unwrap();

    e.clear_relational_residency_budget_bytes(0);
    let q = select("SELECT a, b FROM big ORDER BY a");
    let got = e
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let want = twin
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got.len(), N as usize);
    assert_eq!(got, want, "device locate == host locate");
    assert_eq!(
        got.iter().filter(|r| r[1] == SqlValue::Int4(-5)).count(),
        100,
        "rows 1400..=1499 updated"
    );
}

/// A zero-match range DELETE is a VALID streaming resolve (Some(vec![]) — the conditional 0-row
/// delete), not a decline: the counter fires and nothing changes.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_zero_matches_is_a_resolve() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a > 999999")
        .unwrap();
    assert_eq!(
        e.dml_streaming_resolve_hits(),
        1,
        "0-match locate still resolves on-device"
    );

    let q = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N))]],
        "nothing deleted"
    );
}

/// Without a configured budget the locate DECLINES (no streaming) and the host arm serves —
/// byte-identical default behavior (the activation-gate contract).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_declines_without_budget() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a, b) VALUES (1, 2), (5, 6), (9, 10)")
        .unwrap();
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a > 4").unwrap();
    assert_eq!(e.dml_streaming_resolve_hits(), 0, "no budget -> host arm");
    let q = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(1)]]
    );
}

/// AUDIT M1 (P3): the streaming locate is the FIRST consumer of the DML predicate lowering with
/// NO host recheck (the sibling device arms recheck; the read folds set the no-recheck
/// precedent) — so the non-int4 type matrix must be differential-gated on THIS path. One
/// NULL-bearing mixed-type table; per statement: the budgeted engine must resolve via the
/// streaming fold (counter-gated) and its final state must equal a host-arm twin's exactly.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_dml_locate_type_matrix_differential() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local_cpu_oracle();
    let mut twin_seq = 0u64;

    const N: i32 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        // Every 7th text NULL; every 5th numeric NULL — the 3VL exclusion must match the host's.
        let t = if i % 7 == 0 {
            "NULL".to_string()
        } else {
            format!("'txt{:04}'", (i * 37) % 1000)
        };
        let n = if i % 5 == 0 {
            "NULL".to_string()
        } else {
            format!("{}.{:02}", i % 90, i % 100)
        };
        let d = format!("'2024-{:02}-{:02}'", 1 + (i % 12), 1 + (i % 28));
        let flag = if i % 2 == 0 { "true" } else { "false" };
        let big = i64::from(i) * 1_000_000_007;
        values.push_str(&format!("({i}, {t}, {d}, {n}, {flag}, {big})"));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(
                *s,
                "CREATE TABLE mix (a INT, t TEXT, d DATE, n NUMERIC(10,2), flag BOOL, big BIGINT)",
            )
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO mix VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 4096);

    let statements = [
        // TEXT ordering (byte-lexicographic device kernel vs host compare) over a NULL-bearing col.
        "DELETE FROM mix WHERE t > 'txt0800'",
        // DATE ordering (canonical-text lowering -> days round-trip).
        "DELETE FROM mix WHERE d < '2024-03-15'",
        // NUMERIC scale (rescale-to-column-scale peephole) range.
        "DELETE FROM mix WHERE n >= 44.10",
        // BIGINT (i64 section) range + OR-of-AND groups incl. a bool leaf.
        "DELETE FROM mix WHERE big > 400000000000 OR flag = false AND a < 100",
        // UPDATE through the same locate: text range target, int assignment.
        "UPDATE mix SET a = -1 WHERE t < 'txt0200'",
    ];
    let q = select("SELECT a, t, d, n, flag, big FROM mix ORDER BY big");
    for (i, statement) in statements.iter().enumerate() {
        let hits_before = e.dml_streaming_resolve_hits();
        seq += 1;
        e.execute_text(seq, statement).unwrap();
        assert_eq!(
            e.dml_streaming_resolve_hits(),
            hits_before + 1,
            "statement {i} ({statement}) must resolve via the streaming fold"
        );
        twin_seq += 1;
        twin.execute_text(twin_seq, statement).unwrap();

        e.clear_relational_residency_budget_bytes(0);
        let got = e
            .execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>();
        let want = twin
            .execute_relational_select(&q)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            got, want,
            "statement {i} ({statement}): device locate != host locate"
        );
        e.set_relational_residency_budget_bytes(0, 4096);
    }
}

// ========== P2 (sealed-shards-primary): SV2 tombstone sidecars for cold chunks ==========

/// Multi-row deletes across SEVERAL chunks stamp sidecars (zero rebuilds) and every read shape —
/// aggregate AND row-level projection — masks the tombstoned rows in-kernel, matching a
/// host-path twin exactly.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_sidecar_stamps_mask_rows_across_chunks() {
    // These gates exercise the STORE-DRIVEN patch/stamp machinery (live for non-class tables);
    // without this the table class-enters mid-test and the semantics legitimately change.
    let _class_off = ClassEntryDisabled::new();
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    // Build the cold tier (multi-chunk).
    let q_count = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q_count)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N))]]
    );
    assert!(e.streaming_cold_builds() >= 1);

    // Rows 100, 700 and 1400 live in DIFFERENT chunks (ids ascending; ~12 chunks of ~128 rows).
    // Each Eq-DELETE resolves via the value index (host arm) and must EAGERLY STAMP at commit.
    for a in [100, 700, 1400] {
        seq += 1;
        e.execute_text(seq, &format!("DELETE FROM big WHERE a = {a}"))
            .unwrap();
    }
    assert_eq!(e.streaming_cold_stamps(), 3, "three rows stamped");
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        0,
        "no rebuild for pure deletes"
    );

    // Aggregate through stamped chunks.
    assert_eq!(
        e.execute_relational_select(&q_count)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N) - 3)]]
    );
    // VALUE-SENSITIVE reads through the stamped chunks (COUNT alone cannot catch a mask on the
    // WRONG slot): SUM must reflect exactly WHICH rows are masked (closed form), and a bounded
    // window projection around a deleted row must return exactly the surviving neighbors (small
    // survivor set — no honest-defer; an unbounded ORDER BY here would defer to the CPU and
    // test nothing).
    let expected_sum: i64 = (0..i64::from(N)).sum::<i64>() - 100 - 700 - 1400;
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]],
        "SUM through stamped chunks must miss exactly the deleted rows"
    );
    let q_window = select("SELECT a FROM big WHERE a >= 98 AND a <= 102");
    let got = e
        .execute_relational_select(&q_window)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        got,
        vec![
            vec![SqlValue::Int4(98)],
            vec![SqlValue::Int4(99)],
            vec![SqlValue::Int4(101)],
            vec![SqlValue::Int4(102)],
        ],
        "the window around the deleted row must skip EXACTLY it"
    );
}

/// Mixed workload over a stamped entry: a DELETE stamps, an INSERT tail-patches with the stamped
/// chunk REUSED (its sidecar preserved), an UPDATE (same-id version chain) falls to the REBUILD
/// arm — every step correct, and a stamped entry DECLINES the P1 cold checkpoint (the v1
/// artifact has no sidecar sections; benign skip, ledgered as P2b).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_sidecar_mixed_workload_and_v2_artifact_roundtrip() {
    // A STORE-DRIVEN-era gate (P2 stamps + the P2b artifact round-trip against a replayed twin
    // whose id space must match): the class would shift ids via skipped installs mid-test.
    let _class_off = ClassEntryDisabled::new();
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 900;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let q_count = select("SELECT COUNT(*) FROM big");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    assert_eq!(count(&e), i64::from(N));

    // DELETE -> stamp.
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 10").unwrap();
    assert_eq!(e.streaming_cold_stamps(), 1);
    assert_eq!(count(&e), i64::from(N) - 1);

    // INSERT -> pure tail patch; the STAMPED chunk is REUSED (stamps preserved, still masked).
    let rebuilt_before = e.streaming_cold_chunks_rebuilt();
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a, b) VALUES (100000, 1)")
        .unwrap();
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        rebuilt_before,
        "INSERT stays a pure tail append beside a stamped chunk"
    );
    assert_eq!(
        count(&e),
        i64::from(N),
        "tail row visible AND the stamp still masks"
    );

    // UPDATE (same-id version chain change) -> the classifier must refuse the stamp downgrade;
    // the rebuild arm serves it. Correctness is the assert; the arm split is the counter.
    seq += 1;
    e.execute_text(seq, "UPDATE big SET b = -7 WHERE a = 20")
        .unwrap();
    assert_eq!(count(&e), i64::from(N), "update preserves cardinality");
    let q_probe = select("SELECT b FROM big WHERE a = 20");
    assert_eq!(
        e.execute_relational_select(&q_probe)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int4(-7)]],
        "the updated value must be visible through the streaming read"
    );

    // P2b: a sidecar-bearing entry now QUALIFIES for the v2 artifact — and the sidecar
    // ROUND-TRIPS: a twin engine at the same commit boundary restores the entry and the
    // stamped rows STAY MASKED (no store consultation, no rebuild).
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 30").unwrap();
    assert!(
        e.streaming_cold_stamps() >= 2,
        "premise: the entry carries a sidecar"
    );
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-p2b-ckpt-roundtrip-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    let boundary = e.committed_seq();
    let written = e
        .write_streaming_cold_checkpoint(&base, 1, boundary, boundary + 1)
        .expect("capture runs");
    assert_eq!(
        written, 1,
        "the v2 artifact must carry the sidecar-bearing entry"
    );

    // The twin replays the identical statement history (same commit boundary), restores the
    // artifact directly, and its FIRST streaming read replays the stamped bytes.
    let mut twin = Engine::new_local_cpu_oracle();
    let mut twin_seq = 0u64;
    if !gpu_available(&mut twin, &mut twin_seq) {
        return;
    }
    twin_seq += 1;
    twin.execute_text(twin_seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    twin_seq += 1;
    twin.execute_text(twin_seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    for statement in [
        "DELETE FROM big WHERE a = 10",
        "INSERT INTO big (a, b) VALUES (100000, 1)",
        "UPDATE big SET b = -7 WHERE a = 20",
        "DELETE FROM big WHERE a = 30",
    ] {
        twin_seq += 1;
        twin.execute_text(twin_seq, statement).unwrap();
    }
    assert_eq!(
        twin.committed_seq(),
        boundary,
        "premise: the twin reached the artifact boundary"
    );
    let restored = twin.restore_streaming_cold_checkpoint(&base, 1);
    assert_eq!(
        restored, 1,
        "the twin must restore the sidecar-bearing entry"
    );
    twin.set_relational_residency_budget_bytes(0, 4096);
    assert_eq!(
        count(&twin),
        i64::from(N) - 1,
        "restored stamps still mask (a=10, a=30 gone; tail row present)"
    );
    assert_eq!(
        twin.streaming_cold_builds(),
        0,
        "the restore IS the build — no scan"
    );
    assert!(twin.streaming_cold_hits() >= 1);

    // AUDIT LOW (adopted): a POST-RESTORE delete pins the persisted `payload_copin_s` — the
    // stamp's slot is the id's rank among ids visible at the PAYLOAD boundary; a restore that
    // defaulted the boundary to the seam would exclude the already-stamped rows from the rank,
    // shift the slot, and mask the WRONG row. COUNT is slot-blind; the closed-form SUM bites.
    twin_seq += 1;
    twin.execute_text(twin_seq, "DELETE FROM big WHERE a = 40")
        .unwrap();
    assert!(
        twin.streaming_cold_stamps() >= 1,
        "the post-restore delete must STAMP"
    );
    assert_eq!(count(&twin), i64::from(N) - 2);
    let expected_sum: i64 = (0..i64::from(N)).sum::<i64>() - 10 - 30 - 40 + 100000;
    let sum = twin
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]],
        "the post-restore stamp must mask EXACTLY a=40 (payload-boundary rank)"
    );
}

/// THE CHANGE-LOG REGRESSION (found by P2's stamp counter, but 6c-1-era): `imbl::OrdMap::diff`
/// MISSED a real change — three sequential single-row deletes, each against a freshly pinned
/// generation (the cold-tier entry's exact usage), and the THIRD delete vanished from the diff
/// while the two generations' chains provably differed (deleted_by None vs Some). A missed delta
/// = a patched cold entry silently serving a deleted row. `changed_tuple_ids` therefore reads
/// the store's WRITE-SIDE CHANGE LOG (exact by construction) and structural diffing is BANNED
/// for correctness-bearing deltas. This is the minimal CPU repro, pinned forever.
#[test]
fn cow_change_log_reports_every_pinned_generation_delta() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    e.execute_text(2, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    let g0 = e.read_state.mvcc.table_rows("big").generation_payload();
    e.execute_text(3, "DELETE FROM big WHERE a = 100").unwrap();
    let g1 = e.read_state.mvcc.table_rows("big").generation_payload();
    assert_eq!(g0.rows.changed_tuple_ids(&g1.rows), vec![101]);
    drop(g0);
    e.execute_text(4, "DELETE FROM big WHERE a = 700").unwrap();
    let g2 = e.read_state.mvcc.table_rows("big").generation_payload();
    assert_eq!(g1.rows.changed_tuple_ids(&g2.rows), vec![701]);
    drop(g1);
    e.execute_text(5, "DELETE FROM big WHERE a = 1400").unwrap();
    let g3 = e.read_state.mvcc.table_rows("big").generation_payload();
    // The chains provably differ...
    assert_eq!(g2.rows.chain(1401).map(|c| c[0].deleted_by), Some(None));
    assert_eq!(g3.rows.chain(1401).map(|c| c[0].deleted_by), Some(Some(5)));
    // ...and the delta MUST say so (the imbl structural diff returned [] here).
    assert_eq!(
        g2.rows.changed_tuple_ids(&g3.rows),
        vec![1401],
        "the pinned-generation delta must report the third delete"
    );
}

// ========== P4-1 (chunk-authoritative tables): the REVERSE GATHER ==========

/// The round-trip differential that gates the host columnar decoder: a mixed-type NULL-bearing
/// table streams into cold chunks; the reverse gather must reproduce EXACTLY the store's visible
/// rows (order included — chunk order is scan order), across every section type (int4/date/int2
/// i32, int8/timestamp i64, numeric/uuid b128, bool bitmaps, text blobs, NULL validity bitmaps),
/// and honor the P2 sidecar with the kernel's semantics after a stamped delete.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_reverse_gather_round_trips_all_types_and_sidecars() {
    let mut e = Engine::new_local_cpu_oracle();
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
    const N: i32 = 400;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        // Audit LOW: a NULL in EVERY nullable section type (i32/i64/b128/bool/text/uuid paths).
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
            format!("{}", i64::from(i) * 1_000_000_007)
        };
        let s16 = if i % 13 == 0 {
            "NULL".into()
        } else {
            format!("{}", i % 300 - 150)
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
            "({i}, {s16}, {big}, '2024-{:02}-{:02}', '2024-01-01 00:{:02}:{:02}', {n}, {flag}, {t}, {u})",
            1 + (i % 12),
            1 + (i % 28),
            i % 60,
            (i * 7) % 60,
        ));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO mix VALUES {values}"))
        .unwrap();

    // The ORACLE: the store's visible rows BEFORE streaming (host path, scan order).
    let q = select("SELECT a, s, big, d, ts, n, flag, t, u FROM mix");
    let oracle = e
        .execute_relational_select(&q)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(oracle.len(), N as usize);

    // Stream -> cold chunks; reverse-gather at the current boundary.
    e.set_relational_residency_budget_bytes(0, 4096);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM mix"))
        .unwrap();
    assert!(
        e.streaming_cold_builds() >= 1,
        "premise: a cold entry exists"
    );
    let gathered = e
        .reverse_gather_streamed_rows("mix", e.committed_seq())
        .expect("cold entry present")
        .expect("decode succeeds");
    assert_eq!(
        gathered, oracle,
        "the reverse gather must reproduce the store rows exactly"
    );

    // A stamped DELETE: the gather at the current boundary must exclude EXACTLY that row.
    seq += 1;
    e.execute_text(seq, "DELETE FROM mix WHERE a = 42").unwrap();
    assert!(
        e.streaming_cold_stamps() >= 1,
        "premise: the delete STAMPED"
    );
    let gathered = e
        .reverse_gather_streamed_rows("mix", e.committed_seq())
        .expect("cold entry present")
        .expect("decode succeeds");
    let want: Vec<Vec<SqlValue>> = oracle
        .iter()
        .filter(|row| row[0] != SqlValue::Int4(42))
        .cloned()
        .collect();
    assert_eq!(
        gathered, want,
        "the sidecar mask must apply kernel-identically on the host"
    );
}

// ========== P4-2a (chunk-authoritative tables): chunk-native locate + locate-driven stamp ==========

/// THE LOCATE DIFFERENTIAL: the chunk-native locate (device predicate over the chunks themselves,
/// slots back) must select EXACTLY the rows the store-driven P3 locate selects for the same
/// predicate — compared by ROW VALUES (slots translate to rows through the P4-1 decoder: on an
/// unstamped entry, decoded[slot] IS the slot's row).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_native_locate_matches_store_locate() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    // Build the cold entry (multi-chunk).
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
        .unwrap();
    assert!(e.streaming_cold_builds() >= 1);

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("big")
        .cloned()
        .unwrap();
    // A range + OR shape (value-index-unbindable): a > 1200 OR b < 100.
    let filter_groups: Vec<Vec<(usize, SelectFilterOp, SqlValue)>> = vec![
        vec![(0, SelectFilterOp::Gt, SqlValue::Int4(1200))],
        vec![(1, SelectFilterOp::Lt, SqlValue::Int4(100))],
    ];
    let predicate =
        crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(&table, &filter_groups)
            .expect("lowerable");
    let rtx = e.committed_seq();
    let located = e
        .locate_streaming_cold_slots(&table, &predicate, rtx)
        .expect("locate serves");
    // Translate (chunk, slot) -> row values through the P4-1 decoder (unstamped: slot-addressed).
    let entry_rows: Vec<Vec<Vec<SqlValue>>> = {
        let map = e.read_state.residency.streaming_cold_chunks.load();
        let entry = map.get("big").unwrap();
        entry
            .chunks
            .iter()
            .map(|c| crate::engine_streaming_exec::decode_cold_chunk_rows(&table, c, rtx).unwrap())
            .collect()
    };
    let mut got: Vec<Vec<SqlValue>> = located
        .iter()
        .flat_map(|(chunk_idx, slots)| {
            slots
                .iter()
                .map(|slot| entry_rows[*chunk_idx][*slot as usize].clone())
        })
        .collect();
    // The store-driven P3 locate on the SAME pinned view.
    let visibility = StorageVisibility { read_txn_id: rtx };
    let table_rows = e.read_state.mvcc.table_rows("big");
    let mut want: Vec<Vec<SqlValue>> = e
        .try_streaming_dml_locate(&table, &filter_groups, visibility, &table_rows)
        .expect("store locate serves")
        .into_iter()
        .map(|(_, _, row)| row)
        .collect();
    got.sort();
    want.sort();
    assert_eq!(got.len(), 349, "1201..=1499 (299) + b<100 => a<50 (50)");
    assert_eq!(
        got, want,
        "chunk-native locate == store-driven locate (row values)"
    );
}

/// THE STAMP ISOLATION GATE: locate coordinates on the chunks, stamp them at the current
/// boundary, and every chunk-served view (streaming COUNT/SUM, the reverse gather) must exclude
/// exactly those rows — WITHOUT any store write (the entry's generation is untouched; this is the
/// store-free write primitive in isolation). Already-stamped slots must NOT re-locate (the
/// sidecar visibility composes into the locate).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_locate_driven_stamp_masks_rows_without_store() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 2));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
        .unwrap();

    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("big")
        .cloned()
        .unwrap();
    let filter_groups: Vec<Vec<(usize, SelectFilterOp, SqlValue)>> =
        vec![vec![(0, SelectFilterOp::Gte, SqlValue::Int4(1490))]];
    let predicate =
        crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(&table, &filter_groups)
            .expect("lowerable");
    let rtx = e.committed_seq();
    let located = e
        .locate_streaming_cold_slots(&table, &predicate, rtx)
        .expect("locate serves");
    let located_count: usize = located.iter().map(|(_, s)| s.len()).sum();
    assert_eq!(located_count, 10, "a in 1490..=1499");

    assert!(
        e.stamp_streaming_cold_slots("big", &located, rtx, false),
        "the stamp must install"
    );
    assert_eq!(e.streaming_cold_stamps(), 10);

    // Chunk-served views exclude the stamped rows; the STORE was never written.
    let q_count = select("SELECT COUNT(*) FROM big");
    assert_eq!(
        e.execute_relational_select(&q_count)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(i64::from(N) - 10)]],
        "streaming COUNT masks the stamped rows"
    );
    let expected_sum: i64 = (0..i64::from(N) - 10).sum();
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]],
        "streaming SUM masks EXACTLY the stamped rows"
    );
    let gathered = e
        .reverse_gather_streamed_rows("big", e.committed_seq())
        .unwrap()
        .unwrap();
    assert_eq!(
        gathered.len(),
        (N - 10) as usize,
        "the reverse gather agrees"
    );

    // Idempotence of visibility: re-locating the same predicate finds NOTHING (the sidecar mask
    // composes into the locate — stamped slots are invisible to it).
    let relocated = e
        .locate_streaming_cold_slots(&table, &predicate, rtx)
        .expect("locate serves");
    assert!(
        relocated.is_empty(),
        "already-stamped slots must not re-locate (got {relocated:?})"
    );
}

// ========== P4-2b-i (S-E.P4): the CHUNK-AUTHORITATIVE class — enter, freeze, stream, exit ==========

/// THE CLASS LIFECYCLE GATE: an over-budget, elision-INeligible (text-bearing), keyless FK-free
/// table ENTERS the class at a commit; subsequent INSERTs skip the host store (FROZEN — proven by
/// the store's version count) while the streamed reads see every row (the tail appends are the
/// materialization); an unstreamable read DE-AUTHORITIZES (the post-freeze delta replays into the
/// store) and the host path serves exactly the full data. Differential twin throughout.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_enters_freezes_streams_and_deauths() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    let mut twin = Engine::new_local_cpu_oracle();
    let mut twin_seq = 0u64;

    const N: i32 = 1200;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, 'txt{:04}')", i % 500));
    }
    for (engine, s) in [(&mut e, &mut seq), (&mut twin, &mut twin_seq)] {
        *s += 1;
        engine
            .execute_text(*s, "CREATE TABLE facts (a INT, t TEXT)")
            .unwrap();
        *s += 1;
        engine
            .execute_text(*s, &format!("INSERT INTO facts (a, t) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM facts");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    // Build the cold entry, then the ENTER commit (the eager patch makes the entry fresh at it).
    assert_eq!(count(&e), i64::from(N));
    assert_eq!(e.chunk_class_entries(), 0);
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    twin_seq += 1;
    twin.execute_text(
        twin_seq,
        "INSERT INTO facts (a, t) VALUES (100000, 'enter')",
    )
    .unwrap();
    assert_eq!(
        e.chunk_class_entries(),
        1,
        "the table must ENTER the class at this commit"
    );

    // RECLAIMED (P4): class entry DELETED the host chains — the store-deletion payoff; the
    // chunks are the representation. The count below is 0 and stays 0 through every class write.
    assert!(
        e.chunk_class_reclaimed_rows() > 0,
        "entry must reclaim the host rows"
    );
    let frozen_versions = e
        .read_state
        .mvcc
        .table_rows("facts")
        .store()
        .all_versions()
        .len();
    assert_eq!(frozen_versions, 0, "the class table's host chains are GONE");
    for k in 0..5 {
        seq += 1;
        e.execute_text(
            seq,
            &format!(
                "INSERT INTO facts (a, t) VALUES ({}, 'tail{k}')",
                200000 + k
            ),
        )
        .unwrap();
        twin_seq += 1;
        twin.execute_text(
            twin_seq,
            &format!(
                "INSERT INTO facts (a, t) VALUES ({}, 'tail{k}')",
                200000 + k
            ),
        )
        .unwrap();
    }
    assert_eq!(
        e.chunk_class_skipped_installs(),
        5,
        "five commits skipped the host install"
    );
    assert_eq!(
        e.read_state
            .mvcc
            .table_rows("facts")
            .store()
            .all_versions()
            .len(),
        frozen_versions,
        "the store is FROZEN at the class boundary"
    );
    assert_eq!(
        count(&e),
        i64::from(N) + 6,
        "the streamed read sees every tail row"
    );
    assert_eq!(e.chunk_class_deauths(), 0, "no exit yet");

    // A value-sensitive streamed read through the tails (SUM over a).
    let expected_sum: i64 =
        (0..i64::from(N)).sum::<i64>() + 100000 + (0..5).map(|k| 200000 + k).sum::<i64>();
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM facts"))
        .unwrap();
    assert_eq!(
        sum.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Int8(expected_sum)]]
    );

    // DE-AUTH: clear the budget — streaming deactivates, the CPU-pinned guard replays the
    // post-freeze delta into the store, the class exits, and the host path serves EVERYTHING.
    e.clear_relational_residency_budget_bytes(0);
    let q_rows = select("SELECT a, t FROM facts ORDER BY a");
    let got = e
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        e.chunk_class_deauths(),
        1,
        "the unstreamable read exited the class LOUDLY"
    );
    let want = twin
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got.len(), (N + 6) as usize);
    assert_eq!(
        got, want,
        "post-de-auth host reads == the never-classed twin"
    );
    assert!(
        e.read_state
            .mvcc
            .table_rows("facts")
            .store()
            .all_versions()
            .len()
            > frozen_versions,
        "the delta replayed into the store"
    );

    // Post-exit writes are plain store writes again.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 100000")
        .unwrap();
    twin_seq += 1;
    twin.execute_text(twin_seq, "DELETE FROM facts WHERE a = 100000")
        .unwrap();
    let got = e
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let want = twin
        .execute_relational_select(&q_rows)
        .unwrap()
        .rows
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(got, want);
}

/// P4-2b-ii — CLASS DML STAYS CLASSED: DELETE stamps the chunk-native coordinates (no de-auth,
/// no store touch); UPDATE stamps the old versions and tail-appends the new images; every
/// streamed read reflects them exactly (closed-form SUM); an UNLOWERABLE predicate still falls
/// back to the loud de-auth exit.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_dml_stamps_without_deauth() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT)")
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
    e.execute_text(seq, &format!("INSERT INTO facts (a, t) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM facts");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    let sum_a = |e: &Engine| -> i64 {
        match e
            .execute_relational_select(&select("SELECT SUM(a) FROM facts"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        }
    };
    let _ = count(&e);
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (100000, 'enter')")
        .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // CLASS DELETE: a range WHERE resolves from the chunks (spanning the base AND the tail-
    // absorbed enter row: a=100000 also matches), stamps, and STAYS CLASSED.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a >= 1195")
        .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "the class DELETE must NOT de-auth"
    );
    assert_eq!(
        e.streaming_cold_stamps(),
        6,
        "rows 1195..=1199 AND a=100000 stamped"
    );
    assert_eq!(count(&e), i64::from(N) - 5);
    let expected_sum: i64 = (0..1195i64).sum::<i64>();
    assert_eq!(
        sum_a(&e),
        expected_sum,
        "SUM reflects EXACTLY the stamped rows"
    );

    // CLASS UPDATE: stamp-old + tail-append-new, still classed.
    let stamps_before = e.streaming_cold_stamps();
    seq += 1;
    e.execute_text(seq, "UPDATE facts SET a = -7 WHERE a = 1000")
        .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        0,
        "the class UPDATE must NOT de-auth"
    );
    assert!(
        e.streaming_cold_stamps() > stamps_before,
        "the old version stamped"
    );
    assert_eq!(count(&e), i64::from(N) - 5, "cardinality preserved");
    assert_eq!(
        sum_a(&e),
        expected_sum - 1000 - 7,
        "the new image replaced the old in every streamed read"
    );

    // The class survives further INSERTs after DML.
    let skipped_before = e.chunk_class_skipped_installs();
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (500000, 'post')")
        .unwrap();
    assert!(
        e.chunk_class_skipped_installs() > skipped_before,
        "still classed"
    );
    assert_eq!(sum_a(&e), expected_sum - 1000 - 7 + 500000);

    // The H2 DDL SWEEP exit: any non-DML statement de-authoritizes every class table BEFORE its
    // preflight reads the store — the replayed store must be MVCC-whole (tails inserted at their
    // born boundaries, every post-freeze stamp applied as a tombstone).
    seq += 1;
    e.execute_text(seq, "CREATE TABLE zzz (x INT)").unwrap();
    assert_eq!(e.chunk_class_deauths(), 1, "the DDL sweep exits the class");
    assert_eq!(
        count(&e),
        i64::from(N) - 4,
        "the de-authed store serves the exact post-DML state (stamps replayed, tails present)"
    );
    assert_eq!(
        sum_a(&e),
        expected_sum - 1000 - 7 + 500000,
        "value-exact after the exit"
    );
}

/// P4-3 — THE BORN GATE: a reader boundary below a tail chunk's born commit must not see its
/// rows (the reverse gather and the chunk-native locate are the directly-drivable surfaces; the
/// fold replay shares the same skip). Sidecar stamps above the boundary keep rows visible —
/// exact per-reader MVCC over the chunks.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_born_gate_serves_old_boundaries() {
    let mut e = Engine::new_local_cpu_oracle();
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
    let freeze = e.table_chunk_authoritative("facts").expect("classed");

    // Two post-freeze commits: a tail INSERT, then a class DELETE (a sidecar stamp).
    seq += 1;
    e.execute_text(seq, "INSERT INTO facts (a, t) VALUES (200000, 'tail')")
        .unwrap();
    let after_insert = e.committed_seq();
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 5")
        .unwrap();
    let after_delete = e.committed_seq();

    // The reverse gather AT THE FREEZE: no tails, no post-freeze stamps applied (a=5 visible).
    let at_freeze = e
        .reverse_gather_streamed_rows("facts", freeze)
        .unwrap()
        .unwrap();
    assert_eq!(
        at_freeze.len(),
        (N + 1) as usize,
        "the freeze boundary sees base + enter only"
    );
    assert!(
        at_freeze.iter().any(|r| r[0] == SqlValue::Int4(5)),
        "the pre-delete boundary still sees a=5"
    );

    // At the post-insert boundary: the tail row appears; a=5 still visible (its stamp is later).
    let mid = e
        .reverse_gather_streamed_rows("facts", after_insert)
        .unwrap()
        .unwrap();
    assert_eq!(mid.len(), (N + 2) as usize);
    assert!(mid.iter().any(|r| r[0] == SqlValue::Int4(200000)));
    assert!(mid.iter().any(|r| r[0] == SqlValue::Int4(5)));

    // At the current boundary: the stamp masks a=5.
    let now = e
        .reverse_gather_streamed_rows("facts", after_delete)
        .unwrap()
        .unwrap();
    assert_eq!(now.len(), (N + 1) as usize);
    assert!(!now.iter().any(|r| r[0] == SqlValue::Int4(5)));

    // The chunk-native LOCATE born gate: a predicate matching ONLY the tail row finds it at the
    // current boundary and NOTHING at the freeze boundary.
    let table = e
        .catalog_snapshot()
        .relational_catalog
        .get("facts")
        .cloned()
        .unwrap();
    let groups: Vec<Vec<(usize, SelectFilterOp, SqlValue)>> =
        vec![vec![(0, SelectFilterOp::Eq, SqlValue::Int4(200000))]];
    let predicate =
        crate::engine_dml_prepare::dml_filter_groups_to_device_predicate(&table, &groups).unwrap();
    let now_hits = e
        .locate_streaming_cold_slots(&table, &predicate, after_delete)
        .expect("locate serves");
    assert_eq!(now_hits.iter().map(|(_, s)| s.len()).sum::<usize>(), 1);
    let frozen_hits = e
        .locate_streaming_cold_slots(&table, &predicate, freeze)
        .expect("locate serves");
    assert!(
        frozen_hits.is_empty(),
        "a tail row is invisible to the freeze boundary"
    );
}

/// P4 COMPACTION: a class DELETE that kills most of a chunk triggers the in-install survivor
/// rebuild — the sidecar and dead slots are physically deleted, and every read stays
/// value-exact through the compacted chunk (closed-form SUM + the de-auth exit differential).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_compaction_deletes_dead_slots() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    // Audit note adopted: a NULLABLE b128 column rides the compaction round-trip (the survivor
    // gather + re-encode must preserve NULL validity and numeric mantissas, not just int/text).
    e.execute_text(seq, "CREATE TABLE facts (a INT, t TEXT, n NUMERIC(10,2))")
        .unwrap();
    const N: i32 = 900;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let n = if i % 5 == 0 {
            "NULL".to_string()
        } else {
            format!("{i}.25")
        };
        values.push_str(&format!("({i}, 'txt{:04}', {n})", i % 500));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO facts (a, t, n) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 8192);
    let q_count = select("SELECT COUNT(*) FROM facts");
    let count = |e: &Engine| -> i64 {
        match e.execute_relational_select(&q_count).unwrap().rows.row(0)[0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("count: {other:?}"),
        }
    };
    let sum_a = |e: &Engine| -> i64 {
        match e
            .execute_relational_select(&select("SELECT SUM(a) FROM facts"))
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("sum: {other:?}"),
        }
    };
    let _ = count(&e);
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO facts (a, t, n) VALUES (100000, 'enter', 7.75)",
    )
    .unwrap();
    assert_eq!(e.chunk_class_entries(), 1);

    // Kill MOST of the first chunk's rows (a < 200 spans it): the stamp install must COMPACT.
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a < 200")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0, "stays classed");
    assert!(
        e.chunk_class_compactions() >= 1,
        "the heavily-stamped chunk must compact in the same install"
    );
    assert!(
        e.chunk_class_compacted_slots() >= 150,
        "the dead slots are physically deleted (got {})",
        e.chunk_class_compacted_slots()
    );

    // Value-exact through the compacted chunk.
    assert_eq!(count(&e), i64::from(N) - 200 + 1);
    let expected_sum: i64 = (200..i64::from(N)).sum::<i64>() + 100000;
    assert_eq!(
        sum_a(&e),
        expected_sum,
        "SUM through the compacted chunk is exact"
    );
    // The nullable numeric column survived compaction value-exactly: SUM(n) over survivors
    // (i.25 for i in 200..900 where i % 5 != 0) + the enter row's 7.75.
    let mantissa_sum: i128 = (200..i128::from(N))
        .filter(|i| i % 5 != 0)
        .map(|i| i * 100 + 25)
        .sum::<i128>()
        + 775;
    let sum_n = e
        .execute_relational_select(&select("SELECT SUM(n) FROM facts"))
        .unwrap();
    assert_eq!(
        sum_n.rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![SqlValue::Numeric(gpu_db_sql::Decimal128::new(
            mantissa_sum,
            2
        ))]],
        "the NULL-bearing numeric column round-tripped compaction exactly"
    );

    // Post-compaction DML + the exit both stay correct (coordinates re-slotted: the NEXT delete
    // locates against the fresh epoch).
    seq += 1;
    e.execute_text(seq, "DELETE FROM facts WHERE a = 500")
        .unwrap();
    assert_eq!(e.chunk_class_deauths(), 0);
    assert_eq!(sum_a(&e), expected_sum - 500);
    seq += 1;
    e.execute_text(seq, "CREATE TABLE zzz2 (x INT)").unwrap(); // the DDL-sweep exit
    assert_eq!(e.chunk_class_deauths(), 1);
    assert_eq!(
        sum_a(&e),
        expected_sum - 500,
        "the de-authed store is value-exact"
    );
}

/// P5-0 — THE DEVICE SLOT RECHECK differential: for every slot of a staged mixed-type
/// NULL-bearing chunk, the single-slot device materialization must equal the P4-1 host
/// decoder's row exactly, and the sidecar/born masks must agree (a stamped slot returns
/// Some(None) at-or-above its stamp and the live row below it).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_device_slot_recheck_matches_host_decoder() {
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
    let mut e = Engine::new_local_cpu_oracle();
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
        e.chunk_class_reclaimed_rows() > 0,
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
    let skipped_before = e.chunk_class_skipped_installs();
    seq += 1;
    e.execute_text(seq, "INSERT INTO ku (a, t) VALUES (600000, 'fresh')")
        .unwrap();
    assert!(
        e.chunk_class_unique_probes() > probes_before,
        "the accept path went through the device probe"
    );
    assert!(
        e.chunk_class_skipped_installs() > skipped_before,
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
    let mut e = Engine::new_local_cpu_oracle();
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
        .execute_text(seq, "INSERT INTO cc VALUES (100002, 100000, -1, 'bad-check')")
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
    let inbound_error = e
        .execute_text(seq, "DELETE FROM cp WHERE id = 100003")
        .expect_err("referenced provider delete must reject");
    assert!(format!("{inbound_error:?}").contains("foreign key constraint"));
    assert!(e.chunk_class_device_exact_rechecks() > exact_before);
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
}

/// P5-later — OVER-CAP KEYED CLASS: when the complete retained exact-index set cannot co-reside,
/// compact per-chunk Bloom filters still admit the class. The GPU Bloom probe only chooses
/// chunks; exact predicate + visibility remains authoritative for duplicate checks and point DML.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_over_cap_bloom_candidates_stay_exact() {
    let _forced_over_cap = ChunkKeyIndexCapOverride::tiny();
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kb (a INT PRIMARY KEY, u INT UNIQUE, v INT)")
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
    assert_eq!(e.chunk_class_entries(), 1, "over-cap keyed table enters via Bloom set");
    assert!(e.table_chunk_authoritative("kb").is_some());
    assert!(e.chunk_class_reclaimed_rows() > 0, "host row chains were reclaimed");

    let bloom_0 = e.chunk_key_bloom_probes();
    let exact_0 = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (1100, 999999, 1)")
        .expect_err("base-chunk duplicate must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert!(e.chunk_key_bloom_probes() > bloom_0, "candidate decision ran on GPU Bloom");
    assert!(e.chunk_class_device_exact_rechecks() > exact_0, "conflict was exactly rechecked");
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

    // Structural NULL uniqueness also uses the Bloom candidate set and exact IS NULL predicate.
    seq += 1;
    e.execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (200000, NULL, 3)")
        .unwrap();
    let exact_2 = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let err = e
        .execute_text(seq, "INSERT INTO kb (a, u, v) VALUES (200001, NULL, 4)")
        .expect_err("second NULL unique key must reject");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert!(e.chunk_class_device_exact_rechecks() > exact_2);

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
    assert_eq!(e.chunk_class_deauths(), 0, "Bloom route stays chunk-authoritative");
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
    assert!(e.chunk_class_compactions() > 0, "range delete compacts a keyed chunk");
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
        bloom_ids_before.difference(&bloom_ids_after).next().is_some(),
        "compaction publication purges the replaced chunk-id Bloom"
    );
    let row = e
        .execute_relational_select(&select("SELECT v FROM kb WHERE a = 1100"))
        .unwrap();
    assert_eq!(row.rows.clone().into_boxed(), vec![vec![SqlValue::Int4(4242)]]);
}

/// The Bloom cap is global. A second table that cannot reserve a complete set must roll back every
/// partial buffer, leave the already-authoritative first table intact, and refuse class entry.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_bloom_global_cap_rolls_back_failed_admission() {
    let _forced_over_cap = ChunkKeyIndexCapOverride::tiny();
    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(seq, &format!("INSERT INTO bcb (a, v) VALUES {}", rows(200000)))
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
    seq += 1;
    e.execute_text(seq, "INSERT INTO bcb (a, v) VALUES (300000, 2)")
        .unwrap();
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
        let mut e = Engine::new_local_cpu_oracle();
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
        assert!(e.streaming_cold_spills() > 0, "fixture must be spill-backed");
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

/// Deterministic E1-prime/E2-publication race: the cold read publishes E1 and pauses before its
/// off-lock candidate build; a concurrent UPDATE replaces chunks and publishes E2; E1 then builds.
/// Final epoch validation must remove every now-stale E1 candidate inserted after E2's cleanup.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_bloom_offlock_prime_cannot_strand_stale_ids() {
    let _forced_over_cap = ChunkKeyIndexCapOverride::tiny();
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE kbr (a INT PRIMARY KEY, v INT)")
        .unwrap();
    let values = (0..1200)
        .map(|i| format!("({i}, {i})"))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO kbr (a, v) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let e = std::sync::Arc::new(e);
    let (published_e1, resume_prime) =
        crate::engine_streaming_exec::install_chunk_key_prime_pin_hook();
    let reader = std::sync::Arc::clone(&e);
    let capture = std::thread::spawn(move || {
        reader.execute_relational_select(&select("SELECT COUNT(*) FROM kbr"))
    });
    published_e1.wait();
    seq += 1;
    e.execute_text(seq, "UPDATE kbr SET v = 999999 WHERE a < 400")
        .unwrap();
    assert!(e.streaming_cold_patches() > 0, "interposed UPDATE published E2");
    resume_prime.wait();
    capture.join().expect("capture thread").expect("streaming read");
    assert_eq!(
        e.stale_chunk_key_candidate_count("kbr"),
        0,
        "E1 prime inserted after E2 cleanup must self-retire stale chunk IDs"
    );
}

/// Audit M2 — the temporary exact-predicate batch bound is an authorization seam, not a silent
/// acceptance seam. Exactly 256 fresh rows stay classed and are device-validated; 257 fresh rows
/// loudly de-authorize before succeeding through the restored host reference; and a separate
/// 257-row batch containing a duplicate likewise de-authorizes, then rejects with no partial land.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_unique_batch_bound_deauthorizes_257() {
    fn populate_and_enter(e: &mut Engine, seq: &mut u64, table: &str) {
        let entries_before = e.chunk_class_entries();
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

    let mut e = Engine::new_local_cpu_oracle();
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
    e.execute_text(seq, &format!("INSERT INTO bound_ok VALUES {values_257}"))
        .unwrap();
    assert_eq!(
        e.chunk_class_deauths(),
        deauth_before + 1,
        "257 rows must loudly de-authorize before the host reference accepts"
    );
    assert_eq!(count(&e, "bound_ok"), 600 + 1 + 256 + 257);

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
            &format!("INSERT INTO bound_dup VALUES {}", duplicate_values.join(",")),
        )
        .expect_err("the restored host reference must reject the 257-row duplicate");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(
        e.chunk_class_deauths(),
        deauth_before + 1,
        "the rejecting oversized batch also crosses the visible deauth seam"
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
    let mut e = Engine::new_local_cpu_oracle();
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

    let mut e = Engine::new_local_cpu_oracle();
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

/// P5 charter closure — NULL KEY: unique semantics are STRUCTURAL (NULL == NULL conflicts).
/// Fingerprints do not encode validity, so a NULL tuple bypasses the candidate index and runs
/// an exact `IS NULL` predicate over the chunks on-device. The first NULL succeeds while the
/// class stays authoritative; the second rejects from the device result.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_null_unique_stays_device_native() {
    let mut e = Engine::new_local_cpu_oracle();
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

    // A NULL key uses the raw-placeholder candidate fingerprint; a single NULL misses and passes.
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
        "the structural NULL miss came from the device candidate probe"
    );
    // The SECOND NULL: the same exact device predicate sees the live NULL and rejects.
    let exact_before = e.chunk_class_device_exact_rechecks();
    seq += 1;
    let err = e
        .execute_text(
            seq,
            "INSERT INTO kn (a, u, t) VALUES (2000001, NULL, 'null2')",
        )
        .expect_err("the second NULL is a device-detected structural dup");
    assert!(format!("{err:?}").contains("duplicate key value"));
    assert_eq!(e.chunk_class_deauths(), 0, "the class remains authoritative");
    assert!(
        e.chunk_class_device_exact_rechecks() > exact_before,
        "the matching NULL tuple was confirmed by an exact device predicate"
    );
}

/// Audit M1 — compound partial-NULL uniqueness through every class seam: distinct tuples in one
/// batch pass, an in-batch duplicate rejects on the transient device relation, an existing-row
/// duplicate rejects through the placeholder-fingerprint candidate probe + exact `IS NULL`, a
/// key-preserving UPDATE self-excludes, tombstone/reinsert succeeds, and WAL replay lands the
/// identical accepted history.
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
        let mut e = Engine::new_local_cpu_oracle();
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

        // Same NULL component, distinct second key: legal and device-validated in one batch.
        seq += 1;
        e.execute_text(
            seq,
            "INSERT INTO null_k VALUES (10000, NULL, 7, 1), (10001, NULL, 8, 2)",
        )
        .unwrap();
        // Exact duplicate inside one statement: transient device relation must reject it.
        let exact_before = e.chunk_class_device_exact_rechecks();
        seq += 1;
        let err = e
            .execute_text(
                seq,
                "INSERT INTO null_k VALUES (10002, NULL, 9, 3), (10003, NULL, 9, 4)",
            )
            .expect_err("partial-NULL in-batch duplicate");
        assert!(format!("{err:?}").contains("duplicate key value"));
        assert!(e.chunk_class_device_exact_rechecks() > exact_before);

        // Existing partial-NULL duplicate: candidate index + exact IS NULL rejects.
        seq += 1;
        let err = e
            .execute_text(seq, "INSERT INTO null_k VALUES (10004, NULL, 7, 5)")
            .expect_err("existing partial-NULL duplicate");
        assert!(format!("{err:?}").contains("duplicate key value"));

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
        let mut e = Engine::new_local_cpu_oracle();
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

    // Recovery replays the acked history through the HOST path — it must accept every acked
    // commit (C2) and land value-identical.
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

/// P5-2 — THE TRANSACTION PATH: an explicit-txn INSERT's ONLY unique guard is the preflight
/// (the commit-time de-auth runs AFTER it), so the class probe must reject a dup at statement
/// time inside BEGIN/COMMIT, and accept fresh keys.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_txn_insert_dup_rejected() {
    let mut e = Engine::new_local_cpu_oracle();
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

    // The dup rejects INSIDE the transaction (the preflight probe). A txn's statements all
    // carry the BEGIN's seq — the txn id.
    seq += 1;
    e.execute_text(seq, "BEGIN").unwrap();
    let err = e
        .execute_text(seq, "INSERT INTO kt (a, t) VALUES (500, 'dup')")
        .expect_err("txn dup must reject at preflight");
    assert!(format!("{err:?}").contains("duplicate key value"));
    e.execute_text(seq, "ROLLBACK").unwrap();

    // A fresh key commits through the txn path.
    seq += 1;
    e.execute_text(seq, "BEGIN").unwrap();
    e.execute_text(seq, "INSERT INTO kt (a, t) VALUES (700000, 'fresh')")
        .unwrap();
    e.execute_text(seq, "COMMIT").unwrap();
    let q = select("SELECT COUNT(*) FROM kt WHERE a = 700000");
    let got = match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
        SqlValue::Int8(n) => n,
        ref other => panic!("count: {other:?}"),
    };
    assert_eq!(got, 1, "the txn insert landed");
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
        let mut e = Engine::new_local_cpu_oracle();
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

    // C2: the acked text-key history must replay cleanly through the host path.
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
    let mut e = Engine::new_local_cpu_oracle();
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
        let _ = e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, v) VALUES (100000, -1)"),
        )
        .unwrap();
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
/// deleting enough rows to compact the chunk (including the paused DELETE's key), then resume. The E1
/// write-set must conflict with E2; reloading E2 under the old snapshot would born-skip the
/// compacted chunk, miss the key, and let the stale DELETE erase the concurrent update.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_offlock_prepare_pins_one_entry_across_compaction() {
    let mut engine = Engine::new_local_cpu_oracle();
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
    let (pinned, resume) =
        crate::engine_streaming_exec::install_class_resolve_pin_hook();
    let deleting = std::sync::Arc::clone(&engine);
    let delete_seq = seq + 1;
    let delete = std::thread::spawn(move || {
        deleting.execute_dml_concurrent(delete_seq, "DELETE FROM epoch_t WHERE v = 100")
    });
    pinned.wait(); // DELETE pinned E1 and its read snapshot; it has not located yet.

    let update_seq = seq + 2;
    engine
        .execute_text(
            update_seq,
            "DELETE FROM epoch_t WHERE a < 200 OR a = 900",
        )
        .unwrap();
    assert!(
        engine.chunk_class_compactions() > 0,
        "the interposed write must publish a compacted E2"
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
}

/// P5-3 (audit LOW) — COMPOUND-KEY DML through the probe: the by-key locate on a (a, b) PK
/// rides the FOLDED fingerprint needle (not the raw-i32 fast path) — a needle/build divergence
/// here is a silently MISSED DML match (lost delete/update), so the probe twin (Eq on both key
/// columns) differentials against the fold twin (range predicates) over the identical history.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_chunk_class_keyed_compound_dml_key_locate() {
    let mut e = Engine::new_local_cpu_oracle();
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
        let _ = e
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        seq += 1;
        e.execute_text(
            seq,
            &format!("INSERT INTO {table} (a, b, v) VALUES (100000, 0, -1)"),
        )
        .unwrap();
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
