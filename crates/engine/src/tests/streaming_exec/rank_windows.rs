use super::{gpu_available, select, ClassEntryDisabled};
use crate::{Engine, RelationalSelectResult};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;
use std::sync::Arc;

/// GPU rank windows over an over-budget input: the streaming ordered fold produces the bounded
/// window order, the device rank kernel assigns ROW_NUMBER/RANK/DENSE_RANK, and the final OFFSET/LIMIT
/// is another device window. An unrelated projected NULL column gates NULL-safe materialization.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_rank_windows_over_ordered_input() {
    let _entry_disabled = ClassEntryDisabled::new();
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(
        seq,
        "CREATE TABLE wr (a INT, score INT, bucket INT, note TEXT)",
    )
    .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE TABLE wt (label TEXT, a INT)")
        .unwrap();
    let values = (0..600)
        .map(|i| {
            let note = if i % 5 == 0 {
                "NULL".to_string()
            } else {
                format!("'n{i:04}'")
            };
            format!("({i}, {}, {}, {note})", i / 3, i % 2)
        })
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO wr VALUES {values}"))
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "INSERT INTO wr VALUES (600, NULL, 0, 'p0'), (601, NULL, 1, NULL), \
                               (602, NULL, 0, 'p2'), (603, NULL, 1, 'p3')",
    )
    .unwrap();
    let text_partition_values = (0..300)
        .map(|i| format!("('p{}', {i})", i % 2))
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(
        seq,
        &format!("INSERT INTO wt VALUES {text_partition_values}"),
    )
    .unwrap();
    e.set_relational_residency_budget_bytes(0, 16 * 1024);
    let sql = "SELECT a AS aid, score, note, \
                      row_number() OVER (ORDER BY score) AS rn, \
                      rank() OVER (ORDER BY score) AS rnk, \
                      dense_rank() OVER (ORDER BY score) AS dr \
               FROM wr ORDER BY score LIMIT 40 OFFSET 7";
    let streamed = e
        .execute_resident_expr_select_sql(sql)
        .expect("streaming rank windows");
    assert!(
        e.streaming_window_hits() > 0,
        "GPU rank-window kernel fired"
    );
    assert_eq!(streamed.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(streamed.columns[0].name, "aid");
    assert!(
        e.streaming_fold_peak_chunk_bytes() <= 16 * 1024,
        "ordered input + rank output remain within the configured device budget"
    );
    assert_eq!(streamed.rows.len(), 40);
    assert!(streamed.rows.iter().any(|row| row[2] == SqlValue::Null));
    for row in streamed.rows.iter() {
        let score = match row[1] {
            SqlValue::Int4(score) => score,
            ref other => panic!("score: {other:?}"),
        };
        assert_eq!(row[4], SqlValue::Int8(i64::from(score * 3 + 1)));
        assert_eq!(row[5], SqlValue::Int8(i64::from(score + 1)));
    }
    assert_eq!(streamed.rows.row(0)[3], SqlValue::Int8(8));
    assert!(
        e.execute_resident_expr_select_sql(
            "SELECT row_number(*) OVER (ORDER BY score) FROM wr LIMIT 1"
        )
        .unwrap_err()
        .to_string()
        .contains("arguments or aggregate modifiers"),
        "window FuncCall modifiers must never be silently ignored"
    );
    assert!(
        e.execute_resident_expr_select_sql(
            "SELECT row_number() OVER (ORDER BY score USING <) FROM wr LIMIT 1"
        )
        .unwrap_err()
        .to_string()
        .contains("ORDER BY USING"),
        "custom ORDER BY operators must never be treated as ASC"
    );
    assert!(
        e.execute_resident_expr_select_sql(
            "SELECT row_number() OVER w FROM wr \
             WINDOW w AS (ORDER BY score ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
             LIMIT 1"
        )
        .unwrap_err()
        .to_string()
        .contains("explicit window frames"),
        "named explicit frames must be rejected instead of silently ignored"
    );
    let partitioned_sql = "SELECT score, a, \
                                  row_number() OVER (PARTITION BY score ORDER BY a) AS rn, \
                                  rank() OVER (PARTITION BY score ORDER BY a) AS rnk, \
                                  dense_rank() OVER (PARTITION BY score ORDER BY a) AS dr \
                           FROM wr ORDER BY score, a LIMIT 40 OFFSET 7";
    let streamed_partitioned = e
        .execute_resident_expr_select_sql(partitioned_sql)
        .expect("streaming partitioned rank windows");
    assert_eq!(streamed_partitioned.rows.len(), 40);
    for row in streamed_partitioned.rows.iter() {
        assert_eq!(row[2], row[3]);
        assert_eq!(row[3], row[4]);
    }
    let multi_key_sql = "SELECT bucket, score, a, \
                                row_number() OVER (PARTITION BY bucket, score ORDER BY a, score) AS rn, \
                                rank() OVER (PARTITION BY bucket, score ORDER BY a, score) AS rnk, \
                                dense_rank() OVER (PARTITION BY bucket, score ORDER BY a, score) AS dr \
                         FROM wr ORDER BY bucket, score, a, score LIMIT 40 OFFSET 7";
    let streamed_multi_key = e
        .execute_resident_expr_select_sql(multi_key_sql)
        .expect("multi-partition/multi-order rank window");
    assert_eq!(streamed_multi_key.rows.len(), 40);
    assert!(streamed_multi_key
        .rows
        .iter()
        .all(|row| row[3] == row[4] && row[4] == row[5]));
    let named_window_sql = "SELECT bucket, score, a, \
                                   row_number() OVER w AS rn, \
                                   rank() OVER w AS rnk, \
                                   dense_rank() OVER w AS dr \
                            FROM wr \
                            WINDOW p AS (PARTITION BY bucket, score), \
                                   w AS (p ORDER BY a, score) \
                            ORDER BY bucket, score, a, score LIMIT 40 OFFSET 7";
    let streamed_named_window = e
        .execute_resident_expr_select_sql(named_window_sql)
        .expect("inherited named rank window");
    let filtered_window_sql = "SELECT a, score, \
                                      row_number() OVER (ORDER BY score, a) AS rn, \
                                      rank() OVER (ORDER BY score, a) AS rnk, \
                                      dense_rank() OVER (ORDER BY score, a) AS dr \
                               FROM wr WHERE a >= 300 \
                               ORDER BY score, a LIMIT 40 OFFSET 7";
    let streamed_filtered_window = e
        .execute_resident_expr_select_sql(filtered_window_sql)
        .expect("filtered streaming rank window");
    let empty_window_sql = "SELECT a, row_number() OVER (ORDER BY a) AS rn \
                            FROM wr WHERE a < 0 ORDER BY a LIMIT 10";
    let streamed_empty_window = e
        .execute_resident_expr_select_sql(empty_window_sql)
        .expect("empty filtered rank window");
    assert!(streamed_empty_window.rows.is_empty());
    let null_peer_sql = "SELECT a, score, \
                                row_number() OVER (ORDER BY score DESC) AS rn, \
                                rank() OVER (ORDER BY score DESC) AS rnk, \
                                dense_rank() OVER (ORDER BY score DESC) AS dr \
                         FROM wr ORDER BY score DESC LIMIT 6";
    let null_peers = e
        .execute_resident_expr_select_sql(null_peer_sql)
        .expect("NULL peer rank window");
    for row in null_peers.rows.iter().take(4) {
        assert_eq!(row[1], SqlValue::Null);
        assert_eq!(row[3], SqlValue::Int8(1));
        assert_eq!(row[4], SqlValue::Int8(1));
    }
    assert_eq!(null_peers.rows.row(4)[3], SqlValue::Int8(5));
    assert_eq!(null_peers.rows.row(4)[4], SqlValue::Int8(2));
    e.set_relational_residency_budget_bytes(0, 64 * 1024);
    let null_partition_sql = "SELECT score, a, \
                                     row_number() OVER (PARTITION BY score ORDER BY a) AS rn, \
                                     rank() OVER (PARTITION BY score ORDER BY a) AS rnk, \
                                     dense_rank() OVER (PARTITION BY score ORDER BY a) AS dr \
                              FROM wr ORDER BY score, a LIMIT 4 OFFSET 600";
    let null_partition = e
        .execute_resident_expr_select_sql(null_partition_sql)
        .expect("NULL partition rank window");
    assert_eq!(null_partition.rows.len(), 4);
    for (idx, row) in null_partition.rows.iter().enumerate() {
        assert_eq!(row[0], SqlValue::Null);
        let expected = SqlValue::Int8((idx + 1) as i64);
        assert_eq!(row[2], expected);
        assert_eq!(row[3], expected);
        assert_eq!(row[4], expected);
    }
    let explicit_nulls_sql = "SELECT a, score, \
                                     row_number() OVER (ORDER BY score ASC NULLS FIRST, a) AS rn, \
                                     rank() OVER (ORDER BY score ASC NULLS FIRST, a) AS rnk \
                              FROM wr ORDER BY score ASC NULLS FIRST, a";
    let streamed_explicit_nulls = e
        .execute_resident_expr_select_sql(explicit_nulls_sql)
        .expect("unbounded streaming rank with explicit NULL placement");
    assert_eq!(streamed_explicit_nulls.rows.len(), 604);
    assert!(streamed_explicit_nulls
        .rows
        .iter()
        .take(4)
        .all(|row| row[1] == SqlValue::Null));
    let empty_over_sql = "SELECT a, row_number() OVER () AS rn FROM wr LIMIT 5";
    let streamed_empty_over = e
        .execute_resident_expr_select_sql(empty_over_sql)
        .expect("ROW_NUMBER over an empty window specification");
    assert_eq!(streamed_empty_over.rows.len(), 5);
    assert_eq!(streamed_empty_over.rows.row(0)[1], SqlValue::Int8(1));
    let offset_window_sql = "SELECT bucket, a, note, \
                                    lag(note) OVER (PARTITION BY bucket ORDER BY a) AS previous_note, \
                                    lead(a, 2) OVER (PARTITION BY bucket ORDER BY a) AS next_a \
                             FROM wr ORDER BY bucket, a LIMIT 50";
    let streamed_offset_windows = e
        .execute_resident_expr_select_sql(offset_window_sql)
        .expect("streaming LAG/LEAD windows");
    assert_eq!(streamed_offset_windows.rows.len(), 50);
    assert_eq!(streamed_offset_windows.rows.row(0)[3], SqlValue::Null);
    assert_eq!(streamed_offset_windows.rows.row(0)[4], SqlValue::Int4(4));
    let text_partition_window_sql = "SELECT label, a, \
                                            lag(a) OVER (PARTITION BY label ORDER BY a) AS previous_a, \
                                            lead(a) OVER (PARTITION BY label ORDER BY a) AS next_a \
                                     FROM wt ORDER BY label, a LIMIT 80";
    let streamed_text_partition_window = e
        .execute_resident_expr_select_sql(text_partition_window_sql)
        .expect("streaming LAG/LEAD with TEXT partition key");
    assert_eq!(
        streamed_text_partition_window.rows.row(0),
        &[
            SqlValue::Text("p0".to_string()),
            SqlValue::Int4(0),
            SqlValue::Null,
            SqlValue::Int4(2),
        ]
    );
    assert_eq!(
        streamed_text_partition_window.rows.row(1)[2],
        SqlValue::Int4(0)
    );

    e.clear_relational_residency_budget_bytes(0);
    e.populate_relational_residency_snapshot("wr").unwrap();
    let resident = e
        .execute_resident_expr_select_sql(sql)
        .expect("resident GPU rank oracle");
    let resident_partitioned = e
        .execute_resident_expr_select_sql(partitioned_sql)
        .expect("resident partitioned rank oracle");
    let resident_multi_key = e
        .execute_resident_expr_select_sql(multi_key_sql)
        .expect("resident multi-key rank oracle");
    let resident_named_window = e
        .execute_resident_expr_select_sql(named_window_sql)
        .expect("resident named-window rank oracle");
    let resident_filtered_window = e
        .execute_resident_expr_select_sql(filtered_window_sql)
        .expect("resident filtered-window rank oracle");
    let resident_empty_window = e
        .execute_resident_expr_select_sql(empty_window_sql)
        .expect("resident empty-window rank oracle");
    let resident_null_peers = e
        .execute_resident_expr_select_sql(null_peer_sql)
        .expect("resident NULL peer rank oracle");
    let resident_null_partition = e
        .execute_resident_expr_select_sql(null_partition_sql)
        .expect("resident NULL partition rank oracle");
    let resident_explicit_nulls = e
        .execute_resident_expr_select_sql(explicit_nulls_sql)
        .expect("resident explicit-NULL rank oracle");
    let resident_empty_over = e
        .execute_resident_expr_select_sql(empty_over_sql)
        .expect("resident empty-OVER rank oracle");
    let resident_offset_windows = e
        .execute_resident_expr_select_sql(offset_window_sql)
        .expect("resident LAG/LEAD oracle");
    e.populate_relational_residency_snapshot("wt").unwrap();
    let resident_text_partition_window = e
        .execute_resident_expr_select_sql(text_partition_window_sql)
        .expect("resident TEXT-partition LAG/LEAD oracle");
    let normalize = |result: &RelationalSelectResult| {
        let mut rows: Vec<Vec<SqlValue>> = result
            .rows
            .iter()
            .map(|row| {
                vec![
                    row[0].clone(),
                    row[1].clone(),
                    row[2].clone(),
                    row[4].clone(),
                    row[5].clone(),
                ]
            })
            .collect();
        rows.sort_by(|a, b| crate::rel_exec_helpers::compare_sql_values(&a[0], &b[0]));
        rows
    };
    assert_eq!(
        normalize(&streamed),
        normalize(&resident),
        "streaming rank/dense-rank rows == resident GPU oracle (ROW_NUMBER peer order is unspecified)"
    );
    assert_eq!(
        streamed_partitioned.rows.clone().into_boxed(),
        resident_partitioned.rows.clone().into_boxed(),
        "partitioned multi-key streaming rank window == resident GPU oracle"
    );
    assert_eq!(
        streamed_multi_key.rows.clone().into_boxed(),
        resident_multi_key.rows.clone().into_boxed(),
        "multi-key partition/order ranks == resident GPU oracle"
    );
    assert_eq!(
        streamed_named_window.rows.clone().into_boxed(),
        resident_named_window.rows.clone().into_boxed(),
        "named/inherited window definition == resident GPU oracle"
    );
    assert_eq!(
        streamed_filtered_window.rows.clone().into_boxed(),
        resident_filtered_window.rows.clone().into_boxed(),
        "WHERE is applied on-device before window ordering/ranking"
    );
    assert_eq!(
        streamed_empty_window.rows.clone().into_boxed(),
        resident_empty_window.rows.clone().into_boxed()
    );
    let normalize_null_peers = |result: &RelationalSelectResult| {
        let mut rows: Vec<Vec<SqlValue>> = result
            .rows
            .iter()
            .map(|row| {
                vec![
                    row[0].clone(),
                    row[1].clone(),
                    row[3].clone(),
                    row[4].clone(),
                ]
            })
            .collect();
        rows.sort_by(|a, b| crate::rel_exec_helpers::compare_sql_values(&a[0], &b[0]));
        rows
    };
    assert_eq!(
        normalize_null_peers(&null_peers),
        normalize_null_peers(&resident_null_peers),
        "NULL peer ranks == resident GPU oracle (ROW_NUMBER peer order is unspecified)"
    );
    assert_eq!(
        null_partition.rows.clone().into_boxed(),
        resident_null_partition.rows.clone().into_boxed(),
        "NULL partition ranks == resident GPU oracle"
    );
    assert_eq!(
        streamed_explicit_nulls.rows.clone().into_boxed(),
        resident_explicit_nulls.rows.clone().into_boxed(),
        "unbounded explicit-NULL streaming rank == resident GPU oracle"
    );
    assert_eq!(
        streamed_empty_over.rows.clone().into_boxed(),
        resident_empty_over.rows.clone().into_boxed(),
        "ROW_NUMBER OVER () == resident GPU oracle"
    );
    assert_eq!(
        streamed_offset_windows.rows.clone().into_boxed(),
        resident_offset_windows.rows.clone().into_boxed(),
        "streaming LAG/LEAD == resident GPU oracle"
    );
    assert_eq!(
        streamed_text_partition_window.rows.clone().into_boxed(),
        resident_text_partition_window.rows.clone().into_boxed(),
        "TEXT partition boundaries for LAG/LEAD == resident GPU oracle"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_rank_keeps_one_catalog_data_boundary_across_ddl() {
    let mut engine = Engine::new_local_cpu_oracle();
    let mut seq = 0_u64;
    if !gpu_available(&mut engine, &mut seq) {
        return;
    }
    seq += 1;
    engine
        .execute_text(seq, "CREATE TABLE wrd (a INT, score INT)")
        .unwrap();
    seq += 1;
    engine
        .execute_text(
            seq,
            "INSERT INTO wrd VALUES (1, 30), (2, 10), (3, 20), (4, 20)",
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 4096);
    let e = Arc::new(engine);
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM wrd"))
        .unwrap();
    let pinned = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    let reader = {
        let e = Arc::clone(&e);
        let pinned = Arc::clone(&pinned);
        let resume = Arc::clone(&resume);
        std::thread::spawn(move || {
            e.execute_gpu_rank_window_select_instrumented(
                "SELECT a, score, row_number() OVER (ORDER BY score, a) AS rn \
                 FROM wrd ORDER BY score, a",
                &|| {
                    pinned.wait();
                    resume.wait();
                },
            )
            .unwrap()
        })
    };
    pinned.wait();
    seq += 1;
    e.execute_text(seq, "ALTER TABLE wrd ADD COLUMN extra INT DEFAULT 7")
        .unwrap();
    resume.wait();
    let result = reader.join().expect("rank reader");
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Int4(10), SqlValue::Int8(1)],
            vec![SqlValue::Int4(3), SqlValue::Int4(20), SqlValue::Int8(2)],
            vec![SqlValue::Int4(4), SqlValue::Int4(20), SqlValue::Int8(3)],
            vec![SqlValue::Int4(1), SqlValue::Int4(30), SqlValue::Int8(4)],
        ],
        "the pinned rank fold must use its pre-DDL table shape and cold payload"
    );
}
