use super::{gpu_available, select};
use crate::Engine;
use gpu_db_sql::SqlValue;
use std::sync::Arc;

/// Plain and layered views preserve their stored relational plan when recursively expanded, so an
/// over-budget base relation still reaches the streaming GPU fold rather than the host executor.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_views_expand_into_device_fold() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0_u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE vb (a INT, note TEXT)")
        .unwrap();
    let values = (0..1000)
        .map(|i| {
            let note = if i % 13 == 0 {
                "NULL".to_string()
            } else {
                format!("'v{i:04}'")
            };
            format!("({i}, {note})")
        })
        .collect::<Vec<_>>()
        .join(",");
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO vb VALUES {values}"))
        .unwrap();
    seq += 1;
    e.execute_text(
        seq,
        "CREATE VIEW vv AS SELECT a, note FROM vb WHERE a >= 500",
    )
    .unwrap();
    seq += 1;
    e.execute_text(seq, "CREATE VIEW vv2 AS SELECT * FROM vv")
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);
    let hits = e.streaming_fold_hits();
    let viewed = e
        .execute_relational_select(&select("SELECT * FROM vv2"))
        .expect("layered streaming view");
    assert!(
        e.streaming_fold_hits() > hits,
        "view expansion reached the streaming fold"
    );
    assert_eq!(viewed.rows.len(), 500);
    assert_eq!(viewed.rows.row(0)[0], SqlValue::Int4(500));
    assert!(viewed.rows.iter().any(|row| row[1] == SqlValue::Null));
    let direct = e
        .execute_relational_select(&select("SELECT a, note FROM vb WHERE a >= 500"))
        .unwrap();
    assert_eq!(
        viewed.rows.clone().into_boxed(),
        direct.rows.clone().into_boxed()
    );
}

#[test]
fn layered_view_keeps_one_catalog_data_boundary_across_ddl() {
    let e = Arc::new(Engine::new_local_cpu_oracle());
    e.execute_text(1, "CREATE TABLE vd (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO vd VALUES (1, 10), (2, 20)")
        .unwrap();
    e.execute_text(3, "CREATE VIEW vd1 AS SELECT * FROM vd")
        .unwrap();
    e.execute_text(4, "CREATE VIEW vd2 AS SELECT * FROM vd1")
        .unwrap();
    let query = select("SELECT * FROM vd2");
    let pinned = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    let reader = {
        let e = Arc::clone(&e);
        let pinned = Arc::clone(&pinned);
        let resume = Arc::clone(&resume);
        std::thread::spawn(move || {
            e.execute_relational_select_instrumented(&query, || {
                pinned.wait();
                resume.wait();
            })
            .unwrap()
        })
    };
    pinned.wait();
    e.execute_text(5, "ALTER TABLE vd ADD COLUMN c INT DEFAULT 7")
        .unwrap();
    resume.wait();
    let old_boundary = reader.join().expect("reader");
    assert_eq!(old_boundary.columns.len(), 2);
    assert_eq!(
        old_boundary.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)]
        ]
    );
    let new_boundary = e
        .execute_relational_select(&select("SELECT * FROM vd2"))
        .unwrap();
    assert_eq!(new_boundary.columns.len(), 3);
    assert!(new_boundary
        .rows
        .iter()
        .all(|row| row[2] == SqlValue::Int4(7)));
}
