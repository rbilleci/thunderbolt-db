use super::{gpu_available, select};
use crate::Engine;
use gpu_db_sql::SqlValue;

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
