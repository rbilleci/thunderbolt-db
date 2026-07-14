use super::{gpu_available, select};
use crate::Engine;
use gpu_db_sql::SqlValue;

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
