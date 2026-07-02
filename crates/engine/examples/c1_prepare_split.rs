//! PHASE C slice 1 — measure the DELETE/UPDATE PREPARE share (the host seq_scan, ledger #1).
//!
//! `prepare_delete`/`prepare_update` seq-scan + decode the WHOLE host store to resolve
//! (tuple_id, key, row) for the matched rows — O(table) host work per statement, the measured cap on
//! the incremental tombstone win (2.2x instead of O(rows)). This probe times single-row DELETE and
//! UPDATE end-to-end at growing table sizes under the (now default) sharded data plane; the growth
//! with table size IS the prepare share (the tombstone apply is O(rows touched) and flat by
//! construction — proven by the SV4b/SV5 gates).
//!
//! Run: GPU_DB_BENCH_C1=1 cargo run --release --example c1_prepare_split -p gpu_db_engine
//!      GPU_DB_BENCH_SIZES=16384,65536,262144   GPU_DB_BENCH_OPS=50

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::Engine;

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("GPU_DB_BENCH_C1").ok().as_deref() != Some("1") {
        println!("set GPU_DB_BENCH_C1=1 to run (GPU-touching). Skipping.");
        return Ok(());
    }
    let sizes: Vec<i64> = env::var("GPU_DB_BENCH_SIZES")
        .unwrap_or_else(|_| "16384,65536,262144".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let ops: usize = env::var("GPU_DB_BENCH_OPS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(50);

    println!(
        "c1_prepare_split: single-row DELETE / UPDATE p50 us vs table size (ops={ops}, \
         constrained={})",
        env::var("GPU_DB_BENCH_CONSTRAINED").ok().as_deref() == Some("1")
    );
    println!("{:>10} {:>14} {:>14}", "rows", "DELETE p50", "UPDATE p50");
    let constrained = env::var("GPU_DB_BENCH_CONSTRAINED").ok().as_deref() == Some("1");
    for &rows in &sizes {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(true);
        if env::var("GPU_DB_BENCH_INDEX_OFF").ok().as_deref() == Some("1") {
            e.set_dml_value_index_resolve_enabled(false); // the scan-oracle configuration
        }
        if constrained {
            // 1b coverage: a UNIQUE index + an inbound FK child — the validators run index-driven.
            e.execute_text(1, "CREATE TABLE t (id INT UNIQUE, v INT)")?;
            e.execute_text(2, "CREATE TABLE c (id INT, tid INT)")?;
            e.execute_text(
                3,
                "ALTER TABLE ONLY c ADD CONSTRAINT c_tid_fk FOREIGN KEY (tid) REFERENCES t(id)",
            )?;
        } else {
            e.execute_text(1, "CREATE TABLE t (id INT, v INT)")?;
        }
        let mut seq = 4u64;
        let mut i = 0i64;
        while i < rows {
            let end = (i + 1000).min(rows);
            let values: Vec<String> = (i..end).map(|k| format!("({k},{})", k * 10)).collect();
            e.execute_text(seq, &format!("INSERT INTO t (id, v) VALUES {}", values.join(",")))?;
            seq += 1;
            i = end;
        }
        // Warm the residency + index caches.
        let _ = e.execute_relational_select_text("SELECT id, v FROM t WHERE id = 1");

        let mut del_us: Vec<f64> = Vec::with_capacity(ops);
        for k in 0..ops as i64 {
            let q = Instant::now();
            e.execute_text(seq, &format!("DELETE FROM t WHERE id = {}", 1000 + k))?;
            seq += 1;
            del_us.push(q.elapsed().as_secs_f64() * 1e6);
        }
        let mut upd_us: Vec<f64> = Vec::with_capacity(ops);
        for k in 0..ops as i64 {
            let q = Instant::now();
            e.execute_text(
                seq,
                &format!("UPDATE t SET v = {} WHERE id = {}", k, 5000 + k),
            )?;
            seq += 1;
            upd_us.push(q.elapsed().as_secs_f64() * 1e6);
        }
        del_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        upd_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:>10} {:>14.0} {:>14.0}",
            rows,
            del_us[del_us.len() / 2],
            upd_us[upd_us.len() / 2]
        );
    }
    Ok(())
}
