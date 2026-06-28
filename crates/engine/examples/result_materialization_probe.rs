//! Result-path step-1 — decompose the engine's per-row materialization residual (CPU-only).
//!
//! After the Arc-share, end-to-end read is ~7.8M (127ns/row); the GPU drain is ~33ns/row (30M). The residual
//! ~94ns/row is the COMPLETION's per-row work: convert each row's `Vec<i32>` -> `Vec<SqlValue>`, group by
//! needle into a per-needle `Vec<(row_index, Vec<SqlValue>)>`, sort by row_index, assemble
//! `Vec<Vec<SqlValue>>`. This replicates that EXACT pipeline (the completion at
//! `engine_retained_read.rs`) for N synthetic rows and times it against two flatter representations, to see
//! which stage to kill first:
//!   - CURRENT      : the boxed `Vec<Vec<SqlValue>>` per needle (today's path)
//!   - FLAT-SQLVALUE: one row-major `Vec<SqlValue>` + per-needle row ranges (drops the per-row Vec boxing)
//!   - FLAT-I32     : one row-major `Vec<i32>` (also drops the SqlValue enum-wrap; the wire encoder would
//!                    convert on the fly) — the "device->wire" lower bound on the host side
//! Reports ns/row + implied lookups/s for each. The gap CURRENT->FLAT-* bounds the columnar-layer payoff.
//!
//! Run: cargo run --release --example result_materialization_probe -p gpu_db_engine   (no GPU needed)

use std::time::Instant;
use gpu_db_sql::SqlValue;

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(65536);
    let ncols = 2usize;
    let iters = 50usize;
    // Synthetic GPU output: N needles, ONE matched row each (the batched point-read shape); values [id,bal].
    // (needle_index, row_index, values)
    let projected: Vec<(usize, u64, [i32; 2])> =
        (0..n).map(|i| (i, i as u64, [i as i32, (i as i32) * 7])).collect();

    let per = |total_ns: u128| total_ns as f64 / n as f64;
    let lps = |ns: f64| if ns > 0.0 { 1.0e9 / ns } else { 0.0 };

    // ---- CURRENT: exactly the completion's stages 2-3 (group + i32->SqlValue + sort + assemble) ----
    let mut current = u128::MAX;
    let mut sink = 0i64;
    for _ in 0..iters {
        let t = Instant::now();
        let mut rows_by_select: Vec<Vec<(u64, Vec<SqlValue>)>> = vec![Vec::new(); n];
        for &(ni, ri, vals) in &projected {
            rows_by_select[ni].push((ri, vals.iter().copied().map(SqlValue::Int4).collect()));
        }
        let rows_by_select: Vec<Vec<Vec<SqlValue>>> = rows_by_select
            .into_iter()
            .map(|mut slice| {
                slice.sort_by_key(|(ri, _)| *ri);
                slice.into_iter().map(|(_, row)| row).collect()
            })
            .collect();
        current = current.min(t.elapsed().as_nanos());
        sink += rows_by_select.iter().map(|r| r.len() as i64).sum::<i64>();
    }

    // ---- FLAT-SQLVALUE: row-major Vec<SqlValue> + per-needle (start,len). For 1 row/needle no sort needed;
    //      for multi-row you'd sort an index. Keeps SqlValue (so the wire path is unchanged downstream). ----
    let mut flat_sv = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let mut values: Vec<SqlValue> = Vec::with_capacity(n * ncols);
        let mut ranges: Vec<(u32, u32)> = vec![(0, 0); n]; // (start_row, row_count) per needle
        // single matched row per needle, in needle order (point-read fast path)
        for (row, &(ni, _ri, vals)) in projected.iter().enumerate() {
            ranges[ni] = (row as u32, 1);
            for &v in &vals {
                values.push(SqlValue::Int4(v));
            }
        }
        flat_sv = flat_sv.min(t.elapsed().as_nanos());
        sink += values.len() as i64 + ranges.len() as i64;
    }

    // ---- FLAT-I32: row-major Vec<i32> (no SqlValue enum-wrap); the wire encoder would format from raw. ----
    let mut flat_i32 = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let mut values: Vec<i32> = Vec::with_capacity(n * ncols);
        for &(_ni, _ri, vals) in &projected {
            values.extend_from_slice(&vals);
        }
        flat_i32 = flat_i32.min(t.elapsed().as_nanos());
        sink += values.len() as i64;
    }

    println!("# engine result-materialization residual, N={n} rows x {ncols} cols (point-read shape)");
    println!("# end-to-end today ~127ns/row (7.8M); GPU drain ~33ns/row (30M). Which stage is the residual?\n");
    println!("  {:<14} {:>10} {:>14}", "variant", "ns/row", "implied lookups/s");
    let row = |label: &str, ns: u128| {
        println!("  {label:<14} {:>10.1} {:>14.0}", per(ns), lps(per(ns)));
    };
    row("current", current);
    row("flat-sqlvalue", flat_sv);
    row("flat-i32", flat_i32);
    println!(
        "\n  current/flat-sqlvalue: {:.1}x   current/flat-i32: {:.1}x   (sink {sink})",
        per(current) / per(flat_sv).max(0.01),
        per(current) / per(flat_i32).max(0.01)
    );
    println!("\n# current->flat-sqlvalue = the per-row Vec<SqlValue> boxing+group+sort cost (host-flat win,");
    println!("# wire path unchanged). flat-sqlvalue->flat-i32 = the SqlValue enum-wrap (needs GPU->wire to remove).");
}
