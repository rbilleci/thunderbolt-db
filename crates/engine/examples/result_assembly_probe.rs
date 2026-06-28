//! Result-path optimization — quantify the recoverable host cost of result materialization (CPU-only).
//!
//! The phase-split localized the engine's ~330ns/row host overhead (which caps end-to-end read at ~3M,
//! masking the GPU drain of 23-30M): ~107ns/row in SUBMIT (per-needle `result_columns.clone()` +
//! `access_path.clone()` — cloning identical template data once PER needle) + ~226ns/row in COMPLETE (the
//! `Vec<SqlValue>`-per-row + per-needle `RelationalSelectResult` assembly). Before committing to the (wide)
//! result-type change, this CPU benchmark measures the CEILING of the optimization: assemble N point-read
//! results the CURRENT way (deep-clone columns + access_path per needle, `Vec<SqlValue>` per row) vs a LEAN
//! way (Arc-shared columns + access_path, flat values) and report ns/row for each. The gap is the
//! recoverable host cost; LEAN ns/row vs the GPU drain's ~33ns/row says how close end-to-end can get to the
//! tens-of-millions.
//!
//! Run: cargo run --release --example result_assembly_probe -p gpu_db_engine   (no GPU needed)

use std::sync::Arc;
use std::time::Instant;

use gpu_db_engine::{RelationalAccessPath, RelationalColumn, RelationalSelectResult};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{SqlType, SqlValue};

fn col(name: &str) -> RelationalColumn {
    RelationalColumn {
        id: 0,
        table_oid: 0,
        attnum: 1,
        name: name.to_string(),
        ty: SqlType::Int4,
        domain: None,
        default: None,
        type_oid: 23,
        type_size: 4,
    }
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536);
    let iters: usize = 50;

    let columns = vec![col("id"), col("balance")];
    let access = RelationalAccessPath::EqualityIndex {
        table: "accounts".to_string(),
        column: "id".to_string(),
        matched_keys: 1,
    };
    // Synthetic GPU output: N needles, one (id, balance) row each (the batched point-read shape).
    let raw: Vec<(i32, i32)> = (0..n as i32).map(|i| (i, i * 7)).collect();

    // ---- CURRENT: one RelationalSelectResult per needle, deep-cloning columns + access_path, Vec<SqlValue>
    //      per row. (Exactly what `submit_*` member-build + `complete_*_detached` do today.)
    let mut current_ns = u128::MAX;
    let mut sink_a = 0i64;
    for _ in 0..iters {
        let t = Instant::now();
        let mut results: Vec<RelationalSelectResult> = Vec::with_capacity(n);
        for &(id, bal) in &raw {
            let rows = vec![vec![SqlValue::Int4(id), SqlValue::Int4(bal)]];
            results.push(RelationalSelectResult {
                columns: columns.clone(),
                rows,
                planned_target: DeviceTarget::Gpu(0),
                executed_target: DeviceTarget::Gpu(0),
                fallback_reason: None,
                access_path: access.clone(),
            });
        }
        current_ns = current_ns.min(t.elapsed().as_nanos());
        sink_a += results.len() as i64 + results[0].rows.len() as i64;
    }

    // ---- LEAN: Arc-shared columns + access_path (cheap refcount clone per needle), values still SqlValue
    //      but no per-needle deep clone. (Models RelationalSelectResult holding Arc-shared schema.)
    let columns_arc = Arc::new(columns.clone());
    let access_arc = Arc::new(access.clone());
    struct LeanResult {
        columns: Arc<Vec<RelationalColumn>>,
        rows: Vec<Vec<SqlValue>>,
        access_path: Arc<RelationalAccessPath>,
    }
    let mut lean_ns = u128::MAX;
    let mut sink_b = 0i64;
    for _ in 0..iters {
        let t = Instant::now();
        let mut results: Vec<LeanResult> = Vec::with_capacity(n);
        for &(id, bal) in &raw {
            results.push(LeanResult {
                columns: Arc::clone(&columns_arc),
                rows: vec![vec![SqlValue::Int4(id), SqlValue::Int4(bal)]],
                access_path: Arc::clone(&access_arc),
            });
        }
        lean_ns = lean_ns.min(t.elapsed().as_nanos());
        sink_b += results.len() as i64 + results[0].rows.len() as i64;
    }

    // ---- LEANEST: Arc-shared schema + FLAT column-major i32 values (no per-row Vec<SqlValue> alloc).
    let mut flat_ns = u128::MAX;
    let mut sink_c = 0i64;
    for _ in 0..iters {
        let t = Instant::now();
        let needle_owner: Vec<u32> = (0..n as u32).collect(); // which needle each row belongs to
        let mut values: Vec<i32> = Vec::with_capacity(n * 2); // column-major-ish flat
        for &(id, bal) in &raw {
            values.push(id);
            values.push(bal);
        }
        let _shared_cols = Arc::clone(&columns_arc);
        let _shared_access = Arc::clone(&access_arc);
        flat_ns = flat_ns.min(t.elapsed().as_nanos());
        sink_c += needle_owner.len() as i64 + values.len() as i64;
    }

    let per = |total: u128| total as f64 / n as f64;
    println!("# result-assembly cost, N={n} point-read results (1 row x 2 cols each)");
    println!("# (the host cost that today caps end-to-end read at ~3M; GPU drain ~33ns/row = ~30M)\n");
    println!("  {:<10} {:>12} {:>14} {:>16}", "variant", "total us", "ns/row", "implied lookups/s");
    let row = |label: &str, ns: u128| {
        let perrow = per(ns);
        let lps = if perrow > 0.0 { 1.0e9 / perrow } else { 0.0 };
        println!("  {label:<10} {:>12} {:>14.1} {:>16.0}", ns / 1000, perrow, lps);
    };
    row("current", current_ns);
    row("lean-arc", lean_ns);
    row("flat", flat_ns);
    println!("\n  current/lean speedup: {:.2}x   current/flat: {:.2}x", per(current_ns) / per(lean_ns).max(0.01), per(current_ns) / per(flat_ns).max(0.01));
    println!("  (sinks {sink_a} {sink_b} {sink_c})");
    println!("\n# 'lean-arc' = Arc-share columns/access_path; 'flat' = + drop per-row Vec<SqlValue>. The gap to");
    println!("# 'current' is the recoverable host cost; ns/row vs ~33 (drain) bounds how close end-to-end gets.");
}
