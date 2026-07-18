//! Result-path step-1 — decompose the engine's host-side per-row materialization residual.
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
//!     convert on the fly) — the "device->wire" lower bound on the host side
//!
//! Reports ns/row + implied lookups/s for each. The gap CURRENT->FLAT-* bounds the columnar-layer payoff.
//!
//! Run: cargo run --release --example result_materialization_probe -p gpu_db_engine   (no GPU needed)

use gpu_db_sql::SqlValue;
use std::time::Instant;

/// HIGH-OUTPUT (the cross-kernel-result-path-transfer doc's target): the GENERAL non-unique route
/// (`engine_resident_probe.rs:1367`+`:1519`) materializes `m` needles x `k` matched rows each. Mirrors the
/// real boxed path (CURRENT) vs lpb's flat assemble (`assemble_batched_rows`, general arm). `row_index` is
/// shuffled within each needle (the kernel's `atom.global.add` schedule order) so the within-needle sort that
/// BOTH paths must do is real — isolating the per-row `Vec<SqlValue>` boxing + enum-wrap as the differentiator.
fn high_output(m: usize, k: usize) {
    let ncols = 2usize;
    let n = m * k;
    let iters = 20usize;
    // Synthetic kernel output in schedule order: row i carries (needle_index, row_index, values). Interleave
    // needles (round-robin) and shuffle row_index within a needle via a hash permutation.
    let needle_indices: Vec<u32> = (0..n).map(|i| (i % m) as u32).collect();
    let row_indices: Vec<u64> = (0..n)
        .map(|i| ((i / m) as u64).wrapping_mul(2_654_435_761) % k as u64)
        .collect();
    let row_values: Vec<i32> = (0..n)
        .flat_map(|i| [i as i32, (i as i32).wrapping_mul(7)])
        .collect();
    let rv = |i: usize| &row_values[i * ncols..i * ncols + ncols];

    let per = |total_ns: u128| total_ns as f64 / n as f64;
    let lps = |ns: f64| if ns > 0.0 { 1.0e9 / ns } else { 0.0 };

    // CURRENT — exactly engine_resident_probe.rs:1367 (boxed group + i32->SqlValue per row) + :1519 sort.
    let mut current = u128::MAX;
    let mut sink = 0i64;
    for _ in 0..iters {
        let t = Instant::now();
        let mut rows_by_select: Vec<Vec<(u64, Vec<SqlValue>)>> = vec![Vec::new(); m];
        for i in 0..n {
            rows_by_select[needle_indices[i] as usize].push((
                row_indices[i],
                rv(i).iter().copied().map(SqlValue::Int4).collect(),
            ));
        }
        let assembled: Vec<Vec<Vec<SqlValue>>> = rows_by_select
            .into_iter()
            .map(|mut s| {
                s.sort_by_key(|(ri, _)| *ri);
                s.into_iter().map(|(_, row)| row).collect()
            })
            .collect();
        current = current.min(t.elapsed().as_nanos());
        sink += assembled.iter().map(|s| s.len() as i64).sum::<i64>();
    }

    // FLAT — assemble_batched_rows general arm: O(n) counting-sort scatter by needle + within-needle index
    // sort by row_index + raw-i32 flatten (no per-row Vec, no SqlValue enum-wrap).
    let mut flat = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let mut counts = vec![0u32; m];
        for &ni in &needle_indices {
            counts[ni as usize] += 1;
        }
        let mut ranges = Vec::with_capacity(m);
        let mut acc = 0u32;
        for &c in &counts {
            ranges.push((acc, c));
            acc += c;
        }
        let total = acc as usize;
        let mut slot = vec![0u32; total];
        let mut cursor: Vec<u32> = ranges.iter().map(|&(s, _)| s).collect();
        for (i, &needle_index) in needle_indices.iter().take(n).enumerate() {
            let ni = needle_index as usize;
            slot[cursor[ni] as usize] = i as u32;
            cursor[ni] += 1;
        }
        for &(start, count) in &ranges {
            if count > 1 {
                let s = start as usize;
                slot[s..s + count as usize].sort_by_key(|&i| row_indices[i as usize]);
            }
        }
        let mut values = Vec::with_capacity(total * ncols);
        for &i in &slot {
            values.extend_from_slice(rv(i as usize));
        }
        flat = flat.min(t.elapsed().as_nanos());
        sink += values.len() as i64;
    }

    println!("\n# HIGH-OUTPUT general non-unique route: m={m} needles x k={k} rows = {n} rows x {ncols} cols");
    println!(
        "  {:<14} {:>10} {:>14}",
        "variant", "ns/row", "implied rows/s"
    );
    println!(
        "  {:<14} {:>10.1} {:>14.0}",
        "current(boxed)",
        per(current),
        lps(per(current))
    );
    println!(
        "  {:<14} {:>10.1} {:>14.0}",
        "flat-i32",
        per(flat),
        lps(per(flat))
    );
    println!(
        "  current/flat: {:.1}x   (sink {sink})",
        per(current) / per(flat).max(0.01)
    );
}

/// VM lever (atomic-vs-sort split): the predicate compaction does an atomic-append (unordered) then a HOST
/// `sort_unstable` of the surviving row indices. This bounds the SORT's share of the ~3425us compact at
/// cats=1 (n=262144). The atomic-append output is "locally shuffled, globally roughly-ascending" (grid-
/// stride schedule order), so pdqsort is adaptive -- measure a few orders to bracket it.
fn sort_cost(n: usize) {
    let lcg = |mut x: u64| {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x
    };
    let iters = 30usize;
    let bench = |label: &str, build: &dyn Fn() -> Vec<u32>| {
        let mut best = u128::MAX;
        let mut sink = 0u64;
        for _ in 0..iters {
            let mut v = build();
            let t = Instant::now();
            v.sort_unstable();
            best = best.min(t.elapsed().as_nanos());
            sink += v[0] as u64 + v[n - 1] as u64;
        }
        println!(
            "  sort {label:<22} {:>8.0}us  ({:>5.1} ns/elem)  sink={sink}",
            best as f64 / 1000.0,
            best as f64 / n as f64
        );
    };
    println!("\n# host sort_unstable of {n} u32 (the compaction's host sort -- the prefix-sum lever removes it):");
    bench("already-ascending", &|| (0..n as u32).collect());
    // grid-stride atomic-append order: matches in ~row order but warp-interleaved within small windows.
    bench("roughly-ascending(±64)", &|| {
        (0..n)
            .map(|i| (i as i64 + (lcg(i as u64) % 128) as i64 - 64).clamp(0, n as i64 - 1) as u32)
            .collect()
    });
    bench("fully-shuffled", &|| {
        (0..n).map(|i| (lcg(i as u64) % n as u64) as u32).collect()
    });
}

fn main() {
    if let Ok(n) = std::env::var("SORT_N").map(|v| v.parse().unwrap_or(262144)) {
        sort_cost(n);
        return;
    }
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536);
    let ncols = 2usize;
    let iters = 50usize;
    // Synthetic GPU output: N needles, ONE matched row each (the batched point-read shape); values [id,bal].
    // (needle_index, row_index, values)
    let projected: Vec<(usize, u64, [i32; 2])> = (0..n)
        .map(|i| (i, i as u64, [i as i32, (i as i32) * 7]))
        .collect();

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

    println!(
        "# engine result-materialization residual, N={n} rows x {ncols} cols (point-read shape)"
    );
    println!("# end-to-end today ~127ns/row (7.8M); GPU drain ~33ns/row (30M). Which stage is the residual?\n");
    println!(
        "  {:<14} {:>10} {:>14}",
        "variant", "ns/row", "implied lookups/s"
    );
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

    let m: usize = std::env::var("NEEDLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let k: usize = std::env::var("ROWS_PER_NEEDLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384);
    high_output(m, k);
}
