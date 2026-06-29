//! Grouped-aggregation cardinality probe for the LIVE per-group GROUP BY kernel
//! (`gpu_db_group_by_i32_count_sum_twolevel`, via `group_by_i32_count_sum_from_payload`) -- the kernel the
//! engine actually uses. Sweeps the SAME 8M rows but group columns of varying distinct-count (row % C) to
//! see the throughput-vs-cardinality curve. The two-level shared-mem kernel pre-aggregates per block, so
//! the per-row global-atomic contention that the old `grouped_stats` hash-agg hit at low cardinality is
//! exactly what it targets. (This repointed from the removed `grouped_stats_i32` hash-agg family; the
//! two-level kernel serves COUNT/SUM -- its MIN bit is a no-op, so the MIN column tracks ALL here.)
//!
//!   timeout 200 cargo run --release --example grouped_cardinality_probe -p gpu_db_execution

use std::time::Instant;

use gpu_db_execution::{grouped_agg_mask, CudaDeviceMemoryChunk, CudaDriverRuntime};

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let rows: u64 = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8_388_608);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let n = rows as usize;
    let cards: Vec<u64> = vec![16, 256, 4_096, 65_536, 1 << 18, 1 << 20];

    let Ok(runtime) = CudaDriverRuntime::probe() else {
        eprintln!("no local NVIDIA driver/GPU; skipping");
        return;
    };

    // Layout: [8B row_count][value: N i32][group col per cardinality: N i32 each].
    let off_value = 8u64;
    let group_off = |k: usize| off_value + rows * 4 + (k as u64) * rows * 4;
    let allocated = group_off(cards.len());

    let header = rows.to_le_bytes().to_vec();
    let mut value = Vec::with_capacity(n * 4);
    for row in 0..rows {
        value.extend_from_slice(&((row as i32).wrapping_mul(7)).to_le_bytes());
    }
    // One group column per cardinality: group[i] = (i % C). Distinct-count = C.
    let group_cols: Vec<Vec<u8>> = cards
        .iter()
        .map(|&c| {
            let mut g = Vec::with_capacity(n * 4);
            for row in 0..rows {
                g.extend_from_slice(&((row % c) as i32).to_le_bytes());
            }
            g
        })
        .collect();

    let mut chunks = vec![
        CudaDeviceMemoryChunk { byte_offset: 0, bytes: &header },
        CudaDeviceMemoryChunk { byte_offset: off_value, bytes: &value },
    ];
    for (k, g) in group_cols.iter().enumerate() {
        chunks.push(CudaDeviceMemoryChunk { byte_offset: group_off(k), bytes: g });
    }
    let resident = runtime
        .retain_device_memory_chunks(0, allocated, &chunks)
        .expect("retain resident device memory");

    // Full-table scan: indices = 0..rows (no WHERE), as the executor passes for an unfiltered GROUP BY.
    let indices: Vec<u32> = (0..rows as u32).collect();

    println!("# two-level GROUP BY cardinality sweep. rows={rows}, iters={iters}");
    println!(
        "# aggregate-selection mask: ALL = count+sum+min+max (4 update atomics/row); MIN = 1 atomic;"
    );
    println!("# COUNT = 1 atomic. Reduced masks should be >= ALL at every cardinality (strictly less work).");
    println!(
        "  {:<14} {:>10}  {:>11} | {:>10}  {:>11}  {:>6} | {:>10}  {:>11}  {:>6} | {:>8}",
        "cardinality",
        "ALL us",
        "ALL Melem/s",
        "MIN us",
        "MIN Melem/s",
        "x ALL",
        "CNT us",
        "CNT Melem/s",
        "x ALL",
        "groups",
    );

    // p50 Melem/s for one (cardinality, mask) cell, averaged over `iters` after a warmup.
    let bench = |goff: u64, mask: u32| -> (f64, f64, usize) {
        let run = || {
            resident
                .group_by_i32_count_sum_from_payload(goff, off_value, &indices, mask)
                .unwrap()
                .len()
        };
        for _ in 0..3 {
            run();
        }
        let mut s = Vec::with_capacity(iters);
        let mut groups = 0usize;
        for _ in 0..iters {
            let t = Instant::now();
            groups = run();
            s.push(t.elapsed().as_nanos());
        }
        let us = p50(s) as f64 / 1000.0;
        (us, rows as f64 / us, groups)
    };

    for (k, &c) in cards.iter().enumerate() {
        let goff = group_off(k);
        let (all_us, all_mps, groups) = bench(goff, grouped_agg_mask::ALL);
        let (min_us, min_mps, _) = bench(goff, grouped_agg_mask::MIN);
        let (cnt_us, cnt_mps, _) = bench(goff, grouped_agg_mask::COUNT);
        println!(
            "  {c:<14} {all_us:>9.0}us  {all_mps:>11.1} | {min_us:>9.0}us  {min_mps:>11.1}  {:>5.2}x | {cnt_us:>9.0}us  {cnt_mps:>11.1}  {:>5.2}x | {groups:>8}",
            min_mps / all_mps,
            cnt_mps / all_mps,
        );
    }
    println!("\n# Reduced masks remove per-row atomics unconditionally => expect >= ALL at low AND high card.");
}
