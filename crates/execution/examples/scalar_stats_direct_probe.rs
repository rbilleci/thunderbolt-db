//! Direct scalar-stats reduction over a HIGH-DISTINCT i32 column, vs a HOST oracle + the sum roofline.
//!
//! `SELECT MIN/MAX/AVG(col)` (scalar, no GROUP BY, non-nullable, unfiltered) used to run a self-grouped
//! hash kernel with group == value -- it built an O(distinct)-entry HASH TABLE of every distinct value
//! just to reduce. Over a unique 8M-row column that was an 8M-entry hash table (~182 Melem/s and falling).
//! The direct kernel `scalar_stats_i32_from_payload` is a grid-stride streaming pass + a bar.sync block
//! tree reduction of (count, sum, min, max) + one set of 4 atomics/block -- memory-bound,
//! distinctness-independent. This probe drives it over a unique/scrambled 8M column and reports Melem/s,
//! with the audited direct `sum_i32_from_payload` as the roofline reference. (The dead self-grouped A/B
//! arm was dropped with the `grouped_stats` family; a host oracle now pins the direct (count,sum,min,max).)
//!
//!   timeout 200 cargo run --release --example scalar_stats_direct_probe -p gpu_db_execution

use std::time::Instant;

use gpu_db_execution::{CudaDeviceMemoryChunk, CudaDriverRuntime};

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let rows: u64 = std::env::var("ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8_388_608);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    let n = rows as usize;

    let Ok(runtime) = CudaDriverRuntime::probe() else {
        eprintln!("no local NVIDIA driver/GPU; skipping");
        return;
    };

    // Layout: [8B row_count][value: N i32]. The value column is a SCRAMBLED unique sequence
    // (an LCG-style multiplicative hash of the row index), so distinct-count == rows (~8M distinct) --
    // the worst case for the self-grouped hash path (one hash slot per row).
    let off_value = 8u64;
    let allocated = off_value + rows * 4;
    let header = rows.to_le_bytes().to_vec();
    let mut value = Vec::with_capacity(n * 4);
    for row in 0..rows {
        // Odd multiplier (coprime to 2^32) => a bijection on u32 => a UNIQUE scrambled value per row.
        let scrambled = (row as u32).wrapping_mul(2_654_435_761) as i32;
        value.extend_from_slice(&scrambled.to_le_bytes());
    }

    let chunks = vec![
        CudaDeviceMemoryChunk {
            byte_offset: 0,
            bytes: &header,
        },
        CudaDeviceMemoryChunk {
            byte_offset: off_value,
            bytes: &value,
        },
    ];
    let resident = runtime
        .retain_device_memory_chunks(0, allocated, &chunks)
        .expect("retain resident device memory");

    // Distinct-count of the scrambled column (host): the odd-multiplier hash is a bijection on u32, so
    // every row is a distinct value => ~rows distinct.
    let distinct = {
        let mut s: Vec<i32> = (0..rows)
            .map(|row| (row as u32).wrapping_mul(2_654_435_761) as i32)
            .collect();
        s.sort_unstable();
        s.dedup();
        s.len()
    };

    // Cross-check the direct kernel against an independent HOST oracle (count, sum, min, max) before
    // timing (byte-identity sanity).
    let (d_count, d_sum, d_min, d_max) = resident
        .scalar_stats_i32_from_payload(off_value, rows)
        .expect("direct scalar stats");
    let (h_count, h_sum, h_min, h_max) = {
        let mut sum = 0_i64;
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        for row in 0..rows {
            let v = (row as u32).wrapping_mul(2_654_435_761) as i32;
            sum = sum.wrapping_add(i64::from(v));
            min = min.min(v);
            max = max.max(v);
        }
        (rows, sum, min, max)
    };
    assert_eq!(
        (d_count, d_sum, d_min, d_max),
        (h_count, h_sum, h_min, h_max),
        "direct scalar stats must equal the host oracle"
    );
    println!(
        "# parity OK: count={d_count} sum={d_sum} min={d_min} max={d_max}; distinct={distinct}"
    );

    let p50_us = |mut f: Box<dyn FnMut()>| -> f64 {
        for _ in 0..3 {
            f();
        }
        let mut s = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            f();
            s.push(t.elapsed().as_nanos());
        }
        p50(s) as f64 / 1000.0
    };

    let direct_us = {
        let r = &resident;
        p50_us(Box::new(move || {
            r.scalar_stats_i32_from_payload(off_value, rows).unwrap();
        }))
    };
    let sum_us = {
        let r = &resident;
        p50_us(Box::new(move || {
            r.sum_i32_from_payload(off_value, rows).unwrap();
        }))
    };

    let mps = |us: f64| rows as f64 / us;
    println!("# rows={rows} (~{distinct} distinct), iters={iters}, p50 latency");
    println!("  {:<28} {:>10}  {:>12}", "path", "p50 us", "Melem/s");
    println!(
        "  {:<28} {:>9.0}us  {:>12.1}",
        "direct scalar_stats",
        direct_us,
        mps(direct_us)
    );
    println!(
        "  {:<28} {:>9.0}us  {:>12.1}  (roofline ref)",
        "direct sum (audited)",
        sum_us,
        mps(sum_us)
    );
    println!(
        "\n# direct scalar_stats is {:.0}% of the sum roofline at ~{distinct}-distinct.",
        100.0 * mps(direct_us) / mps(sum_us),
    );
}
