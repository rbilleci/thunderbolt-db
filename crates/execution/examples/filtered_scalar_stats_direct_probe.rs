//! SLICE B: filtered + nullable direct scalar-stats reduction vs the OLD self-grouped hash path, over a
//! HIGH-DISTINCT i32 column.
//!
//! `SELECT MAX(col) WHERE col < k` (filtered scalar) and the unfiltered NULLABLE scalar used to run the
//! self-grouped hash kernel `filtered_grouped_stats_i32_*` / `grouped_stats_i32_nullable_*` with group ==
//! value -- building an O(distinct)-entry HASH TABLE of every surviving distinct value just to reduce.
//! Over a unique ~8M-row column that is a multi-M-entry hash table (~tens of Melem/s and falling). The
//! extended direct kernel `gpu_db_resident_i32_scalar_stats` (slice b: on-device filter predication +
//! NULL-skip) is a grid-stride streaming pass + a bar.sync block tree reduction of (count, sum, min, max)
//! + one set of 4 atomics/block -- memory-bound, distinctness-independent. This probe drives BOTH over the
//! same unique/scrambled column and reports Melem/s, with the audited direct `sum_i32_from_payload` as the
//! roofline reference.
//!
//!   timeout 200 cargo run --release --example filtered_scalar_stats_direct_probe -p gpu_db_execution

use std::time::Instant;

use gpu_db_execution::{
    grouped_agg_mask, CudaDeviceMemoryChunk, CudaDriverRuntime, CudaI32Comparison,
};

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let rows: u64 = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8_388_608);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let n = rows as usize;

    let Ok(runtime) = CudaDriverRuntime::probe() else {
        eprintln!("no local NVIDIA driver/GPU; skipping");
        return;
    };

    // Layout: [8B row_count][value: N i32][validity bitmap: ceil(N/32) u32]. The value column is a
    // SCRAMBLED unique sequence (a multiplicative hash of the row index), so distinct-count == rows -- the
    // worst case for the self-grouped hash path (one hash slot per surviving row). is_null every 9th row.
    let off_value = 8u64;
    let bitmap_words = n.div_ceil(32);
    let off_bitmap = off_value + rows * 4;
    let allocated = off_bitmap + (bitmap_words as u64) * 4;
    let header = rows.to_le_bytes().to_vec();
    let mut value = Vec::with_capacity(n * 4);
    for row in 0..rows {
        // Odd multiplier (coprime to 2^32) => a bijection on u32 => a UNIQUE scrambled value per row.
        let scrambled = (row as u32).wrapping_mul(2_654_435_761) as i32;
        value.extend_from_slice(&scrambled.to_le_bytes());
    }
    let is_null = |i: u64| i % 9 == 0;
    let mut bitmap = vec![0u32; bitmap_words];
    for i in 0..n {
        if !is_null(i as u64) {
            bitmap[i / 32] |= 1u32 << (i % 32);
        }
    }
    let bitmap_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(bitmap.as_ptr().cast::<u8>(), bitmap.len() * 4) };

    let chunks = vec![
        CudaDeviceMemoryChunk { byte_offset: 0, bytes: &header },
        CudaDeviceMemoryChunk { byte_offset: off_value, bytes: &value },
        CudaDeviceMemoryChunk { byte_offset: off_bitmap, bytes: bitmap_bytes },
    ];
    let resident = runtime
        .retain_device_memory_chunks(0, allocated, &chunks)
        .expect("retain resident device memory");

    // A needle near the median so roughly half the rows survive the filter (a realistic selectivity, and a
    // large surviving set so the self-grouped hash table is still huge).
    let needle = 0i32;
    let cmp = CudaI32Comparison::Lt;

    // ---- parity: filtered (no bitmap) direct == reduced self-grouped, before timing. ----
    let direct_f =
        resident.filtered_scalar_stats_i32_from_payload(off_value, rows, needle, cmp, None).unwrap();
    let grouped_f = resident
        .filtered_grouped_stats_i32_compare_from_payload(off_value, off_value, off_value, rows, needle, cmp)
        .unwrap();
    let gf_count: u64 = grouped_f.iter().map(|g| g.count).sum();
    let gf_sum: i64 = grouped_f.iter().map(|g| g.sum).sum();
    let gf_min = grouped_f.iter().map(|g| g.min).min().unwrap();
    let gf_max = grouped_f.iter().map(|g| g.max).max().unwrap();
    assert_eq!(direct_f, (gf_count, gf_sum, gf_min, gf_max), "filtered direct == reduced self-grouped");
    println!(
        "# filtered parity OK (v < {needle}): count={} max={} (surviving distinct groups={})",
        direct_f.0,
        direct_f.3,
        grouped_f.len()
    );

    // ---- parity: nullable (unfiltered) direct == reduced self-grouped. ----
    let direct_nz = resident
        .nullable_scalar_stats_i32_from_payload(off_value, rows, Some(off_bitmap))
        .unwrap();
    let grouped_nz = resident
        .grouped_stats_i32_nullable_from_payload(off_value, off_value, rows, Some(off_bitmap), grouped_agg_mask::ALL)
        .unwrap();
    let gn_count: u64 = grouped_nz.iter().map(|g| g.count).sum();
    let gn_sum: i64 = grouped_nz.iter().map(|g| g.sum).sum();
    let gn_min = grouped_nz.iter().map(|g| g.min).min().unwrap();
    let gn_max = grouped_nz.iter().map(|g| g.max).max().unwrap();
    assert_eq!(direct_nz, (gn_count, gn_sum, gn_min, gn_max), "nullable direct == reduced self-grouped");
    println!(
        "# nullable parity OK: count={} (non-NULL) max={} (surviving distinct groups={})",
        direct_nz.0,
        direct_nz.3,
        grouped_nz.len()
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

    let direct_filtered_us = {
        let r = &resident;
        p50_us(Box::new(move || {
            r.filtered_scalar_stats_i32_from_payload(off_value, rows, needle, cmp, None).unwrap();
        }))
    };
    let grouped_filtered_us = {
        let r = &resident;
        p50_us(Box::new(move || {
            r.filtered_grouped_stats_i32_compare_from_payload(off_value, off_value, off_value, rows, needle, cmp)
                .unwrap();
        }))
    };
    let direct_nullable_us = {
        let r = &resident;
        p50_us(Box::new(move || {
            r.nullable_scalar_stats_i32_from_payload(off_value, rows, Some(off_bitmap)).unwrap();
        }))
    };
    let grouped_nullable_us = {
        let r = &resident;
        p50_us(Box::new(move || {
            r.grouped_stats_i32_nullable_from_payload(off_value, off_value, rows, Some(off_bitmap), grouped_agg_mask::ALL)
                .unwrap();
        }))
    };
    let sum_us = {
        let r = &resident;
        p50_us(Box::new(move || {
            r.sum_i32_from_payload(off_value, rows).unwrap();
        }))
    };

    let mps = |us: f64| rows as f64 / us;
    println!("# rows={rows} (~{rows} distinct), iters={iters}, p50 latency");
    println!("  {:<34} {:>10}  {:>12}", "path", "p50 us", "Melem/s");
    println!("  {:<34} {:>9.0}us  {:>12.1}", "filtered direct (NEW)", direct_filtered_us, mps(direct_filtered_us));
    println!("  {:<34} {:>9.0}us  {:>12.1}", "filtered self-grouped (OLD)", grouped_filtered_us, mps(grouped_filtered_us));
    println!("  {:<34} {:>9.0}us  {:>12.1}", "nullable direct (NEW)", direct_nullable_us, mps(direct_nullable_us));
    println!("  {:<34} {:>9.0}us  {:>12.1}", "nullable self-grouped (OLD)", grouped_nullable_us, mps(grouped_nullable_us));
    println!("  {:<34} {:>9.0}us  {:>12.1}  (roofline ref)", "direct sum (audited)", sum_us, mps(sum_us));
    println!(
        "\n# filtered MAX(v) WHERE v<{needle}: direct is {:.1}x the self-grouped hash path; nullable scalar: {:.1}x.",
        grouped_filtered_us / direct_filtered_us,
        grouped_nullable_us / direct_nullable_us,
    );
    println!(
        "# filtered direct = {:.0}% of the sum roofline; nullable direct = {:.0}%.",
        100.0 * mps(direct_filtered_us) / mps(sum_us),
        100.0 * mps(direct_nullable_us) / mps(sum_us),
    );
}
