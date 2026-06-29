//! Read-kernel roofline sweep: drive each unique READ kernel directly on a resident column at a fixed N
//! and report effective bandwidth vs a measured streaming roofline, to see which kernels are saturated
//! (done) vs which have headroom. Kernel-only timing (the `CudaResidentDeviceMemory` methods are a launch
//! + a sync + a bulk D2H of the small result), so the number reflects the kernel + its transfers, not the
//! engine's host materialization.
//!
//! GB/s is computed as the kernel's PRIMARY input traffic / wall: a single-column scan touches N*4 bytes;
//! the ordered compaction scans the column TWICE (count pass + scatter pass) so it shows ~half the roofline
//! BY DESIGN (it does 2x the traffic) -- that is the signal, not a defect. `% roof` = GB/s / the sum-scan
//! roofline (a pure coalesced column read). >=~80% = bandwidth-saturated; much lower = headroom or a
//! non-streaming access pattern (e.g. the gather's scattered reads).
//!
//! Run (RTX/datacenter box; never --gpu-reset, always under timeout):
//!   timeout 240 cargo run --release --example read_kernel_roofline -p gpu_db_execution

use std::time::Instant;

use gpu_db_execution::{CudaDeviceMemoryChunk, CudaDriverRuntime, CudaI32Comparison};

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let rows: u64 = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(16_777_216);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let n = rows as usize;

    let Ok(runtime) = CudaDriverRuntime::probe() else {
        eprintln!("no local NVIDIA driver/GPU; skipping");
        return;
    };

    // Layout: [8-byte row_count header][col A: N i32][col B: N i32]. A = a scrambled key in [0, 1<<20)
    // (so a `< k` filter has tunable selectivity and GROUP BY has ~1M groups capped); B = a value column.
    let off_a = std::mem::size_of::<u64>() as u64;
    let off_b = off_a + (rows * 4);
    let hash = |row: u64| -> u32 { ((row.wrapping_mul(2_654_435_761)) ^ (row << 13)) as u32 };
    let mut header = Vec::with_capacity(8);
    header.extend_from_slice(&rows.to_le_bytes());
    let mut a_bytes = Vec::with_capacity(n * 4);
    let mut b_bytes = Vec::with_capacity(n * 4);
    for row in 0..rows {
        a_bytes.extend_from_slice(&((hash(row) % (1 << 20)) as i32).to_le_bytes());
        b_bytes.extend_from_slice(&((row as i32).wrapping_mul(7)).to_le_bytes());
    }
    let allocated = off_b + (rows * 4);
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated,
            &[
                CudaDeviceMemoryChunk { byte_offset: 0, bytes: &header },
                CudaDeviceMemoryChunk { byte_offset: off_a, bytes: &a_bytes },
                CudaDeviceMemoryChunk { byte_offset: off_b, bytes: &b_bytes },
            ],
        )
        .expect("retain resident device memory");

    let col_gb = (rows * 4) as f64 / 1e9;
    // Each closure returns (input-bytes-touched-in-GB, the work) so we can print GB/s consistently.
    let bench = |label: &str, passes: f64, mut run: Box<dyn FnMut() -> usize>| -> f64 {
        for _ in 0..3 {
            run();
        } // warmup
        let mut samples = Vec::with_capacity(iters);
        let mut sink = 0usize;
        for _ in 0..iters {
            let t = Instant::now();
            sink += run();
            samples.push(t.elapsed().as_nanos());
        }
        let us = p50(samples) as f64 / 1000.0;
        let gbps = (passes * col_gb) / (us / 1e6);
        println!("  {label:<34} {us:>9.0}us  {:>8.1} GB/s   (sink {sink})", gbps);
        gbps
    };

    println!("# read-kernel roofline. rows={rows} (col={:.0} MB), iters={iters}\n", col_gb * 1000.0);
    println!("  {:<34} {:>9}  {:>8}", "kernel", "p50", "GB/s");

    // ---- roofline: a pure coalesced column read (sum reduce, 1 pass) ----
    let roof = bench(
        "sum_i32 (ROOFLINE: 1-pass read)",
        1.0,
        Box::new(|| resident.sum_i32_from_payload(off_a, rows).unwrap() as usize),
    );

    println!("\n# --- filter / predicate (read the column, emit a mask/indices) ---");
    bench(
        "count_i32_compare (1-pass)",
        1.0,
        Box::new(|| resident.count_i32_compare_from_payload(off_a, rows, 1 << 19, CudaI32Comparison::Lt).unwrap() as usize),
    );
    // The VM lever: ordered compaction -> ascending row indices (2 passes: count + scatter). ~50% sel.
    bench(
        "compare_indices_ordered (2-pass, ~50% sel)",
        2.0,
        Box::new(|| resident.compare_indices_ordered_from_payload(off_a, rows, 1 << 19, 1).unwrap().len()),
    );
    // Ordered VALUES (the project-ordered route; also 2-pass) -- low selectivity so the D2H stays small.
    bench(
        "project_i32_compare ordered (2-pass, ~1% sel)",
        2.0,
        Box::new(|| resident.project_i32_compare_from_payload(off_a, rows, 1 << 13, CudaI32Comparison::Lt).unwrap().len()),
    );

    println!("\n# --- scan-project / gather / grouped ---");
    // equal_any fused scan+project: scan col A for a few needles, project col B (1 pass over the filter).
    let needles: Vec<i32> = (0..8).map(|k| (hash(k * 50_000) % (1 << 20)) as i32).collect();
    bench(
        "match_project_equal_any (1-pass scan+proj)",
        1.0,
        Box::new(|| resident.match_project_i32_equal_any_from_payload(off_a, &needles, &[off_b], rows).unwrap().len()),
    );
    // Gather: read col B at ~1M scattered row indices (random access, not streaming).
    let gather_idx: Vec<u64> = (0..rows).step_by(16).collect();
    let gather_gb = (gather_idx.len() * 4) as f64 / 1e9;
    {
        let g = &gather_idx;
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..3 {
            resident.project_i32_rows_from_payload(off_b, g).unwrap();
        }
        let mut sink = 0usize;
        for _ in 0..iters {
            let t = Instant::now();
            sink += resident.project_i32_rows_from_payload(off_b, g).unwrap().len();
            samples.push(t.elapsed().as_nanos());
        }
        let us = p50(samples) as f64 / 1000.0;
        println!("  {:<34} {us:>9.0}us  {:>8.1} GB/s   (sink {sink}, {} idx, SCATTERED)", "gather (project_i32_rows)", gather_gb / (us / 1e6), g.len());
    }
    // Grouped aggregation: read group col A + value col B (2 columns -> 2 passes of traffic).
    bench(
        "grouped_stats_i32 (2-col read)",
        2.0,
        Box::new(|| resident.grouped_stats_i32_from_payload(off_a, off_b, rows).unwrap().len()),
    );

    println!("\n# roofline = {roof:.0} GB/s. >=~80% of roof on a 1-pass kernel = bandwidth-saturated.");
    println!("# 2-pass kernels show ~half (they stream the column twice); gather is scattered (random).");
}
