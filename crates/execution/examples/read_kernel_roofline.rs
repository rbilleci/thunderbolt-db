//! Read-kernel roofline sweep: drive each unique READ kernel directly on a resident column at a fixed N
//! and report effective bandwidth (or throughput) so we can classify each as saturated (done) vs headroom.
//! Kernel-only timing (the `CudaResidentDeviceMemory` methods = launch + sync + small-result D2H).
//!
//! Three sections:
//!   (1) BANDWIDTH-bound scans -- GB/s = input-bytes-touched / wall, vs the measured streaming roofline
//!       (`equal_any` over a non-matching needle = a pure coalesced read). >=~80% of roof = saturated.
//!       2-pass kernels (the ordered compaction) stream the column twice, so ~half the roof BY DESIGN.
//!   (2) ACCESS-bound gather -- scattered reads, GB/s of useful bytes (cache-line-bound, not streaming).
//!   (3) ALGORITHMIC -- sort / join / grouped: reported as M-elem/s (NOT pure bandwidth; O(n log^2 n) /
//!       random / hash-cardinality bound). Run at a smaller SORT_N.
//!
//! Run (never --gpu-reset, always under timeout):
//!   timeout 300 cargo run --release --example read_kernel_roofline -p gpu_db_execution

use std::time::Instant;

use gpu_db_execution::{
    CudaDeviceMemoryChunk, CudaDriverRuntime, CudaI32Comparison, ExprStep, HashJoinOutcome,
};

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let rows: u64 = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8_388_608);
    let sort_n: u64 = std::env::var("SORT_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_048_576);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let n = rows as usize;

    let Ok(runtime) = CudaDriverRuntime::probe() else {
        eprintln!("no local NVIDIA driver/GPU; skipping");
        return;
    };

    // Layout: [8B row_count][A: N i32][B: N i32][C: N i64][D: N i128]. A = scrambled key in [0,1<<20);
    // B/C/D = value columns of each width (for the per-width filter + gather sweeps).
    let hash = |row: u64| -> u32 { ((row.wrapping_mul(2_654_435_761)) ^ (row << 13)) as u32 };
    let off_a = 8u64;
    let off_b = off_a + rows * 4;
    let off_c = off_b + rows * 4;
    let off_d = off_c + rows * 8;
    let allocated = off_d + rows * 16;
    let mut header = rows.to_le_bytes().to_vec();
    let mut a = Vec::with_capacity(n * 4);
    let mut b = Vec::with_capacity(n * 4);
    let mut c = Vec::with_capacity(n * 8);
    let mut d = Vec::with_capacity(n * 16);
    for row in 0..rows {
        a.extend_from_slice(&((hash(row) % (1 << 20)) as i32).to_le_bytes());
        b.extend_from_slice(&((row as i32).wrapping_mul(7)).to_le_bytes());
        c.extend_from_slice(&((row as i64).wrapping_mul(2_654_435_761)).to_le_bytes());
        d.extend_from_slice(&((row as i128).wrapping_mul(11)).to_le_bytes());
    }
    let _ = &mut header;
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated,
            &[
                CudaDeviceMemoryChunk { byte_offset: 0, bytes: &header },
                CudaDeviceMemoryChunk { byte_offset: off_a, bytes: &a },
                CudaDeviceMemoryChunk { byte_offset: off_b, bytes: &b },
                CudaDeviceMemoryChunk { byte_offset: off_c, bytes: &c },
                CudaDeviceMemoryChunk { byte_offset: off_d, bytes: &d },
            ],
        )
        .expect("retain resident device memory");

    // bench: `gb` = the input bytes touched (GB) for this kernel; reports GB/s at the wall p50.
    let bench = |label: &str, gb: f64, mut run: Box<dyn FnMut() -> usize>| -> f64 {
        for _ in 0..3 {
            run();
        }
        let mut s = Vec::with_capacity(iters);
        let mut sink = 0usize;
        for _ in 0..iters {
            let t = Instant::now();
            sink += run();
            s.push(t.elapsed().as_nanos());
        }
        let us = p50(s) as f64 / 1000.0;
        let gbps = gb / (us / 1e6);
        println!("  {label:<40} {us:>9.0}us  {:>8.1} GB/s   (sink {sink})", gbps);
        gbps
    };
    let throughput = |label: &str, elems: u64, mut run: Box<dyn FnMut() -> usize>| {
        for _ in 0..3 {
            run();
        }
        let mut s = Vec::with_capacity(iters);
        let mut sink = 0usize;
        for _ in 0..iters {
            let t = Instant::now();
            sink += run();
            s.push(t.elapsed().as_nanos());
        }
        let us = p50(s) as f64 / 1000.0;
        println!("  {label:<40} {us:>9.0}us  {:>8.1} Melem/s (sink {sink})", elems as f64 / us);
    };

    let g4 = (rows * 4) as f64 / 1e9;
    let g8 = (rows * 8) as f64 / 1e9;
    let g16 = (rows * 16) as f64 / 1e9;
    println!("# read-kernel roofline. rows={rows}, sort_n={sort_n}, iters={iters}\n");
    println!("  {:<40} {:>9}  {:>8}", "kernel", "p50", "rate");

    println!("\n# (1) BANDWIDTH-bound scans  (vs the equal_any read roofline) ---");
    let needles_miss: Vec<i32> = (0..8).map(|k| -((k as i32) + 1)).collect(); // negative -> never match A
    let roof = bench(
        "equal_any (ROOFLINE: pure 1-pass read)",
        g4,
        Box::new(|| resident.match_project_i32_equal_any_from_payload(off_a, &needles_miss, &[off_b], rows).unwrap().len()),
    );
    bench("sum_i32 (1-pass read+reduce)", g4, Box::new(|| resident.sum_i32_from_payload(off_a, rows).unwrap() as usize));
    bench("count_i32_compare (1-pass)", g4, Box::new(|| resident.count_i32_compare_from_payload(off_a, rows, 1 << 19, CudaI32Comparison::Lt).unwrap() as usize));
    bench("count_i32_between (1-pass)", g4, Box::new(|| resident.count_i32_between_from_payload(off_a, rows, 0, 1 << 19).unwrap() as usize));
    bench("expr_i64_compare_scalar (8B col, 1-pass)", g8, Box::new(|| resident.expr_i64_compare_scalar_filter(off_c, i64::MAX / 2, false, 1, rows).unwrap().len()));
    bench("expr_i128_compare_scalar (16B col, 1-pass)", g16, Box::new(|| resident.expr_i128_compare_scalar_filter(off_d, i128::MAX / 2, false, 1, rows).unwrap().len()));
    let arith = vec![ExprStep::LoadColumn { byte_offset: off_a }, ExprStep::ScalarBinary { op: 0, scalar: 5, scalar_on_left: false }];
    bench("arith_filter a+5<k (load+binop, 2-pass)", 2.0 * g4, Box::new(|| resident.run_expr_arith_filter(&arith, rows, 1, 1 << 19).unwrap().len()));
    bench("compare_indices_ordered (2-pass, ~50% sel)", 2.0 * g4, Box::new(|| resident.compare_indices_ordered_from_payload(off_a, rows, 1 << 19, 1).unwrap().len()));
    bench("project_compare ordered (2-pass, ~1% sel)", 2.0 * g4, Box::new(|| resident.project_i32_compare_from_payload(off_a, rows, 1 << 13, CudaI32Comparison::Lt).unwrap().len()));

    println!("\n# (2) ACCESS-bound gather (scattered, cache-line bound) ---");
    let idx: Vec<u64> = (0..rows).step_by(16).collect();
    let gidx_gb4 = (idx.len() * 4) as f64 / 1e9;
    let gidx_gb8 = (idx.len() * 8) as f64 / 1e9;
    {
        let i = &idx;
        bench("gather_i32 (project_i32_rows)", gidx_gb4, Box::new(|| resident.project_i32_rows_from_payload(off_b, i).unwrap().len()));
        bench("gather_i64 (project_i64_rows)", gidx_gb8, Box::new(|| resident.project_i64_rows_from_payload(off_c, i).unwrap().len()));
    }

    println!("\n# (3) ALGORITHMIC (sort / join / grouped -- M-elem/s, NOT pure bandwidth) ---");
    let keys: Vec<i64> = (0..sort_n).map(|r| hash(r) as i64).collect();
    {
        let k = &keys;
        throughput("bitonic_sort_i64", sort_n, Box::new(|| resident.bitonic_sort_i64(k, false).unwrap().len()));
    }
    let build: Vec<i64> = (0..sort_n).map(|r| r as i64).collect();
    let probe: Vec<i64> = (0..sort_n).map(|r| hash(r) as i64 % sort_n as i64).collect();
    {
        let (bk, pk) = (&build, &probe);
        throughput("hash_join_inner_i64 (build+probe)", sort_n, Box::new(|| {
            match resident.hash_join_inner_i64(bk, pk, None, None) {
                Ok(HashJoinOutcome::Pairs { probe_idxs, .. }) => probe_idxs.len(),
                _ => 0,
            }
        }));
    }
    throughput("grouped_stats_i32 (~1M groups)", rows, Box::new(|| resident.grouped_stats_i32_from_payload(off_a, off_b, rows).unwrap().len()));

    println!("\n# roofline = {roof:.0} GB/s (equal_any read). >=~80% on a 1-pass scan = saturated.");
}
