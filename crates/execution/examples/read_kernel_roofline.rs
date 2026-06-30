//! Read-kernel roofline sweep: drive each unique READ kernel directly on a resident column at a fixed N
//! and report effective bandwidth (or throughput) so we can classify each as saturated (done) vs headroom.
//!
//! WHAT EACH LINE ACTUALLY MEASURES (a representativeness audit corrected this; read before trusting a
//! number as "kernel-only"):
//!   * Section (1) = RESIDENT-INPUT scans. Input is already device-resident (engine indices/columns), so
//!     there is NO per-call input H2D -- wall ~= kernel. THESE are the kernel-clean lines.
//!   * Section (2)/(3) = HOST-INPUT ops (gather/sort/join). They take a HOST slice and upload it to the
//!     device EVERY call (gather ~6MB index, sort ~12MB keys, join ~16MB keys). The engine has those
//!     inputs device-resident (from a filter / a resident column), so the wall here OVERSTATES the kernel
//!     by that per-call H2D. Each line is LABELED with the H2D it includes.
//!
//! THE ROOFLINE is `sum_i32` (a pure 1-pass read+reduce -> HBM streaming peak). `equal_any` is also
//! measured but is NOT the roofline: it does an 8-needle compare per element, so it is compute-bound and
//! runs ~2x slower than a pure read. "% of roofline" is normalized against `sum_i32`.
//!
//! Sections:
//!   (1) BANDWIDTH-bound scans -- GB/s = input-bytes-touched / wall, vs the `sum_i32` read roofline.
//!       >=~80% of roof = saturated. 2-pass kernels (the ordered compaction) stream the column twice, so
//!       ~half the roof BY DESIGN.
//!   (2) ACCESS-bound gather -- scattered reads, GB/s of useful bytes (cache-line-bound, not streaming);
//!       wall includes a per-call index H2D (the single-launch gather kernel is isolated via the CUDA
//!       event when the pooled stream is timed -- see the gather block).
//!   (3) ALGORITHMIC -- sort / join / grouped: reported as M-elem/s (NOT pure bandwidth; O(n log^2 n) /
//!       random / hash-cardinality bound). Run at a smaller SORT_N. sort/join wall includes a per-call key
//!       H2D; the GROUP BY line reports the aggregate KERNEL only (CUDA-event timed) AND its full wall.
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

    println!("\n# (1) RESIDENT-INPUT scans -- kernel-clean, wall ~= kernel (vs the sum_i32 read roofline) ---");
    let needles_miss: Vec<i32> = (0..8).map(|k| -((k as i32) + 1)).collect(); // negative -> never match A
    // ROOFLINE = sum_i32: a pure 1-pass read+reduce over the same 32MB column -> the HBM streaming peak.
    let roof = bench("sum_i32 (ROOFLINE: pure 1-pass read+reduce)", g4, Box::new(|| resident.sum_i32_from_payload(off_a, rows).unwrap() as usize));
    // equal_any is measured for comparison but is NOT the roofline: an 8-needle compare per element makes
    // it compute-bound (~2x slower than a pure read), so it is a data point, not the read ceiling.
    bench(
        "equal_any (8-needle scan -- NOT roofline; ~2x a read)",
        g4,
        Box::new(|| resident.match_project_i32_equal_any_from_payload(off_a, &needles_miss, &[off_b], rows).unwrap().len()),
    );
    bench("count_i32_compare (1-pass)", g4, Box::new(|| resident.count_i32_compare_from_payload(off_a, rows, 1 << 19, CudaI32Comparison::Lt).unwrap() as usize));
    bench("count_i32_between (1-pass)", g4, Box::new(|| resident.count_i32_between_from_payload(off_a, rows, 0, 1 << 19).unwrap() as usize));
    bench("expr_i64_compare_scalar (8B, ~1% sel)", g8, Box::new(|| resident.expr_i64_compare_scalar_filter(off_c, 80_000_i64 * 2_654_435_761, false, 1, rows).unwrap().len()));
    bench("expr_i128_compare_scalar (16B, ~1% sel)", g16, Box::new(|| resident.expr_i128_compare_scalar_filter(off_d, 80_000_i128 * 11, false, 1, rows).unwrap().len()));
    let arith = vec![ExprStep::LoadColumn { byte_offset: off_a }, ExprStep::ScalarBinary { op: 0, scalar: 5, scalar_on_left: false }];
    bench("arith_filter a+5<k (load+binop, 2-pass)", 2.0 * g4, Box::new(|| resident.run_expr_arith_filter(&arith, rows, 1, 1 << 19).unwrap().len()));
    bench("compare_indices_ordered (2-pass, ~50% sel)", 2.0 * g4, Box::new(|| resident.compare_indices_ordered_from_payload(off_a, rows, 1 << 19, 1).unwrap().len()));
    bench("project_compare ordered (2-pass, ~1% sel)", 2.0 * g4, Box::new(|| resident.project_i32_compare_from_payload(off_a, rows, 1 << 13, CudaI32Comparison::Lt).unwrap().len()));

    println!("\n# (2) HOST-INPUT gather -- scattered, cache-line bound; wall INCLUDES a per-call index H2D ---");
    let idx: Vec<u64> = (0..rows).step_by(16).collect();
    let gidx_gb4 = (idx.len() * 4) as f64 / 1e9;
    let gidx_gb8 = (idx.len() * 8) as f64 / 1e9;
    let idx_h2d_mb = (idx.len() * 8) as f64 / 1e6; // u64 index slice uploaded every call
    {
        let i = &idx;
        // gather is a SINGLE-launch kernel and the index H2D happens BEFORE the CUDA event bracket
        // (the event wraps only the kernel launch). So when the pooled stream is event-timed we can
        // isolate the kernel: clear, run, read the event back. Verified < wall (wall = H2D + kernel +
        // D2H + host alloc). If the pool is untimed (event = None) we fall back to the H2D-labeled wall.
        bench(&format!("gather_i32 (project_i32_rows) (+~{idx_h2d_mb:.0}MB idx H2D)"), gidx_gb4, Box::new(|| resident.project_i32_rows_from_payload(off_b, i).unwrap().len()));
        resident.clear_last_kernel_event_elapsed_us();
        let n_gathered = resident.project_i32_rows_from_payload(off_b, i).unwrap().len();
        match resident.last_kernel_event_elapsed_us() {
            Some(ev_us) if ev_us > 0 => {
                let kus = ev_us as f64;
                println!("  {:<40} {:>9.0}us  {:>8.1} GB/s   (kernel-only, CUDA-event; n {n_gathered})", "  ^ gather_i32 KERNEL (no H2D)", kus, gidx_gb4 / (kus / 1e6));
            }
            _ => println!("  {:<40}            (pooled stream untimed -- gather kernel not isolable; use the H2D-labeled wall above)", "  ^ gather_i32 KERNEL"),
        }
        bench(&format!("gather_i64 (project_i64_rows) (+~{idx_h2d_mb:.0}MB idx H2D)"), gidx_gb8, Box::new(|| resident.project_i64_rows_from_payload(off_c, i).unwrap().len()));
    }

    println!("\n# (3) ALGORITHMIC (sort / join / grouped -- M-elem/s, NOT pure bandwidth) ---");
    println!("#     sort/join wall INCLUDES a per-call key H2D (multi-launch -> the CUDA event is only the");
    println!("#     last pass, so the kernel is NOT isolated here -- the H2D is labeled instead).");
    let keys: Vec<i64> = (0..sort_n).map(|r| hash(r) as i64).collect();
    let sort_h2d_mb = (sort_n as usize * 8) as f64 / 1e6; // i64 key slice uploaded every call
    let join_h2d_mb = (sort_n as usize * 8 * 2) as f64 / 1e6; // build + probe key slices
    {
        let k = &keys;
        throughput(&format!("bitonic_sort_i64 (+~{sort_h2d_mb:.0}MB key H2D)"), sort_n, Box::new(|| resident.bitonic_sort_i64(k, false).unwrap().len()));
    }
    let build: Vec<i64> = (0..sort_n).map(|r| r as i64).collect();
    let probe: Vec<i64> = (0..sort_n).map(|r| hash(r) as i64 % sort_n as i64).collect();
    {
        let (bk, pk) = (&build, &probe);
        throughput(&format!("hash_join_inner_i64 build+probe (+~{join_h2d_mb:.0}MB key H2D)"), sort_n, Box::new(|| {
            match resident.hash_join_inner_i64(bk, pk, None, None) {
                Ok(HashJoinOutcome::Pairs { probe_idxs, .. }) => probe_idxs.len(),
                _ => 0,
            }
        }));
    }
    // LIVE per-group GROUP BY (the two-level shared-mem kernel the engine uses), grouping by the
    // ~1M-distinct key column A and summing B over a full-table scan (indices = 0..rows, as the executor
    // passes for an unfiltered GROUP BY). Replaces the removed `grouped_stats_i32` hash-agg measurement.
    //
    // The kernel-timed call (CUDA-event over the LIVE two-level kernel, runs=10) is the HONEST aggregate
    // KERNEL ms. The `from_payload` wall printed alongside is the END-TO-END result path: per-call it pays
    // the 8M-row index H2D (32MB; the engine's indices are already device-resident from its on-device
    // filter -- an artifact here), + the ~2*row_count slot-table setup, + the host build of the result
    // Vec. NONE of that is the kernel. See grouped_cardinality_probe for the end-to-end result path.
    let gb_indices: Vec<u32> = (0..rows as u32).collect();
    {
        let gi = &gb_indices;
        // Honest KERNEL: CUDA-event timed, two-level kernel, 10 runs, full-compute mask.
        let (_rows, kernel_ms) = resident
            .group_by_i32_count_sum_kernel_timed(off_a, off_b, gi, true, 10, gpu_db_execution::grouped_agg_mask::ALL)
            .expect("group_by kernel_timed");
        let kernel_melem_s = rows as f64 / (kernel_ms as f64 * 1e3); // rows / (ms*1000 us) = rows/us = Melem/s
        println!("  {:<40} {:>7.3}ms  {:>8.1} Melem/s (KERNEL only, CUDA-event)", "group_by_i32 KERNEL (~1M groups)", kernel_ms, kernel_melem_s);
        // Full result path (index H2D + ~2*row_count table setup + host Vec build), labeled NON-kernel.
        throughput("  ^ group_by_i32 from_payload (FULL PATH)", rows, Box::new(|| resident.group_by_i32_count_sum_from_payload(off_a, off_b, gi, gpu_db_execution::grouped_agg_mask::ALL).unwrap().len()));
        println!("  {:<40}            (+index H2D + ~2*row_count table setup + host Vec build -- see grouped_cardinality_probe)", "");
    }

    println!("\n# roofline = {roof:.0} GB/s (sum_i32 = pure 1-pass read+reduce ~= HBM peak). >=~80% on a 1-pass scan = saturated.");
    println!("# NOTE: section (1) is resident-input (kernel-clean, wall ~= kernel). gather/sort/join wall");
    println!("#       INCLUDES a per-call input H2D the engine does NOT pay (inputs are device-resident);");
    println!("#       the GROUP BY line shows the aggregate KERNEL (event-timed) vs its full result path.");
}
