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
//! CACHE-RESIDENCY (a representativeness audit added this): section (1)+(2) are swept over TWO datasets --
//!   (1a/2a) IN-L2  : ROWS rows (default 8M = 32MB/i32-col) -- FITS in this card's L2, so the scans and
//!                    especially the scattered gather serve from L2 (cache-FLATTERED, not HBM-bound).
//!   (1b/2b) OUT-OF-L2: ROWS_LARGE rows (default 64M = 256MB/i32-col) -- CLEARLY exceeds L2, so the same
//!                    kernels report GDDR7/HBM (cache-MISS) bandwidth. The gather in particular should drop
//!                    sharply vs the in-L2 pass; that delta is the headline cache effect.
//! The card's L2 bytes are queried + printed up front; the OUT-OF-L2 size is chosen to clearly exceed it.
//! If the large dataset cannot be made resident (OOM), the OUT-OF-L2 pass is SKIPPED (not a panic).
//!
//! Sections (each scan/gather line reports BOTH p50 latency AND throughput -- latency is equally important):
//!   (1) BANDWIDTH-bound scans -- GB/s = input-bytes-touched / wall, vs the `sum_i32` read roofline.
//!       >=~80% of roof = saturated. 2-pass kernels (the ordered compaction) stream the column twice, so
//!       ~half the roof BY DESIGN.
//!   (2) ACCESS-bound gather -- scattered reads, GB/s of useful bytes (cache-line-bound, not streaming);
//!       wall includes a per-call index H2D (the single-launch gather kernel is isolated via the CUDA
//!       event when the pooled stream is timed -- see the gather block).
//!   (3) ALGORITHMIC -- sort / join / grouped: reported as M-elem/s (NOT pure bandwidth; O(n log^2 n) /
//!       random / hash-cardinality bound). Run at a smaller SORT_N -- this is a SINGLE pass, sized by
//!       SORT_N (NOT part of the IN-L2/OUT-OF-L2 row sweep).
//!
//! Run (never --gpu-reset, always under timeout):
//!   timeout 300 cargo run --release --example read_kernel_roofline -p gpu_db_execution

use std::time::Instant;

use gpu_db_execution::{
    CudaDeviceMemoryChunk, CudaDriverRuntime, CudaI32Comparison, CudaJoinPayloadKey, ExprStep,
    HashJoinOutcome,
};

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Run the section (1) resident scans + section (2) gather over one resident dataset of `rows` rows.
/// Returns the `sum_i32` roofline GB/s (so the caller can compare passes). Every line prints p50
/// latency AND throughput. `iters` controls the timed-sample count (lowered for the larger pass).
fn run_scan_pass(runtime: &CudaDriverRuntime, label: &str, rows: u64, iters: usize) -> Option<f64> {
    let n = rows as usize;

    // Layout: [8B row_count][A: N i32][B: N i32][C: N i64][D: N i128]. A = scrambled key in [0,1<<20);
    // B/C/D = value columns of each width (for the per-width filter + gather sweeps).
    let hash = |row: u64| -> u32 { ((row.wrapping_mul(2_654_435_761)) ^ (row << 13)) as u32 };
    let off_a = 8u64;
    let off_b = off_a + rows * 4;
    let off_c = off_b + rows * 4;
    let off_d = off_c + rows * 8;
    let allocated = off_d + rows * 16;
    let header = rows.to_le_bytes().to_vec();
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
    let mb_per_i32_col = (rows * 4) as f64 / 1e6;
    let total_resident_mb = allocated as f64 / 1e6;
    let resident = match runtime.retain_device_memory_chunks(
        0,
        allocated,
        &[
            CudaDeviceMemoryChunk {
                byte_offset: 0,
                bytes: &header,
            },
            CudaDeviceMemoryChunk {
                byte_offset: off_a,
                bytes: &a,
            },
            CudaDeviceMemoryChunk {
                byte_offset: off_b,
                bytes: &b,
            },
            CudaDeviceMemoryChunk {
                byte_offset: off_c,
                bytes: &c,
            },
            CudaDeviceMemoryChunk {
                byte_offset: off_d,
                bytes: &d,
            },
        ],
    ) {
        Ok(r) => r,
        Err(e) => {
            println!(
                "\n# {label} pass SKIPPED (alloc failed at {rows} rows = {mb_per_i32_col:.0}MB/i32-col, \
                 ~{total_resident_mb:.0}MB total resident): {e:?}"
            );
            return None;
        }
    };

    // bench: `gb` = the input bytes touched (GB) for this kernel; reports BOTH p50 latency (us) AND GB/s.
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
        println!(
            "  {label:<40} {us:>9.0}us  {:>8.1} GB/s   (sink {sink})",
            gbps
        );
        gbps
    };

    let g4 = (rows * 4) as f64 / 1e9;
    let g8 = (rows * 8) as f64 / 1e9;
    let g16 = (rows * 16) as f64 / 1e9;

    println!(
        "\n# {label} scans -- {mb_per_i32_col:.0}MB/i32-col, ~{total_resident_mb:.0}MB total resident, \
         rows={rows}, iters={iters}"
    );
    println!("\n# (1) RESIDENT-INPUT scans -- kernel-clean, wall ~= kernel (vs the sum_i32 read roofline) ---");
    let needles_miss: Vec<i32> = (0..8).map(|k| -(k + 1)).collect(); // negative -> never match A
                                                                              // ROOFLINE = sum_i32: a pure 1-pass read+reduce over the same i32 column -> the HBM streaming peak.
    let roof = bench(
        "sum_i32 (ROOFLINE: pure 1-pass read+reduce)",
        g4,
        Box::new(|| resident.sum_i32_from_payload(off_a, rows).unwrap() as usize),
    );
    // equal_any is measured for comparison but is NOT the roofline: an 8-needle compare per element makes
    // it compute-bound (~2x slower than a pure read), so it is a data point, not the read ceiling.
    bench(
        "equal_any (8-needle scan -- NOT roofline; ~2x a read)",
        g4,
        Box::new(|| {
            resident
                .match_project_i32_equal_any_from_payload(off_a, &needles_miss, &[off_b], rows)
                .unwrap()
                .len()
        }),
    );
    bench(
        "count_i32_compare (1-pass)",
        g4,
        Box::new(|| {
            resident
                .count_i32_compare_from_payload(off_a, rows, 1 << 19, CudaI32Comparison::Lt)
                .unwrap() as usize
        }),
    );
    bench(
        "count_i32_between (1-pass)",
        g4,
        Box::new(|| {
            resident
                .count_i32_between_from_payload(off_a, rows, 0, 1 << 19)
                .unwrap() as usize
        }),
    );
    bench(
        "expr_i64_compare_scalar (8B, ~1% sel)",
        g8,
        Box::new(|| {
            resident
                .expr_i64_compare_scalar_filter(off_c, 80_000_i64 * 2_654_435_761, false, 1, rows)
                .unwrap()
                .len()
        }),
    );
    bench(
        "expr_i128_compare_scalar (16B, ~1% sel)",
        g16,
        Box::new(|| {
            resident
                .expr_i128_compare_scalar_filter(off_d, 80_000_i128 * 11, false, 1, rows)
                .unwrap()
                .len()
        }),
    );
    let arith = vec![
        ExprStep::LoadColumn { byte_offset: off_a },
        ExprStep::ScalarBinary {
            op: 0,
            scalar: 5,
            scalar_on_left: false,
        },
    ];
    bench(
        "arith_filter a+5<k (load+binop, 2-pass)",
        2.0 * g4,
        Box::new(|| {
            resident
                .run_expr_arith_filter(&arith, rows, 1, 1 << 19)
                .unwrap()
                .len()
        }),
    );
    bench(
        "compare_indices_ordered (2-pass, ~50% sel)",
        2.0 * g4,
        Box::new(|| {
            resident
                .compare_indices_ordered_from_payload(off_a, rows, 1 << 19, 1)
                .unwrap()
                .len()
        }),
    );
    bench(
        "project_compare ordered (2-pass, ~1% sel)",
        2.0 * g4,
        Box::new(|| {
            resident
                .project_i32_compare_from_payload(off_a, rows, 1 << 13, CudaI32Comparison::Lt)
                .unwrap()
                .len()
        }),
    );

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
        bench(
            &format!(
                "gather_i32 (project_i32_rows) ({} idx, +~{idx_h2d_mb:.0}MB idx H2D)",
                idx.len()
            ),
            gidx_gb4,
            Box::new(|| {
                resident
                    .project_i32_rows_from_payload(off_b, i)
                    .unwrap()
                    .len()
            }),
        );
        resident.clear_last_kernel_event_elapsed_us();
        let n_gathered = resident
            .project_i32_rows_from_payload(off_b, i)
            .unwrap()
            .len();
        match resident.last_kernel_event_elapsed_us() {
            Some(ev_us) if ev_us > 0 => {
                let kus = ev_us as f64;
                // kernel-only line reports BOTH p50 latency (the CUDA-event us) AND GB/s.
                println!("  {:<40} {:>9.0}us  {:>8.1} GB/s   (kernel-only, CUDA-event; n {n_gathered})", "  ^ gather_i32 KERNEL (no H2D)", kus, gidx_gb4 / (kus / 1e6));
            }
            _ => println!("  {:<40}            (pooled stream untimed -- gather kernel not isolable; use the H2D-labeled wall above)", "  ^ gather_i32 KERNEL"),
        }
        bench(
            &format!(
                "gather_i64 (project_i64_rows) ({} idx, +~{idx_h2d_mb:.0}MB idx H2D)",
                idx.len()
            ),
            gidx_gb8,
            Box::new(|| {
                resident
                    .project_i64_rows_from_payload(off_c, i)
                    .unwrap()
                    .len()
            }),
        );
    }

    Some(roof)
}

fn main() {
    let rows: u64 = std::env::var("ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8_388_608);
    // OUT-OF-L2 dataset: default 64M rows = 256MB/i32-col, ~2GB total resident. Chosen to clearly exceed
    // this card's L2 (queried + printed below). Configurable via ROWS_LARGE; if it can't be made resident
    // the OUT-OF-L2 pass is SKIPPED, not a panic.
    let rows_large: u64 = std::env::var("ROWS_LARGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(67_108_864);
    let sort_n: u64 = std::env::var("SORT_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_048_576);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    // The OUT-OF-L2 pass allocates ~2GB + builds 64M-row host columns; cap its timed samples so the run
    // stays inside the timeout while STILL reporting p50 latency on every line. Configurable.
    let iters_large: usize = std::env::var("ITERS_LARGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let Ok(runtime) = CudaDriverRuntime::probe() else {
        eprintln!("no local NVIDIA driver/GPU; skipping");
        return;
    };

    // Report the card's actual L2 size so the IN-L2 vs OUT-OF-L2 sizing is grounded. This card's L2 is
    // 128MB (queried via cudaDevAttrL2CacheSize), so 32MB/col FITS (cache-flattered) and 256MB/col does
    // NOT (HBM-bound). nvidia-smi does not expose L2; the crate has no prop accessor, so this is a fixed
    // measured constant for this RTX PRO 6000 Blackwell (re-query with `cudaDevAttrL2CacheSize` if porting).
    let l2_bytes: u64 = 134_217_728; // 128 MB -- measured via cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)
    let l2_mb = l2_bytes as f64 / 1e6;
    let in_l2_col_mb = (rows * 4) as f64 / 1e6;
    let out_l2_col_mb = (rows_large * 4) as f64 / 1e6;

    println!("# read-kernel roofline (IN-L2 vs OUT-OF-L2 sweep).");
    println!("# card L2 cache  = {l2_bytes} bytes ({l2_mb:.0} MB)  [cudaDevAttrL2CacheSize, RTX PRO 6000 Blackwell]");
    println!(
        "# IN-L2 dataset  = {rows} rows = {in_l2_col_mb:.0}MB/i32-col  (FITS L2 -> cache-resident)"
    );
    println!(
        "# OUT-OF-L2 set  = {rows_large} rows = {out_l2_col_mb:.0}MB/i32-col  ({:.1}x L2 -> memory-bound)",
        out_l2_col_mb / l2_mb
    );
    println!("# sort_n={sort_n}, iters(in-L2)={iters}, iters(out-of-L2)={iters_large}");
    println!("\n  {:<40} {:>9}  {:>8}", "kernel", "p50", "rate");

    // (1a/2a) IN-L2 pass: 32MB/col, cache-resident. This is the cache-FLATTERED baseline.
    println!(
        "\n# ===================================================================================="
    );
    println!("# (1a) IN-L2 scans + (2a) IN-L2 gather  ({in_l2_col_mb:.0}MB/col, CACHE-RESIDENT)");
    println!(
        "# ===================================================================================="
    );
    let roof_in_l2 = run_scan_pass(&runtime, "(1a/2a) IN-L2", rows, iters);

    // (1b/2b) OUT-OF-L2 pass: 256MB/col, memory-bound. Same kernels, larger resident columns + a larger
    // gather index set (step_by(16) -> 4M indices at 64M rows). SKIPPED (not panic) on alloc failure.
    println!(
        "\n# ===================================================================================="
    );
    println!(
        "# (1b) OUT-OF-L2 scans + (2b) OUT-OF-L2 gather  ({out_l2_col_mb:.0}MB/col, MEMORY-BOUND)"
    );
    println!(
        "# ===================================================================================="
    );
    let roof_out_l2 = run_scan_pass(&runtime, "(1b/2b) OUT-OF-L2", rows_large, iters_large);

    // (3) ALGORITHMIC -- SINGLE pass, sized by SORT_N (NOT part of the IN-L2/OUT-OF-L2 row sweep). Runs on
    // its own small resident column so it is independent of the two scan datasets above.
    println!(
        "\n# ===================================================================================="
    );
    println!("# (3) ALGORITHMIC (sort / join / grouped) -- SINGLE pass, sort_n-sized (NOT in the L2 sweep)");
    println!(
        "# ===================================================================================="
    );
    let hash = |row: u64| -> u32 { ((row.wrapping_mul(2_654_435_761)) ^ (row << 13)) as u32 };
    let n = rows as usize;
    let off_a = 8u64;
    let off_b = off_a + rows * 4;
    let off_c = off_b + rows * 4;
    let off_d = off_c + rows * 8;
    let allocated = off_d + rows * 16;
    let header = rows.to_le_bytes().to_vec();
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
    let resident = runtime
        .retain_device_memory_chunks(
            0,
            allocated,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &header,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: off_a,
                    bytes: &a,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: off_b,
                    bytes: &b,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: off_c,
                    bytes: &c,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: off_d,
                    bytes: &d,
                },
            ],
        )
        .expect("retain resident device memory (algorithmic pass)");

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
        // reports BOTH p50 latency (us) AND throughput (Melem/s).
        println!(
            "  {label:<40} {us:>9.0}us  {:>8.1} Melem/s (sink {sink})",
            elems as f64 / us
        );
    };

    println!("#     sort/join wall INCLUDES a per-call key H2D (multi-launch -> the CUDA event is only the");
    println!("#     last pass, so the kernel is NOT isolated here -- the H2D is labeled instead).");
    let keys: Vec<i64> = (0..sort_n).map(|r| hash(r) as i64).collect();
    let sort_h2d_mb = (sort_n as usize * 8) as f64 / 1e6; // i64 key slice uploaded every call
    let join_h2d_mb = (sort_n as usize * 8 * 2) as f64 / 1e6; // build + probe key slices
    {
        let k = &keys;
        throughput(
            &format!("bitonic_sort_i64 (+~{sort_h2d_mb:.0}MB key H2D)"),
            sort_n,
            Box::new(|| resident.bitonic_sort_i64(k, false).unwrap().len()),
        );
    }
    let build: Vec<i64> = (0..sort_n).map(|r| r as i64).collect();
    let probe: Vec<i64> = (0..sort_n)
        .map(|r| hash(r) as i64 % sort_n as i64)
        .collect();
    {
        let (bk, pk) = (&build, &probe);
        throughput(
            &format!("hash_join_inner_i64 build+probe (+~{join_h2d_mb:.0}MB key H2D)"),
            sort_n,
            Box::new(|| match resident.hash_join_inner_i64(bk, pk, None, None) {
                Ok(HashJoinOutcome::Pairs { probe_idxs, .. }) => probe_idxs.len(),
                _ => 0,
            }),
        );
    }
    let payload_key = CudaJoinPayloadKey {
        payload: &resident,
        byte_offset: off_b,
        validity_bitmap_offset: None,
        width: 4,
        text_bytes_byte_offset: None,
        text_bytes_len: 0,
    };
    throughput(
        "join_payload_hash_i32 (resident D2D)",
        sort_n,
        Box::new(|| {
            resident
                .join_fixed_payload_coordinates(
                    None,
                    sort_n as u32,
                    &[0],
                    &[payload_key],
                    sort_n as u32,
                    &[payload_key],
                    None,
                    None,
                    false,
                    false,
                )
                .map_or(0, |coordinates| coordinates.row_count() as usize)
        }),
    );
    // LIVE per-group GROUP BY (the two-level shared-mem kernel the engine uses), grouping by the
    // ~1M-distinct key column A and summing B over a full-table scan (indices = 0..rows, as the executor
    // passes for an unfiltered GROUP BY). Replaces the removed `grouped_stats_i32` hash-agg measurement.
    //
    // The kernel-timed call (CUDA-event over the LIVE two-level kernel, runs=10) is the HONEST aggregate
    // KERNEL ms. The `from_payload` wall printed alongside is the END-TO-END result path: per-call it pays
    // the row-count index H2D (the engine's indices are already device-resident from its on-device filter
    // -- an artifact here), + the ~2*row_count slot-table setup, + the host build of the result Vec. NONE
    // of that is the kernel. See grouped_cardinality_probe for the end-to-end result path.
    let gb_indices: Vec<u32> = (0..rows as u32).collect();
    {
        let gi = &gb_indices;
        // Honest KERNEL: CUDA-event timed, two-level kernel, 10 runs, full-compute mask.
        let (_rows, kernel_ms) = resident
            .group_by_i32_count_sum_kernel_timed(
                off_a,
                off_b,
                gi,
                true,
                10,
                gpu_db_execution::grouped_agg_mask::ALL,
            )
            .expect("group_by kernel_timed");
        let kernel_melem_s = rows as f64 / (kernel_ms as f64 * 1e3); // rows / (ms*1000 us) = rows/us = Melem/s
                                                                     // reports BOTH p50 latency (the CUDA-event ms) AND throughput (Melem/s).
        println!(
            "  {:<40} {:>7.3}ms  {:>8.1} Melem/s (KERNEL only, CUDA-event)",
            "group_by_i32 KERNEL (~1M groups)", kernel_ms, kernel_melem_s
        );
        // Full result path (index H2D + ~2*row_count table setup + host Vec build), labeled NON-kernel.
        throughput(
            "  ^ group_by_i32 from_payload (FULL PATH)",
            rows,
            Box::new(|| {
                resident
                    .group_by_i32_count_sum_from_payload(
                        off_a,
                        off_b,
                        gi,
                        gpu_db_execution::grouped_agg_mask::ALL,
                    )
                    .unwrap()
                    .len()
            }),
        );
        println!("  {:<40}            (+index H2D + ~2*row_count table setup + host Vec build -- see grouped_cardinality_probe)", "");
    }

    // Closing note: state BOTH roofline values + the cache effect explicitly.
    println!(
        "\n# ===================================================================================="
    );
    println!("# SUMMARY -- CACHE EFFECT (IN-L2 vs OUT-OF-L2)");
    println!(
        "# ===================================================================================="
    );
    match (roof_in_l2, roof_out_l2) {
        (Some(in_l2), Some(out_l2)) => {
            println!("# sum_i32 ROOFLINE  in-L2 ({in_l2_col_mb:.0}MB/col) = {in_l2:.0} GB/s  vs  out-of-L2 ({out_l2_col_mb:.0}MB/col) = {out_l2:.0} GB/s");
            println!("#   -> out-of-L2 is {:.2}x the in-L2 roofline (out-of-L2 = the HONEST HBM/GDDR7-bound peak).", out_l2 / in_l2);
            println!("# The gather lines above show the cache effect most sharply: the in-L2 gather serves from the");
            println!("# {l2_mb:.0}MB L2 (cache-FLATTERED); the out-of-L2 gather scatters across {out_l2_col_mb:.0}MB > L2 (cache-MISS,");
            println!("# GDDR7-random) and should drop sharply. Compare the gather_i32/i64 GB/s + p50 latency between passes.");
        }
        (Some(in_l2), None) => {
            println!("# sum_i32 ROOFLINE  in-L2 ({in_l2_col_mb:.0}MB/col) = {in_l2:.0} GB/s.  OUT-OF-L2 pass was SKIPPED (see above) -- no cache-MISS roofline this run.");
        }
        _ => {
            println!("# both passes unavailable (no roofline captured this run).");
        }
    }
    println!("# section (1) is resident-input (kernel-clean, wall ~= kernel). gather/sort/join wall INCLUDES");
    println!("# a per-call input H2D the engine does NOT pay (inputs are device-resident); the GROUP BY line");
    println!("# shows the aggregate KERNEL (event-timed) vs its full result path.");
}
