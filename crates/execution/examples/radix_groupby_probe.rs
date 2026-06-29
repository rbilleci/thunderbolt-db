//! RADIX-PARTITIONED GROUP-BY POTENTIAL PROBE (measurement spike, no production change).
//!
//! The per-group two-level shared-mem kernel (`gpu_db_group_by_i32_count_sum_twolevel`) is fast at
//! low/mid cardinality but FALLS OFF A CLIFF at high cardinality: at ~1M distinct groups over 8M rows
//! its shared-mem table overflows and spills to GLOBAL atomics on a >16MB table = random-access /
//! cache-miss bound (~182 Melem/s measured by `grouped_cardinality_probe`).
//!
//! Proposed fix #2 = RADIX-PARTITIONED AGGREGATION: partition rows by group-key hash into cache-sized
//! buckets, then aggregate each bucket LOCALLY (sequential, not random global access). A clean upper
//! bound on that idea is the SORT-then-SEGMENTED-REDUCE pipeline (a global radix sort IS a maximal
//! radix partition; a sorted run is the most cache-friendly per-bucket layout possible). If sort+reduce
//! does not beat two-level at high card, no cheaper partition scheme will either.
//!
//! This probe measures, at 8M rows, the COST of the radix-agg pipeline vs the two-level kernel across
//! cardinality {4096, 65536, 1<<18, 1<<20} so we can find the crossover and decide whether #2 is worth
//! building. The radix pipeline = three stages:
//!   (1) SORT the 8M i64 group keys -> perm  (the proven LSD-radix argsort via `order_by_sort_i64`;
//!       also `bitonic_sort_i64` as a second data point). This is the dominant, CARD-INDEPENDENT term.
//!   (2) GATHER the value column by the perm (`project_i32_rows_from_payload`, a GPU gather kernel).
//!       NOTE: this path round-trips the perm host->device, so it OVER-counts vs an integrated pipeline
//!       that keeps the perm on-device -- a CONSERVATIVE (upper-bound) gather cost.
//!   (3) SEGMENTED REDUCE over the sorted keys+values -> one row per group. PROXIED here by a coalesced
//!       1-pass streaming scan (`scalar_stats_i32_from_payload`: count+sum+min+max), which is the SAME
//!       memory traffic a real reduce-by-key streams, plus cheap register boundary compares + the
//!       per-group emit. We label it a PROXY and bound the estimate. Correctness of the partition is
//!       checked by asserting the host-side distinct-group count equals the sorted-key run count.
//!
//! Run (never --gpu-reset, always under timeout):
//!   timeout 290 cargo run --release --example radix_groupby_probe -p gpu_db_execution

use std::time::Instant;

use gpu_db_execution::{grouped_agg_mask, CudaDeviceMemoryChunk, CudaDriverRuntime};

fn p50(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

// time a closure `iters` times after a 3x warmup; return p50 wall microseconds + last sink.
fn timed_us(iters: usize, mut run: impl FnMut() -> usize) -> (f64, usize) {
    for _ in 0..3 {
        run();
    }
    let mut s = Vec::with_capacity(iters);
    let mut sink = 0usize;
    for _ in 0..iters {
        let t = Instant::now();
        sink = run();
        s.push(t.elapsed().as_nanos());
    }
    (p50(s) as f64 / 1000.0, sink)
}

fn main() {
    let rows: u64 = std::env::var("ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8_388_608);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    // Two-level CUDA-event runs (kernel-only timing averages min of `runs` internally).
    let kruns: u32 = std::env::var("KRUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let n = rows as usize;
    let cards: Vec<u64> = vec![4_096, 65_536, 1 << 18, 1 << 20];

    let Ok(runtime) = CudaDriverRuntime::probe() else {
        eprintln!("no local NVIDIA driver/GPU; skipping");
        return;
    };

    // Scramble so keys are high-entropy (not pre-sorted): the hashed key still has exactly C distinct
    // values (key = hash(row) % C), so the sort/group is a real high-distinct workload, not a trivially
    // pre-ordered one. The same hash feeds the i32 group col and the i64 sort-key col per cardinality.
    let hash = |row: u64| -> u64 { (row.wrapping_mul(2_654_435_761)) ^ (row << 13) ^ (row >> 7) };

    // Layout: [8B row_count][value: N i32][ per cardinality: i32 group col (N*4) ][ per cardinality: i64
    // sort-key col (N*8) ]. i32 group cols feed the two-level/hash kernels; i64 cols feed the radix sort.
    let off_value = 8u64;
    let g32 = |k: usize| off_value + rows * 4 + (k as u64) * rows * 4; // i32 group col k
    let g32_block = rows * 4 * cards.len() as u64;
    let g64 = |k: usize| off_value + rows * 4 + g32_block + (k as u64) * rows * 8; // i64 sort-key col k
    let allocated = off_value + rows * 4 + g32_block + rows * 8 * cards.len() as u64;

    let header = rows.to_le_bytes().to_vec();
    let mut value = Vec::with_capacity(n * 4);
    for row in 0..rows {
        value.extend_from_slice(&((row as i32).wrapping_mul(7)).to_le_bytes());
    }
    // host-side i64 keys per cardinality (also used to drive the radix sort, which takes a host &[i64]).
    let keys_i64: Vec<Vec<i64>> = cards
        .iter()
        .map(|&c| (0..rows).map(|row| (hash(row) % c) as i64).collect())
        .collect();
    let g32_cols: Vec<Vec<u8>> = keys_i64
        .iter()
        .map(|ks| {
            let mut g = Vec::with_capacity(n * 4);
            for &k in ks {
                g.extend_from_slice(&(k as i32).to_le_bytes());
            }
            g
        })
        .collect();
    let g64_cols: Vec<Vec<u8>> = keys_i64
        .iter()
        .map(|ks| {
            let mut g = Vec::with_capacity(n * 8);
            for &k in ks {
                g.extend_from_slice(&k.to_le_bytes());
            }
            g
        })
        .collect();

    let mut chunks = vec![
        CudaDeviceMemoryChunk { byte_offset: 0, bytes: &header },
        CudaDeviceMemoryChunk { byte_offset: off_value, bytes: &value },
    ];
    for (k, g) in g32_cols.iter().enumerate() {
        chunks.push(CudaDeviceMemoryChunk { byte_offset: g32(k), bytes: g });
    }
    for (k, g) in g64_cols.iter().enumerate() {
        chunks.push(CudaDeviceMemoryChunk { byte_offset: g64(k), bytes: g });
    }
    let resident = runtime
        .retain_device_memory_chunks(0, allocated, &chunks)
        .expect("retain resident device memory");

    // distinct-group ground truth per cardinality (host), used to assert the sort really grouped keys.
    let host_groups: Vec<usize> = keys_i64
        .iter()
        .map(|ks| {
            let mut s: Vec<i64> = ks.clone();
            s.sort_unstable();
            s.dedup();
            s.len()
        })
        .collect();

    // full-table scan: indices = 0..rows (no WHERE), as the executor passes for an unfiltered GROUP BY.
    let indices: Vec<u32> = (0..rows as u32).collect();

    println!("# radix-partitioned GROUP BY potential probe. rows={rows}, iters={iters}, kruns={kruns}");
    println!("# 8M i32 value column; group key = hash(row) %% C (high-entropy, exactly C distinct).");
    println!("# TWO-LEVEL = gpu_db_group_by_i32_count_sum_twolevel (CUDA-event kernel-only ms). NOTE: this");
    println!("#   path sizes its GLOBAL hash table to row_count*2 (=16M slots) -> it does NOT exhibit a");
    println!("#   shared-mem-overflow cliff; it is already a row-count-sized global hash.");
    println!("# HASH-AGG  = gpu_db_resident_i32_grouped_hash_* GLOBAL open-addressing hash (the cliff path;");
    println!("#   produced the 182 number on the older box). Also a 16M-slot global table; slows at high");
    println!("#   distinct from CAS-collision + cache-miss random access = exactly radix's target.");
    println!("# RADIX PIPELINE = radix-argsort(8M i64 keys) + gather(value by perm) + seg-reduce[PROXY].");
    println!("#   seg-reduce PROXY = scalar_stats 1-pass streaming scan (count+sum+min+max) over 8M i32.");
    println!("#   gather here ROUND-TRIPS the perm host->device + values device->host (~96MB PCIe) so it");
    println!("#   OVER-counts heavily; an INTEGRATED pipeline keeps perm+values on-device (gather feeds the");
    println!("#   reduce). 'radix ms' = measured (upper bound); 'integ ms' = sort + 2x reduce-scan (a fused");
    println!("#   on-device gather+reduce lower bound).\n");

    println!(
        "  {:<10} {:>8} | {:>9} {:>9} | {:>9} {:>9} {:>9} {:>9} | {:>9} | {:>9} {:>9}",
        "card", "groups", "2lvl ms", "hash ms", "radix ms", "  sort", "gather", "reduce", "integ ms", "2lvl/integ", "hash/integ",
    );

    for (k, &c) in cards.iter().enumerate() {
        let goff32 = g32(k);
        let ks = &keys_i64[k];

        // ---- TWO-LEVEL kernel (CUDA-event, kernel-only) ----
        let (twolevel_rows, twolevel_ms) = resident
            .group_by_i32_count_sum_kernel_timed(goff32, off_value, &indices, true, kruns, grouped_agg_mask::ALL)
            .expect("two-level kernel");
        assert_eq!(twolevel_rows.len(), host_groups[k], "two-level group count mismatch @card {c}");

        // ---- HASH-AGG path (wall p50; the 182-number path) ----
        let (hash_us, hg) = timed_us(iters, || {
            resident.grouped_stats_i32_from_payload(goff32, off_value, rows, grouped_agg_mask::ALL).unwrap().len()
        });
        assert_eq!(hg, host_groups[k], "hash-agg group count mismatch @card {c}");

        // ---- RADIX PIPELINE ----
        // (1) sort: the proven LSD-radix argsort (order_by_sort_i64 dispatches RADIX at n >= 10k).
        let (sort_us, _) = timed_us(iters, || resident.order_by_sort_i64(ks, false).unwrap().len());
        let perm = resident.order_by_sort_i64(ks, false).unwrap();
        assert_eq!(perm.len(), n, "perm length");
        // assert the sort actually grouped the keys: count runs in the sorted-key sequence == #groups.
        let mut runs_in_sorted = 0usize;
        let mut prev: Option<i64> = None;
        for &p in &perm {
            let key = ks[p as usize];
            if prev != Some(key) {
                runs_in_sorted += 1;
                prev = Some(key);
            }
        }
        assert_eq!(runs_in_sorted, host_groups[k], "sorted-key run count != #groups @card {c}");

        // (2) gather value column by perm (GPU gather; perm uploaded host->device = conservative).
        let perm_u64: Vec<u64> = perm.iter().map(|&p| p as u64).collect();
        let (gather_us, _) = timed_us(iters, || resident.project_i32_rows_from_payload(off_value, &perm_u64).unwrap().len());

        // (3) segmented-reduce PROXY: a coalesced 1-pass count+sum+min+max streaming scan over the value
        // column. Same memory traffic a real reduce-by-key streams; the per-group boundary compares +
        // emit are register-cheap by comparison, so this LOWER-BOUNDs reduce cost (radix total = upper
        // bound on sort+gather + lower bound on reduce; reduce is the small term so the total is tight).
        let (reduce_us, _) = timed_us(iters, || {
            let (cnt, _s, _mn, _mx) = resident.scalar_stats_i32_from_payload(off_value, rows).unwrap();
            cnt as usize
        });

        let radix_us = sort_us + gather_us + reduce_us;
        // Integrated on-device estimate: sort + a fused gather+reduce. The fused gather+reduce reads the
        // value column once via the perm (random-ish, ~1 gather pass) and reduces in registers; bound it
        // by 2x a coalesced reduce-scan (one for the gather-read, one for the reduce-read) -- generous,
        // since neither host round-trip is paid. Sort dominates this estimate.
        let integ_us = sort_us + 2.0 * reduce_us;
        println!(
            "  {c:<10} {:>8} | {:>9.3} {:>9.3} | {:>9.3} {:>9.3} {:>9.3} {:>9.3} | {:>9.3} | {:>8.2}x {:>8.2}x",
            host_groups[k],
            twolevel_ms,
            hash_us / 1000.0,
            radix_us / 1000.0,
            sort_us / 1000.0,
            gather_us / 1000.0,
            reduce_us / 1000.0,
            integ_us / 1000.0,
            (twolevel_ms as f64) / (integ_us / 1000.0),
            (hash_us / 1000.0) / (integ_us / 1000.0),
        );
    }

    // Second sort data point: bitonic over the highest-card key set (a non-radix sort baseline).
    let top = cards.len() - 1;
    let (bit_us, _) = timed_us(iters.min(5), || resident.bitonic_sort_i64(&keys_i64[top], false).unwrap().len());
    let (rdx_us, _) = timed_us(iters, || resident.order_by_sort_i64(&keys_i64[top], false).unwrap().len());
    println!(
        "\n# 8M-key sort: radix {:.3} ms ({:.1} Melem/s)  vs  bitonic {:.3} ms ({:.1} Melem/s).",
        rdx_us / 1000.0,
        rows as f64 / rdx_us,
        bit_us / 1000.0,
        rows as f64 / bit_us,
    );
    println!("# integ ms = sort + 2x reduce-scan (fused on-device gather+reduce lower bound). The radix");
    println!("#   pipeline is SORT-DOMINATED: sort is ~95%+ of integ, gather+reduce are sub-ms on-device.");
    println!("# 2lvl/integ and hash/integ > 1 => radix beats that kernel at that cardinality (vs the");
    println!("#   integrated estimate; the measured 'radix ms' is inflated by host PCIe round-trips).");
}
