//! R2.2c gate-1 probe — can K persistent wave kernels COEXIST on one shared CUDA context?
//!
//! R2.2b-2 enforced an AT-MOST-ONE-resident-wave-kernel invariant because two FULL-OCCUPANCY persistent
//! spin-kernels mutually starve (neither yields its SMs) -> tearing one down (cuStreamSynchronize, infinite
//! backstop) hangs. R2.2b-3's A/B showed the wave wins single-flight but the default flip is GATED on
//! lifting that invariant so multiple point-read SHAPES can each own a coexisting engine (no thrash). The
//! design rule (DECISIONS ADR-008 SM-coexistence gate) is "~1-SM minimal sidecar OR full replacement" — so
//! the hypothesis is that MINIMAL-grid kernels (1 block = 1 SM) leave SMs free and coexist, while FAT
//! (full-occupancy) kernels do not.
//!
//! There are TWO independent unknowns this probe MEASURES (not assumes), for minimal vs fat:
//!   (1) BUILD-while-running: `WaveReadEngine::new` calls `cuMemAlloc` (counters + device result ring), and
//!       (DECISIONS ADR-008 "R2.2 freeze ROOT CAUSE") `cuMemAlloc` DEVICE-SYNCHRONIZES — it blocks until all
//!       GPU work drains, INCLUDING a never-ending wave kernel. So building a 2nd engine while the 1st runs
//!       may block ~backstop (killing the 1st). If so, K-coexistence needs async allocation (cuMemAllocAsync)
//!       or a device-buffer pool BEFORE it is even buildable — independent of SM starvation.
//!   (2) TEARDOWN-while-running: with both engines built + resident, can one be torn down (doorbell +
//!       cuStreamSynchronize) while the other keeps spinning, WITHOUT the survivor starving the torn-down
//!       kernel's blocks (-> sync hangs to backstop)?
//!
//! We use a FINITE backstop so any blocked/starved kernel self-terminates (no zombie); we TIME each op:
//! FAST (<< backstop) = clean; ~backstop = device-sync/starvation. The minimal-vs-fat contrast is the
//! evidence for (or against) lifting the at-most-one invariant with minimal-grid sizing.
//!
//! Run (RTX box; never `--gpu-reset`, always under `timeout`):
//!   timeout 180 cargo run --release --example wave_multikernel_probe -p gpu_db_execution

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use gpu_db_execution::{CudaDriverRuntime, WaveReadEngine};

const BACKSTOP_NS: u64 = 4_000_000_000; // 4s — bounds any block/starvation so the probe can't hang forever
const FAST_MS: u128 = 800; // an op faster than this is "did not device-sync / did not starve to backstop"

/// Build a `WaveReadEngine` over a synthetic `rows`-row int4 table (keys 3r+1, payload 1000r+7) and its
/// R1-format hash index, with a `threads`-wide grid. ring=1024, watchdog=0 (explicit shutdown in this probe).
fn build_engine(
    runtime: &CudaDriverRuntime,
    rows: u64,
    threads: u32,
) -> Result<(WaveReadEngine, Vec<i32>, Vec<i32>), Box<dyn Error>> {
    let keys: Vec<i32> = (0..rows as i32).map(|r| r * 3 + 1).collect();
    let payload: Vec<i32> = (0..rows as i32).map(|r| r * 1000 + 7).collect();
    let mut buf: Vec<u8> = Vec::with_capacity(rows as usize * 8);
    for &k in &keys {
        buf.extend_from_slice(&k.to_le_bytes());
    }
    for &v in &payload {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let resident = Arc::new(runtime.retain_device_memory_copy(0, &buf)?);
    let projections = [0_u64, rows * 4];
    let table_size = ((rows * 2) as u32).next_power_of_two();
    let table_mask = table_size - 1;
    let hash_shift = 32 - table_size.trailing_zeros();
    let mut index = vec![0_u64; table_size as usize];
    for (r, &k) in keys.iter().enumerate() {
        let key = k as u32;
        let mut h = (key.wrapping_mul(0x9E37_79B1) >> hash_shift) & table_mask;
        while index[h as usize] != 0 {
            h = (h + 1) & table_mask;
        }
        index[h as usize] = ((key as u64) << 32) | (r as u64 + 1);
    }
    let index_bytes: Vec<u8> = index.iter().flat_map(|e| e.to_le_bytes()).collect();
    let index_resident = Arc::new(runtime.retain_device_memory_copy(0, &index_bytes)?);
    let engine = WaveReadEngine::new(
        index_resident,
        resident,
        &projections,
        table_mask,
        hash_shift,
        1024,
        threads,
        BACKSTOP_NS,
        0,
    )?;
    Ok((engine, keys, payload))
}

/// Submit `needles` and check every found row gathers (key, payload) correctly. Returns false on a wrong row
/// OR on a submit error (a dead/backstop-killed kernel times out -> Err). CAUTION: a dead engine costs the
/// full DRAIN_TIMEOUT (~20s) here, so only verify engines expected to be alive.
fn verify(engine: &mut WaveReadEngine, keys: &[i32], payload: &[i32], needles: &[i32]) -> bool {
    match engine.submit(needles) {
        Ok(mut rows) => {
            rows.sort_by_key(|row| (row.needle_index, row.row_index));
            rows.len() == needles.len()
                && rows.iter().all(|row| {
                    let r = row.row_index as usize;
                    r < keys.len() && row.values == vec![keys[r], payload[r]]
                })
        }
        Err(_) => false,
    }
}

fn scenario(runtime: &CudaDriverRuntime, label: &str, threads: u32) -> Result<(), Box<dyn Error>> {
    let rows: u64 = 256;
    let needles = vec![1_i32, 3 * 50 + 1, 3 * 200 + 1]; // present keys (rows r=0,50,200)
    println!("\n## scenario: {label}  (threads={threads}, backstop={}s)", BACKSTOP_NS / 1_000_000_000);

    // e0: build + prove alive.
    let (mut e0, k0, p0) = build_engine(runtime, rows, threads)?;
    let e0_ok = verify(&mut e0, &k0, &p0, &needles);
    println!("  e0 built + verified alive: {e0_ok}");

    // (1) BUILD-while-running: time building e1 WHILE e0's kernel spins. Fast => allocs didn't device-sync.
    let t = Instant::now();
    let built = build_engine(runtime, rows, threads);
    let build_e1_ms = t.elapsed().as_millis();
    let build_fast = build_e1_ms < FAST_MS;
    println!(
        "  (1) build e1 while e0 RUNS: {build_e1_ms} ms -> {}",
        if build_fast {
            "FAST: cuMemAlloc did NOT device-sync (coexistence buildable)"
        } else {
            "SLOW ~backstop: cuMemAlloc DEVICE-SYNCED against e0's kernel (e0 killed; needs async alloc)"
        }
    );

    let mut e1 = match built {
        Ok((e1, _k1, _p1)) => e1,
        Err(e) => {
            println!("  build e1 FAILED: {e}");
            e0.shutdown();
            return Ok(());
        }
    };

    // (2) TEARDOWN-while-running: time tearing down e0 while e1 spins. Only meaningful if the build was fast
    // (otherwise e0 was already backstop-killed during the build). Fast teardown => clean coexistence.
    if build_fast {
        let e1_ok = verify(&mut e1, &k0, &p0, &needles);
        println!("  after build, e1 verified alive: {e1_ok}");
        let t = Instant::now();
        e0.shutdown();
        let teardown_ms = t.elapsed().as_millis();
        println!(
            "  (2) teardown e0 while e1 RUNS: {teardown_ms} ms -> {}",
            if teardown_ms < FAST_MS {
                "FAST: clean coexistence (no SM starvation)"
            } else {
                "SLOW ~backstop: SM STARVATION deadlock (e0 blocks never scheduled; killed by backstop)"
            }
        );
        let survivor = verify(&mut e1, &k0, &p0, &needles);
        println!("  survivor e1 still serves after e0 teardown: {survivor}");
    } else {
        println!("  (2) teardown test SKIPPED (e0 already backstop-killed by the build device-sync)");
    }

    e0.shutdown();
    e1.shutdown();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let runtime = match CudaDriverRuntime::probe() {
        Ok(r) => r,
        Err(e) => {
            println!("no GPU ({e}) — skipping");
            return Ok(());
        }
    };
    println!("# R2.2c gate-1 probe — can K persistent wave kernels coexist on one context?");
    println!("# measures (1) build-while-running (cuMemAlloc device-sync) and (2) teardown-while-running (SM starvation)");

    // Minimal (1 block = 1 SM): the hypothesized coexistence-friendly sizing.
    scenario(&runtime, "MINIMAL 1-SM (threads=256, 1 block)", 256)?;
    // Fat (full-occupancy, clamped): the current WAVE_ENGINE_THREADS default — expected to NOT coexist.
    scenario(&runtime, "FAT full-occupancy (threads=8192)", 8192)?;

    println!("\n# Read: scenario MINIMAL both FAST => lift the at-most-one invariant with minimal-grid sizing.");
    println!("# If MINIMAL (1) is SLOW => coexistence needs cuMemAllocAsync/pool first (alloc device-sync, not SM starvation).");
    Ok(())
}
