//! P1-M3 step 4 — engine-level concurrent-read scaling A/B.
//!
//! Measures the gate-2 payoff directly at the engine boundary: the SAME `Arc<Engine>`
//! and the SAME resident query, executed two ways across a concurrency sweep —
//! `serialized` (every call holds a shared lock for its whole duration, the
//! owner-thread/serialized-access model M0 runs on) vs `concurrent` (every call runs
//! `execute_relational_select(&self)` with no lock, the new reader path).
//! The only difference is the lock, so the latency gap is exactly the queue-wait term
//! the P1-M3 substrate exists to remove. This is the controlled engine-level
//! measurement; wiring the same win through the pgwire server (re-running the
//! median-of-N harness) is the remaining step-4 server work and is reads-only over
//! frozen residency, the same safe scenario gate 2 proved.
//!
//! Noise control: each (concurrency, mode) cell runs N repetitions; the reported p50
//! and qps are the median across repetitions. Run on a GPU host so the resident route
//! is taken (otherwise it falls back to the CPU path, which still shows the lock vs
//! no-lock contrast).
//!
//! Env: GPU_DB_SCALE_CONCURRENCY (default 1,2,4,8,16,32,64), GPU_DB_SCALE_OPS_PER_THREAD
//! (default 200), GPU_DB_SCALE_REPS (default 5), GPU_DB_SCALE_ROWS (default 512),
//! GPU_DB_SCALE_RESIDENT (default 1; 0 = CPU path).

use std::env;
use std::error::Error;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Instant;

use gpu_db_engine::Engine;
use gpu_db_protocol::{parse_command, Command, Select};

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn median_u64(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    percentile(values, 0.5)
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if values.is_empty() {
        0.0
    } else {
        values[values.len() / 2]
    }
}

/// One measured run of a (concurrency, mode) cell. Returns (qps, p50_us, p95_us, p99_us).
fn run_cell(
    engine: &Arc<Engine>,
    select: &Select,
    threads: usize,
    ops_per_thread: usize,
    serialize: bool,
    serial_lock: &Arc<Mutex<()>>,
) -> (f64, u64, u64, u64) {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let engine = Arc::clone(engine);
            let select = select.clone();
            let barrier = Arc::clone(&barrier);
            let serial_lock = Arc::clone(serial_lock);
            thread::spawn(move || {
                let mut latencies = Vec::with_capacity(ops_per_thread);
                barrier.wait(); // release together so the read bodies actually overlap
                for _ in 0..ops_per_thread {
                    let started = Instant::now();
                    if serialize {
                        let _guard = serial_lock.lock().unwrap();
                        engine.execute_relational_select(&select).unwrap();
                    } else {
                        engine.execute_relational_select(&select).unwrap();
                    }
                    latencies.push(started.elapsed().as_micros() as u64);
                }
                latencies
            })
        })
        .collect();

    barrier.wait();
    let wall_started = Instant::now();
    let mut all_latencies = Vec::with_capacity(threads * ops_per_thread);
    for handle in handles {
        all_latencies.extend(handle.join().expect("reader thread panicked"));
    }
    let wall = wall_started.elapsed();

    let total_ops = (threads * ops_per_thread) as f64;
    let qps = if wall.as_secs_f64() > 0.0 {
        total_ops / wall.as_secs_f64()
    } else {
        0.0
    };
    all_latencies.sort_unstable();
    (
        qps,
        percentile(&all_latencies, 0.50),
        percentile(&all_latencies, 0.95),
        percentile(&all_latencies, 0.99),
    )
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let targets: Vec<usize> = env::var("GPU_DB_SCALE_CONCURRENCY")
        .unwrap_or_else(|_| "1,2,4,8,16,32,64".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let ops_per_thread = env_usize("GPU_DB_SCALE_OPS_PER_THREAD", 200);
    let reps = env_usize("GPU_DB_SCALE_REPS", 5);
    let rows = env_usize("GPU_DB_SCALE_ROWS", 512);

    // Build a small int4 table and make it GPU-resident (one published generation).
    let mut engine = Engine::new_local();
    engine.execute_text(1, "CREATE TABLE order_line (ol_o_id INT)")?;
    let mut insert = String::from("INSERT INTO order_line (ol_o_id) VALUES ");
    for id in 0..rows {
        if id > 0 {
            insert.push(',');
        }
        insert.push('(');
        insert.push_str(&id.to_string());
        insert.push(')');
    }
    engine.execute_text(2, &insert)?;
    // GPU_DB_SCALE_RESIDENT=1 (default) populates GPU residency so reads take the
    // resident kernel route; =0 leaves it non-resident so reads take the CPU path. The
    // CPU path isolates the &self concurrency win from the GPU-side bottleneck.
    let make_resident = env_usize("GPU_DB_SCALE_RESIDENT", 1) != 0;
    let resident_on_gpu = if make_resident {
        let snapshot = engine.populate_relational_residency_snapshot("order_line")?;
        snapshot.device_memory_proof.is_some()
    } else {
        false
    };

    let engine = Arc::new(engine);
    let serial_lock = Arc::new(Mutex::new(()));
    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM order_line")? else {
        return Err("expected SELECT".into());
    };
    // sanity + warmup
    let warm = engine.execute_relational_select(&select)?;
    assert_eq!(
        warm.rows,
        vec![vec![gpu_db_protocol::SqlValue::Int4(rows as i32)]]
    );

    println!(
        "p1_m3_concurrent_read_scaling: rows={rows} ops_per_thread={ops_per_thread} reps={reps} resident_on_gpu={resident_on_gpu}"
    );
    println!("query=SELECT COUNT(*) FROM order_line  (route: resident count_all when GPU present)");
    println!();
    println!(
        "| conc | serial p50 us | serial p99 us | serial qps | conc p50 us | conc p99 us | conc qps | p50 speedup | p99 speedup | qps speedup |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    let mut json_rows: Vec<String> = Vec::new();
    for &c in &targets {
        let mut serial_p50s = Vec::new();
        let mut serial_p99s = Vec::new();
        let mut serial_qpss = Vec::new();
        let mut conc_p50s = Vec::new();
        let mut conc_p99s = Vec::new();
        let mut conc_qpss = Vec::new();
        for _ in 0..reps {
            let (sq, sp50, _sp95, sp99) =
                run_cell(&engine, &select, c, ops_per_thread, true, &serial_lock);
            let (cq, cp50, _cp95, cp99) =
                run_cell(&engine, &select, c, ops_per_thread, false, &serial_lock);
            serial_p50s.push(sp50);
            serial_p99s.push(sp99);
            serial_qpss.push(sq);
            conc_p50s.push(cp50);
            conc_p99s.push(cp99);
            conc_qpss.push(cq);
        }
        let sp50 = median_u64(&mut serial_p50s);
        let sp99 = median_u64(&mut serial_p99s);
        let sq = median_f64(&mut serial_qpss);
        let cp50 = median_u64(&mut conc_p50s);
        let cp99 = median_u64(&mut conc_p99s);
        let cq = median_f64(&mut conc_qpss);
        let p50_speedup = if cp50 > 0 {
            sp50 as f64 / cp50 as f64
        } else {
            0.0
        };
        let p99_speedup = if cp99 > 0 {
            sp99 as f64 / cp99 as f64
        } else {
            0.0
        };
        let qps_speedup = if sq > 0.0 { cq / sq } else { 0.0 };
        println!(
            "| {c} | {sp50} | {sp99} | {sq:.0} | {cp50} | {cp99} | {cq:.0} | {p50_speedup:.2}x | {p99_speedup:.2}x | {qps_speedup:.2}x |"
        );
        json_rows.push(format!(
            "{{\"concurrency\":{c},\"serial_p50_us\":{sp50},\"serial_p99_us\":{sp99},\"serial_qps\":{sq:.3},\"concurrent_p50_us\":{cp50},\"concurrent_p99_us\":{cp99},\"concurrent_qps\":{cq:.3},\"p50_speedup\":{p50_speedup:.3},\"p99_speedup\":{p99_speedup:.3},\"qps_speedup\":{qps_speedup:.3}}}"
        ));
    }
    println!();
    println!("json={{\"kind\":\"p1_m3_concurrent_read_scaling\",\"rows\":{rows},\"ops_per_thread\":{ops_per_thread},\"reps\":{reps},\"resident_on_gpu\":{resident_on_gpu},\"cells\":[{}]}}", json_rows.join(","));
    Ok(())
}
