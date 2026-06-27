//! OLTP auto-admit A/B micro-benchmark — the STRATA S-F gate.
//!
//! Measures the **per-operation cost vs benefit** of `auto_admit_on_commit` (STRATA S-B), to decide
//! whether flipping it ON by default (S-F) is net-positive for an OLTP workload:
//!   - **read benefit:** point-read service latency, host (non-resident) vs GPU (resident).
//!   - **write cost:**  single-row commit service latency, auto-admit OFF vs ON. With ON, every commit
//!                      re-admits (re-uploads the whole table), so this is the price S-F would charge writes.
//!
//! Closed-loop, single-threaded → pure service latency, no queueing/contention (the open-loop
//! offered-rate harness + a tuned-Postgres baseline are the larger PLAN §1 follow-on). Reports
//! p50/p99/p99.9 and the break-even read:write ratio at which ON pays for itself.
//!
//! Env: `GPU_DB_BENCH_ROWS` (table size, default 20000), `GPU_DB_BENCH_READ_OPS` (default 5000),
//! `GPU_DB_BENCH_WRITE_OPS` (default 500; each ON-write re-uploads the whole table, so keep modest).

use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

use gpu_db_engine::Engine;
use gpu_db_sql::{parse_command, Command, Select};

fn parse_select(sql: &str) -> Select {
    match parse_command(sql).expect("parse") {
        Command::Select(sel) => sel,
        _ => panic!("not a SELECT: {sql}"),
    }
}

struct Lat {
    p50: u128,
    p99: u128,
    p999: u128,
    max: u128,
    ops_per_s: f64,
}

fn summarize(mut micros: Vec<u128>, wall: Duration) -> Lat {
    micros.sort_unstable();
    let n = micros.len();
    let pct = |p: f64| -> u128 {
        if n == 0 {
            return 0;
        }
        let rank = ((n as f64 * p).ceil() as usize).clamp(1, n);
        micros[rank - 1]
    };
    Lat {
        p50: pct(0.50),
        p99: pct(0.99),
        p999: pct(0.999),
        max: *micros.last().unwrap_or(&0),
        ops_per_s: if wall.is_zero() {
            0.0
        } else {
            n as f64 / wall.as_secs_f64()
        },
    }
}

fn print_lat(label: &str, l: &Lat) {
    println!(
        "  {label:<22} p50={:>8}us  p99={:>9}us  p99.9={:>9}us  max={:>9}us  {:>10.1} ops/s",
        l.p50, l.p99, l.p999, l.max, l.ops_per_s
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows: i64 = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let read_ops: usize = env::var("GPU_DB_BENCH_READ_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000);
    let write_ops: usize = env::var("GPU_DB_BENCH_WRITE_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    println!("# OLTP auto-admit A/B  rows={rows}  read_ops={read_ops}  write_ops={write_ops}");

    // --- setup: load `rows` accounts with auto-admit OFF (default) ---
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;
    let mut txn = 2u64;
    let mut id = 0i64;
    while id < rows {
        let mut vals = String::new();
        for _ in 0..1000 {
            if id >= rows {
                break;
            }
            if !vals.is_empty() {
                vals.push(',');
            }
            vals.push_str(&format!("({}, {})", id, (id * 7) % 100_000));
            id += 1;
        }
        e.execute_text(
            txn,
            &format!("INSERT INTO accounts (id, balance) VALUES {vals}"),
        )?;
        txn += 1;
    }
    println!("# loaded {rows} rows");

    // deterministic xorshift id generator (no rand dependency)
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next_id = move || -> i64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % rows as u64) as i64
    };

    // ---------- READ A/B (host non-resident vs GPU resident) ----------
    // host: not warmed, auto-admit off -> non-resident -> host path.
    let host_target = format!(
        "{:?}",
        e.execute_relational_select(&parse_select("SELECT balance FROM accounts WHERE id = 0"))?
            .executed_target
    );
    let mut lat = Vec::with_capacity(read_ops);
    let t = Instant::now();
    for _ in 0..read_ops {
        let q = parse_select(&format!(
            "SELECT balance FROM accounts WHERE id = {}",
            next_id()
        ));
        let s = Instant::now();
        let _ = e.execute_relational_select(&q)?;
        lat.push(s.elapsed().as_micros());
    }
    let host_read = summarize(lat, t.elapsed());

    // GPU: warm the table resident, then read.
    e.populate_relational_residency_snapshot("accounts")?;
    let gpu_target = format!(
        "{:?}",
        e.execute_relational_select(&parse_select("SELECT balance FROM accounts WHERE id = 0"))?
            .executed_target
    );
    let mut lat = Vec::with_capacity(read_ops);
    let t = Instant::now();
    for _ in 0..read_ops {
        let q = parse_select(&format!(
            "SELECT balance FROM accounts WHERE id = {}",
            next_id()
        ));
        let s = Instant::now();
        let _ = e.execute_relational_select(&q)?;
        lat.push(s.elapsed().as_micros());
    }
    let gpu_read = summarize(lat, t.elapsed());

    // ---------- WRITE A/B (auto-admit OFF vs ON; production concurrent path) ----------
    // OFF: writes invalidate residency and stay non-resident -> pure commit cost.
    e.set_auto_admit_on_commit(false);
    let mut lat = Vec::with_capacity(write_ops);
    let t = Instant::now();
    for _ in 0..write_ops {
        let sql = format!(
            "UPDATE accounts SET balance = {} WHERE id = {}",
            next_id(),
            next_id()
        );
        let s = Instant::now();
        e.execute_dml_concurrent(txn, &sql)?;
        lat.push(s.elapsed().as_micros());
        txn += 1;
    }
    let write_off = summarize(lat, t.elapsed());

    // ON: every commit re-admits (re-uploads the whole table).
    e.set_auto_admit_on_commit(true);
    let mut lat = Vec::with_capacity(write_ops);
    let t = Instant::now();
    for _ in 0..write_ops {
        let sql = format!(
            "UPDATE accounts SET balance = {} WHERE id = {}",
            next_id(),
            next_id()
        );
        let s = Instant::now();
        e.execute_dml_concurrent(txn, &sql)?;
        lat.push(s.elapsed().as_micros());
        txn += 1;
    }
    let write_on = summarize(lat, t.elapsed());

    // ---------- report ----------
    println!("\n## reads  (host executed_target={host_target}, gpu executed_target={gpu_target})");
    print_lat("host (non-resident)", &host_read);
    print_lat("gpu (resident)", &gpu_read);
    println!("\n## writes (single-row UPDATE via the concurrent DML path)");
    print_lat("auto-admit OFF", &write_off);
    print_lat("auto-admit ON", &write_on);

    let read_benefit_us = host_read.p50 as i128 - gpu_read.p50 as i128; // us saved per read on GPU
    let write_cost_us = write_on.p50 as i128 - write_off.p50 as i128; // extra us per write under ON
    println!("\n## verdict (p50)");
    println!("  read benefit (host - gpu): {read_benefit_us} us/read");
    println!("  write cost  (on  - off):  {write_cost_us} us/write");
    if read_benefit_us > 0 && write_cost_us > 0 {
        let break_even = write_cost_us as f64 / read_benefit_us as f64;
        println!(
            "  break-even read:write ratio = {break_even:.0}:1  (ON pays off only above ~{break_even:.0} reads per write)"
        );
    } else if read_benefit_us <= 0 {
        println!(
            "  GPU reads are NOT faster at this size — auto-admit ON has no read upside here."
        );
    } else {
        println!("  auto-admit ON adds no write cost at this size (table tiny / GPU absent).");
    }
    Ok(())
}
