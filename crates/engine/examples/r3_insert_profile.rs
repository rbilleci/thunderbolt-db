//! R3 write-path Step 1 — SQL-honest INSERT profiler.
//!
//! The read path is settled (lpb 121.6M lookups/s @b65536). R3 (the write half) is unmeasured: the
//! report card notes a SQL INSERT load runs at ~111K rows/s (~9.4s/1M, "CPU-bound parse+txn") — but that
//! is a GUESS at the attribution. This benchmark turns the guess into a number, with NO optimization yet
//! (charter discipline: measure before optimizing; the read-path lesson was that the bottleneck was NOT
//! where intuition said — host-serial launch, not the GPU).
//!
//! It measures TWO paths over the same single-row OLTP INSERT and reconciles them:
//!
//!   Path A — SQL-honest, end-to-end: `engine.execute_text(seq, "INSERT INTO ... VALUES (...)")` per row.
//!            This is exactly the report card's load loop — it includes SQL PARSE + plan + apply + commit.
//!            This is the headline (should reproduce ~111K rows/s).
//!
//!   Path B — profiled apply+commit: `engine.execute_relational_copy_rows_profiled(seq, &copy, [row])` per
//!            row. It drives the SAME commit machinery (commit_mutation -> apply_insert) but from a
//!            pre-materialized SqlValue row, so it does NOT parse SQL. It returns the 11-stage
//!            `RelationalCopyAdmissionProfile` (render / preflights / row_prepare / mvcc_insert /
//!            value_index_append / residency_invalidation / wal_flush + the commit_total & current_apply
//!            roll-ups). Single-row-per-call so the per-commit/WAL/residency overhead matches real
//!            single-row OLTP (NOT a bulk-load hack — the slow per-row load is the signal).
//!
//!   Reconciliation: wall(A)/row - wall(B)/row  ~=  the SQL parse/plan cost the stage profile cannot see.
//!                   sum(leaf stages)            ~=  wall(B)/row (the rest is wrapper/closure residual).
//!
//! FIDELITY CAVEAT: the engine's profile truncates each timed sub-interval to whole MICROSECONDS
//! (`.as_micros()`). That is fine for bulk COPY (many rows per call) but, at one row per call, blurs
//! SUB-microsecond leaf stages toward 0. So the two headline WALLS (A, B) and the A-B parse estimate are
//! measured here at full nanosecond resolution via `Instant`; only the fine-grained per-stage split below
//! carries the truncation caveat. A sub-us leaf reading ~0 is itself a finding (it is negligible per row).
//!
//! Run:  GPU_DB_BENCH_ROWS=1048576 cargo run --release --example r3_insert_profile -p gpu_db_engine
//! Latency is ALWAYS reported WITH throughput (this is OLTP).

use std::env;
use std::error::Error;
use std::time::Instant;

use gpu_db_engine::{Engine, RelationalCopyAdmissionProfile};
use gpu_db_sql::{CopyFromStdin, CopyOptions, SqlValue};

/// Per-row wall latencies (nanoseconds), full resolution — used for throughput + the latency distribution.
struct WallSamples {
    nanos: Vec<u64>,
}

impl WallSamples {
    fn with_capacity(n: usize) -> Self {
        Self {
            nanos: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, ns: u64) {
        self.nanos.push(ns);
    }

    fn total_secs(&self) -> f64 {
        self.nanos.iter().map(|&n| n as f64).sum::<f64>() / 1e9
    }

    fn rows_per_sec(&self) -> f64 {
        let secs = self.total_secs();
        if secs <= 0.0 {
            0.0
        } else {
            self.nanos.len() as f64 / secs
        }
    }

    /// p-th percentile per-row latency in microseconds (nearest-rank).
    fn pct_us(&self, pct: usize) -> f64 {
        if self.nanos.is_empty() {
            return 0.0;
        }
        let mut sorted = self.nanos.clone();
        sorted.sort_unstable();
        let clamped = pct.clamp(1, 100);
        let rank = (clamped * sorted.len()).div_ceil(100);
        sorted[rank.saturating_sub(1)] as f64 / 1_000.0
    }

    fn max_us(&self) -> f64 {
        self.nanos.iter().copied().max().unwrap_or(0) as f64 / 1_000.0
    }

    fn mean_us(&self) -> f64 {
        if self.nanos.is_empty() {
            return 0.0;
        }
        self.nanos.iter().map(|&n| n as f64).sum::<f64>() / self.nanos.len() as f64 / 1_000.0
    }
}

/// Accumulated `RelationalCopyAdmissionProfile` stage micros across every profiled row.
#[derive(Default)]
struct StageTotals {
    render_sql_wal_payload: u128,
    unique_preflight: u128,
    check_preflight: u128,
    foreign_key_preflight: u128,
    row_prepare: u128,
    mvcc_insert: u128,
    value_index_append: u128,
    residency_invalidation: u128,
    wal_commit_flush_boundary: u128,
    // Roll-ups (NOT leaves — they CONTAIN the stages above; shown as cross-checks, never summed in).
    current_apply_total: u128,
    commit_total: u128,
}

impl StageTotals {
    fn add(&mut self, p: &RelationalCopyAdmissionProfile) {
        self.render_sql_wal_payload += p.render_sql_wal_payload_micros;
        self.unique_preflight += p.unique_preflight_micros;
        self.check_preflight += p.check_preflight_micros;
        self.foreign_key_preflight += p.foreign_key_preflight_micros;
        self.row_prepare += p.row_prepare_micros;
        self.mvcc_insert += p.mvcc_insert_micros;
        self.value_index_append += p.value_index_append_micros;
        self.residency_invalidation += p.residency_invalidation_micros;
        self.wal_commit_flush_boundary += p.wal_commit_flush_boundary_micros;
        self.current_apply_total += p.current_apply_total_micros;
        self.commit_total += p.commit_total_micros;
    }

    /// Sum of the leaf stages (the ones that partition the work; roll-ups excluded).
    fn leaf_sum(&self) -> u128 {
        self.render_sql_wal_payload
            + self.unique_preflight
            + self.check_preflight
            + self.foreign_key_preflight
            + self.row_prepare
            + self.mvcc_insert
            + self.value_index_append
            + self.residency_invalidation
            + self.wal_commit_flush_boundary
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows = env::var("GPU_DB_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1_048_576);
    let device_info = env::var("GPU_DB_BENCH_DEVICE_INFO")
        .unwrap_or_else(|_| "not reported by runner".to_string());

    println!("# R3 INSERT Profiler (Step 1 — measure, no optimization)");
    println!();
    println!("- rows: {rows}");
    println!("- table: accounts(id INT, balance INT)");
    println!("- concurrency: 1");
    println!("- device_info: {device_info}");
    println!();

    let path_a = run_path_a(rows)?;
    let (path_b, stages) = run_path_b(rows)?;

    print_wall(
        "Path A — SQL-honest end-to-end (execute_text: parse + plan + apply + commit)",
        &path_a,
    );
    println!();
    print_wall(
        "Path B — profiled apply + commit only (pre-materialized row, NO parse)",
        &path_b,
    );
    println!();
    print_stage_decomposition(&stages, &path_b, rows);
    println!();
    print_reconciliation(&path_a, &path_b, &stages, rows);

    Ok(())
}

/// Path A: the SQL-honest headline. One CREATE TABLE, then `rows` single-row INSERTs via `execute_text`.
fn run_path_a(rows: usize) -> Result<WallSamples, Box<dyn Error>> {
    let engine = Engine::new_local();
    engine.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;

    let mut samples = WallSamples::with_capacity(rows);
    for i in 0..rows {
        let id = i as i64;
        let balance = (i % 1_000_000) as i64;
        let sql = format!("INSERT INTO accounts (id, balance) VALUES ({id}, {balance})");
        let seq = (i as u64) + 2; // seq 1 was the CREATE TABLE
        let start = Instant::now();
        engine.execute_text(seq, &sql)?;
        samples.push(start.elapsed().as_nanos() as u64);
    }
    Ok(samples)
}

/// Path B: the apply+commit decomposition. Same single-row commits, but from pre-materialized SqlValue
/// rows (so no SQL parse), capturing the 11-stage profile per row.
fn run_path_b(rows: usize) -> Result<(WallSamples, StageTotals), Box<dyn Error>> {
    let mut engine = Engine::new_local();
    engine.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")?;

    let copy = CopyFromStdin {
        table: "accounts".to_string(),
        columns: Some(vec!["id".to_string(), "balance".to_string()]),
        options: CopyOptions::TEXT,
    };

    let mut samples = WallSamples::with_capacity(rows);
    let mut stages = StageTotals::default();
    for i in 0..rows {
        let id = i as i32;
        let balance = (i % 1_000_000) as i32;
        let row = vec![vec![SqlValue::Int4(id), SqlValue::Int4(balance)]];
        let seq = (i as u64) + 2;
        let start = Instant::now();
        let (_n, profile) = engine.execute_relational_copy_rows_profiled(seq, &copy, row)?;
        samples.push(start.elapsed().as_nanos() as u64);
        stages.add(&profile);
    }
    Ok((samples, stages))
}

fn print_wall(name: &str, w: &WallSamples) {
    println!("## {name}");
    println!("- rows_per_sec: {:.0}", w.rows_per_sec());
    println!("- total_wall_s: {:.3}", w.total_secs());
    println!("- per_row_mean_us: {:.3}", w.mean_us());
    println!("- per_row_p50_us: {:.3}", w.pct_us(50));
    println!("- per_row_p99_us: {:.3}", w.pct_us(99));
    println!("- per_row_p99_9_us: {:.3}", percentile_999_us(w));
    println!("- per_row_max_us: {:.3}", w.max_us());
}

/// p99.9 nearest-rank (the .pct_us API takes an integer percentile; p99.9 needs its own rank math).
fn percentile_999_us(w: &WallSamples) -> f64 {
    if w.nanos.is_empty() {
        return 0.0;
    }
    let mut sorted = w.nanos.clone();
    sorted.sort_unstable();
    let rank = (999usize * sorted.len()).div_ceil(1000); // ceil(0.999 * n)
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)] as f64 / 1_000.0
}

fn print_stage_decomposition(stages: &StageTotals, path_b: &WallSamples, rows: usize) {
    let rows_f = rows as f64;
    // Denominator for "% of": the measured Path-B wall (full ns resolution), in microseconds.
    let wall_b_us = path_b.total_secs() * 1e6;
    println!("## Path B stage decomposition (per-row us, % of Path-B wall)");
    println!("- NOTE: stage micros are .as_micros()-truncated per row -> sub-us leaves under-report (->~0).");
    println!("- PLANE: [DATA]=data-plane work to MOVE to GPU; [CTRL]=control/durability that stays host;");
    println!("         [DUAL]=artifact of the host-store<->GPU-resident duality, ELIMINATED (not moved).");
    println!("- leaf stages (these partition the work):");
    print_stage(
        "[DATA] unique_preflight",
        stages.unique_preflight,
        rows_f,
        wall_b_us,
    );
    print_stage(
        "[DATA] check_preflight",
        stages.check_preflight,
        rows_f,
        wall_b_us,
    );
    print_stage(
        "[DATA] foreign_key_preflight",
        stages.foreign_key_preflight,
        rows_f,
        wall_b_us,
    );
    print_stage("[DATA] row_prepare", stages.row_prepare, rows_f, wall_b_us);
    print_stage("[DATA] mvcc_insert", stages.mvcc_insert, rows_f, wall_b_us);
    print_stage(
        "[DATA] value_index_append",
        stages.value_index_append,
        rows_f,
        wall_b_us,
    );
    print_stage(
        "[DUAL] residency_invalidation",
        stages.residency_invalidation,
        rows_f,
        wall_b_us,
    );
    print_stage(
        "[CTRL] render_sql_wal_payload",
        stages.render_sql_wal_payload,
        rows_f,
        wall_b_us,
    );
    print_stage(
        "[CTRL] wal_commit_flush_boundary",
        stages.wal_commit_flush_boundary,
        rows_f,
        wall_b_us,
    );
    println!("- roll-ups (CONTAIN the leaves above; cross-check only, not summed):");
    print_stage(
        "current_apply_total",
        stages.current_apply_total,
        rows_f,
        wall_b_us,
    );
    print_stage("commit_total", stages.commit_total, rows_f, wall_b_us);
}

fn print_stage(name: &str, total_micros: u128, rows_f: f64, wall_b_us: f64) {
    let per_row_us = total_micros as f64 / rows_f;
    let pct = if wall_b_us > 0.0 {
        (total_micros as f64 / wall_b_us) * 100.0
    } else {
        0.0
    };
    println!("  - {name}: {per_row_us:.3} us/row ({pct:.1}% of B wall)");
}

fn print_reconciliation(
    path_a: &WallSamples,
    path_b: &WallSamples,
    stages: &StageTotals,
    rows: usize,
) {
    let a_us = path_a.mean_us();
    let b_us = path_b.mean_us();
    let parse_us = (a_us - b_us).max(0.0);
    let parse_pct = if a_us > 0.0 {
        (parse_us / a_us) * 100.0
    } else {
        0.0
    };

    let leaf_sum_us = stages.leaf_sum() as f64 / rows as f64;
    let b_wall_us = path_b.mean_us();
    let residual_us = (b_wall_us - leaf_sum_us).max(0.0);

    println!("## Reconciliation");
    println!(
        "- Path A mean: {a_us:.3} us/row  ({:.0} rows/s)",
        path_a.rows_per_sec()
    );
    println!(
        "- Path B mean: {b_us:.3} us/row  ({:.0} rows/s)",
        path_b.rows_per_sec()
    );
    println!("- => SQL parse/plan (A - B): {parse_us:.3} us/row ({parse_pct:.1}% of the SQL-honest cost)");
    println!("- leaf-stage sum: {leaf_sum_us:.3} us/row");
    println!("- Path B wrapper residual (B wall - leaf sum): {residual_us:.3} us/row");
    println!(
        "  (residual = Insert-struct build + closure + uncounted commit overhead; large residual or"
    );
    println!(
        "   sub-us truncation both surface here — read it together with the stage notes above.)"
    );
    println!();

    // Target architecture: host = control plane only; data plane -> GPU. Map the measured cost onto it.
    let r = rows as f64;
    let data_plane_us = (stages.unique_preflight
        + stages.check_preflight
        + stages.foreign_key_preflight
        + stages.row_prepare
        + stages.mvcc_insert
        + stages.value_index_append) as f64
        / r;
    let dual_artifact_us = stages.residency_invalidation as f64 / r;
    let durability_log_us =
        (stages.render_sql_wal_payload + stages.wal_commit_flush_boundary) as f64 / r;
    println!("## Target-architecture mapping (host = control plane; data plane -> GPU)");
    println!("  Of the SQL-honest {a_us:.3} us/row:");
    println!("  - [CTRL] parse/plan (stays host):            {parse_us:.3} us/row");
    println!("  - [CTRL] render WAL + fsync/log (stays host): {durability_log_us:.3} us/row");
    println!("  - [DATA] constraints+encode+mvcc+index (-> GPU): {data_plane_us:.3} us/row");
    println!("  - [DUAL] residency invalidation (ELIMINATED):  {dual_artifact_us:.3} us/row");
    println!("  Read: [DATA]+[DUAL] is the cost the host engine carries that the control-plane target removes;");
    println!("  [CTRL] is the floor that remains. (Stage truncation caveat applies to the sub-us [DATA] leaves.)");
}
