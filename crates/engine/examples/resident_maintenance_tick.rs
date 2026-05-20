use std::env;
use std::process;

use gpu_db_engine::{Engine, RelationalResidencyMaintenancePolicy};
use gpu_db_protocol::{parse_command, Command};

#[derive(Debug)]
struct Config {
    scenario: Scenario,
    budget_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
    Basic,
    Invalidated,
    MemoryPressure,
    OversizedBudget,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error={err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args(env::args().skip(1))?;
    let mut engine = seeded_engine()?;
    if config.scenario == Scenario::Invalidated {
        engine.maintain_relational_residency_with_policy(RelationalResidencyMaintenancePolicy {
            tables: vec!["events".to_string()],
            ..RelationalResidencyMaintenancePolicy::default()
        });
        engine
            .execute_text(10, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
            .map_err(|err| err.to_string())?;
    }
    if config.scenario == Scenario::MemoryPressure {
        engine.mark_gpu_memory_pressured(0);
    }

    let mut policy = RelationalResidencyMaintenancePolicy {
        budget_bytes: config.budget_bytes,
        ..RelationalResidencyMaintenancePolicy::default()
    };
    if config.scenario == Scenario::OversizedBudget {
        policy.tables = vec!["oversized".to_string()];
        policy.budget_bytes = Some(1);
    }

    let report = engine.maintain_relational_residency_with_policy(policy);
    println!("scenario={}", scenario_name(config.scenario));
    println!("gpu_id={}", report.gpu_id);
    println!(
        "budget_bytes={}",
        report
            .budget_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unbounded".to_string())
    );
    println!("requested_tables={}", report.requested_tables.join(","));
    println!("entry_count={}", report.entry_count);
    println!("warmed_count={}", report.warmed_count);
    println!("refreshed_count={}", report.refreshed_count);
    println!("already_resident_count={}", report.already_resident_count);
    println!("skipped_count={}", report.skipped_count);
    println!("error_count={}", report.error_count);
    println!("route_ready_count={}", report.route_ready_count);
    println!("route_blocked_count={}", report.route_blocked_count);
    println!("route_ready_tables={}", report.route_ready_tables.join(","));
    for blocker in &report.route_blockers {
        println!("route_blocker.table={}", blocker.table);
        println!("route_blocker.reason={}", blocker.reason);
    }
    for entry in &report.entries {
        println!("entry.table={}", entry.table);
        println!("entry.action={:?}", entry.action);
        println!("entry.reason={}", entry.reason);
    }

    for table in &report.route_ready_tables {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        let Command::Select(select) = parse_command(&sql).map_err(|err| err.to_string())? else {
            return Err("maintenance verification SELECT did not parse as SELECT".to_string());
        };
        let result = engine
            .execute_relational_select(&select)
            .map_err(|err| err.to_string())?;
        println!("verification.table={table}");
        println!("verification.rows={}", result.rows.len());
        println!("verification.executed_target={:?}", result.executed_target);
        println!(
            "verification.fallback_reason={}",
            result
                .fallback_reason
                .map(|reason| format!("{reason:?}"))
                .unwrap_or_else(|| "none".to_string())
        );
    }

    Ok(())
}

fn seeded_engine() -> Result<Engine, String> {
    let mut engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .map_err(|err| err.to_string())?;
    engine
        .execute_text(2, "CREATE TABLE aux (id INT, label TEXT)")
        .map_err(|err| err.to_string())?;
    engine
        .execute_text(3, "CREATE TABLE oversized (id INT, label TEXT)")
        .map_err(|err| err.to_string())?;
    engine
        .execute_text(
            4,
            "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
        )
        .map_err(|err| err.to_string())?;
    engine
        .execute_text(5, "INSERT INTO aux (id, label) VALUES (1, 'aux')")
        .map_err(|err| err.to_string())?;
    engine
        .execute_text(
            6,
            "INSERT INTO oversized (id, label) VALUES (1, 'this-row-is-too-large-for-budget')",
        )
        .map_err(|err| err.to_string())?;
    Ok(engine)
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut config = Config {
        scenario: Scenario::Basic,
        budget_bytes: None,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scenario" => {
                config.scenario = parse_scenario(&next_value(&mut args, "--scenario")?)?
            }
            "--budget-bytes" => {
                config.budget_bytes = Some(parse_u64(&next_value(&mut args, "--budget-bytes")?)?)
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("unsupported argument {other}")),
        }
    }
    Ok(config)
}

fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_u64(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("invalid integer {value}: {err}"))
}

fn parse_scenario(value: &str) -> Result<Scenario, String> {
    match value {
        "basic" => Ok(Scenario::Basic),
        "invalidated" => Ok(Scenario::Invalidated),
        "memory-pressure" => Ok(Scenario::MemoryPressure),
        "oversized-budget" => Ok(Scenario::OversizedBudget),
        _ => Err(format!("unsupported scenario {value}")),
    }
}

fn scenario_name(scenario: Scenario) -> &'static str {
    match scenario {
        Scenario::Basic => "basic",
        Scenario::Invalidated => "invalidated",
        Scenario::MemoryPressure => "memory-pressure",
        Scenario::OversizedBudget => "oversized-budget",
    }
}

fn print_usage() {
    println!("resident_maintenance_tick [--scenario basic|invalidated|memory-pressure|oversized-budget] [--budget-bytes n]");
}
