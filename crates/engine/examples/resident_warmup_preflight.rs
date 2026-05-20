use std::env;
use std::process;

use gpu_db_engine::{Engine, RelationalResidencyWarmupAction, RelationalResidencyWarmupPolicy};
use gpu_db_protocol::{parse_command, Command};

#[derive(Debug)]
struct Config {
    apply: bool,
    scenario: Scenario,
    refresh_invalidated: bool,
    include_missing: bool,
    budget_bytes: Option<u64>,
    max_table_count: Option<usize>,
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
    let mut tables = match config.scenario {
        Scenario::OversizedBudget => vec!["oversized".to_string()],
        _ => vec!["events".to_string(), "aux".to_string()],
    };
    if config.include_missing {
        tables.push("missing".to_string());
    }
    if config.scenario == Scenario::Invalidated {
        engine.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
            tables: vec!["events".to_string()],
            refresh_invalidated: true,
            ..RelationalResidencyWarmupPolicy::default()
        });
        engine
            .execute_text(10, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
            .map_err(|err| err.to_string())?;
        tables = vec!["events".to_string()];
    }
    if config.scenario == Scenario::MemoryPressure {
        engine.mark_gpu_memory_pressured(0);
        tables = vec!["events".to_string()];
    }

    let budget_bytes = if config.scenario == Scenario::OversizedBudget {
        Some(1)
    } else {
        config.budget_bytes
    };
    let report = engine.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables,
        max_table_count: config.max_table_count,
        budget_bytes,
        refresh_invalidated: config.refresh_invalidated,
        ..RelationalResidencyWarmupPolicy::default()
    });

    println!("mode={}", if config.apply { "apply" } else { "dry-run" });
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
    println!("entry_count={}", report.entries.len());

    for entry in &report.entries {
        println!("entry.table={}", entry.table);
        println!("entry.action={}", action_name(&entry.action));
        println!("entry.reason={}", entry.reason);
        println!("entry.resident_bytes={}", entry.resident_bytes);
        println!("entry.evicted_tables={}", entry.evicted_tables.join(","));
        if let Some(route) = entry.route_decision.as_ref() {
            println!("entry.route.accepted={}", route.accepted);
            println!("entry.route.reason={}", route.reason);
            println!("entry.route.query_shape={}", route.query_shape);
            println!("entry.route.cache_state={}", route.cache_state);
            println!(
                "entry.route.has_retained_device_memory={}",
                route.has_retained_device_memory
            );
            println!(
                "entry.route.h2d_bytes_if_resident={}",
                route.h2d_bytes_if_resident
            );
            println!("entry.route.d2h_rows_estimate={}", route.d2h_rows_estimate);
        } else {
            println!("entry.route.accepted=none");
        }
    }

    if config.apply {
        for entry in &report.entries {
            let Some(route) = entry.route_decision.as_ref() else {
                continue;
            };
            if !route.accepted {
                continue;
            }
            let sql = format!("SELECT COUNT(*) FROM {}", entry.table);
            let Command::Select(select) = parse_command(&sql).map_err(|err| err.to_string())?
            else {
                return Err("warmup verification SELECT did not parse as SELECT".to_string());
            };
            let result = engine
                .execute_relational_select(&select)
                .map_err(|err| err.to_string())?;
            println!("verification.table={}", entry.table);
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
        apply: false,
        scenario: Scenario::Basic,
        refresh_invalidated: true,
        include_missing: false,
        budget_bytes: None,
        max_table_count: None,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--apply" => config.apply = true,
            "--dry-run" => config.apply = false,
            "--scenario" => {
                config.scenario = parse_scenario(&next_value(&mut args, "--scenario")?)?
            }
            "--refresh-invalidated" => config.refresh_invalidated = true,
            "--no-refresh-invalidated" => config.refresh_invalidated = false,
            "--include-missing" => config.include_missing = true,
            "--budget-bytes" => {
                config.budget_bytes = Some(parse_u64(&next_value(&mut args, "--budget-bytes")?)?)
            }
            "--max-table-count" => {
                config.max_table_count =
                    Some(parse_usize(&next_value(&mut args, "--max-table-count")?)?)
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

fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
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

fn action_name(action: &RelationalResidencyWarmupAction) -> &'static str {
    match action {
        RelationalResidencyWarmupAction::Warmed => "warmed",
        RelationalResidencyWarmupAction::Refreshed => "refreshed",
        RelationalResidencyWarmupAction::AlreadyResident => "already-resident",
        RelationalResidencyWarmupAction::Skipped => "skipped",
        RelationalResidencyWarmupAction::Error => "error",
    }
}

fn print_usage() {
    println!(
        "Usage: cargo run -p gpu_db_engine --example resident_warmup_preflight -- [--apply] [--scenario basic|invalidated|memory-pressure|oversized-budget] [--budget-bytes N] [--include-missing]"
    );
}
