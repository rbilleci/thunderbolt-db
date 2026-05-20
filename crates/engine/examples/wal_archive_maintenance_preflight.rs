use std::env;
use std::path::{Path, PathBuf};
use std::process;

use gpu_db_engine::Engine;
use gpu_db_wal::WalArchiveTimeline;

#[derive(Debug)]
struct Config {
    control_path: Option<PathBuf>,
    archive_manifest_path: Option<PathBuf>,
    timeline_registry_path: Option<PathBuf>,
    retained_timeline_id: Option<String>,
    current_timestamp_micros: Option<u64>,
    pitr_window_micros: Option<u64>,
    apply: bool,
    recover_retained: bool,
    write_demo_fixture: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error={err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args(env::args().skip(1))?;
    if let Some(dir) = config.write_demo_fixture.as_deref() {
        write_demo_fixture(dir)?;
        println!("demo_fixture={}", dir.display());
        println!("control={}", dir.join("base").join("CONTROL").display());
        println!(
            "archive_manifest={}",
            dir.join("archive").join("MANIFEST").display()
        );
        println!(
            "timeline_registry={}",
            dir.join("TIMELINE_REGISTRY").display()
        );
        println!("retained_timeline=timeline-keep-0002");
        println!("current_timestamp_micros=6000");
        println!("pitr_window_micros=3000");
        return Ok(());
    }

    let control_path = required_path(&config.control_path, "--control")?;
    let archive_manifest_path = required_path(&config.archive_manifest_path, "--archive-manifest")?;
    let timeline_registry_path =
        required_path(&config.timeline_registry_path, "--timeline-registry")?;
    let retained_timeline_id = config
        .retained_timeline_id
        .as_deref()
        .ok_or_else(|| "missing --retain-timeline".to_string())?;
    let current_timestamp_micros = config
        .current_timestamp_micros
        .ok_or_else(|| "missing --current-timestamp-micros".to_string())?;
    let pitr_window_micros = config
        .pitr_window_micros
        .ok_or_else(|| "missing --pitr-window-micros".to_string())?;

    let plan = if config.apply {
        Engine::apply_durable_wal_archive_maintenance_cleanup(
            control_path,
            archive_manifest_path,
            timeline_registry_path,
            retained_timeline_id,
            current_timestamp_micros,
            pitr_window_micros,
        )
    } else {
        Engine::plan_durable_wal_archive_maintenance_cleanup(
            control_path,
            archive_manifest_path,
            timeline_registry_path,
            retained_timeline_id,
            current_timestamp_micros,
            pitr_window_micros,
        )
    }
    .map_err(|err| err.to_string())?;

    println!("mode={}", if config.apply { "apply" } else { "dry-run" });
    println!(
        "current_timestamp_micros={}",
        plan.retention_window_plan.current_timestamp_micros
    );
    println!(
        "pitr_window_micros={}",
        plan.retention_window_plan.pitr_window_micros
    );
    println!(
        "cutoff_timestamp_micros={}",
        plan.retention_window_plan.cutoff_timestamp_micros
    );
    println!("base_txn_id={}", plan.retention_window_plan.base_txn_id);
    println!(
        "base_timestamp_micros={}",
        plan.retention_window_plan.base_timestamp_micros
    );
    println!(
        "retained_record_count={}",
        plan.retention_window_plan
            .retention_plan
            .retained_record_count
    );
    println!(
        "removed_record_count={}",
        plan.retention_window_plan
            .retention_plan
            .removed_record_count
    );
    println!(
        "removed_archive_segments={}",
        format_paths(&plan.retention_window_plan.retention_plan.removed_segments)
    );
    println!(
        "retained_timeline_ids={}",
        plan.timeline_prune_plan.retained_timeline_ids.join(",")
    );
    println!(
        "removed_timeline_ids={}",
        plan.timeline_prune_plan.removed_timeline_ids.join(",")
    );
    println!(
        "removed_timeline_paths={}",
        format_paths(&plan.timeline_prune_plan.removed_timeline_paths)
    );
    println!(
        "removed_branch_manifests={}",
        format_paths(&plan.timeline_prune_plan.removed_branch_manifest_paths)
    );
    println!(
        "removed_timeline_segments={}",
        format_paths(&plan.timeline_prune_plan.removed_segment_paths)
    );

    if config.recover_retained {
        let recovered = Engine::recover_from_registered_durable_wal_archive_timeline(
            timeline_registry_path,
            retained_timeline_id,
        )
        .map_err(|err| err.to_string())?;
        println!(
            "retained_timeline_recovery_wal_records={}",
            recovered.wal_flushed_count()
        );
    }

    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut config = Config {
        control_path: None,
        archive_manifest_path: None,
        timeline_registry_path: None,
        retained_timeline_id: None,
        current_timestamp_micros: None,
        pitr_window_micros: None,
        apply: false,
        recover_retained: false,
        write_demo_fixture: None,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--control" => config.control_path = Some(next_path(&mut args, "--control")?),
            "--archive-manifest" => {
                config.archive_manifest_path = Some(next_path(&mut args, "--archive-manifest")?)
            }
            "--timeline-registry" => {
                config.timeline_registry_path = Some(next_path(&mut args, "--timeline-registry")?)
            }
            "--retain-timeline" => {
                config.retained_timeline_id = Some(next_value(&mut args, "--retain-timeline")?)
            }
            "--current-timestamp-micros" => {
                config.current_timestamp_micros = Some(parse_u64(&next_value(
                    &mut args,
                    "--current-timestamp-micros",
                )?)?)
            }
            "--pitr-window-micros" => {
                config.pitr_window_micros =
                    Some(parse_u64(&next_value(&mut args, "--pitr-window-micros")?)?)
            }
            "--apply" => config.apply = true,
            "--recover-retained" => config.recover_retained = true,
            "--write-demo-fixture" => {
                config.write_demo_fixture = Some(next_path(&mut args, "--write-demo-fixture")?)
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

fn next_path(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<PathBuf, String> {
    Ok(PathBuf::from(next_value(args, flag)?))
}

fn parse_u64(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("invalid unsigned integer {value}: {err}"))
}

fn required_path<'a>(path: &'a Option<PathBuf>, flag: &str) -> Result<&'a Path, String> {
    path.as_deref().ok_or_else(|| format!("missing {flag}"))
}

fn format_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn write_demo_fixture(dir: &Path) -> Result<(), String> {
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let source_timeline_path = dir.join("timeline-main").join("TIMELINE");
    let keep_manifest = dir.join("timeline-keep").join("MANIFEST");
    let keep_segments = dir.join("timeline-keep").join("segments");
    let keep_timeline_path = dir.join("timeline-keep").join("TIMELINE");
    let prune_manifest = dir.join("timeline-prune").join("MANIFEST");
    let prune_segments = dir.join("timeline-prune").join("segments");
    let prune_timeline_path = dir.join("timeline-prune").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");

    let mut engine = Engine::new_local();
    engine
        .execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .map_err(|err| err.to_string())?;
    engine
        .execute_text_at_timestamp_micros(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada')",
            2_000,
        )
        .map_err(|err| err.to_string())?;
    engine
        .persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .map_err(|err| err.to_string())?;
    engine
        .execute_text_at_timestamp_micros(
            3,
            "INSERT INTO people (id, name) VALUES (2, 'Grace')",
            3_000,
        )
        .map_err(|err| err.to_string())?;
    engine
        .execute_text_at_timestamp_micros(
            4,
            "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
            4_000,
        )
        .map_err(|err| err.to_string())?;
    engine
        .execute_text_at_timestamp_micros(
            5,
            "INSERT INTO people (id, name) VALUES (4, 'Dorothy')",
            5_000,
        )
        .map_err(|err| err.to_string())?;
    engine
        .persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .map_err(|err| err.to_string())?;

    Engine::write_durable_wal_archive_timeline(
        &source_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-main-0001".to_string(),
            parent_timeline_id: None,
            fork_txn_id: 0,
            fork_timestamp_micros: None,
            source_manifest_path: manifest_path.clone(),
            branch_manifest_path: manifest_path.clone(),
        },
    )
    .map_err(|err| err.to_string())?;
    Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(
        &manifest_path,
        &keep_manifest,
        &keep_segments,
        &keep_timeline_path,
        "timeline-keep-0002",
        Some("timeline-main-0001"),
        4_000,
    )
    .map_err(|err| err.to_string())?;
    Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(
        &manifest_path,
        &prune_manifest,
        &prune_segments,
        &prune_timeline_path,
        "timeline-prune-0003",
        Some("timeline-main-0001"),
        3_000,
    )
    .map_err(|err| err.to_string())?;
    Engine::register_durable_wal_archive_timeline(&registry_path, &source_timeline_path)
        .map_err(|err| err.to_string())?;
    Engine::register_durable_wal_archive_timeline(&registry_path, &keep_timeline_path)
        .map_err(|err| err.to_string())?;
    Engine::register_durable_wal_archive_timeline(&registry_path, &prune_timeline_path)
        .map_err(|err| err.to_string())?;

    Ok(())
}

fn print_usage() {
    println!(
        "usage: wal_archive_maintenance_preflight --control PATH --archive-manifest PATH --timeline-registry PATH --retain-timeline ID --current-timestamp-micros N --pitr-window-micros N [--apply] [--recover-retained]"
    );
    println!("       wal_archive_maintenance_preflight --write-demo-fixture DIR");
}
