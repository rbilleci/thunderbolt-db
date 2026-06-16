use std::env;
use std::path::{Path, PathBuf};
use std::process;

use gpu_db_engine::Engine;
use gpu_db_sql::{parse_command, Command};

#[derive(Debug)]
struct Config {
    archive_manifest_path: Option<PathBuf>,
    backup_manifest_path: Option<PathBuf>,
    object_dir: Option<PathBuf>,
    restored_manifest_path: Option<PathBuf>,
    restored_segment_dir: Option<PathBuf>,
    recover_timestamp_micros: Option<u64>,
    restore_only: bool,
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
        println!(
            "archive_manifest={}",
            dir.join("archive").join("MANIFEST").display()
        );
        println!("recover_timestamp_micros=3000");
        return Ok(());
    }

    let backup_manifest_path = required_path(&config.backup_manifest_path, "--backup-manifest")?;
    let restored_manifest_path =
        required_path(&config.restored_manifest_path, "--restored-manifest")?;
    let restored_segment_dir =
        required_path(&config.restored_segment_dir, "--restored-segment-dir")?;

    let backup = if config.restore_only {
        gpu_db_wal::read_wal_archive_object_backup_manifest(backup_manifest_path)
            .map_err(|err| err.to_string())?
    } else {
        let archive_manifest_path =
            required_path(&config.archive_manifest_path, "--archive-manifest")?;
        let object_dir = required_path(&config.object_dir, "--object-dir")?;
        Engine::export_durable_wal_archive_object_backup(
            archive_manifest_path,
            backup_manifest_path,
            object_dir,
        )
        .map_err(|err| err.to_string())?
    };

    let restored = Engine::restore_durable_wal_archive_object_backup(
        backup_manifest_path,
        restored_manifest_path,
        restored_segment_dir,
    )
    .map_err(|err| err.to_string())?;

    println!(
        "mode={}",
        if config.restore_only {
            "restore-only"
        } else {
            "export-restore"
        }
    );
    println!(
        "source_record_count={}",
        backup.archive_manifest.checkpoint.durable_record_count
    );
    println!("object_count={}", backup.objects.len());
    println!(
        "restored_record_count={}",
        restored.checkpoint.durable_record_count
    );
    println!("restored_segments={}", restored.segments.len());

    if let Some(timestamp_micros) = config.recover_timestamp_micros {
        let mut recovered = Engine::recover_from_durable_wal_archive_to_timestamp_micros(
            restored_manifest_path,
            timestamp_micros,
        )
        .map_err(|err| err.to_string())?;
        let result = select_people_by_name(&mut recovered, "Grace")?;
        println!("recovered_timestamp_micros={timestamp_micros}");
        println!("recovered_wal_records={}", recovered.wal_flushed_count());
        println!("recovered_grace_rows={}", result.rows.len());
    }

    Ok(())
}

fn select_people_by_name(
    engine: &mut Engine,
    name: &str,
) -> Result<gpu_db_engine::RelationalSelectResult, String> {
    let sql = format!("SELECT id FROM people WHERE name = '{name}'");
    let Command::Select(select) = parse_command(&sql).map_err(|err| err.to_string())? else {
        return Err("expected SELECT command".to_string());
    };
    engine
        .execute_relational_select(&select)
        .map_err(|err| err.to_string())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut config = Config {
        archive_manifest_path: None,
        backup_manifest_path: None,
        object_dir: None,
        restored_manifest_path: None,
        restored_segment_dir: None,
        recover_timestamp_micros: None,
        restore_only: false,
        write_demo_fixture: None,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--archive-manifest" => {
                config.archive_manifest_path = Some(next_path(&mut args, "--archive-manifest")?)
            }
            "--backup-manifest" => {
                config.backup_manifest_path = Some(next_path(&mut args, "--backup-manifest")?)
            }
            "--object-dir" => config.object_dir = Some(next_path(&mut args, "--object-dir")?),
            "--restored-manifest" => {
                config.restored_manifest_path = Some(next_path(&mut args, "--restored-manifest")?)
            }
            "--restored-segment-dir" => {
                config.restored_segment_dir = Some(next_path(&mut args, "--restored-segment-dir")?)
            }
            "--recover-timestamp-micros" => {
                config.recover_timestamp_micros = Some(parse_u64(&next_value(
                    &mut args,
                    "--recover-timestamp-micros",
                )?)?)
            }
            "--restore-only" => config.restore_only = true,
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

fn write_demo_fixture(dir: &Path) -> Result<(), String> {
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let engine = Engine::new_local();
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
        .execute_text_at_timestamp_micros(
            3,
            "INSERT INTO people (id, name) VALUES (2, 'Grace')",
            3_000,
        )
        .map_err(|err| err.to_string())?;
    engine
        .persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .map_err(|err| err.to_string())?;
    Ok(())
}

fn print_usage() {
    println!(
        "usage: wal_archive_object_backup_preflight --archive-manifest PATH --backup-manifest PATH --object-dir DIR --restored-manifest PATH --restored-segment-dir DIR [--recover-timestamp-micros N]"
    );
    println!(
        "       wal_archive_object_backup_preflight --restore-only --backup-manifest PATH --restored-manifest PATH --restored-segment-dir DIR [--recover-timestamp-micros N]"
    );
    println!("       wal_archive_object_backup_preflight --write-demo-fixture DIR");
}
