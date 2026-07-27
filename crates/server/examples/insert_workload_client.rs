//! INSERT-001 deterministic SQL statement-stream generator and verifier.
//!
//! This tool owns the byte-exact input shared by the GPU database and PostgreSQL qualification
//! clients. Replay uses one standalone TCP simple-query client for either backend: every source
//! line is one autocommit statement, with no COPY, prepared protocol, or hidden outer transaction.
//!
//! Examples:
//!
//! ```text
//! cargo run -p gpu_db_server --example insert_workload_client --release -- \
//!   generate --rows 1000000 --chunk 1000 --source target/insert-001/1m.sql \
//!   --manifest target/insert-001/1m.manifest
//!
//! cargo run -p gpu_db_server --example insert_workload_client --release -- \
//!   verify --source target/insert-001/1m.sql \
//!   --manifest target/insert-001/1m.manifest
//!
//! cargo run -p gpu_db_server --example insert_workload_client --release -- \
//!   run --source target/insert-001/1m.sql --manifest target/insert-001/1m.manifest \
//!   --connection "host=127.0.0.1 port=5432 user=postgres dbname=postgres" \
//!   --backend gpu_db --profile development --trial 1
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

const MANIFEST_VERSION: u32 = 1;
const WORKLOAD_ID: &str = "insert001-accounts-int4-v1";
const CREATE_STATEMENT: &str = "CREATE TABLE accounts (id int4, balance int4);\n";
const DEFAULT_ROWS: u64 = 1_000_000;
const DEFAULT_CHUNK: u64 = 1_000;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkloadManifest {
    version: u32,
    workload: String,
    rows: u64,
    chunk: u64,
    insert_statements: u64,
    total_statements: u64,
    source_bytes: u64,
    source_sha256: String,
    statement_stream_sha256: String,
}

impl WorkloadManifest {
    fn serialize(&self) -> String {
        format!(
            "manifest_version={}\n\
             workload={}\n\
             rows={}\n\
             chunk={}\n\
             insert_statements={}\n\
             total_statements={}\n\
             source_bytes={}\n\
             source_sha256={}\n\
             statement_stream_sha256={}\n",
            self.version,
            self.workload,
            self.rows,
            self.chunk,
            self.insert_statements,
            self.total_statements,
            self.source_bytes,
            self.source_sha256,
            self.statement_stream_sha256,
        )
    }

    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("manifest is empty".to_string());
        }
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(format!(
                "manifest exceeds the {MAX_MANIFEST_BYTES}-byte bound"
            ));
        }
        if !bytes.ends_with(b"\n") {
            return Err("manifest must end with one newline".to_string());
        }
        let text =
            std::str::from_utf8(bytes).map_err(|_| "manifest is not valid UTF-8".to_string())?;
        let mut fields = BTreeMap::new();
        for line in text.lines() {
            if line.is_empty() {
                return Err("manifest contains a blank line".to_string());
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("manifest line has no '=': {line:?}"))?;
            if key.is_empty() || value.is_empty() || value.contains('=') {
                return Err(format!("malformed manifest field: {line:?}"));
            }
            if fields.insert(key, value).is_some() {
                return Err(format!("duplicate manifest field: {key}"));
            }
        }
        const EXPECTED_KEYS: [&str; 9] = [
            "manifest_version",
            "workload",
            "rows",
            "chunk",
            "insert_statements",
            "total_statements",
            "source_bytes",
            "source_sha256",
            "statement_stream_sha256",
        ];
        if fields.len() != EXPECTED_KEYS.len() {
            return Err(format!(
                "manifest has {} fields; expected {}",
                fields.len(),
                EXPECTED_KEYS.len()
            ));
        }
        for key in fields.keys() {
            if !EXPECTED_KEYS.contains(key) {
                return Err(format!("unknown manifest field: {key}"));
            }
        }
        let parse_u64 = |key: &str| -> Result<u64, String> {
            parse_canonical_u64(
                fields
                    .get(key)
                    .ok_or_else(|| format!("missing manifest field: {key}"))?,
                &format!("manifest field {key}"),
            )
        };
        let version = parse_u64("manifest_version")?;
        let version =
            u32::try_from(version).map_err(|_| "manifest_version does not fit u32".to_string())?;
        let source_sha256 = required_digest(&fields, "source_sha256")?;
        let statement_stream_sha256 = required_digest(&fields, "statement_stream_sha256")?;
        Ok(Self {
            version,
            workload: fields
                .get("workload")
                .ok_or_else(|| "missing manifest field: workload".to_string())?
                .to_string(),
            rows: parse_u64("rows")?,
            chunk: parse_u64("chunk")?,
            insert_statements: parse_u64("insert_statements")?,
            total_statements: parse_u64("total_statements")?,
            source_bytes: parse_u64("source_bytes")?,
            source_sha256,
            statement_stream_sha256,
        })
    }
}

fn required_digest(fields: &BTreeMap<&str, &str>, key: &str) -> Result<String, String> {
    let value = fields
        .get(key)
        .ok_or_else(|| format!("missing manifest field: {key}"))?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{key} is not a canonical lowercase SHA-256"));
    }
    Ok((*value).to_string())
}

fn parse_canonical_u64(value: &str, label: &str) -> Result<u64, String> {
    let bytes = value.as_bytes();
    let canonical = bytes == b"0"
        || matches!(bytes.first(), Some(b'1'..=b'9')) && bytes[1..].iter().all(u8::is_ascii_digit);
    if !canonical {
        return Err(format!("{label} is not canonical u64"));
    }
    value
        .parse::<u64>()
        .map_err(|_| format!("{label} does not fit u64"))
}

#[derive(Default)]
struct StreamDigests {
    source: Sha256,
    statements: Sha256,
    source_bytes: u64,
    statement_count: u64,
}

impl StreamDigests {
    fn add_statement(&mut self, statement: &[u8]) -> Result<(), String> {
        let len = u64::try_from(statement.len())
            .map_err(|_| "statement length does not fit u64".to_string())?;
        self.source.update(statement);
        self.statements.update(len.to_le_bytes());
        self.statements.update(statement);
        self.source_bytes = self
            .source_bytes
            .checked_add(len)
            .ok_or_else(|| "source byte count overflow".to_string())?;
        self.statement_count = self
            .statement_count
            .checked_add(1)
            .ok_or_else(|| "statement count overflow".to_string())?;
        Ok(())
    }

    fn finish(self) -> (String, String, u64, u64) {
        (
            digest_hex(self.source.finalize().as_slice()),
            digest_hex(self.statements.finalize().as_slice()),
            self.source_bytes,
            self.statement_count,
        )
    }
}

fn digest_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn validate_shape(rows: u64, chunk: u64) -> Result<(), String> {
    if rows == 0 {
        return Err("rows must be greater than zero".to_string());
    }
    if chunk == 0 {
        return Err("chunk must be greater than zero".to_string());
    }
    Ok(())
}

fn insert_statement_count(rows: u64, chunk: u64) -> Result<u64, String> {
    validate_shape(rows, chunk)?;
    Ok(rows.div_ceil(chunk))
}

fn insert_statement(first_id: u64, end_id: u64) -> Result<Vec<u8>, String> {
    if first_id >= end_id {
        return Err("INSERT statement range must be non-empty".to_string());
    }
    let row_count = end_id - first_id;
    let capacity = usize::try_from(row_count)
        .ok()
        .and_then(|rows| rows.checked_mul(24))
        .and_then(|bytes| bytes.checked_add(60))
        .ok_or_else(|| "INSERT statement capacity overflow".to_string())?;
    let mut statement = String::with_capacity(capacity);
    statement.push_str("INSERT INTO accounts (id, balance) VALUES ");
    for id in first_id..end_id {
        if id != first_id {
            statement.push(',');
        }
        let balance = id
            .checked_mul(7)
            .ok_or_else(|| "fixture balance multiplication overflow".to_string())?
            % 100_000;
        write!(&mut statement, "({id}, {balance})").expect("writing to String cannot fail");
    }
    statement.push_str(";\n");
    Ok(statement.into_bytes())
}

fn expected_statement(ordinal: u64, rows: u64, chunk: u64) -> Result<Vec<u8>, String> {
    if ordinal == 0 {
        return Ok(CREATE_STATEMENT.as_bytes().to_vec());
    }
    let insert_ordinal = ordinal - 1;
    let first_id = insert_ordinal
        .checked_mul(chunk)
        .ok_or_else(|| "fixture row offset overflow".to_string())?;
    if first_id >= rows {
        return Err(format!("statement ordinal {ordinal} exceeds workload"));
    }
    insert_statement(first_id, first_id.saturating_add(chunk).min(rows))
}

struct TempArtifact {
    path: PathBuf,
    armed: bool,
}

impl TempArtifact {
    fn create(target: &Path) -> Result<(Self, File), String> {
        let file_name = target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("target has no UTF-8 file name: {}", target.display()))?;
        let temp_name = format!(".{file_name}.tmp-{}", std::process::id());
        let temp_path = target.with_file_name(temp_name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|error| {
                format!(
                    "cannot create temporary artifact {}: {error}",
                    temp_path.display()
                )
            })?;
        Ok((
            Self {
                path: temp_path,
                armed: true,
            },
            file,
        ))
    }

    fn publish(mut self, target: &Path) -> Result<(), String> {
        if target.exists() {
            return Err(format!(
                "refusing to overwrite existing artifact {}",
                target.display()
            ));
        }
        fs::rename(&self.path, target).map_err(|error| {
            format!(
                "cannot publish {} as {}: {error}",
                self.path.display(),
                target.display()
            )
        })?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for TempArtifact {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn generate_workload(
    rows: u64,
    chunk: u64,
    source_path: &Path,
    manifest_path: &Path,
) -> Result<WorkloadManifest, String> {
    validate_shape(rows, chunk)?;
    if source_path == manifest_path {
        return Err("source and manifest paths must differ".to_string());
    }
    for path in [source_path, manifest_path] {
        if path.exists() {
            return Err(format!(
                "refusing to overwrite existing artifact {}",
                path.display()
            ));
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create artifact directory {}: {error}",
                parent.display()
            )
        })?;
    }

    let (source_temp, source_file) = TempArtifact::create(source_path)?;
    let mut writer = BufWriter::new(source_file);
    let mut digests = StreamDigests::default();
    let total_inserts = insert_statement_count(rows, chunk)?;

    for ordinal in 0..=total_inserts {
        let statement = expected_statement(ordinal, rows, chunk)?;
        writer
            .write_all(&statement)
            .map_err(|error| format!("cannot write workload source: {error}"))?;
        digests.add_statement(&statement)?;
    }
    writer
        .flush()
        .map_err(|error| format!("cannot flush workload source: {error}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|error| format!("cannot sync workload source: {error}"))?;
    drop(writer);

    let (source_sha256, statement_stream_sha256, source_bytes, total_statements) = digests.finish();
    let manifest = WorkloadManifest {
        version: MANIFEST_VERSION,
        workload: WORKLOAD_ID.to_string(),
        rows,
        chunk,
        insert_statements: total_inserts,
        total_statements,
        source_bytes,
        source_sha256,
        statement_stream_sha256,
    };

    let (manifest_temp, manifest_file) = TempArtifact::create(manifest_path)?;
    let mut manifest_writer = BufWriter::new(manifest_file);
    manifest_writer
        .write_all(manifest.serialize().as_bytes())
        .map_err(|error| format!("cannot write workload manifest: {error}"))?;
    manifest_writer
        .flush()
        .map_err(|error| format!("cannot flush workload manifest: {error}"))?;
    manifest_writer
        .get_ref()
        .sync_all()
        .map_err(|error| format!("cannot sync workload manifest: {error}"))?;
    drop(manifest_writer);

    // Publish the source first and the manifest last. An interrupted publication can leave an
    // unreferenced source, but never an accepted manifest naming a missing or partial source.
    source_temp.publish(source_path)?;
    manifest_temp.publish(manifest_path)?;
    Ok(manifest)
}

fn read_manifest(path: &Path) -> Result<WorkloadManifest, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("cannot stat manifest {}: {error}", path.display()))?;
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "manifest {} exceeds the {MAX_MANIFEST_BYTES}-byte bound",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| format!("cannot read manifest {}: {error}", path.display()))?;
    WorkloadManifest::parse(&bytes)
}

fn verify_workload(source_path: &Path, manifest_path: &Path) -> Result<WorkloadManifest, String> {
    let manifest = read_manifest(manifest_path)?;
    if manifest.version != MANIFEST_VERSION {
        return Err(format!(
            "unsupported manifest version {}; expected {MANIFEST_VERSION}",
            manifest.version
        ));
    }
    if manifest.workload != WORKLOAD_ID {
        return Err(format!(
            "unexpected workload {}; expected {WORKLOAD_ID}",
            manifest.workload
        ));
    }
    validate_shape(manifest.rows, manifest.chunk)?;
    let expected_inserts = insert_statement_count(manifest.rows, manifest.chunk)?;
    if manifest.insert_statements != expected_inserts {
        return Err(format!(
            "manifest insert statement count {} != expected {expected_inserts}",
            manifest.insert_statements
        ));
    }
    let expected_total = expected_inserts
        .checked_add(1)
        .ok_or_else(|| "total statement count overflow".to_string())?;
    if manifest.total_statements != expected_total {
        return Err(format!(
            "manifest total statement count {} != expected {expected_total}",
            manifest.total_statements
        ));
    }

    let file = File::open(source_path)
        .map_err(|error| format!("cannot open source {}: {error}", source_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut actual = Vec::new();
    let mut digests = StreamDigests::default();
    for ordinal in 0..expected_total {
        actual.clear();
        let read = reader
            .read_until(b'\n', &mut actual)
            .map_err(|error| format!("cannot read statement {ordinal}: {error}"))?;
        if read == 0 {
            return Err(format!("source is truncated before statement {ordinal}"));
        }
        if !actual.ends_with(b"\n") {
            return Err(format!("statement {ordinal} is missing its final newline"));
        }
        std::str::from_utf8(&actual)
            .map_err(|_| format!("statement {ordinal} is not valid UTF-8"))?;
        let expected = expected_statement(ordinal, manifest.rows, manifest.chunk)?;
        if actual != expected {
            return Err(format!(
                "statement {ordinal} does not match the versioned workload"
            ));
        }
        digests.add_statement(&actual)?;
    }
    actual.clear();
    if reader
        .read_until(b'\n', &mut actual)
        .map_err(|error| format!("cannot check for trailing source bytes: {error}"))?
        != 0
    {
        return Err("source contains extra statements or bytes".to_string());
    }

    let (source_sha256, statement_stream_sha256, source_bytes, statement_count) = digests.finish();
    let comparisons = [
        (
            "source_bytes",
            source_bytes.to_string(),
            manifest.source_bytes.to_string(),
        ),
        (
            "total_statements",
            statement_count.to_string(),
            manifest.total_statements.to_string(),
        ),
        (
            "source_sha256",
            source_sha256,
            manifest.source_sha256.clone(),
        ),
        (
            "statement_stream_sha256",
            statement_stream_sha256,
            manifest.statement_stream_sha256.clone(),
        ),
    ];
    for (field, actual, expected) in comparisons {
        if actual != expected {
            return Err(format!(
                "{field} mismatch: actual={actual} manifest={expected}"
            ));
        }
    }
    Ok(manifest)
}

fn expected_sums(rows: u64) -> Result<(i128, i128), String> {
    if rows == 0 {
        return Ok((0, 0));
    }
    let rows_i128 = i128::from(rows);
    let id_sum = rows_i128
        .checked_mul(rows_i128 - 1)
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| "id sum overflow".to_string())?;
    const PERIOD: u64 = 100_000;
    const PERIOD_SUM: i128 = 4_999_950_000;
    let complete_periods = rows / PERIOD;
    let remainder = rows % PERIOD;
    let mut balance_sum = i128::from(complete_periods)
        .checked_mul(PERIOD_SUM)
        .ok_or_else(|| "balance sum overflow".to_string())?;
    for id in 0..remainder {
        balance_sum = balance_sum
            .checked_add(i128::from((id * 7) % PERIOD))
            .ok_or_else(|| "balance sum overflow".to_string())?;
    }
    Ok((id_sum, balance_sum))
}

#[derive(Debug)]
struct ReplayStats {
    create_roundtrip: Duration,
    insert_roundtrip_sum: Duration,
    insert_roundtrip_p50_us: u128,
    insert_roundtrip_p99_us: u128,
    insert_roundtrip_p999_us: u128,
    insert_roundtrip_max_us: u128,
    insert_load_wall: Duration,
    validation_roundtrip_sum: Duration,
    end_to_end_client_wall: Duration,
}

impl ReplayStats {
    fn rows_per_second(&self, rows: u64) -> f64 {
        let seconds = self.insert_load_wall.as_secs_f64();
        if seconds == 0.0 {
            0.0
        } else {
            rows as f64 / seconds
        }
    }
}

fn nearest_rank(sorted: &[u128], numerator: usize, denominator: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = sorted
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .clamp(1, sorted.len());
    sorted[rank - 1]
}

fn require_command_complete(
    messages: Vec<SimpleQueryMessage>,
    expected_rows: u64,
    ordinal: u64,
) -> Result<(), String> {
    match messages.as_slice() {
        [SimpleQueryMessage::CommandComplete(rows)] if *rows == expected_rows => Ok(()),
        [SimpleQueryMessage::CommandComplete(rows)] => Err(format!(
            "statement {ordinal} affected {rows} rows; expected {expected_rows}"
        )),
        _ => Err(format!(
            "statement {ordinal} returned {} messages instead of one command completion",
            messages.len()
        )),
    }
}

async fn scalar_query_i128(client: &Client, sql: &str) -> Result<i128, String> {
    let messages = client
        .simple_query(sql)
        .await
        .map_err(|error| format!("validation query failed: {error}"))?;
    match messages.as_slice() {
        [SimpleQueryMessage::RowDescription(columns), SimpleQueryMessage::Row(row), SimpleQueryMessage::CommandComplete(1)]
            if columns.len() == 1 =>
        {
            row.get(0)
                .ok_or_else(|| "validation row has no first column".to_string())?
                .parse::<i128>()
                .map_err(|_| "validation value is not an integer".to_string())
        }
        _ => Err(format!(
            "validation query returned {} messages instead of one row and completion",
            messages.len()
        )),
    }
}

fn add_duration(total: &mut Duration, value: Duration) -> Result<(), String> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| "duration accumulation overflow".to_string())?;
    Ok(())
}

async fn replay_workload(
    source_path: &Path,
    manifest: &WorkloadManifest,
    connection: &str,
) -> Result<ReplayStats, String> {
    let end_to_end_started = Instant::now();
    let (client, connection_driver) = tokio_postgres::connect(connection, NoTls)
        .await
        .map_err(|error| format!("database connection failed: {error}"))?;
    let connection_task = tokio::spawn(connection_driver);

    let file = File::open(source_path)
        .map_err(|error| format!("cannot reopen source {}: {error}", source_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut actual = Vec::new();
    let mut digests = StreamDigests::default();
    let mut insert_roundtrips = Vec::with_capacity(
        usize::try_from(manifest.insert_statements)
            .map_err(|_| "insert statement count does not fit usize".to_string())?,
    );
    let mut insert_roundtrip_sum = Duration::ZERO;
    let mut create_roundtrip = Duration::ZERO;
    let mut insert_load_started = None;

    for ordinal in 0..manifest.total_statements {
        actual.clear();
        let read = reader
            .read_until(b'\n', &mut actual)
            .map_err(|error| format!("cannot read statement {ordinal}: {error}"))?;
        if read == 0 || !actual.ends_with(b"\n") {
            return Err(format!(
                "source changed after verification at statement {ordinal}"
            ));
        }
        let expected = expected_statement(ordinal, manifest.rows, manifest.chunk)?;
        if actual != expected {
            return Err(format!(
                "source changed after verification at statement {ordinal}"
            ));
        }
        digests.add_statement(&actual)?;
        let sql = std::str::from_utf8(&actual)
            .map_err(|_| format!("statement {ordinal} is not valid UTF-8"))?;
        if ordinal == 1 {
            insert_load_started = Some(Instant::now());
        }
        let roundtrip_started = Instant::now();
        let messages = client
            .simple_query(sql)
            .await
            .map_err(|error| format!("statement {ordinal} failed: {error}"))?;
        let roundtrip = roundtrip_started.elapsed();
        let expected_rows = if ordinal == 0 {
            0
        } else {
            let first_id = (ordinal - 1)
                .checked_mul(manifest.chunk)
                .ok_or_else(|| "fixture row offset overflow".to_string())?;
            manifest.chunk.min(manifest.rows - first_id)
        };
        require_command_complete(messages, expected_rows, ordinal)?;
        if ordinal == 0 {
            create_roundtrip = roundtrip;
        } else {
            add_duration(&mut insert_roundtrip_sum, roundtrip)?;
            insert_roundtrips.push(roundtrip.as_micros());
        }
    }
    let insert_load_wall = insert_load_started
        .ok_or_else(|| "workload contained no INSERT statement".to_string())?
        .elapsed();
    actual.clear();
    if reader
        .read_until(b'\n', &mut actual)
        .map_err(|error| format!("cannot check source tail after replay: {error}"))?
        != 0
    {
        return Err("source grew after verification".to_string());
    }
    let (source_sha256, statement_stream_sha256, source_bytes, statement_count) = digests.finish();
    if source_sha256 != manifest.source_sha256
        || statement_stream_sha256 != manifest.statement_stream_sha256
        || source_bytes != manifest.source_bytes
        || statement_count != manifest.total_statements
    {
        return Err("replayed source identity does not match the manifest".to_string());
    }

    let validation_started = Instant::now();
    let actual_count = scalar_query_i128(&client, "SELECT COUNT(*) FROM accounts;\n").await?;
    let actual_id_sum = scalar_query_i128(&client, "SELECT SUM(id) FROM accounts;\n").await?;
    let actual_balance_sum =
        scalar_query_i128(&client, "SELECT SUM(balance) FROM accounts;\n").await?;
    let validation_roundtrip_sum = validation_started.elapsed();
    let (expected_id_sum, expected_balance_sum) = expected_sums(manifest.rows)?;
    let expected_count = i128::from(manifest.rows);
    if (actual_count, actual_id_sum, actual_balance_sum)
        != (expected_count, expected_id_sum, expected_balance_sum)
    {
        return Err(format!(
            "validation mismatch: actual=({actual_count},{actual_id_sum},{actual_balance_sum}) \
             expected=({expected_count},{expected_id_sum},{expected_balance_sum})"
        ));
    }

    let end_to_end_client_wall = end_to_end_started.elapsed();
    drop(client);
    connection_task.abort();
    let _ = connection_task.await;

    insert_roundtrips.sort_unstable();
    Ok(ReplayStats {
        create_roundtrip,
        insert_roundtrip_sum,
        insert_roundtrip_p50_us: nearest_rank(&insert_roundtrips, 50, 100),
        insert_roundtrip_p99_us: nearest_rank(&insert_roundtrips, 99, 100),
        insert_roundtrip_p999_us: nearest_rank(&insert_roundtrips, 999, 1_000),
        insert_roundtrip_max_us: insert_roundtrips.last().copied().unwrap_or(0),
        insert_load_wall,
        validation_roundtrip_sum,
        end_to_end_client_wall,
    })
}

fn validate_label(value: &str, flag: &str) -> Result<(), String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(usage(&format!(
            "{flag} must contain only ASCII letters, digits, '.', '_', or '-'"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct Cli {
    command: String,
    rows: u64,
    chunk: u64,
    source: PathBuf,
    manifest: PathBuf,
    connection: Option<String>,
    backend: Option<String>,
    profile: Option<String>,
    trial: Option<u64>,
}

fn parse_cli() -> Result<Cli, String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| usage("missing command"))?;
    if !matches!(command.as_str(), "generate" | "verify" | "run") {
        return Err(usage(&format!("unknown command {command:?}")));
    }
    let mut rows = DEFAULT_ROWS;
    let mut chunk = DEFAULT_CHUNK;
    let mut rows_set = false;
    let mut chunk_set = false;
    let mut source = None;
    let mut manifest = None;
    let mut connection = None;
    let mut backend = None;
    let mut profile = None;
    let mut trial = None;
    let mut seen_flags = BTreeSet::new();
    while let Some(flag) = args.next() {
        if !seen_flags.insert(flag.clone()) {
            return Err(usage(&format!("duplicate flag {flag:?}")));
        }
        let value = args
            .next()
            .ok_or_else(|| usage(&format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--rows" => {
                rows = parse_canonical_u64(&value, "--rows").map_err(|error| usage(&error))?;
                rows_set = true;
            }
            "--chunk" => {
                chunk = parse_canonical_u64(&value, "--chunk").map_err(|error| usage(&error))?;
                chunk_set = true;
            }
            "--source" => source = Some(PathBuf::from(value)),
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--connection" => connection = Some(value),
            "--backend" => {
                validate_label(&value, "--backend")?;
                backend = Some(value);
            }
            "--profile" => {
                validate_label(&value, "--profile")?;
                profile = Some(value);
            }
            "--trial" => {
                trial = Some(parse_canonical_u64(&value, "--trial").map_err(|error| usage(&error))?)
            }
            _ => return Err(usage(&format!("unknown flag {flag:?}"))),
        }
    }
    let source = source.ok_or_else(|| usage("--source is required"))?;
    let manifest = manifest.ok_or_else(|| usage("--manifest is required"))?;
    match command.as_str() {
        "generate" => {
            if connection.is_some() || backend.is_some() || profile.is_some() || trial.is_some() {
                return Err(usage("generate does not accept replay-only flags"));
            }
        }
        "verify" => {
            if rows_set
                || chunk_set
                || connection.is_some()
                || backend.is_some()
                || profile.is_some()
                || trial.is_some()
            {
                return Err(usage("verify accepts only --source and --manifest"));
            }
        }
        "run" => {
            if rows_set || chunk_set {
                return Err(usage(
                    "run derives rows and chunk from the verified manifest",
                ));
            }
            if trial == Some(0) {
                return Err(usage("--trial must be greater than zero"));
            }
        }
        _ => unreachable!("command was validated above"),
    }
    Ok(Cli {
        command,
        rows,
        chunk,
        source,
        manifest,
        connection,
        backend,
        profile,
        trial,
    })
}

fn usage(reason: &str) -> String {
    format!(
        "{reason}\nusage: insert_workload_client generate \
         [--rows N] [--chunk N] --source PATH --manifest PATH\n       \
         insert_workload_client verify --source PATH --manifest PATH\n       \
         insert_workload_client run --source PATH --manifest PATH \
         --connection CONN --backend LABEL --profile LABEL --trial N"
    )
}

fn run() -> Result<(), String> {
    let cli = parse_cli()?;
    let started = Instant::now();
    let manifest = match cli.command.as_str() {
        "generate" => generate_workload(cli.rows, cli.chunk, &cli.source, &cli.manifest)?,
        "verify" => verify_workload(&cli.source, &cli.manifest)?,
        "run" => {
            let verification_started = Instant::now();
            let manifest = verify_workload(&cli.source, &cli.manifest)?;
            let verification_wall = verification_started.elapsed();
            let connection = cli
                .connection
                .as_deref()
                .ok_or_else(|| usage("--connection is required for run"))?;
            let backend = cli
                .backend
                .as_deref()
                .ok_or_else(|| usage("--backend is required for run"))?;
            let profile = cli
                .profile
                .as_deref()
                .ok_or_else(|| usage("--profile is required for run"))?;
            let trial = cli
                .trial
                .ok_or_else(|| usage("--trial is required for run"))?;
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|error| format!("cannot build client runtime: {error}"))?;
            let stats = runtime.block_on(replay_workload(&cli.source, &manifest, connection))?;
            println!(
                "insert_workload_replay_status=complete backend={} profile={} trial={} \
                 manifest_version={} workload={} rows={} chunk={} insert_statements={} \
                 source_bytes={} source_sha256={} statement_stream_sha256={} \
                 source_verification_ms={} create_roundtrip_us={} \
                 insert_roundtrip_sum_us={} insert_roundtrip_p50_us={} \
                 insert_roundtrip_p99_us={} insert_roundtrip_p999_us={} \
                 insert_roundtrip_max_us={} insert_load_wall_ms={} \
                 validation_roundtrip_sum_us={} end_to_end_client_wall_ms={} \
                 rows_per_second={:.3} validation=count_id_sum_balance_sum",
                backend,
                profile,
                trial,
                manifest.version,
                manifest.workload,
                manifest.rows,
                manifest.chunk,
                manifest.insert_statements,
                manifest.source_bytes,
                manifest.source_sha256,
                manifest.statement_stream_sha256,
                verification_wall.as_millis(),
                stats.create_roundtrip.as_micros(),
                stats.insert_roundtrip_sum.as_micros(),
                stats.insert_roundtrip_p50_us,
                stats.insert_roundtrip_p99_us,
                stats.insert_roundtrip_p999_us,
                stats.insert_roundtrip_max_us,
                stats.insert_load_wall.as_millis(),
                stats.validation_roundtrip_sum.as_micros(),
                stats.end_to_end_client_wall.as_millis(),
                stats.rows_per_second(manifest.rows),
            );
            return Ok(());
        }
        _ => unreachable!("CLI parser accepted only known commands"),
    };
    let (id_sum, balance_sum) = expected_sums(manifest.rows)?;
    println!(
        "insert_workload_status=complete operation={} manifest_version={} workload={} \
         rows={} chunk={} insert_statements={} total_statements={} source_bytes={} \
         source_sha256={} statement_stream_sha256={} id_sum={} balance_sum={} wall_ms={}",
        cli.command,
        manifest.version,
        manifest.workload,
        manifest.rows,
        manifest.chunk,
        manifest.insert_statements,
        manifest.total_statements,
        manifest.source_bytes,
        manifest.source_sha256,
        manifest.statement_stream_sha256,
        id_sum,
        balance_sum,
        started.elapsed().as_millis(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    if let Err(error) = run() {
        eprintln!(
            "insert_workload_status=failed error={}",
            error.replace('\n', " | ")
        );
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let serial = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "gpu-db-insert-workload-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn paths(&self) -> (PathBuf, PathBuf) {
            (self.0.join("source.sql"), self.0.join("source.manifest"))
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(rows: u64, chunk: u64) -> (TestDir, PathBuf, PathBuf) {
        let dir = TestDir::new();
        let (source, manifest) = dir.paths();
        generate_workload(rows, chunk, &source, &manifest).unwrap();
        (dir, source, manifest)
    }

    #[test]
    fn exact_small_source_and_manifest_round_trip() {
        let (_dir, source, manifest_path) = fixture(3, 2);
        assert_eq!(
            fs::read_to_string(&source).unwrap(),
            "CREATE TABLE accounts (id int4, balance int4);\n\
             INSERT INTO accounts (id, balance) VALUES (0, 0),(1, 7);\n\
             INSERT INTO accounts (id, balance) VALUES (2, 14);\n"
        );
        let manifest = verify_workload(&source, &manifest_path).unwrap();
        assert_eq!(manifest.rows, 3);
        assert_eq!(manifest.chunk, 2);
        assert_eq!(manifest.insert_statements, 2);
        assert_eq!(manifest.total_statements, 3);
        assert_eq!(
            WorkloadManifest::parse(manifest.serialize().as_bytes()).unwrap(),
            manifest
        );
    }

    #[test]
    fn nondivisible_chunk_and_closed_form_sums_are_exact() {
        let (_dir, source, manifest_path) = fixture(5, 2);
        let manifest = verify_workload(&source, &manifest_path).unwrap();
        assert_eq!(manifest.insert_statements, 3);
        assert!(fs::read_to_string(source)
            .unwrap()
            .ends_with("INSERT INTO accounts (id, balance) VALUES (4, 28);\n"));
        assert_eq!(expected_sums(5).unwrap(), (10, 70));
        assert_eq!(expected_sums(100_000).unwrap().1, 4_999_950_000);
        assert_eq!(expected_sums(48_000_000).unwrap().0, 1_151_999_976_000_000);
    }

    #[test]
    fn tampered_source_is_rejected() {
        let (_dir, source, manifest) = fixture(3, 2);
        let mut bytes = fs::read(&source).unwrap();
        let needle = bytes.iter().position(|byte| *byte == b'7').unwrap();
        bytes[needle] = b'8';
        fs::write(&source, bytes).unwrap();
        assert!(verify_workload(&source, &manifest)
            .unwrap_err()
            .contains("does not match"));
    }

    #[test]
    fn truncated_extra_and_boundary_invalid_sources_are_rejected() {
        for mutation in ["truncate", "extra", "boundary"] {
            let (_dir, source, manifest) = fixture(3, 2);
            let mut bytes = fs::read(&source).unwrap();
            match mutation {
                "truncate" => {
                    bytes.pop();
                }
                "extra" => bytes.extend_from_slice(b"SELECT 1;\n"),
                "boundary" => {
                    let newline = bytes.iter().position(|byte| *byte == b'\n').unwrap();
                    bytes.remove(newline);
                }
                _ => unreachable!(),
            }
            fs::write(&source, bytes).unwrap();
            assert!(
                verify_workload(&source, &manifest).is_err(),
                "{mutation} source was accepted"
            );
        }
    }

    #[test]
    fn invalid_utf8_and_malformed_manifest_are_rejected() {
        let (_dir, source, manifest) = fixture(3, 2);
        let mut bytes = fs::read(&source).unwrap();
        bytes[0] = 0xff;
        fs::write(&source, bytes).unwrap();
        assert!(verify_workload(&source, &manifest).is_err());

        let (_dir, source, manifest) = fixture(3, 2);
        let mut text = fs::read_to_string(&manifest).unwrap();
        text.push_str("rows=3\n");
        fs::write(&manifest, text).unwrap();
        assert!(verify_workload(&source, &manifest)
            .unwrap_err()
            .contains("duplicate manifest field"));
    }

    #[test]
    fn existing_targets_and_zero_shapes_fail_closed() {
        let dir = TestDir::new();
        let (source, manifest) = dir.paths();
        assert!(generate_workload(0, 2, &source, &manifest).is_err());
        assert!(generate_workload(2, 0, &source, &manifest).is_err());
        generate_workload(2, 2, &source, &manifest).unwrap();
        assert!(generate_workload(2, 2, &source, &manifest)
            .unwrap_err()
            .contains("refusing to overwrite"));
    }

    #[test]
    fn numeric_manifest_fields_are_canonical() {
        assert_eq!(parse_canonical_u64("0", "value").unwrap(), 0);
        assert_eq!(
            parse_canonical_u64("48000000", "value").unwrap(),
            48_000_000
        );
        for invalid in ["", "00", "01", "+1", "-1", "1 ", " 1", "1_000"] {
            assert!(
                parse_canonical_u64(invalid, "value").is_err(),
                "{invalid:?} was accepted"
            );
        }
    }
}
