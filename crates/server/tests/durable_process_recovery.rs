//! Process-boundary durability proof for the canonical pgwire server.
//!
//! The test deliberately kills, rather than gracefully stops, each writer process after the
//! client has observed its acknowledgement.  The next process must rebuild from exactly that
//! durable WAL prefix and keep appending through the same sole WAL/publication owner.

use std::fs;
use std::io;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use chrono::{NaiveDate, NaiveDateTime};
use futures_util::{stream, SinkExt};
use sha2::{Digest, Sha256};
use tokio::task::JoinHandle;
use tokio_postgres::types::{IsNull, ToSql, Type};
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};
use uuid::Uuid;

static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(1);

/// Small client-bound NUMERIC binary wrapper. The product codec, not this test helper, owns the
/// PostgreSQL representation; the wrapper makes the real W1 Bind carry a finite binary numeric.
#[derive(Debug, Clone)]
struct PgNumeric(Vec<u8>);

impl PgNumeric {
    fn fixed_12345_6700() -> Self {
        Self(vec![
            0, 3, 0, 1, 0, 0, 0, 4, // ndigits, weight, positive sign, display scale
            0, 1, 0x09, 0x29, 0x1a, 0x2c,
        ])
    }
}

impl ToSql for PgNumeric {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut tokio_postgres::types::private::BytesMut,
    ) -> Result<IsNull, Box<dyn std::error::Error + Sync + Send>> {
        if *ty != Type::NUMERIC {
            return Err(Box::new(tokio_postgres::types::WrongType::new::<Self>(
                ty.clone(),
            )));
        }
        out.extend_from_slice(&self.0);
        Ok(IsNull::No)
    }

    fn accepts(ty: &Type) -> bool {
        *ty == Type::NUMERIC
    }

    tokio_postgres::types::to_sql_checked!();
}

struct DurableFixture {
    directory: PathBuf,
    wal: PathBuf,
}

impl DurableFixture {
    fn new() -> io::Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "gpu-db-server-process-recovery-{}-{nonce}-{id}",
            std::process::id()
        ));
        fs::create_dir(&directory)?;
        Ok(Self {
            wal: directory.join("canonical.wal"),
            directory,
        })
    }
}

impl Drop for DurableFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

struct ServerProcess {
    child: Child,
    port: u16,
}

impl ServerProcess {
    fn start(wal: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let port = free_local_port()?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_gpu-db-engine-server"))
            .arg(format!("127.0.0.1:{port}"))
            .env("GPU_DB_WAL_SEGMENT", wal)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        let deadline = Instant::now() + Duration::from_secs(15);
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Ok(Self { child, port });
            }
            if let Some(status) = child.try_wait()? {
                return Err(format!(
                    "gpu-db-engine-server exited before listening on 127.0.0.1:{port}: {status}"
                )
                .into());
            }
            thread::sleep(Duration::from_millis(25));
        }

        let _ = child.kill();
        let _ = child.wait();
        Err(format!("gpu-db-engine-server did not listen on 127.0.0.1:{port}").into())
    }

    fn kill_after_ack(&mut self) -> io::Result<()> {
        if self.child.try_wait()?.is_none() {
            // On Unix, `Child::kill` is SIGKILL.  This deliberately bypasses every graceful
            // shutdown path, so only the WAL state acknowledged to the client may survive.
            self.child.kill()?;
        }
        let _ = self.child.wait()?;
        Ok(())
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.kill_after_ack();
    }
}

fn free_local_port() -> io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    listener.local_addr().map(|address| address.port())
}

async fn connect(port: u16) -> Result<(Client, JoinHandle<()>), tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await?;
    let connection = tokio::spawn(async move {
        // A SIGKILL intentionally closes this task's socket.  The test observes all query
        // acknowledgements directly, so that expected transport error must not hide a failure.
        let _ = connection.await;
    });
    Ok((client, connection))
}

async fn close_client(client: Client, connection: JoinHandle<()>) {
    drop(client);
    connection.abort();
    let _ = connection.await;
}

#[derive(Debug, PartialEq, Eq)]
struct DurableSnapshot {
    rows: Vec<Vec<Option<String>>>,
    transaction_rows: Vec<[String; 2]>,
    catalog_identities: Vec<[String; 3]>,
    numeric_catalog_identity: [String; 3],
    numeric_type_oid: u32,
    numeric_typmod: i32,
    sequence_state: [String; 2],
    digest: [u8; 32],
}

async fn durable_snapshot(client: &Client) -> Result<DurableSnapshot, tokio_postgres::Error> {
    let messages = client
        .simple_query(
            "SELECT id, parent_id, seq_value, i2, i4, i8, amount, flag, note, day, created_at, ident \
             FROM process_restart_rows ORDER BY id",
        )
        .await?;
    let rows = messages
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some(
                [
                    "id",
                    "parent_id",
                    "seq_value",
                    "i2",
                    "i4",
                    "i8",
                    "amount",
                    "flag",
                    "note",
                    "day",
                    "created_at",
                    "ident",
                ]
                .into_iter()
                .map(|name| row.get(name).map(str::to_owned))
                .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect::<Vec<_>>();

    let transaction_rows = client
        .simple_query("SELECT id, note FROM process_restart_txn ORDER BY id")
        .await?
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some([row.get("id")?.to_string(), row.get("note")?.to_string()])
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let catalog_identities = client
        .simple_query(
            "SELECT c.relname, c.oid, c.relkind \
             FROM pg_catalog.pg_class c \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE n.nspname = 'public' \
               AND c.relname IN (\
                   'process_restart_parent', 'process_restart_rows', \
                   'process_restart_seq', 'process_restart_txn'\
               ) \
             ORDER BY c.relname",
        )
        .await?
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some([
                row.get("relname")?.to_string(),
                row.get("oid")?.to_string(),
                row.get("relkind")?.to_string(),
            ]),
            _ => None,
        })
        .collect::<Vec<_>>();

    let numeric = client
        .prepare("SELECT amount FROM process_restart_rows ORDER BY id")
        .await?;
    let numeric = numeric
        .columns()
        .first()
        .expect("prepared numeric projection has one result column");
    let numeric_type_oid = numeric.type_().oid();
    let numeric_typmod = numeric.type_modifier();

    // The server's base-column RowDescription intentionally has no table/attribute origin yet;
    // the canonical GPU catalog relation remains the authoritative source for those identities.
    let numeric_catalog_identity = client
        .simple_query(
            "SELECT a.attrelid, a.attnum, a.atttypmod \
             FROM pg_catalog.pg_attribute a \
             JOIN pg_catalog.pg_class c ON c.oid = a.attrelid \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE n.nspname = 'public' AND c.relname = 'process_restart_rows' \
               AND a.attname = 'amount'",
        )
        .await?
        .into_iter()
        .find_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some([
                row.get("attrelid")?.to_string(),
                row.get("attnum")?.to_string(),
                row.get("atttypmod")?.to_string(),
            ]),
            _ => None,
        })
        .expect("GPU catalog returns the numeric column identity");

    let sequence_state = client
        .simple_query("SELECT last_value, is_called FROM public.process_restart_seq")
        .await?
        .into_iter()
        .find_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some([
                row.get("last_value")?.to_string(),
                row.get("is_called")?.to_string(),
            ]),
            _ => None,
        })
        .expect("sequence state returns exactly one row");

    let mut hasher = Sha256::new();
    for row in &rows {
        for field in row {
            match field {
                Some(field) => {
                    hasher.update([1]);
                    hasher.update((field.len() as u64).to_be_bytes());
                    hasher.update(field.as_bytes());
                }
                None => hasher.update([0]),
            }
        }
    }
    for row in &transaction_rows {
        for field in row {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field.as_bytes());
        }
    }
    for identity in &catalog_identities {
        for field in identity {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field.as_bytes());
        }
    }
    for field in &numeric_catalog_identity {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher.update(numeric_type_oid.to_be_bytes());
    hasher.update(numeric_typmod.to_be_bytes());
    for field in &sequence_state {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    let digest = hasher.finalize();
    let mut exact_digest = [0; 32];
    exact_digest.copy_from_slice(&digest);
    Ok(DurableSnapshot {
        rows,
        transaction_rows,
        catalog_identities,
        numeric_catalog_identity,
        numeric_type_oid,
        numeric_typmod,
        sequence_state,
        digest: exact_digest,
    })
}

fn assert_no_retired_lane_artifacts(wal: &Path) -> io::Result<()> {
    let parent = wal.parent().expect("fixture WAL has a parent directory");
    let stem = wal
        .file_name()
        .expect("fixture WAL has a file name")
        .to_string_lossy();
    let prefix = format!("{stem}.lane-");
    let artifacts = fs::read_dir(parent)?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with(&prefix))
        .collect::<Vec<_>>();
    assert!(
        artifacts.is_empty(),
        "canonical traffic must not create retired physical lane WAL artifacts: {artifacts:?}"
    );
    Ok(())
}

#[tokio::test]
async fn acknowledged_process_writes_survive_sigkill_restart_and_continue_appending(
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = DurableFixture::new()?;

    let mut first = ServerProcess::start(&fixture.wal)?;
    let (client, connection) = connect(first.port).await?;
    client
        .batch_execute("CREATE SEQUENCE process_restart_seq")
        .await?;
    client
        .batch_execute("CREATE TABLE process_restart_parent (id INT PRIMARY KEY)")
        .await?;
    client
        .batch_execute(
            "CREATE TABLE process_restart_rows (\
                 id INT PRIMARY KEY,\
                 parent_id INT,\
                 seq_value INT DEFAULT nextval('process_restart_seq'::regclass),\
                 i2 SMALLINT,\
                 i4 INT,\
                 i8 BIGINT,\
                 amount NUMERIC(12,4),\
                 flag BOOL,\
                 note TEXT,\
                 day DATE,\
                 created_at TIMESTAMP,\
                 ident UUID,\
                 CONSTRAINT process_restart_positive CHECK (amount > 0.00)\
             )",
        )
        .await?;
    client
        .batch_execute(
            "ALTER TABLE ONLY process_restart_rows \
             ADD CONSTRAINT process_restart_parent_fk \
             FOREIGN KEY (parent_id) REFERENCES process_restart_parent(id)",
        )
        .await?;
    client
        .batch_execute("INSERT INTO process_restart_parent VALUES (1)")
        .await?;

    // W1: an extended Parse/Bind/Execute uses the actual pgwire binary codecs for every logical
    // type, then repeats the same prepared route with a typed NULL per nullable column.
    let day = NaiveDate::from_ymd_opt(2024, 1, 15).expect("fixed date");
    let created_at = day
        .and_hms_micro_opt(10, 30, 0, 123_456)
        .expect("fixed timestamp");
    let ident = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")?;
    let numeric = PgNumeric::fixed_12345_6700();
    let prepared = client
        .prepare(
            "INSERT INTO process_restart_rows \
             (id, parent_id, i2, i4, i8, amount, flag, note, day, created_at, ident) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
        )
        .await?;
    assert_eq!(
        client
            .execute(
                &prepared,
                &[
                    &1_i32,
                    &1_i32,
                    &-7_i16,
                    &42_i32,
                    &-9_000_000_000_i64,
                    &numeric,
                    &true,
                    &"Grüße",
                    &day,
                    &created_at,
                    &ident,
                ],
            )
            .await?,
        1,
        "all-types prepared write acknowledgement is durable"
    );
    assert_eq!(
        client
            .execute(
                &prepared,
                &[
                    &2_i32,
                    &1_i32,
                    &Option::<i16>::None,
                    &Option::<i32>::None,
                    &Option::<i64>::None,
                    &Option::<PgNumeric>::None,
                    &Option::<bool>::None,
                    &Option::<String>::None,
                    &Option::<NaiveDate>::None,
                    &Option::<NaiveDateTime>::None,
                    &Option::<Uuid>::None,
                ],
            )
            .await?,
        1,
        "typed NULL Bind acknowledgement is durable"
    );

    // General/autocommit is intentionally distinct from W1 and exercises the same canonical
    // publication owner with text SQL.
    client
        .batch_execute(
            "INSERT INTO process_restart_rows \
             (id, parent_id, i2, i4, i8, amount, flag, note, day, created_at, ident) \
             VALUES (3, 1, NULL, NULL, NULL, 1.0000, NULL, 'simple', NULL, NULL, NULL)",
        )
        .await?;
    client
        .batch_execute("UPDATE process_restart_rows SET note = 'general-updated' WHERE id = 3")
        .await?;

    // One explicit user transaction contains DDL and DML.  Its COMMIT is a separate durable
    // boundary from the prior autocommit/W1 statements.
    client.batch_execute("BEGIN").await?;
    client
        .batch_execute("CREATE TABLE process_restart_txn (id INT PRIMARY KEY, note TEXT)")
        .await?;
    client
        .batch_execute("INSERT INTO process_restart_txn VALUES (1, 'transactional ddl+dml')")
        .await?;
    client.batch_execute("COMMIT").await?;

    let copy = client
        .copy_in(
            "COPY process_restart_rows \
             (id, parent_id, i2, i4, i8, amount, flag, note, day, created_at, ident) \
             FROM STDIN WITH CSV",
        )
        .await?;
    let mut input = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"4,1,4,4,4,2.5000,f,Copied,2024-01-16,2024-01-16 00:00:00,550e8400-e29b-41d4-a716-446655440004\n",
    ))]);
    futures_util::pin_mut!(copy);
    copy.send_all(&mut input).await?;
    assert_eq!(copy.finish().await?, 1, "COPY acknowledgement is durable");

    // A rejected COPY batch cannot leak its valid first row across the process boundary.
    let rejected = client
        .copy_in(
            "COPY process_restart_rows \
             (id, parent_id, seq_value, i2, i4, i8, amount, flag, note, day, created_at, ident) \
             FROM STDIN WITH CSV",
        )
        .await?;
    let mut rejected_input = stream::iter(vec![Ok::<_, tokio_postgres::Error>(
        Bytes::from_static(
            b"7,1,700,7,7,7,40.0000,t,MustNotPublish,2024-01-17,2024-01-17 00:00:00,550e8400-e29b-41d4-a716-446655440007\n\
              8,1,800,8,8,8,-1.0000,f,Rejected,2024-01-18,2024-01-18 00:00:00,550e8400-e29b-41d4-a716-446655440008\n",
        ),
    )]);
    futures_util::pin_mut!(rejected);
    rejected.send_all(&mut rejected_input).await?;
    let rejected = rejected
        .finish()
        .await
        .expect_err("CHECK-violating COPY must reject its entire batch");
    assert_eq!(rejected.code().map(|code| code.code()), Some("23514"));

    // ADR-014 ordinary sequence transitions are separately durable: the nextval acknowledgement
    // survives even when the enclosing user's row is rolled back.  Give that row an explicit
    // sequence value so this proof observes exactly one ordinary transition.
    client.batch_execute("BEGIN").await?;
    let transition = client
        .query_one("SELECT nextval('process_restart_seq'::regclass)", &[])
        .await?;
    assert_eq!(transition.get::<_, i64>(0), 5);
    client
        .batch_execute(
            "INSERT INTO process_restart_rows \
             (id, parent_id, seq_value, i2, i4, i8, amount, flag, note, day, created_at, ident) \
             VALUES (5, 1, 500, NULL, NULL, NULL, 5.0000, NULL, 'must roll back', NULL, NULL, NULL)",
        )
        .await?;
    client.batch_execute("ROLLBACK").await?;

    let before_kill = durable_snapshot(&client).await?;
    let expected_row = |fields: [Option<&str>; 12]| {
        fields
            .into_iter()
            .map(|field| field.map(str::to_owned))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        before_kill.rows,
        vec![
            expected_row([
                Some("1"),
                Some("1"),
                Some("1"),
                Some("-7"),
                Some("42"),
                Some("-9000000000"),
                Some("12345.6700"),
                Some("t"),
                Some("Grüße"),
                Some("2024-01-15"),
                Some("2024-01-15 10:30:00.123456"),
                Some("550e8400-e29b-41d4-a716-446655440000"),
            ]),
            expected_row([
                Some("2"),
                Some("1"),
                Some("2"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]),
            expected_row([
                Some("3"),
                Some("1"),
                Some("3"),
                None,
                None,
                None,
                Some("1.0000"),
                None,
                Some("general-updated"),
                None,
                None,
                None,
            ]),
            expected_row([
                Some("4"),
                Some("1"),
                Some("4"),
                Some("4"),
                Some("4"),
                Some("4"),
                Some("2.5000"),
                Some("f"),
                Some("Copied"),
                Some("2024-01-16"),
                Some("2024-01-16 00:00:00"),
                Some("550e8400-e29b-41d4-a716-446655440004"),
            ]),
        ],
        "the successful Query/Bind/Copy acknowledgements define the exact pre-kill state"
    );
    assert_eq!(before_kill.numeric_type_oid, 1_700);
    assert_eq!(before_kill.numeric_typmod, 786_440);
    assert_eq!(
        before_kill.sequence_state,
        ["5".to_string(), "t".to_string()]
    );
    let table_identity = before_kill
        .catalog_identities
        .iter()
        .find(|identity| identity[0] == "process_restart_rows")
        .expect("catalog identity includes the durable child relation");
    assert_eq!(
        before_kill.transaction_rows,
        vec![["1".to_string(), "transactional ddl+dml".to_string()]],
        "the explicit transaction's DDL and DML recover together"
    );
    assert_eq!(before_kill.numeric_catalog_identity[0], table_identity[1]);
    assert_eq!(before_kill.numeric_catalog_identity[1], "7");
    assert_eq!(before_kill.numeric_catalog_identity[2], "786440");
    assert_eq!(
        before_kill.numeric_catalog_identity[2]
            .parse::<i32>()
            .expect("catalog numeric typmod is int4"),
        before_kill.numeric_typmod,
        "GPU pg_attribute and RowDescription retain the same numeric typmod"
    );
    assert_eq!(before_kill.catalog_identities.len(), 4);
    assert_no_retired_lane_artifacts(&fixture.wal)?;
    close_client(client, connection).await;
    first.kill_after_ack()?;

    let mut reopened = ServerProcess::start(&fixture.wal)?;
    let (client, connection) = connect(reopened.port).await?;
    let after_first_reopen = durable_snapshot(&client).await?;
    assert_eq!(
        after_first_reopen, before_kill,
        "reopen must preserve the exact digest"
    );
    assert_no_retired_lane_artifacts(&fixture.wal)?;

    let post_reopen = client
        .prepare(
            "INSERT INTO process_restart_rows \
             (id, parent_id, i2, i4, i8, amount, flag, note, day, created_at, ident) \
             VALUES ($1, 1, NULL, NULL, NULL, 60.0000, NULL, $2, NULL, NULL, NULL)",
        )
        .await?;
    assert_eq!(
        client
            .execute(&post_reopen, &[&6_i32, &"Katherine"])
            .await?,
        1
    );
    let after_append = durable_snapshot(&client).await?;
    assert_eq!(
        after_append.rows.last(),
        Some(&vec![
            Some("6".to_string()),
            Some("1".to_string()),
            Some("6".to_string()),
            None,
            None,
            None,
            Some("60.0000".to_string()),
            None,
            Some("Katherine".to_string()),
            None,
            None,
            None,
        ])
    );
    close_client(client, connection).await;
    reopened.kill_after_ack()?;

    let mut second_reopen = ServerProcess::start(&fixture.wal)?;
    let (client, connection) = connect(second_reopen.port).await?;
    assert_eq!(
        durable_snapshot(&client).await?,
        after_append,
        "a post-reopen acknowledged append must survive a second SIGKILL/reopen"
    );
    assert_no_retired_lane_artifacts(&fixture.wal)?;
    close_client(client, connection).await;
    second_reopen.kill_after_ack()?;
    Ok(())
}
