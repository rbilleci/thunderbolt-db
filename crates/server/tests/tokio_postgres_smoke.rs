use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use chrono::{NaiveDate, NaiveDateTime};
use futures_util::{pin_mut, stream, SinkExt, TryStreamExt};
use tokio_postgres::types::{FromSql, IsNull, ToSql, Type};
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};
use uuid::Uuid;

/// The Rust client deliberately has no built-in arbitrary-precision NUMERIC type.  Keep that
/// limitation at the client boundary by using one small binary-format wrapper over the canonical
/// finite value used by this smoke; the server still sees a real NUMERIC Bind/Result payload.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PgNumeric(Vec<u8>);

impl PgNumeric {
    fn fixed_12345_6700() -> Self {
        Self(vec![
            0, 3, 0, 1, 0, 0, 0, 4, // ndigits, weight, positive sign, display scale
            0, 1, 0x09, 0x29, 0x1a, 0x2c,
        ])
    }

    fn fixed_1_00() -> Self {
        Self(vec![0, 1, 0, 0, 0, 0, 0, 2, 0, 1])
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

impl<'a> FromSql<'a> for PgNumeric {
    fn from_sql(
        ty: &Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        if *ty != Type::NUMERIC || raw.len() < 8 {
            return Err(Box::new(tokio_postgres::types::WrongType::new::<Self>(
                ty.clone(),
            )));
        }
        let ndigits = usize::from(u16::from_be_bytes([raw[0], raw[1]]));
        let expected_len = 8 + ndigits * 2;
        if raw.len() != expected_len {
            return Err("malformed PostgreSQL binary numeric payload".into());
        }
        Ok(Self(raw.to_vec()))
    }

    fn accepts(ty: &Type) -> bool {
        *ty == Type::NUMERIC
    }
}

struct ServerGuard {
    child: Child,
    port: u16,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_local_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn start_server() -> ServerGuard {
    let port = free_local_port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_gpu-db-engine-server"))
        .arg(format!("127.0.0.1:{port}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn gpu-db-engine-server");

    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return ServerGuard { child, port };
        }
        thread::sleep(Duration::from_millis(25));
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("gpu-db-engine-server did not start listening on 127.0.0.1:{port}");
}

async fn connect(port: u16) -> Result<Client, tokio_postgres::Error> {
    let config = format!(
        "host=127.0.0.1 port={port} user=postgres dbname=postgres application_name=tokio_postgres_smoke"
    );
    let (client, connection) = tokio_postgres::connect(&config, NoTls).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            panic!("tokio-postgres connection task failed: {error}");
        }
    });
    Ok(client)
}

async fn require_failed_transaction_rollback(
    client: &Client,
    violation_sql: &str,
    expected_sqlstate: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    client.batch_execute("BEGIN").await?;
    let violation = client
        .batch_execute(violation_sql)
        .await
        .expect_err("constraint violation must fail its explicit transaction");
    assert_eq!(
        violation.code().map(|code| code.code()),
        Some(expected_sqlstate),
        "violation must preserve its PostgreSQL SQLSTATE"
    );
    let blocked = client
        .query_one("SELECT 1", &[])
        .await
        .expect_err("a constraint failure must abort the explicit transaction");
    assert_eq!(blocked.code().map(|code| code.code()), Some("25P02"));
    client.batch_execute("ROLLBACK").await?;
    assert_eq!(client.query_one("SELECT 1", &[]).await?.get::<_, i32>(0), 1);
    Ok(())
}

#[tokio::test]
async fn canonical_server_tokio_postgres_copy_and_recovery_smoke(
) -> Result<(), Box<dyn std::error::Error>> {
    let server = start_server();
    let mut client = connect(server.port).await?;

    let simple = client.simple_query("SELECT 1 AS one").await?;
    let row = simple
        .iter()
        .find_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some(row),
            _ => None,
        })
        .expect("simple query row");
    assert_eq!(row.get("one"), Some("1"));

    let isolation = client
        .query_one("SHOW TRANSACTION ISOLATION LEVEL", &[])
        .await?;
    assert_eq!(isolation.get::<_, String>(0), "read committed");

    client
        .batch_execute(
            "CREATE TABLE driver_people (id INT PRIMARY KEY, name TEXT);
             INSERT INTO driver_people (id, name)
             VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');",
        )
        .await?;

    client
        .batch_execute(
            "PREPARE dump_lookup(pg_catalog.int4) AS \
             SELECT id, name FROM driver_people WHERE id = $1",
        )
        .await?;
    let executed = client.simple_query("EXECUTE dump_lookup(2)").await?;
    let executed = executed
        .iter()
        .find_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some(row),
            _ => None,
        })
        .expect("SQL EXECUTE row");
    assert_eq!(executed.get("id"), Some("2"));
    assert_eq!(executed.get("name"), Some("Linus"));
    client.batch_execute("DEALLOCATE dump_lookup").await?;

    client.batch_execute("BEGIN").await?;
    client
        .batch_execute(
            "DECLARE _pg_dump_cursor CURSOR FOR \
             SELECT id, name FROM ONLY public.driver_people ORDER BY id",
        )
        .await?;
    let first_fetch = client.simple_query("FETCH 2 FROM _pg_dump_cursor").await?;
    let first_ids = first_fetch
        .iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get("id"),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(first_ids, vec!["1", "2"]);
    let second_fetch = client.simple_query("FETCH 2 FROM _pg_dump_cursor").await?;
    let second_ids = second_fetch
        .iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get("id"),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(second_ids, vec!["3"]);
    client.batch_execute("COMMIT").await?;
    client
        .simple_query("FETCH 1 FROM _pg_dump_cursor")
        .await
        .expect_err("transaction completion must close SQL cursors");

    {
        let transaction = client.transaction().await?;
        let statement = transaction
            .prepare("SELECT id FROM driver_people ORDER BY id")
            .await?;
        let portal = transaction.bind(&statement, &[]).await?;
        let first = transaction.query_portal(&portal, 2).await?;
        let second = transaction.query_portal(&portal, 1).await?;
        let exhausted = transaction.query_portal(&portal, 1).await?;
        assert_eq!(
            first
                .iter()
                .map(|row| row.get::<_, i32>(0))
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(second[0].get::<_, i32>(0), 3);
        assert!(exhausted.is_empty());
        transaction.commit().await?;
    }

    client
        .batch_execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
        .await?;
    let isolation = client
        .query_one("SHOW TRANSACTION ISOLATION LEVEL", &[])
        .await?;
    assert_eq!(isolation.get::<_, String>(0), "repeatable read");
    client.batch_execute("ROLLBACK").await?;

    client.batch_execute("BEGIN").await?;
    client
        .execute(
            "INSERT INTO driver_people (id, name) VALUES ($1, $2)",
            &[&90_i32, &"staged"],
        )
        .await?;
    let duplicate = client
        .execute(
            "INSERT INTO driver_people (id, name) VALUES ($1, $2)",
            &[&90_i32, &"duplicate"],
        )
        .await
        .expect_err("duplicate staged key must fail the explicit transaction");
    assert_eq!(duplicate.code().map(|code| code.code()), Some("23505"));
    let blocked = client
        .query_one("SELECT name FROM driver_people WHERE id = 3", &[])
        .await
        .expect_err("failed transaction must reject subsequent prepared work");
    assert_eq!(blocked.code().map(|code| code.code()), Some("25P02"));
    client.batch_execute("ROLLBACK").await?;
    let rolled_back = client
        .query("SELECT id FROM driver_people WHERE id = 90", &[])
        .await?;
    assert!(rolled_back.is_empty());
    assert_eq!(
        client
            .execute(
                "INSERT INTO driver_people (id, name) VALUES ($1, $2)",
                &[&90_i32, &"reused"],
            )
            .await?,
        1
    );
    let null_name: Option<&str> = None;
    assert_eq!(
        client
            .execute(
                "INSERT INTO driver_people (id, name) VALUES ($1, $2)",
                &[&91_i32, &null_name],
            )
            .await?,
        1
    );
    let null_row = client
        .query_one("SELECT name FROM driver_people WHERE id = $1", &[&91_i32])
        .await?;
    assert_eq!(null_row.get::<_, Option<String>>(0), None);

    // Every exposed logical type crosses an actual Parse/Bind/Execute exchange using this
    // client's native value mappings. PgNumeric is the one test-local binary mapping because
    // tokio-postgres deliberately does not ship an arbitrary-precision NUMERIC implementation.
    client
        .batch_execute(
            "CREATE TABLE driver_all_types (\
                row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),\
                flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID\
             );",
        )
        .await?;
    let numeric = PgNumeric::fixed_12345_6700();
    let day = NaiveDate::from_ymd_opt(1999, 12, 31).expect("fixed valid date");
    let created_at = day
        .and_hms_micro_opt(0, 0, 1, 234_567)
        .expect("fixed valid timestamp");
    let ident = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").expect("fixed UUID");
    let insert_all_types = client
        .prepare("INSERT INTO driver_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)")
        .await?;
    client
        .execute(
            &insert_all_types,
            &[
                &1_i32,
                &-7_i16,
                &42_i32,
                &-9_i64,
                &numeric,
                &true,
                &"Grüße",
                &day,
                &created_at,
                &ident,
            ],
        )
        .await?;
    let absent_i2: Option<i16> = None;
    let absent_i4: Option<i32> = None;
    let absent_i8: Option<i64> = None;
    let absent_numeric: Option<PgNumeric> = None;
    let absent_bool: Option<bool> = None;
    let absent_text: Option<String> = None;
    let absent_day: Option<NaiveDate> = None;
    let absent_timestamp: Option<NaiveDateTime> = None;
    let absent_uuid: Option<Uuid> = None;
    client
        .execute(
            &insert_all_types,
            &[
                &2_i32,
                &absent_i2,
                &absent_i4,
                &absent_i8,
                &absent_numeric,
                &absent_bool,
                &absent_text,
                &absent_day,
                &absent_timestamp,
                &absent_uuid,
            ],
        )
        .await?;
    let all_types = client
        .query_one(
            "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident \
             FROM driver_all_types WHERE row_id = 1",
            &[],
        )
        .await?;
    assert_eq!(
        (
            all_types.get::<_, i16>(0),
            all_types.get::<_, i32>(1),
            all_types.get::<_, i64>(2),
            all_types.get::<_, PgNumeric>(3),
            all_types.get::<_, bool>(4),
            all_types.get::<_, String>(5),
            all_types.get::<_, NaiveDate>(6),
            all_types.get::<_, NaiveDateTime>(7),
            all_types.get::<_, Uuid>(8),
        ),
        (
            -7,
            42,
            -9,
            numeric,
            true,
            "Grüße".to_string(),
            day,
            created_at,
            ident
        )
    );
    let typed_nulls = client
        .query_one(
            "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident \
             FROM driver_all_types WHERE row_id = 2",
            &[],
        )
        .await?;
    assert_eq!(
        (
            typed_nulls.get::<_, Option<i16>>(0),
            typed_nulls.get::<_, Option<i32>>(1),
            typed_nulls.get::<_, Option<i64>>(2),
            typed_nulls.get::<_, Option<PgNumeric>>(3),
            typed_nulls.get::<_, Option<bool>>(4),
            typed_nulls.get::<_, Option<String>>(5),
            typed_nulls.get::<_, Option<NaiveDate>>(6),
            typed_nulls.get::<_, Option<NaiveDateTime>>(7),
            typed_nulls.get::<_, Option<Uuid>>(8),
        ),
        (None, None, None, None, None, None, None, None, None),
        "every declared result type must retain wire NULL"
    );

    client
        .batch_execute("CREATE TABLE driver_constraint_parent (id INT PRIMARY KEY)")
        .await?;
    client
        .batch_execute(
            "CREATE TABLE driver_constraint_child (\
                 id INT PRIMARY KEY, parent_id INT, amount NUMERIC(4,2),\
                 CONSTRAINT driver_constraint_positive CHECK (amount > 0.00)\
             )",
        )
        .await?;
    client
        .batch_execute(
            "ALTER TABLE ONLY driver_constraint_child ADD CONSTRAINT driver_constraint_parent_fk \
             FOREIGN KEY (parent_id) REFERENCES driver_constraint_parent(id)",
        )
        .await?;
    client
        .batch_execute("INSERT INTO driver_constraint_parent VALUES (1)")
        .await?;

    // This is deliberately a real tokio-postgres prepared statement (its generated named
    // statement uses Parse/Bind/Execute), outside an explicit transaction. A failed W1 must be
    // effect-free, retain its typed constraint SQLSTATE, and leave this same client usable.
    let prepared_not_null = client
        .prepare("INSERT INTO driver_constraint_child VALUES ($1, $2, $3)")
        .await?;
    let child_count_before = client
        .query_one("SELECT COUNT(*) FROM driver_constraint_child", &[])
        .await?
        .get::<_, i64>(0);
    let absent_child_id: Option<i32> = None;
    let not_null_error = client
        .execute(
            &prepared_not_null,
            &[&absent_child_id, &1_i32, &PgNumeric::fixed_1_00()],
        )
        .await
        .expect_err("prepared autocommit Bind/Execute must surface NOT NULL");
    assert_eq!(
        not_null_error.code().map(|code| code.code()),
        Some("23502"),
        "the prepared W1 failure must preserve the typed constraint SQLSTATE"
    );
    let child_count_after = client
        .query_one("SELECT COUNT(*) FROM driver_constraint_child", &[])
        .await?
        .get::<_, i64>(0);
    assert_eq!(
        child_count_after, child_count_before,
        "failed W1 must publish no row"
    );
    assert_eq!(
        client.query_one("SELECT 1", &[]).await?.get::<_, i32>(0),
        1,
        "autocommit W1 failure must not poison this client connection"
    );

    for (sql, sqlstate) in [
        (
            "INSERT INTO driver_constraint_child VALUES (NULL, 1, 1.00)",
            "23502",
        ),
        (
            "INSERT INTO driver_constraint_child VALUES (2, 999, 1.00)",
            "23503",
        ),
        (
            "INSERT INTO driver_constraint_child VALUES (3, 1, -1.00)",
            "23514",
        ),
        (
            "INSERT INTO driver_constraint_child VALUES (4, 1, 999.99)",
            "22003",
        ),
    ] {
        require_failed_transaction_rollback(&client, sql, sqlstate).await?;
    }
    assert_eq!(
        client
            .execute(
                "INSERT INTO driver_constraint_child VALUES ($1, $2, $3)",
                &[&10_i32, &1_i32, &PgNumeric::fixed_1_00()],
            )
            .await?,
        1,
        "ROLLBACK must leave the connection reusable"
    );

    let statement = client
        .prepare("SELECT id, name FROM driver_people WHERE id = $1")
        .await?;
    let rows = client.query(&statement, &[&2_i32]).await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, i32>(0), 2);
    assert_eq!(rows[0].get::<_, String>(1), "Linus");

    let empty_rows = client.query(&statement, &[&99_i32]).await?;
    assert!(empty_rows.is_empty());

    let missing_public = client
        .prepare("SELECT oid FROM public.pg_type ORDER BY oid")
        .await
        .expect_err("explicit public lookup must not synthesize pg_catalog.pg_type");
    assert_eq!(missing_public.code().map(|code| code.code()), Some("42P01"));
    client
        .batch_execute(
            "CREATE TABLE pg_type (oid INT PRIMARY KEY);
             INSERT INTO pg_type VALUES (9001);",
        )
        .await?;
    let public_statement = client
        .prepare("SELECT oid FROM public.pg_type ORDER BY oid")
        .await?;
    let public_rows = client.query(&public_statement, &[]).await?;
    assert_eq!(public_rows.len(), 1);
    assert_eq!(public_rows[0].get::<_, i32>(0), 9001);
    client.batch_execute("DROP TABLE pg_type").await?;
    let dropped_public = client
        .query(&public_statement, &[])
        .await
        .expect_err("drop must not rebind public.pg_type to pg_catalog.pg_type");
    assert_eq!(dropped_public.code().map(|code| code.code()), Some("42P01"));

    let mut copy_stream = stream::iter(
        vec![
            Bytes::from_static(b"id|name\n"),
            Bytes::from_static(b"4|Katherine\n5|\"Dorothy|Vaughan\"\n"),
        ]
        .into_iter()
        .map(Ok::<_, tokio_postgres::Error>),
    );
    let copy_sink = client
        .copy_in(
            "COPY driver_people (id, name) FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|')",
        )
        .await?;
    pin_mut!(copy_sink);
    copy_sink.send_all(&mut copy_stream).await?;
    assert_eq!(copy_sink.finish().await?, 2);

    let copied_rows = client
        .query(
            "SELECT id, name FROM driver_people WHERE id >= $1 AND id <= 5 ORDER BY id",
            &[&4_i32],
        )
        .await?;
    assert_eq!(copied_rows.len(), 2);
    assert_eq!(copied_rows[0].get::<_, i32>(0), 4);
    assert_eq!(copied_rows[0].get::<_, String>(1), "Katherine");
    assert_eq!(copied_rows[1].get::<_, i32>(0), 5);
    assert_eq!(copied_rows[1].get::<_, String>(1), "Dorothy|Vaughan");

    let copy_out_statement = client
        .prepare("COPY driver_people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|')")
        .await?;
    let copy_out = client
        .copy_out(&copy_out_statement)
        .await?
        .try_fold(BytesMut::new(), |mut output, chunk| async move {
            output.extend_from_slice(&chunk);
            Ok(output)
        })
        .await?;
    let copy_out = std::str::from_utf8(&copy_out)?;
    assert!(copy_out.contains("id|name\n"));
    assert!(copy_out.contains("4|Katherine\n"));
    assert!(copy_out.contains("5|\"Dorothy|Vaughan\"\n"));

    let unsupported = client
        .simple_query("COPY driver_people FROM STDIN WITH CSV HEADER DELIMITER ','")
        .await
        .expect_err("broader COPY CSV options remain explicitly unsupported");
    assert_eq!(unsupported.code().map(|code| code.code()), Some("0A000"));

    let recovered = client
        .query("SELECT name FROM driver_people WHERE id = 3", &[])
        .await?;
    assert_eq!(recovered[0].get::<_, String>(0), "Grace");

    drop(client);
    let reconnected = connect(server.port).await?;
    let reconnect_simple = reconnected.simple_query("SELECT 1 AS one").await?;
    let reconnect_row = reconnect_simple
        .iter()
        .find_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some(row),
            _ => None,
        })
        .expect("reconnect simple query row");
    assert_eq!(reconnect_row.get("one"), Some("1"));

    Ok(())
}
