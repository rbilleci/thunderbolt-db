use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use futures_util::{pin_mut, stream, SinkExt, TryStreamExt};
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

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
