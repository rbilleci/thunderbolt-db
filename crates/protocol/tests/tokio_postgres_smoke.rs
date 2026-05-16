use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

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
    let mut child = Command::new(env!("CARGO_BIN_EXE_gpu-db-server"))
        .args(["--listen", &format!("127.0.0.1:{port}")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn gpu-db-server");

    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return ServerGuard { child, port };
        }
        thread::sleep(Duration::from_millis(25));
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("gpu-db-server did not start listening on 127.0.0.1:{port}");
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
async fn tokio_postgres_supported_subset_smoke_with_unsupported_recovery(
) -> Result<(), Box<dyn std::error::Error>> {
    let server = start_server();
    let client = connect(server.port).await?;

    let simple = client.simple_query("SELECT 1 AS one").await?;
    let row = simple
        .iter()
        .find_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some(row),
            _ => None,
        })
        .expect("simple query row");
    assert_eq!(row.get("one"), Some("1"));

    client
        .batch_execute(
            "CREATE TABLE driver_people (id INT, name TEXT);
             INSERT INTO driver_people (id, name)
             VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');",
        )
        .await?;

    let statement = client
        .prepare("SELECT id, name FROM driver_people WHERE id = $1")
        .await?;
    let rows = client.query(&statement, &[&2_i32]).await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, i32>(0), 2);
    assert_eq!(rows[0].get::<_, String>(1), "Linus");

    let empty_rows = client.query(&statement, &[&99_i32]).await?;
    assert!(empty_rows.is_empty());

    let unsupported = client
        .simple_query("COPY driver_people FROM STDIN")
        .await
        .expect_err("COPY FROM STDIN remains explicitly unsupported");
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
