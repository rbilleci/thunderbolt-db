use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use sqlx::postgres::{PgConnectOptions, PgPoolOptions, PgSslMode};
use sqlx::{Connection, PgConnection, Row};

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

fn connect_options(port: u16) -> PgConnectOptions {
    PgConnectOptions::new()
        .host("127.0.0.1")
        .port(port)
        .username("postgres")
        .database("postgres")
        .application_name("sqlx_smoke")
        .ssl_mode(PgSslMode::Disable)
}

#[tokio::test]
async fn sqlx_supported_subset_smoke_with_unsupported_recovery(
) -> Result<(), Box<dyn std::error::Error>> {
    let server = start_server();
    let options = connect_options(server.port);
    let mut conn = PgConnection::connect_with(&options).await?;

    let row = sqlx::raw_sql("SELECT 1 AS one")
        .fetch_one(&mut conn)
        .await?;
    assert_eq!(row.try_get::<i32, _>("one")?, 1);
    let null_row = sqlx::raw_sql("SELECT NULL::int4 AS missing")
        .fetch_one(&mut conn)
        .await?;
    assert_eq!(null_row.try_get::<Option<i32>, _>("missing")?, None);

    sqlx::raw_sql(
        "CREATE TABLE sqlx_people (id INT, name TEXT);
         INSERT INTO sqlx_people (id, name)
         VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');",
    )
    .execute(&mut conn)
    .await?;

    let row = sqlx::query("SELECT id, name FROM sqlx_people WHERE id = $1")
        .bind(2_i32)
        .fetch_one(&mut conn)
        .await?;
    assert_eq!(row.try_get::<i32, _>("id")?, 2);
    assert_eq!(row.try_get::<String, _>("name")?, "Linus");

    let empty_rows = sqlx::query("SELECT id, name FROM sqlx_people WHERE id = $1")
        .bind(99_i32)
        .fetch_all(&mut conn)
        .await?;
    assert!(empty_rows.is_empty());

    let unsupported = sqlx::raw_sql("COPY sqlx_people FROM STDIN WITH CSV HEADER DELIMITER ','")
        .execute(&mut conn)
        .await
        .expect_err("broader COPY CSV options remain explicitly unsupported");
    let database_error = unsupported
        .as_database_error()
        .expect("broader COPY CSV option error should be a database error");
    assert_eq!(database_error.code().as_deref(), Some("0A000"));

    let recovered = sqlx::query("SELECT name FROM sqlx_people WHERE id = 3")
        .fetch_one(&mut conn)
        .await?;
    assert_eq!(recovered.try_get::<String, _>("name")?, "Grace");
    drop(conn);

    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await?;
    let reconnect_row = sqlx::raw_sql("SELECT 1 AS one").fetch_one(&pool).await?;
    assert_eq!(reconnect_row.try_get::<i32, _>("one")?, 1);

    Ok(())
}
