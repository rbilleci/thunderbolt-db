use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::str::FromStr;
use std::thread;
use std::time::{Duration, Instant};

use bigdecimal::BigDecimal;
use chrono::{NaiveDate, NaiveDateTime};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions, PgSslMode};
use sqlx::{Connection, PgConnection, Row};
use uuid::Uuid;

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
    let mut child = Command::new(env!("CARGO_BIN_EXE_thunderbolt-db-server"))
        .arg(format!("127.0.0.1:{port}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn thunderbolt-db-server");

    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return ServerGuard { child, port };
        }
        thread::sleep(Duration::from_millis(25));
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("thunderbolt-db-server did not start listening on 127.0.0.1:{port}");
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

    let day = NaiveDate::from_ymd_opt(1999, 12, 31).expect("fixed valid date");
    let created_at = day
        .and_hms_micro_opt(0, 0, 1, 234_567)
        .expect("fixed valid timestamp");
    let ident = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")?;
    let amount = BigDecimal::from_str("12345.6700")?;
    sqlx::raw_sql(
        "CREATE TABLE sqlx_all_types (\
            row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),\
            flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID\
         )",
    )
    .execute(&mut conn)
    .await?;
    sqlx::query("INSERT INTO sqlx_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)")
        .bind(1_i32)
        .bind(-7_i16)
        .bind(42_i32)
        .bind(-9_i64)
        .bind(&amount)
        .bind(true)
        .bind("Grüße")
        .bind(day)
        .bind(created_at)
        .bind(ident)
        .execute(&mut conn)
        .await?;
    sqlx::query("INSERT INTO sqlx_all_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)")
        .bind(2_i32)
        .bind(Option::<i16>::None)
        .bind(Option::<i32>::None)
        .bind(Option::<i64>::None)
        .bind(Option::<BigDecimal>::None)
        .bind(Option::<bool>::None)
        .bind(Option::<String>::None)
        .bind(Option::<NaiveDate>::None)
        .bind(Option::<NaiveDateTime>::None)
        .bind(Option::<Uuid>::None)
        .execute(&mut conn)
        .await?;
    let all_types = sqlx::query(
        "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident \
         FROM sqlx_all_types WHERE row_id = 1",
    )
    .fetch_one(&mut conn)
    .await?;
    assert_eq!(all_types.try_get::<i16, _>("i2")?, -7);
    assert_eq!(all_types.try_get::<i32, _>("i4")?, 42);
    assert_eq!(all_types.try_get::<i64, _>("i8")?, -9);
    assert_eq!(all_types.try_get::<BigDecimal, _>("amount")?, amount);
    assert!(all_types.try_get::<bool, _>("flag")?);
    assert_eq!(all_types.try_get::<String, _>("note")?, "Grüße");
    assert_eq!(all_types.try_get::<NaiveDate, _>("day")?, day);
    assert_eq!(
        all_types.try_get::<NaiveDateTime, _>("created_at")?,
        created_at
    );
    assert_eq!(all_types.try_get::<Uuid, _>("ident")?, ident);
    let typed_nulls = sqlx::query(
        "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident \
         FROM sqlx_all_types WHERE row_id = 2",
    )
    .fetch_one(&mut conn)
    .await?;
    assert_eq!(typed_nulls.try_get::<Option<i16>, _>("i2")?, None);
    assert_eq!(typed_nulls.try_get::<Option<i32>, _>("i4")?, None);
    assert_eq!(typed_nulls.try_get::<Option<i64>, _>("i8")?, None);
    assert_eq!(
        typed_nulls.try_get::<Option<BigDecimal>, _>("amount")?,
        None
    );
    assert_eq!(typed_nulls.try_get::<Option<bool>, _>("flag")?, None);
    assert_eq!(typed_nulls.try_get::<Option<String>, _>("note")?, None);
    assert_eq!(typed_nulls.try_get::<Option<NaiveDate>, _>("day")?, None);
    assert_eq!(
        typed_nulls.try_get::<Option<NaiveDateTime>, _>("created_at")?,
        None
    );
    assert_eq!(typed_nulls.try_get::<Option<Uuid>, _>("ident")?, None);

    sqlx::raw_sql("CREATE TABLE sqlx_numeric_contract (id INT PRIMARY KEY, amount NUMERIC(4,2))")
        .execute(&mut conn)
        .await?;
    sqlx::raw_sql("BEGIN").execute(&mut conn).await?;
    let overflow = sqlx::query("INSERT INTO sqlx_numeric_contract VALUES ($1, $2)")
        .bind(1_i32)
        .bind(BigDecimal::from_str("999.99")?)
        .execute(&mut conn)
        .await
        .expect_err("numeric precision overflow must fail");
    let overflow_sqlstate = overflow
        .as_database_error()
        .expect("numeric range error must be a database error")
        .code()
        .map(|code| code.into_owned());
    assert_eq!(overflow_sqlstate.as_deref(), Some("22003"));
    let blocked = sqlx::raw_sql("SELECT 1")
        .execute(&mut conn)
        .await
        .expect_err("numeric failure must abort explicit transaction");
    let blocked_sqlstate = blocked
        .as_database_error()
        .expect("failed-transaction rejection must be a database error")
        .code()
        .map(|code| code.into_owned());
    assert_eq!(blocked_sqlstate.as_deref(), Some("25P02"));
    sqlx::raw_sql("ROLLBACK").execute(&mut conn).await?;
    sqlx::query("INSERT INTO sqlx_numeric_contract VALUES ($1, $2)")
        .bind(1_i32)
        .bind(BigDecimal::from_str("1.00")?)
        .execute(&mut conn)
        .await?;

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
