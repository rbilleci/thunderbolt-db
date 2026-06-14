//! End-to-end proof (P0-M3): a real PostgreSQL client (`tokio-postgres`) drives
//! the engine-backed façade server over a TCP socket through the simple query
//! protocol, and a CREATE/INSERT/SELECT lifecycle round-trips correctly.

use std::net::TcpListener;
use std::thread;

use tokio_postgres::{NoTls, SimpleQueryMessage};

#[tokio::test]
async fn engine_backed_facade_server_round_trips_over_pgwire() {
    // Bind first so we know the port; the server owns its (non-Send) engine on
    // its own thread, created inside the closure.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        let _ = gpu_db_server::serve(listener);
    });

    let (client, connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await
    .expect("connect to engine-backed facade server");
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE accounts (id INT, name TEXT)")
        .await
        .expect("create table");
    client
        .simple_query("INSERT INTO accounts (id, name) VALUES (1, 'alice')")
        .await
        .expect("insert alice");
    client
        .simple_query("INSERT INTO accounts (id, name) VALUES (2, 'bob')")
        .await
        .expect("insert bob");

    let messages = client
        .simple_query("SELECT id, name FROM accounts WHERE id = 1")
        .await
        .expect("select");

    let mut rows = 0;
    for message in messages {
        if let SimpleQueryMessage::Row(row) = message {
            assert_eq!(row.get("id"), Some("1"));
            assert_eq!(row.get("name"), Some("alice"));
            rows += 1;
        }
    }
    assert_eq!(rows, 1, "expected exactly one row for id = 1");
}

/// P1-M5: the async-ingress server round-trips a real client over the wire (proves the
/// async pgwire framing + the async→blocking-engine bridge via spawn_blocking).
#[tokio::test]
async fn async_ingress_server_round_trips_over_pgwire() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = gpu_db_server::serve_async(listener).await;
    });

    let (client, connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await
    .expect("connect to async-ingress server");
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE accounts (id INT, name TEXT)")
        .await
        .expect("create table");
    client
        .simple_query("INSERT INTO accounts (id, name) VALUES (1, 'alice')")
        .await
        .expect("insert alice");

    let messages = client
        .simple_query("SELECT id, name FROM accounts WHERE id = 1")
        .await
        .expect("select");
    let mut rows = 0;
    for message in messages {
        if let SimpleQueryMessage::Row(row) = message {
            assert_eq!(row.get("id"), Some("1"));
            assert_eq!(row.get("name"), Some("alice"));
            rows += 1;
        }
    }
    assert_eq!(rows, 1);
}

/// P1-M5: many connections are served concurrently as lightweight tasks (no thread per
/// connection), each round-tripping correctly against the shared engine.
#[tokio::test(flavor = "multi_thread")]
async fn async_ingress_serves_many_concurrent_connections() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = gpu_db_server::serve_async(listener).await;
    });
    let conn_str = format!("host=127.0.0.1 port={port} user=postgres dbname=postgres");

    // Seed on one connection.
    {
        let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client.simple_query("CREATE TABLE t (a INT)").await.unwrap();
        for i in 0..25 {
            client
                .simple_query(&format!("INSERT INTO t (a) VALUES ({i})"))
                .await
                .unwrap();
        }
    }

    // 64 concurrent connections each read the table and must see 25 rows.
    let mut handles = Vec::new();
    for _ in 0..64 {
        let conn_str = conn_str.clone();
        handles.push(tokio::spawn(async move {
            let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
            tokio::spawn(async move {
                let _ = connection.await;
            });
            for _ in 0..10 {
                let messages = client.simple_query("SELECT COUNT(*) FROM t").await.unwrap();
                let mut got = None;
                for message in messages {
                    if let SimpleQueryMessage::Row(row) = message {
                        got = row.get(0).map(|s| s.to_string());
                    }
                }
                assert_eq!(got.as_deref(), Some("25"));
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
}
