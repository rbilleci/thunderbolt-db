//! End-to-end proof (P0-M3): a real PostgreSQL client (`tokio-postgres`) drives
//! the engine-backed façade server over a TCP socket through the simple query
//! protocol, and a CREATE/INSERT/SELECT lifecycle round-trips correctly.

use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use gpu_db_engine::Engine;
use gpu_db_execution::CudaDriverRuntime;
use gpu_db_facade::{execute_on_shared_engine, SharedEngine};
use tokio_postgres::{NoTls, SimpleQueryMessage};

/// STRATA golden gate: a real pgwire client reaches the dense GPU point route for one and many shards,
/// returns the same row values over pgwire as the host parity arm, and carries NULL data without disabling an unreferenced
/// projection. The route counter makes the wire-level equality non-vacuous.
#[tokio::test]
#[ignore = "requires a local NVIDIA driver and GPU"]
async fn pgwire_gpu_point_route_is_non_vacuous_across_shards_and_null_data() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    let runtime = runtime.snapshot();
    if !runtime.driver_available || runtime.device_count == 0 {
        return;
    }
    async fn run_arm(
        shard_target: Option<usize>,
    ) -> (Vec<String>, Vec<Option<String>>, u64, usize, usize) {
        let engine = Engine::new_local();
        if let Some(target) = shard_target {
            engine.set_shard_residency_enabled(true);
            engine.set_shard_size_target(target);
            engine.set_shard_index_probe_enabled(true);
            engine.set_shard_batched_point_read_enabled(true);
            engine.set_auto_admit_on_commit(true);
        } else {
            // Parity arm must remain host-pinned after S-F flips production admission defaults.
            engine.set_auto_admit_on_commit(false);
            engine.set_shard_residency_enabled(false);
        }
        let shared = Arc::new(SharedEngine::from_engine(engine));
        execute_on_shared_engine(
            &shared,
            "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, nullable_note INT)",
        )
        .unwrap();
        let row_values = (0..130)
            .map(|id| {
                let note = if id == 1 { "NULL" } else { "9" };
                format!("({id}, {}, {note})", id * 7)
            })
            .collect::<Vec<_>>();
        if shard_target == Some(32) {
            for value in &row_values {
                execute_on_shared_engine(
                    &shared,
                    &format!("INSERT INTO accounts (id, balance, nullable_note) VALUES {value}"),
                )
                .unwrap();
            }
        } else {
            let values = row_values.join(",");
            execute_on_shared_engine(
                &shared,
                &format!("INSERT INTO accounts (id, balance, nullable_note) VALUES {values}"),
            )
            .unwrap();
        }
        let before = shared.gpu_native_activity_snapshot("accounts");
        if shard_target.is_some() && before.resident_shards == 0 {
            return (Vec::new(), Vec::new(), 0, 0, 0);
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let served = Arc::clone(&shared);
        tokio::spawn(async move {
            let _ =
                gpu_db_server::serve_async_with_engine_batching(listener, served, 64, true).await;
        });
        let (client, connection) = tokio_postgres::connect(
            &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
            NoTls,
        )
        .await
        .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let payload_messages = client
            .simple_query("SELECT balance FROM accounts WHERE id = 100")
            .await
            .unwrap();
        let payloads = payload_messages
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
                _ => None,
            })
            .collect();
        let after = shared.gpu_native_activity_snapshot("accounts");
        let null_messages = client
            .simple_query("SELECT nullable_note FROM accounts WHERE id = 1")
            .await
            .unwrap();
        let nulls = null_messages
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => Some(row.get(0).map(str::to_owned)),
                _ => None,
            })
            .collect();
        (
            payloads,
            nulls,
            after.sharded_gpu_probe_batches - before.sharded_gpu_probe_batches,
            before.resident_shards,
            before.valid_resident_tables,
        )
    }

    let host = run_arm(None).await;
    let one = run_arm(Some(1024)).await;
    let many = run_arm(Some(32)).await;
    assert_eq!(host.0, vec!["700"]);
    assert_eq!(host.1, vec![None]);
    assert_eq!(host.3, 0, "host parity arm is explicitly non-resident");
    assert_eq!(
        host.4, 0,
        "host parity arm has no unified resident snapshot"
    );
    assert_eq!(
        (one.0.clone(), one.1.clone()),
        (host.0.clone(), host.1.clone())
    );
    assert_eq!((many.0.clone(), many.1.clone()), (host.0, host.1));
    assert_eq!(one.3, 1, "single-shard arm");
    assert!(many.3 >= 2, "multi-shard arm");
    assert!(
        one.2 > 0 && many.2 > 0,
        "wire reads fired the dense GPU route"
    );
}

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
