//! End-to-end proof (P0-M3): a real PostgreSQL client (`tokio-postgres`) drives
//! the engine-backed façade server over a TCP socket through the simple query
//! protocol, and a CREATE/INSERT/SELECT lifecycle round-trips correctly.

use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use bytes::{Bytes, BytesMut};
use futures_util::{stream, SinkExt, TryStreamExt};
use gpu_db_engine::Engine;
use gpu_db_execution::{CudaDriverRuntime, DeviceTarget};
use gpu_db_facade::SharedEngine;
use tokio_postgres::{NoTls, SimpleQueryMessage};

/// STRATA golden gate: a real pgwire client carries alternating NULL/non-NULL values for every
/// exposed logical type, then reaches the dense GPU point route for coarse- and fine-sharded
/// generations. The all-type wire assertion is deliberately separate from the route counter:
/// that counter proves the specialized keyed resident `accounts` projection, not a claim that
/// the current point kernel covers one wide heterogeneous projection.
#[tokio::test]
#[ignore = "requires a local NVIDIA driver and GPU"]
async fn pgwire_gpu_point_route_is_non_vacuous_across_coarse_and_fine_shards() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    let runtime = runtime.snapshot();
    if !runtime.driver_available || runtime.device_count == 0 {
        return;
    }
    async fn run_arm(
        shard_target: usize,
    ) -> (
        Vec<String>,
        Vec<Option<String>>,
        Vec<Vec<Option<String>>>,
        u64,
        u64,
        usize,
    ) {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(shard_target);
        engine.set_shard_index_probe_enabled(true);
        engine.set_shard_batched_point_read_enabled(true);
        engine.set_auto_admit_on_commit(true);
        let mut txn_id = 1u64;
        engine
            .execute_text(
                txn_id,
                "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, nullable_note INT)",
            )
            .unwrap();
        let row_values = (0..130)
            .map(|id| {
                let note = if id == 1 { "NULL" } else { "9" };
                format!("({id}, {}, {note})", id * 7)
            })
            .collect::<Vec<_>>();
        if shard_target == 32 {
            for value in &row_values {
                txn_id += 1;
                engine
                    .execute_text(
                        txn_id,
                        &format!(
                            "INSERT INTO accounts (id, balance, nullable_note) VALUES {value}"
                        ),
                    )
                    .unwrap();
            }
        } else {
            let values = row_values.join(",");
            txn_id += 1;
            engine
                .execute_text(
                    txn_id,
                    &format!("INSERT INTO accounts (id, balance, nullable_note) VALUES {values}"),
                )
                .unwrap();
        }
        let keyed_device_authoritative_publications = engine.device_authoritative_commits();
        let direct_payload = engine
            .execute_relational_select_text("SELECT balance FROM accounts WHERE id = 100")
            .unwrap();
        assert_eq!(direct_payload.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(direct_payload.fallback_reason, None);
        let direct_null = engine
            .execute_relational_select_text("SELECT nullable_note FROM accounts WHERE id = 1")
            .unwrap();
        assert_eq!(direct_null.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(direct_null.fallback_reason, None);

        // Every logical type crosses pgwire below with a non-NULL value in exactly one row and
        // NULL in the other. The expected text is fixed fixture algebra, not a CPU-executor
        // oracle. The resident point-route proof remains the keyed `accounts` projection above.
        txn_id += 1;
        engine
            .execute_text(
                txn_id,
                "CREATE TABLE wire_all_types (\
                    row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),\
                    flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID\
                 )",
            )
            .unwrap();
        txn_id += 1;
        engine
            .execute_text(
                txn_id,
                "INSERT INTO wire_all_types VALUES \
                    (1, -7, NULL, -9000000000, 12.3400, NULL, 'left', NULL, \
                     '2000-01-01 00:00:01.234567', NULL),\
                    (2, NULL, 42, NULL, NULL, true, NULL, '1999-12-31', NULL, \
                     '550e8400-e29b-41d4-a716-446655440000')",
            )
            .unwrap();

        let shared = Arc::new(SharedEngine::from_engine(engine));
        let before = shared.gpu_native_activity_snapshot("accounts");
        if before.resident_shards == 0 {
            return (Vec::new(), Vec::new(), Vec::new(), 0, 0, 0);
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
        let typed_messages = client
            .simple_query(
                "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident \
                 FROM wire_all_types ORDER BY row_id",
            )
            .await
            .unwrap();
        let typed_rows = typed_messages
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => Some(
                    (0..9)
                        .map(|column| row.get(column).map(str::to_owned))
                        .collect(),
                ),
                _ => None,
            })
            .collect();
        (
            payloads,
            nulls,
            typed_rows,
            after.sharded_gpu_probe_batches - before.sharded_gpu_probe_batches,
            keyed_device_authoritative_publications,
            before.resident_shards,
        )
    }

    let coarse = run_arm(1024).await;
    let fine = run_arm(32).await;
    assert_eq!(coarse.0, vec!["700"]);
    assert_eq!(coarse.1, vec![None]);
    assert_eq!(
        coarse.2,
        vec![
            vec![
                Some("-7".to_string()),
                None,
                Some("-9000000000".to_string()),
                Some("12.3400".to_string()),
                None,
                Some("left".to_string()),
                None,
                Some("2000-01-01 00:00:01.234567".to_string()),
                None,
            ],
            vec![
                None,
                Some("42".to_string()),
                None,
                None,
                Some("t".to_string()),
                None,
                Some("1999-12-31".to_string()),
                None,
                Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            ],
        ],
        "all-type pgwire rows preserve both typed values and per-column NULLs"
    );
    assert_eq!(
        (fine.0.clone(), fine.1.clone(), fine.2.clone()),
        (coarse.0, coarse.1, coarse.2),
        "coarse and fine resident generations agree on the same closed-form wire fixture"
    );
    assert!(coarse.5 > 0, "coarse-sharded arm has resident shards");
    assert!(
        fine.5 > coarse.5,
        "fine-sharded arm has more shards than coarse arm"
    );
    assert!(
        coarse.3 > 0 && fine.3 > 0,
        "wire reads fired the dense GPU route"
    );
    assert!(
        coarse.4 > 0 && fine.4 > 0,
        "the keyed resident accounts fixture published device-authoritative state before pgwire reads"
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

/// PRODUCT-001 COPY slice: async ingress uses the same facade-owned typed admission as blocking
/// ingress. Explicit rollback, CSV NULL-vs-empty, COPY TO, parse failure, and client abort all prove
/// that no buffered host rows become a second publication owner.
#[tokio::test]
async fn async_ingress_copy_is_transactional_typed_and_recovers() {
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
    .unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute("CREATE TABLE copy_t (id INT, name TEXT)")
        .await
        .unwrap();

    client.batch_execute("BEGIN").await.unwrap();
    let mut rollback_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"9,rolled back\n",
    ))]);
    let rollback_sink = client
        .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(rollback_sink);
    rollback_sink.send_all(&mut rollback_data).await.unwrap();
    assert_eq!(rollback_sink.finish().await.unwrap(), 1);
    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM copy_t", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        1
    );
    client.batch_execute("ROLLBACK").await.unwrap();
    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM copy_t", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        0
    );

    let mut typed_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"1,\n2,\"\"\n",
    ))]);
    let typed_sink = client
        .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(typed_sink);
    typed_sink.send_all(&mut typed_data).await.unwrap();
    assert_eq!(typed_sink.finish().await.unwrap(), 2);
    let typed_rows = client
        .query("SELECT id, name FROM copy_t ORDER BY id", &[])
        .await
        .unwrap();
    assert_eq!(typed_rows[0].get::<_, Option<String>>(1), None);
    assert_eq!(
        typed_rows[1].get::<_, Option<String>>(1),
        Some(String::new())
    );

    let copy_out_statement = client
        .prepare("COPY copy_t TO STDOUT WITH CSV")
        .await
        .unwrap();
    let copy_out = client
        .copy_out(&copy_out_statement)
        .await
        .unwrap()
        .try_fold(BytesMut::new(), |mut output, chunk| async move {
            output.extend_from_slice(&chunk);
            Ok(output)
        })
        .await
        .unwrap();
    assert_eq!(&copy_out[..], b"1,\n2,\"\"\n");

    let mut invalid_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"not-an-int,bad\n",
    ))]);
    let invalid_sink = client
        .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(invalid_sink);
    invalid_sink.send_all(&mut invalid_data).await.unwrap();
    let invalid = invalid_sink.finish().await.unwrap_err();
    assert_eq!(invalid.code().map(|code| code.code()), Some("22P02"));

    let mut abort_sink = Box::pin(
        client
            .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
            .await
            .unwrap(),
    );
    abort_sink
        .send(Bytes::from_static(b"3,must-not-publish\n"))
        .await
        .unwrap();
    abort_sink.flush().await.unwrap();
    drop(abort_sink);

    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM copy_t", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        2
    );
}

/// Device-lifetime hazard gate for canonical COPY: three sequential wire copies followed by two
/// truly overlapping copies, then concurrent NULL-vs-zero point reads from the retained device
/// generation. Counters make both mutation publication and GPU read routing non-vacuous.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a local NVIDIA driver and GPU"]
async fn canonical_copy_survives_sequential_and_concurrent_gpu_lifetime_hazard() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    let runtime = runtime.snapshot();
    if !runtime.driver_available || runtime.device_count == 0 {
        return;
    }

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(64);
    engine.set_shard_index_probe_enabled(true);
    engine.set_shard_batched_point_read_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(1, "CREATE TABLE copy_hazard (id INT PRIMARY KEY, note INT)")
        .unwrap();
    let seed = (0..128)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            2,
            &format!("INSERT INTO copy_hazard (id, note) VALUES {seed}"),
        )
        .unwrap();
    let warm = engine
        .execute_relational_select_text("SELECT note FROM copy_hazard WHERE id = 64")
        .unwrap();
    assert_eq!(warm.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(warm.fallback_reason, None);

    let shared = Arc::new(SharedEngine::from_engine(engine));
    let before = shared.gpu_native_activity_snapshot("copy_hazard");
    assert!(
        before.resident_shards > 0,
        "fixture must be device resident"
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let served = Arc::clone(&shared);
    tokio::spawn(async move {
        let _ = gpu_db_server::serve_async_with_engine_batching(listener, served, 64, true).await;
    });
    let conn = format!("host=127.0.0.1 port={port} user=postgres dbname=postgres");

    let (pre_client, pre_connection) = tokio_postgres::connect(&conn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = pre_connection.await;
    });
    pre_client
        .simple_query("SELECT id FROM copy_hazard WHERE id = 64")
        .await
        .unwrap();
    let pre_copy_route = shared.gpu_native_activity_snapshot("copy_hazard");
    assert!(
        pre_copy_route.sharded_gpu_probe_batches > before.sharded_gpu_probe_batches,
        "fixture must reach the GPU point route before COPY: before={before:?} pre={pre_copy_route:?}"
    );

    copy_one_hazard_row(&conn, 1_001, None).await;
    copy_one_hazard_row(&conn, 1_002, Some(0)).await;
    copy_one_hazard_row(&conn, 1_003, Some(7)).await;
    let left = copy_one_hazard_row(&conn, 1_004, None);
    let right = copy_one_hazard_row(&conn, 1_005, Some(0));
    tokio::join!(left, right);

    // Prove that the generation published by the five COPY commits remains usable by the retained
    // GPU point route. Newly appended keys intentionally live outside the dense index's prepared
    // key range, so their NULL-vs-zero checks below prove content while this original key makes
    // execution-target non-vacuity explicit.
    let (probe_client, probe_connection) = tokio_postgres::connect(&conn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = probe_connection.await;
    });
    let probe = probe_client
        .simple_query("SELECT id FROM copy_hazard WHERE id = 64")
        .await
        .unwrap();
    assert!(probe.iter().any(|message| {
        matches!(message, SimpleQueryMessage::Row(row) if row.get(0) == Some("64"))
    }));

    let mut readers = Vec::new();
    for (id, expected) in [
        (1_001, None),
        (1_002, Some(0)),
        (1_003, Some(7)),
        (1_004, None),
        (1_005, Some(0)),
    ] {
        let conn = conn.clone();
        readers.push(tokio::spawn(async move {
            let (client, connection) = tokio_postgres::connect(&conn, NoTls).await.unwrap();
            tokio::spawn(async move {
                let _ = connection.await;
            });
            let sql = format!("SELECT note FROM copy_hazard WHERE id = {id}");
            let messages = client.simple_query(&sql).await.unwrap();
            let actual = messages
                .into_iter()
                .find_map(|message| match message {
                    SimpleQueryMessage::Row(row) => {
                        Some(row.get(0).map(|value| value.parse::<i32>().unwrap()))
                    }
                    _ => None,
                })
                .expect("point query returned one row");
            assert_eq!(actual, expected);
        }));
    }
    for reader in readers {
        reader.await.unwrap();
    }

    let after = shared.gpu_native_activity_snapshot("copy_hazard");
    assert!(
        after.open_shard_append_commits > before.open_shard_append_commits
            || after.device_authoritative_commits > before.device_authoritative_commits,
        "COPY must publish a device-maintained generation: before={before:?} after={after:?}"
    );
    assert!(
        after.sharded_gpu_probe_batches > pre_copy_route.sharded_gpu_probe_batches,
        "the COPY-published generation must remain on the GPU point route: pre={pre_copy_route:?} after={after:?}"
    );
}

async fn copy_one_hazard_row(connection: &str, id: i32, note: Option<i32>) {
    let (client, driver) = tokio_postgres::connect(connection, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = driver.await;
    });
    let note = note.map_or_else(String::new, |value| value.to_string());
    let mut input = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from(format!(
        "{id},{note}\n"
    )))]);
    let sink = client
        .copy_in("COPY copy_hazard (id, note) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(sink);
    sink.send_all(&mut input).await.unwrap();
    assert_eq!(sink.finish().await.unwrap(), 1);
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
