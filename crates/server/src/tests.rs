use super::{
    handle_connection, handle_connection_async, handle_connection_registered, parse_batching_flag,
    read_tagged_frame_async, serve_async_with_engine_batching, CancellationRegistry,
};
use gpu_db_facade::SharedEngine;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[path = "tests/index_lifecycle.rs"]
mod index_lifecycle;

#[path = "tests/sequence_lifecycle.rs"]
mod sequence_lifecycle;

#[path = "tests/view_lifecycle.rs"]
mod view_lifecycle;

/// Thread-3 default-on: an unset `GPU_DB_BATCHING` enables batching. A default-started
/// async server (no env) therefore constructs a `PointLookupBatcher` and routes batchable
/// point-lookups through it. (Decoded via the pure helper so the test does not mutate
/// process-global env, which is racy across the parallel test runner.)
#[test]
fn batching_defaults_on_when_unset() {
    assert!(parse_batching_flag(None));
}

/// The disable escape hatch: `0`/`false`/`off`/`no` (case-insensitive, whitespace-tolerant)
/// turns batching off and restores the per-query path.
#[test]
fn explicit_off_values_disable_batching() {
    for off in ["0", "false", "off", "no", "FALSE", "Off", "  no  "] {
        assert!(
            !parse_batching_flag(Some(off)),
            "{off:?} should disable batching"
        );
    }
}

/// Everything that is not an explicit off token keeps the default-on behavior — including
/// the historical truthy values and any unrecognized/empty value (fail safe = on).
#[test]
fn truthy_and_unrecognized_values_keep_batching_on() {
    for on in [
        "1", "true", "on", "yes", "TRUE", "On", "", "enabled", "garbage",
    ] {
        assert!(
            parse_batching_flag(Some(on)),
            "{on:?} should keep batching on"
        );
    }
}

#[test]
fn blocking_late_auth_frames_error_skip_until_sync_and_recover() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    for (case, payload) in late_auth_payloads() {
        client.write_all(&tagged(b'p', &payload)).unwrap();
        assert_late_auth_error(&read_messages(&mut client, 1));

        // A post-startup auth error is an extended-protocol error: every frame through Sync is
        // ignored. Reusing the exact table name after idle Ready proves that the queued Query was
        // discarded, while the successful CREATE proves the connection remains usable.
        let table = format!("blocking_late_auth_{case}");
        let mut recovery = tagged(
            b'Q',
            &query_payload(&format!("CREATE TABLE {table} (id int4)")),
        );
        recovery.extend(tagged(b'S', &[]));
        client.write_all(&recovery).unwrap();
        assert_eq!(read_messages(&mut client, 1), vec![(b'Z', vec![b'I'])]);

        client
            .write_all(&tagged(
                b'Q',
                &query_payload(&format!("CREATE TABLE {table} (id int4)")),
            ))
            .unwrap();
        assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    }

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_late_auth_frames_error_skip_until_sync_and_recover() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_async_with_engine_batching(
        listener,
        std::sync::Arc::new(SharedEngine::new()),
        2,
        false,
    ));
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client.write_all(&startup_frame()).await.unwrap();
    let _ = read_messages_async(&mut client, 9).await;

    for (case, payload) in late_auth_payloads() {
        client.write_all(&tagged(b'p', &payload)).await.unwrap();
        assert_late_auth_error(&read_messages_async(&mut client, 1).await);

        let table = format!("async_late_auth_{case}");
        let mut recovery = tagged(
            b'Q',
            &query_payload(&format!("CREATE TABLE {table} (id int4)")),
        );
        recovery.extend(tagged(b'S', &[]));
        client.write_all(&recovery).await.unwrap();
        assert_eq!(
            read_messages_async(&mut client, 1).await,
            vec![(b'Z', vec![b'I'])]
        );

        client
            .write_all(&tagged(
                b'Q',
                &query_payload(&format!("CREATE TABLE {table} (id int4)")),
            ))
            .await
            .unwrap();
        assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);
    }

    client.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(client);
    server.abort();
    let _ = server.await;
}

#[test]
fn blocking_backend_key_cancels_simple_and_extended_copy_and_recovers() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let engine = std::sync::Arc::new(SharedEngine::new());
        let cancellations = std::sync::Arc::new(CancellationRegistry::new());
        let (mut primary, _) = listener.accept().unwrap();
        let primary_engine = std::sync::Arc::clone(&engine);
        let primary_cancellations = std::sync::Arc::clone(&cancellations);
        let primary = std::thread::spawn(move || {
            handle_connection_registered(&mut primary, &primary_engine, &primary_cancellations)
                .unwrap();
        });
        for _ in 0..4 {
            let (mut cancel, _) = listener.accept().unwrap();
            handle_connection_registered(&mut cancel, &engine, &cancellations).unwrap();
        }
        primary.join().unwrap();
    });

    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let startup = read_messages(&mut client, 9);
    let (process_id, secret_key) = backend_key(&startup);

    // PostgreSQL treats cancellation while idle as a no-op; it must not poison the next query.
    send_cancel(address, process_id, secret_key);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE blocking_cancel_rows (id int4, name text)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY blocking_cancel_rows FROM STDIN WITH CSV"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 1), vec![b'G']);
    let mut wrong_key = secret_key;
    wrong_key[0] ^= 0x80;
    send_cancel(address, process_id, wrong_key);
    client
        .write_all(&tagged(b'd', b"1,must-not-publish\n"))
        .unwrap();
    send_cancel(address, process_id, secret_key);
    let cancelled = read_messages(&mut client, 2);
    assert_error_sqlstate(&cancelled, b"C57014\0");
    assert_eq!(cancelled[1].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("SELECT COUNT(*) FROM blocking_cancel_rows"),
        ))
        .unwrap();
    let empty = read_messages(&mut client, 4);
    assert_eq!(
        empty.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'Z']
    );
    assert!(
        empty[1].1.ends_with(b"0"),
        "cancelled COPY published rows: {empty:?}"
    );

    let mut extended_copy = Vec::new();
    extended_copy.extend(tagged(
        b'P',
        &parse_payload(
            "blocking_cancel_copy",
            "COPY blocking_cancel_rows FROM STDIN WITH CSV",
            &[],
        ),
    ));
    extended_copy.extend(tagged(
        b'B',
        &bind_payload("blocking_cancel_portal", "blocking_cancel_copy"),
    ));
    extended_copy.extend(tagged(b'E', &execute_payload("blocking_cancel_portal", 0)));
    client.write_all(&extended_copy).unwrap();
    assert_eq!(read_tags(&mut client, 3), vec![b'1', b'2', b'G']);
    send_cancel(address, process_id, secret_key);
    let extended_error = read_messages(&mut client, 1);
    assert_eq!(extended_error[0].0, b'E');
    assert!(extended_error[0]
        .1
        .windows(b"C57014\0".len())
        .any(|window| window == b"C57014\0"));
    client.write_all(&tagged(b'S', &[])).unwrap();
    assert_eq!(read_messages(&mut client, 1), vec![(b'Z', vec![b'I'])]);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("INSERT INTO blocking_cancel_rows VALUES (2, 'recovered')"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_backend_key_cancels_waiting_copy_and_recovers() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let engine = std::sync::Arc::new(SharedEngine::new());
    let executor = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
    let cancellations = std::sync::Arc::new(CancellationRegistry::new());
    let server_engine = std::sync::Arc::clone(&engine);
    let server_executor = std::sync::Arc::clone(&executor);
    let server_cancellations = std::sync::Arc::clone(&cancellations);
    let server = tokio::spawn(async move {
        let (primary, _) = listener.accept().await.unwrap();
        let primary_engine = std::sync::Arc::clone(&server_engine);
        let primary_executor = std::sync::Arc::clone(&server_executor);
        let primary_cancellations = std::sync::Arc::clone(&server_cancellations);
        let primary = tokio::spawn(async move {
            handle_connection_async(
                primary,
                &primary_engine,
                &primary_executor,
                None,
                &primary_cancellations,
            )
            .await
            .unwrap();
        });
        for _ in 0..3 {
            let (cancel, _) = listener.accept().await.unwrap();
            handle_connection_async(
                cancel,
                &server_engine,
                &server_executor,
                None,
                &server_cancellations,
            )
            .await
            .unwrap();
        }
        primary.await.unwrap();
    });
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client.write_all(&startup_frame()).await.unwrap();
    let startup = read_messages_async(&mut client, 9).await;
    let (process_id, secret_key) = backend_key(&startup);

    send_cancel_async(address, process_id, secret_key).await;
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_cancel_rows (id int4, name text)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY async_cancel_rows FROM STDIN WITH CSV"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 1).await, vec![b'G']);
    let mut wrong_key = secret_key;
    wrong_key[3] ^= 0x01;
    send_cancel_async(address, process_id, wrong_key).await;
    let held = executor.acquire().await.unwrap();
    let mut queued_copy = tagged(b'd', b"1,must-not-publish\n");
    queued_copy.extend(tagged(b'd', b"2,must-not-publish-either\n"));
    queued_copy.extend(tagged(b'c', &[]));
    client.write_all(&queued_copy).await.unwrap();
    send_cancel_async(address, process_id, secret_key).await;
    drop(held);
    let cancelled = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&cancelled, b"C57014\0");
    assert_eq!(cancelled[1].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("SELECT COUNT(*) FROM async_cancel_rows"),
        ))
        .await
        .unwrap();
    let empty = tokio::time::timeout(Duration::from_secs(2), read_messages_async(&mut client, 4))
        .await
        .expect("queued COPY frames poisoned the following count request");
    assert_eq!(
        empty.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'Z']
    );
    assert!(
        empty[1].1.ends_with(b"0"),
        "cancelled async COPY published rows: {empty:?}"
    );

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("INSERT INTO async_cancel_rows VALUES (1, 'recovered')"),
        ))
        .await
        .unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), read_tags_async(&mut client, 2),)
            .await
            .expect("queued COPY frames poisoned same-key connection reuse"),
        vec![b'C', b'Z']
    );
    client.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(client);
    server.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_queued_parse_cancels_before_metadata_and_recovers_at_sync() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let engine = std::sync::Arc::new(SharedEngine::new());
    let executor = std::sync::Arc::new(tokio::sync::Semaphore::new(0));
    let cancellations = std::sync::Arc::new(CancellationRegistry::new());
    let server_engine = std::sync::Arc::clone(&engine);
    let server_executor = std::sync::Arc::clone(&executor);
    let server_cancellations = std::sync::Arc::clone(&cancellations);
    let server = tokio::spawn(async move {
        let (primary, _) = listener.accept().await.unwrap();
        let primary_engine = std::sync::Arc::clone(&server_engine);
        let primary_executor = std::sync::Arc::clone(&server_executor);
        let primary_cancellations = std::sync::Arc::clone(&server_cancellations);
        let primary = tokio::spawn(async move {
            handle_connection_async(
                primary,
                &primary_engine,
                &primary_executor,
                None,
                &primary_cancellations,
            )
            .await
            .unwrap();
        });
        let (cancel, _) = listener.accept().await.unwrap();
        handle_connection_async(
            cancel,
            &server_engine,
            &server_executor,
            None,
            &server_cancellations,
        )
        .await
        .unwrap();
        primary.await.unwrap();
    });

    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client.write_all(&startup_frame()).await.unwrap();
    let startup = read_messages_async(&mut client, 9).await;
    let (process_id, secret_key) = backend_key(&startup);
    client
        .write_all(&tagged(
            b'P',
            &parse_payload("queued_parse", "SELECT 1 AS value", &[]),
        ))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while !cancellations.request_is_active(process_id) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("Parse never entered its active request");
    send_cancel_async(address, process_id, secret_key).await;
    executor.add_permits(1);

    let error = read_messages_async(&mut client, 1).await;
    assert_eq!(error[0].0, b'E');
    assert!(error[0]
        .1
        .windows(b"C57014\0".len())
        .any(|window| window == b"C57014\0"));
    client.write_all(&tagged(b'S', &[])).await.unwrap();
    assert_eq!(
        read_messages_async(&mut client, 1).await,
        vec![(b'Z', vec![b'I'])],
        "cancelled queued Parse leaked its synthetic implicit transaction"
    );
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE queued_parse_recovery (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);
    client.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(client);
    server.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_cancelled_queued_extended_sync_rolls_back_staged_insert() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let engine = std::sync::Arc::new(SharedEngine::new());
    let executor = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
    let cancellations = std::sync::Arc::new(CancellationRegistry::new());
    let server_engine = std::sync::Arc::clone(&engine);
    let server_executor = std::sync::Arc::clone(&executor);
    let server_cancellations = std::sync::Arc::clone(&cancellations);
    let server = tokio::spawn(async move {
        let (primary, _) = listener.accept().await.unwrap();
        let primary_engine = std::sync::Arc::clone(&server_engine);
        let primary_executor = std::sync::Arc::clone(&server_executor);
        let primary_cancellations = std::sync::Arc::clone(&server_cancellations);
        let primary = tokio::spawn(async move {
            handle_connection_async(
                primary,
                &primary_engine,
                &primary_executor,
                None,
                &primary_cancellations,
            )
            .await
            .unwrap();
        });
        let (cancel, _) = listener.accept().await.unwrap();
        handle_connection_async(
            cancel,
            &server_engine,
            &server_executor,
            None,
            &server_cancellations,
        )
        .await
        .unwrap();
        primary.await.unwrap();
    });

    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client.write_all(&startup_frame()).await.unwrap();
    let startup = read_messages_async(&mut client, 9).await;
    let (process_id, secret_key) = backend_key(&startup);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE queued_sync_commit (id int4 PRIMARY KEY)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    let mut staged = tagged(
        b'P',
        &parse_payload(
            "queued_sync_insert",
            "INSERT INTO queued_sync_commit VALUES (1)",
            &[],
        ),
    );
    staged.extend(tagged(
        b'B',
        &bind_payload("queued_sync_portal", "queued_sync_insert"),
    ));
    staged.extend(tagged(b'E', &execute_payload("queued_sync_portal", 0)));
    client.write_all(&staged).await.unwrap();
    assert_eq!(
        read_tags_async(&mut client, 3).await,
        vec![b'1', b'2', b'C']
    );

    let held = executor.acquire().await.unwrap();
    client.write_all(&tagged(b'S', &[])).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while !cancellations.request_is_active(process_id) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("Sync never entered its active request");
    send_cancel_async(address, process_id, secret_key).await;
    drop(held);

    let cancelled = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&cancelled, b"C57014\0");
    assert_eq!(cancelled[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("SELECT COUNT(*) FROM queued_sync_commit"),
        ))
        .await
        .unwrap();
    let count = tokio::time::timeout(Duration::from_secs(2), read_messages_async(&mut client, 4))
        .await
        .expect("cancelled queued Sync poisoned the following count request");
    assert!(
        count[1].1.ends_with(b"0"),
        "cancelled queued Sync published its staged row: {count:?}"
    );
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("INSERT INTO queued_sync_commit VALUES (1)"),
        ))
        .await
        .unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), read_tags_async(&mut client, 2),)
            .await
            .expect("cancelled queued Sync poisoned same-key connection reuse"),
        vec![b'C', b'Z']
    );

    client.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(client);
    server.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_extended_copy_cancelled_before_queued_done_frame_recovers_at_sync() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let engine = std::sync::Arc::new(SharedEngine::new());
    let executor = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
    let cancellations = std::sync::Arc::new(CancellationRegistry::new());
    let server_engine = std::sync::Arc::clone(&engine);
    let server_executor = std::sync::Arc::clone(&executor);
    let server_cancellations = std::sync::Arc::clone(&cancellations);
    let server = tokio::spawn(async move {
        let (primary, _) = listener.accept().await.unwrap();
        let primary_engine = std::sync::Arc::clone(&server_engine);
        let primary_executor = std::sync::Arc::clone(&server_executor);
        let primary_cancellations = std::sync::Arc::clone(&server_cancellations);
        let primary = tokio::spawn(async move {
            handle_connection_async(
                primary,
                &primary_engine,
                &primary_executor,
                None,
                &primary_cancellations,
            )
            .await
            .unwrap();
        });
        let (cancel, _) = listener.accept().await.unwrap();
        handle_connection_async(
            cancel,
            &server_engine,
            &server_executor,
            None,
            &server_cancellations,
        )
        .await
        .unwrap();
        primary.await.unwrap();
    });

    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client.write_all(&startup_frame()).await.unwrap();
    let startup = read_messages_async(&mut client, 9).await;
    let (process_id, secret_key) = backend_key(&startup);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE queued_extended_copy (id int4 PRIMARY KEY, name text)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    let mut extended_copy = Vec::new();
    extended_copy.extend(tagged(
        b'P',
        &parse_payload(
            "queued_extended_copy_stmt",
            "COPY queued_extended_copy FROM STDIN WITH CSV",
            &[],
        ),
    ));
    extended_copy.extend(tagged(
        b'B',
        &bind_payload("queued_extended_copy_portal", "queued_extended_copy_stmt"),
    ));
    extended_copy.extend(tagged(
        b'E',
        &execute_payload("queued_extended_copy_portal", 0),
    ));
    client.write_all(&extended_copy).await.unwrap();
    assert_eq!(
        read_tags_async(&mut client, 3).await,
        vec![b'1', b'2', b'G']
    );

    // This raw-wire case deliberately proves cancellation/recovery while CopyDone itself is
    // queued for frame parsing. `copy::tests::copy_done_cancelled_while_queued_never_crosses_the_facade`
    // separately enters the finish helper with a buffered row and a zero-permit executor.
    let held = executor.acquire().await.unwrap();
    client.write_all(&tagged(b'c', &[])).await.unwrap();
    tokio::task::yield_now().await;
    send_cancel_async(address, process_id, secret_key).await;
    drop(held);
    let error = read_messages_async(&mut client, 1).await;
    assert_eq!(error[0].0, b'E');
    assert!(error[0]
        .1
        .windows(b"C57014\0".len())
        .any(|window| window == b"C57014\0"));
    client.write_all(&tagged(b'S', &[])).await.unwrap();
    assert_eq!(
        read_messages_async(&mut client, 1).await,
        vec![(b'Z', vec![b'I'])]
    );

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("SELECT COUNT(*) FROM queued_extended_copy"),
        ))
        .await
        .unwrap();
    let count = read_messages_async(&mut client, 4).await;
    assert!(
        count[1].1.ends_with(b"0"),
        "cancelled queued extended COPY published rows: {count:?}"
    );
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("INSERT INTO queued_extended_copy VALUES (1, 'recovered')"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);
    client.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(client);
    server.await.unwrap();
}

#[test]
fn blocking_multi_statement_simple_query_is_atomic_and_emits_one_ready() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    // Query is a transaction boundary even when its SQL consists only of comments. A preceding
    // Parse has opened the synthetic extended transaction; the comment-only Query must close it
    // and report idle rather than leak it until a later Sync.
    client
        .write_all(&tagged(
            b'P',
            &parse_payload("pending_comment_only", "BEGIN", &[]),
        ))
        .unwrap();
    assert_eq!(read_messages(&mut client, 1), vec![(b'1', Vec::new())]);
    client
        .write_all(&tagged(b'Q', &query_payload("/* comment boundary */")))
        .unwrap();
    assert_eq!(
        read_messages(&mut client, 2),
        vec![(b'I', Vec::new()), (b'Z', vec![b'I'])]
    );

    // Whole-message syntax preflight happens before execution and is also a Query/Sync boundary
    // for an already-open extended implicit cycle.
    client
        .write_all(&tagged(
            b'P',
            &parse_payload("pending_syntax_error", "BEGIN", &[]),
        ))
        .unwrap();
    assert_eq!(read_messages(&mut client, 1), vec![(b'1', Vec::new())]);
    client
        .write_all(&tagged(b'Q', &query_payload("SELECT 1; SELCT broken")))
        .unwrap();
    let pending_syntax = read_messages(&mut client, 2);
    assert_error_sqlstate(&pending_syntax, b"C42601\0");
    assert_eq!(pending_syntax[1].1, vec![b'I']);

    // COPY's protocol classification must not outrank whole-message syntax analysis. The invalid
    // ordinary span wins with 42601, and the prefix DDL is never executed.
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPYfoo blocking_single_copy_prefix"),
        ))
        .unwrap();
    let single_copy_prefix = read_messages(&mut client, 2);
    assert_error_sqlstate(&single_copy_prefix, b"C42601\0");
    assert_eq!(single_copy_prefix[1].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE blocking_copy_preflight_must_not_publish (id int4); \
                 COPYfoo blocking_copy_preflight_must_not_publish FROM STDIN; \
                 COPY blocking_copy_preflight_must_not_publish FROM STDIN",
            ),
        ))
        .unwrap();
    let copy_syntax = read_messages(&mut client, 2);
    assert_error_sqlstate(&copy_syntax, b"C42601\0");
    assert_eq!(copy_syntax[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE blocking_copy_preflight_must_not_publish (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("-- comment only\r/* nested /* block */ comment */"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'I', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "-- leading line\r/* leading block */ \
                 CREATE TABLE trailing_line_comment_commits (id int4); -- trailing line",
            ),
        ))
        .unwrap();
    let trailing_block = read_messages(&mut client, 2);
    assert_eq!(
        trailing_block
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'C', b'Z'],
        "unexpected trailing-comment response: {trailing_block:?}"
    );

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE/* comment is token whitespace */TABLE blocking_comment_spacing (id int4)",
            ),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    // Syntax analysis covers the complete Query before an explicit exit can publish or roll back.
    // An existing explicit transaction becomes failed, keeps its staged work private, and requires
    // a later standalone ROLLBACK.
    client
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'T']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE blocking_preflight_commit (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'T']);
    client
        .write_all(&tagged(b'Q', &query_payload("COMMIT; SELCT broken")))
        .unwrap();
    let commit_syntax = read_messages(&mut client, 2);
    assert_error_sqlstate(&commit_syntax, b"C42601\0");
    assert_eq!(commit_syntax[1].1, vec![b'E']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE blocking_preflight_commit (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'T']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK; SELCT broken")))
        .unwrap();
    let rollback_syntax = read_messages(&mut client, 2);
    assert_error_sqlstate(&rollback_syntax, b"C42601\0");
    assert_eq!(rollback_syntax[1].1, vec![b'E']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'I']);

    // An explicit BEGIN owns the implicit multi-statement segment's exact characteristics. A
    // default synthetic BEGIN must neither hide unsupported isolation nor erase supported READ ONLY.
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("BEGIN ISOLATION LEVEL SERIALIZABLE; COMMIT"),
        ))
        .unwrap();
    let serializable = read_messages(&mut client, 2);
    assert_error_sqlstate(&serializable, b"C0A000\0");
    assert_eq!(serializable[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "BEGIN READ ONLY; \
                 CREATE TABLE blocking_read_only_must_not_publish (id int4); \
                 COMMIT",
            ),
        ))
        .unwrap();
    let read_only = read_messages(&mut client, 3);
    assert_eq!(
        read_only.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'C', b'E', b'Z']
    );
    assert!(read_only[1]
        .1
        .windows(b"C0A000\0".len())
        .any(|window| window == b"C0A000\0"));
    assert_eq!(read_only[2].1, vec![b'E']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE blocking_read_only_must_not_publish (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    // Explicit COMMIT divides a Query string. The two statements after it form a fresh implicit
    // segment, so the missing-relation error rolls back the first post-COMMIT CREATE while the
    // relation committed before the boundary remains published.
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "BEGIN; \
                 CREATE TABLE blocking_segment_committed (id int4); \
                 COMMIT; \
                 CREATE TABLE blocking_segment_rolled_back (id int4); \
                 INSERT INTO missing_blocking_segment VALUES (1)",
            ),
        ))
        .unwrap();
    let segmented = read_messages(&mut client, 6);
    assert_eq!(
        segmented.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'C', b'C', b'C', b'C', b'E', b'Z']
    );
    assert_eq!(segmented[5].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE blocking_segment_rolled_back (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("DROP TABLE blocking_segment_committed"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE trailing_block_comment_commits (id int4) \
                 /* trailing block */; /* comment-only tail */",
            ),
        ))
        .unwrap();
    let trailing_block_after_segment = read_messages(&mut client, 2);
    assert_eq!(
        trailing_block_after_segment
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'C', b'Z'],
        "unexpected post-segment response: {trailing_block_after_segment:?}"
    );
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "/* before BEGIN */ BEGIN; -- before first INSERT\r \
                 INSERT INTO trailing_line_comment_commits VALUES (1); \
                 /* before second INSERT */ \
                 INSERT INTO trailing_block_comment_commits VALUES (2); \
                 /* before COMMIT */ COMMIT; -- after COMMIT",
            ),
        ))
        .unwrap();
    assert_eq!(
        read_tags(&mut client, 5),
        vec![b'C', b'C', b'C', b'C', b'Z']
    );

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE unterminated_comment_rolled_back (id int4); \
                 /* unterminated false COMMIT;",
            ),
        ))
        .unwrap();
    let unterminated = read_messages(&mut client, 2);
    assert_eq!(
        unterminated.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(unterminated[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE unterminated_comment_rolled_back (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE simple_group (id int4); \
                 INSERT INTO simple_group (id) VALUES (7); \
                 SELECT id FROM simple_group",
            ),
        ))
        .unwrap();
    assert_eq!(
        read_tags(&mut client, 6),
        vec![b'C', b'C', b'T', b'D', b'C', b'Z']
    );

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE simple_group_rolled_back (id int4); \
                 INSERT INTO missing_simple_group (id) VALUES (1); \
                 CREATE TABLE must_not_run (id int4)",
            ),
        ))
        .unwrap();
    let failed = read_messages(&mut client, 3);
    assert_eq!(
        failed.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'C', b'E', b'Z']
    );
    assert_eq!(failed[2].1, vec![b'I']);

    // Reusing the first name and creating the skipped third name prove there was neither partial
    // publication nor execution after the failing statement. Keep these as distinct autocommit
    // messages so the assertion is about the failed group, not multi-DDL overlay support.
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE simple_group_rolled_back (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE must_not_run (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    // An escape-string quote must not manufacture the apparent standalone COMMIT below. If it
    // did, transaction-control classification would disable the group wrapper and publish this
    // CREATE before the deliberately invalid second statement failed.
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                r"CREATE TABLE escape_false_commit_rolled_back (id int4); \
                  INVALID E'x\'; COMMIT; y'; \
                  CREATE TABLE escape_false_commit_successor (id int4)",
            ),
        ))
        .unwrap();
    let escape_failed = read_messages(&mut client, 2);
    assert_eq!(
        escape_failed
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(escape_failed[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE escape_false_commit_rolled_back (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE escape_false_commit_successor (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_multi_statement_simple_query_is_atomic_and_emits_one_ready() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_async_with_engine_batching(
        listener,
        std::sync::Arc::new(SharedEngine::new()),
        2,
        true,
    ));
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client.write_all(&startup_frame()).await.unwrap();
    let _ = read_messages_async(&mut client, 9).await;

    client
        .write_all(&tagged(
            b'P',
            &parse_payload("async_pending_comment_only", "BEGIN", &[]),
        ))
        .await
        .unwrap();
    assert_eq!(
        read_messages_async(&mut client, 1).await,
        vec![(b'1', Vec::new())]
    );
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("/* async comment boundary */"),
        ))
        .await
        .unwrap();
    assert_eq!(
        read_messages_async(&mut client, 2).await,
        vec![(b'I', Vec::new()), (b'Z', vec![b'I'])]
    );

    client
        .write_all(&tagged(
            b'P',
            &parse_payload("async_pending_syntax_error", "BEGIN", &[]),
        ))
        .await
        .unwrap();
    assert_eq!(
        read_messages_async(&mut client, 1).await,
        vec![(b'1', Vec::new())]
    );
    client
        .write_all(&tagged(b'Q', &query_payload("SELECT 1; SELCT broken")))
        .await
        .unwrap();
    let pending_syntax = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&pending_syntax, b"C42601\0");
    assert_eq!(pending_syntax[1].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPYfoo async_single_copy_prefix"),
        ))
        .await
        .unwrap();
    let single_copy_prefix = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&single_copy_prefix, b"C42601\0");
    assert_eq!(single_copy_prefix[1].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE async_copy_preflight_must_not_publish (id int4); \
                 COPYfoo async_copy_preflight_must_not_publish FROM STDIN; \
                 COPY async_copy_preflight_must_not_publish FROM STDIN",
            ),
        ))
        .await
        .unwrap();
    let copy_syntax = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&copy_syntax, b"C42601\0");
    assert_eq!(copy_syntax[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_copy_preflight_must_not_publish (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("/* async comment only */ -- line only\n"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'I', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "-- async leading\r/* block */ \
                 CREATE TABLE async_trailing_line_commits (id int4); -- trailing",
            ),
        ))
        .await
        .unwrap();
    let trailing_block = read_messages_async(&mut client, 2).await;
    assert_eq!(
        trailing_block
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'C', b'Z'],
        "unexpected trailing-comment response: {trailing_block:?}"
    );

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE/* async token whitespace */TABLE async_comment_spacing (id int4)",
            ),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .await
        .unwrap();
    assert_eq!(read_messages_async(&mut client, 2).await[1].1, vec![b'T']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_preflight_commit (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_messages_async(&mut client, 2).await[1].1, vec![b'T']);
    client
        .write_all(&tagged(b'Q', &query_payload("COMMIT; SELCT broken")))
        .await
        .unwrap();
    let commit_syntax = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&commit_syntax, b"C42601\0");
    assert_eq!(commit_syntax[1].1, vec![b'E']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .await
        .unwrap();
    assert_eq!(read_messages_async(&mut client, 2).await[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_preflight_commit (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .await
        .unwrap();
    assert_eq!(read_messages_async(&mut client, 2).await[1].1, vec![b'T']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK; SELCT broken")))
        .await
        .unwrap();
    let rollback_syntax = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&rollback_syntax, b"C42601\0");
    assert_eq!(rollback_syntax[1].1, vec![b'E']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .await
        .unwrap();
    assert_eq!(read_messages_async(&mut client, 2).await[1].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("BEGIN ISOLATION LEVEL SERIALIZABLE; COMMIT"),
        ))
        .await
        .unwrap();
    let serializable = read_messages_async(&mut client, 2).await;
    assert_error_sqlstate(&serializable, b"C0A000\0");
    assert_eq!(serializable[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "BEGIN READ ONLY; \
                 CREATE TABLE async_read_only_must_not_publish (id int4); \
                 COMMIT",
            ),
        ))
        .await
        .unwrap();
    let read_only = read_messages_async(&mut client, 3).await;
    assert_eq!(
        read_only.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'C', b'E', b'Z']
    );
    assert!(read_only[1]
        .1
        .windows(b"C0A000\0".len())
        .any(|window| window == b"C0A000\0"));
    assert_eq!(read_only[2].1, vec![b'E']);
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .await
        .unwrap();
    assert_eq!(read_messages_async(&mut client, 2).await[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_read_only_must_not_publish (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "BEGIN; \
                 CREATE TABLE async_segment_committed (id int4); \
                 COMMIT; \
                 CREATE TABLE async_segment_rolled_back (id int4); \
                 INSERT INTO missing_async_segment VALUES (1)",
            ),
        ))
        .await
        .unwrap();
    let segmented = read_messages_async(&mut client, 6).await;
    assert_eq!(
        segmented.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'C', b'C', b'C', b'C', b'E', b'Z']
    );
    assert_eq!(segmented[5].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_segment_rolled_back (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("DROP TABLE async_segment_committed"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE async_trailing_block_commits (id int4) \
                 /* trailing */; /* tail */",
            ),
        ))
        .await
        .unwrap();
    let trailing_block_after_segment = read_messages_async(&mut client, 2).await;
    assert_eq!(
        trailing_block_after_segment
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'C', b'Z'],
        "unexpected post-segment response: {trailing_block_after_segment:?}"
    );
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "/* begin */ BEGIN; \
                 INSERT INTO async_trailing_line_commits VALUES (1); \
                 -- before second insert\r \
                 INSERT INTO async_trailing_block_commits VALUES (2); \
                 /* commit */ COMMIT; -- done",
            ),
        ))
        .await
        .unwrap();
    assert_eq!(
        read_tags_async(&mut client, 5).await,
        vec![b'C', b'C', b'C', b'C', b'Z']
    );

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE async_unterminated_comment (id int4); \
                 /* unterminated false COMMIT;",
            ),
        ))
        .await
        .unwrap();
    let unterminated = read_messages_async(&mut client, 2).await;
    assert_eq!(
        unterminated.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(unterminated[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_unterminated_comment (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE async_group_rolled_back (id int4); \
                 INSERT INTO missing_async_group (id) VALUES (1)",
            ),
        ))
        .await
        .unwrap();
    let failed = read_messages_async(&mut client, 3).await;
    assert_eq!(
        failed.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'C', b'E', b'Z']
    );
    assert_eq!(failed[2].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_group_rolled_back (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                r"CREATE TABLE async_escape_false_commit (id int4); \
                  INVALID E'x\'; COMMIT; y'; \
                  CREATE TABLE async_escape_successor (id int4)",
            ),
        ))
        .await
        .unwrap();
    let escape_failed = read_messages_async(&mut client, 2).await;
    assert_eq!(
        escape_failed
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(escape_failed[1].1, vec![b'I']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_escape_false_commit (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_escape_successor (id int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    client.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(client);
    server.abort();
    let _ = server.await;
}

#[test]
fn pgwire_extended_lifecycle_preserves_transaction_status_and_skip_until_sync() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let startup = read_messages(&mut client, 9);
    assert_eq!(
        startup.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'R', b'S', b'S', b'S', b'S', b'S', b'S', b'K', b'Z']
    );
    assert_eq!(
        startup[1..7]
            .iter()
            .map(|(_, payload)| payload.as_slice())
            .collect::<Vec<_>>(),
        vec![
            b"server_version\x0016.0-gpu-db-engine-facade\0".as_slice(),
            b"server_version_num\x00160000\0".as_slice(),
            b"client_encoding\0UTF8\0".as_slice(),
            b"DateStyle\0ISO, MDY\0".as_slice(),
            b"integer_datetimes\0on\0".as_slice(),
            b"standard_conforming_strings\0on\0".as_slice(),
        ]
    );

    let mut begin = Vec::new();
    begin.extend(tagged(b'P', &parse_payload("begin", "BEGIN", &[])));
    begin.extend(tagged(b'B', &bind_payload("begin_portal", "begin")));
    begin.extend(tagged(b'D', &describe_payload(b'S', "begin")));
    begin.extend(tagged(b'E', &execute_payload("begin_portal", 0)));
    begin.extend(tagged(b'S', &[]));
    client.write_all(&begin).unwrap();
    let begin_messages = read_messages(&mut client, 6);
    assert_eq!(
        begin_messages
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'1', b'2', b't', b'n', b'C', b'Z']
    );
    assert_eq!(begin_messages.last().unwrap().1, vec![b'T']);

    // A duplicate named Parse errors and a malformed queued Bind is discarded without a second
    // ErrorResponse. The terminating Sync still has to decode: its malformed payload adds a
    // protocol ErrorResponse before ReadyForQuery while retaining the failed explicit status.
    let mut failed = Vec::new();
    failed.extend(tagged(b'P', &parse_payload("begin", "BEGIN", &[])));
    failed.extend(tagged(
        b'B',
        &malformed_bind_shape_payload("ignored", "begin"),
    ));
    failed.extend(tagged(b'S', &[99]));
    client.write_all(&failed).unwrap();
    let failed_messages = read_messages(&mut client, 3);
    assert_eq!(failed_messages[0].0, b'E');
    assert_eq!(failed_messages[1].0, b'E');
    assert!(failed_messages[1]
        .1
        .windows(b"C08P01\0".len())
        .any(|window| window == b"C08P01\0"));
    assert_eq!(failed_messages[2], (b'Z', vec![b'E']));

    let mut commit = Vec::new();
    commit.extend(tagged(b'P', &parse_payload("commit", "COMMIT", &[])));
    commit.extend(tagged(b'B', &bind_payload("commit_portal", "commit")));
    commit.extend(tagged(b'E', &execute_payload("commit_portal", 0)));
    commit.extend(tagged(b'D', &describe_payload(b'P', "commit_portal")));
    commit.extend(tagged(b'S', &[]));
    client.write_all(&commit).unwrap();
    let commit_messages = read_messages(&mut client, 5);
    assert_eq!(
        commit_messages
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'1', b'2', b'C', b'E', b'Z']
    );
    assert_eq!(commit_messages[2].1, b"ROLLBACK\0");
    assert_eq!(commit_messages.last().unwrap().1, vec![b'I']);

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn raw_malformed_nonextended_frames_emit_error_and_ready_without_skip() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    // Parse opens the synthetic extended-cycle transaction. A malformed Query is not an
    // extended message, so it rolls that transaction back and emits ErrorResponse + Ready
    // immediately instead of entering ignore-until-Sync.
    client
        .write_all(&tagged(b'P', &parse_payload("pending", "BEGIN", &[])))
        .unwrap();
    assert_eq!(read_messages(&mut client, 1), vec![(b'1', Vec::new())]);
    client.write_all(&tagged(b'Q', b"SELECT 1")).unwrap();
    let malformed_query = read_messages(&mut client, 2);
    assert_eq!(
        malformed_query
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(malformed_query[1].1, vec![b'I']);

    // A following Query is processed immediately, proving there is no lingering skip state.
    client.write_all(&tagged(b'Q', &query_payload(""))).unwrap();
    assert_eq!(
        read_messages(&mut client, 2),
        vec![(b'I', Vec::new()), (b'Z', vec![b'I'])]
    );

    client
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .unwrap();
    let begin = read_messages(&mut client, 2);
    assert_eq!(begin[0], (b'C', b"BEGIN\0".to_vec()));
    assert_eq!(begin[1], (b'Z', vec![b'T']));

    // Sync is also non-extended for error recovery. Its malformed payload fails the explicit
    // transaction and emits Ready immediately with failed-transaction status.
    client.write_all(&tagged(b'S', &[99])).unwrap();
    let malformed_sync = read_messages(&mut client, 2);
    assert_eq!(
        malformed_sync
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(malformed_sync[1].1, vec![b'E']);

    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .unwrap();
    let rollback = read_messages(&mut client, 2);
    assert_eq!(rollback[0], (b'C', b"ROLLBACK\0".to_vec()));
    assert_eq!(rollback[1], (b'Z', vec![b'I']));

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn raw_bind_resolves_missing_statement_before_parameter_format_arity() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    // Two parameter formats for zero supplied values is semantically malformed, but Bind
    // must resolve the named statement first. The frame parser therefore preserves it for the
    // canonical lifecycle owner, which reports missing statement (26000), not 08P01.
    let mut bind = Vec::new();
    push_cstring(&mut bind, "missing_portal");
    push_cstring(&mut bind, "missing_statement");
    bind.extend_from_slice(&2_i16.to_be_bytes());
    bind.extend_from_slice(&0_i16.to_be_bytes());
    bind.extend_from_slice(&1_i16.to_be_bytes());
    bind.extend_from_slice(&0_i16.to_be_bytes());
    bind.extend_from_slice(&0_i16.to_be_bytes());
    client.write_all(&tagged(b'B', &bind)).unwrap();
    let error = read_messages(&mut client, 1);
    assert_eq!(error[0].0, b'E');
    assert!(error[0]
        .1
        .windows(b"C26000\0".len())
        .any(|window| window == b"C26000\0"));

    client.write_all(&tagged(b'S', &[])).unwrap();
    assert_eq!(read_messages(&mut client, 1), vec![(b'Z', vec![b'I'])]);
    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn extended_ddl_commits_at_sync_and_rolls_back_with_a_later_cycle_error() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    assert_eq!(read_tags(&mut client, 9).last(), Some(&b'Z'));

    let mut ddl = Vec::new();
    ddl.extend(tagged(
        b'P',
        &parse_payload("ddl", "CREATE TABLE extended_ddl (id int4)", &[]),
    ));
    ddl.extend(tagged(b'B', &bind_payload("ddl_portal", "ddl")));
    ddl.extend(tagged(b'E', &execute_payload("ddl_portal", 0)));
    ddl.extend(tagged(b'S', &[]));
    client.write_all(&ddl).unwrap();
    let committed = read_messages(&mut client, 4);
    assert_eq!(
        committed.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'1', b'2', b'C', b'Z']
    );

    client
        .write_all(&tagged(b'Q', &query_payload("SELECT id FROM extended_ddl")))
        .unwrap();
    let selected = read_messages(&mut client, 3);
    assert_eq!(
        selected.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'C', b'Z']
    );

    let mut rolled_back = Vec::new();
    rolled_back.extend(tagged(
        b'P',
        &parse_payload("rolled_ddl", "CREATE TABLE rolled_ddl (id int4)", &[]),
    ));
    rolled_back.extend(tagged(b'B', &bind_payload("rolled_portal", "rolled_ddl")));
    rolled_back.extend(tagged(b'E', &execute_payload("rolled_portal", 0)));
    rolled_back.extend(tagged(b'E', &execute_payload("missing_portal", 0)));
    rolled_back.extend(tagged(b'S', &[]));
    client.write_all(&rolled_back).unwrap();
    assert_eq!(
        read_tags(&mut client, 5),
        vec![b'1', b'2', b'C', b'E', b'Z']
    );

    // Reusing the name proves the DDL result returned before the later Execute error remained
    // private and the implicit transaction rolled it back at Sync.
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE rolled_ddl (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn extended_describe_revalidates_catalog_before_emitting_cached_metadata() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    assert_eq!(read_tags(&mut client, 9).last(), Some(&b'Z'));

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE describe_shape (x int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    let mut parse = Vec::new();
    parse.extend(tagged(
        b'P',
        &parse_payload("shape", "SELECT x FROM describe_shape", &[]),
    ));
    parse.extend(tagged(b'S', &[]));
    client.write_all(&parse).unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'1', b'Z']);

    client
        .write_all(&tagged(b'Q', &query_payload("DROP TABLE describe_shape")))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE describe_shape (x text)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    let mut describe = Vec::new();
    describe.extend(tagged(b'D', &describe_payload(b'S', "shape")));
    describe.extend(tagged(b'S', &[]));
    client.write_all(&describe).unwrap();
    let rejected = read_messages(&mut client, 2);
    assert_eq!(
        rejected.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert!(rejected[0]
        .1
        .windows(b"C0A000\0".len())
        .any(|window| window == b"C0A000\0"));

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_extended_describe_revalidates_catalog_before_emitting_cached_metadata() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_async_with_engine_batching(
        listener,
        std::sync::Arc::new(SharedEngine::new()),
        2,
        false,
    ));
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client.write_all(&startup_frame()).await.unwrap();
    assert_eq!(read_tags_async(&mut client, 9).await.last(), Some(&b'Z'));

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_describe_shape (x int4)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    let mut parse = Vec::new();
    parse.extend(tagged(
        b'P',
        &parse_payload("async_shape", "SELECT x FROM async_describe_shape", &[]),
    ));
    parse.extend(tagged(b'S', &[]));
    client.write_all(&parse).await.unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'1', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("DROP TABLE async_describe_shape"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_describe_shape (x text)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut client, 2).await, vec![b'C', b'Z']);

    let mut describe = Vec::new();
    describe.extend(tagged(b'D', &describe_payload(b'S', "async_shape")));
    describe.extend(tagged(b'S', &[]));
    client.write_all(&describe).await.unwrap();
    let rejected = read_messages_async(&mut client, 2).await;
    assert_eq!(
        rejected.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert!(rejected[0]
        .1
        .windows(b"C0A000\0".len())
        .any(|window| window == b"C0A000\0"));

    client.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(client);
    server.abort();
    let _ = server.await;
}

#[test]
fn blocking_parse_and_describe_use_only_the_connection_private_catalog() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let engine = std::sync::Arc::new(SharedEngine::new());
        let mut workers = Vec::new();
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let engine = std::sync::Arc::clone(&engine);
            workers.push(std::thread::spawn(move || {
                handle_connection(&mut stream, &engine).unwrap();
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let mut creator = TcpStream::connect(address).unwrap();
    let mut observer = TcpStream::connect(address).unwrap();
    for client in [&creator, &observer] {
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
    }
    creator.write_all(&startup_frame()).unwrap();
    observer.write_all(&startup_frame()).unwrap();
    assert_eq!(read_tags(&mut creator, 9).last(), Some(&b'Z'));
    assert_eq!(read_tags(&mut observer, 9).last(), Some(&b'Z'));

    creator
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .unwrap();
    assert_eq!(read_tags(&mut creator, 2), vec![b'C', b'Z']);
    creator
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE blocking_private (id int4, value text)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut creator, 2), vec![b'C', b'Z']);

    let mut private_parse = Vec::new();
    private_parse.extend(tagged(
        b'P',
        &parse_payload(
            "private_insert",
            "INSERT INTO blocking_private VALUES ($1, $2) RETURNING id, value",
            &[],
        ),
    ));
    private_parse.extend(tagged(b'D', &describe_payload(b'S', "private_insert")));
    private_parse.extend(tagged(b'S', &[]));
    creator.write_all(&private_parse).unwrap();
    assert_private_parse_description(&read_messages(&mut creator, 4));

    let mut hidden_parse = Vec::new();
    hidden_parse.extend(tagged(
        b'P',
        &parse_payload(
            "hidden_insert",
            "INSERT INTO blocking_private VALUES ($1, $2) RETURNING id, value",
            &[],
        ),
    ));
    hidden_parse.extend(tagged(b'S', &[]));
    observer.write_all(&hidden_parse).unwrap();
    assert_error_sqlstate(&read_messages(&mut observer, 2), b"C42P01\0");

    let mut rowless_parse = Vec::new();
    rowless_parse.extend(tagged(
        b'P',
        &parse_payload(
            "private_rowless",
            "INSERT INTO blocking_private VALUES ($1, $2)",
            &[],
        ),
    ));
    rowless_parse.extend(tagged(b'S', &[]));
    creator.write_all(&rowless_parse).unwrap();
    assert_eq!(read_tags(&mut creator, 2), vec![b'1', b'Z']);
    creator
        .write_all(&tagged(b'Q', &query_payload("not valid sql")))
        .unwrap();
    let failed = read_messages(&mut creator, 2);
    assert_eq!(
        failed.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(failed[1].1, vec![b'E']);
    let mut cached_describe = Vec::new();
    cached_describe.extend(tagged(b'D', &describe_payload(b'S', "private_rowless")));
    cached_describe.extend(tagged(b'S', &[]));
    creator.write_all(&cached_describe).unwrap();
    assert_failed_cached_rowless_description(&read_messages(&mut creator, 3));

    creator
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .unwrap();
    assert_eq!(read_tags(&mut creator, 2), vec![b'C', b'Z']);
    let mut invalidated = Vec::new();
    invalidated.extend(tagged(b'D', &describe_payload(b'S', "private_insert")));
    invalidated.extend(tagged(b'S', &[]));
    creator.write_all(&invalidated).unwrap();
    assert_error_sqlstate(&read_messages(&mut creator, 2), b"C42P01\0");

    creator.write_all(&tagged(b'X', &[])).unwrap();
    observer.write_all(&tagged(b'X', &[])).unwrap();
    drop(creator);
    drop(observer);
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_parse_and_describe_use_only_the_connection_private_catalog() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_async_with_engine_batching(
        listener,
        std::sync::Arc::new(SharedEngine::new()),
        2,
        false,
    ));
    let mut creator = tokio::net::TcpStream::connect(address).await.unwrap();
    let mut observer = tokio::net::TcpStream::connect(address).await.unwrap();
    creator.write_all(&startup_frame()).await.unwrap();
    observer.write_all(&startup_frame()).await.unwrap();
    assert_eq!(read_tags_async(&mut creator, 9).await.last(), Some(&b'Z'));
    assert_eq!(read_tags_async(&mut observer, 9).await.last(), Some(&b'Z'));

    creator
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut creator, 2).await, vec![b'C', b'Z']);
    creator
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE async_private (id int4, value text)"),
        ))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut creator, 2).await, vec![b'C', b'Z']);

    let mut private_parse = Vec::new();
    private_parse.extend(tagged(
        b'P',
        &parse_payload(
            "private_insert",
            "INSERT INTO async_private VALUES ($1, $2) RETURNING id, value",
            &[],
        ),
    ));
    private_parse.extend(tagged(b'D', &describe_payload(b'S', "private_insert")));
    private_parse.extend(tagged(b'S', &[]));
    creator.write_all(&private_parse).await.unwrap();
    assert_private_parse_description(&read_messages_async(&mut creator, 4).await);

    let mut hidden_parse = Vec::new();
    hidden_parse.extend(tagged(
        b'P',
        &parse_payload(
            "hidden_insert",
            "INSERT INTO async_private VALUES ($1, $2) RETURNING id, value",
            &[],
        ),
    ));
    hidden_parse.extend(tagged(b'S', &[]));
    observer.write_all(&hidden_parse).await.unwrap();
    assert_error_sqlstate(&read_messages_async(&mut observer, 2).await, b"C42P01\0");

    let mut rowless_parse = Vec::new();
    rowless_parse.extend(tagged(
        b'P',
        &parse_payload(
            "private_rowless",
            "INSERT INTO async_private VALUES ($1, $2)",
            &[],
        ),
    ));
    rowless_parse.extend(tagged(b'S', &[]));
    creator.write_all(&rowless_parse).await.unwrap();
    assert_eq!(read_tags_async(&mut creator, 2).await, vec![b'1', b'Z']);
    creator
        .write_all(&tagged(b'Q', &query_payload("not valid sql")))
        .await
        .unwrap();
    let failed = read_messages_async(&mut creator, 2).await;
    assert_eq!(
        failed.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_eq!(failed[1].1, vec![b'E']);
    let mut cached_describe = Vec::new();
    cached_describe.extend(tagged(b'D', &describe_payload(b'S', "private_rowless")));
    cached_describe.extend(tagged(b'S', &[]));
    creator.write_all(&cached_describe).await.unwrap();
    assert_failed_cached_rowless_description(&read_messages_async(&mut creator, 3).await);

    creator
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .await
        .unwrap();
    assert_eq!(read_tags_async(&mut creator, 2).await, vec![b'C', b'Z']);
    let mut invalidated = Vec::new();
    invalidated.extend(tagged(b'D', &describe_payload(b'S', "private_insert")));
    invalidated.extend(tagged(b'S', &[]));
    creator.write_all(&invalidated).await.unwrap();
    assert_error_sqlstate(&read_messages_async(&mut creator, 2).await, b"C42P01\0");

    creator.write_all(&tagged(b'X', &[])).await.unwrap();
    observer.write_all(&tagged(b'X', &[])).await.unwrap();
    drop(creator);
    drop(observer);
    server.abort();
    let _ = server.await;
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn simple_query_literal_gpu_path_preserves_typed_null_differential() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    // Both NULLs and their zero/empty controls traverse the same typed transient GPU relation.
    // The contrasting validity payloads prove NULL is not fabricated from the device placeholder.
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "SELECT NULL::int4 AS value; \
                 SELECT 0::int4 AS value; \
                 SELECT NULL::text AS value; \
                 SELECT ''::text AS value",
            ),
        ))
        .unwrap();
    let messages = read_messages(&mut client, 13);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'T', b'D', b'C', b'T', b'D', b'C', b'T', b'D', b'C', b'Z']
    );
    assert_eq!(messages[1].1, vec![0, 1, 255, 255, 255, 255]);
    assert_eq!(messages[4].1, vec![0, 1, 0, 0, 0, 1, b'0']);
    assert_eq!(messages[7].1, vec![0, 1, 255, 255, 255, 255]);
    assert_eq!(messages[10].1, vec![0, 1, 0, 0, 0, 0]);
    assert_eq!(messages[12].1, vec![b'I']);

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn simple_query_commit_and_rollback_end_the_pending_extended_cycle() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE cycle_tx (id int4 PRIMARY KEY)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    send_extended_insert(
        &mut client,
        "atomic_first",
        "INSERT INTO cycle_tx VALUES (3)",
    );
    let mut duplicate = Vec::new();
    duplicate.extend(tagged(
        b'P',
        &parse_payload("atomic_second", "INSERT INTO cycle_tx VALUES (3)", &[]),
    ));
    duplicate.extend(tagged(
        b'B',
        &bind_payload("atomic_second", "atomic_second"),
    ));
    duplicate.extend(tagged(b'E', &execute_payload("atomic_second", 0)));
    client.write_all(&duplicate).unwrap();
    let duplicate = read_messages(&mut client, 3);
    assert_eq!(
        duplicate.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'1', b'2', b'E']
    );
    assert!(duplicate[2]
        .1
        .windows(b"C23505\0".len())
        .any(|window| window == b"C23505\0"));
    client.write_all(&tagged(b'S', &[])).unwrap();
    assert_eq!(read_messages(&mut client, 1), vec![(b'Z', vec![b'I'])]);
    client
        .write_all(&tagged(b'Q', &query_payload("SELECT id FROM cycle_tx")))
        .unwrap();
    let atomic_empty = read_messages(&mut client, 3);
    assert_eq!(atomic_empty[1], (b'C', b"SELECT 0\0".to_vec()));

    send_extended_insert(&mut client, "rolled", "INSERT INTO cycle_tx VALUES (1)");
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .unwrap();
    let rollback = read_messages(&mut client, 2);
    assert_eq!(rollback[0], (b'C', b"ROLLBACK\0".to_vec()));
    assert_eq!(rollback[1], (b'Z', vec![b'I']));
    client
        .write_all(&tagged(b'Q', &query_payload("SELECT id FROM cycle_tx")))
        .unwrap();
    let empty = read_messages(&mut client, 3);
    assert_eq!(empty[1], (b'C', b"SELECT 0\0".to_vec()));

    send_extended_insert(&mut client, "committed", "INSERT INTO cycle_tx VALUES (2)");
    client
        .write_all(&tagged(b'Q', &query_payload("COMMIT")))
        .unwrap();
    let commit = read_messages(&mut client, 2);
    assert_eq!(commit[0], (b'C', b"COMMIT\0".to_vec()));
    assert_eq!(commit[1], (b'Z', vec![b'I']));
    client
        .write_all(&tagged(b'Q', &query_payload("SELECT id FROM cycle_tx")))
        .unwrap();
    let one = read_messages(&mut client, 4);
    assert_eq!(
        one.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'Z']
    );

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_drives_the_async_extended_ingress() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_async_with_engine_batching(
        listener,
        std::sync::Arc::new(SharedEngine::new()),
        2,
        false,
    ));
    let (client, connection) = tokio_postgres::connect(
        &format!(
            "host={} port={} user=postgres",
            address.ip(),
            address.port()
        ),
        tokio_postgres::NoTls,
    )
    .await
    .unwrap();
    let connection = tokio::spawn(async move { connection.await.unwrap() });

    let begin = client.prepare("BEGIN").await.unwrap();
    client.execute(&begin, &[]).await.unwrap();
    let commit = client.prepare("COMMIT").await.unwrap();
    client.execute(&commit, &[]).await.unwrap();

    drop(client);
    connection.await.unwrap();
    server.abort();
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_describes_transaction_private_view_and_forgets_it_on_rollback() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_async_with_engine_batching(
        listener,
        std::sync::Arc::new(SharedEngine::new()),
        2,
        false,
    ));
    let (client, connection) = tokio_postgres::connect(
        &format!(
            "host={} port={} user=postgres",
            address.ip(),
            address.port()
        ),
        tokio_postgres::NoTls,
    )
    .await
    .unwrap();
    let connection = tokio::spawn(async move { connection.await.unwrap() });

    client
        .batch_execute("CREATE TABLE prepared_view_source (id int4, note text)")
        .await
        .unwrap();
    client.batch_execute("BEGIN").await.unwrap();
    client
        .batch_execute(
            "CREATE VIEW prepared_private_view AS \
             SELECT id, note FROM prepared_view_source",
        )
        .await
        .unwrap();
    let statement = client
        .prepare("SELECT * FROM prepared_private_view")
        .await
        .unwrap();
    assert_eq!(
        statement
            .columns()
            .iter()
            .map(|column| (column.name(), column.type_().name()))
            .collect::<Vec<_>>(),
        vec![("id", "int4"), ("note", "text")]
    );

    client.batch_execute("ROLLBACK").await.unwrap();
    let error = client
        .prepare("SELECT * FROM prepared_private_view")
        .await
        .expect_err("rolled-back private view must not remain describable");
    assert_eq!(
        error.code(),
        Some(&tokio_postgres::error::SqlState::UNDEFINED_TABLE)
    );

    drop(client);
    connection.await.unwrap();
    server.abort();
    let _ = server.await;
}

#[test]
fn blocking_simple_copy_from_to_abort_and_transaction_boundaries() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE copy_simple (id INT, name TEXT)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY copy_simple (id, name) FROM STDIN WITH CSV"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 1), vec![b'G']);
    client
        .write_all(&tagged(b'd', b"1,Ada\n2,\n3,\"\"\n"))
        .unwrap();
    client.write_all(&tagged(b'c', &[])).unwrap();
    let copied = read_messages(&mut client, 2);
    assert_eq!(copied[0], (b'C', b"COPY 3\0".to_vec()));
    assert_eq!(copied[1], (b'Z', vec![b'I']));

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY copy_simple (name, id) TO STDOUT WITH CSV HEADER"),
        ))
        .unwrap();
    let copy_out = read_messages(&mut client, 7);
    assert_eq!(
        copy_out.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'd', b'd', b'c', b'C']
    );
    assert_eq!(copy_out[1].1, b"name,id\n");
    assert_eq!(copy_out[2].1, b"Ada,1\n");
    assert_eq!(copy_out[3].1, b",2\n");
    assert_eq!(copy_out[4].1, b"\"\",3\n");
    assert_eq!(read_messages(&mut client, 1), vec![(b'Z', vec![b'I'])]);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY copy_simple (id, name) FROM STDIN WITH CSV"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 1), vec![b'G']);
    client
        .write_all(&tagged(b'd', b"4,must-not-publish\n"))
        .unwrap();
    client.write_all(&tagged(b'f', b"client abort\0")).unwrap();
    let aborted = read_messages(&mut client, 2);
    assert_error_sqlstate(&aborted, b"C57014\0");

    client
        .write_all(&tagged(b'Q', &query_payload("BEGIN")))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'T']);
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY copy_simple (id, name) FROM STDIN WITH CSV"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 1), vec![b'G']);
    client.write_all(&tagged(b'd', b"5,priv\xc3")).unwrap();
    client.write_all(&tagged(b'd', b"\xa9\n")).unwrap();
    client.write_all(&tagged(b'c', &[])).unwrap();
    let private = read_messages(&mut client, 2);
    assert_eq!(private[0], (b'C', b"COPY 1\0".to_vec()));
    assert_eq!(private[1], (b'Z', vec![b'T']));
    client
        .write_all(&tagged(b'Q', &query_payload("ROLLBACK")))
        .unwrap();
    assert_eq!(read_messages(&mut client, 2)[1].1, vec![b'I']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload(
                "CREATE TABLE copy_prefix_must_not_publish (id INT); \
                 COPY copy_simple FROM STDIN",
            ),
        ))
        .unwrap();
    let multi = read_messages(&mut client, 2);
    assert_error_sqlstate(&multi, b"C0A000\0");
    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE copy_prefix_must_not_publish (id INT)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("SELECT COUNT(*) FROM copy_simple"),
        ))
        .unwrap();
    let count = read_messages(&mut client, 4);
    assert_eq!(
        count.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'Z']
    );
    assert!(
        count[1].1.ends_with(b"3"),
        "unexpected COUNT row: {:?}",
        count[1]
    );

    client.write_all(&tagged(b'X', &[])).unwrap();
    server.join().unwrap();
}

#[test]
fn copy_parse_and_completion_bind_to_the_analyzed_relation_generation() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let engine = std::sync::Arc::new(SharedEngine::new());
        let mut workers = Vec::new();
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let engine = std::sync::Arc::clone(&engine);
            workers.push(std::thread::spawn(move || {
                handle_connection(&mut stream, &engine).unwrap();
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let mut copy_client = TcpStream::connect(address).unwrap();
    let mut ddl_client = TcpStream::connect(address).unwrap();
    for client in [&copy_client, &ddl_client] {
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
    }
    copy_client.write_all(&startup_frame()).unwrap();
    ddl_client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut copy_client, 9);
    let _ = read_messages(&mut ddl_client, 9);

    let mut missing_parse = tagged(
        b'P',
        &parse_payload(
            "missing_copy",
            "COPY copy_parse_missing (id) FROM STDIN WITH CSV",
            &[],
        ),
    );
    missing_parse.extend(tagged(b'S', &[]));
    copy_client.write_all(&missing_parse).unwrap();
    let missing = read_messages(&mut copy_client, 2);
    assert_eq!(
        missing.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert_error_sqlstate(&missing, b"C42P01\0");

    ddl_client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE copy_generation_wire (id INT PRIMARY KEY, name TEXT)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut ddl_client, 2), vec![b'C', b'Z']);

    copy_client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY copy_generation_wire (id, name) FROM STDIN WITH CSV"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut copy_client, 1), vec![b'G']);
    for ddl in [
        "DROP TABLE copy_generation_wire",
        "CREATE TABLE copy_generation_wire (id INT PRIMARY KEY, name TEXT)",
    ] {
        ddl_client
            .write_all(&tagged(b'Q', &query_payload(ddl)))
            .unwrap();
        assert_eq!(read_tags(&mut ddl_client, 2), vec![b'C', b'Z']);
    }
    copy_client
        .write_all(&tagged(b'd', b"9,must-not-land\n"))
        .unwrap();
    copy_client.write_all(&tagged(b'c', &[])).unwrap();
    let stale_done = read_messages(&mut copy_client, 2);
    assert_error_sqlstate(&stale_done, b"C40001\0");
    assert_eq!(stale_done[1], (b'Z', vec![b'I']));

    let mut valid_parse = tagged(
        b'P',
        &parse_payload(
            "stale_copy_description",
            "COPY copy_generation_wire (id, name) FROM STDIN WITH CSV",
            &[],
        ),
    );
    valid_parse.extend(tagged(b'S', &[]));
    copy_client.write_all(&valid_parse).unwrap();
    assert_eq!(read_tags(&mut copy_client, 2), vec![b'1', b'Z']);
    for ddl in [
        "DROP TABLE copy_generation_wire",
        "CREATE TABLE copy_generation_wire (id INT PRIMARY KEY, name TEXT)",
    ] {
        ddl_client
            .write_all(&tagged(b'Q', &query_payload(ddl)))
            .unwrap();
        assert_eq!(read_tags(&mut ddl_client, 2), vec![b'C', b'Z']);
    }
    let mut describe = tagged(b'D', &describe_payload(b'S', "stale_copy_description"));
    describe.extend(tagged(b'S', &[]));
    copy_client.write_all(&describe).unwrap();
    let stale_description = read_messages(&mut copy_client, 2);
    assert_error_sqlstate(&stale_description, b"C40001\0");
    assert_eq!(stale_description[1], (b'Z', vec![b'I']));

    let mut execute_stale_parse = tagged(
        b'P',
        &parse_payload(
            "stale_copy_execute",
            "COPY copy_generation_wire (id, name) FROM STDIN WITH CSV",
            &[],
        ),
    );
    execute_stale_parse.extend(tagged(b'S', &[]));
    copy_client.write_all(&execute_stale_parse).unwrap();
    assert_eq!(read_tags(&mut copy_client, 2), vec![b'1', b'Z']);
    for ddl in [
        "DROP TABLE copy_generation_wire",
        "CREATE TABLE copy_generation_wire (id INT PRIMARY KEY, name TEXT)",
    ] {
        ddl_client
            .write_all(&tagged(b'Q', &query_payload(ddl)))
            .unwrap();
        assert_eq!(read_tags(&mut ddl_client, 2), vec![b'C', b'Z']);
    }
    let mut execute_stale = tagged(
        b'B',
        &bind_payload("stale_copy_execute_portal", "stale_copy_execute"),
    );
    execute_stale.extend(tagged(
        b'E',
        &execute_payload("stale_copy_execute_portal", 0),
    ));
    execute_stale.extend(tagged(b'S', &[]));
    copy_client.write_all(&execute_stale).unwrap();
    let stale_execute = read_messages(&mut copy_client, 3);
    assert_eq!(
        stale_execute
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'2', b'E', b'Z'],
        "stale COPY FROM Execute must fail before CopyInResponse (G)"
    );
    assert!(stale_execute[1]
        .1
        .windows(b"C40001\0".len())
        .any(|window| window == b"C40001\0"));

    let mut copy_to_parse_bind = tagged(
        b'P',
        &parse_payload(
            "stale_copy_to",
            "COPY copy_generation_wire TO STDOUT WITH CSV",
            &[],
        ),
    );
    copy_to_parse_bind.extend(tagged(
        b'B',
        &bind_payload("stale_copy_to_portal", "stale_copy_to"),
    ));
    copy_client.write_all(&copy_to_parse_bind).unwrap();
    assert_eq!(read_tags(&mut copy_client, 2), vec![b'1', b'2']);
    ddl_client
        .write_all(&tagged(
            b'Q',
            &query_payload("ALTER TABLE copy_generation_wire ADD COLUMN extra INT DEFAULT 0"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut ddl_client, 2), vec![b'C', b'Z']);
    let mut stale_copy_to_execute = tagged(b'E', &execute_payload("stale_copy_to_portal", 0));
    stale_copy_to_execute.extend(tagged(b'S', &[]));
    copy_client.write_all(&stale_copy_to_execute).unwrap();
    let stale_copy_to = read_messages(&mut copy_client, 2);
    assert_eq!(
        stale_copy_to
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'E', b'Z'],
        "stale COPY TO must fail before CopyOutResponse (H)"
    );
    assert!(
        stale_copy_to[0]
            .1
            .windows(b"C0A000\0".len())
            .any(|window| window == b"C0A000\0"),
        "unexpected stale COPY TO error: {stale_copy_to:?}"
    );

    ddl_client
        .write_all(&tagged(
            b'Q',
            &query_payload("COPY copy_generation_wire (id, id) TO STDOUT WITH CSV"),
        ))
        .unwrap();
    let simple_duplicate = read_messages(&mut ddl_client, 2);
    assert_error_sqlstate(&simple_duplicate, b"C42701\0");

    let mut extended_duplicate = tagged(
        b'P',
        &parse_payload(
            "duplicate_copy_to",
            "COPY copy_generation_wire (id, id) TO STDOUT WITH CSV",
            &[],
        ),
    );
    extended_duplicate.extend(tagged(
        b'B',
        &bind_payload("duplicate_copy_to_portal", "duplicate_copy_to"),
    ));
    extended_duplicate.extend(tagged(
        b'E',
        &execute_payload("duplicate_copy_to_portal", 0),
    ));
    extended_duplicate.extend(tagged(b'S', &[]));
    copy_client.write_all(&extended_duplicate).unwrap();
    let extended_duplicate = read_messages(&mut copy_client, 4);
    assert_eq!(
        extended_duplicate
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>(),
        vec![b'1', b'2', b'E', b'Z'],
        "duplicate extended COPY TO must fail before H/data frames"
    );
    assert!(extended_duplicate[2]
        .1
        .windows(b"C42701\0".len())
        .any(|window| window == b"C42701\0"));

    ddl_client
        .write_all(&tagged(
            b'Q',
            &query_payload("SELECT COUNT(*) FROM copy_generation_wire"),
        ))
        .unwrap();
    let count = read_messages(&mut ddl_client, 4);
    assert!(
        count[1].1.ends_with(b"0"),
        "stale COPY published a row: {count:?}"
    );

    copy_client.write_all(&tagged(b'X', &[])).unwrap();
    ddl_client.write_all(&tagged(b'X', &[])).unwrap();
    drop(copy_client);
    drop(ddl_client);
    server.join().unwrap();
}

fn startup_frame() -> Vec<u8> {
    let mut payload = 196_608_u32.to_be_bytes().to_vec();
    payload.extend_from_slice(b"user\0postgres\0database\0postgres\0\0");
    let mut frame = u32::try_from(payload.len() + 4)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    frame.extend(payload);
    frame
}

fn tagged(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut frame = vec![tag];
    frame.extend_from_slice(&u32::try_from(payload.len() + 4).unwrap().to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

fn backend_key(messages: &[(u8, Vec<u8>)]) -> (u32, [u8; 4]) {
    let payload = &messages
        .iter()
        .find(|(tag, _)| *tag == b'K')
        .expect("startup omitted BackendKeyData")
        .1;
    assert_eq!(payload.len(), 8);
    (
        u32::from_be_bytes(payload[..4].try_into().unwrap()),
        payload[4..].try_into().unwrap(),
    )
}

fn cancel_startup_frame(process_id: u32, secret_key: [u8; 4]) -> Vec<u8> {
    let mut payload = 80_877_102_u32.to_be_bytes().to_vec();
    payload.extend_from_slice(&process_id.to_be_bytes());
    payload.extend_from_slice(&secret_key);
    let mut frame = u32::try_from(payload.len() + 4)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    frame.extend(payload);
    frame
}

fn send_cancel(address: std::net::SocketAddr, process_id: u32, secret_key: [u8; 4]) {
    let mut cancel = TcpStream::connect(address).unwrap();
    cancel
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    cancel
        .write_all(&cancel_startup_frame(process_id, secret_key))
        .unwrap();
    cancel.shutdown(std::net::Shutdown::Write).unwrap();
    let mut response = Vec::new();
    cancel.read_to_end(&mut response).unwrap();
    assert!(response.is_empty(), "CancelRequest emitted a response");
}

async fn send_cancel_async(address: std::net::SocketAddr, process_id: u32, secret_key: [u8; 4]) {
    let mut cancel = tokio::net::TcpStream::connect(address).await.unwrap();
    cancel
        .write_all(&cancel_startup_frame(process_id, secret_key))
        .await
        .unwrap();
    cancel.shutdown().await.unwrap();
    let mut response = Vec::new();
    cancel.read_to_end(&mut response).await.unwrap();
    assert!(response.is_empty(), "CancelRequest emitted a response");
}

async fn read_messages_async(
    stream: &mut tokio::net::TcpStream,
    count: usize,
) -> Vec<(u8, Vec<u8>)> {
    let mut messages = Vec::with_capacity(count);
    for _ in 0..count {
        let frame = read_tagged_frame_async(stream)
            .await
            .unwrap()
            .expect("server closed before the expected response");
        messages.push((frame[0], frame[5..].to_vec()));
    }
    messages
}

async fn read_tags_async(stream: &mut tokio::net::TcpStream, count: usize) -> Vec<u8> {
    read_messages_async(stream, count)
        .await
        .into_iter()
        .map(|(tag, _)| tag)
        .collect()
}

fn assert_private_parse_description(messages: &[(u8, Vec<u8>)]) {
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'1', b't', b'T', b'Z']
    );
    assert_eq!(
        messages[1].1,
        vec![0, 2, 0, 0, 0, 23, 0, 0, 0, 25],
        "private Parse must infer int4/text parameter OIDs"
    );
}

fn assert_failed_cached_rowless_description(messages: &[(u8, Vec<u8>)]) {
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b't', b'n', b'Z']
    );
    assert_eq!(
        messages[0].1,
        vec![0, 2, 0, 0, 0, 23, 0, 0, 0, 25],
        "failed-transaction rowless Describe must retain cached private parameter OIDs"
    );
    assert_eq!(messages[2].1, vec![b'E']);
}

fn assert_error_sqlstate(messages: &[(u8, Vec<u8>)], sqlstate: &[u8]) {
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E', b'Z']
    );
    assert!(messages[0]
        .1
        .windows(sqlstate.len())
        .any(|window| window == sqlstate));
}

fn assert_late_auth_error(messages: &[(u8, Vec<u8>)]) {
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, b'E');
    assert!(messages[0]
        .1
        .windows(b"C08P01\0".len())
        .any(|window| window == b"C08P01\0"));
}

fn late_auth_payloads() -> Vec<(&'static str, Vec<u8>)> {
    let mut sasl_initial = b"SCRAM-SHA-256\0".to_vec();
    let client_first = b"n,,n=,r=late";
    sasl_initial.extend_from_slice(&i32::try_from(client_first.len()).unwrap().to_be_bytes());
    sasl_initial.extend_from_slice(client_first);
    vec![
        ("password", b"late-secret\0".to_vec()),
        ("sasl_initial", sasl_initial),
        (
            "sasl_response",
            b"c=biws,r=late,p=proof-without-trailing-nul".to_vec(),
        ),
    ]
}

fn parse_payload(name: &str, query: &str, type_oids: &[u32]) -> Vec<u8> {
    let mut payload = Vec::new();
    push_cstring(&mut payload, name);
    push_cstring(&mut payload, query);
    payload.extend_from_slice(&i16::try_from(type_oids.len()).unwrap().to_be_bytes());
    for oid in type_oids {
        payload.extend_from_slice(&oid.to_be_bytes());
    }
    payload
}

fn bind_payload(portal: &str, statement: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    push_cstring(&mut payload, portal);
    push_cstring(&mut payload, statement);
    payload.extend_from_slice(&0_i16.to_be_bytes());
    payload.extend_from_slice(&0_i16.to_be_bytes());
    payload.extend_from_slice(&0_i16.to_be_bytes());
    payload
}

fn query_payload(sql: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    push_cstring(&mut payload, sql);
    payload
}

fn send_extended_insert(client: &mut TcpStream, name: &str, sql: &str) {
    let mut request = Vec::new();
    request.extend(tagged(b'P', &parse_payload(name, sql, &[])));
    request.extend(tagged(b'B', &bind_payload(name, name)));
    request.extend(tagged(b'E', &execute_payload(name, 0)));
    client.write_all(&request).unwrap();
    assert_eq!(read_tags(client, 3), vec![b'1', b'2', b'C']);
}

fn malformed_bind_shape_payload(portal: &str, statement: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    push_cstring(&mut payload, portal);
    push_cstring(&mut payload, statement);
    payload.push(0);
    payload
}

fn describe_payload(target: u8, name: &str) -> Vec<u8> {
    let mut payload = vec![target];
    push_cstring(&mut payload, name);
    payload
}

fn execute_payload(portal: &str, max_rows: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    push_cstring(&mut payload, portal);
    payload.extend_from_slice(&max_rows.to_be_bytes());
    payload
}

fn push_cstring(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(value.as_bytes());
    bytes.push(0);
}

fn read_tags(stream: &mut TcpStream, count: usize) -> Vec<u8> {
    read_messages(stream, count)
        .into_iter()
        .map(|(tag, _)| tag)
        .collect()
}

fn read_messages(stream: &mut TcpStream, count: usize) -> Vec<(u8, Vec<u8>)> {
    (0..count)
        .map(|_| {
            let mut tag = [0_u8; 1];
            stream.read_exact(&mut tag).unwrap();
            let mut len = [0_u8; 4];
            stream.read_exact(&mut len).unwrap();
            let payload_len = u32::from_be_bytes(len) as usize - 4;
            let mut payload = vec![0_u8; payload_len];
            stream.read_exact(&mut payload).unwrap();
            (tag[0], payload)
        })
        .collect()
}
