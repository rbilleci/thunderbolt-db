use super::{
    handle_connection, parse_batching_flag, read_tagged_frame_async,
    serve_async_with_engine_batching,
};
use gpu_db_facade::SharedEngine;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

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
    let _ = read_messages(&mut client, 6);

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
    let _ = read_messages_async(&mut client, 6).await;

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
    assert_eq!(
        read_tags(&mut client, 6),
        vec![b'R', b'S', b'S', b'S', b'S', b'Z']
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
    let _ = read_messages(&mut client, 6);

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
    let _ = read_messages(&mut client, 6);

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
    assert_eq!(read_tags(&mut client, 6).last(), Some(&b'Z'));

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
    assert_eq!(read_tags(&mut client, 6).last(), Some(&b'Z'));

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
    assert_eq!(read_tags_async(&mut client, 6).await.last(), Some(&b'Z'));

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
    assert_eq!(read_tags(&mut creator, 6).last(), Some(&b'Z'));
    assert_eq!(read_tags(&mut observer, 6).last(), Some(&b'Z'));

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
    assert_eq!(read_tags_async(&mut creator, 6).await.last(), Some(&b'Z'));
    assert_eq!(read_tags_async(&mut observer, 6).await.last(), Some(&b'Z'));

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
    let _ = read_messages(&mut client, 6);

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
    let _ = read_messages(&mut client, 6);
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
