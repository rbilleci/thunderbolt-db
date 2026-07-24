use super::*;

async fn catalog_has_index(client: &tokio_postgres::Client, name: &str) -> bool {
    !client
        .query(
            "SELECT relname FROM pg_catalog.pg_class WHERE relname = $1",
            &[&name],
        )
        .await
        .unwrap()
        .is_empty()
}

async fn catalog_index_oid(client: &tokio_postgres::Client, name: &str) -> Option<i32> {
    client
        .query_opt(
            "SELECT oid FROM pg_catalog.pg_class WHERE relname = $1",
            &[&name],
        )
        .await
        .unwrap()
        .map(|row| row.get(0))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_runs_private_index_create_rename_multi_drop_and_rollback() {
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
        .batch_execute("CREATE TABLE prepared_index_source (id int4, code int4)")
        .await
        .unwrap();
    client
        .batch_execute(
            "BEGIN;
             CREATE UNIQUE INDEX prepared_index_code
             ON prepared_index_source (code)",
        )
        .await
        .unwrap();
    let private_oid = catalog_index_oid(&client, "prepared_index_code")
        .await
        .expect("private CREATE INDEX must publish a catalog OID");
    client
        .batch_execute(
            "ALTER INDEX prepared_index_code
             RENAME TO prepared_index_code_private;
             CREATE INDEX prepared_index_id
             ON prepared_index_source (id)",
        )
        .await
        .unwrap();
    assert!(catalog_has_index(&client, "prepared_index_code_private").await);
    assert!(catalog_has_index(&client, "prepared_index_id").await);
    assert!(!catalog_has_index(&client, "prepared_index_code").await);
    assert_eq!(
        catalog_index_oid(&client, "prepared_index_code_private").await,
        Some(private_oid),
        "ALTER INDEX must preserve the stable catalog identity"
    );
    client.batch_execute("ROLLBACK").await.unwrap();
    assert!(!catalog_has_index(&client, "prepared_index_code_private").await);
    assert!(!catalog_has_index(&client, "prepared_index_id").await);

    client
        .batch_execute(
            "BEGIN;
             CREATE UNIQUE INDEX prepared_index_code
             ON prepared_index_source (code);
             ALTER INDEX prepared_index_code
             RENAME TO prepared_index_code_final;
             CREATE INDEX prepared_index_id
             ON prepared_index_source (id);
             DROP INDEX IF EXISTS prepared_index_missing, prepared_index_id;
             INSERT INTO prepared_index_source VALUES
             (1, NULL), (2, NULL), (3, 7);
             COMMIT",
        )
        .await
        .unwrap();
    assert!(catalog_has_index(&client, "prepared_index_code_final").await);
    assert!(!catalog_has_index(&client, "prepared_index_code").await);
    assert!(!catalog_has_index(&client, "prepared_index_id").await);
    let duplicate = client
        .execute("INSERT INTO prepared_index_source VALUES (4, 7)", &[])
        .await
        .expect_err("the transactionally published unique index must reject a duplicate");
    assert_eq!(
        duplicate.code(),
        Some(&tokio_postgres::error::SqlState::UNIQUE_VIOLATION)
    );
    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM prepared_index_source", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        3
    );

    drop(client);
    connection.await.unwrap();
    server.abort();
    let _ = server.await;
}
