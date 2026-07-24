use super::*;

async fn catalog_relation_oid(client: &tokio_postgres::Client, name: &str) -> Option<i32> {
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
async fn tokio_postgres_runs_private_sequence_restart_rename_rollback_and_truncate_reset() {
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
        .batch_execute("CREATE TABLE server_sequence_owner (id SERIAL, marker INT)")
        .await
        .unwrap();
    let original_name = "server_sequence_owner_id_seq";
    let original_oid = catalog_relation_oid(&client, original_name)
        .await
        .expect("implicit sequence must have a catalog identity");

    client
        .batch_execute(
            "BEGIN;
             ALTER SEQUENCE server_sequence_owner_id_seq RESTART WITH 40;
             ALTER SEQUENCE server_sequence_owner_id_seq
             RENAME TO server_sequence_private;
             INSERT INTO server_sequence_owner (marker) VALUES (NULL)",
        )
        .await
        .unwrap();
    assert_eq!(
        catalog_relation_oid(&client, "server_sequence_private").await,
        Some(original_oid)
    );
    assert_eq!(
        client
            .query_one(
                "SELECT id, marker FROM server_sequence_owner ORDER BY id",
                &[],
            )
            .await
            .unwrap()
            .get::<_, i32>(0),
        40
    );
    client.batch_execute("ROLLBACK").await.unwrap();
    assert_eq!(
        catalog_relation_oid(&client, original_name).await,
        Some(original_oid)
    );
    assert_eq!(
        catalog_relation_oid(&client, "server_sequence_private").await,
        None
    );
    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM server_sequence_owner", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        0
    );

    client
        .batch_execute(
            "BEGIN;
             ALTER SEQUENCE server_sequence_owner_id_seq RESTART 40;
             ALTER SEQUENCE server_sequence_owner_id_seq
             RENAME TO server_sequence_final;
             INSERT INTO server_sequence_owner (marker) VALUES (NULL), (7);
             COMMIT",
        )
        .await
        .unwrap();
    let rows = client
        .query(
            "SELECT id, marker FROM server_sequence_owner ORDER BY id",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get::<_, i32>(0), 40);
    assert_eq!(rows[0].get::<_, Option<i32>>(1), None);
    assert_eq!(rows[1].get::<_, i32>(0), 41);
    assert_eq!(rows[1].get::<_, Option<i32>>(1), Some(7));
    assert_eq!(
        catalog_relation_oid(&client, "server_sequence_final").await,
        Some(original_oid)
    );

    client
        .batch_execute(
            "BEGIN;
             TRUNCATE server_sequence_owner RESTART IDENTITY;
             INSERT INTO server_sequence_owner (marker) VALUES (NULL);
             COMMIT",
        )
        .await
        .unwrap();
    let row = client
        .query_one(
            "SELECT id, marker FROM server_sequence_owner ORDER BY id",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, i32>(0), 1);
    assert_eq!(row.get::<_, Option<i32>>(1), None);

    drop(client);
    connection.await.unwrap();
    server.abort();
    let _ = server.await;
}
