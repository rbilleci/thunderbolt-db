use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_describes_transaction_private_view_rename_drop_and_recreate() {
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
        .batch_execute(
            "CREATE TABLE prepared_lifecycle_source (id int4, note text);
             CREATE VIEW prepared_lifecycle_view AS
             SELECT id, note FROM prepared_lifecycle_source",
        )
        .await
        .unwrap();
    client
        .batch_execute(
            "BEGIN;
             ALTER VIEW prepared_lifecycle_view
             RENAME TO prepared_lifecycle_renamed",
        )
        .await
        .unwrap();
    let renamed = client
        .prepare("SELECT * FROM prepared_lifecycle_renamed")
        .await
        .unwrap();
    assert_eq!(
        renamed
            .columns()
            .iter()
            .map(|column| (column.name(), column.type_().name()))
            .collect::<Vec<_>>(),
        vec![("id", "int4"), ("note", "text")]
    );
    let old_error = client
        .prepare("SELECT * FROM prepared_lifecycle_view")
        .await
        .expect_err("old private binding must disappear after rename");
    assert_eq!(
        old_error.code(),
        Some(&tokio_postgres::error::SqlState::UNDEFINED_TABLE)
    );
    client.batch_execute("ROLLBACK").await.unwrap();
    client
        .prepare("SELECT * FROM prepared_lifecycle_view")
        .await
        .unwrap();

    client
        .batch_execute(
            "BEGIN;
             ALTER VIEW prepared_lifecycle_view
             RENAME TO prepared_lifecycle_renamed;
             DROP VIEW prepared_lifecycle_renamed;
             CREATE VIEW prepared_lifecycle_final AS
             SELECT note FROM prepared_lifecycle_source",
        )
        .await
        .unwrap();
    let final_statement = client
        .prepare("SELECT * FROM prepared_lifecycle_final")
        .await
        .unwrap();
    assert_eq!(
        final_statement
            .columns()
            .iter()
            .map(|column| (column.name(), column.type_().name()))
            .collect::<Vec<_>>(),
        vec![("note", "text")]
    );
    client.batch_execute("COMMIT").await.unwrap();
    assert!(client
        .query("SELECT * FROM prepared_lifecycle_final", &[])
        .await
        .unwrap()
        .is_empty());
    for missing in ["prepared_lifecycle_view", "prepared_lifecycle_renamed"] {
        let error = client
            .prepare(&format!("SELECT * FROM {missing}"))
            .await
            .expect_err("retired binding must not remain globally visible");
        assert_eq!(
            error.code(),
            Some(&tokio_postgres::error::SqlState::UNDEFINED_TABLE)
        );
    }

    drop(client);
    connection.await.unwrap();
    server.abort();
    let _ = server.await;
}
