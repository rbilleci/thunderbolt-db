use super::*;
use crate::engine_transaction_catalog::index_identity::{index_identity, table_identity};

#[test]
fn transactional_constraint_backed_index_rename_rejects_before_wal() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            4_004,
            parsed(
                "CREATE TABLE constraint_index_owner \
                 (id int4 PRIMARY KEY, code int4 UNIQUE)",
            ),
        )
        .unwrap();
    let original = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(4_005, parsed("BEGIN")).unwrap();
    for sql in [
        "ALTER INDEX constraint_index_owner_pkey RENAME TO constraint_index_owner_id_idx",
        "ALTER INDEX constraint_index_owner_code_key RENAME TO constraint_index_owner_code_idx",
    ] {
        let error = engine
            .submit_transaction(4_005, parsed(sql))
            .expect_err("constraint-backed index rename must fail before private catalog mutation");
        assert!(
            error
                .to_string()
                .contains("cannot rename constraint-backed index with ALTER INDEX"),
            "{error}"
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine
            .transaction_snapshot_handle(4_005)
            .unwrap()
            .transaction_catalog()
            .same_contents(original.as_ref()));
    }
    engine
        .submit_transaction(4_005, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine.catalog_snapshot().same_contents(original.as_ref()));

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(original.as_ref()));
}

#[test]
fn constraint_backed_index_rename_rejects_even_when_destination_is_owner_local_free() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            4_030,
            "CREATE TABLE constraint_scope_owner (id int4 PRIMARY KEY)",
        )
        .unwrap();
    engine
        .execute_text(
            4_031,
            "CREATE TABLE constraint_scope_other \
             (id int4, CONSTRAINT constraint_scope_remote_taken CHECK (id > 0))",
        )
        .unwrap();
    let original = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();

    for destination in ["constraint_scope_free", "constraint_scope_remote_taken"] {
        let sql = format!("ALTER INDEX constraint_scope_owner_pkey RENAME TO {destination}");
        let error = engine.execute_text(4_032, &sql).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot rename constraint-backed index with ALTER INDEX"),
            "{error}"
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine.catalog_snapshot().same_contents(original.as_ref()));
    }
}

#[test]
fn constraint_comment_is_part_of_table_and_index_identity() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            9_001,
            "CREATE TABLE identity_comment_owner (id int4 PRIMARY KEY)",
        )
        .unwrap();
    engine
        .execute_text(
            9_002,
            "COMMENT ON CONSTRAINT identity_comment_owner_pkey \
             ON identity_comment_owner IS 'identity metadata'",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let table = &catalog.relational_catalog["identity_comment_owner"];
    let index = &table.indexes[0];
    let table_with_comment = table_identity(catalog.as_ref(), table).unwrap();
    let index_with_comment = index_identity(catalog.as_ref(), index).unwrap();

    let mut altered = catalog.as_ref().clone();
    altered
        .relational_comments
        .remove(&RelationalCommentTarget::Constraint {
            table: "identity_comment_owner".to_string(),
            constraint: "identity_comment_owner_pkey".to_string(),
        });
    let altered_table = &altered.relational_catalog["identity_comment_owner"];
    let table_without_comment = table_identity(&altered, altered_table).unwrap();
    let index_without_comment = index_identity(&altered, &altered_table.indexes[0]).unwrap();
    assert_ne!(table_with_comment.digest, table_without_comment.digest);
    assert_ne!(index_with_comment.digest, index_without_comment.digest);
}
