use super::*;
use crate::engine_transaction_catalog::index_identity::{index_identity, table_identity};

#[test]
fn transactional_constraint_backed_index_rename_moves_one_identity_and_recovers() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            4_004,
            parsed(
                "CREATE TABLE constraint_index_owner \
                 (id int4 PRIMARY KEY, code int4 UNIQUE, \
                  CONSTRAINT constraint_index_name_taken CHECK (id > 0))",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_005,
            parsed("CREATE TABLE constraint_index_child (id int4 PRIMARY KEY, owner_code int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_006,
            parsed(
                "ALTER TABLE ONLY constraint_index_child \
                 ADD CONSTRAINT constraint_index_child_owner_fk \
                 FOREIGN KEY (owner_code) REFERENCES constraint_index_owner(code)",
            ),
        )
        .unwrap();
    for (txn_id, sql) in [
        (
            4_007,
            "COMMENT ON INDEX constraint_index_owner_pkey IS 'primary lookup'",
        ),
        (
            4_008,
            "COMMENT ON CONSTRAINT constraint_index_owner_pkey \
             ON constraint_index_owner IS 'primary identity'",
        ),
        (
            4_009,
            "COMMENT ON INDEX constraint_index_owner_code_key IS 'code lookup'",
        ),
        (
            4_010,
            "COMMENT ON CONSTRAINT constraint_index_owner_code_key \
             ON constraint_index_owner IS 'code identity'",
        ),
    ] {
        engine.submit_transaction(txn_id, parsed(sql)).unwrap();
    }
    let original = engine.catalog_snapshot();
    let primary = index_named(
        &original.relational_catalog["constraint_index_owner"],
        "constraint_index_owner_pkey",
    )
    .clone();
    let unique = index_named(
        &original.relational_catalog["constraint_index_owner"],
        "constraint_index_owner_code_key",
    )
    .clone();

    engine.submit_transaction(4_011, parsed("BEGIN")).unwrap();
    let collision = engine
        .submit_transaction(
            4_011,
            parsed(
                "ALTER INDEX constraint_index_owner_pkey \
                 RENAME TO constraint_index_name_taken",
            ),
        )
        .expect_err("a backing constraint cannot collide with an existing CHECK constraint");
    assert!(
        collision
            .to_string()
            .contains("relation \"constraint_index_name_taken\" already exists"),
        "{collision}"
    );
    assert!(engine
        .transaction_snapshot_handle(4_011)
        .unwrap()
        .transaction_catalog()
        .same_contents(original.as_ref()));
    engine
        .submit_transaction(
            4_011,
            parsed(
                "ALTER INDEX constraint_index_owner_pkey \
                 RENAME TO constraint_index_owner_id_idx",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_011,
            parsed(
                "ALTER INDEX constraint_index_owner_code_key \
                 RENAME TO constraint_index_owner_code_idx",
            ),
        )
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(4_011)
        .unwrap()
        .transaction_catalog();
    let private_table = &private.relational_catalog["constraint_index_owner"];
    let private_primary = index_named(private_table, "constraint_index_owner_id_idx");
    let private_unique = index_named(private_table, "constraint_index_owner_code_idx");
    assert_eq!(private_primary.oid, primary.oid);
    assert_eq!(private_unique.oid, unique.oid);
    assert!(private_primary.primary_key);
    assert!(private_unique.unique_constraint);
    assert_eq!(
        private
            .relational_comments
            .get(&RelationalCommentTarget::Constraint {
                table: "constraint_index_owner".to_string(),
                constraint: "constraint_index_owner_id_idx".to_string(),
            })
            .map(String::as_str),
        Some("primary identity")
    );
    assert_eq!(
        private
            .relational_comments
            .get(&RelationalCommentTarget::Constraint {
                table: "constraint_index_owner".to_string(),
                constraint: "constraint_index_owner_code_idx".to_string(),
            })
            .map(String::as_str),
        Some("code identity")
    );
    assert!(engine.catalog_snapshot().same_contents(original.as_ref()));
    engine
        .submit_transaction(4_011, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine.catalog_snapshot().same_contents(original.as_ref()));

    engine.submit_transaction(4_012, parsed("BEGIN")).unwrap();
    for sql in [
        "ALTER INDEX constraint_index_owner_pkey \
         RENAME TO constraint_index_owner_id_idx",
        "ALTER INDEX constraint_index_owner_code_key \
         RENAME TO constraint_index_owner_code_idx",
    ] {
        engine.submit_transaction(4_012, parsed(sql)).unwrap();
    }
    engine.submit_transaction(4_012, parsed("COMMIT")).unwrap();

    let committed = engine.catalog_snapshot();
    let committed_table = &committed.relational_catalog["constraint_index_owner"];
    let renamed_primary = index_named(committed_table, "constraint_index_owner_id_idx");
    let renamed_unique = index_named(committed_table, "constraint_index_owner_code_idx");
    assert_eq!(renamed_primary.oid, primary.oid);
    assert_eq!(renamed_primary.key_columns, primary.key_columns);
    assert!(renamed_primary.primary_key);
    assert_eq!(renamed_unique.oid, unique.oid);
    assert_eq!(renamed_unique.key_columns, unique.key_columns);
    assert!(renamed_unique.unique_constraint);
    assert_eq!(
        engine
            .relational_constraint_comment(
                "constraint_index_owner",
                "constraint_index_owner_id_idx"
            )
            .as_deref(),
        Some("primary identity")
    );
    assert_eq!(
        engine
            .relational_constraint_comment(
                "constraint_index_owner",
                "constraint_index_owner_code_idx"
            )
            .as_deref(),
        Some("code identity")
    );
    assert_eq!(
        engine
            .relational_index_comment("constraint_index_owner_id_idx")
            .as_deref(),
        Some("primary lookup")
    );
    assert_eq!(
        engine
            .relational_index_comment("constraint_index_owner_code_idx")
            .as_deref(),
        Some("code lookup")
    );
    let child = &committed.relational_catalog["constraint_index_child"];
    assert_eq!(
        child.foreign_keys[0].referenced_table,
        "constraint_index_owner"
    );
    assert_eq!(child.foreign_keys[0].referenced_column, "code");

    engine
        .submit_transaction(
            4_013,
            parsed("INSERT INTO constraint_index_owner VALUES (1, 7)"),
        )
        .unwrap();
    for (txn_id, sql) in [
        (4_014, "INSERT INTO constraint_index_owner VALUES (1, 8)"),
        (4_015, "INSERT INTO constraint_index_owner VALUES (2, 7)"),
        (4_016, "INSERT INTO constraint_index_child VALUES (1, 999)"),
    ] {
        assert!(
            engine.submit_transaction(txn_id, parsed(sql)).is_err(),
            "{sql}"
        );
    }
    engine
        .submit_transaction(
            4_017,
            parsed("INSERT INTO constraint_index_child VALUES (1, 7)"),
        )
        .unwrap();

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
    assert_eq!(
        recovered
            .relational_constraint_comment(
                "constraint_index_owner",
                "constraint_index_owner_code_idx"
            )
            .as_deref(),
        Some("code identity")
    );
}

#[test]
fn constraint_backed_index_rename_uses_owner_local_constraint_namespace() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            4_030,
            "CREATE TABLE constraint_scope_owner \
             (id int4 PRIMARY KEY, \
              CONSTRAINT constraint_scope_local_taken CHECK (id > 0))",
        )
        .unwrap();
    engine
        .execute_text(
            4_031,
            "CREATE TABLE constraint_scope_other \
             (id int4 PRIMARY KEY, owner_id int4, \
              CONSTRAINT constraint_scope_remote_taken CHECK (id > 0))",
        )
        .unwrap();
    engine
        .execute_text(
            4_032,
            "ALTER TABLE ONLY constraint_scope_other \
             ADD CONSTRAINT constraint_scope_remote_fk \
             FOREIGN KEY (owner_id) REFERENCES constraint_scope_owner(id)",
        )
        .unwrap();
    engine
        .execute_text(
            4_033,
            "COMMENT ON INDEX constraint_scope_owner_pkey IS 'owner index identity'",
        )
        .unwrap();
    engine
        .execute_text(
            4_034,
            "COMMENT ON CONSTRAINT constraint_scope_owner_pkey \
             ON constraint_scope_owner IS 'owner constraint identity'",
        )
        .unwrap();
    let original = engine
        .relational_catalog_table("constraint_scope_owner")
        .unwrap();
    let oid = index_named(&original, "constraint_scope_owner_pkey").oid;

    // Autocommit: an unrelated table's CHECK name is not in this constraint namespace.
    engine
        .execute_text(
            4_035,
            "ALTER INDEX constraint_scope_owner_pkey \
             RENAME TO constraint_scope_remote_taken",
        )
        .unwrap();
    let autocommit = engine
        .relational_catalog_table("constraint_scope_owner")
        .unwrap();
    assert_eq!(
        index_named(&autocommit, "constraint_scope_remote_taken").oid,
        oid
    );
    assert_eq!(
        engine
            .relational_index_comment("constraint_scope_remote_taken")
            .as_deref(),
        Some("owner index identity")
    );
    assert_eq!(
        engine
            .relational_constraint_comment(
                "constraint_scope_owner",
                "constraint_scope_remote_taken"
            )
            .as_deref(),
        Some("owner constraint identity")
    );

    engine.submit_transaction(4_036, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();
    let before_failure = engine
        .transaction_snapshot_handle(4_036)
        .unwrap()
        .transaction_catalog();
    let local_collision = engine
        .submit_transaction(
            4_036,
            parsed(
                "ALTER INDEX constraint_scope_remote_taken \
                 RENAME TO constraint_scope_local_taken",
            ),
        )
        .expect_err("the owner table's CHECK name remains a collision");
    assert!(
        local_collision
            .to_string()
            .contains("relation \"constraint_scope_local_taken\" already exists"),
        "{local_collision}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine
        .transaction_snapshot_handle(4_036)
        .unwrap()
        .transaction_catalog()
        .same_contents(before_failure.as_ref()));

    // Explicit transaction: an unrelated table's FK name is equally owner-local.
    engine
        .submit_transaction(
            4_036,
            parsed(
                "ALTER INDEX constraint_scope_remote_taken \
                 RENAME TO constraint_scope_remote_fk",
            ),
        )
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(4_036)
        .unwrap()
        .transaction_catalog();
    assert_eq!(
        index_named(
            &private.relational_catalog["constraint_scope_owner"],
            "constraint_scope_remote_fk"
        )
        .oid,
        oid
    );
    assert_eq!(
        private
            .relational_comments
            .get(&RelationalCommentTarget::Constraint {
                table: "constraint_scope_owner".to_string(),
                constraint: "constraint_scope_remote_fk".to_string(),
            })
            .map(String::as_str),
        Some("owner constraint identity")
    );
    assert!(engine
        .relational_catalog_table("constraint_scope_owner")
        .unwrap()
        .indexes
        .iter()
        .any(|index| index.name == "constraint_scope_remote_taken"));
    engine.submit_transaction(4_036, parsed("COMMIT")).unwrap();

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let recovered_owner = recovered
        .relational_catalog_table("constraint_scope_owner")
        .unwrap();
    assert_eq!(
        index_named(&recovered_owner, "constraint_scope_remote_fk").oid,
        oid
    );
    assert_eq!(
        recovered
            .relational_index_comment("constraint_scope_remote_fk")
            .as_deref(),
        Some("owner index identity")
    );
    assert_eq!(
        recovered
            .relational_constraint_comment("constraint_scope_owner", "constraint_scope_remote_fk")
            .as_deref(),
        Some("owner constraint identity")
    );
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
