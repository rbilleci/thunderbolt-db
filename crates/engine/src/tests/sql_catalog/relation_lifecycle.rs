use crate::tests::assert_recovered_relational_access_path;
use crate::{
    Engine, RelationalAccessPath, RelationalCheckConstraint, RelationalColumn, RelationalIndex,
    FIRST_USER_COLUMN_ID, FIRST_USER_RELATION_OID, PUBLIC_SCHEMA_NAME,
};
use gpu_db_sql::{parse_command, Command, SelectFilterOp, SqlType, SqlValue};

#[test]
fn relational_catalog_assigns_stable_public_schema_and_type_metadata() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();

    let table = e.relational_catalog_table("people").unwrap();
    assert_eq!(table.schema, PUBLIC_SCHEMA_NAME);
    assert_eq!(table.name, "people");
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);
    assert_eq!(table.columns.len(), 2);
    assert_eq!(
        table.columns[0],
        RelationalColumn {
            id: FIRST_USER_COLUMN_ID,
            table_oid: FIRST_USER_RELATION_OID,
            attnum: 1,
            name: "id".to_string(),
            ty: SqlType::Int4,
            domain: None,
            default: None,
            type_oid: SqlType::Int4.postgres_oid(),
            type_size: SqlType::Int4.type_size(),
        }
    );
    assert_eq!(
        table.columns[1],
        RelationalColumn {
            id: FIRST_USER_COLUMN_ID + 1,
            table_oid: FIRST_USER_RELATION_OID,
            attnum: 2,
            name: "name".to_string(),
            ty: SqlType::Text,
            domain: None,
            default: None,
            type_oid: SqlType::Text.postgres_oid(),
            type_size: SqlType::Text.type_size(),
        }
    );

    e.execute_text(2, "CREATE TABLE teams (id INT)").unwrap();
    assert_eq!(
        e.relational_catalog_table("teams").unwrap().oid,
        FIRST_USER_RELATION_OID + 1
    );
}

#[test]
fn relational_catalog_records_create_index_and_replays_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    let table = e.relational_catalog_table("people").unwrap();
    assert_eq!(
        table.indexes,
        vec![RelationalIndex {
            name: "people_name_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: false,
            primary_key: false,
            unique_constraint: false,
        }]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        vec![RelationalIndex {
            name: "people_name_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: false,
            primary_key: false,
            unique_constraint: false,
        }]
    );

    let duplicate_err = e
        .execute_text(3, "CREATE INDEX people_name_idx ON people (id)")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_err.contains("relation \"people_name_idx\" already exists"),
        "{duplicate_err}"
    );

    let missing = Engine::new_local_cpu_oracle();
    missing
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    assert!(missing
        .execute_text(2, "CREATE INDEX people_missing_idx ON people (missing)")
        .unwrap_err()
        .to_string()
        .contains("column \"missing\" does not exist"));
}

#[test]
fn relational_unique_index_rejects_duplicate_create_insert_update_and_replays_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();
    e.execute_text(3, "CREATE UNIQUE INDEX people_name_uidx ON people (name)")
        .unwrap();

    let indexes = e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        indexes,
        vec![RelationalIndex {
            name: "people_name_uidx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: true,
            primary_key: false,
            unique_constraint: false,
        }]
    );

    let duplicate_insert = e
        .execute_text(4, "INSERT INTO people (id, name) VALUES (3, 'Ada')")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_insert.contains("duplicate key value violates unique index"),
        "{duplicate_insert}"
    );
    let duplicate_update = e
        .execute_text(5, "UPDATE people SET name = 'Ada' WHERE id = 2")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_update.contains("duplicate key value violates unique index"),
        "{duplicate_update}"
    );

    let Command::Select(select) =
        parse_command("SELECT id, name FROM people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Grace".to_string())],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        indexes
    );

    let duplicate_existing = Engine::new_local_cpu_oracle();
    duplicate_existing
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    duplicate_existing
        .execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Ada')",
        )
        .unwrap();
    let create_err = duplicate_existing
        .execute_text(3, "CREATE UNIQUE INDEX people_name_uidx ON people (name)")
        .unwrap_err()
        .to_string();
    assert!(
        create_err.contains("duplicate key value violates unique index"),
        "{create_err}"
    );
}

#[test]
fn relational_unique_constraints_reject_duplicates_and_replay_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT, name TEXT UNIQUE, CONSTRAINT people_id_key UNIQUE (id))",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();

    let indexes = e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        indexes,
        vec![
            RelationalIndex {
                name: "people_name_key".to_string(),
                table: "people".to_string(),
                column: "name".to_string(),
                key_columns: vec!["name".to_string()],
                unique: true,
                primary_key: false,
                unique_constraint: true,
            },
            RelationalIndex {
                name: "people_id_key".to_string(),
                table: "people".to_string(),
                column: "id".to_string(),
                key_columns: vec!["id".to_string()],
                unique: true,
                primary_key: false,
                unique_constraint: true,
            },
        ]
    );

    let duplicate_insert = e
        .execute_text(3, "INSERT INTO people (id, name) VALUES (3, 'Ada')")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_insert.contains("duplicate key value violates unique index"),
        "{duplicate_insert}"
    );
    let duplicate_update = e
        .execute_text(4, "UPDATE people SET id = 1 WHERE name = 'Grace'")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_update.contains("duplicate key value violates unique index"),
        "{duplicate_update}"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        indexes
    );

    let alter = Engine::new_local_cpu_oracle();
    alter
        .execute_text(1, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    alter
        .execute_text(
            2,
            "INSERT INTO teams (id, name) VALUES (1, 'core'), (2, 'db')",
        )
        .unwrap();
    alter
        .execute_text(
            3,
            "ALTER TABLE ONLY public.teams ADD CONSTRAINT teams_name_key UNIQUE (name)",
        )
        .unwrap();
    assert!(alter
        .execute_text(4, "INSERT INTO teams (id, name) VALUES (3, 'core')")
        .unwrap_err()
        .to_string()
        .contains("duplicate key value violates unique index"));
}

#[test]
fn relational_check_constraints_enforce_and_replay_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT, name TEXT, CONSTRAINT people_id_positive CHECK (id > 0))",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();

    let checks = e
        .relational_catalog_table("people")
        .unwrap()
        .check_constraints
        .clone();
    assert_eq!(
        checks,
        vec![RelationalCheckConstraint {
            name: "people_id_positive".to_string(),
            column: "id".to_string(),
            op: SelectFilterOp::Gt,
            value: SqlValue::Int4(0),
        }]
    );

    let invalid_insert = e
        .execute_text(3, "INSERT INTO people (id, name) VALUES (-1, 'Bad')")
        .unwrap_err()
        .to_string();
    assert!(
        invalid_insert.contains("violates check constraint"),
        "{invalid_insert}"
    );
    let invalid_update = e
        .execute_text(4, "UPDATE people SET id = -2 WHERE name = 'Grace'")
        .unwrap_err()
        .to_string();
    assert!(
        invalid_update.contains("violates check constraint"),
        "{invalid_update}"
    );

    let Command::Select(select) =
        parse_command("SELECT id, name FROM people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Grace".to_string())],
        ]
    );

    e.execute_text(
        5,
        "ALTER TABLE ONLY public.people RENAME CONSTRAINT people_id_positive TO people_id_gt_zero",
    )
    .unwrap();
    e.execute_text(6, "ALTER TABLE public.people RENAME COLUMN id TO person_id")
        .unwrap();
    let table = e.relational_catalog_table("people").unwrap();
    assert_eq!(table.check_constraints[0].name, "people_id_gt_zero");
    assert_eq!(table.check_constraints[0].column, "person_id");
    let renamed_checks = table.check_constraints.clone();
    assert!(e
        .execute_text(7, "ALTER TABLE public.people DROP COLUMN person_id")
        .unwrap_err()
        .to_string()
        .contains("index or constraint depends on it"));

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .check_constraints,
        renamed_checks
    );

    let alter = Engine::new_local_cpu_oracle();
    alter
        .execute_text(1, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    alter
        .execute_text(
            2,
            "INSERT INTO teams (id, name) VALUES (1, 'core'), (-1, 'bad')",
        )
        .unwrap();
    let existing_rows = alter
        .execute_text(
            3,
            "ALTER TABLE ONLY public.teams ADD CONSTRAINT teams_id_positive CHECK (id > 0)",
        )
        .unwrap_err()
        .to_string();
    assert!(
        existing_rows.contains("violates check constraint")
            || existing_rows.contains("violated by some row"),
        "{existing_rows}"
    );
    alter
        .execute_text(4, "CREATE TABLE valid_teams (id INT, name TEXT)")
        .unwrap();
    alter
            .execute_text(
                5,
                "ALTER TABLE ONLY public.valid_teams ADD CONSTRAINT valid_teams_id_positive CHECK (id > 0)",
            )
            .unwrap();
    assert!(alter
        .execute_text(6, "INSERT INTO valid_teams (id, name) VALUES (-2, 'bad')")
        .unwrap_err()
        .to_string()
        .contains("violates check constraint"));
}

/// PG 3VL: a CHECK constraint is violated only when its predicate evaluates to FALSE — a NULL
/// operand makes it UNKNOWN, which SATISFIES the constraint ("the check expression should ...
/// yield true or the null value"). Pins: (a) an INSERT with a NULL checked value succeeds; (b) an
/// UPDATE setting the checked column to NULL succeeds; (c) ADD CHECK over existing NULL rows
/// succeeds; (d) a FALSE value still rejects everywhere. Was: NULL wrongly treated as a violation
/// (`select_filter_matches` returns false on NULL — correct for WHERE, wrong for CHECK).
#[test]
fn check_constraint_null_is_satisfied_pg_semantics() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE m (id INT PRIMARY KEY, v INT, CONSTRAINT v_pos CHECK (v > 0))",
    )
    .unwrap();
    // (a) NULL passes the CHECK on INSERT.
    e.execute_text(2, "INSERT INTO m (id, v) VALUES (1, NULL)")
        .unwrap();
    // (d) FALSE still rejects.
    assert!(e
        .execute_text(3, "INSERT INTO m (id, v) VALUES (2, -1)")
        .unwrap_err()
        .to_string()
        .contains("violates check constraint"));
    e.execute_text(4, "INSERT INTO m (id, v) VALUES (3, 5)")
        .unwrap();
    // (b) an UPDATE to NULL passes; an UPDATE to a FALSE value rejects.
    e.execute_text(5, "UPDATE m SET v = NULL WHERE id = 3")
        .unwrap();
    assert!(e
        .execute_text(6, "UPDATE m SET v = -7 WHERE id = 1")
        .unwrap_err()
        .to_string()
        .contains("violates check constraint"));
    // (c) ADD CHECK over existing NULL rows succeeds (both rows now hold v = NULL).
    e.execute_text(7, "ALTER TABLE m ADD CONSTRAINT v_cap CHECK (v < 1000)")
        .unwrap();
    // And ADD CHECK still rejects when a NON-NULL row violates.
    e.execute_text(8, "INSERT INTO m (id, v) VALUES (4, 500)")
        .unwrap();
    assert!(e
        .execute_text(9, "ALTER TABLE m ADD CONSTRAINT v_tiny CHECK (v < 100)")
        .unwrap_err()
        .to_string()
        .to_lowercase()
        .contains("violated"));
}

/// PG 3VL (MATCH SIMPLE): a NULL foreign-key value SATISFIES the constraint — it references
/// nothing, so no provider is required. Pins INSERT-NULL-fk passes, UPDATE-to-NULL-fk passes,
/// and a real missing key still rejects (both validator families share the rule).
#[test]
fn foreign_key_null_is_satisfied_pg_semantics() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE p (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE c (id INT PRIMARY KEY, pid INT)")
        .unwrap();
    e.execute_text(
        3,
        "ALTER TABLE ONLY c ADD CONSTRAINT c_fk FOREIGN KEY (pid) REFERENCES p(id)",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO p VALUES (1, 'a')").unwrap();
    // NULL fk passes (references nothing).
    e.execute_text(5, "INSERT INTO c (id, pid) VALUES (10, NULL)")
        .unwrap();
    // A real missing key still rejects.
    assert!(e
        .execute_text(6, "INSERT INTO c VALUES (11, 999)")
        .unwrap_err()
        .to_string()
        .contains("foreign key"));
    e.execute_text(7, "INSERT INTO c VALUES (12, 1)").unwrap();
    // UPDATE a valid fk to NULL passes.
    e.execute_text(8, "UPDATE c SET pid = NULL WHERE id = 12")
        .unwrap();
    // Deleting the now-unreferenced parent succeeds (NULL fks never pin a provider).
    e.execute_text(9, "DELETE FROM p WHERE id = 1").unwrap();
}

#[test]
fn relational_foreign_keys_enforce_and_replay_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE customers (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT)",
    )
    .unwrap();
    e.execute_text(3, "INSERT INTO customers (id, name) VALUES (1, 'Ada')")
        .unwrap();
    e.execute_text(4, "INSERT INTO orders (id, customer_id) VALUES (10, 1)")
        .unwrap();
    e.execute_text(
            5,
            "ALTER TABLE ONLY orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES customers(id)",
        )
        .unwrap();

    let invalid_insert = e
        .execute_text(6, "INSERT INTO orders (id, customer_id) VALUES (11, 99)")
        .unwrap_err()
        .to_string();
    assert!(invalid_insert.contains("violates foreign key constraint"));
    let invalid_child_update = e
        .execute_text(7, "UPDATE orders SET customer_id = 99 WHERE id = 10")
        .unwrap_err()
        .to_string();
    assert!(invalid_child_update.contains("violates foreign key constraint"));
    let invalid_parent_delete = e
        .execute_text(8, "DELETE FROM customers WHERE id = 1")
        .unwrap_err()
        .to_string();
    assert!(invalid_parent_delete.contains("violates foreign key constraint"));

    e.execute_text(
        9,
        "ALTER TABLE ONLY orders RENAME CONSTRAINT orders_customer_fk TO orders_customer_ref_fk",
    )
    .unwrap();
    e.execute_text(
        10,
        "ALTER TABLE ONLY customers RENAME COLUMN id TO customer_id",
    )
    .unwrap();
    e.execute_text(
        11,
        "ALTER TABLE IF EXISTS ONLY orders DROP CONSTRAINT orders_customer_ref_fk",
    )
    .unwrap();
    e.execute_text(12, "DELETE FROM customers WHERE customer_id = 1")
        .unwrap();

    let replayed = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let orders = replayed.relational_catalog_table("orders").unwrap();
    assert!(orders.foreign_keys.is_empty());
    let customers = replayed.relational_catalog_table("customers").unwrap();
    assert_eq!(customers.columns[0].name, "customer_id");
}

#[test]
fn relational_primary_key_rejects_duplicates_and_replays_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();

    let indexes = e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        indexes,
        vec![RelationalIndex {
            name: "people_pkey".to_string(),
            table: "people".to_string(),
            column: "id".to_string(),
            key_columns: vec!["id".to_string()],
            unique: true,
            primary_key: true,
            unique_constraint: false,
        }]
    );

    let duplicate_insert = e
        .execute_text(3, "INSERT INTO people (id, name) VALUES (1, 'Edsger')")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_insert.contains("duplicate key value violates unique index"),
        "{duplicate_insert}"
    );
    let duplicate_update = e
        .execute_text(4, "UPDATE people SET id = 1 WHERE name = 'Grace'")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_update.contains("duplicate key value violates unique index"),
        "{duplicate_update}"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        indexes
    );

    let alter = Engine::new_local_cpu_oracle();
    alter
        .execute_text(1, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    alter
        .execute_text(
            2,
            "INSERT INTO teams (id, name) VALUES (1, 'core'), (2, 'db')",
        )
        .unwrap();
    alter
        .execute_text(
            3,
            "ALTER TABLE ONLY public.teams ADD CONSTRAINT teams_pkey PRIMARY KEY (id)",
        )
        .unwrap();
    assert!(alter
        .execute_text(4, "INSERT INTO teams (id, name) VALUES (1, 'dup')")
        .unwrap_err()
        .to_string()
        .contains("duplicate key value violates unique index"));

    let duplicate_existing = Engine::new_local_cpu_oracle();
    duplicate_existing
        .execute_text(1, "CREATE TABLE dupes (id INT, name TEXT)")
        .unwrap();
    duplicate_existing
        .execute_text(2, "INSERT INTO dupes (id, name) VALUES (1, 'a'), (1, 'b')")
        .unwrap();
    let add_err = duplicate_existing
        .execute_text(
            3,
            "ALTER TABLE ONLY public.dupes ADD CONSTRAINT dupes_pkey PRIMARY KEY (id)",
        )
        .unwrap_err()
        .to_string();
    assert!(
        add_err.contains("duplicate key value violates unique index"),
        "{add_err}"
    );
}

#[test]
fn relational_catalog_drops_index_and_replays_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(3, "DROP INDEX public.people_name_idx")
        .unwrap();
    assert!(e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .is_empty());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .is_empty());

    e.execute_text(4, "DROP INDEX IF EXISTS people_name_idx")
        .unwrap();
    let missing_err = e
        .execute_text(5, "DROP INDEX people_name_idx")
        .unwrap_err()
        .to_string();
    assert!(
        missing_err.contains("index \"people_name_idx\" does not exist"),
        "{missing_err}"
    );

    let multi = Engine::new_local_cpu_oracle();
    multi
        .execute_text(
            1,
            "CREATE TABLE people (id INT PRIMARY KEY, name TEXT, city TEXT)",
        )
        .unwrap();
    multi
        .execute_text(
            2,
            "INSERT INTO people (id, name, city) VALUES (1, 'Ada', 'London')",
        )
        .unwrap();
    multi
        .execute_text(3, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    multi
        .execute_text(4, "CREATE INDEX people_city_idx ON people (city)")
        .unwrap();
    multi
        .execute_text(
            5,
            "COMMENT ON INDEX public.people_name_idx IS 'name lookup'",
        )
        .unwrap();
    multi
        .execute_text(
            6,
            "COMMENT ON INDEX public.people_city_idx IS 'city lookup'",
        )
        .unwrap();
    let partial_err = multi
        .execute_text(7, "DROP INDEX public.people_name_idx, public.missing_idx")
        .unwrap_err()
        .to_string();
    assert!(
        partial_err.contains("index \"missing_idx\" does not exist"),
        "{partial_err}"
    );
    assert_eq!(
        multi.relational_index_comment("people_name_idx").as_deref(),
        Some("name lookup")
    );
    multi
        .execute_text(
            8,
            "DROP INDEX public.people_name_idx, public.people_city_idx",
        )
        .unwrap();
    let table_indexes = multi
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        table_indexes,
        vec![RelationalIndex {
            name: "people_pkey".to_string(),
            table: "people".to_string(),
            column: "id".to_string(),
            key_columns: vec!["id".to_string()],
            unique: true,
            primary_key: true,
            unique_constraint: false,
        }]
    );
    assert_eq!(multi.relational_index_comment("people_name_idx"), None);
    assert_eq!(multi.relational_index_comment("people_city_idx"), None);
    let Command::Select(select) =
        parse_command("SELECT id, name FROM people WHERE id = 1").unwrap()
    else {
        panic!("expected SELECT");
    };
    let rows = multi.execute_relational_select(&select).unwrap();
    assert_eq!(
        rows.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]]
    );

    let recovered = Engine::recover_from_durable_wal(&multi.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        table_indexes
    );
}

#[test]
fn relational_catalog_renames_index_and_replays_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(3, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(4, "COMMENT ON INDEX public.people_name_idx IS 'lookup'")
        .unwrap();
    e.execute_text(
        5,
        "ALTER INDEX public.people_name_idx RENAME TO people_lookup_idx",
    )
    .unwrap();

    let table = e.relational_catalog_table("people").unwrap();
    let renamed_indexes = table.indexes.clone();
    assert_eq!(
        renamed_indexes,
        vec![RelationalIndex {
            name: "people_lookup_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: false,
            primary_key: false,
            unique_constraint: false,
        }]
    );
    assert_eq!(
        e.relational_index_comment("people_lookup_idx").as_deref(),
        Some("lookup")
    );
    assert_eq!(e.relational_index_comment("people_name_idx"), None);

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Linus'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(2)]]);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        renamed_indexes
    );
    assert_eq!(
        recovered
            .relational_index_comment("people_lookup_idx")
            .as_deref(),
        Some("lookup")
    );

    let duplicate_err = e
        .execute_text(
            6,
            "ALTER INDEX people_lookup_idx RENAME TO people_lookup_idx",
        )
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_err.contains("relation \"people_lookup_idx\" already exists"),
        "{duplicate_err}"
    );
    let missing_err = e
        .execute_text(7, "ALTER INDEX people_name_idx RENAME TO people_old_idx")
        .unwrap_err()
        .to_string();
    assert!(
        missing_err.contains("index \"people_name_idx\" does not exist"),
        "{missing_err}"
    );

    let constrained = Engine::new_local_cpu_oracle();
    constrained
        .execute_text(1, "CREATE TABLE keyed_people (id INT PRIMARY KEY)")
        .unwrap();
    let constraint_err = constrained
        .execute_text(
            2,
            "ALTER INDEX keyed_people_pkey RENAME TO keyed_people_id_idx",
        )
        .unwrap_err()
        .to_string();
    assert!(
        constraint_err.contains("cannot rename constraint-backed index"),
        "{constraint_err}"
    );
}

#[test]
fn relational_catalog_drops_constraints_and_replays_from_wal() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT PRIMARY KEY, name TEXT UNIQUE)",
    )
    .unwrap();
    e.execute_text(
        2,
        "COMMENT ON INDEX public.people_name_key IS 'name uniqueness'",
    )
    .unwrap();
    e.execute_text(
        3,
        "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'row identity'",
    )
    .unwrap();
    e.execute_text(
        4,
        "ALTER TABLE IF EXISTS ONLY public.people DROP CONSTRAINT IF EXISTS people_pkey",
    )
    .unwrap();
    e.execute_text(
        5,
        "ALTER TABLE ONLY public.people DROP CONSTRAINT people_name_key",
    )
    .unwrap();

    let table = e.relational_catalog_table("people").unwrap();
    assert!(table.indexes.is_empty());
    assert_eq!(e.relational_index_comment("people_name_key"), None);
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey"),
        None
    );
    e.execute_text(
        6,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (1, 'Ada')",
    )
    .unwrap();

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .is_empty());
    assert_eq!(recovered.relational_index_comment("people_name_key"), None);
    assert_eq!(
        recovered.relational_constraint_comment("people", "people_pkey"),
        None
    );

    e.execute_text(
        7,
        "ALTER TABLE ONLY public.people DROP CONSTRAINT IF EXISTS people_pkey",
    )
    .unwrap();
    let missing_constraint = e
        .execute_text(
            8,
            "ALTER TABLE ONLY public.people DROP CONSTRAINT people_pkey",
        )
        .unwrap_err()
        .to_string();
    assert!(
        missing_constraint.contains("constraint \"people_pkey\" does not exist"),
        "{missing_constraint}"
    );
    let missing_table = e
        .execute_text(
            9,
            "ALTER TABLE ONLY public.missing_people DROP CONSTRAINT people_pkey",
        )
        .unwrap_err()
        .to_string();
    assert!(
        missing_table.contains("relation \"missing_people\" does not exist"),
        "{missing_table}"
    );
}

#[test]
fn relational_catalog_drops_table_and_replays_from_wal() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT PRIMARY KEY, name TEXT UNIQUE)",
    )
    .unwrap();
    e.execute_text(2, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();
    e.execute_text(4, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(5, "COMMENT ON TABLE public.people IS 'people table'")
        .unwrap();
    e.execute_text(6, "COMMENT ON COLUMN public.people.name IS 'display name'")
        .unwrap();
    e.execute_text(7, "COMMENT ON INDEX public.people_name_idx IS 'lookup'")
        .unwrap();
    e.execute_text(
        8,
        "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'identity'",
    )
    .unwrap();
    e.execute_text(9, "COMMENT ON ROLE postgres IS 'bootstrap role'")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("people").unwrap();
    assert!(snapshot.is_valid());
    // TYPE-COVERAGE #14 (text): a PK'd text table is SHARD-resident (device-authoritative), not
    // single-buffer, so check either representation.
    assert!(
        e.relational_residency_snapshot("people").is_some() || e.resident_shard_count("people") > 0
    );
    e.execute_text(10, "DROP TABLE public.people").unwrap();

    assert!(e.relational_catalog_table("people").is_none());
    assert!(e.relational_catalog_table("teams").is_some());
    assert!(
        e.relational_residency_snapshot("people").is_none()
            && e.resident_shard_count("people") == 0
    );
    assert!(!e.read_state.residency.device_memory.contains_key("people"));
    assert_eq!(e.relational_table_comment("people"), None);
    assert_eq!(e.relational_column_comment("people", 2), None);
    assert_eq!(e.relational_index_comment("people_name_idx"), None);
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey"),
        None
    );
    assert_eq!(
        e.relational_role_comment("postgres").as_deref(),
        Some("bootstrap role")
    );
    let missing_select = e
        .execute_text(11, "INSERT INTO people (id, name) VALUES (3, 'Edsger')")
        .unwrap_err()
        .to_string();
    assert!(
        missing_select.contains("relation \"people\" does not exist"),
        "{missing_select}"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_catalog_table("people").is_none());
    assert!(recovered.relational_catalog_table("teams").is_some());
    assert_eq!(recovered.relational_table_comment("people"), None);
    assert_eq!(
        recovered.relational_role_comment("postgres").as_deref(),
        Some("bootstrap role")
    );

    e.execute_text(12, "DROP TABLE IF EXISTS people").unwrap();
    let missing_drop = e
        .execute_text(13, "DROP TABLE people")
        .unwrap_err()
        .to_string();
    assert!(
        missing_drop.contains("relation \"people\" does not exist"),
        "{missing_drop}"
    );

    let with_view = Engine::new_local_cpu_oracle();
    with_view
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    with_view
        .execute_text(2, "CREATE VIEW public.people_view AS SELECT * FROM people")
        .unwrap();
    let view_drop = with_view
        .execute_text(3, "DROP TABLE people_view")
        .unwrap_err()
        .to_string();
    assert!(
        view_drop.contains("relation \"people_view\" is not a table"),
        "{view_drop}"
    );
    assert!(with_view.relational_catalog_view("people_view").is_some());
}

#[test]
fn relational_catalog_drops_table_batches_atomically_and_replays() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE batch_people (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    e.execute_text(2, "CREATE TABLE batch_teams (id INT, name TEXT UNIQUE)")
        .unwrap();
    e.execute_text(3, "CREATE TABLE batch_keep (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO batch_people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();
    e.execute_text(
        5,
        "INSERT INTO batch_teams (id, name) VALUES (10, 'Compiler')",
    )
    .unwrap();
    e.execute_text(
        6,
        "CREATE INDEX batch_people_name_idx ON batch_people (name)",
    )
    .unwrap();
    e.execute_text(7, "COMMENT ON TABLE public.batch_people IS 'people table'")
        .unwrap();
    e.execute_text(
        8,
        "COMMENT ON COLUMN public.batch_teams.name IS 'team name'",
    )
    .unwrap();
    e.execute_text(
        9,
        "COMMENT ON INDEX public.batch_people_name_idx IS 'lookup'",
    )
    .unwrap();
    assert!(e
        .populate_relational_residency_snapshot("batch_people")
        .unwrap()
        .is_valid());

    e.execute_text(10, "DROP TABLE public.batch_people, public.batch_teams")
        .unwrap();

    assert!(e.relational_catalog_table("batch_people").is_none());
    assert!(e.relational_catalog_table("batch_teams").is_none());
    assert!(e.relational_catalog_table("batch_keep").is_some());
    assert_eq!(e.relational_table_comment("batch_people"), None);
    assert_eq!(e.relational_column_comment("batch_teams", 2), None);
    assert_eq!(e.relational_index_comment("batch_people_name_idx"), None);
    assert!(e.relational_residency_snapshot("batch_people").is_none());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_catalog_table("batch_people").is_none());
    assert!(recovered.relational_catalog_table("batch_teams").is_none());
    assert!(recovered.relational_catalog_table("batch_keep").is_some());

    let atomic = Engine::new_local_cpu_oracle();
    atomic
        .execute_text(1, "CREATE TABLE atomic_people (id INT, name TEXT)")
        .unwrap();
    atomic
        .execute_text(2, "CREATE TABLE atomic_teams (id INT, name TEXT)")
        .unwrap();
    atomic
        .execute_text(
            3,
            "CREATE VIEW public.atomic_view AS SELECT id, name FROM atomic_people",
        )
        .unwrap();
    let missing = atomic
        .execute_text(4, "DROP TABLE atomic_people, missing_atomic")
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("relation \"missing_atomic\" does not exist"),
        "{missing}"
    );
    assert!(atomic.relational_catalog_table("atomic_people").is_some());
    assert!(atomic.relational_catalog_table("atomic_teams").is_some());

    let duplicate = atomic
        .execute_text(5, "DROP TABLE atomic_people, atomic_people")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("table \"atomic_people\" specified more than once"),
        "{duplicate}"
    );
    assert!(atomic.relational_catalog_table("atomic_people").is_some());

    let view_target = atomic
        .execute_text(6, "DROP TABLE IF EXISTS missing_atomic, atomic_view")
        .unwrap_err()
        .to_string();
    assert!(
        view_target.contains("relation \"atomic_view\" is not a table"),
        "{view_target}"
    );
    assert!(atomic.relational_catalog_view("atomic_view").is_some());
    assert!(atomic.relational_catalog_table("atomic_people").is_some());

    atomic
        .execute_text(7, "DROP TABLE IF EXISTS missing_atomic, atomic_people")
        .unwrap();
    assert!(atomic.relational_catalog_table("atomic_people").is_none());
    assert!(atomic.relational_catalog_table("atomic_teams").is_some());
}
