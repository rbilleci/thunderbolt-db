use super::*;

#[test]
fn relational_catalog_truncates_table_and_replays_from_wal() {
    let mut e = Engine::new_local_test_engine();
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
    e.execute_text(4, "INSERT INTO teams (id, name) VALUES (9, 'Infra')")
        .unwrap();
    e.execute_text(5, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(6, "COMMENT ON TABLE public.people IS 'people table'")
        .unwrap();
    e.execute_text(7, "COMMENT ON COLUMN public.people.name IS 'display name'")
        .unwrap();
    e.execute_text(8, "COMMENT ON INDEX public.people_name_idx IS 'lookup'")
        .unwrap();
    e.execute_text(
        9,
        "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'identity'",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("people").unwrap();
    assert!(snapshot.is_valid());

    e.execute_text(10, "TRUNCATE TABLE ONLY public.people")
        .unwrap();

    let Command::Select(empty_people) =
        parse_command("SELECT id, name FROM people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert!(e
        .execute_relational_select(&empty_people)
        .unwrap()
        .rows
        .is_empty());
    assert!(e.relational_catalog_table("people").is_some());
    assert!(e.relational_catalog_table("teams").is_some());
    assert_eq!(
        e.relational_table_comment("people").as_deref(),
        Some("people table")
    );
    assert_eq!(
        e.relational_column_comment("people", 2).as_deref(),
        Some("display name")
    );
    assert_eq!(
        e.relational_index_comment("people_name_idx").as_deref(),
        Some("lookup")
    );
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey")
            .as_deref(),
        Some("identity")
    );
    // TYPE-COVERAGE #14 (text): a PK'd text table is SHARD-resident, so there is no single-buffer
    // snapshot to invalidate — the empty ORDER BY select above already proved TRUNCATE serves no stale
    // rows. Accept either an invalidated single-buffer snapshot (legacy) or the shard representation.
    assert!(e
        .relational_residency_snapshot("people")
        .is_none_or(|snapshot| !snapshot.is_valid()));

    e.execute_text(11, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    let result = e.execute_relational_select(&empty_people).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&empty_people).unwrap();
    assert_eq!(
        recovered_result.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]]
    );
    assert_eq!(
        recovered.relational_table_comment("people").as_deref(),
        Some("people table")
    );
    assert_eq!(
        recovered
            .relational_constraint_comment("people", "people_pkey")
            .as_deref(),
        Some("identity")
    );

    e.execute_text(12, "CREATE TABLE restart_people (id SERIAL, name TEXT)")
        .unwrap();
    e.execute_text(
        13,
        "INSERT INTO restart_people (name) VALUES ('Ada'), ('Grace')",
    )
    .unwrap();
    let restart_seq = e
        .relational_catalog_sequence("restart_people_id_seq")
        .unwrap();
    assert_eq!(restart_seq.last_value, 2);
    assert!(restart_seq.is_called);
    let wal_before_restart = e.durable_wal_records().len();
    let restart_error = e
        .execute_text(14, "TRUNCATE TABLE public.restart_people RESTART IDENTITY")
        .unwrap_err();
    assert!(matches!(restart_error, ExecuteError::Unsupported(_)));
    assert_eq!(e.durable_wal_records().len(), wal_before_restart);
    let restart_seq = e
        .relational_catalog_sequence("restart_people_id_seq")
        .unwrap();
    assert_eq!(restart_seq.last_value, 2);
    assert!(restart_seq.is_called);
    e.execute_text(15, "TRUNCATE TABLE public.restart_people CONTINUE IDENTITY")
        .unwrap();
    e.execute_text(16, "INSERT INTO restart_people (name) VALUES ('Linus')")
        .unwrap();
    let Command::Select(restart_select) =
        parse_command("SELECT id, name FROM restart_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert_eq!(
        e.execute_relational_select(&restart_select).unwrap().rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("Linus".to_string())]]
    );
    let recovered_restart = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered_restart
            .execute_relational_select(&restart_select)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("Linus".to_string())]]
    );
    let recovered_seq = recovered_restart
        .relational_catalog_sequence("restart_people_id_seq")
        .unwrap();
    assert_eq!(recovered_seq.last_value, 3);
    assert!(recovered_seq.is_called);

    let missing_truncate = e
        .execute_text(17, "TRUNCATE TABLE missing_people")
        .unwrap_err()
        .to_string();
    assert!(
        missing_truncate.contains("relation \"missing_people\" does not exist"),
        "{missing_truncate}"
    );

    let with_view = Engine::new_local_test_engine();
    with_view
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    with_view
        .execute_text(2, "CREATE VIEW public.people_view AS SELECT * FROM people")
        .unwrap();
    let view_truncate = with_view
        .execute_text(3, "TRUNCATE people_view")
        .unwrap_err()
        .to_string();
    assert!(
        view_truncate.contains("relation \"people_view\" is not a table"),
        "{view_truncate}"
    );
    assert!(with_view.relational_catalog_view("people_view").is_some());
}
