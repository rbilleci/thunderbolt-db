use super::*;

#[test]
fn snapshot_export_tracks_last_applied_index() {
    let mut e = Engine::new_local_cpu_oracle();
    let token = e.commit_mutation(1, b"SET a=1".to_vec().into()).unwrap();

    let exported = e.export_snapshot_meta();
    let current = e.snapshot_meta();

    assert_eq!(exported.last_included_index, token.index);
    assert_eq!(current.last_included_index, token.index);
    assert_eq!(exported.snapshot_id, 1);
    assert_eq!(current.snapshot_id, 1);
}

#[test]
fn install_snapshot_advances_visible_and_replication_watermarks() {
    let mut e = Engine::new_local_cpu_oracle();
    e.install_snapshot(SnapshotMeta {
        last_included_index: 7,
        last_included_term: 3,
        snapshot_id: 11,
    });

    let marks = e.replication_watermarks();
    assert_eq!(marks.term, 3);
    assert_eq!(marks.max_replication_gap(), 0);
    assert_eq!(marks.total_backlog_items(), 0);
    assert!(marks.is_fully_caught_up());
    assert_eq!(marks.commit_index, 7);
    assert_eq!(marks.applied_index, 7);
    assert_eq!(marks.visible_index, 7);
    assert_eq!(marks.commit_apply_gap, 0);
    assert_eq!(marks.apply_visible_gap, 0);
    assert_eq!(marks.snapshot_id, 11);

    let next = e.commit_mutation(2, b"SET b=2".to_vec().into()).unwrap();
    assert_eq!(next.index, 8);
    assert_eq!(e.get("b").as_deref(), Some("2"));
}

#[test]
fn export_snapshot_meta_is_reflected_in_replication_watermarks() {
    let mut e = Engine::new_local_cpu_oracle();

    assert_eq!(e.replication_watermarks().snapshot_id, 0);

    e.export_snapshot_meta();
    assert_eq!(e.replication_watermarks().snapshot_id, 1);

    e.export_snapshot_meta();
    assert_eq!(e.replication_watermarks().snapshot_id, 2);
}

#[test]
fn relational_residency_snapshot_accounts_bytes_and_invalidates_on_later_wal_apply() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(snapshot.gpu_id, 0);
    assert_eq!(snapshot.table, "events");
    assert_eq!(snapshot.row_count, 2);
    assert_eq!(snapshot.column_count, 2);
    assert_eq!(snapshot.resident_device_int4_columns, vec!["id"]);
    assert!(snapshot.resident_bytes >= 8 + "alpha".len() as u64 + "beta".len() as u64);
    assert_eq!(snapshot.valid_through_index, e.visible_up_to());
    assert!(snapshot.is_valid());
    assert!(!snapshot.memory_pressure_active);
    assert!(snapshot.last_refresh_cost.is_some());

    e.mark_gpu_memory_pressured(0);
    let pressured = e.relational_residency_snapshot("events").unwrap();
    assert!(pressured.memory_pressure_active);
    assert!(pressured.invalidated_by_memory_pressure);
    assert!(!pressured.is_valid());

    let valid_through = pressured.valid_through_index;
    let error = e
        .execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap_err();
    assert!(error.to_string().contains("device DML generation"));
    e.clear_gpu_memory_pressured(0);
    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let refreshed = e.relational_residency_snapshot("events").unwrap();
    assert_eq!(refreshed.row_count, 3);
    assert!(refreshed.resident_bytes > snapshot.resident_bytes);
    assert_eq!(refreshed.valid_through_index, e.visible_up_to());
    assert!(refreshed.is_valid());
    assert!(!refreshed.memory_pressure_active);
    assert!(!refreshed.invalidated_by_memory_pressure);
    let refresh_cost = refreshed.last_refresh_cost.unwrap();
    assert_eq!(refresh_cost.previous_row_count, 2);
    assert_eq!(refresh_cost.refreshed_row_count, 3);
    assert_eq!(refresh_cost.row_delta, 1);
    assert_eq!(
        refresh_cost.previous_resident_bytes,
        snapshot.resident_bytes
    );
    assert_eq!(
        refresh_cost.refreshed_resident_bytes,
        refreshed.resident_bytes
    );
    assert!(refresh_cost.resident_byte_delta > 0);
    assert_eq!(refresh_cost.refreshed_from_index, valid_through);
    assert_eq!(
        refresh_cost.refreshed_through_index,
        refreshed.valid_through_index
    );
    assert_eq!(refresh_cost.invalidated_by_txn_id, Some(3));
    assert!(!refresh_cost.invalidated_by_memory_pressure);
}

#[test]
fn mutation_maintains_only_the_mutated_device_generation() {
    // R3-004: a write maintains its target generation without disturbing another table.
    let mut e = Engine::new_local_cpu_oracle();
    // THE FLIP: this test exercises the SINGLE-BUFFER layer's semantics (a supported, settable
    // configuration; sharded is the default) — pin the layout it tests.
    e.set_shard_residency_enabled(false);
    e.execute_text(1, "CREATE TABLE a (id INT)").unwrap();
    e.execute_text(2, "CREATE TABLE b (id INT)").unwrap();
    e.execute_text(3, "INSERT INTO a (id) VALUES (1)").unwrap();
    e.execute_text(4, "INSERT INTO b (id) VALUES (1)").unwrap();
    e.populate_relational_residency_snapshot("a").unwrap();
    e.populate_relational_residency_snapshot("b").unwrap();
    assert!(e.relational_residency_snapshot("a").unwrap().is_valid());
    assert!(e.relational_residency_snapshot("b").unwrap().is_valid());

    e.execute_text(5, "INSERT INTO a (id) VALUES (2)").unwrap();

    let a = e.relational_residency_snapshot("a").unwrap();
    assert_eq!(a.invalidated_by_txn_id, None);
    assert!(a.is_valid());
    assert_eq!(a.valid_through_index, 5);
    let b = e.relational_residency_snapshot("b").unwrap();
    assert_eq!(
        b.invalidated_by_txn_id, None,
        "table b residency must survive a write to table a (per-table invalidation)"
    );
    assert!(b.is_valid());
}

#[test]
fn create_table_does_not_invalidate_existing_residency() {
    // CREATE TABLE introduces a brand-new table with no prior residency, so it must
    // touch no existing table's snapshot (scope contributes the empty set).
    let mut e = Engine::new_local_cpu_oracle();
    // THE FLIP: this test exercises the SINGLE-BUFFER layer's semantics (a supported, settable
    // configuration; sharded is the default) — pin the layout it tests.
    e.set_shard_residency_enabled(false);
    e.execute_text(1, "CREATE TABLE a (id INT)").unwrap();
    e.execute_text(2, "INSERT INTO a (id) VALUES (1)").unwrap();
    e.populate_relational_residency_snapshot("a").unwrap();
    assert!(e.relational_residency_snapshot("a").unwrap().is_valid());

    e.execute_text(3, "CREATE TABLE c (id INT)").unwrap();
    assert!(
        e.relational_residency_snapshot("a").unwrap().is_valid(),
        "CREATE TABLE must not invalidate an unrelated resident table"
    );
}

#[test]
fn unscoped_ddl_refreshes_previously_authoritative_unrelated_residency() {
    // A schema change still invalidates globally at publication, then refreshes every previously
    // authoritative device generation before service resumes.
    let mut e = Engine::new_local_cpu_oracle();
    // THE FLIP: this test exercises the SINGLE-BUFFER layer's semantics (a supported, settable
    // configuration; sharded is the default) — pin the layout it tests.
    e.set_shard_residency_enabled(false);
    e.execute_text(1, "CREATE TABLE a (id INT)").unwrap();
    e.execute_text(2, "CREATE TABLE b (id INT)").unwrap();
    e.execute_text(3, "INSERT INTO a (id) VALUES (1)").unwrap();
    e.execute_text(4, "INSERT INTO b (id) VALUES (1)").unwrap();
    e.populate_relational_residency_snapshot("a").unwrap();
    e.populate_relational_residency_snapshot("b").unwrap();
    assert!(e.relational_residency_snapshot("b").unwrap().is_valid());

    e.execute_text(5, "ALTER TABLE a ADD COLUMN note INT DEFAULT 0")
        .unwrap();

    let b = e.relational_residency_snapshot("b").unwrap();
    assert!(b.is_valid());
    assert_eq!(b.invalidated_by_txn_id, None);
    assert_eq!(b.valid_through_index, 5);
}

#[test]
fn residency_invalidation_scope_narrows_dml_and_falls_back_on_unknown() {
    fn entry(index: Index, sql: &str) -> LogEntry {
        LogEntry {
            term: 1,
            index,
            payload: sql.as_bytes().to_vec().into(),
        }
    }

    // single-table DML -> exactly that table
    assert_eq!(
        Engine::residency_invalidation_scope(&[entry(1, "INSERT INTO a (id) VALUES (1)")]),
        Some(BTreeSet::from(["a".to_string()]))
    );
    // DML across tables -> the union
    assert_eq!(
        Engine::residency_invalidation_scope(&[
            entry(1, "INSERT INTO a (id) VALUES (1)"),
            entry(2, "INSERT INTO b (id) VALUES (1)"),
        ]),
        Some(BTreeSet::from(["a".to_string(), "b".to_string()]))
    );
    // CREATE TABLE introduces no prior residency -> empty set (narrowed, not global)
    assert_eq!(
        Engine::residency_invalidation_scope(&[entry(1, "CREATE TABLE d (id INT)")]),
        Some(BTreeSet::new())
    );
    // an unscoped command anywhere in the batch -> conservative global (None)
    assert_eq!(
        Engine::residency_invalidation_scope(&[
            entry(1, "INSERT INTO a (id) VALUES (1)"),
            entry(2, "ALTER TABLE a ADD COLUMN note INT DEFAULT 0"),
        ]),
        None
    );
    // an unparseable payload -> conservative global (None)
    assert_eq!(
        Engine::residency_invalidation_scope(&[entry(1, "this is not sql")]),
        None
    );
}

#[test]
fn residency_snapshot_retains_int8_columns_at_the_layout_offset() {
    // Type matrix (doc 19): the general GPU executor reads int8 predicates / projections from the
    // device payload, so residency retains int8 columns as fixed 8-byte row-major data AFTER the int4
    // section. Verify the bookkeeping (the column list + the offset resolver); the on-device read is
    // exercised by the int8 VM slice. CPU-side bookkeeping, so this runs without a GPU.
    let mut e = Engine::new_local_cpu_oracle();
    // i64-SECTION FLIP pin: this test verifies the SINGLE-BUFFER int8 layout bookkeeping —
    // the kill-switch configuration since the 2026-07-03 flip (default = sharded admission).
    e.set_shard_int8_section_enabled(false);
    e.execute_text(1, "CREATE TABLE t (a INT, big BIGINT, b INT, big2 BIGINT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a, big, b, big2) VALUES (1, 100, 2, 7), (3, 200, 4, 8)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();

    // int8 columns retained in catalog order (the int4 columns a, b are interspersed).
    assert_eq!(
        snapshot.resident_device_int8_columns,
        vec!["big".to_string(), "big2".to_string()]
    );

    // Offsets match the payload layout: header(8) + int4 section + int8_ordinal * rows * 8.
    let Command::Select(select) = parse_command("SELECT big FROM t").unwrap() else {
        unreachable!()
    };
    let (table, _bound, _) = e.bind_relational_select_for_execution(&select).unwrap();
    let row_count = snapshot.row_count as u64;
    let int4_section = snapshot.resident_device_int4_columns.len() as u64 * row_count * 4;
    let big_idx = relational_column_index(&table, "big").unwrap();
    let big2_idx = relational_column_index(&table, "big2").unwrap();
    assert_eq!(
        resident_device_int8_column_offset(&snapshot, &table, big_idx).unwrap(),
        8 + int4_section,
        "big is the first int8 column (ordinal 0)"
    );
    assert_eq!(
        resident_device_int8_column_offset(&snapshot, &table, big2_idx).unwrap(),
        8 + int4_section + row_count * 8,
        "big2 is the second int8 column (ordinal 1)"
    );
    // The type guard holds: the int8 resolver rejects an int4 column.
    let a_idx = relational_column_index(&table, "a").unwrap();
    assert!(resident_device_int8_column_offset(&snapshot, &table, a_idx).is_err());
}

#[test]
fn residency_snapshot_retains_numeric_columns_at_the_layout_offset() {
    // Type matrix (doc 19): the general GPU executor reads numeric predicates / projections from the
    // device payload, so residency retains numeric columns as fixed 16-byte i128 mantissas AFTER the
    // int4 AND int8 sections (before text). Verify the bookkeeping (the column list + the offset
    // resolver); the on-device read is exercised by the numeric VM slice. CPU-side, no GPU needed.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE t (a INT, big BIGINT, price NUMERIC(10,2), b INT, tax NUMERIC(10,2), label TEXT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a, big, price, b, tax, label) VALUES \
         (1, 100, 1.50, 2, 0.25, 'x'), (3, 200, 9.99, 4, 1.00, 'y')",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();

    // numeric columns retained in catalog order (the int4 a,b and int8 big are interspersed).
    assert_eq!(
        snapshot.resident_device_numeric_columns,
        vec!["price".to_string(), "tax".to_string()]
    );

    // Offsets: header(8) + the WHOLE int4 section + the WHOLE int8 section + numeric_ordinal*rows*16.
    let Command::Select(select) = parse_command("SELECT price FROM t").unwrap() else {
        unreachable!()
    };
    let (table, _bound, _) = e.bind_relational_select_for_execution(&select).unwrap();
    let row_count = snapshot.row_count as u64;
    let int4_section = snapshot.resident_device_int4_columns.len() as u64 * row_count * 4;
    let int8_section = snapshot.resident_device_int8_columns.len() as u64 * row_count * 8;
    let price_idx = relational_column_index(&table, "price").unwrap();
    let tax_idx = relational_column_index(&table, "tax").unwrap();
    assert_eq!(
        resident_device_numeric_column_offset(&snapshot, &table, price_idx).unwrap(),
        8 + int4_section + int8_section,
        "price is the first numeric column (ordinal 0), after the int4 + int8 sections"
    );
    assert_eq!(
        resident_device_numeric_column_offset(&snapshot, &table, tax_idx).unwrap(),
        8 + int4_section + int8_section + row_count * 16,
        "tax is the second numeric column (ordinal 1)"
    );
    // The type guard holds: the numeric resolver rejects a non-numeric (int8) column.
    let big_idx = relational_column_index(&table, "big").unwrap();
    assert!(resident_device_numeric_column_offset(&snapshot, &table, big_idx).is_err());
}
