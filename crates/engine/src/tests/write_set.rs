use super::*;

/// A snapshot of every piece of engine state a DML `apply_delta` may mutate, used to assert
/// `prepare_*` is pure (no engine mutation).
#[derive(Debug, Clone, PartialEq, Eq)]
struct MutableEngineState {
    versions: Vec<TupleVersion>,
    value_index: BTreeMap<RelationalIndexKey, Vec<String>>,
    next_row_id: u64,
    sequences: BTreeMap<String, (i64, bool)>,
}

fn capture_mutable_state(e: &Engine) -> MutableEngineState {
    let mut versions = e.read_state.mvcc.all_versions();
    versions.sort_by(|a, b| (a.tuple_id, &a.value).cmp(&(b.tuple_id, &b.value)));
    MutableEngineState {
        versions,
        value_index: e.read_state.mvcc.value_index_snapshot(),
        next_row_id: e.read_state.mvcc.current_row_id(),
        sequences: e
            .ddl_catalog()
            .relational_sequences
            .iter()
            .map(|(name, seq)| (name.clone(), (seq.last_value, seq.is_called)))
            .collect(),
    }
}

/// The commit-seq the NEXT commit would receive under serialization (== `entry.index`), which
/// is what a directly-driven `prepare_*`/`apply_delta` pair must use to be byte-identical.
fn next_commit_snapshot(e: &Engine) -> DmlReadSnapshot {
    e.dml_read_snapshot(e.visible_up_to() + 1)
}

fn parse_insert(sql: &str) -> Insert {
    match parse_command(sql).unwrap() {
        Command::Insert(insert) => insert,
        other => panic!("expected INSERT, got {other:?}"),
    }
}

fn parse_update(sql: &str) -> Update {
    match parse_command(sql).unwrap() {
        Command::Update(update) => update,
        other => panic!("expected UPDATE, got {other:?}"),
    }
}

fn parse_delete(sql: &str) -> Delete {
    match parse_command(sql).unwrap() {
        Command::Delete(delete) => delete,
        other => panic!("expected DELETE, got {other:?}"),
    }
}

fn ensure_insert_device_generation(e: &Engine, insert: &Insert) {
    e.ensure_dml_device_generation(&Command::Insert(insert.clone()))
        .unwrap();
}

/// The set of relational row keys whose version chain `apply_delta` touched for `commit_seq`:
/// a NEW version created by `commit_seq` (insert / update-new) OR an EXISTING version
/// tombstoned by `commit_seq` (delete / update-old). Derived purely from the version chains so
/// it is independent of the write-set under test.
fn keys_touched_at(e: &Engine, commit_seq: TxnId) -> BTreeSet<String> {
    e.read_state
        .mvcc
        .all_versions()
        .into_iter()
        .filter(|v| v.created_by == commit_seq || v.deleted_by == Some(commit_seq))
        .map(|v| v.key)
        .collect()
}

#[test]
fn prepare_dml_does_not_mutate_engine_state() {
    // Stage 2 invariant (a): `prepare_*` is PURE — calling it leaves every mutable engine
    // structure (versions, value index, row-id counter, sequences) byte-for-byte unchanged.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (id, label) VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    .unwrap();

    let before = capture_mutable_state(&e);

    // INSERT prepare — no mutation.
    let snapshot = next_commit_snapshot(&e);
    let insert_delta = e
        .prepare_insert(
            &parse_insert("INSERT INTO t (id, label) VALUES (4, 'd')"),
            snapshot,
            None,
            InsertPrepareValidation::Full,
        )
        .unwrap();
    assert_eq!(
        capture_mutable_state(&e),
        before,
        "prepare_insert mutated engine state"
    );

    // UPDATE prepare — no mutation.
    let update_delta = e
        .prepare_update(
            &parse_update("UPDATE t SET label = 'z' WHERE id = 2"),
            next_commit_snapshot(&e),
        )
        .unwrap();
    assert_eq!(
        capture_mutable_state(&e),
        before,
        "prepare_update mutated engine state"
    );

    // DELETE prepare — no mutation.
    let delete_delta = e
        .prepare_delete(
            &parse_delete("DELETE FROM t WHERE id = 3"),
            next_commit_snapshot(&e),
        )
        .unwrap();
    assert_eq!(
        capture_mutable_state(&e),
        before,
        "prepare_delete mutated engine state"
    );

    // The prepared deltas are non-trivial (we actually exercised the work). An INSERT contributes
    // NO row keys to the conflict write-set (it claims a fresh row id at install time, so its row
    // slot can never truly conflict — BUG-1 fix); its non-triviality is the prepared mutation +
    // the row it will consume. UPDATE/DELETE DO record their (stable) row keys for same-row
    // conflict detection.
    assert!(insert_delta.write_set.rows.is_empty());
    assert_eq!(insert_delta.rows_consumed, 1);
    assert!(matches!(
        insert_delta.mutation,
        PreparedMutation::Insert { .. }
    ));
    assert!(!update_delta.write_set.rows.is_empty());
    assert!(!delete_delta.write_set.rows.is_empty());
}

#[test]
fn prepare_insert_with_sequence_default_is_pure_and_advances_on_apply() {
    // The trickiest purity case: a `nextval` (SERIAL) column default. `prepare_insert` must NOT
    // advance the sequence (pure), but the prepared delta must, and `apply_delta` must install
    // exactly the advancement the old in-line apply produced.
    let seq_name = "s_id_seq"; // SERIAL auto-creates `<table>_<col>_seq`.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE s (id SERIAL, v TEXT)")
        .unwrap();
    let seq_before = e
        .ddl_catalog()
        .relational_sequences
        .get(seq_name)
        .map(|s| (s.last_value, s.is_called));
    assert!(seq_before.is_some(), "implicit SERIAL sequence created");

    let before = capture_mutable_state(&e);
    let snapshot = next_commit_snapshot(&e);
    // Two rows, both consuming the default -> two nextval advances captured in the delta.
    let delta = e
        .prepare_insert(
            &parse_insert("INSERT INTO s (v) VALUES ('x'), ('y')"),
            snapshot,
            None,
            InsertPrepareValidation::Full,
        )
        .unwrap();
    assert_eq!(
        capture_mutable_state(&e),
        before,
        "prepare_insert advanced the sequence (not pure)"
    );
    let PreparedMutation::Insert {
        ref seq_advances,
        ref inserted_rows,
        ..
    } = delta.mutation
    else {
        panic!("expected Insert mutation");
    };
    assert!(
        seq_advances.contains_key(seq_name),
        "sequence advancement recorded in the delta"
    );
    // The two rows got the two successive sequence values (1, then 2 on a fresh seq).
    assert_eq!(inserted_rows[0].1[0], SqlValue::Int4(1));
    assert_eq!(inserted_rows[1].1[0], SqlValue::Int4(2));

    // Parity: a fresh engine running the SAME insert through the public path lands on the same
    // sequence state.
    let golden = Engine::new_local_cpu_oracle();
    golden
        .execute_text(1, "CREATE TABLE s (id SERIAL, v TEXT)")
        .unwrap();
    golden
        .execute_text(2, "INSERT INTO s (v) VALUES ('x'), ('y')")
        .unwrap();

    // Apply the prepared delta on `e` and compare the sequence state. A delta carrying nextval
    // advances goes through the SERIALIZED apply (`apply_delta_serialized`), which applies the
    // advance under `&mut self`; the `&self` `apply_delta` deliberately rejects seq-carrying
    // deltas (write-half Stage 4).
    let commit_seq = snapshot.commit_seq;
    {
        let mut cat = e.ddl_catalog();
        e.apply_delta_serialized(&mut cat, delta, commit_seq, None)
            .unwrap();
    }
    let seq_e = e
        .ddl_catalog()
        .relational_sequences
        .get(seq_name)
        .map(|s| (s.last_value, s.is_called));
    let seq_g = golden
        .ddl_catalog()
        .relational_sequences
        .get(seq_name)
        .map(|s| (s.last_value, s.is_called));
    assert_eq!(
        seq_e, seq_g,
        "sequence advanced to the same state as the public path"
    );
}

#[test]
fn prepare_insert_failing_preflight_advances_nothing() {
    // Pins the one deliberate refinement vs. the old in-line apply: when `prepare_insert`
    // fails its (unique) preflight, it advances NOTHING — not the sequence, not the row-id,
    // not the store. (The old apply advanced the sequence before preflighting.) This is the
    // Stage-4 abort-is-side-effect-free property; unobservable on live paths because
    // `execute_text` preflights before committing.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE p (id SERIAL, code INT UNIQUE)")
        .unwrap();
    e.execute_text(2, "INSERT INTO p (code) VALUES (100)")
        .unwrap();

    let before = capture_mutable_state(&e);
    // A second row with the SAME unique `code` -> unique preflight must reject it.
    let err = e.prepare_insert(
        &parse_insert("INSERT INTO p (code) VALUES (100)"),
        next_commit_snapshot(&e),
        None,
        InsertPrepareValidation::Full,
    );
    assert!(err.is_err(), "duplicate unique value must fail preflight");
    assert_eq!(
        capture_mutable_state(&e),
        before,
        "a failed prepare_insert advanced engine state (sequence/row-id/store)"
    );
}

#[test]
fn insert_write_set_is_independent_of_retired_host_apply() {
    // INSERT conflict-write-set invariant (BUG-1 fix): an INSERT claims a FRESH, unique row id at
    // install time, so its row slot can never truly collide with another writer's. Therefore the
    // conflict write-set records NO row keys for an INSERT (putting the predicted snapshot-base
    // key there would spuriously conflict two concurrent disjoint inserts), even though
    // `apply_delta` DOES install those row keys. Inserts conflict ONLY on the unique-index slots
    // they occupy.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO t (id, label) VALUES (1, 'a')")
        .unwrap();

    let snapshot = next_commit_snapshot(&e);
    let commit_seq = snapshot.commit_seq;
    let delta = e
        .prepare_insert(
            &parse_insert("INSERT INTO t (id, label) VALUES (2, 'b'), (3, 'c')"),
            snapshot,
            None,
            InsertPrepareValidation::Full,
        )
        .unwrap();
    // The conflict write-set carries NO insert row keys (BUG-1 fix) and (no unique index here) no
    // unique slots either: a plain INSERT has no genuine conflict dimension.
    assert!(
        delta.write_set.rows.is_empty(),
        "INSERT must contribute NO row keys to the conflict write-set"
    );
    assert!(delta.write_set.unique_slots.is_empty());
    assert_eq!(delta.rows_consumed, 2);

    // R3-004: the host/control-plane apply advances identity state but installs no relational rows;
    // the enclosing commit publishes the corresponding device append.
    e.apply_delta(delta, commit_seq, None).unwrap();
    let touched = keys_touched_at(&e, commit_seq);
    assert!(
        touched.is_empty(),
        "host tuple apply is retired: {touched:?}"
    );
    assert!(e.table_device_authoritative("t"));
}

#[test]
fn insert_write_set_records_unique_slots_but_not_row_keys() {
    // BUG-1 fix, complement: an INSERT into a table WITH a unique index records the unique slot it
    // occupies (the genuine first-committer-wins conflict dimension) but STILL records no row key.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE u (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE UNIQUE INDEX u_id ON u (id)")
        .unwrap();

    let insert = parse_insert("INSERT INTO u (id, label) VALUES (7, 'g')");
    ensure_insert_device_generation(&e, &insert);
    let snapshot = next_commit_snapshot(&e);
    let delta = e
        .prepare_insert(&insert, snapshot, None, InsertPrepareValidation::Full)
        .unwrap();
    assert!(
        delta.write_set.rows.is_empty(),
        "INSERT must contribute NO row keys to the conflict write-set"
    );
    assert_eq!(
        delta.write_set.unique_slots.len(),
        1,
        "INSERT must record the unique-index slot it occupies"
    );
    assert_eq!(delta.write_set.unique_slots[0].column, "id");
}

#[test]
fn update_write_set_remains_exact_after_host_apply_retirement() {
    // Stage 2 invariant (b), UPDATE: the write-set's row keys equal exactly the row keys whose
    // chain apply rewrote (old tombstoned + new created at the same key).
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (id, label) VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    .unwrap();

    let snapshot = next_commit_snapshot(&e);
    let commit_seq = snapshot.commit_seq;
    // Matches id >= 2 -> ids 2 and 3 rewritten; id=1 untouched.
    let delta = e
        .prepare_update(
            &parse_update("UPDATE t SET label = 'z' WHERE id >= 2"),
            snapshot,
        )
        .unwrap();
    let declared: BTreeSet<String> = delta
        .write_set
        .rows
        .iter()
        .map(|r| r.row_key.clone())
        .collect();
    assert_eq!(
        declared.len(),
        2,
        "exactly two rows in the update write-set"
    );

    e.apply_delta(delta, commit_seq, None).unwrap();
    let touched = keys_touched_at(&e, commit_seq);
    assert!(
        touched.is_empty(),
        "host tuple apply is retired: {touched:?}"
    );
    assert_eq!(declared.len(), 2);
    assert!(e.table_device_authoritative("t"));
}

#[test]
fn delete_write_set_remains_exact_after_host_apply_retirement() {
    // Stage 2 invariant (b), DELETE: the write-set's row keys equal exactly the row keys whose
    // version apply tombstoned.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (id, label) VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    .unwrap();

    let snapshot = next_commit_snapshot(&e);
    let commit_seq = snapshot.commit_seq;
    let delta = e
        .prepare_delete(&parse_delete("DELETE FROM t WHERE id >= 2"), snapshot)
        .unwrap();
    let declared: BTreeSet<String> = delta
        .write_set
        .rows
        .iter()
        .map(|r| r.row_key.clone())
        .collect();
    assert_eq!(
        declared.len(),
        2,
        "exactly two rows in the delete write-set"
    );

    e.apply_delta(delta, commit_seq, None).unwrap();
    let touched = keys_touched_at(&e, commit_seq);
    assert!(
        touched.is_empty(),
        "host tuple apply is retired: {touched:?}"
    );
    assert_eq!(declared.len(), 2);
    assert!(e.table_device_authoritative("t"));
}

#[test]
fn write_set_records_unique_index_slots_for_unique_insert() {
    // Stage 2 invariant (b), unique slots: a unique-column insert records exactly the
    // `(table, column, value)` slots it claims — the Stage 4 first-committer-wins conflict
    // points — and nothing for non-unique columns.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE u (id INT UNIQUE, label TEXT)")
        .unwrap();

    let insert = parse_insert("INSERT INTO u (id, label) VALUES (7, 'a'), (8, 'b')");
    ensure_insert_device_generation(&e, &insert);
    let snapshot = next_commit_snapshot(&e);
    let delta = e
        .prepare_insert(&insert, snapshot, None, InsertPrepareValidation::Full)
        .unwrap();

    let slots: BTreeSet<(String, String, String)> = delta
        .write_set
        .unique_slots
        .iter()
        .map(|s| (s.table.clone(), s.column.clone(), s.value.clone()))
        .collect();
    let expected: BTreeSet<(String, String, String)> = [
        (
            "u".to_string(),
            "id".to_string(),
            relational_index_value(&SqlValue::Int4(7)),
        ),
        (
            "u".to_string(),
            "id".to_string(),
            relational_index_value(&SqlValue::Int4(8)),
        ),
    ]
    .into_iter()
    .collect();
    assert_eq!(slots, expected, "unique-slot write-set incorrect");
    // The non-unique `label` column contributes NO unique slots.
    assert!(
        delta
            .write_set
            .unique_slots
            .iter()
            .all(|s| s.column == "id"),
        "spurious unique slot for a non-unique column"
    );
}

#[test]
fn serialized_dml_records_its_write_set_into_the_si_ledger() {
    // C2 (write-path assessment): the recent-commits ledger used to be populated ONLY by the
    // concurrent commit path — a SERIALIZED UPDATE was invisible to a concurrent committer's
    // first-committer-wins check, so a concurrent transaction prepared against an older snapshot
    // could silently overwrite it (a lost update). Every applied serialized DML now records its
    // prepare-computed write-set at its commit seq, exactly like the concurrent path.
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (id, v) VALUES (1, 10)")
        .unwrap();

    // A concurrent transaction pins + REGISTERS its read snapshot BEFORE the serialized UPDATE
    // commits (the RAII guard, exactly as commit_dml_concurrent does — registration is what
    // keeps ledger entries above the snapshot alive), and prepares a write to the SAME row
    // (prepare is pure — nothing is installed yet).
    let stale_snapshot = e.visible_up_to();
    let _snapshot_guard = e.register_active_snapshot(stale_snapshot);
    let stale_delta = e
        .prepare_update(
            &parse_update("UPDATE t SET v = 99 WHERE id = 1"),
            e.dml_read_snapshot(stale_snapshot),
        )
        .unwrap();

    // The conflicting write commits through the SERIALIZED path (execute_text routing).
    e.execute_text(3, "UPDATE t SET v = 50 WHERE id = 1")
        .unwrap();
    let serialized_commit_seq = e.visible_up_to();
    assert!(serialized_commit_seq > stale_snapshot);

    let commit = e.commit_state();
    assert!(
        commit
            .ledger
            .conflicts(&stale_delta.write_set, stale_snapshot),
        "first-committer-wins: the concurrent txn's stale write to the same row must now \
         conflict against the serialized UPDATE's recorded write-set"
    );
    assert!(
        !commit
            .ledger
            .conflicts(&stale_delta.write_set, serialized_commit_seq),
        "a snapshot taken AT the serialized commit already saw it — no conflict"
    );
}

#[test]
fn serialized_unique_insert_records_its_unique_slot_into_the_si_ledger() {
    // C2, unique-slot dimension: a serialized INSERT claiming a unique-index slot must be
    // visible to a concurrent committer preparing a DIFFERENT row with the SAME unique value
    // against an older snapshot (their row keys differ; only the unique slot collides).
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    e.execute_text(2, "CREATE UNIQUE INDEX t_id ON t (id)")
        .unwrap();

    let stale_insert = parse_insert("INSERT INTO t (id, v) VALUES (7, 1)");
    ensure_insert_device_generation(&e, &stale_insert);
    let stale_snapshot = e.visible_up_to();
    let _snapshot_guard = e.register_active_snapshot(stale_snapshot);
    let stale_delta = e
        .prepare_insert(
            &stale_insert,
            e.dml_read_snapshot(stale_snapshot),
            None,
            InsertPrepareValidation::Full,
        )
        .unwrap();

    // A serialized INSERT claims unique slot id=7 after the concurrent txn's snapshot.
    e.execute_text(3, "INSERT INTO t (id, v) VALUES (7, 2)")
        .unwrap();

    let commit = e.commit_state();
    assert!(
        commit
            .ledger
            .conflicts(&stale_delta.write_set, stale_snapshot),
        "the serialized INSERT's unique slot must collide with the stale duplicate-value insert"
    );
}
