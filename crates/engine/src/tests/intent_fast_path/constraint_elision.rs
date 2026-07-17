use super::gpu_ids_of_t;
use crate::Engine;
use gpu_db_sql::{parse_command, Command, SqlValue};
use std::sync::atomic::{AtomicU64, Ordering};

/// CPU-ENGINE RETIREMENT (ADR-006, structural): a CHECK-constrained table now ELIDES — CHECK
/// validation is ROW-LOCAL (`validate_check_constraints_for_rows` evaluates the NEW values only,
/// never the tuple store), so the stale-host-store invariant is unaffected. Pins the full lifecycle:
/// the table elides; a violating INSERT on the ELIDED table errors (and changes nothing); valid DML
/// lands on-device; a violating UPDATE (new image from the device materialize) errors; and ALTER ADD
/// CHECK sees ELIDED-ERA device rows via the rehydrate-first DDL validator (rejects a new constraint
/// an elided-era row violates). GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_check_constrained_table_elides() {
    let mut engine = Engine::new_local_cpu_oracle();
    engine
        .execute_text(
            1,
            "CREATE TABLE t (id INT PRIMARY KEY, v INT, CONSTRAINT v_pos CHECK (v > 0))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    engine
        .execute_dml_concurrent(2, "INSERT INTO t VALUES (1, 10)")
        .unwrap();
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    engine
        .execute_dml_concurrent(3, "INSERT INTO t VALUES (2, 20)")
        .unwrap();
    assert!(
        engine.table_install_elided("t"),
        "a CHECK-constrained (FK-free) table must now ELIDE (CHECK is row-local)"
    );

    // A VIOLATING insert on the ELIDED table errors (row-local validation) and changes nothing.
    let err = engine
        .execute_dml_concurrent(4, "INSERT INTO t VALUES (3, -5)")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("check constraint"),
        "violation must name the CHECK, got: {err}"
    );
    assert!(
        engine.table_install_elided("t"),
        "a rejected insert must not de-elide"
    );
    assert_eq!(
        gpu_ids_of_t(&engine),
        vec![1, 2],
        "the violating row was NOT inserted"
    );

    // A VALID insert lands on-device, still elided.
    engine
        .execute_dml_concurrent(5, "INSERT INTO t VALUES (3, 30)")
        .unwrap();
    assert!(engine.table_install_elided("t"));
    assert_eq!(gpu_ids_of_t(&engine), vec![1, 2, 3]);

    // A VIOLATING UPDATE errors (the candidate new image is built from the DEVICE-materialized row).
    let err = engine
        .execute_dml_concurrent(6, "UPDATE t SET v = -1 WHERE id = 2")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("check constraint"),
        "update violation must name the CHECK, got: {err}"
    );
    assert!(
        engine.table_install_elided("t"),
        "a rejected update must not de-elide"
    );

    // A VALID update works on-device.
    engine
        .execute_dml_concurrent(7, "UPDATE t SET v = 25 WHERE id = 2")
        .unwrap();
    assert_eq!(gpu_ids_of_t(&engine), vec![1, 2, 3]);

    // ALTER ADD CHECK must see ELIDED-ERA rows (ids 2,3 live only on the device): a constraint that
    // an elided-era row violates (v <= 21 fails for v=25 and v=30) must be REJECTED — the DDL
    // row-validator rehydrates first (elision-safe by construction). A permissive one is accepted.
    let err = engine
        .execute_text(8, "ALTER TABLE t ADD CONSTRAINT v_small CHECK (v < 21)")
        .unwrap_err()
        .to_string();
    assert!(
        err.to_lowercase().contains("check") || err.to_lowercase().contains("violat"),
        "ADD CHECK must validate ELIDED-ERA device rows (v=25/30 violate v<21), got: {err}"
    );
    engine
        .execute_text(9, "ALTER TABLE t ADD CONSTRAINT v_cap CHECK (v < 1000)")
        .unwrap();
}

/// CPU-ENGINE RETIREMENT (ADR-006, structural — FK elision, parent side): a table REFERENCED by
/// foreign keys now ELIDES when its referenced column is an i32-section single-column PK — the FK
/// validators' parent-exists / surviving-provider lookups run through `visible_row_with_value`, whose
/// device arm probes the elided parent's PK index + materializes visibility ON THE DEVICE. Pins the
/// lifecycle: the referenced parent elides; a child INSERT referencing an ELIDED-ERA parent key (a
/// device-only row) SUCCEEDS (the device probe must SEE it — the load-bearing pin); a child INSERT
/// referencing a MISSING key is REJECTED and nothing durable/wedging results (later statements work);
/// a parent DELETE of a referenced key is REJECTED; a parent DELETE of an unreferenced key succeeds.
/// GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_fk_referenced_parent_elides() {
    let mut engine = Engine::new_local_cpu_oracle();
    engine
        .execute_text(1, "CREATE TABLE customers (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT)",
        )
        .unwrap();
    engine
        .execute_text(
            20,
            "ALTER TABLE ONLY orders ADD CONSTRAINT orders_fk FOREIGN KEY (customer_id) \
             REFERENCES customers(id)",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    engine
        .execute_dml_concurrent(3, "INSERT INTO customers VALUES (1, 'ada')")
        .unwrap();
    let snap = engine.populate_relational_residency_snapshot("customers");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    engine
        .execute_dml_concurrent(4, "INSERT INTO customers VALUES (2, 'bob')")
        .unwrap();
    assert!(
        engine.table_install_elided("customers"),
        "an FK-REFERENCED parent (i32 PK) must now ELIDE"
    );
    // ELIDED-ERA parent key: device-only (the stale host store cannot see it).
    engine
        .execute_dml_concurrent(5, "INSERT INTO customers VALUES (3, 'eve')")
        .unwrap();
    assert!(
        engine.table_install_elided("customers"),
        "still elided after the device-only insert"
    );

    // THE LOAD-BEARING PIN: a child INSERT referencing the ELIDED-ERA key (3) must SUCCEED — the
    // parent-exists probe must SEE the device-only row (a stale/vacuous answer would reject it).
    engine
        .execute_text(6, "INSERT INTO orders VALUES (100, 3)")
        .unwrap();
    assert!(
        engine.table_install_elided("customers"),
        "the parent probe must run ON-DEVICE (the parent stays elided; a de-elide means the \
         validator fell back to rehydration)"
    );
    // A MISSING parent key is rejected — and nothing durable/wedging results.
    let err = engine
        .execute_text(7, "INSERT INTO orders VALUES (101, 999)")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("foreign key"),
        "missing parent must reject, got: {err}"
    );
    engine
        .execute_text(8, "INSERT INTO orders VALUES (102, 1)")
        .unwrap(); // later statements still work (no wedge)

    // Parent DELETE of a REFERENCED key (3, referenced by order 100) is REJECTED.
    let err = engine
        .execute_text(9, "DELETE FROM customers WHERE id = 3")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("foreign key"),
        "deleting a referenced parent must reject, got: {err}"
    );
    // Parent DELETE of an UNREFERENCED key (2) succeeds.
    engine
        .execute_text(10, "DELETE FROM customers WHERE id = 2")
        .unwrap();
    let Command::Select(s) = parse_command("SELECT id FROM customers").unwrap() else {
        unreachable!()
    };
    let mut ids: Vec<i32> = engine
        .execute_relational_select(&s)
        .unwrap()
        .rows
        .iter()
        .map(|r| match r.first() {
            Some(SqlValue::Int4(n)) => *n,
            other => panic!("unexpected id: {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 3],
        "customer 2 deleted; 1 and 3 (referenced) remain"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, structural — FK elision, CHILD side): a table WITH outbound
/// foreign keys (non-self-referencing, i32 fk columns) now ELIDES. Its own INSERTs validate the
/// parent on-device (item 3); the INBOUND child-reference check on a parent DELETE (`does any child
/// row carry fk = departed key?`) runs ON THE DEVICE via the new Eq scan-locate fallback in
/// `device_visible_row_with_value` — the hash-index probe declines on the DUPLICATE-heavy fk column
/// (two orders share customer 1 to force it), and the child must STAY ELIDED. The load-bearing pin:
/// the referencing child rows are ELIDED-ERA (device-only) — a stale/vacuous answer would wrongly
/// ALLOW the parent delete (an orphan). GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_fk_child_table_elides() {
    let mut engine = Engine::new_local_cpu_oracle();
    engine
        .execute_text(1, "CREATE TABLE customers (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT)",
        )
        .unwrap();
    engine
        .execute_text(
            3,
            "ALTER TABLE ONLY orders ADD CONSTRAINT orders_fk FOREIGN KEY (customer_id) \
             REFERENCES customers(id)",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    engine
        .execute_text(4, "INSERT INTO customers VALUES (1, 'ada')")
        .unwrap();
    engine
        .execute_text(5, "INSERT INTO customers VALUES (2, 'bob')")
        .unwrap();
    engine
        .execute_dml_concurrent(6, "INSERT INTO orders VALUES (100, 1)")
        .unwrap();
    let snap = engine.populate_relational_residency_snapshot("orders");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    engine
        .execute_dml_concurrent(7, "INSERT INTO orders VALUES (101, 1)")
        .unwrap();
    assert!(
        engine.table_install_elided("orders"),
        "an FK-CHILD table (outbound i32 fk, non-self-ref) must now ELIDE"
    );
    // ELIDED-ERA child rows referencing customer 2 (device-only; two rows -> the fk column is
    // duplicate-heavy so the hash-index probe declines -> the Eq scan-locate serves the check).
    engine
        .execute_dml_concurrent(8, "INSERT INTO orders VALUES (102, 2)")
        .unwrap();
    engine
        .execute_dml_concurrent(9, "INSERT INTO orders VALUES (103, 2)")
        .unwrap();
    assert!(
        engine.table_install_elided("orders"),
        "still elided (device-only referencing rows)"
    );

    // THE LOAD-BEARING PIN: deleting customer 2 — referenced ONLY by ELIDED-ERA device rows — must
    // be REJECTED, and the child must STAY ELIDED (the check ran on-device; a de-elide means the
    // ladder fell back to rehydration; a vacuous pass would orphan orders 102/103).
    let err = engine
        .execute_text(10, "DELETE FROM customers WHERE id = 2")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("foreign key"),
        "deleting a parent referenced by elided-era child rows must reject, got: {err}"
    );
    assert!(
        engine.table_install_elided("orders"),
        "the child-reference check must run ON-DEVICE (the child stays elided)"
    );
    // Child INSERT with a missing parent still rejects while elided; a valid one lands.
    let err = engine
        .execute_text(11, "INSERT INTO orders VALUES (104, 999)")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("foreign key"),
        "missing parent must reject, got: {err}"
    );
    engine
        .execute_text(12, "INSERT INTO orders VALUES (105, 1)")
        .unwrap();
    assert!(
        engine.table_install_elided("orders"),
        "valid child insert keeps the child elided"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, audit follow-up — DATE fk column): the Eq scan-locate fallback
/// must lower a DATE fk needle through the CANONICAL date string (`format_date`), NOT a raw-days
/// Int4Literal — the lowering hard-rejects that shape (`date = 5`, PG semantics), and the decline
/// would REHYDRATE the elided Date-fk child on EVERY parent delete (correct answers, systematic
/// thrash). Duplicate fk dates force the hash-index decline -> the scan arm; the child must STAY
/// ELIDED through a rejected referenced-parent delete AND an allowed unreferenced one. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_fk_child_date_fk_stays_elided() {
    let mut engine = Engine::new_local_cpu_oracle();
    engine
        .execute_text(1, "CREATE TABLE days (d DATE PRIMARY KEY, note TEXT)")
        .unwrap();
    engine
        .execute_text(2, "CREATE TABLE events (id INT PRIMARY KEY, on_day DATE)")
        .unwrap();
    engine
        .execute_text(
            3,
            "ALTER TABLE ONLY events ADD CONSTRAINT events_fk FOREIGN KEY (on_day) \
             REFERENCES days(d)",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    engine
        .execute_text(4, "INSERT INTO days VALUES ('2024-03-01', 'kickoff')")
        .unwrap();
    engine
        .execute_text(5, "INSERT INTO days VALUES ('2024-03-02', 'review')")
        .unwrap();
    engine
        .execute_dml_concurrent(6, "INSERT INTO events VALUES (1, '2024-03-01')")
        .unwrap();
    let snap = engine.populate_relational_residency_snapshot("events");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    engine
        .execute_dml_concurrent(7, "INSERT INTO events VALUES (2, '2024-03-01')")
        .unwrap();
    assert!(
        engine.table_install_elided("events"),
        "an FK-CHILD table (outbound DATE fk, non-self-ref) must now ELIDE"
    );
    // ELIDED-ERA rows referencing '2024-03-02' — duplicated so the hash-index probe declines and
    // the Eq scan-locate serves the inbound check with a DATE needle.
    engine
        .execute_dml_concurrent(8, "INSERT INTO events VALUES (3, '2024-03-02')")
        .unwrap();
    engine
        .execute_dml_concurrent(9, "INSERT INTO events VALUES (4, '2024-03-02')")
        .unwrap();
    assert!(
        engine.table_install_elided("events"),
        "still elided (device-only referencing rows)"
    );

    // Deleting the referenced date must REJECT with the child STILL elided (a de-elide means the
    // Date needle declined the device scan and fell back to rehydration — the thrash this pins).
    let err = engine
        .execute_text(10, "DELETE FROM days WHERE d = '2024-03-02'")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("foreign key"),
        "deleting a date referenced by elided-era child rows must reject, got: {err}"
    );
    assert!(
        engine.table_install_elided("events"),
        "the DATE child-reference check must run ON-DEVICE (the child stays elided)"
    );
    // An UNREFERENCED-date parent delete probes the same scan arm (finds rows with the fk value
    // absent) and must SUCCEED — still without de-eliding the child.
    engine
        .execute_text(11, "INSERT INTO days VALUES ('2024-03-03', 'spare')")
        .unwrap();
    engine
        .execute_text(12, "DELETE FROM days WHERE d = '2024-03-03'")
        .unwrap();
    assert!(
        engine.table_install_elided("events"),
        "an allowed parent delete keeps the child elided (no-match scan answered on-device)"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, FK child side — NON-i32 fk columns): UUID / BIGINT / TEXT fk
/// children now ELIDE. Duplicate-heavy child columns are not unique indexes, so the inbound
/// child-reference check rides `device_eq_scan_literal` → the Eq scan-locate — distinct paths (uuid
/// = b128 byte compare via the canonical `format_uuid` round-trip; int8 = CompareScalarI64; text
/// = byte-exact blob compare). Per pair: elided-era referencing rows (duplicated fk values),
/// a rejected referenced-parent delete with the child STAYING elided, and an allowed
/// unreferenced delete. R3-002 also permits the foldable wide-key parents to elide; their
/// parent-exists probes use the fingerprint index plus exact typed device recheck.
/// GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_fk_child_noni32_fk_columns_stay_elided() {
    const TXN0: u64 = 1000;
    let mut engine = Engine::new_local_cpu_oracle();
    for (parent_ddl, child_ddl, fk_ddl) in [
        (
            "CREATE TABLE vendors (vid UUID PRIMARY KEY, vname TEXT)",
            "CREATE TABLE parts (id INT PRIMARY KEY, vendor_id UUID)",
            "ALTER TABLE ONLY parts ADD CONSTRAINT parts_fk FOREIGN KEY (vendor_id) \
             REFERENCES vendors(vid)",
        ),
        (
            "CREATE TABLE accounts (aid BIGINT PRIMARY KEY, aname TEXT)",
            "CREATE TABLE ledgers (id INT PRIMARY KEY, account_id BIGINT)",
            "ALTER TABLE ONLY ledgers ADD CONSTRAINT ledgers_fk FOREIGN KEY (account_id) \
             REFERENCES accounts(aid)",
        ),
        (
            "CREATE TABLE cats (code TEXT PRIMARY KEY, cname TEXT)",
            "CREATE TABLE items (id INT PRIMARY KEY, cat_code TEXT)",
            "ALTER TABLE ONLY items ADD CONSTRAINT items_fk FOREIGN KEY (cat_code) \
             REFERENCES cats(code)",
        ),
        (
            "CREATE TABLE prices (amt NUMERIC(10,2) PRIMARY KEY, pname TEXT)",
            "CREATE TABLE quotes (id INT PRIMARY KEY, quote_amt NUMERIC(10,2))",
            "ALTER TABLE ONLY quotes ADD CONSTRAINT quotes_fk FOREIGN KEY (quote_amt) \
             REFERENCES prices(amt)",
        ),
        (
            "CREATE TABLE slots (at TIMESTAMP PRIMARY KEY, sname TEXT)",
            "CREATE TABLE bookings (id INT PRIMARY KEY, slot_at TIMESTAMP)",
            "ALTER TABLE ONLY bookings ADD CONSTRAINT bookings_fk FOREIGN KEY (slot_at) \
             REFERENCES slots(at)",
        ),
    ] {
        // One monotone facade txn-id stream keeps this older cross-type fixture independent of
        // commit-sequence allocation details.
        engine.execute_text(TXN0 + 1, parent_ddl).unwrap();
        engine.execute_text(TXN0 + 2, child_ddl).unwrap();
        engine.execute_text(TXN0 + 3, fk_ddl).unwrap();
    }
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    // Per pair: (child, parent, pk_col, [kept, departing, missing] key literals).
    let sections: [(&str, &str, &str, [&str; 3]); 5] = [
        (
            "parts",
            "vendors",
            "vid",
            [
                "'00000000-0000-0000-0000-000000000001'",
                "'00000000-0000-0000-0000-000000000002'",
                "'00000000-0000-0000-0000-000000000099'",
            ],
        ),
        // > i32::MAX so a truncating i32 needle could never accidentally match.
        (
            "ledgers",
            "accounts",
            "aid",
            ["4294967300", "4294967301", "9999999999"],
        ),
        ("items", "cats", "code", ["'alpha'", "'beta'", "'zzz'"]),
        ("quotes", "prices", "amt", ["10.25", "20.50", "99.99"]),
        (
            "bookings",
            "slots",
            "at",
            [
                "'2024-06-01 09:00:00'",
                "'2024-06-01 10:00:00'",
                "'2024-06-01 23:00:00'",
            ],
        ),
    ];
    let txn_ids = AtomicU64::new(TXN0 + 10);
    let mut gpu_checked = false;
    for (child, parent, pk_col, [kept, departing, missing]) in sections {
        macro_rules! sql {
            ($s:expr) => {
                engine.execute_text(txn_ids.fetch_add(1, Ordering::Relaxed), &$s)
            };
        }
        sql!(format!("INSERT INTO {parent} VALUES ({kept}, 'keep')")).unwrap();
        sql!(format!("INSERT INTO {parent} VALUES ({departing}, 'ref')")).unwrap();
        assert!(
            engine.table_install_elided(parent),
            "{parent}: a foldable single-wide parent key must elide"
        );
        let parent_validate_before = engine.dml_device_validate_hits();
        sql!(format!("INSERT INTO {child} VALUES (1, {kept})")).unwrap();
        assert!(
            engine.dml_device_validate_hits() > parent_validate_before,
            "{parent}: child provider validation must use the exact device fingerprint recheck"
        );
        if !gpu_checked {
            let snap = engine.populate_relational_residency_snapshot(child);
            if snap
                .map(|s| s.device_memory_proof.is_none())
                .unwrap_or(true)
            {
                return; // no usable GPU
            }
            gpu_checked = true;
        }
        sql!(format!("INSERT INTO {child} VALUES (2, {kept})")).unwrap();
        assert!(
            engine.table_install_elided(child),
            "{child}: a non-i32 fk child (outbound fk, non-self-ref) must now ELIDE"
        );
        // ELIDED-ERA rows referencing the departing key — duplicated, and non-i32 columns have
        // no device index at all, so the Eq scan-locate serves the check with a
        // uuid/int8/text needle.
        sql!(format!("INSERT INTO {child} VALUES (3, {departing})")).unwrap();
        sql!(format!("INSERT INTO {child} VALUES (4, {departing})")).unwrap();
        assert!(
            engine.table_install_elided(child),
            "{child}: still elided (device-only referencing rows)"
        );
        let err = sql!(format!("DELETE FROM {parent} WHERE {pk_col} = {departing}"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("foreign key"),
            "{parent}: deleting a key referenced by elided-era child rows must reject, got: {err}"
        );
        assert!(
            engine.table_install_elided(child),
            "{child}: the child-reference check must run ON-DEVICE (the child stays elided)"
        );
        // A child INSERT with a missing parent still rejects while elided; a valid one lands.
        let err = sql!(format!("INSERT INTO {child} VALUES (9, {missing})"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("foreign key"),
            "{child}: missing parent must reject, got: {err}"
        );
        sql!(format!("INSERT INTO {child} VALUES (10, {kept})")).unwrap();
        assert!(
            engine.table_install_elided(child),
            "{child}: rejected + valid inserts keep the child elided"
        );
        // An UNREFERENCED parent key deletes fine — the no-match scan answers on-device and the
        // child STAYS elided (a de-elide here is the rehydrate-per-parent-delete thrash).
        sql!(format!("INSERT INTO {parent} VALUES ({missing}, 'spare')")).unwrap();
        sql!(format!("DELETE FROM {parent} WHERE {pk_col} = {missing}")).unwrap();
        assert!(
            engine.table_install_elided(child),
            "{child}: an allowed parent delete keeps the child elided"
        );
    }

    // BOOL fk addendum (audit LOW-1): the 1-bit bitmap-vs-validity lowering on the inbound scan.
    // Bool can't follow the loop (only two possible keys — no "missing parent" literal exists):
    // referenced `false` must reject with the child STAYING elided; unreferenced `true` deletes.
    let txn = || txn_ids.fetch_add(1, Ordering::Relaxed);
    engine
        .execute_text(
            txn(),
            "CREATE TABLE toggles (f BOOL PRIMARY KEY, tname TEXT)",
        )
        .unwrap();
    engine
        .execute_text(txn(), "CREATE TABLE states (id INT PRIMARY KEY, flag BOOL)")
        .unwrap();
    engine
        .execute_text(
            txn(),
            "ALTER TABLE ONLY states ADD CONSTRAINT states_fk FOREIGN KEY (flag) \
             REFERENCES toggles(f)",
        )
        .unwrap();
    engine
        .execute_text(txn(), "INSERT INTO toggles VALUES (true, 'on')")
        .unwrap();
    engine
        .execute_text(txn(), "INSERT INTO toggles VALUES (false, 'off')")
        .unwrap();
    assert!(
        engine.table_install_elided("toggles"),
        "a BOOL-PK parent must elide through the bitmap fingerprint index"
    );
    let bool_parent_validate_before = engine.dml_device_validate_hits();
    engine
        .execute_text(txn(), "INSERT INTO states VALUES (1, false)")
        .unwrap();
    assert!(
        engine.dml_device_validate_hits() > bool_parent_validate_before,
        "the BOOL parent provider probe must exact-recheck on device"
    );
    engine
        .execute_text(txn(), "INSERT INTO states VALUES (2, false)")
        .unwrap();
    assert!(
        engine.table_install_elided("states"),
        "states: a bool fk child must now ELIDE"
    );
    let err = engine
        .execute_text(txn(), "DELETE FROM toggles WHERE f = false")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("foreign key"),
        "referenced bool parent must reject, got: {err}"
    );
    assert!(
        engine.table_install_elided("states"),
        "states: the bool child-reference check must run ON-DEVICE (the child stays elided)"
    );
    engine
        .execute_text(txn(), "DELETE FROM toggles WHERE f = true")
        .unwrap();
    assert!(
        engine.table_install_elided("states"),
        "states: an allowed bool parent delete keeps the child elided"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, MIXED-WIDTH groups): a DELETE/UPDATE whose WHERE mixes an INT8
/// scalar leaf with i32-servable leaves (text/bool/int4) resolves ON THE DEVICE — the int8 leaf
/// lowers via the width-safe `LoadColumnI64`+`CompareScalarI64` arm inside an I32 program
/// (`mixed_width_i32_elem`), the shape that used to hard-error ("mixed int8/text") and de-elide.
/// This is EXACTLY the decline recipe the CHECK-bypass repro used — closing it removes that
/// rehydrate trigger. The second DELETE runs over VERSIONED shards (the earlier UPDATE tombstoned
/// + appended), pinning the visibility-branch `mixed_width_i32_elem` fallback. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_mixed_width_dml_resolves_on_device() {
    let mut engine = Engine::new_local_cpu_oracle();
    engine
        .execute_text(
            1,
            "CREATE TABLE m (id INT PRIMARY KEY, big BIGINT, name TEXT, flag BOOL)",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO m VALUES (1, 4294967300, 'keep', true)")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("m");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    // big values straddle i32::MAX so a 4-byte mis-read could never fake the right answer.
    for (id, big, name, flag) in [
        (2i64, 4_294_967_301i64, "drop", "false"),
        (3, 100, "drop", "false"),
        (4, 4_294_967_302, "drop", "true"),
        (5, 4_294_967_303, "hold", "false"),
    ] {
        engine
            .execute_dml_concurrent(
                txn,
                &format!("INSERT INTO m VALUES ({id}, {big}, '{name}', {flag})"),
            )
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_install_elided("m"),
        "the table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM m").unwrap() else {
            unreachable!()
        };
        let mut out: Vec<i64> = engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .map(|r| match r.first() {
                Some(SqlValue::Int4(n)) => *n as i64,
                other => panic!("unexpected id: {other:?}"),
            })
            .collect();
        out.sort_unstable();
        out
    };
    assert_eq!(ids(&engine), vec![1, 2, 3, 4, 5]);

    // MIXED int8+text DELETE (`big > i32::MAX AND name = 'drop'` -> ids 2, 4; id 3's big=100
    // fails the range, id 5's name fails the eq) ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            txn,
            "DELETE FROM m WHERE big > 2147483647 AND name = 'drop'",
        )
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a mixed int8+text DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_install_elided("m"),
        "a mixed int8+text DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![1, 3, 5],
        "exactly the big>i32::MAX 'drop' rows (2,4) deleted"
    );

    // MIXED bool+int8 UPDATE (`flag = false AND big > i32::MAX` -> id 5) ON THE DEVICE.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            txn,
            "UPDATE m SET name = 'held' WHERE flag = false AND big > 2147483647",
        )
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a mixed bool+int8 UPDATE must RESOLVE on the device"
    );
    assert!(
        engine.table_install_elided("m"),
        "a mixed bool+int8 UPDATE must NOT de-elide"
    );

    // VERSIONED-shard mixed READ (the LOAD-BEARING pin for the visibility-branch
    // `mixed_width_i32_elem` fallback — audit note adopted): the UPDATE above tombstoned the old
    // id-5 row and appended its twin, and a VERSIONED shard FORCES the mask VM so the WHERE can
    // compose with the on-device visibility conjuncts (SV3b). Without the fallback this mixed
    // WHERE hard-errors there -> CPU-pinned -> rehydrate/DE-ELIDE, so stays-elided + row-exact
    // (exactly ONE id-5 version, the live 'held' twin, not the tombstoned original) prove the
    // versioned path served it.
    let Command::Select(vsel) =
        parse_command("SELECT id FROM m WHERE flag = false AND big > 2147483647").unwrap()
    else {
        unreachable!()
    };
    let vrows = engine.execute_relational_select(&vsel).unwrap().rows;
    assert_eq!(
        vrows,
        vec![vec![SqlValue::Int4(5)]],
        "versioned-shard mixed read: exactly the LIVE id-5 twin (no tombstoned duplicate)"
    );
    assert!(
        engine.table_install_elided("m"),
        "the versioned-shard mixed read must NOT de-elide (the visibility-branch fallback)"
    );

    // MIXED int4+int8 DELETE over the now-VERSIONED shards (the UPDATE tombstoned + appended a
    // twin): the WHERE composes with the on-device visibility conjuncts at I32 (the
    // `mixed_width_i32_elem` fallback in the visibility branch). id 5 (renamed 'held') matches.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "DELETE FROM m WHERE id > 4 AND big = 4294967303")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a mixed int4+int8 DELETE over versioned shards must RESOLVE on the device"
    );
    assert!(
        engine.table_install_elided("m"),
        "the versioned-shard DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![1, 3],
        "id 5 deleted; 1 ('keep') and 3 (big=100) survive"
    );

    // BOOL INEQUALITY DML (ADR-006: `flag < true` ⇔ `flag = false` — the bool leaves constant-
    // fold PG's false<true ordering; the DML builder now lowers bool comparisons, not just Eq).
    // Survivors: 1 (flag true), 3 (flag false). The DELETE removes exactly id 3, on-device.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn + 1, "DELETE FROM m WHERE flag < true")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a bool-inequality DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_install_elided("m"),
        "a bool-inequality DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![1],
        "flag < true deleted exactly the flag=false row (3)"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, audit HIGH regression pin): a MID-PREFLIGHT REHYDRATE must not
/// bypass CHECK. On an ELIDED CHECK table, an `execute_text` UPDATE whose WHERE the device resolve
/// DECLINES (a mixed int8+text AND — the decline REHYDRATES/de-elides mid-preflight) used to fall back
/// to a scan on the STALE pre-rehydrate store handle: zero visible rows -> the CHECK validated
/// VACUOUSLY -> the apply then wrote the violating value to the real rows. The preflight now RE-PINS
/// the outer store view (+ raises the read boundary) after the decline — the violating UPDATE must
/// ERROR and change nothing. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_check_elided_preflight_rehydrate_no_bypass() {
    let mut engine = Engine::new_local_cpu_oracle();
    engine
        .execute_text(
            1,
            "CREATE TABLE t (id INT PRIMARY KEY, big BIGINT, name TEXT, v INT, \
             CONSTRAINT v_pos CHECK (v > 0))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_host_install_elision_enabled(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine.set_constrained_elision_enabled(true);
    engine.set_resident_delete_tombstone_enabled(true);
    engine.set_resident_update_tombstone_enabled(true);

    engine
        .execute_dml_concurrent(2, "INSERT INTO t VALUES (1, 5, 'x', 10)")
        .unwrap();
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    engine
        .execute_dml_concurrent(3, "INSERT INTO t VALUES (2, 7, 'y', 20)")
        .unwrap();
    assert!(
        engine.table_install_elided("t"),
        "the CHECK table must elide first"
    );
    // THE LOAD-BEARING ROW: inserted AFTER elision entered, so it is ELIDED-ERA (device-only — the
    // stale pre-rehydrate host handle cannot see it). The bypass requires the WHERE to match THIS row.
    engine
        .execute_dml_concurrent(4, "INSERT INTO t VALUES (3, 9, 'z', 30)")
        .unwrap();
    assert!(
        engine.table_install_elided("t"),
        "still elided after the device-only insert"
    );

    // The BYPASS shape: a RANGE-ONLY mixed int8+text AND — the device predicate rejects the mixed
    // widths AND the value index declines (no Eq leaf) -> the else-SCAN runs. The decline
    // REHYDRATES mid-preflight. The violating UPDATE targets the ELIDED-ERA row and must still be
    // REJECTED by the (re-pinned) scan — the stale handle would see zero matches and pass vacuously.
    let err = engine
        .execute_text(5, "UPDATE t SET v = -1 WHERE big > 8 AND name > 'a'")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("check constraint"),
        "the mid-preflight-rehydrate UPDATE must REJECT the CHECK violation (was: silently \
         committed via the stale-scan vacuous pass), got: {err}"
    );
    // The matched (elided-era) row is UNCHANGED (v=30, not -1).
    let Command::Select(s) = parse_command("SELECT v FROM t WHERE id = 3").unwrap() else {
        unreachable!()
    };
    assert_eq!(
        engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .next()
            .and_then(|r| r.first()),
        Some(&SqlValue::Int4(30)),
        "the violating UPDATE must not have changed the elided-era row"
    );
    // And a VALID update through the same declining shape works (the re-pinned scan finds the row).
    engine
        .execute_text(6, "UPDATE t SET v = 31 WHERE big > 8 AND name > 'a'")
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .next()
            .and_then(|r| r.first()),
        Some(&SqlValue::Int4(31)),
        "a valid update through the declining shape must land (the re-pin sees the elided-era row)"
    );
}
