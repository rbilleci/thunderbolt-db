//! E2.1 — covered-INSERT intent fast path: route eligibility, duplicate-key
//! semantics through the wave-batched device validation, and crash-recovery
//! parity of the FUA WAL log (replay reproduces the committed store).

use super::*;
mod constraint_elision;
mod lane_lifecycle;
mod wide_unique_index;

/// Route preparation is a SHAPE PROOF: a table that is not yet elided
/// (device-authoritative), or that has no unique index, must be refused with a
/// clear error instead of silently taking an unvalidated fast path.
#[test]
fn covered_insert_route_requires_covered_shape() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();

    // Missing table.
    let err = engine.prepare_covered_insert_route("nope").unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");

    // Binary WAL records disabled.
    let err = engine.prepare_covered_insert_route("t").unwrap_err();
    assert!(err.to_string().contains("binary WAL records"), "{err}");

    // Flags on, but the table is not elided (no GPU warm-up ran), so the
    // wave-batched device validation is unavailable.
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    let err = engine.prepare_covered_insert_route("t").unwrap_err();
    assert!(
        err.to_string()
            .contains("wave-batched device PK validation"),
        "{err}"
    );

    // A non-INT4 column refuses the route before eligibility is even probed.
    engine
        .execute_text(2, "CREATE TABLE wide (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    let err = engine.prepare_covered_insert_route("wide").unwrap_err();
    assert!(
        err.to_string().contains("every column must be INT4"),
        "{err}"
    );
}

/// COMPOUND KEYS (TYPE-COVERAGE #14 Track 3): a compound PRIMARY KEY over i32-section columns is
/// DEVICE-NATIVE — the table elides, and the wave-batched device write-locate validates uniqueness
/// on the surrogate FINGERPRINT while the authoritative recheck compares the FULL tuple. This proves
/// the DEVICE path FIRES (the `device_write_locate_hits` counter advances — not a silent host
/// fallback) AND that uniqueness is exact (a repeated tuple is 23505; a tuple differing in ONE key
/// column is a distinct row). Compound keys take the CLASSIC covered path (`execute_dml_concurrent`),
/// not the fused intent lane. Self-guards on a driverless box.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_primary_key_elides_and_validates_uniqueness_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE ct (a INT, b INT, v INT, PRIMARY KEY (a, b))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! insert {
        ($sql:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $sql)
        };
    }
    insert!("INSERT INTO ct VALUES (1000000, 0, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU on this box
    }

    // Warm into elision on the classic covered wave path (distinct tuples: unique `a` per row).
    let mut warmed = false;
    for i in 0..10_000_i32 {
        insert!(&format!(
            "INSERT INTO ct VALUES ({}, {}, 0)",
            2_000_000 + i,
            i
        ))
        .unwrap();
        if engine.table_device_authoritative("ct") {
            warmed = true;
            break;
        }
    }
    assert!(
        warmed,
        "compound-PK table never entered elision on a GPU box"
    );

    // The DEVICE write-locate must actually fire for the compound key (non-vacuity).
    let hits_before = engine.device_write_locate_hits();
    // A brand-new distinct tuple commits.
    insert!("INSERT INTO ct VALUES (5000000, 1, 10)").unwrap();
    // The EXACT tuple again -> 23505 (device fingerprint hit, tuple recheck confirms).
    let dup = insert!("INSERT INTO ct VALUES (5000000, 1, 99)")
        .unwrap_err()
        .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate compound tuple must raise 23505, got: {dup}"
    );
    // Same first key column, DIFFERENT second -> a DISTINCT tuple, must commit (not first-col-only).
    insert!("INSERT INTO ct VALUES (5000000, 2, 11)").unwrap();
    // Same second key column, DIFFERENT first -> also distinct, must commit.
    insert!("INSERT INTO ct VALUES (7000000, 1, 12)").unwrap();

    assert!(
        engine.device_write_locate_hits() > hits_before,
        "the compound-key wave validation must run on the DEVICE (write-locate counter advanced)"
    );
    // Uniqueness never de-elided the table (sustained device path).
    assert!(
        engine.table_device_authoritative("ct"),
        "compound-PK table must stay elided across the validated inserts"
    );

    // CHARTER (device-fold consistency): the compound index REBUILD folds the fingerprint ON THE
    // DEVICE (submit_compound_fold_fingerprints), while the probe needle is HOST-folded
    // (compound_key_fingerprint) -- they MUST byte-match. The (1000000, 0) tuple was inserted before
    // elision and has survived every geometric device-fold rebuild during warm-up, so it sits in the
    // index at a DEVICE-folded slot; a duplicate of it (host-folded needle) raising 23505 proves the
    // two folds agree (a constant/order divergence would miss -> no 23505 -> this assert fails).
    let dup_rebuilt = insert!("INSERT INTO ct VALUES (1000000, 0, 55)")
        .unwrap_err()
        .to_string();
    assert!(
        dup_rebuilt.contains("duplicate key value violates unique index"),
        "device-folded index fingerprint must match the host needle fold, got: {dup_rebuilt}"
    );

    // Read-your-writes over the elided compound table: exactly the committed distinct tuples exist.
    let Command::Select(count) =
        parse_command("SELECT COUNT(*) FROM ct WHERE a = 5000000").unwrap()
    else {
        unreachable!()
    };
    let rows = engine.execute_relational_select(&count).unwrap().rows;
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int8(2)]],
        "exactly two visible rows share a=5000000 (b=1 and b=2)"
    );
}

/// COMPOUND KEYS (wider types, Stage 2a): a compound PRIMARY KEY over i64 (Int8/Timestamp) columns — and
/// a MIXED int4+int8 key — elides and enforces uniqueness ON THE DEVICE. Each key column folds its i32
/// WORD decomposition into the surrogate fingerprint (i64 -> [low32, high32], matching the section's LE
/// layout); the device fold kernel reads `widths[k]` words per column. Proves: elision, tuple uniqueness
/// (dup -> 23505; distinct OK), the DEVICE fold matches the host needle even across the HIGH word (a
/// value > 2^32 survives geometric rebuilds and its duplicate is caught), and DELETE by the i64 key stays
/// COMPOUND KEYS (wider types, Stage 2c): a compound PRIMARY KEY over a b128 (UUID) column — mixed with
/// int4 — elides + enforces uniqueness ON THE DEVICE (each b128 key column folds 4 i32 words = the LE
/// section bytes) and DELETE by the key stays device-native (materialize now reassembles b128 for the
/// tuple-verify). Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_b128_uuid_key_elides_and_validates_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE ut (a INT, u UUID, v INT, PRIMARY KEY (a, u))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(2);
    let uuid = |n: u32| format!("00000000-0000-0000-0000-{:012x}", n);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    // Pre-elision sentinel with a HIGH-word-nonzero uuid (exercises all 4 folded words on rebuild).
    sql!(&format!(
        "INSERT INTO ut VALUES (5, '{}', 0)",
        "ffffffff-0000-0000-0000-000000000001"
    ))
    .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ut")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_u32 {
        sql!(&format!(
            "INSERT INTO ut VALUES ({}, '{}', 0)",
            1000 + i,
            uuid(i)
        ))
        .unwrap();
        if engine.table_device_authoritative("ut") {
            warmed = true;
            break;
        }
    }
    assert!(
        warmed,
        "uuid compound-PK table never entered elision on a GPU box"
    );

    // Tuple uniqueness over the b128 key: a distinct uuid commits; the exact (a,u) tuple repeats -> 23505.
    sql!(&format!("INSERT INTO ut VALUES (5, '{}', 1)", uuid(7))).unwrap();
    let dup = sql!(&format!(
        "INSERT INTO ut VALUES (5, '{}', 9)",
        "ffffffff-0000-0000-0000-000000000001"
    ))
    .unwrap_err()
    .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate uuid compound tuple must raise 23505 (4-word fold agreement), got: {dup}"
    );
    assert!(engine.table_device_authoritative("ut"));

    // DELETE by the b128 (uuid) compound key stays DEVICE-NATIVE: the WHERE uuid literal is coerced
    // Text->Uuid (`bind_delete_filter_groups`), the fingerprint probe locates the slot, and materialize
    // reassembles the uuid for the tuple-verify. Assert elision-retention BEFORE any verifying read.
    let resolve_before = engine.dml_device_resolve_hits();
    sql!(&format!(
        "DELETE FROM ut WHERE a = 5 AND u = '{}'",
        "ffffffff-0000-0000-0000-000000000001"
    ))
    .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > resolve_before,
        "uuid compound DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("ut"),
        "uuid compound DELETE must stay device-native, not de-elide"
    );
    // Correctness (may de-elide the versioned table): exactly the (5, uuid(7)) row remains for a=5.
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM ut WHERE a = 5").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&count).unwrap().rows,
        vec![vec![SqlValue::Int8(1)]],
        "deleted the (5, ffff...0001) row; (5, ...0007) remains"
    );
}

/// COMPOUND KEYS (wider types, Stage 2d): a compound PRIMARY KEY over a TEXT column — mixed with int4 —
/// elides and enforces uniqueness ON THE DEVICE. A text key column is variable-length, so it folds to ONE
/// word = the FNV-1a hash of its UTF-8 bytes; the device fold kernel's TEXT branch (`widths[k] == 0`) reads
/// the row's `[start,end)` blob span from the shard's text section and hashes it BYTE-IDENTICALLY to the
/// host `fnv1a_bytes` (so the device index rebuild and the host probe needle agree). A fingerprint collision
/// can only OVER-report a hit, which the full-tuple recheck — now materializing the resident text on-device
/// (`materialize_resident_row_via_hit` Text arm) — separates. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_text_key_elides_and_validates_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE tt (a INT, s TEXT, v INT, PRIMARY KEY (a, s))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    // Pre-elision sentinel: a distinctive text key that must survive the device fold on rebuild.
    sql!("INSERT INTO tt VALUES (5, 'alpha-KEY', 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("tt")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_u32 {
        sql!(&format!(
            "INSERT INTO tt VALUES ({}, 'k{}', 0)",
            1000 + i,
            i
        ))
        .unwrap();
        if engine.table_device_authoritative("tt") {
            warmed = true;
            break;
        }
    }
    assert!(
        warmed,
        "text compound-PK table never entered elision on a GPU box"
    );

    // The DEVICE write-locate must actually fire for the text compound key (non-vacuity).
    let hits_before = engine.device_write_locate_hits();
    // A brand-new distinct tuple commits.
    sql!("INSERT INTO tt VALUES (5, 'beta', 10)").unwrap();
    // The EXACT (a, s) tuple again -> 23505 (device fingerprint hit, on-device text tuple recheck confirms).
    let dup = sql!("INSERT INTO tt VALUES (5, 'beta', 99)")
        .unwrap_err()
        .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate text compound tuple must raise 23505 (text-hash fold agreement), got: {dup}"
    );
    // Re-insert the PRE-ELISION sentinel tuple -> 23505: proves the DEVICE rebuild fold of the resident
    // text blob byte-matches the HOST probe needle's `fnv1a_bytes`.
    let dup_sentinel = sql!("INSERT INTO tt VALUES (5, 'alpha-KEY', 7)")
        .unwrap_err()
        .to_string();
    assert!(
        dup_sentinel.contains("duplicate key value violates unique index"),
        "re-inserting the pre-elision text tuple must raise 23505 (device rebuild == host needle), got: {dup_sentinel}"
    );
    // Same first key column, DIFFERENT text -> a DISTINCT tuple, must commit (not first-col-only).
    sql!("INSERT INTO tt VALUES (5, 'gamma', 11)").unwrap();
    // Same text, DIFFERENT first column -> also distinct, must commit.
    sql!("INSERT INTO tt VALUES (9, 'beta', 12)").unwrap();

    assert!(
        engine.device_write_locate_hits() > hits_before,
        "the text compound-key wave validation must run on the DEVICE (write-locate counter advanced)"
    );
    // Uniqueness never de-elided the table (sustained device path).
    assert!(
        engine.table_device_authoritative("tt"),
        "text compound-PK table must stay elided across the validated inserts"
    );

    // Read-your-writes over the elided text-compound table: the distinct tuples for a=5 are exactly
    // {alpha-KEY, beta, gamma} (3 rows).
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM tt WHERE a = 5").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&count).unwrap().rows,
        vec![vec![SqlValue::Int8(3)]],
        "a=5 holds exactly the 3 distinct text tuples"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006): the DECLINED-shape general-executor read fallback. A wider-type
/// (int8 / numeric / text) SELECT shape the SPECIALIZED resident route does NOT recognize — a scalar
/// aggregate, a filtered projection, DISTINCT, GROUP BY, single-key ORDER BY, `SELECT *` over a text
/// table — used to DE-ELIDE the table and run on the CPU relational engine. It now routes to the GENERAL
/// GPU Expr executor instead: the read stays ON THE DEVICE (`general_read_fallback_hits` advances) and the
/// table STAYS ELIDED (no rehydrate), while the result matches the expected (spec) answer. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_general_read_fallback_serves_declined_wider_type_shapes_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE t (id INT PRIMARY KEY, b INT8, g INT8, s TEXT)",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let mut txn = 2u64;
    // A fixed, KNOWN dataset (id 1..=12; b=id*10; g=id%3; s="v{id}"). Insert the first row, then guard on
    // a usable GPU, then insert the rest — the table elides during ingest and holds the full 12 rows.
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 10, 1, 'v1')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // self-guard: no usable GPU
    }
    for id in 2..=12i64 {
        engine
            .execute_dml_concurrent(
                txn,
                &format!(
                    "INSERT INTO t VALUES ({id}, {}, {}, 'v{id}')",
                    id * 10,
                    id % 3
                ),
            )
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "the wider-type PK'd table must elide before the read fallback is exercised"
    );

    // Each shape: (SQL, expected rows). Set-valued shapes are compared order-insensitively.
    let sorted = |mut rows: Vec<Vec<SqlValue>>| {
        rows.sort_by_key(|r| format!("{r:?}"));
        rows
    };
    let run = |engine: &Engine, shape: &str| -> (bool, bool, Vec<Vec<SqlValue>>) {
        let before = engine.general_read_fallback_hits();
        let Command::Select(select) = parse_command(shape).unwrap() else {
            unreachable!()
        };
        let res = engine.execute_relational_select(&select).unwrap();
        let fired = engine.general_read_fallback_hits() > before;
        let elided_after = engine.table_device_authoritative("t");
        (
            fired,
            elided_after,
            res.rows.iter().map(|r| r.to_vec()).collect(),
        )
    };

    // int8 scalar aggregate: SUM(bigint) -> numeric (PG spec); sum(10..120 step 10) = 780.
    let cases: Vec<(&str, Vec<Vec<SqlValue>>)> = vec![
        (
            "SELECT SUM(b) FROM t",
            vec![vec![SqlValue::Numeric(gpu_db_sql::Decimal128::new(780, 0))]],
        ),
        // int8-filtered projection: b = 50 -> id 5.
        (
            "SELECT id FROM t WHERE b = 50",
            vec![vec![SqlValue::Int4(5)]],
        ),
        // DISTINCT over an int8 column -> {0,1,2}.
        (
            "SELECT DISTINCT g FROM t",
            vec![
                vec![SqlValue::Int8(0)],
                vec![SqlValue::Int8(1)],
                vec![SqlValue::Int8(2)],
            ],
        ),
        // GROUP BY an int8 column -> each residue class has 4 members.
        (
            "SELECT g, COUNT(*) FROM t GROUP BY g",
            vec![
                vec![SqlValue::Int8(0), SqlValue::Int8(4)],
                vec![SqlValue::Int8(1), SqlValue::Int8(4)],
                vec![SqlValue::Int8(2), SqlValue::Int8(4)],
            ],
        ),
        // single-key ORDER BY on an int8 column + LIMIT -> the 3 smallest b (ids 1,2,3).
        (
            "SELECT id FROM t ORDER BY b LIMIT 3",
            vec![
                vec![SqlValue::Int4(1)],
                vec![SqlValue::Int4(2)],
                vec![SqlValue::Int4(3)],
            ],
        ),
        // SELECT * over a TEXT-bearing table (the resident route excludes text from SELECT *).
        (
            "SELECT * FROM t WHERE id = 7",
            vec![vec![
                SqlValue::Int4(7),
                SqlValue::Int8(70),
                SqlValue::Int8(1),
                SqlValue::Text("v7".to_string()),
            ]],
        ),
        // EMPTY-filtered scalar aggregate: SUM over zero rows is NULL (PG spec), served on-device
        // WITHOUT de-eliding (the general executor's empty-set guard returns NULL, not a hard error).
        (
            "SELECT SUM(b) FROM t WHERE b = 999999",
            vec![vec![SqlValue::Null]],
        ),
    ];

    for (shape, expected) in cases {
        let (fired, elided_after, rows) = run(&engine, shape);
        assert!(
            fired,
            "shape {shape:?} must be served by the GENERAL GPU executor (fallback counter must advance), \
             without transferring relational authority off device"
        );
        assert!(
            elided_after,
            "shape {shape:?} must keep the table ELIDED (the on-device read must not rehydrate)"
        );
        assert_eq!(
            sorted(rows),
            sorted(expected),
            "shape {shape:?} result must match the spec answer"
        );
    }
}

/// A ZERO-MATCH DELETE / UPDATE is a data no-op and must retain the current device generation.
/// The commit reports handled without allowing the empty result to establish authority for an
/// otherwise nonresident table. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_zero_match_dml_keeps_table_elided() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, b INT8, s TEXT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 100, 'v1')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // self-guard: no usable GPU
    }
    for id in 2..=6i64 {
        engine
            .execute_dml_concurrent(
                txn,
                &format!("INSERT INTO t VALUES ({id}, {}, 'v{id}')", id * 100),
            )
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "table must elide first"
    );

    let count = |engine: &Engine| -> i64 {
        let Command::Select(s) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
            unreachable!()
        };
        match engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .next()
            .and_then(|r| r.first())
        {
            Some(SqlValue::Int8(n)) => *n,
            other => panic!("unexpected COUNT: {other:?}"),
        }
    };
    assert_eq!(count(&engine), 6, "6 rows committed");

    // Zero-match POINT DELETE / UPDATE (WHERE pk = <absent>) -> data no-op, must STAY ELIDED. (A RANGE
    // zero-match de-elides in the PREPARE phase via a separate trigger — the non-point device resolve —
    // handled in a follow-up slice; this slice closes the point-lookup no-match trigger in the commit path.)
    for stmt in [
        "DELETE FROM t WHERE id = 99999",
        "UPDATE t SET b = 0 WHERE id = 99999",
    ] {
        engine.execute_dml_concurrent(txn, stmt).unwrap();
        txn += 1;
        assert!(
            engine.table_device_authoritative("t"),
            "zero-match {stmt:?} must NOT de-elide"
        );
        assert_eq!(count(&engine), 6, "zero-match {stmt:?} changed no rows");
    }

    // Sanity: a MATCHING DELETE still works + stays elided (the fix didn't break the real path).
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE id = 3")
        .unwrap();
    assert!(
        engine.table_device_authoritative("t"),
        "a matching DELETE stays elided"
    );
    assert_eq!(
        count(&engine),
        5,
        "the matching DELETE removed exactly one row"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, NULL coverage): an INSERT carrying a NULL used to DE-ELIDE the table
/// (the in-place append has no validity-bitmap channel, so it declined -> re-admit). Now a null-carrying
/// batch rolls a DENSE shard whose validity bitmaps the payload builder constructs (like TEXT), so the
/// table STAYS ELIDED and reads NULL-correctly on-device. And the rehydrate GATHER
/// (`gather_resident_table_rows_from_device`) now materializes a null-bearing shard (reads the bitmap ->
/// SqlValue::Null) instead of declining + hard-erroring — so a DML that must rehydrate a null-bearing
/// elided table de-elides SAFELY (correct), never crashes. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_null_insert_keeps_table_elided_and_reads_correctly() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 10)")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // self-guard: no usable GPU
    }
    for id in 2..=6i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {})", id * 10))
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "table must elide first"
    );

    // INSERT a NULL value: must STAY ELIDED (was: de-elide).
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (7, NULL)")
        .unwrap();
    txn += 1;
    assert!(
        engine.table_device_authoritative("t"),
        "a NULL insert must NOT de-elide the table (the rollover builds the validity bitmap)"
    );

    // Read the NULL back ON-DEVICE (the row set stayed elided). Use the text entry so `IS NULL` (which the
    // strict hand-rolled parser rejects) routes through the general executor.
    let read = |engine: &Engine, sql: &str| -> Vec<Vec<SqlValue>> {
        let mut rows: Vec<Vec<SqlValue>> = engine
            .execute_relational_select_text(sql)
            .unwrap()
            .rows
            .iter()
            .map(|r| r.to_vec())
            .collect();
        rows.sort_by_key(|r| format!("{r:?}"));
        rows
    };
    assert_eq!(
        read(&engine, "SELECT id, v FROM t WHERE id = 7"),
        vec![vec![SqlValue::Int4(7), SqlValue::Null]],
        "the NULL reads back as NULL, not a phantom 0"
    );
    assert_eq!(
        read(&engine, "SELECT COUNT(*) FROM t"),
        vec![vec![SqlValue::Int8(7)]],
        "all 7 rows present (6 + the null row)"
    );
    // 3VL: IS NULL finds exactly the null row; a value filter excludes it.
    assert_eq!(
        read(&engine, "SELECT id FROM t WHERE v IS NULL"),
        vec![vec![SqlValue::Int4(7)]],
        "IS NULL finds exactly the null row"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "reads must not de-elide the null-bearing table"
    );

    // When device materialization declines, the RETIRE-002 repair path rehydrates this
    // null-bearing table safely and preserves structural NULLs.
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE id = 5")
        .unwrap();
    assert_eq!(
        read(&engine, "SELECT COUNT(*) FROM t"),
        vec![vec![SqlValue::Int8(6)]],
        "the DELETE removed exactly one row (repair preserved the null-bearing table)"
    );
    assert_eq!(
        read(&engine, "SELECT id, v FROM t WHERE id = 7"),
        vec![vec![SqlValue::Int4(7), SqlValue::Null]],
        "the null row survives repair with its NULL intact"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006): a RANGE / non-point DELETE / UPDATE on an ELIDED table now resolves on
/// the DEVICE via the predicate scan-locate (`try_resolve_dml_via_predicate_scan` -> the WHERE lowered to a
/// ResidentExpr, evaluated per shard by `lower_resident_predicate`, each matching slot materialized with
/// SV3b/SV6 visibility + the full WHERE rechecked) instead of REHYDRATING (de-eliding) in the prepare
/// phase. A zero-match range stays elided (no churn); a matching range deletes/updates the exact rows and
/// STAYS ELIDED, with the device resolve counter advancing (non-vacuity). Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_range_dml_resolves_on_device_without_deelide() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 10)")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // self-guard: no usable GPU
    }
    for id in 2..=8i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {})", id * 10))
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), (1..=8).collect::<Vec<_>>());

    // Zero-match RANGE DELETE -> device resolve, no rows, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE id > 100000")
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "zero-match range DELETE must RESOLVE on the device (counter advances)"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "zero-match range DELETE must NOT de-elide"
    );
    assert_eq!(ids(&engine), (1..=8).collect::<Vec<_>>(), "no rows deleted");

    // Matching RANGE DELETE (id > 6) -> deletes ids 7,8 ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE id > 6")
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "matching range DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "matching range DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        (1..=6).collect::<Vec<_>>(),
        "ids 7,8 deleted exactly"
    );

    // Matching RANGE UPDATE (id <= 2 SET v=0) -> updates ids 1,2 ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "UPDATE t SET v = 0 WHERE id <= 2")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "matching range UPDATE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "matching range UPDATE must NOT de-elide"
    );
    // The update kept the row set (still ids 1..=6) and set v=0 for ids 1,2.
    assert_eq!(
        ids(&engine),
        (1..=6).collect::<Vec<_>>(),
        "UPDATE changed no id set"
    );
    let Command::Select(cnt) = parse_command("SELECT COUNT(*) FROM t WHERE v = 0").unwrap() else {
        unreachable!()
    };
    assert_eq!(
        engine
            .execute_relational_select(&cnt)
            .unwrap()
            .rows
            .iter()
            .next()
            .and_then(|r| r.first()),
        Some(&SqlValue::Int8(2)),
        "exactly ids 1,2 now have v=0"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, wider-type range DML): an INT8 range DELETE/UPDATE on an ELIDED table
/// resolves ON THE DEVICE via `try_resolve_dml_via_predicate_scan` — the WHERE lowers to an int8
/// `ResidentExpr` (`Column(int8) <op> Int8Literal`, the new VM literal) evaluated at I64 width by
/// `CompareScalarI64`, so a LARGE i64 bound (> i32::MAX, e.g. a bigint/timestamp-scale value) that cannot
/// fit an Int4Literal resolves on-device instead of de-eliding. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_int8_range_dml_resolves_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, b INT8)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    // b = id * 2_000_000_000 -> ids 3..=8 have b > i32::MAX (2.1e9), so the bound cannot be an Int4Literal.
    let big = 2_000_000_000i64;
    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES (1, {})", big))
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // self-guard: no usable GPU
    }
    for id in 2..=8i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {})", id * big))
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), (1..=8).collect::<Vec<_>>());

    // A LARGE-bound int8 range DELETE (b > 6e9 -> ids 4..=8) resolves ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, &format!("DELETE FROM t WHERE b > {}", 6 * big))
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "int8 range DELETE with a >i32 bound must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "int8 range DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        (1..=6).collect::<Vec<_>>(),
        "ids 7,8 (b=14e9,16e9) deleted exactly"
    );

    // A LARGE-bound int8 range UPDATE (b <= 4e9 = 2*big -> ids 1,2; 4e9 > i32::MAX) resolves ON THE
    // DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, &format!("UPDATE t SET b = 0 WHERE b <= {}", 2 * big))
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "int8 range UPDATE with a >i32 bound must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "int8 range UPDATE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        (1..=6).collect::<Vec<_>>(),
        "UPDATE changed no id set"
    );
    let Command::Select(cnt) = parse_command("SELECT COUNT(*) FROM t WHERE b = 0").unwrap() else {
        unreachable!()
    };
    assert_eq!(
        engine
            .execute_relational_select(&cnt)
            .unwrap()
            .rows
            .iter()
            .next()
            .and_then(|r| r.first()),
        Some(&SqlValue::Int8(2)),
        "exactly ids 1,2 (b=2e9,4e9) now have b=0"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, wider-type range DML): a single-comparison TIMESTAMP range DELETE (the
/// common data-purge shape, `WHERE ts < '<cutoff>'`) on an ELIDED table resolves ON THE DEVICE — the DML
/// predicate builder emits `Column(ts) <op> Int8Literal(micros)` (a timestamp is i64 micros in the i64
/// section) and the timestamp peephole, now accepting a raw-micros Int8Literal, evaluates it via the i64
/// compare kernel. Stays elided; deletes the exact rows. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_timestamp_range_delete_resolves_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, ts TIMESTAMP)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    // ids 1..=6 at ts = 2020-01..06-01. Purge everything strictly before 2020-04-01 -> ids 1,2,3.
    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, '2020-01-01 00:00:00')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // self-guard: no usable GPU
    }
    for (id, month) in (2..=6i64).zip(2..=6) {
        engine
            .execute_dml_concurrent(
                txn,
                &format!("INSERT INTO t VALUES ({id}, '2020-0{month}-01 00:00:00')"),
            )
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>());

    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE ts < '2020-04-01 00:00:00'")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "timestamp range DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "timestamp range DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![4, 5, 6],
        "ids 1,2,3 (Jan-Mar) purged exactly"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a MULTI-BOUND timestamp range DELETE/UPDATE
/// (`ts >= X AND ts <= Y`) on an ELIDED table resolves ON THE DEVICE. A timestamp is i64 micros in the
/// i64 section, so the AND now lowers on the i64 buffer VM (the SAME path int8 AND/OR uses) — the DML
/// builder emits `And(Column(ts) >= Int8Literal, Column(ts) <= Int8Literal)`, `resident_device_int_
/// column_offset` resolves the timestamp column to the i64 section, and `compile_predicate_program`
/// emits `CompareScalarI64` per bound. No host store, no new kernel. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_timestamp_multibound_range_dml_resolves_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, ts TIMESTAMP)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    // ids 1..=6 at ts = 2020-01..06-01.
    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, '2020-01-01 00:00:00')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    for (id, month) in (2..=6i64).zip(2..=6) {
        engine
            .execute_dml_concurrent(
                txn,
                &format!("INSERT INTO t VALUES ({id}, '2020-0{month}-01 00:00:00')"),
            )
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>());

    // MULTI-BOUND range DELETE (2020-02-01 <= ts <= 2020-04-01 -> ids 2,3,4) — the AND path — ON DEVICE.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            txn,
            "DELETE FROM t WHERE ts >= '2020-02-01 00:00:00' AND ts <= '2020-04-01 00:00:00'",
        )
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a multi-bound timestamp range DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a multi-bound timestamp range DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![1, 5, 6],
        "ids 2,3,4 (Feb-Apr) purged exactly"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a DELETE/UPDATE on a table with NULL-BEARING columns
/// resolves ON THE DEVICE — `materialize_resident_row_via_hit` no longer declines wholesale on a
/// null-bearing shard; it reads each column's validity bitmap per-slot (0 bit -> SqlValue::Null) so the
/// recheck sees the real row (with NULLs). A matched row whose VALUE column is NULL is handled: the
/// device locate excluded NULL predicate operands (3VL), and the recheck re-applies 3VL. No host store.
/// GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nullable_column_dml_resolves_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, notes TEXT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    // id=1 'a', id=2 NULL, id=3 'c', id=4 NULL — a nullable value column that actually holds NULLs.
    engine
        .execute_dml_concurrent(2, "INSERT INTO t VALUES (1, 'a')")
        .unwrap();
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    engine
        .execute_dml_concurrent(3, "INSERT INTO t (id, notes) VALUES (2, NULL)")
        .unwrap();
    engine
        .execute_dml_concurrent(4, "INSERT INTO t VALUES (3, 'c')")
        .unwrap();
    engine
        .execute_dml_concurrent(5, "INSERT INTO t (id, notes) VALUES (4, NULL)")
        .unwrap();
    assert!(
        engine.table_device_authoritative("t"),
        "a nullable-column table must elide (NULL coverage)"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), vec![1, 2, 3, 4]);

    // RANGE DELETE (id 2..=3) — matches a NULL-notes row (id=2) — resolves ON THE DEVICE via the
    // predicate scan + null-aware materialize, and STAYS ELIDED (pre-fix this de-elided: materialize
    // declined on the null-bearing shard).
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(6, "DELETE FROM t WHERE id >= 2 AND id <= 3")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a DELETE on a null-bearing table must RESOLVE on the device (materialize is null-aware)"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a DELETE on a null-bearing table must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![1, 4],
        "ids 2 (NULL) and 3 deleted exactly"
    );

    // The surviving NULL row (id=4) is intact + reads back as NULL.
    let Command::Select(sel) = parse_command("SELECT notes FROM t WHERE id = 4").unwrap() else {
        unreachable!()
    };
    assert_eq!(
        engine
            .execute_relational_select(&sel)
            .unwrap()
            .rows
            .iter()
            .next()
            .and_then(|r| r.first()),
        Some(&SqlValue::Null),
        "id=4's notes is still NULL after the on-device DELETE"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, wider-type range DML): a NUMERIC range DELETE/UPDATE on an ELIDED table
/// resolves ON THE DEVICE — the DML predicate builder emits `Column(num) <op> NumericLiteral(dec)`, lowered
/// by the numeric peephole via the i128 compare kernel (rescaled to the column scale), which ALSO handles
/// AND/OR — so a MULTI-BOUND range (`amt >= a AND amt <= b`) resolves on-device too. Stays elided; touches
/// the exact rows. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_numeric_range_dml_resolves_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, amt NUMERIC(12,2))")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    // ids 1..=6 at amt = id * 50.00 -> 50, 100, 150, 200, 250, 300.
    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 50.00)")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // self-guard: no usable GPU
    }
    for id in 2..=6i64 {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {}.00)", id * 50))
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), (1..=6).collect::<Vec<_>>());

    // Single-bound numeric range DELETE (amt > 250.00 -> id 6) ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE amt > 250.00")
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "numeric range DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "numeric range DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        (1..=5).collect::<Vec<_>>(),
        "id 6 (amt=300) deleted exactly"
    );

    // MULTI-BOUND numeric range UPDATE (100 <= amt <= 200 -> ids 2,3,4) — the AND path — ON THE DEVICE.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            txn,
            "UPDATE t SET amt = 0.00 WHERE amt >= 100.00 AND amt <= 200.00",
        )
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "multi-bound numeric range UPDATE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "numeric range UPDATE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        (1..=5).collect::<Vec<_>>(),
        "UPDATE changed no id set"
    );
    let rows = engine
        .execute_relational_select_text("SELECT id, amt FROM t")
        .unwrap()
        .rows
        .into_boxed();
    let mut zero_ids = rows
        .iter()
        .filter_map(|row| match row.as_slice() {
            [SqlValue::Int4(id), SqlValue::Numeric(value)] if value.mantissa == 0 => Some(*id),
            _ => None,
        })
        .collect::<Vec<_>>();
    zero_ids.sort_unstable();
    assert_eq!(
        zero_ids,
        vec![2, 3, 4],
        "exactly ids 2,3,4 (amt 100,150,200) now have amt=0"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure text coverage): a TEXT-EQUALITY DELETE/UPDATE on an
/// ELIDED table resolves ON THE DEVICE — the DML predicate builder now emits `Column(text) = TextLiteral`,
/// which `lower_resident_predicate` evaluates via the existing DEVICE byte-wise text-equality kernel
/// (`try_lower_text_predicate`); the located slots materialize their text on-device
/// (`materialize_resident_row_via_hit` text arm) for the recheck. NO host store, NO new kernel — reuses
/// the read path. Stays elided; touches exactly the matching rows. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_text_predicate_dml_resolves_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let mut txn = 2u64;
    engine
        .execute_dml_concurrent(txn, "INSERT INTO t VALUES (1, 'alice')")
        .unwrap();
    txn += 1;
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    for (id, name) in [(2i64, "bob"), (3, "carol"), (4, "bob")] {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, '{name}')"))
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "the text table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), vec![1, 2, 3, 4]);

    // TEXT-EQ DELETE (name = 'bob' -> ids 2, 4) ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "DELETE FROM t WHERE name = 'bob'")
        .unwrap();
    txn += 1;
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a text-equality DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a text-equality DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![1, 3],
        "exactly the two 'bob' rows (2,4) deleted"
    );

    // TEXT-EQ UPDATE (name = 'carol' -> id 3) ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(txn, "UPDATE t SET name = 'CAROL' WHERE name = 'carol'")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a text-equality UPDATE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a text-equality UPDATE must NOT de-elide"
    );
    let Command::Select(cnt) =
        parse_command("SELECT COUNT(*) FROM t WHERE name = 'CAROL'").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine
            .execute_relational_select(&cnt)
            .unwrap()
            .rows
            .iter()
            .next()
            .and_then(|r| r.first()),
        Some(&SqlValue::Int8(1)),
        "id 3's name is now 'CAROL' (text UPDATE applied on-device)"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a `LIKE 'prefix%'` DELETE on an ELIDED table resolves
/// ON THE DEVICE — the DML builder lowers `name LIKE 'bo%'` to `Column(name) Like TextLiteral('bo%')`
/// (the escaped pattern, byte-identical to the read path), which `try_lower_text_predicate` evaluates via
/// the existing DEVICE text-LIKE kernel `expr_text_like_scalar_filter`; the recheck re-applies
/// `starts_with`. Matches EVERY row with the prefix ('bob' AND 'bobby'), none without. No host store,
/// no new kernel. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_like_prefix_dml_resolves_on_device() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    engine
        .execute_dml_concurrent(2, "INSERT INTO t VALUES (1, 'alice')")
        .unwrap();
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    for (id, name) in [(2i64, "bob"), (3, "bobby"), (4, "carol")] {
        engine
            .execute_dml_concurrent(
                id as u64 + 1,
                &format!("INSERT INTO t VALUES ({id}, '{name}')"),
            )
            .unwrap();
    }
    assert!(
        engine.table_device_authoritative("t"),
        "the text table must elide first"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), vec![1, 2, 3, 4]);

    // LIKE-prefix DELETE (name LIKE 'bo%' -> 'bob' AND 'bobby' = ids 2,3) ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(9, "DELETE FROM t WHERE name LIKE 'bo%'")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a LIKE-prefix DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a LIKE-prefix DELETE must NOT de-elide"
    );
    assert_eq!(
        ids(&engine),
        vec![1, 4],
        "exactly the 'bo'-prefixed rows (bob, bobby) deleted; alice/carol kept"
    );
}

/// Shared harness: an elided int4-PK table with one extra typed column, seeded 1-per-commit and admitted.
/// Returns `None` (self-guard) with no usable GPU. `col_ddl` is the extra column (e.g. "u UUID").
#[cfg(test)]
fn gpu_elided_pk_table_with_column(col_ddl: &str, seed: &[(i64, &str)]) -> Option<Engine> {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            &format!("CREATE TABLE t (id INT PRIMARY KEY, {col_ddl})"),
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let mut txn = 2u64;
    if let Some((id, v)) = seed.first() {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {v})"))
            .unwrap();
        txn += 1;
    }
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return None; // no usable GPU
    }
    for (id, v) in seed.iter().skip(1) {
        engine
            .execute_dml_concurrent(txn, &format!("INSERT INTO t VALUES ({id}, {v})"))
            .unwrap();
        txn += 1;
    }
    assert!(
        engine.table_device_authoritative("t"),
        "the table must elide first"
    );
    Some(engine)
}

#[cfg(test)]
fn gpu_ids_of_t(engine: &Engine) -> Vec<i64> {
    let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a UUID-EQUALITY DELETE on an ELIDED table resolves ON
/// THE DEVICE — the DML builder lowers `u = 'uuid'` to a `TextLiteral` the existing device
/// `try_lower_uuid_predicate` (byte-wise b128 compare) evaluates; the hit materializes its uuid on-device
/// (b128 arm) for the recheck. No host store, no new kernel. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_uuid_predicate_dml_resolves_on_device() {
    let a = "'00000000-0000-0000-0000-000000000001'";
    let b = "'00000000-0000-0000-0000-000000000002'";
    let Some(engine) = gpu_elided_pk_table_with_column("u UUID", &[(1, a), (2, b), (3, a), (4, b)])
    else {
        return;
    };
    assert_eq!(gpu_ids_of_t(&engine), vec![1, 2, 3, 4]);

    // DELETE WHERE u = <a> -> ids 1, 3, ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            9,
            "DELETE FROM t WHERE u = '00000000-0000-0000-0000-000000000001'",
        )
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a uuid-equality DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a uuid-equality DELETE must NOT de-elide"
    );
    assert_eq!(
        gpu_ids_of_t(&engine),
        vec![2, 4],
        "exactly the two <a>-uuid rows deleted"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): uuid comparisons INSIDE AND/OR resolve ON THE DEVICE
/// via the new `UuidCmpMask` mask-VM step (the same b128 memcmp kernel, composed with MaskBinary) —
/// uuid RANGES (`u >= A AND u <= B`), uuid IN (an OR of `=`), and MIXED uuid+int4 WHEREs. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_uuid_range_and_in_dml_resolve_on_device() {
    let u = |n: u8| format!("'00000000-0000-0000-0000-0000000000{n:02}'");
    let seed_uuids: Vec<(i64, String)> = (1..=5i64).map(|i| (i, u(i as u8))).collect();
    let seed: Vec<(i64, &str)> = seed_uuids.iter().map(|(i, s)| (*i, s.as_str())).collect();

    // uuid RANGE: <02> <= u <= <04> -> ids 2,3,4.
    {
        let Some(engine) = gpu_elided_pk_table_with_column("u UUID", &seed) else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(
                9,
                "DELETE FROM t WHERE u >= '00000000-0000-0000-0000-000000000002' \
                 AND u <= '00000000-0000-0000-0000-000000000004'",
            )
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a uuid RANGE DELETE must RESOLVE on the device (UuidCmpMask in the mask VM)"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "uuid range DELETE must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![1, 5],
            "ids 2,3,4 (the range) deleted"
        );
    }
    // uuid IN: (<01>, <04>) -> ids 1,4 (an OR of uuid equalities through the mask VM).
    {
        let Some(engine) = gpu_elided_pk_table_with_column("u UUID", &seed) else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(
                9,
                "DELETE FROM t WHERE u IN ('00000000-0000-0000-0000-000000000001', \
                 '00000000-0000-0000-0000-000000000004')",
            )
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a uuid IN DELETE must RESOLVE on the device"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "uuid IN DELETE must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![2, 3, 5],
            "ids 1,4 (the IN list) deleted"
        );
    }
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a NULLABLE-uuid RANGE DELETE resolves ON THE DEVICE —
/// the nullable branch's new LOCAL I32-mask gate routes a uuid AND/OR through the mask VM (UuidCmpMask +
/// per-leaf validity AND), so the NULL uuid row is excluded by 3VL (its 16-zero-byte placeholder would
/// otherwise sort below every needle) and SURVIVES a wide range purge. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nullable_uuid_range_dml_resolves_on_device() {
    let u = |n: u8| format!("'00000000-0000-0000-0000-0000000000{n:02}'");
    let seed_uuids: Vec<(i64, String)> =
        vec![(1, u(1)), (2, u(2)), (3, "NULL".to_string()), (4, u(4))];
    let seed: Vec<(i64, &str)> = seed_uuids.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let Some(engine) = gpu_elided_pk_table_with_column("u UUID", &seed) else {
        return;
    };
    // READ 3VL PIN (audit MEDIUM adopted — the decisive assertion): a zero-anchored range READ has NO
    // recheck net (the mask-VM answer is authoritative), and the NULL uuid's device placeholder is 16
    // ZERO BYTES, which MATCHES both bounds — ONLY `compile_uuid_leaf`'s per-leaf validity-AND excludes
    // it. The NULL row (id=3) must be ABSENT from the read result.
    {
        let Command::Select(s) = parse_command(
            "SELECT id FROM t WHERE u >= '00000000-0000-0000-0000-000000000000' \
             AND u <= '00000000-0000-0000-0000-000000000004'",
        )
        .unwrap() else {
            unreachable!()
        };
        let mut got: Vec<i32> = engine
            .execute_relational_select(&s)
            .unwrap()
            .rows
            .iter()
            .map(|r| match r.first() {
                Some(SqlValue::Int4(n)) => *n,
                other => panic!("unexpected id: {other:?}"),
            })
            .collect();
        got.sort_unstable();
        assert_eq!(
            got,
            vec![1, 2, 4],
            "the zero-anchored uuid range READ must EXCLUDE the NULL row (id=3) — the per-leaf \
             validity-AND is the only net (the placeholder's zero bytes match both bounds)"
        );
    }

    // Range starting at the ALL-ZERO uuid: <00> <= u <= <04>. Same 3VL through the DML path (which
    // additionally rechecks). The NULL row (id=3) must SURVIVE, on-device + elided.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            9,
            "DELETE FROM t WHERE u >= '00000000-0000-0000-0000-000000000000' \
             AND u <= '00000000-0000-0000-0000-000000000004'",
        )
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a NULLABLE-uuid RANGE DELETE must RESOLVE on the device (the local I32-mask gate)"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a nullable-uuid range DELETE must NOT de-elide"
    );
    assert_eq!(
        gpu_ids_of_t(&engine),
        vec![3],
        "ids 1,2,4 deleted; the NULL-uuid row survives (3VL: NULL is UNKNOWN, never in range)"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a NULLABLE-timestamp RANGE DELETE resolves ON THE
/// DEVICE — the nullable branch's local gate now also runs {Int8, Timestamp} column sets at I64
/// (`Int8Literal` leaves = 8-byte load + CompareScalarI64 + per-leaf validity AND). The range SPANS the
/// NULL placeholder's epoch (0 micros), so only the validity AND keeps the NULL row out of the device
/// mask; the NULL row (id=3) must SURVIVE the purge and the table stays elided. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nullable_timestamp_range_dml_resolves_on_device() {
    let seed: &[(i64, &str)] = &[
        (1, "'2020-01-01 00:00:00'"),
        (2, "'2020-06-01 00:00:00'"),
        (3, "NULL"),
        (4, "'2021-01-01 00:00:00'"),
    ];
    let Some(engine) = gpu_elided_pk_table_with_column("ts TIMESTAMP", seed) else {
        return;
    };
    // The range spans BOTH plausible epoch-0 anchors (1970 / 2000), so the NULL placeholder (0 micros)
    // is INSIDE the range — the per-leaf validity AND is what excludes it on the device.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            9,
            "DELETE FROM t WHERE ts >= '1960-01-01 00:00:00' AND ts <= '2035-01-01 00:00:00'",
        )
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a NULLABLE-timestamp RANGE DELETE must RESOLVE on the device (the local I64 gate)"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a nullable-timestamp range DELETE must NOT de-elide"
    );
    assert_eq!(
        gpu_ids_of_t(&engine),
        vec![3],
        "ids 1,2,4 deleted; the NULL-timestamp row survives (3VL, placeholder-spanning range)"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, coverage pin): a NULLABLE-numeric RANGE DELETE resolves ON THE
/// DEVICE — `try_lower_nullable_numeric_predicate`'s AND/OR arm (validity-aware
/// `compile_numeric_predicate_program` per side + MaskBinary at I128) already serves it; this pins
/// that capability. The range INCLUDES 0.00 (the NULL placeholder mantissa is 0), so only the
/// per-leaf validity AND keeps the NULL row alive — the load-bearing 3VL case. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nullable_numeric_range_dml_resolves_on_device() {
    let seed: &[(i64, &str)] = &[(1, "10.00"), (2, "50.00"), (3, "NULL"), (4, "90.00")];
    let Some(engine) = gpu_elided_pk_table_with_column("amt NUMERIC(12,2)", seed) else {
        return;
    };
    // The range spans 0.00 (the NULL placeholder mantissa) through 100.00 — every non-null row
    // matches, and ONLY the validity AND excludes the NULL row on the device.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(9, "DELETE FROM t WHERE amt >= 0.00 AND amt <= 100.00")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a NULLABLE-numeric RANGE DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a nullable-numeric range DELETE must NOT de-elide"
    );
    assert_eq!(
        gpu_ids_of_t(&engine),
        vec![3],
        "ids 1,2,4 deleted; the NULL-numeric row survives (3VL, placeholder-mantissa-0 in range)"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a DATE RANGE DELETE (`d >= X AND d < Y` — the month
/// purge shape) resolves ON THE DEVICE — the DML builder lowers a date bound to a raw-days
/// `Int4Literal`, and the new DATE VM leaf (4-byte LoadColumn + CompareScalar + validity, I32-only by
/// width discipline) serves the AND. Includes the nullable 3VL pin: the range SPANS the NULL
/// placeholder's epoch (days 0), so only the validity AND keeps the NULL row alive. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_date_range_dml_resolves_on_device() {
    // Non-null: dates Jan..Apr; delete Feb..Mar (>= '2024-02-01' AND < '2024-04-01') -> ids 2,3.
    {
        let seed: &[(i64, &str)] = &[
            (1, "'2024-01-15'"),
            (2, "'2024-02-15'"),
            (3, "'2024-03-15'"),
            (4, "'2024-04-15'"),
        ];
        let Some(engine) = gpu_elided_pk_table_with_column("d DATE", seed) else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(
                9,
                "DELETE FROM t WHERE d >= '2024-02-01' AND d < '2024-04-01'",
            )
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a date RANGE DELETE must RESOLVE on the device (the DATE VM leaf)"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "date range DELETE must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![1, 4],
            "Feb+Mar (ids 2,3) purged exactly"
        );
    }
    // Nullable 3VL pin: the range spans BOTH plausible epoch-0 anchors (1970/2000), so the NULL
    // placeholder (days 0) is IN-range — only the per-leaf validity AND excludes it.
    {
        let seed: &[(i64, &str)] = &[(1, "'2024-01-15'"), (2, "NULL"), (3, "'2024-03-15'")];
        let Some(engine) = gpu_elided_pk_table_with_column("d DATE", seed) else {
            return;
        };
        // READ 3VL PIN (audit LOW adopted — the decisive assertion, doctrine): a placeholder-spanning
        // range READ has NO recheck net; only the DATE leaf's validity-AND excludes the NULL row.
        {
            let Command::Select(s) =
                parse_command("SELECT id FROM t WHERE d >= '1960-01-01' AND d <= '2035-01-01'")
                    .unwrap()
            else {
                unreachable!()
            };
            let mut got: Vec<i32> = engine
                .execute_relational_select(&s)
                .unwrap()
                .rows
                .iter()
                .map(|r| match r.first() {
                    Some(SqlValue::Int4(n)) => *n,
                    other => panic!("unexpected id: {other:?}"),
                })
                .collect();
            got.sort_unstable();
            assert_eq!(
                got,
                vec![1, 3],
                "the placeholder-spanning date range READ must EXCLUDE the NULL row (id=2) — the \
                 per-leaf validity-AND is the only net (placeholder days 0 is in-range)"
            );
            assert!(
                engine.table_device_authoritative("t"),
                "the date range READ must run ON-DEVICE (stay elided); otherwise the [1,3] result \
                 proves nothing about the device route"
            );
        }
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(
                9,
                "DELETE FROM t WHERE d >= '1960-01-01' AND d <= '2035-01-01'",
            )
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a NULLABLE-date RANGE DELETE must RESOLVE on the device"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "nullable-date range must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![2],
            "ids 1,3 deleted; the NULL-date row survives (3VL, placeholder-spanning range)"
        );
    }
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a BOOL-EQUALITY DELETE on an ELIDED table resolves ON
/// THE DEVICE — the DML builder lowers `flag = true` to a `BoolLiteral` the existing device
/// `try_lower_bool_predicate` (1-bit bitmap → mask) evaluates; the hit materializes its bool on-device
/// (new bitmap arm) for the recheck. No host store, no new kernel. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_bool_predicate_dml_resolves_on_device() {
    let Some(engine) = gpu_elided_pk_table_with_column(
        "flag BOOL",
        &[(1, "true"), (2, "false"), (3, "true"), (4, "false")],
    ) else {
        return;
    };
    assert_eq!(gpu_ids_of_t(&engine), vec![1, 2, 3, 4]);

    // DELETE WHERE flag = true -> ids 1, 3, ON THE DEVICE, STAYS ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(9, "DELETE FROM t WHERE flag = true")
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a bool-equality DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a bool-equality DELETE must NOT de-elide"
    );
    assert_eq!(
        gpu_ids_of_t(&engine),
        vec![2, 4],
        "exactly the two flag=true rows deleted"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a UUID ORDERING (`>`) DELETE on an ELIDED table
/// resolves ON THE DEVICE — uuid is byte-comparable (PG's uuid order == the device compare kernel's
/// cmp code == the recheck `compare_sql_values`), so the DML builder now lowers `<`/`>`/`<=`/`>=` (not
/// just `=`). GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_uuid_inequality_dml_resolves_on_device() {
    let u = |n: u8| format!("'00000000-0000-0000-0000-0000000000{n:02}'");
    // id=5 differs in the FIRST (most-significant) byte — a high-byte-dominant value that is byte-wise
    // GREATER than <02> even though its trailing bytes are all zero. This adversarially exercises the
    // MSB-first byte ordering (not just a last-byte difference).
    let high_byte = "'01000000-0000-0000-0000-000000000000'";
    let Some(engine) = gpu_elided_pk_table_with_column(
        "u UUID",
        &[
            (1, &u(1)),
            (2, &u(2)),
            (3, &u(3)),
            (4, &u(4)),
            (5, high_byte),
        ],
    ) else {
        return;
    };
    // DELETE WHERE u > <02> -> ids 3, 4 (last byte) AND 5 (first byte 01 > 00) ON THE DEVICE, ELIDED.
    let before = engine.dml_device_resolve_hits();
    engine
        .execute_dml_concurrent(
            9,
            "DELETE FROM t WHERE u > '00000000-0000-0000-0000-000000000002'",
        )
        .unwrap();
    assert!(
        engine.dml_device_resolve_hits() > before,
        "a uuid ordering DELETE must RESOLVE on the device"
    );
    assert!(
        engine.table_device_authoritative("t"),
        "a uuid ordering DELETE must NOT de-elide"
    );
    assert_eq!(
        gpu_ids_of_t(&engine),
        vec![1, 2],
        "ids 3,4 (last byte) AND 5 (high byte 01>00) deleted; MSB-first byte order"
    );
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): `col IN (...)` DELETE — parsed as an OR of `=` groups
/// — resolves ON THE DEVICE for int4 and text via the OR mask VM (text needles), reusing the equality
/// coverage. Confirms the IN shape composes from the per-type `=` arms. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_in_list_dml_resolves_on_device() {
    // int4 IN.
    {
        let Some(engine) =
            gpu_elided_pk_table_with_column("v INT", &[(1, "10"), (2, "20"), (3, "30"), (4, "40")])
        else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(9, "DELETE FROM t WHERE id IN (2, 4)")
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "an int4 IN DELETE must RESOLVE on the device"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "int4 IN DELETE must NOT de-elide"
        );
        assert_eq!(gpu_ids_of_t(&engine), vec![1, 3], "ids 2,4 deleted");
    }
    // text IN.
    {
        let Some(engine) = gpu_elided_pk_table_with_column(
            "name TEXT",
            &[(1, "'a'"), (2, "'b'"), (3, "'c'"), (4, "'d'")],
        ) else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(9, "DELETE FROM t WHERE name IN ('a', 'c')")
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a text IN DELETE must RESOLVE on the device"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "text IN DELETE must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![2, 4],
            "the 'a' and 'c' rows deleted"
        );
    }
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure — NEW device kernel): text INEQUALITY
/// (`name < 'x'` / `>` / `<=` / `>=`) DELETE/UPDATE on an ELIDED table resolves ON THE DEVICE via the
/// new lexicographic byte-compare kernel `gpu_db_resident_text_compare_scalar_to_mask` (unsigned memcmp
/// of the common prefix; the shorter string sorts first) — BYTE-IDENTICAL to the host
/// `compare_sql_values` Text order (`str::cmp`) the recheck uses. Adversarial: exercises byte-order
/// (uppercase `B`=0x42 < lowercase `b`=0x62), prefix (first byte decides `ab` vs `b`), and the length
/// tiebreak (`ab` > `a`, common prefix equal → longer sorts after). GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_text_inequality_dml_resolves_on_device() {
    // Byte values: 'a'=0x61, 'b'=0x62, 'c'=0x63, 'ab'=61 62, 'B'=0x42 (< lowercase).
    let seed: &[(i64, &str)] = &[(1, "'a'"), (2, "'b'"), (3, "'c'"), (4, "'ab'"), (5, "'B'")];

    // name < 'b': 'a'(<), 'ab'(first byte 'a'<'b'), 'B'(0x42<0x62) -> delete 1,4,5; keep 'b','c'.
    {
        let Some(engine) = gpu_elided_pk_table_with_column("name TEXT", seed) else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(9, "DELETE FROM t WHERE name < 'b'")
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a text inequality DELETE must RESOLVE on the device"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "text `<` must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![2, 3],
            "a, ab, B deleted (byte-order 'B'<'b' + prefix 'ab'<'b'); 'b','c' kept"
        );
    }
    // name > 'a': 'b','c' (>), 'ab' (common prefix 'a' equal, longer -> 'ab' > 'a'); 'B'=0x42 < 'a'=0x61.
    {
        let Some(engine) = gpu_elided_pk_table_with_column("name TEXT", seed) else {
            return;
        };
        engine
            .execute_dml_concurrent(9, "DELETE FROM t WHERE name > 'a'")
            .unwrap();
        assert!(
            engine.table_device_authoritative("t"),
            "text `>` must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![1, 5],
            "b, c, ab deleted (LENGTH TIEBREAK: 'ab' > 'a'); 'a','B' kept"
        );
    }
    // name >= 'b' (inclusive): only 'b','c' -> delete 2,3; keep 'a','ab','B'.
    {
        let Some(engine) = gpu_elided_pk_table_with_column("name TEXT", seed) else {
            return;
        };
        engine
            .execute_dml_concurrent(9, "DELETE FROM t WHERE name >= 'b'")
            .unwrap();
        assert!(
            engine.table_device_authoritative("t"),
            "text `>=` must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![1, 4, 5],
            "only 'b','c' deleted (>= inclusive); 'a','ab','B' kept"
        );
    }
}

/// CPU-ENGINE RETIREMENT (ADR-006, charter-pure): a TEXT RANGE DELETE (`name >= 'b' AND name < 'd'` —
/// a text inequality INSIDE an AND) resolves ON THE DEVICE via the new `TextCmpMask` mask-VM step
/// (the same lexicographic byte-compare kernel, composed with `MaskBinary` AND). Also exercises the
/// NULLABLE-text composition: a NULL `name` row is excluded by the validity AND (3VL), never matched
/// or mis-ordered by its empty placeholder span. GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_text_range_dml_resolves_on_device() {
    // Non-null range: 'a','b','c','d' with `>= 'b' AND < 'd'` -> delete 'b','c' (ids 2,3).
    {
        let Some(engine) = gpu_elided_pk_table_with_column(
            "name TEXT",
            &[(1, "'a'"), (2, "'b'"), (3, "'c'"), (4, "'d'")],
        ) else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(9, "DELETE FROM t WHERE name >= 'b' AND name < 'd'")
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a text RANGE DELETE must RESOLVE on the device (TextCmpMask in the mask VM)"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "text range DELETE must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![1, 4],
            "'b','c' deleted; 'a','d' kept"
        );
    }
    // NULLABLE-text range: a NULL name row must be excluded by 3VL (its empty placeholder span would
    // otherwise sort below 'b' — but NULL is UNKNOWN, not ''), and the table stays elided.
    {
        let Some(engine) = gpu_elided_pk_table_with_column(
            "name TEXT",
            &[(1, "'a'"), (2, "'b'"), (3, "NULL"), (4, "'c'")],
        ) else {
            return;
        };
        let before = engine.dml_device_resolve_hits();
        engine
            .execute_dml_concurrent(9, "DELETE FROM t WHERE name >= 'a' AND name < 'z'")
            .unwrap();
        assert!(
            engine.dml_device_resolve_hits() > before,
            "a nullable-text range DELETE must RESOLVE on the device"
        );
        assert!(
            engine.table_device_authoritative("t"),
            "nullable-text range must NOT de-elide"
        );
        assert_eq!(
            gpu_ids_of_t(&engine),
            vec![3],
            "'a','b','c' deleted; the NULL row survives (3VL: NULL is UNKNOWN, never < 'z')"
        );
    }
}

/// CPU-ENGINE RETIREMENT (ADR-006, multi-statement elision): a MULTI-ENTRY commit BATCH of INSERTs
/// (the group-commit batcher grouping GpuBatched inserts under load — the SQL-text write path) now
/// KEEPS the table ELIDED via ONE incremental device append per table, instead of de-eliding the
/// whole batch scope to the CPU host store (`to_apply.len() > 1` used to rehydrate every touched
/// elided table). Drives the real path: `commit_mutation_batch` -> `apply_and_publish_committed_inner`
/// with `to_apply.len() == 3`. A CONSTRAINT-FREE int4 table is elision-eligible AND its INSERTs group
/// (a unique-index table takes the immediate single-entry commit and never batches). All rows land +
/// read back on-device, the table stays elided, and every entry took the elided host-install skip
/// (device_authoritative_commits += 3 in the one batch). GPU-gated.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_multi_entry_insert_batch_stays_elided() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);

    let payload = |sql: &str| -> std::sync::Arc<[u8]> { std::sync::Arc::from(sql.as_bytes()) };
    let commit_batch = |engine: &Engine, items: &[(u64, std::sync::Arc<[u8]>)]| {
        engine
            .commit_mutation_batch(items)
            .map_err(|failure| failure.error)
            .expect("group commit");
    };

    // Seed row 1 as a batch of ONE (single-entry path), then admit residency — self-guard on no GPU.
    commit_batch(&engine, &[(2, payload("INSERT INTO t VALUES (1, 10)"))]);
    let snap = engine.populate_relational_residency_snapshot("t");
    if snap
        .map(|s| s.device_memory_proof.is_none())
        .unwrap_or(true)
    {
        return; // no usable GPU
    }
    // Two more single-entry commits drive the (now-resident) table into elision — ENTER needs a
    // handled incremental append.
    commit_batch(&engine, &[(3, payload("INSERT INTO t VALUES (2, 20)"))]);
    commit_batch(&engine, &[(4, payload("INSERT INTO t VALUES (3, 30)"))]);
    assert!(
        engine.table_device_authoritative("t"),
        "the table must be elided before the multi-entry batch"
    );

    let ids = |engine: &Engine| -> Vec<i64> {
        let Command::Select(s) = parse_command("SELECT id FROM t").unwrap() else {
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
    assert_eq!(ids(&engine), (1..=3).collect::<Vec<_>>());

    // THE MULTI-ENTRY BATCH: three INSERTs group-committed as ONE commit (`to_apply.len() == 3`).
    // Before ADR-006 multi-statement elision, the `to_apply.len() > 1` guard de-elided the scope.
    let elisions_before = engine.device_authoritative_commits();
    commit_batch(
        &engine,
        &[
            (10, payload("INSERT INTO t VALUES (4, 40)")),
            (11, payload("INSERT INTO t VALUES (5, 50)")),
            (12, payload("INSERT INTO t VALUES (6, 60)")),
        ],
    );

    assert!(
        engine.table_device_authoritative("t"),
        "a multi-entry INSERT batch must NOT de-elide — it stays device-authoritative"
    );
    assert_eq!(
        engine.device_authoritative_commits() - elisions_before,
        3,
        "all three batched INSERTs took the elided host-install skip (multi-entry stayed elided)"
    );
    assert_eq!(
        ids(&engine),
        (1..=6).collect::<Vec<_>>(),
        "every batched row landed on-device and reads back exactly"
    );
}

/// R3-004: even a zero-row DML statement bootstraps the mandatory device generation. Entering device
/// authority is now safe because admission precedes prepare/apply; it no longer implies that a host
/// install was skipped without device backing.
#[test]
fn zero_row_dml_establishes_device_authority_without_losing_followup_writes() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        .unwrap();
    engine.set_auto_admit_on_commit(true);

    // Empty eligible table: DELETE changes no rows but still establishes the device generation.
    engine
        .execute_text(2, "DELETE FROM t WHERE id > 100000")
        .unwrap();
    assert!(
        engine.table_device_authoritative("t"),
        "a zero-row DELETE must leave the admitted table device-authoritative"
    );
    engine
        .execute_text(3, "UPDATE t SET v = 0 WHERE id > 100000")
        .unwrap();
    assert!(
        engine.table_device_authoritative("t"),
        "a zero-row UPDATE must retain device authority"
    );

    // Subsequent real INSERTs still commit and read back from that generation.
    engine
        .execute_text(4, "INSERT INTO t VALUES (1, 10)")
        .unwrap();
    engine
        .execute_text(5, "INSERT INTO t VALUES (2, 20)")
        .unwrap();
    let Command::Select(s) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
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
        Some(&SqlValue::Int8(2)),
        "both inserts visible (no lost rows / no hard error)"
    );
}

/// COMPOUND KEYS (wider types, Stage 2a): a compound PRIMARY KEY over i64 (Int8/Timestamp) columns — and
/// a MIXED int4+int8 key — elides and enforces uniqueness ON THE DEVICE. Each key column folds its i32
/// WORD decomposition into the surrogate fingerprint (i64 -> [low32, high32] LE, matching the section's LE
/// layout); the device fold kernel reads `widths[k]` words per column. Proves: elision, tuple uniqueness
/// (dup -> 23505; distinct OK), the DEVICE fold matches the host needle even across the HIGH word (a
/// value > 2^32 survives geometric rebuilds and its duplicate is caught), and DELETE by the i64 key stays
/// device-native. Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_i64_and_mixed_key_elides_and_validates_on_device() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE ct (a INT8, b INT8, v INT, PRIMARY KEY (a, b))",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE mt (a INT, b INT8, v INT, PRIMARY KEY (a, b))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(3);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    // Pre-elision sentinel whose first key column exceeds 2^32 (high word non-zero) — it survives the
    // geometric device-fold rebuilds during warm-up, so a later duplicate probes it via the device fold.
    sql!("INSERT INTO ct VALUES (5000000000, 1, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_i64 {
        sql!(&format!(
            "INSERT INTO ct VALUES ({}, {}, 0)",
            6_000_000_000_i64 + i,
            i
        ))
        .unwrap();
        if engine.table_device_authoritative("ct") {
            warmed = true;
            break;
        }
    }
    assert!(
        warmed,
        "i64 compound-PK table never entered elision on a GPU box"
    );

    // Tuple uniqueness over i64 keys: distinct tuples commit; the exact tuple repeats -> 23505.
    sql!("INSERT INTO ct VALUES (5000000000, 2, 10)").unwrap(); // same a, different b -> OK
    let dup = sql!("INSERT INTO ct VALUES (5000000000, 1, 99)")
        .unwrap_err()
        .to_string();
    // DEVICE-FOLD consistency across the HIGH word: (5000000000 = 0x1_2A05F200, high word = 1) sat in a
    // device-folded rebuilt slot; the host-folded duplicate needle must match (a 1-word device fold would
    // miss the high 32 bits -> no 23505).
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "duplicate i64 compound tuple must raise 23505 (2-word fold agreement), got: {dup}"
    );
    assert!(engine.table_device_authoritative("ct"));

    // DELETE + UPDATE by the i64 compound key stay DEVICE-NATIVE (Stage 2b: the SV4b in-place
    // tombstone-locate folds the i64 fingerprint + tuple-verifies the slot, so no int4-predicate
    // de-elide). Run BOTH write ops FIRST and assert elision-retention immediately, then verify through
    // the supported full GPU projection rather than the retired host-pinned predicate path.
    sql!("INSERT INTO ct VALUES (5000000000, 3, 30)").unwrap();
    let resolve_before = engine.dml_device_resolve_hits();
    sql!("DELETE FROM ct WHERE a = 5000000000 AND b = 1").unwrap();
    assert!(
        engine.dml_device_resolve_hits() > resolve_before,
        "i64 compound DELETE must RESOLVE its target on the device"
    );
    assert!(
        engine.table_device_authoritative("ct"),
        "i64 compound DELETE must stay device-native (fingerprint tombstone-locate), not de-elide"
    );
    sql!("UPDATE ct SET v = 777 WHERE a = 5000000000 AND b = 2").unwrap();
    assert!(
        engine.table_device_authoritative("ct"),
        "i64 compound UPDATE must stay device-native (fingerprint tombstone-locate), not de-elide"
    );

    // Correctness: deleted only (5000000000,1) [b=2 and b=3 remain]; the UPDATE set v=777 on
    // exactly (5000000000,2).
    let rows = engine
        .execute_relational_select_text("SELECT a, b, v FROM ct")
        .unwrap()
        .rows
        .into_boxed();
    let mut target_rows = rows
        .into_iter()
        .filter(|row| row.first() == Some(&SqlValue::Int8(5_000_000_000)))
        .collect::<Vec<_>>();
    target_rows.sort_by_key(|row| match row.get(1) {
        Some(SqlValue::Int8(value)) => *value,
        other => panic!("unexpected compound key: {other:?}"),
    });
    assert_eq!(
        target_rows,
        vec![
            vec![
                SqlValue::Int8(5_000_000_000),
                SqlValue::Int8(2),
                SqlValue::Int4(777),
            ],
            vec![
                SqlValue::Int8(5_000_000_000),
                SqlValue::Int8(3),
                SqlValue::Int4(30),
            ],
        ],
        "the full GPU projection proves the delete and update were tuple-exact"
    );

    // MIXED int4+int8 compound key: elide + enforce uniqueness.
    sql!("INSERT INTO mt VALUES (7, 8000000000, 0)").unwrap();
    if engine
        .populate_relational_residency_snapshot("mt")
        .expect("populate mt")
        .device_memory_proof
        .is_some()
    {
        let mut mt_warmed = false;
        for i in 0..10_000_i32 {
            sql!(&format!(
                "INSERT INTO mt VALUES ({}, {}, 0)",
                100 + i,
                9_000_000_000_i64 + i as i64
            ))
            .unwrap();
            if engine.table_device_authoritative("mt") {
                mt_warmed = true;
                break;
            }
        }
        assert!(mt_warmed, "mixed compound-PK table never entered elision");
        sql!("INSERT INTO mt VALUES (7, 8000000001, 1)").unwrap(); // distinct b -> OK
        let mdup = sql!("INSERT INTO mt VALUES (7, 8000000000, 2)")
            .unwrap_err()
            .to_string();
        assert!(
            mdup.contains("duplicate key value violates unique index"),
            "duplicate mixed compound tuple must raise 23505, got: {mdup}"
        );
        assert!(engine.table_device_authoritative("mt"));
    }
}

/// COMPOUND KEYS (operational cases): a DELETE / UPDATE BY a compound key resolves its target ON THE
/// DEVICE (the SQL resolve builds the surrogate fingerprint from the key columns' Eq predicates and
/// probes the compound index; the full `filter_groups` recheck restores tuple exactness), so the table
/// STAYS ELIDED instead of de-eliding to a host rehydrate. This proves: the device resolve FIRES (the
/// `dml_device_resolve_hits` counter advances), the table stays elided across the DELETE + UPDATE, and
/// the ops are TUPLE-EXACT (deleting `(a,b1)` leaves `(a,b2)` — not first-column-only). Driverless-safe.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_delete_update_by_key_stays_device_native() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE ct (a INT, b INT, v INT, PRIMARY KEY (a, b))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! sql {
        ($s:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $s)
        };
    }
    sql!("INSERT INTO ct VALUES (1000000, 0, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    let mut warmed = false;
    for i in 0..10_000_i32 {
        sql!(&format!(
            "INSERT INTO ct VALUES ({}, {}, 0)",
            2_000_000 + i,
            i
        ))
        .unwrap();
        if engine.table_device_authoritative("ct") {
            warmed = true;
            break;
        }
    }
    assert!(
        warmed,
        "compound-PK table never entered elision on a GPU box"
    );
    // Two rows sharing the first key column a=5 but differing in b.
    sql!("INSERT INTO ct VALUES (5, 1, 100)").unwrap();
    sql!("INSERT INTO ct VALUES (5, 2, 200)").unwrap();
    assert!(engine.table_device_authoritative("ct"));

    // DELETE by the FULL compound key -> device resolve; the table must STAY ELIDED (not rehydrate).
    let resolve_before = engine.dml_device_resolve_hits();
    sql!("DELETE FROM ct WHERE a = 5 AND b = 1").unwrap();
    assert!(
        engine.dml_device_resolve_hits() > resolve_before,
        "compound DELETE must resolve its target ON THE DEVICE (counter advanced)"
    );
    assert!(
        engine.table_device_authoritative("ct"),
        "compound DELETE must not de-elide the table"
    );

    // TUPLE EXACTNESS: (5,1) is gone; (5,2) survives (a DELETE keyed on the tuple, not column a).
    let count_a5 = |engine: &Engine| -> i64 {
        let Command::Select(sel) = parse_command("SELECT COUNT(*) FROM ct WHERE a = 5").unwrap()
        else {
            unreachable!()
        };
        match engine.execute_relational_select(&sel).unwrap().rows[0][0] {
            SqlValue::Int8(n) => n,
            ref other => panic!("unexpected count value {other:?}"),
        }
    };
    assert_eq!(
        count_a5(&engine),
        1,
        "exactly one a=5 row remains after deleting (5,1)"
    );

    // UPDATE by the full compound key -> device resolve; stays elided; hits the right tuple.
    sql!("UPDATE ct SET v = 999 WHERE a = 5 AND b = 2").unwrap();
    assert!(
        engine.table_device_authoritative("ct"),
        "compound UPDATE must not de-elide the table"
    );
    let Command::Select(sel) = parse_command("SELECT v FROM ct WHERE a = 5 AND b = 2").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        engine.execute_relational_select(&sel).unwrap().rows,
        vec![vec![SqlValue::Int4(999)]],
        "compound UPDATE must set v=999 on exactly the (5,2) tuple"
    );
}

/// COMPOUND KEYS (audit regression): a compound index's per-shard device-index cache is keyed by
/// `FLAG | ordinal`. Dropping an EARLIER constraint shifts the ordinals of the following indexes, so
/// a SURVIVING cache entry could alias the shifted index (its probe would read an index built from
/// the WRONG key columns -> a missed duplicate / silent UNIQUE violation). This is closed because any
/// index-shape DDL is an "other DDL" in `residency_invalidation_scope` -> the GLOBAL residency
/// invalidation, which purges the PK device-index cache (`purge_shard_pk_index_for_table`) for every
/// table (see `index_probe_key_id`'s CACHE SAFETY note). This test drops the PK so the surviving
/// `(c,d)` UNIQUE shifts from ordinal 1 to 0 and proves it still raises 23505 on a duplicate `(c,d)`
/// tuple — i.e. the ordinal-shift invariant holds end-to-end. Self-guards on a driverless box.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_drop_constraint_shifts_ordinal_without_aliasing_the_device_index() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE ct (a INT, b INT, c INT, d INT, \
             CONSTRAINT ct_pk PRIMARY KEY (a, b), CONSTRAINT ct_cd UNIQUE (c, d))",
        )
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    engine.set_binary_wal_records_enabled(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let txn_ids = AtomicU64::new(2);
    macro_rules! insert {
        ($sql:expr) => {
            engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $sql)
        };
    }
    insert!("INSERT INTO ct VALUES (1000000, 0, 1000000, 0)").unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("ct")
        .expect("populate residency");
    if snapshot.device_memory_proof.is_none() {
        return; // self-guard: no usable GPU
    }
    // Warm into elision; both compound indexes build their device caches during the wave probes.
    let mut warmed = false;
    for i in 0..10_000_i32 {
        insert!(&format!(
            "INSERT INTO ct VALUES ({}, {}, {}, {})",
            2_000_000 + i,
            i,
            3_000_000 + i,
            i
        ))
        .unwrap();
        if engine.table_device_authoritative("ct") {
            warmed = true;
            break;
        }
    }
    assert!(warmed, "compound table never entered elision on a GPU box");
    // A sentinel row establishes the (c,d) = (100, 200) tuple.
    insert!("INSERT INTO ct VALUES (5, 5, 100, 200)").unwrap();

    // Drop the PRIMARY KEY (ordinal 0) -> the surviving `ct_cd` UNIQUE (c,d) shifts to ordinal 0.
    engine
        .execute_text(90_000, "ALTER TABLE ONLY public.ct DROP CONSTRAINT ct_pk")
        .unwrap();

    // The (c,d) uniqueness MUST still be enforced through its device index after the shift: a
    // DISTINCT (a,b) but DUPLICATE (c,d) tuple raises 23505. On the buggy (un-purged) build the
    // (c,d) probe aliased the stale (a,b) index at ordinal 0, missed the duplicate, and committed.
    let dup = insert!("INSERT INTO ct VALUES (6, 6, 100, 200)")
        .unwrap_err()
        .to_string();
    assert!(
        dup.contains("duplicate key value violates unique index"),
        "surviving compound UNIQUE must catch the duplicate after the ordinal shift, got: {dup}"
    );
    // And a genuinely new (c,d) tuple still commits (the index is live, not wedged-declining).
    insert!("INSERT INTO ct VALUES (7, 7, 101, 201)").unwrap();
}
