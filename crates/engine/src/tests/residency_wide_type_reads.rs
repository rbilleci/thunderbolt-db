/// TYPE-COVERAGE #14 (text): a TEXT value column is device-authoritative (elided), ROLLS OVER to
/// multiple DENSE shards (rollover-only — variable-length has no headroom), and READS correctly via
/// the cross-shard gather (each shard's bytes blob byte-copies at a running blob_base; a per-element
/// offset-rebase kernel adds that blob_base to the shard's offsets). VARIED-LENGTH strings incl EMPTY
/// exercise the offset math across shard + blob boundaries. Differential vs the CPU host oracle.
/// Sabotage: dropping the rebase (or the blob segment) diverges the differential.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn text_column_elides_appends_and_reads_multishard() {
    let run = |device: bool| -> (bool, usize, Vec<Vec<SqlValue>>) {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(device);
        e.set_host_install_elision_enabled(device);
        e.set_constrained_elision_enabled(device);
        e.set_device_write_locate_enabled(device);
        e.set_device_write_locate_wave_batch_enabled(device);
        e.set_shard_size_target(64); // force MULTIPLE shards (rollover) -> exercise the text gather
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, s TEXT)")
            .unwrap();
        for (seq, chunk) in (2_u64..).zip(0..2_i64) {
            // s = a varied-length string; k % 7 == 0 -> a GENUINELY EMPTY string (a zero-length blob
            // span, incl. the FIRST row k=0 and spans landing at shard boundaries) — exercises the
            // offset math where consecutive offsets are equal.
            let vals: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| {
                    if k % 7 == 0 {
                        format!("({k}, '')")
                    } else {
                        format!("({k}, 'v{k}-{}')", "x".repeat((k % 7) as usize))
                    }
                })
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO t (id, s) VALUES {}", vals.join(",")),
            )
            .unwrap();
        }
        let elided = e.table_install_elided("t");
        let shard_count = e.resident_shard_count("t");
        let mut rows = e
            .execute_relational_select_text("SELECT id, s FROM t")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        // A FILTERED text projection is NOT a served on-device shape -> CPU-pinned path -> (elided)
        // rehydrate device->host FIRST (the text arm of gather_resident_table_rows_from_device).
        // id=3 -> "v3-xxx" (3 % 7 == 3 -> 3 'x'). Without the text rehydration arm this hard-errors.
        let filtered = e
            .execute_relational_select_text("SELECT id, s FROM t WHERE id = 3")
            .unwrap()
            .rows
            .into_boxed();
        assert_eq!(filtered.len(), 1, "filtered text read returns exactly id=3");
        assert!(
            matches!(filtered[0].get(1), Some(SqlValue::Text(t)) if t == "v3-xxx"),
            "the rehydrated text value is exact, got {:?}",
            filtered[0].get(1)
        );
        (elided, shard_count, rows)
    };
    let (on_elided, on_shards, on_rows) = run(true);
    let (_off_elided, _off_shards, off_rows) = run(false);
    if on_rows.is_empty() {
        return; // driverless box
    }
    assert!(
        on_elided,
        "a TEXT-bearing table must be device-authoritative (elided)"
    );
    assert!(
        on_shards >= 2,
        "the table must roll over to MULTIPLE dense shards (exercise the text gather), saw {on_shards}"
    );
    assert_eq!(
        on_rows, off_rows,
        "elided multi-shard text read == CPU host oracle (blob concat + offset rebase are correct)"
    );
    // A non-empty string materializes with its exact bytes.
    assert!(
        on_rows
            .iter()
            .any(|r| matches!(r.get(1), Some(SqlValue::Text(t)) if t.contains('x'))),
        "a varied-length string materializes with its bytes"
    );
}

/// TYPE-COVERAGE #14 (numeric): a NUMERIC value column rides the device-authoritative / elided
/// fast path — the table ELIDES, INSERTs append device-authoritatively into the b128 (16-byte)
/// section, the table rolls over to MULTIPLE shards, and reads over the numeric column match the
/// CPU (host) oracle byte-for-byte — proving the correctness-critical recompaction GATHER of the
/// b128 section into the unified multi-shard buffer. Sabotage: dropping the numeric gather
/// segment (or the shard/unified descriptor's numeric labels) reads garbage -> the differential
/// diverges.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn numeric_column_elides_appends_and_reads_multishard() {
    let run = |device: bool| -> (bool, usize, Vec<Vec<SqlValue>>) {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(device);
        e.set_host_install_elision_enabled(device);
        e.set_constrained_elision_enabled(device);
        e.set_device_write_locate_enabled(device);
        e.set_device_write_locate_wave_batch_enabled(device);
        e.set_shard_size_target(64); // force MULTIPLE shards (rollover) -> exercise the gather
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, amt NUMERIC(20,4))")
            .unwrap();
        for (seq, chunk) in (2_u64..).zip(0..2_i64) {
            // amt = a distinct 4-scale decimal per row (id.frac) so a wrong gather is visible.
            let vals: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| format!("({k}, {k}.{:04})", (k * 7) % 10000))
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO t (id, amt) VALUES {}", vals.join(",")),
            )
            .unwrap();
        }
        let elided = e.table_install_elided("t");
        let shard_count = e.resident_shard_count("t");
        let mut rows = e
            .execute_relational_select_text("SELECT id, amt FROM t")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        (elided, shard_count, rows)
    };
    let (on_elided, on_shards, on_rows) = run(true);
    let (_off_elided, _off_shards, off_rows) = run(false);
    if on_rows.is_empty() {
        return; // driverless box
    }
    assert!(
        on_elided,
        "a NUMERIC-bearing table must be device-authoritative (elided)"
    );
    assert!(
        on_shards >= 2,
        "the table must roll over to MULTIPLE shards (exercise the recompaction gather), saw {on_shards}"
    );
    assert_eq!(
        on_rows, off_rows,
        "elided multi-shard numeric read == CPU host oracle (the b128 gather is correct)"
    );
    // Spot-check a numeric value is not a zeroed/garbage placeholder.
    assert!(
        on_rows
            .iter()
            .any(|r| matches!(r.get(1), Some(SqlValue::Numeric(_)))),
        "amt materializes as a real NUMERIC (not dropped/zeroed): {:?}",
        on_rows.first()
    );
}

/// TYPE-COVERAGE #14 (bool): a BOOLEAN value column is device-authoritative (elided), APPENDS in
/// place (the device atomicOr bitmap set-range op writes each appended row's bit into the pre-zeroed
/// headroom), ROLLS OVER to multiple shards, and READS correctly via the cross-shard bitmap gather
/// (32-row-aligned byte-copy of each shard's live words). The differential vs the CPU host oracle
/// proves every bit lands right across shard + word boundaries (200 rows, shard_size 64 => ~4 shards,
/// so a word-crossing true/false spread must survive both the append op and the recompaction). Both
/// truth values materialize as SqlValue::Bool. Sabotage: dropping the bool gather segment (or the
/// atomicOr op) diverges the differential; a non-32-aligned shard makes the gather DECLINE, not lie.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn bool_column_elides_appends_and_reads_multishard() {
    let run = |device: bool| -> (bool, usize, Vec<Vec<SqlValue>>) {
        let e = Engine::new_local();
        e.set_auto_admit_on_commit(device);
        e.set_host_install_elision_enabled(device);
        e.set_constrained_elision_enabled(device);
        e.set_device_write_locate_enabled(device);
        e.set_device_write_locate_wave_batch_enabled(device);
        e.set_shard_size_target(64); // force MULTIPLE shards (rollover) -> exercise the bitmap gather
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, flag BOOLEAN)")
            .unwrap();
        for (seq, chunk) in (2_u64..).zip(0..2_i64) {
            // flag = (id % 3 == 0): a word-crossing true/false spread so a wrong bitmap offset, a
            // mis-packed word, or a mis-aligned cross-shard copy is visible in the differential.
            let vals: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                .map(|k| format!("({k}, {})", if k % 3 == 0 { "true" } else { "false" }))
                .collect();
            e.execute_text(
                seq,
                &format!("INSERT INTO t (id, flag) VALUES {}", vals.join(",")),
            )
            .unwrap();
        }
        let elided = e.table_install_elided("t");
        let shard_count = e.resident_shard_count("t");
        let mut rows = e
            .execute_relational_select_text("SELECT id, flag FROM t")
            .unwrap()
            .rows
            .into_boxed();
        rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        // An ORDER BY over a bool-bearing elided table must ALSO route on-device (the bool column
        // is carried through the sort's unified source, not sent to the CPU pinned path where it
        // would rehydrate-decline). Same rows, so the sorted-by-debug comparison folds it in.
        let ordered = e
            .execute_relational_select_text("SELECT id, flag FROM t ORDER BY id")
            .unwrap()
            .rows
            .into_boxed();
        assert_eq!(
            ordered.len(),
            rows.len(),
            "ORDER BY over the bool table serves the same row set on-device"
        );
        // A FILTERED bool projection is NOT a served on-device shape -> it falls to the CPU-pinned
        // path, which for an ELIDED table rehydrates device->host FIRST. Without bool in the
        // rehydration gather this hard-errors ("device-authoritative invariant broken"); with it,
        // the row reads correctly. Row id=3 is true (3 % 3 == 0), id=5 is false.
        let filtered = e
            .execute_relational_select_text("SELECT id, flag FROM t WHERE id = 3")
            .unwrap()
            .rows
            .into_boxed();
        assert_eq!(
            filtered.len(),
            1,
            "a filtered bool projection rehydrates + reads (no device-authoritative hard-error)"
        );
        assert!(
            matches!(filtered[0].get(1), Some(SqlValue::Bool(true))),
            "the rehydrated bool value is correct (id=3 -> true), got {:?}",
            filtered[0].get(1)
        );
        (elided, shard_count, rows)
    };
    let (on_elided, on_shards, on_rows) = run(true);
    let (_off_elided, _off_shards, off_rows) = run(false);
    if on_rows.is_empty() {
        return; // driverless box
    }
    assert!(
        on_elided,
        "a BOOLEAN-bearing table must be device-authoritative (elided)"
    );
    assert!(
        on_shards >= 2,
        "the table must roll over to MULTIPLE shards (exercise the bitmap gather), saw {on_shards}"
    );
    assert_eq!(
        on_rows, off_rows,
        "elided multi-shard bool read == CPU host oracle (the 1-bit/row gather is correct)"
    );
    // Both truth values must materialize as real SqlValue::Bool (not dropped / all-zeroed).
    assert!(
        on_rows
            .iter()
            .any(|r| matches!(r.get(1), Some(SqlValue::Bool(true)))),
        "at least one TRUE flag materializes"
    );
    assert!(
        on_rows
            .iter()
            .any(|r| matches!(r.get(1), Some(SqlValue::Bool(false)))),
        "at least one FALSE flag materializes"
    );
}

/// TYPE-COVERAGE #14: the DEVICE->HOST rehydration gather materializes NUMERIC + UUID (both the
/// 16-byte b128 section) AND BIGINT (the i64 section) — so a read shape the on-device routes cannot
/// serve (here a FILTERED projection of the value column) falls to the CPU-pinned path and
/// rehydrates device->host correctly instead of hard-erroring ("device-authoritative invariant
/// broken"). Differential vs the CPU host oracle over the filtered read, one case per type.
/// Sabotage: declining any of these types in `gather_resident_table_rows_from_device` re-errors.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn b128_and_bigint_filtered_rehydration_reads_from_device() {
    // (value-column DDL type, per-id value SQL). `id` is the INT PK; `val` is the rehydrated column.
    type RehydrationCase = (&'static str, fn(i64) -> String);
    let cases: [RehydrationCase; 6] = [
        ("NUMERIC(20,4)", |k| format!("{k}.{:04}", (k * 7) % 10000)),
        // A NEGATIVE mantissa (i128 high bit): exercises from_le_bytes sign-correctness.
        ("NUMERIC(20,4)", |k| format!("-{k}.{:04}", (k * 7) % 10000)),
        // scale 0 + negative: the mantissa IS the integer, no fractional digits.
        ("NUMERIC(12,0)", |k| format!("{}", k * 7 - 500)),
        ("UUID", |k| {
            format!("'{:08x}-0000-0000-0000-000000000000'", k as u32)
        }),
        ("BIGINT", |k| format!("{}", k * 1_000_000_007)),
        // A NEGATIVE bigint (i64 sign bit) through the two-halves reassembly.
        ("BIGINT", |k| format!("{}", -k * 1_000_000_007 - 1)),
    ];
    for (ty, val_fn) in cases {
        let run = |device: bool| -> (bool, Vec<Vec<SqlValue>>) {
            let e = Engine::new_local();
            e.set_auto_admit_on_commit(device);
            e.set_host_install_elision_enabled(device);
            e.set_constrained_elision_enabled(device);
            e.set_device_write_locate_enabled(device);
            e.set_device_write_locate_wave_batch_enabled(device);
            e.set_shard_size_target(64); // multi-shard -> the gather spans shards
            e.execute_text(1, &format!("CREATE TABLE t (id INT PRIMARY KEY, val {ty})"))
                .unwrap();
            for (seq, chunk) in (2_u64..).zip(0..2_i64) {
                let vals: Vec<String> = (chunk * 100..(chunk + 1) * 100)
                    .map(|k| format!("({k}, {})", val_fn(k)))
                    .collect();
                e.execute_text(
                    seq,
                    &format!("INSERT INTO t (id, val) VALUES {}", vals.join(",")),
                )
                .unwrap();
            }
            let elided = e.table_install_elided("t");
            // A FILTERED projection of the value column is NOT a served on-device shape -> CPU-pinned
            // path -> (elided) rehydrate device->host FIRST. id=3 lands in shard 0.
            let rows = e
                .execute_relational_select_text("SELECT id, val FROM t WHERE id = 3")
                .unwrap()
                .rows
                .into_boxed();
            (elided, rows)
        };
        let (on_elided, on_rows) = run(true);
        let (_off_elided, off_rows) = run(false);
        if on_rows.is_empty() {
            return; // driverless box
        }
        assert!(
            on_elided,
            "{ty}: the table must be device-authoritative (elided)"
        );
        assert_eq!(
            on_rows.len(),
            1,
            "{ty}: the filtered read returns exactly id=3"
        );
        assert_eq!(
            on_rows, off_rows,
            "{ty}: elided filtered read (rehydrated device->host) == CPU host oracle"
        );
    }
}

/// R-ver PART 2: GROUP BY / DISTINCT / ORDER BY over a VERSIONED elided sharded table must
/// HIDE tombstoned rows (they used to hard-REFUSE — "SV3b not wired through the reshaping
/// sub-bridges" — because those bridges dropped `visibility`). Now the visibility is threaded
/// into the survivor `indices` BEFORE group/sort/dedup. Fully tombstoning group 0 makes it
/// vanish from the grouped counts AND the distinct set; the ordered scan starts past it.
/// Sabotage: reverting the `visibility` threading either re-errors or leaks group 0's 40 rows.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn grouped_ordered_distinct_over_versioned_elided_hides_tombstones() {
    let e = Engine::new_local();
    e.set_auto_admit_on_commit(true);
    e.set_host_install_elision_enabled(true);
    e.set_constrained_elision_enabled(true);
    e.set_device_write_locate_enabled(true);
    e.set_device_write_locate_wave_batch_enabled(true);
    e.set_shard_size_target(64);
    e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, g INT)")
        .unwrap();
    let mut seq = 2u64;
    for chunk in 0..2_i64 {
        // g = id / 40 -> ids 0..199 form 5 groups (0..4) of 40 rows each.
        let vals: Vec<String> = (chunk * 100..(chunk + 1) * 100)
            .map(|k| format!("({k},{})", k / 40))
            .collect();
        e.execute_text(
            seq,
            &format!("INSERT INTO t (id, g) VALUES {}", vals.join(",")),
        )
        .unwrap();
        seq += 1;
    }
    if !e.table_install_elided("t") {
        return;
    }
    // Fully tombstone GROUP 0 (ids 0..39) -> versioned shards; g=0 has NO visible rows.
    e.execute_text(seq, "DELETE FROM t WHERE id < 40").unwrap();

    // GROUP BY: group 0 must be ABSENT; each surviving group counts 40; 160 rows total.
    let grouped = e
        .execute_relational_select_text("SELECT g, COUNT(*) FROM t GROUP BY g")
        .unwrap()
        .rows
        .into_boxed();
    let mut counts: std::collections::BTreeMap<i32, i64> = Default::default();
    for row in &grouped {
        if let (Some(&SqlValue::Int4(g)), Some(&SqlValue::Int8(c))) = (row.first(), row.get(1)) {
            counts.insert(g, c);
        }
    }
    assert!(
        !counts.contains_key(&0),
        "the fully-tombstoned group 0 must NOT appear in GROUP BY: {counts:?}"
    );
    assert_eq!(
        counts.get(&1),
        Some(&40),
        "surviving group counts are visible-only"
    );
    assert_eq!(
        counts.values().sum::<i64>(),
        160,
        "160 visible rows across groups 1..4"
    );

    // DISTINCT: {1,2,3,4} — group 0 gone.
    let distinct = e
        .execute_relational_select_text("SELECT DISTINCT g FROM t")
        .unwrap()
        .rows
        .into_boxed();
    let mut gs: Vec<i32> = distinct
        .iter()
        .filter_map(|r| match r.first() {
            Some(&SqlValue::Int4(g)) => Some(g),
            _ => None,
        })
        .collect();
    gs.sort_unstable();
    assert_eq!(
        gs,
        vec![1, 2, 3, 4],
        "DISTINCT g excludes the fully-tombstoned group 0"
    );

    // ORDER BY: the ordered scan starts at id=40 (0..39 tombstoned), 160 visible rows.
    let ordered = e
        .execute_relational_select_text("SELECT id FROM t WHERE id >= 0 ORDER BY id LIMIT 300")
        .unwrap()
        .rows
        .into_boxed();
    assert_eq!(ordered.len(), 160, "ORDER BY returns the 160 visible rows");
    assert_eq!(
        ordered.first().and_then(|r| r.first()),
        Some(&SqlValue::Int4(40)),
        "first ordered id is 40 — ids 0..39 are hidden, not leaked"
    );
    assert!(
        e.table_install_elided("t"),
        "grouped/distinct/ordered reads over a versioned table stay elided (routed on-device)"
    );
}
