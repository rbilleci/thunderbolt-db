use super::*;

#[test]
fn engine_is_send_sync_for_concurrent_reads() {
    // The &self read path (P1-M3 step 3c) is only useful if the engine can be shared
    // across reader threads. Guard that it stays Send + Sync.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Engine>();
}

#[test]
fn concurrent_readers_execute_relational_select_on_shared_engine() {
    // Gate 2 (doc 14): with reads flipped to `&self`, multiple threads run
    // execute_relational_select against ONE shared engine (one published residency
    // generation) concurrently — there is no `&mut self` bottleneck. This is the
    // property the whole P1-M3 substrate exists to enable.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta'), (3, 'alpine')",
    )
    .unwrap();
    // One published residency generation, shared immutably across all readers.
    e.populate_relational_residency_snapshot("events").unwrap();
    let engine = Arc::new(e);

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
        unreachable!()
    };
    // &self call through the Arc — the baseline result every reader must reproduce.
    let expected = engine.execute_relational_select(&select).unwrap();

    const READERS: usize = 8;
    const READS_PER_THREAD: usize = 200;
    let barrier = Arc::new(std::sync::Barrier::new(READERS));
    let readers: Vec<_> = (0..READERS)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let select = select.clone();
            let barrier = Arc::clone(&barrier);
            let expected = expected.clone();
            std::thread::spawn(move || {
                barrier.wait(); // maximize real overlap of the read bodies
                for _ in 0..READS_PER_THREAD {
                    let result = engine.execute_relational_select(&select).unwrap();
                    assert_eq!(result.rows, expected.rows, "concurrent read diverged");
                }
            })
        })
        .collect();
    for reader in readers {
        reader.join().expect("reader thread panicked");
    }
    // Residency survived concurrent reads and is still valid afterward.
    assert!(engine
        .relational_residency_snapshot("events")
        .unwrap()
        .is_valid());
}

#[test]
fn residency_invalidated_error_is_classified_precisely() {
    // BUG 3 seam (classification half). The CPU-fallback decision in
    // `execute_relational_select` keys off `ExecuteError::is_residency_invalidated`. It must be
    // TRUE for exactly the GPU-probe "no retained resident device memory" tombstone error (the
    // case a concurrent committer creates by `publish(None)` mid-statement) and FALSE for every
    // other error, so a genuine device/bind error is never masked by the fallback.
    let invalidated = ExecuteError::Engine(EngineError::ApplyFailed(format!(
        "relation \"{}\" has no retained resident device memory",
        "events"
    )));
    assert!(
        invalidated.is_residency_invalidated(),
        "the exact probe tombstone message must be recognized as residency-invalidated"
    );

    // Genuine, non-fallbackable errors must NOT be misclassified.
    let real_gpu_error = ExecuteError::Engine(EngineError::ApplyFailed(
        "CUDA_ERROR_INVALID_CONTEXT launching kernel".to_string(),
    ));
    assert!(!real_gpu_error.is_residency_invalidated());
    let route_rejected = ExecuteError::Engine(EngineError::ApplyFailed(
        "resident route rejected: query shape not eligible".to_string(),
    ));
    assert!(!route_rejected.is_residency_invalidated());
    let serialization = ExecuteError::Serialization("write-write conflict".to_string());
    assert!(!serialization.is_residency_invalidated());
    let not_leader = ExecuteError::Engine(EngineError::NotLeader);
    assert!(!not_leader.is_residency_invalidated());
}

#[test]
fn execute_relational_select_cpu_pinned_matches_the_public_select() {
    // BUG 3 seam (fallback-target half). When a resident route's residency is invalidated
    // mid-statement, `execute_relational_select` re-serves the statement from
    // `execute_relational_select_cpu_pinned`. That fallback target must produce exactly the
    // result the public CPU path does (a full deterministic resident-route→fallback repro needs a
    // GPU; this asserts the seam the fallback lands on is correct CPU-only). The two are wired to
    // the same bind + pinned MVCC read, so for a non-resident table they must agree on rows,
    // columns, and access path.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    for id in 1..=5 {
        e.execute_text(
            (id + 1) as u64,
            &format!("INSERT INTO t (id, v) VALUES ({id}, {})", id * 10),
        )
        .unwrap();
    }
    for sql in [
        "SELECT id FROM t ORDER BY id",
        "SELECT COUNT(*) FROM t",
        "SELECT id, v FROM t WHERE id = 3",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!("{sql} is a SELECT")
        };
        let via_public = e.execute_relational_select(&select).unwrap();
        let via_fallback = e.execute_relational_select_cpu_pinned(&select).unwrap();
        assert_eq!(
            via_fallback.rows, via_public.rows,
            "{sql}: CPU-fallback rows must equal the public select"
        );
        assert_eq!(
            *via_fallback.columns, *via_public.columns,
            "{sql}: CPU-fallback columns must equal the public select"
        );
        assert_eq!(
            *via_fallback.access_path, *via_public.access_path,
            "{sql}: CPU-fallback access path must equal the public select"
        );
    }
}

// ----------------------------------------------------------------------------------------
// Write-half MVCC — Stage 3: per-table publish-on-commit `&self`-readable data.
// ----------------------------------------------------------------------------------------

/// Drain a sequential cursor into a `Vec` of its visible versions (Stage-3 substrate tests).
#[cfg(test)]
fn drain_cursor(mut cursor: Box<dyn gpu_db_storage::SeqScanCursor + '_>) -> Vec<TupleVersion> {
    let mut rows = Vec::new();
    while let Some(tuple) = cursor.next() {
        rows.push(tuple);
    }
    rows
}

/// Insert one `rel/people/<row_id>` row into `data`, recording its `(name, value)` value-index
/// entry — a tiny stand-in for an `apply_delta` insert, used by the Stage-3 substrate tests.
#[cfg(test)]
fn seed_people_row(
    data: &mut TableVersionData,
    tuple_id: TupleId,
    row_id: u64,
    id: i32,
    name: &str,
    commit_seq: TxnId,
) {
    let row_key = relational_row_key("people", row_id);
    let values = vec![SqlValue::Int4(id), SqlValue::Text(name.to_string())];
    data.rows
        .tuple_insert_reserved_key_with_id(
            tuple_id,
            NewTuple {
                key: row_key.clone(),
                value: encode_relational_row(&values),
            },
            commit_seq,
        )
        .unwrap();
    let index_key = ColumnValueKey {
        column: "name".to_string(),
        value: relational_index_value(&SqlValue::Text(name.to_string())),
    };
    let mut slot = data
        .value_index
        .get(&index_key)
        .cloned()
        .unwrap_or_default();
    std::sync::Arc::make_mut(&mut slot).push(row_key);
    data.value_index.insert(index_key, slot);
}

/// Stage 3 reader-stability: a reader that has `load()`ed a table's generation keeps seeing the
/// SAME rows AND value-index even as the (serialized) writer publishes a new generation that
/// adds rows + value-index entries. New loads see the new generation. This is the per-table
/// `SnapshotCell` discipline applied to the MVCC data (mirrors the residency reader-stability
/// guarantee, now for rows + value-index together in one `Arc<TableVersionData>`).
#[test]
fn stage3_reader_sees_stable_rows_and_value_index_while_writer_publishes() {
    let mvcc = MvccData::new();
    // Generation 1: one row, one value-index entry.
    mvcc.with_table_mut("people", |data| {
        seed_people_row(data, 1, 1, 1, "Ada", 1);
    });
    let vis = StorageVisibility { read_txn_id: 100 };

    // A reader pins generation 1.
    let reader = mvcc.load_table("people").expect("table published");
    let reader_rows_before = drain_cursor(reader.get().rows.seq_scan_open(vis).unwrap());
    assert_eq!(reader_rows_before.len(), 1);
    let ada_index = reader.get().index_keys(
        "name",
        &relational_index_value(&SqlValue::Text("Ada".to_string())),
    );
    assert_eq!(ada_index.len(), 1);
    // No value-index entry for the not-yet-inserted "Linus".
    assert!(reader
        .get()
        .index_keys(
            "name",
            &relational_index_value(&SqlValue::Text("Linus".to_string()))
        )
        .is_empty());

    // The serialized writer publishes generation 2: a second row + its value-index entry.
    mvcc.with_table_mut("people", |data| {
        seed_people_row(data, 2, 2, 2, "Linus", 2);
    });

    // The reader STILL sees exactly generation 1's rows and value-index — stable snapshot.
    let reader_rows_after = drain_cursor(reader.get().rows.seq_scan_open(vis).unwrap());
    assert_eq!(
        reader_rows_after, reader_rows_before,
        "reader's pinned generation changed under a concurrent publish"
    );
    assert!(
        reader
            .get()
            .index_keys(
                "name",
                &relational_index_value(&SqlValue::Text("Linus".to_string()))
            )
            .is_empty(),
        "reader's pinned value-index gained an entry from a later generation"
    );

    // A FRESH load sees generation 2: both rows + both value-index entries.
    let newer = mvcc.load_table("people").expect("table published");
    assert!(newer.generation() > reader.generation());
    assert_eq!(
        drain_cursor(newer.get().rows.seq_scan_open(vis).unwrap()).len(),
        2
    );
    assert_eq!(
        newer
            .get()
            .index_keys(
                "name",
                &relational_index_value(&SqlValue::Text("Linus".to_string()))
            )
            .len(),
        1
    );
}

/// Stage 3 epoch reclamation: an old `Arc<TableVersionData>` generation is freed only after its
/// last reader handle drains — never under an in-flight reader (mirrors the residency / snapshot
/// reclamation tests). We observe this through the generation's reference count.
#[test]
fn stage3_old_table_generation_retired_only_after_last_reader_drains() {
    let mvcc = MvccData::new();
    mvcc.with_table_mut("people", |data| seed_people_row(data, 1, 1, 1, "Ada", 1));

    // A reader pins generation 1.
    let reader = mvcc.load_table("people").expect("table published");
    let g1 = reader.generation();
    // The payload Arc is shared by the reader handle AND the cell's current slot.
    assert_eq!(reader.reader_refcount(), 2, "g1 held by reader + cell slot");

    // Writer publishes generations 2 and 3. g1 is no longer current, but the reader still holds
    // it, so its generation must NOT be reclaimed — only the reader references it now.
    mvcc.with_table_mut("people", |data| seed_people_row(data, 2, 2, 2, "Linus", 2));
    mvcc.with_table_mut("people", |data| seed_people_row(data, 3, 3, 3, "Grace", 3));
    assert_eq!(
        reader.reader_refcount(),
        1,
        "g1 must stay alive (refcount 1 = the in-flight reader) until that reader drains"
    );
    assert_eq!(reader.generation(), g1, "reader still pinned to g1");
    // The reader's rows are still exactly g1's single row.
    assert_eq!(
        drain_cursor(
            reader
                .get()
                .rows
                .seq_scan_open(StorageVisibility { read_txn_id: 100 })
                .unwrap()
        )
        .len(),
        1
    );

    // The current generation is g3 with three rows, independent of the pinned g1.
    let current = mvcc.load_table("people").unwrap();
    assert!(current.generation() > g1);
    assert_eq!(
        drain_cursor(
            current
                .get()
                .rows
                .seq_scan_open(StorageVisibility { read_txn_id: 100 })
                .unwrap()
        )
        .len(),
        3
    );
}

/// Stage 3 read-perf sanity: the equality fast-path still reads the versioned value-index (a
/// per-table map lookup), NOT a version-chain scan. On a many-row table where only a handful
/// match, the chosen access path is `EqualityIndex` with `matched_keys` ≪ the row count, and a
/// direct value-index lookup on the loaded generation returns exactly those keys — i.e. the
/// per-table refactor kept the fast-path index-targeted (the whole reason for per-table cells).
#[test]
fn stage3_resident_equality_read_still_uses_value_index_fast_path() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    // 200 rows; only 2 carry name='Ada'. A version-chain scan would touch all 200; the
    // value-index route touches just the 2 matching keys.
    let mut values = Vec::new();
    for id in 0..200 {
        let name = if id == 7 || id == 142 { "Ada" } else { "Other" };
        values.push(format!("({id}, '{name}')"));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO people (id, name) VALUES {}", values.join(", ")),
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Ada'").unwrap()
    else {
        unreachable!()
    };
    let result = e.execute_relational_select(&select).unwrap();
    // The equality predicate resolved through the per-table versioned value-index, touching
    // only the 2 matching keys (index-targeted, not a 200-row scan).
    assert!(
        matches!(
            *result.access_path,
            RelationalAccessPath::EqualityIndex {
                matched_keys: 2,
                ..
            }
        ),
        "equality fast-path regressed off the value-index: {:?}",
        result.access_path
    );
    assert_eq!(
        result.rows.len(),
        2,
        "both 'Ada' rows returned, no scan miss"
    );

    // Directly: the loaded generation's value-index returns exactly the 2 'Ada' row keys
    // (an O(log) map lookup), and far fewer than the 200 stored rows — proving the read hit
    // the versioned value-index, not a chain scan.
    let handle = e
        .read_state
        .mvcc
        .load_table("people")
        .expect("table published");
    let ada_keys = handle.get().index_keys(
        "name",
        &relational_index_value(&SqlValue::Text("Ada".to_string())),
    );
    assert_eq!(ada_keys.len(), 2);
    let total_rows = drain_cursor(
        handle
            .get()
            .rows
            .seq_scan_open(StorageVisibility {
                read_txn_id: e.visible_up_to(),
            })
            .unwrap(),
    )
    .len();
    assert_eq!(total_rows, 200);
    assert!(
        ada_keys.len() * 10 < total_rows,
        "value-index lookup must be selective (index-targeted), not a full scan"
    );
}
