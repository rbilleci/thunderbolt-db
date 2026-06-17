use super::*;
mod sql_dml;
mod sql_catalog;
mod recovery;
mod write_half;
mod common;
use common::*;
mod write_set;

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
            via_fallback.columns, via_public.columns,
            "{sql}: CPU-fallback columns must equal the public select"
        );
        assert_eq!(
            via_fallback.access_path, via_public.access_path,
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
            result.access_path,
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

use std::sync::atomic::{AtomicU64, Ordering};

use gpu_db_execution::DeviceTarget;
use gpu_db_metrics::GpuParityIssue;
use gpu_db_observability::InMemoryTelemetrySink;

static NEXT_TEST_WAL_PATH_ID: AtomicU64 = AtomicU64::new(1);

fn test_wal_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "gpu-db-engine-{name}-{}-{}.segment",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn planner_targets_mutations_to_gpu() {
    let e = Engine::new_local();
    let plan = e.plan_text("SET a=1").unwrap();

    assert_eq!(plan.nodes().len(), 1);
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(0));
}

#[test]
fn planner_targets_get_to_cpu_fallback_path() {
    let e = Engine::new_local();
    let plan = e.plan_text("GET a").unwrap();

    assert_eq!(plan.nodes().len(), 1);
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Cpu);
}

#[test]
fn planner_config_can_override_default_gpu_target() {
    let e = Engine::with_planner_config(PlannerConfig { default_gpu_id: 3 });
    let plan = e.plan_text("SET a=1").unwrap();

    assert_eq!(plan.nodes().len(), 1);
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(3));
}

#[test]
fn wal_before_visibility_holds() {
    let e = Engine::new_local();
    let t = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
    assert!(e.wal_flushed_count() >= 1);
    assert!(e.visible_up_to() >= t.index);
    assert!(e.applied_len() >= 1);
}

#[test]
fn commit_indices_monotonic() {
    let e = Engine::new_local();
    let a = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
    let b = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
    assert!(b.index > a.index);
    assert!(e.visible_up_to() >= b.index);
}

#[test]
fn execute_set_updates_state_machine() {
    let e = Engine::new_local();
    e.execute_text(1, "SET balance=100").unwrap();
    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
}

#[test]
fn execute_set_accepts_session_and_local_scope_aliases() {
    let e = Engine::new_local();
    e.execute_text(1, "SET SESSION balance=100").unwrap();
    e.execute_text(2, "SET LOCAL balance TO 101").unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("101"));
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn execute_del_removes_existing_key() {
    let e = Engine::new_local();
    e.execute_text(1, "SET balance=100").unwrap();
    e.execute_text(2, "DEL balance").unwrap();

    assert_eq!(e.get("balance"), None);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn execute_delete_alias_removes_existing_key() {
    let e = Engine::new_local();
    e.execute_text(1, "SET balance=100").unwrap();
    e.execute_text(2, "DELETE balance").unwrap();

    assert_eq!(e.get("balance"), None);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn execute_read_text_get_returns_current_value_without_committing() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET balance=100").unwrap();

    let value = e.execute_read_text("GET balance").unwrap();
    assert_eq!(value.as_deref(), Some("100"));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 1);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);
    assert_eq!(
        e.metrics().last_fallback_reason(),
        Some(FallbackReason::NotGpuEligible)
    );
}

#[test]
fn execute_read_text_get_missing_key_does_not_track_d2h_bytes() {
    let mut e = Engine::new_local();
    let value = e.execute_read_text("GET absent").unwrap();

    assert_eq!(value, None);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn execute_read_text_rejects_non_read_commands() {
    let mut e = Engine::new_local();

    let begin_err = e.execute_read_text("BEGIN").unwrap_err();
    assert!(matches!(begin_err, ExecuteError::NonReadCommand("BEGIN")));

    let set_err = e.execute_read_text("SET balance=100").unwrap_err();
    assert!(matches!(set_err, ExecuteError::NonReadCommand("SET")));

    let reset_err = e.execute_read_text("RESET ALL").unwrap_err();
    assert!(matches!(
        reset_err,
        ExecuteError::NonReadCommand("RESET ALL")
    ));

    let discard_err = e.execute_read_text("DISCARD TEMP").unwrap_err();
    assert!(matches!(
        discard_err,
        ExecuteError::NonReadCommand("RESET ALL")
    ));

    let del_err = e.execute_read_text("DELETE balance").unwrap_err();
    assert!(matches!(
        del_err,
        ExecuteError::NonReadCommand("DEL/DELETE")
    ));

    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_read_text_rejects_get_when_not_leader() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET balance=100").unwrap();
    e.become_follower(2);

    let err = e.execute_read_text("GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn execute_text_get_rejects_when_not_leader() {
    let mut e = Engine::new_local();
    e.become_follower(2);

    let err = e.execute_text(1, "GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_text_get_rejects_when_candidate() {
    let mut e = Engine::new_local();
    e.become_candidate(2);

    let err = e.execute_text(1, "GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn execute_text_get_tracks_d2h_bytes_for_hits_only() {
    let e = Engine::new_local();
    e.execute_text(1, "SET balance=100").unwrap();

    e.execute_text(2, "GET balance").unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);

    e.execute_text(3, "GET missing").unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);
}

#[test]
fn execute_read_text_rejects_get_when_candidate() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET balance=100").unwrap();
    e.become_candidate(2);

    let err = e.execute_read_text("GET balance").unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
}

#[test]
fn batching_flushes_on_count_and_updates_metric() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.enqueue_set_text(2, "SET b=2", t0).unwrap();
    assert_eq!(e.get("a").as_deref(), Some("1"));
    assert_eq!(e.get("b").as_deref(), Some("2"));
    assert_eq!(e.metrics().snapshot().batch_flush_count, 1);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
    assert_eq!(
        e.metrics().last_batch_flush_reason(),
        Some(BatchFlushReason::Count)
    );
    assert_eq!(e.metrics().snapshot().batch_wait_samples, 2);
    assert_eq!(e.metrics().snapshot().batch_wait_total_ms, 0);
    assert_eq!(e.metrics().last_batch_wait_ms(), Some(0));
    assert_eq!(
        e.metrics().snapshot().h2d_bytes_total,
        "SET a=1".len() as u64 + "SET b=2".len() as u64
    );
    assert_eq!(e.metrics().snapshot().kernel_exec_samples, 2);
    assert_eq!(e.metrics().snapshot().kernel_exec_total_ms, 2);
    assert_eq!(e.metrics().last_kernel_exec_ms(), Some(1));
    assert_eq!(e.metrics().snapshot().kernel_occupancy_samples, 2);
    assert_eq!(
        e.metrics().snapshot().kernel_occupancy_total_permyriad,
        6400
    );
    assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(3200));
    assert_eq!(e.metrics().snapshot().pending_batch_peak, 2);
    assert_eq!(e.metrics().last_pending_batch_len(), Some(0));
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn batching_flushes_on_time() {
    let mut e = Engine::with_batching(10, Duration::from_millis(2));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=7", t0).unwrap();
    assert!(e.has_pending_batch());
    assert_eq!(e.pending_batch_len(), 1);
    e.tick_batching(t0 + Duration::from_millis(3)).unwrap();
    assert_eq!(e.get("a").as_deref(), Some("7"));
    assert!(!e.has_pending_batch());
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 1);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Time), 1);
    assert_eq!(e.metrics().snapshot().batch_wait_samples, 1);
    assert_eq!(e.metrics().snapshot().batch_wait_total_ms, 3);
    assert_eq!(e.metrics().last_batch_wait_ms(), Some(3));
}

#[test]
fn batching_kernel_occupancy_caps_at_full_utilization() {
    let mut e = Engine::with_batching(1, Duration::from_secs(999));
    let t0 = Instant::now();
    let payload = format!("SET a={}", "x".repeat(512));

    e.enqueue_set_text(1, &payload, t0).unwrap();

    assert_eq!(e.metrics().snapshot().kernel_occupancy_samples, 1);
    assert_eq!(e.metrics().last_kernel_occupancy_permyriad(), Some(10_000));
    assert_eq!(
        e.metrics().snapshot().kernel_occupancy_total_permyriad,
        10_000
    );
}

#[test]
fn admin_flush_tracks_reason() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=9", t0).unwrap();
    assert_eq!(e.pending_batch_len(), 1);
    e.flush_admin().unwrap();

    assert_eq!(e.get("a").as_deref(), Some("9"));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
}

#[test]
fn admin_flush_without_pending_queue_is_noop() {
    let e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();

    e.flush_admin().unwrap();

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.pending_batch_oldest_age(t0), None);
    assert_eq!(e.pending_batch_time_until_deadline(t0), None);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
    assert_eq!(e.metrics().last_batch_flush_reason(), None);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn batching_config_reflects_engine_settings() {
    let e = Engine::with_batching(7, Duration::from_millis(42));
    assert_eq!(e.batching_config(), (7, Duration::from_millis(42)));
}

#[test]
fn batching_and_planner_config_can_be_combined() {
    let e = Engine::with_batching_and_planner_config(
        3,
        Duration::from_millis(9),
        PlannerConfig { default_gpu_id: 5 },
    );

    assert_eq!(e.batching_config(), (3, Duration::from_millis(9)));
    let plan = e.plan_text("SET a=1").unwrap();
    assert_eq!(plan.nodes()[0].op.target, DeviceTarget::Gpu(5));
}

#[test]
fn pending_batch_deadline_counts_down_and_clears_after_flush() {
    let mut e = Engine::with_batching(10, Duration::from_millis(10));
    let t0 = Instant::now();

    assert_eq!(e.pending_batch_time_until_deadline(t0), None);

    e.enqueue_set_text(1, "SET a=9", t0).unwrap();
    assert_eq!(
        e.pending_batch_time_until_deadline(t0 + Duration::from_millis(4)),
        Some(Duration::from_millis(6))
    );
    assert_eq!(
        e.pending_batch_time_until_deadline(t0 + Duration::from_millis(12)),
        Some(Duration::ZERO)
    );

    e.flush_admin().unwrap();
    assert_eq!(
        e.pending_batch_time_until_deadline(t0 + Duration::from_millis(13)),
        None
    );
}

#[test]
fn pending_batch_oldest_age_tracks_then_clears_after_flush() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=9", t0).unwrap();

    let age = e
        .pending_batch_oldest_age(t0 + Duration::from_millis(5))
        .expect("pending batch age should exist");
    assert!(age >= Duration::from_millis(5));

    e.flush_admin().unwrap();
    assert_eq!(
        e.pending_batch_oldest_age(t0 + Duration::from_millis(6)),
        None
    );
}

#[test]
fn batching_can_apply_set_then_del_in_order() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.enqueue_set_text(2, "DEL a", t0).unwrap();

    assert_eq!(e.get("a"), None);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 1);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn deterministic_replay_matches_between_immediate_and_batched_mutation_paths() {
    let trace = [
        (1, "SET acct_a=10"),
        (2, "SET acct_b=25"),
        (3, "DEL acct_a"),
        (4, "SET acct_c=77"),
        (5, "DELETE acct_b"),
        (6, "SET acct_a=99"),
    ];

    let immediate = Engine::new_local();
    for (txn_id, cmd) in trace {
        immediate.execute_text(txn_id, cmd).unwrap();
    }

    let mut batched = Engine::with_batching(64, Duration::from_secs(999));
    let t0 = Instant::now();
    for (txn_id, cmd) in trace {
        batched.enqueue_set_text(txn_id, cmd, t0).unwrap();
    }
    batched.flush_admin().unwrap();

    assert_eq!(
        immediate.commit_state().sm.applied,
        batched.commit_state().sm.applied
    );
    assert_eq!(immediate.commit_state().sm.kv, batched.commit_state().sm.kv);
    assert_eq!(immediate.visible_up_to(), batched.visible_up_to());
    assert_eq!(
        immediate.visible_state_fingerprint(),
        batched.visible_state_fingerprint()
    );
    assert_eq!(immediate.wal_flushed_count(), trace.len());
    assert_eq!(batched.wal_flushed_count(), trace.len());
}

#[test]
fn visible_state_fingerprint_changes_with_visible_kv_state() {
    let e = Engine::new_local();
    let empty = e.visible_state_fingerprint();

    e.execute_text(1, "SET a=1").unwrap();
    let after_set = e.visible_state_fingerprint();
    assert_ne!(after_set, empty);

    e.execute_text(2, "DELETE a").unwrap();
    let after_delete = e.visible_state_fingerprint();
    assert_eq!(after_delete, empty);
}

#[test]
fn flush_command_drains_pending_batch() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=5", t0).unwrap();
    e.execute_text(2, "FLUSH").unwrap();

    assert_eq!(e.get("a").as_deref(), Some("5"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
}

#[test]
fn flush_aliases_drain_pending_batch() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=5", t0).unwrap();
    e.execute_text(2, "FLUSH WAL").unwrap();

    e.enqueue_set_text(3, "SET b=7", t0).unwrap();
    e.execute_text(4, "FLUSH LOG").unwrap();

    assert_eq!(e.get("a").as_deref(), Some("5"));
    assert_eq!(e.get("b").as_deref(), Some("7"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 2);
}

#[test]
fn wal_flush_failure_prevents_visibility_advance() {
    let mut e = Engine::new_local();
    e.simulate_next_wal_flush_failure();
    let res = e.commit_mutation(1, b"SET a=1".to_vec());
    assert!(matches!(res, Err(EngineError::Durability(_))));
    assert_eq!(e.visible_up_to(), 0);
}

#[test]
fn wal_flush_failure_does_not_leak_into_later_successful_commit() {
    let mut e = Engine::new_local();
    e.simulate_next_wal_flush_failure();
    let _ = e.commit_mutation(1, b"SET a=1".to_vec());

    e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

    assert_eq!(e.get("a"), None);
    assert_eq!(e.get("b").as_deref(), Some("2"));
    assert_eq!(e.applied_len(), 1);
}

#[test]
fn wal_flush_failure_discards_unflushed_record_from_buffer() {
    let mut e = Engine::new_local();
    e.simulate_next_wal_flush_failure();

    let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

    assert!(matches!(err, EngineError::Durability(_)));
    assert_eq!(e.wal_flushed_count(), 0);
    assert_eq!(e.wal_buffered_count(), 0);
    assert_eq!(e.wal_unflushed_count(), 0);
}

#[test]
fn durable_wal_records_exclude_failed_commit_attempts() {
    let mut e = Engine::new_local();

    e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
    e.simulate_next_wal_flush_failure();
    let _ = e.commit_mutation(2, b"SET b=2".to_vec());

    let durable = e.durable_wal_records();
    assert_eq!(durable.len(), 1);
    assert_eq!(durable[0].txn_id, 1);
    assert_eq!(durable[0].payload, b"SET a=1".to_vec());
}

#[test]
fn follower_rejects_commit_without_visibility_or_wal_flush() {
    let mut e = Engine::new_local();
    e.become_follower(2);

    let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();

    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(e.visible_up_to(), 0);
    assert_eq!(e.wal_flushed_count(), 0);
    assert_eq!(e.applied_len(), 0);
}

#[test]
fn follower_rejects_batched_enqueue_without_mutating_queue_or_metrics() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_follower(2);

    let t0 = Instant::now();
    let err = e.enqueue_set_text(1, "SET a=1", t0).unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Count), 0);
    assert_eq!(e.metrics().last_batch_flush_reason(), None);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn enqueue_get_rejects_when_not_leader_without_queue_side_effects() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_follower(2);

    let t0 = Instant::now();
    let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn enqueue_get_rejects_when_candidate_without_queue_side_effects() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_candidate(2);

    let t0 = Instant::now();
    let err = e.enqueue_set_text(1, "GET a", t0).unwrap_err();

    assert!(matches!(err, ExecuteError::Engine(EngineError::NotLeader)));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn enqueue_get_tracks_d2h_bytes_for_hits_only() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET balance=100", t0).unwrap();
    e.flush_admin().unwrap();

    e.enqueue_set_text(2, "GET balance", t0).unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);

    e.enqueue_set_text(3, "GET missing", t0).unwrap();
    assert_eq!(e.metrics().snapshot().d2h_bytes_total, "100".len() as u64);
}

#[test]
fn execute_text_mutation_falls_back_to_cpu_when_gpu_is_unavailable() {
    let mut e = Engine::new_local();
    e.mark_gpu_unavailable(0);

    e.execute_text(1, "SET balance=100").unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);

    let snapshot = e.telemetry_snapshot();
    assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
    assert!(snapshot.has_gpu_parity_fallbacks());
    assert!(snapshot.has_gpu_runtime_pressure());
    assert_eq!(snapshot.blocked_gpu_ids(), vec![0]);
    assert_eq!(
        snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
            id: "GPU-120",
            owner: "runtime",
            milestone: "m0-bootstrap",
        }),
        Some(&1)
    );
}

#[test]
fn enqueue_mutation_falls_back_to_cpu_when_gpu_is_memory_pressured() {
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    let t0 = Instant::now();
    e.mark_gpu_memory_pressured(0);

    e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(
        e.metrics().fallback_for(FallbackReason::GpuMemoryPressure),
        1
    );
}

#[test]
fn enqueue_mutation_runtime_saturation_falls_back_before_queueing() {
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    let t0 = Instant::now();
    e.set_gpu_runtime_saturated(true);

    e.enqueue_set_text(1, "SET balance=100", t0).unwrap();

    assert_eq!(e.get("balance").as_deref(), Some("100"));
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 1);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(
        e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
        1
    );

    let snapshot = e.telemetry_snapshot();
    assert!(snapshot.has_gpu_runtime_pressure());
    assert!(snapshot.blocked_gpu_ids().is_empty());
    assert!(snapshot.gpu_runtime.saturated);
}

#[test]
fn candidate_rejects_commit_and_batched_enqueue() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    e.become_candidate(2);

    let commit_err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
    assert!(matches!(commit_err, EngineError::NotLeader));

    let enqueue_err = e
        .enqueue_set_text(1, "SET a=1", Instant::now())
        .unwrap_err();
    assert!(matches!(
        enqueue_err,
        ExecuteError::Engine(EngineError::NotLeader)
    ));

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
    assert_eq!(e.visible_up_to(), 0);
}

#[test]
fn failed_admin_flush_does_not_increment_flush_metrics_or_drop_pending_queue() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(2);

    let err = e.flush_admin().unwrap_err();

    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 0);
    assert_eq!(e.metrics().last_batch_flush_reason(), None);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
    assert_eq!(e.pending_batch_len(), 1);
}

#[test]
fn batch_flush_wal_failure_requeues_items_for_retry() {
    let mut e = Engine::with_batching(2, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.simulate_next_wal_flush_failure();
    let err = e
        .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
        .unwrap_err();

    assert!(matches!(
        err,
        ExecuteError::Engine(EngineError::Durability(_))
    ));
    assert_eq!(e.pending_batch_len(), 2);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().snapshot().batch_wait_samples, 0);
    assert_eq!(e.metrics().snapshot().pending_batch_peak, 2);
    assert_eq!(e.metrics().last_pending_batch_len(), Some(2));
    assert_eq!(e.metrics().snapshot().commits_total, 0);

    e.flush_admin().unwrap();
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.get("a").as_deref(), Some("1"));
    assert_eq!(e.get("b").as_deref(), Some("2"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    assert_eq!(e.metrics().snapshot().commits_total, 2);
}

#[test]
fn enqueue_rejects_new_mutation_when_retry_queue_is_saturated() {
    let mut e = Engine::with_batching(2, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.simulate_next_wal_flush_failure();
    let flush_err = e
        .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
        .unwrap_err();
    assert!(matches!(
        flush_err,
        ExecuteError::Engine(EngineError::Durability(_))
    ));
    assert_eq!(e.pending_batch_len(), 2);

    let saturated_err = e
        .enqueue_set_text(3, "SET c=3", t0 + Duration::from_millis(2))
        .unwrap_err();
    assert!(matches!(
        saturated_err,
        ExecuteError::Engine(EngineError::MutationQueueOverloaded { pending: 2, cap: 2 })
    ));
    assert_eq!(e.pending_batch_len(), 2);
    assert_eq!(
        e.metrics().fallback_for(FallbackReason::GpuQueueSaturated),
        1
    );
    assert_eq!(e.metrics().snapshot().commits_total, 0);

    let snapshot = e.telemetry_snapshot();
    assert!(snapshot.has_gpu_parity_fallbacks());
    assert_eq!(snapshot.gpu_parity_fallback_total(), 1);
    assert_eq!(
        snapshot.gpu_parity_fallbacks.get(&GpuParityIssue {
            id: "GPU-121",
            owner: "runtime",
            milestone: "m0-bootstrap",
        }),
        Some(&1)
    );
}

#[test]
fn failed_time_flush_does_not_drop_pending_queue() {
    let mut e = Engine::with_batching(10, Duration::from_millis(2));
    let t0 = Instant::now();
    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(2);

    let err = e.tick_batching(t0 + Duration::from_millis(3)).unwrap_err();

    assert!(matches!(err, EngineError::NotLeader));
    assert_eq!(e.pending_batch_len(), 1);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn follower_tick_without_pending_batch_is_noop() {
    let mut e = Engine::with_batching(10, Duration::from_millis(2));
    e.become_follower(2);

    e.tick_batching(Instant::now()).unwrap();

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().batch_flush_count, 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn pending_batch_can_be_flushed_after_follower_is_promoted_back_to_leader() {
    let mut e = Engine::with_batching(10, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(2);

    let tick_err = e.tick_batching(t0 + Duration::from_secs(1)).unwrap_err();
    assert!(matches!(tick_err, EngineError::NotLeader));
    assert_eq!(e.pending_batch_len(), 1);
    assert_eq!(e.get("a"), None);

    e.become_leader(3);
    e.flush_admin().unwrap();

    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.get("a").as_deref(), Some("1"));
    assert_eq!(e.metrics().batch_flushes_for(BatchFlushReason::Admin), 1);
    assert_eq!(e.metrics().snapshot().commits_total, 1);
}

#[test]
fn execute_text_non_mutations_count_as_not_gpu_eligible_fallbacks() {
    let e = Engine::new_local();

    e.execute_text(1, "BEGIN").unwrap();
    e.execute_text(1, "COMMIT").unwrap();
    e.execute_text(2, "BEGIN").unwrap();
    e.execute_text(2, "ROLLBACK").unwrap();
    e.execute_text(3, "GET missing").unwrap();
    e.execute_text(4, "FLUSH").unwrap();
    e.execute_text(5, "RESET ALL").unwrap();
    e.execute_text(6, "DISCARD TEMP").unwrap();
    e.execute_text(
        7,
        "CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog",
    )
    .unwrap();

    assert_eq!(e.active_txn_count(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 9);
    assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 9);
    assert_eq!(
        e.metrics().last_fallback_reason(),
        Some(FallbackReason::NotGpuEligible)
    );
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_text_bounds_bootstrap_extension_create() {
    let e = Engine::new_local();

    e.execute_text(1, "CREATE EXTENSION IF NOT EXISTS plpgsql")
        .unwrap();
    e.execute_text(
        2,
        "CREATE EXTENSION IF NOT EXISTS \"plpgsql\" WITH SCHEMA pg_catalog",
    )
    .unwrap();

    let duplicate = e.execute_text(3, "CREATE EXTENSION plpgsql").unwrap_err();
    assert!(matches!(
        duplicate,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "extension \"plpgsql\" already exists"
    ));

    let unsupported = e
        .execute_text(4, "CREATE EXTENSION IF NOT EXISTS hstore")
        .unwrap_err();
    assert!(matches!(
        unsupported,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "only the bootstrap plpgsql extension is supported"
    ));

    let wrong_schema = e
        .execute_text(
            5,
            "CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA public",
        )
        .unwrap_err();
    assert!(matches!(
        wrong_schema,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "plpgsql extension creation is only supported in pg_catalog"
    ));
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn execute_text_accepts_bootstrap_extension_if_exists_cleanup() {
    let e = Engine::new_local();

    e.execute_text(1, "COMMENT ON EXTENSION plpgsql IS 'bootstrap extension'")
        .unwrap();
    e.execute_text(2, "DROP EXTENSION IF EXISTS plpgsql")
        .unwrap();
    assert_eq!(
        e.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    let bootstrap_drop = e.execute_text(3, "DROP EXTENSION plpgsql").unwrap_err();
    assert!(matches!(
        bootstrap_drop,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "cannot drop bootstrap extension \"plpgsql\""
    ));

    let unsupported = e
        .execute_text(4, "DROP EXTENSION IF EXISTS hstore")
        .unwrap_err();
    assert!(matches!(
        unsupported,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "extension \"hstore\" does not exist"
    ));
    assert_eq!(e.metrics().snapshot().commits_total, 1);
}

#[test]
fn execute_text_records_bootstrap_extension_comment_and_replays_from_wal() {
    let e = Engine::new_local();

    e.execute_text(1, "COMMENT ON EXTENSION plpgsql IS 'bootstrap extension'")
        .unwrap();
    assert_eq!(
        e.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.relational_extension_comment("plpgsql").as_deref(),
        Some("bootstrap extension")
    );

    e.execute_text(2, "COMMENT ON EXTENSION plpgsql IS NULL")
        .unwrap();
    assert_eq!(e.relational_extension_comment("plpgsql"), None);

    let missing = e
        .execute_text(3, "COMMENT ON EXTENSION hstore IS 'missing'")
        .unwrap_err();
    assert!(matches!(
        missing,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "extension \"hstore\" does not exist"
    ));
}

#[test]
fn execute_text_replays_bounded_role_metadata_and_acl_grantees() {
    let e = Engine::new_local();

    e.execute_text(1, "CREATE ROLE app_reader WITH LOGIN")
        .unwrap();
    e.execute_text(2, "CREATE USER app_writer").unwrap();
    e.execute_text(3, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(4, "GRANT SELECT ON TABLE people TO app_reader")
        .unwrap();
    e.execute_text(
        5,
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO app_writer",
    )
    .unwrap();
    e.execute_text(6, "COMMENT ON ROLE app_reader IS 'read-only app'")
        .unwrap();

    assert!(e.relational_role("app_reader").unwrap().login);
    assert!(e.relational_role("app_writer").unwrap().login);
    assert_eq!(
        e.relational_role_comment("app_reader").as_deref(),
        Some("read-only app")
    );
    assert!(e
        .relational_catalog_table("people")
        .unwrap()
        .acl
        .contains_key("app_reader"));
    assert!(e
        .ddl_catalog()
        .relational_default_table_acl
        .contains_key("app_writer"));

    e.execute_text(7, "ALTER ROLE app_reader RENAME TO app_analyst")
        .unwrap();
    assert!(e.relational_role("app_reader").is_none());
    assert!(e.relational_role("app_analyst").unwrap().login);
    assert_eq!(
        e.relational_role_comment("app_analyst").as_deref(),
        Some("read-only app")
    );
    assert!(e
        .relational_catalog_table("people")
        .unwrap()
        .acl
        .contains_key("app_analyst"));
    assert!(!e
        .relational_catalog_table("people")
        .unwrap()
        .acl
        .contains_key("app_reader"));

    let dependent_drop = e.execute_text(8, "DROP ROLE app_analyst").unwrap_err();
    assert!(dependent_drop
        .to_string()
        .contains("dependent metadata exists"));

    e.execute_text(9, "REVOKE SELECT ON TABLE people FROM app_analyst")
        .unwrap();
    e.execute_text(10, "COMMENT ON ROLE app_analyst IS NULL")
        .unwrap();
    e.execute_text(11, "DROP ROLE app_analyst").unwrap();
    e.execute_text(12, "DROP USER IF EXISTS app_missing")
        .unwrap();

    assert!(e.relational_role("app_analyst").is_none());
    assert!(e.relational_role("app_writer").is_some());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_role("app_analyst").is_none());
    assert!(recovered.relational_role("app_writer").unwrap().login);
    assert!(recovered
        .ddl_catalog()
        .relational_default_table_acl
        .contains_key("app_writer"));

    let missing_grantee = Engine::new_local()
        .execute_text(1, "GRANT SELECT ON TABLE people TO missing_role")
        .unwrap_err();
    assert!(missing_grantee
        .to_string()
        .contains("relation \"people\" does not exist"));

    let missing_role = Engine::new_local();
    missing_role
        .execute_text(1, "CREATE TABLE people (id INT)")
        .unwrap();
    assert!(missing_role
        .execute_text(2, "GRANT SELECT ON TABLE people TO missing_role")
        .unwrap_err()
        .to_string()
        .contains("role \"missing_role\" does not exist"));
    assert!(missing_role
        .execute_text(3, "DROP ROLE postgres")
        .unwrap_err()
        .to_string()
        .contains("cannot drop bootstrap role"));
    assert!(missing_role
        .execute_text(4, "CREATE ROLE app_password PASSWORD 'secret'")
        .is_err());
    assert!(missing_role
        .execute_text(5, "ALTER ROLE postgres RENAME TO root")
        .unwrap_err()
        .to_string()
        .contains("cannot rename bootstrap role"));
    assert!(missing_role
        .execute_text(6, "ALTER ROLE missing_role RENAME TO renamed_role")
        .unwrap_err()
        .to_string()
        .contains("role \"missing_role\" does not exist"));
}

#[test]
fn execute_text_replays_bounded_database_metadata() {
    let e = Engine::new_local();

    e.execute_text(1, "CREATE DATABASE appdb").unwrap();
    e.execute_text(2, "COMMENT ON DATABASE appdb IS 'application database'")
        .unwrap();
    e.execute_text(3, "CREATE ROLE app_reader").unwrap();
    e.execute_text(
        4,
        "GRANT CONNECT, TEMPORARY ON DATABASE appdb TO app_reader",
    )
    .unwrap();

    let appdb = e.relational_database("appdb").unwrap();
    assert_eq!(appdb.name, "appdb");
    let oid = appdb.oid;
    assert_eq!(
        e.relational_database_acl("appdb")
            .unwrap()
            .get("app_reader")
            .unwrap(),
        &BTreeSet::from([DatabasePrivilege::Connect, DatabasePrivilege::Temporary])
    );
    assert_eq!(
        e.relational_database_comment("appdb").as_deref(),
        Some("application database")
    );

    e.execute_text(5, "ALTER ROLE app_reader RENAME TO app_analyst")
        .unwrap();
    assert!(e
        .relational_database_acl("appdb")
        .unwrap()
        .contains_key("app_analyst"));
    e.execute_text(6, "REVOKE TEMP ON DATABASE appdb FROM app_analyst")
        .unwrap();
    assert_eq!(
        e.relational_database_acl("appdb")
            .unwrap()
            .get("app_analyst")
            .unwrap(),
        &BTreeSet::from([DatabasePrivilege::Connect])
    );

    e.execute_text(7, "ALTER DATABASE appdb RENAME TO appdb_renamed")
        .unwrap();
    let renamed = e.relational_database("appdb_renamed").unwrap();
    assert_eq!(renamed.oid, oid);
    assert_eq!(renamed.name, "appdb_renamed");
    assert!(e
        .relational_database_acl("appdb_renamed")
        .unwrap()
        .contains_key("app_analyst"));
    assert_eq!(
        e.relational_database_comment("appdb_renamed").as_deref(),
        Some("application database")
    );
    assert_eq!(e.relational_database_comment("appdb"), None);

    let duplicate = e
        .execute_text(8, "CREATE DATABASE appdb_renamed")
        .unwrap_err();
    assert!(duplicate
        .to_string()
        .contains("database \"appdb_renamed\" already exists"));
    let duplicate_rename = e
        .execute_text(9, "CREATE DATABASE appdb")
        .and_then(|_| e.execute_text(10, "ALTER DATABASE appdb_renamed RENAME TO appdb"))
        .unwrap_err();
    assert!(duplicate_rename
        .to_string()
        .contains("database \"appdb\" already exists"));

    e.execute_text(11, "DROP DATABASE appdb_renamed").unwrap();
    assert!(e.relational_database("appdb_renamed").is_none());
    assert_eq!(e.relational_database_comment("appdb_renamed"), None);
    e.execute_text(12, "DROP DATABASE IF EXISTS missing_db")
        .unwrap();

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_database("appdb_renamed").is_none());
    assert_eq!(recovered.relational_database_comment("appdb_renamed"), None);

    let kept = Engine::new_local();
    kept.execute_text(1, "CREATE DATABASE appdb").unwrap();
    let recovered_kept = Engine::recover_from_durable_wal(&kept.durable_wal_records()).unwrap();
    assert!(recovered_kept.relational_database("appdb").is_some());

    assert!(Engine::new_local()
        .execute_text(1, "DROP DATABASE postgres")
        .unwrap_err()
        .to_string()
        .contains("cannot drop bootstrap database"));
    assert!(Engine::new_local()
        .execute_text(1, "ALTER DATABASE postgres RENAME TO appdb")
        .unwrap_err()
        .to_string()
        .contains("cannot rename bootstrap database"));
    assert!(Engine::new_local()
        .execute_text(1, "ALTER DATABASE missing_db RENAME TO appdb")
        .unwrap_err()
        .to_string()
        .contains("database \"missing_db\" does not exist"));
    assert!(Engine::new_local()
        .execute_text(1, "CREATE DATABASE templated TEMPLATE template1")
        .is_err());
    assert!(Engine::new_local()
        .execute_text(1, "GRANT CONNECT ON DATABASE missing_db TO PUBLIC")
        .unwrap_err()
        .to_string()
        .contains("database \"missing_db\" does not exist"));
}

#[test]
fn enqueue_non_mutations_count_as_not_gpu_eligible_fallbacks() {
    let mut e = Engine::with_batching(2, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "BEGIN", t0).unwrap();
    e.enqueue_set_text(1, "COMMIT", t0).unwrap();
    e.enqueue_set_text(2, "BEGIN", t0).unwrap();
    e.enqueue_set_text(2, "ROLLBACK", t0).unwrap();
    e.enqueue_set_text(3, "GET missing", t0).unwrap();
    e.enqueue_set_text(4, "FLUSH", t0).unwrap();
    e.enqueue_set_text(5, "RESET ALL", t0).unwrap();
    e.enqueue_set_text(6, "DISCARD TEMP", t0).unwrap();

    assert_eq!(e.active_txn_count(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 8);
    assert_eq!(e.metrics().fallback_for(FallbackReason::NotGpuEligible), 8);
    assert_eq!(e.pending_batch_len(), 0);
    assert_eq!(e.metrics().snapshot().commits_total, 0);
}

#[test]
fn commit_and_rollback_require_active_transaction_context() {
    let e = Engine::new_local();

    let commit_err = e.execute_text(10, "COMMIT").unwrap_err();
    assert!(matches!(
        commit_err,
        ExecuteError::Txn(TxnError::NotFound(10))
    ));

    let rollback_err = e.execute_text(11, "ROLLBACK").unwrap_err();
    assert!(matches!(
        rollback_err,
        ExecuteError::Txn(TxnError::NotFound(11))
    ));

    e.execute_text(12, "BEGIN").unwrap();
    let duplicate_begin_err = e.execute_text(12, "BEGIN").unwrap_err();
    assert!(matches!(
        duplicate_begin_err,
        ExecuteError::Txn(TxnError::AlreadyExists(12))
    ));

    assert_eq!(e.metrics().snapshot().fallback_total, 1);
    assert_eq!(e.active_txn_count(), 1);
}

#[test]
fn and_chain_forms_reopen_transaction_context() {
    let e = Engine::new_local();

    e.execute_text(21, "BEGIN").unwrap();
    e.execute_text(21, "COMMIT AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(22, "COMMIT").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(31, "BEGIN").unwrap();
    e.execute_text(31, "ROLLBACK AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(32, "ROLLBACK").unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn enqueue_non_mutation_chain_forms_reopen_transaction_context() {
    let mut e = Engine::new_local();
    let t0 = Instant::now();

    e.enqueue_set_text(41, "BEGIN", t0).unwrap();
    e.enqueue_set_text(41, "COMMIT AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(42, "COMMIT", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(51, "BEGIN", t0).unwrap();
    e.enqueue_set_text(51, "ROLLBACK AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(52, "ROLLBACK", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn transaction_control_alias_chain_forms_reopen_transaction_context() {
    let e = Engine::new_local();

    e.execute_text(61, "BEGIN").unwrap();
    e.execute_text(61, "END AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(62, "END").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(71, "BEGIN").unwrap();
    e.execute_text(71, "ABORT AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(72, "ABORT").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(75, "BEGIN").unwrap();
    e.execute_text(75, "END WORK AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(76, "COMMIT").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(77, "BEGIN").unwrap();
    e.execute_text(77, "ABORT WORK AND CHAIN").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(78, "ROLLBACK").unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn start_alias_and_work_aliases_drive_transaction_state_transitions() {
    let e = Engine::new_local();

    e.execute_text(73, "START TRANSACTION READ ONLY").unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(73, "COMMIT WORK").unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.execute_text(74, "START WORK, READ WRITE, DEFERRABLE")
        .unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.execute_text(74, "ROLLBACK TRANSACTION").unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn enqueue_transaction_control_alias_chain_forms_reopen_transaction_context() {
    let mut e = Engine::new_local();
    let t0 = Instant::now();

    e.enqueue_set_text(81, "BEGIN", t0).unwrap();
    e.enqueue_set_text(81, "END AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(82, "END", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(91, "BEGIN", t0).unwrap();
    e.enqueue_set_text(91, "ABORT AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(92, "ABORT", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(95, "BEGIN", t0).unwrap();
    e.enqueue_set_text(95, "END WORK AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(96, "COMMIT", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(97, "BEGIN", t0).unwrap();
    e.enqueue_set_text(97, "ABORT WORK AND CHAIN", t0).unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(98, "ROLLBACK", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn enqueue_start_alias_and_work_aliases_drive_transaction_state_transitions() {
    let mut e = Engine::new_local();
    let t0 = Instant::now();

    e.enqueue_set_text(93, "START TRANSACTION READ ONLY", t0)
        .unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(93, "COMMIT WORK", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);

    e.enqueue_set_text(94, "START WORK, READ WRITE, DEFERRABLE", t0)
        .unwrap();
    assert_eq!(e.active_txn_count(), 1);
    e.enqueue_set_text(94, "ROLLBACK TRANSACTION", t0).unwrap();
    assert_eq!(e.active_txn_count(), 0);
}

#[test]
fn commit_and_chain_propagates_txn_id_exhaustion() {
    let e = Engine::new_local();

    e.execute_text(u64::MAX, "BEGIN").unwrap();
    let err = e.execute_text(u64::MAX, "COMMIT AND CHAIN").unwrap_err();

    assert!(matches!(err, ExecuteError::Txn(TxnError::IdExhausted)));
    assert_eq!(e.active_txn_count(), 0);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn replication_watermarks_track_commit_apply_visibility_and_durability() {
    let e = Engine::new_local();

    let before = e.replication_watermarks();
    assert_eq!(before.role, Role::Leader);
    assert_eq!(before.commit_index, 0);
    assert_eq!(before.applied_index, 0);
    assert_eq!(before.visible_index, 0);
    assert_eq!(before.commit_apply_gap, 0);
    assert_eq!(before.apply_visible_gap, 0);
    assert_eq!(before.snapshot_id, 0);
    assert_eq!(before.wal_flushed_count, 0);
    assert_eq!(before.wal_buffered_count, 0);
    assert_eq!(before.wal_unflushed_count, 0);
    assert_eq!(before.pending_batch_len, 0);
    assert_eq!(before.pending_batch_cap, 64);
    assert_eq!(before.pending_batch_remaining_capacity, 64);
    assert_eq!(before.pending_batch_utilization_permyriad, 0);
    assert_eq!(before.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(before.pending_batch_oldest_age_ms, None);
    assert_eq!(before.pending_batch_time_until_deadline_ms, None);
    assert_eq!(before.active_txn_count, 0);
    assert_eq!(before.oldest_active_txn_id, None);
    assert_eq!(before.newest_active_txn_id, None);
    assert!(!before.has_wal_backlog);
    assert!(!before.has_pending_batch_backlog);
    assert!(!before.has_active_txn_backlog);
    assert!(!before.has_commit_apply_gap);
    assert!(!before.has_apply_visible_gap);
    assert!(!before.has_backlog_blockers);
    assert_eq!(before.backlog_blocker_count, 0);
    assert_eq!(before.backlog_blocker_mask, 0);
    assert!(!before.mutation_admission_saturated);
    assert!(before.quiescent_for_failover);
    assert!(!before.follower_promotion_ready);

    let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
    let after = e.replication_watermarks();

    assert_eq!(after.role, Role::Leader);
    assert!(after.term >= before.term);
    assert_eq!(after.commit_index, token.index);
    assert_eq!(after.applied_index, token.index);
    assert_eq!(after.visible_index, token.index);
    assert_eq!(after.commit_apply_gap, 0);
    assert_eq!(after.apply_visible_gap, 0);
    assert_eq!(after.snapshot_id, 0);
    assert!(after.wal_flushed_count >= 1);
    assert_eq!(after.wal_buffered_count, e.wal_buffered_count());
    assert_eq!(after.wal_unflushed_count, e.wal_unflushed_count());
    assert_eq!(after.pending_batch_len, 0);
    assert_eq!(after.pending_batch_cap, 64);
    assert_eq!(after.pending_batch_remaining_capacity, 64);
    assert_eq!(after.pending_batch_utilization_permyriad, 0);
    assert_eq!(after.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(after.pending_batch_oldest_age_ms, None);
    assert_eq!(after.pending_batch_time_until_deadline_ms, None);
    assert_eq!(after.active_txn_count, 0);
    assert_eq!(after.oldest_active_txn_id, None);
    assert_eq!(after.newest_active_txn_id, None);
    assert!(!after.has_wal_backlog);
    assert!(!after.has_pending_batch_backlog);
    assert!(!after.has_active_txn_backlog);
    assert!(!after.has_commit_apply_gap);
    assert!(!after.has_apply_visible_gap);
    assert!(!after.has_backlog_blockers);
    assert_eq!(after.backlog_blocker_count, 0);
    assert_eq!(after.backlog_blocker_mask, 0);
    assert!(!after.mutation_admission_saturated);
    assert!(after.quiescent_for_failover);
    assert!(!after.follower_promotion_ready);
}

#[test]
fn replication_watermarks_do_not_advance_on_rejected_follower_commit() {
    let mut e = Engine::new_local();
    e.become_follower(2);

    let err = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap_err();
    assert!(matches!(err, EngineError::NotLeader));

    let marks = e.replication_watermarks();
    assert_eq!(marks.role, Role::Follower);
    assert_eq!(marks.term, 2);
    assert_eq!(marks.commit_index, 0);
    assert_eq!(marks.applied_index, 0);
    assert_eq!(marks.visible_index, 0);
    assert_eq!(marks.commit_apply_gap, 0);
    assert_eq!(marks.apply_visible_gap, 0);
    assert_eq!(marks.wal_flushed_count, 0);
    assert_eq!(marks.wal_last_durable_txn_id, None);
    assert_eq!(marks.wal_buffered_count, 0);
    assert_eq!(marks.wal_unflushed_count, 0);
    assert_eq!(marks.pending_batch_len, 0);
    assert_eq!(marks.pending_batch_cap, 64);
    assert_eq!(marks.pending_batch_remaining_capacity, 64);
    assert_eq!(marks.pending_batch_utilization_permyriad, 0);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(marks.pending_batch_oldest_age_ms, None);
    assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
    assert_eq!(marks.active_txn_count, 0);
    assert!(!marks.has_wal_backlog);
    assert!(!marks.has_pending_batch_backlog);
    assert!(!marks.has_active_txn_backlog);
    assert!(!marks.has_commit_apply_gap);
    assert!(!marks.has_apply_visible_gap);
    assert!(!marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_mask, 0);
    assert!(!marks.mutation_admission_saturated);
    assert!(!marks.quiescent_for_failover);
    assert!(marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_include_buffered_wal_records() {
    let e = Engine::new_local();

    e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
    e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.wal_buffered_count, 2);
    assert_eq!(marks.wal_flushed_count, 2);
    assert_eq!(marks.wal_last_durable_txn_id, Some(2));
    assert_eq!(marks.wal_unflushed_count, 0);
    assert_eq!(marks.pending_batch_len, 0);
    assert_eq!(marks.pending_batch_cap, 64);
    assert_eq!(marks.pending_batch_remaining_capacity, 64);
    assert_eq!(marks.pending_batch_utilization_permyriad, 0);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 10_000);
    assert_eq!(marks.pending_batch_oldest_age_ms, None);
    assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
    assert_eq!(marks.active_txn_count, 0);
    assert!(!marks.mutation_admission_saturated);
    assert!(marks.quiescent_for_failover);
}

#[test]
fn replication_watermarks_pending_batch_time_fields_clear_after_flush() {
    let mut e = Engine::with_batching(3, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    let before_flush = e.replication_watermarks();
    assert_eq!(before_flush.pending_batch_len, 1);
    assert_eq!(before_flush.pending_batch_cap, 3);
    assert_eq!(before_flush.pending_batch_remaining_capacity, 2);
    assert_eq!(before_flush.pending_batch_utilization_permyriad, 3_333);
    assert_eq!(
        before_flush.pending_batch_remaining_capacity_permyriad,
        6_667
    );
    assert!(before_flush.pending_batch_oldest_age_ms.is_some());
    assert!(before_flush.pending_batch_time_until_deadline_ms.is_some());

    e.flush_admin().unwrap();
    let after_flush = e.replication_watermarks();
    assert_eq!(after_flush.pending_batch_len, 0);
    assert_eq!(after_flush.pending_batch_cap, 3);
    assert_eq!(after_flush.pending_batch_remaining_capacity, 3);
    assert_eq!(after_flush.pending_batch_utilization_permyriad, 0);
    assert_eq!(
        after_flush.pending_batch_remaining_capacity_permyriad,
        10_000
    );
    assert_eq!(after_flush.pending_batch_oldest_age_ms, None);
    assert_eq!(after_flush.pending_batch_time_until_deadline_ms, None);
}

#[test]
fn replication_watermarks_include_pending_batch_depth() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.pending_batch_len, 1);
    assert_eq!(marks.pending_batch_cap, 2);
    assert_eq!(marks.pending_batch_remaining_capacity, 1);
    assert_eq!(marks.pending_batch_utilization_permyriad, 5_000);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 5_000);
    assert!(marks.pending_batch_oldest_age_ms.is_some());
    assert!(marks.pending_batch_time_until_deadline_ms.is_some());
    assert!(marks.has_pending_batch_backlog);
    assert!(marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_count, 1);
    assert_eq!(
        marks.backlog_blocker_mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
    );
    assert!(!marks.has_wal_backlog);
    assert!(!marks.has_active_txn_backlog);
    assert!(!marks.has_commit_apply_gap);
    assert!(!marks.has_apply_visible_gap);
    assert_eq!(marks.wal_buffered_count, 0);
    assert_eq!(marks.wal_unflushed_count, 0);
    assert_eq!(marks.active_txn_count, 0);
    assert!(!marks.quiescent_for_failover);
}

#[test]
fn replication_watermarks_flag_mutation_admission_saturation() {
    let mut e = Engine::with_batching(2, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.simulate_next_wal_flush_failure();
    let err = e
        .enqueue_set_text(2, "SET b=2", t0 + Duration::from_millis(1))
        .unwrap_err();
    assert!(matches!(
        err,
        ExecuteError::Engine(EngineError::Durability(_))
    ));

    let marks = e.replication_watermarks();
    assert_eq!(marks.pending_batch_len, 2);
    assert_eq!(marks.pending_batch_cap, 2);
    assert_eq!(marks.pending_batch_remaining_capacity, 0);
    assert_eq!(marks.pending_batch_utilization_permyriad, 10_000);
    assert_eq!(marks.pending_batch_remaining_capacity_permyriad, 0);
    assert!(marks.has_pending_batch_backlog);
    assert!(!marks.has_wal_backlog);
    assert!(!marks.has_active_txn_backlog);
    assert!(marks.mutation_admission_saturated);
    assert!(!marks.quiescent_for_failover);
    assert!(!marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_follower_promotion_ready_requires_no_backlog() {
    let mut e = Engine::with_batching(8, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();
    e.become_follower(3);

    let marks = e.replication_watermarks();
    assert_eq!(marks.role, Role::Follower);
    assert_eq!(marks.pending_batch_len, 1);
    assert!(!marks.quiescent_for_failover);
    assert!(!marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_follower_promotion_ready_requires_zero_active_txns() {
    let mut e = Engine::new_local();
    e.become_follower(5);
    e.execute_text(9, "BEGIN").unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.role, Role::Follower);
    assert_eq!(marks.active_txn_count, 1);
    assert_eq!(marks.oldest_active_txn_id, Some(9));
    assert_eq!(marks.newest_active_txn_id, Some(9));
    assert!(!marks.follower_promotion_ready);
}

#[test]
fn replication_watermarks_include_active_transaction_count() {
    let e = Engine::new_local();

    e.execute_text(42, "BEGIN").unwrap();

    let marks = e.replication_watermarks();
    assert_eq!(marks.active_txn_count, 1);
    assert_eq!(marks.oldest_active_txn_id, Some(42));
    assert_eq!(marks.newest_active_txn_id, Some(42));
    assert_eq!(marks.pending_batch_len, 0);
    assert_eq!(marks.pending_batch_oldest_age_ms, None);
    assert_eq!(marks.pending_batch_time_until_deadline_ms, None);
    assert!(marks.has_active_txn_backlog);
    assert!(marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_count, 1);
    assert_eq!(
        marks.backlog_blocker_mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
    );
    assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
    assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
    assert!(!marks.has_pending_batch_backlog);
    assert!(!marks.has_wal_backlog);
    assert_eq!(marks.wal_buffered_count, 0);
    assert!(!marks.mutation_admission_saturated);
    assert!(!marks.quiescent_for_failover);
}

#[test]
fn backlog_blocker_enum_roundtrips_through_bits_and_labels() {
    for blocker in BacklogBlocker::ALL {
        assert!(blocker.bit().is_power_of_two());
        assert!(!blocker.as_str().is_empty());
        assert_eq!(BacklogBlocker::from_bit(blocker.bit()), Some(blocker));
        assert_eq!(BacklogBlocker::from_label(blocker.as_str()), Some(blocker));

        let mut marks = Engine::new_local().replication_watermarks();
        marks.backlog_blocker_mask = blocker.bit();
        assert!(marks.has_blocker_kind(blocker));
        assert_eq!(marks.backlog_blockers().collect::<Vec<_>>(), vec![blocker]);
        assert_eq!(
            marks.backlog_blocker_labels().collect::<Vec<_>>(),
            vec![blocker.as_str()]
        );
        assert_eq!(
            marks.backlog_blocker_bits().collect::<Vec<_>>(),
            vec![blocker.bit()]
        );
        assert_eq!(
            ReplicationWatermarks::backlog_blockers_from_mask(marks.backlog_blocker_mask)
                .collect::<Vec<_>>(),
            vec![blocker]
        );
    }
}

#[test]
fn backlog_blocker_display_and_from_str_roundtrip() {
    for blocker in BacklogBlocker::ALL {
        let label = blocker.to_string();
        assert_eq!(label, blocker.as_str());
        assert_eq!(label.parse::<BacklogBlocker>(), Ok(blocker));
    }
}

#[test]
fn backlog_blocker_from_str_reports_unknown_label() {
    let err = "  not-a-real-blocker  "
        .parse::<BacklogBlocker>()
        .expect_err("unknown blocker labels should fail to parse");

    assert_eq!(err.label(), "not-a-real-blocker");
    assert_eq!(
        err.to_string(),
        "unknown backlog blocker label: not-a-real-blocker"
    );
}

#[test]
fn replication_watermarks_backlog_blockers_from_mask_ignores_unknown_bits() {
    let known_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN;
    let unknown_mask = 1 << 7;

    assert_eq!(BacklogBlocker::from_bit(unknown_mask), None);
    assert_eq!(
        ReplicationWatermarks::backlog_blockers_from_mask(known_mask | unknown_mask)
            .collect::<Vec<_>>(),
        vec![BacklogBlocker::Wal, BacklogBlocker::ActiveTxn]
    );
}

#[test]
fn backlog_blocker_mask_helpers_strip_unknown_bits() {
    let unknown_mask = (1 << 5) | (1 << 7);
    let mixed_mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
        | unknown_mask;

    assert_eq!(
        ReplicationWatermarks::known_backlog_blocker_mask(),
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
    assert_eq!(
        ReplicationWatermarks::unknown_backlog_blocker_mask(mixed_mask),
        unknown_mask
    );
    assert_eq!(
        ReplicationWatermarks::sanitize_backlog_blocker_mask(mixed_mask),
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blocker_count_from_mask(mixed_mask),
        2
    );
    assert!(ReplicationWatermarks::has_backlog_blockers_in_mask(
        mixed_mask
    ));
    assert!(!ReplicationWatermarks::has_backlog_blockers_in_mask(
        unknown_mask
    ));
}

#[test]
fn replication_watermarks_backlog_blocker_mask_from_labels_ignores_unknowns() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
        "pending_batch",
        "unknown",
        "active_txn",
        "pending_batch",
    ]);

    assert_eq!(BacklogBlocker::from_label("unknown"), None);
    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blockers_from_mask(mask).collect::<Vec<_>>(),
        vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blocker_labels_from_mask(mask).collect::<Vec<_>>(),
        vec!["pending_batch", "active_txn"]
    );
}

#[test]
fn backlog_blocker_label_decode_normalizes_case_spacing_and_hyphenation() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_labels([
        " WAL ",
        "pending-batch",
        "ACTIVE TXN",
        "commit-apply-gap",
        "apply visible gap",
        "pending.batch",
        "commit--apply  gap",
    ]);

    assert_eq!(
        BacklogBlocker::from_label("PENDING-BATCH"),
        Some(BacklogBlocker::PendingBatch)
    );
    assert_eq!(
        BacklogBlocker::from_label("apply visible gap"),
        Some(BacklogBlocker::ApplyVisibleGap)
    );
    assert_eq!(
        BacklogBlocker::from_label("pending.batch"),
        Some(BacklogBlocker::PendingBatch)
    );
    assert_eq!(
        BacklogBlocker::from_label("commit--apply  gap"),
        Some(BacklogBlocker::CommitApplyGap)
    );
    assert_eq!(
        BacklogBlocker::from_label("__wal__"),
        Some(BacklogBlocker::Wal)
    );
    assert_eq!(
        BacklogBlocker::from_label("___active.txn___"),
        Some(BacklogBlocker::ActiveTxn)
    );
    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_decodes_csv_like_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal, pending-batch; ACTIVE TXN | unknown / apply visible gap : commit apply gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_ignores_empty_segments() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        " , ; | pending_batch || wal ,, ",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_multiline_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal\n pending_batch\r\nACTIVE TXN\t| commit apply gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_jsonish_arrays() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "[\"wal\",\"active_txn\",\"apply visible gap\"]",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_braced_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "{'wal';'active_txn';'apply visible gap'}",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_parenthesized_and_angle_bracket_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "<(wal|active_txn|apply visible gap)>",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_single_quoted_arrays() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "['pending-batch','commit_apply_gap']",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_backtick_quoted_labels() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "`wal`,`active_txn`,`apply_visible_gap`",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_plus_delimiter() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal+active_txn+apply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_windows_style_backslash_delimiter() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal\\active_txn\\apply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_assignment_and_ampersand_delimiters() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "backlog_blockers=wal&active_txn&apply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_mask_from_delimited_labels_accepts_percent_encoded_streams() {
    let mask = ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(
        "wal%2Cpending-batch%7CACTIVE%20TXN%2Fapply_visible_gap",
    );

    assert_eq!(
        mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_WAL
            | ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
            | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP
    );
}

#[test]
fn backlog_blocker_delimited_labels_from_mask_emits_canonical_order() {
    let mask = ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        | ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;

    assert_eq!(
        ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ","),
        "wal,active_txn,apply_visible_gap"
    );
    assert_eq!(
        ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, " | "),
        "wal | active_txn | apply_visible_gap"
    );
}

#[test]
fn backlog_blocker_delimited_mask_roundtrip_is_stable() {
    let mask = ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
        | ReplicationWatermarks::BACKLOG_BLOCKER_COMMIT_APPLY_GAP;
    let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ";");

    assert_eq!(
        ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
        mask
    );
}

#[test]
fn backlog_blocker_delimited_mask_roundtrip_is_stable_with_colon_delimiter() {
    let mask = ReplicationWatermarks::BACKLOG_BLOCKER_WAL
        | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        | ReplicationWatermarks::BACKLOG_BLOCKER_APPLY_VISIBLE_GAP;
    let labels = ReplicationWatermarks::backlog_blocker_delimited_labels_from_mask(mask, ":");

    assert_eq!(
        ReplicationWatermarks::backlog_blocker_mask_from_delimited_labels(&labels),
        mask
    );
}

#[test]
fn replication_watermarks_aggregate_multiple_backlog_blockers() {
    let mut e = Engine::with_batching(8, Duration::from_secs(999));
    let t0 = Instant::now();

    e.enqueue_set_text(7, "SET a=1", t0).unwrap();
    e.execute_text(8, "BEGIN").unwrap();

    let marks = e.replication_watermarks();
    assert!(marks.has_pending_batch_backlog);
    assert!(marks.has_active_txn_backlog);
    assert!(marks.has_backlog_blockers);
    assert_eq!(marks.backlog_blocker_count, 2);
    assert_eq!(
        marks.backlog_blocker_mask,
        ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH
            | ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
    );
    assert_eq!(
        marks.backlog_blocker_count,
        marks.backlog_blocker_mask.count_ones() as u8
    );
    assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH));
    assert!(marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN));
    assert!(!marks.has_backlog_blocker(ReplicationWatermarks::BACKLOG_BLOCKER_WAL));
    assert!(!marks.has_backlog_blocker(1 << 7));
    assert_eq!(
        marks.backlog_blockers().collect::<Vec<_>>(),
        vec![BacklogBlocker::PendingBatch, BacklogBlocker::ActiveTxn]
    );
    assert_eq!(
        marks.backlog_blocker_labels().collect::<Vec<_>>(),
        vec!["pending_batch", "active_txn"]
    );
    assert_eq!(
        marks.backlog_blocker_bits().collect::<Vec<_>>(),
        vec![
            ReplicationWatermarks::BACKLOG_BLOCKER_PENDING_BATCH,
            ReplicationWatermarks::BACKLOG_BLOCKER_ACTIVE_TXN
        ]
    );
    assert_eq!(marks.max_replication_gap(), 0);
    assert_eq!(marks.total_backlog_items(), 2);
    assert!(!marks.is_fully_caught_up());
    assert!(!marks.quiescent_for_failover);
    assert!(!marks.follower_promotion_ready);
}

#[test]
fn snapshot_export_tracks_last_applied_index() {
    let mut e = Engine::new_local();
    let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

    let exported = e.export_snapshot_meta();
    let current = e.snapshot_meta();

    assert_eq!(exported.last_included_index, token.index);
    assert_eq!(current.last_included_index, token.index);
    assert_eq!(exported.snapshot_id, 1);
    assert_eq!(current.snapshot_id, 1);
}

#[test]
fn install_snapshot_advances_visible_and_replication_watermarks() {
    let mut e = Engine::new_local();
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

    let next = e.commit_mutation(2, b"SET b=2".to_vec()).unwrap();
    assert_eq!(next.index, 8);
    assert_eq!(e.get("b").as_deref(), Some("2"));
}

#[test]
fn export_snapshot_meta_is_reflected_in_replication_watermarks() {
    let mut e = Engine::new_local();

    assert_eq!(e.replication_watermarks().snapshot_id, 0);

    e.export_snapshot_meta();
    assert_eq!(e.replication_watermarks().snapshot_id, 1);

    e.export_snapshot_meta();
    assert_eq!(e.replication_watermarks().snapshot_id, 2);
}

#[test]
fn relational_residency_snapshot_accounts_bytes_and_invalidates_on_later_wal_apply() {
    let mut e = Engine::new_local();
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
    assert_eq!(snapshot.last_refresh_cost, None);

    e.mark_gpu_memory_pressured(0);
    let pressured = e.relational_residency_snapshot("events").unwrap();
    assert!(pressured.memory_pressure_active);
    assert!(pressured.invalidated_by_memory_pressure);
    assert!(!pressured.is_valid());

    let valid_through = pressured.valid_through_index;
    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let invalidated = e.relational_residency_snapshot("events").unwrap();
    assert_eq!(invalidated.valid_through_index, valid_through);
    assert_eq!(invalidated.invalidated_by_txn_id, Some(3));
    assert!(invalidated.invalidated_at_index.unwrap() > valid_through);
    assert!(!invalidated.is_valid());

    e.clear_gpu_memory_pressured(0);
    let refreshed = e.populate_relational_residency_snapshot("events").unwrap();
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
    assert_eq!(
        refresh_cost.invalidated_at_index,
        invalidated.invalidated_at_index
    );
    assert!(refresh_cost.invalidated_by_memory_pressure);
}

#[test]
fn mutation_invalidates_only_the_mutated_table_residency() {
    // P1-M3 step 2: per-table residency invalidation. A write to one table must no
    // longer evict every other table's residency (the former stop-the-world bug).
    let mut e = Engine::new_local();
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
    assert_eq!(
        a.invalidated_by_txn_id,
        Some(5),
        "the mutated table is invalidated"
    );
    assert!(!a.is_valid());
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
    let mut e = Engine::new_local();
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
fn unscoped_ddl_conservatively_invalidates_unrelated_residency() {
    // A schema change is not (yet) scoped to a single table, so it conservatively
    // invalidates UNRELATED residency too rather than risk a stale snapshot.
    // Over-invalidation is safe; under-invalidation would serve wrong rows. (The
    // mutated table `a` has its snapshot rebuilt by the schema change itself, so we
    // observe the conservative fallback on the untouched table `b`.)
    let mut e = Engine::new_local();
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
    assert!(
        !b.is_valid(),
        "an unscoped DDL on table a must conservatively invalidate unrelated table b"
    );
    assert_eq!(b.invalidated_by_txn_id, Some(5));
}

#[test]
fn residency_invalidation_scope_narrows_dml_and_falls_back_on_unknown() {
    fn entry(index: Index, sql: &str) -> LogEntry {
        LogEntry {
            term: 1,
            index,
            payload: sql.as_bytes().to_vec(),
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
fn resident_snapshot_probe_reads_valid_snapshot_and_rejects_invalidated_state() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    let Command::Select(select) =
        parse_command("SELECT label FROM events WHERE id = 2 LIMIT 1").unwrap()
    else {
        panic!("expected SELECT");
    };
    let cpu = e.execute_relational_select(&select).unwrap();

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert!(snapshot.is_valid());
    assert_eq!(snapshot.resident_rows.len(), 2);

    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_select_with_resident_snapshot_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    assert!(e
        .execute_relational_select_with_resident_snapshot_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));

    e.populate_relational_residency_snapshot("events").unwrap();
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_select_with_resident_snapshot_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn resident_snapshot_probe_reads_aggregate_distinct_without_transfer() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (id INT, label TEXT, amount INT, category TEXT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount, category) VALUES (1, 'alpha', 10, 'odd'), (2, 'beta', 20, 'even'), (3, 'gamma', 30, 'odd')",
        )
        .unwrap();
    let queries = [
        "SELECT DISTINCT category FROM events ORDER BY category",
        "SELECT category, COUNT(*) FROM events GROUP BY category ORDER BY count DESC",
        "SELECT category, SUM(amount) FROM events GROUP BY category ORDER BY sum DESC",
        "SELECT AVG(amount) FROM events WHERE category = 'odd'",
        "SELECT MIN(amount) FROM events",
        "SELECT MAX(amount) FROM events",
    ];

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert!(snapshot.is_valid());

    for sql in queries {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_select_with_resident_snapshot_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        assert_eq!(after.d2h_bytes_total - before.d2h_bytes_total, 0, "{sql}");
        assert_eq!(
            after.kernel_exec_samples - before.kernel_exec_samples,
            0,
            "{sql}"
        );
    }
}

#[test]
fn resident_snapshot_budget_evicts_oldest_table_before_admission() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE small_a (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO small_a (id, label) VALUES (1, 'a')")
        .unwrap();
    let small_a = e.populate_relational_residency_snapshot("small_a").unwrap();

    e.execute_text(3, "CREATE TABLE small_b (id INT, label TEXT)")
        .unwrap();
    e.execute_text(4, "INSERT INTO small_b (id, label) VALUES (2, 'b')")
        .unwrap();
    let small_b = e.populate_relational_residency_snapshot("small_b").unwrap();

    let budget_bytes = small_b.resident_bytes.saturating_mul(2);
    e.set_relational_residency_budget_bytes(0, budget_bytes);
    e.execute_text(5, "CREATE TABLE small_c (id INT, label TEXT)")
        .unwrap();
    e.execute_text(6, "INSERT INTO small_c (id, label) VALUES (3, 'c')")
        .unwrap();
    let small_c = e.populate_relational_residency_snapshot("small_c").unwrap();

    assert_eq!(small_c.admission_budget_bytes, Some(budget_bytes));
    assert_eq!(small_c.evicted_tables_on_admission, vec!["small_a"]);
    assert!(e.relational_residency_snapshot("small_a").is_none());
    assert!(e.relational_residency_snapshot("small_b").is_some());
    assert!(e.relational_residency_snapshot("small_c").is_some());
    assert_eq!(
        small_c.resident_bytes_after_admission,
        e.relational_resident_bytes_for_gpu(0)
    );
    assert_eq!(small_a.valid_through_index, 2);
    assert_eq!(small_b.valid_through_index, 4);
}

#[test]
fn resident_snapshot_budget_rejects_oversized_snapshot_without_mutation() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO events (id, label) VALUES (1, 'alpha')")
        .unwrap();
    let original = e.populate_relational_residency_snapshot("events").unwrap();

    e.set_relational_residency_budget_bytes(0, original.resident_bytes - 1);
    let err = e
        .populate_relational_residency_snapshot("events")
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeding GPU 0 residency budget"));

    let retained = e.relational_residency_snapshot("events").unwrap();
    assert_eq!(retained, original);
    let status = e.status_snapshot();
    let table = status.relational_residency.table("events").unwrap();
    assert_eq!(table.cache_state, "Valid");
    assert_eq!(table.last_decision_accepted, Some(false));
    assert_eq!(
        table.last_decision_reason.as_deref(),
        Some("resident snapshot exceeds GPU budget")
    );
    assert_eq!(
        table.last_decision_current_bytes_before,
        Some(original.resident_bytes)
    );
    assert_eq!(
        table.last_decision_current_bytes_after,
        Some(original.resident_bytes)
    );
    assert_eq!(
        e.relational_resident_bytes_for_gpu(0),
        original.resident_bytes
    );
}

#[test]
fn resident_snapshot_budget_keeps_wal_and_pressure_invalidation_semantics() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    let original = e.populate_relational_residency_snapshot("events").unwrap();
    let budget_bytes = original.resident_bytes + 128;
    e.set_relational_residency_budget_bytes(0, budget_bytes);

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let invalidated = e.relational_residency_snapshot("events").unwrap();
    assert_eq!(invalidated.invalidated_by_txn_id, Some(3));
    assert!(!invalidated.is_valid());

    let refreshed = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(refreshed.admission_budget_bytes, Some(budget_bytes));
    assert!(refreshed.is_valid());

    e.mark_gpu_memory_pressured(0);
    let pressured = e.relational_residency_snapshot("events").unwrap();
    assert!(pressured.invalidated_by_memory_pressure);
    assert!(pressured.memory_pressure_active);
    assert!(!pressured.is_valid());
}

#[test]
fn resident_snapshot_records_absent_device_memory_proof_when_cuda_unavailable() {
    let mut e = Engine::new_local();
    let _ = e
        .cached_cuda_probe_runtime
        .set(CudaDriverRuntime::unavailable());
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO events (id, label) VALUES (1, 'alpha')")
        .unwrap();

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(snapshot.device_memory_proof, None);
    assert_eq!(e.read_state.residency.device_memory.len(), 0);

    let status = e.status_snapshot();
    assert_eq!(
        status
            .relational_residency
            .table("events")
            .unwrap()
            .device_memory_proof,
        None
    );

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(filtered_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_count_with_resident_device_memory_probe(&filtered_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(text_prefix_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'a%'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&text_prefix_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(membership_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id IN (1, 2)").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&membership_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(range_select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_range_count_with_resident_device_memory_probe(&range_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(sum_select) = parse_command("SELECT SUM(id) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_sum_with_resident_device_memory_probe(&sum_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(avg_select) = parse_command("SELECT AVG(id) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&avg_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(filtered_avg_select) =
        parse_command("SELECT AVG(id) FROM events WHERE id >= 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(
            &filtered_avg_select,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(grouped_sum_select) =
        parse_command("SELECT id, SUM(id) FROM events GROUP BY id").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_grouped_sum_with_resident_device_memory_probe(&grouped_sum_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(projection_select) =
        parse_command("SELECT id FROM events WHERE id >= 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_projection_with_resident_device_memory_probe(&projection_select)
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(ordered_projection_select) =
        parse_command("SELECT id FROM events WHERE id >= 1 ORDER BY id DESC LIMIT 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_ordered_projection_with_resident_device_memory_probe(
            &ordered_projection_select,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));

    let Command::Select(distinct_projection_select) =
        parse_command("SELECT DISTINCT id FROM events ORDER BY id").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(
            &distinct_projection_select,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("has no retained resident device memory"));
}

#[test]
fn gpu_resident_device_memory_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 20), (3, 'gamma', 30)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) =
        parse_command("SELECT amount FROM events WHERE amount >= 20").unwrap()
    else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        2 * std::mem::size_of::<i32>() as u64 + std::mem::size_of::<u64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    e.execute_text(
        3,
        "INSERT INTO events (id, label, amount) VALUES (4, 'delta', 40)",
    )
    .unwrap();
    assert!(e
        .execute_relational_projection_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_sum_probe_parallel_reduction_preserves_scalar_telemetry() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, amount INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, amount) VALUES (1, -5), (2, 0), (3, 7), (4, -2)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let Command::Select(select) = parse_command("SELECT SUM(amount) FROM events").unwrap() else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_sum_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(resident.rows, vec![vec![SqlValue::Int8(0)]]);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        std::mem::size_of::<i64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    e.execute_text(3, "CREATE TABLE empty_events (id INT, amount INT)")
        .unwrap();
    let empty_snapshot = e
        .populate_relational_residency_snapshot("empty_events")
        .unwrap();
    if empty_snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(empty_select) =
        parse_command("SELECT SUM(amount) FROM empty_events").unwrap()
    else {
        unreachable!()
    };
    let empty_cpu = e.execute_relational_select(&empty_select).unwrap();
    let before_empty = e.metrics().snapshot();
    let empty_resident = e
        .execute_relational_sum_with_resident_device_memory_probe(&empty_select)
        .unwrap();
    let after_empty = e.metrics().snapshot();

    assert_eq!(empty_resident.columns, empty_cpu.columns);
    assert_eq!(empty_resident.rows, empty_cpu.rows);
    assert_eq!(
        after_empty.h2d_bytes_total - before_empty.h2d_bytes_total,
        0
    );
    assert_eq!(
        after_empty.d2h_bytes_total - before_empty.d2h_bytes_total,
        std::mem::size_of::<i64>() as u64
    );
    assert_eq!(
        after_empty.kernel_exec_samples - before_empty.kernel_exec_samples,
        1
    );
}

#[test]
fn gpu_resident_device_memory_ordered_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (3, 'gamma', 20), (4, 'delta', 40)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) = parse_command(
        "SELECT amount FROM events WHERE amount >= 20 ORDER BY amount DESC LIMIT 2 OFFSET 1",
    )
    .unwrap() else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_ordered_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(30)], vec![SqlValue::Int4(20)]]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        2 * std::mem::size_of::<i32>() as u64 + std::mem::size_of::<u64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_ordered_projection_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_distinct_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, bucket INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, bucket) VALUES (1, 'alpha', 2), (2, 'beta', 1), (3, 'gamma', 2), (4, 'delta', 3), (5, 'epsilon', 1)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) =
        parse_command("SELECT DISTINCT bucket FROM events ORDER BY bucket DESC LIMIT 2 OFFSET 1")
            .unwrap()
    else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(1)]]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    // Kernel-less projection D2Hs only the i32 column (no device `out_count` readback): the
    // unfiltered distinct path reads exactly the value bytes. The `+ size_of::<u64>()` count term
    // was dropped in 38451a28 (which updated the plain-projection test but missed this DISTINCT
    // one); the filtered-distinct sibling still reads the u64 match-count, so it keeps the term.
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        5 * std::mem::size_of::<i32>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    let Command::Select(unsupported_text) =
        parse_command("SELECT DISTINCT label FROM events ORDER BY label").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(&unsupported_text)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 projection columns"));

    let Command::Select(unsupported_offset_without_order) =
        parse_command("SELECT DISTINCT bucket FROM events OFFSET 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(
            &unsupported_offset_without_order,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("OFFSET proof currently requires same-column ORDER BY and LIMIT"));

    let Command::Select(unsupported_filter) =
        parse_command("SELECT DISTINCT bucket FROM events WHERE bucket >= 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(
            &unsupported_filter,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("optional same-column ORDER BY, LIMIT, and OFFSET"));

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_filtered_distinct_projection_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, bucket INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, bucket) VALUES (1, 'alpha', 1), (2, 'beta', 2), (3, 'gamma', 2), (4, 'delta', 3), (5, 'epsilon', 4), (6, 'zeta', 4)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) = parse_command(
            "SELECT DISTINCT bucket FROM events WHERE bucket >= 2 ORDER BY bucket DESC LIMIT 2 OFFSET 1",
        )
        .unwrap() else {
            unreachable!()
        };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(
        resident.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(2)]]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        5 * std::mem::size_of::<i32>() as u64 + std::mem::size_of::<u64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    let Command::Select(unsupported_cross_column) = parse_command(
        "SELECT DISTINCT bucket FROM events WHERE id >= 2 ORDER BY bucket DESC LIMIT 2",
    )
    .unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
            &unsupported_cross_column,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires predicate and projection to use the same int4 column"));

    let Command::Select(unsupported_offset_without_order) =
        parse_command("SELECT DISTINCT bucket FROM events WHERE bucket >= 2 OFFSET 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
            &unsupported_offset_without_order,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("OFFSET proof currently requires same-column ORDER BY and LIMIT"));

    let Command::Select(unsupported_equality) =
        parse_command("SELECT DISTINCT bucket FROM events WHERE bucket = 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(
            &unsupported_equality,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only non-equality int4 comparisons"));

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_filtered_distinct_projection_with_resident_device_memory_probe(&select,)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn relational_select_grouped_having_filters_engine_aggregate_rows() {
    let e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5), (3, 'zeta', 15)",
        )
        .unwrap();

    let Command::Select(select) = parse_command(
            "SELECT bucket, SUM(amount) FROM events GROUP BY bucket HAVING bucket >= 2 AND sum > 40 ORDER BY sum DESC",
        )
        .unwrap() else {
            unreachable!()
        };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Int8(55)]]
    );

    let Command::Select(or_select) = parse_command(
            "SELECT bucket, MAX(amount) FROM events GROUP BY bucket HAVING bucket = 1 OR max >= 40 ORDER BY bucket",
        )
        .unwrap() else {
            unreachable!()
        };
    let result = e.execute_relational_select(&or_select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(20)],
            vec![SqlValue::Int4(3), SqlValue::Int4(40)]
        ]
    );

    let Command::Select(unsupported_having) =
        parse_command("SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING amount > 10")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_select(&unsupported_having)
        .unwrap_err()
        .to_string();
    assert!(err.contains("HAVING must reference grouped column or aggregate result"));
}

#[test]
fn gpu_resident_device_memory_grouped_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let Command::Select(select) = parse_command(
        "SELECT bucket, SUM(amount) FROM events GROUP BY bucket ORDER BY sum DESC LIMIT 2",
    )
    .unwrap() else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_grouped_sum_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(
        resident.rows,
        vec![
            vec![SqlValue::Int4(3), SqlValue::Int8(40)],
            vec![SqlValue::Int4(2), SqlValue::Int8(35)]
        ]
    );
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        3 * (std::mem::size_of::<i32>()
            + std::mem::size_of::<u64>()
            + std::mem::size_of::<i64>()
            + (2 * std::mem::size_of::<i32>())) as u64
            + std::mem::size_of::<u64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);

    for sql in [
            "SELECT bucket, COUNT(*) FROM events GROUP BY bucket ORDER BY count DESC LIMIT 2",
            "SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING count >= 2 ORDER BY bucket",
            "SELECT bucket, AVG(amount) FROM events GROUP BY bucket ORDER BY avg DESC LIMIT 2",
            "SELECT bucket, MIN(amount) FROM events GROUP BY bucket ORDER BY min DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events GROUP BY bucket ORDER BY max DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events GROUP BY bucket HAVING bucket = 1 OR max >= 40 ORDER BY bucket",
        ] {
            let Command::Select(select) = parse_command(sql).unwrap() else {
                unreachable!()
            };
            let cpu = e.execute_relational_select(&select).unwrap();
            let resident = e
                .execute_relational_grouped_aggregate_with_resident_device_memory_probe(&select)
                .unwrap();
            assert_eq!(resident.columns, cpu.columns, "{sql}");
            assert_eq!(resident.rows, cpu.rows, "{sql}");
            assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.fallback_reason, None);
        }

    let Command::Select(unsupported_having) =
        parse_command("SELECT bucket, SUM(amount) FROM events GROUP BY bucket HAVING amount > 10")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_grouped_aggregate_with_resident_device_memory_probe(&unsupported_having)
        .unwrap_err()
        .to_string();
    assert!(err.contains("HAVING must reference grouped column or aggregate result"));

    e.execute_text(
        3,
        "INSERT INTO events (bucket, label, amount) VALUES (4, 'zeta', 50)",
    )
    .unwrap();
    assert!(e
        .execute_relational_grouped_sum_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_filtered_grouped_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5), (3, 'zeta', 15)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
            "SELECT bucket, COUNT(*) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY count DESC LIMIT 2",
            "SELECT bucket, COUNT(*) FROM events WHERE amount >= 15 GROUP BY bucket HAVING count >= 2 ORDER BY bucket",
            "SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY sum DESC LIMIT 2",
            "SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket HAVING sum > 30 ORDER BY sum DESC",
            "SELECT bucket, AVG(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY avg DESC LIMIT 2",
            "SELECT bucket, MIN(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY min DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events WHERE amount >= 15 GROUP BY bucket ORDER BY max DESC LIMIT 2",
            "SELECT bucket, MAX(amount) FROM events WHERE amount >= 15 GROUP BY bucket HAVING bucket = 2 OR max >= 40 ORDER BY bucket",
        ] {
            let Command::Select(select) = parse_command(sql).unwrap() else {
                unreachable!()
            };
            let cpu = e.execute_relational_select(&select).unwrap();
            let before = e.metrics().snapshot();
            let resident = e
                .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
                    &select,
                )
                .unwrap();
            let after = e.metrics().snapshot();

            assert_eq!(resident.columns, cpu.columns, "{sql}");
            assert_eq!(resident.rows, cpu.rows, "{sql}");
            assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
            assert_eq!(resident.fallback_reason, None);
            assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
            assert!(
                after.d2h_bytes_total > before.d2h_bytes_total,
                "{sql} should read filtered grouped stats from device memory"
            );
            assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);
        }

    let Command::Select(unsupported_equality) =
        parse_command("SELECT bucket, SUM(amount) FROM events WHERE amount = 20 GROUP BY bucket")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
            &unsupported_equality,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only non-equality int4 comparisons"));

    let Command::Select(unsupported_text) = parse_command(
        "SELECT bucket, MAX(amount) FROM events WHERE label >= 'beta' GROUP BY bucket",
    )
    .unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
            &unsupported_text,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 comparison literals"));

    let Command::Select(unsupported_having) = parse_command(
            "SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket HAVING amount > 10",
        )
        .unwrap() else {
            unreachable!()
        };
    let err = e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(
            &unsupported_having,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("HAVING must reference grouped column or aggregate result"));

    let Command::Select(select) =
        parse_command("SELECT bucket, SUM(amount) FROM events WHERE amount >= 15 GROUP BY bucket")
            .unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_filtered_grouped_aggregate_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_membership_count_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (4, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
        "SELECT COUNT(*) FROM events WHERE id IN (1, 3, 99)",
        "SELECT COUNT(*) FROM events WHERE id IN (1, 1, 3)",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_membership_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.fallback_reason, None);
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    }

    let Command::Select(text_membership) =
        parse_command("SELECT COUNT(*) FROM events WHERE label IN ('alpha', 'delta')").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&text_membership)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 membership literals"));

    let Command::Select(cross_column) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 1 OR amount = 40").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&cross_column)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires all membership values to target the same column"));

    let Command::Select(equality_only) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 1").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_membership_count_with_resident_device_memory_probe(&equality_only)
        .unwrap_err()
        .to_string();
    assert!(err.contains("one int4 IN membership predicate"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id IN (1, 3)").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_relational_membership_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_between_count_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (3, 'gamma', 20), (4, 'delta', 40), (5, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
        "SELECT COUNT(*) FROM events WHERE id BETWEEN 2 AND 4",
        "SELECT COUNT(*) FROM events WHERE id BETWEEN 4 AND 2",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_between_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        assert!(after.d2h_bytes_total > before.d2h_bytes_total, "{sql}");
        assert_eq!(
            after.kernel_exec_samples - before.kernel_exec_samples,
            2,
            "{sql}"
        );
    }

    let Command::Select(cross_column) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 2 AND amount <= 40").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_count_with_resident_device_memory_probe(&cross_column)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires both range bounds to target the same column"));

    let Command::Select(text_bounds) =
        parse_command("SELECT COUNT(*) FROM events WHERE label >= 'beta' AND label <= 'gamma'")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_count_with_resident_device_memory_probe(&text_bounds)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 bounds"));

    let Command::Select(equality_only) =
        parse_command("SELECT COUNT(*) FROM events WHERE id = 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_count_with_resident_device_memory_probe(&equality_only)
        .unwrap_err()
        .to_string();
    assert!(err.contains("one int4 BETWEEN predicate"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id BETWEEN 2 AND 4").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_relational_between_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_filter_group_count_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (3, 'gamma', 20), (4, 'delta', 40), (5, 'epsilon', 5), (6, 'zeta', 60)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
        "SELECT COUNT(*) FROM events WHERE id >= 2 AND amount <= 40",
        "SELECT COUNT(*) FROM events WHERE (id = 1 AND amount >= 10) OR (id = 4 AND amount <= 40)",
        "SELECT COUNT(*) FROM events WHERE id <= 2 OR amount >= 50",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_filter_group_count_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        assert!(after.d2h_bytes_total > before.d2h_bytes_total, "{sql}");
        assert!(
            after.kernel_exec_samples > before.kernel_exec_samples,
            "{sql}"
        );
    }

    let Command::Select(unsupported_text_like) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'a%'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filter_group_count_with_resident_device_memory_probe(
            &unsupported_text_like,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 literal predicates"));

    let Command::Select(unsupported_ordered) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 2 ORDER BY count").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filter_group_count_with_resident_device_memory_probe(
            &unsupported_ordered,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("SELECT COUNT(*) with int4 WHERE filter groups"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE id >= 2 AND amount <= 40").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_relational_filter_group_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_text_prefix_count_probe_materializes_text_results() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT, amount INT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, label, amount) VALUES (1, 'alpha', 10), (2, 'alpine', 30), (3, 'beta', 20), (4, 'alphabet', 40), (5, 'gamma', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(
        snapshot
            .resident_device_text_columns
            .iter()
            .map(|layout| layout.name.as_str())
            .collect::<Vec<_>>(),
        vec!["label"]
    );
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'alp%'").unwrap()
    else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&select).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&select)
        .unwrap();
    let after = e.metrics().snapshot();

    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(resident.rows, vec![vec![SqlValue::Int8(3)]]);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert!(after.d2h_bytes_total > before.d2h_bytes_total);
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 0);

    let Command::Select(unsupported_int4) =
        parse_command("SELECT COUNT(*) FROM events WHERE id LIKE '1%'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&unsupported_int4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only text predicates"));

    let Command::Select(unsupported_ordered) =
        parse_command("SELECT COUNT(*) FROM events WHERE label LIKE 'alp%' ORDER BY count")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(
            &unsupported_ordered,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("one text prefix LIKE predicate"));

    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_text_prefix_count_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_scalar_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
        "SELECT AVG(amount) FROM events",
        "SELECT MIN(amount) FROM events",
        "SELECT MAX(amount) FROM events",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.fallback_reason, None);
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
        assert!(
            after.d2h_bytes_total > before.d2h_bytes_total,
            "{sql} should read aggregate stats from device memory"
        );
        assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);
    }

    let Command::Select(unsupported) = parse_command("SELECT AVG(label) FROM events").unwrap()
    else {
        unreachable!()
    };
    assert!(e
        .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&unsupported)
        .unwrap_err()
        .to_string()
        .contains("AVG only supports int4 columns"));

    e.mark_gpu_memory_pressured(0);
    let Command::Select(select) = parse_command("SELECT MAX(amount) FROM events").unwrap() else {
        unreachable!()
    };
    assert!(e
        .execute_relational_scalar_aggregate_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_filtered_scalar_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
        "SELECT SUM(amount) FROM events WHERE amount >= 20",
        "SELECT AVG(amount) FROM events WHERE amount >= 20",
        "SELECT MIN(amount) FROM events WHERE amount >= 20",
        "SELECT MAX(amount) FROM events WHERE amount >= 20",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(resident.fallback_reason, None);
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
        assert_eq!(
            after.d2h_bytes_total - before.d2h_bytes_total,
            (std::mem::size_of::<u64>()
                + std::mem::size_of::<i64>()
                + (2 * std::mem::size_of::<i32>())
                + std::mem::size_of::<u64>()) as u64,
            "{sql}"
        );
        assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 1);
    }

    let Command::Select(unsupported_text) =
        parse_command("SELECT MAX(label) FROM events WHERE label >= 'beta'").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(
            &unsupported_text,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 comparison literals"));

    let Command::Select(unsupported_cross_column) =
        parse_command("SELECT SUM(amount) FROM events WHERE bucket >= 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(
            &unsupported_cross_column,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires the predicate column to match the aggregate column"));

    let Command::Select(empty_max) =
        parse_command("SELECT MAX(amount) FROM events WHERE amount >= 1000").unwrap()
    else {
        unreachable!()
    };
    let cpu = e.execute_relational_select(&empty_max).unwrap();
    let before = e.metrics().snapshot();
    let resident = e
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(&empty_max)
        .unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(resident.columns, cpu.columns);
    assert_eq!(resident.rows, cpu.rows);
    assert_eq!(resident.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(resident.fallback_reason, None);
    assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0);
    assert_eq!(
        after.d2h_bytes_total - before.d2h_bytes_total,
        std::mem::size_of::<i64>() as u64
    );
    assert_eq!(after.kernel_exec_samples - before.kernel_exec_samples, 0);

    let Command::Select(select) =
        parse_command("SELECT SUM(amount) FROM events WHERE amount >= 20").unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn gpu_resident_device_memory_between_scalar_aggregate_probe_materializes_int4_results() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (bucket INT, label TEXT, amount INT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (bucket, label, amount) VALUES (1, 'alpha', 10), (2, 'beta', 30), (1, 'gamma', 20), (3, 'delta', 40), (2, 'epsilon', 5)",
        )
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    for sql in [
        "SELECT SUM(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT AVG(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT MIN(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT MAX(amount) FROM events WHERE amount BETWEEN 10 AND 30",
        "SELECT SUM(amount) FROM events WHERE amount BETWEEN 40 AND 10",
    ] {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let cpu = e.execute_relational_select(&select).unwrap();
        let before = e.metrics().snapshot();
        let resident = e
            .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(&select)
            .unwrap();
        let after = e.metrics().snapshot();

        assert_eq!(resident.columns, cpu.columns, "{sql}");
        assert_eq!(resident.rows, cpu.rows, "{sql}");
        assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(resident.fallback_reason, None, "{sql}");
        assert_eq!(after.h2d_bytes_total - before.h2d_bytes_total, 0, "{sql}");
        if sql.contains("40 AND 10") {
            assert_eq!(after.d2h_bytes_total - before.d2h_bytes_total, 0, "{sql}");
            assert_eq!(
                after.kernel_exec_samples - before.kernel_exec_samples,
                0,
                "{sql}"
            );
        } else {
            assert_eq!(
                after.d2h_bytes_total - before.d2h_bytes_total,
                (std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>())
                    + std::mem::size_of::<u64>()) as u64,
                "{sql}"
            );
            assert_eq!(
                after.kernel_exec_samples - before.kernel_exec_samples,
                1,
                "{sql}"
            );
        }
    }

    let Command::Select(unsupported_text) =
        parse_command("SELECT MAX(label) FROM events WHERE label BETWEEN 'beta' AND 'gamma'")
            .unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(
            &unsupported_text,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("supports only int4 bounds"));

    let Command::Select(unsupported_cross_column) =
        parse_command("SELECT SUM(amount) FROM events WHERE bucket BETWEEN 1 AND 2").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(
            &unsupported_cross_column,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires the predicate column to match the aggregate column"));

    let Command::Select(equality_only) =
        parse_command("SELECT SUM(amount) FROM events WHERE amount = 20").unwrap()
    else {
        unreachable!()
    };
    let err = e
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(
            &equality_only,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("one int4 BETWEEN predicate"));

    let Command::Select(select) =
        parse_command("SELECT SUM(amount) FROM events WHERE amount BETWEEN 10 AND 30").unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    assert!(e
        .execute_relational_between_scalar_aggregate_with_resident_device_memory_probe(&select)
        .unwrap_err()
        .to_string()
        .contains("resident snapshot is invalid"));
}

#[test]
fn status_and_telemetry_surface_relational_residency_state() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE aux (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO aux (id, label) VALUES (1, 'aux')")
        .unwrap();

    let events = e.populate_relational_residency_snapshot("events").unwrap();
    if let Some(proof) = &events.device_memory_proof {
        assert!(proof.retained);
    }
    let aux = e.populate_relational_residency_snapshot("aux").unwrap();
    let budget_bytes = events.resident_bytes;
    e.set_relational_residency_budget_bytes(0, budget_bytes);
    let admitted = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(admitted.evicted_tables_on_admission, vec!["aux"]);

    let status = e.status_snapshot();
    assert_eq!(status.resident_table_count(), 1);
    assert_eq!(status.relational_residency.snapshot_count(), 1);
    assert_eq!(status.relational_residency.valid_snapshot_count(), 1);
    assert_eq!(
        status.relational_residency.total_resident_bytes(),
        events.resident_bytes
    );
    assert_eq!(
        status.relational_residency.budget_bytes_by_gpu.get(&0),
        Some(&budget_bytes)
    );
    assert_eq!(
        status.relational_residency.resident_bytes_by_gpu.get(&0),
        Some(&events.resident_bytes)
    );
    let table = status.relational_residency.table("events").unwrap();
    assert_eq!(table.schema, "public");
    assert_eq!(table.gpu_id, 0);
    assert_eq!(table.row_count, 2);
    assert_eq!(table.column_count, 2);
    assert_eq!(table.resident_bytes, events.resident_bytes);
    assert_eq!(table.admission_budget_bytes, Some(budget_bytes));
    assert_eq!(table.resident_bytes_after_admission, events.resident_bytes);
    assert_eq!(table.evicted_tables_on_admission, vec!["aux"]);
    assert_eq!(table.cache_state, "Valid");
    assert_eq!(table.last_decision_accepted, Some(true));
    assert_eq!(
        table.last_decision_reason.as_deref(),
        Some("admitted after deterministic eviction")
    );
    assert_eq!(
        table.last_decision_current_bytes_before,
        Some(aux.resident_bytes)
    );
    assert_eq!(
        table.last_decision_current_bytes_after,
        Some(events.resident_bytes)
    );
    assert!(table.valid);
    assert!(status.relational_residency.table("aux").is_none());
    status.validate().unwrap();

    e.execute_text(5, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let invalidated = e.telemetry_snapshot();
    let invalidated_table = invalidated.relational_residency.table("events").unwrap();
    assert_eq!(invalidated.resident_table_count(), 1);
    assert_eq!(invalidated_table.cache_state, "Invalidated");
    assert!(!invalidated_table.valid);
    assert_eq!(invalidated_table.invalidated_by_txn_id, Some(5));
    assert_eq!(invalidated.relational_residency.invalid_snapshot_count(), 1);
    if let Some(proof) = &invalidated_table.device_memory_proof {
        assert!(!proof.retained);
    }

    e.mark_gpu_memory_pressured(0);
    let pressured = e.status_snapshot();
    let pressured_table = pressured.relational_residency.table("events").unwrap();
    assert_eq!(pressured_table.cache_state, "InvalidatedByMemoryPressure");
    assert!(pressured_table.memory_pressure_active);
    assert!(pressured_table.invalidated_by_memory_pressure);
    if let Some(proof) = &pressured_table.device_memory_proof {
        assert!(!proof.retained);
    }
    assert_eq!(
        pressured
            .relational_residency
            .memory_pressured_snapshot_count(),
        1
    );
    assert_eq!(
        e.relational_resident_bytes_for_gpu(0),
        events.resident_bytes
    );
    assert!(aux.resident_bytes > 0);
}

#[test]
fn p8_resident_route_decisions_use_cache_state_and_default_fallbacks() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();

    let Command::Select(count_select) = parse_command("SELECT COUNT(*) FROM events").unwrap()
    else {
        unreachable!()
    };
    let absent = e.plan_relational_resident_route(&count_select);
    assert!(!absent.accepted);
    assert_eq!(absent.reason, "relation has no resident snapshot");
    assert_eq!(absent.query_shape, "count_all");
    assert_eq!(absent.d2h_bytes_estimate, 0);
    assert_eq!(absent.last_execution_h2d_bytes, None);
    assert_eq!(absent.last_execution_d2h_bytes, None);
    assert_eq!(absent.last_execution_kernel_samples, None);
    assert_eq!(absent.last_execution_kernel_ms, None);
    assert_eq!(absent.last_execution_kernel_event_elapsed_us, None);
    assert_eq!(absent.last_execution_rows, None);

    let snapshot = e.populate_relational_residency_snapshot("events").unwrap();
    let decision = e.plan_relational_resident_route(&count_select);
    assert_eq!(decision.table, "events");
    assert_eq!(decision.gpu_id, Some(0));
    assert_eq!(decision.query_shape, "count_all");
    assert_eq!(decision.cache_state, "Valid");
    assert!(decision.valid);
    assert_eq!(decision.estimated_rows, 2);
    assert_eq!(decision.resident_bytes, snapshot.resident_bytes);
    assert_eq!(decision.h2d_bytes_if_resident, 0);
    assert_eq!(decision.h2d_bytes_if_cold, snapshot.resident_bytes);
    assert_eq!(
        decision.d2h_bytes_estimate,
        std::mem::size_of::<u64>() as u64
    );
    assert_eq!(decision.d2h_rows_estimate, 1);
    if snapshot.device_memory_proof.is_some() {
        assert!(decision.accepted);
        assert!(decision.has_retained_device_memory);
        assert_eq!(decision.reason, "resident route accepted");
    } else {
        assert!(!decision.accepted);
        assert!(!decision.has_retained_device_memory);
        assert_eq!(
            decision.reason,
            "resident snapshot has no retained device memory"
        );
    }
    assert_eq!(
        e.status_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap(),
        &decision
    );

    let normal = e.execute_relational_select(&count_select).unwrap();
    assert_eq!(normal.planned_target, DeviceTarget::Gpu(0));
    if decision.accepted {
        assert_eq!(normal.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(normal.fallback_reason, None);
    } else {
        assert_eq!(
            normal.fallback_reason,
            Some(FallbackReason::GpuMvccReadParityGap)
        );
    }

    let Command::Select(unsupported) = parse_command("SELECT * FROM events").unwrap() else {
        unreachable!()
    };
    let unsupported = e.plan_relational_resident_route(&unsupported);
    assert!(!unsupported.accepted);
    assert_eq!(unsupported.query_shape, "unsupported_select");
    assert_eq!(unsupported.d2h_bytes_estimate, 0);
    assert_eq!(unsupported.last_execution_h2d_bytes, None);
    assert_eq!(unsupported.last_execution_d2h_bytes, None);
    assert_eq!(unsupported.last_execution_kernel_samples, None);
    assert_eq!(unsupported.last_execution_kernel_ms, None);
    assert_eq!(unsupported.last_execution_kernel_event_elapsed_us, None);
    assert_eq!(unsupported.last_execution_rows, None);
    assert_eq!(
        unsupported.reason,
        "resident routing has no retained-kernel proof for this SELECT shape"
    );

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&count_select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident snapshot is Invalidated");
}

#[test]
fn p8_default_resident_route_executes_accepted_shapes() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (id INT, bucket INT, amount INT, label TEXT)",
    )
    .unwrap();
    // Bucket sums are kept distinct (b1=30, b2=40) so `ORDER BY sum DESC LIMIT 1`
    // below is unambiguous. The GPU sum-order path has no deterministic tie-break
    // for equal sums yet (tracked §9.5 follow-up: "sum-tie 2-key gather"), so a tie
    // would make the cross-path parity check non-deterministic/flaky.
    e.execute_text(
            2,
            "INSERT INTO events (id, bucket, amount, label) VALUES (1, 1, 10, 'alpha'), (2, 1, 20, 'beta'), (3, 2, 40, 'alpine')",
        )
        .unwrap();
    e.populate_relational_residency_snapshot("events").unwrap();

    let Command::Select(count_select) = parse_command("SELECT COUNT(*) FROM events").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&count_select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(route.h2d_bytes_if_resident, 0);

    for sql in [
            "SELECT COUNT(*) FROM events",
            "SELECT COUNT(*) FROM events WHERE id = 2",
            "SELECT COUNT(*) FROM events WHERE id > 1",
            "SELECT COUNT(*) FROM events WHERE label LIKE 'al%'",
            "SELECT COUNT(*) FROM events WHERE id = 1 OR id = 3",
            "SELECT SUM(id) FROM events",
            "SELECT AVG(id) FROM events",
            "SELECT MIN(id) FROM events",
            "SELECT MAX(id) FROM events",
            "SELECT SUM(id) FROM events WHERE id > 1",
            "SELECT AVG(id) FROM events WHERE id < 3",
            "SELECT MIN(id) FROM events WHERE id >= 2",
            "SELECT MAX(id) FROM events WHERE id <= 2",
            "SELECT SUM(id) FROM events WHERE id BETWEEN 1 AND 2",
            "SELECT AVG(id) FROM events WHERE id BETWEEN 2 AND 3",
            "SELECT MIN(id) FROM events WHERE id BETWEEN 1 AND 3",
            "SELECT MAX(id) FROM events WHERE id BETWEEN 1 AND 1",
            "SELECT id FROM events WHERE id > 1",
            "SELECT id FROM events WHERE id >= 1 ORDER BY id DESC LIMIT 2 OFFSET 1",
            "SELECT DISTINCT bucket FROM events ORDER BY bucket DESC LIMIT 2 OFFSET 1",
            "SELECT DISTINCT bucket FROM events WHERE bucket >= 2 ORDER BY bucket DESC LIMIT 2 OFFSET 1",
            "SELECT bucket, COUNT(*) FROM events GROUP BY bucket HAVING count >= 1 ORDER BY bucket",
            "SELECT bucket, SUM(amount) FROM events GROUP BY bucket HAVING sum > 20 ORDER BY sum DESC LIMIT 1",
            "SELECT bucket, AVG(amount) FROM events GROUP BY bucket ORDER BY bucket",
            "SELECT bucket, MIN(amount) FROM events WHERE amount >= 20 GROUP BY bucket HAVING min >= 20 ORDER BY bucket",
            "SELECT bucket, MAX(amount) FROM events WHERE amount > 10 GROUP BY bucket HAVING bucket = 1 OR max >= 30 ORDER BY max DESC",
        ] {
            let Command::Select(select) = parse_command(sql).unwrap() else {
                unreachable!()
            };
            let expected = e
                .execute_relational_select_with_cuda_driver_probe(&select)
                .unwrap();
            let resident = e
                .execute_relational_select_with_resident_route(&select)
                .unwrap_or_else(|err| panic!("{sql}: {err}"));
            let before_default_metrics = e.metrics().snapshot();
            let default = e
                .execute_relational_select(&select)
                .unwrap_or_else(|err| panic!("{sql}: {err}"));
            let after_default_metrics = e.metrics().snapshot();
            assert_eq!(resident.rows, expected.rows, "{sql}");
            assert_eq!(resident.columns, expected.columns, "{sql}");
            assert_eq!(default.rows, expected.rows, "{sql}");
            assert_eq!(default.columns, expected.columns, "{sql}");
            assert_eq!(resident.planned_target, DeviceTarget::Gpu(0), "{sql}");
            assert_eq!(resident.executed_target, DeviceTarget::Gpu(0), "{sql}");
            assert_eq!(resident.fallback_reason, None, "{sql}");
            assert_eq!(default.planned_target, DeviceTarget::Gpu(0), "{sql}");
            assert_eq!(default.executed_target, DeviceTarget::Gpu(0), "{sql}");
            assert_eq!(default.fallback_reason, None, "{sql}");
            assert_eq!(
                e.status_snapshot()
                    .relational_residency
                    .latest_route_decision("events")
                    .unwrap()
                    .h2d_bytes_if_resident,
                0,
                "{sql}"
            );
            let route_decision = e
                .status_snapshot()
                .relational_residency
                .latest_route_decision("events")
                .unwrap()
                .clone();
            let expected_d2h_bytes = match route_decision.query_shape.as_str() {
                "count_all" | "int4_equality_count" | "int4_range_count" | "int4_filter_group_count" => {
                    std::mem::size_of::<u64>() as u64
                }
                "text_prefix_like_count" => e
                    .relational_residency_snapshot("events")
                    .unwrap()
                    .resident_bytes,
                "int4_scalar_aggregate"
                    if matches!(select.projection, SelectProjection::Sum { .. }) =>
                {
                    std::mem::size_of::<i64>() as u64
                }
                // Ungrouped scalar aggregate copies a grouped-stats struct (group key +
                // count + sum + min/max) + result length.
                "int4_scalar_aggregate" => {
                    (std::mem::size_of::<i32>()
                        + std::mem::size_of::<u64>()
                        + std::mem::size_of::<i64>()
                        + (2 * std::mem::size_of::<i32>())
                        + std::mem::size_of::<u64>()) as u64
                }
                // Filtered scalar aggregate copies a scalar-stats struct (count + sum +
                // min/max, no group key) + result length — matches the actual D2H in
                // execute_relational_filtered_scalar_aggregate_with_resident_device_memory_probe
                // and resident_route_d2h_bytes_estimate.
                "int4_filtered_scalar_aggregate" => {
                    (std::mem::size_of::<u64>()
                        + std::mem::size_of::<i64>()
                        + (2 * std::mem::size_of::<i32>())
                        + std::mem::size_of::<u64>()) as u64
                }
                "int4_between_scalar_aggregate" => {
                    (std::mem::size_of::<u64>()
                        + std::mem::size_of::<i64>()
                        + (2 * std::mem::size_of::<i32>())
                        + std::mem::size_of::<u64>()) as u64
                }
                "int4_grouped_aggregate" | "int4_filtered_grouped_aggregate" => route_decision
                    .d2h_rows_estimate
                    .checked_mul(
                        std::mem::size_of::<i32>()
                            + std::mem::size_of::<u64>()
                            + std::mem::size_of::<i64>()
                            + (2 * std::mem::size_of::<i32>()),
                    )
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .unwrap_or(u64::MAX),
                "int4_projection" | "int4_ordered_projection" => route_decision
                    .d2h_rows_estimate
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .unwrap_or(u64::MAX),
                "int4_distinct_projection" | "int4_filtered_distinct_projection" => e
                    .relational_residency_snapshot("events")
                    .unwrap()
                    .row_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .unwrap_or(u64::MAX),
                other => panic!("unexpected resident route shape {other} for {sql}"),
            };
            assert_eq!(
                route_decision.d2h_bytes_estimate, expected_d2h_bytes,
                "{sql}"
            );
            assert_eq!(route_decision.last_execution_h2d_bytes, Some(0), "{sql}");
            assert_eq!(
                route_decision.last_execution_d2h_bytes,
                Some(
                    after_default_metrics
                        .d2h_bytes_total
                        .saturating_sub(before_default_metrics.d2h_bytes_total)
                ),
                "{sql}"
            );
            assert_eq!(
                route_decision.last_execution_kernel_samples,
                Some(
                    after_default_metrics
                        .kernel_exec_samples
                        .saturating_sub(before_default_metrics.kernel_exec_samples)
                ),
                "{sql}"
            );
            assert_eq!(
                route_decision.last_execution_kernel_ms,
                Some(
                    after_default_metrics
                        .kernel_exec_total_ms
                        .saturating_sub(before_default_metrics.kernel_exec_total_ms)
                ),
                "{sql}"
            );
            // Kernel-event timing is recorded only by shapes that actually launch a GPU
            // kernel. Some resident routes are GPU-resident but run no kernel — e.g.
            // text-prefix count and distinct/projection paths finalize on the CPU after a
            // D2H copy — so they record no kernel-event time and add no timing sample.
            // Assert telemetry consistency by what the route observed rather than by
            // hardcoding per-shape: if a kernel timed this execution, the route decision
            // matches the engine's last kernel-event metric and exactly one timing sample
            // lands; otherwise neither moves. (On GPU-less hosts no shape runs a kernel,
            // so every iteration takes the else branch — keeping CI green.)
            let kernel_event_timing_delta = after_default_metrics
                .kernel_event_timing_samples
                .saturating_sub(before_default_metrics.kernel_event_timing_samples);
            if route_decision
                .last_execution_kernel_event_elapsed_us
                .is_some()
            {
                assert_eq!(
                    route_decision.last_execution_kernel_event_elapsed_us,
                    after_default_metrics.last_kernel_event_elapsed_us,
                    "{sql}"
                );
                assert_eq!(kernel_event_timing_delta, 1, "{sql}");
            } else {
                assert_eq!(kernel_event_timing_delta, 0, "{sql}");
            }
            assert_eq!(
                route_decision.last_execution_rows,
                Some(default.rows.len()),
                "{sql}"
            );
            assert!(
                matches!(
                    route_decision.query_shape.as_str(),
                    "count_all"
                        | "int4_equality_count"
                        | "int4_range_count"
                        | "text_prefix_like_count"
                        | "int4_filter_group_count"
                        | "int4_scalar_aggregate"
                        | "int4_filtered_scalar_aggregate"
                        | "int4_between_scalar_aggregate"
                        | "int4_grouped_aggregate"
                        | "int4_filtered_grouped_aggregate"
                        | "int4_projection"
                        | "int4_ordered_projection"
                        | "int4_distinct_projection"
                        | "int4_filtered_distinct_projection"
                ),
                "{sql}"
            );
        }

    for sql in [
        "SELECT DISTINCT label FROM events ORDER BY label",
        "SELECT DISTINCT bucket FROM events WHERE id >= 2 ORDER BY bucket DESC LIMIT 2",
        "SELECT DISTINCT bucket FROM events WHERE bucket = 2",
        "SELECT DISTINCT bucket FROM events OFFSET 1",
        "SELECT id FROM events WHERE id >= 1 ORDER BY id DESC",
        "SELECT id FROM events WHERE id = 1 ORDER BY id DESC LIMIT 1",
        "SELECT id FROM events WHERE amount >= 10 ORDER BY id DESC LIMIT 1",
        "SELECT label FROM events WHERE label LIKE 'a%' ORDER BY label LIMIT 1",
    ] {
        let Command::Select(unsupported) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        let unsupported = e.plan_relational_resident_route(&unsupported);
        assert!(!unsupported.accepted, "{sql}");
        assert_eq!(unsupported.query_shape, "unsupported_select", "{sql}");
        assert_eq!(
            unsupported.reason,
            "resident routing has no retained-kernel proof for this SELECT shape",
            "{sql}"
        );
    }

    let Command::Select(unsupported_group_filter) =
        parse_command("SELECT bucket, COUNT(*) FROM events WHERE amount = 20 GROUP BY bucket")
            .unwrap()
    else {
        unreachable!()
    };
    let unsupported = e.plan_relational_resident_route(&unsupported_group_filter);
    assert!(!unsupported.accepted);
    assert_eq!(unsupported.query_shape, "unsupported_select");
    assert_eq!(
        unsupported.reason,
        "resident routing has no retained-kernel proof for this SELECT shape"
    );
}

#[test]
fn p8_resident_route_batches_int4_equality_projection_literals() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE events (id INT, bucket INT, amount INT, label TEXT)",
    )
    .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, bucket, amount, label) VALUES (1, 1, 10, 'alpha'), (2, 1, 20, 'beta'), (3, 2, 30, 'alpine')",
        )
        .unwrap();
    e.populate_relational_residency_snapshot("events").unwrap();

    let selects = [
        "SELECT id, bucket, amount FROM events WHERE id = 1",
        "SELECT id, bucket, amount FROM events WHERE id = 3",
    ]
    .into_iter()
    .map(|sql| {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        select
    })
    .collect::<Vec<_>>();
    let route = e.plan_relational_resident_route(&selects[0]);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }

    let before = e.metrics().snapshot();
    let results = e
            .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
                &selects,
            )
            .unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].rows,
        vec![vec![
            SqlValue::Int4(1),
            SqlValue::Int4(1),
            SqlValue::Int4(10)
        ]]
    );
    assert_eq!(
        results[1].rows,
        vec![vec![
            SqlValue::Int4(3),
            SqlValue::Int4(2),
            SqlValue::Int4(30)
        ]]
    );
    for result in &results {
        assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
    }
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_multi_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_kernel_samples,
        Some(
            after
                .kernel_exec_samples
                .saturating_sub(before.kernel_exec_samples)
        )
    );
    assert_eq!(decision.last_execution_kernel_samples, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(2));

    let single_column_selects = [
        "SELECT id FROM events WHERE id = 1",
        "SELECT id FROM events WHERE id = 3",
    ]
    .into_iter()
    .map(|sql| {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        select
    })
    .collect::<Vec<_>>();
    let single_column_results = e
            .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
                &single_column_selects,
            )
            .unwrap();
    assert_eq!(single_column_results[0].rows, vec![vec![SqlValue::Int4(1)]]);
    assert_eq!(single_column_results[1].rows, vec![vec![SqlValue::Int4(3)]]);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "int4_equality_projection");
    assert_eq!(decision.last_execution_kernel_samples, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(2));
    let single_column_read_jobs = single_column_selects
        .iter()
        .map(|select| e.prepare_relational_retained_read_job(select))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let single_column_submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
            &single_column_read_jobs,
        )
        .unwrap();
    assert_eq!(
        single_column_submission.job_count,
        single_column_read_jobs.len()
    );
    assert!(single_column_submission.submit_wall_micros > 0);
    let single_column_read_job_results = e
        .complete_relational_retained_read_submission(single_column_submission)
        .unwrap();
    assert_eq!(single_column_read_job_results, single_column_results);

    let mixed_column_selects = [
        "SELECT id, label FROM events WHERE id = 1",
        "SELECT id, label FROM events WHERE id = 3",
    ]
    .into_iter()
    .map(|sql| {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            unreachable!()
        };
        select
    })
    .collect::<Vec<_>>();
    let mixed_column_results = e
            .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
                &mixed_column_selects,
            )
            .unwrap();
    assert_eq!(
        mixed_column_results[0].rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("alpha".into())]]
    );
    assert_eq!(
        mixed_column_results[1].rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("alpine".into())]]
    );
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_mixed_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_kernel_samples, Some(3));
    assert_eq!(decision.last_execution_matched_rows, Some(2));

    let read_jobs = mixed_column_selects
        .iter()
        .map(|select| e.prepare_relational_retained_read_job(select))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(read_jobs.len(), 2);
    assert_eq!(read_jobs[0].snapshot_generation, 1);
    assert!(
        read_jobs[0]
            .route_id
            .starts_with("int4_equality_mixed_column_projection:public:events:id,label:id"),
        "{}",
        read_jobs[0].route_id
    );
    assert_eq!(
        read_jobs[0].params,
        vec![RelationalRetainedReadParam::Int4Eq {
            column: "id".to_string(),
            value: 1
        }]
    );
    let submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&read_jobs)
        .unwrap();
    assert_eq!(submission.route_id, read_jobs[0].route_id);
    assert_eq!(
        submission.snapshot_generation,
        read_jobs[0].snapshot_generation
    );
    assert_eq!(submission.job_count, read_jobs.len());
    assert!(submission.submit_wall_micros > 0);
    let read_job_results = e
        .complete_relational_retained_read_submission(submission)
        .unwrap();
    assert_eq!(read_job_results, mixed_column_results);

    e.execute_text(
        3,
        "INSERT INTO events (id, bucket, amount, label) VALUES (4, 2, 40, 'amber')",
    )
    .unwrap();
    e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    let stale_job = e
        .execute_relational_retained_read_jobs_with_resident_device_memory_probe(&read_jobs)
        .unwrap_err();
    assert!(
        stale_job
            .to_string()
            .contains("snapshot generation mismatch"),
        "{stale_job}"
    );
}

#[test]
fn p8_resident_route_executes_same_column_equality_projection() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, amount INT, label TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO events (id, amount, label) VALUES (1, 10, 'alpha'), (2, 20, 'beta'), (2, 30, 'delta'), (3, 40, 'gamma')",
        )
        .unwrap();
    e.populate_relational_residency_snapshot("events").unwrap();

    let Command::Select(select) = parse_command("SELECT id FROM events WHERE id = 2").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert_eq!(route.query_shape, "int4_equality_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }

    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&select)
        .expect("same-column equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(2)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "int4_equality_projection");
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(2));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );

    let Command::Select(multi_column) =
        parse_command("SELECT id, amount FROM events WHERE id = 2").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&multi_column);
    assert_eq!(route.query_shape, "int4_equality_multi_column_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&multi_column)
        .expect("multi-column equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
            vec![SqlValue::Int4(2), SqlValue::Int4(30)]
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_multi_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(2));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some((2 * 2 * std::mem::size_of::<i32>() + std::mem::size_of::<u32>()) as u64)
    );

    let Command::Select(composite) =
        parse_command("SELECT id, amount FROM events WHERE id = 2 AND amount = 30").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&composite);
    assert_eq!(
        route.query_shape,
        "int4_composite_equality_multi_column_projection"
    );
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&composite)
        .expect("composite equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Int4(30)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_composite_equality_multi_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some((2 * std::mem::size_of::<i32>() + std::mem::size_of::<u32>()) as u64)
    );

    let Command::Select(mixed_composite) =
        parse_command("SELECT id, amount, label FROM events WHERE id = 3 AND amount = 40").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&mixed_composite);
    assert_eq!(route.query_shape, "int4_equality_mixed_column_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&mixed_composite)
        .expect("mixed int4/text equality projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![
            SqlValue::Int4(3),
            SqlValue::Int4(40),
            SqlValue::Text("gamma".to_string())
        ]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_mixed_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(
            (2 * std::mem::size_of::<i32>()
                + std::mem::size_of::<u64>()
                + 2 * std::mem::size_of::<u64>()
                + "gamma".len()
                + std::mem::size_of::<u64>()
                + std::mem::size_of::<u64>()) as u64
        )
    );

    // Single-predicate mixed int4+text projection. The dispatcher delegates this shape to the
    // fused batch path (`..._batch_inner`); the route-execution telemetry observation must be
    // recorded EXACTLY ONCE for the delegation (the dispatcher owns it; the delegated batch
    // path suppresses its own), not double-counted. Assert via the observation counter.
    let Command::Select(mixed_single) =
        parse_command("SELECT id, amount, label FROM events WHERE id = 3").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&mixed_single);
    assert_eq!(route.query_shape, "int4_equality_mixed_column_projection");
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    let observations_before = e.route_execution_observation_count();
    let before = e.metrics().snapshot();
    let result = e
        .execute_relational_select(&mixed_single)
        .expect("single-predicate mixed int4/text projection should use resident route");
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![
            SqlValue::Int4(3),
            SqlValue::Int4(40),
            SqlValue::Text("gamma".to_string())
        ]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    // The delegation records the route-execution observation exactly once (no double-count).
    assert_eq!(
        e.route_execution_observation_count()
            .saturating_sub(observations_before),
        1,
        "single-predicate mixed int4+text route must record its execution observation exactly once"
    );
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("events")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "int4_equality_mixed_column_projection"
    );
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(decision.last_execution_rows, Some(1));
    // The stored d2h-bytes observation reflects the (single) delegated batch execution.
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
}

#[test]
fn p8_opt_in_resident_route_rejects_before_execution() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();

    let Command::Select(absent) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_select_with_resident_route(&absent)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("relation has no resident snapshot"));

    e.populate_relational_residency_snapshot("events").unwrap();
    let Command::Select(unsupported) = parse_command("SELECT * FROM events").unwrap() else {
        unreachable!()
    };
    let err = e
        .execute_relational_select_with_resident_route(&unsupported)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("resident routing has no retained-kernel proof"));

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    let err = e
        .execute_relational_select_with_resident_route(&absent)
        .unwrap_err();
    assert!(err.to_string().contains("resident snapshot is Invalidated"));
    assert_eq!(
        e.telemetry_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap()
            .cache_state,
        "Invalidated"
    );
    let fallback = e.execute_relational_select(&absent).unwrap();
    assert_eq!(fallback.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(
        fallback.fallback_reason,
        Some(FallbackReason::GpuMvccReadParityGap)
    );
}

#[test]
fn p8_resident_route_decisions_reject_evicted_and_memory_pressured_snapshots() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE aux (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO aux (id, label) VALUES (1, 'aux')")
        .unwrap();

    let aux = e.populate_relational_residency_snapshot("aux").unwrap();
    let events = e.populate_relational_residency_snapshot("events").unwrap();
    e.set_relational_residency_budget_bytes(0, events.resident_bytes);
    let admitted = e.populate_relational_residency_snapshot("events").unwrap();
    assert_eq!(admitted.evicted_tables_on_admission, vec!["aux"]);

    let Command::Select(aux_count) = parse_command("SELECT COUNT(*) FROM aux").unwrap() else {
        unreachable!()
    };
    let evicted = e.plan_relational_resident_route(&aux_count);
    assert!(!evicted.accepted);
    assert_eq!(evicted.reason, "relation has no resident snapshot");
    assert_eq!(evicted.h2d_bytes_if_cold, 0);
    assert!(aux.resident_bytes > 0);

    let Command::Select(events_count) = parse_command("SELECT COUNT(*) FROM events").unwrap()
    else {
        unreachable!()
    };
    e.mark_gpu_memory_pressured(0);
    let pressured = e.plan_relational_resident_route(&events_count);
    assert!(!pressured.accepted);
    assert_eq!(pressured.cache_state, "InvalidatedByMemoryPressure");
    assert_eq!(
        pressured.reason,
        "resident snapshot is InvalidatedByMemoryPressure"
    );
    assert!(!pressured.valid);
    assert_eq!(
        e.telemetry_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap()
            .reason,
        pressured.reason
    );
}

#[test]
fn p8_partitioned_resident_count_reduces_valid_partitions_and_rejects_invalidated() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partitions = (0..4_u32)
        .map(|partition_id| {
            let row_count = 256_usize;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id,
                row_start: partition_id as usize * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: Vec::new(),
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM order_line").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_count_all");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 1024);
    assert_eq!(route.resident_bytes, 32);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.h2d_bytes_if_cold, 32);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int8(1024)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_count_all");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(
        decision.last_execution_kernel_samples,
        Some(
            after
                .kernel_exec_samples
                .saturating_sub(before.kernel_exec_samples)
        )
    );
    assert_eq!(decision.last_execution_matched_rows, Some(4));

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (1, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_count_all");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");
}

#[test]
fn p8_partitioned_resident_key_lookup_merges_matches_and_rejects_invalidated() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [Vec<i32>; 4] = [
        vec![42, 1, 42, 2],
        vec![3, 4, 5, 6],
        vec![42, 7, 8, 42],
        vec![9, 10, 11, 12],
    ];
    let partitions = partition_values
        .iter()
        .enumerate()
        .map(|(partition_id, values)| {
            let row_count = values.len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&(*value).to_le_bytes());
            }
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start: partition_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec!["ol_o_id".to_string()],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT ol_o_id FROM order_line WHERE ol_o_id = 42").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_equality_projection");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
            vec![SqlValue::Int4(42)],
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_equality_projection");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_matched_rows, Some(4));

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (42, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(
        invalidated.query_shape,
        "partitioned_int4_equality_projection"
    );
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");
}

#[test]
fn p8_partitioned_resident_multi_column_lookup_merges_projected_rows_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![42, 1, 42, 2],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![500, 501, 502, 503],
        ],
        [
            vec![3, 4, 5, 6],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![600, 601, 602, 603],
        ],
        [
            vec![42, 7, 8, 42],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![700, 701, 702, 703],
        ],
        [
            vec![9, 10, 11, 12],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![800, 801, 802, 803],
        ],
    ];
    let partitions = partition_values
        .iter()
        .enumerate()
        .map(|(partition_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start: partition_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec![
                    "ol_o_id".to_string(),
                    "ol_i_id".to_string(),
                    "ol_quantity".to_string(),
                    "ol_amount".to_string(),
                ],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command(
        "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = 42",
    )
    .unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(
        route.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert!(route.d2h_bytes_estimate > 0);

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(100),
                SqlValue::Int4(5),
                SqlValue::Int4(500)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(102),
                SqlValue::Int4(7),
                SqlValue::Int4(502)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(300),
                SqlValue::Int4(13),
                SqlValue::Int4(700)
            ],
            vec![
                SqlValue::Int4(42),
                SqlValue::Int4(303),
                SqlValue::Int4(16),
                SqlValue::Int4(703)
            ],
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(
        decision.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_matched_rows, Some(4));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision.last_execution_selected_projection_micros.is_some());

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (42, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(
        invalidated.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = partition_values
        .iter()
        .enumerate()
        .map(|(partition_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start: partition_id * row_count + 1,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec![
                    "ol_o_id".to_string(),
                    "ol_i_id".to_string(),
                    "ol_quantity".to_string(),
                    "ol_amount".to_string(),
                ],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(
        missing_layout.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_partitioned_resident_multi_column_lookup_orders_more_than_one_warp_of_matches_per_partition()
{
    // Coverage-gap closer (engine side) for the resident row-index host-sort fix. The
    // partitioned multi-column route iterates partitions in order and, within each partition,
    // materializes one output row per entry of `match_i32_equal_row_indices_from_payload(..)` in
    // that vector's order via a strictly positional gather. The CPU/non-resident reference emits
    // a partition's matching rows in ASCENDING row order. The resident kernel, however, appends
    // matches in `atom.global.add` SCHEDULE order, which is ascending only while all matches in
    // a partition fit in ONE warp (<= 32). Every existing partitioned parity test stays under
    // that boundary (<= 2 matches per partition), so this is the first test that puts MORE THAN
    // ONE WARP of matches in a SINGLE partition.
    //
    // Why this is non-vacuous (would fail/flake WITHOUT the host sort in
    // `launch_cuda_resident_i32_equal_row_indices`): with > 32 interleaved matches in a
    // partition, the kernel's cross-warp append order is non-deterministic and is essentially
    // never ascending, so the positional gather would emit that partition's rows in a
    // non-deterministic, non-ascending order — diverging from the ascending reference asserted
    // below and breaking the partitioned ascending-merge. It passes only because the route now
    // sorts the [0, count) indices host-side. The loop re-runs the query so a sort-less route
    // surfaces a wrong ordering on at least one iteration.
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    const NEEDLE: i32 = 42;
    // Two partitions. Partition 0 carries a MULTI-WARP block of matches: 200 rows where the
    // even rows match the needle (100 matches >> 32, interleaved across many warps and several
    // 128-thread blocks); the projected columns are distinct per row so the asserted order is
    // load-bearing. Partition 1 is a small non-matching tail (exercises the cross-partition
    // merge after the multi-warp partition).
    let p0_rows: usize = 200;
    let p0_ol_o_id: Vec<i32> = (0..p0_rows as i32)
        .map(|row| if row % 2 == 0 { NEEDLE } else { row + 1000 })
        .collect();
    // Make the other three columns unique, monotonic functions of the row so a mis-ordered
    // gather is caught by the exact row comparison (not just by the key column).
    let p0_ol_i_id: Vec<i32> = (0..p0_rows as i32).map(|row| 10_000 + row).collect();
    let p0_ol_quantity: Vec<i32> = (0..p0_rows as i32).map(|row| 20_000 + row).collect();
    let p0_ol_amount: Vec<i32> = (0..p0_rows as i32).map(|row| 30_000 + row).collect();

    let p1_ol_o_id = vec![1, 2, 3, 4];
    let p1_ol_i_id = vec![401, 402, 403, 404];
    let p1_ol_quantity = vec![17, 18, 19, 20];
    let p1_ol_amount = vec![801, 802, 803, 804];

    let partition_columns: Vec<[Vec<i32>; 4]> = vec![
        [p0_ol_o_id, p0_ol_i_id, p0_ol_quantity, p0_ol_amount],
        [p1_ol_o_id, p1_ol_i_id, p1_ol_quantity, p1_ol_amount],
    ];

    // CPU reference: rows from each partition in ASCENDING row order, partitions in order.
    let mut expected_rows: Vec<Vec<SqlValue>> = Vec::new();
    for columns in &partition_columns {
        for (row, key) in columns[0].iter().enumerate() {
            if *key == NEEDLE {
                expected_rows.push(vec![
                    SqlValue::Int4(*key),
                    SqlValue::Int4(columns[1][row]),
                    SqlValue::Int4(columns[2][row]),
                    SqlValue::Int4(columns[3][row]),
                ]);
            }
        }
    }
    assert!(
        expected_rows.len() > 32,
        "test must match more than one warp of rows in a single partition"
    );

    let mut row_cursor = 1usize;
    let partitions = partition_columns
        .iter()
        .enumerate()
        .map(|(partition_id, columns)| {
            let row_count = columns[0].len();
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
            for column in columns {
                for value in column {
                    bytes.extend_from_slice(&(*value).to_le_bytes());
                }
            }
            let row_start = row_cursor;
            row_cursor += row_count;
            BenchmarkRelationalResidencyOwnedPartition {
                partition_id: partition_id as u32,
                row_start,
                row_count,
                resident_bytes: bytes.len() as u64,
                allocated_bytes: bytes.len() as u64,
                resident_device_int4_columns: vec![
                    "ol_o_id".to_string(),
                    "ol_i_id".to_string(),
                    "ol_quantity".to_string(),
                    "ol_amount".to_string(),
                ],
                resident_device_text_columns: Vec::new(),
                chunks: vec![CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes,
                }],
            }
        })
        .collect::<Vec<_>>();

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) = parse_command(
        "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = 42",
    )
    .unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(
        route.query_shape,
        "partitioned_int4_equality_multi_column_projection"
    );
    assert_eq!(route.partition_count, 2);

    // Re-run so a non-deterministic (sort-less) cross-warp order is caught on some iteration.
    for iter in 0..25 {
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        assert_eq!(
            result.rows, expected_rows,
            "partitioned multi-warp resident rows were not in ascending reference order on \
                 iteration {iter} — the resident route's host sort over [0, count) is missing?"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_batched_multi_column_projection_matches_per_query_for_more_than_one_warp_of_matches() {
    // Thread-3 Stage-4 ordered-parity gate (multi-column all-int4). The batched submit/complete
    // path scatters rows per needle in the `equal_any` kernel's `atom.global.add` SCHEDULE
    // order; the per-query path (`execute_relational_select` -> the multi-column probe) now sorts
    // its fused `equal_project` output ascending-by-row_index. Both must return the SAME rows in
    // the SAME (ascending) order for a MULTI-WARP match count (>32, where the atomic-append
    // order is non-deterministic), so the batched output is byte-identical to the per-query path.
    //
    // Non-vacuous: the projected `seq` column is a by-row SCRAMBLED hash, so the ascending-by-row
    // reference is NOT value-sorted and NOT the atomic-append order. WITHOUT the stable-order
    // sort the batched scatter would emit a non-deterministic permutation (caught by the exact
    // comparison and the 25× loop), and it would differ from the per-query path.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (k INT, seq INT)")
        .unwrap();
    const NEEDLE: i32 = 7;
    let rows: usize = 200; // even rows match => 100 matches >> 32 (multi-warp/multi-block).
    let row_value = |row: usize| -> i32 {
        let r = row as u64;
        let h = r.wrapping_mul(2_654_435_761) ^ (r << 13) ^ 0x9E37_79B9;
        (1 + (h % 1_000_000)) as i32
    };
    let mut values = String::new();
    for row in 0..rows {
        if row > 0 {
            values.push_str(", ");
        }
        let k = if row % 2 == 0 {
            NEEDLE
        } else {
            row as i32 + 1000
        };
        values.push_str(&format!("({k}, {})", row_value(row)));
    }
    e.execute_text(2, &format!("INSERT INTO t (k, seq) VALUES {values}"))
        .unwrap();
    e.populate_relational_residency_snapshot("t").unwrap();

    let Command::Select(select) = parse_command("SELECT k, seq FROM t WHERE k = 7").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(route.query_shape, "int4_equality_multi_column_projection");

    // Ascending-by-row reference (independent of either GPU path).
    let expected: Vec<Vec<SqlValue>> = (0..rows)
        .filter(|row| row % 2 == 0)
        .map(|row| vec![SqlValue::Int4(NEEDLE), SqlValue::Int4(row_value(row))])
        .collect();
    assert!(
        expected.len() > 32,
        "test must match more than one warp of rows"
    );
    // The projected by-row sequence must NOT already be value-sorted, else an atomic-append or
    // value-sort impl would pass vacuously.
    let proj_by_row: Vec<i32> = expected
        .iter()
        .map(|row| match row[1] {
            SqlValue::Int4(v) => v,
            _ => unreachable!(),
        })
        .collect();
    let mut proj_sorted = proj_by_row.clone();
    proj_sorted.sort_unstable();
    assert_ne!(
        proj_by_row, proj_sorted,
        "projected by-row sequence is accidentally value-sorted — pick a payload that isn't"
    );

    for iter in 0..25 {
        let per_query = e.execute_relational_select(&select).unwrap();
        assert_eq!(
            per_query.rows, expected,
            "per-query multi-column rows not ascending-by-row on iteration {iter}"
        );

        let job = e.prepare_relational_retained_read_job(&select).unwrap();
        let submission = e
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
                std::slice::from_ref(&job),
            )
            .unwrap();
        let batched = e
            .complete_relational_retained_read_submission(submission)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(
            batched[0].rows, expected,
            "batched multi-column rows not ascending-by-row on iteration {iter} \
                 — the stable-order sort in the result assembly is missing or ineffective?"
        );
        assert_eq!(
            batched[0].rows, per_query.rows,
            "batched multi-column output diverged from the per-query path on iteration {iter}"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn p8_batched_mixed_column_projection_matches_per_query_for_more_than_one_warp_of_matches() {
    // Thread-3 Stage-4 ordered-parity gate (mixed int4/text). The batched path for the mixed
    // shape falls through to `..._batch_inner` (the int4 `equal_any` fast path is int4-only), and
    // the per-query mixed path delegates to the SAME `batch_inner`; both now sort each needle's
    // slice ascending-by-row_index. They must return the SAME rows in the SAME order for a
    // MULTI-WARP match count.
    //
    // Non-vacuous: the projected `label` text is a by-row SCRAMBLED value, so the ascending-by-row
    // reference is neither value-sorted nor the atomic-append order. WITHOUT the sort the scatter
    // is a non-deterministic permutation (caught by the exact comparison + the 25× loop).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (k INT, label TEXT)")
        .unwrap();
    const NEEDLE: i32 = 7;
    let rows: usize = 200; // even rows match => 100 matches >> 32.
                           // Scrambled-by-row label so the ascending-by-row text order is not lexicographically sorted.
    let label_value = |row: usize| -> String {
        let r = row as u64;
        let h = r.wrapping_mul(2_654_435_761) ^ (r << 13) ^ 0x9E37_79B9;
        format!("L{:08}", h % 100_000_000)
    };
    let mut values = String::new();
    for row in 0..rows {
        if row > 0 {
            values.push_str(", ");
        }
        let k = if row % 2 == 0 {
            NEEDLE
        } else {
            row as i32 + 1000
        };
        values.push_str(&format!("({k}, '{}')", label_value(row)));
    }
    e.execute_text(2, &format!("INSERT INTO t (k, label) VALUES {values}"))
        .unwrap();
    e.populate_relational_residency_snapshot("t").unwrap();

    let Command::Select(select) = parse_command("SELECT k, label FROM t WHERE k = 7").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if !route.accepted {
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
        return;
    }
    assert_eq!(route.query_shape, "int4_equality_mixed_column_projection");

    let expected: Vec<Vec<SqlValue>> = (0..rows)
        .filter(|row| row % 2 == 0)
        .map(|row| vec![SqlValue::Int4(NEEDLE), SqlValue::Text(label_value(row))])
        .collect();
    assert!(
        expected.len() > 32,
        "test must match more than one warp of rows"
    );
    let label_by_row: Vec<String> = expected
        .iter()
        .map(|row| match &row[1] {
            SqlValue::Text(v) => v.clone(),
            _ => unreachable!(),
        })
        .collect();
    let mut label_sorted = label_by_row.clone();
    label_sorted.sort();
    assert_ne!(
        label_by_row, label_sorted,
        "projected by-row label sequence is accidentally sorted — pick a payload that isn't"
    );

    for iter in 0..25 {
        let per_query = e.execute_relational_select(&select).unwrap();
        assert_eq!(
            per_query.rows, expected,
            "per-query mixed rows not ascending-by-row on iteration {iter}"
        );

        let job = e.prepare_relational_retained_read_job(&select).unwrap();
        let submission = e
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(
                std::slice::from_ref(&job),
            )
            .unwrap();
        let batched = e
            .complete_relational_retained_read_submission(submission)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(
            batched[0].rows, expected,
            "batched mixed rows not ascending-by-row on iteration {iter} \
                 — the stable-order sort in the result assembly is missing or ineffective?"
        );
        assert_eq!(
            batched[0].rows, per_query.rows,
            "batched mixed output diverged from the per-query path on iteration {iter}"
        );
    }
}

#[test]
fn p8_partitioned_resident_sum_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![42, 1, 42, 2],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![500, 501, 502, 503],
        ],
        [
            vec![3, 4, 5, 6],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![600, 601, 602, 603],
        ],
        [
            vec![42, 7, 8, 42],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![700, 701, 702, 703],
        ],
        [
            vec![9, 10, 11, 12],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![800, 801, 802, 803],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
                    row_count,
                    resident_bytes: bytes.len() as u64,
                    allocated_bytes: bytes.len() as u64,
                    resident_device_int4_columns: vec![
                        "ol_o_id".to_string(),
                        "ol_i_id".to_string(),
                        "ol_quantity".to_string(),
                        "ol_amount".to_string(),
                    ],
                    resident_device_text_columns: Vec::new(),
                    chunks: vec![CudaOwnedDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes,
                    }],
                }
            })
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = 42").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int8(2405)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(4));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision.last_execution_selected_projection_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (42, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_equality_sum");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_between_avg_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 20, 30, 40],
        ],
        [
            vec![15, 25, 35, 45],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![15, 25, 35, 45],
        ],
        [
            vec![50, 60, 70, 80],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![50, 60, 70, 80],
        ],
        [
            vec![20, 21, 22, 23],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![20, 21, 22, 23],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
                    row_count,
                    resident_bytes: bytes.len() as u64,
                    allocated_bytes: bytes.len() as u64,
                    resident_device_int4_columns: vec![
                        "ol_o_id".to_string(),
                        "ol_i_id".to_string(),
                        "ol_quantity".to_string(),
                        "ol_amount".to_string(),
                    ],
                    resident_device_text_columns: Vec::new(),
                    chunks: vec![CudaOwnedDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes,
                    }],
                }
            })
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_o_id BETWEEN 20 AND 35")
            .unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_between_avg");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Numeric(
            Decimal128::parse("24.5000000000000000").unwrap()
        )]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_between_avg");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(8));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision.last_execution_selected_projection_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_o_id BETWEEN 90 AND 99")
            .unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // AVG over zero matched rows now yields the canonical-zero numeric sentinel.
    assert_eq!(
        no_match.rows,
        vec![vec![SqlValue::Numeric(Decimal128::ZERO)]]
    );

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 1, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_between_avg");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_between_avg");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_filtered_max_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 25, 70, 15],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![1, 2, 3, 4],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
                    row_count,
                    resident_bytes: bytes.len() as u64,
                    allocated_bytes: bytes.len() as u64,
                    resident_device_int4_columns: vec![
                        "ol_o_id".to_string(),
                        "ol_i_id".to_string(),
                        "ol_quantity".to_string(),
                        "ol_amount".to_string(),
                    ],
                    resident_device_text_columns: Vec::new(),
                    chunks: vec![CudaOwnedDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes,
                    }],
                }
            })
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= 50").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(80)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(5));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= 100").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    assert_eq!(no_match.rows, vec![vec![SqlValue::Text(String::new())]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 99, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_filtered_max");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_filtered_min_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![90, 25, 70, 85],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![91, 92, 93, 94],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
                    row_count,
                    resident_bytes: bytes.len() as u64,
                    allocated_bytes: bytes.len() as u64,
                    resident_device_int4_columns: vec![
                        "ol_o_id".to_string(),
                        "ol_i_id".to_string(),
                        "ol_quantity".to_string(),
                        "ol_amount".to_string(),
                    ],
                    resident_device_text_columns: Vec::new(),
                    chunks: vec![CudaOwnedDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes,
                    }],
                }
            })
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT MIN(ol_amount) FROM order_line WHERE ol_amount <= 60").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(10)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(6));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT MIN(ol_amount) FROM order_line WHERE ol_amount <= 0").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    assert_eq!(no_match.rows, vec![vec![SqlValue::Text(String::new())]]);

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 5, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_filtered_min");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_partitioned_resident_filtered_avg_reduces_matches_and_rejects_missing_layout() {
    let mut e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
        )
        .unwrap();

    let partition_values: [[Vec<i32>; 4]; 4] = [
        [
            vec![10, 20, 30, 40],
            vec![100, 101, 102, 103],
            vec![5, 6, 7, 8],
            vec![10, 25, 70, 15],
        ],
        [
            vec![50, 60, 70, 80],
            vec![200, 201, 202, 203],
            vec![9, 10, 11, 12],
            vec![20, 30, 40, 10],
        ],
        [
            vec![90, 91, 92, 93],
            vec![300, 301, 302, 303],
            vec![13, 14, 15, 16],
            vec![75, 55, 65, 80],
        ],
        [
            vec![94, 95, 96, 97],
            vec![400, 401, 402, 403],
            vec![17, 18, 19, 20],
            vec![91, 92, 93, 94],
        ],
    ];
    let build_partitions = || {
        partition_values
            .iter()
            .enumerate()
            .map(|(partition_id, columns)| {
                let row_count = columns[0].len();
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
                for column in columns {
                    for value in column {
                        bytes.extend_from_slice(&(*value).to_le_bytes());
                    }
                }
                BenchmarkRelationalResidencyOwnedPartition {
                    partition_id: partition_id as u32,
                    row_start: partition_id * row_count + 1,
                    row_count,
                    resident_bytes: bytes.len() as u64,
                    allocated_bytes: bytes.len() as u64,
                    resident_device_int4_columns: vec![
                        "ol_o_id".to_string(),
                        "ol_i_id".to_string(),
                        "ol_quantity".to_string(),
                        "ol_amount".to_string(),
                    ],
                    resident_device_text_columns: Vec::new(),
                    chunks: vec![CudaOwnedDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes,
                    }],
                }
            })
            .collect::<Vec<_>>()
    };

    let installed = e.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: build_partitions(),
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }

    let Command::Select(select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_amount <= 60").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    assert!(route.accepted, "{route:?}");
    assert_eq!(route.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(route.partition_count, 4);
    assert_eq!(route.estimated_rows, 16);
    assert_eq!(route.h2d_bytes_if_resident, 0);
    assert_eq!(route.d2h_rows_estimate, 1);
    assert_eq!(
        route.d2h_bytes_estimate,
        (4 * std::mem::size_of::<u64>()) as u64
    );

    let before = e.metrics().snapshot();
    let result = e.execute_relational_select(&select).unwrap();
    let after = e.metrics().snapshot();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Numeric(
            Decimal128::parse("25.6250000000000000").unwrap()
        )]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    let decision = e
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .unwrap()
        .clone();
    assert_eq!(decision.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(decision.partition_count, 4);
    assert_eq!(decision.last_execution_h2d_bytes, Some(0));
    assert_eq!(
        decision.last_execution_d2h_bytes,
        Some(after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total))
    );
    assert_eq!(decision.last_execution_rows, Some(1));
    assert_eq!(decision.last_execution_matched_rows, Some(8));
    assert!(decision.last_execution_match_index_micros.is_some());
    assert!(decision
        .last_execution_result_materialization_micros
        .is_some());

    let Command::Select(no_match_select) =
        parse_command("SELECT AVG(ol_amount) FROM order_line WHERE ol_amount <= 0").unwrap()
    else {
        unreachable!()
    };
    let no_match = e.execute_relational_select(&no_match_select).unwrap();
    // AVG over zero matched rows now yields the canonical-zero numeric sentinel.
    assert_eq!(
        no_match.rows,
        vec![vec![SqlValue::Numeric(Decimal128::ZERO)]]
    );

    e.execute_text(
            2,
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (25, 1, 1, 5, 'x')",
        )
        .unwrap();
    let invalidated = e.plan_relational_resident_route(&select);
    assert!(!invalidated.accepted);
    assert_eq!(invalidated.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(invalidated.partition_count, 4);
    assert_eq!(invalidated.cache_state, "Invalidated");
    assert_eq!(invalidated.reason, "resident partition set is Invalidated");

    let mut missing_layout_engine = Engine::new_local();
    missing_layout_engine
            .execute_text(
                1,
                "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)",
            )
            .unwrap();
    let mut missing_layout_partitions = build_partitions();
    missing_layout_partitions[2]
        .resident_device_int4_columns
        .pop();
    let installed = missing_layout_engine.install_benchmark_relational_residency_owned_partitions(
        BenchmarkRelationalResidencyOwnedPartitionInstall {
            table: "order_line",
            gpu_id: 0,
            partitions: missing_layout_partitions,
        },
    );
    if let Err(err) = installed {
        assert!(
            err.to_string().contains("CUDA"),
            "unexpected partition install error: {err}"
        );
        return;
    }
    let missing_layout = missing_layout_engine.plan_relational_resident_route(&select);
    assert!(!missing_layout.accepted);
    assert_eq!(missing_layout.query_shape, "partitioned_int4_filtered_avg");
    assert_eq!(
        missing_layout.reason,
        "resident partition 2 lacks required int4 projection layout"
    );
}

#[test]
fn p8_resident_warmup_policy_warms_refreshes_and_reports_route_readiness() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();

    let report = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(report.gpu_id, 0);
    assert_eq!(report.entries.len(), 1);
    let entry = &report.entries[0];
    assert_eq!(entry.table, "events");
    assert_eq!(entry.action, RelationalResidencyWarmupAction::Warmed);
    assert!(entry.resident_bytes > 0);
    let first_handle = e.relational_retained_snapshot_handle("events").unwrap();
    assert_eq!(first_handle.generation, 1);
    assert_eq!(first_handle.row_count, 2);
    assert!(first_handle.valid);
    assert_eq!(
        first_handle.resident_device_int4_columns,
        vec!["id".to_string()]
    );
    let route = entry.route_decision.as_ref().unwrap();
    assert_eq!(route.query_shape, "count_all");
    assert_eq!(route.snapshot_generation, Some(first_handle.generation));
    if route.has_retained_device_memory {
        assert!(route.accepted);
        let Command::Select(select) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
            unreachable!()
        };
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
    } else {
        assert!(!route.accepted);
        assert_eq!(
            route.reason,
            "resident snapshot has no retained device memory"
        );
    }

    e.execute_text(3, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        Some(3)
    );
    let invalidated_handle = e.relational_retained_snapshot_handle("events").unwrap();
    assert!(first_handle.valid);
    assert_eq!(
        first_handle.has_retained_device_memory,
        route.has_retained_device_memory
    );
    assert_eq!(invalidated_handle.generation, first_handle.generation);
    assert!(!invalidated_handle.valid);
    assert!(!invalidated_handle.has_retained_device_memory);
    let refreshed = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(
        refreshed.entries[0].action,
        RelationalResidencyWarmupAction::Refreshed
    );
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        None
    );
    let refreshed_handle = e.relational_retained_snapshot_handle("events").unwrap();
    assert_eq!(refreshed_handle.generation, 2);
    assert_eq!(refreshed_handle.row_count, 3);
    assert!(refreshed_handle.valid);
    assert_eq!(
        e.status_snapshot()
            .relational_residency
            .latest_route_decision("events")
            .unwrap()
            .estimated_rows,
        3
    );
    assert_eq!(
        e.status_snapshot()
            .relational_residency
            .table("events")
            .unwrap()
            .snapshot_generation,
        refreshed_handle.generation
    );

    let gpu_override = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        gpu_id: Some(7),
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(gpu_override.gpu_id, 7);
    assert_eq!(e.relational_residency_snapshot("events").unwrap().gpu_id, 7);
}

#[test]
fn p8_resident_warmup_policy_applies_budget_and_skips_unsafe_inputs() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE aux (id INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "CREATE TABLE oversized (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    e.execute_text(5, "INSERT INTO aux (id, label) VALUES (1, 'aux')")
        .unwrap();
    e.execute_text(
        6,
        "INSERT INTO oversized (id, label) VALUES (1, 'this-row-is-too-large-for-the-test-budget')",
    )
    .unwrap();

    let events_size = e.populate_relational_residency_snapshot("events").unwrap();
    let aux_size = e.populate_relational_residency_snapshot("aux").unwrap();
    e.clear_relational_residency_budget_bytes(0);
    let report = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec![
            "events".to_string(),
            "aux".to_string(),
            "missing".to_string(),
        ],
        budget_bytes: Some(events_size.resident_bytes),
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(report.budget_bytes, Some(events_size.resident_bytes));
    assert_eq!(report.entries.len(), 3);
    assert!(report.entries.iter().any(|entry| entry.table == "missing"
        && entry.action == RelationalResidencyWarmupAction::Skipped));
    let events = report
        .entries
        .iter()
        .find(|entry| entry.table == "events")
        .unwrap();
    assert!(matches!(
        events.action,
        RelationalResidencyWarmupAction::AlreadyResident | RelationalResidencyWarmupAction::Warmed
    ));
    let aux = report
        .entries
        .iter()
        .find(|entry| entry.table == "aux")
        .unwrap();
    assert!(matches!(
        aux.action,
        RelationalResidencyWarmupAction::Warmed
            | RelationalResidencyWarmupAction::Refreshed
            | RelationalResidencyWarmupAction::AlreadyResident
    ));
    assert!(aux_size.resident_bytes > 0);
    assert!(e.status_snapshot().relational_residency.snapshot_count() <= 1);

    let oversized = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["oversized".to_string()],
        budget_bytes: Some(1),
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(
        oversized.entries[0].action,
        RelationalResidencyWarmupAction::Error
    );
    assert!(oversized.entries[0].reason.contains("exceeding GPU 0"));
    assert!(e.relational_residency_snapshot("oversized").is_none());

    e.mark_gpu_memory_pressured(0);
    let pressured = e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    assert_eq!(
        pressured.entries[0].action,
        RelationalResidencyWarmupAction::Skipped
    );
    assert_eq!(pressured.entries[0].reason, "GPU 0 is memory pressured");
}

#[test]
fn p8_resident_maintenance_tick_summarizes_refresh_and_route_readiness() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE aux (id INT, label TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO events (id, label) VALUES (1, 'alpha'), (2, 'beta')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO aux (id, label) VALUES (1, 'aux')")
        .unwrap();

    e.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
        tables: vec!["events".to_string()],
        refresh_invalidated: true,
        ..RelationalResidencyWarmupPolicy::default()
    });
    e.execute_text(5, "INSERT INTO events (id, label) VALUES (3, 'gamma')")
        .unwrap();
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        Some(5)
    );

    let report = e
        .maintain_relational_residency_with_policy(RelationalResidencyMaintenancePolicy::default());
    assert_eq!(report.gpu_id, 0);
    assert_eq!(report.entry_count, 2);
    assert_eq!(report.refreshed_count, 1);
    assert_eq!(report.warmed_count, 1);
    assert_eq!(report.skipped_count, 0);
    assert_eq!(report.error_count, 0);
    assert_eq!(
        report.route_ready_count + report.route_blocked_count,
        report.entry_count
    );
    assert_eq!(
        e.relational_residency_snapshot("events")
            .unwrap()
            .invalidated_by_txn_id,
        None
    );
    assert!(e.relational_residency_snapshot("aux").is_some());

    let Command::Select(select) = parse_command("SELECT COUNT(*) FROM events").unwrap() else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&select);
    if route.accepted {
        let result = e.execute_relational_select(&select).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        assert_eq!(result.rows, vec![vec![SqlValue::Int8(3)]]);
        assert!(report
            .route_ready_tables
            .iter()
            .any(|table| table == "events"));
    } else {
        assert!(report
            .route_blockers
            .iter()
            .any(|blocker| blocker.table == "events" && blocker.reason == route.reason));
    }
}

#[test]
fn p8_resident_maintenance_tick_reports_pressure_and_budget_blockers() {
    let mut pressured = Engine::new_local();
    pressured
        .execute_text(1, "CREATE TABLE events (id INT, label TEXT)")
        .unwrap();
    pressured
        .execute_text(2, "INSERT INTO events (id, label) VALUES (1, 'alpha')")
        .unwrap();
    pressured.mark_gpu_memory_pressured(0);
    let pressure_report = pressured
        .maintain_relational_residency_with_policy(RelationalResidencyMaintenancePolicy::default());
    assert_eq!(pressure_report.entry_count, 1);
    assert_eq!(pressure_report.skipped_count, 1);
    assert_eq!(pressure_report.route_ready_count, 0);
    assert_eq!(pressure_report.route_blocked_count, 1);
    assert_eq!(pressure_report.route_blockers[0].table, "events");
    assert_eq!(
        pressure_report.route_blockers[0].reason,
        "GPU 0 is memory pressured"
    );
    assert!(pressured.relational_residency_snapshot("events").is_none());

    let mut oversized = Engine::new_local();
    oversized
        .execute_text(1, "CREATE TABLE oversized (id INT, label TEXT)")
        .unwrap();
    oversized
        .execute_text(
            2,
            "INSERT INTO oversized (id, label) VALUES (1, 'this-row-is-too-large')",
        )
        .unwrap();
    let budget_report =
        oversized.maintain_relational_residency_with_policy(RelationalResidencyMaintenancePolicy {
            budget_bytes: Some(1),
            ..RelationalResidencyMaintenancePolicy::default()
        });
    assert_eq!(budget_report.entry_count, 1);
    assert_eq!(budget_report.error_count, 1);
    assert_eq!(budget_report.route_ready_count, 0);
    assert_eq!(budget_report.route_blocked_count, 1);
    assert!(budget_report.route_blockers[0]
        .reason
        .contains("exceeding GPU 0"));
    assert!(oversized
        .relational_residency_snapshot("oversized")
        .is_none());
}

#[test]
fn telemetry_snapshot_reflects_replication_lag_and_runtime_metrics() {
    let mut e = Engine::with_batching(8, Duration::from_secs(60));
    let t0 = Instant::now();

    e.enqueue_set_text(1, "SET a=1", t0).unwrap();

    let snapshot = e.telemetry_snapshot();

    assert_eq!(snapshot.role, Role::Leader);
    assert_eq!(snapshot.replication_lag.commit_index, 0);
    assert_eq!(snapshot.replication_lag.applied_index, 0);
    assert_eq!(snapshot.replication_lag.visible_index, 0);
    assert_eq!(snapshot.replication_lag.commit_apply_gap, 0);
    assert_eq!(snapshot.replication_lag.apply_visible_gap, 0);
    assert_eq!(snapshot.runtime_metrics.pending_batch_peak, 1);
    assert_eq!(snapshot.runtime_metrics.last_pending_batch_len, Some(1));
    assert_eq!(snapshot.runtime_metrics.commits_total, 0);
    assert_eq!(snapshot.snapshot_id, 0);
    assert_eq!(snapshot.wal_flushed_count, 0);
    assert_eq!(snapshot.wal_last_durable_txn_id, None);
    assert_eq!(snapshot.wal_buffered_count, 0);
    assert_eq!(snapshot.wal_unflushed_count, 0);
    assert_eq!(snapshot.pending_batch_len, 1);
    assert_eq!(snapshot.pending_batch_cap, 8);
    assert_eq!(snapshot.active_txn_count, 0);
    assert_eq!(snapshot.backlog_blocker_count, 1);
    assert!(snapshot.has_backlog_blockers());
    assert!(!snapshot.quiescent_for_failover);
    assert!(!snapshot.mutation_admission_saturated);
    assert!(snapshot.gpu_parity_fallbacks.is_empty());
}

#[test]
fn status_snapshot_answers_snapshot_and_replication_health_questions() {
    let mut e = Engine::new_local();
    let token = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
    let exported = e.export_snapshot_meta();

    let status = e.status_snapshot();

    assert_eq!(status.role, Role::Leader);
    assert_eq!(status.term, 1);
    assert_eq!(status.snapshot.snapshot_id, exported.snapshot_id);
    assert_eq!(status.snapshot.last_included_index, token.index);
    assert_eq!(status.snapshot.last_included_term, 1);
    assert_eq!(status.snapshot.visible_index, token.index);
    assert_eq!(status.served_snapshot_frontier(), token.index);
    assert_eq!(status.replication_lag.commit_index, token.index);
    assert_eq!(status.replication_lag.applied_index, token.index);
    assert_eq!(status.replication_distance(), 0);
    assert!(status.why_routed_to_fallback_labels().is_empty());
    assert_eq!(status.latest_fallback_reason(), None);
    assert_eq!(status.backlog_blocker_labels(), Vec::<&'static str>::new());
    status.validate().unwrap();
}

#[test]
fn status_snapshot_surfaces_active_fallback_reasons_and_rollups() {
    let mut e = Engine::new_local();
    e.mark_gpu_unavailable(0);
    e.set_gpu_runtime_saturated(true);

    e.execute_text(1, "SET a=1").unwrap();

    let status = e.status_snapshot();

    assert_eq!(
        status.latest_fallback_reason(),
        Some(FallbackReason::GpuUnavailable)
    );
    assert_eq!(
        status.why_routed_to_fallback_labels(),
        vec!["gpu_unavailable", "gpu_queue_saturated"]
    );
    assert!(status.fallback.is_actively_degraded());
    assert!(status.fallback.has_gpu_parity_fallbacks());
    assert_eq!(status.fallback.gpu_parity_fallback_total(), 1);
    assert_eq!(
        status.fallback.active_reasons,
        vec![
            ActiveFallbackReason::GpuUnavailable { gpu_ids: vec![0] },
            ActiveFallbackReason::GpuQueueSaturated,
        ]
    );
    status.validate().unwrap();
}

#[test]
fn execute_mvcc_query_runs_visibility_filtered_scan_through_execution_layer() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=pending").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(
        result.fallback_reason,
        Some(FallbackReason::GpuMvccReadParityGap)
    );
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(
        e.metrics()
            .fallback_for(FallbackReason::GpuMvccReadParityGap),
        1
    );
}

fn assert_mvcc_query_uses_tracked_cpu_fallback(
    engine: &Engine,
    result: &MvccReadResult,
    expected_total_fallbacks: u64,
) {
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(
        result.fallback_reason,
        Some(FallbackReason::GpuMvccReadParityGap)
    );
    assert_eq!(
        engine
            .metrics()
            .fallback_for(FallbackReason::GpuMvccReadParityGap),
        expected_total_fallbacks
    );
}

#[derive(Debug, Clone)]
struct RecordingMvccBackend {
    executed_target: DeviceTarget,
    rows: Vec<MvccReadRow>,
}

impl MvccExecutionBackend for RecordingMvccBackend {
    fn execute(&self, _query: &MvccReadQuery, _rows: Vec<ResolvedMvccRow>) -> MvccBackendDispatch {
        MvccBackendDispatch::Executed(MvccBackendExecution {
            executed_target: self.executed_target,
            rows: self.rows.clone(),
        })
    }
}


fn first_cuda_slice_support_query() -> MvccReadQuery {
    MvccReadQuery {
        source: MvccReadSource::KeyLookup {
            key: "acct:1".to_string(),
        },
        visibility: StorageVisibility { read_txn_id: 1 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::KeyPrefix("acct:".to_string()),
            MvccReadFilter::ValueEquals("open".to_string()),
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    }
}

#[test]
fn first_cuda_slice_query_gap_accepts_supported_shape() {
    let query = first_cuda_slice_support_query();

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_first_cuda_slice_query(&query));
}

#[test]
fn cuda_native_full_scan_query_requires_full_scan_first_slice_shape() {
    let mut query = first_cuda_slice_support_query();
    assert!(is_cuda_native_source_query(&query));
    assert!(!is_cuda_native_full_scan_query(&query));

    query.source = MvccReadSource::FullScan;
    assert!(is_cuda_native_source_query(&query));
    assert!(is_cuda_native_full_scan_query(&query));

    query.order = Some(MvccReadOrder::ValueAsc);
    assert!(is_cuda_native_source_query(&query));
    assert!(is_cuda_native_full_scan_query(&query));
}

#[test]
fn first_cuda_slice_query_gap_accepts_key_order_for_native_single_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FullScan;
    query.order = Some(MvccReadOrder::KeyDesc);

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::KeyBatchLookup {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query.source = MvccReadSource::FullScan;
    query.order = Some(MvccReadOrder::ValueAsc);
    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_concat_of_native_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:2".to_string(), "acct:3".to_string()],
            },
        ],
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::Concat {
        sources: vec![MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        }],
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));
}

#[test]
fn first_cuda_slice_query_gap_accepts_native_follow_value_chain_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 1,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = None;
    query.order = Some(MvccReadOrder::SourceKeyAsc);
    query.projection = MvccProjection::TargetKeySourceValue;

    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 1,
            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::FollowValueChainBranches {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
        plans: vec![
            MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
        ],
        fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
        provenance: MvccSourceProvenance::Seed,
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));

    query.source = MvccReadSource::FollowValueChainLabeledBranches {
        keys: vec!["acct:1".to_string()],
        branches: vec![MvccLabeledValueChainBranch {
            label: "members".to_string(),
            plan: MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
        }],
        fan_in: MvccValueChainBranchFanIn::AllBranches,
        provenance: MvccSourceProvenance::TerminalInput,
    };
    query.filter = Some(MvccReadFilter::BranchLabelEquals("members".to_string()));
    query.order = Some(MvccReadOrder::BranchLabelAsc);
    query.projection = MvccProjection::BranchLabelTargetValue;
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert!(is_cuda_native_source_query(&query));
}

#[test]
fn first_cuda_slice_query_gap_reports_first_unsupported_boundary() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::Concat {
        sources: vec![MvccReadSource::ConcatDistinct {
            sources: vec![MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            }],
        }],
    };
    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query = first_cuda_slice_support_query();
    query.order = Some(MvccReadOrder::BranchLabelAsc);
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::UnsupportedOrder)
    );

    query = first_cuda_slice_support_query();
    query.projection = MvccProjection::SourceValueOnly;
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::UnsupportedProjection)
    );

    query = first_cuda_slice_support_query();
    query.limit = Some(1);
    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_reports_filter_shape_gaps() {
    let mut query = first_cuda_slice_support_query();
    query.filter = Some(MvccReadFilter::All(vec![]));
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::EmptyLogicalFilterTree)
    );

    query = first_cuda_slice_support_query();
    query.filter = Some(MvccReadFilter::Any(vec![MvccReadFilter::All(vec![
        MvccReadFilter::KeyPrefix("acct:".to_string()),
        MvccReadFilter::SourceKeyPrefix("seed:".to_string()),
    ])]));
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert_eq!(
        first_cuda_slice_filter_gap_detail(query.filter.as_ref().unwrap()),
        None
    );

    query = first_cuda_slice_support_query();
    query.filter = Some(MvccReadFilter::BranchLabelEquals("fallback".to_string()));
    assert_eq!(first_cuda_slice_query_gap(&query), None);
    assert_eq!(
        first_cuda_slice_filter_gap_detail(query.filter.as_ref().unwrap()),
        None
    );
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_filter_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 1,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::ProvenanceKeyPrefix {
            frame: MvccProvenanceFrame::Seed,
            prefix: "acct:".to_string(),
        },
        MvccReadFilter::ProvenanceValueEquals {
            frame: MvccProvenanceFrame::TerminalInput,
            expected: "team:alpha".to_string(),
        },
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query.filter = Some(MvccReadFilter::KeyPrefix("team:".to_string()));
    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_bundle_path_filters() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 2,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::ProvenanceBundlePathContains {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            expected: vec!["profile:2".to_string(), "team:beta".to_string()],
        },
        MvccReadFilter::ProvenanceBundlePathSegmentEquals {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            index: 1,
            expected: "profile:2".to_string(),
        },
        MvccReadFilter::ProvenanceBundleLenEquals {
            bundle: MvccProvenanceFrameBundle::FullPath,
            expected_len: 3,
        },
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_bundle_occurrence_path_filters() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 3,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            left_occurrence_index: 0,
            right_occurrence_index: 1,
            distance: 3,
            expected: vec!["acct:1".to_string()],
        },
        MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            left_occurrence_index: 0,
            left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
            right_occurrence_index: 0,
            right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
            distance: 2,
        },
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_provenance_projection_and_order_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChain {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
        plan: MvccValueChainPlan {
            value_key_hops: 2,
            terminal: MvccValueChainTerminal::CurrentRow,
        },
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::ProvenanceBundleLenEquals {
        bundle: MvccProvenanceFrameBundle::FullPath,
        expected_len: 3,
    });
    query.order = Some(MvccReadOrder::ProvenanceValueDesc {
        frame: MvccProvenanceFrame::TerminalInput,
    });
    query.projection = MvccProjection::TargetKeyProvenanceBundleSummary {
        bundle: MvccProvenanceFrameBundle::FullPath,
        summary: MvccProvenanceSummary::KeyPath,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    query.source = MvccReadSource::FullScan;
    assert_eq!(
        first_cuda_slice_query_gap(&query),
        Some(FirstCudaSliceGap::UnsupportedOrder)
    );
}

#[test]
fn first_cuda_slice_query_gap_accepts_all_order_projection_variants_over_resolved_source() {
    let resolved_source = MvccReadSource::FollowValueChainLabeledBranches {
        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
        branches: vec![MvccLabeledValueChainBranch {
            label: "team".to_string(),
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
        }],
        fan_in: MvccValueChainBranchFanIn::AllBranches,
        provenance: MvccSourceProvenance::Seed,
    };
    let supported_cpu_resolved_filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::SourceKeyPrefix("acct:".to_string()),
        MvccReadFilter::ProvenanceBundleLenEquals {
            bundle: MvccProvenanceFrameBundle::FullPath,
            expected_len: 3,
        },
    ]));
    let occurrence_order_expected = vec!["acct:1".to_string()];
    let mixed_order_left_expected = vec!["acct:1".to_string(), "profile:1".to_string()];
    let mixed_order_right_expected = vec!["team:alpha".to_string()];
    let orders = vec![
        MvccReadOrder::KeyAsc,
        MvccReadOrder::KeyDesc,
        MvccReadOrder::ValueAsc,
        MvccReadOrder::ValueDesc,
        MvccReadOrder::BranchLabelAsc,
        MvccReadOrder::BranchLabelDesc,
        MvccReadOrder::SourceKeyAsc,
        MvccReadOrder::SourceKeyDesc,
        MvccReadOrder::SourceValueAsc,
        MvccReadOrder::SourceValueDesc,
        MvccReadOrder::ProvenanceKeyAsc {
            frame: MvccProvenanceFrame::Seed,
        },
        MvccReadOrder::ProvenanceKeyDesc {
            frame: MvccProvenanceFrame::TerminalInput,
        },
        MvccReadOrder::ProvenanceValueAsc {
            frame: MvccProvenanceFrame::Seed,
        },
        MvccReadOrder::ProvenanceValueDesc {
            frame: MvccProvenanceFrame::TerminalInput,
        },
        MvccReadOrder::ProvenanceBundleKeyPathAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
        },
        MvccReadOrder::ProvenanceBundleKeyPathDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
        },
        MvccReadOrder::ProvenanceBundleValuePathAsc {
            bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
        },
        MvccReadOrder::ProvenanceBundleValuePathDesc {
            bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            expected: occurrence_order_expected.clone(),
            occurrence: MvccProvenanceOccurrence::First,
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::ValuePath,
            expected: occurrence_order_expected.clone(),
            occurrence: MvccProvenanceOccurrence::Last,
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            left_expected: occurrence_order_expected.clone(),
            left_occurrence: MvccProvenanceOccurrence::First,
            right_expected: mixed_order_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::Nth(0),
        },
        MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyValuePath,
            left_expected: occurrence_order_expected.clone(),
            left_occurrence: MvccProvenanceOccurrence::Last,
            right_expected: mixed_order_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::First,
        },
        MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            left_expected: mixed_order_left_expected.clone(),
            left_occurrence: MvccProvenanceOccurrence::First,
            right_expected: mixed_order_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::Last,
        },
        MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairDesc {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::ValuePath,
            left_expected: mixed_order_left_expected,
            left_occurrence: MvccProvenanceOccurrence::Nth(1),
            right_expected: mixed_order_right_expected,
            right_occurrence: MvccProvenanceOccurrence::First,
        },
    ];

    for order in orders {
        let mut query = first_cuda_slice_support_query();
        query.source = resolved_source.clone();
        query.filter = supported_cpu_resolved_filter.clone();
        query.order = Some(order);

        assert_eq!(
            first_cuda_slice_query_gap(&query),
            None,
            "{:?}",
            query.order
        );
    }

    let occurrence_projection_expected = vec!["acct:1".to_string()];
    let mixed_projection_left_expected = vec!["acct:1".to_string(), "profile:1".to_string()];
    let mixed_projection_right_expected = vec!["team:alpha".to_string()];
    let projections = vec![
        MvccProjection::KeyValue,
        MvccProjection::KeyOnly,
        MvccProjection::ValueOnly,
        MvccProjection::BranchLabelTargetValue,
        MvccProjection::SourceKeyTargetValue,
        MvccProjection::SourceValueOnly,
        MvccProjection::TargetKeySourceValue,
        MvccProjection::TargetKeyProvenanceValue {
            frame: MvccProvenanceFrame::TerminalInput,
        },
        MvccProjection::TargetKeyProvenanceSummary {
            summary: MvccProvenanceSummary::KeyValuePath,
        },
        MvccProjection::TargetKeyProvenanceBundleSummary {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
        },
        MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyPath,
            expected: occurrence_projection_expected.clone(),
            occurrence: MvccProvenanceOccurrence::First,
        },
        MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::ValuePath,
            left_expected: occurrence_projection_expected,
            left_occurrence: MvccProvenanceOccurrence::First,
            right_expected: mixed_projection_right_expected.clone(),
            right_occurrence: MvccProvenanceOccurrence::Last,
        },
        MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
            bundle: MvccProvenanceFrameBundle::FullPath,
            summary: MvccProvenanceSummary::KeyValuePath,
            left_expected: mixed_projection_left_expected,
            left_occurrence: MvccProvenanceOccurrence::Nth(1),
            right_expected: mixed_projection_right_expected,
            right_occurrence: MvccProvenanceOccurrence::First,
        },
    ];

    for projection in projections {
        let mut query = first_cuda_slice_support_query();
        query.source = resolved_source.clone();
        query.filter = supported_cpu_resolved_filter.clone();
        query.projection = projection;

        assert_eq!(
            first_cuda_slice_query_gap(&query),
            None,
            "{:?}",
            query.projection
        );
    }
}

#[test]
fn first_cuda_slice_query_gap_accepts_source_relative_filters_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChainLabeledBranches {
        keys: vec!["acct:1".to_string()],
        branches: vec![MvccLabeledValueChainBranch {
            label: "team".to_string(),
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
        }],
        fan_in: MvccValueChainBranchFanIn::AllBranches,
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::All(vec![
        MvccReadFilter::SourceKeyPrefix("acct:".to_string()),
        MvccReadFilter::SourceValueEquals("profile:1".to_string()),
        MvccReadFilter::BranchLabelEquals("team".to_string()),
    ]));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_source_relative_order_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChainLabeledBranches {
        keys: vec!["acct:1".to_string()],
        branches: vec![MvccLabeledValueChainBranch {
            label: "team".to_string(),
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
        }],
        fan_in: MvccValueChainBranchFanIn::AllBranches,
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string()));
    query.order = Some(MvccReadOrder::BranchLabelDesc);

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_source_relative_projection_over_resolved_source() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::FollowValueChainLabeledBranches {
        keys: vec!["acct:1".to_string()],
        branches: vec![MvccLabeledValueChainBranch {
            label: "team".to_string(),
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
        }],
        fan_in: MvccValueChainBranchFanIn::AllBranches,
        provenance: MvccSourceProvenance::Seed,
    };
    query.filter = Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string()));
    query.projection = MvccProjection::TargetKeySourceValue;

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_concat_of_cpu_resolved_sources() {
    let mut query = first_cuda_slice_support_query();
    query.source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
        ],
    };
    query.filter = Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string()));

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_gap_labels_are_stable_for_docs_and_future_routing() {
    assert_eq!(
        FirstCudaSliceGap::UnsupportedSource.label(),
        "unsupported_source"
    );
    assert_eq!(
        FirstCudaSliceGap::UnsupportedOrder.label(),
        "unsupported_order"
    );
    assert_eq!(
        FirstCudaSliceGap::UnsupportedProjection.label(),
        "unsupported_projection"
    );
    assert_eq!(
        FirstCudaSliceGap::UnsupportedFilter.label(),
        "unsupported_filter"
    );
    assert_eq!(
        FirstCudaSliceGap::EmptyLogicalFilterTree.label(),
        "empty_logical_filter_tree"
    );
}


#[test]
fn execute_mvcc_query_keeps_result_contract_stable_across_backend_swap() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();

    let backend = RecordingMvccBackend {
        executed_target: DeviceTarget::Gpu(0),
        rows: vec![MvccReadRow {
            source_key: Some("seed:acct".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("open".to_string()),
        }],
    };

    let result = e
        .execute_mvcc_query_with_backend(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 1 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &backend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(result.rows, backend.rows);
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn cuda_native_full_scan_resolution_feeds_all_versions_to_visibility_kernel() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let kv = e.read_state.mvcc.load_kv();
    let rows = resolve_mvcc_all_versions(kv.get(), StorageVisibility { read_txn_id: 2 }).unwrap();
    let identities = rows
        .iter()
        .map(|row| {
            (
                row.tuple.key.as_str(),
                row.tuple.value.as_str(),
                row.tuple.created_by,
                row.tuple.deleted_by,
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        identities,
        vec![
            ("acct:1", "open", 1, Some(3)),
            ("acct:1", "closed", 3, None),
            ("acct:2", "hold", 2, Some(4)),
        ]
    );
}

#[test]
fn cuda_native_full_scan_fallback_re_resolves_cpu_visible_rows() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_key_lookup_fallback_re_resolves_cpu_visible_row() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:1".to_string()),
            value: Some("open".to_string()),
        }]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_key_batch_fallback_preserves_request_order() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_composition_fallback_re_resolves_cpu_visible_rows() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::IntersectAll {
                    sources: vec![
                        MvccReadSource::KeyBatchLookup {
                            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                        },
                        MvccReadSource::KeyBatchLookup {
                            keys: vec!["acct:2".to_string(), "acct:3".to_string()],
                        },
                    ],
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:2".to_string()),
            value: None,
        }]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn cuda_native_follow_value_chain_fallback_re_resolves_cpu_visible_rows() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET profile:1=team:beta").unwrap();

    let result = e
        .execute_cuda_native_source_query_kv(
            &MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::TargetKeySourceValue,
                limit: None,
            },
            &CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0),
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("profile:1".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_runs_supported_lookup_without_fallback() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 1 },
                filter: Some(MvccReadFilter::ValueEquals("open".to_string())),
                order: None,
                projection: MvccProjection::ValueOnly,
                limit: None,
            },
            &FirstCudaSliceParityBackend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: Some("open".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_runs_supported_full_scan_without_fallback() {
    let mut e = Engine::new_local();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::FullScan,
                visibility: StorageVisibility { read_txn_id: 6 },
                filter: Some(MvccReadFilter::All(vec![
                    MvccReadFilter::KeyPrefix("acct:".to_string()),
                    MvccReadFilter::Any(vec![
                        MvccReadFilter::ValueEquals("closed".to_string()),
                        MvccReadFilter::ValueEquals("archived".to_string()),
                    ]),
                ])),
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &FirstCudaSliceParityBackend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("archived".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_orders_cpu_resolved_rows_by_value_without_fallback()
{
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 4 },
                filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
                order: Some(MvccReadOrder::ValueDesc),
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &FirstCudaSliceParityBackend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn cuda_mvcc_backend_falls_back_when_driver_is_unavailable() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    let backend = CudaMvccExecutionBackend::new(CudaDriverRuntime::unavailable(), 0);

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
                visibility: StorageVisibility { read_txn_id: 1 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyValue,
                limit: None,
            },
            &backend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(result.fallback_reason, Some(FallbackReason::GpuUnavailable));
    assert_eq!(result.rows.len(), 1);
    assert_eq!(e.metrics().fallback_for(FallbackReason::GpuUnavailable), 1);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_historical_key_lookup_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:1".to_string()),
            value: Some("open".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_key_batch_lookup_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:3".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    let kv = e.read_state.mvcc.load_kv();
    let all_version_rows =
        resolve_mvcc_all_versions(kv.get(), StorageVisibility { read_txn_id: 3 }).unwrap();
    let compact_rows = ["acct:3", "acct:1"]
        .iter()
        .flat_map(|key| {
            all_version_rows
                .iter()
                .filter(move |row| row.tuple.key == *key)
                .cloned()
        })
        .collect::<Vec<_>>();
    let compact_h2d_bytes = cuda_mvcc_row_batch_transfer_bytes(&compact_rows);

    assert_eq!(e.metrics().snapshot().h2d_bytes_total, compact_h2d_bytes);
    assert!(compact_h2d_bytes < cuda_mvcc_row_batch_transfer_bytes(&all_version_rows));
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_follow_value_chain_source_resolution_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET profile:1=team:alpha-v2").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_concat_native_sources_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();
    e.execute_text(4, "SET acct:4=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "acct:2".to_string(),
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:4".to_string(), "acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_concat_limit_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();
    e.execute_text(4, "SET acct:4=closed").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "acct:2".to_string(),
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:4".to_string(), "acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_key_order_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_value_order_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_fan_in_order_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=pending").unwrap();
    e.execute_text(4, "SET acct:4=archived").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::KeyLookup {
                        key: "acct:4".to_string(),
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyOnly,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_ordered_limit_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::KeyValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_general_key_range_filter_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:0=cold").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:7=hold").unwrap();
    e.execute_text(4, "SET acct:9=closed").unwrap();
    e.execute_text(5, "SET user:1=active").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::KeyRange {
                start_inclusive: "acct:1".to_string(),
                end_exclusive: "acct:9".to_string(),
            }),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:7".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_prefix_equivalent_key_range_filter_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyRange {
                start_inclusive: "acct:".to_string(),
                end_exclusive: "acct;".to_string(),
            }),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_key_prefix_filter_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_string_value_equals_filter_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ValueEquals("open".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_numeric_value_equals_filter_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=7").unwrap();
    e.execute_text(2, "SET acct:2=8").unwrap();
    e.execute_text(3, "SET acct:3=7").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ValueEquals("7".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("7".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("7".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_filterless_full_scan_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_historical_visibility_mask_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:1=closed").unwrap();
    e.execute_text(4, "DELETE acct:2").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_logical_supported_filters_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:0=cold").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:2=hold").unwrap();
    e.execute_text(4, "SET acct:3=closed").unwrap();
    e.execute_text(5, "SET user:1=open").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyRange {
                    start_inclusive: "acct:1".to_string(),
                    end_exclusive: "acct:4".to_string(),
                },
                MvccReadFilter::Any(vec![
                    MvccReadFilter::ValueEquals("open".to_string()),
                    MvccReadFilter::ValueEquals("hold".to_string()),
                ]),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_filters_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();
    e.execute_text(7, "SET team:beta:2=Bianca").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::ProvenanceKeyPrefix {
                    frame: MvccProvenanceFrame::Seed,
                    prefix: "acct:".to_string(),
                },
                MvccReadFilter::ProvenanceValueEquals {
                    frame: MvccProvenanceFrame::TerminalInput,
                    expected: "team:beta".to_string(),
                },
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_source_relative_filters_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                branches: vec![MvccLabeledValueChainBranch {
                    label: "team".to_string(),
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                }],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::SourceKeyPrefix("acct:".to_string()),
                MvccReadFilter::SourceValueEquals("profile:1".to_string()),
                MvccReadFilter::BranchLabelEquals("team".to_string()),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("Alpha Team".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_source_relative_order_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=member:1").unwrap();
    e.execute_text(4, "SET member:1=Alice").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:1".to_string()],
                branches: vec![
                    MvccLabeledValueChainBranch {
                        label: "member".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                    },
                    MvccLabeledValueChainBranch {
                        label: "team".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::BranchLabelDesc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("member:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("member:1".to_string()),
                value: Some("Alice".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_cpu_resolved_key_value_order_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_source_relative_projection_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=Alpha Team").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:1".to_string()],
                branches: vec![MvccLabeledValueChainBranch {
                    label: "team".to_string(),
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                }],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_concat_cpu_resolved_sources_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:2".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_distinct_cpu_resolved_sources_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

fn seed_native_composition_rows(e: &mut Engine) {
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();
}

fn native_set_composition_cases() -> Vec<(&'static str, MvccReadSource, Vec<&'static str>)> {
    let left = MvccReadSource::KeyBatchLookup {
        keys: vec![
            "acct:1".to_string(),
            "acct:2".to_string(),
            "acct:2".to_string(),
        ],
    };
    let right = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::KeyLookup {
                key: "acct:2".to_string(),
            },
            MvccReadSource::KeyLookup {
                key: "acct:3".to_string(),
            },
        ],
    };
    let sources = || vec![left.clone(), right.clone()];

    vec![
        (
            "concat_distinct",
            MvccReadSource::ConcatDistinct { sources: sources() },
            vec!["acct:1", "acct:2", "acct:3"],
        ),
        (
            "intersect_distinct",
            MvccReadSource::IntersectDistinct { sources: sources() },
            vec!["acct:2"],
        ),
        (
            "intersect_all",
            MvccReadSource::IntersectAll { sources: sources() },
            vec!["acct:2"],
        ),
        (
            "except_distinct",
            MvccReadSource::ExceptDistinct { sources: sources() },
            vec!["acct:1"],
        ),
        (
            "except_all",
            MvccReadSource::ExceptAll { sources: sources() },
            vec!["acct:1", "acct:2"],
        ),
        (
            "symmetric_difference_distinct",
            MvccReadSource::SymmetricDifferenceDistinct { sources: sources() },
            vec!["acct:1", "acct:3"],
        ),
        (
            "symmetric_difference_all",
            MvccReadSource::SymmetricDifferenceAll { sources: sources() },
            vec!["acct:1", "acct:2", "acct:3"],
        ),
    ]
}

fn key_only_rows(keys: &[&str]) -> Vec<MvccReadRow> {
    keys.iter()
        .map(|key| MvccReadRow {
            source_key: None,
            key: Some((*key).to_string()),
            value: None,
        })
        .collect()
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_native_set_composition_without_fallback() {
    let mut e = Engine::new_local();
    seed_native_composition_rows(&mut e);

    for (name, source, expected_keys) in native_set_composition_cases() {
        let result = e
            .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
                source,
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: None,
                order: Some(MvccReadOrder::KeyAsc),
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

        assert_eq!(result.planned_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(result.fallback_reason, None, "{name}");
        assert_eq!(result.rows, key_only_rows(&expected_keys), "{name}");
    }

    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_nested_native_composition_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![MvccReadSource::ConcatDistinct {
                    sources: vec![
                        MvccReadSource::KeyLookup {
                            key: "acct:1".to_string(),
                        },
                        MvccReadSource::KeyLookup {
                            key: "acct:2".to_string(),
                        },
                    ],
                }],
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_bundle_filters_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::ProvenanceBundleKeyPrefix {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    prefix: "profile:".to_string(),
                },
                MvccReadFilter::ProvenanceBundleValueCountAtLeast {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    expected: "team:beta".to_string(),
                    min_count: 1,
                },
                MvccReadFilter::ProvenanceBundleKeyValueEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    key: "profile:2".to_string(),
                    value: "team:beta".to_string(),
                },
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_bundle_path_filters_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::ProvenanceBundlePathContains {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    expected: vec!["profile:2".to_string(), "team:beta".to_string()],
                },
                MvccReadFilter::ProvenanceBundlePathSegmentEquals {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::ValuePath,
                    index: 1,
                    expected: "team:beta".to_string(),
                },
                MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::ValuePath,
                    expected: vec!["team:beta".to_string(), "member:2".to_string()],
                },
                MvccReadFilter::ProvenanceBundleLenEquals {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    expected_len: 3,
                },
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_bundle_occurrence_path_filters_without_fallback()
{
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    distance: 3,
                    expected: vec!["acct:1".to_string()],
                },
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    distance: 2,
                },
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_provenance_projection_order_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=member:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 3,
            }),
            order: Some(MvccReadOrder::ProvenanceValueDesc {
                frame: MvccProvenanceFrame::TerminalInput,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("acct:2 -> profile:2 -> team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("acct:1 -> profile:1 -> team:alpha".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_prefix_terminal_value_chain_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: Some(MvccReadFilter::ProvenanceKeyPrefix {
                frame: MvccProvenanceFrame::TerminalInput,
                prefix: "profile:".to_string(),
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_labeled_branch_source_resolution_without_fallback() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();

    let result = e
        .execute_mvcc_query_with_cuda_driver_probe(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                branches: vec![
                    MvccLabeledValueChainBranch {
                        label: "missing".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                    },
                    MvccLabeledValueChainBranch {
                        label: "members".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::BranchLabelEquals("members".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::BranchLabelTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alpha Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Beta Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Bob".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_supported_lookup_fixture() {
    let mut e = Engine::new_local();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-read-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let query = MvccReadQuery {
        source: MvccReadSource::KeyLookup {
            key: "acct:1".to_string(),
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::ValueEquals("closed".to_string())),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_supported_full_scan_fixture() {
    let mut e = Engine::new_local();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::KeyPrefix("acct:".to_string()),
            MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("closed".to_string()),
                MvccReadFilter::ValueEquals("archived".to_string()),
            ]),
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_supported_key_range_filter() {
    let mut e = Engine::new_local();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::KeyRange {
            start_inclusive: "acct:1".to_string(),
            end_exclusive: "acct:4".to_string(),
        }),
        order: None,
        projection: MvccProjection::KeyOnly,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_concat_native_sources() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=open").unwrap();
    e.execute_text(4, "SET acct:4=closed").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyLookup {
                    key: "acct:2".to_string(),
                },
                MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:4".to_string(), "acct:1".to_string()],
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn first_cuda_slice_query_gap_accepts_distinct_cpu_resolved_composition() {
    let query = MvccReadQuery {
        source: MvccReadSource::ConcatDistinct {
            sources: vec![
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::SourceKeyAsc),
        projection: MvccProjection::TargetKeySourceValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn first_cuda_slice_query_gap_accepts_native_set_composition() {
    for (name, source, _) in native_set_composition_cases() {
        let query = MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: None,
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyOnly,
            limit: None,
        };

        assert_eq!(first_cuda_slice_query_gap(&query), None, "{name}");
    }
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_native_set_composition_variants() {
    for (name, source, expected_keys) in native_set_composition_cases() {
        let mut cpu_engine = Engine::new_local();
        seed_native_composition_rows(&mut cpu_engine);
        let query = MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: None,
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyOnly,
            limit: None,
        };

        let cpu = cpu_engine.execute_mvcc_query(&query).unwrap();
        assert_mvcc_query_uses_tracked_cpu_fallback(&cpu_engine, &cpu, 1);

        let mut backend_engine = Engine::new_local();
        seed_native_composition_rows(&mut backend_engine);
        let backend = backend_engine
            .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
            .unwrap();

        assert_eq!(backend.planned_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(backend.executed_target, DeviceTarget::Gpu(0), "{name}");
        assert_eq!(backend.fallback_reason, None, "{name}");
        assert_eq!(backend.rows, cpu.rows, "{name}");
        assert_eq!(backend.rows, key_only_rows(&expected_keys), "{name}");
        assert_eq!(
            backend_engine.metrics().snapshot().fallback_total,
            0,
            "{name}"
        );
    }
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_distinct_cpu_resolved_sources() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::ConcatDistinct {
            sources: vec![
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::SourceKeyAsc),
        projection: MvccProjection::TargetKeySourceValue,
        limit: None,
    };

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_key_order() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::KeyDesc),
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn first_cuda_slice_query_gap_accepts_value_order_for_native_single_sources() {
    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_value_order() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("pending".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn first_cuda_slice_query_gap_accepts_fan_in_order_for_native_sources() {
    let key_batch = MvccReadQuery {
        source: MvccReadSource::KeyBatchLookup {
            keys: vec!["acct:3".to_string(), "acct:1".to_string()],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyOnly,
        limit: Some(2),
    };
    assert_eq!(first_cuda_slice_query_gap(&key_batch), None);

    let concat = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                },
                MvccReadSource::KeyLookup {
                    key: "acct:4".to_string(),
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyOnly,
        limit: Some(3),
    };
    assert_eq!(first_cuda_slice_query_gap(&concat), None);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_fan_in_order() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=pending").unwrap();
    e.execute_text(4, "SET acct:4=archived").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyBatchLookup {
                    keys: vec!["acct:3".to_string(), "acct:1".to_string()],
                },
                MvccReadSource::KeyLookup {
                    key: "acct:4".to_string(),
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::ValueAsc),
        projection: MvccProjection::KeyOnly,
        limit: Some(3),
    };

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_limit_after_order() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:3=closed").unwrap();
    e.execute_text(3, "SET acct:2=pending").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::KeyDesc),
        projection: MvccProjection::KeyOnly,
        limit: Some(2),
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn mvcc_benchmark_report_summarizes_gpu_coverage_and_fallback_rate() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=closed").unwrap();
    e.execute_text(3, "SET acct:3=pending").unwrap();

    let supported_scan = MvccReadQuery {
        source: MvccReadSource::FullScan,
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::KeyAsc),
        projection: MvccProjection::KeyOnly,
        limit: None,
    };
    let supported_concat = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::KeyLookup {
                    key: "acct:2".to_string(),
                },
                MvccReadSource::KeyLookup {
                    key: "acct:1".to_string(),
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: None,
        order: None,
        projection: MvccProjection::ValueOnly,
        limit: None,
    };
    let supported_nested_distinct = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "acct:1".to_string(),
                    },
                    MvccReadSource::KeyLookup {
                        key: "acct:1".to_string(),
                    },
                ],
            }],
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: None,
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    let results = vec![
        e.execute_mvcc_query_with_backend_fallback(&supported_scan, &FirstCudaSliceParityBackend)
            .unwrap(),
        e.execute_mvcc_query_with_backend_fallback(&supported_concat, &FirstCudaSliceParityBackend)
            .unwrap(),
        e.execute_mvcc_query_with_backend_fallback(
            &supported_nested_distinct,
            &FirstCudaSliceParityBackend,
        )
        .unwrap(),
    ];

    let report = MvccBenchmarkReport::from_results(&results, &e.metrics().snapshot());

    assert_eq!(report.workload_count, 3);
    assert_eq!(report.gpu_executed_count, 3);
    assert_eq!(report.cpu_fallback_count, 0);
    assert_eq!(report.gpu_executed_permyriad, 10_000);
    assert_eq!(report.cpu_fallback_permyriad, 0);
    assert!(report.d2h_bytes_total > 0);
    assert_eq!(report.h2d_bytes_total, 0);
    assert_eq!(report.kernel_exec_samples, 0);
    assert_eq!(report.batch_wait_samples, 0);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_filters() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();
    e.execute_text(7, "SET team:beta:2=Bianca").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 7 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceKeyPrefix {
                frame: MvccProvenanceFrame::Seed,
                prefix: "acct:".to_string(),
            },
            MvccReadFilter::ProvenanceValueEquals {
                frame: MvccProvenanceFrame::TerminalInput,
                expected: "team:beta".to_string(),
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_source_relative_filters() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChainLabeledBranches {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            branches: vec![MvccLabeledValueChainBranch {
                label: "team".to_string(),
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
            }],
            fan_in: MvccValueChainBranchFanIn::AllBranches,
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 6 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::SourceKeyPrefix("acct:".to_string()),
            MvccReadFilter::SourceValueEquals("profile:1".to_string()),
            MvccReadFilter::BranchLabelEquals("team".to_string()),
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("Alpha Team".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_source_relative_order() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=member:1").unwrap();
    e.execute_text(4, "SET member:1=Alice").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChainLabeledBranches {
            keys: vec!["acct:1".to_string()],
            branches: vec![
                MvccLabeledValueChainBranch {
                    label: "member".to_string(),
                    plan: MvccValueChainPlan {
                        value_key_hops: 3,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                },
                MvccLabeledValueChainBranch {
                    label: "team".to_string(),
                    plan: MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                },
            ],
            fan_in: MvccValueChainBranchFanIn::AllBranches,
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: Some(MvccReadOrder::BranchLabelDesc),
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("member:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("member:1".to_string()),
                value: Some("Alice".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_source_relative_projection() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=Alpha Team").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChainLabeledBranches {
            keys: vec!["acct:1".to_string()],
            branches: vec![MvccLabeledValueChainBranch {
                label: "team".to_string(),
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
            }],
            fan_in: MvccValueChainBranchFanIn::AllBranches,
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 3 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: None,
        projection: MvccProjection::TargetKeySourceValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_concat_cpu_resolved_sources() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::Concat {
            sources: vec![
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                MvccReadSource::FollowValueChain {
                    keys: vec!["acct:2".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
            ],
        },
        visibility: StorageVisibility { read_txn_id: 4 },
        filter: Some(MvccReadFilter::SourceKeyPrefix("acct:".to_string())),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_bundle_filters() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 8 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceBundleKeyEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected: "profile:2".to_string(),
            },
            MvccReadFilter::ProvenanceBundleKeyPrefix {
                bundle: MvccProvenanceFrameBundle::FullPath,
                prefix: "profile:".to_string(),
            },
            MvccReadFilter::ProvenanceBundleValueEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected: "team:beta".to_string(),
            },
            MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                key: "profile:2".to_string(),
                value: "team:beta".to_string(),
                min_count: 1,
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_bundle_path_filters() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 8 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceBundlePathContains {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["profile:2".to_string(), "team:beta".to_string()],
            },
            MvccReadFilter::ProvenanceBundlePathSegmentEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
                index: 1,
                expected: "team:beta".to_string(),
            },
            MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::ValuePath,
                expected: vec!["team:beta".to_string(), "member:2".to_string()],
            },
            MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 3,
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("member:2".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_matches_cpu_on_provenance_bundle_occurrence_path_filters(
) {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();

    let query = MvccReadQuery {
        source: MvccReadSource::FollowValueChain {
            keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            plan: MvccValueChainPlan {
                value_key_hops: 3,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            provenance: MvccSourceProvenance::Seed,
        },
        visibility: StorageVisibility { read_txn_id: 5 },
        filter: Some(MvccReadFilter::All(vec![
            MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 3,
                expected: vec!["acct:1".to_string()],
            },
            MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                right_occurrence_index: 0,
                right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                distance: 2,
            },
        ])),
        order: None,
        projection: MvccProjection::KeyValue,
        limit: None,
    };

    assert_eq!(first_cuda_slice_query_gap(&query), None);

    let cpu = e.execute_mvcc_query(&query).unwrap();
    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &cpu, 1);

    let backend = e
        .execute_mvcc_query_with_backend_fallback(&query, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(backend.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(backend.fallback_reason, None);
    assert_eq!(backend.rows, cpu.rows);
    assert_eq!(
        backend.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 1);
}

#[test]
fn execute_mvcc_query_first_cuda_slice_backend_runs_nested_native_composition() {
    let mut e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=hold").unwrap();

    let result = e
        .execute_mvcc_query_with_backend_fallback(
            &MvccReadQuery {
                source: MvccReadSource::Concat {
                    sources: vec![MvccReadSource::ConcatDistinct {
                        sources: vec![
                            MvccReadSource::KeyLookup {
                                key: "acct:1".to_string(),
                            },
                            MvccReadSource::KeyLookup {
                                key: "acct:2".to_string(),
                            },
                        ],
                    }],
                },
                visibility: StorageVisibility { read_txn_id: 2 },
                filter: None,
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            },
            &FirstCudaSliceParityBackend,
        )
        .unwrap();

    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
    assert_eq!(e.metrics().snapshot().fallback_total, 0);
}

#[test]
fn execute_mvcc_query_replays_deterministic_workload_fixture_for_point_lookup() {
    let e = Engine::new_local();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-read-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let historical = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::ValueOnly,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &historical, 1);

    assert_eq!(
        historical.rows,
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: Some("open".to_string()),
        }]
    );

    let current = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "user:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::ValueEquals("active".to_string())),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &current, 2);

    assert_eq!(
        current.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:1".to_string()),
            value: None,
        }]
    );
}

#[test]
fn execute_mvcc_query_replays_deterministic_full_scan_workload_fixture() {
    let e = Engine::new_local();
    for (txn_id, command) in include_str!("../../../../tests/fixtures/mvcc-full-scan-workload.txt")
        .lines()
        .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let historical = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::Any(vec![
                    MvccReadFilter::ValueEquals("open".to_string()),
                    MvccReadFilter::ValueEquals("hold".to_string()),
                ]),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &historical, 1);
    assert_eq!(
        historical.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("open".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("hold".to_string()),
            },
        ]
    );

    let current = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::Any(vec![
                    MvccReadFilter::ValueEquals("closed".to_string()),
                    MvccReadFilter::ValueEquals("archived".to_string()),
                ]),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &current, 2);
    assert_eq!(
        current.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("archived".to_string()),
            },
        ]
    );

    let status = e.status_snapshot();
    assert_eq!(status.fallback.gpu_parity_fallback_total(), 2);
    assert_eq!(
        status.latest_fallback_reason(),
        Some(FallbackReason::GpuMvccReadParityGap)
    );
}

#[test]
fn execute_mvcc_query_replays_deterministic_source_composition_workload_fixture() {
    let e = Engine::new_local();
    for (txn_id, command) in
        include_str!("../../../../tests/fixtures/mvcc-source-composition-workload.txt")
            .lines()
            .enumerate()
    {
        e.execute_text((txn_id + 1) as u64, command).unwrap();
    }

    let multiset_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectAll {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:1".to_string(),
                            "user:2".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &multiset_overlap, 1);

    assert_eq!(
        multiset_overlap.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:1".to_string()),
            value: None,
        }]
    );

    let multiset_imbalance = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceAll {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_mvcc_query_uses_tracked_cpu_fallback(&e, &multiset_imbalance, 2);

    assert_eq!(
        multiset_imbalance.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );

    let status = e.status_snapshot();
    assert_eq!(status.fallback.gpu_parity_fallback_total(), 2);
    assert_eq!(
        status.latest_fallback_reason(),
        Some(FallbackReason::GpuMvccReadParityGap)
    );
}

#[test]
fn execute_mvcc_query_supports_multi_key_lookup_fan_in_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();
    e.execute_text(4, "SET acct:1=closed").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec![
                    "user:1".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:2".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "user:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("closed".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: Some("active".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_concat_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(9, "SET user:1=active").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "user:1".to_string(),
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["acct:2".to_string(), "missing".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    },
                    MvccReadSource::KeyLookup {
                        key: "user:1".to_string(),
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::SourceKeyPrefix("acct:1".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_concat_distinct_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(9, "SET user:1=active").unwrap();

    let deduped = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::KeyLookup {
                        key: "user:1".to_string(),
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "acct:2".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        deduped.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: None,
            },
        ]
    );

    let source_distinction = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_distinction.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_intersect_distinct_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:alpha").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(9, "SET team:beta:1=Bob").unwrap();
    e.execute_text(10, "SET user:1=active").unwrap();

    let exact_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectDistinct {
                sources: vec![
                    MvccReadSource::Concat {
                        sources: vec![
                            MvccReadSource::KeyLookup {
                                key: "user:1".to_string(),
                            },
                            MvccReadSource::KeyBatchLookup {
                                keys: vec!["acct:2".to_string(), "user:1".to_string()],
                            },
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_overlap.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );

    let source_sensitive_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:3".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:3".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: Some(MvccReadFilter::ValueEquals("Alice".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_sensitive_overlap.rows,
        vec![MvccReadRow {
            source_key: Some("acct:3".to_string()),
            key: Some("team:alpha:1".to_string()),
            value: Some("profile:3".to_string()),
        }]
    );
}

#[test]
fn execute_mvcc_query_supports_except_distinct_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:alpha").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(9, "SET team:beta:1=Bob").unwrap();
    e.execute_text(10, "SET user:1=active").unwrap();
    e.execute_text(11, "SET user:2=locked").unwrap();

    let exact_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptDistinct {
                sources: vec![
                    MvccReadSource::Concat {
                        sources: vec![
                            MvccReadSource::KeyBatchLookup {
                                keys: vec![
                                    "user:1".to_string(),
                                    "user:2".to_string(),
                                    "acct:2".to_string(),
                                ],
                            },
                            MvccReadSource::KeyLookup {
                                key: "user:1".to_string(),
                            },
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_difference.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:2".to_string()),
            value: None,
        }]
    );

    let source_sensitive_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:3".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:3".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_sensitive_difference.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_symmetric_difference_distinct_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:alpha").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(9, "SET team:beta:1=Bob").unwrap();
    e.execute_text(10, "SET user:1=active").unwrap();
    e.execute_text(11, "SET user:2=locked").unwrap();
    e.execute_text(12, "SET user:3=standby").unwrap();

    let exact_uniques = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceDistinct {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:2".to_string(),
                            "user:2".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "user:3".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 12 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_uniques.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:3".to_string()),
                value: None,
            },
        ]
    );

    let source_sensitive_uniques = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceDistinct {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:3".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:3".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 12 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Ally".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_sensitive_uniques.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_intersect_all_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET user:1=active").unwrap();
    e.execute_text(9, "SET user:2=locked").unwrap();

    let exact_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectAll {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:2".to_string(),
                            "user:1".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "user:1".to_string()],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string(), "user:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_overlap.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("user:1".to_string()),
            value: None,
        }]
    );

    let join_overlap = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::IntersectAll {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        join_overlap.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_except_all_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET user:1=active").unwrap();
    e.execute_text(9, "SET user:2=locked").unwrap();

    let exact_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptAll {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:1".to_string(),
                            "user:2".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_difference.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:2".to_string()),
                value: None,
            },
        ]
    );

    let join_difference = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ExceptAll {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        join_difference.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_prefixes_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team:alpha:").unwrap();
    e.execute_text(9, "SET prefix:beta:=team:beta:").unwrap();
    e.execute_text(10, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(11, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(12, "SET team:beta:1=Bob").unwrap();
    e.execute_text(13, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(14, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(15, "SET prefix:alpha:v2=team:alpha:v2:")
        .unwrap();
    e.execute_text(16, "SET team:alpha:v2:1=Astra").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_value_key_refs_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=person:1").unwrap();
    e.execute_text(11, "SET team-lead:2=person:2").unwrap();
    e.execute_text(12, "SET person:1=Alice").unwrap();
    e.execute_text(13, "SET person:2=Bob").unwrap();
    e.execute_text(14, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(15, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(16, "SET team-lead:3=person:3").unwrap();
    e.execute_text(17, "SET person:3=Astra").unwrap();
    e.execute_text(18, "SET team-root:3=prefix:ghost:").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 18 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("person:2".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("person:3".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 18 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("person:2".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("person:3".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("person:3".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_value_key_prefixes_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=team:alpha:").unwrap();
    e.execute_text(11, "SET team-lead:2=team:beta:").unwrap();
    e.execute_text(12, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(13, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(14, "SET team:beta:1=Bob").unwrap();
    e.execute_text(15, "SET team:beta:2=Bianca").unwrap();
    e.execute_text(16, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(17, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(18, "SET team-lead:3=team:alpha:v2:")
        .unwrap();
    e.execute_text(19, "SET team:alpha:v2:1=Astra").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 19 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 19 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_ref_value_key_ref_prefixes_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=group-prefix:alpha:")
        .unwrap();
    e.execute_text(11, "SET team-lead:2=group-prefix:beta:")
        .unwrap();
    e.execute_text(12, "SET group-prefix:alpha:=squad:alpha:")
        .unwrap();
    e.execute_text(13, "SET group-prefix:beta:=squad:beta:")
        .unwrap();
    e.execute_text(14, "SET squad:alpha:1=Alice").unwrap();
    e.execute_text(15, "SET squad:alpha:2=Ally").unwrap();
    e.execute_text(16, "SET squad:beta:1=Bob").unwrap();
    e.execute_text(17, "SET squad:beta:2=Bianca").unwrap();
    e.execute_text(18, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(19, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(20, "SET team-lead:3=group-prefix:alpha:v2:")
        .unwrap();
    e.execute_text(21, "SET group-prefix:alpha:v2:=squad:alpha:v2:")
        .unwrap();
    e.execute_text(22, "SET squad:alpha:v2:1=Astra").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 22 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("squad:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("squad:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("squad:alpha:v2:1".to_string()),
                value: Some("Astra".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 22 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Astra".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("squad:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("squad:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("squad:alpha:v2:1".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_generic_follow_value_chain_plan() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team-root:1").unwrap();
    e.execute_text(5, "SET profile:2=team-root:2").unwrap();
    e.execute_text(6, "SET team-root:1=prefix:alpha:").unwrap();
    e.execute_text(7, "SET team-root:2=prefix:beta:").unwrap();
    e.execute_text(8, "SET prefix:alpha:=team-lead:1").unwrap();
    e.execute_text(9, "SET prefix:beta:=team-lead:2").unwrap();
    e.execute_text(10, "SET team-lead:1=group-prefix:alpha:")
        .unwrap();
    e.execute_text(11, "SET team-lead:2=group-prefix:beta:")
        .unwrap();
    e.execute_text(12, "SET group-prefix:alpha:=squad:alpha:")
        .unwrap();
    e.execute_text(13, "SET group-prefix:beta:=squad:beta:")
        .unwrap();
    e.execute_text(14, "SET squad:alpha:1=Alice").unwrap();
    e.execute_text(15, "SET squad:alpha:2=Ally").unwrap();
    e.execute_text(16, "SET squad:beta:1=Bob").unwrap();
    e.execute_text(17, "SET squad:beta:2=Bianca").unwrap();
    e.execute_text(18, "SET team-root:1=prefix:alpha:v2")
        .unwrap();
    e.execute_text(19, "SET prefix:alpha:v2=team-lead:3")
        .unwrap();
    e.execute_text(20, "SET team-lead:3=group-prefix:alpha:v2:")
        .unwrap();
    e.execute_text(21, "SET group-prefix:alpha:v2:=squad:alpha:v2:")
        .unwrap();
    e.execute_text(22, "SET squad:alpha:v2:1=Astra").unwrap();
    e.execute_text(23, "SET squad:alpha:v2:1=talent:1").unwrap();
    e.execute_text(24, "SET squad:beta:1=talent:2").unwrap();
    e.execute_text(25, "SET talent:1=Architect").unwrap();
    e.execute_text(26, "SET talent:2=Builder").unwrap();

    let specialized_prefix = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    let generic_prefix = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(generic_prefix.rows, specialized_prefix.rows);

    let specialized_terminal = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    let generic_terminal = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 26 },
            filter: None,
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(generic_terminal.rows, specialized_terminal.rows);
}

#[test]
fn execute_mvcc_query_supports_follow_value_chain_branches_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(7, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();
    e.execute_text(9, "SET team:beta:2=Bianca").unwrap();

    let branch_grouped = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "acct:3".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        branch_grouped.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("Ally".to_string()),
            },
        ]
    );

    let branch_concat = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec![
                            "acct:2".to_string(),
                            "acct:1".to_string(),
                            "acct:3".to_string(),
                        ],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec![
                            "acct:2".to_string(),
                            "acct:1".to_string(),
                            "acct:3".to_string(),
                        ],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_ne!(branch_concat.rows, branch_grouped.rows);
    assert_eq!(
        branch_concat.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("Bianca".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("Ally".to_string()),
            },
        ]
    );

    let ordered_projection = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Ally".to_string()),
                MvccReadFilter::ValueEquals("team:beta".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyDesc),
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordered_projection.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_chain_branch_first_non_empty_fan_in() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET acct:4=profile:4").unwrap();
    e.execute_text(5, "SET profile:1=team:alpha").unwrap();
    e.execute_text(6, "SET profile:2=team:beta").unwrap();
    e.execute_text(7, "SET profile:4=team:delta").unwrap();
    e.execute_text(8, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(9, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(10, "SET team:beta:1=Bob").unwrap();
    e.execute_text(11, "SET team:delta:1=Dora").unwrap();

    let first_non_empty = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:3".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "acct:4".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: None,
            order: None,
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_non_empty.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:4".to_string()),
                key: Some("acct:4".to_string()),
                value: Some("Dora".to_string()),
            },
        ]
    );

    let all_branches = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:3".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "acct:4".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("team:alpha".to_string()),
                MvccReadFilter::ValueEquals("team:beta".to_string()),
                MvccReadFilter::ValueEquals("team:delta".to_string()),
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
                MvccReadFilter::ValueEquals("Bob".to_string()),
                MvccReadFilter::ValueEquals("Dora".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        all_branches.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Ally".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:4".to_string()),
                key: Some("acct:4".to_string()),
                value: Some("team:delta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:4".to_string()),
                key: Some("acct:4".to_string()),
                value: Some("Dora".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_chain_terminal_input_provenance() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::SourceValueEquals("team:beta".to_string())),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_branch_fan_in_with_terminal_input_provenance() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=profile:3").unwrap();
    e.execute_text(4, "SET profile:1=team:alpha").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();
    e.execute_text(6, "SET profile:3=team:gamma").unwrap();
    e.execute_text(7, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();
    e.execute_text(9, "SET team:beta:2=Bianca").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainBranches {
                keys: vec![
                    "acct:3".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
                plans: vec![
                    MvccValueChainPlan {
                        value_key_hops: 2,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    MvccValueChainPlan {
                        value_key_hops: 1,
                        terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("profile:".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_multi_frame_provenance_filters_and_projection() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();
    e.execute_text(7, "SET team:beta:2=Bianca").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::ProvenanceKeyPrefix {
                    frame: MvccProvenanceFrame::Seed,
                    prefix: "acct:2".to_string(),
                },
                MvccReadFilter::ProvenanceValueEquals {
                    frame: MvccProvenanceFrame::TerminalInput,
                    expected: "team:beta".to_string(),
                },
            ])),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceValue {
                frame: MvccProvenanceFrame::TerminalInput,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_preserves_multi_frame_provenance_identity_under_concat_distinct() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:shared").unwrap();
    e.execute_text(3, "SET team:shared=member:1").unwrap();
    e.execute_text(4, "SET member:1=Alice").unwrap();

    let distinct = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("team:shared".to_string())),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceValue {
                frame: MvccProvenanceFrame::ValueHop(2),
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        distinct.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:shared".to_string()),
                value: Some("member:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:shared".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_provenance_path_summary_projection() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Aria").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: None,
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceSummary {
                summary: MvccProvenanceSummary::KeyValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("acct:1=profile:1 -> profile:1=team:alpha".to_string(),),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("acct:1=profile:1 -> profile:1=team:alpha".to_string(),),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("acct:2=profile:2 -> profile:2=team:beta".to_string(),),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_provenance_summary_projection_keeps_non_join_shapes_stable() {
    let e = Engine::new_local();
    e.execute_text(1, "SET standalone:1=Loose").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 1 },
            filter: Some(MvccReadFilter::KeyPrefix("standalone:".to_string())),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceSummary {
                summary: MvccProvenanceSummary::ValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("standalone:1".to_string()),
            value: None,
        }]
    );
}

#[test]
fn execute_mvcc_query_supports_frame_aware_provenance_ordering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Aria").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 1,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceValueAsc {
                frame: MvccProvenanceFrame::TerminalInput,
            }),
            projection: MvccProjection::TargetKeyProvenanceValue {
                frame: MvccProvenanceFrame::TerminalInput,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("team:beta".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_provenance_frame_bundle_controls() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=member:1").unwrap();
    e.execute_text(6, "SET team:beta=member:2").unwrap();
    e.execute_text(7, "SET member:1=Alice").unwrap();
    e.execute_text(8, "SET member:2=Bob").unwrap();

    let ordered = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundleValuePathAsc {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordered.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("profile:1 -> team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("profile:2 -> team:beta".to_string()),
            },
        ]
    );

    let filtered = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundleValueEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected: "team:beta".to_string(),
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::ValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        filtered.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("profile:2 -> team:beta -> member:2".to_string()),
        }]
    );

    let key_ordered = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundleKeyPrefix {
                bundle: MvccProvenanceFrameBundle::FullPath,
                prefix: "profile:2".to_string(),
            }),
            order: Some(MvccReadOrder::ProvenanceBundleKeyPathAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        key_ordered.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("acct:1 -> profile:2 -> team:beta".to_string()),
        }]
    );

    let key_exact = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundleKeyEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected: "profile:2".to_string(),
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        key_exact.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("acct:1 -> profile:2".to_string()),
        }]
    );

    let key_value_exact = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundleKeyValueEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                key: "profile:2".to_string(),
                value: "team:beta".to_string(),
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        key_value_exact.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some(
                "acct:1=profile:2 -> profile:2=team:beta -> team:beta=member:2".to_string(),
            ),
        }]
    );

    let cross_frame_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundleKeyValueEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                key: "acct:1".to_string(),
                value: "team:beta".to_string(),
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert!(cross_frame_miss.rows.is_empty());

    let key_path_exact = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:1".to_string(), "profile:2".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        key_path_exact.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("acct:1 -> profile:2".to_string()),
        }]
    );

    let value_path_exact = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::ValuePath,
                expected: vec![
                    "profile:2".to_string(),
                    "team:beta".to_string(),
                    "member:2".to_string(),
                ],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::ValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        value_path_exact.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("profile:2 -> team:beta -> member:2".to_string()),
        }]
    );

    let ordered_path_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["profile:2".to_string(), "acct:1".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert!(ordered_path_miss.rows.is_empty());

    let key_path_contains = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathContains {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["profile:2".to_string(), "team:beta".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        key_path_contains.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:beta".to_string()),
            value: Some("acct:1 -> profile:2 -> team:beta".to_string()),
        }]
    );

    let truncated_bundle_contains_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathContains {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
                expected: vec!["team:beta".to_string(), "member:2".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert!(truncated_bundle_contains_miss.rows.is_empty());

    let ordered_subpath_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathContains {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyValuePath,
                expected: vec![
                    "profile:2=team:beta".to_string(),
                    "acct:1=profile:2".to_string(),
                ],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert!(ordered_subpath_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_quantified_and_positional_provenance_bundle_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:noop=profile:noop").unwrap();
    e.execute_text(4, "SET profile:noop=acct:noop").unwrap();

    let key_count = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:noop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleKeyCountAtLeast {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected: "acct:loop".to_string(),
                min_count: 2,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        key_count.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop".to_string()),
        }]
    );

    let value_count = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleValueCountAtLeast {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected: "profile:loop".to_string(),
                min_count: 2,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::ValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        value_count.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let key_value_count = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleKeyValueCountAtLeast {
                bundle: MvccProvenanceFrameBundle::FullPath,
                key: "acct:loop".to_string(),
                value: "profile:loop".to_string(),
                min_count: 2,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        key_value_count.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some(
                "acct:loop=profile:loop -> profile:loop=acct:loop -> acct:loop=profile:loop"
                    .to_string(),
            ),
        }]
    );

    let position_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSegmentEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
                index: 1,
                expected: "acct:loop".to_string(),
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        position_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("profile:loop -> acct:loop".to_string()),
        }]
    );

    let position_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSegmentEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::ValuePath,
                index: 2,
                expected: "profile:loop".to_string(),
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(position_miss.rows.is_empty());

    let threshold_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleKeyCountAtLeast {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected: "acct:loop".to_string(),
                min_count: 3,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(threshold_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_repeated_provenance_bundle_subpath_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let repeated_subpath = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathCountAtLeast {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                min_count: 2,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        repeated_subpath.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
        }]
    );

    let truncated_bundle_repeat_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathCountAtLeast {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                min_count: 2,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_bundle_repeat_miss.rows.is_empty());

    let impossible_repeat_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathCountAtLeast {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["profile:loop".to_string(), "acct:loop".to_string()],
                min_count: 2,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(impossible_repeat_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_relative_provenance_bundle_distance_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathPairAtDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left: "profile:loop".to_string(),
                right: "profile:loop".to_string(),
                distance: 2,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
        }]
    );

    let truncated_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathPairAtDistance {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left: "profile:loop".to_string(),
                right: "profile:loop".to_string(),
                distance: 2,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_distance_miss.rows.is_empty());

    let mismatch_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathPairAtDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left: "profile:loop".to_string(),
                right: "profile:loop".to_string(),
                distance: 1,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(mismatch_distance_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_provenance_bundle_suffix_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let suffix_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec![
                    "profile:loop".to_string(),
                    "acct:loop".to_string(),
                    "profile:loop".to_string(),
                ],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        suffix_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
        }]
    );

    let truncated_suffix_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec![
                    "profile:loop".to_string(),
                    "acct:loop".to_string(),
                    "profile:loop".to_string(),
                ],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_suffix_miss.rows.is_empty());

    let mismatch_suffix_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSuffixEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["profile:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(mismatch_suffix_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_provenance_bundle_prefix_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let prefix_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathPrefixEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        prefix_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
        }]
    );

    let truncated_prefix_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathPrefixEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        truncated_prefix_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop".to_string()),
        }]
    );

    let mismatch_prefix_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathPrefixEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["profile:loop".to_string(), "acct:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(mismatch_prefix_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_provenance_bundle_slice_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let anchored_slice_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSliceEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 1,
                expected: vec!["profile:loop".to_string(), "acct:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        anchored_slice_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string(),),
        }]
    );

    let truncated_slice_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSliceEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        truncated_slice_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop".to_string()),
        }]
    );

    let wrong_anchor_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSliceEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["profile:loop".to_string(), "acct:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_anchor_miss.rows.is_empty());

    let out_of_range_slice_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathSliceEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                start: 2,
                expected: vec!["acct:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(out_of_range_slice_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_whole_bundle_cardinality_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let full_path_len = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 3,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        full_path_len.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop".to_string()),
        }]
    );

    let truncated_bundle_len = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                expected_len: 2,
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        truncated_bundle_len.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop".to_string()),
        }]
    );

    let len_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundleLenEquals {
                bundle: MvccProvenanceFrameBundle::FullPath,
                expected_len: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(len_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_and_nth_occurrence_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:loop".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 3,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:solo".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let first_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 0,
                start_max: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_range_match.rows,
        first_occurrence_match.rows
    );

    let last_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("acct:loop".to_string()),
            value: Some("acct:loop -> profile:loop".to_string()),
        }]
    );

    let last_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 0,
                start_max: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(last_occurrence_range_match.rows, last_occurrence_match.rows);

    let nth_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        nth_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let wrong_first_occurrence_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_occurrence_miss.rows.is_empty());

    let wrong_first_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 1,
                start_max: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_occurrence_range_miss.rows.is_empty());

    let wrong_last_occurrence_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start: 0,
                expected: vec!["profile:loop".to_string(), "acct:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_last_occurrence_miss.rows.is_empty());

    let wrong_last_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 0,
                start_max: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_last_occurrence_range_miss.rows.is_empty());

    let inverted_first_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                start_min: 1,
                start_max: 0,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_first_occurrence_range_miss.rows.is_empty());

    let wrong_nth_occurrence_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 2,
                start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_nth_occurrence_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_range_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 2,
                start_max: 3,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        occurrence_range_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let truncated_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 0,
                start_max: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_occurrence_range_miss.rows.is_empty());

    let wrong_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 3,
                start_max: 3,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_occurrence_range_miss.rows.is_empty());

    let inverted_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                start_min: 3,
                start_max: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_range_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_distance_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let truncated_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_occurrence_distance_miss.rows.is_empty());

    let wrong_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                right_occurrence_index: 1,
                distance: 1,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_occurrence_distance_miss.rows.is_empty());

    let reversed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                right_occurrence_index: 0,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_occurrence_distance_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_distance_range_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 2,
                    max_distance: 3,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        occurrence_distance_range_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let truncated_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 2,
                    max_distance: 3,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_occurrence_distance_range_miss.rows.is_empty());

    let wrong_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 3,
                    max_distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_occurrence_distance_range_miss.rows.is_empty());

    let inverted_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    right_occurrence_index: 1,
                    min_distance: 3,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_occurrence_distance_range_miss.rows.is_empty());

    let reversed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    right_occurrence_index: 0,
                    min_distance: 2,
                    max_distance: 3,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_occurrence_distance_range_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_occurrence_distance_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:loop".to_string()),
            key: Some("profile:loop".to_string()),
            value: Some("acct:loop -> profile:loop -> acct:loop -> profile:loop".to_string()),
        }]
    );

    let first_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 2,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_occurrence_distance_range_match.rows,
        first_occurrence_distance_match.rows
    );

    let last_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_occurrence_distance_match.rows,
        first_occurrence_distance_match.rows
    );

    let last_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 2,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_occurrence_distance_range_match.rows,
        first_occurrence_distance_match.rows
    );

    let truncated_last_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                distance: 2,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_last_occurrence_distance_miss.rows.is_empty());

    let wrong_first_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 3,
                    max_distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_occurrence_distance_range_miss.rows.is_empty());

    let inverted_last_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    min_distance: 3,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_last_occurrence_distance_range_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_occurrence_distance_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_to_ordinal_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
            first_to_ordinal_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some(
                    "acct:loop -> profile:loop -> acct:loop -> profile:loop -> acct:loop -> profile:loop"
                        .to_string(),
                ),
            }]
        );

    let first_to_ordinal_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    min_distance: 4,
                    max_distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_to_ordinal_range_match.rows,
        first_to_ordinal_match.rows
    );

    let ordinal_to_last_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 1,
                    distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(ordinal_to_last_match.rows, first_to_ordinal_match.rows);

    let ordinal_to_last_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 1,
                    min_distance: 2,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_to_last_range_match.rows,
        first_to_ordinal_match.rows
    );

    let truncated_first_to_ordinal_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    distance: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_first_to_ordinal_miss.rows.is_empty());

    let wrong_first_to_ordinal_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    min_distance: 5,
                    max_distance: 6,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_to_ordinal_range_miss.rows.is_empty());

    let inverted_ordinal_to_last_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 1,
                    min_distance: 3,
                    max_distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 3,
                    distance: 2,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_occurrence_offset_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let first_to_ordinal_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start: 0,
                    occurrence_start: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
            first_to_ordinal_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some(
                    "acct:loop -> profile:loop -> acct:loop -> profile:loop -> acct:loop -> profile:loop"
                        .to_string(),
                ),
            }]
        );

    let first_to_ordinal_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start_min: 0,
                    first_start_max: 0,
                    occurrence_start_min: 4,
                    occurrence_start_max: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_to_ordinal_range_match.rows,
        first_to_ordinal_match.rows
    );

    let ordinal_to_last_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                occurrence_start: 2,
                last_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(ordinal_to_last_match.rows, first_to_ordinal_match.rows);

    let ordinal_to_last_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                occurrence_start_min: 2,
                occurrence_start_max: 2,
                last_start_min: 4,
                last_start_max: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_to_last_range_match.rows,
        first_to_ordinal_match.rows
    );

    let truncated_first_to_ordinal_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start: 0,
                    occurrence_start: 4,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_first_to_ordinal_miss.rows.is_empty());

    let wrong_first_to_ordinal_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    occurrence_index: 2,
                    first_start_min: 1,
                    first_start_max: 1,
                    occurrence_start_min: 5,
                    occurrence_start_max: 6,
                    expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_to_ordinal_range_miss.rows.is_empty());

    let inverted_ordinal_to_last_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 1,
                occurrence_start_min: 3,
                occurrence_start_max: 2,
                last_start_min: 5,
                last_start_max: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrenceToLastAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                occurrence_index: 3,
                occurrence_start: 2,
                last_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_offset_projection_and_ordering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:noop=profile:noop").unwrap();
    e.execute_text(4, "SET profile:noop=acct:noop").unwrap();

    let source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:noop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
        ],
    };

    let first_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("0".to_string()),
            },
        ]
    );

    let last_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("4".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let ordinal_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetDesc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Nth(1),
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                occurrence: MvccProvenanceOccurrence::Nth(1),
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let missing_offsets = e
        .execute_mvcc_query(&MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceOffsetAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["missing:key".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceOffset {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                expected: vec!["missing:key".to_string()],
                occurrence: MvccProvenanceOccurrence::First,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        missing_offsets.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_bundle_occurrence_distance_projection_and_ordering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:noop=profile:noop").unwrap();
    e.execute_text(4, "SET profile:noop=acct:noop").unwrap();

    let source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:noop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
        ],
    };

    let same_subpath_distances = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        same_subpath_distances.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("4".to_string()),
            },
        ]
    );

    let mixed_distances = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceDesc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        mixed_distances.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("5".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let missing_distances = e
        .execute_mvcc_query(&MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(MvccReadOrder::ProvenanceBundlePathOccurrenceDistanceAsc {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["missing:left".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["missing:right".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            }),
            projection: MvccProjection::TargetKeyProvenanceBundleOccurrenceDistance {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["missing:left".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["missing:right".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        missing_distances.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_bundle_mixed_occurrence_offset_pair_projection_and_ordering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:noop=profile:noop").unwrap();
    e.execute_text(4, "SET profile:noop=acct:noop").unwrap();

    let source = MvccReadSource::Concat {
        sources: vec![
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "profile:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            MvccReadSource::FollowValueChain {
                keys: vec!["acct:noop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
        ],
    };

    let ascending_pairs = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(
                MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:loop".to_string()],
                    left_occurrence: MvccProvenanceOccurrence::First,
                    right_expected: vec!["profile:loop".to_string()],
                    right_occurrence: MvccProvenanceOccurrence::Last,
                },
            ),
            projection: MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ascending_pairs.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("0,5".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("1,4".to_string()),
            },
        ]
    );

    let descending_pairs = e
        .execute_mvcc_query(&MvccReadQuery {
            source: source.clone(),
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(
                MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairDesc {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:loop".to_string()],
                    left_occurrence: MvccProvenanceOccurrence::First,
                    right_expected: vec!["profile:loop".to_string()],
                    right_occurrence: MvccProvenanceOccurrence::Last,
                },
            ),
            projection: MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:loop".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["profile:loop".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        descending_pairs.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: Some("1,4".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some("0,5".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );

    let missing_pairs = e
        .execute_mvcc_query(&MvccReadQuery {
            source,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: None,
            order: Some(
                MvccReadOrder::ProvenanceBundlePathMixedOccurrenceOffsetPairAsc {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["missing:left".to_string()],
                    left_occurrence: MvccProvenanceOccurrence::First,
                    right_expected: vec!["missing:right".to_string()],
                    right_occurrence: MvccProvenanceOccurrence::Last,
                },
            ),
            projection: MvccProjection::TargetKeyProvenanceBundleMixedOccurrenceOffsetPair {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["missing:left".to_string()],
                left_occurrence: MvccProvenanceOccurrence::First,
                right_expected: vec!["missing:right".to_string()],
                right_occurrence: MvccProvenanceOccurrence::Last,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        missing_pairs.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:loop".to_string()),
                key: Some("acct:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:noop".to_string()),
                key: Some("profile:noop".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_bundle_ordinal_pair_occurrence_offset_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:loop=profile:loop").unwrap();
    e.execute_text(2, "SET profile:loop=acct:loop").unwrap();
    e.execute_text(3, "SET acct:solo=profile:solo").unwrap();
    e.execute_text(4, "SET profile:solo=team:solo").unwrap();

    let ordinal_pair_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string(), "acct:solo".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start: 2,
                right_occurrence_index: 2,
                right_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
            ordinal_pair_match.rows,
            vec![MvccReadRow {
                source_key: Some("acct:loop".to_string()),
                key: Some("profile:loop".to_string()),
                value: Some(
                    "acct:loop -> profile:loop -> acct:loop -> profile:loop -> acct:loop -> profile:loop"
                        .to_string(),
                ),
            }]
        );

    let ordinal_pair_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start_min: 2,
                left_start_max: 2,
                right_occurrence_index: 2,
                right_start_min: 4,
                right_start_max: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(ordinal_pair_range_match.rows, ordinal_pair_match.rows);

    let truncated_ordinal_pair_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start: 2,
                right_occurrence_index: 2,
                right_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_ordinal_pair_miss.rows.is_empty());

    let wrong_ordinal_pair_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start_min: 3,
                left_start_max: 3,
                right_occurrence_index: 2,
                right_start_min: 5,
                right_start_max: 6,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_pair_range_miss.rows.is_empty());

    let inverted_ordinal_pair_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start_min: 3,
                left_start_max: 2,
                right_occurrence_index: 2,
                right_start_min: 5,
                right_start_max: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_pair_range_miss.rows.is_empty());

    let wrong_ordinal_pair_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:loop".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathOccurrencePairAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_start: 2,
                right_occurrence_index: 3,
                right_start: 4,
                expected: vec!["acct:loop".to_string(), "profile:loop".to_string()],
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_pair_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_mixed_occurrence_distance_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();
    e.execute_text(4, "SET acct:2=profile:2").unwrap();
    e.execute_text(5, "SET profile:2=team:beta").unwrap();

    let mixed_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        mixed_occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("acct:1".to_string()),
            value: Some("acct:1 -> profile:1 -> team:alpha -> acct:1".to_string()),
        }]
    );

    let mixed_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 1,
                    max_distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        mixed_occurrence_distance_range_match.rows,
        mixed_occurrence_distance_match.rows
    );

    let truncated_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_mixed_occurrence_distance_miss.rows.is_empty());

    let truncated_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 1,
                    max_distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());

    let wrong_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_mixed_occurrence_distance_miss.rows.is_empty());

    let wrong_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 0,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_mixed_occurrence_distance_range_miss.rows.is_empty());

    let inverted_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    min_distance: 3,
                    max_distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());

    let reversed_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_mixed_occurrence_distance_miss.rows.is_empty());

    let reversed_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 0,
                    left_expected: vec!["team:alpha".to_string(), "acct:1".to_string()],
                    right_occurrence_index: 0,
                    right_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    min_distance: 1,
                    max_distance: 2,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_mixed_occurrence_distance_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_mixed_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_mixed_occurrence_distance_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_mixed_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 1,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_mixed_occurrence_distance_range_match.rows,
        first_mixed_occurrence_distance_match.rows
    );

    let last_mixed_occurrence_distance_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_mixed_occurrence_distance_match.rows,
        first_mixed_occurrence_distance_match.rows
    );

    let last_mixed_occurrence_distance_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 1,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_mixed_occurrence_distance_range_match.rows,
        first_mixed_occurrence_distance_match.rows
    );

    let truncated_last_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_last_mixed_occurrence_distance_miss
        .rows
        .is_empty());

    let wrong_first_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 2,
                    max_distance: 3,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());

    let reversed_last_mixed_occurrence_distance_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(reversed_last_mixed_occurrence_distance_miss.rows.is_empty());

    let inverted_last_mixed_occurrence_distance_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 2,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_last_mixed_occurrence_distance_range_miss
        .rows
        .is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_mixed_occurrence_distance_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_to_ordinal_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 4,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_to_ordinal_mixed_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_to_ordinal_mixed_range_match = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 5,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: Some(
                    MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
                        bundle: MvccProvenanceFrameBundle::FullPath,
                        summary: MvccProvenanceSummary::KeyPath,
                        left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                        right_occurrence_index: 1,
                        right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                        min_distance: 4,
                        max_distance: 4,
                    },
                ),
                order: None,
                projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                },
                limit: None,
            })
            .unwrap();

    assert_eq!(
        first_to_ordinal_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_to_last_mixed_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 1,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_to_last_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let truncated_first_to_ordinal_mixed_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistance {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_first_to_ordinal_mixed_miss.rows.is_empty());

    let wrong_first_to_ordinal_mixed_range_miss = e
            .execute_mvcc_query(&MvccReadQuery {
                source: MvccReadSource::FollowValueChain {
                    keys: vec!["acct:1".to_string()],
                    plan: MvccValueChainPlan {
                        value_key_hops: 5,
                        terminal: MvccValueChainTerminal::CurrentRow,
                    },
                    provenance: MvccSourceProvenance::Seed,
                },
                visibility: StorageVisibility { read_txn_id: 3 },
                filter: Some(
                    MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceDistanceWithin {
                        bundle: MvccProvenanceFrameBundle::FullPath,
                        summary: MvccProvenanceSummary::KeyPath,
                        left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                        right_occurrence_index: 1,
                        right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                        min_distance: 5,
                        max_distance: 6,
                    },
                ),
                order: None,
                projection: MvccProjection::KeyOnly,
                limit: None,
            })
            .unwrap();

    assert!(wrong_first_to_ordinal_mixed_range_miss.rows.is_empty());

    let inverted_ordinal_to_last_mixed_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistanceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    min_distance: 2,
                    max_distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_mixed_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_mixed_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedDistance {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 2,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    distance: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_mixed_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_to_ordinal_mixed_occurrence_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_to_ordinal_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 0,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_to_ordinal_mixed_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_to_ordinal_mixed_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 0,
                    left_start_max: 0,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_to_ordinal_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_to_last_mixed_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let ordinal_to_last_mixed_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 3,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_to_last_mixed_range_match.rows,
        first_to_ordinal_mixed_match.rows
    );

    let truncated_first_to_ordinal_mixed_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceAt {
                    bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 0,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_first_to_ordinal_mixed_miss.rows.is_empty());

    let wrong_first_to_ordinal_mixed_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceToOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 1,
                    left_start_max: 1,
                    right_occurrence_index: 1,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 5,
                    right_start_max: 6,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_to_ordinal_mixed_range_miss.rows.is_empty());

    let inverted_ordinal_to_last_mixed_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 1,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 4,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 5,
                    right_start_max: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_to_last_mixed_range_miss.rows.is_empty());

    let wrong_ordinal_to_last_mixed_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathOccurrenceToLastMixedAt {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_occurrence_index: 2,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_to_last_mixed_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_ordinal_mixed_occurrence_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let ordinal_mixed_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_mixed_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let ordinal_mixed_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start_min: 3,
                left_start_max: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start_min: 4,
                right_start_max: 4,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ordinal_mixed_occurrence_range_match.rows,
        ordinal_mixed_occurrence_match.rows
    );

    let truncated_ordinal_mixed_occurrence_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_ordinal_mixed_occurrence_miss.rows.is_empty());

    let wrong_ordinal_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start_min: 0,
                left_start_max: 2,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start_min: 4,
                right_start_max: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_mixed_occurrence_range_miss.rows.is_empty());

    let inverted_ordinal_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceWithin {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 1,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start_min: 4,
                left_start_max: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start_min: 4,
                right_start_max: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_ordinal_mixed_occurrence_range_miss.rows.is_empty());

    let wrong_ordinal_mixed_occurrence_index_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_occurrence_index: 0,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_occurrence_index: 1,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_ordinal_mixed_occurrence_index_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_bundle_first_last_mixed_occurrence_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha=acct:1").unwrap();

    let first_mixed_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 0,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 1,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_mixed_occurrence_match.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha".to_string()),
            value: Some(
                "acct:1 -> profile:1 -> team:alpha -> acct:1 -> profile:1 -> team:alpha"
                    .to_string(),
            ),
        }]
    );

    let first_mixed_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 0,
                    left_start_max: 0,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 1,
                    right_start_max: 1,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_mixed_occurrence_range_match.rows,
        first_mixed_occurrence_match.rows
    );

    let last_mixed_occurrence_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_mixed_occurrence_match.rows,
        first_mixed_occurrence_match.rows
    );

    let last_mixed_occurrence_range_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 3,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
                },
            ),
            order: None,
            projection: MvccProjection::TargetKeyProvenanceBundleSummary {
                bundle: MvccProvenanceFrameBundle::FullPath,
                summary: MvccProvenanceSummary::KeyPath,
            },
            limit: None,
        })
        .unwrap();

    assert_eq!(
        last_mixed_occurrence_range_match.rows,
        first_mixed_occurrence_match.rows
    );

    let truncated_last_mixed_occurrence_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 2,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceAt {
                bundle: MvccProvenanceFrameBundle::SeedThroughTerminalInput,
                summary: MvccProvenanceSummary::KeyPath,
                left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                left_start: 3,
                right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                right_start: 4,
            }),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(truncated_last_mixed_occurrence_miss.rows.is_empty());

    let wrong_first_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathFirstMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 1,
                    left_start_max: 2,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 1,
                    right_start_max: 1,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(wrong_first_mixed_occurrence_range_miss.rows.is_empty());

    let inverted_last_mixed_occurrence_range_miss = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["acct:1".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(
                MvccReadFilter::ProvenanceBundlePathLastMixedOccurrenceWithin {
                    bundle: MvccProvenanceFrameBundle::FullPath,
                    summary: MvccProvenanceSummary::KeyPath,
                    left_expected: vec!["acct:1".to_string(), "profile:1".to_string()],
                    left_start_min: 4,
                    left_start_max: 3,
                    right_expected: vec!["profile:1".to_string(), "team:alpha".to_string()],
                    right_start_min: 4,
                    right_start_max: 4,
                },
            ),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert!(inverted_last_mixed_occurrence_range_miss.rows.is_empty());
}

#[test]
fn execute_mvcc_query_sorts_missing_provenance_frames_deterministically() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET profile:1=team:alpha").unwrap();
    e.execute_text(3, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(4, "SET standalone:1=Loose").unwrap();
    e.execute_text(5, "SET standalone:2=Leaf").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::Concat {
                sources: vec![
                    MvccReadSource::FullScan,
                    MvccReadSource::FollowValueChain {
                        keys: vec!["acct:1".to_string()],
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 5 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::KeyPrefix("standalone:".to_string()),
                MvccReadFilter::KeyPrefix("team:alpha:".to_string()),
            ])),
            order: Some(MvccReadOrder::ProvenanceKeyDesc {
                frame: MvccProvenanceFrame::TerminalInput,
            }),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("standalone:1".to_string()),
                value: Some("Loose".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("standalone:2".to_string()),
                value: Some("Leaf".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("team:alpha:1".to_string()),
                value: Some("Alice".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_labeled_branch_projection_and_ordering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Alpha Team").unwrap();
    e.execute_text(6, "SET team:beta=Beta Team").unwrap();
    e.execute_text(7, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(8, "SET team:beta:1=Bob").unwrap();

    let query = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                branches: vec![
                    MvccLabeledValueChainBranch {
                        label: "members".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                    },
                    MvccLabeledValueChainBranch {
                        label: "team".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::AllBranches,
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::BranchLabelAsc),
            projection: MvccProjection::BranchLabelTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        query.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alpha Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("members".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Beta Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("members".to_string()),
                value: Some("Bob".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team".to_string()),
                value: Some("Alpha Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team".to_string()),
                value: Some("Beta Team".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_preserves_labeled_branch_identity_and_first_match_filtering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha=Shared Team").unwrap();
    e.execute_text(6, "SET team:beta=Shared Team").unwrap();

    let distinct = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::ConcatDistinct {
                sources: vec![
                    MvccReadSource::FollowValueChainLabeledBranches {
                        keys: vec!["acct:1".to_string()],
                        branches: vec![MvccLabeledValueChainBranch {
                            label: "primary".to_string(),
                            plan: MvccValueChainPlan {
                                value_key_hops: 2,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                        }],
                        fan_in: MvccValueChainBranchFanIn::AllBranches,
                        provenance: MvccSourceProvenance::Seed,
                    },
                    MvccReadSource::FollowValueChainLabeledBranches {
                        keys: vec!["acct:1".to_string()],
                        branches: vec![MvccLabeledValueChainBranch {
                            label: "fallback".to_string(),
                            plan: MvccValueChainPlan {
                                value_key_hops: 2,
                                terminal: MvccValueChainTerminal::CurrentRow,
                            },
                        }],
                        fan_in: MvccValueChainBranchFanIn::AllBranches,
                        provenance: MvccSourceProvenance::Seed,
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: None,
            order: Some(MvccReadOrder::BranchLabelAsc),
            projection: MvccProjection::BranchLabelTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        distinct.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("fallback".to_string()),
                value: Some("Shared Team".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("primary".to_string()),
                value: Some("Shared Team".to_string()),
            },
        ]
    );

    let first_match = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChainLabeledBranches {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
                branches: vec![
                    MvccLabeledValueChainBranch {
                        label: "team".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 2,
                            terminal: MvccValueChainTerminal::CurrentRow,
                        },
                    },
                    MvccLabeledValueChainBranch {
                        label: "members".to_string(),
                        plan: MvccValueChainPlan {
                            value_key_hops: 1,
                            terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                        },
                    },
                ],
                fan_in: MvccValueChainBranchFanIn::FirstNonEmptyBranch,
                provenance: MvccSourceProvenance::TerminalInput,
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::BranchLabelEquals("team".to_string())),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        first_match.rows,
        vec![
            MvccReadRow {
                source_key: Some("profile:1".to_string()),
                key: Some("team:alpha".to_string()),
                value: Some("team:alpha".to_string()),
            },
            MvccReadRow {
                source_key: Some("profile:2".to_string()),
                key: Some("team:beta".to_string()),
                value: Some("team:beta".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_symmetric_difference_all_source_composition() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET user:1=active").unwrap();
    e.execute_text(9, "SET user:2=locked").unwrap();

    let exact_imbalance = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceAll {
                sources: vec![
                    MvccReadSource::KeyBatchLookup {
                        keys: vec![
                            "user:1".to_string(),
                            "user:1".to_string(),
                            "user:2".to_string(),
                        ],
                    },
                    MvccReadSource::KeyBatchLookup {
                        keys: vec!["user:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        exact_imbalance.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:2".to_string()),
                value: None,
            },
        ]
    );

    let join_imbalance = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::SymmetricDifferenceAll {
                sources: vec![
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string(), "acct:1".to_string()],
                    },
                    MvccReadSource::FollowValueKeyRefPrefixes {
                        keys: vec!["acct:1".to_string()],
                    },
                ],
            },
            visibility: StorageVisibility { read_txn_id: 9 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("Alice".to_string()),
                MvccReadFilter::ValueEquals("Ally".to_string()),
            ])),
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        join_imbalance.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_refs_join_adjacent_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:2").unwrap();
    e.execute_text(2, "SET acct:2=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=active").unwrap();
    e.execute_text(4, "SET profile:2=suspended").unwrap();
    e.execute_text(5, "SET acct:3=missing").unwrap();
    e.execute_text(6, "SET acct:1=profile:3").unwrap();
    e.execute_text(7, "SET profile:3=closed").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefs {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:1".to_string()),
                value: Some("active".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:3".to_string()),
                value: Some("closed".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefs {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 7 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("closed".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("profile:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("profile:3".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_prefixes_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=order:1:").unwrap();
    e.execute_text(2, "SET acct:2=order:2:").unwrap();
    e.execute_text(3, "SET order:1:a=paid").unwrap();
    e.execute_text(4, "SET order:1:b=packed").unwrap();
    e.execute_text(5, "SET order:2:a=queued").unwrap();
    e.execute_text(6, "SET order:3:a=orphan").unwrap();
    e.execute_text(7, "SET acct:3=missing:").unwrap();
    e.execute_text(8, "SET acct:1=order:1b:").unwrap();
    e.execute_text(9, "SET order:1b:a=shipped").unwrap();
    e.execute_text(10, "SET order:1b:b=delivered").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("queued".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: Some("shipped".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: Some("delivered".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 10 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("delivered".to_string()),
                MvccReadFilter::ValueEquals("queued".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_prefixes_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET profile:1=order:1:").unwrap();
    e.execute_text(5, "SET profile:2=order:2:").unwrap();
    e.execute_text(6, "SET order:1:a=paid").unwrap();
    e.execute_text(7, "SET order:1:b=packed").unwrap();
    e.execute_text(8, "SET order:2:a=queued").unwrap();
    e.execute_text(9, "SET order:2:b=delivered").unwrap();
    e.execute_text(10, "SET order:3:a=orphan").unwrap();
    e.execute_text(11, "SET profile:1=order:1b:").unwrap();
    e.execute_text(12, "SET order:1b:a=shipped").unwrap();
    e.execute_text(13, "SET order:1b:b=cancelled").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 13 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("queued".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:b".to_string()),
                value: Some("delivered".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: Some("shipped".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:b".to_string()),
                value: Some("cancelled".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 13 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("queued".to_string()),
                MvccReadFilter::ValueEquals("shipped".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:1b:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_refs_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET acct:4=profile:4").unwrap();
    e.execute_text(5, "SET profile:1=team:1").unwrap();
    e.execute_text(6, "SET profile:2=team:2").unwrap();
    e.execute_text(7, "SET profile:4=missing-team").unwrap();
    e.execute_text(8, "SET team:1=gold").unwrap();
    e.execute_text(9, "SET team:2=silver").unwrap();
    e.execute_text(10, "SET team:3=bronze").unwrap();
    e.execute_text(11, "SET profile:1=team:3").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                    "acct:4".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:2".to_string()),
                value: Some("silver".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:3".to_string()),
                value: Some("bronze".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyRefs {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 11 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("silver".to_string()),
                MvccReadFilter::ValueEquals("bronze".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:3".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_follow_value_key_ref_value_key_prefixes_source() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET acct:3=missing-profile").unwrap();
    e.execute_text(4, "SET acct:4=profile:4").unwrap();
    e.execute_text(5, "SET profile:1=team:1").unwrap();
    e.execute_text(6, "SET profile:2=team:2").unwrap();
    e.execute_text(7, "SET profile:4=missing-team").unwrap();
    e.execute_text(8, "SET team:1=order:1:").unwrap();
    e.execute_text(9, "SET team:2=order:2:").unwrap();
    e.execute_text(10, "SET order:1:a=paid").unwrap();
    e.execute_text(11, "SET order:1:b=packed").unwrap();
    e.execute_text(12, "SET order:2:a=queued").unwrap();
    e.execute_text(13, "SET order:2:b=shipped").unwrap();
    e.execute_text(14, "SET order:3:a=orphan").unwrap();
    e.execute_text(15, "SET profile:1=team:3").unwrap();
    e.execute_text(16, "SET team:3=order:3:").unwrap();

    let request_order = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:2".to_string(),
                    "acct:1".to_string(),
                    "missing".to_string(),
                    "acct:3".to_string(),
                    "acct:4".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: None,
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        request_order.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("queued".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:b".to_string()),
                value: Some("shipped".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:3:a".to_string()),
                value: Some("orphan".to_string()),
            },
        ]
    );

    let filtered_and_sorted = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyPrefixes {
                keys: vec![
                    "acct:1".to_string(),
                    "acct:2".to_string(),
                    "acct:2".to_string(),
                ],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::ValueEquals("orphan".to_string()),
                MvccReadFilter::ValueEquals("queued".to_string()),
            ])),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        filtered_and_sorted.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:3:a".to_string()),
                value: None,
            },
        ]
    );

    let join_side_projection = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefValueKeyPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 16 },
            filter: None,
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        join_side_projection.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:a".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("order:2:b".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("order:3:a".to_string()),
                value: Some("profile:1".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_join_side_projection_keeps_non_join_shapes_stable() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: None,
            projection: MvccProjection::TargetKeySourceValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_join_side_source_filters() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=profile:1").unwrap();
    e.execute_text(2, "SET acct:2=profile:2").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(7, "SET team:beta:1=Bob").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::SourceKeyPrefix("acct:1".to_string()),
                MvccReadFilter::SourceValueEquals("profile:1".to_string()),
                MvccReadFilter::KeyPrefix("team:alpha".to_string()),
            ])),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(1),
        })
        .unwrap();

    assert_eq!(
        result.rows,
        vec![MvccReadRow {
            source_key: Some("acct:1".to_string()),
            key: Some("team:alpha:2".to_string()),
            value: Some("profile:1".to_string()),
        }]
    );
}

#[test]
fn execute_mvcc_query_source_filters_are_empty_for_non_join_shapes() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: Some(MvccReadFilter::SourceValueEquals("open".to_string())),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert!(result.rows.is_empty());
}

#[test]
fn execute_mvcc_query_supports_join_side_source_ordering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:2=profile:2").unwrap();
    e.execute_text(2, "SET acct:1=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();
    e.execute_text(7, "SET team:alpha:2=Ally").unwrap();
    e.execute_text(8, "SET team:beta:2=Bianca").unwrap();

    let source_key_ordered = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        source_key_ordered.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:1".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("team:alpha:2".to_string()),
                value: Some("profile:1".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );

    let source_value_ordered = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 8 },
            filter: None,
            order: Some(MvccReadOrder::SourceValueDesc),
            projection: MvccProjection::TargetKeySourceValue,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        source_value_ordered.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:1".to_string()),
                value: Some("profile:2".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("team:beta:2".to_string()),
                value: Some("profile:2".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_source_ordering_keeps_non_join_shapes_stable() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:2=locked").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();

    let result = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyBatchLookup {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 2 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyDesc),
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        result.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_mixed_join_side_projection_controls() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:2=profile:2").unwrap();
    e.execute_text(2, "SET acct:1=profile:1").unwrap();
    e.execute_text(3, "SET profile:1=team:alpha").unwrap();
    e.execute_text(4, "SET profile:2=team:beta").unwrap();
    e.execute_text(5, "SET team:alpha:1=Alice").unwrap();
    e.execute_text(6, "SET team:beta:1=Bob").unwrap();

    let source_key_target_value = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:2".to_string(), "acct:1".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: None,
            order: Some(MvccReadOrder::SourceKeyAsc),
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_key_target_value.rows,
        vec![
            MvccReadRow {
                source_key: Some("acct:1".to_string()),
                key: Some("acct:1".to_string()),
                value: Some("Alice".to_string()),
            },
            MvccReadRow {
                source_key: Some("acct:2".to_string()),
                key: Some("acct:2".to_string()),
                value: Some("Bob".to_string()),
            },
        ]
    );

    let source_value_only = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueKeyRefPrefixes {
                keys: vec!["acct:1".to_string(), "acct:2".to_string()],
            },
            visibility: StorageVisibility { read_txn_id: 6 },
            filter: Some(MvccReadFilter::SourceKeyPrefix("acct:2".to_string())),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::SourceValueOnly,
            limit: Some(1),
        })
        .unwrap();

    assert_eq!(
        source_value_only.rows,
        vec![MvccReadRow {
            source_key: Some("acct:2".to_string()),
            key: None,
            value: Some("profile:2".to_string()),
        }]
    );
}

#[test]
fn execute_mvcc_query_mixed_join_projection_keeps_non_join_shapes_stable() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();

    let source_key_target_value = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 1 },
            filter: None,
            order: None,
            projection: MvccProjection::SourceKeyTargetValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_key_target_value.rows,
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: Some("open".to_string()),
        }]
    );

    let source_value_only = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::KeyLookup {
                key: "acct:1".to_string(),
            },
            visibility: StorageVisibility { read_txn_id: 1 },
            filter: None,
            order: None,
            projection: MvccProjection::SourceValueOnly,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        source_value_only.rows,
        vec![MvccReadRow {
            source_key: None,
            key: None,
            value: None,
        }]
    );
}

#[test]
fn execute_mvcc_query_supports_composite_filter_shapes() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET user:1=active").unwrap();
    e.execute_text(4, "SET user:2=locked").unwrap();

    let all_filter = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::All(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::ValueEquals("locked".to_string()),
            ])),
            order: None,
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();
    assert_eq!(
        all_filter.rows,
        vec![MvccReadRow {
            source_key: None,
            key: Some("acct:2".to_string()),
            value: Some("locked".to_string()),
        }]
    );

    let any_filter = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::Any(vec![
                MvccReadFilter::KeyPrefix("acct:".to_string()),
                MvccReadFilter::ValueEquals("active".to_string()),
            ])),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: None,
        })
        .unwrap();
    assert_eq!(
        any_filter.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("user:1".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_key_range_filter_shapes() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();
    e.execute_text(4, "SET acct:4=suspended").unwrap();

    let ranged = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyRange {
                start_inclusive: "acct:2".to_string(),
                end_exclusive: "acct:4".to_string(),
            }),
            order: Some(MvccReadOrder::KeyAsc),
            projection: MvccProjection::KeyValue,
            limit: None,
        })
        .unwrap();

    assert_eq!(
        ranged.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("locked".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_limit_after_filtering() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:2=locked").unwrap();
    e.execute_text(3, "SET acct:3=locked").unwrap();

    let limited = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: None,
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        limited.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_key_ordering_before_limit() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:2=locked").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:3=closed").unwrap();

    let descending = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 3 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::KeyDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        descending.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn execute_mvcc_query_supports_value_ordering_before_limit() {
    let e = Engine::new_local();
    e.execute_text(1, "SET acct:2=locked").unwrap();
    e.execute_text(2, "SET acct:1=open").unwrap();
    e.execute_text(3, "SET acct:4=closed").unwrap();
    e.execute_text(4, "SET acct:3=closed").unwrap();

    let ascending = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueAsc),
            projection: MvccProjection::KeyValue,
            limit: Some(3),
        })
        .unwrap();

    assert_eq!(
        ascending.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:3".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:4".to_string()),
                value: Some("closed".to_string()),
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: Some("locked".to_string()),
            },
        ]
    );

    let descending = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FullScan,
            visibility: StorageVisibility { read_txn_id: 4 },
            filter: Some(MvccReadFilter::KeyPrefix("acct:".to_string())),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::KeyOnly,
            limit: Some(2),
        })
        .unwrap();

    assert_eq!(
        descending.rows,
        vec![
            MvccReadRow {
                source_key: None,
                key: Some("acct:1".to_string()),
                value: None,
            },
            MvccReadRow {
                source_key: None,
                key: Some("acct:2".to_string()),
                value: None,
            },
        ]
    );
}

#[test]
fn publish_telemetry_emits_snapshot_to_sink() {
    let e = Engine::new_local();
    e.execute_text(1, "SET a=1").unwrap();

    let mut sink = InMemoryTelemetrySink::default();
    e.publish_telemetry(&mut sink);

    assert_eq!(sink.snapshots().len(), 1);
    let snapshot = &sink.snapshots()[0];
    assert_eq!(snapshot.role, Role::Leader);
    assert_eq!(snapshot.replication_lag.commit_index, 1);
    assert_eq!(snapshot.replication_lag.applied_index, 1);
    assert_eq!(snapshot.replication_lag.visible_index, 1);
    assert_eq!(snapshot.replication_lag.commit_apply_gap, 0);
    assert_eq!(snapshot.replication_lag.apply_visible_gap, 0);
    assert_eq!(snapshot.runtime_metrics.commits_total, 1);
    assert_eq!(snapshot.snapshot_id, 0);
    assert_eq!(snapshot.wal_flushed_count, 1);
    assert_eq!(snapshot.wal_last_durable_txn_id, Some(1));
    assert_eq!(snapshot.wal_buffered_count, 1);
    assert_eq!(snapshot.wal_unflushed_count, 0);
    assert_eq!(snapshot.pending_batch_len, 0);
    assert_eq!(snapshot.active_txn_count, 0);
    assert_eq!(snapshot.backlog_blocker_count, 0);
    assert!(!snapshot.has_backlog_blockers());
    assert!(snapshot.quiescent_for_failover);
    assert!(snapshot.gpu_parity_fallbacks.is_empty());
}

#[test]
fn installing_older_snapshot_is_a_status_no_op() {
    let mut e = Engine::new_local();
    let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();
    let baseline = e.status_snapshot();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index.saturating_sub(1),
        last_included_term: 1,
        snapshot_id: 99,
    });

    let marks = e.replication_watermarks();
    assert_eq!(marks.commit_index, committed.index);
    assert_eq!(marks.applied_index, committed.index);
    assert_eq!(marks.visible_index, committed.index);
    assert_eq!(marks.snapshot_id, baseline.snapshot.snapshot_id);
    assert_eq!(e.status_snapshot(), baseline);
    assert_eq!(e.visible_up_to(), committed.index);
    assert_eq!(e.get("a").as_deref(), Some("1"));
}

#[test]
fn installing_higher_index_lower_term_snapshot_is_a_status_no_op() {
    let mut e = Engine::new_local();
    let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index + 2,
        last_included_term: 3,
        snapshot_id: 11,
    });
    let baseline = e.status_snapshot();
    let baseline_marks = e.replication_watermarks();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index + 3,
        last_included_term: 2,
        snapshot_id: 99,
    });

    let marks = e.replication_watermarks();
    assert_eq!(marks, baseline_marks);
    assert_eq!(e.status_snapshot(), baseline);
    assert_eq!(e.visible_up_to(), baseline.snapshot.visible_index);
    assert_eq!(e.get("a").as_deref(), Some("1"));
}

#[test]
fn installing_advanced_snapshot_replaces_snapshot_identity_exactly() {
    let mut e = Engine::new_local();
    let committed = e.commit_mutation(1, b"SET a=1".to_vec()).unwrap();

    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index,
        last_included_term: 1,
        snapshot_id: 11,
    });
    e.install_snapshot(SnapshotMeta {
        last_included_index: committed.index + 2,
        last_included_term: 2,
        snapshot_id: 4,
    });

    let marks = e.replication_watermarks();
    let status = e.status_snapshot();
    assert_eq!(marks.snapshot_id, 4);
    assert_eq!(status.snapshot.snapshot_id, 4);
    assert_eq!(status.snapshot.last_included_index, committed.index + 2);
    assert_eq!(status.snapshot.last_included_term, 2);
}

