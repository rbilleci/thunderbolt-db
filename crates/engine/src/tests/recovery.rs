use super::*;

/// A SERIAL-durable engine, PINNED to the serial backend regardless of the process
/// `GPU_DB_WAL_DURABILITY` env. The tests that use it exercise serial-backend on-disk mechanics —
/// torn-record CRC rejection, torn-append truncation, and checkpoint / segment ROTATION (prefix
/// truncation) — which the FUA fence-pool backend either stores in a DIFFERENT frame-log format
/// (so raw serial-byte corruption is meaningless against it) or DEFERS to a later step (checkpoint
/// prefix truncation is unsupported by the FUA backend today). Pinning serial keeps these tests
/// meaningful (and green) under BOTH modes; in the default (serial) mode this is byte-identical to
/// `Engine::with_durable_wal_segment`. The FUA backend has its own crash-safety coverage in the
/// `gpu_db_wal` crate (roundtrip / segment-roll / reopen-across-lives).
fn serial_durable_engine(path: impl AsRef<std::path::Path>) -> Engine {
    let mut engine = Engine::new_local_cpu_oracle();
    engine.commit_state_mut().wal = WalBuffer::with_durable_segment(path.as_ref());
    engine
}

fn canonical_test_lane_record(
    base: &std::path::Path,
    lane_id: u32,
    local_seq: u64,
    txn_id: u64,
    payload: Vec<u8>,
) -> gpu_db_wal::WalRecord {
    let serial = gpu_db_wal::read_wal_segment(base).expect("serial canonical prefix");
    let boundary = serial
        .iter()
        .rev()
        .find_map(|record| {
            gpu_db_wal::decode_canonical_record_payload(&record.payload)
                .expect("canonical prefix decode")
                .map(|envelope| {
                    (
                        envelope.header.identity,
                        envelope.header.catalog_after_epoch,
                        envelope.header.catalog_after_digest,
                    )
                })
        })
        .expect("serial prefix carries durable identity");
    let commit_seq = serial.len() as u64 + 1 + local_seq;
    let request_digest = gpu_db_wal::canonical_request_digest(&payload);
    Engine::canonical_wal_record_with_boundary_and_request_digest(
        boundary.0,
        boundary.1,
        boundary.2,
        txn_id,
        commit_seq,
        lane_id + 1,
        &std::sync::Arc::from(payload),
        request_digest,
    )
    .expect("canonical test lane record")
}

#[test]
fn relational_access_path_recovers_from_durable_wal_file_after_restart() {
    let path = test_wal_path("restart");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();

    e.persist_durable_wal_to_file(&path).unwrap();
    let recovered = Engine::recover_from_durable_wal_file(&path).unwrap();
    let _ = std::fs::remove_file(path);

    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);
    assert_eq!(table.columns[0].attnum, 1);
    assert_eq!(recovered.wal_unflushed_count(), 0);
    assert_eq!(recovered.wal_flushed_count(), 2);

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace' ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = recovered.execute_relational_select(&select).unwrap();

    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("name".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 1,
        },
    );
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(3)]]);
}

#[test]
fn explicit_transaction_binary_record_is_one_atomic_recoverable_generation() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(
        1,
        "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, marker INT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance, marker) VALUES (1, 100, NULL)",
    )
    .unwrap();
    let existing_row_id = e.read_state.mvcc.current_row_id() - 1;
    let inserted_row_id = e.read_state.mvcc.current_row_id();
    let record = BinaryTransactionRecord {
        allocator_high_water: inserted_row_id + 1,
        sequence_advances: BTreeMap::new(),
        mutations: vec![
            BinaryTransactionMutation::Insert {
                table: "accounts".to_string(),
                row_id: inserted_row_id,
                row_encoded: encode_relational_row(&[
                    SqlValue::Int4(2),
                    SqlValue::Int4(300),
                    SqlValue::Null,
                ]),
            },
            BinaryTransactionMutation::Update {
                table: "accounts".to_string(),
                row_id: inserted_row_id,
                old_row_encoded: encode_relational_row(&[
                    SqlValue::Int4(2),
                    SqlValue::Int4(300),
                    SqlValue::Null,
                ]),
                new_row_encoded: encode_relational_row(&[
                    SqlValue::Int4(2),
                    SqlValue::Int4(350),
                    SqlValue::Null,
                ]),
            },
            BinaryTransactionMutation::Update {
                table: "accounts".to_string(),
                row_id: existing_row_id,
                old_row_encoded: encode_relational_row(&[
                    SqlValue::Int4(1),
                    SqlValue::Int4(100),
                    SqlValue::Null,
                ]),
                new_row_encoded: encode_relational_row(&[
                    SqlValue::Int4(1),
                    SqlValue::Int4(200),
                    SqlValue::Null,
                ]),
            },
            BinaryTransactionMutation::Delete {
                table: "accounts".to_string(),
                row_id: existing_row_id,
                old_row_encoded: encode_relational_row(&[
                    SqlValue::Int4(1),
                    SqlValue::Int4(200),
                    SqlValue::Null,
                ]),
            },
        ],
    };
    let payload = try_encode_binary_transaction(&record).unwrap();
    let wal_before = e.durable_wal_records().len();
    let token = e.commit_mutation(90, Arc::from(payload)).unwrap();
    let durable = e.durable_wal_records();
    assert_eq!(durable.len(), wal_before + 1, "one transaction WAL record");
    assert!(matches!(
        decode_binary_record(&durable.last().unwrap().payload).unwrap(),
        BinaryWalRecord::Transaction(decoded) if decoded == record
    ));
    assert_eq!(e.visible_up_to(), token.index, "one published commit index");

    let select =
        match parse_command("SELECT id, balance, marker FROM accounts ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    let live = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        live.rows.row(0),
        &[SqlValue::Int4(2), SqlValue::Int4(350), SqlValue::Null]
    );
    assert_eq!(live.rows.len(), 1);

    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    let replayed = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(
        replayed.rows, live.rows,
        "live apply and replay are identical"
    );
    assert_eq!(
        recovered.read_state.mvcc.current_row_id(),
        record.allocator_high_water,
        "replay restores the transaction's claimed allocator high-water"
    );
}

// Query the `id` values from `people`, sorted, for crash-recovery assertions.
fn select_people_ids(engine: &mut Engine) -> Vec<i32> {
    let Command::Select(select) = parse_command("SELECT id FROM people ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = engine.execute_relational_select(&select).unwrap();
    result
        .rows
        .iter()
        .map(|row| match row.iter().next().unwrap() {
            SqlValue::Int4(v) => *v,
            other => panic!("expected Int4, got {other:?}"),
        })
        .collect()
}

#[test]
fn durable_engine_recovers_committed_rows_after_simulated_crash() {
    let path = test_wal_path("durable-recover");

    // --- session 1: a durable engine commits two statements, then "crashes" (is dropped). ---
    {
        let e = Engine::with_durable_wal_segment(&path);
        assert!(e.wal_is_durable());
        e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
            .unwrap();
        e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
        )
        .unwrap();
        // Each committed statement fsynced its WAL before becoming visible.
        assert!(e.wal_group_commit_stats().flush_groups >= 2);
        assert_eq!(e.wal_unflushed_count(), 0);
    } // engine dropped == process crash; only the fsynced segment survives.

    // --- session 2: reopen from the durable segment. The committed effects must survive. ---
    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert!(recovered.wal_is_durable());
    assert_eq!(recovered.wal_flushed_count(), 2);
    assert_eq!(select_people_ids(&mut recovered), vec![1, 2]);

    // Post-recovery commits keep appending durably to the SAME segment (history preserved).
    recovered
        .execute_text(3, "INSERT INTO people (id, name) VALUES (3, 'Grace')")
        .unwrap();
    drop(recovered);
    let mut reopened = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(reopened.wal_flushed_count(), 3);
    assert_eq!(select_people_ids(&mut reopened), vec![1, 2, 3]);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn durable_engine_does_not_recover_a_txn_whose_wal_fsync_failed() {
    // The precise WAL-before-visibility boundary: a commit whose WAL fsync does NOT complete
    // must be neither durable NOR visible — no torn state, no visible-but-not-durable row.
    let path = test_wal_path("durable-fsync-boundary");

    let mut e = Engine::with_durable_wal_segment(&path);
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    // txn 1 (CREATE) and txn 2 (INSERT id=1) are fsync-durable and visible.
    assert_eq!(select_people_ids(&mut e), vec![1]);
    let durable_before = e.wal_flushed_count();
    assert_eq!(durable_before, 2);

    // Now the next WAL fsync fails mid-commit (kill-mid-commit). The commit must abort.
    e.simulate_next_wal_flush_failure();
    let failed = e.execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Linus')");
    assert!(
        matches!(
            failed,
            Err(ExecuteError::Engine(EngineError::Durability(_)))
        ),
        "expected the fsync-failed commit to abort with a durability error, got {failed:?}"
    );

    // The failed txn is invisible in the SAME (live) session: not durable, not applied.
    assert_eq!(e.wal_flushed_count(), durable_before);
    assert_eq!(select_people_ids(&mut e), vec![1]);

    // "Crash" and recover from the durable segment: only the fsync-durable txns come back;
    // the failed INSERT (id=2) is absent.
    drop(e);
    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(recovered.wal_flushed_count(), 2);
    assert_eq!(
        select_people_ids(&mut recovered),
        vec![1],
        "a txn whose WAL fsync did not complete must not be visible after recovery"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn durable_recovery_rejects_a_torn_trailing_record() {
    // A crash that leaves a partially-written (torn) trailing record on disk must be detected
    // by the segment CRC at recovery time — recovery fails loudly rather than replaying garbage
    // or silently truncating, so there is never torn state.
    let path = test_wal_path("durable-torn");
    {
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
            .unwrap();
    }

    // Corrupt the last byte of the LOGICAL durable tail (W4a: the physical file carries a
    // preallocated zero tail past it — the file's last byte is a zero, not record data). This
    // damages an ACKNOWLEDGED record below the recorded tail offset, which recovery must
    // reject loudly.
    let valid_bytes = gpu_db_wal::recover_wal_segment(&path).unwrap().valid_bytes;
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[valid_bytes as usize - 1] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();

    let result = Engine::open_durable_wal_segment(&path);
    let _ = std::fs::remove_file(&path);
    match result {
        Err(EngineError::Durability(msg)) => {
            assert!(
                msg.contains("checksum mismatch"),
                "expected a CRC durability error for a torn record, got {msg:?}"
            );
        }
        Err(other) => panic!("expected a CRC durability error, got {other:?}"),
        Ok(_) => panic!("expected recovery to reject a torn trailing record"),
    }
}

#[test]
fn durable_recovery_truncates_a_torn_append_tail_and_continues() {
    // The append-only writer means a crash mid-append leaves a torn record BEYOND the recorded
    // durable tail offset. That commit was never acknowledged (WAL-before-visibility: the fsync
    // never completed), so recovery truncates it and the database keeps serving and appending.
    // Contrast with `durable_recovery_rejects_a_torn_trailing_record`, where the damage is to an
    // ACKNOWLEDGED record (below the recorded tail) and recovery must fail loudly.
    let path = test_wal_path("durable-torn-append");
    {
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
            .unwrap();
    }
    // Simulate the crash mid-append: garbage bytes past the acknowledged tail.
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"\xAB\xCD\xEF torn half-record").unwrap();
    }

    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(recovered.wal_flushed_count(), 2);
    assert_eq!(select_people_ids(&mut recovered), vec![1]);

    // The truncated segment keeps accepting appends, and a further restart sees them.
    recovered
        .execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Linus')")
        .unwrap();
    drop(recovered);
    let mut reopened = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(select_people_ids(&mut reopened), vec![1, 2]);

    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn checkpoint_and_truncate_bounds_the_live_segment_and_recovers_with_checkpoint() {
    // D2: a checkpoint persists the full durable history to a checkpoint segment + control file,
    // then trims the LIVE segment to only post-checkpoint records — the live file is bounded by
    // the checkpoint cadence. Recovery pairs the checkpoint with the live suffix.
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-ckpt-truncate-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let segment_path = dir.join("live.wal");
    let control_path = dir.join("CONTROL");
    let checkpoint_segment_path = dir.join("checkpoint-0001.wal");

    let e = serial_durable_engine(&segment_path);
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    let live_before = e.wal_durable_segment_bytes();

    let meta = e
        .checkpoint_and_truncate_durable_wal(&control_path, &checkpoint_segment_path)
        .unwrap();
    assert_eq!(meta.durable_record_count, 2);
    assert!(
        e.wal_durable_segment_bytes() < live_before,
        "the live segment must shrink at the checkpoint ({} -> {})",
        live_before,
        e.wal_durable_segment_bytes()
    );

    // Post-checkpoint commits land only in the live segment.
    e.execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Linus')")
        .unwrap();
    drop(e);

    let mut recovered =
        Engine::open_durable_wal_segment_with_checkpoint(&control_path, &segment_path).unwrap();
    assert_eq!(recovered.wal_flushed_count(), 3);
    assert_eq!(select_people_ids(&mut recovered), vec![1, 2]);

    // The recovered engine keeps appending durably; a second restart sees everything, and a
    // SECOND checkpoint is self-contained (full history), not just the suffix.
    recovered
        .execute_text(4, "INSERT INTO people (id, name) VALUES (3, 'Grace')")
        .unwrap();
    recovered
        .checkpoint_and_truncate_durable_wal(&control_path, &checkpoint_segment_path)
        .unwrap();
    drop(recovered);
    let mut reopened =
        Engine::open_durable_wal_segment_with_checkpoint(&control_path, &segment_path).unwrap();
    assert_eq!(select_people_ids(&mut reopened), vec![1, 2, 3]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checkpoint_truncation_prunes_the_commit_timestamp_map() {
    // R2 (write-path assessment): the commit-timestamp map grew one entry per commit forever.
    // The checkpoint boundary discards the covered prefix's timestamps; post-checkpoint commits
    // keep recording (and stay strictly monotonic via the running max, which pruning never
    // touches).
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-ts-prune-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let segment_path = dir.join("live.wal");
    let control_path = dir.join("CONTROL");
    let checkpoint_segment_path = dir.join("checkpoint-0001.wal");

    let e = serial_durable_engine(&segment_path);
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    let max_before = {
        let commit = e.commit_state();
        assert_eq!(commit.wal_commit_timestamps_micros.len(), 2);
        commit.max_commit_timestamp_micros
    };

    e.checkpoint_and_truncate_durable_wal(&control_path, &checkpoint_segment_path)
        .unwrap();
    {
        let commit = e.commit_state();
        assert!(
            commit.wal_commit_timestamps_micros.is_empty(),
            "checkpointed records' timestamps are discarded with the prefix"
        );
        assert_eq!(
            commit.max_commit_timestamp_micros, max_before,
            "the monotonicity floor survives the prune"
        );
    }

    // Post-checkpoint commits record fresh (still strictly monotonic) timestamps.
    e.execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Linus')")
        .unwrap();
    {
        let commit = e.commit_state();
        assert_eq!(commit.wal_commit_timestamps_micros.len(), 1);
        assert!(commit.max_commit_timestamp_micros > max_before);
    }

    // Recovery is timestamp-independent: the checkpoint + live suffix still replay fully.
    drop(e);
    let mut recovered =
        Engine::open_durable_wal_segment_with_checkpoint(&control_path, &segment_path).unwrap();
    assert_eq!(select_people_ids(&mut recovered), vec![1, 2]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checkpoint_size_bound_policy_rotates_only_beyond_the_bound() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-ckpt-bound-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let segment_path = dir.join("live.wal");
    let control_path = dir.join("CONTROL");
    let checkpoint_segment_path = dir.join("checkpoint-0001.wal");

    let e = serial_durable_engine(&segment_path);
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();

    // Under a generous bound: no rotation, no checkpoint files.
    let rotated = e
        .checkpoint_and_truncate_durable_wal_if_larger_than(
            &control_path,
            &checkpoint_segment_path,
            1 << 20,
        )
        .unwrap();
    assert!(!rotated);
    assert!(!control_path.exists());

    // Over a tiny bound: rotation runs and the live segment shrinks below it.
    let rotated = e
        .checkpoint_and_truncate_durable_wal_if_larger_than(
            &control_path,
            &checkpoint_segment_path,
            16,
        )
        .unwrap();
    assert!(rotated);
    assert!(control_path.exists());
    assert!(e.wal_durable_segment_bytes() <= 16);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn open_durable_wal_segment_on_missing_path_is_a_fresh_durable_db() {
    let path = test_wal_path("durable-fresh");
    assert!(!path.exists());
    let e = Engine::open_durable_wal_segment(&path).unwrap();
    assert!(e.wal_is_durable());
    assert_eq!(e.wal_flushed_count(), 0);
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (7, 'Ada')")
        .unwrap();
    drop(e);

    let mut recovered = Engine::open_durable_wal_segment(&path).unwrap();
    assert_eq!(select_people_ids(&mut recovered), vec![7]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn relational_state_recovers_from_wal_checkpoint_control_after_restart() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-control-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("CONTROL");
    let segment_path = dir.join("segment-0001.wal");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();

    e.persist_durable_wal_checkpoint(&control_path, &segment_path)
        .unwrap();
    let recovered = Engine::recover_from_durable_wal_checkpoint(&control_path).unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(recovered.wal_unflushed_count(), 0);
    assert_eq!(recovered.wal_flushed_count(), 2);
    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = recovered.execute_relational_select(&select).unwrap();

    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(2)]]);
}

#[test]
fn relational_state_recovers_from_multi_segment_wal_archive() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    e.execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Grace')")
        .unwrap();

    let manifest = e
        .persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    assert_eq!(manifest.segments.len(), 3);

    let recovered = Engine::recover_from_durable_wal_archive(&manifest_path).unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(recovered.wal_unflushed_count(), 0);
    assert_eq!(recovered.wal_flushed_count(), 3);
    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);
    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = recovered.execute_relational_select(&select).unwrap();

    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(2)]]);
}

#[test]
fn relational_state_recovers_from_wal_archive_object_backup() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-object-backup-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("source").join("MANIFEST");
    let segment_dir = dir.join("source").join("segments");
    let backup_path = dir.join("backup").join("BACKUP");
    let object_dir = dir.join("backup").join("objects");
    let restored_manifest_path = dir.join("restored").join("MANIFEST");
    let restored_segment_dir = dir.join("restored").join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    let backup =
        Engine::export_durable_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir)
            .unwrap();
    let restored_manifest = Engine::restore_durable_wal_archive_object_backup(
        &backup_path,
        &restored_manifest_path,
        &restored_segment_dir,
    )
    .unwrap();
    let recovered = Engine::recover_from_durable_wal_archive_to_timestamp_micros(
        &restored_manifest_path,
        3_000,
    )
    .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(backup.objects.len(), 4);
    assert_eq!(restored_manifest.checkpoint.durable_record_count, 3);
    assert_eq!(restored_manifest.record_timestamps.len(), 3);
    assert_eq!(recovered.wal_flushed_count(), 3);
    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);
    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = recovered.execute_relational_select(&select).unwrap();

    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(2)]]);
}

#[test]
fn relational_state_recovers_after_wal_archive_segment_ingestion() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-ingest-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let ingest_segment = segment_dir.join("segment-0002.wal");
    let base = Engine::new_local_cpu_oracle();
    base.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    base.execute_text_at_timestamp_micros(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada')",
        2_000,
    )
    .unwrap();
    base.persist_durable_wal_archive(&manifest_path, &segment_dir, 2)
        .unwrap();

    let tail_records = vec![
        WalRecord {
            txn_id: 3,
            payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                .to_vec()
                .into(),
        },
        WalRecord {
            txn_id: 4,
            payload: b"INSERT INTO people (id, name) VALUES (3, 'Katherine')"
                .to_vec()
                .into(),
        },
    ];
    let tail_timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 3,
            timestamp_micros: 3_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 4,
            timestamp_micros: 4_000,
        },
    ];
    write_wal_segment(&ingest_segment, &tail_records).unwrap();
    let manifest = Engine::ingest_durable_wal_archive_segment(
        &manifest_path,
        &ingest_segment,
        &tail_timestamps,
    )
    .unwrap();

    let recovered = Engine::recover_from_durable_wal_archive(&manifest_path).unwrap();
    let timestamp_recovered =
        Engine::recover_from_durable_wal_archive_to_timestamp_micros(&manifest_path, 3_000)
            .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(manifest.checkpoint.durable_record_count, 4);
    assert_eq!(manifest.checkpoint.last_durable_txn_id, Some(4));
    assert_eq!(manifest.record_timestamps.len(), 4);
    assert_eq!(recovered.wal_flushed_count(), 4);
    assert_eq!(timestamp_recovered.wal_flushed_count(), 3);

    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);
    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let timestamp_grace = timestamp_recovered
        .execute_relational_select(&grace_select)
        .unwrap();
    assert_eq!(timestamp_grace.rows, vec![vec![SqlValue::Int4(2)]]);
    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let timestamp_katherine = timestamp_recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    assert_eq!(timestamp_katherine.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn relational_state_recovers_from_wal_archive_transaction_target() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-target-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    e.execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Grace')")
        .unwrap();
    e.execute_text(4, "INSERT INTO people (id, name) VALUES (3, 'Katherine')")
        .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 2)
        .unwrap();
    let recovered = Engine::recover_from_durable_wal_archive_to_txn(&manifest_path, 3).unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(recovered.wal_unflushed_count(), 0);
    assert_eq!(recovered.wal_flushed_count(), 3);
    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn relational_state_recovers_from_wal_archive_timestamp_target() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-timestamp-target-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();

    let manifest = e
        .persist_durable_wal_archive(&manifest_path, &segment_dir, 2)
        .unwrap();
    assert_eq!(manifest.record_timestamps.len(), 4);
    let recovered =
        Engine::recover_from_durable_wal_archive_to_timestamp_micros(&manifest_path, 3_000)
            .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(recovered.wal_unflushed_count(), 0);
    assert_eq!(recovered.wal_flushed_count(), 3);
    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn relational_state_recovers_from_forked_wal_archive_timeline_branch() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-timeline-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let source_manifest = dir.join("source").join("MANIFEST");
    let source_segments = dir.join("source").join("segments");
    let branch_manifest = dir.join("branch").join("MANIFEST");
    let branch_segments = dir.join("branch").join("segments");
    let pruned_branch_manifest = dir.join("pruned-branch").join("MANIFEST");
    let pruned_branch_segments = dir.join("pruned-branch").join("segments");
    let source_timeline_path = dir.join("source").join("TIMELINE");
    let timeline_path = dir.join("branch").join("TIMELINE");
    let pruned_timeline_path = dir.join("pruned-branch").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();
    e.persist_durable_wal_archive(&source_manifest, &source_segments, 2)
        .unwrap();
    Engine::write_durable_wal_archive_timeline(
        &source_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-main-0001".to_string(),
            parent_timeline_id: None,
            fork_txn_id: 0,
            fork_timestamp_micros: None,
            source_manifest_path: source_manifest.clone(),
            branch_manifest_path: source_manifest.clone(),
        },
    )
    .unwrap();

    let branch = Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(
        &source_manifest,
        &branch_manifest,
        &branch_segments,
        &timeline_path,
        "timeline-branch-0002",
        Some("timeline-main-0001"),
        3_000,
    )
    .unwrap();
    let timeline = Engine::read_durable_wal_archive_timeline(&timeline_path).unwrap();
    Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(
        &source_manifest,
        &pruned_branch_manifest,
        &pruned_branch_segments,
        &pruned_timeline_path,
        "timeline-pruned-0003",
        Some("timeline-main-0001"),
        2_000,
    )
    .unwrap();
    Engine::register_durable_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
    let registry =
        Engine::register_durable_wal_archive_timeline(&registry_path, &timeline_path).unwrap();
    Engine::register_durable_wal_archive_timeline(&registry_path, &pruned_timeline_path).unwrap();
    let selection =
        Engine::select_durable_wal_archive_timeline(&registry_path, "timeline-branch-0002")
            .unwrap();
    let prune_plan =
        Engine::apply_durable_wal_archive_timeline_prune(&registry_path, "timeline-branch-0002")
            .unwrap();
    let missing_selection_err =
        Engine::select_durable_wal_archive_timeline(&registry_path, "timeline-missing")
            .unwrap_err();
    let pruned_selection_err =
        Engine::select_durable_wal_archive_timeline(&registry_path, "timeline-pruned-0003")
            .unwrap_err();
    let recovered = Engine::recover_from_registered_durable_wal_archive_timeline(
        &registry_path,
        "timeline-branch-0002",
    )
    .unwrap();

    assert_eq!(branch.timeline, timeline);
    assert_eq!(selection.timeline, timeline);
    assert_eq!(selection.entry.branch_manifest_path, branch_manifest);
    assert_eq!(selection.manifest.checkpoint.durable_record_count, 3);
    assert!(missing_selection_err
        .to_string()
        .contains("has no timeline timeline-missing"));
    assert!(pruned_selection_err
        .to_string()
        .contains("has no timeline timeline-pruned-0003"));
    assert_eq!(registry.timelines.len(), 2);
    assert_eq!(registry.timelines[1].timeline_id, "timeline-branch-0002");
    assert_eq!(
        prune_plan.retained_timeline_ids,
        vec![
            "timeline-main-0001".to_string(),
            "timeline-branch-0002".to_string()
        ]
    );
    assert_eq!(
        prune_plan.removed_timeline_ids,
        vec!["timeline-pruned-0003".to_string()]
    );
    assert!(!pruned_timeline_path.exists());
    assert!(!pruned_branch_manifest.exists());
    assert!(!pruned_branch_segments.join("segment-0001.wal").exists());
    assert_eq!(
        registry.timelines[1].parent_timeline_id.as_deref(),
        Some("timeline-main-0001")
    );
    assert_eq!(timeline.timeline_id, "timeline-branch-0002");
    assert_eq!(
        timeline.parent_timeline_id.as_deref(),
        Some("timeline-main-0001")
    );
    assert_eq!(timeline.fork_txn_id, 3);
    assert_eq!(timeline.fork_timestamp_micros, Some(3_000));
    assert_eq!(branch.manifest.checkpoint.durable_record_count, 3);
    assert_eq!(recovered.wal_flushed_count(), 3);

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn relational_state_recovers_from_base_checkpoint_plus_wal_archive_transaction_target() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-base-archive-target-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    e.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();
    e.execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Grace')")
        .unwrap();
    e.execute_text(4, "INSERT INTO people (id, name) VALUES (3, 'Katherine')")
        .unwrap();
    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 2)
        .unwrap();

    let recovered = Engine::recover_from_durable_wal_checkpoint_and_archive_to_txn(
        &control_path,
        &manifest_path,
        3,
    )
    .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(recovered.wal_flushed_count(), 3);
    let table = recovered.relational_catalog_table("people").unwrap();
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn relational_state_recovers_from_base_checkpoint_plus_wal_archive_timestamp_target() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-base-archive-timestamp-target-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();
    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 2)
        .unwrap();

    let recovered = Engine::recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(
        &control_path,
        &manifest_path,
        3_000,
    )
    .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(recovered.wal_flushed_count(), 3);
    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn base_checkpoint_plus_wal_archive_rejects_missing_base_overlap() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-base-archive-missing-overlap-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let base = Engine::new_local_cpu_oracle();
    base.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    base.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    base.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();

    let archive = Engine::new_local_cpu_oracle();
    archive
        .execute_text(3, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    archive
        .execute_text(4, "INSERT INTO people (id, name) VALUES (2, 'Grace')")
        .unwrap();
    archive
        .persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();

    let err = match Engine::recover_from_durable_wal_checkpoint_and_archive_to_txn(
        &control_path,
        &manifest_path,
        4,
    ) {
        Ok(_) => panic!("expected missing base overlap error"),
        Err(err) => err,
    };
    let _ = std::fs::remove_dir_all(dir);

    assert!(err.to_string().contains("does not overlap base backup"));
}

#[test]
fn engine_written_wal_archive_timestamps_are_monotonic() {
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET a=1").unwrap();
    e.execute_text(2, "SET b=2").unwrap();

    let timestamps = e.durable_wal_record_timestamps();

    assert_eq!(timestamps.len(), 2);
    assert!(timestamps[0].timestamp_micros < timestamps[1].timestamp_micros);
}

#[test]
fn relational_state_recovers_after_wal_archive_retention_cleanup() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-retention-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    e.execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Grace')")
        .unwrap();
    e.execute_text(4, "INSERT INTO people (id, name) VALUES (3, 'Katherine')")
        .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    let obsolete_tail = segment_dir.join("segment-0004.wal");
    assert!(obsolete_tail.exists());
    let plan = Engine::apply_durable_wal_archive_retention_to_txn(&manifest_path, 3).unwrap();

    let recovered = Engine::recover_from_durable_wal_archive(&manifest_path).unwrap();
    assert!(!obsolete_tail.exists());
    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(plan.removed_record_count, 1);
    assert_eq!(recovered.wal_flushed_count(), 3);

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn relational_state_recovers_after_timestamp_wal_archive_retention_cleanup() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-timestamp-retention-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    let plan =
        Engine::apply_durable_wal_archive_retention_to_timestamp_micros(&manifest_path, 3_000)
            .unwrap();
    let recovered = Engine::recover_from_durable_wal_archive(&manifest_path).unwrap();
    let timestamp_err =
        match Engine::recover_from_durable_wal_archive_to_timestamp_micros(&manifest_path, 4_000) {
            Ok(_) => panic!("expected timestamp target beyond retained archive to fail"),
            Err(err) => err,
        };

    assert_eq!(plan.target_txn_id, 3);
    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(plan.removed_record_count, 1);
    assert_eq!(recovered.wal_flushed_count(), 3);
    assert!(timestamp_err
        .to_string()
        .contains("beyond last durable timestamp"));

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn relational_state_recovers_after_base_checkpoint_archive_retention_cleanup() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-base-archive-retention-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    let plan =
        Engine::apply_durable_wal_archive_retention_from_checkpoint(&control_path, &manifest_path)
            .unwrap();
    let recovered = Engine::recover_from_durable_wal_checkpoint_and_archive_to_txn(
        &control_path,
        &manifest_path,
        3,
    )
    .unwrap();
    let timestamp_recovered =
        Engine::recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(
            &control_path,
            &manifest_path,
            3_000,
        )
        .unwrap();

    assert_eq!(plan.target_txn_id, 2);
    assert_eq!(plan.removed_record_count, 1);
    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(recovered.wal_flushed_count(), 3);
    assert_eq!(timestamp_recovered.wal_flushed_count(), 3);

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_recovered_relational_access_path(
        &grace_result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);

    let timestamp_grace = timestamp_recovered
        .execute_relational_select(&grace_select)
        .unwrap();
    assert_eq!(timestamp_grace.rows, vec![vec![SqlValue::Int4(2)]]);

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(katherine_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn base_checkpoint_archive_retention_rejects_prefix_mismatch_before_cleanup() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-base-archive-retention-mismatch-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let base = Engine::new_local_cpu_oracle();
    base.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    base.execute_text(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    base.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();

    let archive = Engine::new_local_cpu_oracle();
    archive
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    archive
        .execute_text(2, "INSERT INTO people (id, name) VALUES (99, 'Mismatch')")
        .unwrap();
    archive
        .execute_text(3, "INSERT INTO people (id, name) VALUES (2, 'Grace')")
        .unwrap();
    archive
        .persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();

    let original_manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let err = match Engine::apply_durable_wal_archive_retention_from_checkpoint(
        &control_path,
        &manifest_path,
    ) {
        Ok(_) => panic!("expected prefix mismatch error"),
        Err(err) => err,
    };
    let after_manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert!(err.to_string().contains("prefix does not match"));
    assert_eq!(after_manifest, original_manifest);
}

#[test]
fn checkpoint_window_archive_retention_preserves_pitr_recovery() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-checkpoint-window-retention-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        5,
        "INSERT INTO people (id, name) VALUES (4, 'Dorothy')",
        5_000,
    )
    .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    let plan = Engine::apply_durable_wal_archive_retention_from_checkpoint_window(
        &control_path,
        &manifest_path,
        6_000,
        3_000,
    )
    .unwrap();
    let recovered = Engine::recover_from_durable_wal_checkpoint_and_archive_to_txn(
        &control_path,
        &manifest_path,
        4,
    )
    .unwrap();
    let timestamp_recovered =
        Engine::recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(
            &control_path,
            &manifest_path,
            4_000,
        )
        .unwrap();

    assert_eq!(plan.cutoff_timestamp_micros, 3_000);
    assert_eq!(plan.base_txn_id, 2);
    assert_eq!(plan.base_timestamp_micros, 2_000);
    assert_eq!(plan.retention_plan.retained_record_count, 4);
    assert_eq!(plan.retention_plan.removed_record_count, 1);
    assert_eq!(recovered.wal_flushed_count(), 4);
    assert_eq!(timestamp_recovered.wal_flushed_count(), 4);

    let Command::Select(grace_select) =
        parse_command("SELECT id FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let grace_result = recovered.execute_relational_select(&grace_select).unwrap();
    assert_eq!(grace_result.rows, vec![vec![SqlValue::Int4(2)]]);
    let timestamp_grace = timestamp_recovered
        .execute_relational_select(&grace_select)
        .unwrap();
    assert_eq!(timestamp_grace.rows, vec![vec![SqlValue::Int4(2)]]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn maintenance_cleanup_prunes_archive_and_timelines_before_registered_recovery() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-maintenance-cleanup-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let source_timeline_path = dir.join("timeline-main").join("TIMELINE");
    let keep_manifest = dir.join("timeline-keep").join("MANIFEST");
    let keep_segments = dir.join("timeline-keep").join("segments");
    let keep_timeline_path = dir.join("timeline-keep").join("TIMELINE");
    let prune_manifest = dir.join("timeline-prune").join("MANIFEST");
    let prune_segments = dir.join("timeline-prune").join("segments");
    let prune_timeline_path = dir.join("timeline-prune").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();
    e.execute_text_at_timestamp_micros(
        5,
        "INSERT INTO people (id, name) VALUES (4, 'Dorothy')",
        5_000,
    )
    .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    Engine::write_durable_wal_archive_timeline(
        &source_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-main-0001".to_string(),
            parent_timeline_id: None,
            fork_txn_id: 0,
            fork_timestamp_micros: None,
            source_manifest_path: manifest_path.clone(),
            branch_manifest_path: manifest_path.clone(),
        },
    )
    .unwrap();
    Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(
        &manifest_path,
        &keep_manifest,
        &keep_segments,
        &keep_timeline_path,
        "timeline-keep-0002",
        Some("timeline-main-0001"),
        4_000,
    )
    .unwrap();
    Engine::fork_durable_wal_archive_timeline_to_timestamp_micros(
        &manifest_path,
        &prune_manifest,
        &prune_segments,
        &prune_timeline_path,
        "timeline-prune-0003",
        Some("timeline-main-0001"),
        3_000,
    )
    .unwrap();
    Engine::register_durable_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
    Engine::register_durable_wal_archive_timeline(&registry_path, &keep_timeline_path).unwrap();
    Engine::register_durable_wal_archive_timeline(&registry_path, &prune_timeline_path).unwrap();

    let dry_run = Engine::plan_durable_wal_archive_maintenance_cleanup(
        &control_path,
        &manifest_path,
        &registry_path,
        "timeline-keep-0002",
        6_000,
        3_000,
    )
    .unwrap();
    let applied = Engine::apply_durable_wal_archive_maintenance_cleanup(
        &control_path,
        &manifest_path,
        &registry_path,
        "timeline-keep-0002",
        6_000,
        3_000,
    )
    .unwrap();
    let recovered = Engine::recover_from_registered_durable_wal_archive_timeline(
        &registry_path,
        "timeline-keep-0002",
    )
    .unwrap();
    let pruned_err =
        Engine::select_durable_wal_archive_timeline(&registry_path, "timeline-prune-0003")
            .unwrap_err();

    assert_eq!(dry_run, applied);
    assert_eq!(applied.retention_window_plan.base_txn_id, 2);
    assert_eq!(
        applied
            .retention_window_plan
            .retention_plan
            .removed_record_count,
        1
    );
    assert_eq!(
        applied.timeline_prune_plan.retained_timeline_ids,
        vec![
            "timeline-main-0001".to_string(),
            "timeline-keep-0002".to_string()
        ]
    );
    assert_eq!(
        applied.timeline_prune_plan.removed_timeline_ids,
        vec!["timeline-prune-0003".to_string()]
    );
    let (retained_archive, _retained_records) = read_wal_archive(&manifest_path).unwrap();
    assert_eq!(retained_archive.segments[0].first_txn_id, Some(2));
    assert!(!prune_timeline_path.exists());
    assert!(!prune_manifest.exists());
    assert!(keep_timeline_path.exists());
    assert!(keep_manifest.exists());
    assert!(pruned_err
        .to_string()
        .contains("has no timeline timeline-prune-0003"));

    let Command::Select(katherine_select) =
        parse_command("SELECT id FROM people WHERE name = 'Katherine'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let katherine_result = recovered
        .execute_relational_select(&katherine_select)
        .unwrap();
    let Command::Select(dorothy_select) =
        parse_command("SELECT id FROM people WHERE name = 'Dorothy'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let dorothy_result = recovered
        .execute_relational_select(&dorothy_select)
        .unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert_eq!(katherine_result.rows, vec![vec![SqlValue::Int4(3)]]);
    assert_eq!(dorothy_result.rows, Vec::<Vec<SqlValue>>::new());
}

#[test]
fn maintenance_cleanup_rejects_stale_timeline_before_archive_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-maintenance-cleanup-stale-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let source_timeline_path = dir.join("timeline-main").join("TIMELINE");
    let branch_manifest = dir.join("timeline-branch").join("MANIFEST");
    let branch_segments = dir.join("timeline-branch").join("segments");
    let branch_timeline_path = dir.join("timeline-branch").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    Engine::write_durable_wal_archive_timeline(
        &source_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-main-0001".to_string(),
            parent_timeline_id: None,
            fork_txn_id: 0,
            fork_timestamp_micros: None,
            source_manifest_path: manifest_path.clone(),
            branch_manifest_path: manifest_path.clone(),
        },
    )
    .unwrap();
    Engine::fork_durable_wal_archive_timeline_to_txn(
        &manifest_path,
        &branch_manifest,
        &branch_segments,
        &branch_timeline_path,
        "timeline-branch-0002",
        Some("timeline-main-0001"),
        3,
    )
    .unwrap();
    Engine::register_durable_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
    Engine::register_durable_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();
    let original_manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let original_registry = std::fs::read_to_string(&registry_path).unwrap();
    Engine::write_durable_wal_archive_timeline(
        &branch_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-branch-0002".to_string(),
            parent_timeline_id: Some("timeline-main-0001".to_string()),
            fork_txn_id: 2,
            fork_timestamp_micros: None,
            source_manifest_path: manifest_path.clone(),
            branch_manifest_path: branch_manifest.clone(),
        },
    )
    .unwrap();

    let err = Engine::apply_durable_wal_archive_maintenance_cleanup(
        &control_path,
        &manifest_path,
        &registry_path,
        "timeline-branch-0002",
        6_000,
        3_000,
    )
    .unwrap_err();
    let after_manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let after_registry = std::fs::read_to_string(&registry_path).unwrap();
    let _ = std::fs::remove_dir_all(dir);

    assert!(err.to_string().contains("does not match sidecar"));
    assert_eq!(after_manifest, original_manifest);
    assert_eq!(after_registry, original_registry);
}

#[test]
fn checkpoint_window_archive_retention_rejects_unsafe_recent_base_without_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-checkpoint-window-retention-reject-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("base").join("CONTROL");
    let base_segment_path = dir.join("base").join("base.wal");
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text_at_timestamp_micros(1, "CREATE TABLE people (id INT, name TEXT)", 1_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(2, "INSERT INTO people (id, name) VALUES (1, 'Ada')", 2_000)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        3,
        "INSERT INTO people (id, name) VALUES (2, 'Grace')",
        3_000,
    )
    .unwrap();
    e.persist_durable_wal_checkpoint(&control_path, &base_segment_path)
        .unwrap();
    e.execute_text_at_timestamp_micros(
        4,
        "INSERT INTO people (id, name) VALUES (3, 'Katherine')",
        4_000,
    )
    .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    let original_manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let first_segment = segment_dir.join("segment-0001.wal");
    let err = match Engine::apply_durable_wal_archive_retention_from_checkpoint_window(
        &control_path,
        &manifest_path,
        4_000,
        2_000,
    ) {
        Ok(_) => panic!("expected unsafe recent base error"),
        Err(err) => err,
    };
    let after_manifest = std::fs::read_to_string(&manifest_path).unwrap();

    assert!(err.to_string().contains("newer than PITR retention cutoff"));
    assert_eq!(after_manifest, original_manifest);
    assert!(first_segment.exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_archive_transaction_target_rejects_unavailable_durable_boundary() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-engine-wal-archive-target-missing-{}-{}",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(10, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(20, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();

    e.persist_durable_wal_archive(&manifest_path, &segment_dir, 1)
        .unwrap();
    let err = match Engine::recover_from_durable_wal_archive_to_txn(&manifest_path, 15) {
        Ok(_) => panic!("expected unavailable target transaction error"),
        Err(err) => err,
    };
    let _ = std::fs::remove_dir_all(dir);

    assert!(err
        .to_string()
        .contains("does not contain target transaction"));
}

#[test]
fn checkpoint_vacuum_prunes_mvcc_versions_only_at_durable_safe_boundary() {
    let path = test_wal_path("vacuum");
    let e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:1=closed").unwrap();

    assert_eq!(e.read_state.mvcc.version_count(), 2);
    assert_eq!(
        e.read_state
            .mvcc
            .kv_tuple_fetch_by_key("acct:1", StorageVisibility { read_txn_id: 1 })
            .unwrap()
            .map(|version| version.value),
        Some("open".to_string())
    );

    let stats = e.checkpoint_vacuum_mvcc_versions(1).unwrap();
    assert_eq!(
        stats,
        PruneStats {
            removed_versions: 0,
            removed_tuples: 0,
            remaining_versions: 2,
        }
    );

    let stats = e.checkpoint_vacuum_mvcc_versions(2).unwrap();
    assert_eq!(
        stats,
        PruneStats {
            removed_versions: 1,
            removed_tuples: 0,
            remaining_versions: 1,
        }
    );
    assert_eq!(
        e.read_state
            .mvcc
            .kv_tuple_fetch_by_key("acct:1", StorageVisibility { read_txn_id: 2 })
            .unwrap()
            .map(|version| version.value),
        Some("closed".to_string())
    );
    assert_eq!(
        e.read_state
            .mvcc
            .kv_tuple_fetch_by_key("acct:1", StorageVisibility { read_txn_id: 1 })
            .unwrap(),
        None
    );

    e.persist_durable_wal_to_file(&path).unwrap();
    let recovered = Engine::recover_from_durable_wal_file(&path).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(
        recovered
            .read_state
            .mvcc
            .kv_tuple_fetch_by_key("acct:1", StorageVisibility { read_txn_id: 1 })
            .unwrap()
            .map(|version| version.value),
        Some("open".to_string())
    );
}

#[test]
fn checkpoint_vacuum_rejects_unsafe_boundaries() {
    // Stage 4 reasons in `commit_seq`/`Index` space (was façade-`txn_id`): the durable boundary is
    // `committed_seq`, and the active-snapshot guard is the oldest active READ SNAPSHOT.
    let e = Engine::new_local_cpu_oracle();
    let no_commit_err = e.checkpoint_vacuum_mvcc_versions(1).unwrap_err();
    assert!(
        no_commit_err
            .to_string()
            .contains("requires a durable commit boundary"),
        "got: {no_commit_err}"
    );

    // Two committed writes → committed_seq advances to 2 (the durable boundary).
    e.execute_text(1, "SET acct:1=open").unwrap();
    e.execute_text(2, "SET acct:1=closed").unwrap();
    assert_eq!(e.committed_seq(), 2);

    // safe_commit_seq newer than the durable boundary is rejected.
    let newer_than_durable_err = e.checkpoint_vacuum_mvcc_versions(3).unwrap_err();
    assert!(
        newer_than_durable_err
            .to_string()
            .contains("newer than the durable commit boundary 2"),
        "got: {newer_than_durable_err}"
    );

    // An in-flight read snapshot at commit_seq 1 makes safe_commit_seq >= 1 unsafe (it could
    // prune a version that snapshot still needs).
    let guard = e.register_active_snapshot(1);
    let active_err = e.checkpoint_vacuum_mvcc_versions(1).unwrap_err();
    assert!(
        active_err
            .to_string()
            .contains("crosses active read snapshot 1"),
        "got: {active_err}"
    );
    drop(guard);

    // With no active snapshot, pruning strictly below the durable boundary is allowed.
    e.checkpoint_vacuum_mvcc_versions(1).unwrap();
}

/// W1b — the checkpoint-aware AUTO open (the facade's entry point): after a size-bound rotation
/// through the convention paths, a restart recovers checkpoint-then-suffix with the FULL history.
#[test]
fn w1b_auto_open_recovers_full_history_after_rotation() {
    let path = test_wal_path("w1b-auto-open");
    let control = gpu_db_wal::wal_checkpoint_control_path(&path);
    let checkpoint = gpu_db_wal::wal_checkpoint_segment_path(&path);
    {
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
        for i in 0..8 {
            e.execute_text(2 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, {i})"))
                .unwrap();
        }
        // Rotate at a tiny explicit bound (bypasses the env-configured default).
        let rotated = e
            .checkpoint_and_truncate_durable_wal_if_larger_than(&control, &checkpoint, 1)
            .unwrap();
        assert!(rotated, "the live segment must exceed a 1-byte bound");
        // Post-rotation commits land in the truncated live segment.
        for i in 8..12 {
            e.execute_text(2 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, {i})"))
                .unwrap();
        }
    }
    let recovered = Engine::open_durable_wal_segment_auto(&path).unwrap();
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
        unreachable!()
    };
    let rows = recovered.execute_relational_select(&count).unwrap().rows;
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int8(12)]],
        "auto open must recover the checkpointed prefix AND the live suffix"
    );
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&checkpoint));
    let _ = std::fs::remove_file(&control);
    let _ = std::fs::remove_file(&checkpoint);
    let _ = std::fs::remove_file(&path);
}

/// W1b — the rotation CRASH WINDOW: checkpoint segment + control file written, live-segment
/// truncation NOT performed (simulated crash between them). The auto open must (a) not replay
/// the checkpointed records twice, and (b) REPAIR the live segment to the suffix-only layout.
#[test]
fn w1b_auto_open_repairs_the_checkpoint_truncation_crash_window() {
    let path = test_wal_path("w1b-crash-window");
    let control = gpu_db_wal::wal_checkpoint_control_path(&path);
    let checkpoint = gpu_db_wal::wal_checkpoint_segment_path(&path);
    {
        // Pinned SERIAL (audit E2.5c-3 F1): this test fabricates a SERIAL checkpoint pair and
        // exercises the serial live/checkpoint overlap dedup; under the flipped FUA default the
        // live log would be `<path>.fua.*`, the serial recovery would see an empty live file,
        // and the dedup branch would never run — a vacuous pass.
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
        for i in 0..6 {
            e.execute_text(2 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, {i})"))
                .unwrap();
        }
        // Simulate the crashed rotation: write the checkpoint + control file exactly as
        // checkpoint_and_truncate_durable_wal does, but skip the truncation (the crash).
        let records = e.durable_wal_records();
        gpu_db_wal::write_wal_segment(&checkpoint, &records).unwrap();
        gpu_db_wal::write_wal_control_file(
            &control,
            &gpu_db_wal::WalControlFile {
                segment_path: checkpoint
                    .file_name()
                    .map(std::path::PathBuf::from)
                    .unwrap(),
                checkpoint: gpu_db_wal::WalCheckpointMeta {
                    durable_record_count: records.len(),
                    last_durable_txn_id: records.last().map(|r| r.txn_id),
                },
            },
        )
        .unwrap();
        // Engine drops WITHOUT truncating: the live segment still holds the full history.
    }
    let recovered = Engine::open_durable_wal_segment_auto(&path).unwrap();
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
        unreachable!()
    };
    let rows = recovered.execute_relational_select(&count).unwrap().rows;
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int8(6)]],
        "the overlap must be recovered exactly once (6 rows, not 12)"
    );
    // The open repaired the live segment: reopen again and verify convergence.
    drop(recovered);
    let reopened = Engine::open_durable_wal_segment_auto(&path).unwrap();
    let rows = reopened.execute_relational_select(&count).unwrap().rows;
    assert_eq!(rows, vec![vec![SqlValue::Int8(6)]]);
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&checkpoint));
    let _ = std::fs::remove_file(&control);
    let _ = std::fs::remove_file(&checkpoint);
    let _ = std::fs::remove_file(&path);
}

/// W1b — the rotation prunes ALL THREE per-commit unbounded structures in lock-step: the live
/// segment (bytes), the commit-timestamp map, and the replication log's applied prefix.
#[test]
fn w1b_rotation_prunes_timestamps_and_replication_log() {
    let path = test_wal_path("w1b-prune-trio");
    let control = gpu_db_wal::wal_checkpoint_control_path(&path);
    let checkpoint = gpu_db_wal::wal_checkpoint_segment_path(&path);
    let e = serial_durable_engine(&path);
    e.execute_text(1, "CREATE TABLE t (id INT)").unwrap();
    for i in 0..8 {
        e.execute_text(2 + i, &format!("INSERT INTO t (id) VALUES ({i})"))
            .unwrap();
    }
    let bytes_before = e.wal_durable_segment_bytes();
    let entries_before = e.replication_retained_entry_count();
    assert!(
        entries_before >= 9,
        "repl log holds every commit pre-rotation"
    );
    e.checkpoint_and_truncate_durable_wal_if_larger_than(&control, &checkpoint, 1)
        .unwrap();
    assert!(e.wal_durable_segment_bytes() < bytes_before);
    assert_eq!(
        e.replication_retained_entry_count(),
        0,
        "the applied replication prefix is discarded at the checkpoint boundary"
    );
    // The engine keeps serving writes after the rotation.
    e.execute_text(100, "INSERT INTO t (id) VALUES (100)")
        .unwrap();
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&checkpoint));
    let _ = std::fs::remove_file(&control);
    let _ = std::fs::remove_file(&checkpoint);
    let _ = std::fs::remove_file(&path);
}

/// W1b — the SECOND rotation's crash window: after one successful rotation, the live segment
/// holds only the suffix; a second rotation writes a NEW full-history checkpoint, and a crash
/// before its truncation leaves live = the suffix = the new checkpoint's TAIL (not its head —
/// head-to-head prefix matching would find no overlap and double-replay the suffix).
#[test]
fn w1b_auto_open_repairs_a_second_rotation_crash_window() {
    let path = test_wal_path("w1b-crash-window-2");
    let control = gpu_db_wal::wal_checkpoint_control_path(&path);
    let checkpoint = gpu_db_wal::wal_checkpoint_segment_path(&path);
    {
        // Pinned SERIAL (audit E2.5c-3 F1; see the first crash-window test).
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
        for i in 0..4 {
            e.execute_text(2 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, 1)"))
                .unwrap();
        }
        // Rotation 1 completes normally: live = suffix only.
        e.checkpoint_and_truncate_durable_wal_if_larger_than(&control, &checkpoint, 1)
            .unwrap();
        for i in 4..7 {
            e.execute_text(2 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, 2)"))
                .unwrap();
        }
        // Rotation 2 CRASHES between the control-file write and the truncation: write the new
        // full-history checkpoint + control exactly as the rotation does, skip the truncate.
        let records = e.durable_wal_records();
        gpu_db_wal::write_wal_segment(&checkpoint, &records).unwrap();
        gpu_db_wal::write_wal_control_file(
            &control,
            &gpu_db_wal::WalControlFile {
                segment_path: checkpoint
                    .file_name()
                    .map(std::path::PathBuf::from)
                    .unwrap(),
                checkpoint: gpu_db_wal::WalCheckpointMeta {
                    durable_record_count: records.len(),
                    last_durable_txn_id: records.last().map(|r| r.txn_id),
                },
            },
        )
        .unwrap();
    }
    let recovered = Engine::open_durable_wal_segment_auto(&path).unwrap();
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
        unreachable!()
    };
    let rows = recovered.execute_relational_select(&count).unwrap().rows;
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int8(7)]],
        "the second rotation's overlap (the live suffix = the checkpoint's tail) must replay once"
    );
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&checkpoint));
    let _ = std::fs::remove_file(&control);
    let _ = std::fs::remove_file(&checkpoint);
    let _ = std::fs::remove_file(&path);
}

/// W5a — binary WAL records: a covered (elided, constraint-free) concurrent INSERT logs the
/// RESOLVED binary record; restart replay decodes+installs it with the ORIGINAL row ids, and the
/// recovered state is identical to what SQL-text replay produces.
#[test]
fn w5a_binary_wal_records_replay_identically_to_text() {
    let run = |binary: bool, tag: &str| -> Vec<Vec<SqlValue>> {
        let path = test_wal_path(&format!("w5a-replay-{tag}"));
        {
            let e = Engine::with_durable_wal_segment(&path);
            // This fixture manually forces the pre-admission elided state to exercise WAL
            // encoding. Keep S-F out of that setup; otherwise CREATE auto-admits an empty device
            // generation before the synthetic `set_table_device_authoritative` transition.
            e.set_auto_admit_on_commit(false);
            e.set_binary_wal_records_enabled(binary);
            e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
            // Force the covered class deterministically (no GPU needed for the WAL semantics
            // under test): an elided table's prepare skips the value-index, which is exactly
            // reresolve_reuse_eligible / the binary-record condition.
            e.set_table_device_authoritative("t", true);
            for i in 0..6 {
                e.execute_dml_concurrent(
                    2 + i,
                    &format!("INSERT INTO t (id, v) VALUES ({i}, {})", i * 10),
                )
                .unwrap();
            }
        }
        let recovered = Engine::open_durable_wal_segment_auto(&path).unwrap();
        let Command::Select(select) = parse_command("SELECT id, v FROM t ORDER BY id").unwrap()
        else {
            unreachable!()
        };
        let rows: Vec<Vec<SqlValue>> = recovered
            .execute_relational_select(&select)
            .unwrap()
            .rows
            .into_boxed();
        let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
        let _ = std::fs::remove_file(&path);
        rows
    };
    let text_rows = run(false, "text");
    let binary_rows = run(true, "binary");
    assert_eq!(text_rows.len(), 6);
    assert_eq!(
        text_rows, binary_rows,
        "binary replay must produce byte-identical state to text replay"
    );
}

/// W5a — binary records survive a checkpoint rotation (they live inside the checkpoint segment)
/// and interleave correctly with text records (DDL + serialized inserts) across replay: the
/// row-id allocator stays in lock-step because binary applies advance it by rows_consumed.
#[test]
fn w5a_binary_records_interleave_with_text_and_survive_rotation() {
    let path = test_wal_path("w5a-rotation-mix");
    let control = gpu_db_wal::wal_checkpoint_control_path(&path);
    let checkpoint = gpu_db_wal::wal_checkpoint_segment_path(&path);
    {
        let e = Engine::with_durable_wal_segment(&path);
        e.set_binary_wal_records_enabled(true);
        e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
        // Text-record inserts (serialized path) BEFORE elision.
        e.execute_text(2, "INSERT INTO t (id, v) VALUES (100, 1)")
            .unwrap();
        e.set_table_device_authoritative("t", true);
        // Binary-record inserts (covered concurrent path).
        for i in 0..4 {
            e.execute_dml_concurrent(3 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, 2)"))
                .unwrap();
        }
        // Rotate: the binary records move into the checkpoint segment.
        e.checkpoint_and_truncate_durable_wal_if_larger_than(&control, &checkpoint, 1)
            .unwrap();
        // More binary records into the fresh live suffix.
        for i in 4..7 {
            e.execute_dml_concurrent(3 + i, &format!("INSERT INTO t (id, v) VALUES ({i}, 3)"))
                .unwrap();
        }
    }
    let recovered = Engine::open_durable_wal_segment_auto(&path).unwrap();
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
        unreachable!()
    };
    let rows = recovered.execute_relational_select(&count).unwrap().rows;
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int8(8)]],
        "1 text + 7 binary inserts must all replay exactly once through the rotation"
    );
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&checkpoint));
    let _ = std::fs::remove_file(&control);
    let _ = std::fs::remove_file(&checkpoint);
    let _ = std::fs::remove_file(&path);
}

/// E2.5c-1 — LANES-MODE REOPEN: the on-disk lane logs (fabricated exactly as an activated 2-lane
/// engine writes them: WalRecords with binary insert payloads at LANE-LOCAL global seqs tiling
/// [0, N)) replay AFTER the serial log's pre-activation prefix; the reopened engine is ACTIVATED
/// (classic writes refused fail-loud — the v1 intent-only contract survives reopen), its lane
/// cut/oracle continue from the recovered history, and a second reopen is idempotent.
#[test]
fn lanes_reopen_replays_serial_then_lane_merge_and_guards_classic_writes() {
    let path = test_wal_path("lanes-reopen");
    // Serial pre-activation history (pinned to the serial backend, env-proof): DDL + 2 rows.
    let row_base = {
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO t VALUES (1, 10)").unwrap();
        e.execute_text(3, "INSERT INTO t VALUES (2, 20)").unwrap();
        // Lane records must continue the row-id allocation exactly where the serial history
        // left it (the real lane pump claims blocks from this same allocator).
        e.read_state.mvcc.current_row_id()
    };
    // Fabricate the lane logs beside the serial segment: 4 covered-insert records across 2
    // lanes, at LANE-LOCAL global seqs tiling [0, 4).
    {
        let set = gpu_db_wal::FuaWalLaneSet::create(&path, 2, 2, 1 << 20).expect("create lanes");
        for (seq, (id, v)) in [(100, 1000), (101, 1010), (102, 1020), (103, 1030)]
            .iter()
            .enumerate()
        {
            let values = vec![SqlValue::Int4(*id), SqlValue::Int4(*v)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(row_base + seq as u64, values.as_slice())],
            )
            .expect("binary encode");
            let record = canonical_test_lane_record(
                &path,
                (seq % 2) as u32,
                seq as u64,
                100 + seq as u64,
                payload,
            );
            set.append(seq % 2, seq as u64, &[record]).expect("append");
        }
        set.wait_durable(4).expect("lane records durable");
    }
    let select_all = |engine: &Engine| -> Vec<Vec<SqlValue>> {
        let Command::Select(select) = parse_command("SELECT id, v FROM t ORDER BY id").unwrap()
        else {
            unreachable!()
        };
        engine
            .execute_relational_select(&select)
            .unwrap()
            .rows
            .into_boxed()
    };
    let reopened = Engine::open_durable_wal_segment(&path).expect("lanes reopen must succeed");
    let rows = select_all(&reopened);
    assert_eq!(
        rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
            vec![SqlValue::Int4(100), SqlValue::Int4(1000)],
            vec![SqlValue::Int4(101), SqlValue::Int4(1010)],
            vec![SqlValue::Int4(102), SqlValue::Int4(1020)],
            vec![SqlValue::Int4(103), SqlValue::Int4(1030)],
        ],
        "serial prefix then lane merge, in global seq order"
    );
    // The reopened lane state continues the recovered history: durable/applied cuts at 4
    // (lane-local), so the reopened set's next claims can never collide with recovered seqs.
    let stats = reopened.intent_lane_stats().expect("lanes state installed");
    assert_eq!(stats.7, 4, "lane durable cut continues at the merge end");
    assert_eq!(stats.8, 4, "applied cut pre-seeded to the merge end");
    // v1 contract survives reopen: classic DML is refused fail-loud.
    let err = reopened
        .execute_text(9, "INSERT INTO t VALUES (7, 70)")
        .expect_err("classic write after lanes activation must be refused");
    assert!(
        err.to_string().contains("intent lanes are ACTIVE"),
        "expected the intent-only refusal, got: {err}"
    );
    drop(reopened);
    // Idempotent second reopen (nothing new was committed).
    let again = Engine::open_durable_wal_segment(&path).expect("second lanes reopen");
    assert_eq!(select_all(&again), rows, "second reopen is idempotent");
    drop(again);
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
    cleanup_lane_files(&path);
}

/// E2.5c-1 — CRASH-MID-WAVE fault injection: one lane's frame never became durable (a GAP in the
/// global seq space) while other lanes' later frames did (durable ORPHANS above the cut — never
/// acknowledgeable, since acks gate on cut coverage). Reopen must replay exactly the contiguous
/// prefix, durably DISCARD the orphans (they would collide with fresh claims of the same seqs),
/// and be idempotent across a second reopen.
#[test]
fn lanes_reopen_discards_unacked_orphans_above_the_cut() {
    let path = test_wal_path("lanes-orphans");
    let row_base = {
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO t VALUES (1, 10)").unwrap();
        e.read_state.mvcc.current_row_id()
    };
    // Lane 0 holds seqs [0, 3); lane 1 holds seqs [5, 8). Seqs 3..5 belonged to a wave whose
    // frame CRASHED before its FUA fence — the exact on-disk shape of a crash mid-wave.
    {
        let set = gpu_db_wal::FuaWalLaneSet::create(&path, 2, 2, 1 << 20).expect("create lanes");
        let make_record = |seq: u64| {
            let values = vec![SqlValue::Int4(1000 + seq as i32), SqlValue::Int4(0)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(row_base + seq, values.as_slice())],
            )
            .expect("binary encode");
            canonical_test_lane_record(&path, 0, seq, 200 + seq, payload)
        };
        for seq in 0..3u64 {
            set.append(0, seq, &[make_record(seq)]).expect("lane 0");
        }
        for seq in 5..8u64 {
            let values = vec![SqlValue::Int4(1000 + seq as i32), SqlValue::Int4(0)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(row_base + seq, values.as_slice())],
            )
            .expect("binary encode");
            let record = canonical_test_lane_record(&path, 1, seq, 200 + seq, payload);
            set.append(1, seq, &[record]).expect("lane 1");
        }
        // Cut holds at the gap: only [0, 3) is contiguous.
        set.wait_durable(3).expect("contiguous prefix durable");
        // Give the orphan frames time to fence too (they must be ON DISK for the repair to
        // have something to discard; the drop drains the pools either way).
    }
    // Premise: recovery sees the orphan shape (3 contiguous records, 3 orphans above the gap).
    assert_eq!(
        gpu_db_wal::recover_lanes(&path, 2).expect("recover").len(),
        3,
        "premise: the contiguous prefix ends at the gap"
    );
    let reopened = Engine::open_durable_wal_segment(&path).expect("orphaned lanes reopen");
    let reconciled = gpu_db_wal::read_reconciled_transaction_statuses(&path)
        .expect("orphan repair persists stable-ID reconciliation first");
    assert_eq!(
        reconciled
            .iter()
            .map(|status| status.txn_id)
            .collect::<Vec<_>>(),
        vec![205, 206, 207],
        "every complete transaction above the gap remains durably memoized as aborted"
    );
    for txn_id in 205..=207 {
        assert!(matches!(
            reopened
                .commit_state()
                .transaction_status
                .get(&txn_id)
                .map(|status| status.outcome),
            Some(DurableTransactionOutcome::AbortedDiscardedOrphan)
        ));
    }
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
        unreachable!()
    };
    let counted = |engine: &Engine| -> Vec<Vec<SqlValue>> {
        engine
            .execute_relational_select(&count)
            .unwrap()
            .rows
            .into_boxed()
    };
    assert_eq!(
        counted(&reopened),
        vec![vec![SqlValue::Int8(4)]],
        "1 serial row + exactly the 3 contiguous lane rows (orphans discarded, never acked)"
    );
    drop(reopened);
    // The repair was DURABLE and PHYSICAL (audit non-vacuity pin): a raw wal-level reopen
    // FAILS CLOSED whenever any durable frame survives above the cut, so its success here
    // proves the orphan frames are gone from disk — recover_lanes alone truncates at the gap
    // and would pass even against a no-op repair.
    drop(
        gpu_db_wal::FuaWalLaneSet::reopen(&path, 2, 2, 1 << 20)
            .expect("wal-level reopen must succeed: orphans were physically discarded"),
    );
    assert_eq!(
        gpu_db_wal::recover_lanes(&path, 2)
            .expect("recover repaired")
            .len(),
        3,
        "orphan frames were durably discarded"
    );
    let status_path = gpu_db_wal::reconciled_status_path(&path);
    let status_body = std::fs::read_to_string(&status_path).expect("reconciliation sidecar");
    std::fs::write(&status_path, status_body.replace("txn=205", "txn=999"))
        .expect("tamper reconciliation sidecar");
    assert!(
        Engine::open_durable_wal_segment(&path).is_err(),
        "tampered stable-ID reconciliation authority must fail closed"
    );
    std::fs::write(&status_path, status_body).expect("restore reconciliation sidecar");
    let again = Engine::open_durable_wal_segment(&path).expect("second reopen after repair");
    assert_eq!(counted(&again), vec![vec![SqlValue::Int8(4)]]);
    for txn_id in 205..=207 {
        let payload = format!("different request for discarded {txn_id}");
        let error = again
            .commit_state()
            .resolve_transaction_retry(txn_id, payload.as_bytes())
            .expect_err("an abandoned stable ID cannot be reused with another request");
        assert!(error.to_string().contains("different request"), "{error}");
    }
    drop(again);
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(status_path);
    cleanup_lane_files(&path);
}

/// Remove every `<base>.lane-<L>.fua.*` file a lanes test left beside its base path.
fn cleanup_lane_files(path: &std::path::Path) {
    let (Some(parent), Some(stem)) = (path.parent(), path.file_name()) else {
        return;
    };
    let prefix = format!("{}.lane-", stem.to_string_lossy());
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

/// E2.5c-2 — LANES CHECKPOINT + TRUNCATION arc: checkpoint an activated lanes database
/// (checkpoint segment embeds serial ++ lane merge; sidecar records the split + baseline),
/// verify rolled-away lane segments are PHYSICALLY pruned, then fabricate a post-checkpoint
/// lane suffix and prove the auto-open replays checkpoint-then-suffix with row parity.
#[test]
fn lanes_checkpoint_truncates_prunes_and_reopens_with_suffix() {
    let path = test_wal_path("lanes-ckpt");
    let row_base = {
        let e = serial_durable_engine(&path);
        e.execute_text(1, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO t VALUES (1, 10)").unwrap();
        e.read_state.mvcc.current_row_id()
    };
    // Fabricate a 2-lane history with TINY segments (16KiB = 3 one-record frames per segment)
    // so 24 records roll through ~4 segments per lane — the truncation premise.
    let tiny = 16 << 10;
    let make_record = |seq: u64, id: i32| {
        let values = vec![SqlValue::Int4(id), SqlValue::Int4(0)];
        let payload = crate::wal_binary::try_encode_binary_insert(
            "t",
            &[(row_base + seq, values.as_slice())],
        )
        .expect("binary encode");
        canonical_test_lane_record(&path, (seq % 2) as u32, seq, 300 + seq, payload)
    };
    {
        let set = gpu_db_wal::FuaWalLaneSet::create(&path, 2, 2, tiny).expect("create lanes");
        for seq in 0..24u64 {
            set.append(
                (seq % 2) as usize,
                seq,
                &[make_record(seq, 1000 + seq as i32)],
            )
            .expect("append");
        }
        set.wait_durable(24).expect("durable");
    }
    let lane_file_count = || {
        let (parent, stem) = (path.parent().unwrap(), path.file_name().unwrap());
        let prefix = format!("{}.lane-", stem.to_string_lossy());
        std::fs::read_dir(parent)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
            .count()
    };
    let before = lane_file_count();
    assert!(
        before >= 8,
        "premise: rolls produced many segments ({before})"
    );

    let counted = |engine: &Engine| -> i64 {
        let Command::Select(count) = parse_command("SELECT COUNT(*) FROM t").unwrap() else {
            unreachable!()
        };
        match engine
            .execute_relational_select(&count)
            .unwrap()
            .rows
            .row(0)[0]
        {
            SqlValue::Int8(n) => n,
            ref other => panic!("count returned {other:?}"),
        }
    };
    {
        let reopened = Engine::open_durable_wal_segment(&path).expect("lanes reopen");
        assert_eq!(counted(&reopened), 25, "1 serial + 24 lane rows");
        let cut = reopened
            .checkpoint_intent_lanes()
            .expect("lanes checkpoint");
        assert_eq!(cut, 24, "baseline = the cross-lane durable cut");
        let after = lane_file_count();
        assert!(
            after < before,
            "rolled-away lane segments must be pruned: {before} -> {after}"
        );
        // Idempotent re-checkpoint at the same cut (early-returns; the committed
        // checkpoint is untouched).
        assert_eq!(
            reopened.checkpoint_intent_lanes().expect("re-checkpoint"),
            24
        );
    }
    // Post-checkpoint lane SUFFIX (fabricated continuation commits above the baseline).
    {
        let set = gpu_db_wal::FuaWalLaneSet::reopen_from(&path, 2, 2, tiny, 24)
            .expect("wal-level reopen from baseline");
        for seq in 24..30u64 {
            set.append(
                (seq % 2) as usize,
                seq,
                &[make_record(seq, 2000 + seq as i32)],
            )
            .expect("append suffix");
        }
        set.wait_durable(30).expect("suffix durable");
    }
    // AUTO open (exercises the lanes routing past the serial checkpoint-open): replays the
    // checkpoint, then the lane suffix; the reopened engine is intent-only.
    let again = Engine::open_durable_wal_segment_auto(&path).expect("auto reopen");
    assert_eq!(
        counted(&again),
        31,
        "1 serial + 24 checkpointed + 6 suffix rows"
    );
    let err = again
        .execute_text(9, "INSERT INTO t VALUES (7, 70)")
        .expect_err("classic write refused after checkpointed reopen");
    assert!(err.to_string().contains("intent lanes are ACTIVE"), "{err}");
    let stats = again.intent_lane_stats().expect("lanes installed");
    assert_eq!(stats.7, 30, "durable cut continues above the checkpoint");
    assert_eq!(stats.8, 30, "applied cut continues above the checkpoint");
    drop(again);
    // AUDIT (repeat-checkpoint crash window): a NEW generation segment written but whose
    // sidecar commit never happened must be IGNORED — the sidecar is the single commit point,
    // so reopen replays the OLD committed checkpoint + the lane suffix, identically. The
    // orphan's content is DIVERGENT garbage (audit nit): a wrong design that read the highest
    // generation instead of the sidecar-named file would fail loudly, not coincidentally pass.
    std::fs::write(
        gpu_db_wal::lanes_checkpoint_segment_path(&path, 30),
        b"torn uncommitted checkpoint generation",
    )
    .expect("plant an uncommitted next-generation segment");
    let after_crash_window = Engine::open_durable_wal_segment_auto(&path)
        .expect("reopen must ignore an uncommitted checkpoint generation");
    assert_eq!(counted(&after_crash_window), 31, "state unchanged");
    drop(after_crash_window);
    let _ = std::fs::remove_file(gpu_db_wal::lanes_checkpoint_segment_path(&path, 30));
    let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
    let _ = std::fs::remove_file(gpu_db_wal::lanes_checkpoint_sidecar_path(&path));
    let _ = std::fs::remove_file(gpu_db_wal::lanes_checkpoint_segment_path(&path, 24));
    let _ = std::fs::remove_file(&path);
    cleanup_lane_files(&path);
}
