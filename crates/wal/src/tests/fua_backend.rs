// ---- E1 step 1: FUA fence-pool durability backend -----------------------------------------

#[cfg(unix)]
fn fua_test_base(name: &str) -> PathBuf {
    // A per-test DIRECTORY so the `<base>.fua.<id>` segment files don't collide, and cleanup
    // can drop the whole dir.
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-fua-{name}-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create fua test dir");
    dir.join("wal.segment")
}

#[cfg(unix)]
fn rec(txn_id: TxnId, payload: &[u8]) -> WalRecord {
    WalRecord {
        txn_id,
        payload: payload.to_vec().into(),
    }
}

/// (a) Roundtrip + recovery parity: the same logical records recover BYTE-IDENTICALLY through
/// the FUA backend's frame log and through the serial segment reader.
#[cfg(unix)]
#[test]
fn fua_roundtrip_recovers_identically_to_serial_path() {
    let base = fua_test_base("roundtrip");
    let records = vec![
        rec(1, b"CREATE TABLE t (id INT)"),
        rec(2, b"INSERT INTO t (id) VALUES (1)"),
        rec(3, &vec![0xABu8; 9000]), // multi-4KiB payload, exercises frame padding
        rec(4, b"UPDATE t SET id = 2 WHERE id = 1"),
    ];

    // FUA path: append + group-flush all records as one group, then recover from disk.
    {
        let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("fua create");
        assert!(wal.is_durable());
        assert_eq!(wal.durable_segment_path(), Some(base.as_path()));
        for record in &records {
            wal.append(record.clone());
        }
        wal.flush_all().expect("fua flush");
        assert_eq!(wal.flushed_count(), records.len());
        assert_eq!(wal.unflushed_count(), 0);
    }
    let fua_recovered = recover_fua_wal_records(&base).expect("fua recover");

    // Serial path: the same records written to a plain segment and read back.
    let serial_path = base.with_file_name("serial.segment");
    write_wal_segment(&serial_path, &records).expect("serial write");
    let serial_recovered = read_wal_segment(&serial_path).expect("serial read");

    assert_eq!(fua_recovered, records, "fua recovery must match input");
    assert_eq!(
        fua_recovered, serial_recovered,
        "fua and serial recovery must be identical"
    );

    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// Controller decisions are owned by synchronous canonical group commit.  The small pool makes
/// the first decision visibly unfragmented, proving this route is controller-enabled without
/// turning the legacy lane primitive into a second asynchronous feedback system.
#[cfg(unix)]
#[test]
fn fua_canonical_group_commit_owns_controller_decision() {
    let base = fua_test_base("controller-owner");
    let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("fua create");
    wal.append(rec(1, b"controller-owned-group"));
    wal.flush_all().expect("fua flush");

    let telemetry = wal.fua_durability_telemetry();
    assert_eq!(telemetry.logical_groups, 1);
    assert_eq!(telemetry.controller_unfragmented_actions, 1);
    assert_eq!(telemetry.controller_pool_too_narrow, 1);
    assert_eq!(telemetry.controller_sustained_actions, 0);
    assert_eq!(telemetry.controller_qd1_samples, 0);
    assert_eq!(telemetry.controller_pending_qd1_samples, 0);
    assert_eq!(telemetry.controller_action_reconciliation, 1);
    assert_eq!(telemetry.controller_sample_reconciliation, 1);

    drop(wal);
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// The FUA path uses the same incremental lineage proof, but its protected boundary is frame
/// publication rather than serial `io_in_flight`: a substituted or missing sidecar must reject
/// before either `set_published` or the durable cut can move.
#[cfg(unix)]
#[test]
fn fua_identity_binding_rejects_substitution_and_anchor_loss_before_publish() {
    let base = fua_test_base("identity-incremental");
    let identity = identity_test_value(34);
    let foreign = identity_test_value(37);
    let mut wal =
        WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("fua durable create");
    wal.append(canonical_identity_test_record(1, identity));
    wal.flush_all().expect("first FUA flush");
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 1);
    assert_eq!(wal.fua_published_records_for_test(), Some(1));

    write_durable_identity(&base, foreign).expect("install foreign anchor");
    wal.append(canonical_identity_test_record(2, identity));
    let error = match wal.begin_group_flush() {
        Ok(_) => panic!("foreign anchor must reject before FUA ticket/publication"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("another database/timeline"));
    assert_eq!(
        wal.flushed_count(),
        1,
        "identity failure cannot advance durable"
    );
    assert_eq!(
        wal.fua_published_records_for_test(),
        Some(1),
        "identity failure cannot advance the FUA published cursor"
    );
    assert_eq!(wal.durable_identity_verified_records_for_test(), 1);

    write_durable_identity(&base, identity).expect("restore expected anchor");
    wal.flush_all().expect("retry after anchor restore");
    assert_eq!(wal.flushed_count(), 2);
    assert_eq!(wal.fua_published_records_for_test(), Some(2));
    assert_eq!(
        wal.durable_identity_decoded_records_for_test(),
        3,
        "the failed sidecar check leaves the second record unverified"
    );

    for txn_id in 3..=5 {
        wal.append(canonical_identity_test_record(txn_id, identity));
        wal.flush_all().expect("subsequent FUA flush");
        assert_eq!(
            wal.durable_identity_decoded_records_for_test(),
            txn_id as usize + 1,
            "each FUA flush after recovery scans only its new record"
        );
    }
    assert_eq!(wal.flushed_count(), 5);
    assert_eq!(wal.fua_published_records_for_test(), Some(5));

    std::fs::remove_file(durable_identity_path(&base)).expect("remove identity anchor");
    wal.append(canonical_identity_test_record(6, identity));
    let error = match wal.begin_group_flush() {
        Ok(_) => panic!("missing anchor must reject before FUA ticket/publication"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("anchor is missing"));
    assert_eq!(wal.flushed_count(), 5, "anchor loss cannot advance durable");
    assert_eq!(
        wal.fua_published_records_for_test(),
        Some(5),
        "anchor loss cannot hand another frame to FUA"
    );
    assert_eq!(wal.durable_identity_verified_records_for_test(), 5);
    assert_eq!(recover_fua_wal_records(&base).unwrap().len(), 5);
    drop(wal);
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// (b) The durable watermark advances ONLY over the contiguous durable cut and is monotonic,
/// even with MULTIPLE flush jobs in flight completing out of order across the fence pool.
#[cfg(unix)]
#[test]
fn fua_watermark_is_monotonic_under_concurrent_in_flight_jobs() {
    use std::sync::Arc as StdArc;
    use std::sync::Mutex as StdMutex;

    let base = fua_test_base("monotonic");
    let wal = StdArc::new(StdMutex::new(
        WalBuffer::with_fua_durable_segment(&base, 32, 4 << 20).expect("fua create"),
    ));

    // Producer: append records and start group flushes concurrently. Each begin snapshots a
    // disjoint record range under the outer lock; commits run lock-free and may finish out of
    // order, so a watching thread must never see the watermark go backwards or exceed the
    // published count.
    let groups = 40usize;
    let per_group = 5usize;
    let total = groups * per_group;
    // A watcher samples the durable watermark under the outer lock throughout the run and
    // asserts it NEVER decreases (advances only over the contiguous cut) and never overshoots
    // the records appended so far. This is the time-domain monotonicity property; the
    // per-commit return values below are the value-domain property.
    let done = StdArc::new(std::sync::atomic::AtomicBool::new(false));
    let watcher = {
        let wal = StdArc::clone(&wal);
        let done = StdArc::clone(&done);
        std::thread::spawn(move || {
            let mut last = 0usize;
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                let watermark = wal.lock().unwrap().flushed_count();
                assert!(
                    watermark >= last,
                    "watermark regressed: {watermark} < {last}"
                );
                assert!(
                    watermark <= total,
                    "watermark {watermark} exceeds published {total}"
                );
                last = watermark;
                std::thread::yield_now();
            }
        })
    };

    // (job_target, join handle): each commit must return a watermark that already covers its
    // own group (the durable cut reached at least its target) and never exceeds the total.
    let mut handles = Vec::new();
    let mut next_txn = 1u64;
    for group in 0..groups {
        let target = (group + 1) * per_group;
        let begun = {
            let mut guard = wal.lock().unwrap();
            for _ in 0..per_group {
                guard.append(rec(next_txn, format!("op-{next_txn}").as_bytes()));
                next_txn += 1;
            }
            guard.begin_group_flush().expect("begin")
        };
        match begun {
            WalGroupFlushBegin::Clean { .. } => {}
            WalGroupFlushBegin::Job(job) => {
                handles.push((
                    target,
                    std::thread::spawn(move || job.commit().expect("commit")),
                ));
            }
        }
    }

    for (target, handle) in handles {
        let watermark = handle.join().expect("join");
        assert!(
            watermark >= target,
            "commit returned watermark {watermark} below its own group target {target}"
        );
        assert!(
            watermark <= total,
            "watermark {watermark} exceeds published {total}"
        );
    }
    done.store(true, std::sync::atomic::Ordering::Release);
    watcher.join().expect("watcher");

    {
        let mut guard = wal.lock().unwrap();
        guard.flush_all().expect("final flush");
        assert_eq!(guard.flushed_count(), total);
    }
    // Everything recovers, in order.
    let recovered = recover_fua_wal_records(&base).expect("recover");
    assert_eq!(recovered.len(), total);
    for (index, record) in recovered.iter().enumerate() {
        assert_eq!(record.txn_id, index as u64 + 1);
    }
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// (c) Segment roll mid-stream: a small per-segment capacity forces several rolls; the
/// totally-ordered record history recovers contiguously ACROSS the rolled segment files.
#[cfg(unix)]
#[test]
fn fua_segment_roll_recovers_across_segments() {
    let base = fua_test_base("roll");
    let payload = vec![0x5Au8; 2000]; // ~3 frames fit a 12KiB-ish segment before StorageFull
    let total = 60usize;
    {
        // 16KiB data capacity per segment: each ~2KB record pads to 4KiB, so ~4 frames/segment
        // -> many rolls over 60 records.
        let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 16 * 1024).expect("fua create");
        for txn in 1..=total as u64 {
            wal.append(rec(txn, &payload));
            // Flush each record individually so every group is its own frame — maximizes rolls.
            wal.flush_all().expect("flush");
            assert_eq!(wal.flushed_count(), txn as usize);
        }
        let telemetry = wal.fua_durability_telemetry();
        assert_eq!(telemetry.configured_fence_lanes, 8);
        assert_eq!(telemetry.logical_groups, total as u64);
        assert_eq!(telemetry.published_frames, total as u64);
        assert_eq!(telemetry.fenced_frames, total as u64);
        assert_eq!(telemetry.fence_failures, 0);
        assert_eq!(telemetry.stage_copy_frames, total as u64);
        assert_eq!(telemetry.publish_to_claim_frames, total as u64);
        assert_eq!(telemetry.claim_to_write_done_frames, total as u64);
        assert_eq!(telemetry.write_done_to_contiguous_cut_frames, total as u64);
        assert_eq!(telemetry.contiguous_cut_advanced_frames, total as u64);
        assert_eq!(
            telemetry.in_flight_depth_histogram.iter().sum::<u64>(),
            total as u64
        );
        assert_eq!(
            telemetry.logical_payload_bytes, telemetry.payload_bytes,
            "no fragmentation means logical and physical payload bytes match"
        );
        assert_eq!(
            telemetry.single_frame_padded_baseline_bytes, telemetry.padded_bytes,
            "no fragmentation means one-frame baseline and actual padding match"
        );
        assert!(telemetry.waiter_cut_to_observe_count >= total as u64);
    }
    // More than one segment file must exist (a roll happened).
    let dir = base.parent().unwrap();
    let segment_count = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("wal.segment.fua."))
                .unwrap_or(false)
        })
        .count();
    assert!(
        segment_count > 1,
        "expected multiple rolled segments, found {segment_count}"
    );

    let recovered = recover_fua_wal_records(&base).expect("recover across segments");
    assert_eq!(recovered.len(), total);
    for (index, record) in recovered.iter().enumerate() {
        assert_eq!(record.txn_id, index as u64 + 1);
        assert_eq!(&record.payload[..], &payload[..]);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The FUA backend surfaces its config errors and the step-1 unsupported ops fail-closed.
#[cfg(unix)]
#[test]
fn fua_config_and_unsupported_ops_error() {
    let base = fua_test_base("config");
    assert!(WalBuffer::with_fua_durable_segment(&base, 8, 0).is_err());

    let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("create");
    wal.append(rec(1, b"x"));
    wal.flush_all().expect("flush");
    // Prefix truncation is a step-2 capability; it must error rather than silently no-op.
    assert!(wal.truncate_durable_segment_prefix(0).is_err());
    assert_eq!(wal.durable_segment_base_records(), 0);
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// E1 step 3: REOPEN an existing FUA log — write + drop, reopen and verify the recovered
/// history + that appends CONTINUE the contiguous log (new segment above the old id), then
/// drop + recover a THIRD time to prove both segments chain end-to-end.
#[cfg(unix)]
#[test]
fn fua_reopen_continues_the_log_and_recovers_across_lives() {
    let base = fua_test_base("reopen");
    let first = vec![
        rec(1, b"CREATE TABLE t (id INT)"),
        rec(2, b"INSERT INTO t (id) VALUES (1)"),
        rec(3, b"INSERT INTO t (id) VALUES (2)"),
    ];
    // Life 1: write + flush + clean drop (drain).
    {
        let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("create");
        for record in &first {
            wal.append(record.clone());
        }
        wal.flush_all().expect("flush life 1");
    }
    let recovered_1 = recover_fua_wal_records(&base).expect("recover life 1");
    assert_eq!(recovered_1, first, "life-1 recovery matches input");

    // Life 2: reopen seeded with the recovered history, verify watermark, then APPEND more.
    let second = vec![
        rec(4, b"UPDATE t SET id = 3 WHERE id = 1"),
        rec(5, b"DELETE FROM t WHERE id = 2"),
    ];
    {
        let mut wal =
            WalBuffer::with_recovered_fua_durable_segment(&base, recovered_1.clone(), 8, 1 << 20)
                .expect("reopen");
        assert!(wal.is_durable());
        assert_eq!(
            wal.flushed_count(),
            first.len(),
            "reopen reports the recovered records as already durable"
        );
        assert_eq!(wal.unflushed_count(), 0, "nothing unflushed on reopen");
        assert_eq!(wal.durable_segment_path(), Some(base.as_path()));
        for record in &second {
            wal.append(record.clone());
        }
        assert_eq!(wal.unflushed_count(), second.len());
        wal.flush_all().expect("flush life 2");
        assert_eq!(wal.flushed_count(), first.len() + second.len());
    }

    // Life 3: recover across BOTH segments — the chain must be first ++ second, in order.
    let mut expected = first.clone();
    expected.extend(second.clone());
    let recovered_2 = recover_fua_wal_records(&base).expect("recover life 2");
    assert_eq!(
        recovered_2, expected,
        "recovery chains the old and new segments contiguously"
    );
    assert!(
        fua_wal_segments_exist(&base),
        "segments are retained for recovery"
    );
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

#[cfg(unix)]
#[test]
fn raw_recovered_fua_canonical_history_requires_anchor_before_new_segment() {
    let base = fua_test_base("identity-recovered-raw-fua");
    let identity = identity_test_value(41);
    {
        let mut wal =
            WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("create FUA WAL");
        wal.append(canonical_identity_test_record(1, identity));
        wal.flush_all().expect("initial canonical FUA flush");
    }
    let recovered = recover_fua_wal_records(&base).expect("recover canonical FUA history");
    std::fs::remove_file(durable_identity_path(&base)).expect("remove identity anchor");
    let mut files_before = std::fs::read_dir(base.parent().unwrap())
        .expect("list FUA directory")
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    files_before.sort();

    let error = WalBuffer::with_recovered_fua_durable_segment(&base, recovered, 8, 1 << 20)
        .expect_err("raw canonical FUA recovery must require its anchor before creating a segment");
    assert!(error.to_string().contains("anchor is missing"));
    let mut files_after = std::fs::read_dir(base.parent().unwrap())
        .expect("relist FUA directory")
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    files_after.sort();
    assert_eq!(files_after, files_before);
    assert_eq!(read_durable_identity(&base).unwrap(), None);
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// E1 torn-tail crash safety: a crash that leaves the LAST frame's payload corrupt must recover
/// to the durable cut BEFORE the torn frame (the torn commit was never acknowledged), and a
/// reopen must chain new appends contiguously above that cut WITHOUT ever resurrecting the torn
/// frame.
#[cfg(unix)]
#[test]
fn fua_torn_tail_recovers_to_cut_and_reopen_chains_contiguously() {
    // On-disk frame-log layout (crate `gpu_db_write_conveyor::fua_frame_log`): a 4096B file
    // header, then each frame is a 64B header + payload padded up to a 4096B block. With tiny,
    // individually-flushed records every frame is exactly one 4096B block, so frame `i`'s header
    // sits at `4096 * (i + 1)` and its payload starts 64B later.
    const FRAME_LOG_HEADER_BYTES: u64 = 4096;
    const FRAME_ALIGN: u64 = 4096;
    const FRAME_HEADER_BYTES: u64 = 64;

    let base = fua_test_base("torn_tail");
    // txn 5 is the record whose frame we tear; 1..=4 form the durable prefix.
    let pre_tear = vec![
        rec(1, b"CREATE TABLE t (id INT)"),
        rec(2, b"INSERT INTO t (id) VALUES (1)"),
        rec(3, b"INSERT INTO t (id) VALUES (2)"),
        rec(4, b"INSERT INTO t (id) VALUES (3)"),
    ];
    let torn = rec(5, b"INSERT INTO t (id) VALUES (99)");

    // Life 1: append + group-flush each record as its OWN frame (one 4096B block each) into a
    // single large segment (no roll), then clean-drop so the file is fully written and closed.
    {
        let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("create");
        for record in pre_tear.iter().chain(std::iter::once(&torn)) {
            wal.append(record.clone());
            wal.flush_all().expect("flush frame");
        }
        assert_eq!(wal.flushed_count(), pre_tear.len() + 1);
    }
    // All five frames live in segment id 1 (no roll at 1 MiB); before tearing, all recover.
    let seg1 = base.with_file_name("wal.segment.fua.1");
    assert!(seg1.is_file(), "single un-rolled segment at id 1");
    assert_eq!(
        recover_fua_wal_records(&base)
            .expect("pre-tear recover")
            .len(),
        pre_tear.len() + 1,
        "all frames recover before the tear"
    );

    // Corrupt the LAST frame's first payload byte so its payload CRC fails on scan.
    let last_frame_index = pre_tear.len() as u64; // 0-based index of txn 5's frame
    let payload_offset =
        FRAME_LOG_HEADER_BYTES + last_frame_index * FRAME_ALIGN + FRAME_HEADER_BYTES;
    {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&seg1)
            .expect("open segment for corruption");
        file.seek(SeekFrom::Start(payload_offset)).expect("seek");
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).expect("read payload byte");
        byte[0] ^= 0xFF; // flip -> payload CRC no longer matches the frame header
        file.seek(SeekFrom::Start(payload_offset))
            .expect("seek back");
        file.write_all(&byte).expect("write corrupted byte");
        file.sync_all().expect("persist corruption");
    }

    // Recovery stops at the durable cut BEFORE the torn frame: only the pre-tear prefix.
    let recovered_1 = recover_fua_wal_records(&base).expect("torn recover");
    assert_eq!(
        recovered_1, pre_tear,
        "recovery stops at the durable cut before the torn frame"
    );

    // Reopen above the recovered cut, append more records, flush, clean-drop.
    let post = vec![
        rec(6, b"UPDATE t SET id = 4 WHERE id = 1"),
        rec(7, b"DELETE FROM t WHERE id = 2"),
    ];
    {
        let mut wal =
            WalBuffer::with_recovered_fua_durable_segment(&base, recovered_1.clone(), 8, 1 << 20)
                .expect("reopen after tear");
        assert_eq!(
            wal.flushed_count(),
            pre_tear.len(),
            "reopen reports the durable cut as already durable"
        );
        for record in &post {
            wal.append(record.clone());
        }
        wal.flush_all().expect("flush after reopen");
        assert_eq!(wal.flushed_count(), pre_tear.len() + post.len());
    }

    // Re-recover across both segments: the history chains contiguously (pre-tear ++ new), with
    // no gap and the torn frame (txn 5) NEVER resurrected.
    let mut expected = pre_tear.clone();
    expected.extend(post.clone());
    let recovered_2 = recover_fua_wal_records(&base).expect("recover after reopen");
    assert_eq!(
        recovered_2, expected,
        "post-reopen history chains pre-tear ++ new records with no gap"
    );
    assert!(
        !recovered_2.iter().any(|r| r.txn_id == 5),
        "the torn frame (txn 5) is never resurrected"
    );
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// A fresh FUA create must REFUSE to clobber a plain serial WAL file sitting at `<base>` — it
/// might be a durable serial log, and it would later trip the mixed-backend reopen refusal.
#[cfg(unix)]
#[test]
fn fua_create_refuses_to_clobber_a_plain_serial_file() {
    let base = fua_test_base("serial_guard");
    // A leftover plain serial WAL file exactly at `<base>`.
    fs::write(&base, b"pretend durable serial WAL").expect("write serial file");

    let err = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20)
        .expect_err("fresh FUA create must fail when a plain serial file is present");
    let msg = err.to_string();
    assert!(
        msg.contains("plain serial WAL file") && msg.contains("remove it"),
        "error must tell the operator to remove the serial file: {msg}"
    );
    // Fail-closed: the serial file is NOT deleted, and no FUA segment was created.
    assert!(base.is_file(), "serial file must be left intact");
    assert!(
        !fua_wal_segments_exist(&base),
        "no FUA segment should be created on the refused path"
    );
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}
