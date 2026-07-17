#[test]
fn wal_archive_round_trips_ordered_segments() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
    ];

    let manifest = write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
    let (recovered_manifest, recovered_records) = read_wal_archive(&manifest_path).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(recovered_manifest, manifest);
    assert_eq!(recovered_manifest.segments.len(), 2);
    assert_eq!(
        recovered_manifest.checkpoint,
        WalCheckpointMeta {
            durable_record_count: 3,
            last_durable_txn_id: Some(3),
        }
    );
    assert_eq!(recovered_records.len(), 3);
    assert_eq!(&recovered_records[2].payload[..], &b"SET c=3"[..]);
}

#[test]
fn wal_archive_object_backup_exports_and_restores_archive() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-object-backup-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("source").join("MANIFEST");
    let segment_dir = dir.join("source").join("segments");
    let backup_path = dir.join("backup").join("BACKUP");
    let object_dir = dir.join("backup").join("objects");
    let restored_manifest_path = dir.join("restored").join("MANIFEST");
    let restored_segment_dir = dir.join("restored").join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 2_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 3,
            timestamp_micros: 3_000,
        },
    ];

    let source_manifest =
        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 2, &timestamps)
            .unwrap();
    let backup =
        export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
    let restored_manifest = restore_wal_archive_object_backup(
        &backup_path,
        &restored_manifest_path,
        &restored_segment_dir,
    )
    .unwrap();
    let (_validated_manifest, restored_records) =
        read_wal_archive(&restored_manifest_path).unwrap();
    let (_timestamp_manifest, target, timestamp_records) =
        read_wal_archive_to_timestamp_micros(&restored_manifest_path, 2_000).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(backup.archive_manifest, source_manifest);
    assert_eq!(backup.objects.len(), 3);
    assert_eq!(restored_manifest.checkpoint, source_manifest.checkpoint);
    assert_eq!(restored_manifest.record_timestamps, timestamps);
    assert_eq!(restored_records, records);
    assert_eq!(target.target_txn_id, 2);
    assert_eq!(timestamp_records.len(), 2);
}

#[test]
fn wal_archive_object_backup_rejects_corrupt_object_before_manifest_install() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-object-backup-corrupt-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("source").join("MANIFEST");
    let segment_dir = dir.join("source").join("segments");
    let backup_path = dir.join("backup").join("BACKUP");
    let object_dir = dir.join("backup").join("objects");
    let restored_manifest_path = dir.join("restored").join("MANIFEST");
    let restored_segment_dir = dir.join("restored").join("segments");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    }];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let backup =
        export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
    let segment_object = backup
        .objects
        .iter()
        .find(|object| object.source_path != Path::new("MANIFEST"))
        .unwrap();
    let object_path = resolve_manifest_path(&backup_path, &segment_object.object_path);
    fs::write(&object_path, b"corrupt wal object").unwrap();

    let err = restore_wal_archive_object_backup(
        &backup_path,
        &restored_manifest_path,
        &restored_segment_dir,
    )
    .unwrap_err();
    let manifest_installed = restored_manifest_path.exists();
    let _ = fs::remove_dir_all(dir);

    let err = err.to_string();
    assert!(
        err.contains("checksum mismatch") || (err.contains("expected") && err.contains("bytes"))
    );
    assert!(!manifest_installed);
}

#[test]
fn wal_archive_object_backup_rejects_late_corrupt_object_before_segment_install() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-object-backup-late-corrupt-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("source").join("MANIFEST");
    let segment_dir = dir.join("source").join("segments");
    let backup_path = dir.join("backup").join("BACKUP");
    let object_dir = dir.join("backup").join("objects");
    let restored_manifest_path = dir.join("restored").join("MANIFEST");
    let restored_segment_dir = dir.join("restored").join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
    ];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let backup =
        export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
    let second_segment_object = backup
        .objects
        .iter()
        .filter(|object| object.source_path != Path::new("MANIFEST"))
        .nth(1)
        .unwrap();
    let object_path = resolve_manifest_path(&backup_path, &second_segment_object.object_path);
    fs::write(&object_path, b"late corrupt wal object").unwrap();

    let err = restore_wal_archive_object_backup(
        &backup_path,
        &restored_manifest_path,
        &restored_segment_dir,
    )
    .unwrap_err();
    let manifest_installed = restored_manifest_path.exists();
    let segment_dir_installed = restored_segment_dir.exists();
    let staging_segment_dir_installed = restored_segment_dir
        .parent()
        .unwrap()
        .join(format!(
            ".{}.restore-{}",
            restored_segment_dir.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ))
        .exists();
    let _ = fs::remove_dir_all(dir);

    let err = err.to_string();
    assert!(
        err.contains("checksum mismatch") || (err.contains("expected") && err.contains("bytes"))
    );
    assert!(!manifest_installed);
    assert!(!segment_dir_installed);
    assert!(!staging_segment_dir_installed);
}

#[test]
fn wal_archive_object_backup_rejects_manifest_metadata_drift_before_install() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-object-backup-manifest-drift-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("source").join("MANIFEST");
    let segment_dir = dir.join("source").join("segments");
    let backup_path = dir.join("backup").join("BACKUP");
    let object_dir = dir.join("backup").join("objects");
    let restored_manifest_path = dir.join("restored").join("MANIFEST");
    let restored_segment_dir = dir.join("restored").join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 2_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
        .unwrap();
    export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
    let backup_body = fs::read_to_string(&backup_path).unwrap();
    fs::write(
        &backup_path,
        backup_body.replace("record_timestamp=2|2000", "record_timestamp=2|2500"),
    )
    .unwrap();

    let err = restore_wal_archive_object_backup(
        &backup_path,
        &restored_manifest_path,
        &restored_segment_dir,
    )
    .unwrap_err();
    let manifest_installed = restored_manifest_path.exists();
    let segment_dir_installed = restored_segment_dir.exists();
    let _ = fs::remove_dir_all(dir);

    let err = err.to_string();
    assert!(
        err.contains("manifest object does not match backup manifest metadata")
            || err.contains("SHA-256 checksum mismatch")
    );
    assert!(!manifest_installed);
    assert!(!segment_dir_installed);
}

#[test]
fn wal_archive_reads_prefix_to_transaction_target() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-target-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
    ];

    write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
    let (_manifest, target, recovered_records) =
        read_wal_archive_to_txn(&manifest_path, 2).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(
        target,
        WalArchiveRecoveryTarget {
            target_txn_id: 2,
            recovered_record_count: 2,
            last_recovered_txn_id: 2,
        }
    );
    assert_eq!(recovered_records.len(), 2);
    assert_eq!(&recovered_records[1].payload[..], &b"SET b=2"[..]);
}

#[test]
fn wal_archive_target_rejects_before_first_transaction() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-target-before-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![WalRecord {
        txn_id: 10,
        payload: b"SET a=1".to_vec().into(),
    }];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let err = read_wal_archive_to_txn(&manifest_path, 9).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err
        .to_string()
        .contains("before first archived transaction"));
}

#[test]
fn wal_archive_target_rejects_beyond_durable_archive() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-target-beyond-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    }];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let err = read_wal_archive_to_txn(&manifest_path, 2).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("beyond last durable transaction"));
}

#[test]
fn wal_archive_target_rejects_missing_transaction_boundary() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-target-missing-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
    ];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let err = read_wal_archive_to_txn(&manifest_path, 2).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err
        .to_string()
        .contains("does not contain target transaction"));
}

#[test]
fn wal_archive_reads_prefix_to_timestamp_target() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timestamp-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 2_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 3,
            timestamp_micros: 3_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 2, &timestamps)
        .unwrap();
    let (_manifest, target, recovered_records) =
        read_wal_archive_to_timestamp_micros(&manifest_path, 2_000).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(
        target,
        WalArchiveTimestampRecoveryTarget {
            target_timestamp_micros: 2_000,
            target_txn_id: 2,
            recovered_record_count: 2,
            last_recovered_txn_id: 2,
        }
    );
    assert_eq!(recovered_records.len(), 2);
    assert_eq!(&recovered_records[1].payload[..], &b"SET b=2"[..]);
}

#[test]
fn wal_archive_timestamp_target_rejects_missing_metadata() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timestamp-missing-meta-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    }];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let err = read_wal_archive_to_timestamp_micros(&manifest_path, 1_000).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("no timestamp metadata"));
}

#[test]
fn wal_archive_timestamp_target_rejects_unavailable_boundaries() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timestamp-unavailable-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 10,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 20,
            payload: b"SET b=2".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 10,
            timestamp_micros: 10_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 20,
            timestamp_micros: 20_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
        .unwrap();
    let before = read_wal_archive_to_timestamp_micros(&manifest_path, 9_999).unwrap_err();
    let between = read_wal_archive_to_timestamp_micros(&manifest_path, 15_000).unwrap_err();
    let beyond = read_wal_archive_to_timestamp_micros(&manifest_path, 20_001).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(before
        .to_string()
        .contains("before first archived timestamp"));
    assert!(between
        .to_string()
        .contains("falls between archived transaction boundaries"));
    assert!(beyond.to_string().contains("beyond last durable timestamp"));
}

#[test]
fn wal_archive_timestamp_target_rejects_ambiguous_boundary() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timestamp-ambiguous-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 1_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
        .unwrap();
    let err = read_wal_archive_to_timestamp_micros(&manifest_path, 1_000).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("ambiguous"));
}

#[test]
fn wal_archive_ingests_next_segment_and_preserves_timestamps() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-ingest-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let ingest_segment = segment_dir.join("segment-0002.wal");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 2_000,
        },
    ];
    let ingest_records = vec![
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
        WalRecord {
            txn_id: 4,
            payload: b"SET d=4".to_vec().into(),
        },
    ];
    let ingest_timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 3,
            timestamp_micros: 3_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 4,
            timestamp_micros: 4_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 2, &timestamps)
        .unwrap();
    write_wal_segment(&ingest_segment, &ingest_records).unwrap();
    let manifest = append_wal_archive_segment_with_timestamps(
        &manifest_path,
        &ingest_segment,
        &ingest_timestamps,
    )
    .unwrap();
    let (_read_manifest, read_records) = read_wal_archive(&manifest_path).unwrap();
    let (_manifest, target, target_records) =
        read_wal_archive_to_timestamp_micros(&manifest_path, 4_000).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(manifest.segments.len(), 2);
    assert_eq!(manifest.checkpoint.durable_record_count, 4);
    assert_eq!(manifest.checkpoint.last_durable_txn_id, Some(4));
    assert_eq!(manifest.record_timestamps.len(), 4);
    assert_eq!(read_records.len(), 4);
    assert_eq!(&read_records[3].payload[..], &b"SET d=4"[..]);
    assert_eq!(target.target_txn_id, 4);
    assert_eq!(target_records.len(), 4);
}

#[test]
fn wal_archive_ingest_rejects_non_increasing_segment_without_manifest_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-ingest-reject-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let ingest_segment = segment_dir.join("segment-0002.wal");
    let records = vec![WalRecord {
        txn_id: 2,
        payload: b"SET b=2".to_vec().into(),
    }];
    let ingest_records = vec![WalRecord {
        txn_id: 2,
        payload: b"SET duplicate=2".to_vec().into(),
    }];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let before_manifest = fs::read_to_string(&manifest_path).unwrap();
    write_wal_segment(&ingest_segment, &ingest_records).unwrap();
    let err = append_wal_archive_segment(&manifest_path, &ingest_segment).unwrap_err();
    let after_manifest = fs::read_to_string(&manifest_path).unwrap();
    let (_manifest, read_records) = read_wal_archive(&manifest_path).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("not after durable transaction 2"));
    assert_eq!(after_manifest, before_manifest);
    assert_eq!(read_records.len(), 1);
    assert_eq!(&read_records[0].payload[..], &b"SET b=2"[..]);
}

#[test]
fn wal_archive_ingest_requires_timestamp_metadata_when_archive_has_timestamps() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-ingest-timestamp-required-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let ingest_segment = segment_dir.join("segment-0002.wal");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    }];
    let timestamps = vec![WalArchiveRecordTimestamp {
        txn_id: 1,
        timestamp_micros: 1_000,
    }];
    let ingest_records = vec![WalRecord {
        txn_id: 2,
        payload: b"SET b=2".to_vec().into(),
    }];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
        .unwrap();
    let before_manifest = fs::read_to_string(&manifest_path).unwrap();
    write_wal_segment(&ingest_segment, &ingest_records).unwrap();
    let err = append_wal_archive_segment(&manifest_path, &ingest_segment).unwrap_err();
    let after_manifest = fs::read_to_string(&manifest_path).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert!(err
        .to_string()
        .contains("requires timestamp metadata for ingested segment"));
    assert_eq!(after_manifest, before_manifest);
}

#[test]
fn wal_archive_forks_transaction_timeline_with_ancestry() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timeline-txn-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let source_manifest = dir.join("source").join("MANIFEST");
    let source_segments = dir.join("source").join("segments");
    let branch_manifest = dir.join("branch").join("MANIFEST");
    let branch_segments = dir.join("branch").join("segments");
    let timeline_path = dir.join("branch").join("TIMELINE");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                .to_vec()
                .into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                .to_vec()
                .into(),
        },
    ];
    write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();

    let branch = fork_wal_archive_timeline_to_txn(
        &source_manifest,
        &branch_manifest,
        &branch_segments,
        &timeline_path,
        "timeline-0002",
        Some("timeline-0001"),
        2,
    )
    .unwrap();
    let timeline = read_wal_archive_timeline(&timeline_path).unwrap();
    let (_manifest, branch_records) = read_wal_archive(&branch_manifest).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(branch.timeline, timeline);
    assert_eq!(timeline.timeline_id, "timeline-0002");
    assert_eq!(
        timeline.parent_timeline_id.as_deref(),
        Some("timeline-0001")
    );
    assert_eq!(timeline.fork_txn_id, 2);
    assert_eq!(timeline.fork_timestamp_micros, None);
    assert_eq!(branch.manifest.checkpoint.durable_record_count, 2);
    assert_eq!(branch.manifest.checkpoint.last_durable_txn_id, Some(2));
    assert_eq!(branch_records.len(), 2);
    assert_eq!(branch_records[1].txn_id, 2);
}

#[test]
fn wal_archive_forks_timestamp_timeline_and_rejects_self_parent_without_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timeline-timestamp-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let source_manifest = dir.join("source").join("MANIFEST");
    let source_segments = dir.join("source").join("segments");
    let branch_manifest = dir.join("branch").join("MANIFEST");
    let branch_segments = dir.join("branch").join("segments");
    let timeline_path = dir.join("branch").join("TIMELINE");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                .to_vec()
                .into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                .to_vec()
                .into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 2_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 3,
            timestamp_micros: 3_000,
        },
    ];
    write_wal_archive_with_timestamps(&source_manifest, &source_segments, &records, 2, &timestamps)
        .unwrap();

    let self_parent_err = fork_wal_archive_timeline_to_timestamp_micros(
        &source_manifest,
        &branch_manifest,
        &branch_segments,
        &timeline_path,
        "timeline-0002",
        Some("timeline-0002"),
        2_000,
    )
    .unwrap_err();
    assert!(!branch_manifest.exists());
    assert!(self_parent_err
        .to_string()
        .contains("cannot be its own parent"));

    let branch = fork_wal_archive_timeline_to_timestamp_micros(
        &source_manifest,
        &branch_manifest,
        &branch_segments,
        &timeline_path,
        "timeline-0002",
        Some("timeline-0001"),
        2_000,
    )
    .unwrap();
    let timeline = read_wal_archive_timeline(&timeline_path).unwrap();
    let (_manifest, branch_records) = read_wal_archive(&branch_manifest).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(timeline.fork_txn_id, 2);
    assert_eq!(timeline.fork_timestamp_micros, Some(2_000));
    assert_eq!(branch.manifest.record_timestamps.len(), 2);
    assert_eq!(branch_records.len(), 2);
    assert_eq!(branch.timeline, timeline);
}

#[test]
fn wal_archive_timeline_rejects_delimiter_before_sidecar_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-timeline-delimiter-sidecar-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    let existing_path = dir.join("EXISTING_TIMELINE");
    let missing_path = dir.join("MISSING_TIMELINE");
    let sentinel = b"existing timeline bytes";
    fs::write(&existing_path, sentinel).unwrap();
    let base = WalArchiveTimeline {
        timeline_id: "timeline-valid".to_string(),
        parent_timeline_id: None,
        fork_txn_id: 7,
        fork_timestamp_micros: Some(9),
        source_manifest_path: dir.join("source-manifest"),
        branch_manifest_path: dir.join("branch-manifest"),
    };

    let mut invalid_id = base.clone();
    invalid_id.timeline_id = "timeline|injected".to_string();
    let err = write_wal_archive_timeline(&missing_path, &invalid_id).unwrap_err();
    assert!(err.to_string().contains("contains unsupported value"));
    assert!(!missing_path.exists());
    assert!(!temporary_control_path(&missing_path).exists());

    let mut invalid_parent = base;
    invalid_parent.parent_timeline_id = Some("parent|injected".to_string());
    let err = write_wal_archive_timeline(&existing_path, &invalid_parent).unwrap_err();
    assert!(err.to_string().contains("contains unsupported value"));
    assert_eq!(fs::read(&existing_path).unwrap(), sentinel);
    assert!(!temporary_control_path(&existing_path).exists());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn wal_archive_timeline_registry_rejects_delimiter_without_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-timeline-delimiter-registry-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let entry =
        |timeline_id: &str, parent_timeline_id: Option<&str>| WalArchiveTimelineRegistryEntry {
            timeline_id: timeline_id.to_string(),
            parent_timeline_id: parent_timeline_id.map(str::to_string),
            fork_txn_id: 1,
            fork_timestamp_micros: None,
            timeline_path: dir.join(format!("{timeline_id}.timeline")),
            branch_manifest_path: dir.join(format!("{timeline_id}.manifest")),
        };

    let invalid_id = WalArchiveTimelineRegistry {
        timelines: vec![entry("timeline|injected", None)],
    };
    let err = write_wal_archive_timeline_registry(&registry_path, &invalid_id).unwrap_err();
    assert!(err.to_string().contains("contains unsupported value"));
    assert!(!registry_path.exists());
    assert!(!temporary_control_path(&registry_path).exists());

    let valid = WalArchiveTimelineRegistry {
        timelines: vec![entry("timeline-root", None)],
    };
    write_wal_archive_timeline_registry(&registry_path, &valid).unwrap();
    let before = fs::read(&registry_path).unwrap();
    let invalid_parent = WalArchiveTimelineRegistry {
        timelines: vec![
            entry("timeline-root", None),
            entry("timeline-child", Some("timeline|injected")),
        ],
    };
    let err = write_wal_archive_timeline_registry(&registry_path, &invalid_parent).unwrap_err();
    assert!(err.to_string().contains("contains unsupported value"));
    assert_eq!(fs::read(&registry_path).unwrap(), before);
    assert_eq!(
        read_wal_archive_timeline_registry(&registry_path).unwrap(),
        valid
    );
    assert!(!temporary_control_path(&registry_path).exists());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn register_timeline_rejects_delimiter_sidecar_without_registry_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-timeline-delimiter-register-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    let timeline_path = dir.join("INJECTED_TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let missing_registry_path = dir.join("MISSING_REGISTRY");
    fs::write(
        &timeline_path,
        format!(
            "GPUDBWALTIMELINE1\ntimeline_id=timeline|injected\nparent_timeline_id=none\nfork_txn_id=1\nfork_timestamp_micros=none\nsource_manifest_path={}\nbranch_manifest_path={}\n",
            dir.join("source-manifest").display(),
            dir.join("branch-manifest").display()
        ),
    )
    .unwrap();

    let valid = WalArchiveTimelineRegistry {
        timelines: vec![WalArchiveTimelineRegistryEntry {
            timeline_id: "timeline-root".to_string(),
            parent_timeline_id: None,
            fork_txn_id: 0,
            fork_timestamp_micros: None,
            timeline_path: dir.join("root.timeline"),
            branch_manifest_path: dir.join("root.manifest"),
        }],
    };
    write_wal_archive_timeline_registry(&registry_path, &valid).unwrap();
    let before = fs::read(&registry_path).unwrap();

    let err = register_wal_archive_timeline(&missing_registry_path, &timeline_path).unwrap_err();
    assert!(err.to_string().contains("contains unsupported value"));
    assert!(!missing_registry_path.exists());
    assert!(!temporary_control_path(&missing_registry_path).exists());

    let err = register_wal_archive_timeline(&registry_path, &timeline_path).unwrap_err();
    assert!(err.to_string().contains("contains unsupported value"));
    assert_eq!(fs::read(&registry_path).unwrap(), before);
    assert_eq!(
        read_wal_archive_timeline_registry(&registry_path).unwrap(),
        valid
    );
    assert!(!temporary_control_path(&registry_path).exists());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn wal_archive_timeline_registry_requires_parent_before_child_and_unique_ids() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timeline-registry-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let source_manifest = dir.join("source").join("MANIFEST");
    let source_segments = dir.join("source").join("segments");
    let root_timeline_path = dir.join("source").join("TIMELINE");
    let branch_manifest = dir.join("branch").join("MANIFEST");
    let branch_segments = dir.join("branch").join("segments");
    let branch_timeline_path = dir.join("branch").join("TIMELINE");
    let missing_parent_branch_manifest = dir.join("missing-parent").join("MANIFEST");
    let missing_parent_branch_segments = dir.join("missing-parent").join("segments");
    let missing_parent_timeline_path = dir.join("missing-parent").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                .to_vec()
                .into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                .to_vec()
                .into(),
        },
    ];
    write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
    write_wal_archive_timeline(
        &root_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-0001".to_string(),
            parent_timeline_id: None,
            fork_txn_id: 0,
            fork_timestamp_micros: None,
            source_manifest_path: source_manifest.clone(),
            branch_manifest_path: source_manifest.clone(),
        },
    )
    .unwrap();
    fork_wal_archive_timeline_to_txn(
        &source_manifest,
        &branch_manifest,
        &branch_segments,
        &branch_timeline_path,
        "timeline-0002",
        Some("timeline-0001"),
        2,
    )
    .unwrap();
    fork_wal_archive_timeline_to_txn(
        &source_manifest,
        &missing_parent_branch_manifest,
        &missing_parent_branch_segments,
        &missing_parent_timeline_path,
        "timeline-0003",
        Some("timeline-missing"),
        2,
    )
    .unwrap();

    let missing_parent_err =
        register_wal_archive_timeline(&registry_path, &missing_parent_timeline_path).unwrap_err();
    assert!(missing_parent_err
        .to_string()
        .contains("missing parent timeline timeline-missing"));
    assert!(!registry_path.exists());

    let root_registry = register_wal_archive_timeline(&registry_path, &root_timeline_path).unwrap();
    assert_eq!(root_registry.timelines.len(), 1);
    assert_eq!(root_registry.timelines[0].timeline_id, "timeline-0001");

    let registry = register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();
    assert_eq!(registry.timelines.len(), 2);
    assert_eq!(registry.timelines[1].timeline_id, "timeline-0002");
    assert_eq!(
        registry.timelines[1].parent_timeline_id.as_deref(),
        Some("timeline-0001")
    );
    assert_eq!(registry.timelines[1].fork_txn_id, 2);

    let before_registry = fs::read_to_string(&registry_path).unwrap();
    let duplicate_err =
        register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap_err();
    let after_registry = fs::read_to_string(&registry_path).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert!(duplicate_err
        .to_string()
        .contains("already contains timeline timeline-0002"));
    assert_eq!(after_registry, before_registry);
}

#[test]
fn wal_archive_timeline_registry_selects_validated_branch_and_rejects_stale_sidecar() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timeline-select-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let source_manifest = dir.join("source").join("MANIFEST");
    let source_segments = dir.join("source").join("segments");
    let root_timeline_path = dir.join("source").join("TIMELINE");
    let branch_manifest = dir.join("branch").join("MANIFEST");
    let branch_segments = dir.join("branch").join("segments");
    let branch_timeline_path = dir.join("branch").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                .to_vec()
                .into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                .to_vec()
                .into(),
        },
    ];
    write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
    write_wal_archive_timeline(
        &root_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-0001".to_string(),
            parent_timeline_id: None,
            fork_txn_id: 0,
            fork_timestamp_micros: None,
            source_manifest_path: source_manifest.clone(),
            branch_manifest_path: source_manifest.clone(),
        },
    )
    .unwrap();
    fork_wal_archive_timeline_to_txn(
        &source_manifest,
        &branch_manifest,
        &branch_segments,
        &branch_timeline_path,
        "timeline-0002",
        Some("timeline-0001"),
        2,
    )
    .unwrap();
    register_wal_archive_timeline(&registry_path, &root_timeline_path).unwrap();
    register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();

    let selection = select_wal_archive_timeline(&registry_path, "timeline-0002").unwrap();
    assert_eq!(selection.entry.timeline_id, "timeline-0002");
    assert_eq!(selection.timeline.fork_txn_id, 2);
    assert_eq!(selection.manifest.checkpoint.durable_record_count, 2);

    let missing_err = select_wal_archive_timeline(&registry_path, "timeline-missing").unwrap_err();
    assert!(missing_err
        .to_string()
        .contains("has no timeline timeline-missing"));

    write_wal_archive_timeline(
        &branch_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-0002".to_string(),
            parent_timeline_id: Some("timeline-0001".to_string()),
            fork_txn_id: 3,
            fork_timestamp_micros: None,
            source_manifest_path: source_manifest.clone(),
            branch_manifest_path: branch_manifest.clone(),
        },
    )
    .unwrap();
    let stale_err = select_wal_archive_timeline(&registry_path, "timeline-0002").unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(stale_err.to_string().contains("does not match sidecar"));
}

#[test]
fn wal_archive_timeline_prune_keeps_target_ancestry_and_removes_unreferenced_artifacts() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timeline-prune-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let source_manifest = dir.join("source").join("MANIFEST");
    let source_segments = dir.join("source").join("segments");
    let source_timeline_path = dir.join("source").join("TIMELINE");
    let keep_manifest = dir.join("keep").join("MANIFEST");
    let keep_segments = dir.join("keep").join("segments");
    let keep_timeline_path = dir.join("keep").join("TIMELINE");
    let prune_manifest = dir.join("prune").join("MANIFEST");
    let prune_segments = dir.join("prune").join("segments");
    let prune_timeline_path = dir.join("prune").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                .to_vec()
                .into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                .to_vec()
                .into(),
        },
    ];
    write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
    write_wal_archive_timeline(
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
    fork_wal_archive_timeline_to_txn(
        &source_manifest,
        &keep_manifest,
        &keep_segments,
        &keep_timeline_path,
        "timeline-keep-0002",
        Some("timeline-main-0001"),
        3,
    )
    .unwrap();
    fork_wal_archive_timeline_to_txn(
        &source_manifest,
        &prune_manifest,
        &prune_segments,
        &prune_timeline_path,
        "timeline-prune-0003",
        Some("timeline-main-0001"),
        2,
    )
    .unwrap();
    register_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
    register_wal_archive_timeline(&registry_path, &keep_timeline_path).unwrap();
    register_wal_archive_timeline(&registry_path, &prune_timeline_path).unwrap();

    let plan = plan_wal_archive_timeline_prune(&registry_path, "timeline-keep-0002").unwrap();
    let applied = apply_wal_archive_timeline_prune(&registry_path, "timeline-keep-0002").unwrap();
    let registry = read_wal_archive_timeline_registry(&registry_path).unwrap();
    let selection = select_wal_archive_timeline(&registry_path, "timeline-keep-0002").unwrap();
    let pruned_err =
        select_wal_archive_timeline(&registry_path, "timeline-prune-0003").unwrap_err();

    assert_eq!(plan.retained_timeline_id, "timeline-keep-0002");
    assert_eq!(
        plan.retained_timeline_ids,
        vec![
            "timeline-main-0001".to_string(),
            "timeline-keep-0002".to_string()
        ]
    );
    assert_eq!(
        plan.removed_timeline_ids,
        vec!["timeline-prune-0003".to_string()]
    );
    assert_eq!(applied, plan);
    assert_eq!(registry.timelines.len(), 2);
    assert_eq!(selection.manifest.checkpoint.durable_record_count, 3);
    assert!(!prune_timeline_path.exists());
    assert!(!prune_manifest.exists());
    assert!(!prune_segments.join("segment-0001.wal").exists());
    assert!(keep_timeline_path.exists());
    assert!(keep_manifest.exists());
    assert!(keep_segments.join("segment-0001.wal").exists());
    assert!(pruned_err
        .to_string()
        .contains("has no timeline timeline-prune-0003"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn wal_archive_timeline_prune_rejects_stale_sidecar_without_registry_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timeline-prune-stale-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let source_manifest = dir.join("source").join("MANIFEST");
    let source_segments = dir.join("source").join("segments");
    let source_timeline_path = dir.join("source").join("TIMELINE");
    let branch_manifest = dir.join("branch").join("MANIFEST");
    let branch_segments = dir.join("branch").join("segments");
    let branch_timeline_path = dir.join("branch").join("TIMELINE");
    let registry_path = dir.join("TIMELINE_REGISTRY");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                .to_vec()
                .into(),
        },
    ];
    write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
    write_wal_archive_timeline(
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
    fork_wal_archive_timeline_to_txn(
        &source_manifest,
        &branch_manifest,
        &branch_segments,
        &branch_timeline_path,
        "timeline-branch-0002",
        Some("timeline-main-0001"),
        2,
    )
    .unwrap();
    register_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
    register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();
    let before_registry = fs::read_to_string(&registry_path).unwrap();
    write_wal_archive_timeline(
        &branch_timeline_path,
        &WalArchiveTimeline {
            timeline_id: "timeline-branch-0002".to_string(),
            parent_timeline_id: Some("timeline-main-0001".to_string()),
            fork_txn_id: 1,
            fork_timestamp_micros: None,
            source_manifest_path: source_manifest.clone(),
            branch_manifest_path: branch_manifest.clone(),
        },
    )
    .unwrap();

    let err = apply_wal_archive_timeline_prune(&registry_path, "timeline-branch-0002").unwrap_err();
    let after_registry = fs::read_to_string(&registry_path).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("does not match sidecar"));
    assert_eq!(after_registry, before_registry);
}

#[test]
fn wal_archive_retention_plan_keeps_exact_transaction_prefix() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-retention-plan-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
        WalRecord {
            txn_id: 4,
            payload: b"SET d=4".to_vec().into(),
        },
        WalRecord {
            txn_id: 5,
            payload: b"SET e=5".to_vec().into(),
        },
    ];

    write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
    let plan = plan_wal_archive_retention_to_txn(&manifest_path, 3).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(plan.target_txn_id, 3);
    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(plan.removed_record_count, 2);
    assert_eq!(plan.retained_manifest.segments.len(), 2);
    assert_eq!(
        plan.retained_manifest.checkpoint,
        WalCheckpointMeta {
            durable_record_count: 3,
            last_durable_txn_id: Some(3),
        }
    );
    assert_eq!(plan.retained_manifest.segments[1].record_count, 1);
    assert_eq!(plan.retained_manifest.segments[1].last_txn_id, Some(3));
    assert_eq!(
        plan.removed_segments,
        vec![segment_dir.join("segment-0003.wal")]
    );
}

#[test]
fn wal_archive_retention_apply_rewrites_manifest_and_removes_tail_segments() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-retention-apply-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
        WalRecord {
            txn_id: 4,
            payload: b"SET d=4".to_vec().into(),
        },
        WalRecord {
            txn_id: 5,
            payload: b"SET e=5".to_vec().into(),
        },
    ];

    write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
    let removed_tail = segment_dir.join("segment-0003.wal");
    assert!(removed_tail.exists());

    let plan = apply_wal_archive_retention_to_txn(&manifest_path, 3).unwrap();
    let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
    let target_err = read_wal_archive_to_txn(&manifest_path, 4).unwrap_err();

    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(plan.removed_record_count, 2);
    assert!(!removed_tail.exists());
    assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
    assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(3));
    assert_eq!(retained_records.len(), 3);
    assert_eq!(&retained_records[2].payload[..], &b"SET c=3"[..]);
    assert!(target_err
        .to_string()
        .contains("beyond last durable transaction"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn wal_archive_timestamp_retention_rewrites_manifest_and_preserves_timestamp_prefix() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timestamp-retention-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
        WalRecord {
            txn_id: 4,
            payload: b"SET d=4".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 2_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 3,
            timestamp_micros: 3_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 4,
            timestamp_micros: 4_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
        .unwrap();
    let removed_tail = segment_dir.join("segment-0004.wal");
    assert!(removed_tail.exists());

    let plan = apply_wal_archive_retention_to_timestamp_micros(&manifest_path, 3_000).unwrap();
    let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
    let target_err = read_wal_archive_to_timestamp_micros(&manifest_path, 4_000).unwrap_err();

    assert_eq!(plan.target_txn_id, 3);
    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(plan.removed_record_count, 1);
    assert!(!removed_tail.exists());
    assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
    assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(3));
    assert_eq!(
        retained_manifest.record_timestamps,
        vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
        ]
    );
    assert_eq!(retained_records.len(), 3);
    assert_eq!(&retained_records[2].payload[..], &b"SET c=3"[..]);
    assert!(target_err
        .to_string()
        .contains("beyond last durable timestamp"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn wal_archive_timestamp_retention_rejects_between_boundary_without_mutation() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-timestamp-retention-missing-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 10,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 20,
            payload: b"SET b=2".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 10,
            timestamp_micros: 10_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 20,
            timestamp_micros: 20_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
        .unwrap();
    let before_manifest = fs::read_to_string(&manifest_path).unwrap();
    let err = apply_wal_archive_retention_to_timestamp_micros(&manifest_path, 15_000).unwrap_err();
    let after_manifest = fs::read_to_string(&manifest_path).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert!(err
        .to_string()
        .contains("falls between archived transaction boundaries"));
    assert_eq!(after_manifest, before_manifest);
}

#[test]
fn wal_archive_base_retention_plan_keeps_base_boundary_suffix() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-base-retention-plan-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
        WalRecord {
            txn_id: 4,
            payload: b"SET d=4".to_vec().into(),
        },
    ];
    let timestamps = vec![
        WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 2,
            timestamp_micros: 2_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 3,
            timestamp_micros: 3_000,
        },
        WalArchiveRecordTimestamp {
            txn_id: 4,
            timestamp_micros: 4_000,
        },
    ];

    write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
        .unwrap();
    let plan = plan_wal_archive_retention_from_txn(&manifest_path, 2).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(plan.target_txn_id, 2);
    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(plan.removed_record_count, 1);
    assert_eq!(plan.retained_manifest.checkpoint.durable_record_count, 3);
    assert_eq!(
        plan.retained_manifest.checkpoint.last_durable_txn_id,
        Some(4)
    );
    assert_eq!(plan.retained_manifest.segments[0].first_txn_id, Some(2));
    assert_eq!(plan.retained_manifest.record_timestamps[0].txn_id, 2);
    assert_eq!(
        plan.retained_manifest.record_timestamps[0].timestamp_micros,
        2_000
    );
}

#[test]
fn wal_archive_base_retention_apply_rewrites_to_base_suffix() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-base-retention-apply-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        },
        WalRecord {
            txn_id: 4,
            payload: b"SET d=4".to_vec().into(),
        },
    ];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let plan = apply_wal_archive_retention_from_txn(&manifest_path, 2).unwrap();
    let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
    let target_err = read_wal_archive_to_txn(&manifest_path, 1).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(plan.retained_record_count, 3);
    assert_eq!(plan.removed_record_count, 1);
    assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
    assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(4));
    assert_eq!(
        retained_records
            .iter()
            .map(|record| record.txn_id)
            .collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
    assert!(target_err
        .to_string()
        .contains("before first archived transaction"));
}

#[test]
fn wal_archive_retention_rejects_malformed_archive_before_cleanup() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-retention-malformed-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![
        WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        },
        WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        },
    ];

    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let err = apply_wal_archive_retention_to_txn(&manifest_path, 1).unwrap_err();
    assert!(segment_dir.join("segment-0002.wal").exists());
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("non-increasing transaction order"));
}

#[test]
fn wal_archive_rejects_missing_segment() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-missing-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_dir = dir.join("segments");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    }];

    let manifest = write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    fs::remove_file(resolve_manifest_path(
        &manifest_path,
        &manifest.segments[0].segment_path,
    ))
    .unwrap();
    let err = read_wal_archive(&manifest_path).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("failed to open WAL segment"));
}

#[test]
fn wal_archive_rejects_manifest_record_count_mismatch() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-count-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_path = dir.join("segment-0001.wal");
    write_wal_segment(
        &segment_path,
        &[WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }],
    )
    .unwrap();
    let manifest = WalArchiveManifest {
        segments: vec![WalArchiveSegment {
            segment_path: PathBuf::from("segment-0001.wal"),
            record_count: 2,
            first_txn_id: Some(1),
            last_txn_id: Some(1),
        }],
        checkpoint: WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(1),
        },
        record_timestamps: Vec::new(),
    };
    write_wal_archive_manifest(&manifest_path, &manifest).unwrap();

    let err = read_wal_archive(&manifest_path).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("expected 2 records"));
}

#[test]
fn wal_archive_rejects_non_increasing_transaction_order() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-archive-order-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("MANIFEST");
    let segment_a = dir.join("segment-0001.wal");
    let segment_b = dir.join("segment-0002.wal");
    write_wal_segment(
        &segment_a,
        &[WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        }],
    )
    .unwrap();
    write_wal_segment(
        &segment_b,
        &[WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }],
    )
    .unwrap();
    let manifest = WalArchiveManifest {
        segments: vec![
            WalArchiveSegment {
                segment_path: PathBuf::from("segment-0001.wal"),
                record_count: 1,
                first_txn_id: Some(2),
                last_txn_id: Some(2),
            },
            WalArchiveSegment {
                segment_path: PathBuf::from("segment-0002.wal"),
                record_count: 1,
                first_txn_id: Some(1),
                last_txn_id: Some(1),
            },
        ],
        checkpoint: WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(1),
        },
        record_timestamps: Vec::new(),
    };
    write_wal_archive_manifest(&manifest_path, &manifest).unwrap();

    let err = read_wal_archive(&manifest_path).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("non-increasing transaction order"));
}

#[test]
fn v2_archive_manifest_timeline_registry_and_backup_manifest_reject_tampering() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-v2-authority-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let manifest_path = dir.join("archive").join("MANIFEST");
    let segment_dir = dir.join("archive").join("segments");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET v2=1".to_vec().into(),
    }];
    write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
    let manifest_body = fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest_body.starts_with("GPUDBWALARCHIVE2\n"));
    let legacy_manifest = manifest_body
        .replace("GPUDBWALARCHIVE2", "GPUDBWALARCHIVE1")
        .lines()
        .filter(|line| !line.starts_with("sha256="))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&manifest_path, legacy_manifest).unwrap();
    assert_eq!(
        read_wal_archive_manifest(&manifest_path)
            .unwrap()
            .checkpoint
            .durable_record_count,
        1
    );
    fs::write(&manifest_path, &manifest_body).unwrap();
    fs::write(
        &manifest_path,
        manifest_body.replace("durable_record_count=1", "durable_record_count=2"),
    )
    .unwrap();
    assert!(read_wal_archive_manifest(&manifest_path).is_err());
    fs::write(&manifest_path, &manifest_body).unwrap();

    let timeline_path = dir.join("TIMELINE");
    let timeline = WalArchiveTimeline {
        timeline_id: "main".to_string(),
        parent_timeline_id: None,
        fork_txn_id: 1,
        fork_timestamp_micros: None,
        source_manifest_path: manifest_path.clone(),
        branch_manifest_path: manifest_path.clone(),
    };
    write_wal_archive_timeline(&timeline_path, &timeline).unwrap();
    let timeline_body = fs::read_to_string(&timeline_path).unwrap();
    assert!(timeline_body.starts_with("GPUDBWALTIMELINE2\n"));
    fs::write(&timeline_path, timeline_body.replace("timeline_id=main", "timeline_id=evil"))
        .unwrap();
    assert!(read_wal_archive_timeline(&timeline_path).is_err());

    let registry_path = dir.join("REGISTRY");
    write_wal_archive_timeline_registry(
        &registry_path,
        &WalArchiveTimelineRegistry {
            timelines: Vec::new(),
        },
    )
    .unwrap();
    let registry_body = fs::read_to_string(&registry_path).unwrap();
    assert!(registry_body.starts_with("GPUDBWALTIMELINEREGISTRY2\n"));
    fs::write(
        &registry_path,
        registry_body.replace("timeline_count=0", "timeline_count=1"),
    )
    .unwrap();
    assert!(read_wal_archive_timeline_registry(&registry_path).is_err());

    let backup_path = dir.join("backup").join("BACKUP");
    let object_dir = dir.join("backup").join("objects");
    export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
    let backup_body = fs::read_to_string(&backup_path).unwrap();
    assert!(backup_body.starts_with("GPUDBWALOBJECTBACKUP2\n"));
    fs::write(&backup_path, backup_body.replace("objects=2", "objects=3")).unwrap();
    assert!(read_wal_archive_object_backup_manifest(&backup_path).is_err());

    let _ = fs::remove_dir_all(dir);
}
