#[test]
fn flush_commits_all_appended_records() {
    let mut wal = WalBuffer::default();
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    });
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"SET b=2".to_vec().into(),
    });

    wal.flush_all().unwrap();

    assert_eq!(wal.flushed_count(), 2);
}

#[test]
fn fail_next_flush_is_one_shot() {
    let mut wal = WalBuffer::default();
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    });

    wal.fail_next_flush();
    let err = wal.flush_all().unwrap_err();
    assert!(matches!(err, EngineError::Durability(_)));
    assert_eq!(wal.flushed_count(), 0);

    wal.flush_all().unwrap();
    assert_eq!(wal.flushed_count(), 1);
}

#[test]
fn truncate_shrinks_records_and_adjusts_flushed_count() {
    let mut wal = WalBuffer::default();
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    });
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"SET b=2".to_vec().into(),
    });

    wal.flush_all().unwrap();
    assert_eq!(wal.flushed_count(), 2);

    wal.truncate(1);
    assert_eq!(wal.len(), 1);
    assert_eq!(wal.flushed_count(), 1);
}

#[test]
fn unflushed_count_tracks_unpersisted_tail() {
    let mut wal = WalBuffer::default();
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    });
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"SET b=2".to_vec().into(),
    });

    assert_eq!(wal.unflushed_count(), 2);

    wal.flush_all().unwrap();
    assert_eq!(wal.unflushed_count(), 0);

    wal.append(WalRecord {
        txn_id: 3,
        payload: b"SET c=3".to_vec().into(),
    });
    assert_eq!(wal.unflushed_count(), 1);
}

#[test]
fn flushed_records_expose_only_durable_prefix() {
    let mut wal = WalBuffer::default();
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    });
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"SET b=2".to_vec().into(),
    });

    assert!(wal.flushed_records().is_empty());

    wal.flush_all().unwrap();
    wal.append(WalRecord {
        txn_id: 3,
        payload: b"SET c=3".to_vec().into(),
    });

    let durable = wal.flushed_records();
    assert_eq!(durable.len(), 2);
    assert_eq!(durable[0].txn_id, 1);
    assert_eq!(durable[1].txn_id, 2);
}

#[test]
fn flush_failure_does_not_advance_flushed_records() {
    let mut wal = WalBuffer::default();
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    });

    wal.fail_next_flush();
    let _ = wal.flush_all();

    assert!(wal.flushed_records().is_empty());
    assert_eq!(wal.unflushed_count(), 1);
}

#[test]
fn checkpoint_meta_tracks_durable_prefix_and_last_txn_id() {
    let mut wal = WalBuffer::default();
    assert_eq!(
        wal.checkpoint_meta(),
        WalCheckpointMeta {
            durable_record_count: 0,
            last_durable_txn_id: None,
        }
    );

    wal.append(WalRecord {
        txn_id: 7,
        payload: b"SET a=1".to_vec().into(),
    });
    wal.append(WalRecord {
        txn_id: 8,
        payload: b"SET b=2".to_vec().into(),
    });

    assert_eq!(
        wal.checkpoint_meta(),
        WalCheckpointMeta {
            durable_record_count: 0,
            last_durable_txn_id: None,
        }
    );

    wal.flush_all().unwrap();
    assert_eq!(
        wal.checkpoint_meta(),
        WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(8),
        }
    );

    wal.append(WalRecord {
        txn_id: 9,
        payload: b"SET c=3".to_vec().into(),
    });
    assert_eq!(
        wal.checkpoint_meta(),
        WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(8),
        }
    );
}

#[test]
fn checkpoint_meta_does_not_advance_on_failed_flush() {
    let mut wal = WalBuffer::default();
    wal.append(WalRecord {
        txn_id: 11,
        payload: b"SET a=1".to_vec().into(),
    });
    wal.flush_all().unwrap();

    wal.append(WalRecord {
        txn_id: 12,
        payload: b"SET b=2".to_vec().into(),
    });
    wal.fail_next_flush();
    assert!(wal.flush_all().is_err());

    assert_eq!(
        wal.checkpoint_meta(),
        WalCheckpointMeta {
            durable_record_count: 1,
            last_durable_txn_id: Some(11),
        }
    );
}

#[test]
fn durable_flush_persists_records_to_real_segment() {
    let path = test_wal_path("durable-flush");
    let mut wal = WalBuffer::with_durable_segment(&path);
    assert!(wal.is_durable());
    assert_eq!(wal.durable_segment_path(), Some(path.as_path()));

    wal.append(WalRecord {
        txn_id: 1,
        payload: b"CREATE TABLE t (id INT)".to_vec().into(),
    });
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"INSERT INTO t (id) VALUES (1)".to_vec().into(),
    });
    // Nothing on disk until the flush.
    assert!(!path.exists());

    wal.flush_all().unwrap();
    assert_eq!(wal.flushed_count(), 2);

    // The flushed records are now a real, CRC-checked, fsynced segment.
    let recovered = read_wal_segment(&path).unwrap();
    let _ = fs::remove_file(&path);
    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0].txn_id, 1);
    assert_eq!(
        &recovered[1].payload[..],
        &b"INSERT INTO t (id) VALUES (1)"[..]
    );
}

#[test]
fn durable_flush_preserves_history_across_flushes() {
    let path = test_wal_path("durable-history");
    let mut wal = WalBuffer::with_durable_segment(&path);

    wal.append(WalRecord {
        txn_id: 1,
        payload: b"one".to_vec().into(),
    });
    wal.flush_all().unwrap();
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"two".to_vec().into(),
    });
    wal.flush_all().unwrap();

    // The segment must contain BOTH records after the second flush (appended, not clobbered).
    let recovered = read_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0].txn_id, 1);
    assert_eq!(recovered[1].txn_id, 2);
}

fn remove_segment_files(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(wal_tail_offset_path(path));
    let _ = fs::remove_file(durable_identity_path(path));
}

fn identity_test_value(tag: u8) -> CanonicalIdentity {
    CanonicalIdentity {
        database_id: [tag; 16],
        cluster_id: [tag.wrapping_add(1); 16],
        timeline_id: [tag.wrapping_add(2); 16],
        format_epoch: u64::from(tag),
    }
}

fn canonical_identity_test_record(txn_id: TxnId, identity: CanonicalIdentity) -> WalRecord {
    let operation = CanonicalFragment {
        kind: CanonicalFragmentKind::RowMutation,
        body: txn_id.to_le_bytes().to_vec(),
    };
    let request_digest = canonical_request_digest(&operation.body);
    let encoded = encode_canonical_envelope(
        CanonicalPhysicalRange {
            log_epoch: 1,
            lane_id: 0,
            segment_id: 1,
            first_frame_ordinal: txn_id,
        },
        &CanonicalPreApplyHeader {
            identity,
            leader_epoch: 1,
            commit_seq: txn_id,
            stable_transaction_id: txn_id,
            request_digest,
            isolation: CanonicalIsolation::ReadCommitted,
            flags: 0,
            catalog_before_epoch: 0,
            catalog_after_epoch: 0,
            catalog_before_digest: [1; 32],
            catalog_after_digest: [1; 32],
            operation_count: 1,
            table_block_count: 1,
            allocator_high_water: txn_id,
        },
        &[operation],
        &CanonicalOutcome {
            kind: CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 1,
            sqlstate: None,
            constraint_id: 0,
            target_digest: request_digest,
            returning_digest: [0; 32],
        },
    )
    .expect("canonical identity test envelope");
    WalRecord {
        txn_id,
        payload: pack_canonical_record_payload(&encoded)
            .expect("canonical identity test payload")
            .into(),
    }
}

fn prepared_canonical_catalog_test_record(
    txn_id: TxnId,
    identity: CanonicalIdentity,
    catalog_before_epoch: u64,
    catalog_after_epoch: u64,
    catalog_before_digest: CanonicalDigest,
    catalog_after_digest: CanonicalDigest,
    kind: CanonicalFragmentKind,
) -> PreparedCanonicalWalRecord {
    let operation = CanonicalFragment {
        kind,
        body: txn_id.to_le_bytes().to_vec(),
    };
    let request_digest = canonical_request_digest(&operation.body);
    encode_canonical_envelope(
        CanonicalPhysicalRange {
            log_epoch: 1,
            lane_id: 0,
            segment_id: txn_id,
            first_frame_ordinal: txn_id,
        },
        &CanonicalPreApplyHeader {
            identity,
            leader_epoch: 1,
            commit_seq: txn_id,
            stable_transaction_id: txn_id,
            request_digest,
            isolation: CanonicalIsolation::ReadCommitted,
            flags: u32::from(kind as u16),
            catalog_before_epoch,
            catalog_after_epoch,
            catalog_before_digest,
            catalog_after_digest,
            operation_count: 1,
            table_block_count: 1,
            allocator_high_water: txn_id,
        },
        &[operation],
        &CanonicalOutcome {
            kind: CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 1,
            sqlstate: None,
            constraint_id: 0,
            target_digest: request_digest,
            returning_digest: [0; 32],
        },
    )
    .expect("canonical catalog test envelope")
    .into_prepared_record(txn_id)
    .expect("canonical catalog test record")
}

#[test]
fn canonical_append_cache_avoids_tail_decodes_and_preserves_header_chain_bytes() {
    let identity = identity_test_value(37);
    let catalog_digest = [9; 32];
    let mut wal = WalBuffer::new();
    for txn_id in 1..=16 {
        let prepared = prepared_canonical_catalog_test_record(
            txn_id,
            identity,
            0,
            0,
            catalog_digest,
            catalog_digest,
            CanonicalFragmentKind::RowMutation,
        );
        let expected = prepared.as_wal_record().clone();
        wal.append_canonical(prepared);
        assert_eq!(wal.last_record(), Some(&expected));
        let tail = wal.canonical_catalog_tail().unwrap().unwrap();
        assert_eq!(tail.identity, identity);
        assert_eq!(tail.catalog_after_epoch, 0);
        assert_eq!(tail.catalog_after_digest, catalog_digest);
        let envelope = decode_canonical_record_payload(&expected.payload)
            .unwrap()
            .unwrap();
        assert_eq!(envelope.header.catalog_before_epoch, 0);
        assert_eq!(envelope.header.catalog_before_digest, catalog_digest);
        assert_eq!(
            envelope.header.catalog_after_epoch,
            tail.catalog_after_epoch
        );
        assert_eq!(
            envelope.header.catalog_after_digest,
            tail.catalog_after_digest
        );
    }
    assert_eq!(
        wal.canonical_catalog_tail_decodes_for_test(),
        0,
        "sealed append must not re-decode the prior logical tail"
    );
}

#[test]
fn canonical_append_cache_chains_row_catalog_row_boundaries() {
    let identity = identity_test_value(41);
    let genesis = [3; 32];
    let catalog_after = [4; 32];
    let mut wal = WalBuffer::new();
    wal.append_canonical(prepared_canonical_catalog_test_record(
        1,
        identity,
        0,
        0,
        genesis,
        genesis,
        CanonicalFragmentKind::RowMutation,
    ));
    let row_tail = wal.canonical_catalog_tail().unwrap().unwrap();
    wal.append_canonical(prepared_canonical_catalog_test_record(
        2,
        identity,
        row_tail.catalog_after_epoch,
        1,
        row_tail.catalog_after_digest,
        catalog_after,
        CanonicalFragmentKind::CatalogMutation,
    ));
    let catalog_tail = wal.canonical_catalog_tail().unwrap().unwrap();
    wal.append_canonical(prepared_canonical_catalog_test_record(
        3,
        identity,
        catalog_tail.catalog_after_epoch,
        catalog_tail.catalog_after_epoch,
        catalog_tail.catalog_after_digest,
        catalog_tail.catalog_after_digest,
        CanonicalFragmentKind::RowMutation,
    ));

    wal.flush_all().unwrap();
    let headers = wal
        .flushed_records()
        .iter()
        .map(|record| {
            decode_canonical_record_payload(&record.payload)
                .unwrap()
                .unwrap()
                .header
        })
        .collect::<Vec<_>>();
    assert_eq!(headers.len(), 3);
    assert_eq!(
        headers[1].catalog_before_epoch,
        headers[0].catalog_after_epoch
    );
    assert_eq!(
        headers[1].catalog_before_digest,
        headers[0].catalog_after_digest
    );
    assert_eq!(
        headers[2].catalog_before_epoch,
        headers[1].catalog_after_epoch
    );
    assert_eq!(
        headers[2].catalog_before_digest,
        headers[1].catalog_after_digest
    );
    let final_header = &headers[2];
    assert_eq!(final_header.catalog_before_epoch, 1);
    assert_eq!(final_header.catalog_before_digest, catalog_after);
    assert_eq!(final_header.catalog_after_epoch, 1);
    assert_eq!(final_header.catalog_after_digest, catalog_after);
    assert_eq!(wal.canonical_catalog_tail_decodes_for_test(), 0);
}

#[test]
fn canonical_tail_retries_from_surviving_or_empty_rollback_boundary() {
    let identity = identity_test_value(45);
    let first_digest = [6; 32];
    let second_digest = [7; 32];
    let mut wal = WalBuffer::new();
    wal.append_canonical(prepared_canonical_catalog_test_record(
        1,
        identity,
        0,
        0,
        first_digest,
        first_digest,
        CanonicalFragmentKind::RowMutation,
    ));
    wal.append_canonical(prepared_canonical_catalog_test_record(
        2,
        identity,
        0,
        1,
        first_digest,
        second_digest,
        CanonicalFragmentKind::CatalogMutation,
    ));
    wal.truncate(1);
    let surviving = wal.canonical_catalog_tail().unwrap().unwrap();
    assert_eq!(surviving.catalog_after_epoch, 0);
    assert_eq!(surviving.catalog_after_digest, first_digest);
    wal.append_canonical(prepared_canonical_catalog_test_record(
        3,
        identity,
        surviving.catalog_after_epoch,
        surviving.catalog_after_epoch,
        surviving.catalog_after_digest,
        surviving.catalog_after_digest,
        CanonicalFragmentKind::RowMutation,
    ));
    wal.truncate(0);
    assert_eq!(wal.canonical_catalog_tail().unwrap(), None);
    assert_eq!(
        wal.canonical_catalog_tail_decodes_for_test(),
        1,
        "surviving rollback tail decodes once; known-empty rollback does not"
    );
}

#[test]
fn generic_malformed_canonical_tail_fails_closed_and_legacy_prefix_is_neutral() {
    let identity = identity_test_value(49);
    let mut malformed = prepared_canonical_catalog_test_record(
        1,
        identity,
        0,
        0,
        [8; 32],
        [8; 32],
        CanonicalFragmentKind::RowMutation,
    )
    .into_wal_record();
    malformed.payload = malformed.payload[..16].to_vec().into();
    let mut wal = WalBuffer::new();
    wal.append(malformed);
    assert!(wal.canonical_catalog_tail().is_err());

    let mut legacy_first = WalBuffer::new();
    legacy_first.append(WalRecord {
        txn_id: 1,
        payload: b"legacy prefix".to_vec().into(),
    });
    assert_eq!(legacy_first.canonical_catalog_tail().unwrap(), None);
    legacy_first.append_canonical(prepared_canonical_catalog_test_record(
        2,
        identity,
        0,
        0,
        [8; 32],
        [8; 32],
        CanonicalFragmentKind::RowMutation,
    ));
    assert_eq!(
        legacy_first
            .canonical_catalog_tail()
            .unwrap()
            .unwrap()
            .identity,
        identity
    );
}

#[test]
fn recovered_constructor_lazily_recovers_canonical_catalog_tail() {
    let path = test_wal_path("canonical-tail-reopen");
    let identity = identity_test_value(53);
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append_canonical(prepared_canonical_catalog_test_record(
            1,
            identity,
            0,
            0,
            [10; 32],
            [10; 32],
            CanonicalFragmentKind::RowMutation,
        ));
        wal.flush_all().unwrap();
    }
    let recovery = recover_wal_segment(&path).unwrap();
    let mut reopened =
        WalBuffer::with_recovered_durable_segment(&path, recovery.records.clone(), &recovery)
            .unwrap();
    assert_eq!(reopened.canonical_catalog_tail_decodes_for_test(), 0);
    let tail = reopened.canonical_catalog_tail().unwrap().unwrap();
    assert_eq!(tail.identity, identity);
    assert_eq!(tail.catalog_after_digest, [10; 32]);
    assert_eq!(reopened.canonical_catalog_tail_decodes_for_test(), 1);
    drop(reopened);
    remove_segment_files(&path);
}

#[test]
fn serial_identity_anchor_loss_after_binding_rejects_before_physical_wal_handoff() {
    let path = test_wal_path("identity-incremental-serial");
    let identity = identity_test_value(1);
    let mut wal = WalBuffer::with_durable_segment(&path);

    wal.append(WalRecord {
        txn_id: 1,
        payload: b"legacy WAL payload".to_vec().into(),
    });
    wal.flush_all()
        .expect("unbound legacy history remains flushable");
    assert_eq!(read_durable_identity(&path).unwrap(), None);
    assert_eq!(wal.flushed_count(), 1);

    wal.append(canonical_identity_test_record(2, identity));
    wal.flush_all().expect("initial canonical bind");
    assert_eq!(read_durable_identity(&path).unwrap(), Some(identity));
    assert_eq!(wal.flushed_count(), 2);
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 2);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 2);

    fs::remove_file(durable_identity_path(&path)).expect("remove identity anchor");
    wal.append(canonical_identity_test_record(3, identity));
    let error = wal
        .flush_all()
        .expect_err("bound anchor loss must reject before serial WAL IO");
    assert!(error.to_string().contains("anchor is missing"));

    assert_eq!(read_durable_identity(&path).unwrap(), None);
    assert_eq!(wal.flushed_count(), 2, "anchor loss cannot advance durable");
    assert_eq!(read_wal_segment(&path).unwrap().len(), 2);
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 3);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 2);

    write_durable_identity(&path, identity).expect("restore identity anchor");
    wal.flush_all()
        .expect("restored anchor retries the same unverified tail");
    assert_eq!(wal.flushed_count(), 3);
    assert_eq!(read_wal_segment(&path).unwrap().len(), 3);
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 4);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 3);
    drop(wal);
    remove_segment_files(&path);
}

#[test]
fn serial_identity_anchor_failure_preserves_durable_cut_and_retries_unverified_tail() {
    let path = test_wal_path("identity-foreign-serial");
    let identity = identity_test_value(4);
    let foreign = identity_test_value(7);
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(canonical_identity_test_record(1, identity));
    wal.flush_all().expect("first flush");

    write_durable_identity(&path, foreign).expect("install foreign anchor");
    wal.append(canonical_identity_test_record(2, identity));
    let error = match wal.begin_group_flush() {
        Ok(_) => panic!("foreign identity anchor must reject before serial group IO"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("another database/timeline"));
    assert_eq!(
        wal.flushed_count(),
        1,
        "identity failure cannot advance durable"
    );
    assert_eq!(wal.durable_identity_verified_records_for_test(), 1);
    assert_eq!(read_wal_segment(&path).unwrap().len(), 1);

    write_durable_identity(&path, identity).expect("restore expected anchor");
    match wal.begin_group_flush().expect("retry after anchor restore") {
        WalGroupFlushBegin::Job(job) => assert_eq!(job.commit().unwrap(), 2),
        WalGroupFlushBegin::Clean { .. } => panic!("unverified record must still need a flush"),
        WalGroupFlushBegin::Busy => panic!("single test flusher must own an available descriptor"),
    }
    assert_eq!(wal.flushed_count(), 2);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 2);
    assert_eq!(
        wal.durable_identity_decoded_records_for_test(),
        3,
        "the failed sidecar check leaves its record outside the verification cursor"
    );
    drop(wal);
    remove_segment_files(&path);
}

#[test]
fn identity_binding_rejects_mixed_lineage_before_serial_group_io() {
    let path = test_wal_path("identity-mixed-serial");
    let identity = identity_test_value(10);
    let foreign = identity_test_value(13);
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(canonical_identity_test_record(1, identity));
    wal.flush_all().expect("first flush");

    wal.append(canonical_identity_test_record(2, foreign));
    let error = match wal.begin_group_flush() {
        Ok(_) => panic!("mixed canonical lineages must reject before serial group IO"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("multiple durable identities"));
    assert_eq!(wal.flushed_count(), 1);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 1);
    assert_eq!(read_wal_segment(&path).unwrap().len(), 1);
    assert_eq!(read_durable_identity(&path).unwrap(), Some(identity));
    drop(wal);
    remove_segment_files(&path);
}

#[test]
fn identity_binding_allows_legacy_prefix_then_canonical_lineage() {
    let path = test_wal_path("identity-legacy-then-canonical");
    let identity = identity_test_value(16);
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"legacy WAL payload".to_vec().into(),
    });
    wal.flush_all().expect("legacy flush");
    assert_eq!(read_durable_identity(&path).unwrap(), None);

    wal.append(canonical_identity_test_record(2, identity));
    wal.flush_all().expect("canonical flush");
    assert_eq!(read_durable_identity(&path).unwrap(), Some(identity));
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 2);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 2);
    drop(wal);
    remove_segment_files(&path);
}

#[test]
fn identity_binding_truncation_clamps_cursor_but_never_forgets_lineage() {
    let path = test_wal_path("identity-truncate-lineage");
    let identity = identity_test_value(19);
    let foreign = identity_test_value(22);
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(canonical_identity_test_record(1, identity));
    wal.flush_all().expect("first flush");
    wal.truncate(0);
    assert_eq!(wal.flushed_count(), 0);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 0);

    wal.append(canonical_identity_test_record(2, foreign));
    assert!(
        wal.begin_group_flush().is_err(),
        "truncation must not let a different canonical lineage replace the original one"
    );
    assert_eq!(wal.flushed_count(), 0);
    assert_eq!(read_durable_identity(&path).unwrap(), Some(identity));
    drop(wal);
    remove_segment_files(&path);
}

#[test]
fn recovered_bound_serial_buffer_scans_only_new_tail_and_continues() {
    let path = test_wal_path("identity-recovered-bound-serial");
    let identity = identity_test_value(25);
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(canonical_identity_test_record(1, identity));
        wal.append(canonical_identity_test_record(2, identity));
        wal.flush_all().expect("first-life flush");
    }
    let recovery = recover_wal_segment(&path).expect("recover first life");
    let mut wal = WalBuffer::with_recovered_durable_segment_bound_to_identity(
        &path,
        recovery.records.clone(),
        &recovery,
        identity,
    )
    .expect("bound recovery constructor");
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 0);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 2);
    wal.append(canonical_identity_test_record(3, identity));
    wal.flush_all().expect("post-recovery flush");
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 1);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 3);
    drop(wal);
    assert_eq!(recover_wal_segment(&path).unwrap().records.len(), 3);
    remove_segment_files(&path);
}

#[test]
fn raw_recovered_serial_canonical_history_requires_anchor_before_segment_reopen() {
    let path = test_wal_path("identity-recovered-raw-serial");
    let identity = identity_test_value(27);
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(canonical_identity_test_record(1, identity));
        wal.flush_all().expect("initial canonical flush");
    }
    let recovery = recover_wal_segment(&path).expect("recover canonical history");
    let segment_before = fs::read(&path).expect("snapshot live segment");
    fs::remove_file(durable_identity_path(&path)).expect("remove identity anchor");

    let error =
        WalBuffer::with_recovered_durable_segment(&path, recovery.records.clone(), &recovery)
            .expect_err(
                "raw canonical recovery must require its existing anchor before segment mutation",
            );
    assert!(error.to_string().contains("anchor is missing"));
    assert_eq!(fs::read(&path).unwrap(), segment_before);
    assert_eq!(read_durable_identity(&path).unwrap(), None);
    remove_segment_files(&path);
}

#[test]
fn raw_recovered_serial_rejects_suffix_mismatch_before_identity_or_segment_mutation() {
    let path = test_wal_path("identity-recovered-suffix-mismatch");
    let identity = identity_test_value(28);
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(canonical_identity_test_record(1, identity));
        wal.flush_all().expect("initial canonical flush");
    }
    let recovery = recover_wal_segment(&path).expect("recover canonical history");
    let segment_before = fs::read(&path).expect("snapshot live segment");
    fs::remove_file(durable_identity_path(&path)).expect("remove identity anchor");

    let error = WalBuffer::with_recovered_durable_segment(
        &path,
        vec![canonical_identity_test_record(2, identity)],
        &recovery,
    )
    .expect_err("release recovery must reject a mismatched suffix before touching durable state");
    assert!(error.to_string().contains("exact recovered segment suffix"));
    assert_eq!(fs::read(&path).unwrap(), segment_before);
    assert_eq!(read_durable_identity(&path).unwrap(), None);
    remove_segment_files(&path);
}

#[test]
fn recovered_bound_constructor_rejects_sidecar_history_substitution() {
    let path = test_wal_path("identity-recovered-substitution");
    let anchor_identity = identity_test_value(28);
    let record_identity = identity_test_value(31);
    write_durable_identity(&path, anchor_identity).expect("write anchor");
    let error = WalBuffer::with_recovered_durable_segment_bound_to_identity(
        &path,
        vec![canonical_identity_test_record(1, record_identity)],
        &WalSegmentRecovery::empty(),
        anchor_identity,
    )
    .expect_err("sidecar A plus recovered canonical record B must reject");
    assert!(error.to_string().contains("do not match"));
    assert!(
        !path.exists(),
        "rejected construction must not create a WAL segment"
    );
    remove_segment_files(&path);
}

#[test]
fn durable_flush_appends_only_the_unflushed_tail() {
    // D1: the k-th commit costs O(its own bytes), not O(all bytes ever written). The already-
    // durable prefix must be byte-identical after later flushes, and each flush must grow the
    // file by exactly the new records' serialized size.
    let path = test_wal_path("durable-append-only");
    let mut wal = WalBuffer::with_durable_segment(&path);

    let first = WalRecord {
        txn_id: 1,
        payload: b"a large first record payload".to_vec().into(),
    };
    wal.append(first.clone());
    wal.flush_all().unwrap();
    // W4a: the physical file is PREALLOCATED (zero tail); the logical watermark is the
    // durable length, and the logical prefix must stay byte-identical across flushes.
    let first_logical = WAL_SEGMENT_MAGIC.len() as u64 + encoded_record_len(&first);
    assert_eq!(wal.durable_segment_bytes(), first_logical);
    let after_first = fs::read(&path).unwrap()[..first_logical as usize].to_vec();

    let second = WalRecord {
        txn_id: 2,
        payload: b"b".to_vec().into(),
    };
    wal.append(second.clone());
    wal.flush_all().unwrap();
    let second_logical = first_logical + encoded_record_len(&second);
    assert_eq!(wal.durable_segment_bytes(), second_logical);
    let after_second = fs::read(&path).unwrap()[..second_logical as usize].to_vec();
    remove_segment_files(&path);
    assert_eq!(&after_second[..after_first.len()], &after_first[..]);
    // The zero tail past the watermark reads back as clean end-of-log.
}

#[test]
fn recover_truncates_torn_tail_beyond_recorded_offset_and_appends_continue() {
    // A crash mid-append leaves a torn record BEYOND the recorded durable tail: recovery
    // truncates it (that commit was never acknowledged) and the segment keeps accepting
    // appends at the valid boundary.
    let path = test_wal_path("recover-torn-tail");
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        for txn_id in 1..=2 {
            wal.append(WalRecord {
                txn_id,
                payload: format!("record {txn_id}").into_bytes().into(),
            });
            wal.flush_all().unwrap();
        }
        // Drop records the durable tail offset (clean shutdown).
    }
    // W4a: the physical file carries the preallocated zero tail; the LOGICAL valid length
    // is what recovery must report. Plant the torn garbage AT the logical tail (a real torn
    // append writes positionally there), overwriting the first zero bytes.
    let valid_bytes = recover_wal_segment(&path).unwrap().valid_bytes;
    {
        use std::os::unix::fs::FileExt;
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&[0xAB; 17], valid_bytes).unwrap();
    }

    let recovery = recover_wal_segment(&path).unwrap();
    assert_eq!(recovery.records.len(), 2);
    assert_eq!(recovery.valid_bytes, valid_bytes);
    // Discarded = the torn garbage plus the preallocated zero tail behind it.
    assert!(recovery.discarded_torn_bytes >= 17);

    let mut wal =
        WalBuffer::with_recovered_durable_segment(&path, recovery.records.clone(), &recovery)
            .unwrap();
    assert_eq!(wal.flushed_count(), 2);
    wal.append(WalRecord {
        txn_id: 3,
        payload: b"post-recovery".to_vec().into(),
    });
    wal.flush_all().unwrap();
    drop(wal);

    // The torn bytes are gone and the post-recovery append reads back strictly.
    let records = read_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(
        records.iter().map(|r| r.txn_id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn recover_rejects_corruption_below_recorded_tail_offset() {
    // Damage to a record BELOW the recorded durable tail is bit rot of acknowledged data —
    // recovery must fail loudly, never silently truncate it away.
    let path = test_wal_path("recover-bit-rot");
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        for txn_id in 1..=2 {
            wal.append(WalRecord {
                txn_id,
                payload: format!("record {txn_id}").into_bytes().into(),
            });
            wal.flush_all().unwrap();
        }
    }
    // Corrupt the FIRST record's payload (well below the recorded tail).
    let mut bytes = fs::read(&path).unwrap();
    bytes[WAL_SEGMENT_MAGIC.len() + WAL_RECORD_HEADER_LEN] ^= 0x01;
    fs::write(&path, bytes).unwrap();

    let err = recover_wal_segment(&path).unwrap_err();
    remove_segment_files(&path);
    assert!(
        err.to_string().contains("checksum mismatch"),
        "expected loud CRC failure for acknowledged-durable corruption, got {err}"
    );
}

#[test]
fn recover_rejects_segment_ending_short_of_recorded_tail() {
    // A segment that ends CLEANLY before the recorded durable tail (external truncation, a
    // lost file) has lost acknowledged records — loud failure, not a silent fresh database.
    let path = test_wal_path("recover-short");
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"acknowledged".to_vec().into(),
        });
        wal.flush_all().unwrap();
    }
    fs::remove_file(&path).unwrap();
    let err = recover_wal_segment(&path).unwrap_err();
    let _ = fs::remove_file(wal_tail_offset_path(&path));
    assert!(
        err.to_string().contains("before the recorded durable tail"),
        "expected loud short-segment failure, got {err}"
    );
}

#[test]
fn recover_without_sidecar_tolerates_any_trailing_invalid_region() {
    // With no recorded tail offset (sidecar lost), the whole trailing invalid region is
    // treated as torn — the safe direction for an advisory lower bound.
    let path = test_wal_path("recover-no-sidecar");
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"kept".to_vec().into(),
        });
        wal.flush_all().unwrap();
    }
    // W4a: plant the garbage at the LOGICAL tail (inside the preallocated region).
    let valid_bytes = recover_wal_segment(&path).unwrap().valid_bytes;
    {
        use std::os::unix::fs::FileExt;
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&[0xCD; 5], valid_bytes).unwrap();
    }
    fs::remove_file(wal_tail_offset_path(&path)).unwrap();

    let recovery = recover_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(recovery.records.len(), 1);
    assert_eq!(recovery.valid_bytes, valid_bytes);
    assert!(recovery.discarded_torn_bytes >= 5);
}

#[test]
fn w4a_rotation_then_frontier_crossing_group_survives_reopen() {
    // AUDIT 9d6e9f96 BLOCKER regression: after a checkpoint prefix-truncation (rotation),
    // the reopened handle must write POSITIONALLY (not O_APPEND, whose pwrite ignores the
    // offset on Linux) and the preallocation frontier must be re-established — a
    // frontier-crossing group after rotation previously stranded acknowledged records
    // behind a zero hole (loud startup rejection after clean shutdown; silent loss after
    // a crash).
    let path = test_wal_path("w4a-rotation-crossing");
    {
        let mut wal = WalBuffer::with_durable_segment(&path);
        for txn_id in 1..=3 {
            wal.append(WalRecord {
                txn_id,
                payload: format!("pre-rotation {txn_id}").into_bytes().into(),
            });
            wal.flush_all().unwrap();
        }
        // Rotate: records [0,2) move to a (simulated) checkpoint; the live file keeps [2,3).
        wal.truncate_durable_segment_prefix(2).unwrap();
        // A group LARGER than the preallocation chunk forces the extension arm on the
        // post-rotation handle — the exact interaction the blocker corrupted.
        let big = vec![0xBB_u8; (wal_prealloc_chunk_bytes() + 256 * 1024) as usize];
        wal.append(WalRecord {
            txn_id: 4,
            payload: big.into(),
        });
        wal.flush_all().unwrap();
        wal.append(WalRecord {
            txn_id: 5,
            payload: b"after crossing".to_vec().into(),
        });
        wal.flush_all().unwrap();
        // Drop = clean shutdown (records the tail-offset sidecar).
    }
    // Reopen: every post-rotation record must be present and the segment clean.
    let recovery = recover_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(recovery.discarded_torn_bytes, 0);
    let txns: Vec<u64> = recovery.records.iter().map(|r| r.txn_id).collect();
    assert_eq!(txns, vec![3, 4, 5]);
}

#[test]
fn w4a_zero_header_is_never_a_valid_record() {
    // The preallocated zero tail is end-of-log to BOTH readers only because an all-zero
    // 24-byte header is unrepresentable: a real record with txn_id=0 and payload_len=0
    // would carry the FNV checksum of the zero header, which must never itself be 0.
    assert_ne!(
        wal_record_checksum(0, 0, &[]),
        0,
        "FNV checksum of a zero header must be nonzero or zero-tail detection is unsound"
    );
}

#[test]
fn w4a_preallocation_keeps_physical_size_stable_across_flushes() {
    // The whole point of W4a: per-flush fdatasync must not grow the file (size-change
    // journaling is what cost 3x on fdatasync latency).
    let path = test_wal_path("w4a-stable-size");
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"first".to_vec().into(),
    });
    wal.flush_all().unwrap();
    let physical_after_first = fs::metadata(&path).unwrap().len();
    for txn_id in 2..=50 {
        wal.append(WalRecord {
            txn_id,
            payload: format!("record {txn_id}").into_bytes().into(),
        });
        wal.flush_all().unwrap();
    }
    let physical_after_fifty = fs::metadata(&path).unwrap().len();
    assert_eq!(
        physical_after_first, physical_after_fifty,
        "flushes inside the preallocated window must not change the physical size"
    );
    drop(wal);
    // Clean recovery: the zero tail reads as a clean end (no torn bytes).
    let recovery = recover_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(recovery.records.len(), 50);
    assert_eq!(recovery.discarded_torn_bytes, 0);
}

#[test]
fn w4a_growth_crosses_the_prealloc_chunk_inline_flush() {
    // A record larger than the preallocation chunk forces the INLINE flush's extension arm;
    // everything must stay readable and recoverable across the boundary.
    let path = test_wal_path("w4a-growth-inline");
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"small before growth".to_vec().into(),
    });
    wal.flush_all().unwrap();
    let big = vec![0xEE_u8; (wal_prealloc_chunk_bytes() + 1024 * 1024) as usize];
    wal.append(WalRecord {
        txn_id: 2,
        payload: big.clone().into(),
    });
    wal.flush_all().unwrap();
    wal.append(WalRecord {
        txn_id: 3,
        payload: b"small after growth".to_vec().into(),
    });
    wal.flush_all().unwrap();
    drop(wal);
    let recovery = recover_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(recovery.records.len(), 3);
    assert_eq!(recovery.discarded_torn_bytes, 0);
    assert_eq!(&recovery.records[1].payload[..], &big[..]);
}

#[test]
fn w4a_growth_crosses_the_prealloc_chunk_group_job() {
    // Same boundary crossing through the GROUP flush job (the concurrent path's flusher).
    let path = test_wal_path("w4a-growth-job");
    let mut wal = WalBuffer::with_durable_segment(&path);
    let big = vec![0xDD_u8; (wal_prealloc_chunk_bytes() + 512 * 1024) as usize];
    wal.append(WalRecord {
        txn_id: 1,
        payload: big.clone().into(),
    });
    let begun = wal.begin_group_flush().unwrap();
    let flushed = match begun {
        WalGroupFlushBegin::Job(job) => job.commit().unwrap(),
        WalGroupFlushBegin::Clean { flushed_records } => flushed_records,
        WalGroupFlushBegin::Busy => panic!("single test flusher must own an available descriptor"),
    };
    assert_eq!(flushed, 1);
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"after job growth".to_vec().into(),
    });
    wal.flush_all().unwrap();
    drop(wal);
    let recovery = recover_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(recovery.records.len(), 2);
    assert_eq!(recovery.discarded_torn_bytes, 0);
    assert_eq!(&recovery.records[0].payload[..], &big[..]);
}

#[test]
fn truncate_durable_segment_prefix_bounds_live_file_and_keeps_appending() {
    // D2: after a checkpoint, the live segment drops the checkpointed prefix; logical
    // counters are unchanged and appends continue against the trimmed file.
    let path = test_wal_path("truncate-prefix");
    let mut wal = WalBuffer::with_durable_segment(&path);
    for txn_id in 1..=3 {
        wal.append(WalRecord {
            txn_id,
            payload: format!("record {txn_id}").into_bytes().into(),
        });
        wal.flush_all().unwrap();
    }
    let full_bytes = wal.durable_segment_bytes();

    wal.truncate_durable_segment_prefix(3).unwrap();
    assert_eq!(wal.durable_segment_bytes(), WAL_SEGMENT_MAGIC.len() as u64);
    assert!(wal.durable_segment_bytes() < full_bytes);
    assert_eq!(wal.durable_segment_base_records(), 3);
    // Logical counters are untouched — only the FILE was trimmed.
    assert_eq!(wal.len(), 3);
    assert_eq!(wal.flushed_count(), 3);
    assert_eq!(wal.flushed_records().len(), 3);

    wal.append(WalRecord {
        txn_id: 4,
        payload: b"post-checkpoint".to_vec().into(),
    });
    wal.flush_all().unwrap();
    drop(wal);

    let live = read_wal_segment(&path).unwrap();
    remove_segment_files(&path);
    assert_eq!(
        live.iter().map(|r| r.txn_id).collect::<Vec<_>>(),
        vec![4],
        "the live segment holds only post-checkpoint records"
    );
}

#[test]
fn bound_identity_anchor_loss_rejects_empty_live_segment_rotation_without_rewrite() {
    let path = test_wal_path("identity-rotation-anchor-loss");
    let identity = identity_test_value(29);
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(canonical_identity_test_record(1, identity));
    wal.flush_all().expect("initial canonical flush");
    let segment_before = fs::read(&path).expect("snapshot live segment");

    fs::remove_file(durable_identity_path(&path)).expect("remove identity anchor");
    let error = wal
        .truncate_durable_segment_prefix(1)
        .expect_err("bound empty-suffix rotation must require its anchor before rewrite");
    assert!(error.to_string().contains("anchor is missing"));
    assert_eq!(wal.flushed_count(), 1);
    assert_eq!(wal.durable_segment_base_records(), 0);
    assert_eq!(fs::read(&path).unwrap(), segment_before);
    assert_eq!(read_durable_identity(&path).unwrap(), None);
    drop(wal);
    remove_segment_files(&path);
}

#[test]
fn bound_identity_anchor_loss_rejects_nonempty_live_segment_rotation_without_rewrite() {
    let path = test_wal_path("identity-rotation-retained-anchor-loss");
    let identity = identity_test_value(30);
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(canonical_identity_test_record(1, identity));
    wal.flush_all().expect("first canonical flush");
    wal.append(canonical_identity_test_record(2, identity));
    wal.flush_all().expect("second canonical flush");
    let segment_before = fs::read(&path).expect("snapshot live segment");

    fs::remove_file(durable_identity_path(&path)).expect("remove identity anchor");
    let error = wal
        .truncate_durable_segment_prefix(1)
        .expect_err("bound retained canonical rotation must require its anchor before rewrite");
    assert!(error.to_string().contains("anchor is missing"));
    assert_eq!(wal.flushed_count(), 2);
    assert_eq!(wal.durable_segment_base_records(), 0);
    assert_eq!(fs::read(&path).unwrap(), segment_before);
    assert_eq!(read_durable_identity(&path).unwrap(), None);

    write_durable_identity(&path, identity).expect("restore identity anchor");
    wal.truncate_durable_segment_prefix(1)
        .expect("restored anchor permits the retained-suffix rewrite");
    assert_eq!(wal.durable_segment_base_records(), 1);
    assert_eq!(read_wal_segment(&path).unwrap().len(), 1);
    drop(wal);
    remove_segment_files(&path);
}

#[test]
fn durable_group_commit_stats_count_one_group_per_fsync() {
    let path = test_wal_path("durable-groups");
    let mut wal = WalBuffer::with_durable_segment(&path);

    // Two records, then ONE flush => a single group of size 2.
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"a".to_vec().into(),
    });
    wal.append(WalRecord {
        txn_id: 2,
        payload: b"b".to_vec().into(),
    });
    wal.flush_all().unwrap();
    // One more record, separate flush => a second group of size 1.
    wal.append(WalRecord {
        txn_id: 3,
        payload: b"c".to_vec().into(),
    });
    wal.flush_all().unwrap();
    // A flush with nothing new must NOT count as a group (no fsync performed).
    wal.flush_all().unwrap();

    let _ = fs::remove_file(&path);
    let stats = wal.group_commit_stats();
    assert_eq!(stats.flush_groups, 2);
    assert_eq!(stats.durable_records, 3);
    assert_eq!(stats.max_group_size, 2);
    assert!((stats.mean_group_size() - 1.5).abs() < f64::EPSILON);
}

#[test]
fn durable_flush_failure_leaves_no_durable_advance() {
    let path = test_wal_path("durable-fail");
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"a".to_vec().into(),
    });

    wal.fail_next_flush();
    let err = wal.flush_all().unwrap_err();
    assert!(matches!(err, EngineError::Durability(_)));
    // The watermark did not advance and (because the simulated failure short-circuits before
    // any I/O) the segment was never created — nothing partially durable.
    assert_eq!(wal.flushed_count(), 0);
    assert!(!path.exists());

    // A subsequent successful flush makes the record durable.
    wal.flush_all().unwrap();
    let recovered = read_wal_segment(&path).unwrap();
    let _ = fs::remove_file(&path);
    assert_eq!(recovered.len(), 1);
    assert_eq!(wal.flushed_count(), 1);
}

#[test]
fn in_memory_flush_writes_no_segment() {
    // The default buffer is in-memory only: flush advances the watermark but touches no disk.
    let mut wal = WalBuffer::new();
    assert!(!wal.is_durable());
    wal.append(WalRecord {
        txn_id: 1,
        payload: b"a".to_vec().into(),
    });
    wal.flush_all().unwrap();
    assert_eq!(wal.flushed_count(), 1);
    assert_eq!(wal.group_commit_stats(), WalGroupCommitStats::default());
}

#[test]
fn reinstate_durable_records_seeds_flushed_prefix() {
    let mut wal = WalBuffer::new();
    wal.reinstate_durable_records(vec![
        WalRecord {
            txn_id: 1,
            payload: b"a".to_vec().into(),
        },
        WalRecord {
            txn_id: 2,
            payload: b"b".to_vec().into(),
        },
    ]);
    assert_eq!(wal.len(), 2);
    assert_eq!(wal.flushed_count(), 2);
    assert_eq!(wal.unflushed_count(), 0);
}

#[test]
fn wal_segment_round_trips_durable_records() {
    let path = test_wal_path("roundtrip");
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

    write_wal_segment(&path, &records).unwrap();
    let recovered = read_wal_segment(&path).unwrap();
    let _ = fs::remove_file(path);

    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0].txn_id, 1);
    assert_eq!(recovered[0].payload, records[0].payload);
    assert_eq!(recovered[1].txn_id, 2);
    assert_eq!(recovered[1].payload, records[1].payload);
}

#[test]
fn wal_segment_rejects_checksum_mismatch() {
    let path = test_wal_path("checksum");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    }];

    write_wal_segment(&path, &records).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.last_mut().unwrap();
    *last ^= 0x01;
    fs::write(&path, bytes).unwrap();

    let err = read_wal_segment(&path).unwrap_err();
    let _ = fs::remove_file(path);

    assert!(err.to_string().contains("checksum mismatch"));
}

#[test]
fn wal_segment_rejects_truncated_record_header() {
    let path = test_wal_path("truncated");
    fs::write(
        &path,
        [WAL_SEGMENT_MAGIC.as_slice(), &[1_u8, 2, 3]].concat(),
    )
    .unwrap();

    let err = read_wal_segment(&path).unwrap_err();
    let _ = fs::remove_file(path);

    assert!(err.to_string().contains("record header"));
}

#[test]
fn wal_control_file_round_trips_checkpoint_metadata() {
    let control_path = test_wal_path("control").with_extension("control");
    let control = WalControlFile {
        segment_path: PathBuf::from("segment-0001.wal"),
        checkpoint: WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(42),
        },
    };

    write_wal_control_file(&control_path, &control).unwrap();
    let recovered = read_wal_control_file(&control_path).unwrap();
    let _ = fs::remove_file(control_path);

    assert_eq!(recovered, control);
}

#[test]
fn v2_control_and_lanes_checkpoint_sidecars_fail_closed_on_tamper_or_truncation() {
    let base = test_wal_path("checksummed-sidecars");
    let control_path = base.with_extension("control");
    let control = WalControlFile {
        segment_path: PathBuf::from("segment-0001.wal"),
        checkpoint: WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(42),
        },
    };
    write_wal_control_file(&control_path, &control).unwrap();
    let original = fs::read_to_string(&control_path).unwrap();
    assert!(original.starts_with("GPUDBWALCONTROL2\n"));
    let legacy = original
        .replace("GPUDBWALCONTROL2", "GPUDBWALCONTROL1")
        .lines()
        .filter(|line| !line.starts_with("sha256="))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&control_path, legacy).unwrap();
    assert_eq!(read_wal_control_file(&control_path).unwrap(), control);
    fs::write(
        &control_path,
        original.replace("durable_record_count=2", "durable_record_count=3"),
    )
    .unwrap();
    assert!(read_wal_control_file(&control_path).is_err());
    let without_trailer = original
        .lines()
        .filter(|line| !line.starts_with("sha256="))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&control_path, without_trailer).unwrap();
    assert!(read_wal_control_file(&control_path).is_err());

    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET sidecar=1".to_vec().into(),
    }];
    write_lanes_checkpoint(&base, 1, 0, &records).unwrap();
    let sidecar = lanes_checkpoint_sidecar_path(&base);
    let original = fs::read_to_string(&sidecar).unwrap();
    assert!(original.starts_with("GPUDBLANESCHECKPOINT2\n"));
    let segment_name = lanes_checkpoint_segment_path(&base, 0)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    fs::write(
        &sidecar,
        format!("gpu-db-lanes-checkpoint v1 1 0 {segment_name}\n"),
    )
    .unwrap();
    assert!(read_lanes_checkpoint(&base).unwrap().is_some());
    fs::write(
        &sidecar,
        original.replace("serial_records=1", "serial_records=2"),
    )
    .unwrap();
    assert!(read_lanes_checkpoint(&base).is_err());

    let _ = fs::remove_file(control_path);
    let _ = fs::remove_file(sidecar);
    let _ = fs::remove_file(lanes_checkpoint_segment_path(&base, 0));
}

#[test]
fn wal_checkpoint_reads_segment_named_by_control_file() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-checkpoint-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("CONTROL");
    let segment_path = dir.join("segment-0001.wal");
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
    let control = WalControlFile {
        segment_path: PathBuf::from("segment-0001.wal"),
        checkpoint: WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(2),
        },
    };

    write_wal_segment(&segment_path, &records).unwrap();
    write_wal_control_file(&control_path, &control).unwrap();
    let (recovered_control, recovered_records) = read_wal_checkpoint(&control_path).unwrap();
    let _ = fs::remove_dir_all(dir);

    assert_eq!(recovered_control, control);
    assert_eq!(recovered_records.len(), 2);
    assert_eq!(&recovered_records[1].payload[..], &b"SET b=2"[..]);
}

#[test]
fn wal_checkpoint_rejects_control_record_count_mismatch() {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-wal-checkpoint-mismatch-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let control_path = dir.join("CONTROL");
    let segment_path = dir.join("segment-0001.wal");
    let records = vec![WalRecord {
        txn_id: 1,
        payload: b"SET a=1".to_vec().into(),
    }];
    let control = WalControlFile {
        segment_path: PathBuf::from("segment-0001.wal"),
        checkpoint: WalCheckpointMeta {
            durable_record_count: 2,
            last_durable_txn_id: Some(1),
        },
    };

    write_wal_segment(&segment_path, &records).unwrap();
    write_wal_control_file(&control_path, &control).unwrap();
    let err = read_wal_checkpoint(&control_path).unwrap_err();
    let _ = fs::remove_dir_all(dir);

    assert!(err.to_string().contains("expected 2 durable records"));
}
