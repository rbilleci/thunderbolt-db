use super::{gpu_available, select};
use crate::{relational_model::RelationalResidencySnapshot, Engine};
use gpu_db_sql::SqlValue;
use gpu_db_wal::WalBuffer;

// ===================== P1 (sealed-shards-primary): the DURABLE cold checkpoint =====================

/// Pure encode/decode round-trip of the chunk descriptor (no GPU): every persisted field survives;
/// the transient bookkeeping (memory proof, refresh cost, invalidation) restores to its defaults.
#[test]
fn cold_checkpoint_descriptor_round_trips() {
    let descriptor = RelationalResidencySnapshot {
        gpu_id: 3,
        schema: "public".into(),
        table: "t".into(),
        generation: 41,
        row_count: 12,
        capacity: 12,
        column_count: 4,
        resident_bytes: 4096,
        resident_device_int4_columns: vec!["a".into(), "b".into()],
        resident_device_int4_column_stats: vec![
            crate::relational_model::ResidentDeviceInt4ColumnStats {
                name: "a".into(),
                min: -7,
                max: 900,
            },
        ],
        resident_device_int8_columns: vec!["big".into()],
        resident_device_numeric_columns: vec!["price".into()],
        resident_device_bool_columns: vec![
            crate::relational_model::ResidentDeviceBoolColumnLayout {
                name: "flag".into(),
                bitmap_byte_offset: 128,
            },
        ],
        resident_device_text_columns: vec![
            crate::relational_model::ResidentDeviceTextColumnLayout {
                name: "name".into(),
                offsets_byte_offset: 256,
                bytes_byte_offset: 304,
                bytes_len: 77,
            },
        ],
        resident_device_null_columns: vec![
            crate::relational_model::ResidentDeviceNullBitmapLayout {
                name: "b".into(),
                bitmap_byte_offset: 512,
            },
        ],
        valid_through_index: 99,
        invalidated_by_txn_id: None,
        invalidated_at_index: None,
        invalidated_by_memory_pressure: false,
        memory_pressure_active: false,
        last_refresh_cost: None,
        admission_budget_bytes: None,
        resident_bytes_after_admission: 4096,
        evicted_tables_on_admission: Vec::new(),
        device_memory_proof: None,
    };
    let mut w = crate::engine_streaming_exec::ColdCkptWriter {
        inner: Vec::<u8>::new(),
        hash: crate::engine_streaming_exec::FNV_OFFSET,
    };
    crate::engine_streaming_exec::encode_cold_descriptor(&mut w, &descriptor).unwrap();
    let mut r = crate::engine_streaming_exec::ColdCkptReader {
        inner: std::io::Cursor::new(w.inner),
    };
    let decoded = crate::engine_streaming_exec::decode_cold_descriptor(&mut r).unwrap();
    assert_eq!(decoded, descriptor);
}

/// Shared P1 harness: a lanes-mode durable database whose table `t` (plain 2-col int4, NO PK —
/// keeps the shape elision-ineligible so the streaming scan's store premise holds) has
/// `serial_rows` rows from the serial (pre-activation) phase and 24 fabricated lane-commit rows.
/// Returns (wal base path, expected row count, next fabricated row id base, next lane seq).
fn p1_lanes_streaming_fixture(
    tag: &str,
    serial_rows: i32,
) -> Option<(std::path::PathBuf, i64, u64)> {
    let dir = std::env::temp_dir().join(format!(
        "gpu-db-cold-ckpt-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("db.wal");
    let row_base = {
        let mut e = Engine::new_local_cpu_oracle();
        e.commit_state_mut().wal = WalBuffer::with_durable_segment(&base);
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return None; // off-box: skip
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE t (a INT, b INT)")
            .unwrap();
        let mut values = String::new();
        for i in 0..serial_rows {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {})", i * 2));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO t (a, b) VALUES {values}"))
            .unwrap();
        e.read_state.mvcc.current_row_id()
    };
    // Fabricated 2-lane history: 24 one-row binary INSERT commits (the recovery-suite pattern —
    // no live post-activation writes are needed, so the test never trips the classic-write guard).
    let tiny = 16 << 10;
    {
        let set = gpu_db_wal::FuaWalLaneSet::create(&base, 2, 2, tiny).expect("create lanes");
        for seq in 0..24u64 {
            let values = vec![SqlValue::Int4(10_000 + seq as i32), SqlValue::Int4(0)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(row_base + seq, values.as_slice())],
            )
            .expect("binary encode");
            set.append(
                (seq % 2) as usize,
                seq,
                &[gpu_db_wal::WalRecord {
                    txn_id: 500 + seq,
                    payload: payload.into(),
                }],
            )
            .expect("append");
        }
        set.wait_durable(24).expect("durable");
    }
    Some((base, i64::from(serial_rows) + 24, row_base + 24))
}

fn p1_count(e: &Engine) -> i64 {
    let q = select("SELECT COUNT(*) FROM t");
    match e.execute_relational_select(&q).unwrap().rows.row(0)[0] {
        SqlValue::Int8(n) => n,
        ref other => panic!("COUNT returned {other:?}"),
    }
}

/// Recovery now eagerly installs a GPU-resident snapshot. These tests exercise the distinct over-budget
/// streaming/cold-checkpoint path, so explicitly evict that snapshot after setting the tiny budget. The cold
/// tier is separate ownership and deliberately survives this resident-cache eviction across reopen.
fn p1_force_streaming(e: &mut Engine, budget: u64) {
    e.set_relational_residency_budget_bytes(0, budget);
    let catalog = e.ddl_catalog();
    catalog.relational_resident_cache.remove_table(
        "t",
        &e.read_state.residency,
        &e.read_state.route_telemetry,
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_restores_streaming_cold_across_reopen() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("restore", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        assert!(
            e.streaming_fold_hits() >= 1 && e.streaming_cold_builds() >= 1,
            "premise: the read streamed and captured the cold tier (fold {}, builds {})",
            e.streaming_fold_hits(),
            e.streaming_cold_builds()
        );
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(
            e.streaming_cold_checkpointed() >= 1,
            "the lanes checkpoint must persist the quiesced cold tier"
        );
        cut
    };
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    assert!(
        artifact.exists(),
        "artifact {} must exist",
        artifact.display()
    );

    // REOPEN: the seam install restores the cold tier; the first streaming read is a byte REPLAY
    // (a HIT with zero fresh scan-builds), and the answer matches.
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after checkpoint");
    assert!(
        e.streaming_cold_restored() >= 1,
        "reopen must restore the cold tier from the checkpoint artifact"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(p1_count(&e), expected);
    assert!(
        e.streaming_cold_hits() >= 1,
        "the restored entry must serve the first streaming read"
    );
    assert_eq!(
        e.streaming_cold_builds(),
        0,
        "no fresh scan-build: the restore IS the build"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_patches_forward_post_checkpoint_wal_suffix() {
    let Some((base, expected, next_row)) = p1_lanes_streaming_fixture("suffix", 300) else {
        return;
    };
    let budget = 512u64;
    {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(e.streaming_cold_checkpointed() >= 1);
    }
    // Post-checkpoint WAL SUFFIX: 6 more fabricated lane commits ABOVE the checkpoint cut.
    let tiny = 16 << 10;
    {
        let set = gpu_db_wal::FuaWalLaneSet::reopen_from(&base, 2, 2, tiny, 24)
            .expect("wal-level reopen from baseline");
        for seq in 24..30u64 {
            let values = vec![SqlValue::Int4(20_000 + seq as i32), SqlValue::Int4(1)];
            let payload = crate::wal_binary::try_encode_binary_insert(
                "t",
                &[(next_row + (seq - 24), values.as_slice())],
            )
            .expect("binary encode");
            set.append(
                (seq % 2) as usize,
                seq,
                &[gpu_db_wal::WalRecord {
                    txn_id: 900 + seq,
                    payload: payload.into(),
                }],
            )
            .expect("append");
        }
        set.wait_durable(30).expect("durable");
    }
    // REOPEN: restore at the seam, then the 6-record suffix replays THROUGH the restored entry —
    // the 6c-3 commit hooks patch it forward (the WAL suffix IS the delta stream). The first
    // streaming read replays patched bytes and sees the suffix rows.
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after suffix");
    assert!(
        e.streaming_cold_restored() >= 1,
        "the seam restore must land before the suffix replays"
    );
    assert!(
        e.streaming_cold_patches() >= 1,
        "suffix replay must patch the restored entry via the commit hooks"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(
        p1_count(&e),
        expected + 6,
        "the suffix rows must be visible"
    );
    assert!(e.streaming_cold_hits() >= 1);
    assert_eq!(
        e.streaming_cold_builds(),
        0,
        "restore + patches carried the entry — no fresh scan-build"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_corrupt_artifact_is_skipped_never_wrong() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("corrupt", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert!(e.streaming_cold_checkpointed() >= 1);
        cut
    };
    // Flip one byte in the artifact BODY (past the magic): the FNV trailer must reject it.
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    let mut bytes = std::fs::read(&artifact).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    std::fs::write(&artifact, &bytes).unwrap();

    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen with corrupt artifact");
    assert_eq!(
        e.streaming_cold_restored(),
        0,
        "a checksum-failed artifact must restore NOTHING"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(
        p1_count(&e),
        expected,
        "the read rebuilds from the store — never wrong"
    );
    assert!(
        e.streaming_cold_builds() >= 1,
        "the skipped restore leaves the first read to scan + capture"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_boundary_mismatch_is_skipped() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("boundary", 300) else {
        return;
    };
    let budget = 512u64;
    let cut = {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert!(e.streaming_cold_checkpointed() >= 1);
        cut
    };
    // Tamper the artifact's BOUNDARY field (u64 right after the magic) and RECOMPUTE the FNV
    // trailer — a checksum-valid artifact whose boundary does not match the replay seam. The
    // strict-equality guard must skip it (installing would replay bytes from the WRONG commit
    // index — the one guard corruption cannot exercise).
    let artifact = base.with_file_name(format!(
        "{}.cold-checkpoint.{cut}",
        base.file_name().unwrap().to_string_lossy()
    ));
    let mut bytes = std::fs::read(&artifact).unwrap();
    let magic_len = b"GPUDBCOLDCKPT1\n".len();
    let boundary = u64::from_le_bytes(bytes[magic_len..magic_len + 8].try_into().unwrap());
    bytes[magic_len..magic_len + 8].copy_from_slice(&(boundary + 1).to_le_bytes());
    let body_len = bytes.len() - 8;
    let mut hash = crate::engine_streaming_exec::FNV_OFFSET;
    for b in &bytes[..body_len] {
        hash = (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3);
    }
    bytes[body_len..].copy_from_slice(&hash.to_le_bytes());
    std::fs::write(&artifact, &bytes).unwrap();

    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen with tampered boundary");
    assert_eq!(
        e.streaming_cold_restored(),
        0,
        "a boundary-mismatched artifact must restore NOTHING (checksum alone cannot catch it)"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(
        p1_count(&e),
        expected,
        "the read rebuilds from the store — never wrong"
    );
    assert!(e.streaming_cold_builds() >= 1);
}

/// AUDIT HIGH regression (the boundary convention): the LIVE lane pump publishes committed_seq
/// as the EXCLUSIVE frontier (`visible_global_cut = base_seq + cut`), while the recovery seam's
/// replay publishes the INCLUSIVE last record index (`base_seq + cut - 1`) — one less. The
/// pre-fix code stamped the artifact with the live watermark verbatim, so every artifact captured
/// from a pump-published engine carried a boundary ONE HIGH and the restore silently never fired
/// in production (the four sibling tests replay-derive their watermark on BOTH sides, so they
/// cannot see it). This test emulates the pump's convention exactly — it re-publishes the
/// watermark at the frontier before checkpointing — and requires the restore to land anyway.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_cold_checkpoint_restores_under_lane_pump_frontier_watermark() {
    let Some((base, expected, _next_row)) = p1_lanes_streaming_fixture("frontier", 300) else {
        return;
    };
    let budget = 512u64;
    {
        let mut e = Engine::open_durable_wal_segment(&base).expect("lanes reopen");
        p1_force_streaming(&mut e, budget);
        assert_eq!(p1_count(&e), expected);
        assert!(e.streaming_cold_builds() >= 1);
        // Emulate the pump: publish the EXCLUSIVE frontier (base_seq + cut), the value
        // engine_dml_concurrent's settle path publishes after a quiesced wave. The visible set
        // is unchanged (no stamp exists at the frontier).
        let lanes = e.intent_lanes.as_ref().expect("lanes installed");
        let frontier = lanes.base_seq.load(std::sync::atomic::Ordering::Acquire) + 24;
        e.publish_committed_seq(frontier);
        assert_eq!(
            e.committed_seq(),
            frontier,
            "premise: frontier-convention watermark"
        );
        let cut = e.checkpoint_intent_lanes().expect("lanes checkpoint");
        assert_eq!(cut, 24);
        assert!(
            e.streaming_cold_checkpointed() >= 1,
            "the frontier watermark must be ACCEPTED as the quiescence proof"
        );
    }
    let mut e = Engine::open_durable_wal_segment(&base).expect("reopen after frontier checkpoint");
    assert!(
        e.streaming_cold_restored() >= 1,
        "the artifact must carry the SEAM boundary (inclusive last index), not the live \
         frontier — a frontier-stamped artifact never restores"
    );
    p1_force_streaming(&mut e, budget);
    assert_eq!(p1_count(&e), expected);
    assert!(e.streaming_cold_hits() >= 1);
    assert_eq!(e.streaming_cold_builds(), 0);
}
