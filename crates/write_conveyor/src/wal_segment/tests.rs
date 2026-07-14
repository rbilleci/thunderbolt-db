use super::*;
use crate::stats_for_range;
use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, Barrier};

fn temp_wal_path(name: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "gpu-db-write-conveyor-wal-{name}-{}-{nonce}.dat",
        std::process::id()
    ))
}

fn temp_wal_dir(name: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "gpu-db-write-conveyor-wal-dir-{name}-{}-{nonce}",
        std::process::id()
    ))
}

fn custom_intent(client_seq: u64) -> WriteIntent {
    WriteIntent {
        route_id: 17,
        flags: 3,
        txn_id: 10_000 + client_seq,
        tenant_id: 900 + (client_seq % 11),
        key: client_seq.wrapping_mul(97).wrapping_add(5),
        value0: client_seq ^ 0x1357_9bdf_2468_ace0,
        value1: client_seq.rotate_left(9) ^ 0xfeed_face_cafe_beef,
        value2: client_seq.reverse_bits(),
        client_seq,
    }
}

fn stats_for_intents(intents: &[WriteIntent]) -> DrainStats {
    let mut stats = DrainStats::default();
    for intent in intents.iter().copied() {
        stats.observe(intent);
    }
    stats
}

#[test]
fn wal_disk_structures_keep_fixed_sizes() {
    assert_eq!(size_of::<WalSegmentFileHeader>(), 64);
    assert_eq!(size_of::<WalManagerControlRecord>(), 128);
    assert_eq!(size_of::<WalBlockHeader>(), 64);
    assert_eq!(size_of::<WalBlockTrailer>(), 64);
    assert_eq!(WAL_SEGMENT_HEADER_BYTES % page_size(), 0);
}

#[test]
fn wal_segment_recovers_synced_prefix() {
    let path = temp_wal_path("prefix");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 7, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 3).unwrap();
        segment.sync_published_prefix(2).unwrap();
    }

    let recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(recovered.segment_id, 7);
    assert_eq!(recovered.recovered_blocks, 2);
    assert_eq!(recovered.recovered_records, 7);
    assert_eq!(recovered.stats, stats_for_range(0, 7));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_segment_recovers_real_payload_block() {
    let path = temp_wal_path("payload");
    let intents: Vec<_> = [101, 205, 309, 401]
        .into_iter()
        .map(custom_intent)
        .collect();
    {
        let segment = unsafe { MappedWalSegment::create(&path, 13, 16, 4).unwrap() };
        segment.try_publish_intents(&intents).unwrap();
        segment.sync_published_prefix(1).unwrap();
    }

    let recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(recovered.segment_id, 13);
    assert_eq!(recovered.recovered_blocks, 1);
    assert_eq!(recovered.recovered_records, intents.len() as u64);
    assert_eq!(recovered.stats, stats_for_intents(&intents));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_segment_supports_page_aligned_frame_record_count() {
    let path = temp_wal_path("aligned-frame");
    assert_eq!(block_stride(62), 4096);
    {
        let segment = unsafe { MappedWalSegment::create(&path, 22, 124, 62).unwrap() };
        segment.try_publish_block(0, 62).unwrap();
        segment.try_publish_block(62, 17).unwrap();
        segment.sync_published_prefix(2).unwrap();
    }

    let recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(recovered.segment_id, 22);
    assert_eq!(recovered.block_size, 62);
    assert_eq!(recovered.recovered_blocks, 2);
    assert_eq!(recovered.recovered_records, 79);
    assert_eq!(recovered.stats, stats_for_range(0, 79));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_recovery_ignores_valid_blocks_beyond_control_tail() {
    let path = temp_wal_path("control-tail");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 9, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment.sync_published_prefix(1).unwrap();
    }

    let recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(recovered.recovered_blocks, 1);
    assert_eq!(recovered.recovered_records, 4);
    assert_eq!(recovered.stats, stats_for_range(0, 4));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_sync_prefix_is_monotonic() {
    let path = temp_wal_path("monotonic");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 10, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment.sync_published_prefix(2).unwrap();
        segment.sync_published_prefix(1).unwrap();
        assert_eq!(segment.durable_blocks(), 2);
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_sync_prefix_supports_single_shared_flusher() {
    let path = temp_wal_path("shared-sync");
    {
        let segment = Arc::new(unsafe { MappedWalSegment::create(&path, 15, 16, 4).unwrap() });
        for block in 0..4 {
            segment.try_publish_block(block * 4, 4).unwrap();
        }
        let start = Arc::new(Barrier::new(5));
        let mut handles = Vec::new();
        for prefix in 1..=4 {
            let segment = Arc::clone(&segment);
            let start = Arc::clone(&start);
            handles.push(std::thread::spawn(move || {
                start.wait();
                segment.sync_published_prefix(prefix).unwrap();
            }));
        }
        start.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(segment.durable_blocks(), 4);
    }

    let recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(recovered.recovered_blocks, 4);
    assert_eq!(recovered.recovered_records, 16);
    assert_eq!(recovered.stats, stats_for_range(0, 16));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_scan_recovery_observes_data_synced_prefix_without_control_tail() {
    let path = temp_wal_path("scan-data");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 16, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment.sync_published_data_frontier(2).unwrap();
    }

    let control_recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(control_recovered.recovered_blocks, 0);
    let scan_recovered = recover_wal_segment_by_scan(&path).unwrap();
    assert_eq!(scan_recovered.recovered_blocks, 2);
    assert_eq!(scan_recovered.recovered_records, 8);
    assert_eq!(scan_recovered.stats, stats_for_range(0, 8));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_scan_recovery_observes_write_data_synced_prefix() {
    let path = temp_wal_path("scan-write-data");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 18, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment
            .sync_published_data_frontier_with_mode(2, WalDataSyncMode::WriteAndFileData)
            .unwrap();
    }

    let control_recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(control_recovered.recovered_blocks, 0);
    let scan_recovered = recover_wal_segment_by_scan(&path).unwrap();
    assert_eq!(scan_recovered.recovered_blocks, 2);
    assert_eq!(scan_recovered.recovered_records, 8);
    assert_eq!(scan_recovered.stats, stats_for_range(0, 8));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_data_sync_modes_track_upgradeable_frontiers() {
    let path = temp_wal_path("sync-mode-frontiers");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 17, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment.try_publish_block(8, 4).unwrap();
        segment
            .sync_published_data_frontier_with_mode(1, WalDataSyncMode::FileDataOnly)
            .unwrap();
        assert_eq!(segment.data_frontier_blocks.load(Ordering::Acquire), 0);
        assert_eq!(
            segment
                .prewrite_data_frontier_blocks
                .load(Ordering::Acquire),
            0
        );
        assert_eq!(
            segment.write_data_frontier_blocks.load(Ordering::Acquire),
            0
        );
        assert_eq!(segment.file_data_frontier_blocks.load(Ordering::Acquire), 1);

        segment
            .sync_published_data_frontier_with_mode(2, WalDataSyncMode::WriteAndFileData)
            .unwrap();
        assert_eq!(segment.data_frontier_blocks.load(Ordering::Acquire), 0);
        assert_eq!(
            segment
                .prewrite_data_frontier_blocks
                .load(Ordering::Acquire),
            2
        );
        assert_eq!(
            segment.write_data_frontier_blocks.load(Ordering::Acquire),
            2
        );
        assert_eq!(segment.file_data_frontier_blocks.load(Ordering::Acquire), 2);

        segment.sync_published_data_frontier(3).unwrap();
        assert_eq!(segment.data_frontier_blocks.load(Ordering::Acquire), 3);
        assert_eq!(
            segment
                .prewrite_data_frontier_blocks
                .load(Ordering::Acquire),
            3
        );
        assert_eq!(
            segment.write_data_frontier_blocks.load(Ordering::Acquire),
            3
        );
        assert_eq!(segment.file_data_frontier_blocks.load(Ordering::Acquire), 3);
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_prewrite_frontier_waits_for_final_sync() {
    let path = temp_wal_path("prewrite-frontier");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 19, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        assert_eq!(segment.write_published_data_frontier(2).unwrap(), 2);
        assert_eq!(
            segment
                .prewrite_data_frontier_blocks
                .load(Ordering::Acquire),
            2
        );
        assert_eq!(
            segment.write_data_frontier_blocks.load(Ordering::Acquire),
            0
        );
        assert_eq!(segment.file_data_frontier_blocks.load(Ordering::Acquire), 0);

        segment
            .sync_published_data_frontier_with_mode(2, WalDataSyncMode::PrewriteAndFileData)
            .unwrap();
        assert_eq!(
            segment
                .prewrite_data_frontier_blocks
                .load(Ordering::Acquire),
            2
        );
        assert_eq!(
            segment.write_data_frontier_blocks.load(Ordering::Acquire),
            2
        );
        assert_eq!(segment.file_data_frontier_blocks.load(Ordering::Acquire), 2);
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_scan_recovery_observes_prewrite_data_synced_prefix() {
    let path = temp_wal_path("scan-prewrite-data");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 20, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment.write_published_data_frontier(2).unwrap();
        segment
            .sync_published_data_frontier_with_mode(2, WalDataSyncMode::PrewriteAndFileData)
            .unwrap();
    }

    let control_recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(control_recovered.recovered_blocks, 0);
    let scan_recovered = recover_wal_segment_by_scan(&path).unwrap();
    assert_eq!(scan_recovered.recovered_blocks, 2);
    assert_eq!(scan_recovered.recovered_records, 8);
    assert_eq!(scan_recovered.stats, stats_for_range(0, 8));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_scan_recovery_observes_sync_write_data_prefix() {
    let path = temp_wal_path("scan-sync-write-data");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 21, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment
            .sync_published_data_frontier_with_mode(2, WalDataSyncMode::SyncWriteData)
            .unwrap();
        assert_eq!(segment.data_frontier_blocks.load(Ordering::Acquire), 0);
        assert_eq!(
            segment
                .prewrite_data_frontier_blocks
                .load(Ordering::Acquire),
            2
        );
        assert_eq!(
            segment.write_data_frontier_blocks.load(Ordering::Acquire),
            2
        );
        assert_eq!(segment.file_data_frontier_blocks.load(Ordering::Acquire), 2);
    }

    let control_recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(control_recovered.recovered_blocks, 0);
    let scan_recovered = recover_wal_segment_by_scan(&path).unwrap();
    assert_eq!(scan_recovered.recovered_blocks, 2);
    assert_eq!(scan_recovered.recovered_records, 8);
    assert_eq!(scan_recovered.stats, stats_for_range(0, 8));
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_sync_prefix_rejects_bounds_without_panicking() {
    let path = temp_wal_path("sync-bounds");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 10, 16, 4).unwrap() };
        let err = segment
            .sync_published_prefix(segment.block_capacity() + 1)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_segment_empty_payload_publish_is_noop() {
    let path = temp_wal_path("empty-payload");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 14, 16, 4).unwrap() };
        segment.try_publish_intents(&[]).unwrap();
        segment.sync_published_prefix(0).unwrap();
    }

    let recovered = recover_wal_segment(&path).unwrap();
    assert_eq!(recovered.segment_id, 14);
    assert_eq!(recovered.recovered_blocks, 0);
    assert_eq!(recovered.recovered_records, 0);
    assert_eq!(recovered.stats, DrainStats::default());
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_recovery_rejects_corrupt_durable_trailer() {
    let path = temp_wal_path("torn");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 8, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.try_publish_block(4, 4).unwrap();
        segment.sync_published_prefix(2).unwrap();
    }

    let stride = block_stride(4);
    let second_trailer = WAL_SEGMENT_HEADER_BYTES
        + stride
        + size_of::<WalBlockHeader>()
        + 4 * size_of::<WriteIntent>();
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(second_trailer as u64)).unwrap();
    file.write_all(&0_u64.to_ne_bytes()).unwrap();
    file.sync_data().unwrap();

    let err = recover_wal_segment(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_recovery_rejects_unknown_file_header_flags() {
    let path = temp_wal_path("flags");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 11, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.sync_published_prefix(1).unwrap();
    }

    let mut header: WalSegmentFileHeader = {
        let mut file = File::open(&path).unwrap();
        read_struct_at(&mut file, 0).unwrap()
    };
    header.flags = 1;
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(bytes_of(&header)).unwrap();
    file.sync_data().unwrap();

    let err = recover_wal_segment(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_recovery_rejects_file_header_reserved_bits() {
    let path = temp_wal_path("reserved");
    {
        let segment = unsafe { MappedWalSegment::create(&path, 12, 16, 4).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.sync_published_prefix(1).unwrap();
    }

    let mut header: WalSegmentFileHeader = {
        let mut file = File::open(&path).unwrap();
        read_struct_at(&mut file, 0).unwrap()
    };
    header.reserved[0] = 1;
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(bytes_of(&header)).unwrap();
    file.sync_data().unwrap();

    let err = recover_wal_segment(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_file(path);
}

#[test]
fn wal_manager_rolls_and_recovers_across_segments() {
    let dir = temp_wal_dir("roll-recover");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        for block in 0..5 {
            let position = manager.publish_block(block * 4, 4).unwrap();
            assert_eq!(
                position,
                WalPosition {
                    segment_id: block / 2,
                    block_id: block % 2,
                }
            );
        }
        let durable = manager.sync_all_published().unwrap();
        assert_eq!(
            durable,
            WalPosition {
                segment_id: 2,
                block_id: 1,
            }
        );
    }

    let recovered = recover_wal_manager(&config).unwrap();
    assert_eq!(recovered.first_segment_id, 0);
    assert_eq!(recovered.durable_segment_id, 2);
    assert_eq!(recovered.durable_blocks, 1);
    assert_eq!(recovered.recovered_segments, 3);
    assert_eq!(recovered.recovered_records, 20);
    assert_eq!(recovered.stats, stats_for_range(0, 20));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_recovery_caps_segment_tail_to_manager_control() {
    let dir = temp_wal_dir("manager-capped-tail");
    let config = WalSegmentManagerConfig::new(&dir, "events", 16, 4);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        manager.publish_block(0, 4).unwrap();
        manager.sync_all_published().unwrap();
        manager.publish_block(4, 4).unwrap();
        manager.current_segment.sync_published_prefix(2).unwrap();
    }

    let segment_recovered = recover_wal_segment(config.segment_path(0)).unwrap();
    assert_eq!(segment_recovered.recovered_blocks, 2);
    assert_eq!(segment_recovered.recovered_records, 8);

    let manager_recovered = recover_wal_manager(&config).unwrap();
    assert_eq!(manager_recovered.durable_segment_id, 0);
    assert_eq!(manager_recovered.durable_blocks, 1);
    assert_eq!(manager_recovered.recovered_records, 4);
    assert_eq!(manager_recovered.stats, stats_for_range(0, 4));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_rolls_and_recovers_real_payloads() {
    let dir = temp_wal_dir("payload-roll-recover");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    let mut expected = DrainStats::default();
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        for block in 0..5 {
            let intents: Vec<_> = (0..4)
                .map(|offset| custom_intent(1_000 + block * 10 + offset))
                .collect();
            expected.add(stats_for_intents(&intents));
            let position = manager.publish_intents(&intents).unwrap();
            assert_eq!(
                position,
                WalPosition {
                    segment_id: block / 2,
                    block_id: block % 2,
                }
            );
        }
        manager.sync_all_published().unwrap();
    }

    let recovered = recover_wal_manager(&config).unwrap();
    assert_eq!(recovered.first_segment_id, 0);
    assert_eq!(recovered.durable_segment_id, 2);
    assert_eq!(recovered.durable_blocks, 1);
    assert_eq!(recovered.recovered_segments, 3);
    assert_eq!(recovered.recovered_records, 20);
    assert_eq!(recovered.stats, expected);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_handles_read_and_scan_recover_data_fenced_blocks() {
    let dir = temp_wal_dir("handle-scan-recover");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    let mut expected = DrainStats::default();
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        for block in 0..5 {
            let intents: Vec<_> = (0..4)
                .map(|offset| custom_intent(2_000 + block * 10 + offset))
                .collect();
            expected.add(stats_for_intents(&intents));
            let handle = manager.publish_intents_handle(&intents).unwrap();
            assert_eq!(
                handle.position(),
                WalPosition {
                    segment_id: block / 2,
                    block_id: block % 2,
                }
            );
            let mut recovered_payload = Vec::new();
            handle
                .read_published_block_into(&mut recovered_payload)
                .unwrap();
            assert_eq!(recovered_payload, intents);
            handle.sync_published_data_frontier().unwrap();
        }
    }

    let control_recovered = recover_wal_manager(&config).unwrap();
    assert_eq!(control_recovered.recovered_records, 0);
    let scan_recovered = recover_wal_manager_by_scan(&config).unwrap();
    assert_eq!(scan_recovered.recovered_segments, 3);
    assert_eq!(scan_recovered.recovered_records, 20);
    assert_eq!(scan_recovered.stats, expected);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_retention_respects_durable_boundary() {
    let dir = temp_wal_dir("retention");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    let segment0 = config.segment_path(0);
    let segment1 = config.segment_path(1);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        for block in 0..3 {
            manager.publish_block(block * 4, 4).unwrap();
        }
        manager.sync_all_published().unwrap();
        let err = manager.remove_segments_before(2).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(manager.remove_segments_before(1).unwrap(), 1);
        assert!(!segment0.exists());
        assert!(segment1.exists());
    }
    let recovered = recover_wal_manager(&config).unwrap();
    assert_eq!(recovered.first_segment_id, 1);
    assert_eq!(recovered.durable_segment_id, 1);
    assert_eq!(recovered.durable_blocks, 1);
    assert_eq!(recovered.recovered_segments, 1);
    assert_eq!(recovered.recovered_records, 4);
    assert_eq!(recovered.stats, stats_for_range(8, 4));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_scan_recovery_honors_retained_boundary() {
    let dir = temp_wal_dir("scan-retention");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        for block in 0..5 {
            manager.publish_block(block * 4, 4).unwrap();
        }
        manager.sync_all_published().unwrap();
        assert_eq!(manager.remove_segments_before(1).unwrap(), 1);
    }

    let recovered = recover_wal_manager_by_scan(&config).unwrap();
    assert_eq!(recovered.first_segment_id, 1);
    assert_eq!(recovered.durable_segment_id, 2);
    assert_eq!(recovered.durable_blocks, 1);
    assert_eq!(recovered.recovered_segments, 2);
    assert_eq!(recovered.recovered_records, 12);
    assert_eq!(recovered.stats, stats_for_range(8, 12));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_scan_recovery_rejects_layout_mismatch() {
    let dir = temp_wal_dir("scan-layout-mismatch");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        manager.publish_block(0, 4).unwrap();
        manager.sync_all_published().unwrap();
    }

    let wrong_config = WalSegmentManagerConfig::new(&dir, "events", 8, 8);
    let err = recover_wal_manager_by_scan(&wrong_config).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_scan_recovery_rejects_prefix_behind_control() {
    let dir = temp_wal_dir("scan-behind-control");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        for block in 0..3 {
            manager.publish_block(block * 4, 4).unwrap();
        }
        manager.sync_all_published().unwrap();
    }
    std::fs::remove_file(config.segment_path(1)).unwrap();

    let err = recover_wal_manager_by_scan(&config).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_scan_recovery_rejects_segment_after_partial() {
    let dir = temp_wal_dir("scan-gap-after-partial");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        manager.publish_block(0, 4).unwrap();
        manager
            .current_segment
            .sync_published_data_frontier(1)
            .unwrap();
    }
    {
        let _empty_segment =
            unsafe { MappedWalSegment::create(config.segment_path(1), 1, 8, 4).unwrap() };
    }

    let err = recover_wal_manager_by_scan(&config).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_recovery_rejects_segment_layout_mismatch() {
    let dir = temp_wal_dir("control-layout-mismatch");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    let segment0 = config.segment_path(0);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        manager.publish_block(0, 4).unwrap();
        manager.sync_all_published().unwrap();
    }
    std::fs::remove_file(&segment0).unwrap();
    {
        let segment = unsafe { MappedWalSegment::create(&segment0, 0, 8, 8).unwrap() };
        segment.try_publish_block(0, 4).unwrap();
        segment.sync_published_prefix(1).unwrap();
    }

    let err = recover_wal_manager(&config).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_rejects_oversized_publish_before_rollover() {
    let dir = temp_wal_dir("oversized-before-rollover");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    let segment1 = config.segment_path(1);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        manager.publish_block(0, 4).unwrap();
        manager.publish_block(4, 4).unwrap();
        assert_eq!(manager.current_segment_id, 0);
        assert_eq!(manager.current_blocks, 2);

        let err = manager.publish_block(8, 5).unwrap_err();
        assert_eq!(
            err.downcast_ref::<PublishError>().copied(),
            Some(PublishError::CountExceedsBlockSize {
                count: 5,
                block_size: 4,
            })
        );
        assert_eq!(manager.current_segment_id, 0);
        assert_eq!(manager.current_blocks, 2);
        assert!(!segment1.exists());

        let oversized: Vec<_> = (0..5).map(custom_intent).collect();
        let err = manager.publish_intents(&oversized).unwrap_err();
        assert_eq!(
            err.downcast_ref::<PublishError>().copied(),
            Some(PublishError::CountExceedsBlockSize {
                count: 5,
                block_size: 4,
            })
        );
        assert_eq!(manager.current_segment_id, 0);
        assert_eq!(manager.current_blocks, 2);
        assert!(!segment1.exists());
        manager.sync_all_published().unwrap();
    }

    let recovered = recover_wal_manager(&config).unwrap();
    assert_eq!(recovered.durable_segment_id, 0);
    assert_eq!(recovered.durable_blocks, 2);
    assert_eq!(recovered.recovered_records, 8);
    assert_eq!(recovered.stats, stats_for_range(0, 8));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_rejects_empty_publish_without_advancing() {
    let dir = temp_wal_dir("empty-publish");
    let config = WalSegmentManagerConfig::new(&dir, "events", 8, 4);
    {
        let mut manager = unsafe { WalSegmentManager::create(config.clone()).unwrap() };
        let err = manager.publish_block(0, 0).unwrap_err();
        assert_eq!(
            err.downcast_ref::<std::io::Error>().map(|err| err.kind()),
            Some(std::io::ErrorKind::InvalidInput)
        );
        let err = manager.publish_intents(&[]).unwrap_err();
        assert_eq!(
            err.downcast_ref::<std::io::Error>().map(|err| err.kind()),
            Some(std::io::ErrorKind::InvalidInput)
        );
        let position = manager.publish_block(0, 4).unwrap();
        assert_eq!(
            position,
            WalPosition {
                segment_id: 0,
                block_id: 0,
            }
        );
        manager.sync_all_published().unwrap();
    }

    let recovered = recover_wal_manager(&config).unwrap();
    assert_eq!(recovered.recovered_records, 4);
    assert_eq!(recovered.stats, stats_for_range(0, 4));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn wal_manager_rejects_prefix_path_escape() {
    let dir = temp_wal_dir("prefix-escape");
    let config = WalSegmentManagerConfig::new(&dir, "../events", 8, 4);
    let err = unsafe { WalSegmentManager::create(config.clone()) }
        .err()
        .unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    let err = recover_wal_manager(&config).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    let config = WalSegmentManagerConfig::new(&dir, "nested/events", 8, 4);
    let err = unsafe { WalSegmentManager::create(config) }.err().unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    let _ = std::fs::remove_dir_all(dir);
}
