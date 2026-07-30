use super::*;
use crate::{
    prepare_exact_canonical_wal_record, read_wal_segment, CanonicalFragment, CanonicalFragmentKind,
    CanonicalIdentity, CanonicalIsolation, CanonicalOutcome, CanonicalOutcomeKind,
    CanonicalPhysicalRange, CanonicalPreApplyHeader, EngineError, PreparedCanonicalWalRecord,
    TxnId, WalBuffer, WalGroupFlushBegin, WalRecord,
};
use gpu_db_types::{DurabilityBackend, DurabilityFault, DurabilityPoison, DurabilityStage};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

struct ThreadCountingAllocator;

thread_local! {
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn count_allocation_if_scoped() {
    COUNT_ALLOCATIONS.with(|enabled| {
        if enabled.get() {
            ALLOCATION_COUNT.with(|count| count.set(count.get().saturating_add(1)));
        }
    });
}

unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation_if_scoped();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation_if_scoped();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation_if_scoped();
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static THREAD_COUNTING_ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

struct AllocationScope {
    active: bool,
}

impl AllocationScope {
    fn begin() -> Self {
        COUNT_ALLOCATIONS.with(|enabled| {
            assert!(
                !enabled.get(),
                "nested thread allocation scopes are unsupported"
            );
            ALLOCATION_COUNT.with(|count| count.set(0));
            enabled.set(true);
        });
        Self { active: true }
    }

    fn finish(mut self) -> usize {
        self.active = false;
        COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
        ALLOCATION_COUNT.with(Cell::get)
    }
}

impl Drop for AllocationScope {
    fn drop(&mut self) {
        if self.active {
            COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
        }
    }
}

fn assert_zero_allocations<T>(operation: impl FnOnce() -> T) -> T {
    let scope = AllocationScope::begin();
    let result = operation();
    assert_eq!(scope.finish(), 0, "post-handoff transition allocated");
    result
}

static NEXT_TEST_PATH: AtomicU64 = AtomicU64::new(1);

fn test_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "gpu-db-wal-scatter-{label}-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH.fetch_add(1, Ordering::Relaxed)
    ))
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(crate::durable_identity_path(path));
    let _ = std::fs::remove_file(crate::wal_tail_offset_path(path));
}

#[cfg(unix)]
fn fua_test_path(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "gpu-db-wal-scatter-fua-{label}-{}-{}",
        std::process::id(),
        NEXT_TEST_PATH.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).expect("create isolated FUA test directory");
    directory.join("wal.segment")
}

fn descriptor_backing(wal: &WalBuffer, slot: usize) -> (*const GroupEntry, usize) {
    let state = wal.prepared_group_arena.slots[slot]
        .state
        .lock()
        .expect("descriptor state");
    match &*state {
        GroupSlotState::Idle(entries) => (entries.as_ptr(), entries.capacity()),
        GroupSlotState::Poisoned { entries, .. } => (entries.as_ptr(), entries.capacity()),
        GroupSlotState::Busy => panic!("descriptor backing must be sampled outside handoff"),
    }
}

fn exact_fault(error: EngineError) -> DurabilityFault {
    match error {
        EngineError::DurabilityFault(fault) => fault,
        other => panic!("expected fixed durability fault, got {other}"),
    }
}

fn begin_group_error(wal: &mut WalBuffer) -> EngineError {
    match wal.begin_group_flush() {
        Err(error) => error,
        Ok(_) => panic!("poisoned WAL unexpectedly accepted another group flush"),
    }
}

fn assert_descriptor_backing_unchanged(wal: &WalBuffer, expected: (*const GroupEntry, usize)) {
    let actual = descriptor_backing(wal, 0);
    assert_eq!(
        actual.0, expected.0,
        "descriptor allocation pointer changed"
    );
    assert_eq!(
        actual.1, expected.1,
        "descriptor allocation capacity changed"
    );
    assert!(actual.1 >= MAX_WAL_GROUP_RECORDS);
}

fn exact_fault_for(stage: DurabilityStage, raw_os_error: Option<i32>) -> DurabilityFault {
    DurabilityFault::new(DurabilityBackend::SerialWal, stage, raw_os_error, 0, 0)
}

fn exact_record(txn_id: TxnId) -> PreparedCanonicalWalRecord {
    exact_record_with_mutation_bytes(txn_id, 1)
}

fn exact_record_with_mutation_bytes(
    txn_id: TxnId,
    mutation_bytes: usize,
) -> PreparedCanonicalWalRecord {
    prepare_exact_canonical_wal_record(
        txn_id,
        CanonicalPhysicalRange {
            log_epoch: 1,
            lane_id: 0,
            segment_id: 1,
            first_frame_ordinal: 0,
        },
        CanonicalPreApplyHeader {
            identity: CanonicalIdentity {
                database_id: [1; 16],
                cluster_id: [2; 16],
                timeline_id: [3; 16],
                format_epoch: 1,
            },
            leader_epoch: 1,
            commit_seq: txn_id,
            stable_transaction_id: txn_id,
            request_digest: [4; 32],
            isolation: CanonicalIsolation::ReadCommitted,
            flags: u32::from(CanonicalFragmentKind::RowMutation as u16),
            catalog_before_epoch: txn_id.saturating_sub(1),
            catalog_after_epoch: txn_id,
            catalog_before_digest: [7; 32],
            catalog_after_digest: [7; 32],
            operation_count: 2,
            table_block_count: 0,
            allocator_high_water: 0,
        },
        &[
            CanonicalFragment {
                kind: CanonicalFragmentKind::RowMutation,
                body: vec![5; mutation_bytes],
            },
            CanonicalFragment {
                kind: CanonicalFragmentKind::TransactionClaimStatus,
                body: vec![6],
            },
        ],
        CanonicalOutcome {
            kind: CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 1,
            sqlstate: None,
            constraint_id: 0,
            target_digest: [8; 32],
            returning_digest: [0; 32],
        },
    )
    .expect("build exact test record")
}

fn claim_exact(wal: &mut WalBuffer, txn_id: TxnId) -> Arc<[u8]> {
    let prepared = exact_record(txn_id);
    let serialized = prepared
        .exact_authority()
        .expect("exact record authority")
        .serialized_record()
        .clone();
    let mut reservation = wal
        .reserve_typed_exact_append(prepared)
        .expect("preproposal exact reservation");
    wal.append_typed_exact_tentative(&mut reservation)
        .expect("tentative exact append");
    wal.claim_typed_exact_append(reservation)
        .expect("claim exact append");
    serialized
}

fn serial_group_exact_ptr(job: &super::super::WalGroupFlushJob) -> Option<*const u8> {
    match job.kind.as_ref().expect("fresh group job") {
        super::super::WalGroupFlushJobKind::Serial(job) => {
            job.first_exact_serialized_ptr_for_test()
        }
        #[cfg(unix)]
        super::super::WalGroupFlushJobKind::Fua(_) => None,
    }
}

fn next_job(wal: &mut WalBuffer) -> super::super::WalGroupFlushJob {
    match wal.begin_group_flush().expect("begin WAL group") {
        WalGroupFlushBegin::Job(job) => job,
        WalGroupFlushBegin::Clean { .. } => panic!("test WAL has an unflushed serial group"),
        WalGroupFlushBegin::Busy => panic!("test WAL must have a free prepared descriptor"),
    }
}

#[test]
fn serial_exact_group_moves_the_original_arc_and_persists_identical_bytes() {
    let path = test_path("exact-arc");
    let mut wal = WalBuffer::with_durable_segment(&path);
    let serialized = claim_exact(&mut wal, 1);
    let expected_payload = wal
        .last_record()
        .expect("exact record appended")
        .payload
        .clone();

    let job = next_job(&mut wal);
    assert_eq!(serial_group_exact_ptr(&job), Some(serialized.as_ptr()));
    assert_eq!(job.commit().expect("durable exact group"), 1);
    assert!(
        wal.exact_typed_wire_records.is_empty(),
        "successful group drops the moved sparse exact owner"
    );
    drop(wal);
    let raw = std::fs::read(&path).expect("read raw exact serial segment");
    let valid_start = crate::WAL_SEGMENT_MAGIC.len();
    assert_eq!(
        &raw[valid_start..valid_start + serialized.len()],
        serialized.as_ref(),
        "the valid bytes after the segment magic are exactly the prebuilt serialized Arc"
    );
    let records = read_wal_segment(&path).expect("read exact serial segment");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].payload, expected_payload);
    cleanup(&path);
}

#[test]
fn in_memory_exact_uses_the_same_group_handoff_and_retires_the_sidecar() {
    let mut wal = WalBuffer::new();
    let serialized = claim_exact(&mut wal, 1);
    assert_eq!(
        wal.typed_exact_serialized_bytes_for_test(0).as_deref(),
        Some(serialized.as_ref())
    );
    match wal.begin_group_flush().expect("form in-memory group") {
        WalGroupFlushBegin::Clean { flushed_records } => assert_eq!(flushed_records, 1),
        WalGroupFlushBegin::Job(_) | WalGroupFlushBegin::Busy => {
            panic!("in-memory group completes during handoff")
        }
    }
    assert!(wal.exact_typed_wire_records.is_empty());
    assert_eq!(wal.flushed_count(), 1);
}

#[test]
fn mixed_legacy_exact_group_stops_before_later_legacy_and_replays_identically() {
    let path = test_path("mixed");
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(WalRecord {
        txn_id: 1,
        payload: Arc::from(&b"legacy-before"[..]),
    });
    let exact = claim_exact(&mut wal, 2);
    wal.append(WalRecord {
        txn_id: 3,
        payload: Arc::from(&b"legacy-after"[..]),
    });

    let job = next_job(&mut wal);
    assert_eq!(serial_group_exact_ptr(&job), Some(exact.as_ptr()));
    assert_eq!(job.commit().expect("mixed first group"), 2);
    assert_eq!(wal.flushed_count(), 2);
    assert!(wal.exact_typed_wire_records.is_empty());
    assert_eq!(next_job(&mut wal).commit().expect("legacy tail group"), 3);
    drop(wal);
    let records = read_wal_segment(&path).expect("replay mixed serial segment");
    assert_eq!(
        records
            .iter()
            .map(|record| record.txn_id)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    cleanup(&path);
}

#[test]
fn serial_mixed_group_replays_empty_legacy_before_nonempty_legacy_and_exact() {
    let path = test_path("empty-legacy-mixed");
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(WalRecord {
        txn_id: 1,
        payload: Arc::from(&b""[..]),
    });
    wal.append(WalRecord {
        txn_id: 2,
        payload: Arc::from(&b"legacy-after-empty"[..]),
    });
    claim_exact(&mut wal, 3);
    let exact_payload = wal
        .last_record()
        .expect("exact record appended")
        .payload
        .clone();

    assert_eq!(next_job(&mut wal).commit().expect("mixed serial group"), 3);
    drop(wal);

    let records = read_wal_segment(&path).expect("replay mixed serial segment");
    assert_eq!(
        records
            .iter()
            .map(|record| (record.txn_id, record.payload.as_ref()))
            .collect::<Vec<_>>(),
        vec![
            (1, &b""[..]),
            (2, &b"legacy-after-empty"[..]),
            (3, exact_payload.as_ref()),
        ]
    );
    cleanup(&path);
}

#[test]
fn scatter_cursor_skips_empty_legacy_payload_at_boundary_before_partial_later_data() {
    let path = test_path("empty-legacy-boundary");
    let mut wal = WalBuffer::with_durable_segment(&path);
    let empty = WalRecord {
        txn_id: 1,
        payload: Arc::from(&b""[..]),
    };
    let later = WalRecord {
        txn_id: 2,
        payload: Arc::from(&b"legacy-after-empty"[..]),
    };
    wal.append(empty.clone());
    wal.append(later.clone());
    let exact = claim_exact(&mut wal, 3);

    let mut expected = Vec::new();
    for record in [&empty, &later] {
        let mut header = [0; crate::WAL_RECORD_HEADER_LEN];
        crate::encode_record_header_into(&mut header, record);
        expected.extend_from_slice(&header);
        expected.extend_from_slice(record.payload.as_ref());
    }
    expected.extend_from_slice(exact.as_ref());

    let job = next_job(&mut wal);
    let observed = Arc::new(Mutex::new(Vec::<(u64, Vec<u8>)>::new()));
    let observed_write = Arc::clone(&observed);
    let steps = Arc::new(Mutex::new(VecDeque::from([
        crate::WAL_RECORD_HEADER_LEN,
        1usize,
        usize::MAX,
    ])));
    let write_steps = Arc::clone(&steps);
    let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
        write: Box::new(move |iovecs, offset| {
            let offered = iovecs
                .iter()
                .flat_map(|iov| unsafe {
                    std::slice::from_raw_parts(iov.iov_base as *const u8, iov.iov_len)
                })
                .copied()
                .collect::<Vec<_>>();
            let requested = write_steps
                .lock()
                .expect("partial write steps")
                .pop_front()
                .expect("unexpected extra scatter write");
            let written = requested.min(offered.len());
            observed_write
                .lock()
                .expect("observe iovecs")
                .push((offset, offered[..written].to_vec()));
            SerialIoTestWrite::Written(written)
        }),
        sync: Box::new(|| SerialIoTestSync::Complete),
        before_fixed_fault_wake: None,
    });
    assert_eq!(job.commit().expect("empty payload mixed scatter"), 3);

    let seen = observed.lock().expect("read observed scatter bytes");
    let mut expected_offset = crate::WAL_SEGMENT_MAGIC.len() as u64;
    let mut reconstructed = Vec::new();
    for (offset, bytes) in seen.iter() {
        assert_eq!(*offset, expected_offset, "scatter offset is contiguous");
        expected_offset += bytes.len() as u64;
        reconstructed.extend_from_slice(bytes);
    }
    assert_eq!(
        reconstructed, expected,
        "empty legacy payload has no gap, duplicate, or poisoned cursor transition"
    );
    assert!(
        steps
            .lock()
            .expect("all partial write steps used")
            .is_empty(),
        "the first boundary write and following partial write both ran"
    );
    drop(seen);
    drop(_hook);
    drop(wal);
    cleanup(&path);
}

#[test]
fn exact_after_a_full_legacy_descriptor_boundary_remains_pre_admitted() {
    let path = test_path("boundary");
    let mut wal = WalBuffer::with_durable_segment(&path);
    for txn_id in 1..=MAX_WAL_GROUP_RECORDS as u64 {
        wal.append(WalRecord {
            txn_id,
            payload: Arc::from(&b"legacy"[..]),
        });
    }
    let exact = claim_exact(&mut wal, MAX_WAL_GROUP_RECORDS as u64 + 1);
    assert_eq!(
        next_job(&mut wal).commit().expect("full legacy descriptor"),
        MAX_WAL_GROUP_RECORDS
    );
    let job = next_job(&mut wal);
    assert_eq!(serial_group_exact_ptr(&job), Some(exact.as_ptr()));
    assert_eq!(
        job.commit().expect("exact group after descriptor boundary"),
        MAX_WAL_GROUP_RECORDS + 1
    );
    drop(wal);
    cleanup(&path);
}

#[test]
fn scatter_cursor_handles_partial_first_middle_and_last_iovecs() {
    let path = test_path("partials");
    let mut wal = WalBuffer::with_durable_segment(&path);
    let legacy = WalRecord {
        txn_id: 1,
        payload: Arc::from(&b"legacy"[..]),
    };
    wal.append(legacy.clone());
    let exact = claim_exact(&mut wal, 2);
    let mut expected = Vec::new();
    let mut legacy_header = [0; crate::WAL_RECORD_HEADER_LEN];
    crate::encode_record_header_into(&mut legacy_header, &legacy);
    expected.extend_from_slice(&legacy_header);
    expected.extend_from_slice(legacy.payload.as_ref());
    expected.extend_from_slice(exact.as_ref());

    let job = next_job(&mut wal);
    let observed = Arc::new(Mutex::new(Vec::<(u64, Vec<u8>)>::new()));
    let observed_write = Arc::clone(&observed);
    let steps = Arc::new(Mutex::new(VecDeque::from([7usize, 20, 14, usize::MAX])));
    let write_steps = Arc::clone(&steps);
    let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
        write: Box::new(move |iovecs, offset| {
            let offered = iovecs
                .iter()
                .flat_map(|iov| unsafe {
                    std::slice::from_raw_parts(iov.iov_base as *const u8, iov.iov_len)
                })
                .copied()
                .collect::<Vec<_>>();
            let requested = write_steps
                .lock()
                .expect("partial write steps")
                .pop_front()
                .expect("unexpected extra scatter write");
            let written = requested.min(offered.len());
            observed_write
                .lock()
                .expect("observe iovecs")
                .push((offset, offered[..written].to_vec()));
            SerialIoTestWrite::Written(written)
        }),
        sync: Box::new(|| SerialIoTestSync::Complete),
        before_fixed_fault_wake: None,
    });
    assert_eq!(job.commit().expect("partial scatter group"), 2);
    let seen = observed.lock().expect("read observed scatter bytes");
    let mut expected_offset = crate::WAL_SEGMENT_MAGIC.len() as u64;
    let mut reconstructed = Vec::new();
    for (offset, bytes) in seen.iter() {
        assert_eq!(*offset, expected_offset, "scatter offset is contiguous");
        expected_offset += bytes.len() as u64;
        reconstructed.extend_from_slice(bytes);
    }
    assert_eq!(
        reconstructed, expected,
        "scatter cursor neither skips nor duplicates bytes"
    );
    assert!(
        steps
            .lock()
            .expect("all partial write steps used")
            .is_empty(),
        "chosen cuts cover header, middle payload, and final exact iovec"
    );
    drop(seen);
    drop(_hook);
    drop(wal);
    cleanup(&path);
}

#[test]
fn scatter_retries_write_and_sync_eintr_and_poison_paths_retain_the_descriptor() {
    let path = test_path("eintr");
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let job = next_job(&mut wal);
    let writes = Arc::new(Mutex::new(VecDeque::from([
        SerialIoTestWrite::Errno(libc::EINTR),
        SerialIoTestWrite::Written(usize::MAX),
    ])));
    let write_steps = Arc::clone(&writes);
    let syncs = Arc::new(Mutex::new(VecDeque::from([
        SerialIoTestSync::Errno(libc::EINTR),
        SerialIoTestSync::Complete,
    ])));
    let sync_steps = Arc::clone(&syncs);
    let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
        write: Box::new(move |iovecs, _| {
            match write_steps.lock().expect("write steps").pop_front() {
                Some(SerialIoTestWrite::Written(usize::MAX)) => {
                    SerialIoTestWrite::Written(iovecs.iter().map(|iov| iov.iov_len).sum())
                }
                Some(result) => result,
                None => panic!("unexpected extra scatter write"),
            }
        }),
        sync: Box::new(move || {
            sync_steps
                .lock()
                .expect("sync steps")
                .pop_front()
                .expect("unexpected extra sync")
        }),
        before_fixed_fault_wake: None,
    });
    assert_eq!(job.commit().expect("EINTR retry group"), 1);
    drop(_hook);
    assert!(writes.lock().expect("write steps consumed").is_empty());
    assert!(syncs.lock().expect("sync steps consumed").is_empty());
    drop(wal);
    cleanup(&path);

    for (label, write, sync) in [
        (
            "zero",
            SerialIoTestWrite::Written(0usize),
            SerialIoTestSync::Complete,
        ),
        (
            "write-error",
            SerialIoTestWrite::Errno(libc::EIO),
            SerialIoTestSync::Complete,
        ),
        (
            "sync-error",
            SerialIoTestWrite::Written(usize::MAX),
            SerialIoTestSync::Errno(libc::EIO),
        ),
    ] {
        let path = test_path(label);
        let mut wal = WalBuffer::with_durable_segment(&path);
        claim_exact(&mut wal, 1);
        let backing = descriptor_backing(&wal, 0);
        let job = next_job(&mut wal);
        let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
            write: Box::new(move |iovecs, _| match &write {
                SerialIoTestWrite::Written(usize::MAX) => {
                    SerialIoTestWrite::Written(iovecs.iter().map(|iov| iov.iov_len).sum())
                }
                result => *result,
            }),
            sync: Box::new(move || sync),
            before_fixed_fault_wake: None,
        });
        assert!(
            job.commit().is_err(),
            "{label} must poison the serial group"
        );
        drop(_hook);
        assert!(wal.begin_group_flush().is_err(), "{label} wedges the WAL");
        {
            let state = wal.prepared_group_arena.slots[0]
                .state
                .lock()
                .expect("descriptor state");
            match &*state {
                GroupSlotState::Poisoned { entries, .. } => {
                    assert_eq!(
                        entries.as_ptr(),
                        backing.0,
                        "{label} retains descriptor allocation"
                    );
                    assert_eq!(
                        entries.capacity(),
                        backing.1,
                        "{label} retains descriptor capacity"
                    );
                    assert!(
                        entries.capacity() >= MAX_WAL_GROUP_RECORDS,
                        "permanent descriptor backing is at least one complete group"
                    );
                }
                _ => panic!("failed group retains its permanent descriptor backing"),
            }
        }
        drop(wal);
        cleanup(&path);
    }
}

#[test]
fn busy_descriptor_recycles_after_success_and_abandonment_poison_is_fail_closed() {
    let path = test_path("busy");
    let mut wal = WalBuffer::with_durable_segment(&path);
    wal.append(WalRecord {
        txn_id: 1,
        payload: Arc::from(&b"first"[..]),
    });
    let job = next_job(&mut wal);
    assert!(matches!(
        wal.begin_group_flush().expect("bounded backpressure"),
        WalGroupFlushBegin::Busy
    ));
    assert_eq!(job.commit().expect("first descriptor success"), 1);
    wal.append(WalRecord {
        txn_id: 2,
        payload: Arc::from(&b"second"[..]),
    });
    assert_eq!(next_job(&mut wal).commit().expect("recycled descriptor"), 2);
    drop(wal);
    cleanup(&path);

    let path = test_path("abandon");
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let backing = descriptor_backing(&wal, 0);
    drop(next_job(&mut wal));
    assert!(
        wal.begin_group_flush().is_err(),
        "abandoned exact group is fail-closed"
    );
    {
        let state = wal.prepared_group_arena.slots[0]
            .state
            .lock()
            .expect("descriptor state");
        match &*state {
            GroupSlotState::Poisoned { entries, .. } => {
                assert_eq!(
                    entries.as_ptr(),
                    backing.0,
                    "abandonment retains descriptor allocation"
                );
                assert_eq!(
                    entries.capacity(),
                    backing.1,
                    "abandonment retains descriptor capacity"
                );
                assert!(entries.capacity() >= MAX_WAL_GROUP_RECORDS);
            }
            _ => panic!("abandoned group must retain a poisoned descriptor"),
        }
    }
    drop(wal);
    cleanup(&path);
}

#[test]
fn serial_exact_post_handoff_success_allocates_nothing_and_recycles_descriptor() {
    let path = test_path("no-alloc-success");
    let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
        write: Box::new(|iovecs, _| {
            SerialIoTestWrite::Written(iovecs.iter().map(|iov| iov.iov_len).sum())
        }),
        sync: Box::new(|| SerialIoTestSync::Complete),
        before_fixed_fault_wake: None,
    });
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let backing = descriptor_backing(&wal, 0);
    let job = next_job(&mut wal);
    let result = assert_zero_allocations(|| job.commit());
    assert_eq!(result.expect("exact serial success"), 1);
    assert_descriptor_backing_unchanged(&wal, backing);
    drop(_hook);
    drop(wal);
    cleanup(&path);
}

#[test]
fn serial_exact_partial_write_failure_allocates_nothing_and_keeps_first_fault() {
    let path = test_path("no-alloc-partial");
    let writes = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::clone(&writes);
    let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
        write: Box::new(move |_, _| {
            let invocation = write_count.fetch_add(1, Ordering::Relaxed);
            if invocation == 0 {
                SerialIoTestWrite::Written(1)
            } else {
                SerialIoTestWrite::Errno(libc::EIO)
            }
        }),
        sync: Box::new(|| SerialIoTestSync::Complete),
        before_fixed_fault_wake: None,
    });
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let backing = descriptor_backing(&wal, 0);
    let job = next_job(&mut wal);
    let error = assert_zero_allocations(|| job.commit()).expect_err("partial write then EIO");
    let rendered = format!("{error}");
    let fault = exact_fault(error);
    assert_eq!(
        fault,
        exact_fault_for(DurabilityStage::PositionalWrite, Some(libc::EIO))
    );
    assert!(rendered.contains("backend=serial-wal"));
    assert!(rendered.contains("stage=positional-write"));
    assert!(rendered.contains("raw_os_error=Some"));
    assert_descriptor_backing_unchanged(&wal, backing);
    assert_eq!(exact_fault(begin_group_error(&mut wal)), fault);
    drop(_hook);
    drop(wal);
    cleanup(&path);
}

#[test]
fn serial_exact_write_zero_allocates_nothing_and_retains_descriptor() {
    let path = test_path("no-alloc-zero");
    let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
        write: Box::new(|_, _| SerialIoTestWrite::Written(0)),
        sync: Box::new(|| SerialIoTestSync::Complete),
        before_fixed_fault_wake: None,
    });
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let backing = descriptor_backing(&wal, 0);
    let job = next_job(&mut wal);
    let fault =
        exact_fault(assert_zero_allocations(|| job.commit()).expect_err("zero positional write"));
    assert_eq!(
        fault,
        exact_fault_for(DurabilityStage::PositionalWriteZero, None)
    );
    assert_descriptor_backing_unchanged(&wal, backing);
    drop(_hook);
    drop(wal);
    cleanup(&path);
}

#[test]
fn serial_exact_sync_failure_allocates_nothing_and_retains_descriptor() {
    let path = test_path("no-alloc-sync");
    let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
        write: Box::new(|iovecs, _| {
            SerialIoTestWrite::Written(iovecs.iter().map(|iov| iov.iov_len).sum())
        }),
        sync: Box::new(|| SerialIoTestSync::Errno(libc::EIO)),
        before_fixed_fault_wake: None,
    });
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let backing = descriptor_backing(&wal, 0);
    let job = next_job(&mut wal);
    let fault = exact_fault(
        assert_zero_allocations(|| job.commit()).expect_err("sync failure must fail closed"),
    );
    assert_eq!(
        fault,
        exact_fault_for(DurabilityStage::SyncData, Some(libc::EIO))
    );
    assert_descriptor_backing_unchanged(&wal, backing);
    drop(_hook);
    drop(wal);
    cleanup(&path);
}

#[test]
fn serial_exact_abandonment_allocates_nothing_and_retains_descriptor() {
    let path = test_path("no-alloc-abandon");
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let backing = descriptor_backing(&wal, 0);
    let job = next_job(&mut wal);
    assert_zero_allocations(|| drop(job));
    let fault = exact_fault(begin_group_error(&mut wal));
    assert_eq!(fault, exact_fault_for(DurabilityStage::Abandoned, None));
    assert_descriptor_backing_unchanged(&wal, backing);
    drop(wal);
    cleanup(&path);
}

#[test]
fn fixed_poison_is_visible_before_wake_and_first_fault_wins_under_race() {
    let path = test_path("fault-before-wake");
    let mut wal = WalBuffer::with_durable_segment(&path);
    claim_exact(&mut wal, 1);
    let core = Arc::clone(wal.durable.as_ref().expect("serial core"));
    let job = next_job(&mut wal);
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let commit_entered = Arc::clone(&entered);
    let commit_release = Arc::clone(&release);
    let worker = std::thread::spawn(move || {
        let _hook = install_serial_io_hooks_for_test(SerialIoTestHooks {
            write: Box::new(|_, _| SerialIoTestWrite::Errno(libc::EIO)),
            sync: Box::new(|| SerialIoTestSync::Complete),
            before_fixed_fault_wake: Some(Box::new(move |fault| {
                assert_eq!(
                    fault,
                    exact_fault_for(DurabilityStage::PositionalWrite, Some(libc::EIO))
                );
                commit_entered.wait();
                commit_release.wait();
            })),
        });
        job.commit()
    });
    entered.wait();
    let expected = exact_fault_for(DurabilityStage::PositionalWrite, Some(libc::EIO));
    assert_eq!(core.fixed_fault(), Some(expected));
    assert!(
        core.lock_state().io_in_flight,
        "fixed poison installs before descriptor settlement, clear, or wake"
    );

    let waiter_core = Arc::clone(&core);
    let waiter_ready = Arc::new(Barrier::new(2));
    let waiter_ready_thread = Arc::clone(&waiter_ready);
    let woke = Arc::new(AtomicBool::new(false));
    let woke_thread = Arc::clone(&woke);
    let waiter = std::thread::spawn(move || {
        waiter_ready_thread.wait();
        drop(waiter_core.lock_state_idle());
        assert_eq!(waiter_core.fixed_fault(), Some(expected));
        woke_thread.store(true, Ordering::Release);
    });
    waiter_ready.wait();
    std::thread::yield_now();
    assert!(
        !woke.load(Ordering::Acquire),
        "waiter cannot wake while the fixed-fault callback holds the pre-settlement barrier"
    );
    release.wait();
    assert_eq!(
        exact_fault(worker.join().expect("commit worker").unwrap_err()),
        expected
    );
    waiter.join().expect("waiter");
    assert!(woke.load(Ordering::Acquire));
    drop(wal);
    cleanup(&path);

    let first = Arc::new(DurabilityPoison::new());
    let race = Arc::new(Barrier::new(3));
    let left = exact_fault_for(DurabilityStage::SyncData, Some(libc::EIO));
    let right = exact_fault_for(DurabilityStage::Abandoned, None);
    let left_poison = Arc::clone(&first);
    let left_race = Arc::clone(&race);
    let left_worker = std::thread::spawn(move || {
        left_race.wait();
        left_poison.install(left)
    });
    let right_poison = Arc::clone(&first);
    let right_race = Arc::clone(&race);
    let right_worker = std::thread::spawn(move || {
        right_race.wait();
        right_poison.install(right)
    });
    race.wait();
    let published = first.snapshot().expect("one racing fault wins");
    assert!(published == left || published == right);
    assert_eq!(left_worker.join().expect("left fault"), published);
    assert_eq!(right_worker.join().expect("right fault"), published);
    assert_eq!(
        first.install(exact_fault_for(DurabilityStage::FrontierDrift, None)),
        published,
        "later faults cannot overwrite the first published fault"
    );
}

#[test]
fn serial_exact_post_handoff_leaf_forbids_allocation_escape_hatches() {
    let source = include_str!("../group.rs");
    for (begin, end) in [
        (
            "// BEGIN SERIAL_EXACT_POST_HANDOFF_IO_NO_ALLOC",
            "// END SERIAL_EXACT_POST_HANDOFF_IO_NO_ALLOC",
        ),
        (
            "// BEGIN SERIAL_EXACT_POST_HANDOFF_NO_ALLOC",
            "// END SERIAL_EXACT_POST_HANDOFF_NO_ALLOC",
        ),
    ] {
        let (_, marked) = source.split_once(begin).expect("post-handoff begin marker");
        let (leaf, _) = marked.split_once(end).expect("post-handoff end marker");
        for forbidden in [
            "format!",
            ".to_string()",
            "String",
            "io::Error::new",
            "io::Error::other",
            "Vec::with_capacity",
            "try_reserve",
        ] {
            assert!(
                !leaf.contains(forbidden),
                "exact post-handoff leaf must not contain {forbidden}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn exact_fua_scatter_reads_the_permanent_group_regions_without_reencoding() {
    use gpu_db_write_conveyor::FuaScatterSource;

    let mut wal = WalBuffer::new();
    let first = claim_exact(&mut wal, 1);
    let second = claim_exact(&mut wal, 2);
    let mut expected = Vec::with_capacity(first.len() + second.len());
    expected.extend_from_slice(&first);
    expected.extend_from_slice(&second);

    let prefix = wal
        .select_prepared_group_prefix(0, None)
        .expect("select exact group")
        .expect("exact prefix");
    assert_eq!(prefix.target_records, 2);
    let mut group = wal
        .handoff_prepared_group(prefix)
        .expect("handoff exact group")
        .expect("prepared descriptor");
    assert!(group.contains_only_exact());
    assert_eq!(FuaScatterSource::len(&group), expected.len());

    let mut copied = vec![0; expected.len()];
    FuaScatterSource::copy_into(&mut group, 0, &mut copied).expect("copy whole descriptor");
    assert_eq!(
        copied, expected,
        "scatter source preserves both exact outer records"
    );

    let crossing_offset = first.len() - 3;
    let mut crossing = [0u8; 9];
    FuaScatterSource::copy_into(&mut group, crossing_offset, &mut crossing)
        .expect("copy across exact-record boundary");
    assert_eq!(
        crossing,
        expected[crossing_offset..crossing_offset + crossing.len()],
        "scatter cursor crosses exact-record regions without a concatenating materializer"
    );
    group.finish_success();
}

#[cfg(unix)]
#[test]
fn exact_fua_group_is_admitted_pre_wal_and_recovers_identically() {
    let path = fua_test_path("exact-direct");
    let mut wal = WalBuffer::with_fua_durable_segment(&path, 8, 1 << 20).expect("FUA WAL");
    assert!(wal.is_fua_durable());
    let first = claim_exact(&mut wal, 1);
    let second = claim_exact(&mut wal, 2);
    let expected = wal.records.clone();
    let job = next_job(&mut wal);
    assert_eq!(
        assert_zero_allocations(|| job.commit()).expect("exact FUA commit"),
        2
    );
    assert_eq!(wal.flushed_count(), 2);
    assert!(wal.exact_typed_wire_records.is_empty());
    let telemetry = wal.fua_durability_telemetry();
    assert_eq!(telemetry.logical_groups, 1);
    assert_eq!(
        telemetry.logical_payload_bytes,
        (first.len() + second.len()) as u64
    );
    assert_eq!(
        telemetry.single_frame_padded_baseline_bytes,
        gpu_db_write_conveyor::fua_frame_padded_bytes(first.len() + second.len()) as u64
    );
    assert_eq!(telemetry.published_frames, 1, "narrow pool selects F1");
    drop(wal);
    assert_eq!(
        crate::recover_fua_wal_records(&path).expect("recover exact FUA group"),
        expected,
        "FUA recovery must replay the prebuilt exact outer bytes"
    );
    std::fs::remove_dir_all(path.parent().expect("isolated FUA directory"))
        .expect("remove isolated FUA artifacts");
}

#[cfg(unix)]
#[test]
fn exact_fua_legacy_separator_seals_and_drains_before_next_exact_admission() {
    let path = fua_test_path("exact-legacy-exact");
    let mut wal = WalBuffer::with_fua_durable_segment(&path, 8, 1 << 20).expect("FUA WAL");
    let first = claim_exact(&mut wal, 1);
    let legacy = WalRecord {
        txn_id: 2,
        payload: Arc::from(&b"compatibility suffix"[..]),
    };
    wal.append(legacy.clone());

    // The next exact pre-admission must seal the first exact prefix and drain the intervening
    // compatibility record. It may not try to make the legacy bytes part of either exact group.
    let second = claim_exact(&mut wal, 3);
    assert_eq!(wal.flushed_count(), 2);
    assert_eq!(wal.len(), 3);
    wal.flush_all().expect("drain trailing exact group");
    assert_eq!(wal.flushed_count(), 3);

    let telemetry = wal.fua_durability_telemetry();
    assert_eq!(
        telemetry.logical_groups, 3,
        "exact prefix, compatibility separator, and trailing exact owner are distinct groups"
    );
    assert_eq!(
        telemetry.published_frames, 3,
        "narrow pool selects one frame per group"
    );
    assert_eq!(
        telemetry.logical_payload_bytes,
        (first.len() + crate::WAL_RECORD_HEADER_LEN + legacy.payload.len() + second.len()) as u64
    );
    let expected = wal.records.clone();
    drop(wal);
    assert_eq!(
        crate::recover_fua_wal_records(&path).expect("recover exact/legacy/exact sequence"),
        expected,
        "recovery preserves the original logical record order across the admission seal"
    );
    std::fs::remove_dir_all(path.parent().expect("isolated FUA directory"))
        .expect("remove isolated FUA artifacts");
}

#[cfg(unix)]
#[test]
fn exact_fua_seals_and_publishes_the_f16_controller_geometry() {
    let path = fua_test_path("exact-f16");
    let mut wal = WalBuffer::with_fua_durable_segment(&path, 16, 2 << 20).expect("FUA WAL");
    let prepared = exact_record_with_mutation_bytes(1, 64 * 1024);
    let expected = prepared.as_wal_record().clone();
    let mut reservation = wal
        .reserve_typed_exact_append(prepared)
        .expect("preproposal F16 exact reservation");
    wal.append_typed_exact_tentative(&mut reservation)
        .expect("tentative F16 exact append");
    wal.claim_typed_exact_append(reservation)
        .expect("claim F16 exact append");
    assert_eq!(
        next_job(&mut wal).commit().expect("commit F16 exact group"),
        1
    );
    let telemetry = wal.fua_durability_telemetry();
    assert_eq!(telemetry.logical_groups, 1);
    assert_eq!(
        telemetry.published_frames, 16,
        "eligible F16 plan is retained to publish"
    );
    assert_eq!(telemetry.controller_sustained_actions, 1);
    drop(wal);
    assert_eq!(
        crate::recover_fua_wal_records(&path).expect("recover F16 exact group"),
        vec![expected]
    );
    std::fs::remove_dir_all(path.parent().expect("isolated FUA directory"))
        .expect("remove isolated FUA artifacts");
}

#[cfg(unix)]
#[test]
fn exact_fua_rollback_restores_the_unclaimed_physical_reservation() {
    let path = fua_test_path("exact-rollback");
    let mut wal = WalBuffer::with_fua_durable_segment(&path, 8, 1 << 20).expect("FUA WAL");
    let mut reservation = wal
        .reserve_typed_exact_append(exact_record(1))
        .expect("preproposal FUA exact reservation");
    wal.append_typed_exact_tentative(&mut reservation)
        .expect("tentative FUA exact append");
    wal.rollback_typed_exact_append(&mut reservation)
        .expect("rollback restores FUA exact tail");
    assert_eq!(wal.len(), 0);
    claim_exact(&mut wal, 2);
    wal.flush_all().expect("flush after rollback");
    drop(wal);
    assert_eq!(
        crate::recover_fua_wal_records(&path)
            .expect("recover rollback FUA group")
            .iter()
            .map(|record| record.txn_id)
            .collect::<Vec<_>>(),
        vec![2],
        "rolled-back exact bytes cannot survive into the committed FUA group"
    );
    std::fs::remove_dir_all(path.parent().expect("isolated FUA directory"))
        .expect("remove isolated FUA artifacts");
}

#[cfg(unix)]
#[test]
fn exact_fua_full_forming_group_seals_and_readmits_the_next_record() {
    let path = fua_test_path("exact-seal-bound");
    let mut wal = WalBuffer::with_fua_durable_segment(&path, 8, 1 << 20).expect("FUA WAL");
    let total = MAX_WAL_GROUP_RECORDS + 1;
    for txn_id in 1..=total as TxnId {
        claim_exact(&mut wal, txn_id);
    }
    let sealed = wal.flushed_count();
    assert!(
        sealed > 0 && sealed < total,
        "the record or wire bound seals the prior exact group before the next claim (sealed={sealed})"
    );
    wal.flush_all().expect("flush trailing exact group");
    assert_eq!(wal.flushed_count(), total);
    assert_eq!(wal.fua_durability_telemetry().logical_groups, 2);
    drop(wal);
    assert_eq!(
        crate::recover_fua_wal_records(&path)
            .expect("recover bounded exact FUA groups")
            .len(),
        total
    );
    std::fs::remove_dir_all(path.parent().expect("isolated FUA directory"))
        .expect("remove isolated FUA artifacts");
}

#[cfg(unix)]
#[test]
fn exact_fua_rolls_only_after_the_successor_is_prepared_before_claim() {
    fn claim_after_preproposal_successor_readiness(wal: &mut WalBuffer, txn_id: TxnId) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match wal.reserve_typed_exact_append(exact_record(txn_id)) {
                Ok(mut reservation) => {
                    wal.append_typed_exact_tentative(&mut reservation)
                        .expect("tentative exact append after prepared rollover");
                    wal.claim_typed_exact_append(reservation)
                        .expect("claim exact append after prepared rollover");
                    return;
                }
                Err(EngineError::ProposalFailed(_)) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(EngineError::ProposalFailed(_)) => {
                    panic!("FUA successor did not become ready before bounded test admission");
                }
                Err(error) => panic!("exact FUA preproposal rollover admission failed: {error}"),
            }
        }
    }

    let path = fua_test_path("exact-roll");
    let mut wal = WalBuffer::with_fua_durable_segment(&path, 8, 16 * 1024).expect("FUA WAL");
    for txn_id in 1..=8 {
        claim_after_preproposal_successor_readiness(&mut wal, txn_id);
        wal.flush_all()
            .expect("flush exact record through prepared roll");
    }
    drop(wal);
    let segment_count = std::fs::read_dir(path.parent().expect("isolated FUA directory"))
        .expect("list FUA artifacts")
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("wal.segment.fua."))
        })
        .count();
    assert!(
        segment_count > 1,
        "exact FUA fixture must cross a prepared segment roll"
    );
    assert_eq!(
        crate::recover_fua_wal_records(&path)
            .expect("recover exact FUA rolls")
            .len(),
        8
    );
    std::fs::remove_dir_all(path.parent().expect("isolated FUA directory"))
        .expect("remove isolated FUA artifacts");
}

#[cfg(unix)]
#[test]
fn exact_fua_post_handoff_fault_is_fixed_and_blocks_every_later_drain() {
    let path = fua_test_path("exact-fixed-fault");
    let mut wal = WalBuffer::with_fua_durable_segment(&path, 8, 1 << 20).expect("FUA WAL");
    claim_exact(&mut wal, 1);
    let job = next_job(&mut wal);
    wal.fail_next_fua_exact_commit_for_test(DurabilityStage::FuaScatter, Some(libc::EIO));
    let fault = exact_fault(
        assert_zero_allocations(|| job.commit()).expect_err("injected exact FUA fault"),
    );
    assert_eq!(fault.backend, DurabilityBackend::FuaWal);
    assert_eq!(fault.stage, DurabilityStage::FuaScatter);
    assert_eq!(fault.raw_os_error(), Some(libc::EIO));
    assert_eq!(fault.group_first_record, 0);
    assert_eq!(
        exact_fault(begin_group_error(&mut wal)),
        fault,
        "the first fixed exact fault must dominate all later FUA drain errors"
    );
    assert_eq!(wal.flushed_count(), 0);
    drop(wal);
    std::fs::remove_dir_all(path.parent().expect("isolated FUA directory"))
        .expect("remove isolated FUA artifacts");
}

#[test]
fn exact_fua_post_handoff_leaf_forbids_materialization_and_lifecycle_escape_hatches() {
    let source = include_str!("../../fua/exact.rs");
    let (_, marked) = source
        .split_once("// BEGIN FUA_EXACT_POST_HANDOFF_NO_ALLOC")
        .expect("FUA exact post-handoff begin marker");
    let (leaf, _) = marked
        .split_once("// END FUA_EXACT_POST_HANDOFF_NO_ALLOC")
        .expect("FUA exact post-handoff end marker");
    for forbidden in [
        "Vec",
        "encode",
        "format!",
        "String",
        ".roll(",
        "kick_prestage",
        "take_prestaged",
        "spawn",
        "rename",
        "sync_",
        "retry",
        "fallback",
    ] {
        assert!(
            !leaf.contains(forbidden),
            "FUA exact post-handoff leaf must not contain {forbidden}"
        );
    }
}

#[test]
fn exact_preverify_and_tentative_append_preserve_decode_telemetry() {
    let path = test_path("telemetry");
    let mut wal = WalBuffer::with_durable_segment(&path);
    let generic = exact_record(1).into_wal_record();
    wal.append(generic);
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 0);

    let mut reservation = wal
        .reserve_typed_exact_append(exact_record(2))
        .expect("preverify exact proposal");
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 1);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 1);

    wal.append_typed_exact_tentative(&mut reservation)
        .expect("tentative exact append");
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 1);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 2);
    wal.rollback_typed_exact_append(&mut reservation)
        .expect("rollback exact proposal");
    assert_eq!(wal.durable_identity_decoded_records_for_test(), 1);
    assert_eq!(wal.durable_identity_verified_records_for_test(), 1);
    drop(wal);
    cleanup(&path);
}

#[test]
fn typed_exact_leaf_has_no_generic_wire_encoder_escape_hatch() {
    let source = include_str!("../typed_exact.rs");
    assert!(
        !source.contains("encode_record_into"),
        "exact typed ownership may not invoke the generic WAL encoder"
    );
}
