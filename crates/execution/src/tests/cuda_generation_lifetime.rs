use crate::{
    CudaDeviceMemoryChunk, CudaDriverRuntime, CudaI32BatchProjectionRow, CudaResidentDeviceMemory,
};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn published_resident_generation_survives_a_replacement_publish_and_is_freed_after_drain() {
    // P1-M3 step 1 — the real-GPU soundness probe (doc 14 acceptance gate 1).
    //
    // Retires the device-memory-lifetime risk the P1-M2 spike could only model with
    // a leaked-static buffer. With a REAL CudaResidentDeviceMemory whose Drop calls
    // the REAL cu_mem_free, it proves that under the SnapshotCell publish-on-commit
    // model a generation a reader still holds is:
    //   (a) NOT freed when the writer publishes a replacement, and still GPU-valid
    //       (a kernel read of it returns the correct rows), and
    //   (b) freed only after that last reader drains.
    // The probe cannot even compile unless `CudaResidentDeviceMemory: Send + Sync`
    // (the cell must cross the thread boundary), so it also witnesses that change.
    use gpu_db_snapshot::SnapshotCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    // Owner wrapper whose Drop sets `freed` as it is entered — immediately before
    // the inner CudaResidentDeviceMemory field drops (field-declaration order) and
    // calls the REAL cu_mem_free / cu_ctx_destroy on the same synchronous drop path.
    // So `freed == true` means that device free has entered and is about to run —
    // the Drop-observing wrapper doc 14 gate 1 sanctions. For "not freed while held"
    // this is conservative; for "freed after drain" the free is the next,
    // unconditional statements once Drop is entered.
    struct ObservableResident {
        resident: CudaResidentDeviceMemory,
        freed: Arc<AtomicBool>,
    }
    impl Drop for ObservableResident {
        fn drop(&mut self) {
            self.freed.store(true, Ordering::SeqCst);
        }
    }

    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");

    // Known payload: header(row_count) + i32 filter column + i32 projection column.
    let row_count = 5_u64;
    let filter_offset = std::mem::size_of::<u64>() as u64;
    let projection_offset = filter_offset + row_count * std::mem::size_of::<i32>() as u64;
    let mut header = Vec::new();
    header.extend_from_slice(&row_count.to_le_bytes());
    let mut filter = Vec::new();
    for value in [1_i32, 2, 3, 2, 4] {
        filter.extend_from_slice(&value.to_le_bytes());
    }
    let mut projection = Vec::new();
    for value in [10_i32, 20, 30, 21, 40] {
        projection.extend_from_slice(&value.to_le_bytes());
    }
    let allocated_len = projection_offset + projection.len() as u64;

    let build = |freed: &Arc<AtomicBool>| ObservableResident {
        resident: runtime
            .retain_device_memory_chunks(
                0,
                allocated_len,
                &[
                    CudaDeviceMemoryChunk {
                        byte_offset: 0,
                        bytes: &header,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: filter_offset,
                        bytes: &filter,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: projection_offset,
                        bytes: &projection,
                    },
                ],
            )
            .expect("retain resident device memory"),
        freed: Arc::clone(freed),
    };

    // needles [2,4] over filter [1,2,3,2,4] → rows 1,3 (=2) and 4 (=4),
    // projecting [20], [21], [40].
    let expected = vec![
        CudaI32BatchProjectionRow {
            needle_index: 0,
            row_index: 1,
            values: vec![20],
        },
        CudaI32BatchProjectionRow {
            needle_index: 0,
            row_index: 3,
            values: vec![21],
        },
        CudaI32BatchProjectionRow {
            needle_index: 1,
            row_index: 4,
            values: vec![40],
        },
    ];

    let freed_g1 = Arc::new(AtomicBool::new(false));
    // `SnapshotCell<Arc<owner>>` mirrors doc 14's prescribed engine residency shape
    // (the cell wraps each generation in its own `Arc`, so the inner `Arc<owner>` is
    // redundant *here* but is the type the engine application in step 2 adopts).
    let cell = Arc::new(SnapshotCell::new(Arc::new(build(&freed_g1))));

    let barrier = Arc::new(Barrier::new(2));
    let reader = {
        let cell = Arc::clone(&cell);
        let barrier = Arc::clone(&barrier);
        let expected = expected.clone();
        std::thread::spawn(move || {
            let handle = cell.load(); // pins g1 for the whole closure
            assert_eq!(handle.generation(), 1, "reader did not pin g1");
            barrier.wait(); // (1) signal: g1 is pinned
            barrier.wait(); // (2) resume only after the writer published g2

            // g2 is now current, but we still hold g1. A real GPU read of g1 must
            // still return the correct rows — proof its device memory was not freed
            // by the publish. read_view() is derived from the held owner, so the
            // owner remains the lifetime anchor; submit + complete_detached each set
            // the context current on this reader thread.
            let (rows, elapsed_us) = handle
                .get()
                .resident
                .read_view()
                .submit_match_project_i32_equal_any_from_payload(
                    filter_offset,
                    &[2, 4],
                    &[projection_offset],
                    row_count,
                )
                .expect("submit on pinned g1")
                .complete_detached()
                .expect("complete on pinned g1");
            assert_eq!(
                rows, expected,
                "pinned g1 returned wrong rows — freed early?"
            );
            assert!(elapsed_us.is_some(), "no CUDA-event timing from pinned g1");
            // handle drops here → releases the last reference to g1
        })
    };

    barrier.wait(); // (1) g1 is pinned by the reader
    let freed_g2 = Arc::new(AtomicBool::new(false));
    cell.publish(Arc::new(build(&freed_g2))); // writer publishes a replacement
    assert_eq!(cell.current_generation(), 2, "g2 was not published");
    assert!(
        !freed_g1.load(Ordering::SeqCst),
        "g1 was freed while a reader still held it (use-after-free risk)"
    );
    barrier.wait(); // (2) let the reader do its GPU read of g1

    reader.join().expect("reader thread panicked");
    // The reader drained → its handle (the last reference to g1) dropped, and the
    // cell holds g2, not g1. So g1 must now be reclaimed: the real cu_mem_free ran.
    assert!(
        freed_g1.load(Ordering::SeqCst),
        "g1 was not freed after its last reader drained (leak / reclamation broken)"
    );
    // g2 is still current (held by the cell), so it must still be alive.
    assert!(
        !freed_g2.load(Ordering::SeqCst),
        "current generation g2 was freed early"
    );
}
