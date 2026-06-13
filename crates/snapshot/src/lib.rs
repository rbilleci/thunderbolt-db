//! Reader/writer snapshot generations with epoch reclamation (P1-M2 spike).
//!
//! This is the concurrency substrate the engine's Phase 1 reader/writer split
//! adopts, proven here in isolation before refactoring the 24k-line engine:
//!
//! - **Many concurrent readers, lock-free on the read body.** A reader calls
//!   [`SnapshotCell::load`] to obtain a [`SnapshotHandle`] — a cheap `Arc` clone
//!   of the currently-published generation — and then executes against the
//!   immutable payload with **no lock held** and no coordination with other
//!   readers or the writer.
//! - **Serialized writer, atomic publish.** [`SnapshotCell::publish`] swaps in a
//!   new generation. Readers in flight keep executing against the generation they
//!   loaded; new loads see the new one.
//! - **Epoch reclamation by refcount.** An old generation's payload is dropped
//!   only when its last [`SnapshotHandle`] is released — a writer can never free a
//!   generation a reader still holds. This is the safety property GPU-resident
//!   snapshots need: device memory backing a generation stays alive until every
//!   reader of that generation has drained.
//!
//! The payload `T` is opaque and only requires `Send + Sync`, so it can carry a
//! GPU-resident read view holding raw device pointers (the production payload),
//! similar to `CudaResidentDeviceMemoryReadView` which declares
//! `unsafe impl Send + Sync`. The `resident_read_view_…` test exercises the
//! *immutable-read sharing* aspect of that case (a leaked-static buffer). It does
//! **not** model the real device-memory lifetime coupling (the read view is valid
//! only while its owning `!Send` allocation lives and its generation has not been
//! engine-invalidated) — that risk is retired only when this substrate is applied
//! to the real resident types.
//!
//! Note: the publish slot is guarded by a `Mutex` whose critical section is a
//! single `Arc` clone/swap (sub-microsecond) — the read *body* runs outside it.
//! A production version may use `arc-swap`/`RwLock` for a fully lock-free load;
//! the ownership and reclamation semantics proven here are identical.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// An immutable published generation: a monotonically increasing id plus the
/// payload readers execute against.
#[derive(Debug)]
pub struct Generation<T> {
    id: u64,
    payload: T,
}

impl<T> Generation<T> {
    /// The generation id (monotonic across publishes).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The immutable payload.
    pub fn payload(&self) -> &T {
        &self.payload
    }
}

/// A reader's handle to one generation. Holding it keeps that generation (and its
/// payload — e.g. device memory) alive regardless of later publishes.
#[derive(Debug, Clone)]
pub struct SnapshotHandle<T> {
    generation: Arc<Generation<T>>,
}

impl<T> SnapshotHandle<T> {
    /// The id of the generation this handle pins.
    pub fn generation(&self) -> u64 {
        self.generation.id
    }

    /// The immutable payload of the pinned generation.
    pub fn get(&self) -> &T {
        &self.generation.payload
    }

    /// Number of live references to this generation (this handle + the cell slot
    /// if still current + any sibling handles). Exposed for reclamation tests.
    pub fn reader_refcount(&self) -> usize {
        Arc::strong_count(&self.generation)
    }
}

/// The published-generation slot. One writer publishes; many readers load.
#[derive(Debug)]
pub struct SnapshotCell<T> {
    current: Mutex<Arc<Generation<T>>>,
    next_id: AtomicU64,
}

impl<T> SnapshotCell<T> {
    /// Create a cell with an initial generation (id 1).
    pub fn new(initial: T) -> Self {
        Self {
            current: Mutex::new(Arc::new(Generation {
                id: 1,
                payload: initial,
            })),
            next_id: AtomicU64::new(2),
        }
    }

    /// Lock the publish slot, recovering from poison. The critical section only
    /// ever clones/swaps an `Arc` (it cannot leave the slot in a torn state), so
    /// continuing past a poisoned lock is safe — and avoids one panicking thread
    /// taking down every reader and the writer.
    fn slot(&self) -> std::sync::MutexGuard<'_, Arc<Generation<T>>> {
        self.current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Load the currently-published generation as a reader handle. The returned
    /// handle pins that generation until dropped. Lock-free read body afterward.
    pub fn load(&self) -> SnapshotHandle<T> {
        let generation = Arc::clone(&self.slot());
        SnapshotHandle { generation }
    }

    /// Publish a new generation and return its id. The previous generation
    /// remains alive for any readers still holding a handle to it.
    ///
    /// The id is allocated **inside** the lock so that generation-id order always
    /// matches install order — making `current_generation()` monotonic even if the
    /// single-writer contract is ever relaxed to multiple concurrent writers.
    pub fn publish(&self, payload: T) -> u64 {
        let mut slot = self.slot();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        *slot = Arc::new(Generation { id, payload });
        id
    }

    /// The id of the currently-published generation.
    pub fn current_generation(&self) -> u64 {
        self.slot().id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn concurrent_readers_always_see_a_consistent_immutable_generation() {
        // Payload is (n, n*7); a torn read would break the invariant. Because each
        // generation is immutable, every reader always sees a consistent pair.
        //
        // Deterministic: each reader does a FIXED number of loads (no stop-flag
        // race — on a fast machine the writer can otherwise finish before readers
        // are even scheduled). Completing all 8 × READS asserts without panic is
        // the proof; the writer publishes concurrently throughout.
        const READS: usize = 5_000;
        let cell = Arc::new(SnapshotCell::new((1_u64, 7_u64)));

        let writer = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || {
                for n in 2..=5_000_u64 {
                    cell.publish((n, n.wrapping_mul(7)));
                }
            })
        };

        let readers: Vec<_> = (0..8)
            .map(|_| {
                let cell = Arc::clone(&cell);
                thread::spawn(move || {
                    for _ in 0..READS {
                        let handle = cell.load();
                        let (n, derived) = *handle.get();
                        assert_eq!(derived, n.wrapping_mul(7), "torn read of generation");
                    }
                })
            })
            .collect();

        writer.join().unwrap();
        for reader in readers {
            reader.join().unwrap();
        }
    }

    #[test]
    fn reads_actually_overlap_no_global_serialization() {
        // Deterministic (not timing-dependent): every reader loads a generation,
        // records the concurrency count, then rendezvouses at a barrier *while
        // still holding its handle*. The barrier cannot release until all READERS
        // are simultaneously in their read body — which is impossible if the load
        // path serialized reads behind one lock. So peak == READERS exactly.
        const READERS: usize = 8;
        let cell = Arc::new(SnapshotCell::new(0_u64));
        let concurrent = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(std::sync::Barrier::new(READERS));

        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                let cell = Arc::clone(&cell);
                let concurrent = Arc::clone(&concurrent);
                let peak = Arc::clone(&peak);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let _handle = cell.load();
                    let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                    let mut observed = peak.load(Ordering::SeqCst);
                    while now > observed {
                        match peak.compare_exchange_weak(
                            observed,
                            now,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        ) {
                            Ok(_) => break,
                            Err(actual) => observed = actual,
                        }
                    }
                    // All readers hold a handle and meet here at once.
                    barrier.wait();
                    concurrent.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();

        for reader in readers {
            reader.join().unwrap();
        }
        assert_eq!(
            peak.load(Ordering::SeqCst),
            READERS,
            "read bodies did not all overlap; the load path is serializing reads"
        );
    }

    #[test]
    fn old_generation_is_retired_only_after_its_last_reader_drains() {
        // Payload records its label on drop, so we can observe reclamation timing.
        struct Tracked {
            label: &'static str,
            dropped: Arc<Mutex<Vec<&'static str>>>,
        }
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.dropped.lock().unwrap().push(self.label);
            }
        }

        let dropped = Arc::new(Mutex::new(Vec::new()));
        let cell = SnapshotCell::new(Tracked {
            label: "g1",
            dropped: Arc::clone(&dropped),
        });

        // A reader pins g1.
        let reader = cell.load();
        assert_eq!(reader.generation(), 1);

        // Writer publishes newer generations. g1 is no longer current, but the
        // reader still holds it, so it must not be dropped yet.
        cell.publish(Tracked {
            label: "g2",
            dropped: Arc::clone(&dropped),
        });
        cell.publish(Tracked {
            label: "g3",
            dropped: Arc::clone(&dropped),
        });
        assert!(
            !dropped.lock().unwrap().contains(&"g1"),
            "g1 was reclaimed while a reader still held it"
        );

        // Draining the reader releases the last reference to g1 → it is reclaimed.
        drop(reader);
        assert!(
            dropped.lock().unwrap().contains(&"g1"),
            "g1 was not reclaimed after its last reader drained"
        );
    }

    #[test]
    fn resident_read_view_with_raw_pointer_is_shared_across_threads() {
        // Models CudaResidentDeviceMemoryReadView: an immutable payload holding a
        // raw device pointer, declared Send + Sync, read concurrently by many
        // threads against one published generation.
        struct ResidentReadView {
            ptr: *const u8,
            len: usize,
        }
        // SAFETY: the pointed-to buffer is immutable and outlives all readers
        // (leaked to 'static below); concurrent reads of immutable memory are
        // sound. This mirrors the engine's resident read-view contract.
        unsafe impl Send for ResidentReadView {}
        unsafe impl Sync for ResidentReadView {}

        let buffer: &'static [u8] = Box::leak(vec![1_u8, 2, 3, 4, 5].into_boxed_slice());
        let cell = Arc::new(SnapshotCell::new(ResidentReadView {
            ptr: buffer.as_ptr(),
            len: buffer.len(),
        }));

        let readers: Vec<_> = (0..8)
            .map(|_| {
                let cell = Arc::clone(&cell);
                thread::spawn(move || {
                    let mut sum = 0_u32;
                    for _ in 0..1_000 {
                        let handle = cell.load();
                        let view = handle.get();
                        // SAFETY: ptr/len describe the leaked immutable buffer.
                        let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
                        sum = bytes.iter().map(|b| u32::from(*b)).sum();
                    }
                    sum
                })
            })
            .collect();

        for reader in readers {
            assert_eq!(reader.join().unwrap(), 15);
        }
    }
}
