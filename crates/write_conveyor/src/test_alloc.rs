//! One thread-scoped allocator probe shared by write-conveyor unit tests.
//!
//! This test binary has a single global allocator.  Individual proofs opt in through the
//! thread-local scope so background I/O workers and parallel tests do not contaminate a bounded
//! post-handoff assertion.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct ThreadScopedCountingAllocator;

std::thread_local! {
    static ALLOCATION_COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation() {
    let _ = ALLOCATION_COUNT.try_with(|count| {
        if let Some(current) = count.get() {
            count.set(Some(current.saturating_add(1)));
        }
    });
}

unsafe impl GlobalAlloc for ThreadScopedCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static THREAD_SCOPED_COUNTING_ALLOCATOR: ThreadScopedCountingAllocator =
    ThreadScopedCountingAllocator;

struct AllocationScope;

impl AllocationScope {
    fn begin() -> Self {
        ALLOCATION_COUNT.with(|count| {
            assert!(
                count.get().is_none(),
                "thread-scoped allocation assertions may not nest"
            );
            count.set(Some(0));
        });
        Self
    }

    fn finish(self) -> usize {
        ALLOCATION_COUNT.with(|count| {
            count
                .replace(None)
                .expect("thread-scoped allocation assertion remains active")
        })
    }
}

impl Drop for AllocationScope {
    fn drop(&mut self) {
        let _ = ALLOCATION_COUNT.try_with(|count| count.set(None));
    }
}

pub(crate) fn assert_no_allocations<T>(operation: impl FnOnce() -> T) -> T {
    let scope = AllocationScope::begin();
    let result = operation();
    assert_eq!(
        scope.finish(),
        0,
        "bounded post-handoff operation allocated on its host thread"
    );
    result
}
