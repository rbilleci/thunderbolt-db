//! Borrowed bounded sources and exact transient copies for retained S2/S7 decoding.
//!
//! These adapters deliberately expose no aggregate slice. A strict subdecoder can reread only
//! its measured region, and every temporary copy is dropped immediately after its move-only
//! decoded owner enters the already-reserved graph.

use crate::typed_insert_aggregate::codec::DecodedAggregateFraming;
use crate::typed_insert_batch::{CanonicalTypedInsertReadAt, TypedImageReadAt};
use crate::EngineError;

pub(super) struct AggregateRegionSource<'a> {
    framing: &'a DecodedAggregateFraming<'a>,
    section: usize,
    start: u64,
    bytes: u64,
}

impl<'a> AggregateRegionSource<'a> {
    pub(super) fn new(
        framing: &'a DecodedAggregateFraming<'a>,
        section: usize,
        start: u64,
        bytes: u64,
    ) -> Self {
        Self {
            framing,
            section,
            start,
            bytes,
        }
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let requested = u64::try_from(out.len())
            .map_err(|_| fill_error("region source read length overflows"))?;
        offset
            .checked_add(requested)
            .filter(|end| *end <= self.bytes)
            .ok_or_else(|| fill_error("region source read is outside its measured bytes"))?;
        self.framing.with_section_reader(self.section, |reader| {
            reader.skip(
                self.start
                    .checked_add(offset)
                    .ok_or_else(|| fill_error("region source offset overflows"))?,
            )?;
            reader.copy_exact(out)?;
            reader.skip(reader.remaining())?;
            Ok(())
        })
    }
}

impl CanonicalTypedInsertReadAt for AggregateRegionSource<'_> {
    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        self.read(offset, out)
    }
}

impl TypedImageReadAt for AggregateRegionSource<'_> {
    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        self.read(offset, out)
    }
}

/// A fallible exact scratch owner. `resize` is allocation-free after the exact capacity proof;
/// no raw aggregate region survives the caller's immediate strict decoder invocation.
pub(super) fn exact_copy_scratch(bytes: u64, owner: &str) -> Result<Vec<u8>, EngineError> {
    let len = usize::try_from(bytes)
        .map_err(|_| fill_error("measured scratch bytes exceed host addressability"))?;
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(len)
        .map_err(|_| fill_error("measured source scratch reservation failed"))?;
    if scratch.capacity() != len {
        return Err(fill_error("measured source scratch capacity is not exact"));
    }
    #[cfg(test)]
    note_copy(owner)?;
    #[cfg(not(test))]
    let _ = owner;
    scratch.resize(len, 0);
    Ok(scratch)
}

pub(super) fn fill_error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 retained fill: {message}"
    ))
}

#[cfg(test)]
thread_local! {
    static COPY_FAILURE: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static COPY_ATTEMPTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_copy(owner: &str) -> Result<(), EngineError> {
    // Ordinary fills must not leave test-injection state behind for a later scoped failure.
    // Only an armed injection observes copy attempts.
    if COPY_FAILURE.with(|failure| failure.get()).is_none() {
        return Ok(());
    }
    COPY_ATTEMPTS.with(|attempts| {
        let attempt = attempts
            .get()
            .checked_add(1)
            .expect("retained fill copy attempt counter overflow");
        attempts.set(attempt);
        if COPY_FAILURE.with(|failure| failure.get()) == Some(attempt) {
            return Err(fill_error(&format!(
                "injected strict source-copy failure at {owner}"
            )));
        }
        Ok(())
    })
}

#[cfg(test)]
pub(super) fn fail_copy_at_for_test<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    assert_ne!(attempt, 0, "retained source-copy injection is one-based");
    COPY_FAILURE.with(|failure| {
        COPY_ATTEMPTS.with(|attempts| {
            assert!(failure.replace(Some(attempt)).is_none());
            assert_eq!(
                attempts.replace(0),
                0,
                "retained copy injection cannot nest"
            );
            let result = operation();
            failure.set(None);
            attempts.set(0);
            result
        })
    })
}
