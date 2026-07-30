//! Exact decoded-owner reservation seam for canonical typed-INSERT S2 records.
//!
//! This child deliberately owns no record grammar or raw bytes.  The parent decoder names each
//! retained vector owner and calls this fallible seam only after its bounded raw pass has proved
//! the corresponding count/region geometry.

use super::*;

/// Exact source-pass ownership evidence for one strict S2 record.  `maximum_scratch_bytes`
/// excludes the caller-owned raw record copy: aggregate recovery must retain that copy while the
/// strict decoder performs its digest/reencode comparisons, so its peak is their sum.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalTypedInsertDecodeMeasure {
    record_bytes: u64,
    content_fingerprint: gpu_db_wal::CanonicalDigest,
    persistent_bytes: u64,
    persistent_allocation_slots: u64,
    maximum_scratch_bytes: u64,
    maximum_scratch_allocation_slots: u64,
}

#[allow(dead_code)]
impl CanonicalTypedInsertDecodeMeasure {
    pub(crate) fn record_bytes(self) -> u64 {
        self.record_bytes
    }
    /// Immutable raw-source content identity captured by the allocation-free S2 pass.
    ///
    /// Aggregate S7 ownership uses this only to prove that its exact copied record still
    /// represents the measured source; it does not expose raw bytes or decoded vectors.
    pub(crate) fn content_fingerprint(self) -> gpu_db_wal::CanonicalDigest {
        self.content_fingerprint
    }
    pub(crate) fn persistent_bytes(self) -> u64 {
        self.persistent_bytes
    }
    pub(crate) fn persistent_allocation_slots(self) -> u64 {
        self.persistent_allocation_slots
    }
    pub(crate) fn maximum_scratch_bytes(self) -> u64 {
        self.maximum_scratch_bytes
    }
    pub(crate) fn maximum_scratch_allocation_slots(self) -> u64 {
        self.maximum_scratch_allocation_slots
    }
    pub(crate) fn maximum_with_record_copy_bytes(self) -> Result<u64, EngineError> {
        self.record_bytes
            .checked_add(self.maximum_scratch_bytes)
            .ok_or_else(|| codec_error("S2 copy plus decoder scratch overflows"))
    }
    pub(crate) fn maximum_with_record_copy_allocation_slots(self) -> Result<u64, EngineError> {
        self.maximum_scratch_allocation_slots
            .checked_add(u64::from(self.record_bytes != 0))
            .ok_or_else(|| codec_error("S2 copy plus decoder scratch slot count overflows"))
    }
    pub(super) fn new(
        record_bytes: u64,
        content_fingerprint: gpu_db_wal::CanonicalDigest,
        persistent_bytes: u64,
        persistent_allocation_slots: u64,
        maximum_scratch_bytes: u64,
        maximum_scratch_allocation_slots: u64,
    ) -> Self {
        Self {
            record_bytes,
            content_fingerprint,
            persistent_bytes,
            persistent_allocation_slots,
            maximum_scratch_bytes,
            maximum_scratch_allocation_slots,
        }
    }
}

#[cfg(test)]
thread_local! {
    static OBSERVATION: std::cell::Cell<Option<TestState>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct TestState {
    attempts: u64,
    fail_at: Option<u64>,
    persistent_bytes: u64,
    persistent_limit: Option<u64>,
    persistent_slots: u64,
    persistent_slot_limit: Option<u64>,
    scratch_bytes: u64,
    scratch_limit: Option<u64>,
    scratch_slots: u64,
    scratch_slot_limit: Option<u64>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodeReservationStats {
    pub(crate) attempts: u64,
    pub(crate) persistent_bytes: u64,
    pub(crate) persistent_slots: u64,
    pub(crate) scratch_bytes: u64,
    pub(crate) scratch_slots: u64,
}

#[cfg(test)]
fn note(
    owner: &str,
    persistent_bytes: u64,
    persistent_slots: u64,
    scratch_bytes: u64,
    scratch_slots: u64,
) -> Result<(), EngineError> {
    OBSERVATION.with(|observation| {
        if let Some(mut state) = observation.get() {
            state.attempts = state
                .attempts
                .checked_add(1)
                .expect("test S2 reservation counter overflow");
            state.persistent_bytes = state
                .persistent_bytes
                .checked_add(persistent_bytes)
                .expect("test S2 persistent byte counter overflow");
            state.persistent_slots = state
                .persistent_slots
                .checked_add(persistent_slots)
                .expect("test S2 persistent slot counter overflow");
            state.scratch_bytes = state.scratch_bytes.max(scratch_bytes);
            state.scratch_slots = state.scratch_slots.max(scratch_slots);
            observation.set(Some(state));
            if state.fail_at == Some(state.attempts) {
                return Err(codec_error("injected S2 decoded-owner reservation failure"));
            }
            if state
                .persistent_limit
                .is_some_and(|limit| state.persistent_bytes > limit)
            {
                return Err(codec_error(
                    "decoded persistent reservation exceeds measured bound",
                ));
            }
            if state
                .persistent_slot_limit
                .is_some_and(|limit| state.persistent_slots > limit)
            {
                return Err(codec_error(
                    "decoded persistent allocation slots exceed measured bound",
                ));
            }
            if state
                .scratch_limit
                .is_some_and(|limit| state.scratch_bytes > limit)
            {
                return Err(codec_error(
                    "decoded scratch reservation exceeds measured bound",
                ));
            }
            if state
                .scratch_slot_limit
                .is_some_and(|limit| state.scratch_slots > limit)
            {
                return Err(codec_error(
                    "decoded scratch allocation slots exceed measured bound",
                ));
            }
        }
        let _ = owner;
        Ok(())
    })
}

pub(super) fn reserve_exact<T>(
    values: &mut Vec<T>,
    count: usize,
    owner: &str,
) -> Result<(), EngineError> {
    #[cfg(test)]
    if count != 0 {
        note(
            owner,
            u64::try_from(count)
                .ok()
                .and_then(|count| {
                    count.checked_mul(
                        u64::try_from(std::mem::size_of::<T>()).expect("type size fits u64"),
                    )
                })
                .ok_or_else(|| codec_error("decoded owner bytes overflow"))?,
            1,
            0,
            0,
        )?;
    }
    #[cfg(not(test))]
    let _ = owner;
    values
        .try_reserve_exact(count)
        .map_err(|_| codec_error("decoded owner reservation failed"))
}

pub(super) fn reserve_string(
    value: &mut String,
    count: usize,
    owner: &str,
) -> Result<(), EngineError> {
    #[cfg(test)]
    note(
        owner,
        u64::try_from(count).map_err(|_| codec_error("decoded string bytes overflow"))?,
        u64::from(count != 0),
        0,
        0,
    )?;
    #[cfg(not(test))]
    let _ = owner;
    value
        .try_reserve_exact(count)
        .map_err(|_| codec_error("decoded string owner reservation failed"))
}

/// The canonicality reencode is transient compare scratch, not retained decoded evidence.  It
/// still participates in injection/accounting before `Writer::exact` reserves its one buffer.
pub(super) fn reserve_reencode_scratch(bytes: usize) -> Result<(), EngineError> {
    #[cfg(test)]
    note(
        "decoded canonical reencode scratch",
        0,
        0,
        u64::try_from(bytes).map_err(|_| codec_error("reencode scratch bytes overflow"))?,
        u64::from(bytes != 0),
    )?;
    #[cfg(not(test))]
    let _ = bytes;
    Ok(())
}

/// Typed vectors stay owned by their image codec leaf, while S2 owns their aggregate budget.
/// Feed its exact test-only reservation events into the S2 injection stream so source evidence
/// accounts for every retained S2 owner.
#[cfg(test)]
pub(crate) fn note_typed_vector_owner_for_test(
    owner: &str,
    persistent_bytes: u64,
    scratch_bytes: u64,
) -> Result<(), EngineError> {
    if persistent_bytes == 0 && scratch_bytes == 0 {
        return Ok(());
    }
    note(
        owner,
        persistent_bytes,
        u64::from(persistent_bytes != 0),
        scratch_bytes,
        u64::from(scratch_bytes != 0),
    )
}

/// Canonical decoded owners are converted to the engine's Box-backed sealed-vector ABI only
/// after an exact reservation.  Refuse spare capacity so `into_boxed_slice` cannot make a hidden
/// allocation during the post-raw-pass conversion.
pub(super) fn into_exact_boxed_slice<T>(values: Vec<T>) -> Result<Box<[T]>, EngineError> {
    if values.len() != values.capacity() {
        return Err(codec_error(
            "decoded boxed-vector capacity is not exact before conversion",
        ));
    }
    Ok(values.into_boxed_slice())
}

/// Some decoder-private catalogs intentionally retain `Vec` because their narrow borrowed view
/// iterators need stable slices, not a batch-builder conversion.  Keep the same exact-capacity
/// invariant as Box owners so raw-pass accounting never understates retained capacity.
pub(super) fn require_exact_vec<T>(values: &Vec<T>) -> Result<(), EngineError> {
    if values.len() != values.capacity() {
        return Err(codec_error(
            "decoded vector capacity is not exact before retention",
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn observe_stats_for_test<T>(
    operation: impl FnOnce() -> T,
) -> (T, DecodeReservationStats) {
    OBSERVATION.with(|observation| {
        assert!(
            observation
                .replace(Some(TestState {
                    attempts: 0,
                    fail_at: None,
                    persistent_bytes: 0,
                    persistent_limit: None,
                    persistent_slots: 0,
                    persistent_slot_limit: None,
                    scratch_bytes: 0,
                    scratch_limit: None,
                    scratch_slots: 0,
                    scratch_slot_limit: None,
                }))
                .is_none(),
            "S2 reservation observation cannot nest"
        );
        let result = operation();
        let state = observation
            .replace(None)
            .expect("S2 reservation observation remains armed");
        (
            result,
            DecodeReservationStats {
                attempts: state.attempts,
                persistent_bytes: state.persistent_bytes,
                persistent_slots: state.persistent_slots,
                scratch_bytes: state.scratch_bytes,
                scratch_slots: state.scratch_slots,
            },
        )
    })
}

#[cfg(test)]
pub(crate) fn fail_at_for_test<T>(fail_at: u64, operation: impl FnOnce() -> T) -> T {
    fail_with_limits_for_test(fail_at, None, None, None, None, operation)
}

#[cfg(test)]
pub(crate) fn fail_with_limits_for_test<T>(
    fail_at: u64,
    persistent_limit: Option<u64>,
    persistent_slot_limit: Option<u64>,
    scratch_limit: Option<u64>,
    scratch_slot_limit: Option<u64>,
    operation: impl FnOnce() -> T,
) -> T {
    assert_ne!(fail_at, 0, "S2 reservation injection is one-based");
    OBSERVATION.with(|observation| {
        assert!(
            observation
                .replace(Some(TestState {
                    attempts: 0,
                    fail_at: Some(fail_at),
                    persistent_bytes: 0,
                    persistent_limit,
                    persistent_slots: 0,
                    persistent_slot_limit,
                    scratch_bytes: 0,
                    scratch_limit,
                    scratch_slots: 0,
                    scratch_slot_limit,
                }))
                .is_none(),
            "S2 reservation injection cannot nest"
        );
        let result = operation();
        observation
            .replace(None)
            .expect("S2 reservation injection remains armed");
        result
    })
}
