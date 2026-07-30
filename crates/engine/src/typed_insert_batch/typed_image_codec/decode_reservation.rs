//! Raw-pass measurement and exact reservation for the inert typed-image decoder.
//!
//! The parent keeps byte grammar, digest, and typed-vector ownership authoritative.  This leaf
//! owns only the bounded allocation protocol: no raw body escapes and no live result/replay path
//! can reach it.

use super::*;

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
    persistent_slots: u64,
    persistent_limit: Option<u64>,
    scratch_bytes: u64,
    scratch_slots: u64,
    scratch_limit: Option<u64>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecodeReservationStats {
    pub(super) attempts: u64,
    pub(super) persistent_bytes: u64,
    pub(super) persistent_slots: u64,
    pub(super) scratch_bytes: u64,
    pub(super) scratch_slots: u64,
}

#[cfg(test)]
fn note(owner: &str, persistent_bytes: u64, scratch_bytes: u64) -> Result<(), EngineError> {
    // A canonical S2 decode retains these vectors through this image leaf.  Its outer raw-pass
    // budget and failure injection must see the same inner exact owner event; standalone image
    // tests simply have no S2 observation armed, so this remains a no-op there.
    super::super::canonical_codec::note_typed_vector_owner_for_test(
        owner,
        persistent_bytes,
        scratch_bytes,
    )?;
    OBSERVATION.with(|observation| {
        if let Some(mut state) = observation.get() {
            state.attempts = state
                .attempts
                .checked_add(1)
                .expect("test decode reservation counter overflow");
            state.persistent_bytes = state
                .persistent_bytes
                .checked_add(persistent_bytes)
                .expect("test decode persistent byte counter overflow");
            state.persistent_slots = state
                .persistent_slots
                .checked_add(u64::from(persistent_bytes != 0))
                .expect("test decode persistent slot counter overflow");
            state.scratch_bytes = state.scratch_bytes.max(scratch_bytes);
            state.scratch_slots = state.scratch_slots.max(u64::from(scratch_bytes != 0));
            observation.set(Some(state));
            if state.fail_at == Some(state.attempts) {
                return Err(image_error("injected decoded-owner reservation failure"));
            }
            if state
                .persistent_limit
                .is_some_and(|limit| state.persistent_bytes > limit)
            {
                return Err(image_error(
                    "decoded persistent reservation exceeds measured bound",
                ));
            }
            if state
                .scratch_limit
                .is_some_and(|limit| state.scratch_bytes > limit)
            {
                return Err(image_error(
                    "decoded scratch reservation exceeds measured bound",
                ));
            }
        }
        let _ = owner;
        Ok(())
    })
}

#[cfg(test)]
pub(super) fn observe_decode_reservations_for_test<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    OBSERVATION.with(|observation| {
        assert!(
            observation
                .replace(Some(TestState {
                    attempts: 0,
                    fail_at: None,
                    persistent_bytes: 0,
                    persistent_slots: 0,
                    persistent_limit: None,
                    scratch_bytes: 0,
                    scratch_slots: 0,
                    scratch_limit: None,
                }))
                .is_none(),
            "test decode reservation observation cannot nest"
        );
        let result = operation();
        let attempts = observation
            .replace(None)
            .expect("test decode reservation observation remains armed")
            .attempts;
        (result, attempts)
    })
}

#[cfg(test)]
pub(super) fn observe_decode_reservation_stats_for_test<T>(
    operation: impl FnOnce() -> T,
) -> (T, DecodeReservationStats) {
    OBSERVATION.with(|observation| {
        assert!(
            observation
                .replace(Some(TestState {
                    attempts: 0,
                    fail_at: None,
                    persistent_bytes: 0,
                    persistent_slots: 0,
                    persistent_limit: None,
                    scratch_bytes: 0,
                    scratch_slots: 0,
                    scratch_limit: None,
                }))
                .is_none(),
            "test decode reservation observation cannot nest"
        );
        let result = operation();
        let state = observation
            .replace(None)
            .expect("test decode reservation observation remains armed");
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
pub(super) fn fail_decode_reservation_for_test<T>(
    fail_at: u64,
    persistent_limit: Option<u64>,
    scratch_limit: Option<u64>,
    operation: impl FnOnce() -> T,
) -> T {
    assert_ne!(fail_at, 0, "decode reservation injection is one-based");
    OBSERVATION.with(|observation| {
        assert!(
            observation
                .replace(Some(TestState {
                    attempts: 0,
                    fail_at: Some(fail_at),
                    persistent_bytes: 0,
                    persistent_slots: 0,
                    persistent_limit,
                    scratch_bytes: 0,
                    scratch_slots: 0,
                    scratch_limit,
                }))
                .is_none(),
            "test decode reservation injection cannot nest"
        );
        let result = operation();
        observation
            .replace(None)
            .expect("test decode reservation injection remains armed");
        result
    })
}

pub(super) fn reserve_decode_exact<T>(
    values: &mut Vec<T>,
    count: usize,
    persistent_bytes: u64,
    scratch_bytes: u64,
    owner: &str,
) -> Result<(), EngineError> {
    #[cfg(test)]
    note(owner, persistent_bytes, scratch_bytes)?;
    #[cfg(not(test))]
    let _ = (owner, persistent_bytes, scratch_bytes);
    values
        .try_reserve_exact(count)
        .map_err(|_| image_error("decoded owner reservation failed"))
}

pub(super) fn reserve_decode_string(
    value: &mut String,
    count: usize,
    persistent_bytes: u64,
    owner: &str,
) -> Result<(), EngineError> {
    #[cfg(test)]
    note(owner, persistent_bytes, 0)?;
    #[cfg(not(test))]
    let _ = (owner, persistent_bytes);
    value
        .try_reserve_exact(count)
        .map_err(|_| image_error("decoded name reservation failed"))
}

/// `Vec::into_boxed_slice` and `String::into_boxed_str` are allowed to shrink a spare capacity,
/// which would be a hidden post-pass allocation.  Every decoder vector/string is reserved to its
/// raw-pass exact length, and this guard turns any allocator-capacity surprise into a clean
/// refusal before the conversion.
pub(super) fn into_exact_boxed_slice<T>(
    values: Vec<T>,
    owner: &str,
) -> Result<Box<[T]>, EngineError> {
    if values.len() != values.capacity() {
        return Err(image_error(
            "decoded vector capacity is not exact before boxing",
        ));
    }
    let _ = owner;
    Ok(values.into_boxed_slice())
}

pub(super) fn into_exact_boxed_str(value: String, owner: &str) -> Result<Box<str>, EngineError> {
    if value.len() != value.capacity() {
        return Err(image_error(
            "decoded string capacity is not exact before boxing",
        ));
    }
    let _ = owner;
    Ok(value.into_boxed_str())
}

/// Allocation-free raw-pass evidence for a strict typed-image decode.  Persistent terms name
/// only owners retained by `DecodedTypedImage`; the descriptor directory is reusable scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypedImageDecodeMeasure {
    image_bytes: u64,
    content_fingerprint: gpu_db_wal::CanonicalDigest,
    persistent_bytes: u64,
    persistent_allocation_slots: u64,
    maximum_scratch_bytes: u64,
    maximum_scratch_allocation_slots: u64,
}

impl TypedImageDecodeMeasure {
    pub(crate) fn image_bytes(self) -> u64 {
        self.image_bytes
    }

    /// Immutable raw-source content identity captured by the allocation-free image pass.
    /// S7 uses it only to bind an exact copy to the measured borrowed image source.
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

    /// Peak bytes while an aggregate-owned exact raw image copy is live beside this decoder's
    /// transient scratch.  The raw copy is caller-owned and deliberately excluded from the
    /// standalone image measure.
    pub(crate) fn maximum_with_image_copy_bytes(self) -> Result<u64, EngineError> {
        self.image_bytes
            .checked_add(self.maximum_scratch_bytes)
            .ok_or_else(|| image_error("image copy plus decoder scratch overflows"))
    }

    /// Allocation slots matching [`Self::maximum_with_image_copy_bytes`].
    pub(crate) fn maximum_with_image_copy_allocation_slots(self) -> Result<u64, EngineError> {
        self.maximum_scratch_allocation_slots
            .checked_add(u64::from(self.image_bytes != 0))
            .ok_or_else(|| image_error("image copy plus decoder scratch slot count overflows"))
    }

    pub(super) fn from_raw_parts(
        image_bytes: u64,
        content_fingerprint: gpu_db_wal::CanonicalDigest,
        persistent_bytes: u64,
        persistent_allocation_slots: u64,
        maximum_scratch_bytes: u64,
        maximum_scratch_allocation_slots: u64,
    ) -> Self {
        Self {
            image_bytes,
            content_fingerprint,
            persistent_bytes,
            persistent_allocation_slots,
            maximum_scratch_bytes,
            maximum_scratch_allocation_slots,
        }
    }
}

pub(super) fn measure_decoded_typed_image(
    bytes: &[u8],
) -> Result<TypedImageDecodeMeasure, EngineError> {
    super::read_at::measure_decoded_typed_image_from_source(&super::read_at::SliceImageSource::new(
        bytes,
    ))
}

pub(super) fn decode_typed_image_after_measure(
    bytes: &[u8],
    measure: TypedImageDecodeMeasure,
) -> Result<DecodedTypedImage, EngineError> {
    if measure_decoded_typed_image(bytes)? != measure {
        return Err(image_error("decoded image reservation measure drifted"));
    }
    let (header, layout) = decoded_image_layout(bytes)?;
    let descriptors = parse_descriptors_after_measure(&layout)?;
    let mut owned = Vec::new();
    reserve_decode_exact(
        &mut owned,
        layout.columns,
        sized_owner_bytes::<DecodedTypedImageColumn>(layout.columns)?,
        0,
        "decoded column directory",
    )?;
    for descriptor in descriptors {
        let name = descriptor_name(bytes, &descriptor)?;
        let vector = vector_bytes(bytes, &descriptor)?;
        let (validity, validity_len) = decode_typed_validity(vector, header.rows)?;
        let (values, value_len) = decode_typed_values(
            vector
                .get(validity_len..)
                .ok_or_else(|| image_error("vector value boundary is truncated"))?,
            descriptor.ty,
            header.rows,
        )?;
        if validity_len
            .checked_add(value_len)
            .and_then(|length| (length == vector.len()).then_some(length))
            .is_none()
        {
            return Err(image_error("vector length has trailing bytes"));
        }
        if typed_vector_digest(&validity, &values, descriptor.ty, header.rows)?
            != descriptor.vector_digest
        {
            return Err(image_error("vector digest drifted"));
        }
        validate_invalid_placeholders(&validity, &values, header.rows)?;
        owned.push(DecodedTypedImageColumn {
            catalog_column_ordinal: descriptor.catalog_column_ordinal,
            stable_column_id: descriptor.stable_column_id,
            table_ref: descriptor.table_ref,
            attnum: descriptor.attnum,
            ty: descriptor.ty,
            type_oid: descriptor.type_oid,
            type_size: descriptor.type_size,
            result_format: descriptor.result_format,
            name,
            validity,
            values,
            vector_digest: descriptor.vector_digest,
        });
    }
    Ok(DecodedTypedImage {
        facts: DecodedTypedImageFacts {
            role: header.role,
            rows: header.rows,
            columns: header.columns,
            layout_digest: header.layout_digest,
        },
        columns: into_exact_boxed_slice(owned, "decoded column directory")?,
    })
}
