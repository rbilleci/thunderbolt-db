//! One opaque, move-only owner for an unreachable indexed INSERT physical reservation.
//!
//! This is deliberately below the existing prepared/bound INSERT-plan flow, not beside it: the
//! owner may retain physical resources through the would-be WAL boundary, but has no canonical
//! operation, WAL encoding, device apply, cache update, descriptor publication, or acknowledgement
//! API.  The only current consumer is test-only scalar inspection.

#![allow(dead_code)] // intentionally unreachable until the one live handoff is designed

use super::index_delta::PreparedIndexedInPlaceReservation;
use super::index_rollover::PreparedIndexedFixedRolloverReservation;

/// The physical branch is opaque outside residency.  Field declaration order in both contained
/// owners is load-bearing: CUDA work/resources drain before append/budget/mutation release, and
/// named-index lifecycle protection releases last. Boxing either nearly-equal branch would add a
/// new statement-retired host allocation; keep the move-only reservation inline until the sole
/// pre-WAL carrier owns and reports that extra allocation domain.
#[allow(clippy::large_enum_variant)]
enum PreparedIndexedPhysicalReservationBranch<'a> {
    InPlace(PreparedIndexedInPlaceReservation<'a>),
    FixedRollover(PreparedIndexedFixedRolloverReservation<'a>),
}

pub(crate) struct PreparedIndexedPhysicalReservation<'a> {
    branch: PreparedIndexedPhysicalReservationBranch<'a>,
}

impl<'a> PreparedIndexedPhysicalReservation<'a> {
    pub(super) fn in_place(reservation: PreparedIndexedInPlaceReservation<'a>) -> Self {
        Self {
            branch: PreparedIndexedPhysicalReservationBranch::InPlace(reservation),
        }
    }

    pub(super) fn fixed_rollover(reservation: PreparedIndexedFixedRolloverReservation<'a>) -> Self {
        Self {
            branch: PreparedIndexedPhysicalReservationBranch::FixedRollover(reservation),
        }
    }
}

#[cfg(test)]
impl PreparedIndexedPhysicalReservation<'_> {
    /// The inspection seam consumes the owner and gives tests only scalar evidence. It cannot
    /// forward the prepared fused launch, private generation, append plan, or lifecycle guard.
    pub(crate) fn inspect_in_place<R>(
        self,
        inspect: impl FnOnce(super::index_delta::IndexedInPlaceProofReport) -> R,
    ) -> Result<R, crate::ExecuteError> {
        let PreparedIndexedPhysicalReservationBranch::InPlace(reservation) = self.branch else {
            return Err(crate::ExecuteError::Serialization(
                "indexed physical reservation selected the wrong scalar inspection".to_string(),
            ));
        };
        reservation.inspect(inspect)
    }

    /// See `inspect_in_place`: fixed rollover exposes no physical capability through this seam.
    pub(crate) fn inspect_fixed_rollover<R>(
        self,
        inspect: impl FnOnce(super::index_rollover::IndexedFixedRolloverProofReport) -> R,
    ) -> Result<R, crate::ExecuteError> {
        let PreparedIndexedPhysicalReservationBranch::FixedRollover(reservation) = self.branch
        else {
            return Err(crate::ExecuteError::Serialization(
                "indexed physical reservation selected the wrong scalar inspection".to_string(),
            ));
        };
        reservation.inspect(inspect)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn production_owner_is_reservation_only_and_has_no_second_publisher() {
        let source = include_str!("indexed_reservation.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production reservation owner precedes tests");
        assert!(source.contains("enum PreparedIndexedPhysicalReservation"));
        for forbidden in [
            "BoundBinaryInsert",
            "WaveCanonicalOperation",
            "apply_resident_open_shard_append",
            "publish_relational_resident_indexes_for_generation",
            "try_append_to_resident_open_shard",
            "encode_",
        ] {
            assert!(
                !source.contains(forbidden),
                "reservation owner must not become a second WAL/apply/publication authority: {forbidden}"
            );
        }
    }
}
