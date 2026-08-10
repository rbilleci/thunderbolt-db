//! One opaque, move-only owner for an indexed INSERT physical reservation.
//!
//! This is deliberately below the existing prepared/bound INSERT-plan flow, not beside it: the
//! owner retains physical resources through the WAL boundary while the generic codec-5 terminal
//! remains the only canonical operation, WAL, status, apply, publication, and acknowledgement
//! authority. Both in-place and fixed-rollover branches converge on that same terminal handoff.

use super::index_delta::PreparedIndexedInPlaceReservation;
use super::index_rollover::{
    PreparedIndexedDenseRolloverReservation, PreparedIndexedFixedRolloverReservation,
};

/// The physical branch is opaque outside residency.  Field declaration order in both contained
/// owners is load-bearing: CUDA work/resources drain before append/budget/mutation release, and
/// named-index lifecycle protection releases last. Boxing either nearly-equal branch would add a
/// new statement-retired host allocation; keep the move-only reservation inline until the sole
/// pre-WAL carrier owns and reports that extra allocation domain.
#[allow(clippy::large_enum_variant)]
enum PreparedIndexedPhysicalReservationBranch<'a> {
    InPlace(PreparedIndexedInPlaceReservation<'a>),
    FixedRollover(PreparedIndexedFixedRolloverReservation<'a>),
    DenseRollover(PreparedIndexedDenseRolloverReservation<'a>),
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

    pub(super) fn dense_rollover(reservation: PreparedIndexedDenseRolloverReservation<'a>) -> Self {
        Self {
            branch: PreparedIndexedPhysicalReservationBranch::DenseRollover(reservation),
        }
    }

    pub(super) fn take_named_index_publication_guard(
        &mut self,
    ) -> Option<crate::engine_state::TransactionNamedIndexPublicationGuard<'a>> {
        match &mut self.branch {
            PreparedIndexedPhysicalReservationBranch::InPlace(reservation) => {
                reservation.take_named_index_publication_guard()
            }
            PreparedIndexedPhysicalReservationBranch::FixedRollover(reservation) => {
                reservation.take_named_index_publication_guard()
            }
            PreparedIndexedPhysicalReservationBranch::DenseRollover(reservation) => {
                reservation.take_named_index_publication_guard()
            }
        }
    }

    pub(super) fn take_transaction_terminal_device_apply_guard(
        &mut self,
    ) -> Option<std::sync::MutexGuard<'a, ()>> {
        match &mut self.branch {
            PreparedIndexedPhysicalReservationBranch::InPlace(reservation) => {
                reservation.take_transaction_terminal_device_apply_guard()
            }
            PreparedIndexedPhysicalReservationBranch::FixedRollover(reservation) => {
                reservation.take_transaction_terminal_device_apply_guard()
            }
            PreparedIndexedPhysicalReservationBranch::DenseRollover(reservation) => {
                reservation.take_transaction_terminal_device_apply_guard()
            }
        }
    }

    pub(super) fn take_transaction_terminal_budget_guard(
        &mut self,
    ) -> Option<(std::sync::MutexGuard<'a, ()>, u64)> {
        match &mut self.branch {
            PreparedIndexedPhysicalReservationBranch::InPlace(reservation) => {
                reservation.take_transaction_terminal_budget_guard()
            }
            PreparedIndexedPhysicalReservationBranch::FixedRollover(reservation) => {
                reservation.take_transaction_terminal_budget_guard()
            }
            PreparedIndexedPhysicalReservationBranch::DenseRollover(reservation) => {
                reservation.take_transaction_terminal_budget_guard()
            }
        }
    }

    /// Return the data-only successor roots only for a rollover branch. In-place maintenance has
    /// no global manifest swap and therefore cannot act as a predecessor for another rollover.
    pub(super) fn rollover_manifest_successor_predecessor(
        &self,
    ) -> Option<super::prepared_table_index_manifest::PreparedIndexedRolloverManifestPredecessor>
    {
        match &self.branch {
            PreparedIndexedPhysicalReservationBranch::InPlace(_) => None,
            PreparedIndexedPhysicalReservationBranch::FixedRollover(reservation) => {
                Some(reservation.manifest_successor_predecessor())
            }
            PreparedIndexedPhysicalReservationBranch::DenseRollover(reservation) => {
                Some(reservation.manifest_successor_predecessor())
            }
        }
    }

    pub(super) fn apply_after_transaction_wal_claim(
        self,
        engine: &crate::Engine,
        created_by: super::AppendCreatedBy<'_>,
    ) -> Result<(), super::DeviceInsertPlanApplyError> {
        match self.branch {
            PreparedIndexedPhysicalReservationBranch::InPlace(reservation) => {
                reservation.apply_after_transaction_wal_claim(engine, created_by)
            }
            PreparedIndexedPhysicalReservationBranch::FixedRollover(reservation) => {
                reservation.apply_after_transaction_wal_claim(engine, created_by)
            }
            PreparedIndexedPhysicalReservationBranch::DenseRollover(reservation) => {
                reservation.apply_after_transaction_wal_claim(engine, created_by)
            }
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
