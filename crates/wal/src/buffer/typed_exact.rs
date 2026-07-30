//! Move-only exact typed-record ownership beneath the stable `WalBuffer` facade.
//!
//! This leaf owns only the preproposal exact record reservation and its tentative -> claim or
//! rollback transition.  Durable group formation stays in the parent because it owns both serial
//! and FUA persistence; it calls the two private helpers here to reject tentative tails and to
//! reuse already-serialized typed bytes.

use super::group::TypedExactPendingCredit;
use super::{CanonicalCatalogTailCache, DurableIdentityBinding, WalBuffer};
use crate::canonical::ExactCanonicalRecordAuthority;
use crate::{EngineError, PreparedCanonicalWalRecord};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct ExactTypedWireRecord {
    record_index: usize,
    serialized_record: Arc<[u8]>,
    state: ExactTypedWireState,
    /// This scalar owner was reserved before proposal.  It moves into the scatter group with the
    /// immutable serialized record and is released only after durable success or fail-closed
    /// poisoning — never by a post-claim fallback/reprepare path.
    pending_group_credit: TypedExactPendingCredit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactTypedWireState {
    Tentative,
    ClaimedPending,
    InFlight,
}

impl ExactTypedWireRecord {
    fn tentative(
        record_index: usize,
        serialized_record: Arc<[u8]>,
        pending_group_credit: TypedExactPendingCredit,
    ) -> Self {
        Self {
            record_index,
            serialized_record,
            state: ExactTypedWireState::Tentative,
            pending_group_credit,
        }
    }

    pub(super) fn record_index(&self) -> usize {
        self.record_index
    }

    pub(super) fn serialized_len(&self) -> usize {
        self.serialized_record.len()
    }

    pub(super) fn serialized_bytes(&self) -> &[u8] {
        &self.serialized_record
    }

    fn is_tentative(&self) -> bool {
        self.state == ExactTypedWireState::Tentative
    }

    fn is_claimed_pending(&self) -> bool {
        self.state == ExactTypedWireState::ClaimedPending
    }

    pub(super) fn mark_in_flight(&mut self) {
        assert_eq!(
            self.state,
            ExactTypedWireState::ClaimedPending,
            "only a claimed pending typed WAL owner may enter a durability group"
        );
        self.state = ExactTypedWireState::InFlight;
    }

    fn into_pending_group_credit(self) -> TypedExactPendingCredit {
        self.pending_group_credit
    }
}

/// One move-only typed canonical record owner, reserved before replication proposal.
#[must_use = "a typed exact WAL reservation must be claimed or rolled back before it is dropped"]
pub struct WalTypedExactAppendReservation {
    expected_owner_id: u64,
    expected_records_len: usize,
    expected_records_capacity: usize,
    expected_exact_wire_len: usize,
    expected_exact_wire_capacity: usize,
    expected_generation: u64,
    expected_serialized_len: usize,
    prior_catalog_tail: CanonicalCatalogTailCache,
    /// Existing logical history was decoded/anchor-checked before proposal.  Tentative append
    /// advances this cursor using the already-validated exact header; rollback restores it with
    /// the record frontier rather than asking post-claim durability to decode anything.
    prior_durable_identity_binding: DurableIdentityBinding,
    /// Retained only while this one record is rollbackable.  Claimed records keep the compact
    /// serialized Arc sidecar, not a duplicate header/encoding proof per logical WAL entry.
    rollback_authority: Option<ExactCanonicalRecordAuthority>,
    prepared: Option<PreparedCanonicalWalRecord>,
    pending_group_credit: Option<TypedExactPendingCredit>,
    /// FUA's physical cursor/controller proof stays with the tentative owner until its logical
    /// claim commits the extension.  Dropping it restores the prior forming-group seal exactly.
    #[cfg(unix)]
    fua_exact_reservation: Option<crate::fua::FuaExactReservation>,
    state: TypedExactAppendState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypedExactAppendState {
    Prepared,
    Tentative {
        position: usize,
        expected_generation: u64,
    },
}

impl WalTypedExactAppendReservation {
    /// Arc-identical packed payload for replication.  The outer serialized record never crosses
    /// this boundary because replication/state-machine readers consume canonical payload bytes.
    pub fn replication_payload(&self) -> Result<Arc<[u8]>, EngineError> {
        self.prepared
            .as_ref()
            .ok_or_else(|| {
                EngineError::ProposalFailed(
                    "typed exact WAL replication payload was consumed before proposal".to_string(),
                )
            })?
            .exact_packed_payload()
            .cloned()
            .ok_or_else(|| {
                EngineError::ProposalFailed(
                    "typed exact WAL reservation lost its packed replication payload".to_string(),
                )
            })
    }
}

impl WalBuffer {
    /// Reserve typed INSERT's exact record before replication proposal.
    pub fn reserve_typed_exact_append(
        &mut self,
        prepared: PreparedCanonicalWalRecord,
    ) -> Result<WalTypedExactAppendReservation, EngineError> {
        self.prune_durable_typed_exact_wire_records();
        if self.typed_exact_tentative_count != 0 {
            return Err(EngineError::ProposalFailed(
                "typed exact WAL group admission is busy behind a tentative pre-claim owner"
                    .to_string(),
            ));
        }
        let exact = prepared.exact_authority().ok_or_else(|| {
            EngineError::Durability(
                "typed exact WAL reservation requires a prebuilt exact canonical record"
                    .to_string(),
            )
        })?;
        let payload = prepared
            .exact_packed_payload()
            .expect("exact authority checked above");
        let expected_serialized_len =
            usize::try_from(exact.encoding().footprint.serialized_record_bytes).map_err(|_| {
                EngineError::Durability(
                    "typed exact serialized record length exceeds addressable memory".to_string(),
                )
            })?;
        if exact.serialized_record().len() != expected_serialized_len
            || exact
                .serialized_record()
                .get(crate::WAL_RECORD_HEADER_LEN..)
                != Some(payload.as_ref())
            || exact.header().stable_transaction_id != prepared.as_wal_record().txn_id
        {
            return Err(EngineError::Durability(
                "typed exact canonical record binding diverged before WAL reservation".to_string(),
            ));
        }
        // Reserve both the exact bytes and one bounded record share before proposal.  The credit
        // transfers through tentative, claimed-pending, and in-flight ownership, so a successful
        // claim cannot later discover that its forming/in-flight group lacks capacity.
        let pending_group_credit = self
            .prepared_group_arena
            .reserve_typed_exact_credit(expected_serialized_len)?;
        #[cfg(unix)]
        let fua_exact_reservation = self.preflight_typed_exact_fua_admission(
            expected_serialized_len,
            exact.header().identity,
        )?;
        self.records.try_reserve_exact(1).map_err(|_| {
            EngineError::Durability(
                "unable to reserve a typed exact WAL record slot before WAL".to_string(),
            )
        })?;
        self.exact_typed_wire_records
            .try_reserve_exact(1)
            .map_err(|_| {
                EngineError::Durability(
                    "unable to reserve typed exact WAL wire-slot capacity before WAL".to_string(),
                )
            })?;
        // The serial backend's directory, handle, and zero-filled extent are all prepared before
        // proposal.  Once this owner is claimed, group formation performs no creation,
        // preallocation, allocation, or exact-record encoding.
        let prior_durable_identity_binding = self.preflight_typed_exact_serial_admission(
            expected_serialized_len,
            exact.header().identity,
        )?;
        Ok(WalTypedExactAppendReservation {
            expected_owner_id: self.canonical_append_reservation_owner_id,
            expected_records_len: self.records.len(),
            expected_records_capacity: self.records.capacity(),
            expected_exact_wire_len: self.exact_typed_wire_records.len(),
            expected_exact_wire_capacity: self.exact_typed_wire_records.capacity(),
            expected_generation: self.canonical_append_reservation_generation,
            expected_serialized_len,
            prior_catalog_tail: self.canonical_catalog_tail,
            prior_durable_identity_binding,
            rollback_authority: Some(exact.clone()),
            prepared: Some(prepared),
            pending_group_credit: Some(pending_group_credit),
            #[cfg(unix)]
            fua_exact_reservation,
            state: TypedExactAppendState::Prepared,
        })
    }

    /// Move the prebuilt typed record into a tentative, non-flushable tail.
    pub fn append_typed_exact_tentative(
        &mut self,
        reservation: &mut WalTypedExactAppendReservation,
    ) -> Result<(), EngineError> {
        if self.canonical_append_reservation_owner_id != reservation.expected_owner_id
            || self.records.len() != reservation.expected_records_len
            || self.records.capacity() != reservation.expected_records_capacity
            || self.exact_typed_wire_records.len() != reservation.expected_exact_wire_len
            || self.exact_typed_wire_records.capacity() != reservation.expected_exact_wire_capacity
            || self.typed_exact_tentative_count != 0
            || self.canonical_append_reservation_generation != reservation.expected_generation
            || reservation.state != TypedExactAppendState::Prepared
        {
            return Err(EngineError::Durability(
                "typed exact WAL append state drifted before tentative consumption".to_string(),
            ));
        }
        let prepared = reservation.prepared.take().ok_or_else(|| {
            EngineError::Durability("typed exact WAL record was already consumed".to_string())
        })?;
        let pending_group_credit = reservation.pending_group_credit.take().ok_or_else(|| {
            EngineError::Durability(
                "typed exact WAL append lost its pre-proposal bounded group credit".to_string(),
            )
        })?;
        let (record, tail, exact) = prepared.into_parts();
        let exact = exact.ok_or_else(|| {
            EngineError::Durability(
                "typed exact WAL record lost its outer wire authority".to_string(),
            )
        })?;
        if exact.serialized_record().len() != reservation.expected_serialized_len
            || exact
                .serialized_record()
                .get(crate::WAL_RECORD_HEADER_LEN..)
                != Some(record.payload.as_ref())
            || exact.header().stable_transaction_id != record.txn_id
            || reservation
                .rollback_authority
                .as_ref()
                .is_none_or(|authority| {
                    !Arc::ptr_eq(authority.serialized_record(), exact.serialized_record())
                })
        {
            return Err(EngineError::Durability(
                "typed exact WAL record binding drifted before tentative append".to_string(),
            ));
        }
        if self.durable_identity_binding != reservation.prior_durable_identity_binding {
            return Err(EngineError::Durability(
                "typed exact WAL lineage verification frontier drifted before tentative append"
                    .to_string(),
            ));
        }
        let records_capacity = self.records.capacity();
        let wire_capacity = self.exact_typed_wire_records.capacity();
        self.records.push(record);
        self.exact_typed_wire_records
            .push(ExactTypedWireRecord::tentative(
                self.records.len() - 1,
                Arc::clone(exact.serialized_record()),
                pending_group_credit,
            ));
        assert_eq!(
            self.records.capacity(),
            records_capacity,
            "reserved typed exact WAL append unexpectedly grew records"
        );
        assert_eq!(
            self.exact_typed_wire_records.capacity(),
            wire_capacity,
            "reserved typed exact WAL append unexpectedly grew wire authority"
        );
        self.canonical_catalog_tail = CanonicalCatalogTailCache::Known {
            record_count: self.records.len(),
            tail: Some(tail),
        };
        self.durable_identity_binding.identity = Some(exact.header().identity);
        self.durable_identity_binding.verified_records = self.records.len();
        self.invalidate_canonical_append_reservations();
        self.typed_exact_tentative_count = 1;
        reservation.state = TypedExactAppendState::Tentative {
            position: self.records.len() - 1,
            expected_generation: self.canonical_append_reservation_generation,
        };
        Ok(())
    }

    /// Restore record, serialized authority, catalog tail, and both logical frontiers together.
    /// All fallible validation occurs before either vector mutates.
    pub fn rollback_typed_exact_append(
        &mut self,
        reservation: &mut WalTypedExactAppendReservation,
    ) -> Result<(), EngineError> {
        let TypedExactAppendState::Tentative {
            position,
            expected_generation,
        } = reservation.state
        else {
            return Ok(());
        };
        if self.canonical_append_reservation_owner_id != reservation.expected_owner_id
            || position.checked_add(1) != Some(self.records.len())
            || self.exact_typed_wire_records.len()
                != reservation.expected_exact_wire_len.saturating_add(1)
            || self.canonical_append_reservation_generation != expected_generation
            || self.typed_exact_tentative_count != 1
        {
            return Err(EngineError::Durability(
                "typed exact WAL rollback state drifted after proposal".to_string(),
            ));
        }
        let wire = self.exact_typed_wire_records.last().ok_or_else(|| {
            EngineError::Durability(
                "typed exact WAL rollback lost its serialized-byte authority".to_string(),
            )
        })?;
        let tail = match self.canonical_catalog_tail {
            CanonicalCatalogTailCache::Known {
                record_count,
                tail: Some(tail),
            } if record_count == position + 1 => tail,
            _ => {
                return Err(EngineError::Durability(
                    "typed exact WAL catalog-tail frontier drifted before rollback".to_string(),
                ));
            }
        };
        let record = self
            .records
            .last()
            .expect("tentative typed record frontier checked above");
        let rollback_authority = reservation.rollback_authority.as_ref().ok_or_else(|| {
            EngineError::Durability(
                "typed exact WAL rollback lost its immutable authority".to_string(),
            )
        })?;
        if wire.record_index != position
            || !wire.is_tentative()
            || wire.serialized_record.len() != reservation.expected_serialized_len
            || wire.serialized_record.get(crate::WAL_RECORD_HEADER_LEN..)
                != Some(record.payload.as_ref())
            || !Arc::ptr_eq(
                &wire.serialized_record,
                rollback_authority.serialized_record(),
            )
        {
            return Err(EngineError::Durability(
                "claimed or drifted typed exact WAL position cannot be rolled back".to_string(),
            ));
        }
        let record = self.records.pop().expect("validated typed record frontier");
        let wire = self
            .exact_typed_wire_records
            .pop()
            .expect("validated wire frontier");
        let authority = reservation
            .rollback_authority
            .take()
            .expect("validated rollback authority");
        debug_assert!(Arc::ptr_eq(
            &wire.serialized_record,
            authority.serialized_record()
        ));
        reservation.pending_group_credit = Some(wire.into_pending_group_credit());
        #[cfg(unix)]
        reservation.fua_exact_reservation.take();
        reservation.prepared = Some(PreparedCanonicalWalRecord::from_parts(
            record,
            tail,
            Some(authority),
        ));
        self.canonical_catalog_tail = reservation.prior_catalog_tail;
        self.durable_identity_binding = reservation.prior_durable_identity_binding.clone();
        self.invalidate_canonical_append_reservations();
        self.typed_exact_tentative_count = 0;
        reservation.state = TypedExactAppendState::Prepared;
        reservation.expected_generation = self.canonical_append_reservation_generation;
        reservation.expected_records_len = self.records.len();
        reservation.expected_records_capacity = self.records.capacity();
        reservation.expected_exact_wire_len = self.exact_typed_wire_records.len();
        reservation.expected_exact_wire_capacity = self.exact_typed_wire_records.capacity();
        Ok(())
    }

    /// Finalize the only rollbackable typed WAL mutation and make it durable-eligible.
    pub fn claim_typed_exact_append(
        &mut self,
        mut reservation: WalTypedExactAppendReservation,
    ) -> Result<usize, EngineError> {
        let TypedExactAppendState::Tentative {
            position,
            expected_generation,
        } = reservation.state
        else {
            return Err(EngineError::Durability(
                "typed exact WAL claim requires a tentative append".to_string(),
            ));
        };
        if self.canonical_append_reservation_owner_id != reservation.expected_owner_id
            || position.checked_add(1) != Some(self.records.len())
            || self.exact_typed_wire_records.len()
                != reservation.expected_exact_wire_len.saturating_add(1)
            || self.canonical_append_reservation_generation != expected_generation
            || self.typed_exact_tentative_count != 1
        {
            return Err(EngineError::Durability(
                "typed exact WAL claim state drifted after proposal".to_string(),
            ));
        }
        let wire = self.exact_typed_wire_records.last_mut().ok_or_else(|| {
            EngineError::Durability(
                "typed exact WAL claim lost its serialized-byte authority".to_string(),
            )
        })?;
        if wire.record_index != position || !wire.is_tentative() {
            return Err(EngineError::Durability(
                "typed exact WAL position was already claimed".to_string(),
            ));
        }
        #[cfg(unix)]
        if let Some(fua_reservation) = reservation.fua_exact_reservation.take() {
            fua_reservation.claim()?;
        }
        wire.state = ExactTypedWireState::ClaimedPending;
        self.typed_exact_tentative_count = 0;
        self.highest_claimed_typed_exact_record = Some(
            self.highest_claimed_typed_exact_record
                .map_or(position, |prior| prior.max(position)),
        );
        Ok(position + 1)
    }

    pub(super) fn require_no_unclaimed_typed_exact_tail(&self) -> Result<(), EngineError> {
        if self.typed_exact_tentative_count != 0 {
            return Err(EngineError::Durability(
                "cannot flush an unclaimed tentative typed exact WAL tail".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn exact_wire_for_group(
        &self,
        index: usize,
        record: &crate::WalRecord,
    ) -> Result<Option<&ExactTypedWireRecord>, EngineError> {
        match self
            .exact_typed_wire_records
            .binary_search_by_key(&index, |wire| wire.record_index)
        {
            Ok(wire_index) if self.exact_typed_wire_records[wire_index].is_claimed_pending() => {
                let wire = &self.exact_typed_wire_records[wire_index];
                let bytes = &wire.serialized_record;
                if bytes.get(crate::WAL_RECORD_HEADER_LEN..) != Some(record.payload.as_ref()) {
                    return Err(EngineError::Durability(
                        "typed exact WAL wire authority diverged from its packed record"
                            .to_string(),
                    ));
                }
                Ok(Some(wire))
            }
            Ok(_) => Err(EngineError::Durability(
                "attempted to form a WAL group around a non-claimed typed exact record".to_string(),
            )),
            Err(_) => Ok(None),
        }
    }

    pub(super) fn exact_wire_prefix_count(&self, end: usize) -> Result<usize, EngineError> {
        let count = self
            .exact_typed_wire_records
            .iter()
            .take_while(|wire| wire.record_index < end)
            .count();
        for wire in &self.exact_typed_wire_records[..count] {
            if !wire.is_claimed_pending() {
                return Err(EngineError::Durability(
                    "attempted to hand off a non-claimed typed exact WAL owner".to_string(),
                ));
            }
        }
        Ok(count)
    }

    /// Drop serialized Arcs once the durable watermark covers their record positions. This is
    /// called before each new logical mutation/flush and after inline durability completes, so a
    /// 48M recovered history carries no sidecar and a live group retains only its pending tail.
    pub(super) fn prune_durable_typed_exact_wire_records(&mut self) {
        let durable = self.flushed_count();
        let retire = self
            .exact_typed_wire_records
            .iter()
            .take_while(|wire| wire.record_index < durable)
            .count();
        if retire != 0 {
            self.exact_typed_wire_records.drain(..retire);
        }
    }

    pub(super) fn has_typed_exact_wire_at_or_after(&self, len: usize) -> bool {
        self.exact_typed_wire_records
            .iter()
            .any(|wire| wire.record_index >= len)
    }

    /// FUA exact admission is a pre-proposal ownership proof.  Legacy bytes are first drained to
    /// the durable cut, then every forming member is verified as an already-claimed exact owner
    /// before the backend seals the aggregate physical cursor.  A full forming/in-flight group is
    /// internal backpressure: drain it and retry the valid next record, never surface the old
    /// temporary FUA-unsupported response.
    #[cfg(unix)]
    fn preflight_typed_exact_fua_admission(
        &mut self,
        expected_serialized_len: usize,
        exact_identity: crate::CanonicalIdentity,
    ) -> Result<Option<crate::fua::FuaExactReservation>, EngineError> {
        let Some(fua) = self.fua.as_ref().cloned() else {
            return Ok(None);
        };
        loop {
            if let Some(fault) = fua.exact_fault() {
                return Err(EngineError::DurabilityFault(fault));
            }
            if !fua.exact_forming_or_in_flight() {
                self.flush_all()?;
            }
            let first_record = fua.published_records();
            if first_record > self.records.len() {
                return Err(EngineError::Durability(
                    "FUA typed exact admission cursor exceeds the logical WAL frontier".to_string(),
                ));
            }
            if !fua.exact_forming_or_in_flight()
                && (first_record != self.records.len()
                    || fua.durable_records() != self.records.len())
            {
                return Err(EngineError::Durability(
                    "FUA legacy tail did not drain before typed exact admission".to_string(),
                ));
            }
            self.durable_identity_binding.verify_through(
                fua.base_path(),
                &self.records,
                self.records.len(),
            )?;
            match self.durable_identity_binding.identity {
                Some(identity) if identity != exact_identity => {
                    return Err(EngineError::Durability(
                        "typed exact FUA identity diverges from the durable lineage".to_string(),
                    ));
                }
                Some(identity) => {
                    crate::identity::require_durable_identity(fua.base_path(), identity)?
                }
                None => crate::identity::check_or_install_durable_identity(
                    fua.base_path(),
                    exact_identity,
                )?,
            }

            let mut forming_bytes = 0usize;
            let mut legacy_suffix = false;
            for index in first_record..self.records.len() {
                let record = self
                    .records
                    .get(index)
                    .expect("checked logical WAL frontier");
                let Some(exact) = self.exact_wire_for_group(index, record)? else {
                    // A compatibility append can arrive after an exact prefix while that prefix
                    // is still forming or in flight. It belongs to the next logical group: seal
                    // the bounded prefix, drain the compatibility suffix, and retry this new
                    // exact admission from a clean durable cut. It is not a post-WAL fallback.
                    legacy_suffix = true;
                    break;
                };
                forming_bytes = forming_bytes
                    .checked_add(exact.serialized_len())
                    .ok_or_else(|| {
                        EngineError::Durability(
                            "FUA typed exact forming bytes overflow".to_string(),
                        )
                    })?;
            }
            if legacy_suffix {
                // If the exact prefix is already handed off, wait until its controller owner is
                // settled before asking the shared flush path to form the legacy suffix. If it is
                // still forming, `flush_all` owns its exact-prefix handoff and then drains the
                // suffix in order.
                fua.wait_exact_settlement()?;
                self.flush_all()?;
                fua.wait_exact_settlement()?;
                continue;
            }
            let wire_bytes = forming_bytes
                .checked_add(expected_serialized_len)
                .ok_or_else(|| {
                    EngineError::Durability("FUA typed exact wire bytes overflow".to_string())
                })?;
            let target_records = self.records.len().checked_add(1).ok_or_else(|| {
                EngineError::Durability("FUA typed exact record cursor overflow".to_string())
            })?;
            match fua.reserve_exact_forming(first_record, target_records, wire_bytes)? {
                crate::fua::FuaExactReservationAttempt::Reserved(reservation) => {
                    return Ok(Some(reservation));
                }
                crate::fua::FuaExactReservationAttempt::SealPrior => {
                    self.flush_all()?;
                    fua.wait_exact_settlement()?;
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn typed_exact_wire_payload_ptr_for_test(
        &self,
        position: usize,
    ) -> Option<*const u8> {
        self.exact_typed_wire_records
            .binary_search_by_key(&position, |wire| wire.record_index)
            .ok()
            .and_then(|_| {
                self.records
                    .get(position)
                    .map(|record| record.payload.as_ptr())
            })
    }

    #[cfg(test)]
    pub(super) fn typed_exact_serialized_bytes_for_test(&self, position: usize) -> Option<Vec<u8>> {
        self.exact_typed_wire_records
            .binary_search_by_key(&position, |wire| wire.record_index)
            .ok()
            .map(|index| {
                self.exact_typed_wire_records[index]
                    .serialized_record
                    .to_vec()
            })
    }
}
