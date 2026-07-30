//! Checked accounting for host allocations retained by an inert pre-WAL owner.
//!
//! This deliberately counts backing storage only. Inline fields and allocator/`Arc` control
//! headers have no portable byte geometry; the former are not retained allocations and the
//! latter are represented by allocation slots. Generation/snapshot pins are a separate domain.

#![allow(dead_code)] // Inert owner reports are consumed by the next reservation-carrier slice.

use crate::EngineError;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// A mergeable report of live host backing stores and generation pins.
///
/// The owner map is intentionally retained until the final reservation input is assembled:
/// independent owners frequently share `Arc<str>` catalog stores, and scalar reports would
/// otherwise double charge them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct HostRetentionReport {
    backing_bytes: u64,
    backing_allocations: BTreeMap<usize, u64>,
    generation_pin_identities: BTreeSet<usize>,
}

/// Scalar geometry for a predicted owner before its private backing allocation exists.
///
/// Predictions never cross an owner boundary or participate in deduplication; the materialized
/// owner must report its real identities before a reservation can merge it with another owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct HostRetentionGeometry {
    retained_bytes: u64,
    allocation_slots: u64,
    generation_pin_slots: u64,
}

impl HostRetentionReport {
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.backing_bytes
    }

    /// One slot per live backing allocation. This charges neither a Rust inline field nor an
    /// allocator/`Arc` control header; it records the allocation lifetime that owns the bytes.
    pub(crate) fn allocation_slots(&self) -> Result<u64, EngineError> {
        u64::try_from(self.backing_allocations.len()).map_err(|_| overflow("host allocation slots"))
    }

    /// Snapshot/catalog generation pins are distinct from ordinary host backing allocations.
    pub(crate) fn generation_pin_slots(&self) -> Result<u64, EngineError> {
        u64::try_from(self.generation_pin_identities.len())
            .map_err(|_| overflow("host generation-pin slots"))
    }

    pub(crate) fn geometry(&self) -> Result<HostRetentionGeometry, EngineError> {
        Ok(HostRetentionGeometry {
            retained_bytes: self.retained_bytes(),
            allocation_slots: self.allocation_slots()?,
            generation_pin_slots: self.generation_pin_slots()?,
        })
    }

    pub(crate) fn merge(&mut self, other: Self) -> Result<(), EngineError> {
        for (identity, bytes) in other.backing_allocations {
            self.retain_backing(identity, bytes, "merged host backing allocation")?;
        }
        for identity in other.generation_pin_identities {
            self.retain_generation_pin_identity(identity)?;
        }
        Ok(())
    }

    pub(crate) fn retain_boxed_slice<T>(&mut self, values: &[T]) -> Result<(), EngineError> {
        self.retain_slice(values, "boxed host slice", false)
    }

    pub(crate) fn retain_vec<T>(&mut self, values: &Vec<T>) -> Result<(), EngineError> {
        self.retain_raw(
            values.as_ptr() as usize,
            values.capacity(),
            std::mem::size_of::<T>(),
            "host vector",
            false,
        )
    }

    pub(crate) fn retain_string(&mut self, value: &String) -> Result<(), EngineError> {
        self.retain_raw(
            value.as_ptr() as usize,
            value.capacity(),
            std::mem::size_of::<u8>(),
            "host string",
            false,
        )
    }

    /// A plan-private exact-length string owner. Unlike `String`, this has no spare capacity, so
    /// a pre-materialization geometry can be reproduced from its semantic length.
    pub(crate) fn retain_boxed_str(&mut self, value: &str) -> Result<(), EngineError> {
        self.retain_raw(
            value.as_ptr() as usize,
            value.len(),
            std::mem::size_of::<u8>(),
            "boxed host string",
            false,
        )
    }

    pub(crate) fn retain_arc_slice<T>(&mut self, values: &Arc<[T]>) -> Result<(), EngineError> {
        self.retain_raw(
            Arc::as_ptr(values) as *const T as usize,
            values.len(),
            std::mem::size_of::<T>(),
            "Arc host slice",
            true,
        )
    }

    pub(crate) fn retain_arc_str(&mut self, value: &Arc<str>) -> Result<(), EngineError> {
        self.retain_raw(
            Arc::as_ptr(value) as *const u8 as usize,
            value.len(),
            std::mem::size_of::<u8>(),
            "Arc host string",
            true,
        )
    }

    /// An ordinary statement-retired `Arc<T>` allocation. Unlike a generation pin or GPU guard,
    /// its payload and control allocation are part of this plan's host retention.
    pub(crate) fn retain_arc_owner<T>(&mut self, value: &Arc<T>) -> Result<(), EngineError> {
        self.retain_raw(
            Arc::as_ptr(value) as usize,
            1,
            std::mem::size_of::<T>(),
            "ordinary Arc host owner",
            true,
        )
    }

    /// Charge a statement-retired allocation whose bytes live in another logical domain. Typed
    /// canonical shadow bytes, for example, stay out of `host_plan_retained_bytes` but still
    /// consume one generic host allocation slot.
    pub(crate) fn retain_slot_only(&mut self, identity: usize) -> Result<(), EngineError> {
        self.retain_backing(identity, 0, "host allocation slot-only owner")
    }

    /// Merge a backing whose owner is private to a lower crate. Callers may use this only for an
    /// allocation that is structurally unshared outside that lower owner; shared stores must use
    /// one of the identity-aware retain helpers above.
    pub(crate) fn retain_external_backing(
        &mut self,
        identity: usize,
        bytes: u64,
    ) -> Result<(), EngineError> {
        self.retain_backing(identity, bytes, "external host backing allocation")
    }

    /// Record a true catalog/snapshot generation pin only. Device/cache wrappers, contexts,
    /// mutex/lifecycle guards, completion handles, and semantic `Arc<str>` stores are not pins.
    pub(crate) fn retain_generation_pin<T: ?Sized>(
        &mut self,
        value: &Arc<T>,
    ) -> Result<(), EngineError> {
        self.retain_generation_pin_identity(Arc::as_ptr(value) as *const () as usize)
    }

    fn retain_slice<T>(
        &mut self,
        values: &[T],
        domain: &'static str,
        retain_empty_arc_slot: bool,
    ) -> Result<(), EngineError> {
        self.retain_raw(
            values.as_ptr() as usize,
            values.len(),
            std::mem::size_of::<T>(),
            domain,
            retain_empty_arc_slot,
        )
    }

    fn retain_raw(
        &mut self,
        identity: usize,
        elements: usize,
        element_size: usize,
        domain: &'static str,
        retain_zero_byte_slot: bool,
    ) -> Result<(), EngineError> {
        if (elements == 0 || element_size == 0) && !retain_zero_byte_slot {
            return Ok(());
        }
        let bytes = if elements == 0 || element_size == 0 {
            0
        } else {
            u64::try_from(elements)
                .ok()
                .and_then(|elements| {
                    u64::try_from(element_size)
                        .ok()
                        .and_then(|size| elements.checked_mul(size))
                })
                .ok_or_else(|| overflow(domain))?
        };
        self.retain_backing(identity, bytes, domain)
    }

    fn retain_backing(
        &mut self,
        identity: usize,
        bytes: u64,
        domain: &'static str,
    ) -> Result<(), EngineError> {
        if self.generation_pin_identities.contains(&identity) {
            return Err(EngineError::Durability(format!(
                "{domain} identity is already charged as a host generation pin"
            )));
        }
        if let Some(existing) = self.backing_allocations.get(&identity) {
            if *existing != bytes {
                return Err(EngineError::Durability(format!(
                    "{domain} identity has inconsistent retained byte geometry"
                )));
            }
            return Ok(());
        }
        self.backing_bytes = self
            .backing_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow(domain))?;
        self.backing_allocations.insert(identity, bytes);
        Ok(())
    }

    fn retain_generation_pin_identity(&mut self, identity: usize) -> Result<(), EngineError> {
        if self.backing_allocations.contains_key(&identity) {
            return Err(EngineError::Durability(
                "host generation-pin identity is already charged as a host backing allocation"
                    .to_string(),
            ));
        }
        self.generation_pin_identities.insert(identity);
        Ok(())
    }
}

impl HostRetentionGeometry {
    pub(crate) fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }

    pub(crate) fn allocation_slots(self) -> u64 {
        self.allocation_slots
    }

    pub(crate) fn generation_pin_slots(self) -> u64 {
        self.generation_pin_slots
    }

    pub(crate) fn checked_add_backing_elements<T>(
        &mut self,
        elements: usize,
        domain: &'static str,
    ) -> Result<(), EngineError> {
        let element_size = std::mem::size_of::<T>();
        if elements == 0 || element_size == 0 {
            return Ok(());
        }
        let bytes = u64::try_from(elements)
            .ok()
            .and_then(|elements| {
                u64::try_from(element_size)
                    .ok()
                    .and_then(|size| elements.checked_mul(size))
            })
            .ok_or_else(|| overflow(domain))?;
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow(domain))?;
        self.allocation_slots = self
            .allocation_slots
            .checked_add(1)
            .ok_or_else(|| overflow("host allocation slots"))?;
        Ok(())
    }

    /// Add deterministic backing geometry that is known before the backing has an address.  This
    /// is deliberately scalar-only: admission code must not allocate an identity map merely to
    /// size a future owner.  Identity-aware maps remain post-materialization diagnostics.
    pub(crate) fn checked_add_backing_bytes_slots(
        &mut self,
        bytes: u64,
        slots: u64,
        domain: &'static str,
    ) -> Result<(), EngineError> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow(domain))?;
        self.allocation_slots = self
            .allocation_slots
            .checked_add(slots)
            .ok_or_else(|| overflow("host allocation slots"))?;
        Ok(())
    }

    pub(crate) fn checked_add_allocation_slot(&mut self) -> Result<(), EngineError> {
        self.checked_add_backing_bytes_slots(0, 1, "host allocation slot-only owner")
    }

    pub(crate) fn checked_add_generation_pin_slots(
        &mut self,
        slots: u64,
    ) -> Result<(), EngineError> {
        self.generation_pin_slots = self
            .generation_pin_slots
            .checked_add(slots)
            .ok_or_else(|| overflow("host generation-pin slots"))?;
        Ok(())
    }

    /// Compose independently-owned scalar geometry on an allocation-free admission path. The
    /// caller must first establish that the two owner domains cannot share backing identities;
    /// identity-aware report merging remains the post-materialization diagnostic for shared
    /// owners.
    pub(crate) fn checked_add_disjoint(
        &mut self,
        other: Self,
        domain: &'static str,
    ) -> Result<(), EngineError> {
        self.checked_add_backing_bytes_slots(other.retained_bytes, other.allocation_slots, domain)?;
        self.generation_pin_slots = self
            .generation_pin_slots
            .checked_add(other.generation_pin_slots)
            .ok_or_else(|| overflow("host generation-pin slots"))?;
        Ok(())
    }

    pub(crate) fn matches(self, actual: &HostRetentionReport) -> Result<bool, EngineError> {
        Ok(self == actual.geometry()?)
    }

    /// Independent capacity domains reserve their maximum simultaneous demand, never the sum of
    /// owners that were moved between phases.
    pub(crate) fn peak(self, other: Self) -> Self {
        Self {
            retained_bytes: self.retained_bytes.max(other.retained_bytes),
            allocation_slots: self.allocation_slots.max(other.allocation_slots),
            generation_pin_slots: self.generation_pin_slots.max(other.generation_pin_slots),
        }
    }
}

fn overflow(domain: &'static str) -> EngineError {
    EngineError::Durability(format!("{domain} retained byte geometry overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_arc_slice_is_charged_once_and_has_one_backing_slot() {
        let shared: Arc<[u32]> = Arc::from(vec![3_u32, 5, 7]);
        let mut first = HostRetentionReport::default();
        first.retain_arc_slice(&shared).unwrap();
        let mut second = HostRetentionReport::default();
        second.retain_arc_slice(&shared).unwrap();
        first.merge(second).unwrap();
        assert_eq!(
            first.retained_bytes(),
            3 * std::mem::size_of::<u32>() as u64
        );
        assert_eq!(first.allocation_slots().unwrap(), 1);
    }

    #[test]
    fn vector_capacity_not_length_is_the_retained_backing_geometry() {
        let mut values = Vec::<u64>::with_capacity(11);
        values.extend([1, 2]);
        let mut report = HostRetentionReport::default();
        report.retain_vec(&values).unwrap();
        assert_eq!(
            report.retained_bytes(),
            11 * std::mem::size_of::<u64>() as u64
        );
        assert_eq!(report.allocation_slots().unwrap(), 1);
    }

    #[test]
    fn only_explicit_generation_pins_enter_the_pin_domain() {
        let generation = Arc::new(());
        let mut report = HostRetentionReport::default();
        report.retain_generation_pin(&generation).unwrap();
        report.retain_generation_pin(&generation).unwrap();
        assert_eq!(report.generation_pin_slots().unwrap(), 1);
        assert_eq!(report.retained_bytes(), 0);
        assert_eq!(report.allocation_slots().unwrap(), 0);
    }

    #[test]
    fn empty_direct_arc_is_a_zero_byte_but_real_host_allocation_slot() {
        let empty: Arc<str> = Arc::from("");
        let mut report = HostRetentionReport::default();
        report.retain_arc_str(&empty).unwrap();
        report.retain_arc_str(&empty).unwrap();
        assert_eq!(report.retained_bytes(), 0);
        assert_eq!(report.allocation_slots().unwrap(), 1);
    }

    #[test]
    fn distinct_empty_arc_owners_are_not_collapsed_with_a_shared_empty_arc() {
        let first: Arc<str> = Arc::from("");
        let shared = Arc::clone(&first);
        let distinct: Arc<str> = Arc::from("");
        assert_ne!(Arc::as_ptr(&first), Arc::as_ptr(&distinct));
        let mut report = HostRetentionReport::default();
        report.retain_arc_str(&first).unwrap();
        report.retain_arc_str(&shared).unwrap();
        report.retain_arc_str(&distinct).unwrap();
        assert_eq!(report.retained_bytes(), 0);
        assert_eq!(report.allocation_slots().unwrap(), 2);
    }

    #[test]
    fn ordinary_atomic_arc_charges_payload_and_one_generic_slot() {
        let atomic = Arc::new(std::sync::atomic::AtomicU64::new(7));
        let mut report = HostRetentionReport::default();
        report.retain_arc_owner(&atomic).unwrap();
        assert_eq!(
            report.retained_bytes(),
            std::mem::size_of::<std::sync::atomic::AtomicU64>() as u64
        );
        assert_eq!(report.allocation_slots().unwrap(), 1);
        assert_eq!(report.generation_pin_slots().unwrap(), 0);
    }

    #[test]
    fn typed_shadow_slot_only_owner_never_leaks_its_bytes_into_host_plan() {
        let mut report = HostRetentionReport::default();
        report.retain_slot_only(73).unwrap();
        assert_eq!(report.retained_bytes(), 0);
        assert_eq!(report.allocation_slots().unwrap(), 1);
    }

    #[test]
    fn exact_boxed_strings_and_zst_slices_follow_the_owner_taxonomy() {
        let empty = Box::<str>::from("");
        let text = Box::<str>::from("abc");
        let zst: Box<[()]> = vec![(), ()].into();
        let mut report = HostRetentionReport::default();
        report.retain_boxed_str(&empty).unwrap();
        report.retain_boxed_str(&text).unwrap();
        report.retain_boxed_slice(&zst).unwrap();
        assert_eq!(report.retained_bytes(), 3);
        assert_eq!(report.allocation_slots().unwrap(), 1);
    }

    #[test]
    fn scalar_prediction_rejects_one_byte_and_one_slot_sabotage() {
        let mut expected = HostRetentionGeometry::default();
        expected
            .checked_add_backing_bytes_slots(7, 1, "test predicted owner")
            .unwrap();
        let mut actual = HostRetentionReport::default();
        actual.retain_external_backing(1, 7).unwrap();
        assert!(expected.matches(&actual).unwrap());

        let mut one_byte_short = HostRetentionReport::default();
        one_byte_short.retain_external_backing(2, 6).unwrap();
        assert!(!expected.matches(&one_byte_short).unwrap());

        let mut two_slots_expected = expected;
        two_slots_expected.checked_add_allocation_slot().unwrap();
        let mut one_slot_short = HostRetentionReport::default();
        one_slot_short.retain_external_backing(3, 7).unwrap();
        assert!(!two_slots_expected.matches(&one_slot_short).unwrap());
    }

    #[test]
    fn cross_domain_identity_collision_fails_closed() {
        let mut report = HostRetentionReport::default();
        report.retain_external_backing(41, 7).unwrap();
        let error = report.retain_generation_pin_identity(41).unwrap_err();
        assert!(error.to_string().contains("backing allocation"));

        report.retain_generation_pin_identity(43).unwrap();
        let error = report.retain_external_backing(43, 7).unwrap_err();
        assert!(error.to_string().contains("generation pin"));
    }
}
