//! Sealed, flat neutral generation input and its exact pre-launch reservations.
//!
//! This owner is intentionally separate from `super::generation_input_digest`: the latter is
//! the post-quiescence retained-graph validator, while this file builds the only data a reserved
//! builder may observe.  Both spell the frozen digest framing, but neither calls the other's
//! traversal.

use super::super::generation_error;
#[cfg(test)]
use super::super::ROOT_DESCRIPTOR_VERSION;
use super::{directory_ref, range, storage_matches_type};
use crate::typed_insert_aggregate::semantics_v2::retained::{
    graph::{
        ReservedSemanticsV2Graph, RetainedIndexDescriptor, RetainedIndexKeyColumn,
        RetainedKeyEffect, RetainedTable, RetainedTransition,
    },
    SemanticsV2BoundIdentity, SemanticsV2CatalogColumnWitness, SemanticsV2CatalogIndexWitness,
    SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness,
};
use crate::typed_insert_batch::{
    DecodedTypedImage, TypedImageRole, TypedInsertColumnValidity, TypedInsertColumnValues,
};
use crate::{EngineError, SqlType};
use sha2::{Digest, Sha256};

type CanonicalDigest = gpu_db_wal::CanonicalDigest;

const PHYSICAL_MAINTENANCE_ROLE: u8 = 1;

/// The exact builder-private allocation request.  Every owned neutral family is named so a
/// device candidate can reserve its backing without receiving the graph, catalog, allocator
/// proof, expected digest, or any final generation/root.
#[derive(Clone, Copy)]
pub(in super::super::super) struct GenerationBuilderReservation {
    tables: usize,
    rows: usize,
    cells: usize,
    value_bytes: usize,
    indexes: usize,
    keys: usize,
    effects: usize,
    effect_values: usize,
    effect_value_bytes: usize,
    table_outputs: usize,
    index_outputs: usize,
}

impl GenerationBuilderReservation {
    pub(in super::super) fn tables(&self) -> usize {
        self.tables
    }

    pub(in super::super) fn rows(&self) -> usize {
        self.rows
    }

    pub(in super::super) fn cells(&self) -> usize {
        self.cells
    }

    pub(in super::super) fn value_bytes(&self) -> usize {
        self.value_bytes
    }

    pub(in super::super) fn indexes(&self) -> usize {
        self.indexes
    }

    pub(in super::super) fn keys(&self) -> usize {
        self.keys
    }

    pub(in super::super) fn effects(&self) -> usize {
        self.effects
    }

    pub(in super::super) fn effect_values(&self) -> usize {
        self.effect_values
    }

    pub(in super::super) fn effect_value_bytes(&self) -> usize {
        self.effect_value_bytes
    }

    pub(in super::super) fn table_outputs(&self) -> usize {
        self.table_outputs
    }

    pub(in super::super) fn index_outputs(&self) -> usize {
        self.index_outputs
    }
}

/// Owned, flat and reference-neutral builder input.  The only offsets are private offsets into
/// this owned allocation; it has no pointer back to retained S7, catalog, or allocator state.
pub(in super::super) struct SealedGenerationInput {
    identity: NeutralIdentity,
    tables: Vec<NeutralTable>,
    rows: Vec<NeutralRow>,
    cells: Vec<NeutralCell>,
    values: Vec<u8>,
    indexes: Vec<NeutralIndex>,
    keys: Vec<NeutralIndexKey>,
    effects: Vec<NeutralEffect>,
    effect_values: Vec<NeutralTypedValue>,
    effect_value_bytes: Vec<u8>,
    #[cfg(test)]
    retention_sentinel: Option<RetentionSentinel>,
}

#[derive(Clone, Copy)]
struct NeutralIdentity {
    database_id: [u8; 16],
    catalog_epoch: u64,
    catalog_digest: CanonicalDigest,
    stable_transaction_id: u64,
    commit_sequence: u64,
    initial_database_root: CanonicalDigest,
}

#[derive(Clone, Copy)]
struct NeutralTable {
    stable_table_id: u64,
    base_data_generation: u64,
    base_table_root: CanonicalDigest,
    row_allocator_before: u64,
    row_allocator_high_water: u64,
    initial_logical_row_count: u64,
    final_logical_row_count: u64,
    image_layout_digest: CanonicalDigest,
    image_content_digest: CanonicalDigest,
    row_start: u32,
    row_count: u32,
    index_start: u32,
    index_count: u32,
}

#[derive(Clone, Copy)]
struct NeutralRow {
    stable_table_id: u64,
    stable_row_id: u64,
    source_statement_ordinal: u32,
    source_row_ordinal: u32,
    cell_start: u32,
    cell_count: u32,
}

#[derive(Clone, Copy)]
struct NeutralCell {
    catalog_column_ordinal: u32,
    stable_column_id: u32,
    attnum: i16,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
    is_null: bool,
    value_start: u32,
    value_count: u32,
}

#[derive(Clone, Copy)]
struct NeutralIndex {
    owner_stable_table_id: u64,
    stable_index_id: u64,
    flags: u32,
    null_equality_policy: u8,
    base_generation: u64,
    base_root: CanonicalDigest,
    key_start: u32,
    key_count: u32,
    effect_start: u32,
    effect_count: u32,
}

#[derive(Clone, Copy)]
struct NeutralIndexKey {
    key_ordinal: u32,
    owner_catalog_column_ordinal: u32,
    stable_column_id: u32,
    attnum: i16,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
    column_name_digest: CanonicalDigest,
}

#[derive(Clone, Copy)]
struct NeutralEffect {
    stable_table_id: u64,
    stable_index_id: u64,
    stable_row_id: u64,
    source_catalog_ordinal: u32,
    key_arity: u32,
    value_start: u32,
    value_count: u32,
}

#[derive(Clone, Copy)]
struct NeutralTypedValue {
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
    is_null: bool,
    value_start: u32,
    value_count: u32,
}

#[path = "owned/views.rs"]
mod views;

/// Fixed output owners are allocated before launch and are only visible through the builder's
/// private writer.  A slot tracks missing, duplicate, and extra writes without allocating.
pub(in super::super) struct ReservedGenerationOutputs {
    pub(in super::super) header: GenerationOutputHeader,
    pub(in super::super) tables: Vec<GenerationOutputTable>,
    pub(in super::super) indexes: Vec<GenerationOutputIndex>,
    pub(in super::super) extra_write: bool,
    #[cfg(test)]
    retention_sentinel: Option<RetentionSentinel>,
}

#[cfg(test)]
struct RetentionSentinel(std::sync::Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl Drop for RetentionSentinel {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

pub(in super::super) struct GenerationOutputHeader {
    pub(in super::super) filled: bool,
    pub(in super::super) duplicate: bool,
    pub(in super::super) root_descriptor_version: u16,
    pub(in super::super) database_id: [u8; 16],
    pub(in super::super) catalog_epoch: u64,
    pub(in super::super) catalog_digest: CanonicalDigest,
    pub(in super::super) stable_transaction_id: u64,
    pub(in super::super) commit_sequence: u64,
    pub(in super::super) initial_database_root: CanonicalDigest,
    pub(in super::super) generation_input_digest: CanonicalDigest,
    pub(in super::super) final_database_root: CanonicalDigest,
}

pub(in super::super) struct GenerationOutputTable {
    pub(in super::super) filled: bool,
    pub(in super::super) duplicate: bool,
    pub(in super::super) stable_table_id: u64,
    pub(in super::super) final_data_generation: u64,
    pub(in super::super) final_table_root: CanonicalDigest,
    pub(in super::super) final_logical_row_count: u64,
    pub(in super::super) index_start: u32,
    pub(in super::super) index_count: u32,
}

pub(in super::super) struct GenerationOutputIndex {
    pub(in super::super) filled: bool,
    pub(in super::super) duplicate: bool,
    pub(in super::super) stable_index_id: u64,
    pub(in super::super) final_index_generation: u64,
    pub(in super::super) final_index_root: CanonicalDigest,
}

impl GenerationOutputHeader {
    fn sentinel() -> Self {
        Self {
            filled: false,
            duplicate: false,
            root_descriptor_version: 0,
            database_id: [0; 16],
            catalog_epoch: 0,
            catalog_digest: [0; 32],
            stable_transaction_id: 0,
            commit_sequence: 0,
            initial_database_root: [0; 32],
            generation_input_digest: [0; 32],
            final_database_root: [0; 32],
        }
    }
}

impl GenerationOutputTable {
    fn sentinel() -> Self {
        Self {
            filled: false,
            duplicate: false,
            stable_table_id: 0,
            final_data_generation: 0,
            final_table_root: [0; 32],
            final_logical_row_count: 0,
            index_start: 0,
            index_count: 0,
        }
    }
}

impl GenerationOutputIndex {
    fn sentinel() -> Self {
        Self {
            filled: false,
            duplicate: false,
            stable_index_id: 0,
            final_index_generation: 0,
            final_index_root: [0; 32],
        }
    }
}

/// The total builder launch argument.  Its fields remain private to this validation subtree;
/// the builder has only sealed input inspection plus a fixed, already-reserved output writer.
pub(in super::super::super) struct ReservedGenerationLaunch<C, W> {
    input: SealedGenerationInput,
    outputs: ReservedGenerationOutputs,
    candidate: C,
    quarantine: PreReservedQuarantine<C, W>,
}

struct QuarantinePayload<C, W> {
    work: W,
    input: SealedGenerationInput,
    outputs: ReservedGenerationOutputs,
    candidate: C,
}

/// Thread-affine service owner for uncertain launch backing.  The service creates the `Rc`
/// outside individual attempts; registration alone does the fallible one-slot reservation.
pub(in super::super::super) struct GenerationQuarantineRegistry<C, W> {
    inner: std::rc::Rc<std::cell::RefCell<GenerationQuarantineRegistryInner<C, W>>>,
}

impl<C, W> Clone for GenerationQuarantineRegistry<C, W> {
    fn clone(&self) -> Self {
        Self {
            inner: std::rc::Rc::clone(&self.inner),
        }
    }
}

struct GenerationQuarantineRegistryInner<C, W> {
    next_generation: u64,
    next_proof_epoch: u64,
    slots: Vec<Box<[QuarantineSlot<C, W>]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QuarantineId {
    index: usize,
    generation: u64,
}

// `Occupied` deliberately keeps the full generic payload in the one pre-reserved slot.  Boxing
// it at this transition would add exactly the post-launch allocation the quarantine prevents.
#[allow(clippy::large_enum_variant)]
enum QuarantineSlot<C, W> {
    Vacant,
    Armed {
        generation: u64,
    },
    Occupied {
        generation: u64,
        proof_epoch: Option<u64>,
        failure: QuarantineFailure,
        payload: QuarantinePayload<C, W>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum QuarantineFailure {
    QuiescenceUnproven,
    DrainPanicked,
}

/// This unconstructible proof is reserved for the future generation-service control plane that
/// independently observes the device stream idle.  It owns the exact registry it observed and
/// the exact epoch-marked occupied tickets it observed.  A builder and launched attempt never
/// receive it, so they cannot release unknown backing or claim another registry.
pub(in super::super) struct ExternalGenerationQuiescenceProof<C, W> {
    registry: GenerationQuarantineRegistry<C, W>,
    generation_cutoff: u64,
    proof_epoch: u64,
}

pub(in super::super::super) struct AuthorizedGenerationQuarantineReaper<C, W> {
    registry: GenerationQuarantineRegistry<C, W>,
    generation_cutoff: u64,
    proof_epoch: u64,
}

impl<C, W> GenerationQuarantineRegistry<C, W> {
    /// Service construction, not a per-attempt reservation.  The service holds this typed owner
    /// independently through uncertain device completion.
    pub(in super::super::super) fn new() -> Self {
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(GenerationQuarantineRegistryInner {
                next_generation: 1,
                next_proof_epoch: 1,
                slots: Vec::new(),
            })),
        }
    }

    fn register(&self) -> Result<PreReservedQuarantine<C, W>, EngineError> {
        let mut inner = self.inner.borrow_mut();
        let generation = inner.next_generation;
        inner.next_generation = inner
            .next_generation
            .checked_add(1)
            .ok_or_else(|| generation_error("generation quarantine ticket sequence overflows"))?;
        if let Some((index, slot)) =
            inner
                .slots
                .iter_mut()
                .enumerate()
                .find_map(|(index, entry)| match &mut entry[0] {
                    QuarantineSlot::Vacant => Some((index, &mut entry[0])),
                    QuarantineSlot::Armed { .. } | QuarantineSlot::Occupied { .. } => None,
                })
        {
            *slot = QuarantineSlot::Armed { generation };
            return Ok(PreReservedQuarantine {
                registry: Self::clone(self),
                id: QuarantineId { index, generation },
                armed: true,
            });
        }

        #[cfg(test)]
        note_reservation("generation quarantine ticket")?;
        let mut one_slot = Vec::new();
        one_slot
            .try_reserve_exact(1)
            .map_err(|_| generation_error("generation quarantine ticket reservation failed"))?;
        if one_slot.capacity() != 1 {
            return Err(generation_error(
                "generation quarantine ticket reservation is not exact",
            ));
        }
        one_slot.push(QuarantineSlot::Armed { generation });
        #[cfg(test)]
        note_reservation("generation quarantine registry entry")?;
        inner.slots.try_reserve_exact(1).map_err(|_| {
            generation_error("generation quarantine registry entry reservation failed")
        })?;
        inner.slots.push(one_slot.into_boxed_slice());
        Ok(PreReservedQuarantine {
            registry: Self::clone(self),
            id: QuarantineId {
                index: inner.slots.len() - 1,
                generation,
            },
            armed: true,
        })
    }

    /// The future generation-service control plane calls this only after independently observing
    /// device quiescence.  It atomically captures the current ticket cutoff and marks precisely
    /// the occupied slots it observed; Armed slots remain unmarked and cannot be released if
    /// they park later.  This private leaf intentionally exposes no production proof constructor.
    fn observe_external_quiescence(
        &self,
    ) -> Result<ExternalGenerationQuiescenceProof<C, W>, EngineError> {
        let (generation_cutoff, proof_epoch) = {
            let mut inner = self.inner.borrow_mut();
            // Reserve the next epoch before changing any marker, so overflow leaves the registry
            // untouched and a caller cannot mistake a partially marked proof for authority.
            let proof_epoch = inner.next_proof_epoch;
            let next_proof_epoch = proof_epoch
                .checked_add(1)
                .ok_or_else(|| generation_error("generation quarantine proof epoch overflows"))?;
            let generation_cutoff = inner
                .next_generation
                .checked_sub(1)
                .expect("generation quarantine ticket sequence starts at one");
            inner.next_proof_epoch = next_proof_epoch;
            for entry in &mut inner.slots {
                if let QuarantineSlot::Occupied {
                    proof_epoch: marker,
                    ..
                } = &mut entry[0]
                {
                    *marker = Some(proof_epoch);
                }
            }
            (generation_cutoff, proof_epoch)
        };
        Ok(ExternalGenerationQuiescenceProof {
            registry: Self::clone(self),
            generation_cutoff,
            proof_epoch,
        })
    }

    #[cfg(test)]
    pub(in super::super::super) fn authorized_reaper_for_test(
        &self,
    ) -> AuthorizedGenerationQuarantineReaper<C, W> {
        self.observe_external_quiescence()
            .expect("test quiescence observation reserves a fresh proof epoch")
            .authorize_reaper()
    }
}

impl<C, W> ExternalGenerationQuiescenceProof<C, W> {
    /// Consuming the proof makes reaping one-shot.  The reaper retains the proof's exact
    /// registry binding and cannot be redirected to another registry with the same C/W types.
    pub(in super::super) fn authorize_reaper(self) -> AuthorizedGenerationQuarantineReaper<C, W> {
        AuthorizedGenerationQuarantineReaper {
            registry: self.registry,
            generation_cutoff: self.generation_cutoff,
            proof_epoch: self.proof_epoch,
        }
    }
}

impl<C, W> AuthorizedGenerationQuarantineReaper<C, W> {
    /// Reap only the exact occupied tickets marked by the consumed proof observation.  This is
    /// the sole owner release for `Occupied` slots; Armed or later tickets require a fresh proof.
    #[allow(dead_code)]
    pub(in super::super::super) fn reap_after_external_quiescence(self) {
        let slot_count = self.registry.inner.borrow().slots.len();
        for index in 0..slot_count {
            // `payload` leaves its slot while the registry is borrowed, but drops only after
            // this block releases that borrow.  Generic work/input/output/candidate destructors
            // may therefore re-enter the service without observing a mutable RefCell borrow.
            let payload = {
                let mut inner = self.registry.inner.borrow_mut();
                let slot = inner
                    .slots
                    .get_mut(index)
                    .map(|entry| &mut entry[0])
                    .expect("registered quarantine slot remains addressable");
                match slot {
                    QuarantineSlot::Occupied {
                        generation,
                        proof_epoch: Some(proof_epoch),
                        ..
                    } if *generation <= self.generation_cutoff
                        && *proof_epoch == self.proof_epoch =>
                    {
                        match std::mem::replace(slot, QuarantineSlot::Vacant) {
                            QuarantineSlot::Occupied { payload, .. } => Some(payload),
                            QuarantineSlot::Vacant | QuarantineSlot::Armed { .. } => {
                                unreachable!("occupied quarantine slot changed during its borrow")
                            }
                        }
                    }
                    QuarantineSlot::Vacant
                    | QuarantineSlot::Armed { .. }
                    | QuarantineSlot::Occupied { .. } => None,
                }
            };
            drop(payload);
        }
    }

    #[cfg(test)]
    pub(in super::super::super) fn occupancy(&self) -> QuarantineOccupancy {
        let inner = self.registry.inner.borrow();
        let mut occupancy = QuarantineOccupancy::default();
        for entry in &inner.slots {
            match &entry[0] {
                QuarantineSlot::Vacant => occupancy.vacant += 1,
                QuarantineSlot::Armed { .. } => occupancy.armed += 1,
                QuarantineSlot::Occupied {
                    payload, failure, ..
                } => {
                    occupancy.occupied += 1;
                    let _ = payload;
                    occupancy.work_owner += 1;
                    occupancy.input_owner += 1;
                    occupancy.output_owner += 1;
                    occupancy.candidate_owner += 1;
                    occupancy.panicked += usize::from(*failure == QuarantineFailure::DrainPanicked);
                }
            }
        }
        occupancy
    }
}

#[cfg(test)]
#[derive(Default, Debug, PartialEq, Eq)]
pub(in super::super::super) struct QuarantineOccupancy {
    pub(in super::super::super) vacant: usize,
    pub(in super::super::super) armed: usize,
    pub(in super::super::super) occupied: usize,
    pub(in super::super::super) work_owner: usize,
    pub(in super::super::super) input_owner: usize,
    pub(in super::super::super) output_owner: usize,
    pub(in super::super::super) candidate_owner: usize,
    pub(in super::super::super) panicked: usize,
}

/// Non-clone registration.  It cancels only an `Armed` slot on ordinary prelaunch/proven-drop
/// paths; `park` consumes it into `Occupied`, which only the authorized reaper can release.
pub(in super::super) struct PreReservedQuarantine<C, W> {
    registry: GenerationQuarantineRegistry<C, W>,
    id: QuarantineId,
    armed: bool,
}

impl<C, W> PreReservedQuarantine<C, W> {
    pub(in super::super) fn park(
        mut self,
        work: W,
        input: SealedGenerationInput,
        outputs: ReservedGenerationOutputs,
        candidate: C,
        failure: QuarantineFailure,
    ) {
        let mut inner = self.registry.inner.borrow_mut();
        let slot = inner
            .slots
            .get_mut(self.id.index)
            .map(|entry| &mut entry[0])
            .filter(|slot| {
                matches!(slot, QuarantineSlot::Armed { generation } if *generation == self.id.generation)
            })
            .expect("armed generation ticket owns an exact registry slot");
        *slot = QuarantineSlot::Occupied {
            generation: self.id.generation,
            proof_epoch: None,
            failure,
            payload: QuarantinePayload {
                work,
                input,
                outputs,
                candidate,
            },
        };
        self.armed = false;
    }
}

impl<C, W> Drop for PreReservedQuarantine<C, W> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut inner = self.registry.inner.borrow_mut();
        let slot = inner
            .slots
            .get_mut(self.id.index)
            .map(|entry| &mut entry[0])
            .filter(|slot| {
                matches!(slot, QuarantineSlot::Armed { generation } if *generation == self.id.generation)
            })
            .expect("armed generation ticket must deregister its exact slot");
        *slot = QuarantineSlot::Vacant;
    }
}

impl<C, W> Drop for GenerationQuarantineRegistryInner<C, W> {
    fn drop(&mut self) {
        for entry in std::mem::take(&mut self.slots) {
            if matches!(&entry[0], QuarantineSlot::Occupied { .. }) {
                // Losing the final service/reaper handle is not quiescence proof.  Preserve the
                // complete typed payload rather than running C/W destructors against a live GPU.
                let _ = Box::leak(entry);
            }
        }
    }
}

impl<C, W> ReservedGenerationLaunch<C, W> {
    pub(in super::super) fn input(&self) -> &SealedGenerationInput {
        &self.input
    }

    pub(in super::super) fn outputs_mut(&mut self) -> &mut ReservedGenerationOutputs {
        &mut self.outputs
    }

    pub(in super::super) fn into_parts(
        self,
    ) -> (
        SealedGenerationInput,
        ReservedGenerationOutputs,
        C,
        PreReservedQuarantine<C, W>,
    ) {
        (self.input, self.outputs, self.candidate, self.quarantine)
    }

    #[cfg(test)]
    pub(in super::super) fn write_abort_echo(&mut self) {
        self.input.write_abort_echo(&mut self.outputs);
    }

    #[cfg(test)]
    pub(in super::super) fn sabotage_neutral_input(&mut self) {
        self.input.tables[0].initial_logical_row_count ^= 1;
    }
}

impl ReservedGenerationOutputs {
    pub(in super::super) fn set_header(&mut self, value: GenerationOutputHeader) {
        set_output(&mut self.header, value, &mut self.extra_write);
    }

    pub(in super::super) fn set_table(&mut self, ordinal: usize, value: GenerationOutputTable) {
        if let Some(slot) = self.tables.get_mut(ordinal) {
            set_output(slot, value, &mut self.extra_write);
        } else {
            self.extra_write = true;
        }
    }

    pub(in super::super) fn set_index(&mut self, ordinal: usize, value: GenerationOutputIndex) {
        if let Some(slot) = self.indexes.get_mut(ordinal) {
            set_output(slot, value, &mut self.extra_write);
        } else {
            self.extra_write = true;
        }
    }
}

trait SentinelOutput {
    fn is_filled(&self) -> bool;
    fn mark_duplicate(&mut self);
}

impl SentinelOutput for GenerationOutputHeader {
    fn is_filled(&self) -> bool {
        self.filled
    }

    fn mark_duplicate(&mut self) {
        self.duplicate = true;
    }
}

impl SentinelOutput for GenerationOutputTable {
    fn is_filled(&self) -> bool {
        self.filled
    }

    fn mark_duplicate(&mut self) {
        self.duplicate = true;
    }
}

impl SentinelOutput for GenerationOutputIndex {
    fn is_filled(&self) -> bool {
        self.filled
    }

    fn mark_duplicate(&mut self) {
        self.duplicate = true;
    }
}

fn set_output<T: SentinelOutput>(slot: &mut T, value: T, extra_write: &mut bool) {
    if slot.is_filled() {
        slot.mark_duplicate();
        *extra_write = true;
    } else {
        *slot = value;
    }
}

impl SealedGenerationInput {
    /// Builder-side digest traversal.  It hashes only owned neutral values and intentionally
    /// does not call the retained graph validator.
    pub(in super::super) fn builder_input_digest(&self) -> Result<CanonicalDigest, EngineError> {
        let mut digest = begin_digest(b"gpu-db/write001/generation-input/v2");
        digest.update(self.identity.database_id);
        digest.update(self.identity.catalog_epoch.to_le_bytes());
        digest.update(self.identity.catalog_digest);
        digest.update(self.identity.stable_transaction_id.to_le_bytes());
        digest.update(self.identity.commit_sequence.to_le_bytes());
        digest.update(self.identity.initial_database_root);
        digest.update(count_u32(self.tables.len(), "sealed generation table count")?.to_le_bytes());
        let mut previous_table = None;
        for table in &self.tables {
            if previous_table.is_some_and(|previous| table.stable_table_id <= previous) {
                return Err(generation_error(
                    "sealed neutral tables are not stable ordered",
                ));
            }
            digest.update(table.stable_table_id.to_le_bytes());
            digest.update(table.base_data_generation.to_le_bytes());
            digest.update(table.base_table_root);
            digest.update(table.row_allocator_before.to_le_bytes());
            digest.update(table.row_allocator_high_water.to_le_bytes());
            digest.update(table.initial_logical_row_count.to_le_bytes());
            digest.update(table.final_logical_row_count.to_le_bytes());

            let rows = sealed_range(&self.rows, table.row_start, table.row_count)?;
            digest.update(count_u32(rows.len(), "sealed generation row count")?.to_le_bytes());
            let mut previous_row = None;
            for row in rows {
                if previous_row.is_some_and(|previous| row.stable_row_id <= previous) {
                    return Err(generation_error(
                        "sealed neutral rows are not stable ordered",
                    ));
                }
                if row.stable_table_id != table.stable_table_id {
                    return Err(generation_error(
                        "sealed row has the wrong stable table owner",
                    ));
                }
                digest.update(sealed_row_digest(
                    table.stable_table_id,
                    row,
                    &self.cells,
                    &self.values,
                )?);
                previous_row = Some(row.stable_row_id);
            }
            digest.update(table.image_layout_digest);
            digest.update(table.image_content_digest);

            let indexes = sealed_range(&self.indexes, table.index_start, table.index_count)?;
            digest.update(count_u32(indexes.len(), "sealed generation index count")?.to_le_bytes());
            let mut previous_index = None;
            for index in indexes {
                if previous_index.is_some_and(|previous| index.stable_index_id <= previous) {
                    return Err(generation_error(
                        "sealed neutral indexes are not stable ordered",
                    ));
                }
                if index.owner_stable_table_id != table.stable_table_id {
                    return Err(generation_error(
                        "sealed index has the wrong stable table owner",
                    ));
                }
                let shape = sealed_index_shape_digest(table.stable_table_id, index, &self.keys)?;
                digest.update(shape);
                let effects = sealed_range(&self.effects, index.effect_start, index.effect_count)?;
                digest.update(
                    count_u32(effects.len(), "sealed generation effect count")?.to_le_bytes(),
                );
                for effect in effects {
                    if effect.stable_table_id != table.stable_table_id
                        || effect.stable_index_id != index.stable_index_id
                    {
                        return Err(generation_error(
                            "sealed maintenance effect has the wrong explicit owner",
                        ));
                    }
                    digest.update(sealed_effect_digest(
                        table.stable_table_id,
                        effect,
                        shape,
                        &self.effect_values,
                        &self.effect_value_bytes,
                    )?);
                }
                previous_index = Some(index.stable_index_id);
            }
            previous_table = Some(table.stable_table_id);
        }
        Ok(digest.finalize().into())
    }

    #[cfg(test)]
    pub(in super::super) fn write_abort_echo(&self, outputs: &mut ReservedGenerationOutputs) {
        let input_digest = self
            .builder_input_digest()
            .expect("prepared neutral input remains self-addressable");
        outputs.set_header(GenerationOutputHeader {
            filled: true,
            duplicate: false,
            root_descriptor_version: ROOT_DESCRIPTOR_VERSION,
            database_id: self.identity.database_id,
            catalog_epoch: self.identity.catalog_epoch,
            catalog_digest: self.identity.catalog_digest,
            stable_transaction_id: self.identity.stable_transaction_id,
            commit_sequence: self.identity.commit_sequence,
            initial_database_root: self.identity.initial_database_root,
            generation_input_digest: input_digest,
            final_database_root: self.identity.initial_database_root,
        });
        for (table_ordinal, table) in self.tables.iter().enumerate() {
            outputs.set_table(
                table_ordinal,
                GenerationOutputTable {
                    filled: true,
                    duplicate: false,
                    stable_table_id: table.stable_table_id,
                    final_data_generation: table.base_data_generation,
                    final_table_root: table.base_table_root,
                    final_logical_row_count: table.initial_logical_row_count,
                    index_start: table.index_start,
                    index_count: table.index_count,
                },
            );
        }
        for (index_ordinal, index) in self.indexes.iter().enumerate() {
            outputs.set_index(
                index_ordinal,
                GenerationOutputIndex {
                    filled: true,
                    duplicate: false,
                    stable_index_id: index.stable_index_id,
                    final_index_generation: index.base_generation,
                    final_index_root: index.base_root,
                },
            );
        }
    }
}

pub(in super::super) fn measure(
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    identity: SemanticsV2BoundIdentity,
) -> Result<GenerationInputMeasure, EngineError> {
    let mut measure = GenerationInputMeasure {
        tables: graph.tables.len(),
        ..GenerationInputMeasure::default()
    };
    let mut previous_table = None;
    for (table_ordinal, table) in graph.tables.iter().enumerate() {
        let catalog_table =
            catalog_target_table(catalog, table.stable_table_id, table.display_oid)?;
        validate_table(identity, table, catalog_table, table_ordinal)?;
        let transitions = range(
            &graph.transitions,
            table.transition_start,
            table.transition_count,
            "table transition",
        )?;
        let image = graph.images.get(table.image_ref as usize).ok_or_else(|| {
            generation_error("table final image reference is outside retained graph")
        })?;
        validate_image(table, catalog_table, image)?;
        let mut previous_row = None;
        for (ordinal, transition) in transitions.iter().enumerate() {
            validate_transition(table, transition, ordinal, previous_row)?;
            measure.rows = checked_add(measure.rows, 1, "neutral row count")?;
            for column in catalog_table.catalog_columns {
                let bytes = image_cell_len(image, column, transition.image_row_ordinal)?;
                measure.cells = checked_add(measure.cells, 1, "neutral cell count")?;
                measure.value_bytes =
                    checked_add(measure.value_bytes, bytes, "neutral value bytes")?;
            }
            previous_row = Some(transition.stable_row_id);
        }
        let indexes = range(
            &graph.indexes,
            table.owned_index_start,
            table.owned_index_count,
            "table index",
        )?;
        measure.indexes = checked_add(measure.indexes, indexes.len(), "neutral index count")?;
        let mut previous_index = None;
        let mut role_one = 0_usize;
        for (ordinal, index) in indexes.iter().enumerate() {
            validate_index(graph, table, index, catalog, ordinal, previous_index)?;
            measure.keys =
                checked_add(measure.keys, index.key_count as usize, "neutral key count")?;
            for transition in transitions {
                for effect in range(
                    &graph.key_effects,
                    transition.key_effect_start,
                    transition.key_effect_count,
                    "transition key effect",
                )? {
                    if effect.role == PHYSICAL_MAINTENANCE_ROLE
                        && effect.index_ref == index.index_ref
                    {
                        validate_effect(graph, image, transition, effect, index)?;
                        measure.effects = checked_add(measure.effects, 1, "neutral effect count")?;
                        measure.effect_values = checked_add(
                            measure.effect_values,
                            effect.key_arity as usize,
                            "neutral effect-value count",
                        )?;
                        for component in range(
                            &graph.key_components,
                            effect.new_component_start,
                            effect.new_component_count,
                            "maintenance key component",
                        )? {
                            measure.effect_value_bytes = checked_add(
                                measure.effect_value_bytes,
                                image_cell_len_by_ordinal(
                                    image,
                                    component.source_catalog_ordinal,
                                    transition.image_row_ordinal,
                                    component.storage,
                                    component.declared_type_oid,
                                    component.signed_type_size,
                                )?,
                                "neutral effect-value bytes",
                            )?;
                        }
                        role_one = checked_add(role_one, 1, "role-one effect count")?;
                    }
                }
            }
            previous_index = Some(index.stable_index_id);
        }
        let total_role_one = range(
            &graph.key_effects,
            table.key_effect_start,
            table.key_effect_count,
            "table key effect",
        )?
        .iter()
        .filter(|effect| effect.role == PHYSICAL_MAINTENANCE_ROLE)
        .count();
        if role_one != total_role_one {
            return Err(generation_error(
                "neutral generation measure lost a physical maintenance effect",
            ));
        }
        if previous_table.is_some_and(|previous| table.stable_table_id <= previous) {
            return Err(generation_error("neutral tables are not stable ordered"));
        }
        previous_table = Some(table.stable_table_id);
    }
    Ok(measure)
}

#[derive(Default)]
pub(in super::super) struct GenerationInputMeasure {
    tables: usize,
    rows: usize,
    cells: usize,
    value_bytes: usize,
    indexes: usize,
    keys: usize,
    effects: usize,
    effect_values: usize,
    effect_value_bytes: usize,
}

impl GenerationInputMeasure {
    pub(in super::super) fn builder_reservation(&self) -> GenerationBuilderReservation {
        GenerationBuilderReservation {
            tables: self.tables,
            rows: self.rows,
            cells: self.cells,
            value_bytes: self.value_bytes,
            indexes: self.indexes,
            keys: self.keys,
            effects: self.effects,
            effect_values: self.effect_values,
            effect_value_bytes: self.effect_value_bytes,
            table_outputs: self.tables,
            index_outputs: self.indexes,
        }
    }
}

pub(in super::super) fn reserve_and_fill<C, W>(
    measure: GenerationInputMeasure,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    identity: SemanticsV2BoundIdentity,
    candidate: C,
    quarantine_registry: &GenerationQuarantineRegistry<C, W>,
) -> Result<ReservedGenerationLaunch<C, W>, EngineError> {
    let mut tables = reserve_exact(measure.tables, "neutral table")?;
    let mut rows = reserve_exact(measure.rows, "neutral row")?;
    let mut cells = reserve_exact(measure.cells, "neutral cell")?;
    let mut values = reserve_exact(measure.value_bytes, "neutral value")?;
    let mut indexes = reserve_exact(measure.indexes, "neutral index")?;
    let mut keys = reserve_exact(measure.keys, "neutral key")?;
    let mut effects = reserve_exact(measure.effects, "neutral effect")?;
    let mut effect_values = reserve_exact(measure.effect_values, "neutral effect value")?;
    let mut effect_value_bytes = reserve_exact(measure.effect_value_bytes, "neutral effect bytes")?;
    let mut output_tables = reserve_exact(measure.tables, "generation table output")?;
    let mut output_indexes = reserve_exact(measure.indexes, "generation index output")?;
    let quarantine = quarantine_registry.register()?;

    for _ in 0..measure.tables {
        output_tables.push(GenerationOutputTable::sentinel());
    }
    for _ in 0..measure.indexes {
        output_indexes.push(GenerationOutputIndex::sentinel());
    }

    for table in &graph.tables {
        let catalog_table =
            catalog_target_table(catalog, table.stable_table_id, table.display_oid)?;
        let image = graph.images.get(table.image_ref as usize).ok_or_else(|| {
            generation_error("table final image reference vanished while filling neutral input")
        })?;
        let row_start = count_u32(rows.len(), "neutral row start")?;
        let transitions = range(
            &graph.transitions,
            table.transition_start,
            table.transition_count,
            "table transition",
        )?;
        for transition in transitions {
            let cell_start = count_u32(cells.len(), "neutral cell start")?;
            for column in catalog_table.catalog_columns {
                let value_start = count_u32(values.len(), "neutral cell value start")?;
                let (is_null, value_count) = append_image_cell_value(
                    image,
                    column,
                    transition.image_row_ordinal,
                    &mut values,
                )?;
                cells.push(NeutralCell {
                    catalog_column_ordinal: column.catalog_column_ordinal,
                    stable_column_id: column.stable_column_id,
                    attnum: column.attnum,
                    storage: column.storage,
                    declared_type_oid: column.declared_type_oid,
                    signed_type_size: column.signed_type_size,
                    is_null,
                    value_start,
                    value_count,
                });
            }
            rows.push(NeutralRow {
                stable_table_id: table.stable_table_id,
                stable_row_id: transition.stable_row_id,
                source_statement_ordinal: transition.source_statement_ordinal,
                source_row_ordinal: transition.source_row_ordinal,
                cell_start,
                cell_count: table.catalog_column_count,
            });
        }
        let index_start = count_u32(indexes.len(), "neutral index start")?;
        let retained_indexes = range(
            &graph.indexes,
            table.owned_index_start,
            table.owned_index_count,
            "table index",
        )?;
        for index in retained_indexes {
            let key_start = count_u32(keys.len(), "neutral key start")?;
            for key in range(
                &graph.index_key_columns,
                index.key_start,
                index.key_count,
                "index key column",
            )? {
                keys.push(neutral_key(key));
            }
            let effect_start = count_u32(effects.len(), "neutral effect start")?;
            for transition in transitions {
                for effect in range(
                    &graph.key_effects,
                    transition.key_effect_start,
                    transition.key_effect_count,
                    "transition key effect",
                )? {
                    if effect.role != PHYSICAL_MAINTENANCE_ROLE
                        || effect.index_ref != index.index_ref
                    {
                        continue;
                    }
                    let value_start = count_u32(effect_values.len(), "neutral effect value start")?;
                    for component in range(
                        &graph.key_components,
                        effect.new_component_start,
                        effect.new_component_count,
                        "maintenance key component",
                    )? {
                        let bytes_start =
                            count_u32(effect_value_bytes.len(), "neutral effect byte start")?;
                        let (is_null, value_count) = append_image_cell_value_by_ordinal(
                            image,
                            component.source_catalog_ordinal,
                            transition.image_row_ordinal,
                            component.storage,
                            component.declared_type_oid,
                            component.signed_type_size,
                            &mut effect_value_bytes,
                        )?;
                        effect_values.push(NeutralTypedValue {
                            storage: component.storage,
                            declared_type_oid: component.declared_type_oid,
                            signed_type_size: component.signed_type_size,
                            is_null,
                            value_start: bytes_start,
                            value_count,
                        });
                    }
                    effects.push(NeutralEffect {
                        stable_table_id: table.stable_table_id,
                        stable_index_id: index.stable_index_id,
                        stable_row_id: transition.stable_row_id,
                        source_catalog_ordinal: effect.source_catalog_ordinal,
                        key_arity: effect.key_arity,
                        value_start,
                        value_count: effect.new_component_count,
                    });
                }
            }
            indexes.push(NeutralIndex {
                owner_stable_table_id: table.stable_table_id,
                stable_index_id: index.stable_index_id,
                flags: index.flags,
                null_equality_policy: index.null_equality_policy,
                base_generation: index.base_index_generation,
                base_root: index.base_index_root,
                key_start,
                key_count: index.key_count,
                effect_start,
                effect_count: count_u32(effects.len(), "neutral effect end")?
                    .checked_sub(effect_start)
                    .ok_or_else(|| generation_error("neutral effect range underflows"))?,
            });
        }
        tables.push(NeutralTable {
            stable_table_id: table.stable_table_id,
            base_data_generation: table.data_generation_before,
            base_table_root: table.initial_table_root,
            row_allocator_before: table.row_allocator_before,
            row_allocator_high_water: table.row_allocator_high_water,
            initial_logical_row_count: table.initial_logical_row_count,
            final_logical_row_count: table.final_logical_row_count,
            image_layout_digest: table.image_layout_digest,
            image_content_digest: table.image_content_digest,
            row_start,
            row_count: table.transition_count,
            index_start,
            index_count: table.owned_index_count,
        });
    }

    require_exact(&tables, measure.tables, "neutral table")?;
    require_exact(&rows, measure.rows, "neutral row")?;
    require_exact(&cells, measure.cells, "neutral cell")?;
    require_exact(&values, measure.value_bytes, "neutral value")?;
    require_exact(&indexes, measure.indexes, "neutral index")?;
    require_exact(&keys, measure.keys, "neutral key")?;
    require_exact(&effects, measure.effects, "neutral effect")?;
    require_exact(
        &effect_values,
        measure.effect_values,
        "neutral effect value",
    )?;
    require_exact(
        &effect_value_bytes,
        measure.effect_value_bytes,
        "neutral effect bytes",
    )?;
    require_exact(&output_tables, measure.tables, "generation table output")?;
    require_exact(&output_indexes, measure.indexes, "generation index output")?;

    Ok(ReservedGenerationLaunch {
        input: SealedGenerationInput {
            identity: NeutralIdentity {
                database_id: identity.database_id,
                catalog_epoch: identity.catalog_epoch,
                catalog_digest: identity.catalog_digest,
                stable_transaction_id: identity.stable_transaction_id,
                commit_sequence: identity.commit_sequence,
                initial_database_root: identity.initial_database_root,
            },
            tables,
            rows,
            cells,
            values,
            indexes,
            keys,
            effects,
            effect_values,
            effect_value_bytes,
            #[cfg(test)]
            retention_sentinel: None,
        },
        outputs: ReservedGenerationOutputs {
            header: GenerationOutputHeader::sentinel(),
            tables: output_tables,
            indexes: output_indexes,
            extra_write: false,
            #[cfg(test)]
            retention_sentinel: None,
        },
        candidate,
        quarantine,
    })
}

fn neutral_key(key: &RetainedIndexKeyColumn) -> NeutralIndexKey {
    NeutralIndexKey {
        key_ordinal: key.key_ordinal,
        owner_catalog_column_ordinal: key.owner_catalog_column_ordinal,
        stable_column_id: key.stable_column_id,
        attnum: key.attnum,
        storage: key.storage,
        declared_type_oid: key.declared_type_oid,
        signed_type_size: key.signed_type_size,
        column_name_digest: key.column_name_digest,
    }
}

fn validate_table(
    identity: SemanticsV2BoundIdentity,
    table: &RetainedTable,
    catalog_table: &SemanticsV2CatalogTableWitness<'_>,
    ordinal: usize,
) -> Result<(), EngineError> {
    if table.table_ref != ordinal as u32
        || table.image_ref != ordinal as u32
        || table.stable_table_id != catalog_table.stable_table_id
        || table.display_oid != catalog_table.display_oid
        || table.catalog_epoch != identity.catalog_epoch
        || table.schema_digest != catalog_table.schema_digest
        || table.data_generation_before != catalog_table.data_generation
        || table.initial_table_root != catalog_table.data_root
        || table.catalog_column_count as usize != catalog_table.catalog_columns.len()
    {
        return Err(generation_error(
            "neutral table differs from pinned catalog shape",
        ));
    }
    Ok(())
}

fn validate_image(
    table: &RetainedTable,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
    image: &DecodedTypedImage,
) -> Result<(), EngineError> {
    let facts = image.facts();
    if facts.role != TypedImageRole::FinalTableImage
        || facts.rows != table.transition_count
        || facts.columns != table.catalog_column_count
        || facts.layout_digest != table.image_layout_digest
    {
        return Err(generation_error(
            "neutral final image facts differ from retained table",
        ));
    }
    let mut image_columns = image.columns();
    for (ordinal, column) in catalog.catalog_columns.iter().enumerate() {
        let image_column = image_columns.next().ok_or_else(|| {
            generation_error("neutral final image has fewer columns than the catalog")
        })?;
        if image_column.catalog_column_ordinal != ordinal as u32
            || image_column.catalog_column_ordinal != column.catalog_column_ordinal
            || image_column.stable_column_id != column.stable_column_id
            || image_column.attnum != column.attnum
            || image_column.type_oid != column.declared_type_oid
            || image_column.type_size != column.signed_type_size
            || !storage_matches_type(image_column.ty, column.storage)
        {
            return Err(generation_error(
                "neutral final image column differs from catalog",
            ));
        }
    }
    if image_columns.next().is_some() {
        return Err(generation_error(
            "neutral final image has extra catalog columns",
        ));
    }
    Ok(())
}

fn validate_transition(
    table: &RetainedTable,
    transition: &RetainedTransition,
    ordinal: usize,
    previous: Option<u64>,
) -> Result<(), EngineError> {
    if previous.is_some_and(|prior| transition.stable_row_id <= prior)
        || transition.transition_ref
            != directory_ref(table.transition_start, ordinal, "table transition")?
        || transition.table_ref != table.table_ref
        || transition.image_ref != table.image_ref
    {
        return Err(generation_error(
            "neutral transition ordering or ownership is invalid",
        ));
    }
    Ok(())
}

fn validate_index(
    graph: &ReservedSemanticsV2Graph,
    table: &RetainedTable,
    index: &RetainedIndexDescriptor,
    catalog: &SemanticsV2CatalogWitness<'_>,
    ordinal: usize,
    previous: Option<u64>,
) -> Result<(), EngineError> {
    if previous.is_some_and(|prior| index.stable_index_id <= prior)
        || index.index_ref != directory_ref(table.owned_index_start, ordinal, "table index")?
        || index.owner_table_ref != table.table_ref
        || index.owner_stable_table_id != table.stable_table_id
    {
        return Err(generation_error(
            "neutral index ordering or owner is invalid",
        ));
    }
    let catalog_index = catalog_index(catalog, index.stable_index_id)?;
    if index.stable_index_id != catalog_index.stable_index_id
        || index.display_oid != catalog_index.display_oid
        || index.owner_stable_table_id != catalog_index.owner_stable_table_id
        || index.owner_display_oid != catalog_index.owner_display_oid
        || index.catalog_epoch != catalog_index.catalog_epoch
        || index.flags != catalog_index.index_flags
        || index.null_equality_policy != catalog_index.null_equality_policy
        || index.base_index_generation != catalog_index.base_generation
        || index.base_index_root != catalog_index.base_root
        || index.key_count as usize != catalog_index.key_columns.len()
    {
        return Err(generation_error(
            "neutral index differs from pinned catalog shape",
        ));
    }
    for (key_ordinal, (key, catalog_key)) in range(
        &graph.index_key_columns,
        index.key_start,
        index.key_count,
        "index key column",
    )?
    .iter()
    .zip(catalog_index.key_columns.iter())
    .enumerate()
    {
        if key.index_ref != index.index_ref
            || key.key_column_ref
                != directory_ref(index.key_start, key_ordinal, "index key column")?
            || key.key_ordinal != key_ordinal as u32
            || key.key_ordinal != catalog_key.key_ordinal
            || key.owner_catalog_column_ordinal != catalog_key.owner_catalog_column_ordinal
            || key.stable_column_id != catalog_key.stable_column_id
            || key.attnum != catalog_key.attnum
            || key.storage != catalog_key.storage
            || key.declared_type_oid != catalog_key.declared_type_oid
            || key.signed_type_size != catalog_key.signed_type_size
            || key.column_name_digest != catalog_key.column_name_digest
        {
            return Err(generation_error(
                "neutral index key differs from pinned catalog key",
            ));
        }
    }
    Ok(())
}

fn validate_effect(
    graph: &ReservedSemanticsV2Graph,
    image: &DecodedTypedImage,
    transition: &RetainedTransition,
    effect: &RetainedKeyEffect,
    index: &RetainedIndexDescriptor,
) -> Result<(), EngineError> {
    if effect.role != PHYSICAL_MAINTENANCE_ROLE
        || effect.transition_ref != transition.transition_ref
        || effect.index_ref != index.index_ref
        || effect.key_arity != index.key_count
        || effect.new_component_count != index.key_count
    {
        return Err(generation_error(
            "neutral maintenance effect does not bind index shape",
        ));
    }
    let keys = range(
        &graph.index_key_columns,
        index.key_start,
        index.key_count,
        "index key column",
    )?;
    for (ordinal, (component, key)) in range(
        &graph.key_components,
        effect.new_component_start,
        effect.new_component_count,
        "maintenance key component",
    )?
    .iter()
    .zip(keys.iter())
    .enumerate()
    {
        if component.effect_ref != effect.effect_ref
            || component.component_ref
                != directory_ref(
                    effect.new_component_start,
                    ordinal,
                    "maintenance key component",
                )?
            || component.component_ordinal != ordinal as u32
            || component.key_column_ref != key.key_column_ref
            || component.source_catalog_ordinal != key.owner_catalog_column_ordinal
            || component.storage != key.storage
            || component.declared_type_oid != key.declared_type_oid
            || component.signed_type_size != key.signed_type_size
        {
            return Err(generation_error(
                "neutral effect component differs from index key",
            ));
        }
        let _ = image_cell_len_by_ordinal(
            image,
            component.source_catalog_ordinal,
            transition.image_row_ordinal,
            component.storage,
            component.declared_type_oid,
            component.signed_type_size,
        )?;
    }
    Ok(())
}

fn sealed_row_digest(
    table_id: u64,
    row: &NeutralRow,
    cells: &[NeutralCell],
    values: &[u8],
) -> Result<CanonicalDigest, EngineError> {
    let mut digest = begin_digest(b"gpu-db/write001/generation-row-input/v2");
    digest.update(table_id.to_le_bytes());
    digest.update(row.stable_row_id.to_le_bytes());
    digest.update(row.source_statement_ordinal.to_le_bytes());
    digest.update(row.source_row_ordinal.to_le_bytes());
    let cells = sealed_range(cells, row.cell_start, row.cell_count)?;
    digest.update(count_u32(cells.len(), "sealed row cell count")?.to_le_bytes());
    for cell in cells {
        digest.update(cell.catalog_column_ordinal.to_le_bytes());
        digest.update(cell.stable_column_id.to_le_bytes());
        digest.update(cell.attnum.to_le_bytes());
        digest.update(cell.storage);
        digest.update(cell.declared_type_oid.to_le_bytes());
        digest.update(cell.signed_type_size.to_le_bytes());
        digest.update([u8::from(cell.is_null)]);
        let value = sealed_range(values, cell.value_start, cell.value_count)?;
        digest.update(count_u32(value.len(), "sealed cell value length")?.to_le_bytes());
        digest.update(value);
    }
    Ok(digest.finalize().into())
}

fn sealed_index_shape_digest(
    table_id: u64,
    index: &NeutralIndex,
    keys: &[NeutralIndexKey],
) -> Result<CanonicalDigest, EngineError> {
    let mut digest = begin_digest(b"gpu-db/write001/generation-index-shape/v2");
    digest.update(table_id.to_le_bytes());
    digest.update(index.stable_index_id.to_le_bytes());
    digest.update(index.flags.to_le_bytes());
    digest.update([index.null_equality_policy]);
    digest.update(index.base_generation.to_le_bytes());
    digest.update(index.base_root);
    let keys = sealed_range(keys, index.key_start, index.key_count)?;
    digest.update(count_u32(keys.len(), "sealed index key count")?.to_le_bytes());
    for key in keys {
        digest.update(key.key_ordinal.to_le_bytes());
        digest.update(key.owner_catalog_column_ordinal.to_le_bytes());
        digest.update(key.stable_column_id.to_le_bytes());
        digest.update(key.attnum.to_le_bytes());
        digest.update(key.storage);
        digest.update(key.declared_type_oid.to_le_bytes());
        digest.update(key.signed_type_size.to_le_bytes());
        digest.update(key.column_name_digest);
    }
    Ok(digest.finalize().into())
}

fn sealed_effect_digest(
    table_id: u64,
    effect: &NeutralEffect,
    shape: CanonicalDigest,
    values: &[NeutralTypedValue],
    bytes: &[u8],
) -> Result<CanonicalDigest, EngineError> {
    let mut digest = begin_digest(b"gpu-db/write001/generation-index-effect-input/v2");
    digest.update(table_id.to_le_bytes());
    digest.update(effect.stable_row_id.to_le_bytes());
    digest.update(effect.source_catalog_ordinal.to_le_bytes());
    digest.update(shape);
    digest.update(effect.key_arity.to_le_bytes());
    let values = sealed_range(values, effect.value_start, effect.value_count)?;
    if values.len() != effect.key_arity as usize {
        return Err(generation_error(
            "sealed effect value count differs from key arity",
        ));
    }
    for value in values {
        let mut typed = begin_digest(b"gpu-db/write001/s7-typed-key-value/v2");
        typed.update(value.storage);
        typed.update(value.declared_type_oid.to_le_bytes());
        typed.update(value.signed_type_size.to_le_bytes());
        typed.update([u8::from(value.is_null)]);
        let bytes = sealed_range(bytes, value.value_start, value.value_count)?;
        typed.update(count_u32(bytes.len(), "sealed typed value byte count")?.to_le_bytes());
        typed.update(bytes);
        digest.update(typed.finalize());
    }
    Ok(digest.finalize().into())
}

fn image_cell_len(
    image: &DecodedTypedImage,
    column: &SemanticsV2CatalogColumnWitness<'_>,
    row: u32,
) -> Result<usize, EngineError> {
    image_cell_len_by_ordinal(
        image,
        column.catalog_column_ordinal,
        row,
        column.storage,
        column.declared_type_oid,
        column.signed_type_size,
    )
}

fn image_cell_len_by_ordinal(
    image: &DecodedTypedImage,
    ordinal: u32,
    row: u32,
    storage: [u8; 4],
    oid: u32,
    size: i16,
) -> Result<usize, EngineError> {
    let mut length = None;
    append_image_cell_by_ordinal(image, ordinal, row, storage, oid, size, &mut |_, bytes| {
        length = Some(bytes.len());
    })?;
    length.ok_or_else(|| generation_error("neutral image cell length was not produced"))
}

fn append_image_cell_value(
    image: &DecodedTypedImage,
    column: &SemanticsV2CatalogColumnWitness<'_>,
    row: u32,
    target: &mut Vec<u8>,
) -> Result<(bool, u32), EngineError> {
    append_image_cell_value_by_ordinal(
        image,
        column.catalog_column_ordinal,
        row,
        column.storage,
        column.declared_type_oid,
        column.signed_type_size,
        target,
    )
}

fn append_image_cell_value_by_ordinal(
    image: &DecodedTypedImage,
    ordinal: u32,
    row: u32,
    storage: [u8; 4],
    oid: u32,
    size: i16,
    target: &mut Vec<u8>,
) -> Result<(bool, u32), EngineError> {
    let mut cell = None;
    append_image_cell_by_ordinal(
        image,
        ordinal,
        row,
        storage,
        oid,
        size,
        &mut |is_null, bytes| {
            target.extend_from_slice(bytes);
            cell = Some((is_null, bytes.len()));
        },
    )?;
    let (is_null, len) =
        cell.ok_or_else(|| generation_error("neutral image cell was not produced"))?;
    Ok((is_null, count_u32(len, "neutral image cell value length")?))
}

fn append_image_cell_by_ordinal(
    image: &DecodedTypedImage,
    ordinal: u32,
    row: u32,
    storage: [u8; 4],
    oid: u32,
    size: i16,
    sink: &mut impl FnMut(bool, &[u8]),
) -> Result<(), EngineError> {
    let facts = image.facts();
    let row =
        usize::try_from(row).map_err(|_| generation_error("neutral image row is unaddressable"))?;
    let rows = usize::try_from(facts.rows)
        .map_err(|_| generation_error("neutral image row count is unaddressable"))?;
    if row >= rows {
        return Err(generation_error("neutral image row is outside final image"));
    }
    let mut found = None;
    for column in image.columns() {
        if column.catalog_column_ordinal != ordinal {
            continue;
        }
        if found.replace(()).is_some()
            || column.type_oid != oid
            || column.type_size != size
            || !storage_matches_type(column.ty, storage)
        {
            return Err(generation_error(
                "neutral image column is ambiguous or mistyped",
            ));
        }
        let valid = match column.validity {
            TypedInsertColumnValidity::AllValid => true,
            TypedInsertColumnValidity::Bitmap(words) => words
                .get(row / 32)
                .is_some_and(|word| (word & (1_u32 << (row % 32))) != 0),
        };
        if !valid {
            sink(true, &[]);
            return Ok(());
        }
        match (column.values, column.ty) {
            (
                TypedInsertColumnValues::I32(values),
                SqlType::Int2 | SqlType::Int4 | SqlType::Date,
            ) => {
                let value = values
                    .get(row)
                    .ok_or_else(|| generation_error("neutral i32 image vector is short"))?;
                sink(false, &value.to_le_bytes());
            }
            (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
                let value = values
                    .get(row)
                    .ok_or_else(|| generation_error("neutral i64 image vector is short"))?;
                sink(false, &value.to_le_bytes());
            }
            (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => {
                let value = values
                    .get(row)
                    .ok_or_else(|| generation_error("neutral numeric image vector is short"))?;
                sink(false, &value.to_le_bytes());
            }
            (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => {
                let value = values
                    .get(row)
                    .ok_or_else(|| generation_error("neutral UUID image vector is short"))?;
                sink(false, value);
            }
            (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
                let word = words
                    .get(row / 32)
                    .ok_or_else(|| generation_error("neutral bool image vector is short"))?;
                let value = [u8::from((word & (1_u32 << (row % 32))) != 0)];
                sink(false, &value);
            }
            (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
                let start = *offsets
                    .get(row)
                    .ok_or_else(|| generation_error("neutral text offsets are short"))?;
                let end = *offsets
                    .get(row + 1)
                    .ok_or_else(|| generation_error("neutral text offsets are short"))?;
                let start = usize::try_from(start)
                    .map_err(|_| generation_error("neutral text offset is unaddressable"))?;
                let end = usize::try_from(end)
                    .map_err(|_| generation_error("neutral text offset is unaddressable"))?;
                let value = bytes
                    .get(start..end)
                    .ok_or_else(|| generation_error("neutral text range is invalid"))?;
                sink(false, value);
            }
            _ => {
                return Err(generation_error(
                    "neutral image vector arm differs from SQL type",
                ))
            }
        }
        return Ok(());
    }
    Err(generation_error(
        "neutral typed key refers to a missing image column",
    ))
}

fn catalog_target_table<'a>(
    catalog: &'a SemanticsV2CatalogWitness<'a>,
    stable_table_id: u64,
    display_oid: u32,
) -> Result<&'a SemanticsV2CatalogTableWitness<'a>, EngineError> {
    let mut found = None;
    for table in catalog.tables {
        if table.stable_table_id == stable_table_id
            && table.display_oid == display_oid
            && found.replace(table).is_some()
        {
            return Err(generation_error(
                "pinned catalog has duplicate neutral table identity",
            ));
        }
    }
    found.ok_or_else(|| generation_error("neutral target table is absent from pinned catalog"))
}

fn catalog_index<'a>(
    catalog: &'a SemanticsV2CatalogWitness<'a>,
    stable_index_id: u64,
) -> Result<&'a SemanticsV2CatalogIndexWitness<'a>, EngineError> {
    let mut found = None;
    for index in catalog.indexes {
        if index.stable_index_id == stable_index_id && found.replace(index).is_some() {
            return Err(generation_error(
                "pinned catalog has duplicate neutral index identity",
            ));
        }
    }
    found.ok_or_else(|| generation_error("neutral index is absent from pinned catalog"))
}

fn sealed_range<T>(values: &[T], start: u32, count: u32) -> Result<&[T], EngineError> {
    let start = usize::try_from(start)
        .map_err(|_| generation_error("sealed range start is unaddressable"))?;
    let count = usize::try_from(count)
        .map_err(|_| generation_error("sealed range count is unaddressable"))?;
    values
        .get(
            start
                ..start
                    .checked_add(count)
                    .ok_or_else(|| generation_error("sealed range overflows"))?,
        )
        .ok_or_else(|| generation_error("sealed range is outside flat neutral input"))
}

fn count_u32(value: usize, owner: &str) -> Result<u32, EngineError> {
    u32::try_from(value).map_err(|_| generation_error(&format!("{owner} exceeds u32")))
}

fn checked_add(left: usize, right: usize, owner: &str) -> Result<usize, EngineError> {
    left.checked_add(right)
        .ok_or_else(|| generation_error(&format!("{owner} overflows")))
}

fn reserve_exact<T>(count: usize, owner: &'static str) -> Result<Vec<T>, EngineError> {
    #[cfg(test)]
    note_reservation(owner)?;
    #[cfg(not(test))]
    let _ = owner;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| generation_error("neutral generation reservation failed"))?;
    if values.capacity() != count {
        return Err(generation_error(
            "neutral generation reservation is not exact",
        ));
    }
    Ok(values)
}

fn require_exact<T>(values: &Vec<T>, expected: usize, owner: &str) -> Result<(), EngineError> {
    if values.len() != expected || values.capacity() != expected {
        return Err(generation_error(&format!(
            "{owner} fill differs from exact reservation"
        )));
    }
    Ok(())
}

fn begin_digest(domain: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest
}

#[cfg(test)]
thread_local! {
    static RESERVATION_FAILURE: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn note_reservation(_owner: &str) -> Result<(), EngineError> {
    RESERVATION_FAILURE.with(|state| {
        if let Some((fail_at, attempts)) = state.get() {
            let next = attempts + 1;
            state.set(Some((fail_at, next)));
            if fail_at == next {
                return Err(generation_error("injected neutral reservation failure"));
            }
        }
        Ok(())
    })
}

#[cfg(test)]
pub(in super::super) fn fail_reservation_at<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    RESERVATION_FAILURE.with(|state| {
        assert!(
            state.replace(Some((attempt, 0))).is_none(),
            "neutral reservation injection cannot nest"
        );
        let result = operation();
        state.set(None);
        result
    })
}

/// Lifecycle-only test fixture.  It is an owned neutral launch, not a graph or pending-owner
/// constructor, and exercises every allocation family through the same exact-reservation helper.
#[cfg(test)]
pub(in super::super) fn test_launch<C, W>(
    candidate: C,
    quarantine_registry: &GenerationQuarantineRegistry<C, W>,
) -> Result<ReservedGenerationLaunch<C, W>, EngineError> {
    let mut tables = reserve_exact(1, "neutral table")?;
    let rows = reserve_exact(0, "neutral row")?;
    let cells = reserve_exact(0, "neutral cell")?;
    let values = reserve_exact(0, "neutral value")?;
    let indexes = reserve_exact(0, "neutral index")?;
    let keys = reserve_exact(0, "neutral key")?;
    let effects = reserve_exact(0, "neutral effect")?;
    let effect_values = reserve_exact(0, "neutral effect value")?;
    let effect_value_bytes = reserve_exact(0, "neutral effect bytes")?;
    let mut output_tables = reserve_exact(1, "generation table output")?;
    let output_indexes = reserve_exact(0, "generation index output")?;
    let quarantine = quarantine_registry.register()?;
    tables.push(NeutralTable {
        stable_table_id: 1,
        base_data_generation: 1,
        base_table_root: [1; 32],
        row_allocator_before: 1,
        row_allocator_high_water: 1,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        image_layout_digest: [2; 32],
        image_content_digest: [3; 32],
        row_start: 0,
        row_count: 0,
        index_start: 0,
        index_count: 0,
    });
    output_tables.push(GenerationOutputTable::sentinel());
    require_exact(&tables, 1, "neutral table")?;
    require_exact(&rows, 0, "neutral row")?;
    require_exact(&cells, 0, "neutral cell")?;
    require_exact(&values, 0, "neutral value")?;
    require_exact(&indexes, 0, "neutral index")?;
    require_exact(&keys, 0, "neutral key")?;
    require_exact(&effects, 0, "neutral effect")?;
    require_exact(&effect_values, 0, "neutral effect value")?;
    require_exact(&effect_value_bytes, 0, "neutral effect bytes")?;
    require_exact(&output_tables, 1, "generation table output")?;
    require_exact(&output_indexes, 0, "generation index output")?;
    Ok(ReservedGenerationLaunch {
        input: SealedGenerationInput {
            identity: NeutralIdentity {
                database_id: [4; 16],
                catalog_epoch: 1,
                catalog_digest: [5; 32],
                stable_transaction_id: 1,
                commit_sequence: 1,
                initial_database_root: [6; 32],
            },
            tables,
            rows,
            cells,
            values,
            indexes,
            keys,
            effects,
            effect_values,
            effect_value_bytes,
            #[cfg(test)]
            retention_sentinel: None,
        },
        outputs: ReservedGenerationOutputs {
            header: GenerationOutputHeader::sentinel(),
            tables: output_tables,
            indexes: output_indexes,
            extra_write: false,
            #[cfg(test)]
            retention_sentinel: None,
        },
        candidate,
        quarantine,
    })
}

#[cfg(test)]
pub(in super::super) fn test_launch_with_retention_sentinels<C, W>(
    candidate: C,
    quarantine_registry: &GenerationQuarantineRegistry<C, W>,
    input_drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    output_drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> Result<ReservedGenerationLaunch<C, W>, EngineError> {
    let mut launch = test_launch(candidate, quarantine_registry)?;
    launch.input.retention_sentinel = Some(RetentionSentinel(input_drops));
    launch.outputs.retention_sentinel = Some(RetentionSentinel(output_drops));
    Ok(launch)
}

#[cfg(test)]
pub(in super::super) fn test_builder_reservation() -> GenerationBuilderReservation {
    GenerationBuilderReservation {
        tables: 1,
        rows: 0,
        cells: 0,
        value_bytes: 0,
        indexes: 0,
        keys: 0,
        effects: 0,
        effect_values: 0,
        effect_value_bytes: 0,
        table_outputs: 1,
        index_outputs: 0,
    }
}
