//! Private detached-owner boundary between one sealed bootstrap source and a future rebuild.
//!
//! It validates only pre-existing, explicitly retained resident or cold owners. It does not look
//! up live engine state, create roots, launch CUDA, or make a generation visible.

use std::{
    collections::BTreeSet,
    num::{NonZeroU64, NonZeroUsize},
    sync::Arc,
};

use gpu_db_execution::{
    CudaResidentDeviceMemory, RuntimeGenerationRebuildSource, RuntimeGenerationRebuildTarget,
};

use super::{
    bootstrap_publication::{
        BootstrapMaterializationLease, BootstrapResourceLedgerEntry, BootstrapResourceLedgerKind,
        BootstrapResourceLedgerOwner, BootstrapResourceStorageTier,
    },
    DataGenerationError,
};

/// Capability required to inspect a sealed lease's canonical claims. Its private field makes the
/// capability constructible only in this module, even though the lease accessor needs sibling
/// visibility for the private module boundary.
#[derive(Debug)]
pub(super) struct BootstrapResourceLeaseAccess {
    _private: (),
}

/// Opaque, typed identity for the one allocation retained by one source descriptor's later owner
/// bundle. This is not a device pointer and has no public constructor. The physical ledger is
/// deliberately one-to-one here: two claims never share an allocation identity, even for
/// non-overlapping byte ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct BootstrapAllocationIdentity(NonZeroUsize);

/// Typed, pointer-free allocation facts captured from the eventual CUDA owner. `device_ordinal`
/// and `context_identity` are retained for exact owner validation in the attach slice; this
/// foundation can already reject an empty allocation or an invalid byte span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BootstrapAllocationProof {
    device_ordinal: u16,
    context_identity: NonZeroUsize,
    allocated_bytes: u64,
}

/// One prospective allocation record for exactly one canonical source claim. `byte_offset` and
/// `byte_len` describe that claim's exact source range for the later typed-owner attach check;
/// they do not permit sharing an allocation with another claim. Its fields are private: only this
/// module may form records before the concrete pin-owner API exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BootstrapResourceAllocationRecord {
    claim: BootstrapResourceLedgerEntry,
    allocation_identity: BootstrapAllocationIdentity,
    proof: BootstrapAllocationProof,
    byte_offset: u64,
    byte_len: u64,
}

/// Check a sealed lease against prospective typed allocation records. Every canonical source
/// claim must have exactly one record and exactly one allocation identity. This intentionally
/// does not retain an owner, create a CUDA allocation, construct roots, or consult live engine
/// state. The forthcoming concrete pin-attach API will call this validator after it holds exact
/// owners and can validate each exact source range.
pub(super) fn validate_bootstrap_materialization_lease(
    lease: &BootstrapMaterializationLease,
    records: &[BootstrapResourceAllocationRecord],
) -> Result<(), DataGenerationError> {
    let access = BootstrapResourceLeaseAccess { _private: () };
    validate_resource_ledger(lease.resource_claims(&access), records)
}

fn validate_resource_ledger(
    claims: &[BootstrapResourceLedgerEntry],
    records: &[BootstrapResourceAllocationRecord],
) -> Result<(), DataGenerationError> {
    if records.len() < claims.len() {
        return Err(DataGenerationError::Missing(
            "bootstrap resource allocation coverage",
        ));
    }
    if records.len() > claims.len() {
        return Err(DataGenerationError::Unexpected(
            "bootstrap resource allocation record",
        ));
    }

    let mut allocation_identities = BTreeSet::new();
    for (claim, record) in claims.iter().zip(records) {
        if &record.claim != claim {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim",
            ));
        }
        if record.byte_offset != claim.byte_offset || record.byte_len != claim.byte_len.get() {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation range",
            ));
        }
        validate_claim_kind_owner(&record.claim)?;
        validate_allocation_proof(record)?;
        if !allocation_identities.insert(record.allocation_identity) {
            return Err(DataGenerationError::Invalid(
                "duplicate bootstrap resource allocation identity",
            ));
        }
    }
    Ok(())
}

fn validate_claim_kind_owner(
    claim: &BootstrapResourceLedgerEntry,
) -> Result<(), DataGenerationError> {
    let valid = matches!(
        (claim.kind, claim.owner),
        (
            BootstrapResourceLedgerKind::DatabaseManifest,
            BootstrapResourceLedgerOwner::Database
        ) | (
            BootstrapResourceLedgerKind::StatusView,
            BootstrapResourceLedgerOwner::Status
        ) | (
            BootstrapResourceLedgerKind::TablePayload,
            BootstrapResourceLedgerOwner::Table(_)
        ) | (
            BootstrapResourceLedgerKind::IndexPayload,
            BootstrapResourceLedgerOwner::Index { .. }
        ) | (
            BootstrapResourceLedgerKind::Sidecar,
            BootstrapResourceLedgerOwner::Table(_) | BootstrapResourceLedgerOwner::Index { .. }
        )
    );
    if !valid {
        return Err(DataGenerationError::Invalid(
            "bootstrap resource allocation owner",
        ));
    }
    Ok(())
}

fn validate_allocation_proof(
    record: &BootstrapResourceAllocationRecord,
) -> Result<(), DataGenerationError> {
    let _ = record.proof.device_ordinal;
    let _ = record.proof.context_identity;
    if record.proof.allocated_bytes == 0 || record.byte_len == 0 {
        return Err(DataGenerationError::Invalid(
            "bootstrap resource allocation span",
        ));
    }
    let span_end = record
        .byte_offset
        .checked_add(record.byte_len)
        .ok_or(DataGenerationError::CountOverflow)?;
    if span_end > record.proof.allocated_bytes {
        return Err(DataGenerationError::Invalid(
            "bootstrap resource allocation span",
        ));
    }
    Ok(())
}

/// One exact half-open byte range within a detached physical owner. A range may begin at zero,
/// but it may never be empty and is checked against its retained owner before attachment.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BootstrapResourceByteRange {
    offset: u64,
    len: NonZeroU64,
}

impl BootstrapResourceByteRange {
    fn end(self) -> Result<u64, DataGenerationError> {
        self.offset
            .checked_add(self.len.get())
            .ok_or(DataGenerationError::CountOverflow)
    }
}

/// One resident source owner retained independently of the live engine residency map. The raw
/// CUDA pointer remains encapsulated by `CudaResidentDeviceMemory`; this boundary records only
/// its allocation identity, metadata, context identity, and bounded logical range.
struct BootstrapDetachedResidentOwner {
    memory: Arc<CudaResidentDeviceMemory>,
    range: BootstrapResourceByteRange,
}

/// One immutable RAM source owner retained independently of any cold cache or table generation.
struct BootstrapDetachedColdRamOwner {
    bytes: Arc<[u8]>,
    range: BootstrapResourceByteRange,
}

/// An attachment owns exactly one physical representation for one source claim. Placement is
/// intentionally neutral here: a later GPU rebuild decides how to consume each validated owner.
enum BootstrapDetachedResourceOwner {
    Resident(BootstrapDetachedResidentOwner),
    ColdRam(BootstrapDetachedColdRamOwner),
}

impl BootstrapDetachedResourceOwner {
    fn tier(&self) -> BootstrapResourceStorageTier {
        match self {
            Self::Resident(_) => BootstrapResourceStorageTier::Resident,
            Self::ColdRam(_) => BootstrapResourceStorageTier::DetachedRam,
        }
    }

    fn range(&self) -> BootstrapResourceByteRange {
        match self {
            Self::Resident(owner) => owner.range,
            Self::ColdRam(owner) => owner.range,
        }
    }
}

/// One claimed detached owner. Fields stay private so no sibling can fabricate a bundle outside
/// the typed owner construction seam that will accompany the future rebuild module.
struct BootstrapResourceAttachment {
    claim: BootstrapResourceLedgerEntry,
    owner: BootstrapDetachedResourceOwner,
}

/// Exact ordered resource attachments for one sealed lease. It has no public constructor,
/// `Clone`, or `Debug` implementation; tests construct it only under `cfg(test)`.
pub(super) struct BootstrapResourceAttachmentBundle {
    target: BootstrapRebuildTarget,
    attachments: Box<[BootstrapResourceAttachment]>,
}

/// A private, attached-but-uninstalled owner for the future GPU rebuild. It retains the complete
/// sealed lease and every exact physical owner, but deliberately exposes no claims, readers, raw
/// pointers, `Clone`, or `Debug` surface.
pub(super) struct BootstrapAttachedUninstalledResources {
    _lease: BootstrapMaterializationLease,
    _attachments: Box<[BootstrapResourceAttachment]>,
    _target: BootstrapRebuildTarget,
}

/// Opaque, root-free ownership retained by the rebuild compiler. It deliberately keeps only the
/// already-validated detached owners in their canonical attachment order and the target runtime
/// identity. The resource claims (which carry comparator-only shape roots) are discarded here;
/// their separately projected root-free semantics are the compiler's only layout input.
pub(super) struct BootstrapRebuildAttachedOwners {
    attachments: Box<[BootstrapDetachedResourceOwner]>,
    runtime: BootstrapRebuildRuntimeIdentity,
}

/// Private target identity for one future GPU rebuild. This is not a context handle or a device
/// pointer; it is retained solely so later enqueue ownership cannot infer a target from a cache.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct BootstrapRebuildRuntimeIdentity {
    device_ordinal: u16,
    context_identity: NonZeroUsize,
}

impl BootstrapRebuildAttachedOwners {
    pub(super) fn attachment_count(&self) -> usize {
        self.attachments.len()
    }

    pub(super) fn runtime_identity(&self) -> BootstrapRebuildRuntimeIdentity {
        self.runtime
    }
}

/// A failed consuming execution bridge retains the opaque owners so a caller can retry only after
/// choosing a matching actual target. No detached owner, reader, or raw pointer is exposed.
pub(super) struct BootstrapRebuildExecutionSourceFailure {
    error: DataGenerationError,
    owners: BootstrapRebuildAttachedOwners,
}

impl BootstrapRebuildExecutionSourceFailure {
    pub(super) fn into_parts(self) -> (DataGenerationError, BootstrapRebuildAttachedOwners) {
        (self.error, self.owners)
    }
}

/// Consume exact attached owners into execution-only source capabilities. The engine receives no
/// device pointer or RAM slice; it can only place the resulting opaque source in the closed V1
/// rebuild descriptor. Resident allocations are checked against the caller's retained primary
/// target even when other attachments are cold RAM.
pub(super) fn into_runtime_generation_rebuild_sources(
    owners: BootstrapRebuildAttachedOwners,
    target: &RuntimeGenerationRebuildTarget,
) -> Result<Box<[RuntimeGenerationRebuildSource]>, BootstrapRebuildExecutionSourceFailure> {
    if owners.runtime.device_ordinal != target.device_ordinal()
        || owners.runtime.context_identity.get() != target.context_identity()
    {
        return Err(BootstrapRebuildExecutionSourceFailure {
            error: DataGenerationError::PredecessorMismatch("bootstrap rebuild execution target"),
            owners,
        });
    }
    let sources = owners
        .attachments
        .into_vec()
        .into_iter()
        .map(|owner| match owner {
            BootstrapDetachedResourceOwner::Resident(resident) => {
                RuntimeGenerationRebuildSource::resident(
                    resident.memory,
                    resident.range.offset,
                    resident.range.len.get(),
                )
            }
            BootstrapDetachedResourceOwner::ColdRam(cold) => {
                RuntimeGenerationRebuildSource::cold_ram(
                    cold.bytes,
                    cold.range.offset,
                    cold.range.len.get(),
                )
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(sources)
}

/// A failed attach retains the consumed lease and owner bundle for an in-module repair/retry
/// path. It exposes neither the candidate nor detached owners to sibling callers.
pub(super) struct BootstrapResourceAttachmentFailure {
    error: DataGenerationError,
    lease: BootstrapMaterializationLease,
    bundle: BootstrapResourceAttachmentBundle,
}

/// Explicit rebuild target retained even when every source owner is cold RAM. It prevents a
/// future GPU choice from being inferred from a live cache; resident owners must prove the same
/// GPU and primary-context identity during attachment.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BootstrapRebuildTarget {
    device_ordinal: u16,
    context_identity: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BootstrapResidentRuntime {
    device_ordinal: u16,
    context_identity: NonZeroUsize,
}

/// Consume one sealed lease and the exact ordered detached owners that belong to it. No logical
/// root, CUDA launch, reader state, installation, publication, or engine-state lookup occurs.
pub(super) fn attach_bootstrap_resource_bundle(
    lease: BootstrapMaterializationLease,
    bundle: BootstrapResourceAttachmentBundle,
) -> Result<BootstrapAttachedUninstalledResources, Box<BootstrapResourceAttachmentFailure>> {
    let access = BootstrapResourceLeaseAccess { _private: () };
    if let Err(error) = validate_attached_resource_ledger(
        lease.resource_claims(&access),
        bundle.target,
        &bundle.attachments,
    ) {
        return Err(Box::new(BootstrapResourceAttachmentFailure {
            error,
            lease,
            bundle,
        }));
    }
    Ok(BootstrapAttachedUninstalledResources {
        _lease: lease,
        _attachments: bundle.attachments,
        _target: bundle.target,
    })
}

/// Consume the attached source at the one rebuild bridge. Every owner remains retained, but the
/// ledger claims themselves do not cross into the compiler because they contain comparator-only
/// root facts. `bootstrap_publication` consumes the returned lease in the same atomic rebuild
/// preparation transition and emits the independently root-free semantic projection.
pub(super) fn into_bootstrap_rebuild_attached_owners(
    attached: BootstrapAttachedUninstalledResources,
) -> (
    BootstrapMaterializationLease,
    BootstrapRebuildAttachedOwners,
) {
    let BootstrapAttachedUninstalledResources {
        _lease,
        _attachments,
        _target,
    } = attached;
    let attachments = _attachments
        .into_vec()
        .into_iter()
        .map(|attachment| attachment.owner)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    (
        _lease,
        BootstrapRebuildAttachedOwners {
            attachments,
            runtime: BootstrapRebuildRuntimeIdentity {
                device_ordinal: _target.device_ordinal,
                context_identity: _target.context_identity,
            },
        },
    )
}

fn validate_attached_resource_ledger(
    claims: &[BootstrapResourceLedgerEntry],
    target: BootstrapRebuildTarget,
    attachments: &[BootstrapResourceAttachment],
) -> Result<(), DataGenerationError> {
    if attachments.len() < claims.len() {
        return Err(DataGenerationError::Missing(
            "bootstrap resource attachment coverage",
        ));
    }
    if attachments.len() > claims.len() {
        return Err(DataGenerationError::Unexpected(
            "bootstrap resource attachment",
        ));
    }

    let mut allocation_identities = BTreeSet::new();
    let mut resident_runtime = None;
    for (claim, attachment) in claims.iter().zip(attachments) {
        if &attachment.claim != claim {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource attachment claim",
            ));
        }
        validate_claim_kind_owner(&attachment.claim)?;
        if attachment.owner.tier() != claim.tier {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource attachment tier",
            ));
        }
        let owner_range = attachment.owner.range();
        if owner_range.offset != claim.byte_offset || owner_range.len != claim.byte_len {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource attachment range",
            ));
        }
        if let Some((allocation_identity, runtime)) = validate_detached_owner(&attachment.owner)? {
            if !allocation_identities.insert(allocation_identity) {
                return Err(DataGenerationError::Invalid(
                    "duplicate bootstrap resident allocation identity",
                ));
            }
            validate_resident_runtime(runtime, target, &mut resident_runtime)?;
        }
    }
    Ok(())
}

fn validate_detached_owner(
    owner: &BootstrapDetachedResourceOwner,
) -> Result<Option<(BootstrapAllocationIdentity, BootstrapResidentRuntime)>, DataGenerationError> {
    match owner {
        BootstrapDetachedResourceOwner::Resident(resident) => {
            validate_detached_resident_owner(resident).map(Some)
        }
        BootstrapDetachedResourceOwner::ColdRam(cold) => {
            validate_range(
                cold.range,
                u64::try_from(cold.bytes.len()).map_err(|_| DataGenerationError::CountOverflow)?,
                "bootstrap detached RAM range",
            )?;
            Ok(None)
        }
    }
}

fn validate_detached_resident_owner(
    resident: &BootstrapDetachedResidentOwner,
) -> Result<(BootstrapAllocationIdentity, BootstrapResidentRuntime), DataGenerationError> {
    let allocation_identity = BootstrapAllocationIdentity(
        NonZeroUsize::new(resident.memory.allocation_identity()).ok_or(
            DataGenerationError::ZeroIdentity("bootstrap resident allocation"),
        )?,
    );
    let metadata = resident.memory.metadata();
    if !metadata.retained || metadata.allocated_bytes == 0 {
        return Err(DataGenerationError::Invalid("bootstrap resident owner"));
    }
    if resident.range.offset != 0 || resident.range.len.get() != metadata.allocated_bytes {
        return Err(DataGenerationError::Invalid(
            "bootstrap resident whole allocation range",
        ));
    }
    if !resident.memory.has_full_contiguous_initialization() {
        return Err(DataGenerationError::Invalid(
            "bootstrap resident full contiguous initialization",
        ));
    }
    let context_identity = NonZeroUsize::new(resident.memory.context() as usize).ok_or(
        DataGenerationError::ZeroIdentity("bootstrap resident context"),
    )?;
    Ok((
        allocation_identity,
        BootstrapResidentRuntime {
            device_ordinal: metadata.gpu_id,
            context_identity,
        },
    ))
}

fn validate_resident_runtime(
    runtime: BootstrapResidentRuntime,
    target: BootstrapRebuildTarget,
    expected: &mut Option<BootstrapResidentRuntime>,
) -> Result<(), DataGenerationError> {
    if runtime.device_ordinal != target.device_ordinal
        || runtime.context_identity != target.context_identity
    {
        return Err(DataGenerationError::PredecessorMismatch(
            "bootstrap resident target",
        ));
    }
    if let Some(prior) = expected {
        if *prior != runtime {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resident runtime context",
            ));
        }
    } else {
        *expected = Some(runtime);
    }
    Ok(())
}

fn validate_range(
    range: BootstrapResourceByteRange,
    available_bytes: u64,
    label: &'static str,
) -> Result<(), DataGenerationError> {
    if range.end()? > available_bytes {
        return Err(DataGenerationError::Invalid(label));
    }
    Ok(())
}

/// Exact cold V1 bytes for the two-row engine-wrapper fixture. The source layout is sealed by
/// `bootstrap_publication`; this helper merely supplies the detached payload owner it describes.
#[cfg(test)]
fn v1_single_table_int4_gpu_wrapper_payload(all_valid_tail: bool) -> Box<[u8]> {
    let mut bytes = Vec::with_capacity(60);
    for row_id in [2_u64, 5] {
        bytes.extend_from_slice(&row_id.to_le_bytes());
    }
    bytes.extend_from_slice(&if all_valid_tail { 0b11_u32 } else { 0b01_u32 }.to_le_bytes());
    for value in [11_i32, if all_valid_tail { 17 } else { 0 }] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for _ in 0..2 {
        bytes.extend_from_slice(&1_u64.to_le_bytes());
    }
    for _ in 0..2 {
        bytes.extend_from_slice(&3_u64.to_le_bytes());
    }
    assert_eq!(bytes.len(), 60, "sealed two-row V1 test payload length");
    bytes.into_boxed_slice()
}

/// Narrow test-only attached source for actual engine-wrapper CUDA success. It derives the
/// attachment target from the retained execution target, keeps every owner cold, and changes no
/// production construction path or resident-owner surface.
#[cfg(test)]
pub(super) fn attached_v1_single_table_int4_resources_for_gpu_wrapper_test(
    target: &RuntimeGenerationRebuildTarget,
    all_valid_tail: bool,
) -> BootstrapAttachedUninstalledResources {
    let access = BootstrapResourceLeaseAccess { _private: () };
    let lease = super::bootstrap_publication::tests::validated_v1_single_table_int4_materialization_lease_for_gpu_wrapper(&access)
        .expect("valid V1 single-table INT4 GPU-wrapper source");
    let payload = v1_single_table_int4_gpu_wrapper_payload(all_valid_tail);
    let attachments = lease
        .resource_claims(&access)
        .iter()
        .cloned()
        .map(|claim| {
            let owner_bytes = claim
                .byte_offset
                .checked_add(claim.byte_len.get())
                .and_then(|value| usize::try_from(value).ok())
                .expect("test RAM owner length");
            let mut bytes = vec![0_u8; owner_bytes];
            if claim.kind == BootstrapResourceLedgerKind::TablePayload {
                let start = usize::try_from(claim.byte_offset).expect("test payload offset");
                let end = start.checked_add(payload.len()).expect("test payload end");
                assert_eq!(end, bytes.len(), "sealed table payload owner extent");
                bytes[start..end].copy_from_slice(&payload);
            }
            BootstrapResourceAttachment {
                owner: BootstrapDetachedResourceOwner::ColdRam(BootstrapDetachedColdRamOwner {
                    bytes: Arc::from(bytes),
                    range: BootstrapResourceByteRange {
                        offset: claim.byte_offset,
                        len: claim.byte_len,
                    },
                }),
                claim,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    attach_bootstrap_resource_bundle(
        lease,
        BootstrapResourceAttachmentBundle {
            target: BootstrapRebuildTarget {
                device_ordinal: target.device_ordinal(),
                context_identity: NonZeroUsize::new(target.context_identity())
                    .expect("primary CUDA context identity"),
            },
            attachments,
        },
    )
    .unwrap_or_else(|_| panic!("exact V1 cold GPU-wrapper owners attach"))
}

/// Narrow test-only source for the rebuild-proof child. Production code has no constructor for
/// attached owners other than the sealed lease/attachment handoff above.
#[cfg(test)]
pub(super) fn attached_resources_for_bootstrap_rebuild_test(
) -> BootstrapAttachedUninstalledResources {
    let access = BootstrapResourceLeaseAccess { _private: () };
    let lease =
        super::bootstrap_publication::tests::validated_materialization_lease_for_resources(&access)
            .expect("valid synthetic bootstrap source");
    let attachments = lease
        .resource_claims(&access)
        .iter()
        .cloned()
        .map(|claim| {
            let owner_bytes = claim
                .byte_offset
                .checked_add(claim.byte_len.get())
                .and_then(|value| usize::try_from(value).ok())
                .expect("test RAM owner length");
            BootstrapResourceAttachment {
                owner: BootstrapDetachedResourceOwner::ColdRam(BootstrapDetachedColdRamOwner {
                    bytes: Arc::from(vec![0_u8; owner_bytes]),
                    range: BootstrapResourceByteRange {
                        offset: claim.byte_offset,
                        len: claim.byte_len,
                    },
                }),
                claim,
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    match attach_bootstrap_resource_bundle(
        lease,
        BootstrapResourceAttachmentBundle {
            target: BootstrapRebuildTarget {
                device_ordinal: 0,
                context_identity: NonZeroUsize::new(1).expect("test context identity"),
            },
            attachments,
        },
    ) {
        Ok(attached) => attached,
        Err(_) => panic!("exact synthetic detached owners attach"),
    }
}

/// Test-only retention witness for the complete attached-owner handoff. The weak reference is
/// intentionally the only non-owning observer: a prepared build must keep the RAM owner alive,
/// then release it when the build is dropped.
#[cfg(test)]
pub(super) fn attached_resources_with_owner_witness_for_bootstrap_rebuild_test(
) -> (BootstrapAttachedUninstalledResources, std::sync::Weak<[u8]>) {
    let attached = attached_resources_for_bootstrap_rebuild_test();
    let Some(BootstrapResourceAttachment {
        owner: BootstrapDetachedResourceOwner::ColdRam(owner),
        ..
    }) = attached._attachments.first()
    else {
        panic!("synthetic bootstrap source begins with a detached RAM owner");
    };
    let owner = Arc::downgrade(&owner.bytes);
    (attached, owner)
}

#[cfg(test)]
mod tests {
    use std::{
        num::{NonZeroU64, NonZeroUsize},
        sync::Arc,
    };

    use gpu_db_execution::CudaDriverRuntime;

    use super::*;
    use crate::engine_data_generation::{
        bootstrap_publication::tests::{
            validated_materialization_lease_for_resources,
            validated_materialization_lease_with_resident_payloads_for_resources,
        },
        bootstrap_publication::{
            BootstrapPhysicalLayoutDescriptor, BootstrapPhysicalLayoutDescriptorId,
            BootstrapPhysicalLayoutRole, BootstrapPhysicalLayoutRoleKind,
            BootstrapPhysicalStorageType,
        },
        digest::{DatabaseId, RootFormatVersion, StableColumnId, StableIndexId, StableTableId},
    };

    fn table_id(value: u64) -> StableTableId {
        StableTableId::new(value).expect("test table ID")
    }

    fn index_id(value: u64) -> StableIndexId {
        StableIndexId::new(value).expect("test index ID")
    }

    fn source_database_id() -> DatabaseId {
        DatabaseId::new([0x42; 16]).expect("test database ID")
    }

    fn nonzero_len(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).expect("test nonempty range")
    }

    fn layout_descriptor_id(value: u64) -> BootstrapPhysicalLayoutDescriptorId {
        BootstrapPhysicalLayoutDescriptorId::new(value).expect("test layout descriptor")
    }

    fn test_layout(value: u64) -> BootstrapPhysicalLayoutDescriptor {
        BootstrapPhysicalLayoutDescriptor {
            id: layout_descriptor_id(value),
            encoding_version: 1,
            row_start: 0,
            row_count: 0,
            roles: vec![BootstrapPhysicalLayoutRole {
                kind: BootstrapPhysicalLayoutRoleKind::Opaque,
                ordinal: 0,
                column_id: None,
                sql_type: None,
                storage_type: BootstrapPhysicalStorageType::Bytes,
                key_ordinal: None,
                key_shape_root: None,
                byte_offset: 0,
                byte_len: 16,
                byte_stride: nonzero_len(1),
            }]
            .into_boxed_slice(),
        }
    }

    fn database_claim() -> BootstrapResourceLedgerEntry {
        BootstrapResourceLedgerEntry {
            kind: BootstrapResourceLedgerKind::DatabaseManifest,
            owner: BootstrapResourceLedgerOwner::Database,
            resource_id: 1,
            ordinal: 0,
            member_count: 1,
            tier: BootstrapResourceStorageTier::DetachedRam,
            byte_offset: 4,
            byte_len: nonzero_len(16),
            layout: test_layout(1),
            database_id: source_database_id(),
            root_format: RootFormatVersion::V1,
            covered_through: 2,
        }
    }

    fn table_claim(ordinal: u32) -> BootstrapResourceLedgerEntry {
        BootstrapResourceLedgerEntry {
            kind: BootstrapResourceLedgerKind::TablePayload,
            owner: BootstrapResourceLedgerOwner::Table(table_id(7)),
            resource_id: 2 + u64::from(ordinal),
            ordinal,
            member_count: 1,
            tier: BootstrapResourceStorageTier::DetachedRam,
            byte_offset: 4,
            byte_len: nonzero_len(16),
            layout: test_layout(3 + u64::from(ordinal)),
            database_id: source_database_id(),
            root_format: RootFormatVersion::V1,
            covered_through: 2,
        }
    }

    fn status_claim() -> BootstrapResourceLedgerEntry {
        BootstrapResourceLedgerEntry {
            kind: BootstrapResourceLedgerKind::StatusView,
            owner: BootstrapResourceLedgerOwner::Status,
            resource_id: 2,
            ordinal: 0,
            member_count: 1,
            tier: BootstrapResourceStorageTier::DetachedRam,
            byte_offset: 4,
            byte_len: nonzero_len(16),
            layout: test_layout(2),
            database_id: source_database_id(),
            root_format: RootFormatVersion::V1,
            covered_through: 2,
        }
    }

    fn index_claim() -> BootstrapResourceLedgerEntry {
        BootstrapResourceLedgerEntry {
            kind: BootstrapResourceLedgerKind::IndexPayload,
            owner: BootstrapResourceLedgerOwner::Index {
                table_id: table_id(7),
                index_id: index_id(11),
            },
            resource_id: 9,
            ordinal: 0,
            member_count: 1,
            tier: BootstrapResourceStorageTier::DetachedRam,
            byte_offset: 4,
            byte_len: nonzero_len(16),
            layout: test_layout(9),
            database_id: source_database_id(),
            root_format: RootFormatVersion::V1,
            covered_through: 2,
        }
    }

    fn record(
        claim: BootstrapResourceLedgerEntry,
        allocation_identity: usize,
    ) -> BootstrapResourceAllocationRecord {
        let byte_offset = claim.byte_offset;
        let byte_len = claim.byte_len.get();
        let allocated_bytes = byte_offset
            .checked_add(byte_len)
            .expect("test allocation extent");
        BootstrapResourceAllocationRecord {
            claim,
            allocation_identity: BootstrapAllocationIdentity(
                NonZeroUsize::new(allocation_identity).expect("test allocation identity"),
            ),
            proof: BootstrapAllocationProof {
                device_ordinal: 0,
                context_identity: NonZeroUsize::new(1).expect("test context identity"),
                allocated_bytes,
            },
            byte_offset,
            byte_len,
        }
    }

    fn records_for_lease(
        lease: &BootstrapMaterializationLease,
    ) -> Vec<BootstrapResourceAllocationRecord> {
        let access = BootstrapResourceLeaseAccess { _private: () };
        lease
            .resource_claims(&access)
            .iter()
            .cloned()
            .enumerate()
            .map(|(position, claim)| record(claim, position + 1))
            .collect()
    }

    fn byte_range(offset: u64, len: u64) -> BootstrapResourceByteRange {
        BootstrapResourceByteRange {
            offset,
            len: NonZeroU64::new(len).expect("test resource range length"),
        }
    }

    fn target(device_ordinal: u16, context_identity: usize) -> BootstrapRebuildTarget {
        BootstrapRebuildTarget {
            device_ordinal,
            context_identity: NonZeroUsize::new(context_identity).expect("test target context"),
        }
    }

    fn cold_target() -> BootstrapRebuildTarget {
        target(0, 1)
    }

    fn attachment_bundle(
        target: BootstrapRebuildTarget,
        attachments: Vec<BootstrapResourceAttachment>,
    ) -> BootstrapResourceAttachmentBundle {
        BootstrapResourceAttachmentBundle {
            target,
            attachments: attachments.into_boxed_slice(),
        }
    }

    fn attachment_error(
        result: Result<
            BootstrapAttachedUninstalledResources,
            Box<BootstrapResourceAttachmentFailure>,
        >,
    ) -> DataGenerationError {
        match result {
            Ok(_) => panic!("attachment unexpectedly succeeded"),
            Err(failure) => failure.error,
        }
    }

    fn ram_attachment(claim: BootstrapResourceLedgerEntry) -> BootstrapResourceAttachment {
        let allocated_bytes = claim
            .byte_offset
            .checked_add(claim.byte_len.get())
            .expect("test RAM owner extent");
        let range = BootstrapResourceByteRange {
            offset: claim.byte_offset,
            len: claim.byte_len,
        };
        BootstrapResourceAttachment {
            claim,
            owner: BootstrapDetachedResourceOwner::ColdRam(BootstrapDetachedColdRamOwner {
                bytes: Arc::from(vec![
                    0xA5;
                    usize::try_from(allocated_bytes)
                        .expect("test RAM owner size")
                ]),
                range,
            }),
        }
    }

    fn resident_attachment(
        claim: BootstrapResourceLedgerEntry,
        memory: Arc<CudaResidentDeviceMemory>,
    ) -> BootstrapResourceAttachment {
        let range = BootstrapResourceByteRange {
            offset: claim.byte_offset,
            len: claim.byte_len,
        };
        BootstrapResourceAttachment {
            claim,
            owner: BootstrapDetachedResourceOwner::Resident(BootstrapDetachedResidentOwner {
                memory,
                range,
            }),
        }
    }

    fn ram_bundle_for_lease(
        lease: &BootstrapMaterializationLease,
    ) -> BootstrapResourceAttachmentBundle {
        let access = BootstrapResourceLeaseAccess { _private: () };
        BootstrapResourceAttachmentBundle {
            target: cold_target(),
            attachments: lease
                .resource_claims(&access)
                .iter()
                .cloned()
                .map(ram_attachment)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn source_lease() -> BootstrapMaterializationLease {
        let access = BootstrapResourceLeaseAccess { _private: () };
        validated_materialization_lease_for_resources(&access)
            .expect("valid synthetic bootstrap source")
    }

    fn resident_payload_source_lease() -> BootstrapMaterializationLease {
        let access = BootstrapResourceLeaseAccess { _private: () };
        validated_materialization_lease_with_resident_payloads_for_resources(&access)
            .expect("valid synthetic bootstrap source with resident payloads")
    }

    #[test]
    fn resources_sibling_cannot_format_or_exfiltrate_materialization_lease() {
        // This is a compiler-enforced negative trait assertion. If a future change makes the
        // sibling-visible lease implement `Debug`, both implementations become applicable and
        // inference below fails as ambiguous. That prevents `format!("{lease:?}")` from walking
        // the private candidate/replay witness and its retained canonical-envelope bytes.
        trait AmbiguousIfDebug<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfDebug<()> for T {}
        impl<T: ?Sized + std::fmt::Debug> AmbiguousIfDebug<u8> for T {}

        let _ = <BootstrapMaterializationLease as AmbiguousIfDebug<_>>::marker;

        trait AmbiguousIfAttachedDebug<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfAttachedDebug<()> for T {}
        impl<T: ?Sized + std::fmt::Debug> AmbiguousIfAttachedDebug<u8> for T {}

        trait AmbiguousIfAttachedClone<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfAttachedClone<()> for T {}
        impl<T: Clone> AmbiguousIfAttachedClone<u8> for T {}

        let _ = <BootstrapAttachedUninstalledResources as AmbiguousIfAttachedDebug<_>>::marker;
        let _ = <BootstrapAttachedUninstalledResources as AmbiguousIfAttachedClone<_>>::marker;

        trait AmbiguousIfFailureDebug<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfFailureDebug<()> for T {}
        impl<T: ?Sized + std::fmt::Debug> AmbiguousIfFailureDebug<u8> for T {}

        trait AmbiguousIfFailureClone<Marker> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfFailureClone<()> for T {}
        impl<T: Clone> AmbiguousIfFailureClone<u8> for T {}

        let _ = <BootstrapResourceAttachmentFailure as AmbiguousIfFailureDebug<_>>::marker;
        let _ = <BootstrapResourceAttachmentFailure as AmbiguousIfFailureClone<_>>::marker;
    }

    #[test]
    fn resource_ledger_accepts_exact_canonical_allocation_records() {
        let claims = [
            database_claim(),
            status_claim(),
            table_claim(0),
            index_claim(),
        ];
        let records = claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(position, claim)| record(claim, position + 1))
            .collect::<Vec<_>>();
        assert_eq!(validate_resource_ledger(&claims, &records), Ok(()));
    }

    #[test]
    fn sealed_bootstrap_lease_carries_its_full_ordered_ledger_into_validation() {
        let access = BootstrapResourceLeaseAccess { _private: () };
        let lease = validated_materialization_lease_for_resources(&access)
            .expect("valid synthetic bootstrap source");
        let mut records = records_for_lease(&lease);
        let claims = records
            .iter()
            .map(|record| record.claim.clone())
            .collect::<Vec<_>>();
        assert_eq!(claims[0], database_claim());
        assert_eq!(claims[1], status_claim());
        assert_eq!(
            (
                claims[2].kind,
                claims[2].owner,
                claims[2].resource_id,
                claims[2].ordinal,
                claims[2].member_count,
                claims[2].tier,
                claims[2].byte_offset,
                claims[2].byte_len,
                claims[2].database_id,
                claims[2].root_format,
                claims[2].covered_through,
            ),
            (
                BootstrapResourceLedgerKind::TablePayload,
                BootstrapResourceLedgerOwner::Table(table_id(10)),
                3,
                0,
                1,
                BootstrapResourceStorageTier::DetachedRam,
                4,
                nonzero_len(32),
                source_database_id(),
                RootFormatVersion::V1,
                2,
            )
        );
        assert_eq!(claims[2].layout.id, layout_descriptor_id(3));
        assert_eq!(claims[2].layout.encoding_version, 1);
        assert_eq!(claims[2].layout.row_start, 0);
        assert_eq!(claims[2].layout.row_count, 1);
        assert_eq!(claims[2].layout.roles.len(), 5);
        assert!(matches!(
            &claims[2].layout.roles[..],
            [
                BootstrapPhysicalLayoutRole { kind: BootstrapPhysicalLayoutRoleKind::StableRowId, ordinal: 0, column_id: None, sql_type: None, storage_type: BootstrapPhysicalStorageType::Int8, key_ordinal: None, key_shape_root: None, byte_offset: 0, byte_len: stable_row_id_len, byte_stride: stable_row_id_stride },
                BootstrapPhysicalLayoutRole { kind: BootstrapPhysicalLayoutRoleKind::Validity, ordinal: 1, column_id: Some(column), sql_type: Some(gpu_db_sql::SqlType::Int4), storage_type: BootstrapPhysicalStorageType::Bit, key_ordinal: None, key_shape_root: None, byte_offset: 8, byte_len, byte_stride },
                BootstrapPhysicalLayoutRole { kind: BootstrapPhysicalLayoutRoleKind::Value, ordinal: 2, column_id: Some(value_column), sql_type: Some(gpu_db_sql::SqlType::Int4), storage_type: BootstrapPhysicalStorageType::Int4, key_ordinal: None, key_shape_root: None, byte_offset: 12, byte_len: value_len, byte_stride: value_stride },
                BootstrapPhysicalLayoutRole { kind: BootstrapPhysicalLayoutRoleKind::CreatedBy, ordinal: 3, column_id: None, sql_type: None, storage_type: BootstrapPhysicalStorageType::Int8, key_ordinal: None, key_shape_root: None, byte_offset: 16, byte_len: created_len, byte_stride: created_stride },
                BootstrapPhysicalLayoutRole { kind: BootstrapPhysicalLayoutRoleKind::DeletedBy, ordinal: 4, column_id: None, sql_type: None, storage_type: BootstrapPhysicalStorageType::Int8, key_ordinal: None, key_shape_root: None, byte_offset: 24, byte_len: deleted_len, byte_stride: deleted_stride },
            ] if *column == StableColumnId::new(7).expect("stable column ID")
                && *value_column == *column
                && *stable_row_id_len == 8 && *stable_row_id_stride == nonzero_len(8)
                && *byte_len == 4 && *byte_stride == nonzero_len(1)
                && *value_len == 4 && *value_stride == nonzero_len(4)
                && *created_len == 8 && *created_stride == nonzero_len(8)
                && *deleted_len == 8 && *deleted_stride == nonzero_len(8)
        ));
        assert_eq!(
            (
                claims[3].kind,
                claims[3].owner,
                claims[3].resource_id,
                claims[3].ordinal,
                claims[3].member_count,
                claims[3].tier,
                claims[3].byte_offset,
                claims[3].byte_len,
                claims[3].database_id,
                claims[3].root_format,
                claims[3].covered_through,
            ),
            (
                BootstrapResourceLedgerKind::IndexPayload,
                BootstrapResourceLedgerOwner::Index {
                    table_id: table_id(10),
                    index_id: index_id(20),
                },
                4,
                0,
                1,
                BootstrapResourceStorageTier::DetachedRam,
                4,
                nonzero_len(4),
                source_database_id(),
                RootFormatVersion::V1,
                2,
            )
        );
        assert_eq!(claims[3].layout.id, layout_descriptor_id(4));
        assert!(matches!(
            &claims[3].layout.roles[..],
            [BootstrapPhysicalLayoutRole { kind: BootstrapPhysicalLayoutRoleKind::IndexKey, ordinal: 0, column_id: Some(column), sql_type: Some(gpu_db_sql::SqlType::Int4), storage_type: BootstrapPhysicalStorageType::Int4, key_ordinal: Some(0), key_shape_root: Some(_), byte_offset: 0, byte_len, byte_stride }]
                if *column == StableColumnId::new(7).expect("stable column ID")
                    && *byte_len == 4 && *byte_stride == nonzero_len(4)
        ));
        assert_eq!(
            validate_bootstrap_materialization_lease(&lease, &records),
            Ok(())
        );

        records[2].claim.resource_id = 99;
        assert_eq!(
            validate_bootstrap_materialization_lease(&lease, &records),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );

        records = records_for_lease(&lease);
        records[2].claim.covered_through = 3;
        assert_eq!(
            validate_bootstrap_materialization_lease(&lease, &records),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );

        records = records_for_lease(&lease);
        records.swap(2, 3);
        assert_eq!(
            validate_bootstrap_materialization_lease(&lease, &records),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );

        records = records_for_lease(&lease);
        records[3].claim.owner = BootstrapResourceLedgerOwner::Table(table_id(10));
        assert_eq!(
            validate_bootstrap_materialization_lease(&lease, &records),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );
    }

    #[test]
    fn real_source_lease_attaches_exact_detached_ram_owners() {
        let lease = source_lease();
        let bundle = ram_bundle_for_lease(&lease);
        let attached = match attach_bootstrap_resource_bundle(lease, bundle) {
            Ok(attached) => attached,
            Err(_) => panic!("exact detached RAM owners attach"),
        };
        assert_eq!(attached._attachments.len(), 4);
        assert!(attached._attachments.iter().all(|attachment| {
            matches!(
                &attachment.owner,
                BootstrapDetachedResourceOwner::ColdRam(_)
            )
        }));
    }

    #[test]
    fn attached_bundle_rejects_missing_unexpected_reordered_and_foreign_claims() {
        let lease = source_lease();
        let mut missing = ram_bundle_for_lease(&lease).attachments.into_vec();
        missing.pop();
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), missing),
            )),
            DataGenerationError::Missing("bootstrap resource attachment coverage")
        );

        let lease = source_lease();
        let mut unexpected = ram_bundle_for_lease(&lease).attachments.into_vec();
        unexpected.push(ram_attachment(unexpected[0].claim.clone()));
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), unexpected),
            )),
            DataGenerationError::Unexpected("bootstrap resource attachment")
        );

        let lease = source_lease();
        let mut reordered = ram_bundle_for_lease(&lease).attachments.into_vec();
        reordered.swap(2, 3);
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), reordered),
            )),
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment claim")
        );

        let lease = source_lease();
        let mut foreign = ram_bundle_for_lease(&lease).attachments.into_vec();
        foreign[2].claim.database_id =
            DatabaseId::new([0x43; 16]).expect("foreign test database ID");
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), foreign),
            )),
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment claim")
        );

        let lease = source_lease();
        let mut wrong_member_count = ram_bundle_for_lease(&lease).attachments.into_vec();
        wrong_member_count[2].claim.member_count = 2;
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), wrong_member_count),
            )),
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment claim")
        );

        let lease = source_lease();
        let mut wrong_tier = ram_bundle_for_lease(&lease).attachments.into_vec();
        wrong_tier[2].claim.tier = BootstrapResourceStorageTier::Resident;
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), wrong_tier),
            )),
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment claim")
        );

        let lease = source_lease();
        let mut wrong_layout = ram_bundle_for_lease(&lease).attachments.into_vec();
        wrong_layout[2].claim.layout.id = layout_descriptor_id(99);
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), wrong_layout),
            )),
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment claim")
        );
    }

    #[test]
    fn attached_bundle_rejects_foreign_owner_bad_range_and_runtime_context() {
        let lease = source_lease();
        let mut foreign_owner = ram_bundle_for_lease(&lease).attachments.into_vec();
        foreign_owner[2].claim.owner = BootstrapResourceLedgerOwner::Index {
            table_id: table_id(10),
            index_id: index_id(20),
        };
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), foreign_owner),
            )),
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment claim")
        );

        let lease = source_lease();
        let mut bad_range = ram_bundle_for_lease(&lease).attachments.into_vec();
        let BootstrapDetachedResourceOwner::ColdRam(owner) = &mut bad_range[1].owner else {
            panic!("expected detached RAM owner");
        };
        owner.range = byte_range(31, 2);
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), bad_range),
            )),
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment range")
        );

        let lease = source_lease();
        let mut short_ram = ram_bundle_for_lease(&lease).attachments.into_vec();
        let BootstrapDetachedResourceOwner::ColdRam(owner) = &mut short_ram[1].owner else {
            panic!("expected detached RAM owner");
        };
        owner.bytes = Arc::from(vec![0xA5; 19]);
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(cold_target(), short_ram),
            )),
            DataGenerationError::Invalid("bootstrap detached RAM range")
        );

        let mut runtime = Some(BootstrapResidentRuntime {
            device_ordinal: 0,
            context_identity: NonZeroUsize::new(1).expect("first test context"),
        });
        assert_eq!(
            validate_resident_runtime(
                BootstrapResidentRuntime {
                    device_ordinal: 1,
                    context_identity: NonZeroUsize::new(2).expect("foreign test context"),
                },
                cold_target(),
                &mut runtime,
            ),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resident target"
            ))
        );
    }

    #[test]
    fn failed_attach_keeps_the_consumed_lease_and_bundle_for_exact_retry() {
        let lease = source_lease();
        let mut attachments = ram_bundle_for_lease(&lease).attachments.into_vec();
        let BootstrapDetachedResourceOwner::ColdRam(owner) = &mut attachments[0].owner else {
            panic!("expected detached RAM owner");
        };
        owner.range = byte_range(5, 16);
        let failure = match attach_bootstrap_resource_bundle(
            lease,
            attachment_bundle(cold_target(), attachments),
        ) {
            Ok(_) => panic!("invalid attachment unexpectedly accepted"),
            Err(failure) => *failure,
        };
        assert_eq!(
            failure.error,
            DataGenerationError::PredecessorMismatch("bootstrap resource attachment range")
        );
        let mut repaired = failure.bundle;
        repaired.attachments[0] = ram_attachment(repaired.attachments[0].claim.clone());
        assert!(attach_bootstrap_resource_bundle(failure.lease, repaired).is_ok());
    }

    #[test]
    fn cuda_attached_resources_retain_owner_and_reject_duplicate_allocation() {
        let Ok(runtime) = CudaDriverRuntime::probe() else {
            return;
        };
        if runtime.snapshot().device_count == 0 {
            return;
        }
        let Ok(resident_table) = runtime.retain_device_memory_copy(0, &[0x5A; 32]) else {
            return;
        };
        let Ok(resident_index) = runtime.retain_device_memory_copy(0, &[0xA5; 32]) else {
            return;
        };
        let resident_table = Arc::new(resident_table);
        let resident_index = Arc::new(resident_index);
        let target = BootstrapRebuildTarget {
            device_ordinal: resident_table.metadata().gpu_id,
            context_identity: NonZeroUsize::new(resident_table.context() as usize)
                .expect("resident primary context"),
        };

        let sparse_bytes = [0x11_u8; 16];
        let Ok(sparse) = runtime.retain_device_memory_chunks(
            0,
            32,
            &[
                gpu_db_execution::CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &sparse_bytes,
                },
                gpu_db_execution::CudaDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: &sparse_bytes,
                },
            ],
        ) else {
            return;
        };
        assert_eq!(
            sparse.metadata().copied_bytes,
            sparse.metadata().allocated_bytes
        );
        assert!(!sparse.has_full_contiguous_initialization());
        assert_eq!(
            validate_detached_resident_owner(&BootstrapDetachedResidentOwner {
                memory: Arc::new(sparse),
                range: byte_range(0, 32),
            }),
            Err(DataGenerationError::Invalid(
                "bootstrap resident full contiguous initialization"
            ))
        );

        let lease = resident_payload_source_lease();
        let mut duplicate = ram_bundle_for_lease(&lease).attachments.into_vec();
        duplicate[2] = resident_attachment(duplicate[2].claim.clone(), Arc::clone(&resident_table));
        duplicate[3] = resident_attachment(duplicate[3].claim.clone(), Arc::clone(&resident_table));
        assert_eq!(
            attachment_error(attach_bootstrap_resource_bundle(
                lease,
                attachment_bundle(target, duplicate),
            )),
            DataGenerationError::Invalid("duplicate bootstrap resident allocation identity")
        );

        let lease = resident_payload_source_lease();
        let mut exact = ram_bundle_for_lease(&lease).attachments.into_vec();
        exact[2] = resident_attachment(exact[2].claim.clone(), Arc::clone(&resident_table));
        exact[3] = resident_attachment(exact[3].claim.clone(), Arc::clone(&resident_index));
        let attached =
            match attach_bootstrap_resource_bundle(lease, attachment_bundle(target, exact)) {
                Ok(attached) => attached,
                Err(_) => panic!("exact resident owners attach"),
            };
        let allocation = Arc::downgrade(&resident_table);
        drop(resident_table);
        assert!(
            allocation.upgrade().is_some(),
            "attached owner lost resident allocation"
        );
        drop(attached);
        assert!(
            allocation.upgrade().is_none(),
            "resident allocation outlived attached owner"
        );
    }

    #[test]
    fn resource_ledger_rejects_source_mix_and_wrong_owner() {
        let claims = [database_claim(), table_claim(0)];

        let mut source_mix = vec![record(database_claim(), 1), record(table_claim(0), 2)];
        source_mix[1].claim.resource_id = 99;
        assert_eq!(
            validate_resource_ledger(&claims, &source_mix),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );

        let mut wrong_owner = vec![record(database_claim(), 1), record(table_claim(0), 2)];
        wrong_owner[1].claim.owner = BootstrapResourceLedgerOwner::Index {
            table_id: table_id(7),
            index_id: index_id(11),
        };
        assert_eq!(
            validate_resource_ledger(&claims, &wrong_owner),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );
    }

    #[test]
    fn resource_ledger_rejects_shared_allocation_duplicate_missing_and_unexpected_records() {
        let claims = [database_claim(), table_claim(0)];
        let duplicate = vec![record(database_claim(), 1), record(table_claim(0), 1)];
        assert_eq!(
            validate_resource_ledger(&claims, &duplicate),
            Err(DataGenerationError::Invalid(
                "duplicate bootstrap resource allocation identity"
            ))
        );

        let duplicate_claim = [record(database_claim(), 1), record(database_claim(), 2)];
        assert_eq!(
            validate_resource_ledger(&claims, &duplicate_claim),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );

        let missing = [record(database_claim(), 1)];
        assert_eq!(
            validate_resource_ledger(&claims, &missing),
            Err(DataGenerationError::Missing(
                "bootstrap resource allocation coverage"
            ))
        );

        let unexpected = [
            record(database_claim(), 1),
            record(table_claim(0), 2),
            record(index_claim(), 3),
        ];
        assert_eq!(
            validate_resource_ledger(&claims, &unexpected),
            Err(DataGenerationError::Unexpected(
                "bootstrap resource allocation record"
            ))
        );
    }

    #[test]
    fn resource_ledger_rejects_wrong_ordinal_and_invalid_span() {
        let claims = [database_claim(), table_claim(0)];
        let mut wrong_ordinal = vec![record(database_claim(), 1), record(table_claim(0), 2)];
        wrong_ordinal[1].claim.ordinal = 1;
        assert_eq!(
            validate_resource_ledger(&claims, &wrong_ordinal),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation claim"
            ))
        );

        let mut span = [record(database_claim(), 1), record(table_claim(0), 2)];
        span[1].byte_len = 29;
        assert_eq!(
            validate_resource_ledger(&claims, &span),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource allocation range"
            ))
        );
    }

    #[test]
    fn resource_module_is_detached_and_has_no_live_or_public_attach_surface() {
        let source = include_str!("resources.rs");
        for forbidden in [
            ["Arc", "Swap"].concat(),
            ["Read", "State"].concat(),
            ["Cold", "TableChunks"].concat(),
            ["device", "_ptr"].concat(),
            ["pub fn ", "attach"].concat(),
            ["pub(crate) fn ", "attach"].concat(),
            ["fn ", "install"].concat(),
            ["fn ", "publish"].concat(),
            ["fn ", "materialize"].concat(),
        ] {
            assert!(
                !source.contains(&forbidden),
                "resource boundary unexpectedly exposes {forbidden}"
            );
        }
        let forbidden_spill_owner = ["BootstrapDetachedCold", "SpillOwner"].concat();
        assert!(
            source.contains("BootstrapMaterializationLease")
                && source.contains("BootstrapAttachedUninstalledResources")
                && source.contains("BootstrapDetachedResidentOwner")
                && source.contains("BootstrapDetachedColdRamOwner")
                && source.contains("BootstrapRebuildTarget")
                && !source.contains(&forbidden_spill_owner),
            "resource boundary must retain only detached typed owners"
        );
    }
}
