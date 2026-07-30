//! Checked, inert pre-WAL resource geometry.
//!
//! A current statement has the existing two-fragment format. Aggregate final overlays become
//! capacity-eligible only after the additive codec-5 layout fixes every chunk and STATUS2 byte;
//! neither shape owns an encoder, WAL append, apply, or publication capability.

#![allow(dead_code)] // Inert foundation; the next PLAN slice adopts these narrow constructors.

use crate::Index;

/// Stable physical target identity. Device addresses and allocation pointers are deliberately
/// absent: a target can be compared before its private replacement allocation exists.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct GpuTargetKey {
    pub(crate) gpu_id: u16,
    pub(crate) table_oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
}

/// The generation that a replacement target must still match when physical work is later bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuGenerationWitness {
    pub(crate) catalog_seq: Index,
    pub(crate) predecessor_boundary: Index,
    pub(crate) open_shard_id: u32,
    pub(crate) row_start: u64,
    pub(crate) row_count: u64,
    pub(crate) capacity: u64,
    pub(crate) index_mutation_epoch_even: u64,
}

/// One final-overlay GPU replacement contribution. A complete physical set is canonically
/// ordered by `target` and has exactly one contribution per target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalGpuFootprint {
    pub(super) target: GpuTargetKey,
    pub(super) generation: GpuGenerationWitness,
    pub(super) old_generation_pinned_bytes: u64,
    pub(super) new_persistent_bytes: u64,
    pub(super) plan_retained_transient_bytes: u64,
    pub(super) result_retained_device_bytes: u64,
    pub(super) allocation_slots: u64,
    /// Generation pins keep the old authoritative generation observable; they are slot evidence.
    pub(super) generation_pin_slots: u64,
    pub(super) scratch_peak_bytes: u64,
    /// Readback contributes to the serialized host scratch peak, never retained host storage.
    pub(super) max_host_readback_bytes: u64,
}
impl PreWalGpuFootprint {
    fn old_and_new_generation_bytes(&self) -> Result<u64, PreWalFootprintError> {
        checked_add(
            self.old_generation_pinned_bytes,
            self.new_persistent_bytes,
            "GPU old and new generation bytes",
        )
    }

    fn retained_device_bytes(&self) -> Result<u64, PreWalFootprintError> {
        self.old_and_new_generation_bytes()?
            .checked_add(self.plan_retained_transient_bytes)
            .and_then(|bytes| bytes.checked_add(self.result_retained_device_bytes))
            .ok_or(PreWalFootprintError::Overflow("GPU retained device bytes"))
    }

    fn device_peak_bytes(&self) -> Result<u64, PreWalFootprintError> {
        self.retained_device_bytes()?
            .checked_add(self.scratch_peak_bytes)
            .ok_or(PreWalFootprintError::Overflow("GPU device peak bytes"))
    }
}

/// One stable publication target. Explicit aggregation deduplicates this identity only in the
/// final physical set, never by taking a maximum over unrelated targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalPublicationTarget {
    pub(super) target: GpuTargetKey,
    pub(super) bytes: u64,
    pub(super) slots: u64,
}

/// Scalar-only bridge from one sealed indexed residency forecast into the pre-WAL accounting
/// owner.  It contains no pointer, CUDA capability, allocation owner, lock, or publication API.
///
/// Residency constructs this value from the same opaque preview that later consumes the sole
/// materialization permit.  The aggregate shape retains an exact copy so the terminal carrier
/// can reject a forecast/lease mismatch before allocating or launching private physical work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexedPreWalPhysicalForecast {
    pub(crate) target: GpuTargetKey,
    pub(crate) generation: GpuGenerationWitness,
    pub(crate) final_host_retained_bytes: u64,
    pub(crate) final_host_allocation_slots: u64,
    pub(crate) final_host_generation_pin_slots: u64,
    pub(crate) peak_host_retained_bytes: u64,
    pub(crate) peak_host_allocation_slots: u64,
    pub(crate) peak_host_generation_pin_slots: u64,
    pub(crate) old_generation_pinned_bytes: u64,
    pub(crate) new_persistent_bytes: u64,
    pub(crate) retained_device_transient_bytes: u64,
    pub(crate) retained_device_result_bytes: u64,
    pub(crate) incremental_allocation_slots: u64,
    pub(crate) generation_pin_slots: u64,
    pub(crate) maximum_concurrent_device_scratch_bytes: u64,
    pub(crate) maximum_host_readback_bytes: u64,
}

/// Exact coordinator-side target geometry paired with one indexed physical forecast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PreWalPublicationGeometry {
    pub(super) bytes: u64,
    pub(super) slots: u64,
}
/// Stable registration identity for one transaction owner. The nonce prevents a recycled
/// transaction id from accepting a contribution prepared for an earlier registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalTransactionOwnerIdentity {
    pub(super) txn_id: crate::TxnId,
    pub(super) registration_nonce: u64,
}

/// Exact private-overlay transition owned by one statement contribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalOverlayLink {
    pub(super) before_generation: u64,
    pub(super) after_generation: u64,
    pub(super) before_root_digest: gpu_db_wal::CanonicalDigest,
    pub(super) after_root_digest: gpu_db_wal::CanonicalDigest,
}

/// Ownership and ordering identity for one logical statement contribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalStatementIdentity {
    pub(super) owner: PreWalTransactionOwnerIdentity,
    pub(super) overlay_link: PreWalOverlayLink,
    pub(super) statement_ordinal: u32,
}

/// Logical requirements captured for one statement. These are the only fields that explicit
/// statement-time aggregation may combine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalLogicalFootprintInput {
    /// Semantic plan plus binary-template payload, offset, and operation-body allocations.
    pub(super) host_plan_retained_bytes: u64,
    /// Distinct host backing allocations across the semantic plan and typed shadow. This is
    /// separate from byte geometry because control/allocation pressure is not byte-proportional.
    pub(super) host_allocation_slots: u64,
    /// Canonical typed-v1 bytes are inert host evidence only. They never name or size a live WAL
    /// operation body; a later live carrier must obtain that payload from its binary owner.
    pub(super) typed_shadow_retained_bytes: u64,
    /// Arc-backed catalog/snapshot/generation ownership is charged as pins, never heap bytes.
    pub(super) host_generation_pin_slots: u64,
    /// Scratch is a phase-local peak: the future physical owner must serialize scratch/readback
    /// phases and release each temporary lease before advancing to the next target.
    pub(super) host_scratch_peak_bytes: u64,
    pub(super) statement_identity: PreWalStatementIdentity,
    pub(super) row_id_bytes: u64,
    pub(super) row_id_slots: u64,
    pub(super) sequence_effect_bytes: u64,
    pub(super) sequence_effect_slots: u64,
    /// Completed interactive response retained for one statement. Explicit aggregation takes the
    /// peak because each response is retired before the next statement starts.
    pub(super) terminal_response_bytes: u64,
    pub(super) terminal_response_slots: u64,
}
/// Target sets supplied once at the final overlay, never once per statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalFinalPhysicalFootprintInput {
    pub(super) gpus: Box<[PreWalGpuFootprint]>,
    pub(super) publication_targets: Box<[PreWalPublicationTarget]>,
}
/// Input for the current two-fragment shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreWalFootprintInput {
    pub(super) logical: PreWalLogicalFootprintInput,
    pub(super) final_physical: PreWalFinalPhysicalFootprintInput,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreWalLogicalFootprint {
    host_plan_retained_bytes: u64,
    host_allocation_slots: u64,
    typed_shadow_retained_bytes: u64,
    host_generation_pin_slots: u64,
    host_scratch_peak_bytes: u64,
    owner: PreWalTransactionOwnerIdentity,
    overlay_span: PreWalOverlayLink,
    first_statement_ordinal: u32,
    last_statement_ordinal: u32,
    row_id_bytes: u64,
    row_id_slots: u64,
    sequence_effect_bytes: u64,
    sequence_effect_slots: u64,
    terminal_response_bytes: u64,
    terminal_response_slots: u64,
}

/// Complete capacity geometry. It has no encoder, record carrier, or live-state entry point.
/// It is move-only so current capacity admission can transfer its sole demand to one lease.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct PreWalFootprint {
    pub(super) host_plan_retained_bytes: u64,
    pub(super) host_allocation_slots: u64,
    pub(super) typed_shadow_retained_bytes: u64,
    pub(super) host_generation_pin_slots: u64,
    pub(super) host_scratch_peak_bytes: u64,
    pub(super) wal_packed_record_bytes: u64,
    pub(super) wal_serialized_record_bytes: u64,
    pub(super) wal_record_slots: u64,
    pub(super) wal_frame_slots: u64,
    pub(super) row_id_bytes: u64,
    pub(super) row_id_slots: u64,
    pub(super) sequence_effect_bytes: u64,
    pub(super) sequence_effect_slots: u64,
    pub(super) status_index_bytes: u64,
    pub(super) status_index_slots: u64,
    pub(super) terminal_response_bytes: u64,
    pub(super) terminal_response_slots: u64,
    pub(super) publication_bytes: u64,
    pub(super) publication_target_slots: u64,
    pub(super) publication_join_entry_bytes: u64,
    pub(super) publication_join_entry_slots: u64,
    pub(super) completion_bytes: u64,
    pub(super) completion_slots: u64,
    pub(super) gpus: Box<[PreWalGpuFootprint]>,
    publication_targets: Box<[PreWalPublicationTarget]>,
}
/// One current, serial-shape statement. It is intentionally distinct from explicit aggregation.
#[must_use]
pub(super) struct CurrentTwoFragmentShape {
    footprint: PreWalFootprint,
    logical: PreWalLogicalFootprint,
    operation_fragment_body_bytes: u64,
}

/// Move-only logical contribution carries no GPU target, preventing per-statement replacement.
#[must_use]
pub(super) struct LogicalStatementContribution {
    logical: PreWalLogicalFootprint,
    operation_fragment_body_bytes: u64,
}

/// Logical requirements and operation evidence; no WAL/capacity until its format token exists.
#[must_use]
pub(super) struct RequiresAggregateFormatDecision {
    logical: PreWalLogicalFootprint,
    operation_fragment_body_bytes: Box<[u64]>,
}

/// Validated final overlay with unresolved aggregate format and no WAL/capacity accessor.
#[must_use]
pub(super) struct AggregateFinalOverlayFormatDecision {
    logical: PreWalLogicalFootprint,
    operation_fragment_body_bytes: Box<[u64]>,
    final_physical: PreWalFinalPhysicalFootprintInput,
    indexed_physical: Option<IndexedPreWalPhysicalForecast>,
}

/// One explicit typed-INSERT aggregate whose codec-5 layout and final physical set are both
/// fixed. It remains deliberately separate from the current two-fragment route.
#[must_use]
pub(super) struct AggregateTypedInsertShape {
    footprint: PreWalFootprint,
    logical: PreWalLogicalFootprint,
    layout: crate::typed_insert_aggregate::TypedInsertAggregateLayout,
    indexed_physical: Option<IndexedPreWalPhysicalForecast>,
}

/// Immutable aggregate identity retained across scalar capacity admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AggregateTypedInsertAdmissionBinding {
    owner: PreWalTransactionOwnerIdentity,
    overlay_span: PreWalOverlayLink,
    first_statement_ordinal: u32,
    last_statement_ordinal: u32,
    layout: crate::typed_insert_aggregate::TypedInsertAggregateLayout,
    indexed_physical: Option<IndexedPreWalPhysicalForecast>,
}

/// Per-device totals derived only after the target-keyed final set is validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PreWalGpuPoolFootprint {
    pub(super) gpu_id: u16,
    pub(super) old_generation_pinned_bytes: u64,
    pub(super) new_persistent_bytes: u64,
    pub(super) plan_retained_transient_bytes: u64,
    pub(super) result_retained_device_bytes: u64,
    pub(super) allocation_slots: u64,
    /// Device-generation pins; old-generation bytes are envelope evidence, not incremental demand.
    pub(super) generation_pin_slots: u64,
    pub(super) scratch_peak_bytes: u64,
    /// Readback contributes to the serialized host scratch peak, never retained host storage.
    pub(super) max_host_readback_bytes: u64,
}

/// Allocation-free canonical enumeration of the final physical set grouped by GPU.  A footprint
/// has already established `GpuTargetKey` order, so equal GPU ids are contiguous and no map or
/// scratch owner is needed to derive pool admission totals.
pub(super) struct PreWalGpuPoolCursor<'a> {
    gpus: &'a [PreWalGpuFootprint],
    next: usize,
}

impl<'a> PreWalGpuPoolCursor<'a> {
    /// Derives one GPU's total with checked arithmetic. `CurrentTwoFragmentShape::new` runs
    /// this complete cursor before exposing the footprint, so admission can consume the
    /// validated sequence without a map, clone, or fallible mutation phase.
    pub(super) fn try_next(
        &mut self,
    ) -> Result<Option<PreWalGpuPoolFootprint>, PreWalFootprintError> {
        let Some(first) = self.gpus.get(self.next) else {
            return Ok(None);
        };
        let gpu_id = first.target.gpu_id;
        let mut total = PreWalGpuPoolFootprint {
            gpu_id,
            old_generation_pinned_bytes: 0,
            new_persistent_bytes: 0,
            plan_retained_transient_bytes: 0,
            result_retained_device_bytes: 0,
            allocation_slots: 0,
            generation_pin_slots: 0,
            scratch_peak_bytes: 0,
            max_host_readback_bytes: 0,
        };
        while let Some(gpu) = self.gpus.get(self.next) {
            if gpu.target.gpu_id != gpu_id {
                break;
            }
            total.old_generation_pinned_bytes = checked_add(
                total.old_generation_pinned_bytes,
                gpu.old_generation_pinned_bytes,
                "GPU pool old-generation bytes",
            )?;
            total.new_persistent_bytes = checked_add(
                total.new_persistent_bytes,
                gpu.new_persistent_bytes,
                "GPU pool persistent bytes",
            )?;
            total.plan_retained_transient_bytes = checked_add(
                total.plan_retained_transient_bytes,
                gpu.plan_retained_transient_bytes,
                "GPU pool plan-transient bytes",
            )?;
            total.result_retained_device_bytes = checked_add(
                total.result_retained_device_bytes,
                gpu.result_retained_device_bytes,
                "GPU pool result bytes",
            )?;
            total.allocation_slots = checked_add(
                total.allocation_slots,
                gpu.allocation_slots,
                "GPU pool allocation slots",
            )?;
            total.generation_pin_slots = checked_add(
                total.generation_pin_slots,
                gpu.generation_pin_slots,
                "GPU pool generation-pin slots",
            )?;
            total.scratch_peak_bytes = total.scratch_peak_bytes.max(gpu.scratch_peak_bytes);
            total.max_host_readback_bytes = total
                .max_host_readback_bytes
                .max(gpu.max_host_readback_bytes);
            self.next += 1;
        }
        Ok(Some(total))
    }
}

impl PreWalGpuPoolFootprint {
    /// Incremental lease bytes; old authoritative residency is evidence, never a second charge.
    pub(super) fn incremental_device_bytes(&self) -> Result<u64, PreWalFootprintError> {
        self.new_persistent_bytes
            .checked_add(self.plan_retained_transient_bytes)
            .and_then(|bytes| bytes.checked_add(self.result_retained_device_bytes))
            .ok_or(PreWalFootprintError::Overflow(
                "GPU pool incremental device bytes",
            ))
    }

    /// Incremental peak for admission: new persistent plus retained plan/result state and the
    /// serialized scratch peak. See `incremental_device_bytes` for why old pinned bytes are out.
    pub(super) fn incremental_device_peak_bytes(&self) -> Result<u64, PreWalFootprintError> {
        self.incremental_device_bytes()?
            .checked_add(self.scratch_peak_bytes)
            .ok_or(PreWalFootprintError::Overflow(
                "GPU pool incremental device peak bytes",
            ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PreWalFootprintError {
    OperationFragmentBodyEmpty,
    OperationFragmentBodyTooLarge,
    ZeroTransactionOwner,
    ZeroTransactionRegistrationNonce,
    InvalidOverlayLink,
    OwnerIdentityDrift,
    OverlayLinkDrift,
    StatementOrderDrift,
    AggregateLayoutDrift,
    IndexedPhysicalForecastDrift,
    RowIdBytesMismatch { bytes: u64, slots: u64 },
    ByteSlotDomainMismatch(&'static str),
    InvalidGpuTarget(GpuTargetKey),
    InvalidGpuGenerationWitness(GpuTargetKey),
    GpuTargetOrderDrift(GpuTargetKey),
    DuplicateGpuTarget(GpuTargetKey),
    GpuTargetSchemaDrift(GpuTargetKey),
    GpuGenerationWitnessDrift(GpuTargetKey),
    EmptyGpuTargetSet,
    EmptyPublicationTargetSet,
    PublicationTargetOrderDrift(GpuTargetKey),
    DuplicatePublicationTarget(GpuTargetKey),
    PublicationTargetSchemaDrift(GpuTargetKey),
    EmptyPublicationTarget(GpuTargetKey),
    GpuTargetMissingPublication(GpuTargetKey),
    PublicationTargetMissingGpu(GpuTargetKey),
    GlobalGenerationCutDrift(GpuTargetKey),
    TableMutationEpochDrift(GpuTargetKey),
    Overflow(&'static str),
    Wal,
}

impl std::fmt::Display for PreWalFootprintError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OperationFragmentBodyEmpty => {
                formatter.write_str("canonical operation fragment body is empty")
            }
            Self::OperationFragmentBodyTooLarge => {
                formatter.write_str("canonical operation fragment body exceeds WAL limit")
            }
            Self::ZeroTransactionOwner => formatter.write_str("transaction owner id is zero"),
            Self::ZeroTransactionRegistrationNonce => {
                formatter.write_str("transaction registration nonce is zero")
            }
            Self::InvalidOverlayLink => {
                formatter.write_str("statement overlay link is not one exact generation transition")
            }
            Self::OwnerIdentityDrift => {
                formatter.write_str("transaction registration owner changed")
            }
            Self::OverlayLinkDrift => {
                formatter.write_str("statement overlay links are not contiguous")
            }
            Self::StatementOrderDrift => {
                formatter.write_str("statement ordinals are not contiguous")
            }
            Self::AggregateLayoutDrift => {
                formatter.write_str("typed INSERT aggregate layout drifted from logical evidence")
            }
            Self::IndexedPhysicalForecastDrift => {
                formatter.write_str("indexed physical forecast geometry is inconsistent")
            }
            Self::RowIdBytesMismatch { bytes, slots } => {
                write!(
                    formatter,
                    "row-id bytes {bytes} do not equal {slots} slots times eight"
                )
            }
            Self::ByteSlotDomainMismatch(domain) => {
                write!(formatter, "{domain} bytes and slots disagree")
            }
            Self::InvalidGpuTarget(target) => write!(
                formatter,
                "GPU target is invalid: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::InvalidGpuGenerationWitness(target) => write!(
                formatter,
                "GPU generation witness is invalid: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::GpuTargetOrderDrift(target) => write!(
                formatter,
                "GPU target is not canonically ordered: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::DuplicateGpuTarget(target) => write!(
                formatter,
                "GPU target appears more than once: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::GpuTargetSchemaDrift(target) => write!(
                formatter,
                "GPU target schema changed within one final overlay: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::GpuGenerationWitnessDrift(target) => write!(
                formatter,
                "GPU target generation changed: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::EmptyGpuTargetSet => formatter.write_str("final GPU target set is empty"),
            Self::EmptyPublicationTargetSet => {
                formatter.write_str("final publication target set is empty")
            }
            Self::PublicationTargetOrderDrift(target) => write!(
                formatter,
                "publication target is not canonically ordered: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::DuplicatePublicationTarget(target) => write!(
                formatter,
                "publication target appears more than once: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::PublicationTargetSchemaDrift(target) => write!(
                formatter,
                "publication target schema changed within one final overlay: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::EmptyPublicationTarget(target) => write!(
                formatter,
                "publication target has zero slots: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::GpuTargetMissingPublication(target) => write!(
                formatter,
                "GPU target has no publication target: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::PublicationTargetMissingGpu(target) => write!(
                formatter,
                "publication target has no GPU target: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::GlobalGenerationCutDrift(target) => write!(
                formatter,
                "GPU target changed the final-overlay catalog/predecessor cut: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::TableMutationEpochDrift(target) => write!(
                formatter,
                "GPU target changed its table mutation epoch across devices: gpu {} table {}",
                target.gpu_id, target.table_oid
            ),
            Self::Overflow(what) => write!(formatter, "pre-WAL footprint overflows {what}"),
            Self::Wal => formatter.write_str("canonical WAL framing rejects the footprint"),
        }
    }
}

impl std::error::Error for PreWalFootprintError {}
impl CurrentTwoFragmentShape {
    pub(super) fn new(
        input: PreWalFootprintInput,
        operation_fragment_body_bytes: u64,
    ) -> Result<Self, PreWalFootprintError> {
        let logical_statement =
            LogicalStatementContribution::new(input.logical, operation_fragment_body_bytes)?;
        let logical = logical_statement.logical.clone();
        let footprint = footprint_from_current_parts(
            logical.clone(),
            input.final_physical,
            operation_fragment_body_bytes,
        )?;
        let _ = footprint.global_host_retained_bytes()?;
        let mut gpu_pools = footprint.gpu_pool_cursor();
        while let Some(pool) = gpu_pools.try_next()? {
            let _ = pool.incremental_device_bytes()?;
            let _ = pool.incremental_device_peak_bytes()?;
        }
        Ok(Self {
            footprint,
            logical,
            operation_fragment_body_bytes,
        })
    }

    pub(super) fn footprint(&self) -> &PreWalFootprint {
        &self.footprint
    }

    pub(super) fn operation_fragment_body_bytes(&self) -> u64 {
        self.operation_fragment_body_bytes
    }

    pub(super) fn into_logical_statement(self) -> LogicalStatementContribution {
        LogicalStatementContribution {
            logical: self.logical,
            operation_fragment_body_bytes: self.operation_fragment_body_bytes,
        }
    }

    /// The exclusive current-statement exit. The pool receives the sole complete footprint;
    /// explicit transactions instead use `into_logical_statement` and cannot claim current WAL.
    pub(super) fn into_capacity_request(self) -> super::pre_wal_capacity::PreWalCapacityRequest {
        super::pre_wal_capacity::PreWalCapacityRequest::from_validated_footprint(self.footprint)
    }
}
impl LogicalStatementContribution {
    pub(super) fn new(
        input: PreWalLogicalFootprintInput,
        operation_fragment_body_bytes: u64,
    ) -> Result<Self, PreWalFootprintError> {
        validate_operation_fragment_body_bytes(operation_fragment_body_bytes)?;
        Ok(Self {
            logical: checked_logical(input)?,
            operation_fragment_body_bytes,
        })
    }

    /// Resolve a one-statement autocommit overlay through the same aggregate-format decision as
    /// an explicit transaction.  The operation-body byte count remains only logical statement
    /// evidence; codec-5 geometry is supplied exactly once by the later layout token.
    pub(super) fn bind_single_indexed_final_overlay(
        self,
        indexed_physical: IndexedPreWalPhysicalForecast,
        publication: PreWalPublicationGeometry,
    ) -> Result<AggregateFinalOverlayFormatDecision, PreWalFootprintError> {
        let Self {
            logical,
            operation_fragment_body_bytes,
        } = self;
        bind_indexed_final_overlay_parts(
            logical,
            vec![operation_fragment_body_bytes].into(),
            indexed_physical,
            publication,
        )
    }
}
impl RequiresAggregateFormatDecision {
    pub(super) fn from_two(
        first: LogicalStatementContribution,
        second: LogicalStatementContribution,
    ) -> Result<Self, PreWalFootprintError> {
        let operation_fragment_body_bytes = vec![
            first.operation_fragment_body_bytes,
            second.operation_fragment_body_bytes,
        ];
        let logical = aggregate_logical(&first.logical, &second.logical)?;
        Ok(Self {
            logical,
            operation_fragment_body_bytes: operation_fragment_body_bytes.into(),
        })
    }

    pub(super) fn extend(
        self,
        next: LogicalStatementContribution,
    ) -> Result<Self, PreWalFootprintError> {
        let logical = aggregate_logical(&self.logical, &next.logical)?;
        let mut operation_fragment_body_bytes = self.operation_fragment_body_bytes.into_vec();
        operation_fragment_body_bytes.push(next.operation_fragment_body_bytes);
        Ok(Self {
            logical,
            operation_fragment_body_bytes: operation_fragment_body_bytes.into(),
        })
    }

    pub(super) fn bind_final_overlay(
        self,
        final_physical: PreWalFinalPhysicalFootprintInput,
    ) -> Result<AggregateFinalOverlayFormatDecision, PreWalFootprintError> {
        Ok(AggregateFinalOverlayFormatDecision {
            logical: self.logical,
            operation_fragment_body_bytes: self.operation_fragment_body_bytes,
            final_physical: checked_final_physical(final_physical)?,
            indexed_physical: None,
        })
    }

    /// Bind one already-sealed final indexed generation to an explicit transaction aggregate.
    /// Host ownership and peak readback are incorporated here, rather than copied independently
    /// into capacity demand by a caller.
    pub(super) fn bind_indexed_final_overlay(
        self,
        indexed_physical: IndexedPreWalPhysicalForecast,
        publication: PreWalPublicationGeometry,
    ) -> Result<AggregateFinalOverlayFormatDecision, PreWalFootprintError> {
        bind_indexed_final_overlay_parts(
            self.logical,
            self.operation_fragment_body_bytes,
            indexed_physical,
            publication,
        )
    }

    pub(super) fn statement_count(&self) -> usize {
        self.operation_fragment_body_bytes.len()
    }
}
impl AggregateFinalOverlayFormatDecision {
    pub(super) fn statement_count(&self) -> usize {
        self.operation_fragment_body_bytes.len()
    }

    /// Resolve the formerly missing aggregate WAL decision. The layout token has already measured
    /// the exact eight-section stream, canonical chunks, STATUS2 fragment, and outer marker.
    pub(super) fn bind_typed_insert_aggregate_layout(
        self,
        layout: crate::typed_insert_aggregate::TypedInsertAggregateLayout,
    ) -> Result<AggregateTypedInsertShape, PreWalFootprintError> {
        let statement_count = u32::try_from(self.statement_count())
            .map_err(|_| PreWalFootprintError::Overflow("aggregate statement count"))?;
        if layout.measure.stable_transaction_id != self.logical.owner.txn_id
            || layout.measure.statement_count != statement_count
            || layout.measure.insert_statement_count != statement_count
            || layout.measure.original_inserted_row_count != self.logical.row_id_slots
            || layout
                .measure
                .allocator_high_water
                .checked_sub(layout.measure.allocator_before)
                != Some(self.logical.row_id_slots)
        {
            return Err(PreWalFootprintError::AggregateLayoutDrift);
        }
        let mut terminal_logical = self.logical.clone();
        // The terminal codec owner retains one exact boxed body for every chunk and STATUS2
        // while the packed and serialized outer buffers are filled. Typed statement records
        // remain separately charged in `typed_shadow_retained_bytes`.
        terminal_logical.host_plan_retained_bytes = terminal_logical
            .host_plan_retained_bytes
            .checked_add(layout.wal.fragment_body_bytes)
            .ok_or(PreWalFootprintError::Overflow(
                "aggregate fragment-body retained bytes",
            ))?;
        terminal_logical.host_allocation_slots = terminal_logical
            .host_allocation_slots
            .checked_add(u64::from(layout.fragment_count))
            .ok_or(PreWalFootprintError::Overflow(
                "aggregate fragment-body allocation slots",
            ))?;
        let footprint =
            footprint_from_validated_parts(terminal_logical, self.final_physical, layout.wal)?;
        Ok(AggregateTypedInsertShape {
            footprint,
            logical: self.logical,
            layout,
            indexed_physical: self.indexed_physical,
        })
    }
}

impl AggregateTypedInsertShape {
    pub(super) fn footprint(&self) -> &PreWalFootprint {
        &self.footprint
    }

    pub(super) fn layout(&self) -> &crate::typed_insert_aggregate::TypedInsertAggregateLayout {
        &self.layout
    }

    pub(super) fn admission_binding(&self) -> AggregateTypedInsertAdmissionBinding {
        AggregateTypedInsertAdmissionBinding {
            owner: self.logical.owner.clone(),
            overlay_span: self.logical.overlay_span.clone(),
            first_statement_ordinal: self.logical.first_statement_ordinal,
            last_statement_ordinal: self.logical.last_statement_ordinal,
            layout: self.layout,
            indexed_physical: self.indexed_physical.clone(),
        }
    }

    pub(super) fn into_capacity_request(self) -> super::pre_wal_capacity::PreWalCapacityRequest {
        let Self {
            footprint,
            logical: _,
            layout: _,
            indexed_physical: _,
        } = self;
        super::pre_wal_capacity::PreWalCapacityRequest::from_validated_footprint(footprint)
    }
}

impl AggregateTypedInsertAdmissionBinding {
    pub(super) fn owner(&self) -> &PreWalTransactionOwnerIdentity {
        &self.owner
    }

    pub(super) fn overlay_span(&self) -> &PreWalOverlayLink {
        &self.overlay_span
    }

    pub(super) fn statement_ordinals(&self) -> (u32, u32) {
        (self.first_statement_ordinal, self.last_statement_ordinal)
    }

    pub(super) fn layout(&self) -> &crate::typed_insert_aggregate::TypedInsertAggregateLayout {
        &self.layout
    }

    pub(super) fn indexed_physical(&self) -> Option<&IndexedPreWalPhysicalForecast> {
        self.indexed_physical.as_ref()
    }
}

fn bind_indexed_final_overlay_parts(
    mut logical: PreWalLogicalFootprint,
    operation_fragment_body_bytes: Box<[u64]>,
    indexed: IndexedPreWalPhysicalForecast,
    publication: PreWalPublicationGeometry,
) -> Result<AggregateFinalOverlayFormatDecision, PreWalFootprintError> {
    if indexed.peak_host_retained_bytes < indexed.final_host_retained_bytes
        || indexed.peak_host_allocation_slots < indexed.final_host_allocation_slots
        || indexed.peak_host_generation_pin_slots < indexed.final_host_generation_pin_slots
        || indexed.final_host_retained_bytes == 0
        || indexed.final_host_allocation_slots == 0
        || indexed.old_generation_pinned_bytes == 0
        || indexed.generation_pin_slots == 0
        || indexed.maximum_concurrent_device_scratch_bytes == 0
        || publication.slots == 0
    {
        return Err(PreWalFootprintError::IndexedPhysicalForecastDrift);
    }
    let materialization_host_scratch = indexed
        .peak_host_retained_bytes
        .checked_sub(indexed.final_host_retained_bytes)
        .ok_or(PreWalFootprintError::IndexedPhysicalForecastDrift)?
        .max(indexed.maximum_host_readback_bytes);
    logical.host_plan_retained_bytes = logical
        .host_plan_retained_bytes
        .checked_add(indexed.final_host_retained_bytes)
        .ok_or(PreWalFootprintError::Overflow(
            "indexed final host retained bytes",
        ))?;
    // Allocation/pin capacity has no separate scratch ledger. Charge the measured physical peak
    // so the private materializer cannot temporarily exceed the all-resource admission.
    logical.host_allocation_slots = logical
        .host_allocation_slots
        .checked_add(indexed.peak_host_allocation_slots)
        .ok_or(PreWalFootprintError::Overflow(
            "indexed peak host allocation slots",
        ))?;
    logical.host_generation_pin_slots = logical
        .host_generation_pin_slots
        .checked_add(indexed.peak_host_generation_pin_slots)
        .ok_or(PreWalFootprintError::Overflow(
            "indexed peak host generation pins",
        ))?;
    logical.host_scratch_peak_bytes = logical
        .host_scratch_peak_bytes
        .max(materialization_host_scratch);

    let target = indexed.target.clone();
    let final_physical = checked_final_physical(PreWalFinalPhysicalFootprintInput {
        gpus: vec![PreWalGpuFootprint {
            target: target.clone(),
            generation: indexed.generation.clone(),
            old_generation_pinned_bytes: indexed.old_generation_pinned_bytes,
            new_persistent_bytes: indexed.new_persistent_bytes,
            plan_retained_transient_bytes: indexed.retained_device_transient_bytes,
            result_retained_device_bytes: indexed.retained_device_result_bytes,
            allocation_slots: indexed.incremental_allocation_slots,
            generation_pin_slots: indexed.generation_pin_slots,
            scratch_peak_bytes: indexed.maximum_concurrent_device_scratch_bytes,
            max_host_readback_bytes: indexed.maximum_host_readback_bytes,
        }]
        .into(),
        publication_targets: vec![PreWalPublicationTarget {
            target,
            bytes: publication.bytes,
            slots: publication.slots,
        }]
        .into(),
    })?;
    Ok(AggregateFinalOverlayFormatDecision {
        logical,
        operation_fragment_body_bytes,
        final_physical,
        indexed_physical: Some(indexed),
    })
}

impl PreWalFootprint {
    pub(super) fn global_host_retained_bytes(&self) -> Result<u64, PreWalFootprintError> {
        self.host_plan_retained_bytes
            .checked_add(self.typed_shadow_retained_bytes)
            .and_then(|bytes| bytes.checked_add(self.row_id_bytes))
            .and_then(|bytes| bytes.checked_add(self.sequence_effect_bytes))
            .ok_or(PreWalFootprintError::Overflow("global host retained bytes"))
    }

    pub(super) fn gpu_pool_cursor(&self) -> PreWalGpuPoolCursor<'_> {
        PreWalGpuPoolCursor {
            gpus: &self.gpus,
            next: 0,
        }
    }
}
fn validate_operation_fragment_body_bytes(
    operation_fragment_body_bytes: u64,
) -> Result<(), PreWalFootprintError> {
    if operation_fragment_body_bytes == 0 {
        return Err(PreWalFootprintError::OperationFragmentBodyEmpty);
    }
    if operation_fragment_body_bytes > gpu_db_wal::canonical_fragment_body_limit() {
        return Err(PreWalFootprintError::OperationFragmentBodyTooLarge);
    }
    Ok(())
}
fn checked_logical(
    input: PreWalLogicalFootprintInput,
) -> Result<PreWalLogicalFootprint, PreWalFootprintError> {
    if input.statement_identity.owner.txn_id == 0 {
        return Err(PreWalFootprintError::ZeroTransactionOwner);
    }
    if input.statement_identity.owner.registration_nonce == 0 {
        return Err(PreWalFootprintError::ZeroTransactionRegistrationNonce);
    }
    let link = &input.statement_identity.overlay_link;
    if link.before_generation.checked_add(1) != Some(link.after_generation)
        || link.before_root_digest == [0; 32]
        || link.after_root_digest == [0; 32]
        || link.before_root_digest == link.after_root_digest
    {
        return Err(PreWalFootprintError::InvalidOverlayLink);
    }
    let expected_row_id_bytes = input
        .row_id_slots
        .checked_mul(8)
        .ok_or(PreWalFootprintError::Overflow("row-id bytes from slots"))?;
    if input.row_id_bytes != expected_row_id_bytes {
        return Err(PreWalFootprintError::RowIdBytesMismatch {
            bytes: input.row_id_bytes,
            slots: input.row_id_slots,
        });
    }
    for (domain, bytes, slots) in [
        (
            "sequence effect",
            input.sequence_effect_bytes,
            input.sequence_effect_slots,
        ),
        (
            "terminal response",
            input.terminal_response_bytes,
            input.terminal_response_slots,
        ),
    ] {
        if (bytes == 0) != (slots == 0) {
            return Err(PreWalFootprintError::ByteSlotDomainMismatch(domain));
        }
    }
    Ok(PreWalLogicalFootprint {
        host_plan_retained_bytes: input.host_plan_retained_bytes,
        host_allocation_slots: input.host_allocation_slots,
        typed_shadow_retained_bytes: input.typed_shadow_retained_bytes,
        host_generation_pin_slots: input.host_generation_pin_slots,
        host_scratch_peak_bytes: input.host_scratch_peak_bytes,
        owner: input.statement_identity.owner,
        overlay_span: input.statement_identity.overlay_link,
        first_statement_ordinal: input.statement_identity.statement_ordinal,
        last_statement_ordinal: input.statement_identity.statement_ordinal,
        row_id_bytes: input.row_id_bytes,
        row_id_slots: input.row_id_slots,
        sequence_effect_bytes: input.sequence_effect_bytes,
        sequence_effect_slots: input.sequence_effect_slots,
        terminal_response_bytes: input.terminal_response_bytes,
        terminal_response_slots: input.terminal_response_slots,
    })
}
fn footprint_from_current_parts(
    logical: PreWalLogicalFootprint,
    final_physical: PreWalFinalPhysicalFootprintInput,
    operation_fragment_body_bytes: u64,
) -> Result<PreWalFootprint, PreWalFootprintError> {
    validate_operation_fragment_body_bytes(operation_fragment_body_bytes)?;
    let wal = canonical_wal_geometry(operation_fragment_body_bytes)?;
    if wal.fragment_count != 2 || wal.frame_count != 3 {
        return Err(PreWalFootprintError::Wal);
    }
    footprint_from_validated_parts(logical, final_physical, wal)
}

fn footprint_from_validated_parts(
    logical: PreWalLogicalFootprint,
    final_physical: PreWalFinalPhysicalFootprintInput,
    wal: gpu_db_wal::CanonicalWalFootprint,
) -> Result<PreWalFootprint, PreWalFootprintError> {
    let final_physical = checked_final_physical(final_physical)?;
    let mut host_scratch_peak_bytes = logical.host_scratch_peak_bytes;
    for gpu in final_physical.gpus.iter() {
        host_scratch_peak_bytes = host_scratch_peak_bytes.max(gpu.max_host_readback_bytes);
    }
    Ok(PreWalFootprint {
        host_plan_retained_bytes: logical.host_plan_retained_bytes,
        host_allocation_slots: logical.host_allocation_slots,
        typed_shadow_retained_bytes: logical.typed_shadow_retained_bytes,
        host_generation_pin_slots: logical.host_generation_pin_slots,
        host_scratch_peak_bytes,
        wal_packed_record_bytes: wal.packed_record_bytes,
        wal_serialized_record_bytes: wal.serialized_record_bytes,
        wal_record_slots: 1,
        wal_frame_slots: u64::from(wal.frame_count),
        row_id_bytes: logical.row_id_bytes,
        row_id_slots: logical.row_id_slots,
        sequence_effect_bytes: logical.sequence_effect_bytes,
        sequence_effect_slots: logical.sequence_effect_slots,
        status_index_bytes: u64::try_from(crate::durable_transaction_status_index_entry_bytes())
            .map_err(|_| PreWalFootprintError::Overflow("status-index entry bytes"))?,
        status_index_slots: 1,
        terminal_response_bytes: logical.terminal_response_bytes,
        terminal_response_slots: logical.terminal_response_slots,
        publication_bytes: publication_bytes(&final_physical.publication_targets)?,
        publication_target_slots: publication_slots(&final_physical.publication_targets)?,
        publication_join_entry_bytes: u64::try_from(
            crate::engine_commit_coordinator::commit_publication_join_entry_payload_bytes(),
        )
        .map_err(|_| PreWalFootprintError::Overflow("publication-join entry bytes"))?,
        publication_join_entry_slots: 1,
        completion_bytes: u64::try_from(
            crate::engine_dml_concurrent::commit_wave_done_payload_bytes(),
        )
        .map_err(|_| PreWalFootprintError::Overflow("completion payload bytes"))?,
        completion_slots: 1,
        gpus: final_physical.gpus,
        publication_targets: final_physical.publication_targets,
    })
}
fn canonical_wal_geometry(
    operation_fragment_body_bytes: u64,
) -> Result<gpu_db_wal::CanonicalWalFootprint, PreWalFootprintError> {
    let bodies = [
        operation_fragment_body_bytes,
        u64::try_from(crate::engine_durability::canonical_transaction_claim_status_len())
            .map_err(|_| PreWalFootprintError::Overflow("transaction-status length"))?,
    ];
    gpu_db_wal::canonical_wal_footprint(&bodies).map_err(|_| PreWalFootprintError::Wal)
}
fn checked_final_physical(
    final_physical: PreWalFinalPhysicalFootprintInput,
) -> Result<PreWalFinalPhysicalFootprintInput, PreWalFootprintError> {
    if final_physical.gpus.is_empty() {
        return Err(PreWalFootprintError::EmptyGpuTargetSet);
    }
    if final_physical.publication_targets.is_empty() {
        return Err(PreWalFootprintError::EmptyPublicationTargetSet);
    }
    checked_sorted_gpus(&final_physical.gpus)?;
    checked_sorted_publications(&final_physical.publication_targets)?;
    let mut gpu_targets = final_physical.gpus.iter().map(|gpu| &gpu.target);
    let mut publication_targets = final_physical
        .publication_targets
        .iter()
        .map(|publication| &publication.target);
    loop {
        match (gpu_targets.next(), publication_targets.next()) {
            (None, None) => break,
            (Some(gpu), Some(publication)) if gpu == publication => {}
            (Some(gpu), Some(publication)) if gpu < publication => {
                return Err(PreWalFootprintError::GpuTargetMissingPublication(
                    gpu.clone(),
                ));
            }
            (Some(_), Some(publication)) => {
                return Err(PreWalFootprintError::PublicationTargetMissingGpu(
                    publication.clone(),
                ));
            }
            (Some(gpu), None) => {
                return Err(PreWalFootprintError::GpuTargetMissingPublication(
                    gpu.clone(),
                ));
            }
            (None, Some(publication)) => {
                return Err(PreWalFootprintError::PublicationTargetMissingGpu(
                    publication.clone(),
                ));
            }
        }
    }
    Ok(final_physical)
}

fn checked_sorted_gpus(gpus: &[PreWalGpuFootprint]) -> Result<(), PreWalFootprintError> {
    let mut prior: Option<&PreWalGpuFootprint> = None;
    let mut global_cut = None;
    for (position, gpu) in gpus.iter().enumerate() {
        if gpu.target.table_oid == 0 || gpu.target.schema_digest == [0; 32] {
            return Err(PreWalFootprintError::InvalidGpuTarget(gpu.target.clone()));
        }
        if gpu.generation.catalog_seq > gpu.generation.predecessor_boundary
            || gpu.generation.index_mutation_epoch_even & 1 != 0
            || gpu.generation.row_count > gpu.generation.capacity
            || gpu
                .generation
                .row_start
                .checked_add(gpu.generation.capacity)
                .is_none()
        {
            return Err(PreWalFootprintError::InvalidGpuGenerationWitness(
                gpu.target.clone(),
            ));
        }
        let cut = (
            gpu.generation.catalog_seq,
            gpu.generation.predecessor_boundary,
        );
        if global_cut
            .replace(cut)
            .is_some_and(|expected| expected != cut)
        {
            return Err(PreWalFootprintError::GlobalGenerationCutDrift(
                gpu.target.clone(),
            ));
        }
        // Canonical target order is GPU-major, while table witnesses span GPUs.  Compare this
        // bounded final set against preceding entries directly instead of allocating a map.
        for previous_table in &gpus[..position] {
            if previous_table.target.table_oid != gpu.target.table_oid {
                continue;
            }
            if previous_table.target.schema_digest != gpu.target.schema_digest {
                return Err(PreWalFootprintError::GpuTargetSchemaDrift(
                    gpu.target.clone(),
                ));
            }
            if previous_table.generation.index_mutation_epoch_even
                != gpu.generation.index_mutation_epoch_even
            {
                return Err(PreWalFootprintError::TableMutationEpochDrift(
                    gpu.target.clone(),
                ));
            }
        }
        let _ = gpu.retained_device_bytes()?;
        let _ = gpu.device_peak_bytes()?;
        if let Some(previous) = prior {
            if previous.target.gpu_id == gpu.target.gpu_id
                && previous.target.table_oid == gpu.target.table_oid
                && previous.target.schema_digest != gpu.target.schema_digest
            {
                return Err(PreWalFootprintError::GpuTargetSchemaDrift(
                    gpu.target.clone(),
                ));
            }
            match previous.target.cmp(&gpu.target) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Greater => {
                    return Err(PreWalFootprintError::GpuTargetOrderDrift(
                        gpu.target.clone(),
                    ));
                }
                std::cmp::Ordering::Equal if previous.generation != gpu.generation => {
                    return Err(PreWalFootprintError::GpuGenerationWitnessDrift(
                        gpu.target.clone(),
                    ));
                }
                std::cmp::Ordering::Equal => {
                    return Err(PreWalFootprintError::DuplicateGpuTarget(gpu.target.clone()));
                }
            }
        }
        prior = Some(gpu);
    }
    Ok(())
}

fn checked_sorted_publications(
    publication_targets: &[PreWalPublicationTarget],
) -> Result<(), PreWalFootprintError> {
    let mut prior: Option<&GpuTargetKey> = None;
    for (position, publication) in publication_targets.iter().enumerate() {
        if publication.target.table_oid == 0 || publication.target.schema_digest == [0; 32] {
            return Err(PreWalFootprintError::InvalidGpuTarget(
                publication.target.clone(),
            ));
        }
        if publication.slots == 0 {
            return Err(PreWalFootprintError::EmptyPublicationTarget(
                publication.target.clone(),
            ));
        }
        for previous_table in &publication_targets[..position] {
            if previous_table.target.table_oid == publication.target.table_oid
                && previous_table.target.schema_digest != publication.target.schema_digest
            {
                return Err(PreWalFootprintError::PublicationTargetSchemaDrift(
                    publication.target.clone(),
                ));
            }
        }
        if let Some(previous) = prior {
            if previous.gpu_id == publication.target.gpu_id
                && previous.table_oid == publication.target.table_oid
                && previous.schema_digest != publication.target.schema_digest
            {
                return Err(PreWalFootprintError::PublicationTargetSchemaDrift(
                    publication.target.clone(),
                ));
            }
            match previous.cmp(&publication.target) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Greater => {
                    return Err(PreWalFootprintError::PublicationTargetOrderDrift(
                        publication.target.clone(),
                    ));
                }
                std::cmp::Ordering::Equal => {
                    return Err(PreWalFootprintError::DuplicatePublicationTarget(
                        publication.target.clone(),
                    ));
                }
            }
        }
        prior = Some(&publication.target);
    }
    Ok(())
}

fn publication_slots(
    publication_targets: &[PreWalPublicationTarget],
) -> Result<u64, PreWalFootprintError> {
    publication_targets.iter().try_fold(0_u64, |total, target| {
        checked_add(total, target.slots, "publication slots")
    })
}

fn publication_bytes(
    publication_targets: &[PreWalPublicationTarget],
) -> Result<u64, PreWalFootprintError> {
    publication_targets.iter().try_fold(0_u64, |total, target| {
        checked_add(total, target.bytes, "publication bytes")
    })
}

fn aggregate_logical(
    first: &PreWalLogicalFootprint,
    second: &PreWalLogicalFootprint,
) -> Result<PreWalLogicalFootprint, PreWalFootprintError> {
    if first.owner != second.owner {
        return Err(PreWalFootprintError::OwnerIdentityDrift);
    }
    if first.overlay_span.after_generation != second.overlay_span.before_generation
        || first.overlay_span.after_root_digest != second.overlay_span.before_root_digest
    {
        return Err(PreWalFootprintError::OverlayLinkDrift);
    }
    if first.last_statement_ordinal.checked_add(1) != Some(second.first_statement_ordinal) {
        return Err(PreWalFootprintError::StatementOrderDrift);
    }
    Ok(PreWalLogicalFootprint {
        host_plan_retained_bytes: checked_add(
            first.host_plan_retained_bytes,
            second.host_plan_retained_bytes,
            "host-plan retained bytes",
        )?,
        host_allocation_slots: checked_add(
            first.host_allocation_slots,
            second.host_allocation_slots,
            "host allocation slots",
        )?,
        typed_shadow_retained_bytes: checked_add(
            first.typed_shadow_retained_bytes,
            second.typed_shadow_retained_bytes,
            "typed shadow retained bytes",
        )?,
        host_generation_pin_slots: checked_add(
            first.host_generation_pin_slots,
            second.host_generation_pin_slots,
            "host generation-pin slots",
        )?,
        host_scratch_peak_bytes: first
            .host_scratch_peak_bytes
            .max(second.host_scratch_peak_bytes),
        owner: first.owner.clone(),
        overlay_span: PreWalOverlayLink {
            before_generation: first.overlay_span.before_generation,
            after_generation: second.overlay_span.after_generation,
            before_root_digest: first.overlay_span.before_root_digest,
            after_root_digest: second.overlay_span.after_root_digest,
        },
        first_statement_ordinal: first.first_statement_ordinal,
        last_statement_ordinal: second.last_statement_ordinal,
        row_id_bytes: checked_add(first.row_id_bytes, second.row_id_bytes, "row-id bytes")?,
        row_id_slots: checked_add(first.row_id_slots, second.row_id_slots, "row-id slots")?,
        sequence_effect_bytes: checked_add(
            first.sequence_effect_bytes,
            second.sequence_effect_bytes,
            "sequence-effect bytes",
        )?,
        sequence_effect_slots: checked_add(
            first.sequence_effect_slots,
            second.sequence_effect_slots,
            "sequence-effect slots",
        )?,
        // Interactive statement responses are returned and retired before the next statement.
        // A predeclared multi-result owner must use a distinct sum-aggregation footprint.
        terminal_response_bytes: first
            .terminal_response_bytes
            .max(second.terminal_response_bytes),
        terminal_response_slots: first
            .terminal_response_slots
            .max(second.terminal_response_slots),
    })
}

fn checked_add(left: u64, right: u64, what: &'static str) -> Result<u64, PreWalFootprintError> {
    left.checked_add(right)
        .ok_or(PreWalFootprintError::Overflow(what))
}
#[cfg(test)]
#[path = "pre_wal_footprint_consistency_tests.rs"]
mod consistency_tests;
#[cfg(test)]
#[path = "pre_wal_footprint_tests.rs"]
mod tests;
