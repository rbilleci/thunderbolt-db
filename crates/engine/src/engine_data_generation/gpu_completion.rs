//! Sealed GPU SHA-256 completion handoff for the fixed v1 genesis root layout.
//!
//! This module is intentionally private and inert.  It owns the only production conversion from
//! an execution-layer opaque completion token into the concrete roots accepted by the logical
//! generation foundation; it neither installs nor exposes a publication generation.

use gpu_db_execution::{
    CudaDriverRuntime, CudaRuntimeProbeError, CudaSha256BatchId, CudaSha256Completion,
    CudaSha256CompletionPrepareError, CudaSha256DeviceBuffer, CudaSha256UnknownQuiescence,
    OpaqueCudaSha256Batch, PreparedCudaSha256Completion,
    CUDA_RUNTIME_GENERATION_V1_GENESIS_ROOT_SLOTS,
};

use super::digest::{
    DatabaseId, DatabaseRoot, GpuCompletedDigest, RootFormatVersion, StatusViewRoot, TableMapRoot,
};
use super::manifest::GpuRadixEmptyRoots;
use super::publication::GenesisGpuCompletion;
use super::DataGenerationError;

const EMPTY_ROOT_COUNT: usize = 65;
const ROOT_COMPLETION_SLOT_COUNT: usize = EMPTY_ROOT_COUNT * 2 + 1;

const _: () = assert!(ROOT_COMPLETION_SLOT_COUNT == CUDA_RUNTIME_GENERATION_V1_GENESIS_ROOT_SLOTS);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticRootKind {
    TableMapEmpty,
    StatusMapEmpty,
    DatabaseRoot,
}

/// One fixed semantic output slot. This type carries no caller-provided text, generic type code,
/// child digest, or host-built preimage: the closed v1 device kernel owns those exact bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RootCompletionDescriptor {
    root_format: RootFormatVersion,
    database_id: DatabaseId,
    semantic_kind: SemanticRootKind,
    slot_ordinal: u32,
    radix_depth: Option<u8>,
}

/// The one immutable ordered layout this slice accepts: table-map empty roots depth 0..64,
/// status-map empty roots depth 0..64, then the database root.
#[derive(Clone, Debug)]
pub(super) struct RootCompletionLayoutV1 {
    batch_id: CudaSha256BatchId,
    database_id: DatabaseId,
    slots: Box<[RootCompletionDescriptor]>,
}

impl RootCompletionLayoutV1 {
    pub(super) fn new(
        database_id: DatabaseId,
        batch_id: CudaSha256BatchId,
    ) -> Result<Self, RootHandoffError> {
        let mut slots = Vec::with_capacity(ROOT_COMPLETION_SLOT_COUNT);
        for depth in 0..EMPTY_ROOT_COUNT {
            slots.push(RootCompletionDescriptor {
                root_format: RootFormatVersion::V1,
                database_id,
                semantic_kind: SemanticRootKind::TableMapEmpty,
                slot_ordinal: u32::try_from(slots.len())
                    .map_err(|_| RootHandoffError::Layout("root slot ordinal"))?,
                radix_depth: Some(depth as u8),
            });
        }
        for depth in 0..EMPTY_ROOT_COUNT {
            slots.push(RootCompletionDescriptor {
                root_format: RootFormatVersion::V1,
                database_id,
                semantic_kind: SemanticRootKind::StatusMapEmpty,
                slot_ordinal: u32::try_from(slots.len())
                    .map_err(|_| RootHandoffError::Layout("root slot ordinal"))?,
                radix_depth: Some(depth as u8),
            });
        }
        slots.push(RootCompletionDescriptor {
            root_format: RootFormatVersion::V1,
            database_id,
            semantic_kind: SemanticRootKind::DatabaseRoot,
            slot_ordinal: u32::try_from(slots.len())
                .map_err(|_| RootHandoffError::Layout("root slot ordinal"))?,
            radix_depth: None,
        });
        let result = Self {
            batch_id,
            database_id,
            slots: slots.into_boxed_slice(),
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<(), RootHandoffError> {
        if self.slots.len() != ROOT_COMPLETION_SLOT_COUNT {
            return Err(RootHandoffError::Layout("root completion slot count"));
        }
        for (ordinal, descriptor) in self.slots.iter().enumerate() {
            if descriptor.root_format != RootFormatVersion::V1
                || descriptor.database_id != self.database_id
                || descriptor.slot_ordinal
                    != u32::try_from(ordinal)
                        .map_err(|_| RootHandoffError::Layout("root slot ordinal"))?
            {
                return Err(RootHandoffError::Layout("root completion descriptor"));
            }
            let expected = if ordinal < EMPTY_ROOT_COUNT {
                (SemanticRootKind::TableMapEmpty, Some(ordinal as u8))
            } else if ordinal < EMPTY_ROOT_COUNT * 2 {
                (
                    SemanticRootKind::StatusMapEmpty,
                    Some((ordinal - EMPTY_ROOT_COUNT) as u8),
                )
            } else {
                (SemanticRootKind::DatabaseRoot, None)
            };
            if descriptor.semantic_kind != expected.0 || descriptor.radix_depth != expected.1 {
                return Err(RootHandoffError::Layout("root completion layout order"));
            }
        }
        Ok(())
    }

    /// Stage only the database identity for the closed device-resident v1 genesis chain. The CUDA
    /// program itself builds every canonical preimage bottom-up, including each child root and
    /// status subtree count; this host never receives or constructs any digest/preimage bytes.
    pub(super) fn prepare_genesis(
        self,
        runtime: &CudaDriverRuntime,
        gpu_id: u16,
    ) -> Result<PendingRootHandoff, RootHandoffError> {
        self.validate()?;
        let source = runtime.retain_device_memory_copy(gpu_id, &self.database_id.bytes())?;
        let prepared = PreparedCudaSha256Completion::prepare_runtime_generation_v1_genesis(
            CudaSha256DeviceBuffer::new(&source, 0, 16),
            self.batch_id,
        )?;
        Ok(PendingRootHandoff {
            layout: self,
            state: PendingRootHandoffState::Prepared(prepared),
        })
    }
}

enum PendingRootHandoffState {
    Prepared(PreparedCudaSha256Completion),
    Unknown(CudaSha256UnknownQuiescence),
}

/// A sealed in-flight root completion.  It can only retry a failed fence; it cannot acquire a
/// second submission or install a live publication.
#[must_use = "a pending root handoff must be completed, retried, or deliberately parked"]
pub(super) struct PendingRootHandoff {
    layout: RootCompletionLayoutV1,
    state: PendingRootHandoffState,
}

#[allow(
    clippy::large_enum_variant,
    reason = "boxing an unknown-quiescence owner would allocate after its CUDA work was enqueued"
)]
pub(super) enum RootHandoffCompletion {
    Verified(VerifiedRootHandoff),
    Failed(RootHandoffError),
    UnknownQuiescence(PendingRootHandoff),
}

impl PendingRootHandoff {
    pub(super) fn complete(self) -> RootHandoffCompletion {
        let completion = match self.state {
            PendingRootHandoffState::Prepared(prepared) => prepared.enqueue().complete(),
            PendingRootHandoffState::Unknown(unknown) => unknown.retry_complete(),
        };
        match completion {
            CudaSha256Completion::Quiesced(Ok(batch)) => match verify_batch(self.layout, batch) {
                Ok(verified) => RootHandoffCompletion::Verified(verified),
                Err(error) => RootHandoffCompletion::Failed(error),
            },
            CudaSha256Completion::Quiesced(Err(error)) => {
                RootHandoffCompletion::Failed(error.into())
            }
            CudaSha256Completion::UnknownQuiescence(unknown) => {
                RootHandoffCompletion::UnknownQuiescence(Self {
                    layout: self.layout,
                    state: PendingRootHandoffState::Unknown(unknown),
                })
            }
        }
    }
}

/// A fully checked, one-time-consumed root batch.  Its only materializer is the concrete genesis
/// completion below; no generic typed-root relabeling API exists.
pub(super) struct VerifiedRootHandoff {
    table_empty_roots: GpuRadixEmptyRoots<TableMapRoot>,
    status_empty_roots: GpuRadixEmptyRoots<StatusViewRoot>,
    database_root: DatabaseRoot,
}

impl VerifiedRootHandoff {
    pub(super) fn into_genesis_completion(self) -> GenesisGpuCompletion {
        GenesisGpuCompletion {
            empty_table_map_roots: self.table_empty_roots,
            empty_status_map_roots: self.status_empty_roots,
            database_root: self.database_root,
        }
    }
}

#[derive(Debug)]
pub(super) enum RootHandoffError {
    Layout(&'static str),
    Cuda(CudaRuntimeProbeError),
    Prepare(CudaSha256CompletionPrepareError),
    Data(DataGenerationError),
}

impl From<CudaRuntimeProbeError> for RootHandoffError {
    fn from(value: CudaRuntimeProbeError) -> Self {
        Self::Cuda(value)
    }
}

impl From<CudaSha256CompletionPrepareError> for RootHandoffError {
    fn from(value: CudaSha256CompletionPrepareError) -> Self {
        Self::Prepare(value)
    }
}

impl From<DataGenerationError> for RootHandoffError {
    fn from(value: DataGenerationError) -> Self {
        Self::Data(value)
    }
}

fn verify_batch(
    layout: RootCompletionLayoutV1,
    batch: OpaqueCudaSha256Batch,
) -> Result<VerifiedRootHandoff, RootHandoffError> {
    layout.validate()?;
    batch.consume(|batch_id, tokens| {
        if batch_id != layout.batch_id || tokens.len() != layout.slots.len() {
            return Err(RootHandoffError::Layout("root completion batch identity"));
        }
        let mut table_roots = Vec::with_capacity(EMPTY_ROOT_COUNT);
        let mut status_roots = Vec::with_capacity(EMPTY_ROOT_COUNT);
        let mut database_root = None;
        for (expected, token) in layout.slots.iter().zip(tokens) {
            let transport = token.descriptor();
            if transport.batch_id() != layout.batch_id
                || transport.slot_ordinal() != expected.slot_ordinal
            {
                return Err(RootHandoffError::Layout("root completion slot identity"));
            }
            let completion = GpuCompletedDigest::from_cuda_completion(token.opaque_digest());
            match (expected.semantic_kind, expected.radix_depth) {
                (SemanticRootKind::TableMapEmpty, Some(depth)) => {
                    table_roots.push((depth, TableMapRoot::from_gpu_completion(completion)));
                }
                (SemanticRootKind::StatusMapEmpty, Some(depth)) => {
                    status_roots.push((depth, StatusViewRoot::from_gpu_completion(completion)));
                }
                (SemanticRootKind::DatabaseRoot, None) => {
                    if database_root
                        .replace(DatabaseRoot::from_gpu_completion(completion))
                        .is_some()
                    {
                        return Err(RootHandoffError::Layout("duplicate database root"));
                    }
                }
                _ => return Err(RootHandoffError::Layout("root completion slot kind")),
            }
        }
        Ok(VerifiedRootHandoff {
            table_empty_roots: GpuRadixEmptyRoots::from_verified_depths(table_roots)?,
            status_empty_roots: GpuRadixEmptyRoots::from_verified_depths(status_roots)?,
            database_root: database_root
                .ok_or(RootHandoffError::Layout("missing database root"))?,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::super::digest::{
        synthetic_gpu_completion_for_test, CanonicalCatalogDigest, CatalogEpoch, CatalogIdentity,
        SyntheticRootForTest,
    };
    use super::super::publication::DataGenerationBuilder;
    use super::*;

    fn layout(batch: u64) -> RootCompletionLayoutV1 {
        RootCompletionLayoutV1::new(
            DatabaseId::new([7; 16]).expect("database id"),
            CudaSha256BatchId::new(batch).expect("batch id"),
        )
        .expect("layout")
    }

    fn runtime() -> Option<CudaDriverRuntime> {
        let runtime = CudaDriverRuntime::probe().ok()?;
        let snapshot = runtime.snapshot();
        (snapshot.driver_available && snapshot.device_count > 0).then_some(runtime)
    }

    fn completed_batch(
        runtime: &CudaDriverRuntime,
        layout: &RootCompletionLayoutV1,
    ) -> OpaqueCudaSha256Batch {
        let source = runtime
            .retain_device_memory_copy(0, &layout.database_id.bytes())
            .expect("database identity source");
        match PreparedCudaSha256Completion::prepare_runtime_generation_v1_genesis(
            CudaSha256DeviceBuffer::new(&source, 0, 16),
            layout.batch_id,
        )
        .expect("root batch preparation")
        .enqueue()
        .complete()
        {
            CudaSha256Completion::Quiesced(Ok(batch)) => batch,
            CudaSha256Completion::Quiesced(Err(error)) => panic!("root batch failed: {error}"),
            CudaSha256Completion::UnknownQuiescence(_) => panic!("root batch fence unknown"),
        }
    }

    #[test]
    fn layout_rejects_swapped_root_types_and_noncanonical_depth_sets() {
        let mut swapped = layout(41);
        swapped.slots[0].semantic_kind = SemanticRootKind::StatusMapEmpty;
        swapped.slots[EMPTY_ROOT_COUNT].semantic_kind = SemanticRootKind::TableMapEmpty;
        assert!(matches!(
            swapped.validate(),
            Err(RootHandoffError::Layout("root completion layout order"))
        ));

        let mut reversed = layout(42);
        reversed.slots.swap(0, 1);
        assert!(matches!(
            reversed.validate(),
            Err(RootHandoffError::Layout("root completion descriptor"))
        ));

        let mut duplicated = layout(43);
        duplicated.slots[1].radix_depth = Some(0);
        assert!(matches!(
            duplicated.validate(),
            Err(RootHandoffError::Layout("root completion layout order"))
        ));

        let mut missing = layout(44);
        missing.slots[64].radix_depth = Some(63);
        assert!(matches!(
            missing.validate(),
            Err(RootHandoffError::Layout("root completion layout order"))
        ));
    }

    #[test]
    fn layout_is_closed_over_format_database_kind_ordinal_and_depth() {
        let layout = layout(45);
        assert_eq!(layout.slots.len(), ROOT_COMPLETION_SLOT_COUNT);
        assert!(layout
            .slots
            .iter()
            .enumerate()
            .all(|(ordinal, slot)| slot.slot_ordinal == ordinal as u32));
        assert_eq!(
            layout.slots[0].semantic_kind,
            SemanticRootKind::TableMapEmpty
        );
        assert_eq!(layout.slots[0].radix_depth, Some(0));
        assert_eq!(layout.slots[64].radix_depth, Some(64));
        assert_eq!(
            layout.slots[EMPTY_ROOT_COUNT].semantic_kind,
            SemanticRootKind::StatusMapEmpty
        );
        assert_eq!(layout.slots[EMPTY_ROOT_COUNT].radix_depth, Some(0));
        assert_eq!(
            layout.slots[ROOT_COMPLETION_SLOT_COUNT - 1].semantic_kind,
            SemanticRootKind::DatabaseRoot
        );
        assert_eq!(
            layout.slots[ROOT_COMPLETION_SLOT_COUNT - 1].radix_depth,
            None
        );
    }

    #[test]
    fn stale_batch_cannot_be_substituted_into_another_layout() {
        let Some(runtime) = runtime() else {
            return;
        };
        let source_layout = layout(46);
        let stale = completed_batch(&runtime, &source_layout);
        assert!(matches!(
            verify_batch(layout(47), stale),
            Err(RootHandoffError::Layout("root completion batch identity"))
        ));
    }

    #[test]
    fn actual_cuda_fixed_layout_materializes_only_a_local_genesis_completion() {
        let Some(runtime) = runtime() else {
            return;
        };
        let database_id = DatabaseId::new([7; 16]).expect("database id");
        let builder =
            DataGenerationBuilder::new(RootFormatVersion::V1, database_id).expect("builder");
        let pending =
            RootCompletionLayoutV1::new(database_id, CudaSha256BatchId::new(48).expect("batch id"))
                .expect("layout")
                .prepare_genesis(&runtime, 0)
                .expect("root preparation");
        let verified = match pending.complete() {
            RootHandoffCompletion::Verified(verified) => verified,
            RootHandoffCompletion::Failed(error) => panic!("root completion failed: {error:?}"),
            RootHandoffCompletion::UnknownQuiescence(_) => panic!("root completion fence unknown"),
        };
        let catalog = CatalogIdentity::new(
            CatalogEpoch::new(1),
            <CanonicalCatalogDigest as SyntheticRootForTest>::from_synthetic(
                synthetic_gpu_completion_for_test(49),
            ),
        );
        let generation = builder
            .genesis(catalog, verified.into_genesis_completion())
            .expect("local genesis only");
        assert_eq!(generation.tables.count(), 0);
        assert_eq!(generation.status.count(), 0);
    }
}
