//! Opaque identities and roots for the runtime logical-generation authority.
//!
//! This module deliberately has no SHA implementation. Relational commitments arrive only as
//! completed GPU results; production code cannot manufacture a digest from host bytes here.

use std::fmt;
use std::num::NonZeroU64;

use gpu_db_execution::OpaqueCudaSha256Digest;

use super::DataGenerationError;

pub(super) const ROOT_FORMAT_V1: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct RootFormatVersion(u16);

impl RootFormatVersion {
    pub(super) const V1: Self = Self(ROOT_FORMAT_V1);

    pub(super) fn new(value: u16) -> Result<Self, DataGenerationError> {
        if value == ROOT_FORMAT_V1 {
            Ok(Self(value))
        } else {
            Err(DataGenerationError::UnsupportedRootFormat(value))
        }
    }

    pub(super) const fn get(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DatabaseId([u8; 16]);

impl DatabaseId {
    pub(super) fn new(bytes: [u8; 16]) -> Result<Self, DataGenerationError> {
        if bytes == [0; 16] {
            return Err(DataGenerationError::ZeroIdentity("database id"));
        }
        Ok(Self(bytes))
    }

    pub(super) const fn bytes(self) -> [u8; 16] {
        self.0
    }
}

macro_rules! stable_id {
    ($name:ident, $description:literal) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(super) struct $name(NonZeroU64);

        impl $name {
            pub(super) fn new(value: u64) -> Result<Self, DataGenerationError> {
                NonZeroU64::new(value)
                    .map(Self)
                    .ok_or(DataGenerationError::ZeroIdentity($description))
            }

            pub(super) const fn get(self) -> u64 {
                self.0.get()
            }
        }
    };
}

stable_id!(StableTableId, "stable table id");
stable_id!(StableIndexId, "stable index id");
stable_id!(StableRowId, "stable row id");
stable_id!(StableColumnId, "stable column id");
stable_id!(StableTransactionId, "stable transaction id");
stable_id!(CommitSequence, "commit sequence");
stable_id!(DataGeneration, "data generation");
stable_id!(IndexGeneration, "index generation");
stable_id!(PublicationEpoch, "publication epoch");
stable_id!(VisibleNext, "visible next");

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CatalogEpoch(u64);

impl CatalogEpoch {
    pub(super) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(super) const fn get(self) -> u64 {
        self.0
    }
}

/// A nonzero digest received from a completed device operation. Its bytes are never exposed to
/// the host-side generation builder, so that builder cannot use it as input to a host hash.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct GpuCompletedDigest(RootDigest);

impl GpuCompletedDigest {
    pub(super) fn from_cuda_completion(digest: OpaqueCudaSha256Digest) -> Self {
        Self(RootDigest::Cuda(digest))
    }

    const fn root(self) -> RootDigest {
        self.0
    }
}

impl fmt::Debug for GpuCompletedDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GpuCompletedDigest(<opaque>)")
    }
}

/// Bytes remain private to execution's opaque completion token. Every typed root wraps one of
/// these values; test-only synthetic roots use a separate representation that is compiled out of
/// production.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RootDigest {
    Cuda(OpaqueCudaSha256Digest),
    #[cfg(test)]
    Synthetic([u8; 32]),
}

impl RootDigest {
    #[cfg(test)]
    fn new(bytes: [u8; 32]) -> Result<Self, DataGenerationError> {
        if bytes == [0; 32] {
            return Err(DataGenerationError::ZeroDigest);
        }
        Ok(Self::Synthetic(bytes))
    }
}

impl fmt::Debug for RootDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RootDigest(<opaque>)")
    }
}

macro_rules! gpu_root {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(super) struct $name(RootDigest);

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(concat!(stringify!($name), "(<opaque>)"))
            }
        }

        #[cfg(test)]
        impl SyntheticRootForTest for $name {
            fn from_synthetic(digest: GpuCompletedDigest) -> Self {
                Self(digest.root())
            }
        }
    };
}

gpu_root!(CanonicalCatalogDigest);
gpu_root!(ColumnShapeRoot);
gpu_root!(TypedValueRoot);
gpu_root!(CurrentRowLeafRoot);
gpu_root!(IndexEntryLeafRoot);
gpu_root!(RowMapRoot);
gpu_root!(IndexMapRoot);
gpu_root!(TableMapRoot);
gpu_root!(IndexShapeRoot);
gpu_root!(IndexRoot);
gpu_root!(TableRoot);
gpu_root!(DatabaseRoot);
gpu_root!(StatusViewRoot);
gpu_root!(StatusEntryLeafRoot);
gpu_root!(RequestDigest);
gpu_root!(TargetDigest);
gpu_root!(ReturningDigest);
gpu_root!(TerminalEnvelopeDigest);

// Production conversion is deliberately exhaustive and private to the sealed root-completion
// handoff.  There is no generic production relabeling surface.
impl TableMapRoot {
    pub(super) fn from_gpu_completion(digest: GpuCompletedDigest) -> Self {
        Self(digest.root())
    }
}

impl StatusViewRoot {
    pub(super) fn from_gpu_completion(digest: GpuCompletedDigest) -> Self {
        Self(digest.root())
    }
}

impl DatabaseRoot {
    pub(super) fn from_gpu_completion(digest: GpuCompletedDigest) -> Self {
        Self(digest.root())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CatalogIdentity {
    pub(super) epoch: CatalogEpoch,
    pub(super) digest: CanonicalCatalogDigest,
}

impl CatalogIdentity {
    pub(super) const fn new(epoch: CatalogEpoch, digest: CanonicalCatalogDigest) -> Self {
        Self { epoch, digest }
    }
}

#[cfg(test)]
pub(super) trait SyntheticRootForTest: Copy + Eq {
    fn from_synthetic(digest: GpuCompletedDigest) -> Self;
}

#[cfg(test)]
pub(super) fn synthetic_gpu_completion_for_test(seed: u64) -> GpuCompletedDigest {
    // This is intentionally not a hash and is compiled out of production. It models only the
    // opaque completion handoff that the execution SHA-256 task will provide.
    let mut bytes = [0_u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    bytes[8..16].copy_from_slice(&seed.rotate_left(17).to_le_bytes());
    bytes[31] = 0xa5;
    GpuCompletedDigest(RootDigest::new(bytes).expect("synthetic nonzero digest"))
}
