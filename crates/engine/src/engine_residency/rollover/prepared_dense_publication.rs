//! Two-phase dense-rollover publication capability.
//!
//! Dense payload bytes and row identities are already private device allocations. This owner
//! completes the uniform MVCC sidecar and resolves the exact header write before WAL, leaving
//! only one synchronous eight-byte publication after the common transaction claim.

use super::PendingDenseResidentShard;
use crate::{EngineError, ExecuteError, Index};
use gpu_db_execution::CudaOwnedDeviceMemoryChunk;

#[must_use]
pub(in crate::engine_residency) struct PreparedDenseResidentShardPublication {
    pending: PendingDenseResidentShard,
    header_publication: gpu_db_execution::PreparedU64HtoDPublication,
}

impl PendingDenseResidentShard {
    pub(in crate::engine_residency) fn prepare_uniform_commit_pre_wal(
        self,
        commit_seq: Index,
    ) -> Result<PreparedDenseResidentShardPublication, ExecuteError> {
        let row_count = self.payload.final_row_count();
        let rows = usize::try_from(row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed dense rollover row count does not fit host address space".to_string(),
            ))
        })?;
        let exact_stamp_bytes =
            rows.checked_mul(std::mem::size_of::<Index>())
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "sealed dense rollover stamp geometry overflowed".to_string(),
                    ))
                })?;
        if rows == 0
            || exact_stamp_bytes != self.created_by_bytes
            || u64::try_from(exact_stamp_bytes).ok() != Some(self.created_by_bytes_u64)
            || self.payload_bytes < std::mem::size_of::<u64>() as u64
            || self.device_memory.metadata().allocated_bytes < self.payload_bytes
            || self.created_by_region.metadata().allocated_bytes < self.created_by_bytes_u64
            || self
                .row_id_region
                .as_ref()
                .is_some_and(|region| region.metadata().allocated_bytes < self.created_by_bytes_u64)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed dense rollover drifted before pre-WAL uniform publication".to_string(),
            )));
        }

        let mut stamp_payload = Vec::with_capacity(exact_stamp_bytes);
        let stamp = commit_seq.to_le_bytes();
        for _ in 0..rows {
            stamp_payload.extend_from_slice(&stamp);
        }
        self.created_by_region
            .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                byte_offset: 0,
                bytes: stamp_payload,
            }))
            .map_err(super::device_write_error(
                "sealed dense pre-WAL created-by stamps",
            ))?;
        let header_publication = self
            .device_memory
            .prepare_u64_htod_publication(0, row_count)
            .map_err(super::device_write_error(
                "sealed dense pre-WAL final row-count header",
            ))?;
        Ok(PreparedDenseResidentShardPublication {
            pending: self,
            header_publication,
        })
    }
}

impl PreparedDenseResidentShardPublication {
    pub(in crate::engine_residency) fn pending(&self) -> &PendingDenseResidentShard {
        &self.pending
    }

    pub(in crate::engine_residency) fn publish_post_wal(
        self,
    ) -> Result<PendingDenseResidentShard, ExecuteError> {
        let Self {
            pending,
            header_publication,
        } = self;
        header_publication
            .publish()
            .map_err(super::device_write_error(
                "sealed dense prepared final row-count header",
            ))?;
        Ok(pending)
    }
}
