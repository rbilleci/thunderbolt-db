//! Two-phase fixed-rollover publication capability.
//!
//! The parent rollover owner reserves immutable device state. This child owns the final pre-WAL
//! uniform-created-by upload and the only post-WAL publication capability, so the committed path
//! stays a single already-resolved synchronous header write.

use super::PendingFixedResidentShard;
use crate::engine_insert_plan::host_retention::HostRetentionGeometry;
use crate::{EngineError, ExecuteError, Index};
use gpu_db_execution::CudaOwnedDeviceMemoryChunk;

/// Move-only fixed-rollover capability crossing WAL after every mutable sidecar byte has landed.
///
/// `pending` still owns the entirely private generation. The only state left to apply after WAL
/// is the already-resolved, synchronous eight-byte count publication held by
/// `header_publication`; it cannot allocate, encode, rebuild a destination, or resolve CUDA
/// state on the committed path.
#[must_use]
#[allow(dead_code)] // test-only foundation until the live fixed-rollover carrier adopts it
pub(in crate::engine_residency) struct PreparedFixedResidentShardPublication {
    pending: PendingFixedResidentShard,
    header_publication: gpu_db_execution::PreparedU64HtoDPublication,
}

/// Reserve the maximum concurrent local host backing across two non-overlapping phases:
/// materializing the immutable private generation and, later, materializing the exact uniform
/// created-by payload. The latter moves its one backing allocation directly into the pre-WAL HtoD
/// owner, so it must be peak-accounted but is never retained by the sealed shard.
pub(super) fn fixed_rollover_host_materialization_scratch(
    immutable_materialization_peak: HostRetentionGeometry,
    retained_metadata: HostRetentionGeometry,
    uniform_stamp_payload_bytes: u64,
) -> Result<HostRetentionGeometry, ExecuteError> {
    let mut uniform_stamp_phase = retained_metadata;
    uniform_stamp_phase.checked_add_backing_bytes_slots(
        uniform_stamp_payload_bytes,
        u64::from(uniform_stamp_payload_bytes != 0),
        "fixed rollover uniform created-by stamp payload",
    )?;
    let peak = immutable_materialization_peak.peak(uniform_stamp_phase);
    if peak.retained_bytes() < retained_metadata.retained_bytes()
        || peak.allocation_slots() < retained_metadata.allocation_slots()
        || peak.generation_pin_slots() < retained_metadata.generation_pin_slots()
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "fixed rollover host materialization peak underflowed retained metadata".to_string(),
        )));
    }
    let mut scratch = HostRetentionGeometry::default();
    scratch.checked_add_backing_bytes_slots(
        peak.retained_bytes() - retained_metadata.retained_bytes(),
        peak.allocation_slots() - retained_metadata.allocation_slots(),
        "fixed rollover host materialization scratch",
    )?;
    scratch.checked_add_generation_pin_slots(
        peak.generation_pin_slots() - retained_metadata.generation_pin_slots(),
    )?;
    Ok(scratch)
}

impl PendingFixedResidentShard {
    /// Finish every mutable private byte before WAL for a uniform commit sequence.
    ///
    /// This consumes the unpublished shard, validates the sealed row/stamp geometry, uploads one
    /// exact-length created-by image, and resolves the count-header publication while the caller
    /// still owns the pre-WAL reservation. The payload header remains zero throughout this method:
    /// only [`PreparedFixedResidentShardPublication::publish_post_wal`] may make the rows visible.
    #[allow(dead_code)] // test-only foundation until the live fixed-rollover carrier adopts it
    pub(super) fn prepare_uniform_commit_pre_wal(
        self,
        commit_seq: Index,
    ) -> Result<PreparedFixedResidentShardPublication, ExecuteError> {
        let row_count = u64::from_le_bytes(self.final_count_header);
        let rows = usize::try_from(row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover row count does not fit host address space".to_string(),
            ))
        })?;
        let exact_stamp_bytes =
            rows.checked_mul(std::mem::size_of::<Index>())
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "sealed fixed-width rollover stamp geometry overflowed".to_string(),
                    ))
                })?;
        let exact_stamp_bytes_u64 = u64::try_from(exact_stamp_bytes).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover stamp bytes exceed device accounting".to_string(),
            ))
        })?;
        if rows == 0
            || exact_stamp_bytes != self.created_by_stamp_bytes
            || exact_stamp_bytes_u64 > self.created_by_bytes
            || !self
                .created_by_bytes
                .is_multiple_of(std::mem::size_of::<Index>() as u64)
            || self.payload_bytes < std::mem::size_of::<u64>() as u64
            || self.device_memory.metadata().allocated_bytes < self.payload_bytes
            || self.created_by_region.metadata().allocated_bytes < self.created_by_bytes
            || self.row_id_region.is_some() != (self.row_id_bytes != 0)
            || self
                .row_id_region
                .as_ref()
                .is_some_and(|region| region.metadata().allocated_bytes < self.row_id_bytes)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover drifted before pre-WAL uniform publication"
                    .to_string(),
            )));
        }

        // This is the one host allocation captured by `host_materialization_scratch`. Moving it
        // directly into the synchronous HtoD owner avoids a second encoding buffer or retained
        // post-WAL host payload.
        let mut stamp_payload = Vec::with_capacity(self.created_by_stamp_bytes);
        let stamp = commit_seq.to_le_bytes();
        for _ in 0..rows {
            stamp_payload.extend_from_slice(&stamp);
        }
        if stamp_payload.len() != self.created_by_stamp_bytes {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover uniform stamp payload lost exact geometry".to_string(),
            )));
        }
        self.created_by_region
            .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                byte_offset: 0,
                bytes: stamp_payload,
            }))
            .map_err(super::device_write_error(
                "sealed fixed-width pre-WAL created-by stamps",
            ))?;
        let header_publication = self
            .device_memory
            .prepare_u64_htod_publication(0, row_count)
            .map_err(super::device_write_error(
                "sealed fixed-width pre-WAL final row-count header",
            ))?;
        Ok(PreparedFixedResidentShardPublication {
            pending: self,
            header_publication,
        })
    }
}

impl PreparedFixedResidentShardPublication {
    /// Publish the header prepared before WAL, then return the sealed private generation.
    ///
    /// The capability owns the exact allocation, destination, primary context, value, and driver
    /// entry point. This consuming success path intentionally has no alternate encoding,
    /// allocation, device preparation, launch, cache, or fallback surface.
    #[allow(dead_code)] // test-only foundation until the live fixed-rollover carrier adopts it
    pub(super) fn publish_post_wal(self) -> Result<PendingFixedResidentShard, ExecuteError> {
        let Self {
            pending,
            header_publication,
        } = self;
        header_publication
            .publish()
            .map_err(super::device_write_error(
                "sealed fixed-width prepared final row-count header",
            ))?;
        Ok(pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_rollover_host_scratch_reserves_the_exact_peak_of_immutable_and_stamp_phases() {
        let mut immutable_peak = HostRetentionGeometry::default();
        immutable_peak
            .checked_add_backing_bytes_slots(48, 4, "test immutable materialization")
            .unwrap();
        let mut retained = HostRetentionGeometry::default();
        retained
            .checked_add_backing_bytes_slots(16, 1, "test retained metadata")
            .unwrap();

        let scratch = fixed_rollover_host_materialization_scratch(immutable_peak, retained, 32)
            .expect("non-overlapping stamp payload has a checked peak");
        assert_eq!(scratch.retained_bytes(), 32);
        assert_eq!(scratch.allocation_slots(), 3);
        assert_eq!(scratch.generation_pin_slots(), 0);

        let mut materialized = retained;
        materialized
            .checked_add_disjoint(scratch, "test materialized rollover peak")
            .unwrap();
        assert_eq!(materialized, immutable_peak);

        let stamp_dominated =
            fixed_rollover_host_materialization_scratch(immutable_peak, retained, 40)
                .expect("larger stamp phase has a checked peak");
        assert_eq!(stamp_dominated.retained_bytes(), 40);
        assert_eq!(stamp_dominated.allocation_slots(), 3);
        let mut materialized = retained;
        materialized
            .checked_add_disjoint(stamp_dominated, "test stamp-dominated rollover peak")
            .unwrap();
        assert_eq!(materialized.retained_bytes(), 56);
        assert_eq!(materialized.allocation_slots(), 4);
    }

    #[test]
    fn prepared_fixed_rollover_post_wal_surface_is_move_only_and_zero_build() {
        fn assert_send<T: Send>() {}
        assert_send::<PreparedFixedResidentShardPublication>();

        let source = include_str!("prepared_fixed_publication.rs");
        let pre_wal = source
            .split("fn prepare_uniform_commit_pre_wal")
            .nth(1)
            .and_then(|section| {
                section
                    .split("impl PreparedFixedResidentShardPublication")
                    .next()
            })
            .expect("fixed rollover pre-WAL preparation");
        for required in [
            "Vec::with_capacity",
            "append_owned_chunks",
            "prepare_u64_htod_publication",
            "PreparedFixedResidentShardPublication",
        ] {
            assert!(
                pre_wal.contains(required),
                "pre-WAL preparation must own {required}"
            );
        }

        let post_wal = source
            .split("impl PreparedFixedResidentShardPublication")
            .nth(1)
            .and_then(|section| section.split("#[cfg(test)]").next())
            .expect("fixed rollover prepared post-WAL publication");
        assert!(
            post_wal.contains("header_publication") && post_wal.contains(".publish()"),
            "prepared post-WAL publication must consume its sole header capability"
        );
        for forbidden in [
            "Vec",
            "to_vec",
            "encode_u64",
            "CudaOwnedDeviceMemoryChunk",
            "append_owned_chunks",
            "prepare_u64_htod_publication",
            "relational_residency_device_memory",
            "retain_device_memory_",
        ] {
            assert!(
                !post_wal.contains(forbidden),
                "prepared post-WAL publication must not {forbidden}"
            );
        }
    }
}
