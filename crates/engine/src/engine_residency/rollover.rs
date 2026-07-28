//! Fit-aware fixed-width shard rollover planning and private device construction.
//!
//! The plan is deliberately independent from descriptor publication. It first selects a capacity
//! from the exact currently retained budget, then [`PendingResidentShard`] and
//! [`PendingDenseResidentShard`] own every allocation until payload rows, MVCC birth stamps,
//! stable identities, and the row-count header are complete. A failed construction therefore
//! leaves no public descriptor or side-map owner behind.

use super::*;
use crate::typed_insert_batch::PreparedResidentDensePayload;

/// Sealed allocation geometry for one fixed-width, NULL-free rollover.
///
/// This is rebuilt at every authority boundary rather than stored in WAL. The canonical commit
/// boundary serializes the plan and apply observation; callers must reject when the captured open
/// shard no longer matches before consuming the plan.
#[derive(Clone, Debug)]
pub(super) struct ResidentRolloverPlan {
    capacity: usize,
    payload_bytes: u64,
    created_by_bytes: u64,
    row_id_bytes: u64,
    named_index_bytes: u64,
    bool_layouts: Vec<ResidentDeviceBoolColumnLayout>,
    capacity_scan_entries: u64,
}

impl ResidentRolloverPlan {
    /// Shared, checked fixed-width capacity geometry. `Some(target)` applies the rollover's
    /// target floor; `None` returns only the checked `next_power_of_two(2 * rows)` growth shape
    /// for admission paths whose established budget/eviction policy owns the final cap.
    pub(super) fn fixed_width_desired_capacity(
        rows: usize,
        target: Option<usize>,
    ) -> Option<usize> {
        let doubled = rows.checked_mul(2)?.checked_next_power_of_two()?;
        Some(target.map_or(doubled, |target| target.max(doubled)))
    }

    /// Pick the largest capacity in `[rows, desired_capacity]` that fits the exact remaining
    /// resident budget. No configured budget selects a special `2*k` shape: it participates only
    /// as the upper-bound fit search. `None` means even the dense `k`-row generation cannot fit.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(super) fn fixed_width_null_free(
        table: &RelationalTable,
        column_types: &[SqlType],
        rows: usize,
        desired_capacity: usize,
        row_ids_present: bool,
        named_indexes_required: bool,
        remaining_budget: Option<u64>,
    ) -> Result<Option<Self>, ExecuteError> {
        if rows == 0 || desired_capacity < rows || !is_fixed_width(column_types) {
            return Ok(None);
        }
        if desired_capacity > (1_usize << 31) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident rollover capacity is implausibly large".to_string(),
            )));
        }

        let make_plan = |capacity| {
            Self::at_capacity(
                table,
                column_types,
                rows,
                capacity,
                row_ids_present,
                named_indexes_required,
            )
        };
        let Some(remaining_budget) = remaining_budget else {
            let mut plan = make_plan(desired_capacity)?;
            plan.capacity_scan_entries = 1;
            return Ok(Some(plan));
        };

        let minimum = make_plan(rows)?;
        if minimum.total_allocation_bytes() > remaining_budget {
            return Ok(None);
        }

        // Allocation bytes are monotone in capacity: every payload/sidecar section grows with
        // capacity and the named-index estimator is a nondecreasing bounded table/posting shape.
        // Binary search gives the exact largest fitting integer capacity without turning a 4M-row
        // target into an O(target) host loop.
        let mut low = rows;
        let mut high = desired_capacity;
        let mut selected = minimum;
        let mut scans = 1_u64;
        while low <= high {
            let midpoint = low + (high - low) / 2;
            let candidate = make_plan(midpoint)?;
            scans = scans.saturating_add(1);
            if candidate.total_allocation_bytes() <= remaining_budget {
                selected = candidate;
                low = midpoint.saturating_add(1);
            } else {
                high = midpoint.saturating_sub(1);
            }
        }
        selected.capacity_scan_entries = scans;
        Ok(Some(selected))
    }

    fn at_capacity(
        table: &RelationalTable,
        column_types: &[SqlType],
        rows: usize,
        capacity: usize,
        row_ids_present: bool,
        named_indexes_required: bool,
    ) -> Result<Self, ExecuteError> {
        let (payload_bytes, bool_layouts) =
            fixed_width_payload_layout(table, column_types, capacity)?;
        let created_by_bytes = u64::try_from(capacity)
            .ok()
            .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<u64>() as u64))
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident rollover created-by allocation overflowed".to_string(),
                ))
            })?;
        let row_id_bytes = if row_ids_present { created_by_bytes } else { 0 };
        let named_index_bytes = if named_indexes_required {
            estimated_named_index_bytes_for_shard(table, rows, capacity).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has unsupported mandatory index allocation geometry",
                    table.name
                )))
            })?
        } else {
            0
        };
        Ok(Self {
            capacity,
            payload_bytes,
            created_by_bytes,
            row_id_bytes,
            named_index_bytes,
            bool_layouts,
            capacity_scan_entries: 0,
        })
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn total_allocation_bytes(&self) -> u64 {
        self.payload_bytes
            .saturating_add(self.created_by_bytes)
            .saturating_add(self.row_id_bytes)
            .saturating_add(self.named_index_bytes)
    }

    pub(super) fn allocation_bytes_before_indexes(&self) -> u64 {
        self.payload_bytes
            .saturating_add(self.created_by_bytes)
            .saturating_add(self.row_id_bytes)
    }

    pub(super) fn capacity_scan_entries(&self) -> u64 {
        self.capacity_scan_entries
    }

    pub(super) fn bool_layouts(&self) -> &[ResidentDeviceBoolColumnLayout] {
        &self.bool_layouts
    }

    pub(super) fn descriptor_bool_layouts_match(
        table: &RelationalTable,
        column_types: &[SqlType],
        capacity: usize,
        actual: &[ResidentDeviceBoolColumnLayout],
    ) -> bool {
        fixed_width_payload_layout(table, column_types, capacity)
            .is_ok_and(|(_, expected)| expected == actual)
    }
}

/// A fully private fixed-width rollover generation. Its Arcs are handed to the published shard only
/// after all live bytes have landed and the count header has been written last.
pub(super) struct PendingResidentShard {
    pub(super) device_memory: Arc<CudaResidentDeviceMemory>,
    pub(super) created_by_region: Arc<CudaResidentDeviceMemory>,
    pub(super) row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) bool_layouts: Vec<ResidentDeviceBoolColumnLayout>,
    pub(super) int4_stats: Vec<ResidentDeviceInt4ColumnStats>,
    pub(super) live_h2d_bytes: u64,
    pub(super) sidecar_fill_bytes: u64,
    pub(super) persistent_allocation_count: u64,
}

/// A fully private fixed-width generation reserved before WAL. Its immutable column chunks,
/// BoolBits, and exact row-identity sidecar are already device-resident; apply may only stamp
/// birth versions and publish the sealed count header last.
pub(super) struct PendingFixedResidentShard {
    pub(super) device_memory: Arc<CudaResidentDeviceMemory>,
    pub(super) created_by_region: Arc<CudaResidentDeviceMemory>,
    pub(super) row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) bool_layouts: Vec<ResidentDeviceBoolColumnLayout>,
    pub(super) int4_stats: Vec<ResidentDeviceInt4ColumnStats>,
    pub(super) payload_bytes: u64,
    pub(super) created_by_bytes: u64,
    pub(super) row_id_bytes: u64,
    pub(super) created_by_stamp_bytes: usize,
    pub(super) allocation_bytes: u64,
    pub(super) final_count_header: [u8; std::mem::size_of::<u64>()],
    /// Device zero-fill of the capacity-sized created-by sidecar, distinct from H2D.
    pub(super) sidecar_fill_bytes: u64,
    /// Pre-WAL immutable payload/BoolBits/row-ID H2D plus post-WAL stamps and final header.
    pub(super) live_h2d_bytes: u64,
    pub(super) persistent_allocation_count: u64,
}

/// A capacity-sized created-by sidecar allocated before WAL for a typed in-place append whose
/// established open descriptor has no version region yet.
pub(super) struct PendingInPlaceCreatedBy {
    pub(super) region: Arc<CudaResidentDeviceMemory>,
    pub(super) capacity_bytes: u64,
    pub(super) allocation_bytes: u64,
}

/// A fully private nullable/text rollover generation reserved before WAL. The typed plan may
/// carry this opaque allocation set through WAL, but only this allocation owner creates its
/// device buffers or performs the pre-WAL payload uploads.
pub(super) struct PendingDenseResidentShard {
    pub(super) payload: PreparedResidentDensePayload,
    pub(super) device_memory: Arc<CudaResidentDeviceMemory>,
    pub(super) created_by_region: Arc<CudaResidentDeviceMemory>,
    pub(super) row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
    pub(super) payload_bytes: u64,
    pub(super) created_by_bytes: usize,
    pub(super) created_by_bytes_u64: u64,
    pub(super) allocation_bytes: u64,
    /// Exact device-side zero fill for the created-by sidecar. This is not H2D traffic.
    pub(super) sidecar_fill_bytes: u64,
    /// Exact dense traffic across the whole retained plan: pre-WAL payload (including its zero
    /// count header), pre-WAL row IDs when present, post-WAL created-by stamps, and the final
    /// header rewrite.
    pub(super) live_h2d_bytes: u64,
    pub(super) persistent_allocation_count: u64,
}

impl PendingDenseResidentShard {
    /// Allocate and upload the dense physical generation before WAL. The payload count remains
    /// zero until mutation writes its sealed final header after stamping every created-by slot.
    pub(super) fn reserve_pre_wal(
        engine: &Engine,
        gpu_id: u16,
        mut payload: PreparedResidentDensePayload,
        created_by_bytes_u64: u64,
        row_id_payload: Option<Vec<u8>>,
    ) -> Result<Self, ExecuteError> {
        let created_by_bytes = usize::try_from(created_by_bytes_u64).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "dense rollover created-by allocation does not fit host address space".to_string(),
            ))
        })?;
        let row_id_preupload_bytes = row_id_payload
            .as_ref()
            .map(|bytes| u64::try_from(bytes.len()))
            .transpose()
            .map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "dense rollover row-id payload does not fit device accounting".to_string(),
                ))
            })?
            .unwrap_or(0);
        if row_id_preupload_bytes != 0 && row_id_preupload_bytes != created_by_bytes_u64 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "dense rollover row-id payload lost its sealed sidecar geometry".to_string(),
            )));
        }
        let payload_bytes = payload.device_payload_len().ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "dense rollover payload length does not fit device accounting".to_string(),
            ))
        })?;
        let upload = payload.take_pre_wal_upload().ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "dense rollover payload lost its final row-count header".to_string(),
            ))
        })?;
        let payload_preupload_bytes = u64::try_from(upload.len()).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "dense rollover payload upload does not fit device accounting".to_string(),
            ))
        })?;
        if payload_preupload_bytes != payload_bytes {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "dense rollover payload upload lost its sealed byte length".to_string(),
            )));
        }
        let device_memory = engine
            .relational_residency_device_memory(gpu_id, &upload)
            .map(Arc::new)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "dense rollover payload allocation failed".to_string(),
                ))
            })?;
        let created_by_region = Arc::new(
            engine
                .cuda_driver_probe_runtime()
                .retain_device_memory_zeroed(gpu_id, created_by_bytes_u64)
                .map_err(device_allocation_error("dense created-by sidecar"))?,
        );
        let row_id_region = match row_id_payload {
            Some(payload) => Some(
                engine
                    .relational_residency_device_memory(gpu_id, &payload)
                    .map(Arc::new)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "dense rollover row-id allocation failed".to_string(),
                        ))
                    })?,
            ),
            None => None,
        };
        let allocation_bytes = device_memory
            .metadata()
            .allocated_bytes
            .checked_add(created_by_region.metadata().allocated_bytes)
            .and_then(|bytes| {
                bytes.checked_add(
                    row_id_region
                        .as_ref()
                        .map_or(0, |region| region.metadata().allocated_bytes),
                )
            })
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "dense rollover allocation accounting overflowed".to_string(),
                ))
            })?;
        let (sidecar_fill_bytes, live_h2d_bytes) = dense_probe_accounting(
            payload_preupload_bytes,
            row_id_preupload_bytes,
            created_by_bytes_u64,
        )
        .ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "dense rollover live upload accounting overflowed".to_string(),
            ))
        })?;
        let persistent_allocation_count = 2 + u64::from(row_id_region.is_some());
        Ok(Self {
            payload,
            device_memory,
            created_by_region,
            row_id_region,
            payload_bytes,
            created_by_bytes,
            created_by_bytes_u64,
            allocation_bytes,
            sidecar_fill_bytes,
            live_h2d_bytes,
            persistent_allocation_count,
        })
    }
}

impl PendingFixedResidentShard {
    /// Reserve the exact fixed-width rollover generation and upload every immutable byte before
    /// WAL. The payload allocation is zeroed, so stripping the final count chunk leaves the
    /// unpublished header at zero until the post-WAL mutation owner writes it last.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn reserve_pre_wal(
        engine: &Engine,
        gpu_id: u16,
        plan: &ResidentRolloverPlan,
        rows: usize,
        mut chunks: Vec<CudaOwnedDeviceMemoryChunk>,
        bool_uploads: Vec<(u64, Box<[u8]>)>,
        int4_stats: Vec<ResidentDeviceInt4ColumnStats>,
        row_id_payload: Option<Vec<u8>>,
    ) -> Result<Self, ExecuteError> {
        if rows == 0
            || rows > plan.capacity
            || bool_uploads.len() != plan.bool_layouts.len()
            || bool_uploads
                .iter()
                .zip(&plan.bool_layouts)
                .any(|((offset, values), layout)| {
                    *offset != layout.bitmap_byte_offset || values.len() != rows
                })
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover lost its private payload geometry".to_string(),
            )));
        }
        let final_count_header = u64::try_from(rows)
            .map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed fixed-width rollover row count overflowed".to_string(),
                ))
            })?
            .to_le_bytes();
        let header = chunks
            .pop()
            .filter(|chunk| {
                chunk.byte_offset == 0 && chunk.bytes.as_slice() == final_count_header.as_slice()
            })
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed fixed-width rollover omitted its final row-count header".to_string(),
                ))
            })?;
        let immutable_payload_h2d = chunks.iter().try_fold(0_u64, |bytes, chunk| {
            u64::try_from(chunk.bytes.len())
                .ok()
                .and_then(|chunk_bytes| bytes.checked_add(chunk_bytes))
        });
        let bool_h2d = bool_uploads.iter().try_fold(0_u64, |bytes, (_, values)| {
            u64::try_from(values.len())
                .ok()
                .and_then(|value_bytes| bytes.checked_add(value_bytes))
        });
        let created_by_stamp_bytes =
            rows.checked_mul(std::mem::size_of::<u64>())
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "sealed fixed-width rollover stamp geometry overflowed".to_string(),
                    ))
                })?;
        let created_by_stamp_bytes_u64 = u64::try_from(created_by_stamp_bytes).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover stamp bytes exceed device accounting".to_string(),
            ))
        })?;
        let row_id_preupload_bytes = row_id_payload
            .as_ref()
            .map(|bytes| u64::try_from(bytes.len()))
            .transpose()
            .map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed fixed-width rollover row-id bytes exceed device accounting".to_string(),
                ))
            })?
            .unwrap_or(0);
        if row_id_payload.is_some() != (plan.row_id_bytes != 0)
            || (row_id_payload.is_some() && row_id_preupload_bytes != created_by_stamp_bytes_u64)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover row-id geometry drifted".to_string(),
            )));
        }
        let runtime = engine.cuda_driver_probe_runtime();
        let device_memory = Arc::new(
            runtime
                .retain_device_memory_zeroed(gpu_id, plan.payload_bytes)
                .map_err(device_allocation_error("sealed fixed-width payload"))?,
        );
        let created_by_region = Arc::new(
            runtime
                .retain_device_memory_zeroed(gpu_id, plan.created_by_bytes)
                .map_err(device_allocation_error(
                    "sealed fixed-width created-by sidecar",
                ))?,
        );
        let row_id_region = if plan.row_id_bytes == 0 {
            None
        } else {
            Some(Arc::new(
                runtime
                    .retain_device_memory_recompacted(
                        gpu_id,
                        plan.row_id_bytes,
                        &[],
                        &[gpu_db_execution::RecompactFill {
                            byte_offset: 0,
                            len: plan.row_id_bytes,
                            fill_byte: ROW_ID_UNSTAMPED_FILL_BYTE,
                        }],
                        &[],
                    )
                    .map_err(device_allocation_error("sealed fixed-width row-id sidecar"))?,
            ))
        };
        if device_memory.metadata().allocated_bytes < plan.payload_bytes
            || created_by_region.metadata().allocated_bytes < plan.created_by_bytes
            || row_id_region
                .as_ref()
                .is_some_and(|region| region.metadata().allocated_bytes < plan.row_id_bytes)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover allocation was smaller than its pre-WAL geometry"
                    .to_string(),
            )));
        }
        device_memory
            .append_owned_chunks(chunks)
            .map_err(device_write_error("sealed fixed-width payload preupload"))?;
        for (offset, values) in bool_uploads {
            device_memory
                .set_bool_bitmap_range(offset, 0, &values)
                .map_err(device_write_error("sealed fixed-width bool preupload"))?;
        }
        if let (Some(region), Some(payload)) = (&row_id_region, row_id_payload) {
            region
                .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: payload,
                }))
                .map_err(device_write_error("sealed fixed-width row-id preupload"))?;
        }
        let allocation_bytes = device_memory
            .metadata()
            .allocated_bytes
            .checked_add(created_by_region.metadata().allocated_bytes)
            .and_then(|bytes| {
                bytes.checked_add(
                    row_id_region
                        .as_ref()
                        .map_or(0, |region| region.metadata().allocated_bytes),
                )
            })
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed fixed-width allocation accounting overflowed".to_string(),
                ))
            })?;
        let live_h2d_bytes = immutable_payload_h2d
            .and_then(|bytes| bool_h2d.and_then(|bool_bytes| bytes.checked_add(bool_bytes)))
            .and_then(|bytes| bytes.checked_add(row_id_preupload_bytes))
            .and_then(|bytes| bytes.checked_add(created_by_stamp_bytes_u64))
            .and_then(|bytes| bytes.checked_add(u64::try_from(header.bytes.len()).ok()?))
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed fixed-width live upload accounting overflowed".to_string(),
                ))
            })?;
        let sidecar_fill_bytes = plan
            .created_by_bytes
            .checked_add(plan.row_id_bytes)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed fixed-width sidecar fill accounting overflowed".to_string(),
                ))
            })?;
        Ok(Self {
            device_memory,
            created_by_region,
            row_id_region,
            bool_layouts: plan.bool_layouts.clone(),
            int4_stats,
            payload_bytes: plan.payload_bytes,
            created_by_bytes: plan.created_by_bytes,
            row_id_bytes: plan.row_id_bytes,
            created_by_stamp_bytes,
            allocation_bytes,
            final_count_header,
            sidecar_fill_bytes,
            live_h2d_bytes,
            persistent_allocation_count: 2 + u64::from(plan.row_id_bytes != 0),
        })
    }

    /// Finish the only mutable portion after WAL: exact created-by stamps, then the sealed count
    /// header. The generation remains private until mutation publishes its descriptor.
    pub(super) fn finish_post_wal(self, stamps: &[Index]) -> Result<Self, ExecuteError> {
        if stamps.is_empty()
            || std::mem::size_of_val(stamps) != self.created_by_stamp_bytes
            || self.device_memory.metadata().allocated_bytes < self.payload_bytes
            || self.created_by_region.metadata().allocated_bytes < self.created_by_bytes
            || self
                .row_id_region
                .as_ref()
                .is_some_and(|region| region.metadata().allocated_bytes < self.row_id_bytes)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "sealed fixed-width rollover drifted before post-WAL header publication"
                    .to_string(),
            )));
        }
        self.created_by_region
            .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                byte_offset: 0,
                bytes: encode_u64(stamps),
            }))
            .map_err(device_write_error("sealed fixed-width created-by stamps"))?;
        self.device_memory
            .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                byte_offset: 0,
                bytes: self.final_count_header.to_vec(),
            }))
            .map_err(device_write_error(
                "sealed fixed-width final row-count header",
            ))?;
        Ok(self)
    }
}

impl PendingInPlaceCreatedBy {
    /// Reserve a missing open-shard created-by region before WAL. The zero fill preserves the
    /// born-visible meaning until apply stamps the planned live slots and publishes row_count.
    pub(super) fn reserve_pre_wal(
        engine: &Engine,
        gpu_id: u16,
        capacity: usize,
    ) -> Result<Self, ExecuteError> {
        let capacity_bytes = u64::try_from(capacity)
            .ok()
            .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed in-place created-by sidecar geometry overflowed".to_string(),
                ))
            })?;
        let region = Arc::new(
            engine
                .cuda_driver_probe_runtime()
                .retain_device_memory_zeroed(gpu_id, capacity_bytes)
                .map_err(device_allocation_error(
                    "sealed in-place created-by sidecar",
                ))?,
        );
        Ok(Self {
            allocation_bytes: region.metadata().allocated_bytes,
            region,
            capacity_bytes,
        })
    }

    /// Consume the private pre-WAL allocation only if it still has the exact capacity geometry.
    /// Mutation owns the subsequent descriptor and side-map publication.
    pub(super) fn into_region(self, capacity: usize) -> Option<Arc<CudaResidentDeviceMemory>> {
        let expected_bytes = u64::try_from(capacity)
            .ok()
            .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))?;
        (self.capacity_bytes == expected_bytes
            && self.allocation_bytes >= expected_bytes
            && self.region.metadata().allocated_bytes >= expected_bytes)
            .then_some(self.region)
    }
}

impl PendingResidentShard {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        engine: &Engine,
        gpu_id: u16,
        table: &RelationalTable,
        column_types: &[SqlType],
        rows: &[Vec<SqlValue>],
        stamps: &[Index],
        row_ids: Option<&[u64]>,
        plan: &ResidentRolloverPlan,
    ) -> Result<Self, ExecuteError> {
        if rows.is_empty()
            || rows.len() != stamps.len()
            || row_ids.is_some_and(|ids| ids.len() != rows.len())
            || !is_fixed_width(column_types)
            || rows.iter().any(|row| {
                row.len() != column_types.len()
                    || row.iter().any(|value| matches!(value, SqlValue::Null))
            })
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "fixed-width rollover inputs lost their NULL-free parallel geometry".to_string(),
            )));
        }
        if row_ids.is_some() != (plan.row_id_bytes != 0) || rows.len() > plan.capacity {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "fixed-width rollover plan no longer matches apply inputs".to_string(),
            )));
        }

        // Validate and materialize the small live BOOL uploads before the runtime owns any
        // allocation. A type mismatch must fail the private build closed; encoding an arbitrary
        // non-BOOL value as false would publish a device-visible wrong result.
        let bool_uploads = plan
            .bool_layouts
            .iter()
            .map(|layout| {
                let column = table
                    .columns
                    .iter()
                    .position(|candidate| candidate.name == layout.name)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "fixed-width rollover bool layout lost its catalog column".to_string(),
                        ))
                    })?;
                let values = rows
                    .iter()
                    .map(|row| match row[column] {
                        SqlValue::Bool(value) => Ok(u8::from(value)),
                        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "fixed-width rollover bool column \"{}\" lost its Bool value geometry",
                            layout.name
                        )))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((layout.bitmap_byte_offset, values))
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;

        let chunks = compute_open_shard_int4_append_chunks(column_types, plan.capacity, 0, rows)?;
        Self::build_from_chunks(
            engine,
            gpu_id,
            plan,
            rows.len(),
            chunks,
            bool_uploads,
            stamps,
            row_ids,
            plan.bool_layouts.clone(),
            int4_stats(table, column_types, rows),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from_chunks(
        engine: &Engine,
        gpu_id: u16,
        plan: &ResidentRolloverPlan,
        rows: usize,
        mut chunks: Vec<CudaOwnedDeviceMemoryChunk>,
        bool_uploads: Vec<(u64, Vec<u8>)>,
        stamps: &[Index],
        row_ids: Option<&[u64]>,
        bool_layouts: Vec<ResidentDeviceBoolColumnLayout>,
        int4_stats: Vec<ResidentDeviceInt4ColumnStats>,
    ) -> Result<Self, ExecuteError> {
        if rows == 0
            || stamps.len() != rows
            || row_ids.is_some_and(|ids| ids.len() != rows)
            || row_ids.is_some() != (plan.row_id_bytes != 0)
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "fixed-width rollover plan no longer matches apply inputs".to_string(),
            )));
        }
        let header = chunks
            .pop()
            .filter(|chunk| chunk.byte_offset == 0)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "fixed-width rollover payload omitted its final row-count header".to_string(),
                ))
            })?;
        let runtime = engine.cuda_driver_probe_runtime();
        let device_memory = Arc::new(
            runtime
                .retain_device_memory_zeroed(gpu_id, plan.payload_bytes)
                .map_err(device_allocation_error("payload"))?,
        );
        let created_by_region = Arc::new(
            runtime
                .retain_device_memory_zeroed(gpu_id, plan.created_by_bytes)
                .map_err(device_allocation_error("created-by sidecar"))?,
        );
        let row_id_region = if plan.row_id_bytes == 0 {
            None
        } else {
            Some(Arc::new(
                runtime
                    .retain_device_memory_recompacted(
                        gpu_id,
                        plan.row_id_bytes,
                        &[],
                        &[gpu_db_execution::RecompactFill {
                            byte_offset: 0,
                            len: plan.row_id_bytes,
                            fill_byte: ROW_ID_UNSTAMPED_FILL_BYTE,
                        }],
                        &[],
                    )
                    .map_err(device_allocation_error("row-id sidecar"))?,
            ))
        };
        let mut live_h2d_bytes: u64 = chunks.iter().map(|chunk| chunk.bytes.len() as u64).sum();
        device_memory
            .append_owned_chunks(chunks)
            .map_err(device_write_error("payload rows"))?;
        for (bitmap_byte_offset, values) in bool_uploads {
            device_memory
                .set_bool_bitmap_range(bitmap_byte_offset, 0, &values)
                .map_err(device_write_error("bool payload rows"))?;
            live_h2d_bytes = live_h2d_bytes.saturating_add(values.len() as u64);
        }
        created_by_region
            .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                byte_offset: 0,
                bytes: encode_u64(stamps),
            }))
            .map_err(device_write_error("created-by live slots"))?;
        live_h2d_bytes = live_h2d_bytes.saturating_add((rows * std::mem::size_of::<u64>()) as u64);
        if let (Some(region), Some(row_ids)) = (&row_id_region, row_ids) {
            region
                .append_owned_chunks(std::iter::once(CudaOwnedDeviceMemoryChunk {
                    byte_offset: 0,
                    bytes: encode_u64(row_ids),
                }))
                .map_err(device_write_error("row-id live slots"))?;
            live_h2d_bytes =
                live_h2d_bytes.saturating_add((rows * std::mem::size_of::<u64>()) as u64);
        }
        // Header last: no descriptor can later expose a row whose values or sidecars failed.
        live_h2d_bytes = live_h2d_bytes.saturating_add(header.bytes.len() as u64);
        device_memory
            .append_owned_chunks(std::iter::once(header))
            .map_err(device_write_error("final row-count header"))?;
        Ok(Self {
            device_memory,
            created_by_region,
            row_id_region,
            bool_layouts,
            int4_stats,
            live_h2d_bytes,
            sidecar_fill_bytes: plan.created_by_bytes.saturating_add(plan.row_id_bytes),
            persistent_allocation_count: 2 + u64::from(plan.row_id_bytes != 0),
        })
    }
}

/// Accounting for one dense plan's whole device-transfer lifecycle. The initial zero-count header
/// is part of `payload_preupload_bytes`; the final post-WAL header rewrite is always one u64.
fn dense_probe_accounting(
    payload_preupload_bytes: u64,
    row_id_preupload_bytes: u64,
    created_by_bytes: u64,
) -> Option<(u64, u64)> {
    let final_header_bytes = std::mem::size_of::<u64>() as u64;
    let live_h2d_bytes = payload_preupload_bytes
        .checked_add(row_id_preupload_bytes)?
        .checked_add(created_by_bytes)?
        .checked_add(final_header_bytes)?;
    Some((created_by_bytes, live_h2d_bytes))
}

fn is_fixed_width(column_types: &[SqlType]) -> bool {
    !column_types.is_empty()
        && column_types.iter().all(|ty| {
            matches!(
                ty,
                SqlType::Int2
                    | SqlType::Int4
                    | SqlType::Date
                    | SqlType::Int8
                    | SqlType::Timestamp
                    | SqlType::Numeric { .. }
                    | SqlType::Uuid
                    | SqlType::Bool
            )
        })
}

fn fixed_width_payload_layout(
    table: &RelationalTable,
    column_types: &[SqlType],
    capacity: usize,
) -> Result<(u64, Vec<ResidentDeviceBoolColumnLayout>), ExecuteError> {
    let mut bytes = std::mem::size_of::<u64>() as u64;
    let mut bool_layouts = Vec::new();
    for width in [4_u64, 8, 16] {
        let count = column_types
            .iter()
            .filter(|ty| match width {
                4 => matches!(ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date),
                8 => matches!(ty, SqlType::Int8 | SqlType::Timestamp),
                16 => matches!(ty, SqlType::Numeric { .. } | SqlType::Uuid),
                _ => false,
            })
            .count() as u64;
        bytes = bytes
            .checked_add(
                count
                    .checked_mul(capacity as u64)
                    .and_then(|bytes| bytes.checked_mul(width))
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "fixed-width resident payload allocation overflowed".to_string(),
                        ))
                    })?,
            )
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "fixed-width resident payload allocation overflowed".to_string(),
                ))
            })?;
    }
    let bool_bytes = u64::try_from(capacity.div_ceil(32))
        .ok()
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "fixed-width bool bitmap allocation overflowed".to_string(),
            ))
        })?;
    for (index, ty) in column_types.iter().enumerate() {
        if matches!(ty, SqlType::Bool) {
            bool_layouts.push(ResidentDeviceBoolColumnLayout {
                name: table.columns[index].name.clone(),
                bitmap_byte_offset: bytes,
            });
            bytes = bytes.checked_add(bool_bytes).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "fixed-width bool bitmap allocation overflowed".to_string(),
                ))
            })?;
        }
    }
    Ok((bytes, bool_layouts))
}

fn int4_stats(
    table: &RelationalTable,
    column_types: &[SqlType],
    rows: &[Vec<SqlValue>],
) -> Vec<ResidentDeviceInt4ColumnStats> {
    column_types
        .iter()
        .enumerate()
        .filter(|(_, ty)| matches!(ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
        .map(|(column, _)| {
            let (min, max) = rows.iter().fold((i32::MAX, i32::MIN), |(min, max), row| {
                let value = sql_value_as_int4(&row[column]);
                (min.min(value), max.max(value))
            });
            ResidentDeviceInt4ColumnStats {
                name: table.columns[column].name.clone(),
                min,
                max,
            }
        })
        .collect()
}

fn encode_u64(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn device_allocation_error(
    owner: &'static str,
) -> impl FnOnce(gpu_db_execution::CudaRuntimeProbeError) -> ExecuteError {
    move |error| {
        ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident INSERT {owner} allocation failed: {error}"
        )))
    }
}

fn device_write_error(
    owner: &'static str,
) -> impl FnOnce(gpu_db_execution::CudaRuntimeProbeError) -> ExecuteError {
    move |error| {
        ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident INSERT {owner} write failed: {error}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> RelationalTable {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        engine.relational_catalog_table("accounts").unwrap()
    }

    #[test]
    fn dense_probe_accounting_includes_preuploads_and_final_header_for_both_identity_shapes() {
        // Three `(id int4, body text nullable)` rows occupy: u64 count header (8), id (12),
        // one NULL bitmap word (4), four text offsets (32), and the UTF-8 `naïve` bytes (6).
        let payload_bytes = 62;
        let created_by_bytes = 3 * std::mem::size_of::<u64>() as u64;
        assert_eq!(
            dense_probe_accounting(payload_bytes, created_by_bytes, created_by_bytes),
            Some((created_by_bytes, 118)),
            "payload + preuploaded row IDs + post-WAL stamps + final header"
        );
        assert_eq!(
            dense_probe_accounting(payload_bytes, 0, created_by_bytes),
            Some((created_by_bytes, 94)),
            "synthetic no-identity plans omit only the row-ID preupload"
        );
    }

    #[test]
    fn fixed_width_plan_prefers_target_and_budget_only_trims_to_exact_fit() {
        let table = table();
        let types = vec![SqlType::Int4, SqlType::Int4];
        let desired = 64;
        let unconstrained = ResidentRolloverPlan::fixed_width_null_free(
            &table, &types, 3, desired, true, false, None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(unconstrained.capacity(), desired);
        let dense = ResidentRolloverPlan::fixed_width_null_free(
            &table,
            &types,
            3,
            desired,
            true,
            false,
            Some(
                ResidentRolloverPlan::fixed_width_null_free(
                    &table, &types, 3, 3, true, false, None,
                )
                .unwrap()
                .unwrap()
                .total_allocation_bytes(),
            ),
        )
        .unwrap()
        .unwrap();
        assert_eq!(dense.capacity(), 3);
        assert!(dense.capacity_scan_entries() > 1);
    }

    #[test]
    fn fixed_width_plan_rejects_when_dense_generation_cannot_fit() {
        let table = table();
        let types = vec![SqlType::Int4, SqlType::Int4];
        assert!(ResidentRolloverPlan::fixed_width_null_free(
            &table,
            &types,
            4,
            64,
            true,
            false,
            Some(1),
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn fixed_width_descriptor_bool_layout_requires_exact_catalog_identity_and_offset() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(
                1,
                "CREATE TABLE descriptor_shape (id int4, tick int8, amount numeric(10,2), token uuid, flag bool)",
            )
            .unwrap();
        let table = engine.relational_catalog_table("descriptor_shape").unwrap();
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let plan =
            ResidentRolloverPlan::fixed_width_null_free(&table, &types, 2, 16, true, false, None)
                .unwrap()
                .unwrap();
        assert!(ResidentRolloverPlan::descriptor_bool_layouts_match(
            &table,
            &types,
            plan.capacity(),
            plan.bool_layouts(),
        ));
        let mut mislabeled = plan.bool_layouts().to_vec();
        mislabeled[0].name = "wrong_flag".to_string();
        assert!(!ResidentRolloverPlan::descriptor_bool_layouts_match(
            &table,
            &types,
            plan.capacity(),
            &mislabeled,
        ));
        let mut wrong_offset = plan.bool_layouts().to_vec();
        wrong_offset[0].bitmap_byte_offset += 4;
        assert!(!ResidentRolloverPlan::descriptor_bool_layouts_match(
            &table,
            &types,
            plan.capacity(),
            &wrong_offset,
        ));
    }

    #[test]
    fn fixed_width_geometry_separates_growth_from_rollover_target_floor() {
        assert_eq!(
            ResidentRolloverPlan::fixed_width_desired_capacity(1_000, None),
            Some(2_048)
        );
        assert_eq!(
            ResidentRolloverPlan::fixed_width_desired_capacity(1_000, Some(262_144)),
            Some(262_144)
        );
        assert_eq!(
            ResidentRolloverPlan::fixed_width_desired_capacity(200_000, Some(262_144)),
            Some(524_288)
        );
    }

    #[test]
    fn pending_bool_value_type_mismatch_fails_before_device_allocation() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(1, "CREATE TABLE bool_accounts (id int4, flag bool)")
            .unwrap();
        let table = engine.relational_catalog_table("bool_accounts").unwrap();
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let rows = vec![vec![SqlValue::Int4(1), SqlValue::Int4(1)]];
        let plan = ResidentRolloverPlan::fixed_width_null_free(
            &table,
            &types,
            rows.len(),
            4,
            true,
            false,
            None,
        )
        .unwrap()
        .expect("the NULL-free fixed-width bool row has a valid plan");

        let Err(error) = PendingResidentShard::build(
            &engine,
            0,
            &table,
            &types,
            &rows,
            &[1],
            Some(&[11]),
            &plan,
        ) else {
            panic!("a non-Bool payload must be rejected before any device allocation");
        };
        assert!(error
            .to_string()
            .contains("bool column \"flag\" lost its Bool value geometry"));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn rollover_created_pending_shard_is_device_initialized_and_query_visible() {
        let engine = Engine::new_local();
        engine.set_shard_size_target(4);
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .execute_text(
                1,
                "CREATE TABLE rollover_bool (id int4, balance int4, flag bool)",
            )
            .unwrap();
        engine
            .execute_text(
                2,
                "INSERT INTO rollover_bool VALUES (1, 10, false), (2, 20, true)",
            )
            .unwrap();
        engine
            .execute_text(
                3,
                "INSERT INTO rollover_bool VALUES (3, 30, false), (4, 40, true)",
            )
            .unwrap();
        engine
            .execute_text(
                4,
                "INSERT INTO rollover_bool VALUES (11, 110, true), (12, 120, false)",
            )
            .unwrap();

        let table = engine.relational_catalog_table("rollover_bool").unwrap();
        let shards = engine.read_residency_shards();
        let Some(shard) = shards
            .get("rollover_bool")
            .and_then(|table_shards| table_shards.iter().find(|shard| shard.shard_id == 2))
            .cloned()
        else {
            panic!("qualified ignored GPU test must retain the rollover generation");
        };
        let Some(device_memory) = shard.device_memory.as_ref() else {
            panic!("qualified ignored GPU test must retain the rollover device payload");
        };
        let Some(created_by_region) = shard.created_by_region.as_ref() else {
            panic!("rollover-created shard must retain created_by sidecar");
        };
        let Some(row_id_region) = shard.row_id_region.as_ref() else {
            panic!("rollover-created shard must retain stable row identities");
        };

        assert_eq!(shard.row_count, 2);
        assert_eq!(
            shard.capacity, 4,
            "rollover retains the target headroom geometry"
        );
        let descriptor = engine.resident_snapshot_for_shard(&shard, &table);
        assert_eq!(
            device_memory.read_resident_u64_column(0, 1).unwrap(),
            vec![2]
        );

        let id_offset =
            crate::relational_model::resident_device_int4_column_offset(&descriptor, &table, 0)
                .unwrap();
        let balance_offset =
            crate::relational_model::resident_device_int4_column_offset(&descriptor, &table, 1)
                .unwrap();
        assert_eq!(
            device_memory
                .read_resident_i32_column(id_offset, 3)
                .unwrap(),
            vec![11, 12, 0],
            "first unused fixed-width payload slot remains zeroed"
        );
        assert_eq!(
            device_memory
                .read_resident_i32_column(balance_offset, 3)
                .unwrap(),
            vec![110, 120, 0],
            "first unused fixed-width payload slot remains zeroed"
        );
        let bool_offset =
            crate::relational_model::resident_device_bool_column_offset(&descriptor, &table, 2)
                .unwrap();
        assert_eq!(
            device_memory
                .read_resident_i32_column(bool_offset, 1)
                .unwrap()[0] as u32
                & 0b111,
            0b001,
            "live bool bits are written LSB-first and the first unused bit remains false"
        );

        let committed = engine.committed_seq();
        assert_eq!(
            created_by_region.read_resident_u64_column(0, 3).unwrap(),
            vec![
                committed,
                committed,
                u64::from_le_bytes([CREATED_BY_VISIBLE_FILL_BYTE; 8])
            ],
            "created_by stamps only live slots; headroom keeps the visibility fill"
        );
        let row_ids = row_id_region.read_resident_u64_column(0, 3).unwrap();
        assert_ne!(row_ids[0], u64::MAX);
        assert_ne!(row_ids[1], u64::MAX);
        assert_eq!(
            row_ids[2],
            u64::MAX,
            "first unused identity slot keeps its sentinel"
        );

        let visible = engine
            .execute_relational_select_text(
                "SELECT id, balance, flag FROM rollover_bool WHERE id >= 11 ORDER BY id",
            )
            .unwrap();
        assert_eq!(visible.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(visible.fallback_reason, None);
        assert_eq!(
            visible.rows.into_boxed(),
            vec![
                vec![
                    SqlValue::Int4(11),
                    SqlValue::Int4(110),
                    SqlValue::Bool(true)
                ],
                vec![
                    SqlValue::Int4(12),
                    SqlValue::Int4(120),
                    SqlValue::Bool(false)
                ],
            ],
            "the initialized rollover generation is query-visible on the GPU path"
        );
    }

    #[test]
    fn transaction_pre_wal_reservation_uses_the_fixed_width_rollover_plan() {
        let mut engine = Engine::new_local_test_engine();
        engine
            .execute_text(
                1,
                "CREATE TABLE reservation_accounts (id int4, balance int4)",
            )
            .unwrap();
        let table = engine
            .relational_catalog_table("reservation_accounts")
            .unwrap();
        let rows = vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
        ];
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let desired = ResidentRolloverPlan::fixed_width_desired_capacity(
            rows.len(),
            Some(engine.shard_size_target()),
        )
        .unwrap();
        let expected = ResidentRolloverPlan::fixed_width_null_free(
            &table,
            &types,
            rows.len(),
            desired,
            true,
            false,
            None,
        )
        .unwrap()
        .unwrap()
        .total_allocation_bytes();
        let (_, reserved) = engine
            .transaction_resident_append_allocation_bytes(&table, &rows, false)
            .unwrap();
        assert_eq!(reserved, expected);

        engine.set_relational_residency_budget_bytes(0, 0);
        let error = engine
            .transaction_resident_append_allocation_bytes(&table, &rows, false)
            .expect_err("a dense fixed-width generation must reject before WAL when it cannot fit");
        assert!(error
            .to_string()
            .contains("cannot fit a dense fixed-width rollover before WAL"));
    }

    #[test]
    fn exact_resident_accounting_reports_walked_entries_separately_from_bytes() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(
                1,
                "CREATE TABLE accounting_accounts (id int4, balance int4)",
            )
            .unwrap();
        let (bytes, entries) = engine.relational_resident_bytes_and_entries_for_gpu(0);
        assert_eq!(bytes, engine.relational_resident_bytes_for_gpu(0));
        assert!(
            entries >= 2,
            "the empty bootstrap shard is visited for both allocation identity and bytes"
        );
    }
}
