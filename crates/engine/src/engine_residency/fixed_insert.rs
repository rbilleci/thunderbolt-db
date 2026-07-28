//! Typed device-plan compilation for the accepted INSERT residency route.
//!
//! This module owns no CUDA allocation, device write, descriptor publication, side-map mutation,
//! or WAL. It consumes a sealed source into an opaque plan, then the plan re-enters mutation's one
//! publisher at apply time. A post-WAL apply failure is fatal rather than permission to fall back.

use super::append_source::ResidentAppendSource;
use super::*;
use crate::typed_insert_batch::{
    PreparedResidentAppendSource, PreparedResidentBoolUpload, TypedInsertBatch,
};
use std::sync::MutexGuard;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceInsertPlanPrepareError {
    /// The fixed route is not applicable to this source or resident descriptor. Callers may
    /// retain the established pre-WAL legacy preparation in this case.
    UnsupportedShape,
    /// A source proved the exact CREATE bootstrap sentinel, but its first rollover cannot fit the
    /// current residency budget. Legacy must not claim WAL/row identities, then find the same
    /// allocation miss before canonical WAL/status buffering, device apply, or group durability.
    RetryableBoundBootstrapResource,
    /// The exact CREATE bootstrap binding changed while its rollover reservation was being
    /// sealed. This is retryable rather than a license to replace the bound path with legacy.
    RetryableBoundBootstrapState,
}

#[derive(Clone, Copy)]
enum ResidentOpenShardAppendPreparationMode {
    LiveUnindexed,
    #[cfg(test)]
    IndexedInPlaceProof {
        index_scratch_bytes: u64,
    },
}

/// A post-WAL device-apply failure is terminal before physical group durability; the active wave
/// wedges with durable count unchanged rather than attempting legacy re-application or publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceInsertPlanApplyError {
    PlanDrift,
    ShapeDrift,
    PublisherFailure,
}

/// Exact durable row identities sealed into a typed append before WAL. Construction is limited to
/// owned IDs; the synthetic form is test-only so a production cutover cannot accidentally lose a
/// bound entity-identity sidecar.
pub(crate) struct DeviceInsertRowIds {
    kind: DeviceInsertRowIdsKind,
}

enum DeviceInsertRowIdsKind {
    Exact(Box<[u64]>),
    SyntheticNoIdentity,
    ConsumedExact,
}

impl DeviceInsertRowIds {
    pub(crate) fn exact(ids: Box<[u64]>) -> Self {
        Self {
            kind: DeviceInsertRowIdsKind::Exact(ids),
        }
    }

    #[cfg(test)]
    pub(crate) fn synthetic_no_identity() -> Self {
        Self {
            kind: DeviceInsertRowIdsKind::SyntheticNoIdentity,
        }
    }

    fn is_exact(&self) -> bool {
        matches!(
            &self.kind,
            DeviceInsertRowIdsKind::Exact(_) | DeviceInsertRowIdsKind::ConsumedExact
        )
    }

    fn exact_len_matches(&self, rows: usize) -> bool {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(ids) => ids.len() == rows,
            DeviceInsertRowIdsKind::SyntheticNoIdentity => true,
            DeviceInsertRowIdsKind::ConsumedExact => false,
        }
    }

    fn take_exact(&mut self) -> Option<Box<[u64]>> {
        match std::mem::replace(&mut self.kind, DeviceInsertRowIdsKind::SyntheticNoIdentity) {
            DeviceInsertRowIdsKind::Exact(ids) => {
                self.kind = DeviceInsertRowIdsKind::ConsumedExact;
                Some(ids)
            }
            DeviceInsertRowIdsKind::SyntheticNoIdentity | DeviceInsertRowIdsKind::ConsumedExact => {
                None
            }
        }
    }

    /// Row identities are fixed before WAL. Dense plans upload these into their private sidecar
    /// allocation during preflight; apply still owns the IDs to prevent a caller from replacing
    /// identity authority between WAL and descriptor publication.
    fn pre_wal_payload(&self) -> Option<Vec<u8>> {
        match &self.kind {
            DeviceInsertRowIdsKind::Exact(ids) => {
                Some(ids.iter().flat_map(|id| id.to_le_bytes()).collect())
            }
            DeviceInsertRowIdsKind::SyntheticNoIdentity => None,
            DeviceInsertRowIdsKind::ConsumedExact => None,
        }
    }
}

#[derive(Clone)]
struct PreparedOpenShardIdentity {
    shard_id: u32,
    capacity: usize,
    row_count: usize,
    row_start: usize,
    gpu_id: u16,
    schema: String,
    int4_columns: Vec<String>,
    int8_columns: Vec<String>,
    numeric_columns: Vec<String>,
    bool_layouts: Vec<ResidentDeviceBoolColumnLayout>,
    text_layouts: Vec<ResidentDeviceTextColumnLayout>,
    null_layouts: Vec<ResidentDeviceNullBitmapLayout>,
    generation: Arc<()>,
    created_by_region: Option<Arc<CudaResidentDeviceMemory>>,
    row_id_region: Option<Arc<CudaResidentDeviceMemory>>,
}

impl PreparedOpenShardIdentity {
    fn from_open(open: &RelationalResidentShard) -> Self {
        Self {
            shard_id: open.shard_id,
            capacity: open.capacity,
            row_count: open.row_count,
            row_start: open.row_start,
            gpu_id: open.gpu_id,
            schema: open.schema.clone(),
            int4_columns: open.resident_device_int4_columns.clone(),
            int8_columns: open.resident_device_int8_columns.clone(),
            numeric_columns: open.resident_device_numeric_columns.clone(),
            bool_layouts: open.resident_device_bool_columns.clone(),
            text_layouts: open.resident_device_text_columns.clone(),
            null_layouts: open.resident_device_null_columns.clone(),
            generation: Arc::clone(&open.point_route_generation),
            created_by_region: open.created_by_region.clone(),
            row_id_region: open.row_id_region.clone(),
        }
    }

    fn matches(&self, open: &RelationalResidentShard, pressured: bool) -> bool {
        open.int4_appendable
            && open.is_valid(pressured)
            && open.shard_id == self.shard_id
            && open.capacity == self.capacity
            && open.row_count == self.row_count
            && open.row_start == self.row_start
            && open.gpu_id == self.gpu_id
            && open.schema == self.schema
            && open.resident_device_int4_columns == self.int4_columns
            && open.resident_device_int8_columns == self.int8_columns
            && open.resident_device_numeric_columns == self.numeric_columns
            && open.resident_device_bool_columns == self.bool_layouts
            && open.resident_device_text_columns == self.text_layouts
            && open.resident_device_null_columns == self.null_layouts
            && Arc::ptr_eq(&open.point_route_generation, &self.generation)
            && same_optional_device_region(&open.created_by_region, &self.created_by_region)
            && same_optional_device_region(&open.row_id_region, &self.row_id_region)
    }
}

fn same_optional_device_region(
    left: &Option<Arc<CudaResidentDeviceMemory>>,
    right: &Option<Arc<CudaResidentDeviceMemory>>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}

/// CREATE auto-admission deliberately publishes this one zero-capacity descriptor without
/// capacity-sized sidecars. It is not a general missing-identity escape hatch: only the sole,
/// generation-bound shard-0 sentinel may roll over into the first identity-bearing generation.
fn is_empty_bootstrap_sentinel(
    engine: &Engine,
    table: &str,
    table_shards: &[RelationalResidentShard],
    identity: &PreparedOpenShardIdentity,
) -> bool {
    table_shards.len() == 1
        && identity.shard_id == 0
        && identity.row_start == 0
        && identity.row_count == 0
        && identity.capacity == 0
        && identity.created_by_region.is_none()
        && identity.row_id_region.is_none()
        && table_shards[0].deleted_by_region.is_none()
        && engine
            .read_state
            .residency
            .shard_deleted_by_memory
            .get(&(table.to_string(), 0))
            .is_none()
        && engine
            .read_state
            .residency
            .shard_created_by_memory
            .get(&(table.to_string(), 0))
            .is_none()
        && engine
            .read_state
            .residency
            .shard_row_id_memory
            .get(&(table.to_string(), 0))
            .is_none()
}

enum PreparedResidentAppendBranch {
    InPlace(PreparedInPlaceAppend),
    FixedRollover(PreparedFixedRollover),
    DenseRollover(PreparedDenseRollover),
    ConsumedDense,
}

/// The fixed in-place branch keeps the existing descriptor's created-by Arc or a private
/// capacity-sized replacement reserved before WAL. Mutation installs only the latter.
struct PreparedInPlaceAppend {
    pending_created_by: Option<super::rollover::PendingInPlaceCreatedBy>,
}

/// A fixed-width successor generation fully reserved before WAL. The allocation owner has already
/// uploaded all immutable bytes with a zero row-count header; mutation only stamps and publishes.
pub(super) struct PreparedFixedRollover {
    pub(super) pending: super::rollover::PendingFixedResidentShard,
    pub(super) capacity: usize,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) capacity_fit_evaluations: u64,
    pub(super) budget_scan_entries: u64,
}

pub(super) enum PreparedInPlaceCreatedBy {
    Existing(Arc<CudaResidentDeviceMemory>),
    Reserved(super::rollover::PendingInPlaceCreatedBy),
}

/// Private, unpublished dense allocation set. All three CUDA allocations are created while the
/// plan owns the mutation and budget gates, before WAL. Apply may only upload/write them and hand
/// their Arcs to mutation's existing descriptor publisher.
pub(super) struct PreparedDenseRollover {
    pub(super) pending: super::rollover::PendingDenseResidentShard,
    pub(super) new_shard_id: u32,
    pub(super) new_row_start: usize,
    pub(super) budget_scan_entries: u64,
}

fn checked_rollover_coordinates(
    shard_id: u32,
    row_start: usize,
    row_count: usize,
) -> Option<(u32, usize)> {
    Some((shard_id.checked_add(1)?, row_start.checked_add(row_count)?))
}

/// Move-only, opaque pre-WAL device-append plan. `chunks` are host bytes only; device apply occurs
/// after canonical WAL/status buffering but before physical group durability and publication/ack.
pub(super) struct ResidentOpenShardAppendPlan<'a> {
    source: PreparedResidentAppendSource,
    identity: PreparedOpenShardIdentity,
    catalog_seq: Index,
    row_ids: DeviceInsertRowIds,
    bootstrap_sentinel: bool,
    branch: PreparedResidentAppendBranch,
    fixed_chunks: Option<Vec<CudaOwnedDeviceMemoryChunk>>,
    int4_min_max: Vec<(i32, i32)>,
    bool_uploads: Option<Vec<FixedWidthBoolUpload>>,
    // Field order is load-bearing: Rust drops fields in declaration order, so a failed or
    // completed apply releases its budget reservation before it releases the device gate.
    budget_allocation: Option<MutexGuard<'a, ()>>,
    // The plan crosses WAL while owning the only locks that can change its descriptor or consume
    // its sealed budget. This is a reservation, not a best-effort budget snapshot: no other
    // normal device publisher/allocation transaction can invalidate this geometry before apply.
    _device_apply: Option<MutexGuard<'a, ()>>,
}

struct FixedWidthBoolUpload {
    name: String,
    values: Box<[u8]>,
}

impl ResidentOpenShardAppendPlan<'_> {
    pub(crate) fn table_name(&self) -> &str {
        self.source.table_name()
    }

    pub(super) fn source(&self) -> &PreparedResidentAppendSource {
        &self.source
    }

    pub(super) fn row_count(&self) -> usize {
        self.source.row_count()
    }

    pub(super) fn catalog_matches(&self, table: &RelationalTable, catalog_seq: Index) -> bool {
        self.catalog_seq == catalog_seq && source_matches_table(&self.source, table)
    }

    pub(super) fn identity_matches(&self, open: &RelationalResidentShard, pressured: bool) -> bool {
        self.identity.matches(open, pressured)
    }

    pub(super) fn int4_min_max(&self) -> &[(i32, i32)] {
        &self.int4_min_max
    }

    pub(crate) fn is_bound_bootstrap_sentinel(&self) -> bool {
        self.bootstrap_sentinel
    }

    fn take_row_ids(&mut self) -> Option<Box<[u64]>> {
        self.row_ids.take_exact()
    }

    pub(super) fn holds_budget_reservation(&self) -> bool {
        self.budget_allocation.is_some()
    }

    #[cfg(test)]
    pub(super) fn proof_only_open_shard_basis(&self) -> (u32, usize, Index) {
        (
            self.identity.shard_id,
            self.identity.row_count,
            self.catalog_seq,
        )
    }

    fn sidecars_still_match(&self, engine: &Engine) -> bool {
        let table = self.source.table_name();
        let shard_id = self.identity.shard_id;
        same_optional_device_region(
            &engine
                .read_state
                .residency
                .shard_created_by_memory
                .get(&(table.to_string(), shard_id)),
            &self.identity.created_by_region,
        ) && same_optional_device_region(
            &engine
                .read_state
                .residency
                .shard_row_id_memory
                .get(&(table.to_string(), shard_id)),
            &self.identity.row_id_region,
        )
    }

    pub(super) fn is_rollover(&self) -> bool {
        matches!(
            self.branch,
            PreparedResidentAppendBranch::FixedRollover(_)
                | PreparedResidentAppendBranch::DenseRollover(_)
        )
    }

    pub(super) fn chunks_for_in_place(
        &mut self,
        capacity: usize,
        row_start: usize,
    ) -> Option<Vec<CudaOwnedDeviceMemoryChunk>> {
        if self.is_rollover()
            || self.identity.capacity != capacity
            || self.identity.row_count != row_start
        {
            return None;
        }
        self.fixed_chunks.take()
    }

    fn apply_shape_matches(&self, created_by: &AppendCreatedBy<'_>) -> bool {
        created_by.stamps_for(self.row_count()).is_some()
    }

    pub(super) fn take_bool_uploads(
        &mut self,
        layouts: &[ResidentDeviceBoolColumnLayout],
    ) -> Option<Vec<(u64, Box<[u8]>)>> {
        let uploads = self.bool_uploads.take()?;
        (uploads.len() == layouts.len()
            && uploads.iter().zip(layouts).all(|(upload, layout)| {
                upload.name == layout.name && upload.values.len() == self.row_count()
            }))
        .then(|| {
            uploads
                .into_iter()
                .zip(layouts)
                .map(|(upload, layout)| (layout.bitmap_byte_offset, upload.values))
                .collect()
        })
    }

    pub(super) fn dense_rollover_payload_len(&self) -> Option<u64> {
        match &self.branch {
            PreparedResidentAppendBranch::DenseRollover(dense) => Some(dense.pending.payload_bytes),
            _ => None,
        }
    }

    pub(super) fn fixed_rollover_payload_len(&self) -> Option<u64> {
        match &self.branch {
            PreparedResidentAppendBranch::FixedRollover(fixed) => Some(fixed.pending.payload_bytes),
            _ => None,
        }
    }

    pub(super) fn take_fixed_rollover(&mut self) -> Option<PreparedFixedRollover> {
        match std::mem::replace(
            &mut self.branch,
            PreparedResidentAppendBranch::ConsumedDense,
        ) {
            PreparedResidentAppendBranch::FixedRollover(fixed) => Some(fixed),
            branch => {
                self.branch = branch;
                None
            }
        }
    }

    pub(super) fn take_in_place_created_by(
        &mut self,
        capacity: usize,
        row_start: usize,
    ) -> Option<PreparedInPlaceCreatedBy> {
        if self.identity.capacity != capacity || self.identity.row_count != row_start {
            return None;
        }
        let PreparedResidentAppendBranch::InPlace(in_place) = &mut self.branch else {
            return None;
        };
        match in_place.pending_created_by.take() {
            Some(reserved) => Some(PreparedInPlaceCreatedBy::Reserved(reserved)),
            None => self
                .identity
                .created_by_region
                .as_ref()
                .map(|region| PreparedInPlaceCreatedBy::Existing(Arc::clone(region))),
        }
    }

    pub(super) fn take_dense_rollover(&mut self) -> Option<PreparedDenseRollover> {
        match std::mem::replace(
            &mut self.branch,
            PreparedResidentAppendBranch::ConsumedDense,
        ) {
            PreparedResidentAppendBranch::DenseRollover(dense) => Some(dense),
            branch => {
                self.branch = branch;
                None
            }
        }
    }
}

/// The only post-compile token that can reach typed device append. Its physical variant is
/// deliberately private to residency: the concurrent wave can own this opaque plan, but cannot
/// inspect, replace, or construct a resident append source.
pub(crate) struct DeviceInsertPlan<'a>(DeviceInsertPlanKind<'a>);

enum DeviceInsertPlanKind<'a> {
    ResidentOpenShardAppend(ResidentOpenShardAppendPlan<'a>),
}

impl DeviceInsertPlan<'_> {
    pub(crate) fn table_name(&self) -> &str {
        match &self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => plan.table_name(),
        }
    }

    pub(crate) fn is_bound_bootstrap_sentinel(&self) -> bool {
        match &self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => {
                plan.is_bound_bootstrap_sentinel()
            }
        }
    }

    pub(crate) fn apply(
        self,
        engine: &Engine,
        created_by: AppendCreatedBy<'_>,
    ) -> Result<(), DeviceInsertPlanApplyError> {
        match self.0 {
            DeviceInsertPlanKind::ResidentOpenShardAppend(plan) => {
                engine.apply_resident_open_shard_append(plan, created_by)
            }
        }
    }
}

impl Engine {
    /// Compile the one move-only semantic batch into the opaque device plan. Physical branches
    /// are fixed-width append/rollover and dense nullable-or-text rollover; neither adds a
    /// parallel wave carrier or publisher.
    pub(crate) fn compile_typed_insert_device_plan<'a>(
        &'a self,
        batch: TypedInsertBatch,
        row_ids: DeviceInsertRowIds,
    ) -> Result<DeviceInsertPlan<'a>, DeviceInsertPlanPrepareError> {
        let source = batch
            .into_resident_append_source()
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        self.prepare_resident_open_shard_append(source, row_ids)
            .map(|plan| DeviceInsertPlan(DeviceInsertPlanKind::ResidentOpenShardAppend(plan)))
    }

    /// Consume a sealed source and prepare its exact resident append shape before WAL. A decline is
    /// side-effect-free; the source is dropped and the caller may still take the legacy route.
    ///
    /// The caller must consume the returned plan after canonical WAL/status buffering and before
    /// physical group durability, while it owns the canonical commit boundary. The plan deliberately
    /// retains the device gate (and, when needed, allocation gate) only across that device-apply
    /// interval; it is not an asynchronous durability-tail or publication/ack handle.
    fn prepare_resident_open_shard_append<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::LiveUnindexed,
            None,
        )
    }

    fn prepare_resident_open_shard_append_core<'a>(
        &'a self,
        mut source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        mode: ResidentOpenShardAppendPreparationMode,
        preheld_device_apply: Option<MutexGuard<'a, ()>>,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        let live_unindexed = matches!(mode, ResidentOpenShardAppendPreparationMode::LiveUnindexed);
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        // A lane leader already owns this lock, but the current intentional general-path policy
        // does not export a cross-WAL plan after its leader scope ends. It declines before WAL;
        // ordinary callers retain the lock in the returned move-only plan, closing
        // descriptor/generation races through apply.
        if live_unindexed && apply_leader {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let device_apply = match (mode, preheld_device_apply) {
            (ResidentOpenShardAppendPreparationMode::LiveUnindexed, None) => Some(
                self.read_state
                    .residency
                    .mutation_gate
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ),
            #[cfg(test)]
            (ResidentOpenShardAppendPreparationMode::IndexedInPlaceProof { .. }, Some(held)) => {
                Some(held)
            }
            _ => return Err(DeviceInsertPlanPrepareError::UnsupportedShape),
        };
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(source.table_name())
            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
        let catalog_matches = if live_unindexed {
            source.prepared_catalog_seq() == catalog.commit_seq
        } else {
            source.prepared_catalog_seq() <= catalog.commit_seq
        };
        let source_matches = if live_unindexed {
            source_matches_table(&source, table)
        } else {
            #[cfg(test)]
            {
                source_matches_indexed_in_place_proof(&source, table)
            }
            #[cfg(not(test))]
            {
                false
            }
        };
        if source.row_count() == 0 || !catalog_matches || !source_matches {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        let row_ids_present = row_ids.is_exact();
        if !row_ids.exact_len_matches(source.row_count()) {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        // Dense sources are selected before any fixed-width descriptor geometry. Their fresh
        // generation owns exact text/validity layouts, while this identity only binds the OLD
        // open descriptor that will be followed by that rollover.
        let dense_rollover = source.requires_dense_rollover();
        let pressured = self
            .router
            .runtime()
            .snapshot()
            .memory_pressured_gpu_ids
            .clone();
        let (identity, gpu_id, bootstrap_sentinel) = {
            let shards = self.read_state.residency.shards.load();
            let table_shards = shards
                .get(source.table_name())
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let open = table_shards
                .last()
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
            let identity = PreparedOpenShardIdentity::from_open(open);
            // `matches` binds validity and the descriptor generation; the source checks above
            // bind the catalog. Keep the no-sidecar exception narrower than either ordinary
            // empty or ordinary missing-sidecar states, which must still decline before WAL.
            let bootstrap_sentinel =
                is_empty_bootstrap_sentinel(self, source.table_name(), table_shards, &identity);
            let source_types = source.column_types();
            let expected_i32 = source_types
                .iter()
                .filter(|ty| matches!(ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
                .count();
            let expected_i64 = source_types
                .iter()
                .filter(|ty| matches!(ty, SqlType::Int8 | SqlType::Timestamp))
                .count();
            let expected_b128 = source_types
                .iter()
                .filter(|ty| matches!(ty, SqlType::Numeric { .. } | SqlType::Uuid))
                .count();
            let expected_bool = source_types
                .iter()
                .filter(|ty| **ty == SqlType::Bool)
                .count();
            let expected_i32_names: Vec<_> = table
                .columns
                .iter()
                .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
                .map(|column| column.name.clone())
                .collect();
            let expected_i64_names: Vec<_> = table
                .columns
                .iter()
                .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
                .map(|column| column.name.clone())
                .collect();
            let expected_b128_names: Vec<_> = table
                .columns
                .iter()
                .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
                .map(|column| column.name.clone())
                .collect();
            if !identity.matches(open, pressured.contains(&open.gpu_id))
                || identity.int4_columns.len() != expected_i32
                || identity.int8_columns.len() != expected_i64
                || identity.numeric_columns.len() != expected_b128
                || identity.int4_columns != expected_i32_names
                || identity.int8_columns != expected_i64_names
                || identity.numeric_columns != expected_b128_names
                || (!dense_rollover
                    && (identity.bool_layouts.len() != expected_bool
                        || !super::rollover::ResidentRolloverPlan::descriptor_bool_layouts_match(
                            table,
                            &source_types,
                            identity.capacity,
                            &identity.bool_layouts,
                        )))
                || open.resident_device_int4_column_stats.len() != expected_i32
                || (row_ids_present && identity.row_id_region.is_none() && !bootstrap_sentinel)
                || (!row_ids_present && (identity.row_id_region.is_some() || bootstrap_sentinel))
                || identity.created_by_region.as_ref().is_some_and(|region| {
                    u64::try_from(identity.capacity)
                        .ok()
                        .and_then(|capacity| capacity.checked_mul(8))
                        .is_none_or(|required| region.metadata().allocated_bytes < required)
                })
                || identity.row_id_region.as_ref().is_some_and(|region| {
                    u64::try_from(identity.capacity)
                        .ok()
                        .and_then(|capacity| capacity.checked_mul(8))
                        .is_none_or(|required| region.metadata().allocated_bytes < required)
                })
            {
                return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
            }
            (identity, open.gpu_id, bootstrap_sentinel)
        };
        let k = source.row_count();
        #[cfg(test)]
        if matches!(
            mode,
            ResidentOpenShardAppendPreparationMode::IndexedInPlaceProof { .. }
        ) && (dense_rollover
            || bootstrap_sentinel
            || identity
                .row_count
                .checked_add(k)
                .is_none_or(|end| end > identity.capacity))
        {
            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
        }
        // NULL validity or varlen text has no capacity-strided OPEN-shard representation. It was
        // selected before descriptor geometry, so it cannot inherit an in-place arm from headroom.
        let (branch, fixed_chunks, int4_min_max, bool_uploads, budget_allocation) =
            if dense_rollover {
                let payload = source
                    .checked_dense_payload(table)
                    .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let sidecar_bytes = u64::try_from(k)
                    .ok()
                    .and_then(|rows| rows.checked_mul(std::mem::size_of::<u64>() as u64))
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let row_id_bytes = if row_ids_present { sidecar_bytes } else { 0 };
                let payload_bytes = payload
                    .device_payload_len()
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let requested_bytes = payload_bytes
                    .checked_add(sidecar_bytes)
                    .and_then(|bytes| bytes.checked_add(row_id_bytes))
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let row_id_payload = row_ids.pre_wal_payload();
                if row_ids_present != row_id_payload.is_some() {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
                let (new_shard_id, new_row_start) = checked_rollover_coordinates(
                    identity.shard_id,
                    identity.row_start,
                    identity.row_count,
                )
                .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let budget_allocation = self
                    .read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let still_matches = self
                    .read_state
                    .residency
                    .shards
                    .load()
                    .get(source.table_name())
                    .and_then(|shards| shards.last())
                    .is_some_and(|open| identity.matches(open, pressured.contains(&open.gpu_id)));
                let (resident_bytes, budget_scan_entries) =
                    self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
                let remaining_budget = match self.relational_residency_budget_bytes(gpu_id) {
                    Some(budget) => match budget.checked_sub(resident_bytes) {
                        Some(remaining) => Some(remaining),
                        None => {
                            self.read_state
                                .residency
                                .rollover_budget_declines
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            return Err(if bootstrap_sentinel {
                                DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                            } else {
                                DeviceInsertPlanPrepareError::UnsupportedShape
                            });
                        }
                    },
                    None => None,
                };
                if !still_matches
                    || remaining_budget.is_some_and(|remaining| requested_bytes > remaining)
                {
                    self.read_state
                        .residency
                        .rollover_budget_declines
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    });
                }
                let pending = super::rollover::PendingDenseResidentShard::reserve_pre_wal(
                    self,
                    gpu_id,
                    payload,
                    sidecar_bytes,
                    row_id_payload,
                )
                .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
                if remaining_budget.is_some_and(|remaining| pending.allocation_bytes > remaining) {
                    self.read_state
                        .residency
                        .rollover_budget_declines
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    });
                }
                (
                    PreparedResidentAppendBranch::DenseRollover(PreparedDenseRollover {
                        pending,
                        new_shard_id,
                        new_row_start,
                        budget_scan_entries,
                    }),
                    None,
                    Vec::new(),
                    None,
                    Some(budget_allocation),
                )
            } else if !bootstrap_sentinel
                && identity
                    .row_count
                    .checked_add(k)
                    .is_some_and(|end| end <= identity.capacity)
            {
                let chunks = source
                    .checked_append_chunks(identity.capacity, identity.row_count)
                    .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
                #[cfg(test)]
                let index_scratch_bytes = match mode {
                    ResidentOpenShardAppendPreparationMode::LiveUnindexed => 0,
                    ResidentOpenShardAppendPreparationMode::IndexedInPlaceProof {
                        index_scratch_bytes,
                    } => index_scratch_bytes,
                };
                #[cfg(not(test))]
                let index_scratch_bytes = 0;
                let reserve_created_by = identity.created_by_region.is_none();
                // The proof mode always carries the budget guard, including when the append has
                // an existing sidecar.  Its exact pooled index preparation footprint is charged
                // alongside any append-sidecar reservation before the latter allocates.
                let retain_budget_guard = reserve_created_by || !live_unindexed;
                let (pending_created_by, budget_allocation) = if retain_budget_guard {
                    let budget_allocation = self
                        .read_state
                        .residency
                        .budget_allocation_lock
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let still_matches = self
                        .read_state
                        .residency
                        .shards
                        .load()
                        .get(source.table_name())
                        .and_then(|shards| shards.last())
                        .is_some_and(|open| {
                            identity.matches(open, pressured.contains(&open.gpu_id))
                        });
                    let sidecar_bytes = if reserve_created_by {
                        u64::try_from(identity.capacity)
                            .ok()
                            .and_then(|capacity| {
                                capacity.checked_mul(std::mem::size_of::<u64>() as u64)
                            })
                            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?
                    } else {
                        0
                    };
                    let required_before_allocation = sidecar_bytes
                        .checked_add(index_scratch_bytes)
                        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                    let (resident_bytes, _) =
                        self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
                    let remaining_budget = match self.relational_residency_budget_bytes(gpu_id) {
                        Some(budget) => match budget.checked_sub(resident_bytes) {
                            Some(remaining) => Some(remaining),
                            None => return Err(DeviceInsertPlanPrepareError::UnsupportedShape),
                        },
                        None => None,
                    };
                    if !still_matches
                        || remaining_budget
                            .is_some_and(|remaining| required_before_allocation > remaining)
                    {
                        return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                    }
                    let pending_created_by = if reserve_created_by {
                        // Reserve and allocate the capacity-sized created_by sidecar before WAL.
                        // Device apply only installs this exact Arc and stamps its still-invisible slots.
                        let pending = super::rollover::PendingInPlaceCreatedBy::reserve_pre_wal(
                            self,
                            gpu_id,
                            identity.capacity,
                        )
                        .map_err(|_| DeviceInsertPlanPrepareError::UnsupportedShape)?;
                        let actual_required = pending
                            .allocation_bytes
                            .checked_add(index_scratch_bytes)
                            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                        if pending.capacity_bytes != sidecar_bytes
                            || pending.allocation_bytes < sidecar_bytes
                            || remaining_budget.is_some_and(|remaining| actual_required > remaining)
                        {
                            return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                        }
                        Some(pending)
                    } else {
                        None
                    };
                    (pending_created_by, Some(budget_allocation))
                } else {
                    (None, None)
                };
                (
                    PreparedResidentAppendBranch::InPlace(PreparedInPlaceAppend {
                        pending_created_by,
                    }),
                    Some(chunks),
                    source
                        .int4_min_max()
                        .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?,
                    Some(
                        build_bool_uploads(&source, table)
                            .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?,
                    ),
                    budget_allocation,
                )
            } else {
                #[cfg(test)]
                if matches!(
                    mode,
                    ResidentOpenShardAppendPreparationMode::IndexedInPlaceProof { .. }
                ) {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
                // Rollover capacity is an allocation transaction even though this pre-WAL phase makes
                // no allocation. Snapshot the exact budget under the same lock the publisher rechecks.
                let budget_allocation = self
                    .read_state
                    .residency
                    .budget_allocation_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let still_matches = self
                    .read_state
                    .residency
                    .shards
                    .load()
                    .get(source.table_name())
                    .and_then(|shards| shards.last())
                    .is_some_and(|open| identity.matches(open, pressured.contains(&open.gpu_id)));
                if !still_matches {
                    return Err(if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    });
                }
                let (resident_bytes, budget_scan_entries) =
                    self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
                let remaining_budget = match self.relational_residency_budget_bytes(gpu_id) {
                    Some(budget) => match budget.checked_sub(resident_bytes) {
                        Some(remaining) => Some(remaining),
                        None => {
                            self.read_state
                                .residency
                                .rollover_budget_declines
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            return Err(if bootstrap_sentinel {
                                DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                            } else {
                                DeviceInsertPlanPrepareError::UnsupportedShape
                            });
                        }
                    },
                    None => None,
                };
                let desired = super::rollover::ResidentRolloverPlan::fixed_width_desired_capacity(
                    k,
                    Some(self.shard_size_target()),
                )
                .ok_or(if bootstrap_sentinel {
                    DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                } else {
                    DeviceInsertPlanPrepareError::UnsupportedShape
                })?;
                let types = source.column_types();
                let rollover = match super::rollover::ResidentRolloverPlan::fixed_width_null_free(
                    table,
                    &types,
                    k,
                    desired,
                    row_ids_present,
                    false,
                    remaining_budget,
                ) {
                    Ok(Some(plan)) => plan,
                    Ok(None) => {
                        self.read_state
                            .residency
                            .rollover_budget_declines
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Err(if bootstrap_sentinel {
                            DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                        } else {
                            DeviceInsertPlanPrepareError::UnsupportedShape
                        });
                    }
                    Err(_) => {
                        self.read_state
                            .residency
                            .rollover_budget_declines
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Err(if bootstrap_sentinel {
                            DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                        } else {
                            DeviceInsertPlanPrepareError::UnsupportedShape
                        });
                    }
                };
                let chunks = source
                    .checked_append_chunks(rollover.capacity(), 0)
                    .map_err(|_| {
                        if bootstrap_sentinel {
                            DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                        } else {
                            DeviceInsertPlanPrepareError::UnsupportedShape
                        }
                    })?;
                let int4_min_max = source
                    .int4_min_max()
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let expected_int4_stats = table
                    .columns
                    .iter()
                    .filter(|column| {
                        matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date)
                    })
                    .count();
                if int4_min_max.len() != expected_int4_stats {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
                let int4_stats = table
                    .columns
                    .iter()
                    .filter(|column| {
                        matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date)
                    })
                    .zip(int4_min_max)
                    .map(|(column, (min, max))| ResidentDeviceInt4ColumnStats {
                        name: column.name.clone(),
                        min,
                        max,
                    })
                    .collect();
                let bool_uploads = build_bool_uploads(&source, table)
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let bool_uploads = bool_uploads
                    .into_iter()
                    .zip(rollover.bool_layouts())
                    .map(|(upload, layout)| {
                        (upload.name == layout.name)
                            .then_some((layout.bitmap_byte_offset, upload.values))
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or(DeviceInsertPlanPrepareError::UnsupportedShape)?;
                let row_id_payload = row_ids.pre_wal_payload();
                if row_ids_present != row_id_payload.is_some() {
                    return Err(DeviceInsertPlanPrepareError::UnsupportedShape);
                }
                let (new_shard_id, new_row_start) = checked_rollover_coordinates(
                    identity.shard_id,
                    identity.row_start,
                    identity.row_count,
                )
                .ok_or(if bootstrap_sentinel {
                    DeviceInsertPlanPrepareError::RetryableBoundBootstrapState
                } else {
                    DeviceInsertPlanPrepareError::UnsupportedShape
                })?;
                let pending = super::rollover::PendingFixedResidentShard::reserve_pre_wal(
                    self,
                    gpu_id,
                    &rollover,
                    k,
                    chunks,
                    bool_uploads,
                    int4_stats,
                    row_id_payload,
                )
                .map_err(|_| {
                    if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    }
                })?;
                if remaining_budget.is_some_and(|remaining| pending.allocation_bytes > remaining) {
                    self.read_state
                        .residency
                        .rollover_budget_declines
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(if bootstrap_sentinel {
                        DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                    } else {
                        DeviceInsertPlanPrepareError::UnsupportedShape
                    });
                }
                (
                    PreparedResidentAppendBranch::FixedRollover(PreparedFixedRollover {
                        pending,
                        capacity: rollover.capacity(),
                        new_shard_id,
                        new_row_start,
                        capacity_fit_evaluations: rollover.capacity_scan_entries(),
                        budget_scan_entries,
                    }),
                    None,
                    Vec::new(),
                    None,
                    Some(budget_allocation),
                )
            };
        Ok(ResidentOpenShardAppendPlan {
            source,
            identity,
            catalog_seq: catalog.commit_seq,
            row_ids,
            bootstrap_sentinel,
            branch,
            fixed_chunks,
            int4_min_max,
            bool_uploads,
            budget_allocation,
            _device_apply: device_apply,
        })
    }

    /// Apply exactly one pre-WAL plan. Any failure is fatal: a caller that has already written WAL
    /// must wedge/recover instead of falling back to a second write path.
    fn apply_resident_open_shard_append(
        &self,
        mut plan: ResidentOpenShardAppendPlan<'_>,
        created_by: AppendCreatedBy<'_>,
    ) -> Result<(), DeviceInsertPlanApplyError> {
        if !plan.apply_shape_matches(&created_by) {
            return Err(DeviceInsertPlanApplyError::ShapeDrift);
        }
        if !plan_matches_live_open(self, &plan) {
            return Err(DeviceInsertPlanApplyError::PlanDrift);
        }
        let row_ids = plan.take_row_ids();
        let table = plan.source().table_name().to_string();
        self.try_append_to_resident_open_shard(
            &table,
            ResidentAppendSource::DevicePlan(&mut plan),
            created_by,
            row_ids.as_deref(),
        )
        .then_some(())
        .ok_or(DeviceInsertPlanApplyError::PublisherFailure)
    }

    /// Thin cfg(test) adapter over the one resident append preparation core.  It never creates a
    /// `DeviceInsertPlan`; the returned normal append plan is owned only by the inert proof wrapper.
    #[cfg(test)]
    pub(super) fn prepare_resident_open_shard_append_indexed_in_place_proof<'a>(
        &'a self,
        source: PreparedResidentAppendSource,
        row_ids: DeviceInsertRowIds,
        device_apply: MutexGuard<'a, ()>,
        index_scratch_bytes: u64,
    ) -> Result<ResidentOpenShardAppendPlan<'a>, DeviceInsertPlanPrepareError> {
        self.prepare_resident_open_shard_append_core(
            source,
            row_ids,
            ResidentOpenShardAppendPreparationMode::IndexedInPlaceProof {
                index_scratch_bytes,
            },
            Some(device_apply),
        )
    }
}

fn source_matches_table(source: &PreparedResidentAppendSource, table: &RelationalTable) -> bool {
    source.table_name() == table.name
        && source.exact_single_table_dependency()
        && source.table_oid() == table.oid
        && table.indexes.is_empty()
        && table.foreign_keys.is_empty()
        && crate::engine_transaction_reset::table_schema_digest(table)
            .is_ok_and(|digest| digest == source.schema_digest())
        && table.columns.len() == source.columns().len()
        && table
            .columns
            .iter()
            .zip(source.columns())
            .all(|(column, source_column)| {
                is_live_resident_append_type(column.ty)
                    // Scalar defaults were already materialized in the sealed vectors, and
                    // metadata-only domain bindings were revalidated by the prepared plan before
                    // row identities/device compilation. They do not change resident layout.
                    && column.table_oid == table.oid
                    && column.id == source_column.column_id()
                    && column.attnum == source_column.attnum()
                    && column.ty == source_column.ty()
                    && column.type_oid == source_column.type_oid()
                    && column.type_size == source_column.type_size()
                    && source_column.ty() == column.ty
            })
}

/// The cfg(test) proof can inspect an already-published indexed table while preserving the
/// production predicate above verbatim.  It deliberately shares every non-index source/catalog
/// witness with the live route, then lets the proof owner establish complete raw-index coverage.
#[cfg(test)]
fn source_matches_indexed_in_place_proof(
    source: &PreparedResidentAppendSource,
    table: &RelationalTable,
) -> bool {
    source.table_name() == table.name
        && source.exact_single_table_dependency()
        && source.table_oid() == table.oid
        && table.foreign_keys.is_empty()
        && crate::engine_transaction_reset::table_schema_digest(table)
            .is_ok_and(|digest| digest == source.schema_digest())
        && table.columns.len() == source.columns().len()
        && table
            .columns
            .iter()
            .zip(source.columns())
            .all(|(column, source_column)| {
                is_live_resident_append_type(column.ty)
                    && column.table_oid == table.oid
                    && column.id == source_column.column_id()
                    && column.attnum == source_column.attnum()
                    && column.ty == source_column.ty()
                    && column.type_oid == source_column.type_oid()
                    && column.type_size == source_column.type_size()
                    && source_column.ty() == column.ty
            })
}

fn is_live_resident_append_type(ty: SqlType) -> bool {
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
            | SqlType::Text
    )
}

fn build_bool_uploads(
    source: &PreparedResidentAppendSource,
    table: &RelationalTable,
) -> Option<Vec<FixedWidthBoolUpload>> {
    let staged = source.bool_uploads()?;
    staged
        .into_iter()
        .map(|PreparedResidentBoolUpload { column_id, values }| {
            table
                .columns
                .iter()
                .find(|column| column.id == column_id && column.ty == SqlType::Bool)
                .map(|column| FixedWidthBoolUpload {
                    name: column.name.clone(),
                    values,
                })
        })
        .collect()
}

fn plan_matches_live_open(engine: &Engine, plan: &ResidentOpenShardAppendPlan) -> bool {
    let catalog = engine.catalog_snapshot();
    let Some(table) = catalog.relational_catalog.get(plan.source().table_name()) else {
        return false;
    };
    if !plan.catalog_matches(table, catalog.commit_seq) {
        return false;
    }
    let pressured = engine
        .router
        .runtime()
        .snapshot()
        .memory_pressured_gpu_ids
        .clone();
    plan.sidecars_still_match(engine)
        && engine
            .read_state
            .residency
            .shards
            .load()
            .get(plan.source().table_name())
            .and_then(|shards| shards.last())
            .is_some_and(|open| plan.identity_matches(open, pressured.contains(&open.gpu_id)))
}

#[cfg(test)]
mod ownership_tests {
    #[test]
    fn indexed_proof_mode_reuses_the_single_append_preparation_core() {
        let source = include_str!("fixed_insert.rs")
            .split("\n#[cfg(test)]\nmod ownership_tests")
            .next()
            .expect("implementation precedes tests");
        let core = source
            .split("fn prepare_resident_open_shard_append_core")
            .nth(1)
            .and_then(|section| {
                section
                    .split("\n    /// Apply exactly one pre-WAL plan")
                    .next()
            })
            .expect("one append preparation core");
        assert!(core.contains("LiveUnindexed"));
        assert!(core.contains("IndexedInPlaceProof"));
        assert!(core.contains("source_matches_table(&source, table)"));
        assert!(core.contains("source_matches_indexed_in_place_proof(&source, table)"));
        let proof_decline = core
            .find("&& (dense_rollover")
            .expect("proof mode rejects dense/bootstrap/headroom shapes");
        for later in [
            ".checked_dense_payload(table)",
            "fixed_width_desired_capacity",
            "PendingInPlaceCreatedBy::reserve_pre_wal",
            "budget_allocation_lock",
        ] {
            assert!(
                proof_decline < core.find(later).expect("ordinary later preparation branch"),
                "proof mode must decline before {later}"
            );
        }
        assert!(
            ["dense_rollover", "bootstrap_sentinel", ".checked_add(k)", "end > identity.capacity"]
                .into_iter()
                .all(|decline| core[proof_decline..].contains(decline)),
            "proof mode must explicitly decline every fixed/bootstrap/dense shape before allocation"
        );
        let adapter = source
            .split("fn prepare_resident_open_shard_append_indexed_in_place_proof")
            .nth(1)
            .and_then(|section| section.split("\n}\n\nfn source_matches_table").next())
            .expect("proof adapter");
        assert!(adapter.contains("prepare_resident_open_shard_append_core"));
        assert!(!adapter.contains("let catalog ="));
        assert!(!adapter.contains("reserve_pre_wal"));
    }
    #[test]
    fn rollover_coordinates_are_checked_before_allocation() {
        assert_eq!(
            super::checked_rollover_coordinates(7, 11, 13),
            Some((8, 24))
        );
        assert_eq!(super::checked_rollover_coordinates(u32::MAX, 0, 0), None);
        assert_eq!(super::checked_rollover_coordinates(7, usize::MAX, 1), None);
    }
    #[test]
    fn typed_post_wal_paths_cannot_allocate_a_replacement_generation() {
        let mutation = include_str!("mutation.rs");
        let fixed_apply = mutation
            .split("} else if preallocated_fixed_plan {")
            .nth(1)
            .and_then(|section| {
                section
                    .split("} else if !has_text && !batch_has_null {")
                    .next()
            })
            .expect("typed fixed post-WAL apply section");
        assert!(fixed_apply.contains("finish_post_wal"));
        for forbidden in [
            "relational_residency_device_memory",
            "retain_device_memory_",
            "PendingResidentShard::build",
            "fixed_width_desired_capacity",
        ] {
            assert!(
                !fixed_apply.contains(forbidden),
                "typed fixed post-WAL apply must not {forbidden}"
            );
        }

        let rollover = include_str!("rollover.rs");
        let fixed_finish = rollover
            .split("fn finish_post_wal")
            .nth(1)
            .and_then(|section| section.split("impl PendingInPlaceCreatedBy").next())
            .expect("fixed post-WAL finalization");
        for forbidden in [
            "relational_residency_device_memory",
            "retain_device_memory_",
        ] {
            assert!(
                !fixed_finish.contains(forbidden),
                "fixed post-WAL finalization must not {forbidden}"
            );
        }

        let typed_in_place = mutation
            .split("let typed_created_by_region = match &mut source {")
            .nth(1)
            .and_then(|section| section.split("let fused = if").next())
            .expect("typed in-place sidecar handoff");
        assert!(
            typed_in_place.contains("pending.into_region")
                && typed_in_place.contains("get_or_alloc_created_by_region"),
            "mutation must consume and publish the sealed in-place Arc"
        );
        assert!(!typed_in_place.contains("install_preallocated_created_by_region"));
        let pending_in_place = rollover
            .split("impl PendingInPlaceCreatedBy")
            .nth(1)
            .and_then(|section| section.split("impl PendingResidentShard").next())
            .expect("in-place allocation lifecycle");
        for forbidden in [
            "with_shards_mut",
            ".insert_shard(",
            "shard_created_by_memory",
        ] {
            assert!(
                !pending_in_place.contains(forbidden),
                "rollover lifecycle leaf must not publish {forbidden}"
            );
        }
    }

    #[test]
    fn adapter_has_no_second_residency_or_durability_publisher() {
        let source = include_str!("fixed_insert.rs");
        let implementation = source
            .split("#[cfg(test)]\nmod ownership_tests")
            .next()
            .expect("source has an implementation prefix");
        for (prefix, suffix) in [
            ("RelationalResident", "Shard {"),
            ("with_shards_mut", "_for_table("),
            (".insert_", "shard("),
            ("retain_device_", "memory_"),
            (".append_owned_", "chunks("),
            ("write_", "wal"),
            ("append_", "wal"),
        ] {
            let forbidden = format!("{prefix}{suffix}");
            assert!(
                !implementation.contains(&forbidden),
                "fixed INSERT adapter must not own {forbidden}"
            );
        }
        for forbidden in [
            "relational_residency_device_memory",
            "retain_device_memory_",
            "shard_created_by_memory.insert",
            "shard_row_id_memory.insert",
        ] {
            assert!(
                !implementation.contains(forbidden),
                "fixed INSERT adapter must delegate {forbidden} to the allocation/publisher owner"
            );
        }
        assert!(implementation.contains("ResidentAppendSource::DevicePlan"));
        assert_eq!(
            implementation
                .matches("try_append_to_resident_open_shard")
                .count(),
            1,
            "the typed plan must enter mutation through exactly one publisher"
        );
        assert!(implementation.contains("_device_apply"));
        assert!(implementation.contains("budget_allocation"));
        assert!(
            !implementation.contains("pub(crate) fn new"),
            "the move-only plan must have no raw public constructor"
        );
        assert!(
            implementation.contains("row_ids: DeviceInsertRowIds"),
            "the plan must own its row-id input before WAL"
        );
        assert!(
            !implementation.contains("row_ids_present: bool"),
            "a caller-provided row-id presence bit must not cross the WAL/apply boundary"
        );
        let apply = implementation
            .split("fn apply_resident_open_shard_append")
            .nth(1)
            .expect("sealed apply exists")
            .split("#[cfg(test)]")
            .next()
            .expect("sealed apply precedes test-only convenience");
        assert!(
            !apply.contains("row_ids:"),
            "apply must consume only row IDs sealed into the opaque plan"
        );
        assert!(
            source.contains("#[cfg(test)]\n    pub(crate) fn synthetic_no_identity"),
            "synthetic no-identity construction must remain test-only"
        );
    }
}
