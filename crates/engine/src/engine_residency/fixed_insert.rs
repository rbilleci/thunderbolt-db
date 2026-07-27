//! Pre-WAL planning for the accepted INSERT-001 fixed-i32 residency route.
//!
//! This module owns no CUDA allocation, device write, descriptor publication, side-map mutation,
//! or WAL. It consumes a sealed source into an opaque plan, then the plan re-enters mutation's one
//! publisher at apply time. A post-WAL apply failure is fatal rather than permission to fall back.

use super::append_source::ResidentAppendSource;
use super::*;
use crate::prepared_insert_batch::PreparedI32AppendSource;
use std::sync::MutexGuard;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparedI32AppendPrepareError {
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

/// A post-WAL device-apply failure is terminal before physical group durability; the active wave
/// wedges with durable count unchanged rather than attempting legacy re-application or publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparedI32AppendApplyError {
    PlanDrift,
    ShapeDrift,
    PublisherFailure,
}

/// Exact durable row identities sealed into a typed append before WAL. Construction is limited to
/// owned IDs; the synthetic form is test-only so a production cutover cannot accidentally lose a
/// bound entity-identity sidecar.
pub(crate) struct PreparedI32AppendRowIds {
    kind: PreparedI32AppendRowIdsKind,
}

enum PreparedI32AppendRowIdsKind {
    Exact(Box<[u64]>),
    SyntheticNoIdentity,
    ConsumedExact,
}

impl PreparedI32AppendRowIds {
    pub(crate) fn exact(ids: Box<[u64]>) -> Self {
        Self {
            kind: PreparedI32AppendRowIdsKind::Exact(ids),
        }
    }

    #[cfg(test)]
    pub(crate) fn synthetic_no_identity() -> Self {
        Self {
            kind: PreparedI32AppendRowIdsKind::SyntheticNoIdentity,
        }
    }

    fn is_exact(&self) -> bool {
        matches!(
            &self.kind,
            PreparedI32AppendRowIdsKind::Exact(_) | PreparedI32AppendRowIdsKind::ConsumedExact
        )
    }

    fn exact_len_matches(&self, rows: usize) -> bool {
        match &self.kind {
            PreparedI32AppendRowIdsKind::Exact(ids) => ids.len() == rows,
            PreparedI32AppendRowIdsKind::SyntheticNoIdentity => true,
            PreparedI32AppendRowIdsKind::ConsumedExact => false,
        }
    }

    fn take_exact(&mut self) -> Option<Box<[u64]>> {
        match std::mem::replace(
            &mut self.kind,
            PreparedI32AppendRowIdsKind::SyntheticNoIdentity,
        ) {
            PreparedI32AppendRowIdsKind::Exact(ids) => {
                self.kind = PreparedI32AppendRowIdsKind::ConsumedExact;
                Some(ids)
            }
            PreparedI32AppendRowIdsKind::SyntheticNoIdentity
            | PreparedI32AppendRowIdsKind::ConsumedExact => None,
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
            && open.resident_device_int8_columns.is_empty()
            && open.resident_device_numeric_columns.is_empty()
            && open.resident_device_bool_columns.is_empty()
            && open.resident_device_text_columns.is_empty()
            && open.resident_device_null_columns.is_empty()
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

enum PreparedI32AppendBranch {
    InPlace,
    Rollover(super::rollover::ResidentRolloverPlan),
}

/// Move-only, opaque pre-WAL device-append plan. `chunks` are host bytes only; device apply occurs
/// after canonical WAL/status buffering but before physical group durability and publication/ack.
pub(crate) struct PreparedI32OpenShardAppendPlan<'a> {
    source: PreparedI32AppendSource,
    identity: PreparedOpenShardIdentity,
    catalog_seq: Index,
    row_ids: PreparedI32AppendRowIds,
    bootstrap_sentinel: bool,
    branch: PreparedI32AppendBranch,
    chunks: Option<Vec<CudaOwnedDeviceMemoryChunk>>,
    int4_min_max: Vec<(i32, i32)>,
    // Field order is load-bearing: Rust drops fields in declaration order, so a failed or
    // completed apply releases its budget reservation before it releases the device gate.
    budget_allocation: Option<MutexGuard<'a, ()>>,
    // The plan crosses WAL while owning the only locks that can change its descriptor or consume
    // its sealed budget. This is a reservation, not a best-effort budget snapshot: no other
    // normal device publisher/allocation transaction can invalidate this geometry before apply.
    _device_apply: Option<MutexGuard<'a, ()>>,
}

impl PreparedI32OpenShardAppendPlan<'_> {
    pub(crate) fn table_name(&self) -> &str {
        self.source.table_name()
    }

    pub(super) fn source(&self) -> &PreparedI32AppendSource {
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

    pub(super) fn row_ids_required(&self) -> bool {
        self.row_ids.is_exact()
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
        matches!(self.branch, PreparedI32AppendBranch::Rollover(_))
    }

    pub(super) fn rollover_plan(&self) -> Option<&super::rollover::ResidentRolloverPlan> {
        match &self.branch {
            PreparedI32AppendBranch::InPlace => None,
            PreparedI32AppendBranch::Rollover(plan) => Some(plan),
        }
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
        self.chunks.take()
    }

    pub(super) fn chunks_for_rollover(
        &mut self,
        capacity: usize,
    ) -> Option<Vec<CudaOwnedDeviceMemoryChunk>> {
        (self
            .rollover_plan()
            .is_some_and(|plan| plan.capacity() == capacity))
        .then(|| self.chunks.take())
        .flatten()
    }

    fn apply_shape_matches(&self, created_by: &AppendCreatedBy<'_>) -> bool {
        created_by.stamps_for(self.row_count()).is_some()
    }
}

impl Engine {
    /// Consume a sealed source and prepare its exact resident append shape before WAL. A decline is
    /// side-effect-free; the source is dropped and the caller may still take the legacy route.
    ///
    /// The caller must consume the returned plan after canonical WAL/status buffering and before
    /// physical group durability, while it owns the canonical commit boundary. The plan deliberately
    /// retains the device gate (and, when needed, allocation gate) only across that device-apply
    /// interval; it is not an asynchronous durability-tail or publication/ack handle.
    pub(crate) fn prepare_prepared_i32_open_shard_append<'a>(
        &'a self,
        source: PreparedI32AppendSource,
        row_ids: PreparedI32AppendRowIds,
    ) -> Result<PreparedI32OpenShardAppendPlan<'a>, PreparedI32AppendPrepareError> {
        let apply_leader =
            crate::resident_storage::LANE_APPLY_LEADER_ACTIVE.with(std::cell::Cell::get);
        // A lane leader already owns this lock, but the current intentional general-path policy
        // does not export a cross-WAL plan after its leader scope ends. It declines before WAL;
        // ordinary callers retain the lock in the returned move-only plan, closing
        // descriptor/generation races through apply.
        if apply_leader {
            return Err(PreparedI32AppendPrepareError::UnsupportedShape);
        }
        let device_apply = Some(
            self.read_state
                .residency
                .mutation_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(source.table_name())
            .ok_or(PreparedI32AppendPrepareError::UnsupportedShape)?;
        if source.row_count() == 0
            || source.prepared_catalog_seq() != catalog.commit_seq
            || !source_matches_table(&source, table)
        {
            return Err(PreparedI32AppendPrepareError::UnsupportedShape);
        }
        let row_ids_present = row_ids.is_exact();
        if !row_ids.exact_len_matches(source.row_count()) {
            return Err(PreparedI32AppendPrepareError::UnsupportedShape);
        }
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
                .ok_or(PreparedI32AppendPrepareError::UnsupportedShape)?;
            let open = table_shards
                .last()
                .ok_or(PreparedI32AppendPrepareError::UnsupportedShape)?;
            let identity = PreparedOpenShardIdentity::from_open(open);
            // `matches` binds validity and the descriptor generation; the source checks above
            // bind the catalog. Keep the no-sidecar exception narrower than either ordinary
            // empty or ordinary missing-sidecar states, which must still decline before WAL.
            let bootstrap_sentinel =
                is_empty_bootstrap_sentinel(self, source.table_name(), table_shards, &identity);
            if !identity.matches(open, pressured.contains(&open.gpu_id))
                || identity.int4_columns.len() != source.columns().len()
                || open.resident_device_int4_column_stats.len() != source.columns().len()
                || (row_ids_present && identity.row_id_region.is_none() && !bootstrap_sentinel)
                || (!row_ids_present && (identity.row_id_region.is_some() || bootstrap_sentinel))
                || identity.created_by_region.as_ref().is_some_and(|region| {
                    region.metadata().allocated_bytes
                        < (identity.capacity as u64)
                            .saturating_mul(std::mem::size_of::<u64>() as u64)
                })
                || identity.row_id_region.as_ref().is_some_and(|region| {
                    region.metadata().allocated_bytes
                        < (identity.capacity as u64)
                            .saturating_mul(std::mem::size_of::<u64>() as u64)
                })
            {
                return Err(PreparedI32AppendPrepareError::UnsupportedShape);
            }
            (identity, open.gpu_id, bootstrap_sentinel)
        };
        let columns: Vec<&[i32]> = source
            .columns()
            .iter()
            .map(|column| column.values())
            .collect();
        let int4_min_max = columns
            .iter()
            .map(|column| {
                column
                    .iter()
                    .copied()
                    .fold((i32::MAX, i32::MIN), |(min, max), value| {
                        (min.min(value), max.max(value))
                    })
            })
            .collect();
        let k = source.row_count();
        let (branch, chunks, budget_allocation) = if !bootstrap_sentinel
            && identity
                .row_count
                .checked_add(k)
                .is_some_and(|end| end <= identity.capacity)
        {
            let chunks = compute_open_shard_i32_column_append_chunks(
                identity.capacity,
                identity.row_count,
                &columns,
            )
            .map_err(|_| PreparedI32AppendPrepareError::UnsupportedShape)?;
            let budget_allocation = if identity.created_by_region.is_none() {
                // The ordinary append publisher lazily installs a capacity-sized created_by
                // sidecar. Reserve its exact preflight charge before canonical WAL/status buffering, so
                // device apply cannot discover a first-sidecar budget miss before group durability or ack.
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
                let sidecar_bytes = u64::try_from(identity.capacity)
                    .ok()
                    .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<u64>() as u64))
                    .ok_or(PreparedI32AppendPrepareError::UnsupportedShape)?;
                let (resident_bytes, _) =
                    self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
                if !still_matches
                    || self
                        .relational_residency_budget_bytes(gpu_id)
                        .is_some_and(|budget| resident_bytes.saturating_add(sidecar_bytes) > budget)
                {
                    return Err(PreparedI32AppendPrepareError::UnsupportedShape);
                }
                Some(budget_allocation)
            } else {
                None
            };
            (PreparedI32AppendBranch::InPlace, chunks, budget_allocation)
        } else {
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
                    PreparedI32AppendPrepareError::RetryableBoundBootstrapState
                } else {
                    PreparedI32AppendPrepareError::UnsupportedShape
                });
            }
            let (resident_bytes, _) = self.relational_resident_bytes_and_entries_for_gpu(gpu_id);
            let remaining_budget = self
                .relational_residency_budget_bytes(gpu_id)
                .map(|budget| budget.saturating_sub(resident_bytes));
            let desired = super::rollover::ResidentRolloverPlan::fixed_width_desired_capacity(
                k,
                Some(self.shard_size_target()),
            )
            .ok_or(if bootstrap_sentinel {
                PreparedI32AppendPrepareError::RetryableBoundBootstrapState
            } else {
                PreparedI32AppendPrepareError::UnsupportedShape
            })?;
            let types = vec![SqlType::Int4; columns.len()];
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
                        PreparedI32AppendPrepareError::RetryableBoundBootstrapResource
                    } else {
                        PreparedI32AppendPrepareError::UnsupportedShape
                    });
                }
                Err(_) => {
                    self.read_state
                        .residency
                        .rollover_budget_declines
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(if bootstrap_sentinel {
                        PreparedI32AppendPrepareError::RetryableBoundBootstrapState
                    } else {
                        PreparedI32AppendPrepareError::UnsupportedShape
                    });
                }
            };
            let chunks =
                compute_open_shard_i32_column_append_chunks(rollover.capacity(), 0, &columns)
                    .map_err(|_| {
                        if bootstrap_sentinel {
                            PreparedI32AppendPrepareError::RetryableBoundBootstrapState
                        } else {
                            PreparedI32AppendPrepareError::UnsupportedShape
                        }
                    })?;
            (
                PreparedI32AppendBranch::Rollover(rollover),
                chunks,
                Some(budget_allocation),
            )
        };
        Ok(PreparedI32OpenShardAppendPlan {
            source,
            identity,
            catalog_seq: catalog.commit_seq,
            row_ids,
            bootstrap_sentinel,
            branch,
            chunks: Some(chunks),
            int4_min_max,
            budget_allocation,
            _device_apply: device_apply,
        })
    }

    /// Apply exactly one pre-WAL plan. Any failure is fatal: a caller that has already written WAL
    /// must wedge/recover instead of falling back to a second write path.
    pub(crate) fn apply_prepared_i32_open_shard_append(
        &self,
        mut plan: PreparedI32OpenShardAppendPlan<'_>,
        created_by: AppendCreatedBy<'_>,
    ) -> Result<(), PreparedI32AppendApplyError> {
        if !plan.apply_shape_matches(&created_by) {
            return Err(PreparedI32AppendApplyError::ShapeDrift);
        }
        if !plan_matches_live_open(self, &plan) {
            return Err(PreparedI32AppendApplyError::PlanDrift);
        }
        let row_ids = plan.take_row_ids();
        let table = plan.source().table_name().to_string();
        self.try_append_to_resident_open_shard(
            &table,
            ResidentAppendSource::FixedI32Plan(&mut plan),
            created_by,
            row_ids.as_deref(),
        )
        .then_some(())
        .ok_or(PreparedI32AppendApplyError::PublisherFailure)
    }

    #[cfg(test)]
    #[allow(dead_code)] // crate-visible direct-route apply entry retained for focused ownership tests.
    pub(crate) fn try_append_prepared_i32_open_shard(
        &self,
        source: PreparedI32AppendSource,
        created_by: AppendCreatedBy<'_>,
        row_ids: PreparedI32AppendRowIds,
    ) -> bool {
        let Ok(plan) = self.prepare_prepared_i32_open_shard_append(source, row_ids) else {
            return false;
        };
        self.apply_prepared_i32_open_shard_append(plan, created_by)
            .is_ok()
    }
}

fn source_matches_table(source: &PreparedI32AppendSource, table: &RelationalTable) -> bool {
    source.table_name() == table.name
        && source.exact_single_table_dependency()
        && source.table_oid() == table.oid
        && table.indexes.is_empty()
        && table.check_constraints.is_empty()
        && table.foreign_keys.is_empty()
        && crate::engine_transaction_reset::table_schema_digest(table)
            .is_ok_and(|digest| digest == source.schema_digest())
        && table.columns.len() == source.columns().len()
        && table
            .columns
            .iter()
            .zip(source.columns())
            .all(|(column, source_column)| {
                column.ty == SqlType::Int4
                    && column.domain.is_none()
                    && column.default.is_none()
                    && column.table_oid == table.oid
                    && column.id == source_column.column_id()
                    && column.attnum == source_column.attnum()
                    && column.ty == source_column.ty()
                    && column.type_oid == source_column.type_oid()
                    && column.type_size == source_column.type_size()
                    && source_column.values().len() == source.row_count()
            })
}

fn plan_matches_live_open(engine: &Engine, plan: &PreparedI32OpenShardAppendPlan) -> bool {
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
        assert!(implementation.contains("ResidentAppendSource::FixedI32Plan"));
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
            implementation.contains("row_ids: PreparedI32AppendRowIds"),
            "the plan must own its row-id input before WAL"
        );
        assert!(
            !implementation.contains("row_ids_present: bool"),
            "a caller-provided row-id presence bit must not cross the WAL/apply boundary"
        );
        let apply = implementation
            .split("pub(crate) fn apply_prepared_i32_open_shard_append")
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
