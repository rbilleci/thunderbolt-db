//! Physical-index enrollment and private GPU lookup verification for indexed rollover proof.
//!
//! The private build stores only directory words, so launch/status evidence alone cannot prove
//! that a destination received its intended raw key or ordered compound fingerprint. This leaf
//! derives needles from the sealed typed batch, probes every physical destination on the GPU, and
//! reduces each locate result to scalar evidence before control returns to the owner.

#![allow(dead_code)] // part of the deliberately unreachable physical reservation

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::engine_insert_plan::host_retention::{HostRetentionGeometry, HostRetentionReport};
use crate::relational_model::RelationalTable;
use crate::typed_insert_batch::PreparedResidentAppendSource;
use crate::{Engine, ExecuteError};
use gpu_db_execution::{
    multi_shard_i32_write_locate_resource_geometry, resident_index_allocated_bytes,
    CudaAllocationScope, CudaResidentDeviceMemory, CudaWriteLocateResourceGeometry,
    WriteLocateShard,
};

#[cfg(test)]
use super::ReservedPrivateIndexGeneration;
use super::{PreparedFixedRolloverLogicalBindings, PreparedPrivateIndexGeneration};

struct ExpectedPhysicalLookup {
    raw_ordinal: usize,
    key_id: usize,
    needles: Box<[i32]>,
}

/// Private host-side plan for one GPU lookup per distinct destination. It never becomes part of
/// the proof owner, so typed values and GPU result vectors cannot cross the inspection boundary.
pub(super) struct PrivateGpuLookupOracle {
    destinations: Box<[ExpectedPhysicalLookup]>,
    geometry: CudaWriteLocateResourceGeometry,
}

/// Scalar evidence retained after every private GPU lookup result has been checked and dropped.
pub(super) struct PrivateGpuLookupEvidence {
    pub(super) gpu_probe_count: usize,
    pub(super) bounded_readback_bytes: u64,
    pub(super) max_concurrent_readback_bytes: u64,
    pub(super) host_call_peak: HostRetentionGeometry,
}

impl PrivateGpuLookupOracle {
    pub(super) fn max_concurrent_scratch_bytes(&self) -> u64 {
        self.geometry.preparation_bytes
    }

    pub(super) fn incoming_rows(&self) -> usize {
        self.destinations
            .first()
            .map_or(0, |destination| destination.needles.len())
    }

    pub(super) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), ExecuteError> {
        report.retain_boxed_slice(&self.destinations)?;
        for destination in self.destinations.iter() {
            report.retain_boxed_slice(&destination.needles)?;
        }
        Ok(())
    }

    pub(super) fn host_retention_geometry(&self) -> Result<HostRetentionGeometry, ExecuteError> {
        lookup_oracle_host_retention_geometry(self.destinations.len(), self.incoming_rows())
    }
}

pub(super) fn lookup_oracle_host_retention_geometry(
    physical_index_count: usize,
    incoming_rows: usize,
) -> Result<HostRetentionGeometry, ExecuteError> {
    if physical_index_count == 0 || incoming_rows == 0 {
        return Err(super::decline(
            "indexed rollover proof GPU lookup host geometry is empty",
        ));
    }
    let mut geometry = HostRetentionGeometry::default();
    geometry.checked_add_backing_elements::<ExpectedPhysicalLookup>(
        physical_index_count,
        "indexed rollover lookup destination box",
    )?;
    let needle_bytes = u64::try_from(physical_index_count)
        .ok()
        .and_then(|indexes| {
            u64::try_from(incoming_rows)
                .ok()
                .and_then(|rows| indexes.checked_mul(rows))
        })
        .and_then(|values| values.checked_mul(std::mem::size_of::<i32>() as u64))
        .ok_or_else(|| super::decline("indexed rollover lookup needle bytes overflow"))?;
    geometry.checked_add_backing_bytes_slots(
        needle_bytes,
        u64::try_from(physical_index_count)
            .map_err(|_| super::decline("indexed rollover lookup needle slots overflow"))?,
        "indexed rollover lookup needle boxes",
    )?;
    Ok(geometry)
}

/// Derive one raw/folded needle vector for every distinct physical destination before the sealed
/// source moves into the private payload owner. This bounded test-only proof deliberately declines
/// shapes that cannot expose exact i32-section words instead of introducing a host fallback.
pub(super) fn prepare_gpu_lookup_oracle(
    table: &RelationalTable,
    logical: &PreparedFixedRolloverLogicalBindings,
    source: &PreparedResidentAppendSource,
) -> Result<PrivateGpuLookupOracle, ExecuteError> {
    let columns = source.columns();
    let row_count = source.row_count();
    let max_hits = u32::try_from(row_count).map_err(|_| {
        super::decline("indexed rollover proof GPU lookup oracle row count overflows")
    })?;
    if row_count == 0
        || columns.len() != table.columns.len()
        || columns.iter().any(|column| {
            column
                .i32_values()
                .is_none_or(|values| values.len() != row_count)
        })
    {
        return Err(super::decline(
            "indexed rollover proof GPU lookup oracle batch geometry drifted",
        ));
    }
    let geometry = multi_shard_i32_write_locate_resource_geometry(1, row_count, max_hits)
        .ok_or_else(|| {
            super::decline("indexed rollover proof GPU lookup oracle resource geometry overflows")
        })?;
    let mut destinations = Vec::with_capacity(logical.physical.len());
    for physical in logical.physical.iter() {
        let index = table.indexes.get(physical.raw_ordinal).ok_or_else(|| {
            super::decline("indexed rollover proof GPU lookup oracle lost physical index binding")
        })?;
        if index.key_columns.is_empty()
            || index.key_columns.iter().any(|name| {
                table
                    .columns
                    .iter()
                    .position(|column| column.name == *name)
                    .is_none_or(|position| position >= columns.len())
            })
        {
            return Err(super::decline(
                "indexed rollover proof GPU lookup oracle key-column geometry drifted",
            ));
        }
        let uses_fingerprint = super::super::index_uses_fingerprint(table, index);
        let mut needles = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let value = if uses_fingerprint {
                let mut fingerprint = 0x811C_9DC5_u32;
                for name in index.key_columns.iter() {
                    let position = table
                        .columns
                        .iter()
                        .position(|column| column.name == *name)
                        .ok_or_else(|| {
                            super::decline(
                                "indexed rollover proof GPU lookup key column disappeared",
                            )
                        })?;
                    let word = columns[position]
                        .i32_values()
                        .and_then(|values| values.get(row))
                        .copied()
                        .ok_or_else(|| {
                            super::decline(
                                "indexed rollover proof GPU lookup value geometry drifted",
                            )
                        })?;
                    fingerprint ^= word as u32;
                    fingerprint = fingerprint.wrapping_mul(0x0100_0193);
                    fingerprint = fingerprint.rotate_left(13).wrapping_add(0x9E37_79B1);
                }
                fingerprint as i32
            } else if index.key_columns.len() == 1 {
                let position = table
                    .columns
                    .iter()
                    .position(|column| column.name == index.key_columns[0])
                    .ok_or_else(|| {
                        super::decline(
                            "indexed rollover proof GPU lookup raw key column disappeared",
                        )
                    })?;
                columns[position]
                    .i32_values()
                    .and_then(|values| values.get(row))
                    .copied()
                    .ok_or_else(|| {
                        super::decline("indexed rollover proof GPU lookup value geometry drifted")
                    })?
            } else {
                unreachable!("compound bindings must use the fingerprint directory")
            };
            needles.push(value);
        }
        destinations.push(ExpectedPhysicalLookup {
            raw_ordinal: physical.raw_ordinal,
            key_id: physical.key_id,
            needles: needles.into(),
        });
    }
    if destinations.is_empty() {
        return Err(super::decline(
            "indexed rollover proof GPU lookup oracle has no physical destination",
        ));
    }
    Ok(PrivateGpuLookupOracle {
        destinations: destinations.into(),
        geometry,
    })
}

/// GPU-probe every private directory against its own intended raw key or ordered compound
/// fingerprint. The write-locate result remains local to this leaf; only verified scalar counts
/// reach the rollover ledger.
pub(super) fn verify_private_gpu_lookups(
    source: &Arc<CudaResidentDeviceMemory>,
    generations: &[PreparedPrivateIndexGeneration],
    oracle: &PrivateGpuLookupOracle,
) -> Result<PrivateGpuLookupEvidence, ExecuteError> {
    if generations.len() != oracle.destinations.len() {
        return Err(super::decline(
            "indexed rollover proof GPU lookup destination count drifted",
        ));
    }
    let mut host_call_peak = HostRetentionGeometry::default();
    for (generation, expected) in generations.iter().zip(oracle.destinations.iter()) {
        if generation.raw_ordinal != expected.raw_ordinal
            || generation.key_id != expected.key_id
            || expected.needles.is_empty()
        {
            return Err(super::decline(
                "indexed rollover proof GPU lookup binding drifted",
            ));
        }
        let row_count = u32::try_from(expected.needles.len())
            .map_err(|_| super::decline("indexed rollover proof GPU lookup row count overflows"))?;
        CudaAllocationScope::ensure_available(oracle.geometry.preparation_bytes).map_err(
            |error| {
                super::decline(format!(
                    "indexed rollover proof GPU lookup scratch is unavailable: {error}"
                ))
            },
        )?;
        let result = source
            .submit_multi_shard_i32_write_locate(
                &[WriteLocateShard {
                    index: Arc::clone(&generation.memory),
                    table_mask: generation.table_mask,
                    hash_shift: generation.hash_shift,
                    row_count,
                }],
                &expected.needles,
                row_count,
            )
            .map_err(|error| {
                super::decline(format!("indexed rollover proof GPU lookup failed: {error}"))
            })?;
        let mut result_report = HostRetentionReport::default();
        result_report.retain_vec(&result.shard_idx)?;
        result_report.retain_vec(&result.slot)?;
        result_report.retain_vec(&result.count)?;
        let result_geometry = result_report.geometry()?;
        if result.host_peak.bytes != oracle.geometry.host_peak_bytes
            || result.host_peak.allocation_slots != oracle.geometry.host_peak_allocation_slots
            || result.host_peak.bytes < result_geometry.retained_bytes()
            || result.host_peak.allocation_slots < result_geometry.allocation_slots()
        {
            return Err(super::decline(
                "indexed rollover proof GPU lookup host peak drifted",
            ));
        }
        let mut call_geometry = HostRetentionGeometry::default();
        call_geometry.checked_add_backing_bytes_slots(
            result.host_peak.bytes,
            result.host_peak.allocation_slots,
            "indexed rollover proof execution lookup host peak",
        )?;
        host_call_peak = host_call_peak.peak(call_geometry);
        if result.max_hits != row_count
            || result.count.len() != expected.needles.len()
            || result.shard_idx.len() != expected.needles.len() * row_count as usize
            || result.slot.len() != result.shard_idx.len()
        {
            return Err(super::decline(
                "indexed rollover proof GPU lookup result geometry drifted",
            ));
        }
        for (expected_slot, count) in result.count.iter().copied().enumerate() {
            let count = usize::try_from(count)
                .map_err(|_| super::decline("indexed rollover proof GPU lookup count overflows"))?;
            let window_start = expected_slot
                .checked_mul(row_count as usize)
                .ok_or_else(|| {
                    super::decline("indexed rollover proof GPU lookup window overflows")
                })?;
            let window_end = window_start
                .checked_add(count)
                .filter(|end| *end <= result.slot.len())
                .ok_or_else(|| super::decline("indexed rollover proof GPU lookup count drifted"))?;
            if count == 0
                || count > row_count as usize
                || !result.shard_idx[window_start..window_end]
                    .iter()
                    .zip(&result.slot[window_start..window_end])
                    .any(|(&shard, &slot)| shard == 0 && slot == expected_slot as u32)
            {
                return Err(super::decline(
                    "indexed rollover proof GPU lookup missed its intended physical destination",
                ));
            }
        }
    }
    let gpu_probe_count = generations.len();
    let bounded_readback_bytes = oracle
        .geometry
        .readback_bytes
        .checked_mul(
            u64::try_from(gpu_probe_count)
                .map_err(|_| super::decline("indexed rollover proof GPU lookup count overflows"))?,
        )
        .ok_or_else(|| super::decline("indexed rollover proof GPU lookup readback overflows"))?;
    Ok(PrivateGpuLookupEvidence {
        gpu_probe_count,
        bounded_readback_bytes,
        max_concurrent_readback_bytes: oracle.geometry.readback_bytes,
        host_call_peak,
    })
}

/// Match publication's route -> cache -> coverage -> complete -> publications order so this is
/// one exact witness, not an O(1) marker detached from the physical predecessor generation.
pub(super) fn published_index_enrollment_is_complete(
    engine: &Engine,
    table: &RelationalTable,
) -> bool {
    let _route_publish = engine
        .read_state
        .residency
        .sharded_point_route_publish_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let cache = engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let coverage = engine
        .read_state
        .residency
        .named_index_coverage
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let complete = engine
        .read_state
        .residency
        .named_index_coverage_complete
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let publications = engine
        .read_state
        .residency
        .named_index_publications
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut physical_keys = BTreeSet::new();
    if !table
        .indexes
        .iter()
        .enumerate()
        .all(|(raw_ordinal, index)| {
            match super::super::index_probe_key_id(table, index, raw_ordinal) {
                Some(key_id) => {
                    physical_keys.insert(key_id);
                    true
                }
                None => false,
            }
        })
    {
        return false;
    }
    let shards = engine.read_state.residency.shards.load();
    let physical_coverage_is_exact = shards.get(&table.name).is_some_and(|shards| {
        shards
            .iter()
            .filter(|shard| shard.row_count != 0)
            .all(|shard| {
                let Some(payload) = shard.device_memory.as_ref() else {
                    return false;
                };
                physical_keys.iter().all(|key_id| {
                    let key = (table.name.clone(), shard.shard_id, *key_id);
                    let Some(table_size) = super::super::resident_shard_index_table_size(
                        shard.row_count as u64,
                        shard.capacity as u64,
                    ) else {
                        return false;
                    };
                    let Ok(table_mask) = u32::try_from(table_size - 1) else {
                        return false;
                    };
                    let hash_shift = 32 - table_size.trailing_zeros();
                    let Some(allocated_bytes) = resident_index_allocated_bytes(
                        table_mask,
                        shard.capacity.max(shard.row_count) as u64,
                    ) else {
                        return false;
                    };
                    coverage.get(&key) == Some(&(payload.device_ptr(), shard.row_count))
                        && cache.get(&key).is_some_and(|entry| {
                            entry.resident_device_ptr == payload.device_ptr()
                                && entry.row_count == shard.row_count
                                && entry.published_row_count.load(Ordering::Acquire)
                                    == shard.row_count
                                && Arc::ptr_eq(&entry._resident_guard, payload)
                                && entry.device_index.as_ref().is_some_and(|index| {
                                    index.device_ptr() != payload.device_ptr()
                                        && entry.table_mask == table_mask
                                        && entry.hash_shift == hash_shift
                                        && index.metadata().allocated_bytes == allocated_bytes
                                        && entry.published_has_postings.load(Ordering::Acquire)
                                            == entry.has_postings
                                })
                        })
                })
            })
    });
    physical_coverage_is_exact
        && publications.get(&table.oid) == Some(&table.indexes)
        && complete
            .get(&table.name)
            .is_some_and(|(oid, indexes)| *oid == table.oid && indexes == &table.indexes)
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum PrivateIndexBuildSabotage {
    SwapFirstTwoDirectories,
    ReverseCompoundDescriptors,
}

#[cfg(test)]
thread_local! {
    static PRIVATE_INDEX_BUILD_SABOTAGE: std::cell::Cell<Option<PrivateIndexBuildSabotage>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) fn arm_private_index_build_sabotage(sabotage: PrivateIndexBuildSabotage) {
    PRIVATE_INDEX_BUILD_SABOTAGE.with(|slot| slot.set(Some(sabotage)));
}

/// Corrupt only the private test build input. The oracle still derives its expected semantics from
/// the sealed catalog/batch carrier, so either corruption must be rejected before inspection.
#[cfg(test)]
pub(super) fn apply_test_build_sabotage(
    reserved: &mut [ReservedPrivateIndexGeneration],
) -> Result<(), ExecuteError> {
    let Some(sabotage) = PRIVATE_INDEX_BUILD_SABOTAGE.with(|slot| slot.replace(None)) else {
        return Ok(());
    };
    match sabotage {
        PrivateIndexBuildSabotage::SwapFirstTwoDirectories => {
            let (first, rest) = reserved.split_first_mut().ok_or_else(|| {
                super::decline("indexed rollover proof sabotage needs two physical directories")
            })?;
            let second = rest.first_mut().ok_or_else(|| {
                super::decline("indexed rollover proof sabotage needs two physical directories")
            })?;
            if first.build_columns.len() != second.build_columns.len() {
                return Err(super::decline(
                    "indexed rollover proof sabotage needs matching descriptor geometry",
                ));
            }
            std::mem::swap(&mut first.build_columns, &mut second.build_columns);
        }
        PrivateIndexBuildSabotage::ReverseCompoundDescriptors => {
            let generation = reserved
                .iter_mut()
                .find(|generation| {
                    generation
                        .build_columns
                        .iter()
                        .filter(|column| {
                            !matches!(
                                column,
                                gpu_db_execution::CudaCompoundFoldColumn::Validity { .. }
                            )
                        })
                        .count()
                        >= 2
                })
                .ok_or_else(|| {
                    super::decline("indexed rollover proof sabotage needs a compound directory")
                })?;
            let data_columns = generation
                .build_columns
                .iter()
                .position(|column| {
                    matches!(
                        column,
                        gpu_db_execution::CudaCompoundFoldColumn::Validity { .. }
                    )
                })
                .unwrap_or(generation.build_columns.len());
            generation.build_columns[..data_columns].reverse();
        }
    }
    Ok(())
}

/// Reconstruct the exact bounded readback contract from the survivor-only scalar dimensions.
pub(super) fn expected_resource_ledger(
    physical_index_count: usize,
    incoming_rows: usize,
) -> Option<(u64, u64)> {
    let max_hits = u32::try_from(incoming_rows).ok()?;
    let lookup = multi_shard_i32_write_locate_resource_geometry(1, incoming_rows, max_hits)?;
    let count = u64::try_from(physical_index_count).ok()?;
    let build_readback = count.checked_mul(std::mem::size_of::<u32>() as u64)?;
    let lookup_readback = lookup.readback_bytes.checked_mul(count)?;
    Some((
        build_readback.checked_add(lookup_readback)?,
        (std::mem::size_of::<u32>() as u64).max(lookup.readback_bytes),
    ))
}
