//! Sealed logical-input forms for the single resident append publisher.

use super::*;
use crate::typed_insert_batch::{PreparedResidentAppendColumn, PreparedResidentFixedChunk};

/// Both variants describe the same catalog-ordered logical rows; only their encoder ownership
/// differs. Publication, device locks, indexes, sidecars, and descriptors remain in `mutation`.
pub(super) enum ResidentAppendSource<'source, 'plan> {
    Rows(&'source [Vec<SqlValue>]),
    DevicePlan(&'source mut super::fixed_insert::ResidentOpenShardAppendPlan<'plan>),
}

pub(super) enum ResidentAppendI32Columns<'a> {
    Rows(Vec<Vec<i32>>),
    /// The typed source's pre-WAL boxed columns are the only fused descriptors.  Borrow them
    /// directly instead of allocating a `Vec<&[i32]>` at apply time.
    Fixed(&'a [PreparedResidentAppendColumn]),
}

pub(super) enum ResidentAppendInt4MinMax<'a> {
    Rows(Vec<(i32, i32)>),
    Fixed(&'a [(i32, i32)]),
}

impl ResidentAppendInt4MinMax<'_> {
    pub(super) fn as_slice(&self) -> &[(i32, i32)] {
        match self {
            Self::Rows(values) => values,
            Self::Fixed(values) => values,
        }
    }
}

/// The fixed typed plan keeps exact boxed chunks through WAL.  Legacy rows retain their existing
/// growable encoder vector.  Both variants hand owned bytes directly to the CUDA upload iterator
/// without rebuilding an outer `Vec` at the apply boundary.
pub(super) enum ResidentAppendChunks {
    Rows(Vec<CudaOwnedDeviceMemoryChunk>),
    Prepared {
        chunks: Box<[PreparedResidentFixedChunk]>,
        offsets: Box<[u64]>,
    },
}

pub(super) enum ResidentAppendChunkOffsets<'a> {
    Rows(Vec<u64>),
    Prepared(&'a [u64]),
}

impl ResidentAppendChunkOffsets<'_> {
    pub(super) fn as_slice(&self) -> &[u64] {
        match self {
            Self::Rows(offsets) => offsets,
            Self::Prepared(offsets) => offsets,
        }
    }
}

pub(super) enum ResidentAppendPayloadChunks {
    Rows(Vec<CudaOwnedDeviceMemoryChunk>),
    Prepared(Vec<PreparedResidentFixedChunk>),
}

impl ResidentAppendChunks {
    pub(super) fn first_offsets(&self, count: usize) -> Option<ResidentAppendChunkOffsets<'_>> {
        match self {
            Self::Rows(chunks) => Some(ResidentAppendChunkOffsets::Rows(
                chunks
                    .iter()
                    .take(count)
                    .map(|chunk| chunk.byte_offset)
                    .collect(),
            )),
            Self::Prepared { offsets, .. } => (offsets.len() >= count)
                .then(|| ResidentAppendChunkOffsets::Prepared(&offsets[..count])),
        }
    }

    pub(super) fn split_final_header(
        self,
    ) -> Option<(CudaOwnedDeviceMemoryChunk, ResidentAppendPayloadChunks)> {
        match self {
            Self::Rows(mut chunks) => chunks
                .pop()
                .map(|header| (header, ResidentAppendPayloadChunks::Rows(chunks))),
            // `Vec::from(Box<[T]>)` adopts the box allocation; it does not reserve or copy.  The
            // typed plan's backing remains exact and the following upload maps each owner lazily.
            Self::Prepared { chunks, .. } => {
                let mut chunks = Vec::from(chunks);
                chunks.pop().map(|header| {
                    (
                        CudaOwnedDeviceMemoryChunk {
                            byte_offset: header.byte_offset,
                            bytes: header.bytes.into_vec(),
                        },
                        ResidentAppendPayloadChunks::Prepared(chunks),
                    )
                })
            }
        }
    }
}

impl ResidentAppendPayloadChunks {
    pub(super) fn append_to(
        self,
        memory: &CudaResidentDeviceMemory,
    ) -> Result<u64, gpu_db_execution::CudaRuntimeProbeError> {
        match self {
            Self::Rows(chunks) => memory.append_owned_chunks(chunks),
            Self::Prepared(chunks) => memory.append_owned_chunks(chunks.into_iter().map(|chunk| {
                CudaOwnedDeviceMemoryChunk {
                    byte_offset: chunk.byte_offset,
                    bytes: chunk.bytes.into_vec(),
                }
            })),
        }
    }
}

impl ResidentAppendI32Columns<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Rows(columns) => columns.len(),
            Self::Fixed(columns) => columns.len(),
        }
    }

    pub(super) fn all_i32_len(&self, rows: usize) -> bool {
        match self {
            Self::Rows(columns) => columns.iter().all(|column| column.len() == rows),
            Self::Fixed(columns) => columns.iter().all(|column| {
                column
                    .i32_values()
                    .is_some_and(|values| values.len() == rows)
            }),
        }
    }

    pub(super) fn flatten_i32(&self) -> Option<Vec<i32>> {
        match self {
            Self::Rows(columns) => Some(
                columns
                    .iter()
                    .flat_map(|column| column.iter())
                    .copied()
                    .collect(),
            ),
            Self::Fixed(columns) => {
                let mut values = Vec::new();
                for column in *columns {
                    values.extend_from_slice(column.i32_values()?);
                }
                Some(values)
            }
        }
    }

    pub(super) fn rows(&self) -> Option<&[Vec<i32>]> {
        match self {
            Self::Rows(columns) => Some(columns),
            Self::Fixed(_) => None,
        }
    }
}

impl ResidentAppendSource<'_, '_> {
    pub(super) fn row_count(&self) -> usize {
        match self {
            Self::Rows(rows) => rows.len(),
            Self::DevicePlan(plan) => plan.row_count(),
        }
    }

    pub(super) fn rows(&self) -> Option<&[Vec<SqlValue>]> {
        match self {
            Self::Rows(rows) => Some(rows),
            Self::DevicePlan(_) => None,
        }
    }

    pub(super) fn has_null(&self) -> bool {
        match self {
            Self::Rows(rows) => rows
                .iter()
                .any(|row| row.iter().any(|value| matches!(value, SqlValue::Null))),
            Self::DevicePlan(plan) => plan.source().requires_dense_rollover(),
        }
    }

    pub(super) fn requires_dense_rollover(&self) -> bool {
        match self {
            Self::Rows(rows) => rows.iter().any(|row| {
                row.iter().any(|value| matches!(value, SqlValue::Null))
                    || row.iter().any(|value| matches!(value, SqlValue::Text(_)))
            }),
            Self::DevicePlan(plan) => plan.source().requires_dense_rollover(),
        }
    }

    pub(super) fn resets_existing_rows(&self) -> bool {
        match self {
            Self::Rows(_) => false,
            Self::DevicePlan(plan) => plan.resets_existing_rows(),
        }
    }

    /// Recheck a sealed source at the publisher boundary: DDL may commit after its adapter check.
    pub(super) fn matches_current_catalog(
        &self,
        table: &RelationalTable,
        catalog_seq: Index,
    ) -> bool {
        match self {
            Self::Rows(_) => true,
            Self::DevicePlan(plan) => plan.catalog_matches(table, catalog_seq),
        }
    }

    /// A sealed plan is bound to one concrete OPEN descriptor, including its generation Arc.
    /// Legacy row input deliberately remains descriptor-agnostic because its caller may still
    /// re-admit after an ordinary pre-WAL decline.
    pub(super) fn matches_open_descriptor(
        &self,
        open: &RelationalResidentShard,
        pressured: bool,
    ) -> bool {
        match self {
            Self::Rows(_) => true,
            Self::DevicePlan(plan) => plan.identity_matches(open, pressured),
        }
    }

    /// Only the sealed plan can retain an allocation reservation across WAL. The legacy source
    /// owns no such guard and continues to take the allocation mutex locally.
    pub(super) fn holds_budget_reservation(&self) -> bool {
        match self {
            Self::Rows(_) => false,
            Self::DevicePlan(plan) => plan.holds_budget_reservation(),
        }
    }

    pub(super) fn append_chunks(
        &mut self,
        column_types: &[SqlType],
        capacity: usize,
        row_start: usize,
    ) -> Result<ResidentAppendChunks, ExecuteError> {
        match self {
            Self::Rows(rows) => {
                compute_open_shard_int4_append_chunks(column_types, capacity, row_start, rows)
                    .map(ResidentAppendChunks::Rows)
            }
            Self::DevicePlan(plan) => {
                let source = plan.source();
                if !source.column_types_match(column_types) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "sealed typed append no longer matches its fixed-width descriptor"
                            .to_string(),
                    )));
                }
                plan.chunks_for_in_place(capacity, row_start)
                    .map(|owners| ResidentAppendChunks::Prepared {
                        chunks: owners.chunks,
                        offsets: owners.offsets,
                    })
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "sealed typed append plan no longer matches the in-place branch"
                                .to_string(),
                        ))
                    })
            }
        }
    }

    pub(super) fn i32_columns(&self, column_count: usize) -> Option<ResidentAppendI32Columns<'_>> {
        match self {
            Self::Rows(rows) => Some(ResidentAppendI32Columns::Rows(
                (0..column_count)
                    .map(|column| {
                        rows.iter()
                            .map(|row| sql_value_as_int4(&row[column]))
                            .collect()
                    })
                    .collect(),
            )),
            Self::DevicePlan(plan) => {
                let source = plan.source();
                if source.columns().len() != column_count {
                    return None;
                }
                source
                    .columns()
                    .iter()
                    .all(|column| column.i32_values().is_some())
                    .then_some(ResidentAppendI32Columns::Fixed(source.columns()))
            }
        }
    }

    pub(super) fn int4_min_max(
        &self,
        column_types: &[SqlType],
    ) -> Option<ResidentAppendInt4MinMax<'_>> {
        match self {
            Self::Rows(rows) => Some(ResidentAppendInt4MinMax::Rows(
                (0..column_types.len())
                    .filter(|&column| {
                        matches!(
                            column_types[column],
                            SqlType::Int4 | SqlType::Date | SqlType::Int2
                        )
                    })
                    .map(|column| {
                        rows.iter().fold((i32::MAX, i32::MIN), |(min, max), row| {
                            let value = sql_value_as_int4(&row[column]);
                            (min.min(value), max.max(value))
                        })
                    })
                    .collect(),
            )),
            Self::DevicePlan(plan) => plan
                .source()
                .column_types_match(column_types)
                .then(|| ResidentAppendInt4MinMax::Fixed(plan.int4_min_max())),
        }
    }
}
