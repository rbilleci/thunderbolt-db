//! Sealed logical-input forms for the single resident append publisher.

use super::*;

/// Both variants describe the same catalog-ordered logical rows; only their encoder ownership
/// differs. Publication, device locks, indexes, sidecars, and descriptors remain in `mutation`.
pub(super) enum ResidentAppendSource<'source, 'plan> {
    Rows(&'source [Vec<SqlValue>]),
    DevicePlan(&'source mut super::fixed_insert::ResidentOpenShardAppendPlan<'plan>),
}

pub(super) enum ResidentAppendI32Columns<'a> {
    Rows(Vec<Vec<i32>>),
    Fixed(Vec<&'a [i32]>),
}

impl ResidentAppendI32Columns<'_> {
    pub(super) fn slices(&self) -> Vec<&[i32]> {
        match self {
            Self::Rows(columns) => columns.iter().map(Vec::as_slice).collect(),
            Self::Fixed(columns) => columns.clone(),
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
    ) -> Result<Vec<CudaOwnedDeviceMemoryChunk>, ExecuteError> {
        match self {
            Self::Rows(rows) => {
                compute_open_shard_int4_append_chunks(column_types, capacity, row_start, rows)
            }
            Self::DevicePlan(plan) => {
                let source = plan.source();
                if column_types != source.column_types().as_slice() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "sealed typed append no longer matches its fixed-width descriptor"
                            .to_string(),
                    )));
                }
                plan.chunks_for_in_place(capacity, row_start)
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
                    .i32_column_slices()
                    .map(ResidentAppendI32Columns::Fixed)
            }
        }
    }

    pub(super) fn int4_min_max(&self, column_types: &[SqlType]) -> Option<Vec<(i32, i32)>> {
        match self {
            Self::Rows(rows) => Some(
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
            ),
            Self::DevicePlan(plan) => (column_types == plan.source().column_types().as_slice())
                .then(|| plan.int4_min_max().to_vec()),
        }
    }
}
