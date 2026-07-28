//! Resolved INSERT input semantics.
//!
//! This is the one non-owning boundary between SQL cells and the move-only typed batch. It
//! resolves source-list order into catalog order exactly once while retaining source provenance
//! and stable statement/row/column ordinals. It intentionally does not evaluate defaults,
//! sequences, domains, or constraints; later plan operators consume these states rather than
//! attempting to infer them from scalar vectors.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct InsertStatementOrdinal(u32);

impl InsertStatementOrdinal {
    /// Existing one-statement ingress has one stable statement position. Transaction-overlay
    /// preparation will supply its actual operation ordinal when it adopts this boundary.
    pub(crate) const FIRST: Self = Self(0);
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InsertSourceOrdinal {
    pub(crate) statement: InsertStatementOrdinal,
    pub(crate) row: u32,
    pub(crate) column: u32,
}

/// The pre-default state of one catalog-order cell. `ProvidedNull`, `Omitted`, and
/// `ExplicitDefault` deliberately do not share one invalid/absent bit: only a later resolved
/// default/sequence operator is allowed to collapse their SQL meanings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolvedInsertInput<'a> {
    Provided {
        value: &'a SqlValue,
        provenance: gpu_db_sql::InsertValueProvenance,
    },
    ProvidedNull {
        provenance: gpu_db_sql::InsertValueProvenance,
    },
    Omitted,
    ExplicitDefault {
        provenance: gpu_db_sql::InsertDefaultProvenance,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedInsertCell<'a> {
    pub(crate) input: ResolvedInsertInput<'a>,
}

/// A catalog-order target column with one semantic state per statement row.
pub(crate) struct ResolvedInsertColumn<'a> {
    pub(crate) catalog_ordinal: u32,
    pub(crate) column: &'a RelationalColumn,
    pub(crate) source_column_ordinal: Option<u32>,
    pub(crate) cells: Box<[ResolvedInsertCell<'a>]>,
}

/// Borrowed, resolved INSERT semantics. It never owns a second row-major value representation:
/// values still belong to the parsed/bound [`Insert`] until `TypedInsertBatch` compiles them into
/// its single move-only column vectors.
pub(crate) struct ResolvedInsertSemantics<'a> {
    pub(crate) table: &'a RelationalTable,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) row_count: u32,
    pub(crate) columns: Box<[ResolvedInsertColumn<'a>]>,
}

impl<'a> ResolvedInsertSemantics<'a> {
    pub(crate) fn resolve(
        insert: &'a Insert,
        table: &'a RelationalTable,
        statement_ordinal: InsertStatementOrdinal,
    ) -> Result<Self, ExecuteError> {
        let target_indexes = resolve_target_columns(insert, table)?;
        let row_count = u32::try_from(insert.rows.len()).map_err(|_| {
            EngineError::Durability("typed INSERT row count exceeds u32".to_string())
        })?;

        for row in &insert.rows {
            // Preserve the established diagnostic precedence: every row's arity is checked
            // before a later cell is considered for coercion/default lowering.
            if row.len() != target_indexes.len() {
                return Err(EngineError::ApplyFailed(
                    "INSERT value count must match target columns".to_string(),
                )
                .into());
            }
        }

        let source_by_catalog = target_indexes
            .iter()
            .enumerate()
            .map(|(source, target)| u32::try_from(source).map(|source| (*target, source)))
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map_err(|_| {
                EngineError::Durability("INSERT source column ordinal exceeds u32".to_string())
            })?;
        let rows = usize::try_from(row_count).expect("u32 always fits usize on supported hosts");
        let mut cells_by_catalog = (0..table.columns.len())
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();

        for row in &insert.rows {
            for (source_index, target_index) in target_indexes.iter().copied().enumerate() {
                let cell = resolve_cell(&row[source_index])?;
                cells_by_catalog[target_index].push(cell);
            }
        }

        let mut columns = Vec::with_capacity(table.columns.len());
        for (catalog_index, column) in table.columns.iter().enumerate() {
            let catalog_ordinal = u32::try_from(catalog_index).map_err(|_| {
                EngineError::Durability("INSERT catalog column ordinal exceeds u32".to_string())
            })?;
            let source_column_ordinal = source_by_catalog.get(&catalog_index).copied();
            let cells = match source_column_ordinal {
                Some(_) => std::mem::take(&mut cells_by_catalog[catalog_index]),
                None => (0..row_count)
                    .map(|_| ResolvedInsertCell {
                        input: ResolvedInsertInput::Omitted,
                    })
                    .collect(),
            };
            if cells.len() != rows {
                return Err(EngineError::Durability(
                    "resolved INSERT semantic column lost row alignment".to_string(),
                )
                .into());
            }
            columns.push(ResolvedInsertColumn {
                catalog_ordinal,
                column,
                source_column_ordinal,
                cells: cells.into(),
            });
        }

        Ok(Self {
            table,
            statement_ordinal,
            row_count,
            columns: columns.into(),
        })
    }
}

fn resolve_cell<'a>(cell: &'a InsertCell) -> Result<ResolvedInsertCell<'a>, ExecuteError> {
    let input = match cell {
        InsertCell::Value { value, provenance } => {
            match (value, provenance) {
                (
                    SqlValue::Parameter { index, .. },
                    gpu_db_sql::InsertValueProvenance::Parameter {
                        index: provenance_index,
                    },
                ) if index == provenance_index => {
                    return Err(EngineError::ApplyFailed(
                        "unbound INSERT parameter reached semantic preparation".to_string(),
                    )
                    .into());
                }
                (SqlValue::Parameter { .. }, _) => {
                    return Err(EngineError::ApplyFailed(
                        "unbound INSERT parameter provenance does not match its parameter slot"
                            .to_string(),
                    )
                    .into());
                }
                (_, gpu_db_sql::InsertValueProvenance::Parameter { .. }) => {
                    return Err(EngineError::ApplyFailed(
                        "unbound INSERT parameter provenance requires a parameter slot".to_string(),
                    )
                    .into());
                }
                _ => {}
            }
            if matches!(value, SqlValue::Null) {
                ResolvedInsertInput::ProvidedNull {
                    provenance: *provenance,
                }
            } else {
                ResolvedInsertInput::Provided {
                    value,
                    provenance: *provenance,
                }
            }
        }
        InsertCell::Default { provenance } => ResolvedInsertInput::ExplicitDefault {
            provenance: *provenance,
        },
    };
    Ok(ResolvedInsertCell { input })
}

fn resolve_target_columns(
    insert: &Insert,
    table: &RelationalTable,
) -> Result<Vec<usize>, ExecuteError> {
    if insert.columns.is_empty() {
        return Ok((0..table.columns.len()).collect());
    }
    let mut target_indexes = Vec::with_capacity(insert.columns.len());
    let mut seen = BTreeSet::new();
    for name in &insert.columns {
        if !seen.insert(name) {
            return Err(EngineError::DuplicateColumn(name.clone()).into());
        }
        let index = table
            .columns
            .iter()
            .position(|column| column.name == *name)
            .ok_or_else(|| EngineError::UndefinedColumn(name.clone()))?;
        target_indexes.push(index);
    }
    Ok(target_indexes)
}
