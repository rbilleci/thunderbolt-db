//! Immutable fixed-width INSERT batches prepared off the commit sequencer.
//!
//! The direct INSERT-001 carrier owns the only fixed-width values authority for its typed route.
//! It binds that move-only payload to the exact off-lock catalog/dependency proof without copying
//! the mutable catalog, predicted row keys, or a second write/WAL publication authority.

use super::*;
use std::io::Write as _;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedInsertBatchTable {
    name: Arc<str>,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    prepared_catalog_seq: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedInsertDependencyBinding {
    name: Arc<str>,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedInsertBatchColumn {
    /// Stable pg_attribute identity, never a caller-provided column position.
    column_id: u32,
    attnum: i16,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
    values: Box<[i32]>,
}

/// Move-only, column-major i32 payload derived solely by consuming a sealed batch.
///
/// This is intentionally not a general-purpose column container: it has no public constructor,
/// no `Clone`, and no mutable value access.  The residency seam can therefore trust that its
/// table and attribute bindings are the exact bindings proved during off-lock preparation.
// Consumed by the accepted direct INSERT residency route; it remains move-only so that route
// cannot manufacture a second values authority.
pub(crate) struct PreparedI32AppendSource {
    table: PreparedInsertBatchTable,
    row_count: u32,
    columns: Box<[PreparedI32AppendColumn]>,
    dependencies: Box<[PreparedInsertDependencyBinding]>,
}

// Fields are exposed only through the narrow residency accessors below.
pub(crate) struct PreparedI32AppendColumn {
    column_id: u32,
    attnum: i16,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
    values: Vec<i32>,
}

/// A sealed columnar projection of one already-authoritative off-lock INSERT preparation.
///
/// The type has no public fields, no public constructor, and no `Clone`: an individual batch is
/// owned by exactly one `OfflockPreparedDml` carrier. The
/// initial route supports only NULL-free `int4` columns; every broader shape remains legacy.
// Fields are intentionally private to the accepted fixed-width carrier.
pub(crate) struct PreparedInsertBatch {
    table: PreparedInsertBatchTable,
    row_count: u32,
    columns: Box<[PreparedInsertBatchColumn]>,
    /// Sorted compact stable bindings for the exact direct catalog proof. The first route admits
    /// exactly one target dependency, but retaining this as a list makes any later expansion
    /// explicit rather than silently dropping dependency identity.
    dependencies: Box<[PreparedInsertDependencyBinding]>,
}

impl PreparedInsertBatch {
    /// Build the fixed carrier directly from the parsed INSERT and one pinned catalog snapshot.
    /// This is the only off-lock success path that deliberately does not create a `WriteDelta`,
    /// predicted row keys, or a host-row mutation authority.
    fn from_direct_offlock_prepare(
        insert: &Insert,
        catalog: &CatalogSnapshot,
        prepared_catalog_seq: Index,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
    ) -> Result<Option<Self>, ExecuteError> {
        if let Some(expectation) = expected_catalog_version {
            crate::engine_mutation_admission::validate_catalog_version_expectation(
                expectation,
                catalog.commit_seq,
            )?;
        }
        if !insert.returning.is_empty() || catalog.commit_seq != prepared_catalog_seq {
            return Ok(None);
        }
        let Some(table) = catalog.relational_catalog.get(&insert.table) else {
            return Ok(None);
        };
        if !insert.columns.is_empty()
            && (insert.columns.len() != table.columns.len()
                || !insert
                    .columns
                    .iter()
                    .zip(&table.columns)
                    .all(|(provided, column)| provided == &column.name))
        {
            return Ok(None);
        }
        if table.name.len() > u16::MAX as usize
            || insert.rows.is_empty()
            || table.columns.is_empty()
            || !table.indexes.is_empty()
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty()
            || table.columns.iter().any(|column| {
                column.ty != SqlType::Int4
                    || column.domain.is_some()
                    || column.default.is_some()
                    || column.table_oid != table.oid
            })
        {
            return Ok(None);
        }
        let row_count = match u32::try_from(insert.rows.len()) {
            Ok(rows) => rows,
            Err(_) => return Ok(None),
        };
        let mut values = (0..table.columns.len())
            .map(|_| Vec::with_capacity(insert.rows.len()))
            .collect::<Vec<_>>();
        for row in &insert.rows {
            if row.len() != table.columns.len() {
                return Ok(None);
            }
            for (position, value) in row.iter().enumerate() {
                let SqlValue::Int4(value) = value else {
                    return Ok(None);
                };
                values[position].push(*value);
            }
        }
        let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)?;
        let columns = table
            .columns
            .iter()
            .zip(values)
            .map(|(column, values)| PreparedInsertBatchColumn {
                column_id: column.id,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                values: values.into_boxed_slice(),
            })
            .collect::<Vec<_>>();
        Ok(Some(Self {
            table: PreparedInsertBatchTable {
                name: Arc::from(table.name.as_str()),
                oid: table.oid,
                schema_digest,
                prepared_catalog_seq,
            },
            row_count,
            columns: columns.into_boxed_slice(),
            dependencies: vec![PreparedInsertDependencyBinding {
                name: Arc::from(table.name.as_str()),
                oid: table.oid,
                schema_digest,
            }]
            .into_boxed_slice(),
        }))
    }

    #[cfg(test)]
    fn from_authoritative_offlock_prepare(
        insert: &Insert,
        delta: &WriteDelta,
        catalog: &CatalogSnapshot,
        prepared_catalog_seq: Index,
        expected_catalog_version: Option<
            crate::engine_mutation_admission::CatalogVersionExpectation,
        >,
    ) -> Option<Self> {
        // The item retains this same expectation and the sequencer revalidates it under the
        // commit/catalog boundary. Admit only an expectation already exact for this off-lock
        // catalog; a later mismatch remains the established pre-WAL item failure, never a
        // carrier-specific bypass.
        if expected_catalog_version.is_some_and(|expectation| {
            crate::engine_mutation_admission::validate_catalog_version_expectation(
                expectation,
                catalog.commit_seq,
            )
            .is_err()
        }) || !insert.returning.is_empty()
            || catalog.commit_seq != prepared_catalog_seq
        {
            return None;
        }
        let table = catalog.relational_catalog.get(&insert.table)?;
        // Named INSERT values are source ordered, while the sealed vectors are catalog ordered.
        // Accept the named form only when those orders are proven identical; subset, reorder,
        // duplicate, and unknown lists retain the general preparation path.
        if !insert.columns.is_empty()
            && (insert.columns.len() != table.columns.len()
                || !insert
                    .columns
                    .iter()
                    .zip(&table.columns)
                    .all(|(provided, column)| provided == &column.name))
        {
            return None;
        }
        if table.columns.is_empty()
            || !table.indexes.is_empty()
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty()
            || table.columns.iter().any(|column| {
                column.ty != SqlType::Int4
                    || column.domain.is_some()
                    || column.default.is_some()
                    || column.table_oid != table.oid
            })
        {
            return None;
        }
        let PreparedMutation::Insert {
            table: delta_table,
            inserted_rows,
            seq_advances,
        } = &delta.mutation
        else {
            return None;
        };
        if delta_table != &table.name
            || inserted_rows.is_empty()
            || inserted_rows.len() != insert.rows.len()
            || delta.rows_consumed != inserted_rows.len() as u64
            || !seq_advances.is_empty()
            || !delta.foreign_key_dependencies.is_empty()
            || delta.catalog_dependencies.len() != 1
            || delta.catalog_dependencies.get(&table.name) != Some(table)
            || delta.write_set.tables.len() != 1
            || !delta.write_set.tables.contains(&table.name)
            || !delta.write_set.rows.is_empty()
            || !delta.write_set.unique_slots.is_empty()
            || !delta.write_set.unique_slots_i32.is_empty()
        {
            return None;
        }
        let row_count = u32::try_from(inserted_rows.len()).ok()?;
        let schema_digest = crate::engine_transaction_reset::table_schema_digest(table).ok()?;
        // Allocate every typed column exactly once. The source/prepared equality check below
        // prevents this batch from becoming a second coercion/default authority.
        let mut values = (0..table.columns.len())
            .map(|_| Vec::with_capacity(inserted_rows.len()))
            .collect::<Vec<_>>();
        for (source_row, (_row_key, prepared_row)) in insert.rows.iter().zip(inserted_rows) {
            if source_row.len() != table.columns.len() || prepared_row.len() != table.columns.len()
            {
                return None;
            }
            for (position, (source_value, prepared_value)) in
                source_row.iter().zip(prepared_row).enumerate()
            {
                let (SqlValue::Int4(source), SqlValue::Int4(prepared)) =
                    (source_value, prepared_value)
                else {
                    return None;
                };
                if source != prepared {
                    return None;
                }
                values[position].push(*prepared);
            }
        }
        let columns = table
            .columns
            .iter()
            .zip(values)
            .map(|(column, values)| PreparedInsertBatchColumn {
                column_id: column.id,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                values: values.into_boxed_slice(),
            })
            .collect::<Vec<_>>();
        let dependencies = delta
            .catalog_dependencies
            .iter()
            .map(|(name, dependency)| {
                let live = catalog.relational_catalog.get(name)?;
                if live != dependency {
                    return None;
                }
                Some(PreparedInsertDependencyBinding {
                    name: Arc::from(name.as_str()),
                    oid: dependency.oid,
                    schema_digest: crate::engine_transaction_reset::table_schema_digest(dependency)
                        .ok()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            table: PreparedInsertBatchTable {
                name: Arc::from(table.name.as_str()),
                oid: table.oid,
                schema_digest,
                prepared_catalog_seq,
            },
            row_count,
            columns: columns.into_boxed_slice(),
            dependencies: dependencies.into_boxed_slice(),
        })
    }

    #[cfg(test)]
    fn value_bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.values.len() * std::mem::size_of::<i32>())
            .sum()
    }

    /// Derive the sole binary INSERT template from this already-sealed batch. The template
    /// owns no second values authority: it encodes this exact immutable columnar projection only.
    pub(super) fn binary_insert_template(
        &self,
    ) -> Result<crate::wal_binary::PreparedBinaryInsertTemplate, EngineError> {
        crate::wal_binary::PreparedBinaryInsertTemplate::from_sealed_batch(self)
    }

    /// Narrow template-only access to the stable target relation name. The binary template never
    /// accepts caller-provided table metadata.
    pub(crate) fn binary_insert_template_table_name(&self) -> &str {
        &self.table.name
    }

    /// Narrow template-only access to the sealed row count.
    pub(crate) fn binary_insert_template_row_count(&self) -> u32 {
        self.row_count
    }

    /// Recheck the sealed batch at the commit boundary before its move-only source can cross the
    /// typed residency preflight.  The constructor already established these equalities off-lock;
    /// this second proof prevents catalog drift or a mismatched carrier from turning the batch
    /// into an independent coercion, dependency, or write-set authority.
    #[cfg(test)]
    #[allow(dead_code)] // retained direct-carrier proof accessor for fixed-template verification.
    pub(crate) fn matches_fixed_insert_delta(
        &self,
        delta: &WriteDelta,
        expected_write_set: &WriteSet,
        catalog: &CatalogSnapshot,
        prepared_catalog_seq: Index,
    ) -> bool {
        if self.table.prepared_catalog_seq != prepared_catalog_seq
            || catalog.commit_seq != prepared_catalog_seq
            || self.row_count == 0
            || self.columns.is_empty()
            || delta.write_set != *expected_write_set
            || !self.exact_dependencies_match(catalog)
        {
            return false;
        }
        let Some(live_table) = catalog.relational_catalog.get(self.table.name.as_ref()) else {
            return false;
        };
        if live_table.oid != self.table.oid
            || crate::engine_transaction_reset::table_schema_digest(live_table).ok()
                != Some(self.table.schema_digest)
            || live_table.columns.len() != self.columns.len()
        {
            return false;
        }
        let PreparedMutation::Insert {
            table,
            inserted_rows,
            seq_advances,
        } = &delta.mutation
        else {
            return false;
        };
        if table != self.table.name.as_ref()
            || inserted_rows.len() != self.row_count as usize
            || delta.rows_consumed != u64::from(self.row_count)
            || !seq_advances.is_empty()
            || !delta.foreign_key_dependencies.is_empty()
            || delta.catalog_dependencies.len() != 1
            || delta.catalog_dependencies.get(table) != Some(live_table)
            || delta.write_set.tables.len() != 1
            || !delta.write_set.tables.contains(table)
            || !delta.write_set.rows.is_empty()
            || !delta.write_set.unique_slots.is_empty()
            || !delta.write_set.unique_slots_i32.is_empty()
        {
            return false;
        }
        self.columns
            .iter()
            .enumerate()
            .all(|(column_index, column)| {
                let Some(live_column) = live_table.columns.get(column_index) else {
                    return false;
                };
                column.column_id == live_column.id
                && column.attnum == live_column.attnum
                && column.ty == SqlType::Int4
                && column.ty == live_column.ty
                && column.type_oid == live_column.type_oid
                && column.type_size == live_column.type_size
                && column.values.len() == inserted_rows.len()
                && inserted_rows.iter().zip(column.values.iter()).all(|((_key, row), value)| {
                    matches!(row.get(column_index), Some(SqlValue::Int4(actual)) if actual == value)
                })
            })
    }

    /// Commit-bound recheck for a direct off-lock carrier.  There is intentionally no delta to
    /// compare: table-only conflict footprint, catalog/dependency identity, and every typed
    /// column binding are the complete proof retained by this route.
    pub(crate) fn matches_direct_fixed_insert(
        &self,
        expected_write_set: &WriteSet,
        catalog: &CatalogSnapshot,
        prepared_catalog_seq: Index,
    ) -> bool {
        self.table.prepared_catalog_seq == prepared_catalog_seq
            && catalog.commit_seq == prepared_catalog_seq
            && self.row_count != 0
            && !self.columns.is_empty()
            && expected_write_set.tables.len() == 1
            && expected_write_set.tables.contains(self.table.name.as_ref())
            && expected_write_set.rows.is_empty()
            && expected_write_set.unique_slots.is_empty()
            && expected_write_set.unique_slots_i32.is_empty()
            && self.exact_dependencies_match(catalog)
            && catalog
                .relational_catalog
                .get(self.table.name.as_ref())
                .is_some_and(|table| {
                    table.oid == self.table.oid
                        && crate::engine_transaction_reset::table_schema_digest(table).ok()
                            == Some(self.table.schema_digest)
                        && table.columns.len() == self.columns.len()
                        && self
                            .columns
                            .iter()
                            .zip(&table.columns)
                            .all(|(column, live)| {
                                column.column_id == live.id
                                    && column.attnum == live.attnum
                                    && column.ty == SqlType::Int4
                                    && column.ty == live.ty
                                    && column.type_oid == live.type_oid
                                    && column.type_size == live.type_size
                                    && column.values.len() == self.row_count as usize
                            })
                })
    }

    fn exact_dependencies_match(&self, catalog: &CatalogSnapshot) -> bool {
        self.dependencies.len() == 1
            && self.dependencies[0].name == self.table.name
            && self.dependencies[0].oid == self.table.oid
            && self.dependencies[0].schema_digest == self.table.schema_digest
            && catalog
                .relational_catalog
                .get(self.dependencies[0].name.as_ref())
                .is_some_and(|table| {
                    table.oid == self.dependencies[0].oid
                        && crate::engine_transaction_reset::table_schema_digest(table).ok()
                            == Some(self.dependencies[0].schema_digest)
                })
    }

    /// Append one sealed fixed-width row in the existing relational cell encoding. Keeping this
    /// encoding beside the private column vectors prevents a template consumer from observing or
    /// supplying loose typed values, while writing directly into the destination avoids a
    /// per-row relational `Vec` or decimal `String` staging allocation.
    pub(crate) fn append_binary_insert_template_row(
        &self,
        row: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), EngineError> {
        if self.columns.is_empty() || row >= self.row_count as usize {
            return Err(EngineError::Durability(
                "sealed fixed-width INSERT batch row is out of bounds".to_string(),
            ));
        }
        if self.columns.iter().any(|column| {
            column.ty != SqlType::Int4 || column.values.len() != self.row_count as usize
        }) {
            return Err(EngineError::Durability(
                "sealed fixed-width INSERT batch lost int4 column invariants".to_string(),
            ));
        }
        for (position, column) in self.columns.iter().enumerate() {
            if position != 0 {
                out.push(b'|');
            }
            write!(out, "i:{}", column.values[row])
                .expect("writing a relational int4 cell into Vec<u8> cannot fail");
        }
        Ok(())
    }

    /// Consume this sealed batch into the only typed residency input. `Box<[i32]> -> Vec<i32>`
    /// preserves the allocation; this is the ownership handoff that prevents a second column
    /// matrix from entering the eventual device-append route.
    pub(crate) fn into_i32_append_source(self) -> PreparedI32AppendSource {
        PreparedI32AppendSource {
            table: self.table,
            row_count: self.row_count,
            columns: self
                .columns
                .into_vec()
                .into_iter()
                .map(|column| PreparedI32AppendColumn {
                    column_id: column.column_id,
                    attnum: column.attnum,
                    ty: column.ty,
                    type_oid: column.type_oid,
                    type_size: column.type_size,
                    values: column.values.into_vec(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            dependencies: self.dependencies,
        }
    }
}

impl PreparedI32AppendSource {
    pub(crate) fn table_name(&self) -> &str {
        &self.table.name
    }

    pub(crate) fn table_oid(&self) -> u32 {
        self.table.oid
    }

    pub(crate) fn schema_digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.table.schema_digest
    }

    pub(crate) fn prepared_catalog_seq(&self) -> Index {
        self.table.prepared_catalog_seq
    }

    pub(crate) fn row_count(&self) -> usize {
        self.row_count as usize
    }

    pub(crate) fn columns(&self) -> &[PreparedI32AppendColumn] {
        &self.columns
    }

    pub(crate) fn exact_single_table_dependency(&self) -> bool {
        self.dependencies.len() == 1
            && self.dependencies[0].name == self.table.name
            && self.dependencies[0].oid == self.table.oid
            && self.dependencies[0].schema_digest == self.table.schema_digest
    }
}

impl PreparedI32AppendColumn {
    pub(crate) fn column_id(&self) -> u32 {
        self.column_id
    }

    pub(crate) fn attnum(&self) -> i16 {
        self.attnum
    }

    pub(crate) fn ty(&self) -> SqlType {
        self.ty
    }

    pub(crate) fn type_oid(&self) -> u32 {
        self.type_oid
    }

    pub(crate) fn type_size(&self) -> i16 {
        self.type_size
    }

    pub(crate) fn values(&self) -> &[i32] {
        &self.values
    }
}

/// Test-only legacy comparison adapter: build from an already-resolved `WriteDelta` and immutable
/// catalog snapshot. Production direct preparation never calls this path.
#[cfg(test)]
pub(super) fn try_prepare_fixed_insert_batch(
    command: &Command,
    delta: &WriteDelta,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
) -> Option<PreparedInsertBatch> {
    let Command::Insert(insert) = command else {
        return None;
    };
    PreparedInsertBatch::from_authoritative_offlock_prepare(
        insert,
        delta,
        catalog,
        prepared_catalog_seq,
        expected_catalog_version,
    )
}

/// Direct typed off-lock builder. `Ok(None)` is a static ineligible shape and must take the
/// established `prepare_dml` path exactly once; `Err` is an exact catalog expectation failure.
pub(super) fn try_prepare_direct_fixed_insert_batch(
    command: &Command,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
) -> Result<Option<PreparedInsertBatch>, ExecuteError> {
    let Command::Insert(insert) = command else {
        return Ok(None);
    };
    PreparedInsertBatch::from_direct_offlock_prepare(
        insert,
        catalog,
        prepared_catalog_seq,
        expected_catalog_version,
    )
}

#[cfg(test)]
#[path = "prepared_insert_batch_tests.rs"]
mod tests;
