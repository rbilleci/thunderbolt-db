//! Bound INSERT `RETURNING` metadata.
//!
//! This leaf binds projection names exactly once against the target's catalog-order columns. It
//! intentionally owns no value projection, host row materialization, or DML result route.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReturningColumnBinding {
    pub(super) catalog_column_ordinal: u32,
    pub(super) column_id: u32,
    pub(super) attnum: i16,
    pub(super) name: Arc<str>,
    pub(super) ty: SqlType,
    pub(super) type_oid: u32,
    pub(super) type_size: i16,
}

impl ReturningColumnBinding {
    #[cfg(test)]
    pub(crate) fn catalog_column_ordinal(&self) -> u32 {
        self.catalog_column_ordinal
    }

    #[cfg(test)]
    pub(crate) fn column_id(&self) -> u32 {
        self.column_id
    }

    #[cfg(test)]
    pub(crate) fn ty(&self) -> SqlType {
        self.ty
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReturningResultGeometry {
    pub(super) row_count: u32,
    pub(super) column_count: u32,
    pub(super) cell_count: u64,
}

/// Compact RETURNING geometry retained by semantic preparation. It is metadata only and has no
/// host result rows, projection buffer, or result-route authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
pub(crate) struct InsertReturningEffectShape {
    row_count: u32,
    column_count: u32,
    cell_count: u64,
}

#[allow(dead_code)] // The current live route intentionally has no effect-plan consumer.
impl InsertReturningEffectShape {
    pub(crate) const fn row_count(self) -> u32 {
        self.row_count
    }

    pub(crate) const fn column_count(self) -> u32 {
        self.column_count
    }

    pub(crate) const fn cell_count(self) -> u64 {
        self.cell_count
    }
}

/// One SQL-order projection identity. It preserves duplicate and wildcard expansion evidence
/// without exposing a value projection or allocating a result container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
pub(crate) struct ReturningProjectionEffectIdentity<'a> {
    catalog_column_ordinal: u32,
    column_id: u32,
    attnum: i16,
    name: &'a str,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

#[allow(dead_code)] // The current live route intentionally has no effect-plan consumer.
impl ReturningProjectionEffectIdentity<'_> {
    pub(crate) const fn catalog_column_ordinal(self) -> u32 {
        self.catalog_column_ordinal
    }

    pub(crate) const fn column_id(self) -> u32 {
        self.column_id
    }

    pub(crate) const fn attnum(self) -> i16 {
        self.attnum
    }

    pub(crate) fn name(&self) -> &str {
        self.name
    }

    pub(crate) const fn ty(self) -> SqlType {
        self.ty
    }

    pub(crate) const fn type_oid(self) -> u32 {
        self.type_oid
    }

    pub(crate) const fn type_size(self) -> i16 {
        self.type_size
    }
}

/// The prepared output contract for a future typed DML result route. Duplicate projections stay
/// duplicated and projection order stays SQL order; neither is recovered from a set later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundInsertReturning {
    columns: Box<[ReturningColumnBinding]>,
    geometry: ReturningResultGeometry,
}

impl BoundInsertReturning {
    pub(super) fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(super) const fn effect_shape(&self) -> InsertReturningEffectShape {
        InsertReturningEffectShape {
            row_count: self.geometry.row_count,
            column_count: self.geometry.column_count,
            cell_count: self.geometry.cell_count,
        }
    }

    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(super) fn effect_projection_identities(
        &self,
    ) -> impl ExactSizeIterator<Item = ReturningProjectionEffectIdentity<'_>> + '_ {
        self.columns
            .iter()
            .map(|column| ReturningProjectionEffectIdentity {
                catalog_column_ordinal: column.catalog_column_ordinal,
                column_id: column.column_id,
                attnum: column.attnum,
                name: column.name.as_ref(),
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
            })
    }

    #[cfg(test)]
    pub(crate) fn columns(&self) -> &[ReturningColumnBinding] {
        &self.columns
    }

    #[cfg(test)]
    pub(crate) fn geometry(&self) -> ReturningResultGeometry {
        self.geometry
    }

    /// Re-establish the compact output geometry without synthesizing a result container.
    pub(super) fn geometry_is_exact(&self, row_count: u32) -> bool {
        let Ok(column_count) = u32::try_from(self.columns.len()) else {
            return false;
        };
        u64::from(row_count).checked_mul(u64::from(column_count)) == Some(self.geometry.cell_count)
            && self.geometry.row_count == row_count
            && self.geometry.column_count == column_count
    }
}

pub(super) fn bind(
    table: &RelationalTable,
    returning: &[String],
    row_count: u32,
) -> Result<BoundInsertReturning, EngineError> {
    let mut columns = Vec::new();
    for name in returning {
        if name == PROJECTION_WILDCARD_SENTINEL {
            for column in &table.columns {
                columns.push(binding_for(table, column)?);
            }
            continue;
        }
        let column = table
            .columns
            .iter()
            .find(|column| column.name == *name)
            .ok_or_else(|| EngineError::UndefinedColumn(name.to_string()))?;
        columns.push(binding_for(table, column)?);
    }
    let column_count = u32::try_from(columns.len()).map_err(|_| {
        EngineError::Durability("typed INSERT RETURNING column count exceeds u32".to_string())
    })?;
    let cell_count = u64::from(row_count)
        .checked_mul(u64::from(column_count))
        .ok_or_else(|| {
            EngineError::Durability("typed INSERT RETURNING result geometry overflows".to_string())
        })?;
    Ok(BoundInsertReturning {
        columns: columns.into(),
        geometry: ReturningResultGeometry {
            row_count,
            column_count,
            cell_count,
        },
    })
}

fn binding_for(
    table: &RelationalTable,
    column: &RelationalColumn,
) -> Result<ReturningColumnBinding, EngineError> {
    Ok(ReturningColumnBinding {
        catalog_column_ordinal: u32::try_from(
            table
                .columns
                .iter()
                .position(|candidate| candidate.id == column.id)
                .ok_or_else(|| {
                    EngineError::Durability(
                        "typed INSERT RETURNING binding lost target column identity".to_string(),
                    )
                })?,
        )
        .map_err(|_| {
            EngineError::Durability("typed INSERT RETURNING column ordinal exceeds u32".to_string())
        })?,
        column_id: column.id,
        attnum: column.attnum,
        name: Arc::from(column.name.as_str()),
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
    })
}
