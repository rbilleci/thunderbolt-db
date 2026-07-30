//! Allocation-free host-retention geometry for resident append sources.
//!
//! The pre-lease path cannot build an identity map/set report: it is useful to audit a
//! materialized owner, but its allocation would itself alter the peak being admitted. This leaf
//! preserves the report's `Arc<str>` de-duplication with bounded re-scans of the already sealed
//! table/dependency sequence.

use super::*;

pub(super) fn prediction(batch: &TypedInsertBatch) -> Result<HostRetentionGeometry, EngineError> {
    if !batch.sequence_bindings.is_empty() || !batch.returning.is_empty() {
        return Err(EngineError::Durability(
            "typed INSERT resident-source retention prediction requires a supported effect shape"
                .to_string(),
        ));
    }
    let rows = batch.row_count as usize;
    let mut geometry = HostRetentionGeometry::default();
    append_table_and_dependencies(&mut geometry, &batch.table, &batch.dependencies)?;
    for column in batch.columns.iter() {
        if !(matches!(column.presence, TypedInsertColumnPresence::AllProvided)
            && column.all_inputs_are_resolved(rows)
            && is_live_resident_append_type(column.ty)
            && column.values.rows_match(rows))
        {
            return Err(EngineError::Durability(
                "typed INSERT resident-source retention prediction lost source eligibility"
                    .to_string(),
            ));
        }
        append_validity(&mut geometry, &column.validity)?;
        append_values(&mut geometry, &column.values)?;
    }
    geometry.checked_add_backing_elements::<PreparedResidentAppendColumn>(
        batch.columns.len(),
        "typed INSERT resident source column box",
    )?;
    Ok(geometry)
}

pub(super) fn materialized_geometry(
    source: &PreparedResidentAppendSource,
) -> Result<HostRetentionGeometry, EngineError> {
    let mut geometry = HostRetentionGeometry::default();
    append_table_and_dependencies(&mut geometry, &source.table, &source.dependencies)?;
    geometry.checked_add_backing_elements::<PreparedResidentAppendColumn>(
        source.columns.len(),
        "typed INSERT resident source column box",
    )?;
    for column in source.columns.iter() {
        if let Some(validity) = column.validity.as_ref() {
            append_validity(&mut geometry, validity)?;
        }
        if let Some(values) = column.values.as_ref() {
            append_values(&mut geometry, values)?;
        }
    }
    Ok(geometry)
}

fn append_table_and_dependencies(
    geometry: &mut HostRetentionGeometry,
    table: &TypedInsertBatchTable,
    dependencies: &[TypedInsertDependencyBinding],
) -> Result<(), EngineError> {
    append_arc_str(geometry, &table.schema, false)?;
    append_arc_str(
        geometry,
        &table.name,
        Arc::ptr_eq(&table.name, &table.schema),
    )?;
    geometry.checked_add_backing_elements::<TypedInsertDependencyBinding>(
        dependencies.len(),
        "typed INSERT resident source dependency box",
    )?;
    for (position, dependency) in dependencies.iter().enumerate() {
        let schema_seen = arc_seen_before(&dependency.schema, table, dependencies, position, false);
        append_arc_str(geometry, &dependency.schema, schema_seen)?;
        let name_seen = Arc::ptr_eq(&dependency.name, &dependency.schema)
            || arc_seen_before(&dependency.name, table, dependencies, position, false);
        append_arc_str(geometry, &dependency.name, name_seen)?;
    }
    Ok(())
}

fn arc_seen_before(
    candidate: &Arc<str>,
    table: &TypedInsertBatchTable,
    dependencies: &[TypedInsertDependencyBinding],
    position: usize,
    _include_current: bool,
) -> bool {
    Arc::ptr_eq(candidate, &table.schema)
        || Arc::ptr_eq(candidate, &table.name)
        || dependencies[..position].iter().any(|dependency| {
            Arc::ptr_eq(candidate, &dependency.schema) || Arc::ptr_eq(candidate, &dependency.name)
        })
}

fn append_arc_str(
    geometry: &mut HostRetentionGeometry,
    value: &Arc<str>,
    already_seen: bool,
) -> Result<(), EngineError> {
    if already_seen {
        return Ok(());
    }
    let bytes = u64::try_from(value.len()).map_err(|_| {
        EngineError::Durability(
            "typed INSERT resident source Arc string length overflows".to_string(),
        )
    })?;
    geometry.checked_add_backing_bytes_slots(bytes, 1, "typed INSERT resident source Arc string")
}

fn append_validity(
    geometry: &mut HostRetentionGeometry,
    validity: &TypedInsertColumnValidity,
) -> Result<(), EngineError> {
    match validity {
        TypedInsertColumnValidity::AllValid => Ok(()),
        TypedInsertColumnValidity::Bitmap(words) => {
            append_boxed(geometry, words, "typed INSERT validity")
        }
    }
}

fn append_values(
    geometry: &mut HostRetentionGeometry,
    values: &TypedInsertColumnValues,
) -> Result<(), EngineError> {
    match values {
        TypedInsertColumnValues::I32(values) => {
            append_boxed(geometry, values, "typed INSERT i32 values")
        }
        TypedInsertColumnValues::I64(values) => {
            append_boxed(geometry, values, "typed INSERT i64 values")
        }
        TypedInsertColumnValues::I128(values) => {
            append_boxed(geometry, values, "typed INSERT i128 values")
        }
        TypedInsertColumnValues::Bytes16(values) => {
            append_boxed(geometry, values, "typed INSERT bytes16 values")
        }
        TypedInsertColumnValues::BoolBits(values) => {
            append_boxed(geometry, values, "typed INSERT bool values")
        }
        TypedInsertColumnValues::Text { offsets, bytes } => {
            append_boxed(geometry, offsets, "typed INSERT text offsets")?;
            append_boxed(geometry, bytes, "typed INSERT text bytes")
        }
    }
}

fn append_boxed<T>(
    geometry: &mut HostRetentionGeometry,
    values: &[T],
    domain: &'static str,
) -> Result<(), EngineError> {
    geometry.checked_add_backing_elements::<T>(values.len(), domain)
}

#[cfg(test)]
mod tests {
    #[test]
    fn prelease_scalar_leaf_never_names_the_identity_report() {
        let production = include_str!("resident_source_retention.rs")
            .split("\n#[cfg(test)]")
            .next()
            .expect("production leaf precedes tests");
        for forbidden in ["HostRetentionReport", "BTreeMap", "BTreeSet"] {
            assert!(
                !production.contains(forbidden),
                "pre-lease scalar leaf has {forbidden}"
            );
        }
    }
}
