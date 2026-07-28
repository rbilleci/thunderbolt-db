//! Sealing builder for the immutable typed INSERT semantic carrier.
//!
//! This leaf owns only source-order diagnostics and catalog-order vector construction. The
//! parent retains the sealed carrier and every physical-consumption facade.

use super::*;

/// The unconstrained, no-RETURNING semantic builder. It validates the same source order as
/// `prepare_insert`, constructs catalog-order vectors, and reports only still-unsupported
/// semantics explicitly instead of manufacturing a legacy row representation.
pub(super) fn build(
    insert: &Insert,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
    capability: TypedInsertBuildCapability,
) -> Result<TypedInsertBuildResult, ExecuteError> {
    if let Some(expectation) = expected_catalog_version {
        crate::engine_mutation_admission::validate_catalog_version_expectation(
            expectation,
            catalog.commit_seq,
        )?;
    }
    if catalog.commit_seq != prepared_catalog_seq {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::CatalogGeneration,
        ));
    }
    let table = catalog
        .relational_catalog
        .get(&insert.table)
        .ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", insert.table))
        })?;
    let semantics = ResolvedInsertSemantics::resolve(insert, table, InsertStatementOrdinal::FIRST)?;
    debug_assert_eq!(semantics.table.name, table.name);
    let row_count = semantics.row_count;
    let rows = usize::try_from(row_count).expect("u32 always fits usize on supported hosts");
    let mut builders = table
        .columns
        .iter()
        .map(|column| TypedInsertColumnBuilder::new(column.ty, rows))
        .collect::<Result<Vec<_>, _>>()?;
    let mut source_columns = semantics
        .columns
        .iter()
        .enumerate()
        .filter_map(|(catalog_index, column)| {
            column
                .source_column_ordinal
                .map(|source_column_ordinal| (source_column_ordinal, catalog_index))
        })
        .collect::<Vec<_>>();
    source_columns.sort_unstable_by_key(|(source_column_ordinal, _)| *source_column_ordinal);
    let resolved_target_count = if insert.columns.is_empty() {
        table.columns.len()
    } else {
        insert.columns.len()
    };
    debug_assert_eq!(source_columns.len(), resolved_target_count);
    debug_assert!(
        source_columns
            .iter()
            .enumerate()
            .all(|(expected, (actual, _))| usize::try_from(*actual) == Ok(expected)),
        "resolved INSERT source ordinals stay dense and target-list ordered"
    );
    for row_index in 0..rows {
        // Coercion is observable diagnostics: preserve SQL source-list order even though each
        // resulting vector is catalog ordered. Omitted columns were initialized above and
        // deliberately have no source cell to visit.
        for &(_, catalog_index) in &source_columns {
            let semantic_column = &semantics.columns[catalog_index];
            let builder = &mut builders[catalog_index];
            let column = semantic_column.column;
            debug_assert_eq!(semantic_column.catalog_ordinal as usize, catalog_index);
            debug_assert!(
                semantic_column.source_column_ordinal.is_some(),
                "catalog/source mapping remains stable for every resolved cell"
            );
            let cell = &semantic_column.cells[row_index];
            let provenance = TypedInsertInputProvenance::from_input(cell.input)?;
            match cell.input {
                ResolvedInsertInput::Provided { value, .. } => {
                    // Exactly one coercion per supplied scalar. Omitted/default cells never
                    // enter this branch, so a later operator owns their materialization.
                    let value = coerce_insert_value(value.clone(), column.ty, &column.name)?;
                    builder.set_provided(row_index, value, column.ty, provenance)?;
                }
                ResolvedInsertInput::ProvidedNull { .. } => {
                    builder.set_provided_null(row_index, provenance)?;
                }
                ResolvedInsertInput::Omitted => {}
                ResolvedInsertInput::ExplicitDefault { .. } => {
                    builder.set_explicit_default(row_index, provenance)?;
                }
            }
        }
        for builder in &mut builders {
            builder.finish_row(row_index)?;
        }
    }

    // Keep unsupported product semantics explicit only after all input diagnostics have been
    // resolved. A malformed supplied value must not disappear behind a default/domain defer.
    if !insert.returning.is_empty() {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::Returning,
        ));
    }
    let domain_dependencies = collect_domain_dependencies(table, catalog)?;
    let constraints_supported = {
        #[cfg(test)]
        if capability == TypedInsertBuildCapability::ProofOnly {
            table.foreign_keys.is_empty()
                && crate::engine_insert_plan::row_local_constraints::checks_are_device_supported(
                    table,
                )
                && crate::engine_insert_plan::batch_key_constraints::table_has_supported_batch_key_constraints(table)
        } else {
            crate::engine_insert_plan::row_local_constraints::table_has_supported_row_local_checks(
                table,
            )
        }
        #[cfg(not(test))]
        {
            crate::engine_insert_plan::row_local_constraints::table_has_supported_row_local_checks(
                table,
            )
        }
    };
    if !constraints_supported {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::Constraints,
        ));
    }
    let requires_current_table_oid = capability == TypedInsertBuildCapability::ResidentAppend || {
        #[cfg(test)]
        {
            capability == TypedInsertBuildCapability::ProofOnly
        }
        #[cfg(not(test))]
        {
            false
        }
    };
    if requires_current_table_oid
        && table
            .columns
            .iter()
            .any(|column| column.table_oid != table.oid)
    {
        return Ok(TypedInsertBuildResult::Deferred(
            TypedInsertDeferred::CatalogGeneration,
        ));
    }

    let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)?;
    let mut columns = table
        .columns
        .iter()
        .zip(&semantics.columns)
        .zip(builders)
        .map(|((column, semantic_column), builder)| {
            let (validity, presence, input_states, input_provenance, values) = builder.finish(rows);
            TypedInsertColumn {
                column_id: column.id,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                source_column_ordinal: semantic_column.source_column_ordinal,
                validity,
                presence,
                input_states,
                input_provenance,
                default_resolution: TypedInsertDefaultResolution::AllDirect,
                values,
                #[cfg(test)]
                full_invariant_scans: std::sync::atomic::AtomicUsize::new(0),
            }
        })
        .collect::<Vec<_>>();
    match defaults::resolve(&mut columns, table, rows) {
        Ok(()) => {}
        Err(defaults::DefaultResolutionError::Sequence { column_id }) => {
            return Ok(TypedInsertBuildResult::Deferred(
                TypedInsertDeferred::Default { column_id },
            ));
        }
        Err(defaults::DefaultResolutionError::Engine(error)) => return Err(error.into()),
    }
    if columns
        .iter()
        .any(|column| !column.full_invariants_hold(rows))
    {
        return Err(EngineError::Durability(
            "typed INSERT builder produced invalid column vectors".to_string(),
        )
        .into());
    }
    let batch = TypedInsertBatch {
        table: TypedInsertBatchTable {
            name: Arc::from(table.name.as_str()),
            oid: table.oid,
            schema_digest,
            prepared_catalog_seq,
        },
        statement_ordinal: semantics.statement_ordinal,
        row_count,
        columns: columns.into(),
        dependencies: vec![TypedInsertDependencyBinding {
            name: Arc::from(table.name.as_str()),
            oid: table.oid,
            schema_digest,
        }]
        .into(),
        domain_dependencies,
        #[cfg(test)]
        proof_only_indexed_constraints: capability == TypedInsertBuildCapability::ProofOnly
            && !table.indexes.is_empty(),
    };
    #[cfg(not(test))]
    let _ = capability;
    Ok(TypedInsertBuildResult::Ready(batch))
}

fn collect_domain_dependencies(
    table: &RelationalTable,
    catalog: &CatalogSnapshot,
) -> Result<Box<[TypedInsertDomainBinding]>, ExecuteError> {
    let mut seen = BTreeSet::new();
    let mut dependencies = Vec::new();
    for column in &table.columns {
        let Some(name) = column.domain.as_deref() else {
            continue;
        };
        let domain = catalog.relational_domains.get(name).ok_or_else(|| {
            EngineError::ApplyFailed(format!(
                "domain \"{name}\" for relation \"{}\" does not exist",
                table.name
            ))
        })?;
        if column.type_oid != domain.oid || column.ty != domain.base_type {
            return Err(EngineError::ApplyFailed(format!(
                "domain \"{name}\" binding for relation \"{}\" is inconsistent",
                table.name
            ))
            .into());
        }
        if seen.insert(name) {
            dependencies.push(TypedInsertDomainBinding {
                name: Arc::from(name),
                oid: domain.oid,
                base_type: domain.base_type,
            });
        }
    }
    Ok(dependencies.into())
}

struct TypedInsertColumnBuilder {
    validity_words: Vec<u32>,
    presence_words: Vec<u32>,
    input_states: Vec<TypedInsertInputState>,
    input_provenance: Vec<TypedInsertInputProvenance>,
    values: TypedInsertColumnValues,
    /// Text grows only while building. `finish` seals it into the batch's immutable boxed bytes.
    text_bytes: Option<Vec<u8>>,
}

impl TypedInsertInputProvenance {
    fn from_input(input: ResolvedInsertInput<'_>) -> Result<Self, ExecuteError> {
        match input {
            ResolvedInsertInput::Provided { provenance, .. }
            | ResolvedInsertInput::ProvidedNull { provenance } => {
                Self::from_value_provenance(provenance)
            }
            ResolvedInsertInput::Omitted => Ok(Self::Omitted),
            ResolvedInsertInput::ExplicitDefault { provenance } => Ok(match provenance {
                gpu_db_sql::InsertDefaultProvenance::SqlKeyword => Self::SqlDefault,
                gpu_db_sql::InsertDefaultProvenance::Programmatic => Self::ProgrammaticDefault,
            }),
        }
    }

    fn from_value_provenance(
        provenance: gpu_db_sql::InsertValueProvenance,
    ) -> Result<Self, ExecuteError> {
        match provenance {
            gpu_db_sql::InsertValueProvenance::Literal => Ok(Self::Literal),
            gpu_db_sql::InsertValueProvenance::BoundParameter { index } => {
                let index = u32::try_from(index).map_err(|_| {
                    EngineError::ApplyFailed(
                        "bound INSERT parameter index exceeds compact semantic metadata"
                            .to_string(),
                    )
                })?;
                Ok(Self::BoundParameter { index })
            }
            gpu_db_sql::InsertValueProvenance::Programmatic => Ok(Self::ProgrammaticValue),
            // `ResolvedInsertSemantics` rejects this provenance before value lowering. Keep the
            // sealed carrier fail-closed if a future resolver changes its call graph.
            gpu_db_sql::InsertValueProvenance::Parameter { .. } => Err(EngineError::ApplyFailed(
                "unbound INSERT parameter reached compact semantic lowering".to_string(),
            )
            .into()),
        }
    }
}

impl TypedInsertColumnBuilder {
    fn new(ty: SqlType, rows: usize) -> Result<Self, EngineError> {
        Ok(Self {
            validity_words: vec![0; bitmap_words(rows)?],
            presence_words: vec![0; bitmap_words(rows)?],
            input_states: vec![TypedInsertInputState::Omitted; rows],
            input_provenance: vec![TypedInsertInputProvenance::Omitted; rows],
            values: TypedInsertColumnValues::zeroed(ty, rows)?,
            text_bytes: (ty == SqlType::Text).then(Vec::new),
        })
    }

    fn set_provided(
        &mut self,
        row: usize,
        value: SqlValue,
        ty: SqlType,
        provenance: TypedInsertInputProvenance,
    ) -> Result<(), EngineError> {
        set_bit(&mut self.presence_words, row)?;
        if matches!(value, SqlValue::Null) {
            self.input_states[row] = TypedInsertInputState::ProvidedNull;
            self.input_provenance[row] = provenance;
            return Ok(());
        }
        self.input_states[row] = TypedInsertInputState::Provided;
        self.input_provenance[row] = provenance;
        set_bit(&mut self.validity_words, row)?;
        match (&mut self.values, ty, value) {
            (TypedInsertColumnValues::I32(values), SqlType::Int2, SqlValue::Int2(value)) => {
                values[row] = i32::from(value)
            }
            (TypedInsertColumnValues::I32(values), SqlType::Int4, SqlValue::Int4(value))
            | (TypedInsertColumnValues::I32(values), SqlType::Date, SqlValue::Date(value)) => {
                values[row] = value
            }
            (TypedInsertColumnValues::I64(values), SqlType::Int8, SqlValue::Int8(value))
            | (
                TypedInsertColumnValues::I64(values),
                SqlType::Timestamp,
                SqlValue::Timestamp(value),
            ) => values[row] = value,
            (
                TypedInsertColumnValues::I128(values),
                SqlType::Numeric { .. },
                SqlValue::Numeric(value),
            ) => values[row] = value.mantissa,
            (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid, SqlValue::Uuid(value)) => {
                values[row] = value
            }
            (TypedInsertColumnValues::BoolBits(words), SqlType::Bool, SqlValue::Bool(value)) => {
                if value {
                    set_bit(words, row)?;
                }
            }
            (TypedInsertColumnValues::Text { .. }, SqlType::Text, SqlValue::Text(value)) => {
                let bytes = self.text_bytes.as_mut().ok_or_else(|| {
                    EngineError::Durability(
                        "typed INSERT text builder lost its byte vector".to_string(),
                    )
                })?;
                let next_len = bytes.len().checked_add(value.len()).ok_or_else(|| {
                    EngineError::Durability("typed INSERT text bytes overflow".to_string())
                })?;
                u64::try_from(next_len).map_err(|_| {
                    EngineError::Durability("typed INSERT text exceeds u64 offsets".to_string())
                })?;
                bytes.extend_from_slice(value.as_bytes());
            }
            _ => {
                return Err(EngineError::Durability(
                    "coerced typed INSERT value does not match its vector arm".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn set_provided_null(
        &mut self,
        row: usize,
        provenance: TypedInsertInputProvenance,
    ) -> Result<(), EngineError> {
        set_bit(&mut self.presence_words, row)?;
        self.input_states[row] = TypedInsertInputState::ProvidedNull;
        self.input_provenance[row] = provenance;
        Ok(())
    }

    fn set_explicit_default(
        &mut self,
        row: usize,
        provenance: TypedInsertInputProvenance,
    ) -> Result<(), EngineError> {
        // An explicit DEFAULT names the source column but has no concrete scalar until a later
        // default/sequence operator resolves it.
        set_bit(&mut self.presence_words, row)?;
        self.input_states[row] = TypedInsertInputState::ExplicitDefault;
        self.input_provenance[row] = provenance;
        Ok(())
    }

    fn finish_row(&mut self, row: usize) -> Result<(), EngineError> {
        if let TypedInsertColumnValues::Text { offsets, .. } = &mut self.values {
            let bytes = self.text_bytes.as_ref().ok_or_else(|| {
                EngineError::Durability(
                    "typed INSERT text builder lost its byte vector".to_string(),
                )
            })?;
            offsets[row.checked_add(1).ok_or_else(|| {
                EngineError::Durability("typed INSERT text row index overflows".to_string())
            })?] = u64::try_from(bytes.len()).map_err(|_| {
                EngineError::Durability("typed INSERT text exceeds u64 offsets".to_string())
            })?;
        }
        Ok(())
    }

    fn finish(
        mut self,
        rows: usize,
    ) -> (
        TypedInsertColumnValidity,
        TypedInsertColumnPresence,
        Box<[TypedInsertInputState]>,
        Box<[TypedInsertInputProvenance]>,
        TypedInsertColumnValues,
    ) {
        let validity = if bitmap_is_all_set(&self.validity_words, rows) {
            TypedInsertColumnValidity::AllValid
        } else {
            TypedInsertColumnValidity::Bitmap(self.validity_words.into())
        };
        let presence = if bitmap_is_all_set(&self.presence_words, rows) {
            TypedInsertColumnPresence::AllProvided
        } else {
            TypedInsertColumnPresence::Bitmap(self.presence_words.into())
        };
        if let TypedInsertColumnValues::Text { bytes, .. } = &mut self.values {
            *bytes = self
                .text_bytes
                .take()
                .expect("text vector exists for a text column")
                .into();
        }
        (
            validity,
            presence,
            self.input_states.into(),
            self.input_provenance.into(),
            self.values,
        )
    }
}
