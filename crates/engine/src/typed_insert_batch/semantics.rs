//! Move-only typed INSERT semantic lowering.
//!
//! This is the pre-physical boundary: source-order diagnostics and coercion become catalog-order
//! vectors; deterministic defaults, RETURNING identities, domain bindings, and sequence requests
//! are then bound without selecting a resident route or advancing a sequence.

use super::*;

pub(crate) struct PreparedTypedInsert {
    table: TypedInsertBatchTable,
    statement_ordinal: InsertStatementOrdinal,
    row_count: u32,
    columns: Box<[TypedInsertColumn]>,
    dependencies: Box<[TypedInsertDependencyBinding]>,
    domain_dependencies: Box<[TypedInsertDomainBinding]>,
    canonical_catalog: TypedInsertCanonicalCatalog,
    typed_statement_digest: gpu_db_wal::CanonicalDigest,
    returning: returning::BoundInsertReturning,
    sequence_requests: sequence_defaults::SequenceDefaultRequests,
    /// Established bootstrap SQL policy: an omitted/DEFAULT cell for a column with no declared
    /// default is not a live INSERT.  Capture this while semantic inputs and catalog defaults are
    /// adjacent so later physical routes consume one scalar proof instead of rescanning rows.
    missing_required_input: bool,
}

/// Immutable target identity retained by typed semantic preparation for a future effect owner.
/// This scalar view deliberately excludes logical value vectors and physical plan inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
pub(crate) struct PreparedTypedInsertEffectTarget {
    table_oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    prepared_catalog_seq: Index,
}

#[allow(dead_code)] // The current live route intentionally has no effect-plan consumer.
impl PreparedTypedInsertEffectTarget {
    pub(crate) const fn table_oid(self) -> u32 {
        self.table_oid
    }

    pub(crate) const fn schema_digest(self) -> gpu_db_wal::CanonicalDigest {
        self.schema_digest
    }

    pub(crate) const fn prepared_catalog_seq(self) -> Index {
        self.prepared_catalog_seq
    }
}

impl PreparedTypedInsert {
    /// Domain `GPUDBTYPEDINSERTSTATEMENT1`: immutable pre-effect typed intent captured from the
    /// admitted catalog snapshot. It deliberately excludes transaction/receipt/private-outcome
    /// authority and replaces the legacy SQL/JSON statement identity for WRITE-001.
    pub(crate) const fn typed_statement_digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.typed_statement_digest
    }

    #[allow(dead_code)] // Reached by the inert test terminal through the narrow outer wrapper.
    pub(crate) fn canonical_returning_layout_digest(
        &self,
    ) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
        canonical_codec::returning_layout_digest(&self.returning)
    }

    /// Scalar target identity for the inert pre-WAL effect handoff. This borrows no value vector
    /// and cannot split the move-only semantic carrier.
    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(crate) const fn effect_target(&self) -> PreparedTypedInsertEffectTarget {
        PreparedTypedInsertEffectTarget {
            table_oid: self.table.oid,
            schema_digest: self.table.schema_digest,
            prepared_catalog_seq: self.table.prepared_catalog_seq,
        }
    }

    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(crate) const fn effect_statement_ordinal(&self) -> InsertStatementOrdinal {
        self.statement_ordinal
    }

    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(crate) const fn effect_row_count(&self) -> u32 {
        self.row_count
    }

    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(crate) fn effect_sequence_requests(
        &self,
    ) -> impl ExactSizeIterator<Item = sequence_defaults::SequenceDefaultRequestEffectShape<'_>> + '_
    {
        self.sequence_requests.effect_shapes()
    }

    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(crate) const fn effect_returning_shape(&self) -> returning::InsertReturningEffectShape {
        self.returning.effect_shape()
    }

    #[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
    pub(crate) fn effect_returning_projection_identities(
        &self,
    ) -> impl ExactSizeIterator<Item = returning::ReturningProjectionEffectIdentity<'_>> + '_ {
        self.returning.effect_projection_identities()
    }

    #[cfg(test)]
    pub(crate) fn returning(&self) -> &returning::BoundInsertReturning {
        &self.returning
    }

    pub(crate) fn sequence_requests(&self) -> &[sequence_defaults::SequenceDefaultRequest] {
        self.sequence_requests.requests()
    }

    /// Pure whole-bundle preflight for the inert WRITE-001 terminal. It shares the exact seal
    /// contract and performs no vector materialization, so a rejected terminal bundle leaves this
    /// move-only semantic carrier intact until the caller drops it.
    #[cfg(test)]
    pub(crate) fn validate_sequence_bindings_for_terminal(
        &self,
        bindings: &sequence_defaults::SequenceDefaultBindings,
    ) -> Result<(), ExecuteError> {
        sequence_defaults::validate_bindings(
            &self.columns,
            self.statement_ordinal,
            usize::try_from(self.row_count).expect("u32 typed INSERT row count is addressable"),
            &self.sequence_requests,
            bindings,
        )
    }

    /// Consume exact sequence effects and produce the one sealed batch carrier. The effect bundle
    /// cannot be reused after materialization, and every sequence request must bind 1:1 in the
    /// original row-major/catalog-column order.
    pub(crate) fn seal(
        mut self,
        sequence_bindings: sequence_defaults::SequenceDefaultBindings,
    ) -> Result<TypedInsertBatch, ExecuteError> {
        let rows =
            usize::try_from(self.row_count).expect("u32 always fits usize on supported hosts");
        let sequence_bindings = sequence_defaults::materialize(
            &mut self.columns,
            self.statement_ordinal,
            rows,
            &self.sequence_requests,
            sequence_bindings,
        )?;
        if !self.returning.geometry_is_exact(self.row_count)
            || self
                .columns
                .iter()
                .any(|column| !column.full_invariants_hold(rows))
        {
            return Err(EngineError::Durability(
                "typed INSERT semantic seal produced invalid column vectors".to_string(),
            )
            .into());
        }
        Ok(TypedInsertBatch {
            table: self.table,
            statement_ordinal: self.statement_ordinal,
            row_count: self.row_count,
            columns: self.columns,
            dependencies: self.dependencies,
            domain_dependencies: self.domain_dependencies,
            canonical_catalog: self.canonical_catalog,
            typed_statement_digest: self.typed_statement_digest,
            returning: self.returning,
            sequence_bindings,
            missing_required_input: self.missing_required_input,
        })
    }
}

/// Prepare the complete typed INSERT semantic contract before route eligibility is considered.
/// `None` retains the live adapter's stale-generation decline; every other supported SQL shape
/// receives either a move-only prepared carrier or its established semantic diagnostic.
#[allow(dead_code)] // Semantic/effect callers retain this stable direct ingress.
pub(crate) fn prepare_typed_insert_semantics(
    insert: &Insert,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
) -> Result<Option<PreparedTypedInsert>, ExecuteError> {
    prepare_typed_insert_semantics_at(
        insert,
        catalog,
        prepared_catalog_seq,
        expected_catalog_version,
        InsertStatementOrdinal::FIRST,
    )
}

/// Prepare the complete typed INSERT semantic contract at one exact statement position.
///
/// The established wrapper above deliberately remains the first/stable one-statement ingress;
/// transaction-overlay effect preparation supplies its own ordinal here without changing any
/// live physical-route eligibility.
pub(crate) fn prepare_typed_insert_semantics_at(
    insert: &Insert,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
    statement_ordinal: InsertStatementOrdinal,
) -> Result<Option<PreparedTypedInsert>, ExecuteError> {
    let Some(semantics) = resolve_pre_semantic_insert(
        insert,
        catalog,
        prepared_catalog_seq,
        expected_catalog_version,
        statement_ordinal,
    )?
    else {
        return Ok(None);
    };
    prepare_resolved_typed_insert_semantics(semantics, catalog, prepared_catalog_seq).map(Some)
}

/// Resolve the shared non-owning INSERT IR before a physical capability gate decides whether a
/// typed vector should exist.  It preserves expected-catalog, generation, relation, target-list,
/// arity, and parameter diagnostics without evaluating a scalar default or binding a result.
pub(super) fn resolve_pre_semantic_insert<'a>(
    insert: &'a Insert,
    catalog: &'a CatalogSnapshot,
    prepared_catalog_seq: Index,
    expected_catalog_version: Option<crate::engine_mutation_admission::CatalogVersionExpectation>,
    statement_ordinal: InsertStatementOrdinal,
) -> Result<Option<ResolvedInsertSemantics<'a>>, ExecuteError> {
    if let Some(expectation) = expected_catalog_version {
        crate::engine_mutation_admission::validate_catalog_version_expectation(
            expectation,
            catalog.commit_seq,
        )?;
    }
    if catalog.commit_seq != prepared_catalog_seq {
        return Ok(None);
    }
    let table = catalog
        .relational_catalog
        .get(&insert.table)
        .ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", insert.table))
        })?;
    let semantics = ResolvedInsertSemantics::resolve(insert, table, statement_ordinal)?;
    debug_assert_eq!(semantics.table.name, table.name);
    Ok(Some(semantics))
}

pub(super) fn prepare_resolved_typed_insert_semantics(
    semantics: ResolvedInsertSemantics<'_>,
    catalog: &CatalogSnapshot,
    prepared_catalog_seq: Index,
) -> Result<PreparedTypedInsert, ExecuteError> {
    let table = semantics.table;
    let row_count = semantics.row_count;
    let rows = usize::try_from(row_count).expect("u32 always fits usize on supported hosts");
    let primary_key_columns = table
        .indexes
        .iter()
        .filter(|index| index.primary_key)
        .flat_map(|index| index.key_columns.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let missing_required_input = semantics.columns.iter().any(|semantic_column| {
        semantic_column.column.default.is_none()
            // PRIMARY KEY omission materializes NULL and is owned by the ordinary 23502
            // constraint arbitration.  Do not replace that higher-precedence diagnostic with
            // the nullable bootstrap-column policy below.
            && !primary_key_columns.contains(semantic_column.column.name.as_str())
            && semantic_column.cells.iter().any(|cell| {
                matches!(
                    cell.input,
                    ResolvedInsertInput::Omitted | ResolvedInsertInput::ExplicitDefault { .. }
                )
            })
    });
    let domain_dependencies = collect_domain_dependencies(table, catalog)?;
    let domain_ordinals = domain_dependencies
        .iter()
        .enumerate()
        .map(|(ordinal, binding)| {
            u32::try_from(ordinal)
                .map(|ordinal| (binding.name.as_ref(), ordinal))
                .map_err(|_| {
                    EngineError::Durability(
                        "typed INSERT domain dependency count exceeds u32".to_string(),
                    )
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
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
    for row_index in 0..rows {
        for &(_, catalog_index) in &source_columns {
            let semantic_column = &semantics.columns[catalog_index];
            let builder = &mut builders[catalog_index];
            let column = semantic_column.column;
            debug_assert_eq!(semantic_column.catalog_ordinal as usize, catalog_index);
            let cell = &semantic_column.cells[row_index];
            let provenance = TypedInsertInputProvenance::from_input(cell.input)?;
            match cell.input {
                ResolvedInsertInput::Provided { value, .. } => {
                    builder.set_provided(
                        row_index,
                        coerce_insert_value(value.clone(), column.ty, &column.name)?,
                        column.ty,
                        provenance,
                    )?;
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

    let mut columns = table
        .columns
        .iter()
        .zip(&semantics.columns)
        .zip(builders)
        .map(|((column, semantic_column), builder)| -> Result<TypedInsertColumn, EngineError> {
            let (validity, presence, input_states, input_provenance, values) = builder.finish(rows);
            Ok(TypedInsertColumn {
                name: Arc::from(column.name.as_str()),
                column_id: column.id,
                attnum: column.attnum,
                ty: column.ty,
                type_oid: column.type_oid,
                type_size: column.type_size,
                source_column_ordinal: semantic_column.source_column_ordinal,
                domain_dependency_ordinal: column
                    .domain
                    .as_deref()
                    .map(|name| {
                        domain_ordinals.get(name).copied().ok_or_else(|| {
                            EngineError::Durability(
                                "typed INSERT domain ordinal disappeared during semantic preparation"
                                    .to_string(),
                            )
                        })
                    })
                    .transpose()?,
                validity,
                presence,
                input_states,
                input_provenance,
                default_resolution: TypedInsertDefaultResolution::AllDirect,
                values,
                #[cfg(test)]
                full_invariant_scans: std::sync::atomic::AtomicUsize::new(0),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // RETURNING is bound only after every supplied scalar has had its established coercion
    // diagnostic. Its undefined-column error in turn precedes default/sequence binding.
    let returning = returning::bind(table, semantics.returning, row_count)?;
    defaults::resolve(&mut columns, table, rows).map_err(|error| match error {
        defaults::DefaultResolutionError::Engine(error) => ExecuteError::Engine(error),
    })?;
    let sequence_requests =
        sequence_defaults::discover(&columns, table, catalog, semantics.statement_ordinal, rows)?;
    let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)?;
    let (dependencies, canonical_catalog) =
        collect_canonical_catalog(table, catalog, schema_digest)?;
    let typed_statement_digest = canonical_codec::typed_statement_digest_for_prepared(
        canonical_codec::TypedStatementDigestInput {
            table,
            statement_ordinal: semantics.statement_ordinal,
            row_count,
            columns: &columns,
            dependencies: &dependencies,
            domains: &domain_dependencies,
            canonical_catalog: &canonical_catalog,
            returning: &returning,
            sequence_requests: &sequence_requests,
        },
    )?;
    Ok(PreparedTypedInsert {
        table: TypedInsertBatchTable {
            schema: Arc::from(table.schema.as_str()),
            name: Arc::from(table.name.as_str()),
            stable_table_id: table.stable_table_id,
            oid: table.oid,
            schema_digest,
            prepared_catalog_seq,
        },
        statement_ordinal: semantics.statement_ordinal,
        row_count,
        columns: columns.into(),
        dependencies,
        domain_dependencies,
        canonical_catalog,
        typed_statement_digest,
        returning,
        sequence_requests,
        missing_required_input,
    })
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
                schema: Arc::from(domain.schema.as_str()),
                name: Arc::from(name),
                oid: domain.oid,
                base_type: domain.base_type,
            });
        }
    }
    Ok(dependencies.into())
}

/// Capture the complete logical catalog closure while semantic preparation still owns the exact
/// snapshot.  The canonical codec consumes only this inert copy; it must never re-open a catalog
/// or infer an FK parent/index from a later generation.
fn collect_canonical_catalog(
    table: &RelationalTable,
    catalog: &CatalogSnapshot,
    target_schema_digest: gpu_db_wal::CanonicalDigest,
) -> Result<
    (
        Box<[TypedInsertDependencyBinding]>,
        TypedInsertCanonicalCatalog,
    ),
    ExecuteError,
> {
    let mut dependencies = vec![TypedInsertDependencyBinding {
        schema: Arc::from(table.schema.as_str()),
        name: Arc::from(table.name.as_str()),
        oid: table.oid,
        schema_digest: target_schema_digest,
    }];
    let mut dependency_by_oid = BTreeMap::from([(table.oid, 0_u32)]);
    let mut index_oids = BTreeSet::new();
    let mut index_names = BTreeSet::new();
    let indexes = table
        .indexes
        .iter()
        .enumerate()
        .map(|(raw_ordinal, index)| {
            if !index_oids.insert(index.oid) || !index_names.insert(index.name.as_str()) {
                return Err(EngineError::ApplyFailed(
                    "typed INSERT canonical catalog has duplicate target index identity"
                        .to_string(),
                )
                .into());
            }
            canonical_index_binding(table, 0, raw_ordinal, index)
        })
        .collect::<Result<Vec<_>, ExecuteError>>()?;

    let mut foreign_key_names = BTreeSet::new();
    let mut foreign_keys = Vec::with_capacity(table.foreign_keys.len());
    for (raw_ordinal, foreign_key) in table.foreign_keys.iter().enumerate() {
        if foreign_key.name.is_empty() || !foreign_key_names.insert(foreign_key.name.as_str()) {
            return Err(EngineError::ApplyFailed(
                "typed INSERT canonical catalog has duplicate or empty foreign-key identity"
                    .to_string(),
            )
            .into());
        }
        let child_column = canonical_column_binding(table, 0, &foreign_key.column)?;
        let parent = catalog
            .relational_catalog
            .get(&foreign_key.referenced_table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "typed INSERT canonical foreign-key parent is absent".to_string(),
                )
            })?;
        let parent_dependency_ordinal = if let Some(ordinal) = dependency_by_oid.get(&parent.oid) {
            *ordinal
        } else {
            let ordinal = u32::try_from(dependencies.len()).map_err(|_| {
                EngineError::Durability(
                    "typed INSERT canonical dependency count exceeds u32".to_string(),
                )
            })?;
            let parent_schema_digest =
                crate::engine_transaction_reset::table_schema_digest(parent)?;
            dependencies.push(TypedInsertDependencyBinding {
                schema: Arc::from(parent.schema.as_str()),
                name: Arc::from(parent.name.as_str()),
                oid: parent.oid,
                schema_digest: parent_schema_digest,
            });
            dependency_by_oid.insert(parent.oid, ordinal);
            ordinal
        };
        let parent_column = canonical_column_binding(
            parent,
            parent_dependency_ordinal,
            &foreign_key.referenced_column,
        )?;
        // Keep the existing FK semantic contract: compatible domain/base columns share the
        // resolved SQL type, while their catalog OIDs may legitimately differ.
        if child_column.ty != parent_column.ty {
            return Err(EngineError::ApplyFailed(
                "typed INSERT canonical foreign-key columns have incompatible types".to_string(),
            )
            .into());
        }
        let (supporting_ordinal, supporting) = parent
            .indexes
            .iter()
            .enumerate()
            .find(|(_, index)| {
                index.table == parent.name
                    && index.column == foreign_key.referenced_column
                    && index.unique
                    && (index.primary_key || index.unique_constraint)
                    && index.key_columns.as_slice() == [foreign_key.referenced_column.as_str()]
            })
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "typed INSERT canonical foreign-key parent has no single-column UNIQUE/PRIMARY index"
                        .to_string(),
                )
            })?;
        foreign_keys.push(TypedInsertCanonicalForeignKeyBinding {
            raw_ordinal: u32::try_from(raw_ordinal).map_err(|_| {
                EngineError::Durability(
                    "typed INSERT canonical foreign-key ordinal exceeds u32".to_string(),
                )
            })?,
            name: Arc::from(foreign_key.name.as_str()),
            child_column_name: Arc::from(foreign_key.column.as_str()),
            referenced_table_name: Arc::from(foreign_key.referenced_table.as_str()),
            referenced_column_name: Arc::from(foreign_key.referenced_column.as_str()),
            child_column,
            parent_dependency_ordinal,
            parent_column,
            supporting_index: canonical_index_binding(
                parent,
                parent_dependency_ordinal,
                supporting_ordinal,
                supporting,
            )?,
        });
    }
    Ok((
        dependencies.into(),
        TypedInsertCanonicalCatalog {
            indexes: indexes.into(),
            foreign_keys: foreign_keys.into(),
        },
    ))
}

fn canonical_column_binding(
    table: &RelationalTable,
    dependency_ordinal: u32,
    name: &str,
) -> Result<TypedInsertCanonicalColumnBinding, ExecuteError> {
    let (catalog_column_ordinal, column) = table
        .columns
        .iter()
        .enumerate()
        .find(|(_, column)| column.name == name)
        .ok_or_else(|| {
            EngineError::ApplyFailed(
                "typed INSERT canonical catalog column identity is absent".to_string(),
            )
        })?;
    if column.id == 0
        || column.table_oid != table.oid
        || column.type_oid == 0
        || column.type_size != column.ty.type_size()
    {
        return Err(EngineError::ApplyFailed(
            "typed INSERT canonical catalog column identity is inconsistent".to_string(),
        )
        .into());
    }
    Ok(TypedInsertCanonicalColumnBinding {
        dependency_ordinal,
        catalog_column_ordinal: u32::try_from(catalog_column_ordinal).map_err(|_| {
            EngineError::Durability(
                "typed INSERT canonical catalog column ordinal exceeds u32".to_string(),
            )
        })?,
        column_id: column.id,
        attnum: column.attnum,
        name: Arc::from(column.name.as_str()),
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
    })
}

fn canonical_index_binding(
    table: &RelationalTable,
    owner_dependency_ordinal: u32,
    raw_ordinal: usize,
    index: &RelationalIndex,
) -> Result<TypedInsertCanonicalIndexBinding, ExecuteError> {
    if index.oid == 0
        || index.name.is_empty()
        || index.table != table.name
        || index.key_columns.is_empty()
        || index.column != index.key_columns[0]
    {
        return Err(EngineError::ApplyFailed(
            "typed INSERT canonical catalog index identity is inconsistent".to_string(),
        )
        .into());
    }
    let mut key_columns = Vec::with_capacity(index.key_columns.len());
    for name in &index.key_columns {
        key_columns.push(canonical_column_binding(
            table,
            owner_dependency_ordinal,
            name,
        )?);
    }
    Ok(TypedInsertCanonicalIndexBinding {
        owner_dependency_ordinal,
        raw_ordinal: u32::try_from(raw_ordinal).map_err(|_| {
            EngineError::Durability("typed INSERT canonical index ordinal exceeds u32".to_string())
        })?,
        oid: index.oid,
        name: Arc::from(index.name.as_str()),
        table_name: Arc::from(index.table.as_str()),
        first_column_name: Arc::from(index.column.as_str()),
        key_columns: key_columns.into(),
        unique: index.unique,
        primary_key: index.primary_key,
        unique_constraint: index.unique_constraint,
    })
}

struct TypedInsertColumnBuilder {
    validity_words: Vec<u32>,
    presence_words: Vec<u32>,
    input_states: Vec<TypedInsertInputState>,
    input_provenance: Vec<TypedInsertInputProvenance>,
    values: TypedInsertColumnValues,
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
                Ok(Self::BoundParameter {
                    index: u32::try_from(index).map_err(|_| {
                        EngineError::ApplyFailed(
                            "bound INSERT parameter index exceeds compact semantic metadata"
                                .to_string(),
                        )
                    })?,
                })
            }
            gpu_db_sql::InsertValueProvenance::Programmatic => Ok(Self::ProgrammaticValue),
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
