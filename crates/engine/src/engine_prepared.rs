//! Engine-owned prepared-command description.
//!
//! This is the catalog/type boundary for every client protocol. It resolves SQL parameter slots
//! and result columns against one committed catalog generation without executing the command;
//! wire OIDs and format codes remain above the engine.

use super::*;
use gpu_db_sql::SelectFilter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCommandDescription {
    /// Catalog generation used for both parameter inference and result description. Prepared
    /// execution carries this neutral stamp back to mutation admission so a DDL race cannot
    /// change the row type between Bind and durable publication.
    pub catalog_version: u64,
    pub parameter_types: Vec<SqlType>,
    pub result_columns: Vec<RelationalColumn>,
}

impl Engine {
    pub fn describe_prepared_command(
        &self,
        prepared: &PreparedCommand,
        parameter_type_hints: &[Option<SqlType>],
    ) -> Result<PreparedCommandDescription, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        self.describe_prepared_command_in_catalog(prepared, parameter_type_hints, &catalog)
    }

    /// Resolve prepared metadata as one statement of an active explicit transaction. READ
    /// COMMITTED refresh and REPEATABLE READ retention use the same statement-snapshot owner as
    /// execution, while a staged catalog overlay remains visible only through that transaction.
    pub fn describe_prepared_command_in_transaction(
        &self,
        txn_id: TxnId,
        prepared: &PreparedCommand,
        parameter_type_hints: &[Option<SqlType>],
    ) -> Result<PreparedCommandDescription, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        let catalog = snapshot.transaction_catalog();
        self.describe_prepared_command_in_catalog(prepared, parameter_type_hints, &catalog)
    }

    pub(crate) fn describe_prepared_command_in_catalog(
        &self,
        prepared: &PreparedCommand,
        parameter_type_hints: &[Option<SqlType>],
        catalog: &CatalogSnapshot,
    ) -> Result<PreparedCommandDescription, ExecuteError> {
        // PostgreSQL treats an explicit Parse OID array as the declared parameter arity even when
        // trailing positions are unused by the SQL text. Preserve those typed positions for Bind;
        // AST inference still owns the referenced prefix.
        let mut parameter_types =
            vec![None; prepared.parameter_count().max(parameter_type_hints.len())];
        for (slot, hint) in parameter_types
            .iter_mut()
            .zip(parameter_type_hints.iter().copied())
        {
            *slot = hint;
        }

        let result_columns = match prepared.command() {
            Command::ShowTransactionIsolation => vec![RelationalColumn {
                id: 0,
                table_oid: 0,
                attnum: 0,
                name: "transaction_isolation".to_string(),
                ty: SqlType::Text,
                domain: None,
                default: None,
                type_oid: SqlType::Text.postgres_oid(),
                type_size: SqlType::Text.type_size(),
            }],
            Command::SelectLiteral(literal) => {
                infer_value(&literal.value, literal.ty, &mut parameter_types)?;
                vec![RelationalColumn {
                    id: 0,
                    table_oid: 0,
                    attnum: 0,
                    name: literal.column_name.clone(),
                    ty: literal.ty,
                    domain: None,
                    default: None,
                    type_oid: literal.ty.postgres_oid(),
                    type_size: literal.ty.type_size(),
                }]
            }
            Command::SequenceNextVal(_) => vec![sequence_result_column("nextval")],
            Command::SequenceCurrVal(_) => vec![sequence_result_column("currval")],
            Command::SequenceSetVal(_) => vec![sequence_result_column("setval")],
            Command::PreparedCatalog(program) => {
                let parameter = match program {
                    PreparedCatalogProgram::Pg16DomainConstraints { type_oid }
                    | PreparedCatalogProgram::Pg16DomainDefinition { type_oid } => type_oid,
                    PreparedCatalogProgram::Pg16FunctionDefinition { function_oid } => function_oid,
                    PreparedCatalogProgram::Pg16MaterializedViewDependencies => {
                        return Ok(PreparedCommandDescription {
                            catalog_version: catalog.commit_seq,
                            parameter_types: Vec::new(),
                            result_columns: crate::engine_sql_pg::pg_dump_catalog::pg16_prepared_catalog_program_table(program)
                                .columns,
                        });
                    }
                };
                infer_value(parameter, SqlType::Int4, &mut parameter_types)?;
                crate::engine_sql_pg::pg_dump_catalog::pg16_prepared_catalog_program_table(program)
                    .columns
            }
            Command::Select(select) => {
                let table = prepared_select_table(catalog, select)?;
                validate_prepared_catalog_select_shape(&table, select)?;
                // NULL binds safely through every explicit cast and predicate, allowing the
                // existing SELECT binder to remain the sole owner of projection/aggregate types.
                let nulls = vec![SqlValue::Null; prepared.parameter_count()];
                let bound_command = prepared.bind(&nulls)?;
                let Command::Select(bound_select) = bound_command.command() else {
                    unreachable!("a prepared SELECT binds to SELECT")
                };
                let bound = bind_relational_select(&table, bound_select)?;
                infer_select_parameters(
                    &table,
                    select,
                    &bound.selected_columns,
                    &mut parameter_types,
                )?;
                bound.selected_columns
            }
            Command::Insert(insert) => {
                let table = prepared_table(catalog, &insert.table)?;
                infer_insert_parameters(table, insert, &mut parameter_types)?;
                returning_columns(table, &insert.returning)?
            }
            Command::Update(update) => {
                let table = prepared_table(catalog, &update.table)?;
                for assignment in &update.assignments {
                    let ty = prepared_column(table, &assignment.column)?.ty;
                    infer_value(&assignment.value, ty, &mut parameter_types)?;
                }
                infer_filter_parameters(
                    table,
                    update.filter.as_ref(),
                    &update.filters,
                    &update.filter_groups,
                    &mut parameter_types,
                )?;
                returning_columns(table, &update.returning)?
            }
            Command::Delete(delete) => {
                let table = prepared_table(catalog, &delete.table)?;
                infer_filter_parameters(
                    table,
                    delete.filter.as_ref(),
                    &delete.filters,
                    &delete.filter_groups,
                    &mut parameter_types,
                )?;
                returning_columns(table, &delete.returning)?
            }
            _ => Vec::new(),
        };

        let parameter_types = parameter_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| ty.ok_or(ExecuteError::IndeterminateParameterType(index + 1)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedCommandDescription {
            catalog_version: catalog.commit_seq,
            parameter_types,
            result_columns,
        })
    }
}

fn sequence_result_column(name: &str) -> RelationalColumn {
    RelationalColumn {
        id: 0,
        table_oid: 0,
        attnum: 0,
        name: name.to_string(),
        ty: SqlType::Int8,
        domain: None,
        default: None,
        type_oid: SqlType::Int8.postgres_oid(),
        type_size: SqlType::Int8.type_size(),
    }
}

fn validate_prepared_catalog_select_shape(
    table: &RelationalTable,
    select: &Select,
) -> Result<(), ExecuteError> {
    if !matches!(table.schema.as_str(), "pg_catalog" | "information_schema") {
        return Ok(());
    }
    let row_projection = matches!(
        &select.projection,
        SelectProjection::All | SelectProjection::Columns(_)
    );
    if row_projection
        && !select.distinct
        && select.group_by.is_none()
        && select.having_groups.is_empty()
    {
        return Ok(());
    }
    Err(ExecuteError::Engine(EngineError::ApplyFailed(
        "prepared catalog SELECT aggregates, DISTINCT, grouping, and HAVING require the general catalog binder and are not supported at Parse/Describe"
            .to_string(),
    )))
}

fn prepared_select_table(
    catalog: &CatalogSnapshot,
    select: &Select,
) -> Result<RelationalTable, ExecuteError> {
    let name = select.table.as_str();
    if let Some(table) = catalog.relational_catalog.get(name) {
        return Ok(table.clone());
    }
    if catalog.relational_views.contains_key(name) {
        if !select_is_plain_view_scan(select) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "prepared SELECT over public relation {name:?} requires a supported base-table route"
            ))));
        }
        return prepared_view_table(catalog, name, &mut BTreeSet::new());
    }
    if select.public_only {
        return Err(ExecuteError::UndefinedRelation(format!("public.{name}")));
    }
    if public_relation_name_exists(catalog, name) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "prepared SELECT over public relation {name:?} requires a supported base-table route"
        ))));
    }
    synthesize_catalog_relation(name, catalog)
        .map(|(table, _rows)| table)
        .ok_or_else(|| ExecuteError::UndefinedRelation(name.to_string()))
}

fn prepared_view_table(
    catalog: &CatalogSnapshot,
    name: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<RelationalTable, ExecuteError> {
    if !visiting.insert(name.to_string()) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "view dependency cycle is unsupported".to_string(),
        )));
    }
    let result = (|| {
        let view = catalog
            .relational_views
            .get(name)
            .ok_or_else(|| ExecuteError::UndefinedRelation(name.to_string()))?;
        let source = if let Some(table) = catalog.relational_catalog.get(&view.query.table) {
            table.clone()
        } else if catalog.relational_views.contains_key(&view.query.table) {
            if !select_is_plain_view_scan(&view.query) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "only plain SELECT * FROM view is supported for views".to_string(),
                )));
            }
            prepared_view_table(catalog, &view.query.table, visiting)?
        } else {
            return Err(ExecuteError::UndefinedRelation(view.query.table.clone()));
        };
        let columns = bind_relational_select(&source, &view.query)?.selected_columns;
        Ok(RelationalTable {
            schema: view.schema.clone(),
            name: view.name.clone(),
            // A view has a display OID but owns no table data generation.
            stable_table_id: 0,
            oid: view.oid,
            columns,
            indexes: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
            acl: view.acl.clone(),
        })
    })();
    visiting.remove(name);
    result
}

fn prepared_table<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<&'a RelationalTable, ExecuteError> {
    catalog
        .relational_catalog
        .get(name)
        .ok_or_else(|| ExecuteError::UndefinedRelation(name.to_string()))
}

fn prepared_column<'a>(
    table: &'a RelationalTable,
    name: &str,
) -> Result<&'a RelationalColumn, ExecuteError> {
    table
        .columns
        .iter()
        .find(|column| column.name == name)
        .ok_or_else(|| ExecuteError::UndefinedColumn(name.to_string()))
}

fn returning_columns(
    table: &RelationalTable,
    names: &[String],
) -> Result<Vec<RelationalColumn>, ExecuteError> {
    names
        .iter()
        .map(|name| prepared_column(table, name).cloned())
        .collect()
}

fn infer_select_parameters(
    table: &RelationalTable,
    select: &Select,
    result_columns: &[RelationalColumn],
    parameter_types: &mut [Option<SqlType>],
) -> Result<(), ExecuteError> {
    infer_filter_parameters(
        table,
        select.filter.as_ref(),
        &select.filters,
        &select.filter_groups,
        parameter_types,
    )?;
    for group in &select.having_groups {
        for filter in group {
            let column = result_columns
                .iter()
                .find(|column| column.name == filter.column)
                .ok_or_else(|| ExecuteError::UndefinedColumn(filter.column.clone()))?;
            infer_value(&filter.value, column.ty, parameter_types)?;
        }
    }
    if let Some(index) = select.prepared_limit_parameter_index() {
        infer_parameter(index, None, SqlType::Int4, parameter_types)?;
    }
    Ok(())
}

fn infer_insert_parameters(
    table: &RelationalTable,
    insert: &Insert,
    parameter_types: &mut [Option<SqlType>],
) -> Result<(), ExecuteError> {
    let columns = if insert.columns.is_empty() {
        table.columns.iter().collect::<Vec<_>>()
    } else {
        let mut seen = BTreeSet::new();
        let mut columns = Vec::with_capacity(insert.columns.len());
        for name in &insert.columns {
            if !seen.insert(name) {
                return Err(ExecuteError::Engine(EngineError::DuplicateColumn(
                    name.clone(),
                )));
            }
            columns.push(prepared_column(table, name)?);
        }
        columns
    };
    for row in &insert.rows {
        if row.len() != columns.len() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "INSERT row has a different column count than its target list".to_string(),
            )));
        }
        for (cell, column) in row.iter().zip(&columns) {
            if let Some(value) = cell.value() {
                infer_value(value, column.ty, parameter_types)?;
            }
        }
    }
    Ok(())
}

fn infer_filter_parameters(
    table: &RelationalTable,
    filter: Option<&SelectFilter>,
    filters: &[SelectFilter],
    filter_groups: &[Vec<SelectFilter>],
    parameter_types: &mut [Option<SqlType>],
) -> Result<(), ExecuteError> {
    if let Some(filter) = filter {
        infer_filter_parameter(table, filter, parameter_types)?;
    }
    for filter in filters {
        infer_filter_parameter(table, filter, parameter_types)?;
    }
    for group in filter_groups {
        for filter in group {
            infer_filter_parameter(table, filter, parameter_types)?;
        }
    }
    Ok(())
}

fn infer_filter_parameter(
    table: &RelationalTable,
    filter: &SelectFilter,
    parameter_types: &mut [Option<SqlType>],
) -> Result<(), ExecuteError> {
    let ty = prepared_column(table, &filter.column)?.ty;
    infer_value(&filter.value, ty, parameter_types)
}

fn infer_value(
    value: &SqlValue,
    context: SqlType,
    parameter_types: &mut [Option<SqlType>],
) -> Result<(), ExecuteError> {
    let SqlValue::Parameter { index, cast } = value else {
        return Ok(());
    };
    infer_parameter(*index, *cast, context, parameter_types)
}

fn infer_parameter(
    index: usize,
    cast: Option<SqlType>,
    context: SqlType,
    parameter_types: &mut [Option<SqlType>],
) -> Result<(), ExecuteError> {
    let inferred = cast.unwrap_or(context);
    if cast.is_some() && !same_type_family(inferred, context) {
        return Err(ExecuteError::DatatypeMismatch(format!(
            "parameter ${index} cast type {} does not match target type {}",
            inferred.catalog_name(),
            context.catalog_name()
        )));
    }
    let slot = parameter_types
        .get_mut(index.saturating_sub(1))
        .ok_or(ExecuteError::Parse(ParseError::InvalidParameterReference))?;
    if let Some(existing) = *slot {
        if !same_type_family(existing, inferred) {
            return Err(ExecuteError::DatatypeMismatch(format!(
                "inconsistent types deduced for parameter ${index}: {} and {}",
                existing.catalog_name(),
                inferred.catalog_name()
            )));
        }
    } else {
        *slot = Some(inferred);
    }
    Ok(())
}

fn same_type_family(left: SqlType, right: SqlType) -> bool {
    matches!(
        (left, right),
        (SqlType::Numeric { .. }, SqlType::Numeric { .. })
    ) || left == right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn description_resolves_repeated_out_of_order_parameters_and_returning() {
        let engine = Engine::new_local();
        engine
            .execute_text(
                1,
                "CREATE TABLE accounts (tenant_id int4, account_id int8, balance int8, \
                 PRIMARY KEY (tenant_id, account_id))",
            )
            .unwrap();
        let prepared = PreparedCommand::parse(
            "UPDATE accounts SET balance = balance + $2 WHERE tenant_id = $1 \
             AND account_id = $2 RETURNING balance",
        )
        .unwrap();
        let description = engine.describe_prepared_command(&prepared, &[]).unwrap();
        assert_eq!(
            description.parameter_types,
            vec![SqlType::Int4, SqlType::Int8]
        );
        assert_eq!(description.result_columns.len(), 1);
        assert_eq!(description.result_columns[0].name, "balance");
        assert_eq!(description.result_columns[0].ty, SqlType::Int8);
    }

    #[test]
    fn description_infers_an_int4_parameterized_limit() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE limited_accounts (id int4)")
            .unwrap();
        let prepared = PreparedCommand::parse(
            "SELECT id FROM limited_accounts WHERE id >= $1 ORDER BY id LIMIT $2",
        )
        .unwrap();
        let description = engine.describe_prepared_command(&prepared, &[]).unwrap();
        assert_eq!(
            description.parameter_types,
            vec![SqlType::Int4, SqlType::Int4]
        );
    }

    #[test]
    fn description_infers_prepared_int4_scalar_addition() {
        let engine = Engine::new_local();
        let prepared = PreparedCommand::parse("SELECT $1 + 10 AS plus_ten").unwrap();
        let description = engine.describe_prepared_command(&prepared, &[]).unwrap();
        assert_eq!(description.parameter_types, vec![SqlType::Int4]);
        assert_eq!(description.result_columns[0].name, "plus_ten");
        assert_eq!(description.result_columns[0].ty, SqlType::Int4);
    }

    #[test]
    fn description_expands_catalog_mixed_star_against_one_snapshot() {
        let engine = Engine::new_local();
        let prepared = PreparedCommand::parse(
            "SELECT oid, * FROM pg_catalog.pg_type \
             WHERE typname IN ('hstore','geometry','vector')",
        )
        .unwrap();
        let description = engine.describe_prepared_command(&prepared, &[]).unwrap();
        assert!(description.parameter_types.is_empty());
        assert_eq!(
            description
                .result_columns
                .iter()
                .map(|column| column.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "oid",
                "oid",
                "typname",
                "typlen",
                "typtype",
                "typnamespace",
                "typbasetype",
            ]
        );
        assert_eq!(
            description.catalog_version,
            engine.catalog_snapshot().commit_seq
        );

        let unsupported =
            PreparedCommand::parse("SELECT count(*) FROM pg_catalog.pg_policy ORDER BY oid")
                .unwrap();
        let error = engine
            .describe_prepared_command(&unsupported, &[])
            .expect_err("prepared catalog aggregate/order must reject during Describe");
        assert!(error.to_string().contains("Parse/Describe"), "{error}");
    }

    #[test]
    fn description_retains_explicit_public_identity_across_drop_aba() {
        let engine = Engine::new_local();
        let prepared =
            PreparedCommand::parse("SELECT oid FROM public.pg_type ORDER BY oid").unwrap();
        let missing = engine
            .describe_prepared_command(&prepared, &[])
            .expect_err("explicit public lookup must not synthesize pg_catalog.pg_type");
        assert!(missing.to_string().contains("public.pg_type"), "{missing}");

        engine
            .execute_text(1, "CREATE TABLE pg_type (oid INT)")
            .unwrap();
        let description = engine.describe_prepared_command(&prepared, &[]).unwrap();
        assert_eq!(description.result_columns.len(), 1);
        assert_eq!(description.result_columns[0].name, "oid");

        engine.execute_text(2, "DROP TABLE pg_type").unwrap();
        let dropped = engine
            .describe_prepared_command(&prepared, &[])
            .expect_err("drop must not rebind explicit public to a shape-compatible catalog");
        assert!(dropped.to_string().contains("public.pg_type"), "{dropped}");
    }

    #[test]
    fn description_never_synthesizes_behind_public_view_matview_or_sequence_names() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE catalog_shadow_source (oid INT)")
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE VIEW pg_class AS SELECT oid FROM catalog_shadow_source",
            )
            .unwrap();
        engine
            .execute_text(
                3,
                "CREATE MATERIALIZED VIEW pg_type AS SELECT oid FROM catalog_shadow_source WITH NO DATA",
            )
            .unwrap();
        engine.execute_text(4, "CREATE SEQUENCE pg_policy").unwrap();
        engine
            .execute_text(
                5,
                "CREATE INDEX pg_attribute ON catalog_shadow_source (oid)",
            )
            .unwrap();
        engine
            .execute_text(6, "CREATE INDEX pg_trigger ON catalog_shadow_source (oid)")
            .unwrap();

        for sql in [
            "SELECT oid FROM pg_class",
            "SELECT oid FROM pg_type",
            "SELECT oid FROM pg_policy",
            "SELECT oid FROM pg_attribute",
            "SELECT oid FROM pg_trigger",
        ] {
            let prepared = PreparedCommand::parse(sql).unwrap();
            let error = engine
                .describe_prepared_command(&prepared, &[])
                .expect_err("a public relation or index owner must block bare catalog synthesis");
            assert!(
                error.to_string().contains("supported base-table route"),
                "{sql}: {error}"
            );
        }

        let explicit = PreparedCommand::parse("SELECT oid FROM pg_catalog.pg_type").unwrap();
        assert!(engine.describe_prepared_command(&explicit, &[]).is_ok());
    }

    #[test]
    fn description_exposes_typed_literal_projection_without_catalog_lookup() {
        let engine = Engine::new_local();
        let prepared = PreparedCommand::parse("SELECT 1 AS one").unwrap();
        let description = engine.describe_prepared_command(&prepared, &[]).unwrap();
        assert!(description.parameter_types.is_empty());
        assert_eq!(description.result_columns.len(), 1);
        assert_eq!(description.result_columns[0].name, "one");
        assert_eq!(description.result_columns[0].ty, SqlType::Int4);
        assert_eq!(description.result_columns[0].table_oid, 0);
        assert_eq!(description.result_columns[0].attnum, 0);
    }

    #[test]
    fn description_exposes_int8_sequence_value_columns() {
        let engine = Engine::new_local();
        for (sql, name) in [
            ("SELECT nextval('described_sequence'::regclass)", "nextval"),
            ("SELECT currval('described_sequence'::regclass)", "currval"),
            (
                "SELECT setval('described_sequence'::regclass, 9, false)",
                "setval",
            ),
        ] {
            let prepared = PreparedCommand::parse(sql).unwrap();
            let description = engine.describe_prepared_command(&prepared, &[]).unwrap();
            assert!(description.parameter_types.is_empty());
            assert_eq!(description.result_columns.len(), 1);
            assert_eq!(description.result_columns[0].name, name);
            assert_eq!(description.result_columns[0].ty, SqlType::Int8);
            assert_eq!(description.result_columns[0].table_oid, 0);
            assert_eq!(description.result_columns[0].attnum, 0);
        }
    }

    #[test]
    fn description_requires_a_type_for_unused_gap() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4 PRIMARY KEY)")
            .unwrap();
        let prepared = PreparedCommand::parse("SELECT id FROM accounts WHERE id = $2").unwrap();
        assert!(engine.describe_prepared_command(&prepared, &[]).is_err());
        assert_eq!(
            engine
                .describe_prepared_command(&prepared, &[Some(SqlType::Text)])
                .unwrap()
                .parameter_types,
            vec![SqlType::Text, SqlType::Int4]
        );
    }

    #[test]
    fn description_preserves_typed_unused_trailing_parse_parameters() {
        let engine = Engine::new_local();
        let prepared = PreparedCommand::parse("BEGIN").unwrap();
        let description = engine
            .describe_prepared_command(&prepared, &[Some(SqlType::Int4), Some(SqlType::Uuid)])
            .unwrap();
        assert_eq!(
            description.parameter_types,
            vec![SqlType::Int4, SqlType::Uuid]
        );
    }

    #[test]
    fn active_transaction_description_uses_only_its_private_catalog() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(
                40,
                gpu_db_sql::ParsedCommand::parse("BEGIN READ WRITE").unwrap(),
            )
            .unwrap();
        engine
            .submit_transaction(
                40,
                gpu_db_sql::ParsedCommand::parse(
                    "CREATE TABLE private_describe_predecessor (id int4)",
                )
                .unwrap(),
            )
            .unwrap();
        engine
            .submit_transaction(
                40,
                gpu_db_sql::ParsedCommand::parse(
                    "CREATE TABLE private_describe (id int4 PRIMARY KEY, value text)",
                )
                .unwrap(),
            )
            .unwrap();

        let insert = PreparedCommand::parse(
            "INSERT INTO private_describe VALUES ($1, $2) RETURNING id, value",
        )
        .unwrap();
        let private = engine
            .describe_prepared_command_in_transaction(40, &insert, &[])
            .unwrap();
        assert_eq!(private.parameter_types, vec![SqlType::Int4, SqlType::Text]);
        assert_eq!(
            private
                .result_columns
                .iter()
                .map(|column| (column.name.as_str(), column.ty))
                .collect::<Vec<_>>(),
            vec![("id", SqlType::Int4), ("value", SqlType::Text)]
        );
        assert!(matches!(
            engine.describe_prepared_command(&insert, &[]),
            Err(ExecuteError::UndefinedRelation(name)) if name == "private_describe"
        ));

        engine
            .submit_transaction(40, gpu_db_sql::ParsedCommand::parse("ROLLBACK").unwrap())
            .unwrap();
        assert!(matches!(
            engine.describe_prepared_command_in_transaction(40, &insert, &[]),
            Err(ExecuteError::Txn(TxnError::NotFound(40)))
        ));
        assert!(matches!(
            engine.describe_prepared_command(&insert, &[]),
            Err(ExecuteError::UndefinedRelation(name)) if name == "private_describe"
        ));
    }

    #[test]
    fn prepared_catalog_race_is_rejected_before_mutation_publication() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE guarded_write (id int4 PRIMARY KEY)")
            .unwrap();
        let template = PreparedCommand::parse("INSERT INTO guarded_write VALUES (1)").unwrap();
        let version = engine
            .describe_prepared_command(&template, &[])
            .unwrap()
            .catalog_version;

        // Simulate DDL landing after facade revalidation but before the mutation reaches the
        // publication lock. Even unrelated DDL invalidates this exact execution proof; a facade
        // retry may revalidate the unchanged command contract against the new generation.
        engine
            .execute_text(2, "CREATE TABLE catalog_bump (id int4 PRIMARY KEY)")
            .unwrap();
        let bound = template.bind(&[]).unwrap();
        let request = MutationRequest::new(bound).with_expected_catalog_version(version);
        let error = engine.submit_transaction(3, request).unwrap_err();
        assert!(matches!(error, ExecuteError::Unsupported(_)));

        let Command::Select(select) = parse_command("SELECT id FROM guarded_write").unwrap() else {
            panic!("static SELECT must parse as SELECT");
        };
        let rows = engine.execute_relational_select(&select).unwrap();
        assert!(rows.rows.is_empty());
    }
}
