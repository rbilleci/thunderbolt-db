//! Typed prepared-command ownership and direct AST parameter binding.

use std::sync::Arc;

use crate::{
    canonicalize_sql_for_exact_match, command::parse_prepared_command_allowing_catalog,
    lower_sql_parameters, parameter::sql_parameter_arity, Command, ParseError, ParsedCommand,
    PreparedCatalogProgram, SelectFilter, SqlType, SqlValue,
};

/// One parsed SQL template whose `$n` references are typed AST slots rather than reconstructed SQL.
///
/// `bind` clones the already-parsed command and replaces those slots directly. It renders a
/// canonical, parseable request identity for the current transitional SQL-text WAL records, but
/// that text is never reparsed to obtain the executable command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCommand {
    source: Arc<str>,
    command: Command,
    parameter_count: usize,
}

impl PreparedCommand {
    pub fn parse(source: &str) -> Result<Self, ParseError> {
        let raw_parameter_count = sql_parameter_arity(source)?;
        if raw_parameter_count > 0 && !supports_parameterized_command(source) {
            return Err(ParseError::Unsupported(
                "parameters are supported only in prepared SELECT, INSERT, UPDATE, and DELETE"
                    .to_string(),
            ));
        }
        let command = parse_pg16_dump_catalog_program(source)
            .map_or_else(|| parse_prepared_command_allowing_catalog(source), Ok)?;
        let parameter_count = command_parameter_count(&command);
        if parameter_count != raw_parameter_count
            || (parameter_count > 0
                && !matches!(
                    &command,
                    Command::Select(_)
                        | Command::SelectLiteral(_)
                        | Command::Insert(_)
                        | Command::Update(_)
                        | Command::Delete(_)
                        | Command::PreparedCatalog(_)
                ))
        {
            return Err(ParseError::Unsupported(
                "parameter reference is outside a supported prepared SQL value position"
                    .to_string(),
            ));
        }
        Ok(Self {
            source: Arc::from(source),
            command,
            parameter_count,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn command(&self) -> &Command {
        &self.command
    }

    /// Exact PostgreSQL parameter arity: the highest referenced `$n` (repeated and out-of-order
    /// references are legal and gaps still contribute to the Bind arity).
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Bind neutral typed values directly into a cloned AST. No SQL generated here is parsed.
    pub fn bind(&self, parameters: &[SqlValue]) -> Result<ParsedCommand, ParseError> {
        if parameters.len() != self.parameter_count {
            return Err(ParseError::InvalidParameterCount {
                expected: self.parameter_count,
                actual: parameters.len(),
            });
        }
        if parameters
            .iter()
            .any(|value| matches!(value, SqlValue::Parameter { .. }))
        {
            return Err(ParseError::InvalidParameterReference);
        }
        if self.parameter_count == 0 {
            return Ok(ParsedCommand::from_bound_prepared(
                Arc::clone(&self.source),
                self.command.clone(),
            ));
        }

        let mut command = self.command.clone();
        visit_command_values_mut(&mut command, &mut |value| {
            let SqlValue::Parameter { index, cast } = value else {
                return Ok(());
            };
            let supplied = parameters.get(index.saturating_sub(1)).ok_or(
                ParseError::InvalidParameterCount {
                    expected: *index,
                    actual: parameters.len(),
                },
            )?;
            *value = coerce_explicit_cast(supplied, *cast)?;
            Ok(())
        })?;
        bind_select_limit_parameter(&mut command, parameters)?;
        debug_assert_eq!(command_parameter_count(&command), 0);

        // Transitional durability identity only. The executable command above is the directly
        // bound AST; this canonical text is not parsed on the live prepared path.
        let canonical_source = lower_sql_parameters(&self.source, parameters)?;
        Ok(ParsedCommand::from_bound_prepared(
            Arc::from(canonical_source),
            command,
        ))
    }
}

fn supports_parameterized_command(source: &str) -> bool {
    let keyword = source
        .trim_start()
        .split_once(char::is_whitespace)
        .map_or_else(|| source.trim(), |(keyword, _)| keyword);
    ["SELECT", "INSERT", "UPDATE", "DELETE"]
        .iter()
        .any(|allowed| keyword.eq_ignore_ascii_case(allowed))
}

fn parse_pg16_dump_catalog_program(source: &str) -> Option<Command> {
    let canonical = canonicalize_sql_for_exact_match(source).ok()?;
    let parameter = || SqlValue::Parameter {
        index: 1,
        cast: Some(SqlType::Int4),
    };
    if canonical
        == "select tableoid, oid, conname, pg_catalog.pg_get_constraintdef(oid) as consrc, \
            convalidated from pg_catalog.pg_constraint where contypid = $1 order by conname"
    {
        return Some(Command::PreparedCatalog(
            PreparedCatalogProgram::Pg16DomainConstraints {
                type_oid: parameter(),
            },
        ));
    }
    if canonical
        == "select t.typnotnull, pg_catalog.format_type(t.typbasetype, t.typtypmod) as typdefn, \
            pg_catalog.pg_get_expr(t.typdefaultbin, 'pg_catalog.pg_type'::pg_catalog.regclass) as \
            typdefaultbin, t.typdefault, case when t.typcollation <> u.typcollation then \
            t.typcollation else 0 end as typcollation from pg_catalog.pg_type t left join \
            pg_catalog.pg_type u on (t.typbasetype = u.oid) where t.oid = $1"
    {
        return Some(Command::PreparedCatalog(
            PreparedCatalogProgram::Pg16DomainDefinition {
                type_oid: parameter(),
            },
        ));
    }
    if canonical
        == "select proretset, prosrc, probin, provolatile, proisstrict, prosecdef, lanname, \
            proconfig, procost, prorows, pg_catalog.pg_get_function_arguments(p.oid) as funcargs, \
            pg_catalog.pg_get_function_identity_arguments(p.oid) as funciargs, \
            pg_catalog.pg_get_function_result(p.oid) as funcresult, proleakproof, \
            array_to_string(protrftypes, ' ') as protrftypes, proparallel, prokind, prosupport, \
            pg_get_function_sqlbody(p.oid) as prosqlbody from pg_catalog.pg_proc p, \
            pg_catalog.pg_language l where p.oid = $1 and l.oid = p.prolang"
    {
        return Some(Command::PreparedCatalog(
            PreparedCatalogProgram::Pg16FunctionDefinition {
                function_oid: parameter(),
            },
        ));
    }
    None
}

pub(crate) fn command_parameter_count(command: &Command) -> usize {
    let mut highest = 0;
    visit_command_values(command, &mut |value| {
        if let SqlValue::Parameter { index, .. } = value {
            highest = highest.max(*index);
        }
    });
    if let Command::Select(select) = command {
        highest = highest.max(select.prepared_limit_parameter_index().unwrap_or_default());
    }
    highest
}

fn bind_select_limit_parameter(
    command: &mut Command,
    parameters: &[SqlValue],
) -> Result<(), ParseError> {
    let Command::Select(select) = command else {
        return Ok(());
    };
    let Some(index) = select.prepared_limit_parameter_index() else {
        return Ok(());
    };
    let value =
        parameters
            .get(index.saturating_sub(1))
            .ok_or(ParseError::InvalidParameterCount {
                expected: index,
                actual: parameters.len(),
            })?;
    select.limit = match value {
        SqlValue::Null => None,
        SqlValue::Int4(value) if *value >= 0 => Some(*value as usize),
        SqlValue::Int4(_) => return Err(ParseError::NegativeLimit),
        _ => return Err(ParseError::InvalidRelationalSql),
    };
    Ok(())
}

fn visit_command_values(command: &Command, visit: &mut impl FnMut(&SqlValue)) {
    match command {
        Command::Select(select) => {
            visit_filters(
                select.filter.as_ref(),
                &select.filters,
                &select.filter_groups,
                visit,
            );
            for group in &select.having_groups {
                for filter in group {
                    visit(&filter.value);
                }
            }
        }
        Command::SelectLiteral(literal) => visit(&literal.value),
        Command::Insert(insert) => {
            for row in &insert.rows {
                for value in row {
                    visit(value);
                }
            }
        }
        Command::Update(update) => {
            for assignment in &update.assignments {
                visit(&assignment.value);
            }
            visit_filters(
                update.filter.as_ref(),
                &update.filters,
                &update.filter_groups,
                visit,
            );
        }
        Command::Delete(delete) => visit_filters(
            delete.filter.as_ref(),
            &delete.filters,
            &delete.filter_groups,
            visit,
        ),
        Command::CreateTable(create) => {
            for column in &create.columns {
                visit_default(column.default.as_ref(), visit);
            }
            for constraint in &create.check_constraints {
                visit(&constraint.filter.value);
            }
        }
        Command::AddCheckConstraint(constraint) => visit(&constraint.filter.value),
        Command::AddColumn(add) => visit_default(add.column.default.as_ref(), visit),
        Command::AlterColumnDefault(alter) => visit_default(alter.default.as_ref(), visit),
        Command::PreparedCatalog(program) => match program {
            PreparedCatalogProgram::Pg16DomainConstraints { type_oid }
            | PreparedCatalogProgram::Pg16DomainDefinition { type_oid } => visit(type_oid),
            PreparedCatalogProgram::Pg16FunctionDefinition { function_oid } => visit(function_oid),
            PreparedCatalogProgram::Pg16MaterializedViewDependencies => {}
        },
        _ => {}
    }
}

fn visit_default(default: Option<&crate::ColumnDefault>, visit: &mut impl FnMut(&SqlValue)) {
    if let Some(crate::ColumnDefault::Literal(value)) = default {
        visit(value);
    }
}

fn visit_filters(
    filter: Option<&SelectFilter>,
    filters: &[SelectFilter],
    filter_groups: &[Vec<SelectFilter>],
    visit: &mut impl FnMut(&SqlValue),
) {
    if let Some(filter) = filter {
        visit(&filter.value);
    }
    for filter in filters {
        visit(&filter.value);
    }
    for group in filter_groups {
        for filter in group {
            visit(&filter.value);
        }
    }
}

fn visit_command_values_mut(
    command: &mut Command,
    visit: &mut impl FnMut(&mut SqlValue) -> Result<(), ParseError>,
) -> Result<(), ParseError> {
    match command {
        Command::Select(select) => {
            visit_filters_mut(
                select.filter.as_mut(),
                &mut select.filters,
                &mut select.filter_groups,
                visit,
            )?;
            for group in &mut select.having_groups {
                for filter in group {
                    visit(&mut filter.value)?;
                }
            }
        }
        Command::SelectLiteral(literal) => visit(&mut literal.value)?,
        Command::Insert(insert) => {
            for row in &mut insert.rows {
                for value in row {
                    visit(value)?;
                }
            }
        }
        Command::Update(update) => {
            for assignment in &mut update.assignments {
                visit(&mut assignment.value)?;
            }
            visit_filters_mut(
                update.filter.as_mut(),
                &mut update.filters,
                &mut update.filter_groups,
                visit,
            )?;
        }
        Command::Delete(delete) => visit_filters_mut(
            delete.filter.as_mut(),
            &mut delete.filters,
            &mut delete.filter_groups,
            visit,
        )?,
        Command::PreparedCatalog(program) => match program {
            PreparedCatalogProgram::Pg16DomainConstraints { type_oid }
            | PreparedCatalogProgram::Pg16DomainDefinition { type_oid } => visit(type_oid)?,
            PreparedCatalogProgram::Pg16FunctionDefinition { function_oid } => visit(function_oid)?,
            PreparedCatalogProgram::Pg16MaterializedViewDependencies => {}
        },
        _ => {}
    }
    Ok(())
}

fn visit_filters_mut(
    filter: Option<&mut SelectFilter>,
    filters: &mut [SelectFilter],
    filter_groups: &mut [Vec<SelectFilter>],
    visit: &mut impl FnMut(&mut SqlValue) -> Result<(), ParseError>,
) -> Result<(), ParseError> {
    if let Some(filter) = filter {
        visit(&mut filter.value)?;
    }
    for filter in filters {
        visit(&mut filter.value)?;
    }
    for group in filter_groups {
        for filter in group {
            visit(&mut filter.value)?;
        }
    }
    Ok(())
}

fn coerce_explicit_cast(value: &SqlValue, cast: Option<SqlType>) -> Result<SqlValue, ParseError> {
    let Some(cast) = cast else {
        return Ok(value.clone());
    };
    if matches!(value, SqlValue::Null) {
        return Ok(SqlValue::Null);
    }
    if let (SqlType::Numeric { precision, scale }, SqlValue::Numeric(value)) = (cast, value) {
        let rescaled = value
            .rescale(scale)
            .map_err(|_| ParseError::InvalidRelationalSql)?;
        if numeric_exceeds_precision(rescaled.mantissa, precision) {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(SqlValue::Numeric(rescaled));
    }
    let compatible = matches!(
        (cast, value),
        (SqlType::Int2, SqlValue::Int2(_))
            | (SqlType::Int4, SqlValue::Int4(_))
            | (SqlType::Int8, SqlValue::Int8(_))
            | (SqlType::Numeric { .. }, SqlValue::Numeric(_))
            | (SqlType::Bool, SqlValue::Bool(_))
            | (SqlType::Text, SqlValue::Text(_))
            | (SqlType::Date, SqlValue::Date(_))
            | (SqlType::Timestamp, SqlValue::Timestamp(_))
            | (SqlType::Uuid, SqlValue::Uuid(_))
    );
    if compatible {
        Ok(value.clone())
    } else {
        Err(ParseError::InvalidRelationalSql)
    }
}

fn numeric_exceeds_precision(mantissa: i128, precision: u8) -> bool {
    let mut bound = 1_i128;
    for _ in 0..precision {
        let Some(next) = bound.checked_mul(10) else {
            return false;
        };
        bound = next;
    }
    mantissa.unsigned_abs() >= bound.unsigned_abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_replaces_repeated_out_of_order_slots_without_reparsing() {
        let prepared = PreparedCommand::parse(
            "SELECT id FROM accounts WHERE tenant_id = $2 AND account_id = $1 AND id = $2",
        )
        .unwrap();
        assert_eq!(prepared.parameter_count(), 2);
        let bound = prepared
            .bind(&[SqlValue::Int8(90), SqlValue::Int4(7)])
            .unwrap();
        let Command::Select(select) = bound.command() else {
            panic!("expected SELECT");
        };
        assert_eq!(select.filters[0].value, SqlValue::Int4(7));
        assert_eq!(select.filters[1].value, SqlValue::Int8(90));
        assert_eq!(select.filters[2].value, SqlValue::Int4(7));
    }

    #[test]
    fn bind_resolves_parameterized_limit_before_execution() {
        let prepared = PreparedCommand::parse(
            "SELECT id FROM accounts WHERE id >= $1 ORDER BY id LIMIT $002::pg_catalog.int4",
        )
        .unwrap();
        assert_eq!(prepared.parameter_count(), 2);
        let bound = prepared
            .bind(&[SqlValue::Int4(7), SqlValue::Int4(3)])
            .unwrap();
        let Command::Select(select) = bound.command() else {
            panic!("expected SELECT");
        };
        assert_eq!(select.limit, Some(3));
        assert!(matches!(
            prepared.bind(&[SqlValue::Int4(7), SqlValue::Int4(-1)]),
            Err(ParseError::NegativeLimit)
        ));
    }

    #[test]
    fn bind_preserves_prepared_int4_scalar_addition_for_gpu_literal_execution() {
        let prepared = PreparedCommand::parse("SELECT $1 + 10 AS plus_ten").unwrap();
        assert_eq!(prepared.parameter_count(), 1);
        let Command::SelectLiteral(literal) = prepared.command() else {
            panic!("expected bounded scalar projection");
        };
        assert_eq!(literal.ty, SqlType::Int4);
        assert_eq!(literal.add_int4, Some(10));

        let bound = prepared.bind(&[SqlValue::Int4(5)]).unwrap();
        let Command::SelectLiteral(literal) = bound.command() else {
            panic!("expected bound scalar projection");
        };
        assert_eq!(literal.value, SqlValue::Int4(5));
        assert_eq!(literal.add_int4, Some(10));
    }

    #[test]
    fn prepared_int4_addition_binds_edge_operands_without_host_range_folding() {
        for (sql, value, addend) in [
            ("SELECT $1 + 0 AS unchanged", SqlValue::Int4(i32::MAX), 0),
            ("SELECT $1 + -1 AS minus_one", SqlValue::Int4(i32::MIN), -1),
            ("SELECT $1 + 1 AS plus_one", SqlValue::Int4(i32::MAX), 1),
            ("SELECT $1 + 1 AS plus_one", SqlValue::Null, 1),
        ] {
            let prepared = PreparedCommand::parse(sql).unwrap();
            let bound = prepared.bind(std::slice::from_ref(&value)).unwrap();
            let Command::SelectLiteral(literal) = bound.command() else {
                panic!("expected bound scalar arithmetic literal");
            };
            assert_eq!(literal.value, value);
            assert_eq!(literal.add_int4, Some(addend));
        }
    }

    #[test]
    fn ordinary_parse_rejects_unbound_parameter_slots() {
        assert!(matches!(
            ParsedCommand::parse("DELETE FROM accounts WHERE id = $1"),
            Err(ParseError::InvalidParameterReference)
        ));
    }

    #[test]
    fn explicit_parameter_cast_is_preserved_and_checked_at_bind() {
        let prepared = PreparedCommand::parse("DELETE FROM accounts WHERE id = $1::int8").unwrap();
        let bound = prepared.bind(&[SqlValue::Int8(9)]).unwrap();
        assert_eq!(
            ParsedCommand::parse(bound.source()).unwrap().command(),
            bound.command(),
            "canonical durability source must replay the directly bound AST"
        );
        assert!(matches!(
            prepared.bind(&[SqlValue::Int4(9)]),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn null_explicit_cast_has_parseable_canonical_source() {
        let prepared = PreparedCommand::parse("DELETE FROM accounts WHERE id = $1::int8").unwrap();
        let bound = prepared.bind(&[SqlValue::Null]).unwrap();
        assert_eq!(bound.source(), "DELETE FROM accounts WHERE id = NULL::int8");
        assert_eq!(
            ParsedCommand::parse(bound.source()).unwrap().command(),
            bound.command()
        );
    }

    #[test]
    fn bind_rejects_nested_parameter_values_without_panicking() {
        let prepared = PreparedCommand::parse("DELETE FROM accounts WHERE id = $1").unwrap();
        assert!(matches!(
            prepared.bind(&[SqlValue::Parameter {
                index: 2,
                cast: None,
            }]),
            Err(ParseError::InvalidParameterReference)
        ));
    }

    #[test]
    fn zero_arity_bind_reuses_the_owned_command() {
        let prepared = PreparedCommand::parse("DELETE FROM accounts WHERE id = 7").unwrap();
        let bound = prepared.bind(&[]).unwrap();
        assert_eq!(bound.command(), prepared.command());
        assert_eq!(bound.source(), prepared.source());
    }

    #[test]
    fn product_prepared_parse_retains_catalog_mixed_star_and_in_filter() {
        let source = "SELECT oid, * FROM pg_catalog.pg_type \
                      WHERE typname IN ('hstore','geometry','vector')";
        assert!(ParsedCommand::parse(source).is_err());
        let prepared = PreparedCommand::parse(source).unwrap();
        assert_eq!(prepared.parameter_count(), 0);
        let Command::Select(select) = prepared.command() else {
            panic!("expected SELECT");
        };
        assert_eq!(select.table, "pg_catalog.pg_type");
        assert!(!select.public_only);
        assert_eq!(
            select.projection,
            crate::SelectProjection::Columns(vec![
                "oid".to_string(),
                crate::PROJECTION_WILDCARD_SENTINEL.to_string(),
            ])
        );
        assert_eq!(select.filter_groups.len(), 3);
        let bound = prepared.bind(&[]).unwrap();
        assert_eq!(bound.command(), prepared.command());
        assert_eq!(bound.source(), source);

        let quoted = PreparedCommand::parse(r#"SELECT oid, "*" FROM pg_catalog.pg_type"#).unwrap();
        let Command::Select(quoted) = quoted.command() else {
            panic!("expected quoted-star SELECT");
        };
        assert_eq!(
            quoted.projection,
            crate::SelectProjection::Columns(vec!["oid".to_string(), "*".to_string()])
        );

        let explicit_public =
            PreparedCommand::parse("SELECT oid FROM public.pg_type ORDER BY oid").unwrap();
        let Command::Select(explicit_public) = explicit_public.command() else {
            panic!("expected SELECT");
        };
        assert_eq!(explicit_public.table, "pg_type");
        assert!(explicit_public.public_only);
    }

    #[test]
    fn raw_parameters_outside_supported_ast_slots_fail_pre_effect() {
        for sql in [
            "SET x = $1",
            "GET $1",
            "DEL $1",
            "CREATE TABLE t (id int DEFAULT $1)",
        ] {
            assert!(matches!(
                crate::parse_command(sql),
                Err(ParseError::InvalidParameterReference)
            ));
            assert!(PreparedCommand::parse(sql).is_err());
        }
    }

    #[test]
    fn pg16_function_definition_program_is_exact_and_near_misses_fail_closed() {
        let canonical = "SELECT proretset, prosrc, probin, provolatile, proisstrict, prosecdef, \
            lanname, proconfig, procost, prorows, \
            pg_catalog.pg_get_function_arguments(p.oid) AS funcargs, \
            pg_catalog.pg_get_function_identity_arguments(p.oid) AS funciargs, \
            pg_catalog.pg_get_function_result(p.oid) AS funcresult, proleakproof, \
            array_to_string(protrftypes, ' ') AS protrftypes, proparallel, prokind, prosupport, \
            pg_get_function_sqlbody(p.oid) AS prosqlbody \
            FROM pg_catalog.pg_proc p, pg_catalog.pg_language l \
            WHERE p.oid = $1 AND l.oid = p.prolang";
        assert!(matches!(
            PreparedCommand::parse(canonical).unwrap().command(),
            Command::PreparedCatalog(PreparedCatalogProgram::Pg16FunctionDefinition { .. })
        ));

        for near_miss in [
            canonical.replace("prosupport,", "prosupport, p.oid,"),
            canonical.replace("p.oid = $1", "p.oid <> $1"),
            canonical.replace("AND l.oid = p.prolang", "OR l.oid = p.prolang"),
            canonical.replace("FROM pg_catalog.pg_proc", "FROM (pg_catalog.pg_proc"),
            canonical.replace("protrftypes, ' '", "protrftypes, '  '"),
        ] {
            assert!(
                PreparedCommand::parse(&near_miss).is_err(),
                "near-miss pg_dump function program must not enter the fixed-result route: {near_miss}"
            );
        }
    }

    #[test]
    fn numeric_explicit_cast_applies_scale_and_precision() {
        let prepared =
            PreparedCommand::parse("DELETE FROM accounts WHERE balance = $1::numeric(5,2)")
                .unwrap();
        let bound = prepared
            .bind(&[SqlValue::Numeric(crate::Decimal128::new(1005, 3))])
            .unwrap();
        let Command::Delete(delete) = bound.command() else {
            panic!("expected DELETE");
        };
        assert_eq!(
            delete.filter.as_ref().unwrap().value,
            SqlValue::Numeric(crate::Decimal128::new(101, 2))
        );
        assert!(matches!(
            prepared.bind(&[SqlValue::Numeric(crate::Decimal128::new(100_000, 2))]),
            Err(ParseError::InvalidRelationalSql)
        ));
    }
}
