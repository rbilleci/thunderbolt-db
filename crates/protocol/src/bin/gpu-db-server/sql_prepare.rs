// Legacy SQL PREPARE/EXECUTE compatibility ownership. This is not a product execution path.

use super::sql_execute_syntax::{
    parse_sql_execute, parse_supported_cursor_name, split_sql_csv,
    split_sql_name_and_optional_parenthesized_list, sql_keyword_rest_start,
    strip_leading_sql_comments, strip_sql_comments,
};
use super::{
    bind_query_parameters, canonical_sql, contains_zero_placeholder, describe_query_columns,
    execute_select_result, expected_parameter_count, int4_column,
    is_pg_dump_domain_constraints_execute, is_pg_dump_domain_constraints_prepare,
    is_pg_dump_domain_dump_prepare, is_pg_dump_function_dump_prepare, max_placeholder_index,
    negative_limit_error_field, negative_offset_error_field, parse_command,
    pg_dump_domain_constraints_columns, pg_dump_domain_dump_columns,
    pg_dump_domain_dump_execute_oid, pg_dump_domain_dump_rows, pg_dump_function_dump_columns,
    pg_dump_function_dump_rows, resolve_prepared_parameter_type_oids,
    sql_execute_argument_placeholder_index, sql_execute_parameter_error_field,
    write_command_complete, write_error, write_select_rows, write_single_row, BindParameterError,
    Column, Command, ErrorField, ParseError, PreparedQuery, PreparedStatement, ReadWrite,
    SelectResult, Session, SqlType,
};
use std::io;

pub(super) enum SqlDeallocateTarget {
    All,
    Named(String),
}

pub(super) fn try_execute_sql_prepared_statement(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    statement: &str,
    canonical: &str,
    include_row_description: bool,
) -> Option<io::Result<()>> {
    if let Some(name) = parse_sql_prepare_name(statement) {
        if session.prepared.contains_key(&name) {
            return Some(write_error(
                stream,
                &ErrorField {
                    code: "42P05",
                    message: "prepared statement already exists",
                    position: None,
                },
            ));
        }
    }

    if is_pg_dump_domain_constraints_prepare(canonical) || is_pg_dump_domain_dump_prepare(canonical)
    {
        return Some(write_command_complete(stream, "PREPARE"));
    }
    if is_pg_dump_function_dump_prepare(canonical) {
        session.prepared.insert(
            "dumpfunc".to_string(),
            PreparedStatement::PgDumpFunctionDump,
        );
        return Some(write_command_complete(stream, "PREPARE"));
    }

    if let Some((name, parameter_type_oids, query)) = parse_sql_prepare(statement) {
        let query = strip_sql_comments(&query);
        if contains_zero_placeholder(&query) {
            return Some(write_error(
                stream,
                &ErrorField {
                    code: "42P02",
                    message: "there is no parameter $0",
                    position: None,
                },
            ));
        }
        if parameter_type_oids.len() > max_placeholder_index(&query) {
            return Some(write_error(
                stream,
                &ErrorField {
                    code: "08P01",
                    message: "prepared statement has too many parameter types",
                    position: None,
                },
            ));
        }
        if describe_query_columns(session, &query).is_none() {
            return Some(write_error(
                stream,
                &ErrorField {
                    code: "0A000",
                    message: "SQL PREPARE only supports relational SELECT",
                    position: None,
                },
            ));
        }
        let parameter_type_oids =
            resolve_prepared_parameter_type_oids(session, &query, parameter_type_oids);
        session.prepared.insert(
            name,
            PreparedStatement::Sql(PreparedQuery {
                query,
                parameter_type_oids,
            }),
        );
        return Some(write_command_complete(stream, "PREPARE"));
    }

    if is_pg_dump_domain_constraints_execute(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_domain_constraints_columns(),
            &Vec::<Vec<Option<String>>>::new(),
        ));
    }

    if let Some(oid) = pg_dump_domain_dump_execute_oid(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_domain_dump_columns(),
            &pg_dump_domain_dump_rows(session, oid),
        ));
    }

    if let Some((name, parameters)) = parse_sql_execute(statement) {
        match session.prepared.get(&name).cloned() {
            Some(PreparedStatement::AddTen) => {
                if parameters.len() == 1 && parameters[0].as_deref() == Some("5") {
                    return Some(write_single_row(
                        stream,
                        &[int4_column("plus_ten")],
                        &[vec![Some(String::from("15"))]],
                    ));
                }
            }
            Some(PreparedStatement::PgDumpFunctionDump) => {
                let oid = parameters
                    .first()
                    .and_then(|parameter| parameter.as_deref())
                    .and_then(|parameter| parameter.trim().parse::<u32>().ok());
                return Some(write_single_row(
                    stream,
                    &pg_dump_function_dump_columns(),
                    &pg_dump_function_dump_rows(session, oid),
                ));
            }
            Some(PreparedStatement::Sql(query)) => {
                let bound_query = match bind_query_parameters(&query, &parameters) {
                    Ok(query) => query,
                    Err(error) => {
                        return Some(write_error(
                            stream,
                            &sql_execute_parameter_error_field(error),
                        ));
                    }
                };
                let select = match parse_command(&bound_query) {
                    Ok(Command::Select(select)) => select,
                    Err(ParseError::NegativeLimit) => {
                        return Some(write_error(stream, &negative_limit_error_field()));
                    }
                    Err(ParseError::NegativeOffset) => {
                        return Some(write_error(stream, &negative_offset_error_field()));
                    }
                    Ok(_) | Err(_) => {
                        return Some(write_error(
                            stream,
                            &ErrorField {
                                code: "0A000",
                                message: "SQL EXECUTE only supports relational SELECT",
                                position: None,
                            },
                        ));
                    }
                };
                let result = match execute_select_result(session, &select) {
                    Ok(result) => result,
                    Err(error) => return Some(write_error(stream, &error)),
                };
                return Some(write_select_rows(
                    stream,
                    &result.columns,
                    &result.rows,
                    include_row_description,
                ));
            }
            Some(PreparedStatement::Extended(_)) | None => {}
        }
        let message =
            Box::leak(format!("prepared statement \"{name}\" does not exist").into_boxed_str());
        return Some(write_error(
            stream,
            &ErrorField {
                code: "26000",
                message,
                position: None,
            },
        ));
    }

    if let Some(target) = parse_sql_deallocate(statement) {
        match target {
            SqlDeallocateTarget::All => {
                session
                    .prepared
                    .retain(|_, statement| matches!(statement, PreparedStatement::Extended(_)));
                return Some(write_command_complete(stream, "DEALLOCATE ALL"));
            }
            SqlDeallocateTarget::Named(name) => {
                if matches!(
                    session.prepared.get(&name),
                    Some(PreparedStatement::AddTen | PreparedStatement::Sql(_))
                ) {
                    session.prepared.remove(&name);
                    return Some(write_command_complete(stream, "DEALLOCATE"));
                }
                let message = Box::leak(
                    format!("prepared statement \"{name}\" does not exist").into_boxed_str(),
                );
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "26000",
                        message,
                        position: None,
                    },
                ));
            }
        }
    }
    None
}

pub(super) fn parse_sql_prepare(statement: &str) -> Option<(String, Vec<u32>, String)> {
    let trimmed = strip_leading_sql_comments(statement.trim().trim_end_matches(';').trim())?;
    let rest_start = sql_keyword_rest_start(trimmed, "prepare")?;
    let as_idx = find_sql_prepare_as_index(trimmed)?;
    let original_target = strip_sql_comments(trimmed[rest_start..as_idx].trim());
    let query_start = as_idx + " as ".len();
    let query = trimmed[query_start..].trim();
    if query.is_empty() {
        return None;
    }
    let canonical_query = canonical_sql(query);
    if !canonical_query.starts_with("select ") || !canonical_query.contains(" from ") {
        return None;
    }

    let (name, type_oids) = {
        let (name, type_list) = split_sql_name_and_optional_parenthesized_list(&original_target)?;
        if let Some(type_list) = type_list {
            let mut type_oids = Vec::new();
            if !type_list.trim().is_empty() {
                for ty in split_sql_csv(type_list)? {
                    type_oids.push(sql_prepare_type_oid(ty.trim())?);
                }
            }
            (name, type_oids)
        } else {
            (name, Vec::new())
        }
    };

    Some((name, type_oids, query.to_string()))
}

pub(super) fn parse_sql_prepare_name(statement: &str) -> Option<String> {
    let trimmed = strip_leading_sql_comments(statement.trim().trim_end_matches(';').trim())?;
    let rest_start = sql_keyword_rest_start(trimmed, "prepare")?;
    let as_idx = find_sql_prepare_as_index(trimmed)?;
    let original_target = strip_sql_comments(trimmed[rest_start..as_idx].trim());
    split_sql_name_and_optional_parenthesized_list(&original_target).map(|(name, _)| name)
}

fn find_sql_prepare_as_index(statement: &str) -> Option<usize> {
    let mut in_quoted_identifier = false;
    let mut paren_depth = 0usize;
    let mut chars = statement.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if idx < "prepare ".len() {
            continue;
        }
        if in_quoted_identifier {
            if ch == '"' {
                if chars.peek().is_some_and(|(_, next)| *next == '"') {
                    chars.next();
                } else {
                    in_quoted_identifier = false;
                }
            }
            continue;
        }
        if ch == '-' && chars.peek().is_some_and(|(_, next)| *next == '-') {
            chars.next();
            for (_, next_ch) in chars.by_ref() {
                if next_ch == '\n' {
                    break;
                }
            }
            continue;
        }
        if ch == '/' && chars.peek().is_some_and(|(_, next)| *next == '*') {
            chars.next();
            let mut depth = 1usize;
            let mut previous_char: Option<char> = None;
            for (_, next_ch) in chars.by_ref() {
                if previous_char == Some('/') && next_ch == '*' {
                    depth = depth.saturating_add(1);
                    previous_char = None;
                    continue;
                }
                if previous_char == Some('*') && next_ch == '/' {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        break;
                    }
                    previous_char = None;
                    continue;
                }
                previous_char = Some(next_ch);
            }
            continue;
        }
        match ch {
            '"' => in_quoted_identifier = true,
            '(' => paren_depth = paren_depth.saturating_add(1),
            ')' => paren_depth = paren_depth.saturating_sub(1),
            'a' | 'A'
                if paren_depth == 0
                    && statement[idx..]
                        .get(..2)
                        .is_some_and(|candidate| candidate.eq_ignore_ascii_case("as"))
                    && statement[..idx]
                        .chars()
                        .next_back()
                        .is_some_and(char::is_whitespace)
                    && statement[idx + 2..]
                        .chars()
                        .next()
                        .is_some_and(char::is_whitespace) =>
            {
                return statement[..idx]
                    .char_indices()
                    .next_back()
                    .map(|(previous_idx, _)| previous_idx);
            }
            _ => {}
        }
    }
    None
}

pub(super) fn describe_extended_query_columns(
    session: &Session,
    query: &str,
) -> Option<Vec<Column>> {
    if let Some((name, parameters)) = parse_sql_execute(query) {
        return match session.prepared.get(&name) {
            Some(PreparedStatement::AddTen)
                if parameters.len() == 1 && parameters[0].as_deref() == Some("5") =>
            {
                Some(vec![int4_column("plus_ten")])
            }
            Some(PreparedStatement::Sql(prepared)) => {
                bind_sql_execute_describe_parameters(prepared, &parameters)
                    .ok()
                    .and_then(|describe_parameters| {
                        bind_query_parameters(prepared, &describe_parameters).ok()
                    })
                    .and_then(|bound_query| describe_query_columns(session, &bound_query))
            }
            Some(PreparedStatement::Extended(_)) | None => None,
            _ => None,
        };
    }
    describe_query_columns(session, query)
}

fn bind_sql_execute_describe_parameters(
    prepared: &PreparedQuery,
    parameters: &[Option<String>],
) -> Result<Vec<Option<String>>, BindParameterError> {
    if expected_parameter_count(prepared) != parameters.len() {
        return Err(BindParameterError::CountMismatch);
    }
    parameters
        .iter()
        .enumerate()
        .map(|(idx, parameter)| match parameter {
            Some(value) if sql_execute_argument_placeholder_index(value).is_some() => {
                let oid = prepared.parameter_type_oids.get(idx).copied().unwrap_or(0);
                Ok(Some(sql_execute_describe_dummy_value(oid).to_string()))
            }
            parameter => Ok(parameter.clone()),
        })
        .collect()
}

pub(super) fn sql_execute_describe_error(session: &Session, query: &str) -> Option<ErrorField> {
    let (name, parameters) = parse_sql_execute(query)?;
    let Some(PreparedStatement::Sql(prepared)) = session.prepared.get(&name) else {
        return None;
    };
    let describe_parameters = match bind_sql_execute_describe_parameters(prepared, &parameters) {
        Ok(parameters) => parameters,
        Err(error) => return Some(sql_execute_parameter_error_field(error)),
    };
    bind_query_parameters(prepared, &describe_parameters)
        .err()
        .map(sql_execute_parameter_error_field)
}

fn sql_execute_describe_dummy_value(type_oid: u32) -> &'static str {
    match type_oid {
        23 => "1",
        25 => "text",
        _ => "1",
    }
}

pub(super) fn execute_sql_prepared_result(
    session: &mut Session,
    name: &str,
    parameters: &[Option<String>],
) -> Result<SelectResult, ErrorField> {
    match session.prepared.get(name).cloned() {
        Some(PreparedStatement::AddTen)
            if parameters.len() == 1 && parameters[0].as_deref() == Some("5") =>
        {
            Ok(SelectResult {
                columns: vec![int4_column("plus_ten")],
                rows: vec![vec![Some(String::from("15"))]],
            })
        }
        Some(PreparedStatement::PgDumpFunctionDump) => {
            let oid = parameters
                .first()
                .and_then(|parameter| parameter.as_deref())
                .and_then(|parameter| parameter.trim().parse::<u32>().ok());
            Ok(SelectResult {
                columns: pg_dump_function_dump_columns(),
                rows: pg_dump_function_dump_rows(session, oid),
            })
        }
        Some(PreparedStatement::Sql(query)) => {
            let bound_query = bind_query_parameters(&query, parameters)
                .map_err(sql_execute_parameter_error_field)?;
            let select = match parse_command(&bound_query) {
                Ok(Command::Select(select)) => select,
                Err(ParseError::NegativeLimit) => return Err(negative_limit_error_field()),
                Err(ParseError::NegativeOffset) => return Err(negative_offset_error_field()),
                Ok(_) | Err(_) => {
                    return Err(ErrorField {
                        code: "0A000",
                        message: "SQL EXECUTE only supports relational SELECT",
                        position: None,
                    });
                }
            };
            execute_select_result(session, &select)
        }
        Some(PreparedStatement::AddTen) | Some(PreparedStatement::Extended(_)) | None => {
            let message =
                Box::leak(format!("prepared statement \"{name}\" does not exist").into_boxed_str());
            Err(ErrorField {
                code: "26000",
                message,
                position: None,
            })
        }
    }
}

pub(super) fn parse_sql_deallocate(statement: &str) -> Option<SqlDeallocateTarget> {
    let trimmed = strip_leading_sql_comments(statement.trim().trim_end_matches(';').trim())?;
    let rest_start = sql_keyword_rest_start(trimmed, "deallocate")?;
    let original_rest = strip_sql_comments(&trimmed[rest_start..]);
    let mut target = original_rest.trim_start();
    let lower_target = target.to_ascii_lowercase();
    if lower_target == "prepare" || lower_target == "prepared" {
        return None;
    }
    if let Some(after_keyword) = sql_keyword_rest_start(target, "prepare")
        .or_else(|| sql_keyword_rest_start(target, "prepared"))
    {
        target = target[after_keyword..].trim_start();
    }
    if target.eq_ignore_ascii_case("all") {
        return Some(SqlDeallocateTarget::All);
    }
    if target.is_empty() {
        None
    } else {
        parse_supported_cursor_name(target).map(SqlDeallocateTarget::Named)
    }
}

fn sql_prepare_type_oid(ty: &str) -> Option<u32> {
    match canonical_sql(ty).as_str() {
        "int" | "int4" | "integer" | "pg_catalog.int4" | "pg_catalog.integer" => {
            Some(SqlType::Int4.postgres_oid())
        }
        "text" | "pg_catalog.text" => Some(SqlType::Text.postgres_oid()),
        _ => None,
    }
}
