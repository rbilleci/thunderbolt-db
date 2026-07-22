//! SQL-level PREPARE/EXECUTE session syntax over the canonical prepared AST owner.
//!
//! This layer owns only connection-local names and SQL literal decoding. PREPARE delegates parse
//! and catalog description to the facade; EXECUTE binds the retained AST and submits it through
//! the same `SharedEngine::submit` boundary as extended-query protocol portals.

use gpu_db_facade::{
    BoundPreparedStatement, DbError, DbValue, ErrorCategory, LogicalType, PreparedStatement,
};

const MAX_SQL_PREPARE_PARAMETERS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlPreparedAction {
    Prepare {
        name: String,
        query: String,
        parameter_hints: Vec<Option<LogicalType>>,
    },
    Execute {
        name: String,
        arguments: Vec<Option<String>>,
    },
    Deallocate(SqlDeallocateTarget),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlDeallocateTarget {
    All,
    Named(String),
}

pub(crate) fn classify_sql_prepared_statement(
    statement: &str,
) -> Result<Option<SqlPreparedAction>, DbError> {
    let statement = statement.trim();
    if let Some(rest) = strip_keyword(statement, "PREPARE") {
        return parse_prepare(rest).map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "EXECUTE") {
        return parse_execute(rest).map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "DEALLOCATE") {
        return parse_deallocate(rest).map(|target| Some(SqlPreparedAction::Deallocate(target)));
    }
    Ok(None)
}

pub(crate) fn bind_sql_execute(
    statement: &PreparedStatement,
    arguments: &[Option<String>],
) -> Result<BoundPreparedStatement, DbError> {
    let parameter_types = statement.parameter_types().ok_or_else(|| {
        internal_error("SQL prepared statement lost its catalog-resolved parameter types")
    })?;
    if parameter_types.len() != arguments.len() {
        return Err(invalid_request(format!(
            "prepared statement expects {} parameters, got {}",
            parameter_types.len(),
            arguments.len()
        )));
    }
    let values = parameter_types
        .iter()
        .copied()
        .zip(arguments)
        .map(|(ty, value)| decode_sql_execute_argument(ty, value.as_deref()))
        .collect::<Result<Vec<_>, _>>()?;
    statement.bind_values(&values)
}

fn parse_prepare(rest: &str) -> Result<SqlPreparedAction, DbError> {
    let open = rest
        .find('(')
        .ok_or_else(|| syntax_error("PREPARE requires a type list"))?;
    let close = find_matching_parenthesis(rest, open)
        .ok_or_else(|| syntax_error("PREPARE has an unterminated type list"))?;
    let name = normalize_name(&rest[..open])?;
    let after = rest[close + 1..].trim_start();
    let query = strip_keyword(after, "AS")
        .ok_or_else(|| syntax_error("PREPARE requires AS and a query"))?
        .trim();
    if query.is_empty() {
        return Err(syntax_error("PREPARE query is empty"));
    }
    let types = split_csv(&rest[open + 1..close])?;
    if types.len() > MAX_SQL_PREPARE_PARAMETERS {
        return Err(invalid_request(format!(
            "SQL PREPARE supports at most {MAX_SQL_PREPARE_PARAMETERS} parameters"
        )));
    }
    let parameter_hints = types
        .into_iter()
        .map(|ty| parse_parameter_type(ty).map(Some))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SqlPreparedAction::Prepare {
        name,
        query: query.to_string(),
        parameter_hints,
    })
}

fn parse_execute(rest: &str) -> Result<SqlPreparedAction, DbError> {
    let rest = rest.trim();
    let (name, arguments) = if let Some(open) = rest.find('(') {
        let close = find_matching_parenthesis(rest, open)
            .ok_or_else(|| syntax_error("EXECUTE has an unterminated argument list"))?;
        if !rest[close + 1..].trim().is_empty() {
            return Err(syntax_error("unexpected text after EXECUTE argument list"));
        }
        let arguments = split_csv(&rest[open + 1..close])?
            .into_iter()
            .map(parse_sql_literal)
            .collect::<Result<Vec<_>, _>>()?;
        (normalize_name(&rest[..open])?, arguments)
    } else {
        (normalize_name(rest)?, Vec::new())
    };
    Ok(SqlPreparedAction::Execute { name, arguments })
}

fn parse_deallocate(rest: &str) -> Result<SqlDeallocateTarget, DbError> {
    let rest = strip_keyword(rest.trim(), "PREPARE").unwrap_or(rest).trim();
    if rest.eq_ignore_ascii_case("ALL") {
        Ok(SqlDeallocateTarget::All)
    } else {
        normalize_name(rest).map(SqlDeallocateTarget::Named)
    }
}

fn parse_parameter_type(input: &str) -> Result<LogicalType, DbError> {
    match input.trim().to_ascii_lowercase().as_str() {
        "oid" | "pg_catalog.oid" | "int" | "integer" | "int4" | "pg_catalog.int4" => {
            Ok(LogicalType::Int4)
        }
        "bigint" | "int8" | "pg_catalog.int8" => Ok(LogicalType::Int8),
        "smallint" | "int2" | "pg_catalog.int2" => Ok(LogicalType::Int2),
        "text" | "pg_catalog.text" => Ok(LogicalType::Text),
        other => Err(DbError {
            category: ErrorCategory::Unsupported,
            message: format!("SQL PREPARE parameter type {other:?} is not supported"),
        }),
    }
}

fn decode_sql_execute_argument(ty: LogicalType, value: Option<&str>) -> Result<DbValue, DbError> {
    let Some(value) = value else {
        return Ok(DbValue::Null);
    };
    let invalid = || invalid_request(format!("invalid {ty:?} SQL EXECUTE argument"));
    match ty {
        LogicalType::Int2 => value
            .parse::<i16>()
            .map(DbValue::Int2)
            .map_err(|_| invalid()),
        LogicalType::Int4 => value
            .parse::<i32>()
            .map(DbValue::Int4)
            .map_err(|_| invalid()),
        LogicalType::Int8 => value
            .parse::<i64>()
            .map(DbValue::Int8)
            .map_err(|_| invalid()),
        LogicalType::Text => Ok(DbValue::Text(value.to_string())),
        _ => Err(DbError {
            category: ErrorCategory::Unsupported,
            message: format!("SQL EXECUTE does not support {ty:?} parameters"),
        }),
    }
}

fn parse_sql_literal(input: &str) -> Result<Option<String>, DbError> {
    let input = input.trim();
    if input.eq_ignore_ascii_case("NULL") {
        return Ok(None);
    }
    if let Some(inner) = input
        .strip_prefix('\'')
        .and_then(|value| value.strip_suffix('\''))
    {
        return Ok(Some(inner.replace("''", "'")));
    }
    if input.is_empty()
        || input
            .chars()
            .any(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '_' | '.')))
    {
        return Err(syntax_error("unsupported SQL EXECUTE argument"));
    }
    Ok(Some(input.to_string()))
}

fn split_csv(input: &str) -> Result<Vec<&str>, DbError> {
    if input.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    let bytes = input.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\'' if quoted && bytes.get(index + 1) == Some(&b'\'') => index += 1,
            b'\'' => quoted = !quoted,
            b',' if !quoted => {
                parts.push(input[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    if quoted {
        return Err(syntax_error("unterminated string in SQL argument list"));
    }
    parts.push(input[start..].trim());
    if parts.iter().any(|part| part.is_empty()) {
        return Err(syntax_error("empty SQL argument or type"));
    }
    Ok(parts)
}

fn find_matching_parenthesis(input: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut quoted = false;
    let bytes = input.as_bytes();
    let mut index = open;
    while index < bytes.len() {
        match bytes[index] {
            b'\'' if quoted && bytes.get(index + 1) == Some(&b'\'') => index += 1,
            b'\'' => quoted = !quoted,
            b'(' if !quoted => depth += 1,
            b')' if !quoted => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn normalize_name(input: &str) -> Result<String, DbError> {
    let input = input.trim();
    let mut chars = input.chars();
    let Some(first) = chars.next() else {
        return Err(syntax_error("prepared statement name is empty"));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || chars.any(|ch| !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()))
    {
        return Err(syntax_error("invalid prepared statement name"));
    }
    Ok(input.to_ascii_lowercase())
}

fn strip_keyword<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let prefix = input.get(..keyword.len())?;
    if !prefix.eq_ignore_ascii_case(keyword) {
        return None;
    }
    let rest = &input[keyword.len()..];
    rest.chars()
        .next()
        .is_none_or(char::is_whitespace)
        .then_some(rest.trim_start())
}

fn syntax_error(message: impl Into<String>) -> DbError {
    DbError {
        category: ErrorCategory::Syntax,
        message: message.into(),
    }
}

fn invalid_request(message: impl Into<String>) -> DbError {
    DbError {
        category: ErrorCategory::InvalidRequest,
        message: message.into(),
    }
}

fn internal_error(message: impl Into<String>) -> DbError {
    DbError {
        category: ErrorCategory::Internal,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pg16_dump_prepare_and_execute_keep_a_typed_oid_argument() {
        let prepare = classify_sql_prepared_statement(
            "PREPARE getDomainConstraints(pg_catalog.oid) AS SELECT 1 WHERE 1 = $1",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            prepare,
            SqlPreparedAction::Prepare {
                name,
                parameter_hints,
                ..
            } if name == "getdomainconstraints" && parameter_hints == vec![Some(LogicalType::Int4)]
        ));
        assert_eq!(
            classify_sql_prepared_statement("EXECUTE getDomainConstraints('123')").unwrap(),
            Some(SqlPreparedAction::Execute {
                name: "getdomainconstraints".to_string(),
                arguments: vec![Some("123".to_string())],
            })
        );
    }
}
