//! SQL-level PREPARE/EXECUTE session syntax over the canonical prepared AST owner.
//!
//! This layer owns only connection-local names and SQL literal decoding. PREPARE delegates parse
//! and catalog description to the facade; EXECUTE binds the retained AST and submits it through
//! the same `SharedEngine::submit` boundary as extended-query protocol portals.

use gpu_db_facade::{
    pg_adapter, BoundPreparedStatement, DbError, DbValue, ErrorCategory, LogicalType,
    PreparedStatement,
};
use gpu_db_protocol::{canonicalize_sql_for_exact_match, sql_may_start_with_any_keyword};
use pg_query::protobuf::{a_const, AConst, Integer, Node, ScanToken, Token, TypeName};
use pg_query::NodeEnum;

const MAX_SQL_PREPARE_PARAMETERS: usize = 16;
const RELATIONAL_SELECT_ONLY: &str = "extended query protocol only supports relational SELECT";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlPreparedAction {
    Prepare {
        name: String,
        query: String,
        parameter_hints: Vec<Option<LogicalType>>,
    },
    Execute {
        name: String,
        arguments: Vec<SqlExecuteLiteral>,
    },
    Deallocate(SqlDeallocateTarget),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlDeallocateTarget {
    All,
    Named(String),
}

/// An extended-protocol `EXECUTE name($n, ...)` wrapper around a session SQL prepared statement.
///
/// The outer `$n` values belong to the wire statement being Parsed; the target name remains
/// connection-local and is resolved by `ExtendedSession` before Bind. Literal SQL EXECUTE
/// arguments continue to use the simple-query action above.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExtendedSqlExecute {
    pub(crate) name: String,
    pub(crate) arguments: Vec<ExtendedSqlExecuteArgument>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExtendedSqlExecuteArgument {
    OuterParameter {
        index: usize,
        explicit_type: Option<LogicalType>,
    },
    Literal(SqlExecuteLiteral),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SqlExecuteLiteral {
    pub(crate) value: Option<String>,
    pub(crate) explicit_type: Option<LogicalType>,
}

#[derive(Debug, Clone)]
pub(crate) struct SqlPreparedPlan {
    pub(crate) prepared: PreparedStatement,
    pub(crate) deferred_execution_error: Option<DbError>,
}

impl SqlPreparedPlan {
    pub(crate) fn ready(prepared: PreparedStatement) -> Self {
        Self {
            prepared,
            deferred_execution_error: None,
        }
    }
}

pub(crate) fn fill_unused_parameter_holes(
    statement: &str,
    mut hints: Vec<Option<LogicalType>>,
) -> Vec<Option<LogicalType>> {
    let Ok(scan) = pg_query::scan(statement) else {
        return hints;
    };
    let referenced = scan
        .tokens
        .iter()
        .filter_map(|token| {
            (Token::try_from(token.token).ok()? == Token::Param).then_some(())?;
            let start = usize::try_from(token.start).ok()?;
            let end = usize::try_from(token.end).ok()?;
            statement
                .get(start..end)?
                .strip_prefix('$')?
                .parse::<usize>()
                .ok()
        })
        .filter(|index| *index > 0)
        .collect::<Vec<_>>();
    let Some(parameter_count) = referenced.iter().copied().max() else {
        return hints;
    };
    hints.resize(parameter_count.max(hints.len()), None);
    for index in 1..=parameter_count {
        if !referenced.contains(&index) && hints[index - 1].is_none() {
            hints[index - 1] = Some(LogicalType::Text);
        }
    }
    hints
}

pub(crate) fn classify_sql_prepared_statement(
    statement: &str,
) -> Result<Option<SqlPreparedAction>, DbError> {
    if !compat_classifier_gate(statement, &["PREPARE", "EXECUTE", "DEALLOCATE"]) {
        return Ok(None);
    }
    let source = statement;
    let canonical = canonicalize_sql_for_exact_match(statement)
        .map_err(|error| syntax_error(error.to_string()))?;
    let statement = canonical.trim();
    if let Some(rest) = strip_keyword(statement, "PREPARE") {
        return parse_prepare(rest).map(Some);
    }
    if strip_keyword(statement, "EXECUTE").is_some() {
        return parse_postgres_execute(source)
            .and_then(simple_execute_action)
            .map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "DEALLOCATE") {
        return parse_deallocate(rest).map(|target| Some(SqlPreparedAction::Deallocate(target)));
    }
    Ok(None)
}

pub(crate) fn sql_prepare_name(statement: &str) -> Result<Option<String>, DbError> {
    if !compat_classifier_gate(statement, &["PREPARE"]) {
        return Ok(None);
    }
    let canonical = canonicalize_sql_for_exact_match(statement)
        .map_err(|error| syntax_error(error.to_string()))?;
    let Some(rest) = strip_keyword(canonical.trim(), "PREPARE") else {
        return Ok(None);
    };
    take_prepared_name(rest).map(|(name, _)| Some(name))
}

pub(crate) fn classify_extended_sql_execute(
    statement: &str,
) -> Result<Option<ExtendedSqlExecute>, DbError> {
    if !compat_classifier_gate(statement, &["EXECUTE"]) {
        return Ok(None);
    }
    let source = statement;
    let canonical = canonicalize_sql_for_exact_match(statement)
        .map_err(|error| syntax_error(error.to_string()))?;
    let Some(_) = strip_keyword(canonical.trim(), "EXECUTE") else {
        return Ok(None);
    };
    parse_postgres_execute(source).map(Some)
}

fn compat_classifier_gate(statement: &str, candidates: &[&str]) -> bool {
    let admitted = sql_may_start_with_any_keyword(statement, candidates);
    #[cfg(feature = "probe-timing")]
    crate::insert_probe::record_compat_classifier_gate(statement.len() as u64, admitted);
    admitted
}

fn parse_postgres_execute(statement: &str) -> Result<ExtendedSqlExecute, DbError> {
    let scan = pg_query::scan(statement).map_err(|_| sql_execute_shape_error())?;
    validate_execute_escape_strings(statement, &scan.tokens)?;
    let normalized;
    let parsed = match pg_query::parse(statement) {
        Ok(parsed) => parsed,
        Err(_) => {
            normalized = normalize_legacy_execute_lexing(statement, &scan.tokens)?
                .ok_or_else(sql_execute_shape_error)?;
            pg_query::parse(&normalized).map_err(|_| sql_execute_shape_error())?
        }
    };
    let [raw] = parsed.protobuf.stmts.as_slice() else {
        return Err(sql_execute_shape_error());
    };
    let Some(NodeEnum::ExecuteStmt(execute)) =
        raw.stmt.as_deref().and_then(|node| node.node.as_ref())
    else {
        return Err(sql_execute_shape_error());
    };
    let arguments = execute
        .params
        .iter()
        .map(|argument| parse_postgres_execute_argument(argument, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| sql_execute_shape_error())?;
    Ok(ExtendedSqlExecute {
        name: execute.name.clone(),
        arguments,
    })
}

/// Preserve the legacy PRODUCT-001 boundary for lexer forms that libpg_query 17 no longer admits
/// directly: an empty `EXECUTE name()` list, typed national strings, and newline-bearing comments
/// between adjacent string tokens. The scanner supplies exact token ranges, so quoted and
/// dollar-quoted contents cannot be mistaken for comments or prefixes. This is syntax
/// normalization only; the normalized text must still parse to one bounded PostgreSQL
/// `ExecuteStmt` below.
fn normalize_legacy_execute_lexing(
    statement: &str,
    tokens: &[ScanToken],
) -> Result<Option<String>, DbError> {
    if let Some((start, end)) = empty_execute_parentheses(statement, tokens)? {
        let mut normalized = String::with_capacity(statement.len());
        normalized.push_str(statement.get(..start).ok_or_else(sql_execute_shape_error)?);
        normalized.push_str(statement.get(end..).ok_or_else(sql_execute_shape_error)?);
        return Ok(Some(normalized));
    }
    let mut edits = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let kind = scan_token_kind(token)?;
        let (start, end) = scan_token_range(statement, token)?;
        match kind {
            Token::CComment if statement[start..end].contains('\n') => {
                edits.push((start, end, "\n"));
            }
            Token::SqlComment => {
                edits.push((start, end, ""));
            }
            Token::Nchar if typed_national_string_prefix(statement, tokens, index)? => {
                edits.push((start, end, ""));
            }
            _ => {}
        }
    }
    if edits.is_empty() {
        return Ok(None);
    }
    let mut normalized = String::with_capacity(statement.len());
    let mut cursor = 0;
    for (start, end, replacement) in edits {
        if start < cursor {
            return Err(sql_execute_shape_error());
        }
        normalized.push_str(
            statement
                .get(cursor..start)
                .ok_or_else(sql_execute_shape_error)?,
        );
        normalized.push_str(replacement);
        cursor = end;
    }
    normalized.push_str(
        statement
            .get(cursor..)
            .ok_or_else(sql_execute_shape_error)?,
    );
    Ok(Some(normalized))
}

fn empty_execute_parentheses(
    statement: &str,
    tokens: &[ScanToken],
) -> Result<Option<(usize, usize)>, DbError> {
    let significant = tokens
        .iter()
        .filter_map(|token| match scan_token_kind(token) {
            Ok(Token::CComment | Token::SqlComment) => None,
            result => Some(result.map(|kind| (token, kind))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let significant = match significant.as_slice() {
        [execute, name, open, close] => [execute, name, open, close],
        [execute, name, open, close, semicolon] if semicolon.1 == Token::Ascii59 => {
            [execute, name, open, close]
        }
        _ => return Ok(None),
    };
    if significant[0].1 != Token::Execute
        || significant[1].1 != Token::Ident
        || significant[2].1 != Token::Ascii40
        || significant[3].1 != Token::Ascii41
    {
        return Ok(None);
    }
    let (start, _) = scan_token_range(statement, significant[2].0)?;
    let (_, end) = scan_token_range(statement, significant[3].0)?;
    Ok(Some((start, end)))
}

fn typed_national_string_prefix(
    statement: &str,
    tokens: &[ScanToken],
    index: usize,
) -> Result<bool, DbError> {
    if tokens.get(index + 1).map(scan_token_kind).transpose()? != Some(Token::Sconst) {
        return Ok(false);
    }
    if tokens
        .get(index.wrapping_sub(1))
        .map(scan_token_kind)
        .transpose()?
        == Some(Token::TextP)
    {
        return Ok(true);
    }
    let Some(prefix) = index.checked_sub(3) else {
        return Ok(false);
    };
    if scan_token_kind(&tokens[prefix])? != Token::Ident
        || scan_token_kind(&tokens[prefix + 1])? != Token::Ascii46
        || scan_token_kind(&tokens[prefix + 2])? != Token::TextP
    {
        return Ok(false);
    }
    let (start, end) = scan_token_range(statement, &tokens[prefix])?;
    Ok(statement[start..end].eq_ignore_ascii_case("pg_catalog"))
}

/// libpg_query 17 accepts `\\x` without a following hexadecimal digit by dropping the slash.
/// PRODUCT-001 already pins PostgreSQL-compatible rejection at this boundary, so validate only
/// that narrow escape family before consuming the scanner-decoded constant.
fn validate_execute_escape_strings(statement: &str, tokens: &[ScanToken]) -> Result<(), DbError> {
    for token in tokens {
        if scan_token_kind(token)? != Token::Sconst {
            continue;
        }
        let (start, end) = scan_token_range(statement, token)?;
        let raw = &statement[start..end];
        let bytes = raw.as_bytes();
        if bytes.len() < 3
            || !matches!(bytes[0], b'e' | b'E')
            || bytes[1] != b'\''
            || bytes.last() != Some(&b'\'')
        {
            continue;
        }
        let mut index = 2;
        while index + 1 < bytes.len() {
            if bytes[index] == b'\'' && bytes.get(index + 1) == Some(&b'\'') {
                index += 2;
                continue;
            }
            if bytes[index] != b'\\' {
                index += 1;
                continue;
            }
            let Some(escape) = bytes.get(index + 1) else {
                return Err(sql_execute_shape_error());
            };
            if matches!(escape, b'x' | b'X')
                && !bytes.get(index + 2).is_some_and(u8::is_ascii_hexdigit)
            {
                return Err(sql_execute_shape_error());
            }
            index += 2;
        }
    }
    Ok(())
}

fn scan_token_kind(token: &ScanToken) -> Result<Token, DbError> {
    Token::try_from(token.token).map_err(|_| sql_execute_shape_error())
}

fn scan_token_range(statement: &str, token: &ScanToken) -> Result<(usize, usize), DbError> {
    let start = usize::try_from(token.start).map_err(|_| sql_execute_shape_error())?;
    let end = usize::try_from(token.end).map_err(|_| sql_execute_shape_error())?;
    statement
        .get(start..end)
        .ok_or_else(sql_execute_shape_error)?;
    Ok((start, end))
}

fn parse_postgres_execute_argument(
    node: &Node,
    explicit_type: Option<LogicalType>,
) -> Result<ExtendedSqlExecuteArgument, DbError> {
    match node.node.as_ref() {
        Some(NodeEnum::TypeCast(cast)) => {
            let cast_type = parse_execute_cast_type(
                cast.type_name
                    .as_ref()
                    .ok_or_else(|| syntax_error("SQL EXECUTE cast has no target type"))?,
            )?;
            if explicit_type.is_some_and(|outer| outer != cast_type) {
                return Err(unsupported_error(
                    "nested SQL EXECUTE casts must resolve to one supported type",
                ));
            }
            parse_postgres_execute_argument(
                cast.arg
                    .as_deref()
                    .ok_or_else(|| syntax_error("SQL EXECUTE cast has no argument"))?,
                Some(cast_type),
            )
        }
        Some(NodeEnum::ParamRef(parameter)) => {
            let index = usize::try_from(parameter.number)
                .ok()
                .filter(|index| *index != 0)
                .ok_or_else(|| syntax_error("invalid extended SQL EXECUTE parameter reference"))?;
            Ok(ExtendedSqlExecuteArgument::OuterParameter {
                index,
                explicit_type,
            })
        }
        Some(NodeEnum::AConst(constant)) => {
            Ok(ExtendedSqlExecuteArgument::Literal(SqlExecuteLiteral {
                value: execute_constant_text(constant)?,
                explicit_type,
            }))
        }
        Some(NodeEnum::AExpr(expression))
            if expression.kind == pg_query::protobuf::AExprKind::AexprOp as i32
                && expression.lexpr.is_none()
                && matches!(
                    expression.name.as_slice(),
                    [name]
                        if matches!(
                            name.node.as_ref(),
                            Some(NodeEnum::String(operator)) if operator.sval == "+"
                        )
                ) =>
        {
            let operand = expression
                .rexpr
                .as_deref()
                .ok_or_else(sql_execute_shape_error)?;
            if !matches!(
                operand.node.as_ref(),
                Some(NodeEnum::AConst(constant))
                    if matches!(
                        constant.val.as_ref(),
                        Some(a_const::Val::Ival(_) | a_const::Val::Fval(_))
                    )
            ) {
                return Err(sql_execute_shape_error());
            }
            parse_postgres_execute_argument(operand, explicit_type)
        }
        _ => Err(unsupported_error(
            "SQL EXECUTE arguments must be literals or extended-protocol parameters",
        )),
    }
}

fn execute_constant_text(constant: &pg_query::protobuf::AConst) -> Result<Option<String>, DbError> {
    if constant.isnull {
        return Ok(None);
    }
    match constant.val.as_ref() {
        Some(a_const::Val::Ival(value)) => Ok(Some(value.ival.to_string())),
        Some(a_const::Val::Fval(value)) => Ok(Some(value.fval.clone())),
        Some(a_const::Val::Boolval(value)) => Ok(Some(value.boolval.to_string())),
        Some(a_const::Val::Sval(value)) => Ok(Some(value.sval.clone())),
        Some(a_const::Val::Bsval(_)) => Err(unsupported_error(
            "bit-string SQL EXECUTE arguments are not supported",
        )),
        None => Err(syntax_error("SQL EXECUTE constant has no value")),
    }
}

fn parse_execute_cast_type(type_name: &TypeName) -> Result<LogicalType, DbError> {
    if type_name.type_oid != 0
        || type_name.setof
        || type_name.pct_type
        || !type_name.typmods.is_empty()
        || !type_name.array_bounds.is_empty()
    {
        return Err(unsupported_error(
            "SQL EXECUTE casts support only scalar types without modifiers",
        ));
    }
    let name = type_name
        .names
        .iter()
        .map(|node| match node.node.as_ref() {
            Some(NodeEnum::String(part)) => Ok(part.sval.as_str()),
            _ => Err(syntax_error("SQL EXECUTE cast type name is malformed")),
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(".");
    logical_type_for_name(&name)
        // PostgreSQL parses a national-character literal as an implicit bpchar cast. The
        // canonical facade has one bounded text domain, so retain its decoded string as text.
        .or_else(|| {
            matches!(name.as_str(), "bpchar" | "pg_catalog.bpchar").then_some(LogicalType::Text)
        })
        .ok_or_else(|| {
            unsupported_error(format!("SQL EXECUTE cast type {name:?} is not supported"))
        })
}

fn simple_execute_action(execute: ExtendedSqlExecute) -> Result<SqlPreparedAction, DbError> {
    let arguments = execute
        .arguments
        .into_iter()
        .map(|argument| match argument {
            ExtendedSqlExecuteArgument::Literal(literal) => Ok(literal),
            ExtendedSqlExecuteArgument::OuterParameter { .. } => Err(syntax_error(
                "SQL EXECUTE parameters require the extended query protocol",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SqlPreparedAction::Execute {
        name: execute.name,
        arguments,
    })
}

pub(crate) fn bind_sql_execute(
    statement: &PreparedStatement,
    arguments: &[SqlExecuteLiteral],
) -> Result<BoundPreparedStatement, DbError> {
    let parameter_types = statement.parameter_types().ok_or_else(|| {
        internal_error("SQL prepared statement lost its catalog-resolved parameter types")
    })?;
    if parameter_types.len() != arguments.len() {
        return Err(invalid_request(
            "bound parameter count does not match prepared statement",
        ));
    }
    let values = parameter_types
        .iter()
        .copied()
        .zip(arguments)
        .map(|(ty, literal)| decode_sql_execute_literal(ty, literal))
        .collect::<Result<Vec<_>, _>>()?;
    statement.bind_values(&values)
}

/// Return an equivalent query whose negative LIMIT/OFFSET is harmless during effect-free
/// description. PostgreSQL retains these SQL prepared statements and raises the frozen error only
/// when EXECUTE evaluates the bound plan. The caller must retain `parse_error` and reject execution
/// before submitting this description-only clone to the engine.
pub(crate) fn deferred_sql_prepare_analysis_query(
    query: &str,
    parse_error: &DbError,
) -> Option<String> {
    let replace_limit = match parse_error.message.as_str() {
        "LIMIT must not be negative" => true,
        "OFFSET must not be negative" => false,
        _ => return None,
    };
    let mut parsed = pg_query::parse(query).ok()?.protobuf;
    let [raw] = parsed.stmts.as_mut_slice() else {
        return None;
    };
    let NodeEnum::SelectStmt(select) = raw.stmt.as_deref_mut()?.node.as_mut()? else {
        return None;
    };
    let target = if replace_limit {
        &mut select.limit_count
    } else {
        &mut select.limit_offset
    };
    target.as_ref()?;
    *target = Some(Box::new(Node {
        node: Some(NodeEnum::AConst(AConst {
            isnull: false,
            location: -1,
            val: Some(a_const::Val::Ival(Integer { ival: 0 })),
        })),
    }));
    parsed.deparse().ok()
}

fn parse_prepare(rest: &str) -> Result<SqlPreparedAction, DbError> {
    let rest = rest.trim();
    let (name, after_name) = take_prepared_name(rest)?;
    let after_name = after_name.trim_start();
    let (after_types, parameter_hints) = if after_name.starts_with('(') {
        let close = find_matching_parenthesis(after_name, 0)
            .ok_or_else(|| syntax_error("PREPARE has an unterminated type list"))?;
        let types = split_csv(&after_name[1..close])?;
        if types.len() > MAX_SQL_PREPARE_PARAMETERS {
            return Err(invalid_request(format!(
                "SQL PREPARE supports at most {MAX_SQL_PREPARE_PARAMETERS} parameters"
            )));
        }
        let parameter_hints = types
            .into_iter()
            .map(|ty| parse_parameter_type(ty).map(Some))
            .collect::<Result<Vec<_>, _>>()?;
        (&after_name[close + 1..], parameter_hints)
    } else {
        (after_name, Vec::new())
    };
    let query = strip_keyword(after_types.trim_start(), "AS")
        .ok_or_else(|| syntax_error("PREPARE requires AS and a query"))?
        .trim();
    if query.is_empty() {
        return Err(syntax_error("PREPARE query is empty"));
    }
    Ok(SqlPreparedAction::Prepare {
        name,
        query: query.to_string(),
        parameter_hints,
    })
}

fn parse_deallocate(rest: &str) -> Result<SqlDeallocateTarget, DbError> {
    let rest = strip_keyword(rest.trim(), "PREPARE").unwrap_or(rest).trim();
    if rest.eq_ignore_ascii_case("ALL") {
        Ok(SqlDeallocateTarget::All)
    } else {
        let (name, trailing) = take_prepared_name(rest)?;
        if !trailing.trim().is_empty() {
            return Err(syntax_error(
                "unexpected text after DEALLOCATE statement name",
            ));
        }
        Ok(SqlDeallocateTarget::Named(name))
    }
}

fn parse_parameter_type(input: &str) -> Result<LogicalType, DbError> {
    let normalized = input.trim().to_ascii_lowercase();
    logical_type_for_name(&normalized).ok_or_else(|| DbError {
        category: ErrorCategory::Unsupported,
        message: format!("SQL PREPARE parameter type {normalized:?} is not supported"),
    })
}

fn logical_type_for_name(input: &str) -> Option<LogicalType> {
    match input {
        "oid" | "pg_catalog.oid" | "int" | "integer" | "pg_catalog.integer" | "int4"
        | "pg_catalog.int4" => Some(LogicalType::Int4),
        "bigint" | "pg_catalog.bigint" | "int8" | "pg_catalog.int8" => Some(LogicalType::Int8),
        "smallint" | "pg_catalog.smallint" | "int2" | "pg_catalog.int2" => Some(LogicalType::Int2),
        "text" | "pg_catalog.text" => Some(LogicalType::Text),
        _ => None,
    }
}

pub(crate) fn decode_sql_execute_literal(
    target_type: LogicalType,
    literal: &SqlExecuteLiteral,
) -> Result<DbValue, DbError> {
    if literal.value.is_none() {
        return Err(unsupported_error(
            "NULL SQL EXECUTE parameters are not supported",
        ));
    }
    if literal
        .explicit_type
        .is_some_and(|explicit_type| explicit_type != target_type)
    {
        return Err(unsupported_error(format!(
            "SQL EXECUTE argument cast {:?} does not match prepared parameter type {target_type:?}",
            literal.explicit_type.expect("checked as Some")
        )));
    }
    decode_sql_execute_argument(target_type, literal.value.as_deref())
}

pub(crate) fn decode_sql_execute_argument(
    ty: LogicalType,
    value: Option<&str>,
) -> Result<DbValue, DbError> {
    let Some(value) = value else {
        return Ok(DbValue::Null);
    };
    let invalid = || {
        invalid_request(format!(
            "invalid input syntax for parameter type oid {}: {value:?}",
            pg_adapter::logical_type_oid(ty)
        ))
    };
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

fn take_prepared_name(input: &str) -> Result<(String, &str), DbError> {
    let input = input.trim_start();
    if let Some(quoted) = input.strip_prefix('"') {
        let mut name = String::new();
        let mut chars = quoted.char_indices().peekable();
        while let Some((index, ch)) = chars.next() {
            if ch == '"' {
                if chars.peek().is_some_and(|(_, next)| *next == '"') {
                    chars.next();
                    name.push('"');
                    continue;
                }
                if name.is_empty() {
                    return Err(syntax_error("prepared statement name is empty"));
                }
                return Ok((name, &quoted[index + 1..]));
            }
            name.push(ch);
        }
        return Err(syntax_error("unterminated quoted prepared statement name"));
    }
    let end = input
        .find(|ch: char| ch.is_whitespace() || ch == '(')
        .unwrap_or(input.len());
    let name = &input[..end];
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(syntax_error("prepared statement name is empty"));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || chars.any(|ch| !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()))
    {
        return Err(syntax_error("invalid prepared statement name"));
    }
    Ok((name.to_ascii_lowercase(), &input[end..]))
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

fn unsupported_error(message: impl Into<String>) -> DbError {
    DbError {
        category: ErrorCategory::Unsupported,
        message: message.into(),
    }
}

fn sql_execute_shape_error() -> DbError {
    unsupported_error(RELATIONAL_SELECT_ONLY)
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
                arguments: vec![SqlExecuteLiteral {
                    value: Some("123".to_string()),
                    explicit_type: None,
                }],
            })
        );
    }

    #[test]
    fn prepare_without_a_type_list_retains_an_empty_parameter_contract() {
        assert_eq!(
            classify_sql_prepared_statement("PREPARE literal_lookup AS SELECT 1").unwrap(),
            Some(SqlPreparedAction::Prepare {
                name: "literal_lookup".to_string(),
                query: "select 1".to_string(),
                parameter_hints: Vec::new(),
            })
        );
    }

    #[test]
    fn sparse_extended_parameter_holes_default_only_unreferenced_slots_to_text() {
        let hints = fill_unused_parameter_holes(
            "SELECT id FROM accounts WHERE id = $1 OR id = $10 LIMIT $11",
            Vec::new(),
        );
        assert_eq!(hints.len(), 11);
        assert_eq!(hints[0], None);
        assert!(hints[1..9]
            .iter()
            .all(|hint| *hint == Some(LogicalType::Text)));
        assert_eq!(hints[9], None);
        assert_eq!(hints[10], None);
    }

    #[test]
    fn prepare_name_probe_stops_before_parameter_type_semantics() {
        assert_eq!(
            sql_prepare_name("/* lead */ PREPARE \"Existing Name\"(jsonb) AS SELECT $1").unwrap(),
            Some("Existing Name".to_string())
        );
        assert_eq!(sql_prepare_name("SELECT 1").unwrap(), None);
    }

    #[test]
    fn lexical_gate_preserves_every_prepared_action_after_leading_comments_and_case_fold() {
        for statement in [
            "/* gate */ pRePaRe p AS SELECT 1",
            "/* gate */ eXeCuTe p()",
            "/* gate */ dEaLlOcAtE p",
            "PREPARE\u{00a0}\u{2003}unicode_space AS SELECT 1",
        ] {
            assert!(classify_sql_prepared_statement(statement)
                .unwrap()
                .is_some());
        }
        assert_eq!(
            classify_sql_prepared_statement("PREPAREfoo p").unwrap(),
            None
        );
        assert_eq!(sql_prepare_name("PREPAREfoo p").unwrap(), None);
        assert_eq!(
            sql_prepare_name("PREPARE\u{00a0}\u{2003}unicode_space AS SELECT 1").unwrap(),
            Some("unicode_space".to_string())
        );
        assert_eq!(classify_extended_sql_execute("EXECUTEfoo p()"), Ok(None));
    }

    #[test]
    fn quoted_prepared_names_preserve_case_and_escaped_quotes() {
        let prepared =
            classify_sql_prepared_statement("PREPARE \"lookup \"\"quoted\"\"\"(int4) AS SELECT $1")
                .unwrap()
                .unwrap();
        assert!(matches!(
            prepared,
            SqlPreparedAction::Prepare { name, .. } if name == "lookup \"quoted\""
        ));
        assert_eq!(
            classify_sql_prepared_statement("EXECUTE \"lookup \"\"quoted\"\"\"(2)").unwrap(),
            Some(SqlPreparedAction::Execute {
                name: "lookup \"quoted\"".to_string(),
                arguments: vec![SqlExecuteLiteral {
                    value: Some("2".to_string()),
                    explicit_type: None,
                }],
            })
        );
        assert_eq!(
            classify_sql_prepared_statement("DEALLOCATE PREPARE \"lookup \"\"quoted\"\"\"")
                .unwrap(),
            Some(SqlPreparedAction::Deallocate(SqlDeallocateTarget::Named(
                "lookup \"quoted\"".to_string()
            )))
        );
    }

    #[test]
    fn prepare_ignores_comments_and_execute_accepts_postfix_casts() {
        let prepared = classify_sql_prepared_statement(
            "PREPARE comment_lookup(/* id */ int4, /* name */ text) /* target */ AS SELECT $1, $2",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            prepared,
            SqlPreparedAction::Prepare { parameter_hints, .. }
                if parameter_hints == vec![Some(LogicalType::Int4), Some(LogicalType::Text)]
        ));
        assert_eq!(
            classify_sql_prepared_statement("EXECUTE /* stmt */ comment_lookup(3, 'Grace'::text)")
                .unwrap(),
            Some(SqlPreparedAction::Execute {
                name: "comment_lookup".to_string(),
                arguments: vec![
                    SqlExecuteLiteral {
                        value: Some("3".to_string()),
                        explicit_type: None,
                    },
                    SqlExecuteLiteral {
                        value: Some("Grace".to_string()),
                        explicit_type: Some(LogicalType::Text),
                    },
                ],
            })
        );
    }

    #[test]
    fn prepare_without_as_returns_a_syntax_error() {
        let error =
            classify_sql_prepared_statement("PREPARE missing_as(int4) SELECT $1").unwrap_err();
        assert_eq!(error.category, ErrorCategory::Syntax);
        assert_eq!(error.message, "PREPARE requires AS and a query");
    }

    #[test]
    fn execute_literals_accept_parentheses_and_postfix_casts() {
        assert_eq!(
            classify_sql_prepared_statement(
                "EXECUTE lookup(((3)), ('Grace')::pg_catalog.text, (2::int4))",
            )
            .unwrap(),
            Some(SqlPreparedAction::Execute {
                name: "lookup".to_string(),
                arguments: vec![
                    SqlExecuteLiteral {
                        value: Some("3".to_string()),
                        explicit_type: None,
                    },
                    SqlExecuteLiteral {
                        value: Some("Grace".to_string()),
                        explicit_type: Some(LogicalType::Text),
                    },
                    SqlExecuteLiteral {
                        value: Some("2".to_string()),
                        explicit_type: Some(LogicalType::Int4),
                    },
                ],
            })
        );
    }

    #[test]
    fn execute_accepts_legacy_empty_parentheses_and_unary_plus_numeric_literals() {
        assert_eq!(
            classify_sql_prepared_statement("EXECUTE lookup()").unwrap(),
            Some(SqlPreparedAction::Execute {
                name: "lookup".to_string(),
                arguments: Vec::new(),
            })
        );
        assert_eq!(
            classify_sql_prepared_statement("EXECUTE lookup(+1, (+2))").unwrap(),
            Some(SqlPreparedAction::Execute {
                name: "lookup".to_string(),
                arguments: vec![
                    SqlExecuteLiteral {
                        value: Some("1".to_string()),
                        explicit_type: None,
                    },
                    SqlExecuteLiteral {
                        value: Some("2".to_string()),
                        explicit_type: None,
                    },
                ],
            })
        );
        let error = classify_sql_prepared_statement("EXECUTE lookup(+'1')").unwrap_err();
        assert_eq!(error.message, RELATIONAL_SELECT_ONLY);
    }

    #[test]
    fn extended_execute_retains_outer_parameter_order_and_reuse() {
        assert_eq!(
            classify_extended_sql_execute("EXECUTE lookup($2, $1, $2)").unwrap(),
            Some(ExtendedSqlExecute {
                name: "lookup".to_string(),
                arguments: vec![
                    ExtendedSqlExecuteArgument::OuterParameter {
                        index: 2,
                        explicit_type: None,
                    },
                    ExtendedSqlExecuteArgument::OuterParameter {
                        index: 1,
                        explicit_type: None,
                    },
                    ExtendedSqlExecuteArgument::OuterParameter {
                        index: 2,
                        explicit_type: None,
                    },
                ],
            })
        );
        assert_eq!(
            classify_extended_sql_execute("EXECUTE lookup(($2)::int4, 'Ada')").unwrap(),
            Some(ExtendedSqlExecute {
                name: "lookup".to_string(),
                arguments: vec![
                    ExtendedSqlExecuteArgument::OuterParameter {
                        index: 2,
                        explicit_type: Some(LogicalType::Int4),
                    },
                    ExtendedSqlExecuteArgument::Literal(SqlExecuteLiteral {
                        value: Some("Ada".to_string()),
                        explicit_type: None,
                    }),
                ],
            })
        );
    }

    #[test]
    fn postgres_execute_ast_decodes_supported_string_literal_families() {
        for (sql, expected) in [
            ("EXECUTE lookup(E'Ada\\x20Lovelace')", "Ada Lovelace"),
            (
                "EXECUTE lookup($tag$Ada, (Lovelace)$tag$)",
                "Ada, (Lovelace)",
            ),
            ("EXECUTE lookup(U&'Ada\\0020Lovelace')", "Ada Lovelace"),
            (
                "EXECUTE lookup(U&'Ada!0020Lovelace' UESCAPE '!')",
                "Ada Lovelace",
            ),
            ("EXECUTE lookup('Ada'\n' Lovelace')", "Ada Lovelace"),
            ("EXECUTE lookup(N'Ada''s notes')", "Ada's notes"),
            ("EXECUTE lookup(text N'Grace Hopper')", "Grace Hopper"),
            (
                "EXECUTE lookup('Ada' /* keep newline\n */ ' Lovelace')",
                "Ada Lovelace",
            ),
            (
                "EXECUTE lookup('Grace' -- keep newline\n ' Hopper')",
                "Grace Hopper",
            ),
        ] {
            let action = classify_sql_prepared_statement(sql).unwrap().unwrap();
            assert!(matches!(
                action,
                SqlPreparedAction::Execute { arguments, .. }
                    if arguments.len() == 1 && arguments[0].value.as_deref() == Some(expected)
            ));
        }
    }

    #[test]
    fn postgres_execute_ast_retains_cast_types_and_rejects_nonliteral_expressions() {
        let execute = classify_extended_sql_execute(
            "EXECUTE lookup(CAST($1 AS pg_catalog.int4), pg_catalog.text $$Ada$$)",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            execute.arguments,
            vec![
                ExtendedSqlExecuteArgument::OuterParameter {
                    index: 1,
                    explicit_type: Some(LogicalType::Int4),
                },
                ExtendedSqlExecuteArgument::Literal(SqlExecuteLiteral {
                    value: Some("Ada".to_string()),
                    explicit_type: Some(LogicalType::Text),
                }),
            ]
        );
        assert!(classify_extended_sql_execute("EXECUTE lookup($1 + 1)").is_err());
        for sql in [
            "EXECUTE lookup($1 + 1)",
            "EXECUTE lookup($1::jsonb)",
            "EXECUTE lookup('Ada' /* no newline */ 'Lovelace')",
            "EXECUTE lookup(U&'bad\\00xz')",
            "EXECUTE lookup(E'bad\\xzz')",
        ] {
            let error = classify_extended_sql_execute(sql).unwrap_err();
            assert_eq!(error.category, ErrorCategory::Unsupported);
            assert_eq!(error.message, RELATIONAL_SELECT_ONLY);
        }
    }

    #[test]
    fn postgres_execute_compat_normalization_does_not_reinterpret_quoted_contents() {
        for (sql, expected) in [
            ("EXECUTE lookup($$-- not a comment$$)", "-- not a comment"),
            (
                "EXECUTE lookup($tag$/* not a comment\n */$tag$)",
                "/* not a comment\n */",
            ),
            ("EXECUTE lookup(E'\\\\xzz')", "\\xzz"),
        ] {
            let action = classify_sql_prepared_statement(sql).unwrap().unwrap();
            assert!(matches!(
                action,
                SqlPreparedAction::Execute { arguments, .. }
                    if arguments.len() == 1 && arguments[0].value.as_deref() == Some(expected)
            ));
        }
    }

    #[test]
    fn postgres_execute_accepts_schema_qualified_builtin_integer_aliases() {
        for (sql, expected_type) in [
            ("EXECUTE lookup(pg_catalog.integer '1')", LogicalType::Int4),
            ("EXECUTE lookup(pg_catalog.bigint '1')", LogicalType::Int8),
            ("EXECUTE lookup(pg_catalog.smallint '1')", LogicalType::Int2),
        ] {
            let action = classify_sql_prepared_statement(sql).unwrap().unwrap();
            assert!(matches!(
                action,
                SqlPreparedAction::Execute { arguments, .. }
                    if arguments.len() == 1
                        && arguments[0].value.as_deref() == Some("1")
                        && arguments[0].explicit_type == Some(expected_type)
            ));
        }
    }

    #[test]
    fn explicit_execute_cast_must_match_the_prepared_parameter_type() {
        let literal = SqlExecuteLiteral {
            value: Some("1".to_string()),
            explicit_type: Some(LogicalType::Text),
        };
        let error = decode_sql_execute_literal(LogicalType::Int4, &literal).unwrap_err();
        assert_eq!(error.category, ErrorCategory::Unsupported);
        assert!(error.message.contains("does not match"));
    }

    #[test]
    fn sql_execute_literal_null_keeps_the_frozen_unsupported_boundary() {
        let literal = SqlExecuteLiteral {
            value: None,
            explicit_type: None,
        };
        let error = decode_sql_execute_literal(LogicalType::Int4, &literal).unwrap_err();
        assert_eq!(error.category, ErrorCategory::Unsupported);
        assert_eq!(
            error.message,
            "NULL SQL EXECUTE parameters are not supported"
        );
    }

    #[test]
    fn negative_limit_prepare_uses_only_a_description_safe_clone() {
        let error = DbError {
            category: ErrorCategory::Syntax,
            message: "LIMIT must not be negative".to_string(),
        };
        let analysis_query = deferred_sql_prepare_analysis_query(
            "SELECT id FROM accounts ORDER BY id LIMIT -1",
            &error,
        )
        .unwrap();
        assert!(analysis_query.contains("LIMIT 0"));
        PreparedStatement::parse(&analysis_query).unwrap();

        let unrelated = DbError {
            category: ErrorCategory::Syntax,
            message: "some other syntax error".to_string(),
        };
        assert!(deferred_sql_prepare_analysis_query("SELECT 1", &unrelated).is_none());
    }

    #[test]
    fn sql_execute_invalid_literal_uses_the_postgresql_oid_diagnostic() {
        let error =
            decode_sql_execute_argument(LogicalType::Int4, Some("not-a-limit")).unwrap_err();
        assert_eq!(error.category, ErrorCategory::InvalidRequest);
        assert_eq!(
            error.message,
            "invalid input syntax for parameter type oid 23: \"not-a-limit\""
        );
    }
}
