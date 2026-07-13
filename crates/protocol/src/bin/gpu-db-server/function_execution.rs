// Legacy bounded-function ownership. This is not a product execution path.

use super::{
    column_for_sql_type, format_sql_value, function_access_permission_error,
    schema_permission_error, schema_usage_permission_error, write_command_complete, write_error,
    write_select_rows, CatalogCommentTarget, Command, ErrorField, FunctionInfo, ReadWrite,
    SchemaPrivilege, SelectResult, Session, SqlType, SqlValue,
};
use std::collections::BTreeMap;
use std::io;

pub(super) fn rename_function_in_session(
    session: &mut Session,
    old_name: &str,
    new_name: &str,
) -> Result<(), ErrorField> {
    if !session.functions.contains_key(old_name) {
        return Err(ErrorField {
            code: "42883",
            message: "function does not exist",
            position: None,
        });
    }
    if session.functions.contains_key(new_name) {
        return Err(ErrorField {
            code: "42723",
            message: "function already exists with same argument types",
            position: None,
        });
    }
    let mut function = session
        .functions
        .remove(old_name)
        .expect("function existence validated");
    function.name = new_name.to_string();
    session.functions.insert(new_name.to_string(), function);
    session.mark_function_dirty(old_name.to_string());
    session.mark_function_dirty(new_name.to_string());

    let old_target = CatalogCommentTarget::Function {
        function: old_name.to_string(),
    };
    if let Some(comment) = session.comments.remove(&old_target) {
        let new_target = CatalogCommentTarget::Function {
            function: new_name.to_string(),
        };
        session.comments.insert(new_target.clone(), comment);
        session.mark_comment_dirty(old_target);
        session.mark_comment_dirty(new_target);
    }

    session.persist_catalog_snapshot();
    Ok(())
}

pub(super) fn execute_function_command(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
    include_row_description: bool,
) -> io::Result<()> {
    match command {
        Command::CreateFunction(create) => {
            if !session.public_schema_exists {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "3F000",
                        message: "schema does not exist",
                        position: None,
                    },
                );
            }
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if session.functions.contains_key(&create.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42723",
                        message: "function already exists with same argument types",
                        position: None,
                    },
                );
            }
            let oid = session.next_relation_oid;
            let Some(next_oid) = session.next_relation_oid.checked_add(1) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "54000",
                        message: "function OID allocation exhausted",
                        position: None,
                    },
                );
            };
            session.next_relation_oid = next_oid;
            session.functions.insert(
                create.name.clone(),
                FunctionInfo {
                    oid,
                    name: create.name.clone(),
                    return_type: create.return_type,
                    body: create.body,
                    acl: BTreeMap::new(),
                },
            );
            session.mark_function_dirty(create.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE FUNCTION")
        }
        Command::RenameFunction(rename) => {
            if let Err(error) =
                rename_function_in_session(session, &rename.old_name, &rename.new_name)
            {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER FUNCTION")
        }
        Command::DropFunction(drop) => {
            if !drop.if_exists && !session.functions.contains_key(&drop.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42883",
                        message: "function does not exist",
                        position: None,
                    },
                );
            }
            if session.functions.remove(&drop.name).is_some() {
                let target = CatalogCommentTarget::Function {
                    function: drop.name.clone(),
                };
                session.comments.remove(&target);
                session.mark_comment_dirty(target);
            }
            session.mark_function_dirty(drop.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP FUNCTION")
        }
        Command::SelectFunction(call) => {
            let result = match execute_function_result(session, &call) {
                Ok(result) => result,
                Err(error) => return write_error(stream, &error),
            };
            write_select_rows(
                stream,
                &result.columns,
                &result.rows,
                include_row_description,
            )
        }
        _ => unreachable!("function executor called with an unrelated command"),
    }
}

pub(super) fn execute_function_result(
    session: &Session,
    call: &gpu_db_protocol::SelectFunction,
) -> Result<SelectResult, ErrorField> {
    if let Some(error) = schema_usage_permission_error(session, "public") {
        return Err(error);
    }
    let Some(function) = session.functions.get(&call.name) else {
        return Err(ErrorField {
            code: "42883",
            message: "function does not exist",
            position: None,
        });
    };
    if let Some(error) = function_access_permission_error(session, &call.name) {
        return Err(error);
    }
    let value = parse_bounded_sql_function_body(&function.body, function.return_type)?;
    let column = column_for_sql_type(function.return_type, &function.name);
    Ok(SelectResult {
        columns: vec![column],
        rows: vec![vec![Some(format_sql_value(&value))]],
    })
}

fn parse_bounded_sql_function_body(
    body: &str,
    return_type: SqlType,
) -> Result<SqlValue, ErrorField> {
    let Some(rest) = strip_keyword_prefix_case_insensitive(body.trim(), "SELECT") else {
        return Err(unsupported_function_body_error());
    };
    let literal = rest.trim();
    if literal.is_empty()
        || find_keyword_outside_quotes(literal, "FROM").is_some()
        || find_keyword_outside_quotes(literal, "WHERE").is_some()
        || find_keyword_outside_quotes(literal, "ORDER").is_some()
        || find_keyword_outside_quotes(literal, "GROUP").is_some()
        || find_keyword_outside_quotes(literal, "LIMIT").is_some()
        || find_keyword_outside_quotes(literal, "OFFSET").is_some()
        || literal.contains(',')
    {
        return Err(unsupported_function_body_error());
    }
    match return_type {
        SqlType::Int2 => literal
            .parse::<i16>()
            .map(SqlValue::Int2)
            .map_err(|_| unsupported_function_body_error()),
        SqlType::Int4 => literal
            .parse::<i32>()
            .map(SqlValue::Int4)
            .map_err(|_| unsupported_function_body_error()),
        SqlType::Int8 => literal
            .parse::<i64>()
            .map(SqlValue::Int8)
            .map_err(|_| unsupported_function_body_error()),
        SqlType::Numeric { scale, .. } => {
            gpu_db_protocol::Decimal128::parse_at_scale(literal, scale)
                .map(SqlValue::Numeric)
                .ok_or_else(unsupported_function_body_error)
        }
        SqlType::Bool => match literal.to_ascii_lowercase().as_str() {
            "true" | "t" => Ok(SqlValue::Bool(true)),
            "false" | "f" => Ok(SqlValue::Bool(false)),
            _ => Err(unsupported_function_body_error()),
        },
        SqlType::Text => parse_bounded_text_literal(literal)
            .map(SqlValue::Text)
            .ok_or_else(unsupported_function_body_error),
        SqlType::Date => gpu_db_protocol::datetime::parse_date(literal)
            .map(SqlValue::Date)
            .ok_or_else(unsupported_function_body_error),
        SqlType::Timestamp => gpu_db_protocol::datetime::parse_timestamp(literal)
            .map(SqlValue::Timestamp)
            .ok_or_else(unsupported_function_body_error),
        SqlType::Uuid => gpu_db_protocol::uuid::parse_uuid(literal)
            .map(SqlValue::Uuid)
            .ok_or_else(unsupported_function_body_error),
    }
}

fn strip_keyword_prefix_case_insensitive<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    if input.len() < keyword.len() {
        return None;
    }
    let (head, tail) = input.split_at(keyword.len());
    if !head.eq_ignore_ascii_case(keyword) {
        return None;
    }
    if tail
        .chars()
        .next()
        .is_some_and(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())
    {
        return None;
    }
    Some(tail)
}

fn find_keyword_outside_quotes(input: &str, keyword: &str) -> Option<usize> {
    let bytes = input.as_bytes();
    let keyword_bytes = keyword.as_bytes();
    let mut idx = 0;
    let mut in_quote = false;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            if in_quote && idx + 1 < bytes.len() && bytes[idx + 1] == b'\'' {
                idx += 2;
                continue;
            }
            in_quote = !in_quote;
            idx += 1;
            continue;
        }
        if !in_quote
            && idx + keyword_bytes.len() <= bytes.len()
            && input[idx..idx + keyword_bytes.len()].eq_ignore_ascii_case(keyword)
        {
            let before_ok = idx == 0
                || !bytes[idx - 1].is_ascii_alphanumeric()
                    && bytes[idx - 1] != b'_'
                    && bytes[idx - 1] != b'$';
            let after_idx = idx + keyword_bytes.len();
            let after_ok = after_idx == bytes.len()
                || !bytes[after_idx].is_ascii_alphanumeric()
                    && bytes[after_idx] != b'_'
                    && bytes[after_idx] != b'$';
            if before_ok && after_ok {
                return Some(idx);
            }
        }
        idx += 1;
    }
    None
}

fn parse_bounded_text_literal(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    if bytes.len() < 2 || bytes.first() != Some(&b'\'') || bytes.last() != Some(&b'\'') {
        return None;
    }
    let inner = &input[1..input.len() - 1];
    let mut result = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            if chars.peek() == Some(&'\'') {
                chars.next();
                result.push('\'');
            } else {
                return None;
            }
        } else {
            result.push(ch);
        }
    }
    Some(result)
}

fn unsupported_function_body_error() -> ErrorField {
    ErrorField {
        code: "0A000",
        message: "only literal SELECT bodies are supported for SQL function execution",
        position: None,
    }
}
