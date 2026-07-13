use super::frontend_transport::ReadWrite;
use super::{
    begin_copy_from_stdin, bind_parameter_error_field, bind_query_parameters,
    contains_zero_placeholder, describe_extended_query_columns, describe_query_columns,
    execute_copy_to_stdout, execute_declare_cursor, execute_extended_delete,
    execute_extended_insert, execute_extended_update, execute_select_result,
    execute_sql_prepared_result, expected_parameter_count, format_code_count_is_valid,
    is_supported_extended_dml, is_unsupported_declare_cursor_statement, max_placeholder_index,
    negative_limit_error_field, negative_offset_error_field, parse_declare_cursor,
    parse_sql_execute, resolve_prepared_parameter_type_oids, sql_execute_describe_error,
    sql_execute_parameter_type_mapping_error, strip_sql_comments, write_bind_complete,
    write_close_complete, write_command_complete, write_data_row_with_formats, write_error,
    write_no_data, write_parameter_description, write_parse_complete, write_portal_suspended,
    write_row_description, write_row_description_with_formats, BindParameterError, ErrorField,
    Portal, PreparedQuery, PreparedStatement, Session,
};
use gpu_db_protocol::{
    is_copy_statement, is_supported_extended_copy, parse_command, parse_copy_from_stdin,
    parse_copy_to_stdout_table, Command, DescribeTarget, ParseError,
};
use std::io;

fn extended_query_result_column_count(session: &Session, query: &str) -> Option<usize> {
    if parse_declare_cursor(query).is_some()
        || is_supported_extended_dml(session, query)
        || is_supported_extended_copy(query)
    {
        Some(0)
    } else {
        describe_extended_query_columns(session, query).map(|columns| columns.len())
    }
}

pub(super) fn handle_parse(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    statement_name: String,
    query: String,
    parameter_type_oids: Vec<u32>,
) -> io::Result<bool> {
    let query = strip_sql_comments(&query);
    if !statement_name.is_empty() && session.prepared.contains_key(&statement_name) {
        write_error(
            stream,
            &ErrorField {
                code: "42P05",
                message: "prepared statement already exists",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if contains_zero_placeholder(&query) {
        write_error(
            stream,
            &ErrorField {
                code: "42P02",
                message: "there is no parameter $0",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if parameter_type_oids
        .iter()
        .copied()
        .any(|oid| !matches!(oid, 0 | 23 | 25))
    {
        write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "only text and int4 extended-query parameters are supported",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if parameter_type_oids.len() > max_placeholder_index(&query) {
        write_error(
            stream,
            &ErrorField {
                code: "08P01",
                message: "parse message has too many parameter type oids",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if is_copy_statement(&query) && !is_supported_extended_copy(&query) {
        write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "COPY is not supported by the compatibility endpoint",
                position: None,
            },
        )?;
        return Ok(true);
    }
    let parsed_cursor_query = parse_declare_cursor(&query).map(|(_, cursor_query)| cursor_query);
    let describe_query = parsed_cursor_query
        .as_ref()
        .cloned()
        .unwrap_or_else(|| query.clone());
    if let Some(error) = sql_execute_parameter_type_mapping_error(
        session,
        &describe_query,
        parameter_type_oids.as_slice(),
    ) {
        write_error(stream, &error)?;
        return Ok(true);
    }
    if let Some(error) = sql_execute_describe_error(session, &describe_query) {
        write_error(stream, &error)?;
        return Ok(true);
    }
    if parsed_cursor_query.is_none() && is_unsupported_declare_cursor_statement(&query) {
        write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message:
                    "cursor declaration options are not supported by the compatibility endpoint",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if describe_query_columns(session, &describe_query).is_none()
        && describe_extended_query_columns(session, &describe_query).is_none()
        && describe_extended_query_columns(session, &query).is_none()
        && !is_supported_extended_dml(session, &describe_query)
        && !is_supported_extended_copy(&describe_query)
    {
        write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "extended query protocol only supports relational SELECT",
                position: None,
            },
        )?;
        return Ok(true);
    }
    let parameter_type_oids =
        resolve_prepared_parameter_type_oids(session, &describe_query, parameter_type_oids);
    session.replace_extended_statement(
        statement_name,
        PreparedQuery {
            query,
            parameter_type_oids,
        },
    );
    write_parse_complete(stream)?;
    Ok(false)
}

pub(super) fn handle_bind(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    portal_name: String,
    statement_name: String,
    parameter_format_codes: Vec<i16>,
    parameters: Vec<Option<Vec<u8>>>,
    result_format_codes: Vec<i16>,
) -> io::Result<bool> {
    let Some(PreparedStatement::Extended(query)) = session.prepared.get(&statement_name) else {
        write_error(
            stream,
            &ErrorField {
                code: "26000",
                message: "prepared statement does not exist",
                position: None,
            },
        )?;
        return Ok(true);
    };
    let mut query = query.clone();
    query.parameter_type_oids =
        resolve_prepared_parameter_type_oids(session, &query.query, query.parameter_type_oids);
    if !portal_name.is_empty() && session.portals.contains_key(&portal_name) {
        write_error(
            stream,
            &ErrorField {
                code: "42P03",
                message: "portal already exists",
                position: None,
            },
        )?;
        return Ok(true);
    }
    let parameter_format_count = parameter_format_codes.len();
    let expected_parameter_count = expected_parameter_count(&query);
    if parameters.len() != expected_parameter_count {
        write_error(
            stream,
            &ErrorField {
                code: "08P01",
                message: "bind message has wrong number of parameters",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if parameters.iter().any(Option::is_none) {
        write_error(
            stream,
            &bind_parameter_error_field(BindParameterError::NullUnsupported),
        )?;
        return Ok(true);
    }
    if !format_code_count_is_valid(parameter_format_count, expected_parameter_count) {
        write_error(
            stream,
            &ErrorField {
                code: "08P01",
                message: "bind message has wrong number of parameter format codes",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if let Some(result_column_count) = extended_query_result_column_count(session, &query.query) {
        let result_format_count = result_format_codes.len();
        if !format_code_count_is_valid(result_format_count, result_column_count) {
            write_error(
                stream,
                &ErrorField {
                    code: "08P01",
                    message: "bind message has wrong number of result format codes",
                    position: None,
                },
            )?;
            return Ok(true);
        }
    }
    if let Some(error) = binary_result_format_error(session, &query.query, &result_format_codes) {
        write_error(stream, &error)?;
        return Ok(true);
    }
    let mut decoded = Vec::with_capacity(parameters.len());
    for (idx, parameter) in parameters.into_iter().enumerate() {
        let format_code = format_code_at(&parameter_format_codes, idx);
        let type_oid = query.parameter_type_oids.get(idx).copied().unwrap_or(0);
        decoded.push(match parameter {
            Some(bytes) => match decode_bind_parameter(&bytes, format_code, type_oid) {
                Ok(value) => Some(value),
                Err(error) => {
                    write_error(stream, &error)?;
                    return Ok(true);
                }
            },
            None => None,
        });
    }
    if let Err(error) = bind_query_parameters(&query, &decoded) {
        write_error(stream, &bind_parameter_error_field(error))?;
        return Ok(true);
    }
    session.replace_extended_portal(
        portal_name,
        Portal {
            statement_name,
            query: query.clone(),
            parameters: decoded,
            result_format_codes,
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    write_bind_complete(stream)?;
    Ok(false)
}

pub(super) fn handle_describe(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    target: DescribeTarget,
    name: &str,
) -> io::Result<bool> {
    match target {
        DescribeTarget::Statement => {
            let Some(PreparedStatement::Extended(query)) = session.prepared.get(name) else {
                write_error(
                    stream,
                    &ErrorField {
                        code: "26000",
                        message: "prepared statement does not exist",
                        position: None,
                    },
                )?;
                return Ok(true);
            };
            write_parameter_description(stream, &query.parameter_type_oids)?;
            if let Some(columns) = describe_extended_query_columns(session, &query.query) {
                write_row_description(stream, &columns)?;
                Ok(false)
            } else {
                write_no_data(stream)?;
                Ok(false)
            }
        }
        DescribeTarget::Portal => {
            let Some(portal) = session.portals.get(name) else {
                write_error(
                    stream,
                    &ErrorField {
                        code: "34000",
                        message: "portal does not exist",
                        position: None,
                    },
                )?;
                return Ok(true);
            };
            let bound_query = match bind_query_parameters(&portal.query, &portal.parameters) {
                Ok(query) => query,
                Err(error) => {
                    write_error(stream, &bind_parameter_error_field(error))?;
                    return Ok(true);
                }
            };
            if let Some(columns) = describe_extended_query_columns(session, &bound_query) {
                write_row_description_with_formats(stream, &columns, &portal.result_format_codes)?;
                if let Some(portal) = session.portals.get_mut(name) {
                    portal.described = true;
                }
                Ok(false)
            } else {
                write_no_data(stream)?;
                if let Some(portal) = session.portals.get_mut(name) {
                    portal.described = true;
                }
                Ok(false)
            }
        }
    }
}

pub(super) fn handle_execute(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    portal_name: &str,
    max_rows: u32,
) -> io::Result<bool> {
    let Some(portal) = session.portals.get(portal_name) else {
        write_error(
            stream,
            &ErrorField {
                code: "34000",
                message: "portal does not exist",
                position: None,
            },
        )?;
        return Ok(true);
    };
    let bound_query = match bind_query_parameters(&portal.query, &portal.parameters) {
        Ok(query) => query,
        Err(error) => {
            write_error(stream, &bind_parameter_error_field(error))?;
            return Ok(true);
        }
    };
    if let Some((name, query)) = parse_declare_cursor(&bound_query) {
        return execute_declare_cursor(stream, session, name, &query);
    }
    if let Some(copy) = parse_copy_to_stdout_table(&bound_query) {
        execute_copy_to_stdout(stream, session, &copy.table, copy.options)?;
        return Ok(false);
    }
    if let Some(copy) = parse_copy_from_stdin(&bound_query) {
        begin_copy_from_stdin(
            stream,
            session,
            &copy.table,
            copy.columns,
            copy.options,
            false,
        )?;
        return Ok(false);
    }
    if let Some((name, parameters)) = parse_sql_execute(&bound_query) {
        if session
            .portals
            .get(portal_name)
            .and_then(|portal| portal.result.as_ref())
            .is_none()
        {
            let result = match execute_sql_prepared_result(session, &name, &parameters) {
                Ok(result) => result,
                Err(error) => {
                    write_error(stream, &error)?;
                    return Ok(true);
                }
            };
            if let Some(portal) = session.portals.get_mut(portal_name) {
                portal.result = Some(result);
                portal.position = 0;
            }
        }
        execute_portal_batch(stream, session, portal_name, max_rows)?;
        return Ok(false);
    }
    match parse_command(&bound_query) {
        Ok(Command::Insert(insert)) => {
            let tag = match execute_extended_insert(session, insert) {
                Ok(tag) => tag,
                Err(error) => {
                    write_error(stream, &error)?;
                    return Ok(true);
                }
            };
            write_command_complete(stream, &tag)?;
            return Ok(false);
        }
        Ok(Command::Delete(delete)) => {
            let tag = match execute_extended_delete(session, delete) {
                Ok(tag) => tag,
                Err(error) => {
                    write_error(stream, &error)?;
                    return Ok(true);
                }
            };
            write_command_complete(stream, &tag)?;
            return Ok(false);
        }
        Ok(Command::Update(update)) => {
            let tag = match execute_extended_update(session, update) {
                Ok(tag) => tag,
                Err(error) => {
                    write_error(stream, &error)?;
                    return Ok(true);
                }
            };
            write_command_complete(stream, &tag)?;
            return Ok(false);
        }
        _ => {}
    }
    let select = match parse_command(&bound_query) {
        Ok(Command::Select(select)) => select,
        Err(ParseError::NegativeLimit) => {
            write_error(stream, &negative_limit_error_field())?;
            return Ok(true);
        }
        Err(ParseError::NegativeOffset) => {
            write_error(stream, &negative_offset_error_field())?;
            return Ok(true);
        }
        Ok(_) | Err(_) => {
            write_error(
                stream,
                &ErrorField {
                    code: "0A000",
                    message: "limited portal execution only supports relational SELECT",
                    position: None,
                },
            )?;
            return Ok(true);
        }
    };
    if session
        .portals
        .get(portal_name)
        .and_then(|portal| portal.result.as_ref())
        .is_none()
    {
        let result = match execute_select_result(session, &select) {
            Ok(result) => result,
            Err(error) => {
                write_error(stream, &error)?;
                return Ok(true);
            }
        };
        if let Some(portal) = session.portals.get_mut(portal_name) {
            portal.result = Some(result);
            portal.position = 0;
        }
    }
    execute_portal_batch(stream, session, portal_name, max_rows)?;
    Ok(false)
}

pub(super) fn handle_close(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    target: DescribeTarget,
    name: &str,
) -> io::Result<bool> {
    session.close_extended_target(target, name);
    write_close_complete(stream)?;
    Ok(false)
}

fn execute_portal_batch(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    portal_name: &str,
    max_rows: u32,
) -> io::Result<()> {
    let Some(portal) = session.portals.get_mut(portal_name) else {
        return write_error(
            stream,
            &ErrorField {
                code: "34000",
                message: "portal does not exist",
                position: None,
            },
        );
    };
    let Some(result) = portal.result.as_ref() else {
        return write_error(
            stream,
            &ErrorField {
                code: "34000",
                message: "portal does not exist",
                position: None,
            },
        );
    };

    let start = portal.position.min(result.rows.len());
    let requested_end = if max_rows == 0 {
        result.rows.len()
    } else {
        start
            .saturating_add(max_rows as usize)
            .min(result.rows.len())
    };
    let rows = result.rows[start..requested_end].to_vec();
    let columns = result.columns.clone();
    let result_format_codes = portal.result_format_codes.clone();
    let emitted_count = requested_end - start;
    portal.position = requested_end;

    if portal.position < result.rows.len() {
        for row in &rows {
            write_data_row_with_formats(stream, &columns, row, &result_format_codes)?;
        }
        write_portal_suspended(stream)
    } else {
        let completion_count = if portal.completed {
            emitted_count
        } else {
            portal.position
        };
        portal.completed = true;
        for row in &rows {
            write_data_row_with_formats(stream, &columns, row, &result_format_codes)?;
        }
        write_command_complete(stream, &format!("SELECT {completion_count}"))
    }
}

fn format_code_at(format_codes: &[i16], idx: usize) -> i16 {
    match format_codes {
        [] => 0,
        [code] => *code,
        codes => codes[idx],
    }
}

fn decode_bind_parameter(
    bytes: &[u8],
    format_code: i16,
    type_oid: u32,
) -> Result<String, ErrorField> {
    match format_code {
        0 => String::from_utf8(bytes.to_vec()).map_err(|_| ErrorField {
            code: "22021",
            message: "invalid byte sequence for encoding \"UTF8\"",
            position: None,
        }),
        1 => match type_oid {
            23 => {
                let raw: [u8; 4] = bytes.try_into().map_err(|_| ErrorField {
                    code: "22P03",
                    message: "invalid binary representation for int4 parameter",
                    position: None,
                })?;
                Ok(i32::from_be_bytes(raw).to_string())
            }
            25 => String::from_utf8(bytes.to_vec()).map_err(|_| ErrorField {
                code: "22021",
                message: "invalid byte sequence for encoding \"UTF8\"",
                position: None,
            }),
            _ => Err(ErrorField {
                code: "0A000",
                message: "binary parameters are only supported for int4 and text",
                position: None,
            }),
        },
        _ => Err(ErrorField {
            code: "0A000",
            message: "unsupported bind format code",
            position: None,
        }),
    }
}

fn binary_result_format_error(
    session: &Session,
    query: &str,
    result_format_codes: &[i16],
) -> Option<ErrorField> {
    if !result_format_codes.contains(&1) {
        return None;
    }
    let columns = describe_extended_query_columns(session, query)?;
    for (idx, column) in columns.iter().enumerate() {
        if format_code_at(result_format_codes, idx) == 1 && !matches!(column.oid, 23 | 25) {
            return Some(ErrorField {
                code: "0A000",
                message: "binary results are only supported for int4 and text columns",
                position: None,
            });
        }
    }
    None
}

#[cfg(test)]
pub(super) fn test_execute_portal_batch(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    portal_name: &str,
    max_rows: u32,
) -> io::Result<()> {
    execute_portal_batch(stream, session, portal_name, max_rows)
}
