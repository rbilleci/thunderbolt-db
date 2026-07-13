// Legacy cursor parsing and execution ownership. This is not a product execution path.

use super::sql_execute_syntax::{
    parse_sql_execute, parse_supported_cursor_name, sql_keyword_rest_start, strip_sql_comments,
};
use super::sql_prepare::execute_sql_prepared_result;
use super::{
    canonical_sql, execute_select_result, max_placeholder_index, negative_limit_error_field,
    negative_offset_error_field, write_command_complete, write_error, write_rows_with_tag,
    CloseCursorTarget, Cursor, ErrorField, ReadWrite, Session,
};
use gpu_db_protocol::{parse_command, Command, ParseError};
use std::io;

pub(super) fn parse_declare_cursor(statement: &str) -> Option<(String, String)> {
    let stripped = strip_sql_comments(statement);
    let trimmed = stripped.trim().trim_end_matches(';').trim();
    let rest_start = sql_keyword_rest_start(trimmed, "declare")?;
    let rest = trimmed[rest_start..].trim_start();
    let (name, rest) = split_cursor_name_and_rest(rest)?;
    let (token, rest) = take_sql_word(rest)?;
    let rest = match token.as_str() {
        "asensitive" | "insensitive" => rest,
        "no" => {
            let (scroll, rest) = take_sql_word(rest)?;
            if scroll != "scroll" {
                return None;
            }
            rest
        }
        "cursor" => rest,
        _ => return None,
    };
    let (token, rest) = if token == "cursor" {
        (token, rest)
    } else {
        take_sql_word(rest)?
    };
    let rest = if token == "no" {
        let (scroll, rest) = take_sql_word(rest)?;
        if scroll != "scroll" {
            return None;
        }
        let (cursor, rest) = take_sql_word(rest)?;
        if cursor != "cursor" {
            return None;
        }
        rest
    } else if token == "cursor" {
        rest
    } else {
        return None;
    };
    let (token, rest) = take_sql_word(rest)?;
    let rest = if token == "without" {
        let (hold, rest) = take_sql_word(rest)?;
        if hold != "hold" {
            return None;
        }
        let (for_token, rest) = take_sql_word(rest)?;
        if for_token != "for" {
            return None;
        }
        rest
    } else if token == "for" {
        rest
    } else {
        return None;
    };
    let query = canonical_sql(rest);
    if query.is_empty() {
        None
    } else {
        Some((name, query))
    }
}

pub(super) fn is_unsupported_declare_cursor_statement(statement: &str) -> bool {
    let stripped = strip_sql_comments(statement);
    let trimmed = stripped.trim().trim_end_matches(';').trim();
    let Some(rest_start) = sql_keyword_rest_start(trimmed, "declare") else {
        return false;
    };
    trimmed[rest_start..]
        .split_whitespace()
        .any(|token| token.eq_ignore_ascii_case("cursor"))
}

fn split_cursor_name_and_rest(target: &str) -> Option<(String, &str)> {
    let target = target.trim_start();
    if let Some(quoted) = target.strip_prefix('"') {
        let mut chars = quoted.char_indices().peekable();
        while let Some((idx, ch)) = chars.next() {
            if ch == '"' {
                if chars.peek().is_some_and(|(_, next)| *next == '"') {
                    chars.next();
                    continue;
                }
                let name_end = idx + 2;
                let name = parse_supported_cursor_name(&target[..name_end])?;
                return Some((name, &target[name_end..]));
            }
        }
        None
    } else {
        let name_end = target.find(char::is_whitespace)?;
        let name = parse_supported_cursor_name(&target[..name_end])?;
        Some((name, &target[name_end..]))
    }
}

fn take_sql_word(input: &str) -> Option<(String, &str)> {
    let input = input.trim_start();
    let end = input.find(char::is_whitespace).unwrap_or(input.len());
    if end == 0 {
        return None;
    }
    Some((input[..end].to_ascii_lowercase(), &input[end..]))
}

pub(super) fn parse_fetch_forward(statement: &str) -> Option<(String, Option<usize>)> {
    let stripped = strip_sql_comments(statement);
    let trimmed = stripped.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    let rest = lower.strip_prefix("fetch ")?;
    let (name_start_in_rest, count) = parse_forward_cursor_target(rest)?;
    let name_start = "fetch ".len() + name_start_in_rest;
    let name = trimmed[name_start..].trim();
    parse_supported_cursor_name(name).map(|name| (name, count))
}

pub(super) fn parse_move_forward(statement: &str) -> Option<(String, Option<usize>)> {
    let stripped = strip_sql_comments(statement);
    let trimmed = stripped.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    let rest = lower.strip_prefix("move ")?;
    let (name_start_in_rest, count) = parse_forward_cursor_target(rest)?;
    let name_start = "move ".len() + name_start_in_rest;
    let name = trimmed[name_start..].trim();
    parse_supported_cursor_name(name).map(|name| (name, count))
}

fn parse_forward_cursor_target(rest: &str) -> Option<(usize, Option<usize>)> {
    if let Some((cursor_marker_idx, cursor_marker_len, count)) =
        parse_forward_cursor_direction_with_marker(rest)
    {
        return Some((cursor_marker_idx + cursor_marker_len, count));
    }
    parse_forward_cursor_direction_without_marker(rest)
}

fn parse_forward_cursor_direction_with_marker(rest: &str) -> Option<(usize, usize, Option<usize>)> {
    let (cursor_marker_idx, cursor_marker_len) = rest
        .strip_prefix("from ")
        .map(|_| (0, "from ".len()))
        .or_else(|| rest.strip_prefix("in ").map(|_| (0, "in ".len())))
        .or_else(|| rest.find(" from ").map(|idx| (idx, " from ".len())))
        .or_else(|| rest.find(" in ").map(|idx| (idx, " in ".len())))?;
    let direction = rest[..cursor_marker_idx].trim();
    let count = match direction {
        "" | "next" | "forward" => Some(1),
        "all" | "forward all" => None,
        _ => {
            if let Some(count_token) = direction.strip_prefix("forward ") {
                Some(count_token.trim().parse::<usize>().ok()?)
            } else {
                Some(direction.parse::<usize>().ok()?)
            }
        }
    };
    Some((cursor_marker_idx, cursor_marker_len, count))
}

fn parse_forward_cursor_direction_without_marker(rest: &str) -> Option<(usize, Option<usize>)> {
    let skipped = rest.len() - rest.trim_start().len();
    let rest = rest.trim_start();
    if rest.is_empty() {
        return None;
    }
    let first_token_end = rest.find(char::is_whitespace);
    let first_token = first_token_end.map_or(rest, |idx| &rest[..idx]);
    let remainder_start = first_token_end.map(|idx| idx + 1);
    let remainder = remainder_start
        .map(|idx| rest[idx..].trim_start())
        .unwrap_or("");

    match first_token {
        "next" => {
            if remainder.is_empty() {
                None
            } else {
                Some((skipped + rest.len() - remainder.len(), Some(1)))
            }
        }
        "all" => {
            if remainder.is_empty() {
                None
            } else {
                Some((skipped + rest.len() - remainder.len(), None))
            }
        }
        "forward" => {
            if remainder.is_empty() {
                return None;
            }
            let second_token_end = remainder.find(char::is_whitespace);
            let second_token = second_token_end.map_or(remainder, |idx| &remainder[..idx]);
            if second_token == "all" {
                let name = second_token_end
                    .map(|idx| remainder[idx + 1..].trim_start())
                    .unwrap_or("");
                if name.is_empty() {
                    None
                } else {
                    Some((skipped + rest.len() - name.len(), None))
                }
            } else if let Ok(count) = second_token.parse::<usize>() {
                let name = second_token_end
                    .map(|idx| remainder[idx + 1..].trim_start())
                    .unwrap_or("");
                if name.is_empty() {
                    None
                } else {
                    Some((skipped + rest.len() - name.len(), Some(count)))
                }
            } else {
                Some((skipped + rest.len() - remainder.len(), Some(1)))
            }
        }
        "backward" | "prior" | "first" | "last" | "absolute" | "relative" => None,
        _ => {
            if let Ok(count) = first_token.parse::<usize>() {
                if remainder.is_empty() {
                    None
                } else {
                    Some((skipped + rest.len() - remainder.len(), Some(count)))
                }
            } else {
                Some((skipped, Some(1)))
            }
        }
    }
}

pub(super) fn is_unsupported_fetch_cursor_statement(statement: &str) -> bool {
    let stripped = strip_sql_comments(statement);
    let trimmed = stripped.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("fetch ") && parse_fetch_forward(trimmed).is_none()
}

pub(super) fn is_unsupported_move_cursor_statement(statement: &str) -> bool {
    let stripped = strip_sql_comments(statement);
    let trimmed = stripped.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("move ") && parse_move_forward(trimmed).is_none()
}

pub(super) fn parse_close_cursor(statement: &str) -> Option<CloseCursorTarget> {
    let stripped = strip_sql_comments(statement);
    let trimmed = stripped.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    let name = lower.strip_prefix("close ")?;
    if name.trim() == "all" {
        return Some(CloseCursorTarget::All);
    }
    let original = &trimmed["close ".len()..];
    if original.trim().is_empty() {
        None
    } else {
        parse_supported_cursor_name(original.trim()).map(CloseCursorTarget::Named)
    }
}

pub(super) fn execute_declare_cursor(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    name: String,
    query: &str,
) -> io::Result<bool> {
    if session.cursors.contains_key(&name) {
        write_error(
            stream,
            &ErrorField {
                code: "42P03",
                message: "cursor already exists",
                position: None,
            },
        )?;
        return Ok(true);
    }
    if max_placeholder_index(query) > 0 {
        write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "parameterized cursor declarations are not supported",
                position: None,
            },
        )?;
        return Ok(true);
    }
    let result = if let Some((prepared_name, parameters)) = parse_sql_execute(query) {
        match execute_sql_prepared_result(session, &prepared_name, &parameters) {
            Ok(result) => result,
            Err(error) => {
                write_error(stream, &error)?;
                return Ok(true);
            }
        }
    } else {
        let select = match parse_command(query) {
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
                        message:
                            "cursor declarations only support relational SELECT or SQL EXECUTE",
                        position: None,
                    },
                )?;
                return Ok(true);
            }
        };
        match execute_select_result(session, &select) {
            Ok(result) => result,
            Err(error) => {
                write_error(stream, &error)?;
                return Ok(true);
            }
        }
    };
    session.cursors.insert(
        name,
        Cursor {
            columns: result.columns,
            rows: result.rows,
            position: 0,
        },
    );
    write_command_complete(stream, "DECLARE CURSOR")?;
    Ok(false)
}

pub(super) fn execute_fetch_forward(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    name: &str,
    count: Option<usize>,
) -> io::Result<()> {
    let Some(cursor) = session.cursors.get_mut(name) else {
        return write_error(
            stream,
            &ErrorField {
                code: "34000",
                message: "cursor does not exist",
                position: None,
            },
        );
    };
    let start = cursor.position;
    let end = count
        .map(|count| start.saturating_add(count).min(cursor.rows.len()))
        .unwrap_or(cursor.rows.len());
    cursor.position = end;
    write_rows_with_tag(
        stream,
        &cursor.columns,
        &cursor.rows[start..end],
        true,
        &format!("FETCH {}", end - start),
    )
}

pub(super) fn execute_move_forward(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    name: &str,
    count: Option<usize>,
) -> io::Result<()> {
    let Some(cursor) = session.cursors.get_mut(name) else {
        return write_error(
            stream,
            &ErrorField {
                code: "34000",
                message: "cursor does not exist",
                position: None,
            },
        );
    };
    let start = cursor.position;
    let end = count
        .map(|count| start.saturating_add(count).min(cursor.rows.len()))
        .unwrap_or(cursor.rows.len());
    cursor.position = end;
    write_command_complete(stream, &format!("MOVE {}", end - start))
}
