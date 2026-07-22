//! Connection-local SQL cursor syntax for PostgreSQL compatibility clients.
//!
//! Cursor declaration executes the retained SELECT through the facade immediately, so relational
//! authority and snapshot visibility remain on the transaction-pinned GPU route. This module owns
//! only the bounded wire-result window and forward position needed by `pg_dump --inserts`.

use gpu_db_facade::{DbError, ErrorCategory};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlCursorAction {
    Declare { name: String, query: String },
    Fetch { name: String, count: Option<usize> },
    Close(SqlCursorCloseTarget),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlCursorCloseTarget {
    All,
    Named(String),
}

pub(crate) fn classify_sql_cursor_statement(
    statement: &str,
) -> Result<Option<SqlCursorAction>, DbError> {
    let statement = statement.trim().trim_end_matches(';').trim();
    if let Some(rest) = strip_keyword(statement, "DECLARE") {
        return parse_declare(rest).map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "FETCH") {
        return parse_fetch(rest).map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "CLOSE") {
        return parse_close(rest).map(|target| Some(SqlCursorAction::Close(target)));
    }
    Ok(None)
}

fn parse_declare(rest: &str) -> Result<SqlCursorAction, DbError> {
    let (name, rest) = take_name(rest)?;
    let rest =
        strip_keyword(rest, "CURSOR").ok_or_else(|| syntax_error("DECLARE requires CURSOR"))?;
    let query = strip_keyword(rest, "FOR")
        .ok_or_else(|| syntax_error("DECLARE CURSOR requires FOR and a query"))?
        .trim();
    if query.is_empty() {
        return Err(syntax_error("DECLARE CURSOR query is empty"));
    }
    Ok(SqlCursorAction::Declare {
        name,
        query: query.to_string(),
    })
}

fn parse_fetch(rest: &str) -> Result<SqlCursorAction, DbError> {
    let mut rest = rest.trim_start();
    if let Some(after_forward) = strip_keyword(rest, "FORWARD") {
        rest = after_forward.trim_start();
    }
    let marker = keyword_position(rest, "FROM")
        .or_else(|| keyword_position(rest, "IN"))
        .ok_or_else(|| syntax_error("FETCH requires FROM and a cursor name"))?;
    let direction = rest[..marker].trim();
    let after_marker = &rest[marker..];
    let after_marker = strip_keyword(after_marker, "FROM")
        .or_else(|| strip_keyword(after_marker, "IN"))
        .ok_or_else(|| syntax_error("FETCH requires FROM and a cursor name"))?;
    let count = if direction.is_empty() || direction.eq_ignore_ascii_case("NEXT") {
        Some(1)
    } else if direction.eq_ignore_ascii_case("ALL") {
        None
    } else {
        Some(
            direction
                .parse::<usize>()
                .map_err(|_| syntax_error("only forward FETCH counts are supported"))?,
        )
    };
    let (name, trailing) = take_name(after_marker)?;
    if !trailing.trim().is_empty() {
        return Err(syntax_error("unexpected text after FETCH cursor name"));
    }
    Ok(SqlCursorAction::Fetch { name, count })
}

fn parse_close(rest: &str) -> Result<SqlCursorCloseTarget, DbError> {
    if rest.trim().eq_ignore_ascii_case("ALL") {
        return Ok(SqlCursorCloseTarget::All);
    }
    let (name, trailing) = take_name(rest)?;
    if !trailing.trim().is_empty() {
        return Err(syntax_error("unexpected text after CLOSE cursor name"));
    }
    Ok(SqlCursorCloseTarget::Named(name))
}

fn take_name(input: &str) -> Result<(String, &str), DbError> {
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
                    return Err(syntax_error("cursor name is empty"));
                }
                return Ok((name, &quoted[index + 1..]));
            }
            name.push(ch);
        }
        return Err(syntax_error("unterminated quoted cursor name"));
    }
    let end = input.find(char::is_whitespace).unwrap_or(input.len());
    let name = &input[..end];
    let mut chars = name.chars();
    let valid = chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric());
    if !valid {
        return Err(syntax_error("invalid cursor name"));
    }
    Ok((name.to_ascii_lowercase(), &input[end..]))
}

fn strip_keyword<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let input = input.trim_start();
    let prefix = input.get(..keyword.len())?;
    if !prefix.eq_ignore_ascii_case(keyword) {
        return None;
    }
    let rest = &input[keyword.len()..];
    rest.chars()
        .next()
        .is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '$')
        .then_some(rest)
}

fn keyword_position(input: &str, keyword: &str) -> Option<usize> {
    input.char_indices().find_map(|(index, _)| {
        let candidate = input.get(index..index + keyword.len())?;
        if !candidate.eq_ignore_ascii_case(keyword) {
            return None;
        }
        let before = input[..index].chars().next_back();
        let after = input[index + keyword.len()..].chars().next();
        let boundary = |ch: Option<char>| {
            ch.is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '$')
        };
        (boundary(before) && boundary(after)).then_some(index)
    })
}

fn syntax_error(message: impl Into<String>) -> DbError {
    DbError {
        category: ErrorCategory::Syntax,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pg_dump_forward_cursor_sequence() {
        assert_eq!(
            classify_sql_cursor_statement(
                "DECLARE _pg_dump_cursor CURSOR FOR SELECT id FROM ONLY public.accounts"
            )
            .unwrap(),
            Some(SqlCursorAction::Declare {
                name: "_pg_dump_cursor".to_string(),
                query: "SELECT id FROM ONLY public.accounts".to_string(),
            })
        );
        assert_eq!(
            classify_sql_cursor_statement("FETCH 100 FROM _pg_dump_cursor").unwrap(),
            Some(SqlCursorAction::Fetch {
                name: "_pg_dump_cursor".to_string(),
                count: Some(100),
            })
        );
        assert_eq!(
            classify_sql_cursor_statement("CLOSE _pg_dump_cursor").unwrap(),
            Some(SqlCursorAction::Close(SqlCursorCloseTarget::Named(
                "_pg_dump_cursor".to_string()
            )))
        );
    }

    #[test]
    fn rejects_backward_fetch_and_accepts_quoted_names() {
        assert!(classify_sql_cursor_statement("FETCH BACKWARD 1 FROM c").is_err());
        assert_eq!(
            classify_sql_cursor_statement("CLOSE \"Mixed Cursor\"").unwrap(),
            Some(SqlCursorAction::Close(SqlCursorCloseTarget::Named(
                "Mixed Cursor".to_string()
            )))
        );
    }
}
