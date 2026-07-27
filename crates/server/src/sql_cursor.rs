//! Connection-local SQL cursor syntax for PostgreSQL compatibility clients.
//!
//! Cursor declaration executes the retained SELECT through the facade immediately. A declaration
//! inside a transaction uses its pinned engine snapshot; the idle compatibility form retains only
//! the already-materialized result. This module owns syntax only, never relational execution.

use gpu_db_facade::{DbError, ErrorCategory};
use gpu_db_protocol::{canonicalize_sql_for_exact_match, sql_may_start_with_any_keyword};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlCursorAction {
    Declare { name: String, query: String },
    Fetch { name: String, count: Option<usize> },
    Move { name: String, count: Option<usize> },
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
    if !compat_classifier_gate(statement, &["DECLARE", "FETCH", "MOVE", "CLOSE"]) {
        return Ok(None);
    }
    let canonical = canonicalize_sql_for_exact_match(statement)
        .map_err(|error| syntax_error(error.to_string()))?;
    let statement = canonical.trim().trim_end_matches(';').trim();
    if let Some(rest) = strip_keyword(statement, "DECLARE") {
        return parse_declare(rest).map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "FETCH") {
        return parse_cursor_movement(rest, CursorMovement::Fetch).map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "MOVE") {
        return parse_cursor_movement(rest, CursorMovement::Move).map(Some);
    }
    if let Some(rest) = strip_keyword(statement, "CLOSE") {
        return parse_close(rest).map(|target| Some(SqlCursorAction::Close(target)));
    }
    Ok(None)
}

fn compat_classifier_gate(statement: &str, candidates: &[&str]) -> bool {
    let admitted = sql_may_start_with_any_keyword(statement, candidates);
    #[cfg(feature = "probe-timing")]
    crate::insert_probe::record_compat_classifier_gate(statement.len() as u64, admitted);
    admitted
}

fn parse_declare(rest: &str) -> Result<SqlCursorAction, DbError> {
    let declaration = rest;
    let (name, mut rest) = take_name(rest)?;
    if let Some(after_sensitivity) =
        strip_keyword(rest, "ASENSITIVE").or_else(|| strip_keyword(rest, "INSENSITIVE"))
    {
        rest = after_sensitivity;
    }
    if let Some(after_no) = strip_keyword(rest, "NO") {
        rest =
            strip_keyword(after_no, "SCROLL").ok_or_else(|| declare_option_error(declaration))?;
    }
    rest = strip_keyword(rest, "CURSOR").ok_or_else(|| declare_option_error(declaration))?;
    if let Some(after_without) = strip_keyword(rest, "WITHOUT") {
        rest = strip_keyword(after_without, "HOLD")
            .ok_or_else(|| declare_option_error(declaration))?;
    } else if strip_keyword(rest, "WITH").is_some() {
        return Err(declare_option_error(declaration));
    }
    let query = strip_keyword(rest, "FOR")
        .ok_or_else(|| syntax_error("DECLARE CURSOR requires FOR and a query"))?
        .trim();
    if query.is_empty() {
        return Err(syntax_error("DECLARE CURSOR query is empty"));
    }
    if strip_keyword(query, "SELECT").is_none() && strip_keyword(query, "EXECUTE").is_none() {
        return Err(unsupported_error(
            "cursor declarations only support relational SELECT or SQL EXECUTE",
        ));
    }
    Ok(SqlCursorAction::Declare {
        name,
        query: query.to_string(),
    })
}

#[derive(Clone, Copy)]
enum CursorMovement {
    Fetch,
    Move,
}

fn parse_cursor_movement(rest: &str, movement: CursorMovement) -> Result<SqlCursorAction, DbError> {
    let (name, count) =
        parse_forward_cursor_target(rest).ok_or_else(|| movement.unsupported_error())?;
    Ok(match movement {
        CursorMovement::Fetch => SqlCursorAction::Fetch { name, count },
        CursorMovement::Move => SqlCursorAction::Move { name, count },
    })
}

impl CursorMovement {
    fn unsupported_error(self) -> DbError {
        let operation = match self {
            Self::Fetch => "fetch",
            Self::Move => "move",
        };
        unsupported_error(format!(
            "cursor {operation} direction is not supported by the compatibility endpoint"
        ))
    }
}

fn parse_forward_cursor_target(rest: &str) -> Option<(String, Option<usize>)> {
    let rest = rest.trim_start();
    if let Some(target) = strip_cursor_marker(rest) {
        return finish_cursor_target(target, Some(1));
    }
    let (first, after_first) = take_word(rest)?;
    match first {
        "next" => finish_cursor_target(after_first, Some(1)),
        "all" => finish_cursor_target(after_first, None),
        "forward" => parse_forward_target(after_first),
        "backward" | "prior" | "first" | "last" | "absolute" | "relative" => None,
        token if token.starts_with('-') => None,
        token => match token.parse::<usize>() {
            Ok(count) => finish_cursor_target(after_first, Some(count)),
            Err(_) => finish_cursor_target(rest, Some(1)),
        },
    }
}

fn parse_forward_target(rest: &str) -> Option<(String, Option<usize>)> {
    let rest = rest.trim_start();
    if let Some(target) = strip_cursor_marker(rest) {
        return finish_cursor_target(target, Some(1));
    }
    let (token, after_token) = take_word(rest)?;
    if token == "all" {
        finish_cursor_target(after_token, None)
    } else if token.starts_with('-') {
        None
    } else if let Ok(count) = token.parse::<usize>() {
        finish_cursor_target(after_token, Some(count))
    } else {
        finish_cursor_target(rest, Some(1))
    }
}

fn finish_cursor_target(rest: &str, count: Option<usize>) -> Option<(String, Option<usize>)> {
    let target = strip_cursor_marker(rest).unwrap_or(rest);
    let (name, trailing) = take_name(target).ok()?;
    trailing.trim().is_empty().then_some((name, count))
}

fn strip_cursor_marker(input: &str) -> Option<&str> {
    strip_keyword(input, "FROM").or_else(|| strip_keyword(input, "IN"))
}

fn take_word(input: &str) -> Option<(&str, &str)> {
    let input = input.trim_start();
    let end = input.find(char::is_whitespace).unwrap_or(input.len());
    (end > 0).then_some((&input[..end], &input[end..]))
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

fn declare_option_error(declaration: &str) -> DbError {
    if declaration
        .split_whitespace()
        .any(|token| token.eq_ignore_ascii_case("cursor"))
    {
        unsupported_error(
            "cursor declaration options are not supported by the compatibility endpoint",
        )
    } else {
        syntax_error("DECLARE requires CURSOR")
    }
}

fn syntax_error(message: impl Into<String>) -> DbError {
    DbError {
        category: ErrorCategory::Syntax,
        message: message.into(),
    }
}

fn unsupported_error(message: impl Into<String>) -> DbError {
    DbError {
        category: ErrorCategory::Unsupported,
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
                query: "select id from only public.accounts".to_string(),
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
            classify_sql_cursor_statement("MOVE FORWARD ALL _pg_dump_cursor").unwrap(),
            Some(SqlCursorAction::Move {
                name: "_pg_dump_cursor".to_string(),
                count: None,
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
    fn accepts_supported_declaration_options_and_rejects_the_rest() {
        for statement in [
            "DECLARE c NO SCROLL CURSOR FOR SELECT 1",
            "DECLARE c CURSOR WITHOUT HOLD FOR SELECT 1",
            "DECLARE c ASENSITIVE NO SCROLL CURSOR WITHOUT HOLD FOR SELECT 1",
            "DECLARE c INSENSITIVE CURSOR FOR SELECT 1",
        ] {
            assert!(matches!(
                classify_sql_cursor_statement(statement).unwrap(),
                Some(SqlCursorAction::Declare { name, .. }) if name == "c"
            ));
        }
        for statement in [
            "DECLARE c BINARY CURSOR FOR SELECT 1",
            "DECLARE c SCROLL CURSOR FOR SELECT 1",
            "DECLARE c CURSOR WITH HOLD FOR SELECT 1",
            "DECLARE c BINARY INSENSITIVE CURSOR FOR SELECT 1",
        ] {
            let error = classify_sql_cursor_statement(statement).unwrap_err();
            assert_eq!(error.category, ErrorCategory::Unsupported);
            assert_eq!(
                error.message,
                "cursor declaration options are not supported by the compatibility endpoint"
            );
        }
        let error =
            classify_sql_cursor_statement("DECLARE c CURSOR FOR UPDATE accounts SET id = 2")
                .unwrap_err();
        assert_eq!(error.category, ErrorCategory::Unsupported);
        assert_eq!(
            error.message,
            "cursor declarations only support relational SELECT or SQL EXECUTE"
        );
    }

    #[test]
    fn parses_forward_movement_variants_and_rejects_nonforward_directions() {
        for (statement, expected) in [
            ("FETCH c", Some(1)),
            ("FETCH FORWARD c", Some(1)),
            ("FETCH NEXT IN c", Some(1)),
            ("FETCH 0 FROM c", Some(0)),
            ("FETCH FORWARD 2 c", Some(2)),
            ("FETCH ALL c", None),
            ("FETCH FORWARD ALL FROM c", None),
        ] {
            assert_eq!(
                classify_sql_cursor_statement(statement).unwrap(),
                Some(SqlCursorAction::Fetch {
                    name: "c".to_string(),
                    count: expected,
                })
            );
        }
        for statement in [
            "FETCH BACKWARD 1 FROM c",
            "FETCH PRIOR c",
            "FETCH -1 FROM c",
            "MOVE ABSOLUTE 2 FROM c",
            "MOVE -1 c",
        ] {
            let error = classify_sql_cursor_statement(statement).unwrap_err();
            assert_eq!(error.category, ErrorCategory::Unsupported);
            assert!(error.message.contains("direction is not supported"));
        }
    }

    #[test]
    fn comments_and_quoted_cursor_names_are_lexically_preserved() {
        assert_eq!(
            classify_sql_cursor_statement(
                "DECLARE /* name */ \"Escaped \"\" Cursor\" /* option */ ASENSITIVE \
                 NO SCROLL CURSOR WITHOUT HOLD FOR SELECT 'MiXeD' AS value"
            )
            .unwrap(),
            Some(SqlCursorAction::Declare {
                name: "Escaped \" Cursor".to_string(),
                query: "select 'MiXeD' as value".to_string(),
            })
        );
        assert_eq!(
            classify_sql_cursor_statement(
                "MOVE /* count */ 1 /* marker */ FROM /* target */ \"Escaped \"\" Cursor\""
            )
            .unwrap(),
            Some(SqlCursorAction::Move {
                name: "Escaped \" Cursor".to_string(),
                count: Some(1),
            })
        );
        assert_eq!(
            classify_sql_cursor_statement("CLOSE \"Mixed Cursor\"").unwrap(),
            Some(SqlCursorAction::Close(SqlCursorCloseTarget::Named(
                "Mixed Cursor".to_string()
            )))
        );
    }

    #[test]
    fn lexical_gate_preserves_every_cursor_action_after_leading_comments_and_case_fold() {
        for statement in [
            "/* gate */ dEcLaRe c CURSOR FOR SELECT 1",
            "/* gate */ fEtCh c",
            "/* gate */ mOvE ALL c",
            "/* gate */ cLoSe c",
            "DECLARE\u{2003}\u{202f}unicode_space CURSOR FOR SELECT 1",
        ] {
            assert!(classify_sql_cursor_statement(statement).unwrap().is_some());
        }
        assert_eq!(classify_sql_cursor_statement("DECLAREfoo c").unwrap(), None);
        let non_ascii =
            classify_sql_cursor_statement("DECLAREé c CURSOR FOR SELECT 1").unwrap_err();
        assert_eq!(non_ascii.message, "invalid cursor name");
    }
}
