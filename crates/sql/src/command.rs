//! Top-level SQL command and session-control parsing.

use super::{
    normalize_identifier, normalize_relation_identifier, parse_relational_command, split_csv,
    strip_keyword_prefix_case_insensitive, Command, ParseError, SetRoleScope,
    TransactionAccessMode, TransactionCharacteristics, TransactionIsolation,
};
use crate::parameter::{
    dollar_quote_delimiter, is_escape_string_prefix, is_identifier_continuation_byte,
};

/// Split a PostgreSQL simple-query message at top-level semicolons.
///
/// Semicolons inside strings, quoted identifiers, dollar-quoted bodies, line comments, or nested
/// block comments are never treated as boundaries. PostgreSQL treats comments as whitespace, so
/// leading/trailing comments are excluded from each returned statement and comment-only segments
/// are omitted; comments between SQL tokens remain in the slice. An unterminated block comment is
/// retained as executable text so command parsing reports a syntax error instead of silently
/// accepting it as trivia. Callers emit one EmptyQueryResponse only when the whole message has no
/// statement.
pub fn split_simple_query(query: &str) -> Vec<&str> {
    let mut statements = Vec::new();
    let mut statement_code_start = None;
    let mut statement_code_end = 0usize;
    let mut index = 0usize;
    let mut in_single_quote = false;
    let mut single_quote_backslash_escapes = false;
    let mut in_double_quote = false;
    let mut in_line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut block_comment_start = None;
    let mut dollar_delimiter: Option<&str> = None;

    while index < query.len() {
        if let Some(delimiter) = dollar_delimiter {
            if query[index..].starts_with(delimiter) {
                index += delimiter.len();
                dollar_delimiter = None;
            } else {
                index += query[index..]
                    .chars()
                    .next()
                    .expect("index is inside query")
                    .len_utf8();
            }
            statement_code_end = index;
            continue;
        }
        let bytes = query.as_bytes();
        if in_line_comment {
            if matches!(bytes[index], b'\r' | b'\n') {
                in_line_comment = false;
            }
            index += 1;
            continue;
        }
        if block_comment_depth > 0 {
            if bytes[index..].starts_with(b"/*") {
                block_comment_depth += 1;
                index += 2;
            } else if bytes[index..].starts_with(b"*/") {
                block_comment_depth -= 1;
                index += 2;
                if block_comment_depth == 0 {
                    block_comment_start = None;
                }
            } else {
                index += query[index..]
                    .chars()
                    .next()
                    .expect("index is inside query")
                    .len_utf8();
            }
            continue;
        }
        if in_single_quote {
            if bytes[index] == b'\\'
                && single_quote_backslash_escapes
                && bytes.get(index + 1).is_some()
            {
                index += 1;
                index += query[index..]
                    .chars()
                    .next()
                    .expect("backslash has a following character")
                    .len_utf8();
            } else if bytes[index] == b'\'' {
                if bytes.get(index + 1) == Some(&b'\'') {
                    index += 2;
                } else {
                    in_single_quote = false;
                    single_quote_backslash_escapes = false;
                    index += 1;
                }
            } else {
                index += query[index..]
                    .chars()
                    .next()
                    .expect("index is inside query")
                    .len_utf8();
            }
            statement_code_end = index;
            continue;
        }
        if in_double_quote {
            if bytes[index] == b'"' {
                if bytes.get(index + 1) == Some(&b'"') {
                    index += 2;
                } else {
                    in_double_quote = false;
                    index += 1;
                }
            } else {
                index += query[index..]
                    .chars()
                    .next()
                    .expect("index is inside query")
                    .len_utf8();
            }
            statement_code_end = index;
            continue;
        }

        if bytes[index..].starts_with(b"--") {
            in_line_comment = true;
            index += 2;
        } else if bytes[index..].starts_with(b"/*") {
            block_comment_depth = 1;
            block_comment_start = Some(index);
            index += 2;
        } else if bytes[index] == b'\'' {
            statement_code_start.get_or_insert(index);
            in_single_quote = true;
            single_quote_backslash_escapes = is_escape_string_prefix(query, index);
            index += 1;
            statement_code_end = index;
        } else if bytes[index] == b'"' {
            statement_code_start.get_or_insert(index);
            in_double_quote = true;
            index += 1;
            statement_code_end = index;
        } else if bytes[index] == b'$'
            && !(index > 0 && is_identifier_continuation_byte(bytes[index - 1]))
        {
            if let Some((delimiter, after_start)) = dollar_quote_delimiter(query, index) {
                statement_code_start.get_or_insert(index);
                dollar_delimiter = Some(delimiter);
                index = after_start;
                statement_code_end = index;
            } else {
                statement_code_start.get_or_insert(index);
                index += 1;
                statement_code_end = index;
            }
        } else if bytes[index] == b';' {
            if let Some(start) = statement_code_start.take() {
                statements.push(&query[start..statement_code_end]);
            }
            index += 1;
            statement_code_end = index;
        } else {
            let start = index;
            let ch_len = query[index..]
                .chars()
                .next()
                .expect("index is inside query")
                .len_utf8();
            index += ch_len;
            if !query[start..index]
                .chars()
                .next()
                .expect("one character was consumed")
                .is_whitespace()
            {
                statement_code_start.get_or_insert(start);
                statement_code_end = index;
            }
        }
    }

    if block_comment_depth > 0 {
        let invalid_start = block_comment_start.expect("an open block comment records its start");
        statement_code_start.get_or_insert(invalid_start);
        statement_code_end = query.len();
    }
    if let Some(start) = statement_code_start {
        statements.push(&query[start..statement_code_end]);
    }
    statements
}

fn parse_transaction_chain_suffix(tokens: &[&str]) -> Option<bool> {
    if tokens.is_empty() {
        return Some(false);
    }

    if matches!(
        tokens,
        [and, chain]
            if and.eq_ignore_ascii_case("AND") && chain.eq_ignore_ascii_case("CHAIN")
    ) {
        return Some(true);
    }

    if matches!(
        tokens,
        [and, no, chain]
            if and.eq_ignore_ascii_case("AND")
                && no.eq_ignore_ascii_case("NO")
                && chain.eq_ignore_ascii_case("CHAIN")
    ) {
        return Some(false);
    }

    None
}

fn parse_transaction_control_chain(input: &str, keyword: &str) -> Option<bool> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, mut rest) = tokens.split_first()?;
    if !first.eq_ignore_ascii_case(keyword) {
        return None;
    }

    if let Some((scope, tail)) = rest.split_first() {
        if scope.eq_ignore_ascii_case("TRANSACTION") || scope.eq_ignore_ascii_case("WORK") {
            rest = tail;
        }
    }

    parse_transaction_chain_suffix(rest)
}

fn parse_flush_command(input: &str) -> Option<Command> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, rest) = tokens.split_first()?;
    if first.eq_ignore_ascii_case("CHECKPOINT") {
        return match rest {
            [] => Some(Command::Flush),
            _ => None,
        };
    }

    if !first.eq_ignore_ascii_case("FLUSH") {
        return None;
    }

    match rest {
        [] => Some(Command::Flush),
        [target] if target.eq_ignore_ascii_case("WAL") || target.eq_ignore_ascii_case("LOG") => {
            Some(Command::Flush)
        }
        [write_ahead]
            if write_ahead.eq_ignore_ascii_case("WRITE-AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITEAHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD_LOG")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD_WAL") =>
        {
            Some(Command::Flush)
        }
        [write_ahead, target]
            if (write_ahead.eq_ignore_ascii_case("WRITE-AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITEAHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD"))
                && (target.eq_ignore_ascii_case("LOG") || target.eq_ignore_ascii_case("WAL")) =>
        {
            Some(Command::Flush)
        }
        [write, ahead]
            if write.eq_ignore_ascii_case("WRITE") && ahead.eq_ignore_ascii_case("AHEAD") =>
        {
            Some(Command::Flush)
        }
        [write, ahead, target]
            if write.eq_ignore_ascii_case("WRITE")
                && ahead.eq_ignore_ascii_case("AHEAD")
                && (target.eq_ignore_ascii_case("LOG") || target.eq_ignore_ascii_case("WAL")) =>
        {
            Some(Command::Flush)
        }
        _ => None,
    }
}

fn parse_reset_command(input: &str) -> Option<Result<Command, ParseError>> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, rest) = tokens.split_first()?;

    if first.eq_ignore_ascii_case("RESET") {
        return Some(match rest {
            [target] if target.eq_ignore_ascii_case("ALL") => Ok(Command::ResetAll),
            [target]
                if target.eq_ignore_ascii_case("ROLE")
                    || target.eq_ignore_ascii_case("AUTHORIZATION")
                    || target.eq_ignore_ascii_case("AUTH") =>
            {
                Ok(Command::SetRole {
                    role: None,
                    scope: SetRoleScope::Session,
                })
            }
            [scope, role]
                if (scope.eq_ignore_ascii_case("SESSION")
                    || scope.eq_ignore_ascii_case("LOCAL"))
                    && role.eq_ignore_ascii_case("ROLE") =>
            {
                Ok(Command::SetRole {
                    role: None,
                    scope: if scope.eq_ignore_ascii_case("LOCAL") {
                        SetRoleScope::Local
                    } else {
                        SetRoleScope::Session
                    },
                })
            }
            [session, authorization]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH")) =>
            {
                Ok(Command::SetRole {
                    role: None,
                    scope: SetRoleScope::Session,
                })
            }
            [session, authorization, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && default.eq_ignore_ascii_case("DEFAULT") =>
            {
                Ok(Command::SetRole {
                    role: None,
                    scope: SetRoleScope::Session,
                })
            }
            [session, authorization, to, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && to.eq_ignore_ascii_case("TO")
                    && default.eq_ignore_ascii_case("DEFAULT") =>
            {
                Ok(Command::SetRole {
                    role: None,
                    scope: SetRoleScope::Session,
                })
            }
            _ => Err(ParseError::InvalidReset),
        });
    }

    if first.eq_ignore_ascii_case("DISCARD") {
        return Some(match rest {
            [target]
                if target.eq_ignore_ascii_case("ALL")
                    || target.eq_ignore_ascii_case("TEMP")
                    || target.eq_ignore_ascii_case("TEMPORARY")
                    || target.eq_ignore_ascii_case("PLANS")
                    || target.eq_ignore_ascii_case("SEQUENCES") =>
            {
                Ok(Command::ResetAll)
            }
            [scope, kind]
                if (scope.eq_ignore_ascii_case("TEMP")
                    || scope.eq_ignore_ascii_case("TEMPORARY"))
                    && (kind.eq_ignore_ascii_case("TABLE")
                        || kind.eq_ignore_ascii_case("TABLES")) =>
            {
                Ok(Command::ResetAll)
            }
            _ => Err(ParseError::InvalidReset),
        });
    }

    if first.eq_ignore_ascii_case("DEALLOCATE") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "DEALLOCATE") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim_start();
        if rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }

        if let Some(after_prepared) = strip_keyword_prefix_case_insensitive(rest, "PREPARED") {
            let tail = after_prepared.trim_start();
            return Some(
                if parse_reset_identifier(tail).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidReset)
                },
            );
        }

        if let Some(after_prepare) = strip_keyword_prefix_case_insensitive(rest, "PREPARE") {
            let tail = after_prepare.trim_start();
            return Some(
                if parse_reset_identifier(tail).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidReset)
                },
            );
        }

        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("CLOSE") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "CLOSE") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim_start();
        if rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }
        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("UNLISTEN") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "UNLISTEN") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim();
        if rest.is_empty() || rest == "*" || rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }
        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("LISTEN") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "LISTEN") else {
            return Some(Err(ParseError::InvalidReset));
        };
        return Some(
            if parse_reset_identifier(rest.trim()).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("NOTIFY") {
        let Some(raw_rest) = strip_keyword_prefix_case_insensitive(input, "NOTIFY") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let Some((_, tail_after_channel)) = parse_reset_identifier(raw_rest.trim_start()) else {
            return Some(Err(ParseError::InvalidReset));
        };
        let tail_after_channel = tail_after_channel.trim_start();
        if tail_after_channel.is_empty() {
            return Some(Ok(Command::ResetAll));
        }
        let Some(payload) = tail_after_channel.strip_prefix(',') else {
            return Some(Err(ParseError::InvalidReset));
        };
        let payload = payload.trim_start();
        return Some(
            if !payload.starts_with(',') && notify_payload_fragment_is_non_empty(payload) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    None
}

fn parse_reset_identifier(input: &str) -> Option<(&str, &str)> {
    let s = input.trim_start();
    if s.is_empty() {
        return None;
    }

    if s.starts_with('"') {
        let bytes = s.as_bytes();
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                if i == 1 {
                    return None;
                }
                let end = i + 1;
                return Some((&s[..end], &s[end..]));
            }
            i += 1;
        }
        return None;
    }

    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return None;
    }

    let mut end = first.len_utf8();
    for (idx, ch) in chars {
        if ch.is_whitespace() || ch == ',' {
            end = idx;
            break;
        }
        if !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()) {
            return None;
        }
        end = idx + ch.len_utf8();
    }

    Some((&s[..end], &s[end..]))
}

fn notify_payload_fragment_is_non_empty(fragment: &str) -> bool {
    let trimmed = fragment.trim();
    if trimmed.is_empty() || trimmed.trim_matches(',').trim().is_empty() {
        return false;
    }

    let mut in_single_quote = false;
    let mut single_quote_backslash_escapes = false;
    let mut in_double_quote = false;
    let chars: Vec<char> = trimmed.chars().collect();
    let mut idx = 0;

    while idx < chars.len() {
        let ch = chars[idx];
        if in_single_quote {
            if single_quote_backslash_escapes && ch == '\\' && idx + 1 < chars.len() {
                idx += 2;
                continue;
            }

            if ch == '\'' {
                if idx + 1 < chars.len() && chars[idx + 1] == '\'' {
                    idx += 2;
                    continue;
                }
                in_single_quote = false;
                single_quote_backslash_escapes = false;
            }
            idx += 1;
            continue;
        }

        if in_double_quote {
            if ch == '"' {
                if idx + 1 < chars.len() && chars[idx + 1] == '"' {
                    idx += 2;
                    continue;
                }
                in_double_quote = false;
            }
            idx += 1;
            continue;
        }

        if ch == '$' {
            if let Some((delim, after_start)) = parse_notify_dollar_quote_start(&chars, idx) {
                let mut scan = after_start;
                let mut found = false;
                while scan + delim.len() <= chars.len() {
                    if chars[scan..scan + delim.len()] == delim[..] {
                        idx = scan + delim.len();
                        found = true;
                        break;
                    }
                    scan += 1;
                }
                if !found {
                    return false;
                }
                continue;
            }
        }

        if ch.is_whitespace() {
            if chars[idx + 1..].iter().any(|c| !c.is_whitespace()) {
                return false;
            }
            break;
        }

        if ch == '\'' {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 1;
            continue;
        }

        if matches!(ch, 'e' | 'E') && chars.get(idx + 1) == Some(&'\'') {
            in_single_quote = true;
            single_quote_backslash_escapes = true;
            idx += 2;
            continue;
        }

        if matches!(ch, 'b' | 'B' | 'x' | 'X') && chars.get(idx + 1) == Some(&'\'') {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 2;
            continue;
        }

        if matches!(ch, 'u' | 'U')
            && chars.get(idx + 1) == Some(&'&')
            && chars.get(idx + 2) == Some(&'\'')
        {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 3;
            continue;
        }

        match ch {
            '"' => in_double_quote = true,
            ',' => return false,
            _ => {}
        }
        idx += 1;
    }

    !(in_single_quote || in_double_quote)
}

fn parse_notify_dollar_quote_start(chars: &[char], start: usize) -> Option<(Vec<char>, usize)> {
    if chars.get(start) != Some(&'$') {
        return None;
    }
    let mut idx = start + 1;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch == '$' {
            return Some((chars[start..=idx].to_vec(), idx + 1));
        }
        if !(ch.is_ascii_alphanumeric() || ch == '_') {
            return None;
        }
        idx += 1;
    }
    None
}

fn split_set_key_value(rest: &str) -> Option<(&str, &str)> {
    if let Some((k, v)) = rest.split_once('=') {
        return Some((k, v));
    }

    let trimmed = rest.trim();
    let (key, tail) = trimmed.split_once(char::is_whitespace)?;
    let tail = tail.trim_start();
    if tail.len() < 2 {
        return None;
    }

    let (keyword, remainder) = tail.split_at(2);
    if !keyword.eq_ignore_ascii_case("TO") {
        return None;
    }

    if remainder.is_empty() || !remainder.starts_with(char::is_whitespace) {
        return None;
    }

    Some((key, remainder.trim_start()))
}

fn strip_set_scope_prefix<'a>(input: &'a str, scope: &str) -> Option<&'a str> {
    let trimmed = input.trim_start();
    let after_scope = strip_keyword_prefix_case_insensitive(trimmed, scope)?;
    if after_scope.trim().is_empty() {
        return None;
    }
    Some(after_scope.trim_start())
}

fn parse_set_session_command(rest: &str) -> Option<Result<Command, ParseError>> {
    if let Some(after_local) = strip_keyword_prefix_case_insensitive(rest, "LOCAL") {
        let after_local = after_local.trim_start();
        if let Some(after_role) = strip_keyword_prefix_case_insensitive(after_local, "ROLE") {
            let tail = after_role.trim_start();
            return Some(parse_set_role_command(tail, SetRoleScope::Local));
        }

        if let Some(after_transaction) =
            strip_keyword_prefix_case_insensitive(after_local, "TRANSACTION")
        {
            let tail = after_transaction.trim_start();
            return Some(
                parse_begin_mode_list(&normalize_begin_tokens(tail))
                    .map(|transaction| Command::SessionControl {
                        transaction: Some(transaction),
                        access_share_relations: Vec::new(),
                    })
                    .ok_or(ParseError::InvalidSet),
            );
        }
    }

    if let Some(after_session) = strip_keyword_prefix_case_insensitive(rest, "SESSION") {
        let after_session = after_session.trim_start();
        if let Some(after_role) = strip_keyword_prefix_case_insensitive(after_session, "ROLE") {
            let tail = after_role.trim_start();
            return Some(parse_set_role_command(tail, SetRoleScope::Session));
        }
    }

    if let Some(after_role) = strip_keyword_prefix_case_insensitive(rest, "ROLE") {
        let tail = after_role.trim_start();
        return Some(parse_set_role_command(tail, SetRoleScope::Session));
    }

    if let Some(after_transaction) = strip_keyword_prefix_case_insensitive(rest, "TRANSACTION") {
        let tail = after_transaction.trim_start();
        return Some(
            parse_begin_mode_list(&normalize_begin_tokens(tail))
                .map(|transaction| Command::SessionControl {
                    transaction: Some(transaction),
                    access_share_relations: Vec::new(),
                })
                .ok_or(ParseError::InvalidSet),
        );
    }

    if let Some(after_session) = strip_keyword_prefix_case_insensitive(rest, "SESSION") {
        let after_session = after_session.trim_start();

        if let Some(after_authorization) =
            strip_keyword_prefix_case_insensitive(after_session, "AUTHORIZATION")
        {
            let tail = after_authorization.trim_start();
            return Some(
                if parse_reset_identifier(tail)
                    .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }

        if let Some(after_auth) = strip_keyword_prefix_case_insensitive(after_session, "AUTH") {
            let tail = after_auth.trim_start();
            return Some(
                if parse_reset_identifier(tail)
                    .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }

        if let Some(after_characteristics) =
            strip_keyword_prefix_case_insensitive(after_session, "CHARACTERISTICS")
        {
            let after_characteristics = after_characteristics.trim_start();
            if let Some(after_as) =
                strip_keyword_prefix_case_insensitive(after_characteristics, "AS")
            {
                let after_as = after_as.trim_start();
                if let Some(after_transaction) =
                    strip_keyword_prefix_case_insensitive(after_as, "TRANSACTION")
                {
                    let tail = after_transaction.trim_start();
                    return Some(
                        parse_begin_mode_list(&normalize_begin_tokens(tail))
                            .map(|transaction| Command::SessionControl {
                                transaction: Some(transaction),
                                access_share_relations: Vec::new(),
                            })
                            .ok_or(ParseError::InvalidSet),
                    );
                }
            }
            return Some(Err(ParseError::InvalidSet));
        }

        return None;
    }

    None
}

fn parse_set_role_command(tail: &str, scope: SetRoleScope) -> Result<Command, ParseError> {
    if tail.eq_ignore_ascii_case("NONE") || tail.eq_ignore_ascii_case("DEFAULT") {
        return Ok(Command::SetRole { role: None, scope });
    }
    if let Some((role, trailing)) = parse_reset_identifier(tail) {
        if trailing.trim().is_empty() {
            return Ok(Command::SetRole {
                role: Some(normalize_identifier(role)?),
                scope,
            });
        }
    }
    Err(ParseError::InvalidSet)
}

fn is_deferrable_suffix(tokens: &[&str]) -> bool {
    matches!(
        tokens,
        [deferrable] if deferrable.eq_ignore_ascii_case("DEFERRABLE")
    ) || matches!(
        tokens,
        [not, deferrable]
            if not.eq_ignore_ascii_case("NOT") && deferrable.eq_ignore_ascii_case("DEFERRABLE")
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BeginMode {
    Access(TransactionAccessMode),
    Isolation(TransactionIsolation),
    Deferrable(bool),
}

fn parse_begin_mode(tokens: &[String]) -> Option<(usize, BeginMode)> {
    let refs: Vec<_> = tokens.iter().map(String::as_str).collect();

    if refs.len() >= 4
        && matches!(
            &refs[..4],
            [isolation, level, repeatable, read]
                if isolation.eq_ignore_ascii_case("ISOLATION")
                    && level.eq_ignore_ascii_case("LEVEL")
                    && repeatable.eq_ignore_ascii_case("REPEATABLE")
                    && read.eq_ignore_ascii_case("READ")
        )
    {
        return Some((
            4,
            BeginMode::Isolation(TransactionIsolation::RepeatableRead),
        ));
    }

    if refs.len() >= 3 {
        let isolation = match &refs[..3] {
            [isolation, level, serializable]
                if isolation.eq_ignore_ascii_case("ISOLATION")
                    && level.eq_ignore_ascii_case("LEVEL")
                    && serializable.eq_ignore_ascii_case("SERIALIZABLE") =>
            {
                Some(TransactionIsolation::Serializable)
            }
            _ => None,
        };
        if let Some(isolation) = isolation {
            return Some((3, BeginMode::Isolation(isolation)));
        }
    }

    if refs.len() >= 4 {
        let isolation = match &refs[..4] {
            [isolation, level, read, committed]
                if isolation.eq_ignore_ascii_case("ISOLATION")
                    && level.eq_ignore_ascii_case("LEVEL")
                    && read.eq_ignore_ascii_case("READ")
                    && committed.eq_ignore_ascii_case("COMMITTED") =>
            {
                Some(TransactionIsolation::ReadCommitted)
            }
            [isolation, level, read, uncommitted]
                if isolation.eq_ignore_ascii_case("ISOLATION")
                    && level.eq_ignore_ascii_case("LEVEL")
                    && read.eq_ignore_ascii_case("READ")
                    && uncommitted.eq_ignore_ascii_case("UNCOMMITTED") =>
            {
                Some(TransactionIsolation::ReadUncommitted)
            }
            _ => None,
        };
        if let Some(isolation) = isolation {
            return Some((4, BeginMode::Isolation(isolation)));
        }
    }

    if refs.len() >= 2 {
        let two = &refs[..2];
        if matches!(
            two,
            [read, only]
                if read.eq_ignore_ascii_case("READ") && only.eq_ignore_ascii_case("ONLY")
        ) || matches!(
            two,
            [read, write]
                if read.eq_ignore_ascii_case("READ") && write.eq_ignore_ascii_case("WRITE")
        ) {
            return Some((
                2,
                BeginMode::Access(if two[1].eq_ignore_ascii_case("ONLY") {
                    TransactionAccessMode::ReadOnly
                } else {
                    TransactionAccessMode::ReadWrite
                }),
            ));
        }

        if matches!(
            two,
            [not, deferrable]
                if not.eq_ignore_ascii_case("NOT") && deferrable.eq_ignore_ascii_case("DEFERRABLE")
        ) {
            return Some((2, BeginMode::Deferrable(false)));
        }
    }

    if !refs.is_empty() && is_deferrable_suffix(&refs[..1]) {
        return Some((1, BeginMode::Deferrable(true)));
    }

    None
}

fn normalize_begin_tokens(input: &str) -> Vec<String> {
    let mut normalized = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        if ch == ',' {
            normalized.push(' ');
            normalized.push(',');
            normalized.push(' ');
        } else {
            normalized.push(ch);
        }
    }
    normalized.split_whitespace().map(str::to_owned).collect()
}

fn parse_begin_mode_list(tokens: &[String]) -> Option<TransactionCharacteristics> {
    if tokens.is_empty() {
        return None;
    }

    let mut idx = 0;
    let mut characteristics = TransactionCharacteristics::default();
    let mut seen_access_mode = false;
    let mut seen_isolation_level = false;
    let mut seen_deferrable = false;

    while idx < tokens.len() {
        if tokens[idx] == "," {
            return None;
        }

        let (consumed, mode) = parse_begin_mode(&tokens[idx..])?;

        match mode {
            BeginMode::Access(_) if seen_access_mode => return None,
            BeginMode::Isolation(_) if seen_isolation_level => return None,
            BeginMode::Deferrable(_) if seen_deferrable => return None,
            BeginMode::Access(access) => {
                seen_access_mode = true;
                characteristics.access = access;
            }
            BeginMode::Isolation(isolation) => {
                seen_isolation_level = true;
                characteristics.isolation = isolation;
            }
            BeginMode::Deferrable(deferrable) => {
                seen_deferrable = true;
                characteristics.deferrable = deferrable;
            }
        }

        idx += consumed;
        if idx == tokens.len() {
            return Some(characteristics);
        }

        if tokens[idx] == "," {
            idx += 1;
            if idx == tokens.len() || tokens[idx] == "," {
                return None;
            }
        }
    }

    Some(characteristics)
}

fn strip_single_leading_comma(tokens: &[String]) -> Option<&[String]> {
    match tokens {
        [first, rest @ ..] if first == "," && !rest.is_empty() => Some(rest),
        _ => None,
    }
}

fn parse_begin_characteristics(input: &str) -> Option<TransactionCharacteristics> {
    let tokens = normalize_begin_tokens(input);
    let (first, rest) = tokens.split_first()?;

    if first.eq_ignore_ascii_case("BEGIN") {
        return match rest {
            [] => Some(TransactionCharacteristics::default()),
            [second] if second.eq_ignore_ascii_case("TRANSACTION") => {
                Some(TransactionCharacteristics::default())
            }
            [second] if second.eq_ignore_ascii_case("WORK") => {
                Some(TransactionCharacteristics::default())
            }
            [second, mode @ ..]
                if second.eq_ignore_ascii_case("TRANSACTION")
                    || second.eq_ignore_ascii_case("WORK") =>
            {
                parse_begin_mode_list(mode)
                    .or_else(|| strip_single_leading_comma(mode).and_then(parse_begin_mode_list))
            }
            mode => parse_begin_mode_list(mode),
        };
    }

    if first.eq_ignore_ascii_case("START") {
        return match rest {
            [second] if second.eq_ignore_ascii_case("TRANSACTION") => {
                Some(TransactionCharacteristics::default())
            }
            [second] if second.eq_ignore_ascii_case("WORK") => {
                Some(TransactionCharacteristics::default())
            }
            [second, mode @ ..]
                if second.eq_ignore_ascii_case("TRANSACTION")
                    || second.eq_ignore_ascii_case("WORK") =>
            {
                parse_begin_mode_list(mode)
                    .or_else(|| strip_single_leading_comma(mode).and_then(parse_begin_mode_list))
            }
            _ => None,
        };
    }

    None
}

/// Parse a single SQL command (the strict, public entry). A SELECT whose FROM target is a
/// `pg_catalog.`/`information_schema.`-qualified relation is rejected, exactly as before —
/// the legacy compatibility server relies on this (catalog queries fail to parse and route
/// to its compatibility layer). The engine uses [`parse_command_allowing_catalog`] instead.
pub fn parse_command(input: &str) -> Result<Command, ParseError> {
    let command = parse_command_inner(input, false)?;
    // Let the command grammar own malformed quote/syntax diagnostics first. The lexical scan is
    // a defensive backstop for parameter references in positions that a successfully parsed AST
    // does not retain; AST-owned value slots are checked immediately below as well.
    if crate::parameter::sql_parameter_arity(input)? > 0 {
        return Err(ParseError::InvalidParameterReference);
    }
    if crate::prepared::command_parameter_count(&command) > 0 {
        return Err(ParseError::InvalidParameterReference);
    }
    Ok(command)
}

/// Like [`parse_command`] but carries a `pg_catalog.`/`information_schema.` qualifier on a
/// SELECT's FROM target through to the parsed `Select`, for the engine's native catalog
/// support (Phase-3 M2). Behaves identically to [`parse_command`] for every other command.
pub fn parse_command_allowing_catalog(input: &str) -> Result<Command, ParseError> {
    let command = parse_command_inner(input, true)?;
    if crate::parameter::sql_parameter_arity(input)? > 0 {
        return Err(ParseError::InvalidParameterReference);
    }
    if crate::prepared::command_parameter_count(&command) > 0 {
        return Err(ParseError::InvalidParameterReference);
    }
    Ok(command)
}

/// Return whether one statement begins with the `SELECT` keyword after PostgreSQL whitespace and
/// comments. This is only a read-only dispatch classifier: the real PostgreSQL parser still owns
/// validation of the richer statement before execution.
pub fn is_select_statement(input: &str) -> bool {
    let Ok(normalized) = crate::parameter::normalize_sql_comments(input) else {
        return false;
    };
    let input = normalized.trim_start();
    input
        .get(.."select".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("select"))
        && input["select".len()..]
            .chars()
            .next()
            .is_none_or(|next| !next.is_ascii_alphanumeric() && next != '_' && next != '$')
}

pub(crate) fn parse_prepared_command_allowing_catalog(input: &str) -> Result<Command, ParseError> {
    parse_command_inner(input, true)
}

fn parse_command_inner(input: &str, allow_catalog_schemas: bool) -> Result<Command, ParseError> {
    let normalized = crate::parameter::normalize_sql_comments(input)?;
    let input = normalized.as_ref();
    let mut s = input.trim_end();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    while let Some(without_semicolon) = s.strip_suffix(';') {
        s = without_semicolon.trim_end();
    }

    let s = s.trim_start();

    if is_pg16_dump_materialized_view_dependency_program(s) {
        return Ok(Command::PreparedCatalog(
            super::PreparedCatalogProgram::Pg16MaterializedViewDependencies,
        ));
    }
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    if let Some(characteristics) = parse_begin_characteristics(s) {
        return Ok(Command::Begin { characteristics });
    }
    if let Some(chain) = parse_transaction_control_chain(s, "COMMIT")
        .or_else(|| parse_transaction_control_chain(s, "END"))
    {
        return Ok(Command::Commit { chain });
    }
    if let Some(chain) = parse_transaction_control_chain(s, "ROLLBACK")
        .or_else(|| parse_transaction_control_chain(s, "ABORT"))
    {
        return Ok(Command::Rollback { chain });
    }
    if let Some(flush) = parse_flush_command(s) {
        return Ok(flush);
    }
    if let Some(reset) = parse_reset_command(s) {
        return reset;
    }
    let show_tokens = s.split_whitespace().collect::<Vec<_>>();
    if (show_tokens.len() == 4
        && show_tokens[0].eq_ignore_ascii_case("SHOW")
        && show_tokens[1].eq_ignore_ascii_case("TRANSACTION")
        && show_tokens[2].eq_ignore_ascii_case("ISOLATION")
        && show_tokens[3].eq_ignore_ascii_case("LEVEL"))
        || (show_tokens.len() == 2
            && show_tokens[0].eq_ignore_ascii_case("SHOW")
            && show_tokens[1].eq_ignore_ascii_case("transaction_isolation"))
    {
        return Ok(Command::ShowTransactionIsolation);
    }
    if show_tokens.len() == 2
        && show_tokens[0].eq_ignore_ascii_case("SHOW")
        && show_tokens[1].eq_ignore_ascii_case("client_encoding")
    {
        return Ok(Command::SelectLiteral(crate::SelectLiteral {
            column_name: "client_encoding".to_string(),
            ty: crate::SqlType::Text,
            value: crate::SqlValue::Text("UTF8".to_string()),
            add_int4: None,
        }));
    }
    if let Some(relational) = parse_relational_command(s, allow_catalog_schemas) {
        return relational;
    }

    if let Some(relations) = parse_access_share_lock(s)? {
        return Ok(Command::SessionControl {
            transaction: None,
            access_share_relations: relations,
        });
    }

    let mut parts = s.splitn(2, char::is_whitespace);
    if let Some(cmd) = parts.next() {
        if cmd.eq_ignore_ascii_case("SET") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidSet);
            };
            if let Some(alias) = parse_set_session_command(rest) {
                return alias;
            }

            let assignment_rest = strip_set_scope_prefix(rest, "LOCAL")
                .or_else(|| strip_set_scope_prefix(rest, "SESSION"))
                .unwrap_or(rest);
            let Some((k, v)) = split_set_key_value(assignment_rest) else {
                return Err(ParseError::InvalidSet);
            };
            let key = k.trim();
            let value = v.trim();
            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidSet);
            }
            if is_bounded_postgres_session_parameter(key, value) {
                return Ok(Command::SessionControl {
                    transaction: None,
                    access_share_relations: Vec::new(),
                });
            }
            return Ok(Command::SetKv {
                key: key.to_string(),
                value: value.to_string(),
            });
        }

        if cmd.eq_ignore_ascii_case("DEL") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidDel);
            };
            let key = rest.trim();
            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidDel);
            }
            return Ok(Command::DeleteKv {
                key: key.to_string(),
            });
        }

        if cmd.eq_ignore_ascii_case("DELETE") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidDel);
            };
            let rest = rest.trim();
            let key = if let Some((prefix, remainder)) = rest.split_once(char::is_whitespace) {
                if prefix.eq_ignore_ascii_case("FROM") {
                    let candidate = remainder.trim();
                    if candidate.is_empty() || candidate.chars().any(char::is_whitespace) {
                        return Err(ParseError::InvalidDel);
                    }
                    candidate
                } else {
                    return Err(ParseError::InvalidDel);
                }
            } else if rest.eq_ignore_ascii_case("FROM") {
                return Err(ParseError::InvalidDel);
            } else {
                rest
            };

            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidDel);
            }

            return Ok(Command::DeleteKv {
                key: key.to_string(),
            });
        }

        if cmd.eq_ignore_ascii_case("GET") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidGet);
            };
            let key = rest.trim();
            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidGet);
            }
            return Ok(Command::GetKv {
                key: key.to_string(),
            });
        }
    }

    Err(ParseError::Unsupported(s.to_string()))
}

fn is_pg16_dump_materialized_view_dependency_program(input: &str) -> bool {
    let Ok(canonical) = crate::canonicalize_sql_for_exact_match(input) else {
        return false;
    };
    canonical
        == "with recursive w as ( select d1.objid, d2.refobjid, c2.relkind as refrelkind from \
            pg_depend d1 join pg_class c1 on c1.oid = d1.objid and c1.relkind = 'm' join \
            pg_rewrite r1 on r1.ev_class = d1.objid join pg_depend d2 on d2.classid = \
            'pg_rewrite'::regclass and d2.objid = r1.oid and d2.refobjid <> d1.objid join \
            pg_class c2 on c2.oid = d2.refobjid and c2.relkind in ('m','v') where d1.classid = \
            'pg_class'::regclass union select w.objid, d3.refobjid, c3.relkind from w join \
            pg_rewrite r3 on r3.ev_class = w.refobjid join pg_depend d3 on d3.classid = \
            'pg_rewrite'::regclass and d3.objid = r3.oid and d3.refobjid <> w.refobjid join \
            pg_class c3 on c3.oid = d3.refobjid and c3.relkind in ('m','v') ) select \
            'pg_class'::regclass::oid as classid, objid, refobjid from w where refrelkind = 'm'"
}

fn parse_access_share_lock(input: &str) -> Result<Option<Vec<String>>, ParseError> {
    let Some(rest) = strip_keyword_prefix_case_insensitive(input, "LOCK TABLE") else {
        return Ok(None);
    };
    let lower = rest.to_ascii_lowercase();
    let suffix = " in access share mode";
    let Some(relation_list_len) = lower.strip_suffix(suffix).map(str::len) else {
        return Err(ParseError::Unsupported(input.to_string()));
    };
    let relation_list = rest[..relation_list_len].trim();
    if relation_list.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let relations = split_csv(relation_list)?
        .into_iter()
        .map(normalize_relation_identifier)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(relations))
}

fn is_bounded_postgres_session_parameter(key: &str, value: &str) -> bool {
    let key = key.to_ascii_lowercase();
    let value = value.trim().to_ascii_lowercase();
    matches!(
        (key.as_str(), value.as_str()),
        ("datestyle", "iso")
            | ("intervalstyle", "postgres")
            | ("extra_float_digits", "3")
            | ("statement_timeout", "0")
            | ("lock_timeout", "0")
            | ("idle_in_transaction_session_timeout", "0")
            | ("client_encoding", "'utf8'")
            | ("standard_conforming_strings", "on")
            | ("synchronize_seqscans", "off")
            | ("check_function_bodies", "false")
            | ("xmloption", "content")
            | ("client_min_messages", "warning")
            | ("row_security", "off")
            | ("default_tablespace", "''")
            | ("default_table_access_method", "heap")
            | ("default_transaction_read_only", "off")
            | ("search_path", "pg_catalog, public")
            | ("search_path", "public, pg_catalog")
    )
}

#[cfg(test)]
mod transaction_characteristic_tests {
    use super::*;

    #[test]
    fn malformed_read_committed_isolation_is_not_silently_accepted() {
        assert!(parse_command("BEGIN ISOLATION LEVEL COMMITTED").is_err());
        assert!(parse_command("START TRANSACTION ISOLATION LEVEL COMMITTED").is_err());
    }

    #[test]
    fn show_transaction_isolation_is_typed_session_metadata() {
        for sql in [
            "SHOW TRANSACTION ISOLATION LEVEL",
            "show transaction_isolation;",
            "SHOW /* session metadata */ TRANSACTION ISOLATION LEVEL",
        ] {
            assert_eq!(
                parse_command(sql).unwrap(),
                Command::ShowTransactionIsolation
            );
        }
        assert!(parse_command("SHOW TRANSACTION ISOLATION").is_err());
    }

    #[test]
    fn bounded_postgres_set_is_session_control_but_unknown_set_remains_kv() {
        for sql in [
            "SET statement_timeout = 0",
            "SET search_path = pg_catalog, public",
            "SET search_path = public, pg_catalog",
        ] {
            assert!(matches!(
                parse_command(sql).unwrap(),
                Command::SessionControl {
                    transaction: None,
                    ref access_share_relations,
                } if access_share_relations.is_empty()
            ));
        }
        assert!(matches!(
            parse_command("SET application_key = arbitrary_value").unwrap(),
            Command::SetKv { .. }
        ));
    }

    #[test]
    fn pg16_dump_role_login_program_is_bounded() {
        assert_eq!(
            parse_command(
                "ALTER ROLE global_reader WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB \
                 LOGIN NOREPLICATION NOBYPASSRLS"
            )
            .unwrap(),
            Command::AlterRoleLogin(crate::AlterRoleLogin {
                name: "global_reader".to_string(),
                login: true,
            })
        );
        assert!(matches!(
            parse_command("ALTER ROLE global_reader NOLOGIN").unwrap(),
            Command::AlterRoleLogin(crate::AlterRoleLogin { login: false, .. })
        ));
        for unsupported in [
            "ALTER ROLE global_reader WITH SUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS",
            "ALTER ROLE global_reader WITH NOSUPERUSER NOINHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS",
            "ALTER ROLE global_reader WITH CREATEDB",
            "ALTER ROLE global_reader WITH LOGIN PASSWORD 'secret'",
        ] {
            assert!(parse_command(unsupported).is_err(), "{unsupported}");
        }
    }

    #[test]
    fn simple_query_splitter_preserves_quoted_and_commented_semicolons() {
        assert_eq!(
            split_simple_query(
                "SELECT ';'; SELECT \"semi;colon\"; /* outer; /* inner; */ */ SELECT 3;"
            ),
            vec!["SELECT ';'", "SELECT \"semi;colon\"", "SELECT 3"]
        );
        assert_eq!(
            split_simple_query("CREATE FUNCTION f() AS $body$SELECT ';';$body$; SELECT 1"),
            vec!["CREATE FUNCTION f() AS $body$SELECT ';';$body$", "SELECT 1"]
        );
    }

    #[test]
    fn simple_query_splitter_matches_postgres_escape_identifier_and_cr_boundaries() {
        assert_eq!(
            split_simple_query(
                r"CREATE TABLE prefix (id int4); NOTIFY chan, E'x\'; COMMIT; y'; INSERT INTO missing VALUES (1)"
            ),
            vec![
                "CREATE TABLE prefix (id int4)",
                r"NOTIFY chan, E'x\'; COMMIT; y'",
                "INSERT INTO missing VALUES (1)"
            ]
        );
        assert_eq!(
            split_simple_query("SELECT foo$tag$; COMMIT; SELECT bar$tag$; SELECT 1"),
            vec!["SELECT foo$tag$", "COMMIT", "SELECT bar$tag$", "SELECT 1"]
        );
        assert_eq!(
            split_simple_query("SELECT 1 -- comment\r; COMMIT; SELECT 2"),
            vec!["SELECT 1", "COMMIT", "SELECT 2"]
        );
    }

    #[test]
    fn simple_query_splitter_treats_postgres_comments_as_boundary_trivia() {
        assert!(split_simple_query(" /* comment only */ -- and line only\r").is_empty());
        assert_eq!(
            split_simple_query(
                "-- leading line\r/* outer /* nested */ done */ BEGIN; \
                 COMMIT /* trailing block */; -- trailing line"
            ),
            vec!["BEGIN", "COMMIT"]
        );
        assert_eq!(
            split_simple_query("/* leading */ SELECT/* interior */ 1 -- trailing\n; /* tail */"),
            vec!["SELECT/* interior */ 1"]
        );
        assert_eq!(
            split_simple_query("CREATE TABLE comment_rollback (id int4); /* unterminated COMMIT;"),
            vec![
                "CREATE TABLE comment_rollback (id int4)",
                "/* unterminated COMMIT;"
            ]
        );
    }

    #[test]
    fn postgres_comments_are_token_separators_for_command_parsing() {
        assert!(matches!(
            parse_command("SELECT/* interior */ 1 AS one").unwrap(),
            Command::SelectLiteral(crate::SelectLiteral {
                column_name,
                ty: crate::SqlType::Int4,
                value: crate::SqlValue::Int4(1),
                ..
            }) if column_name == "one"
        ));
        assert!(matches!(
            parse_command("SELECT 1/* outer /* nested */ done */ AS one").unwrap(),
            Command::SelectLiteral(crate::SelectLiteral {
                column_name,
                ty: crate::SqlType::Int4,
                value: crate::SqlValue::Int4(1),
                ..
            }) if column_name == "one"
        ));
        assert!(matches!(
            parse_command("CREATE/* interior */ TABLE comment_spacing (id int4)").unwrap(),
            Command::CreateTable(_)
        ));
        let source = "CREATE/* retained WAL source */ TABLE comment_source (id int4)";
        let parsed = crate::ParsedCommand::parse(source).unwrap();
        assert_eq!(parsed.source(), source);
        assert!(parse_command("SELECT 1 /* unterminated").is_err());
    }

    #[test]
    fn select_classifier_is_comment_aware_and_keyword_bounded() {
        assert!(is_select_statement(
            " /* outer /* nested */ comment */ SELECT 1"
        ));
        assert!(is_select_statement("-- lead\r\nselect * from t"));
        assert!(!is_select_statement("SELEC broken"));
        assert!(!is_select_statement("selection FROM t"));
        assert!(!is_select_statement("COPY t FROM STDIN"));
    }

    #[test]
    fn pg16_materialized_dependency_program_is_exact_and_near_misses_fail_closed() {
        let canonical = "WITH RECURSIVE w AS ( \
            SELECT d1.objid, d2.refobjid, c2.relkind AS refrelkind \
            FROM pg_depend d1 \
            JOIN pg_class c1 ON c1.oid = d1.objid AND c1.relkind = 'm' \
            JOIN pg_rewrite r1 ON r1.ev_class = d1.objid \
            JOIN pg_depend d2 ON d2.classid = 'pg_rewrite'::regclass \
                AND d2.objid = r1.oid AND d2.refobjid <> d1.objid \
            JOIN pg_class c2 ON c2.oid = d2.refobjid AND c2.relkind IN ('m','v') \
            WHERE d1.classid = 'pg_class'::regclass \
            UNION \
            SELECT w.objid, d3.refobjid, c3.relkind \
            FROM w \
            JOIN pg_rewrite r3 ON r3.ev_class = w.refobjid \
            JOIN pg_depend d3 ON d3.classid = 'pg_rewrite'::regclass \
                AND d3.objid = r3.oid AND d3.refobjid <> w.refobjid \
            JOIN pg_class c3 ON c3.oid = d3.refobjid AND c3.relkind IN ('m','v') \
            ) \
            SELECT 'pg_class'::regclass::oid AS classid, objid, refobjid \
            FROM w WHERE refrelkind = 'm'";
        assert!(matches!(
            parse_command_allowing_catalog(canonical).unwrap(),
            Command::PreparedCatalog(
                crate::PreparedCatalogProgram::Pg16MaterializedViewDependencies
            )
        ));

        for near_miss in [
            canonical.replace("objid, refobjid", "objid, refobjid, refrelkind"),
            canonical.replace("WHERE refrelkind = 'm'", "WHERE refrelkind = 'v'"),
            canonical.replace("c1.relkind = 'm'", "c1.relkind = 'M'"),
            canonical.replace("UNION", "UNION ALL"),
            canonical.replace("FROM pg_depend d1", "FROM (pg_depend d1"),
        ] {
            assert!(
                parse_command_allowing_catalog(&near_miss).is_err(),
                "near-miss dependency CTE must not enter the fixed-result route: {near_miss}"
            );
        }
    }
}
