//! Top-level SQL command and session-control parsing.

use super::{
    normalize_identifier, parse_relational_command, strip_keyword_prefix_case_insensitive, Command,
    ParseError,
};

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
            [target]
                if target.eq_ignore_ascii_case("ALL")
                    || target.eq_ignore_ascii_case("ROLE")
                    || target.eq_ignore_ascii_case("AUTHORIZATION")
                    || target.eq_ignore_ascii_case("AUTH") =>
            {
                Ok(Command::ResetAll)
            }
            [scope, role]
                if (scope.eq_ignore_ascii_case("SESSION")
                    || scope.eq_ignore_ascii_case("LOCAL"))
                    && role.eq_ignore_ascii_case("ROLE") =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH")) =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && default.eq_ignore_ascii_case("DEFAULT") =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization, to, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && to.eq_ignore_ascii_case("TO")
                    && default.eq_ignore_ascii_case("DEFAULT") =>
            {
                Ok(Command::ResetAll)
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
            return Some(parse_set_role_command(tail));
        }

        if let Some(after_transaction) =
            strip_keyword_prefix_case_insensitive(after_local, "TRANSACTION")
        {
            let tail = after_transaction.trim_start();
            return Some(
                if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }
    }

    if let Some(after_session) = strip_keyword_prefix_case_insensitive(rest, "SESSION") {
        let after_session = after_session.trim_start();
        if let Some(after_role) = strip_keyword_prefix_case_insensitive(after_session, "ROLE") {
            let tail = after_role.trim_start();
            return Some(parse_set_role_command(tail));
        }
    }

    if let Some(after_role) = strip_keyword_prefix_case_insensitive(rest, "ROLE") {
        let tail = after_role.trim_start();
        return Some(parse_set_role_command(tail));
    }

    if let Some(after_transaction) = strip_keyword_prefix_case_insensitive(rest, "TRANSACTION") {
        let tail = after_transaction.trim_start();
        return Some(
            if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidSet)
            },
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
                        if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                            Ok(Command::ResetAll)
                        } else {
                            Err(ParseError::InvalidSet)
                        },
                    );
                }
            }
            return Some(Err(ParseError::InvalidSet));
        }

        return None;
    }

    None
}

fn parse_set_role_command(tail: &str) -> Result<Command, ParseError> {
    if tail.eq_ignore_ascii_case("NONE") || tail.eq_ignore_ascii_case("DEFAULT") {
        return Ok(Command::SetRole { role: None });
    }
    if let Some((role, trailing)) = parse_reset_identifier(tail) {
        if trailing.trim().is_empty() {
            return Ok(Command::SetRole {
                role: Some(normalize_identifier(role)?),
            });
        }
    }
    Err(ParseError::InvalidSet)
}

fn is_isolation_level_suffix(tokens: &[&str]) -> bool {
    matches!(
        tokens,
        [isolation, level, serializable]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && serializable.eq_ignore_ascii_case("SERIALIZABLE")
    ) || matches!(
        tokens,
        [isolation, level, repeatable, read]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && repeatable.eq_ignore_ascii_case("REPEATABLE")
                && read.eq_ignore_ascii_case("READ")
    ) || matches!(
        tokens,
        [isolation, level, read, committed]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && read.eq_ignore_ascii_case("READ")
                && committed.eq_ignore_ascii_case("COMMITTED")
    ) || matches!(
        tokens,
        [isolation, level, read, uncommitted]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && read.eq_ignore_ascii_case("READ")
                && uncommitted.eq_ignore_ascii_case("UNCOMMITTED")
    )
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
enum BeginModeKind {
    AccessMode,
    IsolationLevel,
    Deferrable,
}

fn parse_begin_mode(tokens: &[String]) -> Option<(usize, BeginModeKind)> {
    let refs: Vec<_> = tokens.iter().map(String::as_str).collect();

    if refs.len() >= 4 && is_isolation_level_suffix(&refs[..4]) {
        return Some((4, BeginModeKind::IsolationLevel));
    }

    if refs.len() >= 3 && is_isolation_level_suffix(&refs[..3]) {
        return Some((3, BeginModeKind::IsolationLevel));
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
            return Some((2, BeginModeKind::AccessMode));
        }

        if matches!(
            two,
            [not, deferrable]
                if not.eq_ignore_ascii_case("NOT") && deferrable.eq_ignore_ascii_case("DEFERRABLE")
        ) {
            return Some((2, BeginModeKind::Deferrable));
        }
    }

    if !refs.is_empty() && is_deferrable_suffix(&refs[..1]) {
        return Some((1, BeginModeKind::Deferrable));
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

fn is_begin_mode_list(tokens: &[String]) -> bool {
    if tokens.is_empty() {
        return false;
    }

    let mut idx = 0;
    let mut seen_access_mode = false;
    let mut seen_isolation_level = false;
    let mut seen_deferrable = false;

    while idx < tokens.len() {
        if tokens[idx] == "," {
            return false;
        }

        let Some((consumed, kind)) = parse_begin_mode(&tokens[idx..]) else {
            return false;
        };

        match kind {
            BeginModeKind::AccessMode if seen_access_mode => return false,
            BeginModeKind::IsolationLevel if seen_isolation_level => return false,
            BeginModeKind::Deferrable if seen_deferrable => return false,
            BeginModeKind::AccessMode => seen_access_mode = true,
            BeginModeKind::IsolationLevel => seen_isolation_level = true,
            BeginModeKind::Deferrable => seen_deferrable = true,
        }

        idx += consumed;
        if idx == tokens.len() {
            return true;
        }

        if tokens[idx] == "," {
            idx += 1;
            if idx == tokens.len() || tokens[idx] == "," {
                return false;
            }
        }
    }

    true
}

fn strip_single_leading_comma(tokens: &[String]) -> Option<&[String]> {
    match tokens {
        [first, rest @ ..] if first == "," && !rest.is_empty() => Some(rest),
        _ => None,
    }
}

fn is_begin_with_optional_mode(input: &str) -> bool {
    let tokens = normalize_begin_tokens(input);
    let Some((first, rest)) = tokens.split_first() else {
        return false;
    };

    if first.eq_ignore_ascii_case("BEGIN") {
        return match rest {
            [] => true,
            [second] if second.eq_ignore_ascii_case("TRANSACTION") => true,
            [second] if second.eq_ignore_ascii_case("WORK") => true,
            mode if is_begin_mode_list(mode) => true,
            [second, mode @ ..]
                if second.eq_ignore_ascii_case("TRANSACTION")
                    || second.eq_ignore_ascii_case("WORK") =>
            {
                if is_begin_mode_list(mode) {
                    true
                } else {
                    strip_single_leading_comma(mode).is_some_and(is_begin_mode_list)
                }
            }
            _ => false,
        };
    }

    if first.eq_ignore_ascii_case("START") {
        return match rest {
            [second] if second.eq_ignore_ascii_case("TRANSACTION") => true,
            [second] if second.eq_ignore_ascii_case("WORK") => true,
            [second, mode @ ..]
                if second.eq_ignore_ascii_case("TRANSACTION")
                    || second.eq_ignore_ascii_case("WORK") =>
            {
                if is_begin_mode_list(mode) {
                    true
                } else {
                    strip_single_leading_comma(mode).is_some_and(is_begin_mode_list)
                }
            }
            _ => false,
        };
    }

    false
}

/// Parse a single SQL command (the strict, public entry). A SELECT whose FROM target is a
/// `pg_catalog.`/`information_schema.`-qualified relation is rejected, exactly as before —
/// the legacy compatibility server relies on this (catalog queries fail to parse and route
/// to its compatibility layer). The engine uses [`parse_command_allowing_catalog`] instead.
pub fn parse_command(input: &str) -> Result<Command, ParseError> {
    parse_command_inner(input, false)
}

/// Like [`parse_command`] but carries a `pg_catalog.`/`information_schema.` qualifier on a
/// SELECT's FROM target through to the parsed `Select`, for the engine's native catalog
/// support (Phase-3 M2). Behaves identically to [`parse_command`] for every other command.
pub fn parse_command_allowing_catalog(input: &str) -> Result<Command, ParseError> {
    parse_command_inner(input, true)
}

fn parse_command_inner(input: &str, allow_catalog_schemas: bool) -> Result<Command, ParseError> {
    let mut s = input.trim_end();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    while let Some(without_semicolon) = s.strip_suffix(';') {
        s = without_semicolon.trim_end();
    }

    let s = s.trim_start();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    if is_begin_with_optional_mode(s) {
        return Ok(Command::Begin);
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
    if let Some(relational) = parse_relational_command(s, allow_catalog_schemas) {
        return relational;
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
