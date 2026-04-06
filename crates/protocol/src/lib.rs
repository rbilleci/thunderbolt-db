#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Begin,
    Commit { chain: bool },
    Rollback { chain: bool },
    Flush,
    ResetAll,
    SetKv { key: String, value: String },
    DeleteKv { key: String },
    GetKv { key: String },
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("empty command")]
    Empty,
    #[error("unsupported command: {0}")]
    Unsupported(String),
    #[error("invalid SET syntax; expected: SET key=value or SET key TO value")]
    InvalidSet,
    #[error("invalid DEL/DELETE syntax; expected: DEL key or DELETE [FROM] key")]
    InvalidDel,
    #[error("invalid GET syntax; expected: GET key")]
    InvalidGet,
    #[error("invalid RESET/DISCARD/DEALLOCATE syntax; expected: RESET ALL|ROLE|AUTHORIZATION|AUTH|SESSION AUTHORIZATION|SESSION AUTH, DISCARD {{ALL|TEMP|TEMPORARY|TEMP TABLES|TEMPORARY TABLES|PLANS|SEQUENCES}}, or DEALLOCATE ALL")]
    InvalidReset,
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
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD") =>
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
            [session, authorization]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH")) =>
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
        return Some(match rest {
            [target] if target.eq_ignore_ascii_case("ALL") => Ok(Command::ResetAll),
            _ => Err(ParseError::InvalidReset),
        });
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

pub fn parse_command(input: &str) -> Result<Command, ParseError> {
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

    let mut parts = s.splitn(2, char::is_whitespace);
    if let Some(cmd) = parts.next() {
        if cmd.eq_ignore_ascii_case("SET") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidSet);
            };
            let Some((k, v)) = split_set_key_value(rest) else {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_set() {
        let cmd = parse_command("SET a = 42").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "a".into(),
                value: "42".into()
            }
        );
    }

    #[test]
    fn parses_set_with_non_space_whitespace_separator() {
        let cmd = parse_command("SET\ta = 42").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "a".into(),
                value: "42".into()
            }
        );
    }

    #[test]
    fn parses_set_with_to_assignment_alias() {
        let cmd = parse_command("SET a TO 42").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "a".into(),
                value: "42".into()
            }
        );

        let cmd = parse_command("SET alpha to value words").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "alpha".into(),
                value: "value words".into()
            }
        );
    }

    #[test]
    fn rejects_set_with_whitespace_in_key() {
        assert!(matches!(
            parse_command("SET two words=42"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET a TO"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET a TO42"),
            Err(ParseError::InvalidSet)
        ));
    }

    #[test]
    fn parses_flush() {
        let cmd = parse_command("FLUSH").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE AHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE AHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE AHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE-AHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE-AHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE-AHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITEAHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITEAHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITEAHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);
    }

    #[test]
    fn parses_reset_all() {
        let cmd = parse_command("RESET ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET ROLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET AUTHORIZATION").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET AUTH").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTHORIZATION").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTH").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMP").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMPORARY").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMP TABLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMP TABLES").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMPORARY TABLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMPORARY TABLES").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD PLANS").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD SEQUENCES").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);
    }

    #[test]
    fn parses_transaction_control_commands_case_insensitively() {
        assert_eq!(parse_command("begin").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("COMMIT").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("rOlLbAcK").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("abort").unwrap(),
            Command::Rollback { chain: false }
        );
    }

    #[test]
    fn parses_transaction_control_work_and_transaction_aliases() {
        assert_eq!(parse_command("BEGIN WORK").unwrap(), Command::Begin);
        assert_eq!(parse_command("BEGIN TRANSACTION").unwrap(), Command::Begin);
        assert_eq!(parse_command("BEGIN READ ONLY").unwrap(), Command::Begin);
        assert_eq!(parse_command("BEGIN READ WRITE").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("BEGIN ISOLATION LEVEL SERIALIZABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN ISOLATION LEVEL REPEATABLE READ").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN ISOLATION LEVEL READ COMMITTED").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN ISOLATION LEVEL READ UNCOMMITTED").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("BEGIN DEFERRABLE").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("BEGIN NOT DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN TRANSACTION READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN WORK READ WRITE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN READ WRITE, ISOLATION LEVEL SERIALIZABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN READ ONLY , DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN READ ONLY DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN READ WRITE ISOLATION LEVEL SERIALIZABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("START TRANSACTION").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("BEGIN TRANSACTION, READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN WORK, READ WRITE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION, READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION ISOLATION LEVEL REPEATABLE READ").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("START WORK").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("START WORK, READ WRITE, DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK ISOLATION LEVEL READ COMMITTED").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK, ISOLATION LEVEL REPEATABLE READ, NOT DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION READ ONLY DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK READ WRITE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("COMMIT WORK").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("COMMIT TRANSACTION").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("COMMIT AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("COMMIT AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("COMMIT TRANSACTION AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("COMMIT WORK AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END WORK").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END TRANSACTION").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("END AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END TRANSACTION AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("END WORK AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK WORK").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK TRANSACTION").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ROLLBACK AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK TRANSACTION AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ROLLBACK WORK AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT WORK").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT TRANSACTION").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ABORT AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT TRANSACTION AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ABORT WORK AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
    }

    #[test]
    fn parses_begin_mode_lists_with_mixed_order_and_delimiters() {
        assert_eq!(
            parse_command(
                "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY, NOT DEFERRABLE"
            )
            .unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN READ WRITE ISOLATION LEVEL READ COMMITTED DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK, NOT DEFERRABLE, ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                .unwrap(),
            Command::Begin
        );
    }

    #[test]
    fn rejects_begin_mode_lists_with_duplicate_mode_kinds_even_when_comma_delimited() {
        assert!(matches!(
            parse_command("BEGIN READ ONLY, READ WRITE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN DEFERRABLE, NOT DEFERRABLE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command(
                "START TRANSACTION ISOLATION LEVEL READ COMMITTED, ISOLATION LEVEL SERIALIZABLE"
            ),
            Err(ParseError::Unsupported(_))
        ));
    }

    #[test]
    fn rejects_transaction_control_commands_with_extra_tokens() {
        assert!(matches!(
            parse_command("BEGIN TRANSACTION NOW"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN READ"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN READ COMMITTED"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN ISOLATION LEVEL"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN ISOLATION LEVEL SNAPSHOT"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN NOT"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN READ ONLY, "),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN , READ ONLY"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN READ ONLY,, DEFERRABLE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START TRANSACTION READ"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN TRANSACTION,"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START WORK, "),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START TRANSACTION READ COMMITTED"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START WORK NOW"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN READ ONLY READ WRITE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN READ ONLY, READ WRITE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START TRANSACTION READ ONLY, READ ONLY"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command(
                "START WORK ISOLATION LEVEL READ COMMITTED, ISOLATION LEVEL SERIALIZABLE"
            ),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN DEFERRABLE NOT DEFERRABLE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN ISOLATION LEVEL SERIALIZABLE, ISOLATION LEVEL READ COMMITTED"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("COMMIT WORK PLEASE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("COMMIT AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("COMMIT AND MAYBE CHAIN"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("COMMIT TRANSACTION AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("END WORK PLEASE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("END AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("END WORK AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("ROLLBACK TRANSACTION AGAIN"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("ROLLBACK AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("ROLLBACK WORK AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("ABORT TRANSACTION AGAIN"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("ABORT AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("ABORT WORK AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("FLUSH NOW"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("FLUSH WAL NOW"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("RESET"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("RESET SESSION"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("RESET ALL NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DISCARD"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DISCARD TEMP NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DISCARD ALL NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE PREPARE x"),
            Err(ParseError::InvalidReset)
        ));
    }

    #[test]
    fn accepts_optional_statement_terminator() {
        assert_eq!(parse_command("BEGIN;").unwrap(), Command::Begin);
        assert_eq!(parse_command("START WORK;").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("END AND CHAIN;").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("ABORT AND NO CHAIN;\n").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("SET balance = 42;").unwrap(),
            Command::SetKv {
                key: "balance".into(),
                value: "42".into()
            }
        );
        assert_eq!(
            parse_command("GET balance;\n").unwrap(),
            Command::GetKv {
                key: "balance".into()
            }
        );
        assert_eq!(parse_command("RESET ALL;").unwrap(), Command::ResetAll);
        assert_eq!(parse_command("RESET ROLE;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("RESET AUTHORIZATION;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(parse_command("RESET AUTH;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("RESET SESSION AUTHORIZATION;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("RESET SESSION AUTH;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(parse_command("DISCARD ALL;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("FLUSH WRITE AHEAD LOG;\n").unwrap(),
            Command::Flush
        );
        assert_eq!(parse_command("DISCARD TEMP;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("DISCARD TEMP TABLES;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("DEALLOCATE ALL;\n").unwrap(),
            Command::ResetAll
        );
    }

    #[test]
    fn accepts_repeated_statement_terminators() {
        assert_eq!(parse_command("BEGIN;;").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("SET balance = 42; ; \n").unwrap(),
            Command::SetKv {
                key: "balance".into(),
                value: "42".into()
            }
        );
    }

    #[test]
    fn rejects_input_that_is_only_terminators() {
        assert!(matches!(parse_command(";;;"), Err(ParseError::Empty)));
    }

    #[test]
    fn parses_del() {
        let cmd = parse_command("DEL balance").unwrap();
        assert_eq!(
            cmd,
            Command::DeleteKv {
                key: "balance".into()
            }
        );
    }

    #[test]
    fn parses_delete_alias() {
        let cmd = parse_command("DELETE balance").unwrap();
        assert_eq!(
            cmd,
            Command::DeleteKv {
                key: "balance".into()
            }
        );
    }

    #[test]
    fn parses_delete_from_alias() {
        let cmd = parse_command("DELETE FROM balance").unwrap();
        assert_eq!(
            cmd,
            Command::DeleteKv {
                key: "balance".into()
            }
        );
    }

    #[test]
    fn rejects_del_with_missing_or_extra_tokens() {
        assert!(matches!(parse_command("DEL"), Err(ParseError::InvalidDel)));
        assert!(matches!(
            parse_command("DEL too many"),
            Err(ParseError::InvalidDel)
        ));
        assert!(matches!(
            parse_command("DELETE FROM"),
            Err(ParseError::InvalidDel)
        ));
        assert!(matches!(
            parse_command("DELETE FROM too many"),
            Err(ParseError::InvalidDel)
        ));
        assert!(matches!(
            parse_command("DELETE TABLE balance"),
            Err(ParseError::InvalidDel)
        ));
    }

    #[test]
    fn parses_get() {
        let cmd = parse_command("GET balance").unwrap();
        assert_eq!(
            cmd,
            Command::GetKv {
                key: "balance".into()
            }
        );
    }

    #[test]
    fn rejects_get_with_missing_or_extra_tokens() {
        assert!(matches!(parse_command("GET"), Err(ParseError::InvalidGet)));
        assert!(matches!(
            parse_command("GET too many"),
            Err(ParseError::InvalidGet)
        ));
    }
}
