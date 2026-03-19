#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Begin,
    Commit,
    Rollback,
    Flush,
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
    #[error("invalid SET syntax; expected: SET key=value")]
    InvalidSet,
    #[error("invalid DEL/DELETE syntax; expected: DEL|DELETE key")]
    InvalidDel,
    #[error("invalid GET syntax; expected: GET key")]
    InvalidGet,
}

fn is_transaction_chain_suffix(tokens: &[&str]) -> bool {
    matches!(
        tokens,
        [and, chain]
            if and.eq_ignore_ascii_case("AND") && chain.eq_ignore_ascii_case("CHAIN")
    ) || matches!(
        tokens,
        [and, no, chain]
            if and.eq_ignore_ascii_case("AND")
                && no.eq_ignore_ascii_case("NO")
                && chain.eq_ignore_ascii_case("CHAIN")
    )
}

fn is_transaction_control(input: &str, keyword: &str) -> bool {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let Some((first, mut rest)) = tokens.split_first() else {
        return false;
    };
    if !first.eq_ignore_ascii_case(keyword) {
        return false;
    }

    if let [second] = rest {
        if second.eq_ignore_ascii_case("TRANSACTION") || second.eq_ignore_ascii_case("WORK") {
            return true;
        }
    }

    if keyword.eq_ignore_ascii_case("COMMIT")
        || keyword.eq_ignore_ascii_case("ROLLBACK")
        || keyword.eq_ignore_ascii_case("ABORT")
    {
        if let Some((scope, tail)) = rest.split_first() {
            if scope.eq_ignore_ascii_case("TRANSACTION") || scope.eq_ignore_ascii_case("WORK") {
                rest = tail;
            }
        }
        return rest.is_empty() || is_transaction_chain_suffix(rest);
    }

    rest.is_empty()
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

fn is_begin_mode_suffix(tokens: &[&str]) -> bool {
    matches!(
        tokens,
        [read, only]
            if read.eq_ignore_ascii_case("READ") && only.eq_ignore_ascii_case("ONLY")
    ) || matches!(
        tokens,
        [read, write]
            if read.eq_ignore_ascii_case("READ") && write.eq_ignore_ascii_case("WRITE")
    ) || is_isolation_level_suffix(tokens)
        || is_deferrable_suffix(tokens)
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
    while idx < tokens.len() {
        if tokens[idx] == "," {
            return false;
        }

        let remaining = &tokens[idx..];
        let remaining_refs: Vec<_> = remaining.iter().map(String::as_str).collect();
        let consumed =
            if remaining_refs.len() >= 4 && is_isolation_level_suffix(&remaining_refs[..4]) {
                4
            } else if remaining_refs.len() >= 3 && is_isolation_level_suffix(&remaining_refs[..3]) {
                3
            } else if remaining_refs.len() >= 2 && is_begin_mode_suffix(&remaining_refs[..2]) {
                2
            } else if is_begin_mode_suffix(&remaining_refs[..1]) {
                1
            } else {
                return false;
            };

        idx += consumed;
        if idx == tokens.len() {
            return true;
        }
        if tokens[idx] != "," {
            return false;
        }
        idx += 1;
        if idx == tokens.len() {
            return false;
        }
    }

    true
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
                is_begin_mode_list(mode)
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
                is_begin_mode_list(mode)
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
    if is_transaction_control(s, "COMMIT") || is_transaction_control(s, "END") {
        return Ok(Command::Commit);
    }
    if is_transaction_control(s, "ROLLBACK") || is_transaction_control(s, "ABORT") {
        return Ok(Command::Rollback);
    }
    if s.eq_ignore_ascii_case("FLUSH") {
        return Ok(Command::Flush);
    }

    let mut parts = s.splitn(2, char::is_whitespace);
    if let Some(cmd) = parts.next() {
        if cmd.eq_ignore_ascii_case("SET") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidSet);
            };
            let Some((k, v)) = rest.split_once('=') else {
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

        if cmd.eq_ignore_ascii_case("DEL") || cmd.eq_ignore_ascii_case("DELETE") {
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
    fn rejects_set_with_whitespace_in_key() {
        assert!(matches!(
            parse_command("SET two words=42"),
            Err(ParseError::InvalidSet)
        ));
    }

    #[test]
    fn parses_flush() {
        let cmd = parse_command("FLUSH").unwrap();
        assert_eq!(cmd, Command::Flush);
    }

    #[test]
    fn parses_transaction_control_commands_case_insensitively() {
        assert_eq!(parse_command("begin").unwrap(), Command::Begin);
        assert_eq!(parse_command("COMMIT").unwrap(), Command::Commit);
        assert_eq!(parse_command("END").unwrap(), Command::Commit);
        assert_eq!(parse_command("rOlLbAcK").unwrap(), Command::Rollback);
        assert_eq!(parse_command("abort").unwrap(), Command::Rollback);
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
        assert_eq!(parse_command("START TRANSACTION").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("START TRANSACTION READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION ISOLATION LEVEL REPEATABLE READ").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("START WORK").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("START TRANSACTION DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK READ WRITE").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("COMMIT WORK").unwrap(), Command::Commit);
        assert_eq!(
            parse_command("COMMIT TRANSACTION").unwrap(),
            Command::Commit
        );
        assert_eq!(parse_command("COMMIT AND CHAIN").unwrap(), Command::Commit);
        assert_eq!(
            parse_command("COMMIT AND NO CHAIN").unwrap(),
            Command::Commit
        );
        assert_eq!(
            parse_command("COMMIT TRANSACTION AND CHAIN").unwrap(),
            Command::Commit
        );
        assert_eq!(
            parse_command("COMMIT WORK AND NO CHAIN").unwrap(),
            Command::Commit
        );
        assert_eq!(parse_command("END WORK").unwrap(), Command::Commit);
        assert_eq!(parse_command("END TRANSACTION").unwrap(), Command::Commit);
        assert_eq!(parse_command("ROLLBACK WORK").unwrap(), Command::Rollback);
        assert_eq!(
            parse_command("ROLLBACK TRANSACTION").unwrap(),
            Command::Rollback
        );
        assert_eq!(
            parse_command("ROLLBACK AND CHAIN").unwrap(),
            Command::Rollback
        );
        assert_eq!(
            parse_command("ROLLBACK AND NO CHAIN").unwrap(),
            Command::Rollback
        );
        assert_eq!(
            parse_command("ROLLBACK TRANSACTION AND CHAIN").unwrap(),
            Command::Rollback
        );
        assert_eq!(
            parse_command("ROLLBACK WORK AND NO CHAIN").unwrap(),
            Command::Rollback
        );
        assert_eq!(parse_command("ABORT WORK").unwrap(), Command::Rollback);
        assert_eq!(
            parse_command("ABORT TRANSACTION").unwrap(),
            Command::Rollback
        );
        assert_eq!(parse_command("ABORT AND CHAIN").unwrap(), Command::Rollback);
        assert_eq!(
            parse_command("ABORT AND NO CHAIN").unwrap(),
            Command::Rollback
        );
        assert_eq!(
            parse_command("ABORT TRANSACTION AND CHAIN").unwrap(),
            Command::Rollback
        );
        assert_eq!(
            parse_command("ABORT WORK AND NO CHAIN").unwrap(),
            Command::Rollback
        );
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
            parse_command("START TRANSACTION READ"),
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
    }

    #[test]
    fn accepts_optional_statement_terminator() {
        assert_eq!(parse_command("BEGIN;").unwrap(), Command::Begin);
        assert_eq!(parse_command("START WORK;").unwrap(), Command::Begin);
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
    fn rejects_del_with_missing_or_extra_tokens() {
        assert!(matches!(parse_command("DEL"), Err(ParseError::InvalidDel)));
        assert!(matches!(
            parse_command("DEL too many"),
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
