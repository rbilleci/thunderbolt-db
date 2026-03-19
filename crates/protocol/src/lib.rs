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
    let Some((first, rest)) = tokens.split_first() else {
        return false;
    };
    if !first.eq_ignore_ascii_case(keyword) {
        return false;
    }

    match rest {
        [] => true,
        [second] if second.eq_ignore_ascii_case("TRANSACTION") => true,
        [second] if second.eq_ignore_ascii_case("WORK") => true,
        _ if keyword.eq_ignore_ascii_case("COMMIT") || keyword.eq_ignore_ascii_case("ROLLBACK") => {
            is_transaction_chain_suffix(rest)
        }
        _ => false,
    }
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
    )
}

fn is_begin_with_optional_mode(input: &str) -> bool {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let Some((first, rest)) = tokens.split_first() else {
        return false;
    };

    if first.eq_ignore_ascii_case("BEGIN") {
        return match rest {
            [] => true,
            [second] if second.eq_ignore_ascii_case("TRANSACTION") => true,
            [second] if second.eq_ignore_ascii_case("WORK") => true,
            mode if is_begin_mode_suffix(mode) => true,
            [second, mode @ ..]
                if second.eq_ignore_ascii_case("TRANSACTION")
                    || second.eq_ignore_ascii_case("WORK") =>
            {
                is_begin_mode_suffix(mode)
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
                is_begin_mode_suffix(mode)
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
            parse_command("BEGIN TRANSACTION READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN WORK READ WRITE").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("START TRANSACTION").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("START TRANSACTION READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("START WORK").unwrap(), Command::Begin);
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
        assert_eq!(parse_command("ABORT WORK").unwrap(), Command::Rollback);
        assert_eq!(
            parse_command("ABORT TRANSACTION").unwrap(),
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
            parse_command("ABORT TRANSACTION AGAIN"),
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
