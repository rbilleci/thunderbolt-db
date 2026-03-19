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
    #[error("invalid DEL/DELETE syntax; expected: DEL key")]
    InvalidDel,
    #[error("invalid GET syntax; expected: GET key")]
    InvalidGet,
}

fn is_transaction_control(input: &str, keyword: &str) -> bool {
    let mut tokens = input.split_whitespace();
    let Some(first) = tokens.next() else {
        return false;
    };
    if !first.eq_ignore_ascii_case(keyword) {
        return false;
    }

    match tokens.next() {
        None => true,
        Some(second)
            if second.eq_ignore_ascii_case("TRANSACTION")
                || second.eq_ignore_ascii_case("WORK") =>
        {
            tokens.next().is_none()
        }
        Some(_) => false,
    }
}

fn is_start_begin_alias(input: &str) -> bool {
    let mut tokens = input.split_whitespace();
    matches!(
        (tokens.next(), tokens.next(), tokens.next()),
        (Some(first), Some(second), None)
            if first.eq_ignore_ascii_case("START")
                && (second.eq_ignore_ascii_case("TRANSACTION") || second.eq_ignore_ascii_case("WORK"))
    )
}

pub fn parse_command(input: &str) -> Result<Command, ParseError> {
    let s = input.trim();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }
    let s = s.strip_suffix(';').unwrap_or(s).trim_end();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    if is_transaction_control(s, "BEGIN") || is_start_begin_alias(s) {
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
        assert_eq!(parse_command("START TRANSACTION").unwrap(), Command::Begin);
        assert_eq!(parse_command("START WORK").unwrap(), Command::Begin);
        assert_eq!(parse_command("COMMIT WORK").unwrap(), Command::Commit);
        assert_eq!(
            parse_command("COMMIT TRANSACTION").unwrap(),
            Command::Commit
        );
        assert_eq!(parse_command("END WORK").unwrap(), Command::Commit);
        assert_eq!(parse_command("END TRANSACTION").unwrap(), Command::Commit);
        assert_eq!(parse_command("ROLLBACK WORK").unwrap(), Command::Rollback);
        assert_eq!(
            parse_command("ROLLBACK TRANSACTION").unwrap(),
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
            parse_command("START TRANSACTION READ ONLY"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("COMMIT WORK PLEASE"),
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
            parse_command("ABORT TRANSACTION AGAIN"),
            Err(ParseError::Unsupported(_))
        ));
    }

    #[test]
    fn accepts_optional_statement_terminator() {
        assert_eq!(parse_command("BEGIN;").unwrap(), Command::Begin);
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
