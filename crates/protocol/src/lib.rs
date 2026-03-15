#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Begin,
    Commit,
    Rollback,
    SetKv { key: String, value: String },
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("empty command")]
    Empty,
    #[error("unsupported command: {0}")]
    Unsupported(String),
    #[error("invalid SET syntax; expected: SET key=value")]
    InvalidSet,
}

pub fn parse_command(input: &str) -> Result<Command, ParseError> {
    let s = input.trim();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    let upper = s.to_ascii_uppercase();
    if upper == "BEGIN" {
        return Ok(Command::Begin);
    }
    if upper == "COMMIT" {
        return Ok(Command::Commit);
    }
    if upper == "ROLLBACK" {
        return Ok(Command::Rollback);
    }

    if upper.starts_with("SET ") {
        let rest = &s[4..];
        let Some((k, v)) = rest.split_once('=') else {
            return Err(ParseError::InvalidSet);
        };
        let key = k.trim();
        let value = v.trim();
        if key.is_empty() {
            return Err(ParseError::InvalidSet);
        }
        return Ok(Command::SetKv {
            key: key.to_string(),
            value: value.to_string(),
        });
    }

    Err(ParseError::Unsupported(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_set() {
        let cmd = parse_command("SET a = 42").unwrap();
        assert_eq!(cmd, Command::SetKv { key: "a".into(), value: "42".into() });
    }
}
