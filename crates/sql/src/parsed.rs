//! Owned parse result that keeps the canonical request bytes paired with their typed command.

use std::sync::Arc;

use crate::{parse_command, Command, ParseError};

/// One SQL command parsed exactly once and paired with the exact source text that produced it.
///
/// The private fields prevent a caller from pairing one typed command with different WAL/request
/// bytes. Prepared Bind support can later produce the same invariant from a typed template without
/// making mutation admission accept loose `(Command, text)` arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCommand {
    source: Arc<str>,
    command: Command,
}

impl ParsedCommand {
    pub fn parse(source: &str) -> Result<Self, ParseError> {
        let command = parse_command(source)?;
        Ok(Self {
            source: Arc::from(source),
            command,
        })
    }

    pub fn command(&self) -> &Command {
        &self.command
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn from_bound_prepared(source: Arc<str>, command: Command) -> Self {
        debug_assert_eq!(crate::prepared::command_parameter_count(&command), 0);
        Self { source, command }
    }

    pub fn into_parts(self) -> (Command, Arc<str>) {
        (self.command, self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::ParsedCommand;
    use crate::Command;

    #[test]
    fn parsed_command_keeps_typed_command_and_exact_source_together() {
        let source = "INSERT INTO t (id) VALUES (7::int4)";
        let parsed = ParsedCommand::parse(source).unwrap();
        assert!(matches!(parsed.command(), Command::Insert(_)));
        assert_eq!(parsed.source(), source);
        let (command, retained) = parsed.into_parts();
        assert!(matches!(command, Command::Insert(_)));
        assert_eq!(&*retained, source);
    }
}
