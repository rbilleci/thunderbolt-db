// Legacy parsed session-state ownership. This is not a product execution path.

use super::{
    role_exists, write_command_complete, write_error, Command, ErrorField, ReadWrite, Session,
};
use std::io;

pub(super) fn try_execute_session_command(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: &Command,
    canonical: &str,
) -> Option<io::Result<()>> {
    match command {
        Command::SetRole { role } => {
            if let Some(role) = role {
                if !role_exists(session, role) {
                    return Some(write_error(
                        stream,
                        &ErrorField {
                            code: "42704",
                            message: "role does not exist",
                            position: None,
                        },
                    ));
                }
                session.current_role = Some(role.clone());
            } else {
                session.current_role = None;
            }
            Some(write_command_complete(stream, "SET"))
        }
        Command::Begin { .. } => {
            session.in_transaction = true;
            Some(write_command_complete(stream, "BEGIN"))
        }
        Command::Commit { chain } => {
            session.cursors.clear();
            session.in_transaction = *chain;
            Some(write_command_complete(stream, "COMMIT"))
        }
        Command::Rollback { chain } => {
            session.cursors.clear();
            session.in_transaction = *chain;
            Some(write_command_complete(stream, "ROLLBACK"))
        }
        Command::ResetAll
            if matches!(
                canonical,
                "reset role" | "reset session role" | "reset local role"
            ) =>
        {
            session.current_role = None;
            Some(write_command_complete(stream, "RESET"))
        }
        _ => None,
    }
}
