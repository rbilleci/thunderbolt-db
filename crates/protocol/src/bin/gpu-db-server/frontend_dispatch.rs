use super::frontend_transport::{read_tagged_frame, ReadWrite};
use super::{
    apply_copy_in_rows, handle_bind, handle_close, handle_copy_data, handle_describe,
    handle_execute, handle_parse, run_simple_query, write_command_complete, write_error,
    write_ready_for_query, ErrorField, Session,
};
use gpu_db_protocol::{parse_frontend_message, FrontendMessage, ReadyLoopState};
use std::io::{self, ErrorKind};

pub(super) fn handle_ready_client(
    stream: &mut dyn ReadWrite,
    shared_catalog: bool,
) -> io::Result<()> {
    let mut session = Session::new(shared_catalog);
    let mut extended_error_pending = false;

    loop {
        let Some(frame) = read_tagged_frame(stream)? else {
            return Ok(());
        };

        let message = parse_frontend_message(&frame)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?;
        if !handle_frontend_message(stream, &mut session, &mut extended_error_pending, message)? {
            return Ok(());
        }
    }
}

fn handle_frontend_message(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    extended_error_pending: &mut bool,
    message: FrontendMessage,
) -> io::Result<bool> {
    let mut ready_loop =
        ReadyLoopState::from_flags(session.in_transaction, *extended_error_pending);
    match message {
        FrontendMessage::SimpleQuery(query) => {
            if session.copy_in.is_some() {
                write_error(
                    stream,
                    &ErrorField {
                        code: "08P01",
                        message: "simple query is not allowed during COPY FROM STDIN",
                        position: None,
                    },
                )?;
            } else if ready_loop.should_dispatch_extended_message() {
                run_simple_query(stream, session, &query)?
            }
        }
        FrontendMessage::Parse {
            statement_name,
            query,
            parameter_type_oids,
        } => {
            if ready_loop.should_dispatch_extended_message() {
                *extended_error_pending =
                    handle_parse(stream, session, statement_name, query, parameter_type_oids)?;
                if *extended_error_pending {
                    ready_loop.mark_extended_error();
                }
            }
        }
        FrontendMessage::Bind {
            portal_name,
            statement_name,
            parameter_format_codes,
            parameters,
            result_format_codes,
        } => {
            if ready_loop.should_dispatch_extended_message() {
                *extended_error_pending = handle_bind(
                    stream,
                    session,
                    portal_name,
                    statement_name,
                    parameter_format_codes,
                    parameters,
                    result_format_codes,
                )?;
                if *extended_error_pending {
                    ready_loop.mark_extended_error();
                }
            }
        }
        FrontendMessage::Describe { target, name } => {
            if ready_loop.should_dispatch_extended_message() {
                *extended_error_pending = handle_describe(stream, session, target, &name)?;
                if *extended_error_pending {
                    ready_loop.mark_extended_error();
                }
            }
        }
        FrontendMessage::Execute {
            portal_name,
            max_rows,
        } => {
            if ready_loop.should_dispatch_extended_message() {
                *extended_error_pending = handle_execute(stream, session, &portal_name, max_rows)?;
                if *extended_error_pending {
                    ready_loop.mark_extended_error();
                }
            }
        }
        FrontendMessage::Close { target, name } => {
            if ready_loop.should_dispatch_extended_message() {
                *extended_error_pending = handle_close(stream, session, target, &name)?;
                if *extended_error_pending {
                    ready_loop.mark_extended_error();
                }
            }
        }
        FrontendMessage::Terminate => return Ok(false),
        FrontendMessage::Sync => {
            if ready_loop.clear_extended_error_on_sync(session.copy_in.is_some()) {
                *extended_error_pending = ready_loop.skip_until_sync();
                write_ready_for_query(stream, session.in_transaction)?
            }
        }
        FrontendMessage::Flush => stream.flush()?,
        FrontendMessage::CopyData(bytes) => {
            if let Some(error) = handle_copy_data(session, &bytes) {
                let ready_after_done = session
                    .copy_in
                    .take()
                    .is_none_or(|copy| copy.ready_after_done);
                write_error(stream, &error)?;
                if ready_after_done {
                    write_ready_for_query(stream, session.in_transaction)?;
                } else {
                    ready_loop.mark_extended_error();
                    *extended_error_pending = ready_loop.skip_until_sync();
                }
            } else if session.copy_in.is_none() && ready_loop.should_dispatch_extended_message() {
                write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message:
                            "frontend COPY data flow is not supported by the compatibility endpoint",
                        position: None,
                    },
                )?;
                ready_loop.mark_extended_error();
                *extended_error_pending = ready_loop.skip_until_sync();
            }
        }
        FrontendMessage::CopyDone => {
            if let Some(copy) = session.copy_in.take() {
                let copied = copy.pending_rows.len();
                let ready_after_done = copy.ready_after_done;
                if let Some(error) = apply_copy_in_rows(session, copy) {
                    write_error(stream, &error)?;
                } else {
                    write_command_complete(stream, &format!("COPY {copied}"))?;
                }
                if ready_after_done {
                    write_ready_for_query(stream, session.in_transaction)?;
                }
            } else if ready_loop.should_dispatch_extended_message() {
                write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message:
                            "frontend COPY data flow is not supported by the compatibility endpoint",
                        position: None,
                    },
                )?;
                ready_loop.mark_extended_error();
                *extended_error_pending = ready_loop.skip_until_sync();
            }
        }
        FrontendMessage::CopyFail(_) => {
            if let Some(copy) = session.copy_in.take() {
                write_error(
                    stream,
                    &ErrorField {
                        code: "57014",
                        message: "COPY from stdin was aborted by the client",
                        position: None,
                    },
                )?;
                if copy.ready_after_done {
                    write_ready_for_query(stream, session.in_transaction)?;
                } else {
                    ready_loop.mark_extended_error();
                    *extended_error_pending = ready_loop.skip_until_sync();
                }
            } else if ready_loop.should_dispatch_extended_message() {
                write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message:
                            "frontend COPY data flow is not supported by the compatibility endpoint",
                        position: None,
                    },
                )?;
                ready_loop.mark_extended_error();
                *extended_error_pending = ready_loop.skip_until_sync();
            }
        }
        other => {
            if ready_loop.should_dispatch_extended_message() {
                write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: unsupported_frontend_message(&other),
                        position: None,
                    },
                )?;
                ready_loop.mark_extended_error();
                *extended_error_pending = ready_loop.skip_until_sync();
            }
        }
    }
    Ok(true)
}

fn unsupported_frontend_message(message: &FrontendMessage) -> &'static str {
    match message {
        FrontendMessage::PasswordMessage(_) => "password messages are not supported after startup",
        FrontendMessage::SaslInitialResponse { .. } | FrontendMessage::SaslResponse(_) => {
            "SASL authentication is not supported"
        }
        FrontendMessage::FunctionCall { .. } => {
            "FunctionCall is not supported by the compatibility endpoint"
        }
        FrontendMessage::CopyData(_) | FrontendMessage::CopyDone | FrontendMessage::CopyFail(_) => {
            "frontend COPY data flow is not supported by the compatibility endpoint"
        }
        FrontendMessage::SimpleQuery(_)
        | FrontendMessage::Bind { .. }
        | FrontendMessage::Parse { .. }
        | FrontendMessage::Describe { .. }
        | FrontendMessage::Close { .. }
        | FrontendMessage::Execute { .. }
        | FrontendMessage::Terminate
        | FrontendMessage::Sync
        | FrontendMessage::Flush => "unsupported frontend message",
    }
}

#[cfg(test)]
pub(super) fn test_handle_frontend_message(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    extended_error_pending: &mut bool,
    message: FrontendMessage,
) -> io::Result<bool> {
    handle_frontend_message(stream, session, extended_error_pending, message)
}

#[cfg(test)]
pub(super) fn test_unsupported_frontend_message(message: &FrontendMessage) -> &'static str {
    unsupported_frontend_message(message)
}
