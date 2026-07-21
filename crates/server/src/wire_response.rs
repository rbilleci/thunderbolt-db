//! Canonical pgwire response encoding for every engine-backed server ingress.
//!
//! This module translates neutral facade outcomes and server lifecycle errors into protocol
//! frames. Cancellation may replace only effect-free results before their bytes are published;
//! successful mutation/RETURNING outcomes retain their authoritative result.

use std::io::{self, ErrorKind};

use gpu_db_facade::{pg_adapter, DbError, ErrorCategory, QueryOutcome, SessionTransactionStatus};
use gpu_db_protocol::backend::{BackendColumn, BackendError, BackendWriter};
use gpu_db_protocol::TransactionStatus as WireTransactionStatus;

use crate::cancellation::{cancel_effect_free_success, ActiveRequest, BackendKey};
use crate::extended::{self, ExtendedSession};

pub(crate) fn cancellation_error() -> DbError {
    DbError {
        category: ErrorCategory::Cancelled,
        message: "canceling statement due to user request".to_string(),
    }
}

pub(crate) fn cancellation_checked_outcome(
    active: &ActiveRequest,
    outcome: Result<QueryOutcome, DbError>,
) -> Result<QueryOutcome, DbError> {
    if !active.is_cancelled() {
        return outcome;
    }
    if outcome_can_be_cancelled(&outcome) {
        Err(cancellation_error())
    } else {
        outcome
    }
}

fn outcome_can_be_cancelled(outcome: &Result<QueryOutcome, DbError>) -> bool {
    matches!(
        outcome,
        // These outcomes have no durable/session mutation and can be suppressed safely.
        // Any error may represent an indeterminate post-durable failure.
        // A command/RETURNING result may already be published; neither may be falsely relabelled.
        Ok(QueryOutcome::Rows { .. } | QueryOutcome::CopyIn { .. } | QueryOutcome::Empty)
    )
}

pub(crate) fn encode_cancellable_outcome_messages(
    active: &ActiveRequest,
    outcome: &mut Result<QueryOutcome, DbError>,
) -> io::Result<Vec<u8>> {
    if active.is_cancelled() && outcome_can_be_cancelled(outcome) {
        *outcome = Err(cancellation_error());
    }
    let response = encode_outcome_messages(outcome.clone())?;
    if active.is_cancelled() && outcome_can_be_cancelled(outcome) {
        *outcome = Err(cancellation_error());
        encode_outcome_messages(outcome.clone())
    } else {
        Ok(response)
    }
}

pub(crate) fn cancelled_extended_error() -> extended::ExtendedError {
    extended::ExtendedError::new("57014", "canceling statement due to user request")
}

pub(crate) fn encode_execute_cancellable(
    extended: &mut ExtendedSession,
    portal_name: &str,
    max_rows: u32,
    active: &ActiveRequest,
) -> Result<Vec<u8>, extended::ExtendedError> {
    let can_cancel = extended.portal_outcome_can_be_cancelled(portal_name)?;
    if active.is_cancelled() && can_cancel {
        return Err(cancelled_extended_error());
    }
    let response = extended.encode_execute(portal_name, max_rows);
    cancellation_checked_encoded_response(active, can_cancel, response)
}

fn cancellation_checked_encoded_response(
    active: &ActiveRequest,
    can_cancel: bool,
    response: Result<Vec<u8>, extended::ExtendedError>,
) -> Result<Vec<u8>, extended::ExtendedError> {
    if !can_cancel {
        return response;
    }
    cancel_effect_free_success(active, response, cancelled_extended_error)
}

/// Encode a neutral facade outcome followed by ReadyForQuery.
pub(crate) fn encode_outcome(
    outcome: Result<QueryOutcome, DbError>,
    transaction_status: SessionTransactionStatus,
) -> io::Result<Vec<u8>> {
    let mut buf = encode_outcome_messages(outcome)?;
    buf.extend_from_slice(&encode_ready(transaction_status)?);
    Ok(buf)
}

/// Encode one statement's simple-query messages without ReadyForQuery. A multi-statement Query
/// concatenates these fragments and emits exactly one ReadyForQuery after its final transaction
/// action, matching PostgreSQL's message boundary.
pub(crate) fn encode_outcome_messages(
    outcome: Result<QueryOutcome, DbError>,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = BackendWriter::new(&mut buf);
        match outcome {
            Ok(QueryOutcome::Empty) => {
                writer.empty_query_response()?;
            }
            Ok(QueryOutcome::CopyIn { .. }) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "COPY start outcome reached the ordinary simple-query encoder",
                ));
            }
            Ok(outcome) => {
                let tag = pg_adapter::command_complete_tag(&outcome);
                let returned_rows = match &outcome {
                    QueryOutcome::Rows { columns, rows }
                    | QueryOutcome::Returning { columns, rows, .. } => Some((columns, rows)),
                    _ => None,
                };
                if let Some((columns, rows)) = returned_rows {
                    let backend_columns: Vec<BackendColumn> = columns
                        .iter()
                        .map(|column| {
                            BackendColumn::new(
                                column.name.clone(),
                                pg_adapter::logical_type_oid(column.logical_type),
                                pg_adapter::logical_type_size(column.logical_type),
                            )
                        })
                        .collect();
                    writer.row_description(&backend_columns)?;
                    for row in rows {
                        let values: Vec<Option<String>> =
                            row.iter().map(pg_adapter::db_value_text_opt).collect();
                        writer.data_row(&values)?;
                    }
                }
                writer.command_complete(&tag)?;
            }
            Err(error) => {
                writer.error_response(&BackendError::new(
                    pg_adapter::error_sqlstate(error.category).to_string(),
                    error.message,
                ))?;
            }
        }
    }
    Ok(buf)
}

pub(crate) fn encode_ready(transaction_status: SessionTransactionStatus) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    BackendWriter::new(&mut buf)
        .ready_for_query_status(wire_transaction_status(transaction_status))?;
    Ok(buf)
}

pub(crate) fn encode_optional_error_and_ready(
    error: Option<extended::ExtendedError>,
    transaction_status: SessionTransactionStatus,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    if let Some(error) = error {
        buf.extend_from_slice(&error.encode()?);
    }
    buf.extend_from_slice(&encode_ready(transaction_status)?);
    Ok(buf)
}

fn wire_transaction_status(status: SessionTransactionStatus) -> WireTransactionStatus {
    match status {
        SessionTransactionStatus::Idle => WireTransactionStatus::Idle,
        SessionTransactionStatus::InTransaction => WireTransactionStatus::InTransaction,
        SessionTransactionStatus::FailedTransaction => WireTransactionStatus::FailedTransaction,
    }
}

pub(crate) fn encode_extended_error(error: extended::ExtendedError) -> Vec<u8> {
    error.encode().unwrap_or_default()
}

pub(crate) fn encode_io_error(error: io::Error) -> Vec<u8> {
    encode_extended_error(extended::ExtendedError {
        code: "XX000",
        message: error.to_string(),
    })
}

pub(crate) fn encode_frontend_message_error(
    error: gpu_db_protocol::FrontendMessageError,
) -> Vec<u8> {
    encode_extended_error(extended::ExtendedError {
        code: "08P01",
        message: error.to_string(),
    })
}

/// Build the startup-OK handshake (AuthenticationOk, ParameterStatus x6, BackendKeyData,
/// ReadyForQuery).
pub(crate) fn encode_startup_handshake(backend_key: &BackendKey) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    BackendWriter::new(&mut buf).authentication_ok()?;
    buf.extend_from_slice(&encode_startup_statuses_and_ready(backend_key)?);
    Ok(buf)
}

/// Parameter/status tail shared by trust authentication and SCRAM authentication. SCRAM emits
/// its own AuthenticationSASLFinal + AuthenticationOk before this exact canonical tail.
pub(crate) fn encode_startup_statuses_and_ready(backend_key: &BackendKey) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut writer = BackendWriter::new(&mut buf);
    writer.parameter_status("server_version", "16.0-gpu-db-engine-facade")?;
    writer.parameter_status("server_version_num", "160000")?;
    writer.parameter_status("client_encoding", "UTF8")?;
    writer.parameter_status("DateStyle", "ISO, MDY")?;
    writer.parameter_status("integer_datetimes", "on")?;
    writer.parameter_status("standard_conforming_strings", "on")?;
    writer.backend_key_data(backend_key.process_id_i32(), backend_key.secret_key_i32())?;
    writer.ready_for_query(false)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::cancellation::CancellationRegistry;

    #[test]
    fn post_encoding_error_sentinel_beats_cancellation() {
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));

        let sentinel = extended::ExtendedError::new(
            "XX000",
            "deterministic cached-row response encoding failure",
        );
        let preserved = cancellation_checked_encoded_response(
            &active,
            true,
            Err::<Vec<u8>, _>(sentinel.clone()),
        )
        .unwrap_err();
        assert_eq!(preserved, sentinel);
        assert_ne!(preserved.code, "57014");
    }
}
