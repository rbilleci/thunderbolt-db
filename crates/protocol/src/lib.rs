//! PostgreSQL wire protocol (pgwire) framing and codecs.
//!
//! The neutral SQL vocabulary (command AST, `SqlType`/`SqlValue`, the parser,
//! and `ParseError`) lives in the lower [`gpu_db_sql`] crate and is re-exported
//! here verbatim so this crate's public API is unchanged. Only the wire layer
//! (startup/frontend message parsing, the `backend` encoders) and the legacy
//! `gpu-db-server` binary live here directly (roadmap §9.2).

pub use gpu_db_sql::*;

pub const PG_PROTOCOL_V3: u32 = 196_608;
const PG_SSL_REQUEST_CODE: u32 = 80_877_103;
const PG_GSSENC_REQUEST_CODE: u32 = 80_877_104;
const PG_CANCEL_REQUEST_CODE: u32 = 80_877_102;
const PG_CANCEL_SECRET_KEY_MAX_BYTES: usize = 256;
const PG_PROTOCOL_MAJOR_V3: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupPacket {
    Startup {
        protocol_version: u32,
        params: Vec<(String, String)>,
    },
    SslRequest,
    GssEncRequest,
    CancelRequest {
        process_id: u32,
        secret_key: Vec<u8>,
    },
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum StartupPacketError {
    #[error("startup packet too short")]
    TooShort,
    #[error("startup packet length field is invalid; minimum is 8 bytes, got {declared}")]
    InvalidLengthField { declared: u32 },
    #[error("startup packet length mismatch; expected {expected} bytes, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("startup cancel key exceeds protocol maximum of {max} bytes; got {actual}")]
    InvalidCancelKeyLength { actual: usize, max: usize },
    #[error("unsupported startup protocol code: {0}")]
    UnsupportedProtocolCode(u32),
    #[error("startup parameter payload is not null terminated")]
    UnterminatedParameterPayload,
    #[error("startup parameter payload has odd key/value segment count")]
    InvalidParameterPairing,
    #[error("startup parameter contains invalid UTF-8")]
    InvalidUtf8,
}

fn read_u32_be(bytes: &[u8]) -> Result<u32, StartupPacketError> {
    let arr: [u8; 4] = bytes.try_into().map_err(|_| StartupPacketError::TooShort)?;
    Ok(u32::from_be_bytes(arr))
}

fn protocol_major(version: u32) -> u16 {
    (version >> 16) as u16
}

fn parse_startup_params(payload: &[u8]) -> Result<Vec<(String, String)>, StartupPacketError> {
    let Some(last) = payload.last() else {
        return Err(StartupPacketError::UnterminatedParameterPayload);
    };
    if *last != 0 {
        return Err(StartupPacketError::UnterminatedParameterPayload);
    }

    let mut segments: Vec<&[u8]> = payload[..payload.len() - 1].split(|b| *b == 0).collect();
    if segments.last().is_some_and(|segment| segment.is_empty()) {
        segments.pop();
    }

    if !segments.len().is_multiple_of(2) {
        return Err(StartupPacketError::InvalidParameterPairing);
    }

    let mut params = Vec::with_capacity(segments.len() / 2);
    for pair in segments.chunks_exact(2) {
        if pair[0].is_empty() {
            return Err(StartupPacketError::InvalidParameterPairing);
        }
        let key = std::str::from_utf8(pair[0]).map_err(|_| StartupPacketError::InvalidUtf8)?;
        let value = std::str::from_utf8(pair[1]).map_err(|_| StartupPacketError::InvalidUtf8)?;
        params.push((key.to_owned(), value.to_owned()));
    }
    Ok(params)
}

pub fn parse_startup_packet(frame: &[u8]) -> Result<StartupPacket, StartupPacketError> {
    if frame.len() < 8 {
        return Err(StartupPacketError::TooShort);
    }

    let frame_len = read_u32_be(&frame[..4])? as usize;
    if frame_len < 8 {
        return Err(StartupPacketError::InvalidLengthField {
            declared: frame_len as u32,
        });
    }
    if frame_len != frame.len() {
        return Err(StartupPacketError::LengthMismatch {
            expected: frame_len,
            actual: frame.len(),
        });
    }

    let code = read_u32_be(&frame[4..8])?;
    match code {
        PG_SSL_REQUEST_CODE => {
            if frame_len != 8 {
                return Err(StartupPacketError::LengthMismatch {
                    expected: 8,
                    actual: frame_len,
                });
            }
            Ok(StartupPacket::SslRequest)
        }
        PG_GSSENC_REQUEST_CODE => {
            if frame_len != 8 {
                return Err(StartupPacketError::LengthMismatch {
                    expected: 8,
                    actual: frame_len,
                });
            }
            Ok(StartupPacket::GssEncRequest)
        }
        PG_CANCEL_REQUEST_CODE => {
            if frame_len < 16 {
                return Err(StartupPacketError::LengthMismatch {
                    expected: 16,
                    actual: frame_len,
                });
            }
            let process_id = read_u32_be(&frame[8..12])?;
            let secret_key = frame[12..].to_vec();
            if secret_key.len() > PG_CANCEL_SECRET_KEY_MAX_BYTES {
                return Err(StartupPacketError::InvalidCancelKeyLength {
                    actual: secret_key.len(),
                    max: PG_CANCEL_SECRET_KEY_MAX_BYTES,
                });
            }
            Ok(StartupPacket::CancelRequest {
                process_id,
                secret_key,
            })
        }
        protocol_version if protocol_major(protocol_version) == PG_PROTOCOL_MAJOR_V3 as u16 => {
            let params = parse_startup_params(&frame[8..])?;
            Ok(StartupPacket::Startup {
                protocol_version,
                params,
            })
        }
        other => Err(StartupPacketError::UnsupportedProtocolCode(other)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Startup,
    Authenticating,
    Ready,
    InTransaction,
    Terminating,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionEvent {
    StartupAccepted,
    AuthenticationSucceeded,
    Begin,
    Commit,
    Rollback,
    TerminateRequested,
    ConnectionClosed,
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum SessionTransitionError {
    #[error("invalid session transition from {from:?} using {event:?}")]
    InvalidTransition {
        from: SessionState,
        event: SessionEvent,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionLifecycle {
    state: SessionState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionStatus {
    Idle,
    InTransaction,
}

impl TransactionStatus {
    pub fn ready_for_query_in_transaction(self) -> bool {
        matches!(self, Self::InTransaction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadyLoopState {
    transaction_status: TransactionStatus,
    skip_until_sync: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrontendMessage {
    SimpleQuery(String),
    PasswordMessage(String),
    SaslInitialResponse {
        mechanism: String,
        initial_response: Option<Vec<u8>>,
    },
    SaslResponse(Vec<u8>),
    Bind {
        portal_name: String,
        statement_name: String,
        parameter_format_codes: Vec<i16>,
        parameters: Vec<Option<Vec<u8>>>,
        result_format_codes: Vec<i16>,
    },
    Parse {
        statement_name: String,
        query: String,
        parameter_type_oids: Vec<u32>,
    },
    Describe {
        target: DescribeTarget,
        name: String,
    },
    Close {
        target: DescribeTarget,
        name: String,
    },
    Execute {
        portal_name: String,
        max_rows: u32,
    },
    FunctionCall {
        function_oid: u32,
        argument_format_codes: Vec<i16>,
        arguments: Vec<Option<Vec<u8>>>,
        result_format_code: i16,
    },
    CopyData(Vec<u8>),
    CopyDone,
    CopyFail(String),
    Terminate,
    Sync,
    Flush,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescribeTarget {
    Statement,
    Portal,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum FrontendMessageError {
    #[error("frontend message frame too short")]
    TooShort,
    #[error("frontend message length field is invalid; minimum is 4 bytes, got {declared}")]
    InvalidLengthField { declared: u32 },
    #[error("frontend message length mismatch; expected {expected} bytes, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("unsupported frontend message tag: {0:#x}")]
    UnsupportedTag(u8),
    #[error("simple query payload is not null terminated")]
    UnterminatedSimpleQuery,
    #[error("password message payload is not null terminated")]
    UnterminatedPasswordMessage,
    #[error("sasl-initial-response payload is malformed")]
    InvalidSaslInitialResponsePayload,
    #[error("bind message portal name is not null terminated")]
    UnterminatedBindPortalName,
    #[error("bind message statement name is not null terminated")]
    UnterminatedBindStatementName,
    #[error("bind message payload is malformed")]
    InvalidBindPayload,
    #[error("parse message statement name is not null terminated")]
    UnterminatedParseStatementName,
    #[error("parse message query is not null terminated")]
    UnterminatedParseQuery,
    #[error("parse message parameter type payload is malformed")]
    InvalidParseParameterPayload,
    #[error("describe message target must be S/s (statement) or P/p (portal)")]
    InvalidDescribeTarget,
    #[error("describe message name is not null terminated")]
    UnterminatedDescribeName,
    #[error("close message target must be S/s (statement) or P/p (portal)")]
    InvalidCloseTarget,
    #[error("close message name is not null terminated")]
    UnterminatedCloseName,
    #[error("execute message portal name is not null terminated")]
    UnterminatedExecutePortalName,
    #[error("execute message payload is malformed")]
    InvalidExecutePayload,
    #[error("function-call message payload is malformed")]
    InvalidFunctionCallPayload,
    #[error("copy fail message payload is not null terminated")]
    UnterminatedCopyFail,
    #[error("simple query payload contains invalid UTF-8")]
    InvalidUtf8,
}

fn is_valid_format_code(code: i16) -> bool {
    matches!(code, 0 | 1)
}

fn parse_cstring_payload(
    payload: &[u8],
    unterminated: FrontendMessageError,
) -> Result<&[u8], FrontendMessageError> {
    let Some(bytes) = payload.strip_suffix(&[0]) else {
        return Err(unterminated);
    };
    if bytes.contains(&0) {
        return Err(unterminated);
    }
    Ok(bytes)
}

pub fn parse_frontend_message(frame: &[u8]) -> Result<FrontendMessage, FrontendMessageError> {
    if frame.len() < 5 {
        return Err(FrontendMessageError::TooShort);
    }

    let tag = frame[0];
    let payload_len = u32::from_be_bytes(
        frame[1..5]
            .try_into()
            .map_err(|_| FrontendMessageError::TooShort)?,
    );
    if payload_len < 4 {
        return Err(FrontendMessageError::InvalidLengthField {
            declared: payload_len,
        });
    }
    let expected = payload_len as usize + 1;
    if expected != frame.len() {
        return Err(FrontendMessageError::LengthMismatch {
            expected,
            actual: frame.len(),
        });
    }

    let payload = &frame[5..];
    match tag {
        b'Q' => {
            let query_bytes =
                parse_cstring_payload(payload, FrontendMessageError::UnterminatedSimpleQuery)?;
            let query = std::str::from_utf8(query_bytes)
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();
            Ok(FrontendMessage::SimpleQuery(query))
        }
        b'p' => {
            if let Ok(password_bytes) =
                parse_cstring_payload(payload, FrontendMessageError::UnterminatedPasswordMessage)
            {
                let password = std::str::from_utf8(password_bytes)
                    .map_err(|_| FrontendMessageError::InvalidUtf8)?
                    .to_owned();
                return Ok(FrontendMessage::PasswordMessage(password));
            }

            if let Some(mechanism_end) = payload.iter().position(|&b| b == 0) {
                if mechanism_end == 0 {
                    return Err(FrontendMessageError::InvalidSaslInitialResponsePayload);
                }

                let mechanism = std::str::from_utf8(&payload[..mechanism_end])
                    .map_err(|_| FrontendMessageError::InvalidUtf8)?
                    .to_owned();

                let length_start = mechanism_end + 1;
                let length_end = length_start + 4;
                if payload.get(length_start..length_end).is_none() {
                    return Err(FrontendMessageError::InvalidSaslInitialResponsePayload);
                }

                let length = i32::from_be_bytes(
                    payload[length_start..length_end]
                        .try_into()
                        .map_err(|_| FrontendMessageError::InvalidSaslInitialResponsePayload)?,
                );

                let initial_response = if length == -1 {
                    if length_end != payload.len() {
                        return Err(FrontendMessageError::InvalidSaslInitialResponsePayload);
                    }
                    None
                } else {
                    if length < -1 {
                        return Err(FrontendMessageError::InvalidSaslInitialResponsePayload);
                    }
                    let length = length as usize;
                    let response_end = length_end
                        .checked_add(length)
                        .ok_or(FrontendMessageError::InvalidSaslInitialResponsePayload)?;
                    if response_end != payload.len() {
                        return Err(FrontendMessageError::InvalidSaslInitialResponsePayload);
                    }
                    Some(payload[length_end..response_end].to_vec())
                };

                return Ok(FrontendMessage::SaslInitialResponse {
                    mechanism,
                    initial_response,
                });
            }

            if payload.is_empty() {
                return Ok(FrontendMessage::SaslResponse(Vec::new()));
            }

            Ok(FrontendMessage::SaslResponse(payload.to_vec()))
        }
        b'B' => {
            let Some(portal_end) = payload.iter().position(|&b| b == 0) else {
                return Err(FrontendMessageError::UnterminatedBindPortalName);
            };
            let portal_name = std::str::from_utf8(&payload[..portal_end])
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();

            let statement_start = portal_end + 1;
            let Some(statement_rel_end) = payload[statement_start..].iter().position(|&b| b == 0)
            else {
                return Err(FrontendMessageError::UnterminatedBindStatementName);
            };
            let statement_end = statement_start + statement_rel_end;
            let statement_name = std::str::from_utf8(&payload[statement_start..statement_end])
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();

            let mut offset = statement_end + 1;
            let read_i16 =
                |payload: &[u8], offset: &mut usize| -> Result<i16, FrontendMessageError> {
                    let bytes = payload
                        .get(*offset..*offset + 2)
                        .ok_or(FrontendMessageError::InvalidBindPayload)?;
                    let value = i16::from_be_bytes(
                        bytes
                            .try_into()
                            .map_err(|_| FrontendMessageError::InvalidBindPayload)?,
                    );
                    *offset += 2;
                    Ok(value)
                };

            let format_count = read_i16(payload, &mut offset)?;
            if format_count < 0 {
                return Err(FrontendMessageError::InvalidBindPayload);
            }
            let format_count = format_count as usize;
            let mut parameter_format_codes = Vec::with_capacity(format_count);
            for _ in 0..format_count {
                let format_code = read_i16(payload, &mut offset)?;
                if !is_valid_format_code(format_code) {
                    return Err(FrontendMessageError::InvalidBindPayload);
                }
                parameter_format_codes.push(format_code);
            }

            let parameter_count = read_i16(payload, &mut offset)?;
            if parameter_count < 0 {
                return Err(FrontendMessageError::InvalidBindPayload);
            }
            let parameter_count = parameter_count as usize;
            if !parameter_format_codes.is_empty()
                && parameter_format_codes.len() != 1
                && parameter_format_codes.len() != parameter_count
            {
                return Err(FrontendMessageError::InvalidBindPayload);
            }
            let mut parameters = Vec::with_capacity(parameter_count);
            for _ in 0..parameter_count {
                let len_bytes = payload
                    .get(offset..offset + 4)
                    .ok_or(FrontendMessageError::InvalidBindPayload)?;
                let len = i32::from_be_bytes(
                    len_bytes
                        .try_into()
                        .map_err(|_| FrontendMessageError::InvalidBindPayload)?,
                );
                offset += 4;
                if len == -1 {
                    parameters.push(None);
                    continue;
                }
                if len < -1 {
                    return Err(FrontendMessageError::InvalidBindPayload);
                }
                let len = len as usize;
                let value = payload
                    .get(offset..offset + len)
                    .ok_or(FrontendMessageError::InvalidBindPayload)?;
                offset += len;
                parameters.push(Some(value.to_vec()));
            }

            let result_format_count = read_i16(payload, &mut offset)?;
            if result_format_count < 0 {
                return Err(FrontendMessageError::InvalidBindPayload);
            }
            let result_format_count = result_format_count as usize;
            let mut result_format_codes = Vec::with_capacity(result_format_count);
            for _ in 0..result_format_count {
                let format_code = read_i16(payload, &mut offset)?;
                if !is_valid_format_code(format_code) {
                    return Err(FrontendMessageError::InvalidBindPayload);
                }
                result_format_codes.push(format_code);
            }

            if offset != payload.len() {
                return Err(FrontendMessageError::InvalidBindPayload);
            }

            Ok(FrontendMessage::Bind {
                portal_name,
                statement_name,
                parameter_format_codes,
                parameters,
                result_format_codes,
            })
        }
        b'P' => {
            let Some(statement_name_end) = payload.iter().position(|&b| b == 0) else {
                return Err(FrontendMessageError::UnterminatedParseStatementName);
            };
            let statement_name = std::str::from_utf8(&payload[..statement_name_end])
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();

            let query_start = statement_name_end + 1;
            let Some(query_rel_end) = payload[query_start..].iter().position(|&b| b == 0) else {
                return Err(FrontendMessageError::UnterminatedParseQuery);
            };
            let query_end = query_start + query_rel_end;
            let query = std::str::from_utf8(&payload[query_start..query_end])
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();

            let type_count_start = query_end + 1;
            let type_count_end = type_count_start + 2;
            let Some(type_count_bytes) = payload.get(type_count_start..type_count_end) else {
                return Err(FrontendMessageError::InvalidParseParameterPayload);
            };
            let type_count = i16::from_be_bytes(
                type_count_bytes
                    .try_into()
                    .map_err(|_| FrontendMessageError::InvalidParseParameterPayload)?,
            );
            if type_count < 0 {
                return Err(FrontendMessageError::InvalidParseParameterPayload);
            }
            let type_count = type_count as usize;

            let types_start = type_count_end;
            let expected_types_len = type_count
                .checked_mul(4)
                .ok_or(FrontendMessageError::InvalidParseParameterPayload)?;
            let types_end = types_start
                .checked_add(expected_types_len)
                .ok_or(FrontendMessageError::InvalidParseParameterPayload)?;
            if types_end != payload.len() {
                return Err(FrontendMessageError::InvalidParseParameterPayload);
            }

            let mut parameter_type_oids = Vec::with_capacity(type_count);
            for chunk in payload[types_start..types_end].chunks_exact(4) {
                let oid = u32::from_be_bytes(
                    chunk
                        .try_into()
                        .map_err(|_| FrontendMessageError::InvalidParseParameterPayload)?,
                );
                parameter_type_oids.push(oid);
            }

            Ok(FrontendMessage::Parse {
                statement_name,
                query,
                parameter_type_oids,
            })
        }
        b'D' => {
            let Some((&target_byte, rest)) = payload.split_first() else {
                return Err(FrontendMessageError::TooShort);
            };
            let target = match target_byte {
                b'S' | b's' => DescribeTarget::Statement,
                b'P' | b'p' => DescribeTarget::Portal,
                _ => return Err(FrontendMessageError::InvalidDescribeTarget),
            };

            let name_bytes =
                parse_cstring_payload(rest, FrontendMessageError::UnterminatedDescribeName)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();
            Ok(FrontendMessage::Describe { target, name })
        }
        b'C' => {
            let Some((&target_byte, rest)) = payload.split_first() else {
                return Err(FrontendMessageError::TooShort);
            };
            let target = match target_byte {
                b'S' | b's' => DescribeTarget::Statement,
                b'P' | b'p' => DescribeTarget::Portal,
                _ => return Err(FrontendMessageError::InvalidCloseTarget),
            };

            let name_bytes =
                parse_cstring_payload(rest, FrontendMessageError::UnterminatedCloseName)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();
            Ok(FrontendMessage::Close { target, name })
        }
        b'E' => {
            let Some(portal_end) = payload.iter().position(|&b| b == 0) else {
                return Err(FrontendMessageError::UnterminatedExecutePortalName);
            };
            let portal_name = std::str::from_utf8(&payload[..portal_end])
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();

            let max_rows_start = portal_end + 1;
            let max_rows_bytes = payload
                .get(max_rows_start..max_rows_start + 4)
                .ok_or(FrontendMessageError::InvalidExecutePayload)?;
            if max_rows_start + 4 != payload.len() {
                return Err(FrontendMessageError::InvalidExecutePayload);
            }
            let max_rows = i32::from_be_bytes(
                max_rows_bytes
                    .try_into()
                    .map_err(|_| FrontendMessageError::InvalidExecutePayload)?,
            );
            if max_rows < 0 {
                return Err(FrontendMessageError::InvalidExecutePayload);
            }

            Ok(FrontendMessage::Execute {
                portal_name,
                max_rows: max_rows as u32,
            })
        }
        b'F' => {
            let mut offset = 0;

            let function_oid = {
                let bytes = payload
                    .get(offset..offset + 4)
                    .ok_or(FrontendMessageError::InvalidFunctionCallPayload)?;
                offset += 4;
                u32::from_be_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| FrontendMessageError::InvalidFunctionCallPayload)?,
                )
            };

            let read_i16 =
                |payload: &[u8], offset: &mut usize| -> Result<i16, FrontendMessageError> {
                    let bytes = payload
                        .get(*offset..*offset + 2)
                        .ok_or(FrontendMessageError::InvalidFunctionCallPayload)?;
                    *offset += 2;
                    Ok(i16::from_be_bytes(bytes.try_into().map_err(|_| {
                        FrontendMessageError::InvalidFunctionCallPayload
                    })?))
                };

            let format_count = read_i16(payload, &mut offset)?;
            if format_count < 0 {
                return Err(FrontendMessageError::InvalidFunctionCallPayload);
            }
            let format_count = format_count as usize;
            let mut argument_format_codes = Vec::with_capacity(format_count);
            for _ in 0..format_count {
                let format_code = read_i16(payload, &mut offset)?;
                if !is_valid_format_code(format_code) {
                    return Err(FrontendMessageError::InvalidFunctionCallPayload);
                }
                argument_format_codes.push(format_code);
            }

            let arg_count = read_i16(payload, &mut offset)?;
            if arg_count < 0 {
                return Err(FrontendMessageError::InvalidFunctionCallPayload);
            }
            let arg_count = arg_count as usize;
            if !argument_format_codes.is_empty()
                && argument_format_codes.len() != 1
                && argument_format_codes.len() != arg_count
            {
                return Err(FrontendMessageError::InvalidFunctionCallPayload);
            }
            let mut arguments = Vec::with_capacity(arg_count);
            for _ in 0..arg_count {
                let len_bytes = payload
                    .get(offset..offset + 4)
                    .ok_or(FrontendMessageError::InvalidFunctionCallPayload)?;
                let len = i32::from_be_bytes(
                    len_bytes
                        .try_into()
                        .map_err(|_| FrontendMessageError::InvalidFunctionCallPayload)?,
                );
                offset += 4;
                if len == -1 {
                    arguments.push(None);
                    continue;
                }
                if len < -1 {
                    return Err(FrontendMessageError::InvalidFunctionCallPayload);
                }
                let len = len as usize;
                let value = payload
                    .get(offset..offset + len)
                    .ok_or(FrontendMessageError::InvalidFunctionCallPayload)?;
                offset += len;
                arguments.push(Some(value.to_vec()));
            }

            let result_format_code = read_i16(payload, &mut offset)?;
            if !is_valid_format_code(result_format_code) {
                return Err(FrontendMessageError::InvalidFunctionCallPayload);
            }
            if offset != payload.len() {
                return Err(FrontendMessageError::InvalidFunctionCallPayload);
            }

            Ok(FrontendMessage::FunctionCall {
                function_oid,
                argument_format_codes,
                arguments,
                result_format_code,
            })
        }
        b'd' => Ok(FrontendMessage::CopyData(payload.to_vec())),
        b'c' => {
            if payload_len != 4 {
                return Err(FrontendMessageError::LengthMismatch {
                    expected: 5,
                    actual: frame.len(),
                });
            }
            Ok(FrontendMessage::CopyDone)
        }
        b'f' => {
            let reason_bytes =
                parse_cstring_payload(payload, FrontendMessageError::UnterminatedCopyFail)?;
            let reason = std::str::from_utf8(reason_bytes)
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();
            Ok(FrontendMessage::CopyFail(reason))
        }
        b'X' => {
            if payload_len != 4 {
                return Err(FrontendMessageError::LengthMismatch {
                    expected: 5,
                    actual: frame.len(),
                });
            }
            Ok(FrontendMessage::Terminate)
        }
        b'S' => {
            if payload_len != 4 {
                return Err(FrontendMessageError::LengthMismatch {
                    expected: 5,
                    actual: frame.len(),
                });
            }
            Ok(FrontendMessage::Sync)
        }
        b'H' => {
            if payload_len != 4 {
                return Err(FrontendMessageError::LengthMismatch {
                    expected: 5,
                    actual: frame.len(),
                });
            }
            Ok(FrontendMessage::Flush)
        }
        other => Err(FrontendMessageError::UnsupportedTag(other)),
    }
}

impl Default for SessionLifecycle {
    fn default() -> Self {
        Self {
            state: SessionState::Startup,
        }
    }
}

impl SessionLifecycle {
    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn apply(&mut self, event: SessionEvent) -> Result<SessionState, SessionTransitionError> {
        let next = match (self.state, event) {
            (SessionState::Startup, SessionEvent::StartupAccepted) => SessionState::Authenticating,
            (SessionState::Authenticating, SessionEvent::AuthenticationSucceeded) => {
                SessionState::Ready
            }
            (SessionState::Ready, SessionEvent::Begin) => SessionState::InTransaction,
            (SessionState::InTransaction, SessionEvent::Commit)
            | (SessionState::InTransaction, SessionEvent::Rollback) => SessionState::Ready,
            (SessionState::Ready, SessionEvent::TerminateRequested)
            | (SessionState::InTransaction, SessionEvent::TerminateRequested) => {
                SessionState::Terminating
            }
            (SessionState::Terminating, SessionEvent::ConnectionClosed) => SessionState::Closed,
            (SessionState::Startup, SessionEvent::ConnectionClosed)
            | (SessionState::Authenticating, SessionEvent::ConnectionClosed)
            | (SessionState::Ready, SessionEvent::ConnectionClosed)
            | (SessionState::InTransaction, SessionEvent::ConnectionClosed) => SessionState::Closed,
            _ => {
                return Err(SessionTransitionError::InvalidTransition {
                    from: self.state,
                    event,
                });
            }
        };

        self.state = next;
        Ok(next)
    }
}

impl Default for ReadyLoopState {
    fn default() -> Self {
        Self {
            transaction_status: TransactionStatus::Idle,
            skip_until_sync: false,
        }
    }
}

impl ReadyLoopState {
    pub fn from_flags(in_transaction: bool, skip_until_sync: bool) -> Self {
        Self {
            transaction_status: if in_transaction {
                TransactionStatus::InTransaction
            } else {
                TransactionStatus::Idle
            },
            skip_until_sync,
        }
    }

    pub fn in_transaction(&self) -> bool {
        self.transaction_status.ready_for_query_in_transaction()
    }

    pub fn skip_until_sync(&self) -> bool {
        self.skip_until_sync
    }

    pub fn should_dispatch_extended_message(&self) -> bool {
        !self.skip_until_sync
    }

    pub fn set_transaction_status(&mut self, status: TransactionStatus) {
        self.transaction_status = status;
    }

    pub fn mark_extended_error(&mut self) {
        self.skip_until_sync = true;
    }

    pub fn clear_extended_error_on_sync(&mut self, copy_stream_active: bool) -> bool {
        if copy_stream_active {
            return false;
        }
        self.skip_until_sync = false;
        true
    }
}

pub mod backend {
    use std::io::{self, ErrorKind, Write};

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BackendColumn {
        pub name: String,
        pub oid: u32,
        pub type_size: i16,
    }

    impl BackendColumn {
        pub fn new(name: impl Into<String>, oid: u32, type_size: i16) -> Self {
            Self {
                name: name.into(),
                oid,
                type_size,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BackendError {
        pub code: String,
        pub message: String,
        pub position: Option<String>,
    }

    impl BackendError {
        pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
            Self {
                code: code.into(),
                message: message.into(),
                position: None,
            }
        }

        pub fn with_position(
            code: impl Into<String>,
            message: impl Into<String>,
            position: impl Into<String>,
        ) -> Self {
            Self {
                code: code.into(),
                message: message.into(),
                position: Some(position.into()),
            }
        }
    }

    pub struct BackendWriter<'a, W: Write + ?Sized> {
        inner: &'a mut W,
    }

    impl<'a, W: Write + ?Sized> BackendWriter<'a, W> {
        pub fn new(inner: &'a mut W) -> Self {
            Self { inner }
        }

        pub fn authentication_ok(&mut self) -> io::Result<()> {
            self.message(b'R', &0_i32.to_be_bytes())
        }

        pub fn authentication_sasl(&mut self, mechanisms: &[&str]) -> io::Result<()> {
            let mut payload = 10_i32.to_be_bytes().to_vec();
            for mechanism in mechanisms {
                push_cstring(&mut payload, mechanism);
            }
            payload.push(0);
            self.message(b'R', &payload)
        }

        pub fn authentication_sasl_continue(&mut self, data: &[u8]) -> io::Result<()> {
            let mut payload = 11_i32.to_be_bytes().to_vec();
            payload.extend_from_slice(data);
            self.message(b'R', &payload)
        }

        pub fn authentication_sasl_final(&mut self, data: &[u8]) -> io::Result<()> {
            let mut payload = 12_i32.to_be_bytes().to_vec();
            payload.extend_from_slice(data);
            self.message(b'R', &payload)
        }

        pub fn backend_key_data(&mut self, process_id: i32, secret_key: i32) -> io::Result<()> {
            let mut payload = Vec::with_capacity(8);
            payload.extend_from_slice(&process_id.to_be_bytes());
            payload.extend_from_slice(&secret_key.to_be_bytes());
            self.message(b'K', &payload)
        }

        pub fn parameter_status(&mut self, key: &str, value: &str) -> io::Result<()> {
            let mut payload = Vec::with_capacity(key.len() + value.len() + 2);
            push_cstring(&mut payload, key);
            push_cstring(&mut payload, value);
            self.message(b'S', &payload)
        }

        pub fn ready_for_query(&mut self, in_transaction: bool) -> io::Result<()> {
            let status = if in_transaction { b'T' } else { b'I' };
            self.message(b'Z', &[status])
        }

        pub fn empty_query_response(&mut self) -> io::Result<()> {
            self.message(b'I', &[])
        }

        pub fn command_complete(&mut self, tag: &str) -> io::Result<()> {
            let mut payload = Vec::with_capacity(tag.len() + 1);
            push_cstring(&mut payload, tag);
            self.message(b'C', &payload)
        }

        pub fn parse_complete(&mut self) -> io::Result<()> {
            self.message(b'1', &[])
        }

        pub fn bind_complete(&mut self) -> io::Result<()> {
            self.message(b'2', &[])
        }

        pub fn close_complete(&mut self) -> io::Result<()> {
            self.message(b'3', &[])
        }

        pub fn portal_suspended(&mut self) -> io::Result<()> {
            self.message(b's', &[])
        }

        pub fn no_data(&mut self) -> io::Result<()> {
            self.message(b'n', &[])
        }

        pub fn copy_out_response(&mut self, column_count: usize) -> io::Result<()> {
            self.copy_response(b'H', column_count)
        }

        pub fn copy_in_response(&mut self, column_count: usize) -> io::Result<()> {
            self.copy_response(b'G', column_count)
        }

        fn copy_response(&mut self, tag: u8, column_count: usize) -> io::Result<()> {
            let column_count = i16::try_from(column_count)
                .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "too many COPY columns"))?;
            let mut payload = Vec::with_capacity(1 + 2 + column_count as usize * 2);
            payload.push(0);
            payload.extend_from_slice(&column_count.to_be_bytes());
            for _ in 0..column_count {
                payload.extend_from_slice(&0_i16.to_be_bytes());
            }
            self.message(tag, &payload)
        }

        pub fn copy_data(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.message(b'd', bytes)
        }

        pub fn copy_done(&mut self) -> io::Result<()> {
            self.message(b'c', &[])
        }

        pub fn parameter_description(&mut self, type_oids: &[u32]) -> io::Result<()> {
            let parameter_count = i16::try_from(type_oids.len())
                .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "too many parameters"))?;
            let mut payload = Vec::with_capacity(2 + type_oids.len() * 4);
            payload.extend_from_slice(&parameter_count.to_be_bytes());
            for oid in type_oids {
                payload.extend_from_slice(&oid.to_be_bytes());
            }
            self.message(b't', &payload)
        }

        pub fn select_rows(
            &mut self,
            columns: &[BackendColumn],
            rows: &[Vec<Option<String>>],
            include_row_description: bool,
        ) -> io::Result<()> {
            self.rows_with_tag(
                columns,
                rows,
                include_row_description,
                &format!("SELECT {}", rows.len()),
            )
        }

        pub fn rows_with_tag(
            &mut self,
            columns: &[BackendColumn],
            rows: &[Vec<Option<String>>],
            include_row_description: bool,
            tag: &str,
        ) -> io::Result<()> {
            if include_row_description {
                self.row_description(columns)?;
            }
            for row in rows {
                self.data_row(row)?;
            }
            self.command_complete(tag)
        }

        pub fn row_description(&mut self, columns: &[BackendColumn]) -> io::Result<()> {
            self.row_description_with_formats(columns, &[])
        }

        pub fn row_description_with_formats(
            &mut self,
            columns: &[BackendColumn],
            result_format_codes: &[i16],
        ) -> io::Result<()> {
            let field_count = i16::try_from(columns.len())
                .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "too many columns"))?;
            let mut payload = Vec::new();
            payload.extend_from_slice(&field_count.to_be_bytes());
            for (idx, column) in columns.iter().enumerate() {
                push_cstring(&mut payload, &column.name);
                payload.extend_from_slice(&0_u32.to_be_bytes());
                payload.extend_from_slice(&0_i16.to_be_bytes());
                payload.extend_from_slice(&column.oid.to_be_bytes());
                payload.extend_from_slice(&column.type_size.to_be_bytes());
                payload.extend_from_slice(&(-1_i32).to_be_bytes());
                payload.extend_from_slice(&format_code_at(result_format_codes, idx).to_be_bytes());
            }
            self.message(b'T', &payload)
        }

        pub fn data_row(&mut self, values: &[Option<String>]) -> io::Result<()> {
            let columns = values
                .iter()
                .map(|_| {
                    BackendColumn::new(
                        "",
                        crate::SqlType::Text.postgres_oid(),
                        crate::SqlType::Text.type_size(),
                    )
                })
                .collect::<Vec<_>>();
            self.data_row_with_formats(&columns, values, &[])
        }

        pub fn data_row_with_formats(
            &mut self,
            columns: &[BackendColumn],
            values: &[Option<String>],
            result_format_codes: &[i16],
        ) -> io::Result<()> {
            let value_count = i16::try_from(values.len())
                .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "too many row values"))?;
            let mut payload = Vec::new();
            payload.extend_from_slice(&value_count.to_be_bytes());
            for (idx, value) in values.iter().enumerate() {
                match value {
                    Some(value) => {
                        let encoded;
                        let bytes = if format_code_at(result_format_codes, idx) == 1 {
                            encoded = encode_binary_result_value(value, columns[idx].oid)?;
                            encoded.as_slice()
                        } else {
                            value.as_bytes()
                        };
                        let len = i32::try_from(bytes.len()).map_err(|_| {
                            io::Error::new(ErrorKind::InvalidInput, "row value too large to encode")
                        })?;
                        payload.extend_from_slice(&len.to_be_bytes());
                        payload.extend_from_slice(bytes);
                    }
                    None => payload.extend_from_slice(&(-1_i32).to_be_bytes()),
                }
            }
            self.message(b'D', &payload)
        }

        pub fn error_response(&mut self, error: &BackendError) -> io::Result<()> {
            let mut payload = Vec::new();
            push_error_field(&mut payload, b'S', "ERROR");
            push_error_field(&mut payload, b'V', "ERROR");
            push_error_field(&mut payload, b'C', &error.code);
            push_error_field(&mut payload, b'M', &error.message);
            if let Some(position) = &error.position {
                push_error_field(&mut payload, b'P', position);
            }
            payload.push(0);
            self.message(b'E', &payload)
        }

        pub fn message(&mut self, tag: u8, payload: &[u8]) -> io::Result<()> {
            write_message(self.inner, tag, payload)
        }
    }

    pub fn write_message<W: Write + ?Sized>(
        stream: &mut W,
        tag: u8,
        payload: &[u8],
    ) -> io::Result<()> {
        let total_len = i32::try_from(payload.len() + 4)
            .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "payload too large"))?;
        stream.write_all(&[tag])?;
        stream.write_all(&total_len.to_be_bytes())?;
        stream.write_all(payload)
    }

    fn encode_binary_result_value(value: &str, type_oid: u32) -> io::Result<Vec<u8>> {
        match type_oid {
            23 => {
                let value = value.parse::<i32>().map_err(|_| {
                    io::Error::new(
                        ErrorKind::InvalidInput,
                        "int4 row value cannot be encoded as binary",
                    )
                })?;
                Ok(value.to_be_bytes().to_vec())
            }
            25 => Ok(value.as_bytes().to_vec()),
            _ => Err(io::Error::new(
                ErrorKind::InvalidInput,
                "unsupported binary result type",
            )),
        }
    }

    fn format_code_at(format_codes: &[i16], idx: usize) -> i16 {
        match format_codes {
            [] => 0,
            [code] => *code,
            codes => codes[idx],
        }
    }

    fn push_error_field(payload: &mut Vec<u8>, tag: u8, value: &str) {
        payload.push(tag);
        push_cstring(payload, value);
    }

    fn push_cstring(payload: &mut Vec<u8>, value: &str) {
        payload.extend_from_slice(value.as_bytes());
        payload.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend_messages(bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
        let mut messages = Vec::new();
        let mut idx = 0;
        while idx < bytes.len() {
            let tag = bytes[idx];
            idx += 1;
            let len = u32::from_be_bytes(bytes[idx..idx + 4].try_into().unwrap()) as usize;
            idx += 4;
            let payload_len = len - 4;
            let payload = bytes[idx..idx + payload_len].to_vec();
            idx += payload_len;
            messages.push((tag, payload));
        }
        messages
    }

    #[test]
    fn backend_writer_emits_reusable_startup_result_copy_and_error_messages() {
        let mut output = Vec::new();
        {
            let columns = vec![
                backend::BackendColumn::new(
                    "id",
                    SqlType::Int4.postgres_oid(),
                    SqlType::Int4.type_size(),
                ),
                backend::BackendColumn::new(
                    "name",
                    SqlType::Text.postgres_oid(),
                    SqlType::Text.type_size(),
                ),
            ];
            let mut writer = backend::BackendWriter::new(&mut output);
            writer.authentication_ok().unwrap();
            writer.parameter_status("server_version", "16.0").unwrap();
            writer.backend_key_data(1, 1).unwrap();
            writer.ready_for_query(false).unwrap();
            writer
                .parameter_description(&[SqlType::Int4.postgres_oid()])
                .unwrap();
            writer
                .row_description_with_formats(&columns, &[1, 0])
                .unwrap();
            writer
                .data_row_with_formats(
                    &columns,
                    &[Some("7".to_string()), Some("Ada".to_string())],
                    &[1, 0],
                )
                .unwrap();
            writer.command_complete("SELECT 1").unwrap();
            writer.copy_in_response(2).unwrap();
            writer.copy_out_response(2).unwrap();
            writer.copy_data(b"7,Ada\n").unwrap();
            writer.copy_done().unwrap();
            writer
                .error_response(&backend::BackendError::with_position(
                    "42601",
                    "syntax error",
                    "8",
                ))
                .unwrap();
        }

        let messages = backend_messages(&output);
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'R', b'S', b'K', b'Z', b't', b'T', b'D', b'C', b'G', b'H', b'd', b'c', b'E']
        );
        assert_eq!(messages[0].1, 0_i32.to_be_bytes().to_vec());
        assert_eq!(messages[3].1, vec![b'I']);
        assert_eq!(messages[6].1[6..10], 7_i32.to_be_bytes());
        assert!(messages[12].1.windows(5).any(|window| window == b"42601"));
    }

    #[path = "session_commands.rs"]
    mod session_commands;
    #[path = "transaction_commands.rs"]
    mod transaction_commands;

    #[path = "control_command_rejections.rs"]
    mod control_command_rejections;

    #[path = "command_terminators.rs"]
    mod command_terminators;

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

    #[path = "startup.rs"]
    mod startup;

    #[test]
    fn session_lifecycle_follows_startup_auth_and_transaction_flow() {
        let mut session = SessionLifecycle::default();
        assert_eq!(session.state(), SessionState::Startup);

        assert_eq!(
            session.apply(SessionEvent::StartupAccepted).unwrap(),
            SessionState::Authenticating
        );
        assert_eq!(
            session
                .apply(SessionEvent::AuthenticationSucceeded)
                .unwrap(),
            SessionState::Ready
        );
        assert_eq!(
            session.apply(SessionEvent::Begin).unwrap(),
            SessionState::InTransaction
        );
        assert_eq!(
            session.apply(SessionEvent::Commit).unwrap(),
            SessionState::Ready
        );
        assert_eq!(
            session.apply(SessionEvent::TerminateRequested).unwrap(),
            SessionState::Terminating
        );
        assert_eq!(
            session.apply(SessionEvent::ConnectionClosed).unwrap(),
            SessionState::Closed
        );
    }

    #[test]
    fn session_lifecycle_rejects_invalid_transitions() {
        let mut session = SessionLifecycle::default();
        assert_eq!(
            session.apply(SessionEvent::Begin).unwrap_err(),
            SessionTransitionError::InvalidTransition {
                from: SessionState::Startup,
                event: SessionEvent::Begin,
            }
        );

        session.apply(SessionEvent::StartupAccepted).unwrap();
        session
            .apply(SessionEvent::AuthenticationSucceeded)
            .unwrap();
        assert_eq!(
            session.apply(SessionEvent::Commit).unwrap_err(),
            SessionTransitionError::InvalidTransition {
                from: SessionState::Ready,
                event: SessionEvent::Commit,
            }
        );
    }

    #[test]
    fn ready_loop_state_tracks_sync_recovery_and_transaction_status() {
        let mut ready_loop = ReadyLoopState::default();
        assert!(ready_loop.should_dispatch_extended_message());
        assert!(!ready_loop.in_transaction());

        ready_loop.set_transaction_status(TransactionStatus::InTransaction);
        assert!(ready_loop.in_transaction());

        ready_loop.mark_extended_error();
        assert!(ready_loop.skip_until_sync());
        assert!(!ready_loop.should_dispatch_extended_message());

        assert!(!ready_loop.clear_extended_error_on_sync(true));
        assert!(ready_loop.skip_until_sync());

        assert!(ready_loop.clear_extended_error_on_sync(false));
        assert!(!ready_loop.skip_until_sync());
        assert!(ready_loop.should_dispatch_extended_message());
        assert!(ready_loop.in_transaction());

        let from_flags = ReadyLoopState::from_flags(false, true);
        assert!(!from_flags.in_transaction());
        assert!(from_flags.skip_until_sync());
    }

    fn frontend_frame(tag: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(payload.len() + 5);
        frame.push(tag);
        frame.extend_from_slice(&((payload.len() + 4) as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }

    #[test]
    fn parses_simple_query_and_control_frontend_messages() {
        let query = frontend_frame(b'Q', b"SELECT 1;\0");
        assert_eq!(
            parse_frontend_message(&query).unwrap(),
            FrontendMessage::SimpleQuery("SELECT 1;".to_string())
        );

        let empty_query = frontend_frame(b'Q', b"\0");
        assert_eq!(
            parse_frontend_message(&empty_query).unwrap(),
            FrontendMessage::SimpleQuery(String::new())
        );

        let utf8_query = frontend_frame(b'Q', "SELECT 'héllo';\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&utf8_query).unwrap(),
            FrontendMessage::SimpleQuery("SELECT 'héllo';".to_string())
        );

        let multiline_utf8_query = frontend_frame(b'Q', "SELECT 'héllo'\nFROM tést;\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&multiline_utf8_query).unwrap(),
            FrontendMessage::SimpleQuery("SELECT 'héllo'\nFROM tést;".to_string())
        );

        let password = frontend_frame(b'p', b"secret\0");
        assert_eq!(
            parse_frontend_message(&password).unwrap(),
            FrontendMessage::PasswordMessage("secret".to_string())
        );

        let empty_password = frontend_frame(b'p', b"\0");
        assert_eq!(
            parse_frontend_message(&empty_password).unwrap(),
            FrontendMessage::PasswordMessage(String::new())
        );

        let utf8_password = frontend_frame(b'p', "påsswörd\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&utf8_password).unwrap(),
            FrontendMessage::PasswordMessage("påsswörd".to_string())
        );

        let multiline_utf8_password = frontend_frame(b'p', "påss\nwördΔ\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&multiline_utf8_password).unwrap(),
            FrontendMessage::PasswordMessage("påss\nwördΔ".to_string())
        );

        let mut sasl_initial_payload = Vec::new();
        sasl_initial_payload.extend_from_slice(b"SCRAM-SHA-256\0");
        sasl_initial_payload.extend_from_slice(&5_i32.to_be_bytes());
        sasl_initial_payload.extend_from_slice(b"n,,r=");
        let sasl_initial = frontend_frame(b'p', &sasl_initial_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRAM-SHA-256".to_string(),
                initial_response: Some(b"n,,r=".to_vec()),
            }
        );

        let mut sasl_initial_without_data_payload = Vec::new();
        sasl_initial_without_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
        sasl_initial_without_data_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        let sasl_initial_without_data = frontend_frame(b'p', &sasl_initial_without_data_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial_without_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRAM-SHA-256".to_string(),
                initial_response: None,
            }
        );

        let mut sasl_initial_with_empty_data_payload = Vec::new();
        sasl_initial_with_empty_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
        sasl_initial_with_empty_data_payload.extend_from_slice(&0_i32.to_be_bytes());
        let sasl_initial_with_empty_data =
            frontend_frame(b'p', &sasl_initial_with_empty_data_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_empty_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRAM-SHA-256".to_string(),
                initial_response: Some(Vec::new()),
            }
        );

        let mut sasl_initial_with_binary_data_payload = Vec::new();
        sasl_initial_with_binary_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
        sasl_initial_with_binary_data_payload.extend_from_slice(&3_i32.to_be_bytes());
        sasl_initial_with_binary_data_payload.extend_from_slice(&[0x00, 0xFF, 0x00]);
        let sasl_initial_with_binary_data =
            frontend_frame(b'p', &sasl_initial_with_binary_data_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_binary_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRAM-SHA-256".to_string(),
                initial_response: Some(vec![0x00, 0xFF, 0x00]),
            }
        );

        let mut sasl_initial_with_utf8_data_payload = Vec::new();
        sasl_initial_with_utf8_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
        sasl_initial_with_utf8_data_payload
            .extend_from_slice(&("n,,r=nönce".len() as i32).to_be_bytes());
        sasl_initial_with_utf8_data_payload.extend_from_slice("n,,r=nönce".as_bytes());
        let sasl_initial_with_utf8_data =
            frontend_frame(b'p', &sasl_initial_with_utf8_data_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_utf8_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRAM-SHA-256".to_string(),
                initial_response: Some("n,,r=nönce".as_bytes().to_vec()),
            }
        );

        let mut sasl_initial_with_multiline_utf8_data_payload = Vec::new();
        sasl_initial_with_multiline_utf8_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
        sasl_initial_with_multiline_utf8_data_payload
            .extend_from_slice(&("n,,r=nönce\nΔetail".len() as i32).to_be_bytes());
        sasl_initial_with_multiline_utf8_data_payload
            .extend_from_slice("n,,r=nönce\nΔetail".as_bytes());
        let sasl_initial_with_multiline_utf8_data =
            frontend_frame(b'p', &sasl_initial_with_multiline_utf8_data_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_multiline_utf8_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRAM-SHA-256".to_string(),
                initial_response: Some("n,,r=nönce\nΔetail".as_bytes().to_vec()),
            }
        );

        let mut sasl_initial_with_mixed_utf8_binary_data_payload = Vec::new();
        sasl_initial_with_mixed_utf8_binary_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
        sasl_initial_with_mixed_utf8_binary_data_payload.extend_from_slice(&5_i32.to_be_bytes());
        sasl_initial_with_mixed_utf8_binary_data_payload
            .extend_from_slice(&[b'n', 0xC3, 0xB8, 0xFF, b'!']);
        let sasl_initial_with_mixed_utf8_binary_data =
            frontend_frame(b'p', &sasl_initial_with_mixed_utf8_binary_data_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_mixed_utf8_binary_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRAM-SHA-256".to_string(),
                initial_response: Some(vec![b'n', 0xC3, 0xB8, 0xFF, b'!']),
            }
        );

        let mut sasl_initial_with_utf8_mechanism_payload = Vec::new();
        sasl_initial_with_utf8_mechanism_payload.extend_from_slice("SCRÄM\0".as_bytes());
        sasl_initial_with_utf8_mechanism_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        let sasl_initial_with_utf8_mechanism =
            frontend_frame(b'p', &sasl_initial_with_utf8_mechanism_payload);
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_utf8_mechanism).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRÄM".to_string(),
                initial_response: None,
            }
        );

        let mut sasl_initial_with_utf8_mechanism_and_binary_data_payload = Vec::new();
        sasl_initial_with_utf8_mechanism_and_binary_data_payload
            .extend_from_slice("SCRÄM\0".as_bytes());
        sasl_initial_with_utf8_mechanism_and_binary_data_payload
            .extend_from_slice(&3_i32.to_be_bytes());
        sasl_initial_with_utf8_mechanism_and_binary_data_payload
            .extend_from_slice(&[0x00, 0xFF, 0x7F]);
        let sasl_initial_with_utf8_mechanism_and_binary_data = frontend_frame(
            b'p',
            &sasl_initial_with_utf8_mechanism_and_binary_data_payload,
        );
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_binary_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRÄM".to_string(),
                initial_response: Some(vec![0x00, 0xFF, 0x7F]),
            }
        );

        let mut sasl_initial_with_utf8_mechanism_and_empty_data_payload = Vec::new();
        sasl_initial_with_utf8_mechanism_and_empty_data_payload
            .extend_from_slice("SCRÄM\0".as_bytes());
        sasl_initial_with_utf8_mechanism_and_empty_data_payload
            .extend_from_slice(&0_i32.to_be_bytes());
        let sasl_initial_with_utf8_mechanism_and_empty_data = frontend_frame(
            b'p',
            &sasl_initial_with_utf8_mechanism_and_empty_data_payload,
        );
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_empty_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRÄM".to_string(),
                initial_response: Some(Vec::new()),
            }
        );

        let mut sasl_initial_with_utf8_mechanism_and_utf8_data_payload = Vec::new();
        sasl_initial_with_utf8_mechanism_and_utf8_data_payload
            .extend_from_slice("SCRÄM\0".as_bytes());
        sasl_initial_with_utf8_mechanism_and_utf8_data_payload
            .extend_from_slice(&("n,,r=nönce".len() as i32).to_be_bytes());
        sasl_initial_with_utf8_mechanism_and_utf8_data_payload
            .extend_from_slice("n,,r=nönce".as_bytes());
        let sasl_initial_with_utf8_mechanism_and_utf8_data = frontend_frame(
            b'p',
            &sasl_initial_with_utf8_mechanism_and_utf8_data_payload,
        );
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_utf8_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRÄM".to_string(),
                initial_response: Some("n,,r=nönce".as_bytes().to_vec()),
            }
        );

        let mut sasl_initial_with_utf8_mechanism_and_mixed_data_payload = Vec::new();
        sasl_initial_with_utf8_mechanism_and_mixed_data_payload
            .extend_from_slice("SCRÄM\0".as_bytes());
        sasl_initial_with_utf8_mechanism_and_mixed_data_payload
            .extend_from_slice(&5_i32.to_be_bytes());
        sasl_initial_with_utf8_mechanism_and_mixed_data_payload
            .extend_from_slice(&[b'n', 0xC3, 0xB8, 0xFF, b'!']);
        let sasl_initial_with_utf8_mechanism_and_mixed_data = frontend_frame(
            b'p',
            &sasl_initial_with_utf8_mechanism_and_mixed_data_payload,
        );
        assert_eq!(
            parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_mixed_data).unwrap(),
            FrontendMessage::SaslInitialResponse {
                mechanism: "SCRÄM".to_string(),
                initial_response: Some(vec![b'n', 0xC3, 0xB8, 0xFF, b'!']),
            }
        );

        let sasl_response = frontend_frame(b'p', b"c=biws,r=nonce,p=proof");
        assert_eq!(
            parse_frontend_message(&sasl_response).unwrap(),
            FrontendMessage::SaslResponse(b"c=biws,r=nonce,p=proof".to_vec())
        );

        let empty_sasl_response = frontend_frame(b'p', b"");
        assert_eq!(
            parse_frontend_message(&empty_sasl_response).unwrap(),
            FrontendMessage::SaslResponse(Vec::new())
        );

        let utf8_sasl_response = frontend_frame(b'p', "prøöf".as_bytes());
        assert_eq!(
            parse_frontend_message(&utf8_sasl_response).unwrap(),
            FrontendMessage::SaslResponse("prøöf".as_bytes().to_vec())
        );

        let multiline_utf8_sasl_response =
            frontend_frame(b'p', "c=biws\nr=noncé\np=prøöf".as_bytes());
        assert_eq!(
            parse_frontend_message(&multiline_utf8_sasl_response).unwrap(),
            FrontendMessage::SaslResponse("c=biws\nr=noncé\np=prøöf".as_bytes().to_vec())
        );

        let mut parse_payload = Vec::new();
        parse_payload.extend_from_slice(b"stmt1\0SELECT $1::int4\0");
        parse_payload.extend_from_slice(&1_i16.to_be_bytes());
        parse_payload.extend_from_slice(&23_u32.to_be_bytes());
        let parse = frontend_frame(b'P', &parse_payload);
        assert_eq!(
            parse_frontend_message(&parse).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmt1".to_string(),
                query: "SELECT $1::int4".to_string(),
                parameter_type_oids: vec![23],
            }
        );

        let mut parse_unnamed_statement_payload = Vec::new();
        parse_unnamed_statement_payload.extend_from_slice(b"\0SELECT 1\0");
        parse_unnamed_statement_payload.extend_from_slice(&0_i16.to_be_bytes());
        let parse_unnamed_statement = frontend_frame(b'P', &parse_unnamed_statement_payload);
        assert_eq!(
            parse_frontend_message(&parse_unnamed_statement).unwrap(),
            FrontendMessage::Parse {
                statement_name: String::new(),
                query: "SELECT 1".to_string(),
                parameter_type_oids: vec![],
            }
        );

        let mut parse_unnamed_statement_with_parameter_oid_payload = Vec::new();
        parse_unnamed_statement_with_parameter_oid_payload
            .extend_from_slice(b"\0SELECT $1::text\0");
        parse_unnamed_statement_with_parameter_oid_payload.extend_from_slice(&1_i16.to_be_bytes());
        parse_unnamed_statement_with_parameter_oid_payload.extend_from_slice(&25_u32.to_be_bytes());
        let parse_unnamed_statement_with_parameter_oid =
            frontend_frame(b'P', &parse_unnamed_statement_with_parameter_oid_payload);
        assert_eq!(
            parse_frontend_message(&parse_unnamed_statement_with_parameter_oid).unwrap(),
            FrontendMessage::Parse {
                statement_name: String::new(),
                query: "SELECT $1::text".to_string(),
                parameter_type_oids: vec![25],
            }
        );

        let mut parse_utf8_query_payload = Vec::new();
        parse_utf8_query_payload.extend_from_slice("stmt_utf8\0SELECT 'héllo'\0".as_bytes());
        parse_utf8_query_payload.extend_from_slice(&0_i16.to_be_bytes());
        let parse_utf8_query = frontend_frame(b'P', &parse_utf8_query_payload);
        assert_eq!(
            parse_frontend_message(&parse_utf8_query).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmt_utf8".to_string(),
                query: "SELECT 'héllo'".to_string(),
                parameter_type_oids: vec![],
            }
        );

        let mut parse_empty_query_payload = Vec::new();
        parse_empty_query_payload.extend_from_slice(b"stmt_empty\0\0");
        parse_empty_query_payload.extend_from_slice(&0_i16.to_be_bytes());
        let parse_empty_query = frontend_frame(b'P', &parse_empty_query_payload);
        assert_eq!(
            parse_frontend_message(&parse_empty_query).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmt_empty".to_string(),
                query: String::new(),
                parameter_type_oids: vec![],
            }
        );

        let mut parse_unnamed_empty_query_payload = Vec::new();
        parse_unnamed_empty_query_payload.extend_from_slice(b"\0\0");
        parse_unnamed_empty_query_payload.extend_from_slice(&0_i16.to_be_bytes());
        let parse_unnamed_empty_query = frontend_frame(b'P', &parse_unnamed_empty_query_payload);
        assert_eq!(
            parse_frontend_message(&parse_unnamed_empty_query).unwrap(),
            FrontendMessage::Parse {
                statement_name: String::new(),
                query: String::new(),
                parameter_type_oids: vec![],
            }
        );

        let mut parse_empty_query_with_parameter_oid_payload = Vec::new();
        parse_empty_query_with_parameter_oid_payload.extend_from_slice(b"stmt_empty_typed\0\0");
        parse_empty_query_with_parameter_oid_payload.extend_from_slice(&1_i16.to_be_bytes());
        parse_empty_query_with_parameter_oid_payload.extend_from_slice(&25_u32.to_be_bytes());
        let parse_empty_query_with_parameter_oid =
            frontend_frame(b'P', &parse_empty_query_with_parameter_oid_payload);
        assert_eq!(
            parse_frontend_message(&parse_empty_query_with_parameter_oid).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmt_empty_typed".to_string(),
                query: String::new(),
                parameter_type_oids: vec![25],
            }
        );

        let mut parse_unnamed_empty_query_with_parameter_oid_payload = Vec::new();
        parse_unnamed_empty_query_with_parameter_oid_payload.extend_from_slice(b"\0\0");
        parse_unnamed_empty_query_with_parameter_oid_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        parse_unnamed_empty_query_with_parameter_oid_payload
            .extend_from_slice(&25_u32.to_be_bytes());
        let parse_unnamed_empty_query_with_parameter_oid =
            frontend_frame(b'P', &parse_unnamed_empty_query_with_parameter_oid_payload);
        assert_eq!(
            parse_frontend_message(&parse_unnamed_empty_query_with_parameter_oid).unwrap(),
            FrontendMessage::Parse {
                statement_name: String::new(),
                query: String::new(),
                parameter_type_oids: vec![25],
            }
        );

        let mut parse_multiline_utf8_query_payload = Vec::new();
        parse_multiline_utf8_query_payload
            .extend_from_slice("stmt_utf8_multi\0SELECT 'héllo'\nFROM tést\0".as_bytes());
        parse_multiline_utf8_query_payload.extend_from_slice(&0_i16.to_be_bytes());
        let parse_multiline_utf8_query = frontend_frame(b'P', &parse_multiline_utf8_query_payload);
        assert_eq!(
            parse_frontend_message(&parse_multiline_utf8_query).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmt_utf8_multi".to_string(),
                query: "SELECT 'héllo'\nFROM tést".to_string(),
                parameter_type_oids: vec![],
            }
        );

        let mut parse_utf8_statement_name_payload = Vec::new();
        parse_utf8_statement_name_payload.extend_from_slice("stmté\0SELECT 42\0".as_bytes());
        parse_utf8_statement_name_payload.extend_from_slice(&0_i16.to_be_bytes());
        let parse_utf8_statement_name = frontend_frame(b'P', &parse_utf8_statement_name_payload);
        assert_eq!(
            parse_frontend_message(&parse_utf8_statement_name).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmté".to_string(),
                query: "SELECT 42".to_string(),
                parameter_type_oids: vec![],
            }
        );

        let mut parse_multiline_utf8_statement_name_payload = Vec::new();
        parse_multiline_utf8_statement_name_payload
            .extend_from_slice("stmté\nΔetail\0SELECT 42\0".as_bytes());
        parse_multiline_utf8_statement_name_payload.extend_from_slice(&0_i16.to_be_bytes());
        let parse_multiline_utf8_statement_name =
            frontend_frame(b'P', &parse_multiline_utf8_statement_name_payload);
        assert_eq!(
            parse_frontend_message(&parse_multiline_utf8_statement_name).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmté\nΔetail".to_string(),
                query: "SELECT 42".to_string(),
                parameter_type_oids: vec![],
            }
        );

        let mut parse_utf8_statement_with_parameter_oids_payload = Vec::new();
        parse_utf8_statement_with_parameter_oids_payload
            .extend_from_slice("stmté\0SELECT $1::text, $2::bytea\0".as_bytes());
        parse_utf8_statement_with_parameter_oids_payload.extend_from_slice(&2_i16.to_be_bytes());
        parse_utf8_statement_with_parameter_oids_payload.extend_from_slice(&25_u32.to_be_bytes());
        parse_utf8_statement_with_parameter_oids_payload.extend_from_slice(&17_u32.to_be_bytes());
        let parse_utf8_statement_with_parameter_oids =
            frontend_frame(b'P', &parse_utf8_statement_with_parameter_oids_payload);
        assert_eq!(
            parse_frontend_message(&parse_utf8_statement_with_parameter_oids).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmté".to_string(),
                query: "SELECT $1::text, $2::bytea".to_string(),
                parameter_type_oids: vec![25, 17],
            }
        );

        let mut parse_utf8_query_with_parameter_oids_payload = Vec::new();
        parse_utf8_query_with_parameter_oids_payload
            .extend_from_slice("stmt_utf8\0SELECT 'héllo', $1::text\0".as_bytes());
        parse_utf8_query_with_parameter_oids_payload.extend_from_slice(&1_i16.to_be_bytes());
        parse_utf8_query_with_parameter_oids_payload.extend_from_slice(&25_u32.to_be_bytes());
        let parse_utf8_query_with_parameter_oids =
            frontend_frame(b'P', &parse_utf8_query_with_parameter_oids_payload);
        assert_eq!(
            parse_frontend_message(&parse_utf8_query_with_parameter_oids).unwrap(),
            FrontendMessage::Parse {
                statement_name: "stmt_utf8".to_string(),
                query: "SELECT 'héllo', $1::text".to_string(),
                parameter_type_oids: vec![25],
            }
        );

        let mut bind_payload = Vec::new();
        bind_payload.extend_from_slice(b"portal1\0stmt1\0");
        bind_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_payload.extend_from_slice(&2_i16.to_be_bytes());
        bind_payload.extend_from_slice(&4_i32.to_be_bytes());
        bind_payload.extend_from_slice(&42_i32.to_be_bytes());
        bind_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        bind_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind = frontend_frame(b'B', &bind_payload);
        assert_eq!(
            parse_frontend_message(&bind).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal1".to_string(),
                statement_name: "stmt1".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(42_i32.to_be_bytes().to_vec()), None],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_with_default_formats_payload = Vec::new();
        bind_unnamed_with_default_formats_payload.extend_from_slice(b"\0\0");
        bind_unnamed_with_default_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_default_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_default_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_unnamed_with_default_formats =
            frontend_frame(b'B', &bind_unnamed_with_default_formats_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_default_formats).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![],
                parameters: vec![],
                result_format_codes: vec![],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_payload = Vec::new();
        bind_unnamed_portal_with_named_statement_payload.extend_from_slice(b"\0stmt_named\0");
        bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_unnamed_portal_with_named_statement_payload.extend_from_slice("héllo".as_bytes());
        bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement =
            frontend_frame(b'B', &bind_unnamed_portal_with_named_statement_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_portal_with_named_statement).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_named".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_payload = Vec::new();
        bind_named_portal_with_unnamed_statement_payload
            .extend_from_slice("pörtal_named\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&4_i32.to_be_bytes());
        bind_named_portal_with_unnamed_statement_payload
            .extend_from_slice(&[0xCA, 0xFE, 0xBA, 0xBE]);
        bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement =
            frontend_frame(b'B', &bind_named_portal_with_unnamed_statement_payload);
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_unnamed_statement).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_named".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0xCA, 0xFE, 0xBA, 0xBE])],
                result_format_codes: vec![1],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_text_payload = Vec::new();
        bind_named_portal_with_unnamed_statement_text_payload
            .extend_from_slice("pörtal_text\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_text_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_named_portal_with_unnamed_statement_text_payload.extend_from_slice("héllo".as_bytes());
        bind_named_portal_with_unnamed_statement_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_text =
            frontend_frame(b'B', &bind_named_portal_with_unnamed_statement_text_payload);
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_unnamed_statement_text).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_text".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_empty_text_payload = Vec::new();
        bind_named_portal_with_unnamed_statement_empty_text_payload
            .extend_from_slice("pörtal_empty\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_text_payload
            .extend_from_slice(&0_i32.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_empty_text = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_empty_text_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_unnamed_statement_empty_text).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_empty".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_empty_binary_payload = Vec::new();
        bind_named_portal_with_unnamed_statement_empty_binary_payload
            .extend_from_slice("pörtal_empty_bin\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_binary_payload
            .extend_from_slice(&0_i32.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_empty_binary = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_empty_binary_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_unnamed_statement_empty_binary).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_empty_bin".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![1],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_binary_payload = Vec::new();
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(b"\0stmt_bin_named\0");
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_binary = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_binary_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_portal_with_named_statement_binary).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_bin_named".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0xDE, 0xAD, 0xBE, 0xEF])],
                result_format_codes: vec![1],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_empty_text_payload = Vec::new();
        bind_unnamed_portal_with_named_statement_empty_text_payload
            .extend_from_slice(b"\0stmt_empty_named\0");
        bind_unnamed_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&0_i32.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_empty_text = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_empty_text_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_portal_with_named_statement_empty_text).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_empty_named".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_empty_binary_payload = Vec::new();
        bind_unnamed_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(b"\0stmt_empty_bin_named\0");
        bind_unnamed_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&0_i32.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_empty_binary = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_empty_binary_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_portal_with_named_statement_empty_binary).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_empty_bin_named".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![1],
            }
        );

        let mut bind_named_portal_with_named_statement_empty_binary_payload = Vec::new();
        bind_named_portal_with_named_statement_empty_binary_payload
            .extend_from_slice("pörtal_pair\0stmt_pair\0".as_bytes());
        bind_named_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&0_i32.to_be_bytes());
        bind_named_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_binary_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_named_statement_empty_binary = frontend_frame(
            b'B',
            &bind_named_portal_with_named_statement_empty_binary_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_named_statement_empty_binary).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_pair".to_string(),
                statement_name: "stmt_pair".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![1],
            }
        );

        let mut bind_named_portal_with_named_statement_empty_text_payload = Vec::new();
        bind_named_portal_with_named_statement_empty_text_payload
            .extend_from_slice("pörtal_pair_text\0stmt_pair_text\0".as_bytes());
        bind_named_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&0_i32.to_be_bytes());
        bind_named_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_empty_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_named_portal_with_named_statement_empty_text = frontend_frame(
            b'B',
            &bind_named_portal_with_named_statement_empty_text_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_named_statement_empty_text).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_pair_text".to_string(),
                statement_name: "stmt_pair_text".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_zero_params_payload = Vec::new();
        bind_named_portal_with_unnamed_statement_zero_params_payload
            .extend_from_slice("pörtal_zero\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_zero_params = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_zero_params_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_unnamed_statement_zero_params).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_zero".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![],
                result_format_codes: vec![1],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_zero_params_payload = Vec::new();
        bind_unnamed_portal_with_named_statement_zero_params_payload
            .extend_from_slice(b"\0stmt_zero_named\0");
        bind_unnamed_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_zero_params = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_zero_params_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_portal_with_named_statement_zero_params).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_zero_named".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![],
                result_format_codes: vec![1],
            }
        );

        let mut bind_named_portal_with_named_statement_zero_params_payload = Vec::new();
        bind_named_portal_with_named_statement_zero_params_payload
            .extend_from_slice("pörtal_zero_pair\0stmt_zero_pair\0".as_bytes());
        bind_named_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_named_statement_zero_params = frontend_frame(
            b'B',
            &bind_named_portal_with_named_statement_zero_params_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_named_statement_zero_params).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_zero_pair".to_string(),
                statement_name: "stmt_zero_pair".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![],
                result_format_codes: vec![1],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_zero_params_text_payload = Vec::new();
        bind_named_portal_with_unnamed_statement_zero_params_text_payload
            .extend_from_slice("pörtal_zero_text\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_zero_params_text = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_zero_params_text_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_unnamed_statement_zero_params_text)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_zero_text".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_zero_params_text_payload = Vec::new();
        bind_unnamed_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(b"\0stmt_zero_text_named\0");
        bind_unnamed_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_zero_params_text = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_zero_params_text_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_portal_with_named_statement_zero_params_text)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_zero_text_named".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![],
                result_format_codes: vec![0],
            }
        );

        let mut bind_named_portal_with_named_statement_zero_params_text_payload = Vec::new();
        bind_named_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice("pörtal_zero_text_pair\0stmt_zero_text_pair\0".as_bytes());
        bind_named_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_zero_params_text_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_named_portal_with_named_statement_zero_params_text = frontend_frame(
            b'B',
            &bind_named_portal_with_named_statement_zero_params_text_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_named_statement_zero_params_text)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_zero_text_pair".to_string(),
                statement_name: "stmt_zero_text_pair".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_with_zero_params_and_single_shared_text_format_payload = Vec::new();
        bind_unnamed_with_zero_params_and_single_shared_text_format_payload
            .extend_from_slice(b"\0\0");
        bind_unnamed_with_zero_params_and_single_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_unnamed_with_zero_params_and_single_shared_text_format = frontend_frame(
            b'B',
            &bind_unnamed_with_zero_params_and_single_shared_text_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_zero_params_and_single_shared_text_format)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_with_zero_params_and_single_shared_binary_format_payload = Vec::new();
        bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
            .extend_from_slice(b"\0\0");
        bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_with_zero_params_and_single_shared_binary_format = frontend_frame(
            b'B',
            &bind_unnamed_with_zero_params_and_single_shared_binary_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_zero_params_and_single_shared_binary_format)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![],
                result_format_codes: vec![1],
            }
        );

        let mut bind_unnamed_with_empty_text_parameter_payload = Vec::new();
        bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(b"\0\0");
        bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
        bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_unnamed_with_empty_text_parameter =
            frontend_frame(b'B', &bind_unnamed_with_empty_text_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_empty_text_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_with_empty_binary_parameter_payload = Vec::new();
        bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(b"\0\0");
        bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
        bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_with_empty_binary_parameter =
            frontend_frame(b'B', &bind_unnamed_with_empty_binary_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_empty_binary_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![1],
            }
        );

        let mut bind_unnamed_with_text_parameter_payload = Vec::new();
        bind_unnamed_with_text_parameter_payload.extend_from_slice(b"\0\0");
        bind_unnamed_with_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_text_parameter_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_unnamed_with_text_parameter_payload.extend_from_slice("héllo".as_bytes());
        bind_unnamed_with_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_unnamed_with_text_parameter =
            frontend_frame(b'B', &bind_unnamed_with_text_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_text_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_with_binary_parameter_payload = Vec::new();
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(b"\0\0");
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(&4_i32.to_be_bytes());
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(&[0x00, 0xCA, 0x00, 0xFE]);
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_with_binary_parameter =
            frontend_frame(b'B', &bind_unnamed_with_binary_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_binary_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0x00, 0xCA, 0x00, 0xFE])],
                result_format_codes: vec![1],
            }
        );

        let mut bind_with_zero_params_and_single_shared_format_payload = Vec::new();
        bind_with_zero_params_and_single_shared_format_payload
            .extend_from_slice(b"portal3\0stmt3\0");
        bind_with_zero_params_and_single_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_with_zero_params_and_single_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_with_zero_params_and_single_shared_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_zero_params_and_single_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_with_zero_params_and_single_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_with_zero_params_and_single_shared_format = frontend_frame(
            b'B',
            &bind_with_zero_params_and_single_shared_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_with_zero_params_and_single_shared_format).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal3".to_string(),
                statement_name: "stmt3".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![],
                result_format_codes: vec![1],
            }
        );

        let mut bind_unnamed_with_multiple_result_formats_payload = Vec::new();
        bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(b"\0\0");
        bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_with_multiple_result_formats =
            frontend_frame(b'B', &bind_unnamed_with_multiple_result_formats_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_multiple_result_formats).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![],
                parameters: vec![],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_multiple_result_formats_payload =
            Vec::new();
        bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
            .extend_from_slice("pörtal_results\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_multiple_result_formats = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_multiple_result_formats_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_named_portal_with_unnamed_statement_multiple_result_formats,
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_results".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![],
                parameters: vec![],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_multiple_result_formats_payload =
            Vec::new();
        bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
            .extend_from_slice(b"\0stmt_results_named\0");
        bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_multiple_result_formats = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_multiple_result_formats_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_unnamed_portal_with_named_statement_multiple_result_formats,
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_results_named".to_string(),
                parameter_format_codes: vec![],
                parameters: vec![],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_unnamed_with_default_parameter_formats_payload = Vec::new();
        bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(b"\0\0");
        bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&4_i32.to_be_bytes());
        bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(b"text");
        bind_unnamed_with_default_parameter_formats_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_with_default_parameter_formats =
            frontend_frame(b'B', &bind_unnamed_with_default_parameter_formats_payload);
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_default_parameter_formats).unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![],
                parameters: vec![Some(b"text".to_vec()), None],
                result_format_codes: vec![1],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_default_parameter_formats_payload =
            Vec::new();
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice("pörtal_default\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice(b"text");
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_default_parameter_formats = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_default_parameter_formats_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_named_portal_with_unnamed_statement_default_parameter_formats,
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_default".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![],
                parameters: vec![Some(b"text".to_vec()), None],
                result_format_codes: vec![1],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_default_parameter_formats_payload =
            Vec::new();
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(b"\0stmt_default_named\0");
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(b"text");
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_default_parameter_formats = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_default_parameter_formats_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_unnamed_portal_with_named_statement_default_parameter_formats,
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_default_named".to_string(),
                parameter_format_codes: vec![],
                parameters: vec![Some(b"text".to_vec()), None],
                result_format_codes: vec![1],
            }
        );

        let mut bind_named_portal_with_named_statement_default_parameter_formats_payload =
            Vec::new();
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice("pörtal_default_pair\0stmt_default_pair\0".as_bytes());
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(b"text");
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_default_parameter_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_named_statement_default_parameter_formats = frontend_frame(
            b'B',
            &bind_named_portal_with_named_statement_default_parameter_formats_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_named_portal_with_named_statement_default_parameter_formats
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_default_pair".to_string(),
                statement_name: "stmt_default_pair".to_string(),
                parameter_format_codes: vec![],
                parameters: vec![Some(b"text".to_vec()), None],
                result_format_codes: vec![1],
            }
        );

        let mut bind_with_default_parameter_formats_payload = Vec::new();
        bind_with_default_parameter_formats_payload.extend_from_slice(b"portal2\0stmt2\0");
        bind_with_default_parameter_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_default_parameter_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
        bind_with_default_parameter_formats_payload.extend_from_slice(&4_i32.to_be_bytes());
        bind_with_default_parameter_formats_payload.extend_from_slice(b"text");
        bind_with_default_parameter_formats_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        bind_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_with_default_parameter_formats =
            frontend_frame(b'B', &bind_with_default_parameter_formats_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_default_parameter_formats).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal2".to_string(),
                statement_name: "stmt2".to_string(),
                parameter_format_codes: vec![],
                parameters: vec![Some(b"text".to_vec()), None],
                result_format_codes: vec![1],
            }
        );

        let mut bind_utf8_names_payload = Vec::new();
        bind_utf8_names_payload.extend_from_slice("pörtal\0stmté\0".as_bytes());
        bind_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_utf8_names = frontend_frame(b'B', &bind_utf8_names_payload);
        assert_eq!(
            parse_frontend_message(&bind_utf8_names).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal".to_string(),
                statement_name: "stmté".to_string(),
                parameter_format_codes: vec![],
                parameters: vec![],
                result_format_codes: vec![],
            }
        );

        let mut bind_multiline_utf8_names_payload = Vec::new();
        bind_multiline_utf8_names_payload
            .extend_from_slice("pörtal\nΔetail\0stmté\nΔetail\0".as_bytes());
        bind_multiline_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_multiline_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_multiline_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_multiline_utf8_names = frontend_frame(b'B', &bind_multiline_utf8_names_payload);
        assert_eq!(
            parse_frontend_message(&bind_multiline_utf8_names).unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal\nΔetail".to_string(),
                statement_name: "stmté\nΔetail".to_string(),
                parameter_format_codes: vec![],
                parameters: vec![],
                result_format_codes: vec![],
            }
        );

        let mut bind_with_utf8_text_parameter_payload = Vec::new();
        bind_with_utf8_text_parameter_payload.extend_from_slice(b"portal_utf8\0stmt_utf8\0");
        bind_with_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_utf8_text_parameter_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_with_utf8_text_parameter_payload.extend_from_slice("héllo".as_bytes());
        bind_with_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_with_utf8_text_parameter =
            frontend_frame(b'B', &bind_with_utf8_text_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_utf8_text_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal_utf8".to_string(),
                statement_name: "stmt_utf8".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_with_multiline_utf8_text_parameter_payload = Vec::new();
        bind_with_multiline_utf8_text_parameter_payload
            .extend_from_slice(b"portal_utf8_multi\0stmt_utf8_multi\0");
        bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_multiline_utf8_text_parameter_payload
            .extend_from_slice(&("héllo\nΔetail".len() as i32).to_be_bytes());
        bind_with_multiline_utf8_text_parameter_payload
            .extend_from_slice("héllo\nΔetail".as_bytes());
        bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_with_multiline_utf8_text_parameter =
            frontend_frame(b'B', &bind_with_multiline_utf8_text_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_multiline_utf8_text_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal_utf8_multi".to_string(),
                statement_name: "stmt_utf8_multi".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo\nΔetail".as_bytes().to_vec())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_with_empty_text_parameter_payload = Vec::new();
        bind_with_empty_text_parameter_payload
            .extend_from_slice(b"portal_empty_text\0stmt_empty_text\0");
        bind_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_empty_text_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
        bind_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_with_empty_text_parameter =
            frontend_frame(b'B', &bind_with_empty_text_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_empty_text_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal_empty_text".to_string(),
                statement_name: "stmt_empty_text".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![0],
            }
        );

        let mut bind_with_binary_parameter_payload = Vec::new();
        bind_with_binary_parameter_payload.extend_from_slice(b"portal_bin\0stmt_bin\0");
        bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_binary_parameter_payload.extend_from_slice(&4_i32.to_be_bytes());
        bind_with_binary_parameter_payload.extend_from_slice(&[0x00, 0xCA, 0x00, 0xFE]);
        bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_with_binary_parameter = frontend_frame(b'B', &bind_with_binary_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_binary_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal_bin".to_string(),
                statement_name: "stmt_bin".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0x00, 0xCA, 0x00, 0xFE])],
                result_format_codes: vec![1],
            }
        );

        let mut bind_with_empty_binary_parameter_payload = Vec::new();
        bind_with_empty_binary_parameter_payload.extend_from_slice(b"portal_empty\0stmt_empty\0");
        bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_empty_binary_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
        bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_with_empty_binary_parameter =
            frontend_frame(b'B', &bind_with_empty_binary_parameter_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_empty_binary_parameter).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal_empty".to_string(),
                statement_name: "stmt_empty".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(Vec::new())],
                result_format_codes: vec![1],
            }
        );

        let describe_stmt = frontend_frame(b'D', b"Sstmt1\0");
        assert_eq!(
            parse_frontend_message(&describe_stmt).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Statement,
                name: "stmt1".to_string(),
            }
        );

        let describe_portal = frontend_frame(b'D', b"Pportal1\0");
        assert_eq!(
            parse_frontend_message(&describe_portal).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Portal,
                name: "portal1".to_string(),
            }
        );

        let describe_stmt_lowercase = frontend_frame(b'D', b"sstmt1\0");
        assert_eq!(
            parse_frontend_message(&describe_stmt_lowercase).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Statement,
                name: "stmt1".to_string(),
            }
        );

        let describe_portal_lowercase = frontend_frame(b'D', b"pportal1\0");
        assert_eq!(
            parse_frontend_message(&describe_portal_lowercase).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Portal,
                name: "portal1".to_string(),
            }
        );

        let describe_utf8_portal = frontend_frame(b'D', "Ppörtal\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&describe_utf8_portal).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Portal,
                name: "pörtal".to_string(),
            }
        );

        let describe_utf8_statement = frontend_frame(b'D', "Sstmté\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&describe_utf8_statement).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Statement,
                name: "stmté".to_string(),
            }
        );

        let describe_multiline_utf8_portal = frontend_frame(b'D', "Ppörtal\nΔetail\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&describe_multiline_utf8_portal).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Portal,
                name: "pörtal\nΔetail".to_string(),
            }
        );

        let describe_multiline_utf8_statement = frontend_frame(b'D', "Sstmté\nΔetail\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&describe_multiline_utf8_statement).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Statement,
                name: "stmté\nΔetail".to_string(),
            }
        );

        let describe_unnamed_statement = frontend_frame(b'D', b"S\0");
        assert_eq!(
            parse_frontend_message(&describe_unnamed_statement).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Statement,
                name: String::new(),
            }
        );

        let describe_unnamed_portal_lowercase = frontend_frame(b'D', b"p\0");
        assert_eq!(
            parse_frontend_message(&describe_unnamed_portal_lowercase).unwrap(),
            FrontendMessage::Describe {
                target: DescribeTarget::Portal,
                name: String::new(),
            }
        );

        let close_stmt = frontend_frame(b'C', b"Sstmt1\0");
        assert_eq!(
            parse_frontend_message(&close_stmt).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Statement,
                name: "stmt1".to_string(),
            }
        );

        let close_portal = frontend_frame(b'C', b"Pportal1\0");
        assert_eq!(
            parse_frontend_message(&close_portal).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Portal,
                name: "portal1".to_string(),
            }
        );

        let close_stmt_lowercase = frontend_frame(b'C', b"sstmt1\0");
        assert_eq!(
            parse_frontend_message(&close_stmt_lowercase).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Statement,
                name: "stmt1".to_string(),
            }
        );

        let close_portal_lowercase = frontend_frame(b'C', b"pportal1\0");
        assert_eq!(
            parse_frontend_message(&close_portal_lowercase).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Portal,
                name: "portal1".to_string(),
            }
        );

        let close_utf8_portal = frontend_frame(b'C', "Ppörtal\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&close_utf8_portal).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Portal,
                name: "pörtal".to_string(),
            }
        );

        let close_utf8_statement = frontend_frame(b'C', "Sstmté\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&close_utf8_statement).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Statement,
                name: "stmté".to_string(),
            }
        );

        let close_multiline_utf8_portal = frontend_frame(b'C', "Ppörtal\nΔetail\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&close_multiline_utf8_portal).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Portal,
                name: "pörtal\nΔetail".to_string(),
            }
        );

        let close_multiline_utf8_statement = frontend_frame(b'C', "Sstmté\nΔetail\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&close_multiline_utf8_statement).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Statement,
                name: "stmté\nΔetail".to_string(),
            }
        );

        let close_unnamed_statement = frontend_frame(b'C', b"S\0");
        assert_eq!(
            parse_frontend_message(&close_unnamed_statement).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Statement,
                name: String::new(),
            }
        );

        let close_unnamed_portal_lowercase = frontend_frame(b'C', b"p\0");
        assert_eq!(
            parse_frontend_message(&close_unnamed_portal_lowercase).unwrap(),
            FrontendMessage::Close {
                target: DescribeTarget::Portal,
                name: String::new(),
            }
        );

        let mut execute_payload = Vec::new();
        execute_payload.extend_from_slice(b"portal1\0");
        execute_payload.extend_from_slice(&128_u32.to_be_bytes());
        let execute = frontend_frame(b'E', &execute_payload);
        assert_eq!(
            parse_frontend_message(&execute).unwrap(),
            FrontendMessage::Execute {
                portal_name: "portal1".to_string(),
                max_rows: 128,
            }
        );

        let mut execute_unnamed_portal_unlimited_rows_payload = Vec::new();
        execute_unnamed_portal_unlimited_rows_payload.extend_from_slice(b"\0");
        execute_unnamed_portal_unlimited_rows_payload.extend_from_slice(&0_u32.to_be_bytes());
        let execute_unnamed_portal_unlimited_rows =
            frontend_frame(b'E', &execute_unnamed_portal_unlimited_rows_payload);
        assert_eq!(
            parse_frontend_message(&execute_unnamed_portal_unlimited_rows).unwrap(),
            FrontendMessage::Execute {
                portal_name: String::new(),
                max_rows: 0,
            }
        );

        let mut execute_utf8_portal_payload = Vec::new();
        execute_utf8_portal_payload.extend_from_slice("pörtal\0".as_bytes());
        execute_utf8_portal_payload.extend_from_slice(&64_u32.to_be_bytes());
        let execute_utf8_portal = frontend_frame(b'E', &execute_utf8_portal_payload);
        assert_eq!(
            parse_frontend_message(&execute_utf8_portal).unwrap(),
            FrontendMessage::Execute {
                portal_name: "pörtal".to_string(),
                max_rows: 64,
            }
        );

        let mut execute_multiline_utf8_portal_payload = Vec::new();
        execute_multiline_utf8_portal_payload.extend_from_slice("pörtal\nΔetail\0".as_bytes());
        execute_multiline_utf8_portal_payload.extend_from_slice(&32_u32.to_be_bytes());
        let execute_multiline_utf8_portal =
            frontend_frame(b'E', &execute_multiline_utf8_portal_payload);
        assert_eq!(
            parse_frontend_message(&execute_multiline_utf8_portal).unwrap(),
            FrontendMessage::Execute {
                portal_name: "pörtal\nΔetail".to_string(),
                max_rows: 32,
            }
        );

        let mut function_call_payload = Vec::new();
        function_call_payload.extend_from_slice(&42_u32.to_be_bytes());
        function_call_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_payload.extend_from_slice(&2_i16.to_be_bytes());
        function_call_payload.extend_from_slice(&4_i32.to_be_bytes());
        function_call_payload.extend_from_slice(&7_i32.to_be_bytes());
        function_call_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        function_call_payload.extend_from_slice(&0_i16.to_be_bytes());
        let function_call = frontend_frame(b'F', &function_call_payload);
        assert_eq!(
            parse_frontend_message(&function_call).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 42,
                argument_format_codes: vec![1],
                arguments: vec![Some(7_i32.to_be_bytes().to_vec()), None],
                result_format_code: 0,
            }
        );

        let mut function_call_with_multiple_argument_formats_payload = Vec::new();
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&7_u32.to_be_bytes());
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&3_i32.to_be_bytes());
        function_call_with_multiple_argument_formats_payload.extend_from_slice(b"foo");
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&2_i32.to_be_bytes());
        function_call_with_multiple_argument_formats_payload.extend_from_slice(&[0xCA, 0xFE]);
        function_call_with_multiple_argument_formats_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let function_call_with_multiple_argument_formats =
            frontend_frame(b'F', &function_call_with_multiple_argument_formats_payload);
        assert_eq!(
            parse_frontend_message(&function_call_with_multiple_argument_formats).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 7,
                argument_format_codes: vec![0, 1],
                arguments: vec![Some(b"foo".to_vec()), Some(vec![0xCA, 0xFE])],
                result_format_code: 1,
            }
        );

        let mut function_call_with_zero_args_and_single_shared_format_payload = Vec::new();
        function_call_with_zero_args_and_single_shared_format_payload
            .extend_from_slice(&99_u32.to_be_bytes());
        function_call_with_zero_args_and_single_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_zero_args_and_single_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_zero_args_and_single_shared_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        function_call_with_zero_args_and_single_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let function_call_with_zero_args_and_single_shared_format = frontend_frame(
            b'F',
            &function_call_with_zero_args_and_single_shared_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&function_call_with_zero_args_and_single_shared_format).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 99,
                argument_format_codes: vec![1],
                arguments: vec![],
                result_format_code: 1,
            }
        );

        let mut function_call_with_zero_args_and_single_shared_text_format_payload = Vec::new();
        function_call_with_zero_args_and_single_shared_text_format_payload
            .extend_from_slice(&100_u32.to_be_bytes());
        function_call_with_zero_args_and_single_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_zero_args_and_single_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        function_call_with_zero_args_and_single_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        function_call_with_zero_args_and_single_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let function_call_with_zero_args_and_single_shared_text_format = frontend_frame(
            b'F',
            &function_call_with_zero_args_and_single_shared_text_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&function_call_with_zero_args_and_single_shared_text_format)
                .unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 100,
                argument_format_codes: vec![0],
                arguments: vec![],
                result_format_code: 0,
            }
        );

        let mut function_call_with_embedded_null_binary_arg_payload = Vec::new();
        function_call_with_embedded_null_binary_arg_payload
            .extend_from_slice(&101_u32.to_be_bytes());
        function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&4_i32.to_be_bytes());
        function_call_with_embedded_null_binary_arg_payload
            .extend_from_slice(&[0x00, 0xCA, 0x00, 0xFE]);
        function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        let function_call_with_embedded_null_binary_arg =
            frontend_frame(b'F', &function_call_with_embedded_null_binary_arg_payload);
        assert_eq!(
            parse_frontend_message(&function_call_with_embedded_null_binary_arg).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 101,
                argument_format_codes: vec![1],
                arguments: vec![Some(vec![0x00, 0xCA, 0x00, 0xFE])],
                result_format_code: 1,
            }
        );

        let mut function_call_with_utf8_text_arg_payload = Vec::new();
        function_call_with_utf8_text_arg_payload.extend_from_slice(&103_u32.to_be_bytes());
        function_call_with_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
        function_call_with_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_utf8_text_arg_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        function_call_with_utf8_text_arg_payload.extend_from_slice("héllo".as_bytes());
        function_call_with_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
        let function_call_with_utf8_text_arg =
            frontend_frame(b'F', &function_call_with_utf8_text_arg_payload);
        assert_eq!(
            parse_frontend_message(&function_call_with_utf8_text_arg).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 103,
                argument_format_codes: vec![0],
                arguments: vec![Some("héllo".as_bytes().to_vec())],
                result_format_code: 0,
            }
        );

        let mut function_call_with_multiline_utf8_text_arg_payload = Vec::new();
        function_call_with_multiline_utf8_text_arg_payload
            .extend_from_slice(&105_u32.to_be_bytes());
        function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
        function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_multiline_utf8_text_arg_payload
            .extend_from_slice(&("héllo\nΔetail".len() as i32).to_be_bytes());
        function_call_with_multiline_utf8_text_arg_payload
            .extend_from_slice("héllo\nΔetail".as_bytes());
        function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
        let function_call_with_multiline_utf8_text_arg =
            frontend_frame(b'F', &function_call_with_multiline_utf8_text_arg_payload);
        assert_eq!(
            parse_frontend_message(&function_call_with_multiline_utf8_text_arg).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 105,
                argument_format_codes: vec![0],
                arguments: vec![Some("héllo\nΔetail".as_bytes().to_vec())],
                result_format_code: 0,
            }
        );

        let mut function_call_with_empty_text_arg_payload = Vec::new();
        function_call_with_empty_text_arg_payload.extend_from_slice(&104_u32.to_be_bytes());
        function_call_with_empty_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_empty_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
        function_call_with_empty_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_empty_text_arg_payload.extend_from_slice(&0_i32.to_be_bytes());
        function_call_with_empty_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
        let function_call_with_empty_text_arg =
            frontend_frame(b'F', &function_call_with_empty_text_arg_payload);
        assert_eq!(
            parse_frontend_message(&function_call_with_empty_text_arg).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 104,
                argument_format_codes: vec![0],
                arguments: vec![Some(Vec::new())],
                result_format_code: 0,
            }
        );

        let mut function_call_with_empty_binary_arg_payload = Vec::new();
        function_call_with_empty_binary_arg_payload.extend_from_slice(&102_u32.to_be_bytes());
        function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        function_call_with_empty_binary_arg_payload.extend_from_slice(&0_i32.to_be_bytes());
        function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
        let function_call_with_empty_binary_arg =
            frontend_frame(b'F', &function_call_with_empty_binary_arg_payload);
        assert_eq!(
            parse_frontend_message(&function_call_with_empty_binary_arg).unwrap(),
            FrontendMessage::FunctionCall {
                function_oid: 102,
                argument_format_codes: vec![1],
                arguments: vec![Some(Vec::new())],
                result_format_code: 1,
            }
        );

        let copy_data = frontend_frame(b'd', &[0, 1, 2, 3]);
        assert_eq!(
            parse_frontend_message(&copy_data).unwrap(),
            FrontendMessage::CopyData(vec![0, 1, 2, 3])
        );

        let empty_copy_data = frontend_frame(b'd', &[]);
        assert_eq!(
            parse_frontend_message(&empty_copy_data).unwrap(),
            FrontendMessage::CopyData(Vec::new())
        );

        let binary_copy_data = frontend_frame(b'd', b"row\0chunk\xff");
        assert_eq!(
            parse_frontend_message(&binary_copy_data).unwrap(),
            FrontendMessage::CopyData(b"row\0chunk\xff".to_vec())
        );

        let utf8_copy_data = frontend_frame(b'd', "röwΔ".as_bytes());
        assert_eq!(
            parse_frontend_message(&utf8_copy_data).unwrap(),
            FrontendMessage::CopyData("röwΔ".as_bytes().to_vec())
        );

        let mixed_utf8_binary_copy_data = frontend_frame(b'd', &[b'r', 0xC3, 0xB8, 0xFF, b'!']);
        assert_eq!(
            parse_frontend_message(&mixed_utf8_binary_copy_data).unwrap(),
            FrontendMessage::CopyData(vec![b'r', 0xC3, 0xB8, 0xFF, b'!'])
        );

        let copy_done = frontend_frame(b'c', &[]);
        assert_eq!(
            parse_frontend_message(&copy_done).unwrap(),
            FrontendMessage::CopyDone
        );

        let copy_fail = frontend_frame(b'f', b"bad row\0");
        assert_eq!(
            parse_frontend_message(&copy_fail).unwrap(),
            FrontendMessage::CopyFail("bad row".to_string())
        );

        let empty_copy_fail = frontend_frame(b'f', b"\0");
        assert_eq!(
            parse_frontend_message(&empty_copy_fail).unwrap(),
            FrontendMessage::CopyFail(String::new())
        );

        let utf8_copy_fail = frontend_frame(b'f', "röw mismatch\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&utf8_copy_fail).unwrap(),
            FrontendMessage::CopyFail("röw mismatch".to_string())
        );

        let multiline_utf8_copy_fail = frontend_frame(b'f', "röw mismatch\nΔetail\0".as_bytes());
        assert_eq!(
            parse_frontend_message(&multiline_utf8_copy_fail).unwrap(),
            FrontendMessage::CopyFail("röw mismatch\nΔetail".to_string())
        );

        let terminate = frontend_frame(b'X', &[]);
        assert_eq!(
            parse_frontend_message(&terminate).unwrap(),
            FrontendMessage::Terminate
        );

        let sync = frontend_frame(b'S', &[]);
        assert_eq!(
            parse_frontend_message(&sync).unwrap(),
            FrontendMessage::Sync
        );

        let flush = frontend_frame(b'H', &[]);
        assert_eq!(
            parse_frontend_message(&flush).unwrap(),
            FrontendMessage::Flush
        );
    }

    #[test]
    fn rejects_malformed_frontend_frames() {
        assert_eq!(
            parse_frontend_message(&[b'Q', 0, 0, 0]).unwrap_err(),
            FrontendMessageError::TooShort
        );

        let bad_len = vec![b'Q', 0, 0, 0, 7, b';', 0];
        assert_eq!(
            parse_frontend_message(&bad_len).unwrap_err(),
            FrontendMessageError::LengthMismatch {
                expected: 8,
                actual: 7,
            }
        );

        let invalid_length_field = vec![b'Q', 0, 0, 0, 3];
        assert_eq!(
            parse_frontend_message(&invalid_length_field).unwrap_err(),
            FrontendMessageError::InvalidLengthField { declared: 3 }
        );

        let unterminated = frontend_frame(b'Q', b"SELECT 1;");
        assert_eq!(
            parse_frontend_message(&unterminated).unwrap_err(),
            FrontendMessageError::UnterminatedSimpleQuery
        );

        let query_with_embedded_null = frontend_frame(b'Q', b"SELECT\0 1;\0");
        assert_eq!(
            parse_frontend_message(&query_with_embedded_null).unwrap_err(),
            FrontendMessageError::UnterminatedSimpleQuery
        );

        let password_like_sasl_response = frontend_frame(b'p', b"secret");
        assert_eq!(
            parse_frontend_message(&password_like_sasl_response).unwrap(),
            FrontendMessage::SaslResponse(b"secret".to_vec())
        );

        let binary_sasl_response = frontend_frame(b'p', &[0xFF, 0xFE]);
        assert_eq!(
            parse_frontend_message(&binary_sasl_response).unwrap(),
            FrontendMessage::SaslResponse(vec![0xFF, 0xFE])
        );

        let mixed_utf8_binary_sasl_response = frontend_frame(b'p', &[b'p', b'r', 0xC3, 0xB8, 0xFF]);
        assert_eq!(
            parse_frontend_message(&mixed_utf8_binary_sasl_response).unwrap(),
            FrontendMessage::SaslResponse(vec![b'p', b'r', 0xC3, 0xB8, 0xFF])
        );

        let unsupported_frontend_message = frontend_frame(b'Z', b"");
        assert_eq!(
            parse_frontend_message(&unsupported_frontend_message).unwrap_err(),
            FrontendMessageError::UnsupportedTag(b'Z')
        );

        let malformed_sasl_initial = frontend_frame(b'p', b"SCRAM-SHA-256\0\0\0");
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let empty_mechanism_sasl_initial = frontend_frame(b'p', b"\0\xff\xff\xff\xff");
        assert_eq!(
            parse_frontend_message(&empty_mechanism_sasl_initial).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let invalid_utf8_mechanism_sasl_initial = frontend_frame(b'p', &[0xFF, 0, 0, 0, 0, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_mechanism_sasl_initial).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let malformed_sasl_initial_negative_len = frontend_frame(
            b'p',
            &[b'S', b'C', b'R', b'A', b'M', 0, 0xFF, 0xFF, 0xFF, 0xFE],
        );
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial_negative_len).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_sasl_initial_null_len_with_trailing_payload =
            frontend_frame(b'p', b"SCRAM-SHA-256\0\xFF\xFF\xFF\xFFx");
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial_null_len_with_trailing_payload)
                .unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_sasl_initial_declared_len_mismatch =
            frontend_frame(b'p', b"SCRAM-SHA-256\0\0\0\0\x03xy");
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial_declared_len_mismatch).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_sasl_initial_negative_non_null_len =
            frontend_frame(b'p', b"SCRAM-SHA-256\0\xFF\xFF\xFF\xFE");
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial_negative_non_null_len).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_sasl_initial_empty_mechanism = frontend_frame(b'p', b"\0\0\0\0\0");
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial_empty_mechanism).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_sasl_initial_zero_len_with_trailing_payload =
            frontend_frame(b'p', b"SCRAM-SHA-256\0\0\0\0\0x");
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial_zero_len_with_trailing_payload)
                .unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_parse = frontend_frame(b'P', b"stmt\0SELECT 1\0\0\x01");
        assert_eq!(
            parse_frontend_message(&malformed_parse).unwrap_err(),
            FrontendMessageError::InvalidParseParameterPayload
        );

        let unterminated_parse_statement_name = frontend_frame(b'P', b"stmt");
        assert_eq!(
            parse_frontend_message(&unterminated_parse_statement_name).unwrap_err(),
            FrontendMessageError::UnterminatedParseStatementName
        );

        let unterminated_parse_query = frontend_frame(b'P', b"stmt\0SELECT 1");
        assert_eq!(
            parse_frontend_message(&unterminated_parse_query).unwrap_err(),
            FrontendMessageError::UnterminatedParseQuery
        );

        let truncated_parse_type_count_field = frontend_frame(b'P', b"stmt\0SELECT 1\0\0");
        assert_eq!(
            parse_frontend_message(&truncated_parse_type_count_field).unwrap_err(),
            FrontendMessageError::InvalidParseParameterPayload
        );

        let mut truncated_parse_parameter_oid_field_payload = Vec::new();
        truncated_parse_parameter_oid_field_payload.extend_from_slice(b"stmt\0SELECT 1\0");
        truncated_parse_parameter_oid_field_payload.extend_from_slice(&1_i16.to_be_bytes());
        truncated_parse_parameter_oid_field_payload.extend_from_slice(&[0x00, 0x00, 0x00]);
        let truncated_parse_parameter_oid_field =
            frontend_frame(b'P', &truncated_parse_parameter_oid_field_payload);
        assert_eq!(
            parse_frontend_message(&truncated_parse_parameter_oid_field).unwrap_err(),
            FrontendMessageError::InvalidParseParameterPayload
        );

        let mut malformed_bind_with_truncated_result_format_codes_payload = Vec::new();
        malformed_bind_with_truncated_result_format_codes_payload.extend_from_slice(b"\0\0");
        malformed_bind_with_truncated_result_format_codes_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        malformed_bind_with_truncated_result_format_codes_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        malformed_bind_with_truncated_result_format_codes_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        malformed_bind_with_truncated_result_format_codes_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let malformed_bind_with_truncated_result_format_codes = frontend_frame(
            b'B',
            &malformed_bind_with_truncated_result_format_codes_payload,
        );
        assert_eq!(
            parse_frontend_message(&malformed_bind_with_truncated_result_format_codes).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut malformed_bind_with_zero_params_and_invalid_shared_format_payload = Vec::new();
        malformed_bind_with_zero_params_and_invalid_shared_format_payload
            .extend_from_slice(b"\0\0");
        malformed_bind_with_zero_params_and_invalid_shared_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        malformed_bind_with_zero_params_and_invalid_shared_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        malformed_bind_with_zero_params_and_invalid_shared_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        malformed_bind_with_zero_params_and_invalid_shared_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let malformed_bind_with_zero_params_and_invalid_shared_format = frontend_frame(
            b'B',
            &malformed_bind_with_zero_params_and_invalid_shared_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&malformed_bind_with_zero_params_and_invalid_shared_format)
                .unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let invalid_utf8_parse_statement_name = frontend_frame(
            b'P',
            &[
                0xFF, 0, b'S', b'E', b'L', b'E', b'C', b'T', b' ', b'1', 0, 0, 0,
            ],
        );
        assert_eq!(
            parse_frontend_message(&invalid_utf8_parse_statement_name).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let invalid_utf8_parse_query =
            frontend_frame(b'P', &[b's', b't', b'm', b't', 0, 0xFF, 0, 0, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_parse_query).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let malformed_parse_negative_type_count = frontend_frame(
            b'P',
            &[
                b's', b't', b'm', b't', 0, b'S', b'E', b'L', b'E', b'C', b'T', b' ', b'1', 0, 0xFF,
                0xFF,
            ],
        );
        assert_eq!(
            parse_frontend_message(&malformed_parse_negative_type_count).unwrap_err(),
            FrontendMessageError::InvalidParseParameterPayload
        );

        let malformed_parse_with_zero_type_count_and_trailing_bytes =
            frontend_frame(b'P', b"stmt\0SELECT 1\0\0\0\xFF");
        assert_eq!(
            parse_frontend_message(&malformed_parse_with_zero_type_count_and_trailing_bytes)
                .unwrap_err(),
            FrontendMessageError::InvalidParseParameterPayload
        );

        let mut malformed_parse_with_trailing_oid_bytes_payload = Vec::new();
        malformed_parse_with_trailing_oid_bytes_payload.extend_from_slice(b"stmt\0SELECT 1\0");
        malformed_parse_with_trailing_oid_bytes_payload.extend_from_slice(&1_i16.to_be_bytes());
        malformed_parse_with_trailing_oid_bytes_payload.extend_from_slice(&23_u32.to_be_bytes());
        malformed_parse_with_trailing_oid_bytes_payload.push(0xFF);
        let malformed_parse_with_trailing_oid_bytes =
            frontend_frame(b'P', &malformed_parse_with_trailing_oid_bytes_payload);
        assert_eq!(
            parse_frontend_message(&malformed_parse_with_trailing_oid_bytes).unwrap_err(),
            FrontendMessageError::InvalidParseParameterPayload
        );

        let empty_describe = frontend_frame(b'D', b"");
        assert_eq!(
            parse_frontend_message(&empty_describe).unwrap_err(),
            FrontendMessageError::TooShort
        );

        let truncated_describe_name = frontend_frame(b'D', b"S");
        assert_eq!(
            parse_frontend_message(&truncated_describe_name).unwrap_err(),
            FrontendMessageError::UnterminatedDescribeName
        );

        let truncated_describe_name_lowercase_target = frontend_frame(b'D', b"pportal");
        assert_eq!(
            parse_frontend_message(&truncated_describe_name_lowercase_target).unwrap_err(),
            FrontendMessageError::UnterminatedDescribeName
        );

        let invalid_describe_target = frontend_frame(b'D', b"Xstmt\0");
        assert_eq!(
            parse_frontend_message(&invalid_describe_target).unwrap_err(),
            FrontendMessageError::InvalidDescribeTarget
        );

        let describe_with_embedded_null = frontend_frame(b'D', b"Sstmt\0extra\0");
        assert_eq!(
            parse_frontend_message(&describe_with_embedded_null).unwrap_err(),
            FrontendMessageError::UnterminatedDescribeName
        );

        let invalid_utf8_describe_name = frontend_frame(b'D', b"S\xFF\0");
        assert_eq!(
            parse_frontend_message(&invalid_utf8_describe_name).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let invalid_utf8_describe_name_lowercase_target = frontend_frame(b'D', b"p\xFF\0");
        assert_eq!(
            parse_frontend_message(&invalid_utf8_describe_name_lowercase_target).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let describe_with_trailing_bytes_after_name = frontend_frame(b'D', b"Sstmt\0\xFF");
        assert_eq!(
            parse_frontend_message(&describe_with_trailing_bytes_after_name).unwrap_err(),
            FrontendMessageError::UnterminatedDescribeName
        );

        let describe_unnamed_with_trailing_bytes_after_name = frontend_frame(b'D', b"p\0\xFF");
        assert_eq!(
            parse_frontend_message(&describe_unnamed_with_trailing_bytes_after_name).unwrap_err(),
            FrontendMessageError::UnterminatedDescribeName
        );

        let describe_unnamed_with_extra_null_segment = frontend_frame(b'D', b"p\0\0");
        assert_eq!(
            parse_frontend_message(&describe_unnamed_with_extra_null_segment).unwrap_err(),
            FrontendMessageError::UnterminatedDescribeName
        );

        let describe_unnamed_uppercase_with_trailing_bytes_after_name =
            frontend_frame(b'D', b"S\0\xFF");
        assert_eq!(
            parse_frontend_message(&describe_unnamed_uppercase_with_trailing_bytes_after_name)
                .unwrap_err(),
            FrontendMessageError::UnterminatedDescribeName
        );

        let empty_close = frontend_frame(b'C', b"");
        assert_eq!(
            parse_frontend_message(&empty_close).unwrap_err(),
            FrontendMessageError::TooShort
        );

        let truncated_close_name = frontend_frame(b'C', b"P");
        assert_eq!(
            parse_frontend_message(&truncated_close_name).unwrap_err(),
            FrontendMessageError::UnterminatedCloseName
        );

        let truncated_close_name_lowercase_target = frontend_frame(b'C', b"sstmt");
        assert_eq!(
            parse_frontend_message(&truncated_close_name_lowercase_target).unwrap_err(),
            FrontendMessageError::UnterminatedCloseName
        );

        let invalid_close_target = frontend_frame(b'C', b"Xstmt\0");
        assert_eq!(
            parse_frontend_message(&invalid_close_target).unwrap_err(),
            FrontendMessageError::InvalidCloseTarget
        );

        let close_with_embedded_null = frontend_frame(b'C', b"Sstmt\0extra\0");
        assert_eq!(
            parse_frontend_message(&close_with_embedded_null).unwrap_err(),
            FrontendMessageError::UnterminatedCloseName
        );

        let invalid_utf8_close_name = frontend_frame(b'C', b"S\xFF\0");
        assert_eq!(
            parse_frontend_message(&invalid_utf8_close_name).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let invalid_utf8_close_name_lowercase_target = frontend_frame(b'C', b"s\xFF\0");
        assert_eq!(
            parse_frontend_message(&invalid_utf8_close_name_lowercase_target).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let close_with_trailing_bytes_after_name = frontend_frame(b'C', b"Pportal\0\xFF");
        assert_eq!(
            parse_frontend_message(&close_with_trailing_bytes_after_name).unwrap_err(),
            FrontendMessageError::UnterminatedCloseName
        );

        let close_unnamed_with_trailing_bytes_after_name = frontend_frame(b'C', b"s\0\xFF");
        assert_eq!(
            parse_frontend_message(&close_unnamed_with_trailing_bytes_after_name).unwrap_err(),
            FrontendMessageError::UnterminatedCloseName
        );

        let close_unnamed_with_extra_null_segment = frontend_frame(b'C', b"s\0\0");
        assert_eq!(
            parse_frontend_message(&close_unnamed_with_extra_null_segment).unwrap_err(),
            FrontendMessageError::UnterminatedCloseName
        );

        let close_unnamed_uppercase_with_trailing_bytes_after_name =
            frontend_frame(b'C', b"P\0\xFF");
        assert_eq!(
            parse_frontend_message(&close_unnamed_uppercase_with_trailing_bytes_after_name)
                .unwrap_err(),
            FrontendMessageError::UnterminatedCloseName
        );

        let malformed_execute = frontend_frame(b'E', b"portal\0\0\0");
        assert_eq!(
            parse_frontend_message(&malformed_execute).unwrap_err(),
            FrontendMessageError::InvalidExecutePayload
        );

        let unterminated_execute_portal_name = frontend_frame(b'E', b"portal");
        assert_eq!(
            parse_frontend_message(&unterminated_execute_portal_name).unwrap_err(),
            FrontendMessageError::UnterminatedExecutePortalName
        );

        let execute_with_short_max_rows_payload = frontend_frame(b'E', b"portal\0\0\0\x01");
        assert_eq!(
            parse_frontend_message(&execute_with_short_max_rows_payload).unwrap_err(),
            FrontendMessageError::InvalidExecutePayload
        );

        let invalid_utf8_execute_portal_name = frontend_frame(b'E', &[0xFF, 0, 0, 0, 0, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_execute_portal_name).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let execute_with_embedded_null_portal_name = frontend_frame(b'E', b"portal\0extra\0\0\0\0");
        assert_eq!(
            parse_frontend_message(&execute_with_embedded_null_portal_name).unwrap_err(),
            FrontendMessageError::InvalidExecutePayload
        );

        let mut negative_max_rows_execute_payload = Vec::new();
        negative_max_rows_execute_payload.extend_from_slice(b"portal\0");
        negative_max_rows_execute_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        let negative_max_rows_execute = frontend_frame(b'E', &negative_max_rows_execute_payload);
        assert_eq!(
            parse_frontend_message(&negative_max_rows_execute).unwrap_err(),
            FrontendMessageError::InvalidExecutePayload
        );

        let mut malformed_execute_with_trailing_bytes_payload = Vec::new();
        malformed_execute_with_trailing_bytes_payload.extend_from_slice(b"portal\0");
        malformed_execute_with_trailing_bytes_payload.extend_from_slice(&5_i32.to_be_bytes());
        malformed_execute_with_trailing_bytes_payload.push(0xAA);
        let malformed_execute_with_trailing_bytes =
            frontend_frame(b'E', &malformed_execute_with_trailing_bytes_payload);
        assert_eq!(
            parse_frontend_message(&malformed_execute_with_trailing_bytes).unwrap_err(),
            FrontendMessageError::InvalidExecutePayload
        );

        let malformed_bind = frontend_frame(b'B', b"portal\0stmt\0\0\x01");
        assert_eq!(
            parse_frontend_message(&malformed_bind).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let unterminated_bind_portal_name = frontend_frame(b'B', b"portal");
        assert_eq!(
            parse_frontend_message(&unterminated_bind_portal_name).unwrap_err(),
            FrontendMessageError::UnterminatedBindPortalName
        );

        let unterminated_bind_statement_name = frontend_frame(b'B', b"portal\0stmt");
        assert_eq!(
            parse_frontend_message(&unterminated_bind_statement_name).unwrap_err(),
            FrontendMessageError::UnterminatedBindStatementName
        );

        let mut negative_bind_result_format_code_payload = Vec::new();
        negative_bind_result_format_code_payload.extend_from_slice(b"portal\0stmt\0");
        negative_bind_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_result_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
        negative_bind_result_format_code_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_bind_result_format_code =
            frontend_frame(b'B', &negative_bind_result_format_code_payload);
        assert_eq!(
            parse_frontend_message(&negative_bind_result_format_code).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut negative_bind_result_format_count_payload = Vec::new();
        negative_bind_result_format_count_payload.extend_from_slice(b"portal\0stmt\0");
        negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_result_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_bind_result_format_count =
            frontend_frame(b'B', &negative_bind_result_format_count_payload);
        assert_eq!(
            parse_frontend_message(&negative_bind_result_format_count).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let invalid_utf8_bind_portal_name =
            frontend_frame(b'B', &[0xFF, 0, b's', b't', b'm', b't', 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_bind_portal_name).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let invalid_utf8_bind_statement_name = frontend_frame(
            b'B',
            &[b'p', b'o', b'r', b't', b'a', b'l', 0, 0xFF, 0, 0, 0, 0, 0],
        );
        assert_eq!(
            parse_frontend_message(&invalid_utf8_bind_statement_name).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let mut negative_bind_format_count_payload = Vec::new();
        negative_bind_format_count_payload.extend_from_slice(b"portal\0stmt\0");
        negative_bind_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_bind_format_count = frontend_frame(b'B', &negative_bind_format_count_payload);
        assert_eq!(
            parse_frontend_message(&negative_bind_format_count).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut invalid_bind_format_count_payload = Vec::new();
        invalid_bind_format_count_payload.extend_from_slice(b"portal\0stmt\0");
        invalid_bind_format_count_payload.extend_from_slice(&2_i16.to_be_bytes());
        invalid_bind_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_bind_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
        invalid_bind_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
        invalid_bind_format_count_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        invalid_bind_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        let invalid_bind_format_count = frontend_frame(b'B', &invalid_bind_format_count_payload);
        assert_eq!(
            parse_frontend_message(&invalid_bind_format_count).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut invalid_bind_format_count_with_zero_parameters_payload = Vec::new();
        invalid_bind_format_count_with_zero_parameters_payload.extend_from_slice(b"portal\0stmt\0");
        invalid_bind_format_count_with_zero_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        invalid_bind_format_count_with_zero_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        invalid_bind_format_count_with_zero_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        invalid_bind_format_count_with_zero_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        invalid_bind_format_count_with_zero_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let invalid_bind_format_count_with_zero_parameters = frontend_frame(
            b'B',
            &invalid_bind_format_count_with_zero_parameters_payload,
        );
        assert_eq!(
            parse_frontend_message(&invalid_bind_format_count_with_zero_parameters).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut invalid_bind_format_code_payload = Vec::new();
        invalid_bind_format_code_payload.extend_from_slice(b"portal\0stmt\0");
        invalid_bind_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
        invalid_bind_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
        invalid_bind_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_bind_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        let invalid_bind_format_code = frontend_frame(b'B', &invalid_bind_format_code_payload);
        assert_eq!(
            parse_frontend_message(&invalid_bind_format_code).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut invalid_bind_shared_format_with_multiple_parameters_payload = Vec::new();
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(b"portal\0stmt\0");
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(&42_i32.to_be_bytes());
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        invalid_bind_shared_format_with_multiple_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let invalid_bind_shared_format_with_multiple_parameters = frontend_frame(
            b'B',
            &invalid_bind_shared_format_with_multiple_parameters_payload,
        );
        assert_eq!(
            parse_frontend_message(&invalid_bind_shared_format_with_multiple_parameters)
                .unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut bind_with_multiple_result_formats_payload = Vec::new();
        bind_with_multiple_result_formats_payload.extend_from_slice(b"portal\0stmt\0");
        bind_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_result_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
        bind_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_result_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
        let bind_with_multiple_result_formats =
            frontend_frame(b'B', &bind_with_multiple_result_formats_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_multiple_result_formats).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal".to_string(),
                statement_name: "stmt".to_string(),
                parameter_format_codes: vec![],
                parameters: vec![],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_with_zero_parameters_and_single_format_payload = Vec::new();
        bind_with_zero_parameters_and_single_format_payload.extend_from_slice(b"portal\0stmt\0");
        bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_with_zero_parameters_and_single_format =
            frontend_frame(b'B', &bind_with_zero_parameters_and_single_format_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_zero_parameters_and_single_format).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal".to_string(),
                statement_name: "stmt".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![],
                result_format_codes: vec![],
            }
        );

        let mut bind_with_zero_parameters_and_single_text_format_payload = Vec::new();
        bind_with_zero_parameters_and_single_text_format_payload
            .extend_from_slice(b"portal_text\0stmt_text\0");
        bind_with_zero_parameters_and_single_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_with_zero_parameters_and_single_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_zero_parameters_and_single_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_zero_parameters_and_single_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_with_zero_parameters_and_single_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_with_zero_parameters_and_single_text_format = frontend_frame(
            b'B',
            &bind_with_zero_parameters_and_single_text_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_with_zero_parameters_and_single_text_format).unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal_text".to_string(),
                statement_name: "stmt_text".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![],
                result_format_codes: vec![0],
            }
        );

        let mut bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload =
            Vec::new();
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(b"\0\0");
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(b"\x00\x00\x00\x2a");
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_with_shared_parameter_format_for_multiple_parameters = frontend_frame(
            b'B',
            &bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_unnamed_with_shared_parameter_format_for_multiple_parameters
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_shared_parameter_format_payload =
            Vec::new();
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice("pörtal_shared\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(b"\x00\x00\x00\x2a");
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_shared_parameter_format = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_shared_parameter_format_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_named_portal_with_unnamed_statement_shared_parameter_format,
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_shared".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_shared_parameter_format_payload =
            Vec::new();
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(b"\0stmt_shared_named\0");
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(b"\x00\x00\x00\x2a");
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_shared_parameter_format = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_shared_parameter_format_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_unnamed_portal_with_named_statement_shared_parameter_format,
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_shared_named".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_named_portal_with_named_statement_shared_parameter_format_payload = Vec::new();
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice("pörtal_shared_pair\0stmt_shared_pair\0".as_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(b"\x00\x00\x00\x2a");
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_parameter_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_named_statement_shared_parameter_format = frontend_frame(
            b'B',
            &bind_named_portal_with_named_statement_shared_parameter_format_payload,
        );
        assert_eq!(
            parse_frontend_message(
                &bind_named_portal_with_named_statement_shared_parameter_format,
            )
            .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_shared_pair".to_string(),
                statement_name: "stmt_shared_pair".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_unnamed_with_shared_text_format_for_multiple_parameters_payload = Vec::new();
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(b"\0\0");
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice("héllo".as_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_with_shared_text_format_for_multiple_parameters = frontend_frame(
            b'B',
            &bind_unnamed_with_shared_text_format_for_multiple_parameters_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_with_shared_text_format_for_multiple_parameters)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec()), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_named_portal_with_unnamed_statement_shared_text_format_payload = Vec::new();
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice("pörtal_shared_text\0\0".as_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice("héllo".as_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_unnamed_statement_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_unnamed_statement_shared_text_format = frontend_frame(
            b'B',
            &bind_named_portal_with_unnamed_statement_shared_text_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_unnamed_statement_shared_text_format)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_shared_text".to_string(),
                statement_name: String::new(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec()), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_unnamed_portal_with_named_statement_shared_text_format_payload = Vec::new();
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(b"\0stmt_shared_text_named\0");
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice("héllo".as_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_unnamed_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_unnamed_portal_with_named_statement_shared_text_format = frontend_frame(
            b'B',
            &bind_unnamed_portal_with_named_statement_shared_text_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_unnamed_portal_with_named_statement_shared_text_format)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: String::new(),
                statement_name: "stmt_shared_text_named".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec()), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_named_portal_with_named_statement_shared_text_format_payload = Vec::new();
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice("pörtal_shared_text_pair\0stmt_shared_text_pair\0".as_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice("héllo".as_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_named_portal_with_named_statement_shared_text_format_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_named_portal_with_named_statement_shared_text_format = frontend_frame(
            b'B',
            &bind_named_portal_with_named_statement_shared_text_format_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_named_portal_with_named_statement_shared_text_format)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: "pörtal_shared_text_pair".to_string(),
                statement_name: "stmt_shared_text_pair".to_string(),
                parameter_format_codes: vec![0],
                parameters: vec![Some("héllo".as_bytes().to_vec()), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_with_shared_parameter_format_for_multiple_parameters_payload = Vec::new();
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(b"portal_shared\0stmt_shared\0");
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&4_i32.to_be_bytes());
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(b"\x00\x00\x00\x2a");
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&(-1_i32).to_be_bytes());
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_shared_parameter_format_for_multiple_parameters_payload
            .extend_from_slice(&1_i16.to_be_bytes());
        let bind_with_shared_parameter_format_for_multiple_parameters = frontend_frame(
            b'B',
            &bind_with_shared_parameter_format_for_multiple_parameters_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_with_shared_parameter_format_for_multiple_parameters)
                .unwrap(),
            FrontendMessage::Bind {
                portal_name: "portal_shared".to_string(),
                statement_name: "stmt_shared".to_string(),
                parameter_format_codes: vec![1],
                parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
                result_format_codes: vec![0, 1],
            }
        );

        let mut bind_with_multiple_parameter_formats_and_invalid_code_payload = Vec::new();
        bind_with_multiple_parameter_formats_and_invalid_code_payload
            .extend_from_slice(b"portal\0stmt\0");
        bind_with_multiple_parameter_formats_and_invalid_code_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_with_multiple_parameter_formats_and_invalid_code_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_parameter_formats_and_invalid_code_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_with_multiple_parameter_formats_and_invalid_code_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_parameter_formats_and_invalid_code_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        let bind_with_multiple_parameter_formats_and_invalid_code = frontend_frame(
            b'B',
            &bind_with_multiple_parameter_formats_and_invalid_code_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_with_multiple_parameter_formats_and_invalid_code)
                .unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut bind_with_negative_parameter_format_code_payload = Vec::new();
        bind_with_negative_parameter_format_code_payload.extend_from_slice(b"portal\0stmt\0");
        bind_with_negative_parameter_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_negative_parameter_format_code_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        bind_with_negative_parameter_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_negative_parameter_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        let bind_with_negative_parameter_format_code =
            frontend_frame(b'B', &bind_with_negative_parameter_format_code_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_negative_parameter_format_code).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut bind_with_invalid_single_result_format_code_payload = Vec::new();
        bind_with_invalid_single_result_format_code_payload.extend_from_slice(b"portal\0stmt\0");
        bind_with_invalid_single_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_invalid_single_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        bind_with_invalid_single_result_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
        bind_with_invalid_single_result_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
        let bind_with_invalid_single_result_format_code =
            frontend_frame(b'B', &bind_with_invalid_single_result_format_code_payload);
        assert_eq!(
            parse_frontend_message(&bind_with_invalid_single_result_format_code).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut bind_with_multiple_result_formats_and_invalid_code_payload = Vec::new();
        bind_with_multiple_result_formats_and_invalid_code_payload
            .extend_from_slice(b"portal\0stmt\0");
        bind_with_multiple_result_formats_and_invalid_code_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_result_formats_and_invalid_code_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_result_formats_and_invalid_code_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        bind_with_multiple_result_formats_and_invalid_code_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        bind_with_multiple_result_formats_and_invalid_code_payload
            .extend_from_slice(&2_i16.to_be_bytes());
        let bind_with_multiple_result_formats_and_invalid_code = frontend_frame(
            b'B',
            &bind_with_multiple_result_formats_and_invalid_code_payload,
        );
        assert_eq!(
            parse_frontend_message(&bind_with_multiple_result_formats_and_invalid_code)
                .unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_result_format_payload = Vec::new();
        truncated_bind_result_format_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_result_format_payload.extend_from_slice(&1_i16.to_be_bytes());
        let truncated_bind_result_format =
            frontend_frame(b'B', &truncated_bind_result_format_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_result_format).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_result_format_codes_payload = Vec::new();
        truncated_bind_result_format_codes_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_result_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_result_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_result_format_codes_payload.extend_from_slice(&2_i16.to_be_bytes());
        truncated_bind_result_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_result_format_codes_payload.push(0x00);
        let truncated_bind_result_format_codes =
            frontend_frame(b'B', &truncated_bind_result_format_codes_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_result_format_codes).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_format_count_field_payload = Vec::new();
        truncated_bind_format_count_field_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_format_count_field_payload.push(0x00);
        let truncated_bind_format_count_field =
            frontend_frame(b'B', &truncated_bind_format_count_field_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_format_count_field).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_parameter_format_codes_payload = Vec::new();
        truncated_bind_parameter_format_codes_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_parameter_format_codes_payload.extend_from_slice(&2_i16.to_be_bytes());
        truncated_bind_parameter_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
        let truncated_bind_parameter_format_codes =
            frontend_frame(b'B', &truncated_bind_parameter_format_codes_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_parameter_format_codes).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_parameter_count_field_payload = Vec::new();
        truncated_bind_parameter_count_field_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_parameter_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_parameter_count_field_payload.push(0x00);
        let truncated_bind_parameter_count_field =
            frontend_frame(b'B', &truncated_bind_parameter_count_field_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_parameter_count_field).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut negative_bind_parameter_count_payload = Vec::new();
        negative_bind_parameter_count_payload.extend_from_slice(b"portal\0stmt\0");
        negative_bind_parameter_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_parameter_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_bind_parameter_count =
            frontend_frame(b'B', &negative_bind_parameter_count_payload);
        assert_eq!(
            parse_frontend_message(&negative_bind_parameter_count).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut negative_bind_parameter_length_payload = Vec::new();
        negative_bind_parameter_length_payload.extend_from_slice(b"portal\0stmt\0");
        negative_bind_parameter_length_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_parameter_length_payload.extend_from_slice(&1_i16.to_be_bytes());
        negative_bind_parameter_length_payload.extend_from_slice(&(-2_i32).to_be_bytes());
        negative_bind_parameter_length_payload.extend_from_slice(&0_i16.to_be_bytes());
        let negative_bind_parameter_length =
            frontend_frame(b'B', &negative_bind_parameter_length_payload);
        assert_eq!(
            parse_frontend_message(&negative_bind_parameter_length).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_parameter_length_field_payload = Vec::new();
        truncated_bind_parameter_length_field_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_parameter_length_field_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_parameter_length_field_payload.extend_from_slice(&1_i16.to_be_bytes());
        truncated_bind_parameter_length_field_payload.extend_from_slice(&[0x00, 0x00, 0x00]);
        let truncated_bind_parameter_length_field =
            frontend_frame(b'B', &truncated_bind_parameter_length_field_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_parameter_length_field).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_parameter_value_payload = Vec::new();
        truncated_bind_parameter_value_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_parameter_value_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_parameter_value_payload.extend_from_slice(&1_i16.to_be_bytes());
        truncated_bind_parameter_value_payload.extend_from_slice(&3_i32.to_be_bytes());
        truncated_bind_parameter_value_payload.extend_from_slice(b"ab");
        let truncated_bind_parameter_value =
            frontend_frame(b'B', &truncated_bind_parameter_value_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_parameter_value).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut truncated_bind_result_format_count_field_payload = Vec::new();
        truncated_bind_result_format_count_field_payload.extend_from_slice(b"portal\0stmt\0");
        truncated_bind_result_format_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_result_format_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_bind_result_format_count_field_payload.push(0x00);
        let truncated_bind_result_format_count_field =
            frontend_frame(b'B', &truncated_bind_result_format_count_field_payload);
        assert_eq!(
            parse_frontend_message(&truncated_bind_result_format_count_field).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut negative_bind_result_format_count_payload = Vec::new();
        negative_bind_result_format_count_payload.extend_from_slice(b"portal\0stmt\0");
        negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_bind_result_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_bind_result_format_count =
            frontend_frame(b'B', &negative_bind_result_format_count_payload);
        assert_eq!(
            parse_frontend_message(&negative_bind_result_format_count).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let mut malformed_bind_with_trailing_bytes_payload = Vec::new();
        malformed_bind_with_trailing_bytes_payload.extend_from_slice(b"portal\0stmt\0");
        malformed_bind_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
        malformed_bind_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
        malformed_bind_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
        malformed_bind_with_trailing_bytes_payload.push(0x7F);
        let malformed_bind_with_trailing_bytes =
            frontend_frame(b'B', &malformed_bind_with_trailing_bytes_payload);
        assert_eq!(
            parse_frontend_message(&malformed_bind_with_trailing_bytes).unwrap_err(),
            FrontendMessageError::InvalidBindPayload
        );

        let truncated_function_call_oid_field = frontend_frame(b'F', &[0x00, 0x00, 0x00]);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_oid_field).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let malformed_function_call = frontend_frame(b'F', b"\0\0\0*");
        assert_eq!(
            parse_frontend_message(&malformed_function_call).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_result_format_payload = Vec::new();
        truncated_function_call_result_format_payload.extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        let truncated_function_call_result_format =
            frontend_frame(b'F', &truncated_function_call_result_format_payload);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_result_format).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_result_format_count_field_payload = Vec::new();
        truncated_function_call_result_format_count_field_payload
            .extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_result_format_count_field_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_result_format_count_field_payload
            .extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_result_format_count_field_payload.push(0x00);
        let truncated_function_call_result_format_count_field = frontend_frame(
            b'F',
            &truncated_function_call_result_format_count_field_payload,
        );
        assert_eq!(
            parse_frontend_message(&truncated_function_call_result_format_count_field).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut negative_function_call_format_count_payload = Vec::new();
        negative_function_call_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
        negative_function_call_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_function_call_format_count =
            frontend_frame(b'F', &negative_function_call_format_count_payload);
        assert_eq!(
            parse_frontend_message(&negative_function_call_format_count).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_arg_count_field_payload = Vec::new();
        truncated_function_call_arg_count_field_payload.extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_arg_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_arg_count_field_payload.push(0x00);
        let truncated_function_call_arg_count_field =
            frontend_frame(b'F', &truncated_function_call_arg_count_field_payload);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_arg_count_field).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut negative_function_call_shared_format_code_payload = Vec::new();
        negative_function_call_shared_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
        negative_function_call_shared_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
        negative_function_call_shared_format_code_payload
            .extend_from_slice(&(-1_i16).to_be_bytes());
        negative_function_call_shared_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_function_call_shared_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        let negative_function_call_shared_format_code =
            frontend_frame(b'F', &negative_function_call_shared_format_code_payload);
        assert_eq!(
            parse_frontend_message(&negative_function_call_shared_format_code).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut invalid_function_call_format_code_payload = Vec::new();
        invalid_function_call_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
        invalid_function_call_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
        invalid_function_call_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
        invalid_function_call_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        let invalid_function_call_format_code =
            frontend_frame(b'F', &invalid_function_call_format_code_payload);
        assert_eq!(
            parse_frontend_message(&invalid_function_call_format_code).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut mismatched_function_call_format_count_payload = Vec::new();
        mismatched_function_call_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
        mismatched_function_call_format_count_payload.extend_from_slice(&2_i16.to_be_bytes());
        mismatched_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        mismatched_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
        mismatched_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
        mismatched_function_call_format_count_payload.extend_from_slice(&4_i32.to_be_bytes());
        mismatched_function_call_format_count_payload.extend_from_slice(&7_i32.to_be_bytes());
        mismatched_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        let mismatched_function_call_format_count =
            frontend_frame(b'F', &mismatched_function_call_format_count_payload);
        assert_eq!(
            parse_frontend_message(&mismatched_function_call_format_count).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_format_codes_payload = Vec::new();
        truncated_function_call_format_codes_payload.extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_format_codes_payload.extend_from_slice(&2_i16.to_be_bytes());
        truncated_function_call_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
        let truncated_function_call_format_codes =
            frontend_frame(b'F', &truncated_function_call_format_codes_payload);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_format_codes).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_argument_value_payload = Vec::new();
        truncated_function_call_argument_value_payload.extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_argument_value_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_argument_value_payload.extend_from_slice(&1_i16.to_be_bytes());
        truncated_function_call_argument_value_payload.extend_from_slice(&3_i32.to_be_bytes());
        truncated_function_call_argument_value_payload.extend_from_slice(b"ab");
        let truncated_function_call_argument_value =
            frontend_frame(b'F', &truncated_function_call_argument_value_payload);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_argument_value).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut invalid_function_call_multi_format_code_payload = Vec::new();
        invalid_function_call_multi_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
        invalid_function_call_multi_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
        invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_multi_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
        invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        let invalid_function_call_multi_format_code =
            frontend_frame(b'F', &invalid_function_call_multi_format_code_payload);
        assert_eq!(
            parse_frontend_message(&invalid_function_call_multi_format_code).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut invalid_function_call_format_count_payload = Vec::new();
        invalid_function_call_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
        invalid_function_call_format_count_payload.extend_from_slice(&2_i16.to_be_bytes());
        invalid_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
        invalid_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
        invalid_function_call_format_count_payload.extend_from_slice(&(-1_i32).to_be_bytes());
        invalid_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        let invalid_function_call_format_count =
            frontend_frame(b'F', &invalid_function_call_format_count_payload);
        assert_eq!(
            parse_frontend_message(&invalid_function_call_format_count).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut invalid_function_call_zero_arg_multi_format_payload = Vec::new();
        invalid_function_call_zero_arg_multi_format_payload
            .extend_from_slice(&42_u32.to_be_bytes());
        invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&2_i16.to_be_bytes());
        invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&1_i16.to_be_bytes());
        invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&0_i16.to_be_bytes());
        let invalid_function_call_zero_arg_multi_format =
            frontend_frame(b'F', &invalid_function_call_zero_arg_multi_format_payload);
        assert_eq!(
            parse_frontend_message(&invalid_function_call_zero_arg_multi_format).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut negative_function_call_arg_count_payload = Vec::new();
        negative_function_call_arg_count_payload.extend_from_slice(&42_u32.to_be_bytes());
        negative_function_call_arg_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_function_call_arg_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_function_call_arg_count =
            frontend_frame(b'F', &negative_function_call_arg_count_payload);
        assert_eq!(
            parse_frontend_message(&negative_function_call_arg_count).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut negative_function_call_arg_length_payload = Vec::new();
        negative_function_call_arg_length_payload.extend_from_slice(&42_u32.to_be_bytes());
        negative_function_call_arg_length_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_function_call_arg_length_payload.extend_from_slice(&1_i16.to_be_bytes());
        negative_function_call_arg_length_payload.extend_from_slice(&(-2_i32).to_be_bytes());
        negative_function_call_arg_length_payload.extend_from_slice(&0_i16.to_be_bytes());
        let negative_function_call_arg_length =
            frontend_frame(b'F', &negative_function_call_arg_length_payload);
        assert_eq!(
            parse_frontend_message(&negative_function_call_arg_length).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_arg_length_field_payload = Vec::new();
        truncated_function_call_arg_length_field_payload.extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_arg_length_field_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_arg_length_field_payload.extend_from_slice(&1_i16.to_be_bytes());
        truncated_function_call_arg_length_field_payload.extend_from_slice(&[0x00, 0x00, 0x00]);
        let truncated_function_call_arg_length_field =
            frontend_frame(b'F', &truncated_function_call_arg_length_field_payload);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_arg_length_field).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_arg_value_payload = Vec::new();
        truncated_function_call_arg_value_payload.extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_arg_value_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_arg_value_payload.extend_from_slice(&1_i16.to_be_bytes());
        truncated_function_call_arg_value_payload.extend_from_slice(&3_i32.to_be_bytes());
        truncated_function_call_arg_value_payload.extend_from_slice(b"ab");
        truncated_function_call_arg_value_payload.extend_from_slice(&0_i16.to_be_bytes());
        let truncated_function_call_arg_value =
            frontend_frame(b'F', &truncated_function_call_arg_value_payload);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_arg_value).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut malformed_function_call_with_trailing_bytes_payload = Vec::new();
        malformed_function_call_with_trailing_bytes_payload
            .extend_from_slice(&42_u32.to_be_bytes());
        malformed_function_call_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
        malformed_function_call_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
        malformed_function_call_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
        malformed_function_call_with_trailing_bytes_payload.push(0xAA);
        let malformed_function_call_with_trailing_bytes =
            frontend_frame(b'F', &malformed_function_call_with_trailing_bytes_payload);
        assert_eq!(
            parse_frontend_message(&malformed_function_call_with_trailing_bytes).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut negative_function_call_result_format_count_payload = Vec::new();
        negative_function_call_result_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
        negative_function_call_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_function_call_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_function_call_result_format_count_payload
            .extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_function_call_result_format_count =
            frontend_frame(b'F', &negative_function_call_result_format_count_payload);
        assert_eq!(
            parse_frontend_message(&negative_function_call_result_format_count).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut negative_function_call_result_format_code_payload = Vec::new();
        negative_function_call_result_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
        negative_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        negative_function_call_result_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
        negative_function_call_result_format_code_payload
            .extend_from_slice(&(-1_i16).to_be_bytes());
        let negative_function_call_result_format_code =
            frontend_frame(b'F', &negative_function_call_result_format_code_payload);
        assert_eq!(
            parse_frontend_message(&negative_function_call_result_format_code).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut truncated_function_call_result_format_code_payload = Vec::new();
        truncated_function_call_result_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
        truncated_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        truncated_function_call_result_format_code_payload.push(0x00);
        let truncated_function_call_result_format_code =
            frontend_frame(b'F', &truncated_function_call_result_format_code_payload);
        assert_eq!(
            parse_frontend_message(&truncated_function_call_result_format_code).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let mut invalid_function_call_result_format_code_payload = Vec::new();
        invalid_function_call_result_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
        invalid_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
        invalid_function_call_result_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
        let invalid_function_call_result_format_code =
            frontend_frame(b'F', &invalid_function_call_result_format_code_payload);
        assert_eq!(
            parse_frontend_message(&invalid_function_call_result_format_code).unwrap_err(),
            FrontendMessageError::InvalidFunctionCallPayload
        );

        let malformed_copy_done = frontend_frame(b'c', &[0]);
        assert_eq!(
            parse_frontend_message(&malformed_copy_done).unwrap_err(),
            FrontendMessageError::LengthMismatch {
                expected: 5,
                actual: 6,
            }
        );

        let malformed_terminate = frontend_frame(b'X', &[0]);
        assert_eq!(
            parse_frontend_message(&malformed_terminate).unwrap_err(),
            FrontendMessageError::LengthMismatch {
                expected: 5,
                actual: 6,
            }
        );

        let malformed_sync = frontend_frame(b'S', &[0]);
        assert_eq!(
            parse_frontend_message(&malformed_sync).unwrap_err(),
            FrontendMessageError::LengthMismatch {
                expected: 5,
                actual: 6,
            }
        );

        let malformed_flush = frontend_frame(b'H', &[0]);
        assert_eq!(
            parse_frontend_message(&malformed_flush).unwrap_err(),
            FrontendMessageError::LengthMismatch {
                expected: 5,
                actual: 6,
            }
        );

        let unterminated_copy_fail = frontend_frame(b'f', b"bad row");
        assert_eq!(
            parse_frontend_message(&unterminated_copy_fail).unwrap_err(),
            FrontendMessageError::UnterminatedCopyFail
        );

        let copy_fail_with_embedded_null = frontend_frame(b'f', b"bad\0row\0");
        assert_eq!(
            parse_frontend_message(&copy_fail_with_embedded_null).unwrap_err(),
            FrontendMessageError::UnterminatedCopyFail
        );

        let invalid_utf8_copy_fail = frontend_frame(b'f', &[0xFF, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_copy_fail).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let invalid_utf8_query = frontend_frame(b'Q', &[0xFF, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_query).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let password_with_embedded_null_and_terminator = frontend_frame(b'p', b"secret\0extra\0");
        assert_eq!(
            parse_frontend_message(&password_with_embedded_null_and_terminator).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let invalid_utf8_password = frontend_frame(b'p', &[0xFF, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_password).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let binary_sasl_response_with_embedded_null = frontend_frame(b'p', b"proof\0chunk");
        assert_eq!(
            parse_frontend_message(&binary_sasl_response_with_embedded_null).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let invalid_utf8_parse_statement = frontend_frame(b'P', &[0xFF, 0, b'S', 0, 0, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_parse_statement).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let invalid_utf8_describe_name = frontend_frame(b'D', &[b'S', 0xFF, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_describe_name).unwrap_err(),
            FrontendMessageError::InvalidUtf8
        );

        let unsupported = frontend_frame(b'V', &[]);
        assert_eq!(
            parse_frontend_message(&unsupported).unwrap_err(),
            FrontendMessageError::UnsupportedTag(b'V')
        );
    }

    #[test]
    fn parses_minimal_relational_sql_subset() {
        assert_eq!(
            parse_command("CREATE TABLE people (id INT, name TEXT)").unwrap(),
            Command::CreateTable(CreateTable {
                table: "people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                ],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );

        assert_eq!(
            parse_command("CREATE TABLE keyed_people (id INT PRIMARY KEY, name TEXT)").unwrap(),
            Command::CreateTable(CreateTable {
                table: "keyed_people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                ],
                primary_key: Some(PrimaryKey {
                    name: None,
                    column: "id".to_string(),
                    columns: vec!["id".to_string()],
                }),
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );

        assert_eq!(
            parse_command(
                "CREATE TABLE unique_people (id INT, name TEXT UNIQUE, CONSTRAINT unique_people_id_key UNIQUE (id))"
            )
            .unwrap(),
            Command::CreateTable(CreateTable {
                table: "unique_people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
default: None,
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
default: None,
                    },
                ],
                primary_key: None,
                unique_constraints: vec![
                    UniqueConstraint {
                        name: None,
                        column: "name".to_string(),
                        columns: vec!["name".to_string()],
                    },
                    UniqueConstraint {
                        name: Some("unique_people_id_key".to_string()),
                        column: "id".to_string(),
                        columns: vec!["id".to_string()],
                    },
                ],
                check_constraints: Vec::new(),
            })
        );

        assert_eq!(
            parse_command(
                "CREATE TABLE check_people (id INT, name TEXT, CONSTRAINT check_people_id_positive CHECK (id > 0), CHECK (name = 'Ada'))"
            )
            .unwrap(),
            Command::CreateTable(CreateTable {
                table: "check_people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                ],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: vec![
                    CheckConstraint {
                        name: Some("check_people_id_positive".to_string()),
                        filter: SelectFilter {
                            column: "id".to_string(),
                            op: SelectFilterOp::Gt,
                            value: SqlValue::Int4(0),
                        },
                    },
                    CheckConstraint {
                        name: None,
                        filter: SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Text("Ada".to_string()),
                        },
                    },
                ],
            })
        );
        assert!(matches!(
            parse_command("CREATE TABLE bad_check_people (id INT, CHECK (missing > 0))"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE TABLE bad_check_people (id INT, CHECK (id BETWEEN 1 AND 3))"),
            Err(ParseError::InvalidRelationalSql)
        ));

        assert_eq!(
            parse_command("ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_pkey PRIMARY KEY (id)").unwrap(),
            Command::AddPrimaryKey(AddPrimaryKey {
                table: "keyed_people".to_string(),
                name: "keyed_people_pkey".to_string(),
                column: "id".to_string(),
                columns: vec!["id".to_string()],
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_name_key UNIQUE (name)").unwrap(),
            Command::AddUniqueConstraint(AddUniqueConstraint {
                table: "keyed_people".to_string(),
                name: "keyed_people_name_key".to_string(),
                column: "name".to_string(),
                columns: vec!["name".to_string()],
            })
        );
        assert_eq!(
            parse_command(
                "ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_id_positive CHECK (id > 0)"
            )
            .unwrap(),
            Command::AddCheckConstraint(AddCheckConstraint {
                table: "keyed_people".to_string(),
                name: "keyed_people_id_positive".to_string(),
                filter: SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gt,
                    value: SqlValue::Int4(0),
                },
            })
        );
        assert!(matches!(
            parse_command(
                "ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_id_between CHECK (id BETWEEN 1 AND 3)"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command(
                "ALTER TABLE ONLY public.orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES public.customers(id)"
            )
            .unwrap(),
            Command::AddForeignKey(AddForeignKey {
                table: "orders".to_string(),
                name: "orders_customer_fk".to_string(),
                column: "customer_id".to_string(),
                referenced_table: "customers".to_string(),
                referenced_column: "id".to_string(),
            })
        );
        assert!(matches!(
            parse_command(
                "ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id, tenant_id) REFERENCES customers(id, tenant_id)"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command(
                "ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES private.customers(id)"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command(
                "ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command(
                "ALTER TABLE IF EXISTS ONLY public.keyed_people DROP CONSTRAINT IF EXISTS keyed_people_name_key"
            )
            .unwrap(),
            Command::DropConstraint(DropConstraint {
                table: "keyed_people".to_string(),
                name: "keyed_people_name_key".to_string(),
                table_if_exists: true,
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE ONLY keyed_people DROP CONSTRAINT keyed_people_pkey")
                .unwrap(),
            Command::DropConstraint(DropConstraint {
                table: "keyed_people".to_string(),
                name: "keyed_people_pkey".to_string(),
                table_if_exists: false,
                if_exists: false,
            })
        );
        assert!(parse_command(
            "ALTER TABLE ONLY public.keyed_people DROP CONSTRAINT keyed_people_pkey CASCADE"
        )
        .is_err());
        assert_eq!(
            parse_command(
                "ALTER TABLE IF EXISTS ONLY public.keyed_people RENAME TO archived_people"
            )
            .unwrap(),
            Command::RenameTable(RenameTable {
                old_name: "keyed_people".to_string(),
                new_name: "archived_people".to_string(),
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE public.keyed_people RENAME TO renamed_people").unwrap(),
            Command::RenameTable(RenameTable {
                old_name: "keyed_people".to_string(),
                new_name: "renamed_people".to_string(),
                if_exists: false,
            })
        );
        assert!(
            parse_command("ALTER TABLE public.keyed_people RENAME TO public.renamed_people")
                .is_err()
        );
        assert!(
            parse_command("ALTER TABLE public.keyed_people RENAME TO renamed_people CASCADE")
                .is_err()
        );
        assert_eq!(
            parse_command(
                "ALTER TABLE ONLY public.keyed_people RENAME COLUMN name TO display_name"
            )
            .unwrap(),
            Command::RenameColumn(RenameColumn {
                table: "keyed_people".to_string(),
                old_name: "name".to_string(),
                new_name: "display_name".to_string(),
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE public.keyed_people RENAME id TO person_id").unwrap(),
            Command::RenameColumn(RenameColumn {
                table: "keyed_people".to_string(),
                old_name: "id".to_string(),
                new_name: "person_id".to_string(),
            })
        );
        assert!(parse_command(
            "ALTER TABLE public.keyed_people RENAME COLUMN name TO display_name CASCADE"
        )
        .is_err());
        assert_eq!(
            parse_command(
                "ALTER TABLE IF EXISTS ONLY public.keyed_people RENAME CONSTRAINT keyed_people_pkey TO keyed_people_id_pkey"
            )
            .unwrap(),
            Command::RenameConstraint(RenameConstraint {
                table: "keyed_people".to_string(),
                old_name: "keyed_people_pkey".to_string(),
                new_name: "keyed_people_id_pkey".to_string(),
                table_if_exists: true,
            })
        );
        assert!(parse_command(
            "ALTER TABLE public.keyed_people RENAME CONSTRAINT keyed_people_pkey TO keyed_people_id_pkey CASCADE"
        )
        .is_err());

        assert_eq!(
            parse_command("COMMENT ON DATABASE postgres IS 'primary database'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Database {
                    database: "postgres".to_string(),
                },
                comment: Some("primary database".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON SCHEMA public IS 'application schema'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Schema {
                    schema: "public".to_string(),
                },
                comment: Some("application schema".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON ROLE postgres IS 'bootstrap role'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Role {
                    role: "postgres".to_string(),
                },
                comment: Some("bootstrap role".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON TABLESPACE pg_default IS 'default storage'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Tablespace {
                    tablespace: "pg_default".to_string(),
                },
                comment: Some("default storage".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON TABLE public.people IS 'lookup people'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Table {
                    table: "people".to_string(),
                },
                comment: Some("lookup people".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON COLUMN public.people.name IS 'display name'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Column {
                    table: "people".to_string(),
                    column: "name".to_string(),
                },
                comment: Some("display name".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON COLUMN public.people.name IS NULL").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Column {
                    table: "people".to_string(),
                    column: "name".to_string(),
                },
                comment: None,
            })
        );

        assert_eq!(
            parse_command("COMMENT ON INDEX public.people_name_idx IS 'lookup index'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Index {
                    index: "people_name_idx".to_string(),
                },
                comment: Some("lookup index".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON VIEW public.active_people IS 'active people view'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::View {
                    view: "active_people".to_string(),
                },
                comment: Some("active people view".to_string()),
            })
        );

        assert_eq!(
            parse_command("COMMENT ON CONSTRAINT people_pkey ON public.people IS 'row identity'")
                .unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Constraint {
                    table: "people".to_string(),
                    constraint: "people_pkey".to_string(),
                },
                comment: Some("row identity".to_string()),
            })
        );

        assert_eq!(
            parse_command("CREATE TABLE typed_people (id INT4, owner pg_catalog.text)").unwrap(),
            Command::CreateTable(CreateTable {
                table: "typed_people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                    ColumnDef {
                        name: "owner".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                ],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );

        assert_eq!(
            parse_command("CREATE TABLE public.dump_people (id integer, name text)").unwrap(),
            Command::CreateTable(CreateTable {
                table: "dump_people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                ],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );
        assert!(matches!(
            parse_command("CREATE TABLE private.dump_people (id integer)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command(
                "CREATE TABLE default_people (id INT DEFAULT 7, name TEXT DEFAULT 'Ada''s'::text)"
            )
            .unwrap(),
            Command::CreateTable(CreateTable {
                table: "default_people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: Some(ColumnDefault::Literal(SqlValue::Int4(7))),
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: Some(ColumnDefault::Literal(SqlValue::Text("Ada's".to_string()))),
                    },
                ],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );
        assert!(matches!(
            parse_command("CREATE TABLE invalid_default (id INT DEFAULT 'bad'::text)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("CREATE TABLE serial_people (id SERIAL PRIMARY KEY, name TEXT)").unwrap(),
            Command::CreateTable(CreateTable {
                table: "serial_people".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: Some(ColumnDefault::SequenceNextVal {
                            sequence: "serial_people_id_seq".to_string(),
                            create_if_missing: true,
                        }),
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                ],
                primary_key: Some(PrimaryKey {
                    name: None,
                    column: "id".to_string(),
                    columns: vec!["id".to_string()],
                }),
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );
        assert_eq!(
            parse_command(
                "CREATE TABLE seq_default_people (id INT DEFAULT nextval('public.people_seq'::regclass))"
            )
            .unwrap(),
            Command::CreateTable(CreateTable {
                table: "seq_default_people".to_string(),
                columns: vec![ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
default: Some(ColumnDefault::SequenceNextVal {
                        sequence: "people_seq".to_string(),
                        create_if_missing: false,
                    }),
                }],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );
        assert!(matches!(
            parse_command("CREATE TABLE invalid_serial (id TEXT DEFAULT nextval('people_seq'))"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("CREATE TABLE type_named_columns (integer_col integer, text_col text)")
                .unwrap(),
            Command::CreateTable(CreateTable {
                table: "type_named_columns".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "integer_col".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                    ColumnDef {
                        name: "text_col".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                ],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE ONLY public.default_people ALTER COLUMN name SET DEFAULT 'Grace'::text").unwrap(),
            Command::AlterColumnDefault(AlterColumnDefault {
                table: "default_people".to_string(),
                column: "name".to_string(),
                default: Some(ColumnDefault::Literal(SqlValue::Text("Grace".to_string()))),
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE ONLY public.default_people ALTER COLUMN id SET DEFAULT nextval('public.default_people_id_seq'::regclass)").unwrap(),
            Command::AlterColumnDefault(AlterColumnDefault {
                table: "default_people".to_string(),
                column: "id".to_string(),
                default: Some(ColumnDefault::SequenceNextVal {
                    sequence: "default_people_id_seq".to_string(),
                    create_if_missing: false,
                }),
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE public.default_people ALTER name DROP DEFAULT").unwrap(),
            Command::AlterColumnDefault(AlterColumnDefault {
                table: "default_people".to_string(),
                column: "name".to_string(),
                default: None,
            })
        );
        assert_eq!(
            parse_command(
                "ALTER TABLE ONLY public.default_people ADD COLUMN tag TEXT DEFAULT 'new'::text"
            )
            .unwrap(),
            Command::AddColumn(AddColumn {
                table: "default_people".to_string(),
                column: ColumnDef {
                    name: "tag".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: Some(ColumnDefault::Literal(SqlValue::Text("new".to_string()))),
                },
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE public.default_people ADD bucket INT DEFAULT 4").unwrap(),
            Command::AddColumn(AddColumn {
                table: "default_people".to_string(),
                column: ColumnDef {
                    name: "bucket".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: Some(ColumnDefault::Literal(SqlValue::Int4(4))),
                },
            })
        );
        assert_eq!(
            parse_command(
                "ALTER TABLE public.default_people ADD bucket INT DEFAULT nextval('public.default_bucket_seq'::regclass)"
            )
            .unwrap(),
            Command::AddColumn(AddColumn {
                table: "default_people".to_string(),
                column: ColumnDef {
                    name: "bucket".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
default: Some(ColumnDefault::SequenceNextVal {
                        sequence: "default_bucket_seq".to_string(),
                        create_if_missing: false,
                    }),
                },
            })
        );
        assert!(matches!(
            parse_command(
                "ALTER TABLE private.default_people ADD COLUMN tag TEXT DEFAULT 'bad'::text"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("ALTER TABLE ONLY public.default_people DROP COLUMN tag").unwrap(),
            Command::DropColumn(DropColumn {
                table: "default_people".to_string(),
                column: "tag".to_string(),
            })
        );
        assert_eq!(
            parse_command("ALTER TABLE default_people DROP bucket").unwrap(),
            Command::DropColumn(DropColumn {
                table: "default_people".to_string(),
                column: "bucket".to_string(),
            })
        );
        assert!(matches!(
            parse_command("ALTER TABLE default_people DROP COLUMN tag CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER TABLE default_people DROP COLUMN tag, bucket"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("CREATE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 1 ORDER BY id LIMIT 5").unwrap(),
            Command::CreateView(CreateView {
                name: "active_people".to_string(),
                query: Select {
                    table: "people".to_string(),
                    distinct: false,
                    projection: SelectProjection::Columns(vec![
                        "id".to_string(),
                        "name".to_string(),
                    ]),
                    group_by: None,
                    having_groups: Vec::new(),
                    filter: Some(SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gt,
                        value: SqlValue::Int4(1),
                    }),
                    filters: vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gt,
                        value: SqlValue::Int4(1),
                    }],
                    filter_groups: vec![vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gt,
                        value: SqlValue::Int4(1),
                    }]],
                    order_by: vec![SelectOrder {
                        column: "id".to_string(),
                        descending: false,
                    }],
                    limit: Some(5),
                    offset: None,
                },
                definition: "SELECT id, name FROM people WHERE id > 1 ORDER BY id LIMIT 5"
                    .to_string(),
                or_replace: false,
            })
        );
        assert_eq!(
            parse_command("CREATE OR REPLACE VIEW active_people AS SELECT * FROM people").unwrap(),
            Command::CreateView(CreateView {
                name: "active_people".to_string(),
                query: Select {
                    table: "people".to_string(),
                    distinct: false,
                    projection: SelectProjection::All,
                    group_by: None,
                    having_groups: Vec::new(),
                    filter: None,
                    filters: Vec::new(),
                    filter_groups: Vec::new(),
                    order_by: Vec::new(),
                    limit: None,
                    offset: None,
                },
                definition: "SELECT * FROM people".to_string(),
                or_replace: true,
            })
        );
        assert_eq!(
            parse_command("DROP VIEW public.active_people").unwrap(),
            Command::DropView(DropView {
                names: vec!["active_people".to_string()],
                if_exists: false,
            })
        );
        assert_eq!(
            parse_command("DROP VIEW IF EXISTS active_people").unwrap(),
            Command::DropView(DropView {
                names: vec!["active_people".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("CREATE SCHEMA IF NOT EXISTS public").unwrap(),
            Command::CreateSchema(CreateSchema {
                name: "public".to_string(),
                if_not_exists: true,
            })
        );
        assert_eq!(
            parse_command("DROP SCHEMA IF EXISTS public").unwrap(),
            Command::DropSchema(DropSchema {
                name: "public".to_string(),
                if_exists: true,
            })
        );
        assert!(parse_command("DROP SCHEMA public CASCADE").is_err());
        assert_eq!(
            parse_command("DROP VIEW public.a, public.b").unwrap(),
            Command::DropView(DropView {
                names: vec!["a".to_string(), "b".to_string()],
                if_exists: false,
            })
        );
        assert_eq!(
            parse_command("ALTER VIEW public.active_people RENAME TO renamed_people").unwrap(),
            Command::RenameView(RenameView {
                old_name: "active_people".to_string(),
                new_name: "renamed_people".to_string(),
            })
        );
        assert!(matches!(
            parse_command("DROP VIEW active_people CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER VIEW active_people RENAME TO renamed_people CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("ALTER MATERIALIZED VIEW active_people RENAME TO renamed_people")
                .unwrap(),
            Command::RenameMaterializedView(RenameMaterializedView {
                old_name: "active_people".to_string(),
                new_name: "renamed_people".to_string(),
            })
        );
        assert!(matches!(
            parse_command("ALTER VIEW public.active_people RENAME TO public.renamed_people"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("DROP MATERIALIZED VIEW active_people").unwrap(),
            Command::DropMaterializedView(DropMaterializedView {
                names: vec!["active_people".to_string()],
                if_exists: false,
            })
        );

        assert_eq!(
            parse_command("DROP TABLE public.people").unwrap(),
            Command::DropTable(DropTable {
                names: vec!["people".to_string()],
                if_exists: false,
            })
        );
        assert_eq!(
            parse_command("DROP TABLE IF EXISTS people").unwrap(),
            Command::DropTable(DropTable {
                names: vec!["people".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("DROP TABLE public.a, public.b").unwrap(),
            Command::DropTable(DropTable {
                names: vec!["a".to_string(), "b".to_string()],
                if_exists: false,
            })
        );
        assert!(matches!(
            parse_command("DROP TABLE people CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP TABLE private.people"),
            Err(ParseError::InvalidRelationalSql)
        ));

        assert_eq!(
            parse_command("TRUNCATE TABLE ONLY public.people").unwrap(),
            Command::TruncateTable(TruncateTable {
                name: "people".to_string(),
                restart_identity: false,
            })
        );
        assert_eq!(
            parse_command("TRUNCATE people").unwrap(),
            Command::TruncateTable(TruncateTable {
                name: "people".to_string(),
                restart_identity: false,
            })
        );
        assert_eq!(
            parse_command("TRUNCATE TABLE people RESTART IDENTITY").unwrap(),
            Command::TruncateTable(TruncateTable {
                name: "people".to_string(),
                restart_identity: true,
            })
        );
        assert!(matches!(
            parse_command("TRUNCATE TABLE people CONTINUE IDENTITY"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("TRUNCATE TABLE people CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("TRUNCATE TABLE public.a, public.b"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("TRUNCATE TABLE private.people"),
            Err(ParseError::InvalidRelationalSql)
        ));

        assert_eq!(
            parse_command("CREATE PUBLICATION app_pub FOR TABLE public.people, accounts").unwrap(),
            Command::CreatePublication(CreatePublication {
                name: "app_pub".to_string(),
                target: PublicationTarget::Tables(vec![
                    "people".to_string(),
                    "accounts".to_string(),
                ]),
            })
        );
        assert_eq!(
            parse_command("CREATE PUBLICATION all_pub FOR ALL TABLES").unwrap(),
            Command::CreatePublication(CreatePublication {
                name: "all_pub".to_string(),
                target: PublicationTarget::AllTables,
            })
        );
        assert_eq!(
            parse_command("DROP PUBLICATION IF EXISTS app_pub, all_pub").unwrap(),
            Command::DropPublication(DropPublication {
                names: vec!["app_pub".to_string(), "all_pub".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("COMMENT ON PUBLICATION app_pub IS 'app publication'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Publication {
                    publication: "app_pub".to_string(),
                },
                comment: Some("app publication".to_string()),
            })
        );
        assert!(matches!(
            parse_command("CREATE PUBLICATION app_pub FOR TABLE private.people"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE PUBLICATION app_pub FOR TABLE people WITH (publish = 'insert')"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP PUBLICATION app_pub CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command(
                "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost dbname=postgres' PUBLICATION app_pub, all_pub WITH (connect = false, enabled = false)"
            )
            .unwrap(),
            Command::CreateSubscription(CreateSubscription {
                name: "app_sub".to_string(),
                connection: "host=localhost dbname=postgres".to_string(),
                publications: vec!["app_pub".to_string(), "all_pub".to_string()],
            })
        );
        assert_eq!(
            parse_command("DROP SUBSCRIPTION IF EXISTS app_sub, stale_sub").unwrap(),
            Command::DropSubscription(DropSubscription {
                names: vec!["app_sub".to_string(), "stale_sub".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("COMMENT ON SUBSCRIPTION app_sub IS NULL").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Subscription {
                    subscription: "app_sub".to_string(),
                },
                comment: None,
            })
        );
        assert!(matches!(
            parse_command(
                "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION app_pub"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command(
                "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION app_pub WITH (connect = true, enabled = false)"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command(
                "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION app_pub WITH (connect = false, enabled = true)"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP SUBSCRIPTION app_sub CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));

        assert_eq!(
            parse_command("GRANT SELECT, INSERT ON TABLE public.people TO PUBLIC").unwrap(),
            Command::GrantTable(GrantTable {
                relation: "people".to_string(),
                kind: AclRelationKind::Table,
                grantee: "public".to_string(),
                privileges: vec![TablePrivilege::Select, TablePrivilege::Insert],
            })
        );
        assert_eq!(
            parse_command("GRANT ALL PRIVILEGES ON people TO postgres").unwrap(),
            Command::GrantTable(GrantTable {
                relation: "people".to_string(),
                kind: AclRelationKind::Relation,
                grantee: "postgres".to_string(),
                privileges: vec![
                    TablePrivilege::Select,
                    TablePrivilege::Insert,
                    TablePrivilege::Update,
                    TablePrivilege::Delete,
                ],
            })
        );
        assert_eq!(
            parse_command("REVOKE UPDATE, DELETE ON TABLE people FROM PUBLIC").unwrap(),
            Command::RevokeTable(RevokeTable {
                relation: "people".to_string(),
                kind: AclRelationKind::Table,
                grantee: "public".to_string(),
                privileges: vec![TablePrivilege::Update, TablePrivilege::Delete],
            })
        );
        assert_eq!(
            parse_command("GRANT SELECT ON VIEW public.people_view TO PUBLIC").unwrap(),
            Command::GrantTable(GrantTable {
                relation: "people_view".to_string(),
                kind: AclRelationKind::View,
                grantee: "public".to_string(),
                privileges: vec![TablePrivilege::Select],
            })
        );
        assert_eq!(
            parse_command("REVOKE SELECT ON MATERIALIZED VIEW people_mv FROM PUBLIC").unwrap(),
            Command::RevokeTable(RevokeTable {
                relation: "people_mv".to_string(),
                kind: AclRelationKind::MaterializedView,
                grantee: "public".to_string(),
                privileges: vec![TablePrivilege::Select],
            })
        );
        assert_eq!(
            parse_command("GRANT SELECT, UPDATE ON SEQUENCE people_id_seq TO postgres").unwrap(),
            Command::GrantTable(GrantTable {
                relation: "people_id_seq".to_string(),
                kind: AclRelationKind::Sequence,
                grantee: "postgres".to_string(),
                privileges: vec![TablePrivilege::Select, TablePrivilege::Update],
            })
        );
        assert!(matches!(
            parse_command("GRANT SELECT (id) ON people TO PUBLIC"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("GRANT SELECT ON TABLE private.people TO PUBLIC"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("GRANT SELECT ON people TO app_reader").unwrap(),
            Command::GrantTable(GrantTable {
                relation: "people".to_string(),
                kind: AclRelationKind::Relation,
                grantee: "app_reader".to_string(),
                privileges: vec![TablePrivilege::Select],
            })
        );
        assert_eq!(
            parse_command("CREATE ROLE app_reader WITH LOGIN").unwrap(),
            Command::CreateRole(CreateRole {
                name: "app_reader".to_string(),
                login: true,
            })
        );
        assert_eq!(
            parse_command("CREATE USER app_writer").unwrap(),
            Command::CreateRole(CreateRole {
                name: "app_writer".to_string(),
                login: true,
            })
        );
        assert_eq!(
            parse_command("CREATE ROLE app_batch NOLOGIN").unwrap(),
            Command::CreateRole(CreateRole {
                name: "app_batch".to_string(),
                login: false,
            })
        );
        assert_eq!(
            parse_command("DROP ROLE IF EXISTS app_reader, app_writer").unwrap(),
            Command::DropRole(DropRole {
                names: vec!["app_reader".to_string(), "app_writer".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("ALTER ROLE app_reader RENAME TO app_analyst").unwrap(),
            Command::RenameRole(RenameRole {
                old_name: "app_reader".to_string(),
                new_name: "app_analyst".to_string(),
            })
        );
        assert_eq!(
            parse_command("DROP USER app_batch").unwrap(),
            Command::DropRole(DropRole {
                names: vec!["app_batch".to_string()],
                if_exists: false,
            })
        );
        assert_eq!(
            parse_command("CREATE DATABASE appdb").unwrap(),
            Command::CreateDatabase(CreateDatabase {
                name: "appdb".to_string(),
            })
        );
        assert_eq!(
            parse_command("CREATE DATABASE \"App DB\"").unwrap(),
            Command::CreateDatabase(CreateDatabase {
                name: "App DB".to_string(),
            })
        );
        assert_eq!(
            parse_command("DROP DATABASE IF EXISTS appdb, stale_db").unwrap(),
            Command::DropDatabase(DropDatabase {
                names: vec!["appdb".to_string(), "stale_db".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("ALTER DATABASE appdb RENAME TO appdb_archive").unwrap(),
            Command::RenameDatabase(RenameDatabase {
                old_name: "appdb".to_string(),
                new_name: "appdb_archive".to_string(),
            })
        );
        assert_eq!(
            parse_command("CREATE TABLESPACE appspace LOCATION '/tmp/gpu-db-appspace'").unwrap(),
            Command::CreateTablespace(CreateTablespace {
                name: "appspace".to_string(),
                location: "/tmp/gpu-db-appspace".to_string(),
            })
        );
        assert_eq!(
            parse_command("CREATE TABLESPACE \"App Space\" LOCATION '/tmp/app space'").unwrap(),
            Command::CreateTablespace(CreateTablespace {
                name: "App Space".to_string(),
                location: "/tmp/app space".to_string(),
            })
        );
        assert_eq!(
            parse_command("CREATE TABLESPACE appspace OWNER postgres LOCATION '/tmp/appspace'")
                .unwrap(),
            Command::CreateTablespace(CreateTablespace {
                name: "appspace".to_string(),
                location: "/tmp/appspace".to_string(),
            })
        );
        assert_eq!(
            parse_command("DROP TABLESPACE IF EXISTS appspace, stale_space").unwrap(),
            Command::DropTablespace(DropTablespace {
                names: vec!["appspace".to_string(), "stale_space".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("ALTER TABLESPACE appspace RENAME TO appspace_fast").unwrap(),
            Command::RenameTablespace(RenameTablespace {
                old_name: "appspace".to_string(),
                new_name: "appspace_fast".to_string(),
            })
        );
        assert!(matches!(
            parse_command("CREATE DATABASE appdb OWNER postgres"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP DATABASE appdb WITH (FORCE)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER DATABASE appdb OWNER TO postgres"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER DATABASE appdb RENAME TO appdb_archive SET TABLESPACE pg_default"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE TABLESPACE appspace OWNER app_owner LOCATION '/tmp/appspace'"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE TABLESPACE appspace"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP TABLESPACE appspace CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER TABLESPACE appspace OWNER TO postgres"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER TABLESPACE appspace RENAME TO public.appspace"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE ROLE app_reader PASSWORD 'secret'"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE ROLE app_reader LOGIN NOLOGIN"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER ROLE app_reader RENAME TO app_analyst WITH LOGIN"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP ROLE app_reader CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("GRANT SELECT ON people TO PUBLIC WITH GRANT OPTION"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("GRANT USAGE, CREATE ON SCHEMA public TO PUBLIC").unwrap(),
            Command::GrantSchema(SchemaPrivileges {
                schema: "public".to_string(),
                grantee: "public".to_string(),
                privileges: vec![SchemaPrivilege::Usage, SchemaPrivilege::Create],
            })
        );
        assert_eq!(
            parse_command("REVOKE CREATE ON SCHEMA public FROM postgres").unwrap(),
            Command::RevokeSchema(SchemaPrivileges {
                schema: "public".to_string(),
                grantee: "postgres".to_string(),
                privileges: vec![SchemaPrivilege::Create],
            })
        );
        assert!(matches!(
            parse_command("GRANT USAGE ON SCHEMA private TO PUBLIC"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("GRANT USAGE ON SCHEMA public TO app_reader").unwrap(),
            Command::GrantSchema(SchemaPrivileges {
                schema: "public".to_string(),
                grantee: "app_reader".to_string(),
                privileges: vec![SchemaPrivilege::Usage],
            })
        );
        assert!(matches!(
            parse_command("GRANT USAGE ON SCHEMA public TO PUBLIC WITH GRANT OPTION"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command(
                "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT, INSERT ON TABLES TO PUBLIC"
            )
            .unwrap(),
            Command::GrantDefaultTablePrivileges(DefaultTablePrivileges {
                grantee: "public".to_string(),
                privileges: vec![TablePrivilege::Select, TablePrivilege::Insert],
            })
        );
        assert_eq!(
            parse_command("ALTER DEFAULT PRIVILEGES REVOKE INSERT ON TABLES FROM PUBLIC").unwrap(),
            Command::RevokeDefaultTablePrivileges(DefaultTablePrivileges {
                grantee: "public".to_string(),
                privileges: vec![TablePrivilege::Insert],
            })
        );
        assert!(matches!(
            parse_command(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA private GRANT SELECT ON TABLES TO PUBLIC"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER DEFAULT PRIVILEGES GRANT SELECT ON SEQUENCES TO PUBLIC"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO app_reader").unwrap(),
            Command::GrantDefaultTablePrivileges(DefaultTablePrivileges {
                grantee: "app_reader".to_string(),
                privileges: vec![TablePrivilege::Select],
            })
        );
        assert_eq!(
            parse_command("GRANT CONNECT, TEMPORARY ON DATABASE appdb TO app_reader").unwrap(),
            Command::GrantDatabase(DatabasePrivileges {
                database: "appdb".to_string(),
                grantee: "app_reader".to_string(),
                privileges: vec![DatabasePrivilege::Connect, DatabasePrivilege::Temporary],
            })
        );
        assert_eq!(
            parse_command("REVOKE TEMP ON DATABASE appdb FROM PUBLIC").unwrap(),
            Command::RevokeDatabase(DatabasePrivileges {
                database: "appdb".to_string(),
                grantee: "public".to_string(),
                privileges: vec![DatabasePrivilege::Temporary],
            })
        );
        assert_eq!(
            parse_command("GRANT ALL PRIVILEGES ON TABLESPACE appspace TO postgres").unwrap(),
            Command::GrantTablespace(TablespacePrivileges {
                tablespace: "appspace".to_string(),
                grantee: "postgres".to_string(),
                privileges: vec![TablespacePrivilege::Create],
            })
        );
        assert_eq!(
            parse_command("REVOKE CREATE ON TABLESPACE appspace FROM app_reader").unwrap(),
            Command::RevokeTablespace(TablespacePrivileges {
                tablespace: "appspace".to_string(),
                grantee: "app_reader".to_string(),
                privileges: vec![TablespacePrivilege::Create],
            })
        );
        assert_eq!(
            parse_command("GRANT EXECUTE ON FUNCTION public.answer() TO app_reader").unwrap(),
            Command::GrantFunction(FunctionPrivileges {
                function: "answer".to_string(),
                grantee: "app_reader".to_string(),
                privileges: vec![FunctionPrivilege::Execute],
            })
        );
        assert_eq!(
            parse_command("REVOKE ALL PRIVILEGES ON FUNCTION answer() FROM PUBLIC").unwrap(),
            Command::RevokeFunction(FunctionPrivileges {
                function: "answer".to_string(),
                grantee: "public".to_string(),
                privileges: vec![FunctionPrivilege::Execute],
            })
        );
        assert!(matches!(
            parse_command("GRANT SELECT ON FUNCTION answer() TO PUBLIC"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("GRANT EXECUTE ON FUNCTION answer(int4) TO PUBLIC"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("GRANT SELECT ON DATABASE appdb TO PUBLIC"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("GRANT CREATE ON TABLESPACE appspace TO PUBLIC WITH GRANT OPTION"),
            Err(ParseError::InvalidRelationalSql)
        ));

        assert_eq!(
            parse_command("CREATE INDEX people_name_idx ON public.people (name)").unwrap(),
            Command::CreateIndex(CreateIndex {
                name: "people_name_idx".to_string(),
                table: "people".to_string(),
                column: "name".to_string(),
                columns: vec!["name".to_string()],
                unique: false,
            })
        );
        assert_eq!(
            parse_command("CREATE INDEX people_name_idx ON public.people USING btree (name)")
                .unwrap(),
            Command::CreateIndex(CreateIndex {
                name: "people_name_idx".to_string(),
                table: "people".to_string(),
                column: "name".to_string(),
                columns: vec!["name".to_string()],
                unique: false,
            })
        );
        assert_eq!(
            parse_command(
                "CREATE UNIQUE INDEX people_name_idx ON public.people USING btree (name)"
            )
            .unwrap(),
            Command::CreateIndex(CreateIndex {
                name: "people_name_idx".to_string(),
                table: "people".to_string(),
                column: "name".to_string(),
                columns: vec!["name".to_string()],
                unique: true,
            })
        );
        assert!(matches!(
            parse_command("CREATE INDEX people_name_idx ON people USING hash (name)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE INDEX people_multi_idx ON people (id, name)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("DROP INDEX public.people_name_idx").unwrap(),
            Command::DropIndex(DropIndex {
                names: vec!["people_name_idx".to_string()],
                if_exists: false,
            })
        );
        assert_eq!(
            parse_command("DROP INDEX IF EXISTS people_name_idx").unwrap(),
            Command::DropIndex(DropIndex {
                names: vec!["people_name_idx".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("DROP INDEX public.a, public.b").unwrap(),
            Command::DropIndex(DropIndex {
                names: vec!["a".to_string(), "b".to_string()],
                if_exists: false,
            })
        );
        assert!(matches!(
            parse_command("DROP INDEX CONCURRENTLY people_name_idx"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP INDEX people_name_idx CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("ALTER INDEX public.people_name_idx RENAME TO people_lookup_idx")
                .unwrap(),
            Command::RenameIndex(RenameIndex {
                old_name: "people_name_idx".to_string(),
                new_name: "people_lookup_idx".to_string(),
            })
        );
        assert!(matches!(
            parse_command("ALTER INDEX IF EXISTS people_name_idx RENAME TO people_lookup_idx"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER INDEX people_name_idx RENAME TO public.people_lookup_idx"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER INDEX people_name_idx RENAME TO people_lookup_idx CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));

        assert_eq!(
            parse_command("INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')").unwrap(),
            Command::Insert(Insert {
                table: "people".to_string(),
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![
                    vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                    vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                ],
            })
        );

        assert_eq!(
            parse_command("INSERT INTO people (id, name) VALUES (1, 'O''Brien')").unwrap(),
            Command::Insert(Insert {
                table: "people".to_string(),
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![vec![
                    SqlValue::Int4(1),
                    SqlValue::Text("O'Brien".to_string())
                ]],
            })
        );

        assert_eq!(
            parse_command("INSERT INTO people (id, name) VALUES (-1, 'Minus'), (0, 'Zero')")
                .unwrap(),
            Command::Insert(Insert {
                table: "people".to_string(),
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![
                    vec![SqlValue::Int4(-1), SqlValue::Text("Minus".to_string())],
                    vec![SqlValue::Int4(0), SqlValue::Text("Zero".to_string())],
                ],
            })
        );

        assert_eq!(
            parse_command("INSERT INTO public.people (id, name) VALUES (3, 'Grace')").unwrap(),
            Command::Insert(Insert {
                table: "people".to_string(),
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())]],
            })
        );

        assert_eq!(
            parse_command("INSERT INTO public.people VALUES (4, 'Katherine'), (5, 'Mary')")
                .unwrap(),
            Command::Insert(Insert {
                table: "people".to_string(),
                columns: Vec::new(),
                rows: vec![
                    vec![SqlValue::Int4(4), SqlValue::Text("Katherine".to_string())],
                    vec![SqlValue::Int4(5), SqlValue::Text("Mary".to_string())],
                ],
            })
        );

        assert_eq!(
            parse_command("DELETE FROM public.people WHERE id = 1 OR name LIKE 'Ada%'").unwrap(),
            Command::Delete(Delete {
                table: "people".to_string(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    }],
                    vec![SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::LikePrefix,
                        value: SqlValue::Text("Ada".to_string()),
                    }],
                ],
            })
        );
        assert_eq!(
            parse_command("UPDATE public.people SET name = 'Updated', id = 10 WHERE id = 1 OR name LIKE 'Ada%'").unwrap(),
            Command::Update(Update {
                table: "people".to_string(),
                assignments: vec![
                    UpdateAssignment {
                        column: "name".to_string(),
                        value: SqlValue::Text("Updated".to_string()),
                    },
                    UpdateAssignment {
                        column: "id".to_string(),
                        value: SqlValue::Int4(10),
                    },
                ],
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    }],
                    vec![SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::LikePrefix,
                        value: SqlValue::Text("Ada".to_string()),
                    }],
                ],
            })
        );
        assert!(matches!(
            parse_command("UPDATE public.people SET name = 'Updated'"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("UPDATE public.people SET name = 'Updated' WHERE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert_eq!(
            parse_command("DELETE FROM balance").unwrap(),
            Command::DeleteKv {
                key: "balance".to_string(),
            }
        );
        assert_eq!(
            parse_command("DELETE FROM public.people").unwrap(),
            Command::DeleteKv {
                key: "public.people".to_string(),
            }
        );

        assert_eq!(
            parse_command("SELECT id FROM people WHERE id = +1 LIMIT +1").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }]],
                order_by: Vec::new(),
                limit: Some(1),
                offset: None,
            })
        );
        assert_eq!(
            parse_command("SELECT id FROM public.people WHERE id = 1").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }]],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );
        assert_eq!(
            parse_command("SELECT id, name FROM ONLY public.people").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string(), "name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );
        assert!(matches!(
            parse_command("SELECT id FROM people ORDER BY id LIMIT -1"),
            Err(ParseError::NegativeLimit)
        ));
        assert!(matches!(
            parse_command("SELECT id FROM people ORDER BY id OFFSET -1"),
            Err(ParseError::NegativeOffset)
        ));

        assert_eq!(
            parse_command("SELECT id FROM people ORDER BY id LIMIT 2 OFFSET 1").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: Some(2),
                offset: Some(1),
            })
        );
        assert_eq!(
            parse_command("SELECT id FROM people ORDER BY id OFFSET 1 LIMIT 2").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: Some(2),
                offset: Some(1),
            })
        );

        assert_eq!(
            parse_command("SELECT id, name FROM people WHERE id = 1 ORDER BY name DESC LIMIT 5")
                .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string(), "name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }]],
                order_by: vec![SelectOrder {
                    column: "name".to_string(),
                    descending: true,
                }],
                limit: Some(5),
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE id >= 2 ORDER BY id LIMIT 5").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }]],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: Some(5),
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE 2 <= id ORDER BY id LIMIT 5").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }]],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: Some(5),
                offset: None,
            })
        );

        assert_eq!(
            parse_command(
                "SELECT name FROM people WHERE id >= 2 ORDER BY id LIMIT 5::pg_catalog.int4"
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }]],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: Some(5),
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE name = 'O''Brien'").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("O'Brien".to_string()),
                }),
                filters: vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("O'Brien".to_string()),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("O'Brien".to_string()),
                }]],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command(
                "SELECT id, name FROM people WHERE id = 2::int4 OR name = 'Ada'::text ORDER BY id"
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string(), "name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(2),
                    }],
                    vec![SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    }],
                ],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE id >= 2 AND name = 'Ada'").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                ],
                filter_groups: vec![vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                ]],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE id = 1 OR name = 'Ada'").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    }],
                    vec![SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    }],
                ],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE (id = 1) OR (name = 'Ada')").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    }],
                    vec![SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    }],
                ],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command(
                "SELECT id FROM people WHERE (id = 1 OR id = 3) AND (name = 'Ada' OR name = 'Grace') ORDER BY id"
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                ],
                filter_groups: vec![
                    vec![
                        SelectFilter {
                            column: "id".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Int4(1),
                        },
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Text("Ada".to_string()),
                        },
                    ],
                    vec![
                        SelectFilter {
                            column: "id".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Int4(1),
                        },
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Text("Grace".to_string()),
                        },
                    ],
                    vec![
                        SelectFilter {
                            column: "id".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Int4(3),
                        },
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Text("Ada".to_string()),
                        },
                    ],
                    vec![
                        SelectFilter {
                            column: "id".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Int4(3),
                        },
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Text("Grace".to_string()),
                        },
                    ],
                ],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: None,
                offset: None,
            })
        );
    }

    #[test]
    fn parses_relational_select_in_membership_predicates_as_filter_groups() {
        assert_eq!(
            parse_command("SELECT id FROM people WHERE id IN (1, 3, 5) ORDER BY id").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    }],
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(3),
                    }],
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(5),
                    }],
                ],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE name IN ('Ada', 'Grace') AND id >= 2")
                .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                }),
                filters: vec![
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    },
                ],
                filter_groups: vec![
                    vec![
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Text("Ada".to_string()),
                        },
                        SelectFilter {
                            column: "id".to_string(),
                            op: SelectFilterOp::Gte,
                            value: SqlValue::Int4(2),
                        },
                    ],
                    vec![
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Eq,
                            value: SqlValue::Text("Grace".to_string()),
                        },
                        SelectFilter {
                            column: "id".to_string(),
                            op: SelectFilterOp::Gte,
                            value: SqlValue::Int4(2),
                        },
                    ],
                ],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT id FROM people WHERE id IN ()"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT id FROM people WHERE id NOT IN (1, 2)"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_relational_select_between_predicates_as_filter_groups() {
        assert_eq!(
            parse_command("SELECT id FROM people WHERE id BETWEEN 2 AND 4 ORDER BY id").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    },
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Lte,
                        value: SqlValue::Int4(4),
                    },
                ],
                filter_groups: vec![vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    },
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Lte,
                        value: SqlValue::Int4(4),
                    },
                ]],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE name BETWEEN 'Ada' AND 'Grace' OR id = 4")
                .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Text("Ada".to_string()),
                }),
                filters: vec![
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Lte,
                        value: SqlValue::Text("Grace".to_string()),
                    },
                ],
                filter_groups: vec![
                    vec![
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Gte,
                            value: SqlValue::Text("Ada".to_string()),
                        },
                        SelectFilter {
                            column: "name".to_string(),
                            op: SelectFilterOp::Lte,
                            value: SqlValue::Text("Grace".to_string()),
                        },
                    ],
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(4),
                    }],
                ],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT id FROM people WHERE id NOT BETWEEN 1 AND 3"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_relational_select_prefix_like_predicates_as_filters() {
        assert_eq!(
            parse_command("SELECT id FROM people WHERE name LIKE 'Gra%' ORDER BY id").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("Gra".to_string()),
                }),
                filters: vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("Gra".to_string()),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("Gra".to_string()),
                }]],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: None,
                offset: None,
            })
        );

        assert_eq!(
            parse_command("SELECT name FROM people WHERE name LIKE 'A%' OR id = 3").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["name".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("A".to_string()),
                }),
                filters: vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("A".to_string()),
                }],
                filter_groups: vec![
                    vec![SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::LikePrefix,
                        value: SqlValue::Text("A".to_string()),
                    }],
                    vec![SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(3),
                    }],
                ],
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT id FROM people WHERE name NOT LIKE 'A%'"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT id FROM people WHERE name LIKE '%da'"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT id FROM people WHERE name LIKE 'A_a%'"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_relational_select_distinct_projection() {
        assert_eq!(
            parse_command(
                "SELECT DISTINCT name, id FROM people WHERE name LIKE 'G%' ORDER BY name DESC LIMIT 2 OFFSET 1",
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: true,
                projection: SelectProjection::Columns(vec!["name".to_string(), "id".to_string()]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("G".to_string()),
                }),
                filters: vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("G".to_string()),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("G".to_string()),
                }]],
                order_by: vec![SelectOrder {
                    column: "name".to_string(),
                    descending: true,
                }],
                limit: Some(2),
                offset: Some(1),
            })
        );

        assert!(matches!(
            parse_command("SELECT DISTINCT * FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_relational_count_aggregates() {
        assert_eq!(
            parse_command(
                "SELECT name, COUNT(*) FROM people WHERE id >= 2 GROUP BY name HAVING count >= 2 OR name = 'Ada' ORDER BY count DESC LIMIT 2 OFFSET 1",
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::GroupedCount {
                    column: "name".to_string(),
                },
                group_by: Some("name".to_string()),
                having_groups: vec![
                    vec![SelectFilter {
                        column: "count".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    }],
                    vec![SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    }],
                ],
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }]],
                order_by: vec![SelectOrder {
                    column: "count".to_string(),
                    descending: true,
                }],
                limit: Some(2),
                offset: Some(1),
            })
        );

        assert_eq!(
            parse_command("SELECT COUNT(*) FROM people").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::CountAll,
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT COUNT(*), id FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_relational_having_aggregates() {
        assert_eq!(
            parse_command(
                "SELECT name, SUM(id) FROM people GROUP BY name HAVING sum > 2 AND name LIKE 'G%' ORDER BY sum DESC",
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::GroupedSum {
                    group_column: "name".to_string(),
                    sum_column: "id".to_string(),
                },
                group_by: Some("name".to_string()),
                having_groups: vec![vec![
                    SelectFilter {
                        column: "sum".to_string(),
                        op: SelectFilterOp::Gt,
                        value: SqlValue::Int4(2),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::LikePrefix,
                        value: SqlValue::Text("G".to_string()),
                    },
                ]],
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: vec![SelectOrder {
                    column: "sum".to_string(),
                    descending: true,
                }],
                limit: None,
                offset: None,
            })
        );
    }

    #[test]
    fn parses_relational_sum_aggregates() {
        assert_eq!(
            parse_command(
                "SELECT name, SUM(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY sum DESC LIMIT 2 OFFSET 1",
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::GroupedSum {
                    group_column: "name".to_string(),
                    sum_column: "id".to_string(),
                },
                group_by: Some("name".to_string()),
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }]],
                order_by: vec![SelectOrder {
                    column: "sum".to_string(),
                    descending: true,
                }],
                limit: Some(2),
                offset: Some(1),
            })
        );

        assert_eq!(
            parse_command("SELECT SUM(id) FROM people").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Sum {
                    column: "id".to_string(),
                },
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT SUM(id), name FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_relational_avg_aggregates() {
        assert_eq!(
            parse_command(
                "SELECT name, AVG(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY avg DESC LIMIT 2 OFFSET 1",
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::GroupedAvg {
                    group_column: "name".to_string(),
                    avg_column: "id".to_string(),
                },
                group_by: Some("name".to_string()),
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }]],
                order_by: vec![SelectOrder {
                    column: "avg".to_string(),
                    descending: true,
                }],
                limit: Some(2),
                offset: Some(1),
            })
        );

        assert_eq!(
            parse_command("SELECT AVG(id) FROM people").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Avg {
                    column: "id".to_string(),
                },
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT AVG(id), name FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT DISTINCT AVG(id) FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_relational_min_max_aggregates() {
        assert_eq!(
            parse_command(
                "SELECT name, MIN(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY min DESC LIMIT 2 OFFSET 1",
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::GroupedMin {
                    group_column: "name".to_string(),
                    min_column: "id".to_string(),
                },
                group_by: Some("name".to_string()),
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }]],
                order_by: vec![SelectOrder {
                    column: "min".to_string(),
                    descending: true,
                }],
                limit: Some(2),
                offset: Some(1),
            })
        );

        assert_eq!(
            parse_command("SELECT MAX(name) FROM people").unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Max {
                    column: "name".to_string(),
                },
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT MIN(id), name FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_sequence_catalog_ddl() {
        assert_eq!(
            parse_command("CREATE SEQUENCE public.seq_people").unwrap(),
            Command::CreateSequence(CreateSequence {
                name: "seq_people".to_string(),
            })
        );
        assert_eq!(
            parse_command(
                "CREATE SEQUENCE public.seq_people START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1"
            )
            .unwrap(),
            Command::CreateSequence(CreateSequence {
                name: "seq_people".to_string(),
            })
        );
        assert_eq!(
            parse_command("DROP SEQUENCE IF EXISTS public.seq_people, seq_teams").unwrap(),
            Command::DropSequence(DropSequence {
                names: vec!["seq_people".to_string(), "seq_teams".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("ALTER SEQUENCE public.seq_people RENAME TO seq_person_ids").unwrap(),
            Command::RenameSequence(RenameSequence {
                old_name: "seq_people".to_string(),
                new_name: "seq_person_ids".to_string(),
            })
        );
        assert_eq!(
            parse_command("COMMENT ON SEQUENCE public.seq_people IS 'ids'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Sequence {
                    sequence: "seq_people".to_string(),
                },
                comment: Some("ids".to_string()),
            })
        );

        assert!(matches!(
            parse_command("CREATE SEQUENCE seq_people START WITH 10"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE SEQUENCE seq_people AS bigint"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP SEQUENCE seq_people CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER SEQUENCE IF EXISTS seq_people RENAME TO seq_person_ids"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER SEQUENCE seq_people RENAME TO public.seq_person_ids"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER SEQUENCE seq_people RENAME TO seq_person_ids CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_domain_catalog_ddl() {
        assert_eq!(
            parse_command("CREATE DOMAIN public.account_id AS int4").unwrap(),
            Command::CreateDomain(CreateDomain {
                name: "account_id".to_string(),
                base_type: SqlType::Int4,
            })
        );
        assert_eq!(
            parse_command("CREATE DOMAIN label AS pg_catalog.text").unwrap(),
            Command::CreateDomain(CreateDomain {
                name: "label".to_string(),
                base_type: SqlType::Text,
            })
        );
        assert_eq!(
            parse_command("COMMENT ON DOMAIN public.account_id IS 'domain ids'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Domain {
                    domain: "account_id".to_string(),
                },
                comment: Some("domain ids".to_string()),
            })
        );
        assert_eq!(
            parse_command("DROP DOMAIN IF EXISTS public.account_id, label").unwrap(),
            Command::DropDomain(DropDomain {
                domains: vec!["account_id".to_string(), "label".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("CREATE TABLE accounts (id account_id, label public.label)").unwrap(),
            Command::CreateTable(CreateTable {
                table: "accounts".to_string(),
                columns: vec![
                    ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: Some("account_id".to_string()),
                        default: None,
                    },
                    ColumnDef {
                        name: "label".to_string(),
                        ty: SqlType::Int4,
                        domain: Some("label".to_string()),
                        default: None,
                    },
                ],
                primary_key: None,
                unique_constraints: Vec::new(),
                check_constraints: Vec::new(),
            })
        );

        assert!(matches!(
            parse_command("CREATE DOMAIN account_id AS int4 CHECK (VALUE > 0)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE TABLE accounts (id account_id DEFAULT 1)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        // `bigint` is a storable base type as of Phase-3 M1, so a domain over it now parses
        // (previously only int4/text were supported base types). A parenthesized typmod
        // (e.g. `numeric(12,2)`) on a domain stays rejected — domain typmods are out of M1 scope.
        assert_eq!(
            parse_command("CREATE DOMAIN account_id AS bigint").unwrap(),
            Command::CreateDomain(CreateDomain {
                name: "account_id".to_string(),
                base_type: SqlType::Int8,
            })
        );
        assert!(matches!(
            parse_command("CREATE DOMAIN money_amount AS numeric(12,2)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP DOMAIN account_id CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_function_catalog_ddl() {
        assert_eq!(
            parse_command(
                "CREATE FUNCTION public.answer() RETURNS int4 LANGUAGE sql AS 'SELECT 42'"
            )
            .unwrap(),
            Command::CreateFunction(CreateFunction {
                name: "answer".to_string(),
                return_type: SqlType::Int4,
                body: "SELECT 42".to_string(),
            })
        );
        assert_eq!(
            parse_command(
                "CREATE FUNCTION public.dump_answer() RETURNS integer LANGUAGE sql AS $$SELECT 42$$"
            )
            .unwrap(),
            Command::CreateFunction(CreateFunction {
                name: "dump_answer".to_string(),
                return_type: SqlType::Int4,
                body: "SELECT 42".to_string(),
            })
        );
        assert_eq!(
            parse_command("COMMENT ON FUNCTION public.answer() IS 'metadata only'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Function {
                    function: "answer".to_string(),
                },
                comment: Some("metadata only".to_string()),
            })
        );
        assert_eq!(
            parse_command("COMMENT ON EXTENSION plpgsql IS 'bootstrap extension'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::Extension {
                    extension: "plpgsql".to_string(),
                },
                comment: Some("bootstrap extension".to_string()),
            })
        );
        assert_eq!(
            parse_command("ALTER FUNCTION public.answer() RENAME TO ultimate_answer").unwrap(),
            Command::RenameFunction(RenameFunction {
                old_name: "answer".to_string(),
                new_name: "ultimate_answer".to_string(),
            })
        );
        assert_eq!(
            parse_command("DROP FUNCTION IF EXISTS public.answer()").unwrap(),
            Command::DropFunction(DropFunction {
                name: "answer".to_string(),
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("SELECT public.answer()").unwrap(),
            Command::SelectFunction(SelectFunction {
                name: "answer".to_string(),
            })
        );
        assert_eq!(
            parse_command("SELECT answer()").unwrap(),
            Command::SelectFunction(SelectFunction {
                name: "answer".to_string(),
            })
        );

        assert!(matches!(
            parse_command(
                "CREATE FUNCTION public.echo(int4) RETURNS int4 LANGUAGE sql AS 'SELECT $1'"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        // `bigint` is a storable return type as of Phase-3 M1 (previously unsupported).
        assert_eq!(
            parse_command(
                "CREATE FUNCTION public.answer() RETURNS bigint LANGUAGE sql AS 'SELECT 42'"
            )
            .unwrap(),
            Command::CreateFunction(CreateFunction {
                name: "answer".to_string(),
                return_type: SqlType::Int8,
                body: "SELECT 42".to_string(),
            })
        );
        assert!(matches!(
            parse_command(
                "CREATE FUNCTION public.answer() RETURNS int4 LANGUAGE plpgsql AS 'BEGIN END'"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER FUNCTION public.answer(int4) RENAME TO answer2"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER FUNCTION public.answer() RENAME TO public.answer2"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER FUNCTION public.answer() OWNER TO postgres"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP FUNCTION public.answer() CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT answer(1)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT answer() FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_bootstrap_extension_create() {
        assert_eq!(
            parse_command("CREATE EXTENSION IF NOT EXISTS plpgsql").unwrap(),
            Command::CreateExtension(CreateExtension {
                name: "plpgsql".to_string(),
                if_not_exists: true,
                schema: None,
            })
        );
        assert_eq!(
            parse_command("CREATE EXTENSION IF NOT EXISTS \"plpgsql\" WITH SCHEMA pg_catalog")
                .unwrap(),
            Command::CreateExtension(CreateExtension {
                name: "plpgsql".to_string(),
                if_not_exists: true,
                schema: Some("pg_catalog".to_string()),
            })
        );
        assert_eq!(
            parse_command("CREATE EXTENSION plpgsql").unwrap(),
            Command::CreateExtension(CreateExtension {
                name: "plpgsql".to_string(),
                if_not_exists: false,
                schema: None,
            })
        );

        assert!(matches!(
            parse_command("CREATE EXTENSION IF NOT EXISTS plpgsql VERSION '1.0'"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("CREATE EXTENSION IF NOT EXISTS plpgsql WITH VERSION '1.0'"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command(
                "CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA public VERSION '1.0'"
            ),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_bootstrap_extension_drop_cleanup() {
        assert_eq!(
            parse_command("DROP EXTENSION IF EXISTS plpgsql").unwrap(),
            Command::DropExtension(DropExtension {
                name: "plpgsql".to_string(),
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("DROP EXTENSION \"plpgsql\"").unwrap(),
            Command::DropExtension(DropExtension {
                name: "plpgsql".to_string(),
                if_exists: false,
            })
        );

        assert!(matches!(
            parse_command("DROP EXTENSION IF EXISTS plpgsql CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP EXTENSION IF EXISTS plpgsql, hstore"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_sequence_value_functions() {
        assert_eq!(
            parse_command("SELECT nextval('public.seq_people'::regclass)").unwrap(),
            Command::SequenceNextVal(SequenceNextVal {
                name: "seq_people".to_string(),
            })
        );
        assert_eq!(
            parse_command("SELECT pg_catalog.currval('seq_people'::pg_catalog.regclass)").unwrap(),
            Command::SequenceCurrVal(SequenceCurrVal {
                name: "seq_people".to_string(),
            })
        );
        assert_eq!(
            parse_command("SELECT setval('public.seq_people', 42, false)").unwrap(),
            Command::SequenceSetVal(SequenceSetVal {
                name: "seq_people".to_string(),
                value: 42,
                is_called: false,
            })
        );
        assert_eq!(
            parse_command("SELECT pg_catalog.setval('public.seq_people', 42)").unwrap(),
            Command::SequenceSetVal(SequenceSetVal {
                name: "seq_people".to_string(),
                value: 42,
                is_called: true,
            })
        );

        assert!(matches!(
            parse_command("SELECT nextval('other.seq_people'::regclass)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT setval('seq_people', '42', false)"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("SELECT nextval('seq_people') FROM seq_people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_materialized_view_lifecycle() {
        assert_eq!(
            parse_command(
                "CREATE MATERIALIZED VIEW public.mv_people AS SELECT id, name FROM people ORDER BY id"
            )
            .unwrap(),
            Command::CreateMaterializedView(CreateMaterializedView {
                name: "mv_people".to_string(),
                query: Select {
                    table: "people".to_string(),
                    distinct: false,
                    projection: SelectProjection::Columns(vec![
                        "id".to_string(),
                        "name".to_string(),
                    ]),
                    group_by: None,
                    having_groups: Vec::new(),
                    filter: None,
                    filters: Vec::new(),
                    filter_groups: Vec::new(),
                    order_by: vec![SelectOrder {
                        column: "id".to_string(),
                        descending: false,
                    }],
                    limit: None,
                    offset: None,
                },
                definition: "SELECT id, name FROM people ORDER BY id".to_string(),
                with_data: true,
            })
        );
        assert_eq!(
            parse_command(
                "CREATE MATERIALIZED VIEW mv_people AS SELECT * FROM people WITH NO DATA"
            )
            .unwrap(),
            Command::CreateMaterializedView(CreateMaterializedView {
                name: "mv_people".to_string(),
                query: Select {
                    table: "people".to_string(),
                    distinct: false,
                    projection: SelectProjection::All,
                    group_by: None,
                    having_groups: Vec::new(),
                    filter: None,
                    filters: Vec::new(),
                    filter_groups: Vec::new(),
                    order_by: Vec::new(),
                    limit: None,
                    offset: None,
                },
                definition: "SELECT * FROM people".to_string(),
                with_data: false,
            })
        );
        assert_eq!(
            parse_command("ALTER MATERIALIZED VIEW public.mv_people RENAME TO mv_people_old")
                .unwrap(),
            Command::RenameMaterializedView(RenameMaterializedView {
                old_name: "mv_people".to_string(),
                new_name: "mv_people_old".to_string(),
            })
        );
        assert_eq!(
            parse_command("REFRESH MATERIALIZED VIEW public.mv_people WITH DATA").unwrap(),
            Command::RefreshMaterializedView(RefreshMaterializedView {
                name: "mv_people".to_string(),
            })
        );
        assert_eq!(
            parse_command("DROP MATERIALIZED VIEW IF EXISTS public.mv_people_old, mv_other")
                .unwrap(),
            Command::DropMaterializedView(DropMaterializedView {
                names: vec!["mv_people_old".to_string(), "mv_other".to_string()],
                if_exists: true,
            })
        );
        assert_eq!(
            parse_command("COMMENT ON MATERIALIZED VIEW public.mv_people IS 'snapshot'").unwrap(),
            Command::CommentOn(CommentOn {
                target: CommentTarget::MaterializedView {
                    materialized_view: "mv_people".to_string(),
                },
                comment: Some("snapshot".to_string()),
            })
        );

        assert!(parse_command(
            "CREATE MATERIALIZED VIEW mv_people AS SELECT * FROM people WITH DATA"
        )
        .is_ok());
        assert!(matches!(
            parse_command("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_people"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("REFRESH MATERIALIZED VIEW mv_people WITH NO DATA"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("DROP MATERIALIZED VIEW mv_people CASCADE"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_command("ALTER MATERIALIZED VIEW mv_people RENAME TO public.mv_people_old"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }
}
