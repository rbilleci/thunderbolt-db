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
    FailedTransaction,
}

impl TransactionStatus {
    pub fn ready_for_query_in_transaction(self) -> bool {
        !matches!(self, Self::Idle)
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
                parameter_format_codes.push(format_code);
            }

            let parameter_count = read_i16(payload, &mut offset)?;
            if parameter_count < 0 {
                return Err(FrontendMessageError::InvalidBindPayload);
            }
            let parameter_count = parameter_count as usize;
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
    use super::TransactionStatus;
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
            let status = if in_transaction {
                TransactionStatus::InTransaction
            } else {
                TransactionStatus::Idle
            };
            self.ready_for_query_status(status)
        }

        pub fn ready_for_query_status(&mut self, status: TransactionStatus) -> io::Result<()> {
            let status = match status {
                TransactionStatus::Idle => b'I',
                TransactionStatus::InTransaction => b'T',
                TransactionStatus::FailedTransaction => b'E',
            };
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

    #[path = "kv_commands.rs"]
    mod kv_commands;

    #[path = "startup.rs"]
    mod startup;

    #[path = "session_lifecycle.rs"]
    mod session_lifecycle;

    fn frontend_frame(tag: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(payload.len() + 5);
        frame.push(tag);
        frame.extend_from_slice(&((payload.len() + 4) as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }

    #[path = "frontend_messages_valid.rs"]
    mod frontend_messages_valid;

    #[path = "frontend_messages_malformed.rs"]
    mod frontend_messages_malformed;

    #[path = "relational_sql_facade.rs"]
    mod relational_sql_facade;

    #[path = "relational_select_features.rs"]
    mod relational_select_features;

    #[path = "relational_aggregates.rs"]
    mod relational_aggregates;

    #[path = "catalog_sql_compat.rs"]
    mod catalog_sql_compat;
}
