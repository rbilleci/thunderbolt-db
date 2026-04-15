#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Begin,
    Commit { chain: bool },
    Rollback { chain: bool },
    Flush,
    ResetAll,
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
    #[error("invalid SET syntax; expected: SET key=value or SET key TO value")]
    InvalidSet,
    #[error("invalid DEL/DELETE syntax; expected: DEL key or DELETE [FROM] key")]
    InvalidDel,
    #[error("invalid GET syntax; expected: GET key")]
    InvalidGet,
    #[error("invalid RESET/DISCARD/DEALLOCATE/CLOSE/LISTEN/NOTIFY/UNLISTEN syntax; expected: RESET ALL|ROLE|AUTHORIZATION|AUTH|SESSION AUTHORIZATION|SESSION AUTH, DISCARD {{ALL|TEMP|TEMPORARY|TEMP TABLES|TEMPORARY TABLES|PLANS|SEQUENCES}}, DEALLOCATE {{ALL|name|PREPARE|PREPARED name}}, CLOSE {{ALL|name}}, LISTEN channel, NOTIFY channel[, payload], or UNLISTEN [*|ALL|channel]")]
    InvalidReset,
}

pub const PG_PROTOCOL_V3: u32 = 196_608;
const PG_SSL_REQUEST_CODE: u32 = 80_877_103;
const PG_CANCEL_REQUEST_CODE: u32 = 80_877_102;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupPacket {
    Startup {
        protocol_version: u32,
        params: Vec<(String, String)>,
    },
    SslRequest,
    CancelRequest {
        process_id: u32,
        secret_key: u32,
    },
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum StartupPacketError {
    #[error("startup packet too short")]
    TooShort,
    #[error("startup packet length mismatch; expected {expected} bytes, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
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

fn parse_startup_params(payload: &[u8]) -> Result<Vec<(String, String)>, StartupPacketError> {
    let Some(last) = payload.last() else {
        return Ok(Vec::new());
    };
    if *last != 0 {
        return Err(StartupPacketError::UnterminatedParameterPayload);
    }

    let segments: Vec<&[u8]> = payload[..payload.len() - 1]
        .split(|b| *b == 0)
        .filter(|segment| !segment.is_empty())
        .collect();
    if !segments.len().is_multiple_of(2) {
        return Err(StartupPacketError::InvalidParameterPairing);
    }

    let mut params = Vec::with_capacity(segments.len() / 2);
    for pair in segments.chunks_exact(2) {
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
        PG_CANCEL_REQUEST_CODE => {
            if frame_len != 16 {
                return Err(StartupPacketError::LengthMismatch {
                    expected: 16,
                    actual: frame_len,
                });
            }
            let process_id = read_u32_be(&frame[8..12])?;
            let secret_key = read_u32_be(&frame[12..16])?;
            Ok(StartupPacket::CancelRequest {
                process_id,
                secret_key,
            })
        }
        protocol_version if protocol_version == PG_PROTOCOL_V3 => {
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
    #[error("describe message target must be S (statement) or P (portal)")]
    InvalidDescribeTarget,
    #[error("describe message name is not null terminated")]
    UnterminatedDescribeName,
    #[error("close message target must be S (statement) or P (portal)")]
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

pub fn parse_frontend_message(frame: &[u8]) -> Result<FrontendMessage, FrontendMessageError> {
    if frame.len() < 5 {
        return Err(FrontendMessageError::TooShort);
    }

    let tag = frame[0];
    let payload_len = u32::from_be_bytes(
        frame[1..5]
            .try_into()
            .map_err(|_| FrontendMessageError::TooShort)?,
    ) as usize;
    let expected = payload_len + 1;
    if expected != frame.len() {
        return Err(FrontendMessageError::LengthMismatch {
            expected,
            actual: frame.len(),
        });
    }

    let payload = &frame[5..];
    match tag {
        b'Q' => {
            let Some(query_bytes) = payload.strip_suffix(&[0]) else {
                return Err(FrontendMessageError::UnterminatedSimpleQuery);
            };
            let query = std::str::from_utf8(query_bytes)
                .map_err(|_| FrontendMessageError::InvalidUtf8)?
                .to_owned();
            Ok(FrontendMessage::SimpleQuery(query))
        }
        b'p' => {
            if let Some(password_bytes) = payload.strip_suffix(&[0]) {
                if !password_bytes.contains(&0) {
                    let password = std::str::from_utf8(password_bytes)
                        .map_err(|_| FrontendMessageError::InvalidUtf8)?
                        .to_owned();
                    return Ok(FrontendMessage::PasswordMessage(password));
                }
            }

            if let Some(mechanism_end) = payload.iter().position(|&b| b == 0) {
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
                return Err(FrontendMessageError::UnterminatedPasswordMessage);
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
                b'S' => DescribeTarget::Statement,
                b'P' => DescribeTarget::Portal,
                _ => return Err(FrontendMessageError::InvalidDescribeTarget),
            };

            let Some(name_bytes) = rest.strip_suffix(&[0]) else {
                return Err(FrontendMessageError::UnterminatedDescribeName);
            };
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
                b'S' => DescribeTarget::Statement,
                b'P' => DescribeTarget::Portal,
                _ => return Err(FrontendMessageError::InvalidCloseTarget),
            };

            let Some(name_bytes) = rest.strip_suffix(&[0]) else {
                return Err(FrontendMessageError::UnterminatedCloseName);
            };
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
            let max_rows = u32::from_be_bytes(
                max_rows_bytes
                    .try_into()
                    .map_err(|_| FrontendMessageError::InvalidExecutePayload)?,
            );

            Ok(FrontendMessage::Execute {
                portal_name,
                max_rows,
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
            let Some(reason_bytes) = payload.strip_suffix(&[0]) else {
                return Err(FrontendMessageError::UnterminatedCopyFail);
            };
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

fn parse_transaction_chain_suffix(tokens: &[&str]) -> Option<bool> {
    if tokens.is_empty() {
        return Some(false);
    }

    if matches!(
        tokens,
        [and, chain]
            if and.eq_ignore_ascii_case("AND") && chain.eq_ignore_ascii_case("CHAIN")
    ) {
        return Some(true);
    }

    if matches!(
        tokens,
        [and, no, chain]
            if and.eq_ignore_ascii_case("AND")
                && no.eq_ignore_ascii_case("NO")
                && chain.eq_ignore_ascii_case("CHAIN")
    ) {
        return Some(false);
    }

    None
}

fn parse_transaction_control_chain(input: &str, keyword: &str) -> Option<bool> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, mut rest) = tokens.split_first()?;
    if !first.eq_ignore_ascii_case(keyword) {
        return None;
    }

    if let Some((scope, tail)) = rest.split_first() {
        if scope.eq_ignore_ascii_case("TRANSACTION") || scope.eq_ignore_ascii_case("WORK") {
            rest = tail;
        }
    }

    parse_transaction_chain_suffix(rest)
}

fn parse_flush_command(input: &str) -> Option<Command> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, rest) = tokens.split_first()?;
    if first.eq_ignore_ascii_case("CHECKPOINT") {
        return match rest {
            [] => Some(Command::Flush),
            _ => None,
        };
    }

    if !first.eq_ignore_ascii_case("FLUSH") {
        return None;
    }

    match rest {
        [] => Some(Command::Flush),
        [target] if target.eq_ignore_ascii_case("WAL") || target.eq_ignore_ascii_case("LOG") => {
            Some(Command::Flush)
        }
        [write_ahead]
            if write_ahead.eq_ignore_ascii_case("WRITE-AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITEAHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD_LOG")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD_WAL") =>
        {
            Some(Command::Flush)
        }
        [write_ahead, target]
            if (write_ahead.eq_ignore_ascii_case("WRITE-AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITEAHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD"))
                && (target.eq_ignore_ascii_case("LOG") || target.eq_ignore_ascii_case("WAL")) =>
        {
            Some(Command::Flush)
        }
        [write, ahead]
            if write.eq_ignore_ascii_case("WRITE") && ahead.eq_ignore_ascii_case("AHEAD") =>
        {
            Some(Command::Flush)
        }
        [write, ahead, target]
            if write.eq_ignore_ascii_case("WRITE")
                && ahead.eq_ignore_ascii_case("AHEAD")
                && (target.eq_ignore_ascii_case("LOG") || target.eq_ignore_ascii_case("WAL")) =>
        {
            Some(Command::Flush)
        }
        _ => None,
    }
}

fn parse_reset_command(input: &str) -> Option<Result<Command, ParseError>> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, rest) = tokens.split_first()?;

    if first.eq_ignore_ascii_case("RESET") {
        return Some(match rest {
            [target]
                if target.eq_ignore_ascii_case("ALL")
                    || target.eq_ignore_ascii_case("ROLE")
                    || target.eq_ignore_ascii_case("AUTHORIZATION")
                    || target.eq_ignore_ascii_case("AUTH") =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH")) =>
            {
                Ok(Command::ResetAll)
            }
            _ => Err(ParseError::InvalidReset),
        });
    }

    if first.eq_ignore_ascii_case("DISCARD") {
        return Some(match rest {
            [target]
                if target.eq_ignore_ascii_case("ALL")
                    || target.eq_ignore_ascii_case("TEMP")
                    || target.eq_ignore_ascii_case("TEMPORARY")
                    || target.eq_ignore_ascii_case("PLANS")
                    || target.eq_ignore_ascii_case("SEQUENCES") =>
            {
                Ok(Command::ResetAll)
            }
            [scope, kind]
                if (scope.eq_ignore_ascii_case("TEMP")
                    || scope.eq_ignore_ascii_case("TEMPORARY"))
                    && (kind.eq_ignore_ascii_case("TABLE")
                        || kind.eq_ignore_ascii_case("TABLES")) =>
            {
                Ok(Command::ResetAll)
            }
            _ => Err(ParseError::InvalidReset),
        });
    }

    if first.eq_ignore_ascii_case("DEALLOCATE") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "DEALLOCATE") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim_start();
        if rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }

        if let Some(after_prepared) = strip_keyword_prefix_case_insensitive(rest, "PREPARED") {
            let tail = after_prepared.trim_start();
            return Some(
                if parse_reset_identifier(tail).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidReset)
                },
            );
        }

        if let Some(after_prepare) = strip_keyword_prefix_case_insensitive(rest, "PREPARE") {
            let tail = after_prepare.trim_start();
            return Some(
                if parse_reset_identifier(tail).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidReset)
                },
            );
        }

        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("CLOSE") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "CLOSE") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim_start();
        if rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }
        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("UNLISTEN") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "UNLISTEN") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim();
        if rest.is_empty() || rest == "*" || rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }
        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("LISTEN") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "LISTEN") else {
            return Some(Err(ParseError::InvalidReset));
        };
        return Some(
            if parse_reset_identifier(rest.trim()).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("NOTIFY") {
        let Some(raw_rest) = strip_keyword_prefix_case_insensitive(input, "NOTIFY") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let Some((_, tail_after_channel)) = parse_reset_identifier(raw_rest.trim_start()) else {
            return Some(Err(ParseError::InvalidReset));
        };
        let tail_after_channel = tail_after_channel.trim_start();
        if tail_after_channel.is_empty() {
            return Some(Ok(Command::ResetAll));
        }
        let Some(payload) = tail_after_channel.strip_prefix(',') else {
            return Some(Err(ParseError::InvalidReset));
        };
        let payload = payload.trim_start();
        return Some(
            if !payload.starts_with(',') && notify_payload_fragment_is_non_empty(payload) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    None
}

fn strip_keyword_prefix_case_insensitive<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    if input.len() < keyword.len() || !input[..keyword.len()].eq_ignore_ascii_case(keyword) {
        return None;
    }
    if input.len() > keyword.len()
        && !input[keyword.len()..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        return None;
    }
    Some(&input[keyword.len()..])
}

fn parse_reset_identifier(input: &str) -> Option<(&str, &str)> {
    let s = input.trim_start();
    if s.is_empty() {
        return None;
    }

    if s.starts_with('"') {
        let bytes = s.as_bytes();
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                let end = i + 1;
                return Some((&s[..end], &s[end..]));
            }
            i += 1;
        }
        return None;
    }

    let end = s
        .find(|c: char| c.is_whitespace() || c == ',')
        .unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((&s[..end], &s[end..]))
}

fn notify_payload_fragment_is_non_empty(fragment: &str) -> bool {
    let trimmed = fragment.trim();
    if trimmed.is_empty() || trimmed.trim_matches(',').trim().is_empty() {
        return false;
    }

    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let chars: Vec<char> = trimmed.chars().collect();
    let mut idx = 0;

    while idx < chars.len() {
        let ch = chars[idx];
        if in_single_quote {
            if ch == '\'' {
                if idx + 1 < chars.len() && chars[idx + 1] == '\'' {
                    idx += 2;
                    continue;
                }
                in_single_quote = false;
            }
            idx += 1;
            continue;
        }

        if in_double_quote {
            if ch == '"' {
                if idx + 1 < chars.len() && chars[idx + 1] == '"' {
                    idx += 2;
                    continue;
                }
                in_double_quote = false;
            }
            idx += 1;
            continue;
        }

        if ch == '$' {
            if let Some((delim, after_start)) = parse_notify_dollar_quote_start(&chars, idx) {
                let mut scan = after_start;
                let mut found = false;
                while scan + delim.len() <= chars.len() {
                    if chars[scan..scan + delim.len()] == delim[..] {
                        idx = scan + delim.len();
                        found = true;
                        break;
                    }
                    scan += 1;
                }
                if !found {
                    return false;
                }
                continue;
            }
        }

        match ch {
            '\'' => in_single_quote = true,
            '"' => in_double_quote = true,
            ',' => return false,
            _ => {}
        }
        idx += 1;
    }

    !(in_single_quote || in_double_quote)
}

fn parse_notify_dollar_quote_start(chars: &[char], start: usize) -> Option<(Vec<char>, usize)> {
    if chars.get(start) != Some(&'$') {
        return None;
    }
    let mut idx = start + 1;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch == '$' {
            return Some((chars[start..=idx].to_vec(), idx + 1));
        }
        if !(ch.is_ascii_alphanumeric() || ch == '_') {
            return None;
        }
        idx += 1;
    }
    None
}

fn split_set_key_value(rest: &str) -> Option<(&str, &str)> {
    if let Some((k, v)) = rest.split_once('=') {
        return Some((k, v));
    }

    let trimmed = rest.trim();
    let (key, tail) = trimmed.split_once(char::is_whitespace)?;
    let tail = tail.trim_start();
    if tail.len() < 2 {
        return None;
    }

    let (keyword, remainder) = tail.split_at(2);
    if !keyword.eq_ignore_ascii_case("TO") {
        return None;
    }

    if remainder.is_empty() || !remainder.starts_with(char::is_whitespace) {
        return None;
    }

    Some((key, remainder.trim_start()))
}

fn parse_set_session_command(rest: &str) -> Option<Result<Command, ParseError>> {
    let tokens: Vec<_> = rest.split_whitespace().collect();

    match tokens.as_slice() {
        [role, target]
            if role.eq_ignore_ascii_case("ROLE")
                && (target.eq_ignore_ascii_case("NONE")
                    || target.eq_ignore_ascii_case("DEFAULT")) =>
        {
            Some(Ok(Command::ResetAll))
        }
        [session, authorization, target]
            if session.eq_ignore_ascii_case("SESSION")
                && authorization.eq_ignore_ascii_case("AUTHORIZATION")
                && !target.is_empty() =>
        {
            Some(Ok(Command::ResetAll))
        }
        [session, auth, target]
            if session.eq_ignore_ascii_case("SESSION")
                && auth.eq_ignore_ascii_case("AUTH")
                && !target.is_empty() =>
        {
            Some(Ok(Command::ResetAll))
        }
        [session, characteristics, as_kw, transaction, suffix @ ..]
            if session.eq_ignore_ascii_case("SESSION")
                && characteristics.eq_ignore_ascii_case("CHARACTERISTICS")
                && as_kw.eq_ignore_ascii_case("AS")
                && transaction.eq_ignore_ascii_case("TRANSACTION")
                && !suffix.is_empty() =>
        {
            Some(Ok(Command::ResetAll))
        }
        [role] if role.eq_ignore_ascii_case("ROLE") => Some(Err(ParseError::InvalidSet)),
        [session, authorization]
            if session.eq_ignore_ascii_case("SESSION")
                && authorization.eq_ignore_ascii_case("AUTHORIZATION") =>
        {
            Some(Err(ParseError::InvalidSet))
        }
        [session, auth]
            if session.eq_ignore_ascii_case("SESSION") && auth.eq_ignore_ascii_case("AUTH") =>
        {
            Some(Err(ParseError::InvalidSet))
        }
        [session, characteristics, as_kw, transaction]
            if session.eq_ignore_ascii_case("SESSION")
                && characteristics.eq_ignore_ascii_case("CHARACTERISTICS")
                && as_kw.eq_ignore_ascii_case("AS")
                && transaction.eq_ignore_ascii_case("TRANSACTION") =>
        {
            Some(Err(ParseError::InvalidSet))
        }
        _ => None,
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BeginModeKind {
    AccessMode,
    IsolationLevel,
    Deferrable,
}

fn parse_begin_mode(tokens: &[String]) -> Option<(usize, BeginModeKind)> {
    let refs: Vec<_> = tokens.iter().map(String::as_str).collect();

    if refs.len() >= 4 && is_isolation_level_suffix(&refs[..4]) {
        return Some((4, BeginModeKind::IsolationLevel));
    }

    if refs.len() >= 3 && is_isolation_level_suffix(&refs[..3]) {
        return Some((3, BeginModeKind::IsolationLevel));
    }

    if refs.len() >= 2 {
        let two = &refs[..2];
        if matches!(
            two,
            [read, only]
                if read.eq_ignore_ascii_case("READ") && only.eq_ignore_ascii_case("ONLY")
        ) || matches!(
            two,
            [read, write]
                if read.eq_ignore_ascii_case("READ") && write.eq_ignore_ascii_case("WRITE")
        ) {
            return Some((2, BeginModeKind::AccessMode));
        }

        if matches!(
            two,
            [not, deferrable]
                if not.eq_ignore_ascii_case("NOT") && deferrable.eq_ignore_ascii_case("DEFERRABLE")
        ) {
            return Some((2, BeginModeKind::Deferrable));
        }
    }

    if !refs.is_empty() && is_deferrable_suffix(&refs[..1]) {
        return Some((1, BeginModeKind::Deferrable));
    }

    None
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
    let mut seen_access_mode = false;
    let mut seen_isolation_level = false;
    let mut seen_deferrable = false;

    while idx < tokens.len() {
        if tokens[idx] == "," {
            return false;
        }

        let Some((consumed, kind)) = parse_begin_mode(&tokens[idx..]) else {
            return false;
        };

        match kind {
            BeginModeKind::AccessMode if seen_access_mode => return false,
            BeginModeKind::IsolationLevel if seen_isolation_level => return false,
            BeginModeKind::Deferrable if seen_deferrable => return false,
            BeginModeKind::AccessMode => seen_access_mode = true,
            BeginModeKind::IsolationLevel => seen_isolation_level = true,
            BeginModeKind::Deferrable => seen_deferrable = true,
        }

        idx += consumed;
        if idx == tokens.len() {
            return true;
        }

        if tokens[idx] == "," {
            idx += 1;
            if idx == tokens.len() || tokens[idx] == "," {
                return false;
            }
        }
    }

    true
}

fn strip_single_leading_comma(tokens: &[String]) -> Option<&[String]> {
    match tokens {
        [first, rest @ ..] if first == "," && !rest.is_empty() => Some(rest),
        _ => None,
    }
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
                if is_begin_mode_list(mode) {
                    true
                } else {
                    strip_single_leading_comma(mode).is_some_and(is_begin_mode_list)
                }
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
                if is_begin_mode_list(mode) {
                    true
                } else {
                    strip_single_leading_comma(mode).is_some_and(is_begin_mode_list)
                }
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
    if let Some(chain) = parse_transaction_control_chain(s, "COMMIT")
        .or_else(|| parse_transaction_control_chain(s, "END"))
    {
        return Ok(Command::Commit { chain });
    }
    if let Some(chain) = parse_transaction_control_chain(s, "ROLLBACK")
        .or_else(|| parse_transaction_control_chain(s, "ABORT"))
    {
        return Ok(Command::Rollback { chain });
    }
    if let Some(flush) = parse_flush_command(s) {
        return Ok(flush);
    }
    if let Some(reset) = parse_reset_command(s) {
        return reset;
    }

    let mut parts = s.splitn(2, char::is_whitespace);
    if let Some(cmd) = parts.next() {
        if cmd.eq_ignore_ascii_case("SET") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidSet);
            };
            if let Some(alias) = parse_set_session_command(rest) {
                return alias;
            }
            let Some((k, v)) = split_set_key_value(rest) else {
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

        if cmd.eq_ignore_ascii_case("DEL") {
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

        if cmd.eq_ignore_ascii_case("DELETE") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidDel);
            };
            let rest = rest.trim();
            let key = if let Some((prefix, remainder)) = rest.split_once(char::is_whitespace) {
                if prefix.eq_ignore_ascii_case("FROM") {
                    let candidate = remainder.trim();
                    if candidate.is_empty() || candidate.chars().any(char::is_whitespace) {
                        return Err(ParseError::InvalidDel);
                    }
                    candidate
                } else {
                    return Err(ParseError::InvalidDel);
                }
            } else if rest.eq_ignore_ascii_case("FROM") {
                return Err(ParseError::InvalidDel);
            } else {
                rest
            };

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
    fn parses_set_with_to_assignment_alias() {
        let cmd = parse_command("SET a TO 42").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "a".into(),
                value: "42".into()
            }
        );

        let cmd = parse_command("SET alpha to value words").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "alpha".into(),
                value: "value words".into()
            }
        );
    }

    #[test]
    fn parses_postgres_style_set_session_reset_aliases() {
        assert_eq!(parse_command("SET ROLE NONE").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("SET ROLE DEFAULT").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION AUTHORIZATION DEFAULT").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION AUTH postgres").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command(
                "SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED"
            )
            .unwrap(),
            Command::ResetAll
        );
    }

    #[test]
    fn rejects_set_with_whitespace_in_key() {
        assert!(matches!(
            parse_command("SET two words=42"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET a TO"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET a TO42"),
            Err(ParseError::InvalidSet)
        ));
    }

    #[test]
    fn parses_flush() {
        let cmd = parse_command("CHECKPOINT").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE AHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE AHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE AHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE-AHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE-AHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE-AHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITEAHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITEAHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITEAHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD WAL").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD_LOG").unwrap();
        assert_eq!(cmd, Command::Flush);

        let cmd = parse_command("FLUSH WRITE_AHEAD_WAL").unwrap();
        assert_eq!(cmd, Command::Flush);
    }

    #[test]
    fn parses_reset_all() {
        let cmd = parse_command("RESET ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET ROLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET AUTHORIZATION").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET AUTH").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTHORIZATION").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTH").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMP").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMPORARY").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMP TABLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMP TABLES").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMPORARY TABLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD TEMPORARY TABLES").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD PLANS").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DISCARD SEQUENCES").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE prepared_stmt").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE PREPARE prepared_stmt").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE PREPARED prepared_stmt").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE \"prepared stmt\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE PREPARE \"prepared stmt\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("DEALLOCATE PREPARED \"prepared stmt\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("CLOSE ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("CLOSE cursor_name").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("CLOSE \"cursor name\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("UNLISTEN").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("UNLISTEN *").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("UNLISTEN ALL").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("UNLISTEN updates_channel").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("UNLISTEN \"updates channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("UNLISTEN \"updates,channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("LISTEN updates_channel").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("LISTEN \"updates channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY \"updates channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY \"updates,channel\", 'hello'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, 'hello'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel , 'hello'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel,'hello'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel ,'{\"ok\":true}'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, '{\"ok\":true,\"n\":1}'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, $$hello,world$$").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, $tag$hello,world$tag$").unwrap();
        assert_eq!(cmd, Command::ResetAll);
    }

    #[test]
    fn parses_transaction_control_commands_case_insensitively() {
        assert_eq!(parse_command("begin").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("COMMIT").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("rOlLbAcK").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("abort").unwrap(),
            Command::Rollback { chain: false }
        );
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
        assert_eq!(
            parse_command("BEGIN READ ONLY DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN READ WRITE ISOLATION LEVEL SERIALIZABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("START TRANSACTION").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("BEGIN TRANSACTION, READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN WORK, READ WRITE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION, READ ONLY").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION ISOLATION LEVEL REPEATABLE READ").unwrap(),
            Command::Begin
        );
        assert_eq!(parse_command("START WORK").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("START WORK, READ WRITE, DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK ISOLATION LEVEL READ COMMITTED").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK, ISOLATION LEVEL REPEATABLE READ, NOT DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START TRANSACTION READ ONLY DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK READ WRITE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("COMMIT WORK").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("COMMIT TRANSACTION").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("COMMIT AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("COMMIT AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("COMMIT TRANSACTION AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("COMMIT WORK AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END WORK").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END TRANSACTION").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("END AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("END TRANSACTION AND CHAIN").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("END WORK AND NO CHAIN").unwrap(),
            Command::Commit { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK WORK").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK TRANSACTION").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ROLLBACK AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ROLLBACK TRANSACTION AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ROLLBACK WORK AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT WORK").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT TRANSACTION").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ABORT AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
        assert_eq!(
            parse_command("ABORT TRANSACTION AND CHAIN").unwrap(),
            Command::Rollback { chain: true }
        );
        assert_eq!(
            parse_command("ABORT WORK AND NO CHAIN").unwrap(),
            Command::Rollback { chain: false }
        );
    }

    #[test]
    fn parses_begin_mode_lists_with_mixed_order_and_delimiters() {
        assert_eq!(
            parse_command(
                "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY, NOT DEFERRABLE"
            )
            .unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("BEGIN READ WRITE ISOLATION LEVEL READ COMMITTED DEFERRABLE").unwrap(),
            Command::Begin
        );
        assert_eq!(
            parse_command("START WORK, NOT DEFERRABLE, ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                .unwrap(),
            Command::Begin
        );
    }

    #[test]
    fn rejects_begin_mode_lists_with_duplicate_mode_kinds_even_when_comma_delimited() {
        assert!(matches!(
            parse_command("BEGIN READ ONLY, READ WRITE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN DEFERRABLE, NOT DEFERRABLE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command(
                "START TRANSACTION ISOLATION LEVEL READ COMMITTED, ISOLATION LEVEL SERIALIZABLE"
            ),
            Err(ParseError::Unsupported(_))
        ));
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
            parse_command("BEGIN READ ONLY,, DEFERRABLE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START TRANSACTION READ"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN TRANSACTION,"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START WORK, "),
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
            parse_command("BEGIN READ ONLY READ WRITE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN READ ONLY, READ WRITE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("START TRANSACTION READ ONLY, READ ONLY"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command(
                "START WORK ISOLATION LEVEL READ COMMITTED, ISOLATION LEVEL SERIALIZABLE"
            ),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN DEFERRABLE NOT DEFERRABLE"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("BEGIN ISOLATION LEVEL SERIALIZABLE, ISOLATION LEVEL READ COMMITTED"),
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
            parse_command("END AND"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("END WORK AND"),
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
        assert!(matches!(
            parse_command("FLUSH NOW"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("FLUSH WAL NOW"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("CHECKPOINT NOW"),
            Err(ParseError::Unsupported(_))
        ));
        assert!(matches!(
            parse_command("RESET"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("RESET SESSION"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("RESET ALL NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DISCARD"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DISCARD TEMP NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DISCARD ALL NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE PREPARE"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE PREPARE x y"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE PREPARED"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE PREPARED x y"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE a,b"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE PREPARE a,b"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("DEALLOCATE \"prepared stmt"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("CLOSE"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("CLOSE cursor_name NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("CLOSE \"cursor name"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("UNLISTEN * NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("UNLISTEN a,b"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("UNLISTEN \"updates channel"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("LISTEN"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("LISTEN updates_channel NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel payload"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel ,"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel,"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, ,"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel , ,"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, , ,"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel,,payload"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel,, payload"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, payload, extra"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, 'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, $$unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, $tag$unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY \"updates channel"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("SET ROLE"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET SESSION AUTHORIZATION"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET SESSION AUTH"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION"),
            Err(ParseError::InvalidSet)
        ));
    }

    #[test]
    fn accepts_optional_statement_terminator() {
        assert_eq!(parse_command("BEGIN;").unwrap(), Command::Begin);
        assert_eq!(parse_command("START WORK;").unwrap(), Command::Begin);
        assert_eq!(
            parse_command("END AND CHAIN;").unwrap(),
            Command::Commit { chain: true }
        );
        assert_eq!(
            parse_command("ABORT AND NO CHAIN;\n").unwrap(),
            Command::Rollback { chain: false }
        );
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
        assert_eq!(parse_command("RESET ALL;").unwrap(), Command::ResetAll);
        assert_eq!(parse_command("RESET ROLE;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("RESET AUTHORIZATION;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(parse_command("RESET AUTH;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("RESET SESSION AUTHORIZATION;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("RESET SESSION AUTH;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(parse_command("DISCARD ALL;\n").unwrap(), Command::ResetAll);
        assert_eq!(parse_command("CLOSE ALL;\n").unwrap(), Command::ResetAll);
        assert_eq!(parse_command("UNLISTEN *;\n").unwrap(), Command::ResetAll);
        assert_eq!(parse_command("UNLISTEN ALL;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("LISTEN updates_channel;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("NOTIFY updates_channel;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("NOTIFY updates_channel, 'payload';\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET ROLE NONE;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION AUTHORIZATION DEFAULT;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("FLUSH WRITE AHEAD LOG;\n").unwrap(),
            Command::Flush
        );
        assert_eq!(parse_command("CHECKPOINT;\n").unwrap(), Command::Flush);
        assert_eq!(
            parse_command("FLUSH WRITE_AHEAD_LOG;\n").unwrap(),
            Command::Flush
        );
        assert_eq!(parse_command("DISCARD TEMP;\n").unwrap(), Command::ResetAll);
        assert_eq!(
            parse_command("DISCARD TEMP TABLES;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("DEALLOCATE ALL;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("DEALLOCATE PREPARE prepared_stmt;\n").unwrap(),
            Command::ResetAll
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

    fn with_length_prefix(mut payload: Vec<u8>) -> Vec<u8> {
        let len = (payload.len() + 4) as u32;
        let mut frame = len.to_be_bytes().to_vec();
        frame.append(&mut payload);
        frame
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_params() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice(b"user\0postgres\0database\0gpu\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("user".to_string(), "postgres".to_string()),
                    ("database".to_string(), "gpu".to_string())
                ],
            }
        );
    }

    #[test]
    fn parses_ssl_and_cancel_requests() {
        let ssl = with_length_prefix(PG_SSL_REQUEST_CODE.to_be_bytes().to_vec());
        assert_eq!(
            parse_startup_packet(&ssl).unwrap(),
            StartupPacket::SslRequest
        );

        let mut cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
        cancel_payload.extend_from_slice(&123u32.to_be_bytes());
        cancel_payload.extend_from_slice(&456u32.to_be_bytes());
        let cancel = with_length_prefix(cancel_payload);
        assert_eq!(
            parse_startup_packet(&cancel).unwrap(),
            StartupPacket::CancelRequest {
                process_id: 123,
                secret_key: 456,
            }
        );
    }

    #[test]
    fn rejects_startup_packet_with_length_or_parameter_errors() {
        let short = vec![0, 0, 0, 8, 0, 3, 0];
        assert_eq!(
            parse_startup_packet(&short).unwrap_err(),
            StartupPacketError::TooShort
        );

        let bad_len = vec![0, 0, 0, 10, 0, 3, 0, 0, 0];
        assert_eq!(
            parse_startup_packet(&bad_len).unwrap_err(),
            StartupPacketError::LengthMismatch {
                expected: 10,
                actual: 9,
            }
        );

        let mut bad_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        bad_params.extend_from_slice(b"user\0postgres");
        let bad_params = with_length_prefix(bad_params);
        assert_eq!(
            parse_startup_packet(&bad_params).unwrap_err(),
            StartupPacketError::UnterminatedParameterPayload
        );

        let mut invalid_utf8_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        invalid_utf8_params.extend_from_slice(&[0xFF, 0, b'v', 0, 0]);
        let invalid_utf8_params = with_length_prefix(invalid_utf8_params);
        assert_eq!(
            parse_startup_packet(&invalid_utf8_params).unwrap_err(),
            StartupPacketError::InvalidUtf8
        );
    }

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

        let password = frontend_frame(b'p', b"secret\0");
        assert_eq!(
            parse_frontend_message(&password).unwrap(),
            FrontendMessage::PasswordMessage("secret".to_string())
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

        let sasl_response = frontend_frame(b'p', b"c=biws,r=nonce,p=proof");
        assert_eq!(
            parse_frontend_message(&sasl_response).unwrap(),
            FrontendMessage::SaslResponse(b"c=biws,r=nonce,p=proof".to_vec())
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

        let copy_data = frontend_frame(b'd', &[0, 1, 2, 3]);
        assert_eq!(
            parse_frontend_message(&copy_data).unwrap(),
            FrontendMessage::CopyData(vec![0, 1, 2, 3])
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

        let unterminated = frontend_frame(b'Q', b"SELECT 1;");
        assert_eq!(
            parse_frontend_message(&unterminated).unwrap_err(),
            FrontendMessageError::UnterminatedSimpleQuery
        );

        let unterminated_password = frontend_frame(b'p', b"secret");
        assert_eq!(
            parse_frontend_message(&unterminated_password).unwrap(),
            FrontendMessage::SaslResponse(b"secret".to_vec())
        );

        let malformed_sasl_initial = frontend_frame(b'p', b"SCRAM-SHA-256\0\0\0");
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_sasl_initial_negative_len = frontend_frame(
            b'p',
            &[b'S', b'C', b'R', b'A', b'M', 0, 0xFF, 0xFF, 0xFF, 0xFE],
        );
        assert_eq!(
            parse_frontend_message(&malformed_sasl_initial_negative_len).unwrap_err(),
            FrontendMessageError::InvalidSaslInitialResponsePayload
        );

        let malformed_parse = frontend_frame(b'P', b"stmt\0SELECT 1\0\0\x01");
        assert_eq!(
            parse_frontend_message(&malformed_parse).unwrap_err(),
            FrontendMessageError::InvalidParseParameterPayload
        );

        let invalid_describe_target = frontend_frame(b'D', b"Xstmt\0");
        assert_eq!(
            parse_frontend_message(&invalid_describe_target).unwrap_err(),
            FrontendMessageError::InvalidDescribeTarget
        );

        let invalid_close_target = frontend_frame(b'C', b"Xstmt\0");
        assert_eq!(
            parse_frontend_message(&invalid_close_target).unwrap_err(),
            FrontendMessageError::InvalidCloseTarget
        );

        let malformed_execute = frontend_frame(b'E', b"portal\0\0\0");
        assert_eq!(
            parse_frontend_message(&malformed_execute).unwrap_err(),
            FrontendMessageError::InvalidExecutePayload
        );

        let malformed_bind = frontend_frame(b'B', b"portal\0stmt\0\0\x01");
        assert_eq!(
            parse_frontend_message(&malformed_bind).unwrap_err(),
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

        let malformed_function_call = frontend_frame(b'F', b"\0\0\0*");
        assert_eq!(
            parse_frontend_message(&malformed_function_call).unwrap_err(),
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

        let malformed_copy_done = frontend_frame(b'c', &[0]);
        assert_eq!(
            parse_frontend_message(&malformed_copy_done).unwrap_err(),
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

        let invalid_utf8_password = frontend_frame(b'p', &[0xFF, 0]);
        assert_eq!(
            parse_frontend_message(&invalid_utf8_password).unwrap_err(),
            FrontendMessageError::InvalidUtf8
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
}
