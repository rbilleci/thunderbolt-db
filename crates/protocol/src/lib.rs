#[allow(clippy::large_enum_variant)]
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
    CreateTable(CreateTable),
    Insert(Insert),
    Delete(Delete),
    Select(Select),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTable {
    pub table: String,
    pub columns: Vec<ColumnDef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub ty: SqlType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlType {
    Int4,
    Text,
}

pub const SUPPORTED_SQL_TYPES: [SqlType; 2] = [SqlType::Int4, SqlType::Text];

impl SqlType {
    pub const fn postgres_oid(self) -> u32 {
        match self {
            Self::Int4 => 23,
            Self::Text => 25,
        }
    }

    pub const fn type_size(self) -> i16 {
        match self {
            Self::Int4 => 4,
            Self::Text => -1,
        }
    }

    pub const fn catalog_name(self) -> &'static str {
        match self {
            Self::Int4 => "int4",
            Self::Text => "text",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Insert {
    pub table: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<SqlValue>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delete {
    pub table: String,
    pub filter: Option<SelectFilter>,
    pub filters: Vec<SelectFilter>,
    pub filter_groups: Vec<Vec<SelectFilter>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SqlValue {
    Int4(i32),
    Int8(i64),
    Numeric(String),
    Text(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Select {
    pub table: String,
    pub distinct: bool,
    pub projection: SelectProjection,
    pub group_by: Option<String>,
    pub filter: Option<SelectFilter>,
    pub filters: Vec<SelectFilter>,
    pub filter_groups: Vec<Vec<SelectFilter>>,
    pub order_by: Option<SelectOrder>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectProjection {
    All,
    Columns(Vec<String>),
    CountAll,
    GroupedCount {
        column: String,
    },
    Sum {
        column: String,
    },
    GroupedSum {
        group_column: String,
        sum_column: String,
    },
    Avg {
        column: String,
    },
    GroupedAvg {
        group_column: String,
        avg_column: String,
    },
    Min {
        column: String,
    },
    GroupedMin {
        group_column: String,
        min_column: String,
    },
    Max {
        column: String,
    },
    GroupedMax {
        group_column: String,
        max_column: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectFilter {
    pub column: String,
    pub op: SelectFilterOp,
    pub value: SqlValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectFilterOp {
    Eq,
    Lt,
    Lte,
    Gt,
    Gte,
    LikePrefix,
}

impl SelectFilterOp {
    fn flipped(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Lt => Self::Gt,
            Self::Lte => Self::Gte,
            Self::Gt => Self::Lt,
            Self::Gte => Self::Lte,
            Self::LikePrefix => Self::LikePrefix,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectOrder {
    pub column: String,
    pub descending: bool,
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
    #[error("invalid relational SQL syntax; supported subset: CREATE TABLE name (...), INSERT INTO name (...) VALUES (...), DELETE FROM name WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...], SELECT [DISTINCT] columns|COUNT(*)|SUM(int4_column)|AVG(int4_column)|MIN(column)|MAX(column)|column, COUNT(*)|column, SUM(int4_column)|column, AVG(int4_column)|column, MIN(column)|column, MAX(column) FROM name [WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...]] [GROUP BY column] [ORDER BY selected_column|count|sum|avg|min|max [ASC|DESC]] [LIMIT n] [OFFSET n]")]
    InvalidRelationalSql,
    #[error("LIMIT must not be negative")]
    NegativeLimit,
    #[error("OFFSET must not be negative")]
    NegativeOffset,
    #[error("invalid RESET/DISCARD/DEALLOCATE/CLOSE/LISTEN/NOTIFY/UNLISTEN syntax; expected: RESET ALL|ROLE|AUTHORIZATION|AUTH|SESSION AUTHORIZATION[ [TO] DEFAULT]|SESSION AUTH[ [TO] DEFAULT], DISCARD {{ALL|TEMP|TEMPORARY|TEMP TABLES|TEMPORARY TABLES|PLANS|SEQUENCES}}, DEALLOCATE {{ALL|name|PREPARE|PREPARED name}}, CLOSE {{ALL|name}}, LISTEN channel, NOTIFY channel[, payload], or UNLISTEN [*|ALL|channel]")]
    InvalidReset,
}

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
            [scope, role]
                if (scope.eq_ignore_ascii_case("SESSION")
                    || scope.eq_ignore_ascii_case("LOCAL"))
                    && role.eq_ignore_ascii_case("ROLE") =>
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
            [session, authorization, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && default.eq_ignore_ascii_case("DEFAULT") =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization, to, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && to.eq_ignore_ascii_case("TO")
                    && default.eq_ignore_ascii_case("DEFAULT") =>
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
                if i == 1 {
                    return None;
                }
                let end = i + 1;
                return Some((&s[..end], &s[end..]));
            }
            i += 1;
        }
        return None;
    }

    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return None;
    }

    let mut end = first.len_utf8();
    for (idx, ch) in chars {
        if ch.is_whitespace() || ch == ',' {
            end = idx;
            break;
        }
        if !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()) {
            return None;
        }
        end = idx + ch.len_utf8();
    }

    Some((&s[..end], &s[end..]))
}

fn notify_payload_fragment_is_non_empty(fragment: &str) -> bool {
    let trimmed = fragment.trim();
    if trimmed.is_empty() || trimmed.trim_matches(',').trim().is_empty() {
        return false;
    }

    let mut in_single_quote = false;
    let mut single_quote_backslash_escapes = false;
    let mut in_double_quote = false;
    let chars: Vec<char> = trimmed.chars().collect();
    let mut idx = 0;

    while idx < chars.len() {
        let ch = chars[idx];
        if in_single_quote {
            if single_quote_backslash_escapes && ch == '\\' && idx + 1 < chars.len() {
                idx += 2;
                continue;
            }

            if ch == '\'' {
                if idx + 1 < chars.len() && chars[idx + 1] == '\'' {
                    idx += 2;
                    continue;
                }
                in_single_quote = false;
                single_quote_backslash_escapes = false;
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

        if ch.is_whitespace() {
            if chars[idx + 1..].iter().any(|c| !c.is_whitespace()) {
                return false;
            }
            break;
        }

        if ch == '\'' {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 1;
            continue;
        }

        if matches!(ch, 'e' | 'E') && chars.get(idx + 1) == Some(&'\'') {
            in_single_quote = true;
            single_quote_backslash_escapes = true;
            idx += 2;
            continue;
        }

        if matches!(ch, 'b' | 'B' | 'x' | 'X') && chars.get(idx + 1) == Some(&'\'') {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 2;
            continue;
        }

        if matches!(ch, 'u' | 'U')
            && chars.get(idx + 1) == Some(&'&')
            && chars.get(idx + 2) == Some(&'\'')
        {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 3;
            continue;
        }

        match ch {
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

fn strip_set_scope_prefix<'a>(input: &'a str, scope: &str) -> Option<&'a str> {
    let trimmed = input.trim_start();
    let after_scope = strip_keyword_prefix_case_insensitive(trimmed, scope)?;
    if after_scope.trim().is_empty() {
        return None;
    }
    Some(after_scope.trim_start())
}

fn parse_relational_command(input: &str) -> Option<Result<Command, ParseError>> {
    let first = input.split_whitespace().next()?;
    if first.eq_ignore_ascii_case("CREATE") {
        return Some(parse_create_table(input).map(Command::CreateTable));
    }
    if first.eq_ignore_ascii_case("INSERT") {
        return Some(parse_insert(input).map(Command::Insert));
    }
    if first.eq_ignore_ascii_case("DELETE")
        && strip_keyword_prefix_case_insensitive(input, "DELETE")
            .map(str::trim_start)
            .and_then(|tail| strip_keyword_prefix_case_insensitive(tail, "FROM"))
            .is_some_and(|tail| find_keyword_outside_quotes(tail, "WHERE").is_some())
    {
        return Some(parse_delete(input).map(Command::Delete));
    }
    if first.eq_ignore_ascii_case("SELECT") {
        return Some(parse_select(input).map(Command::Select));
    }
    None
}

fn parse_create_table(input: &str) -> Result<CreateTable, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = rest.rfind(')').ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !rest[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let table = normalize_relation_identifier(rest[..open].trim())?;
    let mut columns = Vec::new();
    for raw_column in split_csv(&rest[open + 1..close])? {
        let mut parts = raw_column.split_whitespace();
        let name = parts
            .next()
            .ok_or(ParseError::InvalidRelationalSql)
            .and_then(normalize_identifier)?;
        let ty = match parts.next().ok_or(ParseError::InvalidRelationalSql)? {
            ty if parse_supported_sql_type_name(ty) == Some(SqlType::Int4) => SqlType::Int4,
            ty if parse_supported_sql_type_name(ty) == Some(SqlType::Text) => SqlType::Text,
            _ => return Err(ParseError::InvalidRelationalSql),
        };
        if parts.next().is_some() {
            return Err(ParseError::InvalidRelationalSql);
        }
        columns.push(ColumnDef { name, ty });
    }
    if columns.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateTable { table, columns })
}

fn parse_supported_sql_type_name(input: &str) -> Option<SqlType> {
    let ty = if input
        .get(.."pg_catalog.".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("pg_catalog."))
    {
        &input["pg_catalog.".len()..]
    } else {
        input
    };
    if ty.eq_ignore_ascii_case("INT")
        || ty.eq_ignore_ascii_case("INT4")
        || ty.eq_ignore_ascii_case("INTEGER")
    {
        Some(SqlType::Int4)
    } else if ty.eq_ignore_ascii_case("TEXT") {
        Some(SqlType::Text)
    } else {
        None
    }
}

fn parse_insert(input: &str) -> Result<Insert, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "INSERT")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "INTO"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let values_pos =
        find_keyword_outside_quotes(rest, "VALUES").ok_or(ParseError::InvalidRelationalSql)?;
    let target = rest[..values_pos].trim();
    let values = rest[values_pos + "VALUES".len()..].trim_start();
    let (table, columns) = if let Some(open) = target.find('(') {
        let close = find_matching_paren(target, open).ok_or(ParseError::InvalidRelationalSql)?;
        if !target[close + 1..].trim().is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let columns = split_csv(&target[open + 1..close])?
            .into_iter()
            .map(|column| normalize_identifier(column.trim()))
            .collect::<Result<Vec<_>, _>>()?;
        if columns.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        (
            normalize_relation_identifier(target[..open].trim())?,
            columns,
        )
    } else {
        (normalize_relation_identifier(target)?, Vec::new())
    };
    let mut rows = Vec::new();
    let mut tail = values;
    loop {
        let open = tail.find('(').ok_or(ParseError::InvalidRelationalSql)?;
        if !tail[..open].trim().is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let close = find_matching_paren(tail, open).ok_or(ParseError::InvalidRelationalSql)?;
        let row = split_csv(&tail[open + 1..close])?
            .into_iter()
            .map(parse_sql_value)
            .collect::<Result<Vec<_>, _>>()?;
        if !columns.is_empty() && row.len() != columns.len() {
            return Err(ParseError::InvalidRelationalSql);
        }
        rows.push(row);
        tail = tail[close + 1..].trim_start();
        if tail.is_empty() {
            break;
        }
        let Some(after_comma) = tail.strip_prefix(',') else {
            return Err(ParseError::InvalidRelationalSql);
        };
        tail = after_comma.trim_start();
    }
    Ok(Insert {
        table,
        columns,
        rows,
    })
}

fn parse_delete(input: &str) -> Result<Delete, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "DELETE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "FROM"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let where_pos =
        find_keyword_outside_quotes(rest, "WHERE").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..where_pos].trim())?;
    let filter_input = rest[where_pos + "WHERE".len()..].trim();
    if filter_input.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let filter_groups = parse_select_filter_groups(filter_input)?;
    let filters = filter_groups.first().cloned().unwrap_or_default();
    Ok(Delete {
        table,
        filter: filters.first().cloned(),
        filters,
        filter_groups,
    })
}

fn parse_select(input: &str) -> Result<Select, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "SELECT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let from_pos =
        find_keyword_outside_quotes(rest, "FROM").ok_or(ParseError::InvalidRelationalSql)?;
    let mut projection_input = rest[..from_pos].trim();
    let distinct = if let Some(after_distinct) =
        strip_keyword_prefix_case_insensitive(projection_input, "DISTINCT")
    {
        projection_input = after_distinct.trim_start();
        true
    } else {
        false
    };
    let projection = parse_projection(projection_input)?;
    if distinct
        && matches!(
            projection,
            SelectProjection::All
                | SelectProjection::CountAll
                | SelectProjection::GroupedCount { .. }
                | SelectProjection::Sum { .. }
                | SelectProjection::GroupedSum { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::GroupedAvg { .. }
                | SelectProjection::Min { .. }
                | SelectProjection::GroupedMin { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::GroupedMax { .. }
        )
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut tail = rest[from_pos + "FROM".len()..].trim_start();
    if let Some(after_only) = strip_keyword_prefix_case_insensitive(tail, "ONLY") {
        tail = after_only.trim_start();
    }
    let table_end = tail.find(char::is_whitespace).unwrap_or(tail.len());
    let table = normalize_relation_identifier(&tail[..table_end])?;
    tail = tail[table_end..].trim_start();

    let mut filter_groups = Vec::new();
    let mut group_by = None;
    let mut order_by = None;
    let mut limit = None;
    let mut offset = None;
    while !tail.is_empty() {
        if let Some(after_where) = strip_keyword_prefix_case_insensitive(tail, "WHERE") {
            let after_where = after_where.trim_start();
            let next = next_clause_pos(after_where).unwrap_or(after_where.len());
            filter_groups = parse_select_filter_groups(after_where[..next].trim())?;
            tail = after_where[next..].trim_start();
        } else if let Some(after_group) = strip_keyword_prefix_case_insensitive(tail, "GROUP") {
            let after_by = strip_keyword_prefix_case_insensitive(after_group.trim_start(), "BY")
                .ok_or(ParseError::InvalidRelationalSql)?
                .trim_start();
            let next = next_clause_pos(after_by).unwrap_or(after_by.len());
            group_by = Some(normalize_identifier(after_by[..next].trim())?);
            tail = after_by[next..].trim_start();
        } else if let Some(after_order) = strip_keyword_prefix_case_insensitive(tail, "ORDER") {
            let after_by = strip_keyword_prefix_case_insensitive(after_order.trim_start(), "BY")
                .ok_or(ParseError::InvalidRelationalSql)?
                .trim_start();
            let next = next_clause_pos(after_by).unwrap_or(after_by.len());
            order_by = Some(parse_select_order(after_by[..next].trim())?);
            tail = after_by[next..].trim_start();
        } else if let Some(after_limit) = strip_keyword_prefix_case_insensitive(tail, "LIMIT") {
            let after_limit = after_limit.trim_start();
            let next = next_clause_pos(after_limit).unwrap_or(after_limit.len());
            limit = Some(parse_select_limit(after_limit[..next].trim())?);
            tail = after_limit[next..].trim_start();
        } else if let Some(after_offset) = strip_keyword_prefix_case_insensitive(tail, "OFFSET") {
            let after_offset = after_offset.trim_start();
            let next = next_clause_pos(after_offset).unwrap_or(after_offset.len());
            offset = Some(parse_select_offset(after_offset[..next].trim())?);
            tail = after_offset[next..].trim_start();
        } else {
            return Err(ParseError::InvalidRelationalSql);
        }
    }

    let filters = filter_groups.first().cloned().unwrap_or_default();
    Ok(Select {
        table,
        distinct,
        projection,
        group_by,
        filter: filters.first().cloned(),
        filters,
        filter_groups,
        order_by,
        limit,
        offset,
    })
}

fn parse_projection(input: &str) -> Result<SelectProjection, ParseError> {
    if input == "*" {
        return Ok(SelectProjection::All);
    }
    if input.eq_ignore_ascii_case("COUNT(*)") {
        return Ok(SelectProjection::CountAll);
    }
    if let Some(column) = parse_aggregate_call(input, "SUM")? {
        return Ok(SelectProjection::Sum { column });
    }
    if let Some(column) = parse_aggregate_call(input, "AVG")? {
        return Ok(SelectProjection::Avg { column });
    }
    if let Some(column) = parse_aggregate_call(input, "MIN")? {
        return Ok(SelectProjection::Min { column });
    }
    if let Some(column) = parse_aggregate_call(input, "MAX")? {
        return Ok(SelectProjection::Max { column });
    }
    let items = split_csv(input)?;
    if items.len() == 2 && items[1].trim().eq_ignore_ascii_case("COUNT(*)") {
        return Ok(SelectProjection::GroupedCount {
            column: normalize_identifier(items[0].trim())?,
        });
    }
    if items.len() == 2 {
        if let Some(sum_column) = parse_aggregate_call(items[1].trim(), "SUM")? {
            return Ok(SelectProjection::GroupedSum {
                group_column: normalize_identifier(items[0].trim())?,
                sum_column,
            });
        }
        if let Some(avg_column) = parse_aggregate_call(items[1].trim(), "AVG")? {
            return Ok(SelectProjection::GroupedAvg {
                group_column: normalize_identifier(items[0].trim())?,
                avg_column,
            });
        }
        if let Some(min_column) = parse_aggregate_call(items[1].trim(), "MIN")? {
            return Ok(SelectProjection::GroupedMin {
                group_column: normalize_identifier(items[0].trim())?,
                min_column,
            });
        }
        if let Some(max_column) = parse_aggregate_call(items[1].trim(), "MAX")? {
            return Ok(SelectProjection::GroupedMax {
                group_column: normalize_identifier(items[0].trim())?,
                max_column,
            });
        }
    }
    if items.iter().any(|item| {
        item.trim().eq_ignore_ascii_case("COUNT(*)") || aggregate_call_name(item.trim()).is_some()
    }) {
        return Err(ParseError::InvalidRelationalSql);
    }
    let columns = items
        .into_iter()
        .map(|column| normalize_identifier(column.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    if columns.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(SelectProjection::Columns(columns))
}

fn parse_aggregate_call(input: &str, expected: &str) -> Result<Option<String>, ParseError> {
    let Some(name) = aggregate_call_name(input) else {
        return Ok(None);
    };
    if !name.eq_ignore_ascii_case(expected) {
        return Ok(None);
    }
    let open = input.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = input.rfind(')').ok_or(ParseError::InvalidRelationalSql)?;
    if close + 1 != input.len() || close <= open + 1 {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(Some(normalize_identifier(input[open + 1..close].trim())?))
}

fn aggregate_call_name(input: &str) -> Option<&str> {
    let open = input.find('(')?;
    if !input.ends_with(')') {
        return None;
    }
    let name = input[..open].trim();
    let mut chars = name.chars();
    let first = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return None;
    }
    if chars.any(|ch| !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())) {
        return None;
    }
    (!name.is_empty()).then_some(name)
}

fn parse_select_limit(input: &str) -> Result<usize, ParseError> {
    match parse_sql_value(input)? {
        SqlValue::Int4(value) if value >= 0 => Ok(value as usize),
        SqlValue::Int4(_) => Err(ParseError::NegativeLimit),
        SqlValue::Int8(_) | SqlValue::Numeric(_) | SqlValue::Text(_) => {
            Err(ParseError::InvalidRelationalSql)
        }
    }
}

fn parse_select_offset(input: &str) -> Result<usize, ParseError> {
    match parse_sql_value(input)? {
        SqlValue::Int4(value) if value >= 0 => Ok(value as usize),
        SqlValue::Int4(_) => Err(ParseError::NegativeOffset),
        SqlValue::Int8(_) | SqlValue::Numeric(_) | SqlValue::Text(_) => {
            Err(ParseError::InvalidRelationalSql)
        }
    }
}

fn parse_select_filter(input: &str) -> Result<SelectFilter, ParseError> {
    let input = trim_wrapping_parentheses(input)?;
    let (left, op, right) = split_select_filter(input)?;
    let left = left.trim();
    let right = right.trim();
    if let Ok(value) = parse_sql_value(right) {
        return Ok(SelectFilter {
            column: normalize_identifier(left)?,
            op,
            value,
        });
    }
    if let Ok(value) = parse_sql_value(left) {
        return Ok(SelectFilter {
            column: normalize_identifier(right)?,
            op: op.flipped(),
            value,
        });
    }
    Err(ParseError::InvalidRelationalSql)
}

fn parse_select_filter_groups(input: &str) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let groups = parse_select_filter_or_groups(input)?;
    if groups.is_empty() || groups.iter().any(Vec::is_empty) {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(groups)
}

fn parse_select_filter_or_groups(input: &str) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let input = trim_wrapping_parentheses(input)?;
    let parts = split_keyword_chain_outside_quotes(input, "OR")?;
    if parts.len() == 1 {
        return parse_select_filter_and_groups(input);
    }
    let groups = parts
        .into_iter()
        .map(parse_select_filter_and_groups)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok(groups)
}

fn parse_select_filter_and_groups(input: &str) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let parts = split_select_and_chain_outside_quotes(input)?;
    if parts.len() == 1 {
        return parse_select_filter_factor(input);
    }
    let mut groups = vec![Vec::new()];
    for part in parts {
        let factor_groups = parse_select_filter_factor(part)?;
        let mut combined = Vec::new();
        for existing in &groups {
            for factor_group in &factor_groups {
                let mut group = existing.clone();
                group.extend(factor_group.iter().cloned());
                combined.push(group);
            }
        }
        groups = combined;
    }
    Ok(groups)
}

fn parse_select_filter_factor(input: &str) -> Result<Vec<Vec<SelectFilter>>, ParseError> {
    let input = input.trim();
    if input.starts_with('(') {
        let close = find_matching_paren(input, 0).ok_or(ParseError::InvalidRelationalSql)?;
        if close == input.len() - 1 {
            return parse_select_filter_or_groups(&input[1..close]);
        }
    }
    if let Some(group) = parse_select_between_filter_group(input)? {
        return Ok(vec![group]);
    }
    if let Some(groups) = parse_select_in_filter_groups(input)? {
        return Ok(groups);
    }
    if let Some(filter) = parse_select_like_prefix_filter(input)? {
        return Ok(vec![vec![filter]]);
    }
    Ok(vec![vec![parse_select_filter(input)?]])
}

fn parse_select_between_filter_group(input: &str) -> Result<Option<Vec<SelectFilter>>, ParseError> {
    let Some(pos) = find_keyword_outside_quotes(input, "BETWEEN") else {
        return Ok(None);
    };
    let column = normalize_identifier(input[..pos].trim())?;
    let bounds = input[pos + "BETWEEN".len()..].trim();
    let Some(and_pos) = find_keyword_outside_quotes(bounds, "AND") else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let lower = bounds[..and_pos].trim();
    let upper = bounds[and_pos + "AND".len()..].trim();
    if lower.is_empty() || upper.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(Some(vec![
        SelectFilter {
            column: column.clone(),
            op: SelectFilterOp::Gte,
            value: parse_sql_value(lower)?,
        },
        SelectFilter {
            column,
            op: SelectFilterOp::Lte,
            value: parse_sql_value(upper)?,
        },
    ]))
}

fn parse_select_in_filter_groups(
    input: &str,
) -> Result<Option<Vec<Vec<SelectFilter>>>, ParseError> {
    let Some(pos) = find_keyword_outside_quotes(input, "IN") else {
        return Ok(None);
    };
    let column = normalize_identifier(input[..pos].trim())?;
    let values = input[pos + "IN".len()..].trim();
    if !values.starts_with('(') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let close = find_matching_paren(values, 0).ok_or(ParseError::InvalidRelationalSql)?;
    if close != values.len() - 1 {
        return Err(ParseError::InvalidRelationalSql);
    }
    let values = split_csv(&values[1..close])?;
    let groups = values
        .into_iter()
        .map(|value| {
            Ok(vec![SelectFilter {
                column: column.clone(),
                op: SelectFilterOp::Eq,
                value: parse_sql_value(value)?,
            }])
        })
        .collect::<Result<Vec<_>, ParseError>>()?;
    Ok(Some(groups))
}

fn parse_select_like_prefix_filter(input: &str) -> Result<Option<SelectFilter>, ParseError> {
    let Some(pos) = find_keyword_outside_quotes(input, "LIKE") else {
        return Ok(None);
    };
    let column = normalize_identifier(input[..pos].trim())?;
    let pattern = parse_sql_value(input[pos + "LIKE".len()..].trim())?;
    let SqlValue::Text(pattern) = pattern else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let Some(prefix) = pattern.strip_suffix('%') else {
        return Err(ParseError::InvalidRelationalSql);
    };
    if prefix.contains('%') || prefix.contains('_') {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(Some(SelectFilter {
        column,
        op: SelectFilterOp::LikePrefix,
        value: SqlValue::Text(prefix.to_string()),
    }))
}

fn split_select_filter(input: &str) -> Result<(&str, SelectFilterOp, &str), ParseError> {
    for (token, op) in [
        ("<=", SelectFilterOp::Lte),
        (">=", SelectFilterOp::Gte),
        ("=", SelectFilterOp::Eq),
        ("<", SelectFilterOp::Lt),
        (">", SelectFilterOp::Gt),
    ] {
        if let Some((column, value)) = input.split_once(token) {
            if column.trim().is_empty() || value.trim().is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            return Ok((column, op, value));
        }
    }
    Err(ParseError::InvalidRelationalSql)
}

fn trim_wrapping_parentheses(input: &str) -> Result<&str, ParseError> {
    let mut trimmed = input.trim();
    loop {
        if !trimmed.starts_with('(') {
            return Ok(trimmed);
        }
        let close = find_matching_paren(trimmed, 0).ok_or(ParseError::InvalidRelationalSql)?;
        if close != trimmed.len() - 1 {
            return Ok(trimmed);
        }
        trimmed = trimmed[1..close].trim();
        if trimmed.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
}

fn parse_select_order(input: &str) -> Result<SelectOrder, ParseError> {
    let mut parts = input.split_whitespace();
    let column = parts
        .next()
        .ok_or(ParseError::InvalidRelationalSql)
        .and_then(normalize_identifier)?;
    let descending = match parts.next() {
        None => false,
        Some(direction) if direction.eq_ignore_ascii_case("ASC") => false,
        Some(direction) if direction.eq_ignore_ascii_case("DESC") => true,
        _ => return Err(ParseError::InvalidRelationalSql),
    };
    if parts.next().is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(SelectOrder { column, descending })
}

fn parse_sql_value(input: &str) -> Result<SqlValue, ParseError> {
    let (s, cast) = split_supported_sql_value_cast(input.trim())?;
    if s.starts_with('\'') {
        if !s.ends_with('\'') || s.len() < 2 {
            return Err(ParseError::InvalidRelationalSql);
        }
        let inner = &s[1..s.len() - 1];
        let value = inner.replace("''", "'");
        return match cast {
            None | Some(SqlType::Text) => Ok(SqlValue::Text(value)),
            Some(SqlType::Int4) => value
                .parse::<i32>()
                .map(SqlValue::Int4)
                .map_err(|_| ParseError::InvalidRelationalSql),
        };
    }
    let value = s
        .parse::<i32>()
        .map_err(|_| ParseError::InvalidRelationalSql)?;
    match cast {
        None | Some(SqlType::Int4) => Ok(SqlValue::Int4(value)),
        Some(SqlType::Text) => Ok(SqlValue::Text(value.to_string())),
    }
}

fn split_supported_sql_value_cast(input: &str) -> Result<(&str, Option<SqlType>), ParseError> {
    let Some(pos) = find_cast_operator_outside_quotes(input) else {
        return Ok((input, None));
    };
    let value = input[..pos].trim();
    let ty = input[pos + 2..].trim();
    if value.is_empty() || ty.is_empty() || find_cast_operator_outside_quotes(ty).is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let ty = if ty
        .get(.."pg_catalog.".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("pg_catalog."))
    {
        &ty["pg_catalog.".len()..]
    } else {
        ty
    };
    let cast = if ty.eq_ignore_ascii_case("int4") || ty.eq_ignore_ascii_case("integer") {
        SqlType::Int4
    } else if ty.eq_ignore_ascii_case("text") {
        SqlType::Text
    } else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok((value, Some(cast)))
}

fn find_cast_operator_outside_quotes(input: &str) -> Option<usize> {
    let bytes = input.as_bytes();
    let mut in_quote = false;
    let mut idx = 0;
    while idx + 1 < bytes.len() {
        match bytes[idx] {
            b'\'' => {
                if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                    idx += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b':' if !in_quote && bytes[idx + 1] == b':' => return Some(idx),
            _ => {}
        }
        idx += 1;
    }
    None
}

fn normalize_identifier(input: &str) -> Result<String, ParseError> {
    let s = input.trim();
    if s.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if let Some(quoted) = s.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        if quoted.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(quoted.replace("\"\"", "\""));
    }
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(ParseError::InvalidRelationalSql);
    }
    if chars.any(|ch| !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())) {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(s.to_ascii_lowercase())
}

fn normalize_relation_identifier(input: &str) -> Result<String, ParseError> {
    let s = input.trim();
    if let Some((schema, table)) = s.split_once('.') {
        if normalize_identifier(schema)? != "public" {
            return Err(ParseError::InvalidRelationalSql);
        }
        return normalize_identifier(table);
    }
    normalize_identifier(s)
}

fn split_csv(input: &str) -> Result<Vec<&str>, ParseError> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut in_quote = false;
    let bytes = input.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\'' => {
                if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                    idx += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b'(' if !in_quote => depth += 1,
            b')' if !in_quote => {
                depth = depth
                    .checked_sub(1)
                    .ok_or(ParseError::InvalidRelationalSql)?;
            }
            b',' if !in_quote && depth == 0 => {
                let part = input[start..idx].trim();
                if part.is_empty() {
                    return Err(ParseError::InvalidRelationalSql);
                }
                parts.push(part);
                start = idx + 1;
            }
            _ => {}
        }
        idx += 1;
    }
    if in_quote || depth != 0 {
        return Err(ParseError::InvalidRelationalSql);
    }
    let part = input[start..].trim();
    if part.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    parts.push(part);
    Ok(parts)
}

fn find_matching_paren(input: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_quote = false;
    let bytes = input.as_bytes();
    let mut idx = open;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\'' => {
                if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                    idx += 2;
                    continue;
                }
                in_quote = !in_quote;
            }
            b'(' if !in_quote => depth += 1,
            b')' if !in_quote => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
        idx += 1;
    }
    None
}

fn find_keyword_outside_quotes(input: &str, keyword: &str) -> Option<usize> {
    let lower = input.to_ascii_lowercase();
    let keyword = keyword.to_ascii_lowercase();
    let bytes = input.as_bytes();
    let mut in_quote = false;
    let mut depth = 0usize;
    let mut idx = 0;
    while idx + keyword.len() <= bytes.len() {
        if bytes[idx] == b'\'' {
            if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                idx += 2;
                continue;
            }
            in_quote = !in_quote;
            idx += 1;
            continue;
        }
        match bytes[idx] {
            b'(' if !in_quote => {
                depth += 1;
                idx += 1;
                continue;
            }
            b')' if !in_quote => {
                depth = depth.saturating_sub(1);
                idx += 1;
                continue;
            }
            _ => {}
        }
        if !in_quote
            && depth == 0
            && lower[idx..].starts_with(&keyword)
            && is_keyword_boundary(input, idx, keyword.len())
        {
            return Some(idx);
        }
        idx += 1;
    }
    None
}

fn is_keyword_boundary(input: &str, start: usize, len: usize) -> bool {
    let before = input[..start]
        .chars()
        .next_back()
        .is_none_or(|ch| !is_identifier_char(ch));
    let after = input[start + len..]
        .chars()
        .next()
        .is_none_or(|ch| !is_identifier_char(ch));
    before && after
}

fn next_clause_pos(input: &str) -> Option<usize> {
    ["WHERE", "GROUP", "ORDER", "LIMIT", "OFFSET"]
        .into_iter()
        .filter_map(|keyword| find_keyword_outside_quotes(input, keyword))
        .min()
}

fn split_keyword_chain_outside_quotes<'a>(
    mut input: &'a str,
    keyword: &str,
) -> Result<Vec<&'a str>, ParseError> {
    let mut parts = Vec::new();
    while let Some(pos) = find_keyword_outside_quotes(input, keyword) {
        let part = input[..pos].trim();
        if part.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        parts.push(part);
        input = input[pos + keyword.len()..].trim_start();
    }
    let tail = input.trim();
    if tail.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    parts.push(tail);
    Ok(parts)
}

fn split_select_and_chain_outside_quotes(input: &str) -> Result<Vec<&str>, ParseError> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut in_quote = false;
    let mut depth = 0usize;
    let mut skip_next_and = false;
    let bytes = input.as_bytes();
    let lower = input.to_ascii_lowercase();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                idx += 2;
                continue;
            }
            in_quote = !in_quote;
            idx += 1;
            continue;
        }
        match bytes[idx] {
            b'(' if !in_quote => {
                depth += 1;
                idx += 1;
                continue;
            }
            b')' if !in_quote => {
                depth = depth.saturating_sub(1);
                idx += 1;
                continue;
            }
            _ => {}
        }
        if !in_quote
            && depth == 0
            && lower[idx..].starts_with("between")
            && is_keyword_boundary(input, idx, "between".len())
        {
            skip_next_and = true;
            idx += "between".len();
            continue;
        }
        if !in_quote
            && depth == 0
            && lower[idx..].starts_with("and")
            && is_keyword_boundary(input, idx, "and".len())
        {
            if skip_next_and {
                skip_next_and = false;
                idx += "and".len();
                continue;
            }
            let part = input[start..idx].trim();
            if part.is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            parts.push(part);
            idx += "and".len();
            start = idx;
            continue;
        }
        idx += 1;
    }

    let tail = input[start..].trim();
    if tail.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    parts.push(tail);
    Ok(parts)
}

fn is_identifier_char(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()
}

fn parse_set_session_command(rest: &str) -> Option<Result<Command, ParseError>> {
    if let Some(after_local) = strip_keyword_prefix_case_insensitive(rest, "LOCAL") {
        let after_local = after_local.trim_start();
        if let Some(after_role) = strip_keyword_prefix_case_insensitive(after_local, "ROLE") {
            let tail = after_role.trim_start();
            return Some(
                if tail.eq_ignore_ascii_case("NONE")
                    || tail.eq_ignore_ascii_case("DEFAULT")
                    || parse_reset_identifier(tail)
                        .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }

        if let Some(after_transaction) =
            strip_keyword_prefix_case_insensitive(after_local, "TRANSACTION")
        {
            let tail = after_transaction.trim_start();
            return Some(
                if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }
    }

    if let Some(after_session) = strip_keyword_prefix_case_insensitive(rest, "SESSION") {
        let after_session = after_session.trim_start();
        if let Some(after_role) = strip_keyword_prefix_case_insensitive(after_session, "ROLE") {
            let tail = after_role.trim_start();
            return Some(
                if tail.eq_ignore_ascii_case("NONE")
                    || tail.eq_ignore_ascii_case("DEFAULT")
                    || parse_reset_identifier(tail)
                        .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }
    }

    if let Some(after_role) = strip_keyword_prefix_case_insensitive(rest, "ROLE") {
        let tail = after_role.trim_start();
        return Some(
            if tail.eq_ignore_ascii_case("NONE")
                || tail.eq_ignore_ascii_case("DEFAULT")
                || parse_reset_identifier(tail)
                    .is_some_and(|(_, trailing)| trailing.trim().is_empty())
            {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidSet)
            },
        );
    }

    if let Some(after_transaction) = strip_keyword_prefix_case_insensitive(rest, "TRANSACTION") {
        let tail = after_transaction.trim_start();
        return Some(
            if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidSet)
            },
        );
    }

    if let Some(after_session) = strip_keyword_prefix_case_insensitive(rest, "SESSION") {
        let after_session = after_session.trim_start();

        if let Some(after_authorization) =
            strip_keyword_prefix_case_insensitive(after_session, "AUTHORIZATION")
        {
            let tail = after_authorization.trim_start();
            return Some(
                if parse_reset_identifier(tail)
                    .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }

        if let Some(after_auth) = strip_keyword_prefix_case_insensitive(after_session, "AUTH") {
            let tail = after_auth.trim_start();
            return Some(
                if parse_reset_identifier(tail)
                    .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }

        if let Some(after_characteristics) =
            strip_keyword_prefix_case_insensitive(after_session, "CHARACTERISTICS")
        {
            let after_characteristics = after_characteristics.trim_start();
            if let Some(after_as) =
                strip_keyword_prefix_case_insensitive(after_characteristics, "AS")
            {
                let after_as = after_as.trim_start();
                if let Some(after_transaction) =
                    strip_keyword_prefix_case_insensitive(after_as, "TRANSACTION")
                {
                    let tail = after_transaction.trim_start();
                    return Some(
                        if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                            Ok(Command::ResetAll)
                        } else {
                            Err(ParseError::InvalidSet)
                        },
                    );
                }
            }
            return Some(Err(ParseError::InvalidSet));
        }

        return None;
    }

    None
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
    if let Some(relational) = parse_relational_command(s) {
        return relational;
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

            let assignment_rest = strip_set_scope_prefix(rest, "LOCAL")
                .or_else(|| strip_set_scope_prefix(rest, "SESSION"))
                .unwrap_or(rest);
            let Some((k, v)) = split_set_key_value(assignment_rest) else {
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
    fn parses_set_with_session_or_local_scope_aliases() {
        let cmd = parse_command("SET LOCAL a = 42").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "a".into(),
                value: "42".into()
            }
        );

        let cmd = parse_command("SET SESSION a TO 42").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "a".into(),
                value: "42".into()
            }
        );

        let cmd = parse_command("SET SESSION statement_timeout = 5s").unwrap();
        assert_eq!(
            cmd,
            Command::SetKv {
                key: "statement_timeout".into(),
                value: "5s".into()
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
            parse_command("SET ROLE app_role").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET ROLE \"app role\"").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET ROLE \"\"\"quoted\"\" role\"").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION ROLE DEFAULT").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION ROLE \"app role\"").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET LOCAL ROLE NONE").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET LOCAL ROLE app_role").unwrap(),
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
            parse_command("SET SESSION AUTHORIZATION \"app user\"").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION AUTH \"app user\"").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command(
                "SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED"
            )
            .unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET TRANSACTION READ ONLY, DEFERRABLE").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE, NOT DEFERRABLE")
                .unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET LOCAL TRANSACTION READ ONLY").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("SET LOCAL TRANSACTION READ WRITE, DEFERRABLE").unwrap(),
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

        let cmd = parse_command("RESET SESSION ROLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET LOCAL ROLE").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET AUTHORIZATION").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET AUTH").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTHORIZATION").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTHORIZATION DEFAULT").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTHORIZATION TO DEFAULT").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTH").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTH DEFAULT").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("RESET SESSION AUTH TO DEFAULT").unwrap();
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

        let cmd = parse_command("CLOSE \"cursor\"\"name\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("CLOSE \"\"\"quoted\"\" cursor\"").unwrap();
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

        let cmd = parse_command("UNLISTEN \"updates\"\"channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("UNLISTEN \"updates\nΔetail\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("LISTEN updates_channel").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("LISTEN \"updates channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("LISTEN \"updates\"\"channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("LISTEN \"updates\nΔetail\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY \"updates channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY \"updates\"\"channel\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY \"updates\nΔetail\", 'héllo\nΔetail'").unwrap();
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

        let cmd = parse_command("NOTIFY updates_channel, \"hello,world\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, \"hello\"\"world\"").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, E'hello\\'world'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, B'101010'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, X'CAFE'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, U&'d\\0061ta'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, e'hello\\'world'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, b'101010'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, x'cafe'").unwrap();
        assert_eq!(cmd, Command::ResetAll);

        let cmd = parse_command("NOTIFY updates_channel, u&'d\\0061ta'").unwrap();
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
            parse_command("COMMIT WORK AND CHAIN").unwrap(),
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
            parse_command("END WORK AND CHAIN").unwrap(),
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
            parse_command("ROLLBACK WORK AND CHAIN").unwrap(),
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
            parse_command("ABORT WORK AND CHAIN").unwrap(),
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
            parse_command("RESET SESSION AUTHORIZATION DEFAULT NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("RESET SESSION AUTHORIZATION TO DEFAULT NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("RESET SESSION AUTH DEFAULT NOW"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("RESET SESSION AUTH TO DEFAULT NOW"),
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
            parse_command("DEALLOCATE \"\""),
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
            parse_command("CLOSE \"\""),
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
            parse_command("UNLISTEN \"\""),
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
            parse_command("LISTEN \"\""),
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
            parse_command("NOTIFY updates_channel, payload extra"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, 'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, \"unterminated"),
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
            parse_command("NOTIFY updates_channel, E'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, B'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, X'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, U&'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, e'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, b'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, x'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY updates_channel, u&'unterminated"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY ;"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("LISTEN ;"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("UNLISTEN @invalid"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY \"updates channel"),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("NOTIFY \"\""),
            Err(ParseError::InvalidReset)
        ));
        assert!(matches!(
            parse_command("SET ROLE"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET SESSION ROLE"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET LOCAL ROLE"),
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
            parse_command("SET ROLE \"unterminated"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET ROLE \"\""),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET SESSION AUTHORIZATION \"unterminated"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET TRANSACTION"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET TRANSACTION NOW"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET TRANSACTION READ ONLY, READ WRITE"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET LOCAL TRANSACTION"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET LOCAL TRANSACTION NOW"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION"),
            Err(ParseError::InvalidSet)
        ));
        assert!(matches!(
            parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION NOW"),
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
            parse_command("RESET SESSION ROLE;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("RESET LOCAL ROLE;\n").unwrap(),
            Command::ResetAll
        );
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
            parse_command("RESET SESSION AUTHORIZATION DEFAULT;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("RESET SESSION AUTHORIZATION TO DEFAULT;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("RESET SESSION AUTH;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("RESET SESSION AUTH DEFAULT;\n").unwrap(),
            Command::ResetAll
        );
        assert_eq!(
            parse_command("RESET SESSION AUTH TO DEFAULT;\n").unwrap(),
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
            parse_command("SET TRANSACTION READ ONLY;\n").unwrap(),
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
        payload.extend_from_slice(b"user\0postgres\0database\0gpu\0application_name\0\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("user".to_string(), "postgres".to_string()),
                    ("database".to_string(), "gpu".to_string()),
                    ("application_name".to_string(), "".to_string())
                ],
            }
        );
    }

    #[test]
    fn parses_ssl_gssenc_and_cancel_requests() {
        let ssl = with_length_prefix(PG_SSL_REQUEST_CODE.to_be_bytes().to_vec());
        assert_eq!(
            parse_startup_packet(&ssl).unwrap(),
            StartupPacket::SslRequest
        );

        let gssenc = with_length_prefix(PG_GSSENC_REQUEST_CODE.to_be_bytes().to_vec());
        assert_eq!(
            parse_startup_packet(&gssenc).unwrap(),
            StartupPacket::GssEncRequest
        );

        let mut cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
        cancel_payload.extend_from_slice(&123u32.to_be_bytes());
        cancel_payload.extend_from_slice(&456u32.to_be_bytes());
        let cancel = with_length_prefix(cancel_payload);
        assert_eq!(
            parse_startup_packet(&cancel).unwrap(),
            StartupPacket::CancelRequest {
                process_id: 123,
                secret_key: 456u32.to_be_bytes().to_vec(),
            }
        );

        let mut extended_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
        extended_cancel_payload.extend_from_slice(&321u32.to_be_bytes());
        extended_cancel_payload.extend_from_slice(b"longer-secret-key");
        let extended_cancel = with_length_prefix(extended_cancel_payload);
        assert_eq!(
            parse_startup_packet(&extended_cancel).unwrap(),
            StartupPacket::CancelRequest {
                process_id: 321,
                secret_key: b"longer-secret-key".to_vec(),
            }
        );

        let mut max_length_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
        max_length_cancel_payload.extend_from_slice(&777u32.to_be_bytes());
        max_length_cancel_payload.extend_from_slice(&vec![0xCD; PG_CANCEL_SECRET_KEY_MAX_BYTES]);
        let max_length_cancel = with_length_prefix(max_length_cancel_payload);
        assert_eq!(
            parse_startup_packet(&max_length_cancel).unwrap(),
            StartupPacket::CancelRequest {
                process_id: 777,
                secret_key: vec![0xCD; PG_CANCEL_SECRET_KEY_MAX_BYTES],
            }
        );

        let mut zero_process_id_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
        zero_process_id_cancel_payload.extend_from_slice(&0u32.to_be_bytes());
        zero_process_id_cancel_payload.extend_from_slice(b"zero-key");
        let zero_process_id_cancel = with_length_prefix(zero_process_id_cancel_payload);
        assert_eq!(
            parse_startup_packet(&zero_process_id_cancel).unwrap(),
            StartupPacket::CancelRequest {
                process_id: 0,
                secret_key: b"zero-key".to_vec(),
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_empty_params() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.push(0);
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_extra_trailing_terminator() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice(b"user\0postgres\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![("user".to_string(), "postgres".to_string())],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_empty_parameter_value() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice(b"options\0\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![("options".to_string(), String::new())],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_multiple_params() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice(b"user\0postgres\0database\0analytics\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("user".to_string(), "postgres".to_string()),
                    ("database".to_string(), "analytics".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_duplicate_keys() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice(b"options\0-a\0options\0-b\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("options".to_string(), "-a".to_string()),
                    ("options".to_string(), "-b".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_mixed_empty_and_non_empty_values() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice(b"application_name\0gpu-db\0options\0\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("application_name".to_string(), "gpu-db".to_string()),
                    ("options".to_string(), String::new()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_utf8_params() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("cliënt\0möde\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![("cliënt".to_string(), "möde".to_string())],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_utf8_duplicate_keys() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0påth=a\0möde\0påth=b\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), "påth=a".to_string()),
                    ("möde".to_string(), "påth=b".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_utf8_empty_value() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![("möde".to_string(), String::new())],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_multiple_utf8_params() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("cliënt\0möde\0rôle\0anályst\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("cliënt".to_string(), "möde".to_string()),
                    ("rôle".to_string(), "anályst".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_utf8_duplicate_keys_and_empty_value() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0påth=a\0möde\0\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), "påth=a".to_string()),
                    ("möde".to_string(), String::new()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_utf8_empty_value_before_utf8_param() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0\0rôle\0anályst\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), String::new()),
                    ("rôle".to_string(), "anályst".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_multiple_utf8_params_and_middle_empty_value() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("cliënt\0möde\0rôle\0\0tëam\0dév\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("cliënt".to_string(), "möde".to_string()),
                    ("rôle".to_string(), String::new()),
                    ("tëam".to_string(), "dév".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_multiple_utf8_params_and_trailing_empty_value() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("cliënt\0möde\0tëam\0dév\0rôle\0\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("cliënt".to_string(), "möde".to_string()),
                    ("tëam".to_string(), "dév".to_string()),
                    ("rôle".to_string(), String::new()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_multiple_utf8_empty_values() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0\0rôle\0\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), String::new()),
                    ("rôle".to_string(), String::new()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_utf8_duplicate_keys_and_multiple_empty_values() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0\0möde\0\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), String::new()),
                    ("möde".to_string(), String::new()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_three_utf8_duplicate_keys() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0påth=a\0möde\0\0möde\0påth=c\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), "påth=a".to_string()),
                    ("möde".to_string(), String::new()),
                    ("möde".to_string(), "påth=c".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_interleaved_utf8_duplicate_keys() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0påth=a\0rôle\0anályst\0möde\0påth=b\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), "påth=a".to_string()),
                    ("rôle".to_string(), "anályst".to_string()),
                    ("möde".to_string(), "påth=b".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_interleaved_utf8_duplicate_keys_and_empty_value() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0påth=a\0rôle\0anályst\0möde\0\0tëam\0dév\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), "påth=a".to_string()),
                    ("rôle".to_string(), "anályst".to_string()),
                    ("möde".to_string(), String::new()),
                    ("tëam".to_string(), "dév".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_interleaved_utf8_duplicate_keys_and_both_empty_values() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0\0rôle\0anályst\0möde\0\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), String::new()),
                    ("rôle".to_string(), "anályst".to_string()),
                    ("möde".to_string(), String::new()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_startup_packet_with_interleaved_utf8_empty_duplicate_before_value() {
        let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        payload.extend_from_slice("möde\0\0rôle\0anályst\0möde\0påth=b\0\0".as_bytes());
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version: PG_PROTOCOL_V3,
                params: vec![
                    ("möde".to_string(), String::new()),
                    ("rôle".to_string(), "anályst".to_string()),
                    ("möde".to_string(), "påth=b".to_string()),
                ],
            }
        );
    }

    #[test]
    fn parses_pg_v3_minor_version_startup_packet() {
        let protocol_version = (PG_PROTOCOL_MAJOR_V3 << 16) | 2;
        let mut payload = protocol_version.to_be_bytes().to_vec();
        payload.extend_from_slice(b"user\0postgres\0\0");
        let frame = with_length_prefix(payload);

        let packet = parse_startup_packet(&frame).unwrap();
        assert_eq!(
            packet,
            StartupPacket::Startup {
                protocol_version,
                params: vec![("user".to_string(), "postgres".to_string())],
            }
        );
    }

    #[test]
    fn rejects_ssl_gssenc_and_cancel_requests_with_invalid_lengths() {
        let mut ssl_payload = PG_SSL_REQUEST_CODE.to_be_bytes().to_vec();
        ssl_payload.push(0);
        let ssl = with_length_prefix(ssl_payload);
        assert_eq!(
            parse_startup_packet(&ssl).unwrap_err(),
            StartupPacketError::LengthMismatch {
                expected: 8,
                actual: 9,
            }
        );

        let mut gssenc_payload = PG_GSSENC_REQUEST_CODE.to_be_bytes().to_vec();
        gssenc_payload.push(0);
        let gssenc = with_length_prefix(gssenc_payload);
        assert_eq!(
            parse_startup_packet(&gssenc).unwrap_err(),
            StartupPacketError::LengthMismatch {
                expected: 8,
                actual: 9,
            }
        );

        let cancel = with_length_prefix(PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec());
        assert_eq!(
            parse_startup_packet(&cancel).unwrap_err(),
            StartupPacketError::LengthMismatch {
                expected: 16,
                actual: 8,
            }
        );

        let mut oversized_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
        oversized_cancel_payload.extend_from_slice(&123u32.to_be_bytes());
        oversized_cancel_payload.extend_from_slice(&vec![0xAB; PG_CANCEL_SECRET_KEY_MAX_BYTES + 1]);
        let oversized_cancel = with_length_prefix(oversized_cancel_payload);
        assert_eq!(
            parse_startup_packet(&oversized_cancel).unwrap_err(),
            StartupPacketError::InvalidCancelKeyLength {
                actual: PG_CANCEL_SECRET_KEY_MAX_BYTES + 1,
                max: PG_CANCEL_SECRET_KEY_MAX_BYTES,
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

        let invalid_length_field = vec![0, 0, 0, 4, 0, 3, 0, 0];
        assert_eq!(
            parse_startup_packet(&invalid_length_field).unwrap_err(),
            StartupPacketError::InvalidLengthField { declared: 4 }
        );

        let unsupported_protocol_code = vec![0, 0, 0, 8, 0x12, 0x34, 0x56, 0x78];
        assert_eq!(
            parse_startup_packet(&unsupported_protocol_code).unwrap_err(),
            StartupPacketError::UnsupportedProtocolCode(0x1234_5678)
        );

        let empty_param_payload = with_length_prefix(PG_PROTOCOL_V3.to_be_bytes().to_vec());
        assert_eq!(
            parse_startup_packet(&empty_param_payload).unwrap_err(),
            StartupPacketError::UnterminatedParameterPayload
        );

        let mut double_null_empty_param_payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        double_null_empty_param_payload.extend_from_slice(b"\0\0");
        let double_null_empty_param_payload = with_length_prefix(double_null_empty_param_payload);
        assert_eq!(
            parse_startup_packet(&double_null_empty_param_payload).unwrap_err(),
            StartupPacketError::InvalidParameterPairing
        );

        let mut bad_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        bad_params.extend_from_slice(b"user\0postgres");
        let bad_params = with_length_prefix(bad_params);
        assert_eq!(
            parse_startup_packet(&bad_params).unwrap_err(),
            StartupPacketError::UnterminatedParameterPayload
        );

        let mut extra_trailing_terminator_payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        extra_trailing_terminator_payload.extend_from_slice(b"user\0postgres\0\0\0");
        let extra_trailing_terminator_payload =
            with_length_prefix(extra_trailing_terminator_payload);
        assert_eq!(
            parse_startup_packet(&extra_trailing_terminator_payload).unwrap_err(),
            StartupPacketError::InvalidParameterPairing
        );

        let mut invalid_utf8_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        invalid_utf8_params.extend_from_slice(&[0xFF, 0, b'v', 0, 0]);
        let invalid_utf8_params = with_length_prefix(invalid_utf8_params);
        assert_eq!(
            parse_startup_packet(&invalid_utf8_params).unwrap_err(),
            StartupPacketError::InvalidUtf8
        );

        let mut invalid_utf8_value_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        invalid_utf8_value_params.extend_from_slice(b"user\0");
        invalid_utf8_value_params.extend_from_slice(&[0xFF, 0, 0]);
        let invalid_utf8_value_params = with_length_prefix(invalid_utf8_value_params);
        assert_eq!(
            parse_startup_packet(&invalid_utf8_value_params).unwrap_err(),
            StartupPacketError::InvalidUtf8
        );

        let mut empty_key_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        empty_key_params.extend_from_slice(b"\0value\0\0");
        let empty_key_params = with_length_prefix(empty_key_params);
        assert_eq!(
            parse_startup_packet(&empty_key_params).unwrap_err(),
            StartupPacketError::InvalidParameterPairing
        );

        let mut dangling_key_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
        dangling_key_params.extend_from_slice(b"user\0postgres\0database\0\0");
        let dangling_key_params = with_length_prefix(dangling_key_params);
        assert_eq!(
            parse_startup_packet(&dangling_key_params).unwrap_err(),
            StartupPacketError::InvalidParameterPairing
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
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                    },
                ],
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
                    },
                    ColumnDef {
                        name: "owner".to_string(),
                        ty: SqlType::Text,
                    },
                ],
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
                    },
                    ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                    },
                ],
            })
        );
        assert!(matches!(
            parse_command("CREATE TABLE private.dump_people (id integer)"),
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
                order_by: None,
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
                order_by: None,
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
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: None,
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
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: Some(SelectOrder {
                    column: "name".to_string(),
                    descending: true,
                }),
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: None,
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
                order_by: None,
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
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }),
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
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "name".to_string(),
                    descending: true,
                }),
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
                "SELECT name, COUNT(*) FROM people WHERE id >= 2 GROUP BY name ORDER BY count DESC LIMIT 2 OFFSET 1",
            )
            .unwrap(),
            Command::Select(Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::GroupedCount {
                    column: "name".to_string(),
                },
                group_by: Some("name".to_string()),
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
                order_by: Some(SelectOrder {
                    column: "count".to_string(),
                    descending: true,
                }),
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
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "sum".to_string(),
                    descending: true,
                }),
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
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "avg".to_string(),
                    descending: true,
                }),
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
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: None,
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
                order_by: Some(SelectOrder {
                    column: "min".to_string(),
                    descending: true,
                }),
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
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: None,
                limit: None,
                offset: None,
            })
        );

        assert!(matches!(
            parse_command("SELECT MIN(id), name FROM people"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }
}
