//! SQL `COPY` statement parsing and row decoding.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyFormat {
    Text,
    Csv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyOptions {
    pub format: CopyFormat,
    pub header: bool,
    pub delimiter: char,
    pub quote: char,
    pub escape: char,
}

impl CopyOptions {
    pub const TEXT: Self = Self {
        format: CopyFormat::Text,
        header: false,
        delimiter: '\t',
        quote: '"',
        escape: '"',
    };

    pub const CSV: Self = Self {
        format: CopyFormat::Csv,
        header: false,
        delimiter: ',',
        quote: '"',
        escape: '"',
    };

    pub const CSV_HEADER: Self = Self {
        format: CopyFormat::Csv,
        header: true,
        delimiter: ',',
        quote: '"',
        escape: '"',
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyFromStdin {
    pub table: String,
    pub columns: Option<Vec<String>>,
    pub options: CopyOptions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyToStdout {
    pub table: String,
    pub options: CopyOptions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyColumn {
    pub name: String,
    pub ty: SqlType,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CopyParseError {
    #[error("COPY data contains invalid UTF-8")]
    InvalidUtf8,
    #[error("COPY NULL values are not supported by the compatibility endpoint")]
    NullNotSupported,
    #[error("invalid input syntax for type smallint")]
    InvalidInt2,
    #[error("invalid input syntax for type integer")]
    InvalidInt4,
    #[error("invalid input syntax for type bigint")]
    InvalidInt8,
    #[error("invalid input syntax for type numeric")]
    InvalidNumeric,
    #[error("invalid input syntax for type boolean")]
    InvalidBool,
    #[error("invalid input syntax for type date")]
    InvalidDate,
    #[error("invalid input syntax for type timestamp")]
    InvalidTimestamp,
    #[error("invalid input syntax for type uuid")]
    InvalidUuid,
    #[error("unterminated COPY escape sequence")]
    UnterminatedEscape,
    #[error("malformed CSV quoted field")]
    MalformedCsvQuotedField,
    #[error("unterminated CSV quoted field")]
    UnterminatedCsvQuotedField,
    #[error("COPY row has wrong number of columns")]
    WrongColumnCount,
    #[error("column does not exist")]
    ColumnDoesNotExist,
}

impl CopyParseError {
    pub const fn postgres_code(&self) -> &'static str {
        match self {
            Self::InvalidUtf8 => "22021",
            Self::NullNotSupported => "0A000",
            Self::InvalidInt2
            | Self::InvalidInt4
            | Self::InvalidInt8
            | Self::InvalidNumeric
            | Self::InvalidBool
            | Self::InvalidDate
            | Self::InvalidTimestamp
            | Self::InvalidUuid => "22P02",
            Self::UnterminatedEscape
            | Self::MalformedCsvQuotedField
            | Self::UnterminatedCsvQuotedField
            | Self::WrongColumnCount => "22P04",
            Self::ColumnDoesNotExist => "42703",
        }
    }

    pub const fn postgres_message(&self) -> &'static str {
        match self {
            Self::InvalidUtf8 => "COPY data contains invalid UTF-8",
            Self::NullNotSupported => {
                "COPY NULL values are not supported by the compatibility endpoint"
            }
            Self::InvalidInt2 => "invalid input syntax for type smallint",
            Self::InvalidInt4 => "invalid input syntax for type integer",
            Self::InvalidInt8 => "invalid input syntax for type bigint",
            Self::InvalidNumeric => "invalid input syntax for type numeric",
            Self::InvalidBool => "invalid input syntax for type boolean",
            Self::InvalidDate => "invalid input syntax for type date",
            Self::InvalidTimestamp => "invalid input syntax for type timestamp",
            Self::InvalidUuid => "invalid input syntax for type uuid",
            Self::UnterminatedEscape => "unterminated COPY escape sequence",
            Self::MalformedCsvQuotedField => "malformed CSV quoted field",
            Self::UnterminatedCsvQuotedField => "unterminated CSV quoted field",
            Self::WrongColumnCount => "COPY row has wrong number of columns",
            Self::ColumnDoesNotExist => "column does not exist",
        }
    }
}

pub fn is_copy_statement(statement: &str) -> bool {
    strip_leading_sql_comments(statement.trim())
        .is_some_and(|statement| canonical_copy_sql(statement).starts_with("copy "))
}

pub fn parse_copy_to_stdout_table(statement: &str) -> Option<CopyToStdout> {
    let statement = strip_leading_sql_comments(statement.trim())?;
    let canonical = canonical_copy_sql(statement);
    let target = canonical.strip_prefix("copy ")?.trim();
    let (target, options) = parse_copy_target_and_options(target, "to stdout")?;
    let table = if let Some(open) = target.find('(') {
        let close = target.rfind(')')?;
        if close <= open || !target[close + 1..].trim().is_empty() {
            return None;
        }
        let table = target[..open].trim();
        let columns = target[open + 1..close]
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        if columns.is_empty()
            || columns
                .iter()
                .any(|column| !is_simple_copy_identifier(column))
        {
            return None;
        }
        table
    } else {
        target
    };
    if table.is_empty()
        || table
            .contains(|ch: char| ch.is_whitespace() || matches!(ch, '(' | ')' | ',' | '\'' | '"'))
    {
        return None;
    }
    Some(CopyToStdout {
        table: table.strip_prefix("public.").unwrap_or(table).to_string(),
        options,
    })
}

pub fn parse_copy_from_stdin(statement: &str) -> Option<CopyFromStdin> {
    let statement = strip_leading_sql_comments(statement.trim())?;
    let canonical = canonical_copy_sql(statement);
    let target = canonical.strip_prefix("copy ")?.trim();
    let (target, options) = parse_copy_target_and_options(target, "from stdin")?;
    let (table, columns) = if let Some(open) = target.find('(') {
        let close = target.rfind(')')?;
        if close <= open || !target[close + 1..].trim().is_empty() {
            return None;
        }
        let table = target[..open].trim();
        let columns = target[open + 1..close]
            .split(',')
            .map(str::trim)
            .map(str::to_string)
            .collect::<Vec<_>>();
        if columns.is_empty()
            || columns
                .iter()
                .any(|column| !is_simple_copy_identifier(column))
        {
            return None;
        }
        (table, Some(columns))
    } else {
        (target, None)
    };
    if !is_simple_copy_table_name(table) {
        return None;
    }
    Some(CopyFromStdin {
        table: table.strip_prefix("public.").unwrap_or(table).to_string(),
        columns,
        options,
    })
}

pub fn is_supported_extended_copy(query: &str) -> bool {
    parse_copy_to_stdout_table(query).is_some() || parse_copy_from_stdin(query).is_some()
}

pub fn parse_copy_row(
    table_columns: &[CopyColumn],
    columns: &[String],
    options: CopyOptions,
    line: &str,
) -> Result<Vec<SqlValue>, CopyParseError> {
    let mut row = Vec::with_capacity(columns.len());
    match options.format {
        CopyFormat::Text => {
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != columns.len() {
                return Err(CopyParseError::WrongColumnCount);
            }
            for (field, column_name) in fields.iter().zip(columns.iter()) {
                let column = table_columns
                    .iter()
                    .find(|candidate| candidate.name == *column_name)
                    .ok_or(CopyParseError::ColumnDoesNotExist)?;
                row.push(parse_copy_text_value(field, column.ty)?);
            }
        }
        CopyFormat::Csv => {
            let fields =
                parse_copy_csv_row(line, options.delimiter, options.quote, options.escape)?;
            if fields.len() != columns.len() {
                return Err(CopyParseError::WrongColumnCount);
            }
            for (field, column_name) in fields.iter().zip(columns.iter()) {
                let column = table_columns
                    .iter()
                    .find(|candidate| candidate.name == *column_name)
                    .ok_or(CopyParseError::ColumnDoesNotExist)?;
                row.push(parse_copy_csv_value(field, column.ty)?);
            }
        }
    }
    Ok(row)
}

fn parse_copy_target_and_options<'a>(
    target: &'a str,
    direction: &str,
) -> Option<(&'a str, CopyOptions)> {
    let marker = format!(" {direction}");
    let direction_idx = target.rfind(&marker)?;
    let table = target[..direction_idx].trim();
    let after_direction = target[direction_idx + marker.len()..].trim();
    if after_direction.is_empty() {
        return Some((table, CopyOptions::TEXT));
    }
    let options = after_direction.strip_prefix("with ")?.trim();
    parse_copy_options(options).map(|options| (table, options))
}

fn parse_copy_options(options: &str) -> Option<CopyOptions> {
    if options == "csv" {
        return Some(CopyOptions::CSV);
    }
    if options == "csv header" {
        return Some(CopyOptions::CSV_HEADER);
    }

    let parenthesized = parenthesized_list(options)?;
    let mut format = None;
    let mut header = false;
    let mut delimiter = ',';
    let mut quote = '"';
    let mut escape = '"';
    let mut quote_set = false;
    let mut escape_set = false;

    for part in split_copy_sql_csv(parenthesized)? {
        let part = part.trim();
        let normalized = canonical_copy_sql(part).replace(" = ", " ");
        if normalized == "format csv" {
            format = Some(CopyFormat::Csv);
        } else if normalized == "header" || normalized == "header true" || normalized == "header on"
        {
            header = true;
        } else if normalized == "header false" || normalized == "header off" {
            header = false;
        } else if canonical_copy_sql(part)
            .strip_prefix("delimiter ")
            .is_some()
        {
            let rest_start = sql_keyword_rest_start(part, "delimiter")?;
            let raw_value = part[rest_start..]
                .trim()
                .strip_prefix('=')
                .unwrap_or(part[rest_start..].trim())
                .trim();
            let decoded = decode_sql_copy_string_literal(raw_value)?;
            let mut chars = decoded.chars();
            delimiter = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            if matches!(delimiter, '"' | '\n' | '\r') {
                return None;
            }
        } else if canonical_copy_sql(part).strip_prefix("quote ").is_some() {
            let rest_start = sql_keyword_rest_start(part, "quote")?;
            let raw_value = part[rest_start..]
                .trim()
                .strip_prefix('=')
                .unwrap_or(part[rest_start..].trim())
                .trim();
            quote = decode_single_copy_option_char(raw_value)?;
            quote_set = true;
        } else if canonical_copy_sql(part).strip_prefix("escape ").is_some() {
            let rest_start = sql_keyword_rest_start(part, "escape")?;
            let raw_value = part[rest_start..]
                .trim()
                .strip_prefix('=')
                .unwrap_or(part[rest_start..].trim())
                .trim();
            escape = decode_single_copy_option_char(raw_value)?;
            escape_set = true;
        } else {
            return None;
        }
    }

    if quote_set && !escape_set {
        escape = quote;
    }
    if delimiter == quote || delimiter == escape {
        return None;
    }

    match format {
        Some(CopyFormat::Csv) => Some(CopyOptions {
            format: CopyFormat::Csv,
            header,
            delimiter,
            quote,
            escape,
        }),
        _ => None,
    }
}

fn decode_single_copy_option_char(raw_value: &str) -> Option<char> {
    let decoded = decode_sql_copy_string_literal(raw_value)?;
    let mut chars = decoded.chars();
    let value = chars.next()?;
    if chars.next().is_some() || matches!(value, '\n' | '\r') {
        return None;
    }
    Some(value)
}

fn parse_copy_text_value(input: &str, ty: SqlType) -> Result<SqlValue, CopyParseError> {
    // COPY TEXT format: the default NULL marker is the unquoted `\N` (M3 — doc 21). It ingests as a SQL
    // NULL for any column type (a literal backslash-N text value arrives escaped as `\\N`, decoded below).
    if input == r"\N" {
        return Ok(SqlValue::Null);
    }
    let text = decode_copy_text(input)?;
    parse_copy_typed_value(&text, ty)
}

/// Parse an already-unescaped COPY field into the column's type. Shared by the text
/// and CSV copy paths so the typed-column vocabulary (int4/int8/numeric/bool/text)
/// is decoded identically.
fn parse_copy_typed_value(text: &str, ty: SqlType) -> Result<SqlValue, CopyParseError> {
    match ty {
        SqlType::Int2 => text
            .parse::<i16>()
            .map(SqlValue::Int2)
            .map_err(|_| CopyParseError::InvalidInt2),
        SqlType::Int4 => text
            .parse::<i32>()
            .map(SqlValue::Int4)
            .map_err(|_| CopyParseError::InvalidInt4),
        SqlType::Int8 => text
            .parse::<i64>()
            .map(SqlValue::Int8)
            .map_err(|_| CopyParseError::InvalidInt8),
        SqlType::Numeric { scale, .. } => Decimal128::parse_at_scale(text, scale)
            .map(SqlValue::Numeric)
            .ok_or(CopyParseError::InvalidNumeric),
        SqlType::Bool => parse_bool_value(text)
            .map(SqlValue::Bool)
            .ok_or(CopyParseError::InvalidBool),
        SqlType::Text => Ok(SqlValue::Text(text.to_string())),
        SqlType::Date => crate::datetime::parse_date(text)
            .map(SqlValue::Date)
            .ok_or(CopyParseError::InvalidDate),
        SqlType::Timestamp => crate::datetime::parse_timestamp(text)
            .map(SqlValue::Timestamp)
            .ok_or(CopyParseError::InvalidTimestamp),
        SqlType::Uuid => crate::uuid::parse_uuid(text)
            .map(SqlValue::Uuid)
            .ok_or(CopyParseError::InvalidUuid),
    }
}

fn decode_copy_text(input: &str) -> Result<String, CopyParseError> {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }
        let Some(escaped) = chars.next() else {
            return Err(CopyParseError::UnterminatedEscape);
        };
        match escaped {
            '\\' => output.push('\\'),
            't' => output.push('\t'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            other => {
                output.push('\\');
                output.push(other);
            }
        }
    }
    Ok(output)
}

#[derive(Debug, PartialEq, Eq)]
struct CopyCsvField {
    text: String,
    quoted: bool,
}

fn parse_copy_csv_row(
    line: &str,
    delimiter: char,
    quote: char,
    escape: char,
) -> Result<Vec<CopyCsvField>, CopyParseError> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    let mut in_quotes = false;
    let mut after_quote = false;

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == escape
                && matches!(chars.peek(), Some(next) if *next == quote || *next == escape)
            {
                if let Some(escaped) = chars.next() {
                    field.push(escaped);
                }
            } else if ch == quote {
                if quote == escape && matches!(chars.peek(), Some(next) if *next == quote) {
                    chars.next();
                    field.push(quote);
                } else {
                    in_quotes = false;
                    after_quote = true;
                }
            } else {
                field.push(ch);
            }
            continue;
        }

        match ch {
            _ if ch == quote && field.is_empty() && !after_quote => {
                quoted = true;
                in_quotes = true;
            }
            _ if ch == delimiter => {
                fields.push(CopyCsvField {
                    text: std::mem::take(&mut field),
                    quoted,
                });
                // Reset BOTH per-field flags: `quoted` must reflect only THIS field, else an unquoted
                // empty field following a quoted one is mis-classified as a quoted empty string (Text(""))
                // instead of the NULL marker (M3 — doc 21). `after_quote` gates the post-close error.
                quoted = false;
                after_quote = false;
            }
            _ if after_quote => return Err(CopyParseError::MalformedCsvQuotedField),
            _ => field.push(ch),
        }
    }

    if in_quotes {
        return Err(CopyParseError::UnterminatedCsvQuotedField);
    }
    fields.push(CopyCsvField {
        text: field,
        quoted,
    });
    Ok(fields)
}

fn parse_copy_csv_value(field: &CopyCsvField, ty: SqlType) -> Result<SqlValue, CopyParseError> {
    // COPY CSV format: an UNQUOTED empty field is the default NULL marker and ingests as a SQL NULL (M3 —
    // doc 21). A QUOTED empty field (`""`) is the empty STRING, not NULL — so the `!quoted` guard matters.
    if !field.quoted && field.text.is_empty() {
        return Ok(SqlValue::Null);
    }
    parse_copy_typed_value(&field.text, ty)
}

fn is_simple_copy_table_name(table: &str) -> bool {
    !table.is_empty()
        && table
            .split('.')
            .all(|part| is_simple_copy_identifier(part) && !part.is_empty())
}

fn is_simple_copy_identifier(identifier: &str) -> bool {
    !identifier.is_empty()
        && identifier
            .chars()
            .all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn canonical_copy_sql(input: &str) -> String {
    let mut sql = input.trim();
    while let Some(stripped) = sql.strip_suffix(';') {
        sql = stripped.trim_end();
    }

    let mut canonical = String::with_capacity(sql.len());
    let mut previous_was_space = false;
    for ch in sql.chars() {
        if ch.is_whitespace() {
            if !previous_was_space {
                canonical.push(' ');
                previous_was_space = true;
            }
        } else {
            canonical.extend(ch.to_lowercase());
            previous_was_space = false;
        }
    }
    canonical.trim().to_owned()
}

fn strip_leading_sql_comments(mut statement: &str) -> Option<&str> {
    loop {
        let trimmed = statement.trim_start();
        let skipped = statement.len() - trimmed.len();
        statement = &statement[skipped..];

        if let Some(comment) = statement.strip_prefix("--") {
            if let Some(newline_idx) = comment.find('\n') {
                statement = &comment[newline_idx + '\n'.len_utf8()..];
                continue;
            }
            return Some("");
        }

        if let Some(comment) = statement.strip_prefix("/*") {
            let comment_end = nested_block_comment_end(comment)?;
            statement = &comment[comment_end..];
            continue;
        }

        return Some(statement);
    }
}

fn nested_block_comment_end(comment_body: &str) -> Option<usize> {
    let mut depth = 1usize;
    let mut previous_char: Option<char> = None;
    for (idx, ch) in comment_body.char_indices() {
        if previous_char == Some('/') && ch == '*' {
            depth = depth.saturating_add(1);
            previous_char = None;
            continue;
        }
        if previous_char == Some('*') && ch == '/' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(idx + ch.len_utf8());
            }
            previous_char = None;
            continue;
        }
        previous_char = Some(ch);
    }
    None
}

fn parenthesized_list(target: &str) -> Option<&str> {
    target
        .strip_prefix('(')?
        .strip_suffix(')')
        .filter(|_| target.ends_with(')'))
}

fn split_copy_sql_csv(input: &str) -> Option<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut chars = input.char_indices().peekable();
    let mut in_quote = false;
    while let Some((idx, ch)) = chars.next() {
        if ch == '\'' {
            if in_quote && matches!(chars.peek(), Some((_, '\''))) {
                chars.next();
                continue;
            }
            in_quote = !in_quote;
            continue;
        }
        if ch == ',' && !in_quote {
            parts.push(input[start..idx].trim());
            start = idx + ch.len_utf8();
        }
    }
    if in_quote {
        return None;
    }
    parts.push(input[start..].trim());
    Some(parts)
}

fn sql_keyword_rest_start(statement: &str, keyword: &str) -> Option<usize> {
    if !statement
        .get(..keyword.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(keyword))
    {
        return None;
    }
    let rest = statement.get(keyword.len()..)?;
    let mut chars = rest.chars();
    let first = chars.next()?;
    if !first.is_whitespace() {
        return None;
    }
    Some(keyword.len() + first.len_utf8())
}

fn decode_sql_copy_string_literal(arg: &str) -> Option<String> {
    let quoted = arg.strip_prefix('\'')?;
    if !quoted.ends_with('\'') {
        return None;
    }
    let inner = &quoted[..quoted.len() - 1];
    let mut decoded = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            if chars.peek() == Some(&'\'') {
                chars.next();
                decoded.push('\'');
            } else {
                return None;
            }
        } else {
            decoded.push(ch);
        }
    }
    Some(decoded)
}
