use std::borrow::Cow;

use crate::{ParseError, SqlValue};

/// Lower PostgreSQL-style `$n` parameters to canonical SQL literals without interpreting
/// placeholders inside quoted strings, quoted identifiers, dollar-quoted bodies, or comments.
///
/// This is the typed control-plane boundary used before the existing SQL parser. Values are rendered
/// from [`SqlValue`], never copied from untrusted text, so a text parameter cannot change SQL structure.
/// The highest parameter number must equal `params.len()`; repeated and out-of-order references are
/// accepted, while `$0`, missing values, surplus values, and numeric overflow fail loudly.
pub fn lower_sql_parameters(input: &str, params: &[SqlValue]) -> Result<String, ParseError> {
    let (output, highest) =
        transform_sql_parameters(input, CommentMode::Preserve, |number, end| {
            let value = params
                .get(number - 1)
                .ok_or(ParseError::InvalidParameterCount {
                    expected: number,
                    actual: params.len(),
                })?;
            render_parameter(value, parameter_has_explicit_cast(input, end)).map(Some)
        })?;
    if highest != params.len() {
        return Err(ParseError::InvalidParameterCount {
            expected: highest,
            actual: params.len(),
        });
    }
    Ok(output)
}

/// Canonicalize a fixed PostgreSQL compatibility program without changing quoted semantics.
///
/// Whitespace and comments outside quoted regions become a single token separator, and unquoted
/// SQL text is ASCII-case-folded. Single-quoted values, double-quoted identifiers, and dollar-
/// quoted bodies remain byte-for-byte intact. Callers may therefore compare a versioned client
/// program exactly without accidentally accepting a different literal or quoted identifier.
pub fn canonicalize_sql_for_exact_match(input: &str) -> Result<String, ParseError> {
    let normalized = normalize_sql_comments(input)?;
    let input = normalized.as_ref();
    let bytes = input.as_bytes();
    let mut output = String::with_capacity(input.len());
    let mut at = 0;
    let mut pending_space = false;

    while at < bytes.len() {
        match bytes[at] {
            byte if byte.is_ascii_whitespace() => {
                pending_space = !output.is_empty();
                at += 1;
            }
            b'\'' => {
                if pending_space {
                    output.push(' ');
                    pending_space = false;
                }
                let backslash_escapes = is_escape_string_prefix(input, at);
                copy_single_quoted(input, &mut at, &mut output, backslash_escapes)?;
            }
            b'"' => {
                if pending_space {
                    output.push(' ');
                    pending_space = false;
                }
                copy_double_quoted(input, &mut at, &mut output)?;
            }
            b'$' if at == 0 || !is_identifier_continuation_byte(bytes[at.saturating_sub(1)]) => {
                if let Some((delimiter, body_start)) = dollar_quote_delimiter(input, at) {
                    if pending_space {
                        output.push(' ');
                        pending_space = false;
                    }
                    copy_dollar_quoted(input, &mut at, &mut output, delimiter, body_start)?;
                } else {
                    if pending_space {
                        output.push(' ');
                        pending_space = false;
                    }
                    output.push('$');
                    at += 1;
                }
            }
            _ => {
                if pending_space {
                    output.push(' ');
                    pending_space = false;
                }
                let ch = input[at..].chars().next().expect("at is inside the input");
                output.push(ch.to_ascii_lowercase());
                at += ch.len_utf8();
            }
        }
    }

    Ok(output)
}

/// Quote/comment/dollar-quote-aware arity of raw PostgreSQL `$n` references.
pub(crate) fn sql_parameter_arity(input: &str) -> Result<usize, ParseError> {
    transform_sql_parameters(input, CommentMode::Preserve, |_number, _end| Ok(None))
        .map(|(_, highest)| highest)
}

/// Replace PostgreSQL comments outside quoted regions with one token-separating space.
///
/// The hand-written command parser consumes whitespace-delimited tokens. PostgreSQL comments are
/// lexical whitespace too, including nested block comments, so normalizing them once at the parser
/// entrance keeps every command grammar consistent without changing strings, quoted identifiers,
/// dollar-quoted bodies, or the original source retained by `ParsedCommand` for WAL identity.
pub(super) fn normalize_sql_comments(input: &str) -> Result<Cow<'_, str>, ParseError> {
    if !input
        .as_bytes()
        .windows(2)
        .any(|pair| pair == b"--" || pair == b"/*")
    {
        return Ok(Cow::Borrowed(input));
    }
    transform_sql_parameters(input, CommentMode::ReplaceWithSpace, |_number, _end| {
        Ok(None)
    })
    .map(|(output, _highest)| Cow::Owned(output))
}

#[derive(Clone, Copy)]
enum CommentMode {
    Preserve,
    ReplaceWithSpace,
}

fn transform_sql_parameters(
    input: &str,
    comment_mode: CommentMode,
    mut replacement: impl FnMut(usize, usize) -> Result<Option<String>, ParseError>,
) -> Result<(String, usize), ParseError> {
    let bytes = input.as_bytes();
    let mut output = String::with_capacity(input.len());
    let mut at = 0;
    let mut highest = 0_usize;

    while at < bytes.len() {
        match bytes[at] {
            b'\'' => {
                let backslash_escapes = is_escape_string_prefix(input, at);
                copy_single_quoted(input, &mut at, &mut output, backslash_escapes)?;
            }
            b'"' => copy_double_quoted(input, &mut at, &mut output)?,
            b'-' if bytes.get(at + 1) == Some(&b'-') => {
                let output_start = output.len();
                copy_line_comment(input, &mut at, &mut output);
                if matches!(comment_mode, CommentMode::ReplaceWithSpace) {
                    output.truncate(output_start);
                    output.push(' ');
                }
            }
            b'/' if bytes.get(at + 1) == Some(&b'*') => {
                let output_start = output.len();
                copy_block_comment(input, &mut at, &mut output)?;
                if matches!(comment_mode, CommentMode::ReplaceWithSpace) {
                    output.truncate(output_start);
                    output.push(' ');
                }
            }
            b'$' => {
                let follows_identifier = at > 0 && is_identifier_continuation_byte(bytes[at - 1]);
                if follows_identifier {
                    output.push('$');
                    at += 1;
                } else if let Some((delimiter, end)) = dollar_quote_delimiter(input, at) {
                    copy_dollar_quoted(input, &mut at, &mut output, delimiter, end)?;
                } else if bytes.get(at + 1).is_some_and(u8::is_ascii_digit) {
                    let digits_start = at + 1;
                    let mut end = digits_start;
                    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
                        end += 1;
                    }
                    let number = input[digits_start..end]
                        .parse::<usize>()
                        .map_err(|_| ParseError::InvalidParameterReference)?;
                    if number == 0 {
                        return Err(ParseError::InvalidParameterReference);
                    }
                    highest = highest.max(number);
                    match replacement(number, end)? {
                        Some(replacement) => output.push_str(&replacement),
                        None => output.push_str(&input[at..end]),
                    }
                    at = end;
                } else {
                    output.push('$');
                    at += 1;
                }
            }
            _ => {
                let ch = input[at..].chars().next().expect("at is inside the input");
                output.push(ch);
                at += ch.len_utf8();
            }
        }
    }

    Ok((output, highest))
}

fn render_parameter(value: &SqlValue, already_cast: bool) -> Result<String, ParseError> {
    Ok(match value {
        SqlValue::Null => "NULL".to_string(),
        SqlValue::Int2(value) if already_cast => value.to_string(),
        SqlValue::Int2(value) => format!("{value}::int2"),
        SqlValue::Int4(value) if already_cast => value.to_string(),
        SqlValue::Int4(value) => format!("{value}::int4"),
        SqlValue::Int8(value) if already_cast => value.to_string(),
        SqlValue::Int8(value) => format!("{value}::int8"),
        SqlValue::Numeric(value) if already_cast => value.to_decimal_string(),
        SqlValue::Numeric(value) => format!(
            "{}::numeric({},{})",
            value.to_decimal_string(),
            crate::NUMERIC_DEFAULT_PRECISION,
            value.scale
        ),
        SqlValue::Bool(value) if already_cast => if *value { "TRUE" } else { "FALSE" }.to_string(),
        SqlValue::Bool(value) => format!("{}::bool", if *value { "TRUE" } else { "FALSE" }),
        SqlValue::Text(value) if already_cast => quote_literal(value),
        SqlValue::Text(value) => format!("{}::text", quote_literal(value)),
        SqlValue::Date(value) if already_cast => {
            quote_literal(&crate::datetime::format_date(*value))
        }
        SqlValue::Date(value) => format!(
            "{}::date",
            quote_literal(&crate::datetime::format_date(*value))
        ),
        SqlValue::Timestamp(value) if already_cast => {
            quote_literal(&crate::datetime::format_timestamp(*value))
        }
        SqlValue::Timestamp(value) => format!(
            "{}::timestamp",
            quote_literal(&crate::datetime::format_timestamp(*value))
        ),
        SqlValue::Uuid(value) => {
            let literal = quote_literal(&crate::uuid::format_uuid(value));
            if already_cast {
                literal
            } else {
                format!("{literal}::uuid")
            }
        }
        SqlValue::Parameter { .. } => return Err(ParseError::InvalidParameterReference),
    })
}

fn parameter_has_explicit_cast(input: &str, parameter_end: usize) -> bool {
    input[parameter_end..].trim_start().starts_with("::")
}

fn quote_literal(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('\'');
    for ch in value.chars() {
        quoted.push(ch);
        if ch == '\'' {
            quoted.push('\'');
        }
    }
    quoted.push('\'');
    quoted
}

pub(super) fn is_escape_string_prefix(input: &str, quote_at: usize) -> bool {
    let bytes = input.as_bytes();
    quote_at > 0
        && matches!(bytes[quote_at - 1], b'e' | b'E')
        && (quote_at == 1 || !is_identifier_continuation_byte(bytes[quote_at - 2]))
}

fn copy_single_quoted(
    input: &str,
    at: &mut usize,
    output: &mut String,
    backslash_escapes: bool,
) -> Result<(), ParseError> {
    let bytes = input.as_bytes();
    let start = *at;
    *at += 1;
    while *at < bytes.len() {
        match bytes[*at] {
            b'\'' if bytes.get(*at + 1) == Some(&b'\'') => *at += 2,
            b'\'' => {
                *at += 1;
                output.push_str(&input[start..*at]);
                return Ok(());
            }
            b'\\' if backslash_escapes && bytes.get(*at + 1).is_some() => {
                *at += 1;
                *at += input[*at..]
                    .chars()
                    .next()
                    .expect("backslash has a following character")
                    .len_utf8();
            }
            _ => *at += input[*at..].chars().next().expect("valid UTF-8").len_utf8(),
        }
    }
    Err(ParseError::InvalidParameterReference)
}

fn copy_double_quoted(input: &str, at: &mut usize, output: &mut String) -> Result<(), ParseError> {
    let bytes = input.as_bytes();
    let start = *at;
    *at += 1;
    while *at < bytes.len() {
        match bytes[*at] {
            b'"' if bytes.get(*at + 1) == Some(&b'"') => *at += 2,
            b'"' => {
                *at += 1;
                output.push_str(&input[start..*at]);
                return Ok(());
            }
            _ => *at += input[*at..].chars().next().expect("valid UTF-8").len_utf8(),
        }
    }
    Err(ParseError::InvalidParameterReference)
}

fn copy_line_comment(input: &str, at: &mut usize, output: &mut String) {
    let start = *at;
    *at = input[*at..]
        .find(['\r', '\n'])
        .map_or(input.len(), |offset| *at + offset + 1);
    output.push_str(&input[start..*at]);
}

fn copy_block_comment(input: &str, at: &mut usize, output: &mut String) -> Result<(), ParseError> {
    let bytes = input.as_bytes();
    let start = *at;
    *at += 2;
    let mut depth = 1_u32;
    while *at < bytes.len() {
        if bytes.get(*at..*at + 2) == Some(b"/*") {
            depth = depth
                .checked_add(1)
                .ok_or(ParseError::InvalidParameterReference)?;
            *at += 2;
        } else if bytes.get(*at..*at + 2) == Some(b"*/") {
            depth -= 1;
            *at += 2;
            if depth == 0 {
                output.push_str(&input[start..*at]);
                return Ok(());
            }
        } else {
            *at += input[*at..].chars().next().expect("valid UTF-8").len_utf8();
        }
    }
    Err(ParseError::InvalidParameterReference)
}

pub(super) fn dollar_quote_delimiter(input: &str, at: usize) -> Option<(&str, usize)> {
    let tail = &input.as_bytes()[at + 1..];
    let tag_end = tail.iter().position(|byte| *byte == b'$')?;
    let tag = &tail[..tag_end];
    if tag
        .first()
        .is_some_and(|byte| !byte.is_ascii_alphabetic() && *byte != b'_' && *byte < 0x80)
        || tag
            .iter()
            .skip(1)
            .any(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_' && *byte < 0x80)
    {
        return None;
    }
    let end = at + 1 + tag_end;
    Some((&input[at..=end], end + 1))
}

pub(super) fn is_identifier_continuation_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$') || byte >= 0x80
}

fn copy_dollar_quoted(
    input: &str,
    at: &mut usize,
    output: &mut String,
    delimiter: &str,
    body_start: usize,
) -> Result<(), ParseError> {
    let end = input[body_start..]
        .find(delimiter)
        .map(|offset| body_start + offset + delimiter.len())
        .ok_or(ParseError::InvalidParameterReference)?;
    output.push_str(&input[*at..end]);
    *at = end;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_canonicalization_preserves_quoted_semantics() {
        assert_eq!(
            canonicalize_sql_for_exact_match(
                " SELECT /* gap */ VALUE, 'MiX  ed', \"Case ID\", $tag$Body  X$tag$  FROM T "
            )
            .unwrap(),
            "select value, 'MiX  ed', \"Case ID\", $tag$Body  X$tag$ from t"
        );
        assert_ne!(
            canonicalize_sql_for_exact_match("SELECT array_to_string(v, ' ')").unwrap(),
            canonicalize_sql_for_exact_match("SELECT array_to_string(v, '  ')").unwrap(),
        );
        assert_ne!(
            canonicalize_sql_for_exact_match("SELECT 'S'::\"char\"").unwrap(),
            canonicalize_sql_for_exact_match("SELECT 's'::\"CHAR\"").unwrap(),
        );
    }

    #[test]
    fn lowers_typed_parameters_without_changing_sql_structure() {
        let uuid = crate::uuid::parse_uuid("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let lowered = lower_sql_parameters(
            "SELECT '$1', \"$2\", $tag$ $3 $tag$, $1, $2, $3, $4 -- $5\n/* $6 */",
            &[
                SqlValue::Int4(7),
                SqlValue::Int8(-9),
                SqlValue::Text("x'); DROP TABLE accounts; --".to_string()),
                SqlValue::Uuid(uuid),
            ],
        )
        .unwrap();
        assert_eq!(
            lowered,
            "SELECT '$1', \"$2\", $tag$ $3 $tag$, 7::int4, -9::int8, \
             'x''); DROP TABLE accounts; --'::text, \
             '550e8400-e29b-41d4-a716-446655440000'::uuid -- $5\n/* $6 */"
        );
    }

    #[test]
    fn repeated_out_of_order_parameters_are_allowed_but_count_is_exact() {
        assert_eq!(
            lower_sql_parameters("SELECT $2, $1, $2", &[SqlValue::Int4(1), SqlValue::Int4(2)])
                .unwrap(),
            "SELECT 2::int4, 1::int4, 2::int4"
        );
        assert!(matches!(
            lower_sql_parameters("SELECT $2", &[SqlValue::Int4(1)]),
            Err(ParseError::InvalidParameterCount {
                expected: 2,
                actual: 1
            })
        ));
        assert!(matches!(
            lower_sql_parameters("SELECT 1", &[SqlValue::Int4(1)]),
            Err(ParseError::InvalidParameterCount {
                expected: 0,
                actual: 1
            })
        ));
        assert!(matches!(
            lower_sql_parameters("SELECT $0", &[]),
            Err(ParseError::InvalidParameterReference)
        ));
    }

    #[test]
    fn nested_comments_and_quoted_utf8_are_preserved() {
        assert_eq!(
            lower_sql_parameters(
                "SELECT 'héllo $1', \"名$2\", $é$ $4 $é$, /* outer /* $3 */ done */ $1",
                &[SqlValue::Bool(true)]
            )
            .unwrap(),
            "SELECT 'héllo $1', \"名$2\", $é$ $4 $é$, /* outer /* $3 */ done */ TRUE::bool"
        );
    }

    #[test]
    fn postgres_identifier_boundaries_do_not_manufacture_parameters_or_dollar_quotes() {
        assert_eq!(
            lower_sql_parameters(
                "SELECT foo$tag$bar$tag$, foo$1, $e\u{301}$ $2 $e\u{301}$, $1",
                &[SqlValue::Int4(9)]
            )
            .unwrap(),
            "SELECT foo$tag$bar$tag$, foo$1, $e\u{301}$ $2 $e\u{301}$, 9::int4"
        );
    }

    #[test]
    fn backslash_before_multibyte_utf8_never_splits_a_character() {
        assert_eq!(
            lower_sql_parameters("SELECT '\\é', $1", &[SqlValue::Int4(7)]).unwrap(),
            "SELECT '\\é', 7::int4"
        );
        assert_eq!(
            lower_sql_parameters("SELECT '\\名', $1", &[SqlValue::Int4(8)]).unwrap(),
            "SELECT '\\名', 8::int4"
        );
    }

    #[test]
    fn postgres_escape_strings_and_cr_line_comments_keep_parameter_scope_exact() {
        assert_eq!(
            lower_sql_parameters("SELECT '\\', $1", &[SqlValue::Int4(7)]).unwrap(),
            "SELECT '\\', 7::int4"
        );
        assert_eq!(
            lower_sql_parameters("SELECT E'\\\'', $1", &[SqlValue::Int4(8)]).unwrap(),
            "SELECT E'\\\'', 8::int4"
        );
        assert_eq!(
            lower_sql_parameters("SELECT 1 -- $2\r WHERE id = $1", &[SqlValue::Int4(9)]).unwrap(),
            "SELECT 1 -- $2\r WHERE id = 9::int4"
        );
    }

    #[test]
    fn supported_non_null_parameter_variants_keep_explicit_sql_types() {
        let values = [
            SqlValue::Int2(7),
            SqlValue::Int4(7),
            SqlValue::Int8(7),
            SqlValue::Numeric(crate::Decimal128::new(700, 2)),
            SqlValue::Bool(true),
        ];
        assert_eq!(
            lower_sql_parameters("SELECT $1, $2, $3, $4, $5", &values).unwrap(),
            "SELECT 7::int2, 7::int4, 7::int8, 7.00::numeric(38,2), TRUE::bool"
        );
    }

    #[test]
    fn canonical_point_read_casts_reparse_as_the_declared_types() {
        let lowered = lower_sql_parameters(
            "SELECT balance_cents, version, status FROM accounts \
             WHERE tenant_id = $1 AND account_id = $2",
            &[SqlValue::Int4(7), SqlValue::Int8(70_001)],
        )
        .unwrap();
        let crate::Command::Select(select) = crate::parse_command(&lowered).unwrap() else {
            panic!("canonical R1 must remain a SELECT")
        };
        assert_eq!(select.filters[0].value, SqlValue::Int4(7));
        assert_eq!(select.filters[1].value, SqlValue::Int8(70_001));
    }

    #[test]
    fn existing_explicit_cast_is_not_duplicated() {
        let lowered = lower_sql_parameters(
            "DELETE FROM accounts WHERE id = $1::int8 AND active = $2 :: bool",
            &[SqlValue::Int8(9), SqlValue::Bool(true)],
        )
        .unwrap();
        assert_eq!(
            lowered,
            "DELETE FROM accounts WHERE id = 9::int8 AND active = TRUE :: bool"
        );
        assert!(crate::ParsedCommand::parse(&lowered).is_ok());
    }

    #[test]
    fn temporal_parameters_render_and_reparse_postgresql_bc_text_at_finite_lower_boundaries() {
        let lowered = lower_sql_parameters(
            "INSERT INTO temporal_boundary (d, t) VALUES ($1, $2)",
            &[
                SqlValue::Date(crate::datetime::PG_DATE_MIN_DAYS),
                SqlValue::Timestamp(crate::datetime::PG_TIMESTAMP_MIN_MICROS),
            ],
        )
        .unwrap();
        assert_eq!(
            lowered,
            "INSERT INTO temporal_boundary (d, t) VALUES ('4714-11-24 BC'::date, \
             '4714-11-24 00:00:00 BC'::timestamp)"
        );
        assert!(
            crate::ParsedCommand::parse(&lowered).is_ok(),
            "the canonical parameter source must remain coercible through the SQL parser"
        );
    }
}
