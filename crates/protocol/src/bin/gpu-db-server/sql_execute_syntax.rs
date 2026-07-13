// Shared legacy SQL EXECUTE syntax and literal decoding. This is not a product execution path.

use super::canonical_sql;

pub(super) fn strip_sql_comments(query: &str) -> String {
    let mut stripped = String::with_capacity(query.len());
    let mut chars = query.char_indices().peekable();
    let mut last_pushed = 0;
    let mut in_quote = false;
    let mut in_quoted_identifier = false;

    while let Some((idx, ch)) = chars.next() {
        if ch == '\'' {
            if in_quoted_identifier {
                continue;
            }
            if in_quote && matches!(chars.peek(), Some((_, '\''))) {
                chars.next();
                continue;
            }
            in_quote = !in_quote;
            continue;
        }
        if ch == '"' {
            if in_quote {
                continue;
            }
            if in_quoted_identifier && matches!(chars.peek(), Some((_, '"'))) {
                chars.next();
                continue;
            }
            in_quoted_identifier = !in_quoted_identifier;
            continue;
        }
        if in_quote || in_quoted_identifier {
            continue;
        }
        if ch == '$' {
            if let Some(tag) = sql_dollar_quote_tag_at(query, idx) {
                let body_start = idx + tag.len();
                if let Some(close_relative) = query[body_start..].find(tag) {
                    let close_end = body_start + close_relative + tag.len();
                    while chars
                        .peek()
                        .is_some_and(|(next_idx, _)| *next_idx < close_end)
                    {
                        chars.next();
                    }
                }
            }
            continue;
        }
        if ch != '-' && ch != '/' {
            continue;
        }

        if ch == '-' && matches!(chars.peek(), Some((_, '-'))) {
            stripped.push_str(&query[last_pushed..idx]);
            chars.next();
            let mut comment_end = query.len();
            for (next_idx, next_ch) in chars.by_ref() {
                if next_ch == '\n' {
                    comment_end = next_idx + next_ch.len_utf8();
                    stripped.push('\n');
                    break;
                }
            }
            if comment_end == query.len() {
                stripped.push(' ');
            }
            last_pushed = comment_end;
            continue;
        }

        if ch == '/' && matches!(chars.peek(), Some((_, '*'))) {
            stripped.push_str(&query[last_pushed..idx]);
            chars.next();
            let mut comment_end = query.len();
            let mut depth = 1usize;
            let mut saw_newline = false;
            let mut previous_char: Option<char> = None;
            for (next_idx, next_ch) in chars.by_ref() {
                if next_ch == '\n' {
                    saw_newline = true;
                }
                if previous_char == Some('/') && next_ch == '*' {
                    depth = depth.saturating_add(1);
                    previous_char = None;
                    continue;
                }
                if previous_char == Some('*') && next_ch == '/' {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        comment_end = next_idx + next_ch.len_utf8();
                        break;
                    }
                    previous_char = None;
                    continue;
                }
                previous_char = Some(next_ch);
            }
            stripped.push(if saw_newline { '\n' } else { ' ' });
            last_pushed = comment_end;
        }
    }

    stripped.push_str(&query[last_pushed..]);
    stripped
}

pub(super) fn sql_keyword_rest_start(statement: &str, keyword: &str) -> Option<usize> {
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

pub(super) fn strip_leading_sql_comments(mut statement: &str) -> Option<&str> {
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

pub(super) fn parse_sql_execute(statement: &str) -> Option<(String, Vec<Option<String>>)> {
    let trimmed = strip_leading_sql_comments(statement.trim().trim_end_matches(';').trim())?;
    let rest_start = sql_keyword_rest_start(trimmed, "execute")?;
    let original_rest = strip_sql_comments(&trimmed[rest_start..]);
    let (name, args) = {
        let (name, arg_list) = split_sql_name_and_optional_parenthesized_list(&original_rest)?;
        if let Some(arg_list) = arg_list {
            let args = if arg_list.trim().is_empty() {
                Vec::new()
            } else {
                split_sql_csv(arg_list)?
                    .into_iter()
                    .map(decode_sql_execute_argument)
                    .collect::<Option<Vec<_>>>()?
            };
            (name, args)
        } else {
            (name, Vec::new())
        }
    };
    Some((name, args))
}

pub(super) fn split_sql_name_and_optional_parenthesized_list(
    target: &str,
) -> Option<(String, Option<&str>)> {
    let target = target.trim();
    if let Some(quoted) = target.strip_prefix('"') {
        let mut chars = quoted.char_indices().peekable();
        while let Some((idx, ch)) = chars.next() {
            if ch == '"' {
                if chars.peek().is_some_and(|(_, next)| *next == '"') {
                    chars.next();
                    continue;
                }
                let name_end = idx + 2;
                let name = parse_supported_cursor_name(&target[..name_end])?;
                let rest = target[name_end..].trim();
                return if rest.is_empty() {
                    Some((name, None))
                } else {
                    parenthesized_list(rest).map(|list| (name, Some(list)))
                };
            }
        }
        None
    } else if let Some(open_idx) = target.find('(') {
        let name = parse_supported_cursor_name(target[..open_idx].trim())?;
        parenthesized_list(target[open_idx..].trim()).map(|list| (name, Some(list)))
    } else {
        parse_supported_cursor_name(target).map(|name| (name, None))
    }
}

fn parenthesized_list(target: &str) -> Option<&str> {
    target
        .strip_prefix('(')?
        .strip_suffix(')')
        .filter(|_| target.ends_with(')'))
}

pub(super) fn split_sql_csv(input: &str) -> Option<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut chars = input.char_indices().peekable();
    let mut in_quote = false;
    let mut in_escape_string = false;
    while let Some((idx, ch)) = chars.next() {
        if !in_quote && ch == '$' {
            if let Some(tag) = sql_dollar_quote_tag_at(input, idx) {
                let body_start = idx + tag.len();
                let close_relative = input[body_start..].find(tag)?;
                let close_end = body_start + close_relative + tag.len();
                while chars
                    .peek()
                    .is_some_and(|(next_idx, _)| *next_idx < close_end)
                {
                    chars.next();
                }
                continue;
            }
        }
        if ch == '\'' {
            if !in_quote {
                in_escape_string = input[..idx]
                    .chars()
                    .next_back()
                    .is_some_and(|prefix| matches!(prefix, 'E' | 'e'));
            }
            if in_quote && matches!(chars.peek(), Some((_, '\''))) {
                chars.next();
                continue;
            }
            in_quote = !in_quote;
            if !in_quote {
                in_escape_string = false;
            }
            continue;
        }
        if in_quote && in_escape_string && ch == '\\' {
            chars.next();
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

fn decode_sql_execute_argument(arg: &str) -> Option<Option<String>> {
    let trimmed = normalize_sql_execute_argument(arg.trim())?;
    if canonical_sql(trimmed) == "null" {
        return Some(None);
    }
    if trimmed.starts_with('\'')
        || trimmed.starts_with("E'")
        || trimmed.starts_with("e'")
        || trimmed.starts_with("N'")
        || trimmed.starts_with("n'")
        || trimmed.starts_with("U&'")
        || trimmed.starts_with("u&'")
    {
        decode_sql_execute_string_literal(trimmed)
            .or_else(|| decode_concatenated_standard_sql_string_literals(trimmed))
            .map(Some)
    } else if sql_dollar_quote_tag_at(trimmed, 0).is_some() {
        decode_dollar_sql_string_literal(trimmed).map(Some)
    } else if !trimmed.is_empty() && !trimmed.contains(char::is_whitespace) {
        Some(Some(trimmed.to_string()))
    } else {
        None
    }
}

fn decode_sql_execute_string_literal(arg: &str) -> Option<String> {
    if let Some(quoted) = arg.strip_prefix('\'') {
        return decode_standard_sql_string_literal(quoted);
    }
    if let Some(quoted) = arg.strip_prefix("N'").or_else(|| arg.strip_prefix("n'")) {
        return decode_standard_sql_string_literal(quoted);
    }
    if arg.starts_with('$') {
        return decode_dollar_sql_string_literal(arg);
    }
    if arg.starts_with("U&'") || arg.starts_with("u&'") {
        return decode_unicode_sql_string_literal(arg);
    }
    let escaped = arg.strip_prefix("E'").or_else(|| arg.strip_prefix("e'"))?;
    decode_escape_sql_string_literal(escaped)
}

fn decode_dollar_sql_string_literal(arg: &str) -> Option<String> {
    let tag = sql_dollar_quote_tag_at(arg, 0)?;
    let body = arg.strip_prefix(tag)?.strip_suffix(tag)?;
    Some(body.to_string())
}

fn sql_dollar_quote_tag_at(input: &str, start: usize) -> Option<&str> {
    let after_open = input.get(start..)?.strip_prefix('$')?;
    let close_relative = after_open.find('$')?;
    let tag_end = start + 1 + close_relative + 1;
    let tag = &input[start..tag_end];
    let tag_body = &tag[1..tag.len() - 1];
    if tag_body
        .chars()
        .all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        Some(tag)
    } else {
        None
    }
}

fn decode_standard_sql_string_literal(quoted: &str) -> Option<String> {
    if !quoted.ends_with('\'') {
        return None;
    }
    let inner = &quoted[..quoted.len() - 1];
    decode_standard_sql_string_body(inner)
}

fn decode_standard_sql_string_body(inner: &str) -> Option<String> {
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

fn decode_concatenated_standard_sql_string_literals(arg: &str) -> Option<String> {
    let mut rest = arg;
    let mut decoded = String::new();
    let mut pieces = 0usize;
    loop {
        let (inner, after_literal) = split_standard_sql_quoted_literal(rest)?;
        decoded.push_str(&decode_standard_sql_string_body(inner)?);
        pieces += 1;
        if after_literal.is_empty() {
            return (pieces > 1).then_some(decoded);
        }
        let separator_len = after_literal.len() - after_literal.trim_start().len();
        let separator = &after_literal[..separator_len];
        if !separator.contains('\n') {
            return None;
        }
        rest = after_literal[separator_len..].trim_start();
        if !rest.starts_with('\'') {
            return None;
        }
    }
}

fn decode_escape_sql_string_literal(quoted: &str) -> Option<String> {
    if !quoted.ends_with('\'') {
        return None;
    }
    let inner = &quoted[..quoted.len() - 1];
    let mut decoded = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\'' => {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                    decoded.push('\'');
                } else {
                    return None;
                }
            }
            '\\' => decode_escape_sql_string_backslash(&mut chars, &mut decoded)?,
            other => decoded.push(other),
        }
    }
    Some(decoded)
}

fn decode_escape_sql_string_backslash<I>(
    chars: &mut std::iter::Peekable<I>,
    decoded: &mut String,
) -> Option<()>
where
    I: Iterator<Item = char>,
{
    match chars.next()? {
        '\\' => decoded.push('\\'),
        '\'' => decoded.push('\''),
        'b' => decoded.push('\u{0008}'),
        'f' => decoded.push('\u{000c}'),
        'n' => decoded.push('\n'),
        'r' => decoded.push('\r'),
        't' => decoded.push('\t'),
        'x' => {
            let codepoint = take_variable_hex_codepoint(chars, 2)?;
            decoded.push(char::from_u32(codepoint)?);
        }
        first if first.is_ascii_digit() && first < '8' => {
            let codepoint = take_octal_codepoint(chars, first, 3)?;
            decoded.push(char::from_u32(codepoint)?);
        }
        other => decoded.push(other),
    }
    Some(())
}

fn take_variable_hex_codepoint<I>(chars: &mut std::iter::Peekable<I>, max_len: usize) -> Option<u32>
where
    I: Iterator<Item = char>,
{
    let mut codepoint = 0u32;
    let mut consumed = 0usize;
    while consumed < max_len {
        let Some(digit) = chars.peek().copied().and_then(|ch| ch.to_digit(16)) else {
            break;
        };
        chars.next();
        codepoint = codepoint.checked_mul(16)?.checked_add(digit)?;
        consumed += 1;
    }
    (consumed > 0).then_some(codepoint)
}

fn take_octal_codepoint<I>(
    chars: &mut std::iter::Peekable<I>,
    first: char,
    max_len: usize,
) -> Option<u32>
where
    I: Iterator<Item = char>,
{
    let mut codepoint = first.to_digit(8)?;
    let mut consumed = 1usize;
    while consumed < max_len {
        let Some(digit) = chars
            .peek()
            .copied()
            .filter(|ch| ch.is_ascii_digit() && *ch < '8')
            .and_then(|ch| ch.to_digit(8))
        else {
            break;
        };
        chars.next();
        codepoint = codepoint.checked_mul(8)?.checked_add(digit)?;
        consumed += 1;
    }
    Some(codepoint)
}

fn decode_unicode_sql_string_literal(arg: &str) -> Option<String> {
    let quoted = arg.strip_prefix("U&").or_else(|| arg.strip_prefix("u&"))?;
    let (inner, rest) = split_standard_sql_quoted_literal(quoted)?;
    let escape_char = unicode_sql_string_escape_char(rest.trim())?;
    decode_unicode_sql_string_body(inner, escape_char)
}

fn split_standard_sql_quoted_literal(arg: &str) -> Option<(&str, &str)> {
    let body = arg.strip_prefix('\'')?;
    let mut chars = body.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if ch == '\'' {
            if chars.peek() == Some(&(idx + ch.len_utf8(), '\'')) {
                chars.next();
                continue;
            }
            return Some((&body[..idx], &body[idx + ch.len_utf8()..]));
        }
    }
    None
}

fn unicode_sql_string_escape_char(rest: &str) -> Option<char> {
    if rest.is_empty() {
        return Some('\\');
    }
    let value_start = sql_keyword_rest_start(rest, "uescape")?;
    let (value, value_rest) = split_standard_sql_quoted_literal(rest[value_start..].trim_start())?;
    if !value_rest.trim().is_empty() {
        return None;
    }
    let mut chars = value.chars();
    let escape = chars.next()?;
    if chars.next().is_some()
        || escape.is_whitespace()
        || escape == '+'
        || escape == '\''
        || escape == '"'
        || escape.is_ascii_hexdigit()
    {
        return None;
    }
    Some(escape)
}

fn decode_unicode_sql_string_body(inner: &str, escape_char: char) -> Option<String> {
    let mut decoded = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\'' => {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                    decoded.push('\'');
                } else {
                    return None;
                }
            }
            _ if ch == escape_char => {
                if chars.peek() == Some(&'+') {
                    chars.next();
                    let codepoint = take_hex_codepoint(&mut chars, 6)?;
                    decoded.push(char::from_u32(codepoint)?);
                } else {
                    let codepoint = take_hex_codepoint(&mut chars, 4)?;
                    decoded.push(char::from_u32(codepoint)?);
                }
            }
            other => decoded.push(other),
        }
    }
    Some(decoded)
}

fn take_hex_codepoint<I>(chars: &mut std::iter::Peekable<I>, len: usize) -> Option<u32>
where
    I: Iterator<Item = char>,
{
    let mut codepoint = 0u32;
    for _ in 0..len {
        let digit = chars.next()?.to_digit(16)?;
        codepoint = codepoint.checked_mul(16)?.checked_add(digit)?;
    }
    Some(codepoint)
}

fn normalize_sql_execute_argument(mut arg: &str) -> Option<&str> {
    loop {
        let parenthesized = strip_parenthesized_sql_execute_argument(arg)?;
        let cast_function_stripped = strip_supported_sql_execute_cast_function(parenthesized);
        let cast_target = cast_function_stripped.unwrap_or(parenthesized);
        let cast_stripped = strip_supported_sql_execute_cast(cast_target);
        let type_prefixed_stripped =
            strip_supported_sql_execute_typed_literal(cast_stripped.unwrap_or(cast_target));
        let normalized = type_prefixed_stripped
            .unwrap_or_else(|| cast_stripped.unwrap_or(cast_target))
            .trim();
        if normalized == arg {
            return Some(normalized);
        }
        if normalized.is_empty() {
            return None;
        }
        arg = normalized;
    }
}

fn strip_supported_sql_execute_cast_function(arg: &str) -> Option<&str> {
    let arg = arg.trim();
    let after_keyword = arg.get(4..)?;
    if !arg
        .get(..4)
        .is_some_and(|keyword| keyword.eq_ignore_ascii_case("cast"))
    {
        return None;
    }
    let after_keyword = after_keyword.trim_start();
    let inner = parenthesized_list(after_keyword)?.trim();
    let as_idx = find_top_level_sql_execute_cast_as(inner)?;
    let value = inner[..as_idx].trim();
    let ty = inner[as_idx + 2..].trim();
    if value.is_empty() {
        return None;
    }
    match canonical_sql(ty).as_str() {
        "int" | "int4" | "integer" | "pg_catalog.int4" | "pg_catalog.integer" | "text"
        | "pg_catalog.text" => Some(value),
        _ => None,
    }
}

fn find_top_level_sql_execute_cast_as(input: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_quote = false;
    let mut in_quoted_identifier = false;
    let mut chars = input.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if in_quote {
            if ch == '\'' {
                if chars.peek().is_some_and(|(_, next)| *next == '\'') {
                    chars.next();
                } else {
                    in_quote = false;
                }
            }
            continue;
        }
        if in_quoted_identifier {
            if ch == '"' {
                if chars.peek().is_some_and(|(_, next)| *next == '"') {
                    chars.next();
                } else {
                    in_quoted_identifier = false;
                }
            }
            continue;
        }
        if ch == '$' {
            if let Some(tag) = sql_dollar_quote_tag_at(input, idx) {
                let body_start = idx + tag.len();
                let close_relative = input[body_start..].find(tag)?;
                let close_end = body_start + close_relative + tag.len();
                while chars
                    .peek()
                    .is_some_and(|(next_idx, _)| *next_idx < close_end)
                {
                    chars.next();
                }
            }
            continue;
        }
        match ch {
            '\'' => in_quote = true,
            '"' => in_quoted_identifier = true,
            '(' => depth = depth.saturating_add(1),
            ')' => depth = depth.saturating_sub(1),
            'a' | 'A'
                if depth == 0
                    && input[idx..]
                        .get(..2)
                        .is_some_and(|candidate| candidate.eq_ignore_ascii_case("as"))
                    && input[..idx]
                        .chars()
                        .next_back()
                        .is_some_and(char::is_whitespace)
                    && input[idx + 2..]
                        .chars()
                        .next()
                        .is_some_and(char::is_whitespace) =>
            {
                return Some(idx);
            }
            _ => {}
        }
    }
    None
}

fn strip_supported_sql_execute_typed_literal(arg: &str) -> Option<&str> {
    let arg = arg.trim();
    let (ty, value_start) = split_leading_sql_type_name(arg)?;
    if !matches!(
        canonical_sql(ty).as_str(),
        "int"
            | "int4"
            | "integer"
            | "pg_catalog.int4"
            | "pg_catalog.integer"
            | "text"
            | "pg_catalog.text"
    ) {
        return None;
    }
    let value = arg[value_start..].trim_start();
    if value.starts_with('\'')
        || value.starts_with("E'")
        || value.starts_with("e'")
        || value.starts_with("N'")
        || value.starts_with("n'")
        || value.starts_with("U&'")
        || value.starts_with("u&'")
        || sql_dollar_quote_tag_at(value, 0).is_some()
    {
        Some(value)
    } else {
        None
    }
}

fn split_leading_sql_type_name(arg: &str) -> Option<(&str, usize)> {
    let mut end = 0usize;
    let mut saw_dot = false;
    for (idx, ch) in arg.char_indices() {
        if ch == '.' {
            if saw_dot || end == 0 {
                return None;
            }
            saw_dot = true;
            end = idx + ch.len_utf8();
            continue;
        }
        if ch == '_' || ch.is_ascii_alphanumeric() {
            end = idx + ch.len_utf8();
            continue;
        }
        if ch.is_whitespace() && end > 0 {
            return Some((&arg[..end], idx + ch.len_utf8()));
        }
        return None;
    }
    None
}

fn strip_parenthesized_sql_execute_argument(mut arg: &str) -> Option<&str> {
    loop {
        let Some(without_open) = arg.strip_prefix('(') else {
            return Some(arg);
        };
        let Some(inner) = without_open.strip_suffix(')') else {
            return Some(arg);
        };
        let inner = inner.trim();
        if inner.is_empty() || !parenthesized_list_is_balanced(inner) {
            return None;
        }
        arg = inner;
    }
}

fn strip_supported_sql_execute_cast(arg: &str) -> Option<&str> {
    let cast_idx = find_top_level_sql_execute_cast(arg)?;
    let value = arg[..cast_idx].trim();
    let ty = arg[cast_idx + 2..].trim();
    if value.is_empty() {
        return None;
    }
    match canonical_sql(ty).as_str() {
        "int" | "int4" | "integer" | "pg_catalog.int4" | "pg_catalog.integer" | "text"
        | "pg_catalog.text" => Some(value),
        _ => None,
    }
}

fn find_top_level_sql_execute_cast(input: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_quote = false;
    let mut in_quoted_identifier = false;
    let mut chars = input.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if in_quote {
            if ch == '\'' {
                if chars.peek().is_some_and(|(_, next)| *next == '\'') {
                    chars.next();
                } else {
                    in_quote = false;
                }
            }
            continue;
        }
        if in_quoted_identifier {
            if ch == '"' {
                if chars.peek().is_some_and(|(_, next)| *next == '"') {
                    chars.next();
                } else {
                    in_quoted_identifier = false;
                }
            }
            continue;
        }
        if ch == '$' {
            if let Some(tag) = sql_dollar_quote_tag_at(input, idx) {
                let body_start = idx + tag.len();
                let close_relative = input[body_start..].find(tag)?;
                let close_end = body_start + close_relative + tag.len();
                while chars
                    .peek()
                    .is_some_and(|(next_idx, _)| *next_idx < close_end)
                {
                    chars.next();
                }
            }
            continue;
        }
        match ch {
            '\'' => in_quote = true,
            '"' => in_quoted_identifier = true,
            '(' => depth = depth.saturating_add(1),
            ')' => depth = depth.saturating_sub(1),
            ':' if depth == 0 && chars.peek().is_some_and(|(_, next)| *next == ':') => {
                return Some(idx);
            }
            _ => {}
        }
    }
    None
}

fn parenthesized_list_is_balanced(input: &str) -> bool {
    let mut depth = 0usize;
    let mut in_quote = false;
    let mut chars = input.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if !in_quote && ch == '$' {
            let Some(tag) = sql_dollar_quote_tag_at(input, idx) else {
                continue;
            };
            let body_start = idx + tag.len();
            let Some(close_relative) = input[body_start..].find(tag) else {
                return false;
            };
            let close_end = body_start + close_relative + tag.len();
            while chars
                .peek()
                .is_some_and(|(next_idx, _)| *next_idx < close_end)
            {
                chars.next();
            }
            continue;
        }
        if ch == '\'' {
            if in_quote && chars.peek().is_some_and(|(_, next)| *next == '\'') {
                chars.next();
                continue;
            }
            in_quote = !in_quote;
            continue;
        }
        if in_quote {
            continue;
        }
        match ch {
            '(' => depth = depth.saturating_add(1),
            ')' => {
                let Some(next_depth) = depth.checked_sub(1) else {
                    return false;
                };
                depth = next_depth;
            }
            _ => {}
        }
    }
    depth == 0 && !in_quote
}

pub(super) fn parse_supported_cursor_name(name: &str) -> Option<String> {
    if name.is_empty() {
        return None;
    }
    if let Some(quoted) = name.strip_prefix('"') {
        if !quoted.ends_with('"') {
            return None;
        }
        let inner = &quoted[..quoted.len() - 1];
        let mut decoded = String::with_capacity(inner.len());
        let mut chars = inner.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    decoded.push('"');
                } else {
                    return None;
                }
            } else {
                decoded.push(ch);
            }
        }
        if decoded.is_empty() {
            None
        } else {
            Some(decoded)
        }
    } else if name.split_whitespace().count() == 1 && !name.contains('"') {
        Some(name.to_ascii_lowercase())
    } else {
        None
    }
}
