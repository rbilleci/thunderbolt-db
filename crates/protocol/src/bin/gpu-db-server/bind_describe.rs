use super::{
    canonical_sql, column_for_sql_type, column_type_oid, int8_column, numeric_column,
    parse_sql_execute, split_sql_csv, sql_dollar_quote_tag_at, BindParameterError, CatalogColumn,
    Column, ErrorField, PreparedQuery, PreparedStatement, Session, Table,
};
use gpu_db_protocol::{parse_command, Command, ParseError, SelectProjection, SqlType};

pub(super) fn bind_query_parameters(
    query: &PreparedQuery,
    parameters: &[Option<String>],
) -> Result<String, BindParameterError> {
    if expected_parameter_count(query) != parameters.len() {
        return Err(BindParameterError::CountMismatch);
    }
    let mut bound = query.query.clone();
    for (idx, parameter) in parameters.iter().enumerate().rev() {
        let placeholder_idx = idx + 1;
        if !contains_unquoted_placeholder_index(&query.query, placeholder_idx) {
            continue;
        }
        let value = parameter
            .as_ref()
            .ok_or(BindParameterError::NullUnsupported)?;
        let placeholder = format!("${placeholder_idx}");
        let literal = encode_parameter_literal(
            value,
            query.parameter_type_oids.get(idx).copied().unwrap_or(0),
        )?;
        bound = replace_unquoted_placeholder(&bound, &placeholder, &literal);
    }
    Ok(bound)
}

pub(super) fn format_code_count_is_valid(count: usize, expected: usize) -> bool {
    matches!(count, 0 | 1) || count == expected
}

pub(super) fn expected_parameter_count(query: &PreparedQuery) -> usize {
    std::cmp::max(
        query.parameter_type_oids.len(),
        max_placeholder_index(&query.query),
    )
}

pub(super) fn resolve_prepared_parameter_type_oids(
    session: &Session,
    query: &str,
    explicit_oids: Vec<u32>,
) -> Vec<u32> {
    let inferred = infer_extended_parameter_type_oids(session, query);
    let max_count = std::cmp::max(
        explicit_oids.len(),
        inferred
            .as_ref()
            .map_or_else(|| max_placeholder_index(query), Vec::len),
    );
    if max_count == 0 {
        return Vec::new();
    }

    let mut resolved = vec![0; max_count];
    if let Some(inferred) = inferred {
        for (idx, oid) in inferred.into_iter().enumerate() {
            resolved[idx] = oid;
        }
    }
    for (idx, oid) in explicit_oids.into_iter().enumerate() {
        if oid != 0 {
            resolved[idx] = oid;
        }
    }
    resolved
}

fn infer_extended_parameter_type_oids(session: &Session, query: &str) -> Option<Vec<u32>> {
    if let Some((name, parameters)) = parse_sql_execute(query) {
        return infer_sql_execute_parameter_type_oids(session, &name, &parameters);
    }
    infer_select_parameter_type_oids(session, query)
        .or_else(|| infer_dml_parameter_type_oids(session, query))
}

fn infer_sql_execute_parameter_type_oids(
    session: &Session,
    name: &str,
    parameters: &[Option<String>],
) -> Option<Vec<u32>> {
    let Some(PreparedStatement::Sql(prepared)) = session.prepared.get(name) else {
        return Some(Vec::new());
    };
    if expected_parameter_count(prepared) != parameters.len() {
        return None;
    }
    let max_idx = parameters
        .iter()
        .filter_map(|parameter| {
            parameter
                .as_deref()
                .and_then(sql_execute_argument_placeholder_index)
        })
        .max()
        .unwrap_or(0);
    if max_idx == 0 {
        return Some(Vec::new());
    }

    let mut oids = vec![0; max_idx];
    for (sql_parameter_idx, parameter) in parameters.iter().enumerate() {
        let Some(extended_parameter_idx) = parameter
            .as_deref()
            .and_then(sql_execute_argument_placeholder_index)
        else {
            continue;
        };
        if extended_parameter_idx == 0 || extended_parameter_idx > oids.len() {
            continue;
        }
        let oid = prepared
            .parameter_type_oids
            .get(sql_parameter_idx)
            .copied()
            .unwrap_or(0);
        let existing = &mut oids[extended_parameter_idx - 1];
        if *existing != 0 && oid != 0 && *existing != oid {
            return None;
        }
        *existing = oid;
    }
    Some(oids)
}

pub(super) fn sql_execute_parameter_type_mapping_error(
    session: &Session,
    query: &str,
    explicit_oids: &[u32],
) -> Option<ErrorField> {
    let (name, parameters) = parse_sql_execute(query)?;
    let Some(PreparedStatement::Sql(prepared)) = session.prepared.get(&name) else {
        return None;
    };
    if expected_parameter_count(prepared) != parameters.len() {
        return None;
    }

    let max_idx = parameters
        .iter()
        .filter_map(|parameter| {
            parameter
                .as_deref()
                .and_then(sql_execute_argument_placeholder_index)
        })
        .max()
        .unwrap_or(0);
    let mut inferred_oids = vec![0; max_idx];
    for (sql_parameter_idx, parameter) in parameters.iter().enumerate() {
        let Some(extended_parameter_idx) = parameter
            .as_deref()
            .and_then(sql_execute_argument_placeholder_index)
        else {
            continue;
        };
        if extended_parameter_idx == 0 || extended_parameter_idx > inferred_oids.len() {
            continue;
        }
        let oid = prepared
            .parameter_type_oids
            .get(sql_parameter_idx)
            .copied()
            .unwrap_or(0);
        let inferred = &mut inferred_oids[extended_parameter_idx - 1];
        if *inferred != 0 && oid != 0 && *inferred != oid {
            return Some(sql_execute_parameter_type_conflict_error());
        }
        *inferred = oid;
    }

    for (idx, explicit_oid) in explicit_oids.iter().copied().enumerate() {
        if explicit_oid == 0 {
            continue;
        }
        let inferred_oid = inferred_oids.get(idx).copied().unwrap_or(0);
        if inferred_oid != 0 && inferred_oid != explicit_oid {
            return Some(sql_execute_parameter_type_conflict_error());
        }
    }
    None
}

fn sql_execute_parameter_type_conflict_error() -> ErrorField {
    ErrorField {
        code: "42P08",
        message: "inconsistent parameter types for SQL EXECUTE placeholder",
        position: None,
    }
}

pub(super) fn sql_execute_argument_placeholder_index(value: &str) -> Option<usize> {
    let digits = value.strip_prefix('$')?;
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let idx = digits.parse::<usize>().ok()?;
    (idx > 0).then_some(idx)
}

fn infer_select_parameter_type_oids(session: &Session, query: &str) -> Option<Vec<u32>> {
    let canonical = canonical_sql(query);
    let max_idx = max_placeholder_index(&canonical);
    if max_idx == 0 {
        return Some(Vec::new());
    }
    let (table_name, _) = describe_parameterized_select_shape(&canonical)?;
    let table = session.tables.get(&table_name)?;
    let mut oids = vec![0; max_idx];

    let where_clause = select_where_clause(&canonical);
    for column in &table.columns {
        for op in ["=", "<=", ">=", "<", ">"] {
            let needle = format!("{} {op} $", column.def.name);
            let mut rest = where_clause.as_str();
            while let Some(pos) = rest.find(&needle) {
                let digits = rest[pos + needle.len()..]
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .collect::<String>();
                if let Ok(idx) = digits.parse::<usize>() {
                    if idx > 0 && idx <= oids.len() {
                        oids[idx - 1] = column_type_oid(session, column);
                    }
                }
                rest = &rest[pos + needle.len()..];
            }
            let suffix = format!(" {op} {}", column.def.name);
            let mut rest = where_clause.as_str();
            while let Some(pos) = rest.find('$') {
                let digits = rest[pos + 1..]
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .collect::<String>();
                if !digits.is_empty() && rest[pos + 1 + digits.len()..].starts_with(&suffix) {
                    if let Ok(idx) = digits.parse::<usize>() {
                        if idx > 0 && idx <= oids.len() {
                            oids[idx - 1] = column_type_oid(session, column);
                        }
                    }
                }
                rest = &rest[pos + 1..];
            }
        }
    }

    if let Some(idx) = select_limit_placeholder_index(&canonical) {
        if idx > 0 && idx <= oids.len() {
            oids[idx - 1] = SqlType::Int4.postgres_oid();
        }
    }
    if let Some(idx) = select_offset_placeholder_index(&canonical) {
        if idx > 0 && idx <= oids.len() {
            oids[idx - 1] = SqlType::Int4.postgres_oid();
        }
    }

    Some(oids)
}

fn infer_dml_parameter_type_oids(session: &Session, query: &str) -> Option<Vec<u32>> {
    let canonical = canonical_sql(query);
    let max_idx = max_placeholder_index(&canonical);
    if max_idx == 0 {
        return Some(Vec::new());
    }
    let dummy_query = replace_parameter_placeholders_with_dummy_literals(&canonical);
    let command = parse_command(&dummy_query).ok()?;
    let mut oids = vec![0; max_idx];
    match command {
        Command::Insert(insert) => {
            let table = session.tables.get(&insert.table)?;
            let indexes = if insert.columns.is_empty() {
                (0..table.columns.len()).collect::<Vec<_>>()
            } else {
                insert
                    .columns
                    .iter()
                    .map(|column| {
                        table
                            .columns
                            .iter()
                            .position(|candidate| candidate.def.name == *column)
                    })
                    .collect::<Option<Vec<_>>>()?
            };
            for (value_idx, value) in insert_value_fragments(&canonical)?.into_iter().enumerate() {
                let column_idx = indexes.get(value_idx % indexes.len()).copied()?;
                assign_fragment_placeholder_oids(
                    value,
                    table.columns[column_idx].def.ty.postgres_oid(),
                    &mut oids,
                );
            }
        }
        Command::Update(update) => {
            let table = session.tables.get(&update.table)?;
            for assignment in update_assignment_fragments(&canonical)? {
                let (column, value) = assignment.split_once('=')?;
                let column = column.trim();
                let column = table
                    .columns
                    .iter()
                    .find(|candidate| candidate.def.name == column)?;
                assign_fragment_placeholder_oids(
                    value,
                    column_type_oid(session, column),
                    &mut oids,
                );
            }
            assign_filter_placeholder_oids(table, &dml_where_clause(&canonical), &mut oids);
        }
        Command::Delete(delete) => {
            let table = session.tables.get(&delete.table)?;
            assign_filter_placeholder_oids(table, &dml_where_clause(&canonical), &mut oids);
        }
        _ => return None,
    }
    Some(oids)
}

pub(super) fn is_supported_extended_dml(session: &Session, query: &str) -> bool {
    let canonical = canonical_sql(query);
    let dummy_query = replace_parameter_placeholders_with_dummy_literals(&canonical);
    match parse_command(&dummy_query) {
        Ok(Command::Insert(insert)) => session.tables.get(&insert.table).is_some_and(|table| {
            if insert.rows.is_empty() {
                return false;
            }
            let expected = if insert.columns.is_empty() {
                table.columns.len()
            } else {
                if insert.columns.iter().any(|column| {
                    !table
                        .columns
                        .iter()
                        .any(|candidate| candidate.def.name == *column)
                }) {
                    return false;
                }
                insert.columns.len()
            };
            insert.rows.iter().all(|row| row.len() == expected)
        }),
        Ok(Command::Update(update)) => session.tables.get(&update.table).is_some_and(|table| {
            !update.assignments.is_empty()
                && update.assignments.iter().all(|assignment| {
                    table
                        .columns
                        .iter()
                        .any(|column| column.def.name == assignment.column)
                })
                && update.filter.is_some()
        }),
        Ok(Command::Delete(delete)) => {
            session.tables.contains_key(&delete.table) && delete.filter.is_some()
        }
        _ => false,
    }
}

fn insert_value_fragments(canonical: &str) -> Option<Vec<&str>> {
    let values_pos = canonical.find(" values ")?;
    let mut tail = canonical[values_pos + " values ".len()..].trim();
    let mut values = Vec::new();
    loop {
        let open = tail.find('(')?;
        if !tail[..open].trim().is_empty() {
            return None;
        }
        let close = matching_paren_index(tail, open)?;
        values.extend(split_sql_csv(&tail[open + 1..close])?);
        tail = tail[close + 1..].trim_start();
        if tail.is_empty() {
            break;
        }
        tail = tail.strip_prefix(',')?.trim_start();
    }
    Some(values)
}

fn update_assignment_fragments(canonical: &str) -> Option<Vec<&str>> {
    let set_pos = canonical.find(" set ")?;
    let where_pos = canonical[set_pos + " set ".len()..].find(" where ")? + set_pos + " set ".len();
    split_sql_csv(canonical[set_pos + " set ".len()..where_pos].trim())
}

fn dml_where_clause(canonical: &str) -> String {
    let Some(where_pos) = canonical.find(" where ") else {
        return String::new();
    };
    canonical[where_pos + " where ".len()..].to_string()
}

fn assign_filter_placeholder_oids(table: &Table, where_clause: &str, oids: &mut [u32]) {
    for column in &table.columns {
        for op in ["=", "<=", ">=", "<", ">"] {
            let needle = format!("{} {op} $", column.def.name);
            let mut rest = where_clause;
            while let Some(pos) = rest.find(&needle) {
                assign_placeholder_digits_oid(
                    &rest[pos + needle.len()..],
                    column.def.ty.postgres_oid(),
                    oids,
                );
                rest = &rest[pos + needle.len()..];
            }
            let suffix = format!(" {op} {}", column.def.name);
            let mut rest = where_clause;
            while let Some(pos) = rest.find('$') {
                let digits = rest[pos + 1..]
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .collect::<String>();
                if !digits.is_empty() && rest[pos + 1 + digits.len()..].starts_with(&suffix) {
                    assign_placeholder_index_oid(&digits, column.def.ty.postgres_oid(), oids);
                }
                rest = &rest[pos + 1..];
            }
        }
    }
}

fn assign_fragment_placeholder_oids(fragment: &str, oid: u32, oids: &mut [u32]) {
    let mut rest = fragment;
    while let Some(pos) = rest.find('$') {
        assign_placeholder_digits_oid(&rest[pos + 1..], oid, oids);
        rest = &rest[pos + 1..];
    }
}

fn assign_placeholder_digits_oid(rest: &str, oid: u32, oids: &mut [u32]) {
    let digits = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    assign_placeholder_index_oid(&digits, oid, oids);
}

fn assign_placeholder_index_oid(digits: &str, oid: u32, oids: &mut [u32]) {
    if let Ok(idx) = digits.parse::<usize>() {
        if idx > 0 && idx <= oids.len() {
            oids[idx - 1] = oid;
        }
    }
}

fn matching_paren_index(input: &str, open: usize) -> Option<usize> {
    let mut chars = input[open..].char_indices().peekable();
    let mut depth = 0usize;
    let mut in_quote = false;
    while let Some((relative_idx, ch)) = chars.next() {
        if ch == '\'' {
            if in_quote && matches!(chars.peek(), Some((_, '\''))) {
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
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(open + relative_idx);
                }
            }
            _ => {}
        }
    }
    None
}

fn select_where_clause(canonical: &str) -> String {
    let Some(where_pos) = canonical.find(" where ") else {
        return String::new();
    };
    let after_where = &canonical[where_pos + " where ".len()..];
    let end = [" order by ", " limit ", " offset "]
        .into_iter()
        .filter_map(|marker| after_where.find(marker))
        .min()
        .unwrap_or(after_where.len());
    after_where[..end].to_string()
}

fn select_limit_placeholder_index(canonical: &str) -> Option<usize> {
    select_clause_placeholder_index(canonical, " limit ")
}

fn select_offset_placeholder_index(canonical: &str) -> Option<usize> {
    select_clause_placeholder_index(canonical, " offset ")
}

fn select_clause_placeholder_index(canonical: &str, clause: &str) -> Option<usize> {
    let clause_pos = canonical.rfind(clause)?;
    let after_clause = canonical[clause_pos + clause.len()..].trim();
    let rest = after_clause.strip_prefix('$')?;
    rest.chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .parse::<usize>()
        .ok()
}

pub(super) fn max_placeholder_index(query: &str) -> usize {
    let mut max_index = 0;
    for token in unquoted_sql_fragments(query) {
        let mut chars = token.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch != '$' {
                continue;
            }

            let mut value = 0usize;
            let mut saw_digit = false;
            while let Some(digit) = chars.peek().copied() {
                let Some(next) = digit.to_digit(10) else {
                    break;
                };
                saw_digit = true;
                value = value.saturating_mul(10).saturating_add(next as usize);
                chars.next();
            }
            if saw_digit {
                max_index = max_index.max(value);
            }
        }
    }
    max_index
}

pub(super) fn contains_zero_placeholder(query: &str) -> bool {
    unquoted_sql_fragments(query)
        .into_iter()
        .any(fragment_contains_zero_placeholder)
}

fn contains_unquoted_placeholder_index(query: &str, target_index: usize) -> bool {
    unquoted_sql_fragments(query)
        .into_iter()
        .any(|fragment| fragment_contains_placeholder_index(fragment, target_index))
}

fn fragment_contains_placeholder_index(fragment: &str, target_index: usize) -> bool {
    let mut chars = fragment.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            continue;
        }

        let mut value = 0usize;
        let mut saw_digit = false;
        while let Some(digit) = chars.peek().copied() {
            let Some(next) = digit.to_digit(10) else {
                break;
            };
            saw_digit = true;
            value = value.saturating_mul(10).saturating_add(next as usize);
            chars.next();
        }
        if saw_digit && value == target_index {
            return true;
        }
    }
    false
}

fn fragment_contains_zero_placeholder(fragment: &str) -> bool {
    let mut chars = fragment.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            continue;
        }

        let mut saw_digit = false;
        let mut all_zero = true;
        while let Some(digit) = chars.peek().copied() {
            if !digit.is_ascii_digit() {
                break;
            }
            saw_digit = true;
            if digit != '0' {
                all_zero = false;
            }
            chars.next();
        }
        if saw_digit && all_zero {
            return true;
        }
    }
    false
}

fn replace_unquoted_placeholder(query: &str, placeholder: &str, literal: &str) -> String {
    let Some(target_index) = placeholder.strip_prefix('$') else {
        return query.to_string();
    };
    let mut rewritten = String::with_capacity(query.len());
    let mut token_start = 0;
    let mut chars = query.char_indices().peekable();
    let mut in_quote = false;

    while let Some((idx, ch)) = chars.next() {
        if ch != '\'' {
            continue;
        }

        if in_quote && matches!(chars.peek(), Some((_, '\''))) {
            chars.next();
            continue;
        }

        if in_quote {
            rewritten.push_str(&query[token_start..=idx]);
            token_start = idx + ch.len_utf8();
            in_quote = false;
        } else {
            push_placeholder_replaced_fragment(
                &mut rewritten,
                &query[token_start..idx],
                target_index,
                literal,
            );
            token_start = idx;
            in_quote = true;
        }
    }

    if in_quote {
        rewritten.push_str(&query[token_start..]);
    } else {
        push_placeholder_replaced_fragment(
            &mut rewritten,
            &query[token_start..],
            target_index,
            literal,
        );
    }

    rewritten
}

fn push_placeholder_replaced_fragment(
    rewritten: &mut String,
    fragment: &str,
    target_index: &str,
    literal: &str,
) {
    let target_index = target_index.parse::<usize>().ok();
    let mut token_start = 0;
    let mut chars = fragment.char_indices().peekable();

    while let Some((idx, ch)) = chars.next() {
        if ch != '$' {
            continue;
        }

        let digit_start = idx + ch.len_utf8();
        let mut digit_end = digit_start;
        while let Some((digit_idx, digit)) = chars.peek().copied() {
            if !digit.is_ascii_digit() {
                break;
            }
            digit_end = digit_idx + digit.len_utf8();
            chars.next();
        }

        if digit_end == digit_start {
            continue;
        }

        rewritten.push_str(&fragment[token_start..idx]);
        let placeholder_index = fragment[digit_start..digit_end].parse::<usize>().ok();
        if placeholder_index == target_index {
            rewritten.push_str(literal);
        } else {
            rewritten.push_str(&fragment[idx..digit_end]);
        }
        token_start = digit_end;
    }

    rewritten.push_str(&fragment[token_start..]);
}

fn unquoted_sql_fragments(query: &str) -> Vec<&str> {
    let mut fragments = Vec::new();
    let mut token_start = 0;
    let mut chars = query.char_indices().peekable();
    let mut in_quote = false;

    while let Some((idx, ch)) = chars.next() {
        if ch != '\'' {
            continue;
        }
        if in_quote && matches!(chars.peek(), Some((_, '\''))) {
            chars.next();
            continue;
        }

        if in_quote {
            token_start = idx + ch.len_utf8();
            in_quote = false;
        } else {
            fragments.push(&query[token_start..idx]);
            in_quote = true;
        }
    }

    if !in_quote {
        fragments.push(&query[token_start..]);
    }
    fragments
}

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

fn replace_parameter_placeholders_with_dummy_literals(query: &str) -> String {
    let mut rewritten = String::with_capacity(query.len());
    let mut chars = query.char_indices().peekable();
    let mut in_quote = false;
    while let Some((_, ch)) = chars.next() {
        if ch == '\'' {
            rewritten.push(ch);
            if in_quote && matches!(chars.peek(), Some((_, '\''))) {
                if let Some((_, escaped)) = chars.next() {
                    rewritten.push(escaped);
                }
                continue;
            }
            in_quote = !in_quote;
            continue;
        }
        if ch != '$' {
            rewritten.push(ch);
            continue;
        }
        if in_quote {
            rewritten.push(ch);
            continue;
        }

        let mut saw_digit = false;
        while let Some((_, digit)) = chars.peek().copied() {
            if !digit.is_ascii_digit() {
                break;
            }
            saw_digit = true;
            chars.next();
        }

        if saw_digit {
            rewritten.push('1');
        } else {
            rewritten.push('$');
        }
    }
    rewritten
}

fn encode_parameter_literal(value: &str, type_oid: u32) -> Result<String, BindParameterError> {
    match type_oid {
        23 => value
            .parse::<i32>()
            .map(|parsed| parsed.to_string())
            .map_err(|_| BindParameterError::InvalidTextRepresentation {
                oid: type_oid,
                value: value.to_string(),
            }),
        25 => Ok(sql_quote_text(value)),
        0 if value.parse::<i32>().is_ok() => Ok(value.to_string()),
        0 => Ok(sql_quote_text(value)),
        _ => Err(BindParameterError::InvalidTextRepresentation {
            oid: type_oid,
            value: value.to_string(),
        }),
    }
}

fn sql_quote_text(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

pub(super) fn bind_parameter_error_field(error: BindParameterError) -> ErrorField {
    match error {
        BindParameterError::CountMismatch => ErrorField {
            code: "08P01",
            message: "bound parameter count does not match prepared statement",
            position: None,
        },
        BindParameterError::NullUnsupported => ErrorField {
            code: "0A000",
            message: "NULL extended-query parameters are not supported",
            position: None,
        },
        BindParameterError::InvalidTextRepresentation { oid, value } => {
            let message = Box::leak(
                format!("invalid input syntax for parameter type oid {oid}: \"{value}\"")
                    .into_boxed_str(),
            );
            ErrorField {
                code: "22P02",
                message,
                position: None,
            }
        }
    }
}

pub(super) fn sql_execute_parameter_error_field(error: BindParameterError) -> ErrorField {
    match error {
        BindParameterError::NullUnsupported => ErrorField {
            code: "0A000",
            message: "NULL SQL EXECUTE parameters are not supported",
            position: None,
        },
        error => bind_parameter_error_field(error),
    }
}

pub(super) fn negative_limit_error_field() -> ErrorField {
    ErrorField {
        code: "2201W",
        message: "LIMIT must not be negative",
        position: None,
    }
}

pub(super) fn negative_offset_error_field() -> ErrorField {
    ErrorField {
        code: "2201X",
        message: "OFFSET must not be negative",
        position: None,
    }
}

pub(super) fn describe_query_columns(session: &Session, query: &str) -> Option<Vec<Column>> {
    let (table_name, projection) = match parse_command(query).ok() {
        Some(Command::Select(select)) => (select.table, select.projection),
        _ => describe_parameterized_select_shape(query)?,
    };
    let table = session.tables.get(&table_name)?;
    match projection {
        // GroupedAggregates and a bare COUNT(DISTINCT) are produced only by the engine Expr parser,
        // not the wire-protocol parser.
        SelectProjection::GroupedAggregates { .. } | SelectProjection::CountDistinct { .. } => None,
        SelectProjection::All => Some(
            table
                .columns
                .iter()
                .map(column_def_to_result_column)
                .collect(),
        ),
        SelectProjection::Columns(columns) => {
            let mut selected = Vec::with_capacity(columns.len());
            for column in columns {
                let column = table
                    .columns
                    .iter()
                    .find(|candidate| candidate.def.name == column)?;
                selected.push(column_def_to_result_column(column));
            }
            Some(selected)
        }
        SelectProjection::CountAll => Some(vec![int8_column("count")]),
        SelectProjection::Sum { column } => {
            let sum_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == column)?;
            matches!(sum_column.def.ty, SqlType::Int4).then(|| vec![int8_column("sum")])
        }
        SelectProjection::Avg { column } => {
            let avg_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == column)?;
            matches!(avg_column.def.ty, SqlType::Int4).then(|| vec![numeric_column("avg")])
        }
        SelectProjection::GroupedCount { column } => {
            let group_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == column)?;
            Some(vec![
                column_def_to_result_column(group_column),
                int8_column("count"),
            ])
        }
        SelectProjection::GroupedSum {
            group_column,
            sum_column,
        } => {
            let group_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == group_column)?;
            let sum_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == sum_column)?;
            matches!(sum_column.def.ty, SqlType::Int4).then(|| {
                vec![
                    column_def_to_result_column(group_column),
                    int8_column("sum"),
                ]
            })
        }
        SelectProjection::GroupedAvg {
            group_column,
            avg_column,
        } => {
            let group_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == group_column)?;
            let avg_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == avg_column)?;
            matches!(avg_column.def.ty, SqlType::Int4).then(|| {
                vec![
                    column_def_to_result_column(group_column),
                    numeric_column("avg"),
                ]
            })
        }
        SelectProjection::Min { column } => {
            let value_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == column)?;
            Some(vec![column_def_to_result_column_with_name(
                value_column,
                "min",
            )])
        }
        SelectProjection::Max { column } => {
            let value_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == column)?;
            Some(vec![column_def_to_result_column_with_name(
                value_column,
                "max",
            )])
        }
        SelectProjection::GroupedMin {
            group_column,
            min_column,
        } => {
            let group_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == group_column)?;
            let value_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == min_column)?;
            Some(vec![
                column_def_to_result_column(group_column),
                column_def_to_result_column_with_name(value_column, "min"),
            ])
        }
        SelectProjection::GroupedMax {
            group_column,
            max_column,
        } => {
            let group_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == group_column)?;
            let value_column = table
                .columns
                .iter()
                .find(|candidate| candidate.def.name == max_column)?;
            Some(vec![
                column_def_to_result_column(group_column),
                column_def_to_result_column_with_name(value_column, "max"),
            ])
        }
    }
}

fn column_def_to_result_column(column: &CatalogColumn) -> Column {
    column_def_to_result_column_with_name(column, &column.def.name)
}

fn column_def_to_result_column_with_name(column: &CatalogColumn, name: &str) -> Column {
    column_for_sql_type(column.def.ty, name)
}

fn describe_parameterized_select_shape(query: &str) -> Option<(String, SelectProjection)> {
    let canonical = canonical_sql(query);
    let dummy_query = replace_parameter_placeholders_with_dummy_literals(&canonical);
    let select = match parse_command(&dummy_query) {
        Ok(Command::Select(select)) => select,
        Err(ParseError::NegativeLimit) => {
            let describe_query = replace_negative_limit_with_zero(&dummy_query)?;
            let Ok(Command::Select(select)) = parse_command(&describe_query) else {
                return None;
            };
            select
        }
        Err(ParseError::NegativeOffset) => {
            let describe_query = replace_negative_offset_with_zero(&dummy_query)?;
            let Ok(Command::Select(select)) = parse_command(&describe_query) else {
                return None;
            };
            select
        }
        Ok(_) | Err(_) => return None,
    };
    Some((select.table, select.projection))
}

fn replace_negative_limit_with_zero(query: &str) -> Option<String> {
    replace_negative_clause_value_with_zero(query, " limit ")
}

fn replace_negative_offset_with_zero(query: &str) -> Option<String> {
    replace_negative_clause_value_with_zero(query, " offset ")
}

fn replace_negative_clause_value_with_zero(query: &str, clause: &str) -> Option<String> {
    let clause_pos = query.rfind(clause)?;
    let after_clause = &query[clause_pos + clause.len()..];
    let trimmed = after_clause.trim_start();
    let minus_len = trimmed.strip_prefix('-')?.len();
    let digit_count = trimmed[1..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return None;
    }
    let leading_ws_len = after_clause.len() - trimmed.len();
    let start = clause_pos + clause.len() + leading_ws_len;
    let end = start + (trimmed.len() - minus_len) + digit_count;
    let mut rewritten = String::with_capacity(query.len());
    rewritten.push_str(&query[..start]);
    rewritten.push('0');
    rewritten.push_str(&query[end..]);
    Some(rewritten)
}

#[cfg(test)]
pub(super) fn test_replace_unquoted_placeholder(
    query: &str,
    placeholder: &str,
    literal: &str,
) -> String {
    replace_unquoted_placeholder(query, placeholder, literal)
}

#[cfg(test)]
pub(super) fn test_replace_parameter_placeholders_with_dummy_literals(query: &str) -> String {
    replace_parameter_placeholders_with_dummy_literals(query)
}

#[cfg(test)]
pub(super) fn test_describe_parameterized_select_shape(
    query: &str,
) -> Option<(String, SelectProjection)> {
    describe_parameterized_select_shape(query)
}
