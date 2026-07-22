//! Relational dispatch and schema, table, index, and DML parsing.

use super::{
    acl, find_char_outside_quotes, find_keyword_outside_quotes, find_matching_paren,
    normalize_identifier, normalize_relation_identifier, parse_alter_role, parse_comment_on,
    parse_create_database, parse_create_domain, parse_create_extension, parse_create_function,
    parse_create_materialized_view, parse_create_publication, parse_create_role,
    parse_create_sequence, parse_create_subscription, parse_create_tablespace, parse_create_view,
    parse_drop_database, parse_drop_domain, parse_drop_extension, parse_drop_function,
    parse_drop_materialized_view, parse_drop_publication, parse_drop_role, parse_drop_sequence,
    parse_drop_subscription, parse_drop_tablespace, parse_drop_view,
    parse_refresh_materialized_view, parse_rename_database, parse_rename_function,
    parse_rename_materialized_view, parse_rename_sequence, parse_rename_tablespace,
    parse_rename_view, parse_select, parse_select_filter, parse_select_filter_groups,
    parse_select_function, parse_select_literal, parse_select_pg_dump_builtin,
    parse_sequence_regclass_arg, parse_sequence_value_function, parse_sql_value,
    parse_supported_sql_type_name, parse_typed_value_from_str, split_csv,
    strip_keyword_prefix_case_insensitive, strip_keyword_suffix_case_insensitive,
    AddCheckConstraint, AddColumn, AddForeignKey, AddPrimaryKey, AddUniqueConstraint,
    AlterColumnDefault, CheckConstraint, ColumnDef, ColumnDefault, Command, CreateIndex,
    CreateSchema, CreateTable, Delete, DropColumn, DropConstraint, DropIndex, DropSchema,
    DropTable, Insert, ParseError, PrimaryKey, RenameColumn, RenameConstraint, RenameIndex,
    RenameTable, SelectFilter, SelectFilterOp, SqlType, SqlValue, TruncateTable, UniqueConstraint,
    Update, UpdateAssignment,
};

pub(super) fn parse_relational_command(
    input: &str,
    allow_catalog_schemas: bool,
) -> Option<Result<Command, ParseError>> {
    let first = input.split_whitespace().next()?;
    if first.eq_ignore_ascii_case("CREATE") {
        let second = input.split_whitespace().nth(1)?;
        if second.eq_ignore_ascii_case("TABLE") {
            return Some(parse_create_table(input).map(Command::CreateTable));
        }
        if second.eq_ignore_ascii_case("SCHEMA") {
            return Some(parse_create_schema(input).map(Command::CreateSchema));
        }
        if second.eq_ignore_ascii_case("DATABASE") {
            return Some(parse_create_database(input).map(Command::CreateDatabase));
        }
        if second.eq_ignore_ascii_case("TABLESPACE") {
            return Some(parse_create_tablespace(input).map(Command::CreateTablespace));
        }
        if second.eq_ignore_ascii_case("INDEX") || second.eq_ignore_ascii_case("UNIQUE") {
            return Some(parse_create_index(input).map(Command::CreateIndex));
        }
        if second.eq_ignore_ascii_case("VIEW") {
            return Some(parse_create_view(input).map(Command::CreateView));
        }
        if second.eq_ignore_ascii_case("MATERIALIZED") {
            let third = input.split_whitespace().nth(2)?;
            if third.eq_ignore_ascii_case("VIEW") {
                return Some(
                    parse_create_materialized_view(input).map(Command::CreateMaterializedView),
                );
            }
        }
        if second.eq_ignore_ascii_case("FUNCTION") {
            return Some(parse_create_function(input).map(Command::CreateFunction));
        }
        if second.eq_ignore_ascii_case("SEQUENCE") {
            return Some(parse_create_sequence(input).map(Command::CreateSequence));
        }
        if second.eq_ignore_ascii_case("DOMAIN") {
            return Some(parse_create_domain(input).map(Command::CreateDomain));
        }
        if second.eq_ignore_ascii_case("PUBLICATION") {
            return Some(parse_create_publication(input).map(Command::CreatePublication));
        }
        if second.eq_ignore_ascii_case("SUBSCRIPTION") {
            return Some(parse_create_subscription(input).map(Command::CreateSubscription));
        }
        if second.eq_ignore_ascii_case("EXTENSION") {
            return Some(parse_create_extension(input).map(Command::CreateExtension));
        }
        if second.eq_ignore_ascii_case("ROLE") || second.eq_ignore_ascii_case("USER") {
            return Some(parse_create_role(input).map(Command::CreateRole));
        }
        if second.eq_ignore_ascii_case("OR") {
            let third = input.split_whitespace().nth(2)?;
            let fourth = input.split_whitespace().nth(3)?;
            if third.eq_ignore_ascii_case("REPLACE") && fourth.eq_ignore_ascii_case("VIEW") {
                return Some(parse_create_view(input).map(Command::CreateView));
            }
        }
        return Some(Err(ParseError::InvalidRelationalSql));
    }
    if first.eq_ignore_ascii_case("DROP") {
        let second = input.split_whitespace().nth(1)?;
        if second.eq_ignore_ascii_case("TABLE") {
            return Some(parse_drop_table(input).map(Command::DropTable));
        }
        if second.eq_ignore_ascii_case("SCHEMA") {
            return Some(parse_drop_schema(input).map(Command::DropSchema));
        }
        if second.eq_ignore_ascii_case("DATABASE") {
            return Some(parse_drop_database(input).map(Command::DropDatabase));
        }
        if second.eq_ignore_ascii_case("TABLESPACE") {
            return Some(parse_drop_tablespace(input).map(Command::DropTablespace));
        }
        if second.eq_ignore_ascii_case("INDEX") {
            return Some(parse_drop_index(input).map(Command::DropIndex));
        }
        if second.eq_ignore_ascii_case("VIEW") {
            return Some(parse_drop_view(input).map(Command::DropView));
        }
        if second.eq_ignore_ascii_case("MATERIALIZED") {
            let third = input.split_whitespace().nth(2)?;
            if third.eq_ignore_ascii_case("VIEW") {
                return Some(
                    parse_drop_materialized_view(input).map(Command::DropMaterializedView),
                );
            }
        }
        if second.eq_ignore_ascii_case("FUNCTION") {
            return Some(parse_drop_function(input).map(Command::DropFunction));
        }
        if second.eq_ignore_ascii_case("SEQUENCE") {
            return Some(parse_drop_sequence(input).map(Command::DropSequence));
        }
        if second.eq_ignore_ascii_case("DOMAIN") {
            return Some(parse_drop_domain(input).map(Command::DropDomain));
        }
        if second.eq_ignore_ascii_case("EXTENSION") {
            return Some(parse_drop_extension(input).map(Command::DropExtension));
        }
        if second.eq_ignore_ascii_case("PUBLICATION") {
            return Some(parse_drop_publication(input).map(Command::DropPublication));
        }
        if second.eq_ignore_ascii_case("SUBSCRIPTION") {
            return Some(parse_drop_subscription(input).map(Command::DropSubscription));
        }
        if second.eq_ignore_ascii_case("ROLE") || second.eq_ignore_ascii_case("USER") {
            return Some(parse_drop_role(input).map(Command::DropRole));
        }
        return Some(Err(ParseError::InvalidRelationalSql));
    }
    if first.eq_ignore_ascii_case("TRUNCATE") {
        return Some(parse_truncate_table(input).map(Command::TruncateTable));
    }
    if first.eq_ignore_ascii_case("REFRESH") {
        return Some(parse_refresh_materialized_view(input).map(Command::RefreshMaterializedView));
    }
    if let Some(command) = acl::parse_acl_command(input) {
        return Some(command);
    }
    if first.eq_ignore_ascii_case("ALTER") {
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("DATABASE"))
        {
            return Some(parse_rename_database(input).map(Command::RenameDatabase));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("TABLESPACE"))
        {
            return Some(parse_rename_tablespace(input).map(Command::RenameTablespace));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("ROLE"))
        {
            return Some(parse_alter_role(input));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("INDEX"))
        {
            return Some(parse_rename_index(input).map(Command::RenameIndex));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("MATERIALIZED"))
        {
            return Some(
                parse_rename_materialized_view(input).map(Command::RenameMaterializedView),
            );
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("VIEW"))
        {
            return Some(parse_rename_view(input).map(Command::RenameView));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("FUNCTION"))
        {
            return Some(parse_rename_function(input).map(Command::RenameFunction));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("SEQUENCE"))
        {
            return Some(parse_rename_sequence(input).map(Command::RenameSequence));
        }
        if find_keyword_outside_quotes(input, "RENAME").is_some() {
            if parse_rename_constraint(input).is_ok() {
                return Some(parse_rename_constraint(input).map(Command::RenameConstraint));
            }
            if parse_rename_table(input).is_ok() {
                return Some(parse_rename_table(input).map(Command::RenameTable));
            }
            return Some(parse_rename_column(input).map(Command::RenameColumn));
        }
        if find_keyword_outside_quotes(input, "DROP").is_some()
            && find_keyword_outside_quotes(input, "DEFAULT").is_none()
        {
            if parse_drop_column(input).is_ok() {
                return Some(parse_drop_column(input).map(Command::DropColumn));
            }
            return Some(parse_drop_table_constraint(input).map(Command::DropConstraint));
        }
        if find_keyword_outside_quotes(input, "ADD").is_some() {
            return Some(parse_alter_table_add(input));
        }
        return Some(parse_alter_column_default(input).map(Command::AlterColumnDefault));
    }
    if first.eq_ignore_ascii_case("COMMENT") {
        return Some(parse_comment_on(input).map(Command::CommentOn));
    }
    if first.eq_ignore_ascii_case("INSERT") {
        return Some(parse_insert(input).map(Command::Insert));
    }
    if first.eq_ignore_ascii_case("UPDATE")
        && find_keyword_outside_quotes(input, "SET").is_some()
        && find_keyword_outside_quotes(input, "WHERE").is_some()
    {
        return Some(parse_update(input).map(Command::Update));
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
        if let Ok(sequence_command) = parse_sequence_value_function(input) {
            return Some(Ok(sequence_command));
        }
        if let Ok(builtin_command) = parse_select_pg_dump_builtin(input) {
            return Some(Ok(builtin_command));
        }
        if let Ok(function_command) = parse_select_function(input) {
            return Some(Ok(function_command));
        }
        if let Ok(literal_command) = parse_select_literal(input) {
            return Some(Ok(literal_command));
        }
        return Some(parse_select(input, allow_catalog_schemas).map(Command::Select));
    }
    None
}

fn parse_alter_column_default(input: &str) -> Result<AlterColumnDefault, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let alter_pos =
        find_keyword_outside_quotes(rest, "ALTER").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..alter_pos].trim())?;
    let rest = rest[alter_pos + "ALTER".len()..].trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(rest);
    if let Some(set_pos) = find_keyword_outside_quotes(rest, "SET") {
        let column = normalize_identifier(rest[..set_pos].trim())?;
        let rest = strip_keyword_prefix_case_insensitive(
            rest[set_pos + "SET".len()..].trim_start(),
            "DEFAULT",
        )
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
        if rest.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(AlterColumnDefault {
            table,
            column,
            default: Some(parse_column_default_expr(rest, None)?),
        });
    }

    let drop_pos =
        find_keyword_outside_quotes(rest, "DROP").ok_or(ParseError::InvalidRelationalSql)?;
    let column = normalize_identifier(rest[..drop_pos].trim())?;
    let rest = strip_keyword_prefix_case_insensitive(
        rest[drop_pos + "DROP".len()..].trim_start(),
        "DEFAULT",
    )
    .ok_or(ParseError::InvalidRelationalSql)?
    .trim();
    if !rest.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(AlterColumnDefault {
        table,
        column,
        default: None,
    })
}

fn parse_add_primary_key(input: &str) -> Result<AddPrimaryKey, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let rest = strip_keyword_prefix_case_insensitive(rest, "PRIMARY")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let columns = parse_constraint_columns(rest)?;
    Ok(AddPrimaryKey {
        table,
        name,
        column: columns[0].clone(),
        columns,
    })
}

fn parse_add_unique_constraint(input: &str) -> Result<AddUniqueConstraint, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let rest = strip_keyword_prefix_case_insensitive(rest, "UNIQUE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let columns = parse_constraint_columns(rest)?;
    Ok(AddUniqueConstraint {
        table,
        name,
        column: columns[0].clone(),
        columns,
    })
}

fn parse_add_check_constraint(input: &str) -> Result<AddCheckConstraint, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let filter = parse_check_constraint_filter(rest)?;
    Ok(AddCheckConstraint {
        table,
        name,
        filter,
    })
}

fn parse_add_foreign_key(input: &str) -> Result<AddForeignKey, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let rest = strip_keyword_prefix_case_insensitive(rest, "FOREIGN")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(rest, open).ok_or(ParseError::InvalidRelationalSql)?;
    let columns = split_csv(&rest[open + 1..close])?;
    let [column] = columns.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let column = normalize_identifier(column.trim())?;
    let rest = rest[close + 1..].trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "REFERENCES")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(rest, open).ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !rest[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let referenced_table = normalize_relation_identifier(rest[..open].trim())?;
    let referenced_columns = split_csv(&rest[open + 1..close])?;
    let [referenced_column] = referenced_columns.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(AddForeignKey {
        table,
        name,
        column,
        referenced_table,
        referenced_column: normalize_identifier(referenced_column.trim())?,
    })
}

fn parse_add_table_constraint(input: &str) -> Result<Command, ParseError> {
    let (_, _, rest) = parse_alter_table_add_constraint(input)?;
    if strip_keyword_prefix_case_insensitive(rest, "PRIMARY").is_some() {
        return parse_add_primary_key(input).map(Command::AddPrimaryKey);
    }
    if strip_keyword_prefix_case_insensitive(rest, "UNIQUE").is_some() {
        return parse_add_unique_constraint(input).map(Command::AddUniqueConstraint);
    }
    if strip_keyword_prefix_case_insensitive(rest, "CHECK").is_some() {
        return parse_add_check_constraint(input).map(Command::AddCheckConstraint);
    }
    if strip_keyword_prefix_case_insensitive(rest, "FOREIGN").is_some() {
        return parse_add_foreign_key(input).map(Command::AddForeignKey);
    }
    Err(ParseError::InvalidRelationalSql)
}

fn parse_alter_table_add(input: &str) -> Result<Command, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let add_pos =
        find_keyword_outside_quotes(rest, "ADD").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..add_pos].trim())?;
    let add_tail = rest[add_pos + "ADD".len()..].trim_start();
    if strip_keyword_prefix_case_insensitive(add_tail, "CONSTRAINT").is_some() {
        return parse_add_table_constraint(input);
    }
    let column_tail = strip_keyword_prefix_case_insensitive(add_tail, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(add_tail);
    Ok(Command::AddColumn(AddColumn {
        table,
        column: parse_column_def(column_tail)?,
    }))
}

fn parse_drop_column(input: &str) -> Result<DropColumn, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let drop_pos =
        find_keyword_outside_quotes(rest, "DROP").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..drop_pos].trim())?;
    rest = rest[drop_pos + "DROP".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(rest);
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let columns = split_csv(rest)?;
    let [column] = columns.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropColumn {
        table,
        column: normalize_identifier(column.trim())?,
    })
}

fn parse_rename_table(input: &str) -> Result<RenameTable, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(remaining) = strip_keyword_prefix_case_insensitive(rest, "IF")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "EXISTS"))
    {
        rest = remaining.trim_start();
        true
    } else {
        false
    };
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_relation_identifier(rest[..rename_pos].trim())?;
    rest = rest[rename_pos + "RENAME".len()..].trim_start();
    let new_tail = strip_keyword_prefix_case_insensitive(rest, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if new_tail.is_empty()
        || find_keyword_outside_quotes(new_tail, "CASCADE").is_some()
        || find_keyword_outside_quotes(new_tail, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameTable {
        old_name,
        new_name: normalize_identifier(new_tail)?,
        if_exists,
    })
}

fn parse_rename_column(input: &str) -> Result<RenameColumn, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..rename_pos].trim())?;
    rest = rest[rename_pos + "RENAME".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(rest);
    let to_pos = find_keyword_outside_quotes(rest, "TO").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_identifier(rest[..to_pos].trim())?;
    let new_tail = rest[to_pos + "TO".len()..].trim();
    if new_tail.is_empty()
        || find_keyword_outside_quotes(new_tail, "CASCADE").is_some()
        || find_keyword_outside_quotes(new_tail, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameColumn {
        table,
        old_name,
        new_name: normalize_identifier(new_tail)?,
    })
}

fn parse_rename_constraint(input: &str) -> Result<RenameConstraint, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let table_if_exists = if let Some(remaining) = strip_keyword_prefix_case_insensitive(rest, "IF")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "EXISTS"))
    {
        rest = remaining.trim_start();
        true
    } else {
        false
    };
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..rename_pos].trim())?;
    rest = rest[rename_pos + "RENAME".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "CONSTRAINT")
        .map(str::trim_start)
        .ok_or(ParseError::InvalidRelationalSql)?;
    let to_pos = find_keyword_outside_quotes(rest, "TO").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_identifier(rest[..to_pos].trim())?;
    let new_tail = rest[to_pos + "TO".len()..].trim();
    if new_tail.is_empty()
        || find_keyword_outside_quotes(new_tail, "CASCADE").is_some()
        || find_keyword_outside_quotes(new_tail, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameConstraint {
        table,
        old_name,
        new_name: normalize_identifier(new_tail)?,
        table_if_exists,
    })
}

/// Split off the leading identifier (a column or type name) from `input`, returning
/// `(name, rest)`. The name runs to the first ASCII whitespace.
fn split_leading_word(input: &str) -> Option<(&str, &str)> {
    let input = input.trim_start();
    if input.is_empty() {
        return None;
    }
    let end = input.find(char::is_whitespace).unwrap_or(input.len());
    Some((&input[..end], input[end..].trim_start()))
}

/// Split a column type token from `input`, returning `(type_token, tail)`. The token
/// is a name optionally followed by a balanced `(...)` typmod group, so a spaced
/// `NUMERIC(12, 2)` is kept intact (unlike a naive whitespace split).
fn split_column_type(input: &str) -> Option<(&str, &str)> {
    let (word, rest) = split_leading_word(input)?;
    // A `(` may begin the word's typmod immediately, or follow after whitespace
    // (`NUMERIC (12,2)`); accept both, balancing parens within the original input.
    let after_word_offset = word.as_ptr() as usize - input.as_ptr() as usize + word.len();
    let rest_trimmed = rest.trim_start();
    if rest_trimmed.starts_with('(') {
        let open = after_word_offset + (rest.len() - rest_trimmed.len());
        let close = find_matching_paren(input, open)?;
        return Some((input[..close + 1].trim(), input[close + 1..].trim_start()));
    }
    Some((word, rest))
}

/// Resolve a column's declared type token to a `(SqlType, domain)` pair. A token that
/// is not a built-in type is treated as a domain reference (defaulting to `Int4`,
/// matching the pre-existing behavior). `serial`/`serial4` is handled by the caller.
fn resolve_column_type(token: &str) -> Result<(SqlType, Option<String>), ParseError> {
    match parse_supported_sql_type_name(token) {
        Some(ty) => Ok((ty, None)),
        None => Ok((SqlType::Int4, Some(normalize_relation_identifier(token)?))),
    }
}

fn parse_column_def(input: &str) -> Result<ColumnDef, ParseError> {
    let (name, after_name) = split_leading_word(input).ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_identifier(name)?;
    let (raw_ty, tail) = split_column_type(after_name).ok_or(ParseError::InvalidRelationalSql)?;
    let (ty, domain) = resolve_column_type(raw_ty)?;
    let tail = tail.to_string();
    if domain.is_some() && !tail.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let default = if tail.is_empty() {
        None
    } else {
        let default_value = strip_keyword_prefix_case_insensitive(&tail, "DEFAULT")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim();
        if default_value.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        Some(parse_typed_column_default(default_value, ty, None)?)
    };
    Ok(ColumnDef {
        name,
        ty,
        domain,
        default,
    })
}

fn parse_column_default_expr(
    input: &str,
    implicit_serial_sequence: Option<String>,
) -> Result<ColumnDefault, ParseError> {
    let trimmed = input.trim();
    if let Some(sequence) = implicit_serial_sequence {
        if !trimmed.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(ColumnDefault::SequenceNextVal {
            sequence,
            create_if_missing: true,
        });
    }
    let function = trimmed
        .strip_prefix("pg_catalog.")
        .or_else(|| trimmed.strip_prefix("PG_CATALOG."))
        .unwrap_or(trimmed);
    if let Some(args) = function
        .strip_prefix("nextval")
        .or_else(|| function.strip_prefix("NEXTVAL"))
    {
        let args = args.trim_start();
        if !args.starts_with('(') {
            return Err(ParseError::InvalidRelationalSql);
        }
        let close = find_matching_paren(args, 0).ok_or(ParseError::InvalidRelationalSql)?;
        if !args[close + 1..].trim().is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let parts = split_csv(&args[1..close])?;
        let [target] = parts.as_slice() else {
            return Err(ParseError::InvalidRelationalSql);
        };
        return Ok(ColumnDefault::SequenceNextVal {
            sequence: parse_sequence_regclass_arg(target.trim())?,
            create_if_missing: false,
        });
    }
    Ok(ColumnDefault::Literal(parse_sql_value(trimmed)?))
}

fn parse_typed_column_default(
    input: &str,
    ty: SqlType,
    implicit_serial_sequence: Option<String>,
) -> Result<ColumnDefault, ParseError> {
    let default = parse_column_default_expr(input, implicit_serial_sequence)?;
    match default {
        // A literal default is coerced to the column's declared type, so e.g. `DEFAULT 0`
        // on a NUMERIC column is stored as a `Numeric` (not the inferred `Int4`), and
        // `DEFAULT TRUE` on a BOOL column is a `Bool`. We re-parse from the rendered
        // literal text rather than trusting the inferred variant.
        // A NULL default is the typeless SQL null — it stays NULL for a column of ANY type, so it
        // skips the type re-coercion (which would otherwise round-trip it through its rendered text
        // "NULL" and mis-parse it as e.g. the text value 'NULL').
        ColumnDefault::Literal(SqlValue::Null) => Ok(ColumnDefault::Literal(SqlValue::Null)),
        ColumnDefault::Literal(value) => {
            let rendered = render_default_literal_for_coercion(&value)?;
            let coerced = parse_typed_value_from_str(&rendered, ty)?;
            Ok(ColumnDefault::Literal(coerced))
        }
        // `nextval(...)` (serial) is integer-only, as before.
        ColumnDefault::SequenceNextVal { .. } if ty == SqlType::Int4 => Ok(default),
        ColumnDefault::SequenceNextVal { .. } => Err(ParseError::InvalidRelationalSql),
    }
}

/// Render an inferred default literal back to the textual form `parse_typed_value_from_str`
/// expects, so it can be re-parsed at the column's declared type.
fn render_default_literal_for_coercion(value: &SqlValue) -> Result<String, ParseError> {
    Ok(match value {
        SqlValue::Null => "NULL".to_string(),
        SqlValue::Int2(value) => value.to_string(),
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) => value.to_decimal_string(),
        SqlValue::Bool(value) => {
            if *value {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        SqlValue::Text(value) => value.clone(),
        SqlValue::Date(value) => crate::datetime::format_date(*value),
        SqlValue::Timestamp(value) => crate::datetime::format_timestamp(*value),
        SqlValue::Uuid(value) => crate::uuid::format_uuid(value),
        SqlValue::Parameter { .. } => {
            return Err(ParseError::InvalidParameterReference);
        }
    })
}

fn parse_drop_table_constraint(input: &str) -> Result<DropConstraint, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let table_if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF")
    {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let drop_pos =
        find_keyword_outside_quotes(rest, "DROP").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..drop_pos].trim())?;
    rest = rest[drop_pos + "DROP".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "CONSTRAINT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let constraint_if_exists = if let Some(after_if) =
        strip_keyword_prefix_case_insensitive(rest, "IF")
    {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let constraints = split_csv(rest)?;
    let [constraint] = constraints.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropConstraint {
        table,
        name: normalize_identifier(constraint.trim())?,
        table_if_exists,
        if_exists: constraint_if_exists,
    })
}

fn parse_alter_table_add_constraint(input: &str) -> Result<(String, String, &str), ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let add_pos =
        find_keyword_outside_quotes(rest, "ADD").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..add_pos].trim())?;
    let rest = rest[add_pos + "ADD".len()..].trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "CONSTRAINT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let primary_pos = find_keyword_outside_quotes(rest, "PRIMARY");
    let unique_pos = find_keyword_outside_quotes(rest, "UNIQUE");
    let check_pos = find_keyword_outside_quotes(rest, "CHECK");
    let foreign_pos = find_keyword_outside_quotes(rest, "FOREIGN");
    let constraint_pos = [primary_pos, unique_pos, check_pos, foreign_pos]
        .into_iter()
        .flatten()
        .min()
        .ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_identifier(rest[..constraint_pos].trim())?;
    Ok((table, name, rest[constraint_pos..].trim_start()))
}

fn parse_check_constraint_filter(rest: &str) -> Result<SelectFilter, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(rest, "CHECK")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if !rest.starts_with('(') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let close = find_matching_paren(rest, 0).ok_or(ParseError::InvalidRelationalSql)?;
    if close != rest.len() - 1 {
        return Err(ParseError::InvalidRelationalSql);
    }
    let filter = parse_select_filter(&rest[1..close])?;
    if matches!(filter.op, SelectFilterOp::LikePrefix) {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(filter)
}

/// Parse a constraint column LIST `(a, b, ...)` — the compound-key form of
/// [`parse_single_constraint_column`]. Returns the ordered, normalized key columns (>= 1). A
/// COMPOUND PRIMARY KEY / UNIQUE constraint is `PRIMARY KEY (a, b)`; a single-column one is the
/// `[a]` special case (so callers get a uniform `Vec`). Rejects an empty list / trailing tokens.
fn parse_constraint_columns(rest: &str) -> Result<Vec<String>, ParseError> {
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(rest, open).ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !rest[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let raw = split_csv(&rest[open + 1..close])?;
    if raw.is_empty() || raw.len() > 32 {
        return Err(ParseError::InvalidRelationalSql);
    }
    raw.iter()
        .map(|column| normalize_identifier(column.trim()))
        .collect()
}

fn parse_create_schema(input: &str) -> Result<CreateSchema, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SCHEMA"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_not_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_not = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "NOT")
            .ok_or(ParseError::InvalidRelationalSql)?;
        let after_exists = strip_keyword_prefix_case_insensitive(after_not.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "AUTHORIZATION").is_some()
        || find_keyword_outside_quotes(rest, "CREATE").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateSchema {
        name: normalize_identifier(rest)?,
        if_not_exists,
    })
}

fn parse_drop_schema(input: &str) -> Result<DropSchema, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SCHEMA"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let schemas = split_csv(rest)?;
    let [schema] = schemas.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropSchema {
        name: normalize_identifier(schema.trim())?,
        if_exists,
    })
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
    let mut primary_key = None;
    let mut unique_constraints = Vec::new();
    let mut check_constraints = Vec::new();
    for raw_column in split_csv(&rest[open + 1..close])? {
        let trimmed = raw_column.trim();
        if let Some(after_constraint) = strip_keyword_prefix_case_insensitive(trimmed, "CONSTRAINT")
        {
            let primary_pos = find_keyword_outside_quotes(after_constraint, "PRIMARY");
            let unique_pos = find_keyword_outside_quotes(after_constraint, "UNIQUE");
            let check_pos = find_keyword_outside_quotes(after_constraint, "CHECK");
            let constraint_pos = [primary_pos, unique_pos, check_pos]
                .into_iter()
                .flatten()
                .min()
                .ok_or(ParseError::InvalidRelationalSql)?;
            let name = normalize_identifier(after_constraint[..constraint_pos].trim())?;
            let rest = after_constraint[constraint_pos..].trim_start();
            if strip_keyword_prefix_case_insensitive(rest, "PRIMARY").is_some() {
                let rest = strip_keyword_prefix_case_insensitive(rest, "PRIMARY")
                    .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
                    .ok_or(ParseError::InvalidRelationalSql)?
                    .trim_start();
                let columns = parse_constraint_columns(rest)?;
                if primary_key.is_some() {
                    return Err(ParseError::InvalidRelationalSql);
                }
                primary_key = Some(PrimaryKey {
                    name: Some(name),
                    column: columns[0].clone(),
                    columns,
                });
            } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "UNIQUE") {
                let columns = parse_constraint_columns(rest.trim_start())?;
                unique_constraints.push(UniqueConstraint {
                    name: Some(name),
                    column: columns[0].clone(),
                    columns,
                });
            } else if strip_keyword_prefix_case_insensitive(rest, "CHECK").is_some() {
                check_constraints.push(CheckConstraint {
                    name: Some(name),
                    filter: parse_check_constraint_filter(rest)?,
                });
            } else {
                return Err(ParseError::InvalidRelationalSql);
            }
            continue;
        }
        if let Some(rest) = strip_keyword_prefix_case_insensitive(trimmed, "PRIMARY") {
            let rest = strip_keyword_prefix_case_insensitive(rest.trim_start(), "KEY")
                .ok_or(ParseError::InvalidRelationalSql)?
                .trim_start();
            let columns = parse_constraint_columns(rest)?;
            if primary_key.is_some() {
                return Err(ParseError::InvalidRelationalSql);
            }
            primary_key = Some(PrimaryKey {
                name: None,
                column: columns[0].clone(),
                columns,
            });
            continue;
        }
        if let Some(rest) = strip_keyword_prefix_case_insensitive(trimmed, "UNIQUE") {
            let columns = parse_constraint_columns(rest.trim_start())?;
            unique_constraints.push(UniqueConstraint {
                name: None,
                column: columns[0].clone(),
                columns,
            });
            continue;
        }
        if strip_keyword_prefix_case_insensitive(trimmed, "CHECK").is_some() {
            check_constraints.push(CheckConstraint {
                name: None,
                filter: parse_check_constraint_filter(trimmed)?,
            });
            continue;
        }
        let (name, after_name) =
            split_leading_word(raw_column).ok_or(ParseError::InvalidRelationalSql)?;
        let name = normalize_identifier(name)?;
        let (raw_ty, tail) =
            split_column_type(after_name).ok_or(ParseError::InvalidRelationalSql)?;
        let mut serial_sequence = None;
        let (ty, domain) =
            if raw_ty.eq_ignore_ascii_case("serial") || raw_ty.eq_ignore_ascii_case("serial4") {
                serial_sequence = Some(format!("{}_{}_seq", table, name));
                (SqlType::Int4, None)
            } else {
                resolve_column_type(raw_ty)?
            };
        let mut tail = tail.to_string();
        let mut column_primary_key = false;
        let mut column_unique = false;
        if let Some(primary_pos) = find_keyword_outside_quotes(&tail, "PRIMARY") {
            let after_primary =
                strip_keyword_prefix_case_insensitive(tail[primary_pos..].trim_start(), "PRIMARY")
                    .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
                    .ok_or(ParseError::InvalidRelationalSql)?;
            if !after_primary.trim().is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            tail = tail[..primary_pos].trim().to_string();
            column_primary_key = true;
        }
        if let Some(unique_pos) = find_keyword_outside_quotes(&tail, "UNIQUE") {
            let after_unique =
                strip_keyword_prefix_case_insensitive(tail[unique_pos..].trim_start(), "UNIQUE")
                    .ok_or(ParseError::InvalidRelationalSql)?;
            if !after_unique.trim().is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            tail = tail[..unique_pos].trim().to_string();
            column_unique = true;
        }
        let default = if tail.is_empty() && serial_sequence.is_none() {
            None
        } else {
            if domain.is_some() {
                return Err(ParseError::InvalidRelationalSql);
            }
            if serial_sequence.is_some() && !tail.is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            let default_value = if serial_sequence.is_some() {
                ""
            } else {
                strip_keyword_prefix_case_insensitive(&tail, "DEFAULT")
                    .ok_or(ParseError::InvalidRelationalSql)?
                    .trim()
            };
            if default_value.is_empty() && serial_sequence.is_none() {
                return Err(ParseError::InvalidRelationalSql);
            }
            Some(parse_typed_column_default(
                default_value,
                ty,
                serial_sequence,
            )?)
        };
        if column_primary_key {
            if primary_key.is_some() {
                return Err(ParseError::InvalidRelationalSql);
            }
            primary_key = Some(PrimaryKey {
                name: None,
                column: name.clone(),
                columns: vec![name.clone()],
            });
        }
        if column_unique {
            unique_constraints.push(UniqueConstraint {
                name: None,
                column: name.clone(),
                columns: vec![name.clone()],
            });
        }
        columns.push(ColumnDef {
            name,
            ty,
            domain,
            default,
        });
    }
    if columns.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if let Some(key) = &primary_key {
        if !columns.iter().any(|column| column.name == key.column) {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    for unique in &unique_constraints {
        if !columns.iter().any(|column| column.name == unique.column) {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    for check in &check_constraints {
        if !columns
            .iter()
            .any(|column| column.name == check.filter.column)
        {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    Ok(CreateTable {
        table,
        columns,
        primary_key,
        unique_constraints,
        check_constraints,
    })
}

fn parse_create_index(input: &str) -> Result<CreateIndex, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (unique, rest) =
        if let Some(after_unique) = strip_keyword_prefix_case_insensitive(rest, "UNIQUE") {
            (true, after_unique.trim_start())
        } else {
            (false, rest)
        };
    let rest = strip_keyword_prefix_case_insensitive(rest, "INDEX")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let on_pos = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_relation_identifier(rest[..on_pos].trim())?;
    let target = rest[on_pos + "ON".len()..].trim_start();
    let open = target.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(target, open).ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !target[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let table_target = target[..open].trim();
    let table_target =
        if let Some((table, method)) = split_optional_create_index_method(table_target) {
            if !method.eq_ignore_ascii_case("btree") {
                return Err(ParseError::InvalidRelationalSql);
            }
            table
        } else {
            table_target
        };
    let table = normalize_relation_identifier(table_target)?;
    // PRODUCT-002: preserve the ordered key list for compound secondary indexes. The engine catalog and
    // resident device-index layer already use this exact order for compound PRIMARY KEY / UNIQUE keys;
    // PRODUCT-002 applies the same ordered descriptor to non-unique BENCH indexes.
    let raw = split_csv(&target[open + 1..close])?;
    if raw.is_empty() || raw.len() > 32 {
        return Err(ParseError::InvalidRelationalSql);
    }
    let columns = raw
        .iter()
        .map(|column| normalize_identifier(column.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    let column = columns[0].clone();
    Ok(CreateIndex {
        name,
        table,
        column,
        columns,
        unique,
    })
}

fn parse_drop_index(input: &str) -> Result<DropIndex, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "INDEX"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "CONCURRENTLY").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty() || find_keyword_outside_quotes(rest, "CASCADE").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if find_keyword_outside_quotes(rest, "RESTRICT").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let indexes = split_csv(rest)?;
    if indexes.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropIndex {
        names: indexes
            .into_iter()
            .map(|index| normalize_relation_identifier(index.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_rename_index(input: &str) -> Result<RenameIndex, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "INDEX"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "ALL").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "CURRENT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_relation_identifier(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameIndex {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn parse_drop_table(input: &str) -> Result<DropTable, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty() || find_keyword_outside_quotes(rest, "CASCADE").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if find_keyword_outside_quotes(rest, "RESTRICT").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let tables = split_csv(rest)?;
    if tables.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropTable {
        names: tables
            .into_iter()
            .map(|table| normalize_relation_identifier(table.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_truncate_table(input: &str) -> Result<TruncateTable, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "TRUNCATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if let Some(after_table) = strip_keyword_prefix_case_insensitive(rest, "TABLE") {
        rest = after_table.trim_start();
    }
    if let Some(after_only) = strip_keyword_prefix_case_insensitive(rest, "ONLY") {
        rest = after_only.trim_start();
    }
    let mut restart_identity = false;
    if let Some(before_restart) = strip_keyword_suffix_case_insensitive(rest, "RESTART IDENTITY") {
        rest = before_restart.trim_end();
        restart_identity = true;
    }
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
        || find_keyword_outside_quotes(rest, "RESTART").is_some()
        || find_keyword_outside_quotes(rest, "CONTINUE").is_some()
        || find_keyword_outside_quotes(rest, "IDENTITY").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let tables = split_csv(rest)?;
    let [table] = tables.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(TruncateTable {
        name: normalize_relation_identifier(table.trim())?,
        restart_identity,
    })
}

fn split_optional_create_index_method(target: &str) -> Option<(&str, &str)> {
    let using_pos = find_keyword_outside_quotes(target, "USING")?;
    let table = target[..using_pos].trim();
    let method = target[using_pos + "USING".len()..].trim();
    (!table.is_empty() && !method.is_empty()).then_some((table, method))
}

fn parse_insert(input: &str) -> Result<Insert, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "INSERT")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "INTO"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let values_pos =
        find_keyword_outside_quotes(rest, "VALUES").ok_or(ParseError::InvalidRelationalSql)?;
    let target = rest[..values_pos].trim();
    let values_and_returning = rest[values_pos + "VALUES".len()..].trim_start();
    let (values, returning) = split_returning_clause(values_and_returning)?;
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
        returning,
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
    let (filter_input, returning) =
        split_returning_clause(rest[where_pos + "WHERE".len()..].trim())?;
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
        returning,
    })
}

fn parse_update(input: &str) -> Result<Update, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "UPDATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let set_pos =
        find_keyword_outside_quotes(rest, "SET").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..set_pos].trim())?;
    let after_set = rest[set_pos + "SET".len()..].trim_start();
    let where_pos =
        find_keyword_outside_quotes(after_set, "WHERE").ok_or(ParseError::InvalidRelationalSql)?;
    let assignment_input = after_set[..where_pos].trim();
    let (filter_input, returning) =
        split_returning_clause(after_set[where_pos + "WHERE".len()..].trim())?;
    if assignment_input.is_empty() || filter_input.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let assignments = split_csv(assignment_input)?
        .into_iter()
        .map(parse_update_assignment)
        .collect::<Result<Vec<_>, _>>()?;
    if assignments.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let filter_groups = parse_select_filter_groups(filter_input)?;
    let filters = filter_groups.first().cloned().unwrap_or_default();
    Ok(Update {
        table,
        assignments,
        filter: filters.first().cloned(),
        filters,
        filter_groups,
        returning,
    })
}

fn parse_update_assignment(input: &str) -> Result<UpdateAssignment, ParseError> {
    let (column, value) = input
        .split_once('=')
        .ok_or(ParseError::InvalidRelationalSql)?;
    let column = normalize_identifier(column.trim())?;
    let value = value.trim();
    if let Some(plus) = find_char_outside_quotes(value, '+') {
        let source_column = normalize_identifier(value[..plus].trim())?;
        if source_column != column || find_char_outside_quotes(&value[plus + 1..], '+').is_some() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let value = parse_sql_value(value[plus + 1..].trim())?;
        return Ok(UpdateAssignment {
            column,
            source_column: Some(source_column),
            value,
        });
    }
    let value = parse_sql_value(value)?;
    Ok(UpdateAssignment {
        column,
        source_column: None,
        value,
    })
}

fn split_returning_clause(input: &str) -> Result<(&str, Vec<String>), ParseError> {
    let Some(position) = find_keyword_outside_quotes(input, "RETURNING") else {
        return Ok((input.trim(), Vec::new()));
    };
    let body = input[..position].trim();
    let projection = input[position + "RETURNING".len()..].trim();
    if body.is_empty() || projection.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let returning = split_csv(projection)?
        .into_iter()
        .map(|column| normalize_identifier(column.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    if returning.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok((body, returning))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frozen_w1_returning_and_checked_update_shape() {
        let insert = parse_insert(
            "INSERT INTO ledger_entries (entry_id, tenant_id) VALUES (7::int8, 2::int8) \
             RETURNING entry_id",
        )
        .unwrap();
        assert_eq!(insert.returning, vec!["entry_id"]);

        let update = parse_update(
            "UPDATE accounts SET balance_cents = balance_cents + 9::int8, \
             version = version + 1::int8 WHERE tenant_id = 2::int8 AND account_id = 3::int8 \
             RETURNING balance_cents, version",
        )
        .unwrap();
        assert_eq!(update.returning, vec!["balance_cents", "version"]);
        assert_eq!(
            update.assignments[0].source_column.as_deref(),
            Some("balance_cents")
        );
        assert_eq!(update.assignments[0].value, SqlValue::Int8(9));

        let delete = parse_delete(
            "DELETE FROM pending_entries WHERE tenant_id = 2::int8 AND pending_id = 3::int8 \
             RETURNING pending_id",
        )
        .unwrap();
        assert_eq!(delete.returning, vec!["pending_id"]);
    }

    #[test]
    fn returning_and_update_expression_boundaries_fail_closed() {
        let update = parse_update(
            "UPDATE notes SET body = 'RETURNING + literal' WHERE id = 1 RETURNING body",
        )
        .unwrap();
        assert_eq!(update.assignments[0].source_column, None);
        assert_eq!(update.returning, vec!["body"]);

        assert!(matches!(
            parse_update("UPDATE t SET a = b + 1 WHERE id = 1 RETURNING a"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_update("UPDATE t SET a = a + 1 + 2 WHERE id = 1 RETURNING a"),
            Err(ParseError::InvalidRelationalSql)
        ));
        assert!(matches!(
            parse_delete("DELETE FROM t WHERE id = 1 RETURNING"),
            Err(ParseError::InvalidRelationalSql)
        ));
    }

    #[test]
    fn parses_bounded_typed_literal_projection_without_misclassifying_table_selects() {
        assert_eq!(
            parse_relational_command("SELECT 1 AS one", false)
                .unwrap()
                .unwrap(),
            Command::SelectLiteral(crate::SelectLiteral {
                column_name: "one".to_string(),
                ty: SqlType::Int4,
                value: SqlValue::Int4(1),
            })
        );
        assert_eq!(
            parse_relational_command("SELECT 'Ada'::text AS \"Display Name\"", false)
                .unwrap()
                .unwrap(),
            Command::SelectLiteral(crate::SelectLiteral {
                column_name: "Display Name".to_string(),
                ty: SqlType::Text,
                value: SqlValue::Text("Ada".to_string()),
            })
        );
        assert!(matches!(
            parse_relational_command("SELECT id FROM people", false)
                .unwrap()
                .unwrap(),
            Command::Select(_)
        ));

        assert_eq!(
            parse_relational_command(
                "SELECT pg_catalog.set_config('search_path', '', false)",
                true,
            )
            .unwrap()
            .unwrap(),
            Command::SelectLiteral(crate::SelectLiteral {
                column_name: "set_config".to_string(),
                ty: SqlType::Text,
                value: SqlValue::Text(String::new()),
            })
        );
        assert!(parse_relational_command(
            "SELECT pg_catalog.set_config('work_mem', '1GB', false)",
            true,
        )
        .unwrap()
        .is_err());
        assert_eq!(
            parse_relational_command("SELECT pg_catalog.pg_is_in_recovery()", true)
                .unwrap()
                .unwrap(),
            Command::SelectLiteral(crate::SelectLiteral {
                column_name: "pg_is_in_recovery".to_string(),
                ty: SqlType::Bool,
                value: SqlValue::Bool(false),
            })
        );
        assert_eq!(
            parse_relational_command("SELECT pg_catalog.current_schemas(false)", true)
                .unwrap()
                .unwrap(),
            Command::SelectLiteral(crate::SelectLiteral {
                column_name: "current_schemas".to_string(),
                ty: SqlType::Text,
                value: SqlValue::Text("{public}".to_string()),
            })
        );
    }
}
