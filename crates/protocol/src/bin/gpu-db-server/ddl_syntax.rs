// Legacy DDL syntax ownership. This is not a product execution path.

use super::{canonical_sql, strip_leading_sql_comments};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ParsedTruncateTable {
    pub(super) table: String,
    pub(super) restart_identity: bool,
}

pub(super) fn parse_truncate_table(statement: &str) -> Option<ParsedTruncateTable> {
    let statement = strip_leading_sql_comments(statement.trim())?;
    let canonical = canonical_sql(statement);
    let mut target = canonical.strip_prefix("truncate ")?.trim();
    target = target.strip_prefix("table ").unwrap_or(target).trim();
    target = target.strip_prefix("only ").unwrap_or(target).trim();
    let mut restart_identity = false;
    if let Some(before_restart) = target.strip_suffix(" restart identity") {
        target = before_restart.trim_end();
        restart_identity = true;
    }
    if target.contains(" cascade")
        || target.contains(" restrict")
        || target.contains(" restart ")
        || target.contains(" continue ")
        || target.contains(" identity")
    {
        return None;
    }
    let mut parts = target.split_whitespace();
    let table = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if !is_simple_copy_table_name(table) {
        return None;
    }
    if table.contains('.') && !table.starts_with("public.") {
        return None;
    }
    Some(ParsedTruncateTable {
        table: table.strip_prefix("public.").unwrap_or(table).to_string(),
        restart_identity,
    })
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct DropTable {
    pub(super) tables: Vec<String>,
    pub(super) if_exists: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct DropConstraint {
    pub(super) table: String,
    pub(super) constraint: String,
    pub(super) table_if_exists: bool,
    pub(super) if_exists: bool,
}

pub(super) fn parse_drop_table(statement: &str) -> Option<DropTable> {
    let statement = strip_leading_sql_comments(statement.trim())?;
    let canonical = canonical_sql(statement);
    let mut target = canonical.strip_prefix("drop table ")?.trim();
    let if_exists = if let Some(remaining) = target.strip_prefix("if exists ") {
        target = remaining.trim();
        true
    } else {
        false
    };
    if target
        .split(',')
        .any(|table| table.split_whitespace().count() != 1)
    {
        return None;
    }
    let tables = target
        .split(',')
        .map(str::trim)
        .map(|table| {
            if !is_simple_copy_table_name(table) {
                return None;
            }
            if table.contains('.') && !table.starts_with("public.") {
                return None;
            }
            Some(table.strip_prefix("public.").unwrap_or(table).to_string())
        })
        .collect::<Option<Vec<_>>>()?;
    if tables.is_empty() {
        return None;
    }
    Some(DropTable { tables, if_exists })
}

pub(super) fn parse_alter_table_drop_constraint(statement: &str) -> Option<DropConstraint> {
    let statement = strip_leading_sql_comments(statement.trim())?;
    let canonical = canonical_sql(statement);
    let mut target = canonical.strip_prefix("alter table ")?;
    let table_if_exists = if let Some(remaining) = target.strip_prefix("if exists ") {
        target = remaining.trim();
        true
    } else {
        false
    };
    target = target.strip_prefix("only ").unwrap_or(target).trim();
    let (table, rest) = target.split_once(" drop constraint ")?;
    if !is_simple_copy_table_name(table) {
        return None;
    }
    let mut rest = rest.trim();
    let if_exists = if let Some(remaining) = rest.strip_prefix("if exists ") {
        rest = remaining.trim();
        true
    } else {
        false
    };
    let mut parts = rest.split_whitespace();
    let constraint = parts.next()?;
    if parts.next().is_some() || !is_simple_copy_identifier(constraint) {
        return None;
    }
    Some(DropConstraint {
        table: table.strip_prefix("public.").unwrap_or(table).to_string(),
        constraint: constraint.to_string(),
        table_if_exists,
        if_exists,
    })
}

pub(super) fn unsupported_foreign_key_option_query(canonical: &str) -> bool {
    canonical.starts_with("alter table ")
        && canonical.contains(" foreign key ")
        && canonical.contains(" references ")
        && (canonical.contains(" on delete ")
            || canonical.contains(" on update ")
            || canonical.contains(" match ")
            || canonical.contains(" deferrable")
            || canonical.contains(" initially "))
}
