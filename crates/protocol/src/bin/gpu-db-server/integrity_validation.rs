// Legacy host-backed relational integrity validation. This is parity/bootstrap debt, not a product path.

use super::{select_filter_matches, CatalogIndex, ErrorField, Session, Table};
use std::collections::BTreeSet;

fn unique_index_violation_error(_index_name: &str) -> ErrorField {
    ErrorField {
        code: "23505",
        message: "duplicate key value violates unique index",
        position: None,
    }
}

pub(super) fn validate_unique_indexes(
    table: &Table,
    indexes: &[CatalogIndex],
) -> Result<(), ErrorField> {
    for index in indexes
        .iter()
        .filter(|index| index.table == table.name && index.unique)
    {
        let Some(column_idx) = table
            .columns
            .iter()
            .position(|column| column.def.name == index.column)
        else {
            continue;
        };
        let mut seen = BTreeSet::new();
        for row in &table.rows {
            if !seen.insert(row[column_idx].clone()) {
                return Err(unique_index_violation_error(&index.name));
            }
        }
    }
    Ok(())
}

fn check_constraint_violation_error(_table: &str, _constraint: &str) -> ErrorField {
    ErrorField {
        code: "23514",
        message: "new row violates check constraint",
        position: None,
    }
}

pub(super) fn validate_check_constraints(table: &Table) -> Result<(), ErrorField> {
    for constraint in &table.check_constraints {
        let Some(column_idx) = table
            .columns
            .iter()
            .position(|column| column.def.name == constraint.column)
        else {
            continue;
        };
        for row in &table.rows {
            if !select_filter_matches(&row[column_idx], constraint.op, &constraint.value) {
                return Err(check_constraint_violation_error(
                    &table.name,
                    &constraint.name,
                ));
            }
        }
    }
    Ok(())
}

fn foreign_key_violation_error() -> ErrorField {
    ErrorField {
        code: "23503",
        message: "insert or update violates foreign key constraint",
        position: None,
    }
}

pub(super) fn validate_foreign_keys(session: &Session) -> Result<(), ErrorField> {
    for table in session.tables.values() {
        for foreign_key in &table.foreign_keys {
            let Some(child_column_idx) = table
                .columns
                .iter()
                .position(|column| column.def.name == foreign_key.column)
            else {
                continue;
            };
            let Some(parent) = session.tables.get(&foreign_key.referenced_table) else {
                continue;
            };
            let Some(parent_column_idx) = parent
                .columns
                .iter()
                .position(|column| column.def.name == foreign_key.referenced_column)
            else {
                continue;
            };
            let parent_values = parent
                .rows
                .iter()
                .map(|row| row[parent_column_idx].clone())
                .collect::<BTreeSet<_>>();
            for row in &table.rows {
                if !parent_values.contains(&row[child_column_idx]) {
                    return Err(foreign_key_violation_error());
                }
            }
        }
    }
    Ok(())
}
