// Legacy extended-DML compatibility ownership. This is not a product execution path.

use super::{
    object_access_permission_error, row_matches_delete_filters, sql_value_matches_type, ErrorField,
    Session,
};
use gpu_db_protocol::TablePrivilege;
use std::collections::BTreeSet;

pub(super) fn execute_extended_insert(
    session: &mut Session,
    insert: gpu_db_protocol::Insert,
) -> Result<String, ErrorField> {
    if !insert.returning.is_empty() {
        return Err(ErrorField {
            code: "0A000",
            message: "DML RETURNING requires GPU engine execution",
            position: None,
        });
    }
    let table_name = insert.table;
    if !session.tables.contains_key(&table_name) {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    }
    if let Some(error) =
        object_access_permission_error(session, &table_name, TablePrivilege::Insert)
    {
        return Err(error);
    }
    let Some(table) = session.tables.get_mut(&table_name) else {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    let indexes = if insert.columns.is_empty() {
        (0..table.columns.len()).collect::<Vec<_>>()
    } else {
        let mut indexes = Vec::with_capacity(insert.columns.len());
        for column in &insert.columns {
            let Some(idx) = table
                .columns
                .iter()
                .position(|candidate| candidate.def.name == *column)
            else {
                return Err(ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                });
            };
            indexes.push(idx);
        }
        indexes
    };
    let inserted_count = insert.rows.len();
    for row in insert.rows {
        if row.len() != indexes.len() {
            return Err(ErrorField {
                code: "42601",
                message: "INSERT value count must match target columns",
                position: None,
            });
        }
        let mut projected = vec![None; table.columns.len()];
        for (source_idx, target_idx) in indexes.iter().copied().enumerate() {
            if !sql_value_matches_type(&row[source_idx], table.columns[target_idx].def.ty) {
                return Err(ErrorField {
                    code: "42804",
                    message: "column type mismatch",
                    position: None,
                });
            }
            projected[target_idx] = Some(row[source_idx].clone());
        }
        if projected.iter().any(Option::is_none) {
            return Err(ErrorField {
                code: "0A000",
                message: "INSERT must provide every column",
                position: None,
            });
        }
        table
            .rows
            .push(projected.into_iter().map(Option::unwrap).collect());
    }
    session.mark_table_dirty(table_name);
    session.persist_catalog_snapshot();
    Ok(format!("INSERT 0 {inserted_count}"))
}

pub(super) fn execute_extended_delete(
    session: &mut Session,
    delete: gpu_db_protocol::Delete,
) -> Result<String, ErrorField> {
    if !delete.returning.is_empty() {
        return Err(ErrorField {
            code: "0A000",
            message: "DML RETURNING requires GPU engine execution",
            position: None,
        });
    }
    let table_name = delete.table.clone();
    if !session.tables.contains_key(&table_name) {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    }
    if let Some(error) =
        object_access_permission_error(session, &table_name, TablePrivilege::Delete)
    {
        return Err(error);
    }
    let Some(table) = session.tables.get_mut(&table_name) else {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    let mut delete_mask = Vec::with_capacity(table.rows.len());
    for row in &table.rows {
        delete_mask.push(row_matches_delete_filters(table, row, &delete)?);
    }
    let deleted_count = delete_mask.iter().filter(|matches| **matches).count();
    let mut delete_mask = delete_mask.into_iter();
    let kept = std::mem::take(&mut table.rows)
        .into_iter()
        .filter(|_| !delete_mask.next().unwrap_or(false))
        .collect();
    table.rows = kept;
    session.mark_table_dirty(table_name);
    session.persist_catalog_snapshot();
    Ok(format!("DELETE {deleted_count}"))
}

pub(super) fn execute_extended_update(
    session: &mut Session,
    update: gpu_db_protocol::Update,
) -> Result<String, ErrorField> {
    if !update.returning.is_empty()
        || update
            .assignments
            .iter()
            .any(|assignment| assignment.source_column.is_some())
    {
        return Err(ErrorField {
            code: "0A000",
            message: "UPDATE expressions and DML RETURNING require GPU engine execution",
            position: None,
        });
    }
    let table_name = update.table.clone();
    if !session.tables.contains_key(&table_name) {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    }
    if let Some(error) =
        object_access_permission_error(session, &table_name, TablePrivilege::Update)
    {
        return Err(error);
    }
    let Some(table) = session.tables.get_mut(&table_name) else {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    let mut seen = BTreeSet::new();
    let mut assignments = Vec::with_capacity(update.assignments.len());
    for assignment in &update.assignments {
        let Some(idx) = table
            .columns
            .iter()
            .position(|column| column.def.name == assignment.column)
        else {
            return Err(ErrorField {
                code: "42703",
                message: "column does not exist",
                position: None,
            });
        };
        if !seen.insert(idx) {
            return Err(ErrorField {
                code: "42601",
                message: "column assigned more than once",
                position: None,
            });
        }
        if !sql_value_matches_type(&assignment.value, table.columns[idx].def.ty) {
            return Err(ErrorField {
                code: "42804",
                message: "column type mismatch",
                position: None,
            });
        }
        assignments.push((idx, assignment.value.clone()));
    }
    let delete_shape = gpu_db_protocol::Delete {
        table: update.table.clone(),
        filter: update.filter.clone(),
        filters: update.filters.clone(),
        filter_groups: update.filter_groups.clone(),
        returning: Vec::new(),
    };
    let mut update_mask = Vec::with_capacity(table.rows.len());
    for row in &table.rows {
        update_mask.push(row_matches_delete_filters(table, row, &delete_shape)?);
    }
    let updated_count = update_mask.iter().filter(|matches| **matches).count();
    for (row, matches) in table.rows.iter_mut().zip(update_mask) {
        if matches {
            for (idx, value) in &assignments {
                row[*idx] = value.clone();
            }
        }
    }
    session.mark_table_dirty(table_name);
    session.persist_catalog_snapshot();
    Ok(format!("UPDATE {updated_count}"))
}
