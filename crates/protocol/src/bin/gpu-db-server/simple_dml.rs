// Legacy simple-query DML ownership. This is not a product execution path.

use super::{
    evaluate_column_default, object_access_permission_error, row_matches_delete_filters,
    sql_value_matches_type, validate_check_constraints, validate_foreign_keys,
    validate_unique_indexes, write_command_complete, write_error, Command, ErrorField, ReadWrite,
    Session,
};
use gpu_db_protocol::TablePrivilege;
use std::collections::BTreeSet;
use std::io;

pub(super) fn execute_simple_dml(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::Insert(insert) => {
            let table_name = insert.table;
            let catalog_indexes = session.indexes.clone();
            let Some(table) = session.tables.get(&table_name).cloned() else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "relation does not exist",
                        position: None,
                    },
                );
            };
            if let Some(error) =
                object_access_permission_error(session, &table_name, TablePrivilege::Insert)
            {
                return write_error(stream, &error);
            }
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
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42703",
                                message: "column does not exist",
                                position: None,
                            },
                        );
                    };
                    indexes.push(idx);
                }
                indexes
            };
            let inserted_count = insert.rows.len();
            let mut new_rows = Vec::with_capacity(inserted_count);
            for row in insert.rows {
                if row.len() != indexes.len() {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42601",
                            message: "INSERT value count must match target columns",
                            position: None,
                        },
                    );
                }
                let mut projected = vec![None; table.columns.len()];
                for (source_idx, target_idx) in indexes.iter().copied().enumerate() {
                    if !sql_value_matches_type(&row[source_idx], table.columns[target_idx].def.ty) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42804",
                                message: "column type mismatch",
                                position: None,
                            },
                        );
                    }
                    projected[target_idx] = Some(row[source_idx].clone());
                }
                for (idx, value) in projected.iter_mut().enumerate() {
                    if value.is_none() {
                        if let Some(default) = table.columns[idx].def.default.clone() {
                            match evaluate_column_default(session, &default) {
                                Ok(default_value) => *value = Some(default_value),
                                Err(error) => return write_error(stream, &error),
                            }
                        }
                    }
                }
                if projected.iter().any(Option::is_none) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "0A000",
                            message: "INSERT must provide every column without a default",
                            position: None,
                        },
                    );
                }
                new_rows.push(projected.into_iter().map(Option::unwrap).collect());
            }
            let mut candidate_table = table.clone();
            candidate_table.rows.extend(new_rows.clone());
            if let Err(error) = validate_unique_indexes(&candidate_table, &catalog_indexes) {
                return write_error(stream, &error);
            }
            if let Err(error) = validate_check_constraints(&candidate_table) {
                return write_error(stream, &error);
            }
            let old_table = session
                .tables
                .insert(table_name.clone(), candidate_table)
                .expect("table existence checked");
            if let Err(error) = validate_foreign_keys(session) {
                session.tables.insert(table_name.clone(), old_table);
                return write_error(stream, &error);
            }
            session.mark_table_dirty(table_name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, &format!("INSERT 0 {inserted_count}"))
        }
        Command::Delete(delete) => {
            let table_name = delete.table.clone();
            let Some(table) = session.tables.get(&table_name).cloned() else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "relation does not exist",
                        position: None,
                    },
                );
            };
            if let Some(error) =
                object_access_permission_error(session, &table_name, TablePrivilege::Delete)
            {
                return write_error(stream, &error);
            }
            let mut delete_mask = Vec::with_capacity(table.rows.len());
            for row in &table.rows {
                match row_matches_delete_filters(&table, row, &delete) {
                    Ok(matches) => delete_mask.push(matches),
                    Err(error) => return write_error(stream, &error),
                }
            }
            let deleted_count = delete_mask.iter().filter(|matches| **matches).count();
            let mut delete_mask = delete_mask.into_iter();
            let mut candidate_table = table.clone();
            candidate_table.rows = table
                .rows
                .into_iter()
                .filter(|_| !delete_mask.next().unwrap_or(false))
                .collect();
            let old_table = session
                .tables
                .insert(table_name.clone(), candidate_table)
                .expect("table existence checked");
            if let Err(error) = validate_foreign_keys(session) {
                session.tables.insert(table_name.clone(), old_table);
                return write_error(stream, &error);
            }
            session.mark_table_dirty(table_name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, &format!("DELETE {deleted_count}"))
        }
        Command::Update(update) => {
            let table_name = update.table.clone();
            let catalog_indexes = session.indexes.clone();
            let Some(table) = session.tables.get(&table_name).cloned() else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "relation does not exist",
                        position: None,
                    },
                );
            };
            if let Some(error) =
                object_access_permission_error(session, &table_name, TablePrivilege::Update)
            {
                return write_error(stream, &error);
            }
            let mut seen = BTreeSet::new();
            let mut assignments = Vec::with_capacity(update.assignments.len());
            for assignment in &update.assignments {
                let Some(idx) = table
                    .columns
                    .iter()
                    .position(|column| column.def.name == assignment.column)
                else {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42703",
                            message: "column does not exist",
                            position: None,
                        },
                    );
                };
                if !seen.insert(idx) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42601",
                            message: "column assigned more than once",
                            position: None,
                        },
                    );
                }
                if !sql_value_matches_type(&assignment.value, table.columns[idx].def.ty) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42804",
                            message: "column type mismatch",
                            position: None,
                        },
                    );
                }
                assignments.push((idx, assignment.value.clone()));
            }
            let delete_shape = gpu_db_protocol::Delete {
                table: update.table.clone(),
                filter: update.filter.clone(),
                filters: update.filters.clone(),
                filter_groups: update.filter_groups.clone(),
            };
            let mut update_mask = Vec::with_capacity(table.rows.len());
            for row in &table.rows {
                match row_matches_delete_filters(&table, row, &delete_shape) {
                    Ok(matches) => update_mask.push(matches),
                    Err(error) => return write_error(stream, &error),
                }
            }
            let updated_count = update_mask.iter().filter(|matches| **matches).count();
            let mut candidate_rows = table.rows.clone();
            for (row, matches) in candidate_rows.iter_mut().zip(update_mask) {
                if matches {
                    for (idx, value) in &assignments {
                        row[*idx] = value.clone();
                    }
                }
            }
            let mut candidate_table = table.clone();
            candidate_table.rows = candidate_rows;
            if let Err(error) = validate_unique_indexes(&candidate_table, &catalog_indexes) {
                return write_error(stream, &error);
            }
            if let Err(error) = validate_check_constraints(&candidate_table) {
                return write_error(stream, &error);
            }
            let old_table = session
                .tables
                .insert(table_name.clone(), candidate_table)
                .expect("table existence checked");
            if let Err(error) = validate_foreign_keys(session) {
                session.tables.insert(table_name.clone(), old_table);
                return write_error(stream, &error);
            }
            session.mark_table_dirty(table_name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, &format!("UPDATE {updated_count}"))
        }
        _ => unreachable!("simple-query DML executor received an unrelated command"),
    }
}
