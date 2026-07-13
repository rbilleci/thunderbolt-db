// Legacy constraint/default catalog ownership. This is not a product execution path.

use super::{
    catalog_constraint_contype, catalog_constraint_entries, catalog_constraint_type,
    format_column_default_expr, int4_column, text_column, write_single_row, ReadWrite, Session,
};
use std::io;

pub(super) fn try_execute_information_schema_constraint_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == information_schema_table_constraints_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("constraint_name"),
                text_column("constraint_type"),
            ],
            &information_schema_table_constraint_rows(session),
        ));
    }
    if canonical == information_schema_key_column_usage_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                text_column("constraint_name"),
                int4_column("ordinal_position"),
            ],
            &information_schema_key_column_usage_rows(session),
        ));
    }
    None
}

pub(super) fn try_execute_pg_constraint_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == pg_catalog_constraints_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("conname"),
                text_column("contype"),
            ],
            &pg_catalog_constraint_rows(session),
        ));
    }
    if canonical == pg_catalog_attrdefs_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("default_expr"),
            ],
            &pg_catalog_attrdef_rows(session),
        ));
    }
    None
}

fn information_schema_table_constraints_query() -> &'static str {
    "select table_schema, table_name, constraint_name, constraint_type from information_schema.table_constraints where table_schema = 'public' order by table_name, constraint_name"
}

fn information_schema_table_constraint_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = catalog_constraint_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("public".to_string()),
                Some(entry.index.table.clone()),
                Some(entry.index.name.clone()),
                Some(catalog_constraint_type(&entry.index).to_string()),
            ]
        })
        .collect::<Vec<_>>();
    for table in session.tables.values() {
        for constraint in &table.check_constraints {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("CHECK".to_string()),
            ]);
        }
        for constraint in &table.foreign_keys {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("FOREIGN KEY".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn information_schema_key_column_usage_query() -> &'static str {
    "select table_schema, table_name, column_name, constraint_name, ordinal_position from information_schema.key_column_usage where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_key_column_usage_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = catalog_constraint_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("public".to_string()),
                Some(entry.index.table.clone()),
                Some(entry.index.column.clone()),
                Some(entry.index.name.clone()),
                Some("1".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    for table in session.tables.values() {
        for constraint in &table.foreign_keys {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.column.clone()),
                Some(constraint.name.clone()),
                Some("1".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[4].cmp(&right[4])));
    rows
}

fn pg_catalog_constraints_query() -> &'static str {
    "select n.nspname, c.relname, con.conname, con.contype from pg_catalog.pg_constraint con join pg_catalog.pg_class c on c.oid = con.conrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' order by c.relname, con.conname"
}

fn pg_catalog_constraint_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = catalog_constraint_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("public".to_string()),
                Some(entry.index.table.clone()),
                Some(entry.index.name.clone()),
                Some(catalog_constraint_contype(&entry.index).to_string()),
            ]
        })
        .collect::<Vec<_>>();
    for table in session.tables.values() {
        for constraint in &table.check_constraints {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("c".to_string()),
            ]);
        }
        for constraint in &table.foreign_keys {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("f".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn pg_catalog_attrdefs_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, pg_catalog.pg_get_expr(d.adbin, d.adrelid) as default_expr from pg_catalog.pg_attrdef d join pg_catalog.pg_class c on c.oid = d.adrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace join pg_catalog.pg_attribute a on a.attrelid = d.adrelid and a.attnum = d.adnum where n.nspname = 'public' order by c.relname, a.attnum"
}

fn pg_catalog_attrdef_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().filter_map(|column| {
                column.def.default.as_ref().map(|default| {
                    vec![
                        Some("public".to_string()),
                        Some(table.name.clone()),
                        Some(column.def.name.clone()),
                        Some(format_column_default_expr(default)),
                    ]
                })
            })
        })
        .collect()
}

#[cfg(test)]
pub(super) fn test_information_schema_table_constraints_query() -> &'static str {
    information_schema_table_constraints_query()
}

#[cfg(test)]
pub(super) fn test_information_schema_table_constraint_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    information_schema_table_constraint_rows(session)
}

#[cfg(test)]
pub(super) fn test_information_schema_key_column_usage_query() -> &'static str {
    information_schema_key_column_usage_query()
}

#[cfg(test)]
pub(super) fn test_information_schema_key_column_usage_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    information_schema_key_column_usage_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_constraints_query() -> &'static str {
    pg_catalog_constraints_query()
}

#[cfg(test)]
pub(super) fn test_pg_catalog_constraint_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_catalog_constraint_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_attrdefs_query() -> &'static str {
    pg_catalog_attrdefs_query()
}

#[cfg(test)]
pub(super) fn test_pg_catalog_attrdef_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_catalog_attrdef_rows(session)
}
