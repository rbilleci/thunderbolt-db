// Legacy information-schema column catalog ownership. This is not a product execution path.

use super::{
    column_type_display_name, column_type_oid, column_type_size, format_column_default_expr,
    int4_column, text_column, write_error, write_single_row, CatalogColumn, ErrorField, ReadWrite,
    Session, SqlType, Table,
};
use std::collections::BTreeSet;
use std::io;

pub(super) fn try_execute_information_schema_column_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if let Some(table) = information_schema_columns_query_table(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_column_rows(session, &table),
        ));
    }
    if canonical == information_schema_all_columns_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_all_column_rows(session),
        ));
    }
    if canonical == information_schema_column_discovery_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_all_column_rows(session),
        ));
    }
    if let Some(tables) = information_schema_columns_in_query_tables(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_column_rows_for_tables(session, &tables),
        ));
    }
    if let Some(table) = information_schema_column_details_query_table(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("column_name"),
                text_column("data_type"),
                text_column("is_nullable"),
                text_column("column_default"),
            ],
            &information_schema_column_detail_rows(session, &table),
        ));
    }
    if let Some(table) = information_schema_column_udt_query_table(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("column_name"),
                text_column("data_type"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_column_udt_rows(session, &table),
        ));
    }
    if canonical == information_schema_rich_columns_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_rich_column_rows(session),
        ));
    }
    if canonical == information_schema_extended_columns_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows(session),
        ));
    }
    if let Some(table) = information_schema_extended_columns_query_table(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_table(session, &table),
        ));
    }
    if let Some(table) = information_schema_extended_columns_catalog_query_table(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_table(session, &table),
        ));
    }
    if let Some(tables) = information_schema_extended_columns_in_query_tables(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_tables(session, &tables),
        ));
    }
    None
}

pub(super) fn try_execute_direct_attribute_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if let Some(table) = catalog_attribute_query_table(canonical) {
        let Some(rows) = catalog_attribute_rows(session, &table) else {
            return Some(write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            ));
        };
        return Some(write_single_row(
            stream,
            &[text_column("attname"), int4_column("atttypid")],
            &rows,
        ));
    }
    if let Some(table) = catalog_attribute_detail_query_table(canonical) {
        let Some(rows) = catalog_attribute_detail_rows(session, &table) else {
            return Some(write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            ));
        };
        return Some(write_single_row(
            stream,
            &[
                int4_column("attnum"),
                text_column("attname"),
                int4_column("atttypid"),
                int4_column("attlen"),
            ],
            &rows,
        ));
    }
    if let Some(table) = pg_catalog_class_attribute_type_query_table(canonical) {
        let Some(rows) = pg_catalog_class_attribute_type_rows(session, &table) else {
            return Some(write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            ));
        };
        return Some(write_single_row(
            stream,
            &[
                int4_column("attnum"),
                text_column("attname"),
                text_column("data_type"),
                text_column("attnotnull"),
            ],
            &rows,
        ));
    }
    None
}

fn information_schema_udt_metadata(column: &CatalogColumn) -> (String, String) {
    if let Some(domain) = column.def.domain.as_ref() {
        ("public".to_string(), domain.clone())
    } else {
        (
            "pg_catalog".to_string(),
            column.def.ty.catalog_name().to_string(),
        )
    }
}

fn information_schema_columns_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_rows(session: &Session, table: &str) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(column.def.name.clone()),
                Some(column.attnum.to_string()),
                Some(column_type_display_name(column)),
            ]
        })
        .collect()
}

fn information_schema_all_columns_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_column_discovery_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name, ordinal_position"
}

fn information_schema_all_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    Some(column_type_display_name(column)),
                ]
            })
        })
        .collect()
}

fn information_schema_columns_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name, ordinal_position";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    Some(column_type_display_name(column)),
                ]
            })
        })
        .collect()
}

fn information_schema_column_details_query_table(canonical: &str) -> Option<String> {
    let prefix = "select column_name, data_type, is_nullable, column_default from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_detail_rows(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                Some("YES".to_string()),
                column.def.default.as_ref().map(format_column_default_expr),
            ]
        })
        .collect()
}

fn information_schema_column_udt_query_table(canonical: &str) -> Option<String> {
    let prefix = "select column_name, data_type, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_udt_rows(session: &Session, table: &str) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            let (udt_schema, udt_name) = information_schema_udt_metadata(column);
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                Some(udt_schema),
                Some(udt_name),
            ]
        })
        .collect()
}

fn information_schema_rich_columns_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_rich_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                let (udt_schema, udt_name) = information_schema_udt_metadata(column);
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    column.def.default.as_ref().map(format_column_default_expr),
                    Some("YES".to_string()),
                    Some(column_type_display_name(column)),
                    Some(udt_schema),
                    Some(udt_name),
                ]
            })
        })
        .collect()
}

fn information_schema_extended_columns_query() -> &'static str {
    "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_extended_columns_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_extended_columns_catalog_query_table(canonical: &str) -> Option<String> {
    let suffix = "' order by ordinal_position";
    let current_database_prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = current_database() and table_schema = 'public' and table_name = '";
    if let Some(table) = canonical
        .strip_prefix(current_database_prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
    {
        return Some(table.to_string());
    }
    let literal_catalog_prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = 'postgres' and table_schema = 'public' and table_name = '";
    canonical
        .strip_prefix(literal_catalog_prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_extended_columns_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name, ordinal_position";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_extended_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(information_schema_extended_column_rows_for_catalog_table)
        .collect()
}

fn information_schema_extended_column_rows_for_table(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    information_schema_extended_column_rows_for_catalog_table(table).collect()
}

fn information_schema_extended_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(information_schema_extended_column_rows_for_catalog_table)
        .collect()
}

fn information_schema_extended_column_rows_for_catalog_table(
    table: &Table,
) -> impl Iterator<Item = Vec<Option<String>>> + '_ {
    table.columns.iter().map(|column| {
        let (numeric_precision, numeric_precision_radix, numeric_scale) =
            information_schema_numeric_metadata(column.def.ty);
        let (udt_schema, udt_name) = information_schema_udt_metadata(column);
        vec![
            Some("postgres".to_string()),
            Some("public".to_string()),
            Some(table.name.clone()),
            Some(column.def.name.clone()),
            Some(column.attnum.to_string()),
            column.def.default.as_ref().map(format_column_default_expr),
            Some("YES".to_string()),
            Some(column_type_display_name(column)),
            None,
            numeric_precision.map(|value| value.to_string()),
            numeric_precision_radix.map(|value| value.to_string()),
            numeric_scale.map(|value| value.to_string()),
            Some(udt_schema),
            Some(udt_name),
        ]
    })
}

fn information_schema_numeric_metadata(ty: SqlType) -> (Option<i32>, Option<i32>, Option<i32>) {
    match ty {
        SqlType::Int2 => (Some(16), Some(2), Some(0)),
        SqlType::Int4 => (Some(32), Some(2), Some(0)),
        SqlType::Int8 => (Some(64), Some(2), Some(0)),
        // For NUMERIC(p,s) PostgreSQL reports the declared precision/scale in radix 10.
        SqlType::Numeric { precision, scale } => {
            (Some(i32::from(precision)), Some(10), Some(i32::from(scale)))
        }
        SqlType::Bool => (None, None, None),
        SqlType::Text => (None, None, None),
        SqlType::Date => (None, None, None),
        SqlType::Timestamp => (None, None, None),
        SqlType::Uuid => (None, None, None),
    }
}

fn catalog_attribute_query_table(canonical: &str) -> Option<String> {
    let prefix = "select attname, atttypid from pg_catalog.pg_attribute where attrelid = '";
    let suffix = "'::regclass and attnum > 0 order by attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn catalog_attribute_detail_query_table(canonical: &str) -> Option<String> {
    let prefix =
        "select attnum, attname, atttypid, attlen from pg_catalog.pg_attribute where attrelid = '";
    let suffix = "'::regclass and attnum > 0 order by attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn catalog_attribute_rows(session: &Session, table: &str) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.def.name.clone()),
                    Some(column_type_oid(session, column).to_string()),
                ]
            })
            .collect(),
    )
}

fn catalog_attribute_detail_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.attnum.to_string()),
                    Some(column.def.name.clone()),
                    Some(column_type_oid(session, column).to_string()),
                    Some(column_type_size(column).to_string()),
                ]
            })
            .collect(),
    )
}

fn pg_catalog_class_attribute_type_query_table(canonical: &str) -> Option<String> {
    let prefix = "select a.attnum, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type, a.attnotnull from pg_catalog.pg_attribute a join pg_catalog.pg_class c on c.oid = a.attrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn pg_catalog_class_attribute_type_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.attnum.to_string()),
                    Some(column.def.name.clone()),
                    Some(column_type_display_name(column)),
                    Some("f".to_string()),
                ]
            })
            .collect(),
    )
}

#[cfg(test)]
pub(super) fn test_information_schema_columns_query_table(canonical: &str) -> Option<String> {
    information_schema_columns_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_information_schema_column_rows(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    information_schema_column_rows(session, table)
}

#[cfg(test)]
pub(super) fn test_information_schema_all_columns_query() -> &'static str {
    information_schema_all_columns_query()
}

#[cfg(test)]
pub(super) fn test_information_schema_column_discovery_query() -> &'static str {
    information_schema_column_discovery_query()
}

#[cfg(test)]
pub(super) fn test_information_schema_all_column_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    information_schema_all_column_rows(session)
}

#[cfg(test)]
pub(super) fn test_information_schema_columns_in_query_tables(
    canonical: &str,
) -> Option<Vec<String>> {
    information_schema_columns_in_query_tables(canonical)
}

#[cfg(test)]
pub(super) fn test_information_schema_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    information_schema_column_rows_for_tables(session, table_names)
}

#[cfg(test)]
pub(super) fn test_information_schema_column_details_query_table(
    canonical: &str,
) -> Option<String> {
    information_schema_column_details_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_information_schema_column_detail_rows(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    information_schema_column_detail_rows(session, table)
}

#[cfg(test)]
pub(super) fn test_information_schema_rich_columns_query() -> &'static str {
    information_schema_rich_columns_query()
}

#[cfg(test)]
pub(super) fn test_information_schema_rich_column_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    information_schema_rich_column_rows(session)
}

#[cfg(test)]
pub(super) fn test_information_schema_extended_columns_query() -> &'static str {
    information_schema_extended_columns_query()
}

#[cfg(test)]
pub(super) fn test_information_schema_extended_columns_query_table(
    canonical: &str,
) -> Option<String> {
    information_schema_extended_columns_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_information_schema_extended_columns_catalog_query_table(
    canonical: &str,
) -> Option<String> {
    information_schema_extended_columns_catalog_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_information_schema_extended_columns_in_query_tables(
    canonical: &str,
) -> Option<Vec<String>> {
    information_schema_extended_columns_in_query_tables(canonical)
}

#[cfg(test)]
pub(super) fn test_information_schema_extended_column_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    information_schema_extended_column_rows(session)
}

#[cfg(test)]
pub(super) fn test_information_schema_extended_column_rows_for_table(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    information_schema_extended_column_rows_for_table(session, table)
}

#[cfg(test)]
pub(super) fn test_information_schema_extended_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    information_schema_extended_column_rows_for_tables(session, table_names)
}

#[cfg(test)]
pub(super) fn test_information_schema_numeric_metadata(
    ty: SqlType,
) -> (Option<i32>, Option<i32>, Option<i32>) {
    information_schema_numeric_metadata(ty)
}

#[cfg(test)]
pub(super) fn test_catalog_attribute_query_table(canonical: &str) -> Option<String> {
    catalog_attribute_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_attribute_detail_query_table(canonical: &str) -> Option<String> {
    catalog_attribute_detail_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_attribute_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    catalog_attribute_rows(session, table)
}

#[cfg(test)]
pub(super) fn test_catalog_attribute_detail_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    catalog_attribute_detail_rows(session, table)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_class_attribute_type_query_table(canonical: &str) -> Option<String> {
    pg_catalog_class_attribute_type_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_class_attribute_type_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    pg_catalog_class_attribute_type_rows(session, table)
}
