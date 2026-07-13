// Legacy table catalog ownership. This is not a product execution path.

use super::{
    acl_display, int4_column, psql_relname_pattern_matches, text_column, write_single_row,
    CatalogCommentTarget, ReadWrite, Session, SqlValue, Table,
};
use std::io;

pub(super) fn try_execute_table_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical
        == "select relname from pg_catalog.pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by relname"
    {
        return Some(write_single_row(
            stream,
            &[text_column("relname")],
            &catalog_table_name_rows(session),
        ));
    }
    if canonical
        == "select oid, relname from pg_catalog.pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by oid"
    {
        return Some(write_single_row(
            stream,
            &[int4_column("oid"), text_column("relname")],
            &catalog_table_oid_rows(session),
        ));
    }
    if canonical == psql_describe_tables_catalog_query()
        || canonical == psql_describe_all_schema_tables_catalog_query()
        || canonical == psql_describe_relations_catalog_query()
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_psql_describe_table_rows(session),
        ));
    }
    if let Some(filter) = psql_describe_tables_catalog_query_filter(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_psql_describe_table_rows_filtered(session, &filter),
        ));
    }
    if canonical == psql_describe_tables_verbose_catalog_query()
        || canonical == psql_describe_all_schema_tables_verbose_catalog_query()
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_psql_describe_table_verbose_rows(session),
        ));
    }
    if let Some(filter) = psql_describe_tables_verbose_catalog_query_filter(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_psql_describe_table_verbose_rows_filtered(session, &filter),
        ));
    }
    if let Some(filter) = psql_describe_table_privileges_catalog_query_filter(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Access privileges"),
                text_column("Column privileges"),
                text_column("Policies"),
            ],
            &catalog_psql_describe_table_privilege_rows_filtered(session, &filter),
        ));
    }
    None
}

fn catalog_table_name_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| vec![Some(table.name.clone())])
        .collect::<Vec<_>>()
}

fn catalog_table_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .tables
        .iter()
        .map(|(name, table)| (table.oid, name))
        .collect::<Vec<_>>();
    rows.sort_by_key(|(oid, _)| *oid);
    rows.into_iter()
        .map(|(oid, name)| vec![Some(oid.to_string()), Some(name.clone())])
        .collect()
}

fn psql_describe_tables_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_all_schema_tables_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') order by 1,2"
}

fn psql_describe_relations_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','v','m','s','f','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_tables_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_all_schema_tables_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') order by 1,2"
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PsqlDescribeTablesFilter {
    pub(super) namespace: String,
    pub(super) relname_pattern: Option<String>,
}

fn psql_describe_tables_catalog_query_filter(canonical: &str) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and ";
    let namespace_prefix = "n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1,2";
    let rest = canonical.strip_prefix(prefix)?;
    if let Some(namespace) = rest
        .strip_prefix(namespace_prefix)
        .and_then(|rest| rest.strip_suffix(namespace_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: namespace.to_string(),
            relname_pattern: None,
        });
    }

    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }
    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_tables_verbose_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and ";
    let namespace_prefix = "n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1,2";
    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let rest = canonical.strip_prefix(prefix)?;

    if let Some(namespace) = rest
        .strip_prefix(namespace_prefix)
        .and_then(|rest| rest.strip_suffix(namespace_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: namespace.to_string(),
            relname_pattern: None,
        });
    }

    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }

    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_table_privileges_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 's' then 'sequence' when 'f' then 'foreign table' when 'p' then 'partitioned table' end as \"type\", pg_catalog.array_to_string(c.relacl, e'\\n') as \"access privileges\", pg_catalog.array_to_string(array( select attname || e':\\n ' || pg_catalog.array_to_string(attacl, e'\\n ') from pg_catalog.pg_attribute a where attrelid = c.oid and not attisdropped and attacl is not null ), e'\\n') as \"column privileges\", pg_catalog.array_to_string(array( select polname || case when not polpermissive then e' (restrictive)' else '' end || case when polcmd != '*' then e' (' || polcmd::pg_catalog.text || e'):' else e':' end || case when polqual is not null then e'\\n (u): ' || pg_catalog.pg_get_expr(polqual, polrelid) else e'' end || case when polwithcheck is not null then e'\\n (c): ' || pg_catalog.pg_get_expr(polwithcheck, polrelid) else e'' end || case when polroles <> '{0}' then e'\\n to: ' || pg_catalog.array_to_string( array( select rolname from pg_catalog.pg_roles where oid = any (polroles) order by 1 ), e', ') else e'' end from pg_catalog.pg_policy pol where polrelid = c.oid), e'\\n') as \"policies\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('r','v','m','s','f','p') and ";
    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1, 2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1, 2";
    let rest = canonical.strip_prefix(prefix)?;

    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }

    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn catalog_psql_describe_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_rows_filtered(
        session,
        &PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        },
    )
}

fn catalog_psql_describe_table_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_verbose_rows_filtered(
        session,
        &PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        },
    )
}

fn catalog_psql_describe_table_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .filter(|table| {
            filter
                .relname_pattern
                .as_deref()
                .is_none_or(|pattern| psql_relname_pattern_matches(pattern, &table.name))
        })
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                Some("postgres".to_string()),
            ]
        })
        .collect()
}

fn catalog_psql_describe_table_verbose_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .filter(|table| {
            filter
                .relname_pattern
                .as_deref()
                .is_none_or(|pattern| psql_relname_pattern_matches(pattern, &table.name))
        })
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                Some("heap".to_string()),
                Some(psql_pretty_table_size(table)),
                session
                    .comments
                    .get(&CatalogCommentTarget::Table {
                        table: table.name.clone(),
                    })
                    .cloned(),
            ]
        })
        .collect()
}

fn psql_pretty_table_size(table: &Table) -> String {
    psql_size_pretty(compat_table_heap_size_bytes(table))
}

fn compat_table_heap_size_bytes(table: &Table) -> u64 {
    table
        .rows
        .iter()
        .map(|row| 24 + row.iter().map(compat_sql_value_size_bytes).sum::<u64>())
        .sum()
}

fn compat_sql_value_size_bytes(value: &SqlValue) -> u64 {
    match value {
        // NULL is sent as a `-1` field length (no bytes); 0 here keeps the size estimate honest.
        SqlValue::Null => 0,
        SqlValue::Int2(_) => 2,
        SqlValue::Int4(_) => 4,
        SqlValue::Text(value) => value.len() as u64,
        SqlValue::Int8(_) => 8,
        // Numeric is sent as text on this endpoint; report its rendered text length.
        SqlValue::Numeric(value) => value.to_decimal_string().len() as u64,
        SqlValue::Bool(_) => 1,
        // Date is sent as its ISO text (YYYY-MM-DD) on this endpoint.
        SqlValue::Date(value) => gpu_db_protocol::datetime::format_date(*value).len() as u64,
        SqlValue::Timestamp(value) => {
            gpu_db_protocol::datetime::format_timestamp(*value).len() as u64
        }
        SqlValue::Uuid(_) => 36, // canonical hyphenated text length
    }
}

fn psql_size_pretty(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes < 10 * KB {
        format!("{bytes} bytes")
    } else if bytes < 10 * MB {
        format!("{} kB", bytes / KB)
    } else if bytes < 10 * GB {
        format!("{} MB", bytes / MB)
    } else {
        format!("{} GB", bytes / GB)
    }
}

fn catalog_psql_describe_table_privilege_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut rows = Vec::new();
    for (name, kind) in session
        .tables
        .keys()
        .map(|name| (name.as_str(), "table"))
        .chain(session.views.keys().map(|name| (name.as_str(), "view")))
        .chain(
            session
                .materialized_views
                .keys()
                .map(|name| (name.as_str(), "materialized view")),
        )
        .chain(
            session
                .sequences
                .keys()
                .map(|name| (name.as_str(), "sequence")),
        )
    {
        if filter
            .relname_pattern
            .as_deref()
            .is_some_and(|pattern| !psql_relname_pattern_matches(pattern, name))
        {
            continue;
        }
        rows.push(vec![
            Some("public".to_string()),
            Some(name.to_string()),
            Some(kind.to_string()),
            relation_acl_display(session, name),
            None,
            None,
        ]);
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn relation_acl_display(session: &Session, relation: &str) -> Option<String> {
    let acl = session.table_acls.get(relation)?;
    acl_display(acl)
}

#[cfg(test)]
pub(super) fn test_catalog_table_name_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_table_name_rows(session)
}

#[cfg(test)]
pub(super) fn test_catalog_table_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_table_oid_rows(session)
}

#[cfg(test)]
pub(super) fn test_psql_describe_tables_verbose_catalog_query() -> &'static str {
    psql_describe_tables_verbose_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_all_schema_tables_catalog_query() -> &'static str {
    psql_describe_all_schema_tables_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_all_schema_tables_verbose_catalog_query() -> &'static str {
    psql_describe_all_schema_tables_verbose_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_relations_catalog_query() -> &'static str {
    psql_describe_relations_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_tables_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    psql_describe_tables_catalog_query_filter(canonical)
}

#[cfg(test)]
pub(super) fn test_psql_describe_tables_verbose_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    psql_describe_tables_verbose_catalog_query_filter(canonical)
}

#[cfg(test)]
pub(super) fn test_psql_describe_table_privileges_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    psql_describe_table_privileges_catalog_query_filter(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_rows(session)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_table_verbose_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_verbose_rows(session)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_table_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_rows_filtered(session, filter)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_table_verbose_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_verbose_rows_filtered(session, filter)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_table_privilege_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_privilege_rows_filtered(session, filter)
}

#[cfg(test)]
pub(super) fn test_relation_acl_display(session: &Session, relation: &str) -> Option<String> {
    relation_acl_display(session, relation)
}
