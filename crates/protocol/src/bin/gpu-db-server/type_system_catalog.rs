// Legacy type-system catalog ownership. This is not a product execution path.

use super::{
    bool_column, catalog_empty_rows, int4_column, sql_type_display_name, text_column,
    write_single_row, Column, ReadWrite, Session, SqlType, PUBLIC_NAMESPACE_OID,
    SUPPORTED_SQL_TYPES,
};
use std::io;

pub(super) fn try_execute_type_system_catalog_query(
    stream: &mut dyn ReadWrite,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == psql_list_conversions_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Source"),
                text_column("Destination"),
                text_column("Default?"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical == psql_list_operators_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Left arg type"),
                text_column("Right arg type"),
                text_column("Result type"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical == psql_list_collations_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Provider"),
                text_column("Collate"),
                text_column("Ctype"),
                text_column("ICU Locale"),
                text_column("ICU Rules"),
                text_column("Deterministic?"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical == psql_list_casts_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Source type"),
                text_column("Target type"),
                text_column("Function"),
                text_column("Implicit?"),
            ],
            &catalog_empty_rows(),
        ));
    }
    None
}

pub(super) fn try_execute_type_system_pg_dump_catalog_query(
    stream: &mut dyn ReadWrite,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical
        == "select tableoid, oid, oprname, oprnamespace, oprowner, oprkind, oprleft, oprright, oprcode::oid as oprcode from pg_operator"
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("tableoid"),
                int4_column("oid"),
                text_column("oprname"),
                int4_column("oprnamespace"),
                int4_column("oprowner"),
                text_column("oprkind"),
                int4_column("oprleft"),
                int4_column("oprright"),
                int4_column("oprcode"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical
        == "select tableoid, oid, collname, collnamespace, collowner, collencoding from pg_collation"
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("tableoid"),
                int4_column("oid"),
                text_column("collname"),
                int4_column("collnamespace"),
                int4_column("collowner"),
                int4_column("collencoding"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical == "select tableoid, oid, conname, connamespace, conowner from pg_conversion" {
        return Some(write_single_row(
            stream,
            &[
                int4_column("tableoid"),
                int4_column("oid"),
                text_column("conname"),
                int4_column("connamespace"),
                int4_column("conowner"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical.starts_with("select tableoid, oid, castsource, casttarget")
        && canonical.contains("from pg_cast")
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("tableoid"),
                int4_column("oid"),
                int4_column("castsource"),
                int4_column("casttarget"),
                int4_column("castfunc"),
                text_column("castcontext"),
                text_column("castmethod"),
            ],
            &catalog_empty_rows(),
        ));
    }
    None
}

fn psql_list_conversions_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.conname as \"name\", pg_catalog.pg_encoding_to_char(c.conforencoding) as \"source\", pg_catalog.pg_encoding_to_char(c.contoencoding) as \"destination\", case when c.condefault then 'yes' else 'no' end as \"default?\" from pg_catalog.pg_conversion c join pg_catalog.pg_namespace n on n.oid = c.connamespace where true and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_conversion_is_visible(c.oid) order by 1, 2"
}

fn psql_list_operators_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", o.oprname as \"name\", case when o.oprkind='l' then null else pg_catalog.format_type(o.oprleft, null) end as \"left arg type\", case when o.oprkind='r' then null else pg_catalog.format_type(o.oprright, null) end as \"right arg type\", pg_catalog.format_type(o.oprresult, null) as \"result type\", coalesce(pg_catalog.obj_description(o.oid, 'pg_operator'), pg_catalog.obj_description(o.oprcode, 'pg_proc')) as \"description\" from pg_catalog.pg_operator o left join pg_catalog.pg_namespace n on n.oid = o.oprnamespace where n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_operator_is_visible(o.oid) order by 1, 2, 3, 4"
}

fn psql_list_collations_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.collname as \"name\", case c.collprovider when 'd' then 'default' when 'c' then 'libc' when 'i' then 'icu' end as \"provider\", c.collcollate as \"collate\", c.collctype as \"ctype\", c.colliculocale as \"icu locale\", c.collicurules as \"icu rules\", case when c.collisdeterministic then 'yes' else 'no' end as \"deterministic?\" from pg_catalog.pg_collation c, pg_catalog.pg_namespace n where n.oid = c.collnamespace and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and c.collencoding in (-1, pg_catalog.pg_char_to_encoding(pg_catalog.getdatabaseencoding())) and pg_catalog.pg_collation_is_visible(c.oid) order by 1, 2"
}

fn psql_list_casts_catalog_query() -> &'static str {
    "select pg_catalog.format_type(castsource, null) as \"source type\", pg_catalog.format_type(casttarget, null) as \"target type\", case when c.castmethod = 'b' then '(binary coercible)' when c.castmethod = 'i' then '(with inout)' else p.proname end as \"function\", case when c.castcontext = 'e' then 'no' when c.castcontext = 'a' then 'in assignment' else 'yes' end as \"implicit?\" from pg_catalog.pg_cast c left join pg_catalog.pg_proc p on c.castfunc = p.oid left join pg_catalog.pg_type ts on c.castsource = ts.oid left join pg_catalog.pg_namespace ns on ns.oid = ts.typnamespace left join pg_catalog.pg_type tt on c.casttarget = tt.oid left join pg_catalog.pg_namespace nt on nt.oid = tt.typnamespace where ( (true and pg_catalog.pg_type_is_visible(ts.oid) ) or (true and pg_catalog.pg_type_is_visible(tt.oid) ) ) order by 1, 2"
}

#[cfg(test)]
pub(super) fn test_psql_list_conversions_catalog_query() -> &'static str {
    psql_list_conversions_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_list_operators_catalog_query() -> &'static str {
    psql_list_operators_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_list_collations_catalog_query() -> &'static str {
    psql_list_collations_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_list_casts_catalog_query() -> &'static str {
    psql_list_casts_catalog_query()
}

pub(super) fn try_execute_builtin_type_catalog_query(
    stream: &mut dyn ReadWrite,
    canonical: &str,
) -> Option<io::Result<()>> {
    if let Some(type_name) = psql_describe_type_catalog_query_type(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_rows(&type_name),
        ));
    }
    if canonical == psql_describe_pg_catalog_types_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_rows_for_supported_types(),
        ));
    }
    if canonical == psql_describe_pg_catalog_types_verbose_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Internal name"),
                text_column("Size"),
                text_column("Elements"),
                text_column("Owner"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_verbose_rows_for_supported_types(),
        ));
    }
    None
}

pub(super) fn try_execute_type_pg_dump_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != pg_dump_type_metadata_query() {
        return None;
    }
    Some(write_single_row(
        stream,
        &pg_dump_type_metadata_columns(),
        &pg_dump_type_metadata_rows(session),
    ))
}

fn psql_describe_type_catalog_query_type(canonical: &str) -> Option<String> {
    let prefix = "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and (t.typname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default or pg_catalog.format_type(t.oid, null) operator(pg_catalog.~) '^(";
    let final_suffix = ")$' collate pg_catalog.default) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2";
    let rest = canonical.strip_prefix(prefix)?;
    let (type_name, rest) = rest.split_once(suffix)?;
    let display_name = rest.strip_suffix(final_suffix)?;
    let matched_type = sql_type_by_catalog_or_display_name(type_name)?;
    (sql_type_by_catalog_or_display_name(display_name) == Some(matched_type))
        .then(|| type_name.to_string())
}

fn psql_describe_pg_catalog_types_query() -> &'static str {
    "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
}

fn psql_describe_pg_catalog_types_verbose_query() -> &'static str {
    "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", t.typname as \"internal name\", case when t.typrelid != 0 then cast('tuple' as pg_catalog.text) when t.typlen < 0 then cast('var' as pg_catalog.text) else cast(t.typlen as pg_catalog.text) end as \"size\", pg_catalog.array_to_string( array( select e.enumlabel from pg_catalog.pg_enum e where e.enumtypid = t.oid order by e.enumsortorder ), e'\\n' ) as \"elements\", pg_catalog.pg_get_userbyid(t.typowner) as \"owner\", pg_catalog.array_to_string(t.typacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
}

fn sql_type_by_catalog_or_display_name(name: &str) -> Option<SqlType> {
    SUPPORTED_SQL_TYPES
        .into_iter()
        .find(|ty| ty.catalog_name() == name || sql_type_display_name(*ty) == name)
}

fn catalog_psql_describe_type_rows(type_name: &str) -> Vec<Vec<Option<String>>> {
    let Some(ty) = sql_type_by_catalog_or_display_name(type_name) else {
        return Vec::new();
    };
    vec![vec![
        Some("pg_catalog".to_string()),
        Some(sql_type_display_name(ty).to_string()),
        None,
    ]]
}

fn supported_sql_types_by_display_name() -> Vec<SqlType> {
    let mut types = SUPPORTED_SQL_TYPES.to_vec();
    types.sort_by_key(|ty| sql_type_display_name(*ty));
    types
}

fn catalog_psql_describe_type_rows_for_supported_types() -> Vec<Vec<Option<String>>> {
    supported_sql_types_by_display_name()
        .into_iter()
        .map(|ty| {
            vec![
                Some("pg_catalog".to_string()),
                Some(sql_type_display_name(ty).to_string()),
                None,
            ]
        })
        .collect()
}

fn catalog_psql_describe_type_verbose_rows_for_supported_types() -> Vec<Vec<Option<String>>> {
    supported_sql_types_by_display_name()
        .into_iter()
        .map(|ty| {
            vec![
                Some("pg_catalog".to_string()),
                Some(sql_type_display_name(ty).to_string()),
                Some(ty.catalog_name().to_string()),
                Some(sql_type_psql_size(ty).to_string()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ]
        })
        .collect()
}

fn sql_type_psql_size(ty: SqlType) -> &'static str {
    match ty.type_size() {
        -1 => "var",
        4 => "4",
        _ => "",
    }
}

fn pg_dump_type_metadata_query() -> &'static str {
    "select tableoid, oid, typname, typnamespace, typacl, acldefault('t', typowner) as acldefault, typowner, typelem, typrelid, case when typrelid = 0 then ' '::\"char\" else (select relkind from pg_class where oid = typrelid) end as typrelkind, typtype, typisdefined, typname[0] = '_' and typelem != 0 and (select typarray from pg_type te where oid = pg_type.typelem) = oid as isarray from pg_type"
}

fn pg_dump_type_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("typname"),
        int4_column("typnamespace"),
        text_column("typacl"),
        text_column("acldefault"),
        int4_column("typowner"),
        int4_column("typelem"),
        int4_column("typrelid"),
        text_column("typrelkind"),
        text_column("typtype"),
        bool_column("typisdefined"),
        bool_column("isarray"),
    ]
}

fn pg_dump_type_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = SUPPORTED_SQL_TYPES
        .into_iter()
        .map(|ty| {
            vec![
                Some("1247".to_string()),
                Some(ty.postgres_oid().to_string()),
                Some(ty.catalog_name().to_string()),
                Some("11".to_string()),
                None,
                None,
                Some("10".to_string()),
                Some("0".to_string()),
                Some("0".to_string()),
                Some(" ".to_string()),
                Some("b".to_string()),
                Some("t".to_string()),
                Some("f".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    let mut domains = session.domains.values().collect::<Vec<_>>();
    domains.sort_by_key(|domain| domain.oid);
    for domain in domains {
        rows.push(vec![
            Some("1247".to_string()),
            Some(domain.oid.to_string()),
            Some(domain.name.clone()),
            Some(PUBLIC_NAMESPACE_OID.to_string()),
            None,
            None,
            Some("10".to_string()),
            Some("0".to_string()),
            Some("0".to_string()),
            Some(" ".to_string()),
            Some("d".to_string()),
            Some("t".to_string()),
            Some("f".to_string()),
        ]);
    }
    rows
}

#[cfg(test)]
pub(super) fn test_psql_describe_type_catalog_query_type(canonical: &str) -> Option<String> {
    psql_describe_type_catalog_query_type(canonical)
}

#[cfg(test)]
pub(super) fn test_psql_describe_pg_catalog_types_query() -> &'static str {
    psql_describe_pg_catalog_types_query()
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_type_rows(type_name: &str) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_type_rows(type_name)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_type_rows_for_supported_types() -> Vec<Vec<Option<String>>>
{
    catalog_psql_describe_type_rows_for_supported_types()
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_type_verbose_rows_for_supported_types(
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_type_verbose_rows_for_supported_types()
}
