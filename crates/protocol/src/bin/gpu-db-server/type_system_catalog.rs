// Legacy empty type-system catalog ownership. This is not a product execution path.

use super::{catalog_empty_rows, int4_column, text_column, write_single_row, ReadWrite};
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
