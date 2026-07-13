// Legacy psql relation-introspection ownership. This is not a product execution path.

use super::{
    bool_column, catalog_constraint_contype, catalog_constraint_definition,
    catalog_index_definition, catalog_index_entries, column_type_display_name,
    foreign_key_definition, format_column_default_expr, format_sql_value, int4_column,
    psql_relname_pattern_matches, sql_type_storage_code, text_column, write_single_row,
    CatalogCheckConstraint, CatalogCommentTarget, ReadWrite, SelectFilterOp, Session, SqlValue,
};
use std::io;

pub(super) fn try_execute_relation_introspection_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if catalog_describe_relation_lookup_query_all_schemas(canonical)
        || catalog_describe_relation_lookup_query_public_namespace(canonical)
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
            ],
            &catalog_describe_relation_lookup_rows_for_public_namespace(session),
        ));
    }
    if let Some(table) = catalog_describe_relation_lookup_query_table(canonical) {
        return Some(write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
            ],
            &catalog_describe_relation_lookup_rows(session, &table),
        ));
    }
    if let Some(oid) = catalog_describe_relation_flags_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                int4_column("relchecks"),
                text_column("relkind"),
                text_column("relhasindex"),
                text_column("relhasrules"),
                text_column("relhastriggers"),
                text_column("relrowsecurity"),
                text_column("relforcerowsecurity"),
                text_column("relhasoids"),
                text_column("relispartition"),
                text_column("?column?"),
                int4_column("reltablespace"),
                text_column("case"),
                text_column("relpersistence"),
                text_column("relreplident"),
                text_column("amname"),
            ],
            &catalog_describe_relation_flags_rows(session, oid),
        ));
    }
    if let Some(oid) = catalog_describe_verbose_attribute_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("attname"),
                text_column("format_type"),
                text_column("pg_get_expr"),
                text_column("attnotnull"),
                text_column("attcollation"),
                text_column("attidentity"),
                text_column("attgenerated"),
                text_column("attstorage"),
                text_column("attcompression"),
                int4_column("attstattarget"),
                text_column("col_description"),
            ],
            &catalog_describe_verbose_attribute_rows(session, oid),
        ));
    }
    if let Some(oid) = catalog_describe_attribute_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("attname"),
                text_column("format_type"),
                text_column("pg_get_expr"),
                text_column("attnotnull"),
                text_column("attcollation"),
                text_column("attidentity"),
                text_column("attgenerated"),
            ],
            &catalog_describe_attribute_rows(session, oid),
        ));
    }
    if let Some(oid) = catalog_describe_index_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("relname"),
                text_column("indisprimary"),
                text_column("indisunique"),
                text_column("indisclustered"),
                text_column("indisvalid"),
                text_column("pg_get_indexdef"),
                text_column("pg_get_constraintdef"),
                text_column("contype"),
                text_column("condeferrable"),
                text_column("condeferred"),
                text_column("indisreplident"),
                int4_column("reltablespace"),
            ],
            &catalog_describe_index_rows(session, oid),
        ));
    }
    if let Some(oid) = catalog_describe_check_constraints_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[text_column("conname"), text_column("pg_get_constraintdef")],
            &catalog_describe_check_constraint_rows(session, oid),
        ));
    }
    if let Some(oid) = catalog_describe_foreign_keys_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                bool_column("sametable"),
                text_column("conname"),
                text_column("condef"),
                text_column("ontable"),
            ],
            &catalog_describe_foreign_key_rows(session, oid),
        ));
    }
    if let Some(oid) = catalog_describe_referenced_by_foreign_keys_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("conname"),
                text_column("ontable"),
                text_column("condef"),
            ],
            &catalog_describe_referenced_by_foreign_key_rows(session, oid),
        ));
    }
    if let Some(oid) = catalog_describe_trigger_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("tgname"),
                text_column("pg_get_triggerdef"),
                text_column("tgenabled"),
                text_column("tgisinternal"),
                text_column("parent"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        ));
    }
    if let Some(oid) = catalog_describe_policy_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("polname"),
                text_column("polpermissive"),
                text_column("array_to_string"),
                text_column("pg_get_expr"),
                text_column("pg_get_expr"),
                text_column("cmd"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        ));
    }
    if let Some(oid) = catalog_describe_statistic_ext_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("stxrelid"),
                text_column("nsp"),
                text_column("stxname"),
                text_column("columns"),
                text_column("ndist_enabled"),
                text_column("deps_enabled"),
                text_column("mcv_enabled"),
                int4_column("stxstattarget"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        ));
    }
    if let Some(oid) = catalog_describe_inherits_parent_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[text_column("oid")],
            &catalog_empty_rows_for_relation_oid(oid),
        ));
    }
    if let Some(oid) = catalog_describe_inherits_child_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("oid"),
                text_column("relkind"),
                text_column("inhdetachpending"),
                text_column("pg_get_expr"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        ));
    }
    None
}

fn catalog_describe_relation_lookup_query_table(canonical: &str) -> Option<String> {
    let prefix = "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(";
    let visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 2, 3";
    if let Some(table) = canonical
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(visible_suffix))
    {
        return Some(table.to_string());
    }

    let namespace_middle =
        ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 2, 3";
    let (table, namespace) = canonical
        .strip_prefix(prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(namespace_middle)?;
    (namespace == "public").then(|| table.to_string())
}

fn catalog_describe_relation_lookup_query_public_namespace(canonical: &str) -> bool {
    canonical
        == "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
}

fn catalog_describe_relation_lookup_query_all_schemas(canonical: &str) -> bool {
    canonical
        == "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace order by 2, 3"
}

fn catalog_describe_relation_lookup_rows(
    session: &Session,
    relname_pattern: &str,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables
        .into_iter()
        .filter(|table| psql_relname_pattern_matches(relname_pattern, &table.name))
    {
        rows.push(vec![
            Some(table.oid.to_string()),
            Some("public".to_string()),
            Some(table.name.clone()),
        ]);
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences
        .into_iter()
        .filter(|sequence| psql_relname_pattern_matches(relname_pattern, &sequence.name))
    {
        rows.push(vec![
            Some(sequence.oid.to_string()),
            Some("public".to_string()),
            Some(sequence.name.clone()),
        ]);
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views
        .into_iter()
        .filter(|view| psql_relname_pattern_matches(relname_pattern, &view.name))
    {
        rows.push(vec![
            Some(view.oid.to_string()),
            Some("public".to_string()),
            Some(view.name.clone()),
        ]);
    }
    rows
}

fn catalog_describe_relation_lookup_rows_for_public_namespace(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables {
        rows.push(vec![
            Some(table.oid.to_string()),
            Some("public".to_string()),
            Some(table.name.clone()),
        ]);
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences {
        rows.push(vec![
            Some(sequence.oid.to_string()),
            Some("public".to_string()),
            Some(sequence.name.clone()),
        ]);
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views {
        rows.push(vec![
            Some(view.oid.to_string()),
            Some("public".to_string()),
            Some(view.name.clone()),
        ]);
    }
    rows
}

fn catalog_describe_relation_flags_query_oid(canonical: &str) -> Option<u32> {
    let plain_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, '', c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let verbose_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', '), c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let verbose_wrapped_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', ') , c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let suffix = "'";
    canonical
        .strip_prefix(plain_prefix)
        .or_else(|| canonical.strip_prefix(verbose_prefix))
        .or_else(|| canonical.strip_prefix(verbose_wrapped_prefix))?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_relation_flags_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    if session
        .sequences
        .values()
        .any(|sequence| sequence.oid == oid)
    {
        return vec![vec![
            Some("0".to_string()),
            Some("s".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some(String::new()),
            Some("0".to_string()),
            Some(String::new()),
            Some("p".to_string()),
            Some("d".to_string()),
            None,
        ]];
    }
    if session
        .materialized_views
        .values()
        .any(|view| view.oid == oid)
    {
        return vec![vec![
            Some("0".to_string()),
            Some("m".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some(String::new()),
            Some("0".to_string()),
            Some(String::new()),
            Some("p".to_string()),
            Some("d".to_string()),
            Some("heap".to_string()),
        ]];
    }
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    let relhasindex = session
        .indexes
        .iter()
        .any(|index| index.table == table.name);
    let relhastriggers = !table.foreign_keys.is_empty()
        || session.tables.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        });

    vec![vec![
        Some(table.check_constraints.len().to_string()),
        Some("r".to_string()),
        Some(if relhasindex { "t" } else { "f" }.to_string()),
        Some("f".to_string()),
        Some(if relhastriggers { "t" } else { "f" }.to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some(String::new()),
        Some("0".to_string()),
        Some(String::new()),
        Some("p".to_string()),
        Some("d".to_string()),
        Some("heap".to_string()),
    ]]
}

fn catalog_describe_attribute_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated from pg_catalog.pg_attribute a where a.attrelid = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_index_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c2.relname, i.indisprimary, i.indisunique, i.indisclustered, i.indisvalid, pg_catalog.pg_get_indexdef(i.indexrelid, 0, true), pg_catalog.pg_get_constraintdef(con.oid, true), contype, condeferrable, condeferred, i.indisreplident, c2.reltablespace from pg_catalog.pg_class c, pg_catalog.pg_class c2, pg_catalog.pg_index i left join pg_catalog.pg_constraint con on (conrelid = i.indrelid and conindid = i.indexrelid and contype in ('p','u','x')) where c.oid = '";
    let suffix = "' and c.oid = i.indrelid and i.indexrelid = c2.oid order by i.indisprimary desc, c2.relname";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_check_constraints_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select r.conname, pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where r.conrelid = '";
    let suffix = "' and r.contype = 'c' order by 1";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_foreign_keys_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_constraint r")
        || !canonical.contains("r.contype = 'f'")
        || !canonical.contains("conparentid = 0")
    {
        return None;
    }
    let marker = "r.conrelid = '";
    let (_, rest) = canonical.split_once(marker)?;
    let (oid, _) = rest.split_once('\'')?;
    oid.parse().ok()
}

fn catalog_describe_referenced_by_foreign_keys_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_constraint c")
        || !canonical.contains("confrelid in (select pg_catalog.pg_partition_ancestors('")
        || !canonical.contains("and contype = 'f'")
        || !canonical.contains("conparentid = 0")
    {
        return None;
    }
    let marker = "pg_partition_ancestors('";
    let (_, rest) = canonical.split_once(marker)?;
    let (oid, _) = rest.split_once('\'')?;
    oid.parse().ok()
}

fn catalog_describe_trigger_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_trigger t")
        || !canonical.contains("pg_catalog.pg_get_triggerdef(t.oid, true)")
    {
        return None;
    }
    let marker = "where t.tgrelid = '";
    let (_, rest) = canonical.split_once(marker)?;
    let (oid, _) = rest.split_once('\'')?;
    oid.parse().ok()
}

fn check_constraint_definition(constraint: &CatalogCheckConstraint) -> String {
    let op = match constraint.op {
        SelectFilterOp::Eq => "=",
        SelectFilterOp::Lt => "<",
        SelectFilterOp::Lte => "<=",
        SelectFilterOp::Gt => ">",
        SelectFilterOp::Gte => ">=",
        SelectFilterOp::LikePrefix => "LIKE",
    };
    let value = match &constraint.value {
        SqlValue::Text(value) => format!("'{}'", value.replace('\'', "''")),
        value => format_sql_value(value),
    };
    format!("CHECK (({} {} {}))", constraint.column, op, value)
}

fn catalog_describe_foreign_key_rows(
    session: &Session,
    table_oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == table_oid) else {
        return Vec::new();
    };
    let mut rows = table
        .foreign_keys
        .iter()
        .map(|constraint| {
            vec![
                Some("t".to_string()),
                Some(constraint.name.clone()),
                Some(foreign_key_definition(constraint)),
                Some(format!("public.{}", table.name)),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn catalog_describe_referenced_by_foreign_key_rows(
    session: &Session,
    table_oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(parent_table) = session.tables.values().find(|table| table.oid == table_oid) else {
        return Vec::new();
    };
    let mut rows = session
        .tables
        .values()
        .flat_map(|table| {
            table
                .foreign_keys
                .iter()
                .filter(|constraint| constraint.referenced_table == parent_table.name)
                .map(|constraint| {
                    vec![
                        Some(constraint.name.clone()),
                        Some(format!("public.{}", table.name)),
                        Some(foreign_key_definition(constraint)),
                    ]
                })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[0].cmp(&right[0]).then_with(|| left[1].cmp(&right[1])));
    rows
}

fn catalog_describe_check_constraint_rows(
    session: &Session,
    table_oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == table_oid) else {
        return Vec::new();
    };
    let mut rows = table
        .check_constraints
        .iter()
        .map(|constraint| {
            vec![
                Some(constraint.name.clone()),
                Some(check_constraint_definition(constraint)),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[0].cmp(&right[0]));
    rows
}

fn catalog_describe_index_rows(session: &Session, table_oid: u32) -> Vec<Vec<Option<String>>> {
    catalog_index_entries(session)
        .into_iter()
        .filter(|entry| entry.table_oid == table_oid)
        .map(|entry| {
            vec![
                Some(entry.index.name.clone()),
                Some(if entry.index.primary_key { "t" } else { "f" }.to_string()),
                Some(if entry.index.unique { "t" } else { "f" }.to_string()),
                Some("f".to_string()),
                Some("t".to_string()),
                Some(catalog_index_definition(&entry.index)),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_definition(&entry.index)),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_contype(&entry.index).to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some("f".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some("f".to_string()),
                Some("f".to_string()),
                Some("0".to_string()),
            ]
        })
        .collect()
}

fn catalog_describe_verbose_attribute_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated, a.attstorage, a.attcompression as attcompression, case when a.attstattarget=-1 then null else a.attstattarget end as attstattarget, pg_catalog.col_description(a.attrelid, a.attnum) from pg_catalog.pg_attribute a where a.attrelid = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_verbose_attribute_rows(
    session: &Session,
    oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                column.def.default.as_ref().map(format_column_default_expr),
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
                Some(sql_type_storage_code(column.def.ty).to_string()),
                Some(String::new()),
                None,
                session
                    .comments
                    .get(&CatalogCommentTarget::Column {
                        table: table.name.clone(),
                        attnum: column.attnum,
                    })
                    .cloned(),
            ]
        })
        .collect()
}

fn catalog_describe_attribute_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                column.def.default.as_ref().map(format_column_default_expr),
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
            ]
        })
        .collect()
}

fn catalog_describe_policy_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select pol.polname, pol.polpermissive, case when pol.polroles = '{0}' then null else pg_catalog.array_to_string(array(select rolname from pg_catalog.pg_roles where oid = any (pol.polroles) order by 1),',') end, pg_catalog.pg_get_expr(pol.polqual, pol.polrelid), pg_catalog.pg_get_expr(pol.polwithcheck, pol.polrelid), case pol.polcmd when 'r' then 'select' when 'a' then 'insert' when 'w' then 'update' when 'd' then 'delete' end as cmd from pg_catalog.pg_policy pol where pol.polrelid = '";
    let suffix = "' order by 1";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_statistic_ext_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select oid, stxrelid::pg_catalog.regclass, stxnamespace::pg_catalog.regnamespace::pg_catalog.text as nsp, stxname, pg_catalog.pg_get_statisticsobjdef_columns(oid) as columns, 'd' = any(stxkind) as ndist_enabled, 'f' = any(stxkind) as deps_enabled, 'm' = any(stxkind) as mcv_enabled, stxstattarget from pg_catalog.pg_statistic_ext where stxrelid = '";
    let suffix = "' order by nsp, stxname";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_inherits_parent_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c.oid::pg_catalog.regclass from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhparent and i.inhrelid = '";
    let suffix = "' and c.relkind != 'p' and c.relkind != 'i' order by inhseqno";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_inherits_child_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c.oid::pg_catalog.regclass, c.relkind, inhdetachpending, pg_catalog.pg_get_expr(c.relpartbound, c.oid) from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhrelid and i.inhparent = '";
    let suffix = "' order by pg_catalog.pg_get_expr(c.relpartbound, c.oid) = 'default', c.oid::pg_catalog.regclass::pg_catalog.text";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_empty_rows_for_relation_oid(_oid: u32) -> Vec<Vec<Option<String>>> {
    Vec::new()
}

#[cfg(test)]
pub(super) fn test_catalog_describe_relation_lookup_query_table(canonical: &str) -> Option<String> {
    catalog_describe_relation_lookup_query_table(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_relation_lookup_query_public_namespace(
    canonical: &str,
) -> bool {
    catalog_describe_relation_lookup_query_public_namespace(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_relation_lookup_query_all_schemas(canonical: &str) -> bool {
    catalog_describe_relation_lookup_query_all_schemas(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_relation_lookup_rows(
    session: &Session,
    relname_pattern: &str,
) -> Vec<Vec<Option<String>>> {
    catalog_describe_relation_lookup_rows(session, relname_pattern)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_relation_lookup_rows_for_public_namespace(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    catalog_describe_relation_lookup_rows_for_public_namespace(session)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_relation_flags_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_relation_flags_query_oid(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_relation_flags_rows(
    session: &Session,
    oid: u32,
) -> Vec<Vec<Option<String>>> {
    catalog_describe_relation_flags_rows(session, oid)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_attribute_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_attribute_query_oid(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_verbose_attribute_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_verbose_attribute_query_oid(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_verbose_attribute_rows(
    session: &Session,
    oid: u32,
) -> Vec<Vec<Option<String>>> {
    catalog_describe_verbose_attribute_rows(session, oid)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_attribute_rows(
    session: &Session,
    oid: u32,
) -> Vec<Vec<Option<String>>> {
    catalog_describe_attribute_rows(session, oid)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_policy_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_policy_query_oid(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_statistic_ext_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_statistic_ext_query_oid(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_inherits_parent_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_inherits_parent_query_oid(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_describe_inherits_child_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_inherits_child_query_oid(canonical)
}
