// Legacy pg_dump relation metadata ownership. This is not a product execution path.

use super::{
    bool_column, catalog_constraint_contype, catalog_constraint_definition, catalog_constraint_oid,
    catalog_foreign_key_oid, catalog_index_definition, catalog_index_entries,
    foreign_key_definition, format_column_default_expr, int4_column, psql_relname_pattern_matches,
    relation_acl_array_display, sql_type_alignment_code, sql_type_display_name,
    sql_type_storage_code, text_column, Column, SelectProjection, Session, SqlType,
    PUBLIC_NAMESPACE_OID,
};
use std::collections::BTreeSet;

pub(super) fn pg_dump_table_oid_lookup_query_table(canonical: &str) -> Option<String> {
    let prefix = "select c.oid from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid operator(pg_catalog.=) c.relnamespace where c.relkind operator(pg_catalog.=) any (array['r', 's', 'v', 'm', 'f', 'p']) and c.relname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid)";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

pub(super) fn pg_dump_table_oid_lookup_rows(
    session: &Session,
    relname_pattern: &str,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        if psql_relname_pattern_matches(relname_pattern, &table.name) {
            rows.push(vec![Some(table.oid.to_string())]);
        }
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        if psql_relname_pattern_matches(relname_pattern, &view.name) {
            rows.push(vec![Some(view.oid.to_string())]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        if psql_relname_pattern_matches(relname_pattern, &view.name) {
            rows.push(vec![Some(view.oid.to_string())]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by_key(|sequence| sequence.oid);
    for sequence in sequences {
        if psql_relname_pattern_matches(relname_pattern, &sequence.name) {
            rows.push(vec![Some(sequence.oid.to_string())]);
        }
    }
    rows
}

pub(super) fn is_pg_dump_class_metadata_query(canonical: &str) -> bool {
    canonical.starts_with("select c.tableoid, c.oid, c.relname, c.relnamespace, c.relkind, c.reltype, c.relowner, c.relchecks, c.relhasindex, c.relhasrules, c.relpages, c.relhastriggers, c.relpersistence, c.reloftype, c.relacl, acldefault(")
        && canonical.contains("from pg_class c left join pg_depend d")
        && canonical.ends_with("where c.relkind in ('r', 's', 'v', 'c', 'm', 'f', 'p') order by c.oid")
}

pub(super) fn pg_dump_class_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("relname"),
        int4_column("relnamespace"),
        text_column("relkind"),
        int4_column("reltype"),
        int4_column("relowner"),
        int4_column("relchecks"),
        bool_column("relhasindex"),
        bool_column("relhasrules"),
        int4_column("relpages"),
        bool_column("relhastriggers"),
        text_column("relpersistence"),
        int4_column("reloftype"),
        text_column("relacl"),
        text_column("acldefault"),
        int4_column("foreignserver"),
        text_column("relfrozenxid"),
        text_column("tfrozenxid"),
        int4_column("toid"),
        int4_column("toastpages"),
        text_column("toast_reloptions"),
        int4_column("owning_tab"),
        int4_column("owning_col"),
        text_column("reltablespace"),
        bool_column("relhasoids"),
        bool_column("relispopulated"),
        text_column("relreplident"),
        bool_column("relrowsecurity"),
        bool_column("relforcerowsecurity"),
        text_column("relminmxid"),
        text_column("tminmxid"),
        text_column("reloptions"),
        text_column("checkoption"),
        text_column("amname"),
        bool_column("is_identity_sequence"),
        bool_column("ispartition"),
    ]
}

pub(super) fn pg_dump_class_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        let relhasindex = session
            .indexes
            .iter()
            .any(|index| index.table == table.name && session.tables.contains_key(&index.table));
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: table.oid,
            name: &table.name,
            relkind: "r",
            relchecks: table.check_constraints.len(),
            relhasindex,
            relhasrules: false,
            amname: Some("heap"),
        }));
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: view.oid,
            name: &view.name,
            relkind: "v",
            relchecks: 0,
            relhasindex: false,
            relhasrules: true,
            amname: None,
        }));
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: view.oid,
            name: &view.name,
            relkind: "m",
            relchecks: 0,
            relhasindex: false,
            relhasrules: true,
            amname: Some("heap"),
        }));
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by_key(|sequence| sequence.oid);
    for sequence in sequences {
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: sequence.oid,
            name: &sequence.name,
            relkind: "S",
            relchecks: 0,
            relhasindex: false,
            relhasrules: false,
            amname: None,
        }));
    }
    rows
}

struct PgDumpClassMetadata<'a> {
    session: &'a Session,
    oid: u32,
    name: &'a str,
    relkind: &'a str,
    relchecks: usize,
    relhasindex: bool,
    relhasrules: bool,
    amname: Option<&'a str>,
}

fn pg_dump_class_metadata_row(metadata: PgDumpClassMetadata<'_>) -> Vec<Option<String>> {
    vec![
        Some("1259".to_string()),
        Some(metadata.oid.to_string()),
        Some(metadata.name.to_string()),
        Some(PUBLIC_NAMESPACE_OID.to_string()),
        Some(metadata.relkind.to_string()),
        Some(metadata.relchecks.to_string()),
        Some("10".to_string()),
        Some("0".to_string()),
        Some(if metadata.relhasindex { "t" } else { "f" }.to_string()),
        Some(if metadata.relhasrules { "t" } else { "f" }.to_string()),
        Some("0".to_string()),
        Some("f".to_string()),
        Some("p".to_string()),
        Some("0".to_string()),
        relation_acl_array_display(metadata.session, metadata.name),
        Some(pg_dump_class_acl_default(metadata.relkind).to_string()),
        Some("0".to_string()),
        Some("0".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some("f".to_string()),
        Some("t".to_string()),
        Some("d".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("0".to_string()),
        None,
        None,
        None,
        metadata.amname.map(str::to_string),
        Some("f".to_string()),
        Some("f".to_string()),
    ]
}

fn pg_dump_class_acl_default(relkind: &str) -> &'static str {
    if relkind == "S" {
        "{postgres=rwU/postgres}"
    } else {
        "{postgres=arwdDxt/postgres}"
    }
}

pub(super) fn pg_dump_attribute_metadata_query_oids(canonical: &str) -> Option<Vec<u32>> {
    let marker = "from unnest('{";
    let (_, rest) = canonical.split_once(marker)?;
    let (oids, rest) = rest.split_once("}'::pg_catalog.oid[]")?;
    if !rest.contains("join pg_catalog.pg_attribute a") {
        return None;
    }
    if oids.trim().is_empty() {
        return Some(Vec::new());
    }
    oids.split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()
}

pub(super) fn pg_dump_attribute_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("attrelid"),
        int4_column("attnum"),
        text_column("attname"),
        int4_column("attstattarget"),
        text_column("attstorage"),
        text_column("typstorage"),
        bool_column("attnotnull"),
        bool_column("atthasdef"),
        bool_column("attisdropped"),
        int4_column("attlen"),
        text_column("attalign"),
        bool_column("attislocal"),
        text_column("atttypname"),
        text_column("attoptions"),
        int4_column("attcollation"),
        text_column("attfdwoptions"),
        text_column("attcompression"),
        text_column("attidentity"),
        text_column("attmissingval"),
        text_column("attgenerated"),
    ]
}

pub(super) fn pg_dump_attribute_metadata_rows(
    session: &Session,
    relation_oids: &[u32],
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for oid in relation_oids {
        if let Some(table) = session.tables.values().find(|table| table.oid == *oid) {
            for column in &table.columns {
                rows.push(pg_dump_attribute_metadata_row(
                    table.oid,
                    column.attnum,
                    &column.def.name,
                    column.def.ty,
                    column.def.domain.as_deref(),
                    column.def.default.is_some(),
                ));
            }
            continue;
        }
        if let Some(view) = session.views.values().find(|view| view.oid == *oid) {
            let Some(table) = session.tables.get(&view.query.table) else {
                continue;
            };
            let columns = match &view.query.projection {
                SelectProjection::All => table.columns.iter().collect::<Vec<_>>(),
                SelectProjection::Columns(names) => names
                    .iter()
                    .filter_map(|name| table.columns.iter().find(|column| column.def.name == *name))
                    .collect::<Vec<_>>(),
                SelectProjection::CountAll
                | SelectProjection::GroupedCount { .. }
                | SelectProjection::Sum { .. }
                | SelectProjection::GroupedSum { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::GroupedAvg { .. }
                | SelectProjection::Min { .. }
                | SelectProjection::GroupedMin { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::GroupedMax { .. }
                | SelectProjection::CountDistinct { .. }
                | SelectProjection::GroupedAggregates { .. } => Vec::new(),
            };
            for (idx, column) in columns.into_iter().enumerate() {
                let Ok(attnum) = i16::try_from(idx + 1) else {
                    continue;
                };
                rows.push(pg_dump_attribute_metadata_row(
                    view.oid,
                    attnum,
                    &column.def.name,
                    column.def.ty,
                    column.def.domain.as_deref(),
                    false,
                ));
            }
            continue;
        }
        if let Some(view) = session
            .materialized_views
            .values()
            .find(|view| view.oid == *oid)
        {
            for column in &view.columns {
                rows.push(pg_dump_attribute_metadata_row(
                    view.oid,
                    column.attnum,
                    &column.def.name,
                    column.def.ty,
                    column.def.domain.as_deref(),
                    false,
                ));
            }
        }
    }
    rows
}

fn pg_dump_attribute_metadata_row(
    relation_oid: u32,
    attnum: i16,
    name: &str,
    ty: SqlType,
    domain: Option<&str>,
    has_default: bool,
) -> Vec<Option<String>> {
    vec![
        Some(relation_oid.to_string()),
        Some(attnum.to_string()),
        Some(name.to_string()),
        Some("-1".to_string()),
        Some(sql_type_storage_code(ty).to_string()),
        Some(sql_type_storage_code(ty).to_string()),
        Some("f".to_string()),
        Some(if has_default { "t" } else { "f" }.to_string()),
        Some("f".to_string()),
        Some(ty.type_size().to_string()),
        Some(sql_type_alignment_code(ty).to_string()),
        Some("t".to_string()),
        Some(
            domain
                .map(str::to_string)
                .unwrap_or_else(|| sql_type_display_name(ty).to_string()),
        ),
        None,
        Some("0".to_string()),
        None,
        Some(String::new()),
        Some(String::new()),
        None,
        Some(String::new()),
    ]
}

pub(super) fn is_pg_dump_index_metadata_query(canonical: &str) -> bool {
    canonical.starts_with("select t.tableoid, t.oid, i.indrelid")
        && canonical.contains("join pg_catalog.pg_index i")
}

pub(super) fn pg_dump_index_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("indrelid"),
        text_column("indexname"),
        text_column("indexdef"),
        text_column("indkey"),
        bool_column("indisclustered"),
        text_column("contype"),
        text_column("conname"),
        bool_column("condeferrable"),
        bool_column("condeferred"),
        int4_column("contableoid"),
        int4_column("conoid"),
        text_column("condef"),
        text_column("tablespace"),
        text_column("indreloptions"),
        bool_column("indisreplident"),
        int4_column("parentidx"),
        int4_column("indnkeyatts"),
        int4_column("indnatts"),
        text_column("indstatcols"),
        text_column("indstatvals"),
        bool_column("indnullsnotdistinct"),
    ]
}

pub(super) fn pg_dump_index_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_index_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("1259".to_string()),
                Some(entry.index_oid.to_string()),
                Some(entry.table_oid.to_string()),
                Some(entry.index.name.clone()),
                Some(catalog_index_definition(&entry.index)),
                Some(entry.attnum.to_string()),
                Some("f".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_contype(&entry.index).to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(entry.index.name.clone()),
                Some("f".to_string()),
                Some("f".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some("2606".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_oid(&entry).to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_definition(&entry.index)),
                Some(String::new()),
                None,
                Some("f".to_string()),
                Some("0".to_string()),
                Some("1".to_string()),
                Some("1".to_string()),
                None,
                None,
                Some("f".to_string()),
            ]
        })
        .collect()
}

pub(super) fn is_catalog_foreign_key_metadata_query(canonical: &str) -> bool {
    canonical.starts_with("select c.tableoid, c.oid, conrelid, conname")
        && canonical.contains("join pg_catalog.pg_constraint c")
}

pub(super) fn catalog_foreign_key_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("conrelid"),
        text_column("conname"),
        int4_column("confrelid"),
        int4_column("conindid"),
        text_column("condef"),
    ]
}

pub(super) fn catalog_foreign_key_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let index_entries = catalog_index_entries(session);
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        for (idx, foreign_key) in table.foreign_keys.iter().enumerate() {
            let referenced_table_oid = session
                .tables
                .get(&foreign_key.referenced_table)
                .map(|table| table.oid)
                .unwrap_or(0);
            let referenced_index_oid = index_entries
                .iter()
                .find(|entry| {
                    entry.index.table == foreign_key.referenced_table
                        && entry.index.column == foreign_key.referenced_column
                        && (entry.index.primary_key || entry.index.unique_constraint)
                })
                .map(|entry| entry.index_oid)
                .unwrap_or(0);
            rows.push(vec![
                Some("2606".to_string()),
                Some(catalog_foreign_key_oid(table.oid, idx).to_string()),
                Some(table.oid.to_string()),
                Some(foreign_key.name.clone()),
                Some(referenced_table_oid.to_string()),
                Some(referenced_index_oid.to_string()),
                Some(foreign_key_definition(foreign_key)),
            ]);
        }
    }
    rows
}

pub(super) fn pg_dump_view_definition_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select pg_catalog.pg_get_viewdef('";
    let suffix = "'::pg_catalog.oid) as viewdef";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

pub(super) fn pg_dump_view_definition_rows(
    session: &Session,
    view_oid: u32,
) -> Vec<Vec<Option<String>>> {
    if let Some(view) = session.views.values().find(|view| view.oid == view_oid) {
        return vec![vec![Some(format!("{};", view.definition))]];
    }
    session
        .materialized_views
        .values()
        .find(|view| view.oid == view_oid)
        .map(|view| vec![vec![Some(format!("{};", view.definition))]])
        .unwrap_or_default()
}

pub(super) fn pg_dump_dependency_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        if let Some(table) = session.tables.get(&view.query.table) {
            rows.push(vec![
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("1259".to_string()),
                Some(table.oid.to_string()),
                Some("n".to_string()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        if let Some(table) = session.tables.get(&view.query.table) {
            rows.push(vec![
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("1259".to_string()),
                Some(table.oid.to_string()),
                Some("n".to_string()),
            ]);
        }
    }
    rows
}

pub(super) fn pg_dump_attrdef_metadata_query_relation_oids(canonical: &str) -> Option<Vec<u32>> {
    let prefix = "select a.tableoid, a.oid, adrelid, adnum, pg_catalog.pg_get_expr(adbin, adrelid) as adsrc from unnest('{";
    let suffix = "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_attrdef a on (src.tbloid = a.adrelid) order by a.adrelid, a.adnum";
    let body = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let oids = body
        .split(',')
        .map(|raw| raw.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!oids.is_empty()).then_some(oids)
}

pub(super) fn pg_dump_attrdef_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("adrelid"),
        int4_column("adnum"),
        text_column("adsrc"),
    ]
}

pub(super) fn pg_dump_attrdef_metadata_rows(
    session: &Session,
    relation_oids: &[u32],
) -> Vec<Vec<Option<String>>> {
    let requested = relation_oids.iter().copied().collect::<BTreeSet<_>>();
    let mut tables = session
        .tables
        .values()
        .filter(|table| requested.contains(&table.oid))
        .collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().filter_map(|column| {
                column.def.default.as_ref().map(|default| {
                    vec![
                        Some("2604".to_string()),
                        Some((30_000_u32 + table.oid + column.attnum as u32).to_string()),
                        Some(table.oid.to_string()),
                        Some(column.attnum.to_string()),
                        Some(format_column_default_expr(default)),
                    ]
                })
            })
        })
        .collect()
}

#[cfg(test)]
pub(super) fn test_pg_dump_index_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_dump_index_metadata_rows(session)
}
