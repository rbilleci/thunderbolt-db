//! PostgreSQL 16 catalog relations used by psql and information-schema clients.
//!
//! This leaf only encodes deterministic rows from one immutable [`CatalogSnapshot`].
//! Consumers upload the rows and perform filtering, joining, ordering, and projection on the GPU.

use super::*;

pub(super) fn synthesize_pg_index(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_index",
        &[
            ("indexrelid", SqlType::Int4),
            ("indrelid", SqlType::Int4),
            ("indnatts", SqlType::Int4),
            ("indnkeyatts", SqlType::Int4),
            ("indisunique", SqlType::Bool),
            ("indnullsnotdistinct", SqlType::Bool),
            ("indisprimary", SqlType::Bool),
            ("indisexclusion", SqlType::Bool),
            ("indimmediate", SqlType::Bool),
            ("indisclustered", SqlType::Bool),
            ("indisvalid", SqlType::Bool),
            ("indcheckxmin", SqlType::Bool),
            ("indisready", SqlType::Bool),
            ("indislive", SqlType::Bool),
            ("indisreplident", SqlType::Bool),
            ("indkey", SqlType::Text),
            ("indcollation", SqlType::Text),
            ("indclass", SqlType::Text),
            ("indoption", SqlType::Text),
            ("indexprs", SqlType::Text),
            ("indpred", SqlType::Text),
        ],
    );
    let rows = catalog_index_entries(catalog)
        .into_iter()
        .map(|entry| {
            let key_count = entry.attnums.len() as i32;
            let indkey = entry
                .attnums
                .iter()
                .map(i16::to_string)
                .collect::<Vec<_>>()
                .join(" ");
            vec![
                SqlValue::Int4(entry.index_oid as i32),
                SqlValue::Int4(entry.table.oid as i32),
                SqlValue::Int4(key_count),
                SqlValue::Int4(key_count),
                SqlValue::Bool(entry.index.unique),
                SqlValue::Bool(false),
                SqlValue::Bool(entry.index.primary_key),
                SqlValue::Bool(false),
                SqlValue::Bool(true),
                SqlValue::Bool(false),
                SqlValue::Bool(true),
                SqlValue::Bool(false),
                SqlValue::Bool(true),
                SqlValue::Bool(true),
                SqlValue::Bool(false),
                SqlValue::Text(indkey),
                SqlValue::Text(String::new()),
                SqlValue::Text(String::new()),
                SqlValue::Text(String::new()),
                SqlValue::Null,
                SqlValue::Null,
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_pg_tables(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_tables",
        &[
            ("schemaname", SqlType::Text),
            ("tablename", SqlType::Text),
            ("tableowner", SqlType::Text),
            ("tablespace", SqlType::Text),
            ("hasindexes", SqlType::Bool),
            ("hasrules", SqlType::Bool),
            ("hastriggers", SqlType::Bool),
            ("rowsecurity", SqlType::Bool),
        ],
    );
    let rows = catalog
        .relational_catalog
        .values()
        .map(|relation| {
            vec![
                SqlValue::Text(relation.schema.clone()),
                SqlValue::Text(relation.name.clone()),
                SqlValue::Text("postgres".to_string()),
                SqlValue::Null,
                SqlValue::Bool(!relation.indexes.is_empty()),
                SqlValue::Bool(false),
                SqlValue::Bool(false),
                SqlValue::Bool(false),
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_pg_attrdef(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_attrdef",
        &[
            ("oid", SqlType::Int4),
            ("adrelid", SqlType::Int4),
            ("adnum", SqlType::Int4),
            ("adbin", SqlType::Text),
        ],
    );
    let mut rows = Vec::new();
    for relation in catalog.relational_catalog.values() {
        for column in &relation.columns {
            let Some(default) = &column.default else {
                continue;
            };
            let expression = render_expression(default)
                .expect("catalog defaults never retain unbound parameters");
            let oid = relation
                .oid
                .wrapping_mul(128)
                .wrapping_add(u32::from(column.attnum as u16));
            rows.push(vec![
                SqlValue::Int4(oid as i32),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Int4(i32::from(column.attnum)),
                SqlValue::Text(expression),
            ]);
        }
    }
    (table, rows)
}

pub(super) fn synthesize_information_schema_table_constraints(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "information_schema",
        "table_constraints",
        &[
            ("constraint_catalog", SqlType::Text),
            ("constraint_schema", SqlType::Text),
            ("constraint_name", SqlType::Text),
            ("table_catalog", SqlType::Text),
            ("table_schema", SqlType::Text),
            ("table_name", SqlType::Text),
            ("constraint_type", SqlType::Text),
            ("is_deferrable", SqlType::Text),
            ("initially_deferred", SqlType::Text),
            ("enforced", SqlType::Text),
            ("nulls_distinct", SqlType::Text),
        ],
    );
    let mut rows = Vec::new();
    for relation in catalog.relational_catalog.values() {
        let mut indexes = relation.indexes.iter().collect::<Vec<_>>();
        indexes.sort_by(|left, right| left.name.cmp(&right.name));
        for index in indexes {
            let kind = if index.primary_key {
                Some("PRIMARY KEY")
            } else if index.unique_constraint {
                Some("UNIQUE")
            } else {
                None
            };
            if let Some(kind) = kind {
                rows.push(information_schema_constraint_row(
                    relation,
                    &index.name,
                    kind,
                ));
            }
        }
        for constraint in &relation.check_constraints {
            rows.push(information_schema_constraint_row(
                relation,
                &constraint.name,
                "CHECK",
            ));
        }
        for constraint in &relation.foreign_keys {
            rows.push(information_schema_constraint_row(
                relation,
                &constraint.name,
                "FOREIGN KEY",
            ));
        }
    }
    (table, rows)
}

fn information_schema_constraint_row(
    relation: &RelationalTable,
    name: &str,
    kind: &str,
) -> Vec<SqlValue> {
    vec![
        SqlValue::Text("postgres".to_string()),
        SqlValue::Text("public".to_string()),
        SqlValue::Text(name.to_string()),
        SqlValue::Text("postgres".to_string()),
        SqlValue::Text(relation.schema.clone()),
        SqlValue::Text(relation.name.clone()),
        SqlValue::Text(kind.to_string()),
        SqlValue::Text("NO".to_string()),
        SqlValue::Text("NO".to_string()),
        SqlValue::Text("YES".to_string()),
        SqlValue::Text("YES".to_string()),
    ]
}

pub(crate) fn synthesize_information_schema_key_column_usage(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "information_schema",
        "key_column_usage",
        &[
            ("constraint_catalog", SqlType::Text),
            ("constraint_schema", SqlType::Text),
            ("constraint_name", SqlType::Text),
            ("table_catalog", SqlType::Text),
            ("table_schema", SqlType::Text),
            ("table_name", SqlType::Text),
            ("column_name", SqlType::Text),
            ("ordinal_position", SqlType::Int4),
            ("position_in_unique_constraint", SqlType::Int4),
        ],
    );
    let mut rows = Vec::new();
    for relation in catalog.relational_catalog.values() {
        let mut indexes = relation.indexes.iter().collect::<Vec<_>>();
        indexes.sort_by_key(|index| {
            index
                .key_columns
                .first()
                .and_then(|name| relation.columns.iter().find(|column| column.name == *name))
                .map_or(i16::MAX, |column| column.attnum)
        });
        for index in indexes {
            if !(index.primary_key || index.unique_constraint) {
                continue;
            }
            for (position, column) in index.key_columns.iter().enumerate() {
                rows.push(information_schema_key_column_row(
                    relation,
                    &index.name,
                    column,
                    position,
                    None,
                ));
            }
        }
        for constraint in &relation.foreign_keys {
            rows.push(information_schema_key_column_row(
                relation,
                &constraint.name,
                &constraint.column,
                0,
                Some(1),
            ));
        }
    }
    (table, rows)
}

fn information_schema_key_column_row(
    relation: &RelationalTable,
    constraint: &str,
    column: &str,
    position: usize,
    position_in_unique_constraint: Option<i32>,
) -> Vec<SqlValue> {
    vec![
        SqlValue::Text("postgres".to_string()),
        SqlValue::Text("public".to_string()),
        SqlValue::Text(constraint.to_string()),
        SqlValue::Text("postgres".to_string()),
        SqlValue::Text(relation.schema.clone()),
        SqlValue::Text(relation.name.clone()),
        SqlValue::Text(column.to_string()),
        SqlValue::Int4((position + 1) as i32),
        position_in_unique_constraint.map_or(SqlValue::Null, SqlValue::Int4),
    ]
}

pub(super) fn synthesize_information_schema_views(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "information_schema",
        "views",
        &[
            ("table_catalog", SqlType::Text),
            ("table_schema", SqlType::Text),
            ("table_name", SqlType::Text),
            ("view_definition", SqlType::Text),
            ("check_option", SqlType::Text),
            ("is_updatable", SqlType::Text),
            ("is_insertable_into", SqlType::Text),
            ("is_trigger_updatable", SqlType::Text),
            ("is_trigger_deletable", SqlType::Text),
            ("is_trigger_insertable", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_views
        .values()
        .map(|view| {
            vec![
                SqlValue::Text("postgres".to_string()),
                SqlValue::Text(view.schema.clone()),
                SqlValue::Text(view.name.clone()),
                SqlValue::Text(view.definition.clone()),
                SqlValue::Text("NONE".to_string()),
                SqlValue::Text("NO".to_string()),
                SqlValue::Text("NO".to_string()),
                SqlValue::Text("NO".to_string()),
                SqlValue::Text("NO".to_string()),
                SqlValue::Text("NO".to_string()),
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_pg_subscription(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_subscription",
        &[
            ("oid", SqlType::Int4),
            ("subdbid", SqlType::Int4),
            ("subname", SqlType::Text),
            ("subowner", SqlType::Int4),
            ("subenabled", SqlType::Bool),
            ("subconninfo", SqlType::Text),
            ("subpublications", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_subscriptions
        .values()
        .map(|subscription| {
            vec![
                SqlValue::Int4(subscription.oid as i32),
                SqlValue::Int4(1),
                SqlValue::Text(subscription.name.clone()),
                SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
                SqlValue::Bool(subscription.enabled),
                SqlValue::Text(subscription.connection.clone()),
                SqlValue::Text(format!("{{{}}}", subscription.publications.join(","))),
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_information_schema_schemata(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "information_schema",
        "schemata",
        &[
            ("schema_name", SqlType::Text),
            ("schema_owner", SqlType::Text),
        ],
    );
    let rows = if catalog.relational_public_schema_exists {
        vec![vec![
            SqlValue::Text("public".to_string()),
            SqlValue::Text("postgres".to_string()),
        ]]
    } else {
        Vec::new()
    };
    (table, rows)
}

pub(super) fn synthesize_pg_extension(
    _catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_extension",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("extname", SqlType::Text),
            ("extnamespace", SqlType::Int4),
            ("extrelocatable", SqlType::Bool),
            ("extversion", SqlType::Text),
            ("extconfig", SqlType::Text),
            ("extcondition", SqlType::Text),
        ],
    );
    (
        table,
        vec![vec![
            SqlValue::Int4(3079),
            SqlValue::Int4(13_500),
            SqlValue::Text("plpgsql".to_string()),
            SqlValue::Int4(PG_CATALOG_NAMESPACE_OID),
            SqlValue::Bool(false),
            SqlValue::Text("1.0".to_string()),
            SqlValue::Null,
            SqlValue::Null,
        ]],
    )
}

pub(super) fn synthesize_pg_publication(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_publication",
        &[
            ("oid", SqlType::Int4),
            ("pubname", SqlType::Text),
            ("pubowner", SqlType::Int4),
            ("puballtables", SqlType::Bool),
            ("pubinsert", SqlType::Bool),
            ("pubupdate", SqlType::Bool),
            ("pubdelete", SqlType::Bool),
            ("pubtruncate", SqlType::Bool),
            ("pubviaroot", SqlType::Bool),
        ],
    );
    let rows = catalog
        .relational_publications
        .values()
        .map(|publication| {
            vec![
                SqlValue::Int4(publication.oid as i32),
                SqlValue::Text(publication.name.clone()),
                SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
                SqlValue::Bool(publication.all_tables),
                SqlValue::Bool(true),
                SqlValue::Bool(true),
                SqlValue::Bool(true),
                SqlValue::Bool(true),
                SqlValue::Bool(false),
            ]
        })
        .collect();
    (table, rows)
}

pub(super) fn synthesize_pg_publication_rel(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_publication_rel",
        &[
            ("oid", SqlType::Int4),
            ("prpubid", SqlType::Int4),
            ("prrelid", SqlType::Int4),
            ("prattrs", SqlType::Text),
            ("prqual", SqlType::Text),
        ],
    );
    let mut association_oid = 90_000_i32;
    let mut rows = Vec::new();
    for publication in catalog.relational_publications.values() {
        for relation_name in &publication.tables {
            let Some(relation) = catalog.relational_catalog.get(relation_name) else {
                continue;
            };
            rows.push(vec![
                SqlValue::Int4(association_oid),
                SqlValue::Int4(publication.oid as i32),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Null,
                SqlValue::Null,
            ]);
            association_oid += 1;
        }
    }
    (table, rows)
}
